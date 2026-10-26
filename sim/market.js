// Seeded agent-based market for Spec 1 (market herding monitor).
//
// One instrument traded on k venues. Each venue has three kinds of
// participant:
//   - market makers quote depth on both sides and widen/withdraw under stress,
//   - noise traders send zero-mean signed flow,
//   - momentum traders send directional flow when the venue is in a one-sided
//     or sweeping regime, and chase the venue's own last return.
// A latent per-venue regime (calm, thin, one-sided, sweeping) sets how each
// group behaves. Regimes follow a sticky Markov kernel. With herding strength
// herd(t), a venue copies a shared common regime (and its direction) at each
// step instead of stepping on its own; herd = 0 gives independent venues.
//
// Planted episodes ramp herding up before a crash. During the crash the
// common regime is forced to a sell-side sweep, so the herded venues sweep
// together and the composite price falls. Before the crash the common regime
// still follows the ordinary kernel, so herding first shows up as cross-venue
// regime agreement and only later as volatility. Whether that ordering yields
// a measurable lead is tested, not assumed (test/market.sim.test.js).
//
// Nothing here models real participants' minds; "agent" means a rule that
// emits order flow.

const { mulberry32 } = require('conscious-agent');
const { sampleIndex } = require('./chains');

const REGIMES = ['calm', 'thin', 'one-sided', 'sweeping'];

// Sticky regime kernel; mostly calm, sweeps are short.
const REGIME_KERNEL = [
  [0.94, 0.03, 0.025, 0.005],
  [0.15, 0.80, 0.04, 0.01],
  [0.15, 0.05, 0.75, 0.05],
  [0.20, 0.10, 0.20, 0.50],
];

// Per-regime behaviour.
const MM_DEPTH = [1, 0.45, 0.7, 0.35]; // market-maker depth multiplier
const MOMENTUM = [0, 0, 0.8, 2.0]; // momentum flow intensity
const SIDE = [0, 0, 0.8, 1.0]; // how one-sided the book gets

function gauss(rng) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

// Herding strength at t from planted episodes. Each episode is
// { start, end, ramp, peak, shape }: herding rises over [start - ramp, start)
// ('ramp': linearly; 'step': jumps to peak at start - ramp) and stays at
// `peak` until `end`. The crash itself is [start, end].
function episodeHerd(episodes, t) {
  let h = 0;
  for (const e of episodes) {
    const from = e.start - e.ramp;
    if (t < from || t > e.end) continue;
    const v = t >= e.start || e.shape === 'step' ? e.peak : e.peak * (t - from + 1) / e.ramp;
    if (v > h) h = v;
  }
  return h;
}

// Simulate T steps on k venues.
//
//   herd      background herding, a number or a function of t (default 0)
//   episodes  planted crash episodes, see episodeHerd (default none)
//   seed      mulberry32 seed
//
// Returns
//   features  one object per venue: Float64Arrays spread, depth, imbalance,
//             flow, mid (log price)
//   labels    herd (per-step herding strength), regimes (latent per-venue
//             regime indices), common (common regime), crash (1 inside a
//             crash), episode (1 from ramp start to crash end)
//   events    [{ start, end, rampStart }] for each crash
function simulateMarket({
  k = 3, T = 10000, seed = 1, herd = 0, episodes = [],
  depth0 = 10, spread0 = 1, noise = 1, impact = 1e-3, priceNoise = 2e-4,
  momentumChase = 0.3, arbitrage = 0.2,
} = {}) {
  if (!(k >= 2 && k <= 4)) throw new Error('simulateMarket: k must be 2..4');
  const rng = mulberry32(seed);
  const baseHerd = typeof herd === 'function' ? herd : () => herd;
  const eps = episodes.map(e => ({ ramp: 0, peak: 0.9, shape: 'ramp', ...e }));

  const features = Array.from({ length: k }, () => ({
    spread: new Float64Array(T), depth: new Float64Array(T), imbalance: new Float64Array(T),
    flow: new Float64Array(T), mid: new Float64Array(T),
  }));
  const regimes = Array.from({ length: k }, () => new Uint8Array(T));
  const labels = {
    herd: new Float64Array(T), regimes, common: new Uint8Array(T),
    crash: new Uint8Array(T), episode: new Uint8Array(T),
  };

  const r = new Array(k).fill(0), dir = new Array(k).fill(1), mid = new Array(k).fill(0), lastRet = new Array(k).fill(0);
  let c = 0, cDir = 1;
  const newDir = (from, to, d) => (to >= 2 && from < 2 ? (rng() < 0.5 ? -1 : 1) : d);

  for (let t = 0; t < T; t++) {
    const crash = eps.some(e => t >= e.start && t <= e.end);
    const h = Math.max(baseHerd(t), episodeHerd(eps, t));
    labels.herd[t] = h;
    labels.crash[t] = crash ? 1 : 0;
    labels.episode[t] = eps.some(e => t >= e.start - e.ramp && t <= e.end) ? 1 : 0;

    // Common regime: ordinary kernel, except a forced sell-side sweep in a crash.
    if (crash) { c = 3; cDir = -1; } else { const nc = sampleIndex(REGIME_KERNEL[c], rng); cDir = newDir(c, nc, cDir); c = nc; }
    labels.common[t] = c;

    for (let i = 0; i < k; i++) {
      if (rng() < h) { r[i] = c; dir[i] = cDir; } else { const nr = sampleIndex(REGIME_KERNEL[r[i]], rng); dir[i] = newDir(r[i], nr, dir[i]); r[i] = nr; }
      regimes[i][t] = r[i];
    }

    const meanMid = mid.reduce((a, b) => a + b, 0) / k;
    for (let i = 0; i < k; i++) {
      const g = r[i], d = dir[i], f = features[i];
      const mm = MM_DEPTH[g] * Math.exp(0.2 * gauss(rng));
      const bid = 0.5 * depth0 * mm * (1 + 0.6 * d * SIDE[g]) * Math.exp(0.25 * gauss(rng));
      const ask = 0.5 * depth0 * mm * (1 - 0.6 * d * SIDE[g]) * Math.exp(0.25 * gauss(rng));
      const chase = momentumChase * Math.tanh(lastRet[i] / (impact * noise));
      const flow = noise * gauss(rng) + d * MOMENTUM[g] * (1 + 0.3 * gauss(rng)) + chase;
      const ret = (impact * flow) / (mm) + priceNoise * gauss(rng) + arbitrage * (meanMid - mid[i]);
      mid[i] += ret;
      lastRet[i] = ret;
      f.depth[t] = bid + ask;
      f.imbalance[t] = (bid - ask) / (bid + ask);
      f.spread[t] = (spread0 / mm) * Math.exp(0.15 * gauss(rng));
      f.flow[t] = flow;
      f.mid[t] = mid[i];
    }
  }

  const events = eps.map(e => ({ start: e.start, end: e.end, rampStart: e.start - e.ramp }));
  return { features, labels, events };
}

// Evenly spaced crash episodes for examples and tests: n crashes of length
// `duration`, each preceded by a herding ramp of length `ramp`, starting
// after `first` and at least `gap` steps apart.
function plantEpisodes({ n, first, gap, ramp = 400, duration = 60, peak = 0.9, shape = 'ramp' }) {
  return Array.from({ length: n }, (_, j) => {
    const start = first + j * gap;
    return { start, end: start + duration - 1, ramp, peak, shape };
  });
}

module.exports = { REGIMES, REGIME_KERNEL, simulateMarket, plantEpisodes, episodeHerd };
