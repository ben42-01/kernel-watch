// Ground truth for the market simulator (sim/market.js).
const test = require('node:test');
const assert = require('node:assert/strict');
const { mulberry32 } = require('conscious-agent');
const { simulateMarket, plantEpisodes, episodeHerd } = require('../sim/market');
const { encodeStates } = require('../src/apps/market');
const { contingency, mutualInformation, bootstrap, fusionRate, pairwiseFusion } = require('../src/info');

const STATES = ['calm', 'thin', 'one-sided', 'sweeping'];

test('same seed gives identical output; another seed does not', () => {
  const opts = { k: 3, T: 2000, episodes: plantEpisodes({ n: 1, first: 1000, gap: 1, ramp: 200 }) };
  const a = simulateMarket({ ...opts, seed: 5 }), b = simulateMarket({ ...opts, seed: 5 }), c = simulateMarket({ ...opts, seed: 6 });
  for (let i = 0; i < 3; i++) for (const f of ['spread', 'depth', 'imbalance', 'flow', 'mid']) {
    assert.deepEqual(a.features[i][f], b.features[i][f]);
  }
  assert.deepEqual(a.labels, b.labels);
  assert.deepEqual(a.events, b.events);
  assert.notDeepEqual(a.features[0].flow, c.features[0].flow);
});

test('episode herding: ramp rises linearly, step jumps, both hold through the crash', () => {
  const ramp = [{ start: 100, end: 150, ramp: 50, peak: 0.8, shape: 'ramp' }];
  const step = [{ ...ramp[0], shape: 'step' }];
  assert.equal(episodeHerd(ramp, 49), 0);
  assert.ok(Math.abs(episodeHerd(ramp, 74) - 0.4) < 1e-12);
  assert.equal(episodeHerd(ramp, 120), 0.8);
  assert.equal(episodeHerd(ramp, 151), 0);
  assert.equal(episodeHerd(step, 50), 0.8);
  const sim = simulateMarket({ k: 2, T: 300, seed: 1, herd: (t) => (t < 10 ? 0.3 : 0), episodes: ramp });
  assert.equal(sim.labels.herd[5], 0.3);
  assert.equal(sim.labels.herd[120], 0.8);
  assert.equal(sim.labels.crash[99], 0);
  assert.equal(sim.labels.crash[100], 1);
  assert.equal(sim.labels.episode[50], 1);
  assert.deepEqual(sim.events, [{ start: 100, end: 150, rampStart: 50 }]);
});

// Cross-venue MI of the encoded states, with a moving-block bootstrap CI and
// a null baseline from circularly shifted pairings (same marginals, no
// alignment), which carries the same plug-in bias. The CI is the normal
// interval estimate ± 2.58·sd: resampling adds its own upward plug-in bias,
// so the percentile interval sits too high and missed the null in ~30% of
// herd-free seeds.
function miWithNull(herd, seed) {
  const T = 8000;
  const { features } = simulateMarket({ k: 2, T, seed, herd });
  const [x, y] = encodeStates(features).streams;
  const pairs = x.map((v, t) => [v, y[t]]);
  const mi = (ps) => mutualInformation(contingency(ps.map(p => p[0]), ps.map(p => p[1]), 4, 4));
  const boot = bootstrap(pairs, mi, { resamples: 200, blockSize: 100, rng: mulberry32(seed + 1) });
  let nullMI = 0;
  const shifts = [1000, 2000, 3000, 4000, 5000, 6000, 7000];
  for (const s of shifts) nullMI += mi(x.map((v, t) => [v, y[(t + s) % T]])) / shifts.length;
  return { ci: [boot.estimate - 2.58 * boot.sd, boot.estimate + 2.58 * boot.sd], nullMI };
}

test('herd = 0: cross-venue state MI is at its null baseline; herding moves it off', () => {
  const free = miWithNull(0, 21);
  assert.ok(free.ci[0] <= free.nullMI && free.nullMI <= free.ci[1], `null ${free.nullMI} outside CI ${free.ci}`);
  const herded = miWithNull(0.5, 21);
  assert.ok(herded.ci[0] > herded.nullMI, `herd 0.5: CI ${herded.ci} vs null ${herded.nullMI}`);
});

test('fusion rate of the encoded streams rises monotonically with herding', () => {
  for (const k of [2, 3]) {
    const F = [0, 0.5, 0.9].map(herd => {
      const { streams } = encodeStates(simulateMarket({ k, T: 8000, seed: 31, herd }).features);
      const sets = streams.map(() => STATES);
      return k === 2 ? fusionRate(streams, sets, { alpha: 0.1 }) : pairwiseFusion(streams, sets, { alpha: 0.1 }).total;
    });
    assert.ok(F[0] < F[1] && F[1] < F[2], `k=${k}: F = ${F}`);
    assert.ok(Math.abs(F[0]) < 0.05, `k=${k}: herd-free F = ${F[0]}`);
  }
});

test('a leading signal exists: regimes align during the ramp before volatility rises', () => {
  const k = 3, T = 26000;
  const episodes = plantEpisodes({ n: 5, first: 6000, gap: 4000, ramp: 600, duration: 80 });
  const { features, labels, events } = simulateMarket({ k, T, seed: 41, episodes });
  const ret = (t) => features.reduce((s, f) => s + f.mid[t] - f.mid[t - 1], 0) / k;
  const segs = { quiet: [], ramp: [], crash: [] };
  for (let t = 1; t < T; t++) {
    const inRamp = events.some(e => t >= e.start - 300 && t < e.start); // second half of the ramp
    const seg = labels.crash[t] ? 'crash' : inRamp ? 'ramp' : labels.episode[t] || events.some(e => t > e.end && t < e.end + 400) ? null : 'quiet';
    if (seg) segs[seg].push(t);
  }
  const agree = (ts) => ts.filter(t => labels.regimes.every(r => r[t] === labels.regimes[0][t])).length / ts.length;
  const vol = (ts) => Math.sqrt(ts.reduce((s, t) => s + ret(t) ** 2, 0) / ts.length);
  // Over 30 seeds: agreement ~0.80 vs ~0.38, ramp vol 1.03-1.38x quiet, crash ~6.3x.
  assert.ok(agree(segs.ramp) > agree(segs.quiet) + 0.3, `agreement ramp ${agree(segs.ramp)} quiet ${agree(segs.quiet)}`);
  assert.ok(vol(segs.ramp) < 1.6 * vol(segs.quiet), `vol ramp ${vol(segs.ramp)} quiet ${vol(segs.quiet)}`);
  assert.ok(vol(segs.crash) > 4 * vol(segs.quiet), `vol crash ${vol(segs.crash)} quiet ${vol(segs.quiet)}`);
});
