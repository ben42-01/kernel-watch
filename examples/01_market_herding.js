// Market herding monitor (docs/spec.md, Spec 1) on the seeded synthetic market.
// Run: node examples/01_market_herding.js [seed]
//
// Three venues, one instrument. Five crash episodes are planted, each after a
// 600-step herding ramp (or a sudden switch-on, in the second scenario). The
// kernel metrics and the two baselines go through the same windows, monitor
// (z = 3 against a rolling baseline) and scoring. Lead = crash start − first
// alert, so positive means the alert came before the crash.
//
// This is a statistical monitor of order-flow streams. It does not show that
// venues are agents in any deeper sense.
const { simulateMarket, plantEpisodes } = require('../sim/market');
const { market } = require('../src');

const seed = Number(process.argv[2] || 1);
const k = 3, T = 26000, horizon = 600;
const SERIES = [...market.METRICS, 'vol', 'imbalance'];
const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');
const pad = (s, n) => String(s).padEnd(n);

function trace(res, events) {
  // Metric value at a few points around each event, as z against the
  // herd-free part of the run before the first ramp.
  const pre = res.windows.filter(w => w.t < events[0].rampStart);
  const z = {};
  for (const m of SERIES) {
    const v = pre.map(w => w[m]).filter(Number.isFinite);
    const mu = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / (v.length - 1));
    z[m] = (x) => (x - mu) / sd;
  }
  const at = (t) => res.windows.reduce((best, w) => (Math.abs(w.t - t) < Math.abs(best.t - t) ? w : best));
  console.log(`  z-scores vs pre-episode windows (first event; ramp ${events[0].rampStart}, crash ${events[0].start}–${events[0].end})`);
  console.log(`  ${pad('t', 8)}${SERIES.map(m => pad(m, 11)).join('')}`);
  const e = events[0];
  for (const t of [e.rampStart - 400, e.rampStart, e.rampStart + 300, e.start - 1, e.end, e.end + 400]) {
    const w = at(t);
    console.log(`  ${pad(w.t, 8)}${SERIES.map(m => pad(fmt(z[m](w[m])), 11)).join('')}`);
  }
}

function table(res) {
  console.log(`  ${pad('series', 12)}${pad('dir', 6)}${pad('recall', 8)}${pad('prec.', 8)}${pad('med.lead', 10)}${pad('leads', 30)}win.F1`);
  for (const m of SERIES) {
    const { events: ev, windows: w } = res.evaluation[m];
    console.log(`  ${pad(m, 12)}${pad(res.monitored[m].direction, 6)}${pad(fmt(ev.recall), 8)}${pad(fmt(ev.precision), 8)}` +
      `${pad(fmt(ev.medianLead, 0), 10)}${pad(ev.leadTimes.join(',') || '—', 30)}${fmt(w.f1)}`);
  }
  const ex = res.evaluation.F.excluded;
  if (ex.length) console.log(`  excluded (inside monitor warm-up): ${ex.map(e => e.start).join(', ')}`);
}

function scenario(title, episodes) {
  const sim = simulateMarket({ k, T, seed, episodes });
  const res = market.analyzeMarket(sim, { horizon });
  console.log(`\n## ${title}`);
  for (const w of res.warnings) console.log(`  warning: ${w}`);
  trace(res, sim.events);
  console.log('  alert onsets:');
  for (const m of SERIES) console.log(`    ${pad(m, 11)}${res.monitored[m].onsets.join(' ') || '—'}`);
  console.log(`  crashes at: ${sim.events.map(e => e.start).join(' ')}`);
  table(res);
  return res;
}

console.log(`Market herding monitor, seed ${seed}: k = ${k} venues, T = ${T}, window 400 / step 40, alpha ${market.MARKET_ALPHA}, z = 3, baseline 100 windows`);

const free = market.analyzeMarket(simulateMarket({ k, T, seed }), { horizon });
console.log('\n## Herd-free run: exceedance rate per window (nominal 0.0013 for a Gaussian at z = 3)');
for (const m of SERIES) {
  const live = free.monitored[m].records.filter(r => r.ready && Number.isFinite(r.value));
  console.log(`  ${pad(m, 12)}${fmt(live.filter(r => r.exceed).length / live.length, 4)}  (${free.monitored[m].onsets.length} alert onsets)`);
}
const nf = Object.entries(free.nonFinite).filter(([, n]) => n).map(([m, n]) => `${m}: ${n}`);
console.log(`  non-finite windows: ${nf.join(', ') || 'none'}`);

scenario('Ramped herding (600 steps) before each 80-step crash',
  plantEpisodes({ n: 5, first: 6000, gap: 4000, ramp: 600, duration: 80 }));
scenario('Sudden herding (step to 0.9, 600 steps) before each 80-step crash',
  plantEpisodes({ n: 5, first: 6000, gap: 4000, ramp: 600, duration: 80, shape: 'step' }));
