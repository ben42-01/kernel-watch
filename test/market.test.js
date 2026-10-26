// Ground truth for the market herding monitor (src/apps/market.js).
const test = require('node:test');
const assert = require('node:assert/strict');
const { simulateMarket, plantEpisodes } = require('../sim/market');
const A = require('../src/apps/market');

const ALL = [...A.METRICS, 'vol', 'imbalance'];

test('encodeStates recovers the latent regimes from L2 features', () => {
  const k = 3, T = 6000;
  const { features, labels } = simulateMarket({ k, T, seed: 51 });
  const enc = A.encodeStates(features);
  assert.equal(enc.streams.length, k);
  assert.equal(enc.thresholds.length, k);
  let hit = 0;
  for (let i = 0; i < k; i++) {
    assert.ok(enc.streams[i].every(s => s >= 0 && s < 4));
    for (let t = 0; t < T; t++) if (enc.streams[i][t] === labels.regimes[i][t]) hit++;
  }
  assert.ok(hit / (k * T) > 0.75, `accuracy ${hit / (k * T)}`);
  assert.throws(() => A.encodeStates([...features, ...features]), /at most 4/);
});

test('thresholds come from the fit segment only', () => {
  const { features } = simulateMarket({ k: 2, T: 4000, seed: 52 });
  const a = A.encodeStates(features, { fit: { start: 0, end: 1000 } });
  const scaled = features.map(f => ({ ...f, flow: f.flow.map((v, t) => (t >= 1000 ? v * 100 : v)) }));
  const b = A.encodeStates(scaled, { fit: { start: 0, end: 1000 } });
  assert.deepEqual(a.thresholds, b.thresholds);
  assert.ok(b.streams[0].slice(1000).filter(s => s === 3).length > 0.9 * 3000);
});

test('marketMetrics warns about sparse joint rows and caps the joint space', () => {
  const { features } = simulateMarket({ k: 4, T: 1200, seed: 53 });
  const { streams } = A.encodeStates(features, { fit: { start: 0, end: 1200 } });
  const res = A.marketMetrics(streams, { size: 400, step: 400 });
  assert.equal(res.windows.length, 3);
  assert.ok(res.warnings.some(w => /per joint row/.test(w)), res.warnings.join('; '));
  assert.ok(res.windows.every(w => w.sparseMass > 0));
  assert.throws(() => A.marketMetrics(streams, { states: ['a', 'b', 'c', 'd', 'e'] }), /max 256/);
});

test('non-finite metric values are counted and skipped by the monitor', () => {
  const windows = Array.from({ length: 30 }, (_, i) => ({ t: i, F: i === 25 ? Infinity : Math.sin(i), Hpi: 1, dobrushin: 0.5, mixingTime: i % 7 ? 10 : NaN }));
  const m = A.runMonitor(windows, { baseline: 10 });
  assert.equal(m.F.nonFinite, 1);
  assert.equal(m.mixingTime.nonFinite, 5);
  assert.equal(m.F.records[25].alert, false);
  assert.equal(m.Hpi.direction, 'down');
});

test('events during monitor warm-up are excluded from scoring', () => {
  const monitored = { F: { readyAt: 1000, onsets: [500, 1500, 3000], records: [] } };
  const events = [{ start: 800, end: 900 }, { start: 1700, end: 1800 }];
  const r = A.evaluate(monitored, events, { horizon: 300 });
  assert.deepEqual(r.F.excluded, [events[0]]);
  assert.equal(r.F.events.recall, 1);
  assert.deepEqual(r.F.events.leadTimes, [200]);
  assert.equal(r.F.events.falsePositives, 1);
});

// The end-to-end runs below are shared by several tests. The crash lasts
// 200 steps so that it fills half a window.
const k = 3, T = 26000;
const runs = {};
const herdFree = () => (runs.free ??= A.analyzeMarket(simulateMarket({ k, T, seed: 61 })));
const withEpisodes = (shape) => (runs[shape] ??= A.analyzeMarket(simulateMarket({
  k, T, seed: 62, episodes: plantEpisodes({ n: 5, first: 6000, gap: 4000, ramp: 600, duration: 200, shape }),
}), { horizon: 600 }));
const pooled = (name, f) => ['ramp', 'step'].reduce((s, shape) => s + f(withEpisodes(shape).evaluation[name]), 0);
const before = (e) => e.events.leadTimes.filter(l => l > 0).length;

test('herd-free run: every series stays near its nominal false-alarm rate', () => {
  const res = herdFree();
  const rates = ALL.map(name => {
    const live = res.monitored[name].records.filter(r => r.ready && Number.isFinite(r.value));
    return live.filter(r => r.exceed).length / live.length;
  });
  // z = 3 is 0.13% for a Gaussian. Overlapping windows cluster exceedances
  // and the metrics are not Gaussian; across 20 seeds single series reached
  // 5% and the mean over series 1.7%.
  rates.forEach((r, i) => assert.ok(r < 0.07, `${ALL[i]}: exceedance rate ${r}`));
  const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
  assert.ok(mean < 0.025, `mean exceedance rate ${mean}`);
});

test('every series alerts on planted herding episodes (ramped and sudden)', () => {
  for (const name of ALL) {
    for (const shape of ['ramp', 'step']) assert.equal(withEpisodes(shape).evaluation[name].excluded.length, 0);
    // H_pi and Dobrushin are weak (see docs/notes.md): require one hit in ten.
    const detected = pooled(name, e => e.events.detected);
    assert.ok(detected >= 1, `${name}: detected ${detected} of 10`);
  }
});

// Over 51 seed offsets F fired before the crash in 6-10 of 10 episodes and
// never less often than realised vol; at two offsets the margin was 0 or the
// count 6, so these bounds are not tight.
test('fusion rate alerts before the crash at least as often as realised volatility', () => {
  const recall = pooled('F', e => e.events.detected) / 10;
  const fBefore = pooled('F', before), volBefore = pooled('vol', before);
  assert.ok(recall >= 0.8, `F recall ${recall}`);
  assert.ok(fBefore >= 6, `F alerted before ${fBefore} of 10 crashes`);
  assert.ok(fBefore >= volBefore, `F before ${fBefore} vs vol before ${volBefore}`);
});
