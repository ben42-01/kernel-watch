const test = require('node:test');
const assert = require('node:assert/strict');
const { mulberry32 } = require('conscious-agent');
const M = require('../src/monitor');

function gauss(rng) {
  return Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
}

test('RollingBaseline keeps the last `size` values', () => {
  const b = new M.RollingBaseline(3);
  [1, 2, 3, 4].forEach(x => b.push(x));
  assert.deepEqual(Array.from(b.values()).sort(), [2, 3, 4]);
  assert.equal(b.mean(), 3);
  assert.equal(b.sd(), 1);
  assert.equal(b.cdf(3), 0.5);
});

for (const method of ['zscore', 'quantile']) {
  test(`${method} monitor: few alerts on a stationary stream, quick alert on a planted shift`, () => {
    const rng = mulberry32(method.length);
    const T = 3000, shiftAt = 2000;
    const flat = Array.from({ length: T }, () => gauss(rng));
    const opts = { method, baseline: 300, threshold: method === 'zscore' ? 3.5 : 0.995, consecutive: 2 };
    const quiet = M.monitorSeries(flat, opts).filter(r => r.alert).length;
    assert.ok(quiet <= 5, `false alerts on flat stream: ${quiet}`);

    const shifted = flat.map((x, t) => (t >= shiftAt ? x + 6 : x));
    const recs = M.monitorSeries(shifted, opts);
    const onsets = M.alertOnsets(recs);
    const first = onsets.find(t => t >= shiftAt);
    assert.ok(first !== undefined && first - shiftAt < 10, `latency ${first - shiftAt}`);
    // The anomaly is kept out of the baseline, so the alert persists.
    assert.ok(recs.slice(shiftAt + 50).filter(r => r.alert).length > 0.8 * (T - shiftAt - 50));
  });
}

test('false-alarm rate stays near nominal on a long stationary stream (no threshold ratchet)', () => {
  const rng = mulberry32(12);
  const xs = Array.from({ length: 20000 }, () => gauss(rng));
  for (const opts of [{ method: 'quantile', threshold: 0.99 }, { method: 'zscore', threshold: 2.326 }]) {
    const recs = M.monitorSeries(xs, { ...opts, baseline: 500 });
    const late = recs.slice(15000).filter(r => r.exceed).length / 5000;
    assert.ok(late > 0.004 && late < 0.02, `${opts.method} late exceedance rate ${late}`);
  }
});

test('direction controls which side alerts', () => {
  const rng = mulberry32(4);
  const xs = Array.from({ length: 400 }, (_, t) => gauss(rng) + (t >= 300 ? -6 : 0));
  assert.equal(M.monitorSeries(xs, { baseline: 200, direction: 'up' }).slice(300).filter(r => r.alert).length, 0);
  assert.ok(M.monitorSeries(xs, { baseline: 200, direction: 'down' }).slice(300).filter(r => r.alert).length > 90);
  assert.ok(M.monitorSeries(xs, { baseline: 200, direction: 'both' }).slice(300).filter(r => r.alert).length > 90);
});

test('non-finite values are skipped, not scored', () => {
  const m = M.createMonitor({ baseline: 2, warmup: 2 });
  m.push(1); m.push(2);
  const r = m.push(Infinity);
  assert.equal(r.alert, false);
  assert.ok(Number.isNaN(r.score));
});

test('evaluateAlerts: precision, recall and signed lead time', () => {
  const events = [{ start: 100, end: 120 }, { start: 300, end: 310 }, { start: 500, end: 505 }];
  const alerts = [95, 104, 200, 305];
  const r = M.evaluateAlerts(alerts, events, { horizon: 10 });
  assert.equal(r.truePositives, 3);
  assert.equal(r.falsePositives, 1);
  assert.equal(r.precision, 0.75);
  assert.equal(r.recall, 2 / 3);
  assert.deepEqual(r.leadTimes, [5, -5]);
});

test('evaluateWindows', () => {
  const r = M.evaluateWindows([true, true, false, false, true], [true, false, true, false, true]);
  assert.deepEqual([r.tp, r.fp, r.fn, r.tn], [2, 1, 1, 1]);
  assert.equal(r.precision, 2 / 3);
  assert.equal(r.recall, 2 / 3);
});
