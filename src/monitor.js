// Baseline calibration and alert streams.
//
// No metric carries a fixed alert level. Each value is scored against a
// rolling baseline of the same metric (z-score or empirical quantile), and the
// threshold is expressed in those calibrated units. Evaluation helpers report
// precision, recall and lead time against labelled events.

const { quantile } = require('./info');

// Fixed-size ring buffer of recent finite values.
class RollingBaseline {
  constructor(size) {
    if (!(size >= 2)) throw new Error('RollingBaseline: size must be >= 2');
    this.size = size;
    this.buf = new Float64Array(size);
    this.n = 0;
    this.head = 0;
  }

  push(x) {
    this.buf[this.head] = x;
    this.head = (this.head + 1) % this.size;
    if (this.n < this.size) this.n++;
  }

  values() { return this.n < this.size ? this.buf.slice(0, this.n) : this.buf; }

  mean() {
    let s = 0;
    for (const v of this.values()) s += v;
    return s / this.n;
  }

  sd() {
    const m = this.mean();
    let s = 0;
    for (const v of this.values()) s += (v - m) ** 2;
    return Math.sqrt(s / Math.max(1, this.n - 1));
  }

  quantile(q) { return quantile(this.values(), q); }

  // Mid-rank empirical CDF of x within the baseline, in [0, 1].
  cdf(x) {
    let below = 0, equal = 0;
    for (const v of this.values()) { if (v < x) below++; else if (v === x) equal++; }
    return (below + 0.5 * equal) / this.n;
  }
}

// Online monitor for one metric.
//
//   method      'zscore' (score = (x - mean) / sd) or 'quantile' (score = baseline CDF of x)
//   threshold   z units for 'zscore', a quantile level in (0.5, 1) for 'quantile'
//   direction   'up', 'down' or 'both'
//   baseline    size of the rolling baseline
//   warmup      values needed before scoring starts (default: baseline size)
//   consecutive exceedances needed before an alert fires
//   learnFromAlerts  if false (default), sustained runs of exceedances (length
//               >= max(consecutive, 2)) are kept out of the baseline so an
//               ongoing anomaly does not become the new normal. Isolated
//               exceedances still enter it: excluding every value above the
//               threshold would truncate the baseline's tail and ratchet the
//               threshold down.
function createMonitor({
  method = 'zscore', threshold = method === 'zscore' ? 3 : 0.99, direction = 'up',
  baseline = 200, warmup = baseline, consecutive = 1, learnFromAlerts = false, minSd = 1e-12,
} = {}) {
  if (!['zscore', 'quantile'].includes(method)) throw new Error(`unknown method "${method}"`);
  if (!['up', 'down', 'both'].includes(direction)) throw new Error(`unknown direction "${direction}"`);
  if (method === 'quantile' && !(threshold > 0.5 && threshold < 1)) throw new Error('quantile threshold must be in (0.5, 1)');
  const base = new RollingBaseline(baseline);
  const excludeAfter = Math.max(consecutive, 2);
  let run = 0, t = 0, pending = [], dropping = false;

  const exceeds = (score) => {
    if (method === 'zscore') {
      return direction === 'up' ? score > threshold : direction === 'down' ? score < -threshold : Math.abs(score) > threshold;
    }
    return direction === 'up' ? score > threshold : direction === 'down' ? score < 1 - threshold : (score > threshold || score < 1 - threshold);
  };

  function push(value, time = t) {
    t = time + 1;
    if (!Number.isFinite(value)) return { t: time, value, score: NaN, exceed: false, alert: false, ready: base.n >= warmup };
    if (base.n < warmup) {
      base.push(value);
      return { t: time, value, score: NaN, exceed: false, alert: false, ready: false };
    }
    const score = method === 'zscore' ? (value - base.mean()) / Math.max(base.sd(), minSd) : base.cdf(value);
    const exceed = exceeds(score);
    run = exceed ? run + 1 : 0;
    if (learnFromAlerts) base.push(value);
    else if (exceed) {
      if (!dropping) pending.push(value);
      if (run >= excludeAfter) { dropping = true; pending = []; }
    } else {
      for (const v of pending) base.push(v);
      pending = [];
      dropping = false;
      base.push(value);
    }
    return { t: time, value, score, exceed, alert: run >= consecutive, ready: true };
  }

  return { push, baseline: base };
}

// Run a monitor over an array; `times` defaults to the array index.
function monitorSeries(values, opts = {}, times = null) {
  const m = createMonitor(opts);
  return values.map((v, i) => m.push(v, times ? times[i] : i));
}

// First time of each run of consecutive alerts (one alert per episode).
function alertOnsets(records) {
  const out = [];
  let prev = false;
  for (const r of records) { if (r.alert && !prev) out.push(r.t); prev = r.alert; }
  return out;
}

// Event-level evaluation. events = [{ start, end }]. An alert at time a
// counts for an event if start - horizon <= a <= end. lead = start - first
// matching alert (positive: before onset; negative: detection latency).
function evaluateAlerts(alertTimes, events, { horizon = 0 } = {}) {
  const hit = (a, e) => a >= e.start - horizon && a <= e.end;
  const tp = alertTimes.filter(a => events.some(e => hit(a, e))).length;
  const leadTimes = [];
  let detected = 0;
  for (const e of events) {
    const first = alertTimes.filter(a => hit(a, e)).reduce((m, a) => Math.min(m, a), Infinity);
    if (first < Infinity) { detected++; leadTimes.push(e.start - first); }
  }
  return {
    precision: alertTimes.length ? tp / alertTimes.length : NaN,
    recall: events.length ? detected / events.length : NaN,
    truePositives: tp,
    falsePositives: alertTimes.length - tp,
    detected,
    missed: events.length - detected,
    leadTimes,
    medianLead: leadTimes.length ? quantile(leadTimes, 0.5) : NaN,
  };
}

// Window-level evaluation of boolean flags against boolean labels.
function evaluateWindows(flags, labels) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  flags.forEach((f, i) => {
    if (f && labels[i]) tp++; else if (f) fp++; else if (labels[i]) fn++; else tn++;
  });
  const precision = tp + fp ? tp / (tp + fp) : NaN;
  const recall = tp + fn ? tp / (tp + fn) : NaN;
  return {
    tp, fp, fn, tn, precision, recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : NaN,
    falsePositiveRate: fp + tn ? fp / (fp + tn) : NaN,
  };
}

module.exports = { RollingBaseline, createMonitor, monitorSeries, alertOnsets, evaluateAlerts, evaluateWindows };
