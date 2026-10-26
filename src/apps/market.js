// Spec 1: market herding monitor.
//
// Each venue's L2 features are coarse-grained into four states (calm, thin,
// one-sided, sweeping). Per rolling window we estimate the joint kernel of
// the venues and report
//   F           fusion rate Σ h(K_i) − h(K_joint) (pairwise sum above k = 2)
//   Hpi         stationary entropy of K_joint
//   dobrushin   Dobrushin contraction coefficient of K_joint
//   mixingTime  spectral mixing-time estimate of K_joint
// Each metric is scored against its own rolling baseline (createMonitor),
// and so are two simple baselines, realised volatility and |imbalance|, on
// the same windows and monitor settings. These are statistical monitors of
// order-flow streams; nothing here says markets are agents or conscious.

const { markov } = require('conscious-agent');
const { countTransitions, kernelFromCounts, encodeJoint, jointStates, rollingWindows } = require('../kernels');
const { fusionRate, pairwiseFusion, quantile } = require('../info');
const { monitorSeries, alertOnsets, evaluateAlerts, evaluateWindows } = require('../monitor');

const STATES = ['calm', 'thin', 'one-sided', 'sweeping'];
const MAX_JOINT = 256;

// Dirichlet pseudo-count for this app. The library default (0.5) adds 8–32
// pseudo-counts to each 16- or 64-state joint row, which on 400-step windows
// swamps the data and pushes F to about -0.3 (k = 2) or -1.5 (k = 3, pairwise).
// 0.1 keeps herd-free F near 0 and roughly doubles its separation per herd step.
const MARKET_ALPHA = 0.1;

// Trailing exponential moving average with span `span` (span 1 = no smoothing).
function ema(xs, span) {
  const out = new Float64Array(xs.length);
  const a = 2 / (span + 1);
  let m = xs[0];
  for (let t = 0; t < xs.length; t++) { m = t === 0 ? xs[0] : a * xs[t] + (1 - a) * m; out[t] = m; }
  return out;
}

// Coarse state per venue and step. Thresholds are quantiles of each venue's
// own (smoothed) features on the fit segment [fit.start, fit.end), which
// should be free of planted events; nothing is hard-coded in feature units.
//   sweeping   |flow| above its `sweep` quantile
//   one-sided  |imbalance| above its `oneSided` quantile
//   thin       depth below its `thin` lower quantile, or spread above the matching upper one
//   calm       otherwise
// Priority runs top to bottom.
function encodeStates(features, {
  fit = { start: 0, end: Math.min(2000, Math.floor(features[0].flow.length / 4)) },
  smooth = 1, sweep = 0.95, oneSided = 0.85, thin = 0.15,
} = {}) {
  if (features.length > 4) throw new Error('encodeStates: at most 4 streams (joint space <= 256 states)');
  const thresholds = [];
  const streams = features.map(f => {
    const flow = ema(f.flow, smooth).map(Math.abs);
    const imb = ema(f.imbalance, smooth).map(Math.abs);
    const depth = ema(f.depth, smooth), spread = ema(f.spread, smooth);
    const seg = (xs) => xs.subarray(fit.start, fit.end);
    const th = {
      flow: quantile(seg(flow), sweep),
      imbalance: quantile(seg(imb), oneSided),
      depth: quantile(seg(depth), thin),
      spread: quantile(seg(spread), 1 - thin),
    };
    thresholds.push(th);
    return Array.from(flow, (_, t) => {
      if (flow[t] > th.flow) return 3;
      if (imb[t] > th.imbalance) return 2;
      if (depth[t] < th.depth || spread[t] > th.spread) return 1;
      return 0;
    });
  });
  return { streams, states: STATES, thresholds };
}

// Per-window metrics on k encoded streams. Every window uses the same `size`
// and `alpha`, because the fusion rate's finite-window bias depends on both
// and can have either sign: compare F with its rolling baseline, never with 0.
//
//   fusion       'joint' (fusionRate on the full product) or 'pairwise'
//                (pairwiseFusion().total); default joint for k = 2, pairwise above
//   minRowVisits joint rows with fewer transitions than this count as sparse
//
// Returns { windows, warnings, nonFinite }. Each window is
// { start, end, t, F, Hpi, dobrushin, mixingTime, sparseMass } where t = end - 1
// (the last step the window has seen) and sparseMass is the share of the
// window's transitions that leave sparse joint rows.
function marketMetrics(streams, {
  size = 400, step = 40, alpha = MARKET_ALPHA, states = STATES,
  fusion = streams.length > 2 ? 'pairwise' : 'joint', minRowVisits = 5,
} = {}) {
  const k = streams.length;
  const n = states.length;
  const nJoint = n ** k;
  if (nJoint > MAX_JOINT) throw new Error(`marketMetrics: joint space has ${nJoint} states (max ${MAX_JOINT})`);
  const sizes = new Array(k).fill(n);
  const stateSets = new Array(k).fill(states);
  const joint = encodeJoint(streams, sizes);
  const jointLabels = jointStates(stateSets);

  const windows = [];
  for (const { start, end } of rollingWindows(joint.length, { size, step })) {
    const slice = streams.map(s => s.slice(start, end));
    const F = fusion === 'joint'
      ? fusionRate(slice, stateSets, { alpha })
      : pairwiseFusion(slice, stateSets, { alpha }).total;
    const counts = countTransitions(joint, nJoint, { start, end });
    const P = kernelFromCounts(jointLabels, counts, { alpha }).matrix;
    // Direct solve is much faster than diagnostics()' power iteration on 64+ states.
    const pi = markov.solveStationary(P) || markov.stationary(P).pi;
    let sparse = 0;
    for (const row of counts) { const ni = markov.rowSum(row); if (ni > 0 && ni < minRowVisits) sparse += ni; }
    windows.push({
      start, end, t: end - 1,
      F, Hpi: markov.entropy(pi), dobrushin: markov.dobrushin(P), mixingTime: markov.mixingTimeEstimate(P, { pi }).mixingTime,
      sparseMass: sparse / (end - start - 1),
    });
  }

  const nonFinite = {};
  for (const m of METRICS) nonFinite[m] = windows.filter(w => !Number.isFinite(w[m])).length;
  const warnings = [];
  const perRow = (size - 1) / nJoint;
  if (perRow < minRowVisits) {
    warnings.push(`window size ${size} gives ${perRow.toFixed(1)} transitions per joint row on average (< ${minRowVisits}); smoothing biases F down on sparse rows`);
  }
  const medSparse = windows.length ? quantile(windows.map(w => w.sparseMass), 0.5) : 0;
  if (medSparse > 0.1) {
    warnings.push(`median window has ${(100 * medSparse).toFixed(0)}% of transitions in joint rows visited < ${minRowVisits} times; consider a larger window or fewer streams`);
  }
  for (const m of METRICS) if (nonFinite[m]) warnings.push(`${m} is non-finite in ${nonFinite[m]} of ${windows.length} windows (skipped by the monitor)`);
  return { windows, warnings, nonFinite };
}

const METRICS = ['F', 'Hpi', 'dobrushin', 'mixingTime'];
const DIRECTIONS = { F: 'up', Hpi: 'down', dobrushin: 'up', mixingTime: 'up', vol: 'up', imbalance: 'up' };

// Realised volatility of the composite (venue-mean) log mid price and mean
// |imbalance| across venues, on the same windows as `windows`.
function baselines(features, windows) {
  const k = features.length, T = features[0].mid.length;
  const ret = new Float64Array(T);
  for (let t = 1; t < T; t++) {
    let r = 0;
    for (const f of features) r += f.mid[t] - f.mid[t - 1];
    ret[t] = r / k;
  }
  return windows.map(({ start, end, t }) => {
    let rv = 0, imb = 0;
    for (let s = start + 1; s < end; s++) rv += ret[s] * ret[s];
    for (let s = start; s < end; s++) for (const f of features) imb += Math.abs(f.imbalance[s]);
    return { start, end, t, vol: Math.sqrt(rv), imbalance: imb / ((end - start) * k) };
  });
}

// One createMonitor per series over the windows. `series` lists the fields
// of each window to monitor; directions default to DIRECTIONS (up for F,
// mixing time, Dobrushin and the baselines; down for H_pi). Alert times are
// the windows' t (causal: the window has ended).
//
// Overlapping windows are strongly autocorrelated, so the baseline must span
// many window lengths: with 50 windows of step 20 (1000 steps, ~2.5
// independent windows) herd-free exceedance rates reached 5-11% at z = 3.
// The default, 100 windows at step 40, spans 4000 steps.
function runMonitor(windows, {
  series = METRICS, method = 'zscore', threshold, baseline = 100, warmup, consecutive = 1, directions = {},
} = {}) {
  const times = windows.map(w => w.t);
  const out = {};
  for (const name of series) {
    const opts = { method, baseline, warmup, consecutive, direction: directions[name] || DIRECTIONS[name] || 'up' };
    if (threshold !== undefined) opts.threshold = threshold;
    const records = monitorSeries(windows.map(w => w[name]), opts, times);
    const ready = records.find(r => r.ready);
    out[name] = {
      direction: opts.direction,
      records,
      onsets: alertOnsets(records),
      readyAt: ready ? ready.t : Infinity,
      nonFinite: records.filter(r => !Number.isFinite(r.value)).length,
    };
  }
  return out;
}

// Precision, recall and lead time for each monitored series. An alert counts
// for an event if it falls in [start - horizon, end]. Events whose
// [start - horizon] lies before the monitor is ready (warm-up) cannot be
// scored fairly and are excluded (and listed), together with alerts before
// readyAt or inside an excluded event's span, which are dropped rather than
// counted as false. Window-level scores label a window
// positive if its t lies in [start - horizon, end] of a scored event.
function evaluate(monitored, events, { horizon = 0 } = {}) {
  const out = {};
  for (const [name, m] of Object.entries(monitored)) {
    const scored = events.filter(e => e.start - horizon >= m.readyAt);
    const excluded = events.filter(e => e.start - horizon < m.readyAt);
    const inExcluded = (t) => excluded.some(e => t >= e.start - horizon && t <= e.end);
    const onsets = m.onsets.filter(a => a >= m.readyAt && !inExcluded(a));
    const kept = m.records.filter(r => r.ready && !inExcluded(r.t));
    out[name] = {
      events: evaluateAlerts(onsets, scored, { horizon }),
      windows: evaluateWindows(kept.map(r => r.alert), kept.map(r => scored.some(e => r.t >= e.start - horizon && r.t <= e.end))),
      excluded,
    };
  }
  return out;
}

// End-to-end run: encode, window, monitor and evaluate the kernel metrics
// and the baselines with identical settings. A window that ends after a
// crash still contains it, so for scoring each event's end is extended by
// size - 1: an alert from such a window is a late detection, not a false
// alarm. Lead times are measured from the event start and are unaffected.
function analyzeMarket({ features, events = [] }, {
  encode = {}, window = {}, monitor = {}, horizon = 600,
} = {}) {
  const enc = encodeStates(features, encode);
  const metrics = marketMetrics(enc.streams, window);
  const size = metrics.windows.length ? metrics.windows[0].end - metrics.windows[0].start : 0;
  const scoring = events.map(e => ({ ...e, end: e.end + Math.max(0, size - 1) }));
  const base = baselines(features, metrics.windows);
  const rows = metrics.windows.map((w, i) => ({ ...w, vol: base[i].vol, imbalance: base[i].imbalance }));
  const monitored = runMonitor(rows, { ...monitor, series: [...METRICS, 'vol', 'imbalance'] });
  return {
    encoding: enc, windows: rows, warnings: metrics.warnings, nonFinite: metrics.nonFinite,
    monitored, evaluation: evaluate(monitored, scoring, { horizon }),
  };
}

module.exports = {
  STATES, METRICS, DIRECTIONS, MARKET_ALPHA,
  ema, encodeStates, marketMetrics, baselines, runMonitor, evaluate, analyzeMarket,
};
