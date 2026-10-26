// Kernel estimation from observed streams: transition counts, joint-state
// encoding for several streams, and rolling windows.
//
// A stream is an array of state indices 0..n-1. Labels live in a separate
// `states` array so the maths never touches strings.

const { MarkovKernel } = require('conscious-agent');

const DEFAULT_ALPHA = 0.5;

// n x n matrix of transition counts seq[t-1] -> seq[t] for t in (start, end).
// The window [start, end) holds end - start states and end - start - 1 transitions.
function countTransitions(seq, n, { start = 0, end = seq.length } = {}) {
  const counts = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let t = Math.max(start, 0) + 1; t < end; t++) {
    const a = seq[t - 1], b = seq[t];
    if (!(a >= 0 && a < n && b >= 0 && b < n)) throw new Error(`state index out of range at t=${t}: ${a} -> ${b}`);
    counts[a][b]++;
  }
  return counts;
}

// alpha > 0 (a symmetric Dirichlet pseudo-count) keeps unseen transitions
// possible. With alpha = 0 the library turns unobserved rows into self-loops,
// i.e. spurious absorbing states, so only use it on data that visits every state.
function kernelFromCounts(states, counts, { alpha = DEFAULT_ALPHA } = {}) {
  return MarkovKernel.fromCounts({ states, counts, alpha });
}

function kernelFromStream(states, seq, { alpha = DEFAULT_ALPHA, start, end } = {}) {
  return kernelFromCounts(states, countTransitions(seq, states.length, { start, end }), { alpha });
}

// Labels of the product space X_1 × … × X_k in mixed-radix order (the first
// stream is the most significant digit). This is the order markov.kron uses,
// so kron(K_1, K_2) and the joint kernel share row indices.
function jointStates(stateSets) {
  return stateSets
    .reduce((acc, s) => acc.flatMap(a => s.map(b => `${a}|${b}`)), [''])
    .map(l => l.slice(1));
}

// Encode k aligned streams as one stream on the product space.
function encodeJoint(streams, sizes) {
  if (streams.length === 0) return [];
  const T = streams[0].length;
  if (streams.some(s => s.length !== T)) throw new Error('encodeJoint: streams must have equal length');
  const out = new Array(T);
  for (let t = 0; t < T; t++) {
    let code = 0;
    for (let i = 0; i < streams.length; i++) code = code * sizes[i] + streams[i][t];
    out[t] = code;
  }
  return out;
}

function decodeJoint(code, sizes) {
  const out = new Array(sizes.length);
  for (let i = sizes.length - 1; i >= 0; i--) { out[i] = code % sizes[i]; code = Math.floor(code / sizes[i]); }
  return out;
}

// Window bounds [start, end) over a stream of `length` items.
function* rollingWindows(length, { size, step = size }) {
  if (!(size >= 2)) throw new Error('rollingWindows: size must be >= 2');
  if (!(step >= 1)) throw new Error('rollingWindows: step must be >= 1');
  for (let start = 0; start + size <= length; start += step) yield { start, end: start + size };
}

// One kernel per window, with its raw counts (the counts are what the
// divergence tests need; the kernel is smoothed).
function rollingKernels(states, seq, { size, step = size, alpha = DEFAULT_ALPHA } = {}) {
  const out = [];
  for (const { start, end } of rollingWindows(seq.length, { size, step })) {
    const counts = countTransitions(seq, states.length, { start, end });
    out.push({ start, end, counts, kernel: kernelFromCounts(states, counts, { alpha }) });
  }
  return out;
}

module.exports = {
  DEFAULT_ALPHA,
  countTransitions, kernelFromCounts, kernelFromStream,
  jointStates, encodeJoint, decodeJoint,
  rollingWindows, rollingKernels,
};
