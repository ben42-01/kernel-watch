// Seeded generators for Markov streams with planted effects. Used by the
// ground-truth tests and as building blocks for the application simulators.

const { mulberry32 } = require('conscious-agent');

// Inverse-CDF draw from a probability row; consumes one rng() draw.
function sampleIndex(row, rng) {
  let u = rng();
  for (let i = 0; i < row.length; i++) { u -= row[i]; if (u < 0) return i; }
  for (let i = row.length - 1; i >= 0; i--) if (row[i] > 0) return i;
  return row.length - 1;
}

// Random n x n kernel with Dirichlet(concentration) rows. Small concentration
// gives peaked rows (strong structure); 1 is uniform on the simplex.
function randomKernel(n, rng = mulberry32(1), { concentration = 1 } = {}) {
  return Array.from({ length: n }, () => {
    const row = Array.from({ length: n }, () => gamma(concentration, rng));
    const s = row.reduce((a, b) => a + b, 0);
    return row.map(v => v / s);
  });
}

function gamma(k, rng) {
  if (k === 1) return -Math.log(1 - rng());
  if (k < 1) return gamma(k + 1, rng) * Math.pow(rng(), 1 / k);
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      // Box–Muller standard normal
      x = Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
}

// T states of the chain P from `start`.
function sampleChain(P, T, rng, { start = 0 } = {}) {
  const seq = new Array(T);
  let s = start;
  for (let t = 0; t < T; t++) { s = sampleIndex(P[s], rng); seq[t] = s; }
  return seq;
}

// Convex mix (1 - eps) P + eps Q, row by row.
function mixKernels(P, Q, eps) {
  return P.map((row, i) => row.map((v, j) => (1 - eps) * v + eps * Q[i][j]));
}

// k streams on the same state space. Stream 0 follows its own kernel; each
// other stream copies stream 0's current state with probability `herd` and
// otherwise steps by its own kernel. herd = 0 gives independent streams.
// `kernels` is one kernel per stream (or one shared kernel).
function herdedStreams({ kernels, T, herd = 0, k = 2, rng }) {
  const K = Array.isArray(kernels[0][0]) ? kernels : Array.from({ length: k }, () => kernels);
  const herdAt = typeof herd === 'function' ? herd : () => herd;
  const streams = Array.from({ length: k }, () => new Array(T));
  const cur = new Array(k).fill(0);
  for (let t = 0; t < T; t++) {
    const h = herdAt(t);
    cur[0] = sampleIndex(K[0][cur[0]], rng);
    for (let i = 1; i < k; i++) cur[i] = rng() < h ? cur[0] : sampleIndex(K[i][cur[i]], rng);
    for (let i = 0; i < k; i++) streams[i][t] = cur[i];
  }
  return streams;
}

module.exports = { sampleIndex, randomKernel, sampleChain, mixKernels, herdedStreams };
