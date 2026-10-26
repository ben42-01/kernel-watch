// Checks the claims made in docs/spec.md against conscious-agent@3.
// Run: node examples/00_claims_check.js
const { FormalConsciousAgent, MarkovKernel, markov, bell, mulberry32 } = require('conscious-agent');

const rng = mulberry32(7);
const rank = (M, tol = 1e-9) => {
  const A = M.map(r => Array.from(r));
  let r = 0;
  for (let c = 0; c < A[0].length && r < A.length; c++) {
    let p = r;
    for (let i = r + 1; i < A.length; i++) if (Math.abs(A[i][c]) > Math.abs(A[p][c])) p = i;
    if (Math.abs(A[p][c]) < tol) continue;
    [A[r], A[p]] = [A[p], A[r]];
    for (let i = r + 1; i < A.length; i++) {
      const f = A[i][c] / A[r][c];
      for (let j = c; j < A[0].length; j++) A[i][j] -= f * A[r][j];
    }
    r++;
  }
  return r;
};
const randKernel = (n) => Array.from({ length: n }, () => {
  const row = Array.from({ length: n }, () => rng());
  const s = row.reduce((a, b) => a + b, 0);
  return row.map(v => v / s);
});

// Claim 1: with M_macro = M_1 ⊗ M_2, rank(M_macro) = rank(M_1)·rank(M_2), so the
// spec's rank-collapse ratio is fixed by the individual kernels and cannot see fusion.
const M1 = randKernel(3), M2 = randKernel(3);
console.log('claim 1  rank(M1)=%d rank(M2)=%d rank(M1⊗M2)=%d  ratio=%s',
  rank(M1), rank(M2), rank(markov.kron(M1, M2)), (rank(markov.kron(M1, M2)) / (rank(M1) + rank(M2))).toFixed(2));

// Claim 2: a perfectly polarised (deterministic, identical) pair has CHSH = 2, the classical bound.
const polarised = bell.fromCorrelators([[1, 1], [1, 1]]);
console.log('claim 2  polarised pair: CHSH=%s class=%s', bell.chsh(bell.correlators(polarised)), bell.classify(polarised));

// Claim 3: the fusion rate h(A)+h(B)-h(AB) of two observed streams is ~0 when
// independent and grows as they herd.
const S = ['calm', 'thin', 'panic'];
const countsOf = (pairs, n, enc) => {
  const C = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let t = 1; t < pairs.length; t++) C[enc(pairs[t - 1])][enc(pairs[t])]++;
  return C;
};
const hRate = (K) => markov.entropyRate(K.matrix);
for (const herd of [0, 0.5, 0.9]) {
  const base = randKernel(3);
  let a = 0, b = 0;
  const pairs = [];
  for (let t = 0; t < 20000; t++) {
    const pick = (row) => { let u = rng(), i = 0; while ((u -= row[i]) > 0 && i < row.length - 1) i++; return i; };
    a = pick(base[a]);
    b = rng() < herd ? a : pick(base[b]);
    pairs.push([a, b]);
  }
  const Kab = MarkovKernel.fromCounts({ states: S.flatMap(x => S.map(y => `${x}|${y}`)), counts: countsOf(pairs, 9, ([x, y]) => 3 * x + y), alpha: 0.5 });
  const Ka = MarkovKernel.fromCounts({ states: S, counts: countsOf(pairs.map(p => [p[0]]), 3, ([x]) => x), alpha: 0.5 });
  const Kb = MarkovKernel.fromCounts({ states: S, counts: countsOf(pairs.map(p => [p[1]]), 3, ([y]) => y), alpha: 0.5 });
  console.log('claim 3  herd=%s fusion rate=%s nats/step', herd, (hRate(Ka) + hRate(Kb) - hRate(Kab)).toFixed(3));
}

// Blueprint: a formal agent, its joint kernel on X×W and its diagnostics.
const agent = new FormalConsciousAgent({
  W: ['liquid', 'stressed'], X: ['tight', 'wide'], G: ['provide', 'withdraw'],
  P: { liquid: [[0.9, 0.1], [0.7, 0.3]], stressed: [[0.3, 0.7], [0.1, 0.9]] },
  D: [[0.9, 0.1], [0.2, 0.8]],
  A: { provide: [[0.95, 0.05], [0.6, 0.4]], withdraw: [[0.5, 0.5], [0.1, 0.9]] },
  rng,
});
const d = agent.diagnostics();
console.log('blueprint ergodic=%s mixingTime=%s world=%j', d.ergodic, d.mixingTime.toFixed(2),
  Object.fromEntries(Object.entries(d.worldMarginal).map(([k, v]) => [k, +v.toFixed(3)])));
