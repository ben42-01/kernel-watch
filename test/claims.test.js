// The claims of docs/spec.md, ported from examples/00_claims_check.js.
const test = require('node:test');
const assert = require('node:assert/strict');
const { FormalConsciousAgent, markov, bell, mulberry32 } = require('conscious-agent');
const { fusionRate } = require('../src/info');
const { randomKernel, herdedStreams } = require('../sim/chains');

function rank(M, tol = 1e-9) {
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
}

test('claim 1: rank of a Kronecker product is the product of ranks, so the rank-collapse ratio cannot see fusion', () => {
  const rng = mulberry32(7);
  for (let rep = 0; rep < 5; rep++) {
    const M1 = randomKernel(3, rng), M2 = randomKernel(3, rng);
    assert.equal(rank(markov.kron(M1, M2)), rank(M1) * rank(M2));
  }
});

test('claim 2: a perfectly polarised pair sits exactly on the classical CHSH bound', () => {
  const p = bell.fromCorrelators([[1, 1], [1, 1]]);
  assert.equal(bell.chsh(bell.correlators(p)), 2);
  assert.equal(bell.classify(p), 'local');
});

test('claim 3: fusion rate is ~0 for independent streams and grows with herding', () => {
  const S = ['calm', 'thin', 'panic'];
  const F = [0, 0.5, 0.9].map((herd, i) => {
    const rng = mulberry32(100 + i);
    const base = randomKernel(3, rng);
    return fusionRate(herdedStreams({ kernels: base, T: 20000, herd, rng }), [S, S]);
  });
  assert.ok(Math.abs(F[0]) < 0.01, `independent F=${F[0]}`);
  assert.ok(F[1] > 0.1 && F[2] > F[1] + 0.1, `F=${F}`);
});

test('blueprint: a formal agent has an ergodic joint kernel on X×W', () => {
  const agent = new FormalConsciousAgent({
    W: ['liquid', 'stressed'], X: ['tight', 'wide'], G: ['provide', 'withdraw'],
    P: { liquid: [[0.9, 0.1], [0.7, 0.3]], stressed: [[0.3, 0.7], [0.1, 0.9]] },
    D: [[0.9, 0.1], [0.2, 0.8]],
    A: { provide: [[0.95, 0.05], [0.6, 0.4]], withdraw: [[0.5, 0.5], [0.1, 0.9]] },
    rng: mulberry32(7),
  });
  const d = agent.diagnostics();
  assert.equal(d.ergodic, true);
  assert.ok(Number.isFinite(d.mixingTime) && d.mixingTime > 0);
  const w = Object.values(d.worldMarginal);
  assert.ok(Math.abs(w.reduce((a, b) => a + b, 0) - 1) < 1e-9);
});
