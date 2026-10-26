const test = require('node:test');
const assert = require('node:assert/strict');
const { markov, mulberry32 } = require('conscious-agent');
const K = require('../src/kernels');
const { randomKernel, sampleChain } = require('../sim/chains');

// Row-wise TV error of an estimate, with a tolerance that scales with the
// number of visits to each row (~2.5x the expected sampling error).
function assertRowsClose(P, Khat, seq) {
  const visits = new Array(P.length).fill(0);
  for (let t = 0; t < seq.length - 1; t++) visits[seq[t]]++;
  P.forEach((row, i) => {
    const tv = markov.totalVariation(row, Khat.matrix[i]);
    assert.ok(tv < Math.sqrt(row.length / visits[i]), `row ${i}: TV ${tv} after ${visits[i]} visits`);
  });
}

test('countTransitions counts each step once and respects window bounds', () => {
  const seq = [0, 1, 1, 2, 0, 1];
  const C = K.countTransitions(seq, 3);
  assert.deepEqual(C, [[0, 2, 0], [0, 1, 1], [1, 0, 0]]);
  assert.deepEqual(K.countTransitions(seq, 3, { start: 2, end: 5 }), [[0, 0, 0], [0, 0, 1], [1, 0, 0]]);
  assert.throws(() => K.countTransitions([0, 5], 3));
});

test('kernelFromStream recovers a known kernel from a long sample', () => {
  const rng = mulberry32(11);
  const P = randomKernel(4, rng);
  const seq = sampleChain(P, 50000, rng);
  assertRowsClose(P, K.kernelFromStream(['a', 'b', 'c', 'd'], seq), seq);
});

test('alpha > 0 keeps unobserved transitions possible and unobserved rows non-absorbing', () => {
  const Khat = K.kernelFromStream(['a', 'b', 'c'], [0, 1, 0, 1, 0]);
  assert.ok(Khat.matrix[2][2] < 1);
  assert.ok(Khat.matrix[0][2] > 0);
});

test('joint encoding matches the row order of markov.kron', () => {
  const sizes = [2, 3];
  const labels = K.jointStates([['a', 'b'], ['x', 'y', 'z']]);
  assert.deepEqual(labels, ['a|x', 'a|y', 'a|z', 'b|x', 'b|y', 'b|z']);
  const code = K.encodeJoint([[1, 0], [2, 1]], sizes);
  assert.deepEqual(code, [5, 1]);
  assert.deepEqual(K.decodeJoint(5, sizes), [1, 2]);
  // Independent chains: the estimated joint kernel approaches kron of the parts.
  const rng = mulberry32(3);
  const A = randomKernel(2, rng), B = randomKernel(3, rng);
  const sa = sampleChain(A, 60000, rng), sb = sampleChain(B, 60000, rng);
  const joint = K.encodeJoint([sa, sb], sizes);
  assertRowsClose(markov.kron(A, B), K.kernelFromStream(labels, joint), joint);
});

test('rollingWindows and rollingKernels tile the stream', () => {
  const w = [...K.rollingWindows(10, { size: 4, step: 3 })];
  assert.deepEqual(w, [{ start: 0, end: 4 }, { start: 3, end: 7 }, { start: 6, end: 10 }]);
  const seq = sampleChain(randomKernel(3, mulberry32(1)), 100, mulberry32(2));
  const rk = K.rollingKernels(['a', 'b', 'c'], seq, { size: 20 });
  assert.equal(rk.length, 5);
  for (const { counts } of rk) assert.equal(counts.flat().reduce((a, b) => a + b, 0), 19);
});
