// Ground-truth tests: each metric must react to a planted effect and stay
// flat (within its null spread) without it.
const test = require('node:test');
const assert = require('node:assert/strict');
const { mulberry32 } = require('conscious-agent');
const I = require('../src/info');
const { countTransitions, kernelFromStream } = require('../src/kernels');
const { randomKernel, sampleChain, mixKernels, herdedStreams, sampleIndex } = require('../sim/chains');

const S4 = ['a', 'b', 'c', 'd'];
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const max = (xs) => Math.max(...xs);
const min = (xs) => Math.min(...xs);

// A clean base kernel, a planted kernel that differs from it, and an
// estimated base from a long clean run (what a deployment would have).
function setup(seed, eps = 0.4) {
  const rng = mulberry32(seed);
  const P = randomKernel(4, rng);
  const Q = mixKernels(P, randomKernel(4, rng, { concentration: 0.3 }), eps);
  const base = kernelFromStream(S4, sampleChain(P, 50000, rng)).matrix;
  return { rng, P, Q, base };
}

test('kl: zero on equal distributions, known value, infinite on support mismatch', () => {
  assert.equal(I.kl([0.5, 0.5], [0.5, 0.5]), 0);
  assert.ok(Math.abs(I.kl([1, 0], [0.5, 0.5]) - Math.LN2) < 1e-12);
  assert.equal(I.kl([0.5, 0.5], [1, 0]), Infinity);
});

test('kernelKL and surprise separate planted windows from clean windows', () => {
  const { rng, P, Q, base } = setup(21);
  const W = 400;
  const clean = Array.from({ length: 30 }, () => sampleChain(P, W, rng));
  const planted = Array.from({ length: 30 }, () => sampleChain(Q, W, rng));
  const klOf = (s) => I.kernelKL(countTransitions(s, 4), base);
  const surOf = (s) => I.surprise(s, base);
  assert.ok(min(planted.map(klOf)) > max(clean.map(klOf)), 'KL');
  assert.ok(mean(planted.map(surOf)) > mean(clean.map(surOf)) + 0.05, 'surprise');
  // G-statistic identity: 2 N D equals the likelihood-ratio statistic.
  const s = clean[0], C = countTransitions(s, 4);
  const mle = C.map(r => { const n = r.reduce((a, b) => a + b, 0); return r.map(c => (n ? c / n : 0)); });
  let G = 0;
  C.forEach((r, i) => r.forEach((c, j) => { if (c) G += 2 * c * Math.log(mle[i][j] / base[i][j]); }));
  assert.ok(Math.abs(G - 2 * (W - 1) * I.kernelKL(C, base)) < 1e-9);
});

test('chiSquareSurvival matches reference values', () => {
  assert.ok(Math.abs(I.chiSquareSurvival(3.841458820694124, 1) - 0.05) < 1e-9);
  assert.ok(Math.abs(I.chiSquareSurvival(18.307038053275146, 10) - 0.05) < 1e-9);
  assert.ok(Math.abs(I.chiSquareSurvival(2, 2) - Math.exp(-1)) < 1e-12);
  assert.ok(Math.abs(I.chiSquareSurvival(124.34211340400407, 100) - 0.05) < 1e-8);
});

test('chiSquare: calibrated p-values under the null, tiny p-values under a planted kernel', () => {
  const { rng, P, Q } = setup(22);
  // Test against the true P so the null is exact.
  const nullP = Array.from({ length: 400 }, () => I.chiSquare(countTransitions(sampleChain(P, 500, rng), 4), P).pValue);
  const rate = nullP.filter(p => p < 0.05).length / nullP.length;
  assert.ok(rate > 0.02 && rate < 0.09, `null rejection rate ${rate}`);
  const planted = Array.from({ length: 20 }, () => I.chiSquare(countTransitions(sampleChain(Q, 500, rng), 4), P).pValue);
  assert.ok(max(planted) < 1e-4, `planted max p ${max(planted)}`);
});

test('benjaminiHochberg matches a hand-worked example', () => {
  const p = [0.01, 0.04, 0.03, 0.005, 0.5];
  // sorted 0.005 0.01 0.03 0.04 0.5 vs (k/5)·0.05 = 0.01 0.02 0.03 0.04 0.05 → k = 4
  assert.deepEqual(I.benjaminiHochberg(p, 0.05).rejected, [true, true, true, true, false]);
  assert.deepEqual(I.benjaminiHochberg([0.2, 0.3], 0.05).rejected, [false, false]);
});

test('mutualInformation: ~0 when independent, positive when coupled, ln 2 for a perfect binary copy', () => {
  assert.ok(Math.abs(I.mutualInformation([[50, 0], [0, 50]]) - Math.LN2) < 1e-12);
  const rng = mulberry32(5);
  const draw = (couple) => {
    const xs = [], ys = [];
    for (let t = 0; t < 5000; t++) {
      const x = sampleIndex([0.3, 0.3, 0.4], rng);
      xs.push(x);
      ys.push(rng() < couple ? x : sampleIndex([0.3, 0.3, 0.4], rng));
    }
    return I.mutualInformation(I.contingency(xs, ys, 3, 3));
  };
  const indep = draw(0), coupled = draw(0.3);
  assert.ok(indep < 0.003, `independent MI ${indep}`);
  assert.ok(coupled > 0.05, `coupled MI ${coupled}`);
});

test('fusionRate: flat across independent replicates, rises with herding', () => {
  const S = ['x', 'y', 'z'];
  const rng = mulberry32(31);
  const base = randomKernel(3, rng);
  const run = (herd) => I.fusionRate(herdedStreams({ kernels: base, T: 3000, herd, rng }), [S, S]);
  const nulls = Array.from({ length: 20 }, () => run(0));
  const planted = Array.from({ length: 20 }, () => run(0.3));
  const sd = Math.sqrt(mean(nulls.map(v => (v - mean(nulls)) ** 2)));
  assert.ok(max(nulls) < min(planted), `null max ${max(nulls)} planted min ${min(planted)}`);
  assert.ok(sd < 0.02 && Math.abs(mean(nulls)) < 0.05, `null mean ${mean(nulls)} sd ${sd}`);
  assert.ok(mean(planted) - mean(nulls) > 5 * sd);
});

test('pairwiseFusion localises the herding pair', () => {
  const S = ['x', 'y', 'z'];
  const rng = mulberry32(32);
  const base = randomKernel(3, rng);
  const [a, b] = herdedStreams({ kernels: base, T: 10000, herd: 0.6, rng });
  const c = sampleChain(base, 10000, rng);
  const { matrix } = I.pairwiseFusion([a, b, c], [S, S, S]);
  assert.ok(matrix[0][1] > 0.1);
  assert.ok(Math.abs(matrix[0][2]) < 0.01 && Math.abs(matrix[1][2]) < 0.01);
});

test('bootstrap: seeded, ~95% CI coverage, block mode runs', () => {
  const rng = mulberry32(9);
  const stat = (xs) => mean(xs);
  let covered = 0;
  for (let rep = 0; rep < 100; rep++) {
    const d = Array.from({ length: 200 }, () => rng());
    const { ci } = I.bootstrap(d, stat, { resamples: 300, rng });
    if (ci[0] < 0.5 && ci[1] > 0.5) covered++;
  }
  assert.ok(covered >= 85 && covered <= 100, `coverage ${covered}/100`);
  const data = Array.from({ length: 500 }, () => rng());
  const a = I.bootstrap(data, stat, { resamples: 500, rng: mulberry32(1) });
  const b = I.bootstrap(data, stat, { resamples: 500, rng: mulberry32(1) });
  assert.deepEqual(a.ci, b.ci);
  assert.ok(Math.abs(a.sd - Math.sqrt(1 / 12 / 500)) < 0.003);
  const blk = I.bootstrap(data, stat, { resamples: 200, blockSize: 25 });
  assert.ok(blk.ci[0] < blk.estimate && blk.estimate < blk.ci[1]);
});

test('quantile uses linear interpolation', () => {
  assert.equal(I.quantile([1, 2, 3, 4], 0.5), 2.5);
  assert.equal(I.quantile([3, 1, 2], 0), 1);
  assert.equal(I.quantile([3, 1, 2], 1), 3);
});
