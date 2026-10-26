// Information-theoretic and test statistics that conscious-agent does not ship.
// All entropies and divergences are in nats.

const { markov, mulberry32 } = require('conscious-agent');
const { kernelFromStream, encodeJoint, jointStates } = require('./kernels');

// ---------------------------------------------------------------------------
// Divergences and mutual information

// KL(p || q) in nats. Infinity when q puts zero mass where p does not.
function kl(p, q) {
  let d = 0;
  for (let i = 0; i < p.length; i++) {
    if (!(p[i] > 0)) continue;
    if (!(q[i] > 0)) return Infinity;
    d += p[i] * Math.log(p[i] / q[i]);
  }
  return Math.max(0, d);
}

// Row-weighted KL of a window's empirical kernel from a base kernel:
//   D = Σ_i (n_i / N) KL(P̂_i || B_i),  P̂_i = counts[i] / n_i (unsmoothed MLE).
// It is the per-transition log-likelihood ratio of "window kernel" against
// "base kernel", and 2·N·D is the G-test statistic for the same counts.
function kernelKL(counts, base) {
  let N = 0, d = 0;
  for (let i = 0; i < counts.length; i++) {
    const row = counts[i];
    const ni = markov.rowSum(row);
    if (ni === 0) continue;
    N += ni;
    d += ni * kl(Array.from(row, c => c / ni), base[i]);
  }
  return N > 0 ? d / N : 0;
}

// Contingency table counts[x][y] from paired samples.
function contingency(xs, ys, nx, ny) {
  const C = Array.from({ length: nx }, () => new Array(ny).fill(0));
  for (let t = 0; t < xs.length; t++) C[xs[t]][ys[t]]++;
  return C;
}

// Plug-in mutual information I(X;Y) of a contingency table, in nats. It is
// biased upward by about (|X|-1)(|Y|-1) / (2N); compare against a baseline or
// a permutation null rather than against zero.
function mutualInformation(counts) {
  const nx = counts.length, ny = counts[0].length;
  const px = new Array(nx).fill(0), py = new Array(ny).fill(0);
  let N = 0;
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) { px[x] += counts[x][y]; py[y] += counts[x][y]; N += counts[x][y]; }
  if (N === 0) return 0;
  let I = 0;
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) {
    const c = counts[x][y];
    if (c > 0) I += (c / N) * Math.log((c * N) / (px[x] * py[y]));
  }
  return Math.max(0, I);
}

// Mean negative log-likelihood per transition of seq under kernel `base`
// (nats/step). Infinity if seq uses a transition `base` forbids; estimate the
// base with alpha > 0 to avoid that.
function surprise(seq, base, { start = 0, end = seq.length } = {}) {
  let s = 0, n = 0;
  for (let t = start + 1; t < end; t++) {
    const p = base[seq[t - 1]][seq[t]];
    if (!(p > 0)) return Infinity;
    s -= Math.log(p);
    n++;
  }
  return n > 0 ? s / n : 0;
}

// ---------------------------------------------------------------------------
// Fusion rate (docs/spec.md, Spec 1)

// F = Σ_i h(K_i) − h(K_joint) in nats/step, where h is markov.entropyRate.
// F ≈ 0 for independent streams because h is additive under ⊗; F > 0 measures
// how much the streams predict one another. On a finite window F is biased
// because the joint kernel has many more cells than the parts: the unsmoothed
// plug-in estimate (alpha = 0) biases it up, while Dirichlet smoothing adds
// entropy to sparsely visited joint rows and can bias it down. Either way,
// calibrate it against a baseline of the same window size and alpha.
function fusionRate(streams, stateSets, { alpha } = {}) {
  const sizes = stateSets.map(s => s.length);
  const joint = kernelFromStream(jointStates(stateSets), encodeJoint(streams, sizes), { alpha });
  const parts = streams.map((s, i) => kernelFromStream(stateSets[i], s, { alpha }));
  return parts.reduce((h, K) => h + markov.entropyRate(K.matrix), 0) - markov.entropyRate(joint.matrix);
}

// Pairwise fusion rates, for when the full product space is too large.
// Returns the symmetric matrix and the sum over pairs.
function pairwiseFusion(streams, stateSets, opts) {
  const k = streams.length;
  const F = Array.from({ length: k }, () => new Array(k).fill(0));
  let total = 0;
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) {
    F[i][j] = F[j][i] = fusionRate([streams[i], streams[j]], [stateSets[i], stateSets[j]], opts);
    total += F[i][j];
  }
  return { matrix: F, total };
}

// ---------------------------------------------------------------------------
// Chi-square test on transition counts

function lnGamma(x) {
  // Lanczos approximation (g = 7, n = 9), accurate to ~1e-15 for x > 0.
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// Regularised upper incomplete gamma Q(a, x) (Numerical Recipes 6.2).
function gammaQ(a, x) {
  if (x <= 0) return 1;
  const lnPre = -x + a * Math.log(x) - lnGamma(a);
  if (x < a + 1) {
    let sum = 1 / a, term = sum;
    for (let n = 1; n < 1000; n++) {
      term *= x / (a + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-15) break;
    }
    return Math.max(0, 1 - sum * Math.exp(lnPre));
  }
  const tiny = 1e-300;
  let b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b; if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return Math.min(1, Math.exp(lnPre) * h);
}

// P(χ²_df > x)
function chiSquareSurvival(x, df) {
  if (df <= 0) return 1;
  return gammaQ(df / 2, x / 2);
}

// Pearson chi-square test of observed transition counts against a base
// kernel: E_ij = n_i · base_ij, row by row. In each row, cells with expected
// count below `minExpected` are pooled into one bin (and that bin into the
// smallest regular cell if it is still too small), so sparse rows do not
// produce a meaningless statistic. Rows with n_i = 0 contribute nothing.
function chiSquare(counts, base, { minExpected = 5 } = {}) {
  let stat = 0, df = 0;
  for (let i = 0; i < counts.length; i++) {
    const row = counts[i];
    const ni = markov.rowSum(row);
    if (ni === 0) continue;
    const bins = [];
    let pooled = { o: 0, e: 0 };
    for (let j = 0; j < row.length; j++) {
      const e = ni * base[i][j];
      if (e >= minExpected) bins.push({ o: row[j], e });
      else { pooled.o += row[j]; pooled.e += e; }
    }
    if (pooled.e > 0) {
      if (pooled.e >= minExpected || bins.length === 0) bins.push(pooled);
      else {
        const k = bins.reduce((m, b, idx) => (b.e < bins[m].e ? idx : m), 0);
        bins[k] = { o: bins[k].o + pooled.o, e: bins[k].e + pooled.e };
      }
    }
    if (bins.length < 2) continue;
    for (const { o, e } of bins) stat += ((o - e) * (o - e)) / e;
    df += bins.length - 1;
  }
  return { statistic: stat, df, pValue: chiSquareSurvival(stat, df) };
}

// Benjamini–Hochberg false-discovery-rate control at level q.
function benjaminiHochberg(pValues, q = 0.05) {
  const m = pValues.length;
  const order = [...pValues.keys()].sort((a, b) => pValues[a] - pValues[b]);
  let k = -1;
  order.forEach((idx, rank) => { if (pValues[idx] <= ((rank + 1) / m) * q) k = rank; });
  const threshold = k >= 0 ? pValues[order[k]] : 0;
  return { rejected: pValues.map(p => k >= 0 && p <= threshold), threshold };
}

// ---------------------------------------------------------------------------
// Bootstrap

// Linear-interpolated empirical quantile (type 7, as in R and NumPy).
function quantile(values, q) {
  const s = Float64Array.from(values).sort();
  if (s.length === 0) return NaN;
  const h = (s.length - 1) * q;
  const lo = Math.floor(h), hi = Math.ceil(h);
  return s[lo] + (h - lo) * (s[hi] - s[lo]);
}

// Percentile bootstrap of statistic(sample). With blockSize > 1 it is a
// circular moving-block bootstrap, which keeps short-range dependence intact
// and is the right choice for state sequences (resampling single steps would
// destroy the transitions being measured).
function bootstrap(data, statistic, { resamples = 1000, blockSize = 1, level = 0.95, rng = mulberry32(1) } = {}) {
  const n = data.length;
  if (n === 0) throw new Error('bootstrap: empty data');
  const b = Math.max(1, Math.min(blockSize, n));
  const samples = new Float64Array(resamples);
  const buf = new Array(n);
  for (let r = 0; r < resamples; r++) {
    for (let filled = 0; filled < n;) {
      const s = Math.floor(rng() * n);
      for (let j = 0; j < b && filled < n; j++) buf[filled++] = data[(s + j) % n];
    }
    samples[r] = statistic(buf);
  }
  let mean = 0;
  for (const v of samples) mean += v / resamples;
  let v2 = 0;
  for (const v of samples) v2 += (v - mean) ** 2;
  const a = (1 - level) / 2;
  return {
    estimate: statistic(data),
    mean,
    sd: Math.sqrt(v2 / Math.max(1, resamples - 1)),
    ci: [quantile(samples, a), quantile(samples, 1 - a)],
    samples,
  };
}

module.exports = {
  kl, kernelKL, contingency, mutualInformation, surprise,
  fusionRate, pairwiseFusion,
  lnGamma, gammaQ, chiSquareSurvival, chiSquare, benjaminiHochberg,
  quantile, bootstrap,
};
