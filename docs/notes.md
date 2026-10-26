# Working notes

Progress log and decisions for agents picking this up. Read `AGENT.md` and
`docs/spec.md` first.

## Status (2026-10-24)

**Milestone 1 (Core) — done, not committed.** `npm test` → 27/27 pass; the
suite also passed under 51 alternative seed offsets (seeds shifted by
`k·7919`), so the assertions are not tuned to lucky seeds.

| File | Contents |
| --- | --- |
| `src/kernels.js` | `countTransitions`, `kernelFromCounts`, `kernelFromStream`, `jointStates`, `encodeJoint`/`decodeJoint`, `rollingWindows`, `rollingKernels` |
| `src/info.js` | `kl`, `kernelKL`, `contingency`, `mutualInformation`, `surprise`, `fusionRate`, `pairwiseFusion`, `chiSquare` (+ `chiSquareSurvival`, `gammaQ`, `lnGamma`), `benjaminiHochberg`, `quantile`, `bootstrap` |
| `src/monitor.js` | `RollingBaseline`, `createMonitor`, `monitorSeries`, `alertOnsets`, `evaluateAlerts`, `evaluateWindows` |
| `src/index.js` | re-exports the three modules |
| `sim/chains.js` | `sampleIndex`, `randomKernel` (Dirichlet rows), `sampleChain`, `mixKernels`, `herdedStreams` (herd may be a function of t) |
| `test/` | `claims` (port of `examples/00_claims_check.js`), `kernels`, `info`, `monitor` |

## Decisions and gotchas

- **Test command.** `node --test test/` fails on Node ≥ 21 (a directory
  argument is treated as a file). `npm test` now runs plain `node --test`,
  which works on Node 18+ and finds `test/*.test.js`. `AGENT.md` updated.
- **alpha.** Default Dirichlet pseudo-count is 0.5 (`DEFAULT_ALPHA`). With
  alpha = 0 the library turns unobserved rows into self-loops (spurious
  absorbing states).
- **`kernelKL(counts, base)`** = Σ_i (n_i/N) KL(P̂_i ‖ B_i), using the
  unsmoothed MLE for the window. 2·N·D is the G-test statistic (tested).
- **`chiSquare`** pools cells with expected count < 5 per row; rows with no
  visits contribute nothing. Null rejection rate at 0.05 was checked to be
  calibrated against the true kernel.
- **Fusion-rate bias has either sign.** Plug-in (alpha = 0) biases F up;
  smoothing on sparse joint rows biases it down (≈ −0.02 nats/step for
  independent streams with a rare state, T = 3000). Never compare F with 0;
  compare with a baseline of the same window size and alpha.
- **Monitor baseline exclusion.** Excluding *every* exceedance from the
  baseline truncates its tail and ratchets the threshold down (seen as excess
  false alarms). Now only runs of length ≥ max(consecutive, 2) are excluded;
  isolated exceedances still enter. Regression test: late false-alarm rate
  on 20 000 stationary samples stays near nominal.
- **Monitor thresholds** are in calibrated units only (z or quantile level);
  defaults are z = 3 and q = 0.99.
- **Bootstrap** uses a circular moving-block resample when `blockSize > 1`.
  Block joins create a few artificial transitions; keep blocks long relative
  to the mixing time.
- **evaluateAlerts** sign convention: lead = event.start − first alert, so a
  positive value means before onset, negative is detection latency.

## Next: Milestone 2 (Spec 2, service traces)

1. `sim/traces.js`: microservice call-graph simulator (services × endpoints ×
   status class), clean traffic as a Markov/n-gram process, injected attack
   scripts (low-and-slow exfiltration, lateral movement, injected call chains)
   whose individual events are drawn from the legitimate vocabulary. Exact
   labels per window.
2. `src/apps/traces.js`: state encoding (last-n tokens or call-graph edges,
   not S_k), `K_base` from clean traffic, per-window surprise, `kernelKL`,
   `chiSquare` + `benjaminiHochberg`, mixing-time shift (bootstrap its clean
   variance first), irreversibility shift, `K.trace(window)` partial observer.
3. Baseline: n-gram frequency detector. Report precision / recall / latency
   for both via `evaluateWindows` / `evaluateAlerts`.
4. Ground-truth tests: every detector reacts to the planted attack and stays
   flat on clean traffic.

## Open questions for the owner

- Commit / branch: nothing is committed yet; the repo is on `master` with no
  commits and `main` does not exist locally. Owner said not to commit for now.
