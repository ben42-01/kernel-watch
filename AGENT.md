# AGENT.md: kernel-watch

Instructions for any coding agent (Claude Code, opencode, etc.) working in this repo.

## What we are building
`kernel-watch` is a Node.js toolkit for monitoring the **Markov kernels of
coarse-grained event streams**. It uses Hoffman's conscious-agent formalism through
the `conscious-agent` npm package (v3, zero dependencies). The full technical spec
is in `docs/spec.md`. Read it before you change any metric.

One core library, three demo applications:
1. **Market herding monitor**: fusion rate, stationary entropy and mixing time across order-book streams.
2. **Service-trace anomaly observer**: surprise, KL/chi-square, mixing-time and irreversibility shift on call sequences.
3. **Framing-contextuality auditor**: Bell/CHSH classification with signalling checks for framed A/B or survey data.

## Ground rules
- **Honest claims.** Nothing here shows that anything is conscious, and no detector
  "sees behind the interface". Write README text, comments and output in that spirit.
- **Use the library; don't reimplement it.** Use `FormalConsciousAgent`,
  `MarkovKernel.fromCounts`, `markov.*`, `bell.*` and `trace` from `conscious-agent`.
  Add to `src/` only what it lacks (mutual information, KL, chi-square, fusion rate,
  rolling windows, bootstrap).
- **Every metric gets a synthetic ground-truth test** before it touches real data.
  The test must show the metric reacts to the planted effect and stays flat without it.
- **No hard-coded alert thresholds.** Calibrate each metric against a rolling
  baseline (z-score or quantile) and report lead time, precision and recall against
  a simple baseline.
- Entropy is in **nats** throughout (as in the library). Convert to bits only for display.
- Use seeded randomness everywhere (`mulberry32` from `conscious-agent`) so runs are reproducible.
- Plain CommonJS, Node ≥ 18, `node --test` for tests, no build step. Avoid
  dependencies unless one clearly earns its place (e.g. a websocket client for live feeds).

## Layout (target)
```
src/
  kernels.js       kernelFromStream, rolling windows, joint-state encoding
  info.js          mutualInformation, kl, chiSquare, fusionRate, bootstrap
  monitor.js       baseline calibration + alert stream (z-score / quantile)
  apps/market.js   Spec 1
  apps/traces.js   Spec 2
  apps/framing.js  Spec 3
sim/               synthetic generators with planted effects (herding, attacks, framing bias)
examples/          runnable demos; 00_claims_check.js checks the spec's claims
test/              node --test suites
docs/spec.md       the specification
docs/notes.md      working notes: status, decisions, next steps (read before continuing)
```

## Milestones
1. **Core**: `src/kernels.js`, `src/info.js`, `src/monitor.js` and tests. Port
   `examples/00_claims_check.js` into tests.
2. **Spec 2 first**: it has the cleanest synthetic ground truth. Simulator →
   detector → precision/recall against an n-gram baseline.
3. **Spec 1**: agent-based market simulator with a herding knob, then a replay of
   public crypto L2 data.
4. **Spec 3**: framing auditor on a split-ballot simulator, then on public survey microdata.
5. Write-up: what each metric caught, what it missed, and how it compares with the baselines.

## Commands
```bash
npm install
node examples/00_claims_check.js
npm test    # node --test: runs test/*.test.js
```
