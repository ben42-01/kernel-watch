# kernel-watch: Specification (conscious-agent v3, Node.js)

Reworked from `hoffman-agents/docs/hoffman_agent_applications_spec.md`. That draft
had three application ideas, a Python/NumPy blueprint, and some metrics that do not
measure what they claim. This version keeps the ideas, fixes the maths, and targets
the Node package `conscious-agent@3.0.0`.

**House rule:** these are statistical monitors built on Markov kernels. Hoffman's
formalism gives us the vocabulary (world, perception, decision, action, combination,
trace). It is not evidence that markets, networks or electorates are conscious, and
no detector here "sees behind the interface". Each one watches the kernel of a
coarse-grained stream that we can observe.

---

## 0. Core model

A conscious agent is the six-tuple `C = (X, G, P, D, A, N)` acting on a world `W`
(Hoffman & Prakash 2014). In `conscious-agent` this is `FormalConsciousAgent`:

| Symbol | Meaning | conscious-agent |
| --- | --- | --- |
| `W` | world states | `W: [...]` labels |
| `X` | experiences | `X: [...]` labels |
| `G` | actions | `G: [...]` labels |
| `P : W × X → Δ(X)` | perception, one X-kernel per world state | `P: { [w]: matrix X×X }` |
| `D : X → Δ(G)` | decision | `D: matrix X×G` |
| `A : G × W → Δ(W)` | action, one W-kernel per action | `A: { [g]: matrix W×W }` |
| `N` | completed cycles | `agent.N` |

**Correction to the draft.** The draft writes the agent's chain as `M = P·D·A` (a
W×W matrix). That product drops the agent's current experience. The right object is
the joint chain on `X × W`:

```
Q((x,w),(x',w')) = P[w](x,x') · Σ_g D(x',g) · A[g](w,w')
```

Use `agent.jointKernel()` to get it. `agent.diagnostics()` returns ergodicity,
period, the stationary distribution, the experience and world marginals, `lambda2`,
the relaxation time and the mixing-time estimate.

**Combination.** `FormalConsciousAgent.combine(c1, c2)` builds the *independent*
combination (`P1⊗P2`, `D1⊗D2`, actions applied in sequence). This matters for
Spec 1. Two real agents that interact do **not** have a product kernel, and the gap
between the observed joint kernel and the product of its marginals is what we
measure.

### Toolbox used throughout

```js
const { FormalConsciousAgent, MarkovKernel, StochasticMatrix, markov, bell, trace, mulberry32 } = require('conscious-agent');

MarkovKernel.fromCounts({ states, counts, alpha })   // estimate a kernel from a transition-count matrix
K.diagnostics()                                       // stationary, ergodic, entropy (nats), lambda2, mixingTime
K.trace(window)                                       // what an observer who sees only `window` experiences
markov.entropy(p)            // Shannon entropy, nats (divide by Math.LN2 for bits)
markov.entropyRate(P)        // h = Σ π_i H(P_i·); additive under independence
markov.mixingTimeEstimate(P) // spectral estimate (an estimate, not a bound, for non-reversible chains)
markov.irreversibility(P)    // 0 = detailed balance, 1 = fully one-way flux
markov.dobrushin(P)          // contraction coefficient: how much memory survives one step
markov.totalVariation(p, q), markov.kron(A, B), markov.marginalize(joint, n, m, axis)
bell.classify(p)             // 'signalling' | 'local' | 'quantum' | 'post-quantum'
bell.chsh(bell.correlators(p)), bell.signalling(p)
```

Metrics the library does not ship, which we write in `src/` (small, pure functions):
mutual information, KL divergence, the chi-square test, and the fusion rate.

---

## Spec 1: Market herding monitor (was "Flash Crash Detector")

### Idea (kept)
Treat each participant group or venue as an agent whose coarse-grained order-flow
state is its experience. A crash is a phase in which these agents stop acting
independently and fall into one shared regime.

### What was wrong
- **Rank-collapse ratio is meaningless here.** For a Kronecker product,
  `rank(M1 ⊗ … ⊗ Mn) = Π rank(Mi)`. So `ρ_R = Π rank / Σ rank` is fixed by the
  individual kernels and cannot see fusion. Checked in `examples/00_claims_check.js`:
  two random 3×3 kernels give a product of rank 9 and `ρ_R = 1.5`, whatever their
  dynamics. Matrix rank is also numerically fragile on estimated kernels.
- "GBM is spatial" is a mischaracterisation. The real contrast is between
  aggregate-price models and models of interacting participants.
- We cannot see individual traders' kernels. We see public order-book streams.

### Formal mapping
| Concept | Definition |
| --- | --- |
| Agents | k observable streams: venues, instruments or order-size buckets |
| `X_i` | per-stream coarse state from L2 book features, e.g. `{calm, thin, one-sided, sweeping}` |
| `G_i` | `{provide, withdraw, cross}` inferred from net order-flow sign/size |
| Observed joint kernel `K_joint` | `MarkovKernel.fromCounts` on `X_1 × … × X_k`, rolling window |
| Independent reference | `K_1 ⊗ … ⊗ K_k` from the per-stream marginal kernels |

### Metrics
1. **Fusion rate**: `F = Σ_i h(K_i) − h(K_joint)` (nats/step), where `h` is
   `markov.entropyRate`. `F ≈ 0` for independent streams. `F > 0` measures how much
   the streams predict one another. It is a multi-information rate, and it uses the
   additivity of `h` under `⊗`. Checked: `F` is −0.001, 0.223 and 0.612 for herding
   strengths 0, 0.5 and 0.9.
2. **Stationary entropy** `H_π` of `K_joint` (`diagnostics().entropy`). This is the
   draft's metric and it is sound. A collapse into few joint states means herding.
3. **Dobrushin coefficient** and **mixing time** of `K_joint`. Slower mixing
   (λ₂ → 1) is the Markov-chain form of "critical slowing down", an established
   early-warning signal.

### Alerts
Do not hard-code thresholds (the draft's 0.35 and 50%). Calibrate each metric
against its own rolling baseline (z-score or empirical quantile). Report the lead
time before labelled events.

### Data & evaluation
- Synthetic: agent-based market (noise traders + market makers + momentum), with a
  herding parameter for ground truth.
- Real: public crypto L2 websocket feeds (free) and the documented flash crashes in
  them. Baselines to beat: realised volatility, order-book imbalance, VPIN.

---

## Spec 2: Service-trace anomaly observer (was "Zero-Trust Intrusion Observer")

### Idea (kept)
Each microservice is an agent. Its experience is the sequence of calls it sees. An
intrusion changes the kernel of legitimate behaviour, and that shows up in mixing,
entropy and irreversibility even when individual events look normal.

### What was wrong
- "Without reading spoofed log contents" overclaims. Traces are logs. If attackers
  forge the trace stream, the fitted kernel is forged too. The honest claim is
  narrower: kernel-level statistics catch attacks whose *individual* events look
  benign but whose *dynamics* differ (low-and-slow exfiltration, lateral movement,
  injected call chains).
- `X = S_k`, the symmetric group, has k! states, which is unusable above k ≈ 8.
  Use n-gram call windows or call-graph edges as states. Keep "decorated
  permutations" (`conscious-agent` `decorated`) as an optional, research-only state
  encoding.
- This is classical sequence anomaly detection (Forrest et al. 1996 onward). The
  value we add is a clean Markov toolkit and the trace-chain view of partial
  observation, not new detection theory.

### Formal mapping
| Concept | Definition |
| --- | --- |
| `W` | global service-graph state, unobserved |
| `X_s` | for service `s`: last-n call tokens `(caller, endpoint, status-class)` |
| `G_s` | `{route, delay, spawn, deny}` |
| Baseline kernel `K_base` | `fromCounts` on clean traffic, `alpha > 0` for unseen transitions |
| Partial observer | `K.trace(window)`: what a sensor that sees only some services experiences |

### Metrics
1. **Mixing-time shift**: `|τ_obs − τ_base| / τ_base` from `markov.mixingTimeEstimate`.
   This is the draft's metric and it is valid, but the estimate is noisy. Bootstrap
   its variance on clean windows before using it as an alert.
2. **Per-window surprise**: the mean negative log-likelihood of the window under
   `K_base`. This is the cheap primary detector.
3. **Divergence**: KL from `K_base` rows to the window's rows, plus a chi-square
   test on transition counts (the draft's p < 0.01, with multiple-testing control).
4. **Irreversibility shift** (`markov.irreversibility`). Request/response traffic
   is close to reversible, and new one-way flows are suspicious.

### Data & evaluation
- Synthetic: a microservice simulator with injected attack scripts, giving exact
  labels.
- Public: OpenTelemetry demo traces, plus attack datasets with syscall or API
  sequences. Report precision/recall/latency against an n-gram frequency baseline.

---

## Spec 3: Framing-contextuality auditor (was "Information Cascade Observer")

### What was wrong (the core claim fails)
The draft used `S_CHSH > 2.1` as a polarisation alarm. That cannot work:
- Polarisation means strong correlation, and perfect correlation is classical. A
  fully polarised pair gives `S = 2` exactly, class `local` (checked:
  `bell.classify(bell.fromCorrelators([[1,1],[1,1]]))` returns `'local'`).
- Data drawn from one joint distribution can never exceed 2 (Fine's theorem). In
  survey data, apparent violations almost always come from **signalling**: one
  question's framing shifts the answer distribution of the other (question-order and
  framing effects). See Dzhafarov's *Contextuality-by-Default*.
- `W = H_A ⊗ H_B` is a Hilbert space with nothing to back it. Beliefs are not qubits.

### What survives, and is worth building
A tool that asks honestly whether a set of framed A/B measurements is classical.
- Settings `a, a'` and `b, b'` are framings, randomly assigned. Outcomes are ±1
  answers.
- Build the behaviour `p[a][b][x][y]` from counts, then run `bell.classify(p)` and
  `bell.signalling(p)`, with bootstrap confidence intervals.
- Report the class (`signalling` / `local` / `quantum` / `post-quantum`) and the
  size of the signalling. The useful product is a **framing-effect auditor** for
  survey designers and A/B tests: "your two questions are not independent
  measurements".
- **Mutual information** `I(A;B)` with a baseline (the draft's 3σ rule) is valid.
  Keep it as the polarisation metric, alongside entropy of the opinion distribution.

### Data & evaluation
Public survey microdata with randomised question order or wording (split-ballot
experiments). Synthetic check: generate data from a local hidden-variable model plus
a controllable framing bias, then verify that the auditor reports `signalling`
before it reports anything "quantum".

---

## Implementation blueprint (Node)

```js
// Rolling-window kernel estimate from a stream of state indices.
function kernelFromStream(states, seq, alpha = 0.5) {
  const n = states.length;
  const counts = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let t = 1; t < seq.length; t++) counts[seq[t - 1]][seq[t]]++;
  return MarkovKernel.fromCounts({ states, counts, alpha });
}

// Fusion rate of k streams: Σ h(K_i) − h(K_joint), nats/step.
function fusionRate(streams, stateSets) {
  const sizes = stateSets.map(s => s.length);
  const enc = (t) => streams.reduce((code, s, i) => code * sizes[i] + s[t], 0);
  const jointStates = stateSets.reduce((acc, s) => acc.flatMap(a => s.map(b => `${a}|${b}`)), ['']).map(l => l.slice(1));
  const joint = kernelFromStream(jointStates, streams[0].map((_, t) => enc(t)));
  const parts = streams.map((s, i) => kernelFromStream(stateSets[i], s));
  return parts.reduce((h, K) => h + markov.entropyRate(K.matrix), 0) - markov.entropyRate(joint.matrix);
}
```

The joint state space grows as `Π |X_i|`. Keep `k·|X|` small (k ≤ 4, |X| ≤ 4), or
compute pairwise fusion rates and aggregate them.

A worked, runnable version of all the claims above is in
`examples/00_claims_check.js`.
