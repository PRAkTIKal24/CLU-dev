# Task C: gamma-read-sweep — is CLU's sequence failure a dissipation artifact? (w21)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gamma-read-sweep.md` · **Branch:** `agent/experiment-engineer/gamma-read-sweep`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/primitive-harness.md` (whose §7.1 flags this as its own sharpest open question) · `.claude/outputs/learned-landscape-write-read.md` §5 (the 2-D γ map) · `.claude/outputs/address-space-dimension-scaling.md` §4
- **⚠ Run this before anything else in the wave. It is cheap and it may invalidate a headline negative.**

## Why — a design choice justified by a finding that was superseded in the same wave
`primitive-harness` fixed **γ = 0.05 per token** as concession #2, justified by w19: *"a readable state must be a fixed point (1.000 at γ=0.02 vs 0.813 at γ=0)."* **`learned-landscape-write-read` §5 then showed that 0.813-at-γ=0 was a SINGLE-PHASE artifact** — with a relaxation phase present, fidelity is *exactly* invariant to γ_read (spread **0.0000** across the whole grid). And `address-space-dimension-scaling` §4 showed **identity retrieval at γ=0 is fine (0.969–1.000) and *improves* with d**; only *value* retrieval needs dissipation.

Meanwhile γ=0.05/token gives a memory half-life of `2ln2/γ ≈ 28 tokens` — and CLU scored **exactly the no-mixing control floor on the adding problem at T=128** (0.182 vs control 0.183) and **chance on parity** (0.538). **The information from the early markers is plausibly gone before the readout.**

⇒ **The hypothesis under test: CLU's 0-of-3 result is partly an artifact of a dissipation setting inherited from a retracted measurement.**

## The two read types (the organizing idea — state which one each cell uses)
| read type | wants | why |
|---|---|---|
| **endpoint** (settled state) | **γ > 0** | must settle to a fixed point for the *value* to be readable |
| **trajectory** (the rollout itself) | **γ ≈ 0** | the oscillation *is* the signal; dissipation erases it |

The shipped harness uses γ>0 **and** reads late — the worst of both. **Both axes must be swept, not just γ.**

## Item 1 — the γ sweep (the decisive measurement)
On `exp-primitive-harness`, sweep **γ ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1}** for the CLU block across **all three families** (adding T=128, parity T=64, MQAR kv=4 T=128). 3 seeds, everything else byte-identical to the shipped harness.

**Pre-register before running:** does the adding-problem MSE show a **sharp transition** as γ falls, and at roughly `γ* ≈ 2ln2/T`? Commit the predicted γ* and the predicted floor. ⚠ **A monotone-but-shallow curve, or no movement at all, is a different and equally publishable result** — it would mean the failure is *not* dissipation and the primitive genuinely cannot integrate, which is the more serious finding.

## Item 2 — trajectory read vs endpoint read
Independently of γ, the block currently emits a **late/settled** state. Add a **trajectory read** (the sequence of intermediate states over the `clu_steps` rollout, linearly projected — the `Prop 11` fiber read) as a second read mode, and cross it with the γ sweep. **Report the 2-D (γ × read-mode) table per family.** Prediction to register: the trajectory read is the one that benefits from γ→0.

## Item 3 — `clu_steps`
The harness ran **`clu_steps=1`** (one Verlet step per token), never varied. Sweep `clu_steps ∈ {1, 2, 4}` at the best γ from Item 1 on the adding problem only, and report the cost multiple alongside. This separates "not enough integration" from "too much dissipation".

## Item 4 — re-run the three families at the best configuration
If Items 1–3 find a better setting, **re-run all three families at it, with the full symmetric LR-rescue pass** (`primitive-harness` §4 — it is monotone and applies to every primitive, so it cannot be a route to a CLU-only win). Report the corrected per-family table **beside** the shipped one.

⚠ **Fairness rule, absolute:** any knob you change for CLU must be either (a) a knob no other primitive has (γ, `clu_steps`, read mode — these are CLU-internal), or (b) swept for **every** primitive. **Do not compare a tuned CLU against untuned baselines.** State explicitly which category each change falls in. The shipped baselines already had 5 full-length-equivalent runs per cell; match that.

## Acceptance
The γ sweep with its pre-registered transition scored, the (γ × read-mode) table, the `clu_steps` sweep with cost, and — **if and only if** a better configuration is found — the corrected three-family table with the fairness category of every change stated. Tests green.

⚠ **The honest outcome may be "0 of 3 stands".** Report it plainly if so; that closes a live hypothesis and is worth as much as a rescue. **Do not tune past the pre-registered grid to find a win** — if you exceed the grid, report the extension as a separate, clearly-labelled exploratory arm.
