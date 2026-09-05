# PREREG — gated-write-performance (w22)

Written **before** any full-length run of the gated-write comparison or the edge search.
Commits the win/tie/lose call per family and per edge-candidate, and how each was derived.
Base: local `main` @ `8519df6`. Budget used for the "full" numbers is pre-registered here:
`train_steps=1200`, `tune_steps=400`, 3 seeds (`42,1042,2042`), LR grid `{3e-4,1e-3,3e-3}`,
symmetric monotone LR-rescue for **every** variant — i.e. the exact published-numbers budget
of `primitive-harness`/`gamma-read-sweep`, so my re-run baselines double as a reproduction check.

## The framing I commit to (fairness, Item 2)
The gate `p += (W_in x)⊙σ(W_gate x)` **imports a capability every baseline already has**
(GRU gates, selective-SSM input-dependent Δ, softmax QK). Adding it to CLU is **levelling,
not beating.** Item 1 can therefore only answer *"is gated CLU now competitive"*. A gated CLU
that merely ties the GRU is **not** a program result. The gate is **category-(b)-adjacent**:
it is a CLU-internal knob (lives in `CLUBlock`, changes no shared slot), but the *capability*
it adds is one the baselines have — so I argue it explicitly rather than calling it category (a).

## Item 1 — gated CLU vs baselines, per family (win = beats best baseline)

| family | γ (CLU) | H1: linear CLU | H2: gated CLU absolute | my call for gated CLU vs best baseline |
|---|---|---|---|---|
| adding T=128 (MSE↓) | 0.05 | at floor ≈0.182 | gets OFF floor to **0.003–0.03** | **LOSE** (GRU/SSM≈0.001, attn≈0.0001). Gate levels; does not win. |
| parity T=64 (acc↑) | 0.05 | chance ≈0.53 | **uncertain**: predict 0.53–0.75, likely still < GRU 1.0 | **LOSE** (GRU=1.000). Gate supplies conjunction, not XOR state-tracking — I do **not** expect parity solved. |
| MQAR T=128 kv4 (acc↑) | 0.0 | 0.386 | gate helps binding → **0.45–0.70** | **LOSE to attention (0.99); TIE/near GRU (0.486)**. |

Derivations. Adding: `gamma-read-sweep §5` measured gated adding MSE 0.0028 (γ=0.02) exploratory,
4× worse than GRU — I expect that to reproduce as a loss on absolute error. Parity: the gate adds a
*multiplicative write*, which the adding/MQAR conjunction needs; parity needs a running XOR
(state that flips), which no write-gate supplies — so I predict little/no parity gain. MQAR: the gate
gives input-conditioned binding; γ=0 is +0.040 free; but a fixed 2·d_clu carry against attention's
O(T) KV cache caps it well below attention.

**Global Item-1 call: gated CLU wins 0 of 3 on absolute performance. It gets off the floor on
adding and improves MQAR, but the levelling caveat holds — no family beaten.**

## Item 3 — WHERE IS THE EDGE? (CLU physics vs a matched, equally-gated GRU/SSM)

| edge | prediction for CLU vs matched gated GRU/SSM | derivation |
|---|---|---|
| **3a long-horizon extrapolation** (train T, test 2T/4T) | **CANDIDATE EDGE — TIE-to-WIN on the *relative* drop.** CLU degrades *less* from T→4T than GRU/SSM on adding; absolute may still trail. | founding CHLU claim (Exp A, 100× stable extrapolation); symplectic/energy structure should hold horizon. But budget-limited, so I hedge to tie-to-win. |
| **3b capacity under item load** (kv 2→16, gate+γ0) | ⭐ **STRONGEST EDGE — WIN vs GRU at kv≥8.** Crossover survives the gate and moves *toward* CLU's favour (lower kv) with γ=0. | `primitive-harness §1b` measured CLU>GRU at kv≥8 **without** the gate and at γ=0.05; γ=0 is +0.040 and the gate lifts the whole curve. This is the single existing CLU-favourable signal. |
| **3c robustness** (input noise at inference) | **WEAK EDGE / TIE.** CLU *may* degrade more gently under input noise (barrier confinement, Prop 2) but the gate does not add robustness, so I do not expect a strong separation. | Prop-2 confinement is a designed-landscape property (`primitive-harness` reconciliation #1 warns it may not transfer to a trained block). Hedge to tie. |

**Headline call: the capacity edge (3b) is the one I expect to survive matched gating + matched
tuning. If it does, it is the wave's headline; if none survives, gated CLU is a competent-but-
undistinguished recurrence — itself a decision-relevant finding.**

## What would falsify each
- Item 1 adding LOSE falsified if gated CLU ≤ 0.001 (matches GRU) — would be a genuine win, unexpected.
- 3b WIN falsified if gated GRU stays ≥ gated CLU at every kv (crossover erased by the gate).
- 3a EDGE falsified if CLU's relative T→4T drop ≥ GRU/SSM's.
- 3c EDGE falsified if CLU's MSE-vs-noise slope ≥ the gated GRU's.
