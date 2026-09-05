# Task: gated-write-performance — can CLU get off the floor and find a real edge as a sequence primitive? (w22)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gated-write-performance.md` · **Branch:** `agent/experiment-engineer/gated-write-performance`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/gamma-read-sweep.md` §5 (the exploratory gated-write arm this promotes) + §2 (the linear-write-current diagnosis) · `.claude/outputs/primitive-harness.md` (the 0/3 result + the drop-in slot + the symmetric-rescue fairness protocol — **reuse verbatim**) · `chlu/core/blocks.py`
- **⭐ This is the direct PERFORMANCE test of the general-primitive claim** (Head, 2026-07-23: *"performance is our main goal… if the property fails but we still get good task-performance that's still a massive win"*). `primitive-harness` scored CLU 0/3; `gamma-read-sweep` found why and showed a fix worth 65× on one family. This task runs that fix properly and asks whether it makes CLU **competitive**, and if so, whether the physics buys an **edge** a plain gate does not.

## Why — the 0/3 has a diagnosed, fixable cause
`gamma-read-sweep` §2: `CLUBlock`'s write current `p += W_in x_t` is **unconditionally linear in the token**, so the state carries `Σv_t` and `Σm_t` but **never their product** — the adding/parity/MQAR tasks all need an input-conditioned conjunction. A multiplicative gate `p += (W_in x_t) ⊙ σ(W_gate x_t)` took adding-MSE **0.1816 (control floor) → 0.0028 (65×)** at matched params, in a 3-cell exploratory arm. **The GRU, selective SSM and attention all already have this ingredient.**

## Item 1 — the pre-registered full run (promote the exploratory arm)
Run `write_mode ∈ {linear, gated}` across **all three families** (adding T=128, parity T=64, MQAR kv=4 T=128), 3 seeds, the **full symmetric monotone LR-rescue** for every primitive (`primitive-harness` §4), matched 40k block params.
**Deliverable: the corrected per-family table — gated CLU vs the shipped baselines (MLP/GRU/SSM/attention), never averaged.** Use γ=0 for MQAR (`gamma-read-sweep` §6: +0.040 free) and state the read mode.

## Item 2 — the honest fairness statement (required)
⚠ The gate **imports a capability every baseline already has**. This is **levelling, not beating** — say so in the report, in those words. The question Item 1 answers is only *"is CLU now competitive"*, not *"does CLU win"*. A gated CLU that merely ties the GRU is **not** a result the program needs.

## Item 3 — ⭐ WHERE IS THE EDGE? (the item that matters most)
Getting off the floor is table-stakes. The Head's bar is a **win**, explained or not. So: with the conjunction fixed, is there any axis where CLU's physics beats a **matched, equally-gated GRU/SSM**? Test the candidates our own results point at:
1. **Long-horizon / extrapolation:** train at T, test at **2T, 4T** (the founding CHLU claim; attention/SSM degrade differently). Does the symplectic/energy structure hold accuracy further?
2. **Capacity under item load:** re-run the `primitive-harness` §1b kv-sweep (the one axis CLU already crossed above the GRU) **with the gate and γ=0** — does the crossover move in CLU's favour, and does it survive the gate? (This is our single strongest existing CLU signal and it was measured *without* the gate.)
3. **Robustness:** input noise / distractor injection at inference — does barrier confinement (Prop 2) show as measured robustness a gated GRU lacks?

**Report each as a performance curve vs a matched gated baseline.** ⚠ **An edge that appears only for CLU and survives matched gating + matched tuning is the wave's headline** — it is the "CLU does something others can't" the program needs. **If no edge survives, report that plainly** — it means gated CLU is a competent-but-undistinguished recurrence, which is itself a decision-relevant finding.

## Item 4 — cost
Report the honest wall-clock and FLOP multiple of gated CLU vs the baselines (`primitive-harness` found 1.41× GRU wall-clock at 0.48× attention FLOPs). Per the Head, cost is stated, not competed on — but an edge that costs 10× must show that in the same table.

## Acceptance
The corrected three-family table with the levelling caveat stated, and the edge search (Item 3) as performance curves vs matched gated baselines, with cost. ≥3 seeds, symmetric rescue, fairness category of every knob stated. Tests green.

⚠ **Pre-register, per family and per edge-candidate, whether you expect CLU to win / tie / lose before running.** ⚠ **The gate is a category-(b)-adjacent change** (it imports a baseline capability) — make that argument explicitly, do not assume it. **Do not tune CLU past the baselines' budget**; the whole value of this task is a trustworthy performance comparison.
