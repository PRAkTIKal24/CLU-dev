# r2-d-sweep-close — the d-sweep at m=4, the arm(a)×arm(b) cell, and R2's close

**Agent:** experiment-engineer. **Worktree MANDATORY** (§3.2). Base local `main @ 082d095` (post-w26).
**Campaign tag: [C1W27].** ⭐ **This is the wave headline and C1's last R2 task.** Head ruling
2026-07-29, decision-queue item 2: *"d-sweep at m=4 (d=6, d=8) — YES, the wave headline; combine
with the arm(a)×arm(b) cell to close R2."* ⚠ **Heaviest compute of the wave — read §Compute before
planning.** Direct follow-up to `.claude/outputs/r2-excursion-reach.md` (read it first, plus its
`PREREG.md`) and `.claude/outputs/readout-channel-theory.md` §4.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** capacity (the R2 law). A law about the primitive — exempt from the masked-recall
  demotion, and **its figure is never framed as beating anything** (CM-23(m)).
- **Laundering control:** the **designed** write at matched geometry must keep reaching its own wall
  `K_designed(d) = 4·2^d` at **every** `d` you run, on the **same payload format** as the learned
  arm. Value-blank control on every reported PASS. If the learned arm only "works" by becoming more
  designed → **N46 scope collapse**, not a win.
- **Falsifies the claim:** at d=6 and/or d=8, the m=4 multi-channel code (± the annealed read) does
  **not** move the wall above the w23-class ceiling at **≥3 seeds with payload read-noise ON** ⇒ the
  w26 unclamping is a **d=4 artefact** and R2 closes with the exponent unchanged. Report that
  outcome as the result; it is a clean, publishable close.
- **Does NOT falsify:** failing to reach exactly `4·2^d` (the prefactor gap is expected and known);
  any comparison to kNN or any external method; losing on a metric-native protocol (standing
  metric-native-ceiling theorem).

## ⭐ THE FIVE BINDING FAIRNESS CONDITIONS (Head ruling B1.3 — carried verbatim, no exceptions)
A violated condition invalidates the arm.
1. **Bits-per-item held constant** across every arm, every `d`, and the baseline.
2. **Byte accounting pinned explicitly** — no capacity smuggled via extra parameters. Publish it.
3. **Payload read-noise ON.** This is the crux and it is what killed w25's `pscale` result
   (0.9995 → 0.5894 under noise) while the constant-Δ code survived. **Never quote `K_learned` at
   `pscale ≠ 1` without it.**
4. **Baselines given the same format** (the designed arm reads the same payload code).
5. **The laundering control travels** with every number.

## Stage 1 — `pass_metric` default → `decode` (do this FIRST; it is a correctness fix)
Head ruling, queue item 3: **YES — a correctness fix, not a lever promotion.** `tol` is **vacuous at
m>1** (a blank payload scores 1.0000). Change `pass_metric` default `"tol"` → `"decode"` in
`chlu/config.py:1604`; keep `"tol"` selectable for backwards reproduction of pre-w27 runs.
- **Add a regression test** that a value-blank store does **not** pass at `m>1` under `decode`.
- **Re-render the w26 d=4 table under `decode`** so the new sweep and the shipped table sit on one
  metric, and **list every w26 number that changes** in your report's first 10 lines. If none change,
  say so explicitly — that is the reconciliation the Hub will check.

## Stage 2 — the d-sweep at m=4 (the headline)
Measure `K_learned(d)` under the m=4 multi-channel payload code at **d=6** and **d=8**, against
`K_designed(d) = 4·2^d` on the same format, and report **the tax at each d** (w26 at d=4: 1/8 → 1/1).
- **N92 budget-adequacy at every first-fail cell** — re-check at 2× atoms; a cell that passes at 2×
  is *budget-limited (a lower bound)*, **never** a wall.
- **≥3 seeds on every decisive cell.** Quote sd and its convention (w26's r2 tables quote
  **population** sd; sample sd is ~20 % larger — match the convention and say which).
- If the tax closes at d=6/d=8: state the exponent you can and cannot re-measure. ⛔ The base-√2 /
  `d^1.62` exponent stays on the do-not-quote list until you have ≥3 d-points on one metric.

## Stage 3 — the arm(a)×arm(b) combination cell
Run the annealed/continuation read **together with** m=4 at the decisive cells. w26's Stage-A
factorial found the two init/width levers were **substitutes** (interaction **+0.459**, 14.3 SE) and
refuted the engineer's own additivity prediction. **Register your prediction for the (a)×(b)
interaction sign before running** and score it. Report the interaction term explicitly, not just the
two main effects.

## Stage 4 — capacity-per-byte, only if the wall moves at d ≥ 6
Addendum-2 §B2 **Candidate 3** is now the strongest of the three contested-win routes *because the
ceiling moved*. ⚠ **w21's bits-per-param (~1.3 vs 2) is STALE** — it was measured *under* the reach
ceiling and nobody has re-derived it. If and only if Stage 2 moves the wall at d≥6, re-measure
**items-per-parameter and bits-per-parameter** with the byte accounting from condition 2 pinned.
This is a measurement, not a claim: **do not** frame it as beating anything.

## ⛔ Defaults
**No default other than `pass_metric` changes in this task** (B1.4 precedent). `atom_init_local`,
`atom_init_width`, the anneal schedule and `m` stay at their shipped values unless you are running a
declared arm. You are measuring, not promoting.

## Compute (declare, do not silently drop)
w25 measured ~1340 s per write per seed at d=8 / 16384 atoms. This wave runs **at most 2 engineer
worktrees**; the w26 thermal incident (load 575 on 8 cores) is the reason. **Cap yourself at ≤4
concurrent background jobs.** Priority order — declare it in your PREREG and follow it:
**Stage 1 → Stage 2 d=6 → Stage 3 at d=4 and d=6 → Stage 2 d=8 → Stage 4.**
Any dimension you cannot reach is reported as **NOT RUN**, never as a null.
Background jobs showing `PPID=1` are **harness-detached, not orphaned** — do not kill them.

## File ownership (standing practice — w26's split produced zero conflicts across four branches)
**You own:** `chlu/experiments/exp_designed_mechanism.py` and the
`ExperimentDesignedMechanismConfig` block in `chlu/config.py` (incl. `pass_metric`).
⛔ **Do NOT touch** `chlu/core/memory_potentials.py` (this is a no-new-core-code task; the
`AtomStorePotential` class and the store/controller config fields are owned by
`deletion-waitlist-stiffness` this wave), `chlu/core/placement.py`, `chlu/core/controller.py`,
`exp_cl_entry.py`, `cl_baselines.py`, `exp_phi_stream.py`. If you believe a core change is
genuinely required, **stop and report it** rather than editing.

## Deliverable
PREREG first (`.claude/outputs/r2-d-sweep-close/PREREG.md`) — the five fairness conditions restated
as checkable items, your registered prediction for the wall at d=6 and d=8 and for the (a)×(b)
interaction sign, and the compute priority order. Report at `.claude/outputs/r2-d-sweep-close.md`,
standard format, PREREG scorecard, **reconciliation list in the first 10 lines** (incl. every w26
number moved by the `decode` default). Full `pytest tests/` green, `ruff` clean, atomic commits on
`agent/experiment-engineer/r2-d-sweep-close`. **Do not push.**

⛔ **Do-not-quote, carried:** `K_learned` at `pscale ≠ 1` without the payload-noise condition · any
`tol`-metric number at m>1 · "the write operator is the ceiling" · width-lock-as-cause · the base-√2
/ `d^1.62` exponent · "~32, d-independent" as settled · "capacity multiplies by sharding" · "24.5×
fewer floats" (now 19.1×) · **quote the curve, not the endpoint.**
