# Task: write-ceiling-break — can any write break the d-independent K≈32 ceiling? (w24)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/write-ceiling-break.md` · **Branch:** `agent/experiment-engineer/write-ceiling-break`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (**§3.2 worktrees mandatory — 3 parallel engineer tasks touch `chlu/` this wave**) · `.claude/outputs/dimension-aware-budget.md` (the result you are extending; its §3 is the ceiling evidence) · `.claude/negative_results.md` **N92** (tier A) · `.claude/claims_matrix.md` v2.2 CM-22(j)/CM-23(h)
- **Status of the law you are attacking:** `K_learned(d) = min(2^d, K_ceiling≈32)` is **pinned and Hub-confirmed**. Geometry is vindicated for d≤5 (capacity doubles per dimension — exactly the designed rate). **This task attacks the ceiling only.**

## Why
This is **the gate on R2** ("the capacity law, unclamped") in the Head's result set. w23 established the ceiling is **not** the terrain (designed writes reach ≥256 under identical numerics), **not** parameter count (d=6 K=64 gets *worse* at 2× atoms: 0.855→0.809), and **not** dimension (d-independent: K=64 unwritable at both d=6 and d=8 despite 4× different geometric room). What remains is the **write operator itself** — one static GLOBAL gradient dig asked to carve K disjoint valleys *jointly*, where the valleys fight over shared atoms. A decisive tell: **write-loss reaches ~0 while retrieval already fails** ⇒ the objective is blind to crowding.

## Item 1 — the locality lever (masked / sequential writes)
Masked writes are measured **bit-local** (local-vs-global corruption advantage **8474× at d=2, 3434× at d=4**). Test whether writing items **one at a time (or in masked groups)** — so no two valleys are dug by the same gradient step — breaks the ceiling. Sweep K past 32 at d ∈ {4, 5, 6, 8}. Report `K_learned(d)` under the masked/sequential write.
⚠ **Fairness category is the whole result here.** If sequential writing needs designed-write assistance (formula-placed centers, hand-set widths), that is **not a learned-write result** — declare exactly what is learned and what is supplied, per N46.

## Item 2 — ⭐ the Head's scale-invariance ablation (closes the numerics hypothesis by measurement)
Rescale well depths / margins / barrier so the **per-item write signal is size-independent** (i.e. the gradient each item contributes does not shrink as K grows). The current evidence *argues* the ceiling is optimization interference rather than quantization/normalization; this ablation **closes it by measurement, not argument**. Run it as its own arm at the frontier cells. If the ceiling moves under rescaling alone, the diagnosis changes from "interference" to "signal dilution" — a different and cheaper fix.

## Item 3 — the objective-side lever (crowding-aware margin)
`write_loss → 0` while retrieval fails is a defect of the objective, not the optimizer. Add a **crowding-aware term** (penalize a written center that lands within `d_safe` of an existing one — reuse the `controller-mvp` spacing test / `admission.admit_site`, which already exists on `main`). Does a write objective that *can see* crowding raise the ceiling?

## Item 4 — the verdict
Report `K_learned(d)` at d ∈ {4,5,6,8} under: baseline global dig (the w23 line) · masked/sequential · scale-invariant · crowding-aware · the best combination. State plainly whether the law becomes `min(2^d, K'_ceiling)` with a **higher** ceiling, or is **unclamped** (capacity tracks `2^d` throughout), or the ceiling **survives all three levers**.

## Acceptance
PREREG **before running** (predict each lever's effect and a falsification bar). Per-point **budget adequacy** per the N92 protocol (2×-atom re-check at every first-fail cell — a stall under an inadequate budget is not a ceiling). **≥3 seeds at the frontier** (w23 found the write is seed-fragile at the 0.9 rung: d=4 K=16 gave 0.876 at 3 seeds vs 0.93–0.98 at 2). Tests green; `ruff` clean on touched files; config knobs registered at **all three sites plus `save_config`** (see ⚠ below).

## ⚠ Standing traps
- **`save_config` is a manual-enumeration trap** — a new config group not added there silently reverts on round-trip. `tests/test_config.py::test_every_group_round_trips_mutated` is the guard. Run it.
- A ceiling that breaks **only** by making the write more designed is a scope collapse (N46), not a win — say so in those words.
- Do not re-quote the base √2 / `d^1.62` exponent (**CM-22(j)**, never-quote).
