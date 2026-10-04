# ADDENDUM — `hub/deletion-2` ⭐ nothing has run on this target yet — the cheapest genuine result on the board
**Read `LAUNCH-COMMON.md` first** (gate ruling, budget, R1–R5).

- **Slug:** `hub-deletion` · **Branch:** `hub/deletion-2` · already created at `945a3e9`
- **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-deletion-2` · already created
- **Threads: 4.** Sub-agents: ≤ 2, inside those 4, **every one on `model: "opus"`**. You run on Opus.

⚠ **Your task file is NOT in bundle v2** — `.claude/tasks/` carries composition, refusal and
generation-table only. The deletion task file is at
`gs://chlu-cloud-forgisprova/handoff/hub-deletion.md` (6,744 B). **Fetch it read-only** into
`$CHLU_HUB_OUTPUTS/hub-deletion/hub-deletion.md` and boot from it:
```bash
gcloud storage cp gs://chlu-cloud-forgisprova/handoff/hub-deletion.md "$CHLU_HUB_OUTPUTS/hub-deletion/"
```
⛔ Do not place it in `.claude/tasks/` — that directory is manifest-pinned.

## ⚠ Relaunch — what your stopped predecessor left (COMMON §-0)
✅ **Smoke-scale determinism PASSED**: `smoke/smoke_A/build.json` and `smoke_B/build.json` from the same
live set (K=5, n_live=2048, 30 steps) hash identical — `labels_sha256 a1c53517…`, `deleted_sha256`
identical. ⭐ Cite it, then **still prove it at your real scale** before anything else; smoke scale is not
the claim. ⚠ The full-scale `det_A` build (`scripts/hub_deletion/build_unit.py --tag det_A`) was killed
mid-run — anything under `runs/` from it is **partial and untrusted; rebuild**. Free provenance from its
log: first training loss `47.49073` matched the Mac manifest to all printed digits (R3 provenance,
⛔ not a parity claim). Untracked in your worktree: `chlu/experiments/exp_gen_deletion.py`,
`scripts/hub_deletion/`. `hub-deletion.md` is already fetched at `$CHLU_HUB_OUTPUTS/hub-deletion/`.
Inherited; read freely; adopt only by a dated bullet. New `START_UTC.txt`, new `RELAUNCH` log section.
⭐ The §5.1 falsifier is CLEAR (generation-table's file).

## Your target (charter)
**Exact structural deletion of one factor value or one atom**, on all four conditions:
unrecoverable from every composition · every other factor preserved · **byte-exact provenance** · the
rivals given the same deletion.

⛔⛔ **The trap your task file names:** *store load-bearing **by construction** is a method-section fact,
never a headline.* ⭐ **"The rival has no delete operator" is worth nothing.** If your result reduces to
that, you have not got a result.

## ⭐ THE MECHANISM — port it, ⛔ do not reinvent it
*Retention You Can Predict, Scope and Delete*, §2.3 / App. F / App. G.8. **Deletion is exact when the
store layout is a canonical function of the live item set and the fixed geometry** (history-independent
placement). So `delete(v)` := remove `v`'s rows from the live set and **recanonicalise (rebuild)**; the
post-delete store is then **byte-identical to a store that never held `v`**.
**Cost: O(n) rebuild per delete** — accepted openly in that paper. ⭐ **Put it in the cost line.**
⚠ Its caveat carries over **verbatim**: *store-level guarantee only; the frozen encoder is a separate
channel; no certified (ε, δ) unlearning is claimed.*

## On the gen store, concretely — in this order
1. ⭐⭐ **Make `train_store` deterministic given the live row set, and PROVE it before anything else.**
   Fixed seed, **fixed batch shape**, no eviction (fixed K × M).
   ⚠ Report I is directly relevant: *same batching ⇒ identical `store_hash`; changed batching ⇒ it moves.*
   **Two independent builds from the same live set must hash identical on this machine.** ⛔ If they do
   not, stop — nothing downstream is meaningful (falsifier §5.5: report the two hashes and the batching,
   ⛔ never loosen "byte-identical" to "close").
2. **`delete_value(v)`** := rebuild from `pool ∖ rows(v)`; assert
   `hash(delete(store, v)) == hash(build(pool ∖ rows(v)))`. ⭐ **That equality IS condition 3.**
3. ⚠ **`delete_atom(j)`** (zeroing the mode's column in dictionary and head) is **NOT canonical** — the
   other modes were fitted with `j` present. ⛔ Report it as a **scoped** operation with its measured
   residue, **never as exact**.

## The four conditions — seed 0, val/DEV corners, no held-out spend until the final
- **Unrecoverable:** external-classifier accuracy on factor `v`'s value drops to chance for every request
  naming it, **and** no rendered tuple contains it.
- **Preserved:** per-factor accuracy on requests **not** naming `v`, within pre-deletion seed spread.
- **Provenance:** the hash equality from step 2.
- **Rivals:** `knn_proto` and atom-mean get row deletion (canonical by construction). `lace` /
  `ebm_compose` have **no exact operator** — retrain-without-`v` if it fits the budget, else **NOT RUN**,
  and either way **a method-section line**.

## ⭐⭐ THE NON-BY-CONSTRUCTION HEADLINE — this is your real result, measure it
**COLLATERAL.** For each deleted value `v`: the **paired** change in per-factor accuracy on compositions
that do **not** contain `v`, **CLU vs `knn_proto`, at matched bytes**, over the 6 factors (one value each:
floor hue, wall hue, object hue, scale, shape, orientation) — **plus rebuild wall-clock per delete.**
⭐ **That comparison is real whichever way it falls.** It is the one number here that cannot be dismissed
as by-construction.

## Pieces already in hand
- `chlu/core/gen_store.delete_store(mode=zero|mean|scramble)` is the **laundering control** (its own
  docstring says so) — ⛔ not your delete operator.
- `gen-store-grouping/runs/shapes3d_recon/seed_*/store_manifest.json` already carries
  `delete_zero_* / delete_scramble_* / delete_mean_*` deltas — **the pre-deletion control is partly in hand.**

## Budget — plan for the rebuild cost
A rebuild ≈ one `train_store`: **546–1258 s per corner at 5 threads** on the predecessor's ward wells.
⭐ **Plan ~20 rebuilds in 12 h, ⛔ not 200.** Time one first and extrapolate before committing.

## Your one scored run
**1 max, seed 0:** the deletion cell on held-out B for the 6 deletions — PREREG'd with **the collateral
bar stated as a number before it runs.**
