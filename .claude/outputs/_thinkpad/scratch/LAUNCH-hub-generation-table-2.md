# ADDENDUM — `hub/generation-table-2` ⭐ THE HEADLINE
**Read `LAUNCH-COMMON.md` first** (it carries the gate ruling, the budget, R1–R5). Then
`.claude/tasks/hub-generation-table.md` is your task, as amended by R1 and §7.1 below.

- **Slug:** `hub-generation-table` · **Branch:** `hub/generation-table-2` · already created at `945a3e9`
- **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-generation-table-2` · already created
- **Threads: 4.** Sub-agents: ≤ 2, inside those 4, **every one on `model: "opus"`**. You run on Opus.

## ✅ §5.1 falsifier — ALREADY DONE by your stopped predecessor. CLEAR. Do not re-run it.
Verdict at `.claude/outputs/hub-generation-table/REENCODE-VERDICT-iter2.md` (see COMMON §-0). Read it;
cite it; move on. ⚠ Your stopped predecessor also left `iter2/dev/{arm2,ctx,dev,exp30_levers,fast}.py`,
`iter2/hubenv.sh`, `LOG-iter2.md`, and an untracked `chlu/core/gen_dict_fit.py` in your worktree.
Inherited; read freely; adopt only by a dated bullet. Start a new `START_UTC.txt` and a `RELAUNCH` log section.

## (superseded — kept for the record) DO THIS FIRST, BEFORE ANYTHING ELSE
**§5.1 falsifier — `reencode_magnitude.json`, ≈ 2 seconds.** The ruling that lets you score rests on the
51–81 ULP divergence being *near-zero post-ReLU dust*, where `np.spacing` has collapsed to the float16
subnormal gap and ULP counting is meaningless. **Measure it, do not assume it.**

Load seed 0's cache, take `crows[:64]`, encode exactly as `_assert_cache_reencodes` does, and for every
differing element record `(|cached|, delta, ulp_delta)`. Write
`$CHLU_HUB_OUTPUTS/hub-generation-table/reencode_magnitude.json` with: the `|cached|` of the argmax-ULP
element; count and max ULP of differing elements with `|cached| ≥ 2^-4`; the same for `< 2^-4`;
`max_abs_delta`.

⛔⛔ **If ANY element with `|cached| ≥ 2^-4` shows `ulp_delta > 2`: ALL FOUR HUBS STOP SCORED WORK AND
REPORT.** The encoder arithmetic on this box genuinely differs and even a no-extension run is suspect.
Say so loudly and immediately — the other three are relying on you for this.
(Expected: the divergence sits entirely on `|cached| < 2^-4`. The pathology is already documented at
`_hub_scratch/reports/I-numerical-divergence.md:212`.)

## Your target
**WIN or TIE (R1) vs `knn_proto` on composition-B aggregate**, matched bytes, external classifier,
shipped cell, **seed 0 this iteration**. ⭐ **On a tie: bank it, then keep pushing for a win.**

## Your starting position — you are already at a tie, so bank it and climb
Predecessor DEV numbers (49 corners): arm **0.8135 / 49.9 tuples** vs `knn_proto` **0.8307 / 51.8** —
Δ = **−0.0172**, CI **[−0.0347, +0.0003]**, 16/49. ⭐ Under R1 that is **already a tie** (CI ∋ 0 *and*
|Δ| ≤ 0.020). ⛔ Do not re-derive it. Bank it, then push.

## Levers the predecessor measured but did not exhaust
- **Partition K** — 5 → 64 lifted round-trip 6 → 47. ⚠ §C.2 **fails at K ≳ 128** (a partition gives K free
  scalars) — **stay below**. A configuration where `store_removed` does not collapse is disqualified
  *before* it is scored, never after.
- **Head rank**; **pair-term selection** — `hue × scale` and `shape × orientation` carried the 0.9734 bound.
- **Medoid-vs-learned margin** — +0.085 at K = 32.
- ⛔ **Not** the PCA basis (R4). ⛔ **Not** the argmax read.

## Your one scored run
After §5.1 clears and your PREREG names object, bar and controls: **the scored cell, seed 0, all arms.**
⭐ **Its `clu` row IS the parity measurement** (COMMON §1c / directive §4) — compare your Windows seed-0
composition-B aggregate and distinct-tuple count against the Mac-published seed-0 `clu` row in
`gen-read-cell-arm/runs/cell/seed_0/…/cell_manifest.json` (the 3-seed row is **0.5949, 9.00 ± 0.82
tuples**). Bound: aggregate within the published 3-seed sd, distinct tuples within ±1.
⭐ **Log it as a labelled ledger row, not as a spend.** Parity costs you nothing extra.

## What you report
- The table **with the 0.5278 constant-output floor drawn on it**.
- ⭐ **The distinct-tuple count beside EVERY aggregate you quote. Every one.**
- The byte-identical **linear-vs-argmax** row on the shipped K = 5 store (**+0.119, 6/6**) as a standalone
  finding — it costs the programme nothing to try and it is your clearest recommendation.
- Per-render tuples persisted **for every arm** (composition-2 reads them instead of spending its own run).
- Every row carries: the §C.1 **medoid** control (report **K** with it — the pass exists at K = 32 and
  vanishes at K = 5), §C.2 `store_removed`, §C.4 distance-to-nearest-atom over atom spread, and `n`.
