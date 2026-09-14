# carried-remeasurements — the three carried items, run properly

**Agent:** results-analyst. **No worktree, no tracked-code edits** (run-only + analysis; flag any
code issue for the engineer). Base local `main` (post-w25). Addendum-2 §B3.7 — *"carried, not
dropped"*.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** compute-adaptive reads (item 1) + lifetimes/isolation (item 2) + capacity (item 3).
  This task **strengthens or retracts existing claims**; it does not open a new one.
- **Laundering control:** each item inherits the control of the claim it tests — item 1 carries
  the kNN-in-φ / feedforward-in-φ floor, item 2 is an attack measurement (the control is the
  paired-placement column, which must stay exactly 0.5000), item 3 carries the monolithic
  laundering line at matched `K_total`.
- **Falsifies:** item 1 — the τ-not-binding observation fails to replicate at ≥3 seeds (then w24's
  "τ over-retries and costs accuracy" stands and the w25 line is retracted). Item 2 — the
  allocator-trace leak is **not** an artefact of near-full occupancy (i.e. it stays ≈1.0 at low
  load), which would make it *worse*, not better, and would raise the stakes on
  `placement-landing`.
- **Does NOT falsify:** item 3 failing at d=8 K=256 (it was pre-registered in w25 as an
  **expected-fail** probe of where geometry takes over — a FAIL is the confirmation).

## Item 1 — multi-seed the "τ is not binding in φ-space" observation ⚠
`cl-entry-build` §Item-3 found that in a crowded φ store **all four retry thresholds
(0.99/0.999/0.9999/1.0) select the same eligible pool and give identical curves** — what produces
the anytime curve is the ranking + the 10%-per-round budget + the lock, not the threshold. ⚠ This
is **one seed**, and it is a **behavioural difference from w24**, where τ=1.0 demonstrably
over-retried and cost accuracy. The gate's threshold clause is currently quoted in two places and
one of them must be wrong.

**Run it at ≥3 seeds** (existing CLI, no code changes: `chlu exp-cl-entry --items retry
--baselines none`, hardest corruption level, `--seed` varied). Report whether the identity holds,
and if it does, state precisely which regime each finding belongs to (φ-space vs pixel-space)
rather than declaring one of them wrong.

## Item 2 — the occupancy sweep on the allocator trace ⭐
`mia-decay-measurement` §2 measured the post-eviction membership oracle at **AUC 0.99985** — but
at **7/8 occupancy**, near-full, which the analyst flagged as *the most favourable case for a
"hole" statistic* and explicitly said should be re-measured before the number is quoted generally.

**Repeat §2's history column at load factors 2/8, 4/8, 6/8, 8/8.** This scopes the leak — and it
also scopes `placement-landing`'s acceptance test, since that task is trying to drive this number
to 0.5. Report the paired-placement column at every load as the sanity check (it must be exactly
0.5000; a deviation is a harness bug, not a finding).

⚠ **File discipline:** `placement-landing` depends on
`.claude/outputs/mia-decay-measurement/mia_harness.py` as its acceptance test **this same wave**.
**Copy it into `.claude/scratch/carried-remeasurements/` and work on your copy. Do not edit the
original.** Note the copy's provenance (source path + git base) in your report.

## Item 3 — the high-load sharding probe (compute-permitting)
`lattice-sharded-store` registered **P4: d=8 K=256, 8×32 → expected FAIL at 0.55–0.85,
geometry-bound** and did not run it (declared, §8.3). Its absence removes a confirmation, not a
claim. Run it if compute allows: `chlu exp-sharded-store`, existing CLI. Context you must respect:
the w23 atom floor `512·√2^d` is **per-store, not per-item** (N107), so a parameter-matched 8-way
split is automatically ~8× starved — **report this cell as budget-confounded unless you can afford
the adequate budget**, exactly as w25 reported the 4×16 collapse. Registry-router (RG) route
accuracy there is expected at 1.000 and union separation 0.714.

⛔ **Never route on post-settle energy (N97) or settling displacement (N104-proposed) — both are
broken; only pre-settle energy (R2) and the classical registry (RG) are admissible, and RG is
declared classical indexing (N89) wherever quoted.**

## Compute note
Four engineer worktrees are running this wave on 8 cores. **Items 1 and 2 are cheap and are the
priority** (item 1 resolves a live contradiction, item 2 scopes a headline number and another
task's acceptance criterion). Item 3 is explicitly droppable — **report it as NOT RUN if you
cannot afford it; never as a null.**

## Deliverable
PREREG first (`.claude/outputs/carried-remeasurements/PREREG.md`) — registered expectations for
items 1 and 2 before running, including what you expect the occupancy curve's *shape* to be.
Report at `.claude/outputs/carried-remeasurements.md`, standard format, PREREG scorecard,
reconciliation list in the first 10 lines. Every number re-derived from a saved metrics JSON, not
transcribed from stdout. **No tracked file may be modified** — verify with `git status --short`
before and after, and say so in the git-footprint section.
