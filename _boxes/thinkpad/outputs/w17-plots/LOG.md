# w17 spoke/plots — LOG

Branch `w17/plots`, base `base/iter8 = 148f3b7`. Worktree `/c/chlu/worktrees/CHLU-w17-plots`.
Output dir `/c/chlu/outputs/w17-plots/`. Deliverables land in `/c/chlu/outputs/w17-paper/figures/`.
ZERO compute, ZERO scored runs. Every number re-read from `/c/chlu/evidence/outputs/`.

## 2026-09-22T21:45Z — start
- START_UTC.txt written.
- Read AMENDMENT-1.md, COMMON.md, plots.md in that order. AMENDMENT 1 binds:
  NO w15/w16 number in any figure; figures needing them are built from w14-and-earlier
  and clearly marked PLACEHOLDER for wave 18.
- Evidence base `/c/chlu/evidence/outputs/` contains NO `w15-*` or `w16-*` directories,
  so the amendment is enforceable by construction for the crossover/ladder/deletion figures.
  Confirmed: `ls /c/chlu/evidence/outputs | grep -E '^w1[56]-'` is empty.

## 2026-09-22T21:50Z — evidence reconnaissance
- `w13-seeds-and-tables/TABLES.md` + `TABLES.json` = the scoreboard with per-cell provenance
  (T1..T12). Read T7/T8/T9/T11/T12 sections.
- `w12b-nr6-and-kstar/` closes NR6: T7 now carries **3 seeds (0,1,2)** at K* = 32, and the
  K* replication of the ladder REVERSES the K=5 verdict (clears at all five rungs).
  This is the w12b artefact AMENDMENT 1 names (`NR6-T7.md`).
