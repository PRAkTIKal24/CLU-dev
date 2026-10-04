# w13 spoke/referee-2 — LOG

| UTC | event |
|---|---|
| 2026-09-22T06:22:17Z | START_UTC.txt written. |
| 2026-09-22T06:22:17Z | GATE checked OPEN: /c/chlu/outputs/w13-draft-revise/DRAFT-v1.md (124769 B) and END_UTC.txt (2026-09-22 08:20 local) both exist. |
| 2026-09-22T06:23Z | Read task file /c/chlu/tasks/w13-spoke-referee-2.md incl. standing-rules line. Zero compute, zero scored runs, 1 thread. |
| 2026-09-22T06:30Z | Read REFEREE-v0.md whole (631 lines, MF1–MF16, SF1–SF20, NTH1–8, promoted B3 item). |
| 2026-09-22T06:40Z | Read DRAFT-v1.md whole (1,308 lines): §0–§11, §A–§J. |
| 2026-09-22T06:50Z | Opened the NEW load-bearing artefact `/c/chlu/outputs/w12b-nr6-and-kstar/` read-only: KSTAR-LADDER.json (all 45 leg records), NR6-T7.json, FINDINGS.md, kstar_ladder/*/LEG.json, rows_*.json. ⛔ Its session/worktree/processes never touched. |
| 2026-09-22T06:55Z | Dumped all 45 (rung x leg x seed) records: store, learned_1x, margin, opponent params, four-part ledger, controls. Re-derived all 15 K* margins; min +3.117 — the draft's claim reproduces. |
| 2026-09-22T07:05Z | ⭐ Found the covariance-convention collision. `w12-composition-ladder/BYTES.json` `/legs/*/single_design_K5/p2_covariance_bytes` = Sum_k d_k^2 x 8 EXACTLY on all three seeds (widths [777,936,216,60,59]->12,268,496 +16,424 = 12,284,920; [814,880,210,72,72]->11,948,136; [836,847,223,68,74]->11,825,496). The K* and crossover ledgers are the symmetric TRIANGLE (K=32: 32*(64*65/2)*8 + 2048*8 + 32*8 = 549,120 exactly; crossover K=5 = 3,380,064 exactly). The same file carries `p2_if_symmetric_triangle_float32` = 3,071,220. |
| 2026-09-22T07:15Z | ⭐ Verified the ladder and bridge legs share IDENTICAL corner id sets (60 at n=2, 120 at n>=3) and the same cache gen 2 / 32,768 fit rows, so a paired substitution is legitimate. Recomputed the ladder store's K=5 margins against w12b's already-fitted same-convention 1,029,776-param opponent, per corner. |
| 2026-09-22T07:25Z | Re-ran `w13-seeds-and-tables/verify_tables.py` myself (read-only, 1 thread, no scored budget): 52/52 sources, 872 cells, 1,786 per-seed values, 8 figure hashes, 10 NOT-RUNs, 19 corrections — PASS. The draft's App. M claim is sound. |
| 2026-09-22T07:40Z | Checked MF1–MF16 item by item against the draft and the artefacts. 14 CLOSED, 2 PARTIAL (MF2, MF14). Eight new must-fixes. |
| 2026-09-22T07:50Z | REFEREE-v1.md written to /c/chlu/outputs/w13-referee-2/. |
| 2026-09-22T07:52Z | Committed to branch w13/referee-2 as cb3b628, under `w13_referee_2/` (the w13 convention — `projects/*/` is in .gitignore and was NOT force-added). |
| 2026-09-22T07:53Z | `git bundle create .../w13-referee-2.bundle w13/referee-2` — that branch only, ⛔ never --all. `git bundle verify` = okay, complete history. `list-heads` shows EXACTLY ONE ref: cb3b628 refs/heads/w13/referee-2. |
| 2026-09-22T07:54Z | END_UTC.txt written last. |

## FINAL REPORT

⛔ **Scored runs: ZERO. Compute budget: ZERO. Fitted models: ZERO. 1 thread.**
⛔ **`$CHLU_CLAUDE_ROOT` NOT read.** ⛔ **No other agent's session, worktree, output dir or process
touched. No background job launched. NO BOX-WIDE KILL.** ⛔ Nothing written outside
`/c/chlu/outputs/w13-referee-2/` and this worktree.

**Verdict:** MAJOR REVISION — no longer a reject, not yet an accept.
R1 3→7 · R2 5→5 (→7 after N1–N4) · R3 6→7 · AC: reject → major revision.

**MF1–MF16:** 14 CLOSED · 2 PARTIAL (MF2, MF14) · 0 OPEN. The draft's "all sixteen CLOSED" overstates
by two.

**Eight new must-fixes, five in the byte ledgers, all closable at zero scored runs** — N1 the
full-square vs symmetric-triangle covariance convention across the K = 5 ↔ K\* comparison; N2 the
same-convention opponent w12b already fitted; N3 the two dropped bridge columns; N4 the weaker K\*
control gate; N5 one ratio against three totals; N6 two objects called "K = 5"; N7 the contribution
paragraph's mislabelled statistic; N8 three contradictory instructions on where the ladder loss prints.

**Independently re-verified and sound:** all 15 K\* margins (min +3.117), the seed means
+4.356/+14.072/+10.022/+8.356/+7.978, every four-part ledger sum, T7's three-seed mean and SD, the
ladder opponent param range, the 13-of-15 control zeros, and `verify_tables.py` (52/52/872/1,786/8,
PASS).

**Could not break:** the n ≥ 3 training-data axis labelling (§4.3) and the n = 3 non-win (§4.2) — both
handled at the right altitude. No must-fix on either.
