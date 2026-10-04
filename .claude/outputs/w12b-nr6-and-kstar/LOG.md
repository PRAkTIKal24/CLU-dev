# w12b hub/nr6-and-kstar — LOG

Standing rules read. Task file `/c/chlu/tasks/w12b-hub-nr6-and-kstar.md` read first.
Worktree: `C:\chlu\worktrees\CHLU-w12b-nr6-and-kstar`, branch `w12b/nr6-and-kstar`, base `e58ab05`.
⛔ `hub/encoder-adapt` is TRAINING on this box — its session/worktree/output dir/PIDs are untouched
throughout. No box-wide kill anywhere in this log. 3 threads.

## 2026-09-21T21:27:20Z — START
- `START_UTC.txt` written.
- Read `/c/chlu/evidence/outputs/w12-advisor-report.md` (158 lines).
- Read `/c/chlu/SCORED-LEDGER.md` (41 lines); `single-design` stands at **1/3 spent, 2 left**
  (row at `SCORED-LEDGER.md:39`, w12 hub/single-design-crossover, 2026-09-21T12:53:06Z).
- Confirmed w12 artefacts on disk: `/c/chlu/outputs/w12-single-design-crossover/`,
  `/c/chlu/outputs/w12-composition-ladder/`, `/c/chlu/outputs/w12-kway-ladder-harness/`.
- All w12 scripts are folded into `base/iter8` at `e58ab05` (`scripts/w12_*.py`).

## 2026-09-21T21:33Z — JOB 1 (NR6): pre-registration and launch

### What was READ before anything was written
- `PREREG-w12-single-design-crossover-1.md` (+ its AMENDMENT 1) — REUSED, not rewritten.
  The w12 hub's output directory is **read-only** to this hub.
- `/c/chlu/outputs/w12-composition-ladder/FINDINGS.md`, `LADDER.json`, `runs/*/LEG.json`,
  `runs/*/ONEX_TARGET.json`, `LEDGER-ACROSS-K.json`, `CACHE-READY.json`,
  `BASELINES-READY.json`.

### ⛔ SEEDS — MEASURED TODAY, NOT INHERITED
| seed | cache rule | `decoder_v1_s<n>.pkl` | `gen-external-classifier/seed_<n>` |
|---|---|---|---|
| 0,1,2 | shipped gen-2 | PRESENT | PRESENT |
| **3** | PASSES (`array_shape [480000,128,4,4]`, 59 blocks, `status READY`) | **ABSENT** | **ABSENT** |
| **4** | PASSES (`array_shape [480000,128,4,4]`, 59 blocks, `status READY`) | **ABSENT** | **ABSENT** |
Paths checked: `C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-decoder-lift/winner/`
(keys `decoder_v1_s0/1/2` only) and `.../gen-external-classifier/` (`seed_0|1|2` only).
⇒ ⭐ **The task file anticipated this for seed 4. Measured, it holds for SEED 3 TOO.**
⇒ **3 seeds, DECLARED (0, 1, 2). ⛔ NEVER padded.**

### The dtype fix — located, not assumed
- The repair `_as_model_input` is at **`1635e0a`** (w12 spoke/deletion-baselines), an
  ancestor of `e58ab05` → **this base carries the fix** (`git merge-base --is-ancestor`).
- A SECOND, functionally equivalent repair **`8e5f202`** lives on `w13/encoder-adapt` and
  is **NOT** in this base.
- The `w12-deletion-baselines` comparators **post-date** the fix: `lstm__as_trained.rival.json`
  records `in_kind: "tokens"`, `param_count 130,721`, 327,680 gradient row-visits, 19.6 s —
  a token family that trained. Before the fix `eqx.nn.Embedding` refused to index at all.
- ⭐ Not relied on: `--refit-tokens` re-fits `transformer` and `lstm` **from scratch in the
  scoring process**, at the same 131,072-param pin on the same fit pool. Smoke-verified
  (`C:/msys64/tmp/claude/smoke_refit.py`): transformer width 45 / 130,433 params,
  lstm width 51 / 130,721 params — identical to the loaded comparators' pins.

### Written BEFORE any run
1. `PREREG-AMENDMENT-2.md` + `.timestamp.txt` = **2026-09-21T21:33:37Z**.
2. Two ledger rows in `/c/chlu/SCORED-LEDGER.md` at **2026-09-21T21:33:51Z**
   (`single-design` 2/3 and 3/3, `w12b hub/nr6-and-kstar`). Ledger 41 → 43 rows.

### Launched 21:34Z — 3 lanes × 1 thread (the 3 threads the task allows)
`scripts/w12b_launch.sh`, PIDs recorded in `logs/PIDS.txt`.
⛔ `hub/encoder-adapt` has 4 python processes on this 3-core box (PIDs 33948/21076/29968
train, 27236 score). **Not touched. No box-wide kill anywhere.**

| lane | log | run | K | seed |
|---|---|---|---|---|
| A | `logs/nr6_run2_K5_s0.log` | 2/3 | 5 | 0 |
| B | `logs/nr6_run3_K32_s1.log` | 3/3 | 32 | 1 |
| C | `logs/nr6_run3_K32_s2.log` | 3/3 | 32 | 2 |

Stores built: K=5 seed 0 → A=64, rank/width 0.1561–0.1565 (matches
`LEDGER-ACROSS-K.json` K=5 row, 0.1565). K=32 seed 1 → A=63, rank/width **0.5938–0.9844**;
K=32 seed 2 → A=63, **0.4375–0.9844**. ⚠ Seed 0 was uniformly 0.9688–0.9844; seeds 1 and 2
carry **rank-deficient wells** the seed-0 row never showed.

## 2026-09-21T21:39Z — ⛔ A DEFECT OF MY OWN, FOUND BEFORE IT COST A NUMBER

`scripts/w12b_scored_leg.py` passed `exp_gen_delete_row7.assert_generation2(seed)`'s dict
to `w12_deletion_baselines.save_comparator`. That dict carries **only**
`{generation, scope, n_rows, root}` — it has **no `features_sha256`**, and
`w4_api.save_learner` requires that key. The full manifest is
`fw.feature_cache_manifest(seed)`.

⚠ **Why this mattered and why it was nearly invisible:** the refit block is wrapped in a
per-family `try/except` that records a DECLARED NOT-RUN. The `KeyError` would have been
swallowed there, the scored rows would have completed looking perfectly healthy, and the
two token families — the one thing the wave's second binding instruction demands — would
have come back as a not-run with a misleading reason. The smoke test did not catch it
because the smoke passed `fw.feature_cache_manifest(0)` directly.

**Fixed in both runners** (`w12b_scored_leg.py`, `w12b_ladder_kstar.py`) with the reason
written at the call site, not silently.

**Restart, by PID and only by PID.** My own three processes — 34684, 8736, 35684 — were
stopped with `Stop-Process -Id`. ⛔ Verified immediately afterwards that
`hub/encoder-adapt`'s PIDs **33948, 21076, 29968, 27236 were still running and untouched**.
⛔ No box-wide kill. Relaunched at **21:39:49Z** as PIDs 140408 / 140412 / 140416
(bash wrappers; recorded in `logs/PIDS.txt`). Cost of the restart: ~6 minutes, none of it a
number — the address books had not finished.

## 2026-09-21T22:21Z — the harness reaped my WAITER SHELLS (not the work)

Claude Code stopped **nine** background wait-loop shells of mine at once because the box was
critically low on memory while the session was idle. ⛔ **This was the harness's action, not
a failure of the runs, and nothing of mine was killed by me here.** Measured immediately
after: all three scored legs were still alive and still scoring —
PIDs **14696** (run 2/3, K=5, seed 0), **35688** (run 3/3, K=32, seed 1),
**29124** (run 3/3, K=32, seed 2). ⛔ `hub/encoder-adapt` also untouched; it has moved on
from `--stage train` to `--stage rivals --seed 1` on three arms (PIDs 33364/30156/28920)
plus `w13_score_leg.py` (27236).

Memory at that moment: **FreeMB 3,663 of 64,949**. My three legs held 15,359 + 6,697 +
6,610 MB; `hub/encoder-adapt` held 11,467 + 2,448 + 2,451 + 2,448 MB.
⇒ ⚠ **This bounds the K\* ladder: 3 concurrent K = 32 legs at ~7 GB each is affordable only
once the NR6 legs exit, and the lanes are started only then.** Waiter shells are kept to
**one at a time** from here.

### The K=5 TEST address book and ledger — the first number of run 2/3
`address book 900.4s  195,584 B` →
**FOUR-PART LEDGER, TEST, K = 5, seed 0: atoms 524,288 + cov/count 3,391,552 +
address 195,584 + coupling 6,610 = 4,118,034 B = 7.8545× the 524,288 B REFERENCE column.**
⭐ **Byte-for-byte identical to `LEDGER-ACROSS-K.json`'s K = 5 row**, which was derived on
`val`. ⇒ the K = 5 1× figure of **1,029,508 float32 params** is now confirmed on **TEST**.

### The K* = 32 ledgers across seeds — the 535,784 pin, measured
| seed | atoms | cov+count | address book | coupling | honest total | 1× params | source |
|---|---|---|---|---|---|---|---|
| 0 | 516,096 | 560,608 | 1,060,864 | 6,610 | 2,144,178 | 536,044 | `w12-single-design-crossover/scored/run1_seed0_K32/scored.json` |
| 1 | 516,096 | 560,608 | **1,054,104** | 6,610 | **2,137,418** | **534,354** | `logs/nr6_run3_K32_s1.log` |
| 2 | 516,096 | 560,608 | **1,058,264** | 6,610 | **2,141,578** | **535,394** | `logs/nr6_run3_K32_s2.log` |
| (val-derived pin) | 516,096 | 560,608 | 1,059,824 | 6,610 | 2,143,138 | 535,784 | `LEDGER-ACROSS-K.json` K=32 row |
⇒ ⭐ **The task file's 535,784-param pin holds across all three seeds to within 0.32 %.**
Only part 3, the address book, varies with the seed; parts 1, 2 and 4 are identical because
they are arithmetic on the partition shape.

## 2026-09-21T23:18Z — JOB 1 (NR6) COMPLETE. Both scored runs spent.

`scripts/w12b_nr6_table.py` → `NR6-T7.json` + `NR6-T7.md`. Every cell names its file.

### ⛔ THE CONTROL GATE FIRST — all four controls, every leg, checked BEFORE any margin
| leg | arm | `landscape_removed` | `landscape_shuffled` | `landscape_shuffled_across_wells` | `random_atoms` | pass |
|---|---|---|---|---|---|---|
| K32 seed 0 | 47.4954 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | YES |
| K32 seed 1 | 47.4037 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | YES |
| K32 seed 2 | 49.1193 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | YES |
| K5  seed 0 | 43.7890 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | YES |
⇒ ⭐ **ALL FOUR LEGS PASS. The rows are claimable.** ⚠ A control that collapsed at seed 0
was not assumed to collapse at seeds 1 and 2 — it was measured, and it did.

### ⭐ T7 NO LONGER PRINTS n = 1
| K | n seeds | row 3 (pair, 60 corners) | row 4 (n-way, 49 corners) |
|---|---|---|---|
| **32 = K\*** | **3** (0,1,2) | **32.350 ± 0.815** (31.600 / 32.233 / 33.217) | **67.177 ± 1.320** (66.959 / 65.980 / 68.592) |
| **5** | **1** (0) | **28.817** | **62.122** |
⭐ The seed-0 headline (31.600 / 66.959) sits INSIDE the three-seed spread on both rows.
**T7 is not a one-seed artefact.**

### ⭐ THE VAL/TEST GAP — the third pre-declared falsification, now testable, DOES NOT FIRE
The frozen rule is `K* = max{K : v(5) − mean_seeds v(K) ≤ 0.02}`, `v` = mean over corners of
the per-corner valid fraction. ⚠ Corner SETS differ (24 val vs 60+49 test), so levels are
never subtracted across conditions; the DROP within a condition is the rule's own quantity.
| condition | v(K=5) | v(K=32) | drop v(5) − v(32) | inside the 0.02 band? |
|---|---|---|---|---|
| val (3 seeds) | 0.89525 | 0.92901 | **−0.03376** | yes |
| TEST (paired, seed 0) | 0.82849 | 0.90125 | **−0.07276** | yes |
⇒ ⭐ **The drop is NEGATIVE on both conditions — K\* is BETTER than K = 5 on the held-out
condition, and by a WIDER margin there than on the selection set.** The rule's band is a
one-sided bound on a drop; there is no drop to bound. ⛔ No adverse val/test gap.

### ⭐ THE TWO TOKEN FAMILIES — pre- vs post-fix, MEASURED not asserted
Re-fitted from scratch in each scoring process at the same 131,072-param pin
(transformer width 45 / 130,433 params; lstm width 51 / 130,721 params — identical to the
loaded comparators' pins) on the same 16,384 fit rows, 327,680 gradient row-visits.
| leg | `transformer` loaded → refit (row3/row4) | `lstm` loaded → refit (row3/row4) |
|---|---|---|
| K5 s0  | 0.667/1.918 → **0.367/1.102** | 2.500/7.531 → **2.567/7.714** |
| K32 s1 | 0.667/0.633 → **0.300/1.143** | 2.700/7.000 → **3.083/5.020** |
| K32 s2 | 1.717/4.735 → **1.733/3.163** | 4.917/14.898 → **4.367/8.061** |
⇒ ⭐ **The re-fits land in the same band as their loaded twins on every leg.** The
`w12-deletion-baselines` comparators therefore genuinely POST-DATE the `1635e0a` repair and
are NOT the flat pre-fix artefacts the wave warned about. ⛔ **The artefact warning applies
to SPOKE A's wave-4 set (`rival_transformer` 0.489–0.531, `rival_lstm` 0.579–0.604), which
the LADDER scores and which is NOT used on any row here.**

### ⛔ DECLARED NOT-RUN on run 2/3
The ledger row for run 2/3 declared **seeds 0, 1, 2** at K = 5 on TEST. **Seed 0 was run;
seeds 1 and 2 are a DECLARED NOT-RUN.** Reason: the box has 3 usable threads shared with
`hub/encoder-adapt`, one K = 32 leg costs ~100 min at one thread (address book alone 2,000 s),
and the acceptance criterion's remaining clause — the K\* ladder at **all five rungs** — needs
every thread. ⛔ **K = 5 on TEST is therefore reported as n = 1, printed, never padded**, and
the val/test gap above is a SEED-0 PAIRED comparison, stated as such.

## 2026-09-22T00:01Z — JOB 2 (K\*): the FIRST leg lands and it FLIPS THE SIGN

`kstar_ladder/2_seed0_K32/` (log `logs/kstar_2_s0_K32.log`). Rung n = 2, seed 0.

| | store | `learned_1x` | opponent params | margin | margin / ceiling(36) |
|---|---|---|---|---|---|
| ladder **K = 5** (`w12-composition-ladder/runs/2_seed0/`) | 28.6333 | 33.2000 | 3,241,076 | **−4.5667** | −0.1269 |
| **K\* = 32** (this hub) | **31.8000** | **25.1333** | **534,866** | **+6.6667** | **+0.1852** |

⭐ **BOTH HALVES MOVE.** The store RISES 28.63 → 31.80 and the opponent FALLS 33.20 → 25.13,
because `learned_1x` is byte-matched to the K\* object's own four-part total —
**2,143,658 B → 535,914 target params**, essentially the task file's 535,784 pin — instead
of the K = 5 object's 12,967,352 B → 3,241,838. ⛔ **The opponent is 6.06× smaller, not 1.92×
smaller**; see the correction below.

⛔ **Control gate at K\*, this leg:** `landscape_removed` 0.0000, `landscape_shuffled` 0.0000,
`random_atoms` 0.0000 against an arm of 31.8000. All collapse. The row is claimable.

⚠ **ONE RUNG, ONE SEED, AND STILL CONFOUNDED.** The K\* arm differs from the ladder's K = 5
arm in TWO ways — K, and the construction (`CovContentStore` on the label-free
`ward_balanced` cut vs `DowndatableStore` on the label-MI `factor` partition). The BRIDGE
arm (`--K 5` in the K\* construction, queued on lane 3) is what separates them. ⛔ No
attribution is made until it lands.

### ⛔⛔ A CORRECTION THE ADVISOR REPORT NEEDS: the "1,029,508-param opponent" is not the ladder's
`w12-advisor-report.md:146` says *"the ladder's verdict stands against a 1,029,508-param
opponent where the main text will use 535,784."* ⛔ **Measured, the ladder's `learned_1x`
carried 3,131,564 – 3,260,030 parameters**, read from
`w12-composition-ladder/runs/<rung>_seed<s>/rows_learned_1x.json:summary.param_count` on all
15 legs (per-rung 3-seed means 3,174,386 / 3,174,386 / 3,214,400 / 3,177,194 / 3,180,704).
The ladder's own FINDINGS says so in words — *"a learner matched to its own 24.4× total"*.

**1,029,508 is a different object's number.** It is the CROSSOVER hub's K = 5 four-part total
in params, from `LEDGER-ACROSS-K.json` — ⚠ the same file whose first key is
`SMOKE_NOT_A_CLAIM: true` and which the advisor report's own §6 already had to correct once.
The two objects price the same statistic differently:
| | K = 5 statistics | convention |
|---|---|---|
| ladder `DowndatableStore` | **12,284,920 B** | FULL-SQUARE float64, `8·Σd_k²` |
| crossover `CovContentStore` | **3,391,552 B** | symmetric-TRIANGLE float64, `8·Σd_k(d_k+1)/2` |
and they sit on different partitions (label-MI `factor` vs label-free `ward_balanced`).

⇒ ⭐ **"the opponent is roughly HALVED" is true WITHIN the crossover convention**
(1,029,508 → 535,784 = 1.92×) **and is NOT a statement about the ladder's opponent.**
Against the ladder's actual opponent the K\* figure is **6.06× smaller** (3,241,838 → 535,914
at rung 2 seed 0). ⛔ That is a much larger change than the wave expected, and it is exactly
why the bridge column exists rather than a two-column before/after.

## 2026-09-22T00:30Z — ⭐ THE BRIDGE LANDS AND THE CONFOUND WAS REAL

`kstar_ladder/2_seed0_K5/` (log `logs/kstar_2_s0_K5.log`). Rung n = 2, seed 0, the crossover
hub's construction held fixed and **K set back to 5**.

| arm | store | `learned_1x` | opponent params | margin |
|---|---|---|---|---|
| ladder **K = 5** (`DowndatableStore`, label-MI `factor`) | 28.6333 | 33.2000 | 3,241,076 | **−4.5667** |
| **BRIDGE K = 5** (`CovContentStore`, label-free `ward_balanced`) | **29.6167** | **29.8667** | **1,029,776** | **−0.2500** |
| **K\* = 32** (same construction) | **31.8000** | **25.1333** | **534,866** | **+6.6667** |

⭐ **The bridge's four-part total measures 4,118,034 B → 1× target 1,029,508 params —
EXACTLY the advisor report's figure.** ⇒ confirmed by direct measurement on the ladder's own
rung-2 pool: **1,029,508 is the CROSSOVER CONSTRUCTION's K = 5 number, and never was the
ladder's opponent.** The ladder's opponent on that same leg was 3,241,076.

### ⭐⭐ THE SWING, DECOMPOSED — and this is why the bridge exists
| step | what moves | margin change | share |
|---|---|---|---|
| ladder K=5 → bridge K=5 | **the CONSTRUCTION** (partition rule + store class + covariance convention) | **+4.317** | 38 % |
| bridge K=5 → K\* = 32 | **K alone** | **+6.917** | **62 %** |
| total | | **+11.233** | 100 % |

⛔ **Without the bridge the whole +11.233 would have been charged to K.** The K step is the
larger part and it is the only part the Head's ruling is about — but **38 % of the swing is
the construction**, and a two-column before/after would have reported that as a K effect.

⭐ Note that the task file's *"the opponent is roughly HALVED"* is **exactly the bridge → K\***
step: 1,029,776 → 534,866 = **1.93×**. The 6.06× figure is the ladder → K\* step and spans
both changes.

⛔ Control gate on the bridge leg: `landscape_removed` 0.0000, `landscape_shuffled` 0.0000,
`random_atoms` 0.0000 against an arm of 29.6167. All collapse.

## 2026-09-22T00:48Z — ⭐⭐ ALL FIVE RUNGS AT K\* = 32, SEED 0. THE VERDICT REVERSES.

Acceptance criterion's ladder clause **MET**: the store arm has a K\* = 32 reading at all
five rungs with `learned_1x` refit at **534,866 params** (target 535,394–535,914 per leg,
the task file's 535,784 pin).

### ⛔ THE CONTROL GATE FIRST — every rung, seed 0
| rung | arm | `landscape_removed` | `landscape_shuffled` | `random_atoms` | exactly 0? | collapses beyond the declared 1.0 margin? |
|---|---|---|---|---|---|---|
| n=2 | 31.800 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=3 | 60.150 | 0.0000 | **0.0083** | 0.0000 | **no** | **YES** |
| n=4 | 65.242 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=5 | 65.642 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
| n=6 | 65.908 | 0.0000 | 0.0000 | 0.0000 | yes | **YES** |
⚠ **Rung 3's `landscape_shuffled` is 0.0083, NOT zero** — one novel tuple surviving across
120 corners, `judge_acc 0.0001`. ⛔ **Printed, not rounded away.** It collapses by the
PRE-DECLARED bar (`LADDER-READY.json`: `DECORATION_MARGIN_NOVELTY = 1.0`) against an arm of
60.150, by a factor of 7,200. ⛔ The bar is not moved, and "exactly 0.0000" — which was the
crossover hub's *observation* on val, never a declared bar — is reported separately.

### ⭐⭐ THE HEADLINE: THE SIGN FLIPS AT 4 OF 5 RUNGS, AND THE STORE NOW WINS AT ALL FIVE
Seed-matched (seed 0 against seed 0 — ⛔ never a 3-seed mean against a 1-seed mean).
| rung | store K=5 | store K\* | opp K=5 params | opp K\* params | opp shrank | margin K=5 | margin K\* | sign flipped |
|---|---|---|---|---|---|---|---|---|
| n=2 | 28.633 | **31.800** | 3,241,076 | 534,866 | 6.06× | −4.567 | **+6.667** | **YES** |
| n=3 ⚠same corners | 56.950 | **60.150** | 3,224,228 | 534,866 | 6.03× | +4.817 | **+17.875** | no (already +) |
| n=4 ⚠same corners | 60.933 | **65.242** | 3,260,030 | 534,866 | 6.10× | −6.242 | **+12.700** | **YES** |
| n=5 ⚠same corners | 63.242 | **65.642** | 3,236,864 | 534,866 | 6.05× | −4.808 | **+10.350** | **YES** |
| n=6 ⚠same corners | 62.692 | **65.908** | 3,238,970 | 534,866 | 6.06× | −5.775 | **+9.375** | **YES** |

⭐ **The sign flips at EXACTLY the four rungs where the store lost at K = 5**, and at the one
rung where it already won the margin grows 3.7×. ⇒ ⭐⭐ **At K\* the store wins at ALL FIVE
RUNGS.** The wave-12 verdict — *"the store loses to a same-bytes MLP at 4 of 5 rungs"* — is
a property of **K = 5**, and under the Head's ruling the main text carries **K\***.
⛔ **This is a CORRECTION to the K = 5 verdict and must be reported as one, not buried.**

⭐ **Both halves move, and it matters which.** The store's own score rises at every rung
(+2.4 to +4.3 novel tuples) — the finer partition genuinely reads better. The opponent falls
6.03–6.10× because it is byte-matched to an object whose four-part total is 6× smaller.

⛔ **AXIS LABEL, unchanged and carried here too:** rungs n = 3, 4, 5, 6 probe the **IDENTICAL
120 corners** (`w12 spoke/seeds-and-tables`); for n ≥ 3 this is a **training-data axis**, not
a composition-depth axis, and only **n = 2 vs n = 6** is a genuine depth contrast. The rungs
are reported; the axis is named for what it is. ⭐ The flip is present at **both** ends of
the only genuine contrast (n=2: −4.567 → +6.667; n=6: −5.775 → +9.375), so it does not rest
on the confounded middle.

## 2026-09-22T02:46Z — all 15 K\* legs finished (lanes completed while this session was
## stopped by an API 529, not by a finding). 2026-09-22T06:06Z — the 5 bridge legs finished.

`logs/KSTAR-PROGRESS.txt` carries every START/END with `rc=0` on all 20 legs.
**0 `FAILED_*.json` anywhere; `failed_arms` empty in all 20 `LEG.json`.**
Legs: 5 rungs × 3 seeds at K\* = 32, plus 5 rungs × 1 seed at K = 5 (the bridge) = **20**,
each with 5 `rows_*.json` (store + 3 controls + `learned_1x`) = **100 arm-runs**.

⛔ The box was free of `hub/encoder-adapt` by 05:51Z (it had finished its own work); this hub
still ran **3 lanes × 1 thread** and never raised the thread count.

### ⛔ CONTROL GATE, ALL 15 K\* LEGS — checked BEFORE any margin
All 15 collapse by the pre-declared bar (`DECORATION_MARGIN_NOVELTY = 1.0`).
**13 of 15 read exactly 0.0000.** ⚠ Two do not: **n=3 seed 0 and n=3 seed 1,
`landscape_shuffled` = 0.0083** (one novel tuple surviving over 120 corners,
`judge_acc 0.0001`). ⛔ Printed, not rounded. Against arms of 60.150 and 62.783 they
collapse by a factor of ~7,200. ⛔ The bar is not moved, and "exactly 0.0000" — the
crossover hub's *observation* on val, never a declared bar — is reported separately.
All 5 bridge legs read exactly 0.0000 on all three controls.

### ⭐⭐ THE HEADLINE, AT 3 SEEDS: THE K = 5 VERDICT REVERSES
Seed-matched, 3 seeds, `margin = store − learned_1x` in novel tuples per corner:
| rung | store K=5 | store K\* | opp K=5 params | opp K\* params | shrank | margin K=5 | margin K\* | sign flipped |
|---|---|---|---|---|---|---|---|---|
| n=2 | 30.539 | **32.644** | 3,177,896 | 536,972 | 5.92× | −2.783 | **+4.356** | **YES** |
| n=3 ⚠ | 59.750 | **61.614** | 3,174,386 | 536,270 | 5.92× | +4.242 | **+14.072** | no (already +) |
| n=4 ⚠ | 62.708 | **66.086** | 3,214,400 | 536,270 | 5.99× | −5.283 | **+10.022** | **YES** |
| n=5 ⚠ | 64.719 | **66.581** | 3,177,194 | 536,972 | 5.92× | −3.900 | **+8.356** | **YES** |
| n=6 ⚠ | 64.556 | **66.603** | 3,180,704 | 536,972 | 5.92× | −4.061 | **+7.978** | **YES** |

### ⭐⭐ THE LADDER'S OWN PRE-DECLARED BAR, EVALUATED AT BOTH K — the cleanest statement
`AMENDMENT-2 §3` (w12 hub/composition-ladder): clears only if it beats the opponent by
**1.0 novel tuple per corner on EVERY declared seed**, not on the seed mean. ⛔ Not moved.
| rung | clears at K=5? | worst seed K=5 | clears at K\*? | worst seed K\* |
|---|---|---|---|---|
| n=2 | no | −4.567 | **YES** | **+3.117** |
| n=3 | no | −0.033 | **YES** | **+11.533** |
| n=4 | no | −6.350 | **YES** | **+8.625** |
| n=5 | no | −4.808 | **YES** | **+6.517** |
| n=6 | no | −5.775 | **YES** | **+6.108** |
⇒ The ladder's published finding was *"There is NO BREAKING POINT, because the store NEVER
CLEARED THE BAR."* ⭐⭐ **At K\* it clears at ALL FIVE RUNGS on EVERY SEED.**
⭐ Every one of the 15 per-seed K\* margins is positive; the smallest is **+3.117**.

### ⭐⭐ AND THE BRIDGE SAYS THE FLIP IS K, NOT THE CONSTRUCTION
| rung | seed 0 margin: ladder K=5 → BRIDGE K=5 → K\* | construction step | K step | K's share |
|---|---|---|---|---|
| n=2 | −4.567 → **−0.250** → +6.667 | +4.317 | **+6.917** | 62 % |
| n=3 | +4.817 → **+8.425** → +17.875 | +3.608 | **+9.450** | 72 % |
| n=4 | −6.242 → **−1.017** → +12.700 | +5.225 | **+13.717** | 72 % |
| n=5 | −4.808 → **−0.517** → +10.350 | +4.292 | **+10.867** | 72 % |
| n=6 | −5.775 → **−2.550** → +9.375 | +3.225 | **+11.925** | 79 % |

⭐⭐ **THE DECISIVE FACT: the BRIDGE still LOSES at 4 of 5 rungs** (wins only at n=3 — the
identical pattern to the ladder's own K = 5 verdict). ⇒ **Changing the construction alone
does NOT flip the verdict. The flip is the K step**, worth 62–79 % of every swing.
⛔ Without the bridge the whole swing would have been charged to K and the claim would have
been unattributable; with it, the claim is *"K\* flips it"* and that is now measured.

⭐ The task file's *"the opponent is roughly HALVED"* is exactly the bridge → K\* step:
**1,029,776 → 536,270–536,972 = 1.92×**. The 5.92–5.99× figure spans both changes.

## 2026-09-22T06:07Z — CLOSE

- `FINDINGS.md` written (§1 is the K\* reversal, first and not buried).
- `NR6-T7.json` / `NR6-T7.md`, `KSTAR-LADDER.json` / `KSTAR-LADDER.md` regenerated from the
  artefacts on disk. ⛔ Nothing re-derived from memory; every cell names its file.
- **Completeness checked, not presence:** 3 scored legs × **12 arms** each with
  `row3_pair 60` + `row4_nway 49` corners and **zero** swallowed refit NOT-RUNs;
  **20 K\* ladder legs × 5 `rows_*.json` = 100 arm-runs**; `failed_arms` empty in all 20
  `LEG.json`; **0** `FAILED_*.json` anywhere.
- **Bundle:** `git bundle create ... w12b/nr6-and-kstar` — **that branch only, never
  `--all`**. `git bundle verify` → *"is okay ... records a complete history"*.
  `git bundle list-heads` → **exactly ONE ref**,
  `3a8522d9cb37d263717ee6e57b8102e5e4c66d86 refs/heads/w12b/nr6-and-kstar`, equal to
  `git rev-parse w12b/nr6-and-kstar`. 6,954,919 B.
- Independently **cloned the bundle** into a scratch dir and confirmed the tip, all 8
  `scripts/w12b_*` files and the 9 commits since `e58ab05`; scratch clone deleted.
- ⛔ `hub/encoder-adapt` untouched throughout. The only processes this hub ever stopped were
  its own three, by PID, at 21:39Z, after which encoder-adapt's four PIDs were verified still
  running. **No box-wide kill at any point.**
