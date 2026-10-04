# DRAFT ledger rows for `/c/chlu/SCORED-LEDGER.md`

⛔ **Nothing here has been appended to the shared ledger.** A row is appended **only** at
the moment a scored run actually starts, and `scripts/w12_scored_leg.py` refuses to run if
its row is not already there (`assert_gates`). This file exists so the row is written from
a prepared text rather than composed in a hurry, and so a reader can see exactly what
would be spent.

**Gate state at the time of writing (11:29Z):**
`/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` — **ABSENT**.
`/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json` — **ABSENT** ⇒ **3 seeds, DECLARED**.

---

## Row template — `single-design` run n/3

```
| <UTC> | w12 hub/single-design-crossover | single-design | n/3 | outputs/w12-single-design-crossover/PREREG-w12-single-design-crossover-1.md | **WHERE DOES COMPOSITION BREAK AS THE PARTITION GETS FINER?** THE PIN CHANGED, NOT THE OBJECT: `CovContentStore` at rank A = min(64, min_k d_k − 1) — 64 per well, capped STRICTLY below the well width — on the LABEL-FREE `ward_balanced` correlation cut at **K\* = <K>**, selected on `val` ONLY by a rule frozen and timestamped at 2026-09-21T11:10:24Z (`SELECTION-RULE.md` + `SELECTION-RULE.timestamp.txt`) before any val composition number existed. FOUR-PART LEDGER PRINTED AND NEVER NETTED (atoms | covariance+count | address book | coupling/hole table) = <B> B; the 524,288 B pin is a REFERENCE COLUMN this wave and the learned baselines are matched to the object's OWN total (1×) — ⚠ or, if `BASELINES-READY.json` names no 1×-matched entry point, the available rivals are matched to 524,288 B (131,072 params), which is a comparison IN THE RIVALS' DISFAVOUR relative to 1× and therefore FLATTERS this hub, so no row measured against them is reported as a 1× result. SCORED: rows 3 (60 pair sub-corners) + 4 (49 n-way corners) on TEST, metric `distinct_valid_novel_tuples` per corner, verdict = `fw.paired_estimand` (the corner is the unit; never a single cell), margin +1.0/corner declared in advance. CONTROL GATE ON EVERY ROW, and no row is claimed unless all pass: `landscape_shuffled` (atom values permuted), `landscape_shuffled_across_wells`, `random_atoms`, `landscape_removed` — ⛔ wave 9's maze arm was decorative at rank/d_k = 0.938 and was caught only by these; rank/width is printed on every row and at K=32 it is 0.9844. DELETION, three scopes with a cost line per arm (seconds · bytes rewritten · data rows read = 0 for us): (a) VALUE red = 11,044 rows; (b) CONJUNCTION red ∧ cube = **0 rows, DECLARED VACUOUS** — `shape=cube` is held out of the split entirely (`exp_gen_delete_row7.py:325`), agreed independently by `spoke/deletion-baselines`; (b′) the DECLARED NON-VACUOUS SUBSTITUTE red ∧ sphere = 5,511 rows, index ADOPTED from that spoke's pin so comparator and arm delete the same rows; (c) ITEM a random 1 % of fit rows. "EXACT" means **STORE-SCOPED**: the frozen encoder and decoder are NOT refit. ⛔ Never "no changes after deletion" — post-deletion adaptation is LIGHT and DATA-FREE. 3 SEEDS (0/1/2), DECLARED: seeds 3 and 4 have no generation-2 cache and that spoke projects ~1 h 50 m per seed. Seed <s>, TEST leg. |
```

## Row template — `generation-table` run 3/3

```
| <UTC> | w12 hub/single-design-crossover | generation-table | 3/3 | outputs/w12-single-design-crossover/PREREG-w12-single-design-crossover-1.md | <RESULT 2 of the prereg: `exp_gen_cell.py` on the same K\* object against the full rival set including the table; `external composition_B aggregate` as the headline and `in_corner` balanced accuracy with its mandatory AUROC companion, each with its `distinct factor_tuples` count — a rate is never reported without its count. ⛔ The cell instrument is a single 72-request corner and is never quoted against a row-3/4 verdict.> |
```
