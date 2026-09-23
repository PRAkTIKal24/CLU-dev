# w18 spoke/seeds-and-tables: FINDINGS (first pass. ⛔ A SECOND PASS IS REQUIRED)

**Branch:** `w18/seeds-and-tables`. **Base:** `base/iter8 = 148f3b7`. **Machine:** batmac. **Scored runs: 0. Compute: 0.** No `.tex` file and no `\PLACEHOLDER{}` was touched.
Every number in this file is a registered cell in `TABLES.md`, printed there with its `[file:line]`.

## 0. What shipped

| file | what |
|---|---|
| `TABLES.md` | The fold specification: the environment register, the w16 CMAZE-T tables (W16-T1…T4), the w16 CMAZE-O tables (W16-O1…O3), the budget ledger and the CMAZE-2b fallback, a placeholder-by-placeholder map, the w15 awaiting and not-folded lists, the 1× bracket, and a seed-column and byte-convention audit of all 24 draft tables |
| `TABLES-w18.json` | Machine-readable: 88 cells, each with its value, source file and line, a `verbatim_on_line: true` check, and the sha256 of each source file |
| `build_tables.py`, `TABLES.template.md` | The table sources. `python build_tables.py` regenerates `TABLES.md` and **fails** if any cell is not verbatim on its cited line, or if any decimal or comma-grouped number enters without a cell. A negative test was run: one cell mutated (`0.693 vs 0.167` → `0.176`) plus one bare `0.7246` in the template. Both were caught, exit 1. Two consecutive builds gave byte-identical output |

## 1. ⛔⛔ The most important finding: the six "landed" w15 spokes have NO artefact on batmac

The launch says `table-fixes`, `crossover-seeds`, `single-design-seeds`, `cmaze-t-harness`, `cmaze-o-harness` and `maze-deletion-2b` have landed. On this machine:
- `evidence/outputs/` has no `w15-*` directory.
- `outputs/w15-table-fixes/` is empty.
- Branch `w15/table-fixes` has no commit past `148f3b7`.
- The harness merge commit `b1ca534`, named at `w16-advisor-report.md:3`, is not an object in this repository.

The sibling `w18/registry-fold` reached the same result independently (its FINDINGS §2).
⇒ **None of the six can be folded without guessing, and this spoke did not guess.** They are listed as **DECLARED NOT-FOLDED: artefact absent** (`TABLES.md` §6.2), with the exact table and row each will fill. The second pass needs their output directories copied into `~/clu/evidence/outputs/`. After that the work is mechanical: add cells, rebuild.
The single w15 number a batmac artefact quotes is folded **as a secondary quote**: the `crossover-seeds` K\* sd **0.0259** against the **0.02** tolerance (`w17-advisor-report.md:199`).

## 2. Wave 16: folded from the Hub's report, because the primary record is not here

`w16-advisor-report.md` is the only w16 artefact on batmac. The primary record, `/c/chlu/outputs/w16-maze-wm/`, is on FORGIS_14. Everything the report prints is folded, and every row it does not print is marked **NOT FOLDABLE**.
- **CMAZE-T** (W16-T1…T4). ENVIRONMENT CMAZE-T ONLY:
  - The A1 store row at n = 3: CLU 0.7246 (0.800 / 0.566 / 0.808) vs the best rival at 0.4035. All four controls collapse, and the cross-maze lookup floor is 0.0000.
  - The rivals re-tested at 80,000 steps.
  - The C-512 replication at n = 3, carrying its seed-1 `random_atoms` caveat.
  - Config B failed validation, and that is an interface finding.
  - ⛔ **Rollout validity = 0.000 for every learned arm. Per-step 0.925, compounding over 43 steps. It is printed as a declared negative, link READ, and not omitted.**
  - Deletion exactness on the winner: relative ΔS = 0.0 on every seed.
- **CMAZE-O** (W16-O1…O3). ENVIRONMENT CMAZE-O ONLY, in a separate table:
  - ⛔ **The 0.693 vs 0.167 margin is printed as WITHDRAWN BY THE HUB, never as a win.** Three scored runs (4–6) are attached to a claim that does not stand.
  - Against converged 30k-step rivals, the CLU reads 0.088 / 0.150 / 0.173 and the best rival 0.315 / 0.413 / 0.463.
  - Declared negative: no store row at any config, link INTERFACE.
  - Configs A and C2 failed validation. Runs 10–12 are declared not-run.
- **CMAZE-2b** Task 3 (runs 13–18): DECLARED NOT-RUN, in its own entry.
- **CMAZE-1, CMAZE-2, CMAZE-2b, CMAZE-T and CMAZE-O never share a table.** The register is `TABLES.md` §1. CMAZE-2b's deletion transfer (`tab:maze-del-2b`, pending w15) is a separate table from `tab:verb`.

## 3. ⚠ The 1× opponent: the BRACKET is printed, not one convention. Why

The ladder (K = 5) sized its 1× opponent from the full-square Σ d_k² ledger. The crossover / K\* replication used the symmetric-triangle float64 four-part total. **The bracket 938,413–1,710,324 is printed wherever the K = 5 opponent appears, and the K\* opponent (536,270–536,972) is labelled symmetric-triangle float64.**

Why one convention was not applied:
1. **Re-pricing is re-deriving.** Moving the ladder's opponent onto the triangle convention means recomputing ledgers from `well_sizes`, which this wave forbids.
2. **No same-convention opponent exists at 3 seeds.** The only in-bracket opponent (1,029,776 params) is **n = 1**, a replication, not scored. Presenting it as the single convention would promote a 1-seed row to the table's standard.
3. **The draft already prints the bracket** (`app:k5`) and drops the out-of-bracket rows. Keeping the bracket stays consistent with the draft and with referee-2's N1/N2.

Remaining gap: `tab:1x`'s caption does not name its convention. It is on the audit edit list (`TABLES.md` §8).

## 4. Seed column and byte conventions (the audit is `TABLES.md` §8)

- **All 24 draft tables carry a `seeds` column.** Single-seed rows print 1. No table prints "9/9 seeds".
- ✘ Four byte-bearing tables do not name their convention **in the table**:
  - `tab:hole`, `tab:rivals-del`: the source defines `bytes_rewritten` as the arm's whole parameter blob.
  - `tab:verb`: the lookup's "matched bytes" is not named.
  - `tab:1x`: the convention is in the prose, not the caption.
- `tab:maze-del` names a dtype for the statistic only.
- These are edits for placeholder-fill or the Head, because this spoke does not own the `.tex`.
- ✔ The ladder's n ≥ 3 training-data note travels with `tab:ladder` (caption and the following paragraph). ⚠ It is only partial under `tab:k5`, which has the same rungs.

## 5. Flags for the second pass and the Hub. None of these is a correction, because the primary record is absent

1. The A1 mean `0.7246` and the C-512 figure `0.725 vs 0.453` agree to three places, although they come from different configs and run ranges. Check against `w16-maze-wm/`.
2. The w16 deletion line does not say which scope it covers (value or conjunction), and it gives no post-deletion composition or cost. P19 and P20 are therefore only partially fillable.
3. The rollout line gives no seed count, and the 80,000-step re-test line does not name the config.
4. The w16 runs are indexed by run number, not seed id.
5. ⚠ **Label collision:** in the draft, `tab:cmaze-del` labels the **CMAZE-T** table. The CMAZE-2b deletion transfer needs its own label (proposed `tab:maze-del-2b`).
6. ⛔ **AMENDMENT-1 still binds:** no w15/w16 number enters a draft until the Head and the Advisor have discussed them. `TABLES.md` is the fold spec. Writing it into the `.tex` is not this spoke's decision.

## 6. DECLARED NOT-DONE

- **w15, AWAITING (three spokes), second pass required.** `bar-d-score` → a new `tab:bard` (never `tab:encloop`), P11 and P31. `coverage-control` → a new row in `tab:ladder`, n = 6 column, P4. `lace-poe-rows` → `tab:t7` LACE and PoE rows, plus LACE and PoE on every rung of `tab:ladder` and generation-table run 3, P3 and P7.
- **w15, NOT FOLDED because the artefact is absent (six spokes):**
  - `crossover-seeds` → `tab:kcurve`, K = 8…28 at seeds 1–2.
  - `single-design-seeds` → `tab:del` and `tab:rivals-del`, seeds 1–2.
  - `maze-deletion-2b` → the new `tab:maze-del-2b`.
  - `cmaze-t-harness` and `cmaze-o-harness` → P12.
  - `table-fixes` → MF2 and MF14(i).
- **w16, NOT FOLDABLE, because the rows are not in any batmac artefact:**
  - CMAZE-T: seen, 2-way and value hold-out; the within-maze lookup; the post-deletion composition and cost for both scopes; rival parameter counts and the store ledger.
  - CMAZE-O: seen, 2-way and value hold-out; controls, lookup and BLIND columns.
- **CMAZE-O deletion:** DECLARED NOT-RUN by construction, because there is no winner. The primary record should confirm this.
- **Not attempted:** any `.tex` edit, any placeholder, any figure, and any rebuild of w13's `build_tables.py`/`verify_tables.py`. Those read `/c/chlu/outputs/...` on FORGIS_14, which does not exist here.
