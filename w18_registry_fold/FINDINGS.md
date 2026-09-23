# w18 spoke/registry-fold — FINDINGS (SECOND PASS; ⛔ THIRD PASS REQUIRED for three w15 spokes)

**Agent:** `doc-curator`. **Branch:** `w18/registry-fold`. **Machine:** batmac.
**Commissioned by:** `tasks/w18/launch-registry-fold-pass2.txt`.
**Scored runs: NONE. Compute: NONE. Background jobs: NONE. Sub-agents: NONE.** tmux was not touched and no process was killed.
⛔ `~/clu/evidence/` was not written. The base md5s are unchanged from pass 1 (registry `5568a6f2…`, matrix `6b8042dd…`), and the `w16-advisor-report.md` md5 is unchanged since arrival (`aace0197…`).

## 0. WHAT SHIPPED IN PASS 2

| file | pass 1 → pass 2 | what pass 2 adds |
|---|---|---|
| `negative_results.md` | 6,250 → **6,819** lines | **`N345`–`N357`** and `◈.4`–`◈.7`, a header note, and inserted `⟲ PASS 2` lines at `N339`, `N340` and `◈.3` |
| `claims_matrix.md` | 1,731 → **1,899** lines, **v2.23 (PROPOSED)** | **`§I8.12`**: `CM-W16-1`…`CM-W16-4`, `CM-W15-1` and `CM-W17-5`. **`§I8.13`**: 11 open flags. Also `⟲` lines at `CM-W17-1` and `CM-W17-4` |

**Additivity was checked with difflib against the pass-1 files:**
- Registry: 6,250 lines equal, 569 inserted, **0 replaced, 0 deleted**.
- Matrix: 1,731 equal, 168 inserted. Line 1 (the title) is the only replaced line, and the prior title is kept verbatim as its tail (`endswith` True).
- Across the registry's 349 `### N` headings there are **zero duplicate numbers**, and each of N345–N357 appears exactly once.

**Number audit.** Every decimal and thousands token in the new text (235 in the registry block, 57 in the matrix block) was checked mechanically and found **verbatim in a source read at this pass**. The sources were `w16-maze-wm/**`, the six w15 dirs, `w18-seeds-and-tables`, `w18-referee-fixes`, `DRAFT-v2.tex`, w12 crossover and pass-1. The audit caught one curator rounding (the N346 control values), which was replaced with the JSON's exact values.

## 1. THE FIVE LAUNCH ITEMS

| # | item | entry | the one-line result (numbers are the primary record's) |
|---|---|---|---|
| 1 | w16 configs that **failed VAL** | `N345` | **CMAZE-T Config B, VB3**: learned blind-seq 0.8781795… against a bar of 0.8757985… (CLU 0.925798…). The interface lifts the rival **+0.40** and the CLU **+0.20**. Link **interface** |
| | | `N346` | **CMAZE-O Config A, VB0**: best 0.020833… (0.0125 / 0.015 / 0.035) against a bar of ≥ 0.05. Link **content/read** |
| | | `N347` | **CMAZE-O Config C2, VB1 against 30k-step rivals**: CLU 0.13667 (0.0875 / 0.15 / 0.1725) against MLP 0.39667 (0.315 / 0.4125 / 0.4625), margin −0.26. ⚠ The **same CLU number PASSED** against the recipe rivals (MLP 0.0342). Link **read** |
| 2 | the w16 scored config that missed | `N348`, `CM-W16-1` | **C-field, runs 4–6.** The **rival margin is WITHDRAWN BY THE HUB**: the MLP at 30k steps reads 0.9875 / 0.99 / 0.99 against the CLU's VAL 0.64 / 0.73 / 0.73. ⭐ The **STORE GATE HOLDS**: all four controls are 0.000 on every seed (`MAZE-WM.json` l.369). **Two statements, never one**. Deletion **ran** |
| 3 | failing links | `N349` | **CMAZE-O has three links**: content/read, interface and read. **CMAZE-T rollout: READ** (validity 0.000 for the CLU and every matched rival, per-step 0.925 against 0.818) |
| 4 | six landed w15 spokes | `N352`–`N357` | Each is folded from its own FINDINGS, with its declared not-runs. The headline negatives: Shapes deletion is **not capability erasure** (`N355`); CMAZE-2b deletion **does not induce refusal** (`N356`); the CMAZE-O harness is **interface-only** (`N353`); **K\* is not resolvable at n = 3** and **K = 20 fails the gate** (`N354`) |
| 5 | ⭐⭐ the Hub report disagreed in **eight** places | **`N351`**, `CM-W16-4` | The artefact won all eight, and the curator checked each primary pointer. **A paraphrase is not a source.** ⚠ New finding: the on-disk report is the **corrected** version (sha256 `a533a67c…`), and §4b's report line pointers resolve only against the **superseded** version (sha256 `3c2115bd…`), which is no longer on batmac. That wording survives only in `TABLES-w18.json`. A residual remains in the corrected report at l.72 |
| — | *(curator-added)* | `N350`, `CM-W16-3` | The shipped across-wells control was **partly the identity** (`rng.permutation(K)`). The w16 fix is a derangement. ⚠ The w15 CMAZE-T smoke (Config-A K 4 / 4 / 8) is exposed |

**M1 and M3 are carried forward OPEN.** Both are HELD per `w18-referee-fixes/FINDINGS.md` l.4–5, and the ⟲ lines at `N339` and `N340` plus `CM-W17-5` record this.
⭐⭐ **M1 recurred LIVE in w16.** C-field's recorded win was withdrawn purely because the rivals were under-trained, and C2 flips WIN → FAIL on the rivals' budget alone. This is recorded against the M1 entry as evidence that the ruling is **load-bearing, not cosmetic**.
The pass-1 instruction to close items against referee-fixes is **done**: M2, M4, M5, M6 and M7(e) are APPLIED, hash-verified against the live draft (`8e45fa98…`).

## 2. ⛔ WHAT DID NOT RUN — THIRD PASS REQUIRED

**Still awaiting (running on the other box). NOT FOLDED, and nothing guessed** (registry `◈.7`):
- **`bar-d-score`**: the BAR D full-decoder GPU row, which goes to `tab:bard` (never `tab:encloop`); P11, P31.
- **`coverage-control`**: the n = 6 row on n = 3's train index, in `tab:ladder`; P4.
- **`lace-poe-rows`**: separate LACE and PoE rows on `tab:t7`, every ladder rung, and generation-table run 3; P3, P7.

**Declared not-done:**
- **Any draft edit.** The greps show `DRAFT-v2.tex` still carries 23.5117 (L1317, L2262) and "+0.012731 at K = 10" (L1247). MF14(i) is not carried. These belong to the Head or `placeholder-fill`, and AMENDMENT-1 applies.
- **Any evidence edit**: the report's l.72 residual, `w14-read-verb/FINAL-REPORT.md`, `w9-maze/LOG.md` l.220 and `w15-maze-deletion-2b/TABLE.md`.
- **Any ruling on M1 or M3.**
- **Adjudicating w12's 0.0093 against 0.0127.**
- **Re-deriving or rounding any number.**

**A path note:** `w18-seeds-and-tables/` is **not** under `evidence/outputs/`, although the launch file says so. It was read from `~/clu/outputs/w18-seeds-and-tables/` (§I8.13 flag 2).

## 3. OPEN FLAGS — `claims_matrix.md` §I8.13 (11 items). Tiers and v2.23 are CURATOR-PROPOSED.

---

# PASS 1 (kept verbatim below, for the record)

**Agent:** `doc-curator`. **Branch:** `w18/registry-fold`. **Base:** `base/iter8 = 148f3b7`. **Machine:** batmac.
**Window:** `START_UTC.txt` → `END_UTC.txt` in `~/clu/outputs/w18-registry-fold/`.
**Scored runs: NONE. Compute: NONE. Background jobs / PIDs: NONE.** ⛔ No machine-wide kill; tmux untouched.
⛔ `~/clu/evidence/` not written (base md5s re-checked at close, unchanged from arrival).

## 0. WHAT SHIPPED

| file | what | lines |
|---|---|---|
| `negative_results.md` | registry + one header note + the **◈ block**: `N338`–`N344`, fold-in ledger `◈.2`, **awaiting list `◈.3`**, declared not-runs | 5,973 → **6,250** |
| `claims_matrix.md` | matrix, title → **v2.22 (PROPOSED)** (prior title kept verbatim as its tail) + `§I8.10` (`CM-W14-3`, `CM-W17-1`…`CM-W17-4`) + `§I8.11` open flags | 1,624 → **1,731** |
| `FINDINGS.md`, `LOG.md`, `START_UTC.txt`, `END_UTC.txt`, `PIDS.txt` | record | — |

**Base:** `evidence/outputs/w14-refusal-displays-closeout/{negative_results,claims_matrix}.md`. This is the latest pair on batmac, and it is byte-identical to the worktree copy under `w14_refusal_displays_closeout/` (md5: registry `5568a6f2…`, matrix `6b8042dd…`). The repo has no `.claude/` directory, so by the w13 precedent the pair lives as a per-spoke copy in `w18_registry_fold/`.

**Additivity was checked with difflib, not just claimed:**
- Registry: 5,974 lines equal, 277 inserted, **0 replaced, 0 deleted**.
- Matrix: 1,624 lines equal, 107 inserted. The title line is the only replaced line, and the prior title is preserved verbatim as its tail (`endswith` True).
- Each of N338–N344 and each of the five CM rows appears as exactly one heading. Across the registry's 336 `### N<number>` headings there are **zero duplicate numbers**. Nothing was renumbered, so no redirect is owed.

## 1. WHAT WAS ADDED vs WHAT WAS ALREADY PRESENT

| task item | already present? | this pass |
|---|---|---|
| **w14 verb-vs-store (+0.83 vs +0.47)** as a standing caveat | ⛔ **Absent.** No `read-verb`, `0.8299` or `0.4749` in either file | **Added:** `N338` (tier A) plus `CM-W14-3`, a standing caveat with forbidden wordings and a re-opening condition. Numbers come from `w14-read-verb/FINAL-REPORT.md` §5.1 (lookup **+0.8299**, CLU **+0.4749**), §5.2 (controls do not collapse) and §5.3 (lookup **0.9648 / 0.9646 / 0.9808**, which is "keep everything") |
| wave-17 referee must-fix items still open | ⛔ Absent (w17 was never folded) | **Added:** all of M1–M7 are **OPEN at this fold**. `N339` M1 (fit pools **16,384 vs 131,072**; data-matched margin **+3.117…+17.875**, primary `w12b-nr6-and-kstar/KSTAR-LADDER.md` l.66/l.77). `N340` M3 (address book carried, not refit; **1,015,808 of 1,060,864** bytes, `w13-seeds-and-tables/TABLES.md` l.822/l.659; **83.5 %**, `w10-physics-section-evidence/EVIDENCE.md` l.382). `N341` M4 (SISA in the bib but uncited; CDM absent from the bib). `N342` M2. `N344` M5/M6/M7. Matrix rows `CM-W17-1`…`4` |
| claims whose artefact referee-3 could not locate | — | referee-3 found **no unlocated number** apart from the four honest `\TODO`s (`N344` item 4). **Added** one upstream mismatch it surfaced: `N343`, where `w14-read-verb/FINAL-REPORT.md` l.297 gives "8.2e-16" but `TABLES.md` l.360/l.462 and the run JSONs give **7.818e-16** |
| *(curator-added)* other wave-17 declared negatives | — | `N344` items 5–8: rendered-compositions set 2 DECLARED; plots not-drawn; lecun2022path uncited; placeholder count disagrees 26 / 29 / 31 |
| every **w16** config that failed val | — | ⛔ **AWAITING w16 — second pass required** |
| every **w16** scored config that missed its bar | — | ⛔ **AWAITING w16 — second pass required** |
| **w15** declared not-runs | — | ⛔ **AWAITING w15 — second pass required** |

## 2. ⛔⛔ AWAITING w15/w16 — SECOND PASS REQUIRED (exact list in registry `◈.3`)

**Wave 15.** No artefact of this programme's wave 15 exists on batmac:
- `evidence/outputs/` has no `w15-*` directory. `transfer-docs-sync-w15.md` belongs to an earlier campaign.
- `~/clu/outputs/w15-table-fixes/` is **empty**.
- Branch `w15/table-fixes` has **no commits** past `148f3b7`.

The launch says 2 of 5 spokes have landed; they have not landed on this machine. **Missing, per spoke:**
- `crossover-seeds`
- `single-design-seeds` (seeds 1–2 at K\*, plus deletion rows)
- `coverage-control`
- `lace-poe-rows`
- `bar-d-score`
- `maze-deletion-2b`
- `table-fixes`
- `cmaze-t-harness` and `cmaze-o-harness` (audits and val smokes)
- every w15 DECLARED NOT-RUN

**Wave 16 (not started).** Missing:
- every config that failed val, with its numbers and bar
- every scored config that missed its bar, with the failing link
- all CMAZE-T / CMAZE-O rows and declared negatives

⚠ `w17-advisor-report.md` §7 item 3 already quotes a w15 `crossover-seeds` figure. **Its primary artefact is not on batmac, so it was not folded.** Under AMENDMENT-1 it must not reach a draft.

## 3. DECLARED NOT-DONE

- **No w15/w16 fold:** there are no artefacts. Nothing was guessed.
- **No ruling on M1, M3 or M4:** those belong to the Advisor (`w17-advisor-report.md` §7). The curator's wordings are labelled as candidates.
- **No judgement on whether `w18/referee-fixes` closed an item:** that spoke is in flight, and its edits are not evidence yet. The second pass should close items against its FINDINGS.
- **M2's mechanism is not folded:** it exists only in that spoke's in-flight LOG.
- **`w14-read-verb/FINAL-REPORT.md` is not edited:** it is read-only evidence (`N343`).
- **No entries of their own for w13 `encoder-adapt`, w13 `seeds-and-tables`, w12b `nr6-and-kstar` or w14 `request-layer-fix`:** these are outside the named items. The registry cites none of the first three (flag `§I8.11` item 3).
- **No statistic re-derived.** The only operations were quoting and mechanical `grep` counts, which are marked.

## 4. OPEN FLAGS — see `claims_matrix.md` §I8.11 (six items). Tiers and v2.22 are CURATOR-PROPOSED.
