# w18 spoke/registry-fold — FINDINGS (first pass; ⛔ SECOND PASS REQUIRED)

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

**Base:** `evidence/outputs/w14-refusal-displays-closeout/{negative_results,claims_matrix}.md`. This is the latest pair on batmac, and it is byte-identical to the worktree copy under `w14_refusal_displays_closeout/` (md5 `6b8042dd…` / `5568a6f2…`). The repo has no `.claude/` directory, so by the w13 precedent the pair lives as a per-spoke copy in `w18_registry_fold/`.

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
