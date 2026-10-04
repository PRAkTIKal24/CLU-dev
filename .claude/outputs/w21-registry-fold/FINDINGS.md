# w21 spoke/registry-fold — FINDINGS (PASS 5: waves 20 and 21 folded)

**Agent:** `doc-curator`. **Branch:** `w21/registry-fold`. **Machine:** batmac. **Brief:** `~/clu/tasks/w21/registry-fold.md`.
**Scored runs: NONE. Compute: none beyond a read-only arithmetic script over evidence JSONs. Background jobs: NONE. Sub-agents: NONE.**
No ledger was written, no draft or `.tex` was read for editing or edited, the `.tex` lock was not taken, `~/clu/evidence/` was not written, tmux was not touched.
Base: `~/clu/outputs/w18-registry-fold/` pass 4 (registry md5 `c9e69d4e5c2f20e7624ab4a7fda0c61e`, matrix md5 `b6a31de6555caf6d3220db5ea09c6986`); unchanged.

## 0. What shipped

| file | pass 4 → pass 5 | what pass 5 adds |
|---|---|---|
| `negative_results.md` | 7,466 → **8,042** lines | **`N372`–`N390`** (19 entries); `◈.15` (what was read), `◈.16` (entries), `◈.17` (fold ledger + not-runs); a header note; `⟲ PASS 5` lines at `N339`, `N340`, `N350`, `N355`, `N360`, `N366`, `N367`, `N368`, `N369`, `N371`, `◈.14` |
| `claims_matrix.md` | 2,131 → **2,307** lines, **v2.26 (PROPOSED)** | **`§I8.18`**: `CM-W20-1`…`CM-W20-4`, `CM-W21-1`…`CM-W21-3`, `CM-W19-5`, `CM-W17-8`. **`§I8.19`**: 11 open flags. `⟲ PASS 5` lines at `CM-W17-1`, `CM-W17-2`, `CM-W15-3`, `CM-W19-3`, `CM-W17-7`, `§I8.17` flag 8. The v2.26 title is **inserted** above the v2.25 title, which is kept verbatim |
| `curator/` | new | `curator_arithmetic.py/.json` (every derived number), `build_pass5.py` (insert-only build), `verify_pass5.py/.json`, `audit_pass5.py/.json` |
| `NEEDS-ARTEFACTS.json` | new | four items batmac lacks (nothing substituted) |

**Acceptance, proved by script** (`curator/verify_pass5.json`, `curator/audit_pass5.json`):
- **difflib against pass 4: registry 576 lines inserted, 0 replaced, 0 deleted; matrix 176 inserted, 0 replaced, 0 deleted.** The base files are an exact subsequence of the new ones.
- **382 `### N` headings, 0 duplicates**; each of N372–N390 appears exactly once. 62 `CM-` rows, 0 duplicates.
- **Number audit:** every decimal and every token of 4+ digits in the inserted lines (757 registry / 132 matrix tokens) appears verbatim in a source file. A stricter pass checks each registry entry against **only the files that entry names**. It passes, with one exception: the w21 reading-key paragraph sits inside `N381`'s split chunk, so its tokens `8,000` and `80,000` are checked against the w21 files instead. Identifiers (§, N, CM-, l., A4.1-style amendment ids) are excluded.

## 1. The results, by unit (the artefacts' numbers)

**w20 `shapes-data-matched`** (3 of 3 spent, target exhausted):
- ⭐⭐ Data-matched margin **+7.03** (+7.49 / +8.60 / +5.02); **+6.54** vs best of all fits; the draft's +39.7…+49.1 ⇒ **~6× shrink, no collapse** (lowest CI +3.73). `N372`, `CM-W20-1`.
- ⭐⭐ **M1's cause was SCHEDULE, not data**: MLP on the same 16,384 rows = 38.6 at 20,480 steps and 5.5 at 1,280; the store goes 47.28 → 48.01 from 16k to 131k rows. `N373`, `CM-W20-2`.
- ⭐⭐ **M3 measured**: under a refit book, `red ∧ sphere` costs survivors −6.13 / −6.43 / −6.30; random-1 % −0.02 / −0.11 / −0.01; (a)/(b) exact within their bars. `N374`. The book's exact statistic is 967304800 bytes (the fit data). `N375`.
- Matched rival deletion, the ladder (+5.13; +10.30…+12.55; K = 5 n = 4/5 not claimable), LACE/PoE at 131k, the five defects (incl. VOIDED seed-1/2 rung stores), all 8 not-runs. `N376`–`N380`.

**w21**:
- `words-full-test`: **LOSS**, Δ̄ −0.0175, σ 0.0140, 0/3; COMPARABLE under two other spreads (now folded); VAL repro bit-identical. `N382`.
- `maze-t-clu-improve`: **WIN** 0.929 vs 0.685, 3/3 — ⛔ ~0.18 of 0.24 is the coding (`+rel` 0.862); CLU keeps +0.066, 2/3; **zero on WORDS-FULL VAL** (0.941 vs 0.937); 68 candidates / 57 passers (ledger says 69). `N383`.
- `maze-t-seed4`: seed 5 (easiest) **−0.185**, two class-A tuples at 0.000 that 8 arms (one a control) solve; seed 3 inflated; rollout 0.000 ×6; cut 0.641 with sensitivity. `N384`.
- ⭐⭐ CLU-REL-A10 seed-4 VAL: `random_atoms` = CLU = 0.8994252873563219, **and `landscape_removed` = 0.2413793103448276** (not in the spoke's prose). `N386`.
- ⭐ The matched-float rule: 4 hidden units on the 954-dim input; seed-3 MLP loss 0.180 → 0.205 with 10× steps; two more non-converged lstm-8k fits in the training table that the spoke did not flag. `N387`.

## 2. ⚠ Where a paraphrase disagreed with the artefact (the artefact wins)

1. **Brief: "(a)/(b) exact (≤1e-8)".** Max (a) 5.383044481277466e-07, max (b2) 7.812795388417726e-08. Both are exact within the declared bars (1e-5, 1e-3), not ≤ 1e-8. `N381`.1
2. **Brief / FINDINGS / Hub: "rivals lose < 1 where the store loses 6".** This holds for the family means of mlp, lstm and transformer only. Per seed it fails: mlp s2 −2.954, cont_mlp s2 −9.569. `N381`.2
3. **w20 Hub report: LACE "against the store's 64.92 on the same corners".** 64.92 is the VAL store. `N381`.3
4. **w20 `headline.rule` "P1-carried".** At seed 2 the headline uses max(DECLARED, TENX); the literal P1-carried mean is 7.342507645259938. The printed +7.03 is the conservative number. `N381`.4
5. **"Item deletion moves nothing".** Seed 1 is −0.110 with a CI that excludes 0. `N381`.5
6. **"−32.1" quoted as replicated.** It is seed 0 only, and it is in no staged JSON. `N381`.6
7. **w18 Hub report MF-2 "CLOSED".** It quotes PREREG A2.1, which is about `learned_ref_ownpool`, not the ladder store. The outcome is harmless. `N381`.7
8. **Brief: "6-seed mean margin +0.046 (was +0.099 on 3)".** This compares two different estimands:
   - per-seed-best: +0.065 on 3 seeds → +0.046 on 6;
   - R\*-by-mean: +0.0988 on 3 seeds → +0.128 on 6, ahead 6/6 (CA).

   "Does not replicate" holds only for the per-seed-best estimand. ⛔ No label was computed. `N385`.
9. *(Found, not a disagreement)* The Hub's **w19 report now relabels BOTH TEST as WIN** and OBS as COMPARABLE under the Head's rule. The curator re-derived every σ. This supersedes the verdict column of `N366` / `N369` / `CM-W19-3`, annotated with ⟲. `N389`.
10. *(Process)* The seed-4 AMENDMENT-1 named "the finding to support" before any number, and the spoke measured it refuted. `N388`.

## 3. Omit nothing (the ban) — what was withheld and is now folded
- **`learned_1x` (w20 did not print it)**: the existing w12 value at the shipped leg is 48.8991 / 49.5413 / 49.4404. That is above w20's K\* store means on every seed, but the two come from different instruments, are unpaired, and are not a margin. On the ladder, K\* beats the same opponent on every seed (worst 3.117). `N380`.
- **Other withheld items:**
  - the w20 VAL 16k comparators (*informational only*);
  - the σ alternatives in words-full (*printed, never the label*);
  - the `+rel` and 10×-float diagnostics (*never a bar*);
  - the atoms-6 cleaner-gate config;
  - the seed-4 `landscape_removed` value;
  - the unflagged lstm losses.

  All are at `N380`, `N382`, `N383`, `N386`, `N387` and `N390`.
- **Left unreported, with reason:** smoke runs (plumbing checks). `N390`.

## 4. What did NOT run
- No scored run, no compute beyond `curator_arithmetic.py` (JSON reads, means, sds, a file count), no GPU, no sub-agent.
- **Not re-derived:**
  - the −32.1 breakdown, and any seed-1/2 breakdown (the corner→content map is not staged);
  - the ~3.2M `learned_1x` (no number on batmac);
  - any persisted-object hash.

  All are listed in `NEEDS-ARTEFACTS.json`.
- **No label computed, no ruling made** on M1, M3, the estimand, `learned_1x` placement or the K = 5 ladder rows. `§I8.19` lists them for the Head and the Advisor.
- No draft or `.tex` was read or edited. Live-draft line numbers for the Shapes headline were **not** checked at this pass (`§I8.19` flag 1).
- ⭐ **No sixth pass is required** for w20 or for the four w21 units in the brief.
