# w22 spoke/registry-fold PASS 6 — FINDINGS (three units folded: rel-newseeds, ladder-rivals-32k, figures3)

**Agent:** `doc-curator`. **Branch:** `w22/registry6`. **Machine:** batmac. **Brief:** `~/clu/tasks/w22/registry6.md`.
**Scored runs: NONE. Compute: none beyond a read-only arithmetic script over evidence JSONs. Background jobs: NONE. Sub-agents: NONE.**
- No ledger was written.
- No draft or `.tex` was read for editing or edited, and the `.tex` lock was not taken.
- `~/clu/evidence/` was not written. tmux was not touched.

**Base:** pass 5 in `~/clu/outputs/w21-registry-fold/` (registry 8,042 lines, md5 `f9259d49…`; matrix 2,307 lines, md5 `c9052456…`). Both equal the
`w21/registry-fold` commit `f1ca948`. Pristine copies are kept in `curator/base_pass5/`.

## 0. What shipped

| file | pass 5 → pass 6 | what pass 6 adds |
|---|---|---|
| `negative_results.md` | 8,042 → **8,415** lines | **`N391`–`N397`** (7 entries); `◈.18` (what was read), `◈.19` (entries), `◈.20` (fold ledger and not-runs); a header note; `⟲ PASS 6` lines at `N372`, `N373`, `N377`, `N381`, `N383`, `N384`, `N386`, `N388`, `◈.17` |
| `claims_matrix.md` | 2,307 → **2,439** lines, **v2.27 (PROPOSED)** | **`§I8.20`**: `CM-W21-4`, `CM-W21-5`, `CM-W21-6`, `CM-W22-1`, `CM-W17-9`. **`§I8.21`**: 7 open flags. `⟲ PASS 6` lines at `CM-W20-1`, `CM-W20-2`, `CM-W20-4`, `CM-W21-2`, `CM-W17-8`, `§I8.19` flag 8. The v2.27 title is **inserted** above the v2.26 title, which is kept verbatim |
| `curator/` | new | `curator_arithmetic_p6.py/.json` (every derived number); `build_pass6.py` (insert-only build); `verify_pass6.py/.json`; `audit_pass6.py/.json`; `rounding_check_p6.py/.json`; `reg_block6.md`, `cm_block6.md`; `src_w22_figures3/` (read-only copies of the figures3 JSONs from branch `w22/figures3` @ `cb5e0ac`); `base_pass5/` |
| `NEEDS-ARTEFACTS.json` | new | three items (nothing substituted) |

The new files are written **in place** in `~/clu/outputs/w21-registry-fold/`, as the brief asks, and copied to `~/clu/outputs/w22-registry-fold6/`.
The base md5 was re-checked immediately before install. Pass 5's own FINDINGS, LOG, START/END and bundle in that folder are untouched.

**Acceptance, proved by script** (`curator/verify_pass6.json`, `verify_pass6_installed.json`, `audit_pass6.json`, `rounding_check_p6.json`):
- **difflib against pass 5:** registry **373 lines inserted, 0 replaced, 0 deleted**; matrix **132 inserted, 0 replaced, 0 deleted**. The base files are
  an exact subsequence of the new ones. The installed copy was re-verified.
- **Headings:** 389 `### N` headings, **0 duplicates**, and N391–N397 each appear exactly once. 67 `CM-` rows, 0 duplicates. No duplicate `◈.` / `§I8.`
  section.
- **Number audit:** every decimal and every token of 4+ digits in the inserted lines (586 registry, 100 matrix) appears verbatim in a source file, not
  glued to a preceding digit. The strict pass checks **each inserted hunk and each entry** against **only the files it names**; a cross-reference such as
  `N393` inherits its entry's sources. **0 failures.** The rounding check finds 0 tokens that occur only as the prefix of a longer number.
  - Two of the curator's own roundings failed the first strict run and were replaced by exact CA6 values: "+2.22" became 2.2194444444444414 and "under
    0.9" became 0.8850152905198776.

## 1. The results, by unit (the artefacts' numbers)

**⚠⚠⚠ The one that bears hardest on the paper: `N394` / `CM-W21-6`, single-fit RNG-draw variance.**
- The ladder's X1 check held rows (32,768), steps (2,560), width, parameters, trainer and schedule identical. A replay with the reference's own keys
  reproduces it per corner on 15/15 legs. ⇒ **only the PRNG draw differs**.
- Per-seed paired Δ **−2.15 … +4.13 novel/corner**; the per-corner CI95 **excludes 0 on 12 of 15 legs**. RMS 2.615173848764028, i.e. about
  1.8492071624427666 per fit if the draws were independent; they are not quite, because the sign follows the seed index.
- ⇒ **Per-corner bootstrap CIs, which every margin in the paper prints, do not contain refit variance.**
  - Shapes headline +7.03: seed 2's +5.02 clears the largest swing by only 0.8850152905198776.
  - Ladder: two per-seed margins sit below it (+3.85 in B, +3.25 in C-sm, both n = 2 seed 2).
- ⭐ **The signs and the 3-seed means are not threatened.**
- ⚠ **Scope:** mlp at 2,560 steps (under-trained), Shapes only. Refit variance at the carried budgets and on CMAZE-T is **unmeasured**
  (`§I8.21` flag 1; `NEEDS-ARTEFACTS.json`).

**w21 `rel-newseeds`** (`N391`, `N392`, `CM-W21-4`):
- ⭐ CLU-REL-A10 BOTH TEST, six seeds: **0.925** against A1's 0.677; at the trained-arm ceiling on every seed.
  - Seed 4 0.533 → 0.900, seed 1 0.566 → 0.894, seed 5 0.682 → 0.939.
  - No WORSE@80k on seeds 3–5; the best rival there (the transformer) is unflagged.
- ⛔⛔ **Three measurements say the store is not what carries the margin:**
  - `+rel` leaves the CLU +0.031 on average (−0.007 / +0.024 / +0.165 / +0.006 / +0.000 / +0.000).
  - Seed 4 TEST: `random_atoms` = CLU = 0.9004 on **16/16 tuples** (re-counted by the curator from the zero-fit replay).
  - Seed 5: CLU 0.939 = best matched rival.
- ⭐ **The claim:** colour-relational request coding, an **input representation**.
- ⚠ **Seed 2 is the exception** (+0.165 beyond the coding; share recovered 0.45).
- ⛔ The value row is **below** the best matched rival on all six seeds. Rollout validity is 0.267 / 0.156 / 0.000.

**w21 `ladder-rivals-32k`** (`N393`, `CM-W21-5`):
- ⭐ MF-2 is real on paper and immaterial when measured. Mean over rungs: A +17.53, B +9.97, **C +10.28**, C-sm +10.75.
- C − B per rung is −1.21 … +1.45, in both directions. C-sm − B reaches +2.2194444444444414.
- Every CI lower bound in B / C / C-sm is above 0 (lowest +1.98). Pool identity 15/15.
- ⭐⭐ At matched steps the pool barely moves the rivals (−3.5 … +3.9, mixed sign); the step budget does. This independently confirms w20's `N373`.
- ⚠ cont_mlp is not converged (up to +19.33). All 60 selections sit at the grid edge. No store was re-scored.

**w22 `figures3`** (`N395`, `CM-W22-1`):
- ⛔ The CMAZE-T panels drew the state while the judge scores the fact vector. **33,412 of 92,628** opposite-mark pairs (A1) were pictorially
  identical: 3,740 in the legal-bit-only class, 16,283 with pocket-next as the reason. For CLU-REL-A10 it was 154,671 of 373,583.
- ⭐ A fact strip brings every count to **0**. ⛔ **No number was affected** (132/132 REL arm scores float-`==` the run JSONs).
- ⚠ Also folded:
  - `c1` is now red in the task figure as in the panels (7/7 discs pixel-identical);
  - the panel palette had **failed** the validator since pass 1 (dE 13.8, CVD 7.2) and is fixed;
  - the 32,768-row ladder figure slot is still `not_landed`.

## 2. ⚠ Where a paraphrase disagreed with the artefact (the artefact wins; `N396`)
1. **Brief: X1 CIs exclude 0 on "10/15".** The artefact says **12/15**. The brief copied a ladder LOG line (18:10Z) that the LOG itself later corrects.
2. **Brief: X1 "±4.1".** The artefact says **−2.15 … +4.13**. The negative side reaches only 2.15.
3. **Brief: REL "fixes A1's two weak seeds (1 and 5)".** A1's hard seeds are **1 and 4**, and its lowest is **4**. There REL gains most (+0.368) and
   **ties `random_atoms`**. Seed 5 is A1's *losing* seed.
4. **Brief: "NOT a store/memory result" as a blanket statement.** It holds at the six-seed mean and on 4 of 6 seeds. **Seed 2 keeps +0.165** beyond the
   coding.
5. **figures3 FINDINGS: palette "ΔE 13.7 / CVD 7.1".** The validator output says **13.8 / 7.2**.
6. **figures3 FINDINGS: the Head's defect file "does not exist".** It **exists** at `evidence/outputs/FIGURE-DEFECT-cmazet-legality.md`, with an mtime
   before figures3 started.
7. **Hub defect note: "88,888 visibly different … figure is fine" and "the pocket is not the problem".** Pixel-level, **59216** pairs are visibly
   different. The pocket statement holds for item 482 only; 16,283 pairs have pocket-next as the reason.
8. **figures3 FINDINGS' cascade mixes levels.** 33,412 is pixel-level; 18,035 and 1,752 are state-level, where the total is 30,930.
9. **LOG timestamps labelled `Z` are non-monotone** in the ladder and figures3 LOGs, and some postdate END. They are consistent with local time.
   **No number depends on them.**

## 3. Omit nothing (the ban)
- **No PREREG** in any of the three units. **One pre-declared bar:** the ladder's X1 bar (LOG 13:58Z). Its verdict *"REPRODUCES"* is quoted as record
  only (`N394`).
- **Folded although the brief did not name them:** the REL value row, the seed-0 `+rel` ceiling (0.915 > 0.908), the `+rel` shares on seeds 0–2,
  the HIGH flags on seeds 4–5, the per-tuple near-ties of controls (s3 15/18, s5 15/16), the seed-3 tuples a rival reaches and the CLU does not,
  rollout, and the REL figure counts. All are in `N391`, `N392` and `N395`.
- Every unit's declared not-runs are listed with their status in `N397`. **Left unreported:** nothing.

## 4. What did NOT run
- No scored run, no compute beyond `curator_arithmetic_p6.py` (JSON reads, means, differences, counts; it fits nothing and scores nothing), no GPU, no
  sub-agent.
- **Not done:**
  - no PRNG-replicate fits (the measurement `N394` says would settle refit variance): zero compute;
  - no persisted object verified: only manifests and JSONs are on batmac, and the spokes' replays are quoted;
  - no figure re-rendered or pixel-counted: one page crop was viewed, and the ambiguity counts are the spoke's, not recomputed.
- **No label computed and no ruling made** on CMAZE-T's paper sentence, the refit-variance caveat, the seed-4 store gate or the ladder figure slot.
  `§I8.21` lists them for the Head and the Advisor.
- No draft or `.tex` was read or edited. The draft state that figures3 reports (placements, `\PLACEHOLDER` 4) is **not** verified here (`§I8.21`
  flag 6).
- ⭐ **No seventh pass is required** for these three units.
