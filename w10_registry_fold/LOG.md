# w10 spoke/registry-fold — LOG

**Agent:** doc-curator. **Branch:** `w10/registry-fold`. **Worktree:** `C:/chlu/worktrees/CHLU-w10-registry-fold`.
**Start:** 2026-09-21T08:26:07Z (`START_UTC.txt`) → **End:** 2026-09-21T08:33:34Z (`END_UTC.txt`), 7 m 27 s.
⚠ All bullet timestamps below are **UTC**, matching those two files (the box clock is UTC+2).
**Scored runs:** ⛔ NONE — this spoke measures nothing; it folds other spokes' measurements.

## Ground rules I am holding myself to
- Evidence base = `/c/chlu/evidence/outputs/` + `/c/chlu/evidence/claims_matrix.md` + `/c/chlu/evidence/negative_results.md`.
  `$CHLU_CLAUDE_ROOT` NOT read (frozen/stale) — confirmed not opened at any point in this session.
- Every row carries **a number and the path that holds it**. A row I cannot source says so in the row.
- Writes: `/c/chlu/outputs/w10-registry-fold/` and my worktree only.

## Timeline

- `08:26` START_UTC written.
- `08:26` Surveyed the registries. Current state on disk:
  - `claims_matrix.md` — 643 lines, **v2.17**, last division = "THE C3 DIVISION" (rows `CM-C3-1…5`, opened 2026-08-13).
    ⛔ **It carries NOTHING from the gen pivot or from iter8 waves 3–9** — grep for `iter8`/`ICLR`/`downdate`/`CMAZE` returns nothing.
  - `negative_results.md` — 3794 lines, main series runs to **N315**, plus a `⟲ 2026-09-11 — GEN PIVOT, PHASE 1` addendum at line 3787.
  - ⚠ **Collision found on arrival:** the 2026-09-11 gen-pivot addendum numbered its two entries **N89 / N90**, and
    `N87` / `N88` already exist in the main series (`negative_results.md:3780-3781`) — i.e. the exact hazard **N315**
    (`negative_results.md:3376`) was filed about. My fold therefore uses a **separate, greppable namespace** rather than
    extending either series. See LOG entry on numbering below.
- Surveyed `_thinkpad/`: wave dirs `w3 w4 w5 w6 w8 w9` (no `w7` dir — see not-run/absence note), plus
  `_thinkpad/outputs/hub-*` (the wave-1/2 hub spokes) and `_thinkpad/scratch/`.
- `08:27` ⚖ **SCOPE CORRECTION FROM THE HUB, mid-session:** `/c/chlu/evidence/` is **READ-ONLY**.
  The task file's "Output: updates to the claims matrix and `/c/chlu/evidence/negative_results.md`" is executed as
  **updated whole copies written into `/c/chlu/outputs/w10-registry-fold/`**, which the Hub folds back.
  ⛔ No byte of `/c/chlu/evidence/` is written by this spoke. Deliverables:
  `claims_matrix.md` (v2.18 proposed) and `negative_results.md` (with the dated ⟲ addendum), both in my output dir.

- `08:27` **EVIDENCE READ, in this order** (⛔ nothing from `$CHLU_CLAUDE_ROOT`):
  `outputs/w{3,4,5,6,8}-advisor-report.md` · `outputs/w4-w6-advisor-ruling.md` ·
  `outputs/advisor-wave-plan-w{4-w7,8-w10}.md` · `_thinkpad/w8/w8-downdate/{LOG.md,DOWNDATE-READY.json}` ·
  `_thinkpad/w8/w8-single-design/LOG.md` · `_thinkpad/w8/w8-encoder-loop/LOG.md` ·
  `_thinkpad/w8/w8-novelty/{NOVELTY.md,LOG.md}` (⭐ **both**, per the task's READ BOTH rule) ·
  `_thinkpad/w8/w8-env-scout/ENV.md` · `_thinkpad/w9/w9-maze/{LOG.md,LEDGER.md}` ·
  `_thinkpad/w9/w9-harness/RESULTS.md`.
- `08:30` Wrote the **ITER8 FOLD** into `negative_results.md`: **21 entries `N-I8-1…21`** + a **17-row
  DECLARED NOT-RUN table** + 6 provenance flags. 3794 → **4739 lines**.
- `08:31` Wrote the **ICLR-LONG DIVISION** into `claims_matrix.md`: **§I8.1** (32 never-quotes),
  **§I8.2** (rows `CM-I8-1…13`), **§I8.3** (6 open flags); title bumped to **v2.18 (PROPOSED)** with the
  entire prior title line preserved verbatim as its tail. 643 → **1039 lines**.
- `08:32` **ADDITIVE-ONLY VERIFIED MECHANICALLY, not asserted:**
  - `diff` of `claims_matrix.md` lines **2–643** vs the evidence copy → **empty**.
  - Line 1: `endswith(old_tail)` → **True** (12,031 → 14,099 chars; only a prefix was added).
  - `diff` of `negative_results.md` lines **1–3794** vs the evidence copy → **empty**.
  - `/c/chlu/evidence/` md5s **unchanged** from arrival: `39be4b5f…` / `c3597411…`. ⛔ Not written.
- `08:32` ⭐ **THE ONE ITEM THE EVIDENCE DOES NOT SUPPORT AS COMMISSIONED** — written into the row, not
  paraphrased away: the task's *"1e-16…1e-14 on Shapes3D (9/9 seeds, two independent implementations)"*.
  **"9/9" is 3 configs × 3 seeds in ONE implementation**; the peer implementation's own band is
  **2.842e-14…1.126e-09**; and A1 **FAILED on test seed 2 at 32.90 ulp** before a declared, un-moved
  variance-share cut `tau = 4.144858e-09`. Recorded at `N-I8-1` §A–§B and `CM-I8-1`(b),(e)(i).
  ⭐ The CMAZE leg IS in the 1e-16 band — **projector exactly 0.000e+00, three seeds, replicated with zero
  numeric differences**.
- `08:33` `FINDINGS.md` written. Deliverables copied into the worktree under
  `projects/w10-registry-fold/`, committed on `w10/registry-fold`, bundled and verified.

## Declared NOT-DONE by this spoke (⛔ never a silent omission)
1. ⛔ **The `N89`/`N90` duplicate was NOT renumbered.** It predates this fold, `/c/chlu/evidence/` is
   read-only to me, and renumbering is the Hub's call. **Flagged as `N-I8-21`(a).**
2. ⛔ **No claims-matrix row was written for the `⟲ 2026-09-11` gen-pivot negatives.** Outside the task's
   item list. **Flagged in §I8.3(2).**
3. ⛔ **No statistic was re-derived.** Every number is quoted from the agent's own artefact with its path.
4. ⛔ **No version was confirmed** — v2.18 is PROPOSED, and v2.17/v2.16/v2.15 are left as they stood.
5. ⛔ **No scored run** — the task declares the budget as none.
