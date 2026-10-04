# w20 spoke/referee4-fixes — APPLY MF-1…MF-8 AND MF-10 FROM REFEREE-v3
**Agent type: paper-writer. Zero compute, zero scored runs.** ⭐ PRE-AUTHORISED; never wait for anyone.
⭐ **Scoped by the Hub 2026-09-23.** `REFEREE-v3.md` returned **4/10** and judged the paper **READY for the Head's editing pass, NOT ready for submission**. It states every must-fix needs **no new scored run**. This spoke applies the ones that are ours.

## Apply in BOTH `DRAFT-v2.tex` and `DRAFT-v2-style.tex`
**MF-1** — the CMAZE-T "store row" gives the store a **different input from the rivals**, the abstract leads with it, and the asymmetry is **not disclosed** in the main text. ⭐ **Disclose it plainly where the row is stated.** ⚠ w19 (`hub/maze-t-symmetric`) is testing symmetric inputs right now; ⛔ **do not pre-empt its result** — disclose the asymmetry, don't predict its resolution.
**MF-2** — the M1/M2 placeholder **states the wrong store fit pool for the ladder** and the draft contradicts itself. Correct it from the artefacts.
**MF-3** — ⭐⭐ **the paper already holds a data-matched, read-path-matched rival and never prints it.** `learned_ref_ownpool`: an MLP at the **130,514-param read-path pin**, fitted on the **same 32,768 rows as the store's content**, **3 seeds on every rung** (`w13-seeds-and-tables/TABLES.md` l.703–710; `w12-composition-ladder/AMENDMENT-1.md` l.29). Print it in §3.1/§3.2 with its source; put the **1× margin (+3.117 minimum)** in the main text as the conservative comparison; fix the abstract to name the fit pools. ⚠ On the 109-corner leg it is fitted on **32,768 rows of the value hold-out pool, NOT 131,072** — say so.
**MF-4** — **N359 exposure:** numbers the paper itself calls **unmeasured** are still printed and argued from. The w8 scorer loaded a rival from the wrong feature space, so **CPU BAR D is UNMEASURED**, ⛔ not "VOID because the baseline scores 0".
**MF-5** — text still written for rows that have since landed, now **contradicting the filled results**.
**MF-6** — ⭐ **three bannered placeholder figures render in the PDF and are not counted among the 11 placeholders.** Either count them or resolve them; ⛔ a placeholder that does not appear in the placeholder index is exactly the silently-closed gap the referee hunts.
**MF-7** — LACE and PoE sit **inside the headline table while performing at control level**. Move or mark them; carry their standing scope (LACE/PoE **as instantiated in this feature space at their own declared budgets**, ⛔ not a refutation of LACE in general).
**MF-8** — the abstract, contribution 1 and the leaving note **overstate the headline** relative to the paper's own disclosures. State the comparison §3.1 endorses: against the **strongest rival of either set**, plus MF-3's data-matched rows. ⭐ **No new number is needed.**
**MF-10** — **K\* in the abstract reads as a clean selection.** The main text is honest; the abstract is not. Make the abstract match: the frozen rule **cannot separate the plateau K = 5…32** on three seeds (seed sd > 0.02); the finding is **the plateau itself**.
⭐ Also fix **SF-7**: contribution 3's *"survives rivals trained ten times longer"* drops two qualifiers — that check is on **validation** (A1); the **scored** 10× replication is **C-512**.
⭐ And the **M7(e) regression**: the w15/w16 fill-in reintroduced pipeline phrasing ("scored run 3 of 3", "runs 1–3"). Strip it again.

## ⛔⛔ NOT YOURS — leave every one of these exactly as it is
- **MF-9 / M3** — deletion wording. **The Head rewrites it.**
- **MF-11** — the page trim (≈18 pp against a 9 pp desk-reject limit). ⛔ **Report the count; do not cut.** The cut forces content decisions that are the Head's.
- **MF-12** — the remaining venue items, including the AI-use statement.
- ⛔ The 11 `\PLACEHOLDER{}` awaiting w19/w20. ⛔ Do not fill or delete them.

## Invariants
⭐ Both drafts compile with **0 undefined citations** — `pdflatex → bibtex` (`BIBINPUTS`/`BSTINPUTS` = `w17-paper/`) `→ pdflatex ×3`, and **check the count**; bibtex silently failed for a long stretch of this campaign. ⭐ Four style diffs EMPTY. ⭐ Every number carries a LaTeX comment naming its artefact.

## Acceptance
`REFEREE4-FIXES.md`: MF-1…MF-8, MF-10, SF-7 and the M7(e) regression each with **before → after** and location; MF-9/MF-11/MF-12 listed as **the Head's, untouched**; the page count reported; four diffs EMPTY; both compile logs with the undefined-citation count shown.

## What would prove this wrong
⭐ If disclosing MF-1's input asymmetry makes the CMAZE-T headline unsupportable **before w19 reports**, say so — ⛔ do not soften the disclosure to protect the claim.
