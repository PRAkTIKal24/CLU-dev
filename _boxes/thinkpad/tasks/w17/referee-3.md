# w17 spoke/referee-3 — REFEREE IT LIKE THE REVIEWER WE WILL ACTUALLY DRAW
**Read `COMMON.md` first.** Slug `referee-3`. Agent type: paper-referee.
⭐ **Gate: start when `DRAFT-v2.tex` and `VENUE.md` exist and draft-v2 has written `END_UTC.txt`.**
⭐ Referee **the un-paraphrased draft** (`DRAFT-v2.tex`), not the style version.

## The one job
`~/clu/outputs/w17-paper/REFEREE-v2.md`: an itemised referee report against **the evidence base** and **VENUE.md**, with a **simulated accept/reject verdict**.

## What to hunt
- **Every claim cross-checked against its artefact** in `~/clu/evidence/outputs/`. A claim whose number you cannot locate is a **must-fix**.
- **Overclaims.** Especially: a **verb** result written as a **store** result; a store row claimed where the charter-C controls did not collapse; the lookup floor described as a rival.
- **Missing baselines.** Is every rival matched naive-vs-naive? Is the lookup floor printed?
- **Scope creep** past what the evidence supports.
- **Byte figures without their convention.** **Seed counts missing from tables.**
- **Two maze environments in one table** (CMAZE-1 / 2 / 2b / T).
- **The ladder's n ≥ 3 training-data caveat** — present and correctly worded?
- **Venue compliance:** anonymity, reproducibility statement, ethics statement, LLM-usage policy, citation style, appendix rules.
- ⭐ The **placeholders** for wave 16: are they honestly marked, and does the paper still stand if they land as **negatives**? ⛔ A paper that only works if wave 16 wins is a paper that will be rejected.

## Output format
Itemised, triaged **must-fix / should-fix / nice-to-have**, each with the file and location, the specific defect, and the minimal change that fixes it. Then a **simulated verdict** with the score and the two-paragraph rationale a real reviewer would write.

## ⭐ Be hostile but fair
⛔ Do not soften because it is our own paper. ⛔ Do not invent defects to appear rigorous. Every item must be actionable.

## Files you own
`REFEREE-v2.md`. ⛔ You do not edit the draft.
