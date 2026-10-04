# w17 spoke/style-paraphrase — THE SAME PAPER IN THE HEAD'S VOICE
**Read `COMMON.md` first.** Slug `style-paraphrase`. Agent type: paper-writer.
⭐ **Gate: start when `~/clu/outputs/w17-paper/DRAFT-v2.tex` exists AND `~/clu/outputs/w17-draft-v2/END_UTC.txt` exists.** ⛔ Do not start on a half-written draft.

## The one job
A SECOND `.tex` — `~/clu/outputs/w17-paper/DRAFT-v2-style.tex` — **same content, paraphrased sentence by sentence** under `~/clu/ship/PJ_Writing_Style_Context.pdf`.

## The style rules to apply
- **ABT (And / But / Therefore)** at each section opening.
- **Macro-to-micro** flow.
- **Objective tone. No hyperbole** — magnitude words only when the numbers support them.
- **Signposting.**
- **Scoping bullets** when a framework is introduced.
- **"We"** for our actions; **passive** for established facts.
- **LaTeX for all symbols.**
- **Bold only for new core concepts.**

⭐ **Apply the RULES only. ⛔ Ignore the thesis content and the HEP persona** in that PDF — they are not this paper.

## ⛔⛔ THE ABSOLUTE CONSTRAINT
**Paraphrase only. NO scientific content added, removed or reordered.**
- ⛔ Not one number changed, dropped or rounded differently.
- ⛔ Not one claim strengthened or weakened.
- ⛔ Not one section, table, figure or placeholder moved.
- ⛔ Every `\PLACEHOLDER{}`, every artefact comment and every `\TODO` survives verbatim.

## The self-check you must run before finishing
Write `STYLE-DIFF-CHECK.md` proving:
1. **Every number** in `DRAFT-v2.tex` appears in `DRAFT-v2-style.tex` — extract both sets mechanically and diff them. **The diff must be empty.**
2. **Every `\PLACEHOLDER{}` key** appears in both. Empty diff.
3. **Every artefact-path comment** appears in both. Empty diff.
4. Section order identical — list both.
⛔ If any diff is non-empty, **fix it before writing `END_UTC.txt`** and record what drifted.

## Acceptance criterion
`DRAFT-v2-style.tex` + `STYLE-DIFF-CHECK.md` with all four diffs empty and shown.

## Files you own
`DRAFT-v2-style.tex`, `STYLE-DIFF-CHECK.md`. ⛔ You NEVER edit `DRAFT-v2.tex`.
