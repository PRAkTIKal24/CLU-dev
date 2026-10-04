# w22 `spoke/paper-visuals` — BRIEF (batmac)
**Wave-plan §7. ZERO compute, ZERO scored runs. ⛔ Never write to a ledger. ⛔ Never edit a `.tex`.**

## THE ONE JOB
**`~/clu/outputs/paper-visuals/` must contain EVERY figure and panel the drafts reference — PNG *and* PDF — with a `README.md` INDEX the Head can read without opening anything else.**

⚠ The folder already exists (built by `w22/figures-redraw2`) and its files are correct. ⛔ **What is MISSING is the index §7 actually asked for.** The current README lists SHA-256s and defers "what it shows" to `FIGURES.md`. **That is not the deliverable.**

## THE INDEX — one line per file, four fields, no exceptions
| field | meaning |
|---|---|
| **what it shows** | one plain sentence, no programme jargon. A reader who has not seen the paper must understand it. |
| **table/section it serves** | the `\label` or section it appears in, e.g. `fig:ladder` → §3.2. ⛔ If a file is referenced by **no** draft, say so explicitly — that is useful, not embarrassing. |
| **data source** | the artefact path the numbers came from. |
| **last-build hash** | the SHA-256 in the installed build manifest. |

⭐ Derive **every** field from artefacts: `\includegraphics`/`\label` greps over both drafts for placement, `PROVENANCE.json` + `FIGURES.md` for sources, `SHA256-w22-build*.txt` for hashes. ⛔ **Do not invent a description from a filename** — open the figure or its build code.

## ALSO REQUIRED
1. **Completeness, proven both ways.** Every `\includegraphics` target in **either** draft has a file here (PNG **and** PDF); and every file here is either referenced or flagged `UNREFERENCED`. ⭐ Print both counts.
2. ⭐ **Rebuild check:** run the existing `build.sh` and confirm the rebuilt figures are **byte-identical** to what is installed. ⛔ If any differs, **report it — do not overwrite**; a figure that will not rebuild identically is a defect the Head must see.
3. **PNG + PDF for every figure.** If a PDF exists without its PNG (or vice versa), generate the missing one **from the same build code**, ⛔ never by converting the other.
4. Keep `PROVENANCE.json`, `FIGURES.md` and the SHA manifests in the folder and referenced from the README.

## ⭐ CONTEXT THE INDEX SHOULD CARRY (verify each before writing it)
- `fig_headline_steps` is the **steps-not-data** result: the MLP at 20,480 steps is within ~1 tuple/corner of itself at 16,384 / 32,768 / 131,072 rows.
- `fig_ladder` has three rival sets; **every header carries pool AND step budget**.
- `fig_maze_conditions` is **six seeds × two configs**, hard seeds (1, 4) shaded by measured ceiling; ⛔ seed 4's **control tie** and seed 5's **rival tie** are marked and must be described as such.
- `fig_deletion` is the **four-measurement** rule, with the **random-1 % control** beside `red ∧ sphere` — that control is what shows the conjunction collapse is content, not instability.
- `fig_cmazet_rel_per_arm` is a **not-landed slot** (gate 2 blocked). ⭐ Two branch bundles are now in `~/clu/ship/` (`w21-rel-newseeds`, `w21-maze-t-clu-improve`) carrying the `w21_clu` module that blocked it. ⛔ **Drawing it is NOT your job** — record it as pending, with the unblock noted.

## ACCEPTANCE
`README.md` indexing every file with all four fields; both completeness counts printed; rebuild-identical check run and its result stated; PNG+PDF parity; `FINDINGS.md`; branch committed and bundled, `git bundle verify` passing.

## OUTPUT
`~/clu/outputs/w22-paper-visuals/` (your report) + the index into `~/clu/outputs/paper-visuals/README.md`.
