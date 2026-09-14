# plain-text-builds — paper-writer report

Task + acceptance criterion: build V2 and V5 as PLAIN unformatted documents in a new `.claude/papers/plain/{v2,v5}/` — every page-fitting typographic device stripped, the author token absent from prose/captions/labels/filenames, and the banked figures restored with judgment — zero number changes, source folders read-only.
Status: **done** — both builds compile (V2 **17 pp**, main text 7.00 pp; V5 **18 pp**, main text 5.00 pp), standalone author-token count outside the bibliography = **0** in both, **8 of the 10 banked figures restored**, 1 restored figure and 1 shipping figure re-rendered token-free with **provably identical plotted values**, 1 banked figure excluded with a reason, all five source folders **byte-identical**.

**DIAL DECLARATION (echoed): none — typesetting/editorial pass. Laundering control: n/a. Falsifies: n/a. Does NOT falsify: n/a.** Zero measurements; zero numbers changed.

**⚠ THIS REPORT CONTAINS A DOWNSTREAM RECONCILIATION LIST AND IT NEEDS AN OWNER (protocol §5 corollary).** Five items, all in §6: (a) ⛔ **the V2 headline figure and one banked V2 figure printed the author token / internal IDs *inside the PNG*** — fixed here for those two, but the same class of defect must be checked before any figure ships anywhere; (b) `sf1_mo_estimator_overlay.png` is **unshippable and un-fixable in this pass** (author token ×3 on canvas, no surviving generator) — an owed re-render from banked data; (c) **two V5 tables do not fit the text block at any permitted size** and were already overfull in the shipped source (91.7 pt and 406.2 pt) — an owed editorial restructure; (d) the **caption-sync worklist is still unexecuted** and this build inherits it (14 `I-J`/`I-R*` tokens in V5); (e) ⭐ **V5's 4.00 pp main-text compliance was typographic** — proved by control build, not asserted.

---

## 1. What I did

1. Created `.claude/papers/plain/{v2,v5}/` (each: `submission.tex`, `submission.pdf`, `figs/`, `BUILD-NOTE.md`, style file, build logs). Sources copied out, never edited in place; `pdflatex` (`/Library/TeX/texbin/pdflatex`) run only inside `plain/` and one scratch control folder.
2. **Stripped every page-fitting device** (item 1): 15 distinct devices in V2, 11 in V5 — all font-size commands on body text/captions/tables/appendices/references, the `multicol` reference list, the `\@startsection` heading-skip redefinitions, float-placement overrides, `\textfloatsep`/`\intextsep`/`\abovecaption`/`\belowcaption` skips, `\raggedbottom`, and every reduced figure width (now `\linewidth`). Both BUILD-NOTEs table them line by line.
3. **Removed the author token everywhere but the bibliography** (item 2), rewriting each mention as a citation carrying the arXiv number, renaming the two labels and one filename that embedded it — **and, on discovering the token printed inside the headline PNG, re-rendering that figure from the render pass's own generator with three text strings changed.**
4. **Restored banked figures with judgment** (item 3): all seven V5 figures and two of three V2 figures, each with a new provenance- and scope-bearing caption; the third excluded with a reason (§6b).
5. Verified: two-way numeric-token check per paper, three sweeps per file with positive controls, a byte manifest of all five source folders before and after, and a control build to attribute V5's page growth.

## 2. How I verified (commands + observed numbers)

**Builds.** `pdflatex … ×3` each.
- V2: `Output written on submission.pdf (17 pages, 990228 bytes)`; page split via `pdftotext -bbox` (the instrument the earlier build notes used): **main text 7.00 pp**, appendix 10.00 pp. Source for reference: 14 pp / 6.14 pp.
- V5: `(18 pages, 1776221 bytes)`; **main text 5.00 pp**, appendix 13.00 pp. Source: 10 pp / 4.00 pp.
- All `\includegraphics` resolve: V2 5 files, V5 11 files (grepped from the logs). No undefined references or citations.

**Overfull/underfull, reported not fought.**
- V2: **1** overfull `\hbox` (3.57 pt, loan-ladder table), 0 overfull `\vbox`, 26 underfull `\hbox`, 14 underfull `\vbox`. **Permitted `\small` exceptions used: 0.**
- V5: **3** overfull `\hbox` after `\small` on five tables (196.50 / 604.92 / 49.44 pt), 0 overfull `\vbox`, 9 underfull `\hbox`, 4 underfull `\vbox`. **Permitted `\small` exceptions used: 5, all listed in the build note with before/after widths.**

**Author-token sweep (per file, positive-controlled).**
| file | `grep -c "\bMo\b"` | what the hit is | Morse / Moser excluded | positive control on the source |
|---|---|---|---|---|
| `plain/v2/submission.tex` | 1 | the bibliography entry (permitted) | Morse ×1, Moser ×2 — survive | 13 occurrences on 9 lines |
| `plain/v5/submission.tex` | 1 | the bibliography entry (permitted) | none present in this file | 2 occurrences on 2 lines |

Pronoun sweep `\bhis\b|\bHis\b|\bhe\b|\bhim\b`: **0** in both (V2's source carried 8; one of them, *"we recover his reported number"*, survived my first pass and was caught on a PDF read-through — a useful reminder that the token grep does not catch pronouns).

**Internal-apparatus / path / program-vocabulary sweep** (`SF-[0-9]|F-[0-9]|F5|C2|C3|CM-[0-9]|Cor-[0-9]|.claude|/Users|scratch/|handover|Advisor|Hub|spoke|never-quote|PREREG|N[0-9]{3}|CSF3|CAMELS|CMAPSS|K5|organizer swap|13.9|bprime|CLU-former`): **0 hits in both files.** Positive control on `tasks/plain-text-builds.md`: **6 hits**.

**Semantic hermeticity:** unchanged from the sources — the theory note is cited as *(Anonymous, 2026)*, and the naming-continuity sentence *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"* is present verbatim in both. Anonymization posture unchanged (empty `\author{}`, `\@notice` suppressed, V5's PDF-metadata scrub retained).

**Two-way numeric-token check** (numeric multiset of the `.tex`, graphics widths and `\setlength` args excluded as typography, my inserted figure blocks removed):
| | distinct tokens | in source not in plain | in plain not in source |
|---|---|---|---|
| V2 | 407 / 407 | `2026`×2 (the two replaced citations), `1`×1 (old filename) | `2605.03338`×3 |
| V5 | 570 / 570 | `2026`×1, plus `4`,`2`,`0`×3 (all deleted typography arguments) | `2605.03338`×1 |
Every number appearing in a new caption was checked against a sentence of the same paper; the lists are in the two build notes §4.

**Figure re-renders, verified by the render pass's own data tap** (`.claude/outputs/figure-render-pass/tap.py`, order-insensitive multiset of every numeric array handed to a plotting call):
| figure | generator | evidence |
|---|---|---|
| V2 headline → `fig_lifetime_headtohead.png` | `figure-render-pass/new_v2f1.py`, **4 string literals changed** | tap digest `baedf7981e…` (9 calls) **identical** to the banked render's digest; same pixel size 2044×1118 |
| V2 `fig1_gmor.png` | `scratch/v2-full-runs/make_figures.py::fig1_gmor`, **2 legend strings changed** | unmodified generator reproduces the banked PNG **byte-for-byte** (sha1 `00eaceca1adf…`); edited run's tap digest `9c284b32c8…` (7 calls) **identical** to the unmodified run's |

**Control build (V5 page attribution).** Plain V5 with the headline figure returned to `0.58\linewidth`: still **5.00 pp main text, 18 pp total** ⇒ the 4.00 → 5.00 pp move is the font/skip stripping, **not** the figure width.

**Source folders byte-identical** (full sha manifests before/after): `v2-short` 21 files, `v5-short` 25, `v2-neurreps-descoped` 10, `neurreps-variants` 11, `palm-variant` 16 — **all five unchanged**; banked generator outputs also unchanged (tapped renders were redirected to scratch).

## 3. Banked-figure dispositions (item 3, with reasons)

**V5 — 7 of 7 restored, all to the appendix, grouped with the result they evidence:** `figB_dlaw`, `figB_signflip`, `figB_massive_vs_flat` → Appendix A (the budget); `figC_lambda_coset`, `figC_register_capacity`, `figC_Tstar` → Appendix B (the emergent arm, beside the paragraph that states exactly those three numbers); `fig2_vault` → Appendix C, beside the designed-arm paragraph. **Multi-seed status printed in each caption**, including the two single-checkpoint figures (seed 44: `figB_dlaw`, `figB_signflip`) and the one **n < 3** figure (`figC_Tstar`: two emergent seeds + one designed control).

**V2 — 2 of 3 restored.** `fig1_gmor` → **main text**, under §4.1, because it is the clearest presentation of the paper's single contribution and §4.1 previously had no figure. `fig3_retention_overlay` (the Head's named figure) → **appendix**, in a new one-figure appendix section, because one of its three curves is a single representative checkpoint and single-seed material is appendix material; its caption says so plainly and points the 5/5-seed statement back to §4.3. **This is the one placement call I would like ruled** (§7 Q1). `sf1_mo_estimator_overlay` → **excluded**, see §6b.

## 4. Formatting minimalism

No bold outside structural headers was introduced; the sources' pre-existing `\textbf{}` run-in lead-ins inside V2's appendices were left as they are (removing them would be a rewrite of body text, which the pass forbids) — **flagged as a judgment call in §7 Q2**. `\texttt{}` is used for software/flag/file names in the new captions (`langevin_noise="fdt"`), matching the sources. Italics in the new captions are limited to the scope/evidence rider, matching the surrounding captions.

## 5. Findings

1. ⛔ **The author token was printed inside PNG canvases, not just in the `.tex`.** The task (and the addendum) assumed filenames and labels; three figures actually carried it in titles, legends and in-axes notes. A filename rename does not satisfy the directive. Two were fixable from surviving generators and are fixed with proof of identical data; one is not (§6b).
2. ⛔ **Internal-apparatus tokens were printed inside a banked PNG too**: `fig1_gmor`'s legend read `F5 exact-map prediction` and `(C2 retention)` — an internal document ID and an internal instrument tag, both of which would have been a hermeticity leak on the page. Fixed by re-render.
3. ⭐ **V5's 4.00 pp main text was a typographic result** (control-proved, §2). At default formatting the same words are 5.00 pp. The Advisor's Addendum-45 standing consequence is confirmed on the measurement, not on inference.
4. **Two V5 tables cannot be made to fit** — they overflow by 196 pt and 605 pt at `\small`, and were already overfull by 92 pt and 406 pt in the shipped source at `\scriptsize`/`\tiny`. The four-instrument table is the severe one.
5. The restored banked figures were rendered for a larger canvas than the re-rendered ones, so at `\linewidth` their in-figure type is proportionally smaller than the purpose-rendered figures'. All are legible in the built PDF (checked by reading the pages), but they are not type-matched to Figures 1/5/9 of V5.
6. Float placement was left to LaTeX as instructed: V2's restored retention figure floats one page past its own appendix section heading.

## 6. Reconciliation list (needs an owner)

a. **Before any figure ships anywhere, sweep the PNG canvases, not just the `.tex`.** The render pass's label-hygiene sweep covered only the six figures it re-rendered; every other banked figure is unswept. Owner: whoever runs the next figure pass.
b. **Owed re-render: `sf1_mo_estimator_overlay.png`** — canvas carries the author token ×3 (title, legend, legend) plus the internal ID `SF-1`; **no generator survives** (0 `.py` files under `.claude` reference `mo_estimator`/`estimator_overlay`). Data is banked and the job is actionable: `.claude/outputs/v2-referee-experiments/mo_estimator_extract.npz` + `mo_estimator_table.json`. Until then V2 ships without the predictor-substitution figure (the result remains in Appendix F's text).
c. **Owed editorial restructure: the two unfittable V5 tables** (the `$T>0$` budget table and the four-instrument shape-claims table). Suggested fix is transposition or a split, not a font size. Pre-existing defect, visible in the source's own log.
d. **The caption-sync worklist is still unexecuted** (`.claude/outputs/figure-caption-sync.md` = BLOCKED, zero files edited). This build inherits its live items — V5 still names `I-J`/`I-R1`/`I-R2`/`I-R3` (14 tokens) in appendix text and in the Fig 9 caption, and the Fig 11 caption still opens *"panels are labelled by the pre-registered prediction each tests"*. ⚠ **There are now three V2 folders and two V5 folders; if the caption-sync pass runs on the sources only, `plain/` silently diverges.** The Hub should decide whether `plain/` becomes the live line or a snapshot.
e. **V5's page-limit compliance needs re-reading at default formatting** (finding 3) — it is a venue-fit question for the Advisor, not a build question.

## 7. Open questions for the Hub / Head

1. **Placement of `fig3_retention_overlay`** — I put the Head's named figure in the appendix because its designed curve is one checkpoint (rule: single-seed material is appendix material), while its baseline and emergent curves are medians over 5 and 3 seeds. Promotion to main text under §4.3 is a one-line change if the Head wants it there.
2. **Run-in `\textbf{}` lead-ins in V2's appendices** ("**The price of the prior (evidence).**", "**Per-step ratios:**", …) — formatting minimalism says no bold outside structural headers, but they are body text the pass may not rewrite. Left as-is; say the word and they become italics or plain.
3. **The citation handle.** With the name banned, V2 needed a repeated handle; I used *"the equivariant-Lyapunov preprint (arXiv:2605.03338)"* at first mention and *"the preprint"*/*"its"* afterwards. If the Head prefers a numbered `\cite` scheme, that is a whole-bibliography conversion and a separate pass.
4. **Does `plain/` supersede the variant folders, or sit beside them?** (Bears on item 6d.)

## Proposed handover updates (for the Hub)

- `.claude/papers/plain/{v2,v5}/` exists: plain builds at **17 pp / 7.00 pp main** and **18 pp / 5.00 pp main**, page counts explicitly not optimised; BUILD-NOTEs carry the device tables, the `\small` exception list, the sweeps and the numeric checks.
- **Standing lesson to record: the anonymization/hermeticity sweep must cover figure canvases.** Three figures printed the author token and two printed internal document IDs inside the image, where every `.tex`-level sweep is blind.
- **Standing lesson, confirmed by measurement:** V5's 4.00 pp main text was typographic — at default formatting it is 5.00 pp, and the figure width is not the cause (control build).
- Two owed spoke tasks: a re-render of `sf1_mo_estimator_overlay` from banked npz (results-analyst, token-free labels, tap-verified), and a table restructure for V5's two unfittable tables (curator/writer).
- `figure-caption-sync` remains unexecuted; its worklist now applies to five folders, not three.
