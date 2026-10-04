# REFEREE-FIXES — w18 spoke `referee-fixes`

**Applies:** REFEREE-v2 **M2, M4, M5, M6, M7** to **both** `~/clu/outputs/w17-paper/DRAFT-v2.tex` and
`DRAFT-v2-style.tex`. **HELD, not applied:** M1, M3.
**Zero compute, zero scored runs.** The only commands that were not reading were `pdflatex`/`bibtex` and
two checkers.
**Mechanism:** every edit is one anchored replacement in `apply_fixes.py`. Each anchor is asserted to
occur exactly once, so an edit either lands where it was aimed or the script stops. 118 replacements
were applied to pristine copies (`pre/`, SHA-256 in `pre/SHA256.txt`). The results were checked and
compiled, then installed into `w17-paper/`. Full diffs: `diff-DRAFT-v2.patch`,
`diff-DRAFT-v2-style.patch`.
**Lines** below are lines of the **installed post-edit files** (SHA-256 in `post-SHA256.txt`:
v2 `8e45fa98…`, style `3999daab…`). "v2" = `DRAFT-v2.tex`, "style" = `DRAFT-v2-style.tex`.

**Scope vs waves 15/16:** none of M2/M4/M5/M6/M7 depends on a w15 or w16 row. **No w15/w16 number
was added**; every number added comes from w ≤ 14 evidence (plus `_thinkpad/w4`). **No
`\PLACEHOLDER{}` argument was changed**: 29 before and 29 after, identical as a multiset. Their
*order* changed only because M5 split `tab:cmazet` (see M5).

---

## M2 — two rival sets an order of magnitude apart on the same corners · **APPLIED**

**What the artefacts say (why the gap exists, as far as artefacts go):**

| | Table 1 rivals (`tab:t7`) | Table 3 rivals (`tab:ladder`) |
|---|---|---|
| source | `w12-deletion-baselines` as-trained comparators | the earlier set, `_thinkpad/w4/spoke-a-rivals` |
| fit pool | **16,384** rows, value hold-out pool, a "BUILD/SMOKE scale" (`w12-deletion-baselines/FINDINGS.md` §4, §10) | **32,256** rows, `montero_combination` pool, `holds_out_any_value: False` (`rivals/seed_{0,1,2}/*.rival.json`) |
| params | 130,433 – 131,891 | 117,878 – 119,858 (`spoke-a-rivals/REPORT.md`) |
| schedule | epochs / lr / wd inherited from the `w4_learners` grids, not tuned (§6.6) | 40 epochs, batch 256, lr 0.003, 5,040 steps, the same for every family and seed (`rivals/seed_*/{mlp,lstm,transformer,cont_mlp}.json`) |
| population caveat | never saw the held-out values | fitted on rows that **carry the held-out values**. Its own record: "not like-for-like … flatters whichever one saw more". The fit-row overlap with the ladder's corners is DECLARED NOT-MEASURED (`w12-composition-ladder/PREREG.md` §7) |

On shared corners the **earlier (ladder) set is the stronger one**. The draft's old caveat called it
"pre-repair", which implied it was the weaker set; that framing is corrected.

**Before → after**
- **§3.1 lead**, v2 l.333–335 / style l.333–335. *Before:* "Every rival is beaten beyond the
  declared 1.0 margin on 109 of 109 corners on every leg; the smallest paired delta … +39.266" led
  the section. *After:* "The rivals of Table 1 are the weaker of the two rival sets in this paper;
  the store's standing against the strongest available rival is given below (Two rival sets). Each
  of them is beaten …". The +39.266 sentence stays but is no longer the lead.
- **New paragraph "Two rival sets, and which is stronger."**, v2/style l.380–410, placed just before
  `tab:ledger-kstar`. It names both sets, their fit pools, parameters, schedules and sources, and the
  population asymmetry. It states that the earlier set is stronger: on the 60 pair corners the MLP
  reads 2.9333 / 3.4333 / 5.2667 against 17.0167 / 22.0000 / 23.3333, and the table floor is
  identical in both. It then reports the store against the **strongest learned rival of either
  set**, per seed:

  | corner set | store | strongest rival | sources |
  |---|---|---|---|
  | all 109 corners of Table 1 | 47.4954 / 47.4037 / 49.1193 | 30.0275 (LSTM) / 28.9174 (LSTM) / 37.5138 (transformer) | `w12b-nr6-and-kstar/NR6-T7.md` (control-gate "arm" column); `w13-seeds-and-tables/TABLES.md` T8 `shipped` leg |
  | ladder n = 2 | 31.800 / 32.500 / 33.633 | 19.4167 / 22.0000 / 24.2500 | `KSTAR-LADDER.md`; TABLES.md T8 |
  | ladder n = 3 (store's lowest n ≥ 3 rung) | 60.150 / 62.783 / 61.908 | 44.4333 / 41.5667 / 51.9917 | same |

  These are printed scores placed side by side. **No margin was computed**, because subtracting them
  would re-derive a number, so the paragraph calls them "score comparisons, not paired margins".
  The paragraph then says: "we do not lead with the margins of Table 1 over its own rivals."
  The rival scores in the `shipped` row are independent of K (`w12-composition-ladder/FINDINGS.md`
  §3), and the paragraph says so.
- **New `\TODO`** (v2/style l.406–409). No artefact isolates which difference causes the gap (fit
  population, pool size, parameter count or schedule). None pairs the K\* store against the earlier
  set with an interval. The value-hold-out refit of the earlier set
  (`spoke-a-rivals/rivals_vhpool/`) exists but is scored in no artefact. **So the gap is described
  from artefacts but not causally explained**, and the TODO says so, as the task required.
- **Table 1 caption** (`tab:t7`): "Rivals are matched on read-path parameters (about 131k)" →
  "… (about 131k) and are the weaker of the paper's two rival sets (§3.1, Two rival sets)".
- **Table 3 caption** (`tab:ladder`, l.454): "Rival rows are the read-path-matched rivals" → "Rival
  rows are the earlier, stronger read-path-matched rival set (§3.1, Two rival sets)".
- **Ladder caveat** (l.499–502). *Before:* "…columns are the earlier rival set's pre-repair numbers
  and were not re-fitted…". *After:* the set was fitted before a later dtype repair and not
  re-fitted, "yet that set is the *stronger* of the paper's two rival sets on shared corners".

⚠ **What this makes visible (reported, not softened):** against its strongest available rival the
store's lead is much smaller than Table 1's +39 to +49. For example, n = 2 seed 2 is 33.633 against
24.2500. The rival set that is stronger here also saw the held-out values, so even this comparison
favours the rival.

⚠ **Overlap with the HELD M1:** M2's own minimal fix requires naming the fit pools, so the new
paragraph prints the Table 1 rivals' 16,384 rows (already in the draft at §3.4 and App. B). I did
**not** print the store's 131,072-row pool beside it. I did not add the 1× margin sentence, did not
touch "naive against naive", and did not touch the abstract or contribution 1. Those are all M1.

## M4 — positioning contradicted by uncited work · **APPLIED (cite-and-distinguish)**
- **Abstract** (v2 l.90–93 / style l.95–97). *Before:* "their intersection is empty: a mechanical
  screen of approximately 4,050 works finds no generative model that deletes training rows exactly."
  *After:* "…empty in a precise sense: a mechanical screen of approximately 4,050 works finds no
  *factor-compositional* generative model (one that generates held-out combinations of factor
  values) that deletes training rows exactly while reading no data and retraining nothing."
- **§1.1 heading**: "…with exact deletion it is empty" → "…with **data-free** exact deletion it is
  empty".
- **§1.1** (v2 l.155–172 / style l.154–171). The orthogonal query now "returns zero works **on the
  screen**". A new paragraph cites **SISA** (`\citep{bourtoule2021machine}`, already in
  `references.bib`): exact, but it reads the shard's retained data and retrains. It also cites
  **Compartmentalized Diffusion Models** (Golatkar, Achille, Swaminathan and Soatto, arXiv:2308.01937):
  they compose models of *data sources*, not held-out combinations of factor values, and forget a
  source by dropping or retraining its model. The paragraph closes: "Neither deletes without reading
  data and retraining, and neither generates held-out combinations of factor values." The position
  sentence is narrowed to "the intersection of factor-compositional generation with exact deletion
  that reads no data and retrains nothing is empty in the works we could screen". The screen's
  scope caveats are kept.
- **§1.2** (v2 l.195–196 / style l.196–197): "can delete only by retraining" → "can delete
  **exactly** only by retraining (the whole model, or a shard of it `\citep{bourtoule2021machine}`)".
- ⚠ **Hand-off to cite-tagger:** CDM is **not in `references.bib`**, and I do not own that file. It
  is named in prose with no `\cite`, following the draft's convention for citation slots. A LaTeX
  comment at that spot gives the full reference and asks for a bib entry. I checked its title,
  authors and arXiv ID by web search (arxiv.org/abs/2308.01937). I read the abstract only, as
  referee-3 did.

## M5 — environments sharing tables · **APPLIED**
- **`tab:req` split** (v2 l.1675–1720) into **`tab:req` = "CMAZE-2 request-layer audit", seed 0** and
  a new **`tab:req2b` = "CMAZE-2b request-layer audit", seeds 0/1/2**. The seed count for the CMAZE-2
  side is from `w14-request-layer-fix/FINAL-REPORT.md` §4–5 ("BEFORE (w9 layer, seed 0)",
  `measure_before_0.json`). The before/after contrast stays in prose (§4.2, which already carries
  it). Its reference is now "Tables req and req2b".
- **`fig_maze_requestlayer` dropped** from both drafts, because it plots CMAZE-2 and CMAZE-2b
  together. A source comment marks where it was. **Hand-off to the figures owner:** re-insert it only
  as two single-environment figures. I do not own `figures/`.
- **`tab:cmazet` split** (v2 l.2019–2070) into **`tab:cmazet` (CMAZE-T)**, which keeps
  `\label{tab:cmaze-del}`, and **`tab:cmazeo` (CMAZE-O)**. The label already existed and now has its
  own table. **The placeholder arguments were moved without being edited.** Their document order is
  now all CMAZE-T cells, then all CMAZE-O cells: old positions 1–11, 13, 15, 17, 18, 20, 22, 24, 12,
  14, 16, 19, 21, 23, 25–29. `placeholder-fill` greps by argument, so nothing it keys on changed.
  The references in §4.4 and the Placeholder index now name both tables.
- **Beyond the referee's list:** **`tab:impl`** mixed the maze store (CMAZE-2) with two Shapes3D
  implementations, against the paper's own rule "no two environments ever share a table". It is split
  into **`tab:impl-maze`** (implementation 1) and **`tab:impl`** (implementations 2–3). The "five
  orders coarser" comparison moved into the caption.
- After the edits, no table in either draft holds two environments.

## M6 — seed-count column · **APPLIED**
A `seeds` column was added to all 12 tables the referee listed and to `tab:ladder`, which also lacked
one. The value was taken from each table's caption or source. The split tables carry one from the
start.

| table | seeds value | table | seeds value |
|---|---|---|---|
| `tab:ledger-kstar` | 3 | `tab:rivals-del` | "1 / 3" (seconds are seed 0; the gap is a 3-seed mean; the caption now says this) |
| `tab:maze-del` | 3 | `tab:unlearn` | 3 |
| `tab:verb` | 3 (last column, because of the `\multicolumn{2}` rows) | `tab:k5` | 3 |
| `tab:cmaze2` | 3 | `tab:1x` | 3 |
| `tab:cmaze1` | 3 | `tab:encloop` | 3 |
| `tab:req` / `tab:req2b` | 1 / 3 | `tab:crr` | 3 |
| `tab:ladder` (added) | 3 | `tab:cmazet` / `tab:cmazeo` | "pending" (no number for a w16 row) |

No claim got weaker from M6. The counts were already in the captions; the one visible change is that
`tab:req`'s CMAZE-2 side is **n = 1**.

## M7 — venue compliance (VENUE.md §§0, 1, 4, 7)
- **(e) Process language · APPLIED.** Removed:
  - "task spine / this spoke does not divide them": the `\TODO` was reworded, and it **stays a
    `\TODO`**, per N1.
  - "dropped from paper rows by ruling" (×2) → "not a scored row of this study".
  - "the Head to confirm" → "the authors to confirm".
  - "one-K ruling" → "one-K rule of this paper".
  - "wave 16 / wave 18 regenerates / wave-14 data / Wave-15 rows / Wave-16 rows / pending a later
    wave".
  - "w9 request layer" in a heading.
  - "the task first named" → "first named in the design".
  - The figure-slot fallback's "owned by the plots spoke".

  "Charter controls" became "pre-declared landscape controls" everywhere in prose and captions (7
  places). "BAR 1/2/3/5" became "Criterion 1/2/3/5", under a paragraph renamed "The pre-registered
  criteria"; prose "BAR D" became "pre-registered criterion D". The **"Corrections absorbed"
  appendix is out of the submission PDF**: it is wrapped in `\iffalse … \fi` and kept in the source
  for the record, which keeps every provenance comment. The reproducibility statement now points to
  App. A and the not-runs appendix instead of `app:corrections`.
  - Remaining, **not mine to touch:** "w15/w16", "charter-C" and "BAR D" inside `\PLACEHOLDER{}`
    arguments. They render until `placeholder-fill` replaces them.
  - Checked at source level: after stripping comments, placeholder arguments and the `\iffalse`
    block, a grep for wave, spoke, Head, ruling, task spine, `BAR `, charter, hub, w9, w14, w15 and
    w16 finds **0** hits in either draft.
- **(c) Bibliography.** Already present: cite-tagger ran, 96 entries, and bibtex runs clean. SISA was
  added as a `\citep`. **CDM needs a bib entry: handed to cite-tagger (DECLARED NOT-DONE here,
  because I do not own `references.bib`).**
- **(a) AI use statement — DECLARED NOT-DONE.** The facts belong to the authors (VENUE.md §7). It
  remains a `\TODO`, with its process wording removed.
- **(b) 9-page limit — DECLARED NOT-DONE.** Trimming is the Head's pass. Measured from the `.aux`
  after the edits: the main text runs from p.1 to §6 on **p.13 (v2) / p.14 (style)**, against the
  strict 9. M2 and M4 each added a paragraph, so the main text grew by about a page.
- **(d) Abstract deadline (Sep 18, 2026 AOE) — DECLARED NOT-DONE.** This is a submission decision for
  the Head, not a drafting item.
- **(f) Zero `\PLACEHOLDER`/`\TODO` at submission — DECLARED NOT-DONE.** There are 29 placeholders,
  which belong to `placeholder-fill`. TODOs went from **4 → 5**; the new one is the honest M2 TODO.

## M1, M3 — **HELD, not applied** (for the Head and the Advisor)
- **M1** (Table 1 rivals trained on 16,384 rows vs the store's pool; whether "naive vs naive"
  survives): **HELD.** It changes the headline claim and the abstract. The abstract's "beating every
  read-path-matched learned rival", contribution 1 and "Rivals: naive against naive" are unchanged:
  each still appears exactly once, the same as before the edits.
- **M3** (whether exactness is rescoped off the address book): **HELD.** It changes what "exact
  deletion" claims. §1.2's "the deletable object equals, to floating-point roundoff …", the scope
  box, "Deletion is store-scoped and exact" in the abstract, and the ethics statement are unchanged:
  each still appears exactly once. My M4 edit in §1.2 touches only the next clause ("…can delete
  exactly only by retraining…").

---

## The four invariant diffs — ALL EMPTY (on the installed shared files)
`style_diff_check.py ~/clu/outputs/w17-paper/DRAFT-v2.tex ~/clu/outputs/w17-paper/DRAFT-v2-style.tex`
→ exit 0 (`STYLE-DIFF-CHECK-shared.md`; the identical run on `work/` is `STYLE-DIFF-CHECK-post.md`;
the pre-edit baseline is `pre/STYLE-DIFF-CHECK-pre.md`):

| diff | before edits | after edits | result |
|---|---|---|---|
| 1. numbers (every digit run, code only) | 2525 = 2525 | **2757 = 2757** | **EMPTY** |
| 2. `\PLACEHOLDER{}` ordered, byte-identical | 29 = 29 | **29 = 29** | **EMPTY** |
| 3. artefact comments, identical order | 265 = 265 | **287 = 287** | **EMPTY** (paths 305 = 305; strings 288 = 288) |
| 4. section order (headings / labels) | 96 / 74 | **97 / 76** | **identical sequences** |
| extra: floats + figslots byte-identical | 31 | 30 | True |
| extra: preamble byte-identical | — | — | True |
| extra: `\TODO` ordered identical | 4 | 5 | True |

⚠ **The count of 265 artefact comments changed to 287, identically in both drafts. This is
deliberate.** Arithmetic: 265, plus 16 SRC comments in the new M2 paragraph, plus 2 SRCBLOCK/ENDSRC pairs (4
comments) from the two new split tables (`tab:req2b` and `tab:impl-maze`), plus 2 SRC comments on the
two request-layer captions, gives 287. Each added number carries its artefact comment, as the task
requires. Separately, removing the wave-number lines dropped 6 of the 7 "(no result number…)"
markers; those are not part of the ordered count. Keeping 265 would have meant adding numbers with no
provenance. The invariant that matters, identical content in identical order across the two drafts,
holds.

**Provenance check** (`w17-draft-v2/check_provenance.py`, which looks up every number token in the
file its SRC names): v2 has **1,864 tokens checked, 1 not found**; the pre-edit file had 1,824
checked and the same 1 not found. The miss is pre-existing and not mine: the K list on the
`\Kstar` rule line. Every number I added was found in its cited artefact. Unsourced lines went from
13 to 6; all 6 remaining are pre-existing template, bib and citation tokens. The style file gives the
same result. Files: `prov_{pre,post}_*.txt`.

## Compile — both performed, both clean
`pdflatex → bibtex → pdflatex → pdflatex`, TeX Live 2026, in `build/` with symlinks to the shared
`iclr2027/`, `figures/` and `references.bib`:

| draft | exits | output | undefined refs / errors (final pass) |
|---|---|---|---|
| `DRAFT-v2.tex` | 0 / 0 / 0 / 0 | 35 pages | 0 |
| `DRAFT-v2-style.tex` | 0 / 0 / 0 / 0 | 36 pages | 0 |

Logs: `compile-DRAFT-v2.log`, `compile-DRAFT-v2-style.log` (all four passes) and
`pdflatex-final-*.log`. PDFs: `DRAFT-v2.pdf`, `DRAFT-v2-style.pdf` in this directory. I did not
overwrite `w17-paper/compile-v2*.log`, because those are another spoke's; they now describe the
pre-edit files.

## What did NOT run / DECLARED NOT-DONE
- **Visual inspection of the rendered PDFs.** No PDF rasteriser or text extractor is installed here
  (no poppler, no pypdf or fitz). Layout was checked through clean compiles, the `.aux` page numbers
  and the source, not by eye.
- **M7 (a), (b), (d), (f):** see above; these are the Head's decisions or other spokes' work.
- **CDM bib entry:** handed to cite-tagger.
- **Split figure for the request layer:** handed to the figures owner. The combined figure is
  dropped.
- **Causal explanation of the M2 gap:** no artefact isolates it, so it is left as a `\TODO`. Scoring
  `rivals_vhpool` on these corners would settle it, but that is a scored or free run, not a drafting
  job.
- **SHOULD-FIX S1–S12 and NICE-TO-HAVE N1–N6:** out of this spoke's scope, not applied.
- **M1, M3:** HELD.
