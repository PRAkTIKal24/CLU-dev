# REFEREE3-FIXES — w20 `referee3-fixes`, items (a)–(g) in both drafts

**Spoke:** w20 `referee3-fixes` on `batmac`. The worktree is `CHLU-w20-referee3-fixes`, on branch `w20/referee3-fixes`, based on `base/iter8` = `148f3b7`.
**Edited (exclusive lock):** `~/clu/outputs/w17-paper/DRAFT-v2.tex` and `DRAFT-v2-style.tex`.
**New sidecar:** `~/clu/outputs/w17-paper/CORRECTIONS-ABSORBED.tex`, which holds the block moved out under item (e).
**Answers:** `~/clu/outputs/w17-paper/REFEREE-v2.md` (M2, M5, M6, M7(e)), under the Head's framing in `tasks/w20/referee3-fixes.md`.
**Zero compute. Zero scored runs.**

**Line numbers.** Unless marked, the line numbers below are those of the edited `DRAFT-v2.tex`. The style file carries every edit at a nearby line; §7 gives both.
**Full diffs:** `DRAFT-v2.w20.diff` and `DRAFT-v2-style.w20.diff`.
**Pre-edit snapshots:** `before-20260923T174636Z/`.
- `DRAFT-v2.tex`: SHA-256 `de7b5fdb…`
- `DRAFT-v2-style.tex`: SHA-256 `2359dfe1…`

**Pre-flight:** `tmux ls` showed only `authtest`, `hubhost` and `w20-referee3-fixes`. None of `w18-placeholder-fill`, `w19-draft-figure-slots` or `w20-citation-prune` was live.

---

## (a) Positioning, rewritten to the Head's framing

| where | before | after |
|---|---|---|
| abstract, L90–94 | "…exact unlearning is an established one, **but their intersection is empty in a precise sense: a mechanical screen of approximately 4,050 works finds no factor-compositional generative model … that deletes training rows exactly while reading no data and retraining nothing.**" | "…exact unlearning is an established one. We work at the level of a single *naive unit* and compare it only with other naive units (an MLP, an LSTM, a transformer and a continuous-input MLP): we do not claim the best compositional-generation or unlearning system, and systems built on top of units, such as sharded retraining or compositional diffusion, are out of scope." |
| §1 heading, L125 | `Introduction: positioning, and why the intersection matters` | `Introduction: a naive unit, and why its three properties matter` |
| §1.1 heading, L129 | `Positioning: compositional generation is crowded; with data-free exact deletion it is empty` | `Positioning: a naive unit, compared with naive units` |
| §1.1, new opening paragraph, L132–138 | (none) | The CLU is a **naive AI unit**. It is compared only with other naive units: MLP, LSTM, transformer and continuous MLP, matched on the read path. The paper does **not** claim the best compositional-generation or unlearning system. Systems built on top of units (sharded retraining, compositional diffusion, physics-informed variants of an MLP) are **out of scope by definition**. Each is cited in one sentence, and the unit/system distinction is stated. |
| §1.1 screen paragraph, L158 | "**The intersection of the two literatures is empty on a systematic screen.** A forward-citation crawl…" | "A systematic screen of both literatures records what we could find. A forward-citation crawl…" The positioning counts from the positioning scout are **kept verbatim**, SRC unchanged. |
| §1.1, L171–188 | "Two works outside the screen … fix the form of our claim." Then SISA and CDM in several sentences, then "Neither deletes…", then "We therefore state the position…: ***the intersection of factor-compositional generation with exact deletion that reads no data and retrains nothing is empty*** in the works we could screen." | (1) "These are the screen's counts, not a claim that no such model exists…", followed by the residual-risk sentence, which was kept. (2) "Three families of systems built on top of units are out of scope, one sentence each." (3) **SISA**, one sentence: a system that shards the data and retrains the unit on one shard; we compare with the unit it wraps. (4) **Compartmentalized diffusion**, one sentence: a system that composes per-source models, not held-out factor values within one unit. (5) **Physics-informed MLP variants** (PINNs, Raissi et al. 2019; Hamiltonian NNs, Greydanus et al. 2019), one sentence: systems that add a physical loss or structure to an MLP unit; the rival we compare with is the plain MLP unit. |
| §1.1, L190 | "One distinction **in this intersection** is ours…" | "One distinction is ours…" |
| §1.2 heading, L198 | `Why the intersection matters: world models need all three at once` | `Why the three properties matter: world models need all three at once` |

**"The intersection is empty" is removed everywhere.** In the code of both files, `grep` for `intersection|is empty` finds 0 hits. The only remaining hits are in `before-*/`.

**Citations.** PINN, HNN and CDM are named in prose only, each with a `% cite-tagger: bib entry needed` comment. None is in `references.bib`, and I did not add bib entries: `references.bib` belongs to the `citation-prune` spoke, and a guessed entry risks fabrication. A `\citep` to a key that does not exist would break the 0-undefined invariant. **`w20-citation-prune` must add SISA's sibling entries: CDM (arXiv:2308.01937, as the referee verified), PINN and HNN.** SISA itself (`bourtoule2021machine`) is already cited.

### ⭐ Required "what would prove this wrong" check: is the positioning left with no claim?

**The positioning keeps a claim, but it is a different kind of claim. The novelty claim is gone, and nothing replaced it.**
- **What remains:** a *scope* claim (a naive unit compared with naive units, with no best-system claim) and the screen's counts, now labelled as counts and not as an emptiness claim. The contribution list (§1.3, items 1–4) never cited the emptiness sentence, and it stands unchanged: composition against read-path-matched rivals, store-scoped exact deletion, the maze, the ablations. The contribution sentences therefore **did not depend** on "the intersection is empty".
- **What is gone:** the paper no longer says, anywhere, that it is the first or only work to do this. The abstract's second sentence used to be a novelty sentence. It is now a scope sentence.
- **For the Head, before the trim pass:** if the paper needs a "why this is new" sentence, it must be written by the Head. I did not invent one, as instructed.

---

## (b) M2: replaced by the wave-20 rival set. It is PLACEHOLDERed, because the set has not landed

- **Location.** §3.1, paragraph *Two rival sets, and which is stronger*: `DRAFT-v2.tex` L502–513 and `DRAFT-v2-style.tex` L502–513. It sits directly after the existing `\TODO` on the cause of the gap. The placeholder is byte-identical in both files.
- **Before.** The paragraph described the two rival sets. No placeholder existed for their replacement.
- **After.** One new `\PLACEHOLDER`. **It names both rival sets and all three pools:**
  - the headline rivals of Table `tab:t7`, fit on **16,384** rows: the deletion study's declared build/smoke scale, reused as the headline comparators;
  - the ladder's rivals of Table `tab:ladder`, fit on **32,256** rows of the combination-split pool;
  - the store's own fit pool, **131,072** rows.

  It states what is pending: Table M-131k and Table M-16k from `tasks/w20/shapes-data-matched.md`, 3 seeds, with convergence evidence, and it says the leading table is not yet decided. **It says the result may land as a NEGATIVE:** the margin may shrink, vanish or reverse. **No matched number is written.**
- **Provenance.** The placeholder sits inside `SRCBLOCK: w12-deletion-baselines/FINDINGS.md (16,384); _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json (n_rows_fit 32256); w13-seeds-and-tables/TABLES.md (131,072: T9 reference refit of the vacuous scope reads the whole pool); ../../tasks/w20/shapes-data-matched.md`.
- **The existing M2 prose is kept.** This covers the two-set description, the "earlier set is stronger" comparison and the `\TODO`. It is all w ≤ 14 evidence. The placeholder says it *replaces* both sets, so the filler rewrites that paragraph when the data lands. I did not delete evidence ahead of the data.
- **Knock-ons.**
  - The placeholder index (App. `app:placeholders`, L2759) now reads **"Eleven rows remain pending"**, and the new row is listed first.
  - The header comment now says "Eleven PLACEHOLDERs remain: ten w19 …, one w20".

## (c) M5: split every shared-environment table and figure. Verified already done; no edit

The audit is `M5-ENV-AUDIT.md`, generated mechanically over all 44 floats and figure slots of `DRAFT-v2.tex`. It checks the maze environment names in each table's data rows and in each figure's caption.

- **Result: 0 floats hold rows from two environments.**
- **One heuristic flag, `tab:cmazet-sym`.** Read by hand, it is a cross-reference ("as on CMAZE-O") inside an untouchable w19 placeholder. It is not a CMAZE-O row.
- **The splits were already present at base 148f3b7:** `tab:req` / `tab:req2b`, `tab:cmazet` / `tab:cmazeo` and `tab:impl-maze` / `tab:impl`, and `fig_maze_requestlayer` had been removed. The claim at L886, "no two environments ever share a table", is now true as written.
- **Before → after:** no change.

## (d) M6: a seeds column on every table

- **Audit.** I read every `table` float's header row: 31 tables.
- **30 already had a `seeds` column.** Rows that are n = 1 already say `1`: `tab:del` per-seed rows, `tab:kcurve` middle rows, `tab:hole` "ours", `tab:req` and `tab:rivals-del` "1 / 3".
- **One table lacked it:** `tab:adapt` (App. `app:adapt`, L1804–1820). It had a per-row `seed` id but no count.

| where | before | after |
|---|---|---|
| `tab:adapt` column spec | `{llccccc}` | `{lclccccc}` |
| `tab:adapt` header | `seed & arm & enc. params trained & …` | `seed & seeds & arm & enc. params trained & …` |
| `tab:adapt` 9 rows | `0 & frozen & …` | `0 & 1 & frozen & …`, and so on. **Each row is one seed's arm-run, so each says n = 1.** |

The float is byte-identical in both files. "Seed count is a column of every table" (L362) is now true.

## (e) Pipeline language stripped. Two terms defined once. "Corrections absorbed" moved out

**Definitions, each made once in §2 and used thereafter:**
- **"pre-declared landscape controls"**: L349–352 in *Rivals: naive against naive*.
  - Before: "Pre-declared landscape controls (…) must collapse…".
  - After: "Four controls, which we call the *pre-declared landscape controls* throughout (…), must collapse…".
- **"pre-registered criterion"**: L359–361, new, at the end of *The judge and the estimand*. It reads: "Where an experiment was pre-registered, its success and falsification criteria were written before its first number; we refer to each as a *pre-registered criterion*, labelled as in that pre-registration (for example, criterion D or criterion S1)."

**Replacements.** All of these are in the rendered text. Every one was applied to both files; floats and paragraph headings are byte-identical across the two.

| id | where (DRAFT-v2.tex) | before | after |
|---|---|---|---|
| E1 | `tab:t7` LACE/PoE note | "LACE and PoE (w15 `lace-poe-rows`) are added" | "LACE and PoE are added" |
| E2 | §3.2 coverage control, L614 | "**Result (w15 `coverage-control`).**" | "**Result.**" |
| E3 | §3.6 paragraph heading | "LACE and PoE, at their own budgets (w15 `lace-poe-rows`)." | "LACE and PoE, at their own budgets." |
| E4 | same paragraph | "in the lace-poe-rows spoke's own words" | "in the words of that experiment's own record" |
| E5 | generation-table paragraph | "added to the generation-table registry" | "added to the arms of the generation table" |
| E6/E7 | same paragraph, L868–870 | "(bar S1 pass) … The pre-declared bar against the shipped read … fails" | "(pre-registered criterion S1, pass) … Pre-registered criterion S2, against the shipped read …, fails" |
| E8 | §4.4 intro | "bars were pre-registered" | "criteria were pre-registered" |
| E9, E11 | §4.4 CMAZE-T / CMAZE-O | "charter-C controls" | "pre-declared landscape controls" |
| E10 | §4.4 CMAZE-O, L1096 | "That rival margin is withdrawn **by the Hub**." | "We withdraw that rival margin." |
| E12 | `fig:maze-wm` caption | "the four charter-C controls" | "the four pre-declared landscape controls" |
| E13 | App. `app:kcurve` | "At three seeds (w15 `crossover-seeds`, 0 scored runs)." | "At three seeds (0 scored runs)." |
| E14 | App. `app:kcurve` | "w12's … comparability with w12" | "the ladder study's … comparability with that study" |
| E15, E17 | App. `app:encloop`, `app:encoder-main` headings | "(w15 `bar-d-score`)" | removed |
| E16 | App. `app:encloop` | "by the bar-d-score artefact" | "by the criterion-D scoring record" |
| E18 | App. `app:mazewm` | "the charter-C controls do not collapse" | "the pre-declared landscape controls do not collapse" |
| E19, E22 | `tab:cmazet`, `tab:cmazeo` rows | "all four charter-C controls" | "all four pre-declared landscape controls" |
| E20, E21 | `tab:cmazeo` caption and row | "WITHDRAWN BY THE HUB" | "WITHDRAWN" |
| E23 | `tab:cmazet-sym` caption | "(wave 19, `maze-t-symmetric`): pending." | ": pending." |
| E24 | App. `app:mazewm` | "validation bars passed" | "validation criteria passed" |
| E25 | `tab:gentable-kstar` caption | "Pre-declared bars: S1 …" | "Pre-registered criteria: S1 …" |
| E26 | `tab:crr` caption and header | "Pre-registered bars …"; header `bar` | "Pre-registered criteria …"; header `criterion` |
| E27, E28, E29 | §5.3 and App. `app:laws` | "pre-registered bars", "statements, bars and misses" | "…criteria…" |
| E30 | App. `app:verb` disclosures | "the gate file's declared primary" | "the pre-registration's declared primary" |
| E31 | App. `app:impl` | "Two caveats live in the gate file it cites" | "…in the pre-registration record it cites" |
| E32 | `tab:cmazet-sym` caption | "hold-outs, bars, controls" | "hold-outs, criteria, controls" |
| P1 | App. `app:placeholders` | "all ten await the w19 wave … (`maze-t-symmetric`). The rows that were pending when the w15 and w16 rows were first written in…" | "Eleven rows remain pending … each awaits a further run: the data-matched rival set … (one); … (three); and the seven rows of Table `tab:cmazet-sym`. Rows that were pending earlier are now written in…" |

**"Corrections absorbed" moved out of the submission.**
- **Before:** App. `app:corrections` sat in the source inside `\iffalse … \fi`. It was hidden from the PDF but still shipped in the `.tex`.
- **After:** the block has been removed from both files and replaced by a 3-line comment pointer. It is kept **verbatim** in `~/clu/outputs/w17-paper/CORRECTIONS-ABSORBED.tex`, with its provenance comments. The script asserted that the block was byte-identical in both drafts before the move.

**Left in place on purpose (declared):**
- **Placeholder arguments.** The 10 w19 placeholders keep "w16" / "w19" / "charter-C" inside their arguments, untouched by instruction. The new w20 placeholder names `w20 shapes-data-matched`, which is the key the filler greps. All placeholders must be gone before submission in any case.
- **The verbatim harness warning identifier** `\texttt{charter\_C\_control\_did\_not\_collapse}` (App. `app:mazewm`). It is quoted as a string the harness printed, and changing it would misquote an artefact.
- **LaTeX comments.** They never render. SRC paths necessarily carry wave names.
- **Generic threshold uses of "bar"**, such as "a bar of 0.575" or "the 0.15 bar". These are ordinary English for a threshold, not labelled "bar k" items.
- **"scored run k of 3" budget language.** This describes a pre-registration budget. I judged it not to be pipeline naming, but the Head may disagree.

## (f) K*: frozen-rule selection AND the plateau

- **Location:** §3 opening paragraph, L375–381, directly after "Every swept K satisfied the rule, so K\* = 32, the finest partition swept…".
- **Before:** the main text stated only the selection. The three-seed non-resolvability was in Appendix `app:kcurve` only.
- **After, added in the main text:**
  > That is the frozen rule's selection; on three seeds, however, the rule cannot separate the plateau K = 5…32: the seed standard deviation of validation composition exceeds the rule's 0.02 tolerance at K = 10, 16, 20, 24 and 28 (0.0211, 0.0203, 0.0285, 0.0235, 0.0276) and at K\* = 32 (0.0259), so three seeds do not distinguish K\* from its neighbours. The finding is the plateau itself: composition is flat across K to within its seed spread, while the honest four-part ledger falls to its minimum near K = 24, 2,094,130 / 2,080,090 / 2,085,810 B on seeds 0 / 1 / 2, a minimum that is resolvable (seed band 7,060 B against neighbour gaps of 12,203 B and 30,507 B; Appendix `app:kcurve`, Table `tab:kcurve3`).
- **Provenance:** every number carries `% SRC: w15-crossover-seeds/CROSSOVER-3SEED.md`. I checked each against that file: the sd column of the table at l.27; the per-seed K = 24 totals at l.71–73; the band and gaps at l.101.
- **Note on AMENDMENT-1.** Its rule that no w15 number enters the draft has already been superseded in practice: w18 `placeholder-fill` wrote the w15 crossover rows into App. `app:kcurve`. This item, from the Head's own task file, instructs me to use these numbers in the main text, and I followed the task file.

## (g) Deletion wording: UNTOUCHED, by instruction

The Head rewrites it in the methods: exact on the store *content*, with the address book carried, not refit. I made no edit to:
- the abstract sentence "Deletion is store-scoped and exact…";
- §1.2's "Exact deletion: … the deletable object equals…";
- §2's retained-statistic paragraph and its scope quote;
- §3.3 and the ethics statement.

`grep` of both diffs for `store-scoped|exactness|deletable object|carried|refit` finds only the new placeholder and the moved Corrections block. Referee M3 is therefore still open and belongs to the Head.

One more thing I left alone: §1.2 still cites SISA in "a learned generator composes but can delete exactly only by retraining (the whole model, or a shard of it)". That is deletion-framing wording, so I did not touch it. As a result SISA is cited in two sentences, one in related work and one in §1.2.

## The Head's items, NOT touched

- **AI-use statement.** The `\TODO` is unchanged. `\TODO` sequences are byte-identical before and after in both files (5 = 5).
- **Page trim.** Both PDFs are 46 pages including appendices. I did not trim.
- **Rest of venue compliance (M7 a–d, f):** the abstract deadline, the placeholder/TODO count at submission, and the bibliography pruning. The pruning is the job of the `w20-citation-prune` spoke, which runs after me.

---

## Invariants: shown

### Four style diffs: ALL EMPTY
From `style_diff_check.py DRAFT-v2.tex DRAFT-v2-style.tex`, exit 0. The full report is `STYLE-DIFF-CHECK-w20.md`.
1. **Numbers:** EMPTY DIFF.
2. **Placeholders:** 11 vs 11, ordered byte-identical: **True**. EMPTY DIFF. `\TODO` 5 vs 5, identical.
3. **Artefact comments:** artefact-path multiset EMPTY; provenance comment strings EMPTY; document order identical (410 vs 410).
4. **Section order:** 108 vs 108 headings, identical; 81 vs 81 labels, identical.

Extra checks: all 44 floats and figure slots are byte-identical, and the preamble is byte-identical. **Verdict: ALL DIFFS EMPTY.**

### The ten w19 placeholders: preserved
In both files, the 10 pre-existing `\PLACEHOLDER` arguments are byte-identical and in the same order as in `before-*/`. The only addition is the w20 placeholder, which is the 1st of 11 in document order.

### Compile: 0 undefined citations in both
`build-final-20260923T175525Z/`. The chain was `pdflatex` → `bibtex` (with `BIBINPUTS` = `BSTINPUTS` = `w17-paper/`, plus `iclr2027/` for the `.bst`) → `pdflatex` ×3. The counts below were read from the logs, not assumed.

| draft | pages | bibtex exit | `\bibitem`s in .bbl | .blg warnings/errors | **undefined citations** | undefined refs | TeX `!` errors |
|---|---|---|---|---|---|---|---|
| `DRAFT-v2.tex` | 46 | 0 | 63 | 0 | **0** | 0 | 0 |
| `DRAFT-v2-style.tex` | 46 | 0 | 63 | 0 | **0** | 0 | 0 |

### SHA-256 after
The hashes of `DRAFT-v2.tex`, `DRAFT-v2-style.tex` and `CORRECTIONS-ABSORBED.tex` are in `SHA256-after.txt`.

## 7. Line map (DRAFT-v2.tex / DRAFT-v2-style.tex)

| item | main | style |
|---|---|---|
| abstract | 90–94 | 95–98 |
| §1 / §1.1 / §1.2 headings | 125 / 129 / 198 | 125 / 129 / 196 |
| unit framing paragraph | 132–138 | 132–138 |
| systems sentences | 174–188 | 173–186 |
| definitions | 349 and 359–361 | 351 and 359–361 |
| K\* plateau | 375–381 | 375–381 |
| w20 placeholder | 503–512 | 503–512 |
| `tab:adapt` | 1804–1822 | 1790–1808 |
| placeholder index | 2759 | 2748 |
| Corrections pointer | 2800–2802 | 2789–2791 |

## What did NOT run / NOT done
- **No compute and no scored run.** The only commands were edits, checks and LaTeX compiles.
- **I did not visually inspect the rendered PDFs.** This box has no `pdftoppm` or `pdftotext`, and no pypdf. Rendering was checked through the compile logs (0 errors, 0 undefined) and by proofreading the source.
- **No bib entries were added** for CDM, PINN or HNN. They are named in prose with `cite-tagger` comments and left to `w20-citation-prune`.
- **The w20 `shapes-data-matched` result is not filled.** It has not landed. The placeholder names the sets and pools only.
- **M1, M3 and M4 as the referee wrote them** are not in my item list. M3 is item (g) and belongs to the Head. M4 is superseded by the Head's framing in (a). M1's fit-pool disclosure is partly carried by the existing *Two rival sets* paragraph and by the new placeholder.
- **The S-items** (SHOULD-FIX S1–S12) are not in my list and were not attempted.
