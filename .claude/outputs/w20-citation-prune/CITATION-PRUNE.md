# CITATION-PRUNE — w20 `citation-prune`: 96 → **55** references

**Spoke:** w20 `citation-prune` on `batmac`. The worktree is `CHLU-w20-citation-prune`, on branch `w20/citation-prune`, based on `base/iter8` = `148f3b7`. This spoke ran after `referee3-fixes` had been reaped.
**Edited (exclusive .tex lock):** `~/clu/outputs/w17-paper/` `DRAFT-v2.tex`, `DRAFT-v2-style.tex`, `references.bib`, `CITATIONS.md` and `CITATIONS-FOR-ZOTERO.md`.
**Pre-edit snapshot:** `before-20260923T181535Z/`. Its SHA-256 values equal referee3-fixes' `SHA256-after.txt`. **After:** `SHA256-after.txt`.
**Zero compute. Zero scored runs.** Network use was limited to Crossref and the arXiv API, to verify the five added identifiers.
**Pre-flight:** `tmux ls` showed only `authtest`, `hubhost` and `w20-citation-prune`. None of `w20-referee3-fixes`, `w18-placeholder-fill`, `w18-referee-4` or `w19-draft-figure-slots` was live, at the start or at install time.

## Result

| | before | after |
|---|---|---|
| `references.bib` entries | 96 | **55** |
| distinct keys cited by each draft | 63 (33 bib entries were never cited) | **55**; every bib entry is cited, and every cite resolves |
| `\citep` / `\nocite` commands per draft | 33 / 6 | 34 / 9 |
| Zotero Block 1 (arXiv) / Block 2 (DOI) | 76 / 48 | **37 / 31** |
| undefined citations, both drafts | 0 | **0** (checked in the logs, below) |

**How the 55 was reached:** 96 − 33 never-cited − 13 cited-for-colour + 5 added = 55.
- The 3 additions the Head required: CDM, PINN and HNN.
- 2 more additions that the same rule demands: Blelloch–Golovin and GMOR. The prose names both and credits both, but neither had a citation.

**55 is at the target.** Getting there cost no citation a referee will name, as argued key by key below. Going lower would. See "What would prove this wrong".

---

## KEPT (55)

Each key below is cited by both drafts. The category is from the task's KEEP list.

**Named prior art, multilinear / frozen-generator line (8).** For every entry except MLC-VAE, the prose names the work by author-year or acronym, so removing the cite would leave a named work unreferenced.
- `tenenbaum2000separating`: Tenenbaum–Freeman (KEEP).
- `vasilescu2002tensorfaces`: TensorFaces (KEEP).
- `vlasic2005face`: named "Vlasic et al., 2005".
- `georgopoulos2020multilinear`: MLC-VAE (KEEP).
- `harkonen2020ganspace`: named "GANSpace, 2020".
- `oldfield2021tensor`: named "TCA, 2021".
- `oldfield2023panda`: PandA (KEEP).
- `oldfield2024bilinear`: named "its 2024 journal version".

**Compositional generation / compositional diffusion (2).**
- `du2020compositional`: canonical compositional EBM, and the method family of the LACE/PoE rows.
- `liu2022compositional`: composable diffusion (KEEP: compositional diffusion).

**Exact unlearning and its contrast class (11).**
- `cao2015towards`: Cao–Yang (KEEP), owner of the downdate primitive.
- `ginart2019making`: named "Ginart et al., 2019".
- `xu2023netflix`: specific claim, bilinear matrix completion.
- `quan2026exact`: exact unlearning on frozen foundation models (KEEP).
- `bourtoule2021machine`: SISA (KEEP), cited ×2.
- `guo2020certified`: certified removal (KEEP), cited ×2.
- `golatkar2020eternal`: canonical approximate forgetting in deep nets. It is now the only approximate-deletion example besides Guo.
- `kong2022approximate`, `moon2024feature`, `kong2023data`, `sun2025generative`: each carries a specific claim about that work in prose ("unlearns a feature rather than rows", "states that redaction is a different task", "approximate by construction").

**Decremental-PCA line (3, KEEP).**
- `hall2002adding`: named "Hall, Marshall and Martin, 2002". It is the source of the "approximating / instability / no error analysis" description.
- `jin2014evd`, `lee2021fast`: the prose says "successors" (plural).

**Screen tool (1).**
- `priem2022openalex`: the crawl behind the positioning counts, which remain in the text as counts.

**World-model anchors (5).**
- `ha2018world`, `hafner2025mastering`: canonical learned world models.
- `valevski2025diffusion`: GameNGen; motivates rule-valid generation.
- `vafa2024evaluating`: exact-judge validity.
- `zhao2022toward`: compositional generalisation as a world-model requirement.

**Datasets and hold-out protocol (3, KEEP).**
- `burgess2018shapes`: 3D Shapes.
- `kim2018disentangling`: introduced 3D Shapes.
- `montero2021role`: provenance of the held-out split.

**Compositional-generalisation anchors (4).**
- `montero2022lost`, `schott2021visual`, `wiedemer2023compositional`, `lachapelle2023additive`: the interface split and the hold-out protocol.

**Baselines and their sources (4, KEEP).**
- `hochreiter1997long`: LSTM.
- `vaswani2017attention`: transformer.
- `nie2021controllable`: LACE.
- `hinton2002training`: PoE.

**Parameter-matching precedent (3, KEEP).**
- `borgeaud2021improving`: RETRO.
- `khandelwal2020generalization`: kNN-LM.
- `lample2019large`: product-key memory. It is also the precedent for a tensor-product address book.

**LDM-class positioning (5, KEEP).**
- `hopfield1982neural`, `krotov2016dense`, `ramsauer2021hopfield`, `hoover2023energy`.
- `kong2024reservoir`: must-cite under CM-21(b).

**Statistics (1).**
- `efron1979bootstrap`: the bootstrap interval named in the prose.

**ADDED by this spoke (5).** Every identifier was resolved this session; the raw responses are in `verify/`.

| key | work | arXiv | DOI | where, and how it is cited |
|---|---|---|---|---|
| `golatkar2023training` | Golatkar, Achille, Swaminathan, Soatto, *Training Data Protection with Compositional Diffusion Models* (CDM) | 2308.01937 (arXiv API: title and all 4 authors match) | **no DOI** (a Crossref search returned no match) | §1.1 systems sentence. `\nocite` inside the existing author parenthesis. |
| `raissi2019physics` | Raissi, Perdikaris, Karniadakis, *Physics-informed neural networks…*, J. Comput. Phys. 378:686–707 | **no arXiv** (the 2017 preprints 1711.10561/.10566 carry different titles and are not this paper) | 10.1016/j.jcp.2018.10.045 (Crossref: title, authors, volume and pages match) | §1.1 systems sentence. `\nocite` after "Karniadakis, 2019". |
| `greydanus2019hamiltonian` | Greydanus, Dzamba, Yosinski, *Hamiltonian Neural Networks*, NeurIPS 2019 | 1906.01563 (arXiv API; its comment gives NeurIPS 2019) | **no DOI** | §1.1 systems sentence. `\nocite` after "Yosinski, 2019". |
| `blelloch2007strongly` | Blelloch, Golovin, *Strongly History-Independent Hashing with Applications*, FOCS'07, pp. 272–282 | **no arXiv** (an arXiv title search returned 0 hits) | 10.1109/FOCS.2007.36 (Crossref match) | App. `app:shorts`: "the placement rule is the Blelloch--Golovin stable-matching table … the theorems follow from theirs and the fix-up cascade is theirs". This attribution is **mandatory** (shorts inventory N118 / P5-5), but it had **no citation**. Now `~\citep` after "stable-matching table". |
| `gellmann1968behavior` | Gell-Mann, Oakes, Renner, *Behavior of Current Divergences under SU3×SU3*, Phys. Rev. 175:2195–2199 | **no arXiv** (predates arXiv) | 10.1103/PhysRev.175.2195 (Crossref match) | App. `app:shorts`, the "GMOR on trained checkpoints" law. The relation was named by acronym with no citation. Now `~\citep` after "GMOR". |

I read these last two as "the shorts' key physics references". They are the only external works the shorts material in this draft names. The rest of the shorts material is the paper's own theory, cited to its artefacts.

---

## DROPPED (46)

### A. Cited once for colour: 13 removed from both drafts
Each was one entry in a list attached to a generic claim. Removing it leaves the claim cited, and no prose names it.

| key | where it was | why dropped |
|---|---|---|
| `nguyen2025survey` | §1.1 "Exact unlearning is also established" | A general survey. The sentence goes on to cite the specific exact results (Ginart, Xu, Quan), so the survey is colour. Its `~\citep{}` was removed whole. |
| `okawa2023compositional` | §1.1 "old and active problem", 4th of 4 | Colour. The claim stays cited by Du 2020 and composable diffusion. |
| `du2023reduce` | same list, 3rd of 4 | Colour here. The LACE sampler caveat it was meant for is not in the prose. |
| `hall1999adding` | `\nocite{hall2002adding,hall1999adding}` | The BMVC'99 conference original of the same work. The prose names only "…2002", and the journal version carries the quoted description. |
| `hafner2019dream` | §1.2 world-model list | Dreamer v1, superseded in the same list by DreamerV3 (`hafner2025mastering`). |
| `schrittwieser2020mastering` | §1.2 world-model list | MuZero concerns planning with a learned model and is not about generation. This is colour for a generic claim. |
| `micheli2023transformers` | §1.2 "states it has not seen" list | IRIS, one of 4 generative world models on a generic claim. `valevski2025diffusion` remains. |
| `bruce2024genie` | same list | Genie: same reason as IRIS. Its venue was also never verified. |
| `alonso2024diffusion` | same list | DIAMOND: same reason. `valevski2025diffusion` is the one that motivates rule-valid generation, so it is the one kept. |
| `izzo2021approximate` | §1.2 "approximate unlearning misses…" list | A second linear-model approximate deletion example beside certified removal (Guo), which is KEEP. |
| `wu2022memorizing` | §2 parameter-matching list, 4th of 5 | The precedent is carried by RETRO, kNN-LM and product-key memory, so a fourth memory-augmented example is colour. |
| `berges2024memory` | same list, 5th of 5 | Same reason. Its venue was also never verified. |
| `kipf2020contrastive` | §1.2 "Composition" definition | C-SWM, the second cite on a generic definition. `zhao2022toward` is the one that makes composition a world-model requirement. |

### B. In the bib but never cited by either draft: 33 removed from the bib
None of these was cited before this spoke, and none is named in the prose. Dropping them changes no rendered text.

| key | why dropped |
|---|---|
| `vasilescu2021causalx` | Never cited. ⚠ It is the "4 of 6 claim atoms" work in the screen count. See the Head note below. |
| `lee2014incremental` | Never cited. The paper makes no incremental-multilinear claim for it to pre-empt. |
| `georgopoulos2022cluster` | Never cited. A frozen-generator variant; the line is carried by GANSpace, TCA and PandA. |
| `ding2022attribute` | Never cited. Same frozen-generator line. |
| `shen2020interpreting` | Never cited. InterFaceGAN, same line. |
| `shen2021closed` | Never cited. SeFa, same line. |
| `liu2024machine` | Never cited. A survey of generative unlearning. |
| `fang2026machine` | Never cited. An approximate subspace-unlearning pre-empt; the draft makes no subspace-unlearning claim for it to pre-empt. |
| `gandikota2023erasing` | Never cited. The concept-erasure line; the draft makes no concept-erasure claim. |
| `kumari2023ablating` | Never cited. Same line. |
| `fan2024salun` | Never cited. Approximate unlearning. |
| `zhang2024unlearncanvas` | Never cited. A benchmark; the benchmark-requirements appendix names no benchmark. |
| `gu1994stable` | Never cited. A numerical primitive; the draft uses no rank-one eigen-update. |
| `locatello2019challenging` | Never cited. Disentanglement context. |
| `liang2025compositional` | Never cited. Theory; the draft does not carry the v1 §10.6 argument that needed it. |
| `mahajan2025compositional` | Never cited. Its companion. |
| `lake2018generalization` | Never cited. SCAN; the draft has no sequence-composition claim. |
| `hupkes2020compositionality` | Never cited. Taxonomy. |
| `keysers2020measuring` | Never cited. CFQ. |
| `fodor1988connectionism` | Never cited. Systematicity background. |
| `lake2017building` | Never cited. Background marked "optional" by the scout. |
| `chevalierboisvert2023minigrid` | Never cited. The draft does not mention MiniGrid. The scout's MiniGrid negative is not in the draft. |
| `gu2020hippo` | Never cited. CM-21(a) makes it must-cite **only if** the draft claims attention or SSMs lack a retention guarantee. It makes no such claim (grep: 0 hits for HiPPO / SSM retention). |
| `gu2022efficiently` | Never cited. S4 positioning precedent; the draft does not use it. |
| `gu2024mamba` | Never cited. Same. |
| `chen2018neural` | Never cited. Neural ODE positioning precedent; not used. |
| `cohen2021gradient` | Never cited. A grammar precedent; not used in prose. |
| `frankle2019lottery` | Never cited. Same. |
| `graves2014neural` | Never cited. CM-21(c) applies only if the draft says "continuity is our NTM/DNC escape hatch". It does not (0 hits). |
| `graves2016hybrid` | Never cited. Same. |
| `karuvally2025exponential` | Never cited. Must-cite only against a "tunable memory permanence" claim, which the draft does not make (0 hits). |
| `kingma2015adam` | Never cited. Conditional on the harness using Adam; the draft does not name the optimiser. |
| `kidger2021equinox` | Never cited. Software; the draft names no framework. |

---

## Proof: prose byte-identical after stripping cites (`PROSE-CHECK.txt`)
Template: w17 cite-tagger's `PROSE-CHECK.txt`. Script: `tools/prose_check.py`. It strips `~?\(no)cite*{…}` from BEFORE and AFTER, then compares.
- **Prose (LaTeX comments removed), both drafts: BYTE-IDENTICAL = YES.** Stripped-prose SHA-256 is equal before and after: `1525ae14…` for `DRAFT-v2.tex` and `0147544b…` for `DRAFT-v2-style.tex`. Word counts are 27,303 = 27,303 and 27,565 = 27,565.
- **Full file (comments included):** exactly **two** lines differ in each draft. These are the two `% cite-tagger: bib entr… needed … (not in references.bib)` comments that referee3-fixes left for this spoke. Both now say the entries were added and name the keys. The script **asserts** that no other line differs.

## Compile: 0 undefined citations in both (`COMPILE-SUMMARY.txt`, `compile-*.log`, `build-final-*/`)
The chain was `pdflatex` → `bibtex` (with `BIBINPUTS` = `BSTINPUTS` = `w17-paper/`, `w17-paper/iclr2027/`) → `pdflatex` ×3. All counts were **read from the logs**.

| draft | pages | bibtex exit | `\bibitem`s | .blg warnings | **undefined citations** | undefined refs | TeX `!` errors |
|---|---|---|---|---|---|---|---|
| `DRAFT-v2.tex` | 45 (was 46: shorter bibliography) | 0 | **55** | 0 (`warning$ -- 0`) | **0** | 0 | 0 |
| `DRAFT-v2-style.tex` | 46 | 0 | **55** | 0 | **0** | 0 | 0 |

Extra check: in both drafts, the set of `\bibitem` keys in the `.bbl` **equals** the set of `references.bib` keys (55 = 55). Every entry is used and every cite resolves.

## Four style diffs: ALL EMPTY (`STYLE-DIFF-CHECK-w20-prune.md`, exit 0)
1. Numbers: **EMPTY**.
2. Placeholders: 11 vs 11, ordered and byte-identical: **EMPTY**. `\TODO`: 5 vs 5, identical.
3. Artefact comments: **EMPTY**. Order is identical (410 vs 410).
4. Section order: **EMPTY**. Headings are 108 = 108 and labels 81 = 81.

Floats (44 = 44) and the preamble are byte-identical.

**Placeholders untouched** (`PLACEHOLDER-CHECK.txt`): every `\PLACEHOLDER{…}` and `\TODO{…}` (with nested braces) is byte-identical before and after, in order, in both files. This covers the 10 w19 placeholders, the 1 w20 placeholder, and the macro definitions.

## Regenerated files
- `references.bib`: 55 entries. The 50 kept entries are byte-identical to the w17 cite-tagger's, in the same order; the 5 new entries are appended; the header is updated. Diff: `references.prune.diff`.
- `CITATIONS-FOR-ZOTERO.md`: regenerated by the w17 `make_zotero.py`, with only its header line changed. **Block 1:** 37 arXiv IDs, one per line, one fenced block. **Block 2:** 31 DOIs, one per line, one fenced block. The omitted counts are stated beneath: 18 of 55 have no arXiv, 24 of 55 have no DOI, and 2 are in neither (burgess2018shapes, montero2021role). Every ID in the blocks is either in the pre-prune (scout-verified) blocks or one of the five verified this session. The check found 0 others and 0 duplicates.
- `CITATIONS.md`: pruned to exactly the 55 bib keys, checked by a script. Added: a header note, §1h (CDM, PINN, HNN), §5b (Blelloch–Golovin, GMOR), and an after-prune line in the summary counts. The full pre-prune file is in `before-*/`.
- Draft diffs: `DRAFT-v2.prune.diff` and `DRAFT-v2-style.prune.diff`, 28 changed lines each. The edits are identical across the two files.

## ⚠ For the Head (decisions I did not take)
1. **CausalX (`vasilescu2021causalx`) is the unnamed "4 of 6 claim atoms" work.** The screen paragraph says "the highest number of the six claim atoms reached by any single work is 4 of 6". A referee may ask which work that is. It was never cited, and I did not add it: this spoke's rule was to drop uncited colour, and naming it would mean new prose. If the Head wants it, adding a `~\citep{vasilescu2021causalx}` after "any single work" would change no prose and would make 56 entries. The entry is verified and is in `before-*/references.bib`.
2. **The CM-21 must-cites (HiPPO, NTM/DNC, EDEN) are dropped because the draft makes none of the claims that trigger them.** If any later edit writes a retention-guarantee, NTM-escape-hatch or memory-permanence sentence, the matching entry must come back. All of them are in `before-*/references.bib`.
3. **GMOR and Blelloch–Golovin** were added on my own reading of "the three shorts' key physics references". Both were uncited named attributions in the prose, and one of them (Blelloch–Golovin) is marked mandatory by the shorts inventory. If the Head reads that KEEP item differently, both are single `~\citep`s and easy to revert.

## What would prove this wrong
- **55 is the smallest defensible number, not a cut to hit the target.** The next candidates would each cost something a referee could point at:
  - `hoover2023energy` is an LDM-class positioning paper, which the task marks KEEP.
  - One of `jin2014evd` / `lee2021fast` would leave "successors" (plural) with one cite, and the decremental-PCA line is KEEP.
  - `golatkar2020eternal` would leave "approximate unlearning" illustrated only by certified removal.
  - `sun2025generative` or `kong2023data` would leave a sentence describing a specific paper with no reference.
  - `hafner2025mastering` or `ha2018world` would thin the world-model anchors to one.
- **The 13 colour drops are judgement calls.** A referee might still name DreamerV3's predecessor, MuZero or Genie in a world-model paragraph. None of them carried a claim in our prose. All 13 remain in `before-*/references.bib`, so each is a one-key revert.

## What did NOT run / NOT done
- **No compute and no scored run.** The only operations were file edits, checks, LaTeX compiles, and Crossref/arXiv API lookups for 5 identifiers.
- **The rendered PDFs were not inspected visually.** This box has no `pdftotext` or `pdftoppm`. Rendering was checked through the logs (0 errors, 0 undefined) and the `.bbl` content.
- **The 50 kept entries were not re-verified.** Their identifiers are the w17 citation-scout's, which were resolved in that session, and are byte-identical here.
- **No prose edits.** The positioning, deletion wording (M3), AI-use `\TODO` and page trim are the Head's.
- **Not touched:** the `\bibliography` comment "this draft carries no \cite yet" (line 1253), which is stale but predates this spoke; and the AMENDMENT-1 tension over w15 numbers that referee3-fixes noted.
- **`OpenReview` was not fetched** (it is blocked, as before). No kept or added entry depends on it except `montero2021role`, which is unchanged.
