# w17 cite-tagger: FINDINGS

**Machine:** batmac (Apple M4). **Compute:** none. **Scored runs:** none. **Numbers added to either draft:** none. I inserted `\cite` commands only, with no w15 or w16 content and no self-citation.

## Deliverables (in `~/clu/outputs/w17-paper/`)
| file | what it is |
|---|---|
| `references.bib` | 96 entries, **generated** by `build_bib.py`. Sources: Crossref for the 48 DOI entries, the arXiv API for arXiv-only entries, and Semantic Scholar for the 6 venue confirmations the scout asked for and for Montero 2021 (DBLP is bot-walled from this box). Two entries come from other records: 3D Shapes from the dataset repository's own README BibTeX, and Montero 2021 from Semantic Scholar. `lecun2022path` is excluded because the scout marked it unverified. |
| `CITATIONS-FOR-ZOTERO.md` | Block 1: 76 arXiv IDs. Block 2: 48 DOIs. Each is one fenced block with one ID per line. Omitted counts are stated: 20 entries lack an arXiv ID, 48 lack a DOI, and 2 appear in neither block. |
| `DRAFT-v2.tex`, `DRAFT-v2-style.tex` | Each has **38 insertions** (32 `\citep`, 6 `\nocite`). The keys and their order are identical in both files. |
| `compile-v2.log`, `compile-v2-style.log` | Real runs of `pdflatex → bibtex → pdflatex ×3` against `iclr2027/`. Every step exited 0. Each produced a 35-page PDF with 62 references, and neither log has an undefined citation or reference or an error. |

## Proofs (in `~/clu/outputs/w17-cite-tagger/`)
- **Prose untouched** (`PROSE-CHECK.txt`): I stripped exactly the inserted `~\citep{…}` and `\nocite{…}` from each cited draft. Each result is **byte-identical** to its pre-cite snapshot (`pre-cite/`, SHA256 recorded). Word counts are equal (v2: 20,786 = 20,786; style: 21,103 = 21,103), and the word-level diff is 0.
- **Style invariants** (`STYLE-DIFF-CHECK-post-cite.md`): I re-ran the style-paraphrase spoke's own `style_diff_check.py` on the cited pair. Numbers, placeholders, artefact comments and section order all show an **EMPTY DIFF**.
- **Provenance**: draft-v2's `check_provenance.py` flags 3 extra "unsourced" tokens (`015`, `019`, `999`). These are the digits inside `\nocite{cao2015towards}`, `\nocite{ginart2019making}` and `\nocite{hall2002adding,hall1999adding}`. The checker's docstring says tokens inside `\cite` are ignored, but its regex covers only `cite[pt]?`. A copy of the checker (`check_provenance_nocite.py`) that differs only by adding `nocite` to that regex returns findings identical to the pre-cite drafts.

## Decisions (pre-authorised, closest to an ICLR main-track result)
1. **Citation style**: VENUE.md §8 specifies natbib author–year with round brackets. I inserted `~\citep{}` wherever the draft names a method or claim without an author–year.
2. **`\nocite` where the draft already prints a hand-typed author–year** (Tenenbaum and Freeman, 2000; Vasilescu and Terzopoulos, 2002; Vlasic et al., 2005; Cao and Yang (2015); Ginart et al., 2019; Hall, Marshall and Martin, 2002 plus Hall 1999). A visible cite there would print the author and year twice, and replacing the string would be a prose change. `\nocite` puts the work in the reference list and leaves the sentence unchanged. **For the Head's editing pass:** convert these hand-typed strings to `\citet`/`\citep`. That is a prose edit, which is outside my remit.
3. **No citation in the abstract.** This follows ICLR convention.
4. **Venues.** I confirmed ICLR 2020 for Dreamer, ICML 2018 for FactorVAE and NeurIPS 2020 for HiPPO on Semantic Scholar by venue and year. For Schott (ICLR 2022), RETRO (ICML 2022) and Hopfield-is-all-you-need (ICLR 2021), Semantic Scholar confirms the venue name but dates the paper by its preprint year, so the venue year is the scout's. Entries the scout marked "venue not verified" are `@misc` arXiv preprints, each with a `note`.
5. **Year source**: every DOI entry takes its year from Crossref's `issued` date. There is one mismatch with the key: `lake2017building` has Crossref year 2016 (the scout noted that BBS volume 40 is dated 2017).
6. **`efron1979bootstrap`**, which the scout marked conditional, is cited at "the paired estimand with a bootstrap interval". The draft's own artefact-sourced text says the interval is a bootstrap one (see also the appendix: "bootstrap 2.5% quantile").
7. **The LACE (`nie2021controllable`) and PoE (`hinton2002training`) method citations** sit inside the w15 `lace-poe-rows` placeholder, where the row is named, and not next to any result. This follows the scout's AMENDMENT-1 flag. The placeholder text is otherwise unchanged and is identical in both drafts.

## Keys in references.bib that the drafts do not cite (34): no prose slot exists
bourtoule2021machine chen2018neural chevalierboisvert2023minigrid cohen2021gradient ding2022attribute fan2024salun fang2026machine fodor1988connectionism frankle2019lottery gandikota2023erasing georgopoulos2022cluster graves2014neural graves2016hybrid gu1994stable gu2020hippo gu2022efficiently gu2024mamba hupkes2020compositionality karuvally2025exponential keysers2020measuring kidger2021equinox kingma2015adam kumari2023ablating lake2017building lake2018generalization lee2014incremental liang2025compositional liu2024machine locatello2019challenging mahajan2025compositional shen2020interpreting shen2021closed vasilescu2021causalx zhang2024unlearncanvas

⚠ **For the Head:** CITATIONS.md marks `gu2020hippo` and `karuvally2025exponential` as **must be cited** (CM-21). The draft has no sentence about retention guarantees or memory permanence to attach them to. The same holds for the NTM/DNC comparison (`graves2014neural`, `graves2016hybrid`), the MiniGrid negative (`chevalierboisvert2023minigrid`), SISA as rebuttal-kit-only (`bourtoule2021machine`), and the R-E positioning precedents (`cohen2021gradient`, `frankle2019lottery`, S4, Mamba, Neural ODE). Each needs a sentence before it can be cited, and writing that sentence is a prose change I may not make. `kingma2015adam` and `kidger2021equinox` would need confirmation from the harness, which this zero-compute spoke did not inspect.

## Names in the draft with no verified entry, left uncited (not fabricated)
- **NegGrad+** (appendix table `tab:unlearn` and §3.4): not in CITATIONS.md.
- **Blelloch–Golovin** stable-matching placement (appendix `app:shorts`): not in CITATIONS.md.
- **Sekhari Def. 3** (Corrections item 14): not in CITATIONS.md.
- **"Guo et al. Def. 1/2"** (Corrections item 14): deliberately not cited, because the sentence says those definitions do not exist.

## DECLARED NOT-DONE
- **No OpenReview fetch.** It returns 403. Montero 2021 therefore uses Semantic Scholar metadata, which has abbreviated first names, plus the scout's forum ID.
- **No DBLP.** It is bot-walled, so Semantic Scholar was used instead.
- **No harness inspection**, so Adam and Equinox stay uncited (U-7).
- **Not written by this spoke:** a JAX `@software` entry, since the draft does not name JAX and the scout did not fetch the URL, and LeCun 2022, since the scout marked it unverified.
