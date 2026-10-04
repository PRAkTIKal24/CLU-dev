# FINDINGS — pw1-draft-citations (cite against the Head's Zotero bib `refs_clu_iclr.bib`, both drafts)

Session `pw1-draft-citations` · branch `pw1/draft-citations` (from `pw1/draft-template` a4de290) · batmac · start 2026-09-25T14:46:03Z.
Zero runs, zero fits, zero GPU. The bib was never written, and its sha256 is unchanged in both places (§7). Stage: `stage-145204/` (V0 = live copy, V2 = installed).

## Status

| brief item | status | one line |
|---|---|---|
| 1. Keys | **done** | Hub map verified independently: 55/55 pairs agree on DOI, arXiv ID or normalised title (0 without an agreeing field). 72 key occurrences renamed per draft; 0 old keys left on code |
| 2. `\citet` / `\citep` | **done** | 56 citation commands per draft: 51 `\citep`, 2 `\citet`, 3 `\citealp` (see choice C4). 9 `\nocite` + hand-typed author-year pairs converted, 14 sites (H1–H14). 0 citations at a line start; 0 doubled `\citet` names |
| 3. Right locations | **done**, with 3 logged exceptions | 13 key occurrences added at 8 sites: 2 at first main-text mention (M1 LSTM/transformer, M2 LACE/PoE), 6 at first appendix mention (P1–P6). Missing-from-the-bib list in §4. Exceptions: the abstract, section headings, and the Sec. 1.1 sentence that defers its citations to the next paragraph (C1–C3) |
| 4. Bibliography lines | **done** | `\IfFileExists{refs_clu_iclr.bib}` + `\bibliography{refs_clu_iclr}`; `\bibliographystyle{iclr2027/iclr2027_conference}` unchanged. One added preamble option, B3 (see C6) |
| 5. Every entry cited | **done** | 55/55 cited before and after; 0 uncited; 0 cited keys missing from the bib. Per-entry counts in §3 |
| Lockstep | **done** | citation sequence, forms, keys and rendered strings identical in both drafts (table compare) |
| Installed | **done** | both `.tex` and both PDFs (+ .aux/.bbl/.blg/.log/.out) cmp-equal to stage V2; a fresh rebuild of the installed tree reproduces .aux/.bbl/.out and the PDF text |

## Choices (logged; closest to an ICLR main-track paper)
- **C1** No citation in the abstract. ICLR abstracts carry none, and its maze sentence is the Head's `\TODO`. 0 citations in the abstract after the pass.
- **C2** No citation in a section heading (it would go into the PDF bookmarks). "world models" first appears in the Sec. 1.2 heading, and the next sentence carries `\citep{ha_world_2018,hafner_mastering_2025}`.
- **C3** The Sec. 1.1 opening sentence ("sharded retraining, compositional diffusion and physics-informed variants … each is cited in one sentence below") gets no citation. The sentence itself defers them to the next paragraph, where SISA, compartmentalised diffusion, PINNs and HNNs are cited. Citing here would contradict the sentence, and no sentence may be rewritten.
- **C4** `\citealp` (natbib's `\citep` without brackets) is used only where a bare hand-typed "Author, Year" is an item inside an existing parenthesis: Tenenbaum & Freeman, Vlasic et al., Hall et al. A `\citep` there would print "((…))". A named method in such a list gets `Method~\citep{…}`, and the hand-typed year it duplicates goes (MLC-VAE, GANSpace, TCA, PandA). "its 2024 journal version~\citep{oldfield_bilinear_2024}" keeps "2024": it is descriptive and was already a `\citep`.
- **C5** "Blelloch--Golovin stable-matching table" uses the authors as a noun modifier ("X's table"), so it becomes `the~\citet{blelloch_strongly_2007} stable-matching table` → "the Blelloch & Golovin (2007) stable-matching table"; the hand-typed names are removed.
- **C6** `\PassOptionsToPackage{hyphens}{url}` is prepended to the existing `\usepackage{hyperref}` line (line count unchanged). The Head's `lee_fast_2021` has a 222-character URL that printed a **133 pt overfull box** in the references (stage V1). With URL breaks at hyphens, 0 overfull boxes. The bib is untouched. This is the one non-citation source change.
- **C7** "First mention in the appendices" = the first mention across the appendix block, captions included, with one citation per work there.
- **C8** Every rule keeps the line count. When a replaced span held a line break, the break moves to the nearest earlier space, never directly before a citation command.

## 1. Key map, old → new → matching field (independent check; `KEYMAP-VERIFY.md`)
| # | old key | new key | verdict |
|---|---|---|---|
| 1 | `blelloch2007strongly` | `blelloch_strongly_2007` | MATCH on DOI + title |
| 2 | `borgeaud2021improving` | `borgeaud_improving_2022` | MATCH on arXiv + title |
| 3 | `bourtoule2021machine` | `bourtoule_machine_2021` | MATCH on DOI + title |
| 4 | `burgess2018shapes` | `3dshapes18` | MATCH on title |
| 5 | `cao2015towards` | `cao_towards_2015` | MATCH on DOI + title |
| 6 | `du2020compositional` | `du_compositional_2020` | MATCH on arXiv + title |
| 7 | `efron1979bootstrap` | `efron_bootstrap_1979` | MATCH on DOI + title |
| 8 | `gellmann1968behavior` | `gell-mann_behavior_1968` | MATCH on DOI + title |
| 9 | `georgopoulos2020multilinear` | `georgopoulos_multilinear_2020` | MATCH on arXiv + title |
| 10 | `ginart2019making` | `ginart_making_2019` | MATCH on arXiv + title |
| 11 | `golatkar2020eternal` | `golatkar_eternal_2020` | MATCH on DOI + title |
| 12 | `golatkar2023training` | `golatkar_training_2024` | MATCH on arXiv + title |
| 13 | `greydanus2019hamiltonian` | `greydanus_hamiltonian_2019` | MATCH on arXiv + title |
| 14 | `guo2020certified` | `guo_certified_2023` | MATCH on arXiv + title |
| 15 | `ha2018world` | `ha_world_2018` | MATCH on DOI + title |
| 16 | `hafner2025mastering` | `hafner_mastering_2025` | MATCH on DOI + title |
| 17 | `hall2002adding` | `hall_adding_2002` | MATCH on DOI + title |
| 18 | `harkonen2020ganspace` | `harkonen_ganspace_2020` | MATCH on arXiv + title |
| 19 | `hinton2002training` | `hinton_training_2002` | MATCH on DOI + title |
| 20 | `hochreiter1997long` | `hochreiter_long_1997` | MATCH on DOI + title |
| 21 | `hoover2023energy` | `hoover_energy_2023` | MATCH on DOI + title |
| 22 | `hopfield1982neural` | `hopfield_neural_1982` | MATCH on DOI + title |
| 23 | `jin2014evd` | `jin_evd_2014` | MATCH on DOI + title |
| 24 | `khandelwal2020generalization` | `khandelwal_generalization_2020` | MATCH on arXiv + title |
| 25 | `kim2018disentangling` | `kim_disentangling_2019` | MATCH on arXiv + title |
| 26 | `kong2022approximate` | `kong_approximate_2022` | MATCH on arXiv + title |
| 27 | `kong2023data` | `kong_data_2023` | MATCH on DOI + title |
| 28 | `kong2024reservoir` | `kong_reservoir-computing_2024` | MATCH on DOI + title |
| 29 | `krotov2016dense` | `krotov_dense_2016` | MATCH on arXiv + title |
| 30 | `lachapelle2023additive` | `lachapelle_additive_2023` | MATCH on DOI + title |
| 31 | `lample2019large` | `lample_large_2019` | MATCH on arXiv + title |
| 32 | `lee2021fast` | `lee_fast_2021` | MATCH on DOI + title |
| 33 | `liu2022compositional` | `avidan_compositional_2022` | MATCH on DOI + title |
| 34 | `montero2021role` | `montero2021role` | MATCH on title |
| 35 | `montero2022lost` | `montero_lost_2022` | MATCH on DOI + title |
| 36 | `moon2024feature` | `moon_feature_2024` | MATCH on DOI + title |
| 37 | `nie2021controllable` | `nie_controllable_2021` | MATCH on arXiv + title |
| 38 | `oldfield2021tensor` | `oldfield_tensor_2021` | MATCH on DOI + title |
| 39 | `oldfield2023panda` | `oldfield_panda_2023` | MATCH on arXiv + title |
| 40 | `oldfield2024bilinear` | `oldfield_bilinear_2024` | MATCH on DOI + title |
| 41 | `priem2022openalex` | `priem_openalex_2022` | MATCH on arXiv + title |
| 42 | `quan2026exact` | `ceci_exact_2027` | MATCH on DOI + title |
| 43 | `raissi2019physics` | `raissi_physics-informed_2019` | MATCH on DOI + title |
| 44 | `ramsauer2021hopfield` | `ramsauer_hopfield_2021` | MATCH on arXiv + title |
| 45 | `schott2021visual` | `schott_visual_2022` | MATCH on arXiv + title |
| 46 | `sun2025generative` | `sun_generative_2025` | MATCH on DOI + title |
| 47 | `tenenbaum2000separating` | `tenenbaum_separating_2000` | MATCH on DOI + title |
| 48 | `vafa2024evaluating` | `vafa_evaluating_2024` | MATCH on DOI + title |
| 49 | `valevski2025diffusion` | `valevski_diffusion_2025` | MATCH on arXiv + title |
| 50 | `vasilescu2002tensorfaces` | `goos_multilinear_2002` | MATCH on DOI + title |
| 51 | `vaswani2017attention` | `vaswani_attention_2023` | MATCH on arXiv + title |
| 52 | `vlasic2005face` | `vlasic_face_2005` | MATCH on DOI + title |
| 53 | `wiedemer2023compositional` | `wiedemer_compositional_2023` | MATCH on DOI + title |
| 54 | `xu2023netflix` | `xu_netflix_2023` | MATCH on arXiv + title |
| 55 | `zhao2022toward` | `zhao_toward_2022` | MATCH on arXiv + title |


Year / author-string differences, old bib → new bib (these change the rendered author-year; reported, bib untouched): `golatkar_training_2024` 2023 → **2024**; `guo_certified_2023` 2020 → **2023**; `kim_disentangling_2019` 2018 → **2019**; `ceci_exact_2027` (old `quan2026exact`) 2026 → **2027**; `vaswani_attention_2023` 2017 → **2023** (the arXiv v-date: "Attention Is All You Need" now renders as Vaswani et al., 2023). Full table: `AUTHOR-YEAR-COMPARE.md`.

## 2. Every citation, before → after, with its sentence (DRAFT-v2 line numbers; the style draft is identical in keys, forms and rendered strings)
| # | line | region | before (old key; hand-typed text shown) | after | rendered (from .bbl) | sentence (source, comments stripped) |
|---|---|---|---|---|---|---|
| 1 | 145 | main | `(added)` | `\citep{hochreiter_long_1997}` | (Hochreiter & Schmidhuber, 1997) | The naive-CLU (n-CLU) is a \emph{naive AI unit}: one model with no system built around it, in the sense in which an MLP, an LSTM~\citep{hochreiter_long_1997} or a transformer~\citep{vaswani_attention_2023} is a unit. |
| 2 | 145 | main | `(added)` | `\citep{vaswani_attention_2023}` | (Vaswani et al., 2023) | The naive-CLU (n-CLU) is a \emph{naive AI unit}: one model with no system built around it, in the sense in which an MLP, an LSTM~\citep{hochreiter_long_1997} or a transformer~\citep{vaswani_attention_2023} is a unit. |
| 3 | 152 | main | `\citep{du2020compositional,liu2022compositional}` | `\citep{du_compositional_2020,avidan_compositional_2022}` | (Du et al., 2020; Liu et al., 2022) | Generating held-out combinations of factors is an old and active problem~\citep{du_compositional_2020,avidan_compositional_2022}. |
| 4 | 153 | main | `H1: (Tenenbaum and Freeman, 2000\nocite{tenenbaum2000separating};` | `\citealp{tenenbaum_separating_2000}` | Tenenbaum & Freeman, 2000 | The multilinear line (\citealp{tenenbaum_separating_2000}; TensorFaces~\citep{goos_multilinear_2002}; \citealp{vlasic_face_2005}; MLC-VAE~\citep{georgopoulos_multilinear_2020}) factorises a data tensor organised by every label combination and renders linearly; |
| 5 | 154 | main | `H2: TensorFaces, Vasilescu and Terzopoulos, 2002\nocite{vasilescu2002tensorfaces}` | `\citep{goos_multilinear_2002}` | (Vasilescu & Terzopoulos, 2002) | The multilinear line (\citealp{tenenbaum_separating_2000}; TensorFaces~\citep{goos_multilinear_2002}; \citealp{vlasic_face_2005}; MLC-VAE~\citep{georgopoulos_multilinear_2020}) factorises a data tensor organised by every label combination and renders linearly; |
| 6 | 154 | main | `H3: Vlasic et al., 2005\nocite{vlasic2005face}` | `\citealp{vlasic_face_2005}` | Vlasic et al., 2005 | The multilinear line (\citealp{tenenbaum_separating_2000}; TensorFaces~\citep{goos_multilinear_2002}; \citealp{vlasic_face_2005}; MLC-VAE~\citep{georgopoulos_multilinear_2020}) factorises a data tensor organised by every label combination and renders linearly; |
| 7 | 154 | main | `H4: MLC-VAE, 2020~\citep{georgopoulos2020multilinear}` | `\citep{georgopoulos_multilinear_2020}` | (Georgopoulos et al., 2020) | The multilinear line (\citealp{tenenbaum_separating_2000}; TensorFaces~\citep{goos_multilinear_2002}; \citealp{vlasic_face_2005}; MLC-VAE~\citep{georgopoulos_multilinear_2020}) factorises a data tensor organised by every label combination and renders linearly; |
| 8 | 155 | main | `H5: GANSpace, 2020~\citep{harkonen2020ganspace}` | `\citep{harkonen_ganspace_2020}` | (Härkönen et al., 2020) | The multilinear line (\citealp{tenenbaum_separating_2000}; TensorFaces~\citep{goos_multilinear_2002}; \citealp{vlasic_face_2005}; MLC-VAE~\citep{georgopoulos_multilinear_2020}) factorises a data tensor organised by every label combination and renders linearly; |
| 9 | 155 | main | `H6: TCA, 2021~\citep{oldfield2021tensor}` | `\citep{oldfield_tensor_2021}` | (Oldfield et al., 2021) | The multilinear line (\citealp{tenenbaum_separating_2000}; TensorFaces~\citep{goos_multilinear_2002}; \citealp{vlasic_face_2005}; MLC-VAE~\citep{georgopoulos_multilinear_2020}) factorises a data tensor organised by every label combination and renders linearly; |
| 10 | 155 | main | `H7: PandA, 2023~\citep{oldfield2023panda}` | `\citep{oldfield_panda_2023}` | (Oldfield et al., 2023) | The multilinear line (\citealp{tenenbaum_separating_2000}; TensorFaces~\citep{goos_multilinear_2002}; \citealp{vlasic_face_2005}; MLC-VAE~\citep{georgopoulos_multilinear_2020}) factorises a data tensor organised by every label combination and renders linearly; |
| 11 | 156 | main | `\citep{oldfield2024bilinear}` | `\citep{oldfield_bilinear_2024}` | (Oldfield et al., 2024) | The multilinear line (\citealp{tenenbaum_separating_2000}; TensorFaces~\citep{goos_multilinear_2002}; \citealp{vlasic_face_2005}; MLC-VAE~\citep{georgopoulos_multilinear_2020}) factorises a data tensor organised by every label combination and renders linearly; |
| 12 | 160 | main | `H8: form of Cao and Yang (2015)\nocite{cao2015towards}` | `\citet{cao_towards_2015}` | Cao & Yang (2015) | In particular we do not claim multilinear or bilinear factored generation, a request-indexed tensor head, tensor-product label conditioning, or the downdate primitive, which follows in one line from the summation form of~\citet{cao_towards_2015} because a cova |
| 13 | 162 | main | `H9: $k$-means (Ginart et al., 2019\nocite{ginart2019making})` | `\citep{ginart_making_2019}` | (Ginart et al., 2019) | Unlearning that matches a retrain without the deleted rows is also established, for $k$-means~\citep{ginart_making_2019}, for bilinear matrix completion~\citep{xu_netflix_2023}, and for ridge heads on frozen backbones~\citep{ceci_exact_2027}, where the most re |
| 14 | 163 | main | `\citep{xu2023netflix}` | `\citep{xu_netflix_2023}` | (Xu et al., 2023) | Unlearning that matches a retrain without the deleted rows is also established, for $k$-means~\citep{ginart_making_2019}, for bilinear matrix completion~\citep{xu_netflix_2023}, and for ridge heads on frozen backbones~\citep{ceci_exact_2027}, where the most re |
| 15 | 163 | main | `\citep{quan2026exact}` | `\citep{ceci_exact_2027}` | (Quan et al., 2027) | Unlearning that matches a retrain without the deleted rows is also established, for $k$-means~\citep{ginart_making_2019}, for bilinear matrix completion~\citep{xu_netflix_2023}, and for ridge heads on frozen backbones~\citep{ceci_exact_2027}, where the most re |
| 16 | 165 | main | `\citep{kong2022approximate}` | `\citep{kong_approximate_2022}` | (Kong & Alfeld, 2022) | The nearest generative attempts are approximate~\citep{kong_approximate_2022}: feature unlearning for pretrained GANs and VAEs~\citep{moon_feature_2024} unlearns a feature rather than rows and fine-tunes the generator; data redaction from pretrained GANs~\cite |
| 17 | 166 | main | `\citep{moon2024feature}` | `\citep{moon_feature_2024}` | (Moon et al., 2024) | The nearest generative attempts are approximate~\citep{kong_approximate_2022}: feature unlearning for pretrained GANs and VAEs~\citep{moon_feature_2024} unlearns a feature rather than rows and fine-tunes the generator; data redaction from pretrained GANs~\cite |
| 18 | 167 | main | `\citep{kong2023data}` | `\citep{kong_data_2023}` | (Kong & Chaudhuri, 2023) | The nearest generative attempts are approximate~\citep{kong_approximate_2022}: feature unlearning for pretrained GANs and VAEs~\citep{moon_feature_2024} unlearns a feature rather than rows and fine-tunes the generator; data redaction from pretrained GANs~\cite |
| 19 | 168 | main | `\citep{sun2025generative}` | `\citep{sun_generative_2025}` | (Sun et al., 2025) | The nearest generative attempts are approximate~\citep{kong_approximate_2022}: feature unlearning for pretrained GANs and VAEs~\citep{moon_feature_2024} unlearns a feature rather than rows and fine-tunes the generator; data redaction from pretrained GANs~\cite |
| 20 | 171 | main | `\citep{priem2022openalex}` | `\citep{priem_openalex_2022}` | (Priem et al., 2022) | A forward-citation crawl~\citep{priem_openalex_2022} over every title and abstract citing the canonical works of either side screened approximately 4,050 unique works: 3,038 citing the generation-side anchors, 759 citing the deletion-side anchors, 254 from 21  |
| 21 | 187 | main | `\citep{bourtoule2021machine}` | `\citep{bourtoule_machine_2021}` | (Bourtoule et al., 2021) | These are the screen's counts, not a claim that no such model exists: abstract-level screening cannot see a match buried in a method section, and indexing of preprints younger than about three months is incomplete (Sec.~\ref{sec:limits}). Three families of sys |
| 22 | 191 | main | `H10: models (Golatkar, Achille, Swaminathan and Soatto\nocite{golatkar2023training})` | `\citep{golatkar_training_2024}` | (Golatkar et al., 2024) | These are the screen's counts, not a claim that no such model exists: abstract-level screening cannot see a match buried in a method section, and indexing of preprints younger than about three months is incomplete (Sec.~\ref{sec:limits}). Three families of sys |
| 23 | 197 | main | `H11: networks (Raissi, Perdikaris and Karniadakis, 2019\nocite{raissi2019physics})` | `\citep{raissi_physics-informed_2019}` | (Raissi et al., 2019) | Physics-informed variants of an MLP, such as physics-informed neural networks~\citep{raissi_physics-informed_2019} and Hamiltonian neural networks~\citep{greydanus_hamiltonian_2019}, are systems that add a physical loss or structure to an MLP unit, and the bas |
| 24 | 197 | main | `H12: networks (Greydanus, Dzamba and Yosinski, 2019\nocite{greydanus2019hamiltonian})` | `\citep{greydanus_hamiltonian_2019}` | (Greydanus et al., 2019) | Physics-informed variants of an MLP, such as physics-informed neural networks~\citep{raissi_physics-informed_2019} and Hamiltonian neural networks~\citep{greydanus_hamiltonian_2019}, are systems that add a physical loss or structure to an MLP unit, and the bas |
| 25 | 203 | main | `H13: (Hall, Marshall and Martin, 2002\nocite{hall2002adding},` | `\citealp{hall_adding_2002}` | Hall et al., 2002 | The decremental-PCA line (\citealp{hall_adding_2002}, and successors~\citep{jin_evd_2014,lee_fast_2021}) subtracts \emph{deflated} eigenspaces; its authors describe the result as approximating the refit, report that subtraction tends to instability, and give n |
| 26 | 203 | main | `\citep{jin2014evd,lee2021fast}` | `\citep{jin_evd_2014,lee_fast_2021}` | (Jin et al., 2014; Lee, 2021) | The decremental-PCA line (\citealp{hall_adding_2002}, and successors~\citep{jin_evd_2014,lee_fast_2021}) subtracts \emph{deflated} eigenspaces; its authors describe the result as approximating the refit, report that subtraction tends to instability, and give n |
| 27 | 213 | main | `\citep{ha2018world,hafner2025mastering}` | `\citep{ha_world_2018,hafner_mastering_2025}` | (Ha & Schmidhuber, 2018; Hafner et al., 2025) | A world model~\citep{ha_world_2018,hafner_mastering_2025} is useful to the extent that it can be asked for states and transitions it has not seen~\citep{valevski_diffusion_2025}, and trusted with the answer. |
| 28 | 214 | main | `\citep{valevski2025diffusion}` | `\citep{valevski_diffusion_2025}` | (Valevski et al., 2025) | A world model~\citep{ha_world_2018,hafner_mastering_2025} is useful to the extent that it can be asked for states and transitions it has not seen~\citep{valevski_diffusion_2025}, and trusted with the answer. |
| 29 | 216 | main | `\citep{vafa2024evaluating}` | `\citep{vafa_evaluating_2024}` | (Vafa et al., 2024) | We name three properties and use them throughout. \textbf{Rule validity}: an output is valid if the task's judge accepts it (on the maze, the engine's judge, which uses integer arithmetic with no tolerance)~\citep{vafa_evaluating_2024}; approximate similarity  |
| 30 | 218 | main | `\citep{zhao2022toward}` | `\citep{zhao_toward_2022}` | (Zhao et al., 2022) | We name three properties and use them throughout. \textbf{Rule validity}: an output is valid if the task's judge accepts it (on the maze, the engine's judge, which uses integer arithmetic with no tolerance)~\citep{vafa_evaluating_2024}; approximate similarity  |
| 31 | 222 | main | `\citep{bourtoule2021machine}` | `\citep{bourtoule_machine_2021}` | (Bourtoule et al., 2021) | The three pull against each other: a lookup table deletes in this sense but composes nothing it has not written; a learned generator composes but meets this definition only by retraining (the whole model, or a shard of it~\citep{bourtoule_machine_2021}); and a |
| 32 | 223 | main | `\citep{guo2020certified,golatkar2020eternal}` | `\citep{guo_certified_2023,golatkar_eternal_2020}` | (Guo et al., 2023; Golatkar et al., 2020) | The three pull against each other: a lookup table deletes in this sense but composes nothing it has not written; a learned generator composes but meets this definition only by retraining (the whole model, or a shard of it~\citep{bourtoule_machine_2021}); and a |
| 33 | 232 | main | `\citep{burgess2018shapes,kim2018disentangling}` | `\citep{3dshapes18,kim_disentangling_2019}` | (Burgess & Kim, 2018; Kim & Mnih, 2019) | item \textbf{Composition at a stated partition, against data-matched, read-path-matched baselines.} On a Shapes3D~\citep{3dshapes18,kim_disentangling_2019} value hold-out at $\Kstar=32$ wells, three scored seeds, with every pre-declared landscape control colla |
| 34 | 323 | main | `\citep{wiedemer2023compositional,lachapelle2023additive,montero2022lost}` | `\citep{wiedemer_compositional_2023,lachapelle_additive_2023,montero_lost_2022}` | (Wiedemer et al., 2023; Lachapelle et al., 2023; Montero et al., 2022) | This is the interface split: unsupervised content, a request-indexed head, and a frozen renderer, with no Cartesian-product training requirement~\citep{wiedemer_compositional_2023,lachapelle_additive_2023,montero_lost_2022}. |
| 35 | 351 | main | `\citep{guo2020certified}` | `\citep{guo_certified_2023}` | (Guo et al., 2023) | The scoping sentence we carry is the one our deletion theory states for its own n-CLU: this is an n-CLU-level guarantee only; the frozen encoder and any residue of past writes in a learned landscape are separate channels, measured separately; we do not claim c |
| 36 | 364 | main | `\citep{hochreiter1997long}` | `\citep{hochreiter_long_1997}` | (Hochreiter & Schmidhuber, 1997) | \paragraph{Baselines: naive against naive.} Learned baselines (MLP, LSTM~\citep{hochreiter_long_1997}, transformer~\citep{vaswani_attention_2023}, and a continuous-input MLP) are matched to the n-CLU's \emph{read path}, and fitted from the same frozen features |
| 37 | 364 | main | `\citep{vaswani2017attention}` | `\citep{vaswani_attention_2023}` | (Vaswani et al., 2023) | \paragraph{Baselines: naive against naive.} Learned baselines (MLP, LSTM~\citep{hochreiter_long_1997}, transformer~\citep{vaswani_attention_2023}, and a continuous-input MLP) are matched to the n-CLU's \emph{read path}, and fitted from the same frozen features |
| 38 | 368 | main | `\citep{borgeaud2021improving,khandelwal2020generalization,lample2019large}` | `\citep{borgeaud_improving_2022,khandelwal_generalization_2020,lample_large_2019}` | (Borgeaud et al., 2022; Khandelwal et al., 2020; Lample et al., 2019) | The memory footprint is not a read-path parameter; it is printed, never matched~\citep{borgeaud_improving_2022,khandelwal_generalization_2020,lample_large_2019}. |
| 39 | 390 | main | `\citep{montero2021role,schott2021visual}` | `\citep{montero2021role,schott_visual_2022}` | (Montero et al., 2021; Schott et al., 2022) | The deletable object is the \emph{single design}: the n-CLU, read and memory footprint of Sec.~\ref{sec:object} on the Shapes3D value hold-out~\citep{montero2021role,schott_visual_2022}, with the partition fixed on validation by a rule frozen before the sweep, |
| 40 | 443 | main | `(added)` | `\citep{nie_controllable_2021}` | (Nie et al., 2021) | LACE~\citep{nie_controllable_2021} and PoE~\citep{hinton_training_2002} are in Tab.~\ref{tab:lace-poe}.} \label{tab:t7} \setlength{\tabcolsep}{3pt} \begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccccc} \toprule arm (optimiser steps) & seeds & all 109 |
| 41 | 443 | main | `(added)` | `\citep{hinton_training_2002}` | (Hinton, 2002) | LACE~\citep{nie_controllable_2021} and PoE~\citep{hinton_training_2002} are in Tab.~\ref{tab:lace-poe}.} \label{tab:t7} \setlength{\tabcolsep}{3pt} \begin{adjustbox}{max width=\linewidth}\begin{tabular}{lcccccc} \toprule arm (optimiser steps) & seeds & all 109 |
| 42 | 660 | main | `\citep{efron1979bootstrap}` | `\citep{efron_bootstrap_1979}` | (Efron, 1979) | Paired per held-out combination against the strongest data-matched baseline, the n-CLU's 95\% bootstrap interval~\citep{efron_bootstrap_1979} lies above zero on every rung and seed (lowest lower bound +2.46, at $n=3$, seed 2), and the smallest seed-level margi |
| 43 | 952 | main | `\citep{nie2021controllable}` | `\citep{nie_controllable_2021}` | (Nie et al., 2021) | \paragraph{LACE and PoE, at their own settings.} LACE~\citep{nie_controllable_2021} and PoE~\citep{hinton_training_2002}, re-implemented at their own settings in this feature space, score at most 0.267 and 0.020 distinct valid novel tuples per held-out combina |
| 44 | 952 | main | `\citep{hinton2002training}` | `\citep{hinton_training_2002}` | (Hinton, 2002) | \paragraph{LACE and PoE, at their own settings.} LACE~\citep{nie_controllable_2021} and PoE~\citep{hinton_training_2002}, re-implemented at their own settings in this feature space, score at most 0.267 and 0.020 distinct valid novel tuples per held-out combina |
| 45 | 1169 | main | `\citep{hopfield1982neural,krotov2016dense,ramsauer2021hopfield,hoover2023energy,kong2024reservoir}` | `\citep{hopfield_neural_1982,krotov_dense_2016,ramsauer_hopfield_2021,hoover_energy_2023,kong_reservoir-computing_2024}` | (Hopfield, 1982; Krotov & Hopfield, 2016; Ramsauer et al., 2021; Hoover et al., 2023; Kong et al., 2024) | We call the class of objects that this n-CLU is the static limit of a \emph{Latent Dynamics Model}: a memory whose content is read by relaxing a latent state under a designed energy~\citep{hopfield_neural_1982,krotov_dense_2016,ramsauer_hopfield_2021,hoover_en |
| 46 | 1290 | appendix | `(added)` | `\citep{hochreiter_long_1997}` | (Hochreiter & Schmidhuber, 1997) | \section{Tab.~\ref{tab:t7}: what it does and does not claim} \label{app:t7more} The baselines of Tab.~\ref{tab:t7} are matched on the read path at 130,514 (MLP), 130,721 (LSTM~\citep{hochreiter_long_1997}), 130,433 (transformer~\citep{vaswani_attention_2023})  |
| 47 | 1291 | appendix | `(added)` | `\citep{vaswani_attention_2023}` | (Vaswani et al., 2023) | \section{Tab.~\ref{tab:t7}: what it does and does not claim} \label{app:t7more} The baselines of Tab.~\ref{tab:t7} are matched on the read path at 130,514 (MLP), 130,721 (LSTM~\citep{hochreiter_long_1997}), 130,433 (transformer~\citep{vaswani_attention_2023})  |
| 48 | 1307 | appendix | `(added)` | `\citep{nie_controllable_2021}` | (Nie et al., 2021) | Declared criteria: S1 (controls collapse) \textsc{pass}; S2 (beats the earlier n-CLU's read on every seed) \textsc{fail}, seed 0; S3 (against the nearest-prototype ceiling) \textsc{lose}; S4 (against LACE~\citep{nie_controllable_2021}, PoE~\citep{hinton_traini |
| 49 | 1307 | appendix | `(added)` | `\citep{hinton_training_2002}` | (Hinton, 2002) | Declared criteria: S1 (controls collapse) \textsc{pass}; S2 (beats the earlier n-CLU's read on every seed) \textsc{fail}, seed 0; S3 (against the nearest-prototype ceiling) \textsc{lose}; S4 (against LACE~\citep{nie_controllable_2021}, PoE~\citep{hinton_traini |
| 50 | 1335 | appendix | `(added)` | `\citep{3dshapes18,kim_disentangling_2019}` | (Burgess & Kim, 2018; Kim & Mnih, 2019) | caption{\textbf{LACE and PoE at their own settings, Shapes3D~\citep{3dshapes18,kim_disentangling_2019} value hold-out, TEST, $n=3$ seeds (0, 1, 2).} Distinct valid novel tuples per held-out combination, on the combinations of Tab.~\ref{tab:t7} (upper block) an |
| 51 | 1372 | appendix | `\citep{nie2021controllable}` | `\citep{nie_controllable_2021}` | (Nie et al., 2021) | \paragraph{LACE and PoE in full.} LACE~\citep{nie_controllable_2021} and PoE~\citep{hinton_training_2002} were scored beside the learned baselines on rows 3--4 and on every ladder rung (Tab.~\ref{tab:lace-poe}), 3 seeds; no learned-baseline column was replaced |
| 52 | 1372 | appendix | `\citep{hinton2002training}` | `\citep{hinton_training_2002}` | (Hinton, 2002) | \paragraph{LACE and PoE in full.} LACE~\citep{nie_controllable_2021} and PoE~\citep{hinton_training_2002} were scored beside the learned baselines on rows 3--4 and on every ladder rung (Tab.~\ref{tab:lace-poe}), 3 seeds; no learned-baseline column was replaced |
| 53 | 1652 | appendix | `(added)` | `\citep{efron_bootstrap_1979}` | (Efron, 1979) | Mean paired $\Delta$ against that baseline on row 4 is $+10.143$ / $+8.673$ / $+7.796$ (bootstrap~\citep{efron_bootstrap_1979} 2.5\% quantile $+6.632$ / $+5.816$ / $+5.082$) and on row 3 $+4.033$ / $+2.483$ / $+2.417$ ($+2.117$ / $+1.033$ / $+1.200$): the bar  |
| 54 | 1744 | appendix | `(added)` | `\citep{guo_certified_2023,golatkar_eternal_2020}` | (Guo et al., 2023; Golatkar et al., 2020) | Approximate unlearning~\citep{guo_certified_2023,golatkar_eternal_2020} (gradient ascent, fine-tuning on the retained rows) was not run against these baselines (App.~\ref{app:notrun}). |
| 55 | 1996 | appendix | `H14: the Blelloch--Golovin stable-matching table~\citep{blelloch2007strongly}` | `\citet{blelloch_strongly_2007}` | Blelloch & Golovin (2007) | The placement rule is the~\citet{blelloch_strongly_2007} stable-matching table with a global priority order; the theorems follow from theirs and the fix-up cascade is theirs. |
| 56 | 2040 | appendix | `\citep{gellmann1968behavior}` | `\citep{gell-mann_behavior_1968}` | (Gell-Mann et al., 1968) | item \textbf{GMOR~\citep{gell-mann_behavior_1968} on trained checkpoints.} With a linear ambient spurion the three objects satisfy $\mu^2F^2=\delta\Sigma$ to a maximum absolute deviation of \sci{1.33}{-15} on 8 trained SO(2) checkpoints. |

BEFORE sites not matched to an AFTER site (a removed citation would show here): none

Rule-level before → after (the exact source spans): `stage-145204/EDIT-REPORT-V2-DRAFT-v2.md` and `…-style.md`.

## 3. Citation count per bib entry (key occurrences in \cite*/\nocite on code), before -> after

| new key | old key | before | after | change |
|---|---|---|---|---|
| `3dshapes18` | `burgess2018shapes` | 1 | 2 | +1 |
| `avidan_compositional_2022` | `liu2022compositional` | 1 | 1 | +0 |
| `blelloch_strongly_2007` | `blelloch2007strongly` | 1 | 1 | +0 |
| `borgeaud_improving_2022` | `borgeaud2021improving` | 1 | 1 | +0 |
| `bourtoule_machine_2021` | `bourtoule2021machine` | 2 | 2 | +0 |
| `cao_towards_2015` | `cao2015towards` | 1 | 1 | +0 |
| `ceci_exact_2027` | `quan2026exact` | 1 | 1 | +0 |
| `du_compositional_2020` | `du2020compositional` | 1 | 1 | +0 |
| `efron_bootstrap_1979` | `efron1979bootstrap` | 1 | 2 | +1 |
| `gell-mann_behavior_1968` | `gellmann1968behavior` | 1 | 1 | +0 |
| `georgopoulos_multilinear_2020` | `georgopoulos2020multilinear` | 1 | 1 | +0 |
| `ginart_making_2019` | `ginart2019making` | 1 | 1 | +0 |
| `golatkar_eternal_2020` | `golatkar2020eternal` | 1 | 2 | +1 |
| `golatkar_training_2024` | `golatkar2023training` | 1 | 1 | +0 |
| `goos_multilinear_2002` | `vasilescu2002tensorfaces` | 1 | 1 | +0 |
| `greydanus_hamiltonian_2019` | `greydanus2019hamiltonian` | 1 | 1 | +0 |
| `guo_certified_2023` | `guo2020certified` | 2 | 3 | +1 |
| `ha_world_2018` | `ha2018world` | 1 | 1 | +0 |
| `hafner_mastering_2025` | `hafner2025mastering` | 1 | 1 | +0 |
| `hall_adding_2002` | `hall2002adding` | 1 | 1 | +0 |
| `harkonen_ganspace_2020` | `harkonen2020ganspace` | 1 | 1 | +0 |
| `hinton_training_2002` | `hinton2002training` | 2 | 4 | +2 |
| `hochreiter_long_1997` | `hochreiter1997long` | 1 | 3 | +2 |
| `hoover_energy_2023` | `hoover2023energy` | 1 | 1 | +0 |
| `hopfield_neural_1982` | `hopfield1982neural` | 1 | 1 | +0 |
| `jin_evd_2014` | `jin2014evd` | 1 | 1 | +0 |
| `khandelwal_generalization_2020` | `khandelwal2020generalization` | 1 | 1 | +0 |
| `kim_disentangling_2019` | `kim2018disentangling` | 1 | 2 | +1 |
| `kong_approximate_2022` | `kong2022approximate` | 1 | 1 | +0 |
| `kong_data_2023` | `kong2023data` | 1 | 1 | +0 |
| `kong_reservoir-computing_2024` | `kong2024reservoir` | 1 | 1 | +0 |
| `krotov_dense_2016` | `krotov2016dense` | 1 | 1 | +0 |
| `lachapelle_additive_2023` | `lachapelle2023additive` | 1 | 1 | +0 |
| `lample_large_2019` | `lample2019large` | 1 | 1 | +0 |
| `lee_fast_2021` | `lee2021fast` | 1 | 1 | +0 |
| `montero2021role` | `montero2021role` | 1 | 1 | +0 |
| `montero_lost_2022` | `montero2022lost` | 1 | 1 | +0 |
| `moon_feature_2024` | `moon2024feature` | 1 | 1 | +0 |
| `nie_controllable_2021` | `nie2021controllable` | 2 | 4 | +2 |
| `oldfield_bilinear_2024` | `oldfield2024bilinear` | 1 | 1 | +0 |
| `oldfield_panda_2023` | `oldfield2023panda` | 1 | 1 | +0 |
| `oldfield_tensor_2021` | `oldfield2021tensor` | 1 | 1 | +0 |
| `priem_openalex_2022` | `priem2022openalex` | 1 | 1 | +0 |
| `raissi_physics-informed_2019` | `raissi2019physics` | 1 | 1 | +0 |
| `ramsauer_hopfield_2021` | `ramsauer2021hopfield` | 1 | 1 | +0 |
| `schott_visual_2022` | `schott2021visual` | 1 | 1 | +0 |
| `sun_generative_2025` | `sun2025generative` | 1 | 1 | +0 |
| `tenenbaum_separating_2000` | `tenenbaum2000separating` | 1 | 1 | +0 |
| `vafa_evaluating_2024` | `vafa2024evaluating` | 1 | 1 | +0 |
| `valevski_diffusion_2025` | `valevski2025diffusion` | 1 | 1 | +0 |
| `vaswani_attention_2023` | `vaswani2017attention` | 1 | 3 | +2 |
| `vlasic_face_2005` | `vlasic2005face` | 1 | 1 | +0 |
| `wiedemer_compositional_2023` | `wiedemer2023compositional` | 1 | 1 | +0 |
| `xu_netflix_2023` | `xu2023netflix` | 1 | 1 | +0 |
| `zhao_toward_2022` | `zhao2022toward` | 1 | 1 | +0 |

entries in the bib: 55; cited after: 55; UNCITED after: none; cited keys not in the bib: none
citation commands: before 45 (9 \nocite), after 56; key occurrences before 59, after 72
commands after by kind: {'citealp': 3, 'citep': 51, 'citet': 2}
citations inside the abstract after: 0
\citet whose first surname also appears in the 80 characters before it (doubled name): none

## 4. Named works with no bib entry ("missing from the bib"; nothing added)
| name in the text | where (first) | note |
|---|---|---|
| Mamba-2 | not named in either draft | named in the brief; no entry, no mention |
| Gated DeltaNet | not named in either draft | named in the brief; no entry, no mention |
| MLP / continuous-input MLP | Sec. 1.1 l.145 | generic architecture; no entry |
| GAN, VAE (as model classes) | Sec. 1.1 l.154, l.166 | generic; the cited works name them |
| diffusion models (generic) | Sec. 1.1 l.191 | generic; the compartmentalised variant is cited |
| PCA / principal-component models | Sec. 1.1 l.164 | generic; the decremental-PCA line is cited |
| approximate-unlearning methods "gradient ascent", "fine-tuning on the retained rows" | App. l.1744 | named unlearning methods with no entry; the family is cited there (Guo et al.; Golatkar et al.) |
| "nearest-prototype ceiling" | Sec. 3 l.969, App. tables | this study's own control; no external work |
| "Latent Dynamics Model" phrase "used in reduced-order modelling and in model-based reinforcement learning" | Sec. 5 l.1170 | two fields are named; no entry |
| fluctuation–dissipation-consistent noise | App. l.2008 | textbook physics; no entry |

Not changed and noted: a later appendix mention of "Blelloch--Golovin placement" (l.2405, "The two laws … in full") stays uncited under C7 (the appendix block cites it at l.1996). The appendix "(ε, δ) unlearning" (l.2000) stays uncited under C7 (Guo et al. is cited at l.1744).

## 5. Acceptance
| check | DRAFT-v2 before → after | DRAFT-v2-style before → after |
|---|---|---|
| exit codes (pdflatex, bibtex, pdflatex ×3) | 0 → 0 (all) | 0 → 0 (all) |
| undefined citations / references | 0 / 0 → **0 / 0** | 0 / 0 → **0 / 0** |
| errors | 0 → 0 | 0 → 0 |
| overfull boxes | 0 → **0** (V1 without C6: 1, 133 pt, the lee_fast_2021 URL) | 0 → **0** (same) |
| underfull boxes | 14 → 20 | 13 → 20 |
| bibitems | 55 → 55 | 55 → 55 |
| bibtex warnings | 0 → 1: `Warning--empty journal in ha_world_2018` | 0 → 1: same |
| pages | **42 → 42** (labels 75 → 75, 0 moved; references pp. 19–23 before and after; appendices from p. 24) | **43 → 44** (labels 75 → 75, 0 renumbered; references pp. 19–23 → 19–24; every appendix label +1 page) |
| end of main text (AI-use statement page) | 18 → 18 | 19 → 19 |
| style diff-check DRAFT-v2 vs style | **ALL DIFFS EMPTY** | |
| `\PLACEHOLDER` / `\TODO` arguments, in order | byte-identical (2 / 4) | byte-identical (2 / 4) |
| provenance (check_provenance.py) | unsourced lines 7 → **3**; not-found tokens 9 → 9 (improved: the removed lines are hand-typed years `'015'`, `'019'`, `'023'`, `'019','019'` at l.160/162/191/197) | 7 → 3; 10 → 10 (same four lines) |
| derived check | byte-identical | byte-identical |
| source diff | 48 lines out, 48 in; 2487 → 2487 lines | 46 out, 46 in; 2478 → 2478 lines |
| rendered non-citation residue (neutral builds + `residue.py`) | **EMPTY**: the 14 hand-typed strings H1–H14 found and removed, remainder before == after | **EMPTY** |
| positive controls | `--plant` → NOT EMPTY (PLANTEDWORD); deleting "renders" → NOT EMPTY | `--plant` → NOT EMPTY |
| bib sha256 (tasks/pw1 and w17-paper) | 26cc298f… → 26cc298f… (both) | |
| blind scan of the installed PDFs | 0 identity tokens; 0 "Zotero"/"/Users/" in text or .bbl (the bib's `file` fields are not printed) | same |

Rendered-text word diff of the full builds (with citations): `stage-145204/checks-V2/WORDDIFF-*.txt`. It is NOT EMPTY by design: it holds the author-year strings and the new reference list. The proof of "only citation text" is the neutral-build diff: `stage-145204/WORDDIFF-neutral-*.txt`, `stage-145204/RESIDUE-*.txt` (+ `-PLANT*`).
The 14 hand-typed strings removed (the only non-command text that left the rendered body): "Tenenbaum and Freeman, 2000"; ", Vasilescu and Terzopoulos, 2002"; "Vlasic et al., 2005"; ", 2020" (MLC-VAE); ", 2020" (GANSpace); ", 2021" (TCA); ", 2023" (PandA); "Cao and Yang (2015)"; "(Ginart et al., 2019)"; "(Golatkar, Achille, Swaminathan and Soatto)"; "(Raissi, Perdikaris and Karniadakis, 2019)"; "(Greydanus, Dzamba and Yosinski, 2019)"; "Hall, Marshall and Martin, 2002"; "Blelloch–Golovin".

## 6. For the Head (printed, not changed)
1. **The bib renders ISBN, ISSN, full URLs and conference locations** for most entries (a Zotero export). The references grew. DRAFT-v2 still fits pp. 19–23, but the style draft's references now run to p. 24, so that draft is 44 pages. Removing those fields, if wanted, is an edit to the Head's bib.
2. `3dshapes18`: its title renders "3d shapes dataset" (the style lowercases it; braces would keep it). `gell-mann_behavior_1968`: renders "SU 3 × SU 3" (the parentheses of SU(3) are missing in the title field).
3. `ha_world_2018` is an `@article` with no journal: bibtex warns "empty journal", and the reference prints "World Models. March 2018. doi: 10.5281/ZENODO.1207631".
4. Five rendered years change with the new bib (§1), for example Vaswani et al., **2023** and Guo et al., **2023** (the arXiv revision dates).
5. Two stale source comments remain (not rendered; not edited): l.20 "No \cite{} command is used: cite-tagger inserts them" and l.1247 "cite-tagger owns references.bib; this draft carries no \cite yet". `references.bib` stays in `w17-paper/` (sha256 226ddd54…), unused now.
6. C3: the Sec. 1.1 opening sentence names three system families and defers their citations to the next paragraph. If the Head wants citations at that first mention, the sentence's "each is cited in one sentence below" would need rewording.

## 7. Hashes
| file | before | after (installed) |
|---|---|---|
| w17-paper/DRAFT-v2.tex | `f9642d0fb44e526c0ea0fbe104395670054d301b6c0a651ce14969633d5e6048` | `42cd4a48e84dd0b2f29fd5761caee64f31c2320ca80ef2df80ec41ef8e5ab011` |
| w17-paper/DRAFT-v2-style.tex | `27babec26b977d96cbc481027f275485a9eeeec891fe9dbf6fb3423b8488054b` | `f7ee48467132ff336acaef466faf2898fa30824c8c8b2867a86c55a827952cb1` |
| w17-paper/DRAFT-v2.pdf | `c05f96897a3a0907b48eaae1ffbbae129f718ab8a3711f86260c77d071fcb42b` | `5defc77ade7fffb9c6d1fd867db274a92718788766dce5c9b73fc89264f5762c` |
| w17-paper/DRAFT-v2-style.pdf | `b03544518b8320c8a0aee16ff884ea607e1d767e78ed12d2860f9b42cd560a2e` | `63814b1afb0ad7c78a0cbc3e084aea30f6753a1de4ed24ceb864e299647f84bd` |
| w17-paper/refs_clu_iclr.bib | `26cc298f0ffabdda3be64b71faeaa3cbb3c04d978b05be59caacdc7c30cd4520` | `26cc298f0ffabdda3be64b71faeaa3cbb3c04d978b05be59caacdc7c30cd4520` |
| w17-paper/references.bib | `226ddd54790f3035a8645c463679e6276c4d1ba49d08871cb06d722669c9a6cc` | `226ddd54790f3035a8645c463679e6276c4d1ba49d08871cb06d722669c9a6cc` |
| tasks/pw1/refs_clu_iclr.bib | `26cc298f0ffabdda3be64b71faeaa3cbb3c04d978b05be59caacdc7c30cd4520` | `26cc298f0ffabdda3be64b71faeaa3cbb3c04d978b05be59caacdc7c30cd4520` |

## What did NOT run
- No `.bib` was written, edited, sorted or regenerated. The bibtex warning and the rendering observations are reported (§6), not fixed.
- No sentence was rewritten. No number or vocabulary word changed. The abstract (and its `\TODO`) is byte-identical. No citation went into a heading.
- Not added (logged): citations at the first mentions in the abstract, in the Sec. 1.2 heading, and in the Sec. 1.1 deferring sentence (C1–C3); citations at later appendix mentions of an already-cited work (C7).
- Nothing was cited that lacks a bib entry (§4 list); Mamba-2 and Gated DeltaNet are not named in either draft.
- `pw1_draft_final`'s residual audit and figure-folder audit were not re-run (no figure, caption vocabulary or float changed beyond inserted citations). The FINAL-INDEX page tables were not regenerated; the page effects are in §5.
- No in-place build inside `w17-paper/`: the stage-built files were installed byte-identically, and a fresh copy of the installed tree was rebuilt in scratch (`stage-145204/post-install/`) to prove reproducibility (.aux/.bbl/.out equal, PDF text equal).
- PDF bytes differ between the stage and the fresh rebuild only by date/ID (the installed PDFs ARE the stage PDFs).
- No measurement, fit, model load or GPU use; `vm_stat` showed ~184k free pages before the builds.
- `outputs/pw1-advisor-report.md` was not written: it belongs to the Hub.
- No mirroring to another machine (no path named). The batmac path is `~/clu/outputs/w17-paper/`.

## Branch and bundle
Branch `pw1/draft-citations` at `2318c1b`: tooling only, `pw1_draft_citations/` (10 files); no outputs committed. Replay check: the committed `cite_edit.py` applied to V0 reproduces both installed `.tex` byte for byte.
Bundle `pw1-draft-citations.bundle` sha256 `032446d4cecf3ec2bdcc1be178313b1693d9fe1a1032a4edd1de8c8d29627cc8`; `git bundle verify`: exit 0, complete history.
