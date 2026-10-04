# cite_edit report: stage-145204/V0/DRAFT-v2-style.tex

| rule | kind | line | before | after |
|---|---|---|---|---|
| H1 | convert | 154 | `(Tenenbaum and Freeman, 2000\nocite{tenenbaum2000separating};` | `(\citealp{tenenbaum2000separating};` |
| H2 | convert | 154 | `TensorFaces, Vasilescu and Terzopoulos, 2002\nocite{vasilescu2002tensorfaces}` | `TensorFaces~\citep{vasilescu2002tensorfaces}` |
| H3 | convert | 155 | `Vlasic et al., 2005\nocite{vlasic2005face}` | `\citealp{vlasic2005face}` |
| H4 | convert | 155 | `MLC-VAE, 2020~\citep{georgopoulos2020multilinear}` | `MLC-VAE~\citep{georgopoulos2020multilinear}` |
| H5 | convert | 156 | `GANSpace, 2020~\citep{harkonen2020ganspace}` | `GANSpace~\citep{harkonen2020ganspace}` |
| H6 | convert | 157 | `TCA, 2021~\citep{oldfield2021tensor}` | `TCA~\citep{oldfield2021tensor}` |
| H7 | convert | 157 | `PandA, 2023~\citep{oldfield2023panda}` | `PandA~\citep{oldfield2023panda}` |
| H8 | convert | 161 | `form of Cao and Yang (2015)\nocite{cao2015towards}` | `form of~\citet{cao2015towards}` |
| H9 | convert | 163 | `$k$-means (Ginart et al., 2019\nocite{ginart2019making})` | `$k$-means~\citep{ginart2019making}` |
| H10 | convert | 192 | `models (Golatkar, Achille, Swaminathan and Soatto\nocite{golatkar2023training})` | `models~\citep{golatkar2023training}` |
| H11 | convert | 197 | `networks (Raissi, Perdikaris and Karniadakis, 2019\nocite{raissi2019physics})` | `networks~\citep{raissi2019physics}` |
| H12 | convert | 198 | `networks (Greydanus, Dzamba and Yosinski, 2019\nocite{greydanus2019hamiltonian})` | `networks~\citep{greydanus2019hamiltonian}` |
| H13 | convert | 203 | `(Hall, Marshall and Martin, 2002\nocite{hall2002adding},` | `(\citealp{hall2002adding},` |
| H14 | convert | 1987 | `the Blelloch--Golovin stable-matching table~\citep{blelloch2007strongly}` | `the~\citet{blelloch2007strongly} stable-matching table` |
| M1 | add | 147 | `an LSTM or a transformer is a unit` | `an LSTM~\citep{hochreiter1997long} or a transformer~\citep{vaswani2017attention} is a unit` |
| M2 | note | | 2 matches; the first (line 443) is edited | |
| M2 | add | 443 | `LACE and PoE are in` | `LACE~\citep{nie2021controllable} and PoE~\citep{hinton2002training} are in` |
| P1 | add | 1279 | `(LSTM),` | `(LSTM~\citep{hochreiter1997long}),` |
| P2 | add | 1280 | `(transformer) and 131,891` | `(transformer~\citep{vaswani2017attention}) and 131,891` |
| P3 | add | 1296 | `(against LACE, PoE)` | `(against LACE~\citep{nie2021controllable}, PoE~\citep{hinton2002training})` |
| P4 | add | 1324 | `at their own settings, Shapes3D value` | `at their own settings, Shapes3D~\citep{burgess2018shapes,kim2018disentangling} value` |
| P5 | add | 1639 | `(bootstrap 2.5\% quantile` | `(bootstrap~\citep{efron1979bootstrap} 2.5\% quantile` |
| P6 | add | 1732 | `Approximate unlearning (gradient ascent` | `Approximate unlearning~\citep{guo2020certified,golatkar2020eternal} (gradient ascent` |
| B1 | bibliography | 1237 | `\IfFileExists{references.bib}{%` | `\IfFileExists{refs_clu_iclr.bib}{%` |
| B2 | bibliography | 1238 | `\bibliography{references}` | `\bibliography{refs_clu_iclr}` |
| B3 | bibliography | 51 | `\usepackage{hyperref}` | `\PassOptionsToPackage{hyphens}{url}\usepackage{hyperref}` |

keys renamed: 72; \nocite left on code: []; lines 2478 -> 2478
problems: []
