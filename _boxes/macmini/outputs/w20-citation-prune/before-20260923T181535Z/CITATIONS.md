# CITATIONS.md: every paper the draft must cite

**Owner:** w17 `spoke/citation-scout` (sole writer of this file). **Consumer:** `spoke/cite-tagger` (writes `references.bib`, `CITATIONS-FOR-ZOTERO.md`).
**Date:** 2026-09-22. **Machine:** batmac. **Scored runs:** none. **CHLU numbers quoted:** none. The only programme figure in this file is the novelty crawl's ≈4,050-work count, quoted with its artefact (`evidence/outputs/_thinkpad/w8/w8-novelty/NOVELTY.md` Addendum §A). This file holds bibliographic identifiers only; it holds no w15/w16 result and no other result.

## How every identifier was verified (read this before trusting any ID)

- **arXiv IDs:** all 76 were resolved against `https://export.arxiv.org/api/query?id_list=…`. For each one, the returned title and first author matched the paper named here. The raw API output is in `~/clu/outputs/w17-citation-scout/verify/arxiv_results.jsonl` and `arxiv_results2.jsonl`.
- **DOIs:** every DOI was resolved against `https://api.crossref.org/works/<DOI>` or found by a Crossref bibliographic search whose top hit's title matched exactly. The one exception is the arXiv-minted Zenodo DOI for Ha & Schmidhuber, which was resolved through `doi.org` (HTTP 302 → zenodo.org/record/1207631). The scripts are `verify/crossref.py` and `verify/arxiv_check.py`.
- ⛔ **One recalled DOI was WRONG and was discarded:** `10.1007/3-540-47969-4_29` resolves to *"Parameter Estimates for a Pencil of Lines"* (Speyer). The correct TensorFaces DOI is `…_30`, verified. This shows why no DOI here comes from memory.
- **`no DOI`** means this scout resolved none. It does **not** assert that none exists. ICLR, ICML/PMLR and pre-2022 NeurIPS papers generally have none. NeurIPS 2022+ proceedings carry Curran `10.52202/…` DOIs, and every NeurIPS 2022+ entry below was swept for one.
- **`no arXiv`** means the paper has no arXiv version that this scout found. Montero 2021 was checked with an arXiv title search that returned 0 hits.
- **Venue** is stated only where the arXiv comment or journal-ref field, or the Crossref container, confirms it. Otherwise the entry says *venue not verified*.
- openreview.net returns `403 ChallengeRequiredError` from this box, as the w12 positioning scout also found. The OpenReview forum ID for Montero 2021 is taken from a web-search hit on `openreview.net/pdf?id=qbH974jKUVy`, **not** from a fetch.

**Cite-key convention:** `firstauthorsurname` + `year` + `firstcontentword`, lowercase ASCII. The year is the one the key is built from, i.e. the venue year where one is verified.

**Sections follow the Head-ruled w17 spine** (`advisor-wave-plan-w15-w18.md` §3): (1) positioning · (2) world-model motivation · (3) the Naive CLU on Shapes · (4) the maze · (5) limitations → leaving note (LDM) · (6) follow-up · related work · appendix. A paper that serves several sections is listed once, under its main section, with the others named in its "supports" line.

⚠ **AMENDMENT-1 flag.** Entries marked **[PLACEHOLDER-ROW]** are method citations for rows that are w15/w16 `\PLACEHOLDER{}`s (LACE, PoE, the CMAZE-T/O world-model rows). Cite the **method** where the placeholder's row is named. ⛔ Never cite them next to a result, because there is no result yet, and the row may land as a negative.

---

## §1 Introduction / positioning: "compositional generation is crowded; compositional + exact-deletion generation is unoccupied"

### 1a. The multilinear / factored-generation line: the claim-STOP owners (cite FIRST, so a referee cannot spring them)

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `tenenbaum2000separating` | Separating Style and Content with Bilinear Models | Tenenbaum | 2000 | no arXiv | 10.1162/089976600300015349 | STOP #1 owner: bilinear style–content factored generation. Neural Computation. |
| `vasilescu2002tensorfaces` | Multilinear Analysis of Image Ensembles: TensorFaces | Vasilescu | 2002 | no arXiv | 10.1007/3-540-47969-4_30 | STOP #1 owner: N-mode SVD, request-indexed tensor synthesis over label-defined principal subspaces. ECCV 2002, LNCS. ⛔ Never the "TensorFaces phrasing" in our own claim. |
| `vlasic2005face` | Face Transfer with Multilinear Models | Vlasic | 2005 | no arXiv | 10.1145/1073204.1073209 | Deep-form re-run of the multilinear line (multilinear 3DMM). ACM TOG / SIGGRAPH 2005. |
| `vasilescu2021causalx` | CausalX: Causal eXplanations and Block Multilinear Factor Analysis | Vasilescu | 2021 | 2102.12853 | 10.1109/ICPR48806.2021.9412780 | The one block-multilinear ancestor, and the highest atom count in the novelty crawl. Supports "wells: claim the use, not the idea". ICPR 2020 (published 2021). |
| `georgopoulos2020multilinear` | Multilinear Latent Conditioning for Generating Unseen Attribute Combinations (MLC-VAE) | Georgopoulos | 2020 | 2009.04075 | no DOI | STOP owner for "tensor-product label conditioning for unseen attribute combinations"; the closest modern statement of the composition claim. ICML 2020. |
| `lee2014incremental` | Incremental N-Mode SVD for Large-Scale Multilinear Generative Models | Lee | 2014 | no arXiv | 10.1109/TIP.2014.2346012 | Pre-empt: the only work bridging eigenspace-update numerics and multilinear generative models. It is **incremental only, with no removal operator**. IEEE TIP. |

### 1b. The frozen-generator / unsupervised-subspace line (owns "unsupervised subspace + frozen decoder")

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `harkonen2020ganspace` | GANSpace: Discovering Interpretable GAN Controls | Härkönen | 2020 | 2004.02546 | no DOI | PCA of frozen-generator features applied layer-wise, i.e. unsupervised principal subspace + frozen decoder. Nearest published thing to "wells". NeurIPS 2020. |
| `oldfield2021tensor` | Tensor Component Analysis for Interpreting the Latent Space of GANs (TCA) | Oldfield | 2021 | 2111.11736 | 10.5244/C.35.430 | Frozen-generator tensor decomposition; no label request, no address book. BMVC 2021. |
| `oldfield2023panda` | PandA: Unsupervised Learning of Parts and Appearances in the Feature Maps of GANs | Oldfield | 2023 | 2206.00048 | no DOI | Parts × appearances on a frozen GAN, which covers well-like partition + unsupervised bases + frozen decoder, but with **no labels and no address book**. ICLR 2023. |
| `oldfield2024bilinear` | Bilinear Models of Parts and Appearances in Generative Adversarial Networks | Oldfield | 2024 | no arXiv | 10.1109/TPAMI.2024.3415506 | Journal PandA. Shows the missing join is still missing in 2024 (still "entirely unsupervised"). IEEE TPAMI. |
| `georgopoulos2022cluster` | Cluster-guided Image Synthesis with Unconditional Models | Georgopoulos | 2022 | 2112.12911 | 10.1109/CVPR52688.2022.01125 | Frozen-generator synthesis by combining clusters across layers, where clusters play the role subspaces play in ours. CVPR 2022. |
| `ding2022attribute` | Attribute Group Editing for Reliable Few-shot Image Generation (AGE) | Ding | 2022 | 2203.08422 | 10.1109/CVPR52688.2022.01091 | Unsupervised sparse dictionary on a frozen generator, with a class mean as a single additive address entry, not a product table. CVPR 2022. |
| `shen2020interpreting` | Interpreting the Latent Space of GANs for Semantic Face Editing (InterFaceGAN) | Shen | 2020 | 1907.10786 | 10.1109/CVPR42600.2020.00926 | Label-supervised directions in a frozen generator. The additive, non-tensor ancestor of our address book. CVPR 2020. |
| `shen2021closed` | Closed-Form Factorization of Latent Semantics in GANs (SeFa) | Shen | 2021 | 2007.06600 | 10.1109/CVPR46437.2021.00158 | Closed-form (eigen) factorisation of frozen-generator weights, the same additive-direction family. CVPR 2021. |

### 1c. Compositional generation (the "crowded" half of the positioning sentence)

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `du2020compositional` | Compositional Visual Generation and Inference with Energy Based Models | Du | 2020 | 2004.06030 | no DOI | Canonical compositional generation by energy summation. Also the method family of the LACE/PoE placeholder rows. NeurIPS 2020. |
| `liu2022compositional` | Compositional Visual Generation with Composable Diffusion Models | Liu | 2022 | 2206.01714 | 10.1007/978-3-031-19790-1_26 | Compositional generation in diffusion models. Shows the field is crowded. ECCV 2022. |
| `du2023reduce` | Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC | Du | 2023 | 2302.11552 | no DOI | Compositional generation; its sampler critique applies to the LACE arm (per `gen-lace-spec.md`). ICML 2023. |
| `okawa2023compositional` | Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task | Okawa | 2023 | 2310.09336 | no DOI | Controlled synthetic study of held-out concept combinations in generative models, which is the same evaluation shape as our held-out corners. NeurIPS 2023 (arXiv comment). |

### 1d. Exact unlearning: owners of the primitive and of adjacent exact results (the "exact deletion" half)

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `cao2015towards` | Towards Making Systems Forget with Machine Unlearning | Cao | 2015 | no arXiv | 10.1109/SP.2015.35 | ⭐ **Owns the downdate primitive** (summation form; a covariance is a summation). We do not claim the primitive. IEEE S&P 2015. |
| `ginart2019making` | Making AI Forget You: Data Deletion in Machine Learning | Ginart | 2019 | 1907.05012 | no DOI | Exact deletion for k-means, i.e. an unsupervised count/centroid model. NeurIPS 2019. |
| `xu2023netflix` | Netflix and Forget: Efficient and Exact Machine Unlearning from Bi-linear Recommendations | Xu | 2023 | 2302.06676 | no DOI | Exact unlearning for bilinear matrix completion (Unlearn-ALS). Venue not verified (arXiv only). |
| `quan2026exact` | Exact Federated Continual Unlearning for Ridge Heads on Frozen Foundation Models | Quan | 2026 | 2603.12977 | 10.1007/978-3-032-37673-2_7 | The "ridge-on-frozen" exact-unlearning analogue: frozen backbone + closed-form head. By its own statement it covers ridge regression only, with **no PCA, eigendecomposition or generative models**. ECML-PKDD 2026, LNCS. |
| `bourtoule2021machine` | Machine Unlearning (SISA) | Bourtoule | 2021 | 1912.03817 | 10.1109/SP40001.2021.00019 | Sharded exact unlearning. Cite where we say sharded ensembles are **rebuttal-kit only, not a paper row** (`advisor-wave-plan-w12-w14.md` l.32). IEEE S&P 2021. |
| `guo2020certified` | Certified Data Removal from Machine Learning Models | Guo | 2020 | 1911.03030 | no DOI | Certified (approximate, ε-bounded) removal for linear models, the contrast class to our exact, store-scoped deletion. ICML 2020. |
| `izzo2021approximate` | Approximate Data Deletion from Machine Learning Models | Izzo | 2021 | 2002.10077 | no DOI | Approximate deletion for linear/logistic models (projective residual update), another approximate-deletion contrast. AISTATS 2021. |
| `golatkar2020eternal` | Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks | Golatkar | 2020 | 1911.04933 | 10.1109/CVPR42600.2020.00932 | Approximate forgetting in deep nets. Supports "the rival deletes only by retrain-without; approximate fine-tune unlearning is rebuttal-kit, not a row". CVPR 2020. |
| `nguyen2025survey` | A Survey of Machine Unlearning | Nguyen | 2025 | 2209.02299 | 10.1145/3749987 | General survey: exact vs approximate taxonomy. ACM TIST (arXiv 2022). |

### 1e. Generative-model deletion: why "retrain-equivalence is out of reach" is the field's own statement

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `kong2022approximate` | Approximate Data Deletion in Generative Models | Kong | 2022 | 2206.14439 | no DOI | The intro's problem sentence: generative deletion is largely open, and this paper settles for a density-ratio **approximation**. Venue not verified (arXiv only). |
| `liu2024machine` | Machine Unlearning in Generative AI: A Survey | Liu | 2024 | 2407.20516 | no DOI | The generative-unlearning literature substitutes behavioural proxies for retrain-equivalence. Venue not verified. |
| `moon2024feature` | Feature Unlearning for Pre-trained GANs and VAEs | Moon | 2024 | 2303.05699 | 10.1609/aaai.v38i19.30138 | Pre-empt: the nearest thing to deletion on a frozen generator. It unlearns a **feature, not rows**, is approximate, and **fine-tunes the generator**. AAAI 2024. |
| `kong2023data` | Data Redaction from Pre-trained GANs | Kong | 2023 | 2206.14389 | 10.1109/SaTML54575.2023.00048 | Use when defining exact deletion. It states that redaction is a different task from data deletion. IEEE SaTML 2023. |
| `sun2025generative` | Generative Adversarial Networks Unlearning | Sun | 2025 | 2308.09881 | 10.1109/TDSC.2025.3564992 | Approximate GAN unlearning by construction. IEEE TDSC 2025. |
| `fang2026machine` | Machine Unlearning in Low-Dimensional Feature Subspace | Fang | 2026 | 2601.22456 | no DOI | Updates a covariance inside a subspace but is **approximate** by its own statement. Pre-empt. Venue not verified. |
| `gandikota2023erasing` | Erasing Concepts from Diffusion Models (ESD) | Gandikota | 2023 | 2303.07345 | 10.1109/ICCV51070.2023.00230 | The concept-erasure line (fine-tuning, approximate), which a referee will name as "unlearning in generators". ICCV 2023. |
| `kumari2023ablating` | Ablating Concepts in Text-to-Image Diffusion Models | Kumari | 2023 | 2303.13516 | 10.1109/ICCV51070.2023.02074 | Same concept-erasure line. ICCV 2023. |
| `fan2024salun` | SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation | Fan | 2024 | 2310.12508 | no DOI | Approximate unlearning in both classifiers and generators. ICLR 2024 (arXiv comment). |
| `zhang2024unlearncanvas` | UnlearnCanvas: Stylized Image Dataset for Enhanced Machine Unlearning Evaluation in Diffusion Models | Zhang | 2024 | 2402.11846 | 10.52202/079017-3055 | Generative-unlearning **benchmark**; supports the benchmark-requirements section (the field scores unlearning with behavioural proxies). NeurIPS 2024 D&B. |

### 1f. The ⭐ one sentence that is ours: "downdate the statistic, not the truncated factors" (the contrast class)

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `hall2002adding` | Adding and subtracting eigenspaces with eigenvalue decomposition and singular value decomposition | Hall | 2002 | no arXiv | 10.1016/S0262-8856(02)00114-2 | ⭐ The decremental-PCA line is **explicitly approximate**: it subtracts deflated eigenspaces, in its own words "approximating", "tends to instability", "no rigorous error analysis". Do **not** concede exactness to it. Image and Vision Computing 20. |
| `hall1999adding` | Adding and Subtracting Eigenspaces | Hall | 1999 | no arXiv | 10.5244/C.13.45 | The BMVC 1999 conference original of the above. The novelty scout read the full text of this version. BMVC 1999. |
| `jin2014evd` | EVD Dualdating Based Online Subspace Learning | Jin | 2014 | no arXiv | 10.1155/2014/429451 | Add+delete on the covariance EVD; optimal only under a low-rank-plus-shift condition. Mathematical Problems in Engineering. |
| `lee2021fast` | Fast and more accurate incremental-decremental principal component analysis algorithm for online learning … | Lee | 2021 | no arXiv | 10.1117/1.JEI.30.4.043012 | Online incremental-decremental PCA, an approximation. Journal of Electronic Imaging. ⚠ The Crossref title is truncated in our log; cite-tagger should take the full title from the DOI metadata. |
| `gu1994stable` | A Stable and Efficient Algorithm for the Rank-One Modification of the Symmetric Eigenproblem | Gu | 1994 | no arXiv | 10.1137/S089547989223924X | The numerical primitive underneath rank-one eigen-updates. SIAM J. Matrix Anal. Appl. 15. |

### 1g. Tooling for the positioning count

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `priem2022openalex` | OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts | Priem | 2022 | 2205.01833 | no DOI | The citation graph behind the ≈4,050-work screen (`NOVELTY.md` Addendum §A). ⛔ This cites the **tool**. The count itself is the programme's own measurement and needs its artefact pointer, not this citation. |

---

## §2 Motivation: "world models need rule-valid (exact-judge), compositional, exactly deletable generation"

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `ha2018world` | World Models | Ha | 2018 | 1803.10122 | 10.5281/zenodo.1207631 | The canonical learned-world-model framing (VAE observation model + dynamics). The DOI is the arXiv-listed Zenodo record, resolved via doi.org. |
| `hafner2019dream` | Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) | Hafner | 2020 | 1912.01603 | no DOI | Latent-imagination world models. Venue not verified from the arXiv record (it was ICLR 2020, but the arXiv comment does not say so; cite-tagger should confirm). |
| `hafner2025mastering` | Mastering Diverse Domains through World Models (DreamerV3); Nature title: "Mastering diverse control tasks through world models" | Hafner | 2025 | 2301.04104 | 10.1038/s41586-025-08744-2 | The current generalist world model. ⚠ The arXiv and Nature titles differ; both were verified. Nature 2025. |
| `schrittwieser2020mastering` | Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero) | Schrittwieser | 2020 | 1911.08265 | 10.1038/s41586-020-03051-4 | Planning with a learned model. Nature 2020. |
| `micheli2023transformers` | Transformers are Sample-Efficient World Models (IRIS) | Micheli | 2023 | 2209.00588 | no DOI | Generative (token) world models. ICLR 2023. |
| `alonso2024diffusion` | Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) | Alonso | 2024 | 2405.12399 | 10.52202/079017-1873 | Diffusion world models, i.e. generative observation models. NeurIPS 2024. |
| `valevski2025diffusion` | Diffusion Models Are Real-Time Game Engines (GameNGen) | Valevski | 2025 | 2408.14837 | no DOI | A generative world model as a game engine. Motivates "rule-valid generation": the output must be a legal next state. ICLR 2025. |
| `bruce2024genie` | Genie: Generative Interactive Environments | Bruce | 2024 | 2402.15391 | no DOI | Generative interactive environments. Venue not verified from the arXiv record. |
| `vafa2024evaluating` | Evaluating the World Model Implicit in a Generative Model | Vafa | 2024 | 2406.03689 | 10.52202/079017-0846 | ⭐ **Closest published support for "rule-valid = valid under an exact judge"**: it scores a generative model's world model against the true rules (DFA-based metrics). NeurIPS 2024. |
| `kipf2020contrastive` | Contrastive Learning of Structured World Models (C-SWM) | Kipf | 2020 | 1911.12247 | no DOI | Compositional/object-factored world models. ICLR 2020. |
| `zhao2022toward` | Toward Compositional Generalization in Object-Oriented World Modeling | Zhao | 2022 | 2204.13661 | no DOI | ⭐ Compositional generalisation **as a world-model requirement**, the "compositional" leg of the motivation sentence. ICML 2022. |
| `lecun2022path` | A Path Towards Autonomous Machine Intelligence | LeCun | 2022 | no arXiv | no DOI | ⚠ **NOT VERIFIED BY THIS SCOUT.** It is an OpenReview position paper, and OpenReview is blocked here. Listed as a candidate only. cite-tagger: **do not insert** unless a later pass verifies it. |

---

## §3 The Naive CLU on Shapes: object, dataset, rivals, metrics, deletion rows

### 3a. Dataset and compositional-generalisation evaluation protocol

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `burgess2018shapes` | 3D Shapes Dataset | Burgess | 2018 | no arXiv | no DOI | ⭐ **3D Shapes (Shapes3D)**, the Shapes environment. Dataset release, `howpublished` = **https://github.com/google-deepmind/3d-shapes** (HTTP 200, checked 2026-09-22). ⚠ The URL in the evidence base's `.bib` stub, `github.com/deepmind/3dshapes-dataset/`, **returns 404**; do not use it. |
| `kim2018disentangling` | Disentangling by Factorising (FactorVAE) | Kim | 2018 | 1802.05983 | no DOI | The paper that introduced and first used 3D Shapes. Cite beside `burgess2018shapes`. ICML 2018 (the arXiv comment names only the workshop version; cite-tagger should confirm). |
| `montero2021role` | The role of Disentanglement in Generalisation | Montero | 2021 | no arXiv | no DOI | ⭐ **Montero et al.** Held-out-combination splits (recombination-to-element / -to-range) on dSprites/3D Shapes; finds only weak combinatorial generalisation. This is the provenance of our held-out-corner split. ICLR 2021; OpenReview forum `qbH974jKUVy` (from a web-search hit, not fetched: OpenReview is blocked). |
| `montero2022lost` | Lost in Latent Space: Disentangled Models and the Challenge of Combinatorial Generalisation (NeurIPS title: "…Examining Failures of Disentangled Models at Combinatorial Generalisation") | Montero | 2022 | 2204.02283 | 10.52202/068431-0736 | Follow-up showing that decoders, not only encoders, fail at combinatorial generalisation. Supports the interface split / frozen-renderer argument. ⚠ The arXiv and NeurIPS titles differ; the DOI's Crossref title is the NeurIPS one. NeurIPS 2022. |
| `schott2021visual` | Visual Representation Learning Does Not Generalize Strongly Within the Same Domain | Schott | 2022 | 2107.08221 | no DOI | Systematic held-out-factor evaluation on dSprites/Shapes3D/MPI3D, showing composition is hard for learned models. Venue not verified from the arXiv record (it was ICLR 2022; cite-tagger should confirm). |
| `locatello2019challenging` | Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations | Locatello | 2019 | 1811.12359 | no DOI | Disentanglement benchmark context for 3D Shapes. ICML 2019 (journal-ref). |
| `wiedemer2023compositional` | Compositional Generalization from First Principles | Wiedemer | 2023 | 2307.05596 | 10.52202/075280-0304 | Theory: compositional generalisation from the structure of the generator. Supports why the read stays compositional up to a fixed renderer. NeurIPS 2023. |
| `lachapelle2023additive` | Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation | Lachapelle | 2023 | 2307.02598 | 10.52202/075280-1092 | Cartesian-product extrapolation from additive decoder structure, the closest theory for generating unseen factor combinations. NeurIPS 2023. |
| `liang2025compositional` | Compositional Generalization via Forced Rendering of Disentangled Latents | Liang | 2025 | 2501.18797 | no DOI | ⭐ Theory ancestor: **latent factorisation can fail to survive the decoder**. Motivation for the plan-vs-latent finding and for the frozen renderer (DRAFT-v1 §10.6). Venue not verified. |
| `mahajan2025compositional` | Compositional Risk Minimization | Mahajan | 2025 | 2410.06303 | no DOI | Companion theory to the above (DRAFT-v1 §10.6). ICML 2025 (arXiv comment). |
| `lake2018generalization` | Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks (SCAN) | Lake | 2018 | 1711.00350 | no DOI | Canonical compositional-generalisation failure of learned sequence models. Background for the "naive rivals fail to compose" rows. ICML 2018. |
| `hupkes2020compositionality` | Compositionality Decomposed: How do Neural Networks Generalise? | Hupkes | 2020 | 1908.08351 | 10.1613/jair.1.11674 | Taxonomy of compositional generalisation (systematicity, productivity, …). Lets us name which kind our held-out corners test. JAIR. |
| `keysers2020measuring` | Measuring Compositional Generalization: A Comprehensive Method on Realistic Data (CFQ) | Keysers | 2020 | 1912.09713 | no DOI | Compound-divergence splits, i.e. principled held-out-combination construction. ICLR 2020. |
| `fodor1988connectionism` | Connectionism and cognitive architecture: A critical analysis | Fodor | 1988 | no arXiv | 10.1016/0010-0277(88)90031-5 | Origin of the systematicity argument. One sentence in the intro or related work. Cognition. |
| `lake2017building` | Building Machines That Learn and Think Like People | Lake | 2017 | 1604.00289 | 10.1017/S0140525X16001837 | Compositionality as a core ingredient. Optional background. Behavioral and Brain Sciences (Crossref issue date 2016, volume 40 = 2017). |

### 3b. Rivals and comparators (naive vs naive, matched on read-path params)

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `hochreiter1997long` | Long Short-Term Memory | Hochreiter | 1997 | no arXiv | 10.1162/neco.1997.9.8.1735 | The **LSTM** rival. Neural Computation. |
| `vaswani2017attention` | Attention Is All You Need | Vaswani | 2017 | 1706.03762 | no DOI | The **transformer** rival (incl. its refuse-token refusal comparator). NeurIPS 2017. |
| `nie2021controllable` | Controllable and Compositional Generation with Latent-Space Energy-Based Models (LACE) | Nie | 2021 | 2110.10873 | no DOI | **[PLACEHOLDER-ROW]** LACE rival, a **w15 placeholder row** (`lace-poe-rows`). Cite the method only, and cite `du2023reduce` for its sampler caveat. NeurIPS 2021 (arXiv comment). |
| `hinton2002training` | Training Products of Experts by Minimizing Contrastive Divergence | Hinton | 2002 | no arXiv | 10.1162/089976602760128018 | **[PLACEHOLDER-ROW]** PoE rival, a **w15 placeholder row**. Method citation for products of experts. Neural Computation. |

(MLP, "continuous MLP" and the lookup-table floor have no method citation; see the unfound list, U-8.)

### 3c. Precedent for matching parameters on the read path and reporting memory separately ("the ledger is memory size, printed separately, never matched")

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `borgeaud2021improving` | Improving language models by retrieving from trillions of tokens (RETRO) | Borgeaud | 2022 | 2112.04426 | no DOI | ⭐ Retrieval-augmented precedent: models are compared by **trainable parameters**, with the retrieval database reported separately. Venue not verified from the arXiv record (ICML 2022; cite-tagger should confirm). |
| `khandelwal2020generalization` | Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM) | Khandelwal | 2020 | 1911.00172 | no DOI | Non-parametric datastore beside a fixed parametric model, with memory not counted as parameters. ICLR 2020. |
| `lample2019large` | Large Memory Layers with Product Keys | Lample | 2019 | 1907.05242 | no DOI | ⭐ Memory reported separately from compute/parameters on the read path. Its product-key addressing is also the nearest ML precedent for a **tensor-product address book**. NeurIPS 2019. |
| `wu2022memorizing` | Memorizing Transformers | Wu | 2022 | 2203.08913 | no DOI | External kNN memory, with memory size reported separately. ICLR 2022. |
| `berges2024memory` | Memory Layers at Scale | Berges | 2024 | 2412.09764 | no DOI | Memory layers compared at matched compute and parameter counts. Venue not verified. |

### 3d. Deletion comparators (only "rival deletes by retrain-without" is a row)
Covered by §1d: `bourtoule2021machine` (SISA, rebuttal kit only), `guo2020certified`, `izzo2021approximate`, `golatkar2020eternal`. ⛔ None of them is a paper row, per `advisor-wave-plan-w12-w14.md` l.31–32. Cite them in the sentence that says why the only rival deletion route in the table is retrain-without.

### 3e. Encoder-adapt paragraph / store-does-not-shape-representation (minor)
No new citation needed beyond `montero2022lost` and `schott2021visual`. See U-9.

---

## §4 The maze: CMAZE-2b deletion transfer, the w14 verb finding (INTERFACE claim), the lookup floor

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `chevalierboisvert2023minigrid` | Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks | Chevalier-Boisvert | 2023 | 2306.13831 | 10.52202/075280-3209 | ⭐ **MiniGrid**, the gridworld family the maze tasks sit next to, and the w8 **negative** "MiniGrid composition-blind" (`advisor-wave-plan-w12-w14.md` l.52). NeurIPS 2023 (Crossref container). ⚠ Cite MiniGrid for the negative and for provenance only. **Do not imply that CMAZE is a MiniGrid environment** unless the harness docs say so; this scout did not check that (U-6). |
| `vafa2024evaluating` | (see §2) | | | | | The exact-judge validity notion behind the maze's engine-judged rows. |
| `zhao2022toward` | (see §2) | | | | | Compositional generalisation in world modelling. Framing for the maze section, and for the **w16 placeholder** CMAZE-T/O rows **[PLACEHOLDER-ROW]**. |

The w14 verb finding (the lookup-floor vs CLU merge lift, and the maze lookup as the within-maze floor) is **our own measurement**, so it cites its artefact, not a paper. No external citation is needed or appropriate.

---

## §5 Limitations → leaving note: the physics-CLU, the "Latent Dynamics Model" class, positioned in R-E (measured-law) grammar

These are the LDM-class positioning papers from `w12-positioning-scout/POSITIONING.md` §§1–2, 8, plus the prior art that CM-21 says **must** be cited.

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `hopfield1982neural` | Neural networks and physical systems with emergent collective computational abilities | Hopfield | 1982 | no arXiv | 10.1073/pnas.79.8.2554 | The associative-memory / energy-landscape origin of "wells". PNAS. |
| `krotov2016dense` | Dense Associative Memory for Pattern Recognition | Krotov | 2016 | 1606.01164 | no DOI | Positioning precedent P1: a new memory primitive introduced on XOR + MNIST. NIPS 2016 (journal-ref). |
| `ramsauer2021hopfield` | Hopfield Networks is All You Need | Ramsauer | 2021 | 2008.02217 | no DOI | Positioning precedent P2, the theorem/identity route (R-A/R-B). The **settling-off identity** sentence should be positioned against it. ICLR 2021 (per the positioning scout; the arXiv record carries no venue field). |
| `hoover2023energy` | Energy Transformer | Hoover | 2023 | 2302.07253 | 10.52202/075280-1197 | Positioning precedent P5: an energy-based primitive with a qualitative flagship section. NeurIPS 2023. |
| `gu2020hippo` | HiPPO: Recurrent Memory with Optimal Polynomial Projections | Gu | 2020 | 2008.07669 | no DOI | ⛔ **Must be cited** (CM-21(a)): HiPPO-LegS has a retention guarantee, so we never claim that attention/SSMs lack one. NeurIPS 2020 (not in the arXiv record; per the positioning scout; cite-tagger should confirm). |
| `gu2022efficiently` | Efficiently Modeling Long Sequences with Structured State Spaces (S4) | Gu | 2022 | 2111.00396 | no DOI | Positioning precedent P3, the matched-budget ablation route (R-C). Our physics ablation runs this route **backwards** (the dynamics cost composition). ICLR 2022 (arXiv comment). |
| `gu2024mamba` | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu | 2024 | 2312.00752 | no DOI | Positioning precedent P4, the mechanism-synthetic route (R-D), which we do **not** hold. COLM 2024, verified by the positioning scout from colmweb.org. ⛔ Do not cite the ICLR-2024 rejection (secondary sources only). |
| `chen2018neural` | Neural Ordinary Differential Equations | Chen | 2018 | 1806.07366 | no DOI | Positioning precedent P6: parity plus a property, with an explicit "Scope and Limitations". NeurIPS 2018. ⛔ Do not cite the Best-Paper status (secondary-only). |
| `cohen2021gradient` | Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability | Cohen | 2021 | 2103.00065 | no DOI | ⭐ **The grammar precedent for the leaving note (R-E):** a measured law with a parameter-free numerical prediction and no rival. ICLR 2021 (arXiv comment). |
| `frankle2019lottery` | The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | Frankle | 2019 | 1803.03635 | no DOI | R-E precedent: a hypothesis articulated as such, at parity. ICLR 2019 (journal-ref). |
| `kong2024reservoir` | Reservoir-computing based associative memory and itinerancy for complex dynamical attractors | Kong | 2024 | no arXiv (not checked) | 10.1038/s41467-024-49190-4 | ⛔ **Must be cited** (CM-21(b)): "retrieval is a rollout, not a weighted sum" is published here and is **not our novelty**. Nature Communications 15. |
| `graves2014neural` | Neural Turing Machines | Graves | 2014 | 1410.5401 | no DOI | ⛔ CM-21(c): the NTM was already soft and end-to-end differentiable, so "continuity is our NTM/DNC escape hatch" is FALSE. Cite it when that comparison comes up. Venue: arXiv only. |
| `graves2016hybrid` | Hybrid computing using a neural network with dynamic external memory (DNC) | Graves | 2016 | no arXiv | 10.1038/nature20101 | Same CM-21(c) point for the DNC. Nature. |
| `karuvally2025exponential` | Exponential Dynamic Energy Network for High Capacity Sequence Memory (EDEN) | Karuvally | 2025 | 2510.24965 | 10.52202/085713-1312 | ⛔ **Must be cited**: it narrows "tunable memory permanence". Only per-item, write-time-chosen permanence survives as ours. NeurIPS 2025. |

---

## §6 Follow-up (full physics-CLU evaluation)
No new external citation. It reuses §5's keys. The benchmark-requirements list cites `zhang2024unlearncanvas` (generative-unlearning benchmarks exist but score behavioural proxies) and `montero2021role` / `schott2021visual` (held-out-combination benchmarks exist but score no deletion). ⚠ See U-5: **no scout surveyed whether a benchmark with both already exists.**

---

## Appendix: software, optimisation, statistics

| cite key | title | first author | year | arXiv | DOI | what in our draft it supports |
|---|---|---|---|---|---|---|
| `kingma2015adam` | Adam: A Method for Stochastic Optimization | Kingma | 2015 | 1412.6980 | no DOI | Optimiser for the learned rivals and decoder, **if** the harness uses Adam. This scout did not check the harness (U-7). ICLR 2015. |
| `kidger2021equinox` | Equinox: neural networks in JAX via callable PyTrees and filtered transformations | Kidger | 2021 | 2111.00254 | no DOI | Equinox (eqx 0.13.4 is pinned in the workspace venv per COMMON.md). NeurIPS 2021 Differentiable Programming workshop. |
| `efron1979bootstrap` | Bootstrap Methods: Another Look at the Jackknife | Efron | 1979 | no arXiv | 10.1214/aos/1176344552 | **Conditional:** cite only if the CI95 on paired Δ is a bootstrap CI. Not confirmed from the evidence (U-7). Annals of Statistics. |

JAX itself (Bradbury et al., 2018, software) has **no arXiv and no DOI** that this scout resolved. If cite-tagger includes it, use a `@software` entry pointing to `github.com/jax-ml/jax`. That URL was not fetched here.

---

## ⛔⛔ CLAIMS FOR WHICH NO CITATION WAS FOUND (as important as the list above)

| # | claim in the draft | what was searched / why it stays uncited | what the draft must do |
|---|---|---|---|
| **U-1** | *"The intersection of compositional generation with exact deletion is empty"* (the positioning count, ≈4,050 works) | This is **our own systematic negative** (`NOVELTY.md` Addendum), not a citable result. No external paper states it. | Cite its artefact in a LaTeX comment; cite `priem2022openalex` only as the tool. Carry the residual-risk sentence ("low, not zero"). |
| **U-2** | *"World models need rule-valid, compositional, **exactly deletable** generation"*, taken as a conjunction | Rule-valid → `vafa2024evaluating`. Compositional → `zhao2022toward`, `kipf2020contrastive`. **Deletion as a world-model requirement: NO PAPER FOUND.** Nothing surfaced in the world-model papers above, and this scout ran no dedicated search for "unlearning in world models / model-based RL". | Write the deletion leg as **our motivation, in motivation grammar** ("a natural requirement is…"), not as a cited fact. ⚠ A dedicated search is DECLARED NOT-DONE here (reason: session scope). |
| **U-3** | *"Downdate the statistic, not the truncated factors"* | Our own sentence. Only the contrast class is citable (`hall2002adding`, `hall1999adding`, `jin2014evd`, `lee2021fast`), and the primitive is owned by `cao2015towards`. | Cite both sides; claim only the distinction. |
| **U-4** | The shorts' laws and validations entering "as this paper's own theory" (CRR, the settle half-life closed form, the "Latent Dynamics Model" class name), including the self-citation in the third person | The shorts are **non-archival**. This scout found **no public identifier** for them, did not search for one, and would break anonymity by looking one up under our name. | Either an anonymised `\cite{anonymous}` placeholder, if the venue's policy allows it (see `VENUE.md`), or no citation, with the theory presented in full. **The Head decides.** |
| **U-5** | *"No benchmark scores composition and deletion on the same object"* (benchmark-requirements section) | **No scout surveyed this** (`POSITIONING.md` NR-7, still true). UnlearnCanvas (deletion-only) and Montero/Schott (composition-only) were found, but that is not a survey. | Keep the draft's existing sentence that this is **not** a claim that the field lacks one. |
| **U-6** | CMAZE-2 / CMAZE-2b / CMAZE-T / CMAZE-O environment provenance | Our own environments. This scout did **not** check whether they are built on MiniGrid code. | Cite `chevalierboisvert2023minigrid` for the MiniGrid negative only, unless the harness docs confirm a derivation. |
| **U-7** | Optimiser, CI95 procedure, p-value test behind the paired Δ | The evidence files read here did not name them. The harness was not inspected (zero-compute spoke; it reads evidence only). | `kingma2015adam` and `efron1979bootstrap` are **conditional**. cite-tagger or draft-v2 must confirm from the harness before inserting them. |
| **U-8** | The MLP, "continuous MLP" and lookup-table rivals/floor | Generic baselines; no canonical paper. The "continuous MLP" appears to be a programme-specific construction. | No citation; describe them in the method section. |
| **U-9** | *"The store does not shape the representation"* (the encoder-adapt negative) | Our own negative. No external paper states it about this object. | Cite its artefact. |
| **U-10** | The exceptional point / defective 2×2 map at `h/h*` ≈ 1 (physics leaving note) | Standard linear algebra / non-Hermitian physics; **not searched** in this session. | If a citation is wanted, a textbook reference (e.g. Kato's perturbation theory) must be verified in a later pass. ⛔ Do not insert an unverified one. |
| **U-11** | LeCun 2022 position paper | OpenReview is blocked here, so it is unverifiable. | Listed as a candidate only; do not insert. |
| **U-12** | Mamba's ICLR-2024 rejection; Neural ODE's Best Paper award | Secondary sources only (`POSITIONING.md` §3.4, P6). | ⛔ Do not cite or state. |

## Summary counts (computed from this file's tables, not by hand)
- **Entries:** 97 unique cite keys. 96 are verified; `lecun2022path` is unverified and must not be inserted. On top of these there is 1 conditional software note (JAX). Two further table rows in §4 are "(see §2)" cross-references, not new entries.
- **arXiv IDs:** 76, every one re-resolved against the arXiv API in a final pass, and 0 invented. Entries without one say `no arXiv`.
- **DOIs:** 48, every one re-resolved in a final pass: 47 via Crossref, and 1 (`10.5281/zenodo.1207631`, a DataCite DOI) via doi.org. **1 recalled DOI was caught as wrong and replaced.** Every other entry says `no DOI`.
- **Claims with no citation found:** 12 (U-1 to U-12).
