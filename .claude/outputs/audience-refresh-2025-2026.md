# audience-refresh-2025-2026 — web-scout report

**DIAL DECLARATION (echoed): none — audience/literature scouting + citation verification; no performance claim; no laundering control applies.**

Task + acceptance criterion: rebuild BOTH audience profiles (NeurReps, PALM) on **2025–2026** material; recover NeurReps 2025 (+2024) accepted lists or declare failure with surfaces tried; settle PALM's lineage and build a 2025 proxy; deliver five Part-3 items per venue.
Status: **done** — with three counts/records declared unverified (F1, F4, F5 below).
Retrieval date: **2026-08-21** (all items unless stated).

## ⚠ DOWNSTREAM RECONCILIATION LIST (first-10-lines rule, AGENT_PROTOCOL §5) — needs an owner

1. ⛔⛔ **A 2026 preprint narrows V2's central novelty claim, in V2's own words.** **Mo, H. H. (2026), "Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks", arXiv:2605.03338** proves that a G-equivariant vector field has **≥ dim(G/H) zero Lyapunov exponents tangent to the group orbit**, and states that when protection is broken the direction "**can acquire a pseudo-gap**" and "**in our controlled breaking experiments this pseudo-gap predicts finite memory lifetime**" — on **S¹ path integration**, against **matched GRU/LSTM/orthogonal-RNN baselines**. That is V2's designed-symmetry ⇒ zero-mode ⇒ pseudo-gap ⇒ lifetime chain, published, three months before today. **Owner: the Advisor, before `v2-neurreps-reframe` writes §1/§2.** See §1.5 N1 for exactly what survives as ours and what does not.
2. **NeurReps 2026 has a THIRD track the prior scout did not see — a "Findings Track": *"High-impact collaborative work between experimentalists and theorists, in any standard preprint format. Single-blind, editorially reviewed by an advisory panel."* No page limit; archival status not stated on the site.** This is directly material to the Add.4 Ruling 1 / track-fit tension logged as reconciliation item 2 of `neurreps-audience-scout`. ⚠ Fit caveat, stated as fact not judgment: it is described as *collaborative work between experimentalists and theorists*; V2 has no experimentalist. **Owner: the Advisor.**
3. **PALM's facts have moved since `v5-scope-scout` (2026-08-19).** Submission deadline now reads **August 29, 2026** on the site (was August 24); the fifth invited speaker is no longer TBA — it is **Mariya Toneva (MPI for Software Systems)**, a brain–language-model alignment researcher, i.e. PALM's reviewer pool now contains a neuro-representational-geometry speaker. **Owner: the Advisor (venue facts), Head (timelines).**
4. **The previous vocabulary map is superseded (it was built on 2022/v197).** Four terms have shifted meaning or availability since 2022 — *symmetry breaking*, *canonicalization*, *world model*, *flow / time-parameterized symmetry* — and two of the shifts are **in our favour** (§3.3). **Owner: whoever writes V2 §2.**

---

# Answer first

Both lists were recovered from primary venue surfaces, not aggregators. **NeurReps 2025 (4th edition, NeurIPS 2025 San Diego, Dec 6–7) is resolved**: 121 poster titles enumerated from `neurips.cc/virtual/2025/workshop/109551`; **NeurReps 2024 (3rd edition, Vancouver, Dec 14) is resolved at 61 posters** from the equivalent 2024 page. **PALM is a first edition** (no predecessor found; site names no prior edition), so its 2025–2026 proxy is built from (i) the immediately preceding dedicated memory workshop, **MemAgents @ ICLR 2026** (70 accepted papers, list recovered — and **Weiwen Liu keynotes both MemAgents and PALM**), (ii) the 2025 adjacent workshops (MUGen, CCFM, Lock-LLM, Memorization/Trustworthy-FM, Long-Context FM), and (iii) the organizers'/speakers' own 2025–26 output. The single most consequential finding is reconciliation item 1: an independent May-2026 preprint has already published the symmetry-protected-zero-mode → pseudo-gap → memory-lifetime chain on S¹. The second is that **the PALM room now speaks physics** ("thermodynamic arbitration", "entropic memory") and **accepts theory papers** — while containing **almost nothing on deletion/right-to-be-forgotten**, which is exactly V5's ground.

---

# PART 1 — NeurReps: the gap closed

## 1.1 Surfaces tried and what each returned (per task instruction: declare, never infer)

| surface | result |
|---|---|
| `neurreps.org/accepted-submissions` | **404** (as prior scout found) |
| `neurreps.org/past-workshops`, `/past-editions`, `/call-for-papers` | **404** — the site is a single-page app; "Past Editions" is an on-page section with no per-year URLs |
| **`neurips.cc/virtual/2025/workshop/109551`** | ✅ **SUCCESS — full poster list + schedule** |
| **`neurips.cc/virtual/2024/workshop/84725`** | ✅ **SUCCESS — full poster list (61) + schedule** |
| OpenReview API v1 and v2 (`api.openreview.net`, `api2.openreview.net`, venueid queries) | **302 → `openreview.net/challenge`** (bot wall). Every `openreview.net/forum`/`/pdf` fetch returned the browser-verification page. **OpenReview is unusable via this toolchain today** |
| `web.archive.org` | **tool-level block** ("Claude Code is unable to fetch from web.archive.org") |
| dblp `db/conf/neurreps` | only **v197 (2022)** and **v228 (2023)**; confirms prior scout's F5 — **no PMLR volume exists for 2024 or 2025** |
| PMLR volume search | no v3/v4 NeurReps volume found |
| `github.com/neurreps` | 3 repos: `neurreps.github.io` (HTML, updated 2026-08-18), `awesome-neural-geometry`, `pmlr-v197-2022`. No accepted-list data file surfaced |

⚠ **F1 — the 2025 count is UNVERIFIED.** Three passes over the same page returned "88", "121" and "128" as the poster count; the page states no count. The **enumerated list below has 121 distinct titles** and its first/last entries match across passes. Treat **121 as an enumerated floor**, not a verified total. The 2024 count (61) is corroborated: the fetched page reported 61 and the workshop's own about-text (via search snippet) says *"61 accepted submissions presented as posters."*

## 1.2 Edition inventory (primary)

| edition | when/where | archival volume | accepted |
|---|---|---|---|
| 1st, 2022 | New Orleans, 2022-12-03 | **PMLR v197** (pub. 2023-02-07) | 21 |
| 2nd, 2023 | New Orleans, 2023-12-16 | **PMLR v228** (pub. 2024-08-02) | 23 |
| 3rd, 2024 | Vancouver, **2024-12-14**, West Ballroom C | **none** | **61** |
| 4th, 2025 | San Diego, **2025-12-07**, Upper Level Ballroom 6A | **none** | **121 enumerated** (F1) |
| 5th, 2026 | **Sydney, Dec 11–12, 2026** | Proceedings track → PMLR | CFP open, deadline **2026-08-24 AoE**, notification 2026-09-29 |

⇒ **The audience grew ~6× between the last archival volume and now.** Any profile built on v197 (21 papers, 2022) is describing 3.5 % of the current room.

## 1.3 NeurReps 2025 — accepted list (121 titles, verbatim, from `neurips.cc/virtual/2025/workshop/109551`)

1 Curvature Estimation on Data Manifolds via Diffusion-augmented Sampling · 2 A New Perspective for Graph Learning Architecture Design: Linearize Your Depth Away · 3 Koopman Autoencoders Learn Neural Representation Dynamics · 4 The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm · 5 SRTD: A Symmetric Divergence for Interpretable Comparison of Representation Topology · 6 Quantifying information stored in synaptic connections rather than in firing activities of neural networks · 7 The Representations of Deep Neural Networks Trained on Dihedral Group Multiplication · 8 An Information-Geometric View of the Platonic Hypothesis · 9 Learning from Frustration: Torsor CNNs on Graphs · 10 CAP_M: Curvature-Aware Pulling on Riemannian Manifolds · 11 A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction · 12 Sheaf Cohomology of Linear Predictive Coding Networks · 13 A Dendritic-Inspired Network Science Generative Model for Topological Initialization of Connectivity in Sparse Artificial Neural Networks · 14 Gauge Fiber Bundle Geometry of Transformers · 15 Persistent Homology Distances for Comparing Disease-Filtered Structural Connectomes · 16 The Binding Problem in Vision Models: Geometric, Functional, and Behavioral Approaches · 17 Causal Geometry of Batch Size and Generalisation · 18 Curvature Dynamic Black-box Attack · 19 Transformers Represent Causal Abstractions · 20 K-theoretic Persistent Cohomology · 21 **Learning rate collapse prevents training recurrent neural networks at scale** · 22 On a Geometry of Interbrain Networks · 23 Filter Equivariant Functions: A symmetric account of length-general extrapolation on lists · 24 Unifying Global Topology Manifolds and Local Persistent Homology for Data Pruning · 25 ECoNets: Rotation Equivariant Contrail Detection Neural Networks in Satellite Imagery · 26 **Slow Transition to Low-Dimensional Chaos in Heavy-Tailed Recurrent Neural Networks** · 27 Model manifold analysis suggests the human visual brain is less like an optimal classifier and more like a feature bank · 28 Beyond Pixels: A Differentiable Pipeline for Probing Neuronal Selectivity in 3D · 29 Sample Efficient Offline RL via T-symmetry Enforced Latent State-Stitching · 30 Brain network science modelling of sparse neural networks… · 31 Balancing Fairness and Accuracy in Graph Learning via Fairness-Constrained Rewiring · 32 Scalable GPU-Accelerated Euler Characteristic Curves · 33 Generalizable Representation Geometry for Grating Stimuli in V1 and ANNs · 34 A Comparative Empirical Study of Relative Embedding Alignment in Neural Dynamical System Forecasters · 35 Do Masked Autoencoders Learn a Human-Like Geometry of Neural Representation?… · 36 REM3DI: Learning smooth, chiral 3D molecular representations from equivariant atomistic foundation models · 37 The Geometry and Topology of Modular Addition Representations · 38 Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement · 39 Geometry matters: insights from Ollivier Ricci Curvature and Ricci Flow into representational alignment · 40 Contrastive Learning with Latent Tension Regularization for Tight Orbits · 41 Context-Dependent Manifold Learning in Dynamical Systems: A Neuromodulated Constrained Autoencoder Approach · 42 Group Convolutional Self-Attention for Roto-Translation Equivariance in ViTs · 43 Exploring Learnability in Dynamical Stochastic Networks: A Field-Theoretic Approach · 44 Unified Generative Latent Representation for Functional Brain Graphs · 45 LFMA: Parameter-Efficient Fine-Tuning via Layerwise Fourier Masked Adapter · 46 On the Impact of Topological Regularization on Geometrical and Topological Alignment in Autoencoders · 47 Saliency Thresholds in Neural Code and its Relation to the Power-Law, Gaussian, and Lambert W Function · 48 Deep neural network model of sound localization replicates "what" and "where" representations in auditory cortex · 49 Affect2Act: Graph Attention Networks for Emotion-Informed Decision Making · 50 The Geometry of Cortical Computation: Manifold Disentanglement and Predictive Dynamics in VCNet · 51 Inferring dynamical features from neural data through joint learning of latents factors and weights · 52 Boundary Guidance for Efficient 3D CT Vision–Language Reasoning · 53 Beyond I-Con: A Roadmap for Representation Learning Loss Discovery · 54 Radial-VCReg: More Informative Representation Learning Through Radial Gaussianization · 55 ⭐ **Flow Equivariant World Models: Structured Dynamics Outside the Field of View** · 56 ⭐⭐ **Symmetry-Regularized Learning of Continuous Attractor Dynamics** · 57 Learning representations on Lp hyperspheres… · 58 **On the geometry of recurrent spiking networks** · 59 From Extrapolation to Generalization: How Conditioning Transforms Symmetry Learning in Diffusion Models · 60 Topological Signatures of Altered Brain Network Centrality in ADHD · 61 Shape-Based Features Complement CLIP Features… · 62 Symmetry as Intervention; Causal Estimation with Data Augmentation · 63 Activation Matching for Explanation Generation and Circuit Discovery · 64 ⭐ **Measuring and Controlling Solution Degeneracy across Task-Trained Recurrent Neural Networks** · 65 ⭐ **Any-Subgroup Equivariant Networks via Symmetry Breaking** · 66 Causality ≠ Decodability, and Vice Versa… · 67 Data Augmentation: A Fourier Analysis Perspective · 68 From Finite to Infinite Groups: A Polynomial-Time Algorithm for Learning with Exact Invariances · 69 Factorized Prefrontal Geometry of Goal and Uncertainty… · 70 Exact Learning Dynamics of In-Context Learning in Linear Transformers… · 71 Logit-Based Losses Limit the Effectiveness of Feature Knowledge Distillation · 72 Hilbert geometry of the symmetric positive-definite bicone · 73 Unifying Regression and Uncertainty Quantification with Contrastive Spectral Representation Learning · 74 Beyond Parallelism: Synergistic Computational Graph Effects in Multi-Head Attention · 75 Composed Program Induction with Latent Program Lattice · 76 **Neurosymbolic Rabbit Brain: Fractal Attractor Geometry for Neural Representations** · 77 Topological Neural Data Analysis with Behavioral Constraint · 78 Neural Manifold Geometry Encodes Feature Fields · 79 Self-Supervised Learning from Structural Invariance · 80 Why all roads don't lead to Rome: Representation geometry varies across the human visual cortical hierarchy · 81 ⭐ **The Cue or not the Cue? A Mechanistic Study of Memory Mechanisms in RNNs** · 82 Model Transferability Informed by Embedding's Topology · 83 ⭐ **Do traveling waves make good positional encodings?** · 84 MAPS: A Dataset for Controlled Probing of Representational Topology in Vision Models · 85 Time-Resolved Circuit Discovery in RNNs via Windowed Causal Interventions and Local Linearization · 86 **Complete Characterization of Gauge Symmetries in Transformer Architectures** · 87 Compositional Symmetry as Compression: Lie-Pseudogroup Structure in Algorithmic Agents · 88 **Emergent Riemannian geometry over learning discrete computations on continuous manifolds** · 89 Towards the Identification of Latent Structures in Language Embeddings · 90 **Dimensionality of population-level latent mechanisms encoding spatial representations** · 91 Curvature Meets Bispectrum: A Correspondence Theory for Transformer Gauge Invariants · 92 Neural Fields Meet Attention · 93 **On neural circuits of working memory sequence permutation: optimizing circuit architectures via Cayley graphs** · 94 Representational Homomorphism Error Predicts Compositional Generalization In Language Models · 95 An Analytical Framework for Multi-Area Balanced Networks · 96 Cannistraci-Hebb Training of Convolutional Neural Networks · 97 Contrast inversion reveals hierarchical asymmetries… · 98 Event2Vec: A Geometric Approach to Learning Composable Representations of Event Sequences · 99 **How does training shape the Riemannian geometry of neural network representations?** · 100 DIET-CP… · 101 Homological Representation Learning for Molecular Graphs · 102 Covering Relations in the Poset of Combinatorial Neural Codes · 103 ⭐⭐ **Poisson-Algebraic Parallel Scan: A Fast Symplectic Framework for Neural Hamiltonians** · 104 Response Patterns to Rotation Angle in a Rotation Pretext Task…: An Observation and a Negative Result · 105 **Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency** · 106 Dual-Stream EEG Decoding for 3D Visual Perception · 107 **Mapping neural representations of topologically non-trivial spaces** · 108 Meta-learning three-factor plasticity rules for structured credit assignment with sparse feedback · 109 Modeling Human Vision with Differential Geometry · 110 Measure Before You Look: Grounding Embeddings Through Manifold Metrics · 111 Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport · 112 Provable Low-Frequency Bias of In-Context Learning of Representations · 113 The Human Brain as a Combinatorial Complex · 114 Tracking Memorization Geometry throughout the Diffusion Model Generative Process · 115 Theoretical Analysis of HyperCube Objective for Group Representation Learning · 116 Graph Mixing Additive Networks · 117 ⭐ **Shaping Latent Geometry with Noise-Injected Hopfield Dynamics** · 118 On Uncertainty Calibration for Invariant Functions · 119 **Equivariance by Local Canonicalization: A Matter of Representation** · 120 Geometric Priors for Generalizable World Models via Vector Symbolic Architecture · 121 Local Predictions, Global Learning: Radial Basis Function Networks for Spatially-Aware Predictive Coding

**Invited speakers 2025 (verified, from the same page):** SueYeon Chung (*Computation-Aware Representation Geometry*), Surya Ganguli (*Mathematical Approaches to Interpretability*), Max Tegmark (*How Symmetry & Geometry Help Generalize*), Katrin Franke (*Visual System Neural Code*), Razvan Pascanu (*ML Models & OOD Generalization*), Yue Song (*Neural Oscillations in ML*). Spotlights: Tahmasebi (#68), Goel (#65), Elumalai (#28), Zavatone-Veth (Riemannian geometry of training), Sun (#17), Aswani (#3), Huang (#64), Wang (#31), Pellegrino (Riemannian geometry & discrete computations).

## 1.4 NeurReps 2024 — accepted list (61, from `neurips.cc/virtual/2024/workshop/84725`)

Full list retrieved. CHLU-relevant subset, verbatim: **#3 A minimalistic representation model for head direction system** · **#15 Dynamical symmetries in the fluctuation-driven regime: an application of Noether's theorem to noisy dynamical systems** ⭐⭐ · **#37 Hamiltonian Matching for Symplectic Neural Integrators** ⭐⭐ · **#47 Storing overlapping associative memories on latent manifolds in low-rank spiking networks** ⭐ · #5 In-Context Symmetries: Self-Supervised Learning through Contextual World Models · #20 Symmetry-Aware Generative Modeling through Learned Canonicalization · #24 Communication subspaces align with training in ANNs · #25 Constrained Belief Updating and Geometric Structures in Transformer Representations · #38 Visualizing Loss Functions as Topological Landscape Profiles · #45 Does equivariance matter at scale? · #46 Geometric Signatures of Compositionality Across a Language Model's Lifetime · #53 Neural Network Symmetrisation in Concrete Settings · #59 Modeling dynamic neural activity by combining naturalistic video stimuli and stimulus-independent latent factors · #60 Neural Representational Geometry of Concepts in Large Language Models · #42 sa-SVAE… preserved neural dynamics across animals. (Remainder is graph/geometric-DL methods, TDA, and applications.)

⭐ **#15 (Noether's theorem applied to *noisy* dynamical systems) and #37 (symplectic integrators) together establish that Noether + symplectic language is already NeurReps-native — the prior 2022-based profile had no such item.** V2/V-series can use both without glossary overhead.

## 1.5 ⭐ Nearest-neighbour set for the NeurReps papers, refreshed (2025–2026 only)

**N1 — Mo, H. H. (2026). "Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks." arXiv:2605.03338 [cs.NE], submitted 2026-05-05.** Sole author (Hanson Hanxuan Mo, Dept. of Applied Mathematics & Computational Neuroscience Center, University of Washington — affiliation single-sourced from a search snippet; arXiv abstract page primary-verified). Abstract, **verbatim in part**: *"For a finite-dimensional autonomous C¹ vector field equivariant under a Lie group G, we prove that any compact invariant set carrying a uniformly nondegenerate group-orbit bundle with stabilizer type H has, at points where the Lyapunov spectrum is defined, at least dim(G/H) zero Lyapunov exponents tangent to the group orbit. … When this protection is explicitly broken, the formerly protected direction can acquire a **pseudo-gap**; in our controlled breaking experiments this **pseudo-gap predicts finite memory lifetime**."* Also: *"we train an exactly equivariant recurrent cell on velocity-input S¹ path integration across six seeds and compare it with matched GRU, LSTM, and orthogonal-RNN baselines. The learned equivariant cell preserves step equivariance to 3.2×10⁻⁸, has a near-zero group-tangent exponent under the zero-input autonomous restriction…"*
⛔ **NARROWS: V2's "designed symmetry ⇒ protected zero mode; break it ⇒ pseudo-gap ⇒ finite lifetime" claim, including the word *pseudo-gap*, the S¹ setting, and the matched-recurrent-baseline protocol.** Also narrows the framing "the flat direction is guaranteed by symmetry, not tuned" — that is his Theorem.
✅ **What is still ours after N1** (stated conservatively, for the Advisor to rule on): Mo gives **existence and qualitative prediction** — zero exponents exist; a pseudo-gap "predicts" a finite lifetime. He does **not** give (as far as the abstract states, and it is the only text verified) **a closed-form exchange rate** n₁/₂ ∝ μ⁻² with a measured slope, **a mass-independent floor**, **an exceptional-point crossover at εμ ≈ γ/2 with √(h−h\*) onset**, or **invariance of those laws under a corrective anchor**. Our μ² is a **curvature of a trained potential** (units 1/time²) whereas his is a **Lyapunov exponent** (units 1/time) — the same conversion caveat the prior scout flagged for Ságodi applies here and is now load-bearing twice. ⚠ **Single-sourced: I read the abstract page only. The full PDF must be read before V2 §1 is written** — if his paper contains a lifetime *law* rather than a lifetime *prediction*, more than the framing is at risk.

**N2 — [authors unavailable] (2025). "Symmetry-Regularized Learning of Continuous Attractor Dynamics." NeurReps 2025 poster; OpenReview forum `W8Gf7CYCo8`; a PDF exists at `openreview.net/pdf/172d42fe...pdf`.** Claim (from search-index text of the PDF, **NOT primary-verified — OpenReview blocked**): augments **variational state space models** with a symmetry regularizer for a chosen continuous invariance; enforces that the learned vector field **approximately commute with the generators of a predefined symmetry group** (rotations, for ring attractors), the regularizer being *"the L2 norm of the Lie bracket"* measuring non-commutativity. ⛔ **NARROWS: "impose a continuous symmetry to obtain a continuous attractor in a learned model."** This is the *emergent-arm* competitor: they get the flat direction by soft regularization; we get it by construction. **⚠ F2 — authors, exact title casing and abstract UNVERIFIED; arXiv has zero results for this title. Do not cite until read.**

**N3 — [authors unavailable] (2025). "Poisson-Algebraic Parallel Scan: A Fast Symplectic Framework for Neural Hamiltonians." NeurReps 2025 poster; also an ICLR 2026 submission, `openreview.net/forum?id=ZjZo4h80XL`, reported **withdrawn**.** Claim (index text, **not primary-verified**): a *"Poisson algebraic decomposition of the learned Hamiltonian"*; *"by embedding polynomial generators explicitly closed under Poisson brackets, PAPS induces an associative Lie-group structure that naturally facilitates parallel-scan (prefix-sum) computation"*; motivated by HNNs' *"inherent sequential integration that prevents parallel computation"* and instability when extrapolating. ⭐ **The single most architecturally adjacent 2025 NeurReps item to CHLU's core** (symplectic + Hamiltonian + sequence-scan). ⚠ Anonymous/withdrawn — **F3: authors not recoverable; cite as a NeurReps 2025 poster only, or not at all.**

**N4 — Huang, A., Singh, S. H., Martinelli, F. & Rajan, K. (2025). "Measuring and Controlling Solution Degeneracy across Task-Trained Recurrent Neural Networks." arXiv:2410.03972v3 (v1 2024-10-04, v3 2025-11-20); comment: *"Accepted to Advances in Neural Information Processing Systems (2025)."*** Primary-verified via arXiv API. 3,400 networks, four neuroscience tasks; task complexity and feature learning **reduce** neural-dynamics degeneracy but **increase** weight-space degeneracy; larger networks and structural regularization reduce degeneracy at all levels. **Relation:** the audience's current standard for *"how much does the trained solution vary across seeds/recipes"* — the exact instrument a referee will ask V2 to apply to its designed-vs-emergent gap. Spotlighted at NeurReps 2025.

**N5 — Lillemark, H. J., Huang, B., Zhan, F., Du, Y. & Keller, T. A. (2026). "Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments." arXiv:2601.01075** (v2 title variant: *"…: Structured Memory for Dynamic Environments"*; NeurReps 2025 poster title: *"…: Structured Dynamics Outside the Field of View"*). ⚠ **Three title variants observed — F4; verify before citing.** Claim (search-index): unifies self-motion and external object motion as **one-parameter Lie-group "flows"**; *"provides a stable latent world representation over hundreds of timesteps"*; **zero-shot generalization to unseen flow velocities** and robustness **far beyond the training horizon**.
**N5b — Keller, T. A. (2025). "Flow Equivariant Recurrent Neural Networks." arXiv:2507.14793v2 (v1 2025-07-20, v2 2025-12-01); comment: *"NeurIPS '25, Spotlight."*** Primary-verified via arXiv API. Verbatim: *"we extend equivariant network theory to this regime of 'flows' — one-parameter Lie subgroups capturing natural transformations over time"*; standard RNNs *"are generally not flow equivariant"*; flow-equivariant models *"significantly outperform their non-equivariant counterparts in terms of training speed, length generalization, and velocity generalization."*
⛔ **NARROWS: "a continuous symmetry carried through time in a recurrent state buys long-horizon stability."** ⭐ **But it also creates our slot:** flow equivariance is the audience's *architectural* answer; it prices nothing and has no damping/temperature. This pair is the **single best positioning anchor for V2 in the 2025–26 room** — same object (one-parameter Lie flow in a recurrent state, long-horizon), orthogonal contribution (they generalize, we price).

**N6 — Haputhanthri, U., Storan, L., Jiang, Y., Raheja, T., Shai, A., Akengin, O., Miolane, N., Schnitzer, M. J., Dinc, F. & Tanaka, H. (2025). "Understanding and controlling the geometry of memory organization in RNNs." arXiv:2502.07256, q-bio.NC, 2025-02-11.** Primary-verified (arXiv abs page). Abrupt learning in short-term-memory tasks: search phase → accuracy plateau → sudden loss drop; **geometric restructuring in phase space precedes the drop**; propose **temporal consistency regularization** that speeds training and **facilitates attractor formation**. ⭐ **Note the author list: Nina Miolane is a NeurReps organizer and Dinc/Tanaka are the authors of the "ghost mechanism" paper the prior scout logged as B9.** This is the organizers' own live line on training-dynamics-shapes-attractors — the strongest "cite your reviewers' current work" candidate.

**N7 — the 2024 pair, still the best in-venue precedents for our formal register:** *"Dynamical symmetries in the fluctuation-driven regime: an application of Noether's theorem to noisy dynamical systems"* and *"Hamiltonian Matching for Symplectic Neural Integrators"*, both NeurReps 2024 posters (records: `neurips.cc/virtual/2024/workshop/84725`; **no author lists retrieved — F5**). ⭐ **Their existence is the finding:** Noether-in-noise and symplectic integration are accepted NeurReps subject matter as of 2024, so CHLU's core need not be introduced apologetically.

**N8 — [authors unavailable] (2025). "Shaping Latent Geometry with Noise-Injected Hopfield Dynamics." NeurReps 2025 poster.** Record only. Relevant because it puts **energy-based associative memory + injected noise** inside this venue in 2025 — the CD/Langevin half of CHLU has a venue precedent. Not primary-verified beyond the title.

## 1.6 Topic distribution, NeurReps 2025 ⚠ MY INFERENCE (keyword-based over the 121 titles; the venue publishes no partition)

⭐ **One partition is NOT my inference — the venue's own 2025 award categories, verbatim from the CFP announcement: "Neuroscience & Interpretability", "Topological & Geometric ML", "Symmetry & Equivariance"** ($500 each, sponsored by NewTheory AI). That is the organizers' three-way view of their own room; the prior profile's clusters (geometric DL / neuro / topology+learning theory) are **close but miss that interpretability has been fused into the neuroscience bucket.**

My keyword tally (titles only; papers can double-count; ±3 per bucket):

| bucket (my label) | approx. count | share |
|---|---|---|
| Symmetry / equivariance / group structure / gauge | **~24** | ~20 % |
| Representational geometry of brains & models, alignment | **~20** | ~17 % |
| Graph / geometric-DL methods & applied | **~30** | ~25 % |
| **Dynamics, attractors, RNNs, Hamiltonian/Koopman/ODE** | **~18** | **~15 %** |
| Topology / TDA / persistent homology / sheaves | **~14** | ~12 % |
| Mechanistic interpretability & LLM internals | **~11** | ~9 % |

⇒ **The "dynamics" bucket (~15 %) did not exist as a bucket in v197 (2022)**, where the nearest items were a single sine-network extrapolation paper and the grid-cell recurrent model. It is now a standing CFP topic ("Dynamics of neural representations"; "Symmetries, dynamical systems, and learning") **and** ~1 in 7 accepted posters. **This is the strongest single fact in favour of V2's fit at NeurReps.**

## 1.7 Method & vocabulary census — what recurs in 2025 titles (⚠ terms whose meaning/availability SHIFTED since 2022 are flagged)

Recurring 2025 terms, by rough frequency: *geometry / geometric* (very high) · *manifold* · *representation(al) geometry* · *topology / topological / persistent homology / cohomology* · *curvature (Riemannian, Ricci, Ollivier-Ricci)* · *equivariant / equivariance / invariance* · *alignment* · *dynamics* · *attractor* · *circuit discovery / mechanistic* · *in-context learning* · *world model* · *gauge* · *bispectrum* · *canonicalization* · *Koopman* · *sheaf* · *symmetry breaking* · *flow*.

⚠ **SHIFTED SINCE 2022 — the four that matter to us:**
1. **"symmetry breaking"** — in 2022 this audience said *"relax equivariance"* (van der Ouderaa & van der Wilk). In 2025 the phrase is literal and titular: *"Any-Subgroup Equivariant Networks via **Symmetry Breaking**"* (#65, a spotlight). ⭐ **CHLU's spontaneous-symmetry-breaking / Goldstone framing no longer needs translation — it is now the room's own word.** This is a change in our favour and the old map's "no-equivalent" marking on this axis is wrong for 2025.
2. **"canonicalization"** — the 2022 term for our coset coordinate was *fundamental-domain projection* (Aslan et al.). The 2024/2025 standard is **canonicalization** (2024 #20 *Symmetry-Aware Generative Modeling through Learned Canonicalization*; 2025 #119 *Equivariance by Local Canonicalization: A Matter of Representation*). ⛔ **Anyone still writing "fundamental domain projection" as the audience's term is four years out of date.**
3. **"world model"** — absent from v197; present from 2024 (#5) and a 2026 CFP topic (*"Equivariant world models for robotics"*); 2025 has two (#55, #120). A latent state that must stay coherent over a long horizon is now discussed under this label.
4. **"flow" / "one-parameter Lie subgroup" / "time-parameterized symmetry"** — created by Keller (2025) and now a live subfield (N5, N5b, #55). ⛔ In 2022 "flow" in this room meant *Ricci flow* or *normalizing flow*; in 2025 it also means a **continuous time symmetry of the input stream**. **Collision hazard: NeurReps 2025 contains BOTH senses in the same poster list (#39 Ricci flow vs #55 flow equivariance).** Disambiguate on first use.

⚠ **Also newly present and NOT in the 2022 vocabulary:** *gauge* (3 papers in 2025: #14, #86, #91) — a physics word that is now safe here; *pseudo-gap* — appears in the adjacent literature (N1) with **our** meaning; *neutral mode / zero Lyapunov exponent* — N1 again. And ⚠ *"geometric approaches to mechanistic interpretability"* was an explicit **2025 CFP topic** and **is absent from the 2026 CFP topic list** — the 2026 list instead adds *statistical learning theory*, *learning and leveraging group structure in data*, *equivariant world models for robotics*, *geometric structure in language*, *geometric/topological analysis of generative models*.

## 1.8 What this audience rewards / rejects — evidenced from 2024–2025 accepted sets

- ⭐ **It accepts negative results explicitly, and not only rhetorically.** The EA track's own text names *"negative findings"*, and **2025 #104 is literally titled "…: An Observation and a Negative Result."** Evidence, not impression.
- ⭐ **It accepts small/toy-scale and single-mechanism studies.** #7 (dihedral group multiplication), #37 (modular addition), #47 (Lambert W in a saliency threshold), #102 (posets of combinatorial neural codes) are not scale papers.
- ⭐ **It accepts pure theory with light or no experiments** — #68 (polynomial-time algorithm for exact invariances, spotlighted), #112 (provable low-frequency bias), #115 (theoretical analysis of an objective), #20 (K-theoretic persistent cohomology), #86 (complete characterization of gauge symmetries).
- ⭐ **It accepts physics-native machinery without apology:** Noether (2024 #15), symplectic integrators (2024 #37), Poisson algebra (2025 #103), field theory (#43), Hopfield + noise (#117), gauge/fiber bundles (#14, #86, #91), chaos/Lyapunov (#26).
- **Scale is discussed but not required** — #45 *"Does equivariance matter at scale?"* (2024) is the room asking the question, not a scale threshold.
- **Three tracks now**, each with a different bar: Proceedings (9 pp, **archival PMLR**, double-blind) · Extended Abstract (4 pp, non-archival, double-blind, framed for early-stage/negative/opinion/datasets) · **Findings (no page limit, single-blind, editorially reviewed by an advisory panel, for "high-impact collaborative work between experimentalists and theorists")**.

---

# PART 2 — PALM: lineage settled, proxy built

## 2.1 Lineage: **first edition** (declared, with the surfaces checked)

- `palm-neurips-2026.github.io` (retrieved 2026-08-21) **names no prior edition and no "2nd"/"3rd" ordinal anywhere**; the header is simply *"PALM · NeurIPS 2026"*.
- No workshop named PALM appears in the NeurIPS 2025 workshop list (56 workshops, enumerated below in 2.2), in the ICML 2025 list (33 workshops, enumerated), or in search under the organizers' names.
- ⇒ **Conclusion: PALM @ NeurIPS 2026 is a first edition.** ⚠ Stated as "no predecessor found across those surfaces", not as a claim the site makes.

**Facts that CHANGED since `v5-scope-scout` (2026-08-19) — both from the venue's own site, 2026-08-21:**
- **Submission deadline reads August 29, 2026** (v5-scope-scout recorded August 24, 2026). Notification 2026-09-29 unchanged; workshop Dec 12 or 13, 2026, **Paris**.
- **The fifth invited speaker is filled: Mariya Toneva (MPI for Software Systems).** Speakers now: Weiwen Liu (SJTU) · Tsendsuren Munkhdalai (Google) · Niloofar Mireshghallah (CMU) · Ali Behrouz (Cornell) · **Mariya Toneva**.
- Organizers unchanged (Fritz, Oh, Abdelnabi, Shen, Lopes, Puerto, Sheth, Jung). Seven topics unchanged. (⚠ I re-verified the topic *headings*; I did not re-transcribe the example lists — `v5-scope-scout` §1.2's verbatim example lists stand and were not contradicted.)

## 2.2 The proxy, and how it is constructed

**Construction rule (stated per acceptance criterion 2):** PALM has no predecessor, so the reviewer-pool proxy = (A) the nearest *dedicated* memory workshop that has already run, weighted highest because it shares a speaker with PALM; (B) the 2025 adjacent workshops in continual learning / unlearning / memorization / long context, which supply the 2025 layer the Head asked for; (C) PALM's own organizers' and speakers' 2025–26 output.

### (A) ⭐ **MemAgents @ ICLR 2026** — the sharpest proxy
*ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems*, **Monday 2026-04-27**, Rio de Janeiro (`iclr.cc/virtual/2026/workshop/10000792`; site `sites.google.com/view/memagent-iclr26`; proposal `openreview.net/forum?id=U51WxL382H`). **70 accepted papers — full list recovered** (see 2.3). Keynotes: Aditi Raghunathan (CMU) · Chelsea Finn (Stanford) · **Weiwen Liu (SJTU)** · Fred Sala (Snorkel) · Mengye Ren (NYU) · Volker Tresp (LMU) · Jeff Clune (UBC) · Jeff Z. Pan (Edinburgh). Organizers include Hinrich Schütze, Yunpu Ma, Ercong Nie (MCML) + international collaborators. ⚠ *">110 submissions"* is single-sourced (MCML news item).
⭐ **Weiwen Liu keynotes MemAgents AND is a PALM invited speaker** — a direct, verified personnel link between the proxy and the target.
**Sibling, same season:** *Lifelong Agents: Learning, Aligning, Evolving* — **1st edition at ICLR 2026** (`iclr.cc/virtual/2026/workshop/10000805`), **2nd at COLM 2026** (Oct 9, 2026; deadline Jul 3, 2026).

### (B) The 2025 layer (the Head's explicit ask) — there was **no dedicated memory workshop in 2025**
That absence is itself the finding: the dedicated agent-memory workshop wave is a **2026** phenomenon (MemAgents ICLR'26, Lifelong Agents ICLR'26/COLM'26, PALM NeurIPS'26, TTCL NeurIPS'26, CL4FMAgents NeurIPS'26). The 2025 predecessors are adjacent, not central:
- **NeurIPS 2025** (56 workshops, list enumerated from `nips.cc/virtual/2025/events/workshop`): #30 **"AI That Keeps Up: Workshop on Continual and Compatible Foundation Model Updates (CCFM)"** (Dec 7, San Diego; speakers Akata, Aljundi, Kanan, Rish, Schmidt; topics *"efficient continual learning at scale"*, *"time-continual learning"*, *"frequent backward-compatible model updates"*, *"dynamic evaluation"*, *"train/test contamination"*; accepted papers on OpenReview only ⇒ **not retrievable, see F6**) · #24 **Lock-LLM: Prevent Unauthorized Knowledge Use from LLMs** · #14 **Multi-Turn Interactions in LLMs** · #45 **LAW 2025: Bridging Language, Agent, and World Models** · #38 Regulatable ML · #34 Scaling Environments for Agents. ⛔ **No workshop with "memory" in the title.**
- **ICML 2025** (33 workshops, enumerated from `icml.cc/virtual/2025/events/workshop`): #14 **"Machine Unlearning for Generative AI" (MUGen)**, Jul 18, Vancouver — organizers Patil, Mazeika, Hodgkins, Basart, Y. Liu, K. Lee, Bansal, B. Li; speakers Carlini, Ling Liu, Mehnaz, Sijia Liu, Triantafillou, Hase, Cooper, Cyphert; CFP topics **verbatim** include *"Irreversible unlearning resistant to fine-tuning attacks"*, *"Differential privacy, exact unlearning, and provable guarantees"*, *"Standardized evaluation frameworks for robust unlearning"* · #19 **"The Impact of Memorization on Trustworthy Foundation Models"** · #33 **"The Second Workshop on Long-Context Foundation Models"** · #4 **"The 1st Workshop on Vector Databases"** · #3 Tiny Titans (on-device FM learning).
⇒ **For V5 specifically:** the 2025 record shows that **"provable/exact unlearning with guarantees"** is an established, invited-speaker-level topic (MUGen), while **the 2026 memory-workshop wave has almost no deletion content** (2.3). V5 sits in the seam.

### (C) PALM's own speakers/organizers, 2025–2026 output (reviewer-pool predictor)
- **Ali Behrouz (invited).** Titans (NeurIPS 2025, arXiv:2501.00663) → **ATLAS: Learning to Optimally Memorize the Context at Test Time, arXiv:2505.23735** → **Nested Learning: The Illusion of Deep Learning Architectures, NeurIPS 2025, arXiv:2512.24695** (Behrouz, Razaviyayn, Zhong, Mirrokni; Google Research blog Nov 2025). Nested Learning's three claims: optimizers **are associative-memory modules**; **self-modifying Titans**; a **"continuum memory system"** generalizing long-/short-term memory; implementation **HOPE**. ⭐⭐ **This is the most CHLU-adjacent thing in PALM's speaker set: a *continuum* of memory timescales, framed as nested optimization with associative memory at every level.** ⚠ V5 and V2 currently cite Titans only (V2) — **Nested Learning/ATLAS are not in either draft** and are the speaker's *current* work, not his 2025-January work.
- **Tsendsuren Munkhdalai (invited).** Infini-attention (arXiv:2404.07143) — bounded-size compressive memory. Unchanged from prior scout.
- **Niloofar Mireshghallah (invited).** 2025–26: *"Position: Privacy Is Not Just Memorization!"* (arXiv:2510.01645) — argues the field over-focuses on verbatim memorization and under-weights **inference-time context leakage and agent capabilities**; **"CIMemories: A Compositional Benchmark for Contextual Integrity of Persistent Memory in LLMs" (ICLR 2026)** — reported finding: violations rise **0.1 % → 9.6 %** as usage scales from 1 to 40 tasks, **25.1 %** with repeated sampling. ⚠ *CIMemories numbers are single-sourced (search summary); verify before quoting.* ⭐ **Relevance: her current instrument is long-horizon persistent-memory leakage — App E.5's AUC/d′ material lands squarely in her expertise, as the prior scout warned, but the framing she now pushes is "not just memorization", which slightly *reduces* the membership-inference exposure and *increases* the contextual-integrity exposure.**
- **Mariya Toneva (invited, NEW).** MPI-SWS; ERC Starting Grant *BrainAlign* (brain-aligned LMs for long-range language understanding). 2026 items reported: *"Temporal Context Reinstatement Drives Episodic-Like Order Memory in Long-Context Language Models"* ⭐ (episodic memory in LMs, cognitive-science framing), *"What Brain Data Adds to Language Model Training"* (CoNLL 2026), *"Tracking Equivalent Mechanistic Interpretations Across Neural Networks"*. ⚠ **All Toneva 2026 titles are single-sourced from a search summary of her dblp/OpenReview profile — F7; verify each before citing.** ⭐ **Consequence: PALM topic 4 ("Neuroscience-Inspired & Cognitive Memory Models") now has a speaker who will actually read it.** That is a fit gain for any CHLU framing that leans on consolidation/replay language.
- **Organizers' centre of mass** (Fritz, Oh, Abdelnabi, Sheth, Puerto — CISPA/KAIST/ELLIS): privacy, security, prompt injection, contextual privacy in agents. Abdelnabi's 2025–26 line (MAGPIE multi-agent contextual privacy; "Got a Secret? LLM Agents Can't Keep It") is **evaluation of leakage in agent systems**.

## 2.3 MemAgents ICLR 2026 — the 70 accepted papers (the proxy census)

Full list retrieved (`iclr.cc/virtual/2026/workshop/10000792`). Topic distribution ⚠ **MY INFERENCE**, keyword tally over 70 titles:

| bucket (my label) | ~count | representative titles (verbatim) |
|---|---|---|
| **Memory architectures for LLM agents** (graph/episodic/hierarchical stores) | **~24** | *"MEMORY IS RECONSTRUCTED, NOT RETRIEVED: GRAPH MEMORY FOR LLM AGENTS"* · *"Human-Like Lifelong Memory: A Neuroscience-Grounded Architecture for Infinite Interaction"* · *"MIRROR: Complementary Encoding and Reconstructive Consolidation for Persistent State in LLM Systems"* · *"GAM: Hierarchical Graph Memory"* · *"SimpleMem: Efficient Lifelong Memory for LLM Agents"* · *"ENGRAM: Effective, Lightweight Memory Orchestration"* |
| **Benchmarks & evaluation of memory** | **~12** | *"AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications"* · *"ShiftBench: Measuring Recovery of Agent Memory Under Distribution Shift"* · *"PROCED-MEM: Benchmarking Procedural Memory Retrieval"* · *"CloneMem: Benchmarking Long-Term Memory for AI Clones"* · *"Evaluating Memory Structure in LLM Agents"* · *"Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory"* |
| **Efficiency / compression / KV-cache** | **~9** | *"CAOTE: Optimizing KV Cache Memory Through Attention Output Error-based Token Eviction"* · *"Norm-Guided KV-Cache Eviction"* · *"R-KVHash"* · *"Agentic Memory Should Localize Compression"* · *"MemFly: On-the-Fly Memory Optimization via Information Bottleneck"* |
| **Continual / test-time / self-evolving learning** | **~10** | *"Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates"* · *"Learning to Continually Learn via Meta-learning Agentic Memory Designs"* · *"Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"* · *"SkillRL"* |
| ⭐ **Theory / provable claims** | **~4** | *"Toward a Theory of Hierarchical Memory for Language Agents"* · **"Tool use is provably more scalable than in-weight memory for Large Language Models"** · *"Provenance-Aware Tiered Memory"* (*"From Lossy to Verified"*) · *"Episodic Memory from Compression Boundaries in Latent Representation Space"* |
| ⭐ **Physics/thermodynamics-framed memory** | **2** | **"Look Before You Leap: Thermodynamic Arbitration of Parametric and Non-Parametric Knowledge in LLM Agents via Self-Regulating Memory Architectures"** · **"Entropic Memory: A Thermodynamics-Inspired Consolidation Mechanism for Lifelong Agent Learning"** |
| **Safety / privacy / attack** | **~3** | *"Memory Injection Attacks on LLM Agents via Query-Only Interaction"* · *"SABER: Small Actions, Big Errors"* · *"Epistemic Memory Failures in Long-Form Narrative Agents"* |
| **Surveys / position** | 2 | *"From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms"* · *"LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents"* |

⭐⭐ **Two findings the Advisor should see immediately.**
1. **"Adaptive Memory Admission Control For LLM Agents"** is an accepted MemAgents 2026 paper (#54 in the recovered list). ⛔ **The word "admission control" — CHLU's own dial name — is now in this room's title vocabulary.** Record only (no authors/abstract retrieved). ⚠ **This may narrow the admission dial's framing novelty; it must be read before any admission claim is written for a PALM-class venue.**
2. ⛔ **Deletion / right-to-be-forgotten is essentially ABSENT from the 70.** Zero titles contain "delete", "deletion", "unlearn", "forget-me", or "right to be forgotten"; **"forgetting" appears once**, and in an efficiency sense (*"Alleviating Forgetfulness of Linear Attention by Hybrid Sparse Attention and Contextualized Learnable Token Eviction"*). Meanwhile **PALM's own CFP names deletion in four of seven topics.** ⇒ The venue is *asking* for a topic its nearest predecessor community did not supply. **That is the clearest evidenced statement of where V5's contribution is scarce.** ⚠ This is a *scarcity* fact, not a strategy recommendation — the venue call is the Advisor's.

## 2.4 PALM vocabulary census (from 2.3's 70 titles + PALM's own CFP + speaker work)

Terms that recur and read as native: *agentic memory* · *episodic / procedural / semantic memory* · *consolidation* · *memory layer / memory substrate* · *retrieval vs. utilization* · *context engineering* · *KV-cache eviction / token eviction* · *compression* · *provenance* · *lifelong / long-horizon* · *test-time learning* · *self-evolving* · *memory injection* · *contextual integrity* · *admission control* (new) · *continuum memory* (Behrouz) · *thermodynamic / entropic* (2 papers).

⚠ **SHIFTED / NEW vs. the 2024–early-2025 material V5 was written against:**
- **"forgetting"** in this room now overwhelmingly means **catastrophic forgetting / efficiency-driven eviction**, *not* deletion-on-request. ⛔ **A bare "forgetting" in a PALM abstract will be read as continual learning.** V5 must say *deletion*, *erasure*, or *right-to-be-forgotten* when it means those.
- **"memory"** in 2026 defaults to an **external store over an LLM**, not a latent state. Any CHLU sentence saying "memory" without a qualifier gets read as a RAG/graph store.
- ⭐ **Physics vocabulary is now permitted here** (thermodynamic arbitration, entropic consolidation) — a change from the 2023–24 baseline where MemoryBank's Ebbinghaus curve was the only physics analogy. **CHLU's temperature/damping register no longer needs an apology in this room**, though it still needs a definition.
- **"admission"** — newly occupied (see 2.3 finding 1).
- **"contextual integrity"** — Mireshghallah's frame; supersedes plain "memorization" as the privacy register.

## 2.5 What PALM's proxy rewards / rejects — evidenced

- ⭐ **Theory is accepted:** *"Toward a Theory of Hierarchical Memory for Language Agents"* and *"Tool use is **provably** more scalable than in-weight memory"* were both accepted at MemAgents. PALM's own CFP names *"theoretical perspectives"*, *"position papers"* and *"negative results"* as contribution types (verbatim, `v5-scope-scout` §1.3).
- **Benchmarks are the single most rewarded artifact type** (~12 of 70 are named benchmarks: AMA-Bench, ShiftBench, PROCED-MEM, CloneMem, ATOD, DialSim, MemoryAgentBench…). A paper with no benchmark is in a minority but not excluded.
- **Toy scale is tolerated when the point is mechanism** (*"LLMs Can't Play Hangman"*, *"Do LLMs Benefit From Their Own Words?"*, *"Evaluating AGENTS.md"*).
- **Negative/diagnostic results appear** (*"Epistemic Memory Failures… A Deployment Study"*, *"Diagnosing Retrieval vs. Utilization Bottlenecks"*).
- ⛔ **What is essentially not present in the proxy:** anything with a **closed-form retention law**, a **temperature**, or a **provable deletion guarantee**. The nearest are the two thermodynamics-*inspired* papers, which (title-level only) read as mechanisms, not laws.

---

# PART 3 — the updated vocabulary maps

Legend: **[E]** exact · **[A]** approximate, difference stated · **[⛔N]** no equivalent.
⚠ **This map SUPERSEDES the map in `outputs/neurreps-audience-scout.md` §"Part 3", which was built from PMLR v197 (2022).** Rows marked **CHANGED** differ from that map; unmarked rows are carried forward unchanged and remain correct.

## 3.1 NeurReps (2025–2026 basis)

| our term | their **current** term | fit | note |
|---|---|---|---|
| flat / neutral direction | *zero Lyapunov exponent tangent to the group orbit*; **symmetry-protected neutral mode**; *tangent direction of a continuous attractor* | **[E]** | **CHANGED** — Mo 2026 (N1) supplies the exact phrase *"symmetry-protected"* and *"neutral mode"*. Prefer these over "flat direction"; they are now the sharpest available. |
| erosion / broken flat direction | **pseudo-gap** | **[E]** | **CHANGED — and this is a claim risk, not just vocabulary.** N1 uses *pseudo-gap* for our object and links it to memory lifetime. Adopt the word, and cite. |
| designed vs emergent symmetry | *exact/architectural equivariance* vs **symmetry breaking** / *approximate or relaxed equivariance* / *learned canonicalization* | **[E]** | **CHANGED** — 2022's *"relax equivariance"* is superseded by titular **symmetry breaking** (#65) and **canonicalization** (#119, 2024 #20). |
| coset register | *canonicalization / canonical form*; *orbit coordinate*; *the encoded continuous variable* | **[A]** | **CHANGED** — the audience's word for the coordinate is now **canonicalization**, not *fundamental-domain projection*. Still ⛔ no equivalent for a coset coordinate used as a **storage device**. |
| symplectic / conformally symplectic map | *symplectic integrator*; *Hamiltonian neural network*; **Poisson algebra / Poisson bracket** | **[E→A]** | **CHANGED — upgraded.** 2024 #37 (Hamiltonian Matching for Symplectic Neural Integrators) and 2025 #103 (Poisson-Algebraic Parallel Scan) make this in-vocabulary. Only the *conformal* qualifier needs the one-clause gloss (J⊤ΩJ = (1−γ)Ω). |
| conservation law / Noether | **Noether's theorem** | **[E]** | **CHANGED — upgraded.** 2024 #15 applies Noether to *noisy* dynamical systems inside this venue. No apology needed. |
| symmetry carried through time in a recurrent state | **flow equivariance** / *one-parameter Lie subgroup* / *time-parameterized symmetry* | **[A]** | **NEW ROW.** Keller's 2025 coinage; now the room's frame for exactly our setting. ⚠ **"flow" collides with Ricci flow and normalizing flow in the same poster list** — disambiguate. |
| latent state that must stay coherent over a long horizon | **world model** (when the task is prediction/control) | **[A]** | **NEW ROW.** Absent in 2022; a 2026 CFP topic. Use only where the task is genuinely predictive. |
| spectral mass μ² | *transverse curvature*; *normal-direction eigenvalue* — ⚠ **and note the unit clash with Lyapunov exponents (1/time) now that N1 makes exponents the default currency** | **[A]** | Carried, and **hardened**: with N1 in the room, printing the μ²(1/time²) ↔ λ(1/time) conversion is no longer optional. |
| exceptional point | *(still not native)* | **[⛔N]** | Carried unchanged. Nothing in 2024/2025 uses it. |
| retention half-life n₁/₂ | *memory lifetime* (N1 uses exactly *"finite memory lifetime"*); *time constant*; *diffusion coefficient D* | **[A→E]** | **CHANGED** — *"memory lifetime"* is now attested verbatim in the adjacent literature. Bridge to it on first use. |
| solution variability across seeds/recipes | **solution degeneracy** | **[E]** | **NEW ROW.** N4 (spotlighted 2025) is the standard instrument; a referee may ask for it. |
| the budget cube (μ, γ, T) | — | **[⛔N]** | Carried unchanged. |
| store / atom / register (CLU nouns) | — | **[⛔N]** | Carried unchanged. |

## 3.2 PALM (2026 basis) — new map (no prior map existed)

| our term | their current term | fit | note |
|---|---|---|---|
| the store / latent memory | **memory layer / memory substrate / agentic memory** | **[A]** | Their default referent is an *external* store over an LLM. Our latent-state store must be distinguished in the same sentence or it is misread. |
| admission (dial) | **memory admission control** | **[E]** | ⚠ **NEWLY OCCUPIED** — a MemAgents ICLR 2026 accepted title. Read it before claiming the dial. |
| lifetimes / decay (dial) | *retention*, *consolidation*, *eviction*, *expiry*, *forgetting* (⚠ = catastrophic forgetting here) | **[A]** | ⛔ Never write bare "forgetting" for deletion. |
| deletion / erasure with a guarantee | *unlearning*, *right-to-be-forgotten*, *invalidation*, *soft delete/tombstone* | **[A]** | The 2025 unlearning community (MUGen) owns *provable* language; the 2026 agent-memory community does not use it at all (2.3 finding 2). |
| isolation (dial) | *memory isolation between agents* (PALM CFP topic 2, verbatim), *contextual integrity* (Mireshghallah), *provenance* | **[A]** | *Contextual integrity* is the rising term; *isolation* is in the CFP verbatim. |
| compute-adaptive reads | *anytime*, *cost-sensitive routing*, *compute allocation* | **[A]** | Attested: *"Did You Check the Right Pocket? Cost-Sensitive Store Routing"*, *"Compute Allocation for Reasoning-Intensive Retrieval Agents"* (both MemAgents 2026). |
| temperature / damping / energy | *thermodynamic*, *entropic* | **[A]** | ⭐ Newly permitted (two 2026 accepted titles). Define once; do not assume the physics. |
| multi-timescale retention | **continuum memory system** (Behrouz) | **[A]** | Invited speaker's own 2025 coinage; the nearest thing to a "spectrum of lifetimes" in this room. |

---

# Confidence & gaps

**Primary-verified (venue's own surfaces, 2026-08-21):** NeurReps 2025 poster list + schedule + speakers (`neurips.cc/virtual/2025/workshop/109551`) · NeurReps 2024 poster list (61) + organizers (`.../2024/workshop/84725`) · NeurReps 2026 three tracks incl. **Findings**, dates, topics (`neurreps.org`) · NeurReps volume inventory (dblp: only v197, v228) · NeurIPS 2025 workshop list (56) · ICML 2025 workshop list (33) · MUGen CFP topics/organizers/speakers/dates (`mugenworkshop.github.io`) · CCFM speakers/topics/date (`sites.google.com/view/ccfm-neurips2025`) · MemAgents accepted list (70) + keynotes + date (`iclr.cc/virtual/2026/workshop/10000792`) · PALM site (organizers, 5 speakers, 7 topics, deadline 2026-08-29) · arXiv abstracts, read from the arXiv record: **2605.03338** (Mo), **2507.14793v2** (Keller), **2410.03972v3** (Huang et al.), **2502.07256** (Haputhanthri et al.).

**Single-sourced / NOT verified — do not cite without a second pass:**
- **F1** NeurReps 2025 total count (88 / 121 / 128 across three passes; 121 enumerated).
- **F2** *Symmetry-Regularized Learning of Continuous Attractor Dynamics* — **authors and abstract unknown**; zero arXiv results; OpenReview blocked.
- **F3** *Poisson-Algebraic Parallel Scan* — authors unknown (withdrawn ICLR 2026 submission ⇒ still anonymous); claim text index-sourced.
- **F4** *Flow Equivariant World Models* — **three title variants** across NeurReps listing / arXiv v1 / arXiv v2; abstract index-sourced (only the sibling 2507.14793 was read primary).
- **F5** NeurReps 2024 items #15 (Noether-in-noise) and #37 (Hamiltonian Matching) — **titles only; no author lists, no abstracts.**
- **F6** CCFM 2025, MUGen 2025, Lock-LLM 2025 **accepted-paper lists are all OpenReview-only** ⇒ unretrievable through this toolchain. The 2025 layer of the PALM proxy is therefore built from CFPs, speakers and organizers, **not** from accepted papers. **Declared, not inferred.**
- **F7** Toneva's and Mireshghallah's 2026 titles/numbers (incl. CIMemories 0.1 %→9.6 %→25.1 %) — search-summary-sourced.
- **F8** MemAgents ">110 submissions" — one aggregator (MCML news).
- **F9** Mo's affiliation (UW Applied Math / Computational Neuroscience Center) — search snippet only.

**⛔ Tool-level blockers to record for the Hub:** **OpenReview (all endpoints, API v1/v2 and web) returns a bot challenge**, and **web.archive.org is blocked at the tool level**. Every OpenReview-hosted accepted list in this ecosystem is therefore out of reach from this agent; a human or an authenticated fetch is required. This is the reason F2, F3, F6 are open.

**What to search next, priority order:** (1) **the full PDF of arXiv:2605.03338** — decides how much of V2's claim survives; (2) the two blocked NeurReps 2025 PDFs (`W8Gf7CYCo8`, `ZjZo4h80XL`) and **"Adaptive Memory Admission Control For LLM Agents"** — three papers that each touch a CHLU claim by name; (3) NeurReps 2026's **Findings track** archival status and reviewer/advisory panel; (4) CCFM/MUGen accepted lists via an OpenReview-capable route; (5) Behrouz's *Nested Learning* (arXiv:2512.24695) full text — the "continuum memory system" formulation.

---

# BibTeX (house pattern; traps in `note`)

```bibtex
@article{mo2026symmetryprotected,
  title={Symmetry-Protected {L}yapunov Neutral Modes in Equivariant Recurrent Networks},
  author={Mo, Hanson Hanxuan},
  journal={arXiv preprint arXiv:2605.03338}, year={2026},
  note={SINGLE author. Submitted 2026-05-05; cs.NE. NO venue on the record -- cite as preprint.
        THEOREM: >= dim(G/H) zero Lyapunov exponents tangent to the group orbit for a G-equivariant C^1 field.
        Uses the word ``pseudo-gap'' for the broken case and links it to ``finite memory lifetime''; S^1 path
        integration vs matched GRU/LSTM/orthogonal-RNN. *** NARROWS V2's central novelty framing -- read the
        PDF before drafting. *** Abstract read from the arXiv abs page 2026-08-21; affiliation single-sourced.}}

@article{keller2025flowequivariant,
  title={Flow Equivariant Recurrent Neural Networks},
  author={Keller, T. Anderson},
  journal={arXiv preprint arXiv:2507.14793}, year={2025},
  note={SOLE author. v1 2025-07-20, v2 2025-12-01. arXiv comment states ``NeurIPS '25, Spotlight'' --
        cite the NeurIPS 2025 record if a proceedings entry exists, else preprint + comment.
        Coins ``flow equivariance'' for one-parameter Lie subgroup symmetries over time.
        ``standard RNNs are generally not flow equivariant''. Retrieved 2026-08-21.}}

@article{lillemark2026flowequivariantwm,
  title={Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments},
  author={Lillemark, Hansen Jin and Huang, Benhao and Zhan, Fangneng and Du, Yilun and Keller, T. Anderson},
  journal={arXiv preprint arXiv:2601.01075}, year={2026},
  note={WARNING -- THREE title variants observed: this one (abs page), ``...: Structured Memory for Dynamic
        Environments'' (html v2), and ``...: Structured Dynamics Outside the Field of View'' (NeurReps 2025
        poster listing). VERIFY the version you cite. Author list and abstract are index-sourced, not primary.}}

@inproceedings{huang2025degeneracy,
  title={Measuring and Controlling Solution Degeneracy across Task-Trained Recurrent Neural Networks},
  author={Huang, Ann and Singh, Satpreet H. and Martinelli, Flavio and Rajan, Kanaka},
  booktitle={Advances in Neural Information Processing Systems}, year={2025},
  note={arXiv:2410.03972 (v1 2024-10-04, v3 2025-11-20); arXiv comment: ``Accepted to Advances in Neural
        Information Processing Systems (2025)''. FOUR authors; Rajan senior -- never ``Rajan et al.''.
        3,400 networks, 4 tasks. Also a NeurReps 2025 spotlight. The audience's standard degeneracy instrument.}}

@article{haputhanthri2025memorygeometry,
  title={Understanding and controlling the geometry of memory organization in {RNN}s},
  author={Haputhanthri, Udith and Storan, Liam and Jiang, Yiqi and Raheja, Tarun and Shai, Adam and
          Akengin, Orhun and Miolane, Nina and Schnitzer, Mark J. and Dinc, Fatih and Tanaka, Hidenori},
  journal={arXiv preprint arXiv:2502.07256}, year={2025},
  note={TEN authors. q-bio.NC, 2025-02-11. NINA MIOLANE IS A NEURREPS ORGANIZER and Dinc/Tanaka wrote the
        ``ghost mechanism'' paper (arXiv:2501.02378). Temporal consistency regularization facilitates
        attractor formation. Best ``cite your reviewers' current work'' candidate for V2.}}

@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={Advances in Neural Information Processing Systems}, year={2025},
  note={arXiv:2512.24695. FIRST AUTHOR IS A PALM 2026 INVITED SPEAKER. Introduces the ``continuum memory
        system'' and HOPE; claims optimizers ARE associative-memory modules. Neither V2 nor V5 cites it.
        Record via arXiv listing + Google Research blog (Nov 2025); abstract NOT primary-verified.}}

@misc{behrouz2025atlas,
  title={{ATLAS}: Learning to Optimally Memorize the Context at Test Time},
  author={Behrouz, Ali and others},
  year={2025}, note={arXiv:2505.23735. FULL AUTHOR LIST NOT VERIFIED -- do not print ``and others'' in a
        submission; resolve first. PALM invited speaker's Titans follow-up.}}

@misc{mireshghallah2025position,
  title={Position: Privacy Is Not Just Memorization!},
  author={Mireshghallah, Niloofar and others},
  year={2025}, note={arXiv:2510.01645. FULL AUTHOR LIST NOT VERIFIED. First author is a PALM 2026 invited
        speaker. Argues the field over-indexes on verbatim memorization vs inference-time/agentic leakage --
        relevant to how App E.5 will be read at PALM.}}

@misc{neurreps2025workshop,
  title={{NeurReps} 2025: 4th Workshop on Symmetry and Geometry in Neural Representations},
  howpublished={NeurIPS 2025 Workshop, San Diego, 2025-12-07},
  year={2025}, note={NON-ARCHIVAL: no PMLR volume exists for 2024 or 2025 (dblp indexes only v197/v228).
        121 poster titles enumerated from neurips.cc/virtual/2025/workshop/109551 on 2026-08-21;
        EXACT COUNT UNVERIFIED (page states none; three fetch passes returned 88/121/128).
        Cite individual posters by title + ``NeurReps 2025 workshop poster'', never as a PMLR entry.}}

@misc{memagents2026workshop,
  title={{MemAgents}: {ICLR} 2026 Workshop on Memory for {LLM}-Based Agentic Systems},
  howpublished={ICLR 2026 Workshop, Rio de Janeiro, 2026-04-27},
  year={2026}, note={70 accepted papers enumerated from iclr.cc/virtual/2026/workshop/10000792 on 2026-08-21.
        Keynote Weiwen Liu is ALSO a PALM 2026 invited speaker -- the verified personnel link between the
        proxy and PALM. Contains ``Adaptive Memory Admission Control For LLM Agents'' (occupies our dial name).}}
```

---

## Proposed handover updates (for the Hub)

1. **NeurReps audience profile is now on 2025–2026 data.** `outputs/neurreps-audience-scout.md` (v197/2022 basis) is **superseded for Part 3 (the vocabulary map)** and **supplemented for Parts 1–2**; its bridge literature (continuous attractors, drift, fine-tuning problem) remains valid and is not re-litigated here. Editions resolved: 2024 = 61 posters, 2025 = 121 enumerated; **no PMLR volume for either**.
2. **NeurReps 2026 has three tracks, not two** — Proceedings (9 pp, archival PMLR, double-blind) · Extended Abstract (4 pp, non-archival, double-blind) · **Findings (no page limit, single-blind, editorially reviewed, "collaborative work between experimentalists and theorists", archival status unstated)**. Add.4 Ruling 1's track analysis should be re-run against three options. Deadline on site: **2026-08-24 AoE**.
3. **PALM is a first edition.** Proxy = MemAgents @ ICLR 2026 (70 papers, recovered) + 2025 adjacent workshops (MUGen, CCFM, Lock-LLM, Memorization/Trustworthy-FM, Long-Context FM) + organizer/speaker output. **No dedicated memory workshop existed in 2025** — the wave is 2026.
4. **PALM venue facts changed since Add.28:** deadline now **Aug 29, 2026**; fifth speaker filled = **Mariya Toneva** (brain–LM alignment, episodic memory in long-context LMs). PALM topic 4 (neuro-inspired memory) now has a matching speaker.
5. ⛔ **New citation fence:** V2's zero-mode/pseudo-gap/lifetime chain must be scoped against **Mo (2026), arXiv:2605.03338** before drafting; V5's/any admission claim must be scoped against **"Adaptive Memory Admission Control For LLM Agents"** (MemAgents 2026). Both are **reconciliation items with no owner until the Hub assigns one.**
6. **Tooling fact worth banking:** OpenReview is fully bot-walled from this agent (API v1, API v2, forum, pdf) and web.archive.org is tool-blocked. **The `neurips.cc/virtual/<year>/workshop/<id>` and `iclr.cc/virtual/<year>/workshop/<id>` pages are the working substitute for any OpenReview-hosted workshop list** — this is the reusable technique that closed the standing NeurReps 2025 gap.

## Flags

- **F1** NeurReps 2025 exact accepted count unverified (88/121/128 across passes; 121 enumerated). Never print a total; print "121 titles listed on the NeurIPS virtual page".
- **F2** *Symmetry-Regularized Learning of Continuous Attractor Dynamics*: authors + abstract **unknown**. ⛔ Do not cite until read.
- **F3** *Poisson-Algebraic Parallel Scan*: anonymous (withdrawn ICLR 2026 submission). Cite as a NeurReps 2025 poster or not at all.
- **F4** *Flow Equivariant World Models*: three title variants; pick and verify one.
- **F5** NeurReps 2024 #15 (Noether-in-noise) and #37 (Hamiltonian Matching): **titles only**, no authors — but their *existence* is the load-bearing fact, and that is primary-verified.
- **F6** CCFM / MUGen / Lock-LLM accepted lists unretrievable (OpenReview-only). The 2025 layer of the PALM proxy rests on CFPs + speakers, and this is declared, not inferred.
- **F7** Toneva 2026 titles and Mireshghallah CIMemories numbers are search-summary-sourced.
- **F8** ⚠ **Terminology collision inside the NeurReps room:** "**flow**" means (a) Ricci flow, (b) normalizing flow, and (c) a one-parameter Lie subgroup of time-parameterized symmetries — all three in the 2025 poster list. Disambiguate on first use. (This joins the prior scout's F8 "conformal" collision.)
- **F9** ⚠ **Claim-widening pressure points, 2026 edition (extends the prior scout's F7):** adopting *"symmetry-protected neutral mode"* or *"pseudo-gap"* without citing Mo (2026) reads as appropriation; using *"memory lifetime"* without our metric-naming rule reads as a claim about a general lifetime; using *"admission control"* at a PALM-class venue without citing the MemAgents 2026 paper of that name is the same failure on the other side.
