# neurreps-audience-scout — web-scout report

**DIAL DECLARATION (echoed): none — literature scouting + citation verification; no performance claim; no laundering control applies.**

Task + acceptance criterion: verify every work in the Head's NeurReps audience census (records + BibTeX + what each actually claims), deliver the bridge literature (continuous attractors/drift, representational drift) with an honest no-claim sentence per item, and a vocabulary map marked exact/approximate/no-equivalent. Read-only.
Status: **done** (rev. 2 — second pass closed 3 of 4 declared gaps and **overturned one of my own first-pass recommendations**; see item 4 below).
Retrieval dates: census + Part 3 on **2026-08-20**; gap-closure pass **2026-08-21**.

## ⚠ DOWNSTREAM RECONCILIATION LIST (first-10-lines rule, AGENT_PROTOCOL §5) — needs an owner

1. **The Head's census is NeurReps *2022* (PMLR v197), not the current audience.** All ~17 named works resolve exactly to the 1st NeurReps volume (21 papers, Dec 2022). It is four years stale. The **NeurReps 2026 CFP topic list** (§1.3b, verbatim) is what V2 §2 should be written to. **Owner: the Advisor, before `v2-neurreps-reframe` writes §2.**
2. **Track-fit flag for the Head (new, not in Add.4 Ruling 1):** the 2026 site describes the Extended Abstract track as *"Early-stage results, negative findings, opinion pieces, or novel datasets."* V2 is none of those. The track matching V2's maturity (Proceedings, 9 pp) is the archival PMLR one Ruling 1 forbids. A real tension, not a citation issue. (§1.3b.)
3. **Do not cite v197 as "NeurReps 2023."** Workshop 2022-12-03; PMLR published 2023-02-07; PMLR keys are `*23a` while the front matter reads "PMLR 197, 2022". Pick one convention. (F1.)
4. ⛔⛔ **I RETRACT my own first-pass novelty sentence.** Rev. 1 of this report proposed that V2 could claim the fine-tuning problem's perturbation is "identified — it is the training objective," with a cure. **The gap-closure pass found prior art that makes that sentence an over-claim in this audience's room: Renart, Song & Wang (2003) already established the destruction-plus-homeostatic-cure pattern for a continuous attractor, and Vafidis et al. (2022) already showed a learning rule that *achieves* the tuning.** The correctly-scoped sentence is in §2.3. **Owner: the Advisor — this must be caught before drafting, not at referee.**

---

# Answer first

The Head's census **is** PMLR volume 197 — *Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations* (NeurReps 2022) — verified in full from the primary PMLR volume page: 21 papers, every Head description matched to a record, including the "topology and learning theory" cluster (McGuire, Davies, Dönmez, Akhtiamov, Tian). The highest-value finding is **Part 2**: this audience already owns our object under another name. A **continuous attractor** is a manifold of equilibria *marginally stable tangent to the manifold and stable normal to it*; its tangent is a **zero eigenvalue / zero Lyapunov exponent**; and its two known failure modes are **noise-driven diffusion along the manifold** (Burak & Fiete 2012, with an explicit diffusion coefficient and ⟨Δθ²⟩ = 2DΔt) and the **"fine-tuning problem"** — destruction of the flat direction by perturbation of the dynamical law, *including by learning* (Ságodi et al., NeurIPS 2024). Our V2 results map onto that vocabulary almost term-for-term. **But the prior art is deeper than it first looks: the destroy-and-homeostatically-restore story already exists (Renart, Song & Wang 2003), so V2's anchor result is a *re-derivation in a new setting*, not a new phenomenon.** What survives as genuinely ours is the quantitative price list — the closed-form μ⁻² half-life with its crossover and floor, measured on a *trained* potential. And throughout: we make no claim about biological systems and model no neural data.

---

# Part 1 — the census, verified

**Primary source for §1.1–§1.3:** `https://proceedings.mlr.press/v197/` (volume index: full titles, full author lists, page ranges), retrieved **2026-08-20**. Abstracts from `proceedings.mlr.press/v197/<slug>.html`, same date. Volume: *Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations*, eds. Sophia Sanborn, Christian Shewmake, Simone Azeglio, Arianna Di Bernardo, Nina Miolane; New Orleans, **2022-12-03**; published **2023-02-07** as PMLR v197.

⚠ **Cluster assignment below is MY INFERENCE.** Records verified; the partition into the Head's three clusters is not something PMLR states. Verified counts: the neuroscience cluster is **exactly 8** (volume items 15–22 — a clean match to the Head's "~8"); geometry/equivariance is 7 unambiguous (items 2–6, 13, 21) with items 7, 8, 14 as the plausible remainder to reach "~9"; topology/learning-theory is 5 (items 9–12, 14), with item 13 straddling.

### 1.1 Geometric DL / equivariance

| # | record (verified) | what it ACTUALLY claims (quote where the fetch returned verbatim) |
|---|---|---|
| G1 | **Shutty, N. & Wierzynski, C. (2023).** "Computing representations for Lie algebraic networks." PMLR 197:1–21. | *"a novel algorithm that finds representations of arbitrary Lie groups given only the structure constants of the associated Lie algebra"*, a toolkit, a relativistic point-cloud benchmark, and *"the first object-tracking model equivariant to the Poincaré group."* ⭐ **Closest census work to CHLU's relativistic framing — and a 2022 firstness claim we must not collide with.** |
| G2 | **Chau, H. Y., Qiu, F., Chen, Y. & Olshausen, B. (2023).** "Disentangling images with Lie group transformations and sparse coding." PMLR 197:22–47. | Lie-group theory + sparse coding in a Bayesian model factoring images into shape + transformation, transformations *constrained to form an n-dimensional torus representation*; on full MNIST recovers *"basic digit shapes and the natural transformations such as shearing and stretching contained in this data."* Self-described as *"the simplest known Bayesian mathematical model for building unsupervised factorized representations."* |
| G3 | **van der Ouderaa, T. F. A. & van der Wilk, M. (2023).** "Sparse Convolutions on Lie Groups." PMLR 197:48–62. | Continuous Lie-group filters from *"a small finite set of basis functions through anchor points"* rather than MLP hypernetworks; *"Regular convolutional layers appear as a special case… at equal memory complexity"*; the basis filters serve networks that **maintain or relax** equivariance. ⭐ *"relax equivariance"* is this audience's own phrase for our designed-vs-emergent axis. |
| G4 | **Klee, D., Biza, O., Platt, R. & Walters, R. (2023).** "Image to Icosahedral Projection for SO(3) Object Reasoning from Single-View Images." PMLR 197:64–80. | *"a novel architecture based on icosahedral group convolutions that reasons in SO(3) by learning a projection of the input image onto an icosahedron"*; approximate rotational equivariance from ordinary 2D images; *"outperforms reasonable baselines."* |
| G5 | **Sangalli, M., Blusseau, S., Velasco-Forero, S. & Angulo, J. (2023).** "Moving frame net: SE(3)-equivariant network for volumes." PMLR 197:81–97. | [paraphrase from the PMLR abstract page] Computes the moving frame **once at the input stage** rather than per layer, proves SE(3)-equivariance, improves on most MedMNIST3D datasets at much lower compute. |
| G6 | **Aslan, B., Platt, D. & Sheard, D. (2023).** "Group invariant machine learning by fundamental domain projections." PMLR 197:181–218. | Pre-processes by *"project[ing] the input data into a geometric space which parametrises the orbits of the symmetry group"*; efficient projection algorithm; improved accuracy on Hodge numbers of CICY matrices. ⭐ The audience's canonical quotient/coset-coordinate paper — nearest published thing to our *coset register* as a representation choice, **but it is a static input encoding, not a memory.** |
| G7 | **Robin, D. A. R., Scaman, K. & Lelarge, M. (2023).** "Periodic signal recovery with regularized sine neural networks." PMLR 197:98–110. | *"multi-layer perceptrons with ReLU activations are provably unable to perform this task"*; sine activations + non-convex regularisation are *"several orders of magnitude better than its competitors for distant extrapolation (beyond 100 periods of the signal)."* ⭐ Long-horizon extrapolation on a periodic signal — V2's Exp-I territory, inside the audience's own volume. |
| G8 | **Thakur, A., Abrol, V. & Sharma, P. (2023).** "Does Geometric Structure in Convolutional Filter Space Provide Filter Redundancy Information?" PMLR 197:111–121. | *"analyses the convolutional layer filter space using simplical geometry to establish a relation between filter relevance and their location on the simplex"*; extremal-point filters are least redundant. Low relevance to V2. |
| G9 | **Xu, D., Gao, R., Zhang, W.-H., Wei, X.-X. & Wu, Y. N. (2023).** "Conformal Isometry of Lie Group Representation in Recurrent Network of Grid Cells." PMLR 197:370–387. | *"Algebraically, we study the Lie group and Lie algebra of the recurrent transformation as a representation of self-motion. Geometrically, we study the conformal isometry of the Lie group representation where the local displacement of the activity vector in the neural space is proportional to the local displacement of the agent in the 2D physical space. Topologically, the compact and connected abelian Lie group representation automatically leads to the torus topology…"*; conformal isometry yields hexagonal patterns and accurate path integration in *"a simple non-linear recurrent model that underlies the continuous attractor neural networks of grid cells."* ⭐⭐ **The most important census item for V2: a Lie-group representation carried in a recurrent state, on a torus, over a continuous attractor — the audience's own instance of everything V2 discusses, and the natural place to say what we add (a price list for time).** |

### 1.2 Neuroscience

| # | record (verified) | what it ACTUALLY claims |
|---|---|---|
| N1 | **Jude, J., Perich, M. G., Miller, L. E. & Hennig, M. H. (2023).** "Capturing cross-session neural population variability through self-supervised identification of consistent neuron ensembles." PMLR 197:234–257. | *"drifts in activity of individual neurons and instabilities in neural recording devices can be substantial, making stable decoding over days and weeks impractical"*; *"self-supervised training of a deep neural network can be used to compensate for this inter-session variability"*, requiring *"only… a single recording session for training the model"*, as *"a step towards reliable, recalibration-free brain computer interfaces."* ⛔ **First author is JUDE — never "Perich et al."** |
| N2 | **Vastola, J. J., Cohen, Z. & Drugowitsch, J. (2023).** "Is the information geometry of probabilistic population codes learnable?" PMLR 197:258–277. | *"we derive a mathematical result that the information geometry of the statistical manifold is directly related to measurable covariance matrices. This suggests a simple but rigorously justified decoding strategy based on principal component analysis, which we illustrate using an analytically tractable PPC."* ⛔ **THREE authors — Zach Cohen sits in the middle.** |
| N3 | **Wang, B. & Ponce, C. R. (2023).** "On the level sets and invariance of neural tuning landscapes." PMLR 197:278–300. | *"we characterize tuning landscapes through the lens of level sets and Morse theory… we developed a statistically reliable signature for these maps based on the change of topology in level sets. We found this topological signature changed progressively throughout the cortical hierarchy, with similar trends found for units in convolutional neural networks."* Hypothesis: *"higher-order units can be locally regarded as isotropic radial basis functions, but not globally."* ⭐ A level set of a tuning landscape is an iso-response set — their version of a flat direction, but in **stimulus** space. Do not conflate. |
| N4 | **Baroni, L., Bashiri, M., Willeke, K. F., Antolík, J. & Sinz, F. H. (2023).** "Learning invariance manifolds of visual sensory neurons." PMLR 197:301–326. | *"Our approach is fully data-driven, allowing the discovery of novel neural invariances, and enables scientists to generate and experiment with novel stimuli along the invariance manifold."* ⛔ **FIVE authors, first author Baroni; Sinz senior — never "Sinz et al."** |
| N5 | **Iyer, R., Siegle, J., Mahalingam, G., Olsen, S. & Mihalas, S. (2023).** "Geometry of inter-areal interactions in mouse visual cortex." PMLR 197:327–353. | Allen Brain Observatory Neuropixels: *"distinct subspaces of a source area mediate interactions with distinct target areas, supporting the notion that cortical areas use distinct channels to communicate"*, and *"these interactions evolve dynamically over tens of milliseconds… Inter-areal subspaces become more aligned with the intra-areal subspaces during epochs in which a feedforward wave of activity propagates."* |
| N6 | **Klindt, D., Gaukstad, S., Vaupel, M., Hermansen, E. & Dunn, B. (2023).** "Topological ensemble detection with differentiable yoking." PMLR 197:354–369. | *"Recent work demonstrated that recordings from individual ensembles exhibit the topological signature of a torus. This is obscured, however, in combined recordings from multiple ensembles."* Unsupervised ensemble identification *"by optimizing a loss function that captures the assumed topological signature."* ⭐ This audience treats "a torus is present in the state space" as an **empirical, testable** signature — the shape of V2's claim about a trained potential. |
| N7 | **Duan, S., Khona, M., Bertagnoli, A., Chandra, S. & Fiete, I. R. (2023).** "See and Copy…" PMLR 197:388–400. | *"A hallmark of biological intelligence and control is combinatorial generalization"*; a modular encoder-RNN + motor-RNN + scheduler generalises to unseen trajectories and to **more segments than seen in training**, and adapts rapidly to perturbations. |
| N8 | (The Head's "~8 neuroscience" = volume items 15–22 = N1–N7 **+ G9**.) | — |

### 1.3 The third cluster — "topology and learning theory" (⚠ my inference; the Head gave a topic, not papers)

| # | record (verified) | claim |
|---|---|---|
| T1 | **McGuire, S., Jackson, S., Emerson, T. & Kvinge, H. (2023).** "Do neural networks trained with topological features learn different internal representations?" PMLR 197:122–136. | *"we find that structurally, the hidden representations of models trained and evaluated on topological features differ substantially compared to those trained and evaluated on the corresponding raw data. On the other hand… these representations can be reconciled (at least to the degree required to solve the corresponding task) using a simple affine transformation. We conjecture that this means that neural networks trained on raw data may extract some limited topological features."* |
| T2 | **Davies, T., Aspinall, J., Wilder, B. & Tran-Thanh, L. (2023).** "Fuzzy c-means clustering in persistence diagram space…" PMLR 197:137–157. | FCM extended to persistence-diagram space with convergence guarantees matching the Euclidean case; *"fuzzy clustering persistence diagrams allows for unsupervised model selection using just the topology of their decision boundaries."* ⚠ PMLR's index inverts the 4th author as "Long, Tran-Thanh"; the surname is **Tran-Thanh**. |
| T3 | **Dönmez, A. (2023).** "On the ambiguity in classification." PMLR 197:158–170. | *"We develop a theoretical framework for geometric deep learning that incorporates ambiguous data in learning tasks. This framework uncovers deep connections between noncommutative geometry and learning tasks."* Learning tasks arise from **groupoids**. Single author. |
| T4 | **Akhtiamov, D. & Thomson, M. (2023).** "Connectedness of loss landscapes via the lens of Morse theory." PMLR 197:171–181. | *"Mode connectivity is a recently discovered property of neural networks stating that two weight configurations of small loss can usually be connected by a path of small loss."* ⭐ **Morse theory on an energy landscape is this audience's register for what V2 does to V_θ. Two volume papers use it (T4, N3). Adopting Morse-theoretic language in V2 §2 is audience-native, costs no claim, needs no new result.** |
| T5 | **Tian, Y., Lubberts, Z. & Weber, M. (2023).** "Mixed-membership community detection via line graph curvature." PMLR 197:219–233. | *"a discrete Ricci curvature flow under which the edge weights of a graph evolve to reveal its community structure."* Low relevance to V2. |

### 1.3b ⭐ The audience's CURRENT agenda (primary, `https://neurreps.org/`, retrieved 2026-08-20)

The site now shows **NeurReps 2026**. CFP topics, **verbatim**:
> Theory and methods for learning invariant and equivariant representations · Statistical learning theory in topology, geometry, and symmetry contexts · Representational geometry in neural data · Learning and leveraging group structure in data · Equivariant world models for robotics · **Dynamics of neural representations** · Topological deep learning and topological data analysis · Geometric structure in language · Geometric and topological analysis of generative models · **Symmetries, dynamical systems, and learning**

Tracks, **verbatim**:
> **Proceedings Track:** *"Self-contained, highly-developed research papers. Archivally published in a dedicated PMLR volume."* — **9 pages, excl. refs + appendices**
> **Extended Abstract Track:** *"Early-stage results, negative findings, opinion pieces, or novel datasets. Non-archival — may be posted to arXiv."* — **4 pages, excl. refs + appendices**

⇒ Add.4 Ruling 1 **re-confirmed on 2026 data**. ⭐ The two bolded bullets are **V2's exact subject line** and should be what the abstract is built to hit. ⚠ See reconciliation item 2 on the EA track's stated purpose.

**Volume inventory (verified 2026-08-21 against dblp's NeurReps index):** only **two** PMLR volumes exist — **v197** (1st, 2022) and **v228** (2nd, held 2023-12-16, published 2024-08-02, eds. Sanborn, Shewmake, Azeglio, Miolane, 23 papers). No 3rd/4th volume is indexed. NeurReps 2025 ran (a CFP is on record; one index states 121 submissions) but **no accepted-paper list was retrievable** — `neurreps.org/accepted-submissions` now 404s. Two v228 items matter to us: **Vastola, J. (2024), "Optimal packing of attractor states in neural representations," PMLR 228:425–442** — *"symmetries in environmental transition statistics imply certain symmetries of the optimal neural representations"*, framing memory-state layout as **sphere packing** (⭐ the audience's own capacity question) — and **Dönmez, A. (2024), "Discovering latent causes and memory modification: A computational approach using symmetry and geometry."**

---

# Part 2 — the bridge literature (the highest-value half)

## 2.1 Continuous attractors and drift — the neuroscience-native name for our flat direction

**B1 — Ságodi, Á., Martín-Sánchez, G., Sokół, P. & Park, I. M. (2024). "Back to the Continuous Attractor." NeurIPS 2024. arXiv:2408.00109.** ⭐⭐ *The single most useful paper in this brief.* Read from the primary PDF (v3, 2025-01-17).
- Abstract, **verbatim**: *"Continuous attractors offer a unique class of solutions for storing continuous-valued variables in recurrent system states for indefinitely long time intervals. Unfortunately, continuous attractors suffer from severe structural instability in general—they are destroyed by most infinitesimal changes of the dynamical law that defines them. … Fast-slow decomposition analysis uncovers the existence of a persistent slow manifold that survives the seemingly destructive bifurcation, relating the flow within the manifold to the size of the perturbation. Moreover, this allows the bounding of the memory error of these approximations… we conclude that continuous attractors are functionally robust and remain useful as a universal analogy for understanding analog memory."*
- The definition to borrow, **verbatim** (§2): *"Let M ⊂ ℝᵈ be a manifold. We say M is a continuous attractor, if (1) every state on the manifold is a fixed point, ∀x ∈ M, f(x) = 0, and (2) the fixed points are **marginally stable tangent to the manifold and stable normal to the manifold**."* ⇒ our μ²=0-along-the-orbit / μ²>0-transverse statement, in their dialect.
- Their line attractor, **verbatim**: *"Linearization of the fixed points on the manifold exhibits two eigenvalues, 0 and −2; the 0 eigenvalue allows the continuum of fixed points, while −2 makes the flow normal to the manifold attractive."*
- The audience's name for our erosion problem, **verbatim** (§1): *"In neuroscience, this vulnerability is well-known and often referred to as the **'fine-tuning problem'**. There are two primary sources of perturbations in the recurrent network dynamics: (1) the stochastic nature of online learning signals that act via synaptic plasticity, and (2) spontaneous fluctuations in synaptic weights."*
- Their memory guarantee, **verbatim** (Eq. 5, §3.3): *"(1/vol M) ∫_M |x(t,x₀) − x₀| dx₀ ≤ t‖φ‖_∞  (error bound)"*, with *"this bound is the worst case and tighter for sufficiently small t ≥ 0."* Plus **Theorem 1 (Persistent Manifold)** and **Proposition 1 (Revival of continuous attractor)**: *"Let the uniform norm of the flow tangent to the manifold be ‖ẏ‖_∞ = η. There exists a perturbation with uniform norm at most η that induces a bifurcation to a continuous attractor manifold."*
- **What our result is NOT.** *We make no claim about biological systems and do not model neural data.* Our object is a 4-dimensional latent state of a trained artificial unit on a synthetic S¹ task; theirs is a class of models of brain circuits. Technical contrast for the writer: **they bound drift by the size of an unknown perturbation (worst-case, linear in t); we do not perturb — we read the transverse curvature spectrum μ² off the trained potential and get the half-life in closed form (n₁/₂ ∝ μ⁻², slope −0.985 measured, predicted −1), with a regime map (floor, exceptional point) a first-order flow cannot exhibit.** Complementary, not competing: theirs is *robustness of the concept*, ours is *the price list*.

**B2 — Burak, Y. & Fiete, I. R. (2012). "Fundamental limits on persistent activity in networks of noisy neurons." PNAS 109(43):17645–17650.** The quantitative diffusion result. Abstract, **verbatim**: *"Here we analytically derive how the stored memory in continuous attractor networks of probabilistically spiking neurons will degrade over time through diffusion. … The noise-induced drift of the memory state over time within the network is strictly lower-bounded by the accuracy of estimation of the network's instantaneous memory state by an ideal external observer. This result takes the form of an information-diffusion inequality."* Two objects to quote: **⟨[θ(t+Δt) − θ(t)]² ⟩ = 2D Δt** and the **information–diffusion inequality D ≥ τ²J⁻¹** (their Eq. 4), J = internal Fisher information.
- ⛔ **NEVER-COPY TRAP (record now primary-verified; content still not):** a correction exists — *"Correction for Burak and Fiete, Fundamental limits on persistent activity in networks of noisy neurons"*, **PNAS 114(20):E4117, published 2017-05-08, doi 10.1073/pnas.1706051114, PMID 28483997, PMCID PMC5441759** (type: **erratum**; verified via the OpenAlex and Europe PMC records, 2026-08-21). **No abstract or body text is exposed by any of the five routes tried** (PNAS 403, PubMed cookie-wall, ADS 405, Europe PMC null abstract, aggregator stub). Search-index text states **Eq. 2 appeared incorrectly owing to a printer's error**, and that remains **single-sourced and unverified**. ⇒ **Do not reproduce their Eq. 2 (the D integral) in any draft.** Quoting ⟨Δθ²⟩ = 2DΔt and Eq. 4 is low-risk.
- **What our result is NOT.** Their D is derived for *probabilistically spiking neurons* and lower-bounded by an information-theoretic inequality about a biological network; **our** D_θ = εT(2−γ)/(2F²γ) is a fluctuation–dissipation statement about a damped symplectic map at temperature T, holding **only** under FDT-consistent noise in a Newtonian kinetic mode (V2 §2 fine print (a)), never under the reference implementation's legacy noise default. *We claim no correspondence between the two constants and fit no neural data.* Honest contrast: **both say a flat direction diffuses; theirs bounds D from the code's Fisher information, ours computes D from the map's own (ε, γ, T, F²) and predicts the half-life.**

**B3 — Khona, M. & Fiete, I. R. (2022). "Attractor and integrator networks in the brain." *Nature Reviews Neuroscience* 23:744–766.** doi:10.1038/s41583-022-00642-0; preprint arXiv:2112.03978. The canonical review — the audience's shared reference for what a continuous attractor is and where it has been identified. *(Record via publisher landing page + Scholar lookup; abstract not transcribed — F3.)*

**B4 — Seung, H. S. (1996). "How the brain keeps the eyes still." PNAS 93(23):13339–13344.** doi:10.1073/pnas.93.23.13339. Origin of the **line-attractor** hypothesis and of the fine-tuning framing: persistent activity in the oculomotor neural integrator forms *an attractive line of fixed points*, produced by *precisely tuned positive feedback*. ✅ **F4 RESOLVED (2026-08-21): sole author, H. Sebastian Seung (Bell Laboratories, Lucent Technologies), verified against the OpenAlex authorship record.** The "Seung-Hopfield" string was a Semantic Scholar URL-slug artifact, not a second author.

**B5 — Kim, S. S., Rouault, H., Druckmann, S. & Jayaraman, V. (2017). "Ring attractor dynamics in the *Drosophila* central brain." *Science* 356(6340):849–853.** doi:10.1126/science.aal4835 (ADS bibcode 2017Sci...356..849K confirms p. 849). Two-photon imaging + optogenetics *overwrote the existing population representation with an artificial one, which was then maintained by the circuit with naturalistic dynamics*, with local excitation + global inhibition enforcing a unique persistent heading representation. ⭐ **"Overwrite the representation, then watch it persist" is literally our write-then-hold protocol, run in a fly** — the best one-sentence analogy for the latch, and the sentence immediately after it must be the no-claim sentence.
- **What our result is NOT.** *We make no claim about biological systems and do not model neural data.* Our write is a momentum impulse into a designed SO(2)-invariant learned potential; theirs is optogenetic stimulation of a real circuit. The analogy is to the *geometric object* (a ring of marginally stable states retaining an externally imposed value), never to the mechanism or the organism.

**B6 — Gardner, R. J., Hermansen, E., Pachitariu, M., Burak, Y., Baas, N. A., Dunn, B. A., Moser, M.-B. & Moser, E. I. (2022). "Toroidal topology of population activity in grid cells." *Nature* 602(7895):123–128.** doi:10.1038/s41586-021-04268-7. Grid-cell population activity *resides on a toroidal manifold that is invariant across environments and brain states*. The empirical anchor under census items N6 and G9, and the audience's proof that **"the state space carries a torus" is a measurable claim** — the shape of V2's Appendix-J/GMOR argument. ⚠ Author list from index records, not the Nature masthead (F3).

## 2.1b ⭐⭐ NEW IN REV. 2 — the fine-tuning problem's KNOWN CURES (the prior art V2's anchor result must be scoped against)

This subsection did not exist in rev. 1 and it is the reason reconciliation item 4 exists. **The pattern "the continuous attractor is destroyed by imperfection → a compensating mechanism restores it" is not new; it is a classic of this literature.**

**B7 — Renart, A., Song, P. & Wang, X.-J. (2003). "Robust spatial working memory through homeostatic synaptic scaling in heterogeneous cortical networks." *Neuron* 38(3):473–485.** doi:10.1016/S0896-6273(03)00255-1. *(Record — title, three authors in order, volume/issue/pages — primary-verified via OpenAlex 2026-08-21; the abstract itself is **index-sourced**, Cell returned 403. F3.)* Claim per the indexed abstract text: heterogeneity in cellular and synaptic properties **destroys the fine tuning**, so that *the network does not support spatial working memory function* and stored spatial information is *lost in a few seconds*; **accurate encoding is recovered when a homeostatic mechanism scales the excitatory synapses to each cell to compensate for the heterogeneity**; the fine-tuning problem is described as *a general feature of systems encoding internal representations of analog features*.
- ⛔ **This is the direct structural analogue of V2 §3.4's energy anchor** — a corrective term that restores a broken flat direction. **V2 must not present "a mechanism that keeps the flat direction alive" as a new idea in this room.** What is different, and defensible: *their* imperfection is static parameter heterogeneity in a hand-built network and *their* cure is synaptic scaling; *ours* is a dynamic imperfection created by the training objective itself in a **learned** potential, and *our* cure is a V(data) energy anchor — and, uniquely, we then show **every headline retention law still holds under the cure**, at ~20× the erosion horizon (GMOR exact to 1.5×10⁻¹², slope −0.956, same floor, EP onset bit-identical at 0.5165). *The laws-survive-the-cure measurement is the claimable part; the destroy-and-restore narrative is not.*

**B8 — Vafidis, P., Owald, D., D'Albis, T. & Kempter, R. (2022). "Learning accurate path integration in ring attractor models of the head direction system." *eLife* 11:e69841.** doi:10.7554/eLife.69841; PMID 35723252. Claim: *"To function as integrators, head direction circuits require precisely tuned connectivity, but it is currently unknown how such tuning could be achieved"*; they propose *a local, biologically plausible learning rule* that *adjusts synaptic efficacies during development, guided by supervisory allothetic cues*, and, applied to the *Drosophila* head-direction system, *"learns to path-integrate accurately and develops a connectivity strikingly similar to the one reported in experiments."* *(Record via eLife landing page + PubMed; abstract text index-sourced — F3.)*
- ⛔ **Consequence for V2: "learning can produce/maintain the tuning of a continuous attractor" is ESTABLISHED.** Our designed-vs-emergent gap (13–14 orders in μ²) must therefore be stated as what it is — *a measurement on our architecture class and training recipe* — and **never** as "learning cannot produce a flat direction." N46 already scopes this correctly (our emergent arm has no coset register); the reframe must not let the audience hear the stronger, false general claim.

**B9 — Dinc, F., Cirakman, E., Kurtkaya, B., Yuksekgonul, M., Jiang, Y., Schnitzer, M. J. & Tanaka, H. (2025/2026). "A ghost mechanism: An analytical model of abrupt learning in recurrent networks." arXiv:2501.02378; comment: *to appear in Physical Review X*.** Abstract, **verbatim in part**: *"Abrupt learning is a common phenomenon in recurrent neural networks (RNNs) trained on working memory tasks. In such cases, the networks develop transient slow regions in state space that extend the effective timescales of computation. … we introduce the ghost mechanism, a process by which dynamical systems exhibit transient slowdown near the remnant of a saddle-node bifurcation. … we identify a critical learning rate that scales as an inverse power law with the timescale of the learned computation. Beyond this rate, learning collapses through two interacting modes: (i) vanishing gradients and (ii) oscillatory gradients near minima. … well-known learning difficulties in RNNs partly arise from the dynamical systems they must learn to implement."*
- ⚠ **Direction check: this paper is about slow structure *appearing* during training, not being destroyed.** It is nonetheless the closest live neighbour to V2's §3.4 training-dynamics story, it is physics-venue (PRX) and therefore highly legible to this audience, and its *"critical learning rate scaling as an inverse power law with the timescale"* is the same *genre* of result as our erosion horizon set by sleep-update frequency racing the wake clamp. **Cite as a neighbour; claim nothing from it.** ⚠ Venue is a **preprint comment ("to appear")** — treat as preprint until the PRX record exists (F6).

## 2.2 Representational drift — as its own phenomenon (⛔ NOT the same thing as 2.1)

⚠ **The trap this section exists to prevent.** "Drift along a continuous attractor" (§2.1: within-session, fast, state-space random walk along a flat direction) and "**representational drift**" (§2.2: cross-session, days-to-weeks, tuning-space reconfiguration of which neurons encode what) are **two different phenomena sharing one English word.** The Head's census item (Jude/Perich/Miller/Hennig, N1) sits in §2.2, not §2.1. **A draft that lets a reader slide from one to the other has mis-described the neighbouring literature — exactly the failure Add.37 FLAG 2 forbids.**

- **D1 — Ziv, Y., Burns, L. D., Cocker, E. D., et al. (2013). "Long-term dynamics of CA1 hippocampal place codes." *Nature Neuroscience* 16:264–266.** doi:10.1038/nn.3329. Over weeks, *each day the ensemble representation of a familiar environment involved a unique subset of cells*, yet the ~15–25 % of cells overlapping between any two days *retained the same place fields, which sufficed to preserve an accurate spatial representation across weeks.*
- **D2 — Driscoll, L. N., Pettit, N. L., Minderer, M., Chettih, S. N. & Harvey, C. D. (2017). "Dynamic reorganization of neuronal activity patterns in parietal cortex." *Cell* 170(5):986–999.e16.** With behaviour stable for a month, the activity–task-feature relationship *was mostly stable on single days but underwent major reorganization over weeks.*
- **D3 — Rule, M. E., O'Leary, T. & Harvey, C. D. (2019). "Causes and consequences of representational drift." *Current Opinion in Neurobiology* 58:141–147.** PMID 31569062. Argues *the recurrent and distributed nature of sensorimotor representations permits drift while limiting disruptive effects*, and that drift may *create error signals between interconnected brain regions that can be used to keep neural codes consistent.*
- **D4 — Rule, M. E. & O'Leary, T. (2022). "Self-healing codes: How stable neural populations can track continually reconfiguring neural representations." *PNAS* 119(7):e2106692119.** doi:10.1073/pnas.2106692119. The mechanism-side companion: a downstream population can track an upstream code that continually reconfigures. *(ADS bibcode 2022PNAS..11906692R + PNAS landing metadata; full text 403 — F3.)* ⛔ **TWO authors — Harvey is not on this one, unlike D3.**
- **D5 — Deitch, D., Rubin, A. & Ziv, Y. (2021). "Representational drift in the mouse visual cortex." *Current Biology* 31(19):4327–4339.e6.** Drift *over timescales spanning minutes to days* across six visual areas, layers and cell types; *representational drift is an inherent property of neural networks*, with population-level organisation yielding time-invariant representations despite drifting single cells.
- **D6 — Aitken, K., Garrett, M., Olsen, S. & Mihalas, S. (2022). "The geometry of representational drift in natural and artificial neural networks." *PLOS Computational Biology* 18(11):e1010716.** doi:10.1371/journal.pcbi.1010716. ⭐ **The bridge inside the bridge:** same senior authors as census item N5 (Olsen, Mihalas, Allen Institute). Verbatim: *"the drift differs from in-session variance and most often occurs along directions that have the most in-class variance, leading to a significant turnover in the neurons used for a given representation… despite this significant change due to drift, linear classifiers trained to distinguish neuronal representations show little to no degradation in performance across days. The features we observe in the neural data are similar to properties of artificial neural networks where representations are updated by continual learning in the presence of dropout."*
- **What our result is NOT — the honest contrast sentence for all of §2.2.** *We make no claim about biological systems and do not model neural data.* Representational drift is a measured biological phenomenon over days and weeks in populations of real neurons; **nothing in V2 measures, models, explains, or predicts it.** The only motion V2 reports is (i) the deterministic **latch** freezing a written coset coordinate at T = 0, (ii) its **diffusion** at T > 0 under FDT-consistent noise, and (iii) **erosion** — a *training objective* driving the order parameter r* → 0 on a synthetic S¹ task. If a draft uses the word "drift" it must be scoped in the same sentence: *"drift of the coset coordinate under our own dynamics"* — never bare "drift."

## 2.3 Where the reframe should place V2 — CORRECTED IN REV. 2

**What this audience already agrees on:** (a) a manifold of marginally stable states stores an analog variable; (b) it is fragile — noise diffuses along it (B2) and perturbations of the dynamical law destroy it (B1); (c) that fragility is the **fine-tuning problem**, known since B4/B7 and named by B1 as arising partly from learning; (d) **compensating mechanisms that restore the tuning exist and are classic (B7)**; (e) **learning rules that produce the tuning exist (B8)**; (f) approximate continuous attractors arise naturally in task-trained RNNs (B1 §4, B9).

⛔ **Therefore the following are NOT available to V2 as novelty in this room:** "flat directions store analog values" · "the flat direction is fragile" · "learning perturbs it" · "a corrective mechanism can keep it alive" · "training can produce the structure."

✅ **What this scout could not find anywhere, and what V2 should therefore lead with:** **a closed-form, measured price list connecting the transverse-curvature spectrum of a *trained, learned* potential to a retention half-life — n₁/₂ ∝ μ⁻² at fitted slope −0.985 over 4.5 decades, with a mass-independent floor and an exceptional-point crossover at εμ ≈ γ/2 with √(h−h*) onset — together with the demonstration that these laws are *invariant under the cure* (all headline laws still hold under the anchor at ~20× the erosion horizon).** The audience has qualitative fragility and qualitative robustness; nobody has priced the trade. **Suggested sentence for the Advisor to rule on (rev. 2, narrowed):** *"This literature knows that a flat direction is fragile and that corrective mechanisms can keep it alive; what has not been available is the exchange rate — how much retention a given transverse curvature buys, where that law stops holding, and whether it survives the correction. We measure all three on trained models."* ⚠ Every clause traces to CM-4/CM-5/CM-6 and §3.1/§3.4 as already approved; it widens nothing.

---

# Part 3 — the vocabulary map (our term → the audience's standard term)

Legend: **[E]** exact · **[A]** approximate, with the difference stated · **[⛔N]** no standard equivalent — keep our term and define it.

| our internal term | audience's standard term | fit | note the writer must carry |
|---|---|---|---|
| **flat / neutral direction** | *tangent direction to the continuous attractor manifold*; *zero mode*; *marginally stable direction*; *zero eigenvalue / zero Lyapunov exponent* | **[E]** | Ságodi's definition ("marginally stable tangent… stable normal") is our statement verbatim. *Goldstone/zero mode* is also legitimate here but physics-native; this audience meets it mainly through Iqbal et al. (2026). Define once. |
| **coset register** | *the position along the continuous attractor*; *the encoded continuous variable*; *the bump phase / heading angle θ*; (geometry side) *orbit coordinate / group parameter / fundamental-domain coordinate* | **[A]** | ⛔ **"Coset register" is ours; there is no standard equivalent as a *storage device*.** Aslan et al. (G6) is the nearest published *coordinate*, but it is an input encoding, not a memory. Suggested first use: *"the coset coordinate — the position along the flat direction, which is what the unit actually stores."* |
| **mode-mass spectrum** | *spectrum of curvatures normal (transverse) to the manifold*; *transverse eigenvalue spectrum*; loosely *relaxation-rate spectrum* | **[A]** | Their eigenvalues are of the **flow Jacobian** (units 1/time, e.g. "0 and −2"); ours are of the **mass-whitened Hessian of V_θ** (units 1/time², μ² = λ(M^{−1/2}∇²V_θ M^{−1/2})). ⚠ **Print the conversion once or readers will compare incommensurable numbers.** |
| **spectral mass μ²** | *transverse curvature*; *normal-direction eigenvalue*; *the (pseudo-)gap* | **[A]** | ⛔ **"Mass" is a trap twice over:** in neuroscience it means nothing and reads as jargon; and our own paper already has the *inertial* mass M running the other way (V2 §1 Nomenclature). Lead with *"transverse curvature μ²"*, keep "spectral mass" as the parenthetical. |
| **latch** | *persistent activity*; *analog working memory*; *the neural integrator*; *a fixed point on the manifold* | **[A]** | ⚠ **Real technical difference — do not blur.** The classical neuroscience latch is a **perfect integrator** (no damping; a marginal drifting integrator). Ours **requires γ > 0**: at γ = 0 the flat mode never freezes a write; any γ > 0 gives an exact latch, q_∞ = q₀ + εp₀/(Mγ) (V2 App. A note (iii)). **We store by damping; they store by not damping.** A genuine differentiator — say it, don't smooth it. |
| **erosion** | closest: *the fine-tuning problem*; *learning-induced degradation of the attractor*; *symmetry restoration* (physics) | **[A]** | The class is right (B1's perturbation source (1) is online learning signals), but ⛔ **the fine-tuning problem is broader and older (B4, B7), and its cures are classic (B7, B8).** Scope as: *"an instance of what this literature calls the fine-tuning problem, here with the perturbation being the training objective itself"* — and **never** "we solve the fine-tuning problem", and **never** imply the destroy-and-restore pattern is new. |
| **the settle** | *relaxation to the attractor*; *decay of the fast/normal modes*; *the fast flow in a fast–slow decomposition*; *transient* | **[A→E]** | Ságodi §3.2's fast–slow decomposition is the exact home: our settle is their **fast normal flow**, our retention their **slow tangent flow**. A gift — it makes the two-timescale story audience-native at zero claim cost. |
| **retention half-life (n₁/₂)** | *memory lifetime*; *time constant τ*; quantitatively *diffusion coefficient D* (⟨Δθ²⟩ = 2DΔt) or *memory-error bound* | **[A]** | ⛔ **This field does not standardly use "half-life."** They report D, τ, or a decoding-accuracy curve. Keep n₁/₂ (it is our defined instrument) but **bridge to D and τ on first use**, and keep V2's rule that *every reported lifetime names its metric* (the envelope/first-crossing bifurcation, §3.1) — that reads as unusual rigour here. |
| **designed vs emergent symmetry** | *exact (architectural / hard-wired) equivariance* vs *learned, approximate, or relaxed equivariance* | **[E]** | Their phrase is in the census: van der Ouderaa & van der Wilk (G3) build filters *"to maintain or relax equivariance"*; neuro side, *hand-tuned connectivity vs task-trained RNN*. ⚠ **Rev.-2 rider:** B8 shows learning *can* produce accurate tuning, so our 13–14 orders of magnitude is a measurement on **our** architecture class and recipe (N46's scope), never a general impossibility claim. |
| **the budget cube ((μ, γ, T))** | — | **[⛔N]** | No equivalent; do not translate. Introduce as a defined object or drop it (it is an Appendix-F.2 supporting result anyway). |
| **the store / atom / register (as CLU nouns)** | — | **[⛔N]** | Program-internal. Here say *"the latent state"* / *"the learned potential V_θ"* / *"a stored value."* |
| **exceptional point** | *(non-Hermitian-physics term; not NeurReps-native)* | **[⛔N]** | Standard in physics, **not** in this audience. Define on first use as *"the overdamped→underdamped crossover, where the 2×2 block becomes defective"*, and let the √(h−h*) onset carry the evidence. |
| **conformally symplectic** | *structure-preserving / symplectic integrator* is known; the conformal qualifier is not | **[A]** | One clause: *J⊤ΩJ = (1−γ)Ω — volume contracts by a fixed factor per step, so the geometry is preserved and only the scale is spent.* ⚠ Collision hazard: **Xu et al. (G9) use "conformal isometry" for something entirely different.** Two "conformal"s in one room — disambiguate or avoid. |
| **Nambu–Goldstone / pseudo-Goldstone cell** | *zero mode of a continuous attractor* / *nearly-zero (slow) mode* | **[A]** | Keep our taxonomy (it is axis 1) but gloss each cell once: unbroken = *no manifold, single fixed point*; NG = *exact continuous attractor*; pseudo-NG = *approximate continuous attractor / slow manifold* — **exactly Ságodi et al.'s object**. Worth one sentence. |

---

# Confidence & gaps

**Verified (primary, authoritative, retrieved 2026-08-20/21):** the entire v197 census — titles, full author lists, page ranges from the PMLR volume index; abstracts from each paper's own PMLR page. NeurReps 2026 CFP topics + track rules from neurreps.org. Ságodi et al. read from the arXiv PDF itself (abstract, definition, Theorem 1, Prop. 1, Eq. 5 transcribed from the page, not summarised). Burak & Fiete abstract + Eq. 4 + the 2DΔt law from PMC full text; the **correction's bibliographic record** from OpenAlex + Europe PMC. **Seung 1996 sole authorship** from OpenAlex. **Renart/Song/Wang 2003 record** from OpenAlex. Aitken et al. abstract from the PLOS article page. Dinc et al. abstract from the arXiv abstract page. Volume inventory from dblp.

**Single-sourced / partially verified — flagged, not checked:** Khona & Fiete (record only) · Gardner et al. (author list from indexes) · Rule & O'Leary 2022 (ADS + landing metadata) · Ziv 2013 / Driscoll 2017 / Deitch 2021 / Rule 2019 (records + claims from landing pages and indexes) · **Renart 2003 and Vafidis 2022 abstract text (index-sourced; both publishers blocked)** — ⚠ these two are now load-bearing for reconciliation item 4, so **their abstracts should be read before the reframe cites them** · PMLR v228 beyond the Vastola entry.

**Still could not verify (declared, not guessed):**
- **The TEXT of the 2017 PNAS correction to Burak & Fiete.** Five routes tried (PNAS 403, PubMed cookie-wall, ADS 405, Europe PMC null abstract, aggregator stub). The record is confirmed; the content ("Eq. 2, printer's error") is not. Gate on this before quoting their Eq. 2. **Likely resolvable only via institutional access or the PMC5441759 PDF.**
- **The NeurReps 2025 accepted-paper list.** `neurreps.org/accepted-submissions` 404s; no PMLR volume; one index reports 121 submissions. The likely home is the workshop's OpenReview venue — **that is the single highest-value remaining search** if the Advisor wants the *current* reviewer pool's own papers.

**Resolved since rev. 1:** F4 (Seung sole author ✅) · F5 (only two PMLR volumes exist ✅) · the correction's bibliographic record ✅ · reconciliation item 4 (my own over-claim, caught and retracted ✅).

**What to search next, priority order:** (1) NeurReps 2025 on OpenReview — the current audience; (2) the Burak & Fiete correction text; (3) primary abstracts for Renart 2003 and Vafidis 2022 (now load-bearing); (4) the NeurReps 2026 organiser/reviewer list, to check for authors of the census works (Fiete, Miolane, Sanborn, Walters, Klindt, Dunn) whose own papers V2 would then be citing.

---

# Flags

- **F1 — v197 citation-year ambiguity.** PMLR metadata says **2023** with keys `*23a`; the front matter says "PMLR 197, **2022**"; the workshop was **2022-12-03**. Choose one convention and state it once. ⛔ Never "NeurReps 2023" for a 2022-presented paper.
- **F2 — page-range anomalies in the v197 index (PMLR's own, not mine):** Akhtiamov ends at **181** and Aslan begins at **181**; van der Ouderaa ends at **62** and Klee begins at **64**. Reproduce as given; do not "fix" them.
- **F3 — records assembled from indexes rather than publisher full text** (PNAS, Nature, Cell, eLife returned 403 or were cookie-walled): Khona & Fiete 2022 · Gardner et al. 2022 · Rule & O'Leary 2022 · **Renart et al. 2003** · **Vafidis et al. 2022** · the Burak & Fiete correction. Standard, heavily-cited records, but **not** primary-verified here.
- **F4 — RESOLVED.** Seung 1996 is **sole-authored** (OpenAlex). The "Seung-Hopfield" string was a Semantic Scholar slug artifact. Rev. 1's warning is withdrawn.
- **F5 — RESOLVED, and it sharpens reconciliation item 1.** Only **two** NeurReps PMLR volumes exist (v197, v228; dblp). The census is the older of the two; **the audience's last four years are not in any archival volume**, which raises the value of the OpenReview search above.
- **F6 — Dinc et al. venue is a preprint comment.** arXiv:2501.02378 states *"to appear in Physical Review X"*. ⛔ Cite as preprint with the comment noted until a PRX record exists; do not print a PRX volume/page.
- **F7 — Add.37 FLAG 2 pressure points, named in advance.** Four audience terms would *widen* our claims if adopted unqualified: **"continuous attractor"** applied to our unit (ours is a designed flat direction of a trained potential in dim 4, not a demonstrated attractor manifold — N46 says the emergent arm has none); **"drift"** unqualified (§2.2 trap); **"solves the fine-tuning problem"** (we exhibit one instance and one cure, on a designed geometry, with N149/N150 fencing the learned-store case); and ⭐ **new in rev. 2** — any phrasing implying that *learning cannot* build a flat direction (B8 refutes it in general) or that a *corrective mechanism* is our idea (B7 predates it by 23 years).
- **F8 — nomenclature collision inside the reframe:** "conformal" means two different things in this room (our conformally symplectic map vs Xu et al.'s conformal isometry).

---

# BibTeX (house pattern; traps in `note`)

```bibtex
@inproceedings{shutty2023lie,
  title={Computing Representations for {L}ie Algebraic Networks},
  author={Shutty, Noah and Wierzynski, Casimir},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={1--21}, year={2023}, publisher={PMLR},
  note={NeurReps 2022 (workshop 2022-12-03; PMLR published 2023-02-07). Claims ``the first object-tracking model equivariant to the Poincare group'' -- the nearest census work to CHLU's relativistic framing. Retrieved 2026-08-20.}}

@inproceedings{chau2023disentangling,
  title={Disentangling Images with {L}ie Group Transformations and Sparse Coding},
  author={Chau, Ho Yin and Qiu, Frank and Chen, Yubei and Olshausen, Bruno},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={22--47}, year={2023}, publisher={PMLR},
  note={FOUR authors; Olshausen is senior -- never ``Olshausen et al.''. Transformations constrained to an n-dimensional TORUS representation. Retrieved 2026-08-20.}}

@inproceedings{vanderouderaa2023sparse,
  title={Sparse Convolutions on {L}ie Groups},
  author={van der Ouderaa, Tycho F. A. and van der Wilk, Mark},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={48--62}, year={2023}, publisher={PMLR},
  note={PMLR slug is `ouderaa23a', NOT `vanderouderaa23a'. Source of the audience-native phrase ``maintain or relax equivariance''. Retrieved 2026-08-20.}}

@inproceedings{klee2023icosahedral,
  title={Image to Icosahedral Projection for {SO}(3) Object Reasoning from Single-View Images},
  author={Klee, David and Biza, Ondrej and Platt, Robert and Walters, Robin},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={64--80}, year={2023}, publisher={PMLR},
  note={Page range starts at 64: PMLR's index skips p.63. Equivariance is APPROXIMATE (icosahedral subgroup of SO(3)), not exact. Retrieved 2026-08-20.}}

@inproceedings{sangalli2023movingframe,
  title={Moving Frame Net: {SE}(3)-Equivariant Network for Volumes},
  author={Sangalli, Mateus and Blusseau, Samy and Velasco-Forero, Santiago and Angulo, Jes{\'u}s},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={81--97}, year={2023}, publisher={PMLR},
  note={Key move: the moving frame is computed ONCE at the input stage, not per layer. Abstract read as paraphrase from the PMLR page, not transcribed verbatim. Retrieved 2026-08-20.}}

@inproceedings{robin2023periodic,
  title={Periodic Signal Recovery with Regularized Sine Neural Networks},
  author={Robin, David A. R. and Scaman, Kevin and Lelarge, Marc},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={98--110}, year={2023}, publisher={PMLR},
  note={``multi-layer perceptrons with ReLU activations are provably unable to perform this task''; sine nets extrapolate ``beyond 100 periods''. The census's own long-horizon-extrapolation paper. Retrieved 2026-08-20.}}

@inproceedings{thakur2023filterspace,
  title={Does Geometric Structure in Convolutional Filter Space Provide Filter Redundancy Information?},
  author={Thakur, Anshul and Abrol, Vinayak and Sharma, Pulkit},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={111--121}, year={2023}, publisher={PMLR},
  note={Simplicial geometry of CNN filter space; low relevance to V2. Retrieved 2026-08-20.}}

@inproceedings{mcguire2023topological,
  title={Do Neural Networks Trained with Topological Features Learn Different Internal Representations?},
  author={McGuire, Sarah and Jackson, Shane and Emerson, Tegan and Kvinge, Henry},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={122--136}, year={2023}, publisher={PMLR},
  note={Conclusion is CONDITIONAL: representations differ but ``can be reconciled ... using a simple affine transformation''; the topological-feature-extraction claim is a CONJECTURE in their own words. Retrieved 2026-08-20.}}

@inproceedings{davies2023fuzzy,
  title={Fuzzy c-Means Clustering in Persistence Diagram Space for Deep Learning Model Selection},
  author={Davies, Thomas and Aspinall, Jack and Wilder, Bryan and Tran-Thanh, Long},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={137--157}, year={2023}, publisher={PMLR},
  note={PMLR's index INVERTS the fourth author as ``Long, Tran-Thanh''; the surname is Tran-Thanh. Retrieved 2026-08-20.}}

@inproceedings{donmez2023ambiguity,
  title={On the Ambiguity in Classification},
  author={D{\"o}nmez, Arif},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={158--170}, year={2023}, publisher={PMLR},
  note={SINGLE author. Groupoids / noncommutative geometry. Same author has a second NeurReps paper in PMLR v228 (2024). Retrieved 2026-08-20.}}

@inproceedings{akhtiamov2023morse,
  title={Connectedness of Loss Landscapes via the Lens of {M}orse Theory},
  author={Akhtiamov, Danil and Thomson, Matt},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={171--181}, year={2023}, publisher={PMLR},
  note={Page range OVERLAPS Aslan et al. (both list p.181) -- PMLR's own numbering; reproduce as given. Retrieved 2026-08-20.}}

@inproceedings{aslan2023fundamental,
  title={Group Invariant Machine Learning by Fundamental Domain Projections},
  author={Aslan, Benjamin and Platt, Daniel and Sheard, David},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={181--218}, year={2023}, publisher={PMLR},
  note={The census's coset/quotient-coordinate paper: projects inputs into ``a geometric space which parametrises the orbits of the symmetry group''. WARNING: an INPUT ENCODING, not a memory -- do not cite as prior art for a coset REGISTER. Retrieved 2026-08-20.}}

@inproceedings{tian2023curvature,
  title={Mixed-Membership Community Detection via Line Graph Curvature},
  author={Tian, Yu and Lubberts, Zachary and Weber, Melanie},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={219--233}, year={2023}, publisher={PMLR},
  note={Discrete Ricci curvature flow on line graphs; low relevance to V2. Retrieved 2026-08-20.}}

@inproceedings{jude2023crosssession,
  title={Capturing Cross-Session Neural Population Variability through Self-Supervised Identification of Consistent Neuron Ensembles},
  author={Jude, Justin and Perich, Matthew G. and Miller, Lee E. and Hennig, Matthias H.},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={234--257}, year={2023}, publisher={PMLR},
  note={FIRST AUTHOR IS JUDE -- never ``Perich et al.''. This is REPRESENTATIONAL-DRIFT territory (cross-session, days-weeks), NOT drift along a continuous attractor. Retrieved 2026-08-20.}}

@inproceedings{vastola2023ppc,
  title={Is the Information Geometry of Probabilistic Population Codes Learnable?},
  author={Vastola, John J. and Cohen, Zach and Drugowitsch, Jan},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={258--277}, year={2023}, publisher={PMLR},
  note={THREE authors -- Zach Cohen is easily dropped. Result: the information geometry of the statistical manifold ``is directly related to measurable covariance matrices'', justifying a PCA decoder. Retrieved 2026-08-20.}}

@inproceedings{wang2023levelsets,
  title={On the Level Sets and Invariance of Neural Tuning Landscapes},
  author={Wang, Binxu and Ponce, Carlos R.},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={278--300}, year={2023}, publisher={PMLR},
  note={Morse theory + level sets on TUNING landscapes (functions of stimulus), NOT on a latent-state energy landscape. Do not conflate their level sets with our flat direction. Retrieved 2026-08-20.}}

@inproceedings{baroni2023invariance,
  title={Learning Invariance Manifolds of Visual Sensory Neurons},
  author={Baroni, Luca and Bashiri, Mohammad and Willeke, Konstantin F. and Antol{\'i}k, J{\'a}n and Sinz, Fabian H.},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={301--326}, year={2023}, publisher={PMLR},
  note={FIVE authors, first author Baroni; Sinz is senior -- never ``Sinz et al.''. The invariance manifold is in STIMULUS space. Retrieved 2026-08-20.}}

@inproceedings{iyer2023interareal,
  title={Geometry of Inter-Areal Interactions in Mouse Visual Cortex},
  author={Iyer, Ramakrishnan and Siegle, Joshua and Mahalingam, Gayathri and Olsen, Shawn and Mihalas, Stefan},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={327--353}, year={2023}, publisher={PMLR},
  note={Allen Brain Observatory Neuropixels. ``distinct subspaces of a source area mediate interactions with distinct target areas'' -- their word for channels. Same senior authors (Olsen, Mihalas) as Aitken et al. 2022 on drift geometry. Retrieved 2026-08-20.}}

@inproceedings{klindt2023topological,
  title={Topological Ensemble Detection with Differentiable Yoking},
  author={Klindt, David and Gaukstad, Sigurd and Vaupel, Melvin and Hermansen, Erik and Dunn, Benjamin},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={354--369}, year={2023}, publisher={PMLR},
  note={Builds on the grid-cell TORUS result (Gardner et al. 2022, Nature); Hermansen and Dunn co-author both. Retrieved 2026-08-20.}}

@inproceedings{xu2023conformal,
  title={Conformal Isometry of {L}ie Group Representation in Recurrent Network of Grid Cells},
  author={Xu, Dehong and Gao, Ruiqi and Zhang, Wen-Hao and Wei, Xue-Xin and Wu, Ying Nian},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={370--387}, year={2023}, publisher={PMLR},
  note={THE closest census work to V2: a Lie-group representation carried in a RECURRENT state on a TORUS, over ``the continuous attractor neural networks of grid cells''. NAMING COLLISION: their ``conformal isometry'' (neural displacement proportional to physical displacement) is UNRELATED to our conformally symplectic map. Retrieved 2026-08-20.}}

@inproceedings{duan2023seeandcopy,
  title={See and Copy: Generation of Complex Compositional Movements from Modular and Geometric {RNN} Representations},
  author={Duan, Sunny and Khona, Mikail and Bertagnoli, Adrian and Chandra, Sarthak and Fiete, Ila R.},
  booktitle={Proceedings of the 1st NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={197}, pages={388--400}, year={2023}, publisher={PMLR},
  note={Khona and Fiete also wrote the Nat Rev Neurosci 2022 attractor review -- the census and the bridge literature share people. Retrieved 2026-08-20.}}

@inproceedings{vastola2024packing,
  title={Optimal Packing of Attractor States in Neural Representations},
  author={Vastola, John},
  booktitle={Proceedings of the 2nd NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  series={Proceedings of Machine Learning Research}, volume={228}, pages={425--442}, year={2024}, publisher={PMLR},
  note={NOT in the Head's census -- added by this scout. Memory-state layout as SPHERE PACKING; ``symmetries in environmental transition statistics imply certain symmetries of the optimal neural representations''. The audience's own capacity question. Abstract read as paraphrase. Retrieved 2026-08-20.}}

@inproceedings{sagodi2024backto,
  title={Back to the Continuous Attractor},
  author={S{\'a}godi, {\'A}bel and Mart{\'i}n-S{\'a}nchez, Guillermo and Sok{\'o}{\l}, Piotr and Park, Il Memming},
  booktitle={Advances in Neural Information Processing Systems 37 (NeurIPS 2024)}, year={2024},
  note={arXiv:2408.00109 (v3, 2025-01-17). THE bridge paper. Quotables: a continuous attractor's fixed points are ``marginally stable tangent to the manifold and stable normal to the manifold''; the ``fine-tuning problem'' with its two perturbation sources, the first being ONLINE LEARNING SIGNALS; Theorem 1 (Persistent Manifold), Prop. 1 (Revival), and the memory error bound Eq. (5): (1/vol M) int_M |x(t,x0)-x0| dx0 <= t ||phi||_inf. Their bound is WORST-CASE and LINEAR IN t from an unknown perturbation size; ours is a closed-form half-life from a measured curvature -- state the contrast, do not claim their theorem. Byline reads ``Il Memming Park''. Retrieved 2026-08-20.}}

@article{burak2012fundamental,
  title={Fundamental Limits on Persistent Activity in Networks of Noisy Neurons},
  author={Burak, Yoram and Fiete, Ila R.},
  journal={Proceedings of the National Academy of Sciences}, volume={109}, number={43}, pages={17645--17650}, year={2012},
  doi={10.1073/pnas.1117386109},
  note={⛔ A CORRECTION EXISTS AND ITS RECORD IS VERIFIED: PNAS 114(20):E4117, 2017-05-08, doi 10.1073/pnas.1706051114, PMID 28483997, PMCID PMC5441759 (type: erratum). ITS TEXT COULD NOT BE RETRIEVED by any of five routes; the claim that Eq. 2 appeared incorrectly (printer's error) is SEARCH-INDEX-SOURCED and UNVERIFIED. DO NOT reproduce their Eq. 2 until the correction is read. Safe quotables: ``<[theta(t+dt)-theta(t)]^2> = 2 D dt'' and the information-diffusion inequality (Eq. 4). Retrieved 2026-08-20/21.}}

@article{khona2022attractor,
  title={Attractor and Integrator Networks in the Brain},
  author={Khona, Mikail and Fiete, Ila R.},
  journal={Nature Reviews Neuroscience}, volume={23}, pages={744--766}, year={2022},
  doi={10.1038/s41583-022-00642-0},
  note={PMID 36329249; preprint arXiv:2112.03978. Record via publisher landing page + Scholar lookup; abstract NOT transcribed by this scout. Retrieved 2026-08-20.}}

@article{seung1996eyes,
  title={How the Brain Keeps the Eyes Still},
  author={Seung, H. Sebastian},
  journal={Proceedings of the National Academy of Sciences}, volume={93}, number={23}, pages={13339--13344}, year={1996},
  doi={10.1073/pnas.93.23.13339},
  note={SOLE AUTHOR -- verified against the OpenAlex authorship record 2026-08-21 (affiliation: Bell Laboratories, Lucent Technologies). The Semantic Scholar slug ``Seung-Hopfield'' is a URL artifact, NOT a second author. Origin of the line-attractor hypothesis and of the fine-tuning framing (precisely tuned positive feedback). Retrieved 2026-08-21.}}

@article{kim2017ring,
  title={Ring Attractor Dynamics in the {D}rosophila Central Brain},
  author={Kim, Sung Soo and Rouault, Herv{\'e} and Druckmann, Shaul and Jayaraman, Vivek},
  journal={Science}, volume={356}, number={6340}, pages={849--853}, year={2017},
  doi={10.1126/science.aal4835},
  note={Some indexes give the article number ``eaal4835'' instead of pp. 849--853; ADS bibcode 2017Sci...356..849K confirms p.849. The ``overwrite the representation, then watch it persist'' experiment -- the best analogy for our write-then-hold protocol, and the sentence that MUST be followed by the no-biological-claim sentence. Retrieved 2026-08-20.}}

@article{gardner2022toroidal,
  title={Toroidal Topology of Population Activity in Grid Cells},
  author={Gardner, Richard J. and Hermansen, Erik and Pachitariu, Marius and Burak, Yoram and Baas, Nils A. and Dunn, Benjamin A. and Moser, May-Britt and Moser, Edvard I.},
  journal={Nature}, volume={602}, number={7895}, pages={123--128}, year={2022},
  doi={10.1038/s41586-021-04268-7},
  note={⚠ Author list assembled from index records, NOT the Nature masthead -- verify before printing. Claim: grid-cell population activity ``resides on a toroidal manifold that is invariant across environments and brain states''. Retrieved 2026-08-20.}}

@article{renart2003robust,
  title={Robust Spatial Working Memory through Homeostatic Synaptic Scaling in Heterogeneous Cortical Networks},
  author={Renart, Alfonso and Song, Pengcheng and Wang, Xiao-Jing},
  journal={Neuron}, volume={38}, number={3}, pages={473--485}, year={2003},
  doi={10.1016/S0896-6273(03)00255-1},
  note={⛔⛔ THE PRIOR ART THAT SCOPES V2's ANCHOR RESULT. Heterogeneity destroys the fine tuning (stored spatial information lost in seconds) and a HOMEOSTATIC SYNAPTIC SCALING mechanism RECOVERS accurate encoding; the fine-tuning problem is called ``a general feature of systems encoding internal representations of analog features''. The destroy-and-restore pattern is therefore 23 years old -- V2 may NOT present a corrective mechanism as a new idea. Record (title/authors/vol/issue/pages/DOI) primary-verified via OpenAlex 2026-08-21; the ABSTRACT TEXT is index-sourced (Cell 403) and should be read before citing. PMID 12741993.}}

@article{vafidis2022learning,
  title={Learning Accurate Path Integration in Ring Attractor Models of the Head Direction System},
  author={Vafidis, Pantelis and Owald, David and D'Albis, Tiziano and Kempter, Richard},
  journal={eLife}, volume={11}, pages={e69841}, year={2022},
  doi={10.7554/eLife.69841},
  note={⛔ THE PRIOR ART THAT SCOPES V2's DESIGNED-VS-EMERGENT GAP. A local, biologically plausible LEARNING RULE tunes the ring attractor and ``learns to path-integrate accurately'', developing connectivity ``strikingly similar to the one reported in experiments''. Therefore ``learning can produce the tuning'' is ESTABLISHED -- our 13--14 orders of magnitude in mu^2 is a measurement on OUR architecture class and recipe (N46 scope), never a general impossibility claim. PMID 35723252. Abstract text index-sourced (publisher not fetched). Retrieved 2026-08-21.}}

@misc{dinc2025ghost,
  title={A Ghost Mechanism: An Analytical Model of Abrupt Learning in Recurrent Networks},
  author={Dinc, Fatih and Cirakman, Ege and Kurtkaya, Bariscan and Yuksekgonul, Mert and Jiang, Yiqi and Schnitzer, Mark J. and Tanaka, Hidenori},
  year={2025}, eprint={2501.02378}, archivePrefix={arXiv},
  note={⚠ PREPRINT: the arXiv comment says ``to appear in Physical Review X'' -- do NOT print a PRX volume/page until the record exists. v1 2025-01-04, v2 2026-04-15. DIRECTION CHECK: slow structure APPEARING during training, not being destroyed. Nearest live neighbour to V2's training-dynamics story and highly legible to a physics-literate audience. Retrieved 2026-08-21.}}

@article{ziv2013longterm,
  title={Long-Term Dynamics of {CA1} Hippocampal Place Codes},
  author={Ziv, Yaniv and Burns, Laurie D. and Cocker, Eric D. and Hamel, Elizabeth O. and Ghosh, Kunal K. and Kitch, Lacey J. and El Gamal, Abbas and Schnitzer, Mark J.},
  journal={Nature Neuroscience}, volume={16}, pages={264--266}, year={2013}, doi={10.1038/nn.3329},
  note={The founding representational-drift observation: a unique subset of cells each day, with ~15-25%% overlap retaining the same place fields. Author list beyond ``Ziv, Burns, Cocker'' taken from index records, not the masthead -- verify. Retrieved 2026-08-20.}}

@article{driscoll2017dynamic,
  title={Dynamic Reorganization of Neuronal Activity Patterns in Parietal Cortex},
  author={Driscoll, Laura N. and Pettit, Noah L. and Minderer, Matthias and Chettih, Selmaan N. and Harvey, Christopher D.},
  journal={Cell}, volume={170}, number={5}, pages={986--999.e16}, year={2017},
  note={``mostly stable on single days but underwent major reorganization over weeks'' while behaviour was stable. PMID 28823559. Retrieved 2026-08-20.}}

@article{rule2019causes,
  title={Causes and Consequences of Representational Drift},
  author={Rule, Michael E. and O'Leary, Timothy and Harvey, Christopher D.},
  journal={Current Opinion in Neurobiology}, volume={58}, pages={141--147}, year={2019},
  note={PMID 31569062. The naming review for the phenomenon. THREE authors -- Harvey IS on this one. Retrieved 2026-08-20.}}

@article{rule2022selfhealing,
  title={Self-Healing Codes: How Stable Neural Populations Can Track Continually Reconfiguring Neural Representations},
  author={Rule, Michael E. and O'Leary, Timothy},
  journal={Proceedings of the National Academy of Sciences}, volume={119}, number={7}, pages={e2106692119}, year={2022},
  doi={10.1073/pnas.2106692119},
  note={Record via ADS bibcode 2022PNAS..11906692R + PNAS landing metadata; full text 403'd, abstract NOT transcribed. TWO authors -- Harvey is NOT on this one, unlike the 2019 review. Retrieved 2026-08-20.}}

@article{deitch2021drift,
  title={Representational Drift in the Mouse Visual Cortex},
  author={Deitch, Daniel and Rubin, Alon and Ziv, Yaniv},
  journal={Current Biology}, volume={31}, number={19}, pages={4327--4339.e6}, year={2021},
  note={Drift ``over timescales spanning minutes to days'' -- explicitly contra the prevailing notion that visual cortex is stable. Retrieved 2026-08-20.}}

@article{aitken2022geometry,
  title={The Geometry of Representational Drift in Natural and Artificial Neural Networks},
  author={Aitken, Kyle and Garrett, Marina and Olsen, Shawn and Mihalas, Stefan},
  journal={PLOS Computational Biology}, volume={18}, number={11}, pages={e1010716}, year={2022},
  doi={10.1371/journal.pcbi.1010716},
  note={Same senior authors as the census's Allen-Institute paper (Iyer et al., PMLR 197). Verbatim: drift ``most often occurs along directions that have the most in-class variance'' yet ``linear classifiers ... show little to no degradation in performance across days''; proposes DROPOUT-LIKE NOISE during continual learning as the mechanism. Abstract transcribed verbatim from the PLOS article page. Retrieved 2026-08-20.}}
```

---

## Proposed handover updates (for the Hub / Shorts Advisor)

1. **Add.37's census is PMLR v197 = NeurReps 2022.** Record the identification (it makes the census fully citable and dates it). The reframe's §2 should be written against the **2026 CFP topic list** (§1.3b, verbatim); v197 supplies *neighbour* citations, not the agenda. ✅ Verified 2026-08-21 that only two NeurReps PMLR volumes exist (v197, v228) — so the audience's last four years live only on OpenReview.
2. **Add.4 Ruling 1 re-confirmed on 2026 data.** ⚠ **New numbered question for the Head:** the EA track is *described* as being for early-stage/negative/opinion/dataset work, while V2 is a developed paper. This does not change Ruling 1, but it bears on how V2 presents itself — e.g. leaning on the paper's honest-negative content, which the track explicitly invites.
3. **Four never-quote / scope items to fold into the registry when this report is accepted:**
   (a) **Burak & Fiete Eq. 2 is under a verified 2017 erratum whose text we could not read** — do not reproduce it;
   (b) **"drift" is ambiguous in this audience** — representational drift (cross-session, biological) vs drift along a continuous attractor (within-session, state-space); every use in the NeurReps variant must be scoped in-sentence;
   (c) ⭐ **Renart, Song & Wang (2003) is prior art for "a corrective mechanism keeps the flat direction alive"** — V2's anchor may not be framed as a new idea;
   (d) ⭐ **Vafidis et al. (2022) is prior art for "learning can produce accurate tuning"** — the designed-vs-emergent gap is a measurement on our recipe (N46 scope), never a general impossibility claim.
4. ⛔ **RETRACTION of rev. 1's recommendation.** Rev. 1 proposed the sentence *"…the perturbation is identified — it is the training objective — …and it is cured by an energy anchor."* **Withdrawn** on (c) and (d). The **rev. 2 replacement** is in §2.3: *"This literature knows that a flat direction is fragile and that corrective mechanisms can keep it alive; what has not been available is the exchange rate — how much retention a given transverse curvature buys, where that law stops holding, and whether it survives the correction. We measure all three on trained models."* Every clause traces to CM-4/CM-5/CM-6 and §3.1/§3.4.
5. **Two small follow-ups worth commissioning, in priority order:** (i) **NeurReps 2025 on OpenReview** — the *current* reviewer pool's own papers, the single highest-value remaining search; (ii) primary abstracts for **Renart 2003** and **Vafidis 2022**, which became load-bearing in this pass and are presently index-sourced.
