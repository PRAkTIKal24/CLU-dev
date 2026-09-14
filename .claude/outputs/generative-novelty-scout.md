# generative-novelty-scout — web-scout report

**Task + acceptance criterion:** Map prior work nearest to the planned latent energy-store contribution across four fronts; name the novelty gap or its absence; census datasets used by that nearest work; return two decisive Part-3 verdicts.
**Status:** done (research-only; no repo edits, no git footprint).
**Dial declaration:** *none — literature/positioning recon.* No CLU numbers, no architecture recommendations, no performance claims. Laundering control / falsifier: N/A (no measurement).
**Downstream reconciliation list:** none. Nothing here retracts an existing CHLU claim.
**Search window:** 2020–2026, emphasis 2024–2026. Date of search: 2026-09-09.

---

## 0. PART-3 VERDICTS (up top, as requested)

### V1 — Has any published work demonstrated the B-vs-C distinction (generate unseen-but-valid AND refuse impossible, **same model**, **no invalid-example tuning**)?

**NO — not found.** No paper located does all three. The three nearest each break exactly one leg:

| Nearest | Does | Breaks |
|---|---|---|
| Tsakonas, Ivaldi, Mouret (2026), *Native Extrapolation Awareness in Flow-Based Conditional Generation* ("Diverging Flows"), arXiv:2602.13061 | One model both generates and detects off-manifold conditions | **Trains on synthesised invalid conditions.** Negatives mined by adversarial PGD: "`c̃_{k+1} = Π_{ℬ_ε^∞(c)}(c̃_k + η·sgn(∇_c̃ℒ_FM(c̃_k)))`", i.e. "seeking perturbations within an ε-ball that locally maximize baseline transport energy". Also treats *all* off-manifold conditions as reject-worthy ⇒ **cannot separate unseen-but-valid from impossible** — that separation is not attempted, let alone measured. |
| Nakamura, Peters, Bajcsy (2025), *Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis*, **RSS 2025**, arXiv:2502.00935 | Refusal criterion **derived from dynamics** (Hamilton–Jacobi reachability) in the latent space of a generative world model | Requires a **learned failure classifier / labelled failure set** in latent space; refusal is over *actions*, not over a requested generation; no unseen-valid-composition claim. |
| Mancini, Naeem, Xian, Akata (2021), *Open World Compositional Zero-Shot Learning*, **CVPR 2021**, arXiv:2101.12609 | Explicitly scores **infeasible** compositions ("*ripe dog*") **without invalid examples** — feasibility is inferred purely from seen compositions: `ρ_obj(s,o) = max_{ô∈O^s} cos(φ(o), φ(ô))` (Eq. 3), `ρ_state(s,o) = max_{ŝ∈S^o} cos(φ(s), φ(ŝ))` (Eq. 4), `ρ(c) = g(ρ_state, ρ_obj)` (Eq. 5), then hard-masks the output space `f_HARD(x) = argmax_{c∈C^t, ρ(c)>τ} cos(ω(x), φ(c))` (Eq. 6) | **Discriminative, not generative.** It is recognition; nothing is generated, nothing is refused-as-a-generation-request. |

**Therefore: the B-vs-C distinction as specified is unclaimed.** ⚠ Caveat: absence of evidence over ~40 queries; I cannot rule out an obscure paper. Confidence: **high** for the mainstream EBM/diffusion/world-model literature, **medium** for niche domain venues (materials, chem, EDA) where "validity" filters are ubiquitous but always post-hoc rule-based (e.g. structural validity = "minimum interatomic distance exceeds 0.5 Å") rather than a model-internal barrier criterion.

### V2 — Has any published work claimed **exact** (not approximate) unlearning in a generative model?

**YES — one, and it is a preprint, and its exactness is architectural (shard-dropping), not a property of a single learned model.**

- Golatkar, Achille, Swaminathan, Soatto (2023, rev. Oct 2024), *Training Data Protection with Compositional Diffusion Models*, **arXiv:2308.01937 — arXiv preprint only; I found no peer-reviewed venue.** Verbatim: separate adapters "trained independently on different data sources, ensuring **perfect (deterministic) isolation**"; "if a data source `D_i` is removed, we only need to remove the corresponding model to remove all information"; "CDMs are the first means to perform both selective forgetting (unlearning) and continual learning on large-scale diffusion models". Composition (Prop. 3.1): `s_θ(x_t,t) = Σ_i w_t(x_t,t)·s_θ^{(i)}(x_t,t)`, `w_t = λ_i p_t^{(i)}(x_t)/p_t(x_t)`.
- **Everything else in the concept-erasure line is explicitly approximate.** SalUn (Fan et al., ICLR 2024 Spotlight, arXiv:2310.12508) positions itself as *narrowing the gap with* "'exact' unlearning (model retraining from scratch)" — 0.2 % gap on CIFAR-10 — i.e. concedes it is not exact.
- The field's own survey agrees the exact notion does not transfer: Feng et al. (2026), *Generative Model Unlearning: A Survey…*, arXiv:2507.19894v2 (15 Aug 2026): "A straightforward solution is to retrain the model after deleting the target data, but this is **computationally prohibitive** for modern GMs and often **infeasible** in real deployment. More importantly, **retraining equivalence … becomes insufficient in generative settings.**" It lists "formal guarantees beyond retraining equivalence" as an **open problem** (Fig. 2, §VII).

**Therefore: "exact unlearning in a generative model" is claimed exactly once, by CDM, and only in the SISA sense (drop a shard), not in the sense of surgically removing a concept's modes from a single learned store with neighbours provably intact.** That latter formulation is **unclaimed**.

---

## 1. FRONT A — Compositional generation with energy-based models

### A1. Du, Li, Mordatch (2020), *Compositional Visual Generation with Energy Based Models*, **NeurIPS 2020 (spotlight)**, arXiv:2004.06030
**Mechanism (verbatim equations).** Independently trained conditional EBMs `E(x|c_i)`, composed at test time:
- Conjunction (product of experts): `p(x|c_1 and … and c_i) = Π_i p(x|c_i) ∝ e^{−Σ_i E(x|c_i)}` (Eq. 4); sampled by `x̃^k = x̃^{k−1} − (λ/2)∇_x Σ_i E_θ(x̃^{k−1}|c_i) + ω^k` (Eq. 5).
- Disjunction: `x̃^k = x̃^{k−1} − (λ/2)∇_x logsumexp(−E(x|c_1),…,−E(x|c_i)) + ω^k` (Eq. 8), valid **only under the assumption that all partition functions `Z(c_i)` are equal** — the paper states this "is not guaranteed to hold in general" and reports it holds empirically.
- Negation: `p(x|not(c_1), c_2) ∝ p(x|c_2)/p(x|c_1)^α ∝ e^{αE(x|c_1) − E(x|c_2)}` (Eq. 9), with `α = 0.01`.
**Vs capabilities.** (1) **Yes** — §4.4 "Cross Product Extrapolation" is the canonical unseen-combination demo. (2) **No.** (3) **No** — you can decline to *include* an EBM, but nothing is deleted and no proof of non-generability exists. (4) **Yes, and this is their headline continual claim** — Table 2: position acc 0.901 → 0.813 → 0.781 as shape then colour EBMs are added, vs GAN 0.941 → 0.111 → 0.117. (5) **Partial** — provenance is trivially known because the operator is user-specified, not discovered.
**Datasets.** MuJoCo Scenes (64×64, rendered spheres/cylinders/boxes, varying size/colour/position + background clutter); CelebA (128×128). Metrics: attribute classifier accuracy (classifier itself at 99.3 % position / 99.9 % colour), FID (composed EBM 45.3 vs SNGAN 64.5), pixel-MSE diversity (64.5 vs 55.4).

### A2. Du, Li, Sharma, Tenenbaum, Mordatch (2021), *Unsupervised Learning of Compositional Energy Concepts* (COMET), **NeurIPS 2021**, arXiv:2111.03042
**⭐ This is the single closest prior work to "LEARNS a factorised energy store rather than composing separately-trained models."**
**Mechanism (verbatim).** Encoder `Enc_θ(x)` emits `K` component codes; each is an energy `E_θ(x, z_k): R^D × R^M → R`. Reconstruction-by-optimisation objective (Eq. 1):
`L_MSE(θ) = ‖ argmin_x ( Σ_k E_θ(x; Enc_θ(x_i)[k]) ) − x_i ‖²`, with the inner argmin approximated by `N` gradient steps `x_i^N = x_i^{N−1} − λ∇_x Σ_k E_θ(x^{N−1}; Enc_θ(x_i)[k])` (Eq. 2), initialised from `x^0 ∼ U(0,1)`, truncated backprop to the previous step. **All `K` components share one energy network `E_θ`**, which is what lets `K` vary.
**Complexity result worth borrowing.** Remark 1: the subspace of energy-compositions `F_energy` is a **strict superset** of segmentation-mask compositions `F_mask`. Remarks 2–3 reduce 3-SAT to an energy `E(x,z)`: evaluating `E` is polynomial while evaluating `f(z) := argmin_x E(x,z)` is worst-case exponential under ETH — i.e. an energy store is strictly cheaper to *learn* than the decoder that realises it.
**Vs capabilities.** (1) **Yes** (Fig. 4 factor recombination across images and across datasets). (2) **No.** (3) **No.** (4) Not claimed. (5) **No** — components are per-image amortised codes, **not a persistent library of modes**; there is no store to attribute to.
**Datasets.** Falcor3D (7 factors: lighting intensity, lighting x/y/z, camera x/y/z), CLEVR, CelebA-HQ, Tetris. Metrics: **BetaVAE metric, MIG, MCC** (COMET* 99.41±0.15 / 19.63±2.49 / 76.55±1.35 vs best β-VAE 84.23±3.51 / 10.90±3.80 / 68.22±6.31 on Falcor3D).

### A3. Nie, Vahdat, Anandkumar (2021), *Controllable and Compositional Generation with Latent-Space Energy-Based Models* (LACE), **NeurIPS 2021**, arXiv:2110.10873
**⭐ Closest structural analogue to "latent energy module on a frozen generator."** EBM lives in the latent space of a **pre-trained, frozen** generative model (StyleGAN); joint `p(z, y)`; sampling reduced to solving an **ODE**; composition of energies via logical operators; "excels at **zero-shot generation of unseen attribute combinations**", first at 1024×1024.
**Vs capabilities.** (1) **Yes, explicitly zero-shot on unseen attribute combinations.** (2) **No.** (3) **No.** (4) Not claimed. (5) **No.**
⚠ **NOT VERIFIED FROM PRIMARY:** exact `E(z,y)` form, ODE, dataset list, and the definition of the unseen-combination split. Abstract-only retrieval; the arXiv landing page did not expose the body. **Flag for follow-up — this is the paper the Hub should have read in full before positioning.**

### A4. Liu, Li, Du, Tenenbaum, Torralba (2022), *Compositional Visual Generation with Composable Diffusion Models*, **ECCV 2022**, arXiv:2206.01714
Reinterprets diffusion models as EBMs so "data distributions defined by energy functions may be explicitly combined"; supplies AND/NOT score-composition operators. Same capability profile as A1, lifted to diffusion. ⚠ dataset list (CLEVR / relational-CLEVR / FFHQ) **NOT VERIFIED FROM PRIMARY** — the arXiv abstract page did not expose the experiments section.

### A5. Du, Durkan, Strudel, Tenenbaum, Dieleman, Grosse, Norouzi, Kushman (2023), *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC*, **ICML 2023**, PMLR 202:8489–8510, arXiv:2302.11552
**The negative result the field is built on:** they "investigate why certain types of composition fail using current techniques and conclude that **the sampler (not the model) is responsible for this failure**", then supply an energy parameterisation admitting Metropolis-corrected samplers. **Directly relevant to a Langevin-rollout claim:** naive annealed Langevin on a composed energy is *known to be the failure point*; any CHLU compositional-rollout result must state its sampler and, ideally, a Metropolis correction or an explicit reason one is unnecessary.

### Extended A (2024–2026, for currency)
- Du & Kaelbling (2024), *Position: Compositional Generative Modeling: A Single Model is Not All You Need*, **ICML 2024**, PMLR 235:11721–11732, arXiv:2402.01103 — the position statement; also notes "compositional components can often be **discovered** from data".
- Su, Liu, Wang, Tenenbaum, Du (2024), *Compositional Image Decomposition with Diffusion Models* (Decomp Diffusion), **ICML 2024**, PMLR 235:46823–46842, arXiv:2406.19298 — unsupervised decomposition into per-component diffusion models; recombination across images/datasets "even if such a scene has never been encountered before". Datasets: CLEVR, CelebA-HQ, **Virtual KITTI 2**, Tetris.
- Wu et al. (2024), *Compositional Generative Inverse Design*, **ICLR 2024**, arXiv:2401.13171 — composes learned energy functions over *subsets of design parameters* so that "each individual generative energy function ensur[es] designs do not locally fall out of distribution"; N-body + 2D multi-airfoil. **The nearest thing to an energy-as-validity-guard, but it is a soft prior inside optimisation, not a refusal.**
- Morshed & Boddeti (2026), *Compositional Generative Modeling from Decentralized Data* (DCFM), **ICML (per arXiv listing)**, arXiv:2606.10153 — enforces global conditional independence across silos without raw-data exchange; **explicitly does not address deletion, provenance, or refusal**. Datasets: Colored MNIST (10 clients, 50 % of 100 attribute combos), OGBench cube manipulation (2 clients, 75 % coverage), NIH ChestX-ray14 (4 clients, 14 attributes, rare co-occurrences).
- Soiffer, Squires, Guan, Hartford, Ravikumar (2026), *Catastrophic Compositional Generation: Why Vanilla Diffusion Models Fail to Extrapolate*, arXiv:2606.23920 (preprint) — **score-estimation error dominates inference-time approximation error when the composed target leaves the training support**; Feynman-Kac correctors can make it *worse*. Datasets: 2D Gaussians, GMMs, "Objects in a Room".
- Wiedemer, Mayilvahanan, Bethge, Brendel (2023), *Compositional Generalization from First Principles*, **NeurIPS 2023**, arXiv:2307.05596 — the theory side: "mild conditions on **only the support of the training distribution and the model architecture**" sufficient for compositional generalisation. This is the theorem a "generate unseen-but-valid" claim should be positioned against.

### ▶ What remains unclaimed on Front A
**A persistent, addressable, learned store of energy *modes* — enumerable, per-concept, editable — as opposed to either (i) separately-trained per-concept EBMs composed at inference (A1/A4/A5/CDM) or (ii) per-image amortised component codes over one shared energy net (A2/Decomp Diffusion): the "wells × modes" factorisation with an explicit inventory is not in this literature.**

---

## 2. FRONT B — Selective / abstaining GENERATION

This is by a wide margin the **thinnest** front. Refusal exists densely for LLM text (RefusalBench, arXiv:2510.10390; abstention surveys) and densely for *input-side* OOD, but **refusal of a generation request on physical/semantic impossibility is essentially unoccupied.**

### B1. Tsakonas, Ivaldi, Mouret (2026), *Native Extrapolation Awareness in Flow-Based Conditional Generation* / "Diverging Flows", arXiv:2602.13061 (preprint; submitted 13 Feb 2026, rev. 15 Jul 2026)
**Nearest overall.** Motivation verbatim: "driven by smoothness biases, flow models yield plausible outputs even for off-manifold conditions, resulting in **silent failures indistinguishable from valid predictions**." Mechanism: `L_DiFlo = E[‖v_t^θ(x_t,c) − u_t^OT‖²] + λL_repel + βL_curve`, with kinetic-energy and geodesic-alignment margins pushing the velocity field for negative condition `c̃` away from optimal transport; detection statistic `S_DOT` = spatiotemporal L1 divergence from a constant-velocity reference; split conformal calibration giving a distribution-free FPR bound at level α.
**Vs capabilities.** (2) **Closest published mechanism — but requires synthesised negatives and does not separate valid-unseen from impossible.** (1)(3)(4)(5) none.
**Datasets.** Spiral manifold (2D cond / 2D state / 10-step); ERA5 temperature forecasting (64×64 heatmaps, 6 h ahead); MNIST→SVHN style transfer with FMNIST/KMNIST/noise as OOD. Metrics: AUROC, FPR@α=0.05, MSE/PSNR/SSIM, FID, LPIPS.

### B2. Nakamura, Peters, Bajcsy (2025), *Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis* ("Latent Safety Filters"), **RSS 2025**, arXiv:2502.00935
"A latent-space generalization of HJ reachability that tractably operates directly on raw observation data … by performing safety analysis in the latent embedding space of a **generative world model**." **This is the only work found where the refusal criterion is genuinely *derived from the dynamics* rather than from a discriminator over samples** — the closest conceptual sibling to an energy-barrier/reachability criterion. Differs on: it refuses *actions*, needs a latent failure indicator, and makes no compositional-generation claim.
Related: *Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures*, arXiv:2505.00779 (2025).

### B3. Regenwetter, Srivastava, Gutfreund, Ahmed (2023), *Constraining Generative Models for Engineering Design with Negative Data*, arXiv:2306.15166 (OpenReview `FNBv2vweBI`)
Trains on **explicit constraint-violating examples**: "generative models are classically shown only 'positive' … datapoints … and are never exposed to 'negative' … datapoints to avoid." Reports 1/6 as many constraint-violating samples with 1/8 the data on some problems; best trade-off in 12 of 14 problems. Domains: ship-hull synthesis with hydrodynamic constraints, vehicle design with impact-safety constraints. **This is the exact negation of the "no tuning on invalid examples" clause — the strongest single differentiator available for capability 2.**

### B4. Mancini, Naeem, Xian, Akata (2021), *Open World Compositional Zero-Shot Learning*, **CVPR 2021**, arXiv:2101.12609 — see V1 table for equations. Datasets: **MIT-States** (53 K images, 245 objects × 115 states, 28,175 compositions, of which **26,114 (~93 %) appear in no split** and are open-world distractors) and **UT-Zappos** (12 objects × 16 materials = 192 compositions, **76 (~40 %) in no split**). Metrics: best-seen, best-unseen, harmonic mean, AUC. **The only published "this combination is infeasible" score computed without invalid supervision — and it is discriminative.**

### B5. Lee, Kim, Kim, Park (2024), *Selective Generation for Controllable Language Models*, **NeurIPS 2024**, arXiv:2307.09254 — two algorithms (`SGen^Sup`, `SGen^Semi`) controlling **FDR-E** (false discovery rate w.r.t. textual entailment) with a theoretical guarantee. Abstention in generation, but the abstain criterion is *correctness/entailment*, is calibrated on labelled data, and is unrelated to physical validity or composition.

### Adjacent, load-bearing for framing (not refusal)
- Liu, Wang, Owens, Li (2020), *Energy-based Out-of-distribution Detection*, **NeurIPS 2020**, arXiv:2010.03759 — the canonical "hard energy threshold" precedent; **input-side**, discriminative, and about samples not requests.
- Bai, Ci, Shou (2025), *Impossible Videos* (IPV-Bench), **ICML 2025**, arXiv:2503.14378 — 4 domains / 14 categories of scenes "that defy physical, biological, geographical, or social laws." **Polarity is inverted**: it asks generators to *succeed* at impossible content and finds they cannot. Useful as a validity-structured benchmark, useless as refusal prior art.
- Judging-not-refusing: **WorldModelBench** (arXiv:2502.20694, 2025) — five violation types: Newton's laws, **mass conservation**, fluid dynamics, penetration, gravity; **VideoPhy** (Bansal et al., arXiv:2406.03520, 2024) — 688 prompts over solid-solid / solid-fluid / fluid-fluid, auto-evaluator VideoCon-Physics, best open model CogVideoX-5B at 53 % physical commonsense; **VideoPhy-2** (arXiv:2503.06800, 2025); **PhyGround** (arXiv:2605.10806, 2026) — "standard video metrics such as FVD, SSIM, PSNR, and CLIP-based similarity … are not designed to detect violations of physical laws." **All post-hoc scorers of produced output; none refuse a request.**
- Contradictory-prompt handling: *Image Generation from Contextually-Contradictory Prompts*, arXiv:2506.01929 (2025) — defines "contextual contradiction" (e.g. "a butterfly in a beehive") and **works around it via LLM-rewritten proxy prompts**, i.e. the field's response to an impossible request is to *satisfy it anyway*, not to refuse.

### ▶ What remains unclaimed on Front B
**No published generative model refuses a requested condition using a criterion intrinsic to its own learned dynamics (energy barrier / reachability), with zero invalid-example supervision, while simultaneously *accepting* unseen-but-valid conditions. The accept/reject asymmetry between "novel-valid" and "impossible" is, as far as this scout can find, unformulated as a task.**

---

## 3. FRONT C — Generative unlearning / concept erasure

### The strongest three (by adoption + evaluation depth)
1. **ESD** — Gandikota, Materzyńska, Fiotto-Kaufman, Bau (2023), *Erasing Concepts from Diffusion Models*, **ICCV 2023**, arXiv:2303.07345. Fine-tunes weights using **negative guidance from a frozen copy as teacher**, "given only the name of the style"; variants **ESD-x** (cross-attention only → style) and **ESD-u** (non-cross-attention → NSFW). Guarantee claimed: **"removes concepts permanently from model weights rather than at inference time, making circumvention impossible even with weight access."** ⚠ **That claim is falsified — see attacks.** Reported: I2P nude generations 796 → 134 (single-sourced summary, not read from the PDF; the ICCV PDF returned HTTP 403).
2. **UCE** — Gandikota, Orgad, Belinkov, Materzyńska, Bau (2024), *Unified Concept Editing in Diffusion Models*, **WACV 2024**, arXiv:2308.14761. **Training-free closed-form** cross-attention weight edit; scales to concurrent edits; unifies bias/copyright/offensive-content editing. **Approximate** — an analytic projection, no non-generability proof.
3. **MACE** — Lu, Wang, Li, Liu, Kong (2024), *MACE: Mass Concept Erasure in Diffusion Models*, **CVPR 2024**, arXiv:2403.06135. Closed-form cross-attention refinement + per-concept LoRA; **scales the erasure scope to 100 concepts**, balancing generality (synonyms) vs specificity (neighbours). **Approximate.**
   - Honourable mention: **SalUn** — Fan, Liu, Zhang, Wong, Wei, Liu (2024), **ICLR 2024 Spotlight**, arXiv:2310.12508 — gradient-based **weight saliency** masks; "the first principled MU approach adaptable enough to … erase … data, classes, or concepts in both image classification and generation"; self-describes as *approaching* exact unlearning (0.2 % gap, CIFAR-10 random forgetting).
   - The one **exact** method: **CDM** — Golatkar et al., arXiv:2308.01937 (preprint) — see V2.

### Attack papers (erased concepts recoverable) — all four verified
- **Pham, Marshall, Cohen, Mittal, Hegde (2024)**, *Circumventing Concept Erasure Methods For Text-to-Image Generative Models*, **ICLR 2024**, arXiv:2308.01508. Breaks **seven** methods — ESD, Selective Amnesia, Forget-Me-Not, Ablating Concepts, UCE, Negative Prompt, Safe Latent Diffusion — via **Concept Inversion** over word-embedding space, "with **zero extra training or fine-tuning**". Verdict: "post hoc concept erasure in generative models provides a **false sense of security**."
- **Zhang, Jia, Chen, Chen, Zhang, Liu, Ding, Liu (2024)**, *To Generate or Not? …* (**UnlearnDiffAtk**), **ECCV 2024**, arXiv:2310.11868 — adversarial prompt generation exploiting the diffusion model's own classification ability, no auxiliary classifier needed; "reveal[s] the lack of robustness of current safety-driven unlearning techniques."
- **Suriyakumar et al. (2024)**, *Unstable Unlearning: The Hidden Risk of Concept Resurgence in Diffusion Models*, arXiv:2410.08074 — **benign** fine-tuning on *unrelated* images causes erased concepts to return, tested against MACE on SD v1.4 and v2.1.
- **"Erased but Not Forgotten: How Backdoors Compromise Concept Erasure"**, arXiv:2504.21072 (2025) — backdoor-based recovery.
- Also: *On the Vulnerability of Concept Erasure in Diffusion Models* / *Rethinking the Vulnerability of Concept Erasure*, arXiv:2502.17537 (2025) — "many state-of-the-art unlearning methods **'conceal' rather than fully erase** latent concept information."

### Ablating Concepts (mechanism, for the census)
Kumari, Zhang, Wang, Shechtman, Zhang, Zhu (2023), **ICCV 2023**, arXiv:2303.13516 — matches the target concept's image distribution to an **anchor** concept's ("Grumpy Cat → Cats"); all results on `stable-diffusion-v1-4`.

### ▶ What remains unclaimed on Front C
**Concept-level deletion that is (a) exact/byte-level, (b) *within a single model's store* rather than by dropping a data-source shard, and (c) accompanied by a *neighbours-intact* guarantee rather than an empirical specificity metric. Every peer-reviewed method is approximate and at least five papers demonstrate recovery of "erased" concepts. CDM gets (a) but not (b) or (c), and is unrefereed.**

---

## 4. FRONT D — Memory-augmented / retrieval-augmented generative models with explicit, editable memory

### D1. Blattmann, Rombach, Oktay, Müller, Ommer (2022), *Retrieval-Augmented Diffusion Models*, **NeurIPS 2022**, arXiv:2204.11824
"Complement the diffusion model with a retrieval-based approach and introduce an **explicit memory in the form of an external database**"; trained with CLIP-retrieved neighbours of each training instance. Key property: **"simply swapping the database for one with different contents transfers a trained model post-hoc to a novel domain."** ⇒ **Editable memory, capability-3-adjacent by deletion-from-DB**, but: memory holds *exemplars*, not modes; no energy landscape; no refusal; no non-generability guarantee (the model's weights retain the concept).

### D2. Golatkar, Achille, Swaminathan, Soatto (2023), CDM, arXiv:2308.01937 — memory = a set of independently-trained shard models composed by posterior weights. Uniquely covers **(3) exact deletion, (4) continual learning, (5) attribution** ("CDMs enable determining the importance of a subset of the data (attribution) in generating particular samples") — **but at data-source granularity, and it does not cover (1) unseen-valid composition as a claim or (2) refusal at all.** Datasets: MSCOCO (split by aesthetic score, 8-split configs), CUB-200, Stanford Cars, Oxford Pets.

### D3. Das, Chaudhury, Nelson, Melnyk, Swaminathan, Dai, Lozano, Kollias, Chenthamarakshan, Navrátil, Dan, Chen (2024), *Larimar: Large Language Models with Episodic Memory Control*, **ICML 2024**, PMLR 235, arXiv:2403.11901 — distributed episodic memory supporting **one-shot updates without gradient descent**, 4–10× speed-up over editing baselines, plus mechanisms for **selective fact forgetting** and context-length generalisation. Closest to "enrol/delete an item in an explicit memory" for a generative model, but the memory is a linear associative slot memory, not an energy landscape, and forgetting is empirical.

### D4. Berges, Oğuz, Haziza, Yih, Zettlemoyer, Ghosh (2024/2025), *Memory Layers at Scale*, **ICML 2025**, arXiv:2412.09764 — trainable key–value lookup adding parameters without FLOPs; scaled to **128 B memory parameters, 1 T tokens**; gains "especially pronounced for factual tasks." Explicit and in principle editable, but no deletion/provenance/refusal claims. ⚠ author list **not verified from primary**.

### D5. Modern Hopfield line (energy-landscape memory, the mechanism CHLU shares)
Ramsauer et al. (2021), *Hopfield Networks is All You Need*, **ICLR 2021** (log-sum-exp energy, exponential capacity, attention equivalence); Hoover et al. (2023), *Energy Transformer*, **NeurIPS 2023**; Santos et al. (2024), *Hopfield-Fenchel-Young Networks*, arXiv:2411.08590; *Modern Hopfield Networks with Continuous-Time Memories*, arXiv:2502.10122 (2025); the ICLR 2025 *New Frontiers in Associative Memory* workshop. **Memories are stored patterns = energy minima; retrieval = relaxation.** ⚠ **Deletion by removing a stored pattern is structurally obvious in classical MHN but I found no paper that claims and *measures* concept deletion with neighbours intact in a modern-Hopfield generative model.** That is a genuine gap and also a genuine "obvious to a reviewer" risk.
Adjacent but classification-only: **MUNKEY** — Laguna, da Silva Gonçalves, Vandenhirtz, Ryser, Cannistraci, Vogt (2026), *Rethinking Machine Unlearning: Models Designed to Forget via Key Deletion*, arXiv:2603.15033 — unlearning literally as `M_u = M \ {(k_i, v_i) | i ∈ D_f}`, ≈0 s unlearning; **ViT classifiers, not generative**; claims approximate not provable forgetting. Datasets: CIFAR-10/100, Tiny-ImageNet, Blood/Path/DermaMNIST.

### ▶ What remains unclaimed on Front D
**An explicit editable memory whose entries are *energy modes* that both (i) compose to generate and (ii) are individually deletable with a stated non-generability consequence. Existing explicit memories store exemplars (RDM), facts (Larimar), key-value parameters (Memory Layers), or whole shard models (CDM) — none stores composable wells×modes, and none pairs deletion with a generation-side guarantee.**

---

## 5. DATASET CENSUS

### 5.1 Main table

| Dataset | Used by (front) | Factor structure / held-out-combination split | Modality | Size | Typical metrics |
|---|---|---|---|---|---|
| **CLEVR** (Johnson et al. 2017) | A: COMET, Decomp Diffusion, Composable Diffusion ⚠; B: (via CoGenT) | Labelled: shape/colour/size/material/position. **CLEVR-CoGenT is the standard compositional split** | Rendered images (+VQA) | 100 K images (CLEVR v1) | decomposition/recomposition quality, MSE, FID, VQA acc |
| **CelebA / CelebA-HQ** | A: Du 2020 (CelebA 128²), COMET (HQ), Decomp Diffusion (HQ), LACE ⚠ | 40 binary attributes labelled; **no standard held-out-combination split** — each paper defines its own attribute conjunctions | Face images | 202,599 / 30,000 (HQ) | attribute-classifier acc, FID, pixel-MSE diversity |
| **MuJoCo Scenes** (Du et al. 2020, bespoke) | A: Du 2020 | Labelled shape/position/size/colour. **Defines its own cross-product extrapolation split: spheres of all sizes appear only in the top-right 1 % / 10 % / 100 % of positions** | Rendered 64×64 | not stated in paper | position acc, colour acc, size/position regression error |
| **Falcor3D** (NVIDIA) | A: COMET | 7 labelled factors: lighting intensity, lighting x/y/z, camera x/y/z. No held-out-combination split used | Rendered images | not stated in COMET | BetaVAE metric, MIG, MCC |
| **Tetris** | A: COMET, Decomp Diffusion | Object-level blocks; no factor split | Synthetic images | small | qualitative decomposition |
| **dSprites** (Matthey et al. 2017) | A/(disentanglement staple); split literature (Montero 2021, Schott 2022) | **6 latent factors**: colour(1, white), shape(3), scale(6), rotation(40), x(32), y(32) → all combinations exhaustively enumerated | 64×64 binary images | **737,280** (full Cartesian product) | MIG/DCI/BetaVAE; for splits: reconstruction error on excluded combos |
| **Shapes3D / 3dshapes** (Burgess & Kim 2018) | same | **6 factors**: floor hue(10), wall hue(10), object hue(10), scale(8), shape(4), orientation(15); exhaustive | 64×64 RGB | **480,000** = 10·10·10·8·4·15 | as above |
| **MPI3D** (Gondal et al., NeurIPS 2019) | same | **7 factors**: object colour(6), shape(6), size(2), camera height(3), background colour(3), DOF1(40), DOF2(40) | real + simulated images | **1,036,800** | as above |
| **MIT-States** | B: OW-CZSL | 245 objects × 115 states = **28,175 compositions**; closed-world split 1,262 seen + 300/400 unseen; **~93 % of compositions in no split → open-world distractors** | Web images | 53 K images | best-seen / best-unseen / HM / AUC |
| **UT-Zappos** | B: OW-CZSL | 12 objects × 16 materials = 192 compositions; 83 seen + 15/18 unseen; **~40 % in no split** | Shoe images | — | as above |
| **ERA5** | B: Diverging Flows | No labelled factors; validity = on-manifold weather state | 64×64 temperature heatmaps, 6 h horizon | reanalysis | AUROC, FPR@α, MSE/PSNR/SSIM |
| **MNIST→SVHN (+FMNIST/KMNIST/noise)** | B: Diverging Flows | OOD-vs-ID, not factors | images | standard | AUROC, FID, LPIPS |
| **Colored MNIST** | A: DCFM | 10 clients, **50 % coverage of 100 colour×digit combinations** — an explicit unseen-combination split | images | 60 K base | combo coverage, FID |
| **OGBench (cube manipulation)** | A: DCFM | 2 clients, **75 % coverage** | robot trajectories | — | success rate |
| **NIH ChestX-ray14** | A: DCFM | 14 disease attributes, focus on **rare co-occurrences** | X-ray images | 112,120 | attribute co-occurrence fidelity |
| **I2P (Inappropriate Image Prompts)** | C: ESD, UCE, MACE, SalUn, SLD, all attacks | Prompt categories: hate, harassment, violence, self-harm, sexual, shocking, illegal | text prompts | **>4,700** (commonly cited 4,703) | NudeNet detections, Q16 classifier rate, attack success rate |
| **Imagenette / ImageNet classes** | C: ESD, SalUn, UnlearnDiffAtk | 10 held-out classes as "concepts" | images | — | unlearning acc, FID |
| **COCO-30K** | C: ESD, UCE, MACE (retainability) | none | text-image | 30 K captions | FID, CLIP score |
| **UnlearnCanvas** (Zhang et al., **NeurIPS 2024 D&B**, arXiv:2402.11846) | C | **Style × object grid** — the only concept-erasure dataset with explicit factor structure | high-res stylized images | — | 7 metrics: unlearning effectiveness, retainability, quality, efficiency |
| **Six-CD** (Ren et al., **CVPR 2025**, arXiv:2406.14855) | C | 6 concept categories; **dual-version malicious/benign prompt pairs** | text prompts | 991 harm + 1,539 nudity prompts; 94 celebrities, 100 characters, 10 objects, 10 styles | in-prompt CLIP score, removal rate |
| **MSCOCO / CUB-200 / Stanford Cars / Oxford Pets** | C+D: CDM | Data-source shards (8-way), not factors | images/captions | standard | FID, attribution, forgetting |
| **CIFAR-10** | C: SalUn; D: MUNKEY | class-level forget sets | images | 60 K | UA/MIA/retain acc |

⚠ Sizes marked "not stated" were not in the primary text I read; I did not substitute a remembered number.

### 5.2 THE COMMON SET (datasets appearing in ≥2 of the four fronts)

Honest answer: **the common set is small, and that is itself the finding.** The four literatures are **almost disjoint in their evaluation venues.**

| Dataset | Fronts | Note |
|---|---|---|
| **CLEVR** (incl. CoGenT) | A (COMET, Decomp Diffusion, Composable Diffusion), and B **only via** the discriminative CoGenT split | The single best-attested compositional venue |
| **CelebA / CelebA-HQ** | A (Du 2020, COMET, Decomp Diffusion, LACE ⚠), C (identity/face erasure appears in erasure benchmarks; **not verified as a named split**) | Attribute conjunctions, no canonical split |
| **MSCOCO / COCO-30K** | C (ESD/UCE/MACE retainability), D (CDM) | Used as a *retention* control, never as a compositional venue |
| **CIFAR-10** | C (SalUn), D (MUNKEY) | Class-level forgetting only |

**Datasets that do NOT bridge, contrary to the task's expectation:** **dSprites, Shapes3D, MPI3D appear in the disentanglement/compositional-split literature (Montero 2021; Schott 2022) but I found them in NONE of the concept-erasure, refusal, or memory-augmented-generation papers surveyed.** Conversely **I2P, UnlearnCanvas, Six-CD appear in NO compositional-generation paper.** ⇒ **There is currently no dataset on which all of capabilities 1–5 could be evaluated end-to-end. Any paper claiming all five must either (a) construct one, or (b) accept a per-capability venue split and defend it.** That is the single most operationally consequential finding of this census.

### 5.3 Standard unseen-combination (compositional) splits — who defined them

| Dataset | Standard compositional split? | Defined by |
|---|---|---|
| **CLEVR** | **YES — CLEVR-CoGenT.** "In Condition A, all cubes are gray, blue, brown, or yellow and all cylinders are red, green, purple, or cyan; in Condition B, cubes and cylinders swap color palettes. Both conditions contain spheres of all colors." | Johnson, Hariharan, van der Maaten, Fei-Fei, Lawrence Zitnick, Girshick (2017), *CLEVR*, **CVPR 2017**, arXiv:1612.06890 |
| **dSprites** | **YES — a de-facto standard pair of conditions:** *recombination-to-element* (held-out combination of values each seen elsewhere) and *recombination-to-range* (held-out **range** of a factor) | Montero, Ludwig, Costa, Malhotra, Bowers (2021), *The role of disentanglement in generalisation*, **ICLR 2021**. Finding: "generalisation was limited to the **Recombination-to-element** condition", suggesting "models are simply generalising on the basis of overall similarity (interpolation)" |
| **Shapes3D** | **YES**, same two conditions as dSprites | Montero et al. (2021), ICLR 2021 |
| **MPI3D** | **YES**, via the recompose/interpolate/extrapolate protocol | Schott, von Kügelgen, Träuble, Gehler, Russell, Bethge, Schölkopf, Locatello, Brendel (2022), *Visual Representation Learning Does Not Generalize Strongly Within the Same Domain*, **ICLR 2022**, arXiv:2107.08221 — evaluates 17 methods on dSprites/Shapes3D/MPI3D + CelebGlow with "recomposing, interpolating, or extrapolating only existing factors" |
| **MuJoCo Scenes** | **YES but bespoke** — the 1 %/10 %/100 % position×size cross-product split | Du, Li, Mordatch (2020), NeurIPS 2020, §4.4 |
| **MIT-States / UT-Zappos (CZSL)** | **YES** — closed-world seen/unseen split; **open-world** variant is the Mancini contribution | closed-world split from Purushwalkam et al. (TMN), adopted by Mancini et al. (2021) CVPR |
| **CelebA/CelebA-HQ** | **NO standard split** — attribute conjunctions are chosen ad hoc per paper | — (NOT-FOUND) |
| **UnlearnCanvas** | Style×object grid, but the **grouped style-object unlearning** scenario is the closest thing to a compositional split | Zhang et al., NeurIPS 2024 D&B, arXiv:2402.11846 |
| **I2P / Six-CD / COCO-30K** | **NO** | — (NOT-FOUND) |

⚠ **A CHLU-relevant trap:** the classic disentanglement splits (Montero; Schott) are the venues where the field's *negative* result lives — "models supported only **weak** combinatorial generalisation" irrespective of disentanglement. Evaluating capability 1 there is high-credibility but also the hardest available bar, and the 2026 result (Soiffer et al., arXiv:2606.23920) says score-estimation error dominates out of support. Reviewers who know this literature will read a strong unseen-combination number there as extraordinary and demand extraordinary controls.

---

## 6. SECONDARY LENS — world models, time series, physical AI (listed separately, not forced)

Where "impossible" is defined by **dynamics or a conservation law rather than a label**:

| Venue | Citation | What "impossible" means |
|---|---|---|
| **WorldModelBench** | arXiv:2502.20694 (2025) | **Five violation types: Newton's laws, mass conservation, fluid dynamics, penetration, gravity.** "detects subtle violations such as irregular changes in object size that breach the mass conservation law" |
| **VideoPhy / VideoPhy-2** | Bansal et al., arXiv:2406.03520 (2024); arXiv:2503.06800 (2025) | Physical commonsense over solid-solid / solid-fluid / fluid-fluid; 688 prompts; auto-evaluator VideoCon-Physics |
| **Physion** | Bear et al., **NeurIPS 2021 D&B**, arXiv:2106.08261 | 8 scenarios (dominoes, support, collide, contain, drop, link, roll, drape); object-contact-prediction task; humans >> models |
| **Physion++** | **NeurIPS 2023 D&B** | requires online inference of latent physical properties (mass, friction, …) |
| **IPV-Bench / Impossible Videos** | Bai, Ci, Shou, **ICML 2025**, arXiv:2503.14378 | 4 domains, 14 categories "that defy physical, biological, geographical, or social laws" — a *taxonomy of impossibility*, ready-made for a refusal benchmark |
| **PhyGround** | arXiv:2605.10806 (2026) | Explicitly argues FVD/SSIM/PSNR/CLIP "are not designed to detect violations of physical laws" |
| **CrashTwin** (per PhyGround-adjacent listing) | arXiv:2606.28757 (2026), physics-grounded multi-agent world-model benchmark | "spatio-temporal consistency, **momentum and energy conservation**, and world-dynamics integrity" |
| **Compositional Generative Inverse Design** | Wu et al., **ICLR 2024**, arXiv:2401.13171 | N-body interaction + 2D multi-airfoil; validity = physics feasibility; energy composition used as an *in-distribution guard* |
| **Latent Safety Filters** | Nakamura et al., **RSS 2025**, arXiv:2502.00935 | Impossibility = **unreachability of a safe set** under learned latent dynamics — the closest published match to a dynamics-derived refusal criterion |
| **ERA5 temperature forecasting** | via Diverging Flows, arXiv:2602.13061 | Time-series/weather; "impossible" = off-manifold condition |
| **Crystal/materials generation** | e.g. LeMat-GenBench, arXiv:2512.04562 | Validity by **hard geometric rules** ("minimum interatomic distance exceeds 0.5 Å"; invalid if unit-cell volume < 0.1 Å³ or polar sine of lattice vectors < 10⁻³) — **rule-based post-hoc filters, explicitly NOT a learned refusal**; the honest contrast case |

**Reading:** the physical-AI world is *rich in impossibility taxonomies and post-hoc judges* and *empty of models that decline*. If the planned capability-2 claim needs a venue where "impossible" is dynamics-defined rather than label-defined, **IPV-Bench's taxonomy + WorldModelBench's five violation types are the two ready-made vocabularies**, and Latent Safety Filters is the one paper that must be cited as the mechanism's nearest neighbour.

---

## 7. Confidence & gaps

**Verified from primary source (read the paper body / equations):** Du et al. 2020 (Eqs. 1–10, Tables 1–2, datasets, cross-product split); COMET (Eqs. 1–2, Remarks 1–3, Alg. 1, Table 1, datasets, metrics); Mancini et al. 2021 (Eqs. 3–8, dataset counts, Tables 1–2); Diverging Flows (objective, PGD negatives, DOT score, conformal calibration, dataset table, metrics); Feng et al. survey (intro + Fig. 2 open problems); CDM (quoted claims + Prop. 3.1 + dataset list, via HTML); 3dshapes factor counts; dSprites size/factors.

**Single-sourced / abstract-only — DO NOT quote as verified:**
- **LACE (arXiv:2110.10873)** — energy form, ODE, datasets, and the *definition of the unseen-combination split* are all unread. **This is the highest-value follow-up: it is the closest architectural sibling (latent EBM on a frozen generator, zero-shot unseen combos) and the Hub is currently positioning against a summary.**
- **Composable Diffusion (arXiv:2206.01714)** — dataset list unread.
- ESD's I2P 796→134 number and the ESD objective equation (ICCV PDF returned HTTP 403; arXiv abstract only).
- Memory Layers at Scale author list.
- I2P count "4,703" (secondary summaries agree; primary says ">4,700").
- Six-CD venue: URLs indicate **CVPR 2025** (openaccess CVPR2025 + cvpr.thecvf.com/virtual/2025); one summary said CVPR 2024 — **I use CVPR 2025**.

**NOT-FOUND (searched, not located — never inferred):**
- Any generative model that refuses a generation request on an intrinsic energy/reachability criterion with zero invalid-example supervision.
- Any paper measuring accept-unseen-valid vs reject-impossible as a *paired* task.
- Any standard held-out-combination split for CelebA-HQ, I2P, Six-CD, or COCO-30K.
- Any modern-Hopfield **generative** paper claiming and measuring concept deletion with neighbours intact.
- Any peer-reviewed (non-preprint) claim of exact unlearning in a generative model.

**What to search next (ranked):** (1) full text of LACE and Composable Diffusion; (2) OpenReview ICLR 2026 / NeurIPS 2026 submissions on "compositional refusal", "feasibility-aware generation", "validity-aware world models" — this is exactly the window where such a paper would first appear and my coverage of unindexed 2026 submissions is weakest; (3) CZSL→generation crossover (does anyone *generate* under a feasibility mask?); (4) the ICLR 2025 *New Frontiers in Associative Memory* workshop proceedings for Hopfield-store deletion; (5) molecular/crystal conditional generators for a *learned* (not rule-based) infeasibility score.

---

## 8. Bibtex-ready refs

```bibtex
@inproceedings{du2020compositional, author={Du, Yilun and Li, Shuang and Mordatch, Igor}, title={Compositional Visual Generation with Energy Based Models}, booktitle={NeurIPS}, year={2020}, note={arXiv:2004.06030}}
@inproceedings{du2021comet, author={Du, Yilun and Li, Shuang and Sharma, Yash and Tenenbaum, Josh and Mordatch, Igor}, title={Unsupervised Learning of Compositional Energy Concepts}, booktitle={NeurIPS}, year={2021}, note={arXiv:2111.03042}}
@inproceedings{nie2021lace, author={Nie, Weili and Vahdat, Arash and Anandkumar, Anima}, title={Controllable and Compositional Generation with Latent-Space Energy-Based Models}, booktitle={NeurIPS}, year={2021}, note={arXiv:2110.10873}}
@inproceedings{liu2022composable, author={Liu, Nan and Li, Shuang and Du, Yilun and Tenenbaum, Joshua B. and Torralba, Antonio}, title={Compositional Visual Generation with Composable Diffusion Models}, booktitle={ECCV}, year={2022}, note={arXiv:2206.01714}}
@inproceedings{du2023reduce, author={Du, Yilun and Durkan, Conor and Strudel, Robin and Tenenbaum, Joshua B. and Dieleman, Sander and Grosse, Rob and Norouzi, Mohammad and Kushman, Nate}, title={Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC}, booktitle={ICML}, year={2023}, series={PMLR 202:8489--8510}, note={arXiv:2302.11552}}
@inproceedings{du2024position, author={Du, Yilun and Kaelbling, Leslie}, title={Position: Compositional Generative Modeling: A Single Model is Not All You Need}, booktitle={ICML}, year={2024}, series={PMLR 235:11721--11732}, note={arXiv:2402.01103}}
@inproceedings{su2024decomp, author={Su, Jocelin and Liu, Nan and Wang, Yanbo and Tenenbaum, Joshua B. and Du, Yilun}, title={Compositional Image Decomposition with Diffusion Models}, booktitle={ICML}, year={2024}, series={PMLR 235:46823--46842}, note={arXiv:2406.19298}}
@inproceedings{wu2024inversedesign, author={Wu, Tailin and others}, title={Compositional Generative Inverse Design}, booktitle={ICLR}, year={2024}, note={arXiv:2401.13171}}
@article{soiffer2026catastrophic, author={Soiffer, Duncan and Squires, Chandler and Guan, Yuan and Hartford, Jason and Ravikumar, Pradeep}, title={Catastrophic Compositional Generation: Why Vanilla Diffusion Models Fail to Extrapolate}, journal={arXiv:2606.23920}, year={2026}}
@inproceedings{morshed2026dcfm, author={Morshed, Mashrur M. and Boddeti, Vishnu Naresh}, title={Compositional Generative Modeling from Decentralized Data}, booktitle={ICML}, year={2026}, note={arXiv:2606.10153}}
@inproceedings{wiedemer2023first, author={Wiedemer, Thadd\"aus and Mayilvahanan, Prasanna and Bethge, Matthias and Brendel, Wieland}, title={Compositional Generalization from First Principles}, booktitle={NeurIPS}, year={2023}, note={arXiv:2307.05596}}
@article{tsakonas2026diverging, author={Tsakonas, Constantinos and Ivaldi, Serena and Mouret, Jean-Baptiste}, title={Native Extrapolation Awareness in Flow-Based Conditional Generation (Diverging Flows)}, journal={arXiv:2602.13061}, year={2026}}
@inproceedings{nakamura2025latentsafety, author={Nakamura, Kensuke and Peters, Lasse and Bajcsy, Andrea}, title={Generalizing Safety Beyond Collision-Avoidance via Latent-Space Reachability Analysis}, booktitle={Robotics: Science and Systems (RSS)}, year={2025}, note={arXiv:2502.00935}}
@article{regenwetter2023negative, author={Regenwetter, Lyle and others}, title={Constraining Generative Models for Engineering Design with Negative Data}, journal={arXiv:2306.15166}, year={2023}}
@inproceedings{mancini2021owczsl, author={Mancini, Massimiliano and Naeem, Muhammad Ferjad and Xian, Yongqin and Akata, Zeynep}, title={Open World Compositional Zero-Shot Learning}, booktitle={CVPR}, year={2021}, pages={5222--5230}, note={arXiv:2101.12609}}
@inproceedings{lee2024selectivegen, author={Lee, Minjae and Kim, Kyungmin and Kim, Taesoo and Park, Sangdon}, title={Selective Generation for Controllable Language Models}, booktitle={NeurIPS}, year={2024}, note={arXiv:2307.09254}}
@inproceedings{liu2020energyood, author={Liu, Weitang and Wang, Xiaoyun and Owens, John and Li, Yixuan}, title={Energy-based Out-of-distribution Detection}, booktitle={NeurIPS}, year={2020}, note={arXiv:2010.03759}}
@inproceedings{bai2025impossible, author={Bai, Zechen and Ci, Hai and Shou, Mike Zheng}, title={Impossible Videos}, booktitle={ICML}, year={2025}, note={arXiv:2503.14378}}
@inproceedings{gandikota2023esd, author={Gandikota, Rohit and Materzy\'nska, Joanna and Fiotto-Kaufman, Jaden and Bau, David}, title={Erasing Concepts from Diffusion Models}, booktitle={ICCV}, year={2023}, note={arXiv:2303.07345}}
@inproceedings{gandikota2024uce, author={Gandikota, Rohit and Orgad, Hadas and Belinkov, Yonatan and Materzy\'nska, Joanna and Bau, David}, title={Unified Concept Editing in Diffusion Models}, booktitle={WACV}, year={2024}, note={arXiv:2308.14761}}
@inproceedings{lu2024mace, author={Lu, Shilin and Wang, Zilan and Li, Leyang and Liu, Yanzhu and Kong, Adams Wai-Kin}, title={MACE: Mass Concept Erasure in Diffusion Models}, booktitle={CVPR}, year={2024}, note={arXiv:2403.06135}}
@inproceedings{fan2024salun, author={Fan, Chongyu and Liu, Jiancheng and Zhang, Yihua and Wong, Eric and Wei, Dennis and Liu, Sijia}, title={SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation}, booktitle={ICLR}, year={2024}, note={Spotlight; arXiv:2310.12508}}
@inproceedings{kumari2023ablating, author={Kumari, Nupur and Zhang, Bingliang and Wang, Sheng-Yu and Shechtman, Eli and Zhang, Richard and Zhu, Jun-Yan}, title={Ablating Concepts in Text-to-Image Diffusion Models}, booktitle={ICCV}, year={2023}, note={arXiv:2303.13516}}
@article{golatkar2023cdm, author={Golatkar, Aditya and Achille, Alessandro and Swaminathan, Ashwin and Soatto, Stefano}, title={Training Data Protection with Compositional Diffusion Models}, journal={arXiv:2308.01937}, year={2023}, note={rev. Oct 2024; preprint, no peer-reviewed venue found}}
@inproceedings{pham2024circumventing, author={Pham, Minh and Marshall, Kelly O. and Cohen, Niv and Mittal, Govind and Hegde, Chinmay}, title={Circumventing Concept Erasure Methods For Text-to-Image Generative Models}, booktitle={ICLR}, year={2024}, note={arXiv:2308.01508}}
@inproceedings{zhang2024unlearndiffatk, author={Zhang, Yimeng and Jia, Jinghan and Chen, Xin and Chen, Aochuan and Zhang, Yihua and Liu, Jiancheng and Ding, Ke and Liu, Sijia}, title={To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now}, booktitle={ECCV}, year={2024}, note={arXiv:2310.11868}}
@article{unstable2024resurgence, title={Unstable Unlearning: The Hidden Risk of Concept Resurgence in Diffusion Models}, journal={arXiv:2410.08074}, year={2024}}
@article{feng2026genmusurvey, author={Feng, Xiaohua and Zhang, Jiaming and Yu, Fengyuan and Wang, Chengye and Zhang, Li and Li, Kaixiang and Li, Yuyuan and Lyu, Lingjuan and Chen, Chaochao and Yin, Jianwei}, title={Generative Model Unlearning: A Survey through Target Events, Unlearning Operators, and Evaluation Protocols}, journal={arXiv:2507.19894v2}, year={2026}}
@inproceedings{zhang2024unlearncanvas, title={UnlearnCanvas: Stylized Image Dataset for Enhanced Machine Unlearning Evaluation in Diffusion Models}, booktitle={NeurIPS Datasets and Benchmarks}, year={2024}, note={arXiv:2402.11846}}
@inproceedings{ren2025sixcd, author={Ren, Jie and others}, title={Six-CD: Benchmarking Concept Removals for Text-to-image Diffusion Models}, booktitle={CVPR}, year={2025}, note={arXiv:2406.14855}}
@inproceedings{schramowski2023sld, author={Schramowski, Patrick and Brack, Manuel and Deiseroth, Bj\"orn and Kersting, Kristian}, title={Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models}, booktitle={CVPR}, year={2023}, note={introduces I2P; arXiv:2211.05105}}
@inproceedings{blattmann2022rdm, author={Blattmann, Andreas and Rombach, Robin and Oktay, Kaan and M\"uller, Jonas and Ommer, Bj\"orn}, title={Retrieval-Augmented Diffusion Models}, booktitle={NeurIPS}, year={2022}, note={arXiv:2204.11824}}
@inproceedings{das2024larimar, author={Das, Payel and others}, title={Larimar: Large Language Models with Episodic Memory Control}, booktitle={ICML}, year={2024}, series={PMLR 235}, note={arXiv:2403.11901}}
@inproceedings{berges2025memorylayers, title={Memory Layers at Scale}, booktitle={ICML}, year={2025}, note={arXiv:2412.09764; author list not verified from primary}}
@article{laguna2026munkey, author={Laguna, Sonia and da Silva Gon\c{c}alves, Jorge and Vandenhirtz, Moritz and Ryser, Alain and Cannistraci, Irene and Vogt, Julia E.}, title={Rethinking Machine Unlearning: Models Designed to Forget via Key Deletion}, journal={arXiv:2603.15033}, year={2026}}
@inproceedings{montero2021disentgen, author={Montero, Milton Llera and Ludwig, Casimir and Costa, Rui Ponte and Malhotra, Gaurav and Bowers, Jeff}, title={The Role of Disentanglement in Generalisation}, booktitle={ICLR}, year={2021}}
@inproceedings{schott2022notgeneralize, author={Schott, Lukas and von K\"ugelgen, Julius and Tr\"auble, Frederik and Gehler, Peter and Russell, Chris and Bethge, Matthias and Sch\"olkopf, Bernhard and Locatello, Francesco and Brendel, Wieland}, title={Visual Representation Learning Does Not Generalize Strongly Within the Same Domain}, booktitle={ICLR}, year={2022}, note={arXiv:2107.08221}}
@inproceedings{johnson2017clevr, author={Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens and Fei-Fei, Li and Lawrence Zitnick, C. and Girshick, Ross}, title={CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning}, booktitle={CVPR}, year={2017}, note={defines CoGenT; arXiv:1612.06890}}
@inproceedings{gondal2019mpi3d, author={Gondal, Muhammad Waleed and others}, title={On the Transfer of Inductive Bias from Simulation to the Real World: A New Disentanglement Dataset}, booktitle={NeurIPS}, year={2019}}
@inproceedings{kim2018factorvae, author={Kim, Hyunjik and Mnih, Andriy}, title={Disentangling by Factorising}, booktitle={ICML}, year={2018}, note={3dshapes/Shapes3D dataset}}
@inproceedings{bear2021physion, author={Bear, Daniel M. and others}, title={Physion: Evaluating Physical Prediction from Vision in Humans and Machines}, booktitle={NeurIPS Datasets and Benchmarks}, year={2021}, note={arXiv:2106.08261}}
@article{bansal2024videophy, author={Bansal, Hritik and others}, title={VideoPhy: Evaluating Physical Commonsense for Video Generation}, journal={arXiv:2406.03520}, year={2024}}
@article{worldmodelbench2025, title={WorldModelBench: Judging Video Generation Models As World Models}, journal={arXiv:2502.20694}, year={2025}}
```

---

## Proposed handover updates (for the Hub)

- **Positioning line that survives adversarial reading:** the contribution's novelty is **not** "compositional generation with an energy model" (owned by Du/COMET/LACE since 2020–21), **not** "concept deletion" (a crowded, broken field), and **not** "editable memory" (RDM/Larimar/Memory Layers). It is **(a) the accept-unseen-valid / refuse-impossible asymmetry from one dynamics-derived criterion with no invalid-example supervision, and (b) an enumerable wells×modes store that makes deletion and provenance structural rather than empirical.** Both are NOT-FOUND in the literature searched.
- **Hard constraint the census imposes:** no single existing dataset supports all five capabilities. A venue plan must be written before any capability-1–5 table is designed.
- **One unread paper is load-bearing:** LACE (arXiv:2110.10873) — latent EBM on a frozen generator, zero-shot unseen attribute combinations. Read it in full before any novelty sentence is written.
- **Sampler discipline:** *Reduce, Reuse, Recycle* (ICML 2023) already established that failures of compositional sampling are the **sampler's** fault. Any Langevin-rollout composition result must declare its sampler and address Metropolis correction, or a reviewer from that line will attribute the result to sampling luck.
