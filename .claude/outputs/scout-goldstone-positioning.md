# scout-goldstone-positioning — web-scout report
Task + acceptance criterion: verdict each of V2's 5 claimed differentiators vs Welling et al. (arXiv:2605.14685) as OPEN / PARTIALLY TAKEN / TAKEN, with evidence + bibtex-ready related-work map.
Status: done

## Answer first
Welling et al. (Iqbal, Keller, Song, Miyato, Welling, arXiv:2605.14685, 14 May 2026) is a **kinematic / initialization-theory** paper: equivariant *feedforward and RNN layers* under exact continuous symmetry (U(1), O(k)), analysed with **mean-field / path-integral** tools, whose "Goldstone mode" is the *phase of a two-input covariance* that is **exactly conserved across depth** (φ^{l+1}=φ^l). It contains **no Hamiltonian, no energy, no symplectic structure, no dissipation/friction, no EFT/coset machinery, and no pseudo-Goldstone/mass analysis** (verified via three independent full-text passes). So V2's differentiators **(i) energy-space account, (ii) friction–Goldstone interplay, (iv) EFT/coset organization, and (v) causal-bound-c are all OPEN against Welling.** The real threat is a sibling paper the task didn't name: **Mo (2026, arXiv:2605.03338), "Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks"** — it already proves ≥ *dim(G/H)* zero-Lyapunov modes AND runs controlled explicit-symmetry-breaking experiments where a **"pseudo-gap" predicts finite memory lifetime**. That **PARTIALLY TAKES differentiator (iii)** (multiplicity + graceful degradation). **V2 must differentiate from Mo, not just Welling.**

---

## (1) Welling paper précis — arXiv:2605.14685 (28pp; cs.LG + cond-mat.stat-mech + AI)
**Authors:** Nabil Iqbal, T. Anderson Keller, Yue Song, Takeru Miyato, Max Welling (U. Amsterdam). Code: github.com/nabiliqbal/ssb-goldstone-deep-info-prop.

**Setting / what "SSB" means operationally.** Internal layers are built **equivariant** under a continuous group G acting on the hidden state: ρ_g f^l(x)=f^l(ρ_g x) (their Eq. 2). Two groups:
- **U(1) (abelian):** complex features, z_i→e^{iα}z_i; phase-preserving nonlinearity φ(z)=tanh(|z|)/|z|·z.
- **O(k) (non-abelian):** ℝ^N split into N/k blocks of k-vectors, each rotated by O(k); weights W_{(αa;βb)}=w_{αβ}δ_{ab} (isotropic within blocks, mixing between).

"SSB" = the *propagated representation stays ordered* (order parameter nonzero) rather than collapsing. Order parameter c^l=(1/N)𝔼[Σ_j φ(z_j^l)φ†(z_j^l)]; large-N recursion c^{l+1}=2∫₀^∞ du·u·e^{-u²}[φ(u σ_W√c^l)]² (their Eq. 8; near-transition expansion c^{l+1}=σ_W²c^l+O((c^l)²), Eq. 9). **Transition at σ_W=1** (weight-init variance): σ_W<1 → symmetric/disordered phase (c→0, info dies); σ_W>1 → **SSB phase** (c finite).

**Main result (the "Goldstone mode").** For two inputs, decompose the covariance into magnitude Δ^l and **phase φ^l**. Magnitude decays slowly, Δ^l∼exp(−l/ξ_Δ). **Phase is exactly conserved: φ^{l+1}=φ^l (their Eq. 11)** — the protected channel. A protected Jacobian component d^L (their Eq. 7) stays O(1) through the SSB phase → no gradient collapse; equivariant nets avoid rank collapse via a representation floor of dimension k. **They explicitly flag the departure from physics:** *"In physics, time-evolution of Goldstone modes is highly constrained and modes generally evolve slowly. In our setting … the mode turns out to be completely independent of time."* (§2.3). Physics dispersions ω=|k| (gapless, Eq. 17) vs ω=√(k²+r) (gapped, Eq. 16) appear **only as Appendix-A review**, never derived for the network. Appendix C: large-N path-integral/mean-field derivation.

**Experiments / metrics.**
- **100-layer MLP, Fashion-MNIST, N=64:** U(1)/O(4) equivariant nets train precisely when σ_W>1; non-equivariant fail at all σ_W; *"larger groups allow more information to be stored in the Goldstone modes, and … correspondingly have stronger large-depth performance"* (Fig. 4 caption).
- **Jacobian analysis:** generic nets rank-collapse; equivariant nets keep healthy rank.
- **Variable-delay copy task** (10 tokens, delay T_max∈{25,100,200}; CE loss + success over 10 seeds): U(1)-RNN N=32 (64 real dims) beats 32/64-unit baselines; GRU-variant solves T_max=200 where others fail.
- **Permuted-seqMNIST:** O(24)-equivariant RNN → **96%**, matching IndRNN/coRNN/LEM without task-specific design.
- Preliminary: 2D conv-RNN develops long-lived **vortices** on the copy task ("precise role remains unclear").

**Stated future work (our collision surface).** (1) **Topological defects** (vortices/domain walls) as a parallel propagation channel — "relatively unexplored in a deep learning context"; (2) **"compatibility with gated / other long-sequence architectures … further exploration of such joint models to future work"**; (3) speculation that conv/transformer symmetries give stronger SSB. **Nothing on Hamiltonian/energy/dissipation/EFT/causal speed.**

---

## (2) Differentiator verdict table

| # | V2 claimed differentiator | Verdict vs **Welling** | Verdict vs **field** | Evidence / who threatens |
|---|---|---|---|---|
| (i) | **Hamiltonian/energy-space account** of Goldstone memory (theirs = equivariant FF/RNN layers) | **OPEN** | **OPEN** | Targeted full-text pass: zero mentions of Hamiltonian, energy conservation, symplectic, phase space, momentum, EBM, Langevin (each explicitly "not found"). Their formalism = mean-field GP/path-integral over *layer index*, not phase-space flow. No HNN paper found that frames Goldstone modes as flat directions of a learned V_θ in a symplectic latent. |
| (ii) | **Dissipation interplay** — "friction kills Goldstone momentum but cannot erase Goldstone displacement" | **OPEN** | **ML-OPEN; physics antecedent exists** | Welling: zero dissipation/friction/damping (verified). Physics: Minami & Hidaka, PRE 97 (2018) 012130 / arXiv:1509.05042 — in dissipative (Fokker-Planck) systems type-A NG modes become **diffusive** instead of propagating. No ML paper makes the friction-vs-Goldstone claim. Cite Minami-Hidaka to look literate; our claim is the *symplectic-latent γp-damping* statement (displacement along flat direction survives; conjugate momentum dies), distinct from their diffusive-dispersion result. |
| (iii) | **Multiplicity / pseudo-Goldstone engineering** — dim(G/H) as designed channel count + graceful degradation with half-life ∝ 1/mass² | **PARTIALLY TAKEN (informally)** | **PARTIALLY TAKEN — by Mo 2026** | Welling has the multiplicity *intuition* only: "larger groups allow more information to be stored" (Fig. 4) — no dim(G/H) formula, no pseudo-Goldstone, no mass/lifetime (verified absent). **Mo 2026 (arXiv:2605.03338): Theorem 1 proves ≥ dim(G/H) zero Lyapunov exponents** tangent to the group orbit for C¹ G-equivariant flows (tested on S¹, T^q, SO(n)/SO(n−1), U(m)/U(m−1)); **and** controlled explicit-breaking experiments where *"this pseudo-gap predicts finite memory lifetime"* — log-lifetime↔pseudo-gap correlation **0.9999999886** (their Fig. 4). That is our "graceful degradation with predictable retention," already published. **Our defensible residue:** (a) *constructive allocation* — choosing G/H to budget channel count by design (Mo *analyzes* a given symmetry, does not *engineer* capacity; verified "No" on design-knob question); (b) the **Hamiltonian curvature mass-law** — retention ∝ 1/ω² with ω²=eig(M⁻¹∂²V_θ), a potential-curvature statement Mo's dissipative-flow Lyapunov framework cannot make (Mo: no Hamiltonian/symplectic/energy/friction — verified). |
| (iv) | **EFT organization of corrections** (HEFT/SMEFT parameterizations, coset / nonlinear-σ-model layers) | **OPEN** | **OPEN — strongest white space** | Welling: no EFT, power counting, Wilsonian RG, coset, or NLSM anywhere (verified; the only expansion is the Eq. 9 Taylor step). Roberts-Yaida-Hanin "EFT of deep learning" (arXiv:2106.10165; + "Structures of NN Effective Theories," arXiv:2305.02334) is a **1/width expansion of initialization statistics** — disjoint from symmetry-breaking memory. No ML instantiation of coset-manifold latent dynamics found. **Cleanest novelty; exactly the Manchester-colleague surface.** |
| (v) | **Causal bound c** on Goldstone signal propagation | **OPEN** | **OPEN (weakest ML motivation)** | Welling's network Goldstone is *time-independent* (φ^{l+1}=φ^l) — they have **no propagation speed at all**; only a decay scale ξ_Δ, no dispersion/velocity derived (verified). Lieb-Robinson bounds are quantum-many-body; no RNN/latent-dynamics instantiation found. Novel but decorative unless tied to an ML-measurable payoff (P1 guard-rail) — e.g., bounded per-step influence radius → provable no-blowup. Otherwise demote to a remark. |

**Net:** 4/5 OPEN against the named target. The month-of-theory risk is concentrated in **(iii)** and comes from **Mo, not Welling**. Reposition (iii) as *"designed allocation + Hamiltonian curvature mass-law"*; lead V2's novelty with **(iv) EFT-of-memory** and **(i) energy-space framing**.

---

## (3) Related-work map (by theme; 1-line relevance each)

**A. SSB / Goldstone modes in deep learning (direct lineage — cite all):**
- **Iqbal, Keller, Song, Miyato, Welling 2026** (arXiv:2605.14685) — the target; equivariant FF/RNN, exactly-conserved covariance phase, σ_W=1 transition.
- **Mo 2026** (arXiv:2605.03338) — dim(G/H) zero-Lyapunov theorem + pseudo-gap→lifetime; **closest prior art to (iii); must cite & distinguish.**
- **Iqbal & Welling 2025** ("Topological defects propagate information in deep neural networks," NeurIPS 2025 AI4Science workshop) — broken ℤ₂ → domain walls carry information; the discrete-symmetry sibling.
- **Löwe et al. 2022/2024** (complex-valued autoencoders / rotating features) — phase-as-information, ancestral to the U(1) construction.
- **Miyato et al. 2024** (Kuramoto oscillatory neurons); **Keller et al. 2023** ("Traveling waves encode the recent past" — source of the pmnist comparison); **Liboni et al. 2023** (traveling waves in RNNs).

**B. Mean-field signal propagation across depth (Welling's framing lineage):**
- Poole et al. 2016; **Schoenholz et al. 2017** ("Deep information propagation"); Yang & Schoenholz 2017 (edge of chaos); Xiao et al. 2018 (dynamical isometry); Chen et al. 2018 (RNN gating & signal propagation) — order-parameter recursions across depth; our energy-drift-across-time is the Hamiltonian analog.

**C. EFT / field theory of neural networks (context for differentiator iv):**
- **Roberts, Yaida, Hanin 2022** (Principles of Deep Learning Theory, arXiv:2106.10165); Banta et al. (arXiv:2305.02334) — EFT = 1/width init statistics, **not** memory/coset dynamics; cite to fence off our different expansion (around the broken-symmetry vacuum).
- NLSM/coset (CCWZ) = textbook HEP with no ML-latent instantiation found.

**D. Symplectic / Hamiltonian / stable RNNs (CHLU's home turf; adjacent must-knows):**
- **Chen, Zhang, Arjovsky, Bottou 2020** ("Symplectic Recurrent Neural Networks," ICLR 2020) — learned H + symplectic integration + multi-step training; nearest architectural cousin (CHLU adds relativistic governor, wake-sleep, Goldstone framing).
- **Erichson, Azencot, Queiruga, Hodgkinson, Mahoney 2021** ("Lipschitz Recurrent Neural Networks," ICLR 2021) — stability via linear + Lipschitz-nonlinearity split; the dynamical-systems-stability baseline.
- Greydanus et al. 2019 (HNN); Cranmer et al. 2020 (LNN); coRNN/LEM (Rusch et al.) as pmnist-class stable RNNs.
- **LyTimeT** (Kong et al., arXiv:2510.19716; already cited in the CHLU paper) — TimeSformer autoencoder + Lyapunov-stability regularizer for state-variable discovery; the "stability as a *learned regularizer*" contrast to our "stability by construction."

**E. Symmetry ↔ conservation in ML (Noether angle):**
- **Alet, Doblar, Zhou, Tenenbaum, Kawaguchi, Finn 2021** ("Noether Networks," NeurIPS, arXiv:2112.03321) — *meta-learns* conserved quantities as prediction-time regularizers; contrast: CHLU builds conservation in via the symplectic form.
- **Kunin et al. 2020** ("Neural Mechanics," arXiv:2012.04728) — symmetry/conservation laws of *SGD training dynamics* (different object; familiar vocabulary).
- Cohen & Welling 2016 (G-CNNs); Kaba & Ravanbakhsh 2023 (symmetry breaking & equivariant NNs) — equivariance backbone.

**F. Continuous attractors (the neuroscience shadow — anticipate this reviewer):**
- Continuous-attractor networks (head direction, grid cells, working memory) hold a **marginally-stable manifold = flat direction ↔ Goldstone mode of the encoded symmetry** (e.g., "Continuous attractors for dynamic memories," eLife 2021; "Symmetries and Continuous Attractors in Disordered Neural Circuits," bioRxiv 2025; Mo 2026 cites this line). A reviewer *will* say "this is a continuous attractor." Pre-empt: CHLU's flat directions live in a **symplectic phase space with momentum** (memory as displacement + conserved dynamics), not a dissipative rate-manifold fixed point.

**G. Certified conservation in latent models (adjacent, post-CHLU — watch):**
- **Wang 2026** (arXiv:2606.24945, "When Do Conservation Laws Survive Learned Representations? Certified Horizons for Latent World Models") — "shell-horizon certificates" bounding rollout steps a *decoded* invariant survives; finds hard symplectic structure extends horizons in known coordinates **but fails across learned coordinate transforms**. V3/ICLR-relevant (certified-horizon claims); "shell" vocabulary overlap is coincidental.

---

## (4) Risks — closest prior art that could sink a review
1. **Mo 2026 (arXiv:2605.03338) — HIGHEST.** Leading V2 with "dim(G/H) protected channels degrade gracefully with an induced gap" invites *"already proved, incl. the gap→lifetime law."* **Mitigation:** cite prominently; claim what Mo cannot: (a) Hamiltonian **curvature** mass-law (1/ω², ω²=eig(M⁻¹∂²V) — potential curvature, not a Lyapunov exponent of a dissipative flow), (b) constructive channel *allocation* as a design interface, (c) friction interplay (ii), (d) EFT organization (iv). Mo has none of these (verified: no Hamiltonian/symplectic/energy/dissipation-as-lever; no design-knob discussion).
2. **Welling's stated future work: "compatibility with gated/other long-sequence architectures."** If V2 reads as "Goldstone + a recurrent architecture," it looks like the obvious execution of their future work. **Mitigation:** thesis = *different mechanism*: their Goldstone is time-independent by construction; ours **propagates on a symplectic flow with finite causal speed and survives/decays under controllable friction**.
3. **Continuous-attractor neuroscience (map F).** "Flat direction = memory" is decades old there. **Mitigation:** novelty is symplectic/momentum + mass-law + EFT corrections, not flat-directions-as-memory per se.
4. **EFT over-claim vs Roberts-Yaida.** **Mitigation:** name ours an EFT of *coset latent memory dynamics* (expansion around the broken vacuum), explicitly disjoint from the 1/width init-statistics EFT.
5. **Differentiator (v)** is the weakest-motivated for ML reviewers — attach a measurable benefit or demote to a remark (P1 guard-rail).

---

## (5) Bibtex
```bibtex
@article{iqbal2026ssb,
  title   = {Spontaneous symmetry breaking and Goldstone modes for deep information propagation},
  author  = {Iqbal, Nabil and Keller, T. Anderson and Song, Yue and Miyato, Takeru and Welling, Max},
  journal = {arXiv preprint arXiv:2605.14685},
  year    = {2026}
}
@article{mo2026lyapunov,
  title   = {Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks},
  author  = {Mo, Hanson Hanxuan},
  journal = {arXiv preprint arXiv:2605.03338},
  year    = {2026}
}
@inproceedings{iqbal2025topological,
  title     = {Topological defects propagate information in deep neural networks},
  author    = {Iqbal, Nabil and Welling, Max},
  booktitle = {NeurIPS 2025 AI for Science Workshop},
  year      = {2025}
}
@inproceedings{chen2020symplectic,
  title     = {Symplectic Recurrent Neural Networks},
  author    = {Chen, Zhengdao and Zhang, Jianyu and Arjovsky, Mart{\'\i}n and Bottou, L{\'e}on},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020}
}
@inproceedings{erichson2021lipschitz,
  title     = {Lipschitz Recurrent Neural Networks},
  author    = {Erichson, N. Benjamin and Azencot, Omri and Queiruga, Alejandro and Hodgkinson, Liam and Mahoney, Michael W.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}
@inproceedings{alet2021noether,
  title     = {Noether Networks: Meta-Learning Useful Conserved Quantities},
  author    = {Alet, Ferran and Doblar, Dylan and Zhou, Allan and Tenenbaum, Joshua B. and Kawaguchi, Kenji and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021}
}
@article{kunin2020neural,
  title   = {Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics},
  author  = {Kunin, Daniel and Sagastuy-Brena, Javier and Ganguli, Surya and Yamins, Daniel L.K. and Tanaka, Hidenori},
  journal = {arXiv preprint arXiv:2012.04728},
  year    = {2020}
}
@book{roberts2022principles,
  title     = {The Principles of Deep Learning Theory: An Effective Theory Approach to Understanding Neural Networks},
  author    = {Roberts, Daniel A. and Yaida, Sho and Hanin, Boris},
  publisher = {Cambridge University Press},
  year      = {2022},
  note      = {arXiv:2106.10165}
}
@article{minami2018nambu,
  title   = {Spontaneous symmetry breaking and Nambu-Goldstone modes in dissipative systems},
  author  = {Minami, Yuki and Hidaka, Yoshimasa},
  journal = {Physical Review E},
  volume  = {97}, number = {1}, pages = {012130},
  year    = {2018},
  note    = {arXiv:1509.05042}
}
@article{wang2026conservation,
  title   = {When Do Conservation Laws Survive Learned Representations? Certified Horizons for Latent World Models},
  author  = {Wang, Hongbo},
  journal = {arXiv preprint arXiv:2606.24945},
  year    = {2026}
}
```
(LyTimeT — Kong et al., arXiv:2510.19716, 2025 — already in the CHLU paper's bibliography; not re-derived here.)

---

## How I verified
- **Welling full text read via ar5iv HTML** (`ar5iv.labs.arxiv.org/html/2605.14685`). The local `docs/2605.14685v1.pdf` **could not be rendered** in the original session (no poppler/pdftoppm on this machine; `Read` on the PDF errored); `Grep` found 5 "Goldstone" hits in the raw PDF but no usable extraction. Précis is therefore **single-source (ar5iv)**, cross-checked against the arXiv abstract page, alphaXiv entry, and the existence/contents of the GitHub repo.
- **Three independent targeted passes** over the full ar5iv text for {Hamiltonian, energy conservation, symplectic, phase space, momentum, dissipation, friction, damping, energy-based models, Langevin, EFT, power counting, coset, NLSM, pseudo-Goldstone, dim(G/H), decay half-lives} — every term returned **explicitly absent**, except the Appendix-A dispersion review (ω=|k| vs ω=√(k²+r)) and the informal "larger groups store more" remark. High confidence in the absence claims.
- **Mo 2026** read via arXiv HTML: Theorem 1 statement, the "No" on design-knob usage, pseudo-gap experiments (log-lifetime↔gap corr 0.9999999886), equivariance error 6.97×10⁻¹⁶, path-integration RMSE 0.041±0.002 @ horizon 256 vs >1.45 baselines — quoted from the paper.
- **Semantic Scholar citations API** on 2605.14685 → **0 citing papers** (paper is <2 months old). Nobody has built on Welling yet; field open, exposure symmetric.
- Sweep searches run for: SSB-in-DL, Goldstone+NN memory, pseudo-Goldstone/NLSM in ML, Noether-in-ML, EFT-of-DL, Lieb-Robinson-in-NN (none found in ML), symplectic/Lipschitz RNNs, LyTimeT, continuous attractors, Iqbal-Welling defects paper.

## Open questions / follow-ups / risks
- **Local-PDF extraction gap:** if the Hub needs exact v1-PDF page/equation refs (vs ar5iv numbering), an agent with Bash should run the task file's `pypdf` route (`/Users/user/opt/miniconda3/bin/python`), or install poppler. Ar5iv content is complete but eq numbers could differ from the PDF.
- **Recommend a dedicated deep-read of Mo 2026 (arXiv:2605.03338) before V2 theory starts** — the single highest-value verification: confirm its pseudo-gap is purely a Lyapunov-exponent object of a dissipative flow, so our curvature/1-over-ω² Hamiltonian law is *provably* distinct, and mine its experimental protocol (their controlled-breaking design is directly reusable for our half-life measurements).
- OpenReview in-review ICLR-2027 submissions are not indexed/searchable — residual scoop risk there unmeasurable.
- Welling et al. have an obvious sequel path (defects/vortices + "joint models with gated architectures"); assume the Amsterdam group is actively extending — V2's speed matters.

## Proposed handover updates (for the Hub)
- **§8 / roadmap V2 — positioning target is now *two* papers.** Welling (2605.14685) is **contrast-clean**: kinematic/equivariance theory with no energy, no dissipation, no EFT, no propagation speed — differentiators (i), (ii), (iv), (v) all **OPEN** vs it. **The scoop risk is Mo (2605.03338)**: ≥dim(G/H) zero-Lyapunov modes (Thm 1) + explicit-breaking "pseudo-gap predicts finite memory lifetime" (corr ≈1.0). Differentiator (iii) is **PARTIALLY TAKEN**.
- **Reframe (iii):** from "pseudo-Goldstone graceful degradation" (generic version now published) to **"dim(G/H) as a *constructive* channel-allocation design knob + Hamiltonian curvature mass-law: retention ∝ 1/ω², ω² = eig(M⁻¹∂²V_θ)"** — ties directly into Thread-5's M-as-budget-allocator; Mo's dissipative-Lyapunov framework cannot express it.
- **Lead V2's novelty with (iv) EFT-of-memory/coset layers + (i) energy-space account** — cleanest white space (verified absent in Welling, Mo, and Roberts-Yaida), lowest scoop risk, and exactly the Manchester-colleague surface.
- **Must-cite/distinguish list for V2:** Mo 2026; Iqbal-Welling 2025 (ℤ₂ defects, NeurIPS-W); Chen 2020 (SymplecticRNN); Erichson 2021 (LipschitzRNN); Alet 2021 (Noether Nets — learned-as-loss vs our by-construction); Kunin 2020 (Neural Mechanics — SGD-dynamics conservation, different object); Minami-Hidaka 2018 (dissipative NG modes → diffusive; the physics antecedent that keeps (ii) honest); Wang 2026 (certified horizons — V3-relevant; notes hard symplectic structure fails across learned coordinate transforms, a caveat for our own claims; "shell" vocab coincidental).
- **Pre-empt the continuous-attractor reviewer** (neuro): CHLU's flat directions are symplectic-with-momentum, not dissipative rate-manifold fixed points.
- **P1 guard-rail flag:** differentiator (v) causal-c needs an attached ML-measurable payoff (e.g., bounded per-step influence radius → no-blowup guarantee) or should be demoted to a remark in V2.
- **§9/workflow note:** this spoke ran without Write/Bash in its first pass (tool-availability gap) and could not extract the local PDF (no poppler); output file was written on a follow-up instruction. Consider noting poppler/pypdf as an environment prerequisite for PDF-reading scout tasks.
