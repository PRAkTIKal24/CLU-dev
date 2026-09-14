# scout-relativistic-samplers — web-scout report
Task + acceptance: novelty/prior-art check on (i) the additive-Gaussian-kick "relativistic Gibbs no-go" (CM-17), (ii) the latent-mass / MJ-as-scale-mixture thermostat (F2), (iii) whether relativistic SGHMC (Lu et al. 2017) carries the bug — verify by reading sources, quote update rules verbatim.
Status: **done** (read primary sources for the two load-bearing items; one item — Monomial-Gamma's exact *wording* of the no-go — flagged single-sourced-via-WebFetch, verify the quote before citing it as "they state the theorem").

---

## BOTTOM LINE (first five lines)
1. **The no-go is essentially a KNOWN corollary, not new.** "An additive-Gaussian momentum kick with fixed covariance cannot leave a *non-Gaussian* momentum marginal invariant" is already the *stated motivation* of the **Monomial-Gamma sampler** literature (Zhang et al., ICML 2017 / NIPS 2016), which adopts non-quadratic kinetic energies and **explicitly avoids additive Gaussian noise by exact momentum refresh**. CHLU's characteristic-function proof is a clean, rigorous, `V`- and `γ`-independent *statement* of this folklore — **cite it as a corollary/sharpening, do not claim it.** The genuinely new, specific piece is the **`d·Θ` dimension-amplification with closed forms** — I found no prior for that.
2. **The latent-mass thermostat is KNOWN mathematics, novel *packaging*.** "Relativistic momentum ∝ e^{−c√(pᵀM⁻¹p+m²c²)} is a Gaussian scale-mixture with inverse-Gaussian mixing" **is** the generalized-hyperbolic / **Normal-Inverse-Gaussian representation (Barndorff-Nielsen 1977/1997)** and the ½-stable-subordinator representation of the relativistic Schrödinger operator √(−Δ+m²) (Carmona–Masters–Simon 1990; Ryznar 2002). The correct *continuous* relativistic thermostat is **Dunkel–Hänggi 2009**. What I could **not** find is anyone using the InvGauss-mixture as a **momentum-persistence-preserving O-step** (an exact underdamped thermostat) — so F2's *engineering form* is plausibly novel but rests entirely on cited prior math + a known alternative fix (exact refresh).
3. **Does relativistic SGHMC have the bug? — YES for the unadjusted stochastic-gradient variant, NO for their main (MH-adjusted HMC) method.** Verbatim: rel-HMC is leapfrog **+ Metropolis** `min(1, exp(−H_L+H_0))` ⇒ **exact** (a no-go escape hatch). rel-SGHMC's update is `p_{t+1} ← p_t − ε∇Ũ − εD M⁻¹(p_t)p_t + N(0, ε(2D−εB̂_t))` — an additive Gaussian kick with **state-independent covariance** as the last operation ⇒ by CHLU's Lemma-9a its invariant momentum marginal is Gaussian-smoothed, **not** the symmetric-hyperbolic target. Two honesty caveats: their friction damps the **velocity** `M⁻¹(p)p = ∇K` (the *correct* FDT structure — CHLU's code damps `p`, the wrong one), so their **continuous limit is correct**; and they never claim the unadjusted chain is exact. So the bug is the standard `O(ε)` unadjusted-SG-MCMC bias, which they do not sharpen.
4. **Strongest honest claim & venue:** an **F5 appendix / methods-note correction**, framed as *"we give a rigorous `V`-/`γ`-independent no-go for additive-noise thermostats on non-quadratic kinetic energies (corollary of the exact-refresh rationale in non-Gaussian-kinetic HMC), quantify its previously-unremarked `d·Θ` dimension amplification, and give a momentum-preserving InvGauss-mixture repair (the sampler realization of the known generalized-hyperbolic / relativistic-Brownian representation)."* **Not** a standalone novelty paper for the no-go or the representation.

---

## Evidence

### Q3 — Relativistic SGHMC (the ⭐ target). VERIFIED from the paper.
- **Lu, Perrone, Hasenclever, Teh, Vollmer (2017), "Relativistic Monte Carlo", AISTATS, PMLR v54; arXiv:1609.04388.** Read via ar5iv. Verbatim:
  - Kinetic energy `K(p) = m c² (pᵀp/(m²c²) + 1)^{1/2}`; relativistic mass `M(p) = m (pᵀp/(m²c²)+1)^{1/2}`; velocity `M⁻¹(p)‖p‖ ≤ c` (asymptotes to c). Momentum marginal `∝ e^{−K(p)}`, "a multivariate generalisation of the **symmetric hyperbolic distribution**." (This is the same object as CHLU's `T(p)`; symmetric-hyperbolic ∈ generalized-hyperbolic family = a Gaussian variance-mean mixture with GIG mixing — i.e. **the paper already names the object F2 re-derives**.)
  - **Rel-HMC (MH-adjusted, their main method):** leapfrog `p_{t+½}←p_t−(ε/2)∇U; θ←θ+εM⁻¹(p_{t+½})p_{t+½}; p_{t+1}←p_{t+½}−(ε/2)∇U`, then **accept `min(1, exp(−H(θ_L,p_L)+H(θ_0,p_0)))`.** ⇒ **exact**; this is exactly the Metropolis escape-hatch of Lemma-9a. **No bug.**
  - **Rel-SGHMC (unadjusted):** `p_{t+1} ← p_t − ε_t ∇Ũ(θ_t) − ε_t D M⁻¹(p_t)p_t + N(0, ε_t(2D − ε_t B̂_t))`. The noise covariance `ε_t(2D−ε_tB̂_t)` is **not a function of the pre-noise momentum** (D is a chosen friction matrix; B̂_t is a stochastic-gradient-noise estimate in θ). ⇒ additive Gaussian kick, fixed covariance, last operation ⇒ **Lemma-9a bites** ⇒ invariant momentum marginal is Gaussian-smoothed, cannot equal the symmetric-hyperbolic target. **The paper states leapfrog "leaves H approximately invariant"; it gives no discussion of a momentum-marginal bias and no MH for the SG variant** (MH is infeasible with minibatch gradients).
  - **Caveat that weakens any "their sampler is broken" claim:** the friction `D M⁻¹(p)p = D∇K(p)` damps the **velocity**, which is the FDT-correct structure (CHLU's `xy-lattice §5v`: correct Langevin damps `∇_pT`; CHLU's code wrongly damps `p`). Hence rel-SGHMC's **continuous-time SDE preserves the exact relativistic Gibbs measure**; only the fixed-covariance discretization is biased. That bias is the ordinary `O(ε)` unadjusted-SG-MCMC bias they implicitly accept.
- **No follow-up found that notes or repairs a relativistic-SGHMC momentum-marginal defect specifically.** Adjacent: recent Wasserstein-bias analyses of stochastic-gradient kinetic Langevin confirm "time discretisation perturbs the invariant distribution" generically (e.g. arXiv:2604.24632), but none single out the non-quadratic-K impossibility or a `d·Θ` law.

### Q1 — is the no-go known? Corollary of non-Gaussian-kinetic HMC.
- **Zhang, Chen, Gan, Henao, Carin (2017), "Stochastic Gradient Monomial Gamma Sampler", ICML 2017; arXiv:1706.01498** (code: github.com/dreasysnail/SGMGT). Monomial kinetic `K(p)=|p|^γ/γ` ⇒ non-Gaussian ("monomial-gamma") momentum marginal. **Momentum handled by exact refresh `p ~ π(p)`, not additive Gaussian noise** — WebFetch-of-PDF reports they motivate this precisely because "additive Gaussian perturbations to non-Gaussian marginals create a mismatch between intended and actual stationary distributions." ⚠ **Single-sourced via WebFetch summary — verify the exact sentence in the PDF before quoting it as "they state the no-go."** Predecessor: Zhang, Wang, Chen, Henao, Fan, Carin (2016), "Towards Unifying HMC and Slice Sampling", NIPS 2016; arXiv:1602.07800 (introduces the Monomial-Gamma family). **⇒ The phenomenon CHLU's Lemma-9a formalizes is recognized in this line; CHLU's contribution is a rigorous general proof (char.-function, `V`/`γ`-independent) + the `d·Θ` amplification.** The Monomial-Gamma fix is CHLU's **F3 (exact refresh)**, *not* F2 (latent-mass).
- General SG-MCMC context (arXiv:1706.01498 and the "Complete Recipe", Ma–Chen–Fox 2015): momentum/thermostat marginals are "typically assumed Gaussian"; going non-Gaussian is known to need care. No one states the `d·Θ` scaling.

### Q2 — is "MJ = Gaussian scale mixture ⇒ randomized-inertia thermostat" known?
- **Representation is classical.** Barndorff-Nielsen (1997), "Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling", *Scand. J. Statist.* 24:1–13 (and the 1977 generalized-hyperbolic paper): **NIG = normal variance-mean mixture with an inverse-Gaussian mixing law**, ≡ Brownian motion subordinated by an IG subordinator. CHLU's `T(p)`/symmetric-hyperbolic momentum is exactly this family — so `p|s ~ N(0,M/2s)`, `s|p ~ InvGauss(...)` is the **known** generalized-hyperbolic representation, not new.
- **Physics side, subordinator form.** `e^{−β√A}` via the ½-stable subordinator is the relativistic Schrödinger operator √(−Δ+m²): Carmona, Masters, Simon (1990), *J. Funct. Anal.* 91:117–142; Ryznar (2002), "Estimates of Green function for relativistic α-stable process", *Potential Analysis* 17:1–23; the relativistic-α-stable process `Z_t = B_{2S_{t,m}}` (subordinated Brownian motion). This is exactly F2's Bernstein/subordinator identity.
- **Correct relativistic thermostat already exists (continuous).** **Dunkel & Hänggi (2009), "Relativistic Brownian motion", *Physics Reports* 471:1–73; arXiv:0812.1996.** The Hänggi–Klimontovich (post-point) relativistic Langevin is the SDE whose stationary law is exactly Jüttner/Maxwell–Jüttner (state-dependent diffusion). **This is the citable source for "the correct thermostat" — the F2 O-step is a discrete, momentum-persistence-preserving realization of it.**
- **What I did NOT find:** an ML/statistics paper that samples a non-Gaussian (relativistic/hyperbolic) momentum marginal by an **auxiliary InvGauss scale variable inside a persistent (underdamped) O-step** — i.e. F2's specific "keep momentum autocorrelation, randomize the inertia per step" construction. The known ML fix is exact refresh (Monomial-Gamma), which kills persistence (CHLU's F3). **F2's novelty is this narrow packaging, and it must be presented on top of Barndorff-Nielsen + Dunkel–Hänggi + the exact-refresh baseline, not as a fresh discovery.**

### Q3(broader) / Q4 — adjacent.
- **Non-quadratic kinetic energies in ML:** Riemannian-manifold HMC (Girolami & Calderhead 2011, JRSS-B) uses position-dependent mass `M(θ)` → non-separable H, **implicit** generalized leapfrog (not an additive-Gaussian O-step, so orthogonal to the no-go). Magnetic HMC, monomial-gamma (above), heavy-tailed kinetics all live here; none give the additive-noise impossibility as a named theorem.
- **Velocity/causal cap as a *memory/stability* guarantee (vs. sampler-efficiency):** no direct hit. The relativistic-sampler literature (Lu 2017) frames the velocity cap purely as a **mixing/robustness-to-stepsize** device. Lieb–Robinson / effective-light-cone bounds (e.g. Nature Comms 2024; arXiv:2305.08334) give causal information-propagation limits in *quantum* dynamics — a conceptual cousin to CHLU's R7 causal-memory-floor but not an ML erasure bound. **CHLU's "light-cone bound on erasure as an adversary-proof memory guarantee" appears genuinely unoccupied in the ML literature** — worth positioning against Lieb–Robinson as the physics precedent for the *idea*, while claiming the ML instantiation.

---

## Relevance to CHLU — what to claim, what to cite, what to differentiate
- **DOWNGRADE the no-go's novelty in CM-17 framing.** It is a rigorous corollary of an already-motivated fact in non-Gaussian-kinetic HMC (Monomial-Gamma). **Claim:** the sharp `V`/`γ`/damping-law-independent statement + the **`d·Θ` amplification** (the part that actually corrects the program's own "c=5 is benign"). **Cite:** Zhang et al. 2017 (the phenomenon), Lu et al. 2017 (the exact object). This protects us from an easy reviewer "this is just unadjusted-discretization bias."
- **F2 latent-mass thermostat:** present as *"the sampler realization of the relativistic-Brownian / NIG scale-mixture, with momentum persistence preserved."* **Cite** Barndorff-Nielsen 1997, Dunkel–Hänggi 2009, and contrast with the exact-refresh fix (Monomial-Gamma = our F3) to justify why F2 (persistence-preserving) is preferable. Do **not** claim the InvGauss representation.
- **Differentiator that survives:** (i) the `d·Θ` dimension law + Exp-C numbers; (ii) F2 as an underdamped, autocorrelation-preserving O-step (vs. HMC's momentum refresh); (iii) the *diagnosis that CHLU's coded friction damps `p` not `∇K`* — note Lu et al. got the friction structurally right, so our defect is partly a coding choice, not intrinsic to relativistic kinetics. Reviewers will check this; own it.
- **R7 causal-memory-floor:** likely the cleaner novelty axis — no ML prior for "velocity cap ⇒ erasure lower bound." Position vs. Lieb–Robinson (physics precedent) not vs. any sampler.

## Confidence & gaps
- **Verified by reading source:** Lu et al. 2017 update rules + MH (ar5iv, quoted verbatim). Barndorff-Nielsen NIG = normal-IG mixture (multiple sources). Dunkel–Hänggi as the relativistic-Langevin/Jüttner authority (Physics Reports). Relativistic-α-stable subordinator representation (Ryznar/Carmona).
- **Single-sourced / verify next:** the *exact wording* by which Monomial-Gamma (arXiv:1706.01498) states the additive-Gaussian-vs-non-Gaussian-marginal mismatch — I have a WebFetch summary, not the sentence. **Before citing it as "prior art states the no-go," pull the exact paragraph** (§ on why they refresh momentum exactly). This is the hinge of the novelty downgrade.
- **Not found (good-faith absence):** (a) any prior additive-noise impossibility *theorem* with the `d·Θ` law; (b) any InvGauss-mixture **persistent O-step** thermostat; (c) any ML velocity-cap-as-erasure-bound. Absence ≠ nonexistence — searched arXiv/PMLR/Scholar surfaces, not every stats-mechanics preprint.

## Bibtex-ready refs
```bibtex
@inproceedings{lu2017relativistic,
  title={Relativistic Monte Carlo},
  author={Lu, Xiaoyu and Perrone, Valerio and Hasenclever, Leonard and Teh, Yee Whye and Vollmer, Sebastian},
  booktitle={AISTATS}, series={PMLR}, volume={54}, pages={1236--1245}, year={2017},
  note={arXiv:1609.04388}}
@inproceedings{zhang2017sgmgt,
  title={Stochastic Gradient Monomial Gamma Sampler},
  author={Zhang, Yizhe and Chen, Changyou and Gan, Zhe and Henao, Ricardo and Carin, Lawrence},
  booktitle={ICML}, year={2017}, note={arXiv:1706.01498}}
@inproceedings{zhang2016unifying,
  title={Towards Unifying Hamiltonian Monte Carlo and Slice Sampling},
  author={Zhang, Yizhe and Wang, Xiangyu and Chen, Changyou and Henao, Ricardo and Fan, Kai and Carin, Lawrence},
  booktitle={NIPS}, year={2016}, note={arXiv:1602.07800}}
@article{barndorffnielsen1997nig,
  title={Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling},
  author={Barndorff-Nielsen, Ole E.},
  journal={Scandinavian Journal of Statistics}, volume={24}, number={1}, pages={1--13}, year={1997}}
@article{dunkel2009relativistic,
  title={Relativistic Brownian motion},
  author={Dunkel, J{\"o}rn and H{\"a}nggi, Peter},
  journal={Physics Reports}, volume={471}, number={1}, pages={1--73}, year={2009},
  note={arXiv:0812.1996}}
@article{carmona1990relativistic,
  title={Relativistic Schr{\"o}dinger operators: Asymptotic behavior of the eigenfunctions},
  author={Carmona, Ren{\'e} and Masters, Wu Chao and Simon, Barry},
  journal={Journal of Functional Analysis}, volume={91}, number={1}, pages={117--142}, year={1990}}
@article{ryznar2002relativistic,
  title={Estimates of Green function for relativistic $\alpha$-stable process},
  author={Ryznar, Micha{\l}},
  journal={Potential Analysis}, volume={17}, number={1}, pages={1--23}, year={2002}}
@article{girolami2011riemann,
  title={Riemann manifold Langevin and Hamiltonian Monte Carlo methods},
  author={Girolami, Mark and Calderhead, Ben},
  journal={JRSS-B}, volume={73}, number={2}, pages={123--214}, year={2011}}
```

## Proposed handover updates (for the Hub)
- **CM-17 novelty framing (owner: Hub, before any standalone-note decision):** the no-go should be presented as a **rigorous corollary** of the exact-refresh rationale in non-Gaussian-kinetic HMC (Zhang et al. 2017/2016), not as a new impossibility result. The **`d·Θ` amplification** and the **momentum-preserving InvGauss O-step (F2)** are the defensible deltas. Reviewer risk if claimed as fully novel: high.
- **F2 attribution:** must cite Barndorff-Nielsen 1997 (NIG scale-mixture) + Dunkel–Hänggi 2009 (correct relativistic thermostat). F2 = persistence-preserving discretization of a known object; F3 (exact refresh) = the Monomial-Gamma precedent.
- **Verify-before-cite action item:** pull the exact sentence in arXiv:1706.01498 where they justify exact momentum refresh over additive noise — this is the hinge of the novelty claim; currently WebFetch-single-sourced.
- **Positioning win:** rel-SGHMC's friction damps `∇K` (correct); CHLU's code damps `p` (wrong). Our defect is partly a coding choice — reviewers will notice; recommend stating it plainly and framing F2/Newtonian-mode as the fix.
- **R7 causal-memory-floor** looks like the cleaner untouched-novelty axis (no ML prior for velocity-cap-as-erasure-bound); position vs. Lieb–Robinson.

Sources:
- [Lu et al. 2017, Relativistic Monte Carlo — PMLR](https://proceedings.mlr.press/v54/lu17b.html) · [arXiv:1609.04388](https://arxiv.org/pdf/1609.04388) · [ar5iv](https://ar5iv.labs.arxiv.org/html/1609.04388)
- [Zhang et al. 2017, Stochastic Gradient Monomial Gamma Sampler (arXiv:1706.01498)](https://arxiv.org/pdf/1706.01498) · [code](https://github.com/dreasysnail/SGMGT)
- [Zhang et al. 2016, Towards Unifying HMC and Slice Sampling (arXiv:1602.07800)](https://arxiv.org/pdf/1602.07800)
- [Barndorff-Nielsen 1997, NIG & Stochastic Volatility (Scand. J. Statist.)](https://onlinelibrary.wiley.com/doi/10.1111/1467-9469.00045)
- [Dunkel & Hänggi 2009, Relativistic Brownian motion (Physics Reports PDF)](https://math.mit.edu/~dunkel/Papers/2009DuHa_PhysRep.pdf)
- [Maxwell–Jüttner distribution — Wikipedia](https://en.wikipedia.org/wiki/Maxwell%E2%80%93J%C3%BCttner_distribution)
- [Relativistic α-stable / subordinator representation (arXiv:2603.11570)](https://arxiv.org/html/2603.11570)
- [Girolami & Calderhead 2011, Riemann manifold HMC (JRSS-B)](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00765.x)
