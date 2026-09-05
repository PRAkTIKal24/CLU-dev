# AI → CLU Primers
### Machine-learning concepts for particle-physics theorists, rebuilt as statements about physics

> **Who this is for.** You are a particle-physics theorist. You know Goldstone bosons, SSB, GMOR/ChPT, custodial symmetry, exceptional points, coset constructions, and Maxwell–Jüttner statistics — you will read the *physics* of the CLU program faster than the people who wrote it. What you do **not** need spelled out is the machine-learning scaffolding the V2 draft silently assumes: what a "recurrent network" is, what "trained" means, what an "LSTM baseline" is, what "RMSE" and "retention protocol" measure, and what a workshop reviewer cares about. This document supplies exactly that, and **only** that. It is the deliberate inverse of the `HEP_primers.md` note (which rebuilds *your* concepts for an ML audience); this one rebuilds *their* concepts for you, every one anchored to a physics analogy you already own.
>
> **How to read.** Seven short sections, one ML concept each. Each has four blocks: **Physics analogy** (your fastest on-ramp — read this first), **What it actually is** (the ML mechanism, plainly), **Look-up** (the ML name, so you can go deeper if you want), and **→ In the V2 draft** (where the paper leans on it). Read §2 first if you read nothing else: the CLU *is* a learned separable Hamiltonian evolved by velocity-Verlet, and once you see that, the rest is bookkeeping.
>
> **What this is not.** It introduces **no physics.** Where a CLU quantity has an exact physics name you already know, it is used without ceremony (a spectral mass is a normal-mode frequency; a flat direction is a Goldstone/modulus; the sampler is Langevin dynamics). The point is to hand you the ML vocabulary so the draft, the deep-dive, and the formal note become fully legible.
>
> **One-line program context (the naming, stated once).** The reference unit is the **CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)**; it is a recurrent primitive whose forward pass is *physics run forward* — a latent state $z=(q,p)$ evolved under a learned Hamiltonian $H=T(p)+V_\theta(q)$ by a damped symplectic step, with physical levers (inertial mass $M$, friction $\gamma$, temperature $T$, causal speed $c$) as its interface. Throughout, **"inertial mass $M$"** and **"spectral mass $\mu$"** are kept strictly distinct and neither is ever called bare "mass" (they run in opposite directions — §2).

---

## §1 Recurrent networks and the exploding/vanishing-gradient problem

**Physics analogy.** A recurrent network is a **discrete-time dynamical map** iterated on a state vector: $h_{t+1}=f(h_t,x_t)$, one tick per input symbol. "Memory" is nothing more mysterious than *information about early inputs surviving in $h_t$ after many iterations* — i.e. a **slow mode / long autocorrelation time** of the map. The pathology the whole field fights is the one you'd predict from the map's Jacobian: iterate $J=\partial h_{t+1}/\partial h_t$ a thousand times and the singular values either blow up (chaos, "exploding gradients") or collapse to zero (over-damped, "vanishing gradients — the state forgets"). A stable, marginally-non-contracting map is exactly the knife-edge that is hard to hit by tuning a black-box $f$.

**What it actually is.** Training a network means optimizing its parameters by gradient descent, and the gradient of the loss with respect to early-time inputs is a **product of these per-step Jacobians** (backpropagation-through-time). If $\|J\|>1$ the product diverges; if $\|J\|<1$ it underflows and no learning signal reaches long-range dependencies. Standard recurrent nets therefore cannot hold information for many steps without careful engineering.

**Look-up.** Recurrent neural network (RNN); backpropagation-through-time (BPTT); exploding/vanishing gradient problem; Long Short-Term Memory (LSTM) — a gated RNN that adds multiplicative "gates" (learned sigmoids) to *hold* a memory cell open against decay. Gates are the ML field's engineered answer to the knife-edge.

**→ In the V2 draft.** The paper's entire premise is that a **structure-preserving physical integrator** hits that knife-edge *by construction* rather than by tuning gates: a symplectic map has Jacobian singular values that come in exact reciprocal pairs $(\sigma,1/\sigma)$, so gradients cannot uniformly vanish, and volume conservation forbids silent contraction. The "exploding/vanishing" framing is the foil the introduction sets up; the LSTM is the engineered incumbent (§4).

---

## §2 The CLU primitive, in one sentence you already know

**Physics analogy.** *The forward pass is a leapfrog integration of a separable Hamiltonian whose potential is learned from data.* That is the whole unit. Take
$$H(q,p)=T(p)+V_\theta(q),\qquad T(p)=\tfrac12 p^\top M^{-1}p\ \ (\text{or a relativistic }c\sqrt{p^\top M^{-1}p+m_0^2c^2}),$$
and advance $(q,p)$ one tick with **velocity-Verlet (kick–drift–kick)**, then apply an optional per-step momentum damping $p\mapsto(1-\gamma)p$. You know this integrator cold. The *only* thing new to you is that $V_\theta$ — the potential — is **a neural network fitted to data**, not written down analytically. The latent "content" of the network lives at position $q$; its "rate of change" is momentum $p$; low $V_\theta$ = plausible/known content, high $V_\theta$ = implausible/garbage. Lead with this picture and everything else in the draft is a corollary.

**What it actually is.** A "neural-network layer" here is just a parameterized function whose parameters $\theta$ are set by optimization. $V_\theta$ is typically a small multilayer perceptron (a stack of affine maps and smooth nonlinearities — tanh/swish) or, for images, a convolutional net. The forward pass composes one Verlet step per input tick via a scan. Because $H$ is **separable** ($T$ depends only on $p$, $V$ only on $q$), the explicit KDK scheme is *exactly* symplectic for any $\theta$ — the guarantees are structural, not trained-for.

**Look-up.** Hamiltonian/symplectic neural networks; velocity-Verlet / Störmer–Verlet / leapfrog (you know it); "learned potential" = the parameters are the discretized $V$; multilayer perceptron (MLP); the damping step is *conformal-symplectic* ($J^\top\Omega J=(1-\gamma)\Omega$, $\det J=(1-\gamma)^d$).

**→ In the V2 draft.** §2 ("Setup") is exactly this. The **nomenclature you must not conflate**: *inertial mass* $M_i$ is the diagonal of $T$ (the per-coordinate stiffness of *response* — larger $M$ ⇒ slower; it is an inverse per-coordinate learning-rate of the state); *spectral mass* $\mu_k^2=\lambda_k(M^{-1/2}\nabla^2V_\theta\,M^{-1/2})$ is the normal-mode frequency at a critical point (larger $\mu$ ⇒ shorter memory). All retention statements use $\mu$; the paper forbids bare "mass." Everything downstream — the latch, the GMOR retention law, the exceptional point — is the physics of *one normal mode of a trained $V_\theta$*.

---

## §3 What "trained" and "learned $V_\theta$" mean — and the wake–sleep objective

**Physics analogy.** "Training" is a variational problem: pick the potential $V_\theta$ that minimizes a **data-fit cost** $L(\theta)$ by gradient flow on $\theta$ (a noisy relaxation in parameter space — think of $\theta$ as coordinates and $L$ as an effective free energy the optimizer rolls downhill). The specific objective the CLU uses to *shape* $V_\theta$ into a good energy landscape is a **contrastive** one you will recognize as a **Boltzmann-machine / energy-based-model learning rule**: lower the energy $V_\theta$ on real data configurations, raise it on "hallucinations" (samples the current model itself generates). At the fixed point, data sit in the valleys and the model's own confabulations have been pushed uphill — precisely the stationarity condition $\langle\partial_\theta V\rangle_{\text{data}}=\langle\partial_\theta V\rangle_{\text{model}}$ of a Boltzmann machine.

**What it actually is.** Two ingredients:
- **Gradient descent on a data-fit loss** (usually mean-squared error between the rolled-out trajectory and the target). This is the "wake" signal — it pins $V_\theta$'s *shape along trajectories* the data actually visit.
- **Wake–sleep / contrastive divergence (CD):** the "sleep" signal generates negative samples by running the dynamics (a short Langevin/MCMC chain — "dreaming"), then pushes their energy up. Because the chain is run for *few* steps and not to equilibrium, CD is a biased-but-cheap estimator of the model term. **Persistent CD (PCD)** keeps a running buffer of negatives between updates.

**Look-up.** Empirical risk minimization; stochastic gradient descent (SGD), Adam; energy-based model (EBM); Boltzmann machine learning rule; wake–sleep (Hinton et al. 1995); contrastive divergence (Hinton 2002) / persistent CD (Tieleman 2008). Known failure mode: short-run/non-convergent CD distorts the learned landscape (Fischer–Igel 2011; Nijkamp et al. 2020).

**→ In the V2 draft.** This is the machinery behind §3.5, which will read to you as a **symmetry-restoration transition**: wake–sleep CD, run to the default horizon, drives the order parameter $r^*$ of a designed degenerate vacuum to zero (the data ring becomes a local *maximum*) — the training objective *restores* the symmetry the architecture broke. The fix ("anchor $V$'s value at data") is a value-clamping term added to the wake loss. You will see immediately that $r^*$ *is* the condensate $\Sigma$ and that this is condensate melting; the ML content you need is only "what CD is and why short-run CD deforms a landscape."

---

## §4 The ML baselines, and why they are the foil (not strawmen)

**Physics analogy.** The paper's claim is that *symmetry-protected* memory (a flat coset direction that latches forever) is qualitatively different from *engineered* memory. To make that stick, it must show that the best hand-engineered recurrent memories the ML field has — the incumbents — **lack the structural signature** (no exact zero mode, no $\mu^{-2}$ law, no basin: their marginally-stable trajectories are perturbation-fragile). The baselines are the "no-symmetry control arm."

**What it actually is.** Three standard recurrent architectures serve as comparators:
- **LSTM** (Hochreiter–Schmidhuber 1997): the workhorse gated RNN; multiplicative gates hold a memory cell.
- **LEM** (Long Expressive Memory, Rusch et al. 2022) and **coRNN** (coupled-oscillator RNN, Rusch–Mishra 2021): *oscillator-based* recurrences — discretized damped oscillators engineered for long-range dependencies. These are the honest neighbors, since the CLU is *also* an oscillator at heart; the difference is that the CLU's oscillator is a **learned Hamiltonian with an exact conservation law**, not a hand-designed ODE.
- **"Well-trained baseline"** means the comparison is *fair*: each baseline is trained with a learning-rate sweep and the best-RMSE checkpoint is kept, so a reviewer cannot dismiss the result as beating a crippled opponent. In the draft the LSTM/LEM reach train-horizon RMSE $0.18$/$0.23$ rad — genuinely competent, not strawmen (V2 draft §3.3, Appendix A.2).

**Look-up.** LSTM; GRU; LEM; coRNN; "learning-rate sweep"; "best-of-sweep" model selection.

**→ In the V2 draft.** §3.3 ("Learned baselines collapse") is the head-to-head. The approved finding is that **the structural retention triad — an infinite-lifetime latch, the $\mu^{-2}$ budget law, and bounded drift — is qualitatively absent** in coRNN/LEM/LSTM (median lifetimes ≈ 5.6/56/69 map-steps, drift randomizes past 1.2 rad, and an LSTM collapses 69→2 map-steps under a 0.1 hidden-state kick), whereas a generically-trained CLU holds ≈263 bounded steps and a symmetry-*designed* CLU latches forever ($\infty$, 5/5 seeds) — all at dim 4, $S^1$ testbed, ≤5 seeds, laptop-CPU. The honest caveat (§5) is that this is a *retention*, not a task-RMSE, comparison (the unit has no native velocity-ingestion path), and the raw map-step advantage does **not** survive per-step-compute normalization (§5).

---

## §5 The ML metrics and protocols the draft uses

**Physics analogy.** Two measurement conventions you need:
- **RMSE (rad).** Root-mean-square error between the network's predicted output and the target, here an angle on the circle $S^1$ — so it is literally an *angular residual in radians* (a $0.2$-rad RMSE ≈ $11^\circ$). It is the ML field's default figure of merit; smaller = better fit.
- **The autonomous-retention protocol.** This is the paper's memory-lifetime assay, and it is exactly a **relaxation-time measurement**: *write* a value into the register (kick the state to encode a phase $\phi_0$), then **hold with zero input** and iterate the map freely, counting **map-steps** until the stored phase has drifted past a tolerance ($0.2$ rad). The count is the memory lifetime. "Map-step" = one application of the recurrent update = the natural unit of time for the dynamical system, the analogue of one integrator tick.

**What it actually is (the caveat that matters).** A map-step is *not* a fixed amount of computation across architectures. One CLU velocity-Verlet step costs two gradient evaluations of $V_\theta$ (two backprops through the potential net), so it is intrinsically heavier than one LSTM cell. Therefore a lifetime advantage measured **in map-steps** can **invert** when you renormalize by **per-step wall-clock or FLOPs** — and the draft says so explicitly (§3.3/§5): the ≈4× map-step advantage is *retired as a compute claim*; only the **qualitative, compute-independent** triad (latch / $\mu^{-2}$ law / bounded drift) is load-bearing.

**Look-up.** RMSE / MSE; "seq-len"/train-horizon vs test-horizon; FLOPs vs wall-clock; per-step-compute normalization; "map-step" (the paper's term for one recurrent update).

**→ In the V2 draft.** §3.3 and its Appendix H carry the metric definitions and the compute-normalization table; the "map-steps ≠ wall-time" caveat travels with every retention number. When you see a lifetime quoted, check whether it is a map-step count (memory-physics statement) or a compute-normalized count (systems statement) — the paper is careful to separate them, and so should any claim you build on it.

---

## §6 "Mo's law" — the published ML lifetime law the paper is head-to-head with

**Physics analogy.** Mo (2026, arXiv:2605.03338) is a recent ML result you can read in an afternoon: it proves that a $C^1$ vector field **exactly equivariant** under a Lie group $G$ (on a compact invariant set, nondegenerate orbit bundle, stabilizer $H$) has at least $\dim(G/H)$ **zero Lyapunov exponents** tangent to the group orbit — i.e. the *kinematics of protection*, the statement that a symmetry forces neutral directions. Empirically Mo then shows that when the symmetry is weakly broken, the formerly-protected direction acquires a "pseudo-gap" $\hat\lambda$ and the memory lifetime is predicted by a **single-exponential** estimator $n_{1/2}\approx\ln2/\hat\lambda$. This is the *ML-native* version of the physics you already know — a neutrality theorem plus a first-order decay fit.

**What it actually is.** "Mo's law" (as the draft uses the phrase) is that single-exponential lifetime predictor and its overdamped regime of validity. It is a *kinematic* account: it tells you a gap implies a finite lifetime, but is silent on **what sets the gap** and cannot represent anything a first-order flow can't (no saturation floor, no ringing, no exceptional point).

**Look-up.** Lyapunov exponents / spectrum; equivariant dynamical systems; continuous-attractor networks; Mo (2026). The draft reuses Mo's own diagnostics (normalized equivariance error $E_{\rm eq}$, group-tangent exponent $\hat\lambda(T)$) and his breaking-and-censoring protocol, for a like-for-like comparison.

**→ In the V2 draft (the headline, Figure 2).** The paper runs **Mo's exact protocol, unchanged, on its trained models** and finds that Mo's single-exponential law is the **overdamped face** of the CLU's mode-mass budget: measured/predicted ratio ≈ 1 in the overdamped band (Mo's published median 1.013; CLU 1.012–1.029, dim 4, 5 seeds), a $2.2\times$ delay spike **at the exceptional point**, and a decline to ≈$0.31\times$ deep underdamped — a *containment, not a conflict* (below the crossover you recover Mo's number; above it the constitutive damped-Hamiltonian structure — floor, ringing, EP — is exactly what a first-order flow cannot exhibit). The three-word summary the draft wants a reviewer to leave with: **"Mo = kinematics of protection; the CLU = a constitutive theory of degradation."** (The ≈$3.2\times$ misprediction is the trained-model number at the deepest tilt tested; a ≈$5\times$ figure exists only on the exact analytic map and is never quoted as trained-model evidence.)

---

## §7 The ML-venue frame, and the "verification vs evidence" discipline

**Physics analogy.** The paper's target venue is a **machine-learning workshop** (ML4PS — Machine Learning for the Physical Sciences; or NeurReps — Neural Representations), which is a very different animal from a PRL or JHEP submission. These are **non-archival, ~4-page** short papers reviewed by ML researchers, not physicists. Three things a workshop reviewer weighs: **novelty** (what is genuinely new vs a re-derivation), **baselines** (is the comparison fair — hence §4), and **honesty** (are the negatives disclosed). There is no expectation of a complete theory; there *is* an expectation that every claim is calibrated to its evidence.

**What it actually is — the labeling discipline you should hold the paper to.** The program enforces a strict two-tier epistemic tag on every result, and you should read the draft through it:
- **Verification** = a result on a **designed testbed**: an architecturally-invariant potential with an analytic tilt/spurion, where the theory is *exact by construction*. These confirm the theory's exactness to machine precision and are **never** presented as discoveries. (Example: the GMOR power law holding to $\mu^2_{\rm meas}/\mu^2_{\rm pred}=1.000000\pm5\times10^{-12}$ over 4.5 decades — a verification, because the potential was *built* $SO(2)$-invariant.)
- **Evidence** = a result on a **learned / trained / anharmonic** system, where the law holds only approximately, with the deviations themselves predicted (typically 2–15%). These are the load-bearing claims: the Mo head-to-head on trained checkpoints (§3.2), the learned-baseline collapse (§3.3), the training-dynamics erosion (§3.5).

**Look-up.** ML4PS; NeurReps; "non-archival workshop"; the calibration language ("verified" vs "evidenced") is the program's own house style — mirror it when you contribute, so a reviewer never mistakes a designed-testbed check for a claim about a trained network.

**→ In the V2 draft.** Every results subsection is tagged in its own header (e.g. *"Designed testbed, analytic tilts ⇒ verification"* vs *"Evidence-grade: a published ML law on trained checkpoints"*). When you assess the physics, the calibration to hold onto is: **the exact machine-precision numbers are verifications of a solvable core; the 2–15%-with-predicted-deviations numbers on trained models are the actual scientific claims.** Certificate-style statements (e.g. BIBO boundedness) carry their scope clause *next to* the claim (coercive-potential / compact-sublevel-set), never buried — you can and should hold contributed claims to the same standard.

---

### One-paragraph recap for the theorist
The CLU is a learned separable Hamiltonian run through velocity-Verlet with optional friction (§2); "trained" means its potential $V_\theta$ was fitted by gradient descent under a contrastive (energy-based, Boltzmann-machine-like) wake–sleep objective (§3); the comparators are gated/oscillator RNNs — LSTM/LEM/coRNN — swept for a fair fight (§4); memory is measured by an autonomous relaxation-time protocol counting map-steps to a phase-drift tolerance, with a compute-normalization caveat (§5); the headline pits the CLU's constitutive retention budget against Mo (2026)'s published single-exponential lifetime law and shows the latter is its overdamped face (§6); and the whole thing is pitched at a non-archival ML workshop under a verification-vs-evidence labeling discipline you should read every number through (§7). Now the draft, the deep-dive, and the formal note are yours.
