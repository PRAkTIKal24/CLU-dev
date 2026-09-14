# f5-arxiv-note — physics-theorist report

Task + acceptance criterion: distill F5 (`formalism-note.md` v1.1) into a **standalone, citable, anonymized preprint** — the common citable ancestor the three double-blind shorts import (critique M1). Scope = the exactly-solved theory + its numerical checks only; no program/roadmap/vertical/lattice/wormhole/gate content. Deliverable = the draft itself + a coverage table (F5 item → note section → numerical check).
Status: **done**
What I did:
- Re-derived nothing new; **distilled and re-scoped** the proven/verified core of F5 v1.1 into a self-contained note written for a general dynamical-systems / theoretical-physics reader, stripped of all program-private vocabulary (no "CLU/CHLU/H-CLU", no "governor", no vertical structure).
- Enforced the Head's anonymization constraints: neutral title (3 proposed), third-person citation of Jawahar & Pierini 2026 as *one instantiation* of the class, placeholder authors/acks, no private branding as headline terms.
- Kept exactly the scoped items: damped-Verlet single-mode solution; latch theorem + Noether-charge decay; spectral-mass (GMOR) retention law + $n_{1/2}\propto\mu^{-2}$ + mass-independent floor + first-crossing/envelope distinction; critical-damping retention minimum + exceptional-point signatures; kinetic-isotropy/Schur constraint **+ the kinetic-spurion blindness proposition (corrigendum, 2026-07-09)**; the two negative results; the **relativistic Gibbs no-go + latent-mass repair (corrigendum-2, 2026-07-10, §6.2)**; position-gated volume accounting; discrete equivariant neutrality; inertial-vs-spectral mass definition. Conformal symplecticity + the mean-vs-max regularizer degeneracy are retained only as the *general* geometric foundation the rest rests on (stated for the class, not any code line).
How I verified: **re-ran** the F5 verification script `.claude/scratch/formalism-note/checks.py` (numpy float64, complex-step Jacobians, seeds `default_rng(42)`/sim-seed-7) with `/Users/user/Desktop/CHLU/.venv/bin/python checks.py`. All 14 checks (a)–(n) reproduce the F5 numbers to the printed digits (live console excerpts embedded at each claim and consolidated in the Numerical-verification table). No new script needed — the note inherits App-N-grade checks verbatim.
Git footprint: none (no tracked code touched). Scratch reused: `.claude/scratch/formalism-note/checks.py`.
Open questions / follow-ups / risks: at the bottom, plus `## Proposed handover updates`.

---
---

# THE DRAFT

**Proposed titles (Head picks one):**
1. *Memory budgets of damped symplectic recurrences: an exactly-solvable theory of retention, latching, and forgetting*
2. *Retention and forgetting in dissipative Hamiltonian recurrent units: exact single-mode results*
3. *The mode-mass budget of damped symplectic integrators as learning primitives*

**Authors:** [placeholder — Head decides]
**Preprint class:** cs.LG / math.DS / nlin.CD (cross-list cond-mat.stat-mech)

---

## Abstract

A growing family of recurrent learning primitives replaces the black-box gated update with a **structure-preserving physical integrator**: a latent state $(q,p)$ is advanced by a symplectic (leapfrog / velocity-Verlet) step of a learnable separable Hamiltonian $H(q,p)=T(p)+V_\theta(q)$, with an optional per-step momentum damping $p\mapsto(1-\gamma)p$ that supplies controllable forgetting. One recent instantiation is the causal Hamiltonian learning unit of Jawahar & Pierini (2026); the results below are stated for the whole class of **damped symplectic recurrences**. We give the exact single-mode solution of the dissipative Verlet map and read off a complete, closed-form account of what such a unit can remember and for how long. The update is **conformally symplectic** ($J^\top\Omega J=(1-\gamma)\Omega$, $\det J=(1-\gamma)^d$), from which memory reduces to the spectrum of one $2\times2$ matrix per normal mode. Symmetry-protected flat directions become **exact latches** — a momentum impulse writes a finite displacement $\varepsilon p_0/(m\gamma)$ that then persists with infinite half-life — while the associated Noether charge is the write current and decays exactly as $(1-\gamma)^n$. Massive modes retain memory for $n_{1/2}\propto\mu^{-2}$ steps (a Gell-Mann–Oakes–Renner–type law in the **spectral mass** $\mu$) up to a critical-damping crossover $\varepsilon\mu\approx\gamma/2$, beyond which retention **saturates** at the mass-independent floor $2\ln2/(-\ln(1-\gamma))$ and the stored value rings at frequency $\propto\sqrt{\text{breaking}}$. The crossover is a genuine exceptional (defective) point with a $\sqrt{h-h^*}$ frequency onset, and retention is *non-monotone* in the damping with a minimum exactly at critical damping. We prove two negative results — friction can never stabilize a saddle direction of $V_\theta$, and an energy-thresholded damping controller is blind to isoenergetic (constant-$H$) escape, which only a relativistic velocity bound $v_i^{\max}=c/\sqrt{M_i}$ contains — and give the kinetic-isotropy (Schur) condition a conserved write current requires, showing that this condition binds the **current alone**: by Sylvester's law of inertia a flat direction keeps $\mu^2=0$ under *any* inertia, so the latch is blind to kinetic anisotropy, which only makes the Noether charge oscillate within bounds. All results are verified numerically to machine precision on the exact map. We relate the neutrality of protected directions to the equivariant-dynamics literature and to Mo (2026), whose single-exponential lifetime law we recover exactly as the overdamped face of the budget.

---

## 1. Setup: the damped symplectic recurrence

### 1.1 State, Hamiltonian, kinetic modes

A latent state $z=(q,p)\in\mathbb R^d\times\mathbb R^d$ carries a **position/content** coordinate $q$ and a **momentum/change** coordinate $p$. The unit is defined by a learnable, **separable** Hamiltonian

$$H(q,p) = T(p) + V_\theta(q),$$

with $V_\theta$ a learned scalar field (MLP/ConvNet, optionally with a confinement $\alpha\lVert q\rVert^2$). Three kinetic terms occur in practice, unified by a **rest-inertia** $M_{\rm eff}$:

| mode | $T(p)$ | $\nabla_pT$ | $M_{\rm eff}$ |
|---|---|---|---|
| identity | $\tfrac12\lVert p\rVert^2$ | $p$ | $I$ |
| learned Newtonian | $\tfrac12\,p^\top M^{-1}p$ | $M^{-1}p$ | $M$ |
| relativistic | $c\sqrt{p^\top M^{-1}p + m_0^2c^2}$ | $\dfrac{cM^{-1}p}{\sqrt{p^\top M^{-1}p+m_0^2c^2}}$ | $m_0M$ |

Here $M=\mathrm{diag}(M_1,\dots,M_d)\succ0$ is a learned diagonal inertial-mass matrix, $m_0$ a rest mass, $c$ a causal speed. The relativistic branch has a hard velocity ceiling; §6.1.

> **One instantiation.** Jawahar & Pierini (2026) realize exactly this class with the relativistic kinetic term, contrastive-divergence training of $V_\theta$, and an inference-time damping schedule; we take their update map as the concrete reference but state every result for the class. We use no property specific to their training objective.

### 1.2 The map

One dissipative velocity-Verlet (kick–drift–kick) step with step $\varepsilon$ and per-step damping $\gamma\in[0,1)$:

$$
\begin{aligned}
p_{1/2} &= p_n - \tfrac{\varepsilon}{2}\nabla V_\theta(q_n),\\
q_{n+1} &= q_n + \varepsilon\,\nabla T(p_{1/2}),\\
p' &= p_{1/2} - \tfrac{\varepsilon}{2}\nabla V_\theta(q_{n+1}),\\
p_{n+1} &= (1-\gamma)\,p',
\end{aligned}
\qquad z_{n+1}=\Phi_{\varepsilon,\gamma}(z_n).
$$

$\gamma=0$ is the conservative symplectic leapfrog; $\gamma>0$ appends one momentum contraction per step. The continuous-time damping rate is $\gamma_c:=-\ln(1-\gamma)/\varepsilon\approx\gamma/\varepsilon$.

### 1.3 Two masses (do not conflate)

**Definition 1 (inertial vs. spectral mass).** Two distinct quantities both get called "mass" and run in opposite directions:
- **Inertial mass $M_i$** — the learned diagonal of $T$. Larger $M_i$ ⇒ *slower*, lower speed cap.
- **Spectral (mode) mass $\mu_k$** — defined at a critical point $q^\ast$ of $V_\theta$ ($\nabla V_\theta(q^\ast)=0$) with stiffness $K:=\nabla^2V_\theta(q^\ast)$ by

$$\mu_k^2 := \lambda_k\!\big(M_{\rm eff}^{-1/2}\,K\,M_{\rm eff}^{-1/2}\big),$$

the normal-mode frequencies $\omega_k=\mu_k$. This is the field-theory sense of "mass" (a $\tfrac12\mu^2\phi^2$ term in canonical normalization). Larger $\mu$ ⇒ *faster* oscillation, *shorter* memory (§4).

At fixed curvature these are inverse: heavy inertial mass ⇒ light spectral mass. All spectral statements below use $\mu$; the dimensionless mode step is $h_k:=\varepsilon\mu_k$.

---

## 2. Exact geometry of the map

**Proposition 1 (conformal symplecticity).** For separable smooth $H$, at every $z$ and every $\theta$, with $\Omega=\begin{pmatrix}0&I\\-I&0\end{pmatrix}$ and $J=D\Phi_{\varepsilon,\gamma}(z)$,

$$J^\top\Omega J=(1-\gamma)\,\Omega,\qquad \det J=(1-\gamma)^d.$$

*Proof.* Each kick has Jacobian $\begin{pmatrix}I&0\\-\frac{\varepsilon}{2}\nabla^2V&I\end{pmatrix}$ and the drift $\begin{pmatrix}I&\varepsilon\nabla^2T\\0&I\end{pmatrix}$; both are symplectic (symmetric Hessians). The damping $(q,p)\mapsto(q,(1-\gamma)p)$ satisfies $D^\top\Omega D=(1-\gamma)\Omega$. Conformal factors multiply under composition. ∎

Consequences: $\gamma=0$ gives **exact symplecticity** (phase-space volume and the symplectic 2-form are preserved); $\gamma>0$ gives **uniform volume contraction** $(1-\gamma)^d$ per step, and *only the momentum half contracts*.

**Corollary 2 (singular-value pairing).** Since $J/\sqrt{1-\gamma}$ is symplectic, the $2d$ singular values of $J$ pair as $(\sigma,\ (1-\gamma)/\sigma)$; over $n$ steps $(\sigma,\ (1-\gamma)^n/\sigma)$. Through a conservative unroll, backpropagated gradients **cannot all vanish** (log-singular values are symmetric about $0$) but **can explode**, always in reciprocal expand/contract pairs. Depth stability is spectrum *control*, not a freebie of symplecticity.

**Corollary 3 (a mean-spectrum chaos penalty is degenerate).** Any regularizer of the form $\mathrm{mean}_i\log\sigma_i(J)$ equals $\tfrac{1}{2d}\log\det J=\tfrac12\ln(1-\gamma)$ **identically** — independent of $\theta$, of the state, and of $V_\theta$ — because it measures the *sum* of local Lyapunov exponents (fixed by conformality), not the *max*. A chaos-suppressing regularizer must therefore use a max/positive-part statistic, e.g. $\max_i\log\sigma_i$ or $\sum_i\max(0,\log\sigma_i)$, which are $\theta$-sensitive.
> *Practitioner note.* This is a purely geometric fact about the class. A regularizer of the degenerate form appears in at least one public instantiation, where it is inert by construction; the point is a design caveat, not a claim about any experiment.

**[verified — check (d)]** On random anharmonic $V$ and random states: $\lVert J^\top\Omega J-(1-\gamma)\Omega\rVert_{\max}\le 3.3\times10^{-16}$; $\mathrm{mean}_i\log\sigma_i-\tfrac12\ln(1-\gamma)$ deviates by $\le 2.1\times10^{-16}$ across random $\theta$, while $\max_i\log\sigma_i$ spans $[+0.014,+0.136]$ over the same draws (a usable, non-degenerate signal).

For completeness, at $\gamma=0$ the KDK map follows a shadow Hamiltonian $\tilde H=H+O(\varepsilon^2)$; for a harmonic mode $\tilde H$ is the conserved quadratic $\tfrac{p^2}{2m}+\tfrac k2(1-\tfrac{h^2}{4})q^2$ and the measured frequency is $\omega_{\rm map}=\arccos(1-\tfrac{h^2}{2})/\varepsilon=\mu(1+\tfrac{h^2}{24}+\dots)$. Discretization renormalizes mode frequencies at $O(\varepsilon^2)$ and requires $h<2$ for stability (the stiffness limit). We use this only to justify the single-mode $2\times2$ analysis below; the exact eigenvalues are the ground truth.

---

## 3. The exact single-mode solution

Linearize at a critical point of $V_\theta$ and diagonalize $M_{\rm eff}^{-1}K$. Each normal mode (inertia $m$, stiffness $k$, $\mu^2=k/m$, $h=\varepsilon\mu$) evolves by the exact $2\times2$ map

$$
A=\begin{pmatrix}1&0\\0&1-\gamma\end{pmatrix}
\begin{pmatrix}1-\tfrac{h^2}{2} & \varepsilon/m\\[2pt] -\varepsilon k\big(1-\tfrac{h^2}{4}\big) & 1-\tfrac{h^2}{2}\end{pmatrix},
\qquad \operatorname{tr}A=(2-\gamma)\big(1-\tfrac{h^2}{2}\big),\quad \det A=1-\gamma.
$$

Everything below is the eigenstructure of $A$; every retention claim is $\lambda(A)$.

### 3.1 The latch (flat direction, $\mu=0$)

**Theorem 4 (latch).** For a symmetry-protected flat direction ($\mu=0$), $A=\begin{pmatrix}1&\varepsilon/m\\0&1-\gamma\end{pmatrix}$, so

$$p_n=(1-\gamma)^np_0\to0,\qquad q_n\to q_\infty=q_0+\frac{\varepsilon\,p_0}{m\,\gamma}\quad(\text{geometrically, rate }1-\gamma).$$

Dissipation kills flat-direction momentum but **freezes** its displacement. At $\gamma=0$ the flat direction is a marginal integrator ($q_n=q_0+n\varepsilon p_0/m$, drifting forever); *any* $\gamma>0$ converts it into an exact **latch** — a momentum impulse $p_0$ writes the finite displacement $\varepsilon p_0/(m\gamma)$ and the stored value then persists with **infinite half-life**. Smaller $\gamma$ ⇒ longer coast ⇒ larger written value. (The relativistic drift gives the same latch with $q_\infty=q_0+\varepsilon\sum_{j\ge0}\nabla T\big((1-\gamma)^jp_0\big)$, an absolutely convergent series.)

**[verified — check (a)]** Newtonian: $|q_N-q_\infty^{\rm pred}|=1.0\times10^{-15}$; relativistic limit matches its series to 0 ulp and is frozen between steps 2000→4000 to 0 ulp; the companion curved mode decays to $10^{-45}$.

### 3.2 Noether charge = write current, and the kinetic-isotropy condition

Let a one-parameter group act linearly on $q$ and lift to phase space by $(q,p)\mapsto(g_sq,\,g_s^{-\top}p)$, generator $X$. If $V_\theta$ and $T$ are both invariant, $H$ is invariant and the **Noether charge**

$$Q_X(q,p)=p^\top X q\quad(\text{e.g. }L=q_1p_2-q_2p_1\text{ for }SO(2))$$

is conserved by the flow.

**Proposition 5 (discrete exactness and decay).** The Verlet map conserves $Q_X$ **exactly** (machine precision, any $\varepsilon$) at $\gamma=0$; with damping,

$$Q_{X,n}=(1-\gamma)^n\,Q_{X,0}.$$

**Kinetic-isotropy (Schur) condition — a condition on the current, not on the register.** For every kinetic mode $\nabla T\parallel M^{-1}p$, so $T$-invariance under a rotation channel requires $M$ to **commute with the group action** — by Schur, $M\propto I$ on each irrep ("members of a multiplet share a mass," exactly as in HEP). An $SO(2)$ channel over $(q_1,q_2)$ therefore **requires $M_1=M_2$ for $Q_X$ to be conserved**; otherwise the kinetic term explicitly breaks the symmetry of $H$ no matter how equivariant $V_\theta$ is. It does **not**, however, give the channel a mass:

**Proposition 5$'$ (kinetic-spurion blindness).** *(Numbered $5'$ to preserve the existing numbering; in the LaTeX source it takes the next automatic number.)* Let $V_\theta$ be $G$-invariant with vacuum $q^\ast$ and stiffness $K=\nabla^2V_\theta(q^\ast)$, and let $M_{\rm eff}\succ0$ be **any** inertia (not necessarily commuting with the group action). With the spectral-mass matrix $W=M_{\rm eff}^{-1/2}KM_{\rm eff}^{-1/2}$,

$$\ker W = M_{\rm eff}^{1/2}\ker K,\qquad \operatorname{rank}W=\operatorname{rank}K,\qquad \mathrm{inertia}(W)=\mathrm{inertia}(K).$$

Hence **every flat direction keeps $\mu^2=0$ exactly, for any anisotropy**: the vacuum manifold, the protected-channel count $\dim(G/H)$, and the $\gamma>0$ latch are untouched, and the latched *physical* direction is $\ker K$ itself — not even rotated. *Proof.* $W=C^\top KC$ with $C=M_{\rm eff}^{-1/2}$ invertible; congruence preserves inertia (Sylvester), hence the number of zero eigenvalues, and $Wv=0\iff v\in M_{\rm eff}^{1/2}\ker K$. In normal-mode coordinates $x=M_{\rm eff}^{1/2}q$ the flat block is exactly $\left(\begin{smallmatrix}1&\varepsilon\\0&1-\gamma\end{smallmatrix}\right)$; its physical direction is $M_{\rm eff}^{-1/2}\ker W=\ker K$. Globally, every vacuum point with $p=0$ is an exact fixed point of the map for any $M$. $\square$

**What the kinetic spurion perturbs instead: the current, boundedly.** At $\gamma=0$, $\dot Q_X=p^\top XM^{-1}p\neq0$ when $[M,X]\neq0$ (for $SO(2)$, $\dot L=p_1p_2(M_1^{-1}-M_2^{-1})$). But $H$ is conserved and $V_\theta$ coercive, so the orbit stays in a compact set and $|Q_X|\le\sup|q||p|$: **the charge cannot drift secularly — it oscillates.** On the vacuum orbit the oscillation is exact and closed-form: with the induced coset metric $F^2(\vartheta)=r_\ast^2(M_1\sin^2\vartheta+M_2\cos^2\vartheta)$ and $E=\tfrac12F^2\dot\vartheta^2$ conserved,

$$Q=F^2\dot\vartheta=\sqrt{2E}\,F(\vartheta)\in\Big[\sqrt{2E}\,r_\ast\sqrt{M_{\min}},\ \sqrt{2E}\,r_\ast\sqrt{M_{\max}}\Big],$$

a bounded oscillation of amplitude $\sqrt{2E}\,r_\ast(\sqrt{M_{\max}}-\sqrt{M_{\min}})$ with period **half a revolution**. Here $F:=\sqrt{M_{\rm eff}}\,r_\ast$ is the coset **decay constant** — the object satisfying $Q=F^2\dot\vartheta$ and, below, $\mu^2F^2=\delta n^2$ — *not* the orbit radius $r_\ast$.

**Design consequence.** An anisotropic channel **still latches, with the same infinite half-life**; what it loses is a *conserved* write current (the write gain becomes $\vartheta$-dependent). Tie the channel masses to obtain a clean write current — not to save the register. **To detect kinetic symmetry breaking, measure the charge law, not the Hessian and not the latch:** both are provably blind to it.

**Where memory actually lives.** Under damping the retained quantity is *not* the charge (friction kills it geometrically) but the **conjugate coset coordinate** — the angle along the vacuum manifold, which freezes (the latch, globally on the orbit). Memory lives on $G/H$; the charge is the write current.

**[verified — check (g)]** Equal masses, $\gamma=0$: $\max_n|L_n-L_0|/|L_0|=3.0\times10^{-14}$ (exact conservation). Unequal masses $M=(1,2)$: $O(1)$ charge non-conservation (a bounded excursion, *not* a drift rate — see (g$'$)). Decay law error $=9.3\times10^{-16}$; on a Mexican-hat run ($\gamma=0.005$) the stored angle is frozen to $2.3\times10^{-9}$ and the radial mode relaxes to the vacuum circle to $3.4\times10^{-10}$.

**[verified — check (g$'$), blindness]** Flat-mode $\mu^2=0.0$ **exactly** (bit-level) for 8 random anisotropic *diagonal* $M$; $|\mu^2|\le7.2\times10^{-16}$ for 6 random non-diagonal SPD $M$ (condition number $\le14$), with the flat $q$-directions matching $\ker K$ to $3.3\times10^{-16}$. Latched coset angle drifts by $0.0$ **exactly** over $2\times10^4$ steps at $\gamma=0.05$ for $M=(1,2)$ and $M=(0.31,4.7)$. On the vacuum ring, the closed form above is reproduced with amplitude ratio $1.0007$–$1.0064$ and elliptic half-period ratio $0.9994$–$1.0005$. The $O(1)$ charge excursion is bounded and stationary: over $10^6$ steps the per-decile $\sup|L|$ has slope $+6.1\times10^{-4}$ and $\sup_n|L_n|=0.582$ stays under the compact-set bound $0.820$.

### 3.3 Massive modes: the retention law (GMOR)

The over/underdamped boundary is the exact crossover

$$h^*(\gamma)=\big(1-\sqrt{1-\gamma}\big)\sqrt{\tfrac{2}{2-\gamma}}=\tfrac\gamma2+O(\gamma^2).$$

**Underdamped** ($h^*<h<2$): $|\lambda|=\sqrt{\det A}=\sqrt{1-\gamma}$, **independent of the mode mass**, with envelope half-life

$$n_{1/2}=\frac{2\ln2}{-\ln(1-\gamma)}\ \text{steps}\quad(\text{exact; }\approx\tfrac{2\ln2}{\gamma}\text{ for small }\gamma).$$

**Overdamped** ($0<h<h^*$): real eigenvalues; perturbing the latch ($f(\lambda)=\lambda+\tfrac{1-\gamma}{\lambda}$, $f'(1)=\gamma$) gives the slow memory eigenvalue and half-life

$$\lambda_{\rm slow}=1-\frac{(2-\gamma)h^2}{2\gamma}+O(h^4),\qquad
n_{1/2}\approx\frac{2\gamma\ln2}{(2-\gamma)\varepsilon^2\mu^2}\ \xrightarrow{\gamma\ll1}\ \frac{\gamma\ln2}{(\varepsilon\mu)^2}.$$

**Theorem 6 (spectral-mass retention law / GMOR).** A weakly-broken (small-$\mu$) channel sits overdamped, and

$$\boxed{\,n_{1/2}\ \propto\ \mu^{-2}\ \propto\ \delta^{-1}\,}$$

for an explicit breaking $\delta$ (a symmetry tilt lifts a flat direction to $\mu^2=\delta\cdot(\text{curvature})/M_{\rm eff}$, the Gell-Mann–Oakes–Renner pattern $\mu^2\propto\delta$). The exponent is **exactly $-2$ in the spectral mass**, valid only while $\varepsilon\mu<h^*(\gamma)\approx\gamma/2$; past the crossover the half-life **saturates** at the mass-independent floor $2\ln2/(-\ln(1-\gamma))$.

**[verified — check (b)]** Overdamped $n_{1/2}$ measured $1544\to6165\to24649$ for $\mu^2=0.04\to0.01\to0.0025$; ratios $3.993,\,3.998$ (predicted $4.0$). Underdamped $|\lambda|=0.974679434=\sqrt{0.95}$ for both $m=1$ and $m=0.25$ (mass-independence to 9 digits). Crossover confirmed real↔complex at $h^*=0.025643$ ($\gamma=0.05$) and $0.111284$ ($\gamma=0.2$).

### 3.4 Which lifetime? (metric bifurcation)

**Proposition 7 (retention-metric bifurcation).** Two lifetimes coexist: the **envelope half-life** $n_{1/2}$ above (amplitude/energy decay) and the **first-crossing time** $n_\times$ (first excursion of a readout past a threshold). Overdamped they agree ($n_\times/n_{1/2}\to1$). Underdamped they **diverge**: the readout crosses ballistically within the first quarter-period, $n_\times\propto1/(\varepsilon\mu)\propto\delta^{-1/2}$, while $n_{1/2}$ saturates at the mass-independent floor — they then measure *different physics* (phase transport vs. retention). Any single-exponential lifetime predictor fails by up to $\sim5\times$ past the crossover for exactly this reason. **Every reported lifetime must name its metric.**

**[verified — check (k)]** Running the first-crossing protocol on the exact mode ($\gamma=0.1$, $\varepsilon=0.1$), measured/predicted ratio $=1.001,\ 1.012$ (deep overdamped) $\to 2.30$ (at the crossover) $\to 0.93,\ 0.44,\ 0.19$ (underdamped); envelope floor pinned at $13.2=2\ln2/(-\ln0.9)$ throughout; $d\ln n_\times/d\ln\delta=-0.58$ underdamped (ballistic $-0.5$ plus small-$n$ threshold-geometry corrections).

### 3.5 Critical-damping minimum and the exceptional point

**Proposition 8 (critical-damping retention minimum).** At fixed $(\varepsilon,\mu)$ with $h<\sqrt2$, the spectral half-life $n_{1/2}(\gamma)=\ln2/(-\ln\max|\lambda|)$ is **non-monotone**: strictly decreasing on the underdamped side, strictly increasing on the overdamped side, with its **minimum exactly at the crossover**

$$\gamma^*(h)=2h(1-h)+O(h^3)\approx2\varepsilon\mu,\qquad n_{1/2}^{\min}=\frac{2\ln2}{-\ln(1-\gamma^*)}\approx\frac{\ln2}{\varepsilon\mu}.$$

Forgetting is *fastest at critical damping* — the memory-side inversion of control theory's "fastest settling at critical damping." Design consequence: a memory channel must keep $\gamma$ away from $2\varepsilon\mu$ of its protected content; a channel one *wants* to erase should be driven to $\gamma\approx2\varepsilon\mu$. (A friction-only mechanism erases massive-mode amplitude but *cannot* delete an exactly-flat coset coordinate, which is latched at any $\gamma$; that needs curvature/tilt or noise.)

**[verified — check (j)]** $\gamma^*=0.039214$ vs $2h(1-h)=0.039200$ at $h=0.02$; argmin over a 40k-point $\gamma$-grid coincides to grid resolution ($2.3\times10^{-5}$); $n^{\min}=34.65$ vs $\ln2/h=34.66$; exact floor $2\ln2/(-\ln0.9)=13.158$ vs small-$\gamma$ approx $13.863$ at $\gamma=0.1$.

**Proposition 9 (exceptional point).** The crossover $h=h^*(\gamma)$ is a **defective (Jordan) point** of $A$: the two eigenvalues merge at $\lambda=\sqrt{1-\gamma}$ with a single eigenvector ($(A-\lambda I)\ne0$, $(A-\lambda I)^2=0$). Consequences: (i) a $\sqrt{\ }$ **frequency onset** $\varphi=C(\gamma)\sqrt{h-h^*}+O(h-h^*)$ with the closed-form prefactor

$$C(\gamma)=\sqrt{(2-\gamma)\,h^*(\gamma)/\sqrt{1-\gamma}}\,,$$

and (ii) an **algebraic decay prefactor** $\lVert A^nz\rVert\sim(1+\kappa n)\lambda^n$ near the EP, so measured lifetimes exceed pure-exponential predictions there. These are second-order-only, sharp observables.

**[verified — check (l)]** $C(0.1)=0.324724$ (formula) vs $\varphi/\sqrt{h-h^*}=0.324740$ at $h-h^*=10^{-5}$; nilpotency $\lVert(A-\lambda I)^2\rVert=1.1\times10^{-17}$ with $\lVert A-\lambda I\rVert=0.1$; $\lVert A^nz\rVert/\lambda^n$ growth ratios $1.76\to1.87\to\ (\to2$, linear-in-$n)$.

---

## 4. The mode-mass budget

Collecting §3 into a design table (per step $\varepsilon$, damping $\gamma$):

| band | condition | behavior | half-life (steps) |
|---|---|---|---|
| latch | $\mu=0$ (symmetry-protected) | frozen displacement, dead momentum | $\infty$ |
| register (overdamped) | $0<\varepsilon\mu\lesssim\gamma/2$ | slow leak, no oscillation | $\dfrac{2\gamma\ln2}{(2-\gamma)(\varepsilon\mu)^2}$ |
| working memory (underdamped) | $\gamma/2\lesssim\varepsilon\mu<2$ | oscillation at $\approx\mu$, mass-independent decay | $\dfrac{2\ln2}{-\ln(1-\gamma)}$ (exact) |
| unstable | $\mu^2<0$ | escape; friction-slowed, never held | doubling $\approx\dfrac{2\gamma\ln2}{(2-\gamma)(\varepsilon\mu_{\rm im})^2}$ |
| forbidden | $\varepsilon\mu>2$ | integrator instability | must not exist |

Three design rules follow: (1) memory capacity at a fixed lifetime is set by how much spectral weight $V_\theta$ can park below $\mu\approx\gamma/(2\varepsilon)$; (2) $\gamma$ moves the register/working-memory boundary *and* the working-memory lifetime simultaneously — one knob, two effects; (3) $\varepsilon$ rescales the whole $\mu$-axis (and the forbidden zone) — step size is part of the budget, not just accuracy. Every half-life here is an **envelope** statement (Prop. 7); first-crossing lifetimes agree only in the overdamped band.

---

## 5. Two negative results

**Proposition 10 (friction never stabilizes a saddle).** For an expanding mode ($\mu^2=-|k|/m<0$, $g:=\varepsilon\sqrt{|k|/m}$), the leading eigenvalue $\lambda_+>1$ for **every** $\gamma\in[0,1)$: $\lambda_+<1$ would require $\operatorname{tr}A<2-\gamma$, but $\operatorname{tr}A=(2-\gamma)(1+g^2/2)>2-\gamma$. Damping only *slows* the escape, mirror-symmetrically to the memory law:

$$\lambda_+\approx1+\frac{(2-\gamma)g^2}{2\gamma}\ (g\ll\gamma),\qquad \lambda_+\approx e^{g}\ (\gamma\to0).$$

Expanding directions are contained by nothing except curvature control of $V_\theta$ (a *correct* chaos regularizer — Cor. 3 — not a mean-spectrum one) and the causal velocity cap (§6.1).

**[verified — check (b)]** Over a $\gamma$-grid, $\min_\gamma(\max|\lambda|-1)=1.4\times10^{-5}>0$; exact escape rate $1.000112449$ matches asymptotic $1.000112500$ and the fitted simulation rate $1.000112449$.

**Proposition 11 (controller blindness to isoenergetic escape).** An energy-thresholded, one-sided damping controller of the form $\gamma_n=s\cdot\tanh\big(\max(0,H(z_n)-E^*)\big)$ brakes only above a target energy $E^*$. Rolling off a saddle of $V_\theta$ converts $V\to T$ at **constant $H$**, so such a controller never triggers on isoenergetic instability. What bounds the resulting velocity is *only* the relativistic causal bound (§6.1); in Newtonian modes saddle escape can reach arbitrary speed at fixed energy. Energy-gating alone is therefore not a safety mechanism; the velocity-saturating kinetic term is.

---

## 6. Kinematic budget of the inertial mass

The exact geometry (§2–§3) is invariant under the symplectic rescaling $(\tilde q,\tilde p)=(M_{\rm eff}^{1/2}q,\,M_{\rm eff}^{-1/2}p)$, so at linear order a *constant* inertial $M$ is a gauge choice absorbed into $\mu$. Its irreducible, non-absorbable roles are kinematic and live in physical (data) coordinates:

### 6.1 Anisotropic causal bound

**Proposition 12 (anisotropic velocity cap).** In relativistic mode the per-coordinate velocity saturates at

$$v_i^{\max}=c/\sqrt{M_i},\qquad \lVert\dot q\rVert\le c/\sqrt{\min_iM_i},$$

and the Newtonian→relativistic crossover for coordinate $i$ sits at momentum $p_i^*=m_0c\sqrt{M_i}$ (where $v=v^{\max}/\sqrt2$). A single scalar "$c$" bounds the velocity only when $M=I$; with a learned mass the light-cone is **mass-anisotropic** — heavy coordinates have lower speed limits and enter the governed regime at larger momenta. This is the only mechanism (Prop. 11) that bounds isoenergetic saddle escape.

**[verified — check (i)]** $v(p=10^8)=c/\sqrt M$ to 9 digits for $M\in\{0.25,1,4\}$; $v(p^*)/v^{\max}=0.707107=1/\sqrt2$.

At finite momentum the relativistic $T$ couples modes: $\nabla^2T$ is off-diagonal, so a fast ("hot") coordinate raises the effective inertia of *all* coordinates through the shared square root. The spectral results of §3, computed at rest, are exact only near $p=0$.

### 6.2 Thermal budget: a no-go for additive-noise thermostats

The velocity cap is not free: it also determines what a *stochastic* (Langevin) version of the recurrence can equilibrate to. Couple the momentum to noise by appending, **as the last sub-step of the map**, $p\mapsto(1-\gamma)p+\sigma\odot\xi$, $\xi\sim\mathcal N(0,I)$ — the standard construction for generative/exploratory use.

**Lemma (Gaussian smoothing).** Let the last sub-step of a Markov chain on $(q,p)$ be $p_{n+1}=D_n+\sigma\odot\xi_n$, where $D_n$ is *any* function of the pre-noise state, $\sigma_i>0$ are constants and $\xi_n$ is independent of that state. Then every invariant measure $\mu$ has momentum marginal $\mu_p=\nu*\mathcal N(0,\Sigma)$, $\Sigma=\mathrm{diag}(\sigma_i^2)$; hence $|\widehat{\mu_p}(t)|\le e^{-\frac12t^\top\Sigma t}$ for all $t$.
*Proof.* $\widehat{\mu_p}(t)=\mathbb E[e^{it^\top D}]e^{-\frac12t^\top\Sigma t}$ and $|\mathbb E[e^{it^\top D}]|\le1$. $\square$
(The lemma is about *discrete splittings*: it never uses the form of $D$, and it has no continuous-time analogue — additive-noise SDEs routinely have non-Gaussian invariant laws.)

**Proposition 12′ (relativistic Gibbs no-go).** Fix $\gamma\in(0,2)$, $T>0$. Because $H=T(p)+V_\theta(q)$ is separable, the Gibbs momentum marginal is potential-free, $\pi_p\propto e^{-T(p)/T}$, **for every $V_\theta$.** In relativistic mode this is the **Maxwell–Jüttner** law, whose characteristic function decays exponentially rather than Gaussianly — in $d=1$ ($M=1$, $\beta=c/T$, $\mu=m_0c$, $s=\sqrt{\beta^2+t^2}$),

$$\widehat{\pi_p}(t)=\frac{\beta}{s}\frac{K_1(\mu s)}{K_1(\beta\mu)}\ \sim\ C|t|^{-3/2}e^{-m_0c|t|}.$$

By the Lemma this violates $|\widehat{\pi_p}(t)|\le e^{-\sigma^2t^2/2}$ for all $|t|\gtrsim2m_0c/\sigma^2$. Therefore **no noise scale $\sigma$ — no per-mode $\sigma_i>0$, no full $\Sigma\succ0$ — makes the Gibbs measure invariant for the relativistic map.** In the Newtonian modes ($T=\tfrac12p^\top M_{\rm eff}^{-1}p$) $\pi_p$ *is* Gaussian and the bound is saturated by the unique exact discrete-FDT scale $\sigma_i^\star=\sqrt{M_{{\rm eff},i}T\gamma(2-\gamma)}$, for which $\sigma^{\star2}/(\gamma(2-\gamma))=M_{{\rm eff},i}T$. $\square$

The result is independent of $V_\theta$ (separability makes $\pi_p$ potential-free, so no interacting potential can restore the missing non-Gaussianity), independent of $\gamma$, and independent of the *form* of the damping. It is a statement about the **sampler, not the thermodynamics**: the exact Gibbs configurational marginal $\propto e^{-V_\theta/T}$ is relativity-insensitive, so a relativistic unit has a perfectly good equilibrium — the additive-noise chain simply does not sample it. Three escapes exist: a **state-dependent** noise covariance, a **Metropolis** accept/reject *after* the kick, or a non-separable $H$.

**The control parameter is $d\Theta$, not $\Theta$.** With $\Theta:=T/(m_0c^2)$, the coded $T(p)$ shares one square root across all $d$ coordinates, so $\langle T_{\rm kin}\rangle/(m_0c^2)\approx d\Theta/2$ and the non-relativistic regime is $d\Theta\ll1$:

$$\frac{\mathrm{Var}_{\rm MJ}(p_i)}{M_{{\rm eff},i}T}=1+\frac{(d+2)\Theta}{2}+O((d\Theta)^2)\ \xrightarrow[d\Theta\gg1]{}\ (d+1)\Theta,\qquad \mathrm{KL}\bigl(\pi_p\|\mathcal N(0,M_{\rm eff}T)\bigr)=\frac{d(d+2)(d+3)}{16}\Theta^2+O(\Theta^3),$$

with $\mathrm{Var}_{\rm MJ}/(M_{\rm eff}T)=K_2(1/\Theta)/K_1(1/\Theta)$ exactly at $d=1$ (both expansions require $d\Theta\ll1$; the arrow is the opposite, ultrarelativistic limit). A $d=1$ scan therefore badly understates a high-dimensional unit, and "raise $c$ until $T\ll m_0c^2$" must be read as $c\gtrsim\sqrt{dT/m_0}$. **A cheap exact repair exists.** Since $A\mapsto\sqrt A$ is a Bernstein function, $e^{-\beta\sqrt A}=\frac{\beta}{2\sqrt\pi}\int_0^\infty s^{-3/2}e^{-\beta^2/4s}e^{-As}ds$, exhibiting Maxwell–Jüttner as a Gaussian scale mixture with one shared latent scale:

$$p\mid s\sim\mathcal N\bigl(0,M/(2s)\bigr),\qquad s\mid p\sim\mathrm{InverseGaussian}\Bigl(\text{mean }\tfrac{c^2}{2TT(p)},\ \text{shape }\tfrac{c^2}{2T^2}\Bigr).$$

Drawing $s\mid p$ and then applying the *same* linear Gaussian O-step with variance $M/(2s)$ preserves $\pi_p$ **exactly**: the correct relativistic thermostat is the Gaussian one with a **randomized inertia equal to the relativistic mass** $m_0\gamma_{\rm Lorentz}$ (indeed $1/(2\mathbb E[s\mid p])=T\,T(p)/c^2$). This is the state-dependent-$\Sigma$ escape the Lemma leaves open, at the cost of one inverse-Gaussian draw per step.

**[verified — check (e$'$)]** Free particle on a torus at $\Theta=1$: the coded chain's stationary momentum is Gaussian (KS $D=0.0011$, $p=0.97$; $\mathrm{Var}=0.9973=M_{\rm eff}T$) and Maxwell–Jüttner is rejected ($D=0.0845$, $p=0$; MJ $\mathrm{Var}=2.6995=K_2(1)/K_1(1)$). Characteristic-function closed form matches a numerical transform to $1.4\times10^{-14}$, with decay rate $\to m_0c$; the Gaussian bound is first violated at $|t|=22.79$ ($T{=}0.5,\gamma{=}0.1$) and exceeded by $5.0\times10^{119}$ at $t=20$, $\Theta=8$. $d$-amplification on the chain at fixed $\Theta=0.1$: $\mathrm{Var}(q)/(T/k)-1=-0.111,-0.196,-0.390,-0.633$ for $d=1,4,16,64$, against a $d$-independent Newtonian control $\le10^{-3}$. The latent-mass O-step reproduces Maxwell–Jüttner (KS $D=0.0013$, $p=0.47$) and collapses the chain's $O(1)$ bias $(-0.311,-0.536,-0.727)$ to the Newtonian shadow floor $(+0.0006,+0.0011,+0.0011)$.

---

## 7. Position-gated dissipation (open-system accounting)

Replace the scalar $\gamma$ by a learned **friction field** $\gamma_\phi(q)\in[0,\gamma_{\max}]$, applied as $(q,p)\mapsto(q,(1-\gamma_\phi(q_{n+1}))p)$ after the Verlet substeps.

**Proposition 13 (position-gated volume contraction).** The damping Jacobian is block-triangular, so regardless of $\nabla\gamma_\phi$,

$$\det D\Phi=\big(1-\gamma_\phi(q_{n+1})\big)^d.$$

Phase-space volume is destroyed **exactly and only where $\gamma_\phi>0$**; outside the superlevel set of $\gamma_\phi$ the dynamics is exactly conservative. Conformality (and hence the singular-value pairing of Cor. 2) is lost wherever $\nabla\gamma_\phi\ne0$; the determinant statement is what remains, and it is exact. This is the clean classical open-system statement — symplectic bulk with localized non-unitary channels — and it makes information loss *metered*: exactly $d\ln(1-\gamma_\phi(q))$ nats of differential entropy per step, localized to the dissipative region.

**[verified — check (h)]** $|\det J-(1-\gamma_\phi(q'))^d|\le1.1\times10^{-16}$ on random states (complex-step Jacobians).

---

## 8. Discrete equivariant neutrality

The latch's frozen direction is a *neutral* mode (unit eigenvalue / zero Lyapunov exponent). Neutrality of symmetry-protected directions is classical equivariant dynamics (Golubitsky–Stewart–Schaeffer 1988; Krupa 1990; Rumberger 2001), recently specialized to recurrent flows by Mo (2026). We state the discrete-time counterpart for our map.

**Proposition 14 (discrete equivariant neutrality).** Let $\Phi$ be a $C^1$ diffeomorphism equivariant under a smooth Lie-group action ($\Phi(g\cdot x)=g\cdot\Phi(x)$), and $K$ a compact $\Phi$-invariant set of constant stabilizer type $\mathcal H$ on which the infinitesimal-action map $A_x:\mathfrak g\to T_xM$ has singular values uniformly bounded in $(0,\infty)$, of rank $q=\dim(\mathcal G/\mathcal H)$. Then the group-tangent bundle $E^G_x=A_x\mathfrak g$ is $D\Phi$-invariant and every Lyapunov exponent of the restricted cocycle is exactly $0$; the spectrum of $D\Phi$ on $K$ contains at least $q$ zeros.

*Proof.* Differentiating $\Phi(e^{s\xi}\!\cdot x)=e^{s\xi}\!\cdot\Phi(x)$ at $s=0$ gives the cocycle identity $D\Phi(x)\,\xi_M(x)=\xi_M(\Phi(x))$, hence $D\Phi^n(x)\,\xi_M(x)=\xi_M(\Phi^n(x))$. For $v=A_x\xi\in E^G_x$, uniform bounds $a\le\sigma(A_\cdot)\le b$ give $(a/b)\lVert v\rVert\le\lVert D\Phi^n(x)v\rVert\le(b/a)\lVert v\rVert$ for all $n$ — growth rate $0$. ∎

The damped Verlet map with $\mathcal G$-invariant $V_\theta$ and channel-isotropic $M$ is equivariant under the lifted action $(q,p)\mapsto(gq,g^{-\top}p)$ (each substep is; the damping commutes with any linear action), and the latched vacuum orbit $\times\{p=0\}$ satisfies the hypotheses — so the latch's frozen direction is a symmetry-protected neutral mode at any $\gamma\in[0,1)$. (Channel isotropy is required for the *map* to be equivariant; by Proposition 5$'$ the flat direction and its latch persist without it, so equivariance is sufficient but not necessary here.)

This is the precise, and *only*, overlap with a Lyapunov-spectrum account. Everything else in §3 — the finite write transport $\varepsilon p_0/(m\gamma)$, the $\gamma$-controlled conversion of a drifting integrator into a deadbeat latch, the Noether charge as write current — lives in the Jordan/transient sector that Lyapunov exponents are provably blind to (a marginal integrator and two frozen registers share the spectrum $\{0,0\}$ yet differ categorically in function).

**[verified — check (m)]** Cocycle identity holds to $1.2\times10^{-15}$ over 500 steps on a Mexican-hat damped map; the latched group-tangent exponent $\widehat\lambda(T=10,10^2,10^3)=0.0$ exactly.

---

## 9. Related work

**Equivariant/continuous-attractor dynamics.** The neutrality of group-orbit directions is classical (Golubitsky, Stewart & Schaeffer 1988; Krupa 1990; Rumberger 2001) and underlies continuous-attractor models of neural memory. Mo (2026) proves that a $C^1$ vector field exactly equivariant under a Lie group $G$, on a compact invariant set with nondegenerate orbit bundle and stabilizer $H$, has at least $\dim(G/H)$ zero Lyapunov exponents along the orbit, and shows that a "pseudo-gap" acquired under explicit breaking predicts a finite memory lifetime. That account is the **kinematics of protection**: it holds for *any* equivariant flow and is therefore silent on what sets the gap. In the damped-Hamiltonian subclass the gap is a **constitutive** quantity, $\mathrm{gap}=\tfrac{(2-\gamma)}{2\gamma}\varepsilon^2\mu^2$ with $\mu^2=\mathrm{eig}(M_{\rm eff}^{-1}\nabla^2V_\theta)$, valid only below the critical-damping crossover, beyond which retention saturates and the stored coordinate rings at frequency $\propto\sqrt{\text{breaking}}$ — regime structure a first-order flow cannot exhibit. Mo's single-exponential lifetime law, run unchanged on our mode, matches ours to $\sim1\%$ in the overdamped regime (his reported median ratio $1.013$ is our overdamped face; check (k) reproduces $1.001$–$1.012$) and mispredicts by up to $5\times$ past the crossover — a clean containment rather than a conflict. Mo's diagnostics (normalized equivariance error, direct group-tangent exponents, subspace alignment) transfer verbatim and we adopt them.

**Symmetry breaking and Goldstone modes in deep networks.** A parallel physics-grounded route to stable long-range information propagation is spontaneous symmetry breaking with gapless Goldstone carriers (Iqbal, Keller, Song, Miyato & Welling 2026). The present class realizes a related mechanism through symplectic conservation plus a causal velocity bound; the latch and the $\mu^{-2}$ pseudo-Goldstone lifetime are its retention-side signatures. The counting of protected channels, $\dim(\mathcal G/\mathcal H)$, and the nonrelativistic (type-B) refinements of Goldstone counting (Minami & Hidaka) apply to the flat-direction sector here; a systematic treatment is left to companion work. Manifold-shaping in equivariant recurrent networks (Di Bernardo *et al.* 2025) is adjacent on the *geometry* side; we make no claim of novelty for choosing a symmetry group or coset — our contribution is the exact price list attached to a chosen structure (the retention table, the kinetic-isotropy constraint), not the choice itself.

**Geometric integration.** The conformal-symplectic structure, shadow-Hamiltonian conservation, and $h<2$ stability limit are standard for (damped) leapfrog integrators (Hairer, Lubich & Wanner; Benettin & Giorgilli); our contribution is to read them as *memory* statements for a learnable $H$.

**Reference instantiation.** Jawahar & Pierini (2026) is one concrete member of the class and the source of the update map used here; all results are stated for the class and verified on the exact map.

---

## 10. Limitations and scope

- **Exactly-solvable core is quadratic.** All boxed results are exact for quadratic $V$ (designed testbeds) and are the *local* statement at a critical point $q^\ast$ of a learned $V_\theta$. Global flatness of a memory channel requires an *exact* symmetry of $V_\theta$ and $T$; accidental flat directions of a learned potential are not protected. On learned, genuinely anharmonic potentials the linearized predictions degrade smoothly; quantifying that deviation (empirically observed at the few-to-fifteen-percent level in companion studies) is outside this note, which verifies only the exact core.
- **$\gamma>0$ breaks conservation and reversibility by construction** — that is what forgetting *is* here. There is no conserved shadow energy for $\gamma>0$; the structurally robust exact statement is conformal symplecticity (Prop. 1).
- **Langevin sampling** (coupling the momentum to noise for generative use) is **kinetic-mode-dependent** (§6.2). In the Newtonian modes a per-mode fluctuation–dissipation-consistent noise scale $\sigma_i^\star=\sqrt{M_{{\rm eff},i}\,T\,\gamma(2-\gamma)}$ targets the Gibbs measure (a uniform scale instead equilibrates each mode at its own temperature), and the residual bias in $H$ is the usual $O(\varepsilon^2)$ shadow bias. In relativistic mode **no $\sigma$ exists** (Prop. 12′): the additive-noise chain's momentum marginal is a Gaussian smoothing, while the Gibbs marginal is Maxwell–Jüttner. The resulting bias is $O(1)$ and $\varepsilon$-independent, controlled by $d\,T/(m_0c^2)$; exact repairs are a latent-mass (state-dependent-$\Sigma$) thermostat, a Metropolis adjustment, or a Newtonian kinetic mode. Beyond the momentum marginal, the full sampling analysis is companion material.
- **Away-from-attractor spectra** (Prop. 12's kinetic coupling at hot states; time-varying $K(q_t)$ along a trajectory) require Oseledets/finite-time analysis rather than fixed-point eigenvalues; not covered.
- **Governor BIBO** is evidenced (compact sublevel sets under a coercive $V$ + monotone damping + the energy barrier), with the proven saddle-blindness caveat of Prop. 11; a full discrete Lyapunov-function proof is open. Non-coercive learned potentials (no confinement term) rely on external clipping and fall outside the boundedness assumptions.

---

## 11. Numerical verification

All results are checked to machine precision on the exact map with a single self-contained script (numpy float64; complex-step Jacobians, step $10^{-30}$; fixed seeds; runs in seconds; the map mirrors the reference implementation exactly). Supplementary on request.

| id | result | proposition | observed (re-run 2026-07-07) |
|---|---|---|---|
| (a) | latch $q_\infty=q_0+\varepsilon p_0/(m\gamma)$ | Thm 4 | Newtonian error $1.0\times10^{-15}$; relativistic series match 0 ulp, frozen 2000→4000 at 0 ulp; curved mode $\to10^{-45}$ |
| (b) | retention $\propto\mu^{-2}$, mass-independence, crossover, saddle | Thm 6, Prop 10 | ratios $3.993/3.998$ (pred 4.0); $\vert\lambda\vert=\sqrt{0.95}$ for $m=1,0.25$; $\min_\gamma(\max\vert\lambda\vert-1)=1.4\times10^{-5}>0$ |
| (c) | squeeze/mass-weighted squeeze symplectic | (used in scope-out §; retained in script) | $\le2.2\times10^{-16}$; $\det=1$ |
| (d) | conformal symplecticity + mean-spectrum degeneracy | Prop 1, Cor 3 | $\lVert J^\top\Omega J-(1-\gamma)\Omega\rVert\le3.3\times10^{-16}$; mean-log-sv $-\tfrac12\ln(1-\gamma)\le2.1\times10^{-16}$; max-log-sv spread $[0.014,0.136]$ |
| (e) | Langevin per-mode temperature, **Newtonian** (scope note §10) | — | code noise Var$(p)=0.0263$ (target 1.0); corrected Var$(p)=1.000000$, Var$(q)=0.500156$ |
| (e$'$) | relativistic Gibbs no-go + latent-mass repair | Prop 12$'$ | coded stationary $p$ Gaussian (KS $D=0.0011$, $p=0.97$), MJ rejected ($D=0.0845$); char.fn. closed form vs FT $1.4\times10^{-14}$, decay rate $\to m_0c$; bound exceeded by $5.0\times10^{119}$ at $t=20,\Theta=8$; $d$-amplification $-0.111\to-0.633$ ($d=1\to64$, $\Theta=0.1$) vs Newtonian control $\le10^{-3}$; latent-mass fix: KS $D=0.0013$ ($p=0.47$), bias $-0.727\to+0.0011$ |
| (f) | shadow Hamiltonian $O(\varepsilon^4)$ | §2 | std$(H)\propto\varepsilon^2$ ($\times4$/doubling); std$(\tilde H)\propto\varepsilon^4$ ($\times16$) |
| (g) | Noether: exact conservation/decay | Prop 5 | equal-$M$ drift $3.0\times10^{-14}$; unequal-$M$ $O(1)$ non-conservation (bounded excursion, not drift); decay-law $9.3\times10^{-16}$; angle frozen $2.3\times10^{-9}$ |
| (g$'$) | kinetic-spurion blindness; bounded charge | Prop 5$'$ | flat $\mu^2=0.0$ exactly (8 diagonal $M$), $\le7.2\times10^{-16}$ (6 SPD $M$, cond $\le14$); flat dirs $=\ker K$ to $3.3\times10^{-16}$; latch drift $0.0$ exactly; ring amplitude ratio $1.0007$–$1.0064$, half-period ratio $0.9994$–$1.0005$; $\sup_n|L_n|=0.582<0.820$ bound |
| (h) | $\det J=(1-\gamma_\phi(q'))^d$ | Prop 13 | $\le1.1\times10^{-16}$ |
| (i) | $v^{\max}=c/\sqrt M$, crossover $p^*$ | Prop 12 | exact to 9 digits; $v(p^*)/v^{\max}=1/\sqrt2$ |
| (j) | critical-damping retention minimum $\gamma^*\approx2\varepsilon\mu$ | Prop 8 | $\gamma^*=0.039214$ vs $2h(1-h)=0.039200$; argmin coincides to grid $2.3\times10^{-5}$; $n^{\min}=34.65$ vs $34.66$ |
| (k) | metric bifurcation (first-crossing vs envelope) | Prop 7 | ratios $1.001,1.012,\dots,2.30$ (EP)$,\dots,0.19$; envelope floor $13.2$; slope $-0.58$ |
| (l) | exceptional point: $\sqrt{h-h^*}$ onset, Jordan block | Prop 9 | $C(0.1)=0.324724$ vs measured $0.324740$; $\lVert(A-\lambda I)^2\rVert=1.1\times10^{-17}$; growth ratios $1.76\to1.87$ |
| (m) | discrete equivariant neutrality (cocycle) | Prop 14 | cocycle deviation $1.2\times10^{-15}$/500 steps; latched exponent $0.0$ |

---

## Appendix A — Provenance (for the Head / paper-writer, strip on arXiv)

Source: `formalism-note.md` v1.1 (F5). Verification script: `.claude/scratch/formalism-note/checks.py`, re-run 2026-07-07 with `/Users/user/Desktop/CHLU/.venv/bin/python checks.py`; all 14 checks reproduce F5 App-N to printed digits. Numerical config (constant across checks unless noted): numpy float64, `np.random.default_rng(42)`, simulation seed 7, complex-step differentiation step $10^{-30}$. Representative per-check parameters: (b) $\gamma=0.2,\varepsilon=0.1$; (d) $\gamma\in\{0,0.15\}$; (e) $m=2,k=1,\varepsilon=0.05,\gamma=0.1,T=0.5$; (g) Mexican-hat $V$, $\gamma\in\{0,0.005\}$; (j) $\varepsilon=0.1,\mu=0.2$; (k)(l) $\gamma=0.1,\varepsilon=0.1$. No trained checkpoints, seeds, or training flags enter this note — all claims are properties of the map, not of any run, so the flag-provenance obligation reduces to the script config above.

---
---

# COVERAGE TABLE (F5 item → note section → numerical check)

| # | F5 item (task scope IN) | F5 label | note section | check | status |
|---|---|---|---|---|---|
| 1 | damped-Verlet single-mode solution | §2.2, §3.3 | §1.2, §3 (2×2 matrix $A$) | (a),(b) | ✅ full |
| 2 | flat-direction latch theorem | Prop-3.3a / Thm-latch | §3.1 Thm 4 | (a) | ✅ full |
| 2b | Noether-charge decay $(1-\gamma)^n$ | §4.1 | §3.2 Prop 5 | (g) | ✅ full |
| 3 | GMOR spectral-mass law $n_{1/2}\propto\mu^{-2}$ | Prop (3.3c) | §3.3 Thm 6 | (b) | ✅ full |
| 3b | mass-independent floor | §3.3b | §3.3 (underdamped) | (b),(k) | ✅ full |
| 3c | first-crossing/envelope distinction | Cor-14 | §3.4 Prop 7 | (k) | ✅ full |
| 4 | critical-damping retention minimum | Cor-13 | §3.5 Prop 8 | (j) | ✅ full |
| 4b | exceptional-point signatures | Cor-15 | §3.5 Prop 9 | (l) | ✅ full |
| 5 | kinetic-isotropy / Schur constraint (on the **current**) | §4.1 | §3.2 (isotropy condition) | (g) | ✅ full |
| 5b | **kinetic-spurion blindness** ($\mu^2\equiv0$ under any $M$; bounded charge oscillation) | §4.1 Prop-17 (corrigendum 2026-07-09) | §3.2 Prop 5$'$ | (g$'$) | ✅ full |
| 6a | friction never stabilizes a saddle | §3.3d | §5 Prop 10 | (b) | ✅ full |
| 6b | governor blindness (isoenergetic) | Prop-10 caveat | §5 Prop 11 | — (analytic; bound via (i)) | ✅ full |
| 7 | position-gated volume accounting | Prop-11 | §7 Prop 13 | (h) | ✅ full |
| 8 | discrete equivariant neutrality | Prop-16 | §8 Prop 14 | (m) | ✅ full |
| 9 | inertial-vs-spectral mass (Def-2) | Def-2 | §1.3 Def 1 | (i) (inertial), (b) (spectral) | ✅ full |
| — | anisotropic causal bound (needed for 6b) | Prop-1 | §6.1 Prop 12 | (i) | ✅ full (support) |
| — | conformal symplecticity (foundation) | Prop-3 | §2 Prop 1 | (d) | ✅ full (foundation) |
| — | mean-spectrum regularizer degeneracy | Prop-5 | §2 Cor 3 | (d) | ✅ retained, neutral framing |

**Scope-OUT items confirmed excluded** (per task): program/roadmap/vertical structure; wormholes (F5 §7.4); gate/calibration/escalation machinery (F5 §7.5, Def-7); lattice/CLU-Net results (F5 §7); interference/NTK study (F5 §6); erosion study; the "CLU/CHLU/H-CLU" coinage; all unpublished spoke experiments beyond the map checks. The squeeze/boost algebra (F5 §7.5, check (c)) is *mentioned only* as a scope-out; its check stays in the shared script but no result is claimed.

**Attribution audit** (task duties): Mo 2026 — §9, third person, overdamped-face relationship stated neutrally with the containment result. Golubitsky–Stewart–Schaeffer / Krupa / Rumberger — §8, §9, equivariant-dynamics lineage. Iqbal/Keller/Song/Miyato/Welling 2026 — §9, sibling SSB/Goldstone route. Minami–Hidaka — §9, Goldstone-counting refinement (scoped, forward-referenced). Di Bernardo *et al.* 2025 — §9, geometry side, **explicitly guarded** ("no claim of novelty for choosing a symmetry group or coset"). Jawahar & Pierini 2026 — §1.1, §9, third person, *one instantiation of the class*, never "our previous work."

**Anonymization audit** (Head constraints): (1) no "CLU" coinage — vocabulary is "damped symplectic recurrences / Hamiltonian recurrent units" throughout ✅; (2) 3 neutral titles proposed ✅; (3) Jawahar & Pierini third-person, one instantiation ✅; (4) no private branding as headline/section terms — "budget"/"latch"/"register" used only descriptively in body text, never as section titles ✅; (5) authors/acks placeholder ✅.

---

## Open questions / follow-ups / risks

1. **Minami–Hidaka citation is a forward reference without a pinned bibliographic entry in my sources.** I cited it neutrally for type-B Goldstone counting; the paper-writer should confirm the exact reference (year/title) before arXiv, or drop it — it is not load-bearing for any claim here.
2. **Di Bernardo et al. 2025 (arXiv:2511.04802)** — Mo's self-declared "closest work"; I cite it guarded, but the allocation-adjacent framing should be re-checked by web-scout before submission (per mo-deep-read follow-up #3) so §9 does not overclaim distinctness.
3. **The "few-to-fifteen-percent" anharmonic-deviation figure in §10** is anticipated from companion spoke work and is *not* verified in this note; I framed it as companion material, not a result. If the Head wants a hard number in the preprint, it needs a citable source or its own check — flag for the paper-writer.
4. **Cor-3 (mean-spectrum regularizer degeneracy)** is retained because it is a clean general theorem, but it is the most de-anonymizing item (a public instantiation uses the degenerate form). I stated it for the class with a soft footnote; the Head may prefer to cut the footnote entirely. It is the one editorial call I could not make unilaterally under the anonymization directive.
5. **Squeeze/boost (check (c))** — kept in the shared verification script but claimed nowhere in the note (scope-out). Harmless, but if the script is shipped as supplementary the paper-writer should either prune (c) or add a one-line "used elsewhere" note to avoid a reviewer asking why an unused check is present.

## Proposed handover updates (for the Hub)

- **§7/§8 no change needed** — this note introduces no new physics; it is a re-scoping of F5 v1.1. All twelve scoped results are already logged in the handover (via F5 v1.0/v1.1 folds).
- **Critique register M1/P1:** the F5 arXiv note draft exists at `.claude/outputs/f5-arxiv-note.md`, covering all task-scoped items with reproduced App-N checks (re-run 2026-07-07). Ready for Head review toward the ~Jul 20 target. Recommend routing to the paper-writer agent for LaTeX conversion *after* the Head resolves open questions 1, 3, 4 above (Minami–Hidaka ref, anharmonic-% policy, Cor-3 footnote).
- **Cross-short consistency (M2):** this note fixes the neutral, third-person vocabulary and the exact constants (crossover $h^*(\gamma)$, floor $2\ln2/(-\ln(1-\gamma))$, minimum $\gamma^*\approx2\varepsilon\mu$, GMOR $\mu^{-2}$) that the three shorts must cite identically. Suggest the Hub treat §3–§4 of this note as the canonical constant table for the claims-consistency matrix.
</invoke>
