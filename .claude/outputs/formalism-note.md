# formalism-note — physics-theorist report

Task + acceptance criterion: F5 — write the unified CLU-Net formalism note (markdown, HEP-colleague-legible) that all shorts import; sections 1–8 mandated; ≥3 numerical sanity checks with reported numbers.
Status: **done** · **v1.1 update applied 2026-07-06 (task f5-v11):** wave-2 audit folded in (Cor-13..15, Prop-16, exact-floor wording, Hyp-3/Def-7 empirical status, App-N jitter annotation); checks extended to (a)–(n); deltas reported in `.claude/outputs/f5-v11.md`.
What I did:
- Read protocol, handover (§1–§2, §7, §8), brainstorm log (all threads), roadmap, and `chlu/core/` + relevant `chlu/training/` lines (ground truth for the map, the Lyapunov-loss call with `gamma=0`, clip locations).
- Derived: exact conformal symplecticity of the coded dissipative Verlet map; singular-value pairing; **proof that the coded Lyapunov regularizer is θ-independent (degenerate)**; the Goldstone latch theorem (exact); the full per-mode retention law (exact 2×2 eigenvalues, overdamped/underdamped regimes, crossover, pseudo-Goldstone exponent **−2 in mode mass**); saddle-never-stabilized corollary + governor blindness to isoenergetic escape; exact discrete Noether conservation/decay conditions (incl. the **mass-isotropy requirement**); the anisotropic causal bound `c/√M_i`; the **Langevin FDT mismatch** in the coded noise scale (+ exact fix); KDK shadow Hamiltonian (validated) + symmetry protection of flat directions; squeeze-operator certificates; lattice composition conditions; interference-kernel formalization.
- Wrote the standalone formalism note below (the deliverable other papers import).
How I verified: `uv run --no-project --with numpy python .claude/scratch/formalism-note/checks.py` — v1.0: 9 checks (a)–(i); v1.1 (2026-07-06): +5 checks (j)–(n) — pure numpy float64 + complex-step Jacobians (no JAX needed; avoids cold-start). Key observed numbers are embedded at the claims and consolidated in Appendix N of the note. Headlines: latch error 1.0e−15; half-life 1/μ² ratios 3.993/3.998 (pred 4.0); underdamped |λ|=√(1−γ) exact to 1e−9 across masses; `‖JᵀΩJ−(1−γ)Ω‖ ≤ 3.3e−16`; mean-log-singular-value = ½ln(1−γ) to 2.1e−16 across random θ (degeneracy proven+observed); code Langevin Var(p)=0.0263 vs Maxwell–Boltzmann target 1.0, corrected noise gives 1.000000; shadow-H fluctuation scales ε⁴ (ratios 4.0/16.0); Noether drift 3.0e−14 over 1e5 steps (equal masses) vs 2.6 (unequal); γ-decay law exact to 9.3e−16; det J = (1−γ(q′))^d to 1.1e−16; v_max = c/√M exact.
Git footprint: none (no tracked code touched). Scratch: `.claude/scratch/formalism-note/checks.py`.
Open questions / follow-ups / risks: listed in note §9 (Open problems) and in the handover-updates section at the bottom.

---

# The CLU Formalism — shared definitions for the CLU program (v1.1)

## Changelog

**v1.1 — 2026-07-06 (task f5-v11; consolidation of the wave-2 audit — mo-deep-read, v2-so2-build, v1-l0-gate, mass-spectrum-peek).** No v1.0 claim renumbered, weakened, or removed; v1.0 numbering is frozen (papers may already cite "F5 Prop-n"). New labels continue the shared Prop/Cor counter; their placement in the document is thematic, not numerical.
- **NEW — Cor-13** critical-damping retention minimum at $\gamma^*$ = root of $h^*(\gamma)=h$ $= 2\varepsilon\mu(1-\varepsilon\mu)+O(h^3)$, plus the trash-region friction spec $\gamma_\phi(q)\approx2\varepsilon\mu(q)$ (§3.5); **Cor-14** first-crossing vs envelope retention-metric bifurcation + mandated metric discipline (§3.5, pointer at §3.4); **Cor-15** exceptional-point signatures at $h^*$, now with the closed-form onset prefactor $C(\gamma)=\sqrt{(2-\gamma)h^*/\sqrt{1-\gamma}}$ (§3.5); **Prop-16** discrete-time equivariant-neutrality corollary (after Mo 2026, arXiv:2605.03338), two-line cocycle proof (§3.3a). Sources: mo-deep-read C1–C3 (verified there); independently re-verified here as checks (j)–(m). Alias map: mo-deep-read C1→Cor-13, C2→Cor-14, C3→Cor-15 — deliberately renamed because Prop-12 already uses "(C1)/(C2)" as internal certificate labels; cite the Cor-n names.
- **CHANGED (wording only):** the underdamped half-life floor is quoted exactly as $2\ln2/(-\ln(1-\gamma))$ everywhere; $2\ln2/\gamma$ is shown once (§3.3b) as its small-$\gamma$ approximation (13.16 vs 13.86 at $\gamma=0.1$).
- **ANNOTATED:** §3.3a neutrality attribution (Mo 2026; Golubitsky–Stewart–Schaeffer 1988; Krupa 1990; Rumberger 2001) + pointer to the canonical V2 positioning prose (mo-deep-read §4 — import, don't fork); §5 Hyp-3 falsifiables (i)/(ii) empirical status (mass signal real but near-uniform under *both* training paths — hierarchy is designed-in doctrine; (ii) untestable-until-banded); §6 first $\Theta(q_A,q_B)$ measurement slot (v3-lattice-build); §7.5/Def-7 single-shell gate evidence + V1 pivot (v1-l0-gate); App-N first-crossing artifact = kick-phase jitter with corrected closed form $\approx\pm1/(2h)$ — v2-so2-build's quoted form mixed the energy-ledger ripple with the amplitude-ledger rate (×2 too big); its *measured* ±10/±12 match the corrected form (check (n)).
- Checks (j)–(n) appended to `checks.py`; Appendix N table extended.

**Purpose.** One notation, one set of definitions, one formalism that the V1/V2/V3 (and possible V2b) short papers and the ICLR paper import verbatim. Self-contained; written for a theoretical-physics reader (HEFT/SMEFT background assumed for §4). The reference implementation is the `chlu` package (Jawahar & Pierini 2026); code is ground truth and every definition below is checked against it.

**Status labels.** Every nontrivial claim carries one of:
**[proven]** derived here (or standard) with proof/derivation; **[verified]** additionally confirmed numerically (check id from Appendix N in brackets); **[evidenced]** supported by standard arguments or numerics but not fully proven here; **[conjectured]** plausible, falsifiable, unproven; **[design hypothesis]** an engineering bet, to be settled by experiment (per program principle P3).

Cross-reference labels: **Def-n** definitions, **Prop-n** propositions, **Cor-n** corollaries (sharing the Prop counter: Prop-12 is followed by Cor-13), **Hyp-n** design hypotheses, **Open-n** open problems. Cite as "F5 Prop-4".

---

## 1. Nomenclature: the CLU family

**Def-1 (CLU).** A **Causal Learning Unit** (CLU, pronounced "clue") is a recurrent learning primitive defined by its **lever interface**, not by any one instantiation:

| Lever | Symbol | Role |
|---|---|---|
| structure-preserving core | $\Phi_{\varepsilon,\gamma}$ | a (conformally-)symplectic update map — geometry is fixed, not learned |
| causal bound | $c$ | hard speed limit on latent transport (light-cone structure) |
| inertial mass matrix | $M$ | learned per-coordinate inertia (budget allocator, §5) |
| friction | $\gamma$ (scalar, per-unit, or field $\gamma(q)$) | the single forgetting knob; volume contraction |
| temperature | $T$ | exploration / generative sampling (Langevin) |
| learned energy | $V_\theta(q)$ | the content: where memories, attractors, and garbage live |
| energy diagnostics | $H(z)$, residual $R$ | universal confidence/anomaly/escalation signal |

Instantiations are declared placeholders (program principle P2): the current implementation realizes the levers through a **Hamiltonian** engine and is the **H-CLU**; a Lagrangian realization is reserved as **L-CLU**; the naming pattern extends. Everything in this note is about the H-CLU unless stated.

**Continuity statement (use verbatim in papers):** *"the CLU, introduced as CHLU in Jawahar & Pierini (2026)"*; the code package and CLI retain the name `chlu` until the scheduled rename.

---

## 2. The H-CLU

### 2.1 State and Hamiltonian

Latent state $z = (q, p) \in \mathbb{R}^d \times \mathbb{R}^d$ (position/content, momentum/change). Learnable Hamiltonian

$$H(q,p) = T(p) + V_\theta(q),$$

**always separable** (this matters: §2.3). The learned potential $V_\theta$ is an MLP/ConvNet scalar field; in the small-`mlp` variant it includes a confinement term $\alpha \lVert q\rVert^2$, $\alpha = 0.05$ (the deep/conv variants omit it — see Prop-10 caveat). Three kinetic modes (code: `kinetic_mode`):

| mode | $T(p)$ | $\nabla_p T$ | effective inertia at $p\!\approx\!0$ |
|---|---|---|---|
| `newtonian_identity` | $\tfrac12 \lVert p\rVert^2$ | $p$ | $M_{\rm eff}=I$ |
| `newtonian_learned` | $\tfrac12\, p^{\!\top} M^{-1} p$ | $M^{-1}p$ | $M_{\rm eff}=M$ |
| `relativistic` | $c\sqrt{p^{\!\top} M^{-1} p + m_0^2 c^2}$ | $\dfrac{c\,M^{-1}p}{\sqrt{p^{\!\top}M^{-1}p+m_0^2c^2}}$ | $M_{\rm eff}=m_0 M$ |

with $M = \mathrm{diag}(\mathrm{softplus}(\text{log\_mass}))$ positive by construction, rest mass $m_0$, causal speed $c$ (fixed hyperparameters in code). Note the code's relativistic $T$ equals $c\sqrt{p^\top M^{-1}p + (m_0c)^2}$, i.e. rest energy $T(0)=m_0c^2$. Two exact kinematic facts:

**Prop-1 (anisotropic causal bound).** In relativistic mode the per-coordinate velocity saturates at
$$v^{\max}_i = c/\sqrt{M_i},\qquad \lVert \dot q\rVert \le c\,/\sqrt{\min_i M_i},$$
and the Newtonian→relativistic crossover for coordinate $i$ sits at momentum $p^*_i = m_0 c \sqrt{M_i}$ (where $v = v^{\max}/\sqrt2$). *The paper's "velocity saturates at $c$" is exact only for $M=I$; with learned mass the light-cone is mass-anisotropic — heavy coordinates have lower speed limits and enter the relativistic (governed) regime at larger momenta.* **[proven; verified (i): $v(10^8)=c/\sqrt M$ to 9 digits for $M\in\{0.25,1,4\}$]**

**Prop-2 (relativistic kinetic mode-coupling).** $\nabla^2_p T = \frac{c}{\sqrt{\cdot}}\left[M^{-1} - \frac{(M^{-1}p)(M^{-1}p)^{\!\top}}{p^{\!\top}M^{-1}p + m_0^2c^2}\right]$: diagonal $\big(=(m_0M)^{-1}\big)$ at $p=0$, but **off-diagonal at finite momentum** — a fast ("hot") coordinate increases the effective inertia of *all* coordinates through the shared square root. Newtonian modes have no such coupling. Consequence: spectra (§3) computed at rest are exact only near $p=0$; boosted/hot states are heavier and kinetically coupled. **[proven]**

### 2.2 The dissipative Verlet map (exact code semantics)

One step of `chlu/core/integrators.py::velocity_verlet_step` with step $\varepsilon$ (code `dt`) and per-step friction $\gamma \in [0,1)$:

$$
\begin{aligned}
p_{1/2} &= p_n - \tfrac{\varepsilon}{2}\, \nabla_q V_\theta(q_n) \\
q_{n+1} &= q_n + \varepsilon\, \nabla_p T(p_{1/2}) \\
p' &= p_{1/2} - \tfrac{\varepsilon}{2}\, \nabla_q V_\theta(q_{n+1}) \\
p_{n+1} &= (1-\gamma)\, p'
\end{aligned}
\qquad \Longleftrightarrow \qquad z_{n+1} = \Phi_{\varepsilon,\gamma}(z_n).
$$

(The code calls $\partial_q H(q,p)$, which equals $\nabla_q V(q)$ by separability.) $\gamma=0$: conservative kick–drift–kick (KDK) leapfrog. $\gamma>0$: momentum damping appended once per step. Dimensionless friction; the continuous-time rate is $\gamma_c := -\ln(1-\gamma)/\varepsilon \approx \gamma/\varepsilon$.

### 2.3 What is and is not preserved

**Prop-3 (exact conformal symplecticity).** Let $\Omega = \begin{pmatrix} 0 & I \\ -I & 0\end{pmatrix}$ and $J = D\Phi_{\varepsilon,\gamma}(z)$. For separable smooth $H$, at every $z$ and for every $\theta$:
$$J^{\!\top} \Omega\, J = (1-\gamma)\,\Omega, \qquad \det J = (1-\gamma)^d .$$
*Proof.* Each kick has Jacobian $\begin{pmatrix} I & 0\\ -\frac{\varepsilon}{2}\nabla^2V & I\end{pmatrix}$, the drift $\begin{pmatrix} I & \varepsilon\nabla^2T\\ 0 & I\end{pmatrix}$; both are symplectic because the Hessians are symmetric. The damping map $(q,p)\mapsto(q,(1-\gamma)p)$ satisfies $D^{\!\top}\Omega D = (1-\gamma)\Omega$. Composition multiplies conformal factors. ∎
Corollaries: $\gamma=0$ ⇒ **exact symplecticity** (phase-space volume and the 2-form preserved); $\gamma>0$ ⇒ uniform volume contraction $(1-\gamma)^d$ per step (only the momentum half contracts). **[proven; verified (d): $\lVert J^\top\Omega J - (1-\gamma)\Omega\rVert_{\max} \le 3.3\times10^{-16}$ on random anharmonic $V$, random states]**

**Prop-4 (singular-value pairing — the honest depth-stability statement).** Since $J/\sqrt{1-\gamma}$ is symplectic, the $2d$ singular values of $J$ pair as $(\sigma,\ (1-\gamma)/\sigma)$; over $n$ steps, $(\sigma,\ (1-\gamma)^n/\sigma)$, and $\prod_i \sigma_i = (1-\gamma)^{nd}$. So through a conservative ($\gamma=0$) unroll, **gradients cannot all vanish** (log-singular values are symmetric around 0) **but can explode**, always in reciprocal expand/contract pairs. Friction shifts the symmetry point to $\tfrac{n}{2}\ln(1-\gamma)$ per pair, a uniform contraction overlay. Depth stability is therefore *spectrum control*, not a freebie of symplecticity. **[proven]**

**Prop-5 (the coded Lyapunov regularizer is degenerate).** `compute_lyapunov_loss` returns $\text{mean}_i \log\sigma_i(J)$. By Prop-3,
$$\text{mean}_i \log \sigma_i \;=\; \tfrac{1}{2d}\log\det J \;=\; \tfrac12 \ln(1-\gamma)$$
**identically — independent of $\theta$, of the state, and of $V_\theta$.** In the wake phase the step is called with $\gamma=0$ (verified in `train.py:155`), so the loss is $\equiv 0$ and its $\theta$-gradient is $0$ up to the $10^{-8}$ epsilon inside the log and float32 round-off. The regularizer cannot penalize chaos even in principle: it measures the *sum* of local Lyapunov exponents (always $d\ln(1-\gamma)$), not the *max*. **[proven; verified (d): deviation from $\tfrac12\ln(1-\gamma)$ ≤ 2.1e−16 across random $\theta$; while $\max_i\log\sigma_i$ varies by O(0.1) across the same draws — a usable replacement signal].** Fix specification for `experiment-engineer`: replace with $\max_i \log\sigma_i$, or $\sum_i (\log\sigma_i)^2$ (hyperbolicity/squeeze magnitude), or $\sum_i \max(0,\log\sigma_i)$; all are $\theta$-sensitive.

**Prop-6 (reversibility).** With $R(q,p)=(q,-p)$: $R\circ\Phi_{\varepsilon,0}\circ R\circ \Phi_{\varepsilon,0} = \mathrm{id}$ (Verlet is symmetric-reversible). This is the precise content of "recognition = time-reversed generation". $\gamma>0$ breaks reversibility by construction — that is what forgetting *is* in this formalism. **[proven, standard]**

**Prop-7 (shadow Hamiltonian, $\gamma=0$).** The KDK map exactly follows, to $O(\varepsilon^4)$, the modified ("shadow") Hamiltonian
$$\tilde H = H + \frac{\varepsilon^2}{12}\Big[(\nabla_pT)^{\!\top}\, \nabla_q^2V \,(\nabla_pT) \;-\; \tfrac12\, (\nabla_qV)^{\!\top}\, \nabla_p^2T\, (\nabla_qV)\Big] + O(\varepsilon^4),$$
and for real-analytic $H$ (tanh/swish networks **are** analytic; ReLU would void this) a truncated $\tilde H$ is conserved to $O(\varepsilon^N)$ over times $e^{O(1/\varepsilon)}$ on compact sets (Benettin–Giorgilli). Energy is bounded-oscillating, not drifting. **[proven for the quadratic case + standard theory; verified (f): on a quartic well, $\mathrm{std}(H)\propto\varepsilon^2$ (×4.0 per doubling), $\mathrm{std}(\tilde H)\propto\varepsilon^4$ (×16.0 per doubling)]**
For the harmonic mode this gives exactly the conserved quadratic $\tilde H \propto \frac{p^2}{2m} + \frac{k}{2}\big(1-\tfrac{(\varepsilon\omega_0)^2}{4}\big)q^2$ and a measured frequency $\omega_{\rm map} = \arccos(1-\tfrac{h^2}{2})/\varepsilon = \omega_0(1 + \tfrac{h^2}{24} + \dots)$, $h:=\varepsilon\omega_0$: discretization renormalizes **masses** (mode frequencies) at $O(\varepsilon^2)$. Stability requires $h < 2$, i.e. $\varepsilon < 2/\omega_{\max}$ — the stiffness limit (paper §D.2).
For $\gamma>0$ there is **no** conserved shadow energy; the structurally robust exact statement is Prop-3 (conformal symplecticity). Backward-error analysis still yields a modified *vector field* = damped-Hamiltonian flow + $O(\varepsilon)$ non-Hamiltonian corrections. **[evidenced, standard]**

**Prop-8 (symmetry protection survives discretization).** If a linear symplectic group action preserves both $T$ and $V_\theta$, every term of the shadow expansion (built from Poisson brackets of $T$ and $V$) is invariant, so $\tilde H$ inherits the symmetry: **an exactly-protected flat (Goldstone) direction stays exactly flat in the shadow Hamiltonian — discretization cannot generate a mass for it.** Massive-mode timescales get $O(\varepsilon^2)$-renormalized (Prop-7); symmetry-protected memory does not degrade with step size. **[proven for linear point symmetries with invariant $T$, $V$]**

### 2.4 Langevin extension (generative/exploratory mode)

Code (`langevin_step`): full Verlet step, then $p \leftarrow (1-\gamma)p$, then $p \leftarrow p + \sigma\,\xi$, $\xi\sim\mathcal N(0,I)$, with $\sigma_{\rm code} = \sqrt{2\gamma T \varepsilon}$ (uniform over coordinates). **The additive Gaussian kick is the *last* sub-step of the map** — this structural fact is what Prop-9′ turns on.

> **⚠ Corrigendum, 2026-07-10 (`f5-corrigendum-2`).** The previous version of this section stated $\sigma_i^\star$ as *the* exact discrete-FDT noise, as a **class-level** statement. It is exact **only for the Newtonian kinetic modes.** In `relativistic` mode **no $\sigma$ whatsoever** gives the coded chain a Gibbs invariant (Prop-9′ below) — the correct statement is an impossibility theorem, not a formula. Open-3 (§9) is thereby **closed**, and its guess that the residual bias is "$O(\varepsilon^2)$" is **retracted**: the bias is $O(1)$, controlled by $d\,T/(m_0c^2)$ and *independent of $\varepsilon$*. Prop-9 below is re-scoped; Prop-9′ is new.

**Prop-9 (exact discrete FDT — Newtonian kinetic modes only).** Let $T(p)=\tfrac12 p^\top M_{\rm eff}^{-1}p$ (`newtonian_identity`, `newtonian_learned`). The damping+noise ("O") sub-step $p' = (1-\gamma)p + \sigma\xi$ is an autonomous linear OU recursion with stationary momentum variance $\sigma^2 / (\gamma(2-\gamma))$ per coordinate. Maxwell–Boltzmann at temperature $T$ requires $\mathrm{Var}(p_i) = M_{{\rm eff},i}\,T$, i.e. the **unique** exact discrete-FDT noise ($\gamma\in(0,2)$)
$$\boxed{\ \sigma_i^{\star} = \sqrt{M_{{\rm eff},i}\; T\; \gamma\,(2-\gamma)}\ }$$
whereas the coded $\sigma_{\rm code}=\sqrt{2\gamma T\varepsilon}$ yields the **effective per-mode temperature**
$$T_{{\rm eff},i} = \frac{2\,\varepsilon\, T}{(2-\gamma)\, M_{{\rm eff},i}} \;\ne\; T .$$
Consequences: (1) code "temperatures" are not in energy units — $\varepsilon$ and $M$ are absorbed; annealing schedules still work (monotone rescale), but $T$ values are not comparable across $dt$/mass/architecture changes. (2) With learned non-uniform $M$, **each mode equilibrates at its own temperature** — FDT is violated per-mode and the chain has *no* Gibbs invariant of the form $e^{-H/T}$. (3) With the corrected $\sigma^\star$, the harmonic chain samples the Gibbs measure of the **shadow** Hamiltonian exactly ($\mathrm{Var}(q) = T/\tilde k$), i.e. $O(\varepsilon^2)$-biased in $H$ — Metropolis-adjust or reduce $\varepsilon$ if exactness matters. **[proven; verified (e): code noise gives $\mathrm{Var}(p)=0.0263$ vs MB target $mT=1.0$ (predicted $T_{\rm eff}=0.0132$ vs nominal $T=0.5$); corrected noise gives $\mathrm{Var}(p)=1.000000$ (exact discrete Lyapunov solve) and $0.9994$ (simulation), $\mathrm{Var}(q)=0.500156 = T/\tilde k$ matching the shadow prediction $0.5\,(1+h^2/4)$]**

**Lemma-9a (Gaussian-smoothing bound).** Suppose the **last** sub-step of a Markov chain on $(q,p)$ is $p_{n+1} = D_n + \sigma\odot\xi_n$, where $D_n$ is *any* (possibly nonlinear) function of the pre-noise state, $\sigma_i>0$ are constants, and $\xi_n\sim\mathcal N(0,I)$ is independent of the pre-noise state. Then **every** invariant probability measure $\mu$ has a momentum marginal that is a Gaussian smoothing,
$$\mu_p = \nu * \mathcal N(0,\Sigma),\qquad \Sigma=\mathrm{diag}(\sigma_i^2),$$
and therefore $\bigl|\widehat{\mu_p}(t)\bigr| \;\le\; e^{-\frac12 t^\top\Sigma\,t}\quad\forall t\in\mathbb R^d.$
*Proof.* $\widehat{\mu_p}(t)=\mathbb E[e^{it^\top D}]\,e^{-\frac12 t^\top\Sigma t}$ by independence, and $|\mathbb E[e^{it^\top D}]|\le1$. $\square$ *(Note: the lemma never uses the form of the damping. It is a statement about discrete splittings; it does **not** apply in continuous time, where additive-noise SDEs routinely have non-Gaussian invariant laws.)*

**Prop-9′ (relativistic Gibbs no-go).** Let $T(p)=c\sqrt{p^\top M^{-1}p+m_0^2c^2}$ with $m_0,c>0$ (`relativistic`), $\gamma\in(0,2)$, $T>0$. Because $H=T(p)+V_\theta(q)$ is **separable**, the Gibbs momentum marginal factorizes off the potential entirely,
$$\pi_p(p)\ \propto\ e^{-T(p)/T}\ =\ \exp\Bigl(-\tfrac{c}{T}\sqrt{p^\top M^{-1}p+m_0^2c^2}\Bigr)\qquad\text{(\textbf{Maxwell–Jüttner}),}$$
**for every $V_\theta$.** Its characteristic function decays *exponentially*, not Gaussianly: in $d=1$ ($M=1$, $\beta=c/T$, $\mu=m_0c$, $s=\sqrt{\beta^2+t^2}$)
$$\widehat{\pi_p}(t)=\frac{\beta}{s}\,\frac{K_1(\mu s)}{K_1(\beta\mu)}\ \sim\ C\,|t|^{-3/2}e^{-m_0c\,|t|},\qquad |t|\to\infty .$$
By Lemma-9a, an invariant $\pi_p$ would need $|\widehat{\pi_p}(t)|\le e^{-\sigma^2t^2/2}$, which fails for all $|t|\gtrsim 2m_0c/\sigma^2$. Hence
$$\boxed{\ \text{\textbf{no }}\sigma\ \text{(no per-mode }\sigma_i>0,\ \text{no full }\Sigma\succ0)\ \text{gives the coded relativistic Langevin a Gibbs invariant.}\ }$$
∎ **Scope, sharply.** (i) The result is **independent of $V_\theta$** — separability makes $\pi_p$ potential-free, so *no interacting potential can restore the missing non-Gaussianity*; a non-separable $H$ would escape (then $\pi_p$ is a $V$-dependent mixture). (ii) It is **independent of $\gamma$** and of the *form* of the damping — Lemma-9a never touches $D$. (iii) The Newtonian case is exactly the boundary: $\pi_p$ is Gaussian and $\sigma^\star$ *saturates* the bound, $\sigma^{\star2}/(\gamma(2-\gamma))=M_{\rm eff}T$. (iv) The three escapes are: **state-dependent** noise covariance (Fix F2/F3), a **Metropolis** accept/reject *after* the kick (Fix F4), or non-separability. **[proven; verified (e′)]**

**The control parameter is $d\Theta$, not $\Theta$.** Write $\Theta := T/(m_0c^2)$. The coded $T(p)$ shares **one** square root across all $d$ coordinates, so equipartition gives $\langle T_{\rm kin}\rangle/(m_0c^2)\approx d\Theta/2$: the non-relativistic regime is $d\Theta\ll1$. Exactly (per coordinate, in the reduced variable $u=M^{-1/2}p$):
$$\frac{\mathrm{Var}_{\rm MJ}(p_i)}{M_{{\rm eff},i}T}=1+\frac{(d+2)\Theta}{2}+O\bigl((d\Theta)^2\bigr) \ \xrightarrow[\ d\Theta\gg1\ ]{}\ (d+1)\,\Theta ,
\qquad \mathrm{KL}\bigl(\pi_p\,\|\,\mathcal N(0,M_{\rm eff}T)\bigr)=\frac{d(d+2)(d+3)}{16}\,\Theta^2+O(\Theta^3)\ \text{nats},$$
(both expansions require $d\Theta\ll1$; the arrow is the opposite, ultrarelativistic limit) with excess kurtosis $3\Theta+O(\Theta^2)$ at $d=1$, where additionally $\mathrm{Var}_{\rm MJ}/(M_{\rm eff}T)=K_2(1/\Theta)/K_1(1/\Theta)$ **exactly**. At *fixed $d$* the whole coded chain reduces to $(\varepsilon\sqrt{k/m_0},\,\gamma,\,\Theta)$ — a scaling lemma verified bit-identically — which is why $(c,T)$ pairs with equal $\Theta$ agree to the last digit. **Do not extrapolate a $d=1$ scan to a high-$d$ unit:** the reference instantiation's generative configuration ($d=784$, $m_0=c=1$, $T:1\!\to\!0.01$) runs at $d\Theta = 784\to7.8$ — **ultrarelativistic throughout**, with the true momentum variance $785\times$ the sampler's at the start of the anneal. Raising $c$ to $5$ leaves $d\Theta=31.4$; **$d\Theta<1$ needs $c\gtrsim\sqrt{dT/m_0}\approx28$** at $T=1$.

**Fixes, by cost (`experiment-engineer` spec).**
| | fix | exact? | cost | note |
|---|---|---|---|---|
| **F1** | raise $c$ or $m_0$ until $d\Theta\ll1$ | no — $O(d\Theta)$ bias | one config line | *not* $\Theta\ll1$; needs $c\gtrsim\sqrt{dT/m_0}$ |
| **F2** | **latent-mass augmentation** (below) | **yes** (momentum marginal exact) | one inverse-Gaussian draw/step | keeps the Gaussian O-step; **dominates F3** |
| **F3** | exact Maxwell–Jüttner momentum refresh (Andersen) | yes | MJ sampler | destroys momentum persistence ⇒ slower $q$-mixing |
| **F4** | Metropolis-adjust the composite step | yes, *and* removes the $O(\varepsilon^2)$ shadow bias | $H$-evals + reversibility care | the only route to exact Gibbs in $H$ |
| **F5** | use a Newtonian kinetic mode | n/a — Prop-9 holds | free | what every $T>0$ law in this note assumes |

**F2, the latent-mass fix (new).** The subordinator identity $e^{-\beta\sqrt{A}}=\frac{\beta}{2\sqrt\pi}\int_0^\infty s^{-3/2}e^{-\beta^2/4s}e^{-As}\,ds$ (valid since $A\mapsto\sqrt{A}$ is a Bernstein function) exhibits Maxwell–Jüttner as a **Gaussian scale mixture** with a *single shared* latent scale $s$:
$$p\mid s\ \sim\ \mathcal N\!\bigl(0,\ M/(2s)\bigr),\qquad s\mid p\ \sim\ \mathrm{InverseGaussian}\Bigl(\text{mean }\tfrac{c^2}{2\,T\,T(p)},\ \text{shape }\tfrac{c^2}{2T^2}\Bigr).$$
Replace the O-step by: *draw $s\mid p$, then* $p\leftarrow(1-\gamma)p+\sqrt{(1-(1-\gamma)^2)\,M/(2s)}\;\xi$. Each stage preserves the joint $\pi(p,s)$, so the composite preserves $\pi_p=$ MJ **exactly**. Since $1/(2\,\mathbb E[s\mid p])=T\,T(p)/c^2$, the physical reading is clean: **the exact relativistic FDT noise is the coded Gaussian noise with a *randomized inertia* equal to the relativistic mass $m_0\gamma_{\rm Lorentz}$.** This is precisely the state-dependent-$\Sigma$ escape hatch that Lemma-9a leaves open. **[proven; verified (e′): O-step alone at $\Theta=8$ reproduces MJ (KS $D=0.0013$, $p=0.47$; $\mathrm{Var}=130.98$ vs exact $130.2555=K_2/K_1\cdot m_0T$); in the full chain the $O(1)$ bias $-0.311/-0.536/-0.727$ collapses to $+0.0006/+0.0011/+0.0011$ — the Newtonian shadow floor]**

*Root cause, in one line (independently reached by `xy-lattice-theory` §5(v)):* the Gibbs-preserving underdamped Langevin damps the **velocity** $\nabla_pT$; the code damps $p$. For Newtonian these coincide ($\Gamma=\gamma M$); for relativistic $T$, $\nabla_pT\propto p/T(p)\ne p/\text{const}$. **The failure is in the sampler, not in the thermodynamics** — the exact Gibbs $q$-marginal $\propto e^{-V_\theta/T}$ is *relativity-insensitive* (the momentum integral factorizes out), so a relativistic unit has a perfectly good equilibrium; the coded chain simply does not sample it.

### 2.5 The energy governor (inference-time control)

Code (`governed_rollout`): per step, $\gamma_n = s\cdot\tanh\big(\max(0,\, H(z_n) - E^*)\big)$ — one-sided braking above a target energy $E^*$ (set from training data, e.g. the 1st-percentile mean energy), frictionless coasting below. Damping strictly decreases $T(p)$ (all three kinetic modes are increasing in $p^\top M^{-1}p$; for the relativistic mode one damping step removes $\approx \gamma\,(T - m_0^2c^4/T)$), so the governor implements a soft energy ceiling → limit-cycle-like behavior at $E^*$.

**Prop-10 (practical BIBO).** Assume (A1) coercive total potential ($V(q)\to\infty$ as $\lVert q\rVert\to\infty$ — holds for `PotentialMLP` via $\alpha\lVert q\rVert^2$; **fails architecturally** for `DeepPotentialMLP`/`ConvPotential`, which rely on `jnp.clip` in the *training* loop, outside the unit); (A2) $T$ coercive in $p$ (all modes); (A3) analytic $H$. Then sublevel sets $\{H\le E\}$ are compact, $\gamma=0$ energy is bounded-oscillating over exponentially long horizons (Prop-7), damping only removes energy, and the governor supplies a restoring barrier at $E^*$: trajectories remain bounded. **[evidenced — full discrete Lyapunov-function proof is Open-1]**
**Sharp caveat (governor blindness):** rolling off a saddle of $V$ converts $V\to T$ at *constant* $H$ — the governor never triggers on isoenergetic instability. What bounds the resulting velocity is **only the relativistic causal bound** (Prop-1); in Newtonian modes saddle escape can reach arbitrary speed at fixed energy. This is the precise sense in which the relativistic mode is the safety feature, and energy-gating alone is not. **[proven]**

**Code↔formalism dictionary:** `dt`↔$\varepsilon$; `gamma`↔$\gamma$; `speed_of_causality`↔$c$; `rest_mass`↔$m_0$; `softplus(log_mass)`↔$M$; confinement $\alpha=0.05$ (`mlp` only); `ConvPotential` output scaled by $1/100$; experiment defaults: Exp A `newtonian_identity`, Exp B `newtonian_learned` (+governor), Exp C `relativistic` (paper's Exp II figure used a relativistic override with $c=5$, project `finalA`).

---

## 3. Mode taxonomy and the mass spectrum (centerpiece)

### 3.1 Linearization and the two masses

Let $q^\ast$ be a critical point of $V_\theta$ ($\nabla V(q^\ast)=0$), $K := \nabla^2 V(q^\ast)$ the **stiffness (curvature) matrix**, $M_{\rm eff}$ the inertia at rest (§2.1 table). Linearized conservative dynamics: $M_{\rm eff}\,\delta\ddot q = -K\,\delta q$. In canonically normalized coordinates ($\tilde q = M_{\rm eff}^{1/2}q$, $\tilde p = M_{\rm eff}^{-1/2}p$ — itself a symplectic map) the whole linear problem is the spectrum of
$$W := M_{\rm eff}^{-1/2}\, K\, M_{\rm eff}^{-1/2},\qquad \mu_k^2 := \lambda_k(W),$$
with normal-mode frequencies $\omega_k = \mu_k$.

**Def-2 (the two masses — do not conflate).**
- **Inertial mass $M_i$** (a.k.a. kinetic mass): the learned diagonal of the kinetic term. Larger $M$ ⇒ *slower*, lower speed cap.
- **Spectral (mode) mass $\mu_k$**: $\sqrt{\text{eigenvalue of } M_{\rm eff}^{-1}K}$ — the HEP-sense mass (a $\tfrac12\mu^2\phi^2$ term in canonical normalization). Larger $\mu$ ⇒ *faster* oscillation, *shorter* memory (see 3.3).

These run in **opposite directions**: at fixed curvature, heavy-$M$ ⇒ light-$\mu$. "Heavy" in EFT language (large $\mu$ = fast = UV) is the opposite of "heavy" in the allocator language (large $M$ = slow = IR). All EFT statements in §4–§5 use $\mu$. **[definitional — this disambiguation corrects loose wording in earlier program notes]**

Everything below is exact for quadratic $V$; for learned $V_\theta$ it is the local statement at $q^\ast$ (global flatness requires symmetry, §4). Define the **dimensionless mode step** $h_k := \varepsilon\,\mu_k$.

### 3.2 Taxonomy

At an attractor, each normal mode falls in one class ($\lambda$ = eigenvalues of the per-mode 2×2 damped-Verlet map $A$, derived in 3.3):

| class | spectrum | dynamics under $\Phi_{\varepsilon,\gamma}$ | functional reading |
|---|---|---|---|
| **flat / Goldstone** | $\mu = 0$ | $\lambda = \{1,\ 1-\gamma\}$ exactly: displacement frozen, momentum dies | transport & dissipation-proof memory (registers) |
| **massive** | $\mu^2 > 0$ | underdamped: decaying oscillation, $\vert\lambda\vert=\sqrt{1-\gamma}$; overdamped: slow leak $\lambda_{\rm slow}\approx 1 - \tfrac{(2-\gamma)h^2}{2\gamma}$ | rationed, designable forgetting (working memory) |
| **expanding** | $\mu^2 < 0$ (saddle of $V$) | $\lambda_+ > 1$ for **every** $\gamma\in[0,1)$ | sensitivity/chaos; creation of distinctions; gradient explosion channel |

plus the honest pairing statement Prop-4: symplectic pairing $(\sigma, (1-\gamma)/\sigma)$ means no mode class ever vanishes silently — every expansion has a paired contraction.

### 3.3 The exact per-mode law **[proven; verified (b)]**

One mode, inertia $m$, stiffness $k$ (so $\mu^2 = k/m$, $h = \varepsilon\mu$), the exact one-step matrix of §2.2:
$$A = \begin{pmatrix}1 & 0\\ 0 & 1-\gamma\end{pmatrix}\begin{pmatrix} 1-\tfrac{h^2}{2} & \varepsilon/m \\ -\varepsilon k\big(1-\tfrac{h^2}{4}\big) & 1-\tfrac{h^2}{2}\end{pmatrix},\qquad \operatorname{tr}A = (2-\gamma)\big(1-\tfrac{h^2}{2}\big),\quad \det A = 1-\gamma .$$

**(a) Goldstone latch (exact).** $\mu=0$: $A = \begin{pmatrix}1 & \varepsilon/m\\ 0 & 1-\gamma\end{pmatrix}$, so
$$p_n = (1-\gamma)^n p_0 \ \to\ 0, \qquad q_n \to q_\infty = q_0 + \frac{\varepsilon\, p_0}{m\,\gamma}\quad(\text{geometrically, rate } 1-\gamma).$$
**Dissipation kills Goldstone momentum but cannot erase Goldstone displacement** — and more: it *freezes* it. At $\gamma=0$ the flat direction is a marginal integrator ($q_n = q_0 + n\varepsilon p_0/m$, drifts forever); any $\gamma>0$ converts it into an exact **latch**: a write (momentum impulse $p_0$) transports the register by the finite amount $\varepsilon p_0/(m\gamma)$, after which the stored value persists with **infinite half-life**. Smaller $\gamma$ ⇒ longer coast ⇒ larger written displacement. The relativistic drift gives the same latch with $q_\infty = q_0 + \varepsilon\sum_{j\ge0} \nabla_pT\big((1-\gamma)^j p_0\big)$ (absolutely convergent; no closed form). **[proven; verified (a): Newtonian $|q_N - q_\infty^{\rm pred}| = 1.0\times10^{-15}$; relativistic limit matches its series to 0 ulp and is frozen to 0 ulp between steps 2000→4000; curved companion mode decayed to $10^{-45}$]**

**Neutrality attribution (v1.1).** That the protected direction is *neutral* (unit eigenvalue / zero Lyapunov exponent) is classical equivariant dynamics — group orbits of equivariant systems carry forced neutral directions (Golubitsky–Stewart–Schaeffer 1988; Krupa 1990; Rumberger 2001) — recently specialized to recurrent flows by Mo 2026 (arXiv:2605.03338), whose Theorem 1 we adapt to our map as Prop-16 below. Everything in the latch *beyond* neutrality — the finite write transport $\varepsilon p_0/(M\gamma)$, the $\gamma$-controlled conversion of a drifting integrator into a deadbeat latch, the Noether charge as write current (§4.1) — lives in the Jordan/transient sector that Lyapunov spectra are provably blind to (mo-deep-read §2b: a marginal integrator and two frozen registers have identical spectra $\{0,0\}$) and is ours. **Canonical V2 related-work prose: mo-deep-read §4 — import it verbatim, do not fork it here.**

**Prop-16 (discrete-time equivariant neutrality; corollary of Mo 2026 Thm 1).** Let $\Phi$ be a $C^1$ diffeomorphism equivariant under a smooth Lie-group action ($\Phi(g\cdot x)=g\cdot\Phi(x)$), and $K$ a compact $\Phi$-invariant set with constant stabilizer type $\mathcal H$ on which the infinitesimal-action map $A_x:\mathfrak g\to T_xM$ has singular values uniformly bounded in $(0,\infty)$, $\operatorname{rank} = \dim(\mathcal G/\mathcal H) =: q$. Then the group-tangent bundle $E^G_x = A_x\,\mathfrak g$ is $D\Phi$-invariant and every Lyapunov exponent of the restricted cocycle is exactly $0$; the spectrum of $D\Phi$ on $K$ contains at least $q$ zeros.
*Proof (two lines).* Differentiating $\Phi(e^{s\xi}\!\cdot x)=e^{s\xi}\!\cdot\Phi(x)$ at $s=0$ gives the cocycle identity $D\Phi(x)\,\xi_M(x)=\xi_M(\Phi(x))$, hence $D\Phi^n(x)\,\xi_M(x)=\xi_M(\Phi^n(x))$ for all $n$. For $v=A_x\xi\in E^G_x$, the uniform bounds $a\le\sigma(A_\cdot)\le b$ on $K$ give $(a/b)\lVert v\rVert \le \lVert D\Phi^n(x)\,v\rVert = \lVert A_{\Phi^nx}\,\xi\rVert \le (b/a)\lVert v\rVert$ for all $n$ — growth rate $0$ on all of $E^G$. ∎
*Application.* The damped Verlet map with $\mathcal G$-invariant $V_\theta$ and channel-isotropic $M$ is equivariant under the lifted action $(q,p)\mapsto(gq,\ g^{-\top}p)$ (each substep is, per §4.1; the damping commutes with any linear action), and the latched vacuum orbit $\times\{p=0\}$ satisfies the hypotheses — so the latch's frozen direction is a symmetry-protected neutral mode in Mo's sense, at any $\gamma\in[0,1)$. *Provenance:* Mo states Theorem 1 for autonomous flows only; the map version above is **ours-stated, his-flow-theorem-inspired** — cite it that way. **[proven; verified (m): cocycle identity holds to $1.2\times10^{-15}$ over 500 steps on the Mexican-hat damped map; latched group-tangent exponent $\widehat\lambda(T{=}10,10^2,10^3) = 0.0$ exactly]**

**(b) Retention half-life of a massive mode.** Underdamped ⟺ complex $\lambda$ ⟺ $h > h^*(\gamma)$, with the exact crossover
$$h^*(\gamma) = \big(1-\sqrt{1-\gamma}\big)\sqrt{\tfrac{2}{2-\gamma}} \;=\; \tfrac{\gamma}{2} + O(\gamma^2).$$
- **Underdamped** ($h^* < h < 2$): $|\lambda| = \sqrt{\det A} = \sqrt{1-\gamma}$ — **independent of the mode mass**. Envelope half-life
$$n_{1/2} = \frac{2\ln 2}{-\ln(1-\gamma)} \approx \frac{2\ln 2}{\gamma}\ \text{steps}.$$
*(v1.1 wording rule: quote the exact first form — at $\gamma=0.1$ it gives 13.16 where the approximation gives 13.86, and measured floors land on the exact value (check (k)). The $2\ln2/\gamma$ form appears only here, as the small-$\gamma$ approximation.)*
- **Overdamped** ($0 < h < h^*$): real $\lambda$; perturbing around the latch ($f(\lambda)=\lambda + \tfrac{1-\gamma}{\lambda}$, $f'(1)=\gamma$) gives the slow "memory" eigenvalue and half-life
$$\lambda_{\rm slow} = 1 - \frac{(2-\gamma)\,h^2}{2\gamma} + O(h^4), \qquad n_{1/2} \approx \frac{2\gamma \ln 2}{(2-\gamma)\,\varepsilon^2\mu^2}\;\; \xrightarrow{\ \gamma\ll1\ }\;\; \frac{\gamma\ln 2}{(\varepsilon\mu)^2}, \qquad t_{1/2} \approx \frac{\gamma_c \ln 2}{\mu^2}.$$
**[proven; verified (b): exact eigenvalues match the asymptotic formula to 5×10⁻⁸ and fitted simulation rates to 9 decimals; underdamped $|\lambda| = 0.974679434 = \sqrt{0.95}$ for both $m=1$ and $m=0.25$ (mass-independence); crossover at $h^*=0.025643$ ($\gamma=.05$), $0.111284$ ($\gamma=.2$) confirmed real↔complex]**

**(c) Pseudo-Goldstone exponent.** A small explicit breaking of the protecting symmetry — **of $V_\theta$**, not of $T$ (Prop-17) — lifts a flat direction to spectral mass $\mu^2 = \delta$-curvature$/M_{\rm eff}$ (for a tilt $\delta\cos(n\vartheta)$ along a vacuum circle of radius $r_\ast$: $\mu^2 = \delta n^2/(M_{\rm eff}r_\ast^2) = \delta n^2/F^2$, the Gell-Mann–Oakes–Renner pattern $\mu^2 \propto \delta$, with the **decay constant** $F := \sqrt{M_{\rm eff}}\,r_\ast$ — see the naming rule in §8). Since a weakly-broken channel sits in the overdamped regime ($h \ll h^*$), its retention is
$$\boxed{\; n_{1/2} \;\propto\; \mu^{-2} \;\propto\; \delta^{-1} \;}$$
— **exponent exactly −2 in the mode mass, −1 in the breaking parameter** — *valid only while $\varepsilon\mu < h^*(\gamma)\approx\gamma/2$; beyond the crossover the half-life saturates at the mass-independent underdamped floor $2\ln 2/(-\ln(1-\gamma))$ steps (exact form; §3.3b).* The brainstorm conjecture "half-life ∝ 1/mass²" is confirmed with this precise scope, constant $\tfrac{2\gamma\ln2}{(2-\gamma)\varepsilon^2}$, and saturation. **[proven; verified (b): $n_{1/2}$ measured 1544 → 6165 → 24649 for $\mu^2 = .04 → .01 → .0025$; ratios 3.993, 3.998 vs predicted 4.0]**

**(d) Expanding modes: friction never stabilizes a saddle.** For $\mu^2 = -|k|/m < 0$ ($g := \varepsilon\sqrt{|k|/m}$), $\lambda_+ > 1$ for **all** $\gamma\in[0,1)$ (proof: $\lambda_+<1$ would need $\operatorname{tr}A < 2-\gamma$, but $\operatorname{tr}A = (2-\gamma)(1+g^2/2) > 2-\gamma$). Friction only slows the escape, mirror-symmetrically to memory decay:
$$\lambda_+ \approx 1 + \frac{(2-\gamma)\,g^2}{2\gamma} \quad (g \ll \gamma), \qquad \lambda_+ \approx e^{g} \quad (\gamma \to 0).$$
Combined with governor blindness (Prop-10): **expanding directions are contained by nothing except curvature control of $V_\theta$ (a correct Lyapunov-type regularizer — not the current one, Prop-5) and the relativistic velocity cap.** **[proven; verified (b): $\min_\gamma(\max|\lambda| - 1) = 1.4\times10^{-5} > 0$ over a $\gamma$-grid; exact escape rate 1.000112449 matches asymptotic 1.000112500 and fitted 1.000112449]**

### 3.4 The mode-mass budget table (per step $\varepsilon$, friction $\gamma$)

| band | condition | behavior | half-life (steps) |
|---|---|---|---|
| latch | $\mu = 0$ (symmetry-protected) | frozen displacement, dead momentum | $\infty$ |
| register (overdamped) | $0 < \varepsilon\mu \lesssim \gamma/2$ | slow leak, no oscillation | $\approx \dfrac{2\gamma\ln2}{(2-\gamma)(\varepsilon\mu)^2}$ |
| working memory (underdamped) | $\gamma/2 \lesssim \varepsilon\mu < 2$ | oscillation at $\approx\mu$, mass-independent decay | $\dfrac{2\ln 2}{-\ln(1-\gamma)}$ (exact) |
| unstable (chaos channel) | $\mu^2 < 0$ | escape; friction-slowed, never held | doubling $\approx \dfrac{2\gamma\ln2}{(2-\gamma)(\varepsilon\mu_{\rm im})^2}$ |
| forbidden (stiffness) | $\varepsilon\mu > 2$ | integrator instability | must not exist |

This table **is** the memory–compute–forgetting budget of the ICLR thesis, now with exact constants. Design rules it implies: (1) memory capacity at a given lifetime is set by how much spectral weight $V_\theta$ can park below $\mu \approx \gamma/(2\varepsilon)$; (2) $\gamma$ moves the register/working-memory boundary and the working-memory lifetime *simultaneously* — one knob, two effects; (3) $\varepsilon$ rescales the whole $\mu$-axis (and the forbidden zone) — step size is part of the budget, not just accuracy.

**Metric discipline (mandated; Cor-14):** every half-life in this table is an **envelope** statement. First-crossing lifetimes agree with it only in the overdamped band; in the underdamped band they measure phase transport, not retention. Any program figure or claim must name its retention metric (§3.5).

### 3.5 Corollaries of the budget table (v1.1; surfaced by the Mo audit, re-verified here)

**Cor-13 (critical-damping retention minimum).** At fixed $(\varepsilon,\mu)$ with $h=\varepsilon\mu<\sqrt2$, the spectral half-life $n_{1/2}(\gamma)=\ln2/(-\ln\max|\lambda|)$ is non-monotone in $\gamma$: strictly decreasing on the underdamped side ($=2\ln2/(-\ln(1-\gamma))$), strictly increasing on the overdamped side, with its **minimum exactly at the crossover** $\gamma^*(h)$ — the unique root of $h^*(\gamma)=h$:
$$\gamma^*(h) = 2h\,(1-h) + O(h^3) \;\approx\; 2\,\varepsilon\mu, \qquad n_{1/2}^{\min} = \frac{2\ln 2}{-\ln(1-\gamma^*)} \;\approx\; \frac{\ln 2}{\varepsilon\mu}.$$
(The overdamped-side slope is infinite at $\gamma^*$ — a $\sqrt{\gamma-\gamma^*}$ discriminant, cf. Cor-15 — so the minimum sits exactly at the exceptional point; for $h\ge\sqrt2$ no crossover exists and forgetting is monotone in $\gamma$.) **Forgetting is fastest at critical damping** — the memory-side inversion of control theory's "fastest settling at critical damping." Design rules: memory channels must keep $\gamma$ away from $2\varepsilon\mu$ of their protected content; **trash regions should target it: $\gamma_\phi(q)\approx 2\,\varepsilon\mu(q)$** (Thread-1 spec — couples the friction field to the local curvature scale; input for the `gamma-field-build` task). Honest scope: friction erases *massive-mode amplitude*; exactly-flat directions are latched, not erased, at any $\gamma$ (§3.3a) — a friction-only trash region cannot delete coset-coordinate content (that needs local curvature/tilt or noise), and a single scalar $\gamma_\phi(q)$ is critical for one $\mu$-band at a time. **[proven from §3.3; verified (j): argmin over a 40k $\gamma$-grid coincides with $\gamma^*$ to grid resolution ($|{\rm argmin}-\gamma^*| = 2.3\times10^{-5}$, spacing $2.5\times10^{-5}$); $\gamma^* = 0.039214$ vs $2h(1-h) = 0.039200$ at $h=0.02$; $n^{\min}$: exact 34.65 vs $\ln2/h = 34.66$; endpoints 276.6 / 1630.2 reproduce mo-deep-read's measured 276.6 / 1630]**

**Cor-14 (retention-metric bifurcation).** Two lifetime metrics coexist: the **envelope half-life** $n_{1/2}$ (canonical-amplitude/energy envelope — the retention notion of §3.4) and the **first-crossing time** $n_\times$ (first excursion of a readout beyond a threshold — Mo's protocol). Overdamped, they measure the same rate ($n_\times/n_{\rm pred}\to1$). Underdamped, they **bifurcate**: the readout crosses ballistically within the first quarter-period, $n_\times \propto 1/(\varepsilon\mu) \propto \delta^{-1/2}$, while $n_{1/2}$ saturates at the mass-independent floor $2\ln2/(-\ln(1-\gamma))$ — the two metrics then measure different physics (phase transport vs retention), and single-exponential lifetime predictors (Mo's) fail by up to 5× past the crossover for exactly this reason. **[proven; verified (k): Mo's code-level protocol run on our mode — measured/predicted ratios 1.001, 1.012 deep overdamped (his median 1.013), 2.300 at the EP, 0.933 → 0.435 → 0.187 underdamped, reproducing mo-deep-read Check-6; $d\ln n_\times/d\ln\delta = -0.58$ underdamped (ballistic $-0.5$ + small-$n$ threshold-geometry corrections); envelope floor pinned at 13.2 $= 2\ln2/(-\ln 0.9)$ throughout]**

**Cor-15 (exceptional-point signatures at $h^*$).** The over/underdamped crossover is a **defective (Jordan) point** of the mode map $A$: at $h=h^*(\gamma)$ the eigenvalues merge at $\lambda=\sqrt{1-\gamma}$ with a single eigenvector ($(A-\lambda I)\ne0$ but $(A-\lambda I)^2=0$). Consequences: (i) **frequency onset** $\varphi = C(\gamma)\,\sqrt{h-h^*} + O(h-h^*)$ with the closed-form prefactor (new in v1.1; one line from the discrete dispersion $\cos\varphi = (2-\gamma)(1-h^2/2)/(2\sqrt{1-\gamma})$):
$$C(\gamma) = \sqrt{\,(2-\gamma)\,h^*(\gamma)\,/\sqrt{1-\gamma}\,}\,;$$
(ii) **algebraic decay prefactor**: $\lVert A^n z\rVert \sim (1+\kappa n)\,\lambda^n$ near the EP, so measured lifetimes exceed pure-exponential predictions there — by a metric- and phase-dependent factor (2.3× under Mo's protocol, ≈2.8× for amplitude first-crossing from a position kick). Sharp, second-order-only observables for the V2 experiment. **[proven; verified (l): $C(0.1) = 0.324724$ from the formula vs measured $\varphi/\sqrt{h-h^*} = 0.324740$ at $h-h^* = 10^{-5}$ (drifting to 0.3398 by $10^{-2}$ — the $O(h-h^*)$ term), matching mo-deep-read's empirical constant 0.3247; nilpotency $\lVert(A-\lambda I)^2\rVert = 1.1\times10^{-17}$ with $\lVert A-\lambda I\rVert = 0.1$; $\lVert A^nz\rVert/\lambda^n$ growth ratios 1.76 → 1.87 (→ 2 = linear-in-$n$); EP delays: 98 vs spectral 34.7 in (j), ratio 2.300 in (k)]**

---

## 4. Symmetry and Goldstone engineering

### 4.1 Noether in the H-CLU **[proven; verified (g), (g′)]**

> **Corrigendum, 2026-07-09.** The previous version of this subsection carried a false corollary — *"if $M$ does not isotropize, the channel is pseudo-Goldstone with $\mu^2\propto$ mass splitting"* — under a `[proven; verified (g)]` tag that over-covered (check (g) tested only the *charge* half). It is replaced below by **Prop-17 (kinetic-spurion blindness)**, which is strictly stronger and points the other way: $\mu^2\equiv0$ under **any** anisotropy. See also the retraction of check (g)'s "2.6" at the end of this subsection.

Let a one-parameter group $g_s$ act linearly on $q$ and lift to phase space by $(q,p)\mapsto(g_s q,\ g_s^{-\top}p)$. If $V_\theta(g_sq)=V_\theta(q)$ **and** $T(g_s^{-\top}p) = T(p)$, then $H$ is invariant and the **Noether charge** $Q_X(q,p) = p^{\!\top} X q$ (with $X = \frac{d}{ds}g_s|_{0}$; e.g. angular momentum $L = q_1p_2 - q_2p_1$ for $SO(2)$) is conserved by the continuous flow.

**Kinetic isotropy condition — it is a condition on the *current*, not on the *register*.** For all three kinetic modes, $\nabla_pT \parallel M^{-1}p$, so $T$-invariance under a rotation channel requires the **inertial masses to be equal within that channel** ($M$ commuting with the group action; by Schur, $M \propto I$ on each irrep — "members of a multiplet have a common mass", exactly as in HEP). An $SO(2)$ channel over coordinates $(q_1,q_2)$ **requires $M_1 = M_2$ for $Q_X$ to be conserved**; otherwise the kinetic term explicitly breaks the symmetry of $H$ no matter how equivariant $V_\theta$ is. **[proven; verified (g): with $M=(1,1)$ the charge is conserved to 3.0×10⁻¹⁴ over $10^5$ steps; with $M=(1,2)$ it is not conserved]**

What that breaking does **not** do is give the channel a mass:

> **Prop-17 (kinetic-spurion blindness).** Let $V_\theta$ be $\mathcal G$-invariant with vacuum $q^\ast$ and stiffness $K = \nabla^2V_\theta(q^\ast)$, and let $M_{\rm eff}\succ0$ be **any** inertia (not necessarily commuting with the group action). With the spectral-mass matrix $W = M_{\rm eff}^{-1/2}KM_{\rm eff}^{-1/2}$ (Def-2),
> $$\ker W = M_{\rm eff}^{1/2}\ker K,\qquad \operatorname{rank}W = \operatorname{rank}K,\qquad \text{inertia}(W) = \text{inertia}(K).$$
> Hence **every flat direction keeps $\mu^2 = 0$ exactly, for any anisotropy**: the vacuum manifold, the channel count $\dim(\mathcal G/\mathcal H)$ (§4.2), and the $\gamma>0$ latch (§3.3a) are untouched, and the latched *physical* $q$-direction is $\ker K$ itself — not even rotated.
>
> *Proof.* $W = C^\top K C$ with $C = M_{\rm eff}^{-1/2}$ invertible, and congruence by an invertible matrix preserves inertia (Sylvester), hence the number of zero eigenvalues. Explicitly $Wv=0 \iff KM_{\rm eff}^{-1/2}v = 0 \iff v \in M_{\rm eff}^{1/2}\ker K$. In normal-mode coordinates $x = M_{\rm eff}^{1/2}q$ the flat block of the one-step map is exactly $\left(\begin{smallmatrix}1&\varepsilon\\0&1-\gamma\end{smallmatrix}\right)$ (§3.3a), whose physical $q$-direction is $M_{\rm eff}^{-1/2}\ker W = \ker K$. *Globally:* every point of the vacuum manifold with $p=0$ is an exact fixed point of the damped-Verlet map ($\nabla V=0 \Rightarrow p_{1/2}=p=0 \Rightarrow q'=q$) for **any** $M$. ∎

**What the kinetic spurion perturbs instead: the Noether current, boundedly.** At $\gamma=0$, $\dot Q_X = p^\top X M^{-1} p \ne 0$ when $[M,X]\ne0$ (for $SO(2)$: $\dot L = p_1p_2(M_1^{-1}-M_2^{-1})$). But $H$ is conserved and $V_\theta$ is coercive, so the orbit stays in a compact set and $|Q_X| \le \sup|q|\,|p|$: **secular drift of the charge is impossible — the violation is a bounded excursion.** On the vacuum orbit it is exactly periodic and closed-form: with the induced coset metric $F^2(\vartheta) = r_\ast^2(M_1\sin^2\vartheta + M_2\cos^2\vartheta)$ and $E = \tfrac12 F^2\dot\vartheta^2$ conserved,
$$Q = F^2\dot\vartheta = \sqrt{2E}\,F(\vartheta) \;\in\; \Big[\sqrt{2E}\,r_\ast\sqrt{M_{\min}},\ \sqrt{2E}\,r_\ast\sqrt{M_{\max}}\Big],$$
a bounded oscillation of amplitude $\sqrt{2E}\,r_\ast(\sqrt{M_{\max}}-\sqrt{M_{\min}})$ with period **half a revolution**.

**Design consequence (this replaces the previous design rule).** An anisotropic channel **still latches, with the same infinite half-life**; what it loses is a *conserved* write current — the write gain becomes $\vartheta$-dependent. Tie the channel masses if you want a clean, $\vartheta$-independent write current (and, at $\dim(\mathcal G/\mathcal H)\ge2$, a degenerate pNG multiplet); do **not** tie them in the belief that the register would otherwise decay. Falsifiable corollary for V2, restated correctly: on symmetric data, does learned $M$ isotropize within the channel? **To detect kinetic symmetry breaking, measure the charge law — not the Hessian and not the latch** (both are blind to it by Prop-17). **[proven; verified (g′): flat mode $\mu^2 = 0.0$ **exactly** (bit-level) for 8 random anisotropic *diagonal* $M$ (the coded `log_mass`); $|\mu^2|\le7.2\times10^{-16}$ and flat $q$-directions coincide with $\ker K$ to $3.3\times10^{-16}$ for 6 random non-diagonal SPD $M$ (cond ≤ 14); coset angle frozen to $0.0$ exactly over $2\times10^4$ steps at $\gamma=0.05$ with $M=(1,2)$ and $M=(0.31,4.7)$; ring closed form: amplitude ratio 1.0007–1.0064, elliptic half-period ratio 0.9994–1.0005]**

⚠ **Retraction of a number.** The "charge drifts by 2.6" quoted in check (g) is a **running-supremum excursion** $\max_n|L_n-L_0|/|L_0|$, **not a drift rate**. Re-run: it is not reproducible across two algebraically identical spellings of $\nabla V$ (2.625 vs 2.734 at $10^5$ steps — the off-ring orbit is a chaotic rosette), it grows with window length as any running supremum must (2.96 at $10^6$ steps), while the **envelope is stationary** (per-decile $\sup|L|$ slope $+6.1\times10^{-4}$; midrange slope $-6.7\times10^{-4}$/decile $= 5.7\times10^{-3}$ of the range) and $\sup_n|L_n| = 0.582$ sits under the compact-set bound $0.820$. Quote it as "$O(1)$ charge non-conservation," never as drift.

**Discrete exactness.** The Verlet map conserves $Q_X$ **exactly** (machine precision, any $\varepsilon$), because each substep does: the kick exerts zero torque (equivariant $\nabla V$), the drift has $\nabla T \parallel M^{-1}p$ (isotropy), and quadratic charges of this form are preserved by both shears. With friction, the damping step gives the **exact decay law**
$$Q_{X,n} = (1-\gamma)^n\, Q_{X,0}.$$
**[proven; verified (g): $\max_n |L_n - (1-\gamma)^nL_0|/|L_0| = 9.3\times10^{-16}$]**

**Charge vs. coordinate (where memory actually lives).** The retained quantity under dissipation is **not** the conserved charge — friction kills $Q_X$ geometrically. What persists is the **conjugate coset coordinate** (the angle $\vartheta$ along the vacuum manifold): the state relaxes onto the orbit, the angle freezes (§3.3a latch, globally on the orbit, not just linearized). Memory lives on $G/H$; the charge is the *write current*. **[proven; verified (g): Mexican-hat run, $\gamma=0.005$: $\vartheta$ frozen to 2.3×10⁻⁹ over the last 4000 steps, radius on the vacuum circle to 3.4×10⁻¹⁰]**

### 4.2 Channel counting

If $V_\theta$ (and $T$) are invariant under a compact group $\mathcal G$ and the attractor orbit spontaneously breaks $\mathcal G \to \mathcal H$ (the stabilizer of $q^\ast$), the flat directions at $q^\ast$ along the orbit number
$$\#\{\text{protected memory channels}\} = \dim \mathcal G - \dim\mathcal H = \dim(\mathcal G/\mathcal H).$$
(The SM's "3" is the instance $SU(2)\times U(1)\to U(1)$.) Choosing $\mathcal G$, the representation, and $\mathcal H$ is **channel allocation by symmetry engineering**. Caveat: this counts orbit directions; additional accidental flat directions of a learned $V_\theta$ are possible (and are not protected). **[proven at the level of the counting; the engineering is Hyp-1]**

### 4.3 Two parameterization routes (the colleague-collaboration surface)

- **Linear realization ("SMEFT-like").** Latent in a linear rep of $\mathcal G$; $V_\theta = f(\text{invariants})$ (invariant/equivariant network). Radial (Higgs-like) modes are massive; angular modes are the Goldstones; explicit breaking enters as spurion terms with definite transformation properties. *Implementable today* (constant diagonal $M$ + equivariant $V_\theta$; respect kinetic isotropy per irrep).
- **Nonlinear realization ("HEFT-like").** Coordinates on the coset: $q = \rho\, U$, $U \in \mathcal G/\mathcal H$; kinetic term = coset (sigma-model) metric with decay constant $F = \sqrt{M_{\rm eff}}\,r_\ast$ (= stiffness of the memory manifold; **not** the orbit radius $r_\ast$ — §8); Goldstones are the coordinates themselves; breaking enters through spurion-dressed operators. **Implementation warning:** a sigma-model kinetic term is a *position-dependent mass matrix* $M(q)$ — this breaks the separability assumption of §2 (the current explicit integrator is no longer symplectic) and is not representable by the constant diagonal `log_mass`. Routes: (i) constrained integrator (RATTLE/projection onto the orbit), (ii) implicit generalized leapfrog, or (iii) stay ambient-linear with a heavy radial mode — since **HEFT is the $\mu_{\rm radial}\to\infty$ decoupling limit of the linear realization**, route (iii) approximates (i) with corrections controlled by $1/\mu_{\rm radial}^2$. **[proven statements about representability; route choice is Hyp-2]**
- **EFT of memory (sketch, flagged for the colleague).** Power counting in $E/\mu_{\rm heavy}$: integrating out the **fast** (large-$\mu$) sector leaves an effective theory on the memory manifold with higher-dimensional operators suppressed by $\mu_{\rm heavy}^{-2}$; leading effects = anharmonic corrections to retention and mode–mode "memory crosstalk"; explicit breaking organized by spurions ⇒ pseudo-Goldstone masses ⇒ §3.3(c) lifetimes. Dictionary: cutoff ↔ stiff sector; Wilson coefficients ↔ learned deviations of $V_\theta$; GMOR ↔ lifetime–breaking law. **[conjectured program, not results]**

---

## 5. The mass matrix as budget allocator

**Def-3 (gas-of-particles reading).** Read the $d$-dimensional latent as $d$ particles $(q_i,p_i)$ each with inertial mass $M_i$ (diagonal $M$ = the code today), or $N$ units of dimension $d_i$ with block masses (lattice, §7). Nothing in §2–§4 changes; this is an interpretation layer that makes $M$'s roles explicit.

**Honest deflation first [proven].** At $\gamma\ge0$ and **linear order, constant $M$ is a gauge choice**: the symplectic rescaling $(\tilde q,\tilde p) = (M_{\rm eff}^{1/2}q,\ M_{\rm eff}^{-1/2}p)$ absorbs it, leaving only the spectral content $W = M_{\rm eff}^{-1/2}KM_{\rm eff}^{-1/2}$. Claims of the form "mass does X" must therefore route through one of $M$'s **irreducible roles**:

1. **Causal budget [proven, Prop-1]:** per-particle speed cap $v_i^{\max} = c/\sqrt{M_i}$ — *not* absorbable (the bound is anisotropic in physical coordinates, where data lives).
2. **Relativistic regime assignment [proven, Prop-1]:** crossover $p_i^* = m_0c\sqrt{M_i}$ — at a common momentum scale, **light particles run relativistic (governor-dominated), heavy ones stay Newtonian**.
3. **Relative curvature pricing [proven]:** one shared $V_\theta$, many particles: $\mu_i^2 = K_{ii}/(M_{{\rm eff},i})$ locally — $M$ redistributes a *shared* landscape's timescales: heavy+flat = deep memory, light+curved = fast scratch. With the exact lifetimes of §3.4.
4. **Boost stiffness / semantic inertia [proven given the spec below]:** under the **mass-weighted squeeze** (define it this way, or the claim is false) $S^{(M)}_\zeta := \mathcal N^{-1} S_\zeta \mathcal N$ with $\mathcal N = \mathrm{diag}(M_{\rm eff}^{1/2}, M_{\rm eff}^{-1/2})$, the position response is $\partial q_i'/\partial\zeta\big|_{\zeta=0} = p_i/M_{{\rm eff},i}$: **light particles reframe strongly, heavy ones barely.** A *raw-coordinate* squeeze ($q' = q\cosh\zeta + p\sinh\zeta$) mixes mass-blind — the Thread-3 L0 statement "the mass matrix automatically makes a global boost region-differential" holds **iff** the squeeze is defined mass-weighted. Spec for `experiment-engineer`: implement $S^{(M)}_\zeta$. **[proven; verified (c): $S^{(M)}$ symplectic to 2.2×10⁻¹⁶]**
5. **Learning-dynamics prior [design hypothesis, Hyp-3]:** `log_mass` is a cheap $d$-parameter knob that gradient descent can move without touching $V_\theta$ — the *learnability* of budget reallocation. Falsifiables (sharpened from Thread 5): (i) trained CLUs develop non-trivial $\mu$-spectra correlated with feature timescales — **measure $\mathrm{eig}(M_{\rm eff}^{-1}\nabla^2V_\theta(q^\ast))$ at attractors, not $M$ alone** (note for `results-analyst`/mass-spectrum-peek: $M$ is half the object); (ii) under $S^{(M)}$-boosts, per-mode displacement scales as $M_i^{-1}$; (iii) mass-hierarchical lattices beat mass-uniform at matched parameters.
   **Hyp-3 empirical status (v1.1; wave-1/2 evidence).** (i) *Tested twice — supported with qualifications.* The mass gradient signal is real, universal, and data-aligned (mass-spectrum-peek: every relativistic checkpoint moved 784/784 components, KS $p\approx0$; global lightening with a hyperparameter-stable ink-light/border-heavy pattern, cross-variant Pearson 0.40–0.93, Spearman vs pixel variance −0.44…−0.72; identity-mode control bit-identical to init validates the pipeline) — but **near-uniform in magnitude**: $\sigma_{\rm struct}$ = 2–23% of the init σ under wake–sleep, and independently log-std ≈ 0.08 across 23 fresh generative-PCD models (v1-l0-gate). **A mass hierarchy does not emerge from current training at this scale. Doctrine: hierarchy is designed-in (banded $M$ — V3) or induced (mass-aware objectives, multi-timescale data) — not awaited.** (ii) ***Untestable-until-banded:*** with near-uniform learned $M$, $1/M_{\rm eff}$ has no dynamic range — v1-l0-gate measured Spearman$(\log|\Delta q_i|,\ \log 1/M_i) = 0.03$ ($n=4608$): no x-axis, not a refutation; the operator-level response $\partial q_i'/\partial\zeta|_0 = p_i/M_{{\rm eff},i}$ is exactly unit-verified ($\le10^{-10}$). Re-test on designed-banded models only. (iii) open — now the V3 design bet.

**Timescale hierarchy & shells [proven math, Hyp-4 architecture].** Fixed shared curvature ⇒ mode frequency $\propto M^{-1/2}$: heavy sectors are slow (IR/backbone), light sectors fast (UV/perception). "Shells by mass" = **spectral banding of $\mu$**, implemented by banding $M$ (and/or curvature). Escalation (§7.5) = moving a query from fast bands to slow bands.

**Composite particles [Hyp-5].** Block-diagonal $M$ = feature groups sharing inertia ("bound states"). Code change (current `log_mass` is diagonal); the linear theory above extends verbatim with block $M_{\rm eff}$.

**Coarse-grained inference — corrected EFT bridge [standard theory; nomenclature fix].** "Integrating out" removes the **fast** (large-$\mu$) sector — which is the *light-inertial-mass or stiff* one — via adiabatic elimination (Born–Oppenheimer: fast modes instantaneously minimize given slow coordinates ⇒ effective potential $V_{\rm eff}$ on the slow/heavy sector) plus, at second order, Mori–Zwanzig memory kernels and effective noise. The earlier phrase "integrate out heavy modes" conflated $M$ with $\mu$ (Def-2): in allocator language one *keeps* the heavy-$M$ (slow) sector and eliminates the fast one; in EFT language the eliminated fast sector is the "heavy-$\mu$" one. Same operation, opposite adjectives — use Def-2's vocabulary. **[flagged correction to Thread 5 wording]**

---

## 6. ⚠ The interference problem (mandated open problem)

**Problem statement (Open-2).** All particles/units sharing a potential share its parameters: $V_\theta: \mathbb{R}^{Nd}\to\mathbb{R}$. A gradient update serving memory/task $A$ moves $V_\theta$ *everywhere*, including where memory $B$'s structure lives — catastrophic interference in $\theta$-space even when $A$ and $B$ are disjoint in state space.

**Formalization.** For contrastive-divergence training (our objective), the parameter update from a wake/sleep pair is $\delta\theta = -\eta\,[\nabla_\theta V_\theta(q_{\rm wake}) - \nabla_\theta V_\theta(q_{\rm sleep})]$, and the induced change of the landscape at any probe point $q$ is exactly
$$\delta V(q) = -\eta\,\big[\Theta(q, q_{\rm wake}) - \Theta(q, q_{\rm sleep})\big], \qquad \boxed{\ \Theta(q,q') := \nabla_\theta V_\theta(q)^{\!\top}\, \nabla_\theta V_\theta(q')\ }$$
— the **potential NTK** (scalar kernel). Dynamics-level interference (what moves states) is governed by the force kernel $\Theta_F(q,q') = \nabla_q\nabla_\theta V^{\top}\,\nabla_{q'}\nabla_\theta V$ ($d\times d$). **Interference of $B$ on $A$ := $\lVert\Theta(q_A, \Omega_B)\rVert / \lVert\Theta(q_A,q_A)\rVert$.** The design problem: make $\Theta$ *controllably* near-block-diagonal over the memory layout — not zero (shared structure is also generalization; P3), but gated.

**Mechanism catalog (assessed):**

| # | mechanism | effect on the kernel | assessment |
|---|---|---|---|
| (i) | **modularity** (per-unit $V_i$, coupling only via $V_c$) | $\Theta$ exactly block-diagonal in unit-owned parameters; cross-talk only through $\theta_{V_c}$, bounded by coupling magnitude/curvature | the only *hard* firewall; this is why V3 scales by mass **and** size **[proven structure]** |
| (ii) | **symmetry/irrep firewalls** | at an $\mathcal H$-symmetric point, $\nabla^2V$ commutes with the action ⇒ block-diagonal over isotypic components (Schur): **modes in different irreps do not mix at linear order** [proven]. The *learning* (NTK) version additionally requires the parameterization to factor across irreps — architecture-dependent | dynamical firewall proven; NTK firewall **[design hypothesis, Hyp-6]** |
| (iii) | **block/separable $V$** ($V=\sum_b V_b + $ weak cross-terms) | explicit zero blocks inside one unit | (i) inside a unit; same math **[proven structure]** |
| (iv) | **local parameterizations** (RBF/local experts) | compactly-supported $\Theta(q,q')$ ⇒ edits are local in state space | strongest per-parameter locality; costs capacity/smoothness; classic interpolation–generalization trade **[evidenced, literature]** |
| (v) | **replay** (the CD buffer) | doesn't shrink $\Theta$; constrains *updates*: sleep negatives at persistent hallucinations + wake positives at old data ≈ projecting $\delta\theta$ to preserve stored energies (GEM-flavored) | anti-interference machinery already present in PCD; persistence of the buffer matters (known §7.4 discrepancy) **[evidenced]** |
| (vi) | **curvature protection ("mass for weights", EWC-like)** | penalty $\tfrac12(\theta-\theta^\ast)^{\top}F(\theta-\theta^\ast)$ multiplies updates by $(F+\lambda I)^{-1}$-type preconditioner along protected directions | **the dual is real but it is *stiffness*, not inertia**: a Fisher penalty is a spring in $\theta$-space (a *spectral mass* $\mu_\theta^2$ for weights), while true $\theta$-inertia is momentum-SGD's $\beta$. By §3's own law, large $\mu_\theta$ ⇒ short-lived deviations ⇒ protected memory — the q-space and θ-space stories unify under the **spectral-mass** language, not the inertial-mass one. Cute → precise, once restated this way **[proven analogy at the linear level]** |

**Success criterion for this section (per task):** sharp problem statement + assessed catalog — done; no solution claimed. Recommended first measurement (V3): estimate $\Theta(q_A,q_B)$ across stored attractors of a trained unit vs. a 2-unit modular lattice at matched parameters. *(v1.1: the `v3-lattice-build` task instantiates exactly this — it will provide the first $\Theta(q_A,q_B)$ modularity measurement; no math change.)* **[Open-2 stays open]**

---

## 7. CLU-Net: the lattice formalism

### 7.1 The object

**Def-4 (CLU lattice).** Units $i = 1..N$ with states $(q_i,p_i)\in\mathbb R^{2d_i}$ on a coupling graph $E$; **one joint Hamiltonian**
$$H_{\rm net}(q,p) = \sum_{i} \Big[T_i(p_i) + V_i(q_i)\Big] + \sum_{(ij)\in E} V_c(q_i, q_j),$$
evolved by the *same* map §2.2 on the concatenated state ("depth" = rollout time, "width" = $N$, "architecture" = $E$).

### 7.2 What survives composition **[proven]**

Everything in §2–§3 survives with $d \to \sum_i d_i$, **iff**:
1. **Kinetic separability:** $T_{\rm net} = \sum_i T_i(p_i)$ — couplings $V_c$ must be **position-only**. Momentum/velocity coupling (e.g. "magnetic", boost-conditioned terms) makes $H$ non-separable ⇒ the explicit integrator loses exact symplecticity ⇒ integrator upgrade required (flag any such proposal).
2. **One global step $\varepsilon$**, kicks using the full joint force $\nabla_{q_i}(V_i + \sum_j V_c)$. (Multirate/per-unit steps: possible but needs dedicated splitting analysis — not covered.)
3. **Damping:** per-unit constants $\gamma_i$ give $\det J = \prod_i (1-\gamma_i)^{d_i}$; **uniform** $\gamma$ preserves global conformal symplecticity with factor $(1-\gamma)$; **heterogeneous** $\gamma_i$ (or fields, below) keep the exact determinant/volume statement but lose the global pairing of Prop-4 — state depth-stability claims accordingly.

Then: symplecticity/conformality (Prop-3), pairing (uniform $\gamma$; Prop-4), shadow $\tilde H_{\rm net}$ (Prop-7), taxonomy & lifetimes on the **joint** spectrum $\mu^2 = \mathrm{eig}\big(M_{\rm eff,net}^{-1}\nabla^2(\sum V_i + \sum V_c)\big)$ (§3), Noether for joint symmetries with **equivariant $V_c$** and channel-isotropic masses (§4.1) — all hold by construction, at any $N$.

**Inter-unit communication has a mass.** Two units with $SO(2)$-symmetric $V_i$ and a $V_c$ invariant only under *simultaneous* rotation: the diagonal channel stays an exact Goldstone latch (shared memory), while the **relative** angle acquires $\mu_{\rm rel}^2 \propto$ coupling curvature $\kappa_c/M_{\rm eff}$: sync timescale $\propto \kappa_c^{-1/2}$, relative-information retention $\propto 1/\kappa_c$ (overdamped, §3.3c). **Coupling strength literally prices communication speed against relative-memory lifetime.** **[proven at quadratic order]**

### 7.3 Friction fields and trash regions (Thread-1 slot)

**Def-5 (friction field).** Replace scalar $\gamma$ by $\gamma_\phi(q) \in [0, \gamma_{\max}]$, applied as $(q,p) \mapsto (q,\ (1-\gamma_\phi(q_{n+1}))\,p)$ after the Verlet substeps.

**Prop-11 (position-gated volume contraction, exact).** The damping Jacobian is block-triangular, so regardless of $\nabla\gamma_\phi$:
$$\det D\Phi = \big(1-\gamma_\phi(q_{n+1})\big)^{d}.$$
Phase-space volume is destroyed **exactly and only where $\gamma_\phi > 0$** — the "event horizon" is the superlevel set of $\gamma_\phi$; outside it the dynamics is exactly conservative. Conformality (and hence sval pairing) is lost wherever $\nabla\gamma_\phi \ne 0$; the determinant statement is what remains, and it is exact. This is the clean classical open-system statement ("symplectic bulk + localized non-unitary channels") for the trash-region program. **[proven; verified (h): $|\det J - (1-\gamma(q'))^d| \le 1.1\times10^{-16}$ with complex-step Jacobians]**

### 7.4 Wormholes

Wormholes = sparse non-local coupling terms $V_c^{\rm wh}(q_i,q_j)$ on distant pairs, energy-gated. **Smooth** gates (gate value a smooth function of state through the potential) keep everything in §7.2. **Hard top-$k$ selection** makes $H$ piecewise-defined: exactly symplectic within a selection epoch, with **energy jumps at switching times** (bounded by the gate magnitude) — account for them explicitly (an energy-budget ledger per switch) or smooth the gate. **[proven structure; ledger design is Hyp-7]**

### 7.5 Squeezes (boosts) and the escalation cascade

**Def-6 (squeeze).** Per conjugate pair $i$, rapidity $\zeta_i$:
$$S_\zeta:\quad q_i' = q_i\cosh\zeta_i + p_i\sinh\zeta_i,\qquad p_i' = q_i\sinh\zeta_i + p_i\cosh\zeta_i,$$
the non-compact (hyperbolic) directions of $Sp(2d)$; **mass-weighted version** $S^{(M)}_\zeta = \mathcal N^{-1}S_\zeta\mathcal N$ (§5.4).
**One-line symplecticity:** $S^{\top}\Omega S = (\cosh^2\zeta - \sinh^2\zeta)\,\Omega = \Omega$, hence $\det S = 1$; conjugation by the symplectic $\mathcal N$ preserves this. **[proven; verified (c): $\lVert S^\top\Omega S-\Omega\rVert \le 2.2\times10^{-16}$, det $=1$; also composed with a Verlet-step Jacobian: $2.2\times10^{-16}$]**

**Prop-12 (retry certificates).** (C1) *Structure:* squeezes are symplectic, so any composition (relax → squeeze → relax…) stays in the (conformally-)symplectic class — every guarantee of §2 applies verbatim to the composed inference trajectory; a retry cannot leave the stability class. (C2) *Bounded energy injection:* in mode-normalized coordinates of a quadratic sector, $H(S_\zeta z) \le e^{2|\zeta|}\, H(z)$ (exact bound from $\cosh 2\zeta + |\sinh 2\zeta| = e^{2|\zeta|}$); bounded rapidity ⇒ bounded injection ⇒ the governor (§2.5) re-absorbs it. For general learned $V_\theta$ the bound is local/compact-set — "cannot destabilize" means C1+C2, **not** that energy never rises. **[proven (quadratic); evidenced (general)]**

**Def-7 (escalation cascade).** Shells $k=1..K$ = mass sectors or units (§5, §7.2). Inference policy for query $x$, retry budget $B$:
1. encode into shell $k$; **relax** under the governor toward $E^*_k$;
2. **residual** $R = H_k(z_{\rm settled}) - \tau_k(x)$ with $\tau_k$ a *learned, training-time-calibrated* margin (calibration objective lives in training — Thread 3; out of scope here);
3. $R \le 0$: answer = read-out of $q$. Else if retries $< B$: apply $S^{(M)}_{\zeta_b}$ (line-search or scheduled rapidity), go to 1 — each retry certified by Prop-12;
4. else **shell-jump** $k{+}1$ via a lift $\Lambda_{k\to k+1}$ (warm-started from the best boosted frame; $\Lambda$'s spec is open — projection vs. coupling-mediated hand-off, Hyp-8).

This is the formal object V1's gate experiments instantiate; its physics content is exactly Props 1–4, 12.

**Def-7 empirical status (v1.1 — v1-l0-gate: single seed, single 32-dim shell, MQAR associative recall).** The *gate* half works: within-model residual energy ranks failures (AUROC 0.65–0.79 wherever failures exist), but raw $R$ is **not comparable across models** (pooled raw AUROC 0.33) — a learned per-instance $\tau$ is empirically mandatory, exactly as Thread 3 anticipated — and $\tau$-gated escalation is a real compute allocator (≈8× relaxation steps saved at ≥ full-budget accuracy on easy/moderate levels). The *retry* half showed **no recovery advantage at that scale**: $S^{(M)}$ retries ≈ raw squeezes ≈ kinetically-matched random kicks ≈ relax-longer (pooled recovery 0.140 vs 0.149), and un-gated retries hurt easy levels ($\tau$-gating repairs that). Prop-12's certificates are untouched — they certify *safety* (structure preservation, bounded energy injection), never recovery power. Scope caveats: learned $M$ was near-uniform, so $S^{(M)}\approx S$ operationally (Hyp-3(ii) untestable there); the hardest level was storage-limited; single seed. Program decision (Head, 2026-07-07): the V1 headline pivots to *calibrated energy-gated compute allocation*; squeeze retries are parked for V3-scale multi-shell / mass-banded experiments — where Def-7's shell-jump step $\Lambda_{k\to k+1}$ (never exercised at single-shell scale) first becomes testable.

---

## 8. Notation table (import verbatim)

| symbol | meaning |
|---|---|
| $z=(q,p)$, $d$ | latent state (position, momentum), latent dimension; lattice: $(q_i,p_i)$, $d_i$ |
| $H(q,p) = T(p)+V_\theta(q)$ | Hamiltonian (always separable); $V_\theta$ includes confinement $\alpha\lVert q\rVert^2$ where present ($\alpha=0.05$, `mlp`) |
| $T(p)$ | kinetic term; modes `newtonian_identity` / `newtonian_learned` / `relativistic` $= c\sqrt{p^\top M^{-1}p+m_0^2c^2}$ |
| $M$, $m_0$, $c$ | learned diagonal inertial mass; rest mass; causal speed |
| $M_{\rm eff}$ | rest-inertia: $I$ / $M$ / $m_0M$ per kinetic mode |
| $\varepsilon$, $\gamma$, $\gamma_c$ | step (`dt`); per-step friction $\in[0,1)$; rate $-\ln(1-\gamma)/\varepsilon$ |
| $\Phi_{\varepsilon,\gamma}$, $J=D\Phi$, $\Omega$ | dissipative Verlet map; its Jacobian; symplectic form matrix |
| $\sigma_i$ | singular values of $J$; pairing $(\sigma,(1-\gamma)/\sigma)$ |
| $K=\nabla^2V(q^\ast)$, $W$, $\mu_k$, $h_k$ | stiffness matrix; $M_{\rm eff}^{-1/2}KM_{\rm eff}^{-1/2}$; **spectral (mode) mass** $=\sqrt{\lambda_k(W)}$; dimensionless step $\varepsilon\mu_k$ |
| $h^*(\gamma)$ | over/underdamped crossover $(1-\sqrt{1-\gamma})\sqrt{2/(2-\gamma)} \approx \gamma/2$ |
| $\gamma^*(h)$, $C(\gamma)$ | critical damping: root of $h^*(\gamma)=h$, $= 2\varepsilon\mu(1-\varepsilon\mu)+O(h^3)$ — retention minimum & EP (Cor-13/15); EP frequency-onset prefactor $\sqrt{(2-\gamma)h^*/\sqrt{1-\gamma}}$ |
| $n_{1/2}$, $t_{1/2}$, $n_\times$ | retention half-life (steps, time) — §3.4 table (**envelope** metric); $n_\times$ = first-crossing lifetime (Mo protocol) — name your metric (Cor-14) |
| $v^{\max}_i$, $p^*_i$ | causal cap $c/\sqrt{M_i}$; relativistic crossover momentum $m_0c\sqrt{M_i}$ |
| $\mathcal G \to \mathcal H$, $X$, $Q_X$, $r_\ast$, $F$, $\delta$ | symmetry breaking pattern; generator; Noether charge $p^\top Xq$; **vacuum radius** $r_\ast$ (the condensate/order parameter $\Sigma$); **decay constant** $F := \sqrt{M_{\rm eff}}\,r_\ast$; explicit-breaking parameter |
| $T$, $T_{{\rm eff},i}$, $\sigma_i^\star$ | temperature (energy units); code-noise effective temp $2\varepsilon T/((2-\gamma)M_{{\rm eff},i})$; exact-FDT noise $\sqrt{M_{{\rm eff},i}T\gamma(2-\gamma)}$ — **Newtonian modes only** (Prop-9/9′) |
| $\Theta$, $d\Theta$ | relativistic-sampling parameter $\Theta:=T/(m_0c^2)$; the **$d$-dimensional control parameter is $d\Theta$** (one shared square root over $d$ coordinates). Non-relativistic $\iff d\Theta\ll1$ |
| Maxwell–Jüttner | the relativistic Gibbs momentum marginal $\propto e^{-T(p)/T}$ — non-Gaussian, exponential tails; **not** what the coded O-step samples (Prop-9′) |
| $E^*$, $s$ | governor target energy and sensitivity; $\gamma_n = s\tanh(\max(0,H-E^*))$ |
| $\tilde H$ | shadow Hamiltonian (Prop-7) |
| $S_\zeta$, $S^{(M)}_\zeta$, $\zeta$, $\mathcal N$ | squeeze; mass-weighted squeeze $\mathcal N^{-1}S_\zeta\mathcal N$; rapidity; $\mathrm{diag}(M_{\rm eff}^{1/2},M_{\rm eff}^{-1/2})$ |
| $H_{\rm net}$, $E$, $V_c$, $\kappa_c$ | lattice Hamiltonian; coupling graph; coupling potential; coupling curvature |
| $\gamma_\phi(q)$ | learned friction field (trash regions); $\det J = (1-\gamma_\phi(q'))^d$ |
| $\Theta(q,q')$, $\Theta_F$ | potential NTK $\nabla_\theta V^\top\nabla_\theta V$; force kernel (interference, §6) |
| $R$, $\tau_k$, $B$, $\Lambda_{k\to k+1}$ | residual energy; learned margin; retry budget; shell-lift operator |

**Terminology rules:** "mass" unqualified is forbidden in program docs — say **inertial mass $M$** or **spectral mass $\mu$** (Def-2). "Heavy" follows the same rule. The unbroken subgroup is $\mathcal H$ (calligraphic); the Hamiltonian is $H(q,p)$.

**Decay-constant naming rule (corrigendum, 2026-07-09).** Earlier drafts called the *orbit radius* "the decay constant $f$". The object that plays $f_\pi$'s role — the one appearing in the current relation and in GMOR — is
$$F := \sqrt{M_{\rm eff}}\;r_\ast, \qquad\text{so that}\qquad Q = F^2\,\dot\vartheta \ \text{ on the vacuum manifold},\qquad \mu^2F^2 = \delta n^2 .$$
Reserve **$F$** for the decay constant and **$r_\ast$** for the vacuum radius (= the condensate $\Sigma$, the order parameter). F5's *formulae* were already correct ($\mu^2 = \delta n^2/(M_{\rm eff}r_\ast^2) \Rightarrow F^2 = M_{\rm eff}r_\ast^2$); only the **name** was misplaced. Consequence for downstream prose: "$f$ buys robustness" must read "$F^2 = M_{\rm eff}r_\ast^2$ buys robustness" — the inertial mass is half of it. **[proven; verified (g′)]**

---

## 9. Open problems (honest list)

- **Open-1:** full discrete Lyapunov-function proof of governor BIBO (Prop-10 is evidenced, with the saddle-blindness caveat proven).
- **Open-2:** the interference problem (§6) — measurement protocol proposed, no solution claimed. *(v1.1: first measurement scheduled in `v3-lattice-build`.)*
- ~~**Open-3:** exact stationary distribution of the corrected-FDT Langevin chain for *relativistic* $T$ (the O-step targets a Gaussian momentum marginal; the relativistic Gibbs marginal $\propto e^{-T(p)/T}$ is non-Gaussian — bias unquantified beyond $O(\varepsilon^2)$ heuristics).~~ **[CLOSED 2026-07-10 by Prop-9′ — the question was mis-posed.]** There *is* no such stationary distribution equal to Gibbs: **no $\sigma$ works** (Lemma-9a + Maxwell–Jüttner's exponential characteristic-function tails). The "$O(\varepsilon^2)$ heuristic" is **retracted** — the bias is $O(1)$ and $\varepsilon$-independent, controlled by $d\Theta=dT/(m_0c^2)$. The stationary law of the coded chain *does* exist and is Gaussian-smoothed; it is simply not $e^{-H/T}$. Exact fixes: F2 (latent-mass), F3 (MJ refresh), F4 (Metropolis) — §2.4.
- **Open-4:** shadow/backward-error theory for the state-dependent friction field $\gamma_\phi(q)$ (Prop-11's determinant is exact; the modified-flow story is not worked out).
- **Open-5:** spectrum theory away from attractors (Prop-2's kinetic coupling at hot states; time-varying $K(q_t)$ along trajectories — Oseledets rather than eigenvalues).
- **Open-6:** $\Lambda_{k\to k+1}$ (shell-lift) specification and its structure-preservation requirements.

## Appendix N — numerical verification summary

Script: `.claude/scratch/formalism-note/checks.py` (numpy float64; complex-step Jacobians, step $10^{-30}$; seeds fixed: `default_rng(42)`, sim seed 7; runs in seconds; mirrors `integrators.py` exactly). Command: `uv run --no-project --with numpy python checks.py`.
Check **(g′)** (added by the 2026-07-09 corrigendum) lives in `.claude/scratch/f5-corrigendum/{verify,verify2,verify3}.py` (numpy 2.0.2, float64, `default_rng(42/7/11)`); same map semantics.

| id | claim (Prop) | observed |
|---|---|---|
| (a) | Goldstone latch, exact $q_\infty = q_0+\varepsilon p_0/(M\gamma)$ (3.3a) | error $1.0\times10^{-15}$; relativistic series match 0 ulp, frozen 2000→4000 steps at 0 ulp; curved mode decayed to $10^{-45}$ |
| (b) | half-life law, $1/\mu^2$, mass-independence, crossover, saddle (3.3b–d) | exact-vs-asymptotic $\lambda$: ≤5×10⁻⁸; fit matches exact to 9 decimals (overdamped); $n_{1/2}$ ratios 3.993/3.998 (pred 4.0); $\vert\lambda\vert=\sqrt{0.95}$ for $m=1$ and $m=0.25$; $h^*$ real↔complex confirmed; $\min_\gamma(\max\vert\lambda\vert-1)=1.4\times10^{-5}>0$ |
| (c) | squeeze symplecticity incl. mass-weighted + composition (Def-6) | all $\le 2.2\times10^{-16}$; $\det S = 1$ exactly |
| (d) | conformal symplecticity; Lyapunov-reg degeneracy (Prop-3,5) | $\lVert J^\top\Omega J-(1-\gamma)\Omega\rVert \le 3.3\times10^{-16}$; mean-log-sv $-\tfrac12\ln(1-\gamma)$ ≤ $2.1\times10^{-16}$ across random $\theta$; max-log-sv spread ≈ 0.014–0.136 |
| (e) | Langevin FDT mismatch + fix, **Newtonian** (Prop-9) | code: $\mathrm{Var}(p)=0.0263$ (target $mT=1.0$; $T_{\rm eff}$ pred 0.0132 vs nominal 0.5); fixed: 1.000000 (Lyapunov solve) / 0.9994 (sim); $\mathrm{Var}(q)=0.500156 = $ shadow prediction |
| (e$'$) | **relativistic Gibbs no-go + latent-mass fix (Prop-9′)** | free particle on the torus, $\Theta=1$: coded stationary $\mathrm{Var}(p)=0.9973$ ($=M_{\rm eff}T$), exc.kurt $-0.014$, KS vs $\mathcal N$ $D=0.0011$ ($p=0.97$, **Gaussian**) vs KS vs MJ $D=0.0845$ ($p=0$, **rejected**; MJ $\mathrm{Var}=2.6995=K_2(1)/K_1(1)$). Char.fn. closed form vs numerical FT $\le1.4\times10^{-14}$; decay rate $\to m_0c$ ($-\!\log|\hat\pi_p|'-m_0c-\tfrac{3}{2t}=-1.7\times10^{-4}$ at $t=100$). Gaussian bound first violated at $\lvert t\rvert=22.79$ ($T{=}0.5,\gamma{=}0.1$), $3.62$ ($T{=}1,\gamma{=}0.5$); at $\Theta=8$, $\lvert\hat\pi_p\rvert/\text{bound}=2.7\!\times\!10^{6}\to5.0\!\times\!10^{119}$ ($t=6\to20$). Chain identity $\hat\mu_{\rm post}=\hat\mu_{\rm pre}e^{-\sigma^2t^2/2}$ to $9.7\times10^{-5}$ ($1.6\times10^8$ samples). $d=1$ table: $\mathrm{Var}$ ratio $1.0150/1.1534/2.6995/16.2819$, exc.kurt $0.030/0.295/1.857/2.907$, KL $7.4\text{e-}5/6.8\text{e-}3/0.384/6.31$ nats at $\Theta=0.01/0.1/1/8$ (all $=K_2/K_1$ to $\le2\times10^{-6}$). **$d$-amplification (chain, $\Theta=0.1$):** $\mathrm{Var}(q)/(T/k)-1 = -0.111/-0.196/-0.390/-0.633$ for $d=1/4/16/64$; Newtonian control $\le10^{-3}$, $d$-independent. **Scaling lemma:** two independent reparameterizations bit-identical ($0.6932785821$, $\Delta=0$) with a working negative control ($0.4669$). **F2:** O-step alone reproduces MJ (KS $D=0.0013$, $p=0.47$); full chain bias $-0.311/-0.536/-0.727 \to +0.0006/+0.0011/+0.0011$ (Newtonian floor $+0.00014$) |
| (f) | KDK shadow Hamiltonian (Prop-7) | $\mathrm{std}(H)$: 1.16e−5→4.63e−5→1.85e−4 (×4/doubling); $\mathrm{std}(\tilde H)$: 1.94e−9→3.11e−8→4.98e−7 (×16/doubling) |
| (g) | discrete Noether: exact conservation/decay; frozen angle (§4.1) | drift 3.0×10⁻¹⁴ (equal $M$, $10^5$ steps); **excursion** 2.6 (unequal $M$ — an envelope, *not* a drift rate; see (g′)); decay-law error 9.3×10⁻¹⁶; angle frozen to 2.3×10⁻⁹; radius→vacuum to 3.4×10⁻¹⁰ |
| (g′) | **Prop-17 kinetic-spurion blindness** + bounded charge (§4.1) | flat $\mu^2 = $ **0.0 exactly** (8 random anisotropic *diagonal* $M$); $\le7.2\times10^{-16}$ (6 random non-diagonal SPD $M$, cond ≤ 14); flat $q$-dirs $=\ker K$ to 3.3×10⁻¹⁶; latch angle drift **0.0 exactly** ($\gamma$=0.05, $2\times10^4$ steps, $M=(1,2)$ and $(0.31,4.7)$); ring amplitude ratio 1.0007–1.0064, elliptic half-period ratio 0.9994–1.0005; $\sup_n|L_n|$=0.582 < compact bound 0.820; per-decile $\sup|L|$ slope $+6.1\times10^{-4}$ over $10^6$ steps |
| (h) | $\det J = (1-\gamma_\phi(q'))^d$ (Prop-11) | ≤ 1.1×10⁻¹⁶ (3 random states) |
| (i) | $v^{\max} = c/\sqrt M$, crossover $p^*$ (Prop-1) | exact to 9 digits; $v(p^*)/v^{\max} = 0.707107 = 1/\sqrt2$ |
| (j) | Cor-13: retention minimum at $\gamma^*$ | argmin$_\gamma$ (40k grid) $-\ \gamma^*$ = 2.3×10⁻⁵ (= grid step); $\gamma^*$ = 0.039214 vs $2h(1-h)$ = 0.039200 ($h$=0.02); $n^{\min}$ 34.65 vs $\ln2/h$ = 34.66; endpoints 276.6/1630.2 (mo-deep-read: 276.6/1630); exact floor at $\gamma$=0.1: 13.158 vs approx 13.863 |
| (k) | Cor-14: metric bifurcation (Mo protocol on our mode) | measured/predicted: 1.001, 1.012, 1.147, **2.300 (EP)**, 0.933, 0.435, 0.187 — reproduces mo-deep-read Check-6 (1.001…2.298…0.187); $d\ln n_\times/d\ln\delta$ = −0.58 underdamped (ballistic −0.5); envelope floor 13.2 throughout |
| (l) | Cor-15: EP onset prefactor + Jordan structure | $C(0.1)$ = 0.324724 (formula) vs $\varphi/\sqrt{h-h^*}$ = 0.324740 @ $10^{-5}$ (mo empirical 0.3247); $\lVert(A-\lambda I)^2\rVert$ = 1.1×10⁻¹⁷ with $\lVert A-\lambda I\rVert$ = 0.1; $\lVert A^nz\rVert/\lambda^n$ ratios 1.76→1.87 (→2 = linear) |
| (m) | Prop-16: discrete cocycle identity; latched exponent | max deviation 1.2×10⁻¹⁵ over 500 steps; $\widehat\lambda(10/10^2/10^3)$ = +0.0 exactly at the latch |
| (n) | App-N artifact: kick-phase jitter of first crossing | log-E ripple measured 0.2627 vs $\gamma(2-\gamma)/(4\sin h)$ = 0.2442; crossing ranges over 64 phases: [17,30] @ $h$=0.1 (jitter pred ±4.8), [15,48] @ $h$=0.05 (pred ±9.5; half ripple period 31); transcribed form (γ/2h)/\|ln√(1−γ)\| = 9.7/19.5 — ×2 too big |

Known diagnostic artifacts (not map errors): underdamped *measured first-crossing* $n_{1/2}$ (23–26 vs predicted 27) and short-window fits differ at the ~1% level from the asymptotic rate because the amplitude envelope ripples within a period; the exact eigenvalues are the ground truth and match theory to all printed digits.
**v1.1 annotation — closed form for the first-crossing artifact (check (n)).** The artifact is **kick-phase-dependent**: friction bites only momentum, so $\log E$ ripples around its secular decay with amplitude $\gamma(2-\gamma)/(4\sin h) \approx \gamma/(2h)$, giving a first-crossing jitter
$$\Delta n \;\approx\; \pm\,\frac{\gamma(2-\gamma)}{4\sin h\ \lvert\ln(1-\gamma)\rvert} \;\approx\; \pm\,\frac{1}{2h}\ \ \text{steps}$$
— **$\gamma$-independent at leading order** — with worst-case slip bounded by ~half the ripple period $\pi/(2h)$ when the rippled envelope grazes the threshold. Verified: measured crossing ranges over 64 kick phases $[17,30]$ at $h{=}0.1$ (pred ±4.8) and $[15,48]$ at $h{=}0.05$ (pred ±9.5; $\pi/2h = 31$) around $n_{1/2}=27.03$; v2-so2-build's measured ±10 @ $h{=}0.05$ and ±12 @ $h{=}0.041$ sit exactly on $1/(2h)$. *Correction (v1.1):* the closed form quoted in v2-so2-build (and copied into task f5-v11), $\pm(\gamma/2h)/\lvert\ln\sqrt{1-\gamma}\rvert$, mixes the energy-ledger ripple with the amplitude-ledger rate — it is ×2 too big and suggests a spurious $\gamma$-dependence; that report's *measured* numbers match the corrected form above.

---

# End of formalism note

## Proposed handover updates (for the Hub)

*(v1.0 block, 2026-07-04 — already folded into handover §7/§8 by the Hub; retained for provenance only. v1.1 deltas are reported separately in `.claude/outputs/f5-v11.md`.)*

**§7 (Known Issues) — upgrades and additions:**
1. **§7.6 upgrade (suspicion → proven):** the Lyapunov regularizer is **provably degenerate**: mean-log-singular-value of the step Jacobian $\equiv \tfrac12\ln(1-\gamma)$ for *any* $\theta$/state (conformal symplecticity, F5 Prop-3/5); the wake phase calls it with $\gamma=0$ (`train.py:155`) so the loss is identically 0 with ~zero gradient (only the $10^{-8}$ log-epsilon + float32 noise leak through). It has never done anything. Fix spec (engineer): $\max_i\log\sigma_i$ or $\sum_i(\log\sigma_i)^2$ or positive-part sum — all θ-sensitive (verified).
2. **NEW discrepancy — Langevin noise violates discrete FDT:** coded $\sigma=\sqrt{2\gamma T\,dt}$ ⇒ per-mode effective temperature $T_{\rm eff,i} = 2\,dt\,T/((2-\gamma)M_{{\rm eff},i})$, not $T$ (F5 Prop-9; verified: Var(p) 0.0263 vs target 1.0). Consequences: temperatures not in energy units (dt/mass absorbed); with learned M, per-mode temperatures differ ⇒ no single Gibbs invariant — **candidate explanation for the MNIST digit-mode imbalance [conjectured; analyst can test on Exp-C checkpoints]**. Exact fix **in a Newtonian kinetic mode**: $\sigma_i^\star=\sqrt{M_{{\rm eff},i}T\gamma(2-\gamma)}$ (verified to give Var(p)=mT exactly; samples shadow-Gibbs). Needs a backward-compat flag (retrained temps shift). **⚠ 2026-07-10 (Prop-9′): in `relativistic` mode $\sigma^\star$ is *not* a fix — no $\sigma$ is.** Exp-C's default is relativistic at $d\Theta=784$; use F2 (latent-mass), F4 (Metropolis), or `newtonian_learned`. Raising $c$ to 5 is **not** sufficient at $d=784$ ($d\Theta=31.4$).
3. **§1.2 wording:** velocity saturation is **anisotropic**: $v^{\max}_i = c/\sqrt{M_i}$ ("saturates at c" exact only for M=I); relativistic rest inertia is $m_0M$, and relativistic T kinetically couples modes at finite p (F5 Prop-1/2). Feature (per-particle causal budgets), but paper wording is imprecise.
4. **Coercivity gap:** Deep/Conv potentials are architecturally non-coercive (no confinement; clipping lives in the training loop, not the unit) — BIBO assumptions fail out-of-unit for them (F5 Prop-10).

**§8 (Open Directions) — results to fold in:**
5. **Pseudo-Goldstone conjecture CONFIRMED with scope:** half-life exponent exactly −2 in *spectral* mass ($\propto\mu^{-2}\propto\delta^{-1}$), valid only overdamped ($\varepsilon\mu\lesssim\gamma/2$); saturates at the mass-independent $2\ln2/\gamma$ steps underdamped. Full budget table with exact constants in F5 §3.4 — this is V2's quantitative backbone.
6. **Goldstone claim strengthened:** dissipation doesn't merely spare the displacement — it **freezes** it (exact latch: $q_\infty = q_0+\varepsilon p_0/(M\gamma)$; γ>0 turns a marginal integrator into deadbeat memory). Memory lives in the coset coordinate; the Noether charge is the write current and decays exactly as $(1-\gamma)^n$.
7. **Two hard negative results for the depth-stability story:** (i) friction can NEVER stabilize a saddle direction (only slow it: escape rate mirrors the memory law); (ii) the governor is blind to isoenergetic saddle escape (V→T at constant H) — only the relativistic cap bounds it. Depth stability = curvature control + relativistic mode, not γ.
8. ~~**V2 design constraint (new):** an SO(2) channel requires **equal inertial masses within the channel** (kinetic isotropy; Schur ⇒ multiplets share mass) — else the kinetic term explicitly breaks the symmetry (verified: O(1) charge drift). Falsifiable: does learned M isotropize on symmetric data?~~
   **[SUPERSEDED 2026-07-09 by Prop-17, §4.1 — do not propagate.]** The correct design constraint: an SO(2) channel requires equal inertial masses for its **Noether write current to be conserved**. The *register* needs no such thing — by Sylvester's law of inertia a flat direction of $V_\theta$ keeps $\mu^2\equiv0$ under **any** invertible $M$, so the vacuum manifold, the channel count and the $\gamma>0$ latch survive kinetic anisotropy exactly (verified: flat $\mu^2 = 0.0$ bit-exact; latch drift $0.0$). The charge non-conservation is a **bounded, non-secular excursion** ($O(1)$), not drift; on the vacuum orbit its amplitude is $\sqrt{2E}\,r_\ast(\sqrt{M_{\max}}-\sqrt{M_{\min}})$ with period half a revolution. Falsifiable (restated): does learned $M$ isotropize on symmetric data? — and it must be tested **on the charge law**, since the Hessian and the latch are provably blind to it.
9. **Thread-5 nomenclature fix (adopt Def-2):** inertial mass M vs spectral mass μ run in opposite directions; "integrate out heavy" must become "adiabatically eliminate fast (large-μ)". At linear order constant M is absorbable; its irreducible roles: causal cap c/√M_i, relativistic regime assignment p*=m₀c√M_i, relative curvature pricing, mass-weighted boost response, learnability prior.
10. **Thread-3 L0 spec:** "global boost is automatically mass-differential" holds **iff** the squeeze is mass-weighted $S^{(M)}=\mathcal N^{-1}S\mathcal N$ — raw squeezes are mass-blind. Engineer must implement $S^{(M)}$.
11. **Thread-1 formal slot:** friction-field damping gives **exact** position-gated volume destruction $\det J=(1-\gamma_\phi(q'))^d$ (verified) — the horizon statement is now a theorem, not a metaphor; global conformality/pairing lost where $\nabla\gamma_\phi\ne0$ (state claims accordingly).
12. **For mass-spectrum-peek (analyst):** M alone is half the object — the budget is $\mathrm{eig}(M_{\rm eff}^{-1}\nabla^2V_\theta(q^\ast))$; measure Hessians at attractors too.

**Corrections to prior program docs:** roadmap v0.2 thesis line "unit-modulus modes compute" is fine, but add pairing caveat per-γ; brainstorm Thread-2 half-life conjecture → confirmed-with-scope (item 5); Thread-5 wording → item 9.
