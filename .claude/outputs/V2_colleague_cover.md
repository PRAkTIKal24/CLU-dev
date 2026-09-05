# Cover note — V2 short for review, and where your theory can move it
### For our HEP-theorist colleague · package assembled 2026-07-19

Thank you for agreeing to look at this. The short paper below is the memory-and-symmetry vertical of the CLU program (the recurrent primitive introduced as **CHLU in Jawahar & Pierini, 2026**). You will read the physics — Goldstone/SSB/GMOR/ChPT, custodial symmetry, exceptional points, coset geometry, Maxwell–Jüttner — faster than we can, so this note does two things: it tells you **what to read in what order**, and it points you at the **genuinely open theory questions** where a particle theorist can contribute most. The one thing the draft assumes and you may not want spelled out — the machine-learning scaffolding — is handled by the AI primer in the package, so you should not have to decode any ML jargon on your own.

---

## 1. The package, and the reading order (~90 min end to end)

Five documents, in this order:

1. **`AI_for_physicists_primer.md`** (~20 min) — *read first.* The inverse of the ML-facing primer: it rebuilds every ML concept the draft uses (recurrent memory, the CLU-as-learned-Hamiltonian, wake–sleep/contrastive-divergence training, the LSTM/LEM/coRNN baselines, RMSE and the autonomous-retention protocol, Mo's lifetime law, and the ML-workshop framing) as a physics analogy. It introduces no physics. Its §2 is your fastest on-ramp: **the CLU is a learned separable Hamiltonian $H=T(p)+V_\theta(q)$ evolved by velocity-Verlet with optional damping $p\mapsto(1-\gamma)p$** — an integrator you know, with a data-fitted potential.

2. **The V2 draft PDF** (`v2-short/draft.pdf`) — the paper itself, ≈4–5 pp + appendices, pitched at a non-archival ML workshop (ML4PS / NeurReps). Headline is **Figure 2**: Mo (2026)'s published single-exponential lifetime law, run unchanged on our trained models, is the *overdamped face* of an exactly-solvable mode-mass retention budget. Read the main text (§§1–5); the appendices are deliberately maximal (every corollary, negative, and robustness check is written out) and are reference material, not required reading.

3. **`v2-symmetry-deepdive.md`** — the rich HEP companion (a running theory note). This is where the physics is developed properly: the exact GMOR relation with a *measured* condensate, the current-theoretic decay constant $F^2=M_{\rm ch}r^{*2}$, the **kinetic-spurion blindness theorem** (a flat direction survives *any* inertial anisotropy — the latch is a **modulus of $V_\theta$, not a Goldstone of the full $H$**), the custodial/$\rho_{\rm CLU}$ structure, the finite-$T$ coset-diffusion law (0+1D Coleman), and the relativistic running-decay-constant thread (§7bis). See §4 below for a housekeeping caveat about reading it.

4. **The F5 note PDF** (`f5-note/f5-note.pdf`) — the rigorous companion: the propositions and machine-precision numerical checks behind everything above (the exact single-mode solution, conformal symplecticity, the GMOR/retention law, the kinetic-spurion blindness proposition, the relativistic-Gibbs no-go and its latent-mass thermostat fix). This is the formal record; go here when you want a proof rather than a narrative. **In the papers it is cited in third person as "Anonymous, 2026"** (anonymization posture, §5 below).

5. **This cover note.**

If you have time for only two documents: the **AI primer** then the **deep-dive**. The draft PDF is the artifact under review; the deep-dive is where you will actually want to work.

---

## 2. What we are asking — and where you can contribute most

We would value **feedback, additions, and amendments** of any kind. But to be concrete, here are the places where the theory is genuinely open and a HEP theorist can move it — these are not polish requests, they are unsolved:

**A. The four open theory questions from the deep-dive (§10, "what I could not prove").** These are stated with the partial results already in hand:
- **O1 — the $(\mu,\gamma,T)$ crossover.** At $T>0$ the retention law is *conjectured* to split into two additive channels, $1/n_{\rm total}\approx 1/n_{\rm relax}(\mu,\gamma,\varepsilon)+1/n_{\rm hop}(\delta,r^*,T)$ (power-law relaxation plus Arrhenius inter-well hopping $\propto e^{-2\delta r^*/T}$), but only *heuristically*. The exact crossover, and the joint law in the **underdamped** band, are underived. The V2 budget table is currently the $T=0$ face of a $(\mu,\gamma,T)$ cube whose interior is open.
- **O2 — the curved-coset diffusion correction.** The coset-diffusion constant $D_\theta=\varepsilon T(2-\gamma)/(2F^2\gamma)$ is derived *exactly* for a strictly flat direction of the coded map; on the curved ring the measured ratio is $1.020\pm0.020$ (free-particle control $0.984\pm0.016$). Both are $1\sigma$-consistent with the flat-direction law, but the curvature/radial-coupling correction is not derived.
- **O6 — the underdamped retention–bandwidth bound.** The causal RB conservation law $n_{1/2}\cdot\dot\theta_{\max}^2 = 2\gamma\ln2\cdot m_0c^2/((2-\gamma)\varepsilon^2\delta\Sigma)$ is derived in the deep-overdamped band with a verified $(\varepsilon\mu/\gamma)^2$ correction; its **underdamped** counterpart (where $n_{1/2}$ is mass-independent) is not — the product presumably collapses to a $\gamma$-only statement, but this is open.
- **O7 — the large-$\zeta$ companding failure.** The relativistic latch stores *rapidity*, not momentum (transport logarithmic in the write impulse — a companding encoder). This is exact on a flat direction; on a curved coset a hard write leaves the vacuum manifold, quantified only to $O(0.2\%)$ for $\zeta\le3$. The large-$\zeta$ geometric correction — and whether a hard write can *dislodge* a latch off the orbit — is the natural failure mode of the register and is open.

**B. The non-abelian V4 seeds** (deep-dive §7, §4.2). These are proven-in-toy but untouched on trained/learned potentials, and they are where a group theorist's instinct is worth most:
- **$SO(3)\to SO(2)$ / symmetric-coset register banks.** $S^2=SO(3)/SO(2)$ is a symmetric space, so write-ordering rotates the addressing *basis* (holonomy $\sim\alpha^2$, Gauss–Bonnet) while leaving the stored *value* nearly order-independent ($O(\alpha^5)$ under the CLU's native charge-impulse write). Non-abelian GMOR/latch on a *curved coset with a learned potential* (not a hand-built hat) is entirely open (O4 in the deep-dive).
- **Torus banks.** The falling-out design rule — pick an **abelian (flat) coset** $T^n=U(1)^n$ for an *independent* register bank (zero holonomy, exactly decoupled registers, at the price of $\pi_1=\mathbb{Z}^n$ aliasing) — deserves a theorist's eye on capacity vs interference.
- **Custodial ↔ kinetic-isotropy.** The proposed $\rho_{\rm CLU}:=\mu_1^2/\mu_2^2=M_2/M_1$ and the *multiplet-universal* invariant $\mu_a^2F_a^2=\delta\Sigma$ (the CLU's "$f_K\neq f_\pi$") give a **retention-anisotropy prediction**: within one pseudo-Goldstone multiplet, $n_{1/2,1}/n_{1/2,2}=M_1/M_2$, independent of the spurion. This is the tightest custodial analogy in the note and is load-bearing only non-abelian — a natural place for you to sharpen or break the correspondence.

**C. The running-decay-constant thread** (deep-dive §7bis). The relativistic kinetic term gives a Lorentz-enhanced coset inertia $F_Q^2(p)=F_Q^2(0)\cosh\zeta$ — a "running decay constant" that we are careful to flag is a *rotor*, **not** a ChPT form factor (Lorentz covariance forbids a running $\langle0|A_\mu|\pi(p)\rangle$). Whether this is the right way to say it, and how far the analogy can be pushed at rest vs in flight, is exactly the kind of question you will have opinions on.

If any of these grabs you, that is the contribution we would most value.

---

## 3. Settled vs open — calibrate your skepticism here

Two axes to keep straight, because the paper is deliberately explicit about both:

**Settled — please don't relitigate the verified numbers.** A set of core claims is locked and cross-checked (the program's claims matrix fixes their exact wording and constants). The ones you'll meet in V2:
- The exact **mode-mass budget**: latch displacement $q_\infty=q_0+\varepsilon p_0/(M\gamma)$ with infinite half-life on a flat direction ($\gamma>0$); Noether charge decaying exactly $(1-\gamma)^n$; retention $n_{1/2}\propto\mu^{-2}$ (GMOR in spectral mass, valid $\varepsilon\mu\lesssim\gamma/2$); the mass-independent floor $2\ln2/(-\ln(1-\gamma))$; the exceptional point at $h^*\approx\gamma/2$ with $\sqrt{h-h^*}$ frequency onset; the critical-damping retention minimum at $\gamma^*\approx2\varepsilon\mu$.
- **GMOR proper on trained checkpoints**: with a linear ambient spurion the three ChPT objects are measured *independently* and satisfy $\mu^2F^2=\delta\Sigma$ to machine precision ($F^2=M_{\rm ch}r^{*2}$ the decay constant, $\Sigma=r^*(\delta)$ the condensate), the leading LEC resonance-saturated by the radial mode. (The shipped *angular* tilt cannot see $\Sigma$ — it measures only the product — which is why it is a clean power-law verification, not a condensate measurement.)

**Open / honest negatives (Appendix F, and the deep-dive's flags) — deliberately unresolved, and fair game.** These are documented, not hidden, and several are future-work anchors:
- No **isotropization** of learned inertial mass on symmetric data (N4) — but the register survives it *exactly*; the breaking is invisible to the Hessian and the latch, showing up only as a *bounded oscillation* of the Noether charge. (This is the measured counterexample to the necessity direction of Mo's hypothesis: equivariance is sufficient, not necessary, for neutral memory.)
- The **continuous coset register is designed-only** (N46): an *emergent* (generic-MLP) unit stores ≈1–1.6 bits on discrete washboard minima, not a real-valued angle. The register must be designed in; the *law* that governs it generalizes.
- The unit **cannot enter the input-driven task-RMSE axis** (N6): no native velocity ingestion, equivariant-control wrapper unbuilt — only the RNN baselines compete on supervised RMSE, and we fabricated no task-RMSE for the CLU.
- The **erosion-study novelty** (§3.5) is pending a literature scout — the *phenomenon* (short-run CD deforming a landscape) is classical; the *specific instance* (a symmetry-restoration transition on a designed degenerate vacuum, with a value-anchor cure) is what we claim as new, and that novelty is not yet confirmed.

**Verification vs evidence — the calibration that governs how hard to push.** Read every number through this tag (primer §7): results on **designed testbeds** (architecturally-invariant potentials, analytic tilts) are **verification of the theory's exactness** — machine-precision, and never dressed up as discoveries; results on **learned/trained/anharmonic** systems (the Mo head-to-head, the baseline collapse, the erosion transition) are **evidence** — laws holding to 2–15% with the deviations themselves predicted. If a number is machine-precise, it is confirming a solvable core; if it is 2–15%-with-predicted-deviations on a trained model, it is an actual scientific claim. Certificate/boundedness statements carry their scope clause (coercive potential / compact sublevel set) *next to* the claim — hold any contributed claim to the same standard, and we all stay honest.

---

## 4. Housekeeping to ignore in the deep-dive (and a recommendation, not an edit)

The deep-dive (`v2-symmetry-deepdive.md`) is a live internal working note, so it contains **program-internal bookkeeping that is not for you to act on**: references to "the Head" and "the Hub" (our human-lead / research-lead roles), wave numbers (`v1.1`, "w13"), task slugs and file paths (`v2-full-runs.md`, `fit-gap-anatomy`, `.claude/scratch/...`), a "Proposed handover updates" section, and an "Open questions for the Head / co-author" list. **These are workflow scaffolding — read straight past them.** The physics you want is in **§§2–7bis** (GMOR/ChPT done properly; the taxonomy; chiral-vs-custodial; finite-$T$; topological capacity; the non-abelian bank; the relativistic thread) plus the correspondence table in §8 and the honest-flags list in §10 (which is genuinely useful — it says exactly which analogies are structural and which are evocative-but-forbidden, e.g. *no chiral logs, no anomalies, no WZW — 0+1D classical, no $\hbar$*).

**Recommendation (we defer the decision to the Head; we did not edit the note):** a *light* de-internalization pass — stripping the "Proposed handover updates" and "Open questions for the Head" sections, and softening the wave-number/task-slug references — would make the deep-dive cleaner to hand to an external collaborator, and is worth doing before any wider circulation. For **this** send, to a trusted potential co-author, the pointer above (read §§2–7bis for the physics, ignore the workflow) **suffices** — the internal references are transparent enough that they will not mislead you, and the physics stands on its own. We flag one substantive internal item you *should* know is real, not noise: the note identifies a **proven error in the F5 formal note** (the old kinetic-isotropy clause "$\mu^2\propto$ mass splitting" is false — the correct statement is $\mu^2\equiv0$ for any invertible $M$, the blindness theorem). **This has since been corrected in the F5 note you have** — the note now states the blindness theorem as Prop.\ (kinetic-spurion blindness) and carries an explicit "Corrigendum note (2026-07-09)" explaining the fix. So the deep-dive's S6 discussion is the *diagnosis* of a now-resolved issue, not a live error in the attached formal note; we mention it only so the two documents read consistently.

---

## 5. Authorship and naming context

- **Authors are a placeholder** (`[AUTHORS PLACEHOLDER]`) in the current draft. Your potential co-authorship is genuinely on the table and is the Head's call — this review is the natural first step, with no commitment implied either way.
- **The naming continuity, stated once so it is not confusing:** the unit is the **CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)**. "CLU" is the current name; "CHLU" is what the same primitive was called in the 2026 introduction. The shorts carry that continuity sentence verbatim.
- **Anonymization posture.** The formal theory note (F5) is, for the moment, **unpublished and cited in third person as "Anonymous, 2026"** in the shorts — a deliberate anonymity strategy while the program decides titles/authors and arXiv timing. So when you see the draft lean on "the theory note (Anonymous, 2026)," that is us, cited hermetically; the shorts otherwise cite only Jawahar & Pierini (2026) and published external work (Mo 2026, Di Bernardo et al. 2025, the standard RNN/ChPT literature). Nothing else in the program is referenced as existing.

---

*Any level of engagement is welcome — a paragraph of reactions, a torn-apart appendix, or a full theory contribution on one of the open questions in §2. The open questions are real; if one of them is a half-hour's work for you, that is a result we could not get on our own.*
