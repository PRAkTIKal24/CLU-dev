# scout-dynamical-memory-priorart — web-scout report

Task + acceptance criterion: Position CLU-as-addressable-dynamical-memory against S4/Mamba/HiPPO, NTM/DNC, modern Hopfield, and Hamiltonian sequence models; four independent novelty verdicts; adversarial read of the transformer-competitor claim.
Status: **done**, with two named retrieval gaps (§0).

> ⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5, first 10 lines).**
> 1. **⛔ THE HEAD'S NTM/DNC PREMISE IS FALSE AS STATED, AND THIS IS THE REPORT'S MAIN RESULT.** The handover (2026-07-21, consequence 1) says NTM/DNC instability "is widely attributed to *hard/discrete* addressing" and that continuous restructuring "may be exactly how CLU escapes the precedent." **The canonical NTM and DNC were already fully soft, continuous and end-to-end differentiable.** The discrete variant (RL-NTM) is a *different, later* line. **The precedent therefore does NOT transfer as an escape — it transfers as a warning**, and two of the three diagnosed DNC failure modes bite CLU's address scheme directly (§2). Every site claiming continuity as our escape hatch must be corrected.
> 2. **"Address → trajectory" is already published prior art.** Kong, Brewer & Lai, *Nature Communications* 2024 build **location-addressable retrieval of dynamical attractors via an index channel** — an index value selects which trajectory a dynamical system produces. This is our architecture's core move, in a peer-reviewed venue, two years old. It is not fatal (not symplectic, not mass-structured) but it must be cited, and "retrieval is a rollout instead of a weighted sum" can no longer be presented as our novelty.
> 3. **Tunable-permanence writes are largely occupied** by Karuvally et al.'s energy-memory line (analytic memory **escape times**, phase transition between static and dynamic regimes). Novelty claim (b) must be narrowed before any paper asserts it.
> 4. Handover's framing "symplectic ⇒ no decay is a retention guarantee attention has no analogue for" — **HiPPO-LegS already has a principled non-decay guarantee** with an explicit error bound and gradient bound (§1). The "no analogue" phrasing is wrong and a reviewer will know it.

---

## 0 · Retrieval gaps (stated, not reconstructed)

- **Ramsauer's exact capacity theorem constant: NOT OBTAINED from primary source.** `ar5iv` for arXiv:2008.02217 exceeds my fetch size limit; OpenReview served a bot-verification page. I have the **abstract verbatim** (primary) and a secondary claim of the form `N ≥ √p · c^((d−1)/4)`. **The exponent form is [SINGLE-SOURCED, secondary] — do not put it in a paper without someone reading the theorem.** The qualitative claim (exponential in dimension) is [VERIFIED].
- **UnICORNN's Hamiltonian statement and integrator: NOT OBTAINED.** The PDF returned compressed streams. Its Hamiltonian framing is [SINGLE-SOURCED, secondary]. Treat as unverified.

---

## Answer first

The sharpest honest finding is negative on two fronts and positive on one narrow front. **(1) The NTM/DNC precedent does not spare us** — those systems were already continuous and soft, and their documented failure modes (flat/noisy address distributions from value-contaminated lookup; exponential degradation when reads are chained) are *continuous-addressing* pathologies that map onto CLU's `(m,q₀,p₀)` address and its wormhole/multi-particle chaining. **(2) Against SSMs, symplectic+nonlinear+mass buys very little that is demonstrable**: HiPPO already supplies a *provable* optimality theorem for what a linear latent memorizes, with error `O(tL/√N)`, exact timescale equivariance, and gradient `Θ(1/t)` — CLU has no theorem of comparable strength, and the fixed-capacity bottleneck that limits SSMs on retrieval is *not* relieved by conservation. **(3) The one genuinely unoccupied space is structure-preserving test-time retry** (novelty c), which is real but low-stakes. Given measured **1–1.6 bits/register** and no win over a trivial baseline, "competitor to the transformer" is currently indefensible and would sink a submission; the defensible framing is a **physics-structured associative memory with *designed* retention**, evaluated on associative-recall diagnostics.

---

## 1 · S4 / Mamba / HiPPO — the sharpest competitor

### 1.1 What HiPPO actually guarantees [VERIFIED — ar5iv primary, arXiv:2008.07669]

HiPPO solves an **online function approximation** problem: given input `f(t)` and a time-varying measure `μ^(t)` on `(-∞,t]`, find coefficients `c(t) ∈ ℝ^N` minimising

> `||f_{≤t} − g^(t)||_{L₂(μ^(t))}`

with `g^(t)` a polynomial of degree `< N`. **The measure is the theory of what is remembered** — it is an explicit, designed statement of which parts of the past matter. That is the thing CLU currently lacks.

HiPPO-LegS: `d/dt c(t) = −(1/t) A c(t) + (1/t) B f(t)`, with `A_{nk} = (2n+1)^{1/2}(2k+1)^{1/2}` for `n>k`, `n+1` for `n=k`, `0` for `n<k`; `B_n = (2n+1)^{1/2}`.

Three guarantees, all quoted from the paper:
- **Approximation error (Prop. 6):** `||f_{≤t} − g^(t)|| = O(tL/√N)` for `L`-Lipschitz `f`; `O(t^k N^{−k+1/2})` with `k` bounded derivatives.
- **Timescale robustness (Prop. 3):** *"For any scalar α > 0, if h(t) = f(αt), then hippo(h)(t) = hippo(f)(αt)"* — exact dilation equivariance, and **the LegS recurrence has no discretization step-size hyperparameter Δt at all.**
- **Gradient bound (Prop. 5):** `||∂c(t₁)/∂f(t₀)|| = Θ(1/t₁)` — polynomial, not exponential, decay.

⭐ **Read Prop. 3 against CLU carefully.** HiPPO-LegS is *exactly* timescale-equivariant and *has no `dt`*. CLU's mass `m` is a learned timescale and CLU has an explicit `dt` (and, per the handover, an unresolved `dt`-vs-cycle units question). **A reviewer who knows HiPPO will ask why a learned mass spectrum is better than an architecture that is provably invariant to timescale by construction.** We do not currently have an answer.

### 1.2 Mamba, selectivity, and the 2026 state [VERIFIED — arXiv:2312.00752 abstract; arXiv:2603.15569 abstract, fetched]

Selectivity = making the SSM parameters **input-dependent**, so the model can "selectively focus on or ignore specific parts of past input history based on their present relevance" — i.e. Mamba's answer to content-dependence, the thing LTI SSMs lack.

**Current state is Mamba-3 (ICLR 2026, arXiv:2603.15569, 16 Mar 2026; Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu).** Verbatim from the abstract: three improvements — *"(1) a more expressive recurrence derived from SSM discretization, (2) a complex-valued state update rule that enables richer state tracking, and (3) a multi-input, multi-output (MIMO) formulation."* At 1.5B, **+0.6 pts** over Gated DeltaNet, **+1.8 pts** with MIMO; *"comparable perplexity to Mamba-2 despite using half of its predecessor's state size."*

⚠ **Directly relevant to CLU:** the field's own frontier model arrived at **complex-valued (i.e. rotational/oscillatory) state updates** as the mechanism for richer state tracking. That is the nearest thing to CLU's oscillatory-conservative latent, it is in the strongest linear-model line, and it is **linear**. Any CLU claim that "oscillatory latent dynamics is the missing ingredient" now has a well-resourced 2026 competitor that got there with a much simpler mechanism. [MY INFERENCE, from the verbatim abstract]

Standing vs transformers: Mamba-3's own framing is **Pareto frontier** (quality vs inference cost), not domination. Secondary sources note transformers remained dominant in the two years after Mamba. [SINGLE-SOURCED for the dominance claim — but note the *paper itself* claims a Pareto improvement, not a replacement, which is the modest framing we should imitate.]

### 1.3 The real gap in SSMs — and whether CLU fills it [VERIFIED — arXiv:2402.01032, ICML 2024]

Jelassi, Brandfonbrener, Kakade & Malach, *"Repeat After Me: Transformers are Better than State Space Models at Copying"*, ICML 2024: **a two-layer transformer can copy strings of exponential length while GSSMs are fundamentally limited by their fixed-size latent state**; pretrained transformers "dramatically outperform state space models at copying and retrieving information from context."

This is the genuine, theorem-backed weakness of the whole latent-dynamics line, and it is a **capacity** argument: a fixed-size state cannot hold unboundedly many key-value pairs.

⛔ **Does CLU fix it? No.** CLU's latent is also fixed-size. **Symplectic conservation prevents *decay*; it does nothing about *capacity*.** Conservation says the information you put in is not lost to dissipation — it says nothing about how much fits. Our own measurement (**1–1.6 bits/register**) is a capacity number, and it is the same bottleneck Jelassi et al. prove is fatal for GSSMs on retrieval. **Anyone claiming CLU beats SSMs on retrieval must confront this; I found no argument that survives.**

### 1.4 ⭐ VERDICT: what symplectic + nonlinear + mass buys over a linear SSM latent

**Adversarially, and this is the sentence the task asked for: on the benchmarks the field cares about, not much, and we cannot presently demonstrate any of it.**

| claimed advantage | honest status |
|---|---|
| symplectic ⇒ no information decay | **Weakest claim.** HiPPO-LegS already remembers all history with error `O(tL/√N)` and gradient `Θ(1/t)`. Ours is a structural property with **no error bound attached**; theirs is a theorem. Strictly worse epistemic standing. |
| mass ⇒ timescale hierarchy | **Undercut twice.** HiPPO-LegS is *exactly* timescale-equivariant with no `dt`. And per the handover, our mass spectrum has been **inert in every run** (N7, `mass_lr_mult` unwired). We are claiming an advantage from a mechanism we have never switched on. |
| nonlinear latent ⇒ more expressive than linear | **Plausible but unevidenced, and contested.** Mamba's selectivity and Mamba-3's complex updates buy content-dependence and state tracking within (near-)linear machinery. Nonlinearity also **forfeits the parallel scan** — the reason SSMs are trainable at scale. This is a real cost we would have to pay and have not priced. |
| fixed-capacity retrieval bottleneck | **Not addressed by CLU at all** (§1.3). |
| tunable retention (`μ²` half-life, friction deletion) | ⭐ **The one place with a defensible edge over SSMs** — SSM forgetting is a learned side-effect of the recurrence, not a *designed, per-item* specification. But see §4: the energy-memory literature has partly taken this too. |

**The honest one-liner for the paper:** the differentiator is **not** performance, expressivity, or retention-as-such — it is that **retention is *specified per item at write time* rather than emerging from a learned recurrence.** That is a controllability claim, not a capability claim, and it should be stated as such.

---

## 2 · ⭐⭐ NTM / DNC — the precedent does NOT spare us

### 2.1 The premise correction [VERIFIED — arXiv:1410.5401 abstract, fetched verbatim]

NTM abstract: *"The combined system is analogous to a Turing Machine or Von Neumann architecture but is **differentiable end-to-end, allowing it to be efficiently trained with gradient descent**."*

The mechanism is explicitly **blurry/soft**: NTMs define *"'blurry' read and write operations that interact to a greater or lesser degree with all the elements in memory (rather than addressing a single element, as in a normal Turing machine)"*, with weightings defining *"a continuous distribution over the memory locations to make the whole operation differentiable."* [VERIFIED — the "blurry"/"greater or lesser degree" language is Graves's own, recovered via secondary quotation of the paper text; the differentiability claim is verbatim primary.]

**The discrete/hard-addressing variant is a separate line:** Zaremba & Sutskever's **RL-NTM** used REINFORCE to learn *where* to access memory precisely *because* hard addressing is non-differentiable. [VERIFIED — secondary description consistent across sources.] It was indeed hard to train — but **it is not the system whose instability the field remembers.**

⇒ **NTM and DNC were unstable *with* continuous, gradient-trained addressing.** Continuity is not the escape hatch.

### 2.2 What actually went wrong [VERIFIED — Csordás & Schmidhuber, ICLR 2019, arXiv:1904.10278, abstract fetched verbatim]

The most precise published diagnosis, and it is entirely within the soft regime:

> *"An analysis of its internal activation patterns reveals three problems: Most importantly, **the lack of key-value separation makes the address distribution resulting from content-based look-up noisy and flat**, since the value influences the score calculation, although only the key should. Second, **DNC's de-allocation of memory results in aliasing**, which is a problem for content-based look-up. Thirdly, **chaining memory reads with the temporal linkage matrix exponentially degrades the quality of the address distribution.**"*

Their fixes improved bAbI mean error rate by **43%** — i.e. these were the dominant defects, and they were fixable *without* abandoning soft addressing.

### 2.3 ⭐ Does this bite CLU? Two of three, directly.

| DNC failure | maps onto CLU? |
|---|---|
| **Key–value entanglement → flat, noisy address distribution** | ⛔ **YES, and severely.** CLU's address is `(m,q₀,p₀)` and the *value* is the trajectory launched from it. **`q₀` is simultaneously the key and the initial condition of the value** — there is no key/value separation *by construction*, which is precisely the defect Csordás & Schmidhuber call "most importantly". [MY INFERENCE, but the structural correspondence is tight.] ⇒ **Actionable: CLU needs an explicit key/value split — an address embedding distinct from the launch state — or it reproduces the DNC's worst diagnosed defect architecturally rather than incidentally.** |
| **De-allocation → aliasing** | ⚠ **YES.** Our deletion mechanism is the `γ_φ` friction/trash region. DNC's lesson: freeing a location without *wiping its contents* makes freed and live addresses collide under content lookup. Their fix was to wipe cell contents, not merely decrement usage. A friction-damped region that retains landscape structure is the aliasing case. |
| **Chained reads degrade exponentially** | ⚠ **YES.** This is our **wormholes** (one particle gathering from far-apart regions) and **multi-particle consolidation** — both are read-chaining. DNC's temporal linkage degraded *exponentially* in chain length. |
| Fixed-size memory, no deallocation → collisions | Partially — CLU's landscape is fixed-capacity (cf. 1–1.6 bits/register). |

**Net:** the Head's amendment (continuous restructuring, gauge-loose, non-degenerate rather than exact addresses) is a **genuinely good design instinct and I found nothing refuting it** — but its *justification* in the handover is wrong. It does not avoid the NTM/DNC precedent by sidestepping discreteness, because there was no discreteness to sidestep. **The correct claim is narrower: CLU avoids the RL-NTM-style non-differentiable-search problem (which soft NTM/DNC also avoided), and must still independently defend against the three soft-addressing pathologies above.** Getting this right in the paper is worth more than any positive result here.

*Why the line stalled, best available account:* not one clean cause. Scaling (dense DNC memory is a severe bottleneck; SDNC/SAM reported **>400× speedup** at 2,000 slots), difficulty scaling to LLM-era sizes, sensitivity to initialization, and the field's absorption of "memory" into attention/KV-cache/retrieval (Transformer-XL, Memorizing Transformers, RETRO/RAG) rather than abandonment. [SINGLE-SOURCED / secondary-synthesis — I found **no** authoritative peer-reviewed retrospective explicitly titled as "why MANNs failed." Treat any confident single-cause story as unsupported.]

---

## 3 · Modern Hopfield ↔ attention — verified, with a capacity reality check

**The bridge is real** [VERIFIED — arXiv:2008.02217 abstract, fetched verbatim]:
> *"The new Hopfield network can store exponentially (with the dimension of the associative space) many patterns, retrieves the pattern with one update, and has exponentially small retrieval errors."* … *"The new update rule is equivalent to the attention mechanism used in transformers."*

Update rule: `ξ^new = X softmax(β Xᵀ ξ)`. [SINGLE-SOURCED, secondary — consistent across sources, and it is the well-known form, but I did not read it in the primary PDF.]

Three energy-minimum types: global fixed points (averaging all patterns), **metastable states** (averaging a subset), and single-pattern fixed points; transformer heads reportedly do global averaging in early layers, partial averaging via metastable states in higher layers. [SINGLE-SOURCED, secondary.]

⇒ **"Energy-landscape retrieval competes with attention" is a principled claim, not an analogy.** This is the strongest available support for the program's framing, and it should be cited prominently.

### Capacity, set against our 1–1.6 bits/register

| model | capacity |
|---|---|
| Classical Hopfield (Hebbian) | `P_max ≈ 0.138 N` patterns, `α_c ≈ 0.138` per spin (Amit, Gutfreund & Sompolinsky 1985); ≈**0.14 bits per synapse** [VERIFIED — the 0.138/α_c result is standard and multiply attested] |
| Modern/continuous Hopfield | **exponential in the dimension** of the associative space [VERIFIED, abstract]; form `N ≥ √p·c^((d−1)/4)` [SINGLE-SOURCED — see §0] |
| EDEN (sequence memory) | `O(γ^N)` vs `O(N)` conventional [VERIFIED, abstract] |
| **CLU (measured)** | **1–1.6 bits per register** |

⚠ **The comparison is not favourable and we should make it ourselves before a reviewer does.** These are different units (bits/register vs patterns/dimension) and **not directly commensurable** — but the *qualitative* gap is the point: the modern-Hopfield line's headline is **exponential** capacity, and ours is order-one bits. **Any framing that invites the capacity comparison loses it.** ⇒ Frame CLU's contribution as **retention control and temporal structure**, explicitly *not* capacity, and say so first, before the question is asked. Re: `exp_v1_hopfield_gate` — the literature's expectation for a well-constructed continuous energy memory is exponential-in-dimension capacity with one-step retrieval; a result near 1 bit/register indicates we are far off that regime, which is a finding worth reporting honestly rather than a bug to hide.

---

## 4 · Physics-structured sequence models & the nearest prior art

**⭐ The nearest prior art to "addressable dynamical memory" — and it is closer than the handover assumes:**

**Kong, Brewer & Lai (2024), "Reservoir-computing based associative memory and itinerancy for complex dynamical attractors", *Nature Communications* 15, doi:10.1038/s41467-024-49190-4** [VERIFIED — PMC full text fetched, mechanism and scaling quoted]:
- **Location-addressable retrieval via an index channel**: *"each is associated with a specific value of the index p… the index value p modulates the dynamics of the RC network through an index input matrix W_index that projects p to the entire hidden layer"*; *"to retrieve a desired attractor, we input its index value through the index channel, and the reservoir machine will generate a dynamical trajectory faithfully representing the attractor."*
- **Capacity scaling**: critical network size `N_c ∝ K^γ` with **γ = 1.08 ± 0.01** (one-hot and binary coding), **1.17 ± 0.02** (2D coding) — i.e. roughly **linear** in the number of stored attractors.
- **Failure mechanism**: switching the index changes the dynamical equations while carrying over the last state `(r_last, v_last)` as the new initial condition; retrieval **fails when that carried-over state lies outside the target attractor's basin.**

⇒ **This is our architecture's core move, published, peer-reviewed, in 2024:** an address selects which trajectory a dynamical system produces, read out as the trajectory. It is **not** symplectic, **not** Hamiltonian, **not** mass-structured, and the address is an injected bias rather than an initial condition — so a genuine gap remains. **But "retrieval is a rollout rather than a weighted sum" is no longer ours to claim as novel.** ⭐ Their failure analysis is also a free gift: **basin-boundary violation on hand-off is the predicted failure mode for CLU's multi-particle consolidation and wormhole hand-offs**, and we can test for it directly.

**Energy-based sequential memory — Karuvally, Sejnowski & Siegelmann.** This line occupies much of the "tunable permanence" ground:
- *General Sequential Episodic Memory Model*, **ICML 2023**, PMLR 202:15900–15910 — **multiple-timescale** architecture, asymmetric synapses + propagation delays produce a **dynamic energy surface** with **metastable states encoding memory sequences**; dense capacity under polynomial activations. [VERIFIED — venue/pages]
- *Exponential Dynamic Energy Network (EDEN)*, **NeurIPS 2025**, arXiv:2510.24965 [VERIFIED — abstract fetched verbatim]: *"a static high-capacity energy network with a slow, asymmetrically interacting modulatory population, enabling robust and controlled memory transitions. We formally derive short-timescale energy functions that govern local dynamics and use them to **analytically compute memory escape times, revealing a phase transition between static and dynamic regimes**"*; capacity `O(γ^N)`.
- *Hidden Traveling Waves bind Working Memory Variables in RNNs*, ICML 2024, arXiv:2402.10163 — stores data **as waves updated by boundary conditions**, explicitly rejecting "static, register-like locations." [SINGLE-SOURCED, secondary] ⚠ Directly adjacent to our coset-register framing and argues *against* register-like storage; worth reading before committing to registers.

**Symplectic/Hamiltonian sequence models:**
- **Chen et al., Symplectic Recurrent Neural Networks, ICLR 2020, arXiv:1909.13334** [VERIFIED — abstract fetched verbatim]: *"leverages symplectic integration, multiple-step training and **initial state optimization**"*. ⭐ **They optimize the initial state**, which is the mechanical half of "learn `(q₀,p₀)` as an address" — but as a *numerical-conditioning device for trajectory fitting*, not as an address. **We could not extract the passage explaining it** (page content insufficient). ⚠ **The engineer or curator must read §Initial State Optimization before we claim learned-`(q₀,p₀)` novelty** — this is the single most likely place a reviewer finds us preempted.
- coRNN (ICLR 2021, arXiv:2010.00951) — networks of controlled nonlinear oscillators, **provable gradient bounds** mitigating vanishing/exploding gradients. UnICORNN (ICML 2021, arXiv:2103.05487) generalizes it as a Hamiltonian system and is invertible in time [SINGLE-SOURCED — see §0].

**Has anyone built a Hamiltonian/symplectic *addressable* memory — storage at chosen locations, retrieval by initial conditions?** **I found none.** The nearest neighbours are Kong et al. (addressable dynamical attractors, non-Hamiltonian) and SRNN (Hamiltonian, initial-state optimization, not addressing). **[Confidence: moderate-to-low.]** This is a negative from four targeted searches, not an exhaustive sweep; the intersection is narrow enough that absence is plausible, but a negative from search is weak evidence. **Recommend one dedicated follow-up** (Semantic Scholar citation-graph walk forward from SRNN and from Kong et al.) before the claim is made in print.

---

## 5 · Four independent novelty verdicts

**(a) Mass / timescale as a retrieval key — *weakly novel; the weakest of the four.***
No prior art found for a *learned mass as the selector of which particle to launch*. But the function it performs — an index that selects which dynamical mode is retrieved — is **exactly Kong et al.'s index channel**, done more directly and with measured scaling. Multi-timescale memory architectures (multi-timescale LSTM, EMNLP 2015; AuGMEnT leaky/non-leaky units) occupy the "different timescales hold different information" ground. ⚠ **And a mass is a scalar** — a one-dimensional address into a large memory is a very low-bandwidth key, which compounds the §3 capacity problem. **Verdict: incremental.** Novel in mechanism, unoriginal in function, and structurally low-capacity.

**(b) Tunable-permanence writes (permanent latch / `μ²` half-life / deletable) — *partly already done.***
**EDEN analytically computes memory escape times and identifies a phase transition between static (permanent) and dynamic (decaying) regimes** — that is permanence as a derived, tunable quantity in an energy memory, published NeurIPS 2025. The trainable-forgetting/machine-unlearning literature independently covers deliberate deletion. **What survives as ours:** permanence as a **per-item design parameter chosen at write time**, with three *qualitatively distinct* modes (exact-zero latch / finite half-life / active deletion) grounded in **symmetry** (`μ²≡0` from an exact coset direction is a genuinely different guarantee from "slow decay"). **Verdict: incremental-to-novel, and only if stated in that narrow form.** The broad claim "we can tune memory permanence" is taken.

**(c) Test-time retry via symplectic boosts — *novel, and cleanly so, but low-stakes.***
The adaptive-compute lineage (ACT; **PonderNet** — halting as a latent-variable model with geometric prior and KL regularization; **DEQ**; path-independent equilibrium models, arXiv:2211.09961) all spend extra test-time compute, but **none preserve a geometric structure while doing it**. Lorentz-equivariant nets (LorentzNet, arXiv:2201.08187; Bogatskiy et al., ICML 2020) apply boosts at test time **only as equivariance *tests***, not as a retry mechanism. **I found no work using a symmetry transformation as an adaptive-compute retry operator. Verdict: novel.** ⚠ But it is a mechanism in search of a demonstrated benefit — novelty here does not carry a paper, and reviewers will ask what retry buys quantitatively.

**(d) Multi-particle parallel retrieval + consolidation — *already done, in two lineages.***
Multi-head attention is parallel retrieval + consolidation. DNC already used **multiple read heads** whose results are combined. Kong et al. handle multi-attractor itinerancy. **Verdict: already-done.** Do not claim this. It is a reasonable engineering choice; presenting it as novel invites an easy rejection. ⚠ And per §2.3 it inherits the chained-read degradation risk.

---

## 6 · The transformer-competitor claim, adversarially

**What a reviewer will demand.** Precedent is well established: **Long Range Arena** (arXiv:2011.04006) enforces that a new model be **"within at best 10% larger in terms of parameters compared to the base Transformer"** [VERIFIED] — parameter-matched comparison is the field's minimum bar, not a courtesy. Beyond that: (i) a **scaling study** across at least 2–3 model sizes, since Mamba's credibility came from 3B-scale parity, not from a single point; (ii) **associative-recall / MQAR** and copying diagnostics, because Jelassi et al. made these *the* discriminating tasks for any fixed-state architecture — **a memory architecture that does not report them will be assumed to fail them**; (iii) throughput/latency, since the whole non-attention case is a Pareto argument; (iv) standard LM perplexity if language is claimed at all.

**Is "competitor" defensible?** ⛔ **No — not as currently supportable.** The field reads "competitor to the transformer" as a **scaling-and-parity claim**, and the bar was set by Mamba (3B, matching 2× size) and Mamba-3 (1.5B, +1.8 pts over the best linear baseline, Pareto framing). Against that, we have **1–1.6 bits/register** and **no result beating a trivial baseline on a real task**. Claiming competitor status invites exactly the comparison we lose, on the axis (capacity/retrieval) where §1.3 shows we have no structural answer.

⚠ Note that **even Mamba-3 does not claim to replace the transformer** — its own abstract claims to *"advance the performance-efficiency Pareto frontier."* **If the strongest linear-model paper of 2026 is that modest, our claim cannot be stronger.**

**⭐ Recommended framing (strongest honest form):**
> *A physics-structured associative memory in which retention is a designed, per-item property — permanent (symmetry-protected), finite half-life (`μ²`-set), or actively deletable — rather than an emergent consequence of a learned recurrence. We characterize what such a landscape can store and for how long, and show retention behaves as the theory predicts.*

This claims **controllability**, which we can evidence, instead of **capability**, which we cannot. It positions against SSMs on an axis where they are genuinely silent, cites Ramsauer for the energy↔attention bridge, and never invites the capacity comparison.

**Framings that would sink us** (name these internally and refuse them):
1. **"A competitor to / replacement for the transformer."** Triggers scaling + parameter-matched + LM-perplexity demands. Instant reject.
2. **"Better memory than SSMs."** Jelassi et al. is a capacity theorem; our capacity is 1–1.6 bits/register. We lose on our own number.
3. **"Symplectic dynamics gives retention guarantees attention has no analogue for."** HiPPO has explicit theorems (§1.1); this reads as not knowing the literature.
4. **"Retrieval as a rollout rather than a weighted sum is novel."** Kong et al. 2024 (§4).
5. **"Continuous addressing lets us escape NTM/DNC's failures."** Factually wrong (§2). A reviewer who knows Graves's work will catch it, and it damages credibility on everything else.
6. **Any headline capacity claim.** §3.

---

## 7 · ⭐ Claims that survive contact with the literature vs claims that do not

**SURVIVE:**
- **Energy-landscape retrieval is a principled competitor-class mechanism to attention** — Ramsauer et al.: the modern Hopfield update *is* the attention mechanism. Well supported; cite prominently.
- **Retention specified per item at write time**, with three qualitatively distinct modes, one of them **symmetry-protected** (`μ²≡0` exactly, not approximately). Narrow but real.
- **Symplectic structure-preserving test-time retry** (novelty c) — no prior art found; genuinely unoccupied.
- **The reframing that CLU is a latent information carrier, not a model of the observed dynamics** — consistent with how HiPPO/SSMs are understood; defensible.
- **A symplectic + mass-structured *addressable* memory has not been built** — moderate-to-low confidence, needs one confirmatory citation-graph sweep (§4).

**DO NOT SURVIVE:**
- ⛔ **"Continuous addressing escapes the NTM/DNC precedent."** NTM/DNC were already continuous and soft. The three diagnosed failure modes are continuous-addressing pathologies, and **key/value entanglement — the "most important" one — is architectural in CLU**, since `q₀` is both key and initial condition.
- ⛔ **"Retrieval as a rollout from a learned address is novel."** Kong et al., *Nat. Commun.* 2024, location-addressable dynamical attractors via an index channel.
- ⛔ **"Symplectic ⇒ retention guarantees with no analogue elsewhere."** HiPPO-LegS: `O(tL/√N)` error, exact timescale equivariance, `Θ(1/t)` gradients. Theirs are theorems; ours is a structural property with no bound attached.
- ⛔ **"Multi-particle parallel retrieval + consolidation is novel."** Multi-head attention; DNC multi-read-heads.
- ⛔ **"Competitor to the transformer."** Unsupportable at 1–1.6 bits/register with no baseline win; and even Mamba-3 only claims a Pareto improvement.
- ⛔ **"Tunable memory permanence" as a broad claim.** EDEN (NeurIPS 2025) computes escape times analytically with a static/dynamic phase transition.
- ⚠ **"Mass as a retrieval key" as a headline novelty.** Functionally equivalent to Kong's index channel, and a scalar key is low-bandwidth.

---

## 8 · Confidence & what to search next

**Verified from primary sources (fetched, quoted):** HiPPO problem statement/ODEs/Props 3,5,6 · Ramsauer abstract · NTM abstract · Csordás & Schmidhuber's three DNC failure modes · Mamba-3 abstract + authors + date · EDEN abstract · SRNN abstract · Kong et al. mechanism, scaling exponents, failure mode · LRA 10% parameter rule.
**Single-sourced / secondary:** Ramsauer capacity exponent · the `ξ^new = X softmax(βXᵀξ)` form · UnICORNN's Hamiltonian claim and integrator · Mamba's post-2024 competitive standing · the "why MANNs declined" synthesis · Graves's "blurry" wording (paper text via secondary quotation).
**Negative results (weak evidence):** no Hamiltonian/symplectic addressable memory found; no symmetry-transformation-as-retry found.

**Search next, in priority order:**
1. ⭐ **SRNN §"initial state optimization" — read the actual passage.** Highest preemption risk for novelty (a)/(c). Needs a tool that can render arXiv:1909.13334 in full.
2. ⭐ **Ramsauer Theorem 3 (capacity) exact statement** — needed before any capacity sentence is printed. Try the ICLR 2021 camera-ready PDF or the `hopfield-layers` GitHub companion.
3. **Citation-graph walk forward from Kong et al. 2024 and SRNN** to confirm the "no Hamiltonian addressable memory" negative.
4. **MQAR / associative-recall benchmark specifics** (Zoology / Based line, Arora et al.) — if we report any retrieval diagnostic, it must be the one the field uses.
5. **Karuvally's traveling-waves paper (arXiv:2402.10163)** — it argues *against* register-like storage; a direct challenge to the coset-register route.

---

## Proposed handover updates (for the Hub)

1. **Correct the 2026-07-21 vision block, consequence 1.** Replace "much of NTM/DNC's instability is attributed to HARD/discrete addressing" with: *NTM/DNC were fully soft and differentiable; their diagnosed failures (Csordás & Schmidhuber, ICLR 2019) are continuous-addressing pathologies. Continuity is not an escape. CLU must independently defend against key/value entanglement, de-allocation aliasing, and chained-read degradation.*
2. **Add a design requirement: key/value separation.** CLU's `q₀` is currently both address and initial condition — architecturally the defect Csordás & Schmidhuber rank most important. Consider an address embedding distinct from the launch state. **Recommend this go to the physics-theorist as a first-order design question.**
3. **Add to the prior-art table:** Kong et al. *Nat. Commun.* 2024 (nearest neighbour — location-addressable dynamical attractors, `N_c ∝ K^1.08`, basin-violation failure mode); Karuvally et al. GSEMM ICML 2023 / EDEN NeurIPS 2025 (energy memory, escape times, `O(γ^N)`); Mamba-3 ICLR 2026 (complex-valued state updates = the field's own oscillatory answer, and it is linear).
4. **Adopt the §6 framing; add the six sinking framings to a "never write this" list.**
5. **Two follow-up scout tasks** are worth spawning: (i) SRNN initial-state-optimization extraction + Ramsauer Theorem 3 — both are preemption/accuracy risks with a tool requirement I could not meet; (ii) the citation-graph sweep confirming the Hamiltonian-addressable-memory negative.
6. **Basin-violation-on-hand-off is a free, cheap, pre-registrable experiment** for multi-particle consolidation and wormholes, handed to us by Kong et al.'s failure analysis.
7. ⚠ **Flag for the Head:** the capacity comparison in §3 is the program's most exposed flank. Recommend deciding *now*, at program level, that CLU does not compete on capacity, and that every write-up says so before a reviewer asks.

---

## Bibtex-ready refs

```bibtex
@inproceedings{gu2020hippo,
  title={HiPPO: Recurrent Memory with Optimal Polynomial Projections},
  author={Gu, Albert and Dao, Tri and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2020},
  note={arXiv:2008.07669}}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri}, journal={arXiv preprint arXiv:2312.00752}, year={2023}}

@inproceedings{lahoti2026mamba3,
  title={Mamba-3: Improved Sequence Modeling using State Space Principles},
  author={Lahoti, Aakash and Li, Kevin Y. and Chen, Berlin and Wang, Caitlin and Bick, Aviv and Kolter, J. Zico and Dao, Tri and Gu, Albert},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2026},
  note={arXiv:2603.15569}}

@inproceedings{jelassi2024repeat,
  title={Repeat After Me: Transformers are Better than State Space Models at Copying},
  author={Jelassi, Samy and Brandfonbrener, David and Kakade, Sham M. and Malach, Eran},
  booktitle={International Conference on Machine Learning (ICML)}, year={2024},
  note={arXiv:2402.01032}}

@article{graves2014ntm,
  title={Neural Turing Machines},
  author={Graves, Alex and Wayne, Greg and Danihelka, Ivo},
  journal={arXiv preprint arXiv:1410.5401}, year={2014}}

@inproceedings{csordas2019improving,
  title={Improving Differentiable Neural Computers Through Memory Masking, De-allocation, and Link Distribution Sharpness Control},
  author={Csord{\'a}s, R{\'o}bert and Schmidhuber, J{\"u}rgen},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2019},
  note={arXiv:1904.10278}}

@inproceedings{ramsauer2021hopfield,
  title={Hopfield Networks is All You Need},
  author={Ramsauer, Hubert and Sch{\"a}fl, Bernhard and Lehner, Johannes and Seidl, Philipp and Widrich, Michael and Adler, Thomas and Gruber, Lukas and Holzleitner, Markus and Pavlovi{\'c}, Milena and Sandve, Geir Kjetil and Greiff, Victor and Kreil, David and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2021},
  note={arXiv:2008.02217}}

@article{amit1985storing,
  title={Storing Infinite Numbers of Patterns in a Spin-Glass Model of Neural Networks},
  author={Amit, Daniel J. and Gutfreund, Hanoch and Sompolinsky, Haim},
  journal={Physical Review Letters}, volume={55}, number={14}, pages={1530--1533}, year={1985}}

@article{kong2024reservoir,
  title={Reservoir-computing based associative memory and itinerancy for complex dynamical attractors},
  author={Kong, Ling-Wei and Brewer, Gene A. and Lai, Ying-Cheng},
  journal={Nature Communications}, volume={15}, year={2024},
  doi={10.1038/s41467-024-49190-4}}

@inproceedings{karuvally2023gsemm,
  title={General Sequential Episodic Memory Model},
  author={Karuvally, Arjun and Sejnowski, Terrence J. and Siegelmann, Hava T.},
  booktitle={International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={202}, pages={15900--15910}, year={2023}}

@inproceedings{karuvally2025eden,
  title={Exponential Dynamic Energy Network for High Capacity Sequence Memory},
  author={Karuvally, Arjun and Lertsaroj, Pichsinee and Sejnowski, Terrence J. and Siegelmann, Hava T.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2025},
  note={arXiv:2510.24965}}

@inproceedings{chen2020symplectic,
  title={Symplectic Recurrent Neural Networks},
  author={Chen, Zhengdao and Zhang, Jianyu and Arjovsky, Martin and Bottou, L{\'e}on},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2020},
  note={arXiv:1909.13334}}

@inproceedings{rusch2021cornn,
  title={Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies},
  author={Rusch, T. Konstantin and Mishra, Siddhartha},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2021},
  note={arXiv:2010.00951}}

@inproceedings{tay2021lra,
  title={Long Range Arena: A Benchmark for Efficient Transformers},
  author={Tay, Yi and Dehghani, Mostafa and Abnar, Samira and Shen, Yikang and Bahri, Dara and Pham, Philip and Rao, Jinfeng and Yang, Liu and Ruder, Sebastian and Metzler, Donald},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2021},
  note={arXiv:2011.04006}}

@article{banino2021pondernet,
  title={PonderNet: Learning to Ponder},
  author={Banino, Andrea and Balaguer, Jan and Blundell, Charles},
  journal={arXiv preprint arXiv:2107.05407}, year={2021}}
```
⚠ **PonderNet's arXiv ID (2107.05407) and author list are from training knowledge, NOT verified this session** — my searches surfaced PonderNet only through secondary/derivative papers. Verify before citing.
