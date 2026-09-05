# di-bernardo-skim — web-scout report
Task + acceptance criterion: Skim Di Bernardo et al. 2025 (arXiv:2511.04802, Mo's "closest work") + Keller 2025 for allocation-adjacent claims; verdict each of 5 questions with quotes; draft V2 positioning prose citing/differentiating from Di Bernardo + Mo + Keller together; quick citation-count check on Mo (2605.03338) & Welling (2605.14685).
Status: done

## ANSWER FIRST
**No red alert. The V2 "allocation with a price list" claim is safe against Di Bernardo and Keller.** Di Bernardo et al. build a *geometric* theory of equivariant-RNN attractor manifolds — which symmetry in the connectivity produces which fixed-point manifold, of what dimension, and whether it is stable or a saddle. They **construct** manifolds but never as a **memory/capacity budget**, and they have **zero temporal-degradation content**: no lifetime, decay rate, forgetting time, dissipation/friction parameter, or capacity-vs-stability *pricing* — verified by two independent term-absence passes ("memory"/"capacity"/"symmetry breaking"/"lifetime" all not found in intro/discussion/conclusion). Keller 2025 is orthogonal again: *flow*-equivariance (hidden state transforms correctly under continuous motion), not retention or allocation. So all four load-bearing pieces of V2's allocation section — (i) task-agnostic channel *budgeting* via chosen G/H, (ii) band placement by the exact (γ, ε, μ) lifetime table, (iii) mixed exactly-protected + deliberately-lifted bands, (iv) the kinetic-isotropy price — remain unclaimed. The one genuine adjacency: Di Bernardo **constructs** equivariant attractor geometry and analyzes **coexistence of subgroup manifolds** (some stable, some saddles) — so V2 must present allocation as *pricing/budgeting a temporal resource*, never as "we choose the architecture's symmetry group" (that construction move is theirs).

**Citation check:** Mo (2605.03338) = **2 citers** (Hongbo Wang 2606.24946; and Iqbal/Welling 2605.14685 itself cites Mo). Welling (2605.14685) = **0 citers**. [Semantic Scholar, this session.]

---

## EVIDENCE

### Target: Di Bernardo, Valente, Mastrogiuseppe, Ostojic (2025)
"Shaping manifolds in equivariant recurrent neural networks", **arXiv:2511.04802**, q-bio.NC, 46pp / 7 figs, v1 6 Nov 2025 / v2 13 Nov 2025. [verified: arXiv abstract page + HTML full text, two independent fetches, consistent]

What the paper *is* (abstract, verbatim): "we introduce a new class of equivariant RNNs, where the connectivity is based on group convolution. Using the group Fourier transform, we reduce such networks to low-rank models… that can be fully analyzed to determine the **symmetry, dimensionality and stability of fixed-point manifolds**." → group-representation-theory account of continuous-attractor **geometry**, motivated by neuroscience (neural manifolds at rest/sleep).

**Q1 — Symmetry group / dim(G/H) as a design knob for memory capacity? → NO (adjacent on "construct", absent on "budget").**
- Constructive about geometry: "Our framework unifies a variety of existing models and offers a principled approach to **build new types of continuous attractor models**" [verbatim]. The symmetry group *is* an engineered knob — but the design target is **manifold shape/dimension/stability**, not a memory budget.
- No capacity counting. The only dimension-count is architectural rank: "at most R=∑ᵏₖ₌₀ dₖ² scalar values" = the **low-rank embedding** from the group-Fourier reduction, **not** memory capacity or a channel budget. [verified]
- "memory" / "capacity" / "working memory" / storing multiple items: **not found** in intro/discussion/conclusion [2-pass verified].

**Q2 — Lifetime / retention / constitutive decay-rate? → NO. (Cleanest differentiator.)**
- "No formulas appear for forgetting timescales, decay rates, or retention windows. The paper focuses on *fixed points* and manifold geometry, not temporal dynamics of memories. **There is no dissipation parameter or friction term** controlling stability over time." [verified]
- Stability is a *binary/geometric* property (stable / marginal / saddle), not a quantified timescale. No parameters→decay-rate map anywhere. This is the axis V2's entire constitutive theory lives on, and it is simply absent here.

**Q3 — Pricing / trade-off language (capacity vs stability vs speed)? → NO.**
- "No explicit pricing or trade-off language. The authors emphasize stability analysis but do not quantify cost functions balancing multiple objectives." [verified] They flag stability *must be considered* (some manifolds are saddles) — a caveat, not a price list.

**Q4 — Multi-channel / mixed exactly-protected + deliberately-lifted bands? → NO (emergent coexistence only, not designed bands).**
- No deliberate symmetry-breaking scheme: "symmetry breaking" / "broken symmetry" / "approximate symmetry" / "perturbation" / "pseudo": **not found** [2-pass verified]. (Contrast Mo, who *does* break symmetry to get pseudo-gaps.)
- Closest object is *emergent* multiplicity, verbatim: "for a connectivity with a given symmetry, depending on parameters, **several manifolds with different symmetry subgroups can coexist, some stable and others consisting of saddle points**." This is **parametric/emergent coexistence** of subgroup manifolds — *not* a designed mix of protected + deliberately-lifted memory bands, and carries no lifetime for the "lifted" (saddle) ones (they're just unstable, not slow-decaying registers).

**Q5 — stability framing (context):** "all fixed points belonging to a given manifold have identical stability properties… if [the manifold] is non-trivial, each fixed point will be marginally stable" [verified]. Marginal stability = their analog of our neutral direction, but characterized geometrically (Jacobian null-space on the orbit), with **no damped/inertial dynamics on top** — so no crossover, no ringing, no saturation, no write-current. Everything the (γ, ε, μ) budget adds is out of their frame.

### Secondary (skim): Keller (2025)
"Flow Equivariant Recurrent Neural Networks", T. Anderson Keller, **arXiv:2507.14793**, **NeurIPS 2025** (poster). [verified: arXiv listing + NeurIPS 2025 poster page]
- Thesis: existing equivariant nets handle *static* transforms; sequences carry *time-parameterized* ("flow") symmetries. FERNNs make the **hidden state transform geometrically with moving stimuli**, improving generalization/speed on motion sequences.
- Verdict: **orthogonal**. About *equivariance to continuous motion* (state co-transforms with a moving stimulus), **not** memory retention, allocation, damping, or capacity budgeting. No lifetime/dissipation content. Shares only the "continuous-symmetry-in-time RNN" umbrella with V2; does not touch what V2 prices.

### Citation-count check (task sub-question)
- **Mo, arXiv:2605.03338** ("Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks", 2026): **citationCount = 2**, influential = 0. Citers: (1) **Hongbo Wang, "Conformal Orbit-Valid Trust Horizons for Equivariant World Models", arXiv:2606.24946, 2026**; (2) **Iqbal et al. (Welling), arXiv:2605.14685** — i.e. the Goldstone paper itself cites Mo. [Semantic Scholar API, this session]
- **Welling/Iqbal, arXiv:2605.14685** ("Spontaneous symmetry breaking and Goldstone modes for deep information propagation", 2026): **citationCount = 0**, influential = 0. [Semantic Scholar API, this session]
- ⚠ Note: handover §10 (2026-07-06) logs "Wang 2026 (**2606.24945**) caveat for V3 certified-horizon claims" — the Mo-citer here is **2606.24946** (adjacent id, same author Hongbo Wang, "…Trust **Horizons**…"). Very likely the same paper/author cluster; reconcile the exact id. Wang couples "equivariant world models" + "trust horizons" and cites Mo → relevant to *both* V2 (equivariance) and V3 (certified horizons).

---

## RELEVANCE TO CHLU (V2 allocation section)
**Net:** Di Bernardo is a **citable neighbor, not a competitor** on allocation. It strengthens V2 by anchoring "here is prior art that *constructs* equivariant attractor geometry," against which our novelty (temporal *budgeting/pricing*) is sharp.
- **Borrow / cite for:** group-rep-theory construction of equivariant attractor manifolds (G-conv connectivity → manifold of prescribed symmetry/dimension); coexistence-of-subgroup-manifolds (stable + saddle) as prior evidence that *which* subgroup survives is parameter-dependent.
- **Differentiate on (all four unclaimed by them):** (i) allocation as a **temporal-memory budget** (channels = protected directions rationed against a retention target), not geometry for its own sake; (ii) **constitutive lifetime** gap = (2−γ)ε²μ²/2γ with crossover + saturation floor — they have *no* time axis; (iii) **deliberate mixed bands** (exactly-protected latch + pseudo-Goldstone slow-decay registers) vs their *emergent* stable/saddle split; (iv) the **kinetic-isotropy price** (Schur multiplet-mass condition) — inexpressible in their (no T/V/M/γ decomposition) framework, exactly as with Mo.
- **Guard-rail:** because Di Bernardo owns "construct the attractor by choosing the connectivity's symmetry group," V2 must **not** headline bare "we choose G/H as an architectural knob." Headline the *pricing*: task-agnostic channel count **sized by a constitutive lifetime table**, with band placement and a kinetic-isotropy cost. This skim confirms the Hub's "allocation with a price list" wording is the defensible framing (upgrading the earlier flat "No", per mo-deep-read §2(e)).
- **Keller:** one-line cite as the flow-equivariant-RNN reference; differentiate in the same breath (equivariance-of-motion, not retention). Also a useful *baseline-family* pointer if V2 needs an equivariant-RNN comparator that is NOT second-order.

### Draft prose to lift — V2 related-work paragraph (Di Bernardo + Keller, dovetails with mo-deep-read §4's Mo paragraph)
> Constructive accounts of equivariant recurrent memory are the closest prior art to our allocation results. Di Bernardo et al. (2025) use group representation theory to link the symmetry of an RNN's (group-convolutional) connectivity to the symmetry, dimensionality, and stability of its fixed-point manifolds, showing that several subgroup manifolds can coexist — some stable, others saddles — as connectivity parameters vary; Keller (2025) extends equivariance from static transforms to continuous *flows*, so that a recurrent state co-transforms correctly with a moving stimulus. Both engineer *which* symmetry a network expresses. What neither prices is **time**: their manifolds are analyzed as geometric fixed-point sets, with no dissipation parameter, decay rate, or retention timescale, and no notion of budgeting protected directions against a memory cost. Our contribution is to treat the choice of coset G/H as an *allocation with a price list*: each protected direction is a memory channel whose retention is set constitutively by the damped-Hamiltonian lifetime law — gap = (2−γ)ε²μ²/2γ below a critical-damping crossover, saturating at a mass-independent floor 2ln2/(−ln(1−γ)) above it — so that channels can be *sized* (exactly-protected latches vs deliberately-lifted pseudo-Goldstone registers of a chosen half-life) and *priced* (the kinetic-isotropy / multiplet-mass constraint of §[X]) rather than merely constructed. Where Di Bernardo et al. obtain marginal-stability geometry and Mo (2026) supplies the kinematics that force dim(G/H) neutral directions, we supply what a purely geometric or purely spectral account cannot: the temporal budget those directions buy, and its cost.

---

## CONFIDENCE & GAPS
- **High confidence** on Di Bernardo verdicts Q1–Q4: the *negatives* (no lifetime, no capacity budget, no pricing, no symmetry-breaking) were each confirmed on **two independent targeted reads** of the HTML full text, consistent with the paper's self-description as a q-bio.NC *fixed-point-geometry* theory. The absence of a friction/decay axis is structural, not a snippet I missed.
- **Single-source / not deep-read:** figures not visually inspected (text + captions only, via LaTeXML HTML — same caveat class as mo-deep-read). Keller assessed at **abstract/summary level only** (task = skim/secondary); a full read is warranted (~half-thread) only if V2 leans on Keller as a baseline or nearer competitor.
- **Citation counts are Semantic-Scholar-single-sourced** and S2 undercounts very recent arXiv; treat Mo=2 / Welling=0 as *lower bounds* as of 2026-07-06. Refresh ~2 weeks pre-freeze (matches mo-deep-read §6). Note: **Welling's Goldstone paper cites Mo** — the two V2 threats are cross-linked; Wang (2606.24946) is the one external builder on Mo so far.
- **Next searches (if wanted):** (1) reconcile Wang 2606.24945 vs .24946 and read it (cites Mo, "trust horizons" → V2 + V3 certified-horizon); (2) full Keller read only if it becomes a baseline; (3) forward-citation watch on Di Bernardo for anyone extending their geometry toward temporal/allocation.

## Bibtex-ready refs
```bibtex
@article{dibernardo2025shaping,
  title  = {Shaping manifolds in equivariant recurrent neural networks},
  author = {Di Bernardo, Arianna and Valente, Adrian and Mastrogiuseppe, Francesca and Ostojic, Srdjan},
  journal = {arXiv preprint arXiv:2511.04802},
  year   = {2025},
  note   = {v1 6 Nov 2025, v2 13 Nov 2025; q-bio.NC; 46pp, 7 figs}
}
@inproceedings{keller2025flow,
  title     = {Flow Equivariant Recurrent Neural Networks},
  author    = {Keller, T. Anderson},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2507.14793}
}
@article{mo2026symmetry,
  title  = {Symmetry-Protected Lyapunov Neutral Modes in Equivariant Recurrent Networks},
  author = {Mo, Hanson Hanxuan},
  journal = {arXiv preprint arXiv:2605.03338},
  year   = {2026}
}
@article{iqbal2026spontaneous,
  title  = {Spontaneous symmetry breaking and Goldstone modes for deep information propagation},
  author = {Iqbal, Nabil and Keller, T. Anderson and Song, Yang and Miyato, Takeru and Welling, Max},
  journal = {arXiv preprint arXiv:2605.14685},
  year   = {2026},
  note   = {author order/affiliations per handover; verify before camera-ready}
}
@article{wang2026conformal,
  title  = {Conformal Orbit-Valid Trust Horizons for Equivariant World Models},
  author = {Wang, Hongbo and others},
  journal = {arXiv preprint arXiv:2606.24946},
  year   = {2026},
  note   = {cites Mo 2605.03338; reconcile id vs handover's 2606.24945}
}
```

## Proposed handover updates (for the Hub)
- **Related-work ledger (V2):** Di Bernardo et al. 2025 (2511.04802) SKIMMED — **contrast-clean on allocation**. They *construct* equivariant attractor manifolds (group-conv connectivity → symmetry/dimension/stability of fixed-point manifolds) and show emergent coexistence of subgroup manifolds (stable + saddle), but have **zero temporal content** (no lifetime, decay, friction, capacity budget, pricing, or deliberate symmetry-breaking — 2-pass term-absence verified). "Allocation with a price list" confirmed defensible; guard-rail = don't headline "we choose G/H as an architectural knob" (their construction move) — headline the *pricing/budgeting of retention*. Keller 2025 (2507.14793, NeurIPS 2025) = orthogonal (flow-equivariance of state under motion, not retention). Draft V2 related-work paragraph in report §"Draft prose to lift".
- **Citation watch:** Mo (2605.03338) = 2 citers [Wang 2606.24946; Welling 2605.14685]; Welling (2605.14685) = 0 citers; both S2 lower-bounds as of 2026-07-06. **Welling's Goldstone paper cites Mo** (the two V2 threats are cross-linked). **Wang 2606.24946** is the sole external builder on Mo and pairs equivariance with "trust horizons" → likely == the handover's V3 caveat paper (logged as 2606.24945; reconcile the id, consider a read).
- **Nomenclature:** Di Bernardo's "marginal stability" is a *geometric* neutrality (Jacobian null-space), distinct from our γ-latched Lyapunov-neutral memory; keep "marginally stable manifold" (theirs) vs "deadbeat latch / pseudo-Goldstone register" (ours) lexically separate in the V2 short.
- **Ops flag:** this web-scout thread initially had Write/Edit disabled (delivered inline first), then was re-enabled and this file was written on retry. Confirm web-scout is spawned with Write enabled so scouts can self-write outputs per protocol §2/§5.

Git footprint: none (read-only research; no tracked code touched; only this untracked `.claude/outputs/` artifact written).
