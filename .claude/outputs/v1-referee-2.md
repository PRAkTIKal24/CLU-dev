# v1-referee-2 — paper-referee report (clean-pass re-review of the revised V1 short)

Task + acceptance criterion: verify MF-1 + F1–F7 closure on `.claude/papers/v1-short/draft.md` (v0.3); hunt NEW inconsistencies introduced by the w12/w13 edits; render the submission-readiness verdict and the F3 determination.
Status: done. **Report only — draft not edited.**

---

## VERDICT: **borderline** (was weak-accept; the revision fixed all seven craft items and *earned* F3, but introduced two hard errors and exposed one pre-existing correctness problem — all three sit in the abstract/§3 headline)

**Meta-review.** The punch-list is genuinely closed. F3 was closed on the strong branch and it works: the certificate now has a measured downstream consequence, and — the thing I most expected to be lost in drafting — **Payoff B's fine print survived intact and in the right places.** "The receipt buys boundedness, not the jump," "`wormhole_blind ≡ no_physics_router` exactly," and "coercive-*component* membership, not the energy ledger, is the operative clause" appear in the abstract, §3.1, §3.2.1(iii), §5, App B.2 item 4, and App D. The attack the task file told me to run — Payoff B sold as "the certificate prevents blow-up" — **does not fire.** CM-3 discipline is likewise clean: I swept for hedged and implied forms of energy-as-superior-signal and found none; the three disclaimers (§4.1 N3, §4.2, §5/F.5) are present and load-bearing. CM-8's noise wall is now *plotted* (Fig 2c), travels with every reversal claim including the abstract, and the cost claim is scoped intra-CLU in all seven places it appears. F6's kv16 is named and the decay (1.57×/1.14×) is stated with the surviving invariant. F1, F4, F5, F7 all closed.

**But.** The revision's new prose, written to make the payoff visible on page 1, fuses two different objects that the sources keep apart: §3's `no_physics_router` is an *untrained analytic constant map* `(q,p)↦(b,p)`; §4.2's router is a *449-param learned decision head that routes through the paper's own det J = 1 wormhole edge* (`v1-router-baseline` L17: "Routes via the **same** direct wormhole edge"). The abstract calls the det J = 0 object "a physics-free **learned** router"; §3.2.1(ii) calls it "**the same** physics-free router"; §5 says it "reaches the same targets **more cheaply**" (a §4.2 FLOPs number, for a different arm, on a different task). As written, the paper's flagship new sentence is contradicted by its own §4.2 apparatus — **MF-A**. Separately, the reach panel's headline falsification ("the squeeze collapses past the causal box") is, by the paper's *own bracket formula* and its *own* Appendix A.1 ζ-grid, an artifact of stopping the line search at ζ = 2.0: the required ζ to land d = 4.0 is **2.0105** — **MF-B**. And the F6 fix, by importing kv24/kv32 numbers, made constructible the exact cross-section contradiction C-7 exists to forbid: §4.1's memory has fidelity 0.717 / accuracy 0.286 at kv32, while §4.3's kv32 has 1.00 / 1.00 and calls the low regime "an under-training artifact" — **MF-C**.

None of the three requires new physics. MF-A and MF-C are rewrites (and MF-A's rewrite makes the paper *better*: decision ≠ transport is the real content). MF-B needs one ~minutes-long analytic run or a restatement of the theorem as a pricing law. Fixed, this is a clean weak-accept and, with the ζ-pricing result, plausibly an accept. Unfixed, quotes #1 and #2 below end the review.

**MF-1 (stated once, not re-itemized):** the theory note is still `Anonymous (2026) [placeholder]`; §2–3's load-bearing Prop-A2/A6/12 remain unreachable by a reviewer. Standing submission dependency across all three shorts; Head critical path, not a referee item.

---

## F1–F7 + MF-1 closure audit

| item | status | note |
|---|---|---|
| **F1** headline occludes hero | **CLOSED** — 2-panel Fig 1; per-arm det-J receipt column in the §3.2 table and legend | fix introduced **S5** (offsets put landing rates above 1.0 / below 0.0) |
| **F2** noise wall in no figure | **CLOSED, exemplary** — Fig 2c plots it, with the fidelity≈1.0 annotation and honest n-mismatch disclosure | — |
| **F3** "beats the router" definitional | **CLOSED on the strong branch** (see determination below) | framing is **MF-A**; the *measurement* is sound |
| **F4** §5 MH wall-of-text | **CLOSED** — 4 numbered design rules + App F | — |
| **F5** BIBO caveat not in main text | **CLOSED and upgraded** — §3.1 paragraph, measured, not asserted | best fix in the revision |
| **F6** 4.8× unscoped | **CLOSED** — kv16 named, decay 1.57×/1.14× stated | fix exposed **MF-C** |
| **F7** internal §-number in figure | **CLOSED** — verified all three PNG titles; no `§7.1` leak | but **S7**: an editorial note leaked into Fig 3's *caption* |
| **MF-1** theory note | **OPEN** (Head) | stated once, per task |

---

## Charter / claims-matrix compliance

- **C-1 (no audit confession):** PASS. §3.3 and §4.3's "under-training artifact" are C-9 negatives about the program's own results, not legacy-paper mechanism numbers.
- **C-2 (designed = verification, learned = evidence):** PASS. §3 header, §3.2.1, App B.2 all carry the verification tag; "evidence" is reserved for §4. Contribution 3 is correctly tagged `[verification; designed testbed, oracle placement.]`
- **C-5 (scale qualifiers):** PASS with one gap — the abstract's payoff sentence ("The receipt is not a label: the wormhole reaches with a receipt…") starts a new claim without re-attaching scope in-sentence; it inherits scope from the *previous* sentence only. No bare "CLUs provide". (S-minor.)
- **C-6 (certificate altitude):** **PASS — and the standout improvement.** Every fine-print clause sits next to its claim: e^{2|ζ|} = matched-quadratic-H (§3.1), frozen-gate det J = 2.05 (§3.1), volume-alone-≠-latch-receipt (§3.1, §3.2.1(i), abstract, App D), coercive-component-not-ledger (§3.1, abstract, §5, App B.2), LTT exchangeability + ECE ≈ 0.100 (§4.1.3). Last round's F5 placement gap is closed.
- **C-7 (flag-provenance / no constructible contradiction):** **FAIL — MF-C.** The A.5 note asserts "no two numbers in this paper are drawn from configs a reviewer could read as contradictory." A reviewer can read A.2 (epochs 400) against A.4 (train_epochs 2000) and §4.1 (kv32: 0.286) against C.4.a (kv32: 1.00) in ninety seconds. The note enumerates γ and `langevin_noise` and omits the one flag that matters.
- **C-8 (hermetic citations):** PASS on structure; **MF-1** open; **S10**: Geifman & El-Yaniv (2017) and Wales & Doye (1997) are in the reference list and cited nowhere in the text.
- **C-9 (negatives):** PASS. App D carries N1/N2/N2b/N3/N24/N30/N31/N23 + three new payoff negatives. **MF-E**: N27 and N21 are *cited* (§4.2 main text; App D N24) and *never defined*.
- **C-10 (appendix maximalism):** PASS. App A–F full; placeholders intact.
- **CM-2 / CM-3 / CM-7 / CM-8 / CM-12 / CM-14:** approved wordings used. **CM-3 forbidden-claim sweep: clean** — no positive-form or hedged energy-as-superior-signal claim anywhere; §3's "energy ledger" is bookkeeping on the reach/escape axis, §5's EBT contrast is structural, F.5 states the deflation outright. **CM-7:** the payoff wording is instantiated verbatim with all three qualifications. **CM-8:** the noise wall travels with every reversal claim (abstract, contribution 6, §4.3.3, Fig 2c caption, App D N2b) and cost is scoped intra-CLU everywhere; never a saving vs Hopfield.

### Claim-evidence audit — abstract + contributions, every numeral traced (task item 7)

Last round: zero mismatches. **This round: two.** Both introduced by the w12/w13 edits.

| draft claim (abstract / contributions) | source | verdict |
|---|---|---|
| $L_i=T\varepsilon c/\sqrt{M_i}$; energy-blind cap | `paid-access-theory` L47, Prop-A2 | exact ✓ |
| injection $\le e^{2|\zeta|}H$; $(\zeta,$ratio,bound$)$ quadruples; $\det S^{(M)}=1.000\pm4$e-6 | `paid-access-experiments` L56 | exact ✓ |
| bracket $[L, L+p_0\sinh\zeta/M_0]$; observed edge $d\approx3.2$ | `paid-access-experiments` L42, L93 | exact ✓ **but see MF-B** |
| $\det J=1$ exact, ledger $=0$ exact, latch shift $p^\top X\Delta$ | `v1-certificate-payoff` L98 | exact ✓ |
| router $\det J = 0$ *exactly* (measured, autodiff) | `v1-certificate-payoff` L14, L60 | exact ✓ |
| std$(Q_{\rm out})$=std$(Q_{\rm in})$=0.0803; router std=0; 16 probes bit-identical; $\Delta Q=p^\top X\Delta$ to 1.2e-7; reconstruction 2.2e-8 | `v1-certificate-payoff` L57–63 | exact ✓ |
| free $\Delta H=0$ exit escapes, $r^\ast\propto T$, growth 1.91 | `v1-certificate-payoff` L84, L82 | exact ✓ |
| "a physics-free **learned** router cannot supply [the receipt]" (abstract) | — | **MISATTRIBUTED — MF-A** |
| $4.81\pm0.44\times$ @kv16, above always-full; $1.57\times$@kv24; $1.14\times$@kv32 | `v1-pivot` L39–40 | exact ✓ |
| LTT 30/30; ECE $0.100\pm0.021$; coverage $0.647\pm0.063$ | `v1-pivot` L46 | exact ✓ |
| Hopfield transfer raw 0.18 → calib 0.88 ≈ CLU 0.43 → 0.87 | `minus-the-physics` L58, L67 | exact ✓ |
| router $1.000/0.948$ @8.81e7 vs gated $0.887/0.715$ @1.18e8; 449 params; chain 1.76→2.94e8; distant 0.41→0.28 | `v1-router-baseline` L10, L38, L44 | exact ✓ |
| Hopfield ceiling ≈1 matvec; 9–10× is intra-CLU | `regime-remap-2000ep` Item 3 | exact ✓ |
| "On the full grid (**198 jobs, $n=8$**)" (contribution 6) | `regime-remap-2000ep` L69, L90, L251 | **SCOPE-WIDENED — MF-D** (n=8 is the corr=0 capacity axis only; stress n=5, frontier n=3) |
| gate 0.36 vs Hopfield 0.71 @σ=0.6/kv32; 0/6; fidelity ≈1.0; 6/15 | `regime-remap-2000ep` L197–198, L206 | exact ✓ |

---

## Itemized findings

### MUST-FIX (blocks submission)

**MF-A — The paper's flagship sentence attributes `det J = 0` to an object that, in this same paper, routes through a `det J = 1` channel.**
*Location:* abstract ("a physics-free **learned** router cannot supply"); §3.2.1(ii) ("**the same** physics-free router"); §5 ("a $\det J=0$ router, which reaches the same targets **more cheaply**").
*The attack:* there are two arms named "router" and they have opposite transport semantics.
- §3 / `no_physics_router` — an **untrained analytic map** `(q,p) ↦ (b,q)`… `(b,p)`. A.5 states plainly: `training: none`. It replaces the state. Hence det J = 0, hence erasure.
- §4.2 / `router_mlp` — a **449-param learned *decision head*** on the raw query cue. `v1-router-baseline` L17: *"Routes via the **same** direct wormhole edge; because its decision needs no settle, a routed query skips phase-1."*

So the §4.2 learned router **transports through the wormhole**, inherits its det J = 1 receipt, and erases nothing. The abstract's claim that "a physics-free learned router cannot supply" the receipt is false *of the only learned router in the paper*. §3.2.1(ii)'s "it destroys phase-space information when it *goes* there" is false of that router. And §5's "more cheaply" imports 8.81e7 vs 1.18e8 — a §4.2 MQAR FLOPs number for the decision head — onto §3's analytic constant map, for which no cost is measured anywhere.
*Why it matters beyond pedantry:* it re-opens F3 one level down. The receipt is a property of the **transport map**, not of physics-free-ness or of learnedness. The measured content is *"a state-replacing (absorbing) jump annihilates phase volume and erases the coset register; a canonical translation transports it."* That is true, defensible, and **strictly stronger** than what is written, because it survives the observation that a learned router can be bolted onto a certified edge — which the paper's own §4.2 does.
*Triage:* **MUST-FIX.** Rewrite: (a) rename §3's arm `absorbing_router` / "state-replacing jump (no-physics baseline)" and say once that it is analytic and untrained; (b) delete "learned" from the abstract; (c) §3.2.1(ii) becomes *"§4.2's learned router is a **decision head** that routes through this same certified edge — decision and transport are orthogonal, and the receipt prices only the latter"* (this is a **better** paragraph and defuses hostile-quote #1); (d) §5 drops "more cheaply" or attributes it to §4.2's routing task explicitly.

**MF-B — "Provably cannot beat the box" is falsified by the paper's own bracket, and the headline collapse at d = 4.0 is a ζ-grid artifact with a margin of Δζ = 0.011.**
*Location:* abstract ("provably cannot beat the box"); contribution 2 ("**cannot** beat the box"); §5 ("provably cannot buy reach"); Fig 1a annotation ("squeeze collapses past the box"); §3.2(i).
*The attack:* the squeeze is not a flow. Applied at the well bottom it delivers an **instantaneous canonical displacement** $p_0\sinh\zeta/M_0$ (`paid-access-theory` L126–129; `paid-access-experiments` L42), which is **unbounded in ζ**. The post-squeeze flow then adds at most $L$. So the reachable radius is $L + p_0\sinh\zeta/M_0$ — the paper's own bracket — and it grows without limit. The squeeze *does* beat $C_T(q_0)$; it beats it by an additive, ζ-controlled amount.

Now do the reviewer's arithmetic from Appendix A.1 alone ($L=2.5$, $p_0=1.2$, $M_0=4.0$, landing criterion $|q-d|<0.4$, ζ grid `[0,…,1.5,2.0]`, "success = any ζ lands"):

| ζ | displacement | max reach | lands d=4.0? (needs 3.6) |
|---|---|---|---|
| 2.0 (**grid max**) | 1.0881 | **3.5881** | **no — by 0.0119** |
| 2.0105 | 1.1000 | 3.6000 | **yes** |
| 2.1 | 1.2066 | 3.7066 | yes |

**One extra ζ point erases Figure 1a's central claim.** The engineer saw this: `paid-access-experiments` follow-up #1 — *"the crossover is a bracket, not a step exactly at L… this **softens the 'drops to 0 for d>L' wording**. A sharper knife-edge would use a heavier reach coord or measure reach from squeezed-momentum-only."* The draft kept the bracket *and* the unsoftened wording. `paid-access-theory` L29 goes further — "[the wormhole] is the **only** mechanism that beats the causal box" — and L244 lists "Squeeze cannot beat relativistic $C_T$ | **[proven]**". That proof is about $Q_T(z) \subseteq C_T(q_0)$ for the *flow*; it does not cover $C_T(q_0 + \delta q_{\rm squeeze})$. This is a G6-class foundational inversion that survived two reports and one referee pass.
*Triage:* **MUST-FIX.** Two routes, both good:
1. **(Preferred — makes the paper stronger and is on-thesis.)** Restate as a **pricing law**, which is what "paid access" is actually about: *the squeeze buys reach excess $p_0\sinh\zeta/M_0$ at energy cost $e^{2\zeta}$ — i.e. **exponential price per unit of reach** — whereas the wormhole buys **unbounded** Δ at a **fixed** ledger $V(b)-V(a)$.* The dichotomy survives, sharpened, and the certificate stack finally *prices* something.
2. **(Minimum.)** Everywhere: "cannot beat the box" → "cannot beat the box **by more than its own instantaneous displacement $p_0\sinh\zeta/M_0$; at the swept $\zeta\le2$ this collapses reach beyond $d\approx3.6$**," and Fig 1a's annotation gains "at $\zeta\le 2$."
Either way, **run the ζ grid out to 2.5–3.0 at $d\in\{4.0,5.0\}$** (analytic, no training, minutes) so the claim is bounded by data rather than by the grid. See missing-experiment #1.

**MF-C — §4.1's escalatable-memory pillar is measured on a memory §4.3 calls under-trained; C-7's "no constructible contradiction" assertion is false.**
*Location:* §4.1 point 2 + A.2 vs §4.3 + C.4.a + A.4 + the A.5 cross-section note.
*The evidence, entirely from the paper's own tables plus one line of the source:*

| kv32, same MQAR vocab-256 apparatus | §4.1 (`v1-pivot`, A.2: **epochs 400**) | §4.3 (`regime-remap`, A.4: **2000 ep**) | §4.3's verdict on the low regime |
|---|---|---|---|
| storage fidelity | **0.717** (`v1-pivot` L31, "storage-limited") | **1.00 ± 0.00** | 500-ep fidelity 0.76 = "**an under-training artifact**" |
| gate accuracy | **0.286 ± 0.037** | **1.00 ± 0.00** | 500-ep gate acc 0.31 = "**an under-training artifact**" |

§4.1's kv32 cell sits *below* the 500-epoch cell that §4.3 spends its opening paragraph disowning. `v1-pivot` follow-up #4 says so itself: *"kv32 remains partially storage-limited (fidelity 0.717) — capacity study still pending **before choosing the short's difficulty band**."* The short chose the band anyway.

The sharper consequence, also constructible from C.4.a: **at 2000 ep the gate accuracy equals the full-budget accuracy** (1.00/1.00, 0.99/0.99, 0.91/0.91). The savings survive (9.9×, intra-CLU); the *accuracy gain* — "landing **above** always-full", the sentence that makes escalatability interesting and that CM-2 encodes as "extra compute buys accuracy" — is visible **only** in the 400-epoch, fidelity-0.717 regime. A reviewer will write: *"escalation buys accuracy only where the memory is broken; train it properly and the gate merely saves compute."*
*Triage:* **MUST-FIX**, and the cheapest of the three. Required: (i) add `epochs` to the A.5 cross-section note and **delete the sentence** "no two numbers in this paper are drawn from configs a reviewer could read as contradictory" — it is an invitation and it is false; (ii) one sentence in §4.1 point 2 owning it: *"§4.1's models are trained 400 ep; §4.3 shows this band is not converged. At convergence (§4.3, C.4.a) the gate's accuracy **matches** full-budget rather than exceeding it, and the payoff is rationing (9.9× intra-CLU), not accuracy — the allocation headroom is a property of an imperfect memory."* This is a C-9 negative, and stating it is worth more than the +4.7 pts it costs; (iii) contribution 4 must not lead with "landing *above* always-full accuracy" without the epoch scope.
*(Note: this does not touch the 4.81×/1.57×/1.14× numbers, which trace exactly. It touches what they mean.)*

**MF-D — contribution 6 attaches $n=8$ to the full 198-job grid.** *Location:* contribution 6, "On the full grid (198 jobs, $n=8$)." *Evidence:* `regime-remap-2000ep` L69/L90/L251 — corr=0 capacity axis n=8 pooled; stress axes n=5; frontier n=3. The noise-wall numbers quoted two clauses later are n=5. §4.3, App C.4 and Fig 2's caption all get this right; only the contributions bullet is wrong, which is the worst place for it. *Triage:* **MUST-FIX** (one parenthesis: "198 jobs; $n=8$ capacity, $n=5$ stress, $n=3$ frontier").

**MF-E — two internal negative-registry references point at nothing.** *Location:* §4.2 main text ("Appendix D, N24/**N27**"); App D N24 ("with N3, **N21**"); App D scope caveat ("N24/**N27**"). Neither N27 nor N21 is defined in Appendix D or anywhere in the draft. A reviewer following a main-text pointer into a void is a gratuitous credibility hit in a paper whose whole pitch is bookkeeping. *Triage:* **MUST-FIX** (trivial: define or drop). Related, SHOULD: the three new payoff negatives in App D carry no N-numbers while their neighbours do; N23 is listed out of order.

### SHOULD-FIX

**S1 — §3.2(v)'s uniqueness claim is contradicted by the table directly above it.** "The wormhole is therefore the *only* arm that reaches all $d$ **with** a $\det J=1$ + ledger certificate." The §3.2 table lists **Newtonian-squeeze (control): $d<L$ = 1, $d>L$ = 1, $\det J = 1.0$.** It reaches all $d$ with det J = 1. The distinguishing property is the *bounded ledger* (and the safe kinetic mode), not the volume certificate. Add "in the relativistic (safe) mode, and at bounded energy cost". Interacts with MF-B: once the squeeze's displacement is honest, the uniqueness claim must be a *pricing* claim, not an existence claim.

**S2 — The column headed "$\det J$ (measured)" contains two symbolic formulas.** §3.2 table: `plain relaxation` and `dense/throat-$V$` are given as $(1-\gamma)^d$, which in the §3.2 rollout ($\gamma=0$, A.1) evaluates to exactly 1. Either print 1.0 (measured) or rename the column.

**S3 — KE$_0 = 0.72$ is a Newtonian, $M=1$ number quoted for a relativistic, $M_0=4.0$ battery.** §3.2: "(escape-blocked: initial kinetic energy $0.72 < \Delta V_b = 1$)" in a sentence that also states "$c=1$, heavy reach coordinate $M_0=4.0$". `paid-access-theory` L142 defines it as $\tfrac12 p_0^2$ ($M=1$, Newtonian, the Appendix-E toy). At $M_0=4$ Newtonian KE is 0.18; relativistic $\sqrt{p^2c^2+m_0^2c^4}-m_0c^2 = 0.562$. The source for the *battery* (`paid-access-experiments` L88) prudently writes only "KE₀<ΔV_b by design." Escape-blocking holds under every reading, but the *number* is imported across configs — the same C-7 class as MF-C, in the paper's most scrutinized paragraph.

**S4 — "measured by autodiff, not asserted" (×5) is measuring a tautology, and says so in the same sentence.** §3.2(iv) writes: "Its map $(q,p)\mapsto(b,p)$ is differentiable with Jacobian $\mathrm{blockdiag}(0_d,I_d)$, so — measured by forward-mode autodiff, not asserted — $\det J=0$ **exactly**." The reader has just been handed the analytic Jacobian; the autodiff adds nothing, and the insistence invites "they ran a derivative on a constant map and called it a measurement." The genuinely measured half — std$(Q_{\rm out})$ = std$(Q_{\rm in})$ = 0.0803, $\Delta Q = p^\top X\Delta$ to 1.2e-7, reconstruction to 2.2e-8, replicated over dim × seed — is strong and under-sold. Reweight: assert the router's det J = 0 analytically (one clause), *measure* the wormhole's transport.

**S5 — Fig 1a plots landing rates above 1.0 and below 0.0.** The ±0.024/±0.008 offsets put the wormhole visibly above the 1.0 gridline (no tick exists there) and the squeeze/plain-relax below 0. The caption discloses the device, but a probability axis exceeding its own range is the kind of thing that gets screenshotted. Alternatives: a broken/stacked receipt strip under the panel, small horizontal jitter in $d$, or a step-plot with distinct markers and a "3 arms coincide at 1.0" brace. (The revision report flagged this risk itself and asked for a second opinion: **do not ship the offsets.**)

**S6 — Figure 4 appears before Figure 3 in document order.** Fig 4 is in App B.2 (line 320); Fig 3 is in App C.4.b (line 401). Renumber at the pruning pass.

**S7 — An internal editorial note is inside a figure caption.** Fig 3: "*(Candidate for the appendix at the pruning pass.)*" This is the F7 failure class (process leaking into the artifact), one layer in. Strip before any build a human sees.

**S8 — Abstract is 552 words; main text is ~7,100 words against a 4–5pp budget.** This is roughly 2× over before figures and tables. Not a "pruning-pass nicety" — at 4–5pp the current draft cannot be laid out. Concrete: abstract → ~200 words (the payoff sentence + the noise wall + the position; the certificate quadruples belong in §3); contributions 6 → 4 (merge 2+3, merge 5+6); §5's four design rules → three sentences pointing at App F; Fig 3 → App C.4.b (its caption already concedes this).

**S9 — Contribution 6 attaches "Δ+0.02; 9–10× intra-CLU rationing" to "clean/**correlated** cues".** On the correlated axis (C.4.c, ρ=0.9) Δ is +0.16/+0.08 and savings are 5.6×/2.6×. The Δ+0.02 and 9–10× are clean-axis (ρ∈{0,0.5}) numbers. §4.3 point 1 gets this right by separating the clauses; the contribution bullet collapses them. Also: the ρ=0.9 "widening" is a Hopfield-collapse artifact — §4.3 and App D say so, contribution 6 does not.

**S10 — Two orphan bibliography entries.** Geifman & El-Yaniv (2017) and Wales & Doye (1997) appear only in the reference list. §5's "basin-hopping" is uncited; if Wales & Doye is meant there, cite it.

**S11 — Fig 2b is titled "Clean cues: the reversal (kv ≤ 64)" but plots kv96**, where the gate (0.91) sits *below* Hopfield (0.95). Honest, but the title contradicts the third plotted point. Retitle "Clean cues: the reversal, and where it stops."

**S12 — Contribution 1's tag "[proven; theory note + Anonymous 2026]" names one source twice.** §1 establishes that the theory note *is* (Anonymous, 2026). Reads as two independent provenances.

**S13 — §4.1 describes the gate's calibration head as "mapping a relaxation residual to $p_{\rm wrong}$", but A.2 lists the calib feature as `r_margin`.** If the deployed head consumes the readout margin as well as the residual, the description is misleading in a section whose N3 negative is precisely "energy adds nothing over the readout margin" — a reviewer who notices will ask whether the 4.81× gate is in fact a margin gate wearing an energy costume. Verify against `v1-pivot` and state the feature set explicitly. (Not a claim error either way; a transparency gap on the most CM-3-adjacent number in the paper.)

### NICE

- **N-a.** The `v1-certificate-payoff` source still says the router's 16 states "all land on `Q = 1.475`" where its own JSON says 1.5; the draft correctly quotes only the load-bearing form. Curator patch, not a paper item (already raised by `v1-revision-2`).
- **N-b.** §3.2's squeeze falsification still rests on two grid points ($d\in\{4.0,5.0\}$). Post-MF-B this becomes a ζ×d surface; plot it.
- **N-c.** App D's three payoff negatives want N-numbers (N37–N39 per `v1-revision-2`).

---

## The F3 determination (task item 1)

**Is the certificate payoff now *demonstrated*? — Yes, at mechanism level, and the fine print is intact.**

- **Payoff A** is embedded as the headline (Fig 1b, §3.2.1, contribution 3) with all three qualifications in main text: volume-alone-is-not-the-receipt (random shift, det J = 1, out-spread 0.2793 = 3.48×); designed-testbed/oracle-placement scope; and the §4.2 boundary restated *inside* the payoff paragraph. The measured content — injective canonical translation ⇒ constant charge shift ⇒ preserved spread, replicated across dim×seed — is real, non-trivial, and correctly reported.
- **Payoff B carries its caveat, everywhere it appears.** I specifically hunted for the failure the task predicted (selling B as "the certificate prevents blow-up" without the coercive-component clause) and **it does not occur**: §3.1's dedicated paragraph, the abstract, §3.2.1(iii), §5's "only under coercive-component screening", App B.2 item 4 ("the receipt, not the jump… `wormhole_blind` and `no_physics_router` coincide *exactly*"), and App D's Payoff-B bullet all state it. B is correctly demoted out of the headline. The "the receipt *refuses* the jump, it does not make it safe" trade (App B.2 item 5) is stated rather than hidden. **F3's inversion does not re-open one level down.**
- **What the demonstration does *not* survive** is MF-A's framing: the payoff is against a **state-replacing analytic map**, not against "a physics-free learned router," and the paper's own §4.2 router is a counterexample to the sentence as written. Fix the wording and the demonstration stands — indeed stands on firmer ground, because "absorbing jumps erase; canonical translations transport" is a claim about maps, not about ideologies, and no reviewer can invert it.

**Net:** F3 is closed on the merits; it is mis-*worded* in three places, and one of them is the abstract.

---

## Missing-experiment list for the Hub

Not empty this time — but only one item is load-bearing, and it is cheap.

1. **[BLOCKING for MF-B, ~minutes, analytic, no training] Extend the ζ line-search to 2.5–3.0 at $d\in\{4.0,5.0\}$ and report the reach-vs-ζ curve.** Predicted from the paper's own bracket: $d=4.0$ lands at $\zeta\ge2.0105$, $d=5.0$ at $\zeta\ge2.6441$. Two outcomes, both publishable: (a) it lands ⇒ **restate the theorem as the pricing law** (reach excess $p_0\sinh\zeta/M_0$ at energy cost $e^{2\zeta}$; exponential price per unit reach vs the wormhole's fixed ledger) — this is a *better* headline and is what "paid access" was always about; (b) it does not land (governor re-brake / relativistic settling prevents capture) ⇒ report the mechanism that saturates it, and the falsification becomes real rather than grid-shaped. Either way the abstract's "provably cannot beat the box" must go. `paid-access-experiments` follow-up #1 already prescribes the sharper variants (heavier reach coord; squeezed-momentum-only reach).
2. **[SHOULD, cheap, analytic] A learned *additive* transport foil: $q\mapsto q+f_\theta(q)$.** The current no-physics arm is absorbing by construction, so det J = 0 and erasure are definitional (S4). An additive learned router has det J ≈ 1 generically and *is invertible* — does it preserve the coset charge? The random-shift arm suggests no (unmatched channel ⇒ scrambles), which would make the true claim **"only a *matched* channel transports, learned or not"** — a claim the paper half-states in §3.2.1(i) but never tests against a learned map. This converts MF-A from a rewrite into a result.
3. **[WIRING ONLY, no run] Reconcile §4.1 with §4.3 (MF-C).** `regime-remap-2000ep` C.4.a already contains the converged gate-vs-full-budget comparison at kv32/64/96 (gate acc = full acc; savings 9.9×/9.5×/6.2×). Cite it in §4.1 as the convergence-regime counterpart. No new compute.

Nothing else in the draft lacks backing. The noise-wall diagnosis, learned entrance-steering, coercive screening on a trained $V_\theta$, and the App F.6 discriminating experiment remain correctly-labelled future work.

---

## The three sentences a hostile reviewer would quote

1. *"The abstract stakes the paper on a receipt 'a physics-free learned router cannot supply' — yet §4.2's physics-free learned router routes through the authors' own $\det J=1$ wormhole edge; the object with $\det J=0$ is not a router at all but an untrained constant map $q\mapsto b$ that the authors wrote down in order to lose."*
2. *"The headline falsification — 'the squeeze collapses past the causal box' — is a property of the ζ grid, not of the physics: the authors' own bracket gives max reach $L+p_0\sinh\zeta/M_0 = 3.588$ at their largest swept $\zeta=2.0$, against a landing threshold of $3.6$ for $d=4.0$; a single additional grid point at $\zeta=2.01$ deletes Figure 1(a)'s central claim, and their own source report warned that this wording had been 'softened'."*
3. *"§4.1 reports storage fidelity 0.717 and gate accuracy 0.286 at kv32; §4.3 reports 1.00 and 1.00 at kv32 and devotes its opening paragraph to explaining that such low numbers are 'an under-training artifact' — so the escalatable-allocation pillar, the paper's one CLU-conditional asset, is measured on precisely the memory the paper's own regime map disowns, and at convergence the gate's accuracy merely *equals* full budget."*

*(Defusal map: #1 → MF-A rewrite, which strengthens the claim; #2 → MF-B, one ζ-grid run and a pricing-law restatement, which strengthens the frame; #3 → MF-C, one owned sentence and a fixed A.5 note, which converts a hidden contradiction into a C-9 negative. None requires retracting a number. All three are currently *unforced* errors.)*

---

## Proposed handover updates (for the Hub)

- **V1 status: v0.3, F1–F7 CLOSED, verdict BORDERLINE.** Regression from last round's weak-accept is caused by three new/exposed items, all in the abstract+§3 headline: **MF-A** (router conflation — the §4.2 learned router routes through the certified edge; the det J=0 arm is an untrained constant map), **MF-B** (the squeeze *does* beat $C_T$ by $p_0\sinh\zeta/M_0$; the d=4.0 collapse fails by Δζ=0.011 at the grid endpoint), **MF-C** (§4.1's kv32 memory is under-trained by §4.3's own criterion; C-7's "no constructible contradiction" note is false). Plus MF-D (n=8 scope), MF-E (dangling N21/N27).
- **F3 determination: the payoff is demonstrated and its fine print is fully intact** — including every Payoff-B caveat the task file demanded. The failure is wording, not evidence. **CM-3 sweep: clean. CM-8 noise wall: plotted, travels everywhere, cost scoped intra-CLU throughout.**
- **Claims-matrix consequence (CM-12/CM-7, needs a Hub decision):** `paid-access-theory` L29 ("the wormhole is the **only** mechanism that beats the causal box") and L244 ("Squeeze cannot beat relativistic $C_T$ | **[proven]**") are **too strong**: the proof covers $Q_T \subseteq C_T(q_0)$ for the flow, not the squeeze's own instantaneous displacement. **CM-12's approved wording should be amended to the pricing form** before V1 (or the F5 note) ships it: *the squeeze buys additive reach $p_0\sinh\zeta/M_0$ at energy cost $e^{2\zeta}$; the wormhole buys unbounded Δ at a fixed ledger.* This is a cross-short item — the F5 note carries the same theorem.
- **Also for the matrix:** CM-7's approved payoff wording should say **"an absorbing/state-replacing jump"**, not "the router" — otherwise V2/V3 inherit MF-A. And the deprecation of "carries no volume certificate" (per `v1-revision-2`) should land alongside it.
- **Highest-leverage next action:** missing-experiment #1 (ζ-grid extension, minutes, analytic). It either rescues the falsification or upgrades the frame to a pricing law — and it is the difference between a reviewer finding this and us reporting it.
- **Pruning is now blocking, not cosmetic:** main text ≈7,100 words + 552-word abstract against 4–5pp. Docket: abstract → ~200w; contributions 6 → 4; §5 design rules → 3 sentences; Fig 3 → App C.4.b (renumber Figs 3/4); strip the editorial note from Fig 3's caption; drop Fig 1a's vertical offsets.
- **MF-1 unchanged:** `Anonymous (2026)` theory note un-arXiv'd; blocks §2–3 provenance for all three shorts. Head critical path.
