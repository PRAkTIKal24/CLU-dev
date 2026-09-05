# v3-referee-2 — paper-referee report (clean-pass adversarial re-review of V3 short, draft v0.4)

Task + acceptance criterion: re-review the revised V3 short as a composite ML4PS/ICLR reviewer; state explicitly whether **MF-1 is CLOSED**; residual MUST/SHOULD; missing-experiment list; three hostile sentences. Report only — draft untouched.
Status: done

**Read:** `AGENT_PROTOCOL.md` · Positioning Charter (philosophy-synthesis L416–438, **C-1 REVERSED**) · `claims_matrix.md` v1.7 (CM-9 RESOLVED, CM-13) · `critique_register.md` (G1–G7, V*, M*) · `v3-referee.md` (prior: MF-1, SF-1…4, N-1…3) · `v3-interference-extra.md` · `v3-revision-3.md` · `v3-lattice-build.md`, `v3-reversible-o1.md`, `minus-the-physics.md`, `fit-gap-anatomy.md`, `v3-interference-ntk.md` + `pricing_predicts.json` · `scout-goldstone-positioning.md` · `papers/v2-short/draft.md` (cross-short) · `papers/v3-short/{draft.md, CHANGELOG.md, figures/*}`.

---

## VERDICT: **weak-accept.** The near-reject is retired. **MF-1 is CLOSED.**

**Meta-review.** The reframe landed, and it landed properly: §3.2 now leads with the exact structural identity (R ≡ 0 off the coupling graph, 0/4,656 over 72 runs ⇒ boxed **S = deg·R̄_edge** to six decimals), demotes the fitted slope to corroboration, states the chain's degree-ramp residual *preemptively in its own main-text paragraph*, and folds the block-monolith control (my prior N-3) into §3.2(iii) with the capacity confound killed in the cleanest available way (1,185-param tied = worst; 18,960-param untied = exactly zero). "O(1) in N" and "flat in N"-for-a-chain are gone from the text; "flat in N" survives only where CM-9 licenses it (the ring, deg ≡ 2) and in the explicit disavowal sentence. SF-1/2/3 and N-1 are closed. The paper is now more honest than most accepted workshop shorts, and §3.2(iii)+App G contain a concession — *"nothing physics-specific buys the firewall — parameter separation does, and `block-untied` is a strictly better firewall than our lattice"* — that I have rarely seen a paper make about its own headline.

**And that concession is now the paper's biggest unaddressed problem, because the draft never confronts what it costs when combined with the rest of the evidence table.** After §3.2(iii), the headline contribution's mechanism is disclaimed as non-physics. What remains as the physics-specific claim is the *priced channel* (§3.3). §3.3 is measured on **2-unit** trained lattices, 3 seeds; App C concedes that on trained lattices the κ_eff exponents are *inconclusive* and "the exponent authority remains the designed lattice" (i.e. a C-2 *verification*, not evidence). So a paper titled **"Scaling a Conservative Memory Primitive"** establishes its scaling result for a mechanism it explicitly disclaims, and its surviving physics result at **N = 2**. Each half is stated honestly, in different sections, hundreds of lines apart. No reviewer will fail to compose them. That composition is MF-2 below and it is the one thing standing between this draft and a clean accept.

Underneath that: five reviewer-reachable numerical/scope defects, three of them the M4 cross-section-drift class this program's register exists to prevent (a main-text bound falsified by the paper's own appendix; a §3.3 table from which none of its own five residuals can be recomputed; a single-unit result attributed to "the lattice" in the abstract), plus a related-work hole sitting exactly where the nearest architectural cousin lives — and the strings for it already exist in our own scout ledger. All MUST-FIXes are editorial or wiring; **none require a new experiment to reach submission.** One new experiment (pricing at N > 2) would move this to accept.

---

## Is MF-1 closed? **YES — with one residue in the asset, not the text.**

| MF-1 closure criterion (task item 1) | status | evidence |
|---|---|---|
| §3.2 leads structural-first | ✅ | (i) identity → (ii) growth → (iii) controls, in that order (L76–117) |
| Boxed identity `S = deg·R̄_edge` in main text | ✅ | L78; `S/(deg·R̄_edge)=1.000000` every cell |
| `R_far ≡ 0` stated as exact, not fitted | ✅ | L76, "0 nonzero of 4,656 off-graph entries"; App H.1 exact-zero audit |
| Growth = corroboration, not headline | ✅ | (ii) is subordinate; App D.1 + App G record the superseded 2-point reading as a process negative |
| "O(1) in N" absent | ✅ | 0 hits |
| "flat in N" absent for a chain | ✅ | 2 hits: the **ring** (CM-9-approved) and the explicit disavowal (L90) |
| CM-9 approved wording used | ⚠ | "degree-bounded" ×8, but **"coordination-bounded" ×3** — a synonym CM-9 does not sanction (N-2) |
| Chain residual preemptive, main text | ✅ | L90, dedicated paragraph, degree-normalized b = +0.17±0.31, ends "we therefore never describe a chain as 'flat in N'" |
| Figure captions free of old framing | ✅ (captions) / ❌ (**rendered figure**) | Fig-1 caption is clean; the **PNG's own title reads "degree-bounded firewall vs O(N) blow-up"** — MF-3 below |

**Verdict on MF-1: CLOSED.** The residue (MF-3) is a rendering artifact in the headline figure, not a framing relapse.

Task items 2–8, checked one by one:

- **Item 2 (chain residual preemptive):** ✅ main text, own paragraph, *before* the growth table. Not a SHOULD-FIX.
- **Item 3 (block-monolith control → §3.2 main text; capacity decoupled):** ✅ full 5-arm table at L105–111; "**Interference is not a capacity effect**" stated explicitly (L113); `block_tied` reported as co-primary foil with "we quote the naive monolith, which understates our case" (L117). N-3 **closed**.
- **Item 4 (SF-2 = ≤7.5%, not rounded back to 8%):** ✅ in abstract / contribution 3 / §3.3 / Fig-3 caption; 0 occurrences of "8%" as a bound. **But the printed table cannot reproduce the 7.5% — MF-4.**
- **Item 5 (SF-1 / SF-3 / N-1):** ✅ all three. `1.38` gone; S=1 phrased as force perturbation with the storage reading explicitly refused and cross-referenced to App C. Per-figure `[verification]`/`[evidence]`/`[structural]` labels on all 8 captions. Contribution 2 carries in-sentence scope.
- **Item 6 (CM-13 scope clauses ride with 946×):** ✅ **in §3.5** (both clauses) and Fig-5 caption. ❌ **in the abstract and in contribution 5** (p.1), where the 946× ships with "exact only at γ=0" but *not* with "not wired into `train_chlu` / untrained models" — SF-3 below.
- **Item 7 (known-pending):** flagged once, no findings spent → (a) 3 bib slots + (b) `(Anonymous, 2026)` are standing dependencies, correctly marked, nothing fabricated; (c) SF-4 (six contributions) not re-raised as a blocker — but see **Craft** for the measured page-budget overrun the pruning pass needs as input.
- **Item 8 (stale 2-point figure):** ✅ `figures/fig1_scaling_curve.png` is 76,481 B — byte-identical to the 6-point regeneration. No caption, table, or sentence describes the superseded version; App D.1 and App G describe it *as superseded*, correctly. **Clean.** (One craft note on the figure's shading annotation — N-3 below.)

---

## Itemized findings

### MUST-FIX (blocks submission)

---

**MF-2 — The paper disclaims the mechanism behind its headline, then evidences its surviving physics claim at N=2, and never puts the two sentences in the same paragraph.**
*Location:* §3.2(iii) L115 + App G L399 (the concession) vs §3.3 L127 + App C L318 (the physics claim's scope). Title; abstract; §1 thesis.

*The attack (G2 "which component buys what", composed with G3 toy-scale).* The draft says, in main text: "nothing physics-specific buys the firewall — parameter separation does, and `block-untied` is a strictly *better* firewall than our lattice." App G goes further: block-untied achieves it "with no Hamiltonian, no symplectic structure and no coupling graph." The paper's answer is that the physics buys **a priced channel through the firewall (§3.3)**. Fine — but then §3.3 is the load-bearing physics contribution, and:
- it is measured on **2-unit** trained lattices, **3 seeds** (§3.3 header, A.3);
- App C ("Trained-coupling scaling exponent (N16)") concedes that *on trained lattices the κ_eff-scaling exponents are inconclusive* and that "the **exponent authority remains the designed lattice** (−0.499/−0.986)" — i.e. a C-2 **verification** result, not evidence;
- the only *scaling* claim about the priced channel (that a wormhole edge adds exactly its own degree contribution) is in §5 **Horizon**, unmeasured.

So the paper's scaling result belongs to a mechanism it disclaims, and its physics result does not scale in the paper. Both facts are individually disclosed with exemplary honesty; their conjunction is never stated, and a reviewer composes it in ten seconds from p.1 (title) + p.4 (concession) + §3.3's header. This is a *stronger* attack than MF-1 ever was, because unlike MF-1 it cannot be refuted by rerunning the same harness with more seeds — the evidence genuinely is N=2.

*Evidence.* `pricing_predicts.json`: 5 rows, one κ_static sweep, `dim = 2-unit SO(2) pair`, seeds {0,1,2}. `v3-interference-extra` §5 limitation 1 + App C confirm the rest.

*Triage: MUST-FIX (framing, not experiment).* Two moves, both cheap:
1. **Own the composition in §1 and §5.** One paragraph: "The firewall is not our physics; parameter isolation buys it, and buys it better (§3.2(iii)). Our claim is the *priced channel* — communication under one conservative dynamics at a cost that is measured, graph-local and O(κ²) — and we demonstrate its predictive price law on trained 2-unit lattices (§3.3), with the exponent verified on designed lattices at N ≤ 8 (§3.1). Scaling the *price list* to N > 2 is the load-bearing follow-up."
2. **Reconsider the title.** `v3-revision-3` §5 item 5 already sniffed this ("the title should perhaps foreground the *price list*, not the firewall"). It is right. "Scaling a Conservative Memory Primitive: … a Degree-Bounded Interference Firewall …" headlines the contribution the paper spends two pages disclaiming.

The permanent fix is the missing experiment (#1 in the Hub list): κ_eff → sync/recall prediction on trained N ∈ {4,8} lattices. That converts §3.3 into a scaling claim and aligns the paper with its own title. **Not required for submission; required for the ICLR long.**

---

**MF-3 — The headline figure's rendered title says "O(N) blow-up." CM-9 forbids bare asymptotics; the mitigation sits 340 lines away.**
*Location:* `figures/fig1_scaling_curve.png` (title text, viewed); mitigation note at L461, inside the appendix-adjacent asset map.

*The attack.* CM-9 (v1.7): *"Drop 'O(1) in N / O(N)' as bare asymptotics; say degree-bounded vs width-linear."* The draft's prose complies perfectly — 0 hits for "O(N)" in text — but the **most-read object in the paper**, the p.2–3 headline figure, carries `O(N)` in 14-pt type across its title. The paper's disclaimer ("the O(N) refers to the monolith, whose fitted exponent is 1.18 ± 0.17; no arm of this paper's modular lattice is described as O(1) or flat in N anywhere in figure or text") is on L461, after Appendix H, in a "Figures (asset map)" section a reviewer will never reach. A reviewer who reads the figure and the abstract sees `O(N)` asserted in one and `N^{1.18±0.17}` fitted in the other, and asks why the paper reports a *fitted* exponent with a CI in the text and a *bare asymptotic* in the figure. Since 1.18 ± 0.17 has a lower CI edge of 1.01, the O(N) label is barely defensible on its own data.

*Triage: MUST-FIX.* Re-render (a `results-analyst` one-liner, not a paper-writer edit): title → "Interference scaling: degree-bounded firewall vs width-linear monolith". The legend already prints `b = +1.18 ± 0.17`. If the re-render cannot happen before the deadline, move the L461 disambiguation **into the Figure-1 caption**.

*Secondary, same asset:* the shading annotation reads "shaded: the two N values of **the original 2-point claim**." To an external reviewer with no access to our internal history, "the original claim" implies a *prior claim by these authors* — inviting "was a two-point version of this submitted elsewhere?" (M3 salami optics). The shading is good science; relabel neutrally: "shaded: the N range accessible to a two-point measurement." Keep the full story in App G, where it belongs.

---

**MF-4 — §3.3 makes a point of "stating the denominator," then prints a table from which *none* of its five residuals can be recovered — and the headline row recomputes to 7.6%, above the abstract's ≤7.5% bound.**
*Location:* §3.3 table (L131–139); abstract; contribution 3; Fig-3 caption.

*The attack (G5 certificate fine print — my prior SF-2, reopened by the fix).* The draft rounds the predictions to integers. Recomputing residuals from the printed table:

| printed pred | printed meas | recomputed \|Δ\|/pred | **draft states** | true pred (`pricing_predicts.json`) | true residual |
|---|---|---|---|---|---|
| 211 | 195 | **7.58%** | 7.5% | 210.894 | 7.537% |
| 123 | 122 | 0.81% | 0.9% | 123.116 | 0.907% |
| 67 | 68 | 1.49% | **2.1%** | 66.622 | 2.07% |
| 39 | 41 | 5.13% | **4.5%** | 39.242 | 4.48% |
| 22 | 23 | 4.55% | **3.2%** | 22.281 | 3.23% |

Every stated residual is correct against the *true* predictions and wrong against the *printed* ones; the max recomputes to **7.6% > 7.5%**. The paper explicitly invites this recomputation — "which is 8.2% if one instead normalizes by the measured value, **so we state the denominator**" — and then supplies a table that fails the test. This is worse than the original ≤8% it replaced, because the original did not advertise its own auditability.

*Triage: MUST-FIX (trivial).* Print predictions to one decimal: 210.9 / 123.1 / 66.6 / 39.2 / 22.3. Then every residual is reproducible and the ≤7.5% bound holds from the table.

---

**MF-5 — §3.3 omits the recall-horizon columns, the observable the "price list" is named for, whose predictions miss by 47–56%.**
*Location:* §3.3 table + prose (L139); abstract claim (2).

*The attack (selective reporting; reprise of my prior hostile sentence #3, which the revision did not close — it *removed the number* instead of printing it).* From `pricing_predicts.json`:

| κ_eff | n₁/₂ pred | n₁/₂ meas | \|Δ\|/pred |
|---|---|---|---|
| 0.0055 | 2773.4 | **∞** (censored, >11.6k) | — |
| 0.0163 | 943.1 | 1468 | **55.7%** |
| 0.0556 | 273.9 | 257 | 6.2% |
| 0.1602 | 92.9 | 137 | **47.5%** |
| 0.4970 | 27.5 | 42 | **52.8%** |

The draft prints only the two sync columns and writes: "The recall horizon n₁/₂ itself carries the expected first-crossing scatter." That sentence is doing a *lot* of work. A reviewer who opens the source (or simply asks) finds that the paper's own named observable — the **recall horizon**, in "predicts recall-horizon ranking" — is mispredicted pointwise by up to 56%, and that this is the one column the table does not show. This reads as selective reporting even though the *claim* (ranking only) is scoped correctly. C-9 (negatives never dropped) and C-6 (fine print next to the claim) both bite.

*Compounding:* "Spearman(n₁/₂^pred, n₁/₂^meas) = **1.0**" is computed on **5 points, one of which is censored** (`n12_meas: Infinity` — a non-measurement used as the top rank). ρ = 1 on n = 5 has exact p ≈ 0.0083 under the null; no n, no p-value, and no censoring disclosure accompany the claim anywhere.

*Triage: MUST-FIX.* (a) Print the n₁/₂ pred/meas columns alongside sync, with the 47–56% residuals stated; (b) state "Spearman ρ = 1.0 (n = 5, exact p = 0.008); the top-ranked lattice's n₁/₂ is right-censored (> 11.6k steps, reported as a near-latch)". This *costs nothing* — the claim was already ranking-only — and it converts the paper's most-quotable weakness into another instance of the honesty that makes §3.2(iii) work.

---

**MF-6 — Cross-section number drift in §3.6 / Appendix F, including one main-text bound falsified by the paper's own appendix.** (M4 class; this is the exact defect the register logged as N36 for V2.)

*Location:* §3.5 L167 + contribution 5 L30 + Fig-5 caption L173 vs App A.6 L289; §3.6 L179–183 vs App F.1/F.2.

Four separate items, all reviewer-reachable by reading this paper alone:

1. **`≤2×10⁻⁶` (main text ×3) vs `≤2.1×10⁻⁶` (App A.6).** The source (`v3-reversible-o1` L93) says "≤ ~2e-6"; the table says 2.1e-6. The draft dropped the tilde and kept the `≤`. **The paper's appendix falsifies its own main-text bound**, in the flattering direction. Fix: `≈2×10⁻⁶` or `≤2.1×10⁻⁶` everywhere. (CM-13's own wording inherits this — recommend the Hub tighten the matrix row in lockstep.)
2. **"Symplectic volume conservation buys bounded (BIBO) dynamics" (L179), immediately followed by the table entry "BIBO settled-fraction 1.00 (CLU) vs 0.33 (broken-volume) vs **1.00 (twin)**."** The physics-free twin scores identically to the CLU on the paper's own BIBO metric. The claim is true *against broken-volume* and false *against the twin*. It is then apparently contradicted again by App F.1, where the same twin diverges to MSE 196 at 5000 steps. The reconciliation (settle-probe vs long-rollout; different horizons) is real and easy — and appears nowhere. Fix: one clause.
3. **"latch coset drift 0.19 (CLU)" (§3.6) vs "CLU conservative (γ=0) … latch drift 0.778" (App F.2).** Same-named quantity, 4× apart, different testbeds (`minus-the-physics` dim-4 vs `fit-gap-anatomy` circle-vacuum). F.2's header says "circle-vacuum" but never says *not comparable to §3.6's 0.19*. Same for the twin's MSE: **0.0128** (`minus-the-physics`, quoted as "0.013" in §3.6) vs **0.0047** (App F.2). Two twins, two fit gaps (≈15× and ≈44×), one section, no bridge.
4. **"Adding a global γ recovers 92% of the twin fit gap"** (§3.6 L183) sits two sentences after the fit gap is defined as `0.013 vs 0.190`. The 92% is 92% of App F.2's **different** gap (0.202, circle-vacuum). A reader recomputing against the §3.6 numbers gets a different arithmetic.

*Cross-short aggravator (C-8).* **V2's abstract carries a qualifier that V3 drops.** V2: *"a licensed global damping recovers 92% of the **absolute** contraction-forbidden fit gap … **the recovered unit still trails the twin by ≈4.6× in ratio**, but is bounded by construction."* V3 states the 92% with neither "absolute" nor the residual 4.6× ratio (0.0216/0.0047 = 4.60). Per C-8, *a reviewer who reads all our submissions must find one coherent program with consistent scope qualifiers.* Right now V2 is more honest than V3 about the same number. **Fix: import V2's exact wording.**

*Triage: MUST-FIX* (items 1 and 4 + the C-8 lockstep; items 2–3 are one clause each).

---

**MF-7 — The abstract attributes a single-unit (N = 1, dim-4) result to "the lattice," in a paper whose thesis is what happens when you go from one unit to many.**
*Location:* abstract L13, final clause of the price-of-physics sentence; §3.6 header (missing scale qualifier).

*The attack (C-5 + scope-widening).* Abstract: *"…and diverges (MSE 196 at 5000 steps vs **the lattice's** bounded 0.20–0.23 plateau)."* Source: `fit-gap-anatomy` item 2 — `dim/hidden/kin = 4/64/newtonian_learned`, arms `chlu, broken_volume, twin(matched), LSTM(hid 64)`. `chlu` is a **single `CHLU(dim=4)` unit**, not a `CLULattice`. Likewise `minus-the-physics` Part A (dim 4). §3.6's body correctly says "the CLU"; only the abstract says "the lattice." And §3.6's own scope header reads *"Dim 2–4, synthetic, laptop-CPU"* — it gives the dimension but **never the unit count**, in the one paper where the unit count is the subject.

A reviewer who checks will report: "the paper's sixth contribution, 'the price of the physics prior,' is measured on a single unit and described in the abstract as a property of the lattice."

*Triage: MUST-FIX.* Abstract → "the single unit's bounded 0.20–0.23 plateau"; §3.6 header → "single CLU unit (N = 1), dim 2–4, 3 seeds". If §3.6 survives the pruning pass at all, it must say N = 1 in its first line.

---

**MF-8 — Abstract claim (2) — the paper's *only surviving physics-specific claim* — carries no scale qualifier at all.** (C-5.)
*Location:* abstract L13.

Abstract claim (1) carries a full in-sentence scope block (`N≤16, chain/ring/circulant-4, 8–12 seeds, 2-dim units, MLP potentials, κ=0.05, at initialization, laptop-CPU`) — exemplary. Claim **(2)** carries **none**: no N, no seeds, no "trained 2-unit lattices." Claim **(3)** carries the spectral-separability qualifier but no N/seeds. Contribution 3 (§1) does carry "(2-unit trained lattices, 3 seeds)" — so this is an abstract-only omission, and it lands on precisely the claim that MF-2 shows is load-bearing. C-5 requires in-sentence, not section-header-level.

*Triage: MUST-FIX.* Add "(trained 2-unit lattices, 3 seeds)" to abstract claim (2) and "(2-unit lattices, 16× ratio, 5 seeds)" to claim (3).

---

**MF-9 — Related work has a hole exactly where the nearest architectural cousin lives; the strings already exist in our own scout ledger.** (Wiring note, not a missing experiment.)
*Location:* §4; references.

§1 opens: *"A structure-preserving alternative to gated recurrence…"* The paper then:
- **cites nothing for gated recurrence** (no Hochreiter & Schmidhuber);
- **uses LSTM as an arm in App F.1** with no citation;
- **cites nothing for learned-Hamiltonian / symplectic recurrence** — no Chen, Zhang, Arjovsky & Bottou 2020 (*Symplectic Recurrent Neural Networks*, ICLR 2020), which our own `scout-goldstone-positioning` §D calls **"the nearest architectural cousin"**; no Greydanus et al. 2019 (HNN); no Cranmer et al. 2020 (LNN); no Erichson et al. 2021 (Lipschitz RNN); no coRNN/LEM (Rusch & Mishra 2021; Rusch et al. 2022);
- **cites nothing for the conformal-symplectic integrator** (no Hairer–Lubich–Wanner), while §2 asserts `J^⊤ΩJ = (1−γ)Ω` as standard;
- **names "deep-sets" four times** (§3.2(iii), §4, App H, Fig-7) with no Zaheer et al. 2017 citation — `v3-revision-3` §5 item 2 correctly escalated this rather than fabricate, and it is still open.

**These are not missing experiments — every string above exists in `scout-goldstone-positioning.md` §D (lines 62–66) and/or is already cited in `papers/v2-short/draft.md` L173.** V2 cites LSTM/LEM/coRNN/Hairer; V3 does not. A ML4PS/ICLR reviewer reading a paper about *a lattice of learned-Hamiltonian symplectic recurrent units* that cites zero Hamiltonian-neural-network work will score it as unaware of its own field, and will say so first.

*Triage: MUST-FIX (splice; ~30 min).* Add a §4 paragraph "Hamiltonian and symplectic recurrence" (Greydanus 2019; Chen et al. 2020 — *nearest architectural cousin: learned H + symplectic integration; we add the lattice composition law, the interference measurement, and the coupling price list*; Erichson et al. 2021; Rusch & Mishra 2021 / Rusch et al. 2022 as the oscillatory-RNN baselines V2 uses). Add Hochreiter & Schmidhuber for App F.1's LSTM arm. Head decision still outstanding on Zaheer (C-8-legal: published, third-party, not a self-citation — no anonymity cost).

---

**MF-10 — Sixteen internal report slugs are embedded in `draft.tex`, the submission artifact.** (M2 anonymity + dangling refs.)
*Location:* `draft.md` (11 prose/caption instances + 14 `Source: \`slug\`` tags); `draft.tex` (16 matches).

Every §3.x header and figure caption ends with ``Source: `v3-interference-extra` `` / ``Source: `v3-band-selection` `` etc.; §4 says ``(Positioning drawn from `mo-deep-read`.)`` and ``(Specific published anchors … see report.)``. These are (a) unresolvable to any reviewer, (b) a direct advertisement of a private multi-agent research pipeline with non-public artifacts, which is exactly the M2 "a reviewer will review the *program*" surface, and (c) formally dangling references under C-8 (shorts cite only J&P 2026 + the F5 note).

I accept these are deliberate drafting scaffolding (the status blurb says "every quantitative statement traces to a source report"). **Nothing in the draft flags them for removal**, and this is precisely the class of thing that ships by accident. Register it on the C-10 pruning checklist as an explicit strip-list item, or move the provenance tags into the Appendix-A tables where they already live.

*Triage: MUST-FIX before submission (not before Hub review).*

---

### SHOULD-FIX

**SF-1 — Contribution 5 and the abstract ship the 946× without the "not wired into `train_chlu` / untrained models" clause.** (CM-13, C-6.) §3.5 and Fig-5's caption carry **both** required scope clauses — exemplary. But p.1 (abstract: *"As a systems corollary … 946× lower peak memory … exact only along the conservative direction"*; contribution 5: *"946× at T=1024,N=2; CPU/small-D; exact only at γ=0"*) carries only the γ=0 clause. A reviewer reading p.1 will believe this is a trainer result. C-6: *"the fine print must never be invertible into the review."* Add five words: "…on untrained models; not yet in the shipped trainer."

**SF-2 — "five-plus orders below the SGD gradient-noise floor, i.e. training-indistinguishable" is an assertion, not a measurement — and the paper elsewhere concedes it has never trained with these gradients.** (§3.5 L167; Fig-5 caption.) No SGD gradient-noise floor is measured anywhere in `v3-reversible-o1` or the draft. §3.5 itself states the result is "validated on untrained models with a final-state loss … not yet wired into the shipped `train_chlu`." A reviewer: *"you claim training-indistinguishable gradients for a method you have never used to train."* Soften to "≈2×10⁻⁶ relative in float32 — five-plus orders below typical SGD gradient noise; whether this is training-indistinguishable is untested (§5)," or measure it.

**SF-3 — The squeeze certificate in §3.6/App F.2 ships without CM-12's mandatory scope clause.** §3.6 L183: *"A squeeze S^(M) rents reach with an exact certificate (det = 1 to 3×10⁻⁷, energy injection ≤ e^{2|ζ|} …)."* CM-12's approved form: *"the squeeze e^{2ζ} bound is a **matched-quadratic-H certificate**"* (and the whole pillar is oracle-placement-scoped). Neither qualifier appears in V3. C-6 requires the altitude next to the claim; C-8 requires the same scope in every short that states it. Add "(a matched-quadratic-H certificate; oracle placement)".

**SF-4 — "at the 16× ratio it *inverts* the ordering" is asserted for "≥30", but the paper's own Appendix E shows no inversion at 30.** (§3.4(iii) L161 vs App E L352, L360.) App E, mult = 30, ratio 16×: align **+0.33** (N=2), **+0.20** (N=4) — degraded, not inverted. Inversion appears only at 100× (−1.00 at N=2; −0.20 at N=4). The main-text sentence "A high multiplier (≥30) is harmful, with a ratio-dependent failure: at the 16× ratio it *inverts* the ordering (alignment −1.00 at 100×)" is defensible only by reading the parenthetical as the scope of the verb. A reviewer flipping to App E finds positive alignment at 30. Fix: "≥30 degrades the ordering (align +1.00 → +0.20…+0.33 at 16×) and 100× inverts it (−1.00 at N=2)". *(CM-5's own wording is loose in the same way — recommend a matrix lockstep.)*

**SF-5 — "explodes the fit (MSE 35×)" hides an s.d. larger than the mean.** (§3.4(iii) L161 vs App E L357 and App B.3: `35.27 ± 47.26`, 3 seeds.) A quantity whose sample s.d. is 1.34× its mean is a single-seed blow-up, not a characterized effect. The main text quotes the mean bare. C-7/M4: state "MSE 35.3 ± 47.3 (3 seeds; seed-dominated)". *(CM-5 inherits: "global-mass runaway (MSE 35×)".)* Also note the runaway is **N-dependent, not purely ratio-dependent**: at N=2, 4×, 100× the ordering holds (+1.00) and MSE is 0.373. The "ratio-dependent failure" framing is incomplete.

**SF-6 — "the modular residual leak is … *tighter* than the theoretical bound" is a dangling comparative.** (§3.2 L101.) Which theoretical bound? The only bound in the paper is the coordination bound `S ≤ 2R̄_edge` — which, for a chain with mean degree < 2, is an algebraic consequence of the identity `S = deg·R̄_edge`, not an independent constraint the data could violate. The sentence then pivots to the κ² sweep, which is about the *leak's coupling dependence*, not about any bound. Either name the bound or cut the clause. *(Same nit in the Fig-1 caption: "the modular lattice **sits at** the coordination bound S ≤ 2R̄_edge (dotted)" presents a tautology as a satisfied constraint.)*

**SF-7 — "sub-linear-to-mild cost growth to D=16" is contradicted by the numbers in the same sentence.** (§3.1 L66.) 233k → 125k → 35k steps/s at N = 2/4/8 (D = 4/8/16): per-step wall grows **6.7× for a 4× increase in D** — *super*-linear in D. (Per *unit*-step it is 1.66×, i.e. sub-linear; that is presumably the intended reading, and it is the charitable one.) The source (`v3-lattice-build` item 3) says "Sub-linear-ish", so the draft inherits the looseness. State it precisely: "per-unit step cost grows 1.7× from N=2 to N=8 while aggregate throughput falls 6.7×."

**SF-8 — Abstract enumerates two of the paper's three evidence grades.** (Abstract final sentence: "designed-testbed numbers are labeled verification, learned-system numbers evidence") vs §1 L33, which defines **three** (adding *structural* for integrator/autodiff-graph results). §3.5 — announced in the abstract — is a `[structural]` result. C-2 reporting discipline should be stated completely where it is first stated. One clause.

---

### NICE

**N-1 — Terminology drift: "coordination-bounded" (×3) vs "degree-bounded" (×8) for one concept.** CM-9's sanctioned phrase is **"degree-bounded vs width-linear."** "Coordination-bounded" appears in the abstract, in the §3.2(ii) sub-heading, and in §4. They are synonyms (coordination number = degree) and the paper defines neither as the other. Pick one — CM-9 says which — and keep "coordination number" for the *quantity*, "degree-bounded" for the *claim*.

**N-2 — Fig-1's y-axis label reads `S = Σ_{A≠B} R_{B←A}`, but the reported S is `mean_B` of that sum** (as §3.2 correctly defines). A reviewer reproducing from the figure will be off by a factor of N. One-character caption fix or an axis re-render.

**N-3 — Fig-1's shading annotation "the original 2-point claim."** See MF-3 secondary. Excellent instinct, wrong words: the honesty belongs in App G (where it is, superbly, as a process negative); the headline figure should not imply the authors previously claimed something else. Relabel.

**N-4 — The paper *can* parameter-match and does so in §3.6 (twin, ±0.05%), but not for its headline foil.** §5 concedes "19,112 modular vs 2,177 monolith at N=16 … a width-matched monolith sweep is not run" and defends it with the block-tied/block-untied capacity decoupling — which is a genuinely strong defense. But a reviewer will notice that §3.6 matches parameters to four significant figures (chlu 4549 / broken_volume 4557 / twin 4551) and ask why the headline comparison could not. Pre-empt it in one clause in §3.2's foil description: "we match *width and family* rather than parameter count, because the capacity confound is settled directly by the block-tied/untied pair (§3.2(iii)) — a param-matched sweep is a follow-up (§5)."

**N-5 — App A.6's XLA-scratch caveat remains exemplary** (`temp_size_in_bytes` = "compiler scratch estimate … a proxy for tape size, not runtime peak-RSS"), and Fig-5's caption now repeats it. It pre-empts the "946× is a compiler artifact" attack. No action. Keep this pattern.

**N-6 — App C's struck-through-and-retained block-monolith entry** (`~~Block-structured monolith (cat. iii)~~ — MEASURED`) is a small masterpiece of C-9 discipline: it shows a reviewer the open item and its discharge in one line. No action.

---

### Known-pending (flagged once per task item 7; no findings spent)

- One bibliography item unresolved (three marked `[·]` slots: Mo 2026; Di Bernardo/Keller; checkpointing-O(√T) + RevNet/momentum-net), awaiting the **Jul-11 venue scout**. The draft marks them explicitly and fabricates nothing — correct behavior. *(Note: MF-9's strings are a **different**, already-available set; do not let them wait on the scout.)*
- The theory note is `(Anonymous, 2026)` pending the Head's title/author call and the live arXiv id.
- **SF-4 (six contributions over budget) is deferred to the C-10 pruning pass** and is **not** re-raised as a blocker. Measured inputs for that pass, below.

---

## Craft / page budget (input for the C-10 pruning pass — not a blocker)

| object | measured | typical 4–5 pp workshop short |
|---|---|---|
| main text, abstract → §5 | **6,414 words** | 2,500–3,500 |
| abstract | **513 words** | 150–250 |
| appendices | 4,427 words | — |
| main-text figures | 5 | 2–3 |
| main-text tables | 6 | 1–2 |
| PDF (appendix-maximal, per C-10) | 19 pp | — |

The main text is **≈2× over** a 4–5 pp budget before LaTeX; the abstract is **≈2–3× over** and is a single unbroken 513-word paragraph in which the three numbered claims are individually longer than most workshop abstracts. Contribution clarity on p.1 is otherwise good: the threefold thesis is stated in §1 ¶2 and the six contributions are individually graded. **Recommendation to the pruning pass:** cut the abstract to three sentences per claim; demote §3.5 and §3.6 to appendix-forward summaries (as SF-4 of the prior report proposed) — noting that MF-7 makes §3.6's N=1 scope a liability in the main text anyway, and MF-2 argues §3.3 should *gain* the space they lose.

Headline figure quality: the 6-point curve is a genuinely good figure — six sizes, 12 seeds, s.e.m. bars, both fitted slopes in the legend, the S=1 reference line, the coordination bound, and the honest shading of the old window. Modulo MF-3's title and N-2/N-3's labels, it is the paper's strongest asset and it now *supports* the claim rather than falsifying it. That is the whole difference between v0.3 and v0.4.

---

## Charter / claims-matrix compliance (spot audit)

| rule | status |
|---|---|
| **C-1** (no audit confession) | ✅ absent; correctly noted in the status blurb |
| **C-2** (verification / evidence / structural) | ✅ per-section **and** per-figure grades (SF-3 of prior report closed). ⚠ abstract lists 2 of 3 grades (SF-8) |
| **C-5** (in-sentence scale qualifiers) | ⚠ abstract claim (1) exemplary; **claims (2) and (3) carry none (MF-8)**; §3.6 gives dim but not unit count (MF-7). Grep for scope-free plurals ("CLUs provide", "the lattice scales"): **0 hits** — good |
| **C-6** (certificate altitude) | ⚠ §3.3's denominator ✅ but its table breaks it (MF-4); n₁/₂ error suppressed (MF-5); §3.5 ✅ in-section but ❌ on p.1 (SF-1); squeeze certificate missing CM-12's matched-quadratic-H clause (SF-3) |
| **C-7** (flag provenance) | ✅ A.1–A.6 + the new **A.2b** (commit `37dc664`, JAX 0.9.0, seeds, N grid, probe path, measurement point, statistics, **bit-exact metric-identity anchor**). A.2 carries an explicit supersession row. Best provenance block in the program. ⚠ but see MF-6 for numbers that drift *between* sections despite it |
| **C-8** (hermetic; one coherent program) | ⚠ no cross-short citation ✅; **but V2 states a qualifier on the shared 92% number that V3 drops (MF-6, cross-short)**; and 16 internal slugs in `draft.tex` (MF-10) |
| **C-9** (negatives never dropped) | ✅ App G, including "**the firewall is not a physics result**" and the superseded 2-point reading as a process negative. ⚠ except the 47–56% n₁/₂ error, which is a negative currently absent from the paper (MF-5) |
| **C-10** (appendix maximalism) | ✅ nothing pruned; App H added |
| **CM-3** (forbidden: energy-as-confidence) | ✅ absent in every form, hedged or implied |
| **CM-9** (RESOLVED wording) | ✅ substance; ⚠ "coordination-bounded" synonym (N-1); ❌ **"O(N)" in the headline figure title (MF-3)** |
| **CM-10** | draft says ≤7.5% (tighter than the matrix's ≤8%) — **matrix needs a lockstep tightening**; MF-4/MF-5 are the draft-side defects |
| **CM-13** | ✅ both scope clauses in §3.5; ❌ one clause missing on p.1 (SF-1); `≤2e-6` vs `≤2.1e-6` (MF-6.1) |
| **CM-1** | ✅ crossing, boundedness-not-lowest-plateau, reach secondary/aggregate-only, γ_φ-wrong-tool all present. ⚠ MF-6.4 + MF-7 |
| **G1** (testbed built to satisfy the theory) | **well defended.** §3.1 is explicitly labeled verification and explicitly disclaimed as "not a discovery"; §3.2–§3.4 are on learned MLP potentials. The one place G1 still bites is MF-2: the *exponent authority* for the price law is the designed lattice |
| **G3** (toy scale) | mostly owned (§5's Scope ¶ is honest and complete). Residual: MF-2's N=2 |
| **G5** (certificate fine print) | MF-4, MF-5, SF-1, SF-3 |
| **G6** (foundational falsifications) | ✅ no legacy number is load-bearing; C-1 respected |
| **M2/M3** (de-anon / salami) | MF-10 (slugs) + MF-3 secondary ("the original 2-point claim") |

**M4 cross-section sweep:** **four contradictions found** (MF-6.1–6.4), one of them across shorts (V2 vs V3 on the 92%). The prior pass found none; the v0.4 revision fixed two pre-existing ones (the "parameter-matched monolith" error and the "persists through training" scope) and introduced none — these four were latent in §3.5/§3.6, which the revision did not touch. **The lesson for the Hub: the M4 sweep must cover the sections a revision did *not* edit.**

---

## Missing-experiment list (for the Hub)

**Genuinely missing (new experiments):**
1. **[HIGHEST — retires MF-2's permanent half] Price-list prediction at N > 2.** Read κ_eff blind from learned couplings on trained N ∈ {4,8} lattices; register sync/recall predictions; measure. Converts the paper's one surviving physics-specific claim from N=2 into a *scaling* claim, aligning it with the title. Extension of register **P13/V3.3**. Same harness as `v3-interference-ntk` item 2; cheap.
2. **Through-training R at N ∈ {4,8,16}.** The analyst's own #1 recommendation and App C's live caveat; the exact zeros are training-invariant, the 6.5e-5 magnitude is not. Blocks nothing at workshop scale; blocks the ICLR claim.
3. **Parameter-matched monolith sweep** (widen hidden until params match). App C admits it is unrun; §3.6 proves we can match to ±0.05% when we want to (N-4).
4. **κ-sweep at N=16** → closes `S = deg·c·κ²` into a fully closed-form price for the firewall (currently N=4 only). The single cheapest upgrade to the paper's structural story.
5. **Dynamical interference half-life.** Still the only thing blocking a *storage* reading of the S=1 crossing; the draft correctly refuses the reading in the meantime.
6. **Trajectory-mediated interference channel** (banding changes which loci drive updates). App C; unmeasured.

**Exists in outputs, not cited — MUST-FIX wiring, not experiments:**
7. **Symplectic-RNN / HNN / oscillatory-RNN / Deep-Sets bibliography.** All strings live in `scout-goldstone-positioning.md` §D (L62–66) and/or `papers/v2-short/draft.md` L173. See MF-9. **Do not let this wait on the Jul-11 scout** — it is a different, already-verified set.
8. **The n₁/₂ pred/meas columns and the Spearman's n/p/censoring** already exist in `v3-interference-ntk/pricing_predicts.json`. See MF-5. Zero compute.

**In-flight / known-pending (no action):** three bib strings (Jul-11 scout); `(Anonymous, 2026)` arXiv id; reversible-BPTT wired into `train_chlu` + GPU/HBM measurement + checkpointing-O(√T) Pareto baseline (blocking for the ICLR systems claim, not for the workshop short).

---

## The three sentences a hostile reviewer would quote

1. *"The authors state, in their own main text, that 'nothing physics-specific buys the firewall — parameter separation does,' that a parameter-separated potential with 'no Hamiltonian, no symplectic structure and no coupling graph' is a **strictly better** firewall than theirs, and that what the physics buys instead is a 'priced channel' — whose predictive price law is demonstrated on **two-unit** lattices with three seeds, and whose scaling exponent, they concede in Appendix C, is 'inconclusive' on trained lattices; so a paper titled 'Scaling a Conservative Memory Primitive' establishes its scaling result for a mechanism it disclaims and its physics result at N = 2."*

2. *"Section 3.3 makes a virtue of 'stating the denominator,' yet not one of its five residuals can be recovered from the table it prints — the headline row (211 predicted, 195 measured) recomputes to 7.6%, above the abstract's ≤7.5% bound — and the table silently omits the recall-horizon columns, the observable the 'price list' is named for, whose predictions miss by 47–56% and whose 'exact' Spearman of 1.0 is computed over five points, the top-ranked of which is a censored non-measurement."*

3. *"The abstract's systems number, 'gradients match to ≤2×10⁻⁶,' is falsified by the authors' own Appendix A.6 (≤2.1×10⁻⁶); the accompanying 946× memory saving is a compiler scratch estimate on untrained models at N=2, D≤16, on a laptop CPU, never wired into the trainer they ship; and the '≈700-step loan' the abstract attributes to 'the lattice' is a single-unit, dim-4 measurement in a paper about what happens when you scale from one unit to many."*

---

## Bottom line for the Hub

- **MF-1: CLOSED.** The reframe is correct, complete, and structural-first. The near-reject is retired. Task items 2, 3, 5, 8 are all satisfied; item 4 is satisfied in wording but broken by table rounding (MF-4); item 6 is satisfied in §3.5 but not on p.1 (SF-1); item 7's known-pendings are correctly handled.
- **Is V3 submission-ready modulo pruning + arXiv id + the pending bib string?** **Not quite — but nothing outstanding requires compute.** Nine MUST-FIXes: one framing paragraph + a title reconsideration (MF-2), one figure re-render (MF-3), two table fixes that cost nothing and *strengthen* the paper (MF-4, MF-5), one number-consistency sweep of the two sections the last revision did not touch (MF-6, incl. a C-8 lockstep with V2), two abstract scope fixes (MF-7, MF-8), one bibliography splice from an existing scout ledger (MF-9), one strip-list item (MF-10). Estimate: one paper-writer pass + one analyst re-render.
- **Then it is a clear weak-accept, and a likely accept** — because the thing that makes this paper unusual is that its most quotable sentence is a concession, and concessions of that quality, *when composed for the reviewer rather than left to be discovered*, are what get honest workshop papers in.

## Proposed handover updates (for the Hub)

- **V3 v0.4: MF-1 CLOSED; near-reject retired.** Nine MUST-FIXes (all editorial/wiring; zero compute). New top attack = **MF-2**: the §3.2(iii) concession + §3.3's N=2 scope compose into "your headline isn't your physics and your physics doesn't scale." Needs a framing paragraph now, and experiment #1 (pricing at N ∈ {4,8}) for the ICLR long.
- **Matrix lockstep required (3 rows):** (a) **CM-10** "sync pointwise ≤8%" → "≤7.5% relative to the registered prediction (max residual on the weakest-coupling lattice; 8.2% if normalized by measurement — state the denominator); n₁/₂ pointwise 47–56%, ranking-only (ρ=1.0, n=5, p=0.008, top rank censored)". (b) **CM-13** "≤2e-6 rel (float32)" → "≤2.1e-6". (c) **CM-5** "≥30 is harmful … inverts ordering at 16×" → "≥30 degrades (align +0.20…+0.33 at 16×); 100× inverts (−1.00 at N=2); the 4× runaway is MSE 35.3 ± 47.3, seed-dominated, N=4 only".
- **CM-9 wording:** the draft uses "coordination-bounded" alongside the sanctioned "degree-bounded." Either bless the synonym in the matrix or purge it from the draft. Recommend purge (one word, three places).
- **C-8 cross-short defect found:** **V2's abstract is more honest than V3's §3.6 about the same 92% number** (V2 says "92% of the **absolute** gap … still trails the twin by ≈4.6× in ratio"; V3 says neither). Import V2's wording into V3. Check whether any other short states the 92% bare.
- **Process lesson for the next referee/revision cycle (worth encoding):** `v3-revision-3` swept §3.2 (which it rewrote) for M4 contradictions and found/fixed two; the four I found (MF-6) are all in §3.5/§3.6 — **sections the revision did not touch**. *The M4 sweep must cover the whole draft, not the diff.*
- **Two Hub task candidates, both cheap, both retiring a hostile sentence:** (1) pricing at N ∈ {4,8} (retires hostile sentence #1's second clause and MF-2's permanent half); (2) a 30-minute bibliography splice from `scout-goldstone-positioning` §D (retires MF-9 — **independent of the Jul-11 scout**, which owns a different three strings).
- **Pruning-pass inputs (measured):** main text 6,414 words (≈2× over), abstract 513 words (≈2–3× over, single paragraph), 5 main-text figures, 6 main-text tables, PDF 19 pp. Add "strip internal report slugs from `draft.tex`" (MF-10) to the pruning checklist as an explicit line item.
