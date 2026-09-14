# v1-referee — paper-referee report (adversarial review of V1 short)

Task + acceptance criterion: simulate an ML4PS/NeurReps composite reviewer for the V1 *position/theory* short (`.claude/papers/v1-short/draft.md`); judge whether the "certified mechanisms, not benchmark-topping" position is honestly earned and internally consistent; MUST/SHOULD/NICE triage; three hostile quotes; ≈empty missing-experiment list.
Status: done. Report only — draft not edited.

---

## VERDICT: **weak-accept** (borderline→weak-accept after SHOULD-FIX craft items)

**Meta-review.** This is, on the evidence, one of the most honestly-scoped short papers I have refereed: every quantitative claim I spot-audited (§3 reach battery, §4.1 gate AUROC/4.8×/LTT, §4.2 router, §4.3 full 198-job regime map incl. the noise wall, §5 MH-kernel) reproduces its source report to the digit, the scale qualifiers are attached in-sentence per C-5, and the certificate fine print (e^{2ζ}=matched-quadratic-H, frozen-gate det J, LTT exchangeability+ECE≈0.10) sits next to its claim per C-6. For a *position/theory* workshop — where principled framing + exact theorems + honest boundary-mapping are the currency and benchmark-topping is explicitly not required — this clears the bar. **But the position is carried almost entirely by one contribution (§3, the certificate stack), which is a machine-precision *verification* on a designed analytic testbed with oracle channel placement and a hand-picked 16× mass band; and every result on a *trained* memory is a boundary or a negative — the gate mechanism is memory-agnostic (§4.1), the energy-gated edge loses to a 449-param router in FLOPs and accuracy (§4.2), and Hopfield keeps both the cost and the noise-robustness advantage (§4.3).** The paper is candid about all of this, loudly, and that candor is its strongest scientific asset at these venues — but it also means a hostile reviewer's core complaint stands: *the physics buys a demonstrated, CLU-specific advantage only where the authors control the testbed.* The paper never exhibits a single learned-memory result in which the certificate stack buys a measurable win a cheaper black box couldn't. That caps it at weak-accept: honest, internally consistent, well-positioned, but thin on demonstrated payoff. Three fixable craft failures (the headline figure occludes its own hero curve; the foregrounded noise wall appears in no figure; the "beats the router" claim is definitional, not demonstrated) currently make the paper read weaker than its own honesty warrants. Fix those and it is a clean weak-accept; leave them and a hostile reviewer will read the visual story as self-refuting.

**On the intentional posture (per task):** the verification-grade lead with oracle placement is a legitimate position-paper move and I do **not** fault it for "not beating baselines." The learned-entrance-steering-is-future-work honesty (§5 first horizon bullet, §3 header, contribution-2 tag) is loud enough that the headline is not a bait-and-switch. The "Hopfield owns the noise axis" stance is Head-confirmed and is stress-tested below, not flagged as a decision.

---

## Charter / claims-matrix compliance (item by item)

- **C-1 (no audit confession):** PASS. No defensive audit paragraph. §3.3 ("why a prior null does not bear") and §4.3 ("the 26/26 map was an under-training artifact") discuss the *program's own* evolving results as C-9 negatives, not the legacy paper's mechanism-numbers. Compliant — but see hostile-quote #3: a reviewer *can* weaponize the §4.3 "under-training artifact" framing.
- **C-5 (scale qualifiers):** PASS. Grepped for scope-free plurals; every generalizing claim carries dim/N/vocab/kv/seeds/laptop in-sentence. §5 scope paragraph is explicit. No bare "CLUs provide."
- **C-6 (certificate altitude):** MOSTLY PASS, **one placement gap** (SHOULD-FIX F5 below): the **BIBO-needs-coercive-exit** caveat — a wormhole exit into a non-coercive region can escape to infinity (`paid-access-theory` §7 issue 7) — lives only in Appendix B, not in §3.1 main text next to the wormhole claim. Task item 5 explicitly requires it next to its claim. Every other fine-print item (e^{2ζ}, frozen-gate det J=2.05, LTT exchangeability+ECE≈0.10) is correctly inline.
- **C-8 (hermetic citations):** PASS on structure (only J&P 2026 + theory note + published lit; no cross-short cites), but see MUST-FIX-1: the theory note is "Anonymous (2026)" and not yet public, while §3's load-bearing propositions (Prop-A2/A6/12) live in it.
- **C-9 (negatives):** PASS. Appendix D carries N1/N2/N2b/N3/N23/N24/N30/N31; the noise wall is foregrounded in main §4.3 point 3.
- **C-10 (appendix maximalism):** PASS. Appendices A–E full; placeholder title/authors.
- **CM-2** (memory-agnostic gate / escalatable asset): approved wording used verbatim, §4.1. PASS.
- **CM-3 (FORBIDDEN energy>margin):** PASS. Explicitly disclaimed in §4.1 (N3), §4.2, §5. I hunted for hedged/implied forms per task item 3: §3's "energy ledger" is a bookkeeping receipt (reach/escape axis), not a confidence signal; §5's EBT contrast is about *structural accounting*, not signal quality. No slide found.
- **CM-7** (router beats gate): stated plainly, §4.2 + C.2. PASS.
- **CM-8** (settled regime map, noise wall travels): final wording, §4.3 + C.4. PASS.
- **CM-12** (paid access, oracle-placement scope, learned steering future work): PASS, §3 header + §5.
- **CM-14** (squeeze-MH kernel, theory-complete toy EBMs, no trained-model runs): PASS — §5 carries every fine-print clause (governor destroys invariant, coset erosion receipt, FDT σ*, "no runs on trained CLU checkpoints claimed").

**Claim-evidence audit result:** zero mismatched/unbacked/scope-widened numbers found. Full trace table below.

| draft claim | source | verdict |
|---|---|---|
| §3.1 injection (0.25,1.13,1.65)…(2.0,27.5,54.6); det S=1.000±4e-6 | paid-access-experiments L56 | exact ✓ |
| §3.1 forbidden gate det J=2.05 | paid-access-experiments L57 | exact ✓ |
| §3.1 latch ΔQ=0.2500=pᵀXΔ; squeeze 1.2e-7; random {0.035,0.157,-0.143,0.302,-0.144} | paid-access-experiments L48–50 | exact ✓ |
| §3.2 reach table; edge at d≈3.2; bracket [L,L+p₀sinhζ/M₀] | paid-access-experiments §7.1 L30–42 | exact ✓ |
| §3.2 KE₀=0.72<ΔV_b=1 (escape-blocked, p₀=1.2, relativistic KE) | paid-access-theory L59,144 | ✓ |
| §4.1 raw 0.431±0.038→calib 0.869±0.015; 4.81±0.44× @629±60 vs 0.847 @3000; LTT 30/30, cov 0.647±0.063 | v1-pivot L33–34,39,46 | exact ✓ |
| §4.1 Hopfield transfer 0.18→0.88 (0.182→0.878) ≈ CLU 0.43→0.87 | minus-the-physics L58,67 | exact ✓ |
| §4.2 router 1.000/0.948 @8.81e7 vs gated 0.887/0.715 @1.18e8; chain 1.76→2.94e8, distant 0.41→0.28; 449 params | v1-router-baseline L34–44,75 | exact ✓ |
| §4.3 noise wall gate 0.36 vs Hop 0.71 @σ=0.6/kv32; 0/6 noise, 6/9 corr, 6/15 overall | regime-remap-2000ep L187,197–198 | exact ✓ |
| §4.3 kv96 reverses @4000 (0.975 vs 0.947); kv128 ties (+0.004); kv32 over-trains 1.00→0.89 | regime-remap-2000ep L133–139 | exact ✓ |
| §5 L1 0.0995→0.0065; T_eff 1.0→0.61; D=1.29e-3 vs 1.25e-3; N_erode=(Δ/s)² | thread9-mh-kernel Checks 1–4 | exact ✓ |

---

## Itemized findings

### MUST-FIX (blocks submission)

**MF-1 — The load-bearing theory note is not public; §3's proofs are nowhere the reviewer can reach.** *Location:* contributions 1–2 tagged "[proven; theory note + Anonymous 2026]"; §2–3 cite Prop-A2/A6/12; References list "Anonymous (2026) [placeholder]." *Attack:* the entire certificate stack — the paper's one physics-specific win — rests on propositions the reviewer cannot verify; Appendix E gives *numerical* checks but not proofs. Per C-8 the note is citable "once live," and the handover confirms it is still an un-arXiv'd placeholder. *Triage:* MUST-FIX **as a submission dependency** — either the theory note is arXiv'd by submission, or §3's three load-bearing propositions get a proof sketch in an appendix. Not the referee's to fix; flag to Hub/Head as critical-path. (Same "(Anonymous, 2026)" placeholder blocks all three shorts per handover.)

### SHOULD-FIX

**F1 — The headline figure (Fig 1) occludes its own hero curve.** *Location:* `paid_access_reach.png`, contribution 2. *Attack:* the wormhole (green) is drawn *underneath* the no-physics router (purple) and Newtonian control (red) — all three are flat at landing-rate 1.0 and perfectly overlap, so the wormhole line is **invisible**. The paper's central thesis — "the wormhole reaches *with* a det J=1 certificate, the router reaches *without* one" — is not representable in a landing-rate plot because both land at 1.0; the distinction is purely textual (the det-J column of Table 2). A reviewer glancing at Fig 1 sees three identical flat lines plus two that drop, and reads "wormhole ≡ router ≡ Newtonian" — the visual *opposite* of the thesis. *Evidence:* rendered figure inspected; green/red not visible under purple. *Triage:* SHOULD-FIX (strong). Add a second panel or annotation carrying the *receipt* that distinguishes the arms (det J / ledger / latch-shift per arm), or split the overlapping lines with markers/offset. The headline figure must headline the contribution.

**F2 — The foregrounded noise wall appears in no figure.** *Location:* §4.3 point 3 ("THE NOISE WALL — the dominant negative, foregrounded"); Figs 2 & 3 both show only the clean corr=0 axis. *Attack:* the paper's *text* leads with the noise wall as the sharpest negative, but both figures show the more favorable clean axis; the noise wall lives only in Appendix C.4.c as a table. A reviewer can fairly say "you foreground your worst result in prose and bury it in an appendix table while two figures show your best axis." *Triage:* SHOULD-FIX. Add a noise-σ panel (gate vs Hopfield across σ∈{0,0.3,0.6}) to Fig 2. This also converts an honesty liability into an honesty asset (the negative becomes visible, matching the loud text).

**F3 — "Beats the router" is definitional, not demonstrated (G1/reviewer crux).** *Location:* §3.2 / contribution 2: the wormhole "beats a no-physics router (which reaches everything but carries no volume certificate)." *Attack:* the same 449-param no-physics router *beats* the energy-gated edge in FLOPs AND accuracy in §4.2. The only sense in which §3's wormhole "beats" it is that the wormhole carries a det J=1 label the router lacks — a label of no *demonstrated* downstream consequence anywhere in the paper. The certificate's value (BIBO/latch preservation) is asserted (§5, App B) but never shown as a failure the router actually incurs. The latch-transport data (§3.1: random-shift baseline erases Q) is the right *shape* of demonstration, but the *router* arm is never shown eroding the latch or landing in a non-coercive region. *Triage:* SHOULD-FIX (strong). Either (a) add one arm where the certificate-free router demonstrably violates a guarantee the wormhole preserves (router jump → non-coercive exit → BIBO blow-up, or router → latch erased vs wormhole → latch transported by exact pᵀXΔ), citing existing data if it exists, or (b) soften §3.2 to "reaches with a receipt the router cannot supply" and state explicitly that the receipt's downstream value is argued (App B), not measured here. Currently this is the sharpest quotable inversion (hostile quote #1).

**F4 — §5 MH-kernel block is a single ~400-word unreadable paragraph in a 4–5pp short.** *Location:* §5 line 144. *Attack:* it is theory-complete-on-toy-EBMs *future work* with no trained-model runs, yet consumes a page-fraction as one impenetrable wall; a workshop reviewer under a 4–5pp budget will resent it. *Triage:* SHOULD-FIX. Break into 3–4 sentences of design-rules in main text + move the derivation/receipts (T_eff annealing, D=½s² coset erosion, FDT σ*) to an appendix at pruning. All the fine print is correct and present per CM-14 — this is a *craft/budget* issue, not a content one.

**F5 — BIBO-coercive-exit caveat not in main text next to the claim (C-6).** *Location:* caveat in App B only; §3.1 wormhole claim has no coercivity qualifier. *Attack:* task item 5 / C-6 require "BIBO needs coercive exits" next to its claim; a reviewer inverting the fine print notes the det J=1 "receipt" is silent on the fact that a jump into a non-coercive region violates BIBO. *Triage:* SHOULD-FIX. One clause in §3.1: "det J=1 certifies volume, not boundedness — the exit must lie in a coercive sub-level set or BIBO can fail (App B)."

**F6 — §4.1's "4.8×" is a kv16-only number sold as "the easy band."** *Location:* §4.1 point 2 / abstract; provenance A.2 (the 4.81× and +accuracy are kv16 specifically). *Attack:* main text says "at the easy band" without naming kv16, and the abstract scopes the whole pillar to "kv≤32"; a reviewer asks what the allocation payoff is at kv24/kv32. If positive only at kv16, the "escalatable asset" is thin. *Triage:* SHOULD-FIX. State the kv at which 4.8× holds and whether the payoff persists at kv24/32 (source has the per-level numbers).

**F7 — Internal source-report section numbers leak into a paper figure.** *Location:* Fig 1 title "§7.1 reach: landing vs distance (crossover at d=L)." *Attack:* "§7.1" is `paid-access-experiments` §7.1, not a section of *this* paper — unprofessional and confusing. *Triage:* SHOULD-FIX (trivial). Relabel to a paper caption; move "crossover at d=L" nuance into caption text (and note the observed edge is d≈3.2, the bracket, not L=2.5, so the title is also slightly misleading).

### NICE

**N-a — Fig 3 → appendix at pruning (task item 6).** In a 3-figure 4–5pp short, the epoch-scaling frontier (`fig_frontier_clean.png`) is the secondary "epoch-budget-wall" story; it is a clean appendix candidate, freeing main-text budget and letting Fig 2 absorb a noise panel (F2). Agree with the task's prompt.

**N-b — Reach falsification rests on 2 points.** §3.2 "squeeze collapses past the box" is demonstrated by exactly d∈{4.0,5.0}=0 (the observed edge is d≈3.2, inside the bracket). With a 6-point d-grid and a crossover bracket spanning ≈[2.5,3.4], "squeeze is box-bounded" is thin. The source itself suggests a heavier reach coord for a knife-edge. Legitimate future-work refinement, not a blocker.

**N-c — "Position" vs contribution profile.** Three of five contributions are boundary/negative results on learned memories. Honest, but a skeptic reads the paper as "the physics doesn't help on trained memories; it wins only on a designed testbed." Consider a one-sentence framing in §1 that owns this shape ("the certificate stack is the contribution; §4 maps precisely where it does and does not translate to a learned-memory advantage") — currently implicit.

---

## Missing-experiment list for the Hub (≈empty, as predicted)

Everything traces to `.claude/outputs/`; the noise-wall diagnosis and the squeeze-MH experiment are legitimate future work, not gaps. Two candidates only, both optional:

1. **(from F3) Certificate-payoff demonstration** — one arm where the no-physics router demonstrably violates a guarantee the wormhole preserves (non-coercive-exit BIBO blow-up, or router-erases-latch vs wormhole-transports-latch). This would convert the definitional "beats the router" into a measured one and is the single highest-leverage addition. *Check first whether `paid-access-experiments` already logged a router-latch or router-exit failure — if so this is a wiring note, not an experiment.*
2. **(from F6, wiring only)** kv24/kv32 allocation-payoff numbers already exist in `v1-pivot`; surface them. Not a new run.

No genuinely-missing experiment blocks submission. The real critical-path item is MF-1 (theory note must go live), which is a Head decision, not an experiment.

---

## The three sentences a hostile reviewer would quote

1. *"In §3 the authors say their wormhole 'beats a no-physics router'; in §4.2 the same 449-parameter router beats their mechanism in both FLOPs and accuracy — the only sense in which the physics 'wins' is that it attaches a det J=1 label of no demonstrated downstream value."*
2. *"Every result on a *trained* memory is a boundary or a negative — the gate is memory-agnostic, the energy-gated edge loses to a cheap classifier, and Hopfield keeps both the cost and the noise-robustness advantage — so the certificate stack's one clean win is a machine-precision self-consistency check on an analytic testbed the authors designed, with oracle channel placement and a hand-tuned 16× mass band."*
3. *"The paper's own headline figure shows the wormhole, the router, and the Newtonian control as three indistinguishable flat lines, and its self-described 'dominant negative,' the noise wall, appears in no figure at all — the visual story is that the physics changes nothing."*

(All three are *defused by fixes already in scope*: #1→F3, #2→own it per N-c and MF-1 posture, #3→F1+F2. None is a matrix violation; each is a framing/craft exposure.)

## Proposed handover updates (for the Hub)
- **V1 verdict: weak-accept**, contingent on MF-1 (theory note live) + SHOULD-FIX craft pass (F1 headline-figure occlusion, F2 noise-panel, F3 certificate-payoff framing, F4 §5 paragraph break, F5 BIBO caveat to main text, F6 kv16 scope, F7 figure relabel). Zero number-mismatches; charter/matrix clean except C-6 F5 placement.
- **Highest-leverage single addition (optional):** F3's certificate-payoff arm (router violates a guarantee the wormhole preserves) — the only thing that would lift the paper above weak-accept, because it is the one place the physics could show a *measured* learned-system win. Check `paid-access-experiments` logs before tasking as a new experiment.
- **Critical-path dependency unchanged:** "(Anonymous, 2026)" theory-note title/authors/arXiv (MF-1) blocks V1's §3 provenance and is shared across all three shorts.
- **Pruning note:** Fig 3 → appendix; §5 MH block → mostly appendix; frees budget for the Fig 2 noise panel.
