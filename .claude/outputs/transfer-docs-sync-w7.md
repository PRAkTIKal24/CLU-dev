# transfer-docs-sync-w7 — doc-curator report
Task + acceptance criterion: bring all transfer docs current through wave-6 + pay the flagged w5 docs-debt — update `HEP_primers.md` (incl. deferred debt), sweep w6 into `negative_results.md`, WRITE the ledger's w6 addendum (duty handed over by Hub). Per-doc edit list + output↔handover contradictions.
Status: **done.** All three docs edited; no code touched; only gitignored `.claude/**` files written. No git footprint.

Inputs read: AGENT_PROTOCOL · doc-curator def · `transfer-docs-sync.md` (w5 pass + docs-debt list, items 1–3 now in scope) · handover §10 w6 review entry (source of truth) · all 8 w6 outputs {v3-interference-ntk, v3-band-selection, v1-router-baseline, anchor-robustness, fit-gap-anatomy, paid-access-theory, v2-short-draft (via handover), post-w6 claims_matrix} · all three transfer docs in full.

Discipline: every number transcribed with report+section citation (no verdict reinterpretation); Def-2 nomenclature (inertial M / spectral μ); two-layer status tags preserved. Where an output and the handover diverge I flagged, did not resolve (see contradictions §).

---

## Doc 1 — `HEP_primers.md` (8 in-place edits, dated `> **Update (wave-6):**` blockquotes + 1 new subsection). Pays w5 docs-debt items 1–3.
1. **Header "Maintenance"** — added the wave-6 line listing the seven touched sections.
2. **§2.3** — NEW wave-6 block: the **ratio-dependent over-lr failure** (CM-5 addendum, `v3-band-selection` item 3). mult≈10 generalizes (align +0.80…+1.00, N∈{2,4}×ratio∈{4×,16×}); ≥30 harmful with ratio-dependent failure — inverts @16× (align −1.00), global-mass runaway MSE 35.3±47.3 @4×; MSE-trap reproduces.
3. **NEW §7.3** — "When training erodes the geometry it was given" (pays docs-debt #3). Demarcation-as-THEORY (μ²-witness + tilt-immunity δ≥0.05 with NO anchor + Exp-B control); anchor envelope (λ=100 bulletproof 5/5 r*=0.911 @~35× cost, λ≈10 gap 1.85 but 1/5 collapse); anchor ⟂ volume (broken-vol diverges 3.5k–176k regardless). Placement note: **V2 App B default, pending Jul-11**. `anchor-robustness` items 1/2.
4. **§8.2** — NEW wave-6 block: **pricing PREDICTIVE** (blind κ_eff → recall-horizon ranking ρ=1.0, sync ≤8% over 91× decade, CM-10) + **banding-as-method** (mis-band price curve matched 1.18 < uniform 2.42 < orthogonal 6.92 < anti 12.79; FFT selector = oracle gap 0.000 5/5, CM-11). `v3-interference-ntk` item 2, `v3-band-selection` items 1–2.
5. **§8.3** — NEW wave-6 measured subsection (pays docs-debt #2). Both legs presented honestly: **Leg 1 (w4)** routing works (0.875/0.812 vs 0.50, AUROC 0.95+, 1-hop beats N-hop); **Leg 2 (w6, CM-7 REWRITTEN)** physics-free router WINS (1.000/0.948 @8.81e7 vs gated 0.887/0.715 @1.18e8, 5 seeds); salvage = 1-hop edge flat-in-N (1.18e8) vs chain scaling (1.76e8→2.94e8). Energy-as-routing-signal forbidden.
6. **§8.4** — heading tag updated `[design hypothesis + one measured leg]` → `[reach/escape dichotomy PROVEN (w6); end-to-end experiments w7]`; NEW wave-6 block making the frame theory-complete: fit-gap falsifiables (a)/(b)/(c) reported (loan called ≈700 steps, +γ 92% / γ_φ −24%, reach +77%@c=0.5 secondary), causal box C_T, squeeze-cures-escape/wormhole-cures-reach [proven], state-dependent-gate design guard ∇g·Δ, certificate table, l0-gate-null-doesn't-test-access. `fit-gap-anatomy` + `paid-access-theory` (CM-12).
7. **Part IX (§9)** — NEW wave-6 block: interference firewall **MEASURED** (CM-9). Monolith R≈0.20 O(N) (S=1.38>self at N=8) vs modular ≈2e-5 O(1), R_far≡0.0, ∝κ² slope 1.99, mass-independent, persists 0/150/300 ep, ratio 1:9,000. **Metric discipline: report basin-displacement R, never the NTK cosine (0.99 both).** Retires F5 Open-2 for the naive question at laptop scale. `v3-interference-ntk`.
8. **§10.3 claim-status ledger** — added 8 w6 rows (firewall / pricing-predictive / banding-method / router-wins / loan-called+ladder / Hopfield-under-trained-PROVISIONAL / anchor-envelope / paid-access), each with key numbers + source report.

**Deliberately left unchanged:** Parts I–IV core derivations (still exact); the §5.2 FDT block (already w5-correct); §9.2 mechanism catalog table (structure claims still hold; the *measurement* is now in the new block). No chapter-level rewrites beyond the mandated new §7.3.

## Doc 2 — `philosophy-synthesis.md` (ledger — WROTE the w6 addendum; ⟲ chapters untouched)
The Hub handed the addendum duty to the curator this wave (task item 3). Appended `# ⟲ Wave-6 addendum (2026-07-07, doc-curator)` with:
- **Chapter deltas:** Ch.2 (sweet-spot grid-generalizes + ratio-dependent over-lr); Ch.3 (γ_φ definitively wrong-tool for fit, −24%); Ch.4 (COURSE-CHANGER 1 router reversal + regime-map PROVISIONAL + paid-access 4th pillar theory-complete); Ch.5 (COURSE-CHANGER 2 favorable — firewall measured + pricing predictive + banding-method; closes ledger's #1 V3 gap); Ch.7 (demarcation-as-THEORY + anchor envelope + anchor⟂volume); Ch.1 (loan CALLED ≈700 steps, boundedness-not-lowest-plateau — fills the w5 supplement's CM-1 "crossing unmeasured" flag).
- **Scorecard deltas** (rows 1,2,4,5,7) + **gap-list updates** (Tier-2 register P9–P14 all CLOSED; remaining = real data + w7 gates) + 5 new **scope guards for drafters** (CM-1/7/8/9/11) + positioning ripples (V3 = wave winner, GO confirmed; V1 routing→CM-2, Hopfield framing held; V2 anchored trainer; C-1 reversal noted).
- ⟲ protocol respected: no chapter rewritten; verified the addendum covers all 8 outputs before writing.

## Doc 3 — `negative_results.md` (w6 sweep: N2 re-tiered + 8 new entries N24–N31)
- **N2 RE-TIERED → PROVISIONAL** (per task): summary-row flag + full w6 note (2000-ep re-map closes 3/3 losing cells fid 0.40→1.00, gate 0.05→0.99, 9–10× savings; but 2 seeds × 2 ep × 3 cells → CM-8 freeze, both verdict-direction AND tier-A prominence frozen until w7 `regime-remap-2000ep`). Cites `anchor-robustness item 2`. **Verdict not resolved** — both directions presented per mandate.
- **8 new entries** (schema-complete: tried · exact numbers+citation · mechanism · scope · tier · vertical · source):
  - **N24** router-beats-energy-gating — **tier A, V1** (the CM-7 reversal)
  - **N25** γ_φ-does-not-recover-fit (−24%, wrong-tool) — tier B, V2/V3
  - **N26** CLU-not-lowest-long-horizon-plateau (broken-vol 0.14/LSTM 0.13 < CLU 0.22) — **tier A, all (CM-1 scope guard)**
  - **N27** reach-secondary + no-per-step-concentration — tier B, V1/V3
  - **N28** masslr_init snap-inversion — tier B, V3
  - **N29** λ≤10 seed-45 instability — tier B, V2
  - **N30** anchor-fails-memory-fidelity — tier B, V1
  - **N31** state-dependent-gate-breaks-volume (∇g·Δ, design guard, proven) — tier B, V1/V3+F5
- Updated summary index (8 rows), header maintenance line, paper-writer appendix-mining notes (per-vertical + CM-1-Pareto + physics-audit incl. C-1-reversal note), and open-provenance flags (N2 double-provisional; N24/N27 harder-band caveat; N25 magnitude caveat; N4/N5 unchanged; V3-GO tier note).
- Cross-refs added: N23 disposition now points to N24 (its P9 blocker resolved); N24 notes it is the 3rd energy-not-a-signal instance (with N3/CM-3, N21).

---

## How I verified
- Every quantitative statement cross-checked against its source-report section before transcription (e.g. router table `v1-router-baseline §1`; firewall R-table + κ-sweep `v3-interference-ntk §1`; loan curve `fit-gap-anatomy item 2`; anchor envelope per-seed `anchor-robustness §1`; paid-access checks A–F `paid-access-theory App N`; band-selection item-3 48-cell table).
- Confirmed the post-w6 `claims_matrix.md` content (CM-7 rewritten, CM-8 provisional, CM-9…CM-12) matches the handover §10 w6 entry and the outputs before transcribing status tags.
- Confirmed all four w5 docs-debt items are now discharged: #1 (primer full-refresh — §8.2/8.3/Part IX/§8.4 done), #2 (§8.3 measured wormhole subsection — done), #3 (§7.3 erosion subsection — done). #4 (frozen-scorecard reconciliation) remains a ⟲-structural feature, unchanged by design (see docs-debt below).
- No tracked code touched; all writes under `.claude/`. No git footprint.

## Output ↔ handover contradictions found (flagged, NOT resolved — per mandate)
- **None substantive.** The 8 w6 outputs, the handover §10 w6 review entry, and the post-w6 claims matrix are mutually consistent.
- **One sanctioned tension (documented, not resolved):** `anchor-robustness item 2` (2000-ep re-map) *directly contradicts the current CM-8/N2 "Hopfield dominant 26/26" wording*. The Hub has **already frozen this PROVISIONAL** (handover §10: "CM-8 frozen PROVISIONAL … V1 drafting HELD on w7 `regime-remap-2000ep`"), so I recorded BOTH the original finding and the reversal-in-progress in N2 and marked verdict+tier provisional. This is the intended state, not an unresolved disagreement.
- **Minor doc-hygiene note (not a contradiction):** `claims_matrix.md` still self-labels "v1.0, wave-5 review (2026-07-07)" in its header while its body already carries w6 content (CM-7 rewritten, CM-9…CM-12). Version-label lag only; a Hub matter (I don't edit the matrix).

## Docs-debt list (bigger than this per-wave mandate — flagged for the Hub)
1. **w5 docs-debt items 1–3 are now PAID** (primer §8.2/8.3/8.4/Part IX/§7.3). The primer is current through w6 on both positive and negative results.
2. **§10.4 "go deeper" reading list** is unchanged since w2 — still adequate, but a paid-access/reachability entry (Lieb–Robinson / basin-hopping / HMC-MCMC per `paid-access-theory` scout flags) would help the w7 reader. Low priority.
3. **Ledger frozen scorecard (lines 318–326)** remains at w3 verdicts by ⟲ design; the current state now lives across THREE addenda (w4/w5/w6). A reader must reconcile four locations. If the Head ever wants a single current-state scorecard that is a Hub call (would violate "chapters never rewritten" in place). Unchanged from the w5 flag.
4. **v2-short-draft** exists and builds (handover §10) but I read it only via the handover — the draft itself lives under `.claude/papers/v2-short/`; the paper-writer/referee own it, not the curator. No transfer-doc action needed; noted for completeness.

## Proposed handover updates (for the Hub)
- Transfer docs current through w6: `HEP_primers.md` (8 edits incl. new §7.3, all w5 docs-debt paid), `philosophy-synthesis.md` (w6 addendum written by curator per handover), `negative_results.md` (N2 re-tiered PROVISIONAL + N24–N31, now 31 entries).
- **CM-8/N2 is the one live scope-freeze the paper-writer must respect:** do not draft any Hopfield-comparison direction until `regime-remap-2000ep` (w7) lands.
- **Five drafter scope-guards now recorded in both the ledger addendum and the negatives registry** (CM-1 boundedness-not-lowest-plateau, CM-7 edge-not-energy-signal, CM-8 frozen, CM-9 report-R-not-cosine, CM-11 FFT-selector-needs-separable-timescales) — ready for the v2/v3 referee passes.
- The negatives registry is ready for w7 paper-writer appendix mining (per-vertical notes updated; C-1-reversal note added so the physics-audit negatives are treated as appendix/F5 material, not a main-text confession).
- Minor: consider re-labeling `claims_matrix.md` header to "wave-6 review" (content already w6).
