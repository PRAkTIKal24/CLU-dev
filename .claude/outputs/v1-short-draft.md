# v1-short-draft — paper-writer report

Task + acceptance criterion: draft the V1 short as an ML4PS position/theory paper ("test-time compute on a conservative memory as PAID ACCESS"), lead with pillar-4 certificate stack (CM-12), + 3 honest supporting pillars, appendix-maximal, build with tectonic. → report evidence map + editorial questions + missing-number flags to `paper-referee` (w9 `v1-referee`).

Status: **done.** `draft.md` (canonical), `draft.tex`, `draft.pdf` (built clean via tectonic, 159 KB, headline Fig 1 embedded), `CHANGELOG.md` — all under `.claude/papers/v1-short/`. No code touched (research/writing only; no git footprint).

## What I did
- Read (in order): AGENT_PROTOCOL → Positioning Charter (philosophy-synthesis §Charter, C-1 REVERSED confirmed) → claims_matrix v1.3 (load-bearing rows CM-2, CM-3[forbidden], CM-7, CM-8, CM-12) → task → all listed evidence reports + negatives registry V1 tier-A.
- Wrote a 5-section main text (Intro / Setup / §3 certificate stack VERIFICATION / §4 three learned-memory pillars EVIDENCE / §5 position+horizon) + 5 appendices (A flag-provenance ×4, B certificate table, C full honest tables, D negatives, E analytic checks).
- Built LaTeX with **tectonic** → `draft.pdf`. Headline figure `paid_access_reach.png` copied from `.claude/outputs/` and embedded as Fig 1. Only cosmetic under/overfull-hbox warnings (the wide 3-col certificate table); no errors.
- 3 working titles proposed (position-paper register) — see below. Title + authors are placeholders per C-6/policy.

## Framing decisions (Head-binding, honored)
- **Venue-shape:** ML4PS position/theory, physics-forward, mechanism-and-certificate led — NOT a benchmark paper. Thesis stated verbatim as the abstract/§1/§5 spine: every capability-buying mechanism carries an explicit physical receipt; certificates are theorems, not empirics.
- **C-1 REVERSED:** NO audit-confession paragraph anywhere (unlike the extant v2-short draft, which still carries one pending v2-referee removal). The corrected-mechanism theorems are left to the F5 note; V1 only cites J&P 2026 for the primitive.
- **C-3 ML-first:** §1 opens on the test-time-compute literature (ACT/PonderNet/CALM/MoD/MoE/EBT) and frames physics as the *derivation apparatus* for pricing access — a reviewer summarizes this as "a principled account of test-time compute," not "a physics problem set."
- **C-2 grade labels:** §3 (paid-access battery on designed analytic testbed, oracle placement) labeled **verification**; §4 (trained MQAR memories) labeled **evidence**. Grade banner at the top of each section.
- **C-5 scope in-sentence** on every generalizing claim (dim 2/4, 5 seeds, oracle placement, laptop; MQAR vocab-256, kv≤32, N≤8). **C-6 certificate fine print next to the claim** (LTT exchangeability + ECE≈0.10; e^{2ζ} = matched-quadratic-H certificate; detJ=1 only for frozen gate; BIBO needs coercive exits).

## Evidence map (claim → source report → CM-x / negative)
| draft location | claim | source | CM / N |
|---|---|---|---|
| Abstract, §2, §3 | reach/escape dichotomy; causal box C_T energy-blind, L_i=Tεc/√M_i | paid-access-theory Prop-A2/Def-A4 | CM-12 |
| §3.1 | squeeze det=1, injection ≤ e^{2ζ}H (matched quad H), governor re-absorb 2ζ/γ_c | paid-access-theory Prop-12 + §4.1; paid-access-experiments §7.3 | CM-12 |
| §3.1 | wormhole detJ=1 exact, ledger=0 exact, latch transport p⊤XΔ exact | paid-access-theory Prop-A6/A7; paid-access-experiments §7.2–7.3 | CM-12 |
| §3.1 fine print | state-dependent gate breaks volume by 1+∇g·Δ (detJ=2.05) | paid-access-theory Prop-A6; paid-access-experiments unit test | CM-12 / N31 |
| §3.2, Fig 1, App C.1 | reach crossover: squeeze bounded (edge ~3.2, bracket [L,L+p₀sinhζ/M₀]); wormhole flat; Newtonian control past L; router no cert; dense fails reach | paid-access-experiments §7.1 (dim 2&4, 5 seeds) | CM-12 |
| §3.3, App D N1 | prior squeeze-retry null does NOT test reach (selection-among-reachable vs crossing-to-unreachable) | paid-access-theory §3.3; v1-l0-gate via negatives N1 | CM-12 / N1 |
| §4.1 | calibration memory-agnostic: raw 0.431→cal 0.869 (CLU); Hopfield 0.18→0.88 ≈ CLU 0.43→0.87 | v1-pivot finding 1; claims_matrix CM-2 (evidence incl. minus-the-physics B) | CM-2 |
| §4.1 | escalatable asset: 4.81±0.44×@kv16 above always-full (0.894@629 vs 0.847@3000) | v1-pivot finding 2 | CM-2 |
| §4.1 | LTT 30/30 valid, cov 0.647@ε0.05 (risk 0.030); ECE 0.100±0.021 under-confident | v1-pivot findings 5,6 | CM-2 / C-6 |
| §4.1 negs | abstention-vs-Hopfield lost (Hopfield 0.983–1.0); energy≈margin ΔAUROC −0.004…+0.024 | v1-pivot findings 3,4 | N2, N3/CM-3 |
| §4.2, App C.2 | 1-hop edge flat 1.18e8 vs chain 1.76e8→2.94e8, distant 0.41→0.28 | v1-router-baseline §2; v1-wormhole-routing | CM-7 (salvage) |
| §4.2 | router 1.000/0.948 @8.81e7 beats gated 0.887/0.715 @1.18e8, 5 seeds, all mixes, both N | v1-router-baseline §1,3 | CM-7, N24 |
| §4.3 cost FINAL | Hopfield ceiling in 1 matvec (0.947–0.979 @β≥5); 9–10× is intra-CLU not vs Hopfield | regime-remap-2000ep Item 3 | CM-8 (cost) |
| §4.3 accuracy PENDING | 2000ep closes 3/3 cells (fid 0.40→1.00, gated 0.05→0.99), 2s×2ep×3cells only | anchor-robustness item 2; v1-hopfield-stress (500ep 26/26) | CM-8 (accuracy, marked slot) |
| §4.3, App D N30 | anchor does not transfer to memory fidelity (pins random init) | anchor-robustness item 2 | N30 |

## CM-compliance self-check
- **CM-12** stated in approved form: causal box energy-blind; squeeze cures escape & provably cannot beat box; crossover **bracket [L, L+p₀sinhζ/M₀]** (not knife-edge); wormhole detJ=1 exact, ledger=0 exact, latch p⊤XΔ exact; beats no-physics router (no volume cert) AND dense-V (fails reach); Newtonian control confirms cap. Scope: **oracle placement** (stated §3 banner + §5).
- **CM-3 FORBIDDEN honored:** nowhere does the draft claim energy is a better signal than a learned alternative — stated explicitly as a *non-claim* in §4.1 (N3) and §4.2 (N24). Router/margin results framed as findings; the paper's value = the certificate stack.
- **CM-2 / CM-7 / CM-8** use approved wordings; CM-8 accuracy left as marked slot with literal `[pending: regime-remap-complete]` token (V3-§3.5 precedent).
- **Canonical constants** (matrix §1) cited exactly: det J=(1−γ)^d and position-gated (1−γφ(q'))^d; latch q∞=q0+εp0/(Mγ) form; inertial M vs spectral μ nomenclature sentence.
- **CLU naming continuity sentence** present verbatim: "the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)."
- **Hermetic citations (C-8/M1):** only J&P 2026 (primitive) + Anonymous 2026 (theory note, third person) + external published (Angelopoulos, Banino, Duane, Geifman, Gladstone/EBT[flagged preprint], Graves, Lieb–Robinson, Neal, Platt, Ramsauer, Raposo, Schuster, Shazeer, Wales–Doye). No cross-short citations. External cites are "precedent for the bound, claim the mechanism-design consequence" per paid-access-theory §4.2.

## Related-work prose provenance (per task)
- Test-time-compute genre map (§1, footnote) and the EBT/CALM/MoD/ACT/PonderNet/Hopfield positioning lifted/condensed from **`scout-adaptive-compute-prior-art.md`** (its "every component genre is crowded; defensible territory is the certificate layer" verdict + the citation list in its §"Lineage citations reviewers will expect").
- Wormhole-vs-attention/MoE, squeeze-vs-basin-hopping/MCMC, cap-vs-Lieb–Robinson positioning from **`paid-access-theory.md` §4.2** (prior-art contamination flags).

## Appendix-maximalism (C-10) — what went where
Main text = main results only. Appendices (all fully written, nothing self-pruned): A = 4 flag-provenance tables (C-7, per source report, commits + seeds + all non-default flags) + cross-section reproducibility note; B = full paid-access certificate table (paid-access-theory §4.1) + BIBO/coercive-exit caveat; C = full reach battery grid, full routing grid, Hopfield iteration-parity grid, preliminary 2000-ep closure grid; D = tier-A negatives N1/N2/N3/N24/N30/N31/N23 with the §3.3 reach-vs-retry disambiguation on N1; E = the 6 analytic float64 checks (A–F).

## Proposed 3 working titles (position-paper register — Head to choose)
1. *Paid Access: Test-Time Compute on a Conservative Memory as a Physically-Metered Resource* (current placeholder)
2. *Every Cheat Has a Receipt: Certified Test-Time Compute on Symplectic Associative Memories*
3. *Reach, Escape, and the Price List: A Certificate Stack for Test-Time Compute in Conservative Memories*

## Open editorial questions (for the Hub / Head → v1-referee)
1. **Grade-of-the-lead tension.** The paper LEADS (per task) with the pillar-4 certificate stack, which is **verification-grade** (designed analytic testbed, oracle placement) — while C-2 reserves "evidence" for learned systems. I resolved this by banner-labeling §3 "verification" and framing the *position* (not a discovery) as the lead. Confirm this is the intended posture for a position/theory paper, or should the abstract foreground §4's learned-memory evidence more?
2. **CM-2 Hopfield 0.18→0.88 number** is carried from the claims-matrix canonical wording (evidence: minus-the-physics Part B), which was NOT in my reading list — I cite it via the matrix + v1-pivot's CLU 0.43→0.87. Confirm acceptable, or should the Hopfield-calibration number be dropped to only what v1-pivot directly measured?
3. **EBT (Gladstone et al. 2025)** is a preprint with unverified venue status (per scout) but is the single highest scoop-risk neighbor ("energy as inference-time verifier"). I cite it prominently in §1 and §5 as the frame's closest black-box foil, flagged "(preprint; venue status unverified)." Confirm prominence is right (scout recommended "cite prominently, not buried").
4. **thread9-mh-kernel optional slot** is a one-sentence horizon hook + a literal `[optional slot: thread9-mh-kernel]` marker in §5. The theorist's note has not landed (no output file yet); leave as marker until it does.
5. **Author-count / anonymization:** built with placeholders visible. If the target venue is ML4PS (double-blind, no rebuttal per scout), an anonymized build blanks title/authors — flag for the pruning/submission pass.

## Missing-number flags (for the Hub — do NOT improvise; fold at revision)
- **CM-8 accuracy grid (BLOCKING the accuracy-reversal claim):** `regime-remap-complete` has **no output file yet** (`.claude/outputs/regime-remap-2000ep.md` has Item 3 only; Items 1/2/4 stubbed). The full ≥5-seed 26-cell map + epoch-scaling frontier + the one-paragraph CM-8 accuracy replacement are needed to (a) upgrade §4.3 from "3/3 sampled cells, 2 seeds" to a grid-wide statement and (b) fill App C.4. Draft carries the literal `[pending: regime-remap-complete]` slot.
- **γ>0 governor re-absorption certificate:** paid-access-experiments used γ=0 for a sharp box; the `t_reabsorb≈2ζ/γ_c` timing row of the certificate table (App B) is verified only to leading order at γ=0 (App E check F). Flagged as future work in §5, not claimed as measured at γ>0.
- **thread9-mh-kernel:** if it lands before the referee pass, a physics-heavy "acceptance as a certified kernel" paragraph folds into §5 at the marked slot.

## Proposed handover updates (for the Hub)
- **§10 / papers:** V1-short DRAFT exists and builds (`.claude/papers/v1-short/{draft.md,draft.tex,draft.pdf,CHANGELOG.md}`), ML4PS position/theory shape, C-1 REVERSED (no audit ¶ — contrast v2-short which still needs its removed). Headline Fig 1 = `paid_access_reach.png`. Ready for w9 `v1-referee`.
- **Claims matrix:** draft consumes CM-12 as adopted (Head-confirm-to-adopt flag in matrix — the draft treats pillar-4 as ADOPTED per task's "LEAD with the physics win"; if the Head has not formally flipped CM-12 from "V1 pillar 4 (Head confirm to adopt)" this should be reconciled before submission).
- **Blocker for accuracy story:** `regime-remap-complete` is the one missing input; everything else is drafted. Spawn it (results-analyst) so §4.3/App C.4 can fold the grid-wide accuracy sentence at the referee revision.
