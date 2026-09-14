# v3-short-draft — paper-writer report

**Task + acceptance criterion:** draft the V3 workshop short (the lattice / CLU-Net paper), ML4PS-shaped, 4–5 pp main + appendix-maximal; charter C-1…C-10 + claims-matrix compliant (load-bearing rows CM-1, CM-5, CM-9, CM-10, CM-11); every number traces; CLU naming continuity; hermetic citations; reversible-O(1) as a marked slot if unlanded.
**Status: done** (draft written, LaTeX builds). §3.5 reversible-O(1) is a marked slot — the `v3-reversible-o1` measurement had not landed at draft time (only the task file exists; no `.claude/outputs/v3-reversible-o1.md`).

## What I wrote
- `.claude/papers/v3-short/draft.md` — canonical V3 short (main §1–3.6 + Appendices A–G, fully written, appendix-maximal per C-10).
- `.claude/papers/v3-short/draft.tex` — LaTeX build (article class, standard packages). **Builds:** `tectonic draft.tex` → `draft.pdf` (exit 0, 132 KB; 4 residual cosmetic underfull hboxes in prose, no overfull, no errors).
- `.claude/papers/v3-short/CHANGELOG.md` — v0.1 line.
- No tracked repo files touched (all under gitignored `.claude/papers/`). **No git footprint.**

## How I verified the build
`cd .claude/papers/v3-short && tectonic draft.tex` → `draft.pdf` exit 0. Numbers transcribed verbatim from source reports (never rounded/smoothed); each section cites its report + CM-x. No content pseudo-verified.

---

## Evidence map (claim → source report → CM-x / charter)

| §/App | claim (headline) | source report + locus | CM / charter |
|---|---|---|---|
| §1 | thesis (ML-first): joint Hamiltonian + sparse coupling scales the primitive; 3-part (firewall/pricing/banding); CLU naming continuity | task Thesis; HEP_primers continuity line | C-3, C-8, naming rule |
| §3.1 | κ=0 bit reduction; joint symplecticity <1e-12; Noether decay (1−γ)ⁿ; single-unit charge non-conserved at κ>0; cap v=c/√M_i; price law slopes −0.986/−0.499, μ_rel²=4κ/M exact; N-flat symplectic err 5.4/4.9/4.4e-8; wormhole skeleton 3.7e-3/1.2e-9 | `v3-lattice-build` items 1–4 | **verification** (C-2: designed testbed, analytic couplings) |
| §3.2 **headline** | firewall: modular R_off 2.26e-5 / R_far 0.0 vs monolith 0.20 unstructured; ∝κ² (slope 1.99); mass-independent (banded≡uniform); persists 0/150/300 ep; O(N) monolith S 0.64→1.38 (>1 at N=8) vs O(1) modular; 1:9,000; metric = R not NTK cosine | `v3-interference-ntk` items 1/3 | **CM-9 (verbatim)**; **evidence** (learned MLP potentials); C-5 scope in-sentence |
| §3.3 | pricing predictive: κ_eff blind → sync ≤8% over 91× decade; ranking Spearman 1.0 / keff↔n½ −1.0; **registered before measured** stated explicitly | `v3-interference-ntk` item 2 | **CM-10 (verbatim)**; evidence |
| §3.4 | degradation curve matched 1.180 < uniform 2.416 < orthogonal 6.924 < anti 12.791 @300ep 5/5; FFT selector gap 0.000 5/5 **when spectrally separable**; masslr-init fragile 4/5 (1 inversion → 12.5); mult≈10 default, ≥30 ratio-dependent harm | `v3-band-selection` items 1–3; `mass-lr-doctrine-test` | **CM-11 + CM-5 (verbatim)**; evidence; C-5, C-6 (separability qualifier next to claim) |
| §3.5 | reversible O(1): **MARKED SLOT** — scan-pure ⇒ invertible; γ>0 float amplification; no claim until measured | task item 5; `v3-reversible-o1` **not landed** | C-10 (slot, not pruned) |
| §3.6 | price of physics: contraction-forbidden dominant (15×, twin 0.013 vs CLU 0.190); BIBO 1.0/0.33; μ² 0.008/0.122; vacuum survives CD vs collapse; **loan called ≈700 steps** (twin 1–2 orders to 500, cross 700, diverge 196@5000 vs bounded 0.20–0.23); NOT lowest plateau (broken-vol 0.14, LSTM 0.13); reach +77%@c=0.5 aggregate-only; +γ 92% preserved, γ_φ −24% | `minus-the-physics` Part A; `fit-gap-anatomy` items 1–3 | **CM-1 (verbatim scope, incl. ≈700-step crossing)**; evidence; C-2 (constitutive-vs-kinematic) |
| §4 related | Mo (kinematics vs constitutive); Di Bernardo/Keller (geometry, guard-rail); modular-vs-monolith interference; RevNet backdrop | `mo-deep-read`; `di-bernardo-skim`; (modular anchors: bib pending) | C-8 (hermetic) |
| §5 | scope + certificate altitude + horizon (unmeasured list foregrounded) | all reports' limitations | C-5, C-6, C-4 |
| App A | flag-provenance per result group (incl. band-selection JAX 0.10.2 caveat) | all source reports | **C-7** |
| App B | mis-banding panel + over-lr N8/N9 both budgets | `v3-band-selection` 1/2, `mass-lr-doctrine-test` | C-9/C-10 |
| App C | honest unmeasured list (trajectory channel, irrep, block monolith, dynamical half-life, N16, intra-unit wormhole) | `v3-interference-ntk` limitations; `fit-gap-anatomy` item 4 | C-9 |
| App D | NTK-cosine-vs-R metric discipline | `v3-interference-ntk` 1(a) caveat | CM-9 tail |
| App E | mass-lr full 16-cell grid + N7 | `v3-band-selection` item 3 | CM-5; N7 |
| App F | loan curve + recovery ladder full | `fit-gap-anatomy` items 2/3 | CM-1; C-9 |
| App G | prominent negatives N7/N8/N9/N12/N16/N22 | `negative_results.md` | **C-9** |

**Compliance self-check:**
- **CM-9** used verbatim incl. the exact numbers (0.20 monolith, 2e-5 modular, R_far≡0.0, ∝κ² slope 1.99, 1:9,000, O(N) S=1.38 at N=8, mass-independent, persists), the "guarantees survive scaling only because of modularity" phrasing with the monolith as measured foil, and the metric-discipline tail (report R never NTK cosine).
- **CM-10** used verbatim: κ_eff blind, ranking Spearman 1.0, sync ≤8% across 91× decade, **registered before measurement** stated in-text.
- **CM-11** used verbatim: degradation curve numbers, FFT selector gap 0.000 5/5 **with the separability qualifier stated**, masslr-init fragile foil.
- **CM-5** used verbatim: ordering inducible / magnitude designed, mult≈10 default, ≥30 ratio-dependent (invert at 16×, runaway MSE 35× at 4×), read align/spread never MSE.
- **CM-1** used verbatim scope: contraction-forbidden (not causal cap, inactive), twin 15×, **crossing MEASURED ≈700 steps**, do-NOT-claim-lowest-plateau caveat, reach secondary/aggregate-only, +γ 92% preserved / γ_φ −24%.
- **CM-3 (forbidden) never appears** — V3 makes no energy-as-confidence-signal claim (no gate/routing content; the wormhole edge is described as flat-in-N reach mechanism only, no energy-superiority).
- **Canonical constants (matrix §1)** cited identically: latch q∞=q0+εp0/(Mγ); charge decay (1−γ)ⁿ; n₁/₂∝μ⁻²; floor 2ln2/(−ln(1−γ)); h*≈γ/2; det J=(1−γ)^d / per-unit ∏(1−γ_i)^{d_i}; inertial M vs spectral μ (never bare "mass" — nomenclature ¶ in §1).
- **Scale qualifiers in-sentence** everywhere (2-dim units, chain, N≤8, 3–5 seeds, laptop-CPU, synthetic) — C-5. No scope-free plurals.
- **C-2:** §3.1 labeled verification (designed, analytic couplings); §3.2/3.3/3.4/3.6 labeled evidence (learned MLP potentials). Constitutive-vs-kinematic contrast foregrounded in §3.6/§4, not buried.
- **Naming:** "the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)" — continuity sentence present (§1).
- **Hermetic (C-8):** cites only J&P 2026 + theory note (Anonymous, 2026) + external published (Mo, Di Bernardo/Keller). No V1/V2 short referenced.
- **Placeholders:** `[WORKING TITLE: …]` (3 candidates below), `[AUTHORS PLACEHOLDER]`.

---

## ⚠ Open editorial questions (for the Hub / Head)

1. **C-1 CONFLICT — no audit paragraph included; confirm.** My system-prompt/task reference a "physics-audit paragraph first (C-1)" and the task says to reuse the V2 canonical wording "once the Hub confirms sign-off." **But the binding charter C-1 (philosophy-synthesis.md L420) was REVERSED by the Head on 2026-07-07: "(a) no defensive audit paragraph in any paper."** Hub sign-off on the V2 canonical text has **not** been confirmed (still open in `v2-short-draft.md` Open-Q2). I therefore followed the charter and **omitted the audit paragraph** from the V3 paper, leaving an editorial note in the header. **Decision needed:** confirm omission (my reading), OR supply the confirmed canonical text if the Head wants it in after all. The V2 canonical wording is preserved verbatim in `v2-short-draft.md` §"canonical C-1 physics-audit paragraph" if you reverse the reversal. This is the single biggest cross-short consistency risk — V2's *current* draft.tex still HAS the audit paragraph (L33), so V2 and V3 currently disagree on C-1. Please reconcile both drafts to one policy.
2. **§3.5 reversible-O(1) slot.** Written as a marked placeholder (scan-pure ⇒ invertible framing, no numbers, no claim) because `v3-reversible-o1` had not landed. When it does, I fold in the measured triple (gradient tol, peak-mem O(1)-vs-O(T) at T=1024, wall-time ×, γ>0 usable horizon). If the run is expected *after* the referee pass, flag whether §3.5 ships as a horizon paragraph instead or is cut for the workshop length.
3. **Headline figure.** I identify **Figure 1 = O(N)-vs-O(1) interference scaling** per the task ("the O(N)-vs-O(1) figure is the headline"). Asset is `.claude/outputs/v3-interference-ntk/fig1_interference_bars.png` — but that is the *bars* figure; the O(N)-vs-N *scaling curve* (item 3, S vs N) may need a dedicated plot (the report gives the table S=0.64→1.38 vs modular ~1e-4 but I did not confirm a standalone scaling-curve PNG exists). **Missing-figure flag:** confirm whether a scaling-curve PNG exists or the referee/engineer should generate one from `through_training.json`/item-3 data. Figures are not embedded (no `\includegraphics`); point me at final PNG paths for the referee pass.
4. **Citation string for the theory note.** V3 cites "(Anonymous, 2026; the theory note)" per matrix §4 open slot. Once the F5 note is live with an arXiv id + Head title, supply the exact string so V2 and V3 agree.
5. **Related-work external bib (C-8/M1).** The modular-network / catastrophic-interference / MoE paragraph in §4 has **no specific author-year citations** — I did not have a scout report drafting V3 modular-related-work prior art (the scout set covers adaptive-compute, goldstone, industrial-datasets, venues; Mo and Di Bernardo come from `mo-deep-read`/`di-bernardo-skim`). I left those anchors as `[·]`/placeholders rather than fabricate citations (binding rule 5). **Request:** a scout pass or a bib for the modular/interference positioning, or confirm the generic framing is acceptable for the workshop.
6. **Venue length.** Drafted appendix-maximal (C-10); main text ~4–5 pp dense. The pruning pass near the Jul-11 venue pick needs the exact page limit (ML4PS-shaped). §3.5 slot and §3.1 wormhole-skeleton paragraph are the first pruning candidates if tight.

## Missing-number flags (none fabricated; genuine gaps)
- **Reversible-O(1) numbers** (§3.5): do not exist yet (`v3-reversible-o1` unlanded). Slot marked, no improvised numbers.
- **Trained-coupling scaling exponent** (App C / N16): inconclusive on trained lattices (weak identifiability) — reported as such, not filled; the designed-lattice exponents (−0.499/−0.986) remain the cited authority.
- **N=4/8 firewall through-training**: `v3-interference-ntk` ran through-training only on the 2-unit case (data generator is 2-unit); N=4/8 firewall is init + structural-argument + 2-unit persistence. Stated as such in §3.2; not over-claimed.
- **O(N)-scaling-curve figure asset**: see editorial Q3 — may need generation.

## Working-title candidates (Head picks / workshops)
1. **Scaling a Conservative Memory Primitive: a Measured Interference Firewall, a Predictive Communication Price List, and a Method for Allocating Timescales** *(current placeholder; ML-first, C-3)*
2. **A Firewall, a Price List, and a Band: What It Costs to Scale a Hamiltonian Memory to Many Units**
3. **CLU-Nets: Modular Symplectic Recurrence with an O(1)-in-Width Interference Firewall and a Curvature-Predicted Communication Cost**

## Positioning prose provenance (which report I lifted from)
- §4 "Symmetry and memory lifetime" (Mo) — lifted/condensed from `mo-deep-read` §4 (same framing as V2's related-work Mo paragraph, kept consistent).
- §4 "Geometry-first structured recurrence" (Di Bernardo/Keller) — from `di-bernardo-skim` "Draft prose to lift", guard-rail applied (headline pricing/budget, not "we choose G/H").
- §4 "Modular vs monolithic sharing" — my synthesis from `v3-interference-ntk` 1(c) (F5 §6 catalog framing); **no external citations** (see editorial Q5).

## Proposed handover updates (for the Hub)
- **Draft exists for paper-referee (next wave):** `v3-short/draft.{md,tex}` (builds, appendix-maximal, not self-pruned). Headline = Fig 1 (O(N)-vs-O(1) firewall).
- **Cross-short C-1 inconsistency to resolve NOW:** V2 draft.tex L33 contains the audit paragraph; V3 omits it per the reversed charter C-1. A reviewer reading both would see a policy split (M2 risk). Recommend the Hub pick one policy and reconcile V2 to it (either drop V2's audit ¶ or supply confirmed canonical text for both). Flagged as V3 editorial Q1.
- **Claims-matrix §4 open slots this draft touches:** theory-note citation string (still needs live arXiv id); V3 modular-related-work bib (new gap — no scout report covers it).
- **CM-5 addendum already reflected:** the w6 ratio-dependent failure ("100× inverts at 16×, global runaway at 4×; mult≈10 safe default") is in §3.4(iii) + App B.3 verbatim.
- **New CM candidate (firewall):** the `v3-interference-ntk` report proposed adding a dedicated CM row for the firewall (currently folded under CM-9). The draft uses the CM-9 v1.0 wording; if the Hub promotes it to its own CM-x, the §3.2 text is already the approved-wording target.
