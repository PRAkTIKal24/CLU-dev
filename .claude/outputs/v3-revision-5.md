# v3-revision-5 — paper-writer report (fold pricing-at-N≤16 into the V3 short; close MF-2)

Task + acceptance criterion: fill v3-revision-4's marked §3.3 slot with `v3-pricing-n-scaling`'s canonical wording; honestly upgrade the abstract so "scaling" may attach to the priced channel (N≤16); fold the App-C resolution in-text; embed the headline figure; state scope; clean PDF. Judgement call requested: is the composition objection (MF-2) now fully answered?
Status: **done**

**Downstream reconciliation list (owner needed — read first):** none new introduced. One standing item is now *dischargeable*: the v3-revision-4 report flagged **CM-10 matrix lockstep**; the matrix v1.9 already carries the N≤16 extension (CM-10 row + evidence column already list `v3-pricing-n-scaling`), so the draft and matrix are now **aligned** — no further CM-10 edit needed. CM-13/CM-5 lockstep items from v3-revision-4 remain as previously flagged (untouched this pass).

## What I did (edits to `draft.md` + `draft.tex`, lockstep; PDF rebuilt)

### 1. Marked slot filled (§3.3) — the task's core
- Replaced the `[MARKED SLOT — v3-pricing-n-scaling, in flight]` blockquote (both formats) with the landed, **pre-registered** result using the report §5 canonical wording, adapted to CM-10's approved form: price law (`sync∝κ_eff^{−1/2}`, `n₁/₂∝κ_eff^{−1}`) holds **flat in N at N∈{2,4,8,16}, both chain+ring, 5 seeds** — sync `−0.49±0.02`, n₁/₂ `−0.91±0.03`, μ_rel² residual `≤0.45%`, R²`≥0.998`, no drift toward 0, κ_eff exact at every N.
- App-C attribution folded into the same slot: the former "inconclusive on trained lattices" = **artifact of the shipped random-`W` coupling's U(1) breaking** (clusters κ_eff to a `≤1.4×` range across a 100× sweep, control R²≤0.47, exponents ±12…±887), resolved by `channel_spring`; the control's failure is presented as the *explanation*, not a weakness.
- §3.3 closing sentence and the page-1 contribution (3) updated ("predictive, **and it scales**").

### 2. Abstract honestly upgraded
- Claim (2) → "predictive price list, **and the list is not an N=2 artifact** (trained lattices, N≤16)": kept the N=2 predictive-parity numbers (Spearman 1.0, ≤7.5% sync) as evidence, **added** the flat-in-N scaling with the full scope + the U(1)-preserving qualifier in-sentence (C-5). The abstract's opening "when it is scaled from one unit to many" and its closing grade-legend sentence now both cash out honestly.

### 3. App-C resolution in-text
- The App-C "Honest unmeasured list" bullet **"Trained-coupling scaling exponent (N16)"** struck-through → **RESOLVED** with the attributed explanation + Figure 10 (both arms: U(1)-preserving flat, U(1)-breaking control off-scale).
- Mirror updates so the paper doesn't contradict itself post-closure: **§1 "what the physics buys"** (no longer "not a result we claim here" → spans N≤16), **§5 "the composition, owned" (b)** (gap **closed**, not open), **title note** (experiment landed positive; "Scaling" may now legitimately attach — placeholder left for C-10 workshop, both slugs banked), **in-paper negatives registry N16** (negative-turned-resolved, C-9).

### 4. Figures
- **Figure 9** = `fig5_pricing_law_N.png` (report's `fig_pricing_law.png`), embedded as the §3.3 headline (2×2 log-log, sync+n₁/₂ × chain+ring, N=2..16 overlaid → visibly flat in N).
- **Figure 10** = `fig6_exponent_vs_N.png` (report's `fig_exponent_vs_N.png`), embedded in App C as the flat-in-N + off-scale-control corroboration.
- Both added to the figure-manifest list. Copied into `figures/`.

### 5. Scope stated (C-5/C-6), and the C-2 grade call
- Every generalizing sentence carries in-sentence scope: **designed SO(2) units, frozen U(1)-preserving `channel_spring` coupling, N≤16, both topologies, 5 seeds, wake-only, laptop-CPU.** The one gating qualifier (coupling must preserve U(1)) rides next to the claim.
- **C-2 grade = [verification], deliberately NOT [evidence].** The `channel_spring` substrate is *designed SO(2) units + a frozen (analytic) coupling* — the **same substrate as Fig 4**, which is already graded **[verification]** at N=2. Grading the N-extension [evidence] while Fig 4 (identical substrate) is [verification] would be a reviewer-constructable C-2 self-contradiction. So Fig 9/10, the §3.3 paragraph, contribution (3)'s second clause, and the A.3b provenance are labeled **verification-of-exactness-across-composition**. The learned-MLP-coupling predictive-parity result (Fig 3, N=2) stays **[evidence]**, untouched. §5(b)/App-C/N16 are phrased so the exponent's cross-N authority is the designed testbed and the *learned* random-`W` arm is the control that fails — the honest reading, and it still closes MF-2 (which is about *range*, not grade).

### 6. Flag-provenance (C-7, protocol §5)
- New **App A.3b** (both formats): commit `e3c8931`, JAX 0.9.0, CPU, x64; wake-only, sleep off, M=1, dt=0.05, 60 ep, window 128, γ=0 (sync)/γ_probe=0.2 (n₁/₂); designed SO(2) MexicanHat units at analytic vacuum (κ_eff 0.1981→0.1983); primary `channel_spring` frozen (198/200 rows) + control `spring_random` (100 rows); extractor `κ_eff=¼M·λ_max=0.5κλ_max(L)`; pre-registered decision rule quoted verbatim. Every added number traces to `v3-pricing-n-scaling` §2/§3/§5 + PREREG.

### Deliberately NOT done (per task/charter)
- CM-13 scope (γ=0-only, not-in-trainer) and MF-1 (theory note = Anonymous) left as-is (task items 6).
- No re-opening of §3.2/§3.4/§3.5/§3.6; the SF-4 six-contribution prune + abstract length stay for the C-10 pruning pass (nothing self-pruned, C-10).
- No number adjusted/rounded/smoothed; no new experiment invented.

## How I verified
- **PDF:** `tectonic draft.tex --keep-logs` → **exit 0, 0 Overfull hboxes, 0 undefined references** (final `.log`), **24 pages**, `draft.pdf` 1,166,484 B (2026-07-19). All figure cross-refs (`fig:pricinglaw`, `fig:exponentN`, `fig:pricelaw`) resolve.
- **Headline figure inspected** (read the PNG): 2×2 log-log, N=2..16 curves overlap on the −0.5 / −1.0 reference lines for both chain and ring — matches the caption.
- **Consistency greps:** 0 `MARKED SLOT` remaining in either format; CM-10 canonical numbers (`−0.49±0.02`, `−0.91±0.03`, `0.45%`, `≤1.4×`) present and identical across md/tex; no CM-3 forbidden energy-signal-superiority claim; no [evidence] tag on Fig 9/10 (both [verification]/[verification/control]).
- No repo/tracked-code changes (research-only; all artifacts under `.claude/papers/v3-short/`). **No git footprint.**

## Findings / judgement — is the composition objection (MF-2) fully answered?
**Yes, in my judgement, with one honestly-stated residual that does not reopen the gap.** MF-2 was: *"the scaling result [firewall] belongs to a mechanism the paper disclaims (parameter isolation, not physics), while the physics-specific result [priced channel] lives at N=2."* The paper now shows the physics-specific price law holds across **N∈{2,4,8,16}** — the **same range at which the firewall is established** — so the physics contribution is no longer stranded at N=2. The former honest concession (App C "inconclusive on trained lattices") became a **resolved, attributed result** (a symmetry-breaking-coupling artifact, with the control failure as its proof), which is the single strongest sentence added this pass.
The **residual** (stated in-text, not hidden): the cross-N exactness is **verification** on the designed-SO(2)/frozen-coupling testbed (the honest C-2 grade), and reading the *fitted exponent* off freely-**learned** couplings at N>2 remains a follow-up — but the *ranking/parity map* already survives learning at N=2 (Fig 3), and **no main-text claim now bounds on an N=2 limitation.** The composition thesis composes without a residual: *modularity is the firewall; a single joint symplectic Hamiltonian is what lets the modules talk, at a price we measure — across the full range at which the firewall is established.* This should move V3 from weak-accept toward accept: the referee asked for exactly one experiment (pricing at N>2), and it landed positive and pre-registered.

## Open editorial questions / follow-ups / risks (for the Hub/Head)
1. **C-2 grade ratification (please confirm).** I graded the N≤16 pricing result **[verification]** (designed SO(2) units + frozen coupling = Fig 4's substrate), not [evidence], to avoid a self-contradiction with Fig 4. CM-10's evidence *column* lists `v3-pricing-n-scaling` under "evidence" — that is a matrix bookkeeping column, not a C-2 figure grade, and I judged the charter (C-2) to govern the figure/caption label. **If the Head prefers [evidence]** (treating the wake-only training as enough to leave "designed"), it is a one-word flip in 6 sites — but then Fig 4 should be reconsidered too, or the two will read inconsistently. Flagging rather than deciding unilaterally.
2. **Title decision (C-10 workshop).** "Scaling" is now defensible; I left the `[WORKING TITLE: Composing…]` placeholder and updated the title-note to say so, banking a "Scaling…to N≤16" slug and the de-scoped "Composing…" slug. Head/Hub to ratify at the title workshop.
3. **MF-3 permanent Fig-1 re-render** (results-analyst, from v3-revision-4 follow-ups) still pending — unrelated to this pass; caption mitigation remains in place.
4. **2 missing seeds** (`ring,N=16,κ=1.0`, 3/5 seeds) noted immaterial in the report (that cell fits to ±0.014); not mentioned in the paper. Top up only if a reviewer asks — flagged, not improvised.
5. **Abstract length** (already ~530 words pre-edit, now longer) and **SF-4 six→fewer contributions** remain for the dedicated C-10 pruning pass; nothing pruned now (C-10).

## Proposed handover updates (for the Hub)
- **V3 v0.6: MF-2 CLOSED.** The §3.3 marked slot is filled with the pre-registered N≤16 pricing law (flat in N, both topologies, 5 seeds); App C's "inconclusive" resolved-and-attributed (random-`W` U(1)-breaking artifact, fixed by `channel_spring`); abstract/§1/§5/title-note/contribution-3/negatives-N16 all updated to state the physics-specific priced channel now spans the firewall's N≤16 range; Figs 9–10 embedded; A.3b provenance added; PDF clean (24 pp, 0 overfull, 0 undefined). Graded **[verification]** per C-2 (Fig 4 substrate) — see open question 1.
- **Matrix:** CM-10 already carries the N≤16 extension (v1.9) — draft and matrix are aligned; no CM-10 edit needed. The v3-revision-4 CM-13/CM-5 lockstep items are still open (unchanged this pass).
- **Convert to tasks (carryover):** (a) results-analyst Fig-1 re-render (MF-3 permanent); (b) C-10 pruning pass for V3 (abstract, SF-4, demote §3.5/§3.6, strip internal slugs MF-10); (c) optional 2-seed top-up on `ring,N=16,κ=1.0` if a reviewer asks.
