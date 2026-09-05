# transfer-docs-sync-w13 — doc-curator report

**Task + acceptance criterion:** four transfer docs current through w13; the "13.9×" sweep done; N10 marked under-re-test and N18 amended; the two Thread-10 dictionary errors corrected everywhere; nothing deleted (C-9); docs-debt list reported.

**Status: done.** All four docs edited. Acceptance met on every clause. **Three items require Hub action** (one matrix action I am not permitted to take, two continuity gaps I refused to backfill). **Zero output↔handover contradictions found** — every number in this wave's reports agrees with §10, including the ones that overturn prior claims.

**Sources read, in protocol order:** `AGENT_PROTOCOL.md` → `handover_context.md` §10 (2026-07-10 WAVE-13 review, + the 2026-07-09 23:5x entry) → `v5-gate.md` → `xy-lattice-theory.md` → `v2-symmetry-deepdive.md` §7bis + §4.1(c) + X4/X5/X6 + O6/O7/O8 → `f5-corrigendum.md` §2 (for the carried-over N4 item) → `negative_results.md`, `future_work.md`, `HEP_primers.md`, `philosophy-synthesis.md`. Matrix v1.8 read, **not edited** (per task).

---

## Edits, per doc (for Hub diff-review)

### 1. `.claude/negative_results.md`
**New entries — N46–N51.** Tiers are curator-proposed with justification on record (task said "propose tier yourself, justify"); Hub may re-tier, the evidence is transcribed either way.

| # | one line | **tier (proposed)** | justification for the tier |
|---|---|---|---|
| **N46** | emergent arm has **no coset register**; Cor-13/CM-16(a) designed-only. *Both halves recorded — the unification generalizes exactly* | **A** (V2/V5) | a **scope collapse of a published-track claim**, not an internal caveat; the sharpest "why designed symmetry matters" statement we own; must travel with every CM-16(a) citation |
| **N47** | ⛔ `fdt` NaN gradient at γ=0; **no FDT-correct model ever trainable at defaults** | **A** (all, physics-audit) | blocking; retroactively bounds what "we ran FDT-correct training" could ever have meant |
| **N48** | `GatedCoupling` returns mean **energy**, not free energy; wormhole **repels its endpoints** | **B** (V1/V3+F5) | **N31 precedent** — a *proven design guard on an un-shipped path*. Carries an explicit **promote-to-A trigger**: the first shipped wormhole-array result |
| **N49** | `spring_coupling`'s random `W` breaks global U(1); trained lattice learns `J/J_true ≈ 0.02` | **A** (V3/Thread-10) | a genuine **experimental** negative about *learned* couplings (not a guard), and the named biggest threat to the Nature-MI thesis — with a free fix |
| **N50** | `two_timescale_orbits` cannot identify a coupling; `r* < 1e-3`, no ring | **B** (V3) | scope limit, not a bug. **Explicitly states it does NOT touch CM-5/CM-10** (banding is about mass, not coupling) — as instructed, so nobody over-reads it |
| **N51** | raw `n₁/₂` exponents **not discriminating**; matched designed control gives the same shallow slopes | **A** (V2/V5) | *not* a drafting defect (unlike tier-C N43/N44): a **measured instrument-validity result with its own control** that *prevented* a false claim. Generalized to the standing rule: never quote `n₁/₂` without `Δ` and `ℓ_θ/Δ` |

**Amendments (nothing deleted — C-9):**
- **N10 → `⚠ RE-OPENED — UNDER RE-TEST`.** Full block added: Exp-C is `relativistic` (`config.py:220`); CM-17 ⇒ both arms were non-Gibbs; the control parameter is `T/(m₀c²)` alone (Exp-C at **1.0**, `KL = 0.384` nats, `Var_MJ/(M_eff T) = 2.6995`; `finalA` at **0.04**). Tier A retained; entry retained; points at `relativistic-gibbs-expc` (F-9). **States explicitly that the Newtonian mechanism is not in doubt — only the attribution verdict.** *Curator did not adjudicate the outcome.*
- **N18 → addendum.** The *fixed* sampler still doesn't sample Gibbs in relativistic mode; free-particle proof; the full `T/(m₀c²)` table; the bit-identical `−0.7290074` cross-check; the lattice confirmation (`⟨cosΔθ⟩/Gibbs = 1.134`, drift ≈ 0); the three fixes in increasing cost. **Preserves the subtlety** that the Gibbs *measure* is relativity-insensitive — the failure is in the **sampler**, not the thermodynamics.
- **N4 → the retracted "2.6" dropped**, replaced by "bounded `O(1)` charge non-conservation — an excursion, not a drift", with a dated Update block recording *why* (running-supremum, window-dependent 2.62→2.96, non-reproducible 2.625 vs 2.734). **This discharges the w12 `f5-corrigendum` reconciliation item that was explicitly assigned to this file and never done.**
- Summary-index rows for N4/N10/N18 annotated; **N46–N51 rows added**.
- Header "Maintained by" extended through w13 + **a standing curator-debt warning** that w11–12 were never swept.
- **Paper-writer notes:** new **V5 appendix list** and **Thread-10/Nature-MI appendix list**; N47 added to the physics-audit line; two new cross-thread patterns recorded (**"the untested derivative" family** N47+N48+N51; **"instrument validity precedes inference"** N10+N18).
- **Provenance flags:** N10 non-adjudication; the CM-16 split the curator cannot make; tier rationales; **N48's non-contamination rests on the *Hub's* call-graph audit, not on the source report** — flagged so a reproducing agent re-verifies; `v5-gate`'s own C-9 superseded artifacts flagged as *intra-report* negatives so the paper-writer doesn't mine them.

### 2. `.claude/future_work.md`
- **Thread-10 section substantially rebuilt** (the two dictionary errors originated *here*):
  - **🔴 Correction 1:** `ρ_s = J = 2κr*²`, **not `F²`**. `F² = M_ch r*²` is an *inertia* and **cancels from the Gibbs measure**; it is a stiffness only in the 0+1D single-register sigma model. ⇒ masses and kinetic mode do not affect equilibrium statistics at all; `tie_channel_mass` is *dynamical*, not thermodynamic.
  - **🔴 Correction 2:** **`n = dim(G/H) + 1`** (S¹⇒XY/O(2), S²⇒Heisenberg/O(3), Z₂⇒Ising/O(1)); O(n) universality **only for spherical cosets**; a torus bank `T^k` is `k` decoupled XY models. *Explicitly notes the primer's `#registers = dim(G/H)` was and remains **correct** — the error was only in the O(n) label.*
  - **🔴 Correction 3:** *"a 2-D array remembers"* replaced. Equilibrium forgets any generic pattern **in every dimension**; what survives below `T_KT` is the **winding number** `(w_x,w_y) ∈ ℤ²`, `τ ∝ L^{πρ_s/T−2}`, exponent vanishing **exactly at** the Nelson–Kosterlitz jump. **The KT transition *is* the memory transition**; `T_KT = 1.786 κ r*²`.
  - **Aliasing and vortices are the same winding** (`S²` ⇒ `π₁ = 0` ⇒ neither); **the honest null** `D_Θ = D_θ/N` in every dimension (verified 15.9 vs 16) + 1D `τ ∝ 1/N` — both marked *must never be sold as a phase transition*.
  - **Prerequisites P1–P7** as a measured table; the `L ≥ 8` size floor (`L=4` is **+13.5%** off `T_KT`); the parameter-free **1-D control** go/no-go (`ξ = −1/ln(I₁/I₀)`, `HELD FOR W15`); the **kill criterion**.
  - Marked `→ SHOWN` with provenance: the designed reduction · γ-independence of the stationary measure · `D_Θ = D_θ/N` · the 1D null · gate-as-annealed-Ising-bond · the γ/T exchange-rate principle (now with its **interior optimum at `T* = B`** — *"cold is not free"*).
- **New section — "Relativistic register physics (deep-dive §7bis)"**: **R5** retention–bandwidth bound (→ `rb-bound-trained`; *the trade-off exists only in relativistic mode* — the Newtonian null is half the result), **R3** rapidity/companding register (`p₀=500` stores `Δθ=7.678` vs Newtonian 625.0), **R7** causal memory-lifetime floor (pathwise; `D_rel` saturates while `D_newt` grows linearly), **R6** *conditional* aliasing protection (**48.86× @ γ=0.05 · 2.479× @ 0.02 · 1.288× @ 0.01 · →1 as γ→0**), **O6/O7/O8**. **O5 marked `→ SHOWN`** with the *honest flag* carried verbatim (**not a ChPT form factor** — a relativistic rotor; CLU term only).
- **V5 section rewritten as GO**, with the scope split stated as the headline (vault + sign flip + V-curve unification generalize; **the latch register does not**), the five named `v5-gate` follow-ups (R1b, R2, R3b, R5-Arrhenius, R7-capacity), and **`T_φ(q)` = `PARKED(Head hold, w13)`** plus the `with_friction_field()` ergonomics blocker.
- **F-6/temperature-lever entry** marked `→ SHOWN` with the **corrected mechanism** (absorb-only ⇒ brake **and** refrigerator; vault **107.77 ± 4.78×**; coupled-bath **13.88×** rejected by **8.11 ± 0.37**), the **DO-NOT-FIX** note on absorb-only, the three `T_φ` spec corrections, and the *write-attenuator* sub-finding.
- **V2 §emergent-SSB** entry reframed by N46 (the induction question is now *"can induction close a 12-order gap?"*; the emergent washboard is a **free `tilt_delta`**).
- Footer: dated w13 sweep note listing corrections, the superseded number, `→ SHOWN` items, new boundaries, and **"nothing deleted (C-9)"**.

### 3. `.claude/outputs/HEP_primers.md`
**Three new subsections, house style (concept → math → CLU connection → status tag → reading pointers):**
- **§5.3 — the linear-OU vs Maxwell–Jüttner no-go** *(the wave's most ML-legible lesson, as tasked)*. Why damping `p` instead of the velocity `∇_pT` breaks Gibbs the moment the kinetic term stops being quadratic; the three-line proof; the `T/(m₀c²)` table; both verifications; **the subtlety stated explicitly** (the Gibbs *measure* is relativity-insensitive; the *sampler* fails). Closes with the general ML statement: *any system that swaps in a non-quadratic kinetic/preconditioner term and keeps a linear damping+noise step has silently changed the distribution it samples.*
- **§8.6 — the lattice *is* the XY model; the KT transition *is* the memory transition.** Exact reduction; the corrected dictionary table (with both ⚠ corrections inline); the memory–vortex correspondence; **aliasing ⟷ vortices** table; the two honest nulls; the five prerequisites.
- **§4.9 — modulus vs Goldstone** (S5), **+ §4.9.1 the binding nomenclature-ledger entry** (Goldstone / modulus / pseudo-Goldstone / Noether charge / inertial `M` / spectral `μ` per Def-2 / decay constant `F` — with *"⚠ never the spin stiffness `ρ_s`"*). Records that **V2's broken-iso battery is an unremarked counterexample to the necessity direction** of the equivariance hypothesis.

**In-place updates (dated `> **Update (wave-13):**` blockquotes, per the pedagogical protocol):**
- **§4.4** — two blockquotes: the **O(n) label correction** (`n = dim(G/H) + 1`; the channel *count* was correct), and **"do the flat directions emerge? No"** (N46 numbers).
- **§5.2** — consequence 2 relabelled *"refuted in w3, then RE-OPENED in w13"*; two blockquotes: the **suspension of the w3 refutation** (both arms non-Gibbs), and **Prop-9 needs a kinetic-mode qualifier everywhere**; consequence 4's *"residual bias unquantified (F5 Open-3)"* struck → now a proven no-go.
- **§10.1** — 8 new dictionary rows (modulus-vs-Goldstone, O(n) label, Maxwell–Jüttner, XY/spin-stiffness, KT, `π₁`, rapidity, light-cone floor).
- **§10.2** — 6 new symbol rows (`J, ρ_s, T_KT, (w_x,w_y)`; `k_r, J₁, J₂`; **`ℓ_θ, Δ` with the "never quote `n₁/₂` without both" rule**; `γ_eff, T_local`; `ζ, F_Q², θ̇_max`; `T/(m₀c²)`).
- **§10.3 ledger** — **9 new rows** (relativistic no-go; `fdt` NaN; vault+refrigerator; designed-only coset + the unification; the non-discriminating exponents; the XY reduction; `spring_coupling` U(1) breaking; `GatedCoupling` force; R5; R7; R6). **Two rows corrected:** the *"charge drift 2.6"* row (**number retracted**, mechanism restated) and the *"MNIST imbalance ← per-mode `T_eff`"* row (**refuted → RE-OPENED, UNDER RE-TEST**).
- **§10.4** — reading pointers added for KT (Kosterlitz–Thouless; Nelson–Kosterlitz; AHNS; JKKN; Hasenbusch; Mermin–Wagner; Chaikin & Lubensky / Tong) and relativistic Langevin (Jüttner; **Dunkel & Hänggi**, *Relativistic Brownian motion*; Leimkuhler & Matthews).
- Masthead maintenance line extended to wave-13 with a full list of what this pass touched.

### 4. `.claude/outputs/philosophy-synthesis.md` (the ledger)
**⟲ protocol honoured: chapters untouched; one dated addendum appended.** The Hub had **not** written a w13 addendum (last was w9+w10), so this is a full curator addendum, not a `(curator supplement)`.

- **Chapter deltas:** **Ch. 5** (R5 turns "M is the budget allocator" from slogan into a **candidate conservation law**, now `TASKED(rb-bound-trained)` — *with the Newtonian null named as the boundary of the claim, not a control*); **Ch. 7** (the wave's defining lesson, written up as tasked: the five-step sequence, the rule **"instrument validity precedes inference"**, *why it will happen again*, and the cheap structural prophylactic); **Ch. 6** (Prop-9 → Prop-9′ no-go; verdict moves *[refuted] attribution* → *[attribution UNDER RE-TEST]*); **Ch. 1** (CM-16 splits; a self-conjecture cleanly falsified; N51); **Ch. 3** (the vault correction; **absorb-only is load-bearing**; `γ_φ` refined from "wrong tool" to *"wrong lever for erasure, right lever for protection"*).
- **New chapter candidate flagged, not opened:** Thread-10 / the lattice as a thermodynamic object — *recommendation: do not open Ch. 8 until the parameter-free 1-D control returns.*
- **Scorecard deltas:** rows 1, 2, 3, 6, 7 moved.
- **Gap-list:** the ⛔ N47 blocker as the new #1 code item; F5 re-blocked; N10 resolves at w14; **CM-16 must be split by the Hub**; the 13.9 sweep result; the two-wave-late retraction with its **recurrence vector** named; **six new scope guards (q)–(v)** for drafters, carrying the corrected wordings verbatim.
- **Positioning ripples:** *the referee stage does not catch this defect class* (w13's defects reproduce perfectly and are still wrong) — the prophylactics are **structural, not editorial**; the **two good process artifacts** written up as policy proposals (**pre-registration caught a wrong published number** and should become mandatory for measured-ratio acceptance criteria; **the theorist self-reported X4 and X5**, and the no-go stands on a proof rather than the harness he discarded); **V5 = GO with the honest scope**; and the recurring-deflation pattern gains **a sixth instance of a new species** — N46 is the *opposite* shape (*the physics buys exactly what we said, but only when we build it*).

---

## The "13.9×" sweep — result, precisely

**Executed. In the four docs I own it was a near-no-op, and that is a finding worth the Hub's attention.**

- **Before my edits, the string `13.9` / `13.88` appeared in ZERO of the four transfer docs.** Verified by grep across `.claude/**/*.md`: the **only** live occurrences were in **`handover_context.md`** (4 lines: §10 w12 entry ¶`t-lever`, and the w13 entry, which already self-flags *"Curator is sweeping it"*). **The task's premise that "it propagated into `future_work.md`" does not hold** — that file carried the *mechanism* ("γ_φ is the wrong lever… retro-explains the −24%") but never the number.
- **What I did instead of a find-replace:** corrected the **mechanism** wherever the vault is discussed, and **named the superseded number** as the ⟲ protocol requires. `future_work.md` now contains `13.9`/`13.88` **only inside the correction block and the footer**, in the form *"the rejected coupled-bath hypothesis, and now the measured value of the scalar-γ control."* Old text is struck through, never deleted (C-9).
- **The absorb-only design is recorded as load-bearing / DO-NOT-FIX** in `future_work.md` (V5 + Ch. 3 of the ledger) and in the primer's §10.3 row.
- ⚠ **Out-of-scope files I did NOT sweep and the Hub should:** `research_roadmap.md`, `brainstorm_log.md`, `claims_matrix.md`, `handover_context.md`. Grep confirms `brainstorm_log.md:273` and `handover_context.md:498` still carry the **`ρ_s ↔ F²`** dictionary error.

---

## What I deliberately left alone
- **`claims_matrix.md` (v1.8)** — task says CM-16/CM-17 already landed; do not re-edit. **Not touched.** I recorded the *outstanding* matrix action (CM-16's split) in three places rather than performing it.
- **`handover_context.md`** — Hub's. Not touched. Proposed updates below.
- **Chapters 1–7 of the ledger** — ⟲ protocol. Untouched; all deltas appended.
- **Waves 11–12 backfill** — **refused, deliberately.** The ledger jumps w9+w10 → w13 and the negatives registry was last swept at w10. Fabricating a retrospective addendum would violate the ⟲ protocol's spirit (chapters/addenda are dated records, not reconstructions) and would be worse than an honest gap. **Flagged in both files' headers.** The one exception is the N4 retraction, folded because the w12 handover *explicitly assigned it to `negative_results.md`*.
- **Verdicts** — I summarized and organized; I did not reinterpret. Where `v5-gate`'s prose band (`1.1e-3 … 3.0e-3`) differs from its own table (`1.060e-3 … 2.963e-3`) I transcribed the **table** and noted the rounding in the ledger. **No output↔handover disagreement was found to flag.**

---

## Docs-debt list (for the Hub)

1. **⚠ CM-16 must be formally split** — (a) designed-only latch face vs the universal unification + `T>0` law. Matrix v1.8 already anticipates it. **Until done, no draft may cite CM-16 as a single claim.** Recorded in all four docs; only the Hub can edit the matrix.
2. **⚠ Ledger continuity gap: no w11 or w12 addendum exists.** Hub call: commission a catch-up addendum, or accept the gap on the record. (The w11/w12 science reached the running log and the matrix, never the ledger.)
3. **⚠ Negatives-registry continuity gap: last swept at w10.** **N45's stated w11 resolution (`v1-certificate-payoff`) is unrecorded and its tier is still provisional** — it was due to discharge-or-promote at w11. Any other w11/w12 negatives are unregistered.
4. **⚠ The `ρ_s ↔ F²` and `n = dim(G/H)` errors survive in Hub-owned files** — `brainstorm_log.md:273`, `handover_context.md:498`. Also grep `research_roadmap.md` and `claims_matrix.md` for both, and for `13.9`/`13.88`.
5. **Recurrence vector, named:** the "2.6" retraction was delivered as a *downstream-reconciliation list inside one agent's report* (`f5-corrigendum` §"downstream"), with **no owner and no wave-boundary check** — and sat live for two waves. Recommend such lists become explicit curator tasks at the review that accepts them.
6. **Provenance-hygiene, unresolved from w10 (N43's lesson) and now recurring:** the wrong "13.9×" vault lived in a *source report's §7 prediction* and propagated into the running log. The flag-provenance rule governs **tables**; predictions and parenthetical asides remain uncovered. `v5-gate`'s pre-registration is the demonstrated cure — consider making it mandatory for measured-ratio acceptance criteria (proposed in the ledger addendum).
7. **Minor:** `v5-gate` §0 rounds the emergent `1−|λ_coset|` band to `1.1e-3 … 3.0e-3`; its §3.2 table gives `1.060e-3 … 2.963e-3`. Cosmetic; the table is transcribed everywhere. Flagged, not resolved.

## Git footprint
**None.** No tracked file created, modified, or deleted. All four edited files are gitignored under `.claude/`. Repo untouched.

## Proposed handover updates (for the Hub)
- **§10 / next-wave block:** *"Transfer docs current through w13. `negative_results.md` at **N51** (N46–N51 added; **N10 RE-OPENED — UNDER RE-TEST**, N18 amended, N4's retracted '2.6' dropped). `future_work.md`: Thread-10 rebuilt with the three corrections; §7bis boundaries folded; **V5 = GO**, `T_φ` HELD. `HEP_primers.md`: new §4.9 (modulus vs Goldstone + nomenclature ledger), §5.3 (the Maxwell–Jüttner no-go), §8.6 (XY/KT + memory–vortex); §4.4/§5.2 corrected; ledger +9 rows. `philosophy-synthesis.md`: ⟲ wave-13 addendum appended (chapters untouched)."*
- **Add to the standing scope-guard list (drafters, verbatim):** (q) cite **CM-16 split**; (r) never quote `n₁/₂` without `Δ` and `ℓ_θ/Δ`, never a designed-vs-emergent claim from raw exponents; (s) the vault is **107.77 ± 4.78×** — *"13.9×" is the rejected hypothesis and the measured scalar-γ control*; (t) `σ*` is **Newtonian-only**; (u) `ρ_s = J = 2κr*²` never `F²`, `n = dim(G/H)+1`, O(n) only for spherical cosets; (v) 2-D stable memory is **topological**, and `D_Θ = D_θ/N` is free in every dimension.
- **Two Hub actions I could not take:** split CM-16 in the matrix; sweep the four Hub-owned docs for `13.9`/`13.88` and the two dictionary errors (`brainstorm_log.md:273`, `handover_context.md:498` confirmed hits).
- **One standing-rule proposal, from the wave's own evidence:** make **pre-registration mandatory** for any task whose acceptance criterion is a measured ratio. It cost `v5-gate` nothing and converted a would-be fit into a decisive `8.11 ± 0.37` rejection — and it is what caught a wrong number this program had been quoting.
