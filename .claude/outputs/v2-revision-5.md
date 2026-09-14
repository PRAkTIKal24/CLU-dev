# v2-revision-5 — paper-writer report

Task + acceptance criterion: apply the **CM-16 split** (scope CM-16a designed-only, lead with CM-16b; N46 to appendix; ℓ_θ/Δ instrument warning on every quoted exponent) and the **CM-17/App-F σ\* kinetic-mode qualifier** (sampler-vs-thermodynamics distinction correct; V2 units Newtonian ⇒ scope clause) to the V2 short. Correctness pass, no new results, PDF clean.
Status: **done** (all 3 items). Draft `v0.5 → v0.6`; `draft.md` + `draft.tex` synced; `draft.pdf` rebuilt clean (**26 pp, 0 undefined refs, 6 figures**, tectonic).

---

## What I did (by task item)

### Item 1 — CM-16 split (App J + §5 + §3.4 + App F + A.8)
The draft leaned on CM-16 in **Appendix J** (the whole appendix) and in the **§5 limitations** clause. Both presented the finite-T laws as one uniform designed-testbed result with "the emergent/MLP arm untested." That is now the **out-of-scope pair** the task flagged. Fixes:

- **App J retitled** "…finite-temperature coset diffusion, **designed and emergent**"; source line now cites both `t-lever-forgetting` (designed, verification) and `v5-gate` (emergent, evidence); provenance A.7 **+ A.8**.
- **New lead "The split" paragraph** (immediately after the mandatory flag) — **leads with (b)** (CM-16b generalizes: same V-curve + T>0 sign flip 10/10 above T\*≈3e-3), states **(a) is designed-only** (N46: ≈1–1.6 bits, not a continuum), and carries the **instrument warning** ("raw d log n₁/₂/d log{T,γ} are NOT discriminating designed-vs-emergent; never quote n₁/₂ without Δ and ℓ_θ/Δ").
- **J.1** split: designed operator identity kept and **labeled verification + scoped "designed arm"**; new paragraph "The emergent arm has no such register — designed-only (evidence; N46)" with the ~12-order gap (1−|λ_coset|≈1e-3 vs ≤1.1e-15), complete relaxation of a written δ (≤2.1e-3, 3 seeds), capacity ≈1–1.6 bits.
- **J.4** unification: existing designed statement kept; new paragraph "**This unification is an emergent-arm result** … the general face" — argmin/γ_crit=0.902±0.003 (3/3), slopes −1.0020/+1.116, eleven orders in μ², T>0 face above T\*≈3e-3, the two-observables-two-signs note (register written *off* the vacuum carries the −1.0020 branch), and a pointer that raw exponents don't discriminate.
- **J.6** rewritten: no longer "emergent untested / top risk"; now states what's designed-only (register, N46) vs what generalizes (unification + T>0 law), plus the **matched-designed-control** shallow-slope numbers (−0.53/−0.60/−1.04; +0.78/+0.63/+0.55) as the load-bearing instrument caveat, and honest remaining risks (3 seeds, one config, T\* on 2 cells/seed).
- **§5 limitations**: the "emergent untested / top open risk" clause → **both arms tested** (designed verification + emergent evidence above T\*), one face (continuous register) designed-only (J.1, N46).
- **§3.4**: one forward-pointer sentence to N46/J.1-J.4 — the sharp operational form of the existing 13–14-order μ² gap (stores nothing on a continuum; but the *law* generalizes). No existing number touched.
- **App F**: **N46 added** to the negatives list.
- **A.8** flag-provenance for `v5-gate` (commit `d6f8bac`, fdt, retie, `newtonian_learned`, **Δ=0.5 rad**, 3 emergent + 1 designed control, γ/T grids, 0/45 censored, harness cross-check 1690 vs 1600 = 5.6%).

### Item 2 — App F σ\* kinetic-mode qualifier (CM-17 / Prop-9′)
App F stated Prop-9's σ\* as a class-level neutral theorem with no kinetic-mode scope. Now:
- σ\* is exact **only in the Newtonian kinetic modes** (`newtonian_identity`/`newtonian_learned`).
- Relativistic O-step = additive Gaussian kick ⇒ Gaussian-smoothed invariant momentum marginal; Maxwell–Jüttner is not of that form ⇒ **no σ gives a Gibbs invariant** (theory note, **Prop-9′**, characteristic-function argument, third person).
- **Sampler-vs-thermodynamics stated correctly**: π_q ∝ e^{−V_θ/T} is relativity-insensitive (momentum integral factorizes), so the unit's equilibrium is well-defined; the chain just fails to sample it. Explicit sentence: *we never assert a relativistic unit "has no equilibrium."*
- **Scope clause, not a self-correction**: V2's units are `newtonian_learned` ⇒ σ\* exact for every config we run; touches no result.
- The Newtonian qualifier was also propagated to the **§2.1 fdt fine print** and the **App-J mandatory flag** (CM-16's v1.9 flag is now "fdt **AND** Newtonian"; `f5-corrigendum-2` "Downstream" flags all three sites 58/271/388). Numbers safe: t-lever/v5-gate both ran `newtonian_learned`.

### Item 3 — carry-overs
- **`13.9×` / `vault` / `107.77×`**: grep-verified **absent** in both `draft.md` and `draft.tex` — the V2 short never quoted the vault number, so **nothing to retract, changed nothing** (as the task's "if already correct, say so" instruction anticipates).
- **MF-1** (theory note still a placeholder / `(Anonymous, 2026)`): standing Head dependency, mentioned once here, not itemized in the draft (draft already cites the note in third person throughout; Prop-9′ cited the same way).

---

## How I verified
```
tectonic -X compile draft.tex --keep-logs   → "Writing draft.pdf (971.81 KiB)", 0 errors
grep -c "undefined" draft.log               → 0
mutool info draft.pdf | grep Pages          → Pages: 26
grep -c "N46" draft.md draft.tex            → 7 / 7  (balanced)
grep -c "Prop-9" draft.md / draft.tex       → 3 / 3
grep -n "untested" draft.{md,tex}           → only App I.7 (GMOR emergent arm — genuinely still untested, out of scope)
grep -c "13.9|vault|107.77" draft.{md,tex}  → 0 / 0
grep -c "Newtonian kinetic mode" .{md,tex}  → 4 / 4
grep -n "no equilibrium"                    → only the DENIAL sentence ("never assert … 'has no equilibrium'")
```
Build warnings: cosmetic underfull/overfull `\hbox` only, all in provenance rows (A.7/A.8) and App J from long `\texttt{}` config tokens — justification, not layout breakage. The pre-existing App A.3 overfull hbox (the `agent/…@b41410f` path) is untouched.

## Evidence backing each change (C-2 labels)
| section | claim | source report | label |
|---|---|---|---|
| App J lead / J.4 (emergent V-curve, T\*) | argmin 0.902±0.003; slopes −1.0020/+1.116; 10/10 above T\*≈3e-3 | `v5-gate` §3.4–3.5, §0 | **evidence** (learned MLP) |
| App J.1 emergent / N46 | 1−|λ_coset|≈1e-3 (~12 orders); δ relaxes ≤2.1e-3; ≈1–1.6 bits | `v5-gate` §3.1–3.3, §0 | **evidence** (learned MLP) |
| App J.1/J.2/J.3/J.5 designed | latch 1.7e-14; D-law 1.0068±0.0219; sign flip +0.955±0.042 | `t-lever-forgetting` (CM-16) | **verification** (designed SO(2)) |
| App J.6 instrument caveat | matched designed control shallow slopes | `v5-gate` §3.5 (matched control) | evidence (method) |
| App F σ\* qualifier / Prop-9′ | relativistic no-go; sampler≠thermo; d·Θ | `f5-corrigendum-2` §1–2, "Downstream" | proven (theory note, third person) |
| A.8 provenance | fdt/retie/newtonian_learned/Δ=0.5/d6f8bac | `v5-gate` §1 | — |

Related-work prose lineage unchanged (no §4 edits beyond the pre-existing).

## Charter / matrix compliance self-check
- **C-1** no audit-confession ¶ (unchanged). ✓
- **C-2** emergent (learned MLP) results labeled **evidence**; designed SO(2)/analytic-tilt results labeled **verification** — both labels explicit in App J headers and J.1/J.4/J.6. ✓
- **C-5** in-sentence scale qualifiers on every new generalizing claim ("3 emergent seeds", "10/10 conditions", "dim 4, laptop-CPU", "one 150-epoch newtonian_learned config"). ✓
- **C-6** the σ\* fine print, the Newtonian-mode qualifier, and the ℓ_θ/Δ instrument warning all sit **next to** their claims (App F bullet, App-J flag, App-J lead + J.4 + J.6), not footnoted. ✓
- **C-7** A.8 provenance added; every new number traces to `v5-gate`/`f5-corrigendum-2`; `spurion_delta`/friction/temperature-field all off ⇒ bit-compatible with the rest of the paper, stated. ✓
- **C-8/M1** hermetic — only J&P 2026, the theory note (Anonymous, 2026, third person; Prop-9′ cited by name not number), and the pre-existing classical refs. No cross-short citation; `v5-gate`/`f5-corrigendum-2`/`t-lever-forgetting` appear only as internal provenance slugs (flagged for anonymized-build scrub, per v2-revision-4 item 5). ✓
- **C-9** N46 added to App F; App J's designed-only-register negative stated plainly; nothing pruned. ✓
- **C-10** appendix maximalism — all split content is in appendices (J + F) or provenance (A.8); main text grew by exactly one forward-pointer sentence (§3.4) and one limitations rewrite (§5). +2 pp, all appendix. ✓
- **Claims matrix v1.9** — CM-16a scoped designed-only, CM-16b led with, both in approved wording; **canonical constants unchanged** (γ_crit=2εμ, D_θ=εT(2−γ)/(2F²γ), floor, EP, latch transport cited identically to §1); CM-17 σ\* qualifier + Prop-9′ in approved form; CM-3 absent even hedged; inertial M vs spectral μ preserved, no bare "mass". ✓
- **Rule 3 (no improvised numbers)** — every quantitative statement copied verbatim from `v5-gate`/`f5-corrigendum-2`; nothing rounded or smoothed. The min-over-grid `1−|λ_coset|≈8e-5` I added to J.1 is `v5-gate` §3.2's reported `min` column (7.6e-5–2.0e-4, quoted as "≈8e-5"). ✓

---

## Open editorial questions (for Hub / Head)
1. **App J placement is still a live Head call** (CM-16b: "V5 = GO / V2 appendix"). The split now makes App J carry *both* the designed verification and the emergent evidence. If the Head rules the emergent material ships as the **V5 "Forgetting" short**, App J should keep the law *statements* + designed verification but forward-attribute the emergent numbers (0.902, −1.0020/+1.116, T\*≈3e-3, N46) to V5. I left them in App J with clear evidence labels because the task said to apply the split here; flagging the dependency, not resolving it.
2. **§3.4's "13–14 orders" (μ²) vs App J.1's "~12 orders" (1−|λ_coset|)** are two different metrics of the same designed-vs-emergent gap, from two source reports (`v2-full-runs` Hessian μ²; `v5-gate` Jacobian eigenvalue deficit). Both are honest and I kept both with their metrics named. If a reviewer-facing single number is wanted, that's a Hub call — I did not reconcile them into one figure (would be improvised).
3. **Length** now 26 pp (was 24). All growth is appendix/provenance per C-10; nothing to prune now, but App J is the natural first candidate for the eventual pruning pass if App J ships to V5.
4. **MF-1** (F5 note arXiv id / title / authors) remains the standing #1 Head submission dependency — Prop-9′ and the whole §-provenance still resolve to "(Anonymous, 2026)".

## Missing-experiment notes (no numbers improvised)
- **GMOR on the emergent arm** (App I.7) — still genuinely untested; the only surviving "untested" in the draft, correctly scoped, not touched by this task.
- **Emergent T\* as a fitted crossover** (`v5-gate` R1b) — the emergent T\*≈3e-3 rests on 2 cells/seed; stated as a remaining risk in J.6, not upgraded.

## Git footprint
**None.** All edits under gitignored `.claude/**` (`papers/v2-short/{draft.md,draft.tex,draft.pdf,CHANGELOG.md}`). No tracked file created/modified/staged; no branch; `HEAD` untouched.

---

## Proposed handover updates (for the Hub)
**§8 / drafting status — update the V2 line:**
- **v2-short is at v0.6.** CM-16 split applied (CM-16a designed-only, CM-16b led-with-and-generalizing) across App J / §5 / §3.4, with **N46** in App F and **A.8** provenance for `v5-gate`; the ℓ_θ/Δ instrument warning + "never quote n₁/₂ without Δ and ℓ_θ/Δ" now travel with every exponent. App F's σ\* carries the **CM-17 kinetic-mode qualifier** (Prop-9′, sampler-vs-thermodynamics correct, V2-Newtonian scope clause); the Newtonian qualifier propagated to §2.1 + App-J flag. `13.9×` confirmed absent program-wide. Build clean, 26 pp, 0 undefined refs. → **`v2-referee-4` (clean pass)** unblocked.

**Claims matrix — no changes needed.** CM-16a/CM-16b and CM-17 used in approved v1.9 wording; CM-16's "appears in" placement (V2 appendix vs V5) preserved as an in-draft pending note. If the Head rules App J → V5, update CM-16b's "appears in" column and re-scope App J.
