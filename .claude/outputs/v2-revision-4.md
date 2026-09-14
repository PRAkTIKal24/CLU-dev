# v2-revision-4 — paper-writer report

Task + acceptance criterion: fold the symmetry-realization taxonomy into the V2 short as a **framing overlay** (all four obligations, all-or-nothing), relabel §3.5 as symmetry restoration, land the S5 modulus/Mo differentiator, add the GMOR-proper appendix, close the corrigendum's three flagged draft sites + App-C, rebuild PDFs.
Status: **done** (all 7 items). Draft at **v0.5**; `draft.md` + `draft.tex` synced; `draft.pdf` rebuilt clean (24 pp, 0 undefined refs, 6 figures).

---

## What I did (by task item)

**1. Taxonomy adopted as a framing overlay — NOT a §3 restructure.** The mode-mass budget remains the quantitative spine untouched.
- Abstract: one two-axis clause ("symmetry realization sets μ²; the map's (ε,γ,T) sets what μ² buys — all three realizations exhibited by trained models here").
- §1: new "**Framing: two axes, not one**" paragraph; ends by flagging that the result the literature would call a training pathology (§3.5) *is* the unbroken cell.
- **New §2.1 "Symmetry realization: the axis that sets μ²"** — three-realization table (Wigner–Weyl / Nambu–Goldstone / pseudo-Goldstone), each mapped to a *this-paper* arm.

**The four obligations, all met in §2.1** (theorist's all-or-nothing condition):
| # | obligation | where / how |
|---|---|---|
| (i) | SSB for a finite-dim deterministic system | ¶ "What SSB means…": classical/tree-level Goldstone theorem **only** — minimiser not G-invariant ⇒ orbit of minimisers ⇒ ∇²V annihilates orbit tangents (dim(G/H) zero eigenvalues). Explicitly disclaims thermodynamic limit, phase transition, ħ. Pre-empts the "no SSB in a 0-dim system" referee. |
| (ii) | Coleman/Mermin–Wagner honest | ¶ "Coleman/Mermin–Wagner, stated honestly": **∞ half-life is a T=0 statement**; at T>0 the coset diffuses, D_θ = εT(2−γ)/(2F²γ), finite computable lifetime, *diffusive register, not long-range order*. Turned into an **asset** (we have the law: CM-16, App J). **Mandatory `langevin_noise="fdt"` fine print inline**, plus "all §3 results are T=0, unaffected". |
| (iii) | NO loop-level ChPT | ¶ "What we do not import": no loops, no chiral logs, no LEC *running*, no anomalies/WZW; only expansions used are ε² (discretization), T (thermal), μ²/μ_rad² (chiral ratio). Repeated in App I.7. |
| (iv) | erosion relabel | item 2 below. |

Also in §2.1: the ¶ "**Symmetry protects the current; flatness protects the register**" (modulus statement, forward-ref to §3.4/§4).

**2. §3.5 relabeled.** New title: "**Symmetry restoration under training, and the anchored recipe that prevents it**". r\* named as *the* order parameter Σ; CD drives Nambu–Goldstone → Wigner–Weyl; **the previously-unnamed degenerate μ² doublet (≈1.0) at the collapsed vacuum is now named as Schur's lemma at a G-invariant vacuum** (`v2-full-runs` Finding 0). Narrative arc sentence added: *architecture breaks the symmetry / the objective restores it / the anchor keeps it broken — with the axis-2 budget intact on the far side.* Honest caveat inline: "restoration" used in the exact classical sense of §2.1; **no thermodynamic phase transition claimed**. Section relabeled **evidence-grade** (training dynamics on learned potentials) per C-2.

**3. S5 modulus/Mo differentiator.** New §4 ¶ "**Sufficient, not necessary: equivariance and neutral memory**".
- Mo's theorem stated as a **sufficiency** statement; I explicitly do *not* attribute a necessity claim to him ("which his theorem does not assert but which is easy to read into it").
- Latch = **modulus** of V_θ; Sylvester ⇒ μ²=0 under any inertial anisotropy (cited as *theory note*, kinetic-spurion blindness).
- **Measured counterexample**: broken-iso battery, non-equivariant map (E^T_eq = 0.0016–0.108, 3 split levels, dim 4), coset exactly flat (μ²_ang ~1e-15), latches (n₁/₂=∞, write-freeze 0).
- Payoff sentence: symmetry buys the conserved **write current**; flatness buys the **register**; the diagnostic that separates them is the charge law, not the Hessian.
- Also seeded in §2.1, §3.4(iii), App C.4, §5 horizon.

**4 + 6. New Appendix I — "GMOR proper: a condensate-resolving spurion on trained checkpoints"** (CM-15 verbatim; labeled **verification**, designed testbed + analytic spurion).
- I.1 why (angular tilt is radius-independent ⇒ measures only μ²F²=δn², "Σ" degenerates to n²); ChPT dictionary table.
- I.2 the identity + full 10-row δ table; **max|μ²F²−δΣ| = 1.33e-15** over 80 (ckpt,δ) pairs.
- I.3 Σ measured three ways (geom; Hellmann–Feynman exact 0.0; envelope-theorem FD 9.2e-11).
- I.4 Σ runs +16.1…+210.1% (linear spurion) vs ≤2.22e-15 (angular tilt) on the *same* checkpoints.
- I.5 LEC resonance-saturated: ratio **0.99999607** (δ≤1e-5), d(ratio)/dx = −1.0504; **δ=0.3 excluded from the NLO claim** (x=0.68, Σ ran +210%).
- I.6 direction independence 6.21e-15. I.7 honest scope (tree-level; probe-only; one architecture; emergent arm untested ⇒ open falsifiable δ→δ+δ_eff).
- **Figure I1** = `figs/fig_gmor_condensate.png` (copied from `outputs/f1-gmor-condensate/`), wired via `\includegraphics`.
- **Nomenclature adopted everywhere**: **F = √(M_ch)·r\*** = decay constant; **Σ = r\*** = condensate/order parameter. A new §3.1 ¶ states this and points to App I.
- **C-6 fine print sits next to the claim**: absolute machine precision at every δ; relative exactness (≤2.7e-14) only for δ≥1e-2; the small-δ relative residual is the **ε/δ autodiff-Hessian roundoff floor**, not the law. Draft explicitly forbids quoting "2.2e-16 relative" for this experiment, and notes the same floor applies to every μ² probe in the paper (all others use δ≥1e-4, floor ~1e-12 ⇒ prior relative agreements sound).

**5. App-C fix + the corrigendum's three flagged sites.** All closed; grep-verified.
| flagged site | before | after |
|---|---|---|
| `draft.md:107` (§3.4) | "Noether-charge **drift** scales linearly with the split" | "**bounded oscillation** — not a secular drift — amplitude linear in the split at small split"; on-orbit envelope √(2E)r\*(√M_max−√M_min), half-revolution period; **off-orbit only boundedness** \|Q\|≤sup\|q\|sup\|p\| (coercivity). Measured 5.4/1.6/0.082e-2 relabeled **amplitudes**. |
| `draft.md:141` (§4 Di Bernardo) | "the retention table and the kinetic-isotropy constraint" | "…and the kinetic-isotropy (Schur) condition **as the price of an equivariant write current**" |
| `draft.md:155` (§5 Horizon) | "the price of an **equivariant channel**" | "the price of an equivariant **write current**, not of the register itself" (+ "latches as exact flat directions (**moduli of V_θ**)") |

- **"≈0.7‖[M,X]‖" deleted** from both md and tex (A/‖[M,X]‖ is *not* constant). Replaced with a qualitative statement + "no proportionality constant should be quoted as a law." **I did not compute a replacement coefficient** (would have been derived arithmetic on reported numbers — flagged rather than improvised).
- **Retracted "2.6" never appears** (verified 0 occurrences of any charge-drift 2.6).
- App C restructured into **C.1 negative / C.2 invisibility (with the Sylvester congruence statement) / C.3 bounded oscillation + two reporting cautions / C.4 attribution**. C.3 carries the reproducibility caution: off-orbit chaotic-rosette running-suprema are window- and float-order-dependent ⇒ *we report amplitudes and boundedness, never a "drift rate."*
- **Corrected design rule** stated in three places (§3.4(i), App C.3(ii), App F/N4): *an anisotropic channel still latches with infinite half-life; tie masses for a clean θ-independent write current (and a degenerate pNG multiplet at dim(G/H)≥2), **not** to save the register; to detect kinetic symmetry breaking measure the charge law — Hessian and latch are provably blind.*
- **N4 registry line rewritten** (App F) accordingly ("the register survives it exactly… bounded oscillation… never a secular drift").

**Bonus (obligation-(ii) driven): new Appendix J — "The T=0 face of the budget cube"** (CM-16 verbatim). Required to discharge the Coleman obligation honestly ("we have the law"). Contents: J.1 |λ_flat|−1 ≤1.7e-14 ∀γ∈[0.002,0.5], drift ≤4.9e-12 rad/200k steps (30/30 cells); J.2 D-law 1.0068±0.0219 over 25 cells; J.3 sign flip 10/10 (+0.955±0.042; 3.77±0.23× for γ 0.05→0.2); **J.4 the unification** (massive-mode n₁/₂(γ) non-monotone, min at γ_crit=2εμ; flat mode = μ→0 corner where γ_crit→0 ⇒ permanently overdamped — *one damping-optimum curve at two values of μ, not two laws*; ties directly to §3.1's measured γ\*≈2εμ minimum); J.5 thermal persistence length ℓ_θ qualifier; J.6 honest scope. **Mandatory fdt flag boxed at the top of the appendix.** Placement explicitly marked **pending Head call** (CM-16 says V5-candidate / V2-appendix).

**7. PDFs.** `v2-short/draft.pdf` rebuilt with tectonic: **0 undefined refs, 24 pp, 955.7 KiB, 6 figures.** `f5-note/f5-note.pdf` is **already current** (pdf mtime 20:15:00 > tex 20:14:44; rebuilt by `f5-corrigendum` at its v0.4). I did **not** touch `papers/f5-note/` — out of my edit scope. If "both PDFs" meant something else, say the word.

---

## How I verified

```
tectonic -X compile draft.tex        → "Writing draft.pdf (955.70 KiB)", 0 errors
tectonic -X compile draft.tex --keep-logs; grep -c "Reference.*undefined" draft.log   → 0
python3 (zlib scan of /Type/Page)    → pages: 24
grep -c "drift scales linearly"      → draft.md:0  draft.tex:0
grep -c "equivariant channel"        → draft.md:0  draft.tex:0
grep -c "0.7\|[M,X]\|"               → draft.md:0  draft.tex:0
grep -n "drift.*2\.6\|2\.6.*drift"   → (no matches)
grep -c "bounded oscillation"        → draft.md:2  draft.tex:2
grep -c "fdt"                        → draft.md:3  draft.tex:3
```
Build warnings: one **pre-existing** cosmetic overfull hbox (34.3 pt) in App A.3 — the unbreakable `agent/experiment-engineer/minus-the-physics@b41410f` path token, untouched by this pass. The §2.1 taxonomy table overfull that my first build introduced was fixed with `\resizebox`.

## Evidence backing each new/changed section

| section | claim | source report | label (C-2) |
|---|---|---|---|
| §2.1 table, ¶ SSB | three realizations; classical Goldstone thm | `v2-symmetry-deepdive` §1, §3.1, §3.3 | framing |
| §2.1 ¶ Coleman | D_θ=εT(2−γ)/(2F²γ); 3.77±0.23× | `v2-symmetry-deepdive` S8 (theory) + `t-lever-forgetting` (CM-16, measured) | verification |
| §2.1 ¶ modulus | Sylvester ⇒ μ²=0 ∀M | `f5-corrigendum` Prop-17 (`prop:kinblind`) | proven (theory note) |
| §3.1 ¶ nomenclature | F²=M_ch r\*²; tilt fixes Σ to 2.2e-15 | `f1-gmor-condensate` §3; `f5-corrigendum` §3 | verification |
| §3.4 isotropization ¶ | bounded oscillation; amplitudes 5.4/1.6/0.082e-2; E^T_eq 0.0016–0.108 | `v2-full-runs` item 5 (numbers) + `f5-corrigendum` §1–2 (interpretation) | evidence |
| §3.5 | r\*→0 restoration; degenerate μ² pair ≈1.0 | `v2-full-runs` Finding 0; `sleep-erosion-study`; `anchor-robustness` | evidence |
| §4 ¶ sufficient-not-necessary | non-equivariant latching map | `v2-symmetry-deepdive` S5 §4.1(c) + `v2-full-runs` item 5 | evidence |
| App C | all of the above + reproducibility caution | `f5-corrigendum` §2 (check-(g) forensics) | evidence + proven |
| App I | μ²F²=δΣ ≤1.33e-15; LEC 0.99999607 | `f1-gmor-condensate` (CM-15) | **verification** |
| App J | latch/D-law/sign-flip/unification/ℓ_θ | `t-lever-forgetting` (CM-16) | **verification** |
| App A.6/A.7 | flag provenance | `f1-gmor-condensate` §Flag provenance; `t-lever-forgetting` §1.1 | — |

Related-work prose lineage unchanged (`mo-deep-read` §4, `di-bernardo-skim`); the new §4 ¶ derives from `v2-symmetry-deepdive` §4.1(c)/S5.

## Charter compliance self-check
- **C-1** no audit-confession ¶ anywhere (unchanged from v0.2 removal). ✓
- **C-2** App I / App J / §3.1 = **verification** (designed testbed, analytic tilt/spurion, even though on trained checkpoints); §3.2/§3.3/§3.4/§3.5 = **evidence**. Labels stated in each section's italic header. ✓
- **C-5** in-sentence scale qualifiers on every new generalizing claim ("8 trained SO(2) checkpoints × 10 δ", "dim 4, 5 seeds, laptop-CPU", "3 mass-split levels", "f64 probe, probe-only"). ✓
- **C-6** GMOR precision fine print + fdt flag sit **next to** their claims (§2.1, §3.1, I.2, J-header), not footnoted. ✓
- **C-7** A.6/A.7 provenance rows added; spurion defaults 0 ⇒ bit-compatibility with all other results stated explicitly (no cross-section contradiction constructible). ✓
- **C-8/M1** hermetic: only Jawahar & Pierini (2026), the theory note (Anonymous, 2026, third person), Mo 2026, Di Bernardo 2025, Keller 2025, Iqbal et al. 2026, and the classical refs. No cross-short citations. Prop-17 cited by *name* ("kinetic-spurion blindness"), not number — the `.md`/`.tex` numbering of F5 differs (5′ vs auto-3), so a number would have been a live footgun. ✓
- **C-9** N4 negative kept and *strengthened*; App J's negative scope (emergent arm untested) stated. ✓
- **C-10** appendix maximalism: nothing pruned; two new full appendices; 24 pp. ✓
- **Claims matrix**: CM-15 and CM-16 used in approved wording; CM-3 absent; canonical constants (latch transport, (1−γ)ⁿ, μ⁻², floor, h\*, γ\*≈2εμ, det J) cited identically to matrix §1. Inertial M vs spectral μ preserved; no bare "mass". ✓
- **Rule 3 (no improvised numbers)**: every number traces. Where the corrigendum's "A/‖[M,X]‖ is not constant" would have required me to divide reported numbers to produce a coefficient, **I stated it qualitatively instead** and flagged it below.

---

## Open editorial questions (for Hub / Head)

1. **App J placement is a live Head call** (CM-16: "V5-candidate / V2 appendix"). I included it because obligation (ii) cannot be discharged honestly without the law, and marked the placement pending in the appendix's own header. **If the Head rules V5**, §2.1's Coleman ¶ must keep the law *statement* but its verification numbers (3.77×, 1.0068±0.0219) would need to move or be attributed forward — as written they are cited to App J. Cheapest alternative: keep a 4-line App J stub with the law + fdt flag and bank the rest.
2. **Length.** 24 pp against a 4–5 pp workshop main text. Main text is now ~6.5 pp (§2.1 added ≈0.9 pp). The taxonomy overlay is the only *main-text* growth; everything else is appendix. Flagging for the dedicated pruning pass — §2.1's "what we do not import" ¶ and the realization table are the two most compressible items if the Head wants main text back under 6.
3. **Derived-arithmetic abstention (rule 3).** The corrigendum says A/‖[M,X]‖ is not constant. The three reported amplitudes ÷ their splits give 0.70/0.73/0.75 (a ~7% drift over a 70× split range) — this would be a *nice* sentence, but it is arithmetic I performed on source numbers, not a measured quantity. **I did not put it in the draft.** If the Hub wants it, it should come from a one-line analyst confirmation (the corrigendum's own toy shows 0.096→0.076→0.062, a *much* larger drift, on a different M-grid — so the two are not obviously the same statement, which is exactly why I abstained).
4. **`v2-full-runs` Finding 0's "channel μ² pair degenerate (~1.0)"** is now load-bearing in §3.5 (it is the Wigner–Weyl evidence). It is currently a parenthetical in a figure caption in the source report. Worth a one-line analyst re-read to confirm the pair is degenerate *to a stated tolerance* — right now the draft says "≈1.0", which is honest but weak. The theorist's F-3 (splitting |μ₁²−μ₂²|/μ̄² on the collapsed ckpts) would upgrade this cheaply; note that on the **designed** arm it is architecturally guaranteed (a pipeline check), so the interesting version is the **emergent** arm.
5. **Internal report names appear in App A provenance rows** (`f1-gmor-condensate`, `t-lever-forgetting`, etc.) — consistent with the pre-existing A.1–A.5 convention, but these are program-internal slugs. Confirm they get scrubbed/renamed at anonymized-build time.
6. **"Rebuild both PDFs"** — I read this as v2-short/draft.pdf (rebuilt) + f5-note.pdf (verified already current from the corrigendum, not touched). Confirm.

## Missing-experiment notes (no numbers improvised)
- **GMOR on the emergent arm** (App I.7): predicted μ²F² = (δ+δ_eff)Σ. Stated as an open falsifiable, not a result.
- **App J on the emergent arm** (J.6): the single-architecture scope is named as "the top open risk of that appendix" in both §5 and J.6.
- **Schur splitting at the collapsed vacuum** (item 4 above): F-3, unrun.

## Git footprint
**None.** All edits are under gitignored `.claude/**` (`papers/v2-short/{draft.md,draft.tex,draft.pdf,CHANGELOG.md,figs/fig_gmor_condensate.png}`). No tracked file created, modified, or staged; no branch created; `HEAD` untouched. One binary added under `.claude/` (the figure, copied from `.claude/outputs/f1-gmor-condensate/`).

---

## Proposed handover updates (for the Hub)

**§8 / drafting status — update the V2 line:**
- **v2-short is at v0.5.** Taxonomy overlay adopted **in full** (four obligations discharged in §2.1); §3.5 relabeled a symmetry-restoration transition; S5 modulus/Mo differentiator landed in §4; **Appendix I (GMOR proper, CM-15)** and **Appendix J (finite-T coset diffusion, CM-16)** added; all three `f5-corrigendum` draft sites + App C closed; retracted "2.6" verified absent. Build clean, 24 pp, 0 undefined refs. → **`v2-referee-3` (clean pass)** is unblocked.

**§7 — one item to close, one to open:**
- **7.17 → ✅ CLOSED in the V2 short.** "drift → bounded oscillation" applied at `draft.md:107` + App C, with the on-orbit/off-orbit distinction and the reproducibility caution; the "≈0.7‖[M,X]‖" proportionality constant is deleted, not re-derived. The `equivariant channel` → `equivariant write current` rewording is applied at **both** flagged sites and grep-confirms 0 remaining occurrences.
- **NEW (drafting-hygiene):** the **ε/δ autodiff-Hessian precision floor** is now stated in the V2 short (App I.2) as applying to *every* μ² probe in the paper. Any future short quoting a μ² relative agreement at δ ≲ 1e-6 must carry it. `f1-gmor-condensate` proposed this as a §7 Known-Issues note; the V2 short has now adopted it in print, so the note should land in the handover too.

**Claims matrix — no changes needed.** CM-15 and CM-16 are used in their approved wordings; CM-16's "Head call" placement is preserved as an in-draft pending note rather than silently resolved. If the Head rules on App J placement, CM-16's "appears in" column should be updated.

**Head decisions requested:** (a) App J placement (V2 appendix vs V5 short); (b) main-text length tolerance for §2.1 ahead of the pruning pass; (c) confirm the "both PDFs" reading.
