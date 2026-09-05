# v2-short-draft — paper-writer report

**Task + acceptance criterion:** draft the V2 workshop short (the "CLU" naming debut) + convert the F5 arXiv note to LaTeX; charter C-1…C-10 + claims-matrix compliant; every number traces; appendix-maximal.
**Status: done.** Both deliverables written, both LaTeX sources build to PDF locally with `tectonic` (present at `/opt/homebrew/bin/tectonic`; no pdflatex/xelatex/latexmk on this machine).

## What I wrote
- `.claude/papers/v2-short/draft.md` — canonical V2 short (main §1–5 + Appendices A–F, fully written).
- `.claude/papers/v2-short/draft.tex` — LaTeX build (article class, standard packages). **Builds:** `tectonic draft.tex` → `draft.pdf` (exit 0; 2 residual ≤14 pt overfull hboxes in prose, cosmetic).
- `.claude/papers/v2-short/CHANGELOG.md` — v0.1 line.
- `.claude/papers/f5-note/f5-note.tex` — arXiv-ready LaTeX of `f5-arxiv-note.md`. **Builds:** `tectonic f5-note.tex` → `f5-note.pdf` (exit 0, **0 overfull**). Cor 2/3 → appendix; open Q 1/3/5 + Cor-3 footnote as `[TODO-HEAD]` margin notes.
- `.claude/papers/f5-note/CHANGELOG.md` — v0.1 line.

Repo: no tracked files touched (all under gitignored `.claude/papers/`). No git footprint.

## How I verified the builds
`cd .claude/papers/v2-short && tectonic draft.tex` → `draft.pdf` 126 KB, exit 0. `cd .claude/papers/f5-note && tectonic f5-note.tex` → `f5-note.pdf` 172 KB, exit 0. Wide tables wrapped in `\resizebox`; `\sloppy` added to relieve prose overfull. No content pseudo-verified — the numbers are transcribed verbatim from the source reports (never rounded/smoothed), each cell citing its report.

---

## V2 short — section-by-section evidence map (claim → source report → CM-x / charter)

| §/App | claim (headline) | source report + locus | CM / charter |
|---|---|---|---|
| §1 audit ¶ | 3 audit findings (dead Lyapunov reg; non-Gibbs sampler; M=I-only cap) | negatives N17/N18/N22; f5-arxiv-note Cor 3 / Prop 9 / Prop 12 | **C-1** (canonical ¶ below) |
| §1 naming | "the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)" | HEP_primers §0.1 continuity line; task Thread-6 | naming rule; C-8 |
| §3.1 | GMOR exact ($\mu^2$ ratio 1.000000±5e-12, 4.5 dec); slope −0.985 (pred −1); floor 27.03; C2 bifurcation ≤3.2×; latch transport ≤1%; C1 min 5/5; EP slope 0.5165 | `v2-full-runs` items 1/3/7 | **verification** (C-2: designed testbed, analytic tilts) |
| §3.2 **headline** | Mo ratio 1.012–1.029 overdamped (his median 1.013) → 2.20±0.16 EP → 0.31±0.01 underdamped; corr 0.9987 overdamped | `v2-full-runs` item 2; `mo-deep-read` §2–4 | **evidence** (C-2: Mo head-to-head on trained models); C-3 (ML-first lead) |
| §3.3 | triad absent in baselines: coRNN/LEM/LSTM 5.6/56/69 map-steps, drift >1.2 rad, LSTM 69→2 perturbed; CLU-emergent 263, 0.35 rad, +12–28%; CLU-designed ∞, coset frozen; **honest input-driven-RMSE gap** | `v2-prefreeze-baselines` items 1/2 | **CM-4 (verbatim)**; C-2 evidence; C-5 scope in-sentence |
| §3.4 | designed vs emergent 13–14 orders in μ²; E_eq^V 15 orders; +12–28% priced; isotropization NO (N4); price of physics: twin ~15× MSE, contraction-forbidden not causal-cap, **short-horizon** | `v2-full-runs` items 4/5; `minus-the-physics` Part A | **CM-1 (verbatim scope)**; C-2 (constitutive-vs-kinematic foregrounded) |
| §3.5 | erosion inverts designed vacuum (116/442/959 ep by freq, chain-length-independent); anchor cure λ=10 depth +0.069/gap +0.244; envelope λ=100 bulletproof 5/5; demarcation erosion∝flatness; anchor ⊥ volume-conservation | `v2-full-runs` Finding 0; `sleep-erosion-study`; `anchor-robustness` | **CM-6** (novelty PENDING scout); C-9 |
| §4 related | Mo (kinematics vs constitutive); Di Bernardo/Keller (geometry, no time — guard-rail applied); Iqbal/Welling (SSB sibling) | `mo-deep-read` §4; `di-bernardo-skim` "Draft prose"; f5-note §9 | C-8 (hermetic; only J&P + theory note + external published) |
| §5 horizon | EFT-of-memory = horizon paragraph only | — | **C-4** (no promissory notes in lead) |
| App A | flag-provenance tables per result group | all source reports' provenance tables | **C-7** |
| App B | erosion law+cure full; novelty **pending scout** | `sleep-erosion-study`, `anchor-robustness` | C-9/C-10; CM-6 |
| App C | isotropization negative + kinetic-breaking invisibility (charge-law discipline) | `v2-full-runs` item 5 | N4; C-9 |
| App D | +12–28% bias decomposition (2/3 settle-offset, s43 anharmonic) | `v2-prefreeze-baselines` item 4 | C-10 |
| App E | EP signatures full | `v2-full-runs` item 7 | C-10 |
| App F | prominent negatives (N4/N5/N6/N12–15/N17/N18/N22/N19) | `negative_results.md` | **C-9** |

**Compliance self-check:** CM-1 wording used with its full scope tail (contraction-forbidden ≠ causal cap; short-horizon, no long-horizon fit-gap claim). CM-4 used verbatim incl. the input-driven-RMSE gap. **CM-3 (forbidden) never appears** — V2 makes no energy-as-confidence-signal claim (no gate/energy-superiority content in the paper at all). Canonical constants (§1 of matrix) cited identically: latch $q_\infty=q_0+\varepsilon p_0/(M\gamma)$; charge decay $(1-\gamma)^n$; $n_{1/2}\propto\mu^{-2}$ valid $\varepsilon\mu\lesssim\gamma/2$; floor $2\ln2/(-\ln(1-\gamma))$; $h^*\approx\gamma/2$ EP $\sqrt{h-h^*}$; $\gamma^*\approx2\varepsilon\mu$; $\det J=(1-\gamma)^d$; inertial M vs spectral μ (never bare "mass"). Scale qualifiers in-sentence everywhere (dim 4, S¹, seeds, laptop-CPU). C-6: BIBO framed "within the coercive-potential / compact-sublevel-set scope" next to the claim; saddle-blindness caveat carried (App F N22).

---

## The canonical C-1 physics-audit paragraph (for Hub sign-off — becomes the shared text for all shorts)

> **A brief audit disclosure (own the falsifications first).** The theory reported here is a product of an audit methodology we apply to our own primitive, and that methodology first turned up three defects in the reference instantiation, which we state up front. *(i)* A chaos-suppressing regularizer of the form $\mathrm{mean}_i\log\sigma_i(J)$ is identically $\tfrac12\ln(1-\gamma)$ — independent of the parameters, the state, and the potential — because conformal symplecticity fixes the *sum* of local Lyapunov exponents; it was therefore inert (zero gradient) for the program's entire history, and a max / positive-part statistic is required instead. *(ii)* The shipped Langevin sampler used a uniform noise scale $\sigma=\sqrt{2\gamma T\varepsilon}$ that equilibrates each mode at its own temperature $T_{\mathrm{eff},i}=2\varepsilon T/((2-\gamma)M_i)$, measured $\approx11\times$ colder than nominal, so it never sampled the intended Gibbs measure; the fluctuation–dissipation-consistent scale $\sigma^\star_i=\sqrt{M_i T\gamma(2-\gamma)}$ restores it. *(iii)* The velocity-saturation ("causal cap") analysis was implicitly $M=I$: with a learned inertial mass the light-cone is mass-anisotropic, $v_i^{\max}=c/\sqrt{M_i}$, so a single scalar speed bounds the state only in the identity-mass case. We report these as audit findings rather than bury them; each is now fixed behind a flag, and (i)/(ii) are today cosmetic — the learned inertial-mass spectrum is narrow — but become load-bearing once a mass hierarchy is designed in.

Provenance for each sentence: (i) = N17 / f5-note Cor 3 (mean-log-sv ≡ ½ln(1−γ), θ-indep, dev ≤2.1e-16). (ii) = N18 / f5-note Prop 9 & check (e) (Var(p) 0.0263→1.000000; ~11× colder). (iii) = N22-adjacent / f5-note Prop 12 (v_i^max = c/√M_i, check (i) to 9 digits). **Hub: please confirm this is the canonical wording before I reuse it in V1/V3 and reconcile with the F5 note's own framing.**

## Working-title candidates (V2 short — Head picks/​workshops)
1. **What a Trained Recurrent Memory Obeys: a Mode-Mass Budget for Retention, Latching, and Forgetting** *(current placeholder; ML-first, C-3)*
2. **The Overdamped Face of a Memory Law: Retention Budgets in Damped Symplectic Recurrent Units**
3. **Latch, Register, Forget: an Exactly-Solvable Retention Budget, Verified on Trained Recurrent Networks**

## F5-note conversion notes
- Faithful LaTeX of the note's THE DRAFT (§1–11 + provenance appendix). All 12 scoped results + both geometric corollaries retained; Cor 2/3 moved to `Appendix: Geometric corollaries` with the Cor-3 practitioner footnote intact (Head 2026-07-07 policy).
- Open questions carried **unresolved** as `[TODO-HEAD]` margin notes: **Q1** Minami–Hidaka ref (confirm/drop), **Q3** the "few-to-fifteen-percent" anharmonic figure (companion vs. citable), **Q5** verification check (c) squeeze/boost pruning; plus a Cor-3-footnote keep/cut flag and a title-choice flag. I resolved none.
- Three titles preserved in the note header as the working-title bracket.

---

## Open editorial questions (for the Hub / Head)
1. **Erosion placement (§3.5 vs. Appendix B vs. standalone).** I put the *recipe* in main-text §3.5 (it is the training method the paper needs) and the *full erosion phase-diagram + cure* in Appendix B, with all novelty claims marked "pending Jul-11 scout confirmation" (CM-6 open slot). Confirm this split, or promote/demote after the scout returns.
2. **C-1 canonical wording sign-off** (above) — needed before V1/V3 reuse.
3. **Venue-specific length.** Drafted appendix-maximal (C-10); main text is ~5 pp of dense prose. The dedicated pruning pass (near the Jul-11 venue pick) will need to know the exact page limit (NeurReps vs. ML4PS differ). Flag which.
4. **F5-note citation string.** The V2 short cites the theory note as "(Anonymous, 2026; the theory note)". Once the F5 note is live with a real arXiv id + Head-chosen title, the Hub should supply the exact citation string (claims-matrix §4 open slot) so both drafts agree.
5. **Headline figure asset.** I identify Fig 2 (`fig2_mo`, Mo head-to-head) as the headline and Fig 3 (`figA_retention_overlay`) as the CM-4 figure. Figures are not embedded (PNG assets live under `.claude/outputs/v2-full-runs/` and `.claude/outputs/v2-prefreeze-baselines/`); the LaTeX has `\resizebox` table placeholders, no `\includegraphics` yet. If the referee pass wants embedded figures, point me at the final PNG paths and I'll wire `\includegraphics`.

## Missing-number flags (none fabricated; these are genuine gaps)
- **No CLU input-driven task-RMSE** exists (N6) — stated as an honest gap in §3.3, not filled. This is by design (no velocity ingestion); do **not** expect a number here until the equivariant-control wrapper is built.
- **Long-horizon fit-gap crossing** (CM-1 tail) is unmeasured until `fit-gap-anatomy` reports; §3.4 explicitly does not claim persistence. If `fit-gap-anatomy` (w6) has landed, its number should be folded into §3.4 — I did not have it in my evidence set.
- **F5-note anharmonic-% figure** (open Q3): the "few-to-fifteen-percent" phrase is companion-anticipated, not a verified constant; left as `[TODO-HEAD]` in the F5 note, and the V2 short instead uses the *measured* +12–28% (v2-prefreeze item 4) with its decomposition (App D).

## Proposed handover updates (for the Hub)
- **Drafts exist for paper-referee (next wave):** `v2-short/draft.{md,tex}` (builds) and `f5-note/f5-note.tex` (builds, 0 overfull). Both are appendix-maximal per C-10; neither self-pruned.
- **CM-6 owner note:** V2 §3.5 + App B are written on the assumption the erosion novelty *may* hold; every novelty sentence is hedged "pending scout confirmation." If the Jul-11 scout finds prior art, only the four App-B novelty sentences + one §3.5 clause need softening — the law/cure numbers stand regardless (they are measurements).
- **Claims-matrix §4 open slots this draft touches:** erosion placement (proposed: §3.5 recipe + App B full); F5-note title (3 candidates carried); F5-note §3 reference string (still needs the live arXiv id).
- **Reused positioning prose:** §4 related-work Mo paragraph lifted from `mo-deep-read` §4; Di Bernardo/Keller paragraph from `di-bernardo-skim` "Draft prose to lift" (guard-rail applied: headline the pricing/budget, not "we choose G/H").
