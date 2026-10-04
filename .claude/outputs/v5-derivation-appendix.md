# v5-derivation-appendix — physics-theorist report
Task + acceptance criterion: add a LEAN proofs appendix to `~/Desktop/V5_PALM_Submission/paper.tex` covering only what V5 asserts, numerically self-checked against the composed map, wired by ≤2 inline `\ref`s, main text otherwise untouched.
Status: **done.**

**DIAL DECLARATION (echoed): NONE.** No experiment, no config change, no registry, no charter. One appendix + two cross-references + this report + scratch check scripts.

> **Reconciliation list? None required downstream** — no printed number conflicts with the algebra (STOP clause never fired). Three *findings about the paper* are listed in §5 for the Head/Hub to adjudicate (undefined `F`, undefined `ℓ_θ`, and the argmin-is-instrument-determined result); none require a text change to ship.

---

## 1. What I did
- Read `paper.tex` in full (423 lines at boot; md5 `a5182217490642da6e62579eca576e7b`, mtime 2026-08-26 02:27:22 — unchanged at apply time, verified).
- Built the worklist from the paper itself (§2 sweep below, with positive control).
- **Pre-registered all check values** (`.claude/outputs/v5-derivation-appendix/PREREG.md`, written before any script ran), then verified every closed form **against the composed elementary map** — damped velocity Verlet re-implemented independently in numpy from `chlu/core/integrators.py`'s step definition, Jacobians by complex-step (machine precision), plus Monte-Carlo Langevin on a ring for the T>0 laws.
- Iterated the appendix in `/tmp/v5da_build/` (never in the live folder), built clean, proved main-text page breaks unmoved, then applied to the live file and rebuilt.

## 2. Classification sweep (deliverable 1)
**Positive control:** `grep -c 'gamma_{\rm crit}=2\varepsilon\mu' paper.tex` → **7 hits** (sweep machinery sees the file; the `.claude`-grep hazard does not apply here — the paper is on Desktop, not gitignored).

| # | assertion (site) | class | disposition |
|---|---|---|---|
| 1 | one damped step = 2×2 matrix fixed by (εμ,γ); spectral radius sets retention (l.52) | ASSERTED | **derived, D.1** |
| 2 | det J=(1−γ)^d; (1−γφ(q′))^d gated (l.76) | ASSERTED | **derived, D.2** (+ ref site H1) |
| 3 | γ_crit=2εμ (abstract l.35, l.78, captions 207/213/222) | ASSERTED | **derived, D.3** |
| 4 | μ⁻² overdamped law, saturation floor, ∓1 asymptotes, branch asymmetry (l.78–85) | ASSERTED | **derived, D.3** |
| 5 | underdamped envelope 2ln2/γ (caption l.213) | ASSERTED | **derived, D.3** |
| 6 | mass-independent floor "exactly 27.03 steps at γ=0.05" (l.183) | ASSERTED | **derived, D.3** (27.0268 — the precedent's bar, matched) |
| 7 | left-branch \|λ\|²=1−γ ⇒ n₁/₂=ln2/(−½ln(1−γ)); 0.998999499 / 692.5 / 693.1 (l.279) | partially DERIVED-IN-PAPER (arithmetic shown) | **derived from the map, D.3** |
| 8 | T=0 latch: \|λ_flat\|=1 at every γ; γ=0 control drifts 142.7 rad (l.93, l.181, Fig 5b) | ASSERTED | **derived, D.4** (Jordan-block shear mechanism) |
| 9 | soft eigenvalue 2α ⇒ τ_max=Γ/2α (negatives, l.399) | ASSERTED | **derived, D.4** (one line; Γ=γ/ε stated) |
| 10 | σ*ᵢ=√(MᵢTγ(2−γ)) (l.67, l.309) | ASSERTED | **derived, D.5** |
| 11 | D_θ=εT(2−γ)/(2F²γ) ⇒ n₁/₂∝γ⁺¹, ∝T⁻¹ sign flip (l.95) | ASSERTED | **derived, D.6** (+ ref site H2) |
| 12 | γ_eff, T_local, hole D_θ, vault=(γ_eff/γ)², 110.25, coupled-bath 13.88, separation 7.942, refrigerator column 0.36249/0.23082/0.17480/0.12591, T_local=1.26e-4 (l.97, l.309–311, table l.321-326) | ASSERTED | **derived, D.7** (incl. the identity: separation factor ≡ refrigerator factor) |
| 13 | "decay commutes with deletion" + byte-identity to never-inserted history (l.109, l.359) | ASSERTED ("it can be shown" class) | **derived, D.8** (uses only the paper's stated conditions) |
| 14 | trilemma "may only optimize two" (l.111, l.378) | ASSERTED | **NOT derivable — §6.1** |
| 15 | n₁/₂=ln2/(−ln\|λ_ret\|) (l.229) | instrument definition | no derivation needed |
| 16 | γ_crit predicted bands 0.082–0.116 / 0.02334/0.01424 from printed μ² (l.183, table l.240) | arithmetic of #3 | verified (R5) |
| 17 | Blelloch–Golovin placement, fix-up cascade, history-independence notions (App D) | CITED-TO-SOURCE | out of scope by rule |
| 18 | capacity "≈1–1.6 bits" (l.290, l.389) | measured statement | not a closed form; see §6.2 |

Nothing on the Advisor's expected list is absent from the paper; every expected item was found and derived. Nothing was derived that the paper does not assert.

## 3. Numerical check table (deliverable 3) — prereg → derived → printed
Scripts: `.claude/scratch/v5-derivation-appendix/{check.py,check2.py}` (+ inline argmin probe); raw output `.claude/outputs/v5-derivation-appendix/check_results.txt`. All rows are **against the composed map / MC**, not the formulas themselves.

| id | quantity | pre-registered | measured vs composed map | paper prints | verdict |
|---|---|---|---|---|---|
| R1 | det J, d=4 anharmonic, γ=0.13 | 0.57289761, rel<1e-9 | 0.572897610000, rel err **0.0** | (1−γ)^d | ✅ |
| R1b | gated det | ((1−γ)(1−φ(q′)))⁴ | rel err 2.2e-16 | (1−γφ(q′))^d | ✅ |
| R2 | closed-form λ± vs composed-map eigs (5 (h,γ) pts) | <1e-12 | worst 2.0e-15 | "exactly solvable" | ✅ |
| R3 | γ=0.002 underdamped identity | 0.9989994995 / 692.46 / 693.147 | 0.998999499 / 692.45 / 693.147; μ-independent across h (R3b) | 0.998999499 / 692.5 / 693.1 | ✅ |
| R4′ | floor at γ=0.05 (h=0.04 and 0.058) | 27.0268 | **27.0268** both h | **"exactly 27.03"** | ✅ **precedent bar matched** |
| R5 | γ_crit=2εμ arithmetic | [0.08185,0.11611]; 0.023343; 0.014245 | idem | 0.082–0.116; 0.02334; 0.01424 | ✅ |
| R6 | exact merge point vs h*(γ)=(1−√(1−γ))√(2/(2−γ)) | agree; γ*/(2εμ)=1−εμ+O((εμ)²) | γ*/(2h)=0.9884/0.9605/0.9521 vs 1−h=0.9883/0.9591/0.9500; map goes complex↔real exactly there | γ_crit=2εμ | ✅ (leading order, as claimed) |
| R7 | slope-below, emergent windows, composed map | −1.002±0.001 | **−1.0023 / −1.0016 / −1.0022** | I-J −1.0023 / −1.0016 / −1.0022 | ✅ **4-decimal reproduction** |
| R8 | slope-above, emergent | +1.10…+1.13 | **+1.1262 / +1.1030 / +1.1254** | I-J +1.1262 / +1.1031 / +1.1254 | ✅ **≤1e-4 reproduction** |
| R8 | slope-above, designed (μ²_rad 0.670…1.348) | +1.20…+1.30 | +1.2352…+1.2877 | "+1.23 to +1.27" | ✅* (see §5.4) |
| R8b | slope-below, designed | −1.004…−1.010 | −1.0047…−1.0060 | −1.006 | ✅ |
| — | argmin/γ_crit, ideal map + 48-grid + parabolic estimator | (post-hoc probe, labelled as such) | **0.8994 / 0.9047 / 0.9056** | 0.8994 / 0.9046 / 0.9055 | ✅ see §5.3 |
| R9 | MC Var(p)/(MT), fdt scale | 1.000±0.02 | 1.0118 (O(h²) Verlet bias + MC) | σ* claim; paper's own cells 0.9981–1.002 | ✅ |
| R10 | refrigerator column MC | 0.36249/0.23082/0.17480/0.12591 ±1% | obs/pred 1.0028/1.0106/1.0050/0.9951 | same four values | ✅ |
| R11′ | D̂_θ vs εT(2−γ)/(2F²γ), F²=Mr*², M=2, r*=1.3 | 1.00±0.05 | ratio 0.9873 | D_θ formula | ✅ |
| R12′ | vault D̂(0)/D̂(0.5) MC | 110.25±10% | **111.50** | 110.25 (pred), 107.77±4.78 (measured) | ✅ |
| R12 | coupled-bath vault MC; identity | 13.881; vault_abs/vault_cpl ≡ T/T_local = 7.9423 | 14.05 MC; identity exact (7.9423 = 7.9423); T_local = 1.2591e-4 | 13.88; 7.942; 1.26e-4 | ✅ |
| R13 | T=0 latch rollout 2e5 steps, γ∈{0.002,0.5} | drift <1e-10 | drift **0.0** (float64); γ=0 control ballistic at exactly εp_θ/(Mr*) = 1.923e-4/step | ≤4.9e-12 (checkpoints); 142.7 rad mechanism | ✅ |
| R14 | decay∘delete = delete∘decay, toy canonical store | byte-equal | layout equal, amplitudes bit-equal | commutation claim | ✅ |

**STOP clause: never fired.** No derivation disagrees with any printed number. One scripting error was found and fixed in the open (first R4 run used h=0.02 < h*(0.05)=0.02564, i.e. the overdamped side — wrong probe, not wrong algebra; corrected run R4′ is the quoted row).

## 4. The appendix + wiring (deliverables 2 & 4)
**Three hunks, nothing else. Proof:** reversing exactly these three hunks from the final live file reproduces the boot md5 `a5182217490642da6e62579eca576e7b` byte-for-byte (`/tmp/v5da_reconstructed_orig.tex`).

- **H1 (ref site 1, §2.1 line 76):** "…when position-gated." → "…when position-gated (this and the closed forms below are derived in App.~\ref{app:derivation})." — placed at the paper's first closed form.
- **H2 (ref site 2, §2.2 line 95):** "…$D_\theta=\varepsilon T(2-\gamma)/(2F^2\gamma)$, which…" → "…$(2F^2\gamma)$ (App.~\ref{app:derivation}), which…"
- **H3 (appendix):** `\section{Derivations of the Closed Forms}\label{app:derivation}` inserted immediately before `\end{document}` (renders as **App. G**, after Prominent Negatives — no existing appendix re-letters). Blocks D.1–D.8 as in §2's disposition column. Zero new numbers (every constant that appears — 27.03, 0.998999499, 692.5, 693.1, 110.25, 13.88, 7.942, 1.26e-4, 0.36249/0.23082/0.17480/0.12591, 142.7, 2α, Γ/2α — already appears in the paper). Zero new claims; the one supplied *definition* (F² = Mr*²) is flagged in §5.1.

**Build evidence:** live rebuild ×2 with `/Library/TeX/texbin/pdflatex`: **0 errors, 0 undefined references, 0 multiply-defined, 0 overfull boxes** (baseline also had 0). Output: **19 pages** (was 17; appendix ≈2 pp, excluded from the venue limit).

**Main-text page measurement, before vs after:** appendices start on p.6 in both builds; References begin on p.5 in both. Per-page pdftotext diff over the original 17 pages: only p.2 and p.3 differ, and only *within* the two edited paragraphs — **page 2 and page 3 begin and end on identical words** in both builds, and **p.5 (where the main text ends) is byte-identical** ⇒ the 4.31-pp main-text measure is exactly unchanged; not a single line was pushed into the main text. (p.17 gains the start of App. G below the last negatives table — appendix zone.)

**Reachability:** both `\ref`s resolve to "App. G" in the built PDF (seen in the page-2/3 text diffs).

**Live-file discipline:** md5+mtime recorded at boot; re-checked immediately before applying — unchanged. All iteration happened in `/tmp/v5da_build/`; the applied live file is byte-identical to the verified scratch file. Side effect: live `paper.pdf/.aux/.log/.out` regenerated by the rebuild (necessarily).

## 5. Findings about the paper (not defects in the algebra; nothing was changed in the main text)
1. **`F` is used but never defined** (l.95, l.309). The derivation forces its meaning: **F² = mass-weighted squared displacement per unit stored coordinate, F² = M r*²** — exactly what the measurement harness computes (`t-lever-forgetting/common.py:140`: `F_sq = M_ch * r_star**2`). The appendix's Assumptions bullet now supplies this definition explicitly (and says it is supplying it). Dimensional aside: D_θ is per unit dynamical time t = nε — also previously unstated, now stated.
2. **`ℓ_θ` (thermal persistence length) is used in ratios (`ℓ_θ/Δ`) but never defined anywhere in the paper.** I did not invent a definition in the appendix (that would be a new claim). Writer-level fix if the Head wants one.
3. **The measured argmin `0.902±0.003×γ_crit` is entirely an instrument+grid property of the ideal integrator map.** The exact map's true minimum sits at the branch-merge point γ* = 2εμ(1−εμ+O(ε²μ²)) ≈ 0.988γ_crit at emergent h; but the ideal 2×2 map evaluated on the 48-point geomspace(0.002,0.5) grid with a 3-point parabolic argmin in log-log returns **0.8994/0.9047/0.9056 — the paper's printed per-seed values to 4 decimals** (the parabola vertex is pulled left by the √(γ−γ*) kink of the overdamped branch). Together with R7/R8 (slopes reproduced to ≤1e-4), the entire I-J shape row of the App.-C table is integrator algebra: a *strengthening* (checkpoints track the ideal map to 4 decimals — the coset really is an exact normal mode) and a *caveat* (the 0.90 number carries no model information; the model-dependent content is γ_crit itself and the finite-γ slope corrections). The paper's own "left branch is an integrator identity" honesty already points this way; this extends it to the argmin ratio. **Not a STOP-clause event** — no printed number is contradicted. The Hub may want this on record before a referee finds it.
4. **Designed slope-above:** ideal map on my assumed 48-pt grid gives +1.235…+1.288 across the five printed μ²_rad; the paper prints "+1.23 to +1.27". The stiffest seed (μ²=1.348) computes +1.288, just outside — with only 5–8 grid points in that window and the designed run's actual grid/window unpublished, this is within instrument ambiguity, and the paper's numbers are measurements I must not touch. Recorded honestly; no action.
5. **Symbol overload (cosmetic):** `Γ` means damping rate γ/ε in the negatives row (τ_max=Γ/2α) but decay rate in App. C's `Γ_jac/Γ_R3` table. D.4 disambiguates locally ("damping rate Γ=γ/ε") without touching either site.

## 6. What the paper asserts that I could NOT derive from its stated assumptions (deliverable 5 — required section)
1. **The trilemma (l.111/l.378): "exact value fidelity, amplitude-independent address hold, amplitude-independent read latency — a system may only optimize two."** Stated as strict ("formally price…a strict trilemma"), but deriving it requires a model linking amplitude to read latency and to basin hold that the paper never states (the empirical section *refutes two repair attempts*, which is evidence for, not proof of, impossibility). I did not patch a proof; per task ⛔ this is reported as a finding. If the Head wants it provable, the missing assumption is a stated monotone latency–depth/amplitude relation for the two-phase read.
2. **Capacity "≈1–1.6 bits" (l.290/l.389):** consistent with log₂ of 2–3 washboard minima, but the paper prints no minima count, so the link is unverifiable from the paper alone. It is presented as a measurement, so nothing is owed — listed for completeness.
3. Everything else asserted is either derived (D.1–D.8), arithmetic of a derived form, a measurement, or cited to a source. **Nothing else is un-derived.**

## 7. Acceptance criteria check
1. Every ASSERTED item derived or listed in §6 with reason — ✅ (§2 table). 2. Zero new numbers/claims; STOP clause honoured (never fired) — ✅. 3. Checks run against the composed map with pre-registration; printed constants reproduced (27.0268→27.03; slopes to 1e-4; argmin to 4 decimals; 110.25/13.88/7.942/1.26e-4/refrigerator column exact) — ✅. 4. Diff = appendix + 2 ref hunks; reversal reproduces boot md5; main-text pages unmoved — ✅. 5. Reachable: both refs resolve to App. G — ✅.

## Flag-provenance table
| item | value |
|---|---|
| repo commit | no tracked file touched; repo HEAD not relevant (object is `~/Desktop/V5_PALM_Submission/`, outside the repo). `chlu/` read-only. |
| object file | `paper.tex` boot md5 `a5182217490642da6e62579eca576e7b` (2026-08-26 02:27:22) → final md5 = md5 of `/tmp/v5da_build/paper.tex` (byte-identical, verified) |
| check env | `/Users/user/Desktop/CHLU/.venv/bin/python`, numpy 2.4.1; pure numpy (no JAX) — the map re-implemented from `chlu/core/integrators.py`'s definition; complex-step Jacobians |
| constants | ε=0.05 throughout; grid geomspace(0.002,0.5,48); windows γ<γ_crit/2.5, γ>2.5γ_crit (paper's); MC ring M=2.0, r*=1.3, μ²_rad=1.0, T=1e-3, γ=0.05, γ_φ∈{0.1,0.2,0.3,0.5}, absorb-only noise σ²=MTγ(2−γ); seeds: numpy default_rng(7,…,42) as in scripts |
| latex | `/Library/TeX/texbin/pdflatex` (TeXLive 2026), 2 passes, baseline rebuild reproduced the Head's shipped PDF byte-size exactly (1743790) |
| prereg | `.claude/outputs/v5-derivation-appendix/PREREG.md` written before any script ran |

## Git footprint
None. No tracked file created/modified/deleted. Artifacts: `.claude/outputs/v5-derivation-appendix{.md,/PREREG.md,/check_results.txt}`, `.claude/scratch/v5-derivation-appendix/{check.py,check2.py}`, `/tmp/v5da_build/` (scratch build, disposable), `/tmp/v5da_reconstructed_orig.tex` (diff-proof witness).

## Open questions / risks
- The live folder's `paper.pdf` is now the 19-page build; if the Head prefers to own the final compile, the tex alone carries everything.
- §5.3 (argmin = instrument property) and §5.4 (designed +1.288 edge) are the two places a referee could push; both are now understood and quantifiable on request.
- §6.1 (trilemma) is the only "strict" word in the paper without a proof behind it; Head's call whether to soften "strict" or leave it (I touched nothing).

## Proposed handover updates (for the Hub)
- **§1.6/CM-16 addendum:** the V5 I-J shape row is now *derived*, not just measured: composed-map algebra reproduces the printed slopes to ≤1e-4 and the printed argmin ratios (0.8994/0.9046/0.9055) to 4 decimals via grid+parabolic estimator; the exact map's true minimum is at γ_crit(1−εμ+O(ε²μ²)). The model-dependent content of the V-curve is γ_crit and the slope corrections 1+γ/(2−γ) — quote accordingly.
- **§7 add (wording debt, V5):** `F` and `ℓ_θ` are used in `paper.tex` without definition; App. G now defines F (F²=Mr*², per-unit-time D). ℓ_θ still needs a one-line definition if any ratio `ℓ_θ/Δ` is ever challenged. The trilemma (l.111) is asserted "strict" without a derivable basis from stated assumptions (§6.1).
- **§8 note:** the identity vault_absorb/vault_coupled ≡ T/T_local (separation factor = refrigerator factor, exactly) is now proven (App. G, D.7) — usable as a one-line consistency check in any future vault harness.
