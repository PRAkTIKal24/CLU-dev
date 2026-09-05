# v1-derivation-appendix — physics-theorist report

Task + acceptance criterion: make V1 (`pj_sub.tex`) self-contained on the two proofs it attributes to the cut theory note (P1 causal box, P2 bounded-injection certificate), as one appendix block + ≤2 `\ref` insertions, every check against the composed map, prereg-first.
Status: **done**
⚠ **Reconciliation list with no owner yet (per protocol §5 corollary):** §"Theory-note sites" below lists 4 main-text/bib sites needing Head rewording now that the note is cut, and §"Findings about the paper" lists 3 wording defects (missing `/M` in the §3.1 displacement law; "kinetic energy 0.72" underivable from A.1; det=2.05 called a "contraction" though it is an expansion). **The Hub should assign these to the Head/curator at review.**

## DIAL DECLARATION (echoed from task)
**Dials touched: NONE.** Derives algebra already implied by the paper; adds one appendix and two cross-references. No experiment, no configuration change, no new measurement.

## Pin check & footprint integrity
- `pj_sub.tex` md5 at start = `727ebee2b8498b4095f8bb7159258f90` — **matches the scoping pin**. Proceeded.
- `pj_sub.tex` md5 after edit = `6867e06b56d97aadc52398558e9e4797`.
- **Diff provably exact** (git can't diff — `.claude/**` gitignored): reverting my three edits in a scratch copy reconstructs a file whose md5 **equals the pin exactly** (`scratch/v1-derivation-appendix/pj_sub_before_reconstructed.tex`). ⇒ diff = one appendix block (G, 76 lines) + two single-line parenthetical `\ref` insertions, nothing else.
- **Protected files byte-untouched**: md5 manifest of `.claude/papers/v1-short/**` (11 files) + `submission.tex` identical before/after (`manifest_before.txt` ≡ `manifest_after.txt`, diff empty; `submission.tex` = `caef2272f9dc96d349b46486563d24ee` both times).
- C-8 hermetic: no sibling-short draft was opened (v1-short files touched only by `md5`).

## What I did
1. Read protocol, handover §1/§7/§8, task file; verified pin.
2. Wrote `outputs/v1-derivation-appendix/PREREG.md` (17 pre-registered check values, derived by hand) **before any script or build existed**.
3. Wrote 4 standalone numpy check scripts (composed maps only), ran them — **all 17 checks pass**.
4. Built the before-state (page split), inserted Appendix G + 2 `\ref`s, built the after-state.
5. This report.

## Prereg-first mtime chain (provable)
```
21:27:50  PREREG.md
21:28:16–21:29:08  check_*.py (4 scripts)
21:29:12–21:29:13  check_*.out (results)
21:29:46  build_before/pj_sub.pdf
21:31:58  pj_sub.tex (edited)
21:32:07  build_after/pj_sub.pdf
```

## STOP clause: **did not fire.**
No derivation disagreed with any number the paper prints. (Three *wording* defects found — see Findings — none is a numeric contradiction; in each case my algebra agrees with the paper's printed numbers and disagrees only with a prose formula/label the paper itself contradicts elsewhere.)

## Derivations (as shipped in Appendix G) — claim → assumptions → result → verdict

**G.1 Causal box (P1).** Assumptions: relativistic kinetic `T(p)=√(c²pᵀM⁻¹p+m₀²c⁴)`, `M≻0` diagonal, `m₀>0`; kick–drift–kick–damp step. Since `∂T/∂p_i = c²p_i/(M_i T)` and `T > c|p_i|/√M_i` strictly, one drift moves `|Δq_i| < εc/√M_i` for **every** p and **every** V_θ; kicks/damping move no position; sum over n≤T steps ⇒ `Q_T ⊆ C_T`, `L_i = Tεc/√M_i`. Corollary (energy-blindness): saturation deficit `1−|q̇_i|√M_i/c ≈ m₀²c²M_i/(2p_i²)`. **Verdict: proven** (symbolic + composed-map machine-precision agreement).

**G.2 Bounded-injection certificate (P2).** The squeeze that reproduces every printed number is the **mass-weighted hyperbolic rotation**: with `u=√M_i q_i, v=p_i/√M_i`: `u'=u coshζ+v sinhζ, v'=v coshζ+u sinhζ`, i.e. `q'=q coshζ+(p/M)sinhζ, p'=p coshζ+Mq sinhζ`; det = cosh²−sinh² = 1 (symplectic). Matched-quadratic H (isotropic in u,v): `u'²+v'² = (u²+v²)cosh2ζ + 2uv sinh2ζ ≤ (u²+v²)e^{2|ζ|}` via `2|uv|≤u²+v²` ⇒ **H(S_ζz) ≤ e^{2|ζ|}H(z)**, equality iff `u=±v`; spectator dims only tighten. Launch state (q=0) ratio = cosh2ζ exactly. Bracket: reach ⊆ `[L, L+(p₀/M₀)sinhζ]`; landing threshold `ζ ≥ arcsinh((d−0.4−L)M₀/p₀)` ⇒ **2.0105** (d=4.0), **2.6441** (d=5.0) — the paper's printed thresholds, independently confirming the Advisor's reproduction. Scaling corollary: `cosh2ζ = 1+2(M₀δ/p₀)²` — **exactly quadratic in excess distance δ**, exponential in rapidity, NOT exponential in distance (matches the paper's already-corrected statement). Scope: matched-quadratic only; quartic can exceed (verified). **Verdict: proven** (in the matched-quadratic scope the paper itself states).

**G.3 Hard-gate Jacobian (third candidate — INCLUDED; judged genuinely underived and cheap).** `q↦q+g(q)Δ` gives position block `I+Δ∇gᵀ` (rank-one); matrix determinant lemma ⇒ `det J = 1+∇g·Δ`; frozen gate restores det J = 1 exactly. **Verdict: proven.** (4 lines; the formula is asserted in §3.1 with only a unit-test number behind it.)

## Numerical verification — every row against the COMPOSED map
Scripts + outputs: `.claude/scratch/v1-derivation-appendix/check_{p1_box,p2_squeeze,bracket,gate}.{py,out}`. Verlet built from elementary kick/drift/kick/damp; squeeze built as weight∘boost∘unweight; Jacobians by central finite differences of the map function; H coded independently.

| # | Check (prereg #) | Pre-registered | Observed (composed map) | Pass |
|---|---|---|---|---|
| 1 | per-step \|Δq_i\| < εc/√M_i, 200 trials, \|p\| up to ~1e7, kicks on | (0.025, 0.1) strict | worst (0.0249973, 0.0999999) | ✓ |
| 2 | 100-step reach ≤ L, p₀∈{1.2,10,1e3,1e6} | ≤ 2.5 from below | 0.4608 / 2.4359 / 2.49999499 / 2.499999999995 | ✓ |
| 3 | deficit at p₀=1e6 vs m₀²c²M₀/2p₀² | 2.0e-12 | **2.000e-12 = predicted exactly** (⭐ this is the paper's App. E "2.0×10⁻¹²") | ✓ |
| 4 | V-independence (double well vs V≡0) | identical box | max\|q₀\| identical to 12 digits | ✓ |
| 5 | composed squeeze det (finite diff) | 1 ± ~1e-6 | max dev 2.5e-8 (consistent w/ paper's ±4e-6) | ✓ |
| 6 | launch-state ratio = cosh2ζ | 1.127626/1.543081/3.762196/27.308233 | identical to 6 d.p. | ✓ |
| 7 | bound e^{2ζ} vs paper's printed bounds | 1.6487/2.7183/7.3891/54.5982 | match paper's 1.65/2.72/7.39/54.6 to printed precision | ✓ |
| 8 | ordering cosh2ζ ≤ printed ratio ≤ e^{2ζ} | all 4 rows | all 4 True | ✓ |
| 9 | single x=0.00703 (from ζ=2 row) predicts other 3 printed ratios | 1.13/1.55/3.79 | 1.1313→1.13, 1.5513→1.55, 3.7877→3.79 | ✓ |
| 10 | bound over 1000 random states (+2 spectator dims), equality at u=v | 0 violations; ~1e-12 | 0 violations (min slack 1.6e-3); equality to 7e-15 | ✓ |
| 11 | quartic can exceed bound | ≥1 violating state | ratio 291.0 > 54.6 | ✓ |
| 12 | ζ(d=4.0)=arcsinh(11/3) | 2.010530 (hand) | 2.010527 → rounds to paper's **2.0105** (hand-arith slip of 3e-6 in prereg; both round identically) | ✓ |
| 13 | ζ(d=5.0)=arcsinh(7) | 2.644121 | 2.644121 = task target **2.6441**, rounds to paper's 2.64 | ✓ |
| 14 | one-sided theorem: NO landing below threshold (composed rollout) | all False below; ζ* above, V-dependent | d=4.0: no landing at thr−{0.3..0.001}, ζ*=2.1505; d=5.0: same, ζ*=2.6741 | ✓ |
| 15 | paper ζ-grid ≤2.0 reproduces C.1 squeeze row | [1,1,1,1,0,0]; max reach 3.588\* < 3.6 | [1,1,1,1,0,0]; 3.5881 (\*prereg hand value 3.5876, arith slip; conclusion identical) | ✓ |
| 16 | gate det J vs 1+∇g·Δ (finite diff, 50 random) | ≤~1e-7; frozen ⇒ 1 | max dev 8.1e-10; frozen det = 1.000000000034 | ✓ |
| 17 | 2.05 ⇒ ∇g·Δ=1.05; (g,Δ) not printed | not reproducible from paper | confirmed; an example (g,Δ) with ∇g·Δ=1.05 gives det 2.050000 | ✓ |

Two prereg hand-arithmetic slips (#12: 2.010530 vs 2.010527; #15: 3.5876 vs 3.5881) are reported verbatim; neither changes any rounding or conclusion.

### Flag-provenance table (mandatory)
| item | value |
|---|---|
| Repo HEAD at run (checks are standalone, no chlu code imported) | `7fcef50` |
| Paper file | `pj_sub.tex` md5 `727ebee2…` (before) → `6867e06b…` (after) |
| Interpreter / libs | main venv python 3.11.13, numpy 2.4.1 (float64 throughout); no JAX |
| Seeds | `numpy.default_rng(0)` in all 4 scripts |
| Physics config (= paper A.1) | ε=0.05, T=100 steps, c=1, m₀=1, M=diag(4.0,0.25), γ=0 (γ=0.3 extra check), p₀=1.2, barrier ΔV_b=1.0, landing tol 0.4, ζ grid = paper's |
| Representative double well (P1/bracket rollouts only; P1 theorem is V-independent) | `V(x)=16·ΔV_b·(x/d)²(1−x/d)²`, wells at 0 and d — the paper does not print its V; only one-sided/kinematic claims were preregistered against it |
| Build | tectonic 0.15.0, scratch outdirs; shipped `pj_sub.pdf` NOT regenerated (see follow-ups) |

## Edits made (deliverables 2+3)
**Appendix block** inserted between the Markov-kernels appendix (currently F) and `\section*{References}`: `\section{Derivations for the Reach and Injection Certificates}\label{app:deriv}` + `\subsection`s labeled `app:deriv:box`, `app:deriv:squeeze`, `app:deriv:gate`. **No letter hard-coded anywhere**; renders as G/G.1–G.3 today and renumbers for free. **Zero new numbers**: every numeric token (2.5, 0.4, 1.2, 4.0, 0.05, 100, 2.0105, 2.64, 2.0, 2.05, e^{2|ζ|}−1…) has a printed ancestor; arcsinh(7) is given to the paper's own rounding "2.64", not 2.6441.

**`\ref` insertion 1** (§2, "Falsifiable reachability constraints" paragraph), byte-context:
- before: `…but cannot exceed, the velocity limit $c/\sqrt{M_i}$. We define the ensuing dichotomy…`
- after: `…but cannot exceed, the velocity limit $c/\sqrt{M_i}$ (Appendix~\ref{app:deriv:box}). We define the ensuing dichotomy…`

**`\ref` insertion 2** (§3.1, squeeze paragraph), byte-context:
- before: `…strictly bounded by $H(S_\zeta z)\le e^{2|\zeta|}H(z)$, and the internal governor…`
- after: `…strictly bounded by $H(S_\zeta z)\le e^{2|\zeta|}H(z)$ (Appendix~\ref{app:deriv:squeeze}), and the internal governor…`

## Build & page split (deliverable 6)
- **0 errors, 0 undefined references** (log grep "undefined" = 0); sole warning = pre-existing overfull hbox (1.4pt) at the C.1 grid, present before my edit.
- **Before:** 15 pp total; main text ends on p. 9 (Appendix A starts p. 9) ≈ 8.3 pp. **After:** 17 pp total; **main text still ends on p. 9 — main-text growth: zero** (both insertions absorbed without reflow). New appendix G occupies pp. 14–16; References p. 16.
- Rendered pages visually verified (G.1/G.2 math correct, "Appendix G.1"/"Figure 1" refs resolve).

## Theory-note sites — exact rewordings for the Head (deliverable 5; ⛔ NOT made by me)
1. **§ Reference architecture (the load-bearing one is #3; this is the first mention):** "The theoretical proofs for the verifiable certificates are detailed in a companion theoretical note (Anonymous, 2026)." → **"The derivations for the causal-box and bounded-injection certificates are given in Appendix~\ref{app:deriv}."** (or delete the sentence; #2's replacement covers it).
2. **§2 "The governed map", last sentence:** "Complete derivations and machine-precision validations are provided in the companion theory note." → **"Derivations of the causal-box, bounded-injection and gate-Jacobian certificates are given in Appendix~\ref{app:deriv}; machine-precision validations of the exactness claims are reported in Appendix E."** ⚠ "Appendix E" would be a hard-coded letter — recommend the Head also add `\label{app:verif}` to the Analytic Verification Protocols section and use `\ref`. ⚠ Note: the conformal-symplecticity derivation (J^TΩJ=(1−γ)Ω) also lived in the note and is NOT in Appendix G (task scope was P1+P2+gate); if the Head wants "complete derivations" to remain literally true, that is one more (short) derivation, currently only verified (App. E), not derived, in-submission.
3. **⛔ THE sentence (§2):** "The theory note proves that in relativistic mode, a single drift step advances position $q$ by at most $\varepsilon c/\sqrt{M_i}$ per coordinate." → **"Appendix~\ref{app:deriv:box} proves that in relativistic mode, a single drift step advances position $q$ by at most $\varepsilon c/\sqrt{M_i}$ per coordinate."** (minimal token swap; everything after "proves" byte-identical).
4. **Bibliography:** delete `\item Anonymous (2026). \emph{[The theory note].}` (Head ruling already; no `\cite` points at it — it is a hand-built itemize, so no build breakage).

## Required section — what I could NOT derive from the paper's own stated assumptions
1. **The exact measured ratios 1.13/1.55/3.79/27.5.** From the paper's stated init (q₀=0 exactly) the ratio is cosh2ζ = 1.128/1.543/3.762/27.31 — *below* all four printed values by a consistent ≈0.7%. The closed form ratio = cosh2ζ + [2uv/(u²+v²)]sinh2ζ explains all four printed digits with a single x≈0.00703 (check 9), i.e. the measurement was taken at a state with slight positive q–p correlation (e.g., after capture or ≥1 flow step) — **but the paper does not state that phase**, so the third digit is not derivable. Not a STOP event: the printed values satisfy the derived two-sided bracket rigorously.
2. **§3.1 displacement law is missing the mass factor.** Main text prints "δq′ = δq coshζ + p₀ sinhζ"; the derivation that reproduces the paper's own bracket `[L, L+p₀sinhζ/M₀]` and thresholds 2.0105/2.6441 requires **δq′ = δq coshζ + (p₀/M₀) sinhζ** (at M₀=4 the printed inline form is 4× off; the literal form would predict ζ(d=4.0)=arcsinh(11/12)=0.85, contradicting the paper's own 2.0105). The Appendix-G derivation states the correct form; **Head should patch the inline formula** (insert `/M_0`, or say "mass-weighted momentum").
3. **"kinetic energy 0.72" (§3.2) is not derivable from A.1.** 0.72 = p₀²/2 (i.e. M=1); with A.1's M₀=4.0, p₀²/2M₀ = 0.18. Either the escape well uses unit mass (unprinted) or the KE quoted is mass-unweighted. Both values are < ΔV_b=1 so the escape-blocked claim survives either way; the number's provenance does not.
4. **The 2.05 unit-test value:** formula proven; the (g, Δ) realizing ∇g·Δ=1.05 is not printed, so 2.05 is a measurement, not derivable (preregistered as such).
5. **Governor timing laws** (≈2ζ/γ_c; ≈(1/γ_c)ln(1+ΔV/E⋆)) — deliberately NOT derived (lean rule; the paper itself grades them "verified to leading order," not exact). Honest status: asserted+evidenced in-paper, underived in-submission.
6. **T(p)'s closed form was never stated in the paper** — the causal-cap theorem was literally underivable in-submission before this task; Appendix G now states it (sourced to the reference architecture + flag table). This is the sharpest instance of the self-containment failure the task predicted.
7. **Minor vocabulary:** §3.1 calls det J = 2.05 an "unpaid contraction"; 2.05 > 1 is an *expansion* (sign of ∇g·Δ decides). Appendix G says "Jacobian/design guard" neutrally; Head may want "unpaid volume change" in main text.

## Acceptance criteria — status
- diff = exactly one appendix block + two single-line `\ref` insertions: **✓ (md5-reconstruction proof)**
- `.claude/papers/v1-short/**` + `submission.tex` byte-untouched, manifest printed: **✓**
- every check row against the composed map: **✓** (see table)
- prereg mtimes precede every result artifact: **✓** (21:27:50 < 21:28:16…)
- build 0 errors / 0 undefined refs; page split reported: **✓** (15→17 pp, main text 8.3 pp unchanged)

## Git footprint
None — all writes under `.claude/` (gitignored). Files written: `pj_sub.tex` (in-place, pinned+verified), `outputs/v1-derivation-appendix{.md,/PREREG.md}`, `scratch/v1-derivation-appendix/*`.

## Open questions / follow-ups / risks
- **Shipped `pj_sub.pdf` is now stale** vs the edited tex (I built only in scratch to keep the diff surface minimal; an advisor was actively building at 21:14). Whoever owns the build should regenerate; my after-build at `scratch/v1-derivation-appendix/build_after/pj_sub.pdf` is a drop-in preview.
- The 4 theory-note/wording items above need a Head pass (owner assignment per the reconciliation rule).
- If the venue port renumbers appendices, G's labels renumber for free, but the paper's pre-existing literal "Appendix B/C.3/E" strings (not mine) will not.

## Proposed handover updates (for the Hub)
- **§1 (V1 paper state):** V1 no longer depends on the theory note for its two load-bearing proofs: causal box and e^{2|ζ|} injection certificate (+ gate Jacobian) are now derived in-submission (Appendix G, labels `app:deriv{,:box,:squeeze,:gate}`), all closed forms verified against composed maps with prereg-first mtimes; STOP clause did not fire; paper constants all confirmed (incl. App. E's 2.0e-12 = m₀²c²M₀/2p₀² at p₀=1e6, and the bracket ζ=2.0105/2.6441 from arcsinh((d−0.4−L)M₀/p₀)).
- **§7 (paper wording debts, new):** (a) §3.1 squeeze displacement law misses `/M₀` (paper's own bracket + thresholds prove the mass-weighted form); (b) §3.2 "kinetic energy 0.72" underivable from A.1 (implies M=1); (c) det=2.05 mislabeled "contraction"; (d) 4 theory-note sites need the rewordings listed in my report; (e) shipped pj_sub.pdf stale vs tex.
- **§8 (minor):** the measured squeeze ratios sit ≈0.7% above cosh2ζ, exactly consistent with one uv-correlation parameter x≈0.007 — if anyone ever needs the third digit, log the phase at which H′/H is measured.
