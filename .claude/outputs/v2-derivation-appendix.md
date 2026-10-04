# v2-derivation-appendix — physics-theorist report

Task + acceptance criterion: add exactly ONE appendix `\section{}` to `condensed_paper.tex` deriving the paper's central closed forms (items 1–5 + optional 6), numerically confirmed against the paper's own constants; file byte-identical everywhere else (proven by diff); zero new numbers in the paper; `\ref` wiring proposed but NOT applied.
Status: **done** — all six items derived (item 6 fit), section = 1 rendered page (net +1 document page), 0 build errors, single-insertion diff proven.
**⚠ Concurrency notice (first-10-lines):** the Head live-edited `condensed_paper.tex` TWICE while I worked (typo pass + one content deletion in `app:loan`). My footprint is proven single-insertion against a reconstructed pre-my-edit state (§Diff proof below). If the Head edits again before Hub review, re-run the one-command proof in §How I verified.
**No downstream reconciliation list** — nothing in the paper needed changing; every derived form reproduced its published constant.

**DIAL DECLARATION (echoed from task):** none — analytical write-up. Zero new measurements, zero new numbers in the paper; every formula derived already appears there. Laundering control: n/a. Falsifies: a derivation disagreeing with a published constant (none did). Does-not-falsify: n/a.

## What I did
- Derived, from the damped KDK velocity-Verlet map with `p→(1−γ)p` on a mass-whitened normal mode (curvature μ², h=εμ): the 2×2 propagator and its eigenvalues; the three bands + the h=2 stability limit; the exceptional point h\*=√(2/(2−γ))(1−√(1−γ)) = γ/2+O(γ²) and its defectiveness; the overdamped half-life 2γln2/[(2−γ)(εμ)²] including the 2γ/(2−γ) coefficient; the μ-independent floor 2ln2/(−ln(1−γ)); and (item 6, it fit) the coset diffusion D_θ = εT(2−γ)/(2F²γ).
- Pre-registered all check values (`outputs/v2-derivation-appendix/PREREG.md`) BEFORE running the harness.
- Verified numerically (`scratch/v2-derivation-appendix/check_derivations.py`, numpy 2.4.1, main venv; the 2×2 block is built by composing K·D·K·damping elementary maps, so closed forms are tested against the map, not themselves).
- Inserted ONE `\section{Derivation of the band structure and closed forms}\label{app:derivation}` immediately after `app:defs` (before `app:curcos`), per the placement instruction. First draft cost 2 rendered pages + 1 new overfull hbox; compressed (det/tr, latch, floor, D_θ moved inline; `\emph` run-ins instead of `\paragraph`) to exactly the 1-page budget with zero new overfull boxes.
- Built COPIES in `scratch/v2-derivation-appendix/build/` (never in `NIPSsubmission/`), baseline vs with-appendix.

## How I verified
- Harness: `cd .claude/scratch/v2-derivation-appendix && /Users/user/Desktop/CHLU/.venv/bin/python check_derivations.py` — full output reproduced below (§Numerical verification).
- Build: `pdflatex×2+bibtex+pdflatex` (`/Library/TeX/texbin`, TeXLive 2026; `refs.bib` reused from `scratch/v2-cite-pass/diagbuild/` since `v2-neurreps/` carries none) on `build/paper_before.tex` (= reconstructed pre-my-edit file) and `build/paper_after.tex` (= frozen current file).
- Diff proof (re-runnable in one atomic command): snapshot live file → strip my section between the `\section{Derivation…}` and `\section{Curvature instantiation…}` markers → `git diff --no-index`.

## Findings/results

### Derivations (all **proven**, elementary linear algebra; assumptions stated)
Assumptions: locally quadratic V_θ at a critical point (the paper's own "solvable core is quadratic" scope, App `app:neg`); mass-whitened normal-mode reduction (paper Sec 3); damping applied as `p→(1−γ)p` after the KDK step. Everything below is exact under these assumptions — no conjectures in the shipped section.
1. **Block:** A = [[1−h²/2, ε], [−(1−γ)εμ²(1−h²/4), (1−γ)(1−h²/2)]]; det A = 1−γ (conformal symplecticity per mode), tr A = (2−γ)(1−h²/2). λ± = ½[tr ± √Δ], Δ = tr² − 4(1−γ).
2. **Bands:** Δ=0 at 1−h²/2 = ±2√(1−γ)/(2−γ); via 2−γ∓2√(1−γ) = (1∓√(1−γ))², roots h\* = √(2/(2−γ))(1−√(1−γ)) = γ/2+O(γ²) and h_f = √(2/(2−γ))(1+√(1−γ)) = 2−O(γ²). λ=−1 ⇔ 1+trA+detA=0 ⇔ **h=2 exactly, every γ** (the quoted stability limit). Latch (μ=0): eigenvalues {1, 1−γ}, q∞ = q₀+εp₀/(Mγ) by geometric sum.
3. **Exceptional point:** at h\*, double eigenvalue λ=√(1−γ); A₁₂=ε≠0 ⇒ A≠λI ⇒ single eigenvector ⇒ **defective** (Jordan block, secular nλⁿ transients). Δ ≃ −4(2−γ)√(1−γ)h\*(h−h\*) ⇒ φ ∝ √(h−h\*) (the paper's exponent-½ onset).
4. **Overdamped half-life:** λ₊ = 1−δ in the char. poly.; constants cancel (1−(2−γ)+(1−γ)=0) leaving −γδ+(2−γ)h²/2=0 ⇒ λ₊ = 1−(2−γ)h²/(2γ)+O(h⁴/γ³) ⇒ **n₁/₂ ≈ 2γln2/[(2−γ)(εμ)²]** — the 2γ/(2−γ) coefficient falls out of the cancellation, nothing is tuned.
5. **Floor:** complex pair ⇒ |λ±|² = det A = 1−γ, **μ enters only the phase** ⇒ envelope (1−γ)^{n/2} ⇒ n₁/₂ = 2ln2/(−ln(1−γ)). That is *why* the law saturates: past h\* the determinant alone fixes the per-step contraction.
6. **Coset diffusion:** AR(1) momentum with FDT σ*²=F²Tγ(2−γ) ⇒ stationary Var(p)=F²T (equipartition); summed autocovariances give factor (2−γ)/γ; with t=Nε, **D_θ = εT(2−γ)/(2F²γ)**. (Assumption flag: derivation is Newtonian/flat-direction, consistent with the paper's own FDT scope note that the relativistic mode has no Gibbs invariant.)

### Numerical verification (script: `.claude/scratch/v2-derivation-appendix/check_derivations.py`; PREREG hit on every row)
| check | predicted (PREREG) | measured | verdict |
|---|---|---|---|
| closed-form A ≡ composed K·D·K·G map | exact | max dev 1.8e−15 over 1000 random draws | ✓ |
| det/tr identities | exact | 2.2e−16 / 4.4e−16 | ✓ |
| floor, γ=0.05 | 27.0268 (**paper: 27.03**) | \|λ\|-based 27.0268; time-domain peak-envelope fits 27.020–27.031 at h∈{0.05,…,1.0} — μ-independent | ✓ |
| overdamped slope d log n₁/₂/d log μ² | −1 (**paper fit −0.985 on ckpts**) | −1.0004 (deep overdamped), −1.0346 (band-filling grid) | ✓ |
| overdamped spot check γ=0.05, h=0.005 | closed form 1421.8 | map-eigenvalue 1408.2; direct iteration 1428 (dev 0.96% ≈ the stated O((h/γ)²)=1% expansion error) | ✓ |
| h\* (γ=0.05) | 0.0256431 exact; γ/2+2.57% | Δ(h\*)=0.0e0; double root 0.97467943=√0.95; eigvec cond 2.8e2→**2.9e9** at h\* (defective) | ✓ |
| φ onset exponent | 0.5 (**paper: 0.5165 on ckpts**) | 0.5010 (log-log fit, h−h\*∈[1e−6,1e−3]) | ✓ |
| stability λ=−1 at h=2 | exact, all γ | min\|λ+1\| = 0.0 at γ∈{0.01,0.05,0.2,0.5} | ✓ |
| latch transport | exact | \|q∞,iter − closed\| = 3.3e−16 | ✓ |
| D_θ (ε=0.05,T=0.1,γ=0.05,F=1) | 0.0975 | 0.09813 (ratio 1.0065, 4000 walkers × 20000 steps); Var(p)=0.0971 vs F²T=0.1 | ✓ |

Honesty notes: (i) PREREG P4 hand-arithmetic said h\*=0.025644; true value 0.0256431 — a 4th-decimal mental-rounding slip in the prereg itself, not in the derivation (the formula is identical; script evaluates it). (ii) No derivation disagreed with any published number — the STOP clause never fired. (iii) The task phrase "slope −1 in log n₁/₂ vs log μ" is strictly slope −1 **vs log μ²** (= vs log δ, the paper's Fig-1 axis, since μ²∝δ by GMOR); n vs log μ itself would be −2. The appendix and the paper both state it against μ²/δ, so there is no inconsistency in the submission.

### Flag-provenance table
| item | value |
|---|---|
| repo commit | n/a — no tracked code touched, no checkpoints loaded; analytic 2×2 maps only |
| script + seeds | `check_derivations.py`; `np.random.default_rng(0)` (matrix draws), `default_rng(7)` (diffusion MC) |
| env | main venv `/Users/user/Desktop/CHLU/.venv`, numpy 2.4.1; no JAX, no CHLU flags in effect |
| TeX | TeXLive 2026 pdflatex, jmlr.cls from texmf-dist; `refs.bib` copy from `scratch/v2-cite-pass/diagbuild/` |

### The file edit — single-insertion proof
- md5 chain of the LIVE file: `7bfbf114…` (my entry snapshot, 23:0x) → [Head's live typo pass: 8 hunks — "isrelationship"→"this relationship" l.38, "magnitutde(s)"→"magnitude(s)" ×7] → [my insertion] → [Head's live deletion of "$\pm0.05\%$ parameters," in `app:loan`] → [my compression re-edit] → **`aac3c6df150a5b3b2280daa5d911250b`** (verified unchanged at 2026-08-24 23:51:20).
- Reconstructed pre-my-edit state `condensed_paper.INTERMEDIATE.tex` md5 `1ad702b61b2b61d855e3d396fa7865f2`; frozen post-edit snapshot `condensed_paper.CURRENT.tex` md5 `aac3c6df…` ≡ live file.
- `git diff --no-index INTERMEDIATE CURRENT` (saved: `scratch/v2-derivation-appendix/insertion_final.diff`): **1 file changed, 33 insertions(+), 0 deletions; exactly one hunk `@@ -109,6 +109,39 @@`** = the new `\section{…}\label{app:derivation}` between `app:defs` and `app:curcos`. Nothing else touched by me — the 8+1 other hunks vs my entry snapshot are the Head's own concurrent edits, itemized above and NOT mine.
- Section internals: 4 numbered equations (`eq:block`,`eq:eigs`,`eq:hstar`,`eq:overdamped` — no label collisions, baseline had zero `eq:` labels), only paper-resident constants (27.03, γ/2, 2εμ, −1, ½, h<2), no `\usepackage`, no author token, no citation of the withdrawn theory note.

### Build (copies only, in `scratch/v2-derivation-appendix/build/`)
| | errors | pages | overfull |
|---|---|---|---|
| `paper_before` (pre-my-edit reconstruction) | 0 | 24 | 2 (29.09pt, 3.63pt — pre-existing tables) |
| `paper_after` (with appendix) | 0 | **25 (+1)** | 2 — byte-same sizes, the same pre-existing boxes; **zero new** |
`app:derivation` = Appendix B, starts p.10 directly under app:defs, ends ≈⅔ down p.11; `app:curcos` p.10→p.11; 0 undefined refs/citations. Visual render checked (pp.10–11).

## Pointer question — proposed `\ref` wiring, ⛔ NOT applied (Head rules separately)
1. Intro (current l.43): `…guided by an exactly-solvable underlying theory.` → `…guided by an exactly-solvable underlying theory (App.~\ref{app:derivation}).`
2. Setup (current l.58): `…is the exceptional point $h^*\approx\gamma/2$, where the block matrix becomes defective.` → `…where the block matrix becomes defective (derived in App.~\ref{app:derivation}).`
3. (Optional, verification sentence, Sec 4.1 l.66): `…serving as exact verification of the theory` → `…serving as exact verification of the theory (App.~\ref{app:derivation})`.

## Git footprint
None — no tracked files touched; all edits inside gitignored `.claude/NIPSsubmission/` (the Head's live artifact, edited per task authorization), reports/scratch under `.claude/`.

## Open questions / follow-ups / risks
- **Live-edit race:** the file is being edited concurrently by the Head. If any edit lands inside Appendix B or before Hub review, re-run the atomic proof (snapshot→strip-by-marker→`git diff --no-index`) rather than trusting the md5s above.
- The condensed paper still has **no `refs.bib` in its folder**; it builds only with the copy from `scratch/v2-cite-pass/diagbuild/`. Not mine to fix — flag to whoever owns the submission build.
- `h_f = √(2/(2−γ))(1+√(1−γ))` (the second discriminant root, = 2−O(γ²)) is a closed form not previously written in the paper — it is algebra, not a measurement, and is needed for the band statement to be exact; flagging for the Hub's zero-new-numbers audit trail.
- The two pre-existing overfull boxes (29.09pt table in `app:loan`, 3.63pt) predate this task.

## Proposed handover updates (for the Hub)
- §1/§7: `condensed_paper.tex` now carries **Appendix B `app:derivation`** (md5 `aac3c6df150a5b3b2280daa5d911250b`, 25pp) deriving all Sec-3 closed forms + D_θ; the referee gap "central closed forms asserted with no derivation" is closed for V2. The `\ref` wiring (3 one-liners above) awaits a Head ruling.
- §7 (env): the Head live-edits `NIPSsubmission/v2-neurreps/condensed_paper.tex` during agent sessions — any spoke editing it must snapshot-and-reconstruct to prove its own footprint (pattern in this report; two collisions absorbed cleanly here).
