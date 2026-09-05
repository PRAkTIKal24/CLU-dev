# PREREG — kt-csf3-tranche (experiment-engineer)

Written **before** the CSF3 tranche runs. My own task's acceptance criterion is
packaging (round-trip parity + smoke + conventions), which is not a measured
exponent — but this task *sizes* a run whose deliverables ARE exponents, and I
measured sizing probes on the laptop. Everything predicted here is therefore
committed before the A100 tranche executes.

Physics constants in force: `kappa=0.05`, `r*=1`, `J = 2 kappa r*^2 = 0.10`,
`T_KT = 1.786 kappa r*^2 = 0.0893` CLU units `= 0.8929 J` (measured 0.898 J).
**Not 0.1786** (retracted, factor 2). `float64`, `langevin_noise="fdt"`,
`newtonian_learned`, no governor.

---

## P1 — Round-trip parity (my acceptance gate). PREDICTED BEFORE RUNNING.
Promoting the five scratch scripts is packaging, so re-running the promoted code
at the laptop seeds/sizings reproduces `.claude/outputs/kt-2d-csf3/*.json`
**bit-exactly** (absdiff 0, not "within error"). Any nonzero difference means I
changed physics and must stop.

**Outcome: CONFIRMED.** absdiff `0.00e+00` on all four checks (§2 of the report).

## P2 — 2-D winding survival at L>=32 (soft exponent (a))
The AHNS law `tau ~ L^(pi rho_s/T - 2)` predicts a NEGATIVE exponent above
`T_KT`. At `L<=16` this was masked by vortex-diffusion traversal (`~L^2`).
Prediction, committed before the full-statistics run:

* **P2a** at `T/J = 1.10`: log-log slope over `L in {32,48,64}` is **< +0.5**
  (i.e. the laptop's apparent `+1.1` collapses once `L>=32`).
* **P2b** at `T/J = 1.30`: slope is **< 0** (the sign change is resolved).
* **P2c** below `T_KT` (`T/J = 0.60, 0.70`) the slope stays **strongly positive
  (> +2)** — memory still improves with size.

Derivation of the threshold in P2a/P2b: with the measured `rho_s`, AHNS gives
`pi(0.247/1.1) - 2 = -1.3` at `T/J=1.1`. I deliberately predict only the SIGN /
a loose bound, not `-1.3`, because first-passage censoring and the residual `L^2`
traversal both bias the measured slope UP; predicting the exact AHNS number
would be over-claiming.

**Laptop probe evidence already in hand (nwalk=4, seed 700, not full statistics):**
`tau_med` at `T/J=1.10` runs `45 (L=16, laptop) -> 89 (L=32) -> 97 (L=48) ->
85 (L=64)`. Slope over `L in {32,48,64}` is already `~ -0.06`, consistent with
P2a and on the edge of P2b. **This is a 4-walker probe and is NOT the result** —
it is the sizing evidence that the run is worth launching.

## P3 — 1-D clean `tau ~ 1/N` slope -1 (soft exponent (b)). **PREDICTED TO FAIL AS SCOPED.**
The scoped fix was "rerun at lower T (T/J=0.5) with long runs". I predict this
**will NOT recover slope -1**, for a reason that is an estimator defect rather
than a compute shortfall:

* **P3a** the MSD through-origin fit is **saturation-dominated** at the laptop's
  settings, so the fitted "rate" falls as the fit window lengthens.
  *Measured, same run/seed:* rate `2.5e-4` over `t<=2500` vs `4.0e-5` over
  `t<=50000` — a factor 6. CONFIRMED before the tranche.
* **P3b** lowering `T/J` 1.0 -> 0.5 changes the N=8 rate by **< 30%** and moves
  the N-slope **toward 0, not toward 1**. *Measured:* `4.04e-5 -> 3.15e-5`;
  slope over `N in {8,32}` `0.39 -> 0.15`. CONFIRMED before the tranche.
* **P3c** root cause: the ring winding is **barely metastable** at these
  parameters. `E_wind(N=8, w=1) = N J (1 - cos 2pi/N) = 0.234` vs `T = 0.10`, so
  `E/T = 2.3`; the winding relaxes in `~1e3` steps and there is no long-lived
  memory whose lifetime could scale as `1/N`. A well-posed diffusive window
  opens only at `T/J <= 0.2` (`E/T >= 11.7`), where MSD is still `<< 1` after
  `5e4` steps. CONFIRMED (probe table in the report §4).

**Consequence I commit to now:** if the tranche runs (b) as originally scoped
(`T/J=0.5`, longer runs, full-range MSD fit), it will return a slope in roughly
`[0.1, 0.5]`, i.e. `tau ~ N^-0.1..-0.5`, and this must **not** be read as
"the exponent is soft" — it is the estimator saturating. The honest routes are
(i) `T/J <= 0.2` **with** `--msd-fit-max 0.3`, or (ii) abandon MSD and measure
1-D winding **first-passage** `tau` (same estimator as the 2-D arm, making the
1-D-vs-2-D contrast apples-to-apples). I recommend (ii).

## P4 — What would falsify my packaging claim
Any of: a nonzero round-trip diff (P1); the `--quick` smoke failing to execute a
mode end-to-end; `postproc` failing to consume laptop-format JSON; the settings
guards not firing on `legacy` noise / relativistic kinetic / governor-on.
All four were checked and none falsified.
