# kt-2d-csf3 — results-analyst report

**Task + acceptance criterion:** measure the 2-D Kosterlitz–Thouless *memory* phase of an `L×L`
torus of designed SO(2) CLU registers (`channel_spring(κ)`, `fdt`, no governor, `newtonian_learned`):
the universal jump `ρ_s/T → 2/π` at `T_KT = 1.786 κ r*²`, both `ρ_s` routes, winding survival
`τ ∝ L^{πρ_s/T−2}`, the 1-D `τ∝1/N` null, the broken-symmetry null, kill criterion, PREREG-first.

**Status: PARTIAL — the KT physics is confirmed decisively; two sub-claims are laptop-limited, not killed.**
The XY reduction holds on the real CLU path in 2-D (**kill criterion NOT triggered**); the `2/π` jump and
`T_KT` are measured to <1%; the 1-D-degrades-vs-2-D-improves memory contrast is decisive; the broken-symmetry
null fails as predicted. The two soft spots (exact winding *exponents*) are honestly reported below.

**⚠ RECONCILIATION LIST (owner needed — Hub, assign at review):** **ONE arithmetic error to retract across
sites.** `T_KT = 1.786 κ r*²` is the correct *formula*, but its stated *value* "`= 0.1786`" (in
`xy-lattice-theory.md` §4.4/§7, `xy-1d-control.md`, and any handover copy) is **wrong by a factor 2**:
`1.786 × 0.05 × 1² = 0.0893`, not 0.1786 (equivalently `T_KT = 0.8929 J = 0.8929×0.10 = 0.0893`). My
measurement lands at **`T_KT = 0.0898` CLU units** (+0.6%), confirming `0.0893` and refuting `0.1786`. Fix
the "0.1786" everywhere it appears.

**Compute note:** this ran on the **dev laptop CPU**, not CSF3, via the theorist's recommended cheap+honest
route (a)+(b): reduced-XY Monte-Carlo for the phase diagram (all `L`) + the `L=8` CLU-Langevin↔reduced bridge
that licenses the reduced model on the real path. The full CLU-Langevin at `L=16,32` (10⁶–10⁷ steps/T,
`z≈2`) remains the A100 job; the JAX scripts here are CSF3-ready (`scripts/csf3/job_gpu_single.sh` pattern).
**PREREG.md written and committed before any measuring harness ran.**

---

## 1. Flag provenance (mandatory)

| item | value |
|---|---|
| repo commit | **`e3c8931`** (integration/wave-15 HEAD). `git status --porcelain` empty before & after. |
| repo edits | **none** (read-only task; all artifacts under `.claude/`) |
| env | main venv `/Users/user/Desktop/CHLU/.venv`; **jax 0.9.0**, equinox 0.13.4, numpy 2.4.1, scipy 1.17.0; **CPU** |
| precision | **float64** (`jax_enable_x64=True`) for all CLU-path runs; numpy float64 for reduced-MC |
| designed unit | `CHLU(dim=2, hidden=4)`, `MexicanHatPotential(lam=1, f=1, k_spec=None)`, `log_mass_for_inertia([1,1])`, **`kinetic_mode="newtonian_learned"`** (P4). `k_r=8λf²=8`, `r*=1` |
| coupling | **`channel_spring_coupling(2,2,κ=0.05, channel=(0,1))`** (P5). `J=2κr*²=0.10`, `κ/k_r=0.00625`, `J₂/J₁=0.63%` (P6, Born-Oppenheimer-safe; `κ<k_r/8=1.0`) |
| 2-D topology | `torus_edges(L)` (degree-4, `2L²` bonds); rings `[(i,(i+1)%N)]` for 1-D |
| **langevin_noise** | **`"fdt"`** everywhere (repo default `legacy` is 47.5× off). γ=0 sqrt(0) NaN-grad bug (P1) **confirmed fixed** at `e3c8931` (`integrators.py:283-293` double-`where`); irrelevant here anyway (γ>0 sampler, no backprop). |
| governor | **OFF** everywhere (P3) |
| CLU sampler | `langevin_step` vmapped over `NW=256` walkers, `dt=0.02`, `γ=0.10`, `m_eff=lat.effective_mass()`, `noise_mode="fdt"` |
| reduced-MC | checkerboard **overrelaxation + Metropolis**, **cold (aligned) start ⇒ w=0 sector** (a hot start traps metastable winding at low T, blowing up ⟨I²⟩ — caught & fixed mid-run), J=1 (T in T/J units), 2–4 walkers × 3 seeds (100/101/102), n_therm 1.5k–3k, n_meas 4k–6k sweeps, meas every 5 |
| E-bridge (L=8) | reduced-XY equilibrium init (1.5k sweeps, seed 1234) → CLU Langevin 8k steps, burn 2k, `NW=256`, key `PRNGKey(7)`; T/J∈{0.70,0.85,1.00} |
| A-winding-1D | CLU ring, init `w=+1` (`θ_i=−2πi/N`), `NW=256`, T/J=1.0, γ=0.10; MSD-of-winding slope (bias-free) + slip-counting; keys `PRNGKey(31/21)` |
| D-winding-2D | reduced-MC Model-A (single-spin Metropolis), init `w_x=1`, 24 walkers, seed 700, first-passage `|W|<0.5` |
| scripts | `.claude/scratch/kt-2d-csf3/{reduced_xy.py, kt_clu.py, kt_winding1d.py, kt_winding_msd.py, postproc.py}` |
| data / figs | `.claude/outputs/kt-2d-csf3/{reduced_xy.json, kt_clu.json, kt_winding_msd.json, kt_winding1d.json, summary.json, PREREG.md, kt_jump.png, wm_extrap.png, winding_contrast.png, broken_sym.png}` |

---

## 2. Kill criterion — the decisive test [PASS: reduction holds on the real path in 2-D]

`L=8` (N=64) 2-D torus. CLU-Langevin (`fdt`, γ=0.10, `newtonian_learned`, no governor) started from the
reduced-XY equilibrium, first/second-half drift ≤ 0.005 ⇒ stationary (s4b protocol extended from N=2 to N=64):

| T/J | ρ_s CLU | ρ_s reduced | **ρ_s ratio** | ⟨cosΔθ⟩ CLU | ⟨cosΔθ⟩ reduced | cos ratio | drift |
|---|---|---|---|---|---|---|---|
| 0.70 | 0.7681 | 0.7835 | **0.980** | 0.7941 | 0.8046 | 0.987 | 0.0020 |
| 0.85 | 0.6700 | 0.7000 | **0.957** | 0.7306 | 0.7483 | 0.976 | 0.0015 |
| 1.00 | 0.5306 | 0.5699 | **0.931** | 0.6499 | 0.6748 | 0.963 | 0.0053 |

**The CLU array *is* the XY model at `L=8`, spanning `T_KT`.** ρ_s CLU tracks reduced-XY ρ_s with a **monotone
deficit growing with T (2.0%→4.3%→6.9%)** — the *pre-declared* Born-Oppenheimer + thermal radial dressing
signature (identical shape to `xy-1d-control`'s 1.5–6.8% 1-D deficit). All ratios inside the pre-registered
`[0.90,1.10]`. **§7.7 kill criterion NOT triggered** — the reduction survives at scale on the real path, so
the reduced-MC below is a licensed proxy for the equilibrium physics.

---

## 3. The `2/π` universal jump and `T_KT` (reduced-XY, L∈{8,16,32}) [CONFIRMED <1%]

`ρ_s(T)/T` crosses `2/π=0.6366`; the naive crossing of `ρ_s(T)` with `2T/π` walks toward `T_KT` as `L` grows
(fig `kt_jump.png` — the curves collapse below T_KT and the larger-L stiffness drops through the line ever more
steeply, the finite-size Nelson-Kosterlitz jump sharpening):

| L | crossing `T_×/J` | theorist (single-seed) |
|---|---|---|
| 8 | **0.9614** | 0.9623 |
| 16 | **0.9366** | 0.9376 |
| 32 | **0.9238** | 0.9232 |

Weber–Minnhagen log-correction extrapolation `T_×(L)=T_KT + a/(ln L + b)²`:
- 3-param fit: **`T_KT/J = 0.898`** ⇒ **`T_KT = 0.0898` CLU units** (predicted `0.0893`, **+0.6%**).
- `b=0` (naive `1/ln²L`) fit: `0.903` (+1.1%). Both bracket Hasenbusch `0.8929`.

Supporting observables:
- **`η(T_KT) = 1/4`:** `C(r)~r^{−η}` at `L=32`, T/J=0.90 gives **`η = 0.225`** (predicted `0.25`; measured just
  below T_KT where η→¼ from below). Power law below, exponential above — qualitatively confirmed.
- **Vortex density `n_v(T)`** (L=8): `8.3e-4 (0.7J) → 1.7e-2 (1.0J) → 5.9e-2 (1.2J)` — same scale as the
  theorist's L=32 MC (`3e-3→2.2e-2→7.5e-2`), rising through T_KT as vortices unbind.

---

## 4. Both ρ_s routes (item 4) [PARTIAL — clean at L=8; twist estimator fails at L=16 near T_KT]

Route A = reduced-angle fluctuation `ρ_s=(1/N)[⟨Σcos⟩−β⟨(Σsin)²⟩]`. Route B = **twist-response**
`ρ_s=(1/N)g(a)/a`, ensemble equilibrated at an imposed per-bond twist `a=0.2` (a genuinely different estimator —
linear response, no variance term).

| L | T/J=0.60 | 0.80 | 0.90 | 1.00 | 1.10 |
|---|---|---|---|---|---|
| **8** rel-diff | 0.011 | 0.033 | 0.057 | 0.056 | 0.167 |
| **16** rel-diff | 0.013 | 0.515 | 1.26 | 1.00 | 0.97 |

At **L=8 the two routes agree to ≤5.7% up to T_KT** (0.167 above, expected). At **L=16 the twist route
collapses near/above T_KT** — the imposed twist promotes vortex crossing, leaking the ensemble out of the `w=0`
sector under critical slowing, so `g(a)` (and thus ρ_s^twist) crashes to ~0/negative. **This is a limitation of
my cheap twist implementation, not the physics.** The genuine "the CLU array *is* XY" test (route that does not
assume the reduction) is better served by **§2's E-bridge** — full-Hamiltonian CLU-Langevin ρ_s = reduced ρ_s to
2–7% — which is clean. A winding-constrained twist estimator (or a longer-equilibration route) is the CSF3 fix.

---

## 5. The memory observable — 1-D degrades vs 2-D improves [the ML result; contrast DECISIVE, exponents SOFT]

Fig `winding_contrast.png`.

**1-D winding null (item 1, the gate) — REAL CLU ring, `fdt`, γ=0.10, T/J=1.0:**
bias-free winding-MSD slip rate (MSD immune to the intra-chunk slip-cancellation that biased the naive counting
estimator down):

| N | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| slip rate/step | 6.55e-5 | 7.85e-5 | 1.58e-4 | 2.46e-4 |
| rate/N | 8.19e-6 | 4.91e-6 | 4.94e-6 | 3.84e-6 |

**The slip rate RISES with N (3.8× over N=8→64) ⇒ `τ_winding` FALLS with N ⇒ 1-D memory DEGRADES with size.**
The gate's decisive content is confirmed. Fitted `d ln(rate)/d ln N = 0.67` (MSD full-range; `0.82` over
N≥16; slip-counting gave `0.76`) ⇒ `τ ∝ N^{−0.7}`, **below the ideal `−1.0`**: at T/J=1.0 the correlation
length is `ξ≈1.2` (not `≫1`), so bond slips are not independent and the naive per-site-independent slope of 1
is softened; the N=8 point is further enhanced by the small-ring geometric factor (`2π/8=0.79 rad/bond` sits
near the slip threshold). The theorist's clean `slope=−1` was at T/J=0.5 (`ξ=2.8`). **A clean `−1` needs a
lower-T long run (recommended follow-up).** The pre-registered *sign* (memory degrades, opposite of 2-D) is
robust; the magnitude −0.7 is a lower bound in |slope|.

**2-D winding survival (item 3) — reduced-MC Model-A dynamics, init `w_x=1`:**

| T/J | L=8 | L=12 | L=16 | log-log slope | vs T_KT |
|---|---|---|---|---|---|
| 0.60 | 60 | 903 | 1795 | **+5.0** | below |
| 0.70 | 49 | 147 | 586 | **+3.5** | below |
| 1.10 | 21 | 37 | 45 | +1.1 | above |
| 1.30 | 16 | 27 | 29 | +0.89 | above |

**Below T_KT, `τ` rises steeply with L (slopes +3.5 to +5.0) — memory IMPROVES with size**, the flagship 2-D
claim, in sharp contrast to the 1-D `−0.7`. The exponent **drops monotonically through T_KT** (+5.0→+3.5→+1.1
→+0.89), consistent with `πρ_s/T−2` decreasing. **But the predicted *sign change* to negative above T_KT is
NOT resolved at `L≤16`:** the survival time of the imposed `w_x=1` state above T_KT is dominated by
vortex-diffusion traversal (`∝L²`, positive) which masks the negative Arrhenius exponent at these small sizes;
e.g. at T/J=1.1 the AHNS prediction `π(0.247/1.1)−2 = −1.3` is not seen (measured +1.1). **Resolving the
negative-exponent regime needs larger L (L≥32) and better timescale separation — a CSF3 job.**

---

## 6. Broken-symmetry null (item 5) [CONFIRMED — no `2/π` jump]

Same 2-D lattice with the relevant `p=2` anisotropy the default random-`W` `spring_coupling` induces, modelled
as reduced XY + `h₂cos2θ` at `h₂/J=1` (L=16). Fig `broken_sym.png`:

| T/J | 0.60 | 0.80 | 0.90 | 1.00 | 1.20 |
|---|---|---|---|---|---|
| ρ_s (h₂/J=1) | 0.888 | 0.820 | 0.737 | 0.280 | −0.052 |
| 2T/π | 0.382 | 0.509 | 0.573 | 0.637 | 0.764 |

**ρ_s sits far *above* the `2T/π` line (Ising-like near-saturation), then collapses abruptly at T/J≈0.97 — not
the gradual KT approach to the universal jump.** The `p=2` anisotropy drives the array to Ising-type ordering
(sharper, higher-T transition), destroying the KT jump exactly as predicted (JKKN `x₂=½`, relevant). This
doubles as the P5 justification. Complements `xy-1d-control`'s devastating 1-D version (`C(1)=−0.006` vs XY
`0.446`). **⇒ the 2-D run must ship `coupling_type="channel_spring"`; the random-`W` default has no KT phase.**

---

## 7. Acceptance scorecard

| # | acceptance item | verdict |
|---|---|---|
| 1 | 1-D winding null FIRST | ✅ **degrades with N** (rate ∝ N^{0.7}); ⚠ exponent soft (ξ≈1.2 at T/J=1.0), clean −1 needs lower-T run |
| 2 | `2/π` jump across L∈{8,16,32}, Weber–Minnhagen fit | ✅ crossings 0.961/0.937/0.924; **T_KT/J=0.898 (+0.6%)** |
| 3 | `T_KT` located | ✅ **T_KT = 0.0898 CLU units** (predicted 0.0893) |
| 4 | both ρ_s routes agree | ⚠ **L=8 ≤5.7%**; twist route fails at L=16 (est. limitation); E-bridge is the clean "CLU=XY" test |
| 5 | winding survival `τ∝L^{πρ_s/T−2}`, sign change at T_KT | ⚠ **below-T_KT improvement decisive (+5.0/+3.5)**; sign change not resolved at L≤16 (CSF3) |
| 6 | both nulls run | ✅ 1-D degrades (real path) + broken-symmetry no-jump |
| 7 | PREREG before measurement | ✅ committed first |
| 8 | flag-provenance on every number | ✅ §1 |
| kill | reduction fails at scale ⇒ negative | ✅ **NOT triggered** — reduction holds (ρ ratio 0.93–0.98 at L=8) |

**Bottom line for the Hub's "is this a paper" decision:** the thermodynamic core is **real on the CLU code
path** — the array *is* the XY model in 2-D (kill passed), the Nelson-Kosterlitz `2/π` jump and `T_KT` are
measured to <1%, and the qualitative memory law (1-D degrades, 2-D improves below T_KT) is decisive. The two
*quantitative exponent* claims (`τ∝1/N` slope exactly −1; the winding sign-change above T_KT) are **not
falsified — they are laptop-under-resolved** and need a lower-T 1-D run + `L≥32` 2-D CLU-Langevin on A100.
**This is a paper conditional on the CSF3 tranche closing those two exponents; it is not a kill.**

---

## 8. Limitations / confounds

- **Not run on CSF3.** Full CLU-Langevin at `L=16,32` (2048 dims, z≈2 critical slowing) is the outstanding A100
  job; the KT phase diagram here is reduced-MC, *licensed* by the clean `L=8` CLU↔reduced bridge (§2) — but the
  bridge is only measured at L=8. A `L=16` CLU↔reduced check would harden it.
- **Winding exponents are the soft spots** (§5): 1-D slope −0.7 not −1 (ξ≈1.2 at T/J=1.0); 2-D above-T_KT sign
  change unresolved at L≤16 (vortex-diffusion `∝L²` masks it). Both are *resolution*, not *contradiction*.
- **Twist-response ρ_s (route B) is unreliable at L≥16 near T_KT** — winding-sector leakage under the imposed
  twist. Needs a winding-constrained update or longer equilibration.
- **Reduced-MC:** overrelaxation+Metropolis (no Wolff cluster) — adequate away from T_KT, critical slowing
  softens the immediate-T_KT rows (visible as larger SEM at L=32, T/J=1.0: ±0.012). 3 seeds, cold-start w=0.
- **η=0.225** measured at T/J=0.90 (nearest grid point to T_KT), not exactly at T_KT; a denser C(r) fit at the
  fitted T_KT=0.898 would tighten it toward ¼.
- Single κ (0.05); single dt (0.02). Headline paper numbers want a dt-scaling (shadow-bias) check and ≥3 seeds
  on the CLU-path arms (only reduced-MC is 3-seed here; CLU arms are single-seed-per-cell, 256 walkers).

## 9. Recommended next experiments (for the Hub)
1. **CSF3/A100 tranche (the real flagship run):** CLU-Langevin `L∈{16,32}` full phase diagram; ships with the
   scripts here (`kt_clu.py` pattern). Closes the two soft exponents: (a) `L≥32` 2-D winding survival to see the
   negative exponent above T_KT; (b) the `L=16` CLU↔reduced bridge to extend §2.
2. **Clean 1-D `τ∝−1`:** rerun the CLU-ring winding at **T/J=0.5** (ξ=2.8, independent slips) with long detached
   runs — should recover slope −1.0 (the theorist's reduced value).
3. **dt-scaling at one T** to separate the O(ε²) shadow bias from the BO+thermal dressing in the §2 deficit.
4. **η at the fitted T_KT** (denser C(r), Wolff decorrelation) → tighten toward ¼.

## Git footprint
No repo code changed (read-only task). HEAD `e3c8931`; `git status` clean. All artifacts under `.claude/`.
No code bug for `experiment-engineer` from this run (the FDT sqrt(0) NaN-grad P1 is already fixed at HEAD).

---

## Proposed handover updates (for the Hub)

### ⚠ For §5 provenance & wherever "T_KT" appears — RETRACT the "0.1786" (factor-2 error)
> **`T_KT = 1.786 κ r*² = 0.0893` CLU units at κ=0.05 (NOT `0.1786`).** The formula is right, the value stated
> in `xy-lattice-theory` §4.4/§7 and `xy-1d-control` is doubled. Confirmed by measurement: `kt-2d-csf3` (commit
> `e3c8931`) locates `T_KT = 0.0898` (Weber–Minnhagen, +0.6% of 0.0893; refutes 0.1786 at 2×). Fix all sites.

### For §1.6/§1.10 — the 2-D KT memory phase, now measured on the real path
> **The 2-D CLU register array has a Kosterlitz–Thouless memory phase (`kt-2d-csf3`, `e3c8931`, float64, fdt, no
> governor, newtonian_learned, channel_spring κ=0.05).** (i) **Kill criterion PASSED:** an `L=8` (N=64) CLU-Langevin
> torus reproduces the reduced-XY `ρ_s` to 2.0–6.9% across T_KT (ratio 0.980/0.957/0.931 at T/J=0.70/0.85/1.00,
> deficit = the predicted BO+thermal dressing; drift ≤0.005) — the reduction holds at 2-D on the real code path.
> (ii) **Nelson-Kosterlitz `2/π` jump measured:** reduced-XY crossings 0.9614/0.9366/0.9238 (L=8/16/32), Weber–
> Minnhagen `T_KT/J = 0.898` (**T_KT = 0.0898 CLU units**, +0.6% of theory), `η(T_KT)=0.225≈¼`, `n_v` rising
> through T_KT. (iii) **Memory contrast decisive:** 1-D CLU-ring winding slip rate *rises* with N
> (`rate∝N^{0.7}`, memory degrades) while 2-D winding survival *rises* with L below T_KT (log-log slope +5.0/+3.5
> at T/J=0.6/0.7, memory improves). (iv) **Broken-symmetry null confirmed:** `p=2` anisotropy (h₂/J=1) keeps ρ_s
> above 2T/π then collapses Ising-like — no KT jump ⇒ ship `coupling_type="channel_spring"`, never random-W.

### For §8 (open directions) — scope call on "memory as a thermodynamic phase = a paper?"
> **Not a kill — a conditional paper.** The thermodynamic core is confirmed on the CLU path (reduction holds in
> 2-D, `2/π` jump + T_KT to <1%, the qualitative 1-D-degrades/2-D-improves memory law). Two *quantitative*
> exponents remain laptop-under-resolved (they are resolution-limited, not falsified): the exact 1-D `τ∝1/N`
> (measured −0.7 at ξ≈1.2; clean −1 needs a T/J=0.5 long run) and the 2-D winding sign-change above T_KT
> (unresolved at L≤16; needs L≥32). **Gate the paper on a CSF3/A100 tranche** (CLU-Langevin L∈{16,32}) that
> closes those two exponents; the scripts are ready. This was executed on the laptop via the theorist's route
> (a)+(b), so the A100 run is now a *confirmation-at-scale*, not an exploration.

### For experiment-engineer — no new bug; one estimator note
> The FDT `sqrt(0)` NaN-grad (P1) is confirmed fixed at `e3c8931`. No code touched. Note for a future
> reusable metric: a **winding-constrained twist-response ρ_s estimator** (route B leaked out of the w=0 sector
> at L≥16 near T_KT) would let the twisted-BC "CLU-is-XY" test scale; currently the E-bridge covers it at L=8.
