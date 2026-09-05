# Q3 — pre-registered predictions for `r2-excursion-reach` (paste-able scorecard)

**Posted early, per the task's timing note.** Source: `readout-channel-theory` §Q3;
identical text is registered in `.claude/outputs/readout-channel-theory/PREREG.md` §3
(written before any measurement script existed). Derivation, assumptions and the
numerical checks are in `.claude/outputs/readout-channel-theory.md`.

**Theorist's headline for the engineer, in one line:** arm (a) is **real** (it is not a
free lunch that cancels under absolute read noise — it trades **1** dimension of value
resolution for **d_eff** dimensions of address packing, net `2^{d_eff−1}` per halving);
arm (b) is **bounded and small** (a hard ceiling `r ≤ Ψ(sep/2)`, worth **one K-rung**);
and **both saturate at d=4 near `r ≈ 0.4`** where the query-jitter floor takes over —
past that point the only remaining lever is `σ_q`, and moving `σ_q` is laundering.

---

## 0. The objects

| symbol | meaning | value in the shipped d=4 harness |
|---|---|---|
| `r` | payload-space Euclidean radius of the code (`= |a|max` at m=1) | 1.0 |
| `m` | number of payload channels | 1 |
| `σ_a` | **absolute** read-out noise on one payload coordinate | 0 (harness is noise-free) |
| `σ_q` | query address jitter (norm) | 0.15 |
| `s` | well width the READ sees | trained 0.30 (init 0.30) |
| `sep(d,K)` | min site separation, `∝ K^{−μ(d)}` | 0.903 (K=16), 0.710 (32), 0.549 (64), 0.451 (128), 0.363 (256) |
| `μ(d)` | packing exponent (measured from `designed_sites`) | 0.590 / 0.407 / **0.350** / 0.283 / 0.243 / 0.185 at d=2/3/**4**/5/6/8 |
| `Ψ(s)` | reach radius `= s·sqrt(2 ln(D/(2α|c|s)) − 1)`, `α=0.05`, `|c|≈0.98` | 0.88 at s=0.302, D=3.2 |
| `κ_R` | `Ψ(s)/s` — w25's `κ`, **a sqrt of a log** | 2.4–2.9 |

Feasibility of a cell = **`r ≤ Ψ(s_read)`** (reach) **and** `2s + c_jσ_q ≤ sep` (merge/jitter).

---

## 1. Scorecard — arm (a), multi-channel payload

| # | prediction | pass band | falsifier |
|---|---|---|---|
| **P3.1** ⭐ | Net item-capacity gain per **halving of `r`** at **fixed absolute** `payload_tol`/`σ_a`: `K_wall(r/2)/K_wall(r) = 2^{d_eff−1}` | **3.6× at d=4** (band 2–6×); at d=6, `2^{3.12}` = **8.7×** (band 4–14×) | ratio ≤ 1.2× ⇒ the lunch cancels; **stop the arm** (this is an acceptable acceptance) |
| **P3.2** ⭐ | `r(m) = zσ_a(K^{1/m} − 1)√m`; monotone ↓ for `m ≤ log₂K`; optimum `m* = ⌈log₂K⌉`; `r(m*) = zσ_a√(log₂K)` | at K=16 with the shipped per-axis spacing `Δ=2/15`: `r` = **1.000 / 0.283 / 0.133** at m = **1 / 2 / 4**. `m=8` at K=16 must NOT help | implemented `r` differs > 20 % from the formula, or m > log₂K helps |
| **P3.3** ⭐ | `K_wall(d,r)` solves `sep(d,K) = max(2r/κ_R, sep_jit)`, `κ_R ≈ 2.44`, `sep_jit ≤ 0.549` | d=4: `K_wall(1.0) ≈ **21**` (obs 16 P / 32 F ✓); `K_wall(0.5) ≈ **78**` (band 60–150; obs K=64 = 0.9922 ✓); **m=2 (r=0.283): strict ≥ 0.9 at K=64, and K=256 FAILS** | K=256 passes at d=4 |
| **P3.4** ⭐ | **Saturation.** Below `r ≈ 0.4` (d=4) the jitter floor binds and the wall stops moving | `K_wall(0.25)` and `K_wall(0.125)` within **1 K-rung** of `K_wall(0.5)` | walls keep moving at fixed `σ_q` ⇒ the jitter floor is wrong |
| **P3.8** | Turning on absolute read noise `σ_a` costs strict `1 − (2Φ(tol/σ_a) − 1)^m` | at `σ_a=0.05, tol=0.1`: **4.6 % (m=1) / 9.0 % (m=2) / 17 % (m=4)**, ±3 % | m-channel loss exceeds the per-axis-independent law ⇒ noise is correlated across channels |

**Design instruction implied by P3.1/P3.2 (the fairness-critical one).** The comparison is
only meaningful if the value criterion is an **absolute** latent-space quantity fixed once
for all arms: `payload_tol` (or `σ_a`) must **not** be rescaled with the codebook. w25's
`pscale` probe scaled both — which is exactly why it was "free there". With `tol` absolute:
- m=1, `r` halved ⇒ `K_value` halves (real cost), `K_addr` ×`2^{1/μ}` (real gain) → net win;
- m channels, `K = n^m` held fixed ⇒ **no value cost at all**, only the `√m`/`K^{1/m}` geometry.
⇒ **arm (a) with `m ≥ 2` dominates "just shrink the codebook", and it is the version that
survives the fairness rule.**

## 2. Scorecard — arm (b), annealed / continuation read

| # | prediction | pass band | falsifier |
|---|---|---|---|
| **P3.5** ⭐ | Hard ceiling **`r ≤ Ψ(sep/2)`**. The anneal's *entire* benefit is running the read at the merge-limited width `sep/2` while the write keeps its preferred `s` (gain `Ψ(sep/2)/Ψ(s_write) = 1.42×` at d=4 K=16) | annealed read moves `K_wall(d=4, r=1)` **16 → 32 and no further**. K=32 annealed strict **0.90–1.00** (from 0.824); **K=64 must still FAIL** (`Ψ(0.2745)=0.80 < 1`) | K=64 passes at r=1 under the anneal ⇒ ceiling wrong; K=32 unchanged ⇒ anneal inert |
| **P3.6** | Schedule: widen during the **address phase only**; be back at native width before the payload settles. Cross-talk bound `s_read ≤ 0.38` (d=4 K=16). Adiabaticity is a non-constraint: `\|ds/dt\| ≤ ω²s/η ≈ 10` vs the 20-time-unit phase ⇒ **shape irrelevant, only endpoints matter** | annealing **through** the read phase at `s_read = 0.5` costs payload abs-err **≥ 0.15** (vs ~1e−4) | annealing through the read phase is harmless ⇒ cross-talk model wrong |
| **P3.7** | (a) and (b) act on the **same** inequality ⇒ **not additive**. Anneal on top of arm (a) adds ≤ 1 rung | ≤ 1 rung | ≥ 2 rungs |

**Cheapest decisive cell for arm (b):** d=4, K=32, r=1, `s_read` swept over
{0.30 (native), 0.38, 0.45, 0.55}, 3 seeds, written landscape only. Predicted strict:
0.824 → ~0.95 (0.38–0.45) → **falls again** at 0.55 (merge: `2s = 1.10 > sep = 0.710`).
A **non-monotone** strict-vs-`s_read` curve with a peak at `s_read ≈ sep/2` is the
signature that confirms both halves of the mechanism at once; a monotone curve refutes it.

## 3. What the theory says NOT to spend compute on

1. **Any excursion arm that scales `payload_tol` with the codebook.** Provably free
   (`r/σ_a` is the value capacity; scaling both holds it fixed), and it is the laundering
   the Head's fairness rule targets. It measures nothing about the primitive.
2. **`r < 0.25` at d=4** (P3.4): the jitter floor binds; the extra compute buys ≤ 1 rung.
3. **Annealing past `s_read ≈ sep/2`** (P3.5/P3.6): merge and payload cross-talk both bite;
   predicted non-monotone.
4. **Trying to buy reach by making the wells deeper.** `κ_R = sqrt(2 ln(D/(2α|c|s)) − 1)`:
   going from `κ_R = 3` to `4` needs `D` ×33; to `6`, `D` ×7·10⁵. **Reach is
   logarithmically un-buyable in the well depth** — which is why the excursion (and `s`,
   and `α`) are the only levers with leverage.
5. ⭐ **One cheap arm the task did not list, ranked above (b):** `learned_confine`
   `α: 0.05 → 0.022`. `Ψ ∝ sqrt(2 ln(1/α) + const)`, so halving `α` raises the reach
   radius by **1.13×** and *costs no discrimination at all* (α does not enter the merge
   condition). One flag, no new code, 3 seeds. Predicted: strict at d=4 K=16 width 0.15
   rises from 0.594 to **0.66 ± 0.06** (reach fraction `Ψ` 0.56 → 0.64). If that fires,
   the (R) mechanism is confirmed by an intervention on the *confinement* rather than on
   the excursion — an independent causal test of the same inequality, for ~1 h of compute.
   ⚠ Check first that `α` is not load-bearing for coercivity elsewhere (F5 Prop-10) and
   that the value-blank control still passes.

---

## 4. ⚠ POST-MEASUREMENT UPDATE (appended after the numerics; §1–§3 above are left verbatim so they can still be scored as registered)

**(i) The criterion changed shape — the *predictions* above stand, their *derivation* is upgraded.**
`Ψ(s) = s√(2 ln(D/(2α|c|s)) − 1)` is systematically **1.36× conservative** (measured ratio
`a*/Ψ = 1.357 ± 0.064` over 7 shipped-integrator sweeps). The exact criterion is the **saddle
condition (U)**: capture ⟺ `|a_i| < R₂`, the middle root of
`(D R/s²)e^{−R²/2s²} = 2α(L − R)` with **`L = √(|c_i|² + a_i²)`** — matched to **1.013 ± 0.025** on
the toy and **31/32 items** on the trained shipped `V` with **zero free parameters**. Use
`.claude/scratch/readout-channel-theory/criterion_U.py::a_ceiling(|c|, s, D, alpha)`, not `Ψ`.
Measured inputs for d=4: `D` = **0.910** at `s_fit` = 0.320 (init width 0.30) and **0.459** at 0.200
(init width 0.15); `a_U` = **1.06** vs `|a|max` = 1.0 — the shipped store sits 6 % inside the wall.

**(ii) P3.5 / P3.6 (arm b) — already measured on the trained shipped `V` at d=4 K=16.** Widening the
atom widths during the **address phase only** (`s_j → √(s_j²+s_a²)`, native widths for the read):

| `s_eff` | `2s_eff/sep` | strict (native `s`=0.184) | strict (native `s`=0.301) | value err if widened **through** the read |
|---|---|---|---|---|
| native | — | 0.5859 | 0.9473 | — |
| 0.310 / 0.392 | 0.69 / 0.87 | **0.8223** | 0.9023 | 3.2e−3 / 7.9e−3 |
| 0.395 / 0.462 | 0.88 / **1.02** | 0.7637 | **0.2520** | 9.7e−3 / **1.7e−1** |
| 0.533 | 1.18 | 0.0625 | 0.0625 | 3.1e−1 |

⇒ **P3.5 confirmed in kind:** the anneal helps **only where reach binds** (+0.236 strict at the
narrow width, +0.004 at the wide one), and the ceiling is the merge condition **`2s_eff = sep`**,
confirmed to one grid step. **P3.6 confirmed:** annealing through the read phase costs 100–2000× in
value error. **The K=32 wall test (16 → 32 rung) is still unrun and is yours.**

**(iii) A warning for the arm-(a) sweep, measured on a designed-Gaussian surrogate.** With
`payload_tol` scaled the w25 way (relative precision fixed), d=4 K=32 goes
0.763 (r=1) → **1.000** (r=0.5) → **0.844** (r=0.283) → 0.842 (r=0.25): **non-monotone**. Below
`r ≈ 0.3` the read's own settling-error floor eats the shrinking tolerance. **Hold `payload_tol`
absolutely fixed** (§1) or you will measure this artefact instead of the capacity.

**(iv) A constant-width Gaussian surrogate does NOT reproduce the trained cells** (5/11), because the
write **adapts the per-item width to the payload excursion** (`corr(s_fit, |a_i|) = +0.82`;
far-payload wells 1.31× wider). If your arm changes the excursion, expect the *write* to re-solve the
width too — measure `trained_well_widths` per item in every cell, not just the median.
