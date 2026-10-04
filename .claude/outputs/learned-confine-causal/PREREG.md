# PREREG — `learned_confine` 0.05 → 0.022: an independent causal test of the saddle criterion (★)

**Written before any measurement script for this task existed.** Base local `main @ 082d095`,
JAX 0.9.0 (main venv). Author: physics-theorist, w27, campaign tag [C1W27].
Everything below is derived from (a) the saddle criterion (★) of `readout-channel-theory` §1 and
(b) **already-existing** w26 data (`.claude/scratch/readout-channel-theory/tr_d4K16_w015.json`,
seed 0) — no new run has been made at the time of writing.

## 0. Dial declaration (protocol §7)
- **Dial:** none — theory/instrument. Causal test of a criterion; **not** a performance claim and
  must not be scored on a performance axis.
- **Laundering control:** n/a (no performance number). Substitute, per the task: this document.
  Plus a **value-blank** control at both α (P-G).
- **Falsifies:** Δstrict ≤ 0, or Δstrict outside the registered band (§3).
- **Does NOT falsify:** a smaller-than-predicted but right-signed, in-band effect; the absolute
  retrieval level; the |a|=0.7333 rung landing anywhere in its (deliberately wide) band.

## 1. The intervention and why it is a *causal* test
Criterion (★) (`readout-channel-theory` §1.0): for a well of depth `D`, width `s`, at
`z_i = (c_i, a_i)` inside a confinement `α|q|²`, every stationary point of `V` lies on the ray
`0 → z_i`, at distance `R` from `z_i` solving

```
(D·R/s²)·e^{−R²/2s²} = 2α(L − R),        L = |z_i| = √(|c_i|² + a_i²)          (★)
```

and the query (launched at `(c_i, 0)`, i.e. a distance `|a_i|` from the site) is captured **iff**
`|a_i| < R₂`, the middle root (the saddle). The confinement enters **only** through the product
`2α·(L−R) ≈ 2α|c_i|`. So α is *not* a free knob: it is one of the four algebraic levers on the
ceiling (`s`, `|a|max`, `α`, `|c|`), and halving it must move the reach ceiling `a_U` by a
**computable** amount. w26 observed this only correlationally (a toy sweep, §4.3: α 0.05→0.025 moved
`a*` +9.1 %). This task intervenes on the shipped learned store.

**The alternative hypothesis being tested against.** If `learned_confine` acts on retrieval through
something *other* than `2α|c_i|` in (★) — e.g. only through the write's conditioning, or not at all —
then Δstrict is ≈ 0 (or negative), and (★)'s status drops from *causal law* to *correlation*.

## 2. Derived quantities (from (★), zero free parameters except the transfer function of §2.2)

### 2.1 The ceiling shift
Feeding the w26 seed-0 per-item fits `(D_i, s_i, |c_i|)` (radial fit R² ≥ 0.9994 at all 16 sites)
into `criterion_U.a_ceiling`:

| α | `a_U` (median over the 15 ball sites) |
|---|---|
| 0.05 | 0.632 |
| 0.022 | 0.690 |

**P-D. `a_U(0.022)/a_U(0.05) = 1.093 ± 0.005`** at every ball site (measured spread over items
1.089–1.098); the one interior site (item 13, `|c| = 0.064`) gives **1.050**. This is the primitive
prediction: **a 2.27× cut in α buys only +9.3 % of reach.** (Consistency check: w26's toy measured
+9.1 % for a 2.0× cut at `s`=0.30, `D`=1.0.) Reach is *algebraically* — not proportionally — coupled
to α, exactly as the `√(2 ln β)` structure requires.

### 2.2 From the ceiling shift to retrieval (the transfer function)
Retrieval per item is not a step in `|a_i|` because the queries are jittered
(`query_sigma` 0.15, `fixed_norm`). Define `x_i ≡ |a_i| / a_U(|c_i|, s_i, D_i, α)`. On the α=0.05
cell the 16 items give a **very sharp** logistic in `x`:

```
Φ(x) = σ((x₀ − x)/w),     x₀ = 1.0022,  w = 0.0306          (10–90 % width = 0.134 in x)
```

`x₀` and `w` are **fitted on the two partial items of the α=0.05 cell** (item 2: x=0.933,
strict 0.906; item 9: x=1.006, strict 0.469) — **declared as a fit, not a derivation.** The other 14
items are saturated (x ≤ 0.762 or ≥ 1.12) and are reproduced exactly. Φ reproduces the α=0.05 cell
mean at **0.5875 vs measured 0.5859**. Note `x₀ = 1.002`: the criterion's own threshold, recovered
from the data to 0.2 %.

**The prediction is `Φ` held fixed and only `a_U` moved by α.** That is the whole content of the
test: nothing about the transfer is allowed to change.

### 2.3 Point prediction
Per-item `x_i` at α=0.022 (same fits), and `Φ(x_i)`:

| item | `|a|` | `x` @0.05 | `x` @0.022 | measured @0.05 | **predicted @0.022** |
|---|---|---|---|---|---|
| 6,13,10,11,3,7,5,1 | ≤ 0.4667 | ≤ 0.762 | ≤ 0.697 | 1.000 | **1.000** |
| 2 | 0.6000 | 0.933 | 0.854 | 0.906 | **0.992** |
| 9 | 0.6000 | 1.006 | 0.917 | 0.469 | **0.941** |
| 0 | 0.7333 | 1.123 | 1.028 | 0.000 | **0.299** |
| 12 | 0.7333 | 1.151 | 1.052 | 0.000 | **0.163** |
| 14, 8 | 0.8667 | 1.319 / 1.483 | 1.209 / 1.353 | 0.000 | **0.001 / 0.000** |
| 4, 15 | 1.0000 | 1.550 / 1.576 | 1.417 / 1.439 | 0.000 | **0.000** |

**Cell mean predicted: `strict(α=0.022) = 0.650`.**

### 2.4 Baseline
Matched baseline is measured **in this task** at α=0.05, same 3 seeds, same script. The external
anchor is `r2-geometry-revival` §1 (3 seeds, shipped `evaluate_arm_cell`): **0.5944 ± 0.0164**
(per-seed 0.5859 / 0.6133 / 0.5840, **population** sd). w26's `fit_reach.py` reproduces its seed-0
value bit-for-bit (0.5859). Predicted per-seed sd ≈ 0.016 ⇒ SE of the 3-seed mean ≈ 0.010, SE of the
paired difference ≈ 0.013.

**⇒ P-1 (headline): `Δstrict = strict(0.022) − strict(0.05) = +0.064`.**

## 3. The registered band, and where it comes from
The point prediction assumes the **write is α-invariant** (same `D_i`, `s_i` at α=0.022). It is not
exactly: the write's `L_min` term needs `V(z_i) < V(z_i+δ) − margin`, and the confinement supplies
`α·E|δ|² = α·0.61` of that margin for free (0.0305 at α=0.05, 0.0134 at α=0.022 against
`margin = 0.15`), so at lower α the write must dig **slightly deeper**; its `L_bar` barrier term
conversely gets **easier** (the midpoint of two ball sites is nearer the origin, where the
confinement is lower). Both are ≤ ~11 % of the relevant loss scale and they oppose each other.
Sensitivity of the predicted cell mean:

| write drift | `D×0.9` | `D×1.0` | `D×1.1` |
|---|---|---|---|
| `s×0.95` | 0.609 | 0.616 | 0.622 |
| `s×1.00` | 0.641 | **0.650** | 0.658 |
| `s×1.05` | 0.685 | 0.696 | 0.705 |

> ### **REGISTERED BAND: `Δstrict ∈ [+0.02, +0.12]`, point `+0.064`.**
> Lower edge = write drift to (`D×0.9`, `s×0.95`) *minus* one SE of the difference; upper edge =
> drift to (`D×1.1`, `s×1.05`) *plus* one SE, i.e. the full write-drift envelope widened by seed
> noise. **Δ ≤ 0 or Δ > +0.12 falsifies (★) as a causal statement** and downgrades w26's "(★) reach
> parameters, not free knobs" to a correlation.

## 4. Secondary registered predictions (the mechanism, not just the number)
- **P-A (the rung that must move).** Item 9 (`|a|=0.60`, `x` 1.006 → 0.917) rises from **0.469** to
  **≥ 0.85** (point 0.941). Item 2 (same `|a|`, better-fitted well) 0.906 → **≥ 0.95**.
- **P-B ⭐ (the rung that must NOT move — the strongest falsifier).** Items 4, 8, 14, 15
  (`|a| ≥ 0.8667`) stay at **0.000**, tolerance ≤ 0.05 each. (★) says α cannot buy this rung: a 9.3 %
  ceiling shift leaves `x ≥ 1.21`, i.e. ≥ 7 logistic widths outside. **If any of these four retrieves
  > 0.20, α is acting through a channel (★) does not contain**, and the causal claim fails even if
  P-1 lands in band. *A right-band Δ obtained by lifting the wrong items is not a confirmation.*
- **P-C (soft).** The `|a| = 0.7333` rung (items 0, 12) partially recovers: predicted **0.299 /
  0.163**, band **[0.00, 0.70]** each. This rung sits inside the transfer's shoulder, where a 5 %
  error in `s_fit` swings the answer; it is registered as *direction only*.
- **P-D.** `a_U` ratio 1.093 ± 0.005 (§2.1). Computed, not measured — recorded so the arithmetic is
  auditable.
- **P-E (write near-invariance).** Measured `D_fit` median ∈ [0.41, 0.51] and `s_fit` median ∈
  [0.19, 0.21] at α=0.022 (i.e. within −10/+10 % and ±5 % of the α=0.05 values 0.459 / 0.200), with
  radial-fit R² ≥ 0.99. If violated, P-1 is re-scored post-hoc **with the measured fits** and both
  scores are reported.
- **P-F (where failures go).** Failed *flows* still collapse toward the ball centre / the interior
  item 13, but with the confinement 2.27× weaker the endpoint radius `|x_end|` for failing items
  **increases** vs the α=0.05 value (0.064). Registered as direction only.
- **P-G (control).** Value-blank (identical write with all payloads set to 0, scored against the
  real codebook) gives `strict_blank ≤ 0.13` at **both** α — the two codewords `|a| = 0.0667` are the
  only ones inside `payload_tol = 0.1` of zero, so the trivial ceiling is 2/16 = 0.125. A blank above
  0.13 invalidates the cell.

## 5. What I will report either way
- The paired Δ with population sd over 3 seeds (**population** sd, matching r2's convention),
  per-seed values, and the per-item table against P-A/P-B/P-C.
- The post-hoc re-score with measured `(D_i, s_i)` **labelled post-hoc**.
- The implication for the two load-bearing knobs (`learned_confine`, ball radius `R`), whichever way
  it goes.

## 6. What this test cannot do (stated in advance)
1. It is **one cell** (d=4, K=16, width 0.15). It says nothing about the three crowded high-K FAIL
   cells, which `readout-channel-theory` §2.5 already flagged as a *multi-well interference* failure
   outside (U)'s single-well scope.
2. Φ's two parameters are fitted on the baseline cell. The falsifiable content is that **`a_U`
   moves the curve by the amount (★) says**, not that the curve has this shape.
3. α also perturbs the write. P-E measures the drift; it does not make the test double-blind.
4. No read noise, deterministic Verlet, no temperature (as in w26).
