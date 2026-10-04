# PREREG — gen-certificate (filed BEFORE any harness existed or any real-data number was computed)

**Written:** 2026-09-11, before `chlu/core/gen_certificate.py` had a single line and before any
feature was loaded. Design was fixed by reading `gen_store.py`, `c_invalid.json` and the R-rulings only.

## 0. What is being predicted
A **derived** refusal criterion for a factor request, computed from quantities the R0 store already
exposes (per-well mode histograms, the mode prior, the modes themselves, the store's own soft-assignment
temperature), with **zero invalid-example supervision** and **no learned refusal head**.

### The certificate, declared in full before measurement
For a request `R = {(factor j, value u)}`:
1. **reach** `E_reach(R) = Σ_t (dmin_t / (s_{j}/2))²`, `dmin` = distance from `u` to the nearest realised
   grid value of factor `j`, `s_j` = that factor's median grid spacing. Units: squared Voronoi radii —
   **dimensionless by construction, so it enters unstandardised.**
2. **composition conflict** `χ(t,t') = − Σ_k log Σ_m q_t[k,m] q_{t'}[k,m] / p̄[k,m]`, with `q_{(j,u)} = `
   the per-well mode histogram of the store's own addressing over **valid train rows** whose factor `j`
   equals the grid value nearest `u` (Laplace α=1), and `p̄` the marginal mode usage. Request statistic:
   `E_pair_max = max over unordered pairs`, `E_pair_mean` reported alongside.
3. **curvature** at the product-of-experts composed point `x̂_k = Σ_m r[k,m] μ_{k,m}`,
   `r ∝ p̄ Π_t (q_t/p̄)`: `λ_min,k = 2 − (2/τ²)·λ_max(Cov_w(μ_k))` (closed form of the Hessian of the
   store's own soft energy `Φ = −2τ² log Σ_m exp(−‖x−μ_m‖²/2τ²)`), `τ = gen_store.tau_end = 0.1`.
   Component = `max_k (−λ_min,k)`.
4. **store energy** `E_store = Σ_k min_m ‖x̂_k − μ_{k,m}‖² / D`.

`cert_total = E_reach + z(E_pair_max) + z(curv) + z(E_store)`, every `z` standardised on the **A** request
set only, through the audited fitter. Higher = refuse.

### Controls, declared before measurement
- `dist_features` — ⭐ **the laundering control**: naive kNN-style composition (mean of per-term prototype
  features) → plain Σ_k min_m ‖x̄_k − μ_{k,m}‖²/D. No composition logic, no reach.
- `dist_plus_reach` — ⭐ **the HARD control**: `z(dist_features) + E_reach`. If this matches `cert_total`,
  the derivation is decoration over naive geometry plus a grid lookup.
- `n_terms` — the syntactic by-construction control (C-duplicate records carry **7** terms, A/B carry 6).

## 1. ⛔ The structural fact that decides where the claim can live
`E_reach` is **identically 0** for every A request, every B request and every C **duplicate_factor**
request (all their values are on-grid), and **> 0** for every C **out_of_range** request. Therefore:
- the `out_of_range` half of C is separable by a grid lookup alone — **any** arm that embeds a request
  will get ~1.0 there, and that number carries no information about the mechanism;
- ⭐ **the entire mechanism claim reduces to the `duplicate_factor` half.** Every headline below is
  therefore reported split by C kind, and the dup-only column is the one the spoke lives on.

## 2. Predictions (committed; measured afterwards)
| # | prediction | threshold | point |
|---|---|---|---|
| P1 | **B-vs-C AUROC, `cert_total`** (primary, R23) | ≥ 0.90 | **0.96** |
| P1a | B-vs-C, **dup only**, `cert_total` | ≥ 0.80 | **0.92** |
| P1b | B-vs-C, **oor only**, `cert_total` | = 1.000 exactly | 1.000 |
| P2 | \|A-vs-C − B-vs-C\| for `cert_total` | ≤ 0.05 | 0.01 |
| P3 | class-balanced A∪B-vs-C **= ½(A-vs-C + B-vs-C)** (weighted Mann–Whitney identity) | \|Δ\| ≤ 1e-12 | exact |
| P4 | ⭐ laundering: `dist_features` B-vs-C **overall** | ≤ 0.70 | 0.55 |
| P4a | `dist_features` **oor only** | ≤ 0.70 | 0.50 (it cannot see OOR: prototypes snap to the grid) |
| P5 | ⭐ **the hard control**: `cert_total` dup-only **minus** `dist_plus_reach` dup-only | ≥ +0.10 | +0.25 |
| P6 | `n_terms` dup-only AUROC (by construction) | = 1.000 | 1.000 |
| P6a | `n_terms` oor-only AUROC (by construction) | = 0.500 | 0.500 |
| P7 | refusal rate on **B** at the A-95th-percentile threshold (the COST on valid-unseen) | ≤ 0.25 | 0.10 |
| P8 | AURC on the balanced B∪C mixture (random ordering ⇒ 0.5) | ≤ 0.15 | 0.06 |
| P9 | correlation-grouping dup-only AUROC **−** random-grouping dup-only AUROC | > 0 in 3/3 seeds | +0.05 |
| P10 | ⭐ audited-fitter assertion: C record ids appearing in any threshold/normaliser fit | **= 0** | 0 |

## 3. The counter-hypothesis, pre-registered so it can win
**H0 (derivation is decoration):** the composition barrier adds nothing — `reach`-only scores within 0.02
of `cert_total` on B-vs-C, and the dup-only AUROC of `cert_total` is ≤ 0.60.
⭐ `reach`-only dup-only AUROC is **0.500 exactly by construction** (all values tie at 0), so H0 is
directly testable and P1a is exactly the test. If P1a fails and P5 fails, **H0 wins and I report that.**

## 4. What would prove the spoke wrong (echoing the task)
- any C record id inside a fit (P10 ≠ 0) ⇒ the headline is void regardless of AUROC;
- P5 fails ⇒ the derived barrier is decoration over naive geometry + a grid lookup.

⛔ No kill-rule verdict is rendered by this spoke (that is `gen-cell-run`'s, against its own prereg).

---

# ⟲ AMENDMENT A1 — filed after the `--quick` plumbing smoke, BEFORE any full-scale number

⚠ Two things were learned from a 40 s quick run (2,048 rows, 8 modes, 30 training steps — ⛔ a store
that is not a store; no number from it is reported anywhere), and both are **mechanical, not outcome
selection**:

**A1.1 — a metric bug in my own risk-coverage code, fixed.** Ties were being resolved in *input order*,
which admitted the valid requests of a tied block first. The `reach` arm ties **every** A request, every
B request and every `duplicate_factor` request at exactly 0.0, so in-order resolution handed it an AURC
it had not earned. Fixed to collapse ties (one curve point per distinct score, the whole tied block
admitted together) before any reported number existed. ⛔ No AUROC was affected (average ranks already).

**A1.2 — a declared SECONDARY combination, `clu_pair_reach = reach + z(pair)`.** The `curv` and `energy`
components are read at the product-of-experts composed point, and that distribution **sharpens with the
number of terms**; the C `duplicate_factor` records carry 7 terms against A/B's 6, so those two
components can inherit the syntactic giveaway that the `count` control measures directly. `pair` (a max
over pairs) and `reach` (a sum of per-term distances that is 0 for every on-grid term) do not.
⛔ **The pre-registered primary `cert_total` is unchanged and is still the headline.** `clu_pair_reach` is
reported beside it with this provenance. Predictions for it, committed now:
| # | prediction | threshold | point |
|---|---|---|---|
| A1-P1 | `clu_pair_reach` B-vs-C, dup only | ≥ 0.80 | 0.90 |
| A1-P2 | `clu_pair_reach` dup-only ≥ `clu` dup-only | ≥ 0 | +0.05 |

---

# ⟲ AMENDMENT A2 — the TERM-COUNT CONTROL (`A7`), filed 2026-09-11 00:0x

⚠ **Full disclosure of when this was written.** It was written **after** seeing seed 0 / correlation's
non-A7 numbers (`clu` B-vs-C 0.9660, dup 0.9321; `distance_reach` dup 0.8866) and after a `--quick` smoke
that exercised the A7 code path on a degenerate 8-mode store (⛔ never reported). **No full-scale A7 number
existed when the predictions below were committed**; seeds 1–2 and the seed-0 re-run produce the first
ones.

**What A7 is.** A request set built from **valid** A requests alone by appending one **redundant** term —
a factor repeated *at the value it already has*. It is satisfiable (it is the same image) and it carries
**7 terms**, exactly like a C `duplicate_factor` record. ⛔ No invalid example is used to construct it, so
the supervision claim is untouched and the audit still runs.

**Why it is necessary and why it was missed in the original prereg.** The C `duplicate_factor` records
carry 7 terms against A/B's 6. The `count` control already showed this is worth **AUROC 1.000 for free**.
Any statistic with the slightest term-count sensitivity therefore inherits a by-construction giveaway when
scored against A or B. ⭐ **`A7`-vs-`C[duplicate_factor]` is the only version of the dup question that is
not contaminated**, and every dup number in the report is now reported in both forms.

| # | prediction | threshold | point |
|---|---|---|---|
| A2-P1 | `count` A7-vs-Cdup (by construction) | = 0.5000 exactly | 0.5000 |
| A2-P2 | `reach` A7-vs-Cdup (by construction: all ties) | = 0.5000 exactly | 0.5000 |
| A2-P3 | ⭐ `clu` A7-vs-Cdup | ≥ 0.75 | 0.85 |
| A2-P4 | `clu` A7-vs-Cdup **<** `clu` A-vs-Cdup (part of the latter is length) | — | −0.05 |
| A2-P5 | ⭐⭐ **the strictest laundering test**: `clu` A7-vs-Cdup − `distance_reach` A7-vs-Cdup | > 0 in 3/3 seeds | +0.05 |

⛔ If A2-P5 fails, the derived barrier does not beat naive geometry once the length giveaway is removed,
and the spoke reports that as its result.
