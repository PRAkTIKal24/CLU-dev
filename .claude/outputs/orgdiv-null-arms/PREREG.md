# PREREG — `orgdiv-null-arms` (C2W5, the matched-capacity organizer audit)

**Filed 2026-08-01 by `experiment-engineer` BEFORE any line of `chlu/core/null_arms.py` or
`chlu/experiments/exp_null_arms.py` existed and before any harness ran.** Protocol §5
pre-registration rule: the acceptance criterion is a *measured threshold* ("does ANY matched-capacity
organizer clear `chance + 0.05` on the rule-4-valid unseen split"), so the numbers are committed here
first, with their derivations.

Base: `main @ eaecc91`; branch `agent/experiment-engineer/orgdiv-null-arms`; worktree
`../CHLU-null-arms`; main venv (`jax 0.9.0`). Frozen interfaces:
`.claude/outputs/orgdiv-cat-test/FROZEN-interfaces.md`. Task: `.claude/tasks/orgdiv-null-arms.md`
(HUB ADDENDUM 2026-08-01 re-scope = the family-solvability audit).

---

## 0. The inputs I am reasoning from (all measured by `orgdiv-cat-test`, not by me)

| # | fact | source |
|---|---|---|
| C1 | chance (constant predictor) = **3.906e-4**; bar = chance + 0.05 = **0.0504** | FROZEN §(iii) |
| C2 | `tol = 0.25 × RMS‖y − ȳ_seen‖` = **0.478** at the registered cell | cat-test §2 |
| C3 | occupancy precision of the **raw launch geometry** (no dynamics) = **0.4061 ± 0.0119** | cat-test §1.2 |
| C4 | family-level matched-filter **EXACT-set** recovery at `d = 4` = **0.006** | cat-test §7.1 |
| C5 | query-only OLS `R²` on the set code at `d = 4` = **0.102** | cat-test §7.1 |
| C6 | the physics arm reads **0.0008 ± 0.0008** unseen; its N3 assignment agreement is **0.211–0.233** | cat-test §1, §6.2 |
| C7 | SP-1: a reader with `≥ N_a·m = 256` fitted params solves the family from the TRUE indicator at **1.000** | cat-test §7.3 |

**The derivation that drives every prediction below.** `y(x) = Σ_{j∈A} v_j` with `v_j` unit-norm in
`R^8` and `F = 4`, so `‖y‖ ≈ 2` and `tol = 0.478`. Getting **three** of the four wells right and one
wrong leaves an error `‖v_wrong − v_right‖ ≈ √2 = 1.41 ≫ tol`. Soft/hedged predictions do no better:
writing `ŷ = Σ_j p_j v_j`, the error is `≈ ‖p − 1_A‖` in a near-orthogonal frame, so a within-`tol`
read needs `‖p − 1_A‖ ≲ 0.24` — i.e. **the metric demands essentially exact set recovery, and hedging
buys nothing.** Therefore every arm's score is bounded by *its own exact-set recovery rate*, and
recovery is bounded by what `φ` (frozen, `d = 4`, `σ_q = 0.15`, `P = 4`) makes available.

---

## 1. ⭐ THE HEADLINE PREDICTION (the audit's binary)

> **REGISTERED: NO arm clears `chance + 0.05`. `max_arm ≤ 0.03` on the unseen split, point estimate
> `0.008`, on every reader, at every capacity point, over the entire registered grid.**
> `P(no arm clears) = 0.85` · `P(exactly one clears) = 0.10` · `P(≥ 2 clear) = 0.05`.

If this survives, the Hub's first branch fires (**the family is refuted for every organizer class
measured**) and the cat-test kill is *not* attributable to the physics write/read specifically.

## 2. Per-arm registered predictions (unseen exact-set accuracy, 5 seeds, mean; matched capacity)

| arm | registered prediction | band | derivation |
|---|---|---|---|
| **N1 hard** (gradient-placed atoms, hard argmax read) | **0.006** | [0.000, 0.030] | C4: per-particle assignment arms inherit the matched-filter exact-set rate at `d = 4`. |
| **N1 soft** (temperature-swept attention read) | **0.008** | [0.000, 0.030] | the hedging bound above: soft weights cannot buy tolerance. |
| **N2 VQ** (k-means++/VQ-STE/product-VQ) | **0.006** | [0.000, 0.020] | same class as N1-hard; the codebook is at best the anchors. |
| **N3 static-geometric** `(c, σ, b)` fitted | **0.006** | [0.000, 0.020] | same class; strictly a power diagram over the launch points. |
| **N4 kNN** | **0.000** | [0.000, 0.002] | rule 4: every unseen `A` differs from **every** stored `B` in ≥ 2 of 4 wells ⇒ `‖y(A) − y(B)‖ ≥ tol` **by construction (K2)**; IDW averaging over `k` rows cannot beat it. |
| **N5 Titans** (surprise-gated fast weights) | **0.000** | [0.000, 0.005] | a smooth `R^4 → R^8` regression fitted on 128 points, against a target that is a 35 960-way lookup. Equivalent to C5's `R² = 0.102` ceiling, which cannot produce a within-`tol` vector. |

**`max_arm` (mechanical max over arms × grid × readers): registered `0.010`, band [0.000, 0.030].**

## 3. Internal-validity anchors (registered — these are how I tell "the family is hard" from
"my arms are broken")

| # | anchor | registered prediction | why it matters |
|---|---|---|---|
| **L1** | N1's accuracy on its own **training** items (SEEN, in-sample) | **≥ 0.50** | if an arm with 5 376 free floats cannot fit 128 items it saw, the audit measures my optimizer, not the family. ⛔ A verdict of "none clears" is INVALID unless L1 holds for at least N1. |
| **L2** | N4 kNN on **SEEN** (leave-one-out is not run; in-sample `k=1`) | **≥ 0.95** | the trivial memoriser must memorise (cat-test measured 0.9984 for the kNN reader on seen). |
| **L3** | shuffle-φ laundering control (every arm re-trained with query codes permuted across queries) | **≤ chance + 0.005** on every arm | any arm that still scores with the query destroyed is scoring on a fitting artifact, not on organization. |
| **L4** | capacity flatness: `|score(a=64) − score(a=12)|` for N1 | **≤ 0.02** | the bottleneck is registered to be `φ`, not capacity. If capacity matters, my derivation is wrong. |

## 4. The declared OUT-OF-CLASS diagnostic — the φ-decodability ceiling

⛔ **Reported, never scored as an arm** (the SP-1 precedent). A combinatorial matched filter that
enumerates all `C(32,4) = 35 960` set codes and returns the nearest one, then reads the true `v_j`:

| quantity | registered prediction | band | derivation |
|---|---|---|---|
| **noiseless** (`σ_q = 0`, exact `φ(x)`) exact-set accuracy | **≥ 0.99** | [0.95, 1.00] | the codes are distinct points; nearest-code is exact decoding. |
| **as-launched** (`P = 4` launches averaged, `σ_q = 0.15`) | **0.20** | [0.05, 0.60] | sufficient statistic `ĉ = mean_p(q0_p − o_p) = φ(x) + (σ_q/√P)ξ`, noise norm ≈ `0.075·2 = 0.15` in `R^4`. 35 960 codes on the 3-sphere of radius 2 (area `2π²R³ = 157.9`) ⇒ per-code cap volume `0.00439` ⇒ NN spacing `≈ (3·0.00439/4π)^{1/3} = 0.102`. Noise (tangential `0.075√3 = 0.13`) **exceeds** the spacing ⇒ ~3 competitors inside the noise ball. |

⭐ **Registered interpretation, committed in advance so it cannot be chosen after the fact:**
- ceiling ≥ 0.20 **and** all arms ≈ 0 ⇒ the family IS decodable through the frozen `φ`, and what fails
  is the **`P`-particle occupancy read protocol** shared by every registered arm (including the
  physics one) — a *protocol* refutation, sharper than a family refutation.
- ceiling ≤ 0.05 ⇒ the frozen `φ` at `d = 4` destroys the set, and the family is refuted for
  **anything** reading through it, physics or not.

## 5. The oracle-imitation row (T5.2 rider (i)) and F5's input

| quantity | registered prediction | band |
|---|---|---|
| N3 fitted **on the physics arm's own assignments**, assignment agreement | **0.45** | [0.25, 0.70] |
| — the same arm's unseen score | **0.001** | [0.000, 0.010] |
| N3 (read-objective-fitted) vs physics agreement, reproduced | **0.22** | [0.15, 0.30] (cat-test measured 0.211–0.233) |
| **F5 fires** (agreement ≥ 0.99)? | **NO** | — |

## 6. Registered grid and selection protocol (committed before it runs)

- Budget per arm: **≥ 5 optimiser points × 3 capacity points × 3 seeds** on SEEN, split
  **96 train / 32 validation**; selection on validation **exact-set accuracy, tie-broken on validation
  MSE** (registered because the accuracy metric is all-or-nothing and will frequently be 0 on 32 items).
- ⛔ **No selection touches `Q_unseen` anywhere.** Selected config is refit on all 128 SEEN, readers
  are fitted on all 128 SEEN, and only then is `Q_unseen` scored, on **5 seeds (0–4)**.
- **Gradient-free arms (N3, N2/k-means, N4):** the "5 learning-rate points" axis is declared
  **substituted** by 5 optimiser restarts / init points (N3), 10 k-means++ restarts × 5 commitment
  costs (N2), and 5 `k` × 2 weightings (N4). Declared, not silently dropped.
- **γ is not an axis for these arms**: no arm has a rollout, so every arm is γ-independent **by
  construction**. Registered as one column, declared, not reported as three identical ones.

## 7. What would make me WRONG in an interesting way (registered in advance)

1. **N1 clears.** Then per the Hub addendum (b) its score is the revival target, the family is
   solvable in the same landscape class, and the cat-test kill is attributable to the physics
   write/read. I put `P = 0.10` on this.
2. **L1 fails** (N1 cannot even fit SEEN). Then my audit is void and the honest report is "the
   optimiser, not the family" — I commit in advance to reporting that as a *failure of my arms*.
3. **The ceiling is high and the arms are at zero.** Then the correct verdict is the protocol
   refutation of §4, **not** the family refutation, and I commit to saying so.
