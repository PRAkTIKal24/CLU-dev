# CERTIFIED ACCESS — seed 0, rung A (DEV, unscored)

**Figure:** `certified_access_seed0_A.png`  **Raw numbers:** `CERTIFIED_ACCESS_FIGURE_seed0_A.json`
**Script:** `scorer/u6_figure.py` (`--plot-only` redraws from the JSON)
**Machine:** FORGIS_14 / Windows AMD64 / CPU (jax 0.9.0, backend `cpu`, numpy 2.4.1)

## The one sentence

> On these 192 rays the **certificate flips at `min_k d_k/σ_k = 2.06`** (50 % of rays certified,
> `d_k ≥ σ_k√(2 ln N)` in some well) while the **plain decode collapses at `4.08`** — the
> certificate flips **2.02 units earlier**, where the decoded output still scores **0.800** against
> the external classifier.

## Method

1. **Object.** seed 0, rung `A`, K = 5 wells, N = 64 atoms/well, `σ = [4.6343, 5.0714, 9.1827,
   6.3054, 5.1006]`, certified radii `σ_k√(2 ln 64) = [13.366, 14.626, 26.484, 18.185, 14.710]`
   (`= 2.8841 σ` in σ-units), `feat_shape = (128,4,4)` → 2048-dim latent.
2. **Launch points.** 192 rows drawn from `ctx["val_rows"]` by `default_rng(20260917)`. Each row's
   own **cached** feature `x0` is read from `ctx["X_store"]` at its position in `ctx["pool"]`.
   ⛔ Nothing is re-encoded; ⛔ `ctx["eval_b"]` is never read.
3. **The ray.** `x(t) = x0 + t·u`, `u` one **fixed unit-norm direction per row** from
   `default_rng(20260917).standard_normal((192, 2048))`, row-normalised — **direction seed 20260917**,
   the same `u` at every radius. 22 radii `t ∈ [0, 86]`, chosen from the measured scale so that
   `min_k d_k/σ_k` sweeps **0.81 → 5.05** (it starts at 0.81, not 0: a val row is not an atom).
4. **Measured at each radius** (`admission_lib`, not re-implemented): `min_k d_k/σ_k`, `max_k E_k`,
   the CLU gate (`E_k > 0` in any well), the certificate (`d_k ≥ σ_k√(2 ln N)` in any well), and the
   whole-latent and factored kNN gates at `q = 0.95`. Both kNN thresholds are fitted on **4096
   `fit_rows` cached features** (⛔ never `val`, ⛔ never B): whole `21.160`, factored
   `[6.600, 7.045, 11.230, 8.145, 7.548]`. The same rivals fitted on the R8 embeddings of those
   corners are in the JSON (`knn_whole_r8_x = 1.008`; the factored-R8 gate never reaches 50 % refusal
   in this range).
5. **Decode leg — it ran.** Decoder = `exp_gen_baselines.load_shared_decoder` → the gen-decoder-lift
   winner `v1_capacity_schedule`, hash-asserted `01ef91f7…`. Classifier =
   `exp_gen_cell.load_external_classifier` → the gated instrument `c8741ac4…` (978 873 params).
   Score = `gen_probe.probe_accuracy` (mean over the 6 factors) against the originating val row's
   factor labels. Chance = 0.1236.
   * **plain decode** = `decode_features(x(t))`.
   * **CLU read** = `gen_landscape.settle_chain(land, x(t), log_w, SettleCfg(steps=1000, the shipped
     `gen_dynamical_read` knobs))`, then decode — the shipped request-steered read launched at `x(t)`
     with the val row's **own** request as the term tensor (`refusal_lib.term_experts`, asserted
     **bit-identical** to the shipped `request_experts` here).
   * **control** = the same call with `log_w = None` (request term off).
6. **Crossings** are linear interpolations on the mean-`min_k d_k/σ_k` axis: the gates at their
   50 %-of-rays point; "decode collapse" where the plain-decode accuracy crosses the midpoint between
   its `t = 0` value (0.9931) and chance, i.e. **0.5583**.

## What the figure shows

| marker | `min_k d_k/σ_k` | plain-decode accuracy there |
|---|---|---|
| kNN factored, q = 0.95 | **1.00** | 0.975 |
| kNN whole-latent, q = 0.95 | **1.19** | 0.946 |
| zero level set (`max_k E_k` crosses 0) | **1.55** | 0.879 |
| **certificate flips** | **2.06** | **0.800** |
| **plain decode collapses** (midpoint to chance) | **4.08** | 0.558 |
| plain decode below 0.5 | 4.73 | 0.500 |

The CLU read is **flat at 0.3507 at every radius** and the request-off control is **flat at 0.1198**
(chance is 0.1236): on this rung the settled state is determined by the request term, not by where
the chain was launched, so the read neither degrades with radius nor recovers the launch's content.
Self-checks that ran and passed: `identity_E_le_0_on_live_store` (max `E_k` at the atoms = −30.39 ≤ 0)
and the `term_experts` ↔ `request_experts` equality (`max_abs_delta = 0.0`, bit-identical).

⛔ **No claim beyond this figure.** One seed, one rung, one ray direction per row, 192 val rows, DEV,
unscored. Nothing here says the ordering holds on B, at another rung, or for a non-random direction.
