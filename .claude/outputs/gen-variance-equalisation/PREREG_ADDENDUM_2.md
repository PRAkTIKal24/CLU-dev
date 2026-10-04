# PREREG ADDENDUM 2 — `gen-variance-equalisation`
⛔ **A NEW FILE.** `PREREG.md` and `PREREG_ADDENDUM_1.md` are **not edited**. Filed **BEFORE** the production run.

## The Leg-1 statistic is NULL-SUBTRACTED. Discovered by a test that failed on purpose.
`PREREG.md` §2 defines `b_{j,f} = Σ_v (n_v/N)(mean_v T_j − mean T_j)²`. ⛔ **That statistic is positively biased
under the null:** with labels that carry *nothing*, `E[b_j] = ((V−1)/N)·λ_j`.
⭐ **The bias is proportional to `λ_j`** — i.e. it is concentrated in exactly the high-variance directions the
hypothesis is about. Left in, *"the factor's signal is in the leading PCs"* would be a **property of the estimator**,
true of every factor including a random one, ⛔ not a fact about the features.

**How it was caught, and it is a real catch:** `tests/test_gen_whiten.py` plants a signal in a **trailing** axis of a
synthetic well and requires the spectrum analysis to find it there. With the raw statistic and `N = 600, V = 4` the
analysis reported the signal in the **leading** axis — the phantom. The test failed; the estimator was wrong.

**The change (`chlu/core/gen_whiten.py::debias_between_class_variance`):**
`b̃_{j,f} = max(0, b_{j,f} − ((V−1)/N)·λ_j)`, clipped at 0 (a negative excess is *no evidence*, not negative
evidence). `variance_explained` and `rank_frac_median` are computed from `b̃`; the **raw** values ship beside them as
`variance_explained_raw` / `rank_frac_median_raw`, and `null_bias_share_of_total_variance = (V−1)/N` ships on every
row so a reader can see the size of what was removed.

## ⛔ Does this move the predictions? Barely, and the direction is declared in advance
At the production cell (`N = 16,384`, `V ≤ 15`) the removed bias totals `(V−1)/N ≤ 0.085 %` of a well's variance,
against a predicted `VE(scale) ≈ 2 %` — ⇒ **`PREREG.md` §4's L1–L11 predictions stand unchanged**, and every one of
them is now measured on the **debiased** statistic. The bias's effect on `rank_frac_median` is the larger one and it
runs **against** the hypothesis (it drags every factor's apparent signal toward rank 1), so removing it **can only
make the premise test easier to pass**. ⭐ **Declared here, before the measurement, so the reader can discount it:**
the premise test in `PREREG.md` §3 is therefore being applied to a statistic that is *more favourable* to my own
hypothesis than the one I pre-registered, and `rank_frac_median_raw` is reported so the stricter reading is available.

## ⛔ Nothing else is revised
The premise test, the stop rule, both falsifiers, the byte ledger, the capacity bound and every predicted value in
`PREREG.md` stand. `PREREG_ADDENDUM_1.md`'s matched-compute control stands.
