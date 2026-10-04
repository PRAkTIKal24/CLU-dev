# POSTHOC_1 — the PR confound (declared before it is computed)

**A NEW FILE, not an edit of `PREREG.md`** (R34's rule). Filed while the production run was still on
its **first** seed's sweep: no exponent existed on disk for any dataset when this was written.
⛔ Nothing here changes a pre-registered answer. §4's rule and §5's table stand exactly as filed.

## The confound
The two datasets' wells do **not** have the same participation-ratio distribution. If the fitted
exponent depended on `PR`, a dSprites-vs-Shapes3D difference could be produced by the **partition's
spectra** rather than by the **factor manifold** — and §11b's claim is about the factor manifold.

## The post-hoc analysis, declared before it runs
1. Pool all `(dataset, seed, well)` cells and regress `fitted_exponent ~ log PR` **within each
   dataset**, reporting slope, Pearson `r` and `p`.
2. Fit the pooled model `fitted_exponent ~ β0 + β1·log PR + β2·1[dsprites]` over all 30 cells and
   report `β2` with its standard error — **the dataset effect after controlling for `PR`**.
3. Report a `PR`-matched subset comparison if one exists (cells whose `PR` overlap between datasets).

## How it may be read
- ⭐ If `β2` keeps the sign and magnitude of the raw dataset difference, the difference is **not** a
  `PR` artefact.
- ⛔ If `β2` collapses toward 0 while `β1` is large, the raw difference is **explained by the
  spectra**, and ⛔ the primary result must be reported with that caveat attached in the same breath.
- ⛔ Either way this is **post-hoc and diagnostic**: it can qualify the §4 answer, ⛔ never replace it,
  and ⛔ it may not be used to rescue a NO.
