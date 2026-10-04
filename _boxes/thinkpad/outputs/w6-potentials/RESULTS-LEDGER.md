# w6 hub/potentials — RESULTS LEDGER (scored run 1 of 3)

Every pre-registered bar, every seed. A leg that did not run is printed **NOT RUN**, never as a null.

| bar | what it tests | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| B1 | median corner \|log10 n_meas/n_pred\| ≤ 0.10 | 0.0433 **PASS** | 0.0435 **PASS** | 0.0431 **PASS** |
| B2 | paired mean Δ vs FITTED one-parameter rival (< 0) | -0.5968 **PASS** | -0.5796 **PASS** | -0.5982 **PASS** |
| B2 | corners better than the fitted rival (≥ 40/49) | 49 | 49 | 49 |
| B3 | underdamped \|d log n / d log μ²\| ≤ 0.10 (median over γ) | 0.0357 **PASS** | 0.0282 **PASS** | 0.0343 **PASS** |
| B3 | …and its max over γ (printed, stricter reading) | 59.1158 | 0.1612 | 1883.2934 |
| B3 | \|log10 n_meas / (2ln2/−ln(1−γ))\| ≤ 0.10 | 0.0378 **PASS** | 0.0378 **PASS** | 0.0348 **PASS** |
| B4 | V-curve argmin / γ_crit within ×2 (D16 small-h form) | 0.3362 **MISS** | 0.3301 **MISS** | 0.3520 **MISS** |
| B4 | …against the EXACT inversion of h*(γ)=h | 0.3782 | 0.3784 | 0.3834 |
| B5 | D17 overdamped slope measured (derived ≈ +1.14) | +1.1368 | +1.1367 | +1.1368 |
| B5 | …mean \|meas − derived\| (must beat naive) | 0.0038 **PASS** | 0.0042 **PASS** | 0.0040 **PASS** |
| B5 | …mean \|meas − naive +1\| | 0.1436 | 0.1430 | 0.1436 |
| B5 | D18 underdamped slope measured (derived ≈ −1.01) | -1.1090 | -1.1090 | -1.1086 |
| B5 | …mean \|meas − derived\| (must beat naive) | 0.1368 **PASS** | 0.1277 **PASS** | 0.1290 **PASS** |
| B5 | …mean \|meas − naive −1\| | 0.1417 | 0.1324 | 0.1333 |
| B6 | Hessian + D14 lift, finite difference ≤ 1e-9 | 3.519e-11 | 1.202e-10 | 2.699e-11 |

| audit | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| corners scored | 49 | 49 | 49 |
| rows | 64335 | 64320 | 64320 |
| field sha | 697d60dfa3e4b70c | 608fc3da332b972d | fca1c4a89befc12e |
| store bytes | 524,288 | 524,288 | 524,288 |
| byte pin matched | True | True | True |
| integrator max\|diff\| | 1.885e-08 | 1.885e-08 | 1.885e-08 |
| instrument floor (exact quadratic) | 0.0349 | 0.0349 | 0.0349 |
| fitted rival C | 0.2007 | 0.206 | 0.2012 |
