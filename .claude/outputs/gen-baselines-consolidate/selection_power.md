# Selection-power diagnostic — can a valid-only criterion pick a refusal configuration?

Per arm-family, over the 36 tuning trials: Spearman rho between the selection criterion (validation aggregate / validation `shape`, both valid-only) and the ORACLE-ONLY AUROC(B vs C).

| family | seed | rho(val_aggregate, AUROC(B vs C)) | rho(val_shape, AUROC(B vs C)) | selected val_agg | selected AUROC | best-AUROC trial's val_agg rank |
|---|---|---|---|---|---|---|
| `knn` | 0 | -0.226 | +0.138 | 0.5068 | 0.4383 | 34/36 |
| `knn` | 1 | -0.184 | -0.041 | 0.4697 | 0.7105 | 33/36 |
| `knn` | 2 | +0.157 | +0.070 | 0.5234 | 0.6441 | 29/36 |
| `lace` | 0 | -0.487 | -0.224 | 0.2454 | 0.9230 | 33/36 |
| `lace` | 1 | -0.615 | -0.511 | 0.2533 | 0.6561 | 17/36 |
| `lace` | 2 | -0.663 | -0.353 | 0.2721 | 0.6647 | 19/36 |
| `knn` | **mean** | **-0.084 ± 0.210** | **+0.056 ± 0.090** | — | — | — |
| `lace` | **mean** | **-0.588 ± 0.091** | **-0.363 ± 0.143** | — | — | — |

## Near-tie sensitivity: top-2 trials by the pre-registered criterion

| family | seed | rank | val_agg | AUROC(B vs C) ORACLE-ONLY | config |
|---|---|---|---|---|---|
| `knn` | 0 | 1 | 0.5068 | 0.4383 | `{'deployment': 'int8', 'k': 5, 'metric': 'cosine', 'normaliser': 'zscore'}` |
| `knn` | 0 | 2 | 0.4974 | 0.6316 | `{'deployment': 'int8', 'k': 5, 'metric': 'euclidean', 'normaliser': 'none'}` |
| `knn` | 1 | 1 | 0.4697 | 0.7105 | `{'deployment': 'int8', 'k': 3, 'metric': 'euclidean', 'normaliser': 'none'}` |
| `knn` | 1 | 2 | 0.4661 | 0.6725 | `{'deployment': 'int8', 'k': 3, 'metric': 'cosine', 'normaliser': 'none'}` |
| `knn` | 2 | 1 | 0.5234 | 0.6441 | `{'deployment': 'int8', 'k': 5, 'metric': 'cosine', 'normaliser': 'none'}` |
| `knn` | 2 | 2 | 0.5156 | 0.7160 | `{'deployment': 'int8', 'k': 5, 'metric': 'euclidean', 'normaliser': 'none'}` |
| `lace` | 0 | 1 | 0.2454 | 0.9230 | `{'form': 'both', 'shrinkage': '1e-3', 'sigma': 0.1}` |
| `lace` | 0 | 2 | 0.2298 | 0.5613 | `{'form': 'both', 'shrinkage': 'ledoit_wolf', 'sigma': 0.03}` |
| `lace` | 1 | 1 | 0.2533 | 0.6561 | `{'form': 'both', 'shrinkage': 'none', 'sigma': 0.03}` |
| `lace` | 1 | 2 | 0.2383 | 0.6020 | `{'form': 'both', 'shrinkage': '1e-3', 'sigma': 0.03}` |
| `lace` | 2 | 1 | 0.2721 | 0.6647 | `{'form': 'both', 'shrinkage': 'ledoit_wolf', 'sigma': 0.03}` |
| `lace` | 2 | 2 | 0.2702 | 0.9990 | `{'form': 'both', 'shrinkage': 'none', 'sigma': 0.1}` |
