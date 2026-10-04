# MAZE-DELETION-2B -- exact deletion on CMAZE-2b

**This table contains no CMAZE-1 or CMAZE-2 row.** Every row: environment **CMAZE-2b**. Seeds = 3. Scored runs spent = 0.

| env | seed | deletion | rows removed / kept | downdate vs refit: statistic (max rel, float64) | subspace projector | head Gram A / Mx | read | read moved by deletion | clean-checkout statistic / projector | source |
|---|---|---|---|---|---|---|---|---|---|---|
| CMAZE-2b | 0 | target=green (w9's banked row) | 224 / 1218 | 7.212e-16 | 0.000e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 | 0.793 | 7.212e-16 / 0.000e+00 | `runs/deletion_primary_seed0.json` |
| CMAZE-2b | 1 | target=green (w9's banked row) | 225 / 1205 | 9.093e-16 | 0.000e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 | 0.506 | 9.093e-16 / 0.000e+00 | `runs/deletion_primary_seed1.json` |
| CMAZE-2b | 2 | target=green (w9's banked row) | 211 / 1240 | 8.937e-16 | 0.000e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 | 0.500 | 8.937e-16 / 0.000e+00 | `runs/deletion_primary_seed2.json` |

## Adaptation cost after deletion (NEVER 'no changes')

| env | seed | path | seconds | address-book floats changed | atom floats changed | VAL other-target validity before → after (Δ) | VAL green refusal before → after | retrained |
|---|---|---|---|---|---|---|---|---|
| CMAZE-2b | 0 | exact head downdate (A−=ΦdᵀΦd, Mx−=ΦdᵀXd, re-solve) | 2.042 | 402656 / 402752 | 7936 / 7936 | 0.2609 → 0.2411 (-0.0198) | 0.6333 → 0.5333 | no |
| CMAZE-2b | 0 | basis-change carry W←WR (w9 as pre-registered) | — | — | — | 0.2609 → 0.0000 (-0.2609) | 0.6333 → 1.0000 | no |
| CMAZE-2b | 1 | exact head downdate (A−=ΦdᵀΦd, Mx−=ΦdᵀXd, re-solve) | 2.328 | 402656 / 402752 | 7936 / 7936 | 0.1849 → 0.1828 (-0.0021) | 0.7248 → 0.4404 | no |
| CMAZE-2b | 1 | basis-change carry W←WR (w9 as pre-registered) | — | — | — | 0.1849 → 0.0000 (-0.1849) | 0.7248 → 1.0000 | no |
| CMAZE-2b | 2 | exact head downdate (A−=ΦdᵀΦd, Mx−=ΦdᵀXd, re-solve) | 2.181 | 402674 / 402752 | 7936 / 7936 | 0.2490 → 0.2431 (-0.0059) | 0.5638 → 0.5319 | no |
| CMAZE-2b | 2 | basis-change carry W←WR (w9 as pre-registered) | — | — | — | 0.2490 → 0.0000 (-0.2490) | 0.5638 → 1.0000 | no |

## Rival deletes only by retrain-without (cost printed)

| env | seed | rival | params (target) | steps × batch | row reads of retained data | seconds | items kept | CLU downdate seconds |
|---|---|---|---|---|---|---|---|---|
| CMAZE-2b | 0 | transformer (matched read-path params) | 405776 (410688) | 4000 × 64 | 256000 | 89.8 | yes (1218 rows) | 0.875 |
| CMAZE-2b | 1 | transformer (matched read-path params) | 405776 (410688) | 4000 × 64 | 256000 | 90.8 | yes (1205 rows) | 0.917 |
| CMAZE-2b | 2 | transformer (matched read-path params) | 405776 (410688) | 4000 × 64 | 256000 | 88.6 | yes (1240 rows) | 0.802 |

## Supplementary deletions (same object, other words) -- labelled supplementary

| env | seed | deletion | rows removed | statistic | projector | read | read moved |
|---|---|---|---|---|---|---|---|
| CMAZE-2b | 0 | target=0 | 234 | 7.266e-16 | 0.000e+00 | 0.0e+00 | 0.459 |
| CMAZE-2b | 0 | target=2 | 258 | 9.011e-16 | 0.000e+00 | 0.0e+00 | 0.465 |
| CMAZE-2b | 0 | target=3 | 235 | 7.228e-16 | 0.000e+00 | 0.0e+00 | 0.770 |
| CMAZE-2b | 0 | target=4 | 222 | 8.753e-16 | 0.000e+00 | 0.0e+00 | 0.550 |
| CMAZE-2b | 0 | target=5 | 269 | 7.487e-16 | 0.000e+00 | 0.0e+00 | 0.576 |
| CMAZE-2b | 0 | maze=0 | 18 | 6.312e-16 | 0.000e+00 | 0.0e+00 | 0.724 |
| CMAZE-2b | 0 | agent=0 | 231 | 9.151e-16 | 0.000e+00 | 0.0e+00 | 0.377 |
| CMAZE-2b | 0 | random 10% of rows (rng 1000) | 134 | 6.820e-16 | 0.000e+00 | 0.0e+00 | 0.302 |
| CMAZE-2b | 1 | target=0 | 259 | 7.501e-16 | 0.000e+00 | 0.0e+00 | 0.635 |
| CMAZE-2b | 1 | target=2 | 245 | 9.194e-16 | 0.000e+00 | 0.0e+00 | 0.608 |
| CMAZE-2b | 1 | target=3 | 247 | 9.135e-16 | 0.000e+00 | 0.0e+00 | 0.473 |
| CMAZE-2b | 1 | target=4 | 231 | 9.150e-16 | 0.000e+00 | 0.0e+00 | 0.522 |
| CMAZE-2b | 1 | target=5 | 223 | 9.024e-16 | 0.000e+00 | 0.0e+00 | 0.607 |
| CMAZE-2b | 1 | maze=0 | 7 | 4.771e-16 | 0.000e+00 | 0.0e+00 | 0.798 |
| CMAZE-2b | 1 | agent=0 | 238 | 9.283e-16 | 0.000e+00 | 0.0e+00 | 0.500 |
| CMAZE-2b | 1 | random 10% of rows (rng 1001) | 132 | 8.561e-16 | 0.000e+00 | 0.0e+00 | 0.373 |
| CMAZE-2b | 2 | target=0 | 260 | 7.478e-16 | 0.000e+00 | 0.0e+00 | 0.699 |
| CMAZE-2b | 2 | target=2 | 259 | 9.037e-16 | 0.000e+00 | 0.0e+00 | 0.431 |
| CMAZE-2b | 2 | target=3 | 239 | 7.253e-16 | 0.000e+00 | 0.0e+00 | 0.597 |
| CMAZE-2b | 2 | target=4 | 259 | 9.128e-16 | 0.000e+00 | 0.0e+00 | 0.452 |
| CMAZE-2b | 2 | target=5 | 223 | 8.968e-16 | 0.000e+00 | 0.0e+00 | 0.872 |
| CMAZE-2b | 2 | maze=0 | 18 | 6.276e-16 | 0.000e+00 | 0.0e+00 | 0.673 |
| CMAZE-2b | 2 | agent=0 | 233 | 9.181e-16 | 0.000e+00 | 0.0e+00 | 0.522 |
| CMAZE-2b | 2 | random 10% of rows (rng 1002) | 164 | 6.952e-16 | 0.000e+00 | 0.0e+00 | 0.417 |
