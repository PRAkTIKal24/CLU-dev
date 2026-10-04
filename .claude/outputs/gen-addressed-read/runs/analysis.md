
## ⭐⭐ THE GENERATION TABLE — composition-B aggregate, external classifier, 3 seeds

| arm | composition-B | per-seed | distinct tuples / 72 | deployed bytes (× the 524,288 B budget) | role |
|---|---|---|---|---|---|
| `store_removed` (**the null**) | **0.1204** ± 0.0131 | .1111 / .1389 / .1111 | 1.00 ± 0.00 | 524,288 + 74,240 head = 1.00× | laundering control |
| `ebm_compose` | **0.1937** ± 0.0182 | .1736 / .1898 / .2176 | 72.00 ± 0.00 | 14,226,660 = 27.14× | like-for-like |
| `lace` | **0.3117** ± 0.0153 | .3009 / .3333 / .3009 | 71.67 ± 0.47 | not byte-matched | like-for-like (cell harness) |
| ⭐ `clu` (**the codebook read**) | **0.5949** ± 0.0082 | .5903 / .6065 / .5880 | 9.00 ± 0.82 | 524,288 + 74,240 head = 1.00× | ⭐⭐ **the matched-bytes rival** |
| ⭐⭐ **`x0` (THE ADDRESSED READ, MATCHED BYTES)** | **0.5008** ± 0.0498 | .4329 / .5185 / .5509 | 18.67 ± 2.36 | 524,288 + 74,240 head = 1.00× | ⭐⭐ **THE HEADLINE ROW** |
| `x0/A_atom_mean_lookup` | **0.3927** ± 0.0986 | .5231 / .2847 / .3704 | 11.00 ± 1.63 | 524,288 + 74,240 head = 1.00× | ⭐ §C.1 floor (the Advisor's named rival) |
| `x0/A_row_lookup` | **0.3272** ± 0.0633 | .2847 / .2801 / .4167 | 2.67 ± 0.47 | 524,288 + 74,240 head = 1.00× | ⭐ §C.1 floor (whole stored row) |
| `x0/A_atoms_shuffled` | **0.1366** ± 0.0422 | .1782 / .0787 / .1528 | 22.33 ± 3.68 | 524,288 + 74,240 head = 1.00× | ⭐ §C.2 control |
| `dynread` (the SETTLED read, same atoms) | **0.2083** ± 0.0514 | .1620 / .1829 / .2801 | 13.33 ± 1.25 | 1,048,576 = 2.00× | ⛔ another spoke's row, shown for contrast |
| `knn_proto` (**ceiling**) | **0.8650** ± 0.0122 | .8588 / .8542 / .8819 | 56.00 ± 3.74 | 475,136 = 0.91× | ⛔ unbounded real-valued channel |

### ⭐ The 8× companion (⛔ never quoted without the multiple)

| row | composition-B | per-seed | distinct tuples / 72 | deployed bytes |
|---|---|---|---|---|
| `x0/B` | **0.6852** ± 0.0173 | .6667 / .7083 / .6806 | 25.33 ± 2.62 | 4,194,304 + 593,920 head = 8.00× | 
| `x0/B_atom_mean_lookup` | **0.5231** ± 0.0842 | .6227 / .5301 / .4167 | 14.00 ± 2.45 | 4,194,304 + 593,920 head = 8.00× | 
| `x0/B_row_lookup` | **0.4421** ± 0.0650 | .4560 / .5139 / .3565 | 5.33 ± 0.47 | 4,194,304 + 593,920 head = 8.00× | 
| `x0/B_atoms_shuffled` | **0.1574** ± 0.0452 | .2060 / .1690 / .0972 | 33.00 ± 2.45 | 4,194,304 + 593,920 head = 8.00× | 
| `dynread/B` (settled) | **0.2114** ± 0.1033 | .0671 / .3032 / .2639 | 16.00 ± 1.41 | 8,388,608 = 16.00× | 

### ⭐⭐ THE VERDICT

⭐⭐ MATCHED BYTES (524288 B store + 74240 B head = 1.00× budget): `x0` = 0.5008 ± 0.0498 vs `clu` 0.5949 ± 0.0082 ⇒ **LOSS** (Δ = -0.0941, m = 0.020) · §C.1 floor ⛔ FAIL · ceiling LOSS · PASS: NO

**Rung A — `x0`, 1.00× the budget**
| rival | role | Δ | paired per-seed | 3/3 same sign | intervals overlap | verdict |
|---|---|---|---|---|---|---|
| `clu` | like-for-like | **-0.0941** | -0.1574 / -0.0880 / -0.0370 | yes | no | **LOSS** |
| `ebm_compose` | like-for-like | **+0.3071** | +0.2593 / +0.3287 / +0.3333 | yes | no | **BEATEN OUTRIGHT** |
| `knn_proto` | ceiling | **-0.3642** | -0.4259 / -0.3356 / -0.3310 | yes | no | **LOSS** |
| `lace` | like-for-like | **+0.1890** | +0.1319 / +0.1852 / +0.2500 | yes | no | **BEATEN OUTRIGHT** |
| `store_removed` | like-for-like | **+0.3804** | +0.3218 / +0.3796 / +0.4398 | yes | no | **BEATEN OUTRIGHT** |
| `x0/A_atom_mean_lookup` | ⭐ §C.1 matched-bytes LOOKUP FLOOR | **+0.1080** | -0.0903 / +0.2338 / +0.1806 | **no** | yes | **LOSS** |
| `x0/A_row_lookup` | ⭐ §C.1 matched-bytes LOOKUP FLOOR | **+0.1736** | +0.1481 / +0.2384 / +0.1343 | yes | no | **BEATEN OUTRIGHT** |
| `x0/A_atoms_shuffled` | ⭐ §C.2 landscape-removed control | **+0.3642** | +0.2546 / +0.4398 / +0.3981 | yes | no | **BEATEN OUTRIGHT** |

- ceiling verdict: **LOSS** · like-for-like all beaten: **NO** · ⭐ §C.1 floor: **⛔ FAIL** · §C.2 control collapses: **YES** · **PASS: NO**
- ⛔⛔ THE ARM DOES NOT BEAT A MATCHED-BYTES LOOKUP TABLE ⇒ the store is DECORATIVE and this row does NOT count as a positive (charter §C.1), whatever the rest of the table says

**Rung B — `x0/B`, 8.00× the budget**
| rival | role | Δ | paired per-seed | 3/3 same sign | intervals overlap | verdict |
|---|---|---|---|---|---|---|
| `clu` | like-for-like | **+0.0903** | +0.0764 / +0.1019 / +0.0926 | yes | no | **BEATEN OUTRIGHT** |
| `ebm_compose` | like-for-like | **+0.4915** | +0.4931 / +0.5185 / +0.4630 | yes | no | **BEATEN OUTRIGHT** |
| `knn_proto` | ceiling | **-0.1798** | -0.1921 / -0.1458 / -0.2014 | yes | no | **LOSS** |
| `lace` | like-for-like | **+0.3735** | +0.3657 / +0.3750 / +0.3796 | yes | no | **BEATEN OUTRIGHT** |
| `store_removed` | like-for-like | **+0.5648** | +0.5556 / +0.5694 / +0.5694 | yes | no | **BEATEN OUTRIGHT** |
| `x0/B_atom_mean_lookup` | ⭐ §C.1 matched-bytes LOOKUP FLOOR | **+0.1620** | +0.0440 / +0.1782 / +0.2639 | yes | no | **BEATEN OUTRIGHT** |
| `x0/B_row_lookup` | ⭐ §C.1 matched-bytes LOOKUP FLOOR | **+0.2431** | +0.2106 / +0.1944 / +0.3241 | yes | no | **BEATEN OUTRIGHT** |
| `x0/B_atoms_shuffled` | ⭐ §C.2 landscape-removed control | **+0.5278** | +0.4606 / +0.5394 / +0.5833 | yes | no | **BEATEN OUTRIGHT** |

- ceiling verdict: **LOSS** · like-for-like all beaten: **YES** · ⭐ §C.1 floor: **PASS** · §C.2 control collapses: **YES** · **PASS: NO**
- ⭐ the arm BEATS every matched-bytes lookup outright — the store is load-bearing on this metric

### ⭐⭐ THE BYTE LEDGER — MEASURED

| | `x0` rung A (measured) | `clu` (published) | `dynread` rung A |
|---|---|---|---|
| store / atoms | **524,288** | **524,288** | 524,288 |
| addressing head | **74,240** | **74,240** | 0 |
| coupling, DEPLOYED | **0** | 0 | 524,288 |
| × the 524,288 B budget | **1.00×** | 1.00× | 2.00× |

- byte-matched to `clu` field for field: **True**
- coupling needed: **False** — bit-identical `x0` across `coupling_None`, `coupling_fitted_at_certificate_lambda`, `coupling_strength_zero`, `request_strength_zero` (max |Δ| = 0.0 on all four)
- coupling bytes that LEAVE the ledger: **524,288 B**

### zero-step proof + §C.4

- seed 0: steps=0 · displacement from launch = 0.0 · decoder input IS the launch bitwise = True · energy trace = None · returns a stored atom = True · distinct index tuples = 19/72 · mean confidence = 0.0666
- seed 1: steps=0 · displacement from launch = 0.0 · decoder input IS the launch bitwise = True · energy trace = None · returns a stored atom = True · distinct index tuples = 22/72 · mean confidence = 0.0510
- seed 2: steps=0 · displacement from launch = 0.0 · decoder input IS the launch bitwise = True · energy trace = None · returns a stored atom = True · distinct index tuples = 29/72 · mean confidence = 0.0478
- `row_lookup` returns 2.67 distinct whole train rows of 72 requests

### instrument-ordering check

- by external: ['knn_proto', 'x0/B', 'clu', 'lace/ld_corrected'] …
- by probe:    ['knn_proto', 'x0/B', 'clu', 'lace'] …
- identical ordering: **False**
