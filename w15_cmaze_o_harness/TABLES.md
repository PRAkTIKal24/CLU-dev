
### `valid` — VAL, 3 seeds, mean ± sd over seeds

Source: `valsmoke/VAL-SMOKE.json` → `table.<arm>.<row>.valid`. `tolerance_used` is `null` on every cell.

| arm | seen | unseen 2-way | unseen n-way (n=4) | value hold-out (strict) | value hold-out (slot) |
|---|---|---|---|---|---|
| AE ceiling (encode the true picture, decode, judge) | 0.7522 ± 0.1767 | 0.2946 ± 0.0574 | 0.4767 ± 0.1203 | 0.0000 ± 0.0000 | 0.0092 ± 0.0014 |
| **CLU — Config-A single design** | 0.2728 ± 0.0913 | 0.0657 ± 0.0315 | 0.1158 ± 0.0396 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| MLP, matched params | 0.1211 ± 0.0301 | 0.0210 ± 0.0119 | 0.0533 ± 0.0270 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| continuous MLP, matched params | 0.0006 ± 0.0010 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| LSTM, matched params | 0.0094 ± 0.0042 | 0.0000 ± 0.0000 | 0.0033 ± 0.0029 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| transformer, matched params | 0.1817 ± 0.0289 | 0.0412 ± 0.0287 | 0.0792 ± 0.0447 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| matched-bytes LOOKUP **floor** (never a rival) | 0.1506 ± 0.0428 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_removed | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_shuffled | 0.2728 ± 0.0913 | 0.0657 ± 0.0315 | 0.1158 ± 0.0396 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| control: landscape_shuffled_across_wells | 0.1733 ± 0.0353 | 0.0421 ± 0.0162 | 0.0650 ± 0.0115 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: random_atoms | 0.1500 ± 0.0347 | 0.0404 ± 0.0101 | 0.0667 ± 0.0267 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| BLIND: layout-free cells | 0.0589 ± 0.0054 | 0.0320 ± 0.0119 | 0.0625 ± 0.0303 | 0.0608 ± 0.0063 | 0.0508 ± 0.0029 |
| BLIND: colour marginal | 0.0061 ± 0.0079 | 0.0034 ± 0.0029 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BLIND: fully factorised marginal | 0.0011 ± 0.0019 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BLIND: no layout word | 0.1117 ± 0.0060 | 0.0530 ± 0.0067 | 0.0100 ± 0.0066 | 0.0992 ± 0.0115 | 0.0817 ± 0.0080 |
| RANDOM state | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

Row sizes per seed: seen n=[600, 600, 600], unseen 2-way n=[396, 396, 396], unseen n-way (n=4) n=[400, 400, 400], value hold-out (strict) n=[400, 400, 400], value hold-out (slot) n=[400, 400, 400].

### `exact_objects` — VAL, 3 seeds, mean ± sd over seeds

Source: `valsmoke/VAL-SMOKE.json` → `table.<arm>.<row>.exact_objects`. `tolerance_used` is `null` on every cell.

| arm | seen | unseen 2-way | unseen n-way (n=4) | value hold-out (strict) | value hold-out (slot) |
|---|---|---|---|---|---|
| AE ceiling (encode the true picture, decode, judge) | 0.7522 ± 0.1767 | 0.2946 ± 0.0574 | 0.4792 ± 0.1227 | 0.0000 ± 0.0000 | 0.0092 ± 0.0014 |
| **CLU — Config-A single design** | 0.2728 ± 0.0913 | 0.0665 ± 0.0312 | 0.1192 ± 0.0416 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| MLP, matched params | 0.1211 ± 0.0301 | 0.0210 ± 0.0119 | 0.0533 ± 0.0270 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| continuous MLP, matched params | 0.0006 ± 0.0010 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| LSTM, matched params | 0.0094 ± 0.0042 | 0.0000 ± 0.0000 | 0.0042 ± 0.0014 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| transformer, matched params | 0.1817 ± 0.0289 | 0.0412 ± 0.0287 | 0.0792 ± 0.0447 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| matched-bytes LOOKUP **floor** (never a rival) | 0.1506 ± 0.0428 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_removed | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_shuffled | 0.2728 ± 0.0913 | 0.0665 ± 0.0312 | 0.1192 ± 0.0416 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| control: landscape_shuffled_across_wells | 0.1739 ± 0.0362 | 0.0429 ± 0.0154 | 0.0692 ± 0.0151 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: random_atoms | 0.1528 ± 0.0323 | 0.0404 ± 0.0101 | 0.0692 ± 0.0267 | 0.0000 ± 0.0000 | 0.0008 ± 0.0014 |
| BLIND: layout-free cells | 0.0589 ± 0.0054 | 0.0320 ± 0.0119 | 0.0625 ± 0.0303 | 0.0608 ± 0.0063 | 0.0508 ± 0.0029 |
| BLIND: colour marginal | 0.0061 ± 0.0079 | 0.0034 ± 0.0029 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BLIND: fully factorised marginal | 0.0011 ± 0.0019 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| BLIND: no layout word | 0.1133 ± 0.0058 | 0.0530 ± 0.0067 | 0.0125 ± 0.0109 | 0.1000 ± 0.0109 | 0.0817 ± 0.0080 |
| RANDOM state | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

Row sizes per seed: seen n=[600, 600, 600], unseen 2-way n=[396, 396, 396], unseen n-way (n=4) n=[400, 400, 400], value hold-out (strict) n=[400, 400, 400], value hold-out (slot) n=[400, 400, 400].

### `legal` — VAL, 3 seeds, mean ± sd over seeds

Source: `valsmoke/VAL-SMOKE.json` → `table.<arm>.<row>.legal`. `tolerance_used` is `null` on every cell.

| arm | seen | unseen 2-way | unseen n-way (n=4) | value hold-out (strict) | value hold-out (slot) |
|---|---|---|---|---|---|
| AE ceiling (encode the true picture, decode, judge) | 0.9744 ± 0.0084 | 0.8561 ± 0.0617 | 0.9292 ± 0.0146 | 0.8508 ± 0.0362 | 0.8983 ± 0.0435 |
| **CLU — Config-A single design** | 0.8600 ± 0.0117 | 0.8089 ± 0.0511 | 0.8575 ± 0.0347 | 0.8275 ± 0.0043 | 0.8092 ± 0.0238 |
| MLP, matched params | 0.8228 ± 0.0019 | 0.8232 ± 0.0397 | 0.8250 ± 0.0278 | 0.8050 ± 0.0000 | 0.8083 ± 0.0184 |
| continuous MLP, matched params | 0.7317 ± 0.0144 | 0.7332 ± 0.0780 | 0.7442 ± 0.0260 | 0.6817 ± 0.0923 | 0.6892 ± 0.0191 |
| LSTM, matched params | 0.8350 ± 0.0209 | 0.7837 ± 0.0316 | 0.8208 ± 0.0302 | 0.7850 ± 0.0090 | 0.8017 ± 0.0624 |
| transformer, matched params | 0.8317 ± 0.0033 | 0.7685 ± 0.0501 | 0.7950 ± 0.0304 | 0.7808 ± 0.0397 | 0.7975 ± 0.0295 |
| matched-bytes LOOKUP **floor** (never a rival) | 0.9072 ± 0.0042 | 0.8779 ± 0.0039 | 0.8850 ± 0.0263 | 0.8775 ± 0.0156 | 0.8608 ± 0.0138 |
| control: landscape_removed | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_shuffled | 0.8600 ± 0.0117 | 0.8089 ± 0.0511 | 0.8575 ± 0.0347 | 0.8275 ± 0.0043 | 0.8092 ± 0.0238 |
| control: landscape_shuffled_across_wells | 0.8067 ± 0.0202 | 0.7938 ± 0.0884 | 0.7917 ± 0.0126 | 0.7917 ± 0.0038 | 0.7608 ± 0.0270 |
| control: random_atoms | 0.8106 ± 0.0063 | 0.7803 ± 0.0291 | 0.8092 ± 0.0411 | 0.7717 ± 0.0063 | 0.7250 ± 0.0150 |
| BLIND: layout-free cells | 0.4161 ± 0.0570 | 0.4251 ± 0.1672 | 0.4475 ± 0.0565 | 0.4183 ± 0.0379 | 0.4083 ± 0.0397 |
| BLIND: colour marginal | 0.8150 ± 0.1603 | 0.8561 ± 0.1247 | 0.8258 ± 0.1510 | 0.8250 ± 0.1516 | 0.8158 ± 0.1595 |
| BLIND: fully factorised marginal | 0.3656 ± 0.0801 | 0.4007 ± 0.1735 | 0.4283 ± 0.0841 | 0.3792 ± 0.0674 | 0.3658 ± 0.0751 |
| BLIND: no layout word | 0.8633 ± 0.0613 | 0.8653 ± 0.0722 | 0.8342 ± 0.0884 | 0.8392 ± 0.0713 | 0.8517 ± 0.0690 |
| RANDOM state | 0.7606 ± 0.0243 | 0.7483 ± 0.0439 | 0.7583 ± 0.0363 | 0.7367 ± 0.0189 | 0.7533 ± 0.0151 |

Row sizes per seed: seen n=[600, 600, 600], unseen 2-way n=[396, 396, 396], unseen n-way (n=4) n=[400, 400, 400], value hold-out (strict) n=[400, 400, 400], value hold-out (slot) n=[400, 400, 400].

### `exact_layout` — VAL, 3 seeds, mean ± sd over seeds

Source: `valsmoke/VAL-SMOKE.json` → `table.<arm>.<row>.exact_layout`. `tolerance_used` is `null` on every cell.

| arm | seen | unseen 2-way | unseen n-way (n=4) | value hold-out (strict) | value hold-out (slot) |
|---|---|---|---|---|---|
| AE ceiling (encode the true picture, decode, judge) | 0.9994 ± 0.0010 | 0.8822 ± 0.0675 | 0.9342 ± 0.0275 | 0.7417 ± 0.1039 | 0.9108 ± 0.0406 |
| **CLU — Config-A single design** | 0.9706 ± 0.0127 | 0.8687 ± 0.0910 | 0.8867 ± 0.0636 | 0.9075 ± 0.0222 | 0.8400 ± 0.0597 |
| MLP, matched params | 0.9922 ± 0.0092 | 0.9663 ± 0.0168 | 0.9858 ± 0.0029 | 0.9850 ± 0.0195 | 0.9783 ± 0.0161 |
| continuous MLP, matched params | 0.3928 ± 0.0712 | 0.5008 ± 0.1048 | 0.5417 ± 0.2532 | 0.3025 ± 0.1345 | 0.3967 ± 0.0569 |
| LSTM, matched params | 0.9956 ± 0.0025 | 0.9781 ± 0.0139 | 0.9858 ± 0.0101 | 0.8183 ± 0.0593 | 0.9108 ± 0.0539 |
| transformer, matched params | 0.9956 ± 0.0025 | 0.9672 ± 0.0219 | 0.9808 ± 0.0113 | 0.8983 ± 0.1227 | 0.9742 ± 0.0128 |
| matched-bytes LOOKUP **floor** (never a rival) | 0.9994 ± 0.0010 | 1.0000 ± 0.0000 | 0.9975 ± 0.0043 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| control: landscape_removed | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| control: landscape_shuffled | 0.9706 ± 0.0127 | 0.8687 ± 0.0910 | 0.8867 ± 0.0636 | 0.9075 ± 0.0222 | 0.8400 ± 0.0597 |
| control: landscape_shuffled_across_wells | 0.8111 ± 0.0728 | 0.6877 ± 0.1494 | 0.7550 ± 0.1070 | 0.7350 ± 0.0978 | 0.6758 ± 0.0906 |
| control: random_atoms | 0.7561 ± 0.0415 | 0.6423 ± 0.0077 | 0.6875 ± 0.0667 | 0.6550 ± 0.0611 | 0.6050 ± 0.0346 |
| BLIND: layout-free cells | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| BLIND: colour marginal | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| BLIND: fully factorised marginal | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| BLIND: no layout word | 0.0989 ± 0.0082 | 0.0505 ± 0.0051 | 0.0000 ± 0.0000 | 0.0867 ± 0.0101 | 0.0733 ± 0.0101 |
| RANDOM state | 0.0711 ± 0.0100 | 0.0682 ± 0.0158 | 0.0742 ± 0.0052 | 0.0842 ± 0.0095 | 0.0883 ± 0.0201 |

Row sizes per seed: seen n=[600, 600, 600], unseen 2-way n=[396, 396, 396], unseen n-way (n=4) n=[400, 400, 400], value hold-out (strict) n=[400, 400, 400], value hold-out (slot) n=[400, 400, 400].

### Memory ledger — printed separately, never matched (COMMON.md)

Source: `valsmoke/VAL-SMOKE.json` → `per_seed_ledgers.<seed>`. Seed count is a column. Byte convention is named on every figure.

| object | seed 0 | seed 1 | seed 2 | seeds | byte convention |
|---|---|---|---|---|---|
| CLU deployed read floats (atoms + address book) | 202,848 | 202,848 | 202,848 | 3 | 4 bytes/float, float32 |
| … of which atoms | 1,536 | 1,536 | 1,536 | 3 | float32 |
| … of which address book | 201,312 | 201,312 | 201,312 | 3 | float32 |
| CLU deployed bytes | 811,392 | 811,392 | 811,392 | 3 | float32 |
| CLU float64 sufficient statistic (bytes) | 9,792 | 9,792 | 9,792 | 3 | float64 — buys exact deletion, **never netted in, never matched** |
| CLU span fraction (n_atoms / d_k) | 0.688 | 0.688 | 0.688 | 3 | — |
| LOOKUP floor items held | 1,584 | 1,584 | 1,584 | 3 | matched to the CLU's read floats |
| rival mlp params (matched) | 203,210 (matched) | 203,210 (matched) | 203,210 (matched) | 3 | params × 4 bytes, float32 |
| rival cont_mlp params (matched) | 202,588 (matched) | 202,588 (matched) | 202,588 (matched) | 3 | params × 4 bytes, float32 |
| rival lstm params (matched) | 202,028 (matched) | 202,028 (matched) | 202,028 (matched) | 3 | params × 4 bytes, float32 |
| rival transformer params (matched) | 204,572 (matched) | 204,572 (matched) | 204,572 (matched) | 3 | params × 4 bytes, float32 |
| shared frozen AE params (encoder + decoder) | 2,292,115 | 2,292,115 | 2,292,115 | 3 | declared OUTSIDE the deletable object; identical for every arm |

### Deletion rows — VAL, colour c = 2

Source: `valsmoke/VAL-SMOKE.json` → `deletion.<seed>`; `valsmoke/seed<N>.json` → `deletion`.

| deletion | seed | rows deleted | store-scoped max abs ΔS | rel. ΔS | Δn | CLU adaptation (s) | rival retrain-without (s) | speed-up |
|---|---|---|---|---|---|---|---|---|
| value | 0 | 5530 (69.1%) | 2.037e-10 | 2.723e-15 | 0 | 3.88 | 15 | 4× |
| value | 1 | 5250 (65.6%) | 1.746e-10 | 7.740e-16 | 0 | 3.48 | 17 | 5× |
| value | 2 | 5959 (74.5%) | 1.746e-10 | 1.604e-15 | 0 | 2.19 | 20 | 9× |
| conjunction | 0 | 2240 (28.0%) | 1.164e-10 | 5.908e-16 | 0 | 3.25 | 17 | 5× |
| conjunction | 1 | 1435 (17.9%) | 2.328e-10 | 4.353e-16 | 0 | 2.02 | 16 | 8× |
| conjunction | 2 | 2168 (27.1%) | 1.746e-10 | 5.952e-16 | 0 | 3.49 | 12 | 4× |

#### Post-deletion composition on the OTHER colours (`valid`)

| deletion | row | side | CLU as-trained | CLU after deletion | CLU refit-without | MLP as-trained | MLP retrain-without | n per seed |
|---|---|---|---|---|---|---|---|---|
| value | seen | other_colours | 0.2621 | 0.3663 | 0.3663 | 0.1285 | 0.1953 | 190+208+162 |
| value | seen | deleted_colour | 0.2764 | 0.0814 | 0.0814 | 0.1181 | 0.0380 | 410+392+438 |
| value | unseen 2-way | other_colours | 0.0495 | 0.0307 | 0.0307 | 0.0354 | 0.0246 | 116+65+149 |
| value | unseen 2-way | deleted_colour | 0.0682 | 0.0148 | 0.0148 | 0.0125 | 0.0081 | 280+331+247 |
| value | unseen n-way (n=4) | other_colours | 0.0725 | 0.0490 | 0.0490 | 0.0490 | 0.0691 | 211+101+67 |
| value | unseen n-way (n=4) | deleted_colour | 0.1125 | 0.0219 | 0.0219 | 0.0497 | 0.0242 | 189+299+333 |
| conjunction | seen | other_colours | 0.2845 | 0.3173 | 0.3173 | 0.1271 | 0.1396 | 441+495+446 |
| conjunction | seen | deleted_colour | 0.2518 | 0.0000 | 0.0000 | 0.1113 | 0.0000 | 159+105+154 |
| conjunction | unseen 2-way | other_colours | 0.0654 | 0.0617 | 0.0617 | 0.0236 | 0.0246 | 317+254+326 |
| conjunction | unseen 2-way | deleted_colour | 0.0590 | 0.0000 | 0.0000 | 0.0094 | 0.0000 | 79+142+70 |
| conjunction | unseen n-way (n=4) | other_colours | 0.1063 | 0.1044 | 0.1044 | 0.0566 | 0.0608 | 350+250+250 |
| conjunction | unseen n-way (n=4) | deleted_colour | 0.1133 | 0.0000 | 0.0000 | 0.0422 | 0.0000 | 50+150+150 |
