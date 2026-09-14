# gen-commitment-collapse — tables (regenerated from disk)

## 1. Diagnosis — posterior, per set, per seed (⛔ is it the posterior or the argmax?)

| set | mean top-1 | mean top1−top2 margin | entropy bits/well (max 6) | degenerate? | argmax distinct |
|---|---|---|---|---|---|
| **A** | 0.7563 ± 0.0053 | 0.6032 ± 0.0080 | 1.0922 ± 0.0231 | **no** | 494 / 493 / 498 of 512 |
| **B** | 0.8326 ± 0.0278 | 0.7384 ± 0.0470 | 0.9117 ± 0.0797 | **no** | 9 / 8 / 10 of 72 |
| **MS** | 0.7655 ± 0.0263 | 0.6288 ± 0.0365 | 1.1277 ± 0.1162 | **no** | 35 / 24 / 24 of 198 |

## 2. ⭐ Factor leverage — which factor moves the committed code (seed-mean)


**A** — `code_flip_rate` (edit one factor ⇒ does the K-tuple move?) / `n_codes_from_factor_alone` / evidence share

| factor | floor_hue | wall_hue | object_hue | scale | shape | orientation |
|---|---|---|---|---|---|---|
| flip rate | 1.00 | 1.00 | 0.96 | 0.71 | 0.79 | 0.83 |
| codes alone | 5.3 | 10.0 | 6.7 | 2.3 | 1.7 | 5.7 |
| evidence share | 0.206 | 0.277 | 0.174 | 0.082 | 0.135 | 0.126 |

**B** — `code_flip_rate` (edit one factor ⇒ does the K-tuple move?) / `n_codes_from_factor_alone` / evidence share

| factor | floor_hue | wall_hue | object_hue | scale | shape | orientation |
|---|---|---|---|---|---|---|
| flip rate | 0.98 | 1.00 | 0.56 | 0.20 | 0.00 | 0.13 |
| codes alone | 2.0 | 2.0 | 1.7 | 1.3 | 1.0 | 1.3 |
| evidence share | 0.213 | 0.274 | 0.177 | 0.075 | 0.124 | 0.137 |

**MS** — `code_flip_rate` (edit one factor ⇒ does the K-tuple move?) / `n_codes_from_factor_alone` / evidence share

| factor | floor_hue | wall_hue | object_hue | scale | shape | orientation |
|---|---|---|---|---|---|---|
| flip rate | 0.99 | 1.00 | 0.63 | 0.28 | 0.84 | 0.17 |
| codes alone | 2.0 | 2.0 | 1.7 | 1.3 | 1.7 | 1.3 |
| evidence share | 0.210 | 0.270 | 0.174 | 0.075 | 0.135 | 0.136 |

## 3. Code-tuple overlap (R29 Q2's owed number), argmax, per seed

| pair | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **|B ∩ A|** | 0 of 9 | 0 of 8 | 0 of 10 |
| **|B ∩ MS|** | 9 of 9 | 7 of 8 | 10 of 10 |
| **|MS ∩ A|** | 0 of 35 | 0 of 24 | 0 of 24 |

## 4. The sweep — distinct codes, ⛔ every row with its MS companion (3-seed mean)

| config | B / 72 | MS / ~198 | A / 512 | realised width | abstain (B) |
|---|---|---|---|---|---|
| `argmax|tau=0.25|margin=0|k=1` | 9.0 ± 0.8 | 27.7 ± 5.2 | 495.0 ± 2.2 | 30.0 bits | 0.00 |
| `argmax|tau=0.5|margin=0|k=1` | 9.0 ± 0.8 | 27.7 ± 5.2 | 495.0 ± 2.2 | 30.0 bits | 0.00 |
| `argmax|tau=1|margin=0|k=1` | 9.0 ± 0.8 | 27.7 ± 5.2 | 495.0 ± 2.2 | 30.0 bits | 0.00 |
| `argmax|tau=2|margin=0|k=1` | 9.0 ± 0.8 | 27.7 ± 5.2 | 495.0 ± 2.2 | 30.0 bits | 0.00 |
| `argmax|tau=4|margin=0|k=1` | 9.0 ± 0.8 | 27.7 ± 5.2 | 495.0 ± 2.2 | 30.0 bits | 0.00 |
| `argmax|tau=8|margin=0|k=1` | 9.0 ± 0.8 | 27.7 ± 5.2 | 495.0 ± 2.2 | 30.0 bits | 0.00 |
| `sample|tau=0.25|margin=0|k=1` | 10.3 ± 1.7 | 35.7 ± 4.5 | 497.3 ± 2.6 | 30.0 bits | 0.00 |
| `sample|tau=0.5|margin=0|k=1` | 15.0 ± 2.2 | 58.0 ± 3.6 | 499.3 ± 4.1 | 30.0 bits | 0.00 |
| `sample|tau=1|margin=0|k=1` | 38.7 ± 0.9 | 120.3 ± 3.3 | 505.7 ± 0.5 | 30.0 bits | 0.00 |
| `sample|tau=2|margin=0|k=1` | 71.7 ± 0.5 | 198.0 ± 0.8 | 512.0 ± 0.0 | 30.0 bits | 0.00 |
| `sample|tau=4|margin=0|k=1` | 72.0 ± 0.0 | 198.7 ± 0.5 | 512.0 ± 0.0 | 30.0 bits | 0.00 |
| `sample|tau=8|margin=0|k=1` | 72.0 ± 0.0 | 198.7 ± 0.5 | 512.0 ± 0.0 | 30.0 bits | 0.00 |
| `margin|tau=0.25|margin=0.1|k=1` | 10.0 ± 1.4 | 34.7 ± 5.4 | 496.0 ± 2.2 | 30.0 bits | 0.00 |
| `margin|tau=0.25|margin=0.25|k=1` | 11.0 ± 2.2 | 38.0 ± 4.1 | 499.3 ± 2.1 | 30.1 bits ⛔ | 0.01 |
| `margin|tau=0.25|margin=0.5|k=1` | 10.3 ± 0.9 | 40.7 ± 7.4 | 499.3 ± 2.5 | 30.1 bits ⛔ | 0.03 |
| `margin|tau=0.25|margin=0.75|k=1` | 11.7 ± 2.5 | 44.7 ± 6.8 | 501.0 ± 2.4 | 30.1 bits ⛔ | 0.05 |
| `margin|tau=0.25|margin=0.9|k=1` | 12.0 ± 4.1 | 47.7 ± 9.5 | 494.0 ± 3.6 | 30.1 bits ⛔ | 0.10 |
| `margin|tau=0.5|margin=0.1|k=1` | 10.7 ± 2.4 | 36.3 ± 4.7 | 497.7 ± 1.7 | 30.0 bits | 0.01 |
| `margin|tau=0.5|margin=0.25|k=1` | 10.3 ± 0.9 | 39.7 ± 6.9 | 499.7 ± 2.9 | 30.1 bits ⛔ | 0.03 |
| `margin|tau=0.5|margin=0.5|k=1` | 11.7 ± 2.9 | 44.0 ± 8.0 | 500.3 ± 1.9 | 30.1 bits ⛔ | 0.06 |
| `margin|tau=0.5|margin=0.75|k=1` | 13.7 ± 2.1 | 47.0 ± 1.6 | 485.7 ± 3.7 | 30.1 bits ⛔ | 0.13 |
| `margin|tau=0.5|margin=0.9|k=1` | 15.3 ± 1.7 | 46.0 ± 3.6 | 450.3 ± 8.1 | 30.1 bits ⛔ | 0.22 |
| `margin|tau=1|margin=0.1|k=1` | 10.3 ± 0.9 | 40.0 ± 6.5 | 499.0 ± 2.4 | 30.1 bits ⛔ | 0.03 |
| `margin|tau=1|margin=0.25|k=1` | 11.0 ± 3.3 | 45.3 ± 7.7 | 498.7 ± 2.4 | 30.1 bits ⛔ | 0.07 |
| `margin|tau=1|margin=0.5|k=1` | 14.7 ± 0.9 | 47.7 ± 1.2 | 472.0 ± 7.8 | 30.1 bits ⛔ | 0.19 |
| `margin|tau=1|margin=0.75|k=1` | 17.7 ± 4.8 | 29.7 ± 6.8 | 365.3 ± 9.9 | 30.1 bits ⛔ | 0.37 |
| `margin|tau=1|margin=0.9|k=1` | 11.7 ± 2.5 | 14.0 ± 3.3 | 209.0 ± 10.6 | 30.1 bits ⛔ | 0.61 |
| `margin|tau=2|margin=0.1|k=1` | 11.7 ± 2.9 | 48.0 ± 7.9 | 494.7 ± 3.3 | 30.1 bits ⛔ | 0.11 |
| `margin|tau=2|margin=0.25|k=1` | 17.7 ± 5.0 | 34.3 ± 10.1 | 393.0 ± 7.8 | 30.1 bits ⛔ | 0.34 |
| `margin|tau=2|margin=0.5|k=1` | 5.3 ± 2.1 | 7.0 ± 1.6 | 86.0 ± 2.2 | 30.1 bits ⛔ | 0.88 |
| `margin|tau=2|margin=0.75|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=2|margin=0.9|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=4|margin=0.1|k=1` | 8.3 ± 2.1 | 11.3 ± 1.9 | 112.7 ± 4.2 | 30.1 bits ⛔ | 0.84 |
| `margin|tau=4|margin=0.25|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=4|margin=0.5|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=4|margin=0.75|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=4|margin=0.9|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=8|margin=0.1|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=8|margin=0.25|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=8|margin=0.5|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=8|margin=0.75|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `margin|tau=8|margin=0.9|k=1` | 1.0 ± 0.0 | 1.0 ± 0.0 | 1.0 ± 0.0 | 30.1 bits ⛔ | 1.00 |
| `topk|tau=0.25|margin=0|k=2` | 32.7 ± 9.8 | 79.0 ± 8.6 | 506.7 ± 0.9 | 54.9 bits ⛔ | 0.00 |
| `topk|tau=0.25|margin=0|k=3` | 46.7 ± 7.1 | 117.7 ± 8.7 | 509.7 ± 0.9 | 76.7 bits ⛔ | 0.00 |
| `topk|tau=0.25|margin=0|k=4` | 50.7 ± 9.9 | 148.3 ± 18.3 | 510.3 ± 0.9 | 96.4 bits ⛔ | 0.00 |
| `topk|tau=0.25|margin=0|k=8` | 70.3 ± 1.2 | 193.3 ± 3.1 | 512.0 ± 0.0 | 160.2 bits ⛔ | 0.00 |
| `topk|tau=0.5|margin=0|k=2` | 32.7 ± 9.8 | 79.0 ± 8.6 | 506.7 ± 0.9 | 54.9 bits ⛔ | 0.00 |
| `topk|tau=0.5|margin=0|k=3` | 46.7 ± 7.1 | 117.7 ± 8.7 | 509.7 ± 0.9 | 76.7 bits ⛔ | 0.00 |
| `topk|tau=0.5|margin=0|k=4` | 50.7 ± 9.9 | 148.3 ± 18.3 | 510.3 ± 0.9 | 96.4 bits ⛔ | 0.00 |
| `topk|tau=0.5|margin=0|k=8` | 70.3 ± 1.2 | 193.3 ± 3.1 | 512.0 ± 0.0 | 160.2 bits ⛔ | 0.00 |
| `topk|tau=1|margin=0|k=2` | 32.7 ± 9.8 | 79.0 ± 8.6 | 506.7 ± 0.9 | 54.9 bits ⛔ | 0.00 |
| `topk|tau=1|margin=0|k=3` | 46.7 ± 7.1 | 117.7 ± 8.7 | 509.7 ± 0.9 | 76.7 bits ⛔ | 0.00 |
| `topk|tau=1|margin=0|k=4` | 50.7 ± 9.9 | 148.3 ± 18.3 | 510.3 ± 0.9 | 96.4 bits ⛔ | 0.00 |
| `topk|tau=1|margin=0|k=8` | 70.3 ± 1.2 | 193.3 ± 3.1 | 512.0 ± 0.0 | 160.2 bits ⛔ | 0.00 |
| `topk|tau=2|margin=0|k=2` | 32.7 ± 9.8 | 79.0 ± 8.6 | 506.7 ± 0.9 | 54.9 bits ⛔ | 0.00 |
| `topk|tau=2|margin=0|k=3` | 46.7 ± 7.1 | 117.7 ± 8.7 | 509.7 ± 0.9 | 76.7 bits ⛔ | 0.00 |
| `topk|tau=2|margin=0|k=4` | 50.7 ± 9.9 | 148.3 ± 18.3 | 510.3 ± 0.9 | 96.4 bits ⛔ | 0.00 |
| `topk|tau=2|margin=0|k=8` | 70.3 ± 1.2 | 193.3 ± 3.1 | 512.0 ± 0.0 | 160.2 bits ⛔ | 0.00 |
| `topk|tau=4|margin=0|k=2` | 32.7 ± 9.8 | 79.0 ± 8.6 | 506.7 ± 0.9 | 54.9 bits ⛔ | 0.00 |
| `topk|tau=4|margin=0|k=3` | 46.7 ± 7.1 | 117.7 ± 8.7 | 509.7 ± 0.9 | 76.7 bits ⛔ | 0.00 |
| `topk|tau=4|margin=0|k=4` | 50.7 ± 9.9 | 148.3 ± 18.3 | 510.3 ± 0.9 | 96.4 bits ⛔ | 0.00 |
| `topk|tau=4|margin=0|k=8` | 70.3 ± 1.2 | 193.3 ± 3.1 | 512.0 ± 0.0 | 160.2 bits ⛔ | 0.00 |
| `topk|tau=8|margin=0|k=2` | 32.7 ± 9.8 | 79.0 ± 8.6 | 506.7 ± 0.9 | 54.9 bits ⛔ | 0.00 |
| `topk|tau=8|margin=0|k=3` | 46.7 ± 7.1 | 117.7 ± 8.7 | 509.7 ± 0.9 | 76.7 bits ⛔ | 0.00 |
| `topk|tau=8|margin=0|k=4` | 50.7 ± 9.9 | 148.3 ± 18.3 | 510.3 ± 0.9 | 96.4 bits ⛔ | 0.00 |
| `topk|tau=8|margin=0|k=8` | 70.3 ± 1.2 | 193.3 ± 3.1 | 512.0 ± 0.0 | 160.2 bits ⛔ | 0.00 |

## 5. ⭐ The accuracy leg — the pre-registered shortlist, 3 seeds

| config | B distinct | MS distinct | in-corner `shape` balacc | AUROC | A aggregate | width | PASS? |
|---|---|---|---|---|---|---|---|
| `argmax|tau=1|margin=0|k=1` | 9.0 ± 0.8 | 27.7 ± 5.2 | **0.6759 ± 0.0107** | 0.7429 ± 0.0080 | 0.6174 ± 0.0070 | 30.0 bits | FAIL |
| `margin|tau=2|margin=0.25|k=1` | 17.7 ± 5.0 | 34.3 ± 10.1 | **0.5872 ± 0.0445** | 0.6583 ± 0.0465 | 0.3735 ± 0.0151 | 30.1 bits ⛔ | FAIL |
| `sample|tau=2|margin=0|k=1` | 71.7 ± 0.5 | 198.0 ± 0.8 | **0.6319 ± 0.0332** | 0.6817 ± 0.0437 | 0.4368 ± 0.0066 | 30.0 bits | FAIL |
| `topk|tau=1|margin=0|k=8` | 70.3 ± 1.2 | 193.3 ± 3.1 | **0.7408 ± 0.0323** | 0.7910 ± 0.0107 | 0.6603 ± 0.0105 | 160.2 bits ⛔ | FAIL |
| *target* `knn_proto`/`proto` | 72 / 72 | — | *0.8817 ± 0.0023* | — | — | unbounded | — |

### per-factor probe accuracy on B (3-seed mean)

| config | floor_hue | object_hue | orientation | scale | shape | wall_hue |
|---|---|---|---|---|---|---|
| `argmax|tau=1|margin=0|k=1` | 0.991 | 0.671 | 0.194 | 0.019 | 0.833 | 1.000 |
| `margin|tau=2|margin=0.25|k=1` | 0.528 | 0.602 | 0.245 | 0.176 | 0.458 | 0.838 |
| `sample|tau=2|margin=0|k=1` | 0.648 | 0.398 | 0.144 | 0.125 | 0.630 | 0.593 |
| `topk|tau=1|margin=0|k=8` | 0.995 | 0.954 | 0.194 | 0.000 | 0.940 | 1.000 |
