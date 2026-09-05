# Rider 1 — the **n = 9 FULL-COLUMN** re-aggregation (F3 follow-up 2)

⭐ **What this retires:** draft App. I.1c's *"un-aggregated columns"* caveat and the mixed-n
labelling. Every column below is at a **uniform n = 9** (seeds 0–8), aggregated with the shipped
`exp_bprime_rivals.audit_table` so the rule is byte-identical to the one that produced the published
table. **Nothing is re-measured** — this is pure re-aggregation of
`.claude/outputs/bprime-rivals-f3/{run400,seeds3to8,repro_c2w4,repro_c2w4_s3to8}`.

**Family:** `aggregate@base` — the SOLE reader-discrimination family (`S = 0.5068`).
**Convention:** sample sd (ddof = 1); SE = sd/√n; **n = 9**. Metric `neg_mae` (higher = better).
**The CLU column is BANKED** (PREREG-Bprime §7) and is never re-derived.

⚠ **Fidelity check:** every quantity the F3 report published at n = 9 reproduces **digit-for-digit**
(raw-table margins −0.4602 / −0.4425 / −0.2732 / −0.2600 / −0.2592; lifts 0.093 ± 0.134 /
−0.071 ± 0.090 / 0.294 ± 0.077 / 0.880 ± 0.227 / 1.025 ± 0.329). What is **new** is the rest of the
row at the same n: the **same-keys null**, the **launder**, the three **+0 B readers**, the paired
`full − null` / `full − blank` statistics, and the **byte ledger**.


## A. `f3` — the F3 report's PRIMARY column, n = 9

*the F3 grid (6 lr x 2 wd), 400 steps, best-of-grid on the fit split — the F3 report's PRIMARY column* · seeds [0, 1, 2, 3, 4, 5, 6, 7, 8]

| arm | `full` | `launder` | **same-keys null** | `blank` | **full − null** (paired) | **lift = full − blank** (paired) | RESCUED? |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | -0.6075 ± 0.1096 | -0.4235 ± 0.0145 | -0.4012 ± 0.0164 | -0.7008 ± 0.0673 | -0.2063 ± 0.1016 | +0.0933 ± 0.1337 | ⛔ |
| **ttt_mlp** | -0.5898 ± 0.0731 | -0.4104 ± 0.0174 | -0.3903 ± 0.0191 | -0.5189 ± 0.0416 | -0.1995 ± 0.0665 | -0.0709 ± 0.0901 | ⛔ |
| **deltanet** | -0.4205 ± 0.0299 | -0.5720 ± 0.0653 | -0.6379 ± 0.0708 | -0.7147 ± 0.0800 | +0.2174 ± 0.0749 | +0.2943 ± 0.0766 | ✅ |
| **gdn** | -0.4073 ± 0.0120 | -1.0033 ± 0.0952 | -0.9715 ± 0.0982 | -1.2869 ± 0.2317 | +0.5642 ± 0.1032 | +0.8796 ± 0.2273 | ✅ |
| **gdn2** | -0.4065 ± 0.0178 | -1.0889 ± 0.0815 | -1.1503 ± 0.1165 | -1.4319 ± 0.3241 | +0.7438 ± 0.1242 | +1.0254 ± 0.3293 | ✅ |

| arm | **+0 B margin** (own table) | **RAW-table margin** | SE below 0 | `knn2_mean_+0B` | `knn2_idw_+0B` | `table_mean_+0B` | P5-vs-raw gap |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | -0.2213 ± 0.1062 | **-0.4602 ± 0.1038** | **4.43 SE** | -0.4074 ± 0.0094 | -0.4091 ± 0.0099 | -0.3938 ± 0.0137 | +0.2762 ± 0.0285 |
| **ttt_mlp** | -0.2095 ± 0.0683 | **-0.4425 ± 0.0869** | **5.09 SE** | -0.4067 ± 0.0115 | -0.4063 ± 0.0117 | -0.3930 ± 0.0135 | +0.2631 ± 0.0307 |
| **deltanet** | -0.0172 ± 0.0263 | **-0.2732 ± 0.0395** | **6.92 SE** | -0.5275 ± 0.0588 | -0.5266 ± 0.0589 | -0.4151 ± 0.0181 | +0.4246 ± 0.0672 |
| **gdn** | -0.0102 ± 0.0229 | **-0.2600 ± 0.0278** | **9.35 SE** | -0.8089 ± 0.0420 | -0.8129 ± 0.0468 | -0.3971 ± 0.0195 | +0.8560 ± 0.0907 |
| **gdn2** | +0.0473 ± 0.0277 | **-0.2592 ± 0.0292** | **8.87 SE** | -0.8752 ± 0.0817 | -0.8796 ± 0.0818 | -0.4538 ± 0.0257 | +0.9416 ± 0.0913 |

| arm | `d_head` | rival **state B** | rival **param B** | matched **table B** | state/table | table lossless? | metric-native |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | 36 | **5220 / 5328** | **5592 / 7944** | **5104 / 5184** | 1.0255 [1.0227, 1.0278] | ✅ | metric-native |
| **ttt_mlp** | 12 | **4656 / 5376** | 5736 | **4608 / 5376** | 1.0012 [1.0000, 1.0104] | ✅ | weakly metric-native |
| **deltanet** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |
| **gdn** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |
| **gdn2** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |

⚠ **A bold pair of values means the ledger is NOT constant across the 9 seeds.** Best-of-grid selects the TTT mini-batch `b` per seed and `b` enters the declared state (the in-flight buffer), so at n = 9 the TTT rows' byte columns are a **per-seed quantity**: `ttt_linear` b = [1, 16, 16, 1, 1, 16, 1, 16, 1], `ttt_mlp` b = [16, 16, 16, 16, 16, 16, 16, 16, 1]. The delta arms are constant. ⛔ A single TTT byte figure must never be quoted as *the* n = 9 value.

## B. C2W4's OWN code path (sequential init split), n = 9

*C2W4's OWN code path (sequential init split), same seeds* · seeds [0, 1, 2, 3, 4, 5, 6, 7, 8]

| arm | `full` | `launder` | **same-keys null** | `blank` | **full − null** (paired) | **lift = full − blank** (paired) | RESCUED? |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | -0.6025 ± 0.0704 | -0.4319 ± 0.0121 | -0.4132 ± 0.0181 | -0.9222 ± 0.1114 | -0.1893 ± 0.0748 | +0.3197 ± 0.0826 | ✅ |
| **ttt_mlp** | -0.5409 ± 0.0744 | -0.3978 ± 0.0129 | -0.4043 ± 0.0213 | -0.6339 ± 0.1079 | -0.1367 ± 0.0845 | +0.0929 ± 0.1072 | ⛔ |
| **deltanet** | -0.4530 ± 0.0230 | -0.4938 ± 0.0538 | -0.5701 ± 0.0653 | -0.5937 ± 0.0329 | +0.1170 ± 0.0764 | +0.1407 ± 0.0461 | ✅ |
| **gdn** | -0.4406 ± 0.0290 | -1.0740 ± 0.0955 | -1.0781 ± 0.0829 | -1.3873 ± 0.1571 | +0.6375 ± 0.0928 | +0.9466 ± 0.1486 | ✅ |
| **gdn2** | -0.4143 ± 0.0302 | -1.0274 ± 0.1518 | -1.0976 ± 0.0989 | -1.7984 ± 0.2900 | +0.6832 ± 0.0846 | +1.3840 ± 0.2764 | ✅ |

| arm | **+0 B margin** (own table) | **RAW-table margin** | SE below 0 | `knn2_mean_+0B` | `knn2_idw_+0B` | `table_mean_+0B` | P5-vs-raw gap |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | -0.2230 ± 0.0685 | **-0.4551 ± 0.0801** | **5.68 SE** | -0.4192 ± 0.0083 | -0.4195 ± 0.0086 | -0.3867 ± 0.0134 | +0.2846 ± 0.0242 |
| **ttt_mlp** | -0.1601 ± 0.0734 | **-0.3936 ± 0.0721** | **5.46 SE** | -0.3893 ± 0.0125 | -0.3898 ± 0.0125 | -0.3894 ± 0.0109 | +0.2505 ± 0.0247 |
| **deltanet** | -0.0640 ± 0.0303 | **-0.3057 ± 0.0316** | **9.67 SE** | -0.4626 ± 0.0498 | -0.4627 ± 0.0493 | -0.4118 ± 0.0220 | +0.3465 ± 0.0486 |
| **gdn** | -0.0392 ± 0.0343 | **-0.2933 ± 0.0408** | **7.19 SE** | -0.8802 ± 0.0583 | -0.8806 ± 0.0608 | -0.4015 ± 0.0207 | +0.9267 ± 0.0827 |
| **gdn2** | +0.0197 ± 0.0603 | **-0.2670 ± 0.0478** | **5.59 SE** | -0.8814 ± 0.1389 | -0.8789 ± 0.1389 | -0.4340 ± 0.0500 | +0.8801 ± 0.1556 |

| arm | `d_head` | rival **state B** | rival **param B** | matched **table B** | state/table | table lossless? | metric-native |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | 29 | **5220 / 5328** | **5592 / 7944** | **5104 / 5184** | 1.0238 [1.0227, 1.0278] | ✅ | metric-native |
| **ttt_mlp** | 12 | 5376 | 5736 | 5376 | 1.0000 [1.0000, 1.0000] | ✅ | weakly metric-native |
| **deltanet** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |
| **gdn** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |
| **gdn2** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |

⚠ **A bold pair of values means the ledger is NOT constant across the 9 seeds.** Best-of-grid selects the TTT mini-batch `b` per seed and `b` enters the declared state (the in-flight buffer), so at n = 9 the TTT rows' byte columns are a **per-seed quantity**: `ttt_linear` b = [16, 1, 1, 16, 16, 16, 16, 16, 16], `ttt_mlp` b = [16, 16, 16, 16, 16, 16, 16, 16, 16]. The delta arms are constant. ⛔ A single TTT byte figure must never be quoted as *the* n = 9 value.

## C. `f3_lite_control` — C2W4's 3-lr sub-grid re-selected from the same fits, n = 9

*C2W4's own 3-lr sub-grid re-selected from the same fits (the init-redraw control)* · seeds [0, 1, 2, 3, 4, 5, 6, 7, 8]

| arm | `full` | `launder` | **same-keys null** | `blank` | **full − null** (paired) | **lift = full − blank** (paired) | RESCUED? |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | -0.5348 ± 0.0541 | -0.4166 ± 0.0157 | -0.3954 ± 0.0146 | -0.7293 ± 0.0730 | -0.1393 ± 0.0534 | +0.1945 ± 0.1055 | ⛔ |
| **ttt_mlp** | -0.5904 ± 0.0733 | -0.4244 ± 0.0162 | -0.4011 ± 0.0197 | -0.5054 ± 0.0367 | -0.1893 ± 0.0719 | -0.0851 ± 0.0823 | ⛔ |
| **deltanet** | -0.4280 ± 0.0306 | -0.6283 ± 0.0600 | -0.6635 ± 0.0628 | -0.7073 ± 0.0815 | +0.2355 ± 0.0686 | +0.2793 ± 0.0790 | ✅ |
| **gdn** | -0.4070 ± 0.0119 | -1.0276 ± 0.0978 | -0.9926 ± 0.0952 | -1.2989 ± 0.2301 | +0.5856 ± 0.1000 | +0.8919 ± 0.2258 | ✅ |
| **gdn2** | -0.4076 ± 0.0184 | -1.0937 ± 0.0799 | -1.1533 ± 0.1141 | -1.4569 ± 0.3152 | +0.7457 ± 0.1226 | +1.0492 ± 0.3203 | ✅ |

| arm | **+0 B margin** (own table) | **RAW-table margin** | SE below 0 | `knn2_mean_+0B` | `knn2_idw_+0B` | `table_mean_+0B` | P5-vs-raw gap |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | -0.1514 ± 0.0505 | **-0.3875 ± 0.0508** | **7.63 SE** | -0.4047 ± 0.0100 | -0.4058 ± 0.0105 | -0.3916 ± 0.0136 | +0.2693 ± 0.0305 |
| **ttt_mlp** | -0.2050 ± 0.0708 | **-0.4431 ± 0.0873** | **5.08 SE** | -0.4138 ± 0.0102 | -0.4143 ± 0.0104 | -0.3931 ± 0.0135 | +0.2771 ± 0.0271 |
| **deltanet** | -0.0197 ± 0.0282 | **-0.2807 ± 0.0397** | **7.07 SE** | -0.5544 ± 0.0560 | -0.5549 ± 0.0556 | -0.4152 ± 0.0191 | +0.4810 ± 0.0599 |
| **gdn** | -0.0105 ± 0.0225 | **-0.2597 ± 0.0275** | **9.43 SE** | -0.8201 ± 0.0431 | -0.8234 ± 0.0479 | -0.3966 ± 0.0194 | +0.8803 ± 0.0932 |
| **gdn2** | +0.0488 ± 0.0273 | **-0.2603 ± 0.0292** | **8.90 SE** | -0.8890 ± 0.0786 | -0.8925 ± 0.0789 | -0.4565 ± 0.0259 | +0.9464 ± 0.0895 |

| arm | `d_head` | rival **state B** | rival **param B** | matched **table B** | state/table | table lossless? | metric-native |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | 36 | **5220 / 5328** | **5592 / 7944** | **5104 / 5184** | 1.0250 [1.0227, 1.0278] | ✅ | metric-native |
| **ttt_mlp** | 12 | **4656 / 5376** | 5736 | **4608 / 5376** | 1.0012 [1.0000, 1.0104] | ✅ | weakly metric-native |
| **deltanet** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |
| **gdn** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |
| **gdn2** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |

⚠ **A bold pair of values means the ledger is NOT constant across the 9 seeds.** Best-of-grid selects the TTT mini-batch `b` per seed and `b` enters the declared state (the in-flight buffer), so at n = 9 the TTT rows' byte columns are a **per-seed quantity**: `ttt_linear` b = [1, 16, 16, 1, 16, 16, 1, 16, 1], `ttt_mlp` b = [16, 16, 16, 16, 16, 16, 16, 16, 1]. The delta arms are constant. ⛔ A single TTT byte figure must never be quoted as *the* n = 9 value.

## D. `f3_val` — declared SECONDARY (held-out selection), n = 9

*declared SECONDARY: best-of-grid on a held-out aux stream* · seeds [0, 1, 2, 3, 4, 5, 6, 7, 8]

| arm | `full` | `launder` | **same-keys null** | `blank` | **full − null** (paired) | **lift = full − blank** (paired) | RESCUED? |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | -0.4461 ± 0.0497 | -0.3933 ± 0.0134 | -0.3973 ± 0.0163 | -0.6522 ± 0.0851 | -0.0488 ± 0.0458 | +0.2062 ± 0.1071 | ⛔ |
| **ttt_mlp** | -0.6390 ± 0.0661 | -0.3971 ± 0.0171 | -0.3976 ± 0.0235 | -0.6606 ± 0.1001 | -0.2415 ± 0.0599 | +0.0216 ± 0.1044 | ⛔ |
| **deltanet** | -0.4267 ± 0.0296 | -0.4885 ± 0.0527 | -0.4795 ± 0.0579 | -0.5035 ± 0.0558 | +0.0528 ± 0.0618 | +0.0768 ± 0.0446 | ⛔ |
| **gdn** | -0.3939 ± 0.0091 | -0.7164 ± 0.1211 | -0.6447 ± 0.0976 | -0.8900 ± 0.2391 | +0.2508 ± 0.0971 | +0.4961 ± 0.2363 | ✅ |
| **gdn2** | -0.3919 ± 0.0202 | -0.8387 ± 0.1032 | -0.9053 ± 0.1437 | -1.0604 ± 0.3420 | +0.5134 ± 0.1413 | +0.6685 ± 0.3389 | ⛔ |

| arm | **+0 B margin** (own table) | **RAW-table margin** | SE below 0 | `knn2_mean_+0B` | `knn2_idw_+0B` | `table_mean_+0B` | P5-vs-raw gap |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | -0.0616 ± 0.0487 | **-0.2987 ± 0.0553** | **5.40 SE** | -0.3978 ± 0.0118 | -0.3974 ± 0.0120 | -0.3901 ± 0.0124 | +0.2460 ± 0.0253 |
| **ttt_mlp** | -0.2516 ± 0.0658 | **-0.4917 ± 0.0721** | **6.82 SE** | -0.4034 ± 0.0175 | -0.4029 ± 0.0174 | -0.3959 ± 0.0133 | +0.2498 ± 0.0312 |
| **deltanet** | -0.0421 ± 0.0237 | **-0.2794 ± 0.0427** | **6.55 SE** | -0.4526 ± 0.0459 | -0.4528 ± 0.0458 | -0.4009 ± 0.0135 | +0.3412 ± 0.0519 |
| **gdn** | -0.0132 ± 0.0134 | **-0.2466 ± 0.0261** | **9.45 SE** | -0.6250 ± 0.0855 | -0.6269 ± 0.0863 | -0.3899 ± 0.0142 | +0.5691 ± 0.1330 |
| **gdn2** | +0.0458 ± 0.0265 | **-0.2446 ± 0.0320** | **7.64 SE** | -0.6629 ± 0.0768 | -0.6691 ± 0.0772 | -0.4377 ± 0.0270 | +0.6913 ± 0.1191 |

| arm | `d_head` | rival **state B** | rival **param B** | matched **table B** | state/table | table lossless? | metric-native |
|---|---|---|---|---|---|---|---|
| **ttt_linear** | 29 | **5220 / 5328** | **5592 / 7944** | **5104 / 5184** | 1.0238 [1.0227, 1.0278] | ✅ | metric-native |
| **ttt_mlp** | 12 | **4656 / 5376** | 5736 | **4608 / 5376** | 1.0069 [1.0000, 1.0104] | ✅ | weakly metric-native |
| **deltanet** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |
| **gdn** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |
| **gdn2** | 36 | 5184 | 9956 | 5184 | 1.0000 [1.0000, 1.0000] | ✅ | metric-native |

⚠ **A bold pair of values means the ledger is NOT constant across the 9 seeds.** Best-of-grid selects the TTT mini-batch `b` per seed and `b` enters the declared state (the in-flight buffer), so at n = 9 the TTT rows' byte columns are a **per-seed quantity**: `ttt_linear` b = [16, 1, 16, 16, 16, 16, 16, 16, 1], `ttt_mlp` b = [1, 16, 1, 1, 1, 16, 1, 1, 16]. The delta arms are constant. ⛔ A single TTT byte figure must never be quoted as *the* n = 9 value.

---

## What the uniform-n table changes, and what it does not

- ⭐ **The headline is unchanged and is now uniform-n on every column:** *at byte-matched state on
  `aggregate`, **0 of 5** rivals beat a zero-extra-byte reader of a RAW table holding the same
  bytes.* Raw margins at n = 9 are **ttt_linear 4.43 · ttt_mlp 5.09 · deltanet 6.92 · gdn 9.35 · gdn2 8.87 SE below zero**
  ⇒ **minimum 4.43 SE**. ⛔ The retired *"≥ 3.6 SE"* wording stays retired (A18.1).
- **Rescue verdicts are the F3 report's, unchanged by the added columns:** `{deltanet, gdn, gdn2}`
  rescued under `f3`; C2W4's own code path additionally rescues `ttt_linear`; **`ttt_mlp` is NOT
  rescued in any column.** ⛔ No n = 3 rescue verdict is quotable (A18.1).
- ⭐⭐ **New, quotable, and it says something the draft did not have: the same-keys null at n = 9,
  PAIRED — and the two rival families fall on opposite sides of it.** `full − null` (positive = the
  arm beats a null handed its own keys):
  **ttt_linear −0.2063 ± 0.1016 · ttt_mlp −0.1995 ± 0.0665 · deltanet +0.2174 ± 0.0749 ·
  gdn +0.5642 ± 0.1032 · gdn2 +0.7438 ± 0.1242** (`f3`, n = 9). ⛔ **Both TTT arms are BELOW their
  own same-keys null** (ttt_mlp by 3.0 SE, ttt_linear by 2.0 SE) — the fitted test-time update is
  worse than doing nothing with the same keys — while all three delta arms are above it by
  2.9–6.0 SE. The same pattern holds on C2W4's own code path (−0.1893 / −0.1367 / +0.1170 / +0.6375
  / +0.6832). This is a *rival-side* statement and changes no CLU claim.
- ⚠ **Unchanged limitation, unsoftened:** one synthetic family, `d_in = 5`, K = 5–6 items, ~10-token
  streams. **Nothing here transfers to a language-model claim.**

