| factor | card | BROKEN val nvr | BROKEN val count ratio | FIXED val nvr | FIXED val count ratio | FIXED val counts min..max |
|---|---|---|---|---|---|---|
| **seed 0** | | | | | | |
| floor_hue | 10 | **1** | 1.000 | **10** | 1.060 | 50..53 |
| wall_hue | 10 | **2** | 1.498 | **10** | 1.040 | 50..52 |
| object_hue | 10 | **10** | 3.043 | **10** | 1.060 | 50..53 |
| scale | 8 | **8** | 1.385 | **8** | 1.032 | 63..65 |
| shape | 4 | **4** | 1.621 | **4** | 1.024 | 126..129 |
| orientation | 15 | **15** | 1.750 | **15** | 1.091 | 33..36 |
| **seed 1** | | | | | | |
| floor_hue | 10 | **1** | 1.000 | **10** | 1.040 | 50..52 |
| wall_hue | 10 | **2** | 1.547 | **10** | 1.060 | 50..53 |
| object_hue | 10 | **10** | 3.524 | **10** | 1.060 | 50..53 |
| scale | 8 | **8** | 1.463 | **8** | 1.032 | 63..65 |
| shape | 4 | **4** | 1.321 | **4** | 1.016 | 127..129 |
| orientation | 15 | **15** | 1.880 | **15** | 1.061 | 33..35 |
| **seed 2** | | | | | | |
| floor_hue | 10 | **1** | 1.000 | **10** | 1.060 | 50..53 |
| wall_hue | 10 | **2** | 1.573 | **10** | 1.060 | 50..53 |
| object_hue | 10 | **10** | 2.630 | **10** | 1.040 | 50..52 |
| scale | 8 | **8** | 1.333 | **8** | 1.082 | 61..66 |
| shape | 4 | **4** | 1.151 | **4** | 1.024 | 127..130 |
| orientation | 15 | **15** | 1.692 | **15** | 1.091 | 33..36 |

### LACE (leg H, 688,128 B) — primary sampler `ode`, and its two Langevin companions
| row | broken split | repaired split | Δmean | per-seed |
|---|---|---|---|---|
| lace/ode in-corner balacc | 0.6797 ± 0.0207 | 0.6873 ± 0.0259 | +0.0076 | 0.7077→0.7191 · 0.6732→0.6557 · 0.6582→0.6872 |
| lace/ode composition B agg | 0.3102 ± 0.0444 | 0.3241 ± 0.0165 | +0.0139 | 0.3727→0.3472 · 0.2731→0.3102 · 0.2847→0.3148 |
| lace/ode composition A agg | 0.3330 ± 0.0111 | 0.3359 ± 0.0179 | +0.0030 | 0.3174→0.3107 · 0.3428→0.3494 · 0.3387→0.3477 |
| lace/ode refusal AUROC(C vs B) | 0.6981 ± 0.0444 | 0.7220 ± 0.0627 | +0.0239 | 0.7568→0.8081 · 0.6883→0.6975 · 0.6493→0.6603 |
| lace/ode A7 dup_factor AUROC | 0.9936 ± 0.0057 | 0.9906 ± 0.0033 | -0.0030 | 1.0000→0.9907 · 0.9861→0.9865 · 0.9946→0.9946 |
| lace/ode A7 out_of_range AUROC | 0.3971 ± 0.0885 | 0.4443 ± 0.1255 | +0.0472 | 0.5135→0.6165 · 0.3789→0.3954 · 0.2990→0.3210 |
| lace/ld_published in-corner balacc | 0.5928 ± 0.1059 | 0.5815 ± 0.0456 | -0.0113 | 0.7418→0.6452 · 0.5052→0.5412 · 0.5314→0.5580 |
| lace/ld_published composition B agg | 0.2616 ± 0.0805 | 0.2870 ± 0.0541 | +0.0255 | 0.3681→0.3565 · 0.1736→0.2245 · 0.2431→0.2801 |
| lace/ld_published composition A agg | 0.2979 ± 0.0056 | 0.2892 ± 0.0121 | -0.0087 | 0.2962→0.2721 · 0.3053→0.2995 · 0.2920→0.2959 |
| lace/ld_published refusal AUROC(C vs B) | 0.7154 ± 0.0510 | 0.8097 ± 0.0210 | +0.0943 | 0.6441→0.7826 · 0.7606→0.8337 · 0.7415→0.8127 |
| lace/ld_published A7 dup_factor AUROC | 0.8797 ± 0.0851 | 0.7846 ± 0.0301 | -0.0951 | 1.0000→0.8248 · 0.8189→0.7523 · 0.8202→0.7766 |
| lace/ld_published A7 out_of_range AUROC | 0.4383 ± 0.1063 | 0.6267 ± 0.0441 | +0.1884 | 0.2882→0.5656 · 0.5216→0.6678 · 0.5050→0.6466 |
| lace/ld_corrected in-corner balacc | 0.6401 ± 0.0193 | 0.6244 ± 0.0113 | -0.0157 | 0.6446→0.6370 · 0.6612→0.6096 · 0.6146→0.6266 |
| lace/ld_corrected composition B agg | 0.2955 ± 0.0351 | 0.2994 ± 0.0240 | +0.0039 | 0.3449→0.3333 · 0.2662→0.2824 · 0.2755→0.2824 |
| lace/ld_corrected composition A agg | 0.3041 ± 0.0173 | 0.3084 ± 0.0151 | +0.0043 | 0.2812→0.2892 · 0.3229→0.3262 · 0.3083→0.3099 |
| lace/ld_corrected refusal AUROC(C vs B) | 0.7195 ± 0.0460 | 0.7465 ± 0.0562 | +0.0270 | 0.7766→0.8131 · 0.7180→0.7508 · 0.6640→0.6757 |
| lace/ld_corrected A7 dup_factor AUROC | 0.8192 ± 0.0674 | 0.7859 ± 0.0322 | -0.0333 | 0.9120→0.7990 · 0.7542→0.7415 · 0.7913→0.8171 |
| lace/ld_corrected A7 out_of_range AUROC | 0.4694 ± 0.1117 | 0.5189 ± 0.1056 | +0.0495 | 0.6130→0.6416 · 0.4545→0.5312 · 0.3407→0.3839 |

### kNN family (leg H, 688,128 B)
| row | broken split | repaired split | Δmean | per-seed |
|---|---|---|---|---|
| knn in-corner balacc | 0.5206 ± 0.0487 | 0.5182 ± 0.0170 | -0.0024 | 0.4621→0.5152 · 0.5182→0.4991 · 0.5813→0.5403 |
| knn composition B agg | 0.4823 ± 0.0480 | 0.4776 ± 0.0289 | -0.0046 | 0.4167→0.5046 · 0.5301→0.4375 · 0.5000→0.4907 |
| knn composition A agg | 0.4976 ± 0.0104 | 0.5069 ± 0.0006 | +0.0093 | 0.4860→0.5068 · 0.4956→0.5062 · 0.5112→0.5076 |
| knn refusal AUROC(C vs B) | 0.5976 ± 0.1159 | 0.8457 ± 0.0238 | +0.2481 | 0.4383→0.8787 · 0.7105→0.8355 · 0.6441→0.8231 |
| knn A7 dup_factor AUROC | 0.3338 ± 0.0692 | 0.3103 ± 0.0013 | -0.0235 | 0.4317→0.3098 · 0.2836→0.3090 · 0.2863→0.3121 |
| knn A7 out_of_range AUROC | 0.8045 ± 0.1002 | 0.9712 ± 0.0070 | +0.1667 | 0.6628→0.9807 · 0.8754→0.9688 · 0.8754→0.9641 |
| knn_proto in-corner balacc | 0.8884 ± 0.0088 | 0.8817 ± 0.0023 | -0.0067 | 0.8763→0.8788 · 0.8920→0.8844 · 0.8970→0.8819 |
| knn_proto composition B agg | 0.8410 ± 0.0330 | 0.8110 ± 0.0257 | -0.0301 | 0.7963→0.7755 · 0.8519→0.8356 · 0.8750→0.8218 |
| knn_proto composition A agg | 0.8111 ± 0.0200 | 0.8116 ± 0.0211 | +0.0005 | 0.7842→0.7832 · 0.8169→0.8179 · 0.8322→0.8338 |
| knn_proto refusal AUROC(C vs B) | 0.5976 ± 0.1159 | 0.8457 ± 0.0238 | +0.2481 | 0.4383→0.8787 · 0.7105→0.8355 · 0.6441→0.8231 |
| knn_proto A7 dup_factor AUROC | 0.3338 ± 0.0692 | 0.3103 ± 0.0013 | -0.0235 | 0.4317→0.3098 · 0.2836→0.3090 · 0.2863→0.3121 |
| knn_proto A7 out_of_range AUROC | 0.8045 ± 0.1002 | 0.9712 ± 0.0070 | +0.1667 | 0.6628→0.9807 · 0.8754→0.9688 · 0.8754→0.9641 |

### kNN family (leg D, deployed 524,288 B — the SHIPPED budget)
| row | broken split | repaired split | Δmean | per-seed |
|---|---|---|---|---|
| knn in-corner balacc | 0.4993 ± 0.0220 | 0.5176 ± 0.0230 | +0.0183 | 0.4893→0.5284 · 0.5298→0.4856 · 0.4788→0.5388 |
| knn composition B agg | 0.4028 ± 0.0774 | 0.4599 ± 0.0406 | +0.0571 | 0.4676→0.5093 · 0.2940→0.4097 · 0.4468→0.4606 |
| knn composition A agg | 0.4731 ± 0.0193 | 0.4954 ± 0.0044 | +0.0222 | 0.4821→0.4956 · 0.4463→0.4899 · 0.4910→0.5007 |
| knn refusal AUROC(C vs B) | 0.6306 ± 0.1471 | 0.8357 ± 0.0218 | +0.2051 | 0.4228→0.8648 · 0.7257→0.8301 · 0.7432→0.8123 |
| knn A7 dup_factor AUROC | 0.3453 ± 0.0737 | 0.3090 ± 0.0145 | -0.0363 | 0.4495→0.2905 · 0.2890→0.3106 · 0.2975→0.3260 |
| knn A7 out_of_range AUROC | 0.8281 ± 0.1299 | 0.9583 ± 0.0048 | +0.1303 | 0.6451→0.9568 · 0.9340→0.9649 · 0.9051→0.9533 |
| knn_proto in-corner balacc | 0.8884 ± 0.0088 | 0.8817 ± 0.0023 | -0.0067 | 0.8763→0.8788 · 0.8920→0.8844 · 0.8970→0.8819 |
| knn_proto composition B agg | 0.8410 ± 0.0330 | 0.8110 ± 0.0257 | -0.0301 | 0.7963→0.7755 · 0.8519→0.8356 · 0.8750→0.8218 |
| knn_proto composition A agg | 0.8111 ± 0.0200 | 0.8116 ± 0.0211 | +0.0005 | 0.7842→0.7832 · 0.8169→0.8179 · 0.8322→0.8338 |
| knn_proto refusal AUROC(C vs B) | 0.6306 ± 0.1471 | 0.8357 ± 0.0218 | +0.2051 | 0.4228→0.8648 · 0.7257→0.8301 · 0.7432→0.8123 |
| knn_proto A7 dup_factor AUROC | 0.3453 ± 0.0737 | 0.3090 ± 0.0145 | -0.0363 | 0.4495→0.2905 · 0.2890→0.3106 · 0.2975→0.3260 |
| knn_proto A7 out_of_range AUROC | 0.8281 ± 0.1299 | 0.9583 ± 0.0048 | +0.1303 | 0.6451→0.9568 · 0.9340→0.9649 · 0.9051→0.9533 |

### structural invariants (P7: must be UNCHANGED)
  no_leak n_holdout_leaked     old=[0, 0, 0] new=[0, 0, 0]  IDENTICAL
  rows.fit                     old=[32256, 32256, 32256] new=[32256, 32256, 32256]  IDENTICAL
  rows.val                     old=[512, 512, 512] new=[512, 512, 512]  IDENTICAL
  rows.A                       old=[1024, 1024, 1024] new=[1024, 1024, 1024]  IDENTICAL
  rows.B                       old=[72, 72, 72] new=[72, 72, 72]  IDENTICAL
  rows.MS                      old=[198, 199, 199] new=[198, 199, 199]  IDENTICAL
  prototype table bytes        old=[475136, 475136, 475136] new=[475136, 475136, 475136]  IDENTICAL
  byte target total            old=[688128, 688128, 688128] new=[688128, 688128, 688128]  IDENTICAL
### ATOM-MEAN CONTROL (ours) — MOVED / UNCHANGED
| row | broken split | repaired split | Δmean | per-seed |
|---|---|---|---|---|
| [A] relative read error on A, k=1 | 0.4618 ± 0.0112 | 0.4578 ± 0.0066 | -0.0040 | 0.4761→0.4640 · 0.4606→0.4609 · 0.4488→0.4486 |
| [A] relative read error on A, best k | 0.4114 ± 0.0355 | 0.4025 ± 0.0326 | -0.0089 | 0.3952→0.3797 · 0.4606→0.3790 · 0.3784→0.4486 |
| [A] relative read error on val, best k | 0.9135 ± 0.0937 | 0.3734 ± 0.0449 | -0.5401 | 0.8987→0.3449 · 1.0349→0.3384 · 0.8068→0.4367 |
| [A] centroid control k=1 (A) | 0.2852 ± 0.0048 | 0.2835 ± 0.0026 | -0.0017 | 0.2915→0.2863 · 0.2800→0.2800 · 0.2841→0.2841 |
| [A] medoid-over-centroid penalty | 1.4260 ± 0.0160 | 1.4065 ± 0.0212 | -0.0195 | 1.4406→1.3820 · 1.4336→1.4336 · 1.4037→1.4037 |
| [A] in-corner shape balacc (external) | 0.6849 ± 0.0202 | 0.6364 ± 0.0645 | -0.0484 | 0.6578→0.6944 · 0.6904→0.6683 · 0.7064→0.5465 |
| [A] retention ladder aggregate on A | 0.5841 ± 0.0184 | 0.5974 ± 0.0138 | +0.0133 | 0.5671→0.5911 · 0.5755→0.5846 · 0.6097→0.6165 |
| [A] distinct factor tuples on B (of 72) | 14.3333 ± 2.0548 | 15.0000 ± 2.1602 | +0.6667 | 17.0000→12.0000 · 12.0000→16.0000 · 14.0000→17.0000 |
| [A] channel bits/read | 90.0000 ± 48.9898 | 70.0000 ± 28.2843 | -20.0000 | 150.0000→90.0000 · 30.0000→90.0000 · 90.0000→30.0000 |
| [A] budget bytes | 524288.0000 ± 0.0000 | 524288.0000 ± 0.0000 | **bit-identical** | 524288.0000→524288.0000 · 524288.0000→524288.0000 · 524288.0000→524288.0000 |
| [B] relative read error on A, k=1 | 0.3299 ± 0.0053 | 0.3298 ± 0.0054 | -0.0002 | 0.3375→0.3375 · 0.3266→0.3259 · 0.3257→0.3259 |
| [B] relative read error on A, best k | 0.2819 ± 0.0394 | 0.3054 ± 0.0290 | +0.0235 | 0.3375→0.2644 · 0.2565→0.3259 · 0.2517→0.3259 |
| [B] relative read error on val, best k | 0.5864 ± 0.0740 | 0.2713 ± 0.0296 | -0.3150 | 0.6908→0.2303 · 0.5396→0.2851 · 0.5287→0.2986 |
| [B] centroid control k=1 (A) | 0.1695 ± 0.0021 | 0.1695 ± 0.0021 | **bit-identical** | 0.1710→0.1710 · 0.1665→0.1665 · 0.1711→0.1711 |
| [B] medoid-over-centroid penalty | 1.3604 ± 0.0108 | 1.3604 ± 0.0108 | **bit-identical** | 1.3492→1.3492 · 1.3571→1.3571 · 1.3750→1.3750 |
| [B] in-corner shape balacc (external) | 0.6685 ± 0.0927 | 0.6997 ± 0.0400 | +0.0312 | 0.5511→0.6900 · 0.6764→0.7528 · 0.7779→0.6563 |
| [B] retention ladder aggregate on A | 0.6565 ± 0.0181 | 0.6491 ± 0.0191 | -0.0074 | 0.6452→0.6370 · 0.6423→0.6341 · 0.6820→0.6761 |
| [B] distinct factor tuples on B (of 72) | 24.3333 ± 5.3125 | 22.0000 ± 2.8284 | -2.3333 | 18.0000→20.0000 · 24.0000→20.0000 · 31.0000→26.0000 |
| [B] channel bits/read | 105.0000 ± 42.4264 | 75.0000 ± 42.4264 | -30.0000 | 45.0000→135.0000 · 135.0000→45.0000 · 135.0000→45.0000 |
| [B] budget bytes | 4194304.0000 ± 0.0000 | 4194304.0000 ± 0.0000 | **bit-identical** | 4194304.0000→4194304.0000 · 4194304.0000→4194304.0000 · 4194304.0000→4194304.0000 |
| fitted exponent (request-addressed, best k) | -0.1845 ± 0.0844 | -0.1334 ± 0.0438 | +0.0511 | -0.0759→-0.1740 · -0.2816→-0.0726 · -0.1960→-0.1537 |
| implied PR (request-addressed, best k) | 14.5512 ± 8.4363 | 17.3554 ± 7.2412 | +2.8042 | 26.3469→11.4927 · 7.1029→27.5583 · 10.2039→13.0153 |
| fitted exponent (centroid control) | -0.2501 ± 0.0050 | -0.2472 ± 0.0024 | +0.0029 | -0.2564→-0.2478 · -0.2499→-0.2499 · -0.2440→-0.2440 |
| knn_proto regression: probe in-corner | 0.8884 ± 0.0088 | 0.8817 ± 0.0023 | -0.0067 | 0.8763→0.8788 · 0.8920→0.8844 · 0.8970→0.8819 |

### ATOM-MEAN — the rows `gen-atom-mean-control.md` actually published (the BAR)
| row | broken split | repaired split | Δmean | per-seed |
|---|---|---|---|---|
| [A] read error, oracle-feature-addressed k=1 (A rows) | 0.4067 ± 0.0094 | 0.3986 ± 0.0023 | -0.0081 | 0.4199→0.3957 · 0.4014→0.4014 · 0.3989→0.3989 |
| [A] read error, oracle-feature-addressed best k (A rows) ⭐THE BAR | 0.3567 ± 0.0319 | 0.3451 ± 0.0382 | -0.0116 | 0.3401→0.3138 · 0.4014→0.3225 · 0.3286→0.3989 |
| [A] read error, oracle-feature-addressed best k (val rows) | 0.8189 ± 0.0986 | 0.3204 ± 0.0468 | -0.4985 | 0.8045→0.2858 · 0.9463→0.2889 · 0.7061→0.3866 |
| [A] distinct ATOM CONFIGURATIONS on B (of 72) | 24.3333 ± 8.8066 | 22.0000 ± 5.7155 | -2.3333 | 32.0000→19.0000 · 12.0000→30.0000 · 29.0000→17.0000 |
| [B] read error, oracle-feature-addressed k=1 (A rows) | 0.2306 ± 0.0038 | 0.2306 ± 0.0038 | **bit-identical** | 0.2308→0.2308 · 0.2260→0.2260 · 0.2352→0.2352 |
| [B] read error, oracle-feature-addressed best k (A rows) ⭐THE BAR | 0.1851 ± 0.0324 | 0.2062 ± 0.0347 | +0.0211 | 0.2308→0.1574 · 0.1595→0.2260 · 0.1651→0.2352 |
| [B] read error, oracle-feature-addressed best k (val rows) | 0.3993 ± 0.0649 | 0.1820 ± 0.0304 | -0.2173 | 0.4908→0.1398 · 0.3472→0.1960 · 0.3599→0.2103 |
| [B] distinct ATOM CONFIGURATIONS on B (of 72) | 37.3333 ± 11.6142 | 28.6667 ± 6.2361 | -8.6667 | 21.0000→37.0000 · 47.0000→22.0000 · 44.0000→27.0000 |
| fitted exponent (oracle-feature, best k) ⭐THE EXPONENT | -0.3204 ± 0.1054 | -0.2523 ± 0.0656 | +0.0682 | -0.1865→-0.3317 · -0.4439→-0.1711 · -0.3310→-0.2540 |
| implied PR (oracle-feature, best k) | 7.0916 ± 2.6457 | 8.5306 ± 2.3560 | +1.4390 | 10.7264→6.0293 · 4.5055→11.6875 · 6.0430→7.8751 |

### the PUBLISHED exponent row (`tables.md` line 15: oracle_feature_addressed/k1)
| | N=64 (budget A) | N=512 (budget B) | fitted exponent | implied PR |
|---|---|---|---|---|
| broken split (published) | 0.4067 ± 0.0094 | 0.2306 ± 0.0038 | **-0.2727 ± 0.0141** | 7.33 |
| &nbsp;&nbsp;per-seed | 0.4199 · 0.4014 · 0.3989 | 0.2308 · 0.2260 · 0.2352 | -0.2878 · -0.2763 · -0.2540 | |
| repaired split | 0.3986 ± 0.0023 | 0.2306 ± 0.0038 | **-0.2632 ± 0.0095** | 7.60 |
| &nbsp;&nbsp;per-seed | 0.3957 · 0.4014 · 0.3989 | 0.2308 · 0.2260 · 0.2352 | -0.2593 · -0.2763 · -0.2540 | |
### CELL HARNESS — MOVED / UNCHANGED (external instrument)
| arm | statistic | broken split | repaired split | Δmean | per-seed |
|---|---|---|---|---|---|
| `clu` | in-corner shape balacc | 0.6568 ± 0.0226 | 0.6568 ± 0.0226 | **bit-identical** | 0.6540→0.6540 · 0.6307→0.6307 · 0.6858→0.6858 |
| `clu` | in-corner AUROC | 0.7653 ± 0.0123 | 0.7653 ± 0.0123 | **bit-identical** | 0.7812→0.7812 · 0.7513→0.7513 · 0.7633→0.7633 |
| `clu` | composition B aggregate | 0.5949 ± 0.0082 | 0.5949 ± 0.0082 | **bit-identical** | 0.5903→0.5903 · 0.6065→0.6065 · 0.5880→0.5880 |
| `clu` | distinct factor tuples on B | 9.0000 ± 0.8165 | 9.0000 ± 0.8165 | **bit-identical** | 9.0000→9.0000 · 8.0000→8.0000 · 10.0000→10.0000 |
| `ebm_compose` | in-corner shape balacc | 0.5408 ± 0.0087 | 0.5477 ± 0.0216 | +0.0069 | 0.5455→0.5183 · 0.5287→0.5551 · 0.5483→0.5697 |
| `ebm_compose` | in-corner AUROC | 0.6077 ± 0.0134 | 0.5917 ± 0.0124 | -0.0160 | 0.5910→0.5793 · 0.6239→0.5872 · 0.6080→0.6086 |
| `ebm_compose` | composition B aggregate | 0.2130 ± 0.0132 | 0.1937 ± 0.0182 | -0.0193 | 0.2245→0.1736 · 0.1944→0.1898 · 0.2199→0.2176 |
| `ebm_compose` | distinct factor tuples on B | 71.6667 ± 0.4714 | 72.0000 ± 0.0000 | +0.3333 | 71.0000→72.0000 · 72.0000→72.0000 · 72.0000→72.0000 |
| `knn_proto` | in-corner shape balacc | 0.9119 ± 0.0222 | 0.9165 ± 0.0250 | +0.0046 | 0.9040→0.9198 · 0.9422→0.9453 · 0.8894→0.8844 |
| `knn_proto` | in-corner AUROC | 0.9875 ± 0.0088 | 0.9847 ± 0.0095 | -0.0028 | 0.9840→0.9783 · 0.9995→0.9981 · 0.9789→0.9777 |
| `knn_proto` | composition B aggregate | 0.9005 ± 0.0182 | 0.8650 ± 0.0122 | -0.0355 | 0.8843→0.8588 · 0.8912→0.8542 · 0.9259→0.8819 |
| `knn_proto` | distinct factor tuples on B | 54.6667 ± 8.0554 | 56.0000 ± 3.7417 | +1.3333 | 50.0000→57.0000 · 48.0000→51.0000 · 66.0000→60.0000 |
| `lace` | in-corner shape balacc | 0.6347 ± 0.0177 | 0.6278 ± 0.0289 | -0.0069 | 0.6143→0.5890 · 0.6324→0.6582 · 0.6575→0.6361 |
| `lace` | in-corner AUROC | 0.7186 ± 0.0147 | 0.7146 ± 0.0130 | -0.0039 | 0.7034→0.7075 · 0.7139→0.7036 · 0.7385→0.7328 |
| `lace` | composition B aggregate | 0.3056 ± 0.0208 | 0.3117 ± 0.0153 | +0.0062 | 0.3056→0.3009 · 0.3310→0.3333 · 0.2801→0.3009 |
| `lace` | distinct factor tuples on B | 71.3333 ± 0.9428 | 71.6667 ± 0.4714 | +0.3333 | 70.0000→72.0000 · 72.0000→71.0000 · 72.0000→72.0000 |
| `lace/ld_published` | in-corner shape balacc | 0.5580 ± 0.0564 | 0.5708 ± 0.0479 | +0.0128 | 0.6288→0.6187 · 0.5544→0.5885 · 0.4908→0.5053 |
| `lace/ld_published` | in-corner AUROC | 0.7599 ± 0.0604 | 0.7540 ± 0.0443 | -0.0059 | 0.7860→0.7588 · 0.8173→0.8057 · 0.6763→0.6975 |
| `lace/ld_published` | composition B aggregate | 0.2276 ± 0.0165 | 0.2554 ± 0.0022 | +0.0278 | 0.2222→0.2569 · 0.2500→0.2569 · 0.2106→0.2523 |
| `lace/ld_published` | distinct factor tuples on B | 56.3333 ± 4.1096 | 55.3333 ± 4.4969 | -1.0000 | 57.0000→55.0000 · 61.0000→61.0000 · 51.0000→50.0000 |
| `lace/ld_corrected` | in-corner shape balacc | 0.6605 ± 0.0197 | 0.6448 ± 0.0115 | -0.0158 | 0.6338→0.6294 · 0.6669→0.6480 · 0.6808→0.6569 |
| `lace/ld_corrected` | in-corner AUROC | 0.7299 ± 0.0203 | 0.7411 ± 0.0156 | +0.0111 | 0.7321→0.7619 · 0.7040→0.7242 · 0.7537→0.7372 |
| `lace/ld_corrected` | composition B aggregate | 0.2855 ± 0.0159 | 0.2816 ± 0.0180 | -0.0039 | 0.2755→0.2824 · 0.3079→0.3032 · 0.2731→0.2593 |
| `lace/ld_corrected` | distinct factor tuples on B | 71.3333 ± 0.4714 | 72.0000 ± 0.0000 | +0.6667 | 71.0000→72.0000 · 72.0000→72.0000 · 71.0000→72.0000 |
| `store_removed` | in-corner shape balacc | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | **bit-identical** | 0.5000→0.5000 · 0.5000→0.5000 · 0.5000→0.5000 |
| `store_removed` | in-corner AUROC | 0.5008 ± 0.0012 | 0.5008 ± 0.0012 | **bit-identical** | 0.5000→0.5000 · 0.5025→0.5025 · 0.5000→0.5000 |
| `store_removed` | composition B aggregate | 0.1204 ± 0.0131 | 0.1204 ± 0.0131 | **bit-identical** | 0.1111→0.1111 · 0.1389→0.1389 · 0.1111→0.1111 |
| `store_removed` | distinct factor tuples on B | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **bit-identical** | 1.0000→1.0000 · 1.0000→1.0000 · 1.0000→1.0000 |

### cell: which selections moved
  seed 0 ebm_compose: MOVED {'step_size': 0.03, 'prior_sigma': 2.0, 'n_steps': 200, 'clip': False} -> {'step_size': 0.03, 'prior_sigma': 2.0, 'n_steps': 200, 'clip': True}
  seed 0 lace (INHERITED): None -> None
  seed 1 ebm_compose: MOVED {'step_size': 0.03, 'prior_sigma': 2.0, 'n_steps': 50, 'clip': False} -> {'step_size': 0.03, 'prior_sigma': 2.0, 'n_steps': 200, 'clip': True}
  seed 1 lace (INHERITED): None -> None
  seed 2 ebm_compose: same  {'step_size': 0.03, 'prior_sigma': 2.0, 'n_steps': 200, 'clip': False} -> {'step_size': 0.03, 'prior_sigma': 2.0, 'n_steps': 200, 'clip': False}
  seed 2 lace (INHERITED): None -> None

### cell: structural invariants
  store deployed bytes     IDENTICAL  old=[524288, 524288, 524288] new=[524288, 524288, 524288]
  clu channel bits         IDENTICAL  old=[30.0, 30.0, 30.0] new=[30.0, 30.0, 30.0]
  decoder param hash       IDENTICAL  old=['01ef91f75eeaae7be8d83a220c9cc6629120345357da9033db22375c278ab196', '001011bc603a7cb914cfaa592f43087685727431902ffc133755de30dbb21e4f', '2d7eea75add770441fcd0068ffc7a3dcc0b02cd16d6a31b3e095f47b64aa2455'] new=['01ef91f75eeaae7be8d83a220c9cc6629120345357da9033db22375c278ab196', '001011bc603a7cb914cfaa592f43087685727431902ffc133755de30dbb21e4f', '2d7eea75add770441fcd0068ffc7a3dcc0b02cd16d6a31b3e095f47b64aa2455']
  K                        IDENTICAL  old=[5, 5, 5] new=[5, 5, 5]
  D                        IDENTICAL  old=[2048, 2048, 2048] new=[2048, 2048, 2048]
  rows.B                   IDENTICAL  old=[72, 72, 72] new=[72, 72, 72]
  rows.MS                  IDENTICAL  old=[198, 199, 199] new=[198, 199, 199]
  rows.fit                 IDENTICAL  old=[32256, 32256, 32256] new=[32256, 32256, 32256]
  rows.val                 IDENTICAL  old=[512, 512, 512] new=[512, 512, 512]
### Is the selection IDENTIFIED after the repair?  (atom-mean, 24-trial grid, 256 rendered val rows)
| seed | budget | halves disagree (BROKEN, contiguous) | halves disagree (REPAIRED, contiguous) | halves disagree (REPAIRED, interleaved ⭐) | best-to-2nd gap old → new | trial spread new |
|---|---|---|---|---|---|---|
| 0 | A | DISAGREE | agree | agree | 0.0221 → 0.0130 | 0.0970 |
| 0 | B | DISAGREE | DISAGREE | DISAGREE | 0.0026 → 0.0072 | 0.0678 |
| 1 | A | DISAGREE | DISAGREE | DISAGREE | 0.0033 → 0.0046 | 0.0942 |
| 1 | B | DISAGREE | DISAGREE | DISAGREE | 0.0007 → 0.0007 | 0.0721 |
| 2 | A | agree | DISAGREE | DISAGREE | 0.0182 → 0.0039 | 0.1111 |
| 2 | B | DISAGREE | agree | DISAGREE | 0.0039 → 0.0039 | 0.0842 |

**5/6 → 4/6 (contiguous) and 5/6 (interleaved, the exchangeable halving).**
### `knn_proto` under FOUR independent stratified splits (seed 0, deployed 524,288 B)
⭐ `knn_proto` has no tuning axis: every move below is the `fit`-row swap, ⛔ never a selection rule.

| split | in-corner balacc | composition B agg | composition A agg | selected kNN trial |
|---|---|---|---|---|
| ⛔ published (BROKEN head/tail split) | 0.8763 | 0.7963 | 0.7842 | k=5 cosine zscore int8 |
| ⭐ the shipped repair (seed_offset=20260914) | 0.8788 | 0.7755 | 0.7832 | k=5 euclidean none int8 |
| alternative stratified split salt_a (seed_offset=20260915) | 0.8763 | 0.7778 | 0.7821 | k=5 euclidean none int8 |
| alternative stratified split salt_b (seed_offset=20260916) | 0.8763 | 0.7755 | 0.7826 | k=5 cosine none int8 |
| alternative stratified split salt_c (seed_offset=20260917) | 0.8788 | 0.7755 | 0.7845 | k=5 euclidean none int8 |

**Spread across the four equally-valid stratified splits** (⛔ all the same rule, only the salt differs): in-corner 0.8775 ± 0.0013 (range 0.0025) · composition B 0.7760 ± 0.0010 (range 0.0023) · composition A 0.7831 ± 0.0009 (range 0.0024)
