# gen-baselines-consolidate — extracted tables (all numbers from on-disk artefacts)

⚠ **± is the POPULATION sd (ddof=0), the convention the shipped aggregate and the R26/R27 ledger use**; the sample sd (ddof=1) is given in brackets.

Source: `.claude/outputs/gen-baselines/runs/shapes3d_recon/seed_{0,1,2}/baselines_manifest.json` and `.claude/outputs/gen-baselines/runs/shapes3d_recon/baselines_results.json`


## Composition — in-corner `shape` balanced accuracy (R23 gate statistic)

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.4621 | 0.5182 | 0.5813 | 0.5206 ± 0.0487 [ddof1 0.0596] (n=3) |
| `knn_proto` | 0.8763 | 0.8920 | 0.8970 | 0.8884 ± 0.0088 [ddof1 0.0108] (n=3) |
| `knn_compsel` | 0.4621 | 0.4912 | 0.5805 | 0.5113 ± 0.0504 [ddof1 0.0617] (n=3) |
| `lace_ode` | 0.7077 | 0.6732 | 0.6582 | 0.6797 ± 0.0207 [ddof1 0.0254] (n=3) |
| `lace_ld_published` | 0.7418 | 0.5052 | 0.5314 | 0.5928 ± 0.1059 [ddof1 0.1297] (n=3) |
| `lace_ld_corrected` | 0.6446 | 0.6612 | 0.6146 | 0.6401 ± 0.0193 [ddof1 0.0236] (n=3) |

## Composition — in-corner AUROC (companion, NEVER the gate)

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.5020 | 0.5105 | 0.6197 | 0.5441 ± 0.0536 [ddof1 0.0656] (n=3) |
| `knn_proto` | 0.9945 | 0.9994 | 0.9798 | 0.9912 ± 0.0084 [ddof1 0.0102] (n=3) |
| `knn_compsel` | 0.5020 | 0.5193 | 0.6390 | 0.5534 ± 0.0609 [ddof1 0.0746] (n=3) |
| `lace_ode` | 0.7973 | 0.7721 | 0.7838 | 0.7844 ± 0.0103 [ddof1 0.0126] (n=3) |
| `lace_ld_published` | 0.8376 | 0.7395 | 0.7229 | 0.7667 ± 0.0506 [ddof1 0.0620] (n=3) |
| `lace_ld_corrected` | 0.6872 | 0.7974 | 0.7322 | 0.7389 ± 0.0452 [ddof1 0.0554] (n=3) |

## Composition — B `shape` probe accuracy

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.5556 | 0.7500 | 0.6250 | 0.6435 ± 0.0805 [ddof1 0.0985] (n=3) |
| `knn_proto` | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 [ddof1 0.0000] (n=3) |
| `knn_compsel` | 0.5556 | 0.2639 | 0.5278 | 0.4491 ± 0.1314 [ddof1 0.1610] (n=3) |
| `lace_ode` | 0.5972 | 0.4167 | 0.4722 | 0.4954 ± 0.0755 [ddof1 0.0925] (n=3) |
| `lace_ld_published` | 0.6250 | 0.0556 | 0.2639 | 0.3148 ± 0.2352 [ddof1 0.2881] (n=3) |
| `lace_ld_corrected` | 0.4861 | 0.4028 | 0.3750 | 0.4213 ± 0.0472 [ddof1 0.0578] (n=3) |

## Composition — B aggregate probe accuracy

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.4167 | 0.5301 | 0.5000 | 0.4823 ± 0.0480 [ddof1 0.0588] (n=3) |
| `knn_proto` | 0.7963 | 0.8519 | 0.8750 | 0.8410 ± 0.0330 [ddof1 0.0404] (n=3) |
| `knn_compsel` | 0.4167 | 0.3681 | 0.4907 | 0.4252 ± 0.0504 [ddof1 0.0618] (n=3) |
| `lace_ode` | 0.3727 | 0.2731 | 0.2847 | 0.3102 ± 0.0444 [ddof1 0.0544] (n=3) |
| `lace_ld_published` | 0.3681 | 0.1736 | 0.2431 | 0.2616 ± 0.0805 [ddof1 0.0985] (n=3) |
| `lace_ld_corrected` | 0.3449 | 0.2662 | 0.2755 | 0.2955 ± 0.0351 [ddof1 0.0430] (n=3) |

## Composition — A aggregate probe accuracy

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.4860 | 0.4956 | 0.5112 | 0.4976 ± 0.0104 [ddof1 0.0127] (n=3) |
| `knn_proto` | 0.7842 | 0.8169 | 0.8322 | 0.8111 ± 0.0200 [ddof1 0.0245] (n=3) |
| `knn_compsel` | 0.4860 | 0.4956 | 0.5133 | 0.4983 ± 0.0113 [ddof1 0.0139] (n=3) |
| `lace_ode` | 0.3174 | 0.3428 | 0.3387 | 0.3330 ± 0.0111 [ddof1 0.0136] (n=3) |
| `lace_ld_published` | 0.2962 | 0.3053 | 0.2920 | 0.2979 ± 0.0056 [ddof1 0.0068] (n=3) |
| `lace_ld_corrected` | 0.2812 | 0.3229 | 0.3083 | 0.3041 ± 0.0173 [ddof1 0.0211] (n=3) |

## Refusal — AUROC(B vs C) POOLED  [⛔ CONTAMINATED: C-dup carries 7 terms vs B's 6]

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.4383 | 0.7105 | 0.6441 | 0.5976 ± 0.1159 [ddof1 0.1419] (n=3) |
| `knn_proto` | 0.4383 | 0.7105 | 0.6441 | 0.5976 ± 0.1159 [ddof1 0.1419] (n=3) |
| `knn_compsel` | 0.4383 | 0.5608 | 0.7160 | 0.5717 ± 0.1137 [ddof1 0.1392] (n=3) |
| `lace_ode` | 0.7568 | 0.6883 | 0.6493 | 0.6981 ± 0.0444 [ddof1 0.0544] (n=3) |
| `lace_ld_published` | 0.6441 | 0.7606 | 0.7415 | 0.7154 ± 0.0510 [ddof1 0.0625] (n=3) |
| `lace_ld_corrected` | 0.7766 | 0.7180 | 0.6640 | 0.7195 ± 0.0460 [ddof1 0.0563] (n=3) |

## Refusal — AUROC(B vs C), kind = `duplicate_factor`  [⛔ CONTAMINATED — needs A7]

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.2137 | 0.5455 | 0.4128 | 0.3907 ± 0.1364 [ddof1 0.1670] (n=3) |
| `knn_proto` | 0.2137 | 0.5455 | 0.4128 | 0.3907 ± 0.1364 [ddof1 0.1670] (n=3) |
| `knn_compsel` | 0.2137 | 0.3846 | 0.5150 | 0.3711 ± 0.1234 [ddof1 0.1511] (n=3) |
| `lace_ode` | 1.0000 | 0.9977 | 0.9996 | 0.9991 ± 0.0010 [ddof1 0.0012] (n=3) |
| `lace_ld_published` | 1.0000 | 0.9996 | 0.9780 | 0.9925 ± 0.0103 [ddof1 0.0126] (n=3) |
| `lace_ld_corrected` | 0.9402 | 0.9815 | 0.9873 | 0.9697 ± 0.0210 [ddof1 0.0257] (n=3) |

## Refusal — AUROC(B vs C), kind = `out_of_range`  [✅ A7-CLEAN: control is 6-term valid, detector 0.5000]

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.6628 | 0.8754 | 0.8754 | 0.8045 ± 0.1002 [ddof1 0.1227] (n=3) |
| `knn_proto` | 0.6628 | 0.8754 | 0.8754 | 0.8045 ± 0.1002 [ddof1 0.1227] (n=3) |
| `knn_compsel` | 0.6628 | 0.7369 | 0.9171 | 0.7722 ± 0.1068 [ddof1 0.1308] (n=3) |
| `lace_ode` | 0.5135 | 0.3789 | 0.2990 | 0.3971 ± 0.0885 [ddof1 0.1084] (n=3) |
| `lace_ld_published` | 0.2882 | 0.5216 | 0.5050 | 0.4383 ± 0.1063 [ddof1 0.1302] (n=3) |
| `lace_ld_corrected` | 0.6130 | 0.4545 | 0.3407 | 0.4694 ± 0.1117 [ddof1 0.1368] (n=3) |

## Refusal — AUROC(A vs C) pooled [⛔ contaminated on the dup half]

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.7621 | 0.8572 | 0.8228 | 0.8140 ± 0.0393 [ddof1 0.0481] (n=3) |
| `knn_proto` | 0.7621 | 0.8572 | 0.8228 | 0.8140 ± 0.0393 [ddof1 0.0481] (n=3) |
| `knn_compsel` | 0.7621 | 0.8044 | 0.8331 | 0.7999 ± 0.0292 [ddof1 0.0357] (n=3) |
| `lace_ode` | 0.7482 | 0.7239 | 0.7451 | 0.7391 ± 0.0108 [ddof1 0.0132] (n=3) |
| `lace_ld_published` | 0.8513 | 0.7723 | 0.7755 | 0.7997 ± 0.0365 [ddof1 0.0447] (n=3) |
| `lace_ld_corrected` | 0.7322 | 0.6979 | 0.7252 | 0.7184 ± 0.0148 [ddof1 0.0181] (n=3) |

## Refusal — AUROC(A∪B vs C) pooled [⛔ contaminated on the dup half]

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.7408 | 0.8475 | 0.8111 | 0.7998 ± 0.0443 [ddof1 0.0542] (n=3) |
| `knn_proto` | 0.7408 | 0.8475 | 0.8111 | 0.7998 ± 0.0443 [ddof1 0.0542] (n=3) |
| `knn_compsel` | 0.7408 | 0.7884 | 0.8255 | 0.7849 ± 0.0346 [ddof1 0.0424] (n=3) |
| `lace_ode` | 0.7487 | 0.7216 | 0.7388 | 0.7364 ± 0.0112 [ddof1 0.0137] (n=3) |
| `lace_ld_published` | 0.8377 | 0.7716 | 0.7733 | 0.7942 ± 0.0308 [ddof1 0.0377] (n=3) |
| `lace_ld_corrected` | 0.7351 | 0.6992 | 0.7212 | 0.7185 ± 0.0148 [ddof1 0.0181] (n=3) |

## Refused fraction on C (threshold = A-only 95th pct)

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.5139 | 0.5694 | 0.5972 | 0.5602 ± 0.0346 [ddof1 0.0424] (n=3) |
| `knn_proto` | 0.5139 | 0.5694 | 0.5972 | 0.5602 ± 0.0346 [ddof1 0.0424] (n=3) |
| `knn_compsel` | 0.5139 | 0.4444 | 0.6528 | 0.5370 ± 0.0866 [ddof1 0.1061] (n=3) |
| `lace_ode` | 0.5417 | 0.5139 | 0.5000 | 0.5185 ± 0.0173 [ddof1 0.0212] (n=3) |
| `lace_ld_published` | 0.6111 | 0.5694 | 0.5139 | 0.5648 ± 0.0398 [ddof1 0.0488] (n=3) |
| `lace_ld_corrected` | 0.4306 | 0.4861 | 0.4722 | 0.4630 ± 0.0236 [ddof1 0.0289] (n=3) |

## Refused fraction on B (false refusal of valid-unseen)

| arm | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `knn` | 0.5000 | 0.4167 | 0.4167 | 0.4444 ± 0.0393 [ddof1 0.0481] (n=3) |
| `knn_proto` | 0.5000 | 0.4167 | 0.4167 | 0.4444 ± 0.0393 [ddof1 0.0481] (n=3) |
| `knn_compsel` | 0.5000 | 0.4861 | 0.4028 | 0.4630 ± 0.0429 [ddof1 0.0526] (n=3) |
| `lace_ode` | 0.0139 | 0.0833 | 0.2778 | 0.1250 ± 0.1117 [ddof1 0.1368] (n=3) |
| `lace_ld_published` | 0.3194 | 0.0417 | 0.1806 | 0.1806 ± 0.1134 [ddof1 0.1389] (n=3) |
| `lace_ld_corrected` | 0.0139 | 0.0278 | 0.0556 | 0.0324 ± 0.0173 [ddof1 0.0212] (n=3) |

## ORACLE-selected upper bounds (max over all 36 trials, using C — ⛔ not achievable)

| family | metric | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|---|
| `knn` | c_vs_b | 0.7031 | 0.7892 | 0.7998 | 0.7640 ± 0.0433 [ddof1 0.0530] (n=3) |
| `knn` | c_vs_aub | 0.8232 | 0.8498 | 0.8387 | 0.8373 ± 0.0109 [ddof1 0.0134] (n=3) |
| `lace` | c_vs_b | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 [ddof1 0.0000] (n=3) |
| `lace` | c_vs_aub | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 [ddof1 0.0000] (n=3) |

## Ceilings (re-measured in-run, real features through THE frozen decoder)

| ceiling | value |
|---|---|
| decoder ceiling, in-corner `shape` balacc | 0.9414 ± 0.0179 (n=3) |
| decoder ceiling, A aggregate | 0.9902 ± 0.0030 (n=3) |
| STORE ceiling, in-corner balacc (R22/R23, carried constant) | 0.6866 |
