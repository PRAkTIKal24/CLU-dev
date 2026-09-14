## seeds present: [0, 1, 2] | NOT RUN: []

### ⭐ THE GATE — in-corner `shape`, balanced accuracy (R23.1)
| quantity | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| **balacc held-out (GATE)** | 0.9518 | 0.9751 | 0.9579 | 0.9616 ± 0.0099 |
| balacc seen normaliser | 0.9474 | 0.9678 | 0.9508 | 0.9553 ± 0.0089 |
| **ratio held/seen** | 1.0046 | 1.0076 | 1.0075 | 1.0066 ± 0.0014 |
| AUROC held-out (companion) | 0.9997 | 0.9980 | 0.9995 | 0.9991 ± 0.0008 |
| AUROC seen (companion) | 0.9966 | 0.9956 | 0.9957 | 0.9960 ± 0.0005 |
| TPR held-out (⛔ detection) | 1.0000 | 0.9822 | 0.9995 | 0.9939 ± 0.0083 |
| TNR held-out | 0.9036 | 0.9680 | 0.9163 | 0.9293 ± 0.0279 |
| balacc on REAL images (instrument) | 0.9870 | 0.9854 | 0.9873 | 0.9866 ± 0.0008 |
| AUROC on REAL images | 1.0000 | 0.9993 | 0.9997 | 0.9997 ± 0.0003 |
| **VERDICT** | PASS | PASS | PASS | |

### populations / provenance
- seed 0: pops {'negatives': 4096, 'positives': 6292, 'seen': 4096} pools {'corner_total': 18876, 'negatives_pool': 12444, 'positives_pool': 6292, 'seen_pool': 22482} | dec ccc6635b460cb56d (1515073 params, recipe v1_capacity_schedule, 1800 steps) | enc e15cfe0266f8f330 | probe 7499bd52c5f6e956 | cache used/15508 sha 55abddfe3becc1fa | side==cache True | no_leak 6292/6292 leaked 0 | mse {'negatives': 0.0015540857566520572, 'positives': 0.0005179322906769812, 'seen': 0.0004521778319031}
- seed 1: pops {'negatives': 4096, 'positives': 6292, 'seen': 4096} pools {'corner_total': 18876, 'negatives_pool': 12442, 'positives_pool': 6292, 'seen_pool': 22494} | dec 194d37be319b1512 (1515073 params, recipe v1_capacity_schedule, 1800 steps) | enc 3eefabfd20e33cc2 | probe 858af570db673b5a | cache used/15508 sha 7e80b8cabd1c2c3b | side==cache True | no_leak 6292/6292 leaked 0 | mse {'negatives': 0.001560249482281506, 'positives': 0.0005115388776175678, 'seen': 0.0004366367938928306}
- seed 2: pops {'negatives': 4096, 'positives': 6292, 'seen': 4096} pools {'corner_total': 18876, 'negatives_pool': 12452, 'positives_pool': 6292, 'seen_pool': 22500} | dec 70fb2f430f8bcb18 (1515073 params, recipe v1_capacity_schedule, 1800 steps) | enc 29d7e2a983b35ed6 | probe 552aacf2536a5917 | cache used/15508 sha b65f4290bb4a6d7e | side==cache True | no_leak 6292/6292 leaked 0 | mse {'negatives': 0.0015519836451858282, 'positives': 0.0005116748507134616, 'seen': 0.0004172389453742653}

### bridge aggregate ceiling (context — ⛔ NOT the gate)
| quantity | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| ceiling_A aggregate | 0.5154 | 0.4922 | 0.4932 | 0.5003 ± 0.0107 |
| ceiling_B aggregate | 0.5307 | 0.5182 | 0.5187 | 0.5225 ± 0.0058 |
| ratio B/A aggregate | 1.0296 | 1.0528 | 1.0518 | 1.0447 ± 0.0107 |
| ceiling_A `orientation` | 0.1650 | 0.1475 | 0.1426 | 0.1517 ± 0.0096 |
| ceiling_B `orientation` | 0.0615 | 0.0598 | 0.0609 | 0.0607 ± 0.0007 |
| ceiling_A `posX` | 0.4775 | 0.4746 | 0.4648 | 0.4723 ± 0.0054 |
| ceiling_B `posX` | 0.4100 | 0.4286 | 0.4469 | 0.4285 ± 0.0151 |
| ceiling_A `posY` | 0.4883 | 0.4727 | 0.4766 | 0.4792 ± 0.0066 |
| ceiling_B `posY` | 0.6086 | 0.5768 | 0.5815 | 0.5889 ± 0.0140 |
| ceiling_A `scale` | 0.6006 | 0.5684 | 0.5645 | 0.5778 ± 0.0162 |
| ceiling_B `scale` | 0.5734 | 0.5435 | 0.5046 | 0.5405 ± 0.0282 |
| ceiling_A `shape` | 0.8457 | 0.7979 | 0.8174 | 0.8203 ± 0.0196 |
| ceiling_B `shape` | 1.0000 | 0.9822 | 0.9995 | 0.9939 ± 0.0083 |
| probe_on_REAL A aggregate | 0.5557 | 0.5387 | 0.5457 | 0.5467 ± 0.0070 |
| probe_on_REAL A `shape` | 0.9434 | 0.9277 | 0.9551 | 0.9421 ± 0.0112 |
| probe_on_REAL A `orientation` | 0.1582 | 0.1396 | 0.1445 | 0.1475 ± 0.0079 |
| decode MSE A | 0.00097 | 0.00098 | 0.00097 | 0.00097 ± 0.00001 |
| probe_on_REAL B aggregate | 0.5322 | 0.5159 | 0.5220 | 0.5234 ± 0.0067 |
| probe_on_REAL B `shape` | 0.9998 | 0.9752 | 0.9986 | 0.9912 ± 0.0113 |
| probe_on_REAL B `orientation` | 0.0621 | 0.0594 | 0.0658 | 0.0625 ± 0.0026 |
| decode MSE B | 0.00052 | 0.00051 | 0.00051 | 0.00051 ± 0.00000 |

### 3-way `shape` companion (⛔ a DIFFERENT quantity)
- seed 0: negatives: macro-recall 0.6725 (n=4096, classes [0, 2]) | positives: macro-recall 1.0000 (n=6292, classes [1]) | seen: macro-recall 0.9912 (n=4096, classes [1])
- seed 1: negatives: macro-recall 0.6648 (n=4096, classes [0, 2]) | positives: macro-recall 0.9822 (n=6292, classes [1]) | seen: macro-recall 0.9675 (n=4096, classes [1])
- seed 2: negatives: macro-recall 0.7076 (n=4096, classes [0, 2]) | positives: macro-recall 0.9995 (n=6292, classes [1]) | seen: macro-recall 0.9854 (n=4096, classes [1])
