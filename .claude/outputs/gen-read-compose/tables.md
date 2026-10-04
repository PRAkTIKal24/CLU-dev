3 seeds: [0, 1, 2]

### gate statistic: in-corner shape balanced accuracy
| arm | mean ± sd | per seed |
|---|---|---|
| decoder ceiling | **0.9414 ± 0.0179** | 0.9223 / 0.9365 / 0.9653 |
| STORE ceiling (measured) | **0.6866 ± 0.0681** | 0.5972 / 0.7001 / 0.7624 |
| read | **0.6759 ± 0.0107** | 0.6610 / 0.6809 / 0.6858 |
| clu_wide | **0.7484 ± 0.0307** | 0.7146 / 0.7889 / 0.7415 |
| store_removed | **0.5000 ± 0.0000** | 0.5000 / 0.5000 / 0.5000 |
| proto | **0.8817 ± 0.0023** | 0.8788 / 0.8844 / 0.8819 |

### AUROC companion
| decoder ceiling | **0.9886 ± 0.0119** | 0.9720 / 0.9939 / 0.9998 |
| STORE ceiling | **0.7414 ± 0.0295** | 0.7004 / 0.7549 / 0.7688 |
| read | **0.7429 ± 0.0080** | 0.7512 / 0.7453 / 0.7321 |
| clu_wide | **0.7931 ± 0.0095** | 0.8059 / 0.7898 / 0.7835 |
| store_removed | **0.5000 ± 0.0000** | 0.5000 / 0.5000 / 0.5000 |
| proto | **0.9908 ± 0.0085** | 0.9928 / 1.0000 / 0.9796 |

### TPR / TNR
| read | TPR 0.8333 ± 0.1227 | TNR 0.5185 ± 0.1234 |
| clu_wide | TPR 0.9583 ± 0.0589 | TNR 0.5384 ± 0.0781 |
| store_removed | TPR 0.0000 ± 0.0000 | TNR 1.0000 ± 0.0000 |
| proto | TPR 1.0000 ± 0.0000 | TNR 0.7634 ± 0.0046 |

### A: aggregate and shape (separate, always)
| arm | aggregate | shape |
|---|---|---|
| decoder ceiling | 0.9910 ± 0.0029 | 0.9733 ± 0.0133 |
| STORE ceiling | 0.6215 ± 0.0021 | 0.6107 ± 0.0252 |
| read | 0.6174 ± 0.0070 | 0.5977 ± 0.0375 |
| clu_wide | 0.6610 ± 0.0103 | 0.6445 ± 0.0345 |
| store_removed | 0.1196 ± 0.0151 | 0.2467 ± 0.0136 |
| proto | 0.8104 ± 0.0163 | 0.6960 ± 0.0367 |

### B: aggregate and shape (separate, always)
| arm | aggregate | shape |
|---|---|---|
| decoder ceiling | 0.9830 ± 0.0061 | 0.9398 ± 0.0131 |
| STORE ceiling | 0.6312 ± 0.0109 | 0.8565 ± 0.0755 |
| read | 0.6181 ± 0.0204 | 0.8333 ± 0.1227 |
| clu_wide | 0.6844 ± 0.0129 | 0.9583 ± 0.0589 |
| store_removed | 0.0833 ± 0.0393 | 0.0000 ± 0.0000 |
| proto | 0.8102 ± 0.0254 | 1.0000 ± 0.0000 |

### falsifier: read − store_removed (in-corner shape balacc)
delta = +0.1759 ± 0.0107  per-seed +0.1610 / +0.1809 / +0.1858
largest arm spread = 0.0107 -> delta/spread = 16.4x; seeds positive = 3/3
delta B shape = +0.8333 ± 0.1227

### fractions of ceilings
| read | 0.7181 ± 0.0069 of decoder | 0.9930 ± 0.0858 of store |
| clu_wide | 0.7952 ± 0.0336 of decoder | 1.0987 ± 0.0936 of store |
| store_removed | 0.5313 ± 0.0100 of decoder | 0.7357 ± 0.0756 of store |
| proto | 0.9369 ± 0.0168 of decoder | 1.2972 ± 0.1307 of store |

### diagnostics
code agreement A: exact-set 0.5104 ± 0.0106, per-well 0.8710 ± 0.0021
code agreement B: exact-set 0.5833 ± 0.3007, per-well 0.8954 ± 0.0806
code agreement MS: exact-set 0.4331 ± 0.0993, per-well 0.8504 ± 0.0367
request posterior confidence on B: 0.8326 ± 0.0278
distinct code configs A: 495.0 ± 2.2 (of 512)
distinct code configs B: 9.0 ± 0.8 (of 72)
distinct rendered vectors on B, read: 9.0 ± 0.8 (of 72)
distinct rendered vectors on B, clu_wide: 72.0 ± 0.0 (of 72)
distinct rendered vectors on B, store_removed: 1.0 ± 0.0 (of 72)
distinct rendered vectors on B, proto: 72.0 ± 0.0 (of 72)

bytes: 524288 excluded coupling: 163840 addressing head: 74240 proto table: 475136
channel: {'clu_wide': 320, 'proto': 2048, 'read': 30.0}
posterior entropy bits: [4.285, 4.269, 5.122]
invariance: [(True, -0.0019), (True, -0.0049), (True, 0.0078)]
wall clock s: [4908, 872, 899]
decoder recipes: {0: ('v1_capacity_schedule', '01ef91f75eea'), 1: ('v1_capacity_schedule', '001011bc603a'), 2: ('v1_capacity_schedule', '2d7eea75add7')}
store training final loss: [31.647, 29.742, 34.355]
rows: {'A': 512, 'B': 72, 'MS': 198, 'store_train': 32768} [{'A': 512, 'B': 72, 'MS': 198, 'store_train': 32768}, {'A': 512, 'B': 72, 'MS': 199, 'store_train': 32768}, {'A': 512, 'B': 72, 'MS': 199, 'store_train': 32768}]
