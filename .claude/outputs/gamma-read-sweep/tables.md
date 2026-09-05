files: ['exploratory_gated_write.json', 'item1_gamma.json', 'item2_read.json', 'item3_steps.json']

## Item 1 — gamma sweep (CLU only; baselines untouched)

| family (metric) | γ=0 | γ=0.001 | γ=0.005 | γ=0.01 | γ=0.02 | γ=0.05 | γ=0.1 | shipped γ=0.05 | spread | best γ |
|---|---|---|---|---|---|---|---|---|---|---|
| **adding_T128** (mse) | 0.1825 ± 0.0032 | 0.1822 ± 0.0032 | 0.1821 ± 0.0031 | 0.1817 ± 0.0045 | 0.1816 ± 0.0043 | 0.1831 ± 0.0060 | 0.1839 ± 0.0075 | 0.1825 | **0.0023** | **0.02** |
| ↳ seed-σ within cell | 0.0032 | 0.0032 | 0.0031 | 0.0045 | 0.0043 | 0.0060 | 0.0075 | | (mean 0.0045) | |
| ↳ best lr | 0.003 | 0.001 | 0.001 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | | | |
| ↳ diverged seeds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | | | |
| **parity_T64** (accuracy) | 0.5334 ± 0.0053 | 0.5323 ± 0.0021 | 0.5228 ± 0.0042 | 0.5349 ± 0.0032 | 0.5207 ± 0.0047 | 0.5233 ± 0.0051 | 0.5368 ± 0.0054 | 0.5380 | **0.0161** | **0.1** |
| ↳ seed-σ within cell | 0.0053 | 0.0021 | 0.0042 | 0.0032 | 0.0047 | 0.0051 | 0.0054 | | (mean 0.0043) | |
| ↳ best lr | 0.001 | 0.003 | 0.0003 | 0.001 | 0.0003 | 0.0003 | 0.003 | | | |
| ↳ diverged seeds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | | | |
| **mqar_T128_kv4** (accuracy) | 0.3864 ± 0.0032 | 0.3857 ± 0.0086 | 0.3773 ± 0.0081 | 0.3770 ± 0.0060 | 0.3669 ± 0.0127 | 0.3464 ± 0.0080 | 0.3337 ± 0.0024 | 0.3464 | **0.0527** | **0** |
| ↳ seed-σ within cell | 0.0032 | 0.0086 | 0.0081 | 0.0060 | 0.0127 | 0.0080 | 0.0024 | | (mean 0.0070) | |
| ↳ best lr | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | | | |
| ↳ diverged seeds | 0 | 0 | 0 | 0 | 0 | 0 | 0 | | | |

adding_T128: min=0.1816 max=0.1839 spread=0.0023 mean-within-cell-σ=0.0045 spread/σ=0.50 control=0.1825 best-baseline=('attention', 0.0001)
parity_T64: min=0.5207 max=0.5368 spread=0.0161 mean-within-cell-σ=0.0043 spread/σ=3.76 control=0.5109 best-baseline=('gru', 1.0)
mqar_T128_kv4: min=0.3337 max=0.3864 spread=0.0527 mean-within-cell-σ=0.0070 spread/σ=7.52 control=0.0124 best-baseline=('attention', 0.9945)

## Item 2 — (gamma x read-mode), per clu_steps


### adding_T128, clu_steps=1  ⚠ read modes are the SAME MAP at clu_steps=1 (fiber of one)
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.1825 ± 0.0032 | 0.1831 ± 0.0060 |
| trajectory | 83 | 0.1825 ± 0.0032 | 0.1831 ± 0.0060 |
| **traj − end** | | +0.0000 | +0.0000 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0000

### adding_T128, clu_steps=2
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.1827 ± 0.0036 | 0.1820 ± 0.0036 |
| trajectory | 53 | 0.1832 ± 0.0038 | 0.1819 ± 0.0039 |
| **traj − end** | | +0.0004 | -0.0001 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0006

### parity_T64, clu_steps=1  ⚠ read modes are the SAME MAP at clu_steps=1 (fiber of one)
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.5334 ± 0.0053 | 0.5233 ± 0.0051 |
| trajectory | 83 | 0.5334 ± 0.0053 | 0.5233 ± 0.0051 |
| **traj − end** | | +0.0000 | +0.0000 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0000

### parity_T64, clu_steps=2
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.5337 ± 0.0056 | 0.5345 ± 0.0048 |
| trajectory | 53 | 0.5253 ± 0.0086 | 0.5321 ± 0.0051 |
| **traj − end** | | -0.0084 | -0.0023 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = -0.0061

### mqar_T128_kv4, clu_steps=1  ⚠ read modes are the SAME MAP at clu_steps=1 (fiber of one)
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.3864 ± 0.0032 | 0.3464 ± 0.0080 |
| trajectory | 83 | 0.3864 ± 0.0032 | 0.3464 ± 0.0080 |
| **traj − end** | | +0.0000 | +0.0000 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0000

### mqar_T128_kv4, clu_steps=2
| read mode | d_clu | γ=0 | γ=0.05 |
|---|---|---|---|
| endpoint | 83 | 0.2952 ± 0.0069 | 0.3298 ± 0.0092 |
| trajectory | 53 | 0.2881 ± 0.0089 | 0.3034 ± 0.0092 |
| **traj − end** | | -0.0072 | -0.0264 |

interaction (Δ@γ=0) − (Δ@γ=0.05) = +0.0192

## Item 3 — clu_steps (adding, at the best gamma)

| cell | d_clu | half-life (tok) | best lr | mse |
|---|---|---|---|---|
| {'clu_gamma': 0.02, 'clu_steps': 1} | 83 | 68.6 | 0.0003 | 0.1816 ± 0.0043 |
| {'clu_gamma': 0.02, 'clu_steps': 2} | 83 | 34.3 | 0.003 | 0.1824 ± 0.0031 |
| {'clu_gamma': 0.02, 'clu_steps': 4} | 83 | 17.2 | 0.0003 | 0.1817 ± 0.0034 |
