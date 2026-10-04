# gen-atom-mean-resweep — assembled tables

## T1 ⭐ DISAGREEMENT vs n_val  (⛔ the headline)

| n_val | tune_n | rows/half | cells | ⭐ INTERLEAVED disagree | contiguous disagree | gap best→2nd | median gap | trial spread | realised n_fit | fit traded | σ_c(half) ± se | ⭐ median gap / σ_c | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 256 | 128 | 6 | **5 / 6** | 4 / 6 | 0.0007–0.0130 | 0.0042 | 0.0678–0.1111 | 32,256 | 0 | 0.0061 ± 0.0010 | **0.69** | 398 |
| 512 | 512 | 256 | 6 | **3 / 6** | 5 / 6 | 0.0020–0.0094 | 0.0028 | 0.0678–0.1061 | 32,256 | 0 | 0.0046 ± 0.0014 | **0.61** | 546 |
| 1024 | 1024 | 512 | 6 | **3 / 6** | 5 / 6 | 0.0003–0.0042 | 0.0013 | 0.0686–0.1042 | 31,744 | 512 | 0.0023 ± 0.0006 | **0.56** | 901 |
| 2048 | 2048 | 1024 | 6 | **4 / 6** | 4 / 6 | 0.0008–0.0044 | 0.0019 | 0.0669–0.1082 | 30,720 | 1536 | 0.0032 ± 0.0004 | **0.60** | 1721 |
| 4096 | 4096 | 2048 | 6 | **1 / 6** | 3 / 6 | 0.0002–0.0061 | 0.0020 | 0.0699–0.1043 | 28,672 | 3584 | 0.0014 ± 0.0006 | **1.42** | 2732 |
| 8192 | 8192 | 4096 | 6 | **2 / 6** | 4 / 6 | 0.0002–0.0041 | 0.0015 | 0.0674–0.1063 | 24,576 | 7680 | 0.0009 ± 0.0003 | **1.72** | 5856 |

## T2 The six cells, per n_val (⛔ every cell, never a chosen few)

| n_val | cell | agree (interleaved) | selected | ihalf1 | ihalf2 | gap | c_full | c_h1 | c_h2 |
|---|---|---|---|---|---|---|---|---|---|
| 512/256 | s0/A | ✅ | k3/euc/none/per | k3/euc/none/per | k3/euc/none/per | 0.0130 | +0.0130 | +0.0156 | +0.0104 |
| 512/256 | s0/B | ❌ | k3/euc/none/per | k1/cos/none/per | k3/euc/none/per | 0.0072 | +0.0072 | +0.0000 | +0.0143 |
| 512/256 | s1/A | ❌ | k3/cos/none/per | k1/cos/none/per | k3/cos/none/per | 0.0046 | +0.0046 | -0.0026 | +0.0117 |
| 512/256 | s1/B | ❌ | k1/euc/none/per | k1/euc/none/per | k1/cos/none/per | 0.0007 | +0.0007 | +0.0052 | -0.0039 |
| 512/256 | s2/A | ❌ | k1/cos/none/per | k1/cos/none/per | k1/euc/none/per | 0.0039 | +0.0039 | +0.0013 | +0.0065 |
| 512/256 | s2/B | ❌ | k1/cos/none/per | k3/cos/none/per | k1/cos/none/per | 0.0039 | +0.0039 | -0.0013 | +0.0091 |
| 512/512 | s0/A | ✅ | k3/euc/none/per | k3/euc/none/per | k3/euc/none/per | 0.0055 | +0.0055 | +0.0072 | +0.0039 |
| 512/512 | s0/B | ❌ | k3/euc/none/per | k1/cos/none/per | k3/euc/none/per | 0.0026 | +0.0026 | -0.0052 | +0.0104 |
| 512/512 | s1/A | ❌ | k5/cos/none/per | k5/cos/none/per | k3/cos/none/per | 0.0020 | +0.0020 | +0.0039 | +0.0000 |
| 512/512 | s1/B | ✅ | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | 0.0029 | +0.0029 | +0.0046 | +0.0013 |
| 512/512 | s2/A | ✅ | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | 0.0094 | +0.0094 | +0.0072 | +0.0117 |
| 512/512 | s2/B | ❌ | k1/cos/none/per | k3/cos/none/per | k1/cos/none/per | 0.0020 | +0.0020 | -0.0046 | +0.0085 |
| 1024/1024 | s0/A | ❌ | k3/cos/none/per | k5/euc/none/per | k3/euc/none/per | 0.0003 | +0.0003 | +0.0039 | -0.0033 |
| 1024/1024 | s0/B | ✅ | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | 0.0013 | +0.0013 | +0.0003 | +0.0023 |
| 1024/1024 | s1/A | ❌ | k5/cos/none/per | k3/cos/none/per | k3/euc/none/per | 0.0008 | +0.0008 | +0.0036 | -0.0020 |
| 1024/1024 | s1/B | ❌ | k1/cos/none/per | k1/cos/none/per | k3/euc/none/per | 0.0042 | +0.0042 | +0.0065 | +0.0020 |
| 1024/1024 | s2/A | ✅ | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | 0.0013 | +0.0013 | +0.0007 | +0.0020 |
| 1024/1024 | s2/B | ✅ | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | 0.0021 | +0.0021 | +0.0013 | +0.0029 |
| 2048/2048 | s0/A | ❌ | k3/cos/none/per | k3/euc/none/per | k3/cos/none/per | 0.0008 | +0.0008 | -0.0010 | +0.0026 |
| 2048/2048 | s0/B | ❌ | k3/cos/none/per | k3/cos/none/per | k3/euc/none/per | 0.0019 | +0.0019 | +0.0055 | -0.0018 |
| 2048/2048 | s1/A | ❌ | k5/cos/none/per | k5/cos/none/per | k3/euc/none/per | 0.0010 | +0.0010 | +0.0041 | -0.0021 |
| 2048/2048 | s1/B | ❌ | k1/cos/none/per | k1/cos/none/per | k1/euc/none/per | 0.0024 | +0.0024 | +0.0047 | +0.0000 |
| 2048/2048 | s2/A | ✅ | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | 0.0044 | +0.0044 | +0.0072 | +0.0016 |
| 2048/2048 | s2/B | ✅ | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | 0.0020 | +0.0020 | +0.0034 | +0.0005 |
| 4096/4096 | s0/A | ✅ | k3/euc/none/per | k3/euc/none/per | k3/euc/none/per | 0.0021 | +0.0021 | +0.0021 | +0.0020 |
| 4096/4096 | s0/B | ✅ | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | 0.0020 | +0.0020 | +0.0017 | +0.0023 |
| 4096/4096 | s1/A | ❌ | k5/cos/none/per | k3/cos/none/per | k3/euc/none/per | 0.0002 | +0.0002 | +0.0007 | -0.0002 |
| 4096/4096 | s1/B | ✅ | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | 0.0061 | +0.0061 | +0.0045 | +0.0077 |
| 4096/4096 | s2/A | ✅ | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | 0.0035 | +0.0035 | +0.0068 | +0.0001 |
| 4096/4096 | s2/B | ✅ | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | 0.0013 | +0.0013 | +0.0002 | +0.0023 |
| 8192/8192 | s0/A | ✅ | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | 0.0012 | +0.0012 | +0.0011 | +0.0013 |
| 8192/8192 | s0/B | ❌ | k3/euc/none/per | k3/cos/none/per | k3/euc/none/per | 0.0002 | +0.0002 | +0.0004 | +0.0001 |
| 8192/8192 | s1/A | ✅ | k5/cos/none/per | k5/cos/none/per | k5/cos/none/per | 0.0018 | +0.0018 | +0.0036 | +0.0000 |
| 8192/8192 | s1/B | ✅ | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | 0.0041 | +0.0041 | +0.0044 | +0.0038 |
| 8192/8192 | s2/A | ✅ | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | 0.0038 | +0.0038 | +0.0032 | +0.0045 |
| 8192/8192 | s2/B | ❌ | k3/cos/none/per | k1/cos/none/per | k3/cos/none/per | 0.0008 | +0.0008 | -0.0004 | +0.0020 |

## T3 ⭐ THE PRICE — knn_proto (no tuning axis) re-fitted on the surviving fit rows

| n_val | realised n_fit | % of 32,256 | s0 | s1 | s2 | mean ± sd | Δ vs published | bit-identical |
|---|---|---|---|---|---|---|---|---|
| 512/256 | 32,256 | 100.0 % | 0.8788 | 0.8844 | 0.8819 | **0.8817 ± 0.0023** | +0.0000 | ✅ |
| 512/512 | 32,256 | 100.0 % | 0.8788 | 0.8844 | 0.8819 | **0.8817 ± 0.0023** | +0.0000 | ✅ |
| 1024/1024 | 31,744 | 98.4 % | 0.8763 | 0.8869 | 0.8844 | **0.8825 ± 0.0046** | +0.0008 | ⚠ waived |
| 2048/2048 | 30,720 | 95.2 % | 0.8763 | 0.8894 | 0.8794 | **0.8817 ± 0.0056** | -0.0000 | ⚠ waived |
| 4096/4096 | 28,672 | 88.9 % | 0.8687 | 0.8794 | 0.8844 | **0.8775 ± 0.0066** | -0.0042 | ⚠ waived |
| 8192/8192 | 24,576 | 76.2 % | 0.8838 | 0.8920 | 0.8719 | **0.8826 ± 0.0083** | +0.0008 | ⚠ waived |

## T4 ⭐ THE DEPLOYABLE (request-addressed) EXPONENT — ⛔ SUSPENDED, NON-ADJUDICATING

| n_val | leg | s0 | s1 | s2 | mean ± sd |
|---|---|---|---|---|---|
| 512/256 | `exp_req_k1` | -0.153105 | -0.166621 | -0.153665 | **-0.1578 ± 0.0062** |
| 512/256 | `exp_req_ksel` | -0.174023 | -0.072574 | -0.153665 | **-0.1334 ± 0.0438** |
| 512/256 | `exp_oracle_k1` | -0.259336 | -0.276301 | -0.253965 | **-0.2632 ± 0.0095** |
| 512/512 | `exp_req_k1` | -0.153105 | -0.169507 | -0.153665 | **-0.1588 ± 0.0076** |
| 512/512 | `exp_req_ksel` | -0.174023 | -0.083264 | -0.071056 | **-0.1094 ± 0.0459** |
| 512/512 | `exp_oracle_k1` | -0.259336 | -0.273411 | -0.253965 | **-0.2622 ± 0.0082** |
| 1024/1024 | `exp_req_k1` | -0.151259 | -0.169391 | -0.153253 | **-0.1580 ± 0.0081** |
| 1024/1024 | `exp_req_ksel` | -0.050228 | -0.083292 | -0.071167 | **-0.0682 ± 0.0137** |
| 1024/1024 | `exp_oracle_k1` | -0.259596 | -0.273411 | -0.253965 | **-0.2623 ± 0.0082** |
| 2048/2048 | `exp_req_k1` | -0.152736 | -0.168857 | -0.151902 | **-0.1578 ± 0.0078** |
| 2048/2048 | `exp_req_ksel` | -0.174024 | -0.083866 | -0.069925 | **-0.1093 ± 0.0461** |
| 2048/2048 | `exp_oracle_k1` | -0.259596 | -0.273411 | -0.253965 | **-0.2623 ± 0.0082** |
| 4096/4096 | `exp_req_k1` | -0.153996 | -0.171853 | -0.154572 | **-0.1601 ± 0.0083** |
| 4096/4096 | `exp_req_ksel` | -0.180509 | -0.084834 | -0.072633 | **-0.1127 ± 0.0482** |
| 4096/4096 | `exp_oracle_k1` | -0.256358 | -0.273411 | -0.253965 | **-0.2612 ± 0.0087** |
| 8192/8192 | `exp_req_k1` | -0.152011 | -0.169886 | -0.155699 | **-0.1592 ± 0.0077** |
| 8192/8192 | `exp_req_ksel` | -0.166331 | -0.082949 | -0.196279 | **-0.1485 ± 0.0480** |
| 8192/8192 | `exp_oracle_k1` | -0.262574 | -0.273411 | -0.253965 | **-0.2633 ± 0.0080** |

## T5 NOT RUN / wall clock

- 512/256: seeds [0, 1, 2] · wall 14.0 min
- 512/512: seeds [0, 1, 2] · wall 16.5 min
- 1024/1024: seeds [0, 1, 2] · wall 23.3 min
- 2048/2048: seeds [0, 1, 2] · wall 36.5 min
- 4096/4096: seeds [0, 1, 2] · wall 52.8 min
- 8192/8192: seeds [0, 1, 2] · wall 105.5 min
## X1 ⭐ THE SELECTED WINNER AT EVERY n_val — does the ladder converge?

| cell | 512/256 | 512/512 | 1024/1024 | 2048/2048 | 4096/4096 | 8192/8192 | distinct |
|---|---|---|---|---|---|---|---|
| s0/A | k3/euc/none/per | k3/euc/none/per | k3/cos/none/per | k3/cos/none/per | k3/euc/none/per | k3/cos/none/per | **2** |
| s0/B | k3/euc/none/per | k3/euc/none/per | k1/cos/none/per | k3/cos/none/per | k3/cos/none/per | k3/euc/none/per | **3** |
| s1/A | k3/cos/none/per | k5/cos/none/per | k5/cos/none/per | k5/cos/none/per | k5/cos/none/per | k5/cos/none/per | **2** |
| s1/B | k1/euc/none/per | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | **2** |
| s2/A | k1/cos/none/per | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | k3/cos/none/per | **2** |
| s2/B | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | k1/cos/none/per | k3/cos/none/per | **2** |

## X2 ⛔ n_val = 4,096 vs n_val = 8,192 — two draws that BOTH pass the gate — pick the SAME winner in only **3 of 6** cells

| cell | n_val 4,096 | n_val 8,192 | |
|---|---|---|---|
| s0/A | k3/euc/none/per | k3/cos/none/per | ⛔ **DIFFER** |
| s0/B | k3/cos/none/per | k3/euc/none/per | ⛔ **DIFFER** |
| s1/A | k5/cos/none/per | k5/cos/none/per | ✅ match |
| s1/B | k1/cos/none/per | k1/cos/none/per | ✅ match |
| s2/A | k3/cos/none/per | k3/cos/none/per | ✅ match |
| s2/B | k1/cos/none/per | k3/cos/none/per | ⛔ **DIFFER** |

## X3 ⭐ PER-AXIS IDENTIFICATION — which parts of the tuple are real?

### (a) ACROSS n_val points

| axis | cells stable over ALL 6 points | over the 4 points n_val ≥ 1024 | verdict |
|---|---|---|---|
| `k` | 2 / 6 | 4 / 6 | ⛔ **NOT identified** (4/6 stable even at n_val ≥ 1024) |
| `metric` | 3 / 6 | 4 / 6 | ⛔ **NOT identified** (4/6 stable even at n_val ≥ 1024) |
| `normaliser` | 6 / 6 | 6 / 6 | ✅ **IDENTIFIED** (unanimous) |
| `scope` | 6 / 6 | 6 / 6 | ✅ **IDENTIFIED** (unanimous) |

### (b) WITHIN a draw — the interleaved-half test, decomposed per axis

| n_val | `k` | `metric` | `normaliser` | `scope` | whole tuple |
|---|---|---|---|---|---|
| 512/256 | 3/6 | 3/6 | 0/6 | 0/6 | **5/6** |
| 512/512 | 3/6 | 1/6 | 0/6 | 0/6 | **3/6** |
| 1024/1024 | 3/6 | 3/6 | 0/6 | 0/6 | **3/6** |
| 2048/2048 | 1/6 | 4/6 | 0/6 | 0/6 | **4/6** |
| 4096/4096 | 1/6 | 1/6 | 0/6 | 0/6 | **1/6** |
| 8192/8192 | 1/6 | 1/6 | 0/6 | 0/6 | **2/6** |

### (c) every value ever selected, over all 36 cell-points

- `k`: {3: 18, 1: 13, 5: 5}
- `metric`: {'cosine': 29, 'euclidean': 7}
- `normaliser`: {'none': 36}
- `scope`: {'per_well': 36}
