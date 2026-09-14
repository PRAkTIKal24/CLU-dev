### THE TABLE — in-corner `shape` balanced accuracy on IMAGES (external classifier)

| arm | deployed bytes | tuning trials | IMAGE balacc (ext) | per seed | AUROC | TPR / TNR | distinct vectors of 72 | ⭐ distinct factor-tuples of 72 | probe companion |
|---|---|---|---|---|---|---|---|---|---|
| `clu` | 524,288 + 74,240 head | 0 of 36 | **0.6568 ± 0.0226** | 0.6540 / 0.6307 / 0.6858 | 0.7653 ± 0.0123 | 0.7361 ± 0.2051 / 0.5776 ± 0.2313 | 9.0000 ± 0.8165 | **9.0000 ± 0.8165** | 0.6759 ± 0.0107 |
| `store_removed` | 524,288 + 74,240 head | 0 of 36 | **0.5000 ± 0.0000** | 0.5000 / 0.5000 / 0.5000 | 0.5008 ± 0.0012 | 0.0000 ± 0.0000 / 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** | 0.5000 ± 0.0000 |
| `knn_proto` | 475,136 | 0 of 36 | **0.9119 ± 0.0222** | 0.9040 / 0.9422 / 0.8894 | 0.9875 ± 0.0088 | 1.0000 ± 0.0000 / 0.8238 ± 0.0445 | 72.0000 ± 0.0000 | **54.6667 ± 8.0554** | 0.8884 ± 0.0088 |
| `lace` | — | 36 of 36 | **0.6347 ± 0.0177** | 0.6143 / 0.6324 / 0.6575 | 0.7186 ± 0.0147 | 0.3935 ± 0.0524 / 0.8759 ± 0.0312 | 72.0000 ± 0.0000 | **71.3333 ± 0.9428** | 0.6389 ± 0.0112 |
| `ebm_compose` | 14,226,660 | 36 of 36 | **0.5408 ± 0.0087** | 0.5455 / 0.5287 / 0.5483 | 0.6077 ± 0.0134 | 0.1991 ± 0.0327 / 0.8825 ± 0.0157 | 72.0000 ± 0.0000 | **71.6667 ± 0.4714** | 0.5318 ± 0.0090 |
| `lace/ld_published` | — | 36 of 36 | **0.5580 ± 0.0564** | 0.6288 / 0.5544 / 0.4908 | 0.7599 ± 0.0604 | 0.1898 ± 0.1029 / 0.9262 ± 0.0349 | 72.0000 ± 0.0000 | **56.3333 ± 4.1096** | 0.5924 ± 0.1010 |
| `lace/ld_corrected` | — | 36 of 36 | **0.6605 ± 0.0197** | 0.6338 / 0.6669 / 0.6808 | 0.7299 ± 0.0203 | 0.4352 ± 0.0346 / 0.8859 ± 0.0050 | 72.0000 ± 0.0000 | **71.3333 ± 0.4714** | 0.6420 ± 0.0120 |
| *decoder ceiling (real features)* | — | — | *0.9357 ± 0.0193* | 0.9085 / 0.9472 / 0.9514 | — | — | 72 | — | *0.9414 ± 0.0179* |

### B-set composition, on IMAGES

| arm | B aggregate (ext) | B `shape` (ext) | distinct factor-tuples on MS of ~199 |
|---|---|---|---|
| `clu` | 0.5949 ± 0.0082 | 0.7361 ± 0.2051 | 24.3333 ± 5.4365 |
| `store_removed` | 0.1204 ± 0.0131 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `knn_proto` | 0.9005 ± 0.0182 | 1.0000 ± 0.0000 | 148.6667 ± 12.6842 |
| `lace` | 0.3056 ± 0.0208 | 0.3935 ± 0.0524 | 195.3333 ± 0.9428 |
| `ebm_compose` | 0.2130 ± 0.0132 | 0.1991 ± 0.0327 | 198.0000 ± 0.0000 |
| `lace/ld_published` | 0.2276 ± 0.0165 | 0.1898 ± 0.1029 | 117.0000 ± 8.2865 |
| `lace/ld_corrected` | 0.2855 ± 0.0159 | 0.4352 ± 0.0346 | 194.6667 ± 0.4714 |

### Instrument invariance — does swapping the reader re-order the arms?

| arm | external | probe | Δ |
|---|---|---|---|
| `clu` | 0.6568 | 0.6759 | -0.0191 |
| `store_removed` | 0.5000 | 0.5000 | +0.0000 |
| `knn_proto` | 0.9119 | 0.8884 | +0.0235 |
| `lace` | 0.6347 | 0.6389 | -0.0041 |
| `ebm_compose` | 0.5408 | 0.5318 | +0.0090 |
| `lace/ld_published` | 0.5580 | 0.5924 | -0.0344 |
| `lace/ld_corrected` | 0.6605 | 0.6420 | +0.0185 |

ranking by external: ['knn_proto', 'lace/ld_corrected', 'clu', 'lace', 'lace/ld_published', 'ebm_compose', 'store_removed']
ranking by probe:    ['knn_proto', 'clu', 'lace/ld_corrected', 'lace', 'lace/ld_published', 'ebm_compose', 'store_removed']
identical: False

### Regression check vs the published probe numbers

| arm | seed | measured (probe) | published | Δ | bit-identical | REGRESSED |
|---|---|---|---|---|---|---|
| `clu` | 0 | 0.6610 | 0.6610 | +0.0000 | ✅ | no |
| `knn_proto` | 0 | 0.8763 | 0.8763 | +0.0000 | ✅ | no |
| `lace` | 0 | 0.6237 | 0.7077 | -0.0840 | — | ⛔ YES |
| `clu` | 1 | 0.6809 | 0.6809 | +0.0000 | ✅ | no |
| `knn_proto` | 1 | 0.8920 | 0.8920 | +0.0000 | ✅ | no |
| `lace` | 1 | 0.6423 | 0.6732 | -0.0309 | — | no |
| `clu` | 2 | 0.6858 | 0.6858 | +0.0000 | ✅ | no |
| `knn_proto` | 2 | 0.8970 | 0.8970 | +0.0000 | ✅ | no |
| `lace` | 2 | 0.6505 | 0.6582 | -0.0077 | — | no |

### Sampler-noise floor

- seed 0 `lace`: sd **0.0204** over 3 independent sampler keys; values [0.6023, 0.6383, 0.5903]

### NOT RUN

- seed 0: nothing — every requested arm completed
- seed 1: nothing — every requested arm completed
- seed 2: nothing — every requested arm completed
- seeds requested [0, 1, 2], not run []

### Hash discipline (⛔ asserted per arm per seed)

- `clu`: decoder ['001011bc603a', '01ef91f75eea', '2d7eea75add7'] · classifier ['aa69b436989b', 'ad16c4e53273', 'c8741ac417d8']
- `store_removed`: decoder ['001011bc603a', '01ef91f75eea', '2d7eea75add7'] · classifier ['aa69b436989b', 'ad16c4e53273', 'c8741ac417d8']
- `knn_proto`: decoder ['001011bc603a', '01ef91f75eea', '2d7eea75add7'] · classifier ['aa69b436989b', 'ad16c4e53273', 'c8741ac417d8']
- `lace`: decoder ['001011bc603a', '01ef91f75eea', '2d7eea75add7'] · classifier ['aa69b436989b', 'ad16c4e53273', 'c8741ac417d8']
- `ebm_compose`: decoder ['001011bc603a', '01ef91f75eea', '2d7eea75add7'] · classifier ['aa69b436989b', 'ad16c4e53273', 'c8741ac417d8']
- `lace/ld_published`: decoder ['001011bc603a', '01ef91f75eea', '2d7eea75add7'] · classifier ['aa69b436989b', 'ad16c4e53273', 'c8741ac417d8']
- `lace/ld_corrected`: decoder ['001011bc603a', '01ef91f75eea', '2d7eea75add7'] · classifier ['aa69b436989b', 'ad16c4e53273', 'c8741ac417d8']

### LACE best-of-family (R9.4: *report LACE's best*), probe-vs-probe

| seed | published best (sampler) | measured best (sampler) | Δ |
|---|---|---|---|
| 0 | 0.7418 (`ld_published`) | 0.7348 (`ld_published`) | -0.0069 |
| 1 | 0.6732 (`ode`) | 0.6423 (`ode`) | -0.0309 |
| 2 | 0.6582 (`ode`) | 0.6531 (`ld_corrected`) | -0.0052 |

**best-of-family: published 0.6911 ± 0.0364 · measured 0.6767 ± 0.0413 · Δ -0.0143**
