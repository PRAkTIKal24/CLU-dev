# gen-encoder-pretrain — bar 0.9 on ['scale', 'orientation']

**VERDICT: PASS** — at least one variant reaches the bar on BOTH bar factors on the SPREAD grid with the headline LINEAR probe

## probe=linear · pool=grid (n=4096)

| n_values_realised | 10 | 10 | 10 | 8 | 4 | 15 | | |
| chance | 0.100 | 0.100 | 0.100 | 0.125 | 0.250 | 0.067 | | |
| variant | floor_hue | wall_hue | object_hue | scale | shape | orientation | aggregate | bar |
|---|---|---|---|---|---|---|---|---|
| random_init | 0.9988 ± 0.0003 | 0.9989 ± 0.0005 | 0.9991 ± 0.0003 | 0.9361 ± 0.0030 | 0.9812 ± 0.0022 | 0.9865 ± 0.0023 | 0.9834 | **PASS** |
| recon_control | 0.9995 ± 0.0002 | 0.9994 ± 0.0003 | 0.9993 ± 0.0004 | 0.9408 ± 0.0072 | 0.9960 ± 0.0010 | 0.9912 ± 0.0009 | 0.9877 | **PASS** |

## probe=linear · pool=corner (n=209)

| n_values_realised | 2 | 2 | 2 | 3 | 3 | 3 | | |
| chance | 0.500 | 0.500 | 0.500 | 0.333 | 0.333 | 0.333 | | |
| variant | floor_hue | wall_hue | object_hue | scale | shape | orientation | aggregate | bar |
|---|---|---|---|---|---|---|---|---|
| random_init | 0.9890 ± 0.0155 | 0.9969 ± 0.0044 | 0.9937 ± 0.0089 | 0.8801 ± 0.0372 | 0.9114 ± 0.0558 | 0.9859 ± 0.0167 | 0.9595 | **FAIL** |
| recon_control | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.8470 ± 0.0475 | 1.0000 ± 0.0000 | 0.9968 ± 0.0045 | 0.9740 | **FAIL** |

## compute (⭐ a compute win is not a representation win)

| variant | steps | enc params | pretext-head params | D | train s (mean) |
|---|---|---|---|---|---|
| random_init | 0 | 428576 | 250467 | 2048 | 2.0 |
| recon_control | 1500 | 428576 | 250467 | 2048 | 2492.7 |
