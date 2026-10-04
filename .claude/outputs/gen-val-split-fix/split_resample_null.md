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
