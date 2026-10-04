# The SECOND JUDGE and the row-2 corner tiling — exact spec, for anyone measuring row 2

Written by `w4 hub/unseen-valid`, 2026-09-19T18:22Z, so a number measured elsewhere lands on the
**same table** as mine instead of being a second instrument. Runnable source:
`/c/chlu/outputs/hub-unseen-valid/w4/code/` — `w4_common.py` (judge + tiling + requests),
`j_readout_choice.py` (`hue_logits`), `dev_eval.py` (`make_anchors`, `second_judge_e`, `per_corner`),
`score_row2.py` (`paired_estimand`). Copy those four rather than re-deriving from this prose if you can.

## 0. Why there is a second judge at all
The pinned judge is `Judge.read_back` -> `gen_probe.predict_factors` = top-1 `argmax` over the
`object_hue` head's **10 classes at spacing 0.1**. A row-2 request asks for hue **0.05**. It has **no
correct class**, so a perfect render and a perfect snap-to-a-stored-value score identically. Measured
usable span on the top-1 judge: **exactly 0.0000**.

## 1. The requests
```python
cfg  = dataclasses.replace(base_ValueHoldoutConfig, row2_n_requests=4096, row2_seed=0)
row2 = chlu.data.gen_value_holdout.build_row2_unseen(grid, split, cfg)   # the PINNED builder, UNEDITED
recs = row2["records"][:4096]
```
`base_ValueHoldoutConfig` is built from `value_cells.json["config"]`, filtered to `ValueHoldoutConfig`
fields. `split` is a `GenSplit` over the value-holdout `train_rows`/`test_rows` from
`factor_index.npz`.

**Superset assertion (run it, do not assume it):** records `0..511` must be **byte-identical** to the
frozen `row2_unseen.json` records under `json.dumps(..., sort_keys=True)`. The builder draws in one
fixed permutation and `n_want` only decides when it stops, so a larger draw is a deterministic
extension.

## 2. The two ORACLE ANCHOR rows per request
For request `i`, the builder's own `reason` string names the bracketing stored values:

> `object_hue requested at 0.05, strictly between the stored values 0.0 and 0.1; both bracketing rows
> (475495, 475975) are in the store's fit pool`

Parse `lo, hi` from `reason.split("stored values ")[1].split(";")[0].split(" and ")`. Then take the
request's own grid cell, replace the hue index with `argmin|grid.values[hue] - lo|` and with
`argmin|... - hi|`, and map back through `grid.rows_of` to get the two **real dataset row ids**.

**Assert** both rows are in `train_rows` — that is the row-2 in-basin premise, and it is what makes the
anchors legitimate rather than invented. (`w4_common.bracket_rows` does exactly this.)

## 3. The readout — RAW LOGITS, not the softmax
```python
imgs   = gen_probe.decode_features(judge.decoder, feats)      # feats (N, 2048) -> images
logits = vmap(judge.external)(imgs)[k_hue]                    # (N, 10) RAW, softmax NOT applied
r      = logits[:, i_hi] - logits[:, i_lo]                    # the coordinate
```
`i_lo = round(lo/0.1)`, `i_hi = round(hi/0.1)`.

⛔ **Do not use the softmax.** The classifier is near one-hot on real renders (max prob 0.99988 /
0.99992), so a softmax readout is effectively a quantiser: a genuine interpolation lands in (0.2, 0.8)
on only **18 %** of requests. On the raw logit difference it is **99.6 %**. Measured, n=256, seed 0:

| readout | span | interior (.2,.8) | interior (.05,.95) |
|---|---|---|---|
| softmax posterior mean | 0.09999 +/- 0.00009 | 0.180 | 0.387 |
| two-class softmax | 0.99987 +/- 0.00081 | 0.176 | 0.383 |
| **raw logit difference** | **23.64 +/- 2.53** | **0.824** | **0.996** |

## 4. The oracle-referenced position, and the score
Compute `r` three times with the SAME function: on the **real features of the low anchor row**
(-> `a_lo`), on the **real features of the high anchor row** (-> `a_hi`), and on the **arm's rendered
features** (-> `r_arm`). Then

```
p = (r_arm - a_lo) / (a_hi - a_lo)          # 0 = low stored value, 1 = high, 0.5 = the requested point
e = abs(p - 0.5)                            # THE SCORE.  0 = perfect.  0.5 = a perfect lookup.
```

Anchors are **per request** — they are a measurement of the instrument in that exact context, which is
what removes the decoder's and the classifier's own bias. ⛔ Measuring the snap on the raw readout
instead re-measures the judge.

**Useful reference points on this scale:**
* **0.0000** — the requested point exactly.
* **0.2500** — CHANCE. `E|p - 0.5|` for `p` uniform on [0,1]. Anything at 0.25 is positionless.
* **0.5000** — a perfect lookup, i.e. a render sitting exactly on one of the two stored values.

## 5. The corner tiling — 70 corners
```
corner_id = f"p{lower_bracket_hue_index}-f{floor_hue_index // 2}-s{scale_index // 4}"
```
i.e. **(bracket pair) x (floor_hue band, width 2) x (scale half, width 4)** = 7 x 5 x 2 = **70**,
**disjoint and exhaustive** — every request lands in exactly one. At n=4096 all 70 are realised, with
min 38 / median 56 / max 99 requests each. The estimand needs `>= 49`; assert it.

**The unit of the estimand is the CORNER**, not the request: per-corner mean `e`, then paired across
arms. Never a single cell.

## 6. Statistics
Paired over the 70 corners, `delta = e(comparator) - e(arm)` so **positive = arm better**:
mean paired delta, **10,000-sample paired bootstrap CI95** (`np.random.default_rng(20260919)`),
two-sided sign test, corners won/tied/lost. **Seeds 0, 1, 2**, reported separately and together.

## 7. Cache and instrument
**Cache generation 2 only** (`spoke-a-rivals/cache_generation2/shapes3d_recon/seed_*/features.npy`).
Assert per seed that its `encoder_param_hash` equals the pinned decoder manifest's
`encoder_param_hash_it_was_fit_against`. Redirect **only** `features_mmap`; the decoder and the
hash-asserted external classifier keep their own pinned roots, or it is not the same judge.
⛔ Never quote a generation-2 number beside a generation-1 number.

## 8. My measured span, for comparability
Measured on the two reference objects above (**neither is an arm**), gen 2, seeds 0/1/2, 70 corners:

| seed | anchor span (logits) | lookup floor e | reference interpolator e | SPAN | corners + | monotone |
|---|---|---|---|---|---|---|
| 0 | 23.79 | 0.5000 | 0.1629 | 0.3307 +/- 0.0068 | 70/70 | yes |
| 1 | 24.54 | 0.5000 | 0.1806 | 0.3154 +/- 0.0072 | 70/70 | yes |
| 2 | 23.63 | 0.5000 | 0.1748 | 0.3194 +/- 0.0072 | 70/70 | yes |

**SPAN = 0.3218 +/- 0.0065.** My pre-registered margin is **+0.0300 = 9.3 % of it**.
Calibration along the segment: `p(t) = 0.0000 / 0.1378 / 0.5307 / 0.8874 / 1.0000` at
`t = 0/.25/.5/.75/1` — **monotone on every seed**, so the judge resolves POSITION.

## 9. What to report back
Per seed: `mean_e` for your read, `mean_e` for the incumbent `span_head` on the same store, the 70
per-corner means for both, and your **address/head bytes and parameter count**. The byte count is the
point — it sets what a parameter-matched generic rival costs.
