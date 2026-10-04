# gen-montero-splits-spec — web-scout report

Task + acceptance criterion: Pin, from primary source, the exact recombination-to-element train/test split of Montero et al. (ICLR 2021) for Shapes3D and dSprites, in enough detail to regenerate it blind. Deliver `montero_splits.json` with a `status` on every field.

Status: **done** (acceptance met; 6 fields honestly marked NOT OBTAINED, see §Gaps)

⭐ **DOWNSTREAM RECONCILIATION LIST — HAS AN OWNER? NO. See §Reconciliation (needs a Hub-assigned owner this review).** The paper's prose rule and the shipped code's rule **materially disagree for BOTH datasets**. Anyone who implements from the paper text — which is the obvious thing to do — builds the wrong split.

## ⭐ DIAL DECLARATION (echoed per protocol §7)
- **Dial:** *none — instrument / spec recon.* No CLU number, no architecture opinion, no performance claim made below.
- **Laundering control:** N/A (no measurement).
- **Falsifies:** N/A. Failure mode guarded against was a silently reconstructed split.

---

## Answer first

The official implementation **exists** (`github.com/mmrl/disent-and-gen`), and per the task's primacy rule it is the source I pinned: the splits live in **`scripts/configs/datasplits.py`** as deterministic boolean masks, **no seed, no sampling** — train is the exact complement of test over the full factor grid. **The paper's prose rule contradicts the code for both datasets**, and the paper additionally **contradicts itself** on dSprites rotation. Under the code, held-out sets are **72 images for Shapes3D** (of 480,000) and **5,808 for dSprites** (of 737,280); the paper reports **no** counts of its own, so no author-vs-me discrepancy can be computed.

---

## What I did
- Read the ICLR 2021 camera-ready PDF in full (pp. 1–10) after OpenReview's bot-wall blocked direct extraction — recovered it via the authors' lab mirror and read the PDF pages directly.
- Located the official code, pinned it at commit **`bff35a7570d08fdef0d10c21b093b9a24d98dde6`** (2022-05-28), traced the dispatch chain `get_data_spliters` → `configs.datasplits.{Dsprites,Shapes3D}.get_splits`.
- Extracted the filter expressions, index constants and angle constants; cross-checked factor grids against the DeepMind dataset repos and the TFDS catalog.
- Computed `n_train`/`n_test` myself from the code filters and the factor grids.

## How I verified
- **Angle-constant trap, verified twice via two different CDNs** (`raw.githubusercontent` and `cdn.jsdelivr.net`), because the first extraction produced a *logically impossible* clause and I did not accept it. Both returned the identical line:
  `a90, a120, a180, a240 = np.pi / 2, 4 * np.pi / 3, np.pi, 2 * np.pi / 3`
  The **names are bound to swapped values**: `a120` = 4π/3 = **240°**, `a240` = 2π/3 = **120°**. So `(rot < a120) & (rot > a240)` is not impossible — it means **120° < θ < 240°**, which reconciles with §2.1 prose. An engineer who "fixes" the apparent bug inverts the held-out rotation band.
- **Column-offset check (dSprites):** confirmed `load_sprites` slices the colour column off at load — `latent_values = data_zip['latents_values'][:, 1:]` — so `shp,scl,rot,tx,ty = 0,1,2,3,4` are valid **only post-slice**. Applied to the raw 6-column npz, the recomb-to-element filter selects **nothing** (colour class is always 0, so `== 1` is empty). This is the single most likely silent-failure mode.
- **Shape-array cross-check:** recomb-to-**element** uses `latent_classes[:,shp]==1` → **ellipse**; recomb-to-**range** uses `latent_values[:,shp]==1` → **square**. Two different arrays, same literal, different shapes — and *both* match the paper's prose for their respective conditions. That internal consistency is strong evidence my reading of the indexing is correct.
- Counts computed by hand from the grids; grid totals independently confirmed (737,280 dSprites; 480,000 Shapes3D) against DeepMind READMEs and TFDS.

---

## Findings — the load-bearing result

### Shapes3D recombination-to-element
Code (`scripts/configs/datasplits.py`, class `Shapes3D`, `leave1out`), test filter:
```
(oh >= 0.8) & (wh >= 0.8) & (fh >= 0.8) & (scl >= 1.1) & (shp == 1) & (rot > 20)
```
Paper (§2.2, p.6), verbatim: *"we excluded one combination from training: [floor-hue > 0.5, wall-hue > 0.5, object-hue > 0.5, object-shape=cylinder, object-scale=1, object-orientation=0]."*

**The prose is not merely looser — it is not executable and partly inverted:**
- `object-scale = 1` **does not exist** in the scale grid (`np.linspace(0.75,1.25,8)`: nearest 0.964, 1.036). Proof the prose is a gloss.
- `object-orientation = 0` names the **centre** of the range; code selects the **top end** (`> 20`).
- Hues `> 0.5` (4/10 values) vs code `>= 0.8` (2/10) — a 64× larger hue block.
- **Tie-breaker:** Fig. 4 caption says *"the model has not seen magenta floors with purple cylinders"* — hues at the 0.8–0.9 end, corroborating the **code**.

⇒ **n_test = 2·2·2·3·1·3 = 72**; n_train = 479,928.

### dSprites recombination-to-element
Code, test filter (class `Dsprites`, `leave1out`):
```
(latent_classes[:,shp] == 1) & (latent_values[:,scl] > 0.6) &
(rot > 0.0) & (rot < a120) & (rot > a240) &        # ⇒ 120° < θ < 240°
(tx > 0.66) & (ty > 0.66)
```
Paper §2.1, p.4, verbatim: *"…we split each range of values of a generative factor into three bins, so that we had 3 × 3 × 3 × 3 × 3 = 243 such combinations… We then remove one of these 243 combinations during training, namely those that satisfied [shape=ellipsis, position-x >= 0.6, , position-y >= 0.6, 120° <= rotation <= 240°, scale < 0.6]."*

- **`scale < 0.6` (prose) vs `scale > 0.6` (code) — sign-inverted.** Code is backed by the p.3 gloss *"a **large** ellipse"* and Fig. 2's *"scale = 1"*. The §2.1 prose is very likely a typo.
- **The paper contradicts itself on rotation:** §2.1 says 120–240° (agrees with code); p.3 bullet and Fig. 2 caption both say *"orientation < 120°"*.
- **posX/posY threshold appears at three different values across the paper** (`> 0.5` p.3 & Fig. 2; `>= 0.6` §2.1); code uses `0.66`.
- **The "243 bins" story is not what the code implements.** `scale > 0.6` selects **4 of 6** values — two-thirds, not a tertile. Expected size under the prose is 737,280/243 ≈ 3,035; the code's cell holds **5,808** (≈1.9×).

⇒ **n_test = 1·4·12·11·11 = 5,808**; n_train = 731,472.

### ⚠ Float-boundary hazard (dSprites only, severity HIGH)
Two dSprites thresholds land *exactly* on grid values, so `n_test` is decided by float64 round-off, not by intent:
- `scale > 0.6` — stored 0.6 vs literal 0.6 ⇒ 4 or 5 scale values (n_test 5,808 or 7,260).
- 120°/240° land exactly on `np.linspace(0,2π,40)` indices 13 and 26 ⇒ 12, 13 or 14 rotation values (5,808 / 6,292 / 6,776).

**Mandatory instruction carried into the JSON:** do **not** regenerate the grid and reason about nominal values — evaluate the filter verbatim against the actual npz array (post `[:, 1:]` slice) and report the *realised* counts, reconciling against my 5,808. Shapes3D has no such hazard (LOW).

## Provenance (pointers only — nothing downloaded)
- dSprites: `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`, **19,318,135 bytes** (GitHub contents API), zlib/libpng licence, Matthey et al. 2017. No published checksum upstream.
- Shapes3D: `3dshapes.h5`, 255.18 MiB (TFDS), Burgess & Kim 2018. No published checksum; flat GCS URL marked RECONSTRUCTED.

---

## Reconciliation list (⭐ needs a Hub-assigned owner)
1. **Any B-cell design doc or task file that states the held-out combination in the paper's words must be rewritten to the code's rule** — thresholds `>=0.8 / >=1.1 / >20` for Shapes3D, `scale > 0.6 / 120–240° / 0.66` for dSprites.
2. **The data spoke must be handed the two traps explicitly** (colour-column slice; swapped angle-constant names) — both are silent-failure modes, one produces an empty test set.
3. Any statement that the dSprites hold-out is "1 of 243 bins" should not be repeated as fact.

## Confidence & gaps
- **Verified (2+ independent fetches or code+paper cross-check):** angle constants, Shapes3D filter, colour-column slice, factor grids, grid totals, commit hash.
- **Single-sourced:** the exact dSprites `leave1out` expression came from the same file via 3 fetches but one underlying blob — the *reading* is corroborated by its agreement with §2.1 prose on rotation and by the shape-array cross-check, but I could not view the file as plain text (the fetch layer refuses bulk verbatim and saved it as unreadable binary).
- **NOT OBTAINED (6):** sha256 of `datasplits.py` (no shell — git blob SHA-1 `a59ef1be…` and 11,666-byte size given instead); published checksums for both datasets (none exist upstream); exact bytes of `3dshapes.h5`; dSprites recomb2range scale/rotation variants and extrapolation x∈{0.16,0.25,0.75} (in prose, absent from pinned code); author-reported n_train/n_test (**the paper reports none**).
- **Not verified:** that the pinned commit's code is identical to what produced the published results. The camera-ready code is the *prior* commit `9b34e902…`; `bff35a7` only adds a licence — but I did not diff them. Cheap next check if it matters.

## Bibtex-ready refs
```bibtex
@inproceedings{montero2021role,
  title     = {The role of Disentanglement in Generalisation},
  author    = {Montero, Milton Llera and Ludwig, Casimir J.H. and Ponte Costa, Rui
               and Malhotra, Gaurav and Bowers, Jeffrey S.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://openreview.net/forum?id=qbH974jKUVy}
}
@misc{dsprites17,
  author = {Matthey, Loic and Higgins, Irina and Hassabis, Demis and Lerchner, Alexander},
  title  = {dSprites: Disentanglement testing Sprites dataset},
  howpublished = {https://github.com/deepmind/dsprites-dataset/}, year = {2017}
}
@misc{3dshapes18,
  title = {3D Shapes Dataset}, author = {Burgess, Chris and Kim, Hyunjik},
  howpublished = {https://github.com/deepmind/3dshapes-dataset/}, year = {2018}
}
```

Git footprint: **none** — read-only spoke, zero branches, zero worktrees, zero tracked-file edits. Artifacts written only under `.claude/outputs/`.

## Proposed handover updates (for the Hub)
- Montero splits are **PINNED from code**, not prose, at `mmrl/disent-and-gen@bff35a7`. Held-out: **Shapes3D 72 / 480,000**, **dSprites 5,808 / 737,280** (dSprites ±float-boundary; realised count to be reported by the data spoke).
- **Paper prose is unusable as a spec for both datasets** and self-contradictory for dSprites. Route the JSON, not the paper, to the data spoke.
- Assign an owner to the 3-item reconciliation list above at this review.
