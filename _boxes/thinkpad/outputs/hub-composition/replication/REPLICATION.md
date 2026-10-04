# Independent replication — `hub/composition`, three headline numbers

**Spoke:** independent replication. **Date:** 2026-09-16. **Charter §A.5** — *Unreplicated = not a result.*

Nothing under `.claude/outputs/hub-composition/mech/`, `.../instrument/` or `.../probes/` was read,
imported or copied. Every number below comes from code in this directory
(`claim1.py`, `claim2.py`, `claim3.py`, `claim3_sweep.py`, `claim3_extra.py`, `claim3_overlap.py`,
`merge.py`). Image features were **never encoded**: all features are read from the frozen cache via
`gen_bridge.features_for_rows`, which raises on an uncached row.

Shell preamble used for every command:

```sh
. /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/hub-composition/hubenv.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$OUT/replication"
```

---

## CLAIM 1 — the frozen table's exact-match column

**Command**

```sh
"$PY" claim1.py "$OUT/replication/claim1.json"
```

Source: the three frozen Mac manifests
`.claude/outputs/gen-read-cell-arm/runs/cell/seed_{0,1,2}/seed_{0,1,2}/cell_manifest.json`
(sha256[:16] — seed 0 `10093b69b46c7edf`, seed 1 `98fdb076d84ca6a0`, seed 2 `5111428316cf08ec`;
also recorded in `claim1.json` under `manifest_sha256_16`).
Statistic: `arms.<arm>.external.composition_B.exact_match * 72`, mean ± sd over the three seeds.

| row | seed 0 | seed 1 | seed 2 | my mean ± sd(ddof=0) | hub | verdict |
|---|---|---|---|---|---|---|
| **ceiling** | 67 | 71 | 65 | **67.667 ± 2.494** | 67.67 ± 2.49 | **CONFIRMED** |
| **knn_proto** | 12 | 10 | 21 | **14.333 ± 4.784** | 14.33 ± 4.78 | **CONFIRMED** |
| clu | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |
| lace | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |
| lace/ld_corrected | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |
| lace/ld_published | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |
| ebm_compose | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |
| dynread | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |
| dynread/B | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |
| dynread/A_request_off | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |
| dynread/B_request_off | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |
| store_removed | 0 | 0 | 0 | 0.000 ± 0.000 | 0 | **CONFIRMED** |

All 11 arms present in the manifests are listed; `composition_B.n == 72` on every arm and every seed.
The zero rows are **exactly** 0.0 in the JSON, not a rounded small number.

`distinct.B.factor_tuples` (requested alongside), seeds 0/1/2:

| arm | s0 | s1 | s2 |
|---|---|---|---|
| clu | 9 | 8 | 10 |
| dynread | 12 | 15 | 13 |
| dynread/A_request_off | 1 | 1 | 1 |
| dynread/B | 15 | 18 | 15 |
| dynread/B_request_off | 1 | 1 | 1 |
| ebm_compose | 72 | 72 | 72 |
| knn_proto | 57 | 51 | 60 |
| lace | 72 | 71 | 72 |
| lace/ld_corrected | 72 | 72 | 72 |
| lace/ld_published | 55 | 61 | 50 |
| store_removed | 1 | 1 | 1 |

### Discrepancy, stated as a number

**The hub's `±` is a population sd (ddof = 0), not a sample sd.** With n = 3 seeds the sample sd
(ddof = 1) is **3.055** for the ceiling (hub prints 2.49, difference **0.561**) and **5.859** for
knn_proto (hub prints 4.78, difference **1.075**). Both of my figures are in `claim1.json`
(`sd_pop`, `sd_sample`). The hub's numbers reproduce only under ddof = 0. For a 3-seed spread quoted
as "± sd", ddof = 0 understates the spread by a factor of sqrt(3/2) = 1.225 and is the less
conservative of the two conventions.

---

## CLAIM 2 — "all 72 held-out B target tuples are distinct, and none occurs in training"

**Command**

```sh
"$PY" claim2.py "$OUT/replication/claim2.json"
```

Split built independently via `dataset_module("shapes3d").build_split(cfg.gen_data.data_root)`;
tuples via `chlu.eval.gen_probe.factor_labels(split.grid, rows)`; counting by plain `set` arithmetic.

| quantity | hub | mine | verdict |
|---|---|---|---|
| `len(split.test_rows)` | 72 | **72** | **CONFIRMED** |
| distinct factor tuples among them | 72 | **72** | **CONFIRMED** |
| those tuples that also occur in `split.train_rows` | 0 | **0** | **CONFIRMED** |

Supporting facts I measured (same file): `n_train_rows = 479_928`, `n_distinct_train_tuples = 479_928`,
total `480_000` rows = the full 10×10×10×8×4×15 grid, and the row→tuple map is **injective on the whole
grid**. The held-out cell is exactly the box
`floor_hue ∈ {8,9} × wall_hue ∈ {8,9} × object_hue ∈ {8,9} × scale ∈ {5,6,7} × shape = 1 × orientation ∈ {12,13,14}`
= 2·2·2·3·3 = 72.

### The identity `distinct_valid_novel == 72 * exact_match` — DOES NOT FOLLOW

Because the grid is fully partitioned and the map is a bijection, "valid **and** novel" is *identical*
to "is one of the 72 held-out tuples". So

- `72 * exact_match` = #{ i : pred_i == target_i };
- `distinct_valid_novel` = | { pred_i } ∩ { the 72 held-out tuples } |.

Claim 2 buys exactly **one** direction: each exact match contributes a *distinct* element of that
intersection (the targets are distinct), and every target is valid-and-novel, so

> **distinct_valid_novel ≥ 72 · exact_match**, always.

Equality requires an extra premise Claim 2 does not supply: **no off-diagonal corner hit** — no arm may
emit, for target *i*, a tuple that equals a *different* held-out target *j*, unless *j* was already
matched exactly. That premise is not measured anywhere I can see, and it is not a technicality:

**Named case where it fails: `knn_proto`.** From the frozen manifest, seed 0, per-factor accuracy on B
is floor 1.000, wall 1.000, object 1.000, orientation 1.000, shape 0.986, **scale 0.167**. So all 72
generated tuples already have floor/wall/object/orientation *correct*, hence inside the corner's ranges,
and 71/72 have shape = 1. Corner membership therefore turns on one thing only: whether the predicted
scale lands in **{5,6,7}** — which is a *3-of-8* window, while an exact match needs the *one* correct
value. There are 59 rows that are correct on all five other factors and wrong on scale; every one of
them whose scale error stays inside {5,6,7} is a valid, novel, non-exact tuple. Scale accuracy of
0.167 sits between 0.125 (uniform over all 8 scale values) and 0.333 (uniform over the 3 in-corner
values), which is the signature of a predictor that lands in the window a substantial fraction of the
time. I therefore expect `distinct_valid_novel` for knn_proto to be well above 12 on seed 0, not equal
to it.

⚠ **Honesty note on this point:** I could **not** measure `distinct_valid_novel` directly. Doing so
needs the per-row predicted tuples, which are not in the manifests (I grepped: no key matching
`*novel*` exists in `cell_manifest.json`) and would otherwise require re-running the samplers or
reading the forbidden directories. What I *have* proved is that Claim 2 does not entail the identity,
and that the arm the identity is most load-bearing for is the arm most structurally exposed to its
failure mode.

**Second named case: the zero rows.** Under the identity, `exact_match = 0` forces
`distinct_valid_novel = 0` — i.e. the strictly stronger statement that *not one* of the arm's outputs
lands in the held-out cell. `ebm_compose` and `lace/ld_corrected` emit **72 distinct** tuples on every
seed and `lace` has an in-corner TPR (the manifest's own shape-channel gate statistic) of 0.264 / 0.472 /
0.403 across seeds. Reporting `distinct_valid_novel = 0` for them *by identity from `exact_match = 0`*
would assert something never measured. If the hub's table literally derives the novelty column from
the exact-match column, that column carries no independent information and should not be presented as
evidence of "no novel content".

---

## CLAIM 3 — the store round-trip bound and the residual control

Seed 0, the 512 `val` rows. Every artifact **loaded**, nothing fitted here:

- store: loaded from `.../gen-read-compose/runs/shapes3d_recon/seed_0/store_s0.pkl`; I set
  `cfg.gen_cell.allow_store_training = False` so the run would *fail* rather than silently train, and
  asserted `provenance.trained_here is False`.
- grouping: `exp_gen_dream.resolve_grouping` reported *"reproduces the plan BIT-IDENTICALLY on its own seed"*.
- decoder: `winner`, recipe `v1_capacity_schedule`, param hash `01ef91f75eeaae7b…`.
- external classifier: param hash `c8741ac417d88ebf…`, 978 873 params.
- rows: `pool = 32 768`, `fit = 32 256`, `val = 512` (no silent truncation).
- cache: 480 000 rows, D = 2048, `feat_shape = (128, 4, 4)`, scope `full`.

**Commands**

```sh
"$PY" claim3.py          "$OUT/replication/claim3.json"
PYTHONPATH="$OUT/replication:$PYTHONPATH" "$PY" claim3_sweep.py "$OUT/replication/claim3_sweep.json"
PYTHONPATH="$OUT/replication:$PYTHONPATH" "$PY" claim3_extra.py "$OUT/replication/claim3_extra.json"
"$PY" claim3_overlap.py  "$OUT/replication/claim3_overlap.json"
"$PY" merge.py
```

| # | quantity | hub | mine | verdict |
|---|---|---|---|---|
| a | decoder ceiling, exact 6/6 | 491 / 512 | **491 / 512** | **CONFIRMED** (exactly) |
| b | store round-trip, exact 6/6 | 10 / 512 | **10 / 512** | **CONFIRMED** (exactly) |
| b | mean per-row ‖rt−X‖/‖X‖ | 0.3486 | **0.34860397** | **CONFIRMED** (4 d.p.) |
| c | X + norm-matched Gaussian | 393 / 512 | **384.4 ± 9.5**, range 366–402 over 15 draws | **CONFIRMED within RNG spread** |
| c | …its rel-err | 0.3486 | **0.34860397**, max per-row ‖g‖−‖e‖ mismatch **7.6e-08** | **CONFIRMED** |
| c | X − e | 162 / 512 | **162 / 512** | **CONFIRMED** (exactly) |
| c | X + e[permutation] | 95 / 512 | **117.3 ± 8.0**, range 104–132 over 15 draws | **REFUTED as stated** |

Per-factor accuracy (floor, wall, object, scale, shape, orientation):

| condition | floor | wall | object | scale | shape | orient | exact |
|---|---|---|---|---|---|---|---|
| X (ceiling) | 1.000 | 1.000 | 1.000 | 0.973 | 0.988 | 0.998 | 491 |
| rt (round-trip) | 0.998 | 0.982 | 0.691 | **0.238** | 0.609 | **0.174** | 10 |
| X + g (‖g‖=‖e‖) | 1.000 | 1.000 | 0.998 | 0.807 | 0.971 | 0.940 | 377 |
| X − e | 1.000 | 1.000 | 0.975 | 0.492 | 0.908 | 0.787 | 162 |
| X + e[perm] | 1.000 | 1.000 | 0.875 | 0.533 | 0.863 | 0.533 | 117 |

### Discrepancies, stated as numbers

1. **Gaussian control, 393 vs my 384.4.** Difference **+8.6**, which is **0.90** of my measured
   across-draw sd (9.52). One of my 15 draws landed on **exactly 393**. A single draw of this control
   is a *sample*, not a constant: my draws span **366 – 402**, a 36-count range. Reporting it as
   "393" without a spread overstates its precision by roughly ±20 counts.
2. **Permuted-residual control, 95 vs my 117.3.** Difference **−22.3**, which is **2.77** of my
   across-draw sd (8.03), and **below every one of my 15 random draws** (minimum 104). This does not
   replicate under "a random permutation". It *is* reachable under other permutation conventions —
   I measured `roll by 1` = **64**, `roll by 256` = 113, `reverse` = 128, forced derangement = 139,
   legacy `RandomState(0)` = 116, legacy `RandomState(20260914)` = 114. So the hub's 95 most likely
   comes from a *structured* shuffle (a roll or a sort-order-dependent shift), not an i.i.d.
   permutation; on rows sorted by index, adjacent val rows share factor values, which makes
   `e[i−1]` a *more* damaging perturbation than a random one. **The stated number is not wrong so
   much as unlabelled**: the convention has to be named for the number to mean anything, and the
   spread (104–132 for random, 64–139 across conventions) has to be shown.
3. The permuted control is **not** at "the identical rel-err 0.3486" — permuting `e` breaks the
   per-row norm match. Measured mean per-row rel-err is **0.348548 – 0.348591** across draws vs
   **0.348604** for the round-trip: a difference of up to **5.6e-05**. Negligible in effect,
   but "identical" is true only of the Gaussian and the `−e` controls (where it is exact by
   construction to 7.6e-08).

### Where the hub's INTERPRETATION does not follow

**(i) The isotropic Gaussian is close to the weakest control available, so it cannot carry the
conclusion on its own.** In D = 2048, a random direction is essentially orthogonal to whatever
low-dimensional subspace the decoder+classifier actually reads; "a perturbation of the same norm is
survivable" is nearly guaranteed a priori and is not strong evidence that the store's residual is
*specially* structured. The `X − e` control (162) is the stronger one and the hub already has it:
it is the *same* norm **and** the same direction up to sign, and it still scores 16× the round-trip.

**(ii) The comparison that actually licenses the conclusion is a dose-response, and I had to run it
myself.** Decoding `X + α·e`:

| α | 0.125 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| rel-err | 0.0436 | 0.0872 | 0.1743 | 0.2615 | 0.3486 |
| exact, residual direction | 458 | **373** | 121 | 40 | **10** |
| exact, Gaussian at same norm | 488 | 485 | 468 | 431 | **382** |

The residual direction at **α = 0.25** (rel-err 0.087) is already as damaging as isotropic noise at
**α = 1.0** (rel-err 0.349). That is a clean ~**4×** damage-per-unit-norm factor, and it is the
quantitative statement the "structured residual" claim needs. It supports the hub's reading — but the
hub's published control does not establish it; the 393 alone is compatible with much weaker versions
of the claim.

**(iii) "Round-trip bound" is measured IN-SAMPLE for the store.** `claim3_overlap.py`:
**512 / 512** of the `val` rows are inside `store_train`, i.e. the store was fitted on every row its
round-trip is scored on. That direction is *conservative* for the hub (an out-of-sample round-trip can
only be worse, so 10/512 really is an upper bound) — but the 491-vs-10 headline then compares a
near-held-out decoder number against an in-sample store number, and the asymmetry should be stated.
For completeness: **12 / 512** val rows (2.34 %) are also in `bridge_train`, 0 are in `eval_a`,
`eval_b`, `fit`, or `split.test_rows`. I could **not** verify the *winner* decoder's own training
rows: `gen-decoder-lift/winner/winner_hashes.json` records the recipe, hashes and the encoder it was
fit against, but **no row list**.

**(iv) The round-trip is selectively, not uniformly, lossy — and "10/512" hides that.** The
round-trip keeps floor_hue at 0.998 and wall_hue at 0.982 while collapsing orientation to 0.174 and
scale to 0.238. A conjunction metric over 6 factors turns "2 of 6 factors destroyed" into "10/512",
which reads as total failure. Two extra facts I measured that bear on it: the round-trip is
genuinely row-specific (`rt` permuted across rows scores **0/512**, per-factor ≈ chance, 0.066–0.268),
and the reconstruction leaves the feature manifold — the cached features are **100 % non-negative**
(post-ReLU) while `rt` is only **87.0 %** non-negative. Neither fact is fatal to the hub's claim;
both change what the number is evidence *of*.

**(v) A control the hub does not have.** All three published controls perturb `X` *additively*, so
they all preserve `X`'s information and merely add error. The round-trip *replaces* `X` with a
codebook reconstruction (501 distinct codes over 512 rows, 5 code channels). "Additive noise of the
same norm is survivable" and "lossy quantisation of the same norm is not" is partly a comparison of
two different kinds of map, not purely of structure vs magnitude. The dose-response in (ii) is the
control that closes that gap, because it stays inside the residual's own direction.

---

## Files

| file | what |
|---|---|
| `claim1.py`, `claim1.json` | manifest re-read, per-arm mean ± sd (both ddof conventions) |
| `claim2.py`, `claim2.json` | split rebuilt, tuple counts and train-overlap |
| `claim3.py`, `claim3.json` | ceiling, round-trip, the three controls |
| `claim3_sweep.py`, `claim3_sweep.json` | 12 further draws of each random control |
| `claim3_extra.py`, `claim3_extra.json` | alternative permutation conventions, dose-response, `rt[perm]` |
| `claim3_overlap.py`, `claim3_overlap.json` | provenance of the 512 `val` rows |
| `merge.py`, `replication.json` | merged raw results with verdicts |
