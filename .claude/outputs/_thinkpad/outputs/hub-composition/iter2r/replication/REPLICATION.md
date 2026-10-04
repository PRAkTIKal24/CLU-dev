# INDEPENDENT REPLICATION — `hub/composition-2`, scored run 1, seed 0, `eval_b`

Charter §A.5 spoke. **Every number below was produced by code written in this directory
from the pinned `chlu` library only.** No file under `iter2r/dev/`, `iter2/dev/`,
`instrument/` or `mech/` was read, opened, imported or copied; `SCORED_B_seed0.json`,
`analysis_*.json` and `rivals_ledger.jsonl` were never opened. My four numbers were
written to `RESULTS_main.json` **before** the §COMPARE block of the brief was consulted.

Nothing under `chlu/` was edited. Nothing was deleted. No feature was encoded
(measured shortfall **0**). All writes are under this directory.

---

## 1. Environment, verbatim

```bash
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
export CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2
cd "$CHLU_ROOT" && . ./env_hub.sh
export PYTHONUTF8=1
export PY="$CHLU_ROOT/.venv/Scripts/python.exe"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WT=/c/Users/PratikJawahar/chlu/CHLU-hub-composition-2
export HUBENV=/c/Users/PratikJawahar/chlu/hub-composition-env
export PYTHONPATH="$WT:$CHLU_ROOT:$HUBENV/pylibs"
export CHLU_GEN_DATA_ROOT="$HUBENV/data/gen"
export REP="$CHLU_CLAUDE_ROOT/outputs/hub-composition/iter2r/replication"
mkdir -p "$REP"
cd "$REP"
$PY rep_context.py
$PY rep_score.py --tag main
```

`cd "$REP"` before every `$PY` invocation — the `sys.path[0]` trap. Verified in-process.

## 2. Provenance the loaders printed

| | |
|---|---|
| `chlu.__file__` | `C:\Users\PratikJawahar\chlu\CHLU-hub-composition-2\chlu\__init__.py` ✅ hub branch, not the bundle |
| Python | 3.11.16 (main, Sep 1 2026) MSC v.1944 64-bit AMD64 |
| numpy / jax | 2.4.1 / 0.9.0 |
| bridge dir | `…/.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0` |
| feature cache | **`n_rows = 480000`, `scope = "full"`**, `D = 2048`, `dtype = float16`, `feat_shape = (128,4,4)` |
| cache encoder hash | `98d326028316ae2cd6b960b26a922cd47fe8a2c9e73199fb1ba2927864a29c0d` (== bridge manifest) |
| **coverage gate** | need 32,840 rows (32,768 store-train ∪ 72 eval_b) → **shortfall = 0**. ⛔ nothing encoded. |
| decoder | `gen-decoder-lift/winner/decoder_v1_s0.pkl`, recipe `v1_capacity_schedule`, hash `01ef91f75eeaae7be8d83a220c9cc6629120345357da9033db22375c278ab196`, 1,516,233 params — hash **and** recipe asserted by `resolve_decoder` |
| external classifier | `gen-external-classifier/seed_0/external_classifier.pkl`, hash `c8741ac417d88ebf8ee82163ba9d7090ed66856aa9ee951e621791f08794669e`, 978,873 params, 1500 steps, gate bal-acc on real = 1.0 |
| bridge probe | `c03bfb8916d00932754df37973ff74de3ecec97a5d07ffc306414e466ba7960b` (== manifest) |
| rows | `store_train` 32,768 · `eval_a` 1,024 · **`eval_b` 72** |
| factors | `floor_hue, wall_hue, object_hue, scale, shape, orientation`, cards `[10,10,10,8,4,15]` |

## 3. One construction the PREREG does not pin, and how I resolved it

`chlu.core.gen_factored_read.mode_basis` **refuses a ragged store**:

```
ValueError: the factored head needs equal wells; got sizes
[36, 37, 38, 41, 42, 49, 52, 53, 56, 57, 58, 59, 60, 61, 62, 64, 65, 67, 72, 74, 79,
 80, 94, 96, 110, 113, 156]
```

Those are the sizes the *shipped* `gen_store.correlation_grouping` returns at K = 32 on
the seed-0 store-train pool. So the object in the PREREG **cannot** be built from the
shipped grouping directly. The only construction in the pinned library that is both the
correlation grouping and equal-welled is
`chlu.core.gen_partition.balanced_correlation_partition` — same `|corr|`, same ward
linkage, cut by dendrogram **leaf order** into equal blocks. I used that:

```
partition: balanced_correlation (dendrogram leaf order, equal blocks) linkage=ward
dead=61  well sizes [64]*32  in-block |corr|^2 = 0.1266
  (the RAGGED shipped grouping at K=32 scores in-block |corr|^2 = 0.1364 and is REFUSED)
```

61 dead channels + 1987 live is exactly the count that makes the dead-channel
back-fill land on 64/64/…/64 — strong evidence this is the intended construction.
The store modes are then **per-well Lloyd k-means, M = 26, seed 0**, seeded k-means++
off the pinned library's own RNG stream (`gen_store.init_store`'s
`default_rng(seed*2654435761 % 2**32)` consumed well-by-well), run to convergence
(max 100 iters; mean 65.8 iters to a zero-movement fixed point).

**⚠ The PREREG does not pin the Lloyd seeding rule, the iteration count, or the
equal-well partition.** §7 of this report shows those choices move R1 between 18 and 36.

## 4. Observed stdout — the scored run (`LOG_main.txt`)

```
[    4.0s] chlu.__file__ = C:\Users\PratikJawahar\chlu\CHLU-hub-composition-2\chlu\__init__.py
[    5.3s] context: store (32768, 2048) B (72, 2048)  K=32 M=26
[   10.1s] partition: balanced_correlation (dendrogram leaf order, equal blocks) linkage=ward dead=61 well sizes [64 x 32] in-block |corr|^2=0.1266
[   10.2s]   (for reference, the RAGGED shipped grouping at K=32 scores in-block |corr|^2=0.1364 and is REFUSED by mode_basis)
[   48.9s] k-means done: mean Lloyd iters 65.8
[   49.1s] store_bytes = 212992  (K=32 M=26 D=2048)
[   49.1s] pairs = ((0, 3), (2, 3))  ([('floor_hue', 'scale'), ('object_hue', 'scale')])
[   49.7s] head bytes = 256512  (n_add=58 n_pair=160 rank=16)
[   49.7s] OBJECT BYTES = 212992 + 256512 = 469504
[   50.0s] assert_request_only_factored: {"signature_ok": true,
           "coefficients_re_derived_from_requests_bit_identical": true,
           "read_is_pure_weighted_mode_sum": true,
           "max_abs_delta_vs_independent_sum": 2.86102294921875e-06,
           "no_feature_argument_on_the_path": true, "n_checked": 72}
[   50.0s] clu_linear features (72, 128, 4, 4)  mean|.|=0.4062
[   50.1s] store_removed_zero features  max|.| = 0 (expect 0)
[   51.6s] knn_proto bytes = 475136  (57 prototypes, D=2048)
[   51.6s]   decoder: winner recipe=v1_capacity_schedule hash=01ef91f75eeaae7b...
[   51.6s]   external classifier c8741ac417d88ebf... (978873 params)
[   51.6s] store was fitted on 32768 distinct factor tuples (32768 rows)
[   51.6s] === SCORING ===
[   53.4s]   [R1 clu_linear]          exact_match=0.347222  aggregate=0.8657  n_valid=25  n_novel=72  HEADLINE=25  72*exact_match=25.0000  identity_holds=True
[   54.7s]   [R2 knn_proto]           exact_match=0.166667  aggregate=0.8565  n_valid=12  n_novel=72  HEADLINE=12  72*exact_match=12.0000  identity_holds=True
[   56.0s]   [R3 store_removed_zero]  exact_match=0.000000  aggregate=0.0556  n_valid=0   n_novel=72  HEADLINE=0   72*exact_match=0.0000   identity_holds=True
[   57.4s]   [R4 decoder_ceiling]     exact_match=0.930556  aggregate=0.9861  n_valid=67  n_novel=72  HEADLINE=67  72*exact_match=67.0000  identity_holds=True
[   57.4s] paired Delta (R1-R2) = +0.1806 aggregate = +13.00 tuples; CI95 [+0.0538,+0.3073] = [+3.88,+22.12] tuples
```

## 5. MY FOUR NUMBERS (written to disk before comparing)

| # | arm | my HEADLINE | my `exact_match` | `72 × exact_match` | identity |
|---|---|---|---|---|---|
| **R1** | `clu_linear` (the object) | **25** | 0.347222 | 25.0000 | ✅ holds |
| **R2** | `knn_proto` | **12** | 0.166667 | 12.0000 | ✅ holds |
| **R3** | landscape-removed (modes zeroed) | **0** | 0.000000 | 0.0000 | ✅ holds |
| **R4** | decoder ceiling | **67** | 0.930556 | 67.0000 | ✅ holds |

Byte ledger, computed with `gen_store.store_bytes` / `head.byte_ledger()` /
`protos.byte_ledger()`:

| | bytes |
|---|---|
| store (K=32, M=26, D=2048) | **212,992** |
| factored head (n_add 58, n_pair 160, rank 16) | **256,512** |
| **object total** | **469,504** |
| `knn_proto` (57 prototypes + global mean, D=2048) | **475,136** |
| ratio | **0.98814** |

**Identity `HEADLINE == 72 × exact_match`: it HOLDS, for all four arms, in my hands.**
It holds for a checkable reason, which I verified rather than assumed: all 72 `eval_b`
requests are distinct (`all_requests_distinct = true`) and **all 72 are novel** —
`n_novel = 72` for every arm, and the store's 32,768 fit rows realise 32,768 *distinct*
tuples, none of which is an `eval_b` tuple. So "distinct valid novel tuples" degenerates
to "count of valid items". PREREG P10 is confirmed.

Per-factor accuracies (the same frozen external classifier for every arm):

| arm | floor_hue | wall_hue | object_hue | scale | shape | orientation |
|---|---|---|---|---|---|---|
| R1 `clu_linear` | 1.000 | 1.000 | 1.000 | **0.542** | 0.986 | **0.667** |
| R2 `knn_proto` | 1.000 | 1.000 | 1.000 | **0.167** | 0.972 | 1.000 |
| R3 zeroed | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.333 |
| R4 ceiling | 1.000 | 1.000 | 1.000 | 0.958 | 0.958 | 1.000 |

Diversity companion — distinct factor tuples the instrument *reads*, out of 72
(`gen_probe.predict_factors`, all arms): R1 **38**, R2 **56**, R3 **1** (one constant
render, as it must be), R4 **70**. ⚠ Worth noting beside R1: the object is both *less
accurate per item* than the ceiling and *less diverse in what it produces* than its own
rival — it collapses 72 requests onto 38 read tuples where `knn_proto` reaches 56.

## 6. VERDICTS (compared only after the above was on disk)

Hub reported: R1 = 42, R2 = 12, R3 = 0, R4 = 67; object 469,504 B; `knn_proto` 475,136 B.

| # | hub | mine | verdict |
|---|---|---|---|
| **R1** `clu_linear` | 42 | **25** | ⛔ **DIFFERS** — 17 tuples (0.236 in aggregate units) below the hub's |
| **R2** `knn_proto` | 12 | **12** | ✅ **CONFIRMED** (exact) |
| **R3** landscape-removed | 0 | **0** | ✅ **CONFIRMED** (exact; the zeroed read is the all-zeros feature, `max|.| = 0`) |
| **R4** decoder ceiling | 67 | **67** | ✅ **CONFIRMED** (exact) |
| object bytes | 469,504 | **469,504** | ✅ **CONFIRMED** (exact) |
| `knn_proto` bytes | 475,136 | **475,136** | ✅ **CONFIRMED** (exact) |
| identity `72 × exact_match` | asserted | **holds** | ✅ **CONFIRMED** |

Three of four numbers, both byte ledgers and the asserted identity replicate **exactly**.
The object's own headline does not.

## 7. Why R1 differs, stated plainly

Everything the two runs share replicated bit-exactly: the same cache, the same rows, the
same decoder, the same classifier, the same rival, the same control, the same ceiling,
the same byte ledgers down to the byte. The *only* thing that differs is the store the
object is read out of — and **the PREREG does not pin how that store is fitted.**

I re-ran my own pipeline over five reasonable readings of "K=32 wells from the shipped
correlation grouping, M=26 modes per well, k-means (Lloyd) per well, seed 0". R2, R3 and
R4 came back **12 / 0 / 67 in every single one**. R1 did not:

| variant | partition linkage | k-means++ RNG | Lloyd iters | **R1** | R2 | R3 | R4 |
|---|---|---|---|---|---|---|---|
| `main` (reported) | ward | library stream | to convergence (mean 65.8) | **25** | 12 | 0 | 67 |
| `lib25` | ward | library stream | 25 (gen_knn's default) | **23** | 12 | 0 | 67 |
| `lib0_initonly` | ward | library stream | 0 (k-means++ seed only) | **36** | 12 | 0 | 67 |
| `indep100` | ward | per-well independent | to convergence | **18** | 12 | 0 | 67 |
| `indep0_initonly` | ward | per-well independent | 0 | **20** | 12 | 0 | 67 |
| `avg100` | average | library stream | to convergence | **29** | 12 | 0 | 67 |

⚠ **A seventh variant did not finish and I report no number for it.** The pinned
library's *own* store fit — `gen_store.init_store` + `gen_store.train_store`, the
adam/soft-read objective `exp_gen_compose.run_seed` uses, 1200 steps — instead of Lloyd
(`$PY rep_score.py --store-fit shipped --tag shipped`). It was **killed by its own 2400 s
cap at step 720 / 1200**, so `RESULTS_shipped.json` does not exist. All that exists is the
training trace:

```
[   17.6s]     step   1/1200  loss=5.84438  tau=1.000
[  412.7s]     step 240/1200  loss=4.38532  tau=0.632
[  835.6s]     step 480/1200  loss=4.34700  tau=0.399
[ 1299.4s]     step 720/1200  loss=4.38617  tau=0.251
   <killed at 2400 s, before step 960>
```

⛔ I am not guessing its R1. ⭐ **It is the single highest-value thing for whoever picks
this up to run first** (~40 min at 4 threads): adam-trained modes are not cluster
centroids and span more of each 64-dim well than Lloyd centroids do, which is exactly the
mechanism in point 2 below, so it is the variant most likely to close the gap to 42. If
it lands at 42, the disagreement in §6 resolves into "the PREREG says Lloyd and the run
used `train_store`" — a wording defect, not a measurement one.

**R1 ranges over 18–36 across nuisance choices the PREREG leaves free.** That is a spread
of 18 tuples, i.e. 0.25 in aggregate units — an order of magnitude larger than R1's own
tie margin m = 0.020, and comparable to the ±13-tuple corner sd the PREREG itself warns
about. The hub's 42 sits above my whole range, but my whole range is wide enough that I
cannot call 42 *wrong*; I can only say **I could not reach it, and the quantity is not
pinned tightly enough for "42" to be a reproducible statement.**

Three specific things I think the hub got wrong, stated without softening:

1. ⛔ **The PREREG's object specification is under-determined and therefore not
   replicable.** "K = 32 wells from the shipped correlation grouping" cannot be executed
   as written — `mode_basis` *raises* on the shipped grouping at K = 32 (sizes 36…156).
   Some equal-well construction was substituted and the PREREG does not say which.
   "k-means (Lloyd) per well, seed 0" names neither the seeding RNG stream nor the
   iteration budget, and I measure those as worth ±9 tuples. A PREREG that pins ridge to
   one decimal place but leaves the store's fit unpinned has pinned the wrong things.
   ⭐ The fix is one line: ship the store checkpoint's `param_hash` in `PREREG-object.json`,
   the way `resolve_decoder` and `load_external_classifier` already do for the other two
   instruments. Both of *those* replicated exactly — that is not a coincidence.

2. ⚠ **The direction of the sensitivity is diagnostic and it is not flattering.** My
   *least* converged store (k-means++ seeds, zero Lloyd steps — i.e. 26 raw training rows
   per well, no fitting at all) scores the **highest** R1 (36), and every step of honest
   k-means fitting makes the headline *worse* (36 → 23 → 25 at convergence). The
   per-factor table says why: the object's losses are concentrated in `orientation`
   (0.667, against 1.000 for both `knn_proto` and the ceiling) and `scale` (0.542).
   Lloyd centroids collapse exactly the within-cluster variation that `orientation` lives
   in, and the factored head can only emit vectors inside the 26-dimensional mode span of
   each 64-dimensional well. So R1 is substantially a measurement of *how much
   unconverged noise is left in the codebook*, not of composition. That is a mechanism the
   hub should name before it speaks any number from this column.

3. ⚠ **The B-cell verdict is being over-read relative to its own declared precision.**
   Even taking the hub's 42 at face value, my paired statistic on the numbers I *did*
   reproduce gives Δ(R1−R2) = +0.181 aggregate = +13.0 tuples, CI95 [+3.9, +22.1] —
   nominally a WIN under R1, against the PREREG's predicted TIE (P4) and predicted
   Δ = +3 (P3). With the hub's 42 it would be +30 tuples, three times P3's point estimate
   and at the top of P3's interval [−24, +30]. Two runs of the *same* pre-registered
   object landing at +13 and +30 tuples, both nominally "wins", is the clearest possible
   demonstration that the single-cell B read is dominated by nuisance variance. The
   PREREG says this in §3 and then reports the number anyway. ⭐ I agree with §3 and
   disagree with the reporting.

**What is solid.** R2 = 12, R3 = 0, R4 = 67, object = 469,504 B, `knn_proto` = 475,136 B,
and the `72 × exact_match` identity are all **independently confirmed, exactly, on a
clean build from the pinned library**. The §C.2 falsifier does not fire: the
landscape-removed control is 0 and the zeroed read is provably the all-zeros feature.
`assert_request_only_factored` passes here too (signature OK, coefficients re-derived
from the requests bit-identically, read is a pure weighted mode sum, no feature argument
anywhere on the path, 72/72 checked) — falsifier 3 does not fire. Falsifier 2 does not
fire. Falsifier 4 does not fire (shortfall 0). **The only number I cannot stand behind is
the object's own headline, and the reason is that the object is not fully specified.**

## 8. Files written (all under `$REP`, nothing else touched)

| file | what |
|---|---|
| `rep_context.py` | stage 1 — instruments, rows, the no-encode coverage gate, cached features, grouping |
| `rep_score.py` | stage 2 — store, factored head, the four arms, the metric, the byte ledgers |
| `context.npz`, `context_meta.json` | the frozen context + full provenance |
| `RESULTS_main.json` | ⭐ **the reported run** |
| `RESULTS_{lib25,lib0_initonly,indep100,indep0_initonly,avg100}.json` | the §7 sensitivity variants |
| `RESULTS_shipped.json` | ⛔ **does not exist** — the `--store-fit shipped` variant was killed at step 720/1200 by its own 2400 s cap. See §7. |
| `LOG_main.txt` | raw stdout of the reported run |
| `REPLICATION.md` | this file |
