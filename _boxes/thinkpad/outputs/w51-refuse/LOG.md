# w5.1/refuse — LOG  (AUTO-HUB RETRY, kind R-A robustness)

**Parent:** `w5 hub/refuse` (finished 2026-09-20T00:36Z, 1 h 06 m of 10 h). Branch `w5.1/refuse` from
its tip `10b0734`. **Allowance 8 h from `START_UTC.txt` = 2026-09-20T00:39:18Z ⇒ deadline 08:39Z.**
⛔⛔ **SCORED BUDGET: ZERO.** The `refusal` target has 0 of 3 remaining. Nothing below scores a
held-out row: every number here is a **re-measurement of the SPREAD of an already-scored quantity**
under a wider instrument, plus split-index arithmetic. ⛔ No bar is set, no PREREG file is opened,
because a PREREG exists to bind a scored run and there is no scored run to bind.

**Dial (from the target file): the INSTRUMENT, not the object.** Nothing about the field store
changes. **Falsified if** the bridge-seed spread is wide enough to move a quantity the parent
reported with a ±, in which case the parent's ± figures are understated and this report says so.

## 0. Boot and premise checks — 2026-09-20T00:39Z–01:05Z

- **2026-09-20T00:39:18Z** — `START_UTC.txt` written FIRST, before any other work. No live launcher
  of mine exists (this is the first process of this hub). Read, in order and nothing else:
  `autonomous-hub-charter.md` → `w5-hub-common.md` → `w5.1-refuse.md`.
- **2026-09-20T00:40Z** — ⚠ **The parent's LOG is READING, not truth** (retry instruction). Every
  number I put beside mine is **re-derived from the parent's own artifacts on disk**
  (`/c/chlu/outputs/w5-refuse/runs/r1/seed_*/deletion.npz`), ⛔ never copied out of its prose.
- **2026-09-20T00:41Z** — ⭐⭐ **PREMISE 1 CONFIRMED — generation 2 is full scope at ALL THREE bridge
  seeds.** Read from each cache's own `feature_cache.json` under
  `C:\chlu\outputs\spoke-a-rivals\cache_generation2\shapes3d_recon\seed_{s}\`:

  | bridge seed | generation | n_rows | scope | `features_sha256` (12) | `encoder_param_hash` (12) |
  |---|---|---|---|---|---|
  | 0 | 2 | 480,000 | full | `ef6359a4cd54` | `98d326028316` |
  | 1 | 2 | 480,000 | full | `86b44d0f1e64` | `fc05e2c5f396` |
  | 2 | 2 | 480,000 | full | `3ab14e583c3f` | `154d498610648` |

  ⭐ Three **distinct** encoder parameter hashes at three **distinct** feature matrices, each
  1,966,080,128 B on disk. ⛔ The target file's kill-switch ("bridge seeds 1/2 turn out not to be
  full scope ⇒ stop and report") does **not** fire. The premise is alive.
- **2026-09-20T00:42Z** — ⭐ **PREMISE 2 CONFIRMED — the pin is real and it is a one-token pin.**
  In `chlu/experiments/exp_gen_refuse.py` the bridge seed enters at exactly four sites, all reading
  `PINS["bridge_seed"] = 0` regardless of the store seed:
  `build_field_arm` → `features_mmap(...)` (the ENCODER) and `feature_cache_manifest(...)`;
  `run_queries` → `Judge(...)`; `run_deletion` → `Judge(...)`.
  `Judge(s)` loads `decoder_v1_s{s}.pkl` (hash-asserted against `winner_hashes.json`) **and**
  `classifier_dir(s)/external_classifier.pkl` (hash-asserted against its manifest). So one token
  controls **encoder + decoder + classifier + cache** together — the parent's §5b is exactly right.
- **2026-09-20T00:43Z** — ⭐⭐ **PREMISE 3 CONFIRMED — after the change the two hubs' seed conventions
  become IDENTICAL, not merely similar.** Read out of `w5/delete`'s own committed source
  (`git show w5/delete:chlu/experiments/exp_gen_delete_row7.py`), not out of its LOG:

  | what seed *s* selects | `w5 hub/delete` | `w5 hub/refuse` (parent) | this hub (joint mode) |
  |---|---|---|---|
  | fit-pool draw | `pool_rng_seed + stride*s` | `pool_rng_seed + stride*s` | same |
  | atom selection | `field_spec(s)` | `field_spec(s)` | same |
  | encoder / feature cache | `features_mmap(s)` | ⛔ `features_mmap(0)` | ⭐ `features_mmap(s)` |
  | decoder | `Judge(s)` | ⛔ `Judge(0)` | ⭐ `Judge(s)` |
  | external classifier | `Judge(s)` | ⛔ `Judge(0)` | ⭐ `Judge(s)` |
  | channel→well partition | `build_partitions(out/"partitions")`, seed-0, shared | identical | identical |

  ⭐ `scored_pool` (delete) and `refuse_fit_pool` (parent) are the **same two lines of arithmetic**.
  ⛔ Note the last row honestly: **neither** hub varies the partition with the seed — it is built
  once from `PINS["seed"]`'s features and shared. So "joint" means five of six things, in both hubs,
  and that is what makes them comparable rather than what makes them complete.
- **2026-09-20T00:43Z** — ⛔ **The third kill-switch also does not fire.** Varying encoder/decoder/
  classifier here **does** require refitting the store (a different encoder means different atoms),
  but that is a **rebuild of a different store**, not a refit of a deployed one: `assert_no_refit`
  binds the *deletion*, which still happens inside a finished image. So the instrument can be varied;
  it costs three store builds, not a broken invariant.

## 1. The parent's baseline, RE-DERIVED from its artifacts — 01:10Z–01:25Z

- **2026-09-20T00:44Z** — ⭐ **The parent's fixed-instrument numbers, re-derived by me from
  `/c/chlu/outputs/w5-refuse/runs/r1/seed_{0,1,2}/deletion.npz` through its own frozen
  `score_deletion`, ⛔ not copied from its prose.** Its `deletion.npz` is the CLEAN sequential
  re-run; the quarantined `deletion.npz.overlap-suspect` files at seeds 0 and 1 are the
  two-writer artifacts it withdrew at its 03:00Z bullet and I do **not** read them as data.

  | quantity | per seed 0 / 1 / 2 | mean | max − min |
  |---|---|---|---|
  | EVICTION bytes changed | 17,569 / 20,113 / 20,218 | 19,300.0 | 2,649 |
  | EVICTION bytes OUTSIDE permitted ⭐ the claim | 0 / 0 / 0 | 0 | 0 |
  | EVICTION items evicted | 6 / 7 / 7 | 6.7 | 1 |
  | NOVEL post-HOLE `field_geom` | 2,284 / 2,490 / 2,178 | 2,317.3 | 312 |
  | NOVEL post-HOLE `pair_presence` | 1,428 / 1,584 / 1,362 | 1,458.0 | 222 |
  | NOVEL post-EVICT `field_geom` | 1,970 / 2,143 / 1,781 | 1,964.7 | 362 |

  ⭐ These reproduce the parent's report exactly (`2,317.3 ± 312`, `1,458.0 ± 222`,
  `1,964.7 ± 362`, `17,569 / 20,113 / 20,218`, `6/7/7`). Its report is arithmetically sound.
- **2026-09-20T00:44Z** — ⭐ **Its §5b is TRUE OF ITS ARTIFACTS, not just of its prose.** All three
  of its `deletion.json` files carry the *same* `features_sha256` (`ef6359a4cd54`), the *same*
  `encoder_param_hash` (`98d326028316`), the *same* `decoder_param_hash` (`01ef91f75eea`) and the
  *same* `classifier_param_hash` (`c8741ac417d8`). ⭐ One instrument, three stores, confirmed at
  the hash level. The thing this retry exists to widen is real and is exactly as described.
- **2026-09-20T00:44Z** — ⚠⚠ **A CONVENTION FINDING that changes how the whole comparison must be
  read, and it runs IN THE PARENT'S FAVOUR.** Its `±` is **not** a half-range and **not** an sd: in
  its own `exp_gen_refuse_report._agg` it is `spread = float(a.max() - a.min())` — the **full
  range** — printed as `mean ± spread`. So `2,317.3 ± 312` denotes an interval
  `[2,005.3, 2,629.3]` that is **about twice as wide as the three numbers it saw**
  (`[2,178, 2,490]`). ⛔ The retry's framing assumed "narrower than the truth"; on the notation
  the parent's stated intervals are **conservative**, not understated. ⭐ I adopt its statistic
  verbatim so the two columns are like-for-like, and I print **two** verdicts per row — against
  the published `mean ± spread` and against the observed `[min, max]` — because they can disagree
  and the disagreement is the finding, not a nuisance.
- **2026-09-20T00:44Z** — ⛔ **What I am NOT doing.** No held-out row is scored; the budget is 0/3
  and stays 0/3. Both columns re-read artifacts of rows the parent already paid for. I set no bar,
  so there is nothing a PREREG could bind, and opening one would be theatre.

## 2. The joint-instrument run — 01:25Z onward

- **2026-09-20T00:45Z** — ⭐ **The change is ONE token, and `pinned` stays the default so nothing
  already on disk can be reinterpreted by accident.** `exp_gen_refuse` gains
  `BRIDGE_MODE` / `bridge_seed_for(seed)` and a `--bridge-mode {pinned,joint}` flag.
  `pinned` returns 0 (the parent, bit-for-bit); `joint` returns the store seed. The four call
  sites — the encoder (`features_mmap`), the cache manifest, and the two `Judge(...)`
  constructions — now all read `bridge_seed_for(seed)`. ⛔ No threshold, no rule, no pin, no
  metric and nothing about the store moved.
- **2026-09-20T00:45Z** — ⭐⭐ **`--stage seed0check` PASSES: at store seed 0 the two modes coincide
  on ALL FIVE hashes**, asserted in code rather than argued (`seed0_modes_coincide.json`):
  `bridge_seed` 0=0 · `features_sha256` `ef6359a4cd54` · `store_sha256` `617e322125fb…` ·
  `decoder_param_hash` `01ef91f75eea` · `classifier_param_hash` `c8741ac417d8`; deployed
  524,288 B, image 2,054,866 B in both. ⛔ Had this failed, every seed-1/2 number below would be
  measuring something other than the instrument and none of it could be believed.
- **2026-09-20T00:46Z** — ⚠ **A self-inflicted fault, logged because it failed in the safe
  direction.** `seed0check` exited 1 *after* writing its JSON: Windows stdout is cp1252 and its
  success line contains `⭐`. The measurement had completed and the assertion had passed; only
  the congratulatory `print` died. ⛔ An exit code that reports a log line as a failed measurement
  is exactly the fault `w5 hub/delete` recorded at its own BAR 4. Fixed at the driver
  (`PYTHONIOENCODING=utf-8`) rather than by deleting the character. ⛔ No process of any other
  agent was touched: my python child was identified by command line as PID 28900 and my driver as
  27905; PIDs 32052 / 3756 (`w5-delete`) and 31752 (`w5-compose-field`) were left alone, and the
  fits are resumable so nothing needed killing at all.
- **2026-09-20T00:46Z** — ⭐⭐ **A FALSIFIABLE PREDICTION, written BEFORE my seeds 1 and 2 finish.**
  `w5 hub/delete` draws its pool with the same two lines (`pool_rng_seed + stride*s`), takes atoms
  with the same `field_spec(s)`, shares the same seed-0 partition, and calls `features_mmap(s)` —
  which is what `joint` now does. So if "comparable" means anything, my joint eviction byte counts
  must come out **at its numbers, not at my parent's**:

  | seed | ⭐ predicted (= `w5 hub/delete`, `run1/seed_*/bytes.json`) | the parent's fixed instrument |
  |---|---|---|
  | 0 | 17,569 B / 0 outside / 6 items | 17,569 B / 0 / 6 ⭐ these agree already |
  | 1 | **16,764 B / 0 outside / 6 items** | 20,113 B / 0 / 7 |
  | 2 | **19,114 B / 0 outside / 7 items** | 20,218 B / 0 / 7 |

  Negative control, same prediction: **5,735 / 5,726 / 5,783** (delete's), ⛔ not 5,735 / — / 5,762
  (the parent's). ⚠ If the joint numbers land on neither set, "comparable" is false and I say so.

## 3. ⚠⚠ A CONFOUND FOUND BEFORE IT CONTAMINATED THE TABLE — 01:50Z

- **2026-09-20T00:47Z** — ⚠⚠ **My seed-0 store is NOT bit-identical to the parent's, and I found
  it by checking rather than by assuming.** Direct hash of the two `arm.pkl` images:
  parent seed 0 `sha256[:32] = 697d60dfa3e4b70c6822e04af1abeb3b` — ⭐ which is *exactly* the hash
  `w5 hub/delete` published for its own seed-0 image, so **the parent's and delete's seed-0 stores
  really are one image, verified by me from the pickles, not from either LOG**. ⛔ **Mine is
  `617e322125fb6754…`.** Both are 2,054,866 B.
- **2026-09-20T00:47Z** — ⭐ **The disagreement is confined and I measured exactly where.** Segment
  by segment, my build against the parent's, at store seed 0 in PINNED mode (i.e. the parent's own
  configuration):

  | segment | elements differing | max abs | max rel |
  |---|---|---|---|
  | `atoms` (the landscape, 524,288 B) | **0 / 131,072** | 0 | 0 |
  | `atom_row`, `atom_basin`, `atom_alive`, `atom_life` | **0** | 0 | 0 |
  | `basin_sigma`, `basin_gap`, `basin_n`, `well_sigma` | **0** | 0 | 0 |
  | `term_mu`, `term_n`, `global_mu` | **0** | 0 | 0 |
  | `coupling`, `coupling_w`, `coupling_dp_fg`, `coupling_dp_gf`, `coupling_hole` | **0** | 0 | 0 |
  | ⚠ `coupling_u` | 30 / 10,576 | 1.49e-08 | 8.81e-04 |
  | ⚠ `coupling_v` | 1,441 / 245,760 | 9.97e-08 | 1.13e-04 |

  ⭐ Everything the refusal rule and the deletion verbs touch — atoms, basins, the term map, the
  coupling weights and the hole flags — is **bit-identical**. The disagreement is float32 ULP
  noise in the two **low-rank coupling factors** only, on 0.6 % of their elements. The partitions
  are bit-identical too (all 11 arrays and the JSON), so it is not a partition drift.
- **2026-09-20T00:47Z** — ⛔⛔ **This is a CONFOUND for the table I was about to print, and I am not
  printing it.** The eviction byte count cannot see `coupling_u/v` (the verb does not touch them),
  but the **novel-tuple count can**: it goes through `field_read` → decoder → classifier, so a
  1e-7 perturbation can flip a tuple. A column of mine against a column of the parent's would
  confound **instrument** with **my build environment**, and instrument is the only thing this
  retry is allowed to measure.
- **2026-09-20T00:47Z** — ⭐⭐ **THE FIX, and it makes the retry stronger rather than weaker: I run
  BOTH arms myself.** `--bridge-mode pinned` at seeds 0/1/2 into `runs/fixed`, and `joint` into
  `runs/joint`, in one environment with one thread setting. The headline table is then **my fixed
  vs my joint**, where the ONLY thing that differs is which bridge seed the store seed selects —
  which is the dial, exactly. ⭐ The parent's own three numbers are then reported as a **third
  column**, and my-fixed-vs-parent-fixed becomes an independent **reproducibility** statement
  about the pinned arm rather than a hidden term inside the spread. ⛔ Costs three more deletion
  stages and no scored run.

- **2026-09-20T00:49Z** — ⚠ **Correction to this log's own timestamps, made as soon as I noticed.**
  Every bullet above between 00:40Z and 00:47Z was first written with a **local** (UTC+2) clock
  reading and an `Z` suffix — `01:50Z` where the true UTC was `00:47Z`. ⛔ A provenance field that
  is wrong is worse than one that is absent, so I corrected the stamps in place rather than
  leaving two hours of phantom elapsed time on the record, and I am recording the correction here
  rather than making it silently. ⭐ The ORDER of the bullets and every number in them is
  unchanged. Real elapsed at this bullet: **10 minutes** of the 8-hour allowance
  (`START_UTC = 00:39:18Z`, deadline **08:39Z**).

## 4. ⭐⭐ The ULP disagreement, DIAGNOSED not shrugged at — 00:48Z

- **2026-09-20T00:48Z** — ⭐⭐ **The cause is the BLAS THREAD COUNT, and I proved it by rebuilding.**
  The same store, same seed, same pinned mode, same code, rebuilt at `OMP_NUM_THREADS=3`:

  | build | store image `sha256[:32]` |
  |---|---|
  | mine at **1 thread** | `617e322125fb6754da93b1107c96ed73` |
  | ⭐ mine at **3 threads** | **`697d60dfa3e4b70c6822e04af1abeb3b`** |
  | the parent (`w5 hub/refuse`) | **`697d60dfa3e4b70c6822e04af1abeb3b`** |
  | `w5 hub/delete` (its own published seed-0 hash) | **`697d60dfa3e4b70c6822e04af1abeb3b`** |

  ⭐ **At 3 threads my build is BIT-IDENTICAL to both hubs' — a fourth independent build of the
  same 2,054,866-byte image.** The mechanism is named, not guessed: `gen_field.build_field` takes
  a LAPACK `np.linalg.svd` for the rank-8 coupling correction, and LAPACK's reduction order
  depends on the thread count. ⭐ The code already canonicalises the SVD's **signs** with a comment
  saying that without it "a REBUILD of the identical store differs by a few hundred bytes"; ⚠ what
  it does not pin is the **magnitudes**, which move by ~1e-7 with threading. ⛔ It touches
  `coupling_u`/`coupling_v` and nothing else.
- **2026-09-20T00:49Z** — ⭐ **Adopted as a number-bearing setting, not a speed knob.** The driver
  now exports `OMP_NUM_THREADS=3` and runs the stages **one at a time** (3 threads × 1 process =
  the hub's 3-thread allowance, and a 3-at-a-time launch would have been 9). ⛔ The 1-thread store
  artifacts I had already written were **deleted**, not reinterpreted. ⭐ Both arms are then
  bit-comparable with the parent AND with `w5 hub/delete`, which is strictly better than the
  isolated-arms fix alone — I keep the isolated arms as well, so the headline contrast has no
  environment term in it at all.
- **2026-09-20T00:49Z** — ⚠ **For the programme, not just for me:** a store image's reproducibility
  is **thread-count-dependent** at 1e-7 in `coupling_u`/`coupling_v`. Any hub quoting a store
  image hash should quote its BLAS thread count beside it, and any replication that rebuilds on a
  different thread setting will see a hash mismatch that is **not** a real disagreement. ⭐ The
  landscape itself (`atoms`, all 524,288 B), the basins, the term map, the coupling weights and
  the hole flags are bit-identical at either setting, so ⛔ no byte claim of either hub is affected.

## 5. The measurement design, fixed at 00:52Z and not moved after

- **2026-09-20T00:52Z** — ⭐ **Three columns, and the reason each exists.**
  1. **parent, fixed** — the parent's own `runs/r1` tree, scored by its own frozen
     `score_deletion`. ⭐ This is what the target file means by "your parent's fixed-instrument
     spread", so it goes in the table literally rather than as my reconstruction of it.
  2. **mine, FIXED** — the parent's configuration (`--bridge-mode pinned`), rebuilt by me at
     3 threads. ⭐ `parent` vs `mine-fixed` is a **reproducibility** statement.
  3. ⭐ **mine, JOINT** — `--bridge-mode joint`, the wider instrument. ⭐ `mine-fixed` vs
     `mine-joint` is **the dial, clean**: same machine, same threads, same code, same
     partition; the only thing that differs is which bridge seed the store seed selects.
  ⛔ The headline contrast is (2) vs (3), ⛔ not (1) vs (3), because (1) vs (3) would carry a
  build-environment term inside it and the dial is only allowed to carry the instrument.
- **2026-09-20T00:52Z** — ⭐ **The verdict rule, written before the numbers exist.** Each row
  reports whether the joint mean falls **inside the parent's published `mean ± spread`** and,
  separately, whether it falls **inside the parent's observed `[min, max]`**. ⚠ Two verdicts
  because the parent's `±` is the full range, so the two can disagree — and a quantity that is
  inside the published interval but outside the three numbers the parent actually saw was
  covered **only by the generous notation**. That distinction is the finding, not a nuisance.
- **2026-09-20T00:52Z** — ⛔ **Saturated quantities are NOT chased, and that is a decision, not an
  omission.** Refusal 1.0000, `bitmap8` Δreject 0.0000, the 1-byte hole, `0` bytes outside — each
  is bounded above or below by construction and a wider instrument cannot move it. They are
  printed once as **fixed points**, so "did not move" is on the record as measured rather than
  assumed. ⭐ Chasing them would be the drift the retry charter warns about.

- **2026-09-20T00:53Z** — ⭐ **`--stage seed0check` PASSES AGAIN at 3 threads, now at the parent's
  own hash.** `pinned` and `joint` both build store seed 0 to `697d60dfa3e4b70c…`, identical on
  all five hashes (`seed0_modes_coincide.json`, `all_identical: true`). ⭐ So the mode switch is a
  pure reparametrisation, asserted in code, and my environment now reproduces the parent exactly.
- **2026-09-20T00:54Z** — ⭐ **The partition is thread-INDEPENDENT, checked rather than hoped.**
  The `partitions.npz` I copied into both arms (built at 1 thread) and both freshly built at 3
  threads are **all bit-identical to the parent's**, on every one of the 11 arrays and on the JSON.
  ⛔ So the partition is not a second uncontrolled term: the ULP sensitivity is confined to
  `build_field`'s SVD, exactly where I located it, and not in the label path.

## 6. ⭐⭐ A stronger design than the one I was asked for — the 3×3 grid, 00:58Z

- **2026-09-20T00:58Z** — ⭐⭐ **`pinned` vs `joint` answers "is the spread wider?" but ⛔ cannot
  answer "wider because of what", and the second question is the one the programme actually
  needs.** The joint spread mixes the store draw with the instrument in one number. So I am
  adding a third setting, `--bridge-seed N`, which fixes the instrument at one bridge seed
  *independently* of the store seed and opens the full **3×3 grid, (store seed) × (bridge seed)**:

  | read | what it isolates | who has ever measured it |
  |---|---|---|
  | down column `bridge = 0` | the **store** effect at a fixed instrument | ⭐ the parent — this IS its spread |
  | across a row at fixed store | ⭐⭐ the **instrument** effect alone | ⛔ **nobody** |
  | the diagonal | store + instrument together | `joint`, and `w5 hub/delete` |

  ⛔ Still zero scored runs: every cell re-reads rows the parent already paid for, on the same
  frozen scorer. ⭐ It costs 6 more fits and 6 more deletion stages (`bridge=1` and `bridge=2`
  columns; the `bridge=0` column is the pinned arm I am already running).
- **2026-09-20T00:58Z** — ⚠ **Declared before it runs, so it cannot become a post-hoc story:** if
  the row-wise (instrument) variation is **small** relative to the column-wise (store) variation,
  then the parent's ± was narrow for a *harmless* reason and its numbers stand. If the row-wise
  variation is **comparable or larger**, the parent's ± genuinely understates the truth and I say
  so plainly. ⛔ Either way the answer is the result; neither is a failure.

## 7. ⭐⭐⭐ THE COMPARABILITY QUESTION IS SETTLED AT THE BYTE LEVEL — 00:57Z

- **2026-09-20T00:57Z** — ⭐⭐⭐ **All six stores built. The prediction I wrote at 00:46Z held, and
  it held in the strongest possible form: not "the numbers agree" but "it is the same image".**
  Store image `sha256[:32]` of every `arm.pkl`, hashed by me from the pickles:

  | store seed | bridge seed | mine, PINNED | mine, JOINT | the PARENT | ⭐ `w5 hub/delete` (published) |
  |---|---|---|---|---|---|
  | 0 | 0 / 0 | `697d60df…abeb3b` | `697d60df…abeb3b` | `697d60df…abeb3b` | `697d60df…abeb3b` |
  | 1 | 0 / **1** | `d0551f14…e8dc5c` | ⭐ **`1719deda…6c5a02ac`** | `d0551f14…e8dc5c` | **`1719deda…6c5a02ac`** |
  | 2 | 0 / **2** | `3d9dc66b…ec2b21` | ⭐ **`db12e26a…f474392`** | `3d9dc66b…ec2b21` | **`db12e26a…f474392`** |

  ⭐⭐ **My JOINT arm is BIT-IDENTICAL to `w5 hub/delete`'s store on all three seeds.** ⭐⭐ **My
  PINNED arm is BIT-IDENTICAL to the parent's on all three seeds.** ⛔ And the two columns differ
  at seeds 1 and 2, which is precisely the disagreement the two hubs found at 00:34Z and could
  not resolve without this run.
- **2026-09-20T00:57Z** — ⭐⭐ **ACCEPTANCE ITEM 2, ANSWERED, and the answer is mechanical rather
  than statistical.** ⭐ **YES — after `--bridge-mode joint` the parent's numbers ARE comparable to
  `w5 hub/delete`'s**, because at every seed the two hubs are *the same 2,054,866-byte image*,
  reached by three independent code paths. ⛔ **Under the parent's original `pinned` convention
  they were NOT**, and the reason is now exhibited rather than argued: at seeds 1 and 2 the two
  hubs were building different stores from different encoders. ⛔ Neither hub's numbers were
  wrong; they were answers to different questions.
- **2026-09-20T00:57Z** — ⚠ **The honest limit on that "yes".** It is a statement about the STORE
  and the INSTRUMENT, which are now identical. It is ⛔ **not** a claim that the two hubs' *metrics*
  are interchangeable — they score different rows (`refuse` scores refusal pairs and novelty on
  109 corners; `delete` scores its own row-7 bars) — so a shared row still needs a shared metric.
  ⭐ What is now legitimate is exactly what was illegitimate before: putting the two hubs' **byte
  counts** for the same operation on the same store in one table.
- **2026-09-20T00:57Z** — ⭐ **A by-product worth recording: the pinned arm REPLICATES the parent
  exactly.** Three stores, three bit-identical images, on a different agent's process tree with
  the thread count restored. ⛔ That is not the §A.5 independent replication (this is the parent's
  own code, and the parent already had a clean-checkout spoke); it is the environment control the
  dial contrast needs, and it is reported as that and nothing more.

- **2026-09-20T00:58Z** — Deletion stages running sequentially at 3 threads; the judge pass is the
  cost (5,688 corner + 512 accept-leg requests, re-judged after the eviction because the eviction
  DOES change the render). ⭐ The `--stage deletion` half-checkpoints (`_ck_hole.pkl` /
  `_ck_evict.pkl`) the parent added after being killed mid-stage are inherited unchanged, so a
  kill costs at most one half. ⛔ The 3×3 grid is CHAINED behind the main driver rather than run
  beside it, so the box never sees more than one of my python processes at a time.

- **2026-09-20T01:09Z** — ⭐⭐ **The PINNED arm reproduces the parent at the ARRAY level, not just
  in summary.** Seeds 0 and 1 complete (seed 2 running). Compared element-wise from the two
  `deletion.npz` files:

  | seed | evict bytes mine / parent | items | outside | hole bytes | ⭐ judged corner tuples identical | ⭐ post-eviction tuples identical |
  |---|---|---|---|---|---|---|
  | 0 | 17,569 / 17,569 | 6 / 6 | 0 / 0 | 1 / 1 | **True** | **True** |
  | 1 | 20,113 / 20,113 | 7 / 7 | 0 / 0 | 1 / 1 | **True** | **True** |

  ⭐ `corner_tuples` is the full (5,688 × 6) matrix the judge produced, and it is **bit-identical**
  to the parent's — so the environment control is exact, and any difference the JOINT arm shows
  is the instrument and nothing else. ⛔ This is the control, ⛔ not a result.

- **2026-09-20T01:26Z** — ⭐⭐ **THE ENVIRONMENT CONTROL IS PERFECT — my PINNED arm is the parent,
  on every number I intend to report.** All three seeds complete:
  **21/21 per-seed byte cells identical** (evict bytes, bytes outside, items evicted, hole bytes,
  hole outside, image bytes, negative control) and **8/8 novel-tuple triples identical**:

  | quantity | mine, fixed (s0/s1/s2) | the parent (s0/s1/s2) |
  |---|---|---|
  | NOVEL post-HOLE `field_geom` / `bitmap8` / `pair_deletion` | 2,284 / 2,490 / 2,178 | 2,284 / 2,490 / 2,178 |
  | NOVEL post-HOLE `pair_presence` | 1,428 / 1,584 / 1,362 | 1,428 / 1,584 / 1,362 |
  | NOVEL post-EVICT `field_geom` / `bitmap8` / `pair_deletion` | 1,970 / 2,143 / 1,781 | 1,970 / 2,143 / 1,781 |
  | NOVEL post-EVICT `pair_presence` | 1,458 / 1,582 / 1,260 | 1,458 / 1,582 / 1,260 |

  ⭐ So the dial contrast has **no environment term left in it at all**: whatever the JOINT arm
  shows against this column is the instrument, and only the instrument. ⛔ Reported as the control
  it is — a hub running its own predecessor's code on its own box is ⛔ **not** a §A.5 independent
  replication and I do not call it one.

- **2026-09-20T01:28Z** — ⭐⭐⭐ **THE 00:46Z PREDICTION HIT EXACTLY, CELL FOR CELL, AT SEED 1.**
  I wrote it down before the run; here it is measured:

  | seed 1 | parent / mine-FIXED (bridge 0) | ⭐ mine-JOINT (bridge 1) | ⭐ `w5 hub/delete` published | predicted? |
  |---|---|---|---|---|
  | evict bytes changed | 20,113 | **16,764** | **16,764** | ⭐ **YES** |
  | evict bytes OUTSIDE | 0 | **0** | **0** | ⭐ YES |
  | items evicted | 7 | **6** | **6** | ⭐ **YES** |
  | negative control bytes | 5,730 | **5,726** | **5,726** | ⭐ **YES** |
  | decoder param hash | `01ef91f75eea` | `001011bc603a` | (its own seed-1 decoder) | — |

  ⭐⭐ **A 3,349-byte move in the eviction count and a whole evicted item, from changing NOTHING
  but which frozen encoder the store seed reads.** ⛔ That is the instrument, not the object, and
  it is the exact quantity the parent reported with a ±.

## 8. ⭐⭐⭐ THE RESULT — 01:36Z

- **2026-09-20T01:36Z** — ⭐ **All six main cells complete. Every SATURATED quantity is identical
  across all three columns**, exactly as the parent predicted and as I pre-declared I would not
  chase: hole **1 byte**, **0** outside, `field_geom` reject deleted conjunction **1.0000**,
  accept never-held conjunction **1.0000**, `bitmap8` Δreject **0.0000**, `pair_presence` accept
  never-held **0.0000**, evict reject **1.0000**, accept survivors **1.0000**, rendered features
  bit-identical **True/True/True** — ⭐ all at `± 0.0000` in `parent-fixed`, `mine-fixed` AND
  `mine-joint`. ⛔ The wider instrument cannot move a bounded quantity, and now that is measured
  in nine places rather than assumed.
- **2026-09-20T01:36Z** — ⚠⚠ **THE RANGE STATISTIC HIDES THE EFFECT, and reporting only it would
  have been a mistake I nearly made.** Compare the two spreads and several quantities look calmer
  under the wider instrument (`EVICTION bytes changed` ×0.89, `NOVEL post-EVICTION` ×0.51). ⛔ That
  is an artifact: a range over **three** cells can stay the same width while **every cell moves**.
  ⭐ The right test is **PAIRED** — same store seed, two instruments — and it is available to me
  precisely because `mine-fixed` and `mine-joint` differ in nothing else.
- **2026-09-20T01:37Z** — ⭐⭐⭐ **THE PAIRED TEST, AND IT FALSIFIES THE PARENT'S ± FIGURES.** Each
  row: the same three store seeds, read by the pinned instrument and by the joint one.

  | quantity | fixed (s0/s1/s2) | ⭐ joint (s0/s1/s2) | Δ per seed | **max abs Δ** | the parent's WHOLE published ± | ⛔ ratio |
  |---|---|---|---|---|---|---|
  | EVICTION bytes changed | 17,569 / 20,113 / 20,218 | 17,569 / **16,764** / **19,114** | 0 / −3,349 / −1,104 | **3,349** | 2,649 | ⛔ **1.26×** |
  | EVICTION items evicted | 6 / 7 / 7 | 6 / **6** / 7 | 0 / −1 / 0 | **1** | 1 | 1.00× |
  | NEGATIVE CONTROL bytes | 5,735 / 5,730 / 5,694 | 5,735 / **5,726** / **5,783** | 0 / −4 / +89 | **89** | 41 | ⛔ **2.17×** |
  | NOVEL post-HOLE `field_geom` / `bitmap8` / `pair_deletion` | 2,284 / 2,490 / 2,178 | 2,284 / **2,030** / **2,570** | 0 / −460 / +392 | **460** | 312 | ⛔ **1.47×** |
  | NOVEL post-HOLE `pair_presence` | 1,428 / 1,584 / 1,362 | 1,428 / **1,330** / **1,712** | 0 / −254 / +350 | **350** | 222 | ⛔ **1.58×** |
  | NOVEL post-EVICTION `field_geom` / `bitmap8` / `pair_deletion` | 1,970 / 2,143 / 1,781 | 1,970 / **1,787** / **1,876** | 0 / −356 / +95 | **356** | 362 | 0.98× |
  | NOVEL post-EVICTION `pair_presence` | 1,458 / 1,582 / 1,260 | 1,458 / **1,386** / **1,438** | 0 / −196 / +178 | **196** | 322 | 0.61× |

  ⭐ Seed 0 is **0 in every row** — the two instruments coincide there by construction, which is
  the internal control and it fires correctly.
- **2026-09-20T01:37Z** — ⛔⛔ **SO I SAY IT PLAINLY, AS §5 OF MY OWN TARGET FILE REQUIRES: THE
  PARENT'S ± FIGURES ARE UNDERSTATED.** On four of the seven quantities, swapping the instrument
  at a **fixed** store seed moves the number by **more than the parent's entire published ±** —
  up to **2.17×** on the negative control and **1.47×** on its headline novel-tuple count. ⛔ Its
  `±` never contained the instrument, and the instrument is the larger term. ⭐ Per the retry
  charter this is **the result, not a failure**, and per the dial declaration the retry is
  **FALSIFIED in its stated direction** — which is the outcome it was built to be able to reach.
- **2026-09-20T01:37Z** — ⭐⭐ **BUT — and this is the sentence that matters most — NOTHING THE
  PARENT CLAIMED MOVED.** Every one of its claims is a **bounded** quantity (1.0000, 0.0000, 1
  byte, 0 outside, bit-identical renders) and every one is **unchanged at ± 0.0000** under the
  wider instrument. ⭐ What moved is exactly what both hubs had already agreed to call
  **provenance**: byte *counts* and novel-tuple *counts*. ⛔ The parent's own 00:34Z bullet —
  "the invariant is not the count; the invariant is 0 outside" — is vindicated by the very
  measurement that falsifies its error bars.

- **2026-09-20T01:39Z** — ⚠ **A rule breach of my own, found and fixed in one commit.** `MY_PIDS.txt`
  landed in the **worktree root** and got committed: in `cd <outputs> && nohup ... & echo ... |
  tee -a MY_PIDS.txt` the `&` backgrounds the whole `cd`-chain, so the `tee` ran in the original
  working directory. ⛔ `CHLU_HUB_OUTPUTS` is the only place this hub writes. Removed from the
  index and from disk; the real record was and remains
  `/c/chlu/outputs/w51-refuse/MY_PIDS.txt`, and the missing line is restored there. ⛔ No code
  affected. Logged rather than quietly dropped.

## 9. ⭐⭐⭐ THE 3×3 GRID — and a CORRECTION to my own 01:37Z verdict, 02:20Z

- **2026-09-20T02:15Z** — ⭐ **All nine grid cells complete, and the grid is internally consistent:
  36/36 diagonal cells identical** between the grid's `(s, s)` cells (`--bridge-seed s --seed s`)
  and the separately-run `--bridge-mode joint` arm — two code paths, same twelve quantities, same
  three seeds, no disagreement. ⛔ No process of mine was running when I collected; the four
  python processes on the box at that moment were `w5-delete`'s and `w5-compose-field`'s and I
  touched none of them.
- **2026-09-20T02:16Z** — ⭐⭐ **THE GRID SEPARATES THE TWO EFFECTS, AND THE INSTRUMENT IS THE
  BIGGER ONE.** Row range = instrument at a fixed store; column `b=0` range = the parent's own
  store spread.

  | quantity | ⭐ max INSTRUMENT range (a row) | the parent's STORE range (`b=0`) | ratio |
  |---|---|---|---|
  | NOVEL post-EVICTION `field_geom` / `bitmap8` / `pair_deletion` | **1,009** | 362 | ⛔ **2.79×** |
  | NOVEL post-EVICTION `pair_presence` | **782** | 322 | ⛔ **2.43×** |
  | NOVEL post-HOLE `pair_presence` | **496** | 222 | ⛔ **2.23×** |
  | NEGATIVE CONTROL bytes | **89** | 41 | ⛔ **2.17×** |
  | NOVEL post-HOLE `pair_deletion` | **644** | 312 | ⛔ **2.06×** |
  | NOVEL post-HOLE `field_geom` | **603** | 312 | ⛔ **1.93×** |
  | EVICTION bytes changed | **3,349** | 2,649 | ⛔ **1.26×** |
  | EVICTION items evicted | 1 | 1 | 1.00× |
  | ⭐ **EVICTION bytes OUTSIDE permitted** | **0** | **0** | ⭐ **bounded — 0 in every direction** |

  ⚠ Note what the columns also say: the parent's store range at `b=0` for `EVICTION bytes
  changed` is **2,649**, but at `b=1` it is **273** and at `b=2` only **96**. ⛔ So even the
  *store* effect it reported is not a stable property — it is itself a function of which
  instrument you hold fixed. ⭐ On 7 of 9 varying quantities the instrument is the larger term.
- **2026-09-20T02:20Z** — ⚠⚠⚠ **CORRECTION TO MY OWN 01:37Z BULLET, and it is the most important
  thing this hub found.** I wrote "**NOTHING THE PARENT CLAIMED MOVED**". ⛔ **That is true of the
  DIAGONAL and FALSE of the full grid, and I had only measured the diagonal when I wrote it.**
  The parent's claim has an **accept half** — "still answering **1.0000** of the 2,160 requests
  naming a never-held conjunction". Audited across all nine cells:

  | store seed \ bridge seed | b=0 | b=1 | b=2 |
  |---|---|---|---|
  | store 0 | 1.0000 | 1.0000 | 1.0000 |
  | store 1 | 1.0000 | 1.0000 | ⛔ **0.8333** |
  | store 2 | 1.0000 | 1.0000 | 1.0000 |

  ⛔⛔ **The accept leg FAILS in 1 of 9 cells.** ⭐ A refusal rule gets the *reject* half free by
  refusing everything, so the accept half is the half where a claim can die — and it dies here.
- **2026-09-20T02:22Z** — ⭐⭐ **THE MECHANISM, NAMED EXACTLY — it is `empty_basin`, and it is an
  UNDER-POPULATED BASIN, not the deletion machinery.** Firing rates on the 5,688 scored corner
  requests of the **base** store (⛔ before any deletion), all nine cells:

  | cell | accept | `empty_basin` | `out_of_range` | `hole` | `duplicate` |
  |---|---|---|---|---|---|
  | every cell except (s1, b2) | **1.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
  | ⛔ **(store 1, bridge 2)** | **0.8312** | ⛔ **0.1688** | 0.0000 | 0.0000 | 0.0000 |

  ⭐ **And the dead term is named.** Every cell has exactly 2 terms with zero live atoms —
  `object_hue[5]=0.5` and `shape[0]=0`, which are **the held-out values** and are *supposed* to
  be empty. At bridge seed 2 the different encoder leaves a **third, legitimate** term with no
  atoms out of the 64:
  - (store 0, bridge 2): `orientation[5] = −8.571` dies — ⭐ but no scored corner names it, so
    accept stays 1.0000. **A silent near-miss.**
  - ⛔ (store 1, bridge 2): `orientation[13] = 25.714` dies — and the scored corners **do** name
    it, so 16.88 % of them are refused outright.
- **2026-09-20T02:23Z** — ⛔ **What this costs the parent's table, stated exactly.** In that cell
  `field_geom` produces **403 fewer** post-hole and **263 fewer** post-eviction distinct valid
  novel tuples than `bitmap8` (2,271 vs 2,674; 1,923 vs 2,186), while `bitmap8` holds 1.0000 on
  both legs. ⛔ **So the parent's exact TIE with the 8-byte rule on survivor novel tuples — which
  it reported as identical on all three of its seeds — becomes a LOSS in 1 of 9 instrument cells.**
  ⭐ In the other 8 cells the tie is exact, so the parent's finding is right about the typical case
  and wrong to have been stated without an instrument qualifier.

---

# FINAL REPORT — w5.1/refuse (AUTO-HUB RETRY, kind R-A robustness)

**Parent:** `w5 hub/refuse`. **Branch** `w5.1/refuse` from its tip `10b0734`.
**START_UTC** 2026-09-20T00:39:18Z · **finished** ~02:30Z · **≈ 1 h 51 m of an 8 h allowance.**
⛔⛔ **SCORED RUNS: 0 of 0. The `refusal` target still has 0 of 3 remaining and I spent none.**
Every number below re-reads rows the parent already paid for, through the frozen
`score_deletion` / `score_novelty_row`. ⛔ No PREREG was opened, because a PREREG binds a scored
run and there was none to bind. ⛔ Nothing about the field store, the rule, the thresholds, the
pins or the metric was changed — **the dial was the instrument, exactly as declared.**
⭐ **Labels** are used ONCE at fit time to align wells and build the address book, and at the
interface. ⛔ Never in the landscape, the read or the write.

## 1. ⭐⭐⭐ THE HEADLINE — two sentences, and the second one is against the parent

⭐⭐ **NOTHING THE PARENT CLAIMED MOVED UNDER THE INSTRUMENT IT WAS ASKED ABOUT.** Every claim it
makes is a **bounded** quantity — refuse the deleted conjunction **1.0000**, accept the never-held
conjunction **1.0000**, `bitmap8` Δreject **0.0000**, the hole **1 byte**, **0 bytes outside**,
every rendered feature bit-identical — and every one is **unchanged at ± 0.0000** across the
parent's instrument, my rebuild of it, and the full joint instrument. ⭐ What moved is exactly what
both hubs had already agreed to call **provenance**: byte *counts* and novel-tuple *counts*. The
parent's own words — *"the invariant is not the count; the invariant is 0 outside"* — are
**vindicated by the very measurement that breaks its error bars**.

⛔⛔ **But its ± figures are understated, and one half of its claim fails outright in 1 of 9
instrument cells.** On **7 of 13** quantities, swapping the instrument at a **fixed** store seed
moves the number by **more than the parent's entire published ±**. And the **accept** half of its
headline claim — *"still answering 1.0000 of the 2,160 requests naming a never-held conjunction"* —
drops to **0.8333** at (store seed 1, bridge seed 2), because a legitimate term loses every one of
its atoms at that encoder. ⛔ With one instrument the parent could not have seen either.

## 2. ⭐ The one-line verdict the target file asks for

> ⭐⭐ **The wider instrument moves NOTHING the parent CLAIMED and EVERYTHING the parent
> QUANTIFIED — its bounded claims are untouched at ±0.0000, its ± figures are understated by up to
> 2.79×, and its accept leg is not instrument-invariant: it fails in 1 of 9 cells.**

## 3. ⭐⭐ THE ONE JOB — the bridge-seed spread, both instruments in ONE table

Full table: **`BRIDGE-SPREAD.md`** · machine-readable: **`bridge_spread.json`**.

⚠ **A convention finding that changes how the parent must be read, and it runs in its favour.**
Its `±` is **not** a half-range and **not** an sd. In its own `exp_gen_refuse_report._agg` it is
`spread = max − min`, the **full range**, printed as `mean ± spread`. So `2,317.3 ± 312` writes an
interval about **twice as wide as the three numbers it saw**. ⛔ The retry brief assumed the
notation was tight; it is generous. I adopt the parent's statistic verbatim so the columns are
like-for-like, and print each column's observed `[min, max]` beside it.

### 3a. ⚠ The range statistic HIDES the effect — the mistake I nearly made

Compared as spreads, several quantities look **calmer** under the wider instrument (`EVICTION bytes
changed` ×0.89, `NOVEL post-EVICTION` ×0.51). ⛔ That is an artifact: **a range over three cells can
stay the same width while every cell moves.** The sharp test is **paired** — the same store seed,
two instruments — and it was available only because both arms were run here.

### 3b. ⭐⭐ THE PAIRED TEST — same store seed, two instruments

| quantity | fixed (s0/s1/s2) | ⭐ joint (s0/s1/s2) | **max abs Δ** | the parent's WHOLE published ± | ⛔ ratio |
|---|---|---|---|---|---|
| EVICTION bytes changed | 17,569 / 20,113 / 20,218 | 17,569 / **16,764** / **19,114** | **3,349** | 2,649 | ⛔ **1.26×** |
| ⭐ EVICTION bytes OUTSIDE permitted | 0 / 0 / 0 | 0 / 0 / 0 | **0** | 0 | ⭐ bounded |
| EVICTION items evicted | 6 / 7 / 7 | 6 / **6** / 7 | 1 | 1 | 1.00× |
| EVICTION read rel. change on survivors | .0693 / .0632 / .0836 | .0693 / **.0890** / **.0853** | **.0258** | .0204 | ⛔ **1.26×** |
| NEGATIVE CONTROL bytes | 5,735 / 5,730 / 5,694 | 5,735 / **5,726** / **5,783** | **89** | 41 | ⛔ **2.17×** |
| NOVEL post-HOLE `field_geom`/`bitmap8`/`pair_deletion` | 2,284 / 2,490 / 2,178 | 2,284 / **2,030** / **2,570** | **460** | 312 | ⛔ **1.47×** |
| NOVEL post-HOLE `pair_presence` | 1,428 / 1,584 / 1,362 | 1,428 / **1,330** / **1,712** | **350** | 222 | ⛔ **1.58×** |
| NOVEL post-EVICTION `field_geom`/`bitmap8`/`pair_deletion` | 1,970 / 2,143 / 1,781 | 1,970 / **1,787** / **1,876** | **356** | 362 | 0.98× |
| NOVEL post-EVICTION `pair_presence` | 1,458 / 1,582 / 1,260 | 1,458 / **1,386** / **1,438** | **196** | 322 | 0.61× |

⭐ **Store seed 0 is exactly 0 in every row** — the two instruments coincide there by construction.
That is the internal control and it fires correctly.
⛔ **7 of 13 quantities exceed the parent's entire published ±. Its ± never contained the
instrument, and on these quantities the instrument is the larger term.**

### 3c. ⭐ The fixed points — "did not move" MEASURED, not assumed

| quantity | parent, fixed | mine, fixed | ⭐ mine, joint |
|---|---|---|---|
| HOLE bytes changed | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| HOLE bytes changed OUTSIDE permitted | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | **0.0000 ± 0.0000** |
| HOLE reject deleted conjunction, `field_geom` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| HOLE accept never-held conjunction, `field_geom` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| HOLE Δreject, `bitmap8` (THE BAR) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | **0.0000 ± 0.0000** |
| HOLE accept never-held conj., `pair_presence` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | **0.0000 ± 0.0000** |
| EVICT reject deleted value, `field_geom` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| EVICT reject deleted value, `bitmap8` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| EVICT accept survivors, `field_geom` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | **1.0000 ± 0.0000** |
| HOLE rendered features bit-identical | [T, T, T] | [T, T, T] | **[T, T, T]** |

## 4. ⭐⭐ ACCEPTANCE ITEM 2 — ARE THE TWO HUBS COMPARABLE NOW? **YES, and mechanically so**

⭐⭐ **YES.** After `--bridge-mode joint` my store image is **bit-identical to `w5 hub/delete`'s on
all three seeds**, and my pinned arm is **bit-identical to the parent's on all three**:

| store seed | mine, PINNED | mine, JOINT | the PARENT | ⭐ `w5 hub/delete` |
|---|---|---|---|---|
| 0 | `697d60df…abeb3b` | `697d60df…abeb3b` | `697d60df…abeb3b` | `697d60df…abeb3b` |
| 1 | `d0551f14…e8dc5c` | ⭐ **`1719deda…6c5a02ac`** | `d0551f14…e8dc5c` | **`1719deda…6c5a02ac`** |
| 2 | `3d9dc66b…ec2b21` | ⭐ **`db12e26a…f474392`** | `3d9dc66b…ec2b21` | **`db12e26a…f474392`** |

⭐ **12/12 cross-hub byte cells agree exactly** (evict bytes, bytes outside, items evicted,
negative control at all three seeds) — ⛔ read from `w5 hub/delete`'s own
`run1/seed_*/bytes.json`, not from its LOG. ⭐ The agreement is **not statistical**: at every seed
the two hubs are *the same 2,054,866-byte image*, reached by three independent code paths.
⛔ **Under the parent's original `pinned` convention they were NOT comparable**, and the reason is
now exhibited rather than argued: at seeds 1 and 2 the two hubs were building different stores from
different encoders. ⛔ Neither hub's numbers were wrong; they answered different questions.

⚠ **The honest limit on that YES.** It is a statement about the **store and the instrument**, which
are now identical. It is ⛔ **not** a claim that the two hubs' *metrics* are interchangeable — they
score different rows — so a shared row still needs a shared metric. ⭐ What is now legitimate is
exactly what was illegitimate before: **the two hubs' byte counts for the same operation on the
same store may share a table.**

## 5. ⭐⭐ BEYOND THE BRIEF — the 3×3 grid, and what it caught

Full table: **`BRIDGE-GRID.md`** · machine-readable: **`bridge_grid.json`**.

`pinned` vs `joint` says the spread is wider but ⛔ cannot say **why**, because the diagonal mixes
the store draw with the instrument. `--bridge-seed N` opens the full **(store seed) × (bridge seed)**
grid: column `b=0` is the parent's own store spread, a **row** is the instrument alone.

| quantity | ⭐ max INSTRUMENT range (a row) | parent's STORE range (`b=0`) | ratio |
|---|---|---|---|
| NOVEL post-EVICTION `field_geom`/`bitmap8`/`pair_deletion` | **1,009** | 362 | ⛔ **2.79×** |
| NOVEL post-EVICTION `pair_presence` | **782** | 322 | ⛔ **2.43×** |
| NOVEL post-HOLE `pair_presence` | **496** | 222 | ⛔ **2.23×** |
| NEGATIVE CONTROL bytes | **89** | 41 | ⛔ **2.17×** |
| NOVEL post-HOLE `pair_deletion` | **644** | 312 | ⛔ **2.06×** |
| NOVEL post-HOLE `field_geom` | **603** | 312 | ⛔ **1.93×** |
| EVICTION bytes changed | **3,349** | 2,649 | ⛔ **1.26×** |
| EVICTION items evicted | 1 | 1 | 1.00× |
| ⭐ **EVICTION bytes OUTSIDE permitted** | **0** | **0** | ⭐ **bounded — 0 in all nine cells** |

⚠ **Even the store effect is not a stable property.** `EVICTION bytes changed` has store range
**2,649** at `b=0`, **273** at `b=1` and **96** at `b=2`. ⛔ The parent happened to hold fixed the
one instrument at which the store spread looks largest.
⭐ **Internal consistency: 36/36 diagonal cells identical** between the grid's `(s,s)` cells and the
separately-run `joint` arm — two code paths, twelve quantities, three seeds, no disagreement.

### 5a. ⛔⛔ THE ACCEPT-LEG FAILURE — the finding only a grid could reach

| store seed \ bridge seed | b=0 | b=1 | b=2 |
|---|---|---|---|
| store 0 | 1.0000 | 1.0000 | 1.0000 |
| store 1 | 1.0000 | 1.0000 | ⛔ **0.8333** |
| store 2 | 1.0000 | 1.0000 | 1.0000 |

⭐ **The mechanism is named, not guessed: `empty_basin`, firing at 0.1688 on the BASE store before
any deletion.** Every cell has exactly 2 terms with zero live atoms — `object_hue[5]` and
`shape[0]`, **the held-out values**, which are supposed to be empty. At bridge seed 2 the different
encoder leaves a **third, legitimate** term with none of the 64 atoms:
- (store 0, bridge 2): `orientation[5] = −8.571` dies — ⭐ no scored corner names it. **A silent
  near-miss**, and it is the more alarming of the two.
- ⛔ (store 1, bridge 2): `orientation[13] = 25.714` dies — the scored corners **do** name it, so
  16.88 % of them are refused outright.

⛔ **Consequence for the parent's table:** in that cell `field_geom` yields **403 fewer** post-hole
and **263 fewer** post-eviction novel tuples than `bitmap8` (2,271 vs 2,674; 1,923 vs 2,186) while
`bitmap8` holds 1.0000 on both legs. ⛔ **The parent's exact TIE with the 8-byte rule becomes a LOSS
in 1 of 9 instrument cells.** ⭐ In the other 8 the tie is exact — so the parent is right about the
typical case and wrong to have stated it without an instrument qualifier.

## 6. ⚠ Two faults of the INSTRUMENT the programme should know about

1. ⭐⭐ **A store image's reproducibility is BLAS-thread-count dependent.** At
   `OMP_NUM_THREADS=1` my seed-0 store hashed `617e322125fb6754…`; at **3** it hashed
   `697d60dfa3e4b70c6822e04af1abeb3b` — **bit-identical to the parent's and to `w5 hub/delete`'s**.
   The difference is float32 ULP noise confined to `coupling_u`/`coupling_v` (30/10,576 and
   1,441/245,760 elements, max abs **1e-7**), because `build_field` takes a LAPACK
   `np.linalg.svd` whose reduction order depends on the thread count. ⭐ The code already
   canonicalises the SVD's **signs**, with a comment saying that without it a rebuild "differs by a
   few hundred bytes"; ⚠ what it does not pin is the **magnitudes**.
   ⛔ **Every other segment — `atoms` (all 524,288 B), basins, term map, coupling weights, hole
   flags — is bit-identical at either setting, so no byte claim of either hub is affected.**
   ⭐ **Recommendation: a hub quoting a store image hash should quote its BLAS thread count beside
   it**, or a replication on a different setting will see a mismatch that is not a disagreement.
2. ⚠ **The atom budget can starve a legitimate term.** 64 atoms spread over 57 terms leaves terms
   with as few as 0; which term starves depends on the encoder. ⭐ **Recommendation: assert at
   build time that every term the scored corners name has ≥ 1 live atom**, and fail loudly rather
   than refusing 17 % of a row silently. ⛔ Filed as an instrument property, not a hub result.

## 7. ⭐ Controls and what each proves

| control | verdict |
|---|---|
| `--stage seed0check` — the two modes must coincide at store seed 0 | ⭐ **PASSES on all five hashes** (bridge seed, features sha256, store sha256, decoder hash, classifier hash). ⛔ Had it failed, every seed-1/2 number would be measuring something other than the instrument. |
| PINNED arm vs the PARENT — the environment control | ⭐ **29/29 per-seed cells identical**, and the full **(5,688 × 6) judged tuple matrices bit-identical**. ⛔ So the dial contrast has NO environment term in it. |
| Store seed 0 in the paired test | ⭐ **Exactly 0 in every row**, as it must be by construction. |
| Grid diagonal vs the `joint` arm — two code paths | ⭐ **36/36 identical.** |
| Partition thread- and bridge-independence | ⭐ **Bit-identical to the parent's** on all 11 arrays and the JSON, at 1 and at 3 threads, in every tree. ⛔ Not a second uncontrolled term. |
| Negative control (1e-3 nudge to one address-book row) | ⭐ Fires in every cell: 5,694–5,795 B and `assert_no_refit` FIRES. The instrument is not blind. |
| `EVICTION bytes OUTSIDE permitted` | ⭐ **0 in all nine grid cells**, both deletion flavours. ⛔ The one thing that is genuinely a property of the object. |

## 8. ⛔ §A/§C bookkeeping

- **§A.1/A.2 — scored runs: 0 spent, 0 available, 0 used.** ⭐ The target file's kill-switch
  ("you find yourself wanting a scored run ⇒ stop and report") **never fired**: no quantity here
  needed one, because every column re-reads already-scored rows.
- **§A.4 — frozen assets untouched.** `score_deletion`, `score_novelty_row`, `score_refusal_row`,
  `accept_rule`, `value_presence_bitmap`, `gen_field` and every threshold are imported verbatim.
  ⛔ My only code change to the parent's runner is `bridge_seed_for(seed)`, and `pinned` (the
  default) reproduces it bit-for-bit — **proved, not asserted**, by the 29/29 control.
- **§A.5 — independent replication: ⛔ DECLARED NOT RUN for this retry.** ⚠ My pinned arm
  reproducing the parent is an **environment control run on the parent's own code by me**; that is
  ⛔ **not** an independent re-derivation and I do not call it one. ⭐ The parent already carries
  its own §A.5 clean-checkout spoke. ⛔ A declared NOT-RUN is never a null.
- **§C floor checks — ⛔ NOT RE-RUN, and deliberately so.** This hub moved no read parameter, no
  rule and no threshold, so §C.1 (matched-bytes lookup), §C.2 (landscape removed/shuffled) and
  §C.4 are **unchanged from the parent's** and re-running them would spend compute to reproduce a
  number nobody disputes. ⚠ **One §C.3 qualification is now owed, and it is mine to raise:** the
  parent's tie-with-8-bytes narrative rests on "a deletion that changes the output exactly", which
  still holds — ⛔ but §5a shows the accompanying *accept* leg is not instrument-invariant, so the
  tie clause should be quoted with the instrument named.

## 9. Cost line

Local CPU, **3 threads**, ⛔ **no GPU** — no run here had a headline number as its best case, so
§D never engaged and no GPU justification line was written. Wall clock **≈ 1 h 51 m** of the 8 h
allowance. Compute: 15 store builds at 35–65 s, 12 deletion stages at 200–400 s each (each carrying
two judge passes over 5,688 corner + 512 accept-leg requests), all **sequential, one process at a
time**. ⛔ **No box-wide kill was ever issued**; the only processes stopped were PIDs 27905/28900
and 28472/14784, identified by command line as mine, and the three other agents' processes were
left alone throughout.
**Scored-run count, travelling with every number above: refusal 0/3 spent by me, 3/3 spent by the
parent, 0 remaining.**

## 10. What would have proved this retry wrong, and what actually happened

| the target file's falsifier | outcome |
|---|---|
| "The spread moves a reported quantity beyond its stated ± ⇒ say the ± figures are understated" | ⭐ **FIRED, on 7 of 13 quantities. Said plainly in §1 and §3b.** That is the result. |
| "Bridge seeds 1/2 turn out not to be full scope ⇒ stop, the premise is dead" | ⛔ Did not fire — 480,000 rows, `scope: full`, generation 2, three distinct encoder hashes, verified from each cache's own manifest. |
| "You cannot vary the encoder/decoder/classifier without refitting the store ⇒ say so" | ⚠ Partly applies and is stated: varying the instrument **does** require rebuilding the store, but that is a *different store*, not a refit of a deployed one. `assert_no_refit` binds the deletion and passes 14/14 everywhere. |
| "You find yourself wanting a scored run ⇒ you have drifted; stop and report" | ⛔ Never fired. 0 scored runs. |

## 11. Artifacts

| file | what it is |
|---|---|
| `LOG.md` | this file — dated bullets throughout, final report as the last section |
| `BRIDGE-SPREAD.md` / `bridge_spread.json` | ⭐ the one-job table: parent / mine-fixed / mine-joint, the paired test, the fixed points, the cross-hub comparability test |
| `BRIDGE-GRID.md` / `bridge_grid.json` | ⭐ the 3×3 grid, the store-vs-instrument decomposition, the accept-leg audit |
| `runs/seed0_modes_coincide.json` | the five-hash assertion that the mode switch is a pure reparametrisation |
| `runs/{pinned,joint,bridge1,bridge2}/seed_*/deletion.{npz,json}` | every cell's raw measurement and its instrument provenance |
| `runs/logs/*.log` | every stage's stdout, one file per (stage, mode, seed) |
| `run_joint.sh`, `run_grid.sh` | the two drivers, with the thread-count rationale in the comments |
| `MY_PIDS.txt` | every PID this hub launched, captured at launch |
| `w51-refuse.bundle` | the branch, bundled whole |

---

## 12. Close-out verification — 2026-09-20T02:25Z

- **2026-09-20T02:25Z** — ⭐ **CODE: committed, tree clean.** 6 commits on `w5.1/refuse` from the
  parent tip `10b0734` to `ef13542`; `git status --short` is **empty**. Files: `exp_gen_refuse.py`
  (+143, the mode switch and `seed0check`), `exp_gen_refuse_bridge_spread.py` (new),
  `exp_gen_refuse_bridge_grid.py` (new). ⛔ No frozen-harness file touched.
- **2026-09-20T02:25Z** — ⭐ **BUNDLE: whole and verified.** `w51-refuse.bundle` (4,874,171 B),
  `git bundle create --all`. `git bundle verify` reports **"The bundle records a complete
  history."** and it carries `refs/heads/w5.1/refuse = ef13542…` plus every other branch.
- **2026-09-20T02:25Z** — ⭐ **OUTPUTS: verified by directory listing, ⛔ not by `git status`**
  (which cannot see a gitignored outputs tree — the w4 failure mode w5-common §5 warns about).
  Top level: `LOG.md`, `BRIDGE-SPREAD.md` + `bridge_spread.json`, `BRIDGE-GRID.md` +
  `bridge_grid.json`, `START_UTC.txt`, `MY_PIDS.txt`, `run_joint.sh`, `run_grid.sh`,
  `driver.log`, `grid.log`, `w51-refuse.bundle`, `runs/`. Under `runs/`: 14 files each in
  `pinned` / `joint` / `bridge1` / `bridge2`, 27 stage logs, 4 each in the two `seed0check` trees,
  and ⭐ **all 12 `deletion.npz` cells present** (4 trees × 3 seeds). Total **33 MB**.
- **2026-09-20T02:25Z** — ⭐ **NO SUB-AGENT AND NO PROCESS OF MINE IS RUNNING.** I spawned no
  sub-agents. Process check at close: the only `python.exe` on the box belong to
  `exp_gen_delete_row7` (`w5-delete`) and `exp_w5_compose_field` — ⛔ **neither is mine and I
  touched neither.** ⭐ **No box-wide kill was ever issued at any point in this hub.**
- **2026-09-20T02:25Z** — ⛔ **FINAL BUDGET STATEMENT: `refusal` scored runs remain 0 of 3.**
  I spent none, opened no PREREG, and wrote no row to `/c/chlu/SCORED-LEDGER.md` because there was
  nothing to write. ⭐ Every number in this log is a re-measurement of an already-scored row's
  spread under a different frozen instrument.
