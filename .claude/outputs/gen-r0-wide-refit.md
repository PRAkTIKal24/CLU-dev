# gen-r0-wide-refit — experiment-engineer report

**Task + acceptance criterion:** re-fit R0 (grouping **and** store) on the **wide** feature cache, report the
adjusted-Rand agreement with Montero's grouping, and give an explicit `CONSISTENT` / `PLUMBING FAULT — STOP`
verdict on the wide-vs-Montero **read margin** — plus R0's wide ladder coordinates.
**Status: done.** 3 seeds {0, 1, 2}, all complete. ⭐ **THE GATE: `CONSISTENT`** (0.70× the pooled seed spread on
both the aggregate and `shape`). `assert_plan_matched` passes. Montero R0 is untouched and reproduces bit-identically.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 2 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐⛔ **THE LADDER'S x-AXIS INVERTS SIGN BETWEEN THE TWO CONFIGURATIONS, and it is an artefact of the metric,
>    not of the store.** Montero (R22, corrected): pooled leakage margin **`+0.1337 ± 0.0592`** (random leaks
>    *more*). Wide: **`−0.0660 ± 0.0144`, 3/3 seeds negative, 4.59× its own spread** (random leaks *less*).
>    ⭐ **Diagnosed, not guessed** (§4.2): the numerator set and the denominator set do not realise the same number
>    of classes — on the wide split the 13 corners fix `scale` and `orientation`, so the numerator is a **6-class**
>    `orientation` problem against a **15-class** denominator. Balanced accuracy is normalised *within* a class
>    count, not *across* class counts, so the weaker arm gains more from the easier numerator and the margin flips.
>    ⛔ **R1 must not be run on this x-axis as it stands.** Options in §4.3. **Owner needed before R1.**
> 2. ⚠ **R24 rider 1's premise does not hold, and the number it asks for cannot mean what it was meant to mean.**
>    ARI(Montero, wide) = **`0.1150 ± 0.0126`** — but **Montero's own cross-seed ARI is `0.0638`** (§2.3), i.e.
>    *lower*. ⛔ The two caches are **different encoder fits** (by construction — a wide encoder must not have seen
>    the wide cells) and channel **identity** does not survive a re-fit: mean diagonal correlation **0.333**, and
>    the diagonal is the best match for only **8.1 %** of channels (§2.4). A grouping is a partition of channel
>    *indices*, so a cross-encoder ARI is bounded by encoder reproducibility, which is itself ~0.06. **The
>    agreement number is in the ledger as ruled; the ruling's interpretation of it needs a Head/Hub decision.**

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *isolation* — R0, re-established on the configuration the ladder will actually run on.
- **Laundering control:** ⭐ **random grouping at the same `K` and the same well-size histogram — a full arm on
  every number below**, never a footnote.
- **Falsifies:** wide-R0 and Montero-R0 diverge beyond seed spread on the read margin ⇒ **plumbing fault, STOP.**
  ⛔ **It did not fire:** 0.70× the pooled spread on both statistics, against a pre-registered tolerance of 3×.
- **Does NOT falsify:** a *level* difference between the configurations (they hold out different rows). ⚠ One did
  arise and is reported as a non-falsifier: the wide correlation read is `0.5956` vs Montero's `0.6107`.

## FLAG-PROVENANCE TABLE (protocol §5, mandatory)
| | value |
|---|---|
| branch / commits | `agent/experiment-engineer/gen-r0-wide-refit` — **`194b66a`** (one commit; ⛔ every artefact of this spoke is under the gitignored `.claude/`); base **`main @ 9a4457d`** |
| worktree | `/Users/user/Desktop/CHLU-gen-r0wide` (fresh; ⛔ the shared checkout was on another agent's branch `pilot-ttt-nan-and-d5-wiring @ 7fcef50` the whole session and was never touched) |
| dataset / pretext | `shapes3d` / `recon`; `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen` |
| **condition** | ⭐ **`recomb2element_wide`** (⛔ non-default) — 13 cells, `n_test = 936`, `n_train = 479,064` |
| `gen_data.wide_n_cells / wide_seed / wide_include_montero` | 12 / 0 / True (shipped defaults) |
| ⛔ **decoder recipe** | **`v1_capacity_schedule`** — asserted **three ways** before any number was recorded: the run's `dec_recipe.name`, the resolved config recipe dict, and the decoder **parameter hash** against `gen-bridge-wide`'s published per-seed hashes. ⛔ It is also asserted **≠** Montero's baseline decoder hash. (`gen_bridge.dec_recipe` still defaults to the superseded `"baseline"`; `gen_r0_wide.dec_recipe` defaults to the winner.) |
| frozen wide stack (per seed 0/1/2) | encoder `e159c6b5…` / `5b7a04d1…` / `35b72d38…`; decoder `cedb6933…` / `12a8dbeb…` / `0cb8f032…`; probe `fcdd49a0…` / `d02e1fae…` / `c92f7f2d…` — all == `gen-bridge-wide`'s published values |
| feature cache | scope **`full`**, 480,000 rows, `float16`, sha256 verified: `cbc713ad…` / `910be2d1…` / `9c53bcb8…`. ⛔ **0 rows encoded on the fly** (unlike Montero's seeds 1–2) |
| Montero encoders (read-only, for the diagnostics) | `98d32602…` / `fc05e2c5…` / `154d498e…` — ⛔ **different by construction**, see §2.4 |
| `K` | **5, loaded verbatim from `store_plan.json`** (`k_choice.rule = "PINNED …"`); ⛔ `choose_k` is never called on this rung. `assert_plan_matched` **PASSED**: `K=5, n_modes=64, D=2048, store_bytes=524288` |
| store config (all shipped `GenStoreConfig` defaults) | `n_modes 64` · `n_store_train 32768` · `n_corr 16384` · `n_probe_train 8192` · `train_steps 1200` · `train_batch 128` · `train_lr 3e-3` · `commitment_beta 0.25` · `tau 1.0 → 0.1` geometric · `grouping_method "auto"` → **ward** on all seeds · `min_well_frac 0.10` · `eval_batch 256` |
| leakage probes | `probe_steps 400` · `probe_batch 512` · `probe_lr 3e-3`; key chain continued from `train_store`'s returned key exactly as `run_arm` does |
| new knobs (`GenR0WideConfig`, all defaults) | `gate_tolerance_sigmas 3.0` · `underpowered_spread_multiple 3.0` · `exclude_shared_corner_from_pooled True` · `shared_corner_cells ["montero","wide-000"]` |
| rows per seed | `store_train` 32,768 · `eval_a` 1,024 · `eval_b` 936 · leakage numerator 3,103 / 3,112 / 3,120 · rows excluded as fitted 41,414 / 41,427 / 41,417 |
| seeds / spread | {0, 1, 2}; `±` = **population std over seeds (ddof = 0)**, the convention every gen spoke uses. Per-seed values printed under every table |
| PRNG | store key `seed*10007 + 20260911 + (0 correlation | 1 random)` — ⛔ identical to the shipped R0 chain; row draws `default_rng(seed*7919 + 20260910)`; diagnostics `default_rng(20260911+seed)` / `default_rng(20260912+seed)` |
| JAX / env | **main venv** `/Users/user/Desktop/CHLU/.venv`, jax **0.9.0**, equinox 0.13.4, numpy 2.4.1, Python 3.11.13, CPU, float32. ⛔ **no `uv sync` in the worktree** (protocol §4) |
| wall clock | **4,223 / 4,412 / 4,476 s** per seed, 3 concurrent + 1 foreign agent process on 8 cores |
| ⛔ CSF3 | **NOT RUN** — unreachable (2FA). Nothing here needed it; no job id exists and none is reported as a pass |
| ⚠ interrupted run | the first launch (2026-09-11 23:36) was **killed by a machine sleep/restart at ~00:26** with no seed finished. Relaunched under `caffeinate -i -s` at 10:38. ⭐ The deterministic prefix reproduced **exactly** across the restart (ARI `0.1027`, stability `0.4961`, channel identity `0.3192` on seed 0, to 4 dp) — an unplanned reproducibility check |

---

## 1. ⛔ THE GATE (acceptance 3) — `CONSISTENT`

**Read margin = correlation − random, raw multi-class top-1 on `eval_a`** — ⛔ deliberately the *same statistic*
as Montero's shipped comparison values (`gen-r0-rescore` §3, V1 decoder), because a gate that compares two
different statistics is not a gate.

| statistic | **wide-R0** | Montero-R0 (shipped, V1) | Δ | Δ in pooled spreads | verdict |
|---|---|---|---|---|---|
| aggregate | **+0.1410 ± 0.0097** | +0.1504 ± 0.0094 | −0.0094 | **0.70×** (tol 3×) | ⭐ **CONSISTENT** |
| `shape` | **+0.1813 ± 0.0228** | +0.2008 ± 0.0165 | −0.0195 | **0.70×** (tol 3×) | ⭐ **CONSISTENT** |

Per-seed wide margins — aggregate `+0.13704 / +0.13151 / +0.15430`; `shape` `+0.17090 / +0.16016 / +0.21289`.
**3/3 seeds positive on both.** Pooled spread = `sqrt(sd_wide² + sd_montero²)` = 0.0135 (aggregate) / 0.0281 (`shape`).

⭐ **Verdict: `CONSISTENT`. The two feature caches behave like the same encoder's on the axis the ladder reads.**
⛔ The ladder is **not** stopped. R1 is unblocked *by this gate* — but see reconciliation item 1, which blocks it
for a different reason (the x-axis, not the caches).

⚠ **Honest note on the tolerance.** 3× was pre-registered in `PREREG.md` and lives in `GenR0WideConfig`, not in the
harness body. The measured 0.70× clears it with room; ⛔ I did not choose the tolerance after seeing the number.

---

## 2. Grouping agreement (acceptance 1, R24 rider 1)

### 2.1 The number the ruling asks for
| quantity | per seed (0 / 1 / 2) | mean ± sd |
|---|---|---|
| ⭐ **ARI(Montero grouping, wide grouping)** | +0.1027 / +0.1099 / +0.1323 | **+0.1150 ± 0.0126** |
| best-permutation agreement (same-well fraction) | 0.3892 / 0.4326 / 0.4546 | 0.4255 ± 0.0272 |
| ⛔ chance floor — ARI vs **Montero's random arm** | −0.0008 / +0.0061 / −0.0021 | **+0.0011 ± 0.0036** |

**The agreement is real but low: 0.115, against a chance floor of 0.001.** ⛔ `K` was **not** re-derived to improve
it (ladder constant). Well sizes: wide `[295,286,286,744,437]` / `[422,228,290,233,875]` / `[351,182,313,358,844]`
vs Montero `[303,305,573,501,366]` / `[397,300,307,300,744]` / `[283,283,390,616,476]`; `ward` accepted on every
seed and arm; dead channels 81 / 64 / 58 (Montero, seed 0: 71).

### 2.2 ⭐ Before interpreting that as instability — the like-for-like control
**Same cache, same `K`, same method, a DISJOINT sample of 16,384 train rows:**

| | per seed | mean ± sd |
|---|---|---|
| ARI (row-split, encoder held fixed) | 0.4961 / 0.5030 / 0.6252 | **0.5414 ± 0.0593** |
| best-permutation agreement | 0.7290 / 0.7676 / 0.8237 | 0.7734 ± 0.0389 |

⭐ **Even with the encoder held completely fixed, re-drawing the correlation rows moves ~23 % of the channels
(ARI 0.54).** So a substantial part of any grouping disagreement is the ward cut's own sampling sensitivity —
consistent with `gen-store-grouping` §2's finding that this is a low-contrast correlation matrix (mean
off-diagonal |corr| 0.143) and that `K=5` won its eigengap narrowly (0.0514 vs 0.0452).

### 2.3 ⭐ THE DECISIVE COMPARISON — Montero's own cross-seed agreement
| | 0v1 | 0v2 | 1v2 | mean |
|---|---|---|---|---|
| **ARI between Montero R0 groupings at different seeds** | 0.0565 | 0.0765 | 0.0583 | **0.0638** |
| ARI between wide R0 groupings at different seeds | 0.0541 | 0.0642 | 0.0561 | 0.0581 |

⭐ **Montero's grouping agrees with the wide grouping (0.115) BETTER than it agrees with itself at another seed
(0.064).** ⛔ Therefore the wide re-fit is **not** a source of grouping instability: *the grouping is
encoder-specific, and the encoder is seed-specific.* Low agreement across conditions is the **expected** value,
not a finding about the wide split.

### 2.4 ⭐ Why — measured, not assumed (the diagnostic R24 rider 1 needs)
The two caches are different **encoder fits** (⛔ by construction: a wide-condition encoder must never have seen
the wide cells). They share an architecture and a PRNG chain (same init) and 99.8 % of their eligible training
rows. On the **same 8,192 rows**, per-coordinate Pearson correlation between the two caches:

| | per seed | mean ± sd |
|---|---|---|
| mean **diagonal** corr (channel *i* vs channel *i*) | 0.3192 / 0.3168 / 0.3643 | **0.3334 ± 0.0219** |
| fraction with diagonal corr > 0.9 | 0.039 / 0.032 / 0.023 | 0.0312 ± 0.0065 |
| fraction whose **best match is its own index** | 0.077 / 0.085 / 0.082 | **0.0812 ± 0.0031** |
| mean **best-match** \|corr\| over all *j* | 0.7303 / 0.7195 / 0.7111 | **0.7203 ± 0.0079** |

⭐ **The feature *space* largely survives the re-fit (best-match 0.72); the channel *indexing* does not (diagonal
0.33; own index is the best match for 8 % of channels).** A grouping is a partition of channel **indices**, so an
ARI across two encoder fits is bounded above by exactly this. ⛔ **This is the reason the gate is on the margin
and not on the level, and it is why a low ARI here is a non-issue rather than a plumbing fault.**

---

## 3. R0's ladder coordinates on the wide configuration (acceptance 4)

### 3.1 The read metric — `shape` separate from the aggregate
⛔ Raw multi-class top-1 (chance = per-factor majority rate, **not** 0.5). ⛔ Do not place beside §3.2.

| | aggregate | `shape` |
|---|---|---|
| no-store ceiling (real cached features, same decoder + probe) | **0.9897 ± 0.0021** | 0.9733 ± 0.0072 |
| ⭐ correlation grouping | **0.5956 ± 0.0112** | 0.5671 ± 0.0438 |
| ⭐ random grouping (same `K`, same well sizes) | 0.4547 ± 0.0107 | 0.3857 ± 0.0211 |
| ⭐ **margin** | **+0.1410 ± 0.0097 (14.5×)** | **+0.1813 ± 0.0228 (8.0×)** |

Per-seed correlation aggregate `0.6068 / 0.5802 / 0.5999`; random `0.4697 / 0.4487 / 0.4456`. 3/3 positive.
⚠ The absolute read is low for **both** arms (0.60 against a 0.99 ceiling) because the store is a **30-bit**
channel by design (`K·log₂M = 5·6`, a declared design constant) — named in advance as a non-falsifier.
⭐ `assert_no_bypass` passed on all 6 arms: the read is a pure codebook gather, **30 bits/read**.

### 3.2 The in-corner read — ⭐ balanced accuracy is the gate statistic, AUROC the mandatory companion
Positives = a cell's held-out rows; negatives = the same corner's **matched-seen other shapes**. ⛔ Chance = 0.5.
⛔ Per R23.3 the headline pools the **other 11 cells**; the shared-corner pair is quoted separately.

| arm | balacc (11 cells) | AUROC (11 cells) | balacc (all 13) |
|---|---|---|---|
| real features (the ceiling) | **0.9894 ± 0.0023** | 0.9993 ± 0.0005 | 0.9798 ± 0.0091 |
| ⭐ correlation, store read | **0.6899 ± 0.0292** | **0.8223 ± 0.0303** | 0.6835 ± 0.0282 |
| ⭐ random, store read | 0.5854 ± 0.0311 | 0.6903 ± 0.0511 | 0.5800 ± 0.0318 |
| **margin (corr − rand)** | **+0.1045 ± 0.0510 (2.0×)** | **+0.1320 ± 0.0606 (2.2×)** | |

⭐ **Independent provenance check:** the real-feature row reproduces R23.3's Hub-recomputed **0.9894 ± 0.0021**
(11 cells) and **0.9798 ± 0.0088** (all 13) — through my own harness, with the negatives *further* restricted by my
store-training rows. I am reading the same instrument as `gen-bridge-wide`.
⚠ The store read (0.69) sits far below the real-feature ceiling (0.99) for both arms — the same **store-capacity**
statement R23.5 made on Montero (0.6866 vs 0.9414), reproduced at the wide configuration. ⛔ Not a grouping claim.

⛔ **The shared-corner pair (R23.3), never pooled:**

| cell | real features | correlation | random |
|---|---|---|---|
| `montero` | 1.0000 ± 0.0000 | 0.7825 ± 0.0786 | 0.6002 ± 0.0720 |
| `wide-000` | 0.8542 ± 0.0967 | 0.5133 ± 0.0188 | **0.5000 ± 0.0000** (exactly chance) |

### 3.3 Code-space companions (decoder-independent)
| | MIG | DCI-C | DCI-D | block-diagonality |
|---|---|---|---|---|
| correlation | **0.1838 ± 0.0178** | **0.1630 ± 0.0061** | 0.2957 ± 0.0123 | 0.4891 ± 0.0045 |
| random | 0.0365 ± 0.0104 | 0.0330 ± 0.0079 | 0.3175 ± 0.0050 | 0.2556 ± 0.0138 |
| ⭐ **margin (corr − rand)** | **+0.1473 ± 0.0154 (9.5×)** | **+0.1299 ± 0.0036 (36×)** | **−0.0218 ± 0.0155** | 1.91× |

⭐ Both companions reproduce Montero's structure (`+0.1823` MIG, `+0.175` DCI-C) at the wide configuration, 3/3
seeds. ⚠ **DCI-D again fails to favour the clustered arm** (margin **−0.0218**, i.e. it slightly prefers *random*) exactly as `gen-store-grouping` §9 warned — ⛔ do not
quote DCI-D alone as a factorisation measure in this setting.

---

## 4. ⛔ The leakage x-axis — the one place the two configurations genuinely disagree

### 4.1 The numbers (R23.2's definition, implemented verbatim)
`leakage = 1 − balacc(held-out ∪ matched-seen-other-shapes, same corner) ÷ balacc(same, seen corners)`, per well,
per factor. Numerator 3,103–3,120 rows over the 13 corners; denominator `eval_a` (1,024 rows, fitted by nothing).

| | pooled leakage | `shape` leakage |
|---|---|---|
| correlation | **−0.0960 ± 0.0246** | −0.0854 ± 0.0143 |
| random | **−0.1620 ± 0.0389** | −0.1276 ± 0.0349 |
| ⭐ **margin (rand − corr)** | ⛔ **−0.0660 ± 0.0144 (4.59×)** | ⛔ **−0.0422 ± 0.0271 (1.56×)** |
| Montero, same definition (R22) | +0.1337 ± 0.0592 (2.26×) | +0.0147 ± 0.0771 (0.19×) |

Per-seed pooled margin `−0.0458 / −0.0746 / −0.0778` — **3/3 negative**, i.e. **consistently inverted** relative to
Montero. ⛔ On the wide configuration the *random* grouping has the *lower* leakage, which on the ladder's own
reading would make the laundering control look **more** factorised.

**Non-degeneracy proof (acceptance 4):** **180** (well × factor) cells over both arms and all 3 seeds —
**0 non-finite**, range **[−0.6645, +0.1569]**, **0 cells with |leakage| > 1**. ⭐ The corrected metric is bounded
and well-behaved on the wide configuration too (the old definition produced 55/180 NaN and a worst cell of −198.0).

### 4.2 ⭐ The diagnosis — it is the metric, not the store
Per-factor pooled leakage (mean over seeds):

| | floor_hue | wall_hue | object_hue | scale | `shape` | orientation |
|---|---|---|---|---|---|---|
| correlation | −0.054 | −0.014 | −0.131 | −0.183 | −0.085 | −0.139 |
| random | −0.015 | −0.031 | **−0.290** | −0.207 | −0.128 | **−0.325** |

The inversion is carried by `orientation` and `object_hue`. Realised class counts, measured (seed 0, well 0):

| factor | classes in the **numerator** | classes in the **denominator** (`eval_a`) |
|---|---|---|
| orientation | **6** | **15** |
| scale | **6** | 8 |
| shape / hues | 4 / 10 | 4 / 10 |

⭐ **Each wide cell fixes a `scale` corner and an `orientation` corner, so the numerator is a 6-class problem
scored against a 15-class denominator.** Balanced accuracy is normalised *within* a class count, not *across*
class counts: the easier numerator lifts **both** arms above 1.0 transfer (hence negative leakage for both), and it
lifts the **weaker** arm more — which is exactly what flips the margin. ⛔ This is a property of the ratio, not of
the wells.

⚠ What **does** survive from R22: the *shape* of the per-well table. Correlation's five wells are heterogeneous
(`−0.046 … −0.181`, spread 0.045) while random's are near-uniform (`−0.147 … −0.204`, spread 0.021) — the same
2.2× "clustered wells differentiate, random wells are five copies of the same profile" signature.

### 4.3 Options for the Hub (⛔ not mine to choose)
1. **Match the class structure**: score the numerator and denominator over the **same realised classes** (restrict
   `eval_a` to the corners' `scale`/`orientation` values, or drop those two factors from the pooled x-axis).
2. **Per-corner normalisation**: compute leakage per cell against a *matched* seen set drawn from that cell's own
   hue bands (the `seen_pos` set `gen-bridge-wide` already builds), never against the whole grid.
3. ⭐ **Use the code-space axis** (MIG / DCI-C), which separates cleanly and identically on *both* configurations
   (+0.147 / +0.130 wide, +0.182 / +0.175 Montero) — R23.2 pre-registered them as companions, not substitutes, so
   promoting them is a Head decision.

### 4.4 ⭐ The ruling-2 fallback (acceptance 5) — **TRIGGERED**
`shape`-leakage separates the groupings by **1.56× the seed spread** (margin `−0.0422 ± 0.0271`), below the
pre-registered ~3×. ⛔ **`shape`-leakage is therefore declared UNDERPOWERED, in those words** — and note it is
underpowered **and sign-inverted**, so the widened split did **not** restore the resolution R22's option (b) hoped
for. ⛔ This is a pre-registered outcome, not a failure, and it is not mine to soften or to fix with another
statistic. ⭐ R0's claim on the wide configuration rests where R23.4 put it: on the **read** axis
(**+0.1410 aggregate / +0.1813 `shape`**, 14.5× / 8.0× spread) and on the **code-space companions**.

---

## 5. ⛔ Montero R0 is untouched, and reproduces bit-identically (acceptance 6)
- **Real data, seed 0** (`montero_reproduction_s0.json`): the new entry point, handed **Montero's own** train
  cache and plan, returns **2048/2048 labels identical** to the shipped `store_manifest.json` for **both** arms,
  well sizes **`[303, 305, 573, 501, 366]`** exactly as shipped. ⛔ Seeds 1–2 were not re-derived because their
  Montero caches are `scope="used"` and would need on-the-fly encoding — that is a *re-run*, which this task forbids.
- **Synthetic, in the suite:** `test_the_wide_entry_point_is_the_shipped_path_bit_for_bit` asserts the entry point
  equals `grouping_labels(channel_correlation(X), …, dead=dead_channels(X))` element-for-element for both arms,
  **and** that the store trained on top of it is bit-identical across repeated runs.
- The store half of the bit-identity was already established by `gen-r0-rescore`, which re-materialised both
  Montero arms with `|Δ| == 0.0`; ⛔ the new entry point does not touch `train_store` at all (it is additive code
  below every existing line of `gen_store.py`).
- ⛔ **No Montero artifact was written, re-fitted or modified.** Montero's numbers in this report are **read** from
  `gen-r0-rescore/summary.json` and `gen-store-grouping`'s manifests.

---

## 6. Pre-registration scorecard

**`PREREG.md` mtime `2026-09-11 23:16:14`** — before the harness existed (`wider0.py`'s first version ran its
first smoke at **23:17:29**) and ~11 h before the first production number (**10:38:38** the next morning for the grouping agreement,
**10:57:46** for the first read).
⚠ **One wrinkle, stated rather than smoothed over:** the **row-split stability control** (§2.2) and the
random-row fix to the channel-identity diagnostic were written at **~23:20, after the first *smoke* run had
already shown a low ARI and a low diagonal correlation.** ⛔ They are therefore **diagnostics added in response to
an observation, not pre-registered predictions**, and no prediction rides on them; every P-number below was fixed
at 23:16:14 and none was edited afterwards (`PREREG.md` has a single mtime).

| # | prediction | threshold | measured | verdict |
|---|---|---|---|---|
| **P1** | channel identity survives the re-fit: mean diagonal corr ≥ 0.80 | ≥ 0.80 | **0.3334** | ⛔ **FALSIFIED** |
| **P1-alt** | the registered counter-hypothesis: diagonal ≈ 0 (< 0.30) | < 0.30 | 0.3334 | ⚠ **also missed, narrowly** — ⭐ the truth is *between* them and is reported as such: the feature **set** survives (best match 0.72), the **indexing** does not (own index best for 8 %) |
| **P2** | ARI(Montero, wide) ≥ 0.30 | ≥ 0.30 | **0.1150** | ⛔ **FALSIFIED** — ⭐ and the control I registered it against explains it: Montero's own cross-seed ARI is **0.0638** |
| **P3** | ⛔ **THE GATE: CONSISTENT**, aggregate margin +0.15 [+0.10, +0.20] | ≤ 3× pooled spread | **+0.1410 at 0.70×** | ⭐ **HELD** (point and interval both) |
| **P3b** | `shape` margin +0.20 [+0.13, +0.27] | — | **+0.1813** | ✅ in interval |
| **P4** | the ruling-2 fallback **triggers** (`shape` leakage < 3× spread), point +0.05 [−0.05, +0.15] | < 3× | **1.56×**, margin **−0.0422** | ⭐ **HELD** (and the point is inside the interval, at its edge) |
| **P5** | pooled leakage margin > 0 on ≥ 2/3 seeds, point +0.13 | ≥ 2/3 | **0/3 — margin −0.0660** | ⛔ **FALSIFIED, sign-inverted** — reconciliation item 1 |
| **P6** | MIG margin ≥ +0.10 and DCI-C margin ≥ +0.10 | ≥ +0.10 | **+0.1473 / +0.1299** | ⭐ **HELD**, both |
| **P7** | 0 NaN cells; max ≤ +1.0; min ≥ −1.5 | — | **0 NaN**, range [−0.6645, +0.1569] | ⭐ **HELD** |
| **P8** | correlation aggregate read 0.61 [0.50, 0.72] against a ~0.99 ceiling | — | **0.5956** vs ceiling **0.9897** | ✅ **HELD** |
| **P9** | `assert_plan_matched` passes; `K=5`, 64 modes, `D=2048`, 524,288 B | exact | **PASSED, all four** | ⭐ **HELD** |

**5 held, 1 held with a caveat, 3 falsified, 1 split** — and ⭐ **the two most consequential falsifications (P1,
P5) are the two findings in the reconciliation list.** ⛔ No prediction was edited after the fact; the file's mtime
is 23:16:14, and the only numbers that existed before the production run were the **smoke**'s, at reduced
settings (⛔ never reported).

---

## 7. How I verified (commands + observed output)
```
# ⚠ plumbing smoke (NOT a result): the whole path at reduced settings
R0WIDE_SMOKE=1 python .claude/scratch/gen-r0-wide-refit/wider0.py fit 0
  ⭐ POSITIVE CONTROL (smoke only): assert_plan_matched REFUSED the run —
     ladder rungs disagree on n_modes: 8 != 64 — only the grouping may change between rungs
  (⭐ the ladder contract firing on purpose, on a run that deliberately broke it)

# production, per seed (3 concurrent, under caffeinate):
BASE_COMMIT=194b66a CHLU_GEN_DATA_ROOT=… PYTHONPATH=… python …/wider0.py fit {0,1,2}
  == seed 0: enc e159c6b5257ed125 dec cedb6933c9c1da1f probe fcdd49a02c265383 recipe=v1_capacity_schedule — verified
  ⭐ no-leak: 936/936 held-out excluded from 32768 store rows, 0 leaked; per cell 0 leaked over 13 cells
  ⭐ K = 5 (pinned) wells [295, 286, 286, 744, 437] dead=81 method=ward
  ⭐ GROUPING AGREEMENT vs Montero: ARI=0.1027 perm-agreement=0.3892 (chance floor … ARI=-0.0008)
  ⭐ GROUPING STABILITY (same cache, 16384 DISJOINT rows): ARI=0.4961 perm-agreement=0.7290
  ⭐ CHANNEL IDENTITY (P1): mean diag corr=0.3192 frac>0.9=0.039 argmax==diag on 0.077 of live channels
  [correlation] READ aggregate=0.6068 shape=0.5508 rel_mse=0.2740  (30 bits/read)
  [random]      READ aggregate=0.4697 shape=0.3799 rel_mse=0.4249  (30 bits/read)
  ⭐ assert_plan_matched PASSED: {'K': 5, 'n_modes': 64, 'D': 2048, 'store_bytes': 524288, 'n_rungs': 2}
  == seed 0 done in 4223s

# ⛔ the acceptance-6 control, on REAL data
python …/montero_repro.py
  correlation: 2048/2048 labels identical -> ⭐ BIT-IDENTICAL; wells [303, 305, 573, 501, 366]
  random:      2048/2048 labels identical -> ⭐ BIT-IDENTICAL; wells [303, 305, 573, 501, 366]

# the gate + tables + figure
python …/agg.py
  {"gate": {"aggregate": "CONSISTENT", "shape": "CONSISTENT"},
   "ari_vs_montero": 0.11498003492589938, "ruling2_triggered": true}

# targeted tests, then lint
pytest tests/test_gen_r0_wide.py -q -p no:randomly --no-cov   ->  20 passed in 52.03s
ruff check chlu/core/gen_store.py chlu/config.py tests/test_gen_r0_wide.py + all scratch harnesses
  -> All checks passed!
⚠ ruff FORMAT not applied: `chlu/core/gen_store.py` and `chlu/config.py` are NOT ruff-format-clean at the
  base commit (verified against `git show main:<file>`), so formatting them would produce a large
  out-of-scope diff; the new test file is left in the style of its sibling `tests/test_gen_store.py`,
  which is likewise not format-clean.
```

## 8. Suite
### 8.1 The run
```
pytest -q -p no:randomly --no-cov          (main venv, worktree, under caffeinate)
  -> 2119 passed, 13 skipped, 0 failed, 45 warnings in 3280.77s (0:54:40)
```

| | tests | note |
|---|---|---|
| **before** (`main @ 9a4457d`) | **2112 passed, 0 failed** | ⚠ **inherited, not re-measured** — `gen-bridge-wide` measured it on the branch whose merge *is* `9a4457d` (`e4f3d71..9a4457d` is exactly its 4 commits and nothing else) |
| **after** (`agent/experiment-engineer/gen-r0-wide-refit @ 194b66a`) | **2119 passed + 13 skipped = 2132 collected** | |
| delta | **+20 = exactly the 20 new tests in `tests/test_gen_r0_wide.py`** | ⭐ exact accounting: 2112 + 20 = 2132 |

⛔ **0 failed. No pre-existing test changed state.**

### 8.2 ⚠ The 13 skips are an environment artefact, and I checked every one
They are **not** mine and **not** new behaviour: a **worktree has no `.claude/`**, so the dataset-presence guards
fire (`Shapes3D h5 not at <worktree>/.claude/data/gen/...`). Located and re-run with the data root pointed at the
real data:

| file | skipped in the worktree | with `CHLU_GEN_DATA_ROOT` set |
|---|---|---|
| `tests/test_gen_data_splits.py` | 8 | pass |
| `tests/test_gen_bridge_wide.py` | 3 | pass |
| `tests/test_gen_wide_split.py` | 2 | pass |
| combined re-run (+ `test_gen_store.py`, `test_gen_r0_wide.py`) | — | **127 passed, 0 skipped** |

```
CHLU_GEN_DATA_ROOT=…/.claude/data/gen pytest -q -rs tests/test_gen_data_splits.py \
    tests/test_gen_bridge_wide.py tests/test_gen_store.py tests/test_gen_r0_wide.py
  -> 109 passed, 0 skipped in 156.27s
CHLU_GEN_DATA_ROOT=… pytest -q tests/test_gen_wide_split.py  -> 18 passed, 0 skipped
```
⭐ So the honest statement is **2132 green, 0 red**, of which 13 need `$CHLU_GEN_DATA_ROOT` to be exercised from a
worktree. ⚠ This is why `gen-bridge-wide` could report "0 skipped" at 2112 and I see 13: the difference is one
environment variable, not a code change.

---

## 9. Artefacts (all under `.claude/`, gitignored)
⭐ **The ladder contract for R1–R5: `outputs/gen-r0-wide-refit/store_plan_wide.json`** (schema
`chlu-gen-store-plan/1`, `rung="R0-wide"`, `condition="recomb2element_wide"`, seed-0 labels as `k_choice_seed=0`
demands). ⛔ `assert_plan_matched([montero_plan, wide_plan])` passes — `K=5`, `n_modes=64`, `D=2048`,
`store_bytes=524288` are **identical** to Montero's plan and only `labels` differ, which is exactly what the ladder
allows. It also carries the wide encoder/decoder/cache hashes and the agreement record. **R1–R5 load this file, not
Montero's.**

`outputs/gen-r0-wide-refit/` — `PREREG.md` · `wide_r0_seed_{0,1,2}.json` (full per-seed records: every arm, every
well, every cell) · `summary.json` · `tables.md` · `wide_r0_summary.png` (gate · agreement · x-axis) ·
`store_plan_wide.json` · `montero_reproduction_s0.json` · `suite_after.log` · `smoke_wide_r0_seed_0.json`.
`scratch/gen-r0-wide-refit/` — `wider0.py` (the harness) · `agg.py` · `montero_repro.py` · `fit_s{0,1,2}.log` ·
`smoke*.log`. ⛔ No run artifact was written inside the repo.

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-r0-wide-refit`, base **`main @ 9a4457d`**, worktree
  `/Users/user/Desktop/CHLU-gen-r0wide`.
- **Commits:** `194b66a` — *add the wide-condition R0 entry point, the grouping-agreement metric and THE GATE*.
- **Files touched (all additive):**
  - `chlu/core/gen_store.py` **+220, 0 deleted** — a new section 9 **below every existing line**:
    `adjusted_rand_index`, `grouping_agreement`, `fit_grouping_from_plan`, `read_margin_gate`, + 4 names in `__all__`.
  - `chlu/config.py` **+67, 0 deleted** — `GenR0WideConfig` (inserted immediately **above** `ProjectConfig`, because
    a dataclass referenced by `CHLUConfig` cannot be defined after it — ⚠ see the conflict note below) plus the
    three house wiring sites (`CHLUConfig` field, `load_config`, `save_config`), each marked
    `# --- gen-r0-wide-refit (additive; R24) ---`.
  - `tests/test_gen_r0_wide.py` **new, 20 tests**.
  - ⛔ Nothing under `chlu/eval/`, `chlu/data/`, `chlu/core/gen_bridge.py`, `chlu/experiments/`, or `scripts/csf3/`.
- **Rebased onto local `main`** (⛔ not `origin/main`, §7.21): `main` had not moved from `9a4457d`, so it was a
  no-op and **no `config.py` conflict materialised** — the other three spokes had not landed at the time of writing.
  ⚠ **My `config.py` block is NOT at the literal end of the file** (it cannot be: `CHLUConfig` must be able to
  reference it). Whoever merges next will meet it just above `ProjectConfig` and in the three wiring sites.
  ⭐ **Keep both blocks; delete, rename or reorder nothing.**
- **Ref verified from the MAIN repo** (protocol §3.2), working tree clean:
  `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/gen-r0-wide-refit` → `194b66a`;
  `git show --stat 194b66a` → **583 insertions, 0 deletions**, 3 files. ⚠ The worktree
  `/Users/user/Desktop/CHLU-gen-r0wide` is **left in place** for review — ⛔ I did not remove it.
- **Unresolved conflicts:** none. ⛔ The shared checkout (another agent's branch, `7fcef50`) was never touched.

## Open questions / follow-ups / risks
1. ⛔ **Reconciliation item 1 (the x-axis) blocks R1** just as surely as the gate would have. The gate says the
   caches are fine; the x-axis says the *ruler* is not. §4.3 has three options; the choice is a Head/Hub ruling.
2. ⚠ **The agreement number is in the ledger (0.1150 ± 0.0126) — but R24 rider 1's dichotomy ("high agreement
   documents a non-issue; low agreement is a finding about grouping stability") has a third branch that the ruling
   did not anticipate and that the data selects: *the number is bounded by encoder reproducibility*, which §2.3
   measures at 0.064. I report it; I do not adjudicate it.
3. ⚠ **`K = 5` is now pinned into a second configuration.** Its eigengap margin was narrow on Montero (0.0514 vs
   0.0452 at `k=6`) and §2.2 shows the ward cut moves under row resampling. ⛔ I did not re-derive it. If the Hub
   ever wants `K=6`, that is a *decision* taken before R1, not a re-derivation inside a rung.
4. ⚠ **The wide store is not checkpointed** (same as R0): the per-seed JSONs are the artifact. Re-running
   reproduces it from the same key chain — verified across the machine restart (§flag table).
5. ⚠ **`wide-000` reads at exactly chance (0.5000 ± 0.0000) for the random arm** and 0.5133 for correlation, while
   its real-feature ceiling is 0.8542. It is the hardest cell in both configurations, and it is half of the
   shared-corner pair. ⛔ Quoted separately, never pooled (R23.3).
6. ⚠ **Scope note — the deletion delta was NOT re-measured on the wide configuration.** Acceptance did not ask for
   it and each mode costs a full decode pass per arm; the shipped R0 numbers (`+0.4752` aggregate on `zero`) stand
   as Montero-configuration numbers. ⭐ The *structural* no-bypass result does not need re-measuring — it is a
   property of `read_from_codes`'s signature and `assert_no_bypass` passed on all 6 wide arms (30 bits/read).
   If the Hub wants the wide deletion delta, it is ~15 min/seed on top of a re-run.
7. ⚠ **CSF3 was not run** (2FA, unreachable). Everything is local CPU. The launcher's live smoke (R21) still gates
   the arms; ⛔ it does not gate this re-fit and I claim no cluster run.

## Proposed handover updates (for the Hub)
- **§3 config:** new **`GenR0WideConfig`** block (dataclass + `load_config` + `save_config`, additive), accessor
  `config.gen_r0_wide`. Notable defaults: `condition="recomb2element_wide"`, ⭐ **`dec_recipe="v1_capacity_schedule"`**
  (⛔ deliberately *not* `gen_bridge.dec_recipe`'s superseded `"baseline"`), `gate_tolerance_sigmas=3.0`,
  `underpowered_spread_multiple=3.0`, `shared_corner_cells=["montero","wide-000"]`.
- **§7 NEW known issue — a grouping is encoder-specific, and encoders are seed-specific.** Two fits of the same
  encoder architecture from the *same init* on 99.8 % of the same rows preserve the feature **space** (best-match
  channel correlation 0.72) but not the channel **indexing** (diagonal 0.33; own index is the best match for 8 % of
  channels). ⛔ Any cross-fit comparison of channel-index partitions (ARI or otherwise) is bounded by this:
  Montero's own cross-seed grouping ARI is **0.064**. Measure the diagonal before interpreting such a number.
- **§7 NEW known issue — R23.2's leakage ratio is not comparable across sets with different realised class
  counts.** On the wide split the numerator realises 6 `orientation` classes against the denominator's 15, which
  drives both arms' leakage negative and **inverts the margin** (Montero +0.1337 → wide −0.0660, 3/3 seeds).
  ⛔ Fix the class structure before reading any rung off this axis.
- **§7 NEW discipline note — a worktree suite silently SKIPS 13 dataset tests.** A worktree has no `.claude/`, so
  `tests/test_gen_data_splits.py` (8), `tests/test_gen_bridge_wide.py` (3) and `tests/test_gen_wide_split.py` (2)
  skip on "Shapes3D h5 not at `<worktree>/.claude/data/gen/...`". ⛔ A worktree count is therefore **13 lower** than
  a main-checkout count unless `$CHLU_GEN_DATA_ROOT` is exported — compare totals (passed **+** skipped), not
  "passed". All 13 pass with the variable set (verified, §8.2).
- **§7 NEW discipline note — `caffeinate`.** A machine sleep killed three multi-hour fits at ~25 min in with no
  checkpoint. Long local runs should be launched under `caffeinate -i -s`.
- **Rulings ledger (R24):** rider 1 **delivered** (ARI `0.1150 ± 0.0126`, per seed `0.1027 / 0.1099 / 0.1323`, with
  the chance floor `0.0011`, the row-split stability control `0.5414`, and Montero's cross-seed `0.0638`);
  rider 2 **honoured** (Montero R0 untouched, bit-identical, off the ladder); rider 3 ⭐ **`CONSISTENT` at 0.70× the
  pooled seed spread** — ⛔ **the gate does not stop the ladder.** R23.2's fallback is **TRIGGERED**:
  `shape`-leakage is **underpowered**. ⛔ R1 is still blocked, by reconciliation item 1, not by the caches.
