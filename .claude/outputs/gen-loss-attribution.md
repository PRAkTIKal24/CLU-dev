# gen-loss-attribution — experiment-engineer report

**Task + acceptance criterion:** probe `scale` and `orientation` at **three points on the same rows** — frozen-encoder
features → per-well sub-vector → store code — and say where the information disappears; acceptance = a verdict of
**World A / World B / explicitly in-between**, with the random-code control and the store's own MI beside it.
**Status: done.**

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 4 items, needs a Hub owner (protocol §5 corollary, first-10-lines rule)
> 1. ⛔⛔ **R40.E's encoder falsifier is ALREADY MET by the CURRENT encoder.** R40.E scopes `gen-encoder-pretrain`
>    against *"`scale` + `orientation` probe **≥ ~0.9 on seen data**"*. On the shipped frozen `recon` encoder, on
>    held-out rows of the spread grid, that probe returns **`scale` 0.9352 ± 0.0058** and
>    **`orientation` 0.9915 ± 0.0018**, 3/3 seeds. ⇒ **a new encoder cannot be justified by "the current one lacks
>    `scale`/`orientation`"** — that premise is false as stated. ⚠ The *other* clause of R40.E's falsifier (**swap
>    within the family**) is ⛔ **NOT** measured here and is untouched. **Owner: Hub, before spoke 2 launches.**
> 2. ⭐⭐ **The partition is EXONERATED; the k-means codebook is the culprit.** Of `scale`'s total loss, **9.5 %**
>    lands at the well partition and **90.5 %** at quantisation; for `orientation` the partition costs **0.0 %**
>    (retention 1.0037 ± 0.0016) and quantisation costs **all** of it. ⇒ any repair aimed at the *grouping*
>    (factor-aligned wells) addresses ≤ 10 % of the `scale` loss and none of the `orientation` loss.
>    **Owner: Hub, into the R40.D dynamical-read design note.**
> 3. ⚠ **"the 30-bit read" is NOT a capacity statement and must not be written as the cause.** The store assigns
>    **3,598 / 3,594 / 3,723 distinct codes to 4,096** spread rows. Distinctness is not the binding constraint
>    (consistent with R31's 160.2-bit result). ⛔ A draft sentence attributing this loss to "30 bits" is unsupported.
> 4. ⭐ **The loss profile is GRADED, not binary, and the current shorthand is imprecise.** R31/R34's *"the read
>    carries HUE and SHAPE"* holds for floor/wall hue (retention 0.997 / 0.995) but **`object_hue` retains 0.665 and
>    `shape` 0.555** — both well below the other hues and well above `scale` 0.225 / `orientation` 0.243.
>    **Owner: Hub, at the next table/caption review.**

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *attribution*. ⛔ **No fix, no new mechanism, no arm** — a known loss was located, not repaired.
- **Laundering control:** the **same probe class, same hyper-parameters, same PRNG key, same train/test rows** at all
  three points, plus the **random-code control** of the same width at the code point. ⛔ A different probe at each
  stage would measure the probes, not the pipeline.
- **Falsifies:** nothing of ours — both outcomes were informative by design.
- **Would prove this spoke wrong:** the random-code control scoring above chance. ⭐ **It did not:**
  `scale` **0.1199 ± 0.0051** (chance 0.1250), `orientation` **0.0649 ± 0.0011** (chance 0.0667) — *below* chance on
  both, 3/3 seeds. The code-level numbers are store content, not probe capacity.

## ⭐ THE VERDICT — one line, in the pre-registered words

⭐ **IN-BETWEEN, and the in-between sits entirely on the "code ≈ chance" clause: `scale` in-between · `orientation`
in-between — ⛔ World B is REJECTED (the encoder has both factors at ≥ 0.93), and the store discards 77.6 % / 75.7 %
of the above-chance signal at the k-means codebook, not at the partition.**

⛔ **Not rounded to World A.** World A needs *both* clauses. The **encoder clause PASSES outright** (0.9352 ≥ 0.90 and
0.9915 ≥ 0.90) but the **code is not at chance**: `scale` **0.3069** against a no-information floor of **0.1226**
(2.50×) and `orientation` **0.2909** against **0.0690** (4.22×). The code keeps ~22–24 % of the signal — far above
nothing, far below the encoder. **3/3 seeds return `in-between` independently.**

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| harness commit | **`d1b2dd4`** (branch tip; every number below produced at this tree — see §"Config drift" for the one default changed *before* the production run and *after* the plumbing smoke) |
| branch / base | `agent/experiment-engineer/gen-loss-attribution` off **`main @ 994ca9b`**; rebase onto `main` is a no-op (`main` did not move) |
| worktree / venv | `../CHLU-gen-attrib`, **main venv** (`/Users/user/Desktop/CHLU/.venv`), ⛔ no `uv sync`. **JAX 0.9.0** |
| dataset · pretext · condition | `shapes3d` · `recon` · `recomb2element` (Montero). Grid 480,000 rows = 10·10·10·8·4·15 |
| seeds | **0, 1, 2** (3 seeds, spread on every number; `±` is **population sd, ddof = 0**, declared in code) |
| encoder | ⛔ **LOADED, never fitted.** `gen-feature-bridge` per-seed `encoder.pkl`, parameter hash asserted against `bridge_manifest.json` **and** against the feature-cache manifest |
| feature cache | seed 0 `scope="full"` (480,000 rows) · seeds 1–2 `scope="used"` (9,288 rows). ⭐ Cache-extension **proved**: 64 genuinely cached rows re-encoded through the loaded frozen encoder, `max |Δ| = 0.0` on all three seeds |
| store | ⛔ **LOADED, never retrained.** `gen-read-compose` per-seed checkpoints, hashes `3e3424a1ea40…` / `76827ad534ee…` / `8449e6057dac…` — **asserted against `gen-commitment-collapse`'s manifest**. `K=5`, `M=64`, `D=2048`, **30 bits/read** |
| grouping | `gen-commitment-collapse`'s cached `labels_vec`; ⭐ the store's **well channels asserted identical** to the re-derived partition (⛔ not just their sizes). Well sizes s0 `[303,305,573,501,366]` · s1 `[397,300,307,300,744]` · s2 `[283,283,390,616,476]` |
| decoder | ⛔ **NONE.** This measurement never renders an image; `dec_recipe`'s `"baseline"` trap is structurally unreachable here |
| rows | `SPREAD_train` **12,288** · `SPREAD` (test) **4,096** · `CORNER_B` **72** · `CORNER_MS` **198/199/199**; pool = `split.train_rows \ store_train`, seeded `default_rng(seed·7919 + 20260914)` |
| row-set hygiene (measured) | `SPREAD_train ∩ SPREAD` = **0** · `SPREAD ∩ store_train` = **0** · `SPREAD ∩ eval_b` = **0** · ⚠ `SPREAD ∩ bridge_train` = **64 / 82 / 77 of 4,096 (1.6–2.0 %)**, the encoder's own fit rows — see limitation L1 |
| probe (⛔ identical at every point) | `WellProbe` (one `eqx.nn.Linear` head per factor, multinomial logistic, Optax Adam), `probe_steps=3000`, `probe_batch=512`, `probe_lr=3e-3`, key `PRNGKey(20260914 + seed)`, per-feature standardisation inside `train_well_probe` |
| random-code control | `default_rng(20260914 + 977·seed + 13·i)`, uniform on `[0, 64)^5`, one-hot to the **same 320 dims** |
| ⛔ NOT used | any decoder, `soft_read`, `read_from_codes`, `DreamLandscape`, `dream_rollout`, `request_posterior`, the address book, the external classifier |
| wall clock | **137.7 s / 346.0 s / 399.4 s** per seed (seeds 1–2 re-encode ~16.6k rows through the frozen encoder). CPU only |
| ⛔ CSF3 | not used; local only, as the task states |

**Pre-registration:** `.claude/outputs/gen-loss-attribution/PREREG.md`, mtime **07:58:06**, ⛔ never edited since
(⭐ R34's seventh strike: any addendum would be a NEW file, never appended in place). Production artefacts:
`attrib_s0.json` **08:02:30** → `attrib_s1.json` **08:08:17** → `attrib_s2.json` **08:14:56** → `summary.json`
**08:14** → `tables.md` **08:20**. ⇒ **every number postdates the prereg on disk.** The World A / World B table is
in it **verbatim**.

---

## 1. ⭐ THE HEADLINE TABLE — SPREAD grid, 3 seeds (full tables: `tables.md`)

⛔ `n_values_realised` on every row (R31, fourth strike). "retention" = `(acc − chance) / (acc_encoder − chance)`.

| factor | n_val | chance | **encoder** 2048-d | **best well** d_k | **code** 320-d | ⭐ **code_random** 320-d | ret. best-well | ret. code | MI max/H |
|---|---|---|---|---|---|---|---|---|---|
| `floor_hue` | 10 | 0.100 | 0.9994 ± 0.0003 | 0.9999 | **0.9966 ± 0.0019** | 0.0954 | 1.000 | **0.997** | 0.995 |
| `wall_hue` | 10 | 0.100 | 0.9997 ± 0.0001 | 1.0000 | **0.9948 ± 0.0032** | 0.1007 | 1.000 | **0.995** | 0.959 |
| `object_hue` | 10 | 0.100 | 0.9994 ± 0.0003 | 0.9999 | 0.6981 ± 0.0389 | 0.0989 | 1.001 | 0.665 | 0.651 |
| **`scale`** | **8** | 0.125 | **0.9352 ± 0.0058** | 0.8753 ± 0.0151 | **0.3069 ± 0.0042** | **0.1199 ± 0.0051** | **0.926** | **0.225** | **0.169** |
| `shape` | 4 | 0.250 | 0.9960 ± 0.0008 | 0.9895 | 0.6638 ± 0.0248 | 0.2533 | 0.991 | 0.555 | 0.428 |
| **`orientation`** | **15** | 0.067 | **0.9915 ± 0.0018** | 0.9949 ± 0.0003 | **0.2909 ± 0.0390** | **0.0649 ± 0.0011** | **1.004** | **0.243** | 0.353 |

⭐ **Read three ways and they agree.** The hue/shape rows are the **positive control**: the *same* 320-d probe on the
*same* codes reads `floor_hue` at **0.9966** and `scale` at **0.3069**. ⛔ **No capacity argument can explain a
0.997-vs-0.307 gap on an identical input** — that is the decisive answer to trap 1, stronger than any bound.

## 2. ⭐⭐ WHERE IT GOES — the partition is innocent, the codebook is not

| factor | encoder → best well | best well → code | share of the total loss at the **partition** | share at **k-means + argmax** |
|---|---|---|---|---|
| **`scale`** | 0.9352 → 0.8753 (**−7.4 %** of signal) | 0.8753 → 0.3069 (**−70.2 %**) | **9.5 %** | ⭐ **90.5 %** |
| **`orientation`** | 0.9915 → 0.9949 (**+0.4 %**, i.e. none) | 0.9949 → 0.2909 (**−76.1 %**) | **0.0 %** | ⭐ **100 %** |

⭐ **What "best well" means and why the argument is a fortiori.** The wells **partition** the 2048 coordinates, so
their concatenation *is* the feature vector up to a column permutation — the only informative point-2 statistic is the
**best single well**. That best well holds **≤ 744 of 2048 coordinates** and still reads `scale` at 0.8753 and
`orientation` at 0.9949. ⇒ **no cross-well combination is needed to read either factor.** The `code` point then has
**all five wells**, i.e. strictly more wells than the best-well point, and scores **0.3069 / 0.2909**. ⛔ Since the
code point is *not* information-starved relative to the well point in well count, the drop can only be the
**quantisation inside the wells**.

⭐ **And it is an information deficit, not an estimation one — visible IN-SAMPLE.** On its own **training** rows the
320-d code probe reaches `floor_hue` **0.998 / 1.000 / 0.999** but `scale` **0.390 / 0.398 / 0.391** and
`orientation` **0.377 / 0.409 / 0.470**. ⛔ A probe that cannot fit the training data has nothing to fit.

⚠ **What this does NOT establish.** R40.B's *mechanism* — "k-means keeps high-variance directions and
`scale`/`orientation` are low-variance in these features" — is ⛔ **not tested here.** This spoke measured **where**,
not **why**. The variance claim remains a hypothesis.

## 3. ⭐ THE FREE INSTRUMENT — the store's own MI, and it AGREES with the probe

`I(code_k ; factor_f)` in bits, max over wells, normalised by `H(f)` (SPREAD, 3-seed means):

| | floor_hue | wall_hue | object_hue | **scale** | shape | **orientation** |
|---|---|---|---|---|---|---|
| **MI max / H(f)** | 0.995 | 0.959 | 0.651 | **0.169** | 0.428 | 0.353 |
| **probe retention** | 0.997 | 0.995 | 0.665 | **0.225** | 0.555 | 0.243 |

⭐ **Spearman ρ between the two columns = 1.000 — the orderings are identical.** ⛔ **There is no disagreement to
report.** Two independent readings of the same store, one of which (MI) was already on disk and cost nothing.
`scale`'s best well carries **0.495 / 0.493 / 0.529 bits of its 3.00** — ⭐ **the flat row, seen from the address
book's own side.** (⚠ The raw-accuracy ordering of `scale` 0.307 vs `orientation` 0.291 *inverts* relative to
retention and MI, purely because their chance levels differ, 0.125 vs 0.067. ⛔ Never rank factors by raw accuracy.)

## 4. ⚠ THE CORNER — reported SEPARATELY, ⛔ never as the global profile

| factor | n_val | encoder | best well | code | code_random | ret. code |
|---|---|---|---|---|---|---|
| `floor_hue` | **2** | 1.0000 | 1.0000 | 0.9074 ± 0.1309 | 0.0509 | 0.897 |
| `wall_hue` | **2** | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 0.0972 | 1.000 |
| `object_hue` | **2** | 1.0000 | 1.0000 | 0.7593 ± 0.1762 | 0.1065 | 0.733 |
| **`scale`** | **3** | **0.7824 ± 0.1023** | 0.8889 ± 0.0196 | **0.2824 ± 0.0236** | 0.1528 | 0.241 |
| `shape` | ⛔ **1** | ⛔ **NOT APPLICABLE** | ⛔ **NOT APPLICABLE** | ⛔ **NOT APPLICABLE** | — | — |
| **`orientation`** | **3** | 1.0000 | 1.0000 | **0.2315 ± 0.1721** | 0.0417 | 0.177 |

⭐ **R31's rule fires again, as designed.** `CORNER_B/shape` realises **one** value; the raw cell would have read
`code = 0.2546 ± 0.2516` — a confident-looking number that means *"there is nothing to discriminate against."*
⛔ It is stamped, never emitted bare.
⚠ **The corner's encoder `scale` drops to 0.7824 ± 0.1023** while every other factor is at 1.000 — consistent with
R16/`gen-feature-bridge`'s **photometric** account of the corner (the three-hue clause collapses object/background
contrast) and ⛔ **not** with a compositional one. On the matched-seen companion the same number is
**0.8372 ± 0.0446**, i.e. the deficit is present on *seen* combinations too.
⭐ **P11 held:** corner `code` `scale` **0.2824** < spread **0.3069**, corner `orientation` **0.2315** < spread
**0.2909** — the corner is *slightly* worse, ⛔ nowhere near the near-zero a corner-only measurement would have
suggested. **The loss is global, exactly as R34 said.**

## 5. ⛔ THE CAPACITY BOUND FAILED — reported as a failure, with its cause measured

**P9 predicted** the non-parametric tuple-lookup bound would be **≥** the linear code probe. ⛔ **FALSIFIED:**
`scale` **0.1797 ± 0.0046** vs the linear probe's **0.3069**; `orientation` **0.1413** vs **0.2909**.
**Cause, measured, not guessed:** lookup **coverage on SPREAD is 0.461 / 0.473 / 0.386** — the store assigns
**8,893 / 8,768 / 9,506 distinct tuples to 12,288 training rows**, so over half the test rows land on a tuple never
seen and fall back to the global majority. ⇒ ⛔ **the "upper bound" is not achieved and must not be quoted as one.**
⭐ The capacity objection is instead settled by the two controls that *did* work: the random code (≈ chance) and the
hue positive control (0.997 on the identical 320-d input). ⭐ **And the failure is itself a finding:** a 30-bit code
that is **near-injective on the spread grid** (3,598–3,723 distinct codes for 4,096 rows) is ⛔ **not** short of
channel — reconciliation item 3.

## 6. Pre-registered predictions vs measurement (PREREG §3, verbatim ids)

| id | predicted | 80 % interval | measured | verdict |
|---|---|---|---|---|
| **P1** | encoder `scale` 0.75 | [0.55, 0.92] | **0.9352 ± 0.0058** | ⛔ **FALSIFIED — above the interval.** I under-predicted the encoder badly |
| **P2** | encoder `orientation` 0.60 | [0.35, 0.85] | **0.9915 ± 0.0018** | ⛔ **FALSIFIED — above** |
| **P3** | best-well `scale` 0.60 | [0.35, 0.85] | **0.8753 ± 0.0151** | ⛔ **FALSIFIED — above** |
| **P4** | best-well `orientation` 0.45 | [0.20, 0.75] | **0.9949 ± 0.0003** | ⛔ **FALSIFIED — above** |
| **P5** | code `scale` 0.20 | [0.13, 0.35] | **0.3069 ± 0.0042** | ⭐ **HELD** (upper half of the interval) |
| **P6** | code `orientation` 0.17 | [0.07, 0.35] | **0.2909 ± 0.0390** | ⭐ **HELD** |
| **P7** | ⭐ code_random `scale` 0.128 | [0.11, 0.16] | **0.1199 ± 0.0051** | ⭐ **HELD — the spoke is not wrong** |
| **P8** | ⭐ code_random `orientation` 0.068 | [0.05, 0.10] | **0.0649 ± 0.0011** | ⭐ **HELD** |
| **P9** | lookup ≥ linear code probe | gap ≤ 0.15 | **−0.127 / −0.150** | ⛔ **FALSIFIED** (§5; cause measured) |
| **P10** | code hues ≥ 0.70 | [0.55, 0.95] | 0.9966 / 0.9948 / **0.6981** | ⭐ **HELD on the interval**; ⚠ `object_hue` misses the 0.70 *point* by 0.002 |
| **P11** | corner code `scale` < spread | — | 0.2824 < 0.3069 | ⭐ **HELD** |
| **P12** | MI `scale` max ≤ 0.6 bits | [0.2, 1.0] | **0.506 bits** of 3.00 | ⭐ **HELD** |
| **§5 word** | "in-between" | — | **in-between**, 3/3 seeds | ⭐ **HELD — ⛔ for the OPPOSITE reason.** I predicted in-between because the *encoder* would miss 0.90; in fact the encoder clears it and the *code* misses "≈ chance" |
| **§5 split** | partition ~0.15, k-means ~0.40; **k-means the larger culprit** | — | partition **9.5 %**, k-means **90.5 %** (`scale`) | ⭐ **Direction HELD, magnitude under-predicted** — the partition is even more innocent than I guessed |

⭐ **8 held, 5 falsified, and the falsified ones are the informative half.** Every encoder/well prediction was too
low: I derived them from a *decoder→CNN-probe* chain and assumed a linear head on raw features would lose a lot. It
does not — these features are **linearly** factor-decodable almost perfectly.

## 7. ⚠ Limitations, stated rather than buried
- **L1 — 1.6–2.0 % of the probe's test rows are the encoder's own fit rows** (64/82/77 of 4,096 land in
  `bridge_train`). ⛔ Cannot move 0.9352 materially, but it is nonzero and is now on the record. `SPREAD ∩ store_train`
  and `SPREAD ∩ eval_b` are **exactly 0**.
- **L2 — the encoder probe overfits in-sample** (train aggregate 0.9998 / 0.9999 / 1.0000 at 2048 dims). The reported
  number is the **held-out** one (0.986 aggregate), on 4,096 rows whose factor combinations are disjoint from train
  (the grid is a bijection), so this is an estimation caveat, ⛔ not an inflation of the headline.
- **L3 — the probe is LINEAR.** "The encoder has `scale`" means *linearly decodable at 2048 dims with 12,288 training
  rows*. A lower-capacity or non-linear-necessary criterion could return a different answer, and R40.E's **swap-
  within-the-family** clause is untested.
- **L4 — `code` is all five wells; `best well` is one.** The chain is ⛔ **not** strictly nested; §2 spells out why the
  conclusion is a fortiori rather than a decomposition.
- **L5 — shapes3d only.** ⛔ Nothing here is claimed for dSprites.

## 8. Config drift (declared)
Two defaults were raised **after** the `--quick` plumbing smoke and **before** the production run, both purely
capacity/precision and neither part of the protocol: `n_spread_train` 6144 → **12288**, `n_spread_test` 2048 →
**4096**, `probe_steps` 1500 → **3000**. Reason: the smoke showed the encoder probe hitting train-accuracy 1.0 on 512
rows, i.e. a data-starved probe would have **understated** the encoder and biased the result toward World B. ⛔ No
quick number is reported anywhere. The shipped defaults are the ones every number above used.

## How I verified
```
# plumbing smoke (⛔ numbers discarded, separate --out so it cannot poison the artefacts)
python -m chlu.experiments.exp_gen_attribution --quick --out .../scratch/gen-loss-attribution/quick   # EXIT=0
# production, 3 seeds
python -m chlu.experiments.exp_gen_attribution                                                        # EXIT=0
```
with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`,
`CHLU_CLAUDE_ROOT=/Users/user/Desktop/CHLU/.claude`,
`CHLU_GEN_BRIDGE_ROOT=/Users/user/Desktop/CHLU/.claude/outputs/gen-feature-bridge/runs` (a worktree has no `.claude/`).

**Targeted tests, run from the worktree with `CHLU_GEN_DATA_ROOT` set (R31's fifth strike):**
`tests/test_gen_attribution.py tests/test_config.py tests/test_gen_store.py tests/test_gen_compose.py`
⇒ ⭐ **95 passed · 0 skipped · 0 failed (3:51).** **Zero skips — no skip to name.**
(`tests/test_gen_attribution.py` alone: **22 passed · 0 skipped · 0 failed**.)
⛔ **No full suite started** — the Hub is running one.
`ruff check` clean on all four touched files. ⚠ `ruff format --check` would reformat them — and would equally
reformat `chlu/core/gen_store.py`, `chlu/experiments/exp_gen_navigate.py` and `tests/test_gen_store.py`; the repo is
not format-clean, so I matched the house style rather than reformatting shared-adjacent files.

**One test failed on first run and was fixed:** `test_verdict_is_in_between_and_is_NOT_rounded` asserted
`leaning == "World A"` for `acc_code = 0.20` against a floor of `0.13 + 0.05`; by the pre-registered rule that code is
**not** at chance, so the correct leaning is `"neither"`. ⭐ The **test** was wrong, ⛔ not the rule — the rule had
already been filed in `PREREG.md` and was not touched. The test now pins both sub-cases.

## Artefacts
`.claude/outputs/gen-loss-attribution/` — `PREREG.md` (filed 07:58, ⛔ unedited) · `summary.json` ·
`tables.md` (9 sections, incl. per-seed, in-sample, probe-convergence, full MI matrices, lookup coverage) ·
`attribution.png` · `runs/shapes3d_recon/attrib_s{0,1,2}.json`.

## Git footprint
- **Branch** `agent/experiment-engineer/gen-loss-attribution` off `main @ 994ca9b`; **worktree `../CHLU-gen-attrib`,
  left parked** (commits verified visible from the primary repo: `git -C /Users/user/Desktop/CHLU log main..<branch>`
  lists all four).
- `64a7e3d` `chlu/config.py` — **one contiguous append-only block at EOF**, +112 lines, ⛔ 0 deletions, no
  `CHLUConfig` field / `load_config` / `save_config` site. `ast.parse` clean.
- `42f72d6` `chlu/eval/gen_attribution.py` (new)
- `77763c8` `chlu/experiments/exp_gen_attribution.py` (new)
- `d1b2dd4` `tests/test_gen_attribution.py` (new)
- ⛔ Nothing else touched: `gen_store.py`, `gen_compose.py`, `gen_dream.py`, `gen_bridge.py`, `chlu/eval/rivals/`,
  `chlu/data/`, `scripts/csf3/` are **unmodified**. ⛔ **No CLI hook added** — `chlu/cli/experiment_cmd.py` is outside
  the files I own, and the precedent is that recent gen experiments (`exp_gen_navigate`, `exp_gen_compose`'s newer
  siblings) run via `python -m`. **Flagging it: if the Hub wants a `chlu exp-gen-attribution` verb, that is a
  separate one-line task.**
- No conflicts. Rebase onto `main` was a no-op (`main` has not moved from `994ca9b`).
- ⛔ Not pushed, no PR, left for review.

## Open questions / follow-ups / risks
1. ⭐ **The obvious next measurement, ~30 min, not run here (out of dial):** is R40.B's *mechanism* right? Project the
   features onto their per-well principal axes and ask whether `scale`/`orientation` live in the **low**-variance
   directions that k-means discards. That would turn "k-means threw it away" into "k-means threw it away *because*".
2. ⚠ **The fix implied by item 2 of the reconciliation list is a WRITE-side one and the Head has already ruled the
   quantiser is KEPT as the write-side organiser** (R40.C). ⛔ That ruling and this finding are in tension worth a
   sentence at the design note: the organiser we are keeping is the component that loses 90 % of the signal.
   ⛔ Not mine to resolve.
3. ⚠ **`orientation`'s corner numbers ride on 72 rows and 3 realised values** (`code` 0.2315 ± 0.1721 — the sd is
   74 % of the mean). ⛔ Not quotable on its own; the SPREAD row is.
4. ⚠ The `code_random` control lands *slightly below* chance (0.1199 vs 0.1250; 0.0649 vs 0.0667). Expected for a
   probe fitting noise, but worth one line if anyone quotes it as "exactly chance".

## Proposed handover updates (for the Hub)
- **§7 NEW — attribution finding (this spoke).** On the spread grid, the frozen `recon` encoder holds `scale`
  **0.9352 ± 0.0058** and `orientation` **0.9915 ± 0.0018** (linear probe, held-out rows, 3 seeds); the store code
  retains **0.225 / 0.243** of the above-chance signal; **the well partition costs 9.5 % / 0 % and the per-well
  k-means codebook costs 90.5 % / 100 %**. Random-code control at chance (0.1199 / 0.0649). Verdict
  **in-between — World B rejected**. ⛔ Never write this as "the 30-bit read is too narrow": the code is
  near-injective on the grid (3,598–3,723 distinct codes on 4,096 rows).
- **§7 NEW — standing trap for every worktree gen run.** `CHLU_GEN_DATA_ROOT` alone is **not** enough. A worktree also
  needs **`CHLU_CLAUDE_ROOT`** and **`CHLU_GEN_BRIDGE_ROOT`**, or the run dies at `bridge_dir` (it did, for me, on the
  first attempt). ⭐ Worth folding all three into the task template beside R31's fifth strike.
- **§7 amend — the loss-profile shorthand.** *"The read carries HUE and SHAPE"* is imprecise: `object_hue` 0.665 and
  `shape` 0.555 retention sit **between** the other hues (≈0.996) and `scale`/`orientation` (≈0.23). The profile is
  graded.
- **§3 config — new block `GenAttributionConfig`** (append-only at `config.py` EOF, reached via
  `gen_attribution_config(config)`; ⛔ deliberately **not** a `CHLUConfig` field, same rationale as
  `gen_navigate_config`). Defaults: 3 seeds, `n_spread_train=12288`, `n_spread_test=4096`, `probe_steps=3000`,
  `probe_batch=512`, `probe_lr=3e-3`, `world_a_encoder_floor=0.90`, `world_a_code_slack=0.05`.
- **Worktree ledger:** `../CHLU-gen-attrib` added, **parked** (not removed), 0 uncommitted files.
