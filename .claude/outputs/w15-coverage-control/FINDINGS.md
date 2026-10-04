# w15 spoke/coverage-control — FINDINGS

**Is the ladder's rise depth, or coverage?** Branch `w15/coverage-control`, base `base/iter8 = 148f3b7`.
`chlu.__file__ = C:\chlu\worktrees\CHLU-w15-coverage-control\chlu\__init__.py`. I asserted it before any number and wrote it into every `val/seed*_K32/LEG.json`.
Every number below names the file it came from. **Scored runs spent: 0.**

---

## 0. THE ANSWER

⭐⭐ **For rungs n = 3…6 the rise is coverage, and coverage is the only thing it can be.** Restricting n = 6's fit pool to n = 3's train index gives an object that **is rung n = 3**:
- the same pool draw, so the same store and address book;
- the same judge;
- the same acceptance;
- the same novelty mask;
- the same corner rows.

So on the test ladder, "n = 6 at n = 3's coverage" equals n = 3: **61.614** against n = 6's **66.603** (K\* = 32, 3 seeds). The whole n = 3 → n = 6 rise of **+4.989** is fit-pool coverage, and none of it is depth.

- ⛔ **The test-split number for that cell is DERIVED, not measured.** A new configuration on test would be a scored run, and `composition` is exhausted (§1). The derivation is an identity, and I checked all four of its conditions on the instrument (§3).
- ⭐ **Val measures it directly.** C equals A to every printed decimal, on every seed, for the store and for `learned_1x` alike. B sits above A on all 3 seeds.
- ⚠ **The val rise is small.** Val is near its ceiling (24 tuples per corner), and the rise is below the pre-declared 1.0 margin on 2 of 3 seeds. The PREREG verdict on val is therefore **"NO RISE TO DECOMPOSE ON VAL"**, and I report it as that.

**⇒ The Head's note stands, and the reason is stronger than "they differ in coverage": for n ≥ 3, fit-pool coverage is the only thing that differs.**

### The inference note (verbatim, for the paper)
> *We beat the naive fair rivals at every n; the CLU's own 2-way → 6-way rise is not to be inferred, because rungs 3–6 share corners and differ in coverage.*

⛔ **The paper's ladder table is unchanged.** This spoke adds one control row (§2) and the note above.

---

## 1. Budget: the ledger, verified BEFORE any number

`/c/chlu/SCORED-LEDGER.md` header rule: *"Three scored runs per TARGET … counted from 2026-09-16"*. The `composition` rows since then are:

| line | who | label |
|---|---|---|
| L8 | wave 2 | 1/3 |
| L11 + L12 | hub/factor-wells | one run, the same run with a clock correction |
| L15 | hub/factor-wells | 3/3 |
| L27 | w5 compose-field | "1/3" |
| L30 + L32 | w6 compose-physics | "2/3 SPENT; 3/3 BANKED" |

- That is **5 spends against a cap of 3**.
- The "2/3" reading comes from a count restarted at w5. No written rule authorises that restart.
- ⇒ **`composition` is DECLARED EXHAUSTED. 0 runs are spent, and only the val leg ran.**
- I appended a **NOTE row (0 runs)** to the ledger at `2026-09-23T07:13:25Z`, before any number, naming `outputs/w15-coverage-control/PREREG.md`.
- ⚠ If the Head rules that the w5 restart is valid, 1 run remains. By §3, it would buy only a re-measurement of an identity.

---

## 2. The control row (for the ladder table's appendix)

K\* = 32 `CovContentStore` on the label-free `ward_balanced` cut. Estimand: `distinct_valid_novel_tuples` per corner.

| rung | fit pool drawn from | n seeds | test: mean (per seed) | how the test number was obtained | val: mean (per seed) |
|---|---|---|---|---|---|
| n = 3 | train_3 | 3 | **61.614** (60.150 / 62.783 / 61.908) | published | **22.767** (21.767 / 23.050 / 23.483) |
| **n = 6 at n = 3's coverage** | **train_3** | 3 | **61.614** (= n = 3, identical by construction) | ⛔ **DERIVED, not measured** (§3) | **22.767** (21.767 / 23.050 / 23.483) — **measured, = n = 3** |
| n = 6 | train_6 | 3 | **66.603** (65.908 / 66.250 / 67.650) | published | **23.478** (22.967 / 23.617 / 23.850) |

Sources for the table:
- Test numbers: `/c/chlu/outputs/w12b-nr6-and-kstar/kstar_ladder/{3,6}_seed{0,1,2}_K32/rows_single_design_K32.json` `/summary/distinct_valid_novel_tuples_mean`, collected into `COVERAGE-CONTROL.json` `/test_context`.
- Val numbers: `val/seed{s}_K32/rows_{A,C,B}_store_pool{3,3,6}_est{3,6,6}.json` `/summary`, collected into `COVERAGE-CONTROL.json` `/summary_val/store`.

Per-corner ceiling: test 72 rows per corner, val 24.

### Decomposition on val (`COVERAGE-CONTROL.json` `/summary_val`, `/val/{s}/decomposition`), 3 seeds

| quantity | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| rise B − A | +1.200 | +0.567 | +0.367 | +0.711 |
| **coverage B − C** (store pool alone) | **+1.200** | **+0.567** | **+0.367** | **+0.711** |
| **mask C − A** (estimand masks alone) | **0.000** | **0.000** | **0.000** | **0.000** |
| coverage share s = (B − C)/(B − A) | 1.00 | 1.00 | 1.00 | 1.00 |
| paired B − C over 60 corners: corners +, − (sign test p) | 17+, 7− (0.064) | 14+, 4− (0.031) | 9+, 5− (0.424) | — |
| diagnostic D (pool 6, n = 3 masks) | 22.967 = B | 23.617 = B | 23.850 = B | = B |

`learned_1x` shows the same structure (`/summary_val/learned_1x`): C = A on every seed (22.100 / 21.800 / 23.333), and B − A = −0.150 / +0.917 / +0.567.

---

## 3. ⭐⭐ Why n = 6 at n = 3's coverage IS n = 3 — the identity, checked

Source: `IDENTITY-CHECK.txt`. Each condition was checked on the instrument, and none of the checks produces a model output on the test split.

1. **The pool.** `fw.fit_pool` is one draw, `rng(pool_rng_seed = 20260919)`, from the train index, and it is the same for every seed. Recomputing the pool from train_3 reproduces the w12 ladder's rung-3 pool exactly: `True`. It also reproduces the rung-6 pool from train_6: `True` at seed 0 in the check file, and at every seed in each leg's `LEG.json` `/audit/pool_recompute_equals_w12_test_pool`. ⇒ The store and the address book are n = 3's.
2. **Acceptance.** The test corners of both rungs are accepted at **1.000000** under the train_3 bitmap and under the train_6 bitmap.
3. **Novelty.** `judge_valid` requires all six factors to equal the request. A valid read-back is therefore a held-out corner tuple, which counts as novel under either rung's mask. Measured: `distinct_valid == distinct_valid_novel` in **3,600 / 3,600** published test per-corner rows (K\* rungs 3 and 6) and **1,800 / 1,800** val rows.
4. **The probe.** Rung 3 and rung 6 test corners are the identical **8,640 rows**.

⇒ The only thing that separates rung 3 from rung 6 is the store's fit pool. The mask channel is **identically zero**, as the C − A row above shows. **The coverage share of the n = 3 → 6 rise is 1 by construction.** Val confirms this empirically: C = A to every printed decimal on 3 of 3 seeds, for the store and for `learned_1x`.

The test-side coverage effect is B − C (= B − A) = **+5.758 / +3.467 / +5.742**, mean **+4.989** (`COVERAGE-CONTROL.json` `/test_context/C_test_DERIVED_not_measured/coverage_B_minus_C_test`). That is DERIVED, 3 seeds.

⚠ **What the derivation assumes.** A refit of the same pool reproduces the same store and address book. Within a process, val shows it does (C and A come from the same object). Across processes I did not re-verify it on test, because that would be the scored run.

---

## 4. What the ladder can and cannot say about depth

- **n = 3…6:** there is no depth contrast; §3 shows only the pool differs.
- **n = 2 vs n = 6** is the only genuine depth contrast, and it also changes the corner:
  - n = 2 has 60 corners × 36 rows; n = 6 has 120 × 72 (`w12-composition-ladder/FINDINGS.md` §5).
  - The raw 2 → 6 "rise" at K\* (32.644 → 66.603, `w12b-nr6-and-kstar/KSTAR-LADDER.md`, seed-matched table, 3 seeds) is mostly the doubled ceiling.
  - Ceiling-normalised, it is 32.644 / 36 = **0.907** against 66.603 / 72 = **0.925** (my arithmetic on those two files).
  - ⇒ Normalised by corner size, the "2-way → 6-way rise" is about **+0.02**. The note's instruction not to infer it is correct on this ground too.
- ⛔ I did not run an n = 2 coverage control. The n = 2 rung carves a different family (pair cells), so "n = 2 at n = 6's coverage" is not defined by this instrument.

---

## 5. The instrument (what is held fixed)

- **Store.** K\* = 32 `CovContentStore` on the `ward_balanced` label-free cut, A = 63, with the `R.READ_PINS` address book (rank 128, 40 sweeps), built by `w12b_ladder_kstar.build_cov_store` / `address_book` unchanged.
- **Byte ledgers** (crossover hub four-part convention: atoms + covariance/count + address book + coupling/hole, never netted), from `COVERAGE-CONTROL.json` `/val/{s}/store_bytes`:

  | seed | pool 3 | pool 6 |
  |---|---|---|
  | 0 | 2,142,098 B | 2,143,658 B |
  | 1 | 2,145,738 B | 2,148,338 B |
  | 2 | 2,149,378 B | 2,153,018 B |

  On val, `learned_1x` is matched to 1× that total.
- **Estimand and judge.** `fw.per_corner_scores`, with `FastJudge` asserted element-for-element identical to `fw.Judge` in every leg (`LEG.json` `/fastjudge_positive_control`).
- **Controls (charter C), post-fit form as in w12b.** `landscape_removed`, `landscape_shuffled` and `random_atoms` read **0.0000 on every cell (A, B, C, D) and every seed** (`/val/{s}/controls`). The gate passes, so the store is not decorative on val.
- **Val geometry** (no val set existed; the ladder's full sweep uses every coordinate option):
  - The ladder's own builder, with `scale` swapped to the middle value **0.964** (index 3), which no test cell touches.
  - val n = 3: 6 cells / 18,000 rows, train 162,000.
  - val n = 6: 60 cells / 1,440 rows, train 277,920.
  - Both probe the identical **60 corners × 24 rows**.
  - Every val row is asserted to be a TRAIN row of both test rungs.
  - Source: `val/seed*_K32/LEG.json` `/audit`.
- **Pool size.** 131,072 in every cell; the pin is served.

### ⚠ One failed launch, disclosed (`AMENDMENT-1.md`)

The PREREG's first val geometry, a width-2 middle scale band, read **0.0000 with judge_acc 0.0000 on every cell, controls included**. Together with the test n = 3 cells, it removed every row of the ladder's object hues from train, so the acceptance bitmap refused every val request. That was an instrument fault, not a result. The rows are archived in `val_INVALID_amend0_width2/` and the logs in `logs/amend0/`.

The amendment changed one thing: the band width, from 2 to 1. It also added a hard no-absent-value assertion. The dry run then showed accept fraction 1.0 (`logs/amend1_dryrun.txt`). No bar moved.

---

## 6. The PREREG bars, applied verbatim (`PREREG.md`; `COVERAGE-CONTROL.json` `/bars`)

| bar | outcome |
|---|---|
| control gate: every cell's controls < store − 1.0, every seed | **PASS**: A, B, C and D all True |
| a rise exists: B − A > 1.0 on every seed | **FAIL**: +1.200 / +0.567 / +0.367 |
| COVERAGE bar: s ≥ 0.5 and B − C > 1.0 on every seed | not met, because B − C = B − A < 1.0 on 2 seeds |
| DEPTH bar: \|B − C\| ≤ 1.0 on every seed | not evaluated as a verdict: the rise bar failed first |
| **VERDICT (val)** | **NO RISE TO DECOMPOSE ON VAL** |

I report the verdict as declared. What val does show, without a bar:
- The mask channel is exactly zero.
- Everything B gains over A, it gains over C.
- The direction is the same on all 3 seeds.

The decisive evidence is the identity in §3, not the val magnitude. The val magnitude is compressed by the 24-tuple ceiling: A is already at 0.949 of it.

---

## 7. DECLARED NOT-RUN

- **The test-split C cell.** It would be a scored run, and `composition` is exhausted (§1). Its value follows from the identity in §3 and is printed as DERIVED.
- **`pca_tensor` / the K = 5 ladder store.** Not the main-text object. The identity in §3 does not depend on the store, so it would read C = A there too.
- **The stronger `random_atoms`** (random atoms with the address book refit). About 1.8 CPU-hours per book on this shared box. The post-fit form already collapses to 0.0000.
- **The naive fair rivals (524,288 B) on val.** Not rerun. On test they are exactly constant across n = 3…6 (`w12-composition-ladder/FINDINGS.md` §5b). The note's clause "we beat the naive fair rivals at every n" is carried from the published tables, and this spoke did not re-verify it.
- **Seeds 3 and 4.** The frozen decoder, classifier and rivals do not exist for them (`w12-composition-ladder/FINDINGS.md` §1).
- **An n = 2 coverage control.** Not defined by this instrument (§4).

---

## 8. What would have proved this wrong

| falsifier (task file) | outcome |
|---|---|
| n = 6 at n = 3's coverage **matches n = 6** ⇒ the rise is depth and the caveat is too strong | **did not happen.** It matches **n = 3** exactly, by identity on test and by measurement on val (3/3 seeds). |
| it **drops to n = 3's level** ⇒ coverage, and the note stands | ⭐ **HAPPENED**, in the strongest form: it is not only at n = 3's level, it is n = 3's object. |
| (mine) the mask channel C − A is non-zero ⇒ "coverage" would mix store and scoring | **did not happen:** 0.000 on every seed, and 0 novelty mismatches in 5,400 rows |

---

## 9. Artefacts (`/c/chlu/outputs/w15-coverage-control/`)

| file | what |
|---|---|
| `COVERAGE-CONTROL.json` | **acceptance artefact**: all cells per seed, decomposition, bars, test context, the derived C_test, the note verbatim |
| `IDENTITY-CHECK.txt` | the four-condition proof that n = 6 at n = 3's coverage equals n = 3 |
| `PREREG.md` (+ `.timestamp.txt` 07:13:25Z), `AMENDMENT-1.md` (+ timestamp) | the declarations |
| `val/seed{0,1,2}_K32/` | `LEG.json` (audit, pools, ledgers), `rows_*.json` per cell/control/learner (per-corner), `pool{3,6}/` stores' address books |
| `val_INVALID_amend0_width2/`, `logs/amend0/` | the failed first geometry, kept |
| `logs/` | `val_seed*.log`, `amend1_dryrun.txt` |
| `LOG.md`, `PIDS.txt`, `START_UTC.txt`, `END_UTC.txt` | the record |
| `w15-coverage-control.bundle` | branch `w15/coverage-control` |

Code on the branch:
- `scripts/w15_coverage_leg.py`: the val leg.
- `scripts/w15_coverage_report.py`: the aggregation and bars.
- `scripts/w15_env.sh`: the env, including the Windows-form `PYTHONPATH` fix. `env_w3.sh`'s mixed-form path lost the site-packages under git-bash; `chlu` still resolved inside the worktree, but equinox did not import.
