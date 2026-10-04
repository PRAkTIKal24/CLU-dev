# w15 spoke/crossover-seeds — FINDINGS

**The one job:** run K = 8, 10, 12, 16, 20, 24, 28 at seeds 1 and 2 (14 legs) on the same
frozen rule and the same instrument as the w12 crossover, and regenerate the crossover
figure with **n = 3 at every K**.

**Status: COMPLETE.** All 14 legs ran. Every K in the sweep now carries n = 3 seeds
{0, 1, 2}. **Scored runs spent: ZERO** — the whole spoke is `val`, the free selection
condition (charter A.2), and every number is `SMOKE_NOT_A_CLAIM`. No row was written to
`/c/chlu/SCORED-LEDGER.md` because no scored run was spent.

Acceptance artefacts, all in `/c/chlu/outputs/w15-crossover-seeds/`:

| artefact | sha256 |
|---|---|
| `CROSSOVER-3SEED.json` | `c5d68ee3abf1e203436f00360aa0ddaae27c028776d519db515ffc1d631a9290` |
| `CROSSOVER-3SEED.md` | `509f03128e77f408d12b3503a9d0ef7947d1423b844a4ffc69c5628b5775b176` |
| `fig_crossover_3seed.pdf` (vector) | `e9c1acf4394e6d250b9350f3a15deaf1711d1cb46017af4e7efb1c10a4d993ce` |
| `fig_crossover_3seed.png` | `0b6b4e4fd2cdfb780513cc3897010caa00e457050b960dffcf7ca0a4bc1ab14a` |

Hashes from `ARTEFACTS.sha256`, produced by `scripts/w15_rebuild_check.sh`, which builds
all four twice from the same `row.json` files and diffs the hashes:
**BYTE-IDENTICAL ON REBUILD: PASS** (`REBUILD-pass1.sha256` vs `REBUILD-pass2.sha256`).

---

## THE BYTE CONVENTION — named here, in the JSON, and on the figure

> **HONEST FOUR-PART TOTAL, uncompressed raw array bytes, SUMMED AND NEVER NETTED:**
> (1) atoms = `sum_k modes_k.size × 4 B` (float32, the deployed content);
> (2) covariance + count = `sum_k stat_floats_k × 8 B` (float64 — the precision actually
> retained; counting it at 4 would be netting by dtype) + 11,488 B of count table;
> (3) address book = `TensorAddress.byte_ledger()['address_bytes']`;
> (4) coupling/hole table = 5,288 + 1,322 = 6,610 B.
> No entropy coding, no deduplication, no part subtracted from any other.
> The 524,288 B pin is a **REFERENCE COLUMN** for continuity with waves 5–8, **not** a
> budget in this sweep.

**Notation.** `K` = wells. `n` in this artefact is the **seed count** and is a column on
every table. (`n` = factors jointly composed is a different quantity and is not used
here.) Every table below carries its seed count.

---

## 1. The headline: the U-shape is CONFIRMED, and the minimum does not move

The test was declared in `LOG.md` section 5 at **2026-09-22T21:31:05Z**, when
`/c/chlu/outputs/w15-crossover-seeds/val_sweep/` held **no** `row.json` for any cell —
the four lanes had been running under two minutes and a single address-book fit takes
500–1,830 s. Seed 0's totals were already known (they are w12's, and they are what the
sentence was built on); the test decides the sentence on seeds 1 and 2, whose numbers did
not yet exist.

- **S1** — U-shaped iff `T(·, s)` strictly decreases to its minimum and strictly increases
  after it.
- **S2** — "near K = 24" iff `argmin_K T(K, s) ∈ {20, 24, 28}`, the grid point and its two
  ladder neighbours. The ladder was not re-spaced.
- **S3** — claimable at n = 3 iff S1 and S2 hold on **all three** seeds.
- **S4** — resolvability, reported beside S3 and never instead of it: the gap to each
  neighbour must exceed the seed band.

Honest four-part total, bytes, per K per seed
(`CROSSOVER-3SEED.json:u_shape.per_seed.<s>.totals_by_K`):

| seed | K=5 | K=8 | K=10 | K=12 | K=16 | K=20 | **K=24** | K=28 | K=32 | argmin | S1 | S2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4,118,034 | 2,959,602 | 2,606,754 | 2,393,714 | 2,177,330 | 2,100,786 | **2,094,130** | 2,126,890 | 2,143,138 | **24** | yes | yes |
| 1 | 4,118,034 | 2,959,602 | 2,606,754 | 2,393,714 | 2,177,330 | 2,095,586 | **2,080,090** | 2,107,130 | 2,136,378 | **24** | yes | yes |
| 2 | 4,118,034 | 2,959,602 | 2,606,754 | 2,393,714 | 2,177,330 | 2,100,266 | **2,085,810** | 2,117,530 | 2,142,098 | **24** | yes | yes |

- per-seed minima `{0: 24, 1: 24, 2: 24}`; distinct minima `[24]`
- **S1 = True**, **S2 = True**, **S3 VERDICT = CONFIRMED**
- **S4:** at K = 24 the seed band is **sd 7,060 B**; the gap to K = 20 is **12,203 B** and
  to K = 28 is **30,506 B**. Both clear the band, so `minimum_resolvable = true`.

**The sentence may be said at n = 3: the honest four-part total is U-shaped in K and its
minimum sits at K = 24 on every seed, with the minimum resolvable against the seed band.**
The falsifier named in the task file — "if the minimum moves materially across seeds, the
sentence cannot be claimed" — did not fire. (`CROSSOVER-3SEED.json:u_shape`)

**Why the minimum is so stable, stated rather than left to look like luck.** Three of the
four parts do not depend on the seed at all. The `ward_balanced` cut splits the same
2,048 − 67 = 1,981 live channels into K near-equal wells whatever the seed, so the atom
block (524,288 B at A = 64) and the covariance+count block are fixed by K alone; the
coupling/hole table is a constant 6,610 B. **Only the address book carries seed variance**,
and only where a well deploys a rank below 64 — which first happens at K = 20. Hence the
seed sd on the total is exactly **0 B at K = 5, 8, 10, 12 and 16** and opens to 2,864 /
7,060 / 9,885 / 3,640 B at K = 20 / 24 / 28 / 32
(`CROSSOVER-3SEED.json:per_K.<K>.honest_total_bytes.sd` and `.parts_across_seeds`). The
U-shape is therefore a property of the K-dependence of two parts pulling in opposite
directions, with a small seed-dependent wobble on one of them:

| part | K = 5 (n=3) | K = 24 (n=3, mean) | K = 32 (n=3, mean) |
|---|---|---|---|
| 1 atoms | 524,288 | 524,288 | 516,096 |
| 2 covariance + count | 3,391,552 | 735,328 | 560,608 |
| 3 address book | 195,584 (sd 0) | **820,451 (sd 7,060)** | 1,057,224 (sd 3,640) |
| 4 coupling / hole table | 6,610 | 6,610 | 6,610 |
| **honest total** | **4,118,034** | **2,086,677** | **2,140,538** |

The covariance falls like `1/K` and the address book grows with K; K = 24 is where they
cross. **The object never reaches the retired 524,288 B pin at rank 64: the floor over
the whole sweep and all three seeds is 3.9675 × the reference, at K = 24 seed 1**
(`CROSSOVER-3SEED.json:per_K.24.per_seed.1.ledger_four_part.times_reference_pin`).

---

## 2. The finding that corrects w12: seed 0 was the pessimistic seed, and the
## composition-vs-K story inverts at n = 3

w12 ran the seven interior K at **seed 0 alone**. Composition at n = 1 fell below the K = 5
reference at K = 8…20. At n = 3 it does not fall below K = 5 **anywhere**.

| K | n | val comp mean | seed sd | seed band | drop vs K=5, **n = 3** | drop vs K=5, **w12 n = 1** |
|---|---|---|---|---|---|---|
| 5 | 3 | 0.8953 | 0.0065 | [0.8912, 0.9028] | +0.0000 | +0.0000 (n=3) |
| 8 | 3 | 0.9037 | 0.0197 | [0.8837, 0.9230] | **−0.0085** | +0.0116 (n=1) |
| 10 | 3 | 0.9014 | 0.0211 | [0.8825, 0.9242] | **−0.0062** | +0.0127 (n=1) |
| 12 | 3 | 0.9082 | 0.0141 | [0.8929, 0.9207] | **−0.0129** | +0.0023 (n=1) |
| 16 | 3 | 0.9111 | 0.0203 | [0.8906, 0.9311] | **−0.0158** | +0.0046 (n=1) |
| 20 | 3 | 0.9178 | 0.0285 | [0.8877, 0.9444] | **−0.0226** | +0.0075 (n=1) |
| 24 | 3 | 0.9223 | 0.0235 | [0.8981, 0.9450] | **−0.0270** | −0.0029 (n=1) |
| 28 | 3 | 0.9267 | 0.0276 | [0.8976, 0.9525] | **−0.0314** | −0.0023 (n=1) |
| 32 | 3 | 0.9290 | 0.0259 | [0.9016, 0.9531] | **−0.0338** | −0.0338 (n=3) |

Sources: `CROSSOVER-3SEED.json:per_K.<K>.val_composition` and `.drop_from_K5`;
w12 column from `/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json:per_K.<K>.drop_from_K5`
with that file's own `n_seeds` printed beside it.

**Every drop at n = 3 is ≤ 0: the largest drop anywhere in the sweep is 0.0000, at K = 5
itself.** w12's "largest drop anywhere is 0.0093 at K = 10" was a seed-0 artefact. The
partition can be made as fine as K = 32 without any loss of composition at all on three
seeds — composition rises monotonically from 0.8953 to 0.9290 across the ladder.

**K\* and the crossover under the frozen rule, applied unchanged at n = 3**
(`CROSSOVER-3SEED.json:K_star`, `.K_star_candidates`, `.K_crossover`):

- **K\* = 32.** The satisfying set is now **every K in the sweep**, `[5, 8, 10, 12, 16, 20, 24, 28, 32]`,
  because no K drops below K = 5 at all. w12 reached K\* = 32 too, so the selection is
  unchanged — but it is reached for a different reason and from a much weaker position.
- **K_crossover = None.** No K drops more than 0.10 below K = 5. ⛔ The range was **not**
  extended to manufacture one, exactly as `SELECTION-RULE.md` pre-committed.

### 2b. ⛔ The frozen rule is NOT resolvable at n = 3, and that is said plainly

The frozen tolerance is 0.02. The **seed sd on composition exceeds it at six of the nine
K**: K = 10, 16, 20, 24, 28, 32
(`CROSSOVER-3SEED.json:per_K.<K>.selection_resolvability.seed_sd_exceeds_tolerance`).
A rule whose tolerance is narrower than the seed band cannot separate two K on three
seeds. **K\* = 32 is what the rule returns; it is not a K that three seeds can
distinguish from K = 24 or K = 28.** The rule was not moved and the tolerance was not
widened — this is reported as a limit of the instrument at n = 3, not repaired.

This is the honest cost of the finding in §2: making the interior rows n = 3 did not only
flip a sign, it revealed that the seed band is wider than the tolerance the selection was
built on. Closing that needs more seeds, which this box does not have (see §5).

---

## 3. The claim gate: charter-C controls, and one cell that does NOT collapse

Worst `valid` (max over the 24 val corners, then max over the 3 seeds)
(`CROSSOVER-3SEED.json:per_K.<K>.control_gate`):

| K | n | landscape_removed | landscape_shuffled | landscape_shuffled_across_wells | random_atoms | strict gate |
|---|---|---|---|---|---|---|
| 5 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| 8 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| 10 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| 12 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| 16 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| **20** | 3 | 0.0000 | 0.0000 | **0.0139** (mean 0.000193) | 0.0000 | **FAIL** |
| 24 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| 28 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |
| 32 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | PASS |

The single non-collapsing cell, named rather than averaged away
(`CROSSOVER-3SEED.json:per_K.20.controls_not_collapsing`):

| K | seed | control | valid max over 24 corners | valid mean over 24 corners | the arm at that cell |
|---|---|---|---|---|---|
| 20 | 1 | `landscape_shuffled_across_wells` | 0.0139 | 0.000579 | 0.9213 |

One corner of twenty-four, at 1 request in 72. **Under the strict gate — exactly zero on
the worst corner of the worst seed — K = 20 FAILS and no row may be claimed there.** That
is the reported verdict and it is not softened.

Beside it, and never instead of it: `landscape_shuffled`, the **shipped** charter C.2
definition (atom *values* permuted inside each well) and the one w12's own module declares
the verdict is taken from, **collapses to exactly 0 at every K and every seed**, as do
`landscape_removed` and `random_atoms`.
`landscape_shuffled_across_wells` is the task file's literal wording and is weaker by
construction: `scripts/w12_crossover_lib.py:perturb_modes` records that its permutation is
uniform with only the identity rejected, so about one well in K keeps its own atoms. I
checked whether that explains this cell and **it does not** — a fixed point is present at
K = 20 on seed 1 *and* on seed 2, and seed 2 collapsed to 0.0000. The 0.0139 is a residual
of the permuted read at that one cell, not the fixed point.

**The store claim stands at K = 24, the U-shape minimum**, where all four controls collapse
to exactly 0 on all three seeds. It does not stand at K = 20.

---

## 4. Store-scoped exactness, worst case over seeds, every K

Definition: the frozen encoder and decoder are declared **outside** the deletable object
and are **not** refit. Exactness is store-scoped — `downdate(delete the red rows)` against
a store fitted from scratch without those rows, same partition, same rank. Post-deletion
adaptation is light and data-free; it is **not** "no changes". A rival deletes only by
retrain-without, and that cost is printed in w12's own deletion artefacts, not here.

(`CROSSOVER-3SEED.json:per_K.<K>.exactness_worst`)

| K | n | max abs mode deviation | max subspace projector distance | ranks agree | rows refitted | data passes during deletion |
|---|---|---|---|---|---|---|
| 5 | 3 | 1.455e-11 | 1.430e-11 | True | 0 | 0 |
| 8 | 3 | 1.819e-12 | 1.370e-12 | True | 0 | 0 |
| 10 | 3 | 2.910e-11 | 1.981e-11 | True | 0 | 0 |
| 12 | 3 | 2.328e-10 | 2.126e-10 | True | 0 | 0 |
| 16 | 3 | 1.164e-10 | 1.164e-10 | True | 0 | 0 |
| 20 | 3 | 7.431e-08 | 1.976e-07 | True | 0 | 0 |
| 24 | 3 | 3.254e-08 | 5.865e-08 | True | 0 | 0 |
| 28 | 3 | 2.472e-08 | 4.667e-08 | True | 0 | 0 |
| 32 | 3 | 2.147e-07 | 2.578e-07 | True | 0 | 0 |

Worst anywhere in the sweep on three seeds: mode deviation **2.147e-07**, subspace
projector distance **2.578e-07**, ranks agree at every K on every seed, zero rows refitted
and zero data passes during deletion. 11,024 red rows deleted per cell.

---

## 5. What did NOT run — DECLARED NOT-RUN, with reasons

- **No scored run, no `test` condition.** By the task file: "Scored: ZERO. All val. Free."
  The `test` condition and the scored control gate at K\* were **not** re-run at n = 3.
  DECLARED NOT-RUN by design. No row was added to `/c/chlu/SCORED-LEDGER.md`.
- **Seeds 3 and 4: DECLARED NOT-RUN.** `C:/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/`
  holds `seed_0`, `seed_1`, `seed_2` and nothing else. There is no generation-2 feature
  cache for seeds 3 or 4 on this box, and w5-common 3.8 forbids scoring on generation 1.
  This is what caps the seed band at n = 3 and is the reason §2b cannot be closed here.
- **K outside {5, 8, 10, 12, 16, 20, 24, 28, 32}: DECLARED NOT-RUN.** The ladder was not
  extended, re-spaced, or refined around K = 24, because `SELECTION-RULE.md` pre-committed
  that the range would not be extended and because a finer grid near the minimum would be
  a new selection quantity, not this one.
- **A derangement version of `landscape_shuffled_across_wells`: DECLARED NOT-RUN.** The
  control definition was held FIXED across every row rather than improved mid-sweep, so
  the val rows here carry exactly the control w12's rows carry. Fixing the permutation to
  a derangement would make the two waves' controls incomparable.
- **w12's artefacts were read and never written.** `CROSSOVER-val.json`,
  `fig_crossover_val.png`, `SELECTION-RULE.md`, the `val_sweep/` rows and the `scored/`
  and `deletion/` trees are untouched; `git status` in this worktree shows no change to
  any w12 file, and this spoke wrote only under `/c/chlu/outputs/w15-crossover-seeds/`.

---

## 6. What would have proved this wrong

1. **The named falsifier.** If `argmin_K T(K, s)` had landed outside {20, 24, 28} on any
   seed, S3 would read REFUTED and the per-seed minima would be the reported result. It
   did not: all three seeds minimise at exactly K = 24.
2. **A minimum inside the band.** If the gap from K = 24 to K = 20 (12,203 B) or K = 28
   (30,506 B) had been smaller than the seed sd at K = 24 (7,060 B), the honest statement
   would have become "the total is flat over K = 20–28" and the word *minimum* would have
   been dropped. Both gaps clear the band; S4 says `minimum_resolvable = true`.
3. **A non-collapsing control at K = 24.** Any of the four charter-C controls scoring above
   zero at the minimum would make the store decorative there, and the U-shape would be a
   statement about the bytes of an object that does nothing. All four collapse to exactly
   0 at K = 24 on all three seeds. (They do **not** all collapse at K = 20 — §3 — and that
   K is reported as failing the gate.)
4. **A rebuild that moved.** If `CROSSOVER-3SEED.json`, `.md`, the PDF or the PNG had
   differed between two builds from the same rows, the artefact would not be reproducible
   and the numbers would not be checkable. `scripts/w15_rebuild_check.sh` passes.
5. **The import trap.** If `chlu` had resolved to the Head's checkout, or
   `CHLU_GEN_FEATURES_ROOT` had been unset so the run silently read generation-1 features,
   every number here would be void. Both are asserted in-process at the top of every leg
   (`scripts/w15_crossover_seeds.py:_assert_own_checkout`, and
   `exp_gen_delete_row7.assert_generation2` inside the w12 instrument). All four lanes
   printed `cache {'generation': 2, 'scope': 'full', 'n_rows': 480000, ...}` on their
   first line — `logs/laneA_s1_lowK.log`, `logs/laneB_s1_hiK.log`,
   `logs/laneC_s2_lowK.log`, `logs/laneD_s2_hiK.log`.

---

## 7. Method, and the one divergence from w12

The instrument was **imported, not reimplemented**. `scripts/w15_crossover_seeds.py`
imports `scripts/w12_val_sweep.py` and calls its `main()` verbatim; the only thing it
changes is the module-level `OUT` path, so rows land in this spoke's output dir instead of
w12's read-only tree. Same partition (`ward_balanced_labels`), same rank rule
(A = min(64, min_k d_k − 1)), same read (`TensorAddress`, `R.READ_PINS`, 90/10,
`pinned_basis`), same 24-corner val metric, same four charter-C controls, same
store-scoped exactness check. A leg here is therefore comparable line by line with w12's
seed-0 leg at the same K, because it *is* the same code.

Reporting is new, and for a reason that is itself the point of the spoke:
`w12_crossover_report.py` printed `per[ss[0]]["ledger_four_part"]` — **seed 0's ledger** —
as the ledger for a K. That is exactly the quantity the U-shape sentence rests on.
`scripts/w15_crossover_report.py` carries the four parts **per K per seed** with the mean
and band beside them; `scripts/w15_crossover_figure.py` draws the total as a four-segment
stack so "never netted" is a property a reader can check rather than a promise, with an
inset zoom on K ≥ 16 where the U actually lives.

**Files written by this spoke:** `scripts/w15_crossover_seeds.py`,
`scripts/w15_crossover_report.py`, `scripts/w15_crossover_figure.py`,
`scripts/w15_launch.sh`, `scripts/w15_rebuild_check.sh` in the worktree; everything else
under `/c/chlu/outputs/w15-crossover-seeds/`. Nothing else, anywhere.

**Environment note, recorded because it silently voids numbers.** `source
/c/chlu/env_w3.sh` alone does not give a working interpreter on this box: it joins
`PYTHONPATH` with a POSIX `:`, native Windows python splits on `;`, and the wave-2
site-packages ends up off the path entirely (`ModuleNotFoundError: No module named
'equinox'`). `/c/chlu/outputs/w15-crossover-seeds/env.sh` and `scripts/w15_launch.sh`
re-export `PYTHONPATH` with `;` and Windows-form paths after sourcing. Nothing else about
the environment was touched; no venv was built, no `uv sync` was run. Verified
`jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4`, python 3.11.9, and
`chlu.__file__ = C:\chlu\worktrees\CHLU-w15-crossover-seeds\chlu\__init__.py`.

**Compute.** 14 legs, 4 lanes (one BLAS thread each on a 4-core box), PIDs 174637, 174640,
174643, 174646 recorded in `PIDS.txt` at launch. No box-wide kill was ever run; all four
lanes exited on their own. Launched 2026-09-22T21:29:49Z, last leg landed 2026-09-22T23:06:26Z — wall clock 1 h 37 m. The address-book fit is the whole cost
of a leg (507–1,829 s); the store build is ~10 s and the five 24-corner reads ~250 s.
