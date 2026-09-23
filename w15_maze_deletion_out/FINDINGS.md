# w15 maze-deletion-2b — FINDINGS

**Environment on every row: CMAZE-2b.** ⛔ **This table contains no CMAZE-1 or CMAZE-2 row.**
Seeds: 3 (0, 1, 2), all of them. **Scored runs spent: 0** (free replication; no SCORED-LEDGER row needed or written).
Paths are relative to `/c/chlu/outputs/w15-maze-deletion-2b/` unless absolute.
Acceptance artefact: `MAZE-DELETION-2B.json` (machine-readable) + `TABLE.md` (the same rows, rendered).

## Verdict

**Exactness transfers to CMAZE-2b.** On all 3 seeds, deleting the green target from the store
by downdate gives the same result as refitting the store without those rows, to float64
rounding:
- statistic residual **7.212e-16 / 9.093e-16 / 8.937e-16** (max relative, float64);
- subspace projector **0.000e+00** on all three seeds;
- the address book's Gram (A, Mx) and the deployed read are also **0.000e+00** on all three.

A from-scratch clean checkout reproduces this **bit-identically**. The clean checkout rebuilt
the banks, the pool and the encoders itself. The maze section keeps its differentiator,
now measured on the environment the paper prints.

## 1. What was run

- **Environment: CMAZE-2b.** This is the configuration of the CMAZE-2b rows the paper prints
  (`w14_read_verb.leg` defaults):
  - banks: `w14_request_layer.task2.build_site_v2`, `bank_after_seed{0,1,2}.pkl`, 96 sites each;
  - GRID_REBASED budget words;
  - CMAZE-2b encoder pool: 270 sites (`w14_read_verb.pool`);
  - store: K = 8 wells, n_atoms = 31, ridge = 0.01; encoder trained for 12000 steps.

  Nothing in `w14_request_layer`, `w14_read_verb` or `w9_maze` was edited; everything is
  imported. The driver is `w15_maze_deletion/run_deletion.py` and the assembler is
  `w15_maze_deletion/assemble.py`, both on branch `w15/maze-deletion-2b`.
- **Deletable object:** the store's sufficient statistic (S, t, n), the atoms derived from it,
  and the address book's retained Gram (A, Mx).
  - ⛔ **The frozen encoder and its frozen decoder are declared OUTSIDE the deletable object.**
- **Primary row:** delete every item written under target = green. This is the same word as
  the banked w9 row.
- **Refit-without:** a fresh `CovContentStore` fitted on the surviving rows, with the same well
  partition, and the address book re-accumulated and re-solved on the same survivors.
- **Measurement set:** the exactness rows read only TRAIN items. The driver asserts that the
  train set is identical under the val and test splits, so no held-out row is touched.
- **Metrics:**
  - `statistic` and `projector` are computed by `w9_maze.display.downdate_vs_refit`, imported
    unmodified, so they are the same quantities as the banked numbers.
  - This spoke adds residuals on the address book's Gram and on the read. The read is
    compared on 2038 / 2015 / 2055 probe requests (`deletion_rows[*].n_probe_requests`) (train requests plus VAL grounded rows).
  - It also adds a sanity column: how far the deletion **moves** the read. A zero residual
    only means something next to a non-zero change.

## 2. The deletion rows — CMAZE-2b, 3 seeds

| env | seed | deletion | rows removed / kept | downdate vs refit: statistic (max rel, float64) | subspace projector | Gram A / Mx residual | read residual | read moved by deletion | clean checkout: statistic / projector | source |
|---|---|---|---|---|---|---|---|---|---|---|
| CMAZE-2b | 0 | target = green | 224 / 1218 | **7.212e-16** | **0.000e+00** | 0.0e+00 / 0.0e+00 | 0.0e+00 | 0.793 | 7.212e-16 / 0.000e+00 | `runs/deletion_primary_seed0.json`, `clean/runs/deletion_clean_seed0.json` |
| CMAZE-2b | 1 | target = green | 225 / 1205 | **9.093e-16** | **0.000e+00** | 0.0e+00 / 0.0e+00 | 0.0e+00 | 0.506 | 9.093e-16 / 0.000e+00 | `runs/deletion_primary_seed1.json`, `clean/runs/deletion_clean_seed1.json` |
| CMAZE-2b | 2 | target = green | 211 / 1240 | **8.937e-16** | **0.000e+00** | 0.0e+00 / 0.0e+00 | 0.0e+00 | 0.500 | 8.937e-16 / 0.000e+00 | `runs/deletion_primary_seed2.json`, `clean/runs/deletion_clean_seed2.json` |

On every row: refit = False, 0 passes over retained data, items never kept (the `receipt` in
each run JSON).

**Supplementary deletions.** These are labelled supplementary and are not the printed row:
- the other 5 target words, maze = 0, agent = 0, and a random 10 % of rows;
- 8 deletions per seed × 3 seeds = 24 rows (`TABLE.md` §Supplementary, `MAZE-DELETION-2B.json` → `supplementary_rows`);
- worst statistic residual across all 27 deletions: **9.283e-16**;
- projector max **0.000e+00**; read residual max **0.000e+00** (`MAZE-DELETION-2B.json` → `summary`);
- how far the deletion moves the read: 0.302–0.872 (per-row `sanity_read_change_by_deletion`).

⚠ **How to read the 0.000e+00 projector.** The projector is built from the deployed atoms,
which are **float32**. It is therefore exactly zero whenever downdate and refit round to the
same float32 atoms. It cannot resolve differences below float32. The fine-grained exactness
number is the **float64 statistic residual** (7.2e-16 – 9.1e-16). Quote the two together.

Deployed ledger (the convention is named): the read has 410688 floats = 1642752 bytes at
**float32 deployed** (atoms 7936 + address book 402752). The sufficient statistic is 35904 bytes
at **float64 retained**; this is what buys exact deletion, and it is declared beside the
deployed bytes, never netted in. Source: `runs/deletion_primary_seed*.json` → `ledger`, same
on all 3 seeds. Deployed rank per well: 30 of 32 channels (rank/d_k 0.94, the saturated regime
w9 already declared).

## 3. Clean-checkout replication

- **Checkout.** I committed branch `w15/maze-deletion-2b` at `f9f524647cb9675e32f320f13e26061f9f8a428d`,
  bundled it (`clean_src.bundle`) and `git clone`d it into `clean_checkout/` (0 dirty files).
  `chlu.__file__` resolved to `clean_checkout\chlu\__init__.py` (the env assert passed; see
  `logs/clean_run_seed*.log`).
- **Inputs rebuilt from scratch inside the clean checkout:** 3 CMAZE-2b banks (`w14_request_layer.build_bank after s 96`)
  and 3 pool shards (`w14_read_verb.pool s 90`).
  **All 6 are sha256-IDENTICAL** to the live inputs the printed rows used (`clean/INPUTS_SHA.txt`).
- **Encoders retrained from scratch** (12000 steps, about 7100–7260 s each under box contention).
  **They are sha256-bit-identical to the frozen encoders behind the printed CMAZE-2b rows.**
  `Ztrain` is bit-identical on all 3 seeds (`MAZE-DELETION-2B.json` → `primary_rows[*].clean_checkout_replication`:
  `encoder_bit_identical_to_printed: true`, `Ztrain_bit_identical: true`, `identical_to_primary: true`).
- **Result:** all 27 deletion residuals, and the adaptation-cost validities, are **identical**
  between the primary and the clean run (`clean/runs/deletion_clean_seed*.json`).
- ⚠ **One non-bit-identical quantity, stated.** On the carry path (w9 as pre-registered), the
  count of address-book floats that change differs by a handful between runs. For example,
  seed 0 changes 402728 floats in the primary run and 402727 in the clean run; seed 2 changes
  402737 and 402728. This is float32 BLAS rounding under different thread counts
  (OMP 3 vs 4). It does not touch any exactness number or any validity.

## 4. Adaptation cost after deletion — ⛔ never "no changes"

Deletion is exact at the scope of the store. What happens to the deployed object afterwards
is a light, data-free re-solve. It is **not** "no changes":

| env | seed | path | seconds | address-book floats changed | atom floats changed | VAL other-target validity before → after (Δ) | VAL green refusal before → after | retrained |
|---|---|---|---|---|---|---|---|---|
| CMAZE-2b | 0 | exact head downdate (A −= ΦdᵀΦd, Mx −= ΦdᵀXd, closed-form re-solve) | 2.042 | 402656 / 402752 | 7936 / 7936 | 0.2609 → 0.2411 (**−0.0198**) | 0.6333 → 0.5333 | no |
| CMAZE-2b | 1 | same | 2.328 | 402656 / 402752 | 7936 / 7936 | 0.1849 → 0.1828 (**−0.0021**) | 0.7248 → 0.4404 | no |
| CMAZE-2b | 2 | same | 2.181 | 402674 / 402752 | 7936 / 7936 | 0.2490 → 0.2431 (**−0.0059**) | 0.5638 → 0.5319 | no |
| CMAZE-2b | 0 | basis-change carry W ← WR (w9 as pre-registered) | 0.009 | 402728 / 402752 | 7936 / 7936 | 0.2609 → **0.0000** | 0.6333 → 1.0000 | no |
| CMAZE-2b | 1 | same | 0.004 | 402728 / 402752 | 7936 / 7936 | 0.1849 → **0.0000** | 0.7248 → 1.0000 | no |
| CMAZE-2b | 2 | same | 0.012 | 402737 / 402752 | 7936 / 7936 | 0.2490 → **0.0000** | 0.5638 → 1.0000 | no |

Source for every cell: `runs/deletion_primary_seed{s}.json` → `adaptation_cost`. There are
90 / 109 / 94 VAL green rows and 506 / 476 / 510 VAL other-target rows.

- The honest description of the adaptation: **every atom and ≈ 99.98 % of the address book
  change.** The change is a closed-form re-solve that reads only the deleted rows handed in;
  it makes 0 passes over retained data and does no retraining. It is light in compute
  (about 2 s), **not** in extent.
- The carry path is w9's pre-registered mechanism. On CMAZE-2b it again destroys the
  surviving targets (→ 0.0000), exactly as it did on the earlier layer. The exact-head
  downdate (w9 Amendment 1) is the path to print: the surviving-target cost is −0.0021 to
  −0.0198 on VAL.
- ⚠ **Deletion does not induce refusal of the deleted word.** On the exact-head path, green
  refusal *falls* on all 3 seeds (0.63 → 0.53, 0.72 → 0.44, 0.56 → 0.53). The paper must not
  claim "deleted ⇒ refused". The claim is store-scoped exactness, nothing more.
- Timing depends on the box. In the clean run, under contention, the same exact-head step took
  8.4–12.3 s (`clean/runs/deletion_clean_seed*.json`). The primary-run seconds were measured
  with less contention.

## 5. The rival deletes only by retrain-without (cost printed)

| env | seed | rival | params (target) | steps × batch | row reads of retained data | seconds | must keep items | CLU downdate seconds (store + head) | same-object refit-without seconds |
|---|---|---|---|---|---|---|---|---|---|
| CMAZE-2b | 0 | transformer (matched read-path params) | 405776 (410688) | 4000 × 64 | 256000 | 89.8 | yes, 1218 rows | 0.875 | 1.103 |
| CMAZE-2b | 1 | transformer | 405776 (410688) | 4000 × 64 | 256000 | 90.8 | yes, 1205 rows | 0.917 | 1.172 |
| CMAZE-2b | 2 | transformer | 405776 (410688) | 4000 × 64 | 256000 | 88.6 | yes, 1240 rows | 0.802 | 1.084 |

Source: `runs/deletion_primary_seed{s}.json` → `rival_retrain_without`, `deletion_rows[0]`.
Both sides share the frozen encoder, so its cost is excluded from both.

⚠ **Stated plainly:** at this scale the CLU's downdate is **not** much faster than refitting its
own store and head from the survivors (≈ 0.8–0.9 s vs ≈ 1.1–1.2 s). The differentiator is not
speed. It is that the downdate is exact **without keeping the items** and without any pass over
retained data. The rival must keep all ~1200 rows and read them 256000 times.

## 6. Relation to the banked numbers — cited here, never in the table

The banked numbers were measured on CMAZE-2, the earlier layer (`/c/chlu/outputs/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.txt:66`).
⚠ The seed-2 artefact says **7.818e-16**, not the "8.2e-16" that `w9-maze/LOG.md:220` and later
documents quote. The replacement numbers for the paper are the CMAZE-2b rows in §2.
⛔ The two environments are not merged into one table and not averaged.

## 7. What did NOT run

- **Test-leg adaptation composition: DECLARED NOT-RUN.** Reading test rows would be a scored
  act, and this spoke spends zero scored runs. The adaptation validities in §4 are on VAL only.
  The exactness rows need no held-out row, because the train set is identical under both splits.
- **Rival retrain-without for the MLP / LSTM: DECLARED NOT-RUN.** Only the matched transformer,
  the strongest printed rival, was timed. The LSTM and MLP retrain costs are not printed here.
- **Charter-C controls:** not re-run. They are not part of a deletion claim.
- The first primary pass (`runs_old/`) is superseded by `runs/`. It gave the same residuals,
  without the sanity and rival fields.

## 8. What would have proved this wrong

- Any seed with a float64 statistic residual well above rounding (≫ 1e-15), a non-zero
  projector, or a non-zero read residual. That would mean exactness does not transfer to
  CMAZE-2b, and the maze section would lose its only banked differentiator. **None occurred:**
  27 of 27 deletions are at ≤ 9.283e-16, and the projector and read residuals are 0.
- A zero residual next to a deletion that did not move the object, which would make the result
  vacuous. **Ruled out:** the deletion moves the read by 0.30–0.87 relative on every row.
- A clean checkout that failed to reproduce the result. **Ruled out:** the inputs, the encoders
  and every residual are bit-identical.

## Environment notes

- The env assert passed in both checkouts:
  - worktree: `C:\chlu\worktrees\CHLU-w15-maze-deletion-2b\chlu\__init__.py`;
  - clean checkout: `clean_checkout\chlu\__init__.py`.
- Versions: jax 0.9.0, numpy 2.4.1, equinox 0.13.4.
- ⚠ Under Git Bash, `env_w3.sh`'s `:`-separated PYTHONPATH is collapsed by MSYS into one entry,
  so equinox cannot be imported. I re-exported the same two entries, in the same order, with the
  Windows `;` separator (`logs/launch.sh`, `logs/clean_run.sh`). I built no venv and installed nothing.
- PIDs of everything I launched are in `PIDS.txt`. I killed no processes.
