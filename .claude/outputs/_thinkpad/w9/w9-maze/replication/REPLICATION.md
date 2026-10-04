# REPLICATION — w9/maze SCORED RUN 1 (seed 0, test leg)

Independent replication spoke, charter section A.5 ("a separate spoke re-runs the hub's winning
configuration from its PREREG file on a clean checkout of the hub's branch. Unreplicated = not a
result."). Run 2026-09-21.

## 1. What was run

- **Commit sha: `7db71f64b0a34ca3147ff3a0910e5e77c03c96fc`** — `[w9/maze] CMAZE-2: the request layer
  over CMAZE-1, and the CLU ported onto it`. Identical to the hub worktree's HEAD.
- **Clean checkout:** `git clone -b w9/maze /c/chlu/outputs/w9-maze/bundle/w9-maze-interim.bundle
  /c/chlu/worktrees/CHLU-w9-maze-replica`. `git status --porcelain` empty at clone time and at the
  end of the run.
- **No code was modified.** Nothing was written inside the hub worktree
  (`C:/chlu/worktrees/CHLU-w9-maze`) or `C:/chlu/outputs/w9-harness`. Imports were verified to
  resolve inside the replica: `chlu` -> `C:\chlu\worktrees\CHLU-w9-maze-replica\chlu\__init__.py`,
  `w9_maze` -> `C:\chlu\worktrees\CHLU-w9-maze-replica\w9_maze\__init__.py`, with the replica
  directory first on `PYTHONPATH`. Environment: jax 0.9.0, numpy 2.4.1, equinox 0.13.4,
  `OMP_NUM_THREADS=3`.
- **Command** (run from the replica directory):

      OMP_NUM_THREADS=3 python -u scripts/w9_maze_dev.py --seed 0 --which test \
        --ae-steps 12000 --n-atoms 31 --K 8 --ridge 0.01 --lowrank-atoms 8 \
        --tag replica --display --prereg C:/chlu/outputs/w9-maze/PREREG-w9-maze-1.md \
        --ae-cache C:/chlu/outputs/w9-maze/replication/ae_replica_seed0.eqx

  The encoder was trained **from scratch** (2127 s total; the hub's scored run took 239 s because
  it loaded an encoder cached from its DEV leg). No hub artefact was copied in.
- **Config matches PREREG-w9-maze-1 section 2:** 96 mazes, per_side 3, latent D = 256, encoder pool
  (543 bank-train items + 4860 pool items over 270 pool sites), AE 12000 steps, store K = 8,
  n_atoms = 31, ridge = 0.01 (rank/d_k = 0.9375, read = 410,688 floats), second arm `clu_lowrank`
  K = 8 / n_atoms = 8 / ridge = 0.01 (rank/d_k = 0.21875, 105,984 floats), test leg, seed 0,
  baselines matched to 410,688 floats within 2 percent.

## 2. Data determinism (maze bank rebuild)

Site 0 of `banks2/bank_seed0.pkl` (seed 1000001) was rebuilt from the replica's own
`w9_maze.task.build_site(seed, per_side=3, min_pairs=12, wall_target=0)` and compared to the hub's
cached bank. **Everything matched on the first try at `wall_target = 0`; no sweep over 0..5 was
needed.**

| field | result |
|---|---|
| grid | equal |
| tint | equal |
| door | equal |
| pairs | equal |
| routes | equal |
| walled_target | 2 (replica) vs 2 (hub) |

The maze bank is reproducible from seed alone.

## 3. Side-by-side RESULT-row validities (hub scored1 vs replica)

**Every arm on every RESULT row matched to full float precision. Maximum absolute difference over
all 36 arm x row validities: 0.0000.**

### ROW `seen` (n = 1208)

| arm | hub | replica | abs diff |
|---|---|---|---|
| clu | 1.0000 | 1.0000 | 0.0000 |
| clu_lowrank | 0.0240 | 0.0240 | 0.0000 |
| landscape_removed | 0.0000 | 0.0000 | 0.0000 |
| landscape_shuffled | 0.9396 | 0.9396 | 0.0000 |
| random_atoms | 0.9222 | 0.9222 | 0.0000 |
| lookup_matched | 1.0000 | 1.0000 | 0.0000 |
| mlp | 0.9048 | 0.9048 | 0.0000 |
| lstm | 0.6507 | 0.6507 | 0.0000 |
| transformer | 0.9892 | 0.9892 | 0.0000 |

### ROW `unseen_pair` (n = 195)

| arm | hub | replica | abs diff |
|---|---|---|---|
| clu | 0.1590 | 0.1590 | 0.0000 |
| clu_lowrank | 0.0000 | 0.0000 | 0.0000 |
| landscape_removed | 0.0000 | 0.0000 | 0.0000 |
| landscape_shuffled | 0.1590 | 0.1590 | 0.0000 |
| random_atoms | 0.1692 | 0.1692 | 0.0000 |
| lookup_matched | 0.1590 | 0.1590 | 0.0000 |
| mlp | 0.1385 | 0.1385 | 0.0000 |
| lstm | 0.1897 | 0.1897 | 0.0000 |
| transformer | 0.1897 | 0.1897 | 0.0000 |

### ROW `nway` (n = 183 bundles / 389 routes)

| arm | hub | replica | abs diff |
|---|---|---|---|
| clu | 0.0383 | 0.0383 | 0.0000 |
| clu_lowrank | 0.0000 | 0.0000 | 0.0000 |
| landscape_removed | 0.0000 | 0.0000 | 0.0000 |
| landscape_shuffled | 0.0383 | 0.0383 | 0.0000 |
| random_atoms | 0.0273 | 0.0273 | 0.0000 |
| lookup_matched | 0.0164 | 0.0164 | 0.0000 |
| mlp | 0.0273 | 0.0273 | 0.0000 |
| lstm | 0.0383 | 0.0383 | 0.0000 |
| transformer | 0.0492 | 0.0492 | 0.0000 |

### ROW `budget_interp` (n = 434)

| arm | hub | replica | abs diff |
|---|---|---|---|
| clu | 0.1313 | 0.1313 | 0.0000 |
| clu_lowrank | 0.0046 | 0.0046 | 0.0000 |
| landscape_removed | 0.0000 | 0.0000 | 0.0000 |
| landscape_shuffled | 0.1267 | 0.1267 | 0.0000 |
| random_atoms | 0.1106 | 0.1106 | 0.0000 |
| lookup_matched | 0.1198 | 0.1198 | 0.0000 |
| mlp | 0.1290 | 0.1290 | 0.0000 |
| lstm | 0.1498 | 0.1498 | 0.0000 |
| transformer | 0.1452 | 0.1452 | 0.0000 |

## 4. Ceilings, ledger, baselines, display — all identical

**Ceilings** (hub = replica, diff 0.0000 on every entry):

| row | representation | pipeline(rows) | store-span(rows) |
|---|---|---|---|
| seen | 1.0000 | 1.0000 | 1.0000 |
| unseen_pair | 1.0000 | 0.6205 | 0.6205 |
| nway | 1.0000 | 0.5090 | 0.5090 |
| budget_interp | 1.0000 | 0.5438 | 0.5461 |

pipeline(train) 1.0000 in both.

**Ledger** — identical integer for integer: `read_floats` 410,688 (atoms 7,936 + address book
402,752), deployed 1,642,752 B, float64 statistic 4,488 floats / 35,904 B, K = 8, n_atoms = 31,
dim = 256, design width 1,624, dead channels 0, rank/width 0.9375 on all 8 wells.
`ledger_lowrank.read_floats` 105,984, rank/width 0.21875 on all 8 wells.

**Matched baseline parameters** — identical: mlp 410,920 (width 482, matched), lstm 409,846
(width 205, matched), transformer 405,776 (width 149, matched). All within 2 percent of 410,688.

**Display section** — identical:

| quantity | hub | replica | bar |
|---|---|---|---|
| `delete_green_exactness.statistic_max_rel_residual` | 7.5217e-16 | 7.5217e-16 | <= 1e-10 |
| `delete_green_exactness.subspace_max_projector_residual` | 0.0000e+00 | 0.0000e+00 | <= 1e-6 |
| rows removed / refit / passes / items kept | 216 / False / 0 / False | 216 / False / 0 / False | — |

Refusal rates (refusal, validity, n — all identical):

| condition | hub refusal | replica refusal | diff | validity (both) | n |
|---|---|---|---|---|---|
| budget_infeasible | 0.9714 | 0.9714 | 0.0000 | 0.0000 | 280 |
| walled | 0.9606 | 0.9606 | 0.0000 | 0.0000 | 837 |
| mt_unseen | 0.9721 | 0.9721 | 0.0000 | 0.0035 | 573 |
| grounded_possible | 0.7979 | 0.7979 | 0.0000 | 0.0616 | 584 |

Interface deletion (move-left) and content deletion (green) receipts identical, including the
post-deletion composition figure on surviving targets (0.1569 -> 0.0000, delta -0.1569) and the
green-request refusal change (0.7642 -> 1.0000).

**ENV COLUMNS** identical: BLIND-SEQUENTIAL 0.9077 on this split (published CMAZE-1: 0.4980),
bend fraction 0.513 (published 0.816), FRAG same-maze r1 0.0599 / r2 0.5339, FRAG cross-maze
r1 0.0017 / r2 0.0000 (published 0.0000 / 0.0000).

**Whole-file check.** A full recursive diff of the hub's `scored1_seed0_test.json` against the
replica's `replica_seed0_test.json` found exactly **four** differing leaves, all of them
provenance rather than result: `/seconds` (238.80 vs 2126.85), `/tag` (`scored1` vs `replica`),
`/ae_cache` (path), and `/ae_loss_history` (empty in the hub because it loaded a cached encoder;
25 entries in the replica, which trained one). **Zero numeric differences anywhere else in the
file.** The independently retrained encoder is byte-identical to the hub's cached one
(md5 `a2c96e817f58ce5e7f4c5e2965b0b27b` for both `replication/ae_replica_seed0.eqx` and
`runs/ae/ae_seed0_steps12000.eqx`), so encoder training is bit-deterministic on this machine and
this is an exact, not merely statistical, replication.

## 5. The three claims

### (a) A matched-parameter LSTM or transformer BEATS the CLU on the unseen-pair row — **YES, REPLICATES**

| | clu | lstm | transformer | best baseline minus clu |
|---|---|---|---|---|
| hub | 0.1590 | 0.1897 | 0.1897 | **+0.0307** |
| replica | 0.1590 | 0.1897 | 0.1897 | **+0.0307** |

Both the LSTM and the transformer, matched to the CLU's 410,688 floats within 2 percent, score
0.1897 against the CLU's 0.1590 on `unseen_pair`. This is BAR 1's explicit trip-wire ("If a
matched-parameter LSTM or transformer beats the CLU on any of these rows, the Shapes3D result does
not transfer on that row, and THAT IS THE WAVE ANSWER"). The replication confirms it, exactly.
The CLU also fails to exceed `lookup_matched` (0.1590, a tie), is matched by `landscape_shuffled`
(0.1590) and is beaten by `random_atoms` (0.1692) on the same row.

### (b) The charter C.2 controls do NOT collapse — **YES, REPLICATES**

BAR 3 requires `landscape_removed` and `landscape_shuffled` validity <= 0.05 on the `seen` row.

| arm | hub `seen` | replica `seen` | collapses (<= 0.05)? |
|---|---|---|---|
| clu | 1.0000 | 1.0000 | — |
| landscape_removed | 0.0000 | 0.0000 | yes |
| **landscape_shuffled** | **0.9396** | **0.9396** | **NO** |
| **random_atoms** | **0.9222** | **0.9222** | **NO** |

`landscape_shuffled` (0.9396) and `random_atoms` (0.9222) sit within 0.06 and 0.08 of the CLU's
1.0000 — nowhere near the 0.05 bar. Only `landscape_removed` collapses, and it collapses because
the landscape is gone entirely, not because the store's content is doing work. At
rank/d_k = 0.9375 the store is **saturated**: a shuffled landscape and random atoms span
essentially the same subspace, so the read is carried by the span, not by the learned content.
The store is decorative on this content. **BAR 3 fails identically in the replication, so under
the hub's own pre-registration no row is claimed.**

### (c) Exact deletion holds — **YES, REPLICATES**

| quantity | hub | replica | bar | passes |
|---|---|---|---|---|
| downdate-vs-refit statistic max relative residual | 7.5217e-16 | 7.5217e-16 | <= 1e-10 | yes |
| subspace max projector residual | 0.0000e+00 | 0.0000e+00 | <= 1e-6 | yes |

216 rows removed by rank-m subtract on the float64 sufficient statistic, `refit = False`,
`passes_over_retained_data = 0`, `items_kept = False`. The deletion machinery is exact in the
replication, to the same digits.

Note (not part of claim (c), but adjacent in BAR 6): the post-deletion composition row does **not**
meet its own bar in either the hub or the replica — validity on the OTHER, surviving targets goes
0.1569 -> 0.0000, a delta of -0.1569 against a +/- 0.05 tolerance. That failure reproduces exactly
too.

## 6. What did NOT reproduce

**Nothing.** There is no quantity in the hub's scored run 1 that the replication failed to
reproduce. Every RESULT-row validity, every bend/no-bend split, every ceiling, every ledger
integer, every baseline parameter count, every refusal rate and both deletion residuals matched to
full precision, and the independently retrained encoder came out byte-identical.

Two observations that are properties of the hub's run rather than replication failures, recorded
here because they reproduce faithfully and bear on how the numbers should be read:

1. **`nway` row denominators are mixed.** The arm-level `validity` on the `nway` row is scored per
   *bundle* (n = 183; all routes in the bundle must be correct), while `pipeline_ceiling_rows` and
   `span_ceiling_rows` for `nway` are scored per *route* (n = 389, validity 0.5090), as are the
   bend / no-bend splits (197 + 192 = 389). The `nway` arm validity of 0.0383 is therefore **not**
   directly comparable to its quoted 0.5090 ceiling. The per-route CLU figure is available in the
   JSON as `table.nway.clu.per_route.validity = 0.1183`. Identical in hub and replica; flagged,
   not changed.
2. **The hub's scored run loaded a cached encoder** (`runs/ae/ae_seed0_steps12000.eqx`, built in
   its DEV leg) rather than training one, which is why `ae_loss_history` is empty in its JSON and
   its wall clock was 239 s. This replication trained the encoder from scratch in 2127 s and
   arrived at a byte-identical file, so the cache is not a confound.

## 7. Artefacts

All under `C:/chlu/outputs/w9-maze/replication/`:

- `replica_seed0.log` — full run log
- `replica_seed0_test.json` — the replication's full result record (copied from
  `C:/chlu/outputs/w9-maze/runs/`, where the script writes it)
- `replica_seed0_test.txt` — the printed table
- `ae_replica_seed0.eqx` — the independently trained encoder

Replica checkout: `C:/chlu/worktrees/CHLU-w9-maze-replica` at `7db71f6`, unmodified.

## Verdict

**SCORED RUN 1 of hub `w9/maze` REPLICATES EXACTLY** on a clean checkout of `7db71f6` from the
hub's own bundle, with the encoder retrained from scratch and no code changed. All three claims
hold with identical numbers: (a) YES — matched LSTM and transformer both beat the CLU,
0.1897 vs 0.1590 on `unseen_pair`; (b) YES — the C.2 controls do not collapse, `landscape_shuffled`
0.9396 and `random_atoms` 0.9222 against the CLU's 1.0000 on `seen`, so the store is saturated and
decorative at rank/d_k = 0.9375; (c) YES — deletion is exact, statistic residual 7.5217e-16 and
subspace projector residual 0.0000e+00.
