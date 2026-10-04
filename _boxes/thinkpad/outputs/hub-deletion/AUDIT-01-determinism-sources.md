# AUDIT-01 — sources of non-determinism on the build path
**2026-09-17, T+0:03.** Read-only static audit of `canonical_build` / `gen_store.train_store`,
commissioned by me and run by a sub-agent on Opus, then checked by me against the files.
⚠ This is an audit of the **code**, not a measurement. The measurement is `det_A` vs `det_B`
(DETERMINISM.json). An audit that says "should be deterministic" is not the gate and does not replace it.

## Verdict: CONDITIONALLY DETERMINISTIC
The source is clean: **no** unseeded PRNG, **no** `time`/`os.environ`/`hash()`/set-ordering/device-count
dependence, **no** argument mutation anywhere on the build path. Every draw derives from the passed
`seed`/`key`. The conditions are entirely numeric-environment ones:
1. identical BLAS thread count in both processes;
2. identical JAX backend (CPU) and XLA intra-op thread count;
3. identical numpy / scipy / jax / optax builds.

⚠ **The gap I am carrying, stated rather than closed:** `build_unit.py` **records**
`OMP/MKL/OPENBLAS_NUM_THREADS` into `build.json` but never **sets** them, and never pins
`JAX_PLATFORMS=cpu`. Both my processes are launched from the same `hubenv.sh` with the same values, so
the two-process comparison is still valid *on this machine, in this session* — which is exactly the
claim the falsifier asks for. ⛔ I am **not** upgrading it to "deterministic anywhere". The
post-hoc check that both builds saw the same thread values is in `DETERMINISM.json["threads"]`.
⚠ `JAX_PLATFORMS` is deliberately **not** changed between `det_A` and `det_B` — altering the
environment mid-gate would make the pair incomparable. jaxlib here is the CPU build and this box has no
CUDA, so there is no second backend to pick up; pinning is added only for units launched after the gate.

## The ranked risks (file:line)
1. ⭐ **`gen_store.py:131`** `C = (A.T @ A) / n` on `(16384, 2048)` float64 → multithreaded BLAS, so the
   last ULPs of `C` are a function of thread count/blocking. `scipy.cluster.hierarchy.linkage` is a
   **greedy sequential merge**, so one flipped ULP can flip a merge and produce a *completely different
   partition*. **This is the highest-risk path and it is why the gate is run at all.**
2. `gen_store.py:788-799` — 1200 `eqx.filter_jit` Adam steps in float32; a 1-ULP divergence at step 0 is
   amplified. XLA-CPU is deterministic at fixed shapes + fixed intra-op pool, not across backends.
3. `gen_store.py:797` `jax.random.choice(sk, n, (batch,), replace=False)` — deterministic given
   `(sk, n)`, but the without-replacement algorithm has changed between JAX releases ⇒ library drift
   silently reshuffles every batch. Same at `:993`.
4. `gen_store.py:190-215` `_cluster_live` — **scipy**, not sklearn: `linkage` + `fcluster(maxclust)`.
   No seed and none needed; version-coupled.
5. `gen_dream.py:304` `M = int(C.max()) + 1` — the book's `M` comes from the **observed maximum code**,
   not `cfg.n_modes`, so an unused top mode changes the book's *shape*. Still pure; and
   `store_digest` hashes `str(a.shape)`, so it is **caught, not hidden**.
6. `SpectralClustering(random_state=0)` at `:206-214` — **not reached**; `GROUPING_ORDER=("average","ward")`.
7. `_kmeanspp_init` `:690-701` — float32 reductions, deterministic per numpy build. Low.
8. ⚠ **`build.json` files will never be byte-identical** (`timing_s`, `unit_wall_s`, `utc`). The
   comparison is on `digest.store_digest` and on the `modes.npz` array bytes — `compare_det.py`
   re-hashes the arrays independently so the verdict does not rest on the digest agreeing with itself.

## ⭐⭐ CORRECTIONS TO MY OWN PRIOR BELIEFS — recorded because they change the claim
- ⛔ **`n_corr` is 16,384, not 1,024** (`config.py:3486`; 1,024 was the *smoke* override).
  ⇒ `Xc = X_live[:16384]` is **the first 16,384 rows of the sorted live vector**. Deleting rows changes
  *which* rows those are, so **the channel partition itself generally changes on deletion.**
  ⭐ **A post-delete store is a wholly different object, not a perturbation of the original.** That is
  correct by design — it is what makes the layout a canonical function of the live set — but it is a
  **claim-scoping fact** and it is why PREREG §3's size-matched placebo is not optional.
- ⛔ **There is no `min_well_frac` retry loop.** I had assumed `grouping_info["attempts"]` was a walk
  over relaxing floors. It is not: `floor` is computed **once** (`gen_store.py:253`) and `attempts` is a
  walk over **methods** (`average` then `ward`), breaking on the first whose `sizes.min() >= floor`. If
  none passes it **raises** — ⭐ **the floor is never relaxed.** Good: there is no hidden loosening here.
- ⚠ **`grouping="correlation"` does not use `seed` at all** (`:344-346`); the seed reaches the grouping
  only for `grouping="random"`. So the partition is a function of the live features alone.
- ⚠ **`store_removed_read(..., "mean", seed=…)` ignores the seed** — only the `"scramble"` branch
  consumes it (`gen_store.py:673`). The laundering floor is a deterministic function of
  `(store, codes)`. ⭐ A strengthening, but ⛔ the floor must **not** be described as "seed-averaged".

## The two facts the deletion claim leans on — both CONFIRMED in code
- **`fit_address_book`, `gen_dream.py:313-318`**, smoothing `a = 1.0`: a factor value with **zero** live
  rows gives `cond[v,k] = (0 + 1)/(0 + 1·M) = 1/M` — the *identical float64 scalar* for every mode, so
  the row is **exactly uniform, bit-for-bit**. No NaN, no division by zero.
  ⚠ **Precision correction to the comment in `exp_gen_deletion.py:214-215`**, which says the term
  "contributes nothing". In `address_requests` the score accumulates `lc[v] − lp` per asked factor, so
  for the deleted `v` the bracket is `(const − log P(m))`: **constant in `m`, so it cannot move an
  argmax by itself, but not identical to omitting the factor** — it leaves the score one extra
  `−log P(m)` lighter than the `R[:,f] = −1` "not requested" path. ⭐ And if a request names *only* the
  deleted value, the score is exactly `−log M` everywhere and `np.argmax` returns **mode 0 in every
  well** — a degenerate constant read. The deployed eval always sends the full 6-factor tuple, so the
  other five terms dominate; ⛔ that is the measured open question, not an assumption.
- **`FactorPrototypes.fit`, `gen_request.py:306-311`**: `n_k[c] == 0 < min_per_value=1` ⇒ `P[c] = mu`,
  the global mean over the **live** rows; the lazy conditional never evaluates the NaN branch, and the
  zero count is surfaced in `n_per_value`. ⭐ So both arms degrade the deleted term to *uninformative*
  by their own shipped code — neither is handed the deletion for free.

## Bytes, exactly (⛔ not "matched" by assertion)
`knn_proto` `byte_ledger()` = `(sum_k C_k + 1) × D × 4` = `(57 + 1) × 2048 × 4` = **475,136 B**.
CLU store = `64 × 2048 × 4` = **524,288 B**. ⭐ **The rival is 90.6 % of the CLU's bytes — it is the
*smaller* model, which is conservative against the CLU**, and the report says so rather than claiming
parity.

## Config values that reach the store bytes (`config.py`)
`n_corr=16384` · `n_modes=64` · `train_steps=1200` · `train_batch=128` · `train_lr=3e-3` ·
`tau 1.0→0.1` · `commitment_beta=0.25` · `min_well_frac=0.10` · `grouping_method="auto"` ·
`n_store_train=32768` (consumed by `select_store_rows`, not by `canonical_build`). `k=5` is hard-coded
at `build_unit.py:166` from the pinned `store_plan.json`, ⛔ not read from cfg.

**Batch shape vs `n_live`:** `batch = min(train_batch, n) = 128` for every planned deletion (the
smallest live set here is 24,534 rows), and `steps`/`tau` do not depend on `n`. ⭐ **Shape invariant;
contents fully `n`-dependent** — `jax.random.choice(sk, n, …)` draws from `[0,n)`, so the index stream
differs from step 0. ⚠ The only shape hazard is `n_live < 128`, which cannot arise here.

## Config aliasing
`CHLUConfig()` builds a fresh `GenStoreConfig` per instance (all `default_factory`), so there is no
shared mutable default. ⚠ `g = config.gen_store` is a **reference**, and `dataclasses.replace` on a
`CHLUConfig` is a **shallow** copy that shares the `GenStoreConfig` instance. Safe as written because
`build_unit.py` is one fresh process per unit and mutates before the single `canonical_build` call —
⛔ but it would not survive looping builds inside one process. Noted so I do not later "optimise" the
driver into a loop and break the very property I am proving.
