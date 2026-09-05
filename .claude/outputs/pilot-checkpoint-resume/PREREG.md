# PREREG — pilot-checkpoint-resume

Filed **before** running any toy gate or RSS harness, at implementation-complete
(`ruff` green, nothing executed yet). Base `main @ 8efc1d8`; branch
`agent/experiment-engineer/pilot-checkpoint-resume`.

Machine: Apple M1 Pro, **CPU backend**, main venv reused (protocol §4). Every
GPU/CSF3 number below is a **projection**, labelled as such — the cluster is
unreachable from this machine.

---

## P1 — old path vs new path is BITWISE on every decision-bearing field

**Prediction: bitwise identical.** Derivation: the change set is IO + cache
management. The three things that could move a number, and why they do not:

1. `release_host_memory()` drops *executables*, never values. XLA compilation of
   the same HLO is deterministic, so a re-compiled program returns the same
   bits. (This is the same argument `pilot-placement-probe` used when it jitted
   the plan-pass stages, which was measured bitwise.)
2. `monitors_final` moves from *after* the eval phases to *inside* the training
   segment. It is a pure function of `(m, reg, x0)`; nothing between the two
   positions writes to `m` or calls `reg.observe`. ⇒ inert.
3. The `[train/…]` print and the journal writes touch no array.

**Excluded from "bitwise" by construction (declared in advance):** wall-clock
fields — `wall_s`, `wall_s_total`, `plan_pass_s`, `plan_pass_frac`,
`train_log[*].wall_s`/`plan_s`, `monitors_*.wall_s`, `anytime_curve[*].wall_s`,
`gradient_probe_*.wall_s*`, `store_health[*].wall_s`. These are timings and were
never reproducible.

**Falsifies P1:** any non-timing float differing at all. There is no tolerance
band here — unlike `csf3-memory-fit`'s remat (a genuinely re-associated VJP), no
arithmetic is re-associated by this change set, so the correct prediction is
**exact equality, not ≤1 ULP.**

## P2 — kill-after-arm-1 + `--resume` matches an uninterrupted run BITWISE on the
remaining arms

**Prediction: bitwise identical on arms 2..N**, and on arm 1's re-emitted rows.
Derivation of the stream guarantee (this is the claim the task asks me to verify
and state): `_train_batches` materialises `list(random_batches(tr, batch, seq_len,
n_batches=steps, seed=pcfg.seed))` ONCE and each arm consumes `iter(batches)`, a
fresh iterator over that same list. `random_batches` draws from a
`np.random.default_rng(seed)` created inside the call. Therefore **arm k's batch
sequence is a function of `(seed, steps, batch, seq_len)` alone and carries
nothing out of arms 0..k-1** — no fast-forwarding is needed and none is
performed. Eval iterators are `contiguous_batches` (deterministic, unseeded).
Model init is `PRNGKey(1000 + seed)` split 3, and `build_arm` re-splits
deterministically ⇒ the deserialisation template is reproducible from `seed`.
Optimiser state is re-initialised per arm inside `train_arm`.

**Predicted single point of drift, registered in advance:** the persistent
monitor registry (monitor #6's `(write_loss, acq)` window) cannot be serialised.
I have handled it by taking `monitors_final` inside the training segment; **I
predict this makes even monitor #6 bitwise across a resume.** If instead I had
left `monitors_final` after the eval phases, I predict monitor #6 would flip
`applicable → inapplicable` on a resumed arm.

**Falsifies P2:** any non-timing float differing on arms 2..N.

## P3 — WHERE the host-RSS peak lives (attribution, at toy)

Registered ranking of the eval-block phases by their contribution to peak host
RSS, highest first:

1. `clu_store/gradient_probe_final` — 4 backwards, and it is the ONE phase that
   is **not** under `filter_jit` (`eqx.filter_value_and_grad(loss_fn)` is called
   directly), so its residuals are eager buffers with no XLA reuse, plus it
   rebuilds a second model via `_swap_mcfg`;
2. `clu_store/anytime_curve` — 5 one-shot compiles of a program whose `verlet`
   is static, none re-used;
3. `clu_store/dyneval` — a backward, ×3 LRs in the grid;
4. `clu_store/static`, `clu_store/blank_store` — forward only, and they share
   one executable.

**Prediction: `hygiene` gives measurable RSS back** — i.e. at ≥1 phase boundary
`rss_gb` after `release_host_memory()` is **lower** than before it, by ≥ 50 MB at
toy. **Falsifies:** hygiene returns < 10 MB at every boundary at toy (⇒ the lever
is inert at this scale and the CSF3 projection is unsupported by measurement, and
I must say so rather than claim the fix works).

## P4 — the projection to pilot, against the < 100 GB budget

⚠ **Registered honestly: I predict I CANNOT bound this from the laptop.**
Attempt 1's MaxRSS is *truncated by the kill* (≥ 125.6 GB, upper bound unknown),
so the quantity to be reduced has no measured value, and toy RSS at
`d_model=64, n_layers=2, batch=4` is dominated by the Python+JAX floor
(predicted 0.5–2 GB total), not by the eval block. **I therefore pre-register
that my deliverable-2 output is (a) the measured toy per-phase attribution,
(b) the mechanism, and (c) an explicitly-labelled UNBOUNDED projection — not a
"< 100 GB" claim.** A "< 100 GB projection" asserted from toy numbers would be
the kind of number this protocol exists to prevent.

**Registered instead as the actionable prediction:** if hygiene alone is
insufficient, the ordered cut list is (i) `plan_workers` 8→4 (predicted saving:
4 × the per-worker RSS, i.e. 4 × ~0.3–0.8 GB = **1.2–3.2 GB** — *predicted too
small to matter against a ≥ 125.6 GB peak*), (ii) drop `--d5` (removes 5 of the
one-shot compiles), (iii) `-G 2 -c 24` for 240 GB (Head's call). I register in
advance that **(i) is a decoy** and that if the true peak is compile-driven the
only levers that can move it by tens of GB are (ii) and (iii).

## P5 — suite

Baseline **1348 passed / 0 failed**. Prediction: **1348 + (new tests) passed,
0 failed**, and ruff green.
