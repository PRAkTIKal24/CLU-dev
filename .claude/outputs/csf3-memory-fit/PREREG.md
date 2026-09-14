# PREREG — `csf3-memory-fit`

Protocol §5 pre-registration rule. Filed **2026-08-01**, on branch
`agent/experiment-engineer/csf3-memory-fit` @ `be258f1`.

## 0. Honest ordering statement (read this first)

The acceptance criterion of this task contains **two ratios** and they were
pre-registered at different moments. I record both, and which measurements were
already in hand when each was written.

| Prediction | Registered by | Registered when | Measured when |
|---|---|---|---|
| P1: activations ÷ ~`n_chunks` (16) from chunk remat | **the task file** (`.claude/tasks/csf3-memory-fit.md` §1), i.e. by the Hub, before I ran anything | before the branch existed | ledger part 1 (C-sweep) |
| P2: remat costs ≤ 2× the chunk-interior compute | **the task file** §1/§6 | as above | toy wall-clock |
| P3–P5 below (layer/batch scaling, the projected pilot peak, the toy slowdown) | this file | **after** ledger part 1 (the C-sweep) had printed, **before** ledger part 2 and before any wall-clock run | ledger part 2 + toy runs |

⛔ So: P1's confirmation is a genuine pre-registered result. P3 (layer scaling)
is registered here **while `ledger2.py` is running and has produced no output**
(`wc -c` = 0 at the moment of writing) — the derivation below is therefore a
commitment, not a rationalisation. P4/P5 likewise precede their runs.

## 1. What ledger part 1 already measured (the inputs to P3–P5)

CPU `memory_analysis().temp_size_in_bytes` of the compiled
`value_and_grad(loss_fn)` at the **true pilot store geometry** (`addr 8 /
payload 4 / capacity 32 / atoms_per_item 256` ⇒ 8192 atoms, `dim 12`,
`address_steps = read_steps = 64`, `retry_rounds = 1`, `traj_stride 8`,
`d_model 512`, `chunk 64`, `write_inner_steps ∈ {4, 40}`), at `batch = 1`,
`n_layers = 1`, sweeping `n_chunks`:

| `n_chunks` | remat OFF | `remat_chunks=True` |
|---|---|---|
| 1 | 645.5 MB | 645.5 MB |
| 2 | 1935.1 MB | 648.2 MB |
| 4 | 3223.9 MB | 651.3 MB |

Least-squares on `C ∈ {2, 4}` (the `C = 1` point is degenerate — a length-1 scan
has nothing to stack):

* OFF: `temp(C) ≈ 646.2 MB + 644.4 MB · C`
* ON : `temp(C) ≈ 645.1 MB + 1.55 MB · C`

⇒ **P1 CONFIRMED**: the per-chunk marginal term falls **644.4 → 1.55 MB, a
416× reduction**, and the residual 1.55 MB/chunk is *quantitatively* the
`StoreState` carry (8192·12 + 2·8192 + 32·12 floats = 458 KB) plus the shell's
own per-chunk residuals (conv window 4·64·512·4 B = 512 KB + MLP hidden
64·2048·4 B = 512 KB ≈ 1.02 MB) — 1.48 MB predicted vs 1.55 MB measured. The
mechanism is identified, not just the number.

## 2. The predictions

**P3 — layer and batch scaling of the rematted peak.** Under `remat_chunks`,
only one chunk interior is live at a time *within a layer*. Layers are
sequential (layer `l`'s backward needs layer `l+1`'s cotangent), so I predict
the ~645 MB transient is paid **once, not `n_layers` times**, while the
`1.55 MB · C` persistent term is paid per layer. `vmap` over the batch has no
such serialisation, so **everything multiplies by `batch`**. Formally:

```
peak_ON(B, L, C)  ≈  B · [ 645 MB  +  1.55 MB · C · L ]        (P3a)
```

The competing hypothesis I am registering against is that XLA keeps every
layer's interior live:

```
peak_ON(B, L, C)  ≈  B · L · [ 645 MB + 1.55 MB · C ]          (P3b)
```

**These differ by 12× at pilot scale and that is the whole question.** The
discriminating cells are `(B=1, L∈{1,2,3}, C=2)`: P3a predicts
`648 / 651 / 654 MB` (flat, +3.1 MB per layer); P3b predicts
`648 / 1296 / 1944 MB` (doubling, tripling). ⚠ I put **0.75** on P3a, 0.25 on
P3b — the CPU backend's buffer assignment is a different scheduler from the
GPU's and I have seen it hold buffers longer than the data dependency requires.

**P4 — the projected pilot peak.** The GPU's own number for the OFF config is
the 97.82 GiB in run 1's crash. I do **not** extrapolate the CPU's absolute
bytes to the GPU (the CPU model would predict ≈ 1.05 TB for
`B=8, L=12, C=16`, i.e. 10× the GPU's figure — the GPU's XLA has already
applied automatic rematerialization, which is exactly why its message quotes an
auto-remat floor of 76.70 GiB). What I extrapolate is the **ratio measured at
the true geometry**:

```
projected_ON  =  97.82 GiB · [ (a + b_on·16) / (a + b_off·16) ]
              =  97.82 GiB · (671.0 / 10956.6)
              =  5.99 GiB              (P4, central)
```

I pre-register the **central estimate 5.99 GiB** and, because a single-scheduler
ratio transfer is the weak link, a **conservative bracket of 5× that = 30 GiB**.
Both are **< 72 GB**, so I pre-commit: *the fix passes the acceptance memory bar
if the measured layer scaling is P3a, and it still passes under P3b* (P3b's
worst case is `8 · 12 · 645 MB = 61.9 GiB` transient + `8 · 12 · 16 · 1.55 MB =
2.4 GiB` persistent = 64.3 GiB, which clears 72 GB but with only 11 % headroom
— so **P3b would make me report the fix as MARGINAL and recommend
`accum_steps = 2` in the submission line**, which halves it to 32 GiB).

**P5 — the wall-clock price.** `jax.checkpoint` recomputes the chunk interior's
forward once inside the backward. The forward is ~1 unit, the backward ~2 units,
so the step goes `3 → 4` units ⇒ I predict a **1.33× slowdown of the
differentiable pass**, and — because the pilot step *also* pays the plan pass
and the optimiser — a **smaller end-to-end slowdown, which I register as
≤ 1.35× and predict at ≈ 1.15–1.30× measured on the toy end-to-end runner.**
⛔ Registered falsifier for the wallclock half of acceptance: an end-to-end toy
slowdown **> 2.0×** would mean 4000 steps × 3 seeds no longer fits `-t 12:00:00`
and I must say so and stop.

**P6 — rung 2 (`remat_read_segments`) buys nothing once rung 1 is on.**
Ledger part 1 already showed `remat_chunks + remat_read_segments=4` at
`C = 4` costs 655.5 MB vs 651.3 MB for rung 1 alone — i.e. **+0.6 %, a
regression, not a saving**. I pre-register the *explanation* to be checked: the
rematted chunk interior's peak is set by the **write** (40 inner steps × 8
perturbations of a differentiated `write_loss`), not by the read, so segmenting
the read cannot lower it. The check: with `write_inner_steps = 4` instead of 40
the ON floor should be **materially lower** if the write dominates. ⚠ Ledger
part 1's `w=4` and `w=40` rows are byte-identical (645,503,176 both), which
already *contradicts* that explanation — so I register the alternative: the
645 MB floor is neither read nor write residuals but a single large buffer
(most plausibly the `(n_atoms, dim)` broadcast inside one force evaluation,
vmapped/fused), and rung 2 is therefore **structurally unable to help** and
should be reported as a lever that exists, is bit-identical, and is **not
recommended for the resubmission**.

## 3. What would falsify the whole task

* any bpc or `WritePlan` field changing beyond the declared tolerance
  (**already measured: loss and all 7 plan fields BITWISE identical**, gradient
  `||dg||/||g|| = 8.67e-10` vs a 1e-8 gate and a 1.19e-7 float32 ULP);
* a projected pilot peak **> 72 GB** with all levers on;
* a toy end-to-end slowdown **> 2.0×**.

Losing wall-clock to remat is *not* a falsifier — it is the trade the lever
exists to make (task dial declaration, "does NOT falsify").
