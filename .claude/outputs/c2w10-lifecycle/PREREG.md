# PREREG — c2w10-lifecycle-mechanics (spoke S2)

**Filed by the experiment-engineer spoke, 2026-08-10, BEFORE the lifecycle harness ran and BEFORE the
L5-b equivalence test was pinned to any number.** This file does **not** re-register L1–L7 — those are
registered in `PREREG-C2W10.md` §4 and I implement them as written. What is registered here is only
what the parent prereg left to the spoke: **the parameter derivations, the L5-b tolerance, the f_max
decision, the decimation-ladder selection rule, and the operating point** — i.e. every number I would
otherwise be free to choose after seeing a result.

Base: `main @ 9e0bb25`. Branch `agent/experiment-engineer/c2w10-lifecycle-mechanics`, worktree
`../CHLU-c2w10`. Main venv reused (no worktree `uv sync`); JAX **0.9.0**.

---

## P0. The L5-b reconciliation, resolved from the RAW artifact BEFORE the test is pinned

The task file and `PREREG-C2W10.md` §4 L5-b both attribute `events 27/40/70, rates 0.593/0.050/0.043`
to the **`p1_on_i1_on`** cell. **That attribution is wrong**, and I state the correction before I write
the equivalence test, per the reconciliation the task assigns me.

Read directly out of `.claude/outputs/c2w6-anti-erosion/erosion_*_records.json` (field
`n_rewrite_events`, `n_rewrite_violations`, `rewrite_violation_rate`, per seed):

| C2W6 cell | flags | events/seed | pre-guard violations | pre-guard rate |
|---|---|---|---|---|
| `p1_off` | P1 OFF, I1 guard OFF | **27 / 40 / 70** | 16 / 2 / 3 | **0.593 / 0.050 / 0.043** |
| `p1_on` | P1 ON, I1 guard OFF | 37 / 78 / 53 | 1 / 0 / 0 | 0.027 / 0.0 / 0.0 |
| `p1_on_i1_on` | P1 ON, **I1 guard ON** | **44 / 62 / 59** | **6 / 0 / 0** | 0.136 / 0.0 / 0.0 |

⇒ **`[0.593, 0.050, 0.043]` is the `p1_off` GUARD-OFF BASELINE, not the I1 arm.** Add.7 §A22's
"OFF `[0.593,0.050,0.043]` → ON `[0.027,0.0,0.0]` under P1" is the **P1** contrast (`p1_off` →
`p1_on`) and is **correct as printed**. §A23.2's "0 post-guard violations" is the **I1** leg
(`p1_on_i1_on`), where the raw `n_rewrite_violations = 6/0/0` counts **pre-guard** events — i.e. the
events the guard *repaired* — because `exp_anti_erosion.post_guard_violations`' own docstring says the
write's `violation` flag is the pre-guard verdict and "on a guard-ON arm it counts REPAIRS, not
failures". `n_rewrite_violations_post_guard` is **absent (null) in all three artifacts** (the function
post-dates them), so the "0 post-guard violations" claim must be **recomputed** from the per-event
depths that ARE in the artifact (`depth_before`, `depth_after`, `depth_guarded`, `refresh_factor`).

**Registered prediction P0-a:** recomputing `post_guard_violations` (imported read-only from
`exp_anti_erosion`) over the stored `rewrite_events` gives **0 / 0 / 0** for `p1_on_i1_on` and
**16 / 2 / 3** for `p1_off`. Prior 0.85. If it does not, §A23.2's "zero post-guard violations" is
itself unsupported and that is a finding I report rather than paper over.

## P1. The L5-b equivalence test and its DECLARED TOLERANCE (registered before it is run)

The store-level guard is validated **on C2W6's own recorded rewrite events** — the 44/62/59
`p1_on_i1_on` events and the 27/40/70 `p1_off` events are replayed as `(depth_before, depth_after)`
pairs through my store-level guard's arithmetic, and its output is compared to `blocks.py`'s recorded
`refresh_factor` and `depth_guarded`.

Registered equivalence claims and tolerances:
- **E1 (violation flag):** my pre-guard violation predicate `d_after < d_before` reproduces the
  recorded `violation` flag on **100.0 %** of events. Tolerance: **0 mismatches**.
- **E2 (refresh factor):** my factor, expressed in `blocks.py`'s **amplitude** units
  `f = clip(sqrt(d_before/d_after), 1, refresh_max_gain)` on violation events and exactly `1.0`
  otherwise, matches the recorded `refresh_factor` to **rtol 1e-6** (both are float32 products of the
  same two float32 depths). Tolerance: **max relative deviation ≤ 1e-6**.
- **E3 (post-guard monotonicity):** replaying through my guard leaves **0** post-guard violations on
  every seed of both cells, at `tol = 1e-6` (the shipped `post_guard_violations` tolerance).
- ⛔ **A divergence beyond E1/E2/E3 FAILS L5-b** and the leg ships `false` with the numbers.

⚠ The two implementations differ in *where* the factor is applied (`blocks.py`: `amp * f` on the
slot's rows; store level: `LearnedVStore.scale_group_amplitude(slot, f**2)`, whose docstring says it
scales **depth** by its argument and the amplitude parameter by its square root). **This is the same
arithmetic in different units, and E2 is stated in the amplitude units so the comparison is direct.**

## P2. L1's hysteresis parameters — the derivation (this is the part I must not choose after the fact)

`PREREG-C2W10.md` §4 L1: "`read_hits` in the trailing window ≥ `h_hi` sustained over ≥ `d_dwell`
chunks", designed negative "a single burst reaching `h_hi` does not promote".

**Derivation (arithmetic, not taste).** Let the trailing window be `W` chunks. A single burst of
`h_hi` hits in chunk `c` keeps the trailing-window count at `≥ h_hi` for chunks `c, c+1, …, c+W-1`,
i.e. for **exactly `W` consecutive chunks**, and for no more. Therefore the dwell requirement binds —
and the burst negative can fire at all — **iff `d_dwell > W`**. With `d_dwell ≤ W` the designed
negative is *arithmetically incapable of failing the burst*, which is precisely the vacuous-guard
defect class C2W8 caught twice.

⇒ **Registered:** `window = 2`, `d_dwell = 3` (`d_dwell > window` is asserted at construction and
raises `ValueError`), `h_hi = 2`, `h_lo = 1`, `d_demote = 2`, `k_streams = 3`.
**Registered prediction P2-a:** with these values the burst negative passes on the first build
(a burst of `h_hi` hits reaches `dwell = 2 < 3`). Prior 0.90.

## P3. L4 `f_max` — NOT re-derived

I adopt the Hub default **`f_max = 0.25` of the well budget** and explicitly do **not** re-derive it;
the parent prereg permits either and requires the choice be declared. Protected cap
`= floor(f_max · budget)`, evaluated **before** each promotion; the promotion is **refused** (never
partially applied) when the cap is reached, and `protected_saturation` trips at that moment.
**Registered prediction P3-a:** forcing every item's usage high leaves `n_protected ==
floor(0.25·budget)` exactly, with `≥ 1` refused promotion and the monitor tripped. Prior 0.90.

## P4. L6 netting — the three assertions, and how each is made non-vacuous

Netting is `designed_decay_factors` (imported read-only from `chlu/core/well_lifecycle.py`), replayed
per item: `depth_netted(i,t) = depth_raw(i,t) / Π factors(i, ≤t)`.
- **N-a** `leak = 0` ⇒ every factor is exactly `1.0` ⇒ `netted` is returned **bitwise** as `raw`
  (asserted with `==` on the float64 bit pattern, not `allclose`).
- **N-b** `leak > 0, Δt > 0` ⇒ `netted > raw` **strictly**.
- **N-c** a well with **no writes after admission** nets to the analytic `exp(−leak·Δt)`:
  `Π factors == exp(−leak·Δt)` to **1e-9**, `Δt` = ticks the item was live.
**Registered prediction P4-a:** N-c holds to ≤ 1e-9 because the controller's per-tick law is exactly
`amps *= exp(−leak)` in float64 and the factor is the recorded ratio. Prior 0.85 (the residual risk is
float32 rounding inside `scale_group_amplitude`, which is why the assertion is on the **controller's**
factors, not on the learned store's depths).

## P5. Decimation — the ladder and the selection rule, registered BEFORE the probe

Ladder `m ∈ {1, 2, 5, 10}` (registered in `PREREG-C2W10.md` §2.2b). Selection rule, fixed now:
**the smallest `m` on the ladder whose measured wall-clock for one seed of the lifecycle cell is
≤ 2 h**, subject to the structure-preservation pytest passing at that `m`. Structure preservation is
**asserted, not claimed**: all three cycles and both change points present at the chosen `m`, **with
counts**, and the same test **fails** at an `m` that destroys either.
⚠ `BENCHMARK-GATE.json` did not exist at spoke start (`.claude/outputs/c2w10-benchmark-gate/` is an
empty directory). **If it is still absent when I reach the real-stream legs, those legs are a declared
NOT-RUN with that reason, the decimation *machinery* + structure-preservation pytest still ship, and
they are exercised on the synthetic (whose cycles and change points are known by construction).**
I do **not** re-download or re-freeze the stream.

## P6. The operating point, and the `n_live_max ≥ 64` risk, declared in advance

The I2 analyst spoke gates on `n_live_max ≥ 64`. Registered target: `well_budget = 64`,
`capacity = 72`, `addr_dim = 12` (d = 16 is a declared NOT-RUN, measured inert), carried rig facts ON
(`atom_site_local_init = True`, Wendland kernel, width co-scaled to the seed's **measured** key
spacing), `write_steps` fixed by the pricing probe with an **N94 floor of 40** and a **≤ 2 h/seed**
target, `n_seeds = 3`.
**Registered prediction P6-a:** `n_live_max = 64` is reachable within the wall-clock target. Prior
0.70. **If it is not, the telemetry file says so explicitly and it is a NOT-RUN, not a null** — the
parent prereg's I2-a rule, applied to my own deliverable.

## P7. What would make me report a leg `false`

Stated in advance so it is not negotiated later: any leg whose designed negative **cannot be shown to
fail** (guard disabled ⇒ the bad behaviour does not appear) ships `false` with the reason string, even
if its positive test is green. K-C: if L3's target population is empty at the measured operating
point, the verb is reported **UNEXERCISED** — `l3.designed_negatives` still have to be green on the
planted population, but `l3.exercised_on_stream = false`.
