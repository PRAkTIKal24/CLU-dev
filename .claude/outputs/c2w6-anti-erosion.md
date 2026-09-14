# c2w6-anti-erosion — experiment-engineer report

**Task + acceptance criterion:** build §A20.6's **P1 stop-gradient partition**, **I1
refresh-on-rewrite monotonicity** and **I2 usage-vs-erosion telemetry**, with K1/K2 green as
TESTS before any science cell; deliver the erosion curves, adjudicate E1–E3/I1/I2 and the
P-residual interaction against the registered bands **mechanically**, and emit the CSF3
run-3 gate verdict + flag block either way. **Status: done** (see §0 for the two declared
cuts).

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5, first-10-lines rule)
> **R1 — ⭐⭐ N223's erosion DOES NOT REPRODUCE at the CSF3 run-2 config.** Over **1000**
> outer steps × 3 paired seeds the partition-OFF arm's fitted depth ends at
> **0.708 ± 0.57×** its untrained value and **3.74 ± 3.0×** its own step-200 value — it does
> not decay, it **recovers**. Registered **E1 is REFUTED 0/3 seeds** (rule: ≤0.5×). The
> banked "depth → 4.95e-63 after 200 steps" (probe §7 R3) was measured at the **shipped**
> config (no placement, `write_margin=0.15`, pre-ψ-fix); placement + margin + the ψ payload
> residual have between them removed it. *(Owner: Hub — every artifact that says "the outer
> loop destroys the store" needs the qualifier "**at the shipped config**"; §7.27's watch
> item and the C2W6 charter row are both scoped to a config that is no longer the one being
> submitted.)*
> **R2 — the gate verdict is `FAILS_FLATTEN`, and the failing leg is the BASELINE, not P1.**
> E2 passes 3/3, K3 passes (P1 is **better** by −0.0049 ± 0.0008 bpc, 6.2 SE, 3/3 paired
> seeds), K4 does not fire; the gate fails only because there was no erosion to flatten.
> *(Owner: Advisor — the registered rule is applied verbatim and the verdict stands as
> printed; whether "no harm + no erosion to fix" earns the run-3 slot is a promotion
> decision, not a measurement.)*
> **R3 — ADDENDUM-1's own prediction A1-3 is REFUTED and the parent prereg's I1-a is
> CONFIRMED.** I registered (before the cells, from an untrained-model count) that rewrite
> events were structurally impossible at this rig. After training they occur: **27/40/70
> events per run**, violation rate **0.593 / 0.050 / 0.043**, mean **0.228 ± 0.182** — inside
> the parent prereg's registered 10–40 % point band. φ's drift changes the allocator's slot
> choices; an untrained count is not a rig property. *(Owner: me — recorded here; no
> downstream artifact quotes A1-3.)*

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)

- **Dial:** **lifetimes/isolation** (training-time protection of written content) — tier-iii
  component build. ⛔ **No paper number**; CSF3 run-3 config evidence + ablation rows.
- **Laundering control:** live vs blank vs memory-deleted, paired seeds, on BOTH arms;
  **K4** (the relocation detector) is applied to the gate verdict and **did not fire**
  (ON-arm memory-deleted margin +0.105/+0.124/+0.226 bpc, i.e. the memory is a net *cost*
  that the swap detects, and |live−blank| is 3.1e-04…1.9e-03, off the float32 floor).
- **Falsifies / does not falsify:** per prereg §3 + ADDENDUM 1; scored mechanically in
  §4 (the gate) and §7 (the full scorecard).
- ⚠ Monitor #13 / N94 travels with every w4 reading; the **w40 pair** is the undemoted
  confirmation (§3–§4) — and it is the cell where erosion actually appears.
- ⛔ **Depth is still NOT quotable as feature importance.** I2's primary statistic reads
  ρ = **−0.257 ± 0.151** (read-selection) / **+0.067 ± 0.163** (leave-one-well-out Δbpc) —
  the registered branch is **NO_USAGE_STRUCTURE**, so the charter §A21 caveat **stays
  active**. The Head's hypothesis (ρ ≥ +0.5) is not confirmed.
- ⛔ TOY scale (0.16 M). ⛔ "CLU-former" is a placeholder and appears nowhere.

---

## 0. FLAG PROVENANCE (every number in this report)

| item | value |
|---|---|
| branch | **`c2w6-anti-erosion`** (the wave branch), worktree `../CHLU-c2w6`, base local `main @ 104ca19` |
| commits | `ffe7440` · `ae47a66` · `251c2c1` · `cc5e49b` · `d8c9fa7` · `b623d56` (see §10) |
| env | **main venv reused** (`/Users/user/Desktop/CHLU/.venv`), **no worktree `uv sync`** (w6 hazard avoided). **JAX 0.9.0**, equinox 0.13.4, optax 0.2.6, CPU |
| **seeds** | **0, 1, 2 on every cell**, paired (identical shell, φ-gain, data order, plan pass at step 0; cells differ ONLY in the lever). SE = sample sd (ddof=1)/√n |
| **scale** | ⛔ **TOY (0.16 M): `d_model 64`, `n_layers 2`, `seq_len 512`, `batch 4`, `chunk 32`** |
| data | **enwik8**, byte-level, canonical 90/5/5 split, first 4 000 000 B; curve/telemetry on the **valid** split's first contiguous batch (lane 0), held-out bpc on **test** (4 batches) |
| rig | the **CSF3 run-2 config**: `atom_place_radius=0.3`, `write_margin=0.6`, `psi_payload_residual=True`, `psi_residual_source=q_star`, `psi_residual_gain=1.0`, `psi_residual_trainable=True`, all stage flags TRUE, `soft_certificate=True`, `leak 0.02`, `retry_tau 0.5`, `retry_max_rounds 1`, `capacity 8`, `budget 6`, `n_atoms 1024`, `atom_width 0.3`, `atom_depth_init 1e-4`, `dim 3` (`addr 2` + `payload 1`) |
| read | `dt 0.05`, γ_addr 0.05, γ_read 0.02 (both trainable), 24+24 (+24 gated retry) Verlet, `traj_stride 8`, `kinetic newtonian_learned`, read_mode **trajectory**, ψ = `DeepSetsPsi(hidden 32, depth 2)` |
| write | masked/local, **4 inner sign-SGD steps at lr 0.05** (**40** in the `w40_*` cells), `n_perturb 8`, σ_addr 0.25, σ_pay 0.6, barrier 0.2, `barrier_pairs "nn"` |
| **the levers under test (both ship OFF)** | `erosion_partition` (False) · `refresh_monotonic` (False) · `refresh_max_gain` (4.0) · `refresh_amp_ceiling` (0.0) |
| optimiser | AdamW, warmup-cosine, peak `lr 1e-3`, `warmup 100`, `grad_clip 1.0`, `wd 0.0` |
| **horizon** | **1000 outer steps** (w4 cells) / **400** (w40 pair); depth sampled every **`monitor_every = 25`** ⇒ 41 (17) readings/run |
| langevin / temperature | **N/A** — deterministic read, `T = 0`, `p₀ = 0`, no Langevin step |
| wake–sleep / Lyapunov | **N/A** — third training path (`train_cluformer.py`), byte-LM cross-entropy |
| artifacts | `.claude/outputs/c2w6-anti-erosion/` — `PREREG-AntiErosion-ADDENDUM-1.md` · `erosion_{p1_off,p1_on,p1_on_i1_on,w40_p1_off,w40_p1_on,resoff_p1_off,resoff_p1_on}_records.json` · `erosion_aggregate.json` · `TABLES.md` · `erosion_curves.png` · `k2_reference_seed0.json` + `k2_capture.py` · `logs/` |
| CSF3 | **0 A100-hours.** Not reachable from this machine |

**⛔ Declared cuts (never reported as nulls).** (i) The retrained `none` (memory-deleted) arm
is **flag-independent** — no flag under test can reach it — so it is trained once per seed
inside the `p1_off` cell and quoted for every cell at that seed; the *eval-time* swap
(`NullMemoryCell` in the trained block) is computed on **every** cell/seed. (ii) The
trainer's own `store_watch` second depth series is ON for the three w4 claim cells and OFF
for the w40 pair and the diagnostics, where it costs a 40-step write replay per window for a
quantity the telemetry already measures.

---

**Wall clock.** 21 cell-seeds, ≈ **6.9 h** measured, 3–4 processes in parallel on 8 cores:
w4 claim cells 1348–1706 s each · **w40 pair 1906–2986 s each** (priced at ~2.0 s/step
before it ran, per the task; **no seed was cut** — the cell fit) · diagnostic rider
1459–1877 s each. Tests 63 s (new) + 768 s (regressions).

---

## 1. THE BUILD (what shipped, and what it deliberately does not do)

**P1 — `StreamMemoryConfig.erosion_partition` (ships OFF).** A `stop_gradient` at the **write
boundary**, in three legs, because the outer loss reaches the depth-determining leaves by
three routes and severing one is not a partition:

| leg | site | why it exists |
|---|---|---|
| 1 | the `StoreState` a write **returns** | the chunk chain: every later read of a written store |
| 2 | `init_state()`'s atom leaves | the **chunk-0** read sees the init leaves directly — same parameters, same erosion, one chunk earlier |
| 3 | `self._atoms` inside `write` | the **eviction re-draw** reads the same init leaves |

⛔ **The read channel is deliberately untouched** and that is half of the acceptance: ψ, the
friction selectors, the mass selector and φ's **query** gradient all keep their gradients
(§2). Off, it is **bit-identical and parameter-count-identical** — no leaf is added; this is
gradient plumbing, so `cell_ledger()` is unchanged on every arm.

**I1 — `refresh_monotonic` (ships OFF)** + `refresh_max_gain` (4.0) + `refresh_amp_ceiling`
(0.0 = off). An admitted **rewrite** — a write into a slot that already holds an item and is
**not** being evicted — may never leave the well shallower *at the incoming item's own site*
than the state the inner write started from; the slot's `amp` rows are rescaled by
`f = sqrt(D_before/D_after)` clipped into `[1, refresh_max_gain]`. **Designed channels are
excluded from the accounting by construction** (the reference is taken after decay, eviction
re-draw and localized placement), and the guard **only restores** — it can never end deeper
than `max(D_before, D_after)`, adds no leaf and no state byte (K5), and multiplies `amp` by
exactly `1.0` when it does not fire, which is bit-identical (I1-b).

**I2 — the telemetry** (always on in the harness, decision-inert to the model): per well, per
monitor window — fitted depth · last-write chunk · read-selection proxy ·
`‖∂L_outer/∂(this well's atom rows)‖` · leave-one-well-out probe bpc at 4 checkpoints; plus
the **rewrite audit taken from the write's own code path** (`write_and_diag`, one pass — two
passes would pay the inner unroll twice, which at w40 is the whole instrument budget) and
the **interference audit** (§5.2).

⛔ **Untouched, as declared:** `chlu/config.py`, `psi_readout.py` (the AttentionPsi
quarantine), `monitors.py` (I1 wants no monitor row — the check lives in the block and the
experiment; **flagged for the Hub** in case the registry should gain one), `clu_system.py`,
`eval/**`, `scripts/csf3/*` (no new env passthrough was needed: every flag reaches the
cluster through the existing `--mem` path).

---

## 2. ⭐⭐ K1 — PARTITION INTEGRITY, MEASURED IN-SYSTEM (not asserted)

The gradient probe on the **full two-layer block** at the run-2 config, seed 0, one batch
(`k1_in_system.json`, reproducible with `k1_insystem.py`):

| leaf | P1 OFF (L0 / L1) | P1 ON (L0 / L1) | verdict |
|---|---|---|---|
| atom `centers` | 1.967e-03 / 2.388e-03 | **0.0 / 0.0** | severed |
| atom `log_width` | 1.669e-03 / 1.559e-03 | **0.0 / 0.0** | severed |
| atom `amp` | **9.347e-03** / 7.154e-03 | **0.0 / 0.0** | severed — this is N223's channel |
| `log_mass` | 2.960e-03 / 1.594e-03 | 2.990e-03 / 1.594e-03 | ✅ read channel alive |
| `log_gamma_addr` | 1.162e-03 / 9.428e-04 | 1.192e-03 / 9.428e-04 | ✅ alive |
| `log_gamma_read` | 7.924e-04 / 7.986e-04 | 8.017e-04 / 7.986e-04 | ✅ alive |
| `psi` | 6.398e-02 / 3.180e-02 | 6.423e-02 / 3.180e-02 | ✅ alive |
| `psi_res_gate` | 7.070e-03 / 6.589e-03 | 7.114e-03 / 6.589e-03 | ✅ alive |
| **`phi`** | 9.083e-02 / 2.768e-02 | **6.593e-02** / 4.350e-02 | ⭐ see below |
| loss | 5.764082908630371 | 5.764082908630371 | **identical bit-for-bit** |

`exact_zero: true` on **all three leaves in both layers** — not small, exactly 0.0 (K1 ✅).
The forward is unchanged (identical loss), so this is a backward-only intervention.

> ⭐⭐ **A mechanism finding nobody had named: at the run-2 config the write→φ channel is NOT
> closed by sign-SGD.** The `write_sign=True` docstring argues that `jnp.sign`'s zero
> derivative severs `d(store state)/d(phi)`, making the trajectory read the only channel to
> φ "by construction as well as by theorem". That is true of the *inner-loop* path — but
> **H1b's localized placement re-opens it**: `centers[:, :addr] = z[:addr] + jig` is a
> plain differentiable assignment of φ's output into the store's state, outside the
> sign-gated loop. It is live in the run-2 config (`atom_place_radius = 0.3`). Measured:
> **27 % of φ's layer-0 gradient (0.0908 → 0.0659) flows through the write's placement
> path**, and P1 is what closes it. *(Layer 1's φ gradient RISES 0.0277 → 0.0435 — the
> layers are not independent once layer 0's write path is cut.)*

**K2 — bit-identity OFF, in its strong form.** Fingerprints captured from `main @ 104ca19`
**before** the build (`k2_capture.py` → `k2_reference_seed0.json`) and inlined into
`tests/test_anti_erosion.py`: the 16-chunk written state's four leaves (sha256 of the float32
bytes), the read output, the loss `5.764082908630371` and the three atom-gradient hashes all
reproduce **exactly** with the flags off, as does `cell_ledger()`
(`params 8617 / state_floats 5144 / state_bytes 20576`). **K2 ✅.**

**K5 — I1 is not a hidden capacity increase.** `cell_ledger()` is identical with both flags on
(same leaf count, same state bytes); the guarded depth is `<= max(D_before, D_after)` on a
constructed violation; the factor is capped. **K5 ✅** (three tests).

---

## 3. ⭐⭐⭐ THE HEADLINE — N223's EROSION DOES NOT REPRODUCE AT THE RUN-2 CONFIG

**Erosion curve** = median fitted well depth (live items, own site on the launch manifold),
lane 0 of a fixed validation batch, every 25 outer steps, 41 readings/run, 3 paired seeds.
Figure: `erosion_curves.png`. Full per-seed table: `TABLES.md` T1.

| cell | depth final | **final / untrained** | final / step-200 | bpc |
|---|---|---|---|---|
| `p1_off` (w4) | 0.1134 ± 0.093 | **0.708 ± 0.57** | 3.74 ± 3.0 | 3.710 ± 0.045 |
| `p1_on` (w4) | 0.1142 ± 0.055 | 0.603 ± 0.23 | 3.51 ± 1.9 | 3.706 ± 0.045 |
| `p1_on_i1_on` (w4) | 0.1495 ± 0.102 | 0.744 ± 0.34 | 4.73 ± 3.4 | 3.696 ± 0.038 |
| **`w40_p1_off`** | 0.2116 ± 0.11 | 1.206 ± 0.80 | 1.00 ± 0.50 | 4.307 ± 0.015 |
| **`w40_p1_on`** | 0.3176 ± 0.16 | 1.832 ± 1.20 | 0.913 ± 0.38 | 4.313 ± 0.010 |
| `resoff_p1_off` ⛔DIAG | 0.03164 ± 0.025 | **0.141 ± 0.095** | 1.55 ± 0.83 | 3.696 ± 0.035 |
| `resoff_p1_on` ⛔DIAG | 0.1073 ± 0.069 | 0.495 ± 0.27 | 0.986 ± 0.28 | 3.702 ± 0.033 |

- ⛔⛔ **E1 IS REFUTED, 0/3 seeds.** The partition-OFF arm's step-1000 depth is **9.78 /
  0.903 / 0.530 ×** its step-200 value against a registered rule of ≤0.3× (band ≤0.5×). The
  store does not decay over 1000 steps at the run-2 config — it dips and **recovers**.
  Against the untrained reading it ends at 0.135 / 1.85 / 0.140 ×, i.e. one seed ends
  **deeper** than it started. Nothing here resembles `0.0288 → 4.95e-63`.
- ⭐ **The diagnostic rider says why, and it is the §A20.6 "intrinsic vs symptom" question
  answered:** with the **ψ payload residual OFF** — the store useless to the loss — depth
  falls to **0.141 ± 0.095 ×** untrained (0.019 / 0.075 / 0.329), a 5.0× lower retention than
  the same arm with the residual on (0.708 ± 0.57), and at seed 0 the store is measurably
  inert (`live − blank = +1.7e-06`, the pilot's float32 signature). **Erosion is a symptom of
  uselessness, not an intrinsic property of a differentiably-unrolled write.** ⛔ DIAGNOSTIC
  label travels: this is the labelled 2×2 corner, never a claim cell.
- ⭐⭐ **At N94's maturity floor the picture changes and P1 wins on depth.** In the `w40`
  pair — the only readings monitor #13 does not demote — the paired ON/OFF **final-depth
  ratio is 21.53 / 1.52 / 1.46 (3/3 seeds ON deeper, geometric mean 3.63×)**, and at seed 0
  the OFF arm does collapse (0.0730 → 2.12e-04, ×0.0029) while the ON arm holds 21.5× more
  depth (4.57e-03). **The w4 cells are the ones that show no erosion; the undemoted cell
  shows it on 1/3 seeds and P1 protects against it on 3/3.**
- **P1 is not paid for in bpc.** Paired Δbpc(ON−OFF): **−0.00485 ± 0.00078** at w4 (3/3 seeds
  *better*, 6.2 SE — inside E3's ±0.01 band and on the good side), **+0.00568 ± 0.00561** at
  w40 (not significant, inside the band). P1+I1 vs OFF: **−0.0140 ± 0.0078**.
- ⚠ **Seed variance dominates everything at this scale.** The untrained depth itself is
  0.046 / 0.162 / 0.249 across seeds; SEs on the ratios are as large as the means. No number
  in this section is a performance claim, and none may be quoted without its SE.

---

## 4. THE GATE — `FAILS_FLATTEN`, on the baseline leg, at BOTH budgets

`aggregate()` applies prereg §4 verbatim (`TABLES.md` T5, `erosion_aggregate.json`):

| leg | w4 (`p1_on` vs `p1_off`) | w40 (`w40_p1_on` vs `w40_p1_off`) |
|---|---|---|
| **E2** ON flattens (≥0.5×, 3/3) | ✅ **passed** (7.07 / 0.539 / 2.94) | ❌ 2/3 (0.154 / 1.401 / 1.183) |
| **E1** OFF decays (≤0.5×, ≥2/3) | ❌ **0/3** (9.78 / 0.903 / 0.530) | ❌ 1/3 (0.160 / 1.877 / 0.969) |
| **K3** bpc not worse | ✅ (−0.0049 ± 0.0008) | ✅ (+0.0057 ± 0.0056) |
| **K4** not relocated | ✅ did not fire | ✅ did not fire |
| **VERDICT** | **`FAILS_FLATTEN`** | **`FAILS_FLATTEN`** |

⛔ **The verdict stands as printed** — the rule was registered and is applied mechanically.
**But the leg that fails is E1, the BASELINE**: there was no erosion to flatten. P1 passes
every leg that is about P1. That distinction is the whole content of R2 and it is the
Advisor's call, not a measurement.

**K4, explicitly (the laundering control, mandatory on the verdict).** It does **not** fire
on either arm at either budget: the ON arm's `|live − blank|` is 3.1e-04 … 1.95e-03 (off the
float32 floor) and the **memory-deleted margin is positive on 6/6 cell-seeds**
(+0.105/+0.124/+0.226 bpc at w4; +0.0035/+0.0125/+0.0233 at w40). ⚠ **Positive means the
memory is still a NET COST** — the block does better with the memory deleted — which is the
pilot's standing finding, unchanged by this wave, and it is why "protected" cannot be
upgraded to "useful" here. The retrained `none` arm agrees with the eval-time swap
(+0.072/+0.091/+0.155 vs +0.078/+0.125/+0.197).

---

## 5. I1 — THE REWRITE AUDIT

### 5.1 I1-a (guard OFF) — measurable after training, and ADDENDUM-1's own prediction is refuted

| cell | rewrite events / seed | violation rate / seed | mean ± SE |
|---|---|---|---|
| `p1_off` | 27 / 40 / 70 | 0.593 / 0.050 / 0.043 | **0.228 ± 0.182** |
| `p1_on` | 37 / 78 / 53 | 0.027 / 0 / 0 | 0.009 ± 0.009 |
| `p1_on_i1_on` | 44 / 62 / 59 | (guard fired 6 / 0 / 0) | — |
| `resoff_p1_off` ⛔DIAG | 34 / 43 / 67 | 0.912 / 0.535 / 0.090 | **0.512 ± 0.238** |
| `w40_p1_off` | 0 / 34 / 30 | 0 / 0 / 0 | 0.000 |

- ✅ **The parent prereg's I1-a is CONFIRMED**: 0.228 ± 0.182 sits inside the registered
  10–40 % point band and inside the 2–60 % band.
- ⛔ **My ADDENDUM-1 §A1-3 prediction ("0 events, structurally") is REFUTED.** It was derived
  from an *untrained* count (13/19/16 admits, 0 slot reuse, 3 seeds) plus the arithmetic
  `n_chunks = 16 ≤ capacity = 8`. After training, φ's drift changes the allocator's slot
  choices and clean rewrites appear. **An untrained count is not a rig property** — a
  pre-registered prediction that fails is a finding, and this one is mine.
- ⭐ **The rate tracks uselessness, not the partition:** the residual-OFF arm rewrites
  destructively **51.2 ± 23.8 %** of the time (0.912 at seed 0) against 22.8 % with the
  residual on and **0.9 %** with P1 on. A store the loss cannot use is also a store whose
  rewrites damage it.
- ⚠ At **w40 the channel is silent**: 64 events, **0 violations** — a 40-step inner write
  always re-digs deeper than it found. I1's value is a w4-budget artifact at this rig.

### 5.2 I1-b (guard ON) — CONFIRMED, and the pre/post distinction matters

`p1_on_i1_on`: the guard **fired 6 times at seed 0** and **post-guard violations are
0 / 0 / 0 on 3/3 seeds** across 165 rewrite events. ⚠ The first version of this table read
"6/44 violated" because the write's own `violation` flag is the **pre-guard** verdict — it
counts the events the guard exists to repair. `post_guard_violations()` recomputes I1-b's
actual statistic from the per-event depths in the artifact (commit `d8c9fa7`); both counts
are reported everywhere. Bit-identity when the guard does not fire is asserted **in-process
and under `filter_jit`**: loss and every gradient leaf agree to 0.0e+00 between
`p1_on` and `p1_on_i1_on` at step 0. ⚠ Two *separate processes* running the two configs
drift at ~1e-4 in nll by step 50 — that is CPU/XLA cross-process float noise (the
same-process comparison is bitwise), and it is the paired noise floor, ~100× below E3's band.

### 5.3 ⭐ The interference channel (ADDENDUM-1 §A1-4) — a hard C3-locality receipt

Over **1530–2060 (w4) / 147–630 (w40) / 1708–1943 (resoff) events per seed**, splitting the
fitted depth at a live item's own site into its **own** rows and every **foreign** row:

- ✅✅ **own leg: 0 violations of the decay law, on every cell and every seed.** The residual
  against the designed prediction `D_after = D_before · group_scale²` is **≤ 3.3e-07**
  (float32 ULP). The masked write is C3-local **exactly**; every bit of an untouched item's
  own-well change is the designed leak. *This is now a live regression check, not a result.*
- **foreign leg (the real #9/#12 channel):** foreign-atom depth at a live item's site RISES on
  **0.383 ± 0.086** of events at w4 (0.541 / 0.243 / 0.365) and **0.571 ± 0.082** at w40
  (0.731 / 0.461 / 0.520). ⛔ My registered A1-4 clause ("rate ≥ 0.5 on ≥2/3 seeds") is
  **REFUTED at w4 (1/3)** and **CONFIRMED at w40 (3/3)**; the second clause (median relative
  change > 0) holds 2/3 at w4 and 3/3 at w40. **Neighbours do crowd in, and they crowd in
  more when the write budget is at N94's floor.**

---

## 6. ⭐⭐ I2 — THE HEAD'S HYPOTHESIS IS NOT CONFIRMED; THE QUOTATION CAVEAT STAYS

Registered: on the partition-OFF arm, Spearman **ρ(usefulness, erosion rate) ≥ +0.5** (the
most-useful wells erode fastest); ≤ −0.3 is the registered refutation branch; |ρ| < 0.3 is
"no usage structure" and the caveat stays.

| arm | ρ(read-selection) per seed | mean ± SE | pooled (n wells) | ρ(LOO Δbpc) mean ± SE |
|---|---|---|---|---|
| `p1_off` (w4) | +0.029 / −0.486 / −0.314 | **−0.257 ± 0.151** | **−0.323** (18) | +0.067 ± 0.163 |
| `w40_p1_off` | +0.872 / +0.143 / −0.771 | +0.081 ± 0.475 | +0.205 (17) | +0.219 ± 0.229 |
| `resoff_p1_off` ⛔DIAG | −0.257 / +0.200 / −0.086 | −0.048 ± 0.135 | −0.018 (18) | +0.200 ± 0.271 |

- ⛔ **Registered branch = `NO_USAGE_STRUCTURE`** on the primary arm and statistic
  (|−0.257| < 0.3). The +0.5 hypothesis is **not** confirmed anywhere; the two proxies
  **disagree in sign** on the primary arm (read-selection −0.257, loss-contribution +0.067),
  and the w40 arm's per-seed ρ spans +0.872 to −0.771. **⛔ Depth therefore remains
  non-quotable as feature importance (charter §A21 caveat ACTIVE).**
- The third column — `ρ(‖∂L/∂atoms‖, erosion)` — is +0.220 (w4) / +0.108 (w40) / +0.176
  (resoff) pooled: the gradient magnitude into a well is **weakly** predictive of its erosion,
  which is the mechanism the hypothesis assumed, but far too weak to carry it.
- ⭐ **The partition's in-flight receipt:** on every P1-ON arm this column is `nan` because
  `‖∂L/∂(atom rows)‖ ≡ 0` for **every well at every one of the 41 readings** — K1 holding
  continuously through training, not just at init.
- ⚠ **Provisional, as instructed.** The analyst adjudicates ρ on the raw per-well series
  (`erosion_*_records.json → telemetry[*].wells`, 18–21 wells × 41 readings × 7 cells ×
  3 seeds, with per-reading site addresses so allocation drift is checkable). Slot index is
  the proxy for well identity; `n_wells = 5–6` per seed is small and is why the per-seed ρ is
  so unstable.

---

## 7. THE PREREG SCORECARD (registered → measured → verdict)

| # | registered | measured | verdict |
|---|---|---|---|
| **E1** | OFF depth(1000) ≤ 0.3× depth(200), ≥2/3 seeds (band ≤0.5) | 9.78 / 0.903 / 0.530 | ⛔⛔ **REFUTED 0/3** — no erosion to protect against at the run-2 config |
| **E2** | ON ≥ 0.7× (band [0.5, 1.05]), 3/3 | 7.07 / 0.539 / 2.94 → ≥0.5 on 3/3, **inside [0.5,1.05] on 1/3, above it on 2/3** | ◐ **met on the gate's clause, ABOVE the registered band** — the curve does not flatten, it re-deepens |
| **E3** | paired &#124;Δbpc&#124; ≤ 0.01 | **−0.00485 ± 0.00078** (3/3 seeds better) | ✅ **CONFIRMED**, and on the favourable side |
| **I1-a** | 10–40 % of rewrite events (band 2–60 %) | **0.228 ± 0.182** (0.593/0.050/0.043) | ✅ **CONFIRMED** (point band, on the mean) |
| **I1-b** | depth-reduction events = exactly 0 | **0 / 0 / 0** post-guard, 165 events, guard fired 6× | ✅ **CONFIRMED** |
| **I2** | ρ ≥ +0.5 (refute ≤ −0.3) | **−0.257 ± 0.151** / +0.067 ± 0.163 | ⛔ **NOT CONFIRMED — `NO_USAGE_STRUCTURE`**; caveat stays |
| **P-residual interaction** | ON final depth ≥ 0.1321 on ≥2/3 seeds; ON collapsing BELOW OFF disproves P1 | ON final 0.0068 / 0.1455 / 0.1903 → **2/3 ≥ banked**; ON below OFF on **1/3** seeds (no collapse) | ✅ **met; P1 is NOT disproved** — at w40 ON is deeper on 3/3 (geo-mean 3.63×) |
| **A1-3** (mine) | rewrite events = 0, structurally | 27 / 40 / 70 events | ⛔ **REFUTED — the finding of §5.1** |
| **A1-4** (mine) | own-leg violations 0, residual ≤1e-5 | **0**, residual ≤ **3.3e-07** | ✅✅ **CONFIRMED exactly** |
| **A1-4** (mine) | foreign-up rate ≥ 0.5 on ≥2/3 seeds | w4 **1/3** (0.541/0.243/0.365); w40 **3/3** (0.731/0.461/0.520 — 2/3 strictly ≥0.5) | ◐ **refuted at w4, confirmed at w40** |

**Score: 4 confirmed (1 exactly), 2 partial, 3 refuted** — two of the refutations are mine and
one (E1) retires the premise the wave was built on.

---

## 8. ⭐ THE CSF3 RUN-3 FLAG BLOCK (emitted with the verdict attached, per task §5)

```bash
# scripts/csf3/job_gpu_cluformer.sh — RUN 3 candidate
# ⛔ VERDICT AT EMISSION: FAILS_FLATTEN (w4 AND w40). Submitted only on the Advisor's decision.
sbatch --export=ALL,SEEDS="0 1 2",STAGE=pilot,STG=s4,\
MEM="atom_place_radius=0.3 write_inner_steps=40 psi_payload_residual=True psi_residual_source=q_star erosion_partition=True refresh_monotonic=True",\
STORE="write_margin=0.6",\
SET="monitor_every=25 plan_workers=8" \
       -c 12 --mail-user=$CLU_MAIL -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh
```

The diff against run 2 is **exactly two `MEM` entries** (`erosion_partition=True`
`refresh_monotonic=True`); zero module edits on the cluster, zero new env passthrough, and
the byte ledger and matched GRU/TTT swap geometry are **unchanged** (both flags add no leaf).
⚠ **What the Advisor is deciding**, stated plainly: the gate says NO because the *baseline*
did not erode; the component's own legs are (a) K1 exact, (b) no bpc harm — a paired
*improvement* at w4, (c) 3.63× geometric-mean deeper final wells at N94's undemoted budget,
(d) K4 clean. If `refresh_monotonic` is dropped, remove that one token: at w40 it has nothing
to do (0 violations in 64 events).

---

## 9. HOW I VERIFIED (commands + observed output)

```
# kill-conditions first, before any science cell
PYTHONPATH=. .venv/bin/python -m pytest tests/test_anti_erosion.py -q      -> 18 passed (63 s)
# K1 in-system on the full block            -> k1_in_system.json (§2 table; exact_zero true)
# K2 reference captured at main @ 104ca19   -> k2_reference_seed0.json, inlined into the test
# 21 cell-seeds, ~6.9 h wall (3-4 processes in parallel on 8 cores):
#   p1_off/p1_on/p1_on_i1_on  3 seeds x 1000 steps   (1348-1706 s each)
#   w40_p1_off/w40_p1_on      3 seeds x 400  steps   (1906-2986 s each)
#   resoff_p1_off/_on ⛔DIAG   3 seeds x 1000 steps   (1459-1877 s each)
PYTHONPATH=. .venv/bin/python -m chlu.experiments.exp_anti_erosion --cells <cell> \
    --seeds 0 1 2 --steps <1000|400> --monitor-every 25 --eval-batches 4 \
    --loo-batches 2 --loo-checkpoints 4 [--no-none-arm] [--no-store-watch]
PYTHONPATH=. .venv/bin/python -m chlu.experiments.exp_anti_erosion --aggregate-only --out .
.venv/bin/python -m ruff check chlu/ tests/                                -> All checks passed!
```

⚠ `ruff format --check` reports "would reformat" on my files **and on unmodified files on
`main`** (e.g. `train_cluformer.py` before my edit) — `ruff format` is not this repo's
convention; `ruff check` is the gate and it is green.

### 9.1 Regression run (at branch HEAD `d8c9fa7`)

```
PYTHONPATH=. .venv/bin/python -m pytest tests/test_anti_erosion.py tests/test_blocks.py \
    tests/test_cluformer_pilot.py tests/test_placement_probe.py tests/test_psi_residual.py \
    tests/test_pilot_checkpoint_resume.py -q -p no:randomly --no-cov
-> 109 passed in 767.72s (0:12:47)
```

**0 failures, 0 skips** across the six affected suites (18 of the 109 are this wave's).
The **full-suite run before merge is the Hub's gate, not my claim.**

### 9.2 ⚠ One artifact-hygiene bug I caught and fixed (stated, not buried)

`--aggregate-only` globs `erosion_*_records.json`, which **also matched the three throwaway
smoke runs** (26 and 50 steps, seed 0) I used to validate the instrument; the first
`erosion_aggregate.json` it wrote therefore carried **24** records, not 21, and the figure's
legend showed the duplicates. Caught by reading the figure. The smoke files are moved to
`smoke/` and everything was regenerated: **`n_records: 21`**, and the verdict, E1's per-seed
ratios (9.782 / 0.903 / 0.530) and K3 (−0.004853) are **unchanged**. `TABLES.md` was never
affected (`make_tables.py` filtered `smoke` from the start), so every number in §3–§7 was
computed on the clean 21.

---

## 10. GIT FOOTPRINT

**Branch `c2w6-anti-erosion`** (the wave branch, per the task file — not the `agent/…`
default), worktree `../CHLU-c2w6`, base local `main @ 104ca19`. **Not pushed** (neither
`origin` nor `clu-dev`), worktree left in place for Hub review.

| commit | what |
|---|---|
| `ffe7440` | P1 partition (3 legs) + I1 guard + `fitted_well_depth` + `write_and_diag`/`write_diag` + `train_arm`'s `probe`/`probe_out` hook |
| `ae47a66` | `exp_anti_erosion.py` (harness, cells, I2 telemetry, mechanical gate) + `tests/test_anti_erosion.py` (K1/K2/K5) |
| `251c2c1` | the interference audit corrected: designed decay vs foreign-atom crowding; `occupied`/`evicting` on the diag; the ADDENDUM-1 collapse floor |
| `cc5e49b` | `--no-store-watch` declared cut, `plot_curves`, `store_health` carried into the record |
| `d8c9fa7` | I1-b is the POST-guard count (`post_guard_violations`); E2's band reported two-sided |
| `b623d56` | the erosion figure faceted by seed (the arms are paired *within* a seed) |

**Files touched (all declared MINE in the task file):** `chlu/core/blocks.py` (additive:
4 config fields, `fitted_well_depth`, `init_state`/`write` legs, the guard, the diag,
`write_and_diag`), `chlu/training/train_cluformer.py` (**only** `train_arm`'s signature +
the two `_probe` call sites + one return key — default `probe=None` is the shipped loop
unchanged), `chlu/experiments/exp_anti_erosion.py` (**new**),
`tests/test_anti_erosion.py` (**new**). ⛔ `scripts/csf3/job_gpu_cluformer.sh` **not**
touched — no new env passthrough was needed. **No conflicts**: C2W7 (wt2) touches none of
these; `git status` in the worktree was clean of foreign changes throughout.

---

## 11. OPEN QUESTIONS / FOLLOW-UPS / RISKS

1. **⭐ The premise is retired, and the charter row should be re-scoped.** "The outer loop
   destroys the store" is a **shipped-config** statement (no placement, `write_margin=0.15`,
   pre-ψ-fix). At the config CSF3 is actually submitting, 1000 toy steps produce no net
   erosion. The remaining erosion lives at **w40, seed 0** and in the **residual-off**
   corner. C2W6's gate was written against a baseline that no longer exists.
2. **⭐ The write→φ placement leak (§2) is a standalone finding with no owner.** `write_sign`'s
   docstring's "sign-SGD severs `d(store state)/d(phi)`" is **false whenever
   `atom_place_radius > 0`**, which is the run-1/2/3 config. 27 % of φ's layer-0 gradient
   flows that way. It should be a §7 entry and probably an N-registry number; it also means
   every "the write cannot train φ" statement needs the placement qualifier.
3. **I1's home is not here.** At w40 the channel is silent (0/64); at w4 it fires at 22.8 %
   and the guard cleans it (0/165). Its designed use-case — capacity pressure, repeated
   writes of the same item, cross-stream persistence — is **C2W8/C2W10**. Shipping it OFF in
   run 3 costs nothing and buys the invariant when those waves land.
4. **The seed variance is the real methodological risk.** Untrained depth varies 5.4× across
   3 seeds and the SEs on every ratio are comparable to the means. 3 seeds cannot resolve a
   2× depth effect at this rig; the w40 pair's 3/3 sign consistency is the only depth
   statement I would defend, and only as a sign.
5. **A monitor row for I1?** The task forbade me `monitors.py` (C2W7 owns it). If the
   registry should gain a "rewrite reduced a well's depth" row, the check exists as
   `CluStoreCell.write_diag`'s `violation`/`depth_*` fields and needs ~5 lines to wire.
6. **`n_wells = 5–6`** per seed makes every ρ a 6-point rank correlation. If I2 is to be
   adjudicated properly, the rig needs either more live items (capacity/budget up, or longer
   sequences) or pooling across lanes — both change the rig and need registering first.

---

## Proposed handover updates (for the Hub)

- **§7.27 [RE-SCOPE, not resolved]** — "200 outer steps drive the in-block store's well depth
  to ~1e-63" must gain **"at the shipped config (no placement, `write_margin=0.15`,
  pre-ψ-fix)"**. Measured this wave at the **run-2** config over **1000** steps × 3 seeds:
  final depth **0.708 ± 0.57 ×** untrained (no net erosion). The collapse survives only in
  the **residual-off** corner (0.141 ± 0.095 ×) and at **w40 seed 0** (×0.0029). The
  in-flight watch stays valuable; its trigger threshold should not assume monotone collapse.
- **§7.NEW [OPEN, mechanism]** — **the localized-placement write→φ gradient leak.**
  `StreamMemoryConfig.write_sign`'s docstring claims sign-SGD severs `d(store state)/d(phi)`.
  With `atom_place_radius > 0` (run-1/2/3) the placement assignment
  `centers[:, :addr] = z[:addr] + jig` re-opens it: **27 % of φ's layer-0 gradient** (0.0908
  → 0.0659) flows through the write. `erosion_partition=True` closes it.
- **§3 config defaults** — four new `StreamMemoryConfig` knobs, all shipping OFF and all
  bit-identical off: `erosion_partition=False`, `refresh_monotonic=False`,
  `refresh_max_gain=4.0`, `refresh_amp_ceiling=0.0`. None adds a leaf or a state byte.
- **N-registry candidates** (numbers, not narrative): (a) **erosion is a symptom of
  uselessness** — residual-off retention 0.141 ± 0.095 vs residual-on 0.708 ± 0.57 ×
  untrained, 3 paired seeds, 1000 steps; (b) **the masked write is C3-local to float32** —
  0 violations of `D·group_scale²` in 5553 + 1355 + 5546 events, max residual 3.3e-07;
  (c) **P1's w40 depth protection** — paired ON/OFF final-depth ratio 21.5/1.52/1.46,
  3/3 seeds; (d) **I1-a's rate tracks uselessness** — 0.9 % (P1 on) / 22.8 % (run-2) /
  51.2 % (residual off).
- **Charter §A21 caveat — KEEP ACTIVE.** I2 reports `NO_USAGE_STRUCTURE`
  (ρ = −0.257 ± 0.151 read-selection, +0.067 ± 0.163 loss-contribution, proxies disagreeing
  in sign). **Depth is still not quotable as feature importance.** The Head's hypothesis is
  not confirmed and not refuted — it is unmeasurable at 6 wells/seed.

---

## ⛔ DATED ERRATUM BANNER (Hub, 2026-08-05, after charter ADDENDUM 7 §A23 — body above UNTOUCHED, C-3 precedent)

**Three corrections, all from the Head's rulings; none moves a measured number.**

1. **§8's run-3 flag block ships ONE flag, not two.** Ruling §A23.2: `erosion_partition=True`
   ships; **`refresh_monotonic` stays `False`** (0 post-guard violations at w40; its home is
   C2W8/C2W10). The corrected `MEM` entry — the diff against run 2 is **exactly one token**:
   ```
   MEM="atom_place_radius=0.3 write_inner_steps=40 psi_payload_residual=True psi_residual_source=q_star erosion_partition=True"
   ```
   §8's own fallback sentence ("if `refresh_monotonic` is dropped, remove that one token") is the
   path taken.
2. **The gate verdict is SUPERSEDED, not overturned.** `FAILS_FLATTEN` was computed correctly
   against prereg §4 and stands as the record of the registered rule. §A21's trigger has since been
   **re-scoped by the Head** (the baseline failed it, not P1) and P1 satisfies the replacement rule
   §A23.2(i)–(iv). ⛔ Quote the verdict only with its re-scope attached; the component ships.
3. **Two claim-scope constraints the Advisor attached (§A22), binding on every downstream
   quotation of this report:**
   - the **3/3 depth protection is `w40`-SCOPED** (N94's undemoted floor; w4 is **2/3**, geo 1.421×);
   - the K3 claim is **"no harm"**, never "P1 improves bpc" — the w4 improvement is real
     (−0.004853 ± 0.000780, 6.23 SE) but the **w40 point estimate is slightly WORSE** (+0.005676 ±
     0.005613, n.s.), so the defensible claim is no harm at the 2-SE bar on both budgets.

⭐ **Ship-safety condition (Head): "no silent bugs" — discharged at `main @ d1149a4` by §A23.2(i)
K1's exact-zero in-system probe + (iv) K2's pre-build sha256 fingerprint test
(`k2_reference_seed0.json` @ `104ca19`), both green in the 1397/0 post-merge suite. ⛔ Any future
change that reds either test un-ships `erosion_partition` until it is green again.**

---

## ⛔ DATED ERRATUM BANNER 2 (Hub, 2026-08-06, after `c2w6-erosion-adjudication` + `doc-curator-c2w6-fold` — body above UNTOUCHED)

**The adjudication re-derived every number in this report from raw and found ZERO transcription or
arithmetic error** (21 records × 5 curve scalars, worst relative deviation **0.00e+00**; all 12
cell-seed ρ triples MATCH; gate legs and `aggregate()` identical). **No verdict in this report
changes.** The five corrections below are about **estimators, instrument validity and claim form**.

1. ⭐⭐ **§3/R1's claim form is SUPERSEDED — and it is an UPGRADE, not a retraction.**
   *"0.708 ± 0.57× — it does not decay, it recovers"* is not supportable: the direction is **not
   significant under either estimator** (arith 0.708 ± 0.570 = **0.51 SE** from 1; geo **0.327**,
   log-mean/SE **−1.29**), and the two estimators **disagree in direction on the same three seeds**.
   The replacement, from the curves rather than their endpoints: **a large TRANSIENT TROUGH** — depth
   falls to **0.9 % (step 150) / 5.3 % (step 275)** of untrained on 2/3 seeds (112× and 19× losses,
   vs reading-to-reading noise of only 1.05–1.33×), then partially recovers (final/min geo 4.20×).
   ⭐ **The banked N223 anchor was taken at 200 steps — INSIDE this trough — so E1's `final/step-200`
   had its denominator in the trough and was structurally measuring recovery, not decay.**
   ⇒ ⛔ **"the premise is retired" / "there is no erosion at the run-2 config" are NEVER-QUOTE.**
   ✅ **The defensible record: N223's mechanism FIRES at the run-2 config but is NOT TERMINAL.**
   ✅ **What survives intact is the COMPARATIVE claim** (§3's real R1): the residual-off corner erodes
   **significantly** (geo 0.0775×, log-mean/SE **−3.10**) while the run-2 arm does not — erosion
   tracks uselessness, and that comparison is estimator-robust.
2. ⛔ **§6's ρ(LOO) = +0.067 is UNDEFINED, not a null.** The leave-one-well-out proxy's **ICC(1,1) is
   NEGATIVE on 3/3 seeds** (−0.205 / −0.218 / −0.248) ⇒ attenuation ceiling **0.000**: no true
   correlation of any size could have been observed. ⛔ **Never quote it as a measured correlation.**
   (Also: the two registered usefulness proxies are **anti-correlated**, −0.505 ± 0.125, 4.04 SE.)
3. ⛔ **§6's "pooled (n wells)" ρ column is a NON-REGISTERED estimator and it flips sign** on the
   mechanism row (pooled **+0.2198** vs the registered per-seed-then-mean **−0.1619 ± 0.2806**). The
   sentence *"the gradient magnitude … is weakly predictive … the mechanism the hypothesis assumed"*
   **does not survive the registered estimator** — read it as relabelled, not as evidence.
4. ⚠ **§3's curve is RAW — it is not netted of the designed decay, and the decay exponent DRIFTS**
   (`last_write_chunk` moves 0→12 within a slot; 6–10 distinct values per slot per run). The decay
   law itself is exact (predicted per-tick drop 0.039211 vs measured median **0.039211** over 717
   readings). Netting moves E1 at seed 0 from **9.78× → 6.47× (−34 %)** — i.e. up to a third of the
   apparent "recovery" is allocator drift, not depth restoration — and it makes the arm look **more**
   eroded, never less. ⛔ **No C2W6 verdict changes** (E1 0/3 raw *and* netted). Netting is a **build
   requirement for C2W8/C2W10**, not a C2W6 defect.
5. ⚠ **R1's parenthetical states E1's rule as "(rule: ≤0.5×)"**; the registered rule is **≤0.3×**
   with **≤0.5× as the band** (this report's own §3/§7 and the Hub's record agree). REFUTED 0/3 under
   either, so nothing moves — but ⛔ do not quote R1's line as the rule.

⭐ **Also confirmed by the adjudication, in this report's favour:** the **P3 re-price trigger did NOT
fire** on the clean within-wave paired leg (partition-ON below partition-OFF on **1/3** seeds only;
geo 1.421× at w4, **3.628× on 3/3 at w40**) — **the partition did not starve the write of its one
useful gradient.** And §5.2's I1-b and §5.3's own-leg C3-locality receipt both reproduce exactly.
