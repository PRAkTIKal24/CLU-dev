# psi-payload-residual — experiment-engineer report

**Task + acceptance criterion:** build a payload-carrying residual in ψ, deliver the per-stage spread
ledger, and adjudicate §A20.3(a) — **decoded spread reaches `q*` spread (ratio ≥ 0.5 across cells)
with the blank-store decode at chance** — or locate the gap exactly; end with the decoder-fixed
CSF3 **run-2 flag block**. **Status: done** — **5 cells × 3 paired seeds untrained** (including the
`run1_w40` cell at N94's maturity floor, the only reading monitor #13 does not demote) **+ 1 cell ×
3 seeds × 2 paired arms trained.** ⭐ **Bar MET on 5/5 cells; blank-leak check GREEN on 5/5.**

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5, first-10-lines rule)
> **R1 — ⭐ The probe's `acq(live) − acq(blank) = 0.000 exactly on 14/14 cells × 3 seeds` NO LONGER
> HOLDS with the residual on.** At the *same* seeds, lanes, plans and a bit-identically reproduced
> gate-0 column, `acq − blank` = **+0.125 ± 0.033 / +0.141 ± 0.043 / +0.141 ± 0.043 / +0.248 ± 0.048**
> (3.3–5.2 SE, 3 paired seeds, 13–18 items/seed) — **and +0.211 ± 0.051 (4.15 SE) at N94's 40-step
> maturity floor, the one cell monitor #13 does not demote.** ⛔ It is a **toy, UNTRAINED**
> number and it does **not** survive 200 outer steps significantly (§5). *(Owner: Hub — this is the
> first non-zero write-effect measured anywhere in the tier-iii block; every artifact that says "the
> in-block store's acquisition is at chance in every configuration measured" needs the qualifier
> "…with the shipped read-out".)*
> **R2 — ⚠ the task file's "the current 0.04–0.15" is the reciprocal of §6's *7–25× compression*
> band, not a per-cell ratio.** Recomputed consistently off §6's own raw artifacts the per-cell
> before-column is **0.160 / 0.058 / 0.191** (`PREREG ADDENDUM 1`, filed before any cell ran). The
> bar (≥ 0.5) is unaffected. *(Owner: curator — the §A20.3(a) wording if it is ever re-quoted.)*
> **R3 — ⛔ the acceptance ratio is a RATIO and R3's store destruction inflates it.** After 200 outer
> steps the *residual-off* control's ratio reads 0.577 ± 0.26 (per seed 0.661 / 0.084 / 0.985) purely
> because `sd(q*)` collapsed with the wells. **Never quote the ratio without the absolute spreads
> beside it** (§5 does). *(Owner: whoever writes the read-fix section.)*

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)

- **Dial:** **compute-adaptive reads** (the read-out's fidelity) — tier-iii instrument + tier-ii
  design input. ⛔ **No paper number**; this is run-2 config evidence.
- **Laundering control:** live vs blank vs the same-init blank, paired seeds; **blocking check —
  the blank-store decode must stay at chance.** Measured: blank `acq` = **0.260 ± 0.025 = chance,
  exactly, at EVERY gate in {0, 0.5, 1, 1.5, 2, 3, 4, 6} on 5/5 cells × 3 seeds**; the residual's own
  blank contribution `sd(q*_blank)/sd(q*)` = **0.0022–0.0064** against a 0.05 bar; and the residual
  adds **nothing** to the blank decode's spread (0.01995 vs psi-only-blank 0.02010). **LEAK CHECK
  GREEN 5/5.**
- **Falsifies:** the bar not met AND the compression not localized further.
- **Does NOT falsify:** acquisition still at chance with the spread restored (that would isolate the
  assignment rule — §A20.3(b)'s territory); anything tier-iii at scale (monitor #13 / N94 travel).
- ⚠ **Monitor #13 / N94 travels with every reading:** 4 inner write steps against a floor of 40 ⇒
  **every number here is formally NON-PROMOTABLE** except the `run1_w40` cell (§2.4), which is at the
  floor and carries the same verdict.
- ⛔ TOY scale (0.16 M). ⛔ "CLU-former" is a placeholder and appears nowhere: this is *the tier-iii
  block*.

---

## 0. FLAG PROVENANCE (every number in this report)

| item | value |
|---|---|
| branch | `agent/experiment-engineer/psi-payload-residual`, worktree `../CHLU-psires`, base local `main @ 9b2d4db` |
| commits | `bbe3734` · `26c0d1b` · `c319328` — see §10 |
| env | **main venv reused** (`/Users/user/Desktop/CHLU/.venv`), **no worktree `uv sync`** (w6 hazard avoided). **JAX 0.9.0**, equinox 0.13.4, optax 0.2.6, CPU |
| **seeds** | **0, 1, 2 on every cell**, paired (identical shell, φ-gain, data order, plan pass; cells differ ONLY in the lever under test). SE = sample sd (ddof = 1)/√n |
| **scale** | ⛔ **TOY (0.16 M): `d_model 64`, `n_layers 2`, `seq_len 512`, `batch 4`, `chunk 32`** |
| data | **enwik8**, byte-level, canonical 90/5/5 split, first 4 000 000 B; ledger/self-probe on the **valid** split's first contiguous batch (all 4 lanes), held-out bpc on **test** (4 batches) |
| store | `DesignFreedomPotential(rung="free_mlp", family="atoms")`, `dim 3` (`addr_dim 2` + `payload_dim 1`), `capacity 8`, `budget 6`, `n_atoms 1024`, `atom_width 0.3`, `atom_init_scale 1.0`, `atom_depth_init 1e-4`, `confine 0.05`, `ball_radius 1.0`, `query_sigma 0.15`, masked write ON, all stage flags TRUE, `soft_certificate=True`, `leak 0.02`, `retry_tau 0.5`, `retry_max_rounds 1` |
| read | `dt 0.05`, `γ_address 0.05`, `γ_read 0.02` (both trainable), 24 + 24 (+24 gated retry) Verlet steps, `traj_stride 8`, `kinetic_mode newtonian_learned`, read_mode **trajectory**, ψ = `DeepSetsPsi(hidden 32, depth 2)` |
| write | masked/local, **4 inner sign-SGD steps at lr 0.05** (40 in `run1_w40`), `n_perturb 8`, `σ_addr 0.25`, `σ_pay 0.6`, `barrier 0.2`, `barrier_pairs "nn"` |
| **cells** (probe-paired) | `baseline` (none) · `h1b_r0.3` (`atom_place_radius=0.3`) · `h1b_m1.0` (`+ write_margin=1.0`) · **`run1`** (`atom_place_radius=0.3`, `write_margin=0.6` — the CSF3 **run-1** config at w4) · `run1_w40` (`+ write_inner_steps=40`) |
| **the lever under test (NEW, ships OFF)** | `psi_payload_residual` (False) · `psi_residual_source` ("q_star") · `psi_residual_gain` (1.0) · `psi_residual_trainable` (True) |
| ledger tier | residual wired with `source="both"` and the gate used as the dial; gate rows `[[0],[0]]/[[1],[0]]/[[0],[1]]` recover `psi_only`, `q*[payload]`, `traj_mean[payload]` **exactly** (residual is additive+linear; asserted at 1e-5, measured max **2.4e-07**) |
| trained tier | AdamW, warmup-cosine, peak `lr 1e-3`, `warmup 20`, `grad_clip 1.0`, `wd 0.0`, **200 steps**, `source="q_star"`, gate init 1.0 trainable (arm `residual_on`) vs gate pinned 0.0 (arm `residual_off` — bit-identical to the shipped read-out) |
| langevin / temperature | **N/A** — deterministic, `T = 0`, `p₀ = 0`, no Langevin step |
| wake–sleep / Lyapunov | **N/A** — third training path (`train_cluformer.py`), byte-LM cross-entropy |
| artifacts | `.claude/outputs/psi-payload-residual/` — `PREREG.md` (+ ADDENDUM 1) · `psires_ledger_records.json` (12) · `psires_trained_records.json` (3 × 2 arms) · `psires_w40_records.json` · `psires_smoke_records.json` · `{ledger,trained,w40}_run.log` · `TABLES.md` |
| wall clock | ≈ **2.2 h** measured: 12 ledger cell-seeds at 123–181 s · **3 `run1_w40` cell-seeds at 824–836 s** · 6 trained arm-seeds at 176–231 s · 100 tests at 297 s |
| CSF3 | **0 A100-hours.** Not reachable from this machine |

---

## 1. THE BUILD (what shipped, and what it deliberately does not do)

`chlu/core/blocks.py`, additive:

```
decode = psi(traj, ReadState)             # the shipped read-out, untouched
       + gate · source[payload]           # NEW: payload coordinates only
```

- **Payload coordinates only, and that is the leak guard, not a preference.** The read launches on
  the payload-zero manifold (`q0[a:a+m] = 0` by construction), so the residual's source carries
  **zero** information about `phi(x)` at launch. An address-block residual would be exactly N68's
  query bypass and **is not offered by the API**.
- **Ships OFF and off it is bit-identical *and* parameter-count-identical**: the gate leaf is `None`,
  not `zeros`, so `cell_ledger()["params"]`, the byte ledger and the solved GRU/TTT matched-swap
  geometry are unchanged. ⭐ Verified end-to-end, not just by unit test: the gate-0 column of §2
  reproduces `pilot-placement-probe`'s untrained table **digit-for-digit** (§9).
- **Additive and exactly linear in the gate**, which is what makes the whole ledger + the 8-point
  gate sweep + both leak arms come out of **three reads per item** instead of one grid per gate.
  Measured linearity error **≤ 2.4e-07** (float32 ULP); the `q*` source recovered through the
  residual agrees with the independently-coded `read_diag` settle to **≤ 6.0e-08**.
- ⛔ **`AttentionPsi` is untouched.** The quarantine is not routed through, not relaxed, not measured.
- ⛔ **N46 holds:** the write is bit-identical with and without the residual (asserted).

---

## 2. ⭐⭐ THE PER-STAGE SPREAD LEDGER — §A20.3(a) ADJUDICATED

Spread = sample sd (ddof = 1) over a lane's live items, meaned over the 4 lanes, then over 3 seeds
(±SE across seeds). 13–18 items/seed.

### 2.1 The four stages, and the acceptance ratio

| cell | sd(true) | sd(`q*`) | sd(psi only) | **sd(decoded)** | **ratio decoded/`q*`** | ratio psi-only/`q*` | **§A20.3(a)** |
|---|---|---|---|---|---|---|---|
| `baseline` | 0.3227 ± 0.073 | 0.1013 ± 0.033 | 0.0191 ± 0.0067 | **0.0985 ± 0.030** | **1.085 ± 0.097** | 0.360 ± 0.21 | ✅ **MET** |
| `h1b_r0.3` | 0.3227 ± 0.073 | 0.2070 ± 0.048 | 0.0181 ± 0.0052 | **0.2018 ± 0.045** | **0.9997 ± 0.041** | 0.124 ± 0.025 | ✅ **MET** |
| `h1b_m1.0` | 0.3227 ± 0.073 | 0.3015 ± 0.053 | 0.0163 ± 0.0044 | **0.2973 ± 0.048** | **1.005 ± 0.024** | 0.102 ± 0.025 | ✅ **MET** |
| **`run1`** | 0.3227 ± 0.073 | 0.2947 ± 0.044 | 0.0155 ± 0.0040 | **0.2894 ± 0.038** | **0.979 ± 0.022** | 0.067 ± 0.011 | ✅ **MET** |
| **`run1_w40`** ⭐ | 0.3227 ± 0.073 | 0.2729 ± 0.036 | 0.0159 ± 0.0040 | **0.2711 ± 0.033** | **0.9966 ± 0.022** | 0.074 ± 0.012 | ✅ **MET** |

⭐ **The bar (≥ 0.5) is MET on 5/5 cells, adjudicated mechanically by `aggregate()`** (`ACCEPTANCE_MET`
in the artifact), **including the `run1_w40` cell that monitor #13 does not demote.** The read-out's
compression goes from **2.8–15×** (psi only) to **1.00–1.09×**.
⛔ **Declared in the PREREG, before measuring: at gate 1 this bar is close to being met BY
CONSTRUCTION** — the residual *is* `q*`, so `sd(decoded) ≈ sd(q*)` unless the two channels
anti-correlate. Passing it is the precondition, not the finding. The findings are §3 and §4.

### 2.2 The stage that is NOT closed — the payload's **scale**, per cell

| cell | `sd(q*) / sd(true)` | median &#124;`q*`&#124;/&#124;true&#124; | reading |
|---|---|---|---|
| `baseline` | 0.421 ± 0.13 | 0.357 ± 0.11 | the dynamics deliver ~36 % of the payload (probe §6: 30 %) |
| `h1b_r0.3` | 0.777 ± 0.14 | 0.986 ± 0.21 | placement alone puts the delivered payload **on scale** |
| `h1b_m1.0` | 1.047 ± 0.034 | 1.531 ± 0.27 | margin 1.0 **overshoots** (probe §6: 130–175 %) |
| `run1` | 1.341 ± 0.30 | 1.465 ± 0.22 | the run-1 config overshoots too |
| `run1_w40` | 1.159 ± 0.22 | 1.383 ± 0.22 | 10× the write budget changes the scale by ~0.9× — **not the constraint** |

⭐ **This re-prices probe §6.2's headline.** Pooled over 4 lanes and 3 seeds, the "30–50 % of the way"
figure is the **`baseline`** cell's property; with placement the delivered payload is already at
**0.99–1.53× the true scale** (1.38× at the 40-step floor). So after the residual there is **no residual scale gap to close on the
placement cells** — the remaining error is *direction/identity*, not magnitude (§4 shows the gate
sweep is flat above g ≈ 1, which is the same statement).

### 2.3 The `traj_mean` stage (measured, free, not shipped)

sd(`traj_mean[payload]`) = **0.0319 / 0.0746 / 0.1451 / 0.1180** (`run1_w40`: 0.1309) — i.e.
**31–49 % of sd(`q*`)** on
every cell. The strided trajectory's mean payload carries the same signal at ~⅓ the amplitude
(it averages in the payload-zero launch), so it is a **weaker** source and the run-2 block ships
`q_star`. ⛔ Declared NOT-RUN as a shipped configuration; kept as a ledger stage.

### 2.4 ⭐ `run1_w40` — the only monitor-#13-undemoted cell, and it is the STRONGEST

The expectation was written into this section **before any `run1_w40` number had been read**
(*"the ratio is a read-out property and the write budget does not enter it, so w40 should reproduce
`run1`'s 0.98 ± 0.02 to within its own SE"*). It is **CONFIRMED**: **0.9966 ± 0.022** vs `run1`'s 0.979 ± 0.022, per seed
1.040 / 0.976 / 0.974. Everything else moves in the same direction as `run1`, only further:

| quantity | `run1` (w4, demoted by #13) | **`run1_w40` (at N94's floor)** |
|---|---|---|
| ratio decoded/`q*` | 0.979 ± 0.022 | **0.9966 ± 0.022** |
| ratio psi-only/`q*` | 0.067 ± 0.011 | 0.074 ± 0.012 |
| `acq` at gate 1 | 0.508 ± 0.029 | **0.471 ± 0.050** |
| **`acq − blank`** (per seed) | +0.154, +0.278, +0.312 | **+0.154, +0.167, +0.312** |
| **`acq − blank`** mean ± SE | +0.248 ± 0.048 (5.15 SE) | **+0.211 ± 0.051 (4.15 SE)** |
| blank `acq` at every gate | 0.260 = chance | **0.260 = chance** |
| depth | 0.1489 ± 0.044 | 0.1947 ± 0.042 |
| bpc live − blank (g0 → g1) | +5.24e-04 → +1.01e-02 | +5.84e-04 → +9.95e-03 |
| wall clock / seed | 140–167 s | **824–836 s** |

⭐ **This is the one row in this report that N94/monitor #13 does not demote, and it carries the
same verdict as the four demoted ones.** ⛔ It is still TOY scale and still not a paper number.

## 3. ⛔ THE BLANK-STORE LEAK CHECK — GREEN, on three clauses

| clause | bar | `baseline` | `h1b_r0.3` | `h1b_m1.0` | `run1` | `run1_w40` |
|---|---|---|---|---|---|---|
| blank `acq` at chance | `≤ chance` | 0.260 = 0.260 ✅ | ✅ | ✅ | ✅ | ✅ |
| the residual's own blank share `sd(q*_blank)/sd(q*)` | < 0.05 | **0.0064** | **0.0032** | **0.0022** | **0.0022** | **0.0024** |
| the residual adds no blank spread `sd(dec_blank) ≤ 1.05·sd(psi_only_blank)` | — | 0.01995 ≤ 0.0211 ✅ | ✅ | ✅ | ✅ | ✅ |

- ⭐ **Blank acquisition is 0.260 = chance EXACTLY at every gate in {0, 0.5, 1, 1.5, 2, 3, 4, 6}, on
  every cell (5/5), on every seed.** A gate that multiplies the blank store's payload channel by 6 buys
  nothing, because there is nothing there: `sd(q*_blank)` = **6.51e-04 ± 1.4e-04** against a live
  0.10–0.30 (**150–460×**).
- ⚠ **One honest correction to my own instrument, made before the numbers were read into the
  report:** the first-written clause compared the *whole* blank decode spread to the live one and
  fired on every cell. That clause charges the residual for **ψ's own query-driven variation on a blank
  store** (0.0201) — which is **shipped behaviour, present at gate 0** and identical in the
  residual-off arm. The three clauses above are residual-specific; the ψ-side number is reported
  here rather than hidden (commit `c319328`; the records were not re-measured, only re-adjudicated).
- ⛔ **This is not a claim that ψ has no `q0` channel.** It has one (0.0201 of blank decode spread,
  and it *grows* with training, §5). It is a claim that **the residual does not add one**.

---

## 4. ⭐⭐⭐ ACQUISITION COMES OFF CHANCE — the probe's exact-zero is broken

`acq` is the pooled multi-lane self-probe (nearest-**stored**-payload assignment, N110), live and
blank, at the same plan.

| cell | `acq` at gate 0 (= shipped) | `acq` at gate 1 | blank | **`acq − blank`, paired per seed** | mean ± SE | SE-multiple |
|---|---|---|---|---|---|---|
| `baseline` | **0.260** | 0.385 ± 0.030 | 0.260 | +0.077, +0.111, +0.188 | **+0.125 ± 0.033** | **3.83** |
| `h1b_r0.3` | **0.260** | 0.401 ± 0.022 | 0.260 | +0.077, +0.222, +0.125 | **+0.141 ± 0.043** | **3.31** |
| `h1b_m1.0` | **0.260** | 0.401 ± 0.022 | 0.260 | +0.077, +0.222, +0.125 | **+0.141 ± 0.043** | **3.31** |
| **`run1`** | **0.260** | **0.508 ± 0.029** | 0.260 | +0.154, +0.278, +0.312 | **+0.248 ± 0.048** | **5.15** |
| **`run1_w40`** ⭐ | **0.260** | 0.471 ± 0.050 | 0.260 | +0.154, +0.167, +0.312 | **+0.211 ± 0.051** | **4.15** |

- ⭐ **The gate-0 column is 0.260 = chance to the digit on 5/5 cells — the probe's own result,
  reproduced exactly** — and the *same model, same state, same queries* at gate 1 is
  **0.385–0.508**. The lever is the read-out and nothing else.
- ⭐ **`acq(live) − acq(blank)`, which the probe measured at 0.000 exactly on 14/14 cells × 3 seeds,
  is now +0.125…+0.248 at 3.3–5.2 paired SE, and +0.211 ± 0.051 (4.15 SE) at N94's 40-step floor.**
  This is the **first non-zero write-effect measured anywhere in the tier-iii block.**
- ⛔ **What it is not:** promotable (monitor #13), trained (§5), or a performance number. It is a
  label-free self-probe on 13–18 items at `dim = 3`, `capacity = 8`.
- ⚠ The per-seed binomial SE is ~0.13, so **no single seed clears `chance + 2 SE`**; the statistic
  that clears is the **paired** `acq − blank` across 3 seeds, which is the pre-registered
  write-effect statistic (`signal_a_write_effect`'s form).

### 4.1 The gate sweep (arithmetic, verified against the forward probe at g = 0 and g = 1)

live / blank, pooled over lanes and seeds; chance 0.260 everywhere:

| cell | g=0 | g=0.5 | **g=1** | g=1.5 | g=2 | g=3 | g=4 | g=6 |
|---|---|---|---|---|---|---|---|---|
| `baseline` | .260/.260 | .299/.260 | **.385**/.260 | .387/.260 | .360/.260 | .338/.260 | .415/.260 | .390/.260 |
| `h1b_r0.3` | .260/.260 | .348/.260 | **.401**/.260 | .443/.260 | .380/.260 | .364/.260 | .367/.260 | .367/.260 |
| `h1b_m1.0` | .260/.260 | .399/.260 | **.401**/.260 | .401/.260 | .401/.260 | .422/.260 | .385/.260 | .404/.260 |
| `run1` | .260/.260 | .379/.260 | **.508**/.260 | .443/.260 | .422/.260 | .404/.260 | .404/.260 | .385/.260 |
| `run1_w40` | .260/.260 | .395/.260 | **.471**/.260 | .383/.260 | .383/.260 | .346/.260 | .327/.260 | .346/.260 |

⭐ **The curve is a step at g ≈ 0.5–1 and then FLAT.** ⛔ **PREREG P4 is refuted in its mechanism:** I
predicted the win would need `g* ∈ [2, 4]` to close a 2.1–3.3× *scale* gap; measured, **gate 1 is
already at or past the optimum on 4 of 5 cells** and gates 2–6 do not help — on `run1_w40` the curve
**decays** above g = 1 (.471 → .327 at g = 4). §2.2 says why — with
placement the delivered payload is already on scale, so what the residual buys is **spread**, not
gain. ⚠ Non-monotone wiggles (`baseline` at g = 3 vs 4) are 13–18-item quantisation, not structure.
The forward probe run through the real read at g = 0 and g = 1 reproduces the arithmetic values
exactly (e.g. `baseline` s0: forward 0.38462, arithmetic 0.385).

---

## 5. THE TRAINED TIER (`run1`, 200 outer steps, 3 seeds, paired arms)

`residual_off` is the *same model* with the gate pinned to 0 (bit-identical to the shipped
read-out), so the two arms differ by **one leaf**.

| arm | gate init → trained | bpc | live − blank | **ratio dec/`q*`** | sd(`q*`) | `acq` / chance | `acq − blank` | **depth** |
|---|---|---|---|---|---|---|---|---|
| `residual_off` | 0.00 → 0.0000 | 4.6099 ± 0.016 | −1.8e-05 ± 1.5e-05 | 0.577 ± 0.26 | 0.233 ± 0.057 | 0.1691 / 0.1691 | **0.000** | 0.0322 ± 0.017 |
| `residual_on` | 1.00 → **0.9664 ± 0.0135** | 4.6152 ± 0.015 | +5.7e-05 ± 8.3e-04 | **1.048 ± 0.044** | 0.298 ± 0.063 | 0.272 ± 0.11 / 0.182 | +0.106 ± 0.106 | **0.1321 ± 0.078** |

- ✅ **P5 CONFIRMED — the residual survives the outer loop.** The gate barely moves: per seed
  **0.9917 / 0.9454 / 0.9621**, per layer 0.945–0.985 (6 gates, all within 5.5 % of pass-through).
  ⛔ It does **not** train to zero: the outer loss does not *penalise* the payload channel.
- ✅ **The acceptance ratio survives training**: 1.048 ± 0.044 (per seed 0.971 / 1.124 / 1.049).
- ⛔ **The acquisition gain does NOT survive significantly.** `acq − blank` per seed is
  **0.000 / +0.318 / 0.000** — one seed of three. **+0.106 ± 0.106 is 1.0 SE and is a null.** ⚠ This
  is the honest headline of §5 and it is the one the CSF3 run will see.
- ⛔⛔ **R3's well-destruction reproduces, and the residual measurably RESISTS it.** Untrained `run1`
  depth is 0.0727 / 0.1490 / 0.2250; after 200 steps `residual_off` is **0.0011 / 0.0582 / 0.0373**
  and `residual_on` is **0.0021 / 0.2732 / 0.1210** — **higher on 3/3 paired seeds (1.9× / 4.7× /
  3.2×), paired Δ = +0.0999 ± 0.0623**. ⛔ **PREREG P6 predicted no effect and is refuted on
  direction** (the magnitude stayed inside the registered band). *Mechanism, stated as a hypothesis
  and not measured here: with the gate open, the outer loss can reach `V_θ` through the payload
  channel, so the wells acquire a reason to exist that the collapsed read-out never gave them.*
- ⚠ **bpc is slightly WORSE with the residual on 3/3 seeds** (paired Δ **+0.0053 ± 0.0030** bpc).
  At 200 steps × 2048 tokens every arm sits at 4.59–4.65 bpc, barely past unigram statistics, so this
  is a paired-margin observation, not a performance claim — but it is the correct sign to worry
  about, and it is stated, not buried.
- ⚠ **ψ's own blank-store spread GROWS with training** (`sd(psi_only_blank)` 0.020 untrained →
  0.022 / 0.161 / 0.066 trained). Blank acquisition still does not clear chance
  (`acq_decoded_blank` 0.142 / 0.183 / 0.175 vs chance 0.190 / 0.182 / 0.174), but this is the
  channel the N68 launder exists for and it should be watched at scale, **not** attributed to the
  residual (the residual-off arm shows the same growth).

---

## 6. PREREG SCORECARD (registered → measured → verdict)

`PREREG.md` + ADDENDUM 1, both filed before the harness ran.

| # | registered | measured | verdict |
|---|---|---|---|
| **P1** | ratio 0.95–1.10, band [0.85, 1.20]; bar met 3/3 | **0.979–1.085**, met **5/5** (incl. `run1_w40` 0.9966 ± 0.022) | ✅✅ **CONFIRMED** (and declared in advance as near-by-construction) |
| **P2** | blank spread ≤ 0.006 of the residual's channel; blank acq exactly at chance; `acq−blank` = 0 on the blank arm | blank `sd(q*)` **6.5e-04**, blank acq **0.260 = chance at all 8 gates × 5 cells × 3 seeds** | ✅✅ **CONFIRMED exactly** |
| **P3** | `acq(g=1)` **still exactly at chance** (the uniform-under-shoot argument) | **0.385–0.508 vs chance 0.260**, `acq−blank` +0.125…+0.248 at 3.3–5.2 SE (w40: +0.211 ± 0.051) | ⛔⛔ **REFUTED — and this is the report's central finding.** My derivation used §6's three lane-0 items at one seed, where the decode does collapse onto the smallest-magnitude payload; pooled over 4 lanes × 3 seeds it does not, and §2.2 shows why (with placement the delivered scale is already ~1.0, not 0.3) |
| **P4** | `g* ∈ [2, 4]`, best acq 0.35–0.60 | best acq **0.385–0.508**, but at **g ≈ 1–1.5**; the sweep is flat above and *decays* at w40 | ◐ **PARTIAL: the magnitude is inside the band, the mechanism is refuted.** There is no scale gap to close on the placement cells |
| **P5** | gate stays in [0.3, 3.0] of 1.0; ratio ≥ 0.5 after training | gate **0.966 ± 0.014**; ratio **1.048 ± 0.044** | ✅ **CONFIRMED** |
| **P6** | depth 0.02–0.15, residual has **no** effect on R3 | 0.1321 ± 0.078 (in band) but **higher than the paired control on 3/3 seeds**, Δ = +0.0999 ± 0.0623 | ◐ **REFUTED on direction, confirmed on band** — a finding: a read-out lever measurably resists R3 |
| **P7** | &#124;bpc live−blank&#124; ≤ 1e-3 at g = 1; gap 1.5–10× the probe's | **+2.3e-03…+1.0e-02**, i.e. **14–84×** | ⛔ **REFUTED on both clauses** (untrained bpc, so no performance content; the store's content reaches the output far more strongly than predicted) |

**Score: 3 confirmed (2 exactly) · 2 partial · 2 refuted.** An eighth prediction was written into
§2.4 **before any `run1_w40` number had been read** (*"w40 reproduces `run1`'s ratio within its own
SE"*) and is **CONFIRMED** (0.9966 ± 0.022 vs 0.979 ± 0.022). ⭐ The refutation that matters is **P3**:
I pre-registered that restoring the spread would be *necessary and not sufficient*, and the
measurement says it is sufficient to move acquisition off chance at toy scale, untrained.

---

## 7. ⭐⭐ THE RUN-2 FLAG BLOCK (the deliverable that gates CSF3 run 2)

**§A20.4 discipline: run-1's exact config, plus ONLY the ψ flags.** Every knob is a flag through the
`--set/--mem/--store` path added by `pilot-placement-probe` (`85a557d`); **zero module edits on the
cluster.** The diff against run 1 is exactly two `MEM` entries.

```bash
# scripts/csf3/job_gpu_cluformer.sh — RUN 2 (decoder-fixed), the pre-registered designed ablation
sbatch --export=ALL,SEEDS="0 1 2",STAGE=pilot,STG=s4,\
MEM="atom_place_radius=0.3 write_inner_steps=40 psi_payload_residual=True psi_residual_source=q_star",\
STORE="write_margin=0.6",\
SET="monitor_every=25 plan_workers=8" \
       -c 12 --mail-user=$CLU_MAIL -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh
```

| # | knob | run 1 | **run 2** | evidence |
|---|---|---|---|---|
| 1 | `atom_place_radius` | 0.3 | **0.3 (unchanged)** | probe §10 row 2 |
| 2 | `write_inner_steps` | 40 | **40 (unchanged)** | probe §10 row 3 (N94's floor) |
| 3 | `write_margin` | 0.6 | **0.6 (unchanged)** | probe §10 row 4 |
| 4 | `monitor_every` | 25 | **25 (unchanged)** | probe §10 row 7 |
| 5 | `plan_workers` / `-c` | 8 / 12 | **8 / 12 (unchanged)** | `lane-parallel-controller` (4.93× measured) |
| 6 | ⭐ **`psi_payload_residual`** | *(absent = False)* | **True** | §2: ratio 0.067–0.36 → **0.98–1.09** on 5/5 cells; §4: `acq − blank` 0.000 → **+0.125…+0.248**. ⭐ **At run 1's own `write_inner_steps=40`** (the `run1_w40` cell, the only #13-undemoted reading): ratio **0.9966 ± 0.022**, `acq − blank` **+0.211 ± 0.051** |
| 7 | ⭐ **`psi_residual_source`** | — | **`q_star`** | §2.3: `traj_mean` carries only 31–49 % of `q*`'s spread |
| 8 | `psi_residual_gain` | — | **1.0 (default, not passed)** | §4.1: the sweep is flat above g ≈ 1; §5: training leaves the gate at 0.966 |
| 9 | `psi_residual_trainable` | — | **True (default, not passed)** | §5: the gate does not run away and does not collapse to 0 |
| 10 | ⚠ **what to watch** | — | the **paired `acq − blank`** and the **untrained-vs-trained depth**, per seed, both already in the §7.27 in-flight watch | §5: the untrained gain is a **null after 200 toy steps**; if it is still a null at 4000 steps the read-out is fixed and the **assignment rule / the write's payload direction** is the next bottleneck (§A20.3(b)) |

⚠ **Two ledger consequences the submission must carry.** (i) The residual adds **`payload_dim`
parameters per layer** (1 per layer at the pilot's `m`); it is on `cell_ledger()["params"]`
automatically, so the matched GRU/TTT arms re-solve around it — the swap stays matched, but the
matched hidden sizes are **not** bit-identical to run 1's and must be re-quoted from the artifact,
never copied. (ii) The **STATE** column is unchanged (a read-out gate is parameters, not state).

---

## 8. ⛔ DECLARED CUTS AND NOT-RUNs (never reported as nulls)

- **NOT RUN — the trained tier on `baseline` / `h1b_r0.3` / `h1b_m1.0`.** The trained tier runs the
  **run-1 config**, per the task's default; the other three cells are untrained-only.
- **NOT RUN — `traj_mean` or `both` as a *shipped* configuration** (measured as a ledger stage, §2.3).
- **NOT RUN — the 5-arm swap table (GRU / TTT / echo / memory-deleted) per cell.** The residual is a
  read-out lever *inside* the CLU arm; the swap protocol is the pilot's and is unchanged by it.
- **NOT RUN — `AttentionPsi` in any form.** QUARANTINED; not routed through, not relaxed.
- **NOT RUN — anything at 26–47 M / on CSF3**, WikiText-103, deeper stacks, larger `capacity`, or any
  plot (the deliverable is a ledger table).
- **NOT RUN — a CLI hook in `chlu/cli/experiment_cmd.py`.** Deliberate: the file is outside the
  task's ownership list and the tier-iii precedent (`exp_placement_probe`) runs as
  `python -m chlu.experiments.exp_psi_residual`. Flagged for the Hub as a one-line follow-up.
- **UNDER-POWERED, declared:** 3 seeds; 13–18 probed items/seed; `dim = 3`, `capacity = 8`, 4 lanes,
  2 layers. The per-seed binomial SE on `acq` is ~0.13 — **only the paired `acq − blank` statistic is
  defensible**, and §5's trained `acq` is explicitly a null.
- **NOT RUN — the trained tier at w40.** 200 outer steps × 40 inner write steps was outside the
  budget (the untrained w40 cell alone cost 42 min); the trained tier runs at w4 and says so.

---

## 9. HOW I VERIFIED (commands + observed output)

| check | command | observed |
|---|---|---|
| the ledger grid | `PYTHONPATH=. python -u -m chlu.experiments.exp_psi_residual --tier ledger --cells baseline h1b_r0.3 h1b_m1.0 run1 --seeds 0 1 2 --tag ledger` | 12 records, 123–181 s each; `ledger_run.log` |
| the trained tier | `… --tier trained --cells run1 --seeds 0 1 2 --steps 200 --tag trained` | 3 records × 2 arms, 176–231 s per arm; `trained_run.log` |
| the w40 cell | `… --tier ledger --cells run1_w40 --seeds 0 1 2 --tag w40` | 3 records, 824–836 s each; `w40_run.log` |
| ⭐ **the residual-off path is the SHIPPED path — measured end-to-end, not asserted** | gate-0 columns vs `pilot-placement-probe` §2 | **depth** 0.02882 / 0.08545 / 0.1668 / 0.1489 vs the probe's **0.02882 / 0.08545 / 0.1668 / 0.1489**; **bpc live−blank** +2.78e-05 / +1.74e-04 / +7.12e-04 / +5.24e-04 vs the probe's **+2.80e-05 / +1.74e-04 / +7.12e-04 / +5.24e-04**; **acq** 0.260 vs 0.260 — **digit-for-digit** |
| ⭐ **the ledger reproduces §6's raw artifacts through an independent path** | lane 0, seed 0, `baseline` | `q*[payload]` **−0.23329 / −0.18052 / −0.21142** (`qstar_payload.json`: −0.233294 / −0.180517 / −0.211423); psi-only decode **−0.06862 / −0.06392 / −0.06015** (`decode_dispersion.json`: −0.068617 / −0.063920 / −0.060149); blank `q*` **−0.00092 / +0.00018 / −0.00017** (−0.000924 / +0.000185 / −0.000168) |
| instrument integrity (every cell/seed, blocking) | in-run assertion | linearity ≤ **2.4e-07**, `q*`-source vs `read_diag` ≤ **6.0e-08**, tol 1e-5 |
| ⭐ **the §7 run-2 `MEM` string actually parses and reaches the cell** (a flag that silently no-ops on the cluster is the provenance hole the flag path exists to avoid) | `_parse_kv` on the literal §7 `MEM` entries → `memory_cfg()` → `CluStoreCell` | `{'atom_place_radius': 0.3, 'write_inner_steps': 40, 'psi_payload_residual': True, 'psi_residual_source': 'q_star'}` (bool and string inferred correctly) → `mcfg` True/`'q_star'`/gain 1.0/w40/0.3 → **gate `[[1.]]`, params +1 vs the residual-off cell** |
| **new tests** | `pytest tests/test_psi_residual.py -q` | **15 passed in 32 s** |
| **regression (the file I touched that others import)** | `pytest tests/test_psi_residual.py tests/test_placement_probe.py tests/test_blocks.py tests/test_cluformer_pilot.py -q` | see §9.1 |
| lint | `ruff check chlu/ tests/` | **All checks passed** at every commit |

⚠ **Full-suite status, stated honestly:** I ran the affected modules, not the whole suite (~25 min,
and the machine was held by the grids). `chlu/core/blocks.py` is the only file I touched that is
imported outside my own tests. **A full-suite run before merge is owed and is the Hub's gate.**

### 9.1 Regression run (at branch HEAD `c319328`)
```
pytest tests/test_psi_residual.py tests/test_placement_probe.py tests/test_blocks.py \
       tests/test_cluformer_pilot.py tests/test_psi_readout.py -q
→ 100 passed in 297.22s (0:04:57), 0 failed
```
(15 mine + 22 placement-probe + 22 blocks + 16 pilot + 25 psi_readout. ⭐ `test_psi_readout.py`
is included on purpose: it is the quarantine's own regression suite and it passes untouched.)

---

## 10. GIT FOOTPRINT

**Branch** `agent/experiment-engineer/psi-payload-residual`, **worktree `../CHLU-psires`**, base
local `main @ 9b2d4db` (did not move under me; `git rebase main` is a **no-op**). ⛔ **Not pushed,
not merged.** ⛔ `origin` never touched. ⛔ `clu`/`clu-dev` never pushed.

| commit | subject |
|---|---|
| `bbe3734` | blocks: the psi PAYLOAD RESIDUAL (charter §A20.3(a)) |
| `26c0d1b` | the psi-residual instrument: the per-stage spread ledger + 15 tests |
| `c319328` | make the blank-leak criterion RESIDUAL-SPECIFIC |

**Files touched — my declared list, nothing else:**
`chlu/core/blocks.py` (**additive**: 4 config fields + `PSI_RESIDUAL_SOURCES`/`psi_residual_sources`
+ 1 cell field + 1 init hunk + 1 read hunk + `CluStoreCell.payload_residual`) ·
`chlu/experiments/exp_psi_residual.py` (**new**) · `tests/test_psi_residual.py` (**new, 15 tests**).

⛔ **NOT touched:** `chlu/core/psi_readout.py` (**the AttentionPsi quarantine is untouched**) ·
`chlu/training/train_cluformer.py` (incl. all lane-parallel code) · `chlu/config.py` ·
`chlu/cli/experiment_cmd.py` · the factored-store / null-arms files · `chlu/core/monitors.py` ·
`chlu/experiments/exp_placement_probe.py` (imported **read-only**). **No collision with any
concurrent agent's declared files.**

⚠ **Worktree deliberately NOT removed** — left for Hub review; remove with
`git worktree remove ../CHLU-psires` afterwards. Branch ref verified from the MAIN repo. ⚠ The runner
writes artifacts relative to its cwd, so the JSONs were produced under the worktree's gitignored
`.claude/` and **copied** to the main repo's `.claude/outputs/psi-payload-residual/`; both copies are
byte-identical.

### 10.1 The rider (Head item 3) — DONE
`.claude/outputs/orgdiv-cat-test/FROZEN-interfaces.md`, corrected **in place**, with the Hub's dated
erratum banner left beneath as the record: the matched-capacity ledger's store-parameters row
(`a = 32`, `1024·14` floats = **57 344 B**) and byte-ratio row (**9.67×**), and the reader-parameter
counts **R2 `well_table` 16 → 72** and **R4 `mlp_small` 88 → 92**. Each correction carries an
inline *(corrected in place 2026-08-01)* marker. ⛔ Nothing else in that file was touched.

---

## 11. OPEN QUESTIONS / FOLLOW-UPS / RISKS

1. ⭐⭐ **The next bottleneck is now measurable and it is NOT the spread.** With the read-out fixed,
   the remaining error is the **direction/identity** of the delivered payload: `sd(q*)/sd(true)` is
   0.42–1.34 and the gate sweep is flat, yet acquisition is 0.39–0.51 against a chance of 0.26 — i.e.
   the store delivers *a* payload of roughly the right size that is only partly the *right* one.
   That is §A20.3(b)'s territory (the assignment rule) **and** the write's payload objective.
2. ⭐ **The untrained gain does not survive 200 outer steps** (`acq − blank` +0.106 ± 0.106, 1/3
   seeds). Before anything is claimed at scale, someone should measure whether the gain returns with
   a store that is *not* being destroyed by its own outer loop (R3) — the two are confounded here,
   and §5's depth result says the residual is already pushing against R3.
3. ⚠ **The residual is not purely read-side once training runs.** It changes the gradient path into
   φ and `V_θ`, and the paired depth difference (1.9–4.7× on 3/3 seeds) is the evidence. Any
   statement of the form "a read-out lever cannot affect the write" is now false in this harness.
4. ⚠ **ψ's own blank-store spread grows with training** (0.020 → 0.022–0.161). Blank acquisition
   stayed at/below chance here, but the trajectory launder (`chlu.eval.dividend.trajectory_launder`)
   should be run on any scale reading with a trained ψ.
5. **Risk on the record:** everything is `dim = 3`, `capacity = 8`, 2 layers, 0.16 M params, 4 lanes,
   3 seeds, 4 inner write steps. ⛔ **Nothing here transfers to a 26–47 M claim**, and §7 is a
   best-informed config, not a validated one.
6. **Cheap follow-up (≈ 20 min):** the `psi_residual_trainable=False` arm at the trained tier — the
   designed-mechanism control that separates "the gate is useful" from "the gate learned something".
   Wired and tested; not run (budget went to the w40 cell).

---

## Proposed handover updates (for the Hub)

**§7 Known Issues — ADD / AMEND:**
- ⭐ **The in-block store's acquisition is NO LONGER at chance in every configuration** (probe §2's
  headline). With `psi_payload_residual=True`, untrained toy `acq − blank` = **+0.125…+0.248 at
  3.3–5.2 paired SE** on 5/5 cells (**+0.211 ± 0.051 at N94's 40-step floor**); at gate 0 the
  same models reproduce the probe's exact chance.
  Any artifact carrying the old absolute needs the qualifier *"with the shipped read-out"*.
- ⚠ **The acceptance ratio `sd(decoded)/sd(q*)` is inflated by R3's store destruction** (a collapsed
  `q*` shrinks the denominator): the trained residual-OFF control reads 0.577 ± 0.26. Never quote the
  ratio without the absolute spreads.
- ⚠ **A read-out lever measurably affects the WRITE once outer training runs:** `residual_on` keeps
  1.9–4.7× more well depth than the paired `residual_off` control after 200 steps (3/3 seeds).
- ⚠ **ψ's own blank-store output spread grows with training** (0.020 → 0.022–0.161 at 200 steps),
  independent of the residual. This is the N68 channel; run the trajectory launder on trained-ψ
  readings.
- ⚠ **`pilot-placement-probe` §6.2's "30–50 % of the payload" is the `baseline` cell's property.**
  Pooled over 4 lanes × 3 seeds, median |`q*`|/|true| is **0.36 (baseline) / 0.99 (placement) / 1.53
  (placement × margin 1.0) / 1.47 (run-1 config)** — with placement the delivered payload is on scale
  or overshooting.

**§3 config defaults:** **none changed.** Four new `StreamMemoryConfig` knobs ship OFF
(`psi_payload_residual=False`, `psi_residual_source="q_star"`, `psi_residual_gain=1.0`,
`psi_residual_trainable=True`); off, the cell is bit-identical **and parameter-count-identical**
(the gate leaf is `None`), verified end-to-end against the probe's published columns.
`chlu/config.py` untouched. **New CLI: none** (deliberate, §8) — reachable through the existing
`--mem` flag path, which is what §7's run-2 block uses.

**§10 running log / N-registry:**
- ⭐ **§A20.3(a) is MET at toy scale, untrained: 5/5 cells, ratio 0.979–1.085 against a bar of 0.5**
  (including `run1_w40`, at N94's 40-step maturity floor, 0.9966 ± 0.022),
  blank-leak check green on three residual-specific clauses. ⛔ It is **near-by-construction at
  gate 1** and was declared so in the PREREG before measuring.
- ⭐ **The acquisition result (R1 above) is the first non-zero write-effect in the tier-iii block**,
  and it does **not** survive 200 outer toy steps significantly — both halves belong in the log.
- ⛔ **PREREG P3 refuted:** restoring the spread turned out to be *sufficient* to move acquisition off
  chance, not merely necessary; the uniform-under-shoot argument held only on §6's single lane.
