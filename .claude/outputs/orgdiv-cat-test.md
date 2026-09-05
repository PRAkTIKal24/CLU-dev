# orgdiv-cat-test — experiment-engineer report

**Task + acceptance criterion (one line):** build `PREREG-TierII.md`'s factored store and run steps
1–3 (effective-`s` + K1 · K2 · K3/K4/K5) with every K-verdict stated, publish the frozen interfaces,
score the physics arm — *a clean kill is a full acceptance*.
**Status: done.** All five K-verdicts stated multi-seed; frozen interfaces published; physics arm
scored over the registered γ axis and a `d` sweep; deletion curve run; tests green.
⛔ **Verdict: the vehicle dies at K5, and the ORGANIZER SWAP WAS NEVER REACHED.** Reported as a
pre-condition kill, **not** as a tier-ii null.

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, in the first 10 lines)
1. **`PREREG-TierII.md` §2.3 rule 4's *second* assertion is unsatisfiable at its own registered
   `m = 1`** — measured **0.5 %** of held-out queries pass. Needs `m ≥ 8`. §2.4's feasibility check
   verified only the *set* half of rule 4, never the payload half.
2. **§3.4's registered K1 prediction ("passes at `a ≥ 12`") is REFUTED**: K1 fails at `a = 12` on the
   SC-6 capture leg (0.69–0.88 vs bar 0.90), 3/3 seeds. It passes only at **`a = 32`**, whose byte
   ratio is **9.67×**, not §5.2's 5.00×.
3. ⚠ **`bprime-c6`'s `s = 0.40` may be inflated by the confinement term.** My estimator returns
   **0.438** on the same store family *without* subtracting `α‖q‖²` and **0.304** *with* — a **1.44×**
   factor. **Whoever owns `s` must check whether `CluSystem._well_fit` / the ∇V-ratio law subtract the
   confinement.** Every `d/s` statement in the program rides on this ruler.
4. **Program-wide baseline correction:** occupancy / co-activation precision above `F/N_a` is **not**
   evidence of a store effect when wells are placed at the query codes. The admissible baseline is the
   **blank store / raw launch geometry**. Measured here: launch **0.406**, settled read **0.297**.
5. `chlu/training/train_memory.py::write_loss` cannot pin a `payload_dim > 1` payload block (scalar
   `payload_index`). Not edited (out of ownership); generalised in `factored_store.py` and asserted
   equal to the shipped objective at `m = 1`.
6. **The prereg's `ratio = 1.4·A + 0.8` spelling is only valid at `d=4, m=1`.** It agrees with the
   corrected `[A(D+2)+d]/(d+m)` exactly there (both 5.00×) and disagrees at `m = 8` (12.0 vs 9.67).

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial / pillar:** TIER ii — the organization dividend.
- **Control:** the ORGANIZER SWAP. ⛔ The settle-deleted / matched-bytes launder is TIER i's control;
  run and reported below labelled **inherited diagnostic**, never as tier-ii evidence.
- **Falsifies:** F1–F5 at their registered signs/thresholds/seed counts; K1–K5 as pre-conditions that
  can kill the family *before* any arm comparison.
- **Does NOT falsify:** a TIE (`|OD_min| ≤ 0.05`); losing to a table on SEEN queries (Thm O1/D2a);
  dividend ≈ 0 on the inherited tier-i launder.

## 0. File ownership (zero-conflict rule) + compute declaration
**Created (mine):** `chlu/core/factored_store.py` · `chlu/experiments/exp_cat_test.py` ·
`tests/test_factored_store.py`.
**Edited (minimal, in-scope):** `chlu/core/monitors.py` (+14/−1, the task's named rider) ·
`tests/test_monitors.py` (+18) · `chlu/cli/experiment_cmd.py` (+41: one new subcommand + handler,
inserted before `exp-paid-access`; no existing line altered).
⛔ **Not touched:** `eval/race.py`, `eval/fb4_gate.py`, `core/psi_readout.py`, `chlu/config.py`,
`training/train_memory.py`, `core/clu_system.py`.
**Compute (declared before the first run, prereg §8.7):** wave budget 5 seeds × 5 arms × 4 readers ×
3 γ = 300 cells; **my share = the physics arm, 60 cells** + the K-sweeps. **Stagger: strictly
sequential in ONE worktree** (worktree 1 of ≤ 3), never two JAX processes at once. Spent ≈ **2 h 55 m**
wall (calibrate 586 s · controls+arm+d-sweep+deletion 5 720 s · instruments 380 s).

---

# 1. ⭐ FIRST SCREEN — the five K-verdicts, and where the wave dies

| id | check | bar | **measured (5 seeds unless noted)** | verdict |
|---|---|---|---|---|
| **K1** | write admissibility, per `a` | loss ≤ 0.05 · `λ_min>0` ≥ 90 % · capture ≥ σ_q at ≥ 90 % | `a=4`: 0.153 / 1.00 / **0.354** · `a=12`: 0.049 / 1.00 / **0.812** · **`a=32`: 0.0093 / 1.00 / 0.958** (3 seeds each) | ⛔ FAIL at 4, ⛔ **FAIL at 12**, ✅ **PASS at 32** |
| **K2** | rule 4, per held-out query | 100 % | set half **100 %** (max overlap 2 = `F−2`) · payload half **100 % at `m=8`**, **0.5 % at the registered `m=1`** | ✅ **PASS at `m = 8`** (registered deviation, §3) |
| **K3** | nearest-item table + strongest +0 B substitute | ≤ 0.60 | **0.0000** and **0.0008** | ✅ **PASS**, by a mile |
| **K4** | 4 leak controls | ≤ chance + 0.05 (= **0.0507**) | blank **0.0000** · query-only **0.0000** · permuted payloads **0.0000** · address-leak dividend **−0.109** | ✅ PASS — ⚠ **but see §1.1: vacuous** |
| ⭐ **K5** | per-item table launder, same reader class | read must beat it by **> 0.10** on ≥ 1 reader | **margin = 0.0000 on all 4 readers** (read 0.0000, table 0.0000) | ⛔⛔ **FAIL ⇒ F3 FIRES ⇒ tier ii is dead at this vehicle** |

> ## ⛔ THE ONE-LINE VERDICT
> **At an operating point that passes K1 (`a = 32`, measured `d/s = 2.70`), K2 and K3, the physics arm
> reads `0.0008 ± 0.0008` exact-set accuracy on rule-4-valid unseen combinations against a chance of
> `0.0004` — and so does every control, including the `K`-row table K5 exists to beat. K5 therefore
> fails *vacuously*: the read is not table-expressible because it is not expressible at all.
> The organizer swap was never reached, so `orgdiv-null-arms`' arms cannot be scored against a
> physics arm that has nothing to score.**

### 1.1 ⚠ Why K3 and K4 passing is NOT good news, stated where it can't be missed
K3 and K4 are *upper-bound* controls: they pass when the launder scores low. Here **every** number in
the cell is ≈ 0, so they pass **vacuously**. ⛔ A reader of this report must not quote
"K3 ✅ K4 ✅" as evidence the family is sound. The informative controls are K1 (which the vehicle
barely survives, and only at 2.7× the registered atom budget) and **K5, which it fails**.

### 1.2 ⭐⭐ THE MECHANISM, and it is the most transferable thing this wave produced

| statistic (unseen queries, 5 seeds, `a=32`, `d/s=2.70`, γ=0.05, read 400+800) | mean ± 2 SE |
|---|---|
| occupancy precision of the **raw launch geometry** (no dynamics at all) | **0.4061 ± 0.0119** |
| occupancy precision **after the two-phase damped settle** | **0.2967 ± 0.0253** |
| **the settle's dividend** | ⛔ **−0.1094** (2 SEs do not overlap) |
| chance `F/N_a` | 0.125 |

> ⭐ **The physics read does not organize the query; it destroys 27 % of the set information that the
> frozen launch geometry already contained.** The wells sit at the frozen query codes, so
> nearest-anchor assignment of the *launch point* is already a matched filter at 0.406; running the
> shipped two-phase relaxation moves particles into a *worse* assignment. This is measured against the
> tightest possible control — same φ, same anchors, same wells, the dynamics as the only difference.

**Corollary the whole program should absorb (reconciliation 4):** any co-activation / occupancy
statistic quoted against `F/N_a` overstates the store's contribution. Here the store is **above
chance (0.297 vs 0.125) and simultaneously below its own launder (0.406)**.

---

# 2. Flag provenance (mandatory — every quantitative result in this report)

Commit **`c2cc6a6`** (tree of record; `ad25d37` + the ruler pin) · branch
`agent/experiment-engineer/orgdiv-cat-test` · worktree `../CHLU-orgdiv-cat-test` · **main venv**
(`jax 0.9.0`, `equinox 0.13.4`, `optax`, `numpy 2.4.1`, float32) · **seeds 0–4** (3 seeds where noted).

| flag | value | note |
|---|---|---|
| `n_wells / f_subset / n_items / n_unseen` | 32 / 4 / 128 / 512 | registered design point, unchanged |
| **`atoms_per_well`** | **32** | ⚠ **deviation**: registered `a = 12`; K1 fails there (§3) |
| `addr_dim d` | 4 (registered) + sweep {4, 8, 16, 32} | |
| **`payload_dim m`** | **8** | ⚠ **deviation**: registered `m = 1`; K2b unsatisfiable there (§3) |
| **`payload_radius`** | **1.0**, unit-norm `v_j` | ⚠ **deviation / designed mechanism** (§3) |
| **`atom_payload_init_radius`** | **1.0** | ⭐ **designed mechanism**, 0 parameters (§3, §5.2) |
| `atom_local_radius` | 0.25 | ⭐ pilot §5.3 tooling (N98 localized init), deployed |
| `lambda_traj` | **0.0** | ⛔ pilot §5.3 trajectory write term **NOT deployed** (declared NOT-RUN, §8) |
| `s_measured` (the ruler) | **0.318** | ⭐ MEASURED, confinement-subtracted; `sep = 2.7 × 0.318 = 0.859` |
| measured `d/s` | **2.70 / 2.71 / 2.71** (`a=32`) | inside the registered band [2.5, 2.9] |
| `target_ds` / `depth_ratio` | 2.7 / 3.0 | depth heterogeneity applied to alternate wells |
| `gamma_address` / `gamma_read` | 0.05 / 0.02, axis {0.02, 0.05, **0.2**} | γ=0.2 = internal VQ-collapse control, never a claim cell |
| read budget | **400 + 800** Verlet steps, `dt = 0.05` | ⚠ every γ statement is read-budget-scoped |
| `kinetic_mode` / `M` / `p₀` | `newtonian_learned` / `I` / 0 | |
| `query_sigma σ_q` | 0.15 | |
| `n_particles P` / `launch_radius` | 4 / 0.6 | designed launch offsets = parameters, ledgered |
| write | 300 steps, adamw lr 3e-3, wd 1e-4, σ_addr 0.25, σ_pay 0.6, margin 0.15, barrier 0.2, 32 perturbs, **masked per well** | |
| organizer | **implicit** channel, 200 Adam steps @ 3e-3, batch 32, settle 150+150, ridge **0.0** | ⚠ reduced training read budget, declared |
| `confine α` | 0.05 | 2α = 0.1000 |
| soft certificate `B` | **0.542** — ⭐ `bprime-c6` §2's re-located edge, unrefuted | dependency LANDED and used |
| `tol` | `0.25 × RMS‖y − ȳ_seen‖` = 0.478 | **chance = 3.906e-4** (constant predictor) |
| bytes | store **57 344 B**, φ **576 B**, ratio_corrected **9.67×** | ⛔ reported, never claimed |

---

# 3. ⛔ REGISTERED DEVIATIONS — argued, never silent (task header requirement)

| # | prereg says | I ran | the measurement that forced it |
|---|---|---|---|
| **D1** | `m = 1` (§5.1 byte arithmetic) | **`m = 8`** | K2's own second assertion `min_B‖y(A)−y(B)‖ ≥ tol` passes on **0.5 %** of held-out queries at `m=1`. Sweep (5 seeds): `m` = 1/2/4/6/8/12 → **0.005 / 0.119 / 0.802 / 0.987 / 1.000 / 1.000**. `m` is the one symbol §2.1 leaves free (`v_j ∈ R^m`); `(N_a,F,K)` untouched. **`m=8` is the smallest passing value.** |
| **D2** | `a = 12` (§7) | **`a = 32`** | K1 **fails at `a=12`** on the SC-6 capture leg, 3/3 seeds. §3.4's registered prediction is refuted. Cost: byte ratio 5.00× → **9.67×** (reported, not claimed; tier ii is matched *across arms*, so no byte claim is at stake). |
| **D3** | `v_j` drawn (distribution unstated) | **unit-norm on `S^{m−1}`** | An i.i.d. `N(0,I_m)` draw puts targets at `‖v‖≈√m=2.83` while the read launches from payload 0 — outside basin reach. Unit radius also makes reach identical at every well, so a depth difference is a depth difference and not a distance artefact. The metric is scale-invariant in this radius. |
| **D4** | — (new) | **`atom_payload_init_radius = 1.0`**, a declared **designed mechanism** costing 0 parameters | With the historical `N(0,1)` scatter the payload block of every atom starts at radius `√m` from its target ⇒ `exp(−m/2s²) ≈ 2e-11` of signal. **Measured consequence: every well relaxed to the ORIGIN, `λ_min = 0.1000 = 2α` exactly.** N46 fairness preserved: the block is rescaled onto the target *shell*, never toward the target *direction*. |
| **D5** | organizer "implicit gradients and/or trajectory reads" | **implicit only** | Measured at init, both channels are dead (§5.1); post-write the implicit channel is 1.2–4.0× stronger. φ is frozen, so nothing upstream of `∇V` needs a trajectory read. |

---

# 4. Stage 1 — the effective-`s` instrument (OQ-1, BLOCKING) and K1

⭐ **The operating point is set on MEASURED `s`, not on the `atom_init_width` ruler** (`bprime-c6`
showed that ruler overstates the span by 1.74×). Estimator: fit `A e^{−r²/2s²}` to each written well's
radial profile on 16 random rays, **with `α‖q‖²` subtracted analytically**.

| `a` | seed | **`s` measured** | fit `R²` | well depth | **`d/s` (measured)** | `d/s` (atom-width ruler) | endpoint write loss | `λ_min>0` | **capture ≥ σ_q** | K1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0/1/2 | 0.284 / 0.281 / 0.296 | 1.000 | **0.0000** | 2.59 / 2.77 / 3.80 | 2.46 / 2.59 / 3.76 | 0.125 / 0.160 / 0.175 | 1.00 | 0.62 / 0.31 / 0.12 | ⛔ FAIL ×3 |
| 12 | 0/1/2 | 0.304 / 0.295 / 0.299 | 1.000 | 0.284 | 2.81 / 2.70 / 2.85 | 2.85 / 2.65 / 2.84 | 0.039 / 0.036 / 0.071 | 1.00 | **0.88 / 0.88 / 0.69** | ⛔ **FAIL ×3** |
| **32** | 0/1/2 | **0.312 / 0.321 / 0.320** | 0.998 | 0.289 | **2.70 / 2.71 / 2.71** | 2.80 / 2.90 / 2.89 | **0.018 / 0.010 / 0.0002** | 1.00 | **0.94 / 0.94 / 1.00** | ✅ **PASS ×3** |

- ⭐ **`s = 0.3176 ± 0.0049` at `a = 32`**, `R² = 0.998` — a *measurement*, discharging §7's "bracket,
  not a measurement" caveat for this store. **⚠ It is 21 % below `bprime-c6`'s `s = 0.40`, and I can
  account for the gap: the confinement.** Without subtracting `α‖q‖²` my own estimator returns
  **0.438** on the same store (1.44× inflation) — reconciliation item 3.
- **`a = 4` digs no wells at all** (depth 0.0000): its K1 failure is not a tolerance question.
- **The `λ_min` leg is uninformative** (1.00 everywhere, including where depth is 0.0000, because
  `2α = 0.1 > 0`). **The capture leg is the one that discriminates** — an independent reproduction of
  T4.2 / `bprime-c6`'s rider that `λ_min > 0` does not certify a nonempty basin. Two wells at `a = 12`
  had `λ_min > 0` with capture radius exactly **0.000**.
- **F4 does NOT fire:** K1 is satisfiable, at `a = 32`, ratio 9.67× — tier-ii-legal (arms matched to
  each other; no byte claim needed). The vehicle is affordable, at **2.7× the registered atom budget**.

# 5. Design-rule compliance (prereg §6) — satisfied / waived / **VIOLATED**, per row

| # | rule | status | evidence |
|---|---|---|---|
| 1 | organizer trains only through trajectory reads / implicit gradients; report `‖∂L/∂·‖` at init for every trainable group | ✅ **SATISFIED** | φ **frozen** ⇒ `∂q*/∂q₀ = 0` discharged by construction. Grad-norm table §5.1. |
| 2 | γ/M trainable only through that channel | ✅ **WAIVED as registered** | γ is a designed, swept operating point; never learned. |
| 3 | allocation collapse is a gradient-flow attractor; monitor `S_eff ∈ [S/2, S]` | ⛔⛔ **VIOLATED — the run is reported COLLAPSED, not null** | §5.2 |
| 4 | the 2α coercivity floor carried on any lifetime statement | ✅ CARRIED | `2α = 0.1000`; measured `λ_min` floor **0.0993** at undug wells — the floor *is* what `λ_min` reports when nothing was written. |
| 5 | soft certificate is the sharing precondition; use `bprime-c6`'s re-located `B` | ✅ **DEPENDENCY LANDED AND USED** | `B ≥ 0.542` (`bprime-c6` §2), declared in the config as `soft_cert_B`. ⛔ The stale `B = 0.33` is quoted nowhere. |
| 6 | if `allocate` is used: pure function of `(query, shared policy params)`; allocation-shuffle test BLOCKING | ✅ **SATISFIED, vacuously and by construction** | `allocate` is **not used**. Placement is a deterministic relaxation of the frozen codes under `(R, sep)` — **no per-item table exists**, so the shuffle test is exact. Asserted in `test_placement_is_pure_and_shuffle_invariant`. |
| 7 | the flow-map warning ⇒ K5 + report the flow-map pair per arm | ✅ ran, ⛔ **K5 FAILED** | §1, §6.3 |

## 5.1 `‖∂L/∂·‖` at init, both channels (rule 1's mandatory table; 3 seeds)

| channel | `centers` | `log_width` | `amp` |
|---|---|---|---|
| **implicit, at init (before the write)** | 7.88e-10 | 1.52e-09 | 7.26e-09 |
| **trajectory, at init (before the write)** | 1.67e-10 | 3.17e-10 | 1.57e-09 |
| **implicit, after the write** | **1.4664** | **1.1318** | **1.2450** |
| **trajectory, after the write** | 1.2489 | 0.2847 | 1.0441 |
| *prereg reference scales* | *0.0 implicit / 2.654e-9 unroll / 6.421e-3 trajectory* | | |

⭐ **Finding: the organizer is untrainable until the wells exist.** At init both channels sit at
`1e-10`–`1e-9`, i.e. at or below the prereg's *unroll* reference and **six to seven orders below its
trajectory reference** — the atoms are too far from the launch manifold for any gradient to exist.
After the write both are `O(1)`. **The write is not an initialisation detail; it is the precondition
for gradient to exist at all.** The implicit channel dominates the trajectory channel on `log_width`
by **4.0×**, which is why D5 selects it.

## 5.2 ⛔⛔ RULE 3 VIOLATED — the physics arm COLLAPSED

`S_eff = K·F/#(wells ever occupied)`, nominal `S = 16`, registered band **[8, 16]**:

| seed | 0 | 1 | 2 |
|---|---|---|---|
| wells ever occupied (of 32) | **15** | **10** | **14** |
| **`S_eff`** | **34.1** | **51.2** | **36.6** |
| in band [8, 16]? | ⛔ no | ⛔ no | ⛔ no |

> **Per prereg §6 rule 3 this run is reported as COLLAPSED, not as a null.** The read concentrates on
> 31–47 % of the well vocabulary — the factored analogue of allocation collapse, and exactly the
> gradient-flow attractor the rule was written to catch. ⚠ **This means the `0.0000` scores of §1 are
> not a clean measurement of "physics-trained organization"; they are a measurement of a collapsed
> one.** A future revival must fix collapse first — it is a concrete, named, pre-registered target.

---

# 6. The physics arm, the γ axis, and the falsifiers

## 6.1 Scores (5 seeds; `a=32`, measured `d/s=2.70`; chance **3.906e-4**; read 400+800)

| reader | params | **unseen** | **seen** | ⭐ accuracy-vs-`tol` curve (×0.25 / ×0.5 / **×1** / ×2 / ×4) |
|---|---|---|---|---|
| `sum_linear` | 104 | 0.0008 | 0.0109 | 0.0000 / 0.0000 / **0.0008** / 0.0187 / **0.5539** |
| `well_table` | 72 | 0.0008 | 0.0031 | 0.0000 / 0.0000 / **0.0008** / 0.0195 / **0.5801** |
| `knn` | 0 (+128 stored) | 0.0000 | **0.9984** | 0.0000 / 0.0000 / **0.0000** / 0.0125 / 0.4473 |
| `mlp` | 92 | 0.0004 | 0.0000 | 0.0000 / 0.0000 / **0.0004** / 0.0199 / 0.5687 |

- ⭐ **Quote the curve, not the endpoint** — and the curve earns its keep: at the registered `tol` the
  arm reads `0.000`, but at `4×tol` it reads **0.45–0.58**. **The store is POOR, not INERT.** It puts
  the read in the right neighbourhood of `y(x)` and never on it. (Contrast `cluformer-pilot` §5, where
  live == blank to float32 round-off; that is *not* what is happening here.)
- ⭐ **The kNN reader that beat us in C2W1 scores 0.9984 on SEEN and 0.0000 on unseen.** That is
  rule 4 doing exactly its job: a memorising reader is *structurally* excluded from the held-out
  combinations. It is the cleanest evidence in this report that the family's construction is sound
  even though its vehicle is not.

## 6.2 `OD_min` against the two in-house nulls (N3 = F5's fitted static-geometric rule, N4 = kNN)

⚠ `null*` here is **in-house only** (N3, N4). N1/N2/N5 belong to `orgdiv-null-arms` and are NOT
included; this is **not** the wave's `null*`.

| γ_address | `OD_min` (mean ± 2 SE, 5 seeds) | F5 assignment agreement | occupancy precision | claim cell? |
|---|---|---|---|---|
| 0.02 | **−0.0012 ± 0.0010** | 0.211 | 0.276 | no (below band) |
| **0.05** | **−0.0016 ± 0.0015** | 0.228 | 0.299 | ✅ **yes** |
| 0.20 | **−0.0012 ± 0.0010** | 0.233 | 0.300 | ⛔ no — internal VQ-collapse control |

## 6.3 Falsifiers, adjudicated mechanically against the prereg's own bars

| # | bar | measured | verdict |
|---|---|---|---|
| **F1** | fires iff `mean+2SE < +0.05`; ambiguous band `\|OD_min\| ≤ 0.05` ⇒ **TIE** | `−0.0016 ± 0.0015` | ⭐ **TIE** — the prereg's registered most-likely outcome (prior 0.55), and per §3.1's explicit ambiguous-band rule it is reported as TIE. ⚠ Its literal fire condition is also satisfied; **both stated, because the number is vacuous** (every arm reads 0.000). |
| **F2** | fires iff max `OD > +0.05` while min `OD < 0` | all four readers within `±0.004` of 0 | ✅ **does not fire** |
| **F3** | fires iff K3 **or** K5 fails | **K5 fails** (margin 0.0000, bar 0.10) | ⛔⛔ **FIRES.** ⚠ *Interpretation matters:* it fires because the read does not BEAT a table, not because the answer was in one — the `K`-row table also scores 0.0000, and K3 is 0.0000. |
| **F4** | fires iff K1 unsatisfiable at any affordable `a` | K1 passes at `a=32`, ratio 9.67× | ✅ **does not fire** — the vehicle is affordable, at 2.7× the registered budget |
| **F5** | fires iff the fitted static-geometric null reproduces the physics assignment on ≥ 99 % | **0.211 / 0.228 / 0.233** at γ = 0.02 / 0.05 / 0.2 | ✅ **does not fire at ANY γ** — including γ=0.2, where the prereg registered a fire. ⚠ **Vacuous in the other direction:** the physics assignment is not reproduced by a static rule because it is close to *noise*, not because it carries a structurally non-VQ channel. The prereg's registered irreducible disagreement was 0.193–0.203; measured **0.77–0.79**. |

## 6.4 The inherited tier-i diagnostic (⛔ NOT tier-ii evidence — CM-27(b) adjacency honoured)
Settle-deleted launder (payload channel of the settled read zeroed), same reader class:
**0.0000 on all four readers vs the physics arm's 0.0008** ⇒ dividend **≈ 0**, exactly as CM-27(b)
predicts by design at tier ii. ⭐ **The organizer-swap result sits in the same paragraph, as the
adjacency rule requires: the in-house organizer swap is a TIE at `−0.0016 ± 0.0015` (§6.2), and the
wave's real swap was never reached because K5 killed the cell first.**

---

# 7. ⭐⭐ SP-2 — the structural result, pre-registered and measured

**Registered before the harness existed** (`.claude/outputs/orgdiv-cat-test/PREREG.md` §SP-2): the
address dimension `d` is squeezed from both sides — the store needs `d ≳ 2F ln(N_a/F) = 16.6` to
recover `A(x)` from the set-code at all, while the query-only control needs `d ≪ N_a` to stay at
chance — **and I predicted the window is EMPTY**.

## 7.1 The family-level sweep (5 seeds, numpy only)
| `d` | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| rank ceiling `d/N_a` | 0.125 | 0.250 | 0.500 | 0.750 | 1.000 | 1.500 | 2.000 |
| **query-only `R²` (the leak)** | 0.102 | 0.172 | 0.388 | 0.676 | **0.983** | 0.987 | 0.988 |
| matched-filter set precision | 0.371 | 0.527 | 0.694 | 0.807 | 0.861 | 0.958 | 0.989 |
| **matched-filter EXACT-set** | 0.006 | 0.021 | 0.140 | 0.337 | 0.481 | 0.832 | 0.956 |

## 7.2 The measured arm sweep (3 seeds, full store, `a = 32`)
| `d` | **physics best** | **query-only best** | K4 bar | K4 ok | **occupancy precision (unseen)** | write loss |
|---|---|---|---|---|---|---|
| **4** (registered) | **0.0000** | 0.0000 | 0.0507 | ✅ | 0.293 | 0.0042 |
| 8 | **0.0000** | 0.0000 | 0.0507 | ✅ | **0.417** ← peak | 0.0001 |
| 16 | **0.0013** | 0.0026 | 0.0507 | ✅ | 0.344 | 0.0001 |
| 32 | **0.0000** | **0.0339** | 0.0507 | ✅ | **0.188** ← collapses | 0.0029 |

> ⭐ **SP-2a CONFIRMED, and in a stronger form than I registered.** There is no `d` at which the
> physics arm exceeds chance + 0.05 — not one. The store's own set recovery is **non-monotone**: it
> peaks at `d = 8` (0.417) and *collapses* to 0.188 (below chance 0.125's neighbourhood, and below
> every other cell) by `d = 32`, exactly where the query-side leak finally switches on
> (`R² = 0.983`, query-only accuracy 0.034). **The store gets worse in precisely the regime the query
> gets easy.**
> ⚠ **Where I was wrong, and it matters:** I predicted K4 would *fail* for `d ≥ 16`. It does not —
> because at `m = 8` chance collapses to `3.9e-4` and exact-set accuracy is all-or-nothing, which
> punishes the query-only reader's `R² = 0.98` just as hard as it punishes the store. My `R²`
> predictions were right (0.388 at `d=16`, 0.983 at `d=32`); my translation of them into exact-set
> accuracy was derived at `m = 1`'s arithmetic and is refuted.

## 7.3 SP-1 — the family's structural ceiling (⛔ DECLARED OUT-OF-CLASS DIAGNOSTIC)
An OLS fit of `y` on the **true indicator** (`N_a·m = 256` dof) fitted on the 128 SEEN items:
**exact-set accuracy `1.0000` on the 512 unseen queries, `‖v̂ − v‖∞ = 4.25e-15`** (5 seeds).
⭐ **Registered prediction was ≥ 0.95 and `< 1e-6` — CONFIRMED, sharply.**
⛔ **This is reported and never scored as an arm or as a K4 leg.** It is why the reader class is
capacity-bounded below `N_a·m`: the ground truth `1_A ↦ y` is a 256-parameter *linear code*, so a
reader that big solves the family with no store on any arm, including a blank one. **The escape is
`K < N_a`** (verified: at `K=12 < N_a=16` the probe cannot recover `v`); the registered design point
has `K = 128 > N_a = 32` and is squarely in the solvable regime.

---

# 8. PREREG SCORECARD (`.claude/outputs/orgdiv-cat-test/PREREG.md`, filed 10:23Z before any harness code)

| # | registered | measured | verdict |
|---|---|---|---|
| **SP-1** | out-of-class probe ≥ 0.95, `‖v̂−v‖∞ < 1e-6` | **1.0000**, **4.25e-15** | ✅✅ |
| **SP-2a** | no `d` with K4-pass **and** physics > chance+0.05 | none, at any `d` | ✅ **CONFIRMED** |
| SP-2b | physics at `d=8` ∈ [0.20, 0.28] | **0.0000** | ⛔ REFUTED (over-predicted) |
| SP-2c | at `d=32`: physics ≥ 0.50, query-only ≥ 0.90 | **0.0000**, **0.034** | ⛔ REFUTED — the `R²` half was right (0.983), the accuracy half was derived at the wrong `m` |
| SP-2d | occupancy monotone in `d`, crosses 0.50 between 16 and 32 | **non-monotone**, peaks 0.417 at `d=8`, never crosses 0.50 | ⛔ REFUTED |
| I1 | reproduce `bprime-c6`'s `s = 0.40 ± 0.06` on their rig | ⛔ **NOT RUN** (declared, §9) — but the 1.44× confinement mechanism is measured | — |
| I2 | `s ∈ [0.28, 0.55]`, point 0.40 | **0.3176 ± 0.0049** | ✅ in range; point prediction **26 % high** |
| I3 | `sep ∈ [0.76, 1.49]`, point 1.08 | **0.859** | ✅ in range; point high |
| K1 | passes at `a ≥ 12`, fails at `a ≤ 4` | fails at 4 **and 12**; passes at 32 | ⛔ **REFUTED** (and so is the prereg's own §3.4) |
| K2 | 23 193 ± 2 300 valid held-out; 100 % of queries pass | **24 046**; ✅ · **0.5 %** at registered `m=1` | ✅ count · ⛔ **the 100 % is REFUTED at `m=1`** |
| K3 | 0.15 ± 0.10 | **0.0000 / 0.0008** | ◐ **sharper than registered** (outside the band, on the good side) |
| K4 | legs within ±0.03 of chance; query-only 0.20–0.30; `P(fail) = 0.35` | all **0.0000**; did not fail | ✅ pass · ⛔ the 0.20–0.30 figure was `m=1` arithmetic |
| K5 | **PASSES**, margin +0.15 | ⛔ **FAILS**, margin **0.0000** | ⛔⛔ **REFUTED — and this is the wave's verdict** |
| F1 | TIE, `OD_min = 0.00 ± 0.04` | **−0.0016 ± 0.0015** | ✅✅ |
| F2 | does not fire | does not fire | ✅ |
| F3 | does not fire | ⛔ **FIRES** | ⛔ REFUTED |
| F4 | does not fire; predicted a 4.40-vs-5.00 byte-law discrepancy | does not fire; **both laws give 5.00** at `d=4,m=1` (`D = 5`, not 4) | ✅ verdict · ⛔ **my discrepancy prediction was wrong** |
| F5 | does not fire at γ=0.05 (agreement 0.85–0.98, point 0.93); **fires** at γ=0.2 | **0.228** at γ=0.05; **0.233** at γ=0.2 | ⛔ **REFUTED at both ends** — no fire anywhere, and the agreement is 4× lower than predicted |
| A1 | physics unseen 0.35 ± 0.15 | **0.0008** | ⛔ REFUTED |
| A2 | occupancy precision 0.45 ± 0.15 | **0.297 ± 0.025** | ◐ just inside the band — **but the baseline moved**: the launder scores 0.406 |
| A3 | seen 0.15 above unseen | `sum_linear` +0.010; `knn` **+0.998** | ◐ mixed; the kNN row is the sharp one |
| **A4** | `P(physics indistinguishable from chance) ≈ 0.40` | **it landed** | ⭐ the risk I named in advance is the outcome |
| M11 | `saddle_reach_threshold` still raises via `s*s` underflow | reproduced in one line (`s=1e-200`); denominator guard is the fix | ✅✅ |
| DEL | exactness 1.0 by construction; degradation monotone, ≥ 2× from `p=0.0045` to `p=0.094` | exactness ✅; degradation **0.0000 at every `p`** | ✅ · ⛔ REFUTED (**vacuously** — the read is already at 0) |

**Score: 9 ✅ · 4 ◐ · 12 ⛔.** ⭐ **The scorecard's own shape is a finding: I systematically
over-predicted the store's capability, at every single point where I put a number on it.** The two
predictions that survived sharply (SP-1, SP-2a) are both *structural* claims about the family; every
refuted one is a claim about what the physics would achieve.

---

# 9. The deletion curve (prereg §5.3 — TWO series, ⛔ never a scalar)

| `p` (private parameter-mass fraction) | 0.0045 | 0.027 | 0.036 | 0.094 |
|---|---|---|---|---|
| private atoms deleted (of 1 024) | 5 | 28 | 37 | 96 |
| **series 1 — exactness on the private fraction** (byte equality) | **1.0 ✓** | **1.0 ✓** | **1.0 ✓** | **1.0 ✓** |
| **series 2 — measured degradation on the shared fraction** | −0.0007 | −0.0007 | −0.0007 | −0.0007 |

⛔ **Series 2 is VACUOUS and is reported as such, not as "deletion is free."** The shared-fraction read
was already at `0.0000` before the deletion, so there is nothing left to degrade. Series 1 is exact by
construction (the private rows are zeroed byte-for-byte, verified with float equality) and is
therefore **not a result either** — it is the definition. ⚠ **A deletion claim on this substrate needs
a working read first**; this curve is a declared placeholder, not evidence.

---

# 10. How I verified (commands + observed output)

```
# worktree, main venv reused (protocol §4 w6 lesson): jax 0.9.0 / equinox 0.13.4 / numpy 2.4.1
git worktree add ../CHLU-orgdiv-cat-test -b agent/experiment-engineer/orgdiv-cat-test main
PYTHONPATH=$PWD /Users/user/Desktop/CHLU/.venv/bin/python -m pytest tests/ -q     # see below
/Users/user/Desktop/CHLU/.venv/bin/python -m ruff check chlu/ tests/              # All checks passed
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-cat-test --help                      # CLI hook live
PYTHONPATH=$PWD .venv/bin/python -m chlu exp-cat-test --quick                     # 286 s, all 5 stages
# the reported runs (sequential, one process):
  stage_family(seeds=0..4)                       9 s
  stage_calibrate(a=4,12,32 x seeds 0..2)      586 s
  run_cat_test(stages=controls,arm,d_sweep,deletion; atoms_per_well=32)  5 720 s
  design-rule instruments (grad norms, S_eff)   380 s
```
Artifacts: `.claude/outputs/orgdiv-cat-test/results/{stage_family,stage_calibrate,stage_controls,
stage_arm,stage_d_sweep,stage_deletion,design_rules,cat_test_summary}.json` ·
`.claude/outputs/orgdiv-cat-test/cat_test_summary.png` (6 panels) ·
`.claude/outputs/orgdiv-cat-test/FROZEN-interfaces.md` · `.../PREREG.md`.

**Tests (on my branch):**
- `tests/test_factored_store.py` — **26 passed** (new).
- `tests/test_monitors.py` — **42 passed** (41 + the new underflow guard test).
- ⭐ **FULL SUITE ON MY BRANCH: `1216 passed, 31 warnings in 1205.85s (0:20:05)` — zero failures,
  zero errors, zero skips.** (Baseline at the C2W4 close was 1190; +26 are mine.)

---

# 11. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)

1. **The organizer swap itself (N1/N2/N5).** Not run — `orgdiv-null-arms` owns them, and K5 killed the
   cell before they were needed. `OD_min` in §6.2 is against **in-house N3/N4 only** and is **not**
   the wave's `null*`.
2. **The pilot's §5.3 trajectory write term (`lambda_traj > 0`)** and the **ψ payload residual.** The
   plumbing is present (`CatTestConfig.lambda_traj`) but both are **NOT deployed**. Only the third
   candidate fix (localized atom init) was, plus the new `atom_payload_init_radius` mechanism. ⚠ A
   revival should try these before concluding the physics cannot organize.
3. **The tuning grid.** Neither arm received the registered `≥5 lr × 3 capacity × 3 seeds`. The
   physics arm ran at ONE configuration. ⛔ **Therefore §6's numbers may not carry a headline as
   "physics loses" — they carry "physics, untuned and collapsed, does not clear its own K5".**
4. **`bprime-c6`'s rig, re-measured with my estimator (I1).** Not run; the 1.44× confinement mechanism
   is measured on *my* store only. Reconciliation item 3 is a *flag*, not a refutation of their number.
5. **`m` between 1 and 8, and `a` between 12 and 32.** K2 and K1 were bracketed on a coarse grid; the
   exact thresholds are bracketed, not located.
6. **Plots of the landscape itself** (force fields / potential surfaces). Not produced.

---

# 12. Git footprint

- **Branch:** `agent/experiment-engineer/orgdiv-cat-test` (off local `main` @ `29fc22b`), in worktree
  `../CHLU-orgdiv-cat-test`. ⛔ Not pushed, no PR, no merge — left for review.
- **Commits** (verified from the MAIN repo, protocol §3.2):
  - `91e65e1` `[experiment-engineer] move monitor #11's zero-depth guard into monitors.py`
  - `ad25d37` `[experiment-engineer] the factored store + the cat test (C2W5 tier-ii vehicle)`
  - `c2cc6a6` `[experiment-engineer] pin the operating point to MEASURED s; endpoint write loss`
- **Files touched:** `chlu/core/factored_store.py` (new, 1 239 L) · `chlu/experiments/exp_cat_test.py`
  (new, 833 L) · `tests/test_factored_store.py` (new, 386 L) · `chlu/core/monitors.py` (+14/−1) ·
  `tests/test_monitors.py` (+18) · `chlu/cli/experiment_cmd.py` (+41).
- **Rebase:** onto local `main` (⚠ **not** `origin/main`, §7.21) — **no-op, base unmoved** (`main`
  still at `29fc22b`).
- **Branch ref verified FROM THE MAIN REPO** before finishing (protocol §3.2, the wave-4 lesson):
  `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/orgdiv-cat-test`
  lists all three commits. **The worktree is left in place** (not removed) for Hub review.
- ⚠ **A concurrent worktree exists** (`../CHLU-placement`, `pilot-placement-probe`). No file overlap:
  it touches `blocks.py`/`exp_cluformer_pilot.py`/`exp_placement_probe.py`/`train_cluformer.py`/
  `scripts/csf3/`/`test_placement_probe.py` — **verified by diffing its branch against `main`: zero
  file overlap with mine, and it does NOT touch `monitors.py`.** No merge conflict is expected.
- **Conflicts:** none. No other agent's uncommitted work was present in the shared checkout.

---

# 13. Open questions / follow-ups / risks

1. ⭐ **Is the destruction in §1.2 a property of the CLU or of my launch protocol?** The settle moves
   particles *off* the correct wells. Candidate causes, none tested: `P = 4` particles launched from
   one set-code with fixed offsets all fall into the *same* deepest basin (consistent with the `S_eff`
   collapse); or the payload block's 8 extra dimensions dominate the metric that decides which basin
   wins. **A one-day probe: measure how many DISTINCT wells the P particles occupy per query.**
2. **The collapse (§5.2) is the first thing to fix, and it is named and pre-registered.** Nothing in
   this report distinguishes "physics cannot organize" from "this physics arm collapsed."
3. **`d/s` band:** the whole cell ran at 2.70–2.71, inside the registered band. The band was derived on
   a **1-D/2-D single-atom toy at `p₀=0, T=0`**; this is the first learned multi-atom store to run in
   it, and the transfer is still bracketed.
4. **Risk to the wave plan:** `orgdiv-null-arms` can build its arms against the frozen interfaces, but
   **there is no physics arm worth swapping against**. Suggest the Hub re-scope it to *"score the null
   arms on the same family and report whether ANY organizer clears chance"* — if none do, the family
   is refuted for everyone and that is a much stronger, cheaper statement than a tier-ii null.
5. **The `m ≥ 8` requirement changes the byte story** (ratio 9.67× at `a=32`, `m=8`). Any future
   byte-frontier ambition for tier ii is now more expensive than prereg §5.2 costed it.

---

## Proposed handover updates (for the Hub)

- **§7 Known Issues — RESOLVED:** `saddle_reach_threshold`'s `ZeroDivisionError` (the pilot's R4 /
  `cluformer-pilot` §6.4) is fixed **inside `monitors.py`** with a denominator guard + test
  (`91e65e1`). The caller-side guard in `train_cluformer.py` (`7bc166a`) is left in place as a
  semantic exclusion. The known-issues entry can move to resolved.
- **§7 Known Issues — NEW (open):** `chlu/training/train_memory.py::write_loss` takes a **scalar**
  `payload_index` and therefore cannot pin or jitter a `payload_dim > 1` payload block; any caller
  with `m > 1` silently gets a query manifold that does not match its read. Not fixed (ownership).
- **§7 Known Issues — NEW (open, program-wide ruler):** the effective-`s` estimator must subtract
  `α‖q‖²`, or a pure confinement bowl log-fits as a well. Measured inflation **1.44×** on this store.
  Flags `bprime-c6`'s `s = 0.40` for a check (reconciliation 3).
- **§3 config:** new `chlu exp-cat-test` CLI command; new `CatTestConfig` (in
  `chlu/core/factored_store.py`, **not** `chlu/config.py`, per the `SoftCertificateConfig` precedent).
- **Registry/doctrine candidates:** (i) *occupancy/co-activation precision must be scored against the
  blank store, never `F/N_a`* — measured 0.406 launder vs 0.297 settled; (ii) *the organizer is
  untrainable until the wells exist* — grad norms `1e-10` at init, `O(1)` after the write; (iii) *an
  additive-payload compositional family with `K > N_a` is a linear code and is solvable by any reader
  with `≥ N_a·m` parameters* (SP-1, `1.0000` / `4.25e-15`) — this constrains every future
  compositional gym family the program builds, not just this one.

---

## ⚠ DATED CURATOR PROVENANCE FOOTNOTE (2026-08-06, `doc-curator-c2w7-fold`, [C2W7] — body above UNTOUCHED, C-3 precedent) — ⛔ **NO VERDICT MOVES**

**§6.1's published SEEN column is UNREPRODUCIBLE FROM THIS REPORT'S OWN ARTIFACTS.** Hub-verified at
the C2W7 close: **`sum_linear = 0.0109375`** and **`well_table = 0.003125`** appear **nowhere in any
`orgdiv-cat-test/results/*.json`**, and `stage_arm.json` carries **no `seen`/`unseen` keys at all**
(top-level: `cells` / `readers` / `aggregate`).

⭐ **The values WITH provenance are `reader-fitting-audit`'s re-measurement on the same cell
(`γ = 0.05`, 5 seeds): SEEN `sum_linear` 0.0094 · `well_table` 0.0047** (also re-measured: `knn`
0.9969, `mlp` 0.0016, `sum_identity` 0.0031, `well_identity` 0.0000). ⇒ **cite the re-measured values;
read the published SEEN column as unreproducible-from-artifacts.**

⛔⛔ **This is a PROVENANCE FOOTNOTE, NOT AN ERRATUM.** **SEEN was never the metric** — unseen is, and
the unseen column **reproduces BIT-FOR-BIT** through the audit's reproduction gate, including the
non-zero seeds 1–2 (`sum_linear` / `well_table` **0.00078 ± 0.00096**, `mlp` 0.00039 ± 0.00078, `knn`
0.0000). **K5's kill, §5.2's collapse reading and the published `0.0008 ± 0.0008` all stand unchanged.**

⚠ **Two further C2W7 notes on this cell, filed so a reader does not inherit them wrongly:**
1. **The identity reader is STRICTLY WORSE than the fitted one here** (unseen **0.0000** vs
   **0.00078**, on 2 of 5 seeds) — reported as a negative, and it is part of why a zero-parameter
   reader is **added to** a class, never substituted for it (charter §A26.3; **N237**).
2. ⛔ **`sum_linear`'s least-squares fit INFLATES at this cell** (`diag(W) = +46.53`, mean residual
   **13.48** against the identity's **1.84**) ⇒ **a fitted reader's residual is never evidence that a
   store is "close"** (**N238**). ⚠ The engineer flags this may be a **store** fact (the settled
   payload block sits at ~1/46 of the family's scale) — **owner needed; not resolved here.**

⚠ **`S_eff` re-labelling (charter §A26.4, **N239**): this report's `S_eff` 34.1 / 51.2 / 36.6 is
CONCENTRATION, and "COLLAPSED" is now reserved for exactly that — so §5.2's label SURVIVES.** ⛔ What
does not survive is the same word at the band's upper edge (C2W7's 16.77 = **30.6 of 32 wells
visited**, i.e. slight UNDER-usage): the `[8, 16]` band's lower half is unreachable by construction.

**Sources:** `.claude/outputs/reader-fitting-audit.md` §5/§6/§7/reconciliation 2 · charter
**ADDENDUM 8 §A26.3/§A26.4** · the 2026-08-06 `[C2W7]` §10 entry. ⛔ **`PREREG-TierII.md` is not
edited and no `ERRATA-TierII.md` is owed** (the audit's verdict was SURVIVES).
- **`PREREG-TierII.md` errata:** reconciliation items 1, 2 and 6 above.
