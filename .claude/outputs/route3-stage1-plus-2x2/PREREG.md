# PREREG — `route3-stage1-plus-2x2` (C2W3, experiment-engineer)

**Written before any measured run of D1–D4.** Base `main @ 6ff4c1d`, worktree `../CHLU-route3`,
branch `agent/experiment-engineer/route3-stage1-plus-2x2`. Env: main venv reused, **JAX 0.9.0**
(verified: `import 4.04 s`, `chlu.__file__ = /Users/user/Desktop/CHLU-route3/chlu/__init__.py`).

Only runs executed before this file was written: (a) a `--quick` smoke of the *shipped*
`exp_memory_gym.run_cell` (warm-up), (b) a wall-clock timing probe of three shipped gym cells.
Neither measures any quantity this prereg predicts (both are shipped-code health checks).

---

## 1. The §A9.4 bar, VERBATIM (charter Addendum 2 §A9.4; Head pre-registered, applied arithmetically)

> **Stage 2 unlocks iff the per-slot store-attributable discriminability (full − settle-deleted
> launder, per slot `t`) clears the launch-noise floor BEYOND 2 SE, at ≥3 seeds, on ≥1 family, at ANY
> `t` — q-slots and p-slots scored separately (a live p-channel unlocks even with a dead q-channel).**

### 1.1 The exact arithmetic I will apply

For each `(family, channel c ∈ {q,p}, slot t)`:

1. per seed `s ∈ {0,1,2}`: `margin(t,c,s) = [ full(t,c,s) − launder(t,c,s) ] − floor(t,c,s)`
2. `mean = mean_s margin`, `sd = sample sd (ddof=1)`, `SE = sd/√3`
3. **clears ⇔ `mean − 2·SE > 0`** (strict), n = 3 seeds, all three admissible.
4. `unlock = true` ⇔ ≥1 `(family, channel, slot)` clears **and** the family is admissible
   (≥1 admissibly-written cell; a family with 0 admissible cells **ABSTAINS** after one bounded
   escalation and neither unlocks nor blocks).
5. ⛔ **§A9.5 overrides:** if the per-slot matched-bytes table launder reproduces the slotted read,
   Route 3 fails **regardless** of (4).

### 1.2 The three quantities, defined operationally (declared, because the charter names them but the
repo has no such object yet)

* **Discriminability `D(t,c)`** — the per-slot read is the family's own answer channel taken from the
  slot state: for `overload`/`aggregate` the **payload block** of `q_t` (channel `q`) or of `p_t`
  (channel `p`); for `manifold` the **spectator block**. `D = |Spearman ρ|` between that scalar and
  the query's own target. ⭐ Spearman is chosen **before** any run because it is **scale-free**: at
  small `t` the momentum channel is `O(t)` small in magnitude but (per §A8.1) *proportional* to the
  store's payload, and any magnitude-based decode would confound "small" with "uninformative". No
  fitted parameters, no bytes, identical instrument on every arm and both channels.
* **The per-slot settle-deleted launder `launder(t,c)`** — the identical read (identical launch,
  identical integrator, identical slot index, identical instrument) on the **store-deleted** system:
  the harness's own blank/unwritten store (`build_system(replace(cfg, seed=seed+991))`, the control
  `exp_memory_gym.run_cell` already builds). ⚠ **Declared mapping:** the shipped
  `settle_deleted_launder` is a *settled-point* table object and has **no slot index**; the per-slot
  instantiation of "delete the settle" is "delete the store that creates the settle", which leaves the
  launch and the dynamics intact. The matched-bytes TABLE launder is the separate §A9.5 object (§4).
* **The launch-noise floor `floor(t,c)`** — the same `D` measured on the store-deleted arm under an
  **independently re-drawn launch perturbation of the same law** (`+N(0, σ_q)` on the address block,
  σ_q = the family's own `query_sigma`): the discriminability that launch-cloud noise alone produces
  at that slot. It is subtracted **in addition** to the launder, so the bar is conservative by
  construction (a slot must beat both its store-deleted twin and the noise level of that twin). The
  un-floored dividend `full − launder` is reported beside it so the Hub sees both.

## 2. The slot/stride grid — DECLARED BEFORE LOOKING AT ANYTHING

`CluSystem.read` records a strided buffer at `traj_stride = 8`: **phase 1** (`address_steps = 400`)
⇒ 50 points, **phase 2** (`read_steps = 800`) ⇒ 100 points; point index `j` within a phase is
integrator step `8j + 1`, i.e. `t = 0.05·(8j+1)` time units at `dt = 0.05`. Slots are the
concatenated-buffer point indices

```
S = {0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 49,   # phase 1 (launch → address settle)
     54, 62, 74, 99, 124, 149}                      # phase 2 (→ settled point)
```

19 slots, **log-dense at the small-`t` end** because that is where §A8.1 makes its prediction. If a
retry fires and appends points, slots are still indexed off the phase-1/phase-2 boundaries recorded
in `res.phase`; extra retry points are not added to the grid.

## 3. Predicted SHAPE of the attribution curve (q and p separately)

Derivation (from the shipped read, not from hope): the read launches at `p₀ = 0` **exactly**
(`clu_system.read`: `p0 = jnp.zeros_like(q0)`) and with the **payload channels of `q₀` zeroed**.
Therefore, to leading order in `t`,

* `p_t = −∫₀ᵗ ∇V(q) dt' ≈ −t·∇V(q₀)` — and `∇V` **is** the store. The payload component of `∇V(q₀)`
  is (for a Gaussian well at payload `a_i`) `∝ (0 − a_i)`, i.e. **exactly proportional to the stored
  payload**. ⇒ `D_p(t)` should be **high at the very first slot** and roughly flat until dissipation
  and well-crossing scramble it.
* `q_t − q₀ = O(t²)`, and `q₀`'s payload block is identically 0 ⇒ `D_q(t) ≈ floor` at small `t`,
  rising monotonically as the particle falls into the well, saturating near the phase-1/phase-2
  settle where the gym's shipped `decode = 1.000`.

**P1 (the Advisor's on-the-record prediction, confirmed/refuted here):** `D_p` at the smallest slot
(`j = 0`, `t = 0.05`) ≥ **0.80**, and `D_q` at `j = 0` ≤ **0.20** (both `|ρ|`, `overload`).
**P2:** `D_q` crosses 0.5 between `j = 8` and `j = 49`, and is ≥ 0.85 by `j = 99`.
**P3 (the unlock):** **`unlock = true`, driven by the p-channel at small `t`** on `overload`
(`load1x_shipped`), with margin ≥ +0.5 at `j ∈ {0,1,2}`; q clears only at large `t`.
**P3′ (the registered alternative):** if the store-deleted arm's confinement gradient alone
correlates with the payload (it should not — the blank store has no payload structure), both
channels' margins collapse to ≈ 0 and `unlock = false`. Registered so the null is not post-hoc.
**P4 (Jacobian, §A8.2):** the flow map is **contractive within the launch cloud** — within-item
spread ratio `w(t)/w(0) < 1` for all `t` beyond the address settle — and **separated across items**:
between-item spread `b(t)/b(0)` stays ≥ 1 in the address block. Predicted crossover (contraction
begins to dominate) at `j ≈ 16–32`.

## 4. §A9.5 — the per-slot matched-bytes TABLE launder, and whether I expect it to KILL

**Object:** at each slot `t` and channel `c`, a table of **K rows** (one per live item) holding that
item's **mean slot content**, keyed by the query's nearest stored key; evaluated **leave-one-query-out**
(the evaluated query never contributes to its own row) and scored with the *identical* instrument.
Bytes: `K × dim × 4 B` per slot — strictly **less** than the store's own byte ledger (≥ 478× on
`overload`), so it is a *cheaper* reader. Margin reported per slot as `read − table`, signed.

**P5 — I expect §A9.5 to FIRE.** Derivation: on the existing merged rig nothing couples slots; the
slot content of a query is (to the accuracy of nearest-key assignment) a deterministic function of
which item the query addresses, so K time-indexed rows can express it. Registered prediction: the
table's `D` is **within ±0.05 of the read's at ≥ 80 % of slots**, and the read beats the table by
> 0.10 at **no** slot at which it also clears §A9.4. If that holds, **Route 3 fails regardless of the
unlock arithmetic** (intervention §8.2) and I report it in the first 10 lines.
**Registered alternative P5′:** the read beats the table where nearest-key assignment is *wrong*
(interference/ambiguity under overload) — if the margin is > +0.10 at ≥ 3 slots in a channel that
also clears §A9.4, §A9.5 does **not** fire and Route 3 keeps a headline claim.

## 5. Predicted 2×2 outcome (D4), with the derivation

Arms: store ∈ {`gauss`, `shell` (learned radius)} × write ∈ {`endpoint` (control),
`path@λ_path=0.3`}, families `overload/load1x_shipped` and `manifold/base`, 3 seeds.
λ_path = 0.3 is the largest coefficient at which C2W2 measured the write still admissible
(λ_path ≥ 3 drives `λ_min` to −0.48…−1.30).

**P6:** both bit-identity gates PASS — (a) at `λ_traj = λ_path = 0` the written `V` is bit-identical
to `main`'s (the terms sit behind a Python-level `>0` branch), and (b) at `r = 0` the shell reduces to
the Gaussian bit-identically **including with the path term on** (the term touches the write loss, not
the potential's algebra).
**P7 — the 2×2's unrun cell does NOT rescue the shell.** Derivation: C2W2 measured Route 1's path/traj
penalty as an **exactly constant** `L = L_endpoint + 0.55·λ` — the optimiser never descends it — so it
supplies no gradient to any store parameter, learned radius included. Predicted `|Δr|` on the
`shell × path` cell < **0.01** (C2W2's radius moved 0.500 → 0.501 with the term OFF), and its dividend
≤ 0, within 0.05 of the `shell × endpoint` cell.
**P7′ (registered alternative):** the path term *is* the signal the radius is blind to ⇒ `|Δr| ≥ 0.05`
and the `shell × path` dividend exceeds `shell × endpoint` by ≥ 0.05. Either outcome is a result; the
2×2 is a declared re-price, **not** a second gate.
**P8:** the `gauss × endpoint` cell reproduces a direct unpatched `exp_memory_gym.run_cell` **digit
for digit** (identical code path ⇒ bitwise identical scores).

## 6. Falsifiers (task §5, restated as what I will report)

* no slot on any family clears §A9.4 in either channel ⇒ **Route 3 DEAD at this weight class**;
* the §A9.5 table launder reproduces the slotted read ⇒ **fails regardless of dividend** (overrides);
* either bit-identity gate fails ⇒ the 2×2 is uninterpretable, stop and report;
* the trajectory launder fires on any ψ I use ⇒ no ψ number from that arm is quotable.
  ⚠ I use **no learned ψ** in D1–D4: the attribution instrument is a fixed rank correlation on raw
  slot channels, and `AttentionPsi` is being quarantined (D5), not run.
* **Does NOT falsify:** dead q-channel + live p-channel (unlocks by design) · a flat curve on a family
  whose store was never admissibly written (admissibility exclusion, reported with its reason) ·
  the 2×2 coming out ≤ 0.

## 7. Admissibility (declared before the run)

A cell is admissibly written iff (Head ruling (i), Route-1 convention, `race.WriteRecord`):
`max_site endpoint write loss ≤ 0.05` **and** `λ_min ≥ 0`. Coverage per family is reported
**first-class, before any verdict**. A family at 0/n gets **one bounded escalation**
(`write_steps 300 → 900`) and then ABSTAINS. ⚠ `overload` is quoted **only** at `load1x_shipped`
(478×) — at the base atom budget it was 0/18 admissible including the Gaussian control.
