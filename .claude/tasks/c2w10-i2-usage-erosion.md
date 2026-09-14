# c2w10-i2-usage-erosion — the I2 re-measurement, at the only live-well count that can carry it

**Campaign 2, C2W10 ("The persistent store"). Agent:** results-analyst. **ZERO worktrees.**
Banked artifacts only — you run no store, you train nothing, you touch no tracked code.
Writes `.claude/outputs/c2w10-i2-usage-erosion.md` + artifacts to `.claude/outputs/c2w10-i2/`.
**Budget:** ≈ half a day.

## ⛔⛔ MECHANICAL PRECONDITION — check this FIRST and refuse if it is not met
This spoke may not run until **`.claude/outputs/c2w10-lifecycle/USAGE-TELEMETRY.json`** exists **and**
contains `n_live_max >= 64` **and** `n_seeds >= 3`.
**If the file is absent, or either condition fails: write your report with status `BLOCKED`, name the
missing condition and its measured value, and stop.** ⛔ Do not substitute another rig, do not lower
the threshold, do not "make do" with the C2W8 census (16 items — that is precisely the power failure
Add.7 ruling 5 deferred this measurement to escape). A `BLOCKED` return here is the correct, expected
and previously-rewarded outcome (C2W6's analyst returned `BLOCKED` and was right).

**Binding documents, read first, in this order:**
- `.claude/outputs/c2w10-persistent-store/PREREG-C2W10.md` **§5 IN FULL** — legs I2-a…I2-d and **the
  LIFT RULE**. It is registered; you apply it and you compute its booleans. ⛔ **You do not lift the
  caveat — only the Hub lifts it.** Your job is to report which branch fired.
- charter **§A20.6** (I2 as filed: the Head's hypothesis that under a net-cost store the MOST useful
  wells erode FASTEST), **Add.7 §A23.5** (`NO_USAGE_STRUCTURE` accepted, re-measurement deferred to
  THIS wave, caveat REMAINS ACTIVE), **Add.9 §A27.1** (⛔ `ρ(LOO) = +0.067` is **UNDEFINED, not a
  null** — ICC(1,1) negative 3/3 ⇒ attenuation ceiling 0.000; and the pooled-ρ column that **flips
  sign** is a non-registered estimator, relabelled, never evidence), **§A28.3(ii)** (depth ≠
  usefulness; the optimizer's erosion is churn, not curation).
- `.claude/outputs/c2w8-well-lifecycle.md` §3.4 (the `U` instrument as built: registered primary proxy
  `read_hits(i)`, item-id keyed, surviving eviction; LOO reported only beside ICC).

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** none — **instrument re-measurement.** ⛔ No performance claim, no CLU verdict, no cell run.
- **Laundering control:** N/A. The controls here are statistical: ICC(1,1), the power precondition,
  and the pre-registered two-sided lift rule.
- **Falsifies:** I2-a or I2-b failing ⇒ the test is UNDERPOWERED/UNDEFINED and is a declared NOT-RUN.
- **Does NOT falsify:** an INDETERMINATE result — that is the registered modal outcome (Hub prior
  0.60) and it is a result, not a shortfall.
- ⛔ Depth is not feature importance (§A23.5 ACTIVE — and this spoke is the only venue that can change
  that, in one direction, by one registered rule, adjudicated by the Hub).

---

## What to compute

Population: **live wells** at the declared measurement points in `USAGE-TELEMETRY.json`, per seed.
- **`U_i` = item-id-keyed `read_hits`** — the registered primary proxy. ⛔ **Depth never enters `U`.**
- **`E_i` = per-well erosion rate on the NETTED depth curve.** ⛔ Never the raw curve (Add.9 §A27.1:
  un-netted curves overstate recovery by up to 34 %, and the decay exponent drifts with
  `last_write_chunk`). State the estimator you use in closed form, on the **log** scale (C2W6's
  estimator correction: geometric/log-scale depth, and censored readings are **censoring statements**,
  never point estimates — "4.95e-63" is the standing example).
- Report the **whole curve**, not endpoints — C2W6's E1 erratum happened because an endpoint statistic
  could not see a 112× transient trough whose recovery it then measured instead of the decay.

| leg | what you report |
|---|---|
| **I2-a POWER** | `n_live` per seed at the measurement point; the Fisher-z 2-SE half-width `2/√(n−3)`; **the smallest honestly detectable \|ρ\| quoted beside every ρ** (≈ 0.25 at n = 64) |
| **I2-b ICC** | `ICC(1,1)` of the usage proxy, per seed. ⛔ **ICC ≤ 0 ⇒ the LOO leg is `UNDEFINED` and no ρ from it is quotable** |
| **I2-c EROSION** | `ρ(U, E)` per seed with 2-SE bounds. **CONFIRM branch:** ρ ≤ −0.2 beyond 2 SE on ≥ 2/3 seeds ⇒ useful wells erode fastest ⇒ the caveat HARDENS |
| **I2-d DEPTH-USAGE** | `ρ(U, depth)` per seed on the netted curve, with 2-SE bounds |

## The registered LIFT RULE — compute its booleans, do not interpret them
The caveat lifts **only if BOTH**, on **3/3 seeds**, with I2-a and I2-b satisfied:
1. `ρ(U, E)`'s 2-SE **lower** bound is above **−0.10** (the anti-correlation is refuted); **and**
2. `ρ(U, depth) ≥ +0.30` beyond 2 SE (positive evidence, not merely absence of the negative).
Anything else ⇒ **INDETERMINATE ⇒ the caveat stays ACTIVE**, reported as a result.
⛔⛔ **AUTHORITY (Head ruling 3, 2026-08-10):** the caveat lives in charter **§A23.5, an ADVISOR-OWNED
document ratified by the Head** ⇒ **the Hub measures and PROPOSES; the ADVISOR makes the amendment.**
Neither you nor the Hub edits §A23.5. **Two conditions carried from C2W6:** ⭐ **only a
POSITIVE-STRUCTURE finding lifts anything — a second `NO_USAGE_STRUCTURE` leaves the caveat exactly
where it is** (which is why the rule needs I2-d's positive leg and not merely I2-c's refutation); and
⛔ **any ICC ≤ 0 reading is `UNDEFINED`, not a null, and is never quotable as a measured correlation.**

Emit `.claude/outputs/c2w10-i2/I2-VERDICT.json` with:
`n_live_by_seed, icc_by_seed, rho_U_E_by_seed(+2SE), rho_U_depth_by_seed(+2SE), detectable_rho,
i2a_pass, i2b_pass, branch ∈ {CONFIRM, REFUTE_BOTH_LEGS, INDETERMINATE, NOT_RUN}, lift_rule_satisfied (bool)`
— every field computed arithmetically. ⛔ `lift_rule_satisfied = true` is a **measurement, not a lift**;
your report states in one line: *"the Hub proposes; the Advisor amends §A23.5; this spoke does neither."*

## Riders (cheap, and they close standing debts)
1. **Estimator hygiene:** any pooled-across-seeds correlation is a **non-registered estimator** — if
   you report one, label it as such and report the registered per-seed form beside it. C2W6's pooled-ρ
   **flipped sign** against the registered estimator.
2. **Report the censoring fraction** (wells whose netted depth hits the floor) — a correlation computed
   over a censored population needs its censoring fraction quoted with it.
3. If the branch is CONFIRM, state explicitly what it implies for the lifecycle: **depth remains
   unusable as the usefulness criterion**, which is why L3's trash criterion is keyed on `read_hits`
   and not on depth — that design choice is then vindicated by measurement rather than by assumption.

## FILE OWNERSHIP (declared)
**You own:** `.claude/outputs/c2w10-i2/**`. ⛔ **You touch NO tracked code and no other spoke's
outputs.** You read `USAGE-TELEMETRY.json` and the banked C2W6/C2W8 artifacts read-only.

## Acceptance (mechanical)
1. The precondition was checked **first** and its measured values are quoted in the report.
2. `I2-VERDICT.json` exists with every field above, computed arithmetically.
3. ICC beside every LOO number; `UNDEFINED` labelled where ICC ≤ 0; the detectable-\|ρ\| quoted beside
   every ρ; erosion computed on the **netted** curve, on the log scale, with the censoring fraction.
4. The lift rule's two legs reported separately, and the one-line statement that the Hub lifts.
5. Reconciliation list in the **first 10 lines**; NOT-RUNs declared as NOT-RUNs, never nulls.
