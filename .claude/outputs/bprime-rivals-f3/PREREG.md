# PREREG — `bprime-rivals-f3` (C2W4 rider, experiment-engineer)

**Filed 2026-08-01, BEFORE any full-F3 grid point was run.** Protocol §5 pre-registration rule.
Base local `main @ 21a6dc4`, branch `agent/experiment-engineer/bprime-rivals-f3`, worktree `../CHLU-f3`.
Governing documents: `.claude/tasks/bprime-rivals-f3.md`; `.claude/outputs/bprime-rivals.md` (§1.1, §3, §4,
§9 — the incumbent numbers I am testing); `.claude/outputs/bprime-rivals/PREREG.md` (the registered bands);
`.claude/outputs/rival-recon.md` §F3 + standing rule 5 (the grid).

⭐ **The only measured inputs used to derive the predictions below are (a) the C2W4 audit table and
(b) the C2W4 artefact's *fit-split loss surface*** (`exp_bprime_rivals_metrics.json`,
`cells[*].rivals[*].fit.grid`) — i.e. the incumbent run's own tuning surface, which is the legitimate and
the *only* honest basis for predicting what widening that grid does. **No point of the widened grid had
been evaluated when this file was written.**

---

## 0. DIAL DECLARATION (echoed)
- **Dial / pillar:** **none — TIER-i instrument hardening.** No new claim. Re-scoring an existing audit
  under the program's own standing tuning rule (N78 / `rival-recon` F3).
- **Laundering control:** unchanged and inherited — the full C2W4 audit column set per cell (matched-byte
  projected table launder · +0 B reader set · raw-metric table at the same bytes · same-keys null · blank
  store · two-sided byte ledger · identical φ, enforced in code).
- **Falsifies:** nothing of mine. **The job is to give the C2W4 numbers the chance to be falsified.**
  Pre-registered consequence (Head, 2026-08-01): **any outcome that changes, changes the paper.**
- **Does NOT falsify:** an arm improving and still losing to the raw +0 B table (that is the
  metric-native-ceiling theorem); an arm staying at its blank floor (NOT-RESCUED is a verdict, not a null).

---

## 1. THE INCUMBENT NUMBERS I AM TESTING (C2W4, `aggregate@base`, 3 seeds, mean ± SE)

Reproduced verbatim from `.claude/outputs/bprime-rivals.md` §1.1/§4 and the artefact's `audit_table`.
**These are the priors. Every one of them is on the table.**

| arm | `d_head` | rescue status | `full` | dividend vs own **arg-min** table | **+0 B margin (R5)** | **raw-table margin** | blank | lift over blank |
|---|---|---|---|---|---|---|---|---|
| ttt_linear | 29 | ✅ RESCUED | −0.4546 ± 0.0312 | −0.0302 ± 0.0118 | **−0.0523 ± 0.0165** | **−0.2465 ± 0.0371** | −0.8426 | +0.3879 ± 0.0869 |
| ttt_mlp | 12 | ⛔ NOT | −0.6324 ± 0.2036 | −0.2216 ± 0.2033 | **−0.2284 ± 0.1999** | **−0.4242 ± 0.2114** | −0.6031 | −0.0293 ± 0.1090 |
| deltanet | 36 | ⛔ NOT | −0.4652 ± 0.0402 | +0.2006 ± 0.0576 | **−0.0047 ± 0.0549** | **−0.2571 ± 0.0356** | −0.5657 | +0.1004 ± 0.1296 |
| gdn | 36 | ✅ RESCUED | −0.3961 ± 0.0208 | +1.0197 ± 0.0593 | **+0.0448 ± 0.0591** | **−0.1880 ± 0.0203** | −1.3220 | +0.9259 ± 0.2387 |
| gdn2 | 36 | ✅ RESCUED | −0.3964 ± 0.0220 | +0.8771 ± 0.3006 | **+0.0445 ± 0.0613** | **−0.1883 ± 0.0227** | −1.6618 | +1.2654 ± 0.4968 |

Derived incumbent facts also on trial: **R5 count = 3 of 5 ≤ 0** (ttt_linear, ttt_mlp, deltanet);
**R5-raw count = 5 of 5 ≤ 0**; **P5-vs-raw gap** (the §4 finding) per arm = **0.216 / 0.203 / 0.458 /
1.208 / 1.065**; **rescued set = {ttt_linear, gdn, gdn2}**.

---

## 2. ⭐ WHAT COUNTS AS "AN OUTCOME CHANGED" (pre-committed thresholds; adjudication is mechanical)

An outcome is **CHANGED** iff any of these fires, on any arm:

- **T1 — rescue flip.** An arm's `RESCUED_above_own_blank_2se` verdict differs from §1 (either direction).
- **T2 — R5 sign flip.** The signed +0 B margin crosses 0 on any arm **and** the crossing exceeds
  **2 SE of the new estimate** (so a −0.0047 ± 0.0549 arm wobbling to +0.004 is *not* a change).
- **T3 — raw-table margin crosses 0 by > 2 SE.** i.e. any arm becomes a genuine winner over a raw table
  holding the same bytes. **This is the one that would rewrite B′'s headline sentence.**
- **T4 — the R5 count changes.** "3 of 5 ≤ 0" becomes anything else (again, counting only crossings that
  exceed 2 SE, so ties are declared ties).
- **T5 — the P5-vs-raw gap collapses.** On any arm the projected-vs-raw control gap (§1's 0.203–1.208)
  falls below 2 SE of 0 — which would demote the §4 methodological finding.

Anything else (a `full` moving inside its SE, a launder moving, a `d_head` unchanged) is **UNCHANGED**.
⛔ I commit now to reporting a CHANGED verdict as a **finding**, not as a nuisance, and to naming the
paper claim it invalidates in the report's first screen.

---

## 3. ⭐ THE PREDICTIONS, WITH DERIVATIONS

### 3.0 The mechanism I am predicting from (the incumbent fit-split surface)
Best-of-grid is selected on the **fit split** = the auxiliary streams the outer parameters are optimised
on (F2a: never the eval stream). The C2W4 artefact contains that surface for all 15 (arm × seed) cells.
Two facts, read off it and stated before running:

1. **The incumbent arg-min is never at the grid's low-lr edge.** In **15 of 15** (arm, seed) cells the
   selected lr is **3.16e-3 or 1e-2**, never 1e-3, and the fit loss at 1e-3 is ≥ the winner's in 15/15.
   Examples: `ttt_linear@s0/b16` 0.2351 → 0.2037 → 0.1917 as lr goes 1e-3 → 3.16e-3 → 1e-2;
   `ttt_mlp@s0/b16` 0.1420 → 0.1142 → 0.1080; `gdn@s0` 0.23756 → 0.23665 → 0.23710.
2. **The delta arms' surface is flat to the 4th decimal** (`gdn@s0` spans 0.00091 across the whole lr
   grid; `gdn2@s1` spans 0.00028). Their outer loop is *not* lr-limited; whatever they are doing, it is
   what they do at every lr in the range.

### 3.1 P1 — the widened **lr** axis selects nothing new. **Predicted: 0 of 15 cells select a new lr.**
The three new lrs (1e-4, 3.16e-4, 5e-4) are **all below** the incumbent grid's *worst* point (1e-3) on a
surface that is monotone-improving with lr toward 1e-2 in 13 of 15 cells (the 2 exceptions are interior
minima at 3.16e-3, still above 1e-3). At a fixed 400 Adam steps, an lr 10–100× smaller travels
correspondingly less from init. ⛔ Note the F3 grid's *upper* edge is 1e-2 — the same as C2W4's — so the
widening is **entirely on the side the surface says is worse.** This is the single most important
derivation in this file and it is why I predict UNCHANGED.

### 3.2 P2 — the **wd** axis is near-inert **under this harness's selection rule**.
**Predicted: wd = 0.1 selected in ≤ 2 of 15 cells; where selected, Δfit-loss < 0.005.**
Decoupled decay (AdamW) adds a shrinkage that is not a descent direction for the objective; the selection
criterion here **is** that objective on the fit split. A regulariser essentially never lowers the loss it
is not optimising. At lr = 1e-2, wd = 0.1, 400 steps the unopposed shrink factor is
`(1 − 1e-3)^400 ≈ 0.67` — large enough to *hurt* visibly if it does anything at all.
⭐ **Corollary I register as a finding-in-waiting:** if P2 holds, then *F3's 6×2 grid is operationally a
6×1 grid on this harness*, and the honest fix is a **held-out fit-validation stream** for selection. I
therefore run the same grid a second way — selection on a held-out auxiliary stream (seed+103), never the
eval stream — and report it as a declared **secondary** column. **Predicted:** the val-selected column
picks wd = 0.1 in **≥ 3 of 15** cells and still changes **no** §2 threshold.

### 3.3 P3 — per-arm predicted deltas on the eval metric (vs §1), full-F3 grid, 400 steps
| arm | predicted Δ`full` | predicted Δ raw-table margin | reasoning |
|---|---|---|---|
| deltanet / gdn / gdn2 | **\|Δ\| < 0.010** | **\|Δ\| < 0.010** | fit surface flat to 1e-3 (§3.0.2); the selected point cannot move far |
| ttt_linear | **\|Δ\| < 0.050** (< 2 SE) | **\|Δ\| < 0.050** | the winner (lr 1e-2) is already at the grid edge that survives |
| ttt_mlp | **\|Δ\| < 0.250** (≈ 1 SE of its own 0.2036) | **\|Δ\| < 0.250** | the only arm whose C2W4 seed scatter is larger than any plausible tuning effect |

### 3.4 P4 — rescue statuses. **Predicted: all five UNCHANGED** ({ttt_linear, gdn, gdn2} rescued).
⚠ **The fragile one, named in advance: `ttt_mlp`** (lift −0.0293 ± 0.1090 — it sits *at* its blank floor
with an SE 3.7× its lift). If it flips to RESCUED I pre-commit to checking my **F3-lite control column**
(§4) before crediting the flip to tuning: a flip that also appears in the control is an **init redraw**,
not a rescue. `deltanet` (+0.1004 ± 0.1296) is the second-most fragile by the same test.

### 3.5 P5 — the 2000-step re-check rescues nothing. **Predicted: no §2 threshold fires at 5× budget.**
Derived from C2W4 §3's own measurement on the frontier family: at 2000 steps the *fit* MAE reached 0.024
while eval MAE went to 0.75 (**31×**), and decode fell (gdn2 0.0417 → 0.0000). The failure is a
**generalisation gap across item geometries**, forced by F2a, not an optimisation gap. **Predicted at
`aggregate`:** fit-split loss drops ≥ 20% at 2000 steps on the TTT arms, `full` changes by < 1 SE on
every arm, and the raw-table margin stays negative on **5 of 5**.

### 3.6 P6 — the §4 methodological finding survives. **Predicted: T5 does not fire on any arm**;
the P5-vs-raw gap stays ≥ 0.15 on all five arms and ≥ 0.9 on `gdn`/`gdn2`. The gap is a property of
*where the table is read* (projected vs raw space), not of how well the projections were tuned — and
tuning `θ_K, θ_V` for the recurrence is exactly what *makes* the projected space a worse metric, so a
better-tuned arm should if anything widen it.

### 3.7 ⭐ The honest alternative I am registering AGAINST myself
The task asks me to register the possibility that the low-lr half rescues nothing **because the 31×
fit→eval gap is geometric, not optimisation-limited** — that is precisely P1+P5 above, and it is my
primary prediction. **The alternative that would make me wrong:** the delta arms' flat fit surface is flat
because 3.16e-3 is already *too large* for them (they are bouncing, not converged), in which case
lr ∈ {1e-4 … 5e-4} would find a genuinely lower fit loss and a materially better `full`. **If that
happens, T3 is live and I will say the paper's headline changes.** I put its prior at **≤ 15%**: a
bouncing optimiser at 3 lrs spanning 10× would show *scatter* across seeds at fixed lr, and the seed-wise
surfaces are instead smooth and ordered.

---

## 4. DECLARED PROTOCOL DEVIATIONS (one variable was supposed to move; here is the full list)
1. ⭐ **Init-key scheme changed, unavoidably.** C2W4 drew each grid point's init from a *sequential*
   `jax.random.split`, so the init depended on the grid's *length and order*. Widening the grid therefore
   re-draws every model even at an unchanged (lr, b). I change the scheme to **one init per
   (arm, seed, mini-batch b), shared across all (lr, wd)** — standard tuning practice and the only way the
   grid surface is an lr/wd surface rather than an lr/wd/init surface. ⛔ Consequence: the F3 column is
   **not** bit-comparable to C2W4. **Priced, not hidden:** I report an **F3-LITE CONTROL column** —
   the C2W4 sub-grid (lr ∈ {1e-3, 3.16e-3, 1e-2}, wd = 0) selected under the *new* key scheme, from the
   same fits at zero extra cost. **`control − C2W4` = the init-redraw effect; `F3 − control` = the
   tuning effect.** Every CHANGED verdict is adjudicated on `F3 − control`.
2. **A held-out fit-validation stream** (seed + 103, one stream) is *computed* for every grid point and
   used **only** for the declared secondary selection (§3.2). ⛔ The training data of the outer loop is
   **byte-identical** to C2W4 (2 streams, seeds +101/+102, same trim), and the eval stream is untouched.
3. **F3's optimiser sub-clauses are NOT adopted:** `rival-recon` F3 also names AdamW β=(0.9, 0.98) and
   cosine decay. I keep C2W4's β=(0.9, 0.999) and a constant lr, because the task's dial declaration says
   **one variable moves** and because it keeps the F3-lite control meaningful. ⛔ Declared as a deviation,
   not claimed as compliance. `wd = 0.1` is implemented with `optax.adamw` (decoupled); `wd = 0` keeps
   `optax.adam` so the control column has *no* optimiser change at all.
4. **Family:** `aggregate@base` only. The frontier column is NOT re-run (task §1; it was declared
   non-informative), unless an arm is newly rescued at `aggregate`.
5. Seeds 0, 1, 2 — identical to C2W4. Iso-state budget, head widths, φ, scorer, rescue gate: unchanged.
