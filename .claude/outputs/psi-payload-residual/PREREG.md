# PREREG — `psi-payload-residual` (charter §A20.3(a); protocol §5 pre-registration rule)

**Filed 2026-08-01, BEFORE the harness that measures any of it was run.** The acceptance criterion is
a measured *ratio*, so this file is mandatory. Predictions are stated with the derivation that
produced them; a prediction that survives is evidence, one that fails is a finding.

Branch `agent/experiment-engineer/psi-payload-residual`, worktree `../CHLU-psires`, base local
`main @ 9b2d4db`. Code under test: `StreamMemoryConfig.psi_payload_residual` +
`CluStoreCell.payload_residual` (additive hunks in `chlu/core/blocks.py`), instrument
`chlu/experiments/exp_psi_residual.py`.

## 0. The spread convention (fixed before measuring)

`spread` = **sample sd, ddof = 1**, over the live items of one lane; pooled across lanes as an
unweighted mean of the per-lane sds, ±SE across lanes, then meaned across 3 paired seeds.
⭐ The convention is chosen to reproduce the task's own "current 0.04–0.15" column: recomputing
`pilot-placement-probe` §6's lane-0/seed-0 numbers as sds gives **0.131 / 0.048 / 0.156**
(baseline / h1b_r0.3 / h1b_m1.0), which is that band. The `range` column is also recorded but is not
the adjudicated statistic.

## 1. The design being tested (what "pass-through" means arithmetically)

`decode = psi(traj, state) + g · q*[payload]`, gate `g` initialised at 1.0, payload coordinates only.
Because the residual is **additive and linear in `g`**, `decode(g) = decode(0) + g·source` exactly,
and the per-stage ledger + the whole gate sweep are derivable from three reads per item. That
linearity is itself asserted at runtime (`_LINEARITY_TOL = 1e-5`); if the assertion fires, every
derived number in the report is void.

## 2. Predictions

### P1 — the acceptance ratio at pass-through (the §A20.3(a) bar)
**Predicted: ratio `decoded / q*` = 0.95–1.10 on all cells; bar (≥ 0.5) MET on 3/3.**
*Derivation:* `sd(psi_only + q*) ≈ sqrt(sd(q*)² + sd(psi_only)² + 2·cov)`. Measured (probe §6):
`sd(psi_only)` = 0.0028–0.0048 and `sd(q*)` = 0.027–0.058, i.e. `psi_only` is 5–17 % of `q*`, so
the sum's sd is within ~2 % of `sd(q*)` unless the two are strongly anti-correlated; §6 measured
`psi_only` anti-correlated with the TRUTH (r = −0.89…−0.99) and `q*` only weakly correlated with it
(−0.60/−0.54/+0.75), so the cross term is small and can go either way. Band [0.85, 1.20].
⛔ **Declared in advance: at `g = 1` this bar is close to being met BY CONSTRUCTION.** Passing it is
therefore *not* the finding; it is the precondition. The findings are P3–P6.

### P2 — the blank-store leak check (BLOCKING; laundering control)
**Predicted: blank-store decoded spread ≤ 0.006 (≤ 5 % of the live decoded spread), blank
acquisition exactly at chance, `acq(live) − acq(blank)` = 0.000 on every cell × seed.**
*Derivation:* the read launches on the payload-zero manifold (`q0[a:a+m] = 0` by construction), so
the residual's source is 0 at launch; §6 measured blank `q*[payload]` = **9.2e-4 / 1.8e-4 / 1.7e-4**
against a live 0.18–1.14, i.e. 3 orders down. A payload-restricted residual therefore cannot carry
`phi(x)`. ⛔ If blank acquisition rises off chance, the build is N68's leak in a new coat and is
**withdrawn**, whatever P1 says.

### P3 — acquisition at pass-through: **STILL EXACTLY AT CHANCE** (the second gap)
**Predicted: `acq(g=1)` = chance to the digit on 3/3 cells, and `acq − blank` = 0.000.**
*Derivation — this is the report's central pre-registered claim, and it is a SCALE argument, not a
spread one.* Acquisition is nearest-**stored**-payload assignment (N110). With the residual at
`g = 1` the decode is ≈ `q*[payload]` = 30–50 % of the true value (probe §6.2). On §6's own lane-0
items that is decoded −0.41 / −0.30 / −0.37 against true payloads −0.622 / −0.779 / −0.850: **all
three decodes are nearest to the smallest-magnitude stored payload (−0.622)**, so the assignment is
constant, giving exactly one hit ⇒ `acq = 1/n = chance`, identically — the same arithmetic that
produced the probe's exact-chance result, now caused by a *uniform under-shoot* instead of by a
collapsed spread. ⭐ **Restoring the spread is necessary and not sufficient; the residual exposes a
SECOND deficit — the delivered payload's SCALE.**

### P4 — the gate sweep finds the scale, and acquisition comes off chance
**Predicted: there exists `g*` in [2, 4] with `acq(g*) ≥ chance + 2 SE` on ≥ 1 of the 3 §6 cells;
best pooled `acq` in 0.35–0.60 against a chance of ~0.26.**
*Derivation:* the scale gap is 1/0.30–1/0.48 ≈ **2.1–3.3×**; at `g ≈ 1/frac` the decode lands on the
true payload scale and the rank information in `q*` (the only information there is) becomes
resolvable. On §6's three items `g = 2.5` gives decoded −1.03 / −0.74 / −0.93 ⇒ 2 of 3 hits (0.67).
⚠ `q*`'s correlation with the truth is weak and sign-inconsistent across cells (−0.60/−0.54/+0.75),
so this is predicted to be a **partial, cell-dependent** rescue, not a clean one.
⛔ Falsifier of P4: no gate anywhere in {0…6} clears `chance + 2 SE` on any cell ⇒ the payload's
*identity* information is absent from `q*`, not merely mis-scaled, and the next bottleneck is the
WRITE (which payload the dynamics deliver), not the read-out. That would be a finding, and it names
§A20.3(b)'s territory as the owner.

### P5 — the trained tier: does the residual survive 200 outer steps?
**Predicted: the learned gate stays within [0.3, 3.0] of its 1.0 init (median across seeds/layers
1.0 ± 0.4) and the acceptance ratio stays ≥ 0.5.**
*Derivation:* the outer signal is byte-LM cross-entropy at 4.6 bpc after 409 600 tokens (probe §7),
i.e. the model is barely past unigram statistics; the measured live-vs-blank contribution to that
loss is ~1e-4 bpc, so the gradient into a single scalar gate is tiny and the gate should barely
move. ⛔ Falsifier: the gate trains toward 0 (|g| < 0.1) ⇒ the outer loss *prefers* the collapsed
decode and the residual is not merely useless but actively penalised — a hard finding for the
read-fix programme.

### P6 — the trained tier: depth, and R3's well-destruction
**Predicted: with the residual on, well depth after 200 steps at the run-1 config lands in
0.02–0.15 (the probe's placement+margin value was 0.0616 ± 0.037), i.e. the residual does NOT
rescue and does NOT worsen R3.** *Derivation:* the residual is read-side only and adds one scalar
per layer; R3's destruction is upstream (the write/φ path). Band [0.005, 0.30] — outside it, the
read-out lever has an unexplained upstream effect and must be reported as such.

### P7 — bpc
**Predicted: |bpc(live) − bpc(blank)| stays ≤ 1e-3 at `g = 1` untrained, i.e. the residual does not
manufacture a held-out win.** The live-vs-blank gap should *grow* relative to the probe's
+2.8e-05…+7.1e-04 (the store's content now reaches the output more directly) — predicted 1.5–10×
larger, same sign. ⛔ A bpc gap that grows with the BLANK store is a leak (see P2).

## 3. Declared NOT-RUNs (never reported as nulls)
- Any 26–47 M / CSF3 number. No cluster route from this machine.
- `AttentionPsi` — QUARANTINED, not routed through, not measured.
- The 5-arm swap table per cell (GRU/TTT/echo): the residual is a read-out lever inside the CLU arm;
  the swap protocol is the pilot's and is not re-run per cell here.
- The `traj_mean` source as a *shipped* configuration: it is measured as a ledger stage (it is free
  from the same three reads) but the run-2 flag block will name one source, not two, unless the
  measurement says otherwise.
- Any figure/plot: the deliverable is a ledger table.

## 4. The cut order, if wall clock runs short
1. `run1_w40` (the N94-floor cell) — cut LAST, it is the only non-demoted reading.
2. The trained tier's `residual_off` control — never cut (it is the paired control).
3. `h1b_m1.0` — cut FIRST if needed (its own §6 row already says the margin collapses `q*` spread).

---

# ADDENDUM 1 (2026-08-01, filed BEFORE any cell of the harness ran) — the "before" column's arithmetic

⛔ **§0 above mis-states the probe's before-column, and the correction is filed rather than edited in
(the `pilot-placement-probe` addendum precedent).** §0 quoted 0.131 / 0.048 / 0.156 by dividing
`decode_dispersion.json`'s `decoded_std` (numpy default, **ddof = 0**) by a `q*` sd recomputed at
**ddof = 1**. Recomputed consistently from the same raw artifacts (either ddof; the ratio is
ddof-invariant), §6's lane-0/seed-0 before-column is:

| cell | sd(decoded) | sd(`q*`) | **ratio** | compression |
|---|---|---|---|---|
| `baseline` | 0.004242 | 0.026517 | **0.1600** | 6.25× |
| `h1b_r0.3` | 0.003374 | 0.057845 | **0.0583** | 17.14× |
| `h1b_m1.0` | 0.005879 | 0.030788 | **0.1909** | 5.24× |

So the task file's *"the current 0.04–0.15"* is the **reciprocal of the report's 7–25× compression
band** (1/25 = 0.04, 1/7 = 0.14), not a per-cell ratio; the per-cell before-column is
**0.058–0.191**, i.e. one cell (`h1b_m1.0`) already sits above the top of the quoted band. ⭐ This
does not move the acceptance bar (≥ 0.5), and it does not move any prediction P1–P7 — it makes the
"before" column honest, and it is the number `PROBE_BEFORE_RATIO` now carries.

Two derived quantities are also pinned here so they are pre-registered rather than chosen later:
`q*`/true spread ratio **0.228 / 0.497 / 0.264** and median |`q*`|/|true| **0.249 / 0.436 / 1.399**
(the last is the margin-1.0 overshoot). P4's `g*` band [2, 4] was derived from those fractions and is
unchanged.
