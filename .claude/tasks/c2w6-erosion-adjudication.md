# c2w6-erosion-adjudication — I2 hypothesis test + gate re-derivation from raw artifacts

**Campaign 2, wave C2W6. Agent:** results-analyst. **No worktree** (reads `main` after the Hub
merges `c2w6-anti-erosion`, or the branch's committed artifacts if spawned pre-merge — the Hub's
spawn line says which). **GATED on `c2w6-anti-erosion-build` landing its artifacts.** Writes
`.claude/outputs/c2w6-erosion-adjudication.md`.

**Read first:** `.claude/outputs/c2w6-anti-erosion/PREREG-AntiErosion.md` (the registered bands —
you adjudicate against THESE, not the engineer's report) · the engineer's report + raw JSONs ·
charter §A20.6 (I2's specification + the Head's hypothesis) · §A21 C2W6 row (the gate).

## The job (advisor-precedent: every decision-grade number re-derived from raw artifacts,
independently of the engineer's tables)

1. **I2 — the Head's hypothesis, adjudicated:** from the per-well telemetry series, compute
   ρ(usefulness, erosion rate) on the partition-OFF arm — both usefulness measures (read-selection
   frequency; leave-one-well-out loss contribution), per seed and pooled, with the gradient-proxy
   as the mechanism check. Verdict against the registered bands (ρ ≥ +0.5 confirms; ρ ≤ −0.3
   refutes; |ρ| < 0.3 no-structure). **State the quotation consequence explicitly:** confirmed ⇒
   the "depth is not feature importance" caveat stays until the fix ships; refuted ⇒ recommend the
   Hub lift it (Hub lifts, not you).
2. **Gate re-derivation:** recompute E1/E2/E3 and the K3/K4 legs from the raw curves; state the
   run-3 verdict in the prereg §4's own vocabulary and whether it matches the engineer's
   `aggregate()` output digit-for-digit. Any mismatch is a finding, reported, never reconciled
   silently.
3. **Designed-decay separation audit:** verify the erosion curves net out the designed decay law
   correctly (a mis-netted decay would fake both flattening and erosion) — recompute one seed's
   curve from the per-well raw series end-to-end.
4. **The P-residual interaction row** (prereg §3 last item): adjudicate it, and if the partition
   starved the write of its one useful gradient (partition-ON below partition-OFF), say so plainly
   — that is the P3 re-price trigger, Head-owned.

---

## ⭐ ADDENDUM 1 (Hub, 2026-08-05, at the C2W6 review) — RE-SPAWN: the gate is now satisfied

**Your first run returned `BLOCKED` correctly** (the build had not landed; you verified liveness and
mtimes, took no recovery action, and fabricated nothing — that was the right call, and your
pre-cell audit of my prereg is accepted in full). **The build has since landed:** branch
`c2w6-anti-erosion`, 21 cell-seeds, per-well telemetry present at
`erosion_*_records.json → records[*].telemetry` and `records[*].i2.wells`. Re-run your §1–§4.

**What changed under you, and the one epistemic consequence you must carry.** Your F1/F2/F3 were
filed 11:39; the engineer's first science cell ran 13:17; but the engineer never saw them (parallel
spoke, no Hub in session between) and filed its own `PREREG-AntiErosion-ADDENDUM-1.md` at 12:09
instead. ⛔ **So F1/F2/F3 are now POST-HOC with respect to the cells and may NOT be applied as if
pre-registered.** Apply them as **reporting-side estimators, labelled post-hoc**, never as a
re-scoring that changes a registered verdict. The Hub has already done the first pass (below);
your job is to check it and go deeper.

**Hub's own re-derivation, for you to check independently (I reproduced every engineer number
digit-for-digit from raw):** E1 per-seed 9.782/0.9035/0.5305 · K3 w4 −0.004853 ± 0.00078 (6.23 SE)
· w40 ON/OFF final-depth 21.53/1.516/1.464 (geo 3.628, 3/3) · ρ(read-sel) −0.2571 ± 0.1512 ·
ρ(LOO) +0.0667 ± 0.1627 · `n_records` 21 · verdict `FAILS_FLATTEN` (both budgets).

**The four questions this wave now turns on:**
1. ⭐⭐ **Your F3, applied to the headline — the Hub's provisional finding, please adjudicate it.**
   The engineer's R1 rests on `final/untrained = 0.708 ± 0.57×` ("does not decay, it recovers").
   That is the **arithmetic** mean of 0.1351 / 1.847 / 0.1404 — the **geometric** mean is
   **0.327×**, i.e. 2 of 3 seeds lost ~7× depth and one gained 1.85×. On the *registered* E1
   statistic (final/step-200) the refutation is robust under both estimators (arith 3.739, geo
   1.674), so **the gate verdict does not move** — but the narrative "there is no erosion at the
   run-2 config" does. Rule on the defensible claim form and the never-quote wording.
2. **I2 at n_wells = 5–6.** Is `NO_USAGE_STRUCTURE` a null or an *underpowered* reading? The two
   proxies disagree in sign on the primary arm and the w40 per-seed ρ spans +0.872 to −0.771.
   Your own pre-registered call was |ρ| < 0.3 — state whether the data can distinguish your
   prediction from "unmeasurable," and price what rig change would (it needs registering first).
3. **The designed-decay separation audit** (your original §3) on the real curves — a mis-netted
   decay law fakes both flattening and erosion, and the whole E1/E2 story depends on it.
4. **The P-residual interaction row under your F2 repair:** the engineer scored it "2/3 ≥ banked"
   = met, using the pooled-mean reference you flagged as defective. Re-score it as a **paired
   per-seed** comparison at matched horizon and say whether the verdict survives your own fix.

## Constraints
Multi-seed statistics only (paired, ddof=1, SE across seeds) · no new runs without a dated prereg
addendum (you are an adjudicator; if a diagnostic re-run is genuinely needed, ≤ 30 min total,
declared) · every caveat rides (monitor #13/N94 on w4 cells; toy scale; depth-not-importance until
your own §1 verdict) · ⛔ no edits outside your report file · ⛔ never push anything.
