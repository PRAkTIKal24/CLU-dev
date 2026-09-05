# c2w7-read-cardinality — "Finish the read" (the cardinality iteration; charter §A21 C2W7 row)

**Campaign 2, wave C2W7. Agent:** experiment-engineer. **Worktree 2 of ≤3** (wt1 is C2W6
anti-erosion's — scoped concurrently, live; read its 2026-08-05 §10 coordination note. CSF3
occupies zero). Branch **`c2w7-read-cardinality`**. Writes
`.claude/outputs/c2w7-read-cardinality.md` + artifacts. Binding scope: charter **ADDENDUM 6 §A21's
C2W7 row**, on top of **ADDENDUM 5 §A20.3** and the two Advisor amendments **A1/A2** (recorded in
`.claude/tasks/tierii-read-fix.md`, still binding). Tier-ii control = the **ORGANIZER SWAP** (§A13);
the settle-deleted launder is an inherited diagnostic only. §A14.1 stands: **inference-read claims
are CLOSED** — everything here is training-time organization + read-protocol machinery, and reducing
the read to a table after training is a computational win.

**Your substrate is iteration 1, MERGED on `main @ 104ca19`:** `chlu/core/multiwell_read.py`,
`chlu exp-tierii-read`, `tests/test_multiwell_read.py`. Its verdict (`.claude/outputs/
tierii-read-fix.md`, read IN FULL) is your specification: **addressing is SOLVED** (K0 1.0000,
distinct wells 11.27 settled, S_eff 16.00 in band, settle destruction 4.6 % raw / +0.076 gated) and
**expressivity is NOT** (exact-set 0.0023 vs clear bar 0.02; OD_min a vacuous tie; **G1 FIRES:
read − live launder = −0.0016 ± 0.0019**). The named, measured blocker is **CARDINALITY**: the read
visits ~11.3 wells with no mechanism that commits to exactly `F = 4`; the gated set has 5.79 ± 0.90
members (never 4); the gate throws away correct wells (coverage_gated 0.053); and **π-sharpening
provably cannot supply the commitment** (top-F(π) == A(x) at 0.000 for every β — the ranking is
depth-driven, not query-driven, at `depth_ratio = 3`). The launch's own top-F ranking is currently a
better set estimator than the settled occupancy's.

**Read first (order):** charter §A21 C2W7 row + §A20.3 + §A20.2's re-labelling · `tierii-read-fix.md`
in full (esp. §4.1, §6, §8, §13) · `orgdiv-null-arms.md` §3–§4 (the mechanism + the ceiling) ·
`PREREG-TierII.md` + `ERRATA-TierII.md` (the falsifiers that govern your endgame; the ERRATA's
deviations D1/D2 and the §A20.2 scoping are binding) · the `[C2W7]` §10 scoping entry.

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial:** TIER ii — the organization dividend, **read-cardinality iteration** (iteration 2).
- **Control:** the ORGANIZER SWAP (gated, §Gate below), against `N1′` under the F3-equal tuning
  standard both ways. Until the gate fires, your controls are the four §A20.3(c) guards — above all
  **G1 recomputed LIVE on your OWN learned launches** (amendment A1's strong form).
- **Falsifies:** your re-registered falsifiers (below), each with sign/threshold/seeds filed BEFORE
  the harness runs (w14 rule: numeric predictions committed).
- **Does NOT falsify:** the nulls' failure (not your claim); anything tier i/iii; ψ's marginal value
  (A2 — stays in its own ablation).

## The five deliverables (§A21 C2W7 row, each one measured)

1. **Multiplicity-as-counting-code.** The k particles distribute over wells with **learned
   multiplicity** — the read's answer is no longer a set/occupancy but a **counting code**
   (well j carries multiplicity n_j, Σ n_j = k), and the F-cardinality commitment is **query-driven**:
   the head (not well depth) decides how many particles a well gets. This is the §4.1 fix: the
   information §4.1 proved absent from the settled occupancy must be carried by the launch/multiplicity
   channel. Design freedom is yours (soft counts, learned budget/stopping, dedupe/evolve-unique now
   in its live regime — iteration 1 measured the verb INERT because successive-suppression never let
   particles co-occupy; a multiplicity head is exactly the regime where it goes live).
2. **Overlap-as-importance weighting** in the prediction: per-well contributions weighted by particle
   overlap/multiplicity (confidence), replacing binary occupancy + noisy-or. ⛔ **This changes the
   reader class ⇒ the reader-class RE-REGISTRATION (deliverable 6) must be filed BEFORE any run.**
3. **The batch-level anti-collapse regularizer.** Penalize **MARGINAL well-usage collapse only** —
   the across-inputs marginal of well usage (batch-level launch distribution) collapsing to few wells.
   ⛔ **Per-query concentration is CONFIDENCE and is NEVER penalized** (a confident query putting all
   k in F wells is the design working). Doctrine §3.3 activation order: **monitored first, regularized
   second** — the regularizer ships built and OFF; it turns on only when the monitor (4) fires, and
   both states are reported.
4. **The launch-collapse monitor** — the new anti-collapse row: all-k-in-one-well **marginally**
   (across the batch) = the codebook-collapse mode of a learned launch head. Statistic: your choice
   (marginal S_eff of launches / batch usage perplexity), with a band and a demonstrated designed
   negative (N74: a guard that cannot fire is vacuous). The row lands in `chlu/core/monitors.py`
   (ADDITIVE only) — **C2W6 has ceded `monitors.py` to C2W7 on the §10 record** (its 2026-08-05
   scoping entry, coordination note); C2W6 owns `train_cluformer.py` + `blocks.py`, which you do
   not touch.
5. **Recomputed launch-only launders** (amendment A1, strong form): every claim cell carries
   `read − launder` where the launder is **this cell's OWN learned launches with the landscape
   deleted and the written payload table retained**, scored through the same re-registered reader
   class at the same k. ⛔ 0.272 and 0.695 are stale out-of-class reference ceilings — quote them only
   as labelled reference lines WITH their `(d, draws)` noise model (reconciliation 1 of iteration 1),
   never as bars. Recompute the store-free ceiling at YOUR head's noise model as a reference line.

**(6) Reader-class re-registration — BLOCKING, before any run.** Multiplicity/overlap weighting
changes what a reader consumes (weighted counting code, not a set). File the one-line amendment in
your `PREREG.md` **and** as a dated `AMENDMENT-C2W7` block appended to
`.claude/outputs/orgdiv-prereg/ERRATA-TierII.md` (the prereg itself is NEVER edited). The class stays
capacity-bounded **< N_a·m = 256 params per reader** (SP-1: a ≥256-param reader solves the family
storeless), stays ≥ 4 architectures incl. a non-quantising twin (iteration 1's D8), and is frozen
before the first arm runs. Note on the record: iteration 1 §13.3 asks whether the cap itself is what
zeroes every arm — your re-registration is where that gets addressed *within* the 256 bound, not by
raising it.

## The learned launch head goes live
Iteration 1 scored only the DESIGNED head (`train_launch_head` built + guard-tested, declared
NOT-RUN as an arm). Your multiplicity head is LEARNED — so all four §A20.3(c) guards apply live:
- **G1** launch-only launder recomputed live (deliverable 5) — ⭐ this is the wave's decisive number:
  iteration 1's G1 fired, meaning the launches carried everything. **Your store must add value over
  your own learned launches or the vehicle fails again regardless of cardinality.**
- **G2** soft-occupancy/soft-count training signal (hard assignments don't backprop; ratio test).
- **G3** staged store-then-launch co-training (w20; blank-store head gradient is 1 300× weaker —
  ordering stays mandatory; the designed init from iteration 1 §8 is your init).
- **G4** k on the byte ledger, every arm matched (k = 12 registered; k ∈ {16, 24} declared NOT-RUN
  unless you argue and register otherwise). Learned-head params on the ledger, all arms.
- **Learned p₀ = reach lever only** (§A14.1): keep the confidence-gated ballistic kick as a
  registered lever (it buys coverage ×12; it does not buy selectivity) and ablate it once.
- **K0 re-adjudicated store-free FIRST** for the learned head (standing rule): P(≥F distinct wells
  reachable) ≥ 0.90 at your registered cell before anything is built on it, and re-checked after
  training (a trained head can collapse K0 — that is exactly what monitor (4) watches).

## Protocol constraints carried (iteration 1's cell, unchanged unless argued)
`d = 8` (D6 stands; ceiling 0.695) · one-draw query noise `σ_q = 0.15` (D7) · `m = 8`, `a = 32`
(measured constraints) · `payload_radius = 0.5` (basin-reach constraint, §7 known issue) · measured
`s` confinement-subtracted (§7.28; expect ≈ 0.288, re-measure), `d/s ∈ [2.5, 2.9]` · γ_address 0.05 /
γ_read 0.02, read budget 400+800, every γ statement budget-scoped · 5 seeds before any number,
`n_unseen ≥ 256`, rule-4-valid splits (K2 at m=8) · K1/K3/K4/K5 pre-conditions re-run on your cell
before the arms (kill-conditions first; build the kill-condition before the thing it can kill) ·
⛔ `@jax.checkpoint` on every settle body you differentiate through (the silent exit-0 OOM,
program-wide hazard) · ⛔ no byte-matched tier-ii promise (ERRATA E-T3) · §2.6 claim form (no well
named semantically) · never-quotes carried: "K0 repaired / S_eff in band" never without R1 + G1 in
the same paragraph.

**ψ pinning (amendment A2):** your PREREG declares WHICH ψ every cell uses — shipped or
payload-residual (now merged on main) — with a one-line argument, **uniform across ALL arms, no
mid-task hot-swap**. ψ's marginal value is NOT measured here (it belongs to the CSF3 run-1/run-2
ablation).

**depth_ratio (Hub pre-registration, flagged to the Head):** the claim cell KEEPS the registered
≥3× depth heterogeneity (it exists so F5 is falsifiable — a binding prereg falsifier does not move
mid-iteration). `depth_ratio = 1` runs as a **registered diagnostic axis, never a claim cell**
(iteration 1 measured it raising reach/coverage and lowering gated precision — your multiplicity
head must beat the depth-driven ranking WITH heterogeneity present). If the Head rules otherwise at
spawn, that ruling supersedes this paragraph.

## ⚖ THE GATE (pre-registered here; the wave's one decision point)
**Expressivity guards, adjudicated mechanically on your claim cell (5 seeds):**
- **R1** exact-set accuracy clears its bar: `mean − 2 SE > 0.02`;
- **G1** the read beats its own live launder: `read − launder` mean − 2 SE > 0;
- **S_eff** in band [8, 16] (else COLLAPSED, reported, gate fails);
- the launch-collapse monitor did not fire unresolved on the claim cell.

**If ALL clear ⇒ run the ORGANIZER SWAP** — the tier-ii verdict per `PREREG-TierII.md`'s own
falsifiers (F1 bar `OD_min > +0.05` worst reader; F2; F5 with E-T5's imitability caveat), against
`N1′` (same store parameterisation, same objective, static read, no dynamics) with the F3-equal
tuning standard BOTH ways and k + head-params on every arm's ledger. Primary swap form: **the same
trained launch head FROZEN onto both arms (bit-identical launches; only the organizer varies)**;
robustness arm (runs only if the primary swap is non-vacuous): N1′ with the launch head refit at
your training budget. ⛔ **The verdict is adjudicated by the ADVISOR against raw artifacts — you
report, you do not adjudicate.**
**If any guard fails ⇒ NO swap.** Report the negatives as negatives (iteration 1's format), and the
wave's product is the cardinality mechanism + the monitor/regularizer measurements, which feed the
CSF3 run-3 config decision at the Advisor's review (§A21 adjudication order). Either way this is a
paper section, not a discard.

## Pre-registration obligation (before the harness runs)
`PREREG.md` in your output dir: K0 bar · R1/R2/G1–G4 re-registered with signs/thresholds/seeds ·
the reader-class re-registration (deliverable 6) · the launch-collapse monitor's band + designed
negative · the regularizer's OFF-until-monitor-fires activation rule · ψ declaration · your own
numeric predictions (w14 rule) · declared NOT-RUN list. Kill-conditions and K0 run before the arms.

## Acceptance
K0 (learned head) adjudicated first · all five deliverables built and measured · the four guards
each shown firing on a designed negative (pytest-asserted, iteration 1's precedent) · launch-collapse
monitor live with band + negative · launders recomputed live per cell · the gate adjudicated
mechanically and the swap run iff it fires · S_eff in band or labelled COLLAPSED · multi-seed ·
tests green (baseline 1363/0 on `main @ 104ca19`) · report → Hub, spawn nothing.
**Ownership (declared):** `chlu/core/multiwell_read.py` · `chlu/core/factored_store.py` (additive) ·
a new launch-head/multiplicity module (yours to name) · `chlu/core/monitors.py` (**additive only** —
the launch-collapse row; ceded by C2W6 on the §10 record) · `chlu/experiments/exp_tierii_read.py`
(extend; or a new exp module) · your tests · `chlu/cli/experiment_cmd.py` (additive only).
⛔ **NOT yours:** `chlu/training/train_cluformer.py`, `chlu/core/blocks.py`, `scripts/csf3/`
(C2W6/CSF3 territory) · `chlu/core/psi_readout.py`'s quarantined path ·
`chlu/core/null_arms.py` beyond its public API. **Git:** branch + scoped worktree off local `main`;
verify commits from the main repo (§3.2); never push `origin`; `clu-dev` only.
