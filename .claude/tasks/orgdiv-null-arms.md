# orgdiv-null-arms — the merciless matched-capacity organizer controls (N1–N5, `null*`)

**Campaign 2, wave C2W5. Agent:** experiment-engineer. **Worktree 3 of ≤3 — ⚠ GATED: spawns only
after `orgdiv-cat-test` publishes `.claude/outputs/orgdiv-cat-test/FROZEN-interfaces.md`** (frozen
φ instance · frozen reader class · frozen split · frozen launch protocol). Branch
`orgdiv-null-arms`, scoped worktree. Writes `.claude/outputs/orgdiv-null-arms.md` +
`.claude/outputs/orgdiv-null-arms/*`. Charter **ADDENDUM 4 §A19 task 2**, executing
**`PREREG-TierII.md` §4 VERBATIM (binding).**

**Read first:** `.claude/AGENT_PROTOCOL.md`; **`PREREG-TierII.md` in full** (esp. §1 the ledger ·
§4 the arms · §3 the falsifiers your arms feed); charter **§A13/§A19**; the `FROZEN-interfaces.md`
checkpoint (your ground truth for φ/readers/split/launches — ⛔ if anything there is ambiguous,
STOP and flag the Hub; never improvise a match); `.claude/outputs/bprime-rivals-f3.md` (the tuning
standard you inherit); the `[C2W4-CLOSE]` §10 entry.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **TIER ii — the organizer swap's NULL SIDE.** You build the competition. ⭐
  **A hobbled null is the same referee attack as a hobbled rival, in mirror image** (§A19.2) — your
  arms get **the F3-grade tuning standard**: full registered grid, held-out selection, honest power.
- **Control:** you ARE the control. Your own guards: identical φ (the frozen instance — byte-compare
  it), identical bytes/capacity (two-sided ledger, corrected law, learned-initial-state rule: init =
  PARAMETERS, per-stream deviation = STATE, both declared per arm), identical launch protocol,
  identical reader class + fitting budget.
- **Falsifies:** nothing of yours — you produce `null*`. But ⛔ **selection on `Q_unseen` anywhere
  in your pipeline invalidates the wave** (tuning selects on a held-out-from-SEEN validation split,
  never on unseen combinations).
- **Does NOT falsify:** an arm of yours beating the physics arm is a legitimate outcome of the
  experiment, not a defect to fix quietly.

---

## 1. The five arms (prereg §4.2 — each in its STRONGEST registered form)
1. **N1 — gradient-placed atoms** ⭐ (the cleanest, most damaging swap): **identical store
   parameterisation** (`c_j, log s_j, amp_j`) trained by plain Adam on the read objective with a
   **static** assignment rule — no rollout anywhere. DOF/bytes identical to the physics arm by
   construction.
2. **N2 — VQ:** best of {k-means++ ×10 restarts, VQ-STE with EMA, product-VQ}; commitment cost
   swept ≥ 5 points; `N_a` codes + per-code payload.
3. **N3 — fitted static-geometric rule:** `argmin_j [‖z−c_j‖²/2σ_j² − b_j]` (power/Apollonius),
   `(c, σ, b)` fitted jointly on SEEN. **This is F5's null** (fires if it reproduces the physics
   arm's assignment on ≥ 99 % of held-out queries). ⭐ **Plus the oracle-imitation variant: N3
   fitted on the physics arm's own assignments** (T5.2 rider (i)) — a physics arm that cannot beat
   an imitation of itself has no organization claim.
4. **N4 — kNN:** no training; raw keys + payloads; `k ∈ {1, 2, 3, 5, 10}`, uniform and IDW (the
   C2W1 `knn2_idw` substitute was the arm that beat us — treat it with respect).
5. **N5 — Titans-style write:** surprise-gated fast-weight rule, momentum + weight-decay per the
   published rule, chunk granularity matched to the physics arm's; matched param count;
   learned-initial-state rule declared.

## 2. The tuning standard (prereg §4.3 + the F3/A17.4 inheritance)
- **Registered budget, committed:** ≥ 5 lr points × 3 capacity points × 3 seeds per arm on SEEN,
  selected on held-out-from-seen validation. The physics arm gets the same budget, no more —
  cross-check the declared budgets with `orgdiv-cat-test` before either of you scores a final cell.
- **`null* = max over ALL arms AND their entire registered grid — computed, not estimated.** ⛔ The
  weak null (one sampled config) may not carry a headline.
- ⭐ **A17.4 standing practice:** any gate/verdict whose control has learned-init variance uses
  paired or multi-init controls, or n ≥ 9 (the C2W4 lesson: n=3 rescue verdicts were coin flips).
- **Per-query compute rule (prereg §1):** report per-arm read compute; if arms differ by > 2×, the
  cheaper arm additionally runs at the richer arm's budget, else the comparison is void.

## 3. Deliverables
- Per-arm: score on SEEN-validation and `Q_unseen` per reader × seed × γ claim cell, full grid
  results (JSON), byte ledger row, compute row, the selected config + selection trace.
- `null*` per cell, with the arg (which arm/config attains it).
- The N3-vs-physics assignment-agreement number (F5's input) and the oracle-imitation row.
- Declared NOT-RUNs listed, never as nulls. Multi-seed (5 for scored cells) before any number.

## 4. Acceptance
All five arms land in their strongest form with full grids and ledgers; `null*` computable per cell
by mechanical max; ledger identity checks pass (φ byte-compare; bytes/capacity/launch/reader
identity vs `FROZEN-interfaces.md`); tests green on your branch.

**File ownership:** you own `chlu/**` files you create for N1–N5 + your tests. ⛔ Do NOT touch the
factored-store family files, `monitors.py` (both `orgdiv-cat-test`'s), or anything
`pilot-placement-probe` declares. Declare your exact list in your report's first section.
**Git:** branch + scoped worktree; never push `origin`; `clu-dev` only. Report → Hub, spawn nothing.

---

# HUB ADDENDUM (2026-08-01, at the wave review — ✅ RE-SCOPE APPROVED BY THE HEAD, ruling 2, same
session; this addendum is now the binding scope and the spawn line is released)

**Head clarifications on the record (ruling 2):** the physics arm you are auditing against was the
FULL factored store as pre-registered — nothing nerfed, nothing unbuilt; "no physics arm to swap"
means "reads at chance," not "absent." **(b) PRE-REGISTERED HERE, before you run: if N1 clears
chance + 0.05, its score IS the revival target the tuned physics arm of the next iteration must
beat** — record N1's score per reader × seed with that framing in your report. (c) The organizer
swap is DEFERRED, not cancelled — a working physics-arm iteration follows per ruling 1, carrying
the K5 kill's design inputs (the S_eff collapse fix · the ψ payload residual · m=8/a=32).

**The situation changed:** `orgdiv-cat-test` died at K5 — the physics arm reads `0.0008 ± 0.0008`
vs chance `0.0004` on unseen queries (COLLAPSED per rule 3: `S_eff` 34–51 vs band [8,16]; untuned;
reported as a pre-condition kill, not a tier-ii null). **There is no physics arm to swap against**,
so this task as scoped above cannot produce an `OD`.

**The re-scoped question (recommended): the FAMILY-SOLVABILITY AUDIT.** Build N1–N5 exactly as
specified above, against the same `FROZEN-interfaces.md`, and report: **does ANY matched-capacity
organizer clear chance + 0.05 on the rule-4-valid unseen split?** Two decision-grade outcomes:
- **None clears** ⇒ the family is refuted for every organizer class measured — a stronger, cheaper
  statement than a tier-ii null, and the family (not the physics) is the first fix.
- **Any clears — especially N1 (identical store parameterisation, non-physics training)** ⇒ the
  family is solvable *within the same landscape class*, and the cat-test kill becomes ATTRIBUTABLE:
  the physics write/read specifically, not the family, is what failed. N1's score then becomes the
  target any physics-arm revival must meet, and the revival inherits the cat-test §13 fix list
  (S_eff collapse first; the ψ payload-residual mechanism from the probe's §6; `m=8`/`a=32` as
  measured constraints).
Everything else above stands: strongest registered forms, the full tuning grid (the F3 standard —
tuning an arm that might WIN the audit matters even more now), identical φ/bytes/launch/readers,
selection never on `Q_unseen`, 5 seeds on scored cells, A17.4 power rule. Report `max_arm` per
reader with the same curve discipline; the settle-deleted launder column is not needed (no physics
arm); K-verdict context quoted from the cat-test report, never re-adjudicated.
