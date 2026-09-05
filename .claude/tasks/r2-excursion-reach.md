# r2-excursion-reach — the two-arm excursion task + the init×width factorial

**Agent:** experiment-engineer. **Worktree MANDATORY** (§3.2). Base local `main` (post-w25).
Carries **R2's revival** (addendum-2 §B1.3, §B1.4, §B3.4, §B3.5). ⚠ **The heaviest compute task
of the wave** — read §Compute before planning.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** capacity (the R2 law). A law about the primitive — exempt from the masked-recall
  demotion, and **its figure is never framed as beating anything** (CM-23(m)).
- **Laundering control:** the designed write at matched geometry must keep reaching its own wall
  (`K_designed(4) = 128` on this harness). If a lever "works" only by making the learned write
  more designed → **N46 scope collapse**, not a win. Report the designed arm at every format
  change.
- **Falsifies:** neither excursion arm moves the wall at **≥3 seeds** under a budget-adequate
  atom count **with payload read-noise ON**.
- **Does NOT falsify:** failing to reach the designed `4·2^d` (the 4× prefactor gap is expected
  and known); any comparison to kNN or external methods.

## ⭐ THE FIVE BINDING FAIRNESS CONDITIONS (Head ruling B1.3 — both arms, no exceptions)
The Head ruled the read-out excursion **a legitimate interface parameter** — an *output code*, not
a difficulty knob; every architecture chooses its output encoding. That ruling comes with five
conditions, and a violated condition invalidates the arm:
1. **Bits-per-item held constant** across every arm and the baseline.
2. **Byte accounting pinned explicitly** — no capacity smuggled via extra parameters. Publish it.
3. **Payload read-noise ON.** ⭐ This is the crux. w25's harness launches queries at *exactly*
   `payload = 0` with a noise-free value channel, which is precisely why shrinking the excursion
   was free there — and why w25 forbade quoting any `K_learned` at `pscale ≠ 1`. **The free lunch
   must survive noise or it is not a result.**
4. **Baselines given the same format** (the designed arm reads the same payload code).
5. **The laundering control travels** with every number.

## Stage A — the init×width factorial (do this FIRST; it is cheap and it gates a default)
Two large levers landed in w25 and they may be **one effect**: `lattice-sharded-store`'s
localized atom init (N98: +0.051 monolithic / +0.082 sharded at d=6 K=64, and it repaired the
router 0.913→0.970) and `r2-geometry-revival`'s width finding (`atom_init_width` 0.30→0.15
*destroys* retrieval; 0.30 sits at a measured optimum). Both attack *"atoms in the wrong place"*
from different sides.

**Design (Advisor-2): a 2×2 factorial with the interaction term, 3 seeds.**
`atom_init_local ∈ {False, True}` × `atom_init_width ∈ {0.30, 0.15}`, at **d=6 K=64**, monolithic,
value-blank controlled. Report the two main effects **and the interaction term explicitly**.
**Registered prediction on file (Advisor-2): substantially ONE effect.**
⛔ **Neither lever becomes a default before this cell reads out** (Head, B1.4) — you are measuring,
not promoting. Do not change any default in this task.

## Stage B — the two excursion arms (gated on nothing; run after Stage A)
Both test the same mechanism from w25 §4: *the read launches from the payload-zero manifold, and a
well whose payload sits far from that manifold is invisible from where the ball is dropped*
(`s ≳ σ_q` from below, `s ≳ |a|_max/κ`, κ≈3, from above).

**Arm (a) — multi-channel payload.** Split the fixed value range across `m` payload channels, each
of excursion `|a|_max/√m`, carrying the **same total value precision**. Lowers per-axis reach
demand at constant information content. Sweep `m` (e.g. 1, 2, 4, 8) at d=4 and d=6.

**Arm (b) — the annealed / continuation read [Advisor-2].** Widen the atom widths *during
settling* on a schedule, `σ_eff(t) = √(σ² + s(t)²)`. **Analytically free for Gaussians** (a
Gaussian convolved with a Gaussian is a Gaussian), and it decouples storage width from read reach
with **no change to the stored payload format at all** — which makes it the cheaper and more
elegant arm if it works. Derive the schedule from the theorist's predictions
(`readout-channel-theory`, this wave) if they land in time; otherwise register your own schedule
before running and say so.

**For both arms:** N92 budget-adequacy at **every first-fail cell** (re-check at 2× atoms; a cell
that passes at 2× is *budget-limited (lower bound)*, never a wall). ≥3 seeds on every decisive
cell. Value-blank control on every reported PASS.

## If the wall moves
Say so precisely and **re-measure capacity-per-byte** (addendum-2 §B2 Candidate 3): w21's
bits-per-param loss (~1.3 vs 2) was measured *under* the reach ceiling, so it is stale if either
arm unclamps. The log-linear capacity figure returns **with a mechanism story attached** — that is
the R2 deliverable. If the wall does not move, that is a clean, publishable close of the R2
direction with a named mechanism, and you should report it as such.

## Compute (declare, do not silently drop)
w25 measured ~1340 s per write per seed at d=8/16384 atoms, and 4 engineer worktrees share 8 CPU
cores this wave. **Prioritise: Stage A → Arm (b) at d=4 → Arm (a) at d=4 → d=6 → d=8.** Report any
dimension you could not reach as **NOT RUN**, never as a null. Declare the priority order in your
PREREG before you start.

## File ownership
**You own:** `chlu/experiments/exp_designed_mechanism.py` + `ExperimentDesignedMechanismConfig`,
and the atom-init/width knobs in `chlu/core/memory_potentials.py`. ⚠ `placement-landing` is
editing `chlu/core/controller.py` and `AtomStorePotential.evict` this wave — **do not touch
either**. Do not touch `exp_cl_entry.py`, `cl_baselines.py`, `exp_phi_stream.py`.

## Deliverable
PREREG first (`.claude/outputs/r2-excursion-reach/PREREG.md`) — the five fairness conditions
restated as checkable items, your registered predictions for Stage A's interaction term and each
arm's wall movement, and the compute priority order. Report at
`.claude/outputs/r2-excursion-reach.md`, standard format, PREREG scorecard, reconciliation list in
the first 10 lines. Full `pytest tests/` green, `ruff` clean, atomic commits on
`agent/experiment-engineer/r2-excursion-reach`. Do not push.

⛔ **Do-not-quote, carried:** `K_learned` at `pscale ≠ 1` **unless the payload-noise condition is
satisfied** (that is the whole point of condition 3) · "the write operator is the ceiling" ·
width-lock-as-cause · the base-√2 / `d^1.62` exponent · "~32, d-independent" as settled.
