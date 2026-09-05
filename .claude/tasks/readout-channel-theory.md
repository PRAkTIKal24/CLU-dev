# readout-channel-theory — one rigorous pass over the read-out channel

**Agent:** physics-theorist. **No worktree** (you touch no tracked code — flag what the engineer
should change, per your standing scope). Base local `main` (post-w25).
Addendum-2 §B3.6, consolidated: three questions, **one object**.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** capacity (the reach condition) + lifetimes (the payload-dependence ruling).
- **Laundering control:** N/A for derivations; **every numerical sanity check you run must use
  the shipped `V`**, not an idealised Gaussian, and you must say which you used for each claim.
  Where an analytic result does not transfer to the shipped store, say so — that is exactly the
  failure mode w25 caught in `lattice-capacity-theory` (its §5.2 causal prediction was refuted by
  experiment, and its §5.0 correlational one survived; the P6 magnitudes did not transfer at all).
- **Falsifies:** a derivation that contradicts the measured reach numbers in
  `r2-geometry-revival` §4 (force collapse 5.1×, corr −0.887 with |a_i|, the half-excursion
  intervention 0.824 → 1.000).
- **Does NOT falsify:** predicting a *smaller* wall movement than the engineers hope for; an
  impossibility result on either excursion arm is an acceptance, not a failure.

## The object
The read launches every query on the **payload-zero manifold** (the anti-decoration guard), a
distance `|a_i|` along the payload channel from the stored target. A Gaussian well of width `s`
and depth `D` exerts force `D·r/s²·exp(−r²/2s²)`, which collapses super-exponentially in `1/s`.
Everything below is a consequence of that one fact. **Source: `.claude/outputs/r2-geometry-revival.md`
§4/§4.1/§8 — read it before anything else; it is the measurement your theory must match.**

## Q1 — Formalize the reach condition
w25 states it as a two-sided bound and derived κ by eyeball: `s ≳ σ_q` (the basin must contain the
jittered query — the floor the previous theory task identified) **and** `s ≳ |a|_max/κ` with
κ = O(3) (the basin must be visible from the launch manifold — the ceiling-side condition nobody
had written down). **Derive κ properly.** What sets it? How does it depend on the well depth `D`,
the confinement `α`, the read schedule (`γ_address 0.05 × 400 → γ_read 0.02 × 800`, `dt 0.05`),
and the competing payload hill? Give the bound in a form the engineer can evaluate for a proposed
configuration, and state its regime of validity.

**Then close the loop the w25 report left open:** the measured `sep/width` ratio is a *perfect*
PASS/FAIL classifier at d ≥ 4 (boundary in (2.30, 2.49)) yet **not** the causal variable (forcing
it to 4.90 destroys retrieval). Explain *why a correlate that sharp arises* from the reach
account — w25 conjectures that both numerator and denominator drift smoothly with `d`. Confirm,
refute, or replace that conjecture.

## Q2 — Rule on the payload-dependent lifetime (this currently undercuts the lifetime dial)
`mia-decay-measurement` §5 measured: retention at the amplitude floor correlates **r = −0.846
with `a_i²`** and **r = +0.015 with `|c_i|`**. Mechanism from the shipped `V`: the payload term is
`0.5κ(q₂ − S(q_addr))²` and the guard forces `q₂(0) = 0`, so at launch the item's own site carries
a payload **hill** of height `0.5κa_i²` (up to 0.5 for `a_i = ±1`) competing with an address
**well** of depth `A` (0.05 at the floor — 10× smaller). The full-`V` probe shows `s5` crossing
zero between A = 0.4 and 0.3 and reaching −0.113 at the floor: **the decayed site becomes a net
maximum of `V` at `q₂ = 0` for large `|a_i|`.**

⇒ **Two items given the same `leak` do not have the same effective half-life.** Rule on the fix.
Three candidates are on the table; recommend one and justify it against the others:
- **(a)** scale the payload bump by the amplitude (`payloads * amps`);
- **(b)** launch reads at `q₂ = S(q_addr)` instead of 0 — ⚠ check this against the anti-decoration
  guard the zero-launch exists to enforce; if it breaks it, say so;
- **(c)** ⭐ **the cheap calibration option [Advisor-2, explicitly requested]:** leave the
  potential **untouched** — the user specifies a half-life and the store *solves* the per-item
  `leak` numerically to hit it. No physics changes; the dial becomes calibrated rather than raw.
  Evaluate honestly: does this make the dial *honest*, or merely *look* honest? If the retention
  curve's shape (not just its half-life) is what a referee will probe, say so.

Your ruling gates any change to the potential; the engineer will not act without it.

## Q3 — Theory for both excursion arms (pre-register predictions `r2-excursion-reach` can score)
- **Arm (a) multi-channel payload:** the fixed value range split across `m` channels of excursion
  `|a|_max/√m` at constant total precision. Predict the wall movement as a function of `m`, with a
  falsifier. Where does it stop helping, and why?
- **Arm (b) the annealed / continuation read:** widen atom widths during settling,
  `σ_eff(t) = √(σ² + s(t)²)` — analytically free for Gaussians. Derive the **schedule**: what
  `s(t)` should be, over which phase of the two-phase read, and what it costs in address
  discrimination (a wider effective well is a less selective one — quantify the trade). Predict
  whether the anneal must terminate before the read phase and why.
- ⭐ **The condition that decides both:** w25's harness has a **noise-free** payload channel, and
  the Head's fairness rules put payload read-noise ON. **Derive what read-noise `σ_a` does to each
  arm.** Arm (a) shrinks per-axis excursion — does the noise scale with it (free lunch cancels) or
  not (real gain)? This single question is worth more than the rest of Q3.

Deliver these as **numbered, falsifiable predictions with bands**, formatted so the engineer can
paste them into a PREREG scorecard. If an arm is provably a free lunch that cancels under noise,
**say so now and save the wave the compute** — an impossibility result here is a full acceptance.

## Deliverable
PREREG first if you run numerics (`.claude/outputs/readout-channel-theory/PREREG.md`). Report at
`.claude/outputs/readout-channel-theory.md`: derivations with assumptions stated, numerical sanity
checks (say for each whether it used shipped `V` or an idealisation), the Q2 ruling with its
recommendation, the numbered predictions for Q3, and a reconciliation list in the first 10 lines.
Small numerical checks in `.claude/scratch/readout-channel-theory/`. **No tracked code.**

⚠ **Timing:** `r2-excursion-reach` starts on its Stage A (the init×width factorial) and does not
block on you — but your Q3 predictions should land before its Stage B. Post them as soon as they
exist rather than holding them for the full report.
