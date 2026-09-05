# ERRATA-ARMB — dated addenda to `PREREG.md` (`c2w8p2-emission-head`)

Filed **before** the cells each block governs. `PREREG.md` is **not edited**.

---

## §1 — 2026-08-07, filed BEFORE the second (final) arm-B census run: the first configuration produced a DEGENERATE CELL, and the arm was re-configured once

### What happened

The arm's **first** configuration was run on the real rig, 3 seeds, `main @ 80d7d4b` +
`47116c8`. It did not produce a scoreable cell:

| seed | `n_admitted` / target | `n_live` | `overdig` | emitted depth (median) |
|---|---|---|---|---|
| 0 | 2 / 16 | 2 | 0.25 | 0.1005 |
| 1 | 1 / 16 | 1 | 0.12 | 0.0589 |

⇒ **the admission gate refused 38 of 40 offers**: the head emitted near-identical centers, so
every offer after the first fell inside `d_safe = 0.124` of a live one. ⛔ **This is a broken
cell, not a result** — a gate scored on 1–2 wells is not the pass-2 gate, and the three legs are
reported for completeness only, never as arm B's reading.

### The cause, diagnosed mechanically (not guessed)

A pretraining probe on a synthetic-φ stand-in (`.claude/scratch/` → `/tmp/diag2.py`, parameters
logged in the report) isolates it in one line:

> **With a plain random-init MLP, every item's emitted center starts at ≈ 0, the head sits at an
> almost exact permutation symmetry between items, and gradient descent from there converges to a
> CONSTANT placement map.** Measured: median pairwise center separation **0.027 – 0.069** against a
> φ nearest-neighbour spacing of **0.339**; emitted depth driven to its floor (0.06) because
> overlapping wells make `L_grad` blow up, so the cheapest escape is to have no wells at all.

Nothing about this is specific to the write objective's weighting: adding the w24 crowding lever at
`crowd_d_safe`, raising the reach weight ×100, raising the depth floor, extending to 4 000 steps and
raising the learning rate ×3 all leave the map constant (separation 0.027 – 0.11, always ≪ the φ
spacing).

⭐ **The second, sharper half of the diagnosis, and it is the finding this arm carries:**
**the reach hinge alone cannot supply the missing pressure.** At the emitted widths a Gaussian
basin (`ρ·s`, `ρ = 2`) is comparable to the whole address ball, so *"the launch must be inside the
basin"* is near-vacuous, and a constant map at the pool's centroid satisfies it. The only way to
make the reach hinge bite is to shrink its slack to the key spacing — **at which point it is a pin,
and the Head's binding prohibition forbids it.**

### The two changes, and why each is a *designed* choice rather than a tuning knob

**(1) A trainable linear skip on φ in the head, initialised at `gain · I`
(`emission_center_skip_gain = 1.0`).** ⭐ This is an **initialisation**, not a constraint — the
`atom_local_radius` / N98 precedent, where the codebase already treats a localized init as a
designed lever. ⛔ It is not a pin, and the direction of the only pressure that acts on it
afterwards proves it: decoupled weight decay shrinks `skip` toward **ZERO**, i.e. toward *ignoring*
φ, which is the opposite of what a pin does. Nothing in the objective references `|c − φ|`.
Effect, same probe, same seed: separation **0.027 → 0.47**, `0/16` pairs inside `d_safe`, and
`|c − φ|` **0.21** (non-zero, ≈ 62 % of the φ spacing).

**(2) `attribution_margin_penalty` — THE designed write→φ organization gradient (charter §A28.1),
in the only form that is not a pin.** `relu(margin + |q_i − z_i| − min_{j≠i}|q_i − z_j|)²`: the
launch from item *i*'s own query must be attributed to item *i*'s well **rather than to anybody
else's**. It is **competitive** (its gradient depends on the nearest *other* well, so it says
nothing about `|c − φ|` alone), its zero set is an open **Voronoi** region rather than the point
`c = φ`, and it is **vacuous for a single item** — exactly where a pin is most active. It is
declared as a designed mechanism in the arm's ledger.

Two shipped levers are engaged alongside them, both at **measured** operating points, not guesses:
the w24 **crowding** term at the rig's own admission radius (`crowd_d_safe = d_safe`), the emitted
**width floor 0.30** (the payload reach: a well of width `s` exerts `exp(−a²/2s²)` of its force at
the read's `payload = 0` launch manifold — `0.007` at `s = 0.16`, `0.25` at `s = 0.30`), and the
emitted **depth floor 1.5** (pass 1's measured foreign background at a live site, median
`0.611 – 1.261`; a shallower well is a minority of the landscape at its own site, which is pass 1's
diagnosis restated).

### Discipline attached

* ⛔ **Nothing was selected against a gate leg.** The configuration was chosen on a *synthetic-φ
  pretraining probe*, against three build-sanity criteria only — (a) items are admitted at all,
  (b) emitted centers clear `d_safe`, (c) emitted centers stay within reach of φ. `capture_radius`,
  `decode` and `site_drift` were **not** evaluated at any point during the selection.
* The **first** configuration's numbers are reported in full in the spoke report beside the final
  ones. Neither is hidden.
* The final configuration is run **once** per seed. No seed is re-run after its result is seen.
* `PREREG.md`'s predictions (K7, and the Hub's P1–P4/P6) are **unchanged** — none of them mentions
  the head's training configuration.

*Filed 2026-08-07 by `experiment-engineer` (wt2), before the final arm-B census executed.*
