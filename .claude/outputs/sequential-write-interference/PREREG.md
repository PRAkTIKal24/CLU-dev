# PREREG — sequential-write-interference (w21, experiment-engineer)

Written **before** any harness in `chlu/experiments/exp_sequential_write.py` was executed
(the module did not exist yet when this file was committed to disk).
Base: local `main` @ `31c3e15`. Branch `agent/experiment-engineer/sequential-write-interference`.

The claim under test (Hub): *"unconstrained writes destroy stored items; an admission gate
suppresses it by ~5 orders — locality is a certificate the controller CHECKS, not a property
learning provides."* Reference points supplied by the Hub: designed+ungated `0.000`;
learned+ungated `2.9e-2 … 5.0e-1`; theorist designed+gated `8.0e-5`.

---

## Derivations that the predictions rest on (stated first, so they can be wrong)

**D1 — the spacing gate is arithmetically vacuous on the exact w20 geometry.**
w20 §3 writes A on a K=4 site ring of radius `f = 1.0`; nearest-neighbour site distance is
`2 f sin(pi/4) = 1.4142`. MVC-0's gate is `d_safe = 4.4 s`, with `s` the write's spatial
width. On the learned rungs the only spatial width in the write objective is
`write_sigma_addr = 0.25`, giving `d_safe = 1.10`. Since `1.4142 > 1.10`, **every w20 write
is already admissible** and `refuse-and-relocate` can never fire. The gate therefore cannot
change any w20 number by construction. (Same arithmetic at K=8: spacing `0.765 < 1.10`, so
the gate *would* fire there — a K=8 ring is inadmissible under MVC-0, which independently
predicts w19's ~8-item ceiling from an admission rule.)

**D2 — the theorist's 8.0e-5 is a property of the WRITE OPERATOR, not of the gate.**
S3's writes are atom writes: `dV = -A exp(-|q-c|^2/2s^2)`, whose gradient at a stored
minimum decays as `A (d/s^2) exp(-d^2/2s^2)`. At `d = d_safe = 4.4 s` that factor is
`exp(-9.68) = 6.3e-5` — i.e. **the 5 orders come from the exponential tail of a LOCAL
basis function, and the gate only guarantees the argument stays in that tail.** A learned
MLP `V_theta` has global support: `<d_theta V(q*_i), dtheta>` has no `exp(-d^2)` factor at
all. So a spacing gate placed around a *global-support* write is expected to buy ~nothing.

**D3 — ring capacity under the gate.** Sites confined to the `r = f = 1` ring with
`d_safe = 1.10` admit at most `pi / arcsin(d_safe/2f) = 5.39 -> 5` items.
Sites free in a box `U[-2,2]^2` with `d_safe = 1.54` (theorist's `s = 0.35`) admit at most
`~7.8` by hexagonal packing (area bound; realistic random sequential adsorption gives
fewer). Theorist measured 13/20 admitted in a slightly larger relocation window.

---

## Item 1 — the gate on the w20 failing setup (>= 5 seeds)

| # | prediction | falsifier |
|---|---|---|
| **P1** | **The spacing gate changes nothing on the exact w20 ring geometry.** For every rung, `\|corruption_gated_spacing - corruption_ungated\| < 1e-3` and the admission rate is **1.000** (zero refusals, zero relocations). | any refusal, or a corruption change > 1e-3 |
| **P2** | **`sites_learned_payload` stays destroyed under the spacing gate:** strict-A-after-B `<= 0.10` (w20: 1.000 -> 0.000), corruption in `[2e-2, 8e-1]` — i.e. **the theorist's 8.0e-5 does NOT transfer to a learned landscape via the spacing gate.** I am pre-registering the *negative*, per D2. | strict-A-after-B >= 0.5 or corruption < 1e-2 under spacing-gating alone |
| **P3** | With a **crowded proposal distribution** (B proposed uniformly in the disk `r <= 1.3` instead of at the 4th ring site) the gate becomes non-vacuous: refusal-or-relocation rate `>= 0.35`, and on the **designed atom** landscape gated max fixed-point drift `< 1e-3` while ungated `> 1e-1`. | drift ratio < 10x |
| **P4** | **Only the C3 admissibility check changes the learned outcome, and it does so by REFUSING.** With `delta_budget = 0.10`, the C3-gated learned write is truncated to `< 25 %` of its step budget on `>= 80 %` of cells; corruption falls `< 1e-2` **but the new item B is not stored** (strict-B `< 0.5`). | corruption < 1e-2 *and* strict-B >= 0.9 (would mean the gate rescues at no capacity cost) |
| **P5** | Anchoring the stored items in the loss during B's write (C3 option (b), rehearsal from the codebook) **is** able to rescue: corruption `< 5e-2` with strict-B `>= 0.9`. This is a *structured write operator*, not an admission gate — reported as the mechanism that actually works. | corruption >= 5e-2 |
| **P6** | The first-order law `\|\|H^-1 grad dV(q*)\|\|` predicts the *measured* fixed-point drift within 2x on `>= 60 %` of designed-atom writes (theorist: median ratio 1.0002, 100 % within 2x). On learned writes the same law is expected to be far worse (large perturbation): within 2x on `< 60 %`. | either half |

## Item 2 — the sequential-write curve (K = 1..16)

| # | prediction | falsifier |
|---|---|---|
| **P7** | **CLU learned + ungated: retention of item 1 falls below 0.5 after exactly ONE subsequent write** (w20: 1.000 -> 0.000 on one write) and stays `<= 0.2` thereafter. Mean retention over stored items at K=16: `<= 0.15`. | item-1 retention > 0.5 after 2 writes |
| **P8** | **CLU designed + gated: retention of item 1 stays `>= 0.95` for every admitted write, all the way to K=16**, and mean retention over admitted items `>= 0.90`. | any drop below 0.9 |
| **P9** | **Admission count**: of 16 sequential proposals in `U[-2,2]^2` at `d_safe = 4.4 s`, the gate admits **8-14** (point estimate **11**; theorist 13/20 = 0.65 -> 10.4/16, area bound 7.8, so the interval brackets both). Ungated admits 16/16 by definition. | outside [8, 14] |
| **P10** | **CLU learned + gated: the gate does NOT rescue the learned landscape.** Retention of item 1 at K=16 is closer to the learned+ungated arm than to designed+gated: `retention_learned_gated < 0.5 * (retention_designed_gated)`, unless the C3 check refuses so aggressively that fewer than 4 items are ever stored. Explicitly: **I predict the gate's benefit on the learned arm comes entirely from refusal, not from locality.** | learned+gated retention >= 0.9 at K=16 with >= 8 items stored |
| **P11 (crossover K)** | The **crossover** — the K at which designed+gated mean retention exceeds learned+ungated mean retention — is at **K = 2** (i.e. immediately; a single subsequent write already separates the arms). I additionally predict the *gap* is >= 0.6 by K = 4. | crossover K >= 4 |

## Item 3 — the cross-primitive comparison

| # | prediction | falsifier |
|---|---|---|
| **P12** | **Sequential parametric writes catastrophically forget in EVERY primitive.** For transformer, GRU, MLP and CLU-block alike, retention of item 1 after 15 subsequent writes is `<= 0.30`, and mean retention at K=16 is `<= 0.40`. **No primitive is qualitatively better; the axis that matters is the gate, not the architecture.** | any primitive holds item-1 retention >= 0.6 at K=16 without rehearsal |
| **P13** | The spread across primitives at K=16 mean retention is **small** (max - min `<= 0.35`) compared to the CLU designed+gated vs CLU learned+ungated gap of P11 (`>= 0.6`). | primitive spread > gap |
| **P14 (compute-to-criterion)** | Steps for item K to reach criterion **rises with K for every primitive** (>= 2x from K=2 to K=16). ⚠ The Head's hypothesis is that this is *wasted compute reorganising to conserve key info*; the discriminating measurement is whether the rise differs across primitives. **I predict the rise is broadly FLAT across primitives (all within 2x of each other), which would mean the compute cost does not distinguish architectures.** | one primitive's growth factor > 3x another's |
| **P15** | The **joint** criterion (item K correct *without* dropping items 1..K-1 below threshold) is **unreachable within the step budget for >= 50 % of (primitive, K>=8) cells**. Catastrophic forgetting under sequential writes is not something more steps fixes. | < 50 % censoring |
| **P16 (retrieval cost scaling)** | **Parametric** read cost is flat in K for ALL primitives (forward FLOPs identical to the digit; wall-clock variation < 15 %). **Contextual** attention read cost grows with K: log-log slope of FLOPs vs K in `[0.9, 2.1]` (linear-to-quadratic in sequence length). CLU landscape rollout cost is flat in K (< 15 % variation over K = 1..16). | attention contextual slope < 0.5, or CLU rollout variation > 15 % |

## Item 4 — scope statement (not a prediction, a commitment)
CLU's writes here are **parametric** (into `V_theta` / into the block's weights, at training
time). Attention's memory in the *contextual* arm is a KV cache written at inference. These
are different capabilities. **No contextual-memory claim will be made from the parametric
experiment**, and the transformer's parametric arm — not its contextual arm — is the one
compared on retention.

---

### Instrument commitments (registered so they cannot be tuned after the fact)
1. **Blank control over the strongest read on every reported cell** (w20 method finding): a
   landscape written with all-zero payloads, scored with the *real* payload targets, under
   both the value read (strict) and nearest-centroid. Any cell whose blank fails is reported
   as **unmeasured**, not as a pass or a fail.
2. **Value-recovery (strict = basin AND |read - stored| < payload_tol), never classification**,
   is the primary retention metric.
3. Seeds `0..4` (>= 5) everywhere; every arm sees the identical seed list and identical
   proposal sequence per seed, so arms differ only in the controller.
4. Cross-primitive: identical LR grid, identical step budgets, identical seeds, and the
   **symmetric monotone rescue pass** of `primitive-harness` (adopt an alternative LR only
   if it wins on the full n-seed mean — the winner's-curse fix).
