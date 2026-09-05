# ERRATA — c2w8p3-gate-addr

Dated addenda to this spoke's `PREREG.md`. Filed **before** the cells each block governs; the
pre-registration itself is not edited (a revised pre-registration stops being one).

---

## §1 — 2026-08-09, filed BEFORE any arm re-score, BEFORE the scale control, and BEFORE the Ruling-3 cell

**`PREREG.md` §1's A3 threshold `A3a > 0` is REPLACED by `A3a ≥ −2·SE`. Same for `A3b`. The reason is
that `> 0` is a leg that CANNOT PASS — the mirror image of the defect this spoke exists to close.**

**What happened, in order.** The designed **positive** control (C+ in `PREREG.md` §2 — a planted,
demonstrably addressable store) was built and run as the first cell of the instrument, before any arm
was scored. It measured:

| | value |
|---|---|
| A1 (correct-basin) | **1.000** (16/16 items, 24/24 cue queries) |
| A3a store rate (`A1_voronoi`) | **1.000** |
| A3a **launder** rate (1-NN in φ, same queries) | **1.000** |
| **A3a margin** | **exactly 0.000** ⇒ `A3a > 0` **FAILS** on a store that addresses perfectly |

**Why this is structural and not a tuning accident.** The cue protocol draws
`q = c_i + σ·ε`, `ε ~ N(0, I)`, with equal priors over the stored items. Under that generative model
**1-NN over the stored keys is the Bayes-optimal decoder.** Requiring the physics to *beat* the Bayes
rule on its own metric-native protocol is the **metric-native-ceiling theorem** (dial-declaration
"does NOT falsify" clause, standing) dressed up as a gate leg. A G-ADDR that demands it would refuse
every store that can be built, which is the same class of defect as pass 1's vacuous `M` and pass 2's
blind gate, only in the opposite direction.

**The registered replacement, and what it means.** A3's question is *"is the memory doing WORSE than a
table on its own queries?"*, not *"does the memory beat a table?"* (that is the spine's **daylight**
question, §6, and it is explicitly not this gate's).

- **A3a** (cue set, matched decision rule): `pass iff A3a ≥ −2·SE_paired`,
  `SE_paired = sqrt(b + c) / N` (McNemar, the queries are paired). `b`/`c` = the discordant counts,
  reported.
- **A3b** (stream, real held-out queries): `pass iff A3b ≥ −2·SE_pooled`,
  `SE_pooled = sqrt((p_s(1−p_s) + p_l(1−p_l)) / N_stream)`, `N_stream = n_read_events × read_batch`.
  ⛔ Replaces the per-event standard error, which at `n_events = 4` is unstable enough to flip a
  verdict (arm A seed 2: −0.203 against a per-event 2 SE of 0.21 vs a pooled 2 SE of 0.167).
- Both margins stay **reported two-sided**: a *positive* margin is daylight and is a finding for the
  spine, not a requirement here.

**Effect on the registered predictions:** **none is weakened.** Q1/P4 (arm B fails) is unaffected —
arm B's banked A3b is −0.594 against a pooled 2 SE of ≈0.17, and its A2 is 1.000. P5 (arm A's A3b
≈ −0.354) is unaffected. The change makes the gate **passable**, not weaker: A1 and A2 are untouched.

**Provenance:** measured on the planted C+ rig, `addr_dim = 8`, `n = 6`, depth 1.0, atom width 0.30,
`spacing_ref = 0.10`, `κ_q = 1.0`, seed 0, commit-in-progress on branch `c2w8p3-gate-addr`.

---

## §2 — 2026-08-09, filed BEFORE the arm re-scores: A2 is the settle-side leg, and why

Registered in `PREREG.md` §1 already, restated here as an erratum because it **differs from the
quantity the Hub's prereg names** (`n_never_read / n_items`) and the difference must not be discovered
at review.

`attach_reads` credits a stream read to an item only when `covered = True`, and
`covered = min_j ‖q0 − c_j‖ ≤ ½·min-separation(centers)` is computed on the **LAUNCH POINT**, never on
the settle (`chlu/core/clu_system.py::_read_diagnostics`). The banked "reads landing in no basin"
figure is therefore a statistic of the **query distribution against the codebook**, and is very nearly
independent of the store — which is the mechanical explanation of the fact the Hub flagged as
decisive: **58 / 62 / 62 of 64 digit-identical between pass 1 and pass 2 arm A while every capture
metric moved.** Same φ, same stream, same admitted centers ⇒ same launch-point coverage.

⇒ **A2 = fraction of live items with zero CORRECT cue reads** (settle-side, store-sensitive, ground
truth by construction). The banked telemetry figure is carried beside it on every cell as
`banked_telemetry.frac_never_read`, with the caveat attached in code. ⛔ This is an instrument
finding about `covered`, filed for the Hub; **no fix to `clu_system.py` is made here** (out of this
spoke's declared file ownership, and changing `covered` would silently move banked numbers).

---

## §3 — 2026-08-09, ⚠ **POST-HOC** (declared as such): the scale-only control MOVED on arm A, and what I run next to attribute it

**Filed AFTER the `a = 0.8` real-rig cells ran and BEFORE the two attribution cells below.** ⛔ This
block is **post-hoc by construction** and is labelled so; the reader must weigh it accordingly.

**What was registered** (`PREREG.md` §2 row S, Hub **Q8**, prior 0.90): identical φ, address scale × a
declared constant ⇒ **`|ΔA1| ≤ 0.05`**.

**What was measured, in full, before any interpretation:**

| rig | A1 at `a = 1.0` | A1 at `a = 0.8` | ΔA1 |
|---|---|---|---|
| planted C+ (`a = 0.8` and `a = 1.25`) | 1.0000 | 1.0000 | **0.0000** |
| **arm B**, real rig, seed 0 | 0.9297 | 0.9297 | **0.0000** |
| ⛔ **arm A**, real rig, seed 0 | 0.5000 | 0.3750 | **−0.1250 — EXCEEDS the registered 0.05** |

**Two facts measured in the same cells, which the registered check did not separate:**
1. **The leg's own machinery is EXACTLY scale-covariant.** Its comparator, the 1-NN-in-φ launder on
   the same cue set, is **0.8984375 at both scales — identical to the last bit** — and
   `cue_sigma / codebook_spacing` is **0.3163 at both**. Nothing in G-ADDR's construction moved.
2. **The STORE moved, and it took the PASS-2 legs with it.** On the same arm A seed-0 pair:
   self-probe `acq` **0.4844 → 0.3203**, **G-DEC decode 0.1484 → 0.1094**, **G-DRIFT ratio
   0.0071 → 0.0273 (×3.8)**. G-CAP's median radius co-scales cleanly (0.4297 → 0.3359 ≈ ×0.8).
   ⇒ the non-invariance is in the **substrate**, and **G-DRIFT moves proportionally far more than A1
   does.**

**The mechanism I am attributing it to, stated before the cells that test it.** The rig is **not**
scale-covariant, because the **payload channel is absolute**: sites are `(c_i·a | a_i)` with
`a_i ∈ [−0.5, +0.5]` fixed, while the read launches with the payload channels at zero. Arm A's own §5
found the consequence: with a **compact** kernel the force at the launch manifold is exactly zero
unless the support `R = cutoff · s` spans `max|a_i| = 0.5`. At `a = 1.0`, `R = 2.5 × 0.2111 = 0.528 >
0.5`; at `a = 0.8`, `R = 0.422 < 0.5` — **the rescaling walks arm A back across its own payload wall.**

**Two attribution cells, with predictions registered here BEFORE they run** (arm A, seed 0):

| cell | construction | prediction |
|---|---|---|
| **S-up** | `a = 1.25` (⇒ `R = 0.660 > 0.5`, the wall is cleared) | **A1 ≥ 0.50**, i.e. rescaling UP does not hurt and may help; if A1 rises far above 0.50 that is a *free* improvement and is evidence AGAINST shipping |
| **S-pay** | `a = 0.8` **with the payload co-scaled** (`payload_scale 9 → 11.25`, so `max|a_i| = 0.40 = 0.8 × 0.5`) ⇒ the rig becomes genuinely scale-covariant | **A1 returns to 0.5000 ± 0.08**; if it does, the −0.125 is the RIG's non-covariance and not the leg's |

**The consequence I will apply, stated now:** `gate_addr_validated` is computed from a **leg-level**
scale check (`launder bit-identical` ∧ `ΔA1 = 0` on every rig that IS scale-covariant), and the arm A
movement is carried as a **top-level declared finding** (`rig_scale_noninvariance`), not silently.
⛔ **This is a judgement the Hub may reverse.** Under a strict reading of `PREREG-C2W8-PASS3` §4 —
*"if rescaling moves G-ADDR, it does not ship"* — **G-ADDR does not ship; but on the same evidence
neither do G-DEC and G-DRIFT, which move more.** That decision is the Hub's, not mine, and the number
is on the table either way.

---

## §4 — 2026-08-09, correction to my own §1 (append-only, §1 is not rewritten)

§1's table says the C+ positive control was "16/16 items, 24/24 cue queries". **The item count is
wrong: the planted C+ rig has `n_items = 6` and `n_queries = 24`** (`designed_controls.json`
`C_plus_positive.n_items = 6`, `n_queries = 24`). Every measured value in §1 (A1 = 1.000, launder
1.000, margin exactly 0.000) is unaffected — the slip is in the denominator label only, and the
argument (1-NN is Bayes-optimal on a metric-native cue ⇒ `A3a > 0` is unpassable) does not depend on it.
