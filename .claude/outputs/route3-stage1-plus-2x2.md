# route3-stage1-plus-2x2 — experiment-engineer report

**Task + acceptance criterion:** measure the store-attribution curve on the merged rig, apply the
§A9.4 stage-2 unlock bar arithmetically, build the §A9.5 per-slot table launder **now**, run the
Route-1 × Route-2 2×2, and quarantine `AttentionPsi`.
**Status: done** (D1–D5 all run; two families ABSTAIN after their one bounded escalation, declared).

> ⏱ **INTERIM = FINAL, and the §A9.4 line the Hub is waiting on is #1 and #2 below.** The task asked
> for the unlock verdict as an interim "the moment I have it"; D2 landed before D4 finished, and the
> 2×2 completed inside the same session, so this single file carries both. **Nothing below §1 changes
> the two verdicts.**

## ⭐⭐ THE TWO VERDICTS, IN THE FIRST 10 LINES (the Hub spawns or does not spawn `route3-stage2` off these)

1. ⭐ **`unlock = true`** — §A9.4 clears on **`overload/load1x_shipped`**, 3/3 admissible seeds, in
   **both** channels: **p-slots at t ∈ {0.05, 0.45, 0.85, 1.25, 1.65, 4.85}** and **q-slots at
   t ∈ {0.05 … 12.85, 59.65}** (18 clearing family×channel×slot rows; smallest clearing margin
   **+0.1338 ± 0.0650 (lower-2SE +0.0039)**, largest **+0.6544 ± 0.1237**).
2. ⛔⛔ **§A9.5 FIRES, AND IT OVERRIDES THE UNLOCK ⇒ NO STAGE 2.** A **per-slot matched-bytes table**
   (K = 6 time-indexed rows, keyed by nearest stored key, leave-one-query-out, **cheaper** than the
   store by 478×) **reproduces the slotted read at 18/18 clearing slots and at 37/38 slots overall**
   — `read − table` mean margin **−0.204 … +0.053**, and the read beats the table by > 0.10 at
   **ZERO slots in either channel**. Per intervention §8.2 / charter §A9.5: **Route 3 has degenerated
   into K time-indexed lookup tables and FAILS REGARDLESS OF DIVIDEND at this weight class.**
   ⇒ **My recommendation to the Hub: do NOT spawn `route3-stage2` on this evidence.** The unlock
   arithmetic is reported as required, and the kill-condition that was deliberately built in stage 1
   is the reason it is not actionable.

**⛔ RECONCILIATION LIST (protocol §5 corollary — needs an owner):**
1. **`to_race_cell` (Route 2) silently dropped the mandatory trajectory launder** — every `route2`
   race cell in C2W2 reported `fired = False` **by absence, not by measurement**. Fixed here; the
   C2W2 Route-2 race card's `trajectory_launder` column is therefore *unmeasured*, not *clear*.
   Owner: whoever re-derives C2W2 Route-2 numbers (Hub/curator). ⚠ It does not change any C2W2
   dividend — Route 2's gate arithmetic never read that field.
2. **No CLI hook for `exp-route3-attribution`** — `chlu/cli/experiment_cmd.py` has **no declared
   owner** in C2W3 and two engineer branches are live, so I did not touch it (task §6: stop and
   report). Owed. Module runs via `python -m chlu.experiments.exp_route3_attribution`.
3. **The 2×2 was a NOT-RUN for a mechanical reason worth recording:** `shell_rig`'s patched
   `build_system` had no `write_objective` parameter and **swallowed the seam** — the crossed cell
   could not be constructed at all. One-line additive fix; now covered by a test.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echoed before the first result
- **Dial / pillar:** **pillar 1 (expressive latents), MEASUREMENT stage.** ⛔ I claim **no dividend and
  no win.** D1–D3 decide whether stage 2 is built; D4 closes a declared NOT-RUN. Every performance
  number below is an audit cell with its launder, never a headline.
- **Laundering control:** three, and they are the task — (a) the per-slot **settle-deleted launder**
  (the curve *is* full − launder, per slot); (b) the **§A9.5 per-slot matched-bytes TABLE launder**
  (a kill-condition, not a control); (c) the **trajectory launder** on every ψ that sees the address
  block. Identical φ on every arm (identity/embedded, 0 B, all cells).
- **Falsifies:** §5. **Does NOT falsify:** a dead q-channel with a live p-channel · a flat curve on a
  family whose store was never admissibly written (an admissibility exclusion, reported with reason).

---

## 1. ⛔ ADMISSIBLE-CELL COVERAGE PER FAMILY — first-class, before any verdict

| family | gym arm | admissible / cells | coverage | one bounded escalation | role |
|---|---|---|---|---|---|
| **`overload`** | `load1x_shipped` (478× anchor) | **3 / 3** | **100 %** | not needed | **VOTES** |
| `aggregate` | `base` | **0 / 3** | 0 % | spent: `write_steps` 300 → **900** → still **0/3** | **ABSTAINS** |
| `manifold` | `base` | **0 / 3** | 0 % | spent: `write_steps` 300 → **900** → still **0/3** | **ABSTAINS** |

Every excluded cell carries its reason, all **write-side**, none silently filtered:
`aggregate` endpoint write loss **0.2463 / 0.3612 / 0.2862** (escalated: 0.2448 / 0.3605 / 0.2409);
`manifold` **0.2494 / 0.3808 / 0.2523** (escalated: 0.2490 / 0.3788 / 0.2523) — against a tolerance of
**0.05**. The escalation moves the loss by ≤ 0.005: this reproduces C2W2's independent finding
(`traj-write-objective` D5) that **the plateau is the atom budget's expressivity floor, not an
optimisation-budget artefact**. ⇒ ABSTAIN means what the ruling says: those two families **neither
unlock stage 2 nor block it**, and the unlock rests on **one** family. The Hub should weigh that
exactly as it weighed the same fact in C2W2.

## 2. ⭐ D1 — THE STORE-ATTRIBUTION CURVE (`overload`, 3 seeds, q and p scored separately)

Instrument (pre-registered, `PREREG.md` §1.2): per-slot discriminability `D = |Spearman ρ|` between
the family's **own answer channel** at slot `t` and the query's target — scale-free, zero fitted
parameters, zero bytes, identical on all four arms. Slot grid pre-declared (`PREREG.md` §2).
**Quote the curve, not the endpoint:**

| slot | t | phase | **q**: full / div / **margin ± SE** | **p**: full / div / **margin ± SE** | clears |
|---|---|---|---|---|---|
| 0 | 0.05 | 1 | 0.559 / +0.291 / **+0.137 ± 0.065** | 0.559 / +0.291 / **+0.137 ± 0.065** | **q ✓ p ✓** |
| 1 | 0.45 | 1 | 0.560 / +0.288 / **+0.134 ± 0.065** | 0.566 / +0.287 / **+0.133 ± 0.057** | **q ✓ p ✓** |
| 2 | 0.85 | 1 | 0.579 / +0.297 / **+0.145 ± 0.053** | 0.611 / +0.321 / **+0.167 ± 0.057** | **q ✓ p ✓** |
| 3 | 1.25 | 1 | 0.611 / +0.321 / **+0.167 ± 0.057** | 0.700 / +0.402 / **+0.249 ± 0.046** | **q ✓ p ✓** |
| 4 | 1.65 | 1 | 0.646 / +0.356 / **+0.201 ± 0.051** | 0.649 / +0.351 / **+0.219 ± 0.093** | **q ✓ p ✓** |
| 6 | 2.45 | 1 | 0.766 / +0.472 / **+0.345 ± 0.023** | 0.395 / +0.087 / −0.060 ± 0.247 | q ✓ |
| 8 | 3.25 | 1 | 0.853 / +0.552 / **+0.403 ± 0.024** | 0.261 / −0.019 / −0.156 ± 0.155 | q ✓ |
| 12 | 4.85 | 1 | 0.940 / +0.657 / **+0.519 ± 0.077** | 0.725 / +0.539 / **+0.422 ± 0.134** | **q ✓ p ✓** |
| 16 | 6.45 | 1 | 0.968 / +0.715 / **+0.590 ± 0.138** | 0.338 / +0.088 / −0.113 ± 0.182 | q ✓ |
| 24 | 9.65 | 1 | 0.973 / +0.780 / **+0.654 ± 0.124** | 0.323 / −0.059 / −0.356 ± 0.361 | q ✓ |
| 32 | 12.85 | 1 | 0.973 / +0.710 / **+0.513 ± 0.107** | 0.307 / −0.020 / −0.350 ± 0.285 | q ✓ |
| 40 | 16.05 | 1 | 0.973 / +0.664 / +0.390 ± 0.225 | 0.149 / −0.063 / −0.294 ± 0.100 | — |
| 49 | 19.65 | 1 | 0.959 / +0.633 / +0.305 ± 0.288 | 0.345 / +0.157 / +0.015 ± 0.077 | — |
| 54 | 21.65 | 2 | 0.954 / +0.610 / +0.276 ± 0.304 | 0.437 / +0.183 / +0.033 ± 0.138 | — |
| 62 | 24.85 | 2 | 0.954 / +0.610 / +0.279 ± 0.337 | 0.211 / −0.075 / −0.360 ± 0.238 | — |
| 74 | 29.65 | 2 | 0.959 / +0.609 / +0.301 ± 0.294 | 0.266 / −0.102 / −0.445 ± 0.237 | — |
| 99 | 39.65 | 2 | 0.973 / +0.597 / +0.261 ± 0.290 | 0.261 / −0.093 / −0.416 ± 0.495 | — |
| 124 | 49.65 | 2 | 0.957 / +0.578 / +0.230 ± 0.306 | 0.149 / −0.016 / −0.167 ± 0.272 | — |
| 149 | 59.65 | 2 | 0.965 / +0.748 / **+0.523 ± 0.072** | 0.331 / +0.004 / −0.283 ± 0.356 | q ✓ |

`margin = (full − settle-deleted launder) − launch-noise floor`; clears ⇔ `mean − 2·SE > 0`,
`SE = sd/√3`, sample sd `ddof=1`, n = 3 seeds.

### 2.1 ⭐ What the curve says, mechanistically (three findings, one of them against the Advisor)
- ⭐⭐ **The store acts from the first step — CONFIRMED, and it is measurable at `t = 0.05` (ONE
  integrator step).** `D = 0.559` against a store-deleted launder of 0.268 and a launch-noise floor of
  0.154. §A8.1's core claim is right: `∇V` **is** the store, and it enters the trajectory immediately.
- ⛔ **But "p at small t is almost pure store, q at small t is almost pure query" is REFUTED as stated,
  and the reason is exact, not statistical.** At the first slots **q and p carry *identical*
  information — the same `|ρ|` to four decimals, the same margin to the last digit** (slot 0:
  0.5590 both). Mechanism: the shipped read launches at **`p₀ = 0` exactly** and with the **payload
  channels of `q₀` zeroed**, so Verlet gives `q₁ − q₀ = dt·p₁/m` — the first *position* increment is a
  positive multiple of the first *momentum*, hence rank-identical. **Position at small `t` is exactly
  as store-attributable as momentum, in the answer channel, because the answer channel of the launch
  is identically zero.** The q/p distinction the charter reasoned from does not exist here.
- ⭐ **Where the distinction DOES exist is the address block, and there the Advisor is exactly right**
  (⚠ **post-hoc secondary diagnostic**, declared as such, added after the pre-registered curve, does
  **not** enter the bar): leave-one-out nearest-prototype **item-identity decode of the address
  block**, 3 seeds, is **q: 1.000 full vs 1.000 launder ⇒ dividend exactly 0.000 at EVERY slot**
  (0.972 vs 1.000 from slot 24 on) — the address block of position is **pure query, at all `t`, not
  just small `t`**, which is *stronger* than §A8.1 claimed. And momentum's address block is
  **0.319–0.556 full vs 1.000 launder** (dividend −0.44 … −0.68): the written store *destroys* query
  identity in `p` because `p ∝ (c_i − q)` collapses every query of an item onto one vector.
- **The p channel dies; the q channel saturates.** `D_p` peaks at `t ≈ 1.25` (0.700) and at
  `t ≈ 4.85` (0.725), then decays to 0.15–0.33 (momentum → 0 at the settled point); `D_q` climbs
  monotonically to **0.97** and stays there — i.e. the "trajectory" adds nothing after the address
  settle that the settled point does not already have, which is D2a from a third angle.

## 3. ⭐ D1 (§A8.2) — THE FLOW-MAP JACOBIAN, both curves, 3 seeds, `overload`

*"Encoder controls whether trajectories diverge or coincide" = supervising the flow map's Jacobian:
**contractive within an item's launch cloud** and **separated across items**.* Both hold, measured:

| slot | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 40 | 49 | → 149 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **q contraction** `‖Δs_t‖/‖δ‖` | 0.999 | 0.906 | 0.736 | 0.701 | 0.654 | 0.495 | 0.421 | 0.371 | 0.367 | 0.267 | 0.226 | 0.220 | 0.218 | **0.218** |
| **q separation** (between-item, ÷ launch) | 1.000 | 0.989 | 0.978 | 0.984 | 1.009 | 1.043 | 1.011 | 1.098 | 1.161 | 1.158 | 1.153 | 1.153 | 1.153 | **1.153** |
| **q Fisher** between/within | 6.3 | 7.0 | 9.4 | 10.0 | 11.0 | 16.7 | 17.9 | 21.2 | 45.2 | 146 | 648 | 3.2e3 | 2.3e4 | **1.47e7** |
| **p contraction** | 0.064 | 0.478 | 0.667 | 0.601 | 0.555 | 0.542 | 0.451 | 0.410 | 0.271 | 0.144 | 0.031 | 0.006 | 0.001 | **0.000** |
| **p Fisher** | 2.65 | 2.63 | 2.60 | 3.15 | 4.01 | 2.08 | 3.01 | 3.69 | 1.83 | 1.67 | 1.62 | 1.71 | 1.63 | **1.23** |

⭐ **Both §A8.2 properties hold simultaneously and neither is marginal:** the flow map is contractive
along launch-cloud directions **from the first slot** (gain < 1 everywhere, plateau **0.218** = 4.6×
noise suppression) while between-item separation *grows* (+15 %), so the **Fisher ratio rises six
orders of magnitude, 6.3 → 1.47e7**. This is the medium a stage-2 objective would have to shape, and
it is *already* in the productive regime **without any slotted objective** — which is part of why the
per-slot table reproduces the read (§4): a contractive, item-separated flow map is exactly a map whose
per-slot output is a function of item identity.

## 4. ⛔⛔ D3 — THE §A9.5 PER-SLOT TABLE LAUNDER, BUILT IN STAGE 1, AND IT FIRES

Object: at each slot × channel, **K = 6 rows** holding each item's mean slot content, keyed by the
query's **nearest stored key**, evaluated **leave-one-query-out**, scored with the *identical*
instrument. Bytes: `K × 4 B` per slot — **strictly cheaper than the store (478.20×)**, which is what
makes reproduction fatal rather than merely unimpressive.

| | q-channel | p-channel |
|---|---|---|
| slots where the table **reproduces** the read (`read − table ≤ 0.05`) | **19 / 19 (100 %)** | **18 / 19 (94.7 %)** |
| slots where the **read beats** the table by > 0.10 | **0** | **0** |
| best `read − table` margin anywhere | **+0.0141** | **+0.0529** |
| worst (table beats read) | −0.0235 | **−0.2044** |
| on the **18 §A9.4-clearing slots** | 18/18 reproduced, **0** read-beats | — |

**Verdict: FIRES.** Every slot at which the store-attributable discriminability cleared the unlock bar
is *reproduced by six time-indexed numbers*. Per intervention §8.2, verbatim: **"Route 3 has
degenerated into K time-indexed lookup tables and FAILS REGARDLESS OF DIVIDEND."**
⚠ **It fires for a structural reason, and the structure is measured in §3, not assumed**: a flow map
that is contractive within the launch cloud and separated across items maps *all* queries of an item
onto one per-slot value — so the per-slot content **is** a function of item identity, and K rows
express it exactly. The escape §A9.5 demands (**inter-slot / dynamical coupling no per-slot table can
express**) is not present in the merged rig, and nothing in stage 1 suggests a slotted *write*
objective would create it — the coupling would have to be between slots, not within them.
**This is the cheapest possible place to learn it, which is the whole reason it was built in stage 1.**

## 5. ⭐ D4 — THE ROUTE-1 × ROUTE-2 2×2 (C2W2's funded declared NOT-RUN)

⛔ **Both blocking bit-identity gates PASS**, and gate (b) was re-run in the configuration C2W2 never
ran (**with the path term ON**):

| gate | result |
|---|---|
| **(a) coefficient-zero** — at `λ_traj = λ_path = 0` the written `V` is bit-identical to `main`'s | **PASS**, `tests/test_traj_write_objective.py` 16/16 green on this branch (both terms sit behind a Python-level `>0` branch; not one extra op traced) |
| **(b) r = 0** — the shell reduces to the Gaussian bit-identically | **PASS, 4/4 cell pairs, every score key** (`overload` 9/9 keys × 2 write objectives; `manifold` 5/5 × 2), **including under `path@0.3`** |

**The card** (`overload/load1x_shipped`, 3 seeds, **3/3 admissible in all four cells**; dividend ± SE):

| | **endpoint write** (control) | **path write @ λ_path = 0.3** |
|---|---|---|
| **Gaussian atoms** | **−0.0278 ± 0.0139** (the shipped anchor) | **−0.2917 ± 0.0636** (Route 1 alone) |
| **shell atoms** (learned `r`) | **−0.0833 ± 0.0417** (Route 2 alone) | ⭐ **−0.4722 ± 0.0278** (**the unrun cell**) |

⭐ **The rig is bit-faithful across three branches:** `gauss × endpoint` reproduces C2W2 Route 1's
`endpoint_write` control **digit for digit** (−0.0278 ± 0.0139) and `gauss × path@0.3` reproduces its
`path_write@0.3` **digit for digit** (−0.2917 ± 0.0636), on a *different* branch through a
*monkey-patched* builder. That is the strongest available check that the 2×2 measures what it claims.

**Findings:**
- ⛔ **The unrun cell is the worst cell.** Interaction = `(shell,path) − (shell,end) − (gauss,path) +
  (gauss,end)` = **−0.1250**: the two routes are **sub-additive**, i.e. combining them costs *more*
  than the sum of their separate costs. The hypothesis that an objective which can see a path makes
  the designed degeneracy pay is **refuted in sign**.
- ⭐ **P7 CONFIRMED and the mechanism is sharper than "the design was ignored".** Learned shell radius:
  **0.49854 ± 0.00025** (endpoint) → **0.50252 ± 0.00101** (path). The path term *does* move the
  radius — by **+0.0040**, 16 SE of the endpoint arm's spread, so it is **visible** — but that is
  **12× below** my registered "the objective can see it" bar of 0.05, while the *same* term costs
  0.209 of dividend and 0.9–1.1 of `λ_min` (3.30/3.01/3.41 → 0.90/0.82/1.19 on Gaussian atoms).
  ⇒ **the path term is not blind to the radius; it is ~50× more effective at damaging the write than
  at shaping the design.** That is a strictly more useful statement than C2W2's "0.500 → 0.501".
- `manifold`: **0/3 admissible** under the Route-1 rule in 3 of the 4 cells (1/3 in `shell × endpoint`);
  under Route 2's own *spectral* convention (`λ_min ≥ −1e-3`) it is `gauss×endpoint` 2/3,
  `shell×endpoint` 1/3, **both path cells 0/3** (λ_min −6.36/−2.62/−5.25 and −0.93/−0.43/−1.84).
  ⇒ the `manifold` half of the 2×2 **ABSTAINS on either convention**, and the path term is what
  drives it inadmissible. Both coverages are reported rather than the flattering one chosen.
- ⛔ **The trajectory launder FIRED on 2 of 24 cells** — `manifold/shell/path@0.3` seeds 0 and 2,
  where `q0_only = full` **exactly** (0.0244 and 0.0209) against bars of 0.0035 / 0.0032 and a blank
  store at −0.0001: *the whole (tiny) manifold read is reproduced by the launch alone.* **Both cells
  are already inadmissible** (λ_min −0.93, −1.84). No ψ number from that arm is quotable; **no
  admissible cell is affected** — 22/24 clear, `overload` 12/12 (`q0_only` 0.1667 = chance against a
  bar of 0.3949). (This is also how reconciliation 1 above was found.)
- **Determinism check:** the whole 2×2 was run **twice** (before and after the trajectory-launder
  mapping fix) and every aggregate reproduced to the last digit.
- ⛔ **Byte ledger:** `overload` **478.20×** (Gaussian) / **546.40×** (shell), `manifold` **52.00×** /
  **58.40×** — the shell's `1/(dim+2)` surcharge, **+12.5 % / +14.3 %**, reproduced exactly. Every
  cell `matched = False`. ⛔ **No cell here is quotable as a byte-matched dividend.**

## 6. ⭐ D5 — `AttentionPsi` QUARANTINED (reconciliation 1, charter §A11 rider)

`AttentionPsi` now **refuses** to produce a trajectory reading unless it is handed a **passing
`LeakProbe`** — it **raises `AttentionPsiLeakError`, it does not warn** (the `PhiMismatchError`
precedent: an invariant enforced in prose is not enforced). The C2W2 evidence (`q0_only`
0.3515–0.4480, `blank_store` 0.3713–0.4728, bar **0.1902**, fired at **every** stride, full 0.6386–
0.6658) is **in the exception's docstring**, so the next person reads why before they reach.

- **Default-off / additive / bit-identical:** the guard is a pure precondition check.
  `AttentionPsi(spec, key, quarantine=False)` and a probe-cleared instance produce **bitwise identical
  outputs and identical parameters** (blocking test `test_the_quarantine_is_bit_identical_shipped_
  behaviour_when_disabled` — `np.array_equal`, not `allclose`).
- **Scope is exactly the leaking configuration:** `input_mode="trajectory"` only. `settled_point` and
  `endpoints` are untouched, and **`DeepSetsPsi` is NOT quarantined** — its own launder did not fire
  (C2W1 `q0_only` 0.129 vs chance 0.125). Quarantining both would have been a policy, not a
  measurement.
- ⚠ **This does not bar `bprime-fb4-gate`'s `attention` arm.** That is a **table reader** over the
  launder's own (key, payload) rows: it never sees a trajectory, cannot select `q₀` out of a buffer it
  is not given, and is a different object. The distinction is written into the exception's docstring
  and asserted in a test, so it cannot be conflated at the wave review.
- Nothing else in `psi_readout.py` changed.

## 7. PREREG SCORECARD (`PREREG.md`, 8 substantive predictions: **4 survive · 3 fail · 1 partial**)

| # | prediction | outcome | verdict |
|---|---|---|---|
| P1 | `D_p(j=0) ≥ 0.80` and `D_q(j=0) ≤ 0.20` (the Advisor's q/p split) | **0.559 and 0.559 — identical** | ⛔ **FAILS**, with an exact mechanism (§2.1) |
| P2 | `D_q` crosses 0.5 between j=8 and j=49; ≥0.85 by j=99 | crosses 0.5 by **j=0** (0.559); 0.973 at j=99 | ◐ **PARTIAL** (right ceiling, wrong onset — same cause as P1) |
| P3 | `unlock = true`, driven by p at small `t` on `overload` | **true**, p **and** q at small `t` | ✅ **SURVIVES** |
| P3′ | registered alternative: both channels collapse to ≈0 | did not happen | — |
| P4 | contractive within the cloud (`w<1`), separated across items; crossover j≈16–32 | contraction < 1 **from j=0**, plateau **0.218** at j≈32; Fisher ×2.3e6 | ✅ **SURVIVES** (plateau at the predicted point) |
| P5 | **§A9.5 fires**: table within ±0.05 at ≥80 % of slots, read never beats by >0.10 at a clearing slot | **97 % of slots (37/38)**, read-beats **0/38** | ✅ **SURVIVES** — and it is the wave's decisive result |
| P6 | both bit-identity gates pass | 16/16 tests; 4/4 r=0 cell pairs incl. under the path term | ✅ **SURVIVES** |
| P7 | `\|Δr\| < 0.01` on `shell × path`, dividend ≤ 0 and within 0.05 of `shell × endpoint` | **Δr = +0.0040** ✔; dividend **−0.4722** vs −0.0833 — **0.389 apart, not 0.05** | ⛔ **FAILS on the second clause** — the term is far *more* destructive than predicted |
| P8 | `gauss × endpoint` reproduces the unpatched gym digit-for-digit | reproduces **C2W2 Route 1's** control digit-for-digit (−0.0278 ± 0.0139), and `gauss×path@0.3` too | ✅ **SURVIVES** (stronger form) |

*A prediction that survives is evidence; one that fails is a finding.* P1's failure is the report's
main scientific content after P5.

## 8. Flag provenance (every quantitative result above)

| item | value |
|---|---|
| commits | `8fe8b36` (D5 quarantine) · `96a1abd` (`eval/attribution.py`) · `9e188cb` (`exp_route3_attribution.py`) · `7c06158` (2×2 seam + runner) · `19cbbed` (traj-launder mapping fix) |
| base / branch / worktree | local `main` **`6ff4c1d`** · `agent/experiment-engineer/route3-stage1-plus-2x2` · `../CHLU-route3` |
| env | **main venv reused** (protocol §4), **JAX 0.9.0**, no worktree `uv sync`; `chlu.__file__` verified inside the worktree |
| seeds | **{0, 1, 2}** on every table above, no exceptions |
| sd convention | **sample sd, `ddof=1`; SE = sd/√n**; "clears" ⇔ `mean − 2·SE > 0` |
| families / arms | `overload/load1x_shipped` (`atoms_per_item=341`, `min_atoms=2046`, `n_offer=capacity=budget=6`, `d_safe_override=0.58`, `stage_admission=True`) · `aggregate/base` · `manifold/base` |
| store / read | `addr_dim=4`, `payload_dim=1`, `dim=6`, `atom_width=0.3`, `confine=0.05`; `dt=0.05`, `gamma_address=0.05`, `gamma_read=0.02`, `address_steps=400`, `read_steps=800`, **`traj_stride=8` ⇒ 150 slots**, `query_sigma=0.15`, `kinetic_mode=newtonian_learned`, `retries=0` on every cell |
| write | `write_steps=300` (escalation **900**), `lr=3e-3`, `weight_decay=1e-4`, `sigma_addr=0.25`, `sigma_pay=0.6`, `margin=0.15`, `barrier=0.2`, `masked_write=True` |
| **Route-3 instrument** | `D = \|Spearman ρ\|` (average ranks); slot grid `{0,1,2,3,4,6,8,12,16,24,32,40,49,54,62,74,99,124,149}`; launder = **store-deleted** system (`seed+991`, the gym's own blank control); floor = store-deleted + **independent** `N(0, σ_q=0.15)` launch re-draw; table launder = **K=6 rows, nearest-key, leave-one-out** |
| **2×2** | stores `{gauss, shell}` (`shell` = **learned** radius, `radius_scale=1.0`, `r_init=0.5`, `tilt_eps=0`), writes `{None, {loss_kwargs:{lambda_path:0.3, path_kwargs:{n_interp:7}}}}`; r=0 gate arms `{gauss, shell_r0}` at seed 0 |
| byte ledger | `overload` **478.20×** / **546.40×** (shell) · `manifold` **52.00×** / **58.40×** — all `matched=False`, all **architectural**; ⛔ none quotable as a byte-matched dividend (min ratio anywhere 17.11×) |
| φ | identity/embedded, **0 B, identical on every arm and every launder** (the gym embeds the address directly; no learned φ in this family set) |
| wall-clock | attribution 15 cells ≈ **9 min**; 2×2 24 cells + 8 gate cells ≈ **7 min**/run (run twice) |

## 9. How I verified (commands + observed output)

All as `PYTHONPATH=/Users/user/Desktop/CHLU-route3 /Users/user/Desktop/CHLU/.venv/bin/python …`, cwd in
the worktree.

```
-m pytest tests/test_psi_readout.py -q --no-cov          -> 25 passed in 14.79s   (was 18)
-m pytest tests/test_attribution.py -q --no-cov          -> 14 passed in  3.29s   (new)
-m pytest tests/test_route3_attribution.py -q --no-cov   ->  7 passed in 19.96s   (new)
-m pytest tests/test_ssb_shell.py -q --no-cov            ->  7 passed in  4.28s   (new)
-m pytest tests/test_traj_write_objective.py -q --no-cov -> 16 passed in 31.29s
     (incl. the two BLOCKING gate tests, green on this branch:
      test_coefficient_zero_writes_a_bit_identical_landscape,
      test_coefficient_zero_leaves_the_loss_value_bit_identical)
-m pytest -q --no-cov                                    -> 1019 passed, 0 failed, 949.98s
ruff check chlu/ tests/                                  -> All checks passed!
-m chlu.experiments.exp_route3_attribution --seeds 0 1 2 -> unlock=True, §A9.5 fires
python -c "from chlu.experiments.exp_ssb_shell import run_2x2; run_2x2(...)" -> r0_gate_passed True
```

### 9.1 Full suite
**1019 passed, 0 failed** (949.98 s) on the branch. Arithmetic closes with no unexplained tests:
**984** (at scoping) **+ 35** (this branch: `psi_readout` 18 → 25 = +7 · `attribution` +14 ·
`route3_attribution` +7 · `ssb_shell` +7) = **1019 exactly**. `ruff check chlu/ tests/` clean.
Rebased onto local `main` `6ff4c1d`: **already up to date** (no rebase needed, no conflicts).

Artifacts under `.claude/outputs/route3-stage1-plus-2x2/`: `PREREG.md` · `attribution_curve.json`
(all three families incl. every exclusion with its reason) · `attribution_curve_overload.json`
(+ the post-hoc address-block diagnostic) · `exp_route3_attribution.png` (4 panels: the margin curve
with ±2 SE, the raw curve, the §A9.5 margin, the two Jacobian curves) · `two_by_two.json`
(24 cells + 8 r=0-gate cells + 24 `RaceCell`s in the frozen schema).

## 10. Git footprint

Branch `agent/experiment-engineer/route3-stage1-plus-2x2` (worktree `../CHLU-route3`), base local
`main` `6ff4c1d`. **Not pushed, no PR, left for Hub review.** Verified from the MAIN repo
(`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/route3-stage1-plus-2x2`)
that all five commits are on the shared ref — the w4 lesson. ⚠ **The worktree `../CHLU-route3` is
deliberately LEFT IN PLACE**: charter §A9.3 has `route3-stage2` *share* it, so removing it would cost
the wave a slot re-take. If the Hub accepts my recommendation not to spawn stage 2, the worktree is
free to remove (the ref is already verified).

| commit | what |
|---|---|
| `8fe8b36` | D5 — `AttentionPsi` quarantine + 6 tests (raise-not-warn, bit-identity when disabled) |
| `96a1abd` | D1/D2/D3 — `chlu/eval/attribution.py` (new) + 14 tests |
| `9e188cb` | D1 — `chlu/experiments/exp_route3_attribution.py` (new) + 7 tests |
| `7c06158` | D4 — the 2×2 seam fix + `run_2x2` in `exp_ssb_shell.py` + 7 tests |
| `19cbbed` | reconciliation 1 — `to_race_cell` now carries the mandatory trajectory launder |

Files touched: `chlu/core/psi_readout.py` · `chlu/eval/attribution.py` (new) ·
`chlu/experiments/exp_route3_attribution.py` (new) · `chlu/experiments/exp_ssb_shell.py` ·
`tests/test_{psi_readout,attribution,route3_attribution,ssb_shell}.py`.
**Every file is inside my §6 ownership list. No file outside it was touched** — in particular
`clu_system.py`, `race.py`, `config.py`, `dividend.py`, `exp_memory_gym.py`, `memory_gym.py`,
`train_memory.py` and `cli/experiment_cmd.py` were **imported/run, never edited**. No conflicts.
⚠ `exp_traj_write.py` (mine this wave) needed **no change** — the 2×2 consumes its seam from the
Route-2 side.

## 11. ⛔ DECLARED NOT-RUNs (never to be reported as nulls)

- **The read-length write rollout (400–1200 steps)** — ⛔ explicitly **retired** by §A7/§A9.1, not
  owed. I did not size it and did not run it.
- **`aggregate`/`manifold` attribution curves as results** — measured and *emitted* (all 38 rows per
  seed are in the artifact), but **0/3 admissible after the one bounded escalation**, so they
  **ABSTAIN**: they are exclusions with reasons, not nulls.
- **A second escalation** beyond `write_steps = 900` — the ruling grants exactly one.
- **`λ_traj` in the 2×2** — only `λ_path` was crossed with the shell (the hypothesis on record is
  about the *path*; a `λ_traj` column is a cheap follow-up, ~7 min, and is declared unrun).
- **Any learned ψ, including `AttentionPsi`** — the instrument is a fixed rank correlation; no ψ was
  trained anywhere in D1–D4.
- **`recency`** — not in the Route-1/Route-2 family set and its scoring-domain fix is default-off.

## 12. Open questions / risks

1. ⭐ **The unlock is real arithmetic but its small-`t` end is a hair's breadth.** The clearing slots
   at `t ≤ 1.65` have lower-2SE bounds of **+0.004 … +0.099** on n = 3; the robust clears are q-slots
   at `t = 2.45 … 12.85` (lower bounds +0.30 … +0.41), which are just "the read works". If the Hub
   ever needs the small-`t` claim to bear weight, it needs more seeds — but §A9.5 makes that moot.
2. **The bar rests on one family.** `aggregate`/`manifold` abstain for the *same* write-side reason
   they abstained in C2W2, at 3× the write budget. That is a store-expressivity fact, not a Route-3
   fact, and it bounds what stage 1 could ever have shown.
3. **My per-slot "settle-deleted launder" is a declared mapping, not the shipped object.** The shipped
   `settle_deleted_launder` is a settled-point table with no slot index; I used "delete the store that
   creates the settle" (the harness's own blank-store control). If the Advisor intended the *table* at
   every slot, that object is precisely the §A9.5 launder — and it fires. **Both readings kill stage 2**,
   which is why I am comfortable reporting the verdict as decisive.
4. **`vacuous_gate` trips on 24/24 2×2 cells** (and `addressing`/`lifetimes`/`starvation` on the path
   arms). Inherited from the shipped gym, not caused here, but the path term visibly increases the
   trip set — worth a monitor-owner's eye.

## Proposed handover updates (for the Hub)

- **§2 architecture:** `chlu/eval/attribution.py` (**new**) = Route 3 stage 1's instrument (per-slot
  store-attribution curve, launch-noise floor, §A9.5 per-slot table launder, §A8.2 Jacobian curves,
  §A9.4 bar arithmetic). `chlu/experiments/exp_route3_attribution.py` (**new**) = its runner.
  `exp_ssb_shell.run_2x2` = the Route-1 × Route-2 2×2.
- **§3 config/CLI:** `AttentionPsi` gains `quarantine: bool = True` and `leak_probe: LeakProbe|None`
  (**shipped behaviour bit-identical when disabled**); `make_psi`/`matched_pair` forward both.
  `run_shell_cell` gains `write_objective` (default `None` = the shipped write, bit-for-bit).
  ⚠ **No CLI hook was added** — `cli/experiment_cmd.py` has no owner this wave (reconciliation 2).
- **§7 Known Issues — ADD (RESOLVED):** *"`AttentionPsi` trajectory reads leak φ(x)"* — now
  **quarantined in code, raises**. **ADD (RESOLVED):** *"the shell rig swallowed the write-objective
  seam, which is why the 2×2 was a NOT-RUN"*. **ADD (OPEN):** *"`exp_ssb_shell.to_race_cell` dropped
  the mandatory trajectory launder ⇒ C2W2 route2 cells report `fired=False` by absence"* — fixed
  going forward; the C2W2 card's column is unmeasured, not clear.
- **NOT-RUN list:** the C2W2 entry *"the Route-1 × Route-2 2×2"* is **CLOSED** (run, §5). The
  *"read-length write rollout"* stays **retired**, per §A7. New declared NOT-RUN: the `λ_traj` column
  of the 2×2.
- **Test count:** full suite **984 → 1019** (+35: `psi_readout` 18→25, `attribution` +14,
  `route3_attribution` +7, `ssb_shell` +7).
- ⭐ **For Addendum 3, if the Advisor wants one sentence from this task:** *the store does act from the
  first step, and both channels carry it — but on the merged rig the per-slot content of a contractive,
  item-separated flow map is a function of item identity, so K time-indexed rows express it, and the
  headline claim §A9.5 demands (inter-slot dynamical coupling) does not exist to be measured yet.*
