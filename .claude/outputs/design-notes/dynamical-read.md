# DESIGN NOTE — the dynamical read (R40.C / R40.D)
**Hub, 2026-09-14. ⛔ A DESIGN FOR REVIEW, NOT A BUILD ORDER.**
⭐⭐ **REVISED after `gen-variance-equalisation` and `gen-encoder-pretrain` landed. §0.5 is new and it changes the
central argument: the arm's justification is no longer "unaddressed wells" — it is a MEASURED scaling law.** Nothing here is implemented. R40.D: no new mechanism
is built before Head + Advisor have seen its design.

---

## 0. ⛔ THE BINDING CONSTRAINT (Advisor, 2026-09-14) — it removes the obvious design
> **The landscape must NOT take the `M` k-means centres as its minima**, or the dynamical read **inherits the codebook
> loss**. Build it over the **written atoms** (density / kernel form, wells emergent), **or** a **much larger
> codebook** with the byte ledger **re-declared**.

⭐ **Why this is right, and why it kills the cheapest design.** `gen-loss-attribution` measured the loss as **90.5 %
`scale` / 100 % `orientation` at quantisation**, and the partition as exonerated. A landscape whose minima *are* the
64 centres per well can only ever settle **onto those centres** — so the settled state carries exactly the
information the centres carry. ⛔ **A continuous read over discrete minima is still a discrete read.** The
dynamics would be decoration.

---

## 0.5 ⭐⭐ THE SCALING LAW — NEW, AND IT REPLACES THIS ARM'S JUSTIFICATION
`gen-variance-equalisation` measured the thing that explains everything: **the well features have effective dimension
`PR = (Σλ)²/Σλ²` of only 3.12 … 13.63**, against nominal widths `d_k` = 283 … 744. High-rate quantisation theory then
gives a distortion ratio **ε ≈ M^(−2/PR)**, and the spoke confirmed it **parameter-free** (predicted 0.253 vs measured
0.278 unwhitened; 0.979 vs 0.934 whitened).

### ⛔⛔ CORRECTION, 2026-09-14 — THE LAW SURVIVES AS A *LEVEL* AND FAILS AS A *RATE*
⭐ **`gen-codebook-allocation` measured the within-well exponent of `ε` in `M`: it is `−0.2631 ± 0.0555`**
⚠ **CORRECTED 2026-09-15 by `gen-dsprites-exponent` (7 `M` points, not 3): it is NOT "constant".** The exponent
depends weakly on `log PR` **within** each dataset (slope `+0.0702 ± 0.0136`, `r = 0.82`, `p = 2.1e−4`), running
`−0.287 → −0.200` over `PR = 3 → 13`. ⭐ **The headline survives — the law's `−2/PR` still overpredicts the RATE by
~3× (it would run −0.667 → −0.154) — ⛔ but "constant" is the wrong word and was recorded twice.** — against `−2/PR`, which would span **−0.642 … −0.147**, a 4.4×
range **the data does not show**. ⭐ **`gen-atom-mean-control` agrees independently** (across-budget slopes
**−0.2501 ± 0.0050** over centres, **−0.2727 ± 0.0141** over atoms).
⇒ ⭐⭐ **`ε ≈ M^(−2/PR)` holds as a LEVEL (pooled `r = 0.905`, median ratio 1.078) and is REFUTED as a RATE. The
exponent is universal, ≈ −0.26; the wells differ in LEVEL, not in slope.**
⛔ **Every table below that extrapolated `M = ε^(−PR/2)` is therefore WRONG — including the "6.5 million codewords"
row.** They are kept, struck, for the record. ⚠ The measurement is POST-HOC and covers `M ∈ [4, 186]` only; ⛔ it is
not itself extrapolable either.

**Restated under the measured constant exponent (`ε ∝ M^(−0.2631)`), to reach ε = 0.10:**
| well | ε at `M` = 64 | ⭐ `M` needed | bytes |
|---|---|---|---|
| best (`PR` 3.12) | 0.0695 | **already there** | — |
| typical (`PR` 6.06) | 0.2535 | ~2,200 | **34×** |
| worst (`PR` 13.63) | 0.5432 | ~40,000 | ⛔ **623×** |
⭐ **The qualitative conclusion SURVIVES — the hard wells are out of reach at any budget we would accept — but the
magnitude was overstated by ~160×.** ⛔ **Never quote "6.5 million codewords" again.**

### ⭐ Consequence 1 — the wells are NOT equally hard, and a uniform `M` is therefore the wrong design
| well `PR` | ε at today's `M = 64` |
|---|---|
| 3.12 (best) | **0.070** |
| ~6 (typical) | 0.250 |
| 13.63 (worst) | ⛔ **0.543** |

### ⭐⭐ Consequence 2 — NO FEASIBLE UNIFORM `M` FIXES THE HARD WELLS
`M` needed per well to reach a target ε (`M = ε^(−PR/2)`):
| target ε | `PR` = 3.12 | `PR` = 6 | `PR` = 13.63 |
|---|---|---|---|
| 0.278 (today's average) | 7 | 47 | ⛔ **6,149** |
| 0.10 | 36 | 1,000 | ⛔ **6.5 million** |
| 0.05 | 107 | 8,000 | ⛔ **7.4 × 10⁸** |

Uniform-`M` byte cost (`M · D · 4`): `M` = 512 → **8×** budget · `M` = 6,120 → **96×** · `M` = 8,870 → **139×**.
⇒ ⛔ **Candidate C ("just use a much larger codebook") is now quantitatively bounded: it rescues the low-`PR` wells
cheaply and CANNOT rescue the high-`PR` ones at any budget we would accept.**

### ⭐⭐⭐ Consequence 3 — this explains the whitening failure exactly, in one line
**Whitening sets `PR = d_k` by construction** (283 … 744). Predicted ε at `M = 64` is then **0.971 … 0.989**;
measured **0.934**. ⇒ ⭐ **whitening maximised the very quantity the distortion depends on.** *"k-means' reliance on
high variance is not a bug; it is the compression"* is now a **prediction that was tested and held**, not a slogan.

### ⭐⭐ Consequence 4 — THE ARM'S REAL JUSTIFICATION, and it is much stronger than the old one
⛔ §1 showed the stated reason (*"unaddressed wells take values through the coupling"*) is **nearly vacuous on a full
request** — only **0.58 of 5** wells are unaddressed.
⭐ **The measured reason is better: a continuous settled state pays a DIFFERENT AND BETTER EXPONENT.**
⛔ **STRUCK, on the Advisor's ruling — the earlier wording ("no `M` in its error term") was WRONG.** A landscape over
stored atoms still pays an error: **INTERPOLATION error along the factor manifold**. Nearest-neighbour quantisation is
piecewise-constant (error `~ N^(−2/PR)`); interpolation is higher-order (`~ N^(−4/PR)` for a second-order scheme).
⇒ ⭐ **the claim is a BETTER EXPONENT AT MATCHED BYTES, never zero error — and it must be MEASURED, not asserted.**
⭐ **The arm is the only proposed repair the law permits for the high-`PR` wells, which are provably beyond any
affordable `M`.** ⛔ **This replaces the "unaddressed wells take values through the coupling" sentence wherever the
arm is motivated — in this note, the claims matrix and every draft (Head ruling D1).**

### ⭐ A cheap candidate nobody has tried, which falls straight out of the law
**Per-well `M` allocation by `PR`** — spend codewords where the distortion actually is, instead of uniformly.
It is a **re-allocation at constant total bytes**, so ⛔ it needs **no** byte-ledger renegotiation. ⭐ It cannot save
the worst wells (consequence 2), but it is the cheapest test of whether the law is actionable, and it is **strictly
better-motivated than factor-aligned wells**, which the exonerated partition already de-prioritised.

## 0.6 ⭐ THE OTHER WAVE-8 RESULTS THE DESIGN DEPENDS ON
### ⭐⭐ The encoder is CLOSED as a candidate explanation — three independent ways
1. **It has the factors:** frozen `recon` encoder reads `scale` **0.9352 ± 0.0058**, `orientation` **0.9915 ± 0.0018**.
2. **Training it harder makes things WORSE:** at matched compute, unfreezing *lowers* the distinct-code count
   (**−6** / **−29**; request-side **7 → 1** of 72).
3. ⭐⭐ **An UNTRAINED random-init encoder of the identical architecture scores the same** — `scale`
   **0.9361 ± 0.0030**, `orientation` **0.9865 ± 0.0023** vs the trained control's **0.9408** / **0.9912**. The gap
   (**+0.0047**) is **smaller than the control's own seed spread**, for **1,246× the compute**.
⇒ ⛔⛔ **"≥0.9 on seen data" does not measure the pretext.** It measures that a 2,048-d random non-linear projection
of a 64×64 image is linearly decodable. ⭐ **A random encoder also carrying the factors makes *"the encoder is
innocent, the codebook is the culprit"* STRONGER, not weaker** — which is the premise this whole note rests on.
⛔ **Consequence that binds future work: every encoder bar is SATURATED and cannot discriminate. Any future bar must
be a MARGIN OVER THE D-MATCHED RANDOM-INIT FLOOR, never an absolute accuracy.** ⚠ R40.E's other clause — *swap within
the family* — is **still not measured by anyone.**

### ⭐ The one intervention that moved "distinct of 72" upward — and it is the CHEAP one
**Diagonal standardisation (16,384 B, 3.1 % of the store budget) took the distinct configuration count on the 72
held-out requests from `9.00 ± 0.82` to `15.00 ± 2.94`** — ⭐ **the first thing in this phase to move that number in
the right direction** — and lifted `orientation` retention 0.2425 → 0.2985.
⛔ **It is NOT the pre-registered pass**: `scale` did not move, and it **lost** `object_hue` (0.665 → 0.463) and
`shape` (0.555 → 0.423). ⛔ It must not be promoted to primary.
⚠ **But it is 3 % of budget and it moved the number the pivot named.** See **D7**.

## 1. ⭐ DREAMING'S AUTOPSY — read first, as ruled. One fact reframes the whole design.
From `gen-dream-sampler` (R25), Hub-checked against the report:
- ⛔ **The coupling WAS fitted** — an inter-well **PMI field over discrete codes** from 32,768 train codes,
  `coupling_strength = 1.0`, **+163,840 B**. ⭐ **"The coupling was never fitted" is NOT the correct autopsy** and must
  not be carried into the design as a premise.
- **Acceptance was 0.952.** At that acceptance the corrected and uncorrected chains are nearly the same chain: ⭐ **the
  chain barely moved per step.**
- The chain **sampled a mode** rather than settling; the `no_coupling` control took essentially all the apparent gain
  (steering-attributable **+0.0015 ± 0.0204 = ZERO**).
- ⭐⭐ **THE FACT THAT MATTERS MOST: the addressing head clamps `4.42 ± 0.03` of 5 wells, leaving `0.58` relaxed.**

### ⛔ The consequence the Head and Advisor must rule on BEFORE this is built
R40.C's stated reason for the arm is: *"unaddressed wells take values consistent with the addressed ones through the
coupling — this is the thing a codebook cannot do."*
⛔ **But on a FULL request there are only 0.58 unaddressed wells.**
⭐ **RULED (D1): full requests remain the HEADLINE cell, and the arm's job there is RESOLUTION INSIDE the addressed
wells** — the settled state carries the `scale`/`orientation` the codebook cannot. **Partial requests become a
PRE-REGISTERED SECONDARY cell with its own falsifier** (rival: `knn_proto` with the marginal mean for unaddressed
factors). ⇒ the old framing is superseded, not merely caveated. ⭐ **Where it bites is PARTIAL requests — which R26.3 deliberately banked
as "a DIFFERENT capability, secondary, not scored in this cell."**
⇒ ⭐ **Decision needed (D1):** either the dynamical read is scored on a cell that **includes partial requests**, or it
is scored where its own mechanism is mostly idle. ⛔ **Building before this is settled repeats dreaming's error** —
that spoke measured steering in a regime with almost nothing to steer.

---

## 2. Landscape form per well — three candidates, ⛔ the first is excluded by §0
| # | form | minima | verdict |
|---|---|---|---|
| ~~A~~ | ~~potential with the `M` k-means centres as minima~~ | the 64 centres | ⛔ **EXCLUDED by §0** — inherits the codebook loss |
| **B** | ⭐ **kernel-density / energy over the WRITTEN ATOMS** — `E_k(x) = −log Σ_i w_i · κ(x, a_i)` over the well's stored atoms, **wells emergent, minima not enumerated** | wherever the atoms put them | ⭐ **recommended primary** |
| C | a **much larger codebook** (`M' ≫ 64`) as minima | `M'` centres | permitted by §0 **only with the byte ledger re-declared** |

**Why B.** It is the only form whose resolution is not capped by a mode count: the settled state can land **between**
atoms, which is precisely what `scale` and `orientation` need — they were quantised away, not absent (the encoder
carries them at **0.9352 / 0.9915**).
**The cost of B, stated honestly:** the atoms must be retained. If the well keeps `N_k` atoms at width `d_k`, the
store becomes `Σ_k N_k · d_k · 4` bytes — ⭐ **linear in the number of atoms kept, and it must be pre-declared.**
At today's budget, 524,288 B buys **64 atoms per well** (exactly the current codebook). ⛔ **B is only a real change
if it keeps more atoms than that, so B implies a byte-ledger renegotiation. Say so up front rather than discovering it.**
⚠ **C is the honest fallback** and is cheaper to build; it tests "is it capacity?" directly.

## 3. ⭐ How the coupling is fitted — LABEL-FREE, and NOT the PMI field again
The old coupling was **PMI over discrete codes** — a table over the very quantisation we are trying to escape.
**Proposal:** fit a **continuous** inter-well coupling on the **train features only**, label-free:
`E_couple(x_1..x_K) = Σ_{k<l} xᵀ_k W_kl x_l`, with `W_kl` fitted by maximum-likelihood / score-matching on the joint
feature distribution. ⛔ **No factor labels anywhere** — this is the generative path.
**Byte cost:** `Σ_{k<l} d_k·d_l·4` = **6,592,368 B** at today's well sizes ⇒ ⭐ **12.6× the store budget. This is the
design's single biggest cost and it must be decided, not discovered.**
⚠ **Cheaper variants to price in the brainstorm:** low-rank `W_kl = U_kᵀV_l` (rank `r` ⇒ `Σ r(d_k+d_l)·4`), or
coupling only on a **shared low-dimensional projection**.

## 4. Launch and stopping rule
- **Launch:** particles initialised from the **address-book clamp on the addressed wells**; unaddressed wells
  initialised from the well's own marginal. ⭐ **The launch must be a function of request + store ONLY** — R40.C
  requires the request-only assertion to be **extended to cover the launch**, so ⛔ no feature tensor may enter it.
- **Steps: 1000** (R40.C, down from 4000).
- ⭐ **Stopping rule and its rider, binding:** a chain **still descending at 1000 steps is reported NOT CONVERGED**,
  ⛔ never as a null. **Log the energy trace every step and report the fraction of chains still descending.**
- ⚠ **Dreaming's acceptance was 0.952** — a chain that barely moves. **Report acceptance and per-step displacement
  from step 1**, or we will not know whether it settled or simply sat still.

## 5. Decoder interface
`settled continuous state → store→decoder at PROTOTYPE WIDTH (the CLU-wide decoder)`.
⛔ **Never encoder→decoder.** The existing `CLU-wide` arm already has this path and its mixture weights come from the
**request**; ⛔ `soft_read(store, X, τ)` takes **FEATURES** and using it voids the structural guarantee.
⭐ **Re-prove the bypass guarantee as an assertion over the WHOLE path — launch, settle and read** — and report the
realised channel width. ⚠ **The settled state is continuous, so `K·log₂M = 30 bits` no longer describes it.** The
channel must be **re-derived and re-declared**, ⛔ not quietly inherited.

## 6. ⛔ Falsifiers (R40.C, verbatim — 3 seeds)
1. Beats **`store_removed`** AND the **codebook-argmax read** on the generation table, **beyond seed spread**.
2. **≥ ~60 / 72 distinct settled configurations.**
3. **`scale` + `orientation` probe on the settled state above the codebook read's** (baseline: `scale` 0.225,
   `orientation` 0.243).
**Ceiling stays `knn_proto` on the same features (0.9119 external).**

## 7. Cost and artifact paths
| item | estimate |
|---|---|
| coupling fit (full `W_kl`) | ~1 day, dominated by the 6.3 MB fit |
| landscape + settle implementation | ~1 day |
| 3-seed scored run at 1000 steps | ~0.5 day |
**Artifacts:** `.claude/outputs/gen-dynamical-read/` — `PREREG.md`, `runs/<dataset>_<pretext>/seed_*/`,
`energy_traces_*.json`, `settled_states_*.npz`, `tables.md`.

## 8. ⛔ Open, for the brainstorm — the Hub will not decide these
- **D1 (blocking):** partial requests in the scored cell, or accept that the mechanism is idle on full requests? (§1)
- **D2:** landscape **B** (atoms, byte ledger renegotiated) or **C** (larger codebook)? ⭐ **§0.5 now bounds C: it
  rescues low-`PR` wells cheaply and CANNOT rescue high-`PR` ones at any acceptable budget.** ⇒ **Hub reads this as
  B-or-continuous for the hard wells, with C or per-well `M` allocation as the cheap partial fix.**
- **D3:** full `W_kl` at **12.6× budget**, or low-rank / projected coupling? (§3)
- **D4:** what replaces `K·log₂M` as the declared channel for a continuous settled state? (§5)
- ⭐ **D5 — ANSWERED, and it did NOT pre-empt the design; it sharpened it.** The total-retained line did **not** go
  flat: it **collapsed** (3.678 → 1.172), so the pre-registered capacity reading did not fire. What fired instead is
  §0.5's scaling law. ⛔ The old wording is kept below for the record only.
- ~~**D5 (superseded):** `gen-variance-equalisation` may pre-empt all of this — **if its total-retained line is flat and only the
  distribution moves, CAPACITY is binding**, and the repair is `M` (or the representation), ⛔ not the read. **This note should be re-read after that result lands**, exactly as ruled.~~

- ⭐ **D6 (NEW, cheap, untried):** **per-well `M` allocation by effective dimension** — spend codewords where the
  distortion is, instead of uniformly. ⭐ **A re-allocation at CONSTANT TOTAL BYTES, so it needs no ledger
  renegotiation**, and it is strictly better-motivated than factor-aligned wells. ⛔ It cannot save the high-`PR`
  wells (§0.5 consequence 2), but it is the cheapest test of whether the law is actionable. **Spoke before anything
  dynamical is built?** (§0.5)
- ⭐ **D7 (NEW):** the **16,384 B diagonal arm doubled distinct-of-72 (9 → 15)** without being the pre-registered
  pass. **Worth a follow-up arm at 3 % of budget, or dropped?** (§0.6)

## 9. Riders
- **Factor-aligned wells: SECONDARY rider only** — the partition is exonerated, so it addresses **≤ 10 %** of the
  `scale` loss and **none** of the `orientation` loss.
- ⛔ **No draft blames 30 bits.** The aggregate code is *sufficient* (30 bits vs 18.87 needed); the binding
  constraint is **per-well `M`**, and distinctness is not the limiter (3,598 distinct codes on 4,096 rows).
