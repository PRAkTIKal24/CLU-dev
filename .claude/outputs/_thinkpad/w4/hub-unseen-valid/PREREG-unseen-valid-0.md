# PREREG-unseen-valid-0 — scoreboard ROW 2, in-basin hue interpolation

**Written 2026-09-18 22:50Z, BEFORE the first DEV number of this hub.** Nothing below was chosen after seeing a
result. Amendments are appended and dated, never edited (charter §A.3).

Hub `hub/unseen-valid` · branch off pin `4125fce` + `w3-value-holdout` · target file `hub-unseen-valid.md`.
Scored budget: **fresh target, 3 runs, 0 spent at the time of writing.**

---

## 0. Why this file has to exist
Row 2 has never been measured. My target file's words: *a first measurement on an unmeasured row with no
registered expectation is unfalsifiable — whatever comes out will look like the answer.* So the expectation, the
bar, the metric, the corner structure, the stop rule and the abandon rule are all fixed here, in advance.

## 1. The question, stated so it can fail
Does a basin carry a **continuous coordinate** that can be addressed **between** its stored values?
Operationally: request `object_hue = 0.05` when the store holds 0.0 and 0.1 (both bracketing rows in its fit pool),
render through the frozen decoder, and read the hue back with the frozen external judge.

## 2. THE EXPECTED DIRECTION — registered before any number

**I expect the arm to TIE the control, and I expect the control to be strong.**

Reasoning, from the known facts I am forbidden to rediscover and from what I read in the code today:
- w3-common §1: on the basis+head object **the head does the work and the store is a compression basis**; medoid
  atoms tie fitted atoms exactly. That is a direct prediction that the head, not the store, supplies interpolation.
- w3-common §4.4/§4.5: the read is a **signed linear span** over modes and the modes **already span the answer**.
  A signed linear span is exactly the hypothesis class the control's linear head already has. Two objects with the
  same hypothesis class should interpolate equally well.
- LOG FACT B: `FactorPrototypes.prototype` **already** linearly interpolates off-grid values. The control is an
  interpolator by construction, declared in R8, and not something I get to weaken.

**Quantitative registration.** Primary metric `HUEMAE` (§3). Let `D = HUEMAE(control) - HUEMAE(arm)`, paired over
corners, in units of the grid spacing Delta = 0.1. My registered prior, before looking:

| outcome | registered probability |
|---|---|
| **tie**, abs(D) < 0.2 Delta | **0.55** |
| arm wins, D >= 0.2 Delta | **0.20** |
| arm loses, D <= -0.2 Delta | **0.25** |

=> **My modal pre-registered prediction is that this hub reports row 2 as WON BY THE HEAD, and reports the CLU
mechanism as decoration in those words** (w3-common §1). I am registering that now so that if it happens I cannot
present it as a surprise, and if the arm wins instead, the win is against a stated 0.20 prior.

**Second registered prediction — the SNAP.** I expect the *shipped-style* discrete-code CLU read to **snap** to a
bracket: its rendered hue lands nearer a stored value than the requested midpoint. Registered:
`SNAPFRAC(discrete CLU) > 0.5` and `SNAPFRAC(control) < 0.5`. (Definitions §3.) My target file calls this the most
informative thing this hub can produce, so it is reported whatever it says.

**Third registered prediction — the laundering control.** `store_removed` must collapse on row 2: registered as
`HUEMAE(store_removed) >= 0.8 Delta`. If it does **not** collapse, every row-2 number here came from somewhere
other than the store and I report that first.

## 3. The metrics, fixed now

The judge is categorical (LOG FACT A), so the readout is the **posterior-derived hue**, and hue is **circular**:
- `p(image)` in R^10 = `gen_probe.factor_probs(judge, image, hue_head)`.
- **Circular posterior hue** `h_hat = angle( sum_c p_c * exp(2*pi*i*h_c) ) / (2*pi)`, wrapped to `[0,1)`. The
  linear posterior mean is reported beside it as a diagnostic, never as the metric.
- **Circular distance** `cdist(a,b) = min(abs(a-b), 1-abs(a-b))`.

**Primary — `HUEMAE`** = mean over requests of `cdist(h_hat, h_requested)`, in units of Delta = 0.1. Lower is
better. Continuous, no threshold, paired request-for-request between arm and control.

**Gated primary — `ROW2ACC`.** A row-2 success requires the arm to render the in-between hue **and not destroy the
rest of the tuple**: the other five factors must all be top-1 correct by the same judge, AND
`cdist(h_hat, h_requested) <= 0.5 Delta` (the render is nearer the requested midpoint than either bracket). Both
the gated and the ungated numbers are reported; neither is reported alone.

**The SNAP, measured explicitly (target-file mandate).** With brackets `h_lo < h_hi`, `Delta = h_hi - h_lo`,
project the readout onto the bracket segment: `t = (h_hat - h_lo)/Delta` (circular-unwrapped onto the segment).
- `d_req = abs(t - 0.5) * Delta` — distance from the rendered value to the **requested** value.
- `d_near = min(abs(t), abs(1-t)) * Delta` — distance from the rendered value to the **nearest stored** value.
- `SNAPFRAC` = fraction of requests with `d_near < d_req`. `SNAPFRAC -> 1` is a lookup with extra steps;
  `SNAPFRAC -> 0` is genuine interpolation. Both `mean d_req` and `mean d_near` are reported as raw numbers.
- Note registered in advance: because a row-2 request is the **exact midpoint**, the two brackets are equidistant
  from it by construction, so "nearest stored value" is the nearer bracket *of the render*, not of the request.
  This is why the segment projection `t` is the well-posed form and a raw nearest-neighbour distance is not.

**Distinct-output count is reported beside every accuracy** (harness honesty rule 3): distinct judged factor
tuples, and distinct judged hue readings, per corner.

## 4. The estimand — corners, fixed now
w3-common §3: **>= 49 corners, paired, margin declared in advance. A single cell decides nothing.**

**Corner := (bracket pair, shape, scale tercile).** Realised: 7 bracket pairs x 3 shapes x 3 scale terciles =
**63 corners** (>= 49). Scale terciles are cut on the 8-value scale grid as `{0.750,0.821,0.893}` /
`{0.964,1.036}` / `{1.107,1.179,1.250}`, a fixed rule written here and not tuned.

Pairing is **within corner, request-for-request**: the arm and both controls see the byte-identical request list,
so the pairing is exact and the corner is the unit of the estimand.

**Estimand** = mean over the 63 corners of the per-corner paired difference `D_c`, with the paired 95% CI over
corners. **Declared margin: the arm wins only if the paired mean `D >= 0.2 Delta` AND the paired 95% CI over
corners excludes 0.** Anything else is a tie or a loss and is reported as such.

**Request set.** Primary = the frozen `row2_unseen.json` (n=512, seed 0), used verbatim. If the pinned builder
`build_row2_unseen` extends deterministically to a larger `n` with the **first 512 records byte-identical** (it
should: the RNG stream is fixed by a single leading `permutation` call and `n_want` only sets the break), I will
extend to N=4096 for corner density and **assert the byte-identity of the first 512** before using it. If that
assertion fails, I fall back to the frozen 512 and say so. Either way the corner definition above is unchanged.

## 5. THE JUDGE STOP RULE — run first, before any arm exists
My target file: *the judge cannot resolve the in-between value at all => the row is unmeasurable with this
instrument. Establish the judge's resolution on real images first. If it fails, stop and report.*

**Protocol, fixed now.** shapes3d has no off-grid hue rows, so an in-between image must be constructed — and the
construction is validated **against the dataset's own hue mechanism** before it is trusted:
1. **Mechanism check.** For matched rows differing only in `object_hue`, recover the object mask and verify that
   the dataset's hue step is an HSV hue rotation with S and V preserved.
2. **Construction check (the one that makes it "real").** Build an **on-grid** hue image from a *different*
   on-grid hue by the same rotation (e.g. synthesise 0.2 from 0.0 by rotating +0.2) and compare **pixel-wise** to
   the **real** 0.2 image. Registered pass criterion: **mean absolute pixel error <= 0.02** (on [0,1] images).
   If that passes, an image built the same way at 0.05 is produced by the dataset's own hue mechanism and is as
   real as this dataset gets. It is always described as **real-derived**, never as "real", in the report.
3. **Resolution measurement.** Sweep true hue across a bracket in steps of 0.1 Delta and measure `h_hat` on those
   images. Registered pass criteria, BOTH required:
   - **monotonicity**: Spearman rho between true hue and `h_hat` across the sweep >= **0.9**;
   - **midpoint resolution**: `mean cdist(h_hat, h_true)` at the true midpoints <= **0.25 Delta**, and AUROC
     separating midpoint images from each bracket's real images >= **0.9**.

**If either criterion fails, this hub STOPS, spends no scored run, and reports "row 2 is unmeasurable with this
instrument" as its result**, with the numbers. That is a real finding about the instrument and the Head is told
immediately. It is registered here so it cannot later be described as a failure to deliver.

## 6. The bar — the control I must build first, and build well
**Control 1 (the bar): fitted basis + linear head at matched bytes** — w3-common §1's object, with the linear head
**interpolating between prototypes**, which is what makes it the sharp control for *this* row. It is built and
measured **before the arm exists**, per my target file.
**Control 2 (second row): `knn_proto`**, the pin's rival, at its declared byte position.
**Laundering control: `store_removed`** on every row-2 number (charter §C.2).
**Matched bytes**: deployed store 524,288 B; the head's bytes declared beside it via `byte_ledger()`; the ledger
assertion `gen_landscape.landscape_byte_ledger` must pass. A number whose ledger does not balance is void.

## 7. Abandon rule — what makes me stop pursuing the arm
- The judge stop rule (§5) fires => stop, report the instrument result.
- `D < 0.2 Delta` on the 63-corner paired estimand, or the CI includes 0 => **the row is the head's, not the
  store's.** I abandon the arm as a claim, report it as decoration in w3-common §1's words, and do **not** spend
  further scored runs trying to win it back.
- `store_removed` fails to collapse (§2, third prediction) => every number is void and reported as void.
- The arm renders a valid image that is not the requested value (high `SNAPFRAC`) => reported as *a lookup with
  extra steps*, in those words, with the snap numbers.

## 8. Floors carried on every number (charter §C, w3-common §7)
beats the matched-bytes lookup beyond seed spread · landscape-removed collapses · settled state not on its nearest
atom (else it is a lookup table and is reported as one) · seed 0 only this wave, so a scored run buys a **verdict,
not a paper number** (w3-common §5).

## 9. Scope statement carried with every row-2 number
The hold-out is from **the store's fit pool**, not from the encoder, decoder or judge — those three are frozen,
hash-pinned, and were fitted on a split containing every hue. "The store never held a row carrying that value."
A row-2 number quoted without this sentence is a claim this hub did not make.

---

## AMENDMENT 1 — 2026-09-18 23:23Z (appended, not edited; charter §A.3)

**Written after §5 step 2 FAILED its registered criterion (worst whole-image pixel MAE 0.03062 > 0.02) and
BEFORE the re-run.** The failure stands and is reported as a failure. The diagnosis is that the mechanism
(step 1) is exact on pixels where hue is well defined (S > 0.2: rotation error <= 1.1e-04, |dS| = |dV| = 0.0000)
and my construction applied it to every mask pixel, including anti-aliased boundary blends where a per-pixel
hue rotation cannot be correct in principle.

**A1.1 — the construction is restricted to its measured domain.** Rotate only **interior object pixels**: in the
object mask AND S > 0.2 in *both* endpoint images. Boundary pixels keep their source-image values, and the
fraction of mask pixels left unrotated is reported. **Pixel criterion is unchanged: worst whole-image MAE <= 0.02.**

**A1.2 — a second, OPERATIVE criterion: JUDGE AGREEMENT.** Pixel MAE is not the quantity this hub depends on;
the only instrument that reads these images is the frozen external classifier. What must hold is that it reads a
*constructed* on-grid image the way it reads the *real* one. Registered here, before the number:
- (a) top-1 `object_hue` class agreement, constructed vs real, **>= 0.95**;
- (b) mean `abs(h_hat_constructed - h_hat_real)` **<= 0.1 Delta = 0.01**.

**A1.3 — the stop rule is unchanged and now binds on both criteria.** If the refined construction fails **either**
A1.1 or A1.2, PREREG-0 §5's stop rule fires: the hub stops, spends no scored run, and reports row 2 as
unmeasurable with this instrument.

**A1.4 — the resolution criteria are NOT amended.** Spearman rho >= 0.9 across the sweep; midpoint
`mean cdist(h_hat, h_true) <= 0.25 Delta`; AUROC separating midpoints from each bracket's real images >= 0.9.
Those were registered before any number and stand exactly as written.

---

## AMENDMENT 2 — 2026-09-19 00:45Z (appended; written BEFORE any row-2 number exists)

My target file: *"The control is the sharp one here and you must build it well ... Build the control that can kill
the idea before you build the idea."* Two strengthenings, registered before the first row-2 measurement.

**A2.1 — a SECOND fitted-basis variant, and the bar is the STRONGER of the two.**
The read under test is a **signed linear span** (w3-common §4.5), and for a signed span over 64 atoms per well the
**optimal** basis in squared error is the leading rank-64 PCA subspace — strictly stronger than a k-means++
medoid dictionary. Both are built at identical bytes (64 x d_k float32 per well, 524,288 B total, byte-identical
to the deployed store):
- `basis_head` — k-means++ atoms (the wave-2-style medoid dictionary; w3-common §1 records that medoid atoms tie
  fitted atoms exactly);
- `basis_head_pca` — the per-well leading rank-64 PCA subspace, with the well mean retained in the span.

⭐ **The bar is whichever of the two has the LOWER `HUEMAE`.** The arm must beat the *better* control, not the
one that happens to be convenient. Both are reported on every row, whichever wins.

**A2.2 — a second, harder laundering control.** `store_removed_*` uses the pin's `mean` deletion, which makes all
64 modes per well identical, so a signed span over them is rank-1 **by construction** and its collapse is close to
trivial. A `scramble` deletion (the pin's third mode) destroys content while preserving each well's shape, scale
and marginals exactly — so `store_scrambled_span` keeps full rank and is a genuinely hard control. Registered
now: **if the store's content is load-bearing, `store_scrambled_span` must also collapse**, at the same
`HUEMAE >= 0.8 Delta` threshold PREREG-0 §2 registered for `store_removed`.

⛔ Neither amendment relaxes anything: both raise the bar the arm must clear, and both were written before the
first row-2 number.

---

## AMENDMENT 3 — 2026-09-19 01:05Z (appended; the CONSTRUCTION changes, no threshold does)

**Written after A1.1 failed TWICE** (worst whole-image pixel MAE 0.0306 naive, 0.0293 interior-restricted, against
the registered 0.02) and after A1.2 PASSED (top-1 agreement 0.9996, mean |dh| 0.00010). The two A1.1 failures
stand as failures and are reported as such.

**A3.0 — the 23:21Z diagnosis was wrong and is withdrawn.** The residual is not predominantly in anti-aliased
boundary pixels: the interior mask retained 98.76% of object pixels, so at most ~0.2% of the image changed hands,
which cannot move a 0.030 MAE. The residual is in the object's **reflection on shapes3d's reflective floor** — a
blend of object and floor colour, which moves along a chord and does not rotate with the object's hue. No
per-pixel hue rotation on any mask can reproduce it.

**A3.1 — the IN-BRACKET construction.** Row 2 asks for a value strictly inside a bracket, so the construction is
built and validated inside a bracket, between two REAL endpoint images `I_a`, `I_b`:
- a pixel is **rotational** iff its measured a->b change IS the hue rotation: `|dhue - step| <= 0.01`,
  `|dS| <= 0.02`, `|dV| <= 0.02`, and `S >= 0.2` at both ends. Those pixels are hue-rotated by `f*(b-a)`, which
  step 1 measured to be exact (error <= 1.1e-04, |dS| = |dV| = 0.0000).
- every other pixel is **RGB-lerped** `(1-f)*I_a + f*I_b` -- the correct first-order behaviour of an anti-aliased
  or reflected blend, and exact at `f = 0` and `f = 1`.
Tolerances are fixed here, before the run.

**A3.2 — the validation becomes on-grid and therefore checkable.** Take a bracket of width `2*Delta` (e.g. 0.0 and
0.2), construct `f = 0.5`, and compare **pixel-wise to the REAL midpoint image** (0.1). This is a genuine
in-between construction checked against a genuine in-between image, which the src->dst rotation never was.
⛔ **The threshold is NOT relaxed and NOT moved: worst whole-image MAE <= 0.02**, the number registered at 22:50Z.

**A3.3 — the resolution criteria are untouched.** Spearman rho >= 0.9; midpoint `mean cdist <= 0.25 Delta`;
AUROC >= 0.9. Unchanged since 22:50Z, before any number.

**A3.4 — what the verdict rests on.** The stop rule is decided on the in-bracket construction, because that is the
one whose fidelity is demonstrated. ⛔ I will not declare "row 2 is unmeasurable" on images that failed their own
pixel check, and I will not declare "row 2 is measurable" on them either. Both readings of the first construction
are reported as provisional, and the verdict is the in-bracket one.

---

## AMENDMENT 4 — 2026-09-18 22:52:57Z (appended; still BEFORE any row-2 arm number)

**A4.1 — the request set is EXTENDED, and the superset assertion PASSED.** `build_row2_unseen` was run at the
pinned `row2_seed = 0` for `n = 4096`, and the **first 512 records are byte-identical to the frozen
`row2_unseen.json`** — asserted on the records, not assumed. PREREG-0 §4's condition for using the extension is
therefore met. `runs/row2_unseen_n4096.json`.

**A4.2 — the corner rule takes scale QUARTILES, not terciles, because terciles do not reach the bar.**
PREREG-0 §4 predicted 63 corners from `7 brackets x 3 shapes x 3 scale terciles`. **Measured: only 48 realise**
— 15 of the 63 cells are empty, at n = 512 and at n = 4096 alike. **48 < 49**, so the registered rule misses
w3-common §3's estimand bar on the query set as it actually exists.

Same rule, one cut finer: **corner := (bracket pair, shape, scale QUARTILE)**, quartiles
`{0.750,0.821} / {0.893,0.964} / {1.036,1.107} / {1.179,1.250}`. Measured realisation:

| partition | corners | sizes | median |
|---|---|---|---|
| bracket x shape x scale-tercile (registered) | **48** | 51–123 | 90 |
| **bracket x shape x scale-quartile (adopted)** | **64** | **49–83** | **63** |
| bracket x shape x tercile x orientation-half | 96 | 20–65 | 42 |

⭐ 64 corners, every one with at least 49 requests. ⛔ This is a property of the **query set**, measured with no
arm, no control and no model output in the room; it is not a bar on any arm's performance and no arm number
existed when it was written. The 48-corner shortfall of the registered rule is reported as a finding in its own
right.
