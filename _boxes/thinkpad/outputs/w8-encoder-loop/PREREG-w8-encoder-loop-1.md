# PREREG — w8 hub/encoder-loop, scored run 1 of 3

**Written:** 2026-09-20, BEFORE any held-out number exists.
**Target:** `encoder-loop`. **Budget:** 3 scored runs. This is **run 1/3**.
**Branch/base:** `w8/encoder-loop` off `base/iter7 = ec2ef7c`.
**Charter:** A.1 (pre-registered before it runs) · A.2 (selection on `val` only) · A.3 (a bar seen a
result against is never changed; amendments are appended, dated, never edited).

---

## 1. THE QUESTION, in one line

**Does the store force better features?** An encoder trained ALONE versus an encoder trained WITH
THE STORE IN THE LOOP, at EQUAL COMPUTE, where the loop's objective is the **swap loss plus
reconstruction** — scored on rows 3 and 4 with **every arm and every baseline given the same
features**.

---

## 2. THE DIAL, and what is held fixed

**Dial: the encoder.** One shared warm start per seed; the arms differ only in stage 2.

| arm | store on the gradient path | encoder in Optax | swap loss | role |
|---|---|---|---|---|
| `alone` | ⛔ no | yes | ⛔ no | ⭐ **CONTROL — "the encoder trained ALONE"** |
| `loop_nosw` | yes | yes | ⛔ no | ⭐ **unfrozen WITHOUT the swap loss — its own column** |
| `loop_swap` | yes | yes | ⭐ **yes** | ⭐ **THE ARM** |
| `alone_wc` | ⛔ no | yes | ⛔ no | CONTROL — `alone` matched on **wall clock**, not steps |

**Held fixed and shared bit-identically across arms:** the warm-start `(encoder, store, decoder)`
triple and its three parameter hashes; the PRNG stream and therefore the minibatch sequence; the
training rows; the well partition; the **fixed ownership vector** (derived once on the warm start,
Hungarian on the measured transfer matrix, then constant — `gen_swap_probe`: *"a refinement loop
must fix it, or the objective moves under the optimiser"*); and ⭐ **THE EXTERNAL CLASSIFIER**,
hash-asserted, which reads IMAGES and is therefore the one instrument that survives a change of
feature space.

**Not shared, and it cannot be: the decoder.** An unfrozen encoder's features are not the frozen
decoder's input space (w8 rule 1). Every arm gets a fresh decoder **refit from its own frozen
features by the identical recipe at the identical step count**, fit downstream of the encoder
exactly as R18 shapes the shipped pipeline.

---

## 3. ⚠ DECLARED DEVIATIONS FROM THE INCUMBENT'S INSTRUMENT — read before any number

1. ⛔ **The decoder is `w8_v1_short`, not the gated `v1_capacity_schedule`.** Same architecture
   (width ×2, 2 blocks/stage, cosine, 5 % warmup, same loss); **only `dec_steps` is cut**, under a
   different recipe name so no manifest can read as the gated artefact, and a code assertion checks
   that nothing but the step count differs. **Why:** measured on this box, the gated recipe costs
   **2.724 s/step**, i.e. **82 min per decoder**, and this hub needs one decoder per arm.
   ⛔ **Consequence, stated in advance: absolute row-3/row-4 values from this hub DO NOT SHARE A
   TABLE with the incumbent's.** They share an architecture, not an instrument. What they are
   exactly comparable with is each other.
2. ⚠ **`--quick` numbers are VOID** and every manifest they write says so.
3. ⚠ **The swap loss trains against the scoring instrument.** Precedent exists (hub/placement used
   the judge as a fit-time selection loss) and the task commissions it in terms. It carries the
   CROSS-JUDGE control in §5.

---

## 4. THE ESTIMAND

Unmodified frozen harness: `exp_w5_compose_field` → `exp_gen_read` → `exp_gen_factor_wells`, driven
by `scripts/w8_encoder_loop_score.sh`, which edits nothing and only points the harness's own
documented env redirections at one encoder arm.

- **Row 4** — `r4_comp_nway`, **49 n-way corners**. **Row 3** — `r3_comp_2way`, **60 pair
  sub-corners**. ⭐ ≥ 49 paired corners on both, as w8-common §2 requires.
- **Metric:** `distinct_valid_novel_tuples` per corner — judged valid by the external classifier,
  novel against the fit pool.
- **Statistic:** `paired_estimand`, unit = the corner, paired in order; mean paired difference, a
  two-sided sign test and a 10,000-draw paired bootstrap CI95.
- **Seeds 0, 1, 2**, reported individually and never pooled into one number.
- ⚠ Row 3's ceiling is **36** tuples/corner and row 4's is **72** (`object_hue` and `shape` are both
  fixed in a row-3 corner, so each tuple is requested twice). ⛔ The two rows are not on one
  absolute scale; `frac_of_ceiling` is the comparable column and is reported beside both.

---

## 5. ⭐ THE BARS — declared now, in advance, and never moved

### BAR A — the hub's question (PRIMARY)
`pca_tensor@loop_swap` − `pca_tensor@alone`, paired by corner, on **row 4** and on **row 3**.

- **PASS:** mean paired delta **> +1.0** novel tuples/corner **AND** bootstrap CI95 low **> +1.0**,
  on **all three seeds**, on row 4. (+1.0 is the programme's own `DECORATION_MARGIN_NOVELTY`.)
- **EXPECTED DIRECTION:** positive, if the store shapes the representation.
- ⭐ **SECONDARY READING, declared now as secondary and ⛔ never quotable as BAR A:** mean paired
  delta > 0 on all three seeds with sign-test p < 0.05. A hub-sized effect on a *representation*
  may be real and smaller than a *mechanism* margin; saying so in advance is the only honest way to
  report it.

### BAR B — THE FALSIFIER (this is the one the task names)
If the mean paired delta is **≤ 0 on two or more of the three seeds** on row 4, then
**store-in-the-loop features do not beat encoder-alone features at equal compute**, and this hub
reports that **the store does not shape the representation**. ⭐ That is a clean answer to a claim
the programme has carried a long time and it is reported as the result, not as a failure.

### BAR C — EQUAL COMPUTE, or nothing
1. `gradient_steps` and `images_seen` must be **identical** across `alone`, `loop_nosw`, `loop_swap`.
2. `alone_wc.train_seconds` must be **≥ `loop_swap.train_seconds`**.
- ⛔ If BAR A passes against `alone` on equal STEPS but fails against `alone_wc` on equal SECONDS,
  the result is reported as a **COMPUTE WIN, not a representation win**.

### BAR D — do the baselines close the gap?
`pca_tensor@X` − `rival_mlp_180k@X`, computed **within each arm X** (the baseline refit on that
arm's own features, `w4_learners`' protocol verbatim, matched parameters 179,968, `dropout=False`).
- ⭐ If that CLU-minus-baseline margin does **not grow** from `alone` to `loop_swap`, then the
  features are better **for everyone** — a finding **about the encoder, ⛔ not about the store** —
  and this hub says which, in those words.

### BAR E — the factor probe, as a MARGIN
Linear per-factor readout of the features, identical budget for every arm, reported **only** as
`accuracy(arm) − accuracy(random_init)` against an untrained encoder of the identical architecture
carried through the identical probe.
- ⛔ **No absolute probe number is a claim.** If the probe improves in absolute terms but not over
  random init, there is **no claim**.

### BAR F — the CROSS-JUDGE control on the swap loss
The swap `exact` rate scored under a **different seed's** external classifier than the one in the
loop. ⛔ If the treatment's advantage exists only under its own judge, it is **adversarial to the
instrument, not a representation gain**, and is reported as such.

### Charter §C floors, on the scored table
- **C.1/C.4** `pca_lookup_argmax` — the matched-bytes lookup over the same atoms.
- **C.2** `pca_landscape_removed` — atoms zeroed; **must collapse**.

---

## 6. WHAT IS NOT IN THIS RUN, declared as NOT-RUN rather than omitted

- ⛔ **The w8 single design.** It did not exist when this hub built: `w8-hub-single-design` wrote its
  `START_UTC.txt` in the same minute as this hub and is itself gated on
  `w8-downdate/DOWNDATE-READY.json`. This run scores **`pca_tensor` as-is** — wave 5's incumbent —
  on the encoder-loop features. If the single design lands in time it is a second scored run under
  its own appended PREREG; if it does not, it is a **DECLARED NOT-RUN with its launch line emitted**
  and ⛔ never a null.
- ⛔ **`frozen` and `alone_wc` in the rows-3/4 scoring.** `alone_wc` is reported on training compute
  and probe margin (which is where the equal-seconds question is actually decided); `frozen` is
  reported if wall clock allows. Both declared, neither silently dropped.
- ⛔ **The GPU leg.** `gcloud` is not installed on this box. The spec is emitted, verified and
  ready at `cloud/LAUNCH-SPEC.md`; the Head executes the authenticated step.

---

## 7. Provenance carried with every number

Cache generation 2, full scope, built on one CPU device (`device_guard`), per-block sha256.
Labels are used at fit time to align wells, to build the address book and to form the swap loss's
target on **fitting rows only**; they never enter the landscape, the read or the write (charter
B.6). ⭐ `pca_tensor`'s content remains **unsupervised**. Store 524,288 B exact; the address book is
reported beside it and never netted in. `assert_no_feature_bypass` and `assert_no_bypass` run before
any number is kept.

---

## AMENDMENT 1 — 2026-09-20T22:10Z (appended, never an edit; charter A.3)

**What this amends:** the LOG's 18:27Z pre-committed contingency, which said that if the prep
phase were not finished by 22:30Z the scored run would be cut from three encoder arms to two
(`alone`, `loop_swap`), losing BAR A′.

**The fact that triggers it:** prep will not be finished by 22:30Z. `alone_wc` is at ~295/883
steps and will land ≈22:38Z; probes and the two remaining caches follow, so prep ends ≈23:25Z.

**The decision: the cut is NOT taken, and here is the measurement that justifies not taking it.**
The contingency was written at 18:27Z against a projection, before any arm had been timed. Since
then every stage has been measured on this box:

| remaining work | measured basis | estimate |
|---|---|---|
| `alone_wc` ×3 (parallel) | 3.3 s/step × ~880 steps | to ≈22:38Z |
| probes ×4 (parallel) | cache-free, encoder-only | ≈25 min |
| caches ×2 (`loop_swap`, `loop_nosw`) | 10.5 min measured on the `alone` caches | ≈21 min |
| `val` floor-effect check | fires automatically on the `loop_swap` cache | ≈25 min, OVERLAPPED |
| `loop_swap` rivals refit ×3 (parallel) | 663–829 s measured on `alone` | ≈14 min |
| 3 score legs per seed (parallel across seeds) | 25 min/leg measured from the w5 scored log | ≈75 min |

⭐ That lands the scored run at **≈01:05Z against a 02:36Z bound — ~1.5 h of slack.** The cut
would buy ~32 min and cost BAR A′, the `loop_nosw − alone` column, which is ⭐ **the column the
task file specifically asks to be isolated** ("unfreezing alone made things worse in an earlier
wave; the swap loss is the new ingredient — isolate it"). Spending 1.5 h of slack to keep the
commissioned column is the right trade and it is made on measured numbers, not on hope.

⛔ **The contingency is not cancelled, it is re-based.** New trigger: **if the three `test` score
legs have not STARTED by 00:15Z**, the scored run is cut to `alone` and `loop_swap`, and BAR A′ is
reported as a DECLARED NOT-RUN with its launch line emitted. ⛔ No bar, margin or estimand in
sections 4–5 is touched by this amendment; it concerns scope under wall clock and nothing else.
