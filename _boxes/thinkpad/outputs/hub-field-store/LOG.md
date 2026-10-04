# hub/field-store — LOG

**Hub:** `hub/field-store` · **branch** `w4/field-store` · **base** `9dfed5e` (`base/iter4`)
**Mandate:** BUILD ONLY. ⛔ Scored budget ZERO. No claim, no held-out number.
**Output dir:** `/c/chlu/outputs/hub-field-store` (the only place this hub writes).

---

## 2026-09-19

- **16:53Z** `START_UTC.txt` written into `/c/chlu/outputs/hub-field-store`; no live start file existed
  beforehand (directory was empty). Boot order honoured: charter → `w4-hub-common.md` → `w4-hub-field-store.md`,
  and nothing else. ⛔ No wave-2/wave-3 report read.
- **16:55Z** Environment verified. `chlu.__file__` = `C:\chlu\worktrees\CHLU-w4-field-store\chlu\__init__.py`
  — inside the worktree, so numbers from this session are attributable to this branch. python 3.11.9.
  Frozen instrument present: bridge `seed_0/1/2` (`features.npy` 1,966,080,128 B, sha
  `d6728f01…`, 480,000 × 2048, feat_shape 128×4×4), declift winner decoders `decoder_v1_s{0,1,2}.pkl`,
  external classifier `seed_0/1/2`. Data root has `shapes3d/value_holdout` complete.
- **16:56Z** `/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` does **not** exist yet. Irrelevant to this
  hub: the rivals gate binds scored runs and this hub has none. Recorded so the absence is on the record.
- **16:58Z** Inherited the factor-aligned partition (`exp_gen_factor_wells --stage partition`, pins untouched,
  fit pool 131,072 rows, 0 shared with the hold-out, 64 dead channels, top-factor MI 0.5458 bits).
  Wells (K=5, D=2048): `0=floor_hue(777) · 1=wall_hue+scale(939) · 2=object_hue(216) · 3=shape(58) ·
  4=orientation(58)`. ⭐ **`object_hue` and `shape` land in DIFFERENT wells (2 and 3)** — so the red-cylinder
  conjunction is genuinely a coupling between two wells and the hole is expressible as designed.
  ⚠ Wells 3 and 4 are rank-starved (d_k=58 < 64 atoms): their atom span is the whole well, so those two
  wells are read exactly. Recorded now because it will explain part of the smoke.
- **17:05Z** Design fixed before any code. The field store is **its own byte image**: one `uint8` buffer with a
  segment table and typed views into it, so "changes zero bytes outside X" is a literal statement about that
  buffer and not a re-serialisation artefact. Segments: `well_of_channel · atoms · atom_life · atom_alive ·
  atom_row · atom_basin · well_sigma · basin_sigma · basin_gap · basin_n · coupling · coupling_hole ·
  term_mu · term_n · global_mu`. Atoms are **A=64 training ITEMS, each written into every well** (item i
  occupies d_k floats in well k) → 64 x 2048 x 4 = **524,288 B, byte-identical to the wave-3 store ledger**.
  One LIFETIME per item, as the task specifies.
- **17:05Z** Declared design choice, so it is on the record rather than discovered later: **the address book
  survives deletion.** Deletion evicts ATOMS; `term_mu` (the term->coordinate count table) is an interface
  object built from pool counts and is NOT rewritten, because rewriting it is a refit. The consequence is the
  intended one: after eviction the word still addresses a place, and there is nothing there — a GAP — which is
  what refusal reads. This is also what makes the byte property exact.
- **17:40Z** `chlu/core/gen_field.py` written and unit-smoked on synthetic factor-aligned data
  (3 factors, 24 dims, 3 wells, 16 atoms). All four byte properties hold **exactly**:
  evict 413 B changed / **0 outside**; hole **1 B** changed / 0 outside; decay 206 B / 0 outside;
  per-item lifetime 2 B / 0 outside. `assert_no_refit` passes on all four (no fitted segment moves).
  Read is request-only by signature and bit-stable under a feature perturbation.
- **17:45Z** ⚠ **CORRECTION to the first coupling design, found by measurement, not by argument.**
  The first version used raw smoothed PMI as a multiplicative read gain. On synthetic data that
  *doubled* the read error (rel-err 0.155 / 0.214 vs 0.051 / 0.071) and fired false refusals
  (gap score 1.66 and 2.85 against a 1.50 threshold) on perfectly valid requests. Cause: on a
  fully-crossed pool the raw PMI is 1/sqrt(n) sampling wobble, and an unseen pair's raw PMI is a
  large NEGATIVE number (~ -7), which would have suppressed exactly the row-3/row-4 requests.
  ⛔ **That is wave 3's joint-table trap re-appearing in soft form.**
- **17:45Z** ⭐ The fix, and it is the load-bearing part of the design: the prior a pair backs off to
  is **INDEPENDENCE**, with evidence weight `n/(n+m)` (m=32) and a 2-sigma noise threshold on top.
  An unseen pair therefore gets `pmi` **exactly 0**, a read gain of **exactly 1**, and is **answered**.
  A DELETED pair is a different number in a different segment (`coupling_hole`, 1 byte).
  After the fix, synthetic rel-err 0.051 / 0.031 / 0.071, zero false refusals, and 0 of 47 coupling
  entries survive the shrink on truly-independent synthetic data — i.e. the table is measurably flat
  where it should be flat.
- **18:10Z** Smoke corners built and STRUCTURALLY VERIFIED (`--stage corners`): row 3 = **60** val
  two-way corners of 36 rows over three `(object_hue, shape)` conjunctions **every row of which is
  removed from the fit pool** (36,000 grid rows), row 4 = **24** val Montero n-way corners of 72 rows,
  row 1 = 512 requests drawn from the pool itself. ⭐ `rows_in_fit_pool = 0` for rows 3 and 4,
  `rows_in_scored_test_index = 0` for both, all corner rows in the value-holdout TRAIN index, corner
  sets pairwise disjoint. ⛔ **The scored test corners and the shipped query sets are not touched, so
  this hub spends 0 of its 0 scored runs.** Pool 131,072 rows after excluding 37,728.
- **18:10Z** ⭐ **BANDS DECLARED IN ADVANCE, before the first smoke number** (charter A.3 — not moved
  afterwards): "within the wave-3 spread" means |mean paired per-corner delta| <= **1.0 distinct valid
  novel tuple** on rows 3 and 4 (the wave-3 hub's own decoration margin) and |delta| <= **0.02**
  absolute judge accuracy on row 1, against `aligned_w3` refitted on the identical pool, paired corner
  by corner, at seeds 0/1/2.
- **18:12Z** Field store built, seed 0, in 20.4 s. Ledger: atoms **524,288 B** (= the wave-3 store
  ledger exactly) + interface **486,318 B**; whole image 1,013,658 B. ⭐ For comparison the wave-3
  object's addressing head is **1,994,240 B**, so the field store's interface is about **4.1x cheaper**
  than the head it replaces.
- **18:12Z** Three measurements from the build, each of which a later hub would otherwise rediscover:
  1. ⚠ **basin gap / kernel width per well = 1.34 / 0.81 / 0.75 / 0.71 / 0.40.** The basins **OVERLAP**
     in raw feature space: the within-value spread is as large as the between-value gap. The gaps are
     real objects and are stored, but on this encoder they are NOT a clean geometric separation, and
     wave 5's refusal hub must know that before it is scoped on them.
  2. Atoms per term: min 0, max 23 — and the only two terms with **zero** atoms are `object_hue=0.5`
     and `shape=0.0`, the condition's own held-out values, which the pool does not contain. Correct.
  3. The coupling table is **not** flat here: 105 of 1322 term pairs were never held (they back off to
     the independence prior, `pmi` exactly 0) and 40 survive the 2-sigma shrink, `|pmi|` up to 1.090,
     concentrated in `object_hue x shape` — which is the value hold-out's OWN depletion structure
     showing up, not a property of shapes3d (which is fully crossed by construction).
- **18:35Z** 12/12 unit tests pass (`tests/test_gen_field.py`): ledger, atoms-are-items, request-only
  read, landscape-removed collapse (read exactly 0), eviction/hole/lifetime/decay byte counts,
  the unseen-vs-deleted trap test, and a **negative control** proving the byte instrument is not blind
  (nudging one address-book row by 1e-3 moves bytes and fails `assert_no_refit`).
- **18:48Z** ⭐ **SELECTION on val, by feature reconstruction error, no judge, no scored run**
  (`--stage sweep`, 16 configurations x 3,888 val corner rows). Best field config:
  `coupling_eta=0.0, sigma_mult=0.5, ridge_frac=1e-3` -> **rel_err 0.3566 / cos 0.9326**.
  Two things measured there and now pinned:
  - the coupling as a READ GAIN **hurt** (0.3566 -> 0.3617 at `coupling_eta=1`). Reason, measured: the
    40 PMI entries that survive the shrink are the value hold-out's OWN depletion structure
    (concentrated in `object_hue x shape`), not a property of shapes3d, which is fully crossed.
    ⭐ So `coupling_eta` is pinned to **0.0**: the coupling table's job is the HOLE and the
    unseen-pair prior, and both are measured on their own terms. Declared, not hidden.
  - the atom projection adds **nothing** over the address book alone (0.3566 vs 0.3541). The atoms are
    not yet earning their place on reconstruction.
- **18:48Z** ⛔ **THE INCUMBENT IS AHEAD ON THIS PROXY: `aligned_w3` rel_err 0.2798 / cos 0.9574 vs the
  field store's 0.3566 / 0.9326**, on the same rows, the same pool, the same partition. Reconstruction
  error is a proxy, not the acceptance metric — the acceptance metric is the judged rows. Going to the
  judge now before changing anything, because the right question is whether that gap survives the
  decoder and the classifier.
- **19:10Z** ⛔⛔ **FIRST SMOKE, seed 0, and it FALSIFIES the first build of the object.** Same pool,
  same partition, same judge, same corners, paired corner by corner:
  | row | field | aligned_w3 | paired delta (field - aligned) |
  |---|---|---|---|
  | r1 seen (judge acc) | **0.1367** | **0.5391** | -0.4024 |
  | r3 2-way (novel tuples/corner) | **0.233** | **10.217** | **-9.983** (95% CI -12.47..-7.60, p=3.1e-11, 2W/14T/44L) |
  | r4 n-way (novel tuples/corner) | **1.667** | **24.083** | **-22.417** (95% CI -27.58..-17.25, p=1.2e-07, 0W/0T/24L) |
  ⛔ Against a declared band of +-1.0 novel tuples and +-0.02 accuracy, that is **far outside**, not
  within. Reported here before it is fixed, per the target file: the redesign as first built COSTS
  composition, and the number is on the record.
- **19:15Z** ⭐ **Cause located, by measurement, in one number.** The field store's address is a
  MAIN-EFFECTS count table; the incumbent's head carries a **pair-interaction block**. On the same
  3,888 val rows: additive count table **0.3541** · incumbent head **0.2798** · additive + the full
  **2-way pair-conditional** count table **0.2416**. ⭐ So the pair interaction is the entire gap, and a
  pair table does not merely close it, it **overtakes the head**. 2,160 of 58,320 evaluation pair-cells
  are conjunctions the pool never held; they back off to no correction, which is the same
  independence prior the coupling scalar already uses.
- **19:15Z** Design consequence, and it makes the coupling table load-bearing instead of decorative:
  **the coupling table between wells now carries a COORDINATE correction, low-rank, gated by the same
  smoothed evidence weight `n/(n+m)`.** One object, three jobs: (1) the pair coordinate correction,
  (2) the smoothed prior — an unseen pair gets weight 0, no correction, and is **answered**, (3) the
  HOLE — one byte, refused. Measuring the rank the correction needs now; the constraint is that the
  whole interface must stay under the incumbent's 1,766,400 B head.
- **19:55Z** Coupling coordinate correction implemented (rank chosen on val: r=4 -> 0.2652, **r=8 ->
  0.2524**, r=16 -> 0.2461, full -> 0.2406, incumbent head 0.2798; r=8 is the smallest rank that beats
  the head, at 1,025,344 B inside the head's 1,766,400 B). All 12 unit tests still pass — the byte
  properties are unaffected by the new segments.
- **19:55Z** ⭐ **Second smoke, seed 0, same corners, same judge. The fix moved row 4 into the band:**
  | row | field (was) | field (now) | aligned_w3 | paired delta | in band? |
  |---|---|---|---|---|---|
  | r1 seen (acc) | 0.1367 | **0.4355** | 0.5391 | -0.1036 | ⛔ no (band 0.02) |
  | r3 2-way (novel/corner) | 0.233 | **5.300** | 10.217 | **-4.917** (CI -7.47..-2.45, p=0.0021) | ⛔ no |
  | r4 n-way (novel/corner) | 1.667 | **24.375** | 24.083 | **+0.292** (CI -4.29..+4.79, p=1.00, 11W/1T/12L) | ⭐ **YES** |
  Side effect worth recording: the better address also sharpened the basins — **gap / kernel width per
  well went from 1.34/0.81/0.75/0.71/0.40 to 2.68/1.62/1.50/1.42/0.81**, i.e. four of five wells now
  separate their values by more than one kernel width. Refusal has more to read than it did.
- **19:56Z** Remaining gap is on rows 1 and 3. Measuring whether it is the ADDRESS or the atom
  PROJECTION that costs it, per row, before changing anything further.
- **20:25Z** Projection probe (val rel-err, rows 1/3/4). ⭐ **The ADDRESS is not the problem; the ATOMS
  are.** Address alone (term map + rank-8 coupling, no atoms): **0.1610 / 0.3015 / 0.1910** — better
  than the wave-3 head's 0.1911 / 0.3264 / 0.2216 on every row, at fewer interface bytes. The field
  read (that address projected onto the 64-item span): 0.2519 / 0.3379 / 0.2550. So the 64-item span
  costs 0.091 / 0.036 / 0.064 and that is the whole remaining deficit. `sigma_mult` saturates at >= 1.0
  (0.2519 at 1.0, 0.2517 at 1e6) and 0.5 is worse, so `sigma_mult` is re-pinned from 0.5 to **1.0** —
  the earlier 0.5 was selected under the additive address and did not survive the better one.
- **20:35Z** ⭐ Atom SELECTION changed from a Latin design over the factor grid to **greedy span
  selection against the address manifold, under a basin-coverage constraint** (each step the
  least-covered term names the candidate set; within it the item removing the most residual energy
  from a rank-128 sketch of the address manifold is written). Val rel-err 0.2048 / 0.3297 / 0.2410 vs
  the Latin design's 0.2509 / 0.3382 / 0.2542. Labels enter only at fit time, at the address book.
- **20:40Z** **Third smoke, seed 0.** field r1 **0.5840** / r3 **7.933** / r4 **30.375** against
  aligned_w3 0.5391 / 10.217 / 24.083. Deltas: r1 **+0.0449**, r3 **-2.283** (CI -4.25..-0.37,
  p=0.054), r4 **+6.292** (CI -0.29..+12.71, p=0.134). r4 and r1 now favour the field store; r3 is
  still short.
- **20:45Z** ⭐ **Per-factor read-back located r3's deficit in ONE factor.** On r3 the field store reads
  `shape` at **0.491** against the head's **0.750** (it is ahead on `object_hue`, 0.952 vs 0.840).
  Cause: the fit pool is deliberately unbalanced — whole `(object_hue, shape)` conjunctions are
  removed — so a MARGINAL mean carries the absence of those conjunctions inside the main effect, and
  the additive address leaks it. The wave-3 head does not have this problem because its main effects
  are estimated jointly with the pair block. Fix: the term -> coordinate map is now fitted by
  **backfitting** (4 sweeps of per-cell means against the running additive residual), which is still a
  count table and estimates the main effects jointly. Re-running the smoke.
- **21:30Z** ⛔ **CORRECTION to the 20:45Z bullet.** The backfitted term map was implemented and it was
  WRONG twice over. First it had a real bug: the re-centring step shed a constant that the intercept
  never absorbed, which drove the `object_hue` read-back to **0.086** (r1 accuracy 0.082, r3 novel
  0.000). Fixed by handing the shed constant to `global_mu` as a pure reparametrisation. Second, once
  correct, **backfitting is still worse than the marginal map on every row**: val address rel-err
  0.2556 / 0.3537 / 0.2810 after one sweep and 0.2837 / 0.4061 / 0.2823 after four, against the
  marginal map's **0.1610 / 0.3015 / 0.1910**. The joint fit buys pool residual with compensating
  effects that do not survive to held-out corners.
- **21:55Z** ⭐ Decision, recorded with its reason: **the term -> coordinate map stays MARGINAL.**
  `term_map="backfit"` is kept in the code so the negative result is reproducible rather than folklore.
  ⛔ The consequence is honest and stands as a measured property of the object: on the two-way corners
  the field store reads `shape` at **0.491** against the wave-3 head's **0.750**, because a marginal
  mean of an unbalanced pool carries the absence of the removed conjunction inside it. That is the
  whole of the remaining row-3 deficit and wave 5 should know it.
- **21:56Z** Launching the full matrix on seeds 0/1/2 with the pinned configuration: marginal term map,
  rank-8 coupling coordinate correction, greedy span atom selection, `sigma_mult=1.0`,
  `coupling_eta=0.0`, 64 items / 524,288 B.
- **22:35Z** ⛔ **DECLARED NOT-RUN, with its reason — this is not a null.** The three-seed requirement
  (w4-hub-common section 2) cannot be met on the BRIDGE seed on this machine: the frozen feature cache
  is full scope for **seed 0 only**. `features.npy` is 480,000 rows at seed 0 and **9,288 rows** at
  seeds 1 and 2 (verified by shape, not inferred). Extending it means writing inside the read-only
  frozen instrument and re-encoding ~960,000 images, and a hub may do neither (charter A.4, and
  `CHLU_HUB_OUTPUTS` is the only place this hub writes). The bridge-seed spread is therefore **NOT RUN**.
- **22:35Z** ⭐ What replaces it, and it is the spread the acceptance actually asks for: **three
  STORE-FIT seeds at bridge seed 0**. The store seed moves (a) the fit-pool draw — three genuinely
  different pools of 131,072 rows, verified distinct — (b) the incumbent's k-means++/adam stream
  (`store_key_seed`), and (c) the field store's atom selection. It is applied **identically to both
  arms**, so the pairing survives. That is the spread of the OBJECT at a fixed instrument, which is
  what "within the spread of the wave-3 aligned store" means; the instrument's own spread is declared
  NOT RUN above and wave 5 should re-measure it when a full-scope generation-2 cache exists at seeds
  1 and 2.
- **23:30Z** ⭐⭐ **THE DELETION PROOF, seed 0, asserted BYTE-WISE against a 2,044,290 B image.**
  Every count is the number of bytes whose value actually differs, from a direct comparison of the two
  buffers — there is no re-serialisation step in the path.
  | operation | bytes changed | **OUTSIDE the evicted atoms / the hole** | segments touched |
  |---|---|---|---|
  | **evict** `object_hue = 0.0` (5 items) | **14,735** | **0** | atoms, atom_life, atom_alive, atom_row, atom_basin |
  | ⭐⭐ **hole** — the RED CYLINDER (`object_hue=0.0 AND shape=cylinder`) | **1** | **0** | `coupling_hole` only |
  | decay x0.5 on 8 items | **8** | **0** | `atom_life` only |
  | decay to zero (8 items die) | **23,322** | **0** | atoms + the per-item rows |
  | per-item lifetime := 0.25 on 3 items | **3** | **0** | `atom_life` only |
  | ⚠ NEGATIVE CONTROL: nudge ONE address-book row by 1e-3 | **5,762** | — | `term_mu` |
  `assert_no_refit` passes on every operation: `well_of_channel · well_sigma · basin_sigma ·
  basin_gap · basin_n · term_mu · term_n · global_mu · coupling` are **byte-identical** across all of
  them. ⭐ **The acceptance property holds EXACTLY, not approximately.**
- **23:32Z** ⭐⭐ **The hole is exactly scoped, and the trap does NOT re-appear.** After one byte:
  hole flag on the conjunction **0.0 -> 1.0** (256 probes); on `object_hue=red` alone **0.0**; on
  `shape=cylinder` alone **0.0**; on neither **0.0**. Refusal on the 256 general probes changed on
  **0.0000** of them. ⭐ And the load-bearing one: **96 UNSEEN two-way conjunctions — pairs the pool
  never held, i.e. exactly what every row-3 request is — are hole-flagged at 0.0000 after the hole.**
  The red cylinder's own coupling entry reads `pmi = 0.0, evidence weight = 0.0` both before and after:
  it was already an unseen pair and was being ANSWERED; one byte in a different array turned it into a
  refusal without touching any other unseen pair. That is the escape from wave 3's joint-table trap,
  made mechanical.
- **23:35Z** Eviction's behaviour, reported including its cost: requests for the deleted value go from
  0.167 to **1.0000** refused (via the now-EMPTY basin: `live_atoms_in_deleted_basin_after = 0`), while
  surviving requests refuse at 0.0086. ⚠ **But the surviving reads do move**: evicting 5 shared items
  changes the rendered features of **100%** of surviving requests by **4.88%** relative. That is the
  honest price of item-level deletion — an atom is a whole training item and it serves every well — and
  it is exactly why a CONJUNCTION deletion must be a hole and not an eviction.
- **23:40Z** Refusal calibration, on val: the gap rule is a distance cut, **not** a discriminator — at
  `gap_tau=1.5` it refuses 0.195 of SEEN and 0.292 of unseen two-way requests, because the within-value
  spread is comparable to the between-value gap on this encoder. Pinned to **4.0**, where it costs
  0.0039 on seen and 0.0000 on unseen two-way. ⭐ The two mechanisms that DO discriminate are
  independent of it: the empty basin (1.0000 on a deleted value at every threshold tried) and the hole.
  ⛔ Wave 5 should not build a refusal claim on the geometric gap of THIS encoder.
- **23:55Z** Reproducibility fix found by rebuilding: two builds of the identical store differed by
  362 / 1,823 / 1,081 bytes, all inside `coupling_v`/`coupling_u` — LAPACK does not promise the same
  sign per singular component. Signs are now canonicalised (largest-magnitude entry of each right
  singular vector positive). **Rebuild-vs-rebuild now differs by 0 bytes at all three seeds**
  (sha `c4ab607d…`, `4245cfe4…`, `bef0130e…`). Re-running every field-derived smoke on the canonical
  store so the whole artifact set is one object.
- **2026-09-19 19:58Z — ⚠ CORRECTION, TIMESTAMPS.** The bullets above from "18:10Z" onwards carry
  times that are too late, by up to about four hours. Cause: this box's local clock is **UTC+0200**
  and I read file modification times as if they were UTC. ⛔ The bullets are NOT rewritten (charter
  section E.2); this bullet is the correction. Authoritative anchors, from `date -u`:
  `START_UTC` **16:53:13Z** · partition written **16:58Z** · seed-0 field rows (final, canonical store)
  **19:44Z** · this bullet **19:58Z**. **Total wall clock at this point: 3 h 05 m of the 10 h budget.**
  The ORDER of every bullet is correct and is what the log is for; only the stamps drifted. Times from
  here are taken from `date -u`.
- **20:05Z** ⭐ **The row-3 deficit is an ADDRESS problem, not a CONTENT problem — measured, via a
  declared NON-ARM.** `--stage ceiling` projects the TRUE encoder feature onto the 64-item span and
  judges it: that is the best any read of these atoms could do. (Construction borrowed from
  `hub/read`'s `oracle_span`, and kept in the same declared NON-ARM status: it reads a feature, so it
  can never be quoted as a mechanism.) On the two-way corners the atom span reads **shape at 0.981**
  while the deployed read was getting 0.453 — the content is there and the address was losing it.
  ⛔ By contrast `scale` (0.610) and `orientation` (0.721) are CONTENT ceilings: no address fixes them.
- **20:10Z** ⭐ Fix adopted, selected on val: **one-step DE-CONTAMINATION of the marginal term
  coordinates, applied only where the pool is unbalanced.** `a_f[v] = b_f[v] - sum_g sum_u
  (p(u|f=v) - p(u)) b_g[u]`, stored as two more coupling columns (`coupling_dp_fg`, `coupling_dp_gf`,
  10,576 B). The bracket is **exactly zero on a balanced pool**, so it touches only the cells a
  held-out conjunction actually distorted — which is what separates it from backfitting, which
  re-estimated everything and was worse. Strength chosen at **0.5**: shape on the two-way corners
  0.453 -> 0.568 (1.0 gives 0.629 but costs `object_hue` 0.998 -> 0.893 and row-1 exact 0.584 -> 0.522).
- **20:20Z** Byte properties re-asserted on the new 2,054,866 B image, unchanged in kind: evict
  **23,332** B / **0 outside** · hole **1** B / **0 outside** · decay-half **8** B / 0 · decay-to-zero
  **23,465** B / 0 · lifetime **3** B / 0 · negative control 5,762 B. Unseen-pair refusal after the
  hole: **0.0000**. `ALL_BYTE_PROPERTIES_HOLD: True`.

- **20:27Z** ⛔ **A real defect found by re-reading the numbers, and fixed.** After the atom set changed
  (de-contamination moves the address manifold the atoms are selected against), evicting
  `object_hue = 0.0` by PROVENANCE left **1 live atom** in the deleted basin — an atom written for a
  neighbouring hue whose nearest basin is this one. The basin therefore still looked occupied and
  refusal on the deleted value stayed at **0.0417** instead of going to 1.0000. ⛔ The byte property
  was never in question; the deletion-to-refusal LINK was.
- **20:27Z** ⭐ Fix: `evict_term` now vacates the basin **geometrically as well as by provenance** —
  it takes every live atom whose nearest basin in the host well is the deleted term, in addition to
  every atom written holding it. It is still an operation on atoms, it takes strictly MORE of them
  (9 = 8 by provenance + 1 by geometry alone), and more is the conservative direction for an erasure.
  After the fix: **live atoms in the deleted basin = 0**, refusal on the deleted value
  **0.0417 -> 1.0000**, surviving requests refused 0.0043, bytes changed **26,275 / 0 outside**.
- **20:30Z** The FINAL REPORT section below was regenerated at this point so it carries the corrected
  eviction numbers. ⛔ **No running bullet above was touched** — the drafting of the report is itself
  logged here, and the report is the summary, not the log.

---

# FINAL REPORT — `hub/field-store`

**BUILD ONLY. ⛔ Scored runs spent: 0 of 0.** No held-out test corner and no shipped query set was
touched by this hub; every number below is measured on the **val selection condition** (charter §A.2),
on corners whose rows are removed from the fit pool of **every** arm. ⛔ **Nothing here is a claim on
the generation table.** Branch `w4/field-store`, base `9dfed5e`.

## 1. What was delivered

`chlu/core/gen_field.py` (≈1,100 lines) + `chlu/experiments/exp_gen_field_store.py` (≈1,050 lines) +
`tests/test_gen_field.py` (12 tests, all passing). Every element the target file names is present and
is load-bearing:

| element | what it is | where |
|---|---|---|
| factor-aligned wells | inherited verbatim: `0=floor_hue(777) · 1=wall_hue+scale(939) · 2=object_hue(216) · 3=shape(58) · 4=orientation(58)` | `well_of_channel` |
| atoms = training ITEMS in every well | **64 items**, item `i` contributing `x_i[chan_k]` to well `k`; 64 × 2048 × 4 = **524,288 B**, the wave-3 store ledger exactly | `atoms` |
| per-item LIFETIME | one float per item, shared across wells because the item is one thing | `atom_life` |
| kernel width per well | median within-value spread per well: `6.17 / 9.60 / 5.62 / 0.78 / 1.66` | `well_sigma` |
| basins per value, **with gaps** | basin = term `(f,v)`; centre = the term coordinate in its host well; radius = within-value spread; gap = distance to the nearest sibling basin | `basin_sigma`, `basin_gap` |
| **coupling table between wells** | four columns: a rank-8 **coordinate correction**, the smoothed **evidence weight**, the shrunk **PMI**, and a one-byte **HOLE** per pair; plus two de-contamination columns | `coupling_u/_v/_w/_dp_fg/_dp_gf/_hole` |
| term → coordinate map | a marginal count table, 57 terms × 2048 float32 | `term_mu`, `global_mu` |
| deletion = evict or decay | `evict_term`, `decay_items`, `set_lifetime`, `punch_hole`. ⛔ **No function in the module refits anything** | — |

**The store IS a byte image.** One `uint8` buffer with a segment table; every accessor is a typed view
into it. That is not a packaging choice — it is what makes the acceptance property a literal statement
about bytes rather than about a re-serialisation.

**Byte ledger** (seed 0, sha `a6cb923d2ecc6d21…`, rebuild-reproducible to 0 bytes):

| | field store | wave-3 incumbent |
|---|---|---|
| **landscape (the budget)** | **524,288 B** ✓ matched | 524,288 B |
| interface / addressing | **1,527,526 B** (coupling 1,047,818 · term map 475,136 · rest 4,572) | head **1,766,400 B** |
| provenance (write/delete only, never read) | 2,732 B | — |
| **whole deployed image** | **2,054,866 B** | 2,290,688 B |

⭐ The field store's addressing is **236,874 B cheaper** than the head it replaces, at a byte-identical
landscape ledger.

## 2. ⭐⭐ ACCEPTANCE 2 — a deletion changes ZERO bytes outside the evicted atoms and the hole

**It holds EXACTLY.** Counts are changed-byte positions from a direct comparison of the two buffers.

| operation | bytes changed | **OUTSIDE the permitted region** | segments touched |
|---|---|---|---|
| **evict** `object_hue = 0.0` (**9 items** = 8 by provenance + 1 by geometry) | **26,275** of 2,054,866 | **0** | atoms, atom_life, atom_alive, atom_row, atom_basin |
| ⭐⭐ **hole** — the RED CYLINDER, `object_hue=0.0 ∧ shape=cylinder` | **1** | **0** | `coupling_hole` only |
| decay ×0.5 on 8 items | **8** | **0** | `atom_life` only |
| decay to zero (8 items die) | **23,465** | **0** | atoms + the per-item rows |
| per-item lifetime := 0.25 on 3 items | **3** | **0** | `atom_life` only |
| ⚠ **NEGATIVE CONTROL** — nudge ONE address-book row by 1e-3 | **5,762** | — | `term_mu` |

`assert_no_refit` passes on every operation: `well_of_channel · well_sigma · basin_sigma · basin_gap ·
basin_n · term_mu · term_n · global_mu · coupling · coupling_w · coupling_u · coupling_v ·
coupling_dp_fg · coupling_dp_gf` are **byte-identical** across all of them. The negative control is
there because an assertion that has never fired is not an assertion.

### ⭐⭐ The hole, and why it is the escape from wave 3's trap

Wave 3 proved a per-factor gate changes **zero bits** on a conjunction deletion, and that the joint
table which *can* express it refuses every row-3 and row-4 request. Here, unseen and deleted are
**different numbers in different arrays**:

| after ONE byte | rate |
|---|---|
| hole flag on the deleted conjunction (256 probes) | **0.0000 → 1.0000** |
| hole flag on `object_hue=red` alone | **0.0000** |
| hole flag on `shape=cylinder` alone | **0.0000** |
| hole flag on neither | **0.0000** |
| refusal changed anywhere on the 256 general probes | **0.0000** |
| ⭐ **96 UNSEEN two-way conjunctions refused** — i.e. exactly what every row-3 request is | **0.0000** |

The red cylinder's own coupling entry reads `pmi = 0.0, evidence weight = 0.0` **both before and
after**: it was already an unseen pair and was being answered; one byte in a different array turned it
into a refusal **without touching any other unseen pair**. ⭐ **The trap does not re-appear.**

⚠ **Declared honestly, because wave 5 must plan around it:** the hole changes the *refusal*, not the
render — `rendered_features_bit_identical = True`. That is the correct semantics for refusing a
conjunction, and it is stated rather than hidden. Eviction is the operation that changes the render.

### Eviction: what it buys and what it costs

⭐ **It buys the deletion-to-refusal link, exactly.** After evicting `object_hue = 0.0`:
`live_atoms_in_deleted_basin_after = 0`, refusal on requests for the deleted value goes
**0.0417 → 1.0000**, and refusal on surviving requests is **0.0043**. Eviction vacates the basin
**geometrically as well as by provenance** — every live atom whose nearest basin in the host well is
the deleted term goes too, not only the atoms written holding it. ⚠ That is not decoration: taking
provenance alone left one interloper behind, the basin still looked occupied, and refusal on the
deleted value stayed at 0.0417. It is still an operation on atoms and it takes strictly MORE of them,
which is the conservative direction for an erasure.

⚠ **And eviction's price, declared:** evicting 9 shared items moves the rendered features of **100 %**
of surviving requests by **7.07 %** relative. An atom is a whole training item and it serves every
well. ⭐ That is precisely why a conjunction deletion has to be a hole and not an eviction — a hole
leaves every render bit-identical and refuses exactly one conjunction.

## 3. ACCEPTANCE 1 — smoke on rows 1, 3, 4 within the wave-3 aligned store's spread

Same pool, same partition, same judge, same corners, paired corner by corner, 3 store-fit seeds.
Row 1 = 512 seen requests drawn from the pool · row 3 = **60** val two-way corners over three
`(object_hue, shape)` conjunctions *every row of which* is removed from the pool · row 4 = the **24**
val Montero n-way corners. Rows 3/4 score **distinct valid NOVEL tuples per corner**.
⭐ Labels are used ONCE, at fit time, to align the wells and to build the address book.

| arm | r1 judge acc | r3 novel/corner | r4 novel/corner | seeds |
|---|---|---|---|---|
| **field** | **0.5820 ± 0.0256** | **9.767 ± 1.135** | **31.375 ± 3.927** | 3 |
| `aligned_w3` (incumbent) | 0.5579 ± 0.0171 | 10.733 ± 0.481 | 26.472 ± 3.783 | 3 |
| `field_landscape_removed` | 0.0000 | 0.000 | 0.000 | 1 |
| `field_landscape_shuffled` (coordinates scrambled) | 0.0000 | 0.050 | 0.000 | 1 |
| `field_landscape_index_permuted` | 0.6055 | 11.000 | 35.125 | 1 |
| `aligned_landscape_removed` | 0.0000 | 0.000 | 0.000 | 1 |

**Bands declared before the first smoke number** (charter §A.3, never moved): ±1.0 novel tuple per
corner on rows 3 and 4 — the wave-3 hub's own decoration margin — and ±0.02 absolute accuracy on row 1.

| row | per-seed paired delta (field − incumbent) | mean | band | verdict |
|---|---|---|---|---|
| **r3 two-way** | +0.783 · −2.400 · −1.283 | **−0.967** | 1.0 | ⭐ **WITHIN the declared band** |
| r4 n-way | +11.042 · +7.208 · −3.542 | **+4.903** | 1.0 | outside — **field ahead** (+1.30 incumbent seed SD) |
| r1 seen | +0.066 · +0.014 · −0.008 | **+0.024** | 0.02 | outside by 0.004 — **field ahead** |

⭐ **Read plainly: the redesign does not cost composition.** The row it was at risk on, row 3, is
inside the band; rows 1 and 4 fall outside it in the *favourable* direction, which is not what "the
redesign costs something" looks like. ⚠ The seed-to-seed swing on row 4 is large in both directions
(+11.0, +7.2, −3.5) and larger than the mean; **row 4 should be read as a tie with wide spread, not as
a win**, and this hub claims nothing from it.

### The charter §C controls
- ⭐ **§C.2 landscape-removed:** zeroing the atoms takes the field read to **exactly 0** on every row
  — by construction, since the read is a projection onto the atom span. The incumbent's control
  collapses to 0.0000 too.
- ⭐ **§C.2 landscape-shuffled:** permuting each well's **channel axis** — same atoms, same bytes, same
  multiset of numbers, coordinates scrambled — collapses to 0.0000 / 0.050 / 0.0000.
- ⭐ **A DECLARED NO-OP, reported as a positive structural result.** Permuting atom **indices**
  reproduces the field arm *exactly* (0.6055 / 11.000 / 35.125 at seed 0). Wave 3's shuffle works by
  breaking a head's coefficient→mode correspondence; this store has none. It weights atoms by
  **distance**, so an index permutation is invisible to it. ⭐ That is charter §B.2 — *addressed, not
  indexed* — made testable, and the arm is kept so the statement stays testable.
- ⛔ **§C.1 / §C.3 / §C.4 (matched-bytes lookup floor, tie-is-the-win, nearest-atom diagnostic) are
  NOT RUN here and are NOT a null.** They gate a *scored* claim, this hub has none, and the full rival
  set (`RIVALS-READY.json`) did not exist while this hub ran. Wave 5 runs them.

## 4. What wave 5 must know before it is scoped on this object

1. ⭐ **The address book alone already beats the wave-3 head** on val reconstruction — 0.1610 / 0.3015 /
   0.1910 on rows 1/3/4 against 0.1911 / 0.3264 / 0.2216 — at fewer interface bytes. ⛔ **The 64-item
   span is what gives that lead back** (it costs 0.091 / 0.036 / 0.064). The content, not the address,
   is now the binding constraint.
2. ⭐ **The content ceiling, per factor** (declared NON-ARM: it reads a feature — true feature
   projected onto the atom span / true feature, seed 0):
   | row | floor_hue | wall_hue | object_hue | scale | shape | orientation |
   |---|---|---|---|---|---|---|
   | r1 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | **0.857**/0.994 | 0.998/0.998 | **0.877**/0.994 |
   | r3 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | **0.610**/0.969 | 0.981/0.986 | **0.721**/1.000 |
   | r4 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | **0.882**/1.000 | 1.000/1.000 | **0.748**/0.988 |
   ⛔ **`scale` and `orientation` are CONTENT limits — no better address fixes them.** They are where
   the headroom is, and changing what the basins hold is the intervention that would move them.
3. ⛔ **Do not build a refusal claim on this encoder's geometric gap.** Gap / kernel width per well is
   `2.68 / 1.62 / 1.50 / 1.42 / 0.81`: the basins **overlap**. At `gap_tau=1.5` the gap rule refuses
   0.195 of SEEN requests and 0.292 of unseen two-way ones — it does not separate them. Pinned to
   **4.0**, where it costs 0.0039 / 0.0000. ⭐ The two mechanisms that DO discriminate are independent
   of it: **the empty basin** (0.0417 → **1.0000** on a deleted value, with
   `live_atoms_in_deleted_basin_after = 0`) and **the hole** (1.0000 on its conjunction, 0.0000
   everywhere else). ⚠ And the empty-basin rule only works because eviction vacates the basin
   geometrically — see §2; deleting by provenance alone silently breaks it.
4. ⛔ **Three negative results, each debugged before being abandoned, so nobody repeats them.**
   (a) The coupling as a multiplicative read **gain** HURT (val 0.3566 → 0.3617): the surviving PMI on
   this pool is the value hold-out's own depletion structure, not a property of shapes3d, which is
   fully crossed. Pinned to 0; the table earns its place through its *coordinate* correction instead.
   (b) **Backfitting** the term map is worse than marginal means on every row (0.2556 / 0.3537 / 0.2810
   after one sweep, 0.2837 / 0.4061 / 0.2823 after four, converged by twelve, against 0.1610 / 0.3015 /
   0.1910). It buys pool residual with compensating effects that do not survive to held-out corners.
   `term_map="backfit"` is kept in the code so the negative result is reproducible rather than folklore.
   (c) A plain **Latin design** over the factor grid is a worse atom set than greedy span selection
   against the address manifold (0.2509 / 0.3382 / 0.2542 against 0.2048 / 0.3297 / 0.2410).
5. ⚠ The remaining per-factor gap on row 3 is `shape`: field **0.600** against the incumbent's 0.746
   (mean of 3 seeds), against a span ceiling of 0.981. ⭐ The field store is ahead on `object_hue`
   (0.928 vs 0.864) and on `orientation` (0.802 vs 0.683). One-step de-contamination at strength 1.0
   takes shape to 0.629 but costs `object_hue` and row 1; **0.5 is the pinned compromise and the
   trade-off curve is in the code beside the pin.**
6. ⛔ **DECLARED NOT-RUN, not a null: the bridge-seed spread.** The frozen feature cache is full scope
   for **seed 0 only** — `features.npy` is 480,000 rows at seed 0 and **9,288 rows** at seeds 1 and 2
   (verified by shape). Extending it means writing inside the read-only frozen instrument and
   re-encoding ~960,000 images; a hub may do neither. The three seeds reported here are **store-fit**
   seeds at bridge seed 0: they move the fit-pool draw, the incumbent's k-means++/adam stream and the
   field store's atom selection, **identically for both arms**. Wave 5 should re-measure the
   instrument's own spread when a full-scope generation-2 cache exists at seeds 1 and 2.

## 5. Reproduce it

```
python -m chlu.experiments.exp_gen_field_store --stage corners --out DIR
python -m chlu.experiments.exp_gen_field_store --stage sweep   --seed 0 --out DIR
python -m chlu.experiments.exp_gen_field_store --stage run  --arm field      --seed {0,1,2} --out DIR
python -m chlu.experiments.exp_gen_field_store --stage run  --arm aligned_w3 --seed {0,1,2} --out DIR
python -m chlu.experiments.exp_gen_field_store --stage run  --arm field_landscape_{removed,shuffled,index_permuted} --seed 0 --out DIR
python -m chlu.experiments.exp_gen_field_store --stage ceiling --seed 0 --out DIR
python -m chlu.experiments.exp_gen_field_store --stage delete  --seed 0 --out DIR
python -m chlu.experiments.exp_gen_field_store --stage report  --out DIR
```
Every stage is resumable and wall-clock capped; a kill costs one corner, not the run. The pinned
configuration is in `exp_gen_field_store.PINS` with the measurement that chose each dial written beside
it. Store rebuilds are **byte-identical** (SVD signs canonicalised), sha
`a6cb923d2ecc6d21…` / `a96b26227ab87795…` / `d45219d8b1867aa1…` at store seeds 0 / 1 / 2.

## 6. Cost line

**Wall clock 3 h 40 m** of the 10 h budget (`START_UTC` 2026-09-19T16:53:13Z -> 20:33Z).
**Local CPU only, 3 threads**, as allocated; ⛔ **no GPU was requested and none is justified** — this
hub's best case is a built object, not a headline number (charter §D). ⛔ **0 scored runs spent of 0
allowed.** ⛔ No sub-agent was spawned, so none can still be running. Artifacts ≈ 70 MB under
`/c/chlu/outputs/hub-field-store`, the only directory this hub wrote to.

## 7. The one-line verdict

⭐ **The field store is built, and its central property holds exactly: a deletion changes zero bytes
outside the evicted atoms and the hole — 26,275 B for an eviction, 1 B for the hole, 0 outside either
— and the hole refuses one conjunction, `object_hue=red ∧ shape=cylinder`, while 96 unseen
conjunctions stay answered at 0.0000 refusal and every other render stays bit-identical.**
The smoke lands inside the declared band on row 3 and ahead of the incumbent on rows 1 and 4, at a
byte-identical landscape ledger and a cheaper interface. ⛔ It claims nothing; wave 5 scores it.

---
- **20:35Z — CLOSE.** `w4-field-store.bundle` written to `/c/chlu/outputs/hub-field-store/`
  (4,447,836 B, `git bundle verify`: *the bundle records a complete history*, ref
  `refs/heads/w4/field-store` at `1f86711`, 951 commits — the branch WHOLE). Artefacts verified by
  directory listing, not by `git status`: `LOG.md` · `START_UTC.txt` · `w4-field-store.bundle` ·
  `runs/{base,fs1}` with `corners.json`, `sweep_seed0.json`, `report.json`, per-seed
  `deletion.json` / `content_ceiling.json` and every arm's `arm.json` / `rows.npz`; 57 MB total, all
  inside `CHLU_HUB_OUTPUTS`. ⛔ **No sub-agent was spawned by this hub, so none can still be running**
  (`ps` shows 0 of this hub's processes alive). ⛔ **0 scored runs spent of 0 allowed.**

---

## REPORT ADDENDUM — 2026-09-19, after `hub/read`'s scored-corner ceiling

⛔ The FINAL REPORT above is unchanged. This addendum is appended, not merged, because it rests
partly on a measurement **another hub made and I did not verify**, and partly on new work of mine.

- **20:40Z — INPUT FROM `hub/read` (reported, NOT verified by this hub).** It measured `oracle_span`
  per factor on the **scored** corners (seed 1, 49 n-way, generation 2, its judge) and reports: rival
  `mlp_180k` **55.653** novel tuples/corner with scale **0.8171** / orientation **0.9691**;
  `oracle_span` **50.898** with 0.8024 / 0.8804; its own arm 44.408; the wave-3 read 45.000. ⭐ Its
  conclusion, which I am passing on rather than endorsing: a parameter-matched generic learner beats
  the **oracle address** on the wave-3 store, so no read of that store can reach it and the binding
  constraint is what the atoms carry — this hub's dial.
- **20:40Z — ⛔ THE WARNING THAT MATTERS TO MY OWN REPORT.** `hub/read` measured the same ceiling on
  val first, got **orientation 0.550**, concluded "the atoms cannot carry orientation", and was wrong:
  on the scored corners it is **0.8804**. `hub/placement` independently saw a val margin shrink 8.6x
  on the scored set. ⛔ **My §4 item 2 ceiling is VAL-measured**, on Montero slots 49–72 of the same
  enumeration whose slots 0–48 are scored. ⛔ **It bounds this OBJECT on those corners and must not be
  read as a statement about the scored rows.** ⭐ Corrected reading of my own §4 item 2: `scale` and
  `orientation` are the factors with the headroom — that part survives — but the LEVELS (0.610 /
  0.721) are not the scored-corner levels and the scored-corner ceiling is the first thing wave 5
  should measure.
- **20:45Z — ⭐ STRUCTURAL CHECK I RAN MYSELF, at zero scored cost** (it reads the condition's own cell
  definitions, no held-out number; now emitted as `corner_geometry` in `corners.json`). `hub/read`'s
  claim about the figure of merit is **exactly right**, and — the part it could not know —
  **my val corners have the IDENTICAL geometry to the scored ones**:
  | corner set | floor/wall/object hue | scale | shape | orientation |
  |---|---|---|---|---|
  | SCORED row 3 (60) | 2 / 2 / **1 (constant 60/60)** | 3 | **1 (constant 60/60)** | 3 |
  | SCORED row 4 (49) | 2 / 2 / 2 | 3 | **1 (constant 49/49)** | 3 |
  | my val row 3 (60) | 2 / 2 / **1 (constant 60/60)** | 3 | **1 (constant 60/60)** | 3 |
  | my val row 4 (24) | 2 / 2 / 2 | 3 | **1 (constant 24/24)** | 3 |
  ⭐ So the *shape of the figure of merit* transfers even though the *level* does not: shape is a
  per-corner GATE, and distinct-tuple COUNT is driven by scale x orientation. Every arm I measured
  reads all three hues at 1.0000.
- **20:50Z — ⭐ That re-reads my own row-3 and row-4 results, and it is a better decomposition than the
  one in §3.** Per-corner, pooled over 3 seeds:
  | row | arm | shape | scale | orientation | novel/corner | corr(scale x orient, novel) |
  |---|---|---|---|---|---|---|
  | r4 | **field** | 1.000 | **0.637** | **0.677** | **31.375** | 0.994 |
  | r4 | aligned_w3 | 1.000 | 0.604 | 0.596 | 26.472 | 0.994 |
  | r3 | **field** | **0.600** | 0.599 | **0.802** | 9.767 | — |
  | r3 | aligned_w3 | 0.746 | 0.598 | 0.683 | 10.733 | — |
  ⭐ On row 4 the novel-tuple count is **determined** by scale x orientation (r = 0.994 for both arms),
  and the field store is ahead on **both** — that is the whole of its +4.903.
  ⭐ On row 3 the deficit is **the shape gate and almost nothing else**: restricted to the 42 of 180
  corner-seeds where **both** arms read shape correctly, the paired delta goes **-0.967 -> -0.429**.
  On the two factors that actually generate distinct tuples the field store is level on scale
  (0.599 vs 0.598) and **ahead on orientation** (0.802 vs 0.683).
- **20:52Z — ⭐ The handoff, sharpened.** `hub/read`'s bar is orientation **0.9691** at scale ~0.82,
  against an atom-span ceiling of 0.8804 / 0.8024 on the wave-3 store. ⭐ **A field store is free to
  move that ceiling, because it chooses WHICH 64 items are written.** This hub already selects them
  greedily against the address manifold; selecting them against **orientation and scale
  reconstruction specifically** is a one-function change in `_span_items` (swap the target set) and is
  the obvious first experiment for wave 5.
- **20:55Z — ⛔ DECLARED NOT-RUN, and it is not a null: the per-factor ceiling on the SCORED corners.**
  It is a number on the held-out set, therefore a scored run under charter §A.2, and this hub's budget
  is **ZERO**. ⭐ The capability is now wired and **guarded**: `--stage ceiling --condition test`
  refuses unless `CHLU_FIELD_SPEND_SCORED_RUN` is set, and the refusal message names the PREREG and
  `/c/chlu/SCORED-LEDGER.md` requirements. Verified to refuse.
- **20:56Z — ⚠ A latent bug found while wiring the above and fixed.** `--stage corners` had referenced
  an undefined `seed` since the store-seed refactor (it should have been `args.seed`). ⛔ It affected
  nothing already reported — no smoke, fit, run, delete or report stage touches that line, and
  `corners.json`'s structural assertions had been produced before the refactor — but the stage would
  have crashed for wave 5. Fixed, rerun, and `corners.json` regenerated with `corner_geometry` in it.
- **20:59Z — CLOSE (2).** Addendum work committed as `e567048`; bundle regenerated
  (4,450,184 B, complete history, tip `e567048`). 12/12 unit tests still pass. ⛔ Still **0 scored
  runs spent of 0 allowed** — the scored-corner ceiling is wired, guarded and NOT RUN. Wall clock
  4 h 06 m of the 10 h budget.
- **21:05Z — ⚠ CORRECTION, and it is `hub/read`'s catch, not mine.** My 20:50Z bullet reported that
  restricting to the 42 of 180 corner-seeds where **both** arms read shape correctly moves the row-3
  paired delta from **-0.967 to -0.429**. ⛔ **That is not a corrected effect size and I should not
  have written it as one.** The subset is selected on an outcome both arms produce, so the survivors
  are not a random half and the comparison is conditional-on-success. ⭐ **The row-3 headline stands at
  -0.967**, which is what §3 of the report says and what it continues to say.
- **21:10Z — ⭐ The unconditioned replacement, now computed for EVERY corner and emitted by
  `--stage report` as `paired_estimand_gate_held`.** The scoreboard's geometry says `shape` is held
  constant in every corner, so it can only GATE validity and can never generate a distinct tuple. So
  score the same metric with the gate supplied by the **corner** instead of by the judge: the judged
  non-shape factors must all be right, and the tuple is completed with the corner's own constant
  shape. ⛔ It conditions on nothing the arm produced — all 60 (r3) and 24 (r4) corners are in it at
  every seed.
  | row | scoreboard metric (the headline) | **gate-held decomposition** | shape read-back: field / incumbent |
  |---|---|---|---|
  | r3 | +0.783 · -2.400 · -1.283 → **-0.967** | +4.533 · +1.483 · +1.083 → **+2.367** | 0.600 / 0.746 |
  | r4 | +11.042 · +7.208 · -3.542 → **+4.903** | +11.042 · +7.208 · -3.542 → **+4.903** | 1.000 / 1.000 |
  ⭐ **The whole row-3 deficit is the shape gate**: hold the gate and the field store is +2.367 ahead
  on the axis that actually produces distinct tuples, having been -0.967 behind on the metric.
  ⭐ **And the decomposition validates itself on row 4**: there both arms read the gate at **1.000**, so
  holding it should change nothing — and it changes nothing, to the digit, at all three seeds. A
  diagnostic that moved row 4 would have been measuring its own construction.
  ⛔ **`paired_estimand_gate_held` is a DIAGNOSTIC and is not the scoreboard metric.** It is not
  quotable as a result and this hub quotes nothing. The headline for row 3 is **-0.967**, inside the
  declared +-1.0 band.
- **21:12Z — Where that leaves the handoff.** Two separable jobs for wave 5, and they are not the same
  job: (1) **the gate** — `shape` on two-way corners, field 0.600 against the incumbent's 0.746, with
  an atom-span ceiling of 0.981 on my val corners, i.e. an ADDRESS problem; (2) **the distinctness
  axis** — `scale` x `orientation`, where `hub/read` reports a matched MLP at 0.8171 / 0.9691 on the
  scored corners against an oracle-span ceiling of 0.8024 / 0.8804 on the wave-3 store, i.e. a CONTENT
  problem, and the one a field store can attack because it chooses which items are written.
- **21:15Z — CLOSE (3).** Decomposition committed; bundle regenerated (complete history, tip below).
  12/12 tests pass, worktree clean. ⛔ Still **0 scored runs spent of 0 allowed**. Wall clock
  4 h 22 m of the 10 h budget.
