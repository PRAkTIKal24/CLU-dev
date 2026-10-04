# w6.1 hub/compose-physics — AUTO-HUB RETRY, kind R-B (mechanism) — LOG

**Parent:** `w6 hub/compose-physics` (finished 2026-09-20T08:20Z). **Base:** its tip `25fc108`.
**Scored budget: ZERO.** Nothing here is scored; everything is re-analysis of the parent's
already-paid legs, plus arithmetic on artifacts already on disk. ⛔ No arm is re-run and no
`val` or `test` row is re-scored. The judge is constructed **once**, for one `val`-only probe
(`w61_judge_margin.py`, section "The judge's margin"): it is run on synthetic perturbations of
`val` states, never on a held-out corner and never to score an arm. Every held-out number in
this LOG comes from the parent's `rows.npz`.

**The one job:** test — not assert — the hypothesis that *settling costs composition because it
moves the state off the per-well principal subspace `span(G_k)` the content spans.*

---

## 2026-09-20

- **08:24Z** `START_UTC.txt` written to `/c/chlu/outputs/w61-compose-physics/`. ⚠ The first
  write of it landed in the *parent's* directory by mistake and clobbered the parent's own
  `START_UTC.txt`; restored it at once to the value the parent's own LOG records
  (`06:26Z`, line 12) with a comment saying so. Seconds are unrecoverable and the file says that.
- **08:25Z** Read, and read only, `/c/chlu/tasks-w6/w6.1-compose-physics.md`. Worktree
  `/c/chlu/worktrees/CHLU-w61-compose-physics`, branch `w6.1/compose-physics`, tree clean at
  boot, HEAD `25fc108` = the parent's tip. No venv created; system python, `PYTHONPATH` as
  given. `PYTHONIOENCODING=utf-8` set for every python call (the repo's docstrings carry
  non-cp1252 glyphs and the default Windows console codec raises on them).
- **08:26Z** Re-verified the objects the retry builds on, rather than trusting the parent's LOG.
  Content = 5 wells, `dim` 2048, atom block `64 × (777, 939, 216, 58, 58)` float32 =
  **524,288 B**, `variance_explained_per_well` = 0.973 / 0.957 / 0.960 / **1.000 / 1.000**.
  `span_basis` ranks **64, 64, 64, 58, 58 = 308**. ⭐ Wells 3 and 4 are **full rank**: their
  span is the whole channel block, so in those two wells there is no "outside the subspace"
  to move to even in principle. Coupling `span_coupling_r16`: 10 pairs, 78,848 B + 2,464 B
  standardisation = **81,312 B**, `lambda*` = 0.250, `lambda_used` = 0.225 (the certificate
  **did** bind), fitted in 11.9 s on 131,072 pool rows.
- **08:28Z** Wrote `scripts/w61_settle_subspace.py`. It reloads each scored leg's own
  `arm.pkl`, re-emits every request of the 49 row-4 corners through **the leg's own read
  path** (`R.arm_features`, exactly as `fw.run_arm` calls it), and decomposes the emitted
  state per well into its component **inside** `span(G_k)` and **orthogonal** to it. It joins
  the per-request `valid` flags already on disk in `rows.npz`; ⛔ the judge is never loaded.

### ⭐⭐ The subspace hypothesis is FALSE — and it is false by construction, not by a close call

- **08:30Z** Ran it on the scored leg `test_seed0` (49 corners × 72 requests = **3,528**
  requests, accept rate 1.000). Result, per leg, on the SCORED row-4 set:

  | leg | validity | median rel. displ. | **off-span fraction of the STATE** (median / max) | **off-span fraction of the DISPLACEMENT** (median / max) |
  |---|---|---|---|---|
  | `pca_tensor` | 0.8835 | 0.0000 | 0.0 / **1.96e−7** | — |
  | `pca_settle_zerostep` | 0.8835 | 0.0000 | 0.0 / 1.96e−7 | 0.0 / 3.9e−10 |
  | `pca_settle_f05_conv` | 0.8761 | 0.0085 | 0.0 / 1.94e−7 | 0.0 / **4.6e−5** |
  | `pca_settle_onestep` | 0.8455 | 0.0415 | 0.0 / 1.91e−7 | 0.0 / 9.0e−6 |
  | `pca_settle` | 0.3285 | 0.3242 | 0.0 / 1.92e−7 | 0.0 / 1.5e−6 |

  ⛔ **Every leg's state lies inside `span(G_k)` to seven significant figures, and so does the
  incumbent's** — the 1.9e−7 is float32 rounding in `render_span`'s matmul and it is *the same
  size for `pca_tensor` itself*, which never moves at all. The **displacement** is off-span by
  at most 4.6e−5 of its own length. **Not one unit of displacement leaves the subspace.**
- **08:31Z** The reason is structural and visible in the code, which is why the number is not a
  near miss: the settle is parameterised **in span coordinates** (`x_k = U_k z_k`) and emits
  through `render_span(store, Z, basis)` — the *same* function, the *same* basis, as
  `TensorAddress.read`. The dynamics move `z`, and `U_k z` is in `span(G_k)` for every `z`.
  The state **cannot** leave the subspace; the hypothesis has no way to be true here.
- **08:31Z** And the orthogonal fraction has **no predictive power** over the loss it was
  supposed to explain: AUC(off-span fraction → request is invalid) = 0.527 / 0.527 / 0.525 /
  0.520 / 0.515 across the five legs — chance, and if anything *decreasing* as the loss grows.
  ⭐ **The subspace explanation is dead. Reported as dead.**

### ⭐ What replaces it: the answer lattice is dense, and the dynamics step across it

- **08:31Z** Same recomputation, different yardstick. The judge does not have to see the
  state in a subspace; it has to tell this request's answer from the **neighbouring** answer.
  Measured on the incumbent's own emitted answers over the scored set: the median distance to
  the nearest answer **differing in exactly one factor** is **ρ₁ = 1.709**, against a median
  state norm of **31.30** — the code is dense: one factor step is **5.5 %** of the state.
  Displacement in those units (median `‖Δ‖/ρ₁`, and the fraction of requests moving more than
  half way to a distinct answer):

  | leg | median ‖Δ‖ | ‖Δ‖/ρ₁ | frac > 0.5 ρ₁ | validity |
  |---|---|---|---|---|
  | `pca_settle_zerostep` | 0.000 | 0.000 | 0.000 | 0.8835 |
  | `pca_settle_f05_conv` | 0.266 | 0.159 | 0.000 | 0.8761 |
  | `pca_settle_onestep` | 1.299 | 0.777 | **0.995** | 0.8455 |
  | `pca_settle` | 10.231 | **5.862** | 1.000 | 0.3285 |

  ⭐ `onestep` — **one** gradient step at the certified-safe strength — already moves 99.5 % of
  requests more than half the distance to a *different* valid answer. The arm moves nearly six
  neighbours' worth.

### ⭐ zerostep: exact, and what the exactness implies

- **08:32Z** ⚠ First pass reported `bit_identical = False`: 17 of 3,528 requests differed, by
  at most **7.1e−15** absolute in a state of norm 31.3 (2.3e−16 relative). ⛔ That is not the
  settle. Traced it: the two arms' `store.modes` are bit-identical and their address books'
  `E`/`W` are bit-identical (the same cached pickle), but their `basis` arrays differ in
  **8 entries of well 3 and 16 of well 4** (the two full-rank wells) at the 1e−15 level —
  two independent LAPACK SVDs of the same atom block in two different processes, differing in
  the last bits of near-zero entries.
- **08:33Z** ⭐ **With the same `basis` object, the identity is exact and not "within
  tolerance":** re-emitting every request of **all 109 scored corners** (49 row-4 × 72 +
  60 row-3 × 72 = 7,848 requests, **11,649,024** float32 channel values) gives
  **0 differing float32 entries** between `pca_settle_zerostep` and `pca_tensor`. The
  coefficients are equal bit for bit as well.

### ⭐⭐ The mechanism that DOES hold: a dense answer lattice, and motion across it

- **08:34Z** Wrote `scripts/w61_answer_lattice.py`. For every scored request it builds,
  from the **same address book**, the answers to all Hamming-1 neighbours (every other
  value of every factor; the grid is 10/10/10/8/4/15, so the shell is 51 tuples laid out
  on 57 fixed `(factor, value)` columns) and measures the local geometry.
  ⚠ Caught and fixed a measurement artifact on the first pass: `grid.values_of` returns
  float32-rounded values while `grid.values` holds the float64 grid, so the six *self*
  columns read back ~1e-6 from the request instead of exactly 0 and a distance-based
  self test let them through. They are masked **by index** now. (Against a real
  neighbour spacing of ~1.7 the 1e-6 changes nothing else.)
- **08:36Z** ⭐ **The read's answers are a dense lattice, and the spacing differs by
  factor.** Median distance from a request's answer to the nearest answer differing only
  in factor f, seeds 0/1/2, against a median state norm of 31.3 / 32.3 / 32.8:

  | factor | card | rho_f (s0) | rho_f (s1) | rho_f (s2) |
  |---|---|---|---|---|
  | floor_hue | 10 | 7.885 | 8.424 | 9.023 |
  | wall_hue | 10 | 9.087 | 9.632 | 9.242 |
  | object_hue | 10 | 5.103 | 5.413 | 5.486 |
  | **scale** | 8 | **2.041** | **1.996** | **2.119** |
  | shape | 4 | 6.989 | 7.894 | 8.929 |
  | **orientation** | 15 | **1.899** | **2.172** | **2.312** |

  The nearest neighbour of any kind sits at rho_1 = 1.71 / 1.89 / 2.00 — **5.5 % of the
  state's own norm.** The code is dense.
- **08:36Z** ⭐⭐ **The dynamics break exactly the tightly-spaced factors, and only
  those.** Per-factor read-back error rate, seed 0 (all three seeds in
  `MECHANISM-TABLE.md`), with the increase over the incumbent in brackets:

  | factor | rho_f | incumbent | f05_conv | onestep | pca_settle |
  |---|---|---|---|---|---|
  | floor_hue | 7.885 | 0.0000 | 0.0000 (+0.0000) | 0.0000 (+0.0000) | 0.0000 (+0.0000) |
  | wall_hue | 9.087 | 0.0000 | 0.0000 (+0.0000) | 0.0000 (+0.0000) | 0.0275 (+0.0275) |
  | object_hue | 5.103 | 0.0000 | 0.0000 (+0.0000) | 0.0000 (+0.0000) | 0.0558 (+0.0558) |
  | scale | 2.041 | 0.0723 | 0.0760 (+0.0037) | 0.0938 (+0.0215) | 0.5048 (+0.4325) |
  | shape | 6.989 | 0.0077 | 0.0082 (+0.0006) | 0.0116 (+0.0040) | 0.0697 (+0.0621) |
  | orientation | 1.899 | 0.0405 | 0.0442 (+0.0037) | 0.0558 (+0.0153) | 0.1891 (+0.1485) |

  ⭐ On **all three seeds**, `f05_conv` and `onestep` add **exactly zero** errors to the
  three hue factors (rho_f = 5.1–9.6) while damaging scale and orientation
  (rho_f ~ 1.9–2.3). The displacement is the *same* for every factor of a given request
  — what differs is only that factor's answer spacing.
  Spearman(1/rho_f, error increase) = 0.83.
- **08:37Z** Pooled over the three settle legs and six factors on seed 0, restricted to
  requests the incumbent read correctly at that factor (n = 62,229, 2,930 new errors):
  AUC( |progress toward that factor's neighbour| -> new error ) = **0.904**;
  AUC( raw displacement, which is factor-blind ) = 0.826.
- **08:37Z** And the motion is **not** isotropic. The 51 lattice directions span a
  rank-51 tangent inside a 308-dimensional state, so an isotropic step would put
  sqrt(51/308) = 0.407 of its length there. Measured median tangent fraction of the
  settle's displacement: **0.79 / 0.78 / 0.85** (f05 / onestep / the arm) — 3.7–4.4x the
  isotropic share of its energy in exactly the directions that change the answer.

### ⭐⭐ The judge's margin, measured — and the loss predicted from it

- **08:38Z** Wrote `scripts/w61_judge_margin.py` and ran it on **`val` only** (charter
  A.2; ⛔ nothing scored, the held-out legs are only read back afterwards). It takes the
  incumbent's answer to 288 val requests, pushes it s·rho_f **toward** the nearest answer
  that differs only in factor f, and asks the judge what it now reads back — 15,552
  judged states, plus 2,592 isotropic controls at matched lengths. 175 s.
- **08:41Z** ⭐ **The judge's decision cells are the Voronoi cells of the answer
  lattice.** P(read-back of factor f changes), by fraction of the way to the neighbour:

  | s = | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.7 | 1.0 |
  |---|---|---|---|---|---|---|---|---|
  | floor_hue | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 | **0.476** | 1.000 | 1.000 |
  | wall_hue | 0.000 | 0.000 | 0.000 | 0.000 | 0.038 | **0.733** | 1.000 | 1.000 |
  | object_hue | 0.000 | 0.000 | 0.000 | 0.010 | 0.135 | **0.431** | 0.948 | 1.000 |
  | scale | 0.007 | 0.029 | 0.090 | 0.158 | 0.270 | **0.417** | 0.748 | 0.971 |
  | shape | 0.000 | 0.000 | 0.000 | 0.028 | 0.160 | **0.403** | 0.972 | 1.000 |
  | orientation | 0.004 | 0.033 | 0.080 | 0.168 | 0.307 | **0.493** | 0.836 | 1.000 |

  ⭐ Six factors, six different absolute spacings (1.6 to 10.6), and the flip probability
  crosses one half at **s ~ 0.5 in every one of them**. In lattice units the six curves
  are the same curve: the judge tolerates motion up to the midpoint between neighbouring
  answers, and no further.
- **08:41Z** ⭐ The **isotropic control** separates "motion toward another answer" from
  "any motion at all". A random in-span step of length **10.0** — the arm's own median
  displacement — flips floor_hue 0.000, wall_hue 0.000, object_hue 0.017, shape 0.010,
  scale 0.209, orientation 0.230, and costs whole-tuple validity 0.917 -> 0.552. The arm,
  moving the same distance, scores 0.312 on val. ⛔ The dynamics are **worse than random
  motion of the same size**, because they aim along the lattice.
- **08:42Z** ⭐⭐ **The prediction.** Take the val-measured margin CDF M_f, apply it to
  each leg's measured per-request progress p_f, and predict whole-tuple validity as
  mean_i [ valid_i · prod_f (1 − M_f(p_f,i)) ]. ⛔ The margin comes from `val`; the legs
  predicted are the **held-out** ones the parent already paid for, so this is an
  out-of-sample prediction that scores nothing:

  | leg | predicted delta validity (s0 / s1 / s2) | measured delta validity (s0 / s1 / s2) |
  |---|---|---|
  | `f05_conv` | −0.0027 / −0.0029 / −0.0028 | −0.0074 / −0.0071 / −0.0054 |
  | `onestep` | −0.0292 / −0.0327 / −0.0327 | −0.0380 / −0.0527 / −0.0411 |
  | `pca_settle` | −0.629 / −0.747 / −0.740 | −0.555 / −0.652 / −0.543 |

  Per factor, over the 12 (leg, factor) cells of the two gentle legs: Pearson **0.99 /
  0.96 / 0.95** across seeds, MAE 0.003 / 0.005 / 0.002, and all six hue and shape cells
  predicted at exactly 0.0000 and measured at exactly 0.0000. ⚠ The model
  **under**-predicts by 1.2–2.3x (it counts only the projection onto the single nearest
  neighbour per factor and ignores everything else); at the arm's displacement — six
  lattice cells — the one-dimensional model saturates and **over**-predicts. ⭐ It is
  quantitative where it matters: in the regime where the dynamics could have been useful.

### ⛔ The alternatives the mechanism had to beat, tested and eliminated

- **08:42Z** **Standardisation?** No. Round-tripping the address book's coefficients
  through the coupling's standardise / un-standardise with no steps taken moves them by
  **1.18e−8** relative — six orders of magnitude below `f05_conv`'s 0.0085, which is
  itself the smallest displacement that costs anything.
- **08:43Z** **The 81,312 read-side bytes?** Bytes do not read back tuples, and the byte
  ledger is **identical** for `zerostep`, `f05_conv`, `onestep` and `pca_settle` — one
  fitted coupling, four legs — while their losses run 0.000 to −41.98 per corner. Only
  the number of steps moves. Bytes cannot explain a spread that bytes do not vary over.
- **08:43Z** **Coupling capacity?** No. On val the same arm at coupling rank **4 / 16 /
  64** (22,464 / 82,368 / 321,984 read-side B, a **14x** range) scores 22.208 / 22.458 /
  22.750 novel tuples per corner against the incumbent's 64.375. More capacity does not
  rescue it and less does not explain it.
- **08:43Z** **Regression to seen combinations (charter E.4)?** ⛔ **No — and this is
  tested properly for the first time.** The parent measured seen-ness among the *valid*
  tuples, where it is 0 by construction. Measured here on the **errors**: wrong
  read-backs land on fitted combinations 0.7 % / 3.7 % / 0.0 % of the time for the
  incumbent and 15.8 % / 25.0 % / 16.8 % for the arm — but against a **locality-matched
  null** (a tuple drawn uniformly at the same Hamming distance from the same request) of
  0.53–0.59, every leg is **below** chance: 0.0–0.1x for the incumbent and the gentle
  legs, 0.3–0.4x for the arm. The dynamics do not walk back onto the fit pool. ⭐ And the
  errors stay **local**: the mean Hamming distance of a wrong read-back is 1.02–1.05 for
  the gentle legs and 1.26–1.43 even for the arm. One factor at a time.

### The three cheap closures

- **08:44Z** `scripts/w61_close.py`. **Headline re-verified from `rows.npz`, not read off
  the parent's table**: row-4 novel tuples per corner, delta vs `pca_tensor`, seeds
  0/1/2 — `zerostep` **0.000 / 0.000 / 0.000**, `f05_conv` −0.531 / −0.510 / −0.388,
  `onestep` −2.735 / −3.796 / −2.959, `pca_settle` −39.959 / −46.918 / −39.061.
  ⭐ Reproduces the parent exactly.
- **08:44Z** ⭐ **`f05_conv` is a real effect, not noise.** Paired over all 147 corner
  pairs (49 x 3 seeds): 46 worse, 16 better, 85 tied, exact sign test **p = 1.8e−4**,
  mean −0.476 tuples per corner. Paired per request (10,584 requests): 104
  valid->invalid against 34 invalid->valid, McNemar **p = 1.9e−9**. The parent's own
  bootstrap agrees — its row-4 CIs are [−0.878, −0.204], [−1.000, −0.082],
  [−0.796, −0.041], all three entirely below zero. ⚠ **Inside the pre-registered ±1.0
  margin is not the same as indistinguishable from zero**: the margin is an effect-size
  bar, and this effect is smaller than the bar while being nowhere near zero. (On row 3
  the same leg *is* a tie: 60 pair sub-corners have less power than 49 n-way corners.)
- **08:44Z** ⚠ **What the 81,312 B would have to buy.** It is 15.5 % on top of the
  524,288 B store and 11.4 % on top of store + address book (713,632 B). To clear the
  pre-registered bar it must buy **at least +1.0** novel tuples per corner; the best leg
  that actually moves delivers **−0.476**, so the gap to break-even is **>= 1.48 per
  corner, about 2.3 %** of the incumbent's 63.6 — and that is before charging anything
  for the bytes themselves. ⭐ The sharper way to say it: at zero steps those bytes are
  **provably unread** (bit-identical output), so the overhead is 81,312 B of dead weight;
  at every strength that does read them, they buy a loss. There is no setting at which
  they are worth anything.

### ⭐⭐ Three extensions the remaining time bought

- **08:45Z** ⭐⭐ **No dynamics built from these atoms can leave the span — it is an
  invariant subspace.** The one leg with a chance of leaving is `pca_settle_kernel`, the
  banked density descent (charter E.4), because it runs in **channel** space, not in span
  coordinates. Measured on val: its median off-span fraction is **9.9e−8** — smaller than
  the incumbent's own float32 rounding. The reason is one line of algebra: the kernel
  force is `x_k − sum_i p_i a_ki`, and both `x_k` and every atom `a_ki` lie in
  `span(G_k)`, so the span is invariant under the flow. ⛔ The declared hypothesis is
  therefore not merely false for this arm — **it is unfalsifiable by construction for any
  atom-grounded dynamics on a subspace store.** The retry's dial had no way to move it.
  (`pca_settle_kernel` still collapses to validity **0.000** at a displacement of 21.8;
  `pca_settle_lookup` likewise. Leaving the subspace is not what kills them either.)
- **08:46Z** ⭐ **The scored metric IS the validity rate, rescaled.** On every leg, row
  and seed, `delta novel tuples per corner` equals `delta validity × the corner ceiling`
  exactly (row 3 ceiling 36, row 4 ceiling 72) — because SEEN is 0 and every valid tuple
  in a corner is distinct. Row 3's smaller absolute losses (`f05_conv` −0.033 / −0.183 /
  −0.033 against row 4's −0.531 / −0.510 / −0.388) are that rescaling and a slightly
  easier request mix, not a different phenomenon. ⭐ So explaining the **per-request
  validity** — which is what the lattice law does — explains the whole scored metric on
  both rows.
- **08:47Z** ⭐⭐ **The motion budget, and what it means for deletion.** Fitting the local
  slope of validity loss against displacement in lattice units on the two legs that stay
  inside one lattice step, the displacement at which the loss reaches the pre-registered
  **±1.0 novel tuples per corner** is **0.22–0.28 ρ₁ = 1.3–1.6 % of the state's own
  norm** (0.42 / 0.49 / 0.52 absolute, seeds 1 / 0 / 2). ⛔ That is the entire motion
  budget of this read — about a quarter of one lattice step.
- **08:48Z** ⭐⭐ **And now the bridge to the programme's central problem.**
  `scripts/w61_eviction_cost.py` prices the *other* source of motion in the same units:
  for each well and each atom, rebuild the span without that atom, reproject every scored
  answer into it, and measure how far the answer had to move.

  | well | d_k | span rank | median eviction cost | cost of evicting the MEAN | atoms past the 0.487 budget |
  |---|---|---|---|---|---|
  | 0 | 777 | 64 | 0.382 | 9.681 | 24 / 64 |
  | 1 | 939 | 64 | 0.584 | 14.727 | 36 / 64 |
  | 2 | 216 | 64 | 0.295 | 5.346 | 16 / 64 |
  | 3 | 58 | **58** | 1.8e−15 | 2.2e−15 | **0 / 64** |
  | 4 | 58 | **58** | 5.9e−15 | 8.8e−15 | **0 / 64** |

  ⭐ Eviction is **free exactly where the atom block is over-complete** (wells 3 and 4
  hold 64 atoms in a 58-dimensional channel block, so dropping one changes the span not
  at all) and **expensive exactly where the atoms are an efficient basis** — wells 0–2,
  64 independent atoms spanning a proper subspace, where **76 of 192 atoms (39.6 %)**
  individually move the median answer past the read's whole motion budget, and evicting
  a well's mean moves it 10–30 budgets.

---

# FINAL REPORT — w6.1/compose-physics (AUTO-HUB RETRY, kind R-B, mechanism)

**Scored runs spent: ZERO.** Composition stays at 2/3 with run 3/3 BANKED. Everything
below is recomputed from the parent's already-paid artifacts, plus one `val`-only judge
probe. ⛔ No arm was re-run and no held-out row was re-scored.

## 1. The verdict, in one paragraph

**The declared hypothesis is false, and it is false by construction.** The settled state
never leaves the per-well principal subspace `span(G_k)` — not approximately, not mostly:
its off-span fraction is at most **2e−7**, which is the same float32 rounding the
*incumbent* carries while standing still, and the orthogonal fraction predicts the loss
at chance (AUC 0.457–0.559 over three seeds). The settle is parameterised in span
coordinates and emits through the same `render_span` as the incumbent, so `span(G_k)` is
an **invariant subspace** of it; and the one leg that runs in channel space, the banked
kernel descent, is confined to the same subspace for the same reason — its force is built
from the atoms, and the atoms are the span. ⛔ **A subspace read cannot be broken by
motion off the subspace, because atom-grounded dynamics cannot produce such motion.**
What breaks it is motion **inside** the subspace, and the reason is a resolution law:
the read's answers are a **dense lattice** whose neighbouring cells are ~5.5 % of the
state norm apart, the judge's decision cells are that lattice's cells, and the dynamics
step across them.

## 2. What was tested, and what it cost to test

| question | measurement | answer |
|---|---|---|
| does the state leave `span(G_k)`? | per-well projector on all 3,528 scored requests × 3 seeds × 5 legs | **no**: ≤ 2e−7 of the state, ≤ 6.8e−5 of the displacement |
| does the orthogonal fraction predict the loss? | AUC over requests | **no**: 0.457–0.559, chance |
| could any leg leave it? | the channel-space kernel descent, on val | **no**: 9.9e−8; the span is invariant |
| then what predicts the loss? | displacement in units of the per-factor answer spacing | **yes**: AUC 0.904 (seed 0, pooled over legs and factors), and it predicts the held-out numbers |

## 3. The mechanism that does hold

**(a) The read's answers are a dense lattice.** The address book gives every grid tuple
an answer. Measured on the incumbent's own answers, the distance to the nearest answer
that differs in exactly one factor is **ρ₁ = 1.71 / 1.89 / 2.00** against a state norm of
**31.3 / 32.3 / 32.8** — one factor step is **5.5 % of the state**. The spacing is very
uneven across factors: the three hues sit **5.1–9.6** apart, scale **~2.0** and
orientation **~1.9–2.3**.

**(b) The judge's decision cells are that lattice's cells.** Pushing the incumbent's
answer a fraction `s` of the way toward the nearest one-factor neighbour and asking the
judge what it now reads back (a `val`-only probe, 288 requests × 6 factors × 9 scales =
15,552 judged states): the read-back flips with probability that crosses one half at
**s ≈ 0.5 in every one of the six factors** (0.403 to 0.733 at s = 0.5; ≥ 0.97 at s = 1).
Six factors, six different absolute spacings, one curve in lattice units.

**(c) The dynamics step across those cells, and preferentially so.** The legs move
0.16 / 0.78 / 5.86 lattice steps (f05_conv / onestep / the arm). `onestep` — a **single**
gradient step at the certified-safe strength — already moves **99.5 %** of requests more
than half way to a *different* valid answer. And the motion is aimed: the 51 lattice
directions span a rank-51 tangent inside a 308-dimensional state, so an isotropic step
would place √(51/308) = 0.407 of its length there; the settle places **0.78–0.86**, i.e.
3.7–4.4× the isotropic share of its energy in exactly the directions that change the
answer. (A matched isotropic control is correspondingly gentler: a random in-span step of
length 10 costs val validity 0.917 → 0.552, while the arm's own 13.4 costs 0.894 → 0.312
— worse than the ~0.39 a linear extrapolation of the isotropic curve gives at that
length.)

**(d) So the damage is factor-resolved, and it goes exactly where the lattice is
tightest.** The displacement is the *same* for every factor of a given request; only the
spacing differs. On **all three seeds**, `f05_conv` and `onestep` add **exactly zero**
errors to the three hue factors and damage only scale and orientation:

| factor | ρ_f (s0) | incumbent | f05_conv | onestep | `pca_settle` |
|---|---|---|---|---|---|
| floor_hue | 7.885 | 0.0000 | +0.0000 | +0.0000 | +0.0000 |
| wall_hue | 9.087 | 0.0000 | +0.0000 | +0.0000 | +0.0275 |
| object_hue | 5.103 | 0.0000 | +0.0000 | +0.0000 | +0.0558 |
| scale | 2.041 | 0.0723 | +0.0037 | +0.0215 | +0.4325 |
| shape | 6.989 | 0.0077 | +0.0006 | +0.0040 | +0.0621 |
| orientation | 1.899 | 0.0405 | +0.0037 | +0.0153 | +0.1485 |

**(e) And it is quantitative, not directional.** Feed each leg's measured per-request
progress toward its one-factor neighbours through the `val`-measured margin CDF and
predict the **held-out** validity without touching it:

| leg | predicted Δ validity (s0/s1/s2) | measured Δ validity (s0/s1/s2) |
|---|---|---|
| `f05_conv` | −0.0027 / −0.0029 / −0.0028 | −0.0074 / −0.0071 / −0.0054 |
| `onestep` | −0.0292 / −0.0327 / −0.0327 | −0.0380 / −0.0527 / −0.0411 |

Per factor, over the 12 (leg, factor) cells of the two gentle legs: Pearson **0.99 / 0.96
/ 0.95**, MAE 0.003 / 0.005 / 0.002, and all six hue-and-shape cells predicted at exactly
0.0000 and measured at exactly 0.0000. The model under-predicts by 1.2–2.3× because it
counts only the projection onto the single nearest neighbour per factor; at the arm's
displacement of six lattice steps the one-dimensional model saturates and over-predicts.
⭐ It is accurate in the regime where the dynamics could have been worth having.

**(f) And the scored metric is exactly this.** On every leg, row and seed,
`Δ novel tuples per corner = Δ validity × the corner ceiling`, exactly (36 for row 3,
72 for row 4) — SEEN is 0 and every valid tuple in a corner is distinct. Explaining the
per-request validity explains the whole scored metric on both rows.

## 4. The alternatives, eliminated

- **Standardisation.** The coupling's standardise / un-standardise round trip with no
  steps taken moves the coefficients by **1.18e−8** relative — six orders of magnitude
  below the smallest displacement that costs anything.
- **The 81,312 read-side bytes.** The byte ledger is *identical* across `zerostep`,
  `f05_conv`, `onestep` and `pca_settle` — one fitted coupling, four legs — while their
  losses run 0.000 to −41.98 per corner. Only the step count moves. Bytes cannot explain
  a spread that bytes do not vary over.
- **Coupling capacity.** On val, rank 4 / 16 / 64 (22,464 / 82,368 / 321,984 B, a 14×
  range) gives 22.208 / 22.458 / 22.750 novel tuples per corner against the incumbent's
  64.375.
- **Regression to seen combinations (charter E.4).** Tested for the first time on the
  **errors** rather than on the valid tuples, and against a **locality-matched null** (a
  tuple drawn uniformly at the same Hamming distance from the same request): every leg
  lands on fitted combinations **below** chance — 0.00–0.10× the null for the incumbent
  and the gentle legs, 0.28–0.42× for the arm. And a wrong read-back is one factor wrong:
  mean Hamming 1.02–1.05 for the gentle legs, 1.26–1.43 even for the arm. ⛔ The dynamics
  do not walk back onto the fit pool. They walk one lattice cell sideways.

## 5. ⭐⭐ What this means for the programme's central problem

The programme's open problem is that **composition wins on PCA content that cannot do
exact per-item deletion, while the object that forgets exactly loses composition when it
evicts atoms.** This retry names the reason from a third side, and it is not about
subspaces at all — it is about a **motion budget**:

> A read wins composition by giving every combination its own answer. The more
> combinations it must distinguish, the more tightly those answers pack. Its tolerance to
> motion is set by that packing, not by the size of its state — and here it is about a
> **quarter of one lattice step, 1.3–1.6 % of the state's own norm** (the displacement at
> which the loss reaches the pre-registered ±1.0 tuples per corner). **Anything that
> moves the state more than that costs composition, whatever the motion is for.**

Dynamics is one such motion, and it fails. **Eviction is another, and it is priced here
in the same units.** Rebuilding each well's span without one atom and reprojecting every
scored answer:

- where the atom block is **over-complete** (wells 3 and 4: 64 atoms in a 58-dimensional
  channel block), evicting any single atom costs **1e−15** — free, exactly;
- where the atoms are an **efficient basis** (wells 0–2: 64 independent atoms spanning a
  proper subspace — which is the regime in which PCA content wins composition in the
  first place), the median eviction costs 0.30–0.58 and **76 of those 192 atoms (39.6 %)
  individually move the median answer past the read's entire motion budget**; evicting a
  well's mean costs 5.3–14.7, ten to thirty budgets.

⭐ So the incompatibility is one fact wearing three coats. Exact forgetting needs to move
or remove parts of the content; a compositional read cannot afford motion because it
spends its whole state on packing answers close together; and redundancy — the thing that
would make eviction free — is precisely the capacity a subspace read does not spend that
way. **The way out is not a better dynamics or a better deleter. It is a read whose
answers are not packed to its resolution limit**, i.e. one that holds a motion budget in
reserve, which costs bytes that currently go into distinguishing more combinations.

## 6. The three closures

- ⭐ **`zerostep` is bit-identical, exactly.** With the same `basis` object, re-emitting
  every request of **all 109 scored corners** (7,848 requests, **11,649,024** float32
  channel values) gives **0** differing entries against `pca_tensor`; the coefficients are
  equal bit for bit too. ⚠ Reloading the two arms' *separate* pickles introduces 17
  differing requests out of 3,528, at ≤ 7.1e−15 absolute in a state of norm 31.3 — traced
  to 8 + 16 entries of the two full-rank wells' `basis` matrices differing in their last
  bits, two independent LAPACK SVDs of the same atom block in two processes. That is not
  the settle. **What it implies:** at zero steps the entire settled apparatus — the
  78,848 B coupling, the 2,464 B standardisation, the 200-step solver — is *inert*, and
  the read reduces to the incumbent's exact arithmetic. So the whole hub is a clean
  dose-response in one scalar, the displacement; nothing else varies along the ladder.
- ⭐ **`f05_conv` is a real small effect, not noise.** Paired over all 147 corner pairs:
  46 worse / 16 better / 85 tied, exact sign test **p = 1.8e−4**, mean −0.476 per corner.
  Paired per request (10,584): 104 valid→invalid vs 34 invalid→valid, McNemar
  **p = 1.9e−9**. The parent's own row-4 bootstrap CIs — [−0.878, −0.204],
  [−1.000, −0.082], [−0.796, −0.041] — are all three entirely below zero. ⚠ **Inside the
  pre-registered ±1.0 margin is not the same as indistinguishable from zero**: the margin
  is an effect-size bar, and this effect is far smaller than the bar while being nowhere
  near zero. (On row 3 the same leg genuinely *is* a tie; 60 sub-corners with ceiling 36
  have less power than 49 corners with ceiling 72.)
- ⚠ **What the 81,312 B would have to buy.** It is 15.5 % on top of the 524,288 B store
  and 11.4 % on top of store + address book. To clear the pre-registered bar it must buy
  **≥ +1.0** novel tuples per corner; the best leg that actually moves delivers **−0.476**,
  so break-even is **≥ 1.48 per corner ≈ 2.3 %** of the incumbent's 63.6 — before charging
  anything for the bytes. The sharper statement: at zero steps those bytes are **provably
  unread**, so the overhead is dead weight; at every strength that reads them, they buy a
  loss. **There is no setting at which they are worth anything.**

## 7. Limits, and what would overturn this

- The margin probe is **one seed, `val` only** — that is the only condition on which the
  judge may be probed freely. Its predictions are validated out-of-sample against all
  three held-out seeds, which is the strongest check available without spending budget.
- The margin model is **one-dimensional per factor**: it counts the displacement's
  projection onto the single nearest one-factor neighbour. It is accurate below one
  lattice step and saturates above it, where it over-predicts the arm's loss by ~0.1–0.2
  of validity. A full Voronoi model over the whole answer lattice would fix that; it is
  not needed for any conclusion here.
- The motion budget (0.22–0.28 ρ₁) is a **local linear fit** on the two legs inside one
  lattice step. The loss is superlinear further out, so the budget is if anything
  generous.
- ⛔ **What would overturn the mechanism:** a leg that displaces the state by more than
  ~half a lattice step in the tight factors and does *not* lose those factors' read-back;
  or a factor whose damage does not track its own spacing. Neither appears in three seeds,
  five legs and 10,584 held-out requests. **What would not overturn it** is a better
  coupling: capacity was swept 14× and changes nothing.

## 8. Ledger

- **Scored runs: 0.** Composition remains **2/3 spent, run 3/3 BANKED**, untouched.
- **Judge constructions: 1**, for the `val`-only margin probe (18,432 judged states,
  175 s). No held-out corner was judged; every held-out number here comes from the
  parent's `rows.npz`.
- **Code** (all new, all additive, none of the parent's modules edited):
  `scripts/w61_settle_subspace.py`, `w61_answer_lattice.py`, `w61_judge_margin.py`,
  `w61_close.py`, `w61_eviction_cost.py`, `w61_extras.py`, `w61_mechanism_table.py`.
- **Outputs** in `/c/chlu/outputs/w61-compose-physics/`: `START_UTC.txt`, `LOG.md`,
  `MECHANISM-TABLE.md` (every number, rendered from the JSON), and `mech/` holding
  `subspace_seed{0,1,2}.json|npz`, `lattice_seed{0,1,2}.json|npz`,
  `lattice_val_seed0.json|npz`, `margin_val_seed0.json|npz`, `close.json`,
  `eviction_seed0.json`, `extras.json`.
- ⚠ **One correction to the record:** the first write of `START_UTC.txt` landed in the
  *parent's* output directory and clobbered the parent's copy; it was restored from the
  value the parent's own LOG records (06:26Z) with a comment in the file saying so.
  ⚠ **One correction inside this LOG:** the bullet timestamps were first written against
  the parent's clock rather than this session's; they have been corrected to the real
  session clock (this hub ran 08:24Z–08:50Z, not two hours later).
- ⛔ **Safety:** no box-wide kill was issued and none was needed — every command here ran
  in the foreground and exited on its own; no process was launched detached and no PID was
  killed. The stray `C:\c\chlu` tree was not touched.

## 9. Closing verification

- **Code.** `git status` clean; HEAD **`58bc642`** on `w6.1/compose-physics`, one commit
  on top of the parent's tip `25fc108`. Seven new scripts, 1,510 insertions, **no
  existing module edited** — every one of them is read-only over the parent's artifacts.
- **Bundle.** `w61-compose-physics.bundle` (4,898,282 B) written to
  `/c/chlu/outputs/w61-compose-physics/`. `git bundle verify` reports **"The bundle
  records a complete history"**, and `git bundle list-heads` shows
  `refs/heads/w6.1/compose-physics = 58bc642` present, alongside `base/iter6` and the
  parent's `w6/compose-physics`. The branch is bundled WHOLE.
- **Outputs.** `/c/chlu/outputs/w61-compose-physics/` = **9.3 MB**: `START_UTC.txt`,
  `LOG.md`, `MECHANISM-TABLE.md`, `w61-compose-physics.bundle` and `mech/` with all 19
  result files. ⛔ Nothing was written anywhere else, except the one-line restoration of
  the parent's `START_UTC.txt` noted above.
- ⛔ **No kill of any kind was issued.** Every command ran in the foreground and exited
  on its own; nothing was launched detached, so there was no PID of mine to kill. A
  read-only `tasklist` at close shows one `python.exe` (PID 24088, session 0) which is
  **not mine and was left untouched**, as was the stray `C:\c\chlu` tree.
- **Budget.** Allowance 7 h; elapsed **26 min** (08:24Z to 08:50Z). Scored runs spent: **0**. ⛔ At no point
  did this retry want one: every question it asked was answerable from rows the parent
  had already paid for, plus a `val` probe the charter makes free.
