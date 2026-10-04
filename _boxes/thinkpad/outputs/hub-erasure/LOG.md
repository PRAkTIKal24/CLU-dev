# hub/erasure — LOG

**Target:** deleted content is GONE from the CLU store, not merely GATED in a byte-matched baseline,
under the same adversary.
**Branch/worktree:** `hub/erasure` @ `afd37cf` · `C:/chlu/worktrees/CHLU-hub-erasure`
**Scored budget: ZERO.** The `deletion` target keeps all 3 banked. No number here is a scored run.
**Outputs:** `C:/chlu/outputs/hub-erasure` (only write location).

---

## 2026-09-18 (UTC)

- **22:20** — Booted in the prescribed order: charter → `w3-hub-common.md` → `hub-erasure.md`. Nothing
  else read. No campaign ledger, no wave-2 report.
- **22:20** — `START_UTC.txt` written; no live one pre-existed, so this launcher is the only one.
- **22:20** — Environment verified. `chlu.__file__ = C:\chlu\worktrees\CHLU-hub-erasure\chlu\__init__.py`
  — inside the worktree, so numbers from this session are attributable to the pinned tree. jax 0.9.0 ·
  numpy 2.4.1 · equinox 0.13.4 · system python, no venv created, nothing installed.
- **22:20** — Standing constraints I am *not* going to rediscover (§4 of the common file, §5 of the
  target): learned well placement is null · codebook loses 90.5% scale / 100% orientation · density
  descent regresses · read verb is the big effect and the store is a SIGNED basis · >2x bytes is not the
  lever · dreaming is dead · no threshold may take its length scale from atom spacing · energy machinery
  is correct · atoms beyond channel dim buy nothing. And: **scale and orientation are physical defaults
  and are forbidden as deletion targets.**
- **22:20** — Also standing, from §1 of the target: **refusal is dead as a byte-matched claim** — an
  8-byte value-presence bitmap scores 1.0000 on rows 1/2/5/6 against a 524,288-byte budget. I am not
  pursuing a refusal win. Rows 5-7's refusal behaviour is reported as a negative with the 8-byte rule
  printed beside it.
- **22:35** — Instrument survey done. Everything the target needs is on disk and frozen; **no
  encoding, no cache extension, therefore no re-encode-gate hazard and no GPU case.**
  - feature cache `shapes3d_recon/seed_{0,1,2}/features.npy` — `(480000, 128, 4, 4)` float16,
    `scope: full`, sha `d6728f01…`. All 480,000 rows cached, so every store I fit only *reads*.
  - decoder `decoder_v1_s{0,1,2}.pkl`, hash `01ef91f7…` for seed 0 — matches
    `classifier_gate.json`'s `decoder.param_hash_per_seed["0"]`.
  - external classifier `seed_0/external_classifier.pkl`, hash `c8741ac4…` — matches its own
    manifest. `classifier_gate.json` reads **`gate: PASS`**, floor 0.95, per-seed 1.0/1.0/1.0.
  - store grouping `store_plan.json`: K=5, D=2048, ward, well sizes `[303,305,573,501,366]`.
    ⇒ at the 524,288 B budget the store is **N = 64 atoms per well** (`64*2048*4 = 524,288` exactly).
  - value-holdout split: train 284,472 / test 195,528; held out `object_hue = 0.5` (index 5) and
    `shape = 0.0` (cube), plus 3 pair cells and 49 n-way corners.
- **22:35** — **The fact that makes this hub's target measurable at all:** `AtomStore.atoms[k] =
  Z_train[atom_rows][:, cols_k]`, and `assert_atoms_are_train_rows` enforces it bit-for-bit. The
  atom row index is **shared across wells**, so the CLU store's entire content is **64 verbatim
  train feature rows**. An adversary holding the weights can decode them directly. Symmetrically,
  `FactorPrototypes` holds one mean feature vector **per (factor, value)** — including the deleted
  value — at `(57+1)*2048*4 = 475,136 B`. ⇒ the gate-vs-erasure contrast is not a metaphor here: it
  is 4,096 bytes of the prototype table that *are* the deleted content, against a set of 64 rows
  none of which carries it.
- **22:35** — Smoke test (3.1 s total): grid + decoder + classifier + 64 cached rows →
  decode → classify. Per-factor read-back accuracy on REAL cached train features:
  `floor_hue 1.000 · wall_hue 1.000 · object_hue 1.000 · scale 0.563 · shape 0.859 ·
  orientation 1.000`. **Consequence for the adversary's instrument:** `object_hue` is read at
  1.000 and is the reliable recovery axis; `shape` at 0.859 is usable but noisier; `scale` at
  0.563 would have been unusable — which is a third, independent reason (beside §5's physical-
  default argument) not to target it. Recorded now, before any adversary exists.
- **22:48** — ⭐⭐ **ADVERSARY CALIBRATION 1 — the unconstrained span sweep is measuring the
  DECODER, not the store. Caught by the positive control, before any claim existed.**
  Setup: 64 basis vectors; free signed coefficients `c ∈ R^64`; `x = c @ A`; 60 Adam-free GD steps
  at lr 0.05 on judge-0's `log p(object_hue = 5)` through the frozen decoder; 64 candidates;
  then scored by all three external classifiers independently.

  | span | attacked-judge hit | **all-3-judge hit** | mean log p |
  |---|---|---|---|
  | real atoms, from a pool that NEVER held hue 0.5 | 0.781 | **0.594** | −1.54 |
  | **RAND-GAUSS**, 64 random vectors matched only in per-coordinate mean/std | 0.859 | **0.672** | −0.97 |
  | the same atoms with each coordinate independently shuffled across atoms | 0.797 | **0.562** | −1.40 |

  **Reading:** a random subspace of the same dimension recovers the "deleted" value *better*
  than the real store does. The three-judge unanimity guard does **not** rescue it — this is not
  an adversarial example that fails to transfer, it is the frozen decoder genuinely rendering
  hue 0.5, which it can do because it was trained on every hue (the split artifact's own scope
  note says so: *"The decoder can still render the value and the judge can still recognise it"*).
  ⇒ **An adversary free to choose the decoder's input recovers any value from any store, including
  one whose parameters are noise.** Had I run this against `clu` first and reported it, I would
  have reported the decoder's generative range as a failure of erasure.
- **22:48** — **What this rules in and out, and it is a scope ruling I will carry into the PREREG:**
  "recoverable from the store" cannot mean "the frozen decoder can be driven to draw it". It can
  only mean recoverable **from the store's parameters and its deployed read**. So the move-set
  splits into moves that are admissible as stated (read the parameters; drive the deployed read;
  request the value directly or indirectly) and coefficient sweeps, which are admissible **only
  in a constrained form whose own positive control returns ≈ 0**. Calibrating that constraint next
  — sensitivity (must recover from a store that HOLDS the value) against specificity (must recover
  nothing from a dimension-matched random subspace). ⛔ No claim is made from an uncalibrated move.
- **23:05** — **ADVERSARY CALIBRATION 2 — the L1 cap does not rescue the signed sweep.**
  Same sweep, coefficients initialised at one stored atom (one-hot) and projected back onto
  the L1 ball each step. At the tightest cap tried, `L1 <= 1.0` (i.e. the signed convex hull):
  HOLDS-the-value span **0.625** · deleted span 0.562 · **random Gaussian span 0.484**.
  ⇒ separation sensitivity − specificity = 0.141, and the deleted span is indistinguishable
  from the random one. ⛔ **The signed sweep is inadmissible at every cap I tried.**
- **23:05** — **ADVERSARY CALIBRATION 3 — the ZERO-MOVE baseline, and it localises the fault to
  the SIGN.** No optimisation at all; just decode and judge. Unanimous over 3 judges:

  | candidate set (nothing optimised) | reads `hue = 5` | reads `shape = cube` |
  |---|---|---|
  | global mean of the deleted pool | **0.000** | **0.000** |
  | 128 random **convex** combinations of 64 deleted-pool atoms | **0.000** | **0.000** |
  | 128 random **signed**, L1 = 1 combinations of the same atoms | **0.125** | **0.195** |
  | the 64 stored atoms, verbatim | **0.000** | **0.000** |
  | 128 REAL `hue = 5` rows — the instrument's ceiling | **1.000** | 0.281 |
  | 128 REAL cube rows — the instrument's ceiling | 0.109 | **0.930** |

  - ⭐ **Correction to a guess I had formed and had not yet acted on:** I expected the global mean
    to read as the mid-hue and therefore to poison the hue target. It does not — 0.000. The
    confound is not "0.5 is the average hue"; it is **the sign**.
  - ⭐ The two off-diagonal numbers are an internal consistency check the judges pass: real cube
    rows read `hue = 5` at 0.109 and real hue-5 rows read cube at 0.281, which are exactly the
    base rates 1/10 and 1/4. The judges are reading truth, not noise.
  - ⭐ **The instrument ceiling is 1.000 (hue) and 0.930 (cube)** on real carrying rows, so a
    zero on a deleted store cannot be blamed on the judges.
  - ⛔ Free **signed** combinations leave the feature manifold and the frozen decoder renders
    the off-manifold point as something the judges will call anything — 0.125/0.195 with *no
    optimisation whatsoever*, which is the floor a gradient then climbs from. **Convex
    combinations have a clean 0.000 floor.**
- **23:05** — **Next, and it decides the move-set:** testing whether a CONVEX sweep
  (`c = softmax(θ)`, always on the simplex, fully differentiable) is admissible — sensitivity on
  a span that HOLDS the value against specificity on a random span and on the landscape-removed
  (coordinate-shuffled) span, on BOTH targets. Whatever it returns, the signed sweep is reported
  as a failed move with the numbers above, per §7 of the target file.
- **22:36** — **CALIBRATION 2 addendum:** `L1 <= 1.5` before I stopped the sweep — HOLDS 0.625 ·
  DELETED 0.578 · RAND **0.453**. Same picture as `L1 <= 1.0`, so I stopped the remaining (looser)
  caps: they can only be worse, and the box is at 84 % across four hubs. ⛔ **Verdict: the signed
  span sweep is INADMISSIBLE at every cap measured.** It stays in the code as a reported failed
  move and its numbers are never quoted as evidence.
- **22:36** — Stopped a second calibration job for the same reason: `random_span` and
  `clu_shuffled` are **conditions inside the claim-1 run**, so the sensitivity/specificity
  calibration is measured per target and per seed as part of the run itself, and
  `claim1_verdict` refuses to read a cell that fails it. A separate calibration job was
  duplicating that work on a contended box.
- **22:37** — **`PREREG-erasure-1.md` written** — the adversary's move-set, its budget, the
  calibration gate (`sensitivity − specificity >= 0.10`), the nine claim-1 conditions, the byte
  ledger, the churn design and the three conjunctions, all before any claim cell ran. ⛔ **Scored
  runs spent: ZERO.** There is no `SCORED-LEDGER.md` in the shared hub root to append to, and I
  spend nothing, so I write no ledger row — recorded here instead.
- **22:37** — Byte ledger, asserted at run open: `clu` **524,288 B exactly** (`64 × 2048 × 4`) ·
  `knn_proto` **475,136 B** · `basis_head` **522,288 B** at `m = 62`, which the code asserts is
  the **largest** `m` the budget allows (an under-resourced like-for-like is not a like-for-like).
  The 8-byte gate is ADDED to the gated arms, never netted off ours.
- **22:37** — Plumbing smoke at `n_store_train = 4096`, `adv_steps = 3` (⛔ reported nowhere as a
  result) confirmed the pipeline end-to-end and showed the expected shape already: on
  `clu_full` the direct request recovers the "deleted" hue at 0.580 unanimous and the weight read
  at 0.094 — **which is the 1/10 base rate of hue-5 rows among 64 atoms, i.e. the move is
  reading exactly what is there** — while on `clu_deleted` the same moves give 0.012 and **0.000**.
- **22:38** — **Claim 1 launched for real, seed 0, both targets, all 9 conditions.** 4 threads,
  900 s per-cell cap, every cell writes its own JSON and is skipped on resume.
- **22:42** — **Correction to the like-for-like, made before any `basis_head` cell was written.**
  My first `basis_head` was a dictionary + a head over the 57 single-value one-hots. That is
  weaker than the object w3-hub-common §1 names: wave 2 measured that **the pair-interaction
  block, not the store, is what beats `knn_proto`** on it. I stopped the claim-1 run (only the
  pools file had been written), rebuilt the head over **singles + ALL 1,322 pairwise interactions
  + bias = 1,380 request features**, and re-solved for the largest dictionary the budget allows:
  **m = 38**, `used = 521,056 B` of the 524,288 B budget.
  ⭐ The arm is **declared at the full 524,288 B** with `used_bytes` and `unused_bytes` in its
  ledger — its parameterisation cannot spend the last 3,232 B, and **over-stating the control's
  allowance is the conservative direction** for a claim made against it. ⛔ Quietly scoring it at
  521,056 B would have been under-resourcing the like-for-like.
  ⇒ the adversary's M2 weight read now decodes **1,418 vectors** for this arm: 38 dictionary
  atoms + 1,379 head rows, **including every pair-interaction row** — which is exactly where a
  conjunction's content would sit, so the strongest place to look is now looked at.
  Byte ledger as re-asserted: `clu` **524,288 B exactly** · `knn_proto` **475,136 B** ·
  `basis_head` **524,288 B declared / 521,056 B used** · gate **+8 B** on the gated arms.
- **22:42** — Claim 1 relaunched, seed 0, both targets, 9 conditions.
- **22:55** — **Claim 1, first two real cells (seed 0, target `object_hue = 0.5`, 32,768-row live
  set, 524,288 B).** Unanimous over three judges; judge 0 is the only one attacked.

  | move | `clu_full` (HOLDS it) | `clu_deleted` (refit) |
  |---|---|---|
  | M1 direct request | **0.543** | **0.004** |
  | M3 off-grid bracket | 0.484 | **0.000** |
  | M4a convex span sweep | 0.333 | 0.031 |
  | M2 weight read | **0.109** | **0.000** |
  | M5 read-reachable | 0.051 | 0.016 |
  | M3 term omitted | 0.016 | 0.000 |
  | **best move** | **0.543** | **0.031** |

  - ⭐ **M2 = 0.109 on `clu_full` is the 1/10 base rate of hue-5 rows among 64 atoms.** The move
    is reading exactly what is there and nothing else — the same internal check the zero-move
    baseline passed. On the refitted store it is **0.000**: not one of the 64 stored train rows
    carries the deleted value, which is what "refitted with no row carrying it" means, verified
    by decoding all 64 rather than asserted.
  - ⛔ Not yet a verdict: the calibration gate needs `random_span` (specificity), and the
    like-for-like `basis_head_gated` has not run. **No claim until the full cell block is in.**
  - Cost: **464 s per cell**. 18 cells for seed 0.
- **22:55** — **Claim 2 plumbing verified, and the precondition holds: the identical-live-set
  rebuild gives churn EXACTLY 0.0000 for all three arms.** Without that the pipeline would be
  nondeterministic and no churn number would be readable. (Levels from the smoke are at a
  2,048-row pool with 32 requests and are ⛔ reported nowhere.)
- **22:55** — **Claim 3 plumbing verified**, and the contrast is already visible at smoke scale
  (⛔ not a result): on `objhue2_x_oblong`, `clu` pre 0.812 → post **0.000** with placebo 0.562,
  while `basis_head` goes 0.875 → 0.625 and `knn_proto` 0.812 → 0.375. Both tables re-compose
  the deleted conjunction from parts they still hold; the CLU does not.
- **22:55** — A real bug the smoke caught: `run_claim2` wrote its per-request prediction `.npz`
  before anything created `claim2/`. Fixed. ⭐ This is why the smoke exists — it would have
  killed a two-hour queued run at its last step.
- **23:08** — ⛔⛔ **PROCESS FAULT AGAINST MYSELF, and it is wave 2's §6.1 fault exactly: TWO
  writers were executing claim 1 into one artefact set.** Caught by an inconsistency, not by
  luck: `clu_deleted`'s best move read **0.031** at 22:55 and **0.042** at 23:05 — the same cell
  file, rewritten.
  - **Cause, verified:** my two earlier "stopped" bullets are **wrong and are corrected here**.
    `ps aux | grep … | awk '{print $1}' | xargs kill` under MSYS printed the wrong column, and a
    `Stop-Process -Id` I issued named a PID that was not the worker. Enumerating
    `Win32_Process` command lines showed `run_claim1.py` at PID **29964** (the pre-patch launch)
    **and** PID **31628** (the relaunch) both alive, plus `calib5.py` at 27840 still running
    after I logged it stopped.
  - **What it could have corrupted:** the pre-patch process held the OLD module, so any
    `basis_head` cell it wrote would have carried the singles-only head at `m = 62`, mixed into
    a directory with `m = 38` pair-block cells and indistinguishable afterwards.
  - **Action:** all three processes killed **by verified PID**, and ⛔ **every claim-1 cell
    discarded and recomputed under a single writer.** ⛔ No number written before 23:08 is used.
  - **Lesson, recorded as a rule for the rest of this hub:** every kill is confirmed by
    re-enumerating `Win32_Process` command lines, never by the exit status of the kill.
  - ⚠ The two numbers differed by exactly 1 of 96 candidates (0.0104), i.e. XLA thread-order
    non-determinism, not a logic error. That is a second thing worth knowing: **these cells are
    reproducible to about one candidate, not bit-exactly**, so a margin of 0.10 is the right
    granularity to read them at and 0.01 differences are noise.
- **23:10** — **Single-writer run launched, PID 13044, verified by re-enumerating
  `Win32_Process` command lines: exactly one `run_all.py` and nothing else of mine.** Stages, in
  order, all resumable per cell: claim 1 (seed 0, both targets, 9 conditions) → claim 2 (churn,
  seeds 0/1/2) → claim 3 (3 conjunctions, seeds 0/1/2) → the seven-row scoreboard → claim 1
  seeds 1 and 2 if the clock allows. 4 threads. Per-cell cap 900 s.
- **23:10** — Scoreboard plumbing validated end-to-end (smoke, ⛔ reported nowhere): all seven
  rows emit for all five arms — `clu`, `basis_head`, `knn_proto`, the pinned no-model
  `grid_syntax_rule`, and **`value_bitmap_8B`**, which I added as a scored arm so the §1 negative
  is printed *as a row* rather than as prose. Even at smoke scale it does what §1 says: **1.0000
  on row 5 at 8 bytes, and 0.0 on rows 1–4 because it generates nothing.**
  ⚠ One design decision recorded now, before the real scoreboard runs: **every arm is scored
  UNGATED on the scoreboard**, so each arm's deployed ledger is constant across its seven rows
  (the harness asserts that). The gate belongs to claim 1. ⇒ row 7's `fit_fingerprint` will
  differ pre/post for every arm, because this hub's deletion **is a refit**, and
  `score_deletion_row` will record that as a fault. ⭐ That is the honest reading and it is the
  hub's point: **row 7 as specified scores the NO-RETRAINING variant, which is the gate's move;
  the CLU buys erasure with a rebuild.** Reported, not engineered around.
- **23:29** — ⛔⛔ **TWO SCORING FAULTS FOUND AND FIXED, BOTH OF WHICH FLATTERED MY OWN CLAIM.**
  Caught by reading the first `basis_head_gated` cell rather than its summary number. All
  claim-1 cells discarded again and recomputed; ⛔ no number written before 23:30 is used.
  1. ⛔ **Interface moves were scored BEFORE the gate.** `basis_head_gated` reported
     `M1_direct_request = 0.934` as its best recovery, but the cell's own fields show
     `gate_served_frac = 0.000` and `unanimous_hit_after_gate = 0.000` — the gate refuses every
     one of those requests. Crediting the adversary with a request the gate blocks **overstates
     the gated baseline's leakage, which is the one direction this hub must not err in.**
     Fixed: `move_recovery()` reads every interface move **after the gate**, in one place.
  2. ⛔ **The weight read had the wrong denominator.** On `basis_head_gated` it scored
     `0.005` = 7 recovering parameters / 1,418 decoded. But look at *which* 7:
     `head_single (factor 2, value 5)` — the `object_hue = 5` coefficient itself — plus
     `head_pair_f2v5_f3v5`, `…f3v6`, `…f3v7`, `…f4v1` and a dictionary atom. **The adversary
     found precisely the deleted value's own head rows.** An adversary does not need half the
     parameters to leak; it needs one, and it knows which one to read. Fixed: M2 now also
     reports `targeted_hit`, the rate over the parameters **labelled with the deleted value**,
     and that is the rate the headline uses. ⭐ `clu` has **no targeted set at all** — its atoms
     are whole train rows indexed by nothing — so its all-parameter rate is the right and only
     statistic. That asymmetry is not a measurement artefact; **it is the thing this hub is
     about**, and it is now visible in the metric instead of hidden by it.
- **23:29** — **A third fault, and it turned into a finding.** The off-grid move was not
  off-grid for `clu` or `basis_head`: the code set the on-grid index and so re-ran M1. Fixed
  properly, per arm:
  - `knn_proto` and `basis_head` are asked a **genuinely** off-grid term (for `basis_head`, the
    linear blend of the two bracketing one-hots, matching what `FactorPrototypes.prototype`
    already does, so both table arms get the same question);
  - `clu` is **NOT APPLICABLE** and says so rather than scoring 0.0 — the pinned request is a
    factor → *grid value* assignment (`atom_posterior` indexes `log_cond[f][value_index]`), so
    **the CLU's interface has no off-grid channel to exploit.** That is a property of the
    interface, reported, not a gap in the adversary.
  - ⭐⭐ **And this is where the 8-byte gate is structurally blind:** its vocabulary *is* the
    factor grid, so an off-grid term has **no bit to test** and the rule can only accept it.
    Whatever the interpolating arms hand back for a request at hue 0.49, the gate did not stop
    it. `gate_served_frac = 1.0` is recorded on that move for exactly this reason.
- **23:30** — 30 tests pass (6 new, covering both scoring rules and the off-grid encoding).
  One of the new tests initially failed on **my own fixture**, not the code — an identity-like
  `W` discarded every request feature, so the two bracketing values were indistinguishable and
  the "did it actually move?" assertion could not fire. Replaced with a dense random head.
- **23:30** — Run relaunched, single writer PID 29276, verified by command line.
- **00:33** — ⭐⭐ **CLAIM 1, SEED 0, COMPLETE — and the two targets do NOT get the same verdict.**
  Every number unanimous over three independent judges; judge 0 is the only one attacked;
  interface moves are read after the gate; M2 is read at its targeted rate.

  **Target `object_hue = 0.5` — 32,768-row live set, 524,288 B:**

  | condition | bytes | best recovery | by which move |
  |---|---|---|---|
  | `clu_full` (HOLDS it) — sensitivity | 524,288 | **0.543** | M1 direct request |
  | `clu_placebo` (equal rows deleted, none carrying) | 524,288 | **0.562** | M4a convex sweep |
  | `clu_deleted` (refit, never held it) | 524,288 | **0.042** | M4a convex sweep |
  | `clu_shuffled` (landscape-removed, §C.2) | 524,288 | **0.016** | M5 |
  | `knn_proto_gated` | 475,144 | **1.000** | **M2 weight read** |
  | `basis_head_gated` — THE LIKE-FOR-LIKE | 524,296 | **0.629** | **M3 off-grid bypass** |
  | `basis_head_refit` | 524,288 | **0.000** | — |
  | `knn_proto_refit` | 475,136 | **0.000** | — |
  | `random_span` — specificity | 524,288 | **0.000** | — |

  - **Calibration: sensitivity 0.543 − worst-case specificity 0.016 = +0.527.** ⇒ readable.
  - **Verdict: ERASED vs GATED.** Best gated baseline 1.000 − refitted CLU 0.042 = **+0.958**.
  - ⭐ `knn_proto_gated` = **1.000 via the weight read**: `protos[object_hue][5]` is the mean
    cached feature of every deleted row and it is still sitting in the deployed table. **4,096
    of its 475,136 bytes ARE the deleted content.** The gate refuses the request; the parameter
    answers it anyway.
  - ⭐ `basis_head_gated` = **0.629 via the OFF-GRID BYPASS**, with `gate_served_frac = 1.00`.
    Its on-grid direct request is blocked cleanly (`0.934 → 0.000` after the gate) — the gate
    *works*, on-grid. But the 8-byte rule's vocabulary **is** the factor grid, so a request at
    hue 0.49 has no bit to test, the rule serves it, and the interpolating head hands back the
    deleted value. Its weight read is **0.179 targeted** (5 of the 28 parameters labelled
    `object_hue = 0.5` decode to it, including the `head_single` for it and four of its
    pair-interaction rows). ⛔ Two independent leaks in the like-for-like.
  - ⭐ `clu_placebo` 0.562 ≈ `clu_full` 0.543 ≫ `clu_deleted` 0.042: the collapse is caused by
    deleting the **carrying** rows, not by deleting rows.
  - ⚠ **The honest qualification, stated first rather than buried:** `basis_head_refit` and
    `knn_proto_refit` are **0.000** — *lower* than the refitted CLU's 0.042. **Erasure here is a
    property of REFITTING, not of the CLU specifically.** What the CLU has is that deletion-as-
    refit is its native exact operation on atoms (charter §B.4), while the gate is what the
    deployed baselines *are* in the literature. ⛔ The claim this hub can make is
    **gate vs erasure**, ⛔ not CLU vs table.
- **00:33** — ⛔⛔ **Target `shape = cube`: ADVERSARY NOT CALIBRATED. The cell is NOT readable,
  and I am reporting it as a null rather than as the second win it superficially looks like.**
  `clu_full` 0.385 · `clu_deleted` 0.188 · `clu_placebo` 0.396 · **`clu_shuffled` 0.344** ·
  `random_span` 0.000 · `knn_proto_gated` 0.646 · `basis_head_gated` 0.438.
  - The **landscape-removed control does not collapse**: a store whose content has been
    destroyed by coordinate-shuffling recovers the cube at 0.344 against a ceiling of 0.385.
    That is a charter **§C.2 failure on this target**, so `clu_deleted = 0.188` cannot be read
    as erasure — sensitivity − specificity = **+0.042**, under the pre-registered 0.10 margin.
  - **Mechanism, and it is worth more than the cell would have been:** the convex sweep over
    *shuffled real atoms* finds cube-reading points, while the same sweep over a *Gaussian*
    matched to the marginals finds none (0.000). ⇒ a shuffled feature map is incoherent, the
    frozen decoder renders incoherence as something blocky and hard-edged, and the judges call
    that a **cube**. **The cube is the shape a degraded render degrades towards under THIS
    decoder** — the same class of hazard that rules out `scale` and `orientation` as physical
    defaults (target §5), and the mirror image of the value-hold-out builder's own argument for
    rejecting the sphere. ⛔ On this instrument the cube is not an admissible target for any
    synthesis move, and the value-hold-out's stated reason for choosing it (*"the shape most
    distant from the retained set under the only degradation mode we have measured"*) does not
    hold for the degradation mode measured here.
- **00:33** — **Corrected the verdict rule, and this is the correction that mattered most.**
  `claim1_verdict` gated on the random span alone; it now gates on the **worst case over both
  negative controls** (random span AND landscape-removed). Under the old rule the cube cell
  read `ERASED vs GATED`; under the corrected rule it reads `ADVERSARY NOT CALIBRATED`.
  ⭐ A Gaussian matched only to the marginals is a **weaker** control than shuffled real atoms,
  and the weaker control must not be the one that licenses the claim. The hue cell is unchanged
  (binding control `clu_shuffled` at 0.016, separation +0.527).
  ⚠ The running process holds the pre-patch function in memory, so any verdict file it writes
  for seeds 1–2 uses the old rule; I recompute every verdict from the cells at the end.
- **00:51** — ⭐ **CLAIM 2 COMPLETE — rebuild churn, 3 store seeds, per-request predictions
  PERSISTED (which is what wave 2 could not do).** 32,768-row live set, **8,234 rows deleted**
  (wave 2's size, reproduced exactly), **4,096 fixed requests**, the same request set for every
  arm and every store. "Churn" = the fraction of requests whose read-back **factor tuple**
  changed. Per-request predictions from all three judges are on disk as
  `claim2__<arm>__seed<s>__predictions.npz` (`requests`, `before`, `null`, `deleted`), so the
  ratio is auditable row by row rather than taken on trust.

  | arm | null (identical live set) | churn after deletion | mean factors changed (of 6) |
  |---|---|---|---|
  | `clu` | **0.0000 / 0.0000 / 0.0000** | **0.9954 ± 0.0012** | **2.87** |
  | `basis_head` (like-for-like) | **0.0000 / 0.0000 / 0.0000** | **0.7110 ± 0.0335** | 1.05 |
  | `knn_proto` | **0.0000 / 0.0000 / 0.0000** | **0.0922 ± 0.0019** | 0.096 |

  - ⭐ **The null is exactly 0.0000 in all nine (arm, seed) blocks.** The rebuild is
    deterministic, so every non-zero number below it is a real behavioural change and not
    pipeline noise. That is the precondition wave 2 never established.
  - Ratios: **clu / knn_proto = 10.8×** on tuple churn and **29.9×** on mean factors changed;
    clu / basis_head = 1.40× and 2.7×. The wave-2 prior under test was 13.4× (18.8 % vs 1.40 %,
    aggregates only). ⭐ **Same direction, same order of magnitude, now auditable.**
  - ⚠ **Read the ceiling, not just the ratio:** `clu` is saturated at 0.995 on the tuple metric,
    so the tuple ratio *understates* the gap. `mean_factors_changed` (2.87 vs 0.096) is the
    ceiling-free statistic and is the one to quote.
  - ⚠ **What this does and does not say.** It says the store's output is strongly a function of
    *which rows it holds*, and that a byte-matched prototype table's output is very nearly
    independent of them — the same coin as wave 2's "the head does the work and the store is a
    compression basis". ⛔ It is **not** by itself a virtue: 99.5 % churn also means the CLU's
    answers are unstable under rebuild, and a referee is entitled to read it that way. The
    defensible claim is the narrow one: **deletion changes this store's output and barely
    changes the table's.**
- **01:18** — ⭐ **CLAIM 3 COMPLETE — conjunction deletion, 3 conjunctions × 3 store seeds ×
  3 arms = 27 cells, each with a size-matched placebo drawn from inside the declared cell.**
  Cell: **`shape = 3` (oblong/capsule), NON-CYLINDER** — not the cylinder (wave 2's scored cell
  could not test a shape conjunction at all), not the cube (this hub's value target), not the
  sphere. Conjunctions: `object_hue=2 ∧ shape=3`, `floor_hue=7 ∧ shape=3`, `wall_hue=4 ∧ shape=3`.
  Retention = **unanimous over three judges on BOTH factors at once**.

  | arm | **pre-deletion** | post-deletion | placebo | post/pre | **post/placebo** |
  |---|---|---|---|---|---|
  | `clu` | **0.382** | **0.104** | 0.332 | 0.259 | **0.321** |
  | `basis_head` (like-for-like) | **0.833** | **0.437** | 0.824 | 0.520 | **0.557** |
  | `knn_proto` | **0.858** | **0.385** | 0.864 | 0.449 | **0.446** |

  - ⚠ **The pre-deletion level is printed first because the target file requires it and because
    it changes the reading.** The CLU holds these conjunctions **weakly to begin with (0.382)**
    against the tables' 0.83–0.86 — the same weakness wave 2 flagged at 0.3672. ⛔ A large part
    of the CLU's low post-deletion 0.104 is that it never held the conjunction strongly.
  - ⚠ **And the CLU's PLACEBO also drops** (0.382 → 0.332) while the tables' placebos do not
    move at all (0.833 → 0.824, 0.858 → 0.864). The CLU is perturbed by *any* deletion —
    the same fact claim 2 measures as 99.5 % churn — which inflates its apparent specificity.
    ⇒ **`post/placebo` is the honest statistic**, and on it the CLU forgets the conjunction to
    **0.321** against the like-for-like's **0.557** and the prototype table's **0.446**.
    A real effect, ~1.7× the like-for-like, ⛔ but nothing like wave 2's 2.9 % vs 60.4 %, and I
    am not going to quote it as if it were.
  - ⭐⭐ **The sharp result here is the GATE contrast, and it is exact rather than measured:**
    after a conjunction deletion **both values still occur separately in the live set, so the
    8-byte per-factor bitmap changes ZERO bits.** It cannot express a conjunction deletion at
    all, so a gated arm goes on serving the deleted conjunction at its **full pre-deletion
    rate, 0.833–0.858**, against the refitted CLU's **0.104**. The only gate that *can* express
    it is the full joint table — **60,000 B over 480,000 combinations** — which then refuses
    every combination the live set does not contain, i.e. **every row-3 and row-4 request**.
    ⇒ the gate can have the deletion or compositional generation, not both. **That is the
    clearest thing this hub produced on claim 3**, and unlike the arm comparison it does not
    depend on a weakly-held pre-deletion level.
- **01:30** — **Two corrections to the scoreboard, both of which had been flattering the store,
  and both fixed before the reported table is generated.**
  1. ⛔ **The 8-byte rule was only clearing ONE bit.** Row 5 carries a `held_out_hue` half and a
     `held_out_shape` half; I cleared `object_hue = 5` only, so the rule refused half the row
     and scored **0.7565** instead of its true score. Fixed to clear **every** held-out value,
     and it now reproduces the target file's §1 negative exactly: **`value_bitmap_8B` = 1.0000
     on row 5, at 8 bytes, against every store arm's 0.0000.** ⛔ Under-reporting the rule this
     hub is measured against is the one kind of error that would make the §1 negative look
     weaker than it is.
  2. **Rows 1–2 were unreadable as printed.** My `valid` is *unanimous over three judges on
     every requested factor at once* — far stricter than the programme's usual unweighted mean
     over factors — so every arm sits near zero and the row says nothing. The strict rate still
     scores the row (it is applied identically to every arm, so the comparison is fair), but
     the **per-factor accuracy is now recorded beside it** in the artifact's provenance, so the
     row is readable and ⛔ nobody compares this hub's row 1 with another hub's without seeing
     that the criterion differs.
  - ⚠ Declared, not fixed: the bitmap scores **0.433** on row 6 rather than the §1 table's
    1.0000 because I did not give it the free duplicate-factor check that table assumes. Row 6
    is `evidential: false` by the Head's ruling and every store arm scores 0.0000 on it, so the
    rule already wins the row; adding the check would only widen a gap that is not evidence for
    anything. Recorded so the difference from the §1 table is not mistaken for a discrepancy.
  - ⭐ Worth recording: the pinned no-model laundering rule `grid_syntax_rule` scores **0.0000
    on both refusal rows** under AMENDMENT 1's pairing. That is the amendment working — with
    row 2 as the accept-leg it scored a perfect pair for free; with row 1 (on-grid, stored) it
    cannot separate the legs at all. The surviving free rule is the 8-byte bitmap, and it is
    the one printed beside rows 5–7.
- **01:30** — Claim 1 seeds 1 and 2 are running (36 cells, ~1.9 h). The reported scoreboard is
  regenerated after them, with both fixes in.
- **02:57** — ⭐⭐ **CLAIM 1, ALL 54 CELLS (9 conditions × 2 targets × 3 store seeds), FINAL.**
  Verdicts recomputed from the cells under the corrected two-control calibration rule.

  **`object_hue = 0.5` — CALIBRATED ON 3/3 SEEDS, ERASED vs GATED on all three.**

  | condition | s0 | s1 | s2 | mean ± sd |
  |---|---|---|---|---|
  | `clu_full` (HOLDS it) — sensitivity | 0.543 | 0.427 | 0.646 | **0.539 ± 0.089** |
  | `clu_placebo` (equal rows, none carrying) | 0.562 | 0.543 | 0.715 | **0.607 ± 0.077** |
  | `clu_deleted` (refit) | 0.042 | 0.198 | 0.023 | **0.088 ± 0.078** |
  | `clu_shuffled` (landscape-removed) | 0.016 | 0.020 | 0.023 | **0.020 ± 0.003** |
  | `knn_proto_gated` | 1.000 | 1.000 | 1.000 | **1.000 ± 0.000** |
  | `basis_head_gated` — THE LIKE-FOR-LIKE | 0.629 | 0.672 | 0.625 | **0.642 ± 0.021** |
  | `basis_head_refit` | 0.000 | 0.000 | 0.000 | **0.000 ± 0.000** |
  | `knn_proto_refit` | 0.000 | 0.000 | 0.000 | **0.000 ± 0.000** |
  | `random_span` — specificity | 0.000 | 0.000 | 0.000 | **0.000 ± 0.000** |

  Calibration per seed: +0.527 / +0.408 / +0.622 (binding control `clu_shuffled` every time —
  the random span is the weaker control and never binds).
  **Gap (best gated − refitted CLU): +0.958 / +0.802 / +0.977.**
  ⚠ `clu_deleted` is **not uniformly zero**: seed 1 reached 0.198 via the convex sweep, against
  0.023–0.042 on the others. Reported as the spread it is, not as "0".
  ⚠ `clu_deleted` (0.088) sits **above** `clu_shuffled` (0.020): a small residual signal
  survives the refit. Declared.

  **`shape = cube` — CALIBRATED ON ONLY 1/3 SEEDS ⇒ ⛔ NOT READABLE, reported as a null.**
  Per seed, sensitivity − worst specificity: **+0.042 / −0.292 / +0.156**. On seed 1 the
  landscape-removed store recovers the cube at **0.635 against a ceiling of 0.344** — the
  content-destroyed store beats the content-holding one. ⛔ One seed clearing the margin out of
  three is not a result and I am not quoting the cube's `ERASED vs GATED` verdict from seed 2.
- **02:57** — ⭐ **The mechanism behind the cube null, which is worth more than the cell would
  have been.** The convex sweep over *coordinate-shuffled real atoms* finds cube-reading points
  (0.250–0.635) while the same sweep over a *Gaussian matched to the same marginals* finds none
  (0.000 on every seed). A shuffled feature map is incoherent; the frozen decoder renders
  incoherence as something blocky and hard-edged; the judges call that a **cube**.
  ⇒ **on this instrument the cube is the shape a degraded render degrades towards** — the same
  class of hazard that rules out `scale` and `orientation` as physical defaults (target §5),
  and the mirror image of the value-hold-out builder's own argument for rejecting the *sphere*.
  ⛔ Its stated reason for choosing the cube — *"the shape most distant from the retained set
  under the only degradation mode we have measured"* — does not survive the degradation mode
  measured here. **A later hub should treat `shape = cube` as ineligible for any synthesis-based
  recovery measurement on this decoder, and the eligible-target registry is now: hue yes,
  cube no, scale no, orientation no.**
- **03:22** — ⭐ **CHARTER §A.5 REPLICATION — PASSED, and it reproduces the decisive cells
  EXACTLY.** A fresh `git clone --branch hub/erasure` of the hub's own branch at `0a9a442` into
  a separate directory, a separate process, its own output tree, `PYTHONPATH` pointing at the
  clone (asserted in the spoke: it refuses to run if it imports the worktree). Only the frozen
  instrument is shared, and that is read-only for both.

  | condition | original (seed 0) | replication | Δ |
  |---|---|---|---|
  | `clu_full` | 0.543 | **0.543** | 0.000 |
  | `clu_deleted` | 0.042 | **0.042** | 0.000 |
  | `clu_shuffled` | 0.016 | **0.016** | 0.000 |
  | `basis_head_gated` | 0.629 | **0.629** | 0.000 |
  | `knn_proto_gated` | 1.000 | **1.000** | 0.000 |
  | `random_span` | 0.000 | **0.000** | 0.000 |

  Calibration separation **+0.527**, verdict **ERASED vs GATED**, gap **+0.958** — all identical.
  840 s. ⚠ Note against the earlier non-determinism bullet: run-to-run drift of one candidate
  (0.010) is possible and was observed once; here the replication landed on the same values in
  all six cells.
- **03:22** — **Final scoreboard regenerated with both fixes.** Headline readings:
  `value_bitmap_8B` = **1.0000 on row 5 at 8 bytes** against every store arm's 0.0000;
  `grid_syntax_rule` = 0.0000 on both refusal rows (AMENDMENT 1 working as designed);
  row 7 records the fault **"fit fingerprint moved (32768 → 29182) — the arm RETRAINED"** for
  every arm, with `delta_reject = 0.0`, exactly as declared in advance at 23:10.

---

# FINAL REPORT — `hub/erasure`

**Target:** show that deleted content is **UNRECOVERABLE** from the CLU store and merely
**GATED** in a byte-matched table baseline, under the same adversary.
**Branch:** `hub/erasure` @ `0a9a442`, off the pin `4125fce` via `afd37cf`.
**⛔⛔ SCORED RUNS SPENT: 0 / 3. The `deletion` target keeps all three banked.** Nothing here
touched the held-out sets A/B/C; every number is a mechanism measurement on the pinned
`value_holdout` condition with its own controls.

## 1. The one-line answer

⭐ **On `object_hue = 0.5`, calibrated on 3/3 store seeds: the deleted value is recoverable from
the byte-matched gated baselines at 1.000 (`knn_proto`) and 0.642 ± 0.021 (`basis_head`, the
like-for-like), and from the refitted CLU store at 0.088 ± 0.078 — a gap of +0.912.**

⛔ **On `shape = cube` the adversary failed its own calibration on 2 of 3 seeds, and that target
is reported as a NULL, not as a second win.**

⚠ **The qualification that belongs in the first paragraph rather than a footnote: the refitted
`basis_head` and `knn_proto` score 0.000 ± 0.000 — *better* erasure than the CLU's 0.088.
Erasure here is a property of REFITTING, not of the CLU. The defensible claim is `gate` vs
`erasure`, ⛔ not CLU vs table.**

## 2. Pre-registered runs — all of them, including the ones that lost

`PREREG-erasure-1.md` was written before any claim cell ran. No bar was changed after a result
was seen; the corrections logged during the run are to *instruments*, and every one was made
before the numbers it affects existed.

| # | pre-registered run | outcome |
|---|---|---|
| 1 | Claim 1: 9 conditions × 2 targets × 3 seeds = **54 cells** | hue **ERASED vs GATED** 3/3; cube **NOT CALIBRATED** 2/3 ⇒ **null** |
| 2 | Claim 2: rebuild churn, 3 arms × 3 seeds, predictions persisted | **clean** — null exactly 0.0000 in 9/9 blocks |
| 3 | Claim 3: 3 conjunctions × 3 seeds × 3 arms = **27 cells** + placebos | effect real but **modest**; the exact gate result is the strong part |
| 4 | Seven-row scoreboard, 5 arms | CLU is **decoration** on rows 1–3 and 5–7 |
| 5 | M4b signed span sweep (unconstrained, and L1-capped) | ⛔ **FAILED ITS OWN CONTROL — reported, never quoted** |
| 6 | §A.5 replication of the winning configuration, clean checkout | **PASSED, Δ = 0.000 on all six decisive cells** |

## 3. The best replicable configuration

`ErasureConfig()` shipped defaults, pinned completely (w3 §3): bridge seed 0 · 32,768-row live
set · **64 atoms** (`64 × 2048 × 4 = 524,288 B` exactly) · `kmeanspp` selection · frozen ward
grouping K = 5 from `store_plan.json` · `nn_median` bandwidths · address smoothing 1.0 · launch
`clamp_atom`, read `state`, **zero settling steps** · adversary 96 candidates × 100 steps, lr
0.2, **convex** (`softmax`) coefficients, judge 0 attacked and judges 1–2 held out ·
`rng_base = 20260919`, `adv_seed = 4242`.
**Replicated from a clean clone at Δ = 0.000 on all six decisive cells.**

## 4. The three claims

### Claim 1 — gate vs erasure (`object_hue = 0.5`, 3 seeds, all at ≤ budget + 8 B)

| condition | mean ± sd | what it establishes |
|---|---|---|
| `clu_full` | 0.539 ± 0.089 | **sensitivity** — the adversary works when the value is there |
| `clu_placebo` | 0.607 ± 0.077 | deleting *rows* does nothing; deleting *carrying* rows does |
| **`clu_deleted`** | **0.088 ± 0.078** | the refitted store |
| `clu_shuffled` | 0.020 ± 0.003 | **specificity**, §C.2 landscape-removed — collapses ✓ |
| **`knn_proto_gated`** | **1.000 ± 0.000** | `protos[object_hue][5]` **is** the deleted content |
| **`basis_head_gated`** | **0.642 ± 0.021** | the like-for-like, leaking two independent ways |
| `basis_head_refit` | 0.000 ± 0.000 | a refitted table erases completely |
| `knn_proto_refit` | 0.000 ± 0.000 | so does the other one |
| `random_span` | 0.000 ± 0.000 | specificity, dimension-matched noise |

Calibration per seed (sensitivity − worst-case specificity): **+0.527 / +0.408 / +0.622**, with
`clu_shuffled` binding every time — the random span is the weaker control and never binds.
**Gap (best gated − refitted CLU) per seed: +0.958 / +0.802 / +0.977.**

**Two distinct leaks in the gated like-for-like, and neither is a straw man:**

1. **Weight read (M2).** 5 of the 28 parameters *labelled* `object_hue = 0.5` decode straight to
   it — the `head_single` for the value and four of its pair-interaction rows. For `knn_proto`
   it is one prototype row and the recovery is **1.000**: **4,096 of its 475,136 deployed bytes
   ARE the deleted content.** ⛔ No gate can refuse this move, because it never asks.
2. **Off-grid bypass (M3).** The 8-byte rule's vocabulary *is* the factor grid, so a request at
   hue 0.49 has **no bit to test** and the rule serves it (`gate_served_frac = 1.00`); the
   interpolating head then returns the deleted value at **0.642**. The same arm's on-grid
   request is blocked cleanly (0.934 → **0.000** after the gate). **The gate works exactly where
   it is defined and not one step outside it.**
   ⭐ `clu` has **no off-grid channel at all** — its pinned request is a factor → grid-value
   assignment — so this move is `NOT APPLICABLE` to it, reported as such, never as 0.0.

⚠ Two honesty notes inside the win: `clu_deleted` is **not uniformly zero** (0.023 / 0.042 /
0.198 — seed 1 is an outlier), and at 0.088 it sits **above** its own landscape-removed control
at 0.020, so a small residual signal survives the refit.

**`shape = cube`: NULL.** Sensitivity − worst specificity = **+0.042 / −0.292 / +0.156** across
seeds; on seed 1 the *content-destroyed* store beat the content-holding one (0.635 vs 0.344).
**Mechanism:** the convex sweep over coordinate-shuffled real atoms finds cube-reading points,
while the same sweep over a marginal-matched Gaussian finds none (0.000, 3/3). A shuffled
feature map is incoherent, the frozen decoder renders incoherence as something blocky and
hard-edged, and the judges call that a cube. ⇒ **the cube is the shape a degraded render
degrades towards on this decoder** — the same hazard class as `scale` and `orientation`.
**Eligible-target registry after this hub: `object_hue` yes · `shape = cube` NO · `scale` no ·
`orientation` no.**

### Claim 2 — rebuild churn, per-request predictions persisted (3 seeds)

| arm | null (identical live set) | churn after deleting 8,234 / 32,768 | mean factors changed |
|---|---|---|---|
| `clu` | **0.0000** ×3 | **0.9954 ± 0.0012** | **2.87 / 6** |
| `basis_head` | **0.0000** ×3 | 0.7110 ± 0.0335 | 1.05 |
| `knn_proto` | **0.0000** ×3 | **0.0922 ± 0.0019** | 0.096 |

Ratios **10.8×** (tuple) and **29.9×** (factors) against `knn_proto`; wave 2's prior was 13.4×.
⭐ **The null is exactly 0.0000 in all nine blocks** — the precondition wave 2 never established.
Per-request predictions for all three judges are on disk (`claim2/*__predictions.npz`), so the
ratio is auditable row by row rather than taken on trust.
⚠ `clu` is **saturated** at 0.995, so quote the factor count, not the tuple rate.
⚠ This says the store's output depends strongly on which rows it holds and the table's barely
does; ⛔ it is **not** by itself a virtue — 99.5 % churn is also instability, and a referee is
entitled to read it that way.

### Claim 3 — conjunction deletion, cell `shape = 3` (oblong, NON-cylinder), 3 conjunctions × 3 seeds

| arm | pre-deletion | post-deletion | placebo | **post / placebo** |
|---|---|---|---|---|
| `clu` | **0.382** | **0.104** | 0.332 | **0.321** |
| `basis_head` | 0.833 | 0.437 | 0.824 | 0.557 |
| `knn_proto` | 0.858 | 0.385 | 0.864 | 0.446 |

⚠ The CLU **holds these conjunctions weakly to begin with** (0.382 against 0.83–0.86) and **its
placebo also drops** while the tables' do not — so `post / placebo` is the honest statistic, and
on it the CLU forgets ~1.7× more than the like-for-like. ⛔ Nothing like wave 2's 2.9 % vs 60.4 %,
and it is not quoted as if it were.

⭐⭐ **The exact result is the strong one.** After a conjunction deletion both values still occur
separately, so the 8-byte per-factor bitmap changes **ZERO bits** — it cannot express the
deletion at all and keeps serving the conjunction at its full pre-deletion rate (0.833–0.858)
against the refitted CLU's 0.104. The only gate that *can* express it is the full joint table
(**60,000 B over 480,000 combinations**), which then refuses every combination absent from the
live set — **every row-3 and row-4 request**. **A gate can have the deletion or compositional
generation, not both.**

## 5. The seven-row scoreboard (w3 §2) — reported; rows outside the target are not claimed

| row | `clu` | `basis_head` | `knn_proto` | `grid_syntax_rule` | `value_bitmap_8B` (8 B) |
|---|---|---|---|---|---|
| 1 seen | 0.0039 | 0.0059 | **0.0996** | 0.0000 | 0.0000 |
| 2 unseen-valid | 0.0000 | 0.0020 | **0.1133** | 0.0000 | 0.0000 |
| 3 comp 2-way | 0 | 0 | **12** | 0 | — |
| 4 comp n-way | 3 | 0 | **118** | 0 | — |
| 5 comp refusal | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **1.0000** |
| 6 invalid refusal | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5467 |
| 7 deletion | 0.0000 | 0.0000 | 0.0000 | — | — |

⛔⛔ **On rows 1, 2, 3, 5, 6 and 7 the CLU ties or loses the fitted basis + linear head. Per
w3-hub-common §1 that is DECORATION, and it is reported in those words.** It beats `basis_head`
on row 4 only (3 novel tuples against 0) and loses that row to `knn_proto` by 118 to 3.

⚠ `valid` here is **unanimous over three judges on every requested factor at once** — stricter
than the programme's usual per-factor mean, and applied identically to every arm. The per-factor
means are in the artifact: rows 1/2 `clu` 0.517 / 0.564, `basis_head` 0.671 / 0.659, `knn_proto`
0.823 / 0.824. ⛔ Do not compare these row-1 numbers with another hub's without that.

**Rows 5–7 with the §1 negative printed beside them, as required:** an **8-byte** bitmap scores
**1.0000 on row 5** against every store arm's 0.0000, and row 7's `delta_reject` is **0.0** for
every arm, with the recorded fault *"fit fingerprint moved (32768 → 29182) — the arm
RETRAINED"*. ⛔ No row-5, row-6 or row-7 number here is evidence for the store. (The bitmap's
row 6 is 0.5467 rather than the §1 table's 1.0000 only because I did not give it the free
duplicate-factor check; declared, and it already beats every store arm's 0.0000.)
⭐ The pinned no-model `grid_syntax_rule` scores **0.0000 on both refusal rows** — AMENDMENT 1
working exactly as intended: with row 1 as the accept-leg, a grid-syntax rule can no longer
separate the legs for free.

## 6. §C floor checks — one line each

- **§C.1 — every arm beats the matched-bytes lookup.** ⛔ **FAILED on the generation table:**
  `clu` loses `knn_proto` on rows 1–4 at matched bytes, so **the store is decorative there and
  is reported as such.** On this hub's own target the comparison is a *recovery* measurement,
  not a lookup one, and the lookup-style moves inside it (`M5_read_reachable`, `M2_weight_read`)
  are reported per cell.
- **§C.2 — landscape-removed control must collapse.** ✓ **hue: 0.020 ± 0.003** against a 0.539
  ceiling. ⛔ **cube: 0.250–0.635, does NOT collapse** — which is precisely why the cube target
  is reported as a null.
- **§C.3 — "a tie is the win narrative".** Dead by the 2026-09-19 amendment; not invoked here.
- **§C.4 — a settled state sitting on its nearest atom is a lookup table.** ⛔ **Conceded and
  declared in the PREREG, not discovered by a referee:** `x0` takes **zero settling steps** and
  returns the posterior-argmax **stored atom** per well by construction. The arm as deployed in
  this hub **is** a lookup table over per-well atom recombinations, and is reported as one.

## 7. What would have proved this hub wrong, and what actually happened

- *"The adversary recovers from the CLU as readily as from the gated table."* — Not on hue
  (0.088 against 1.000). **On cube it effectively did, and that cell is reported as a null.**
- *"The gated `basis_head` is indistinguishable from the CLU."* — It is distinguishable, but
  **the refitted `basis_head` beats the CLU at erasure (0.000 against 0.088)**, so the claim
  narrowed to gate-vs-erasure accordingly.
- *"Churn does not survive persisting per-request predictions."* — It survived; the null is
  exactly 0.0000 and the predictions are on disk.
- *"Deleting a conjunction breaks rows 3–4 for the survivors."* — Not tested to destruction; the
  conjunction effect is real but modest and the CLU's pre-deletion hold is weak.
- *"Your adversary is weak and you cannot tell."* — ⭐ **This is the one that fired, three times,
  and the controls caught it every time:** the unconstrained signed sweep (random span 0.672 >
  real atoms 0.594), the L1-capped sweep (specificity 0.484 against sensitivity 0.625), and the
  cube target (landscape-removed 0.635 > ceiling 0.344). **All three are reported as failed
  moves and none is quoted as evidence.**

## 8. Cost line

**Wall clock:** 2026-09-18 22:20 → 2026-09-19 03:30 UTC, **5 h 10 m**, one hub, one writer.
**Compute:** local CPU only, **4 threads**, on the 64 GB box shared with three other hubs.
**Measured in-cell compute:** 72 timed cells summing **3.28 h** (claim 1: 54 cells; claim 2: 9;
claim 3: 9), plus two scoreboard builds (~6 min each) and the 840 s replication spoke.
**GPU: NONE, and no GPU case ever existed.** The feature cache is `scope: full` at 480,000 rows
with `shortfall = 0`, asserted at every run open, so every run here only *reads* it — the
re-encode gate cannot bind and w3 §8's two hazards are structurally out of scope. ⛔ The three
justification lines were never written because there was nothing to justify.
**Peak memory:** comfortably inside budget; the largest resident objects are the 32,768 × 2048
float32 pool (268 MB) and the basis head's design matrix (181 MB).
**⛔ Scored-run budget consumed: 0 / 3.**

## 9. What the next hub should inherit

1. ⛔ **"Recoverable from the store" cannot mean "the decoder can be driven to draw it."** A
   random 64-dim span reaches a held-out hue at **0.672** unanimous over three independent
   judges. Any recovery move needs a sensitivity **and** a specificity control measured in the
   same run, or it is measuring the decoder.
2. ⛔ **Free *signed* coefficient sweeps are inadmissible on this instrument; convex ones are
   fine.** Zero-move floors, no optimisation at all: convex 0.000 / 0.000, signed-L1=1
   0.125 / 0.195 (hue / cube).
3. ⛔ **A Gaussian matched to the marginals is a WEAKER control than shuffled real atoms.** Gate
   on the worst of the two, or the cube's null reads as a win.
4. ⛔ **Eligible deletion targets on this decoder: `object_hue` yes; `shape = cube` NO** (it is
   the degradation attractor for incoherent features); `scale` and `orientation` no.
5. ⭐ **The gate is blind off-grid.** Any bitmap or lookup keyed on the factor grid serves every
   off-grid request, and an interpolating read then leaks the gated value straight back.
6. ⭐ **A per-factor gate cannot express a conjunction deletion at all (zero bits change), and
   the joint table that can costs rows 3–4 entirely.** That contrast is exact rather than
   measured, and it is the most robust thing this hub produced.

---

## Close-out (appended after the report, per §E.2 — corrections are new bullets)

- **03:25 (UTC)** — **Return artefacts, both of them (w3 §6.5: a return needs a report AND a
  bundle, and `git status` does not show the outputs).**
  - `hub-erasure.bundle` — 41,036 B, `afd37cf..hub/erasure`, verified; carries `0a9a442`
    (`gen_erasure.py` 2,172 lines + `test_gen_erasure.py` 387 lines, 31 tests).
  - `MANIFEST.md` — the **directory listing** of all **96 output files / 2.0 MB**, written
    because a worktree holding an entire unsaved outputs tree reports clean.
  - The worktree itself is clean; both commits are on `hub/erasure` and nothing is uncommitted.
- **03:25** — ⭐ **Sub-agents: NONE were spawned at any point.** The §A.5 replication was run as
  a separate *process* from a clean `git clone` rather than as an agent, which satisfies the
  substance (clean checkout, independent process, its own output tree, asserted to import the
  clone and not the worktree) without leaving anything running. Re-enumerated
  `Win32_Process` command lines at close: **no `CHLU-hub-erasure` process is alive.**
- **03:25** — **`START_UTC.txt` is left in place and marked done below.** It is the one-launcher
  lock of §6.1; a later launcher should see that this hub ran and finished, not an empty slot.
- **03:25** — **Scored-run ledger: nothing to append.** There is no `SCORED-LEDGER.md` in the
  shared hub root, and **this hub spent 0 of its 3** scored runs, so there is no row to write.
  The `deletion` target's budget is untouched and banked at **3/3 remaining**.
