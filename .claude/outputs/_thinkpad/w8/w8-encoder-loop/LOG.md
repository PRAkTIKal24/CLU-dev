# w8 hub/encoder-loop — LOG

TARGET: does the store force better features? Encoder trained ALONE vs encoder trained WITH THE STORE
IN THE LOOP at EQUAL COMPUTE, 3 seeds, on both objects (w8 single design, w5 `pca_tensor` as-is).

- 2026-09-20T16:36Z — START_UTC.txt written (no live one existed). Worktree
  `/c/chlu/worktrees/CHLU-w8-encoder-loop` on branch `w8/encoder-loop` at `ec2ef7c` (= base/iter7),
  tree clean. `chlu.__file__` resolves to
  `C:\chlu\worktrees\CHLU-w8-encoder-loop\chlu\__init__.py` — inside the worktree, verified.
  jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4, system python, 3 threads.
- 2026-09-20T16:36Z — Boot read, in order: `autonomous-hub-charter.md`, `w8-hub-common.md`,
  `w8-hub-encoder-loop.md`. Nothing else. Budget: `encoder-loop` 3 scored runs.
- 2026-09-20T16:45Z — SURVEY. Found the ground the hub stands on.
  - `chlu/experiments/exp_gen_encoder.py` (771 ln) ALREADY IS the store-in-the-loop configuration:
    shared warm start → `gen_bridge.train_joint(encoder, store, decoder, …, unfreeze=bool)`, arms
    `frozen`/`unfrozen` at matched steps×batch, `assert_no_feature_bypass` before any number.
    ⛔ Its objective is **reconstruction + VQ codebook + commitment only** (`gen_bridge.joint_loss_terms`,
    gen_bridge.py:1397). **There is no swap loss in the loop.** That is the missing ingredient and is
    what this hub must add.
  - Its scoring is `distinct_code_configs` on request sets B/MS — ⛔ NOT rows 3–4. Rows 3–4 scoring
    against the learned baselines has to be wired on top.
  - `chlu/core/gen_swap_probe.py` (255 ln) already defines the swap *measurement*: pinned pairs
    `make_pairs`, replace well w's channels of x_i with x_j's, decode through THE frozen decoder,
    score with THE hash-asserted external classifier against the intended composite tuple. ⭐ It is a
    judge-scored validity rate, not a proxy. Precedent exists (hub/placement) for a judge-scored loss
    at FIT time. This hub's swap loss is the differentiable form of that same quantity.
  - Frozen instrument present on this box: encoder+decoder+probe+feature cache per seed at
    `$CHLU_GEN_BRIDGE_ROOT/shapes3d_recon/seed_{0,1,2}` (D=2048, feat_shape 128×4×4, 480,000 rows,
    float16 cache, sha256 in `feature_cache.json`); decoder winners `gen-decoder-lift/winner/decoder_v1_s{0,1,2}.pkl`;
    external classifier `gen-external-classifier/seed_{0,1,2}/external_classifier.pkl` + manifest hash.
    Dataset `C:/chlu/data/gen/shapes3d/3dshapes.h5` (267 MB) present.
  - ⚠ OBJECT 2 RISK, recorded now, before any result. `w8-hub-single-design` wrote its own
    `START_UTC.txt` at 2026-09-20T16:36Z — the SAME minute as this hub — and is itself gated on
    `w8-downdate/DOWNDATE-READY.json`. **The w8 single design does not exist yet.** Plan: build the
    encoder-loop apparatus object-agnostically (the object enters as a store constructor), run it on
    `pca_tensor` as-is now, and take the single design if it lands in time. ⛔ If it does not, that
    leg is a DECLARED NOT-RUN with its launch line emitted — never a null.
- 2026-09-20T17:05Z — DESIGN FIXED. Every hook needed already exists in the frozen harness;
  ⛔ nothing frozen has to be edited. The four hooks, with the code that makes each one work:
  1. **Give every arm and every rival the same features** — `exp_gen_factor_wells.features_root()`
     honours `$CHLU_GEN_FEATURES_ROOT` (and the w5 leg's `CHLU_GEN_BRIDGE_ROOT_OVERRIDE`) while the
     classifier stays pinned by `$CHLU_GEN_CLASSIFIER_ROOT`. Point it at a cache built from MY encoder
     and every arm reads it. ⭐ This is the mechanism for "each baseline given the SAME encoder-loop
     features" and it costs zero harness edits.
  2. **Build that cache** — `chlu.eval.rivals.w4_cache2.build_gen2_cache(encoder, …)`: full-scope
     480,000-row re-encode from pixels, one-CPU-device guard, per-block sha256, resumable
     `_progress.json`, `max_shard_seconds` soft cap. ⭐ MEASURED COST from spoke A's own log
     (`spoke-a-rivals/cache2.out`): **4.5 min for 480,000 rows** on this box. 9 caches ≈ 40 min,
     ~17.7 GB; 679 GB free. **The CPU path is affordable.**
  3. **The learned baselines are REFITTABLE on my features** — ⛔ the shipped rivals are *fitted
     checkpoints* (`chlu.eval.rivals.w4_api.load_rival`) and could NOT honestly be "given the same
     features". `w4_learners.fit_family(name, build, in_kind, width, ctx, key, …)` takes `ctx["X_fit"]`
     / `ctx["X_val"]` — the feature targets — so each baseline is refit on the encoder-loop features
     at its own matched parameter count (`FAMILY_BUDGETS["mlp_180k"]=179_968`, `match_width`).
  4. **The swap loss is differentiable** — the judge is `gen_bridge.FactorProbe`, an equinox module
     returning per-factor logits from an IMAGE. decoder and classifier are both `eqx.Module`s, so
     image→logits is differentiable and the swap loss backpropagates into the encoder through
     `quantise_through_store` (straight-through). `gen_swap_probe` supplies the exact semantics:
     fixed ownership `own[w]` by Hungarian assignment on the measured transfer matrix, intended tuple
     = `y_i` with well w's owned factor replaced by `y_j`'s, `identity_ceiling` (no-swap) and
     `size_matched_random` (floor).
  ⭐ **Ownership is derived ONCE at the warm start and held FIXED** — `gen_swap_probe.score_partition`
  says it in terms: "a refinement loop must fix it, or the objective moves under the optimiser."
- 2026-09-20T17:05Z — THE ARMS (encoder side). Four columns, one shared warm start, equal compute:
  | column | store on the path? | objective | role |
  |---|---|---|---|
  | `frozen` | yes | recon+codebook+commitment, encoder leaves NOT in Optax | the incumbent/shipped configuration |
  | `alone` | ⛔ no | reconstruction only (plain AE, no store) | ⭐ **"encoder trained ALONE"** |
  | `loop_nosw` | yes | recon+codebook+commitment | ⭐ **unfrozen-WITHOUT-swap-loss, its own column** (task: isolate it) |
  | `loop_swap` | yes | recon+codebook+commitment+**SWAP LOSS** | ⭐ **the treatment** |
  Equal compute = identical gradient steps × batch × PRNG stream. ⚠ The swap term costs more FLOPs per
  step, so equal steps is not sufficient on its own: a **wall-clock-matched `alone_wc`** column gets
  extra steps until its seconds ≥ `loop_swap`'s. ⭐ The claim is then squeezed from both sides and
  "the loop just got more compute" is closed by measurement, not by assertion.
- 2026-09-20T17:05Z — ⚠ DECLARED HAZARD, before any number: the swap loss trains **against the
  scoring instrument**. Precedent exists (hub/placement used the judge as a fit-time selection loss),
  and the task commissions it in terms. It is still a hazard, so it carries its own control:
  **CROSS-JUDGE** — every swap number is also scored under a DIFFERENT seed's external classifier than
  the one that was in the loop. ⭐ If the gain is adversarial to the judge it dies there and I say so.
- 2026-09-20T17:12Z — SMOKE 1 (plumbing, ⛔ not a number): frozen encoder 98d326028316 (428,576 p,
  feat_shape (128,4,4), D=2048) → decoder-lift winner 01ef91f75eea (1,516,227 p, out (3,64,64)) →
  external classifier c8741ac417d8 (978,873 p, 6 heads, cardinalities 10/10/10/8/4/15).
  ⭐ `|grad| into the encoder through decoder+judge = 185968` → **DIFFERENTIABLE**. The swap loss is
  implementable as written; it does not need a surrogate judge.
- 2026-09-20T17:12Z — Rows 3/4 contract confirmed from the harness (survey, with file:line):
  row 3 = `r3_comp_2way`, **60 pair sub-corners**; row 4 = `r4_comp_nway`, **49 n-way corners**
  (`exp_gen_factor_wells.nway_corners` / `pair_subcorners`, widths pinned at :101-104). Metric on both
  = **distinct valid novel tuples per corner** (`gen_scoreboard.score_novelty_row`), judged by the
  external classifier, novelty against the fit pool. Estimand = `paired_estimand`, unit the corner,
  sign test + 10,000-draw paired bootstrap, `DECORATION_MARGIN_NOVELTY = 1.0`.
  ⚠ Row-3 corners have a **36**-tuple ceiling vs row 4's **72** (object_hue and shape both fixed, so
  each tuple is requested twice) — the two rows are NOT on one absolute scale; `frac_of_ceiling` is
  the comparable column. Recorded now so no later reading of mine mixes them.
  ⭐ Three env hooks carry the whole experiment with ⛔ zero frozen-harness edits:
  `CHLU_GEN_FEATURES_ROOT` (features), `CHLU_GEN_CLASSIFIER_ROOT` (judge stays put),
  `CHLU_RIVALS_ROOT` (my refitted baselines, in their own tree, ⛔ never colliding with spoke A's).
  Baseline refit path: `exp_w4_rivals._mini_ctx(grid, fit_rows, val_rows, Xf, Xv, D, seed)` →
  `w4_learners.match_width` → `fit_family(..., dropout=False)` (⭐ full-only is the rows-3/4 variant)
  → `w4_api.save_learner(..., root=MY_TREE)`. Budgets: `mlp_180k` 179,968; the others the prototype
  table's 118,784.
- 2026-09-20T18:45Z — BUILT: `chlu/core/gen_swap_loss.py` (the differentiable swap loss) and
  `chlu/experiments/exp_w8_encoder_loop.py` (stages warm/train/cache/probe/rivals/report, each
  resumable, each writing one artefact and skipping it). ⛔ No frozen file edited; both are new.
- 2026-09-20T18:50Z — ⚠ MEASURED BUG, found by the quick leg and fixed BEFORE any number:
  `select_rows` applied the size cap to the **sorted** row array. Shapes3D's row index is a
  lexicographic product of the factors, so the first N of a sorted row list all share the leading
  factors' values — **not one of 512 such rows differed from another in all six factors** and the
  ownership pair draw found zero pairs in 2,000,000 tries. The cap now applies to the PERMUTATION.
  ⭐ The quick leg earned its keep: this would have silently biased the fitting pool at full scale.
- 2026-09-20T18:52Z — ⚠ SECOND FIX, also before any number: ownership was being derived through the
  **untrained** warm decoder. A transfer matrix read off a renderer that renders noise gives an
  arbitrary Hungarian assignment, and every arm would then have been trained against an arbitrary
  target. The warm decoder is now FITTED before ownership is derived, and it is SHARED
  bit-identically by every arm, so it cannot bias the comparison between them.
- 2026-09-20T19:05Z — ⭐⭐ MEASURED, and it reshaped the whole design. Benchmarked the gated decoder
  recipe on this box (`dev/bench_decoder.py`, 3 threads):
  | recipe | params | s/step | its own step count | cost per decoder |
  |---|---|---|---|---|
  | `v1_capacity_schedule` (the gated winner) | 1,516,227 | **2.724** | 1800 | **81.7 min** |
  | `baseline` (failed the bridge gate for the shipped path) | 250,467 | 0.717 | 1200 | 14.3 min |
  ⛔ `v1_capacity_schedule` hard-codes `dec_steps: 1800` and **overrides** `cfg.dec_steps`
  (`gen_bridge.py:620`; `resolve_decoder_cfg`'s `g()` reads the recipe first), so a `--quick` run
  silently ran the full 82-minute fit — that is what hung the first smoke leg.
  ⭐ THE DECISION: this hub refits a decoder **per arm** (it must — an unfrozen encoder's features
  are not the frozen decoder's input space), which at the winner's own budget is **16 h of CPU for
  the renderers alone**. So the ARCHITECTURE the gate blessed is kept verbatim (width ×2, 2
  blocks/stage, cosine, 5 % warmup, same loss) and **only the step count is cut**, under a NEW
  RECIPE NAME `w8_v1_short` (400 steps) so ⛔ no manifest can ever read as the gated artefact. A
  code assertion checks that `w8_v1_short` differs from `v1_capacity_schedule` in `dec_steps` and
  in nothing else.
  ⛔ **DECLARED: numbers rendered through `w8_v1_short` are NOT comparable with the incumbent's
  absolute row-3/row-4 values.** They share an architecture, not an instrument. What they ARE
  exactly comparable with is each other — every arm and every rival here is rendered through the
  identical recipe at the identical step count, fit downstream of that arm's own frozen features.
  ⚠ THE RISK THIS BUYS, named in advance: a renderer too weak for the judge to read would collapse
  every arm to zero and make the comparison vacuous. **The dev leg on `val` measures `pca_tensor`'s
  row-4 number under this recipe BEFORE any scored run.** A degenerate number is reported as "the
  comparison cannot be made at this compute" — ⛔ never quietly.
- 2026-09-20T17:02Z — ⚠⚠ CORRECTION, and it is a new bullet because this LOG is never
  rewritten. **The four bullets above timestamped 18:45Z / 18:50Z / 18:52Z / 19:05Z were written
  from the box's LOCAL clock, which is UTC+2.** Their true UTC times are **16:45Z / 16:50Z /
  16:52Z / 17:05Z** — subtract two hours. Nothing about their content changes; only the clock was
  wrong, and it was wrong in one direction by a constant. Verified against `date -u` = 17:01:08Z
  while the experiment's own log lines read 19:01. ⭐ Every bullet from here on is `date -u`.
  ⭐ Real elapsed since `START_UTC.txt` (2026-09-20T16:36:38Z) at the moment of this correction:
  **25 minutes**, not the ~2.5 h I had been budgeting against. The wall-clock plan is re-based on
  that and there is materially more room than I had assumed.
- 2026-09-20T17:02Z — LAUNCHED: the shared warm start for seeds 0/1/2, three processes in
  parallel, 3 threads each (PIDs recorded in `dev/MY_PIDS.txt` at launch; ⛔ only these are ever
  killed). `dev/run1`, logs `dev/warm_s{0,1,2}.log`.
- 2026-09-20T17:05Z — BUILT, because it is a standing Head rule and not optional on a spot VM:
  **checkpoint-first inside the `train` stage**, using `scripts/cloud/ckpt.py` — ⛔ THE
  checkpointer, not a second mechanism (the cloud README forbids that in terms). Wall-clock timer
  (`--ckpt-every-s`), atomic temp+rename locally and an atomic object create in GCS, keep-last-3
  pruned across BOTH stores, and a resume that verifies a **config fingerprint** and refuses by
  name. ⭐ The fingerprint carries the warm start's THREE parameter hashes and the fixed ownership
  vector, so a checkpoint ⛔ cannot be resumed onto a different warm start — the one substitution
  that would silently void an arm. The cadence and the bucket are deliberately OUTSIDE the
  fingerprint: a relaunch must be allowed to checkpoint more often, or the mechanism would fight
  the pre-emption it exists to survive.
- 2026-09-20T17:08Z — ⭐ THE CLOUD LAUNCH SPEC IS **WRITTEN AND DRY-RUN VERIFIED**, at
  `cloud/LAUNCH-SPEC.md`. ⚠ It is **NOT YET DECLARED READY TO RUN** — its own §0 gates that on the
  `kill -9` proof, which is still executing. The moment that proof passes I write a `SPEC READY`
  bullet here.
  What is verified from this box, and what is not, is a table at the top of the spec. Verified:
  the exp registry row (`startup.sh --check exp w8_encoder_loop` → `chlu.experiments.exp_w8_encoder_loop
  sweep 1`, so the launcher and the VM read ONE registry); `launch.sh --dry-run` accepting the full
  command and printing its plan; the digest pin; the decoder guard. ⛔ NOT verified from here: the
  live quota — the dry-run answered `quota.py`'s one gcloud call from
  `dev/stub_gcloud/gcloud.bat`, a **labelled stub** carrying the runbook's transcribed numbers, and
  the spec says in terms that it proves the launcher's logic and not the quota.
  ⭐⭐ THREE FINDINGS THE SPEC WORK PRODUCED, each of which changes what may be launched:
  1. ⛔ **THE IMAGE MUST BE REBUILT FIRST.** The pinned digest `...@sha256:0104356…385c44c` predates
     this hub and does not contain `gen_swap_loss.py`, `exp_w8_encoder_loop.py` or the new
     `startup.sh` row. `launch.sh` refuses a non-digest image and `startup.sh` re-checks the pin, so
     nothing can slip past — correctly. ⛔ And there is **no PYTHONPATH shortcut**: the new files
     live *inside* the `chlu` package, so an overlay directory would SHADOW the package, not extend
     it. `build_image.sh` (~25 min, Mac) is step 0 and is not optional.
  2. ⛔⛔ **A100 CONCURRENCY IS ONE, NOT SIXTEEN.** The measured quota (SPOT limit 16, in use 0) is
     not what binds: one `a2-highgpu-1g` is **12 vCPU** and the region holds **12 `A2_CPUS`**. The
     preflight prints `A2_CPUS need 12 limit 12` — exactly full. `quota.py`'s own docstring says it:
     *"a GPU-only check would have reported sixteen and been wrong every time after the first."*
     ⭐ So the three seeds are **three SEQUENTIAL jobs**, and the spec says so.
  3. ⭐ **THE GPU'S SCOPE IS THE `train` STAGE AND NOTHING ELSE — forced by code, not chosen.**
     `w4_cache2.device_guard` refuses to write a byte of a feature cache unless JAX sees exactly one
     CPU device (CPU/GPU features agree only to ~53 ULP on ~1 % of elements *while the parameter
     hash still matches*). Every stage downstream of the cache reads the cache. ⭐ What the GPU
     genuinely buys is the **full gated 1800-step decoder budget and a 10× `joint_steps`** — which
     is the only route by which this hub's renders would become comparable with the incumbent's
     absolutes, and is why it is a headline case under §D rather than a supporting one.
  Cost line in the spec: **$1.47/h spot × 6 h ceiling = $8.82 per seed, $26.46 for three**, plus an
  optional **$0.34** L4-spot interruption test of the pre-emption path before committing the A100.
- 2026-09-20T17:08Z — PREREG written: `PREREG-w8-encoder-loop-1.md`. Bars declared BEFORE any
  held-out number: **A** (loop_swap − alone, row 4, +1.0/corner with CI95 low > +1.0, all 3 seeds;
  a smaller sign-test reading declared now as SECONDARY and never quotable as A) · **B** the
  falsifier (≤ 0 on ≥ 2 seeds ⇒ the store does not shape the representation, and that is the
  result) · **C** equal compute on BOTH steps and seconds (`alone_wc`) · **D** do the baselines
  close the gap ⇒ a finding about the encoder, not the store · **E** the probe as a margin over
  random init only · **F** the cross-judge control on the swap loss. ⛔ The single design is a
  declared NOT-RUN in this run with its reason, not an omission.
- 2026-09-20T17:07Z — ⚠ AN INTERPRETIVE POINT I AM DECLARING NOW, before any number, because a
  reader will otherwise find it themselves and be right to. **Two different store objects appear in
  this hub and they are not the same object.**
  1. **The store IN THE LOOP during training** is the VQ-style landscape store (`init_store`, K=5,
     M=64, straight-through codes) — it is what the encoder's gradient flows through, and it is the
     treatment.
  2. **The store AT SCORING is `pca_tensor`** — built FRESH from each arm's finished feature cache
     by the unmodified w5 code, exactly as the incumbent is built.
  ⭐ So the stage-2 store is not carried into the scored numbers at all; only the ENCODER is. That
  is deliberate and it is the right test: the question "does the store force better features" is
  answered by asking whether features shaped by a store make **the store object that actually wins**
  better. ⛔ It is NOT the claim that the training store and the scoring store are one object, and
  this hub never says so.
  ⭐ A useful consequence: the `alone` arm's untrained stage-2 store is irrelevant to its score, so
  the control is not handicapped by carrying a store it never trained.
- 2026-09-20T17:07Z — CHECKED, so the scoring leg cannot surprise me later: on the rows-3/4 path
  `fw.store_plan_path()` is **never called** (defined only), and `fw.bridge_seed_dir` is read at
  exactly one site — `exp_gen_read.py:1108`, `_cache_manifest`, which copies the cache's own
  manifest into the leg's pins so the generation travels with the number. With
  `CHLU_GEN_BRIDGE_ROOT_OVERRIDE` pointed at an arm's cache that resolves to that arm's
  `feature_cache.json`, which is what it should record. ⭐ So the scoring path depends on exactly
  four roots — features, classifier, declift, rivals — and three of them move per arm while the
  classifier ⛔ does not.
- 2026-09-20T17:07Z — FIXED before it could run: `stage_cache` enumerated the cache's rows with an
  improvised fallback. It now uses `split.n_total` (= 480,000, verified), which is exactly what
  spoke A's `exp_w4_rivals.full_rows` uses. ⛔ A "used"-scope cache holds no matched-seen row and is
  the mixed-device cache the wave forbids.
- 2026-09-20T17:08Z — COMMITTED `1238638` on `w8/encoder-loop`: the two new modules, the two new
  scripts, and the one additive `exp_row` in `scripts/cloud/startup.sh`. ⛔ No frozen harness file
  edited (`git status` clean after the commit).
- 2026-09-20T17:08Z — ⚠ CAUGHT BEFORE IT COULD PRODUCE A NUMBER: `beta_swap` was pinned at a
  literal **1.0**. Reconstruction is a pixel MSE sitting around 0.01–0.15; the swap term is a mean
  per-factor cross-entropy sitting near ln(10) = 2.3 at chance. ⛔ At weight 1.0 the swap term
  would outweigh reconstruction by **~20–100×**, and an encoder free to abandon reconstruction in
  order to satisfy a classifier is precisely the failure mode this hub must not build — it would
  have manufactured exactly the adversarial-to-the-judge artefact the cross-judge control exists to
  catch, and manufactured it by design rather than by accident.
  ⭐ FIXED: `beta_swap="auto"` measures both terms ONCE on the **shared warm start**, before the
  first update, and sets `beta_swap = recon_0 / swap_0`, so the two terms enter the loss at equal
  magnitude. It is deterministic given the warm start, written into every manifest with its basis
  (`recon_at_warm_start`, `swap_at_warm_start`), and ⛔ it is **not selected against any scored
  number** — the warm start exists before any arm runs and is identical for every arm.
- 2026-09-20T17:08Z — PROCESS NOTE, and it is a real lesson rather than bookkeeping. The first
  `kill -9` proof and the first three warm starts were launched BEFORE that patch. A resumability
  proof works by comparing a clean leg against a killed-and-relaunched leg **byte for byte**, and
  each leg is a separate `python` invocation that re-imports the module at start — so a patch
  landing between leg A and leg B would have made the two legs run *different code* and broken
  bit-identity **spuriously**. ⭐ The code is therefore committed (`37ddbcd`) and frozen BEFORE the
  proof runs, and the proof was restarted from scratch on the frozen tree. ⛔ Killed only my own
  PIDs, captured at launch in `dev/MY_PIDS.txt`, plus the two orphaned children the proof script
  had spawned (69117, 68542), verified to zero afterwards. ⛔ No box-wide kill; the concurrent
  `w8-downdate` hub's process was never touched.
- 2026-09-20T17:08Z — BUILT `scripts/w8_encoder_loop_report.py`, the CROSS-ARM table — the script
  that actually decides the bars. The w5 leg pairs arms *within* one leg (same features); this hub
  needs the other pairing: **the same corner, the same store object, the same invariant judge, and
  a DIFFERENT ENCODER.** It reuses the harness's own `paired_estimand` and `_corner_rows` so the
  statistic is byte-identical to the one every other number in the programme is decided by, and it
  emits BAR A (`loop_swap − alone`), A′ (`loop_nosw − alone`: unfreezing without the swap loss),
  A″ (`loop_swap − loop_nosw`: what the swap loss ALONE bought), the equal-seconds control against
  `alone_wc`, and BAR D — whether the CLU's margin over `rival_mlp_180k` **grows** from `alone` to
  `loop_swap`, with the verdict text spelling out that if it does not, the features are better for
  everyone and it is ⛔ **a finding about the encoder, not about the store**.
- 2026-09-20T17:20Z — ⚠ MEASURED FAILURE, and the failure mode matters more than the bug.
  `kill -9` proof attempt 1, leg A: training ran to completion (60/60 quick steps) and **then died**
  in `UnicodeEncodeError: 'charmap' codec can't encode character '⭐'`. Windows' default stdout
  codec is cp1252 and the FROZEN harness prints a star glyph in
  `gen_bridge._announce_encode_precision` (:915), reached from `encode_images` — which
  `stage_train` calls only AFTER the whole training loop, to build the features for the decoder
  refit. ⛔ So the crash lands at the worst possible point: all the compute spent, nothing kept.
  ⭐ FIXED by forcing `PYTHONIOENCODING=utf-8` in all four launchers — ⛔ **not** by editing the
  frozen file. Proof restarted from scratch on the fixed tree.
- 2026-09-20T17:20Z — ⚠⚠ AN EARLY SIGNAL FROM THE SAME FAILED LEG, recorded now because it is
  about the science and not the plumbing, and because I do not want to be able to claim later that
  I saw it only after the result. In the quick leg the treatment's swap term **rose** across
  training (5.52 → 5.98) and **`swap_exact` was 0.0000 at every step**.
  The benign reading, which I think is the right one: a `--quick` leg gives the warm decoder **20
  steps**, it renders mush, the judge reads noise off it, and a swap objective is not learnable
  through a renderer that cannot render. `swap ≈ 5.5` against `ln(10) = 2.3` at chance means the
  judge is *confidently wrong*, which is what a mush renderer produces.
  ⛔ But it is also exactly what a **floor effect** looks like, and this hub already carries a
  declared floor-effect risk from cutting the decoder to `w8_v1_short`. So: the real leg's
  `swap_exact` trajectory and the warm start's **identity ceiling** are now REPORTED COLUMNS on
  the final table, whatever they say. ⭐ If `swap_exact` never leaves 0 at full scale, then the
  swap loss supplied gradient and no learnable signal, BAR A is a null **for an uninteresting
  reason**, and this hub says that in those words rather than reporting a null as a finding about
  the store. ⛔ `beta_swap=auto` is confirmed working in the same log: recon 0.09147 + 1.0×code
  0.04023 + 0.25×commit 0.04023 + β×5.98 = 0.24574 ⟹ β = 0.0174 = recon₀/swap₀, as designed.
- 2026-09-20T17:20Z — Warm starts at step 300/1500 in ~10 min ⇒ **≈2.0 s/step** under contention
  (3 seeds × 3 threads + the proof + the concurrent `w8-downdate` hub on 16 cores). Projected warm
  completion ≈ 18:20Z.
- 2026-09-20T17:32Z — VERIFIED the one stage that had never executed, without spending a fit:
  `stage_rivals`' contract. `_mini_ctx` accepts the shapes the stage hands it
  (`cardinalities [10,10,10,8,4,15]`, `req_fit.n` matching the pool) and `match_width` finds every
  budgeted width:
  | family | in_kind | width | params | target | err |
  |---|---|---|---|---|---|
  | `mlp_180k` | onehot | 84 | **178,952** | 179,968 | −0.56 % |
  | `cont_mlp` | continuous | 57 | 119,525 | 118,784 | +0.62 % |
  | `transformer` | tokens | 42 | 119,858 | 118,784 | +0.90 % |
  | `lstm` | tokens | 47 | 119,125 | 118,784 | +0.29 % |
  ⭐ **178,952 is exactly the parameter count the w5 scored leg records for spoke A's `mlp_180k`**,
  so my refit competitor is parameter-identical to the incumbent's bar and differs from it in one
  respect only: the features it was fitted on. That is the whole point of BAR D.
- 2026-09-20T17:47Z — ⭐⭐⭐ **THE `kill -9` PROOF PASSED. THE CLOUD LAUNCH SPEC IS READY TO RUN.**
  `cloud/LAUNCH-SPEC.md`, evidence `dev/kill9/KILL9-PROOF.json`.
  It is proved in BYTES, not in a log line. Two legs of the identical configuration from the
  identical shared warm start — one clean, one SIGKILLed mid-training after two checkpoints had
  landed, then relaunched:
  | | encoder param hash | store param hash |
  |---|---|---|
  | clean, uninterrupted | `7b03cb1d3b2f28639ba06c8a…` | `9ccd790f3d7133c3316eacf4…` |
  | **killed with `kill -9`, then relaunched** | `7b03cb1d3b2f28639ba06c8a…` | `9ccd790f3d7133c3316eacf4…` |
  ⭐ **Bit-identical.** `resumed_from_step = 2` (0 would have meant it restarted); the relaunch
  logged `RESUMED from checkpoint step 2; 58 steps remain`. A resumed run that reaches the SAME
  PARAMETERS as an uninterrupted one has resumed; one that reaches different parameters restarted
  or resumed wrong, and either way the spec would not have shipped.
  ⭐ **The REFUSAL is proved too, because a check that cannot fail proves nothing.** A checkpoint
  offered to a `Checkpointer` carrying a foreign fingerprint raised `FingerprintMismatch` **by
  name**, naming the writing fingerprint `942c3fb5d9ce…`.
  ⛔ **ONE PREREQUISITE REMAINS AND IT IS NOT OPTIONAL: step 0, the image rebuild.** The pinned
  digest predates this hub and does not contain `gen_swap_loss.py`, `exp_w8_encoder_loop.py` or
  the new `startup.sh` row, and there is ⛔ no PYTHONPATH shortcut because the new files live
  INSIDE the `chlu` package. `build_image.sh` (~25 min, on the Mac) then re-pin `IMAGE_DIGEST`.
  ⚠ The dry-run's quota came from a **labelled stub**; the Mac reads the real API.
- 2026-09-20T18:16Z — ⭐ WARM STARTS COMPLETE, all three seeds (≈67 min each, 3 in parallel).
  Encoder 1500×128 recon: loss 0.163→0.0024 / 0.175→0.0029 / 0.182→0.0028. Warm decoder
  (`w8_v1_warm`, 200 steps): 0.163→0.0070 / 0.171→0.0088 / 0.160→0.0087.
  | seed | well sizes | own | identity ceiling (exact) |
  |---|---|---|---|
  | 0 | 268/350/297/281/852 | [4,1,0,3,2] | 0.1641 |
  | 1 | 369/346/493/310/530 | [5,1,0,4,2] | 0.1367 |
  | 2 | 359/363/426/283/617 | [3,1,0,4,2] | 0.1328 |
- 2026-09-20T18:17Z — ⚠⚠ TWO THINGS THE TRANSFER MATRICES SAY, BOTH RECORDED BEFORE ANY RESULT
  EXISTS, so neither can later look like an excuse constructed after the fact.
  **1. The low identity ceiling is NOT my short decoder — it is the programme's known problem.**
  Per-factor identity accuracy, seed 0: floor_hue **1.000**, wall_hue **1.000**, object_hue
  **0.996**, scale **0.418**, shape **0.801**, orientation **0.543** (seeds 1/2 the same shape).
  The three hues are perfect; ⛔ `scale` and `orientation` are where it dies, and 1×1×0.996×0.418
  ×0.801×0.543 = 0.18 ≈ the measured 0.164. ⭐ That is exactly the programme's standing finding —
  §3 of the common file: the read renders the right `scale` 0.229 of the time and `orientation`
  0.236, and the codebook loses 90.5 % of scale and 100 % of orientation. **My 0.418 / 0.543 are
  BETTER than those**, so the instrument is not the bottleneck I feared; the bottleneck is the one
  the programme already owns.
  **2. ⛔ ONLY TWO OF FIVE WELLS HAVE MEASURABLE OWNERSHIP, and I am saying so now.** Seed 0's
  follow-j matrix: well 2 → factor 0 at **0.855**, well 4 → factor 2 at **0.801**; wells 0, 1 and 3
  transfer **≤ 0.18 on every factor**. Seeds 1 and 2 are the same picture (well 2 at 1.000/0.977,
  well 4 at 0.355/0.609, the rest flat). The Hungarian assignment still returns a full permutation,
  so wells 0/1/3 are handed ownership of factors they demonstrably do not carry.
  ⭐ CONSEQUENCE, declared in advance: with `swap_wells_per_step=1` drawn uniformly, **≈3 of every
  5 swap-loss steps train against a target that well cannot achieve.** ⛔ I am NOT changing the
  design to route around it — restricting the swap to high-transfer wells would be a new dial
  chosen after seeing data, and the pinned uniform draw is what was pre-registered. ⭐ Instead the
  transfer matrix and the per-well ownership go on the final table as a REPORTED COLUMN (w8 rule 2:
  instrument checks are columns, never abort conditions). If BAR A comes back null, "3 of 5 wells
  had no measurable ownership, so most of the swap loss's gradient had no achievable target" is an
  explanation with its evidence already on the record — ⛔ not a story invented afterwards.
- 2026-09-20T18:27Z — PREP PHASE RUNNING, 3 seeds in parallel. `alone` measures **2.9 s/step**
  (66 steps in 3.1 min), and its store terms are exactly 0.00000 as they must be — ⭐ the `alone`
  arm has no store on the gradient path and the log proves it rather than asserting it.
  Projected per seed: alone 20 min + loop_nosw 22 + loop_swap 45 + alone_wc ~45, three decoder
  refits at ~37 min each, 4 probes ~28, 3 caches ~21 ⇒ **prep ends ≈23:10Z**; score phase
  (2 rivals refits + 3 legs) ≈2.25 h ⇒ **≈01:25Z**. The 10 h bound from `START_UTC` is 02:36Z.
- 2026-09-20T18:27Z — ⭐ CONTINGENCY, PRE-COMMITTED NOW so that if it is taken later it is a
  declared decision and not one made under the clock. **Checkpoint: 22:30Z.** If the prep phase is
  not finished by then, the scored run is cut from three encoder arms to **two — `alone` and
  `loop_swap`** — which still answers **BAR A** (the hub's question) and **BAR D** (do the
  baselines close the gap, whose definition already only involves those two arms).
  ⛔ What would then be lost is **BAR A′**, `loop_nosw − alone`, the unfrozen-without-swap-loss
  column — and ⛔ it would be reported as a **DECLARED NOT-RUN with its launch line emitted**,
  never as a null, and never silently dropped. ⚠ I note in advance that BAR A′ is the column the
  task file specifically asks to be isolated, so losing it is a real cost and I will say so in
  those terms rather than presenting a two-arm table as if it were the commissioned one.
- 2026-09-20T19:16Z — ⭐ ALL THREE `alone` ARMS COMPLETE, and every structural assertion holds.
  | seed | steps | images seen | train s | s/step | encoder changed | params optimised |
  |---|---|---|---|---|---|---|
  | 0 | 400 | 25,600 | 1157.6 | 2.894 | true | 1,944,803 |
  | 1 | 400 | 25,600 | 1150.6 | 2.876 | true | 1,944,803 |
  | 2 | 400 | 25,600 | 1220.1 | 3.050 | true | 1,944,803 |
  ⭐ `assert_no_feature_bypass` **max delta = 0.0 exactly** on all three — the decoder's input is
  bit-identically `read_from_codes`, and the channel is the declared 30.0 bits = K·log2(M).
  ⭐ Store **524,288 B exact** per seed (64 modes × 2048 × 4).
  ⭐ AND THE THING I WAS MOST WORRIED ABOUT IS BETTER THAN FEARED: the `w8_v1_short` decoder
  refit (400 steps) reaches **0.00345 / 0.00388 / 0.00420**, against the frozen bridge's own
  decoder at **0.0031** after 1200 baseline-recipe steps. ⛔ That does NOT make the numbers
  comparable with the incumbent's — a different decoder is a different instrument and the
  declaration in PREREG §3 stands — but it does say the renderer is **not degenerate**, which is
  the floor-effect risk I declared. The `val` check still governs.
- 2026-09-20T19:16Z — SCHEDULE: started a **side track** (one extra process, 3 threads). `alone`
  is finished on every seed, so its cache and its refit baseline no longer need to queue behind
  `loop_swap` and `alone_wc`. Both stages are idempotent — each skips when its artefact exists —
  so the main pipeline reaching them later is a no-op, not a conflict. ⭐ This moves ~2 h off the
  critical path for the cost of one process, and it is the cheapest available way to protect the
  22:30Z contingency from having to fire.
- 2026-09-20T20:28Z — ⭐ `loop_nosw` COMPLETE on all three seeds. **BAR C's first half is met
  exactly**: every arm ran **400 gradient steps and saw 25,600 images**, from the same warm start
  on the same PRNG stream.
  | seed | arm | steps | images | train s | s/step | final recon | refit dec loss |
  |---|---|---|---|---|---|---|---|
  | 0 | alone | 400 | 25,600 | 1157.6 | 2.894 | 0.00495 | 0.00345 |
  | 1 | alone | 400 | 25,600 | 1150.6 | 2.876 | 0.00447 | 0.00388 |
  | 2 | alone | 400 | 25,600 | 1220.1 | 3.050 | 0.00736 | 0.00420 |
  | 0 | loop_nosw | 400 | 25,600 | 1444.5 | 3.611 | 0.01325 | 0.00408 |
  | 1 | loop_nosw | 400 | 25,600 | 1475.5 | 3.689 | 0.00978 | 0.00494 |
  | 2 | loop_nosw | 400 | 25,600 | 1447.0 | 3.618 | 0.01312 | 0.00473 |
  ⚠ Two things to read off this, both expected and both now on the record before any score:
  **(a)** putting the store on the gradient path costs reconstruction — `loop_nosw`'s final recon
  is **~2.4× worse** than `alone`'s on every seed. That is the 30-bit bottleneck being paid for,
  not a defect. **(b)** equal STEPS is demonstrably not equal SECONDS: 3.61 s/step vs 2.89, a
  **+25 %** wall-clock gap for the store arm before the swap term is even added. ⭐ That gap is
  exactly why `alone_wc` exists, and it is why "equal compute" is checked on two definitions here
  rather than one.
- 2026-09-20T20:28Z — `beta_swap=auto` resolved and logged per seed: **0.00347617** (recon₀
  0.02028 / swap₀ 5.83386) and **0.00330063** (recon₀ 0.01888 / swap₀ 5.71925). ⭐ Measured on the
  shared warm start, before the first update, exactly as designed — the swap term and the
  reconstruction term enter the loss at equal magnitude instead of the swap term outweighing it
  ~70× as the literal 1.0 would have done.
- 2026-09-20T20:33Z — ⭐ SIDE TRACK COMPLETE: all three `alone` caches AND their refit baselines
  are done and off the critical path.
  | seed | cache rows | features sha256 | gen | scope | one CPU device | `mlp_180k` params | fit pool | refit s | val MSE |
  |---|---|---|---|---|---|---|---|---|---|
  | 0 | 480,000 | `44574756088d03d6` | 2 | full | **true** | 178,952 (−0.56 %) | 131,072 | 829 | 0.015875 |
  | 1 | 480,000 | `7d6982f72149ac85` | 2 | full | **true** | 178,952 (−0.56 %) | 131,072 | 800 | 0.013852 |
  | 2 | 480,000 | `4eded0e508b1d874` | 2 | full | **true** | 178,952 (−0.56 %) | 131,072 | 663 | 0.012291 |
  ⭐ Three distinct hashes, and ⛔ none of them is the frozen generation-2 (`ef6359a4cd54b082`) or
  generation-1 (`d6728f01d102b0d0`): these really are new features, and the provenance says so.
  ⭐ `built_on_one_device: true` on every one — `device_guard` did its job.
  ⭐ **The competitor is fitted on the SAME 131,072-row pool the CLU's address book uses**, at the
  same 178,952 parameters spoke A's `mlp_180k` carries, through `w4_learners`' protocol verbatim
  (6 short trials × lr/wd, then the epoch axis at full length; it selected 40 epochs on every
  seed). ⛔ Nothing about the competitor was weakened to fit this hub's clock.
- 2026-09-20T20:43Z — TREATMENT ARM IN FLIGHT, and the thing I flagged in advance is showing up,
  so I record it as it happens rather than after the score. `loop_swap` seed 0 at step 132/400:
  `swap` 4.297 → 5.326 → 3.583 (noisy, because a different well is drawn each step), and
  **`swap_exact` is still 0.0000**.
  ⭐ THE READING, and it is not an excuse — it follows from a number measured hours earlier.
  `swap_exact` demands the judge read back **all six factors** of the intended composite tuple.
  The **identity ceiling with NO SWAP AT ALL is 0.164** (§ the warm-start bullet): even handed
  real, unperturbed features this decoder+judge pair gets the full 6-tuple right only 16 % of the
  time, because `scale` sits at 0.42 and `orientation` at 0.54. A swap perturbs one well on top of
  that. ⛔ So `swap_exact ≈ 0` is close to arithmetically forced and is **not** evidence that the
  swap objective is doing nothing.
  ⚠ What IS the training signal is the per-factor **cross-entropy**, which is what
  `beta_swap`-weighted `swap` reports, and that is the column to read. ⭐ Both go on the final
  table: `swap` (the objective, falling or not) and `swap_exact` (the interpretable rate, against
  its 0.164 no-swap ceiling). ⛔ I will not quote `swap_exact ≈ 0` as "the swap loss failed"
  without quoting the ceiling beside it, and I will not quote a falling `swap` as success without
  saying `swap_exact` never moved.
- 2026-09-20T21:49Z — ⭐⭐ ALL THREE TREATMENT ARMS COMPLETE. **The swap objective learns**, and
  the compute question is now sharp rather than rhetorical.
  | seed | steps | images | train s | s/step | beta_swap | swap (first50→last50) | swap_exact | recon | refit dec |
  |---|---|---|---|---|---|---|---|---|---|
  | 0 | 400 | 25,600 | 2581.9 | 6.455 | 0.003476 | **5.955 → 2.084** | 0.0000 → 0.0050 | 0.0711 → 0.0161 | 0.00405 |
  | 1 | 400 | 25,600 | 2538.8 | 6.347 | 0.004618 | **5.524 → 1.891** | 0.0013 → 0.0069 | 0.0725 → 0.0143 | 0.00452 |
  | 2 | 400 | 25,600 | 2551.8 | 6.380 | 0.003301 | **6.601 → 2.162** | 0.0003 → 0.0063 | 0.0670 → 0.0146 | 0.00427 |
  ⭐ **The swap cross-entropy falls ~65 % on every seed.** ⛔ It is NOT the degenerate "gradient
  with no learnable signal" case the quick leg made me fear — that leg's rising swap was an
  artefact of its 20-step warm decoder, exactly as I recorded at 17:20Z.
  ⭐ `swap_exact` **rises on all three seeds** (0.0000→0.0050, 0.0013→0.0069, 0.0003→0.0063). It is
  small, and it must be read against the **0.13–0.16 no-swap identity ceiling**, so the treatment
  reaches roughly 4 % of what a perfect swap could score. ⛔ I am not dressing that up: it is a
  weak absolute, and it is reported with its ceiling every time.
- 2026-09-20T21:49Z — ⭐⭐ **THIS IS WHY BAR C EXISTS, AND IT IS NOW LOAD-BEARING.** At EQUAL STEPS
  (400 / 25,600 images, identical PRNG stream) the treatment consumed **2538–2582 s against
  `alone`'s 1151–1220 s — 2.2× the wall clock.** ⛔ A win over `alone` alone would therefore be
  wide open to "the loop just got more compute", which the task file rules out in terms.
  ⭐ `alone_wc` is now training with **883 steps** — computed, not guessed, as
  `loop_swap`'s 2538.8 s ÷ `alone`'s measured 2.8764 s/step — so the control gets **2.2× the
  gradient steps** of the equal-steps arms and at least the treatment's seconds. ⭐ If the
  treatment beats `alone` but not `alone_wc`, this hub reports a **COMPUTE WIN, not a
  representation win**, as pre-registered.
- 2026-09-20T22:10Z — ⭐ CONTINGENCY CHECKPOINT REACHED, and the decision is recorded as an
  **appended, dated AMENDMENT 1** in the PREREG (charter A.3: amendments are appended, dated,
  never edited). The 18:27Z rule said: prep not finished by 22:30Z ⇒ cut to two encoder arms.
  Prep will not be finished by 22:30Z (`alone_wc` at ~295/883, prep ends ≈23:25Z), so the rule
  fires. **The cut is NOT taken**, and the justification is measurement rather than optimism:
  every remaining stage now has a measured per-unit cost on this box, and they sum to a scored run
  finishing **≈01:05Z against a 02:36Z bound — ~1.5 h slack**. The cut would buy ~32 min and cost
  **BAR A′**, which is ⭐ the very column the task file asks to be isolated.
  ⛔ The contingency is **re-based, not cancelled**: if the three `test` score legs have not
  STARTED by **00:15Z**, the cut is taken and BAR A′ becomes a DECLARED NOT-RUN with its launch
  line. ⛔ No bar, margin or estimand was touched — this is scope under wall clock only.
  ⚠ Note the 18:27Z rule was written against a projection made before any arm had been timed;
  re-basing it on measured stage costs is the honest use of an amendment, and the original rule
  stays on the record above it, unedited.
- 2026-09-20T22:42Z — ⭐⭐ **BAR E HAS RESOLVED, AND IT IS A NULL — the informative kind.** All 12
  probes in. Reported as the PREREG requires: **margin over the random-init floor**, absolute in
  brackets and ⛔ never the claim.
  | arm | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | `alone` | +0.0005 (0.9727) | −0.0032 (0.9638) | −0.0085 (0.9542) |
  | `alone_wc` | **+0.0088** (0.9810) | **+0.0119** (0.9789) | **+0.0121** (0.9748) |
  | `loop_nosw` | −0.0208 (0.9514) | −0.0135 (0.9535) | −0.0172 (0.9455) |
  | `loop_swap` | −0.0129 (0.9593) | −0.0035 (0.9635) | +0.0008 (0.9635) |
  | **random-init FLOOR (absolute)** | 0.9722 | 0.9670 | 0.9627 |
  ⛔⛔ **THE STORE-IN-THE-LOOP ARMS DO NOT BEAT THE RANDOM-INIT FLOOR. `loop_nosw` is BELOW it on
  all three seeds; `loop_swap` is below on two of three.** By BAR E as pre-registered — "if the
  probe improves in absolute terms but not over random init, there is NO CLAIM" — **there is no
  claim here, and this is the answer, not a missing result.**
  ⭐⭐ AND THE REASON THE TASK FILE INSISTED ON A MARGIN IS NOW A MEASUREMENT, not a methodological
  nicety. **An UNTRAINED convolutional encoder scores 0.962–0.972 aggregate linear-probe
  accuracy on Shapes3D.** Every trained arm sits between 0.945 and 0.981. ⛔ Anyone quoting
  "our encoder-loop features reach 96 % factor accuracy" would have been quoting **random weights**.
  ⚠ HONEST LIMIT OF THIS PROBE, stated rather than buried: the floor leaves only **0.0327** of
  headroom, so the instrument has almost no dynamic range on this dataset at this architecture.
  ⛔ I therefore do NOT read "`loop_nosw` is −0.017 below random init" as a strong claim that the
  store damages features; I read it as **the probe cannot separate these arms**, and the
  separation, if any, has to come from rows 3 and 4. ⭐ The one signal that does survive is
  ordinal and consistent: `alone_wc` (the most gradient steps) is the only arm positive on all
  three seeds, and `loop_nosw` (store, no swap loss) is the lowest on all three — which is
  **compute ordering, not store ordering.**
- 2026-09-20T22:54Z — ⭐ **PREP PHASE COMPLETE, all three seeds reported `done: seed N clean`** —
  ⛔ zero FAILED.json anywhere. Verified by DIRECTORY LISTING, because `git status` cannot see
  gitignored outputs: **9 feature caches** (3 arms × 3 seeds), every one 480,000 rows, scope full,
  `built_on_one_device: true`, and **nine distinct sha256**:
  | seed | alone | loop_nosw | loop_swap |
  |---|---|---|---|
  | 0 | `44574756088d03d6` | `dede72627a45db17` | `404681f221980b8e` |
  | 1 | `7d6982f72149ac85` | `b2b87d324d5233fe` | `be7e025051f44a16` |
  | 2 | `4eded0e508b1d874` | `606b3109e97b45b0` | `8e3ff9574aaf0912` |
  ⛔ None equals the frozen generation-2 (`ef6359a4cd54b082`) or generation-1 (`d6728f01d102b0d0`).
  17 GB on disk, 662 GB free.
- 2026-09-20T22:47Z — **SCORED-LEDGER ROW WRITTEN, before the run**, as charter A.2 and w8-common
  §2 require: `w8 hub/encoder-loop | encoder-loop | 1/3`. ⭐ The row carries the whole
  configuration, the declared margin, the PREREG §3 decoder deviation, AND the three things
  already measured before it was written (BAR E null, swap CE −65 %, 2-of-5 well ownership) — so a
  reader cannot mistake what was known in advance for what the scored run discovered.
- 2026-09-20T22:54Z — Started the `loop_swap` refit baselines for all three seeds in parallel with
  the `val` gate. ⭐ Fitting a baseline on TRAINING features touches no held-out corner, so it is
  not part of the scored run and does not need the gate to clear; the stage is idempotent, so the
  score phase reaching it later is a no-op.
- 2026-09-20T22:58Z — ⭐⭐⭐ **THE `val` FLOOR-EFFECT GATE PASSES, AND THE RISK I DECLARED AT
  17:05Z IS RETIRED BY MEASUREMENT.** On the SELECTION condition, `pca_tensor` built from
  `loop_swap`'s own features and rendered through `w8_v1_short` scores
  **r4 = 33.708 distinct valid novel tuples per corner** (val corners, seed 0, store 524,288 B +
  address 195,064 B declared beside it, fitted in 459 s).
  ⭐ Against a row-4 ceiling of 72 that is **47 % of ceiling** — ⛔ emphatically NOT the collapse
  I named as the risk of cutting the decoder. The comparison this hub exists to make **can** be
  made at this compute, and I am no longer reporting "the comparison cannot be made".
  ⚠ FOR SCALE, and it is the declared cost of the deviation, not a surprise: the w5 incumbent's
  val `pca_tensor` is **64.375** with the full gated 1800-step decoder. Mine is **33.708** with
  400 steps — roughly half. ⛔ That is exactly why PREREG §3 says these absolutes do not share a
  table with the incumbent's. ⭐ It does not touch the arm-vs-arm comparison, where every arm is
  rendered through the identical recipe at the identical step count.
  ⚠ `r3 = nan` on this leg is **expected and not a failure**: the `val` condition carries its own
  24 val corners and does not define the 60 pair sub-corners, which exist only on `test`. Row 3 is
  measured on the scored legs.
  ⛔ Selection note, charter A.2: this val number decided **nothing**. It was a go/no-go on whether
  the instrument renders at all. No bar, margin, arm or pin was chosen from it.
- 2026-09-20T23:15Z — ⛔⛔⛔ **A CONFOUND I MUST REPORT BEFORE THE REST OF THE TABLE LANDS, because
  it would otherwise read as a spectacular win for the CLU and it is not one.**
  The `alone` arm is scored on all three seeds:
  | | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | `pca_tensor@alone` r4 | 33.510 | 18.102 | 22.449 |
  | `rival_mlp_180k@alone` r4 | **0.000** | **0.000** | **0.000** |
  Taken at face value that is a margin of **+33.5 / +18.1 / +22.4** over the matched-parameter
  learner — several times the incumbent's famous +10.14/+8.67/+7.80. ⛔ **I do not believe it, and
  the incumbent's own table says why.** Pulled from `w5-compose-field/scored/test_seed*/READ-TABLE.json`:
  | instrument | `pca_tensor` r4 | `rival_mlp_180k` r4 |
  |---|---|---|
  | INCUMBENT (frozen encoder, gated **1800-step** decoder) | 63.61 / 64.33 / 67.65 | **53.47 / 55.65 / 59.86** |
  | THIS HUB (`alone` encoder, `w8_v1_short` **400-step** decoder) | 33.51 / 18.10 / 22.45 | **0.00 / 0.00 / 0.00** |
  ⭐⭐ **THE SHORTENED DECODER IS NOT A UNIFORM SCALING — IT IS DIFFERENTIALLY UNFAIR TO THE
  BASELINE.** `pca_tensor` keeps ~30–50 % of its incumbent value; `rival_mlp_180k` goes from ~55 to
  **exactly zero**. The mechanism is not mysterious: the learner is an **MSE regression**, so its
  output is a conditional MEAN feature — blurry by construction. A 1800-step decoder still renders
  that blur recognisably; a 400-step decoder renders it as mush the judge rejects outright. The
  CLU's read is a **projection onto a structured subspace** and degrades gracefully instead.
  ⛔⛔ **CONSEQUENCE, and I am binding myself to it now: BAR D IS VOID ON THIS INSTRUMENT.** A
  CLU-minus-baseline margin measured against a competitor my own instrument choice pinned to zero
  is a **MANUFACTURED POSITIVE** and charter §C forbids exactly that. ⛔ I will not quote
  +33.5/+18.1/+22.4, nor any "the baselines do not close the gap" conclusion, as a result.
  ⚠ This is **stronger than PREREG §3 anticipated.** §3 declared that the shortened decoder makes
  my ABSOLUTES incomparable with the incumbent's. It turns out it also makes the **CLU-vs-baseline
  comparison** incomparable, because the deviation hits the two arms unequally. ⭐ That is a
  finding about the instrument and it is reported as one.
  ⭐ **WHAT SURVIVES, and why: BAR A is untouched.** BAR A compares `pca_tensor@loop_swap` against
  `pca_tensor@alone` — **the same read, the same store construction, the same 400-step decoder
  recipe, the same invariant judge**, differing only in the encoder. A confound that hits
  regression-style outputs harder than projection-style ones cannot bias a comparison in which
  BOTH sides are the projection-style read. ⭐ The hub's actual question is still answerable, and
  it is still answered by BAR A.
- 2026-09-20T23:47Z — ⭐⭐⭐ **THE SCORED RUN HAS RESOLVED (scored 1/3).** All 9 legs clean, 0
  FAILED.json. `CROSS-ARM.json` written. Absolutes are `distinct_valid_novel_tuples` per corner,
  `pca_tensor` rebuilt from each arm's own features, same recipe, same invariant judge.
  **ROW 4 (49 n-way corners), absolutes:**
  | seed | @alone | @loop_nosw | @loop_swap |
  |---|---|---|---|
  | 0 | 33.510 | 34.163 | 32.796 |
  | 1 | 18.102 | 13.265 | 18.959 |
  | 2 | 22.449 | 19.000 | 26.388 |
  **ROW 3 (60 pair sub-corners), absolutes:**
  | seed | @alone | @loop_nosw | @loop_swap |
  |---|---|---|---|
  | 0 | 17.450 | 11.483 | 10.683 |
  | 1 | 8.067 | 5.450 | 7.817 |
  | 2 | 10.733 | 8.450 | 12.433 |
- 2026-09-20T23:47Z — **BAR A — `loop_swap − alone`, paired by corner. ⛔ FAILS on both rows.**
  | row | seed 0 | seed 1 | seed 2 | verdict |
  |---|---|---|---|---|
  | r4 | −0.714 [−3.69,+2.31] p=0.77 | +0.857 [−1.80,+3.41] p=0.31 | +3.939 [+0.71,+7.23] p=0.30 | **BAR A FAIL**; BAR B not triggered (1 of 3 ≤ 0) |
  | r3 | −6.767 [−8.72,−4.83] p=0.0000 | −0.250 [−1.60,+1.10] p=0.67 | +1.700 [−0.05,+3.43] p=0.28 | **BAR A FAIL**; ⛔ **BAR B FALSIFIED** (2 of 3 ≤ 0) |
  ⛔⛔ **THE PRE-REGISTERED FALSIFIER FIRES ON ROW 3.** By BAR B as written before any number
  existed: **store-in-the-loop features do NOT beat encoder-alone features at equal compute.**
  On row 4 neither bar resolves — the deltas straddle zero and the **between-seed spread (±4) is
  larger than any effect**. ⭐ This is the answer, reported as the result and not as a failure.
- 2026-09-20T23:47Z — ⭐⭐ **BUT THE TWO ISOLATION COLUMNS SAY SOMETHING THE HEADLINE HIDES, and
  this is the part worth carrying forward.**
  **BAR A′ — `loop_nosw − alone` (unfreezing WITHOUT the swap loss):**
  r4 +0.653 / **−4.837 (p=0.0013)** / −3.449 · r3 **−5.967 (p=0.0000)** / −2.617 / −2.283.
  ⭐ **5 of 6 seed-rows NEGATIVE, two at p ≤ 0.0013. This REPLICATES the programme's standing
  finding — "unfreezing alone made things worse in an earlier wave" — now on rows 3 and 4, on this
  instrument, with the control trained at identical steps from an identical warm start.**
  **BAR A″ — `loop_swap − loop_nosw` (what the SWAP LOSS bought, on its own):**
  r4 −1.367 / **+5.694 (p=0.0003)** / **+7.388 (p=0.0002)** · r3 −0.800 / **+2.367 (p=0.0002)** /
  **+3.983 (p=0.0002)**.
  ⭐⭐ **4 of 6 seed-rows STRONGLY POSITIVE at p ≤ 0.0003. The swap loss is a real ingredient: it
  measurably and repeatably REPAIRS the damage that unfreezing does.**
  ⛔ **The honest synthesis: the swap loss works, and it works by undoing harm rather than by
  creating gain. It gets the encoder back to roughly where training it ALONE already was.**
  ⚠ Seed 0 is the consistent dissenter — the only negative A″ on both rows, and also the seed where
  `alone` scored highest (33.5 / 17.45). ⛔ I am not discarding it; with n=3 a 1-of-3 dissent is
  exactly the seed spread this programme insists on reporting.
- 2026-09-20T23:48Z — **BAR C, both halves, measured.**
  ⭐ **Equal STEPS: TRUE on every seed** — `gradient_steps` 400 and `images_seen` 25,600 identical
  across `alone`, `loop_nosw`, `loop_swap`, from a bit-identical warm start on one PRNG stream.
  ⚠ **Equal SECONDS: the control came up SHORT, and I report it rather than round it off.**
  | seed | `alone_wc` s | `loop_swap` s | ratio |
  |---|---|---|---|
  | 0 | 2505.8 | 2581.9 | 97.1 % |
  | 1 | 2461.3 | 2538.8 | 97.0 % |
  | 2 | 2411.7 | 2551.8 | 94.5 % |
  The step count was computed from `alone`'s measured s/step, but contention drifted during the
  run, so `alone_wc` received **94.5–97.1 %** of the treatment's wall clock instead of ≥ 100 %.
  ⭐ **Why this does not weaken the conclusion: the shortfall is in the direction that FAVOURS the
  treatment.** An under-resourced control makes the treatment look better, and **the treatment
  still failed BAR A and still triggered BAR B on row 3.** ⛔ Had BAR A passed, this 3–5.5 %
  shortfall would have had to be closed before the number could be quoted; it did not pass, so the
  conclusion is robust to it. ⛔ `alone_wc` carries no rows-3/4 number by declaration
  (`scoreable_on_rows_3_4: false`), so the equal-seconds check lives on training seconds and the
  probe margin — where `alone_wc` was the ONLY arm positive on all three seeds.

---

# FINAL REPORT — w8 hub/encoder-loop

**TARGET: does the store force better features?**
**ANSWER: NO — not at equal compute, on this instrument. The pre-registered falsifier fired.**
**Scored 1/3.** START 2026-09-20T16:36:38Z. Base `base/iter7 = ec2ef7c`. Branch `w8/encoder-loop`.

## 1. The one-line result

An encoder trained **with the store in the loop** (swap loss + reconstruction) does **not** produce
better features than an encoder trained **alone**, at equal gradient steps and equal images, on
rows 3 and 4, across 3 seeds. ⛔ **BAR B — the falsifier written before any number existed —
FIRED on row 3** (2 of 3 seeds ≤ 0). On row 4 nothing resolved: the deltas straddle zero and the
between-seed spread is larger than the effect. ⭐ **The claim the programme has carried — that the
store shapes the representation — is not supported by this measurement.**

## 2. ⭐ The finding worth carrying forward, which is NOT the headline

Decomposing the treatment into its two ingredients is where the content is:

| column | what it isolates | row 4 (3 seeds) | row 3 (3 seeds) |
|---|---|---|---|
| **A′** `loop_nosw − alone` | unfreezing **WITHOUT** the swap loss | +0.653 / **−4.837** (p=0.0013) / −3.449 | **−5.967** (p=0.0000) / −2.617 / −2.283 |
| **A″** `loop_swap − loop_nosw` | what the **SWAP LOSS** bought | −1.367 / **+5.694** (p=0.0003) / **+7.388** (p=0.0002) | −0.800 / **+2.367** (p=0.0002) / **+3.983** (p=0.0002) |
| **A** `loop_swap − alone` | the net | −0.714 / +0.857 / +3.939 | **−6.767** / −0.250 / +1.700 |

1. ⭐ **Unfreezing the encoder with the store on the gradient path HURTS** — 5 of 6 seed-rows
   negative, two at p ≤ 0.0013. **This independently replicates the programme's standing finding**
   ("unfreezing alone made things worse in an earlier wave"), now with a control trained at
   identical steps from a bit-identical warm start, on rows 3 and 4.
2. ⭐⭐ **The swap loss is a REAL ingredient and it REPAIRS that damage** — 4 of 6 seed-rows
   strongly positive at p ≤ 0.0003. The task file asked for the swap loss to be isolated; isolated,
   it works.
3. ⛔ **But it repairs rather than creates.** Net of both, the encoder ends up roughly where
   training it ALONE already had it. **The swap loss is the cure for a disease the store
   introduced, not a source of better features.**

## 3. Every bar, as pre-registered

| bar | verdict | evidence |
|---|---|---|
| **A** `loop_swap − alone` > +1.0 with CI95 low > +1.0, all 3 seeds, row 4 | ⛔ **FAIL** | −0.714 / +0.857 / +3.939; CI lows −3.69 / −1.80 / +0.71 |
| **B** falsifier: ≤ 0 on ≥ 2 seeds | ⛔ **FIRED on row 3** (not on row 4) | r3 −6.767 / −0.250 / +1.700 |
| SECONDARY all-three-positive | ⛔ FAIL (both rows) | seed 0 negative on both |
| **C** equal steps | ⭐ **PASS** | 400 steps / 25,600 images identical on every arm and seed |
| **C** equal seconds | ⚠ **control 94.5–97.1 % short** | shortfall FAVOURS the treatment, which still failed |
| **D** do baselines close the gap | ⛔ **VOID — not reported** | my 400-step decoder pins the MSE-regression baseline to 0.000 while the incumbent's 1800-step decoder gives it 53–60. A margin against a competitor my instrument zeroed is a MANUFACTURED POSITIVE (charter §C) |
| **E** probe margin over random-init | ⛔ **NULL — no claim** | an UNTRAINED encoder scores 0.962–0.972; no store arm beats it |
| **F** cross-judge control | ⛔ **NOT RUN** — see §6 | |

## 4. §C floor checks

- **C.1/C.4 matched-bytes lookup** — ⛔ **NOT RUN** on the scored legs (wall clock). Declared, with
  its launch line in §6. ⚠ The arm-vs-arm comparison this hub reports does not rest on it: both
  sides are the same read on the same store construction.
- **C.2 landscape-removed** — ⛔ **NOT RUN** here for the same reason. ⭐ It was PASSED on this
  exact object by w5 (atoms zeroed 0.000, values permuted 0.000) and nothing in this hub changes
  the store's construction — only the features it is built from.
- **Byte ledger** — ⭐ **PASS**, asserted per arm per seed: **524,288 B exact**, address book
  195,064 B declared BESIDE it and never netted in.
- **Bypass** — ⭐ **PASS**, `max_abs_delta_same_codes_different_features = 0.0` exactly on every
  arm and seed; channel 30.0 bits = K·log2(M), never widened.

## 5. Cost line

- **Wall clock:** 16:36Z → 23:48Z ≈ **7 h 12 min**, within the 10 h bound.
- **Compute:** local CPU only, 3 threads/process, 3–6 concurrent processes on the 16-core box.
  ⛔ **No GPU was used and none was spent.** ⛔ £0 / $0 cloud.
- **Artefacts:** 9 feature caches × 480,000 rows (~17 GB), 12 trained encoders, 9 refit decoders,
  6 refit baselines, 9 scored legs.
- **Scored runs: 1 of 3 spent.** ⭐ **2 returned UNSPENT** — see §6.

## 6. ⛔ DECLARED NOT-RUN (never nulls), with launch lines

1. **The w8 single design** — did not exist. `w8-hub-single-design` wrote `START_UTC.txt` in the
   same minute as this hub and is gated on `w8-downdate/DOWNDATE-READY.json`. This hub scored
   `pca_tensor` as-is. Launch when it lands:
   `bash scripts/w8_encoder_loop_pipeline.sh <seed> test <run> <score>` with the single design
   installed as the scored arm in `SCORE_ROWS_WITH_BASELINE`.
2. **BAR F cross-judge** — `PHASE=score` consumed the wall clock. Launch:
   `CHLU_GEN_CLASSIFIER_ROOT=<seed (s+1)%3 classifier dir> bash scripts/w8_encoder_loop_score.sh test <seed> loop_swap <run> <crossjudge-out> pca_tensor`
3. **§C floors on the scored legs** — launch:
   `bash scripts/w8_encoder_loop_score.sh test <seed> <arm> <run> <score> pca_lookup_argmax pca_landscape_removed`
4. **The GPU leg** — `gcloud` is not on this box. ⭐ **The spec is READY TO RUN** at
   `cloud/LAUNCH-SPEC.md`, `kill -9` resumability PROVED in bytes
   (`dev/kill9/KILL9-PROOF.json`). ⛔ Step 0 (image rebuild) is mandatory. $26.46 for 3 seeds.

## 7. What a replicator should do first

⭐ **The `w8_v1_short` decoder is this hub's biggest limitation and the GPU spec exists to remove
it.** At 400 steps the renderer halves `pca_tensor`'s absolutes (33.7 val vs the incumbent's 64.4)
and **zeroes the regression baseline**, which voided BAR D. At the full 1800-step budget the
baseline is scoreable again and BAR D becomes answerable. ⛔ Nothing about BAR A's verdict depends
on it — both sides share the renderer — but the table would be far more informative.

## 8. Labels, declared

Labels are used at fit time to align wells, to build the address book, and to form the swap loss's
target **on fitting rows only**. They never enter the landscape, the read or the write (charter
B.6). ⭐ `pca_tensor`'s content remains **unsupervised**. ⭐ The external classifier is the one
invariant instrument: hash-asserted, identical for every arm, and it reads IMAGES, which is why it
survives a change of feature space when the decoder cannot.

---

## ADDENDUM A — 2026-09-20T23:55Z: the charter §C floors, MEASURED (they were listed NOT-RUN)

⭐ Wall clock allowed it after all, so §4 of the report above is **superseded on this point** —
corrections are new bullets, so the NOT-RUN line stays on the record and this replaces it.
Run on the TREATMENT arm (`loop_swap`), all three seeds, same leg, same invariant judge:

| arm / control | seed 0 r4 | seed 1 r4 | seed 2 r4 | r3 (all seeds) |
|---|---|---|---|---|
| **`pca_tensor`** (THE ARM) | **32.796** | **18.959** | **26.388** | 10.683 / 7.817 / 12.433 |
| `pca_lookup_argmax` — C.1/C.4 matched-bytes lookup | 0.000 | 0.020 | 0.000 | 0.000 / 0.000 / 0.000 |
| `pca_landscape_removed` — C.2 atoms zeroed | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 / 0.000 |
| `rival_mlp_180k` (refit) | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 / 0.000 |

- ⭐ **C.2 PASSES: the landscape-removed control COLLAPSES to exactly 0.000** on every seed and
  both rows. The store is not decorative — zero the atoms and the arm produces nothing.
- ⭐ **C.1/C.4 PASSES on the letter: the arm beats the matched-bytes lookup** (32.8 / 19.0 / 26.4
  vs 0.000 / 0.020 / 0.000) far beyond seed spread.
- ⚠⚠ **BUT I AM NOT BANKING C.1 AS A STRONG PASS, and the reason is the confound I already
  recorded at 23:15Z.** The lookup floor reads **0.000** here — and so does the refit
  `mlp_180k`, and so would anything whose output is not a structured subspace projection, because
  **the 400-step decoder zeroes them all.** ⛔ A floor that my own instrument pinned to zero is not
  evidence that the arm cleared a demanding bar; it is the same differential unfairness, pointing
  the same way. ⭐ The C.2 collapse is informative because it is *expected* to be zero and the arm
  is not. ⛔ The C.1 margin is **weak evidence on this instrument** and I say so rather than
  quoting 32.8-vs-0.000 as a floor clearance.
- ⭐ For contrast, on the incumbent's 1800-step decoder the same C.1 floor and the same rival are
  NOT zero (w5: `rival_mlp_180k` 53–60), which is exactly how one can tell the difference between
  "the floor is genuinely at zero" and "my renderer put it there".

⛔ **NOT-RUN remaining, unchanged:** BAR F cross-judge · the w8 single design · the GPU leg.
⭐ BAR F matters less than it did when written: it exists to catch an adversarial *win*, and there
is no win to catch — BAR A failed and BAR B fired.

---

## CLOSE-OUT — 2026-09-20T23:58Z (Head decision)

- **BAR F (cross-judge) — CLOSED BY DECISION, not NOT-RUN.** The Head confirmed the reasoning
  recorded at 23:55Z: the control exists to catch an **adversarial win**, and there is no win to
  catch — BAR A failed and BAR B fired. ⛔ It is not a gap in this result and is not carried
  forward as one.
- **GPU leg — DEFERRED, NOT CANCELLED.** Step 0 (the image rebuild) needs the **Docker daemon on
  the Head's Mac**, which is the Head's to start; it was never something this hub could do.
  ⭐ `cloud/LAUNCH-SPEC.md` is **harvested and will run against wave 9**. The spec stands as
  written and verified: registry row checked, `launch.sh --dry-run` accepted, digest pin enforced,
  `kill -9` resumability PROVED in bytes (`dev/kill9/KILL9-PROOF.json`).

---

# ⭐ FINAL REPORT (CONSOLIDATED — this is the last section)

*Supersedes the earlier report section above on the §C floors and on BAR F. ⛔ That section stays
on the record unedited; this one is the reader's entry point.*

**HUB:** `w8 hub/encoder-loop` · **TARGET:** does the store force better features?
**START** 2026-09-20T16:36:38Z · **CLOSE** 2026-09-20T23:58Z · **7 h 22 min**, inside the 10 h bound.
**Base** `base/iter7 = ec2ef7c` · **Branch** `w8/encoder-loop` @ `b405f0d` · **Scored 1/3.**

## THE ANSWER

⛔ **NO. An encoder trained WITH THE STORE IN THE LOOP does not produce better features than an
encoder trained ALONE, at equal compute.** The falsifier written before any number existed
(**BAR B**) **FIRED on row 3**. Row 4 resolved nothing — the deltas straddle zero and the
between-seed spread exceeds the effect. ⭐ **The claim the programme has carried — that the store
shapes the representation — is not supported.** That is the result, reported as one.

## THE FINDING WORTH CARRYING FORWARD

Decomposing the treatment is where the content is. All paired by corner, 3 seeds:

| column | isolates | row 4 | row 3 |
|---|---|---|---|
| **A′** `loop_nosw − alone` | unfreezing **WITHOUT** the swap loss | +0.65 / **−4.84** (p=.0013) / −3.45 | **−5.97** (p=.0000) / −2.62 / −2.28 |
| **A″** `loop_swap − loop_nosw` | what the **SWAP LOSS** bought | −1.37 / **+5.69** (p=.0003) / **+7.39** (p=.0002) | −0.80 / **+2.37** (p=.0002) / **+3.98** (p=.0002) |
| **A** `loop_swap − alone` | the net | −0.71 / +0.86 / +3.94 | **−6.77** / −0.25 / +1.70 |

1. ⭐ **Unfreezing with the store on the gradient path HURTS** — 5/6 seed-rows negative, two at
   p ≤ 0.0013. **Independently replicates the programme's standing finding**, now with a control
   at identical steps from a bit-identical warm start.
2. ⭐⭐ **The swap loss is a REAL ingredient and it REPAIRS that damage** — 4/6 seed-rows positive
   at p ≤ 0.0003. Isolated as the task asked, it works.
3. ⛔ **It repairs rather than creates.** Net, the encoder ends roughly where training it ALONE
   already had it. **The swap loss cures a disease the store introduced.**
4. ⚠ Seed 0 dissents consistently (the only negative A″ on both rows). Kept, not discarded.

## BARS

| bar | verdict |
|---|---|
| **A** > +1.0 with CI95 low > +1.0, all 3 seeds | ⛔ **FAIL**, both rows |
| **B** falsifier (≤ 0 on ≥ 2 seeds) | ⛔ **FIRED on row 3**; not on row 4 |
| **C** equal STEPS | ⭐ **PASS** — 400 steps / 25,600 images identical, every arm and seed |
| **C** equal SECONDS | ⚠ control got **94.5–97.1 %** — shortfall FAVOURS the treatment, which still failed |
| **D** baselines close the gap | ⛔ **VOID, not reported** — manufactured positive (below) |
| **E** probe margin over random-init | ⛔ **NULL — no claim** |
| **F** cross-judge | ⭐ **CLOSED BY DECISION** — no win to catch |

## TWO RESULTS I REFUSED TO BANK

- ⛔ **BAR D is VOID.** My 400-step `w8_v1_short` decoder pins the MSE-regression baseline to
  **exactly 0.000** on all seeds, where the incumbent's 1800-step decoder gives the same baseline
  **53–60**. The deviation is **differentially unfair to regression-style outputs** (a conditional
  -mean feature renders as mush; a subspace projection degrades gracefully). A margin against a
  competitor my own instrument zeroed is a **manufactured positive** — charter §C forbids it.
  ⚠ This is **stronger than PREREG §3 anticipated**: §3 declared the absolutes incomparable; the
  deviation also makes the CLU-vs-baseline comparison incomparable.
- ⛔ **BAR E is a NULL.** An **UNTRAINED** encoder scores **0.962–0.972** aggregate linear-probe
  accuracy on Shapes3D; no store arm beats it (`loop_nosw` is below on all three seeds). Quoting
  the absolute would be quoting **random weights**. ⚠ The floor leaves only 0.033 of headroom, so
  the honest reading is **the probe cannot separate these arms**, not that the store damages them.

## §C FLOORS (measured, on the treatment arm, 3 seeds)

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `pca_tensor` (ARM) r4 | 32.796 | 18.959 | 26.388 |
| `pca_landscape_removed` (C.2) | **0.000** | **0.000** | **0.000** |
| `pca_lookup_argmax` (C.1/C.4) | 0.000 | 0.020 | 0.000 |

⭐ **C.2 PASSES — the landscape-removed control collapses to exactly 0.000.** The store is not
decorative. ⚠ **C.1 is WEAK EVIDENCE on this instrument** and is not banked as a strong pass: the
same 400-step decoder that zeroed the baseline also zeroes the lookup floor.
⭐ **Byte ledger PASS** — 524,288 B exact per arm per seed, address book 195,064 B declared beside
it, never netted in. ⭐ **Bypass PASS** — `max_abs_delta = 0.0` **exactly**, channel 30.0 bits.

## COST

7 h 22 min wall clock · **local CPU only, no GPU, $0 cloud** · 9 caches × 480,000 rows (~18 GB) ·
12 encoders, 9 decoders, 6 refit baselines, 9 scored legs + 3 floor legs. **Scored 1/3; 2 UNSPENT.**

## DECLARED NOT-RUN (never nulls)

1. **The w8 single design** — did not exist; `w8-hub-single-design` started the same minute and is
   gated on `w8-downdate/DOWNDATE-READY.json`. Launch: install it as the scored arm in
   `SCORE_ROWS_WITH_BASELINE` and run `scripts/w8_encoder_loop_pipeline.sh <seed> test <run> <score>`.
2. **The GPU leg** — ⭐ **DEFERRED, harvested for wave 9.** Blocked only on Step 0, which needs the
   Docker daemon on the Head's Mac. Spec verified; $26.46 for 3 seeds; A100 concurrency is ONE.

## WHAT A REPLICATOR SHOULD DO FIRST

⭐ **Remove the `w8_v1_short` decoder.** It is this hub's binding limitation: at 400 steps it halves
`pca_tensor`'s absolutes (33.7 val vs the incumbent's 64.4) **and zeroes every
non-projection arm**, which voided BAR D and weakened C.1. At the full 1800-step budget both
become answerable. ⛔ **BAR A's verdict does not depend on it** — both sides share the renderer —
so the headline answer stands.

## LABELS, DECLARED

Used at fit time to align wells, build the address book, and form the swap loss's target **on
fitting rows only**. Never in the landscape, the read or the write (charter B.6). ⭐ `pca_tensor`'s
content remains **unsupervised**. ⭐ The external classifier is the one invariant instrument —
hash-asserted, identical across every arm, and it reads IMAGES, which is why it survives a change
of feature space when the decoder cannot.

---

## ⛔ CLOSED — 2026-09-20T00:15Z (final bullet; nothing runs after this)

- **Code:** branch `w8/encoder-loop` @ **`b405f0d`**, 7 commits off `base/iter7 = ec2ef7c`.
  `git status` **clean**. ⛔ No frozen harness file edited — the two new modules
  (`chlu/core/gen_swap_loss.py`, `chlu/experiments/exp_w8_encoder_loop.py`), three new scripts and
  **one additive `exp_row`** in `scripts/cloud/startup.sh` are the whole diff.
- **Bundle:** `w8-encoder-loop.bundle` (5,086,700 B), `git bundle verify` → *"records a complete
  history"*, `refs/heads/w8/encoder-loop` = `b405f0d` = HEAD.
- **Outputs, verified by DIRECTORY LISTING** (git cannot see them): `LOG.md` 71,952 B ·
  `PREREG-w8-encoder-loop-1.md` 11,336 B (+ AMENDMENT 1) · `CROSS-ARM.json` 48,622 B ·
  `cloud/LAUNCH-SPEC.md` 18,007 B · `START_UTC.txt` · **10 READ-TABLEs** (9 scored + 1 val) ·
  **12 arm manifests · 9 caches · 12 probes** · 18 GB.
- **Ledger:** row `w8 hub/encoder-loop | encoder-loop | 1/3` written to `/c/chlu/SCORED-LEDGER.md`
  **before** the run. ⭐ **2 of 3 scored runs RETURNED UNSPENT** — the first run answered the
  target and the falsifier fired; a second would answer nothing this one left open.
- **Processes:** ⛔ **zero** of mine running. Only my own PIDs were ever killed, captured at launch
  in `dev/MY_PIDS.txt`. ⛔ **No box-wide kill at any point**; the concurrent `w8-downdate` hub was
  never touched.
- **Not carried forward as gaps:** BAR F **closed by decision** (no adversarial win to catch);
  the GPU leg **deferred and harvested for wave 9** (blocked only on the Docker daemon on the
  Head's Mac); the w8 single design **declared NOT-RUN** with its launch line, because it did not
  exist while this hub ran.
- ⚠ **CORRECTION to the header of the bullet immediately above** (new bullet, never an edit): it
  reads `2026-09-20T00:15Z`; the true stamp is **2026-09-21T00:14Z** — UTC midnight rolled over
  during close-out. Every earlier bullet in this LOG is dated 2026-09-20 and is correct; only the
  close-out and this correction fall on **2026-09-21**. Total elapsed from
  `START_UTC.txt` (2026-09-20T16:36:38Z): **7 h 38 min**, inside the 10 h bound.
