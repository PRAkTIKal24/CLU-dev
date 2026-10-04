# LOG - w4-spoke-a-rivals

Dial: the instrument and the comparison set. I run no arm and make no claim.
Falsified if a rival cannot be brought to matched parameters within +/-10%, or if generation-2
re-encoding does not reproduce its own hashes on a second pass.

- 2026-09-19 16:53 UTC - START_UTC.txt written into /c/chlu/outputs/spoke-a-rivals/ as first action. Directory was
  empty; no live claim existed. status=LIVE, pid recorded.
- 2026-09-19 16:53 UTC - Read task file /c/chlu/tasks-w4/w4-spoke-a-rivals.md. Scope confirmed: (a) rival set
  (prototype table, LACE with selection repaired in my own module, PoE-EBM at own budget run twice,
  generic learners MLP/transformer/LSTM/continuous-input-MLP at matched params +/-10%, 3 seeds),
  (b) cache generation 2 for bridge seeds 1-2, re-encoded from pixels, CPU-only, hashed, with a
  re-encode determinism check as my own falsifier, (c) re-score every arm on gen2, (d)
  retrain-from-scratch deletion comparator with cost line, (e) RIVALS-READY.json published PARTIAL
  as soon as the first rival is scorable - four hubs are blocked on it.
- 2026-09-19 16:53 UTC - Beginning read-only survey of the instrument before writing any code.

## Survey of the instrument (read-only)

- 2026-09-19 16:56 UTC - Verified chlu.__file__ = C:\chlu\worktrees\CHLU-w4-spoke-a\chlu\__init__.py. Numbers from
  this process are my worktree's.
- 2026-09-19 16:56 UTC - jax 0.9.0 reports backend=cpu, devices=[CpuDevice(id=0)], local_device_count=1. There is
  exactly ONE device visible in this process, so the "never split" rule is satisfied structurally,
  not just by intent. I will additionally pin JAX_PLATFORMS=cpu on every cache launch and record
  the resolved precision record in each manifest.
- 2026-09-19 16:56 UTC - FOUND THE REASON THREE SEEDS WERE BLOCKED. Generation-1 feature caches under
  $CHLU_GEN_BRIDGE_ROOT/shapes3d_recon:
    seed_0: scope="full",  n_rows=480000, features.npy 1,966,080,128 B
    seed_1: scope="used",  n_rows=9288,   features.npy    38,043,776 B
    seed_2: scope="used",  n_rows=9288,   features.npy    38,043,776 B
  Seeds 1 and 2 only cache train+eval_A+eval_B. Every matched-seen row on those seeds is currently
  served by exp_gen_bridge.py's SIDE BUFFER, i.e. re-encoded on the fly, per run, on whatever device
  the caller happens to be on. That is precisely the CPU/GPU-mixture hazard the task names, and it
  is why seeds 1-2 could not carry a scored row. Generation 2 = full-scope re-encode of seeds 1-2.
- 2026-09-19 16:56 UTC - DECISION (declared, with its cost): generation 2 will cover seeds 0, 1 AND 2, not just 1-2.
  Reason: a generation whose seed 0 came from the gen-1 pass on an unrecorded device and whose seeds
  1-2 came from this CPU is exactly the half-and-half cache the task forbids. Re-encoding seed 0 on
  the same device, same code path, same dtype makes the generation homogeneous, and it also buys a
  free instrument check: if my CPU re-encode of seed 0 reproduces gen-1's features_sha256
  d6728f01...f2e2 bit-for-bit, then gen-1 seed 0 was CPU-written and gen1-vs-gen2 is a no-op on that
  seed; if it does not, that difference is a finding the hubs must have before they score. Cost is
  one extra ~26 min local-CPU pass, which is free. Seeds 1-2 remain the mandated deliverable.
- 2026-09-19 16:56 UTC - Generation-1 manifests carry NO "encode_precision" key; core/gen_bridge.build_feature_cache
  writes one now. So gen-2 manifests will record backend/device/precision and gen-1's cannot. That
  asymmetry is itself the argument for not mixing the two.
- 2026-09-19 16:56 UTC - LACE breakage understood and located. chlu/config.py:5168 pins
  lace_selection_source="gen-arms-a7-wiring/runs/historical_688128"; exp_gen_cell.lace_selection()
  (line ~377) inherits (sigma, form, shrinkage) from that manifest rather than re-selecting.
  But exp_gen_cell builds its fit/val rows with split_fit_val() in STRATIFIED mode
  (exp_gen_cell.py:1717-1722), while historical_688128 selected its configuration under the old
  head_tail cut. The inherited configuration is therefore selected for a different fit set than the
  one it is scored on -> 0.6136 measured vs 0.7077 published, flagged regressed by
  regression_check() at exp_gen_cell.py:2173. The repair is to re-run the SAME 36-trial selection
  (exp_gen_baselines.run_lace_arm, lines 1045-1105: sigma x form x shrinkage, short schedule,
  criterion = validation aggregate probe accuracy) on the stratified rows, in MY module.
  chlu/config.py is NOT edited.
- 2026-09-19 16:56 UTC - Next: the re-encode determinism check, before anything expensive. It is my declared
  falsifier and it gates the whole cache track.

## The falsifier ran first, and it passed

- 2026-09-19 17:02 UTC - RE-ENCODE DETERMINISM CHECK: **PASSED**. Seed 1, first shard = rows 0..8191 (8192 rows),
  encoded twice in the same process through the frozen encoder, quantised to float16 (the cache
  dtype, so the hash covers the bytes that actually land in features.npy, not a float32
  intermediate).
    pass 1 sha256 = 1f6748ea769437571bd3f8de716c662ea308a749f7b18a2f73ef830f8eeb6af0
    pass 2 sha256 = 1f6748ea769437571bd3f8de716c662ea308a749f7b18a2f73ef830f8eeb6af0
  Identical. Wall clock 7.0s for both passes. Record at
  /c/chlu/outputs/spoke-a-rivals/determinism_check.json.
  Generation 2 is NOT void; the cache track proceeds. Had this failed I would have stopped here
  and reported, per the dial declaration.
- 2026-09-19 17:02 UTC - Device guard passes and is mechanical, not a promise: jax sees exactly one device,
  cpu:0, backend=cpu, resolved matmul precision HIGHEST, jax 0.9.0, numpy 2.4.1, host FORGIS_14.
  build_gen2_cache() refuses to write a byte unless n_devices==1 and the device is a CPU, and it
  writes that record into every manifest. JAX_PLATFORMS=cpu is pinned in the module before jax is
  imported.
- 2026-09-19 17:02 UTC - MEASURED THROUGHPUT, correcting the planning assumption: 8192 rows encode in ~3.5 s on
  this box, so a 480,000-row full-scope pass projects to ~4 min, not the ~26 min quoted from wave
  3. Three full seeds is therefore ~12 min of encode plus sha256 over 3 x 1.97 GB. This is well
  inside local CPU and there is no GPU case here whatsoever; no GPU justification lines are being
  written because no GPU run is being requested.
- 2026-09-19 17:02 UTC - Two environment repairs needed before the frozen path would run from this worktree, both
  external to the read-only files (nothing under the read-only list was touched):
  (a) gen_data.data_root must stay None so shapes3d.shapes3d_root() appends /shapes3d to
      $CHLU_GEN_DATA_ROOT; setting it to $CHLU_GEN_DATA_ROOT directly looks for the h5 one level
      up and raises FileNotFoundError.
  (b) stdout must be UTF-8. chlu/core/gen_bridge.py:915 prints a non-ASCII marker and a cp1252
      Windows console raises UnicodeEncodeError there - which would have killed a multi-hour
      cache pass over a log line. exp_w4_rivals reconfigures stdout/stderr to utf-8 at import.
- 2026-09-19 17:02 UTC - Launching the generation-2 full-scope build for seeds 1, 2 (the mandated deliverable) and
  then 0 (the homogeneity/cross-check seed), resumable block by block, 3 threads, one device.

## Generation 2 building; matched parameters solved

- 2026-09-19 17:11 UTC - GENERATION 2 / SEED 1 COMPLETE. 480,000 rows, scope=full, float16,
  features_sha256=86b44d0f1e6470e7f1e0e0c37c961974b8dcf8a083546c0e6600d71d53eb5b08.
  Built on cpu:0, one device, precision HIGHEST, 59 blocks of 8192, ~4.5 min. Every block
  carries its own sha256 in the manifest, so a partial cache is auditable and resumable.
- 2026-09-19 17:11 UTC - CORRECTION / my own error, recorded rather than tidied away: while trying to kill a
  slow foreground probe I ran taskkill against the first python PID the task list returned and
  killed the CACHE BUILD instead (PID 22692), at seed 2 block 37/59. No data was lost and nothing
  was corrupted: _progress.json had 37 blocks recorded with their hashes, the relaunch asserted
  the build description was unchanged and resumed at block 38. This is the resumability
  requirement doing its job under an unplanned kill, which is better evidence than a clean run
  would have been. Going forward I kill by matching the command line, not by list position.
- 2026-09-19 17:11 UTC - MATCHED TOTAL PARAMETERS: solved, all four generic learners inside +/-0.8%, far inside
  the +/-10% band. TARGET = 118,784 parameters. That target is the PROTOTYPE TABLE's own count,
  (sum_k C_k + 1) x D = (57 + 1) x 2048, and it is not a number I chose freely: at float32 it is
  475,136 B, which is exactly GenCellConfig.prototype_table_bytes_reference. So the generic
  learners are matched to the incumbent specialist they are being compared against, on the
  measure the pinned config already uses. Realised counts (count = every array leaf, including
  biases, layer norms, positional embeddings and the output head - nothing excluded):
      mlp          width 55   117,878 params   -0.763%   input: 57-dim concatenated one-hots
      cont_mlp     width 57   119,183 params   +0.336%   input: 6 normalised factor VALUES
      transformer  width 42   119,606 params   +0.692%   input: 6 factor tokens, 2 layers
      lstm         width 47   118,843 params   +0.050%   input: 6 factor tokens, recurrent
  For scale, the CLU's deployed store is 524,288 B (131,072 float32) plus a 74,240 B addressing
  head; I report the ratio against that too so a hub can read the rivals against either
  reference, but the MATCHING is to the prototype table and is declared as such.
- 2026-09-19 17:11 UTC - match_width() measures the count by BUILDING the model, never from a formula, so an
  architecture edit cannot silently drift the count; and it raises with the achievable counts
  either side if a family cannot be brought inside the band. It bisects (counts are monotone in
  width) after a linear sweep proved to cost minutes per family.

## FINDING FOR THE HEAD AND EVERY HUB, BEFORE ANYONE SCORES

- 2026-09-19 17:18 UTC - GENERATION 2 COMPLETE FOR ALL THREE SEEDS, full scope, 480,000 rows each, CPU-only,
  one device (cpu:0), precision HIGHEST, float16, every block hashed.
      seed 0  features_sha256=ef6359a4cd54b08287cf3709a2b4f7ed39c0fb504b32f5ddb65d483aefa813c3  4.5 min
      seed 1  features_sha256=86b44d0f1e6470e7f1e0e0c37c961974b8dcf8a083546c0e6600d71d53eb5b08  4.8 min
      seed 2  features_sha256=3ab14e583c3f500da060638cb9255072c20cf4053c212321d797469a080d3a3a  1.9 min (resumed)
  Root: /c/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/. Generation 1 is
  untouched on disk and still hashed; I only ever read it.
- 2026-09-19 17:18 UTC - ⚠ GENERATION 1 AND GENERATION 2 DISAGREE, AND THE DISAGREEMENT HAS THE PREDICTED
  SIGNATURE. Measured element-by-element on the rows the two generations share:
      seed 0:  5,004,082 / 983,040,000 elements differ = 0.509%, max |diff| 0.00390625
      seed 1:     90,684 /  19,021,824 elements differ = 0.477%, max |diff| 0.00195312
      seed 2:    116,435 /  19,021,824 elements differ = 0.612%, max |diff| 0.00390625
  Encoder parameter hashes are IDENTICAL in every case, so this is not a different encoder -
  it is the same encoder computing different bytes. The max deviations are 2^-8 and 2^-9,
  i.e. exactly ONE float16 ULP at those magnitudes, on about HALF A PERCENT of elements.
  That is the ~1%-of-elements / few-ULP signature the task warned about for a cache that was
  not all produced on one device. Generation 1 for these seeds was not written by the code path
  and device that just wrote generation 2.
  This is why generation 1 and generation 2 must never be mixed, and it is why every arm has to
  be re-scored rather than carried over. Record: /c/chlu/outputs/spoke-a-rivals/gen1_vs_gen2.json.
  ⭐ It does NOT by itself invalidate any generation-1 number; it says those numbers belong to a
  different cache and cannot be quoted beside generation-2 numbers. Whether it MOVES a published
  arm by more than its seed spread is a separate measurement, and it is the next thing I run.
- 2026-09-19 17:18 UTC - RIVALS-READY.json is LIVE at /c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json,
  status=partial, and it is rewritten automatically every time an artifact lands, so it can
  never be stale or ahead of the evidence. It already carries all three generation-2 caches with
  their sha256s, the device record, the determinism-check verdict, and the prototype table
  scored on seed 1. A hub can start scoring against the table now.
- 2026-09-19 17:18 UTC - First scored rival, on generation 2, seed 1: PROTOTYPE TABLE (the incumbent),
  118,784 parameters / 475,136 B. Judge (external classifier) in-corner balanced accuracy
  0.9453; probe companion 0.8844; distinct factor_tuples on B = 51 of 72. For orientation, the
  pinned knn_proto probe reference for seed 1 is 0.8920 on generation 1, so the table's
  generation-2 probe number sits 0.0076 below it - well inside the 0.05 regression tolerance.
  One seed is not a result; the other two follow.

## A hub's request changed what I ship, and found a real defect

- 2026-09-19 17:31 UTC - hub/read (session chlu-w4-read-33) asked for the rivals as a request->features
  CALLABLE rather than only as cell-harness scores. Its scoreboard is the value-holdout one -
  49 n-way corners of 72 requests for row 4, 60 pair sub-corners of 36 for row 3 - which are
  different request sets from my B / matched-seen pair, so it cannot pair its per-corner numbers
  against a composition_B aggregate. It offered me the alternative of emitting per-corner records
  on its corner ids. I took the callable: it is cheaper for me and four hubs can reuse one object.
  Answered within the hour, before it had committed to building a rival-runner of its own.
- 2026-09-19 17:31 UTC - ⚠ THAT REQUEST FOUND A REAL DEFECT IN MY GENERIC LEARNERS, and I am fixing it rather
  than handing four hubs a dash. arm_features' contract spells an unconstrained factor as nan,
  and rows 3 and 4 ask exactly those partially-specified requests. As first trained, all four
  generic learners had only ever seen fully-specified 6-factor requests, so on any pair corner
  they would have been answering out of distribution - and a rival answering blind is a rival
  weakened by my code, which is the one thing this spoke is told not to do. Repair: the learners
  are retrained with REQUEST DROPOUT (half the training requests fully specified, half a random
  subset of 1-5 factors, re-drawn every epoch). Declared as training-data augmentation over the
  request distribution the programme actually uses; the matched parameter count and the tuning
  protocol are unchanged, and validation MSE is still measured on FULL requests so the cost of
  the augmentation is visible rather than hidden inside the selection. Both numbers (full-request
  and partial-request val MSE) are stored per rival.
  The absent-factor encoding per family: one-hot -> the factor's block is all zeros; tokens ->
  a reserved token per factor at sum(C_k)+k (vocabulary widened by 6, widths re-matched); the
  continuous MLP -> a second channel per factor carrying a requested flag (input 6 -> 12 dims,
  width re-matched to 118,784 +/-10% again). The prototype table needed nothing: it already
  omits unrequested terms natively, which is exactly what arm_features does today.
- 2026-09-19 17:31 UTC - What the callable does NOT do, and why that is the honest answer rather than a gap:
  LACE and the PoE-EBM are not offered as request->features callables at all. Both are SAMPLER
  arms - two calls on the same request return different features by construction - so a
  "function of the request" would be a fiction. Both also have closed categorical vocabularies
  and cannot express an off-grid value; that limit is documented at
  chlu.core.gen_request.DenseRequests and is the METHOD's, not mine. The loader raises
  RivalRefusal with the reason rather than returning a nearest-grid-value number, because
  quietly snapping an off-grid request onto the grid would have the rival scoring a request it
  never received. Of the callables, only the table and the continuous MLP accept off-grid values;
  the one-hot and token families refuse, and that is their dash and footnote on row 2.
- 2026-09-19 17:31 UTC - Module: chlu/eval/rivals/w4_api.py, in MY tree. I did not put it on main - landing code
  on main is not mine to do from this worktree, and I told the hub so rather than doing it.
  API: load_rival(seed, name) -> callable(values (N,6) float, nan = not requested)
  -> (features (N,2048) float32, out_of_range (N,) bool). available(seed) reports what is
  loadable and why anything missing is missing.

## The incumbent is live on all three seeds

- 2026-09-19 17:34 UTC - PROTOTYPE TABLE scored on generation 2, all three seeds, and saved as a callable.
  118,784 params / 475,136 B. Judge in-corner balanced accuracy 0.9165 +/- 0.0306 (sd over 3
  seeds); probe companion 0.8817; judge composition-B aggregate 0.8650; distinct factor tuples
  on B 56.0 of 72. RIVALS-READY.json now reads table:true, seeds:[0,1,2], status still partial.
  ⭐ The seed spread of 0.0306 on the headline statistic is worth every hub's attention BEFORE
  it reads a table: a between-rival difference smaller than that is not a difference.
- 2026-09-19 17:34 UTC - Callable contract verified against the four cases that matter, not just imported:
  full on-grid request -> (N,2048) float32, oor all False; partial request (hue only, rest nan)
  -> returns, and the features genuinely DIFFER from the full request, so the omitted terms are
  omitted rather than defaulted; off-grid midpoint hue -> returns, oor False (inside range);
  value past the end of the grid -> oor True on every affected row. Told hub/read it can start.
- 2026-09-19 17:34 UTC - Report generator (chlu/eval/rivals/w4_report.py) reads every number back out of the
  JSON on disk and renders the table from that, one way only. No number in the final table is
  carried in a variable from the run that produced it, and no reference constant is ever derived
  from a rendered table.

## Learners landing; two things worth flagging early

- 2026-09-19 17:38 UTC - Widths re-matched after the partial-request encodings changed the input dimension, and
  everything still lands inside the band: mlp width 55 -> 117,878 (-0.76%), cont_mlp width 57 ->
  119,525 (+0.62%, input 6 -> 12 because of the requested-flag channels). Count is measured by
  building the model, so the re-match was automatic rather than something I had to remember.
- 2026-09-19 17:38 UTC - The two-stage tuning repair earned its keep immediately: all three seeds SELECTED
  epochs=40 for the mlp on the full-length leg. Under the original grid that axis would have been
  picked by the RNG stream, and a 20-epoch mlp would have gone into the table as "the matched
  MLP" on roughly half the seeds.
- 2026-09-19 17:38 UTC - ⚠ EARLY NUMBER, all three seeds not yet aggregated, but hubs should see it: the matched
  MLP's judge in-corner balanced accuracy is 0.9523 (seed 1), 0.8308 (seed 0), 0.8568 (seed 2).
  Two things follow. First, at MATCHED PARAMETERS a plain one-hot MLP is competitive with the
  prototype table (table: 0.9165 +/- 0.0306) and beats it on one seed. Second, the MLP's spread
  across seeds is far wider than the table's. A single-seed comparison between these two rivals
  would be meaningless, and I will report the spread beside every mean.
- 2026-09-19 17:38 UTC - The request-dropout cost is measured, not asserted: mlp seed 1 validation MSE is 0.02898
  on fully specified requests and 0.07963 on partially specified ones. Partial requests are
  genuinely harder for it - which is the honest reading, and exactly why rows 3-4 need the number
  rather than an assumption that a rival trained on full requests would have coped.

## Which population these numbers are on - measured, because four hubs are about to quote them

- 2026-09-19 17:41 UTC - hub/read sent a measurement that changes how any "test_rows" number should be read: the
  VALUE-holdout test index is ~80% rows carrying a value no training row holds, while its corner
  sets are retained-values-only at 1.0000. It cost that hub an hour - a read scoring R^2 = -7.08
  on random test rows turned out to be the best read it had on the corner population. It asked me
  to state which population each rival number is on. Fair, and I measured it rather than
  asserting it.
- 2026-09-19 17:41 UTC - MEASURED (seed 1; the split is identical across seeds): every one of my scored sets is
  RETAINED-VALUES-ONLY, zero rows carrying a value absent from the fit pool.
      B  (Montero corner)          72 rows   0 novel-value rows
      MS (matched-seen control)   199 rows   0
      fit                      32,256 rows   0
      val                         512 rows   0
  B realises shape={cylinder} and object_hue={8,9}, both present in the fit pool, in a
  CONJUNCTION the pool never carries. So this spoke's split is a COMBINATION hold-out and the
  value-holdout population is a different one; the 80% figure does not touch my rival numbers.
  Artifact: population.json, plus a `population` block in RIVALS-READY.json recording per
  statistic which population it is on, so the two cannot be quoted against each other by accident.
- 2026-09-19 17:41 UTC - The one exception, stated rather than buried: the deletion comparator's forget/retain
  numbers ARE about a value the arm never held (object_hue=0.5, grid index 5, the value
  gen_value_holdout declared before any build). They are the only numbers from this spoke that
  are row-2/5-7 territory and they are labelled as such.
- 2026-09-19 17:41 UTC - Row eligibility is now a declared artifact too (row_eligibility.json), decided in advance
  on structural grounds rather than discovered later by whoever assembles the table: which rival
  can enter row 2 (off-grid value), rows 3-4 (partial requests) and rows 5-7 (value hold-out),
  with a named footnote on every dash and, where the limit is the METHOD's rather than mine
  (LACE's and the PoE-EBM's closed categorical vocabularies), that is said explicitly. A referee
  reading an unexplained dash assumes we chose it.

## CORRECTION: the dropout variant is the wrong primary, and two variants now ship

- 2026-09-19 18:05 UTC - hub/read corrected itself and the correction matters. Its 60 pair sub-corners and 49
  n-way corners are built by exp_gen_factor_wells.run_arm via _request_values_from_rows =
  grid.values_of(rows), which names ALL SIX factors. So rows 3 and 4 ask FULLY SPECIFIED
  requests, not partial ones. Partial requests reach the learners only on rows 1, 2, 5 and 6,
  through the query sets.
- 2026-09-19 18:05 UTC - ⚠ CORRECTING MY EARLIER BULLET: I wrote that the request-dropout retrain was needed for
  rows 3-4. That was wrong - it is needed for rows 1, 2, 5, 6. The earlier bullet stands as
  written and this one corrects it.
  More importantly, the correction inverts which variant should be PRIMARY. Dropout training
  costs real accuracy on full requests - MEASURED at 20 epochs on seed 0, full-request val MSE
  ~0.0259 without dropout against ~0.0343 with it, about +32%. Shipping only the dropout variant
  would therefore hand the generic learners a handicap on exactly the population where they are
  judged, which is weakening a competitor through my own code. So BOTH variants ship, at the same
  matched parameter count:
      mlp / mlp_180k / cont_mlp / transformer / lstm   full-requests-only  = THE PRIMARY,
                                                        the variant rows 3-4 use
      *_dropout                                         partial requests in distribution,
                                                        for the query-set rows 1, 2, 5, 6
  The full-only variants now REFUSE a partial request rather than answering out of distribution,
  and the refusal names the _dropout variant, so the wrong one cannot be quoted by accident.
  The twelve already-trained dropout artifacts were renamed rather than rebuilt; no work thrown
  away.
- 2026-09-19 18:05 UTC - NEW RIVAL AT A SECOND BUDGET: mlp_180k, width 84, 178,952 parameters, -0.57% of 179,968.
  hub/read asked for it and the reason is good: NO arm on that hub can be parameter-matched to
  this rival set, because the pinned store ALONE is 131,072 parameters (524,288 B / 4), 10.3%
  above the table's 118,784. That is a property of the programme's byte pin - 524,288 B store
  against a 475,136 B table reference - not a choice any hub made. Its arm (store + product
  address book, R=128) sits at 179,968, 1.51x the rival target. Without a rival at that budget
  the "you only won on a budget gap" objection can only be answered with an argument. Same
  architecture as mlp, same tuning protocol, same seeds, distinct name so both rows can appear.
  Recorded in w4_learners.FAMILY_BUDGETS with that reasoning attached.
  ⛔ Scope limit I stated to the hub rather than letting it infer: mlp_180k tests whether a
  GENERIC LEARNER improves with 1.51x parameters. It does not license "parameter count does not
  matter" in general - a table and an address book are different objects.
- 2026-09-19 18:05 UTC - DELETION COMPARATOR RAN, cost line measured (seed 1): deleting object_hue=0.5 costs
  2.37 s and a refit of 29,067 rows by the gold standard, against ~0 s and 0 rows by
  mean-substitution. On the FORGET set the judge reads the deleted factor at 0.0196 (gold) vs
  0.0588 (edit); on the RETAIN set 1.0 vs 1.0. Seed 0: 2.84 s / 29,099 rows, forget 0.0 vs 0.0,
  retain 1.0 vs 1.0. Both methods preserve the retain set perfectly, so neither is merely damaged;
  the gold standard forgets at least as hard as the edit. ⚠ The cost ratio prints as "inf" because
  the edit is below timer resolution - I will report the edit cost as "below timer resolution,
  O(D) work on one prototype row" rather than as a ratio against zero.

- 2026-09-19 18:11 UTC - CORRECTION to row_eligibility.json, following from the same fact. I had mapped rows 3-4
  to the "partial request" capability. Wrong: rows 3-4 ask FULL six-factor requests, so every
  rival in the set can enter them. The artifact now separates four columns - rows 3-4 (full
  requests, every rival yes), row 2 (off-grid value), partial requests via the query sets
  (rows 1/2/5/6), and rows 5-7 (value hold-out) - and the previous version would have told a hub
  that its main rows needed a capability they do not need.
- 2026-09-19 18:11 UTC - Also fixed: the deletion cost line printed the gold/edit ratio as "inf" because the edit
  is a single O(D) row copy that lands below the clock's resolution. A ratio against a measured
  zero is a timer artifact, not a measurement, so it is now reported as "an O(D) copy of one
  prototype row and 0 refitted rows, below timer resolution, so no ratio is quoted". The stale
  seed-0/1 deletion artifacts were deleted so the running pass rebuilds them with the honest line.

## The aggregate was hiding the only interesting thing, and a hub's result made me look

- 2026-09-19 18:14 UTC - hub/read sent a per-factor result from its own val corners (its population, not mine, so
  I quote it only as the prompt it was): the prototype table reads orientation 0.909 where the
  oracle address into its store's atoms reads 0.550, and reads scale 0.105 where its arm reads
  0.751. Its point - that this is a cleaner statement of what the table is FOR than any aggregate
  - applies directly to my own table, which was printing aggregates only.
- 2026-09-19 18:14 UTC - I had the per-factor numbers in my artifacts all along and was not reporting them. Added
  a per-factor section. MEASURED on MY corner (B, 3 seeds, judge = external classifier):
      rival                floor_hue  object_hue  orientation  scale   shape   wall_hue  agg
      prototype table        1.000      1.000       0.995      0.204   0.991    1.000   0.865
      MLP (matched params)   1.000      1.000       0.644      0.741   0.981    1.000   0.894
  ⭐ This is the finding, and the aggregate destroys it. The MLP's aggregate is HIGHER than the
  table's (0.894 vs 0.865), and the two arms are good at different things: the table reads
  orientation at 0.995 and scale at 0.204; the MLP reads orientation at 0.644 and scale at 0.741.
  A rival that wins an aggregate while losing a factor another arm carries is not better at the
  task - it is better at a different mixture of it. Any hub ranking these two on the aggregate
  alone will draw a conclusion the per-factor numbers do not support.
  ⚠ Note the shapes agree across two different populations and two different instruments: the
  table is strong on orientation and weak on scale on my corner, and hub/read measures the same
  direction on its val corners. I am NOT combining those numbers - different populations - but
  the agreement is worth someone's attention.
- 2026-09-19 18:14 UTC - Report now prints per-factor accuracy for every rival above the aggregate, with the
  aggregate last and a line saying to read the factors first.

- 2026-09-19 18:16 UTC - ⛔ CAUGHT BEFORE IT COULD MATTER, in the re-score path. exp_gen_cell runs its two
  MECHANICAL preconditions in main(), at line 2711 - NOT inside run_seed. My rescore stage calls
  run_seed directly, so as first written it would have skipped both gates (classifier_gate.json
  must read PASS; gen-arms-a7-wiring/refusal_scores.json must read status=complete) and produced
  a completely normal-looking run. That is precisely the silent-bypass class the harness is built
  to prevent, and I would have re-scored every arm without the gates ever firing. Fixed: the
  rescore stage now calls check_preconditions() exactly as main() does and attaches the record to
  every manifest, so a re-scored number cannot be quoted without the evidence that its gates
  fired. Verified both pass right now: classifier gate PASS (floor 0.95, 3 seeds), refusal scores
  status=complete (not_run=[]).
- 2026-09-19 18:16 UTC - mlp_180k at 1.51x parameters does improve the generic learner on full-request validation
  MSE: 0.0198-0.0234 against the matched mlp's 0.0258-0.0290 across seeds. So the extra budget
  buys something real, which is what makes it a fair test of hub/read's budget-gap question
  rather than a straw man.

## The budget gap matters, and the measurement says so

- 2026-09-19 18:42 UTC - Full-only (PRIMARY) learners complete on all three seeds for mlp, mlp_180k and cont_mlp.
  Judge in-corner balanced accuracy, mean over 3 seeds, with distinct factor tuples on B:
      mlp_180k        178,952 params (1.51x)   0.9298   distinct 58.7/72
      prototype table 118,784 params           0.9165   distinct 56.0/72
      mlp             117,878 params (matched) 0.8983   distinct 46.0/72
      cont_mlp        119,525 params (matched) 0.6377   distinct 45.0/72
  Per seed, mlp_180k beats mlp on every one, on both validation MSE (0.0168/0.0171/0.0201 against
  0.0226/0.0232/0.0272) and judge in-corner (0.8782/0.9397/0.9716 against 0.8409/0.9271/0.9270).
- 2026-09-19 18:42 UTC - ⭐ THE RESULT hub/read asked for, and it does not go its way in the direction it might
  have hoped: at 1.51x parameters a plain one-hot MLP OVERTAKES the prototype table on the
  judge's in-corner statistic (0.9298 vs 0.9165) AND on distinct factor tuples (58.7 vs 56.0),
  while the parameter-MATCHED MLP loses to the table on both (0.8983, 46.0). So for the generic
  learner family the budget gap is not cosmetic - it is worth about +0.03 balanced accuracy and
  +12 distinct tuples between 1.0x and 1.51x. Any hub arguing "my arm wins and the budget gap is
  irrelevant" now has a measurement pointing the other way for at least one rival family, and it
  should be reported that way rather than left as an argument.
  ⛔ What it does NOT establish, stated so nobody over-reads it: this is one architecture family
  over one budget ratio on one corner. 1.51x of an MLP is not 1.51x of a store, and nothing here
  says a store scales the same way.
- 2026-09-19 18:42 UTC - cont_mlp is far the weakest rival at matched parameters (0.6377 judge, val MSE ~0.05-0.06
  against mlp's ~0.023). That is expected and is the price of its input being six NUMBERS rather
  than 57 one-hots - and it is the only learner that can enter row 2 at all. It is reported as a
  weak rival that can do something the others cannot, not hidden and not excused.

## A second hub found a real gap in MY artifacts, and it was a call-site gap

- 2026-09-19 18:48 UTC - hub/unseen-valid (chlu-w4-unseen-valid-15) reported that my fitted rivals SAW
  object_hue=0.5 - the value the value-holdout condition forbids - so they are not like-for-like
  with a value-holdout arm on rows 2 and 5. It read the counts straight off my checkpoint:
  table.rival.pkl n_per_value[object_hue] = [3236, 3223, 3164, 3243, 3312, 3157, 3324, 3114,
  3272, 3211] against its own knn_proto's [11044, 16295, 16228, 11205, 16515, 0, 16556, 11035,
  16113, 16081]. 3,157 rows at hue 0.5 against exactly 0.
  It is right, and it did the work to rule out the alternatives before reporting: it refit its
  own arm at my exact 32,256 rows (0.2590), at 65,536 (0.2586) and at 131,072 (0.2558) - flat, so
  not fit-pool size - and confirmed my _table_fn makes the identical embed_many call, so not the
  composition code. The difference is WHICH ROWS.
- 2026-09-19 18:48 UTC - ⭐ THE PART THAT WAS MY FAULT, and it is not the population itself. README section 3 and
  population.json already said the population plainly. But a consumer loads a callable, passes
  4096 requests, gets out_of_range=0 on every one - correct, because the value IS inside the grid
  - and receives no signal at all that the population is wrong for their row. available(seed)
  reported callable/offgrid/partial/param_count; population eligibility was documented for a
  READER and absent at the CALL SITE. Documentation that only reaches a reader is not a guard.
- 2026-09-19 18:48 UTC - Fixed, and fixed as data rather than as a sentence:
  (a) Every checkpoint now carries a population record - fit_population, n_rows_fit, and
      n_per_value per factor, the actual fitted counts. 30 checkpoints backfilled across 3 seeds.
  (b) RivalRead exposes .fit_population and .n_per_value, and gains
      .like_for_like({factor: [values]}) which returns a verdict naming every held-out value
      this rival DID fit, with the row count. Verified: like_for_like({"object_hue":[0.5],
      "shape":[0.0]}) returns False and reports 3,157 and 7,972 fitted rows respectively.
  (c) available(seed) now returns fit_population, values_never_fitted and the caveat.
  (d) RIVALS-READY.json carries a fit_population block naming BOTH declared hold-out values as
      present in my pool, with the measured counts.
  (e) row_eligibility.json marks rows 2 and 5-7 as ELIGIBLE BUT WRONG POPULATION rather than
      plain eligible - the mechanism can enter, the fit cannot.
  ⭐ Counts rather than a flag, deliberately: a consumer holding out ANY value can check
  mechanically without knowing how my pool was built, and the check keeps working if someone
  later holds out a different value.
- 2026-09-19 18:48 UTC - ⛔ I am NOT treating the hub's measured 0.1581 for my table against its knn_proto's
  0.2558 as a rival-versus-arm result, and I said so. Same mechanism, same bytes, different fit
  population; the number is about the pool, not about the method. Its decision to footnote my
  numbers and not count my table as beating its arm is correct and I endorse it.

## Refitting two rivals on another hub's pool, so it gets a comparable rival not a caveat

- 2026-09-19 18:54 UTC - hub/unseen-valid accepted the offer and exported its fit pool as row indices:
  C:/chlu/outputs/hub-unseen-valid/w4/runs/value_holdout_fit_pool.npz key "pool", 131,072 int64
  rows, seed-independent. New stage `refit_pool` fits the two families that CAN enter a
  value-holdout row - the prototype table and the continuous-input MLP - on THAT pool, on the
  generation-2 cache, for all three seeds. The one-hot and token families are excluded on purpose:
  their vocabularies are closed over the grid and my own RivalRefusal rejects every off-grid
  request, so refitting them changes nothing and they keep their dash and footnote.
- 2026-09-19 18:54 UTC - ⛔ I VERIFIED THE POOL MYSELF rather than trusting its label, because a pool labelled
  "holds out hue 0.5" that does not would make every refitted rival silently wrong, and the stage
  refuses to run on a pool that holds out nothing. Measured:
      object_hue [11044, 16295, 16228, 11205, 16515, 0, 16556, 11035, 16113, 16081] -> index 5 (0.5) ABSENT
      shape      [0, 43616, 43527, 43929]                                            -> index 0 (0.0) ABSENT
- 2026-09-19 18:54 UTC - ⚠ TWO FINDINGS ABOUT THEIR POOL, reported to them before their PREREG is final:
  (a) The pool holds out SHAPE 0.0 as well as object_hue 0.5, but their manifest's
      "held_out_values_absent" names only object_hue. Consistent with gen_value_holdout's declared
      held_out_shape_value=0.0 so almost certainly intended - but a consumer reading that key
      concludes one value is held out when two are, and any row-5 claim about shape rests on the
      second.
  (b) The pool OVERLAPS the Montero test rows on 36 rows. Irrelevant to their own split, but it
      is exactly why nothing fitted on their pool is scored on my B corner, and why nobody should
      cross-compare these refitted rivals against a Montero-corner number. Recorded in the refit
      manifest.
- 2026-09-19 18:54 UTC - Design choices, declared: the TABLE is fitted on the full 131,072 rows, matching the
  n_rows_fit their knn_proto reports, so that pair is genuinely like-for-like. The cont_mlp needs
  a selection set, so 512 rows are carved OUT of their pool rather than borrowed - the validation
  rows are then value-holdout-consistent too, instead of coming from a pool that contains the
  held-out value. Output under /c/chlu/outputs/spoke-a-rivals/rivals_vhpool/, every checkpoint
  carrying population.n_per_value so they can assert object_hue[5]==0 (and shape[0]==0) before
  scoring rather than trusting my label.
- 2026-09-19 18:54 UTC - ⚠ A CONFOUND I FLAGGED IN THEIR PLAN: they intend my cont_mlp as an independent
  implementation of their named falsifier, theirs being 573,440 params in-harness. Mine is at
  ~119k, a 4.8x gap. If the two AGREE that is a strong independence result; if they DISAGREE the
  likeliest explanation is the parameter gap, not implementation. Told them to pre-register
  either a same-budget comparison or an explicit acknowledgement that a disagreement is
  confounded - before the numbers exist, not after.

- 2026-09-19 18:58 UTC - ⚠ THE POOL REFIT CRASHED and has been relaunched. It died silently after writing seed 0's
  table - no traceback in the log, so a kill rather than an exception. Diagnosis: memory. The
  stage held the full 131,072 x 2048 float32 pool gather (~1.07 GB) while train() additionally
  materialised the whole fit input AND target set as device arrays (~1.07 GB each), several GB
  live at once in one process with four others running. Free physical memory measured 14.3 GB of
  63.4 GB after the fact, so it was a transient spike rather than a standing shortage.
  Fixed properly rather than by retrying:
    (a) train() now keeps inputs and targets in NUMPY and converts only the 256-row minibatch to
        a device array. Peak drops by ~2 GB per training run and every learner in the spoke
        benefits, not just this one.
    (b) the refit stage frees the full-pool gather as soon as the closed-form table is fitted.
    (c) the refit stage is now resumable per seed, so a second kill costs one seed, not all three.
  Seed 0's table was valid and survived; it is being rebuilt anyway so the manifest is written by
  one clean run.
- 2026-09-19 18:58 UTC - hub/unseen-valid confirmed both pool findings in its own worktree, corrected its manifest
  with a dated CORRECTION_ key rather than a silent patch, and asserted that its pool's overlap
  with its OWN value-holdout test rows is 0 - which is the check that actually governs its
  numbers; my 36-row figure was against the Montero test rows, a different split.
  It also changed its pre-registered pairing on the confound I flagged: instead of comparing my
  ~119k cont_mlp against its 573,440-param in-harness MLP (a 4.8x gap that would have confounded
  any disagreement), it will compare mine against its cont_mlp_TABLEMATCHED at 118,153 params.
  Budget-matched and population-matched. The 573k arm-matched MLP stays its falsifier of record.
- 2026-09-19 18:58 UTC - One asymmetry it raised and I offered to remove: my cont_mlp selects lr/wd/epochs on 512
  rows carved from its pool, while its in-harness MLPs used a fixed lr and fixed step count with
  no selection at all. That favours mine, which is conservative for its claim, so its plan to
  state it on the row is sound. I offered an additional cont_mlp_notune at its exact protocol so
  the independence row differs only in implementation - shipped ALONGSIDE the tuned one, never
  replacing it, because an untuned rival is a weakened rival and R27.3 forbids weakening a
  competitor through our own code.

- 2026-09-19 19:03 UTC - cont_mlp_notune added: an INDEPENDENT reimplementation of hub/unseen-valid's exact
  in-harness protocol, built from the spec after READING its code
  (/c/chlu/outputs/hub-unseen-valid/w4/code/cont_mlp.py) rather than trusting its summary of it -
  the two agreed. Protocol: depth 2, three Linear layers 6 -> h -> h -> 2048 with GELU between
  and none on the output, default equinox init from PRNGKey(seed) split depth+1; six raw factor
  values min-max normalised by their own realised grid range, no presence flag, no bracketing;
  raw float32 feature targets; optax.adam(1e-3), no weight decay, no schedule; 4000 steps of
  batch 512 sampled WITH REPLACEMENT from np.random.default_rng(seed); all 131,072 pool rows,
  no selection set, no early stopping. My independent width calc reproduces theirs exactly:
  h=55, 118,153 params, 99.5% of the 118,784 target - the same number their
  cont_mlp_TABLEMATCHED reports.
  ⭐ It ships ALONGSIDE the tuned cont_mlp, never instead of it. The tuned one is the proper
  rival; an untuned rival is a weakened rival and R27.3 forbids weakening a competitor through
  our own code. The untuned one exists only so the implementation-independence row differs in
  implementation alone rather than in implementation AND training protocol.
- 2026-09-19 19:03 UTC - Their reference numbers for that protocol, seed 0: final minibatch MSE 0.0341 at 118,153
  params and 0.0108 at 573,440. If mine lands near 0.034 at the matched budget the protocols
  agree; that is the check, and it is theirs to make, not mine.
- 2026-09-19 19:03 UTC - ⚠ Resource note worth recording for the programme, not just for me: hub/unseen-valid
  reports its scored run at 9.4 GB resident with a single ~8.6 GB materialised tensor. Five
  agents on 64 GB is tight, and anything holding a full 131,072 x 2048 gather plus device copies
  will be the process that dies. My train() no longer does. That is the second time today a
  memory-shaped failure was environmental rather than algorithmic.

## The deletion comparator has a real result, and wave 5 should see it early

Deleting object_hue=0.5 (grid index 5) from the fit pool. FORGET = validation rows carrying that
hue; RETAIN = validation rows carrying any other hue. "judge(del)" is the judge's accuracy on the
DELETED FACTOR; "agg" is its six-factor aggregate. Seeds 0 and 2 below; seed 1 still running.

      method                          wall clock   rows refit   forget(del)  retain(del)
      table_full_pool (reference)      18.7 s        32,256       1.0          1.0
      table_retrain_from_scratch        3.1 s        29,099       0.0          1.0
      table_edit_mean_substitution     ~0 s               0       0.0          1.0
      mlp_retrain_from_scratch         69.5 s        29,099       0.0          1.0
      mlp_180k_retrain_from_scratch    68.3 s        29,099       0.0          1.0
      lstm_retrain_from_scratch        91.9 s        29,099       0.0          0.996
      transformer_retrain_from_scratch 178.2 s       29,099       0.0          0.998
      cont_mlp_retrain_from_scratch    75.6 s        29,099       0.0          0.229   <-- damaged

- 2026-09-19 19:07 UTC - ⭐ FINDING 1, and it is the one wave 5 most needs: FOR A PROTOTYPE TABLE, THE CHEAP EDIT
  IS THE GOLD STANDARD. Retraining from scratch without the value and substituting the global
  mean for that value's prototype agree on every statistic measured - forget 0.0 vs 0.0, retain
  1.0 vs 1.0, forget aggregate 0.620 vs 0.620 on seed 0 and 0.611 vs 0.618 on seed 2, retain
  aggregate 0.805 vs 0.802. The reason is structural rather than lucky: FactorPrototypes.fit
  falls back to the GLOBAL MEAN for any value with zero fitted rows, which is exactly what the
  edit writes. Only the second-order change to the OTHER prototypes differs, and it is inside
  seed noise. Cost: 3.1 s and a 29,099-row refit against an O(D) copy of one row.
  ⛔ Scope: this is a property of a per-value-mean table, NOT a general licence. It says nothing
  about whether an edit approximates retraining for a store, an address book or a learner, and
  the learner rows below show the answer is not automatically yes.
- 2026-09-19 19:07 UTC - ⭐ FINDING 2: the comparator DETECTS A REAL DIFFERENCE BETWEEN RIVALS, which is what it
  was built for. Deletion leaves mlp and mlp_180k undamaged (retain 1.000), lstm and transformer
  barely touched (0.993-0.998), but DAMAGES cont_mlp badly - its retain accuracy on the deleted
  factor collapses to 0.229 from the table's 1.000. cont_mlp did not delete the value, it lost
  the factor. A method that forgets by getting worse everywhere has not deleted anything, and
  this is exactly the failure the forget/retain pair exists to expose. Reporting it plainly: the
  weakest rival in my set is also the one that cannot survive a deletion.
- 2026-09-19 19:07 UTC - ⭐ FINDING 3, the cost line wave 5 asked for, and it scales with the rival rather than
  being one number: exact deletion costs 3.1 s for the table, ~69 s for either MLP, ~92 s for the
  LSTM and ~178 s for the transformer, all on the same 29,099 retained rows. The table is ~57x
  cheaper to delete from exactly than the transformer, and for the table the exact route is not
  even necessary.
- 2026-09-19 19:07 UTC - Sanity checks that make the above readable rather than suspicious: the full-pool table
  reads the deleted factor at 1.0 on the forget set (it saw the value), every deletion method
  drops it to 0.0 (not to chance 0.1 - the global-mean prototype renders a specific WRONG hue,
  so the judge is systematically wrong rather than guessing), and retain stays at 1.0 for the
  methods that are not damaged. The aggregate drop of ~0.197 is close to the 1/6 expected from
  losing one of six factors, so there is little collateral damage beyond the deleted factor.

- 2026-09-19 19:09 UTC - An unplanned cross-check on the generation change, from the LACE repair's whitening
  diagnostics. chlu/config.py documents the values MEASURED on the shipped (generation-1) cache
  as |diag-1| = 0.0591 for shrinkage "none", 0.1436 for ledoit_wolf and 0.2322 for 1e-3. On
  GENERATION 2 I measure, per seed:
      seed 0   none 0.05844   ledoit_wolf 0.1482   1e-3 0.2360
      seed 1   none 0.05652   ledoit_wolf 0.1450   1e-3 (pending)
      seed 2   none 0.05031   ledoit_wolf 0.1414   1e-3 0.2127
  Same ordering, same exactness verdicts (only "none" passes the 0.10 tolerance, on every seed),
  but the values have moved in the third decimal. That is what a half-percent of elements
  differing by one float16 ULP should do to a downstream second-moment statistic: visible, small,
  and not a reordering. It is an independent confirmation that the generation change propagates
  into derived quantities rather than staying in the cache - which is the argument for
  re-scoring every arm rather than carrying numbers over.
  ⚠ Held loosely: the config's three constants are not labelled per-seed, so I cannot tell which
  seed they came from, and part of the gap may be seed rather than generation. I am not treating
  this as a measurement of the generation effect - the element-level comparison in
  gen1_vs_gen2.json is that measurement. This is corroboration, not evidence.

- 2026-09-19 19:11 UTC - Pipelined the generation-2 re-score rather than waiting for the whole rival set: a
  per-seed waiter launches that seed's rescore the moment its poe_ebm.json lands. The process
  count stays flat at three - a rival run ends as its re-score begins - instead of the box
  sitting idle on two cores while one straggling seed finishes. The re-score is the last
  acceptance item and the longest, so starting each seed early is worth the small orchestration.
- 2026-09-19 19:11 UTC - Current state: 3 rival runs in LACE (36-trial repaired selection now running on seed 0),
  1 pool refit in cont_mlp tuning, 3 rescore waiters sleeping. All verified consuming CPU rather
  than assumed alive - the earlier liveness check gave a false death report and I no longer trust
  a single zero reading.

- 2026-09-19 19:16 UTC - ⛔ AN INTEGRITY CHECK ON WHAT I HAD ALREADY PUBLISHED CAUGHT A REAL GAP. I ran every
  loadable callable (33 across 3 seeds) through a shape/dtype/finiteness check plus a
  like_for_like() assertion. Thirty passed; the three `lstm` full-only checkpoints FAILED - they
  carried no population record. Cause: they were written after the backfill had already run, by
  processes that had imported the pre-population version of save_learner. So a hub loading
  `lstm` would have got like_for_like() -> None ("no population record") instead of a verdict,
  on a rival I had already told hubs was available. Re-stamped; all 33 now carry it.
  ⭐ This is the second time today that publishing something and then checking it found a defect
  that reasoning about it would not have. The gap existed for roughly twenty minutes in an
  artifact four hubs were told to read.
- 2026-09-19 19:16 UTC - Guard added so it cannot recur silently: publish_ready() now scans every callable's
  metadata and reports `callables_missing_population_record` and `population_record_complete`
  in RIVALS-READY.json itself, and status cannot reach "ready" while any are missing. The check
  lives in the file hubs actually read, not in a script I have to remember to run.
- 2026-09-19 19:16 UTC - LACE REPAIR PRODUCED A DIFFERENT SELECTION, which is the point of the exercise. On the
  stratified split the 36-trial sweep selects sigma=0.03, form=both, shrinkage=ledoit_wolf
  (val agg 0.2949), where the inherited historical selections are sigma=0.1/both/1e-3 (seed 0)
  and sigma=0.03/both/none (seed 1). ⚠ Note for the report: the repaired selection lands on
  ledoit_wolf, whose |diag-1| is ~0.145 and therefore NOT whitening-exact by the pinned 0.10
  tolerance, so R27.3's exactness gate fires on the repaired leg. My module records
  selected_is_whitening_exact per seed; that flag must be printed beside the repaired number
  rather than left implicit, because "the repair" is not automatically "the better-licensed
  configuration".

## Memory pressure on the box, and a plan change because of it

- 2026-09-19 19:22 UTC - ⚠ THE HARNESS KILLED TWO OF MY BACKGROUND WAITERS because the SYSTEM was critically low
  on memory. Neither was a failure of the waiter and neither is being restarted on my own
  initiative. What was stopped: the "wait for all rival stages" watcher and the "robust refit"
  watcher. Both were sleeping shell loops consuming almost nothing; they were reaped because the
  BOX was short, not because they were expensive. No compute was lost - all four of my python
  processes survived and are still working.
- 2026-09-19 19:22 UTC - Measured at the time: 4.2 GB free of 63.4 GB. My four processes held ~16 GB
  (rivals_s0 4.3, rivals_s1 4.2, rivals_s2 5.2, refit_pool 2.3) and a single hub/unseen-valid
  scoring process held 10.9 GB. Free memory has since recovered to 18.7 GB and all four of mine
  are healthy.
- 2026-09-19 19:22 UTC - ⛔ PLAN CHANGE, made because of this rather than after it bites: the chained per-seed
  re-score is DISARMED. Pipelining the rescue to start as each seed's rivals finished was a good
  idea under normal conditions, but it would have launched three MORE full cell-harness processes
  - each loading a store, a decoder, a judge and its own feature gathers - into a box that had
  4.2 GB free. Verified nothing fired: no chain.log, no rescore output files, and the waiters
  were already gone. The launcher script is moved aside so it cannot re-trigger.
  The re-score will instead be run SEQUENTIALLY, one seed at a time, once the rival runs have
  finished and released their memory. Same total work, a third of the peak. It is the last
  acceptance item and it is slower this way; that is the right trade against putting another
  agent's scored run at risk of an OOM kill.
- 2026-09-19 19:22 UTC - Worth recording for the programme: this is the third memory-shaped event today (my refit
  crash, hub/unseen-valid's 8.6 GB tensor, now a system-level reap). Five agents on 64 GB is
  genuinely tight, and the failure mode is not gradual - a process dies with no traceback. Any
  spoke holding a full-scope gather plus device copies should assume it is the one that dies.

- 2026-09-19 19:25 UTC - ⛔ MY OWN BUG, caught by the monitor: the pool refit crashed with KeyError 'req_B' after
  completing the full 18-minute cont_mlp fit on seed 0. Cause: fit_family() ends by producing
  features for the B and MS request sets, but the refit deliberately hands it a context with
  NEITHER - because producing B/MS features from a rival fitted on another hub's pool is exactly
  the population mistake this whole rival set is careful about. The absence was a design choice
  and the code treated it as an error.
  Fixed so the design choice is expressed rather than tripped over: fit_family now produces
  features only for the request sets the context actually carries, with the reason in a comment
  so nobody "fixes" it back by adding req_B to the mini context.
  ⭐ And I dry-ran both refit code paths on a 3,000-row slice before relaunching, rather than
  spending another 18 minutes to find the next line that fails. That is the lesson from this one:
  a long fit at the end of an untested path is a bad bet, and the path was untested because I
  wrote the mini context specifically to omit things.
  ⚠ The three main rival runs were untouched by this - all three are mid-LACE and healthy.

- 2026-09-19 19:47 UTC - ⭐ THE IMPLEMENTATION-INDEPENDENCE CHECK AGREES, on seed 0. My independent build of
  hub/unseen-valid's protocol - written from its published spec after reading its code, not by
  importing it - reaches final minibatch MSE 0.03532 at 118,153 params on 131,072 pool rows.
  Its own in-harness number at the same budget on the same seed is 0.0341. A 3.6% relative gap
  on a statistic that is a SINGLE minibatch of 512 samples, so well inside what the RNG alone
  would produce. Two independent implementations of the same protocol at the same budget on the
  same pool and the same generation-2 cache land on the same number.
  ⚠ Stated with its limits: this is one seed, and "final minibatch MSE" is a noisy one-batch
  statistic rather than a held-out evaluation. It establishes that the two implementations
  TRAIN THE SAME WAY. It does not by itself establish anything about either one's performance on
  the hub's 70 corners - that is the hub's scored measurement to make, not mine.
  My independent width calculation had already reproduced theirs exactly (h=55, 118,153 params),
  which is a separate and cleaner agreement because it is arithmetic rather than stochastic.

## EXTERNAL: my budget finding replicated on a second population, and it is bigger there

⚠ Everything in this section is hub/read's measurement on hub/read's population with hub/read's
judge. It is recorded as independent corroboration and attributed as such. ⛔ None of it is mixed
into my own table, and no number below is mine.

- 2026-09-19 19:48 UTC - hub/read scored my rivals on its 49 n-way and 60 pair corners (seed 0, generation 2,
  paired, its own judge) and reports that mlp_180k BEATS ITS ARM on both target rows:
      row 4:  tensor_addr 40.837  vs  mlp_180k 53.469   paired delta -12.633  [-17.327, -7.837]  sign p 6.2e-05
      row 3:  tensor_addr 16.883  vs  mlp_180k 25.683   paired delta  -8.800  [-11.033, -6.467]  sign p 7.7e-10
  Judge accuracy on the 49: mlp_180k 0.7426, tensor_addr 0.5672, span_head 0.5513, mlp 0.5173,
  table 0.1550. Its hub is falsified, by a rival it asked me to build for exactly that purpose.
- 2026-09-19 19:48 UTC - ⭐⭐ MY BUDGET FINDING REPLICATES ON A COMPLETELY INDEPENDENT CORNER SET, WITH A DIFFERENT
  INSTRUMENT, AND IS LARGER THERE. Same architecture, same 1.0x -> 1.51x parameter step:
      my corner B, my judge:        +0.033 balanced accuracy, +12.7 distinct factor tuples
      its 49 n-way corners, its judge: 37.245 -> 53.469 novel tuples/corner, i.e. +16.2
  Two populations, two instruments, same direction. On its scoreboard the parameter step buys
  the generic learner MORE than perfecting its dial could: it measured its own read ceiling at
  +9.08 tuples/corner (deployed 25.458 -> oracle address 34.542 on val), and the budget step is
  +16.2. That is a strong independent confirmation that the 118,784 matched budget is a real
  constraint on the generic-learner family and not an artefact of my corner.
- 2026-09-19 19:48 UTC - ⛔⛔ A CAUTION ABOUT MY OWN HEADLINE TABLE THAT I WOULD NOT HAVE FOUND ALONE, and it is
  the most important thing in this exchange for my report. The RIVAL ORDERING IS
  POPULATION-DEPENDENT, severely:
      on MY corner B:              table 0.9165   mlp_180k 0.9298   gap 0.013
      on ITS 49 n-way corners:     table 0.1550   mlp_180k 0.7426   gap 0.588
  The prototype table is NEARLY COMPETITIVE on my corner and COLLAPSES on its n-way conjunctions.
  My B is a single Montero cell - 72 rows, one shape value, retained values throughout - and an
  additive per-value-mean composition handles it well. Its corners are n-way conjunctions, which
  the additive rule cannot represent. Anyone reading my table as "the prototype table is a strong
  rival" would be generalising from the easiest compositional population in the programme. I am
  putting this in the report beside my own numbers, with its numbers attributed.
- 2026-09-19 19:48 UTC - Its second point is one I should carry too: had it scored only the parameter-MATCHED mlp
  it would have measured +3.592 [-0.816, +8.103] over "the MLP", read that as a win, and
  published a wrong conclusion - while the store alone (131,072 params) put its arm permanently
  outside the matched budget. The mlp_180k rival existed only because it asked and I said yes to
  a side request. ⭐ The lesson for this spoke: a rival set matched to ONE budget is not a
  comparison set, it is a comparison at one budget, and the second budget is what made the first
  interpretable.
- 2026-09-19 19:48 UTC - My caveat survives contact and it is printing it: this is one architecture family over
  one budget ratio, and 1.51x of an MLP is not 1.51x of a store. Its ablation against the
  incumbent read is +1.143 [-2.898, +5.061] - nothing - so the budget effect is specific to the
  generic learner rather than a general statement about parameters.

## CORRECTING TWO ERRORS OF MINE, both caught by hub/read

- 2026-09-19 19:50 UTC - ⛔ ERROR 1, mine: I described mlp_180k to hub/read as a "budget-MISMATCHED generic
  learner" and offered that as a softening of its falsification. That is backwards and the
  arithmetic says so. mlp_180k is 178,952 parameters against its arm's 179,968 - MATCHED to
  -0.57%. It is the MATCHED rival for that hub. The mismatched one is mlp at 117,878, and that
  is the comparison its +3.592 [-0.816, +8.103] came from.
  I conflated two different reference budgets: mlp_180k is 1.51x the PROTOTYPE TABLE's 118,784
  (which is what my own table is matched to) and simultaneously 0.994x hub/read's ARM. Both are
  true and I quoted the first while reasoning about the second. It declined the softening, which
  was correct of it, and the honest statement is the harsher one: at a matched parameter budget
  a plain one-hot MLP with no store, no atoms, no landscape and no address book beats the
  addressed read by +12.6 novel tuples/corner on row 4 and +8.8 on row 3, intervals nowhere near
  zero.
  ⭐ The report must carry BOTH framings for mlp_180k or it will mislead in one direction or the
  other: mismatched to the table's budget, matched to that hub's arm.
- 2026-09-19 19:50 UTC - ⛔ ERROR 2, mine: I attributed a row-2 `table` number of 0.1581 to hub/read. It is
  hub/unseen-valid's, from a different scoreboard with a second judge, and row 2 is not
  hub/read's target at all. I nearly carried a number under the wrong hub's name into my own
  report. Withdrawn from anything addressed to hub/read; the 0.1581 belongs where it was first
  reported and only hub/unseen-valid has the standing to interpret it.
  ⚠ The speculation I attached to it - that the additive composition rule rather than the fit
  population might be what fails - is NOT retracted, but it is now correctly addressed to
  hub/unseen-valid and it is a question, not a finding.
- 2026-09-19 19:50 UTC - ⭐ KEPT AND SHARPENED, from its reply: its ablation against the incumbent read is
  +1.143 [-2.898, +5.061], indistinguishable from zero. So BOTH addressed reads - its variant and
  the incumbent - sit below the matched generic learner on those corners. The finding is about
  the addressed-read FAMILY on that population, not about its particular variant. That is a
  stronger and more useful statement than the one I offered it, and it is its own.

## The pool question is settled by subtraction, not by a scored run

- 2026-09-19 19:54 UTC - hub/unseen-valid settled it without scoring anything: it loaded my refitted table and its
  own knn_proto and SUBTRACTED them. Same pool -> the same object.
      n_rows_fit            131,072 both
      n_per_value[object_hue] identical, including the 0 at index 5
      mu_global             max abs delta 2.42e-05
      per-factor prototypes 4.89e-06 to 2.42e-05, on features of order 1
  Two objects agreeing to 1e-5 cannot score 0.1581 and 0.2558 on the same judge. So the original
  gap IS the fit population, my refitted table will land at its knn_proto's value, and that
  agreement is the CONFIRMATION rather than a failed refit. It recorded that prediction in its
  own LOG before measuring, so the null cannot later be misread.
- 2026-09-19 19:54 UTC - ⛔ MY HYPOTHESIS IS EXCLUDED, and I accept the exclusion. I had suggested the additive
  per-value-mean composition rule, rather than the fit population, might be what fails. For the
  row-2 gap that is now ruled out by an identity argument, which is stronger than any score
  comparison could have been. ⚠ What remains open, and is a DIFFERENT question I should not
  conflate with the one just closed: whether the additive rule fails on hub/read's n-way
  CONJUNCTIONS. That was never the 0.1581 gap and it is not settled by this.
- 2026-09-19 19:54 UTC - ⛔ A SECOND OVER-CLAIM OF MINE, corrected by them and accepted: I called hub/read's 0.1550
  "weak evidence" for the composition-rule story. It is not weak evidence, it is NOT EVIDENCE -
  different estimand (49 n-way corners vs 70 row-2 corners), different judge, different requests.
  Its closeness to 0.1581 is a coincidence of scale, not a measurement of the same quantity.
  Neither of us will cite it. That is the second time today I have let two numbers being NEAR
  EACH OTHER stand in for their being ABOUT THE SAME THING; the first was the whitening
  cross-check, where I caught it myself and flagged it. Worth naming as a pattern rather than as
  two incidents.
- 2026-09-19 19:54 UTC - ⭐ A POSITIVE FOR MY OWN REPORT, from the same comparison: my prototype-table
  implementation and hub/unseen-valid's independently built knn_proto produce the SAME FITTED
  OBJECT to 1e-5 on the same pool. That is a cross-implementation validation of the incumbent
  rival that neither of us set out to run - two codebases, same mechanism, same bytes, same
  numbers. It belongs in the report as evidence the table rival is correctly implemented.
- 2026-09-19 19:54 UTC - ⭐ THE LESSON, which is cheap and general: COMPARE THE OBJECTS BEFORE SCORING THEM. A
  pickle.load and a max(abs(A-B)) answered in seconds, at zero decodes and zero scored-run
  budget, a question we were both about to spend a scored run on. It freed that hub's run 2/3 for
  the cont_mlp legs where the question is genuinely open. I will apply this to my own re-score:
  where two arms should be identical by construction, assert the identity rather than measure it.

## LACE, seed 2: the repair changes NOTHING on this seed, and that is the finding

- 2026-09-19 20:09 UTC - First complete LACE result, seed 2, both legs. ⭐ THE REPAIRED SELECTION IS IDENTICAL TO
  THE INHERITED ONE. Re-running the full 36-trial selection on the STRATIFIED split lands on
  exactly the configuration the historical_688128 manifest carries: sigma=0.03, form=both,
  shrinkage=ledoit_wolf. The declared second rule (validation `shape` rather than aggregate)
  agrees with it too. So delta_repaired_minus_historical = 0.0 EXACTLY, and both legs produce
  bit-identical numbers because they are the same configuration run twice.
      published (gen 1) probe reference   0.6582
      historical selection, probe         0.6511   regressed? False
      repaired selection, probe           0.6511   regressed? False
  Measured 0.6511 against a published 0.6582 is -0.0071, well inside the 0.05 tolerance.
- 2026-09-19 20:09 UTC - ⚠ SO THE REGRESSION MY TASK DESCRIBES IS NOT PRESENT ON SEED 2. The task states the
  harness flags `lace` as regressed at 0.6136 measured against 0.7077 published - and 0.7077 is
  the SEED 0 entry of lace_ode_probe_reference_per_seed. On seed 2 the reference is 0.6582, the
  measurement is 0.6511, and nothing is regressed. ⛔ I am NOT concluding the regression is
  seed-0-only until seeds 0 and 1 report; that is the obvious hypothesis and it is exactly the
  kind of thing that is wrong half the time. But it already means "LACE is regressed" cannot be
  stated as a property of the arm - it is at most a property of one seed, and the repair is a
  no-op on at least one other.
- 2026-09-19 20:09 UTC - ⭐ LACE'S BEST LEG IS NOT ITS PUBLISHED ONE, and never-weaken-a-rival means saying so.
  On seed 2, by the judge's in-corner balanced accuracy:
      ld_corrected  0.6638   distinct 72/72
      ode           0.6089   distinct 72/72     <- the published composition row
      ld_published  0.5009   distinct 44/72     <- chance
  The Langevin leg at the CORRECTED temperature beats the published ODE row by +0.055. And
  ld_published sits at 0.5009, i.e. exactly chance, which is the ~200x-too-cold sampler
  temperature gen-lace-spec F2 derived and the config documents. Quoting only the `ode` row would
  understate LACE by 0.055 on this seed; quoting only ld_published would understate it by 0.16.
  The report prints all three with the primary sampler named.
- 2026-09-19 20:09 UTC - All three samplers render 72 distinct factor tuples out of 72 except ld_published at 44.
  That is the distinct-output column doing its job: the chance-level arm is also the degenerate
  one, and an accuracy alone would not have shown that.

## ⭐⭐ THE LACE REPAIR DOES NOT FIX THE REGRESSION. My task's premise is falsified.

- 2026-09-19 20:14 UTC - Seed 0, both legs, on generation 2:
      published (generation-1) ode probe reference   0.7077
      HISTORICAL selection (sigma=0.1, both, 1e-3)   0.6313   regressed = True
      REPAIRED  selection (sigma=0.1, both, ledoit_wolf) 0.6231   regressed = True
      delta repaired minus historical                -0.0082
  ⛔ The repair does NOT recover the published number. It lands 0.0846 below it, and it is
  SLIGHTLY WORSE than the inherited selection it was supposed to fix. On seed 2 the repair
  selected exactly the inherited configuration, so it was a no-op there. Two seeds, no repair.
- 2026-09-19 20:14 UTC - ⛔ THE PREMISE I WAS GIVEN IS THEREFORE WRONG, and this is the finding the Head needs.
  My task states that lace_selection_source points at "the broken-split historical selection" and
  that this is why the harness flags lace as regressed (0.6136 vs 0.7077). I re-ran the entire
  36-trial selection on the stratified split - the same grid, the same short schedule, the same
  criterion - and the resulting configuration scores no better. So WHATEVER causes the
  0.7077 -> 0.63 gap, it is not that the selection was made on the wrong split. The repair was
  the right thing to build and the right thing to test; the test says the hypothesis is false.
  ⭐ I am reporting the repair as BUILT, RUN and NEGATIVE rather than quietly reporting the
  historical number, because a repair that does not repair is a result about the instrument.
- 2026-09-19 20:14 UTC - ⚠ A CONFOUND I MUST NOT LET STAND AS AN EXPLANATION WITHOUT SAYING SO: 0.7077 is a
  GENERATION-1 constant, and both my numbers are GENERATION 2. The harness's regression_check
  compares a gen-2 measurement against a gen-1 reference, which is precisely the mixing this
  whole spoke forbids. So the "regressed = True" flags above are NOT clean evidence of a
  regression - they are a cross-generation comparison shown for orientation. ⛔ The honest
  statement is narrower than either "LACE is regressed" or "LACE is fine": the repair changes
  nothing (that part IS within-generation and clean, -0.0082), and the level cannot be compared
  to 0.7077 until an arm is re-scored on generation 2, which is what my rescore stage is for.
- 2026-09-19 20:14 UTC - ⭐ LACE'S BEST PROBE NUMBER ON SEED 0 IS 0.7601, ABOVE the published 0.7077 - and it comes
  from ld_published, not from ode:
      historical  ld_published  probe 0.7601  judge 0.6477  distinct 58/72
      historical  ld_corrected  probe 0.6301  judge 0.6553  distinct 71/72
      historical  ode           probe 0.6313  judge 0.6143  distinct 71/72
  So on the probe statistic the arm's best leg CLEARS its own published reference, while the leg
  the reference names does not. Reporting only `ode` understates LACE by 0.129 on the probe here.
  ⚠ Note the judge and the probe DISAGREE about which leg is best (judge prefers ld_corrected,
  probe prefers ld_published) and ld_published is the least diverse at 58/72 distinct. I am not
  adjudicating that; I am printing all three legs on both instruments with the distinct counts,
  which is the only honest presentation.

## The 573k falsifier replicate agrees with hub/unseen-valid

- 2026-09-19 20:14 UTC - cont_mlp_notune_572k, independently built at its protocol, on its pool, generation 2:
      seed 0: 573,440 params (100.1%), final minibatch MSE 0.01094   its reference 0.0108
      seed 1: 573,440 params (100.1%), final minibatch MSE 0.01209
  Seed 0 agrees to 1.3% relative on a single-minibatch statistic. Its falsifier of record now has
  an independent implementation on the same pool and the same cache. Seed 2 is fitting now.

## LACE COMPLETE, all three seeds. The repair is never positive, and the regression is one seed.

      seed  published(gen1)  historical(gen2)  repaired(gen2)  repair delta  hist regressed?
        0       0.7077           0.6313           0.6231         -0.0082          True
        1       0.6732           0.6353           0.6278         -0.0075          False
        2       0.6582           0.6511           0.6511          0.0000          False
      mean      0.6797           0.6393           0.6340         -0.0052
      sd        0.0254           0.0105           0.0150

- 2026-09-19 20:18 UTC - ⭐ CONCLUSION 1: THE REPAIR IS NEVER POSITIVE. Per seed it is -0.0082, -0.0075 and exactly
  0.0000; mean -0.0052. On seed 2 it re-selects the inherited configuration exactly, so it cannot
  differ. On the other two it selects a different configuration and scores slightly WORSE. There
  is no seed on which re-running the 36-trial selection on the stratified split helps. This is a
  WITHIN-GENERATION comparison - both legs on generation 2, same rows, same code - so it is clean
  and does not inherit the generation caveat.
  ⛔ The premise I was handed is falsified: the inherited selection is not what is wrong with LACE.
- 2026-09-19 20:18 UTC - ⭐ CONCLUSION 2: "LACE IS REGRESSED" IS A STATEMENT ABOUT ONE SEED, and probably about the
  REFERENCE rather than the measurement. The gap to the published constant is 0.0764, 0.0379 and
  0.0071 - monotone in the published value itself. My three generation-2 measurements are tightly
  clustered (sd 0.0105, range 0.0198) while the three published generation-1 references are
  spread nearly three times as wide (sd 0.0254, range 0.0495), with seed 0 the high outlier at
  0.7077 against a 0.6797 mean.
  So the pattern is not "my seed 0 fell" but "the published seed 0 is high relative to its own
  siblings, and my seeds agree with each other". ⚠ Held with its caveat: published is generation
  1, mine is generation 2, and I will not assert a cause across that boundary. What I can say
  without crossing it is that the measurements are more self-consistent than the references.
- 2026-09-19 20:18 UTC - ⛔ NOT-RUN, declared rather than silently skipped: R27.3's second route - the general ODE
  form WITH the prior score in the drift - remains NOT RUN. exp_gen_cell.lace_selection() records
  it as needing a new drift term in sample_ode that nobody has written, and I did not write it
  either. My repair changed WHICH configuration is selected, not the sampler. So the possibility
  that the published 0.7077 came from a differently-drifted sampler is untested, and it is the
  most obvious remaining candidate for the seed-0 gap.
- 2026-09-19 20:18 UTC - ⚠ Every repaired selection lands on a NON-whitening-exact shrinkage (ledoit_wolf, 1e-3,
  ledoit_wolf), so R27.3's exactness gate fires on all three repaired legs. Only "none" clears
  the 0.10 tolerance on this cache. The repair therefore trades a licence for a validation score
  and loses on both, which is a further reason not to prefer it.

## Pool refit COMPLETE: four objects x three seeds, delivered to hub/unseen-valid

- 2026-09-19 20:32 UTC - /c/chlu/outputs/spoke-a-rivals/rivals_vhpool/seed_{0,1,2}/ now carries table,
  cont_mlp (tuned, budget-matched), cont_mlp_notune (118k at its protocol) and
  cont_mlp_notune_572k (573k at its protocol) on all three seeds, with refit_manifest.json.
  Pool verified independently on every launch: 131,072 rows, object_hue[5] and shape[0] both
  absent, fit_population=value_holdout. No dashes, so the replicate covers 3/3 seeds.
      cont_mlp_notune_572k   seed 0 0.01094   seed 1 0.01209   seed 2 0.012682
      its references         seed 0 0.0108    -                seed 2 0.01306
      cont_mlp_notune (118k) seed 0 0.03532   seed 1 0.03515   seed 2 0.04333
  Both of its 573k references matched within 3% (+1.3% and -2.9%) on a single-minibatch
  statistic. Claim kept narrow and repeated to it: the implementations TRAIN THE SAME WAY.
  Nothing about its corners, which are its scored measurement.
- 2026-09-19 20:32 UTC - ⚠ Flagged to it rather than left for it to trip over: the 118k notune is noticeably
  seed-sensitive (0.0353/0.0352/0.0433) while the 573k is stable (0.0109-0.0127). If its own
  118k numbers show the same seed-2 spread, that is the small model being genuinely
  seed-sensitive rather than an implementation difference - and a 118k disagreement should not
  be read as one without checking.
- 2026-09-19 20:32 UTC - ⛔ None of the vhpool objects is scored on my corner B, and none will be. They belong to
  that hub's population; producing a B number for them is the population error this whole
  exchange was about. The stage refuses to compute one by construction - its context carries no
  B/MS request set at all.

## PoE-EBM run 1, seed 0: the byte multiple, and where this arm's weakness actually is

- 2026-09-19 20:35 UTC - ⭐ THE BYTE MULTIPLE, which the task requires printed beside every row of this arm:
  3,556,665 parameters = 14,226,660 B = **27.1x** the 524,288 B deployed store. This arm runs at
  its OWN budget and it is not a small overrun - it is twenty-seven times the store's bytes. Any
  number it produces is not comparable in level to a byte-held arm, and the multiple goes on the
  row, not in a footnote.
- 2026-09-19 20:35 UTC - ⚠ AN OBSERVATION ABOUT WHERE THIS ARM FAILS, which the aggregate alone would not show.
  Its six independent experts are essentially perfect classifiers on their own validation data -
  per-factor accuracy [1.0, 1.0, 1.0, 0.998, 0.9941, 0.998], mean 0.9984. But the SAMPLER's
  selected configuration reaches only val_aggregate 0.1953 (val_compositional 0.3477) on the
  decoded images. So the energy model knows the factors and the SGLD sampler cannot produce
  feature vectors that decode to them. The weakness is in the sampling, not the experts, and
  reporting only the arm's final accuracy would attribute it to the wrong component.
  ⛔ This is a description of a measured gap inside the rival, not a criticism of the method: the
  36-trial budget tuned the sampler as far as its own grid allows, and the grid is the one the
  pinned config specifies.
- 2026-09-19 20:35 UTC - Selected sampler configuration on seed 0 run 1: step_size=0.03, prior_sigma=2.0,
  n_steps=200, clip=True. Note it selected the LARGEST step size and the LARGEST step count in
  the grid, and the non-negativity clip - i.e. it went to the edge of its tuning grid on two
  axes. ⚠ An arm selecting at a grid boundary is an arm whose grid may be too small; that is a
  caveat about the budget the rival was given, and it belongs beside its number.

## An audit prompted by another hub's warning, and one defect it found

- 2026-09-19 20:49 UTC - hub/unseen-valid reported that ITS Monitor notifications have produced NINE spurious
  values against its on-disk artifacts, the worst on the very diagnostic it was using to decide
  whether its headline held - a notification said linear_span_ARM sd=10.674 / 1204 tuples /
  25.63 per corner where the artifact says 5.479 / 4028 / 57.70. Those imply OPPOSITE verdicts.
  Only its artifacts-only rule stopped it mis-calling its own result.
- 2026-09-19 20:49 UTC - ⛔ I HAVE BEEN QUOTING SOME NUMBERS FROM STREAMED OUTPUT, so I audited every one of them
  against its artifact rather than assuming my stream was clean. Re-read from JSON:
      PoE byte multiple    logged 27.1x     artifact 27.1352x  (3,556,665 params, 14,226,660 B)
      PoE run 1 judge      logged 0.5417    artifact 0.5417
      PoE run 2 judge      logged 0.5701    artifact 0.5701
      573k MSE seed 0      logged 0.01094   artifact 0.01094
      573k MSE seed 1      logged 0.01209   artifact 0.01209
      573k MSE seed 2      logged 0.012682  artifact 0.01268 (same value, 5 dp in the log)
  ⭐ Every number matches. No spurious value has entered my log or any message I have sent. The
  structured results (table, learners, LACE, deletion, caches) were read from JSON by script from
  the start, so only these few were ever at risk.
- 2026-09-19 20:49 UTC - ⛔ BUT THE AUDIT FOUND A REAL DEFECT, in a place I had already pointed a hub at:
  rivals_vhpool/refit_manifest.json documented ONLY SEED 2. The refit ran in two phases as two
  separate processes, each writing that file, so phase 2 silently deleted phase 1's record while
  twelve perfectly valid checkpoints sat on disk beside it. I had told hub/unseen-valid the
  manifest was there.
  Fixed twice over: (a) the manifest is rebuilt from the CHECKPOINTS - structural facts from each
  checkpoint's own metadata, which is authoritative, and the final-minibatch MSE recovered from
  the run log and LABELLED with that source because it is a training diagnostic the checkpoint
  does not store; (b) the stage now MERGES per seed instead of overwriting, so a phased re-run
  cannot do this again.
  ⭐ The lesson is the same shape as this morning's lstm population gap: the artifact a consumer
  is pointed at must be checked, not assumed, and "the checkpoints are all there" is not the same
  claim as "the manifest describes them".

- 2026-09-19 20:52 UTC - ⛔ A THIRD DEFECT IN WHAT I HAD ALREADY PUBLISHED, reported by hub/unseen-valid:
  load_rival refused cont_mlp_notune_572k - the object its PREREG-2 names as its PRIMARY leg -
  while the checkpoint sat valid on disk. Cause: the loader gated on a hand-maintained tuple of
  family names, and a budget-suffixed name was never added to it. The fit was fine; the dispatch
  was wrong.
  Fixed as a mechanism rather than as an entry, because the list was the wrong idea:
    (a) loadability is now decided by what the CHECKPOINT DECLARES (in_kind), not by name
        membership - a protocol-matched build exists at whatever budget it was asked for and a
        name list will always lag the objects;
    (b) available(seed) now ENUMERATES FROM DISK instead of iterating a fixed list. That was the
        more dangerous half: it would have reported the 573k as ABSENT rather than raising, and
        an absence is the one failure mode a consumer does not think to question.
  Verified on all three seeds: 4096 requests, (4096,2048) float32, oor 0, object_hue[5]==0 and
  shape[0]==0, off-grid accepted, partial correctly refused.
- 2026-09-19 20:52 UTC - ⭐ The hub declined to hand-load the .eqx to route around my refusal, and said why: if it
  reconstructed the architecture and got the input normalisation or layer count subtly wrong, the
  number would look like an independent replication while actually being its own code twice. That
  is the right instinct and it was vindicated - the refusal was pointing at a real defect. Worth
  recording as a norm: a rival API's refusal is a signal to report back, not an obstacle to work
  around, precisely because the failure it prevents is invisible in the output.
- 2026-09-19 20:52 UTC - Manifest writer now MERGES per seed (the earlier patch silently failed to apply - I
  asserted on the old text, the assertion fired, and I caught it because the shell reported the
  AssertionError rather than because I re-read the file). Re-applied and import-checked.

- 2026-09-19 20:53 UTC - ⭐ AN UNPLANNED CROSS-CHECK THAT VALIDATES THE 2-D CONTEXT FIX. The re-score's own
  ebm_compose arm on seed 0 - run by the SHIPPED exp_gen_cell through its own gather_features -
  reports IMAGE-space in-corner shape balacc 0.5417, distinct 72 vectors / 72 factor-tuples of
  72. My separate PoE stage's run 1 on seed 0, which calls run_ebm_arm with the same seed and
  therefore the same key stream, reported 0.5417. Identical to four decimal places.
  That is two independent context constructions - the shipped one and mine - driving the same
  frozen arm to the same number, and it is exactly the check that would have failed before I
  fixed gather_gen2 to return (N, D). ⛔ It is a validation of my CONTEXT, not of the arm: the
  arm is the same code in both cases. But it retires the worry that my rival numbers were
  computed in a subtly different setting from the cell's.
- 2026-09-19 20:53 UTC - Also note the distinct-output column on that row: 72/72 factor tuples for ebm_compose.
  The arm is fully diverse and still only reads 0.5417 - so its weakness is genuinely
  compositional accuracy, NOT the degenerate-output failure mode. That is the opposite diagnosis
  from LACE's ld_published leg (44/72 distinct at chance), and the two would be
  indistinguishable from the accuracy alone.

## ⭐⭐⭐ GENERATION 1 FAILS ITS OWN ENCODER'S RE-ENCODE CHECK. ALL THREE SEEDS.

- 2026-09-19 20:56 UTC - The re-score crashed on seed 1 with exp_gen_store._assert_cache_reencodes firing on MY
  generation-2 cache (3 ULP, 3 of 131,072 elements). Rather than work around my own gate failing,
  I ran the SAME check against BOTH generations on all three seeds, each with the encoder stored
  beside that cache. Result:
      seed   generation 1                              generation 2
        0    FAIL  max 51 ULP, 1280/131072 (0.98%)     PASS  max 1.00 ULP, 1/131072 (7.6e-06)
        1    FAIL  max 81 ULP,  640/131072 (0.49%)     FAIL  max 3.00 ULP, 3/131072 (2.3e-05)
        2    FAIL  max 59 ULP, 1231/131072 (0.94%)     PASS  max 1.00 ULP, 2/131072 (1.5e-05)
- 2026-09-19 20:56 UTC - ⛔⛔ THE GENERATION-1 CACHES ARE NOT REPRODUCED BY THE ENCODER SHIPPED BESIDE THEM, on
  this box, by a factor of 51 to 81 ULP on roughly half to one percent of elements. That is not a
  marginal miss - the check's per-element bound is ONE ULP and its fraction bound is 1%, and
  generation 1 sits at 50-80x the first and right at the edge of the second on every seed.
- 2026-09-19 20:56 UTC - ⭐ AND THE SIGNATURE IS THE ONE MY TASK PREDICTED, INDEPENDENTLY MEASURED. The task
  states "CPU and GPU features agree to ~53 ULP on ~1% of elements". I measure generation 1 at
  51, 81 and 59 ULP on 0.98%, 0.49% and 0.94% of elements. I did not go looking for that number;
  it fell out of running the harness's own gate against the shipped cache. The most economical
  explanation is that generation 1 was produced on different hardware from the encoder's home
  here - which is exactly the hazard generation 2 was commissioned to remove, and it was real.
- 2026-09-19 20:56 UTC - ⭐ GENERATION 2 IS ONE TO TWO ORDERS OF MAGNITUDE CLOSER. 1 ULP on 1-2 elements out of
  131,072 on two seeds, 3 ULP on 3 elements on the third, against generation 1's 51-81 ULP on
  640-1280 elements. The residual is batch-shape sensitivity - the gate re-encodes 64 rows
  standalone while the cache was written in 8192-row blocks at eval_batch 256, so the
  accumulation order differs - and it is at the floor that reordering allows on this box.
- 2026-09-19 20:56 UTC - ⛔ THE CONSEQUENCE FOR THE PROGRAMME, and it is larger than my spoke: A RE-SCORE ON
  GENERATION 1 WAS NEVER POSSIBLE. The shipped cell calls this gate at the top of every run_seed.
  Run against generation 1 it would have refused on all three seeds. Any historical run that
  proceeded did so because the gate only fires ENFORCED when the cache is EXTENDED, and a
  full-scope cache with no misses never triggers it. So the check existed, was correct, and was
  structurally unable to see the thing it was built to catch.
- 2026-09-19 20:56 UTC - ⚠ What this does NOT say, stated because the temptation is obvious: it does not
  invalidate any published generation-1 number. Those numbers were computed on those features,
  and the features are what they are. It says the generation-1 cache cannot be reproduced from
  its own encoder here, so it cannot be extended, audited or re-scored on this box - and that a
  generation-2 number and a generation-1 number are measurements on different feature sets, which
  is what I have said from the start and can now demonstrate rather than assert.

- 2026-09-19 20:59 UTC - The seed-1 gate failure is NOT sampling luck and I checked rather than assumed: the gate
  takes cache_rows[:64] deterministically - no randomness at all. So seed 1's 3-ULP miss is
  reproducible, and re-running it hoping for a pass would be meaningless as well as dishonest.
  The cause is batch shape: the gate encodes those 64 rows standalone (one batch of 64) while the
  cache wrote them inside an 8192-row block at eval_batch 256, so the accumulation order differs.
  Seeds 0 and 2 land at 1 ULP under the same reordering and seed 1 at 3; the difference is which
  encoder, not which cache.
  ⛔ I considered rebuilding seed 1 at eval_batch 64 so the gate's probe would reproduce
  bit-identically, and rejected it: the gate only ever looks at the first 64 rows, so matching my
  build to its probe would be tuning the cache to pass its own check. The number would then mean
  nothing. Reported instead.
- 2026-09-19 20:59 UTC - ⛔ MY OWN ENGINEERING DEFECT, fixed: seed 1's gate failure raised out of run_seed and
  killed the whole re-score process, so seeds 1 AND 2 were lost to a refusal that concerned only
  seed 1. stage_rescore now catches a per-seed failure, writes seed_N/NOT_RUN.json with the
  reason, and continues. A gate firing is a RESULT about that seed; letting it abort the others
  turns one measurement into three missing ones.
  compare_generations now also reports seeds_compared, seeds_NOT_RUN and a coverage_caveat, and
  its verdict says "on only N of 3 seeds" rather than reading as though every seed contributed.

## A recommendation for whoever owns exp_gen_store.py (NOT implemented here - read-only)

- 2026-09-19 21:09 UTC - The re-encode gate has TWO bounds - per-element (<=1 ULP) and fraction (<=1%) - and my
  measurements show they discriminate very differently between the two failure modes it faces:
      failure mode                          per-element ULP     differing fraction
      different DEVICE (generation 1)       51-81  (50-80x over)   0.49-0.98%  (at the edge)
      batch-shape reorder (generation 2)    1-3    (1-3x over)     0.002%      (500x under)
  ⭐ The FRACTION bound separates them cleanly - three orders of magnitude apart - while the
  per-element ULP bound fires on both. So the gate as written refuses a same-device cache for
  reordering noise, and its genuinely discriminating statistic is the one that is not doing the
  refusing. Generation 2 seed 1 is refused at 3 elements of 131,072; generation 1 would be
  refused at 640-1280, for a completely different reason.
  ⛔ NOT implemented by me: exp_gen_store.py is on my read-only list and I did not touch it. This
  is a recommendation with the measurement attached, for its owner: consider whether the
  per-element bound should scale with the differing fraction, or whether a cache with no misses
  (full scope, nothing to extend) should be checked at all - the gate's own docstring says the
  hazard it exists to prevent is "a run that WRITES NEW ROWS into the cache's feature space", and
  a full-scope cache never writes one.
- 2026-09-19 21:09 UTC - ⛔ I also did NOT use the gate's own `will_extend=False` escape hatch to get seed 1
  through. It exists and takes a `reason`, and a full-scope cache genuinely cannot extend - so
  the argument would have been available and arguably correct. I declined because the call site
  is inside exp_gen_cell.run_seed, which is read-only and passes the default; reaching past it
  would mean running the scored cell in a configuration the cell did not choose. A declared
  NOT-RUN with the measurement is worth more than a number obtained by loosening someone else's
  gate from outside.
- 2026-09-19 21:09 UTC - PoE-EBM progress: seed 0 runs 0.5417 / 0.5701, seed 1 runs 0.5482 / 0.5595, all at 27.1x
  the store's bytes. Across four independent runs on two seeds the arm sits in 0.542-0.570, a
  spread of 0.028 - which is itself the answer to why the task demanded two runs: a single draw
  anywhere in that band would have looked like a point estimate.

## ⭐⭐ RE-SCORE ON GENERATION 2 COMPLETE. One arm trips the seed-spread rule.

      arm                     gen-1 mean  gen-1 sd  gen-2 mean  gen-2 sd  max|delta|  moved?
      clu                       0.6759     0.0132     0.6734     0.0176    0.0000     False
      knn_proto                 0.8817     0.0028     0.8803     0.0022    0.0000     False
      store_removed             0.5000     0.0000     0.5000     0.0000    0.0000     False
      ebm_compose               0.5148     0.0171     0.5215     0.0206    0.0069     False
      lace (ode)                0.6399     0.0025     0.6412     0.0140    0.0139     False
      lace/ld_corrected         0.6161     0.0151     0.6327     0.0037    0.0177     TRUE
      lace/ld_published         0.5547     0.0757     0.6436     0.1648    0.1181     False
      dynread (+3 legs)         0.5577     0.0372     NOT RUN    -         -          -

- 2026-09-19 21:25 UTC - ⛔ REPORTING THIS BEFORE ANY HUB SCORES, as my task requires: lace/ld_corrected moves by
  up to 0.0177 between generations against a generation-1 seed spread of 0.0151. The rule fired.
  Per seed the movement is 0.0177 (seed 0) and 0.0025 (seed 2), so it is one seed, marginal at
  1.17x the spread, on a COMPANION sampler leg rather than on LACE's published ode row.
- 2026-09-19 21:25 UTC - ⭐ THE READING THAT MATTERS MORE, and it is reassuring: THE THREE DETERMINISTIC ARMS DID
  NOT MOVE AT ALL. clu, knn_proto and store_removed are 0.0000 on every seed - not "small", not
  "within tolerance", EXACTLY zero to four decimals. Every arm that moves is a SAMPLER arm (the
  two LACE Langevin legs and the energy-based arm), and a stochastic arm differing between two
  independent runs is expected whether or not the cache changed. So the generation change is
  invisible to every arm that is a deterministic function of the features, and the one trip is a
  sampler leg edging past a small spread on one seed.
- 2026-09-19 21:25 UTC - ⚠ Held with its limits, three of them: (a) the comparison covers 2 of 3 seeds, because
  seed 1's generation-2 re-score is a declared NOT-RUN (the re-encode gate, logged above), so
  every gen-2 spread here is computed from two points; (b) the gen-1 column is a
  GENERATION-1 number and is shown as a separate column, never combined; (c) dynread and its
  three legs are NOT RUN on generation 2 - they were requested but did not produce rows in the
  seeds that completed, so they are a declared absence, not a zero.
- 2026-09-19 21:25 UTC - ⛔ A DEFECT IN MY OWN COMPARISON, caught before it misled anyone and worth recording in
  full because of what it would have said. The first version of this artifact reported
  "no published arm moves by more than its seed spread" with EVERY generation-1 cell reading
  NOT RUN - i.e. a reassuring verdict computed from ZERO generation-1 data. Cause: the shipped
  manifests nest one level deeper than I assumed
  (runs/cell/seed_N/seed_N/cell_manifest.json), so my loader found nothing and the comparison
  compared against an empty dict. Fixed: the loader tries both layouts and falls back to a glob,
  and compare_generations now emits "VACUOUS - no generation-1 manifest was found" instead of a
  clean bill of health when the reference side is missing.
  ⭐ The general lesson, and it is the sharpest one of the day: AN EMPTY COMPARISON IS NOT A
  PASSING COMPARISON, and a verdict function that cannot tell the difference will always fail
  in the reassuring direction. The vacuous version sat on disk for about fifteen minutes and no
  hub was pointed at it in that window.

- 2026-09-19 21:31 UTC - ⛔ LACE WAS PRINTING A BARE DASH IN THE PARAMETER COLUMN, which made the largest arm in
  the set look budget-free next to a PoE-EBM carrying a 27.1x label. The shipped
  exp_gen_cell.run_lace_arm records no parameter count, so nothing was available to print.
  Counted from the method's own structure instead:
      LACE classifier (LaceClassifier over dense_embedder_dims 2048->1408->768->128
        plus per-factor heads)                                     4,073,663 params
      LACE whitening: W (2048x2048) + W_inv (2048x2048) + mu       8,390,656 params
      TOTAL                                                       12,464,319 params
                                                                = 49,857,276 B = 95.1x the store
  ⭐ So LACE is 95.1x the 524,288 B store - more than THREE TIMES the PoE-EBM's 27.1x, and the
  single largest object in the rival set by a wide margin. That was invisible in my table until
  the dash was chased. Artifact: lace_byte_ledger.json.
  ⚠ Declared honestly: this is a STRUCTURAL count I computed from the implementation, not a
  measurement the arm reported, and the artifact says so. The dominant term is the pair of
  2048x2048 whitening matrices stored on gen_lace.Whitening - a property of whitening at D=2048,
  not a choice anyone made per-run.
- 2026-09-19 21:31 UTC - ⭐ THE RIVAL TABLE IS COMPLETE AND THE BYTE COLUMN NOW TELLS THE REAL STORY: the two arms
  that beat nothing are the two that are 27x and 95x over budget, and the incumbent prototype
  table at 0.91x the store budget leads the set at 0.9165. Reported as measured, in both
  directions.
- 2026-09-19 21:31 UTC - FINAL REPORT regenerated and appended as the last section of this LOG.md, and written
  separately to REPORT.md. The earlier appended copy was removed first so there is exactly one
  report block; the 138 dated bullets above it are untouched.

---

# REPORT - w4-spoke-a: the rival set, and cache generation 2

Generated 2026-09-19 21:31 UTC by reading the artifacts on disk. Every number below traces to a JSON file in this directory; none is carried in a variable from the run that produced it.

## Dial declaration, discharged

**Dial: the instrument and the comparison set.** This spoke ran no arm of ours and makes no claim about the mechanism.

**Falsified if** a rival could not be brought to matched parameters within +/-10 %, or if generation-2 re-encoding did not reproduce its own hashes on a second pass. Neither happened: every rival matched inside +/-1 %, and the determinism check passed. Both are evidenced below.

## The rules that travel with these numbers

1. **Generation 2 only.** Never quote a number here beside a generation-1 number, in either direction.
2. **Generation 1 is untouched** on disk and still hashed.
3. **Know the population.** Every scored set here is the Montero COMBINATION hold-out and is retained-values-only (measured). It is not the value-holdout population. Only the deletion comparator's numbers concern an unheld value.
4. **A dash is never a win for us.** Every dash is footnoted and names whose limit it is.
5. **No accuracy without its distinct-output count.**

### Cache generation 2

**The falsifier passed.** One shard (seed 1, 8192 rows) re-encoded twice in the same process; the sha256 of the quantised cache bytes matched: `1f6748ea769437571bd3f8de716c662e…` both times. Had it not, generation 2 would be noise and this spoke would have stopped here.

| seed | rows | scope | dtype | device | precision | features sha256 |
|---:|---:|---|---|---|---|---|
| 0 | 480,000 | full | float16 | cpu:0 | HIGHEST | `ef6359a4cd54b08287cf3709…` |
| 1 | 480,000 | full | float16 | cpu:0 | HIGHEST | `86b44d0f1e6470e7f1e0e0c3…` |
| 2 | 480,000 | full | float16 | cpu:0 | HIGHEST | `3ab14e583c3f500da060638c…` |

**GENERATION 1 DOES NOT PASS THE SHIPPED RE-ENCODE GATE, ON ANY SEED.** `exp_gen_store._assert_cache_reencodes` - the gate the scored cell calls at the top of every `run_seed` - re-encodes 64 cached rows through the encoder stored beside that cache and requires agreement within ONE ULP of the cache dtype on under 1% of elements. Run against both generations:

| seed | generation 1 | generation 2 |
|---:|---|---|
| 0 | **FAIL** - 51 ULP, 1280/131072 (0.98%) | PASS - 1 ULP, 1/131072 (0.00%) |
| 1 | **FAIL** - 81 ULP, 640/131072 (0.49%) | **FAIL** - 3 ULP, 3/131072 (0.00%) |
| 2 | **FAIL** - 59 ULP, 1231/131072 (0.94%) | PASS - 1 ULP, 2/131072 (0.00%) |

The generation-1 caches are not reproduced by their own stored encoder here, by **51-81 ULP on 0.49-0.98 % of elements**. That is the CPU-versus-GPU signature this spoke's task documented in advance (~53 ULP on ~1 % of elements), measured independently by running the programme's own gate. Generation 2 sits at 1-3 ULP on ~1e-5 of elements - the floor that batch-shape reordering allows, since the gate encodes 64 rows standalone while the cache was written in 8192-row blocks.

⛔ **What this does not say.** It does not invalidate any published generation-1 number: those numbers were computed on those features, and the features are what they are. It says generation 1 cannot be reproduced, extended or audited on this box - so it cannot be re-scored here, and a generation-1 number and a generation-2 number are measurements on different feature sets.

⭐ **What bounds the blast radius**, and it should be read beside the table above rather than after it: in the generation-2 re-score, the three DETERMINISTIC arms (`clu`, `knn_proto`, `store_removed`) move by **exactly 0.0000**. The regeneration is invisible to anything that is a deterministic function of the features. On its own the ULP table reads as alarming; the 0.0000 is what makes it precise.

⭐ **A consequence for the programme, not just this spoke.** A re-score on generation 1 was never possible: that gate would have refused all three seeds. Historical runs proceeded because the gate only fires ENFORCED when a cache is EXTENDED, and a full-scope cache with no misses never triggers it. The check existed, was correct, and was structurally unable to see the thing it was built to catch.

**Generation 1 and generation 2 disagree, with the predicted signature.** Same encoder parameter hash in every case, so this is the same encoder computing different bytes:

| seed | elements compared | differing | share | max abs diff |
|---:|---:|---:|---:|---:|
| 0 | 983,040,000 | 5,004,082 | 0.509% | 0.00390625 |
| 1 | 19,021,824 | 90,684 | 0.477% | 0.001953125 |
| 2 | 19,021,824 | 116,435 | 0.612% | 0.00390625 |

The max deviations are one float16 ULP at those magnitudes, on about half a percent of elements. That is the few-ULP / ~1 %-of-elements signature of a cache that was not all produced on one device. It does not invalidate any generation-1 number; it says those numbers belong to a different cache and cannot be quoted beside a generation-2 one.

### The rival set on GENERATION 2 (3 seeds: 0, 1, 2)

Judge = the frozen external classifier. Probe = its companion, on the same renders. `in-corner` is balanced accuracy on the held-out corner against its matched-seen control; chance is 0.5. `distinct` is distinct factor tuples the judge reads on B, out of 72 - an accuracy is never printed without it.

| rival | params | judge in-corner | probe in-corner | judge comp-B | distinct/72 | seeds |
|---|---:|---:|---:|---:|---:|---:|
| prototype table (incumbent) | 118,784 | 0.9165 ± 0.0306 | 0.8817 | 0.8650 | 56.0 | 3 |
| LACE (selection repaired) | 12,464,319 (95.1x) | 0.6155 ± 0.0261 | 0.6340 | 0.3048 | 71.7 | 3 |
| product-of-experts EBM | 3,556,665 (27.1x) | 0.5500 ± 0.0094 | 0.5139 | 0.1968 | 72.0 | 3 |
| MLP (one-hots) | 117,878 | 0.8983 ± 0.0497 | 0.8919 | 0.8943 | 46.0 | 3 |
| continuous-input MLP | 119,525 | 0.6377 ± 0.0743 | 0.6030 | 0.7184 | 45.0 | 3 |
| transformer (6 tokens, 2 layers) | 119,858 | 0.8463 ± 0.0201 | 0.8464 | 0.8526 | 48.3 | 3 |
| LSTM (6 tokens) | 119,125 | 0.8946 ± 0.0406 | 0.8961 | 0.8565 | 48.7 | 3 |

### Rivals at a second budget, and the request-dropout variants

Judge = the frozen external classifier. Probe = its companion, on the same renders. `in-corner` is balanced accuracy on the held-out corner against its matched-seen control; chance is 0.5. `distinct` is distinct factor tuples the judge reads on B, out of 72 - an accuracy is never printed without it.

These are kept out of the matched table above so they cannot be mistaken for it.

`mlp_180k` is at **179,968** parameters, 1.51x the matched target, because no arm on the read hub can be parameter-matched to this rival set: the pinned store alone is 131,072 parameters (524,288 B / 4), 10.3 % above the table's 118,784. That is a property of the programme's byte pin, not a choice any hub made. It answers whether a GENERIC LEARNER improves with 1.51x the parameters; it does not license 'parameter count does not matter' in general, because a table and an address book are different objects.

The `_dropout` variants are the SAME architectures at the SAME matched counts, trained with half their requests partially specified. They exist for rows 1, 2, 5 and 6, where an unrequested factor is genuinely `nan`. Rows 3 and 4 ask full six-factor requests, so the primary (full-only) variants above are the right rivals there — and the stronger ones: dropout training costs the MLP about 32 % on full-request validation MSE.

| rival | params | judge in-corner | probe in-corner | judge comp-B | distinct/72 | seeds |
|---|---:|---:|---:|---:|---:|---:|
| MLP at 179,968 params (the read hub's arm budget) | 178,952 | 0.9298 ± 0.0475 | 0.9620 | 0.9614 | 58.7 | 3 |
| MLP, request-dropout variant | 117,878 | 0.8800 ± 0.0640 | 0.8800 | 0.8326 | 40.3 | 3 |
| MLP at 179,968 params, request-dropout variant | 178,952 | 0.9229 ± 0.0662 | 0.9286 | 0.9367 | 51.7 | 3 |
| continuous-input MLP, request-dropout variant | 119,525 | 0.7108 ± 0.0324 | 0.6983 | 0.6073 | 30.0 | 3 |
| transformer, request-dropout variant | 119,858 | 0.8558 ± 0.0802 | 0.9017 | 0.8434 | 53.3 | 3 |
| LSTM, request-dropout variant | 119,125 | 0.8770 ± 0.0475 | 0.9043 | 0.8279 | 41.7 | 3 |

### Per-factor judge accuracy on B (mean over seeds)

The aggregate is the last thing to read. On this corner the prototype table reads every factor except `scale` at or near 1.000, and `scale` near chance — which is the whole story about what a per-value mean can and cannot represent, and it is invisible in the aggregate.

| rival | `floor_hue` | `object_hue` | `orientation` | `scale` | `shape` | `wall_hue` | aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|
| prototype table (incumbent) | 1.000 | 1.000 | 0.995 | 0.204 | 0.991 | 1.000 | 0.865 |
| LACE (selection repaired) | 0.310 | 0.273 | 0.231 | 0.324 | 0.352 | 0.338 | 0.305 |
| product-of-experts EBM | 0.171 | 0.208 | 0.167 | 0.306 | 0.199 | 0.130 | 0.197 |
| MLP (one-hots) | 1.000 | 1.000 | 0.644 | 0.741 | 0.981 | 1.000 | 0.894 |
| continuous-input MLP | 1.000 | 0.963 | 0.292 | 0.616 | 0.440 | 1.000 | 0.718 |
| transformer (6 tokens, 2 layers) | 1.000 | 1.000 | 0.755 | 0.454 | 0.907 | 1.000 | 0.853 |
| LSTM (6 tokens) | 1.000 | 1.000 | 0.801 | 0.421 | 0.917 | 1.000 | 0.856 |

Read a rival by the factors it carries, not by its mean. A rival that wins an aggregate while losing a factor another arm carries is not better at the task; it is better at a different mixture of it.

### LACE: the historical selection and the repaired one, both printed

`GenCellConfig.lace_selection_source` was NOT edited. The historical leg is still run from the pinned value; the repair re-runs the SAME 36-trial selection (same grid, same short schedule, same criterion) on the STRATIFIED rows the arm is actually scored on, in this spoke's own module.

| seed | published (gen 1) | historical selection | repaired selection | delta | historical regressed? | repaired regressed? | repaired config | repaired whitening-exact? |
|---:|---:|---:|---:|---:|:--:|:--:|---|:--:|
| 0 | 0.7077 | 0.6313 | 0.6231 | -0.0082 | True | True | `sigma=0.1 both ledoit_wolf` | False |
| 1 | 0.6732 | 0.6353 | 0.6278 | -0.0075 | False | False | `sigma=0.03 both 1e-3` | False |
| 2 | 0.6582 | 0.6511 | 0.6511 | 0.0000 | False | False | `sigma=0.03 both ledoit_wolf` | False |

The last column is not decoration. R27.3's exactness gate asks whether the SELECTED shrinkage leaves `p0(u) = N(0, I)` near-exact; only `none` clears the pinned 0.10 tolerance on this cache. A repaired selection that lands on `ledoit_wolf` is better on the validation criterion AND outside that licence, so "repaired" does not automatically mean "better licensed" and the flag must be read with the number.

Statistic: probe-companion in-corner balanced accuracy on the primary sampler, which is the statistic the harness's own regression check uses. The published column is the generation-1 reference constant from `GenCellConfig.lace_ode_probe_reference_per_seed` and is shown for orientation ONLY — it is a generation-1 number and must not be quoted beside a generation-2 one.

### The product-of-experts EBM, at its OWN budget, run twice

| seed | params | deployed bytes | byte multiple vs the 524,288 B store | run 1 judge in-corner | run 2 judge in-corner | spread |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,556,665 | 14,226,660 | 27.1x | 0.5417 | 0.5701 | 0.0201 |
| 1 | 3,556,665 | 14,226,660 | 27.1x | 0.5482 | 0.5595 | 0.0080 |
| 2 | 3,556,665 | 14,226,660 | 27.1x | 0.5603 | 0.5703 | 0.0071 |

Two independent runs, each with its own expert fit and its own sampler keys, because a 72-item corner read from a stochastic sampler cannot carry a confident claim from one draw. The byte multiple belongs beside every number this arm produces: it is not held to the store's budget and its numbers are not comparable in level to an arm that is.

### Did the PoE-EBM win outright?

The PoE-EBM does not win outright: the best judge in-corner mean is table at 0.9165, against the PoE-EBM's 0.5500. Reported either way.

### Which rival can enter which row

A dash below is a structural fact about the rival, decided in advance and footnoted. It is never a win for us.

| rival | rows 3-4 (FULL requests) | row 2 (off-grid value) | partial requests (query sets) | rows 5-7 (value hold-out) | callable |
|---|:--:|:--:|:--:|:--:|:--:|
| table | yes | yes | yes | yes | yes |
| cont_mlp | yes | yes | — | yes | yes |
| mlp | yes | — | — | yes | yes |
| transformer | yes | — | — | yes | yes |
| lstm | yes | — | — | yes | yes |
| mlp_180k | yes | — | — | yes | yes |
| *_dropout (every learner) | yes | — | yes | yes | yes |
| lace_repaired | yes | — | yes | yes | — |
| poe_ebm | yes | — | — | yes | — |

**Population caveat on rows 2 and 5-7.** Every rival here is fitted on the Montero combination pool, which contains every factor value. On a value-holdout row it saw the value the arm under test never did - the fitted table has 3,157 rows at `object_hue` 0.5 against a value-holdout arm's 0. Eligible, but NOT like-for-like: report with a population footnote, or refit on your own pool. Check with `load_rival(seed, name).like_for_like({factor: [values]})`.

Footnotes for every dash:

- **cont_mlp / partial** — PRIMARY variant, full requests only; load cont_mlp_dropout for a partial request
- **mlp / offgrid** — closed vocabulary: this rival's input is a class index over the realised grid, so a value between or outside the grid points cannot be expressed at all. Snapping it to the nearest grid value would score the rival on a request it never received, so it is refused instead. (one-hot over the grid values)
- **mlp / partial** — the PRIMARY variant is trained on fully specified requests only, which is what rows 3 and 4 ask. For a partial request load mlp_dropout, which is the same architecture at the same matched count.
- **transformer / offgrid** — closed vocabulary: this rival's input is a class index over the realised grid, so a value between or outside the grid points cannot be expressed at all. Snapping it to the nearest grid value would score the rival on a request it never received, so it is refused instead. (token over the grid values)
- **transformer / partial** — PRIMARY variant, full requests only; load transformer_dropout for a partial request
- **lstm / offgrid** — closed vocabulary: this rival's input is a class index over the realised grid, so a value between or outside the grid points cannot be expressed at all. Snapping it to the nearest grid value would score the rival on a request it never received, so it is refused instead. (token over the grid values)
- **lstm / partial** — PRIMARY variant, full requests only; load lstm_dropout for a partial request
- **mlp_180k / offgrid** — closed vocabulary: this rival's input is a class index over the realised grid, so a value between or outside the grid points cannot be expressed at all. Snapping it to the nearest grid value would score the rival on a request it never received, so it is refused instead. (one-hot over the grid values)
- **mlp_180k / partial** — PRIMARY variant, full requests only; load mlp_180k_dropout for a partial request
- ***_dropout (every learner) / offgrid** — same as the primary variant of that family: only cont_mlp_dropout accepts an off-grid value
- **lace_repaired / offgrid** — LACE's categorical head has a closed vocabulary over the realised grid and cannot express an off-grid value (chlu.core.gen_request.DenseRequests). THE METHOD'S limit, not this spoke's.
- **lace_repaired / callable** — this arm's output is a stochastic draw, not a function of the request: two calls on the same request return different features by construction. Ask this spoke to score your corners.
- **poe_ebm / offgrid** — the per-factor experts are categorical over the realised grid. THE METHOD'S limit, not this spoke's.
- **poe_ebm / partial** — the experts are composed by summing one energy per factor over a FULL assignment; an absent factor has no expert term and the product of experts is undefined for it
- **poe_ebm / callable** — this arm's output is a stochastic draw, not a function of the request: two calls on the same request return different features by construction. Ask this spoke to score your corners.

### The deletion comparator: retrain from scratch without the value

**Seed 0 cost line.** deleting object_hue=0.5 costs 3.08s and a refit of 29,099 rows by the gold standard, against an O(D) copy of one prototype row and 0 refitted rows by mean-substitution (below timer resolution, so no ratio is quoted); on the forget set the judge reads the deleted factor at 0.0 (gold) vs 0.0 (edit), and on the retain set 1.0 vs 1.0

**Seed 1 cost line.** deleting object_hue=0.5 costs 3.75s and a refit of 29,067 rows by the gold standard, against an O(D) copy of one prototype row and 0 refitted rows by mean-substitution (below timer resolution, so no ratio is quoted); on the forget set the judge reads the deleted factor at 0.0196 (gold) vs 0.0588 (edit), and on the retain set 1.0 vs 1.0

**Seed 2 cost line.** deleting object_hue=0.5 costs 3.20s and a refit of 29,011 rows by the gold standard, against an O(D) copy of one prototype row and 0 refitted rows by mean-substitution (below timer resolution, so no ratio is quoted); on the forget set the judge reads the deleted factor at 0.0 (gold) vs 0.0 (edit), and on the retain set 1.0 vs 1.0

SCOPE OF THE DELETION COMPARATOR. The value is held out of the ARM'S FIT POOL only, not from the encoder, the decoder or the judge, which are frozen and were fitted on a split containing every hue. 'Never trained on' means the arm never fitted a row carrying that value. The decoder can still render it and the judge can still recognise it, which is what makes the comparison scorable at all.

| seed | method | wall clock | rows refitted | judge on the deleted factor, FORGET set | same, RETAIN set |
|---:|---|---:|---:|---:|---:|
| 0 | cont_mlp_retrain_from_scratch | 75.572s | 29,099 | 0.0000 | 0.2294 |
| 0 | lstm_retrain_from_scratch | 91.949s | 29,099 | 0.0000 | 0.9957 |
| 0 | mlp_180k_retrain_from_scratch | 68.298s | 29,099 | 0.0000 | 1.0000 |
| 0 | mlp_retrain_from_scratch | 69.547s | 29,099 | 0.0000 | 1.0000 |
| 0 | table_edit_mean_substitution | 0.000s | 0 | 0.0000 | 1.0000 |
| 0 | table_full_pool | 18.744s | 32,256 | 1.0000 | 1.0000 |
| 0 | table_retrain_from_scratch | 3.081s | 29,099 | 0.0000 | 1.0000 |
| 0 | transformer_retrain_from_scratch | 178.227s | 29,099 | 0.0000 | 0.9978 |
| 1 | cont_mlp_retrain_from_scratch | 77.821s | 29,067 | 0.0000 | 0.1800 |
| 1 | lstm_retrain_from_scratch | 88.801s | 29,067 | 0.0000 | 0.9957 |
| 1 | mlp_180k_retrain_from_scratch | 75.195s | 29,067 | 0.0000 | 1.0000 |
| 1 | mlp_retrain_from_scratch | 69.634s | 29,067 | 0.0000 | 1.0000 |
| 1 | table_edit_mean_substitution | 0.000s | 0 | 0.0588 | 1.0000 |
| 1 | table_full_pool | 19.450s | 32,256 | 1.0000 | 1.0000 |
| 1 | table_retrain_from_scratch | 3.754s | 29,067 | 0.0196 | 1.0000 |
| 1 | transformer_retrain_from_scratch | 185.623s | 29,067 | 0.0392 | 1.0000 |
| 2 | cont_mlp_retrain_from_scratch | 54.109s | 29,011 | 0.0000 | 0.2343 |
| 2 | lstm_retrain_from_scratch | 76.248s | 29,011 | 0.0000 | 0.9935 |
| 2 | mlp_180k_retrain_from_scratch | 59.953s | 29,011 | 0.0000 | 1.0000 |
| 2 | mlp_retrain_from_scratch | 60.976s | 29,011 | 0.0000 | 1.0000 |
| 2 | table_edit_mean_substitution | 0.000s | 0 | 0.0000 | 1.0000 |
| 2 | table_full_pool | 16.781s | 32,256 | 1.0000 | 1.0000 |
| 2 | table_retrain_from_scratch | 3.198s | 29,011 | 0.0000 | 1.0000 |
| 2 | transformer_retrain_from_scratch | 140.611s | 29,011 | 0.0000 | 0.9935 |

Both sets are needed or neither number means anything: performance on the FORGET set should fall, and performance on the RETAIN set should not. A method that forgets the value by getting worse everywhere has not deleted anything, it has been damaged.

### Every existing arm, re-scored on generation 2

generation 1 and generation 2 are separate columns. The delta says whether the instrument change is visible; it is never a result and the two columns are never combined.

| arm | gen-1 mean | gen-1 seed sd | gen-2 mean | gen-2 seed sd | max abs delta | moved more than seed spread? |
|---|---:|---:|---:|---:|---:|:--:|
| clu | 0.6759 | 0.0132 | 0.6734 | 0.0176 | 0.0000 | False |
| dynread | 0.5577 | 0.0372 | NOT RUN | NOT RUN | NOT RUN | None |
| dynread/A_request_off | 0.5000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | None |
| dynread/B | 0.5742 | 0.1047 | NOT RUN | NOT RUN | NOT RUN | None |
| dynread/B_request_off | 0.5000 | 0.0000 | NOT RUN | NOT RUN | NOT RUN | None |
| ebm_compose | 0.5148 | 0.0171 | 0.5215 | 0.0206 | 0.0069 | False |
| knn_proto | 0.8817 | 0.0028 | 0.8803 | 0.0022 | 0.0000 | False |
| lace | 0.6399 | 0.0025 | 0.6412 | 0.0140 | 0.0139 | False |
| lace/ld_corrected | 0.6161 | 0.0151 | 0.6327 | 0.0037 | 0.0177 | True |
| lace/ld_published | 0.5547 | 0.0757 | 0.6436 | 0.1648 | 0.1181 | False |
| store_removed | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | False |

**Verdict: AT LEAST ONE ARM MOVES BY MORE THAN ITS SEED SPREAD - report this to every hub before it scores against generation 2**

### External corroboration, and a caution about the table above

Everything in this section is **hub/read's measurement, on hub/read's population, with hub/read's judge**, reported here with attribution. None of it is mixed into the tables above and no number here is this spoke's.

**The rival ordering is population-dependent, severely.** Same two rivals, two corner sets:

| | prototype table | `mlp_180k` | gap |
|---|---:|---:|---:|
| this spoke's corner B, this spoke's judge | 0.9165 | 0.9298 | 0.013 |
| hub/read's 49 n-way corners, its judge | 0.1550 | 0.7426 | 0.588 |

The prototype table is nearly competitive here and collapses there. Corner B is a single Montero cell - 72 rows, one `shape` value, retained values throughout - and an additive per-value-mean composition handles it well. hub/read's corners are n-way conjunctions, which the additive rule cannot represent.

⛔ **So do not read the table above as 'the prototype table is a strong rival'.** It is a strong rival *on the easiest compositional population in the programme*. The levels across the two rows are not on a common scale (different populations, judges and request sets); it is the change in ORDERING that is the finding.

**The matched-budget result replicates independently, and is larger elsewhere.** The same 1.0x -> 1.51x parameter step on the same architecture:

- this spoke, corner B: +0.033 balanced accuracy, +12.7 distinct factor tuples
- hub/read, 49 n-way corners: 37.245 -> 53.469 novel tuples per corner, i.e. +16.2

Two populations, two instruments, same direction. On hub/read's scoreboard that step exceeds its dial's entire measured headroom (+9.08 tuples/corner from its deployed read to an oracle address), and it reports its hub falsified by `mlp_180k` on both of its target rows.

⛔ **`mlp_180k` carries TWO budget framings and both must be stated, or it misleads in one direction or the other.** It is **1.51x** the prototype table's 118,784 - the budget this spoke's own rival set is matched to - and simultaneously **0.994x** hub/read's arm at 179,968, i.e. matched to that arm within -0.57%. Against the table it is the over-budget rival; against that hub's arm it is THE MATCHED RIVAL. This spoke initially quoted the first while reasoning about the second and offered hub/read a softening it correctly declined.

⚠ The caveat this spoke attached to `mlp_180k` survives and is repeated: it tests whether a GENERIC LEARNER improves with 1.51x the parameters, measured against the TABLE's budget. It does not license 'parameter count beats mechanism', and 1.51x of an MLP is not 1.51x of a store. hub/read's own ablation against its incumbent read is +1.143 [-2.898, +5.061], i.e. nothing - so on those corners BOTH addressed reads sit below the matched generic learner, which is a statement about the addressed-read FAMILY rather than about any one variant.

⭐ **Why the second budget existed at all.** `mlp_180k` was built on a side request, not from this spoke's plan. Without it, hub/read would have compared against the parameter-matched `mlp` alone, measured +3.592 [-0.816, +8.103], and read that as a win - while the pinned store's 131,072 parameters put its arm permanently outside the matched budget. A rival set matched to ONE budget is a comparison at one budget; the second budget is what made the first interpretable.

## Readiness

`RIVALS-READY.json` status: **ready**, seeds [0, 1, 2], rivals present: ['cont_mlp', 'lace_repaired', 'lstm', 'mlp', 'poe_ebm', 'table', 'transformer'].
