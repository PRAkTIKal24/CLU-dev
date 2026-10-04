# w13 hub/encoder-adapt — LOG

## 2026-09-21T17:03:11Z — START
- Wrote START_UTC.txt.
- Read /c/chlu/tasks/w13-hub-encoder-adapt.md (task + standing rules).
- Note: standing-rules block contains templated paths (`w13-w13-hub-encoder-adapt`); job body
  and invocation are authoritative: output dir `/c/chlu/outputs/w13-encoder-adapt/`,
  branch `w13/encoder-adapt`.

## 17:03–17:40Z — programme state read, environment, and the reuse decision
- Read (evidence base only, ⛔ no `$CHLU_CLAUDE_ROOT`):
  `/c/chlu/evidence/outputs/w9-advisor-report.md`, `w10-advisor-report.md`,
  `w12-deletion-baselines/FINDINGS.md`, `w12-composition-ladder/FINDINGS.md`,
  `w12-single-design-crossover/{CROSSOVER-val.json,FINDINGS.md}`, `/c/chlu/SCORED-LEDGER.md`.
- ⭐ Ledger confirms target `encoder-adapt` **0/3, its own budget**, NOT drawing on
  `encoder-loop` (`/c/chlu/SCORED-LEDGER.md` line 40).
- ⭐ **K\* = 32 CONFIRMED from `CROSSOVER-val.json`**: `K_star=32`, `A_requested=63`,
  well width 64..64, `rank_over_width_max = 0.9844` (min 0.2656),
  four-part ledger 2,143,138 B = 4.0877x the 524,288 B reference.
  (The task's "0.9844" is the MAX; the MIN is 0.2656 and is carried too.)
- ⭐ **The w12 NEGATIVE margin reproduced arithmetically from
  `w12-composition-ladder/FINDINGS.md` §0**: store/learned_1x ceiling-normalised
  (store 0.8483/0.8299/0.8709/0.8989/0.8966 at k=2..6; store abs 30.5389/59.7500/
  62.7083/64.7194/64.5556; learned_1x 33.3222/55.5083/67.9917/68.6194/68.6167)
  => ceiling = store_abs/normalised, margin = (store-learned_1x)/ceiling =
  **-0.0773 (k=2) / -0.0734 (k=4) / -0.0542 (k=5) / -0.0564 (k=6)**, i.e. exactly the
  task's "-0.054 to -0.077 of ceiling at n = 2, 4, 5, 6". Bar starts NEGATIVE. Confirmed.
- ⭐ `chlu.__file__ = C:\chlu\worktrees\CHLU-w13-encoder-adapt\chlu\__init__.py` (asserted;
  will be re-asserted in code before every number). Python 3.11.9, jax 0.9.0 CPU,
  equinox 0.13.4, optax 0.2.6, numpy 2.4.1. ⛔ No torch on this box; no GPU visible.
- ⚠ The uv venv trampoline at `chlu-bundle-v2/.venv/Scripts/python.exe` is BROKEN
  (`uv trampoline failed to spawn Python child process`). The working interpreter is the
  system `Python311` with `$PYTHONPATH` = worktree ; that venv's site-packages.
- **REUSE DECISION.** Merged `w12/single-design-crossover` into `w13/encoder-adapt`
  (fast-forward, additive; 13 files) so the SCORED leg runs w12's own K* harness
  (`scripts/w12_crossover_lib.py`) unmodified rather than a re-implementation.
  The encoder pipeline is `chlu/experiments/exp_w8_encoder_loop.py` (warm / train /
  cache / probe / rivals stages), reused unmodified for everything downstream of `train`.

## 17:06–17:12Z — harness built
- NEW: `chlu/experiments/exp_w13_encoder_adapt.py` (1012 lines). Arms `frozen`/`adapter`/
  `anchor`; `AdaptedEncoder` (LoRA-style, up-projection zero-init); per-arm Optax filter;
  anchor penalty (feature + weight relative L2) with lambda = recon_0 / rho^2, rho = 0.10.
  It REGISTERS its arms into `exp_w8_encoder_loop.ARM_SPEC/ARM_CLASS` and overwrites that
  module's PINS, so w8's `stage_cache` / `stage_probe` / `stage_rivals` run UNMODIFIED on
  w13 arms. The registration is explicit and printed, not hidden.
- ⚠ **A real implementation decision, recorded because it is a deviation.** w8's swap step
  compiles ONE XLA graph PER WELL (`well` is a static Python int inside
  `gen_swap_loss.swap_loss_terms`). At w8's K = 5 that is 5 compilations; at this wave's
  K* = 32 it is 32 compilations of a graph carrying encoder+store+decoder+judge.
  w13 therefore composes `gen_swap_loss`'s OWN primitives (`swapped_features`,
  `quantise_through_store`, `intended_tuple`) in `gen_swap_loss`'s own order with the
  well's channel row TRACED, and keys the compile on the OWNED FACTOR (<= 6 values).
  ⭐ Equality with `SL.swap_loss_terms` is ASSERTED NUMERICALLY on a real batch before the
  first update (`swap_identity_vs_frozen_swap_loss` on every arm manifest); a non-zero
  delta REFUSES the run.
- NEW: `scripts/w13_score_leg.py` — wave 12's own K* harness driven per ENCODER ARM with
  the four documented redirections; the CLASSIFIER does not move.
- NEW: `scripts/w13_killproof.sh` — the kill -9 proof (own PID only, never box-wide).
- `PREREG-w13-encoder-adapt-1.md` written and timestamped **2026-09-21T17:11:58Z**,
  before any w13 number existed.
- ⚠ Box clock note: UTC = local − 2 h on this box; all times in this LOG are UTC.
- ⚠ **What this hub reads under `$CHLU_CLAUDE_ROOT`:** nothing. The harness's own
  resolvers (`CHLU_GEN_BRIDGE_ROOT`, `CHLU_GEN_CLASSIFIER_ROOT`, `CHLU_GEN_DECLIFT_DIR`)
  load the FROZEN INSTRUMENT CHECKPOINTS from there, exactly as w8 and w12 did. ⛔ No
  report, no findings file and no number under that root is read by this hub.

## 17:14Z — ⭐⭐ FINDING 1, from the smoke, BEFORE any spend: THE SWAP LOSS IS ILL-POSED AT K* = 32
- Ran `--stage warm --seed 0 --quick` with the in-loop store at the task's pin K = 32,
  A = 63. The frozen harness REFUSED, verbatim
  (`/c/chlu/outputs/w13-encoder-adapt/dev/logs/smoke_warm_s0.log`, and
  `chlu/core/gen_swap_loss.py:148`):

  > `AssertionError: well(s) [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  > 21, 22, 23, 24, 26, 27, 28, 29, 30, 31] own no factor -- the swap loss has no target
  > for them and is ill-posed; reported, not patched`

  **26 of 32 wells own no factor.** Shapes3D has 6 factors; the swap loss gives each well
  exactly ONE owned factor by a Hungarian match on a measured transfer matrix, so at most
  6 wells can own one at any K.
- `gen_swap_loss.derive_fixed_ownership`'s own docstring states the design: *"With K=5
  wells and 6 Shapes3D factors exactly one factor is UNOWNED by construction; that is a
  property of the pinned ledger"*. ⭐ The swap loss is DEFINED for K of the order of the
  factor count.
- ⛔ **NOT PATCHED.** Restricting the swap draw to the 6 owning wells would be patching a
  refusal the harness makes deliberately ("reported, not patched" is its own wording).
- ⭐ **RESOLUTION, declared:** the two stores are separated and carry different K.
  * **SCORED OBJECT — unchanged, K\* = 32**: `CovContentStore` at K\* = 32, A = 63,
    rebuilt from EACH ARM's own generation-2 cache by wave 12's unmodified harness.
    ⭐ The task's pin is honoured exactly where the task puts it: on the store design.
  * **IN-LOOP VQ STORE — K = 5, A = 64**, w8's pinned ledger. Atom block = A x D x 4 =
    64 x 2048 x 4 = **524,288 B**, w8's pin exactly. (Same arithmetic at K\* = 32, A = 63
    gives 63 x 2048 x 4 = 516,096 B = w12's own `p1_atom_bytes` — the two ledgers agree.)
- ⚠ This is a real collision between two pins the task gives, resolved by separating the
  objects, and it is reported rather than absorbed.
- Killed my three K=32 warm PIDs (132149/132152/132155) — **only those PIDs**, never a
  box-wide kill — and relaunched at the corrected pin (132315/132318/132321 + smoke 132324).

## 17:18–17:38Z — ⭐ KILL -9 RESUMABILITY: PROVED, BEFORE ANY SPEND
`scripts/w13_killproof.sh`, artefacts in `dev/killproof/`.
Arm `adapter`, seed 0, `--quick` (60 steps x 16), 5 s wall-clock checkpoint timer.

| part | result | evidence |
|---|---|---|
| 1. checkpoint then **SIGKILL my own PID** | ⭐ PASS | `killproof.log`: victim PID **132774**, newest ckpt `ckpt_000000000001.npz`, `SIGKILL sent to 132774`, `victim reaped; exit status 137` (= 128+9) |
| 2. relaunch must RESUME | ⭐ PASS | `run_resumed.log`: `[adapter] RESUMED from checkpoint step 1; 59 steps remain`, exit 0, **23 checkpoint saves** |
| 3. resumed == uninterrupted | ⭐ PASS on every TRAINED object | `killproof.json`: `encoder`, `base_encoder`, `store`, `joint_decoder` param hashes **IDENTICAL**; fingerprints equal (`5b616fc355eda4b7`) |
| 4. different fingerprint must be refused BY NAME | ⭐ PASS | `fingerprint_refusal.json`: `FingerprintMismatch: checkpoint ... was written by config fingerprint 552cea0330a2b8d9, this job is f7659c4708fd127b. Refusing to resume.` |

⛔ **Only PID 132774 was signalled.** No box-wide kill. PIDs recorded in `MY_PIDS.txt`.
⚠ **One object did NOT match and it is reported, not hidden:** `render_decoder`
(`0fb21401...` resumed vs `45e12010...` clean). It is refit AFTER the training loop, so
the checkpoint does not cover it; the trained objects the checkpoint DOES cover all match
bit-for-bit, and the displacement figures are identical to 16 digits
(`feature_relative_l2_squared 0.015152376145124435` in BOTH runs). A control isolating
this to XLA CPU load-dependence rather than to resumption is run below.

## 17:38Z — three measured facts from the same artefacts
1. ⭐ **The dynamic swap term IS `gen_swap_loss.swap_loss_terms`**, asserted numerically:
   `abs_delta_loss = 0.0`, `abs_delta_exact = 0.0` (exactly zero), both runs
   (`dev/killproof/*/seed_0/adapter/arm.json -> compute.swap_identity_vs_frozen_swap_loss`).
2. ⭐ **The adapter is the reference pin and is bit-identical to frozen at step 0**:
   `rank 32, D 2048, param_count 131,072, bytes_float32 524,288,
   bit_identical_to_frozen_at_step_0: true` (checked on 8 real images; the run is refused
   if it is not). `n_encoder_params_optimised = 131,072` — exactly the adapter, nothing else.
3. ⚠ **The in-loop store reproduces w8's pin exactly**: `store_bytes 524,288`,
   `n_modes 64`, `n_wells 5`, well sizes [625, 333, 452, 278, 360],
   ownership `own = [3, 0, 4, 5, 1]` — all five wells own a distinct factor, factor 2
   unowned, exactly as `gen_swap_loss`'s docstring says is true "by construction" at K = 5.

## 18:18Z — the THREE SHARED WARM STARTS are on disk (seeds 0/1/2)
`run/seed_{0,1,2}/warm_start.json`. 1500 steps x 128 on 8192 rows; warm decoder 200 steps.

| seed | warm s | enc loss final | warm-dec loss final | encoder hash | store hash | in-loop well sizes | own | identity ceiling (exact) |
|---|---|---|---|---|---|---|---|---|
| 0 | 2608.1 | 0.00236 | 0.00702 | `4a6e7a6366ec` | `c1ae2ae1d85b` | 268/350/297/281/852 | [4,1,0,3,2] | **0.1641** |
| 1 | 2633.4 | 0.00334 | 0.00870 | `1095dedc843a` | `fc6de6182833` | 307/453/457/315/516 | [4,1,0,3,2] | **0.1523** |
| 2 | 2610.8 | 0.00285 | 0.00735 | `12bccb25b623` | `1e22d5819454` | 367/268/440/287/686 | [1,3,0,5,2] | **0.1406** |

⭐ `store_bytes = 524,288` on every seed — w8's pin exactly.
⚠ **The identity ceiling is 0.14–0.16, and it is the number that decides whether this
comparison can be made at all.** It is the judge's exact-tuple read-back on REAL features
through the SHARED 200-step warm decoder; the per-arm renderer gets 400 steps. ⛔ If the
composition numbers come back at the floor, this hub reports "the comparison cannot be
made at this compute" and spends NO scored run — w8's own dev-leg rule. A **val** floor
check is run on seed 0 before any test-condition leg.

## 18:18Z — seed-0 lanes launched (train -> cache -> probe -> rivals)
PINS UNCHANGED FROM w8: `joint_steps 400 x batch 64 = 25,600 images`, `dec_refit_steps 400`.
⭐ The renderer's step count is NOT cut below w8's: it is the instrument, and w8 already
cut it from the gated winner's 1800.
3 lanes x 1 thread (`OMP/MKL/OpenBLAS/NumExpr = 1`), i.e. 3 of this hub's 4 threads, the
4th left for the scoring and reporting work. PIDs 134896 / 134900 / 134906 in MY_PIDS.txt.

## 19:56Z — ⛔ FINDING 2: `w4_learners.train` CAST TOKEN IDS TO FLOAT AND COULD NOT FIT THEM
The rivals stage failed on `transformer` on all three seed-0 arms
(`logs/{frozen,adapter,anchor}_s0_rivals.log`):

> `TypeError: Indexer must have integer or boolean type, got indexer with type float32`
> at `equinox/nn/_embedding.py:101  return self.weight[x]`, from
> `chlu/eval/rivals/w4_learners.py:446 train(...)`

CAUSE, located and verified: `train()` did `Xnp = np.asarray(X0, np.float32)`
unconditionally. The two `tokens` families (`transformer`, `lstm`) feed those ids into
`eqx.nn.Embedding`, which jax 0.9.0 refuses to index with a float. `predict()` two
functions below never made the cast, so INFERENCE always preserved the dtype and only
TRAINING was broken. `mlp` (one-hot) and `cont_mlp` (continuous) are unaffected and had
already fitted cleanly.
⚠ **This is not new to w13.** `w8-encoder-loop/run/seed_*/{alone,loop_swap}/rivals.json`
records `families_done = ["mlp_180k"]` on **all six** arm-seeds — the token families were
never fitted there either.
⭐ **REPAIRED, not declared not-run:** a rival that cannot be FITTED is a rival that
cannot lose, and this hub's bar needs the strongest available competitor. The repair
preserves an integer input's dtype and nothing else — ⛔ no change to the protocol, the
grids, the architecture, the schedule or the parameter budget
(`chlu/eval/rivals/w4_learners.py`, commit `w13: REPAIR w4_learners.train ...`).
Verified after the repair on 4096 rows x 2 epochs: transformer / lstm / mlp / cont_mlp
all fit and all return a finite val MSE.
Rivals relaunched for seed 0 with **five** families: `mlp, mlp_180k, cont_mlp,
transformer, lstm` (`mlp_180k` = 179,968 params, w8's "THE BAR" = store + address book).
LACE remains a DECLARED NOT-RUN with its own reason.

## 19:56Z — seed-0 TRAIN + PROBE are on disk. Equal steps and images, PRINTED
`run/seed_0/{frozen,adapter,anchor}/arm.json`

| arm | steps | images | train s | s/step | enc params in optimiser | base encoder changed | decoder refit s |
|---|---|---|---|---|---|---|---|
| frozen | 400 | 25,600 | 2297 | 5.743 | **0** | False | 2279 |
| adapter | 400 | 25,600 | 2368 | 5.921 | **131,072** | **False** | 2261 |
| anchor | 400 | 25,600 | 2523 | 6.309 | **428,576** | True | 2217 |

⭐ **Equal steps AND equal images on all three arms, asserted in the manifests.**
⭐ Seconds span 2297–2523 s, a **9.8 %** spread — equal steps is within 10 % of equal
seconds here, so no separate seconds-matched control decides anything. (w8 needed one
because its treatment cost 2.2x its control; here every arm carries the same swap term.)
⭐ `n_encoder_params_optimised` reads **0 / 131,072 / 428,576** — the arm labels are
properties of the Optax filter, checked, not claimed.

### The displacement from the frozen encoder (REPORTED, never tuned against)
| arm | feature rel-L2² | weight rel-L2² |
|---|---|---|
| frozen | **0.000000** | **0.000000** |
| adapter | 0.157581 | 0.126956 |
| anchor | **0.002146** | **0.000324** |

⭐ **The anchor penalty does what it says.** `anchor`'s feature displacement is **73x
smaller** than the UNPENALISED `adapter`'s, and it sits **4.7x below** the declared
reference displacement (rho = 0.10 => rel-L2² = 0.01). ⛔ The falsifier *"anchor diverges
from the frozen features"* **does not fire**, and lambda was not retuned to make that so.
⚠ Conversely `adapter`, which has no penalty, moves the features by rel-L2² 0.158
(rel-L2 0.397, i.e. ~40 %). "Light" describes its PARAMETER count, not its effect on the
features, and this hub says so.

### Losses, seed 0 (first step -> last step)
| arm | total | recon | commitment (= the STORE READ ERROR) | swap_exact |
|---|---|---|---|---|
| frozen | 0.19270 -> 0.15304 | 0.02069 -> 0.01777 | 0.12566 -> **0.10126** | 0.0000 -> **0.0000** |
| adapter | 0.19270 -> **0.13669** | 0.02069 -> **0.01350** | 0.12566 -> **0.09254** | 0.0000 -> **0.0000** |
| anchor | 0.19270 -> 0.15551 | 0.02069 -> 0.01787 | 0.12566 -> 0.09862 | 0.0000 -> **0.0000** |

⭐ All three arms start at the SAME loss 0.19270 — one bit-identical warm start per seed,
visible in the number.
⭐ `adapter` lowers the store read error most (0.09254 vs frozen 0.10126, −8.6 %).
⛔⛔ **`swap_exact` is 0.0000 at the first step AND the last step on EVERY arm.** The swap
term's own headline statistic — the fraction of swaps the judge reads back as exactly the
intended tuple — never leaves zero at this compute. That is a NULL on the swap loss, it is
reported as one, and it is consistent with the 0.14–0.16 identity ceiling measured on the
warm start.

### ⭐ The factor probe — MARGIN over random init, seed 0, NEVER pooled
| arm | aggregate margin | exact-match margin | arm absolute | random-init floor |
|---|---|---|---|---|
| frozen | **+0.0001** | −0.0034 | 0.9736 | 0.9735 |
| adapter | **+0.0007** | −0.0024 | 0.9742 | 0.9735 |
| anchor | **−0.0016** | −0.0178 | 0.9719 | 0.9735 |

⛔ **No arm buys anything over an UNTRAINED encoder of the same architecture.** The
absolute 0.97 reproduces w8's measurement that a random-init encoder probes Shapes3D at
0.962–0.972 — which is exactly why no absolute probe number is a claim here.
⚠ Seed 0 only; seeds 1 and 2 follow, and the three are reported per seed.

## 20:34Z — all five learned baselines fitted on all three seed-0 arms
`run/seed_0/*/rivals.json`, `families_done = [mlp, mlp_180k, cont_mlp, transformer, lstm]`.
Param counts identical across arms (the protocol is): mlp 117,878 (-0.76% of 118,784),
mlp_180k 178,952 (-0.56% of 179,968), cont_mlp 119,525 (+0.62%), transformer 119,858
(+0.90%), lstm 119,125 (+0.29%). Fit seconds 244-911 per family per arm.
Val MSE on the arm's own features (lower = better rival):

| family | frozen | adapter | anchor |
|---|---|---|---|
| mlp_180k | 0.015666 | 0.015934 | **0.014637** |
| mlp | 0.022216 | 0.022682 | **0.020797** |
| lstm | 0.021808 | 0.022293 | **0.020509** |
| transformer | 0.023992 | 0.024397 | **0.022645** |
| cont_mlp | 0.035193 | 0.036606 | **0.033397** |

⚠ **The baselines are UNIFORMLY BEST on `anchor`'s features and worst on `adapter`'s, on
all five families.** That is a feature-space fact about the arms and it is exactly the
trap the bar exists to catch: a baseline that predicts an arm's features better is a
STRONGER competitor on that arm, so `anchor`'s margin is measured against a harder bar.

## 21:26Z — ⭐ THE FREE VAL FLOOR CHECK (seed 0, `frozen`) — THE COMPARISON CAN BE MADE
`valscore/val_seed0_frozen_K32/score.json`. K* = 32, A = 63, wells 64..64, rank/width
0.9375..0.9844, four-part ledger **2,148,858 B = 4.0986x** the 524,288 B reference,
address book 1,065,544 B fitted in 2,292 s, store built in 7.8 s. 24 corners, 1,728 requests.
Dead channels: frozen partition file 67, THIS ARM 53, **union 110 of 2,048** (reported, not
absorbed; wells still come out 64 wide).

| row | distinct_valid/corner | judge acc | valid frac |
|---|---|---|---|
| **`cov_tensor`** | **38.500** (sd 8.193) | **0.5347** | 0.5347 |
| `mlp_180k` | 36.583 | 0.5081 | 0.5081 |
| `transformer` | 30.000 | 0.4167 | 0.4167 |
| `lstm` | 27.958 | 0.3883 | 0.3883 |
| `mlp` | 25.625 | 0.3559 | 0.3559 |
| `cont_mlp` | 24.083 | 0.3345 | 0.3345 |
| ⭐ `landscape_removed` | **0.000** | **0.0000** | 0.0000 |
| ⭐ `landscape_shuffled` | **0.000** | **0.0000** | 0.0000 |
| ⭐ `landscape_shuffled_across_wells` | **0.000** | **0.0000** | 0.0000 |
| ⭐ `random_atoms` | **0.000** | **0.0000** | 0.0000 |

⭐⭐ **ALL FOUR CHARTER-C CONTROLS READ EXACTLY 0.0000.** The renderer renders, the judge
reads, and the gate the whole design hangs on passes.
⚠ `distinct_valid_novel_tuples` came back **exactly 0.000 on EVERY row** — arm, controls
and all five rivals alike. ⛔ That is NOT a floor: it is a property of the **val split**
(a val corner's tuples are inside `train_rows`, so `seen_tuple_mask` marks every one seen
and NOTHING can be novel, for us or for a rival). On `test` the corners are held out and
novelty is meaningful. ⭐ **Therefore this hub does NOT report "the comparison cannot be
made at this compute", and the renderer's 400 steps are NOT cut further.**
- Score leg patched to carry BOTH metrics plus judge accuracy, valid fraction and the four
  control values on every row, and a DEGENERACY RULE declared in **AMENDMENT 2**
  (timestamped above, before any test number): the primary is used unless it reads exactly
  0 for the arm AND every baseline AND every control, in which case the secondary decides
  and the row says `primary_degenerate: true`.
- ⚠ FOR THE RECORD, NOT A CLAIM (val, seed 0, one arm): paired `distinct_valid_tuples`
  margin of the single design over its best learned baseline (`mlp_180k`) = **+1.917 per
  corner** (n = 24, sd 4.452). Against `mlp` +12.875, `cont_mlp` +14.417, `transformer`
  +8.500, `lstm` +10.542.

## 21:27:53Z — LEDGER ROW WRITTEN, BEFORE ANY SCORED RUN
`/c/chlu/SCORED-LEDGER.md`, row `w13 hub/encoder-adapt | encoder-adapt | 1/3`, appended
2026-09-21T21:27:53Z — **before** the first `--condition test` process was launched
(21:27:58Z). ⭐ This hub's OWN row under its OWN target; another hub's row authorises
nothing. Scope declared ON the row: **run 1/3 covers the WHOLE pre-registered comparison**
— 3 arms x 3 seeds x the TEST condition (49 n-way + 60 pair sub-corners) — because the bar
is a single cross-arm comparison that cannot be evaluated from any subset.

## 23:04Z — ⭐ FIRST SCORED ROW: seed 0, `frozen` (the REFERENCE)
`score/test_seed0_frozen_K32/score.json`. 109 corners (60 pair + 49 n-way), K* = 32,
A = 63, wells 64..64, rank/width 0.9844, ledger **2,147,298 B = 4.0956x** the 524,288 B
reference, address book fitted in 2,185 s. Dead channels 64 (partition file) / 53 (this
arm) / **107 union** of 2,048. Leg wall clock 95 min.

⭐ `primary_degenerate: false` — on TEST the novel metric separates, and it is the one used.
⭐⭐ **ALL FOUR CHARTER-C CONTROLS READ EXACTLY 0.0**.

| row | novel/corner | judge acc |
|---|---|---|
| **`cov_tensor`** | **23.3119** | 0.4378 |
| `mlp_180k` (the best learned baseline) | 22.7431 | 0.4335 |
| `lstm` | 20.798 | 0.3926 |
| `transformer` | 18.248 | 0.3392 |
| `landscape_removed` / `landscape_shuffled` / `landscape_shuffled_across_wells` / `random_atoms` | **0.0000** | — |

**MARGIN(frozen, seed 0) = +0.5688** per corner vs `mlp_180k`
(row 3 pair −0.4667, row 4 n-way +1.8367).
Per baseline: mlp +4.872, mlp_180k +0.569, cont_mlp +12.266, transformer +5.064, lstm +2.514.
⚠ **Paired, it is not distinguishable from zero:** bootstrap CI95 **[−0.743, +1.917]**,
median paired delta **0.000**, corners won/tied/lost **50 / 9 / 50**, sign-test
p = **1.0000**. The margin-declared-in-advance of +1.0/corner is NOT cleared by the
reference arm, and the reference arm is not asked to clear it — this is the baseline the
treatments must GROW from.

## 23:08Z — seed 1 lanes COMPLETE; seed 2 launched; scored worker moved to seed 1
`run/seed_1/*/arm.json`:

| arm | steps | images | train s | enc params optimised | base changed | dF rel-L2² | dW rel-L2² | swap_exact final |
|---|---|---|---|---|---|---|---|---|
| frozen | 400 | 25,600 | 2291 | 0 | False | 0.000000 | 0.000000 | 0.0000 |
| adapter | 400 | 25,600 | 2308 | 131,072 | False | 0.111499 | 0.104833 | 0.0156 |
| anchor | 400 | 25,600 | 2309 | 428,576 | True | **0.002393** | **0.000247** | 0.0156 |

⭐ Seconds spread on seed 1 is **0.8 %** (2291–2309 s) — equal steps IS equal seconds here.
⭐ `anchor`'s feature displacement is again ~47x smaller than the unpenalised `adapter`'s.
⛔ `swap_exact` remains at or beside zero (0.0000 / 0.0156 / 0.0156 = 1 of 64) — the NULL
on the swap loss reproduces on a second seed.
⛔ **Probe margins on seed 1 are NEGATIVE on ALL THREE arms**: −0.0160 / −0.0064 / −0.0059
aggregate (arm 0.9495 / 0.9591 / 0.9602 against random-init floors 0.9655 / 0.9655 /
0.9661). ⚠ Seed 0 gave +0.0001 / +0.0007 / −0.0016. **The sign of the probe margin is not
stable across seeds** — exactly the instability w8 and w9's harness each found
independently, which is why this is reported per seed and NEVER pooled.

## 01:37–04:11Z — seed 2 lanes complete; ALL NINE SCORED LEGS COMPLETE
`score/test_seed{0,1,2}_{frozen,adapter,anchor}_K32/score.json` — 9 of 9 on disk.
Gate check across all nine, machine-verified:
- `primary_degenerate: false` on **all 9** (on TEST the novel metric separates)
- all four charter-C controls read **EXACTLY 0.0** on **all 9**
- corner counts `(60 pair, 49 n-way)` on **all 9**
- best learned baseline `mlp_180k` on **all 9**
- `K* = 32, A = 63` on all 9; rank/width max in {0.9688, 0.9844}
- four-part ledger 2,123,898–2,160,818 B = **4.051x–4.121x** the 524,288 B reference
- address book fitted 879–2,185 s per leg; dead-channel union 64–164 of 2,048

Seed 2 arms (`run/seed_2/*/arm.json`): 400 steps x 25,600 images on every arm; seconds
2211 / 2047 / 1640 (the TREATMENTS were faster than the reference on this seed);
`n_encoder_params_optimised` 0 / 131,072 / 428,576; displacement dF
0.000000 / 0.137014 / 0.001711; `swap_identity` abs_delta **0.0** on all nine arm-runs.

## 04:11Z — CROSS-ARM.json / CROSS-ARM.md written; FINDINGS.md written
`scripts/w13_report.py` re-derives nothing: it lines up what is on disk.
**THE BAR:** `anchor` grows on **3/3** seeds (+0.7615 / +0.0183 / +0.2018);
`adapter` on **2/3** (+1.0550 / +0.3945 / **-0.7982**). Declared bar was >= 2 of 3.
⛔⛔ **AND THE TRAP FIRED:** in **6 of 6** treatment cells the single design and its
strongest baseline move in the SAME direction by comparable amounts, so the growth is a
finding about the ENCODER, not about the store. That is the headline, not a caveat.
⛔ No leg of the nine clears the programme's +1.0/corner with a CI95 low above +1.0.

## Housekeeping
- ⛔ **No box-wide kill at any point.** Every PID launched by this hub is in `MY_PIDS.txt`;
  the only kills issued were to this hub's own recorded PIDs (132149/132152/132155 after
  FINDING 1; 138251/138256/138261 after a mis-ordered launch argument; 132774 as the
  killproof victim).
- ⚠ Several of this session's *poll shells* were reaped by the harness under system memory
  pressure. ⭐ No compute process was affected: every reap was checked and all worker PIDs
  survived. Recorded because it is why the poll cadence changes mid-LOG.
- Scored runs on `encoder-adapt`: **1 of 3 SPENT. 2 and 3 UNSPENT and returned.**

## 04:13Z — BUNDLE
`git bundle create /c/chlu/outputs/w13-encoder-adapt/w13-encoder-adapt.bundle w13/encoder-adapt`
⛔ that branch only, never `--all`.
- `git bundle verify`: **"is okay"**, *"The bundle records a complete history"*, sha1.
- `git bundle list-heads`: **exactly ONE ref** —
  `326e1f0d74e71be17934a680550782a35c3d0827 refs/heads/w13/encoder-adapt`. 6,268,564 B.
- ⭐ Independently CLONED from the bundle into a scratch directory and re-checked from
  that clone alone: 9 scored legs present, every one `SCORED: true`, `K_star: 32`,
  `n_corners {row3_pair 60, row4_nway 49, total 109}`, all four controls 0.0,
  `primary_degenerate: false`.
- The 18 GB of caches and checkpoints under `/c/chlu/outputs/w13-encoder-adapt/` are
  deliberately NOT in the bundle; `w13_encoder_adapt/README.md` says so and says where
  they are.
