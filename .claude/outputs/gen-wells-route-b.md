# gen-wells-route-b — experiment-engineer report

**Task + acceptance criterion:** learn the well placement with the store in the loop, under a between-well
factorisation + within-well multimodality objective, scored **only** as a margin over a D-matched random-placement
floor, and deliver a **submit-ready CSF3 long job** with a resume path demonstrated by killing it.
**Status: done** (every cluster leg `NOT RUN` — CSF3 needs 2FA and I cannot reach it; see *What is NOT run*).

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 4 items, needs a Hub owner (protocol §5 corollary, first-10-lines rule)
> 1. ⛔ **`3dshapes.h5` IS NOT IN THE STAGED ARTEFACT SET, and this job needs it.** `STAGED.json` stages 44 files /
>    1.95 GiB; under `data/gen/` it stages **exactly one** — `shapes3d/c_invalid.json`. `build_split`/`factor_labels`
>    read the label array from the h5 and `_assert_cache_reencodes` streams images from it. ⛔ I did **not** improvise
>    a staging path (the task forbids it and a peer owns that file set); the jobscript now **names the missing file at
>    job start**. **Owner: Hub → `gen-cluster-stage-artefacts`, before leg 3b is submitted.**
> 2. ⛔ **`chlu/core/gen_store.py::train_store` recompiles its jitted step on EVERY iteration** — a repo-wide
>    performance defect affecting every gen spoke, not just this one. `tau` is a **Python float** passed into an
>    `eqx.filter_jit`-ed `step` (`gen_store.py:788-798`), and `filter_jit` treats non-arrays as **static**, so each
>    annealed `tau` is a fresh trace+compile. ⭐ Fixing the identical bug in my own code took the placement loop from
>    seconds-per-step to ~0.05 s/step. ⛔ I did not touch `gen_store.py` — not my file this wave.
>    **Owner: Hub → whoever holds `gen_store.py` next.** One-line fix: `jnp.asarray(tau)`.
> 3. ⚠ **The task file's "partition inherited from `k`-means" is WRONG and would have propagated.** The shipped
>    partition is **hierarchical linkage** (`correlation_grouping`, `method_used: "ward"` in the store plan);
>    **k-means++ seeds the MODES**, not the partition. Corrected in `PREREG.md` §0a and in the module docstring.
>    **Owner: Hub, at the next task/table review.**
> 4. ⚠ **`GenBridgeConfig.dec_recipe` still defaults to `"baseline"`** — the decoder that FAILED the gate. My block
>    defaults to the winner and asserts it by name **and** per-seed hash, but the trap is still armed for anyone who
>    forgets. **Owner: Hub (standing item, already §7-adjacent).**

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)

- **Dial:** ⭐ **WHERE THE WELLS GO** — well placement learned with the store in the loop, replacing the inherited
  partition. ⛔ Not the read, not the decoder, not the encoder.
- **Laundering control:** ⭐⭐ **D-matched RANDOM placement at matched bytes** — built and measured **FIRST**; the
  harness refuses to score a learned arm until the floor artefact exists on disk.
- **Falsifies:** the claim that our wells are semantic. ⚠ It may show placement was never the lever.
- **Does NOT falsify:** failing to recover `scale`/`orientation` information that `gen-loss-attribution` already
  localised at the **codebook** (90.5 % / 100 %) rather than at the partition. That is a known confirmation of a
  measured result, not news.

## ⛔ MECHANICAL PRECONDITION — checked FIRST

`.claude/outputs/gen-cluster-stage-artefacts/STAGED.json`: **`"status": "complete"`, `"checksum_match": true`**,
`n_files: 44`, `n_missing: 0`, `n_checksum_mismatch: 0`, `n_load_failed: 0`, 2,097,001,955 B.
⭐ **PRECONDITION MET** — the task proceeded. (⚠ But see reconciliation item 1: *complete* means every file in the
manifest is present and checksums; it does **not** mean the file set is sufficient for this job.)

---

## What I did

1. **Filed `PREREG.md` before any measurement** — the objective, both arms, the random-placement floor, the metric
   with its degenerate case, and **what result would refute semantic placement**. It commits to the **null**
   (§6b) and says why. `PREREG-addendum-1.md` records the one optimisation decision taken during the build.
2. **`chlu/core/gen_wells.py` (new)** — the placement relaxation, the D-matched floor, the size-constrained
   hardening, the label-free objective + the declared weakly-supervised term, the label-free gate, the byte ledger,
   the cache-coverage gate, the address-book sharpness metric, the collapse guard, and a
   checkpoint/resume implementation following `pilot-checkpoint-resume` (⛔ not a second invented pattern).
3. **`chlu/experiments/exp_gen_wells_route_b.py` (new)** — the harness: preflight → **floor first** → the other arms
   → margins + collapse verdicts.
4. **`tests/test_gen_wells.py` (new)** — 44 tests, weighted towards the guards that must **fail on purpose**.
5. **`scripts/csf3/env.sh`** — the artefact-root export block (the inherited item), and
   **`scripts/csf3/job_gpu_wells.sh` (new)** — the first jobscript in that directory with checkpoint/resume logic.
6. **`chlu/config.py`** — one appended EOF block; **`chlu/cli/experiment_cmd.py`** — one marker-delimited region.

## How I verified

### Targeted tests (acceptance 8)

```
$ CHLU_GEN_DATA_ROOT=…/.claude/data/gen CHLU_CLAUDE_ROOT=…/.claude \
  .venv/bin/python -m pytest tests/test_gen_wells.py -q
2 failed, 42 passed in 70.22s
```
⛔ **Both failures were my own over-tight assertions, not code defects**, and both are now corrected (re-run below):
- `test_random_codes_score_near_zero_and_a_perfect_code_scores_one` — a *perfect* code scored **0.8830**, not
  `> 0.95`. ⭐ Correct behaviour: at `smoothing = 1.0` Laplace adds `M·V = 16` pseudo-counts against `N = 400` real
  ones, a ~4 % dilution that lands directly in the conditional entropy. The test now measures the delta end at
  negligible smoothing (`> 0.99`) **and** pins the dilution at the shipped smoothing (`0.85 < S < 0.99`) — which is
  worth knowing before anyone reads a sharpness of 0.88 as "not quite a delta".
- `test_the_weak_sup_term_survives_JIT` — `-4.8398e-06` vs `-4.8075e-06` under `rel=1e-6`. ⭐ On an untrained state
  the term sits at ~`5e-6`, where float32 accumulation order (which XLA may reorder under `jit`) moves the last
  digits; a *relative* tolerance there is meaningless. Now `abs=1e-6`. The claim under test is "the jitted and
  un-jitted paths compute the same quantity", not bitwise equality.

Re-run after the corrections:
```
$ … -m pytest tests/test_gen_wells.py -q -k "perfect_code or survives_JIT"
2 passed, 42 deselected in 21.54s
```
⭐ **passed / skipped / failed = 44 / 0 / 0** on `tests/test_gen_wells.py`
(42 green on the first pass + the 2 corrected assertions).
⛔ **No full suite was run** — 16 GB box, and a peer spoke (`exp_gen_atom_mean`) was holding CPU throughout.

### ⭐ The gates, each seen to FIRE

| gate | acquits | ⛔ convicts |
|---|---|---|
| **label-free** | `learned_label_free`: loss **and gradients bit-identical** under a label permutation (`abs_loss_delta = 0.0`, `max_abs_grad_delta = 0.0`), signature admits no labels | `learned_weak_sup`: `\|Δloss\| = 9.567e-03`, `max\|Δgrad\| = 9.721e-04` ⇒ **`LEAKY-AS-DECLARED`** — ⭐ the check has teeth |
| **byte ledger** | 524,288 B = `M·D·4`, coupling identity holds | raises `BYTE LEDGER MISMATCH` on a 4-byte perturbation |
| **cache coverage** | seed 0, `scope="full"`: 3144/3144, 0 shortfall | names `SEED 1`, `9288`, `32768` and **dies at startup** |
| **collapse** | `occupancy held`, ratio 1.000 | reports `⛔ COLLAPSE` when a well is emptied |
| **`env.sh` roots** | derives all five, honours an explicit `--export` | prints `CHLU_GEN_DECLIFT_DIR -> <path>` and `exit 1` |

### ⭐ The verification run, end-to-end (acceptance 3) — ⛔ NOT SCORED

Seed 0, `--quick`, 200 placement steps, arms `random_dmatched` + `learned_label_free`.
⛔ **No number below enters any table**; the harness refuses `--quick --scored`, and `summary.json` carries
`"scored": false` with the verdict `NOT RENDERED`.

Preflight, in order, all on real artefacts:
`artefact roots present` → `plan (VERBATIM): K=5 n_modes=64 D=2048 store_bytes=524,288 (method 'ward')` →
`cache coverage: 3144/3144 rows cached (seed 0, scope='full') — 0 shortfall` →
`cache re-encode check: 64 rows bit-identical through the frozen encoder` →
`decoder: recipe 'v1_capacity_schedule' asserted by NAME and by param hash 01ef91f7…`.

| arm | well sizes | store bytes | used modes | occupancy | coords moved |
|---|---|---|---|---|---|
| `random_dmatched` (**floor**) | [374, 268, 279, 371, 756] | **524,288** | **64.0 / 64** | 5.818 b | — |
| `learned_label_free` | [374, 268, 279, 371, 756] | **524,288** | **64.0 / 64** | 5.820 b | **1503 / 2048 (73.4 %)** |

⭐ Identical well-size histogram ⇒ **D-matched exactly**; identical bytes ⇒ the comparison is byte-matched by
construction, not by luck. ⭐ **Occupancy held (ratio 1.000)** — so this is not the collapse artefact.

**Per-factor sharpness as a margin over the floor** (acceptance 4). ⛔ `n_values_realised` on every row:

| factor | V realised | learned | floor | ΔS |
|---|---|---|---|---|
| `wall_hue` | 10 | 0.2374 | 0.2399 | −0.0025 |
| `floor_hue` | 10 | 0.2069 | 0.2180 | −0.0111 |
| `object_hue` | 10 | 0.1089 | 0.1085 | +0.0004 |
| `shape` | 4 | 0.1061 | 0.1257 | −0.0196 |
| `scale` | 8 | 0.0612 | 0.0617 | −0.0005 |
| `orientation` | 15 | 0.0530 | 0.0506 | +0.0023 |
| **mean** | — | — | — | **−0.0052** |

⚠ **What this is and is not.** It is a *plumbing* result at 2,048 store rows / 1,024 eval rows / 200 placement steps.
⛔ It is **not** the scored run and it does **not** discharge `PREREG.md` §6b. What it does establish is that the
path runs end-to-end on real artefacts and that the machinery can express both outcomes. ⚠ The **direction** is
consistent with the pre-registered null (§6b predicted mean `ΔS < 0.05` with no factor above `+0.10`; observed mean
`−0.0052`, max `+0.0023`) — ⛔ but a `--quick` agreement is not evidence, and it is recorded here as an
indication only.

### ⭐⭐ RESUME, DEMONSTRATED BY KILLING IT (acceptance 7)

⛔ A resume path that has never survived an actual kill is not a resume path. This one was killed.

**(A) uninterrupted reference** — seed 0, `--quick --steps 200`, arms `random_dmatched` + `learned_label_free`,
run to completion; final placement checkpoint `ckpt_learned_label_free_step00000200.eqx`.

**(B) the same config, `kill -9` mid-placement** —
```
python PID = 76363  (/Users/user/Desktop/CHLU/.venv/bin/python)
banked before the kill: 3 (newest: ckpt_learned_label_free_step00000110.json)
…: 76363 Killed: 9   ( exec … exp_gen_wells_route_b … )
⛔ KILLED with -9 at 2026-09-15T15:15:52Z — no clean shutdown, no final write
   process gone: YES
   result artefact written by the killed run? NO — as expected
```

**(C) restarted with `--resume`** — ⭐ it **picks up rather than restarting**:
```
[resume] ⭐ RECOVERED 'learned_label_free' from ckpt_learned_label_free_step00000110:
         resuming at step 110 of 200 (config fingerprint a522c9cf4c1b… matches;
         loss at checkpoint 129.42300415039062; 3 good checkpoint(s) on disk)
```

**The bit-identity check against the uninterrupted run** — the same gate `pilot-checkpoint-resume` passed:

| quantity | uninterrupted (A) | killed + resumed (C) |
|---|---|---|
| final placement checkpoint **sha256** | `f7405dc18bbb3180dc799f9f826796ab518aa053278b909ff20978197758389d` | **identical** |
| `cmp` on the 7,988,232-byte checkpoint | — | ⭐⭐ **BYTE-IDENTICAL** |
| delivered `labels` (all 2048 coordinates) | — | ⭐ **identical** |
| coordinates moved | 1503 | **1503** |
| `resumed_from` / `steps_run` | 0 / 200 | ⭐ **110 / 90** — it really did resume |
| config fingerprint | `a522c9cf4c1b…` | identical |

⭐ **The guarantee is structural, not lucky:** the batch for step `i` is `fold_in(base_key, i)` — a pure function of
`(base_key, i)` and ⛔ **not** of a key carried through the loop — so a run resumed at step `i` is handed the
identical batches in the identical order.

⚠ **Take 1 of this demonstration was INVALID, and it is worth recording why**, because it is the same class of
error the acceptance item exists to catch. `cd X && py … &` backgrounds the whole AND-list **in a subshell**, so
`$!` is the *subshell's* PID; `kill -9 $!` killed the subshell and left python **orphaned**. It ran to completion,
and the "resume" then found a finished step-200 checkpoint and correctly had nothing to do — a green result that
proved nothing. ⭐ Fixed by capturing the python PID with `exec`, and by asserting **`process gone: YES`** and
**"the killed run wrote no result artefact"** before the resume leg is allowed to run. ⛔ A kill harness that does
not verify the kill is as untrustworthy as a resume path that has never been killed.

⚠ Legs B and C used `--store-train-steps 2` (a new flag, plumbing/demo only) so the floor arm's `train_store` did
not dominate a shared box; ⛔ it does not touch the placement, which is computed before and independently of the
store fit, and the reference (A) it is compared against used the same placement config. ⛔ A scored run uses the
config default.

## Findings / results

### 1. ⛔⭐ A null that was manufacturable by a learning rate — caught in the build

The first end-to-end run hardened with **`0/2048` coordinates moved**. That observation is *verbatim* the
pre-registered falsifier ("learned placement does not beat D-matched random placement ⇒ placement was never the
lever") and it would have been reported as the pre-registered null with a straight face. It was an **optimisation
artefact**: the relaxation carries 655,360 mode floats against 10,240 placement logits, and at a shared Adam rate
the modes absorb the gradient. Fixed with a separate logit rate (`placement_logit_lr_mult = 30.0`,
`optax.multi_transform`), declared in `PREREG-addendum-1.md` §A1, and **guarded** by a test that fails if the
relaxation is frozen. After the fix: **73.4 % of coordinates move.**
⭐ **Standing rule for whoever reads the scored run: a "no margin" result is only a finding when
`placement.fraction_moved` is non-zero.** That field is written into every learned arm's record for this reason.

### 2. ⛔⭐ The objective collapsed, and the repo already knew why

With `tau` in **absolute** units the occupancy entropy went to **exactly 0.0000** by step 60 and stayed there —
every row picking one mode per well — with both regularisers **inert** at that point (`loss == rec` to five
decimals). ⛔ A collapsed store is a fixed point of the absolute-`tau` objective: it cannot climb back out.

This is documented in this repo already. `chlu/core/gen_dream.py::mode_distance_scale`:

> *"`exp(-d²/2τ²)` is saturated one-hot whenever `tau` is small against this scale, and a saturated soft occupancy
> has vanishing gradient … Specifying `tau` in absolute units … is therefore a sampler defect, and it is the one
> that froze this module's first run."*

⭐ Fixed the same way, per row: `_soft_occupancy` divides `d²` by **each row's own mean distance to that well's
modes**, making `tau` scale-free in `D`, in `M` and in feature magnitude.

**Measured, on real seed-0 features (4,096 rows, 120 steps), after the fix** — ⛔ the criterion is occupancy alone;
no sharpness, margin or factor label was read:

| `lambda_occ` | used modes / 64 | occupancy H (nats, max 4.159) | coords moved |
|---|---|---|---|
| **0.1 (the declared default)** | 17.2 | **4.0063** | 1552 |
| 1.0 | 17.4 | 4.0128 | 1553 |
| 10.0 | 17.8 | 4.1126 | 1545 |
| 50.0 | 34.2 | 4.1518 | 1556 |

⭐ **The `tau` normalisation alone removed the collapse at every `lambda_occ`**, so the declared default
`lambda_occ = 0.1` was **left unchanged** — ⛔ I did not tune it. (Raising it to 50 roughly doubles the
relaxation's hard-argmax mode usage if the Hub ever wants that; it is not needed to avoid collapse.)

### 3. ⚠ Two silent defects in my own build, both now guarded

- **`--quick` was not quick**: `select_store_rows` reads `config.gen_store.n_store_train`, not my block's, so a
  smoke gathered the full **32,768** rows while logging a perfectly normal `rows: store_train=32768`.
- **The weakly-supervised arm could not survive `jit`**: `weak_supervision_term` inferred its one-hot width with
  `y.max()`, reading a **traced** array. The failure mode is nasty — the *un-jitted* label-free gate passes and logs
  `label-free check FIRED as designed` immediately before the arm dies.

### 4. The metric's second trap, closed in code rather than by a referee

`S` is normalised by `log V`, not `H(f)`, so a factor with a **skewed marginal** scores above zero while carrying no
information. Every row therefore also carries `marginal_only_sharpness` (what `S` would be if the code said
nothing) and `sharpness_above_marginal`. On the shipped cache the marginals are near-uniform
(`marginal_only ≈ 0.002`), so `S ≈ I(f; code_k)/H(f)` — ⭐ but a number that needs another arm to be interpretable
is a number that will eventually be quoted alone.

## ⭐ Flag provenance (mandatory, critique M4)

| item | value |
|---|---|
| harness commits | **`3ab0e6e` · `13d169e` · `8674559` · `65d15b4`**; every number below produced at `65d15b4`'s tree |
| branch / base | `agent/experiment-engineer/gen-wells-route-b` off **`main @ 115d4b5`**; rebase onto `main` is a no-op |
| worktree / venv | `../CHLU-routeb`, **main venv** (`/Users/user/Desktop/CHLU/.venv`), ⛔ no `uv sync`. **JAX 0.9.0** |
| dataset · pretext · condition | `shapes3d` · `recon` · `recomb2element` (Montero) |
| seeds | **0 only** (⚠ seeds 1/2 cannot reach production size — see follow-up 3; ⛔ no `±` is quoted anywhere) |
| encoder | ⛔ **LOADED, never fitted.** `gen-feature-bridge` seed-0 `encoder.pkl`, `98d32602…`, asserted against the bridge manifest **and** the feature-cache manifest |
| decoder | ⭐ **`dec_recipe = "v1_capacity_schedule"`**, asserted by NAME and by param hash **`01ef91f7…`** (⛔ the config default `"baseline"` FAILED the gate) |
| feature cache | 480,000 rows, `scope="full"`, `float16`, `D = 2048`, `feat_shape = (128, 4, 4)`; re-encode bit-identity checked on 64 rows |
| pinned plan (VERBATIM) | `K = 5`, `n_modes = 64`, `D = 2048`, `store_bytes = 524,288`, `bits_per_read = 30.0` |
| `n_store_train` / `n_corr` / `n_eval` | **2,048 / 1,024 / 1,024** (⚠ `--quick`; production is 32,768 / 16,384 / 8,192) |
| placement | `steps = 200` (CLI), `batch = 128`, `lr = 3e-3`, **`logit_lr_mult = 30.0`**, `tau 1.0 → 0.25`, `lambda_occ = 0.1`, `lambda_fact = 1.0`, `warm_start_from_reference = True` |
| store fit (delivered) | `store_train_steps = 30` (`--quick`), `batch = 1024`, `lr = 3e-3`, `beta = 0.25`, `tau 1.0 → 0.10` — ⛔ identical call in every arm |
| arm keys | ⭐ `fold_in` on the arm label (`gen_landscape.arm_key`), ⛔ never a lengthened `split` |
| checkpointing | time-based (`--ckpt-every-seconds`) **and** every 10 steps under `--quick`; `keep_last = 3`; atomic temp+rename |
| ⛔ scored? | **NO.** `summary.json` carries `"scored": false`; the verdict is `NOT RENDERED`; `--quick --scored` is refused |

## ⭐ Acceptance, item by item

| # | item | status |
|---|---|---|
| 1 | `PREREG.md` filed **before** any measurement | ⭐ **done** — objective, both arms, the floor, and the refutation condition; + `PREREG-addendum-1.md` |
| 2 | random-placement floor **built and measured first** | ⭐ **done** — `floor_first/floor_s0.json` written before any learned arm; enforced in code |
| 3 | short verification run, end-to-end, ⛔ no scored verdict | ⭐ **done** — local CPU, seed 0; `"scored": false` |
| 4 | per-factor sharpness with `n_values_realised`, as a margin over the floor | ⭐ **done** — 6 factors, every row carries `V`, margins only |
| 5 | byte ledger asserted in code, aborts on mismatch | ⭐ **done** — 524,288 B both arms; gate seen to raise |
| 6 | one copy-able block of CSF3 submit lines + wall time + resume command | ⭐ **done** — `submit.sh` |
| 7 | ⭐⭐ **resume demonstrated by killing it**, with bit-identity | ⭐ **done** — `kill -9` at step 110/200; resumed at 110; final checkpoint **byte-identical** (sha256 match) |
| 8 | targeted tests green, passed/skipped/failed reported | ⭐ **done** — **44 / 0 / 0** |
| 9 | cost line: card, wall hours, dollars | ⭐ **done** — local CPU $0; CSF3 `gpuA` £0, free at point of use |

## ⛔ What is NOT run, naming the leg

⛔ **CSF3 needs 2FA and I cannot reach it. Every cluster leg is `NOT RUN`:**

| leg | status |
|---|---|
| `push_repo.sh` (ship code + `uv.lock`) | ⛔ **NOT RUN** |
| `stage_artefacts.sh` (⛔ not mine) | ⛔ **NOT RUN** — and **blocked**: `3dshapes.h5` is not in the staged set |
| `setup_env_job.sh` (env build on a compute node) | ⛔ **NOT RUN** |
| `job_gpu_wells.sh` QUICK smoke (leg 3a) | ⛔ **NOT RUN** |
| `job_gpu_wells.sh` SCORED run (leg 3b) | ⛔ **NOT RUN** |
| `job_gpu_wells.sh` RESUME (leg 3c) | ⛔ **NOT RUN** on the cluster (⭐ but the resume path itself **was** killed and resumed locally — §"Resume") |
| artefact sync back | ⛔ **NOT RUN** |

⭐ The copy-able submit block is at `.claude/outputs/gen-wells-route-b/submit.sh`.

## Cost line (acceptance 9)

| leg | card | wall | cost |
|---|---|---|---|
| everything in this report | **local CPU** (Apple Silicon, the main venv) | ~3 h wall, one seed | **$0** |
| CSF3 leg 3b (scored, seed 0) | **A100-SXM4-80GB, `-p gpuA`** | ⚠ **~6–10 h extrapolated**, `-t 2-0` submitted (~5× headroom) | **£0 — free at point of use** |

⚠ ⛔ **The A100-80GB above is CSF3's, and is NOT the GCP A100-80GB that `PREREG.md` §12 records as quota-zero for
this project.** They are different resources; ⛔ neither line may be quoted as the other. **No cloud GPU was used**
— the verification run is local CPU, so the L4/A100-40GB-spot constraint (6 free L4s; ⛔ do not evict `tsfm-pair-a`)
was never engaged. ⚠ The A100 wall estimate is an **extrapolation from CPU**, not a measurement; the generous `-t`
is deliberate, and with checkpointing it costs nothing on a free partition.

## Git footprint

- **Branch:** `agent/experiment-engineer/gen-wells-route-b`, **worktree** `../CHLU-routeb`, base **`main @ 115d4b5`**
  (the Hub's tip — it carries the `gen-cluster-stage-artefacts` merge this task's precondition depends on).
- **Commits** (4, ⭐ **verified from the MAIN repo**, protocol §3.2 — the check that lost 8 commits in wave-4):

| hash | subject |
|---|---|
| `3ab0e6e` | `gen_wells: learned well placement, its D-matched random floor, and the gates` |
| `13d169e` | `the route-B harness: floor FIRST, then everything is a margin over it` |
| `8674559` | `tests: the guards that must fail on purpose, and two that already have` |
| `65d15b4` | `CSF3: a job inherits the artefact root, and a job that dies resumes` |

⭐ `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/gen-wells-route-b` lists all four.
⭐ **Rebase onto the named base is a no-op** — local `main` has not moved from `115d4b5`. ⛔ `origin/main` was never
touched (§7.21: it is frozen at `40c2f31` and rebasing onto it would replay dozens of foreign commits).
⛔ Nothing pushed, no PR opened.

- **Files touched** — exactly the ones I own:

| file | change | deletions |
|---|---|---|
| `chlu/core/gen_wells.py` | **new** | — |
| `chlu/experiments/exp_gen_wells_route_b.py` | **new** | — |
| `tests/test_gen_wells.py` | **new** | — |
| `scripts/csf3/job_gpu_wells.sh` | **new** | — |
| `chlu/config.py` | ONE appended EOF block, `BEGIN/END gen-wells-route-b`, **no `CHLUConfig` field** | **0 semantic** ⚠ |
| `chlu/cli/experiment_cmd.py` | marker-delimited region ×2 | **0** |
| `scripts/csf3/env.sh` | ONE added export block | **0** |

⚠ **`chlu/config.py` reads `162 insertions(+), 1 deletion(-)`, and the "deletion" is not one.** The previous block's
final line `# --- END gen-dsprites-exponent ---…` had **no trailing newline**; adding one makes git re-emit the
*identical* line as a delete+add pair (`\ No newline at end of file`). ⭐ **0 semantic deletions** — verified by
inspecting the diff directly.

⭐ **`chlu/config.py` verified BEHAVIOURALLY, not by presence** (⚠ a presence check that assumes fields reports a
false loss here — it has, twice): the block instantiates, memoises on the config, accepts overrides, survives a
`save_config`/`load_config` round-trip, and the **accessor still resolves on the reloaded config**, while
`gen_wells_route_b` is confirmed **absent** from `CHLUConfig`'s dataclass fields — which is the design.

## Open questions / follow-ups / risks

1. ⛔ **The scored run has not happened.** Everything above is a build plus a `--quick` verification. The
   pre-registered verdict is deliberately `NOT RENDERED`.
2. ⚠ **`warm_start_from_reference = True`** starts the relaxation at the inherited partition. Defensible (route B
   *replaces* that partition) but it is a prior, and a cold-start arm would be a cleaner test of "learn where the
   wells go from nothing". One flag, not run here.
3. ⚠ **Seeds 1 and 2 cannot run at production size** — 9,288 cached rows against 32,768 needed. The gate names it
   and dies. Either re-stage those seeds at `scope="full"` or accept seed 0 only; ⛔ a 3-seed claim is not available
   today, and `±` over one seed is not a spread.
4. ⚠ **`lambda_fact` was never swept.** It is at its declared default and the between-well MI term's contribution is
   unseparated from the reconstruction term's.

---

## Proposed handover updates (for the Hub)

- **§7-CURRENT, new item:** **`gen_store.train_store` recompiles its jitted step every iteration** (`tau` is a Python
  float under `eqx.filter_jit`, `gen_store.py:788-798`). Repo-wide across the gen phase; one-line fix
  (`jnp.asarray(tau)`); measured ~60× on the identical pattern in `gen_wells`. **No owner yet.**
- **§7-CURRENT, new item:** **the staged gen artefact set lacks `3dshapes.h5`**, so every gen job that needs factor
  labels or the cache re-encode check will fail on CSF3. Owner: `gen-cluster-stage-artefacts`.
- **Correction to circulate:** the inherited partition is **ward linkage**, ⛔ not k-means; **k-means++ seeds the
  modes**. The task file for this spoke said k-means and it would have reached a table.
- **New standing rule proposed:** *a placement null is only a finding when `fraction_moved > 0`* — the frozen
  relaxation reproduced the pre-registered falsifier exactly, and nothing in the report format would have caught it.
- **`scripts/csf3/env.sh` now exports and asserts the five `CHLU_*` roots**, so `--export` no longer has to carry
  them by hand and an omission fails at job start rather than after the GPU is allocated.
