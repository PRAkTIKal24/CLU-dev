# gen-encoder-unfreeze — experiment-engineer report

**Task + acceptance criterion:** build a training path in which the encoder trains **with the store in the loop**
without breaking the structural no-bypass guarantee, and **smoke it locally end to end** at toy scale.
**Status: done.** ⛔ **Every cluster item: NOT RUN. No job, no job id, no inferred number.**

## ⚠ DOWNSTREAM RECONCILIATION LIST — needs a Hub owner (read this in the first 10 lines)
1. ⛔ **`scripts/csf3/job_gpu_gen.sh` hard-whitelists `ARMS` to `clu store_removed lace knn`** (§1, line 127), so
   `ARMS="frozen unfrozen"` is **refused at submit time**. The script is read-only to me. I worked around it *inside my
   module* (`--arm clu` ⇒ both arms in one process, explicit and tested). **If the Hub wants per-arm workspace dirs,
   the launcher needs `frozen|unfrozen` added to that whitelist — a launcher change I did not make.**
2. ⚠ **`STAGE=decoder` still names `chlu.experiments.exp_gen_decoder`, which does not exist.** `STAGE=encoder`'s module
   (`exp_gen_encoder`) did not exist either until this branch; I filled that one. The `decoder` hole is still open and
   the launcher's own existence check will `die` on it. **Owner needed.**
3. ⭐ **A resume bug the launcher-shaped invocation caught, now fixed — but the CLASS is programme-wide.**
   `skip_if_exists` reused an arm manifest written before a field existed and the run died in the aggregate with
   `KeyError`. On a cluster that is a **resumed job failing in the aggregate**. Fixed here by comparing a `ARM_SCHEMA`
   constant, not only `status == "ok"`. ⚠ **Every other gen module's skip/resume checks `status` alone.** Worth a sweep.
4. ⚠ **R26.7 (`gen_data.data_root` must fail loudly, no default) is still globally unfixed** — it lives in
   `chlu/config.py` / `chlu/data/`, which my task forbids. I implemented the loud failure **on my path only**
   (`exp_gen_encoder.resolve_data_root`). ⛔ Do not read this as R26.7 discharged.

---

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** *the encoder* — frozen in every shipped phase-1 artefact; here it trains with the store in the loop.
- **Laundering control:** the **frozen-encoder arm at matched compute** — same warm-start object, same PRNG key, same
  minibatch sequence, same loss, same step count; the **only** difference is whether the encoder's leaves are in Optax.
- **Falsifies:** the hypothesis that the commitment collapse is caused by the encoder being frozen.
- **Does NOT falsify:** absolute distinct-code counts far below 72 at smoke scale; losing reconstruction MSE to the
  shipped pipeline (a different loop).

## Precondition — verified on disk
```
$ cat .claude/outputs/gen-cluster-smoke/smoke_result.json
  "status": "ok", "git_commit": "22f9dc5",
  runs: encoder/clu_s0 ok(20344190) certificate/clu_s0 ok(20344191) s1 ok(20344192) s2 ok(20344193)
```
Top-level `"status" == "ok"`. ⭐ Gate satisfied; I started.

## ⭐ PREREG
`.claude/outputs/gen-encoder-unfreeze/PREREG.md`, written **before** `exp_gen_encoder.py` had ever executed
(P1…P6). Verdicts in §4.

---

## 1. What I did

### 1.1 `chlu/core/gen_bridge.py` — **+441 lines at EOF, 0 deletions, 0 modifications**
| symbol | what it is |
|---|---|
| `CODE_GRADS` | `("straight_through", "stop")` — how the encoder may receive gradient across the integer bottleneck |
| `store_codes_traced` / `gather_from_codes_traced` | the **jnp/jittable** twins of `address_codes` / `read_from_codes` (the shipped ones return NumPy and cannot sit in a jitted step). ⭐ Bit-equality with the shipped pair is **asserted at runtime**, not assumed |
| `quantise_through_store` | `features -> (decoder_input, pure_gather, codes)`. ⛔ carries the **test-only** `bypass_alpha` hook |
| `assert_no_feature_bypass` | the four-leg check (§2) |
| `realised_channel_bits` | design constant `K·log2(M)` **plus** the width realised on a request set |
| `joint_filter_spec` · `joint_loss_terms` · `train_joint` | **ONE** stage-2 loop with an `unfreeze` flag — the matched-compute control is structural, not bolted on |

### 1.2 `chlu/config.py` — one block at EOF + the three standard wiring hunks
`GenEncoderUnfreezeConfig` appended at the shared EOF anchor, plus the `CHLUConfig` field, the `load_config` branch and
the `save_config` entry — **each inside its own `BEGIN/END gen-encoder-unfreeze` marker**, the idiom every gen spoke
uses, so a concurrent gen spoke's append does not conflict. ⛔ **Declaring the three extra hunks** since the task said
"ONE new block, APPENDED AT EOF": without them the block is unreachable and `save_config`/`load_config` silently drop it.

### 1.3 `chlu/experiments/exp_gen_encoder.py` — new (the module `STAGE=encoder` has always named and never had)
`request → codes → read_from_codes → decoder`, two arms from one shared warm start, fresh output root, skip-if-exists,
per-request **code tuples persisted** (R30's lesson), `--quick` profile.

### 1.4 `tests/test_gen_encoder_unfreeze.py` — new, **29 tests**
§1 frozen-path fingerprints · §2 the bypass check (shown failing, three `alpha`s, plus the `soft_read` trap) ·
§3 the matched-compute control · §4 the config block · §5 the experiment module (loud `data_root`, fresh output root,
the 30-bit quick profile, the budgets-not-matched guard, the stale-schema resume guard, the launcher arm alias).

---

## 2. ⛔⭐ THE GUARANTEE — and the check shown FAILING before it was trusted passing

**The guarantee, restated precisely for a *trained* encoder:** the tensor handed to the decoder is a function of the
`K` integer codes and the store **alone**. Operationally: **any two feature vectors that address the same K modes
produce a BIT-IDENTICAL decoder input.** That is R20's 30-bit channel written as something a test can execute.

`assert_no_feature_bypass` has four legs; **leg (2), the collision test, is the discriminating one**:
1. the traced codes/gather equal the shipped `address_codes` / `read_from_codes` **exactly**;
2. ⭐ **collision** — replace every feature vector by *the modes it addressed* (a provably different tensor with
   provably identical codes) and re-run: the decoder input must come back bit-identical;
3. **jitter** — perturb the features; on rows whose codes did not move, the decoder input must not move;
4. **value purity** — the straight-through surrogate must add **exactly zero** to the value.

### ⭐ Shown FAILING on a deliberately bypassed variant (`bypass_alpha = 0.1`), verbatim:
```
AssertionError: ⛔ NO-BYPASS CHECK FAILED on the joint (unfrozen) path — 3 leg(s):
  - (2) COLLISION: two DIFFERENT feature tensors that address the SAME K codes produced DIFFERENT
        decoder inputs (max |delta| = 0.2649548053741455). ⛔ ... the K*log2(M)-bit channel claim is void.
  - (3) JITTER: perturbing the features by 0.001 moved the decoder input by 0.0003899410367012024
        on rows whose codes did not change
  - (4) VALUE: the decoder's input is NOT the pure codebook gather (max |delta| = 0.365947961807251)
        — a feature tensor is reaching the decoder.
```
Parametrised over `alpha ∈ {1e-3, 1e-1, 1.0}`; the test asserts the **collision** leg is among those that fire.
⭐ Separately, `test_collision_criterion_catches_R27s_named_soft_read_trap` shows the *criterion* catching the Hub's own
named trap: `soft_read` (same arithmetic as the composed read, but it **takes the features**) moves under the collision
construction while `read_from_codes` does not.

### On the real path, on real trained features (512 rows, unfrozen arm, after training):
```
max_abs_delta_decoder_input_vs_pure_gather        0.0
max_abs_delta_same_codes_different_features       0.0     (over 512 rows; the two feature tensors differ by 1.857)
max_abs_delta_under_jitter                        0.0
design_bits 30.0  (K·log2 M = 5·6)   widened: false   store_bytes 524288
```

### ⚠ DECLARED, NOT HIDDEN — the straight-through gradient
The codes are integers, so **there is no differentiable path through them at all**; any encoder training through a
store necessarily uses a *surrogate* gradient (VQ-VAE's straight-through estimator). `"straight_through"` adds
`z − stop_gradient(z)`, which is **numerically zero**: the decoder's input **value** is bit-identical to
`read_from_codes(store, codes)` at every step of training and at evaluation.
⛔ **So the guarantee this programme states in its method section — "a bypass is structurally inexpressible" — is a
statement about the FORWARD VALUE, and it survives unfreezing intact.** The surrogate is a *gradient* route, carries
no forward information, and is ablatable (`code_grad="stop"`, in which case the encoder gets gradient only from the
commitment term). ⭐ **If the Head/Advisor want the method section to say "no gradient path either", that is a stricter
claim than the code makes and it must be written as `code_grad="stop"`, not assumed.** Flagged for a ruling.

**Realised channel width, every configuration run:** `30.0 bits` (`K=5`, `M=64`), identical in both arms, both runs.
⛔ No widening. `--quick` deliberately does **not** shrink `M` for exactly this reason.

---

## 3. ⛔ THE FROZEN PATH IS BIT-IDENTICAL BEFORE AND AFTER
Fingerprints recorded on the **unmodified** tree (`main @ 994ca9b`) by
`.claude/scratch/gen-encoder-unfreeze/baseline_fingerprint.py` **before a line of the block was written**, then
re-measured on the modified tree:
```
FROZEN-PATH FINGERPRINT DIFF: NONE — bit-identical on all 12 pinned quantities
```
(encoder/decoder-v1/decoder-baseline/store parameter hashes, `feat_shape`, well sizes, `codes_sum`, `feat_sum`,
`read_sum`, `decoded_sum`, `bits_per_read`, `store_bytes`.) Pinned as constants in the test file (§1 of the tests), so
a future edit that moves the frozen path goes red. `git diff --stat` on the two tracked files: **584 insertions,
0 deletions.**

---

## 4. ⭐ THE ACCEPTANCE CRITERION — the local smoke, end to end

**It is a SMOKE.** ⛔ Every number below is void as a reported result.

**Scale:** Shapes3D, Montero `recomb2element`, seed 0, **CPU**, `--quick`: `n_train = 512` train rows (0 held-out rows
fitted, asserted), warm start **40** pretext steps @ batch 16, stage 2 **60** joint steps @ batch 16, `D = 2048`
(`128×4×4`), `K = 5`, `M = 64` ⇒ **30 bits**, decoder recipe **`v1_capacity_schedule`** (asserted). Wall clock **168 s**
for both arms including data streaming. Exit 0.

### 4.1 Matched compute — stated for both arms
| | frozen | unfrozen |
|---|---|---|
| warm-start steps (shared object) | 40 | 40 |
| stage-2 gradient steps | **60** | **60** |
| batch | **16** | **16** |
| images seen in stage 2 | **960** | **960** |
| PRNG key / minibatch sequence | identical | identical |
| loss function | identical | identical |
| parameters **updated** per step | 1,647,299 | 2,075,875 |
| encoder changed (param hash) | **false** | **true** |
| stage-2 wall clock | 30.9 s | 37.8 s |
⭐ The step and data budgets are **exactly** matched; the unfrozen arm updates strictly **more parameters per step** —
that *is* the treatment. `budgets_matched: true` is written into `seed_summary.json` and the frozen arm's encoder hash
is asserted unchanged inside `train_joint` (it raises otherwise).

### 4.2 ⭐ Distinct-code count on B **and** its matched-seen (MS) companion, both arms
| address-book fit rows | arm | **distinct(B)** | /72 | **distinct(MS)** | /n | mean conf. B | feature-side B | feature-side MS |
|---|---|---|---|---|---|---|---|---|
| 512 (= the fitting rows) | frozen | **30** | 72 | **46** | 216 | 0.2169 | 8 | 14 |
| 512 | **unfrozen** | **1** | 72 | **1** | 216 | 0.0438 | **20** | **37** |
| 16,384 | frozen | **7** | 72 | **10** | 209 | 0.6140 | 8 | 14 |
| 16,384 | **unfrozen** | **1** | 72 | **1** | 209 | 0.1515 | **20** | **37** |

⚠ Two address-book budgets are shown because the request-side count is an **estimate** over `Σ_f V_f · K · M` counting
cells; at 512 rows those cells are empty, the Dirichlet smoothing dominates and the argmax degenerates — which *looks*
like a commitment collapse and is not. Training is bit-identical between the two rows of each arm (the feature-side
counts are identical, 8/14 and 20/37, which is the internal check that only the book changed).
⭐ **At 16,384 book rows the frozen arm reproduces the published collapse magnitude: 7 of 72, against R29 Q2's 9 of 72.**

### 4.3 ⭐⭐ THE RESULT — **unfreezing does NOT lift the distinct-code count on B. It lowers it.**
`unfrozen − frozen` on B: **−29** (512-row book) and **−6** (16,384-row book); on MS **−45** and **−9**; ⭐ sign
identical on B and MS, as P2 predicted, i.e. the effect is a property of the **colour corner**, not of held-out-ness.
⛔ **P1 is FALSIFIED at this scale, in the opposite direction, at matched compute, at two independent book budgets.**
Per the task's own framing: *"If unfreezing does not lift the distinct-code count on B, say so — that is a finding and
it narrows the problem."* **It does not. Reported.**

⭐ **And the two counts move in OPPOSITE directions**, which is the substantive part:
- **feature-side** (images → codes; no address book involved): **8 → 20 on B**, **14 → 37 on MS** — the unfrozen
  encoder+store spread real images over **2.5×** as many code tuples.
- **request-side** (request → address book → codes): **7 → 1**.
⇒ the unfrozen encoder makes the *store* discriminate images **better** and the *addressing head* **worse**.

### 4.4 The disk-only tuple analysis the persisted tuples now make free (R30's fix in action)
`.claude/outputs/gen-encoder-unfreeze/code_tuple_analysis.json`, 16,384-row book:
| arm | req-B tuples that are a realised feature code on **seen** (MS) rows | on B rows | req-B ∩ req-MS |
|---|---|---|---|
| frozen | **0 / 7** | 0 / 7 | **7 / 7** |
| unfrozen | **0 / 1** | 0 / 1 | **1 / 1** |
⭐ In **both** arms the codes the address book commits to are (a) **off-manifold** — not one of them is ever produced by
a real image, seen or held out — and (b) **shape-blind** — the B requests and the matched-seen requests land on the
*same* tuples. At smoke scale that is an **addressing** failure, not a capacity failure (30 bits ≫ 72 codes), which is
the same direction as R29 Q1's *"the gap is ADDRESSING"*.
⛔ **60 joint steps and 512 training images. A direction, not a result.** The cluster run is what decides it.

### 4.5 Reproducibility
The default-configuration run was executed **twice** into two fresh directories; every summary field except wall clock
is **identical** (`frozen IDENTICAL / unfrozen IDENTICAL / warm True`). The second run was after the `book_rows`
refactor, so it doubles as the regression check that the refactor preserved the default path exactly.

---

## 5. Flag provenance (mandatory table)
| | value |
|---|---|
| commit | **`6190629`** (branch tip); the measurements in §4 were produced at `87329fe` and re-verified identical at `6190629` |
| JAX | **0.9.0** (⭐ the MAIN venv, reused per protocol §4 — no worktree `uv sync`, so no 0.10.2 drift) |
| seeds | `0` (single seed; ⛔ a smoke, not a 3-seed measurement) |
| dataset / condition | `shapes3d` / `recomb2element` (Montero), `pretext=recon` (R1) |
| `--quick` | **ON** (⛔ every number void as a result) |
| `n_train` / `book_rows` | 512 / `None` (=512) and 16,384 |
| warm start | `enc_steps=40`, `enc_batch=16`, Adam `lr=1e-3` |
| stage 2 | `joint_steps=60`, `joint_batch=16`, `joint_lr=1e-3`, Adam |
| `code_grad` | `straight_through` |
| `beta_codebook` / `beta_commitment` | `1.0` / `0.25` |
| `k_wells` / `n_modes` / channel | `5` / `64` / **30.0 bits**, `store_bytes = 524,288 B` |
| `dec_recipe` | **`v1_capacity_schedule`** (⛔ NOT the `"baseline"` default), asserted per run |
| `max_matched_seen` | 256 (realised MS: 216 / 209) |
| `bypass_alpha` | `0.0` everywhere except the named falsifier tests |
| data root | `$CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen` |
| device | CPU (laptop), 168 s / 252 s per run |

---

## 6. ⛔ What this configuration VOIDS if it is ever promoted (R18, not optional)
R18 is explicit: the store trains on cached features from the **frozen** encoder and the decoder is downstream. ⭐ **A
changed encoder changes the feature cache and voids every downstream number in this phase.** This spoke therefore
writes only to `.claude/outputs/gen-encoder-unfreeze/runs/…`; ⛔ **nothing existing was re-fitted, overwritten or
invalidated** (verified: the only files created are the 3 run trees + logs listed in §8).

**Every number that would have to be RE-MEASURED if this configuration replaced the frozen one:**
1. **The decoder ceiling / kill-rule-(i) statistic** — R22: baseline `0.7191 ± 0.0305`, **V1 `0.9414 ± 0.0179`**,
   AUROC `0.9886`, TPR `0.9398`, TNR `0.9429`. Fitted against the frozen features.
2. **The wide-configuration gate table** — R23 ruling 3: 11-cell headline `0.9894 ± 0.0021` (seen `0.9861`, ratio
   `1.0034`, AUROC `0.9993`), `montero` `1.0000`, `wide-000` `0.8542 ± 0.0967`, worst-of-11 `wide-001` `0.9470`.
3. **The dSprites gate** — R37: `0.9616` (per seed `0.9518 / 0.9751 / 0.9579`), AUROC `0.9991`, ratio `1.007`.
4. **Every store number** — R19.2 / R20: `delete_zero_delta_aggregate = +0.4752`; correlation − random `+0.1504`
   aggregate / **`+0.2015`** on `shape` (R23.4's read-axis `+0.2008` at 12× spread); R23.5's in-corner store read
   **`0.6866`** against the `0.9414` real-feature ceiling. ⭐ `+0.4752` survives a *decoder* change (Head, R20); it does
   **not** survive an *encoder* change.
5. **The whole headline table** — R37: `knn_proto` `0.9119 / 0.8884`, `lace/ld_corrected` `0.6605`, `clu`
   `0.6568 / 0.6759`, `lace(ode)` `0.6347`, `lace/ld_published` `0.5580`, `ebm_compose` `0.5408`, `store_removed`
   `0.5000`, ceiling `0.9357 / 0.9414` — **every row**, because every arm decodes through the one frozen decoder that
   was fitted against the frozen features.
6. **Kill rule (ii)'s verdict** — R29 Q2: `CLU-wide − knn_proto = −0.1400`, 0/3 seeds; and R27.1's channel-matched
   lookup at **chance `0.5206`**.
7. **The commitment collapse itself** — R29 Q2's **9 of 72**, and R29 Q1's `0.510` per K-tuple / `0.871` per well.
8. **The certificate** — R26: barrier `0.9146` vs naive `0.5630`; attribution up to `1.000` vs chance `0.167`; R27.2's
   oracle `0.8373 ± 0.0109` and deployable `0.7998 ± 0.0443`; the per-kind rival table (kNN oor `0.8045` / dup
   `0.3907`; LACE oor `0.3971` / dup `0.9991`) — all computed on frozen features.
9. **DCI-C's invariance qualification** — R28: relabelling `1.28e-16`, class-count `0.367 / 0.019`, 6/6 sign agreement,
   magnitude ratio `1.34`. Measured on the frozen features; the axis would need re-qualifying.
10. **Artefacts, not just numbers** — the feature caches (seed 0 `full`, seeds 1–2 `used`), `store_plan.json`, the
    per-seed groupings and their agreement statistics, `montero`/`wide` bridge checkpoints, the byte ledger
    (`524,288 B` store + the certificate's ~74 KB), R24's grouping re-derivation, R34's navigation numbers.
11. **LACE's whitening feasibility** — R9.1: exactness is a property of the feature covariance at `D = 2048`. A new
    encoder ⇒ re-fit the shrinkage Gaussian and re-run R27.3's exactness gate per seed.

**⛔ What is NOT voided:** R11's split counts (Shapes3D 72 / dSprites 6,292), the C set + A7 control, the request sets —
all defined in **factor space** and encoder-independent (R2/R8). The probe's *instrument* caveats (R37's dSprites
`orientation` at 0.15) are dataset facts and also survive.

---

## 7. ⛔ CLUSTER: **NOT RUN — every item**
| item | status |
|---|---|
| `STAGE=encoder` real run on CSF3 (any seed) | ⛔ **NOT RUN** |
| `--quick` canary through `exp_gen_encoder` on CSF3 | ⛔ **NOT RUN** |
| `PLAN=1` dry run on a login node | ⛔ **NOT RUN** |
| dataset staging onto `$SCRATCH` | ⛔ **NOT RUN** |
| GPU preflight / wallclock sizing for this stage | ⛔ **NEVER OBSERVED** |
⛔ Per the Head's 2026-09-14 scope change I submitted nothing. **There is no job id anywhere in this report.**

### 7.1 ⚠ The input-path gap for THIS stage (folding `gen-launcher-fixes` §3.3)
Unlike `certificate`, this stage needs **no bridge/store artefacts** — it fits everything itself, and `STAGE=encoder`
has **no upstream** in `job_gpu_gen.sh`, so ⛔ **no `ALLOW_MISSING_UPSTREAM=1` is needed or wanted**.
It *does* need the **dataset**, and `push_repo.sh` excludes `.claude/`, so `3dshapes.h5` (255 MB) is **not** on the
cluster. ⭐ **It is the only file this stage needs** (`build_split → load_labels`, then `stream_images`; the C set and
A7 are not read by this module). ⭐ `resolve_data_root` **fails loudly** if it is absent (R26.7) — a missing dataset
cannot become a null result here.

### 7.2 ⭐ THE SUBMIT LINES — one literal command line each
⭐ The whole `--export` value is **ONE quoted shell word**; `sbatch --export` splits on **commas only**.
⛔ Do not build it in a variable and expand it — zsh does not word-split an expanded variable and that has submitted
garbage on this programme before. Every submitting line carries `GEN_GIT_COMMIT=$(git rev-parse HEAD)`.
⚠ `EXTRA_ARGS` contains spaces and is passed as one comma-delimited field; `gen-launcher-fixes` §4.4 verified locally
that it reaches the payload intact and is recorded verbatim in `run_manifest.json`.

**(0) laptop — push the code, then stage the ONE data file (`push_repo.sh` will not carry it):**
```bash
export GEN_GIT_COMMIT=$(git rev-parse HEAD); export CSF3_USER=<username>; scripts/csf3/push_repo.sh
rsync -avP .claude/data/gen/shapes3d/3dshapes.h5 $CSF3_USER@csf3.itservices.manchester.ac.uk:~/scratch/CHLU-gen/data/shapes3d/
```
**(1) login node — PLAN first (prints the exact `-a` range; submits nothing):**
```bash
DATASET=shapes3d PRETEXT=recon STAGE=encoder ARMS=clu SEEDS="0 1 2" PLAN=1 bash scripts/csf3/job_gpu_gen.sh
```
**(2) the canary — ONE seed, `--quick`, real module, real data path (⭐ do this before the real run):**
```bash
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=encoder,ARMS=clu,SEEDS=0,EXTRA_ARGS=--arm clu --quick --data-root /mnt/iusers01/<path>/scratch/CHLU-gen/data/shapes3d,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -a 0-0 -t 0:30:00 --job-name=clu-gen-unfreeze-canary scripts/csf3/job_gpu_gen.sh
```
**(3) the real run — 3 seeds, both arms per job:**
```bash
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=encoder,ARMS=clu,SEEDS=0 1 2,EXTRA_ARGS=--arm clu --data-root /mnt/iusers01/<path>/scratch/CHLU-gen/data/shapes3d --book-rows 65536,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -a 0-2%3 -t 04:00:00 --job-name=clu-gen-unfreeze scripts/csf3/job_gpu_gen.sh
```
**(4) before walking away:**
```bash
squeue --me; ls -l logs/clu-gen-unfreeze-*.out | wc -l
```
**Realised output paths** (`$SCRATCH = ~/scratch`):
```
~/scratch/CHLU-gen/workspaces/shapes3d_recon/encoder/clu_s{0,1,2}/
    artifacts/seed_<n>/{warm_start.json, seed_summary.json,
                        frozen/{manifest.json,history.json,encoder.pkl,store.pkl,decoder.pkl,codes_*.npy},
                        unfrozen/{...}}
    artifacts/unfreeze_summary.json    logs/stage.log    run_manifest.json
```
⚠ **`-t 04:00:00` is a GUESS.** It is 168 s for the `--quick` profile on a laptop CPU; the full profile is ~10× the
steps at 2× the batch on 8× the rows. ⛔ Nothing about A100 wallclock for this stage has ever been observed — that is
exactly what the canary at `-t 0:30:00` is for. ⛔ `--book-rows 65536` is likewise untested at scale; drop it if the
canary is slow (the request-side count is then reported with its `n_fit_rows`, which every manifest carries).
⛔ `<path>` must be filled in by whoever runs `rsync` — I cannot resolve `$SCRATCH` from here.

---

## 8. How I verified — commands and observed output
| what | command | result |
|---|---|---|
| frozen-path fingerprint, **pre-edit** | `python .claude/scratch/gen-encoder-unfreeze/baseline_fingerprint.py` on `main @ 994ca9b` | 12 quantities recorded (JAX 0.9.0) |
| frozen-path fingerprint, **post-edit** | same script, modified tree | **`DIFF: NONE — bit-identical on all 12`** |
| the bypass check **fails on purpose** | direct call, `bypass_alpha=0.1` | `AssertionError` naming legs (2),(3),(4) — verbatim in §2 |
| new tests | `pytest tests/test_gen_encoder_unfreeze.py -q` | **29 passed in 23.05 s** |
| targeted regression | `pytest tests/test_gen_bridge.py tests/test_gen_store.py tests/test_gen_compose.py tests/test_gen_bridge_dsprites.py tests/test_gen_launcher.py tests/test_config.py -q` | **166 passed, 0 failed, 5 warnings (266.59 s)** |
| lint | `ruff check chlu/ tests/test_gen_encoder_unfreeze.py` | **All checks passed** |
| the smoke | `python -m chlu.experiments.exp_gen_encoder … --quick` | **exit 0**, 168 s, artefacts written |
| reproducibility | same command into a fresh dir, after the `book_rows` refactor | `frozen IDENTICAL / unfrozen IDENTICAL / warm True` |
| book-budget diagnostic | `… --book-rows 16384 --quick` | exit 0, 252 s |
| ⭐ **the launcher-shaped invocation** | `python -m … --stage encoder --arm clu --data-root <dir> --quick` | **exit 0**; caught the resume bug (§1 item 3), then reproduced `frozen B=30/72 MS=46/216`, `unfrozen B=1/72 MS=1/216` **exactly** |
⛔ **Full suite NOT started** (task item 9, and the 16 GB machine).
⚠ `ruff format --check` reports the two new files (and `gen_store.py`, `gen_bridge.py`, `exp_gen_compose.py` on
**unmodified `main`**) as "would reformat" — the repo is **not** ruff-format-clean, so I did **not** run
`ruff format`; it would have produced a large unrelated diff. `ruff check` is the bar the repo actually meets.

**Artefacts:** `.claude/outputs/gen-encoder-unfreeze/{PREREG.md, code_tuple_analysis.json, logs/*.log,
runs/shapes3d_recon/, runs/shapes3d_recon_rerun/, runs/shapes3d_recon_book16384/}` and
`.claude/scratch/gen-encoder-unfreeze/{baseline_fingerprint.py, baseline_fingerprint.json, after_fingerprint.json}`.

## 9. PREREG verdicts
| | prediction | verdict |
|---|---|---|
| **P1** | unfrozen ≥ frozen on distinct(B), small strict lift `+1…+4` | ⛔ **FALSIFIED** — `−6` (16,384-row book) / `−29` (512-row book), at matched compute |
| **P2** | B and MS move together, `|Δ_B − Δ_MS| ≤ 3` | ⚠ **half held**: signs agree (both negative, both budgets); the magnitude bound fails (`|−6 − −9| = 3` ✅ at 16,384; `|−29 − −45| = 16` ✗ at 512, where the book is the binding constraint) |
| **P3** | channel `= K·log2(M)` exactly, distinct ≤ `M^K` | ✅ **HELD** — `30.0` bits both arms, both runs, `widened: false` |
| **P4** | the bypass check raises for `alpha > 0`, passes at `0` | ✅ **HELD** — §2, three alphas, collision leg fires |
| **P5** | frozen arm's encoder hash unchanged; budgets matched | ✅ **HELD** — `encoder_changed: false`, `budgets_matched: true` |
| **P6** | frozen path bit-identical | ✅ **HELD** — 12/12 |
⭐ P1 was pre-registered as "the distinct-code count on B", which in R29's sense is **request-side**; the feature-side
companion (which moved the *other* way, `8 → 20`) was **not** pre-registered and is reported as a companion, not as a
rescue of P1.

## 10. Git footprint
- **Branch** `agent/experiment-engineer/gen-encoder-unfreeze`, base **`main @ 994ca9b`**, worktree
  `../CHLU-gen-unfreeze` (fresh; verified **from the main repo** that `main..<branch>` shows all four commits).
- `ebdf45f` `[experiment-engineer] add the unfrozen path beside the frozen one, with a bypass check that can fail`
- `e266be2` `[experiment-engineer] add GenEncoderUnfreezeConfig (one block, appended at EOF)`
- `87329fe` `[experiment-engineer] exp_gen_encoder: the two-arm unfreeze harness + its falsifiable tests`
- `6190629` `[experiment-engineer] resume must compare the SCHEMA, not only the status`
- **Files:** `chlu/core/gen_bridge.py` (+441/−0, EOF block) · `chlu/config.py` (+143/−0: EOF block + 3 marked wiring
  hunks) · `chlu/experiments/exp_gen_encoder.py` (new) · `tests/test_gen_encoder_unfreeze.py` (new, 29 tests). **Branch tip: `6190629`.**
- `git rebase main` → *"up to date"* (main did not move). ⛔ **Not pushed, no PR.** Working tree clean; worktree
  removed **after** verifying from the main repo that `main..agent/experiment-engineer/gen-encoder-unfreeze` shows all
  **4** commits (the w4 lost-commits precedent). Whole-branch diff: **1,848 insertions, 0 deletions.**
- ⚠ **Two concurrent gen spokes were live during this run**: `../CHLU-gen-attrib` on
  `agent/experiment-engineer/gen-loss-attribution` (also at `994ca9b`), and something writing
  `.claude/outputs/gen-encoder-pretrain/` — ⭐ **adjacent to this spoke's subject matter; the Hub should read the two
  reports together.** **Overlap risk: `chlu/config.py`'s three wiring anchors.** My hunks are inside
  `BEGIN/END gen-encoder-unfreeze` markers immediately after the `gen-bridge-dsprites` markers, which is the idiom
  that has produced zero conflicts across four concurrent gen spokes (R19). ⛔ I did not touch the shared checkout.

## 11. Open questions / risks
1. ⭐ **A ruling is needed on the surrogate-gradient wording** (§2). The shipped guarantee is a **forward-value**
   guarantee and it survives unfreezing bit-exactly. It is **not** "no gradient path" — with integer codes that is
   only achievable by `code_grad="stop"`, which trains the encoder from the commitment term alone. **Which sentence
   goes in the method section is the Head's, not mine.**
2. ⛔ **One seed, 60 joint steps.** The §4.3 finding is a *direction*, not a result. The cluster run decides it.
3. ⚠ **The frozen arm here is NOT the shipped pipeline.** It is the identical joint loop with the encoder out of the
   optimiser — the right control *for this experiment*. ⛔ Its numbers are not comparable to `gen-store-grouping`'s or
   `gen-decoder-lift`'s and must never be quoted against them.
4. **If the collapse is addressing, unfreezing the encoder is the wrong lever.** §4.4 says the committed codes are
   off-manifold and shape-blind in **both** arms. The obvious next spoke is an **addressing** one (train the request →
   code head against realised codes, or commit through a rule that can only emit realised tuples), not a bigger
   encoder run. ⛔ That is a scoping opinion, not a result.
5. ⚠ `--book-rows` is a new knob with a `None` default that preserves existing behaviour exactly (verified by the
   bit-identical re-run). It changes the MS set when used (MS excludes the book's rows), which is why MS is 216 vs 209
   between the two runs — ⛔ not a bug, and the counts are in every manifest.

---

## Proposed handover updates (for the Hub)
1. **§7 new entry — `job_gpu_gen.sh`'s `ARMS` whitelist cannot express non-cell arms.** `clu|store_removed|lace|knn`
   only; `ARMS="frozen unfrozen"` dies at submit time. Any future spoke with its own arm names hits this. Current
   workaround: the arm axis is collapsed inside the payload module (`--arm clu` ⇒ both). **Owner: whoever next edits
   the launcher.**
2. **§7 new entry — `STAGE=decoder` names `chlu.experiments.exp_gen_decoder`, which does not exist.** The launcher's
   own module-existence check will `die` on it. `STAGE=encoder`'s hole is now filled by this branch.
3. **§7 amend — R26.7 is only *locally* discharged.** `exp_gen_encoder.resolve_data_root` fails loudly; the shared
   `GenDataConfig.data_root` default is unchanged and still silently resolves to a path that exists on neither a
   worktree nor a cluster node. ⛔ Do not mark R26.7 done.
4. **§3 config note — new block `gen_encoder_unfreeze` (`GenEncoderUnfreezeConfig`)**, defaults preserve all existing
   behaviour, `dec_recipe` pinned to `v1_capacity_schedule` (⚠ `gen_bridge.dec_recipe` still defaults to the
   gate-failing `"baseline"` — that trap is still live for every other entry point).
5. **Rulings ledger — a candidate row.** *"Unfreezing the encoder with the store in the loop does not lift the
   distinct-code count on B at smoke scale; at matched compute it lowers it (7→1 of 72), while the feature-side count
   rises (8→20). The collapse is in the ADDRESS BOOK, not the store's capacity."* ⛔ **Smoke scale, one seed — this is
   a candidate for the negatives registry only after the cluster run, not now.**
6. **Method-section wording needs a Head ruling** — see §11.1 (forward-value guarantee vs "no gradient path").
7. ⭐ **§7-style sweep worth scheduling: skip/resume across the gen modules checks `status == "ok"` alone.** This spoke
   hit the failure mode for real (reused a pre-field manifest ⇒ `KeyError` in the aggregate) and now pins an
   `ARM_SCHEMA`. ⛔ A resumed cluster job is exactly where this bites and exactly where it is most expensive.
8. ⚠ **`.claude/outputs/gen-encoder-pretrain/` is being written by a concurrent spoke** whose subject overlaps this
   one (encoder pretext). The Hub should reconcile the two before either result is quoted.
