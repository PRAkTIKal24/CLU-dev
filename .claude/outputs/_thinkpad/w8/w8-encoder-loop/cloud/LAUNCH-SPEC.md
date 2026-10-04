# w8 hub/encoder-loop — READY-TO-RUN CLOUD LAUNCH SPEC

> ## ⭐⭐ STATUS: **READY TO RUN**, as of 2026-09-20T17:47Z.
> The gate in §0 — the `kill -9` resumability proof — **PASSED**. After a SIGKILL
> mid-training and a relaunch, the resumed leg reached **bit-identical parameters** to an
> uninterrupted leg of the identical configuration (encoder `7b03cb1d3b2f2863…`, store
> `9ccd790f3d7133c3…`, both equal; `resumed_from_step = 2`), and a foreign config
> fingerprint was **refused by name**. Evidence: `dev/kill9/KILL9-PROOF.json`.
> ⛔ **One prerequisite remains and it is not optional: STEP 0, the image rebuild.**
>
> ## ⭐ DISPOSITION, 2026-09-20T23:58Z (Head): **DEFERRED, NOT CANCELLED — harvested for WAVE 9.**
> Step 0 needs the **Docker daemon on the Head's Mac**, which is the Head's to start and was never
> something the hub box could do. Nothing in this spec is stale: the exp-registry row, the
> `--dry-run` acceptance, the digest-pin enforcement and the `kill -9` resumability proof all
> stand as verified. ⛔ The only open item is the daemon, then `build_image.sh`, then re-pin
> `IMAGE_DIGEST`.

**Written by:** `w8 hub/encoder-loop`, on the Windows hub box, 2026-09-20.
**Executed by:** the Head, from the Mac where `gcloud` is authenticated for project `forgisprova`.
**Results return to:** `/c/chlu/outputs/w8-encoder-loop/cloud/`.

> ⛔ **This hub cannot run any of this itself.** `gcloud` is not installed on the hub box and the
> account-level actions (create, poll, fetch, build) need the Head's credentials. Everything below
> has been **built and verified offline on this box**; what is left is the authenticated step.

---

## 0. WHAT IS VERIFIED, AND WHAT IS NOT — read this before spending anything

| claim | status | evidence |
|---|---|---|
| the swap loss backpropagates through store → decoder → judge into the encoder | ✅ **MEASURED** | `dev/smoke_pieces.py`: `\|grad\| = 185968` |
| the whole CPU pipeline runs end to end at small scale | ✅ **MEASURED** | `dev/` legs, see `LOG.md` |
| `kill -9` mid-training RESUMES, bit-identically | ✅ **PROVED 17:47Z** | `dev/kill9/KILL9-PROOF.json`; clean == killed encoder AND store hashes, `resumed_from_step=2` |
| a foreign config fingerprint is REFUSED on resume | ✅ **PROVED 17:47Z** | `FingerprintMismatch` raised by name against `942c3fb5d9ce…` |
| `launch.sh` accepts this spec and its preflight passes | ✅ **DRY-RUN** | §5, run offline against a labelled stub |
| the launcher and the VM share one exp registry row | ✅ **CHECKED** | `startup.sh --check exp w8_encoder_loop` → `chlu.experiments.exp_w8_encoder_loop sweep 1` |
| the live quota at launch time | ⚠ **NOT VERIFIED FROM HERE** | the dry-run used a stub; the Mac reads the real API |
| the pinned image contains this hub's new modules | ⛔ **NO — SEE STEP 0** | they were written today |

### ⛔ STEP 0 IS MANDATORY: the image must be rebuilt

The pinned image
`us-central1-docker.pkg.dev/forgisprova/chlu/chlu-cuda@sha256:010435659b510551b64d2008361ac5c906bf127246d0d4fc678b8c204385c44c`
was built before this hub existed. It does **not** contain:

- `chlu/core/gen_swap_loss.py` (new)
- `chlu/experiments/exp_w8_encoder_loop.py` (new)
- the `w8_encoder_loop` row in `scripts/cloud/startup.sh` (new)

`launch.sh` refuses any image not pinned by digest, and `startup.sh` re-checks the pin on the VM,
so there is no way to slip new code past it — which is correct, and it means the rebuild is not
optional. There is also **no PYTHONPATH shortcut**: the new files live *inside* the `chlu` package,
so an overlay directory on `PYTHONPATH` would shadow the whole package rather than extend it.

```bash
# on the Mac, from a checkout of this hub's branch (w8/encoder-loop)
bash scripts/cloud/build_image.sh              # ~25 min; throwaway e2-standard-8 builder, self-deleting
# it writes the new digest to scripts/cloud/IMAGE_DIGEST — commit that, or pass --image explicitly
```

⚠ `build_image.sh` ships `uv.lock` explicitly (it is gitignored) and **refuses if it differs from
the checked-out commit**. Build from a clean checkout of `w8/encoder-loop`.

---

## 1. WHAT THE GPU IS FOR — and what must never go there

⭐ **GPU = the `train` stage only.** That is not a preference; it is forced by the code.

`chlu/eval/rivals/w4_cache2.device_guard` **refuses to write a byte** of a feature cache unless JAX
sees exactly one device and that device is a **CPU**. The reason is in its own docstring: CPU and
GPU features agree only to ~53 ULP on ~1 % of elements *while the parameter hash still matches*, so
a cache half-built on each is silently void and nothing downstream notices. The cloud README carries
the same warning for `gen_store`/`gen_cell`.

Every stage downstream of the cache reads the cache. So:

| stage | where | why |
|---|---|---|
| `warm` — shared warm start + fixed ownership | ⭐ **GPU** | it is the encoder+decoder fit that `train` starts from |
| `train` — the four arms' stage 2 | ⭐ **GPU** | the treatment's step carries a **second decoder pass and a judge pass**; the decoder alone measures **2.724 s/step** on this box's CPU |
| `cache` — the 480,000-row re-encode | ⛔ **CPU, enforced** | `device_guard` |
| `probe`, `rivals`, the rows-3/4 leg | ⛔ **CPU** | they read the cache |

**What the GPU actually buys.** On CPU this hub is forced down to `joint_steps=400` and a decoder
refit of 400 steps under a renamed short recipe (`w8_v1_short`), and its renders are therefore
**not comparable with the incumbent's absolute row-3/row-4 values**. On an A100 the **full gated
`v1_capacity_schedule` budget (1800 steps)** and a much larger `joint_steps` come into range. That
is the only route by which this hub's numbers join the incumbent's table, which is why it is a
headline case under charter §D and not a supporting one.

---

## 2. THE BINDING CONSTRAINT: A100 concurrency is ONE, not sixteen

The measured quota says A100-40 SPOT limit 16, in use 0. **The GPU metric is not what binds.**

```
metric                                   need    limit     used     free  ok
PREEMPTIBLE_NVIDIA_A100_GPUS                1       16        0       16  yes
A2_CPUS                                    12       12        0       12  yes      <-- EXACTLY full
```

One `a2-highgpu-1g` is 12 vCPU and the region holds 12 `A2_CPUS`. `quota.py`'s own docstring states
it: *"the real concurrency is ONE, not sixteen. A GPU-only check would have reported sixteen and
been wrong every time after the first."*

⭐ **So the three seeds run as three SEQUENTIAL jobs, not three parallel ones.** A parallel launch
does not fail fast — it burns an attempt and returns a stockout-shaped error minutes later.

---

## 3. PRECONDITIONS to check on the Mac before launching

```bash
gcloud config get-value project                      # expect: forgisprova
gcloud storage ls gs://chlu-cloud-forgisprova/artefacts/outputs/gen-external-classifier/
#   ^ THE INVARIANT INSTRUMENT of this hub. If it is not staged, nothing below is scoreable.
gcloud storage ls gs://chlu-cloud-forgisprova/artefacts/outputs/gen-decoder-lift/winner/
#   ^ startup.sh's decoder guard asserts it by name AND by sha256 before touching the GPU.
bash scripts/cloud/stage_gcs.sh --dry-run            # re-stage only if this reports PARTIAL/STALE
bash scripts/cloud/startup.sh --check exp w8_encoder_loop   # expect: ...exp_w8_encoder_loop sweep 1
bash scripts/cloud/startup.sh --check roots
```

---

## 4. ⭐ THE COMMANDS — run these three, in order, one at a time

Each is one seed. **Wait for each to finish before starting the next** (§2).

```bash
export CHLU_W8_CKPT_REMOTE=gs://chlu-cloud-forgisprova/jobs   # optional; the payload also defaults sanely

for SEED in 0 1 2; do
  scripts/cloud/run_until_done.sh \
      --job      "w8enc-s${SEED}" \
      --card     a100-40 --spot \
      --exp      w8_encoder_loop \
      --units    "${SEED}" \
      --artefacts gs://chlu-cloud-forgisprova/artefacts \
      --exp-args "--stage cloud --ckpt-every-s 300 --joint-steps 4000 --dec-steps 1800 --warm-steps 4000 --n-train 32768" \
      --max-hours 6 --max-attempts 8 --max-usd 12.00
done
```

`run_until_done.sh`, not bare `launch.sh`: spot VMs are taken away, and the supervisor relaunches
under an attempt cap **and** a spend cap until `out/result.json` exists. Its exit codes:
`0` done · `1` incomplete · `3` quota refusal · `4` attempt cap · `5` spend cap · `6` two consecutive
payload failures with no pre-emption.

### What each flag is doing, and why it is that value

| flag | value | why |
|---|---|---|
| `--card a100-40 --spot` | | on-demand A100 quota is **0 free**; spot is 16 and the job is built to survive pre-emption |
| `--units "${SEED}"` | one seed per job | §2: concurrency is one, and one unit = one seed = warm + all arms |
| `--stage cloud` | | the module's own entry point: warm start, then every arm's stage 2, each skipped when its artefact exists |
| `--ckpt-every-s 300` | 5 min | the wall-clock checkpoint timer. ⛔ Deliberately **outside** the config fingerprint, so a relaunch may checkpoint more often without being refused |
| `--joint-steps 4000` | vs 400 on CPU | 10× the CPU budget. **Equal for every arm in the launch, by construction** (`apply_compute_overrides`) |
| `--dec-steps 1800` | ⭐ | **the full gated `v1_capacity_schedule` budget** — this is the point of going to the GPU |
| `--warm-steps 4000` | vs 1500 | a better shared warm start; shared bit-identically, so it cannot bias between arms |
| `--n-train 32768` | vs 8192 | 4× the fitting rows |
| `--max-hours 6` | | watchdog; the VM self-deletes. At $1.47/h spot the ceiling is **$8.82/seed** |
| `--max-usd 12.00` | | the supervisor's own cap, above the watchdog ceiling so the watchdog binds first |

⚠ **`--joint-steps` and `--n-train` are inside the config fingerprint.** Changing either between
attempts of the same job makes the checkpoint unresumable *by design* — it will refuse by name
rather than silently resume onto a different configuration. Fix the numbers before the first launch.

### Cost line

| item | rate | ceiling | note |
|---|---|---|---|
| a2-highgpu-1g, A100-40, SPOT | **$1.47/h** | 6 h | list price us-central1, 2026-09-15 |
| per seed | | **$8.82** | |
| **three seeds** | | **$26.46** | sequential; the realised figure will be lower unless pre-empted repeatedly |
| the optional L4 interruption test (§7) | $0.343/h | 1 h | **$0.34** |

---

## 5. THE DRY-RUN, verified on this box

Run offline against `dev/stub_gcloud/gcloud.bat` — a **labelled stub** that answers the single call
`launch.sh --dry-run` makes (`quota.py`'s `gcloud compute regions describe`) with the runbook's
**transcribed** measured quota. ⛔ It proves the launcher's logic, **not** the live quota; the Mac
reads the real API.

```
== plan
   job           w8enc-s0          vm chlu-w8enc-s0
   card          a100-40 (nvidia-tesla-a100) x1 on a2-highgpu-1g [12 vCPU]
   provisioning  SPOT -- expect pre-emption; the job resumes
   run image     ...chlu-cuda@sha256:0104356...385c44c          <-- MUST BE THE REBUILT DIGEST
   experiment    w8_encoder_loop -> chlu.experiments.exp_w8_encoder_loop (out_mode sweep, artefacts 1)
   units         0
   exp args      --stage cloud --ckpt-every-s 300 --joint-steps 4000 --dec-steps 1800
   checkpoints   gs://chlu-cloud-forgisprova/jobs/w8enc-s0/ckpt   every 20s
   sync-back     gs://chlu-cloud-forgisprova/jobs/w8enc-s0/out    every 15s (DURING the run)
   watchdog      6 h, then the VM deletes itself
   est. cost     $1.47/h x 6 h ceiling = $8.82
== --dry-run: nothing created.
```

To reproduce it on the Mac **without** the stub (it will read the real quota):

```bash
scripts/cloud/launch.sh --card a100-40 --spot --job w8enc-s0 --exp w8_encoder_loop \
  --units '0' --artefacts gs://chlu-cloud-forgisprova/artefacts \
  --exp-args '--stage cloud --ckpt-every-s 300 --joint-steps 4000 --dec-steps 1800' \
  --max-hours 6 --dry-run
```

---

## 6. ⭐ CHECKPOINT-FIRST, and the `kill -9` proof that gates this spec

**The mechanism is `scripts/cloud/ckpt.py` — THE checkpointer.** The cloud README forbids a second
one in terms, and this hub does not write one. What it gives:

- a **wall-clock** timer (`--ckpt-every-s`), not per-epoch;
- **atomic** twice: local temp + `os.replace`, then an atomic object create in GCS;
- **persisted off the VM** — a save is not complete until the object exists in the bucket;
- **keep-last-3**, pruned across *both* stores;
- a resume that **verifies a config fingerprint and refuses by name**;
- **sync-back every 15 s while the job runs**, from the host loop *and* the payload — because a
  pre-empted VM never reaches its last write.

**The fingerprint** (`train_fingerprint`) carries the arm, the seed, the step budget, every loss
weight, and ⭐ **the warm start's three parameter hashes and the fixed ownership vector** — so a
checkpoint cannot be resumed onto a different warm start, which is the one substitution that would
silently void an arm. The cadence and the bucket are deliberately outside it.

**The proof** — `dev/kill9_proof.sh`, results in `dev/KILL9-PROOF.json`. It is not a log line; it is
bytes. Two legs of the identical configuration from the identical warm start: one clean, one
`kill -9`'d mid-training and relaunched. The verdict is whether the killed leg's **encoder and store
parameter hashes equal the clean leg's**, with `resumed_from_step > 0`.

> A resumed run that reaches the SAME PARAMETERS as an uninterrupted one has resumed. One that
> reaches different parameters restarted, or resumed wrong — and either way the spec does not ship.

It also proves the **refusal**: a checkpoint is offered to a `Checkpointer` carrying a foreign
fingerprint and must raise `FingerprintMismatch`. A check that cannot fail proves nothing.

**Granularity of a pre-emption:** the steps since the last checkpoint (≤ 5 min), never the arm and
never the unit. Completed arms are skipped by their `arm.json`; the warm start is skipped by its own.

---

## 7. OPTIONAL, and cheap: the L4 interruption test

Before committing $26 of A100, one L4-spot run proves the *pre-emption path* end to end for $0.34.
L4 spot quota is 8 free and `g2-standard-8` draws on the generic `CPUS` metric, so it does not
contend with the A100 job.

```bash
scripts/cloud/run_until_done.sh --job w8enc-l4probe --card l4 --spot \
  --exp w8_encoder_loop --units '0' \
  --artefacts gs://chlu-cloud-forgisprova/artefacts \
  --exp-args '--stage cloud --quick --ckpt-every-s 60' \
  --max-hours 1 --max-attempts 4 --max-usd 1.00
```

⚠ `--quick` marks every number it writes **VOID** in its own manifest. This leg is a *path* test.

---

## 8. Returning the results

```bash
scripts/cloud/fetch_results.sh w8enc-s0 /c/chlu/outputs/w8-encoder-loop/cloud/w8enc-s0
scripts/cloud/fetch_results.sh w8enc-s1 /c/chlu/outputs/w8-encoder-loop/cloud/w8enc-s1
scripts/cloud/fetch_results.sh w8enc-s2 /c/chlu/outputs/w8-encoder-loop/cloud/w8enc-s2
scripts/cloud/cost.sh w8enc-s0 a100-40 1    # and s1, s2 — for the hub's cost line
```

⚠ `fetch_results.sh` defaults its destination to `<repo>/.claude/outputs/cloud/<job>`, and **this
worktree has no `.claude/`**. Pass the destination explicitly, exactly as above.

It fetches `gs://…/jobs/<job>/out` only — **not** `…/ckpt`. That is correct: the checkpoints are
scaffolding, the artefacts are the result.

**What comes back, per seed:** `w8_encoder_loop/seed_<S>/warm_start.json`, `warm_*.pkl`,
`warm_rows.npz`, and per arm `arm.json` · `encoder.pkl` · `store.pkl` · `decoder.pkl` ·
`joint_decoder.pkl` · `history.json` · `declift/`.

### Then, back on the CPU hub box — the stages a GPU may not run

```bash
cd /c/chlu/worktrees/CHLU-w8-encoder-loop
OUT=/c/chlu/outputs/w8-encoder-loop/cloud/merged      # the fetched seed_<S>/ trees, side by side
for SEED in 0 1 2; do for ARM in alone frozen loop_nosw loop_swap alone_wc; do
  python -u -m chlu.experiments.exp_w8_encoder_loop --stage cache  --seed $SEED --arm $ARM --out $OUT
  python -u -m chlu.experiments.exp_w8_encoder_loop --stage probe  --seed $SEED --arm $ARM --out $OUT
  python -u -m chlu.experiments.exp_w8_encoder_loop --stage rivals --seed $SEED --arm $ARM --out $OUT
done; done
python -u -m chlu.experiments.exp_w8_encoder_loop --stage report --out $OUT
# then the rows-3/4 leg per (arm, seed), with the three env hooks pointed at that arm:
#   CHLU_GEN_BRIDGE_ROOT_OVERRIDE=$OUT/seed_<S>/<ARM>/cache
#   CHLU_GEN_CLASSIFIER_ROOT=<the frozen classifier tree>      # the judge does NOT move
#   CHLU_GEN_DECLIFT_DIR=$OUT/seed_<S>/<ARM>/declift
#   CHLU_RIVALS_ROOT=$OUT/seed_<S>/<ARM>/rivals
```

---

## 9. ⛔ Standing prohibitions carried into this spec

- ⛔ Never touch the instances `tsfm-pair-a`, `tsfm-pair-b`, `jonas-vm-a100`, `a100-dev-machine`.
- ⛔ Do not edit `scripts/cloud/startup.sh` while a supervisor is running — it is uploaded as VM
  metadata at create time, so attempt N+1 would run a different launcher than attempt N.
- ⛔ An image pinned by tag is refused, by `launch.sh` and again by `startup.sh`. Keep it a digest.
- ⚠ Known defect in the shipped tooling, reported not patched by this hub: `shutdown.sh:24` rsyncs
  `/var/chlu/out` **without** the `-x '.*ckpt.*'` exclusion that `startup.sh` and `payload_smoke.py`
  both carry. On the pre-emption path — the *common* ending on spot — that copies checkpoints into
  `gs://…/jobs/<job>/out/ckpt/`, an unpruned duplicate prefix. It costs storage, not correctness,
  and `fetch_results.sh` will pull them. Delete that prefix after fetching if it appears.
