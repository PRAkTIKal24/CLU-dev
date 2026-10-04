# CLU/CHLU on GCP — runbook

The cloud counterpart to `scripts/csf3/`. **Cloud is additive**: nothing in
`scripts/csf3/` changed, `push_repo.sh` and `stage_artefacts.sh` are untouched,
and CSF3 remains the place to run anything that wants an 80 GB A100.

Prepared 2026-09-15 against the **measured** quota of project `forgisprova`,
not against a card list. Status legend as in the CSF3 runbook:
**[MEASURED]** — read from the live API on 2026-09-15 ·
**[RAN]** — executed end to end on real hardware ·
**[HEAD]** — needs the Head's account; an agent cannot do it.

---

## 1. The one-paragraph version

`launch.sh` creates one GPU VM from a pinned boot image, which pulls **one
container image pinned by sha256 digest** and runs a job that **checkpoints to
Cloud Storage on a wall-clock timer** and **syncs its outputs to the same
bucket while it runs**. Spot VMs are taken away — so the job resumes from the
bucket by default, verifying a config fingerprint before it does, and the VM
deletes itself on every exit path. `fetch_results.sh` brings the outputs to
`.claude/outputs/`.

---

## 2. Measured quota — build against this, not against the assumed cards

`us-central1` **[MEASURED 2026-09-15]**, via
`gcloud compute regions describe us-central1 --project=forgisprova`:

| card | metric | limit | in use | free | verdict |
|---|---|---|---|---|---|
| **L4** | `NVIDIA_L4_GPUS` | 8 | **2** | 6 | ⭐ iteration card |
| L4 (spot) | `PREEMPTIBLE_NVIDIA_L4_GPUS` | 8 | 0 | 8 | cheap interruption testing |
| A100-40GB on-demand | `NVIDIA_A100_GPUS` | 1 | **1** | 0 | ⛔ consumed by another team |
| **A100-40GB SPOT** | `PREEMPTIBLE_NVIDIA_A100_GPUS` | 16 | 0 | 16 | ⭐ the training path |
| A100-80GB | `NVIDIA_A100_80GB_GPUS` | **0** | 0 | 0 | ⛔ not requested yet — §7 |
| H100 / H200 | — | — | — | — | ⛔ not offered to this project |

**Two limits that are not in any card table, and an honest account of how far they bind:**

1. **`A2_CPUS` = 12 — one `a2-highgpu-1g` worth of vCPU.** Read literally, that caps A100
   concurrency at **one VM**, however many of the 16 spot GPUs are free. ⚠ **How strictly it
   binds is NOT established, and the measurement argues against the literal reading:** another
   team's `jonas-vm-a100` (`a2-highgpu-1g`, 12 vCPU) was RUNNING in `us-central1-f` while our
   own spot `a2-highgpu-1g` launched successfully in `us-central1-a` — and the API reported
   `A2_CPUS usage = 0` throughout. **The `limit` is reliable; the reported `usage` is not.**
   `quota.py` therefore treats this metric as a **conservative** check: it may refuse a launch
   GCE would have accepted, which is the right direction for a preflight, but do not quote
   "concurrency is exactly 1" as a measured fact. It is an upper bound derived from a metric
   whose usage field we have caught lying.
2. **`PREEMPTIBLE_CPUS` = 0.** This is *not* a block: with a zero preemptible CPU quota GCE bills
   Spot VMs against the regular `CPUS` quota (200 here). `quota.py` encodes that rather than
   refusing every spot launch — reading the 0 literally would have refused the entire training path.

### ⭐ A100-80GB: the brief was regionally, not globally, true
All regions were scanned on 2026-09-15. **`us-east4` has `NVIDIA_A100_80GB_GPUS` = 2, one in use,
ONE FREE, on-demand** (`a2-ultragpu-1g`, zone `us-east4-c` only). Spot 80GB is 0 everywhere, and
**no region has any H100 or H200 quota** — that part of the brief is confirmed.

```bash
scripts/cloud/launch.sh --card a100-80 --region us-east4 --job <name>   # preflight passes today
```
⭐ **So the quota request in §7.1 may be unnecessary — check whether `us-east4` is enough first.**
Caveats: it is on-demand only (**$5.07/h**, no spot), stock in `us-east4-c` is intermittent, and
the image and bucket live in `us-central1` so an 80GB run pays cross-region egress on the ~7 GB
pull. **CSF3 remains the right home for 80 GB work** — `-p gpuA`, 19 nodes × 4, free at point of use.

⛔ **`us-central1` holds other teams' running instances.** `tsfm-pair-a` **and `tsfm-pair-b`
(both `g2-standard-4`)** are the 2 in-use L4s — *the task named one; there are two* — and
`jonas-vm-a100` (`a2-highgpu-1g`, `us-central1-f`) consumes the single on-demand A100-40GB.
`a100-dev-machine` (`a2-ultragpu-1g`, `us-east4-c`) holds one of the two 80GB A100s. **None of
these may be stopped, resized or evicted.** Everything this path creates is named `chlu-*` and
labelled `owner=chlu`, so it is separable at a glance:
`gcloud compute instances list --filter="labels.owner=chlu"`.

---

## 3. Files

| File | Runs where | Purpose |
|---|---|---|
| `config.sh` | sourced | project/region/bucket/boot-image settings; thin readers over `cards.json` |
| `cards.json` | data | **single source of truth**: card → accelerator, machine type, both quota metrics, vCPU, list price |
| `quota.py` | laptop | the preflight. Refuses by name, with the measured limit. Also a standalone CLI |
| `Dockerfile` | builder | the pinned runtime image (§4) |
| `build_image.sh` | laptop | builds the image on a throwaway GCE VM, prints the **digest** |
| `launch.sh` | laptop | **the entry point for ONE attempt**: preflight → create → (optionally wait) → cost line |
| `run_until_done.sh` | laptop | **the entry point for a SPOT job**: relaunches until `result.json` exists, under a hard attempt cap **and** a hard spend cap |
| `startup.sh` | the VM | GCE startup script: sync loop, **artefact-root + decoder assertions**, GPU preflight, digest-pinned pull, **the EXP registry and the unit sweep**, self-delete |
| `shutdown.sh` | the VM | GCE shutdown script: the pre-emption path — SIGTERM the payload, final sync |
| `ckpt.py` | in-container | atomic, time-based, fingerprinted, GCS-backed checkpoints |
| `payload_smoke.py` | in-container | the trivial deterministic GPU payload (hash chain) |
| `status.sh` / `cost.sh` / `fetch_results.sh` | laptop | where is it / what did it cost / bring it home |

---

## 4. The image

Mirrors **CSF3 README §3.1** exactly: `uv sync --frozen --extra cuda`, JAX's
**self-contained pip CUDA wheels**, and **no CUDA toolkit anywhere on the
image** — a second CUDA on `LD_LIBRARY_PATH` shadows the bundled libs and
breaks jax, which is the same trap `module load libs/cuda/...` sets on CSF3.
The host contributes the NVIDIA **driver** only.

Everything is pinned by digest or checksum, never by tag:

| layer | pin |
|---|---|
| base | `python:3.11-slim-bookworm@sha256:528257d4…bdf84` |
| uv | `ghcr.io/astral-sh/uv:0.9.7@sha256:ba4857bf…5e142` |
| gcloud CLI | version `585.0.0` by versioned URL + `sha256:7b97198e…c76a36` |
| GPU boot image | `common-cu129-ubuntu-2204-nvidia-580-v20260909` (exact name, **not** a family — a family is a moving pointer exactly like a tag) |
| our runtime image | `…/chlu-cuda@sha256:…` — see `IMAGE_DIGEST` |

`uv.lock` is gitignored, so `build_image.sh` ships it explicitly in the build
context; a `git clone` on the builder would resolve fresh and drift silently
off the laptop's pins. **`launch.sh` and `startup.sh` both refuse a
tag-pinned image reference outright** rather than warning about it.

```bash
scripts/cloud/build_image.sh          # ~25 min; prints and writes IMAGE_DIGEST
```

The builder is a throwaway `e2-standard-8` that takes its context from GCS,
needs no SSH key, and **deletes itself on every exit path** — a failed build
cannot leave an instance billing.

---

## 5. Running a job

```bash
# refuse-and-explain, nothing created:
scripts/cloud/launch.sh --card a100-80 --job x --dry-run

# L4, on demand, iteration:
scripts/cloud/launch.sh --card l4 --job smoke1 --steps 200 --wait

# A100-40GB on SPOT -- the training path. It WILL be pre-empted eventually;
# that is fine, relaunch the same --job and it resumes from the bucket:
scripts/cloud/launch.sh --card a100-40 --spot --job train1 --max-hours 6

# where is it / what has it cost / bring it home:
scripts/cloud/status.sh train1
scripts/cloud/cost.sh   train1 a100-40 1
scripts/cloud/fetch_results.sh train1
```

Useful flags: `--gpus N` · `--region/--zone` · `--fresh` (discard checkpoints
and start over) · `--keep-vm` (skip self-delete; **you must then delete it**) ·
`--max-hours H` (watchdog) · `--ckpt-every-s` · `--sync-every-s`.

### ⭐ 5.1 Running a REAL experiment — `--exp`

`--steps/--size` above drive the **synthetic smoke payload**, which runs no
science. A real arm is selected by name:

```bash
# the tier-III pilot, three seeds, on a spot L4, supervised (see §5.3):
scripts/cloud/run_until_done.sh --job pilot1 --card l4 --spot \
    --exp cluformer --units '0 1 2' --exp-args '--stage s1 --scale toy' \
    --max-hours 1 --max-attempts 8 --max-usd 3.00

# a gen arm: the artefact set is REQUIRED and the launcher refuses without it
scripts/cloud/run_until_done.sh --job cell1 --card a100-40 --spot \
    --exp gen_cell --units '0 1 2' \
    --artefacts gs://chlu-cloud-forgisprova/artefacts \
    --exp-args '--dataset shapes3d --pretext recon --stage cell'
```

⭐ **One launcher, parameterised.** The experiment is a row in `exp_row()` in
`startup.sh`; adding an arm is that row and nothing else. ⛔ **Never a second
launcher per experiment** — two launchers drift and only one gets the next fix.
`launch.sh` asks `startup.sh --check exp <name>` rather than holding a second
copy of the table, so the laptop and the VM cannot disagree.

| `--exp` | module | `--out` is | artefacts |
|---|---|---|---|
| `smoke` | `payload_smoke.py` | — | no |
| `cluformer` | `chlu.experiments.exp_cluformer_pilot` | sweep root | no (downloads enwik8) |
| `gen_encoder` | `chlu.experiments.exp_gen_encoder` | sweep root | **yes** |
| `gen_store` | `chlu.experiments.exp_gen_store` | sweep root | **yes** |
| `gen_cell` | `chlu.experiments.exp_gen_cell` | sweep root | **yes** |
| `gen_dream` | `chlu.experiments.exp_gen_dream` | sweep root | **yes** |
| `gen_certificate` | `chlu.experiments.exp_gen_certificate` | sweep root | **yes** |
| `gen_atom_mean` | `chlu.experiments.exp_gen_atom_mean` | sweep root | **yes** |
| `gen_baselines` | `chlu.experiments.exp_gen_baselines` | **the unit dir** | **yes** |

(The last row is not a typo: `exp_gen_baselines` does **not** append `seed_<N>/`
to `--out` and `exp_gen_atom_mean` does — the same split `job_cpu_gen.sh` §7
encodes. Getting it backwards writes every seed over one manifest.)

**One unit per seed, and a pre-emption costs ONE UNIT.** `--units '0 1 2'` runs
three units; each completed one is banked in a GCS-backed journal, so a
relaunched VM skips it. The unit that was in flight re-runs.

### ⭐ 5.2 What is asserted before the GPU is touched

Both checks run **before** the driver wait, the docker install and the 7 GB
image pull, because all three cost minutes and neither check costs anything.

1. **The five `CHLU_*` artefact roots**, each `${VAR:-default}` over the staged
   GCS prefix, in the same relative layout `scripts/csf3/stage_artefacts.sh`
   publishes — so the bucket and CSF3 scratch are the same tree in two places:

   | variable | default |
   |---|---|
   | `CHLU_CLAUDE_ROOT` | `/var/chlu/artefacts` (the `--artefacts` prefix, fetched) |
   | `CHLU_GEN_DATA_ROOT` | `$CHLU_CLAUDE_ROOT/data/gen` |
   | `CHLU_GEN_BRIDGE_ROOT` | `$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs` |
   | `CHLU_GEN_STORE_RUNS` | `$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs` |
   | `CHLU_GEN_DECLIFT_DIR` | `$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner` |

   A missing root is refused **by name**, naming the variable *and* its value —
   ⛔ a job must never reach the GPU to discover it cannot open an artefact.
   `--skip-artefact-check` opts out for a run that genuinely needs none.
   ⚠ `CHLU_DATA_ROOT` is **not** one of the five and is not asserted: it is the
   download cache, pointed at the host mount so a sweep of N units does not
   download enwik8 N times.

2. ⛔ **The decoder recipe, by name AND by hash.** `GenBridgeConfig.dec_recipe`
   still defaults to **`"baseline"`** — the recipe that **FAILED** the gate
   (0.7191 balanced accuracy, 0/3 seeds) — deliberately, so Montero's history
   reproduces. ⛔ **A baseline run looks entirely normal in the logs and is
   void.** So three things are checked against `gen-decoder-lift`'s own
   `winner_hashes.json`, which is the authority the experiments use:
   * the requested recipe is `v1_capacity_schedule` (⛔ not merely "not
     baseline" — `v2_object_weighted` also lost);
   * the **staged** decoder declares that recipe itself; and
   * `sha256(decoder_v1_s<seed>.pkl)` matches `pkl_sha256`, per seed.

### ⭐ 5.2b The dataset is DOWNLOADED ON THE VM, never staged

`gen-cloud-artefact-bridge` stages 44 files (1.95 GiB) and deliberately **not**
`3dshapes.h5`: it comes from a public bucket whose owner pays the egress, so a VM
download costs **$0.00** while staging it costs storage plus a 255 MiB upload.
Its report calls this *"the only place the two platforms differ"*, and
`startup.sh` closes it — the dataset name is read out of the experiment's own
`--dataset` argument, and `chlu.data.<ds>.download_<ds>(verify=True)` (the
function `chlu data <ds> --download` itself calls) asserts the sha256 **and** the
exact byte count, so a truncated fetch fails loudly instead of yielding a short
file. Measured on `gencell4`:

```
[dataset] shapes3d/3dshapes.h5: 267573662 bytes sha256=0a0f6ed9…be1785
```

⛔ **The CLI command is not used**, because it also *builds the split* — which the
arm builds for itself moments later. The download is the part a job needs.

### ⛔⛔ 5.2c KNOWN BLOCKER — the gen feature cache is PINNED TO THE DEVICE THAT WROTE IT

`exp_gen_store._assert_cache_reencodes` re-encodes 64 cached rows through the
frozen encoder and requires `max |delta| == 0.0` **exactly**. The shipped cache is
`float16` and was written on CPU. On an L4, measured twice:

| container env | `max \|delta\|` |
|---|---|
| default | `0.00390625` = 2⁻⁸ |
| `JAX_DEFAULT_MATMUL_PRECISION=highest` + deterministic XLA | `0.001953125` = 2⁻⁹ |

Those are **exactly one float16 ULP** at the affected element's magnitude (the 64
rows top out at 4.13 ≈ 2², where one float16 ULP *is* 2⁻⁸). ⭐ The re-encode
therefore agrees with the cache **to the last representable bit of the cache's own
dtype** — and the check demands zero, which no cross-device float32 matmul can
deliver. Forcing precision halves the delta and **cannot** reach zero.

⇒ ⛔ **`gen_store` / `gen_cell` cannot run on any GPU against the current cache.**
This is a `chlu/**` decision (a tolerance of one storage ULP, or a per-device
cache), not a cloud-path defect: everything up to and including the arm's own
decoder-hash assertion runs correctly on GCP. **Nothing else about the gen path
is blocked** — the artefacts, the roots, the decoder and the dataset all work.

### ⭐⭐ 5.3 `run_until_done.sh` — the relaunch supervisor

⛔ **A fallback that needs a human to notice and relaunch is not a fallback.**
`gen-cloud-enablement` proved resume works; nothing re-invoked it, and its
`det-ctl` needed **three launches** to finish 600 steps with a human driving all
three. Two of this project's spot L4s were pre-empted by Google **unprompted**
inside one short acceptance window.

```bash
scripts/cloud/run_until_done.sh --job <job> --card l4 --spot \
    --exp <exp> --units '0 1 2' [--max-attempts N] [--max-usd X] [--backoff-s S]
```

* **Completion sentinel**: `out/result.json`, written by `startup.sh` **only**
  when every requested unit is banked. Its absence is the relaunch condition and
  nothing else is.
* ⛔ **Never an unbounded loop.** Two independent hard caps, both refusing by
  name and both reported: `--max-attempts` (default 6) and `--max-usd`
  (default 5.00). The spend cap is checked against spend-so-far **plus the
  ceiling of the attempt about to be launched**, so it cannot be breached by
  exactly the attempt that tests it. A card with no price in `cards.json` is
  refused outright — a loop whose cost cannot be computed is an unbounded loop
  wearing a cap.
* **Stockout back-off**: a stockout creates no VM and is charged \$0, and the
  wait doubles (`--backoff-s`, capped by `--backoff-max-s`). A real VM having
  run resets it: the region is not stocked out.
* It never races a VM of its own — a previous attempt still `RUNNING` is waited
  out, not launched alongside.
* State (attempts used, estimated spend, stockouts) is written to
  `.claude/outputs/gen-cloud-runner/supervisor/<job>.json` after every
  transition, plus one log per attempt.
* Unknown flags are passed straight through to `launch.sh`. ⛔ The flag list is
  not re-declared here; two copies is how the two drift apart.
* **It stops rather than retries on two classes of failure that retrying cannot
  fix**: a **quota refusal** (stock comes back, a quota metric of zero does not)
  and **two consecutive payload failures with no pre-emption** (`out/FAILED`
  present, `out/PREEMPTED` absent — the VM ran and the unit failed, and the next
  attempt reproduces it exactly). Exit codes: `0` done · `1` incomplete ·
  `3` quota · `4` attempt cap · `5` spend cap · `6` payload.
* ⚠ **Do not edit `startup.sh` while a supervisor is running.** `launch.sh`
  uploads it as metadata at create time, so attempt *N+1* would run a different
  launcher than attempt *N*.

### Checkpoint / resume contract
- **Time-based**, not per-epoch: pre-emption does not wait for an epoch, so
  neither does the checkpointer.
- **Atomic**: temp file + `os.replace` locally, then a GCS object create —
  both atomic, so no reader ever sees a torn checkpoint and a process killed
  mid-write still has its previous one.
- **Persisted off the VM**: `save()` is not complete until the object exists
  in the bucket. Instance-local disk dies with the VM, and on Spot that is the
  expected outcome, not the unlucky one.
- **Keep last N** (default 3), locally and remotely.
- **Resume verifies and refuses**: the step counter and a SHA-256 of the
  result-determining config must match, or the resume is refused *by name*.
  `--ckpt-every-s`/`--step-sleep-s` are deliberately outside the fingerprint —
  a relaunch must be free to checkpoint more often.
- **Pre-emption is handled twice**: a thread in the payload long-polls
  `instance/preempted`, and GCE's `shutdown-script` `docker stop`s the
  container so the payload's SIGTERM handler fires. 30 seconds is not enough
  time to discover that your only mechanism did not fire.

### Sync-back contract
Outputs go to `gs://chlu-cloud-forgisprova/jobs/<job>/out/` **on a 15 s timer
while the job runs**, from two independent places (the payload after every
checkpoint, and a host-side loop started *before* the payload). A result that
is not synced back does not exist, and a pre-empted VM never reaches its last
write — so at-exit-only syncing produces nothing in the common case.

---

## 6. Cost

`launch.sh` prints a projected cost at launch and `cost.sh` prints the actual
line at exit:

```
== COST
   job smoke1: l4 (on-demand)  0.128 h @ $0.854/h  =  $0.11
   basis: live VM uptime; list price as of 2026-09-15 …
```

Rates live in `cards.json` and are **public list prices, an estimate, not an
invoice** — spot rates float and discounts are not modelled. Standing costs
outside any job: Artifact Registry storage for the image (~7 GB ⇒ ~$0.70/mo)
and the bucket (negligible).

---

## 7. [HEAD] — the items an agent cannot do

**7.1 [CHECK FIRST, THEN MAYBE FILE] The A100-80GB quota request.**

⭐ **Before filing anything: `us-east4` already has 2 `NVIDIA_A100_80GB_GPUS`, one of them free.**
`scripts/cloud/launch.sh --card a100-80 --region us-east4 --job <name>` passes the preflight today.
If one on-demand 80GB A100 at a time is enough, **there is nothing to file** — and CSF3 gives you
80GB A100s for free anyway.

If more is genuinely needed, this posts under the Head's account:

- Console: **IAM & Admin → Quotas & System Limits** —
  <https://console.cloud.google.com/iam-admin/quotas?project=forgisprova>
- Filter on *Service:* **Compute Engine API** (`compute.googleapis.com`) and the *Dimension*
  `region` = the region you want.

  | metric ID | console display name | us-central1 | us-east4 |
  |---|---|---|---|
  | `NVIDIA_A100_80GB_GPUS` | NVIDIA A100 80GB GPUs | **0** | 2 (1 free) |
  | `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` | Preemptible NVIDIA A100 80GB GPUs | **0** | **0** |
  | `A2_CPUS` | CPUs (A2 machine family) | 12 | 12 |

  Ask for `A2_CPUS` alongside the GPU metrics: `a2-ultragpu-1g` is 12 vCPU, so 12 A2_CPUS is one
  VM's worth however many GPUs are granted. ⚠ But see §2 — we have caught that metric's *usage*
  field reporting 0 while a 12-vCPU A2 instance was running, so treat it as a limit to raise
  rather than as a measurement you can trust.

  Justification text that matches reality: single-node JAX training, 1–4 concurrent VMs,
  checkpoint/resume already in place and demonstrated against real pre-emptions.

**7.2 Set a billing budget + alert** on project `forgisprova` before the first
long spot run — <https://console.cloud.google.com/billing/budgets>. The
`--max-hours` watchdog bounds a single job; it does not bound a mistake
repeated.

**7.3 Tear down when the programme is done:**
```bash
gcloud artifacts repositories delete chlu --location=us-central1 --project=forgisprova
gcloud storage rm -r gs://chlu-cloud-forgisprova
```

**7.4 Never touch** `tsfm-pair-a` or `tsfm-pair-b` (another team, RUNNING,
holding the 2 in-use L4s).

---

## 8. Caveats

- **`us-central1-a` only, by default.** L4 and A100 stock varies by zone; a
  `ZONE_RESOURCE_POOL_EXHAUSTED` on create is a stock problem, not a quota
  problem (`quota.py` already passed), and the fix is `--zone us-central1-b|c`.
- **Bit-identity is a same-card-model claim.** Two runs of the same payload on
  two L4s agree; an L4 and an A100 need not, and comparing across card models
  is not a resume test.
- **No dataset is staged into the image** (task scope). `chlu` is installed;
  data plumbing for a real arm is a separate job.
- **The payload is a smoke test, not science.** `payload_smoke.py` exists to
  prove the *path*. Wiring a real experiment means pointing `startup.sh`'s
  `docker run` at the real entry point and reusing `ckpt.py` — ⛔ do not write
  a second checkpoint mechanism.
- **JAX cold start** is slow on first compile; the image sets a persistent XLA
  cache at `/var/chlu/jax_cache`, which survives container restarts on the
  same VM but **not** a pre-emption (the disk goes with the VM).
