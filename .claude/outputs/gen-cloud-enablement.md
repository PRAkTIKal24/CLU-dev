# gen-cloud-enablement — experiment-engineer report

**Task + acceptance criterion:** make GCP runnable — a working, checkpoint-first path from this
repo to a running GPU job and back — accepted on **a real L4 job end to end**, **a spot instance
taken away mid-run and the job resuming bit-identically**, a launcher that refuses an un-quota'd
card, a digest-pinned image, mid-run sync-back, the quota items for the Head, green tests, a cost line.

**Status: done.** All seven acceptance items met **on the image that ships**
(`@sha256:0104356…5c44c`). ⭐ **Pre-emption was tested four times and twice GCE did it to us
unprompted** — `det-ctl` was genuinely pre-empted at step 80 and again at step 172 and still
finished correctly. ⭐ **Five independent runs converge on the same 64-character hash chain over
all 600 steps**, including two whose VM was destroyed mid-run. ⛔ **Bit-identity required forcing
XLA determinism**; at default flags a resumed run agrees only to ~6 significant figures, for a
reason that is about XLA autotuning and *not* about the resume (§4.3). **Total spend: $1.37.**

⚠ **Two corrections to the task's own premises, both measured (§4.1):** the A100-80GB card the
brief says does not exist **exists in `us-east4` today** (2 quota, 1 free) — the quota request may
not need filing — and my own `A2_CPUS ⇒ concurrency 1` claim is **weaker than I first wrote it**;
the metric's usage field demonstrably lies, so it is a conservative bound, not a measurement.

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** ⛔ **none — infrastructure.** No arm, no mechanism, no number here enters any table.
- **Laundering control:** n/a for a dial, but the bit-identity claim has one and it earned its
  keep: an **uninterrupted reference on a separate VM** (`final-ref` / `det-ref`) against the
  killed-and-resumed run. Without it, "the chains differ" cannot be told from "the resume is
  broken" — and at default flags they *did* differ (§4.3). ⛔ **No PREREG** — the acceptance is a
  mechanism, not a measurement.
- **Falsifies:** nothing (infrastructure). **Does NOT falsify:** spot pre-emptions, zone stockouts
  and list-price-vs-invoice drift are properties of GCP, not defects in this path.

## ⚠ RECONCILIATION LIST (first-10-lines rule, protocol §5) — **six sites, an owner is needed**
1. ⭐ **The A100-80GB premise is regionally, not globally, true.** `us-east4` has
   `NVIDIA_A100_80GB_GPUS` = 2, **one free, on-demand, today**. Every region was scanned. Anything
   that says "no 80 GB card exists for this project" needs qualifying by region.
2. **`scripts/csf3/README.md` §3.2 is WRONG and I relied on it:** it says `uv.lock` is gitignored
   and must be rsync'd. **`uv.lock` is tracked.** It cost this spoke one image build (§4.4).
3. **§7 candidate — `uv sync --frozen` does not catch extras-only pyproject↔lock drift.**
4. **The task's quota table is missing `A2_CPUS`,** and my first reading of it was an overclaim —
   see §4.1 for what is actually established.
5. **The task names one untouchable instance; there are four running** (§4.1).
6. **A spot job needs a relaunch supervisor and this spoke did not ship one** (§7.1). Resume works;
   nothing re-invokes it. I relaunched by hand; `det-ctl` needed three launches to finish.

---

## 0. FLAG PROVENANCE (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/gen-cloud-enablement` @ `a1d6f48`, `8389f82`, `80be4ad`, `dac5330`, `7837602`, `acbb17c`, `eeddb71`, `ab4bc4d`, `d63e014`, `b81c2cd` (**10**). Task base was `main @ 115d4b5`; **`main` advanced 12 commits during the work (other Hubs merging), so the branch is rebased onto `main @ 8a404de`** per protocol §3.5. `115d4b5` is still an ancestor; the rebase was conflict-free and the diff vs `main` is **purely additive (2222 insertions, 0 deletions)**. |
| worktree | `../CHLU-cloud` (as tasked). No concurrent spoke touched `scripts/cloud/**` or `tests/test_cloud_launch.py`. |
| project / region | `forgisprova` / `us-central1` (zone chosen by stockout fallback, recorded per run); `us-east4` for the 80 GB probe |
| ⭐ **shipped run image** | `us-central1-docker.pkg.dev/forgisprova/chlu/chlu-cuda@sha256:010435659b510551b64d2008361ac5c906bf127246d0d4fc678b8c204385c44c` |
| superseded images | `@sha256:6e8e2c92…dc8d` (**wrong lock**, §4.4) — ⭐ **DELETED from the registry** so it cannot be pulled by accident. `@sha256:8b814166…9c7a` and `@sha256:94002e30…1dc6` kept (correct provenance, just superseded; ~$1.40/mo — `gcloud artifacts docker images delete …` to trim). |
| image base pins | `python:3.11-slim-bookworm@sha256:528257d4…bdf84` · `ghcr.io/astral-sh/uv:0.9.7@sha256:ba4857bf…5e142` · gcloud CLI `585.0.0` + `sha256:7b97198e…c76a36` |
| boot image | `common-cu129-ubuntu-2204-nvidia-580-v20260909` (`deeplearning-platform-release`) — exact name, **not** a family |
| measured driver | **580.178.04** on the L4 VMs (CSF3 is 580.126.09 — same major, so the bundled-CUDA-wheel argument is unchanged) |
| in-image JAX | **jax 0.9.0**, `jax-cuda12-plugin` 0.9.0, `nvidia-cudnn-cu12` 9.24.0.43 — identical to `uv.lock`, i.e. laptop/CSF3 parity |
| local env | main venv reused (protocol §4); **no worktree `uv sync`** |
| payload config (`det-*`, `final-*`) | `--steps 600 --size 2048 --seed 0 --ckpt-every-s 15 --step-sleep-s 0.5`, sync every 15 s, `--max-hours 1`; acceptance `--steps 120`; A100 `--size 4096` |
| determinism | `XLA_FLAGS=--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0` on `det-*`/`final-*`; **absent** on `preempt-*`, and that is the whole of §4.3 |
| ⛔ NOT done | no quota request filed (Head's account) · no existing instance touched · no `scripts/csf3/**`, `chlu/**`, `pyproject.toml` or `uv.lock` modified · no dataset in the image · no science |

---

## 1. What I did

| # | deliverable | where | status |
|---|---|---|---|
| 1 | pinned image (CSF3 recipe, digest-pinned) | `scripts/cloud/Dockerfile`, `build_image.sh` | ✅ |
| 2 | launch script w/ quota refusal + `--dry-run` | `launch.sh`, `quota.py`, `cards.json` | ✅ |
| 3 | checkpoint-first: time-based, atomic, keep-N, fingerprinted, GCS-persisted | `ckpt.py` | ✅ |
| 4 | sync-back **on a timer during** the job | `startup.sh` + `payload_smoke.py` | ✅ |
| 5 | cost line | `cost.sh`, printed by `launch.sh` | ✅ |
| 6 | runbook + Head's checklist + quota items | `README.md` §7 | ✅ |
| 7 | tests | `tests/test_cloud_launch.py` — **36 passed** | ✅ |

The repo had **zero** cloud tooling before this. `scripts/csf3/**` is byte-unchanged; cloud is additive.

---

## 2. ⭐ ACCEPTANCE 1 — a real GPU job, end to end, on an L4

`chlu-final-l4`, **on-demand L4**, `us-central1-b`, on the shipped image. Real VM, image pulled
**by digest**, real sync-back, EXIT 0.

```
NVIDIA L4, 580.178.04, 23034 MiB
Digest: sha256:010435659b510551b64d2008361ac5c906bf127246d0d4fc678b8c204385c44c
[payload] jax 0.9.0 backend=gpu devices=['cuda:0']
{'chain': '7ddaf2f11bc4d2fb15b0ab0d5f231c30ba1c40d958725dfa03d41f0d1991f209',
 'steps': 120, 'backend': 'gpu', 'devices': ['cuda:0'], 'resumed_from': 0}
```
**Cost: 0.103 h × $0.854/h = $0.09.** (On-demand L4 was stocked out region-wide for ~3 minutes
first; the run is attempt 2 of a retry loop. See §4.2 — this was the normal case today, not bad luck.)

⭐ **The A100-40GB SPOT training path was validated the same way** (`chlu-accept-a100`,
`a2-highgpu-1g`, `--size 4096`, EXIT 0, chain `ab340217…`): **0.102 h × $1.470/h = $0.15.**
This matters because on-demand A100 quota is 1 and is consumed by another team — spot is the only
A100 route, and it now has a demonstrated end-to-end run rather than an assumption.

---

## 3. ⭐⭐ ACCEPTANCE 2 — PRE-EMPTION SURVIVED, DEMONSTRATED

### 3.1 The design, stated before the evidence
Two independent detectors, because 30 seconds is not enough time to discover that your only
mechanism did not fire: a thread inside the payload long-polling
`metadata/instance/preempted?wait_for_change=true`, and GCE's `shutdown-script` running
`docker stop` so the payload's SIGTERM handler fires. Either sets the stop flag; the loop then
**force-checkpoints**, ignoring the timer, and exits 0.

### 3.2 The instance really was taken away — four times, two of them by Google

| run | what happened to the VM | resumed from |
|---|---|---|
| ⭐ `final-kill` | **I destroyed it** — `instances delete` at step 139/600 in `us-central1-c`; relaunched into `us-central1-a` | **139** |
| `det-kill` | **I destroyed it** at step 151/600 in `us-central1-b`; relaunched into `us-central1-a` | **151** |
| ⭐⭐ `det-ctl` (1st) | **GCE pre-empted it for real**, unprompted, at step 80/600 | — |
| ⭐⭐ `det-ctl` (2nd) | **GCE pre-empted it for real again**, at step 172/600 | **80**, then **172** |

Evidence the pre-emption path fired, read from what the dying VM left in the bucket:
```
progress.json after the kill:  "step": 139, "preempted": true,
                               "stop_reason": ["signal SIGTERM"], "n_saves": 7
checkpoints in GCS:            ckpt_…103.npz  ckpt_…127.npz  ckpt_…139.npz
```
`det-ctl`'s log names the *other* detector: `[payload] STOP requested: GCE pre-emption notice` →
`checkpointed at step 80 … -> exit 0`. On `det-kill` the cadence was ~25 steps and the final
checkpoint landed at **151, one step after 150** — an off-cadence forced save, the signal
handler's signature.

⭐ **And the thing the whole design exists for.** On `det-kill`, `progress.json` was left at step
**150** while the checkpoint reached step **151**. The final `progress.json` write never got off
the VM — but the checkpoint did, because `save()` publishes to GCS atomically at save time instead
of waiting for the sync timer. **A pre-empted VM loses its last write. It does not lose its last
checkpoint.**

### 3.3 Resume, not restart — and bit-identical, on the shipped image
```
final-ref   resumed_from=  0  chain=664526ada28ee37e39d9988cf48b85a266e0e424a40e438f64e138b9ba9d28a7
final-kill  resumed_from=139  chain=664526ada28ee37e39d9988cf48b85a266e0e424a40e438f64e138b9ba9d28a7
            acc_sum -57979.22265625 · acc_first -552.1762084960938 · acc_last 514.4844970703125
            — identical to the last bit on all three
BIT-IDENTITY: PASS      [payload] RESUMED from step 139 … -- 461 steps remain
```
⭐ **Five runs, one chain.** `final-ref`, `final-kill`, `det-ref`, `det-ctl`, `det-kill` all land on
`664526ad…` — across two image builds, three zones, two destroyed VMs and two real GCE
pre-emptions, with `resumed_from` ∈ {0, 139, 151, 172}.

⭐ **Why the chain and not the endpoint.** It is `sha256(chain_{t-1} ‖ bytes(acc_t))` folded at
every step, so a run that restarted from zero, dropped a step, repeated a step, or produced a
different float **anywhere in its 600-step history** cannot reach the same digest. Comparing final
values would not catch a repeated step; this does.

---

## 4. Findings

### 4.1 ⭐ Two measured corrections to the task's quota premises

**(a) ⭐⭐ The 80 GB A100 exists — in `us-east4`.** The brief says A100-80GB is `0` with the quota
request unfiled, and warns against designing around a card that does not exist. True of
`us-central1` and `europe-west4`. **False of `us-east4`**, which nobody checked: I scanned **every
region in the project**.

| | us-central1 | europe-west4 | **us-east4** | every other region |
|---|---|---|---|---|
| `NVIDIA_A100_80GB_GPUS` | 0 | 0 | ⭐ **2, one in use ⇒ 1 FREE** | 0 |
| `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` | 0 | 0 | 0 | 0 |
| `NVIDIA_H100_80GB_GPUS` / `NVIDIA_H200_GPUS` | 0 | 0 | 0 | 0 — ⛔ **brief confirmed, no H100/H200 anywhere** |

`scripts/cloud/launch.sh --card a100-80 --region us-east4` **passes the preflight today**
(`NVIDIA_A100_80GB_GPUS need 1, limit 2, used 1, free 1 → yes`). ⛔ **I did not get a run onto it:**
`us-east4-c` is the only zone offering `a2-ultragpu-1g`, and **18 create attempts over ~45 minutes
all returned STOCKOUT** (no VM was created by any of them — a stockout costs nothing). So this is **quota-verified and dry-run-verified, not a ran card** — stated that
way deliberately, because §"what would prove this spoke wrong" is exactly about claiming a path
that has not been exercised. Caveats: on-demand only (**$5.07/h**), and image+bucket are in
`us-central1`, so an 80 GB run pays cross-region egress on the ~7 GB pull.
⭐ **CSF3 remains the right home for 80 GB work** — free at point of use, 19 nodes × 4.
**Consequence for the Head: check whether one `us-east4` card is enough before filing anything.**

**(b) ⚠ I overstated `A2_CPUS`, and I am correcting it.** My first write-up said "`A2_CPUS` = 12 ⇒
A100 concurrency is exactly ONE, not sixteen." That is a literal reading of the metric, and **the
measurement argues against it**: another team's `jonas-vm-a100` (`a2-highgpu-1g`, 12 vCPU) was
RUNNING in `us-central1-f` while our own spot `a2-highgpu-1g` launched successfully in
`us-central1-a` — and the API reported `A2_CPUS usage = 0` throughout. **The `limit` is reliable;
the reported `usage` is not.** What is established: the limit is 12, one `a2-highgpu-1g` is 12
vCPU, and `quota.py` refuses `--gpus 2` on that basis. That refusal is **conservative** — it may
refuse a launch GCE would accept — which is the right direction for a preflight, but "concurrency
is 1" must not be quoted as a measured fact. Every other row of the task's table is **confirmed**:
L4 8 with 2 in use, A100-40 on-demand 1 consumed, **A100-40 SPOT 16 free**.

**(c) ⛔ Four instances are running, not one.** The task flags `tsfm-pair-a`. Also live:
`tsfm-pair-b` (`g2-standard-4` — the two together are the 2 in-use L4s), `jonas-vm-a100`
(`a2-highgpu-1g`, `us-central1-f` — the consumer of the single on-demand A100-40GB), and
`a100-dev-machine` (`a2-ultragpu-1g`, `us-east4-c` — one of the two 80 GB A100s). **None were
touched.** Everything I created is named `chlu-*` and labelled `owner=chlu`; **zero remain**
(`--filter="labels.owner=chlu"` → empty).

### 4.2 ⛔ `us-central1` GPU stock is genuinely scarce, and spot L4 gets pre-empted for real
Sixteen create attempts hit `STOCKOUT` across `-a`/`-b`/`-c` in one afternoon; one relaunch needed
three tries over several minutes. **A stockout is not a quota problem** — the preflight has already
passed — so `launch.sh` walks the zones that actually offer the machine type (`us-central1-f` has
no `g2` at all) and names which failure it saw. Without it the majority of my launches would have
needed a hand retry. Two of my spot L4s were then pre-empted by Google mid-run, unprompted.

### 4.3 ⛔ BIT-IDENTITY REQUIRES FORCING XLA DETERMINISM — the finding most likely to bite later
The **first** kill/resume pair, run at default flags:
```
preempt-ref   chain=664526ad…  acc_sum=-57979.22265625   (uninterrupted)
preempt-kill  chain=42f02df7…  acc_sum=-57979.25         (resumed from 179)
```
~6 significant figures of agreement, disagreement in the last bits. **The cause is not the resume.**
XLA autotunes matmul algorithm selection **per process**; a resumed job is a new process, autotunes
afresh, and can pick a different algorithm for the remaining steps. The two *uninterrupted* runs
agreed because they happened to pick the same one — note `preempt-ref` lands on `664526ad…`, the
same chain all five deterministic runs reach.

⭐ With `--deterministic` (`--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`) the same
experiment is bit-identical **across a destroyed VM, across zones, and across a resume** (§3.3).

**Consequence, stated plainly:** *a GPU job that is resumed is not bit-reproducible against an
uninterrupted one unless determinism is forced — and a spot job is resumed by definition.* Any
future claim of the form "the resumed leg reproduces the straight leg" must either pass
`--deterministic` or state its tolerance. `result.json` now records `xla_flags`, so the question
can be answered after the fact instead of argued about.

### 4.4 Defects found and fixed in this spoke's own work
| what | how it was caught | fix |
|---|---|---|
| ⭐ **Built the image from the wrong `uv.lock`.** I trusted CSF3 README §3.2 ("gitignored") and copied the main repo's lock into the worktree. `pyproject.toml@115d4b5` declares a `gen`/h5py extra that lock lacks, and **`uv sync --frozen` did not object** — the image recorded commit `115d4b5` but was built from another branch's lock. | `git status` showed ` M uv.lock` at commit time | lock restored; `build_image.sh` now **refuses to build** when `uv.lock` differs from the checkout; image rebuilt |
| ⭐ **"Keep last N" silently became "keep last N per process".** A resumed process downloads only the checkpoint it restarts from, so pruning by the local directory never sees what the pre-empted process left in the bucket. Measured: steps **137 and 163 orphaned** after the first kill. | inspecting the bucket after the kill | `_prune()` prunes the union of local ∪ remote; regression test |
| ⭐ **Both syncers duplicated every checkpoint into `out/ckpt/`**, a prefix nothing prunes — unbounded growth across a campaign of pre-emptions. **I fixed one of the two and thought I was done**; `payload_smoke.py::_sync` kept doing it. | bucket listing, twice | both excluded; test asserts on **both** invocations |
| **…and the first exclusion did nothing.** `-x '^ckpt/.*'` matched nothing: gcloud matches the regex against the **full path**, not the path relative to the source. | `out/ckpt/` filled up anyway | `-x '.*ckpt.*'`; verified empty on the final runs |
| **`launch.sh` died on every on-demand launch**: bash 3.2 (macOS `/bin/bash`) treats `"${ARR[@]}"` on an empty array as unbound under `set -u`. | first real launch | `${PROV[@]+"${PROV[@]}"}` |
| ⛔ **The DLVM boot image has no Docker.** `common-cu129-ubuntu-2204-nvidia-580` ships the NVIDIA **driver** but `docker: command not found`, so the first L4 job died at `docker pull`. | the failed job's log — **which reached the bucket because the sync loop starts before the payload** | `startup.sh` installs docker + the NVIDIA container toolkit idempotently |

⭐ That last row is worth reading twice: **the job that failed still produced a usable artefact**,
because sync-back starts before anything can go wrong. That is the argument for §5 of the task.

---

## 5. The rest of the acceptance

### 5.1 ACCEPTANCE 5 — sync-back mid-run, not at exit
Captured while the VM was still `RUNNING`, on the shipped image:
```
{'step': 53, 'steps': 600, 'done': False, 'n_saves': 3, 'preempted': False}
out/ prefix at that moment: job.log  progress.json        <- and no duplicated ckpt/
```
`done: false`, three checkpoints already in the bucket. Two independent writers push during the
run: the payload after every checkpoint, and a host-side loop started **before** the payload on a
15 s timer.

### 5.2 ACCEPTANCE 3 & 4 — refusals and the digest
Full transcript: `.claude/outputs/gen-cloud-enablement/refusals.txt`. Condensed:
```
a100-80    -> REFUSING TO LAUNCH. card 'a100-80' (nvidia-a100-80gb) has NO QUOTA in this
              project/region: PREEMPTIBLE_NVIDIA_A100_80GB_GPUS limit = 0. …
h200       -> unknown card 'h200'. Known: a100-40, a100-80, l4, t4. Not available to this
              project at all: … no NVIDIA_H200_GPUS quota in ANY region (all regions scanned)
a100-40 x2 -> 2x a2-highgpu-1g needs 24 vCPU, but A2_CPUS limit = 12 … caps concurrency at 1
:latest    -> image … is pinned by TAG. Refusing: a tag is a mutable pointer, so a re-run would
              silently get a different environment and we would never know.
```
All four refuse **before anything is created** (`Nothing was created.`). Shipped digest:
**`sha256:010435659b510551b64d2008361ac5c906bf127246d0d4fc678b8c204385c44c`**.

### 5.3 ACCEPTANCE 7 — tests and the cost line
`tests/test_cloud_launch.py`: **36 passed, 0 failed, 0 skipped** (3.0 s; main venv, no GPU, no
network, no JAX import — quota logic runs against a recorded `regions describe` payload so the
refusals stay pinned after the live quota moves). `ruff check scripts/cloud tests/` → clean.
⛔ `ruff format` **not** applied: 102 files on `main` would reformat, so it is not an enforced gate
here and I did not introduce a formatting delta.
⛔ **The full suite was not run.** Nothing added is on any `chlu/**` import path — `scripts/cloud/**`
plus one new test file — so the suite count is unchanged by construction.

**COST — measured from the GCE operations log (insert→delete per successful VM):**

| instance | runs | hours | $/h | USD |
|---|---|---|---|---|
| `final-l4` (L4 on-demand) · `accept-l4` | 1 · 1 | 0.103 · 0.103 | 0.854 | 0.09 · 0.09 |
| `accept-a100` (A100-40 **spot**) | 1 | 0.102 | 1.470 | **0.15** |
| `final-ref` · `final-kill` (L4 spot) | 1 · 2 | 0.195 · 0.283 | 0.343 | 0.07 · 0.10 |
| `det-ref` · `det-ctl` · `det-kill` (L4 spot) | 1 · 3 · 2 | 0.201 · 0.500 · 0.262 | 0.343 | 0.07 · 0.17 · 0.09 |
| `preempt-ref` · `preempt-kill` (L4 spot) | 1 · 2 | 0.188 · 0.259 | 0.343 | 0.06 · 0.09 |
| `shake1` · `shake2` (L4 on-demand, debugging) | 1 · 1 | 0.039 · 0.091 | 0.854 | 0.03 · 0.08 |
| 4 × `chlu-builder-*` (e2-standard-8 image builds) | 4 | 1.070 | 0.268 | **0.29** |
| **TOTAL** | **17 VMs** | **3.40 h** | | **$1.37** |

GPU VMs $1.08, image builders $0.29. ⛔ **List-price estimate, not an invoice** — spot rates float
and discounts are not modelled; `cost.sh` prints the same line per job. Standing cost outside any
job: **3 images × ~7 GB ≈ $2.10/month** in Artifact Registry (only `@sha256:0104…` is needed;
the other two are superseded and can be deleted), bucket negligible. §7.3 of the runbook tears
down both in two commands.

---

## 6. Git footprint
- **Branch** `agent/experiment-engineer/gen-cloud-enablement`, task base `main @ 115d4b5`, now rebased onto `main @ 8a404de`, worktree `../CHLU-cloud` (left in place for Hub review).
- **10 commits, rebased onto `main @ 8a404de`:** `a1d6f48` image+quota · `8389f82` checkpointing ·
  `80be4ad` launch/run/price · `dac5330` tests · `7837602` runbook · `acbb17c` sync-exclusion fix ·
  `eeddb71` digest pin · `ab4bc4d` quota corrections · `d63e014` generic fetch destination ·
  `b81c2cd` honest zone line. Verified **from the main repo** per protocol §3.2:
  `git log --oneline main..agent/experiment-engineer/gen-cloud-enablement` → 10, and
  `git diff --stat main..HEAD` → **16 files, additions only, 0 deletions**. ⚠ `main` moved 12 commits under me while
  I worked (three other Hubs merged); the rebase was clean because nothing I touch is shared.
- **Files added (all new; nothing existing modified):**
  `scripts/cloud/{Dockerfile,README.md,build_image.sh,cards.json,ckpt.py,config.sh,cost.sh,fetch_results.sh,launch.sh,payload_smoke.py,quota.py,shutdown.sh,startup.sh,status.sh,IMAGE_DIGEST}`,
  `tests/test_cloud_launch.py`.
- ⛔ **Untouched:** `scripts/csf3/**`, `chlu/**`, `pyproject.toml`, `uv.lock`, `push_repo.sh`,
  `stage_artefacts.sh`. No push, no PR — branch left for Hub/Head review.
- **GCP resources created** (all new): Artifact Registry repo `us-central1/chlu`, bucket
  `gs://chlu-cloud-forgisprova`, and the `chlu-*` instances above — **all deleted, none remain.**
  I **enabled `artifactregistry.googleapis.com`** on the project: a project-level change, additive
  and non-destructive, required by the one job. Flagging it explicitly.
  ⛔ **No existing instance was stopped, resized, evicted, or inspected beyond `instances list`.**
- **Artefacts:** `.claude/outputs/gen-cloud-enablement/` — per-job `result.json`, `progress.json`,
  `job.log` (incl. `det-kill/job-attempt1.log`, the killed attempt), checkpoints, build logs,
  `refusals.txt`.

---

## 7. Open questions / follow-ups / risks
1. ⭐ **A spot job needs a relaunch supervisor and I did not ship one.** Resume works; nothing
   re-invokes it. `det-ctl` needed three launches to finish 600 steps and I drove all of them by
   hand. A ~20-line `run_until_done.sh` (relaunch while no `result.json`, back off on stockout)
   closes this and is the highest-value follow-up. Left out as scope — the task named five deliverables.
2. ⭐ **Decide the 80 GB question before filing anything** (§4.1a). `us-east4` may already be enough;
   if not, request `NVIDIA_A100_80GB_GPUS` + `A2_CPUS` there rather than in `us-central1`.
3. **The `us-east4` 80 GB path is quota-verified but never ran** — stock was absent all window.
   Someone should land one job on it before it is relied on.
4. **The image carries no dataset and no real entry point.** Wiring a real arm means pointing
   `startup.sh`'s `docker run` at it and reusing `ckpt.py` — ⛔ **do not write a second checkpoint
   mechanism**; that was explicit in the task and remains right.
5. **Host docker + NVIDIA container toolkit are the one unpinned thing.** Deliberate: the host
   contributes a driver and a runtime, the image contributes python/jax/CUDA, and only the latter
   determines a result. If a run must be defensible to the host package version, the answer is a
   custom boot image, not a pin in `startup.sh`.
6. **`--max-hours` bounds one job, not a mistake repeated.** A billing budget is a [HEAD] item.
7. **Bit-identity is a same-card-model claim.** Two L4s agree under `--deterministic`; an L4 and an
   A100 need not, and comparing across card models is not a resume test.

## Proposed handover updates (for the Hub)

**§7-CURRENT additions:**
- **7.30 [ENV, standing] `scripts/csf3/README.md` is WRONG: `uv.lock` is TRACKED,** not gitignored.
  Two sites — the `push_repo.sh` table row (line 52) and §3.2 (line 84, "rsync is the ONLY way it
  reaches the cluster"). **Still present at `main @ 8a404de`**, i.e. it survived today's CSF3 README
  pass. It cost this spoke an image build.
- **7.31 [ENV, standing] `uv sync --frozen` does NOT detect extras-only drift** between
  `pyproject.toml` and `uv.lock`. A lock from another branch builds green and yields an environment
  whose recorded provenance is a different commit's. Anything shipping an environment must compare
  the lock against the checkout itself.
- **7.32 [cloud, standing] GPU results are not bit-reproducible across processes at default XLA
  flags.** Autotuning selects matmul algorithms per process, so a **resumed** run diverges from an
  uninterrupted one in the last bits (measured: `acc_sum` −57979.22265625 vs −57979.25).
  `--deterministic` restores bit-identity. **Any resumed-leg reproduction claim must say which
  regime it is in**; `result.json` records `xla_flags`.
- **7.33 [cloud, standing] ⭐ A100-80GB DOES exist for this project — in `us-east4`** (limit 2, 1
  free, on-demand; spot 80GB is 0 in every region). `us-central1`/`europe-west4` are 0. **No H100 or
  H200 quota in any region.** Quota-verified, not yet run — `us-east4-c` stock was absent.
- **7.34 [cloud, standing] `A2_CPUS` = 12 is a limit you can trust; its reported `usage` is not.**
  Measured 0 while another team's 12-vCPU A2 instance was running. Treat "A100 concurrency = 1" as
  a conservative bound, not a measurement.
- **7.35 [cloud, standing] Four instances of other teams are live** — `tsfm-pair-a`, `tsfm-pair-b`
  (the 2 in-use L4s), `jonas-vm-a100` (the 1 on-demand A100-40), `a100-dev-machine` (1 of the 2
  us-east4 80GB A100s). None may be touched. `us-central1` GPU stock is scarce enough that zone
  fallback and launch retries are mandatory, not optional.

**New capability for the running log:** GCP is runnable, checkpoint-first, and has survived real
pre-emptions. `scripts/cloud/README.md` is the runbook;
`scripts/cloud/launch.sh --card <l4|a100-40> [--spot] --job <name>` is the entry point; results land
in `gs://chlu-cloud-forgisprova/jobs/<job>/` and come home with `fetch_results.sh`.
**CSF3 remains the place for 80 GB A100s at scale** — free at point of use, and the cloud has one.
