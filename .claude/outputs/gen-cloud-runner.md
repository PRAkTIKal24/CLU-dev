# gen-cloud-runner — experiment-engineer report

**Task + acceptance criterion:** make GCP run a real CHLU experiment **unattended, to completion,
across pre-emption** — a real entry point (`EXP=`, one parameterised launcher, the five `CHLU_*`
roots asserted, the decoder asserted by name and hash, `ckpt.py` reused) plus a relaunch supervisor
with a hard attempt cap and a hard spend cap.

**Status: done — 6/6 acceptance items met.** ⭐ **A spot L4 was destroyed mid-sweep and the job
finished itself**: `run_until_done.sh` relaunched, the new VM read the journal, **skipped the banked
unit and re-ran only the one in flight**, and wrote `result.json` — **no human in the loop.**
**Total spend: $0.50, 13 VMs, none left running.**

⭐ **The end-to-end leg RAN** — its precondition landed mid-task and is satisfied (`STAGED-GCS.json`:
`status=complete`, `checksum_match=true`, 0 failed to load, verified on a VM). ⛔ **But it does not
finish, and the reason is in `chlu/**`, not in the cloud path.** Everything this spoke owns is green
against the real 2 GB artefact set — roots asserted, decoder asserted by name and hash, dataset
downloaded and checksummed, the arm launched on the GPU with its real config and passing its **own**
decoder guard. The arm then dies on the **encoder-cache re-encode gate**: on an L4, **20.853% of
elements differ at max 4389 float16 ULP** (§5.1). ⛔ I changed nothing — `chlu/**` is out of my scope.
**That blocker, not the plumbing, is what stands between GCP and a gen run.**

⚠ **§5.1 also carries a RETRACTION of this spoke's own first number.** I initially reported the
disagreement as *"exactly one float16 ULP"*; a per-element instrument that landed on `main` while I
worked refutes it, and ⛔ **"one ULP" must not be quoted for the L4.** The reasoning that produced the
wrong number, and why it does not follow, is written out in §5.1 rather than quietly replaced.

⚠ **The most important thing this spoke found is a defect in what it inherited, not in what it
built:** ⛔ **the shipped image could not open Shapes3D at all** — no `h5py` — so **no gen experiment
could have run on GCP**, and nothing said so. It is invisible on the laptop, where those tests
`importorskip` and pass as SKIPs. Found only by driving a real gen arm to the GPU (§5).

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** ⛔ **none — infrastructure.** **Laundering control:** n/a. **Falsifies:** nothing.
- ⛔ **No PREREG** — the acceptance is a mechanism, not a measurement.
- ⛔ **No science, no arm, no number here enters any table.** Every experiment run was `--stage s1
  --scale toy` or `--quick`, which the modules themselves label *plumbing smoke, NOT a result*.

## ⚠ RECONCILIATION LIST (first-10-lines rule, protocol §5) — **five sites, an owner is needed**
0. ⚠⚠ **A RETRACTION OF MY OWN NUMBER.** This report first said the encoder-cache disagreement on an
   L4 was *"exactly one float16 ULP"*. ⛔ **It is not: 20.853% of elements, max 4389 ULP** (§5.1).
   Nothing downstream has quoted it yet — this is the only copy — but **anyone who read an earlier
   draft must drop "one ULP"**, and ⛔ **`gen-reencode-tolerance` must not be read as "GPU gen runs
   are unblocked"** (§5.1, handover 7.36).
1. ⛔⛔ **`scripts/cloud/IMAGE_DIGEST` points at an image that CANNOT RUN GEN WORK, and whose `chlu`
   is 7 `chlu/` files stale vs `main`.** I fixed the Dockerfile and **built + verified the replacement**, but
   ⛔ **I did not repoint `IMAGE_DIGEST` — it is not in my file list and it is shared state.**
   **The Hub's action is one line** (§8). Until then, pass `--image <digest>` explicitly.
2. **`tests/test_gen_data_splits.py` hides a missing `h5py` as 8 SKIPs.** The pyproject comment
   already says so; it has now cost a GPU-minute. Any environment that ships gen work must assert
   the import, not skip it. The image does now.
3. ⭐ **`PREEMPTIBLE_NVIDIA_L4_GPUS = 8, all free, in `us-east4` too** — the peer measured only
   `us-central1`. A second L4 region matters: **`us-central1` spot AND on-demand L4 were both
   stocked out region-wide for ~35 minutes today**, and ~20 stockouts landed across the day (§9.3).
4. **An interrupted `gcloud storage cp` on a destroyed VM leaves a 0-byte `job.log_.gstmp`** in the
   output prefix, which `fetch_results.sh` then copies down. Cosmetic; it lives in `ckpt.py`'s
   neighbour path and is nobody's bug yet.

---

## 0. FLAG PROVENANCE (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/gen-cloud-runner` @ `97c567c`, `edcbf33`, `934ac8f`, `5ed3471`, `6c148a9`, `8980072`, `c7a6980`, `e81f720`, `21c8660`, `2ccdd70`, `7536bd3`, `043f206` (**12**) |
| base | task base `main @ 9bf6422`; ⚠ **rebased onto `main @ dcd667e`** after `gen-reencode-tolerance` merged mid-task. Worktree `../CHLU-runner`. |
| project / region | `forgisprova` / `us-central1` (`us-east4` probed) |
| image, acceptance 1+3 | `…/chlu-cuda@sha256:010435659b510551b64d2008361ac5c906bf127246d0d4fc678b8c204385c44c` — the **shipped** pin (`chlu` @ `84e1ab8`; `exp_cluformer_pilot` is byte-identical to `main`'s, which is why this run's provenance is exact) |
| interim image | `…@sha256:16334204…12676` — `chlu` = `main`, still no h5py. Superseded; **delete**. |
| ⛔ `IMAGE_DIGEST` | **left pointing at the old `@sha256:01043565…5c44c`** — not my file, and shared state a peer spoke was launching through (reconciliation 1). |
| ⭐ **final image** | `…/chlu-cuda@sha256:3f9d8ec681e2179ff563ac534d95bb3da976cd542803b8ba1626bd7def3c739e` — built from the **rebased** branch (`chlu` = `main @ dcd667e`, i.e. **with the new re-encode tolerance**) **plus `--extra gen` (h5py)**. `gencell6`/`gencell7` used it. ⛔ **This is the one the Hub must pin** (§8). |
| interim image 2 | `…@sha256:496b765c…86186` — `c7a6980` + h5py, pre-rebase. Superseded; **delete**. |
| boot image | `common-cu129-ubuntu-2204-nvidia-580-v20260909`; measured driver **580.178.04**, NVIDIA L4 23034 MiB |
| card | **L4, spot**, `us-central1-a`, `g2-standard-8`, $0.343/h list |
| experiment (acceptance 1) | `EXP=cluformer` → `chlu.experiments.exp_cluformer_pilot`, `--stage s1 --scale toy`, **units (seeds) 0 1 2**, `--max-hours 1` |
| experiment (end-to-end) | `EXP=gen_cell` → `chlu.experiments.exp_gen_cell`, `--dataset shapes3d --pretext recon --stage cell --quick`, unit 0, artefacts `gs://chlu-cloud-forgisprova/artefacts` |
| dec_recipe | `v1_capacity_schedule` asserted by name **and** by `pkl_sha256` per seed against `winner_hashes.json` |
| supervisor caps | `--max-attempts 30 --max-usd 3.00 --backoff-s 90 --backoff-max-s 240` (pilot1) |
| local env | main venv reused (protocol §4); **no worktree `uv sync`**; `uv.lock` untouched |
| ⛔ NOT done | no `chlu/**`, `pyproject.toml`, `uv.lock`, `ckpt.py`, `stage_gcs.sh`/`verify_gcs.sh`, `scripts/csf3/**`, `IMAGE_DIGEST` modified · no other team's instance touched · no push, no PR · no science |

---

## 1. What I did

| # | deliverable | where | status |
|---|---|---|---|
| 1 | ⭐ **a real entry point** — `EXP=` resolved through one registry, args passed through | `startup.sh` §0 + §15 | ✅ |
| 2 | the five `CHLU_*` roots, `${VAR:-default}`, asserted at job start **by name** | `startup.sh` §1 | ✅ |
| 3 | ⛔ the decoder recipe asserted **by name AND by hash** | `startup.sh` §2 | ✅ |
| 4 | the unit journal — **`ckpt.py` reused**, not a second mechanism | `startup.sh` §14 | ✅ |
| 5 | ⭐⭐ **the relaunch supervisor**, two hard caps, stockout back-off | `run_until_done.sh` **(new)** | ✅ |
| 6 | the dataset fetched on the VM (the bridge's decision, wired here) | `startup.sh` §9b | ✅ |
| 7 | tests | `tests/test_cloud_runner.py` **(new)** — **93 passed** with the peer's file | ✅ |
| 8 | runbook §5.1–5.3 | `README.md` | ✅ |

⭐ **One launcher, parameterised.** Adding an arm is one row in `exp_row()`; `launch.sh` *asks*
`startup.sh --check exp <name>` rather than holding a second copy, so the laptop and the VM cannot
disagree. ⛔ There is no second launcher and no second checkpoint mechanism — `ckpt.py` is
byte-unchanged (asserted by a test that diffs it against `main`).

---

## 2. ⭐ ACCEPTANCE 1 — a real CHLU experiment, on a GPU, to completion

**`EXP=cluformer` → `chlu.experiments.exp_cluformer_pilot --stage s1 --scale toy`, seeds 0 1 2, on a
SPOT NVIDIA L4** (`g2-standard-8`, `us-central1-a`, driver 580.178.04). ⛔ Not the synthetic payload.

```
[exp] cluformer -> module=chlu.experiments.exp_cluformer_pilot out_mode=sweep needs_artefacts=0
[image] built from commit 84e1ab8 · NVIDIA L4, 580.178.04, 23034 MiB
wrote /var/chlu/out/cluformer/pilot_toy_seed0_S1.json (149s, stages S1)   [sweep] unit 0 ok in 159s
wrote /var/chlu/out/cluformer/pilot_toy_seed1_S1.json (153s, stages S1)   [sweep] unit 1 ok in 163s
wrote /var/chlu/out/cluformer/pilot_toy_seed2_S1.json (137s, stages S1)   [sweep] unit 2 ok in 147s
[result] {"complete": true, "units_banked": ["0","1","2"], "units_requested": ["0","1","2"],
          "dec_recipe": "v1_capacity_schedule", "sweep_wall_s": 332, ...}
```
**Artefacts synced back** to `.claude/outputs/gen-cloud-runner/jobs/pilot1/cluformer/` — three real
per-seed manifests (~34 KB each) plus the partials. ⛔ **`--stage s1 --scale toy` is the acceptance
criterion "the block does not collapse" at toy scale — it is plumbing evidence and no number from
it enters any table.**

**Card L4 (spot) · wall 1084 s of VM life across two attempts · $0.10.**

---

## 3. ⭐⭐ ACCEPTANCE 2 — UNATTENDED ACROSS PRE-EMPTION

⛔ **The machine was destroyed, not the process.** At 20:20:58 UTC, with unit 0 banked and **unit 1
in flight**, I ran `gcloud compute instances delete chlu-pilot1`. I then touched nothing.

| time (UTC) | what happened | who did it |
|---|---|---|
| 20:18:46 | `[journal] unit 0 banked (1 done) -> gs://…/pilot1/journal` | the VM |
| 20:20:03 | `=== [sweep] unit 1 …` starts | the VM |
| ⭐ 20:20:58 | **`Deleted …/instances/chlu-pilot1`** — the VM ceases to exist | **me, once** |
| 20:21:26 | `attempt 1 ended WITHOUT result.json (890s, ~$0.0848)` | **the supervisor** |
| ⭐ 20:21:28 | `=== attempt 2/30` — **relaunched itself** | **the supervisor** |
| 20:24 | `[resume] restoring gs://…/pilot1/out -> /var/chlu/out` | the new VM |
| ⭐⭐ 20:24 | `[journal] fingerprint 406073d0fc73fb29 :: 1 unit(s) already done: 0` | the new VM |
| ⭐⭐ 20:24 | **`[sweep] unit 0 already banked -- skipping`** | the new VM |
| 20:29 / 20:31 | units 1 and 2 run and bank | the new VM |
| ⭐ 20:31:59 | `result.json` written, `"complete": true` | the new VM |
| 20:32:15 | `✅ attempt 2: result.json present. 645s, ~$0.0615` → `SUPERVISOR FINISHED: done` | the supervisor |

```
attempts used: 2 of 30   (stockouts 0, incomplete 1)
estimated spend: $0.1463 of $3.00
```
⭐ **A new VM picked up the checkpoint and re-ran only the unit that was in flight.** The journal is
three `ckpt_*.npz` objects at `gs://…/pilot1/journal/` (steps 1, 2, 3 — `keep=3`), written by
`ckpt.py`, not by anything new.

---

## 4. ACCEPTANCE 3–5 — every gate demonstrated firing

Full transcript: `.claude/outputs/gen-cloud-runner/refusals.txt`. Condensed:

### 4.1 The five roots (acceptance 3) — ⭐ **on a real VM, and before the GPU**
`gate1`, a spot L4 pointed at a deliberately absent prefix. **Six seconds from boot to refusal**,
ahead of the driver wait, the docker install and the 7 GB pull:
```
=== chlu startup 2026-09-15T20:36:58Z on chlu-gate1 ===
[artefacts] fetching gs://…/artefacts-deliberately-absent -> /var/chlu/artefacts
FATAL[startup]: gen artefact root(s) MISSING, by name:
    CHLU_GEN_DATA_ROOT -> /var/chlu/artefacts/data/gen
    CHLU_GEN_BRIDGE_ROOT -> /var/chlu/artefacts/outputs/gen-feature-bridge/runs
    CHLU_GEN_STORE_RUNS -> /var/chlu/artefacts/outputs/gen-store-grouping/runs
    CHLU_GEN_DECLIFT_DIR -> /var/chlu/artefacts/outputs/gen-decoder-lift/winner
=== finishing rc=2 2026-09-15T20:37:04Z ===
```
⚠ **That run also found a defect in my own check**: `CHLU_CLAUDE_ROOT` passed *vacuously*, because
`startup.sh` creates that directory itself. An **empty** root is now refused too (`8980072`), which
the offline check confirms names all five.

### 4.2 The decoder (acceptance 4) — refused three independent ways
Against the **real** `gen-decoder-lift/winner`:
```
POSITIVE  seed_0/1/2: recipe v1_capacity_schedule · pkl sha256 bcf39c42…/7a31fa49…/55957abd… OK
BY NAME   dec_recipe='baseline'           -> rc=5  "…FAILED the gate… is VOID"
BY NAME   dec_recipe='v2_object_weighted' -> rc=5  (⛔ "not baseline" is not enough: v2 also LOST)
BY NAME   staged decoder declares 'baseline' with the flag correct -> rc=5
BY HASH   one flipped bit in decoder_v1_s0.pkl -> rc=5
          sha256 97d88b91… != winner_hashes bcf39c42…   (seed_1 still verified OK)
```
⭐ **And it passed on the real staged bytes on a VM** (`gencell4`/`gencell5`), after which
`exp_gen_cell`'s **own** guard independently agreed: `⭐ decoder: winner recipe=v1_capacity_schedule
hash=01ef91f75eeaae7b… (hash + recipe asserted)`.

### 4.3 Both caps (acceptance 5) — ⭐ **demonstrated refusing LIVE, against the production launcher**
⛔ Not stubs. Real `launch.sh`, real GCP calls, **$0.00 spent** in both:
```
SPEND CAP    spent so far $0 + next attempt ceiling (1h x $0.343/h) = $0.343 > cap $0.10
             === SUPERVISOR FINISHED: spend_cap   attempts used: 0 of 10   (nothing created)
ATTEMPT CAP  (tag-pinned image, so the launcher refuses and no VM is ever created)
             attempt 1/3 … 2/3 … 3/3, each "created NO VM … $0.00", backoff 1s → 2s → 2s (capped)
             ⛔ REFUSING to relaunch: attempt cap reached (3 of 3 used)   spend $0 of $100
QUOTA        a100-80 spot has 0 quota in every region → STOPPING, 1 of 9 attempts, $0
PAYLOAD      gencell2: two payload failures, no pre-emption → STOPPING, 2 of 4 attempts
STOCKOUT     gate1: live, 2 real region-wide stockouts, backoff 120s → 240s, charged $0.00
```
The spend cap is checked against **spend so far plus the ceiling of the attempt about to start**, so
it cannot be breached by exactly the attempt that tests it. A card with no price in `cards.json` is
refused outright — ⛔ *a loop whose cost cannot be computed is an unbounded loop wearing a cap.*
The stub-driven equivalents are in `tests/test_cloud_runner.py`, which is how the caps are asserted
without spending the budget they protect.

---

## 5. ⭐ THE END-TO-END LEG — precondition SATISFIED, and what it found

⭐ **The precondition landed mid-task and is met**, so this leg RAN (it is not `NOT RUN`):
`STAGED-GCS.json` — `status="complete"`, `checksum_match=true`, `n_load_failed=0`, `n_missing=0`,
`loaders_all_opened=true`, 44 files, 2.097 GB, verified on a VM. ⭐ Its `artefact_root` is
`/var/chlu/artefacts` — **exactly the default this spoke's `startup.sh` exports, independently.**
The two halves agreed without a negotiation.

`EXP=gen_cell` → `exp_gen_cell --dataset shapes3d --pretext recon --stage cell --quick`, spot L4,
against `gs://chlu-cloud-forgisprova/artefacts`. **Status: the runner's half is green end to end;
the arm is blocked inside `chlu/**`, which is out of scope.** Four VMs, each one finding the next
defect — ⭐ *every one of them a real gap in the inherited path, not in code I wrote:*

| run | got as far as | what it found |
|---|---|---|
| `gencell1` | the GPU, 11 s into the arm | ⛔⛔ **the image has no `h5py`** — `chlu/data/shapes3d.py` refuses to open the HDF5 dataset ⇒ **no gen experiment could run on GCP at all.** Fixed: `Dockerfile` now syncs `--extra gen` and asserts the import at build time |
| `gencell2` | the arm's own decoder guard, which **passed** | ⛔ **nothing downloaded `3dshapes.h5`.** The bridge decided VM-download-not-staged and handed the wiring to `startup.sh`; it was not there. Fixed (§1 row 6) — and the same job exposed the **stale-sentinel** bug (§6.1) |
| `gencell4` | the encoder cache check (old gate) | `max \|delta\| = 0.00390625` at default precision |
| `gencell5` | the encoder cache check (old gate) | `max \|delta\| = 0.001953125` with `JAX_DEFAULT_MATMUL_PRECISION=highest` + deterministic XLA |
| ⭐ `gencell6` | the encoder cache check (**the new per-element gate**, after rebasing onto `main @ dcd667e` and rebuilding) | **27332/131072 elements differ (20.853%), max 4389.000 float16 ULP** — ⚠ **this refutes my first one-ULP reading** (§5.1) |

**Everything the runner owns works on the real artefact set:**
```
[artefacts] all five roots present: CHLU_CLAUDE_ROOT=/var/chlu/artefacts
[decoder] seed_0: recipe v1_capacity_schedule · pkl sha256 bcf39c42bc7885d8… OK (1516227 params)
[dataset] shapes3d/3dshapes.h5: 267573662 bytes sha256=0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785
[     0.1s]   ⭐ decoder: winner recipe=v1_capacity_schedule hash=01ef91f75eeaae7b… (hash + recipe asserted)
[     0.1s]   ⭐ external classifier c8741ac417d88ebf… (978873 params) — THE instrument
```

### ⛔⛔ 5.1 THE BLOCKER — and ⚠ **A CORRECTION TO MY OWN FIRST READING OF IT**

⚠ **Read this before quoting anything from an earlier draft of this section.** I first concluded
that the disagreement was *exactly one float16 ULP*. **That was wrong, and a better instrument — one
a peer spoke landed on `main` while I worked — refutes it.** The retraction and the replacement are
both below, because the wrong version is the kind that propagates.

**What I measured first (gencell4 / gencell5, the old gate).** `_assert_cache_reencodes` demanded
`max|delta| == 0.0` exactly. On an L4: `0.00390625` at default precision, `0.001953125` with
`JAX_DEFAULT_MATMUL_PRECISION=highest` + deterministic XLA. I noted that the cache is **float16**
(`feature_cache.json`: `dtype=float16`, `D=2048`, 480 000 rows), that the 64 probe rows top out at
**4.13 ≈ 2²**, and that **one float16 ULP there is exactly 2⁻⁸ = 0.00390625** — and inferred a
one-ULP rounding flip.

⛔ **The inference does not follow.** Matching the *max absolute delta* against the ULP of the
*largest value in the block* only works if the deviating element **is** that largest value. The old
gate reported a single scalar, so nothing in my evidence said which element moved. It was a
coincidence of magnitudes presented as a mechanism.

⭐ **What the new gate measures (gencell6, per-element, on the same card and the same rows):**
```
cache re-encode check: 64 rows, 27332/131072 elements differ (20.853%),
max |delta| = 0.00390625 = 4389.000 float16 ULP   (bounds: 1 ULP, 1.0% of elements)
```
**4389 ULP, and a fifth of the block.** The peer's tolerance scales each element by `np.spacing` of
**its own** cached value, so an element whose cached value is small has a correspondingly small ULP —
and the deviation is spread across the block, not parked on the one large element. ⛔ **This is not a
rounding flip.**

⭐ **And the precision knob is most of the answer, which the old scalar could never have shown.**
Same card, same rows, same gate, one variable (`gencell6` vs `gencell7`):

| container env | elements differing | max ULP | vs the bounds (1 ULP, 1%) |
|---|---|---|---|
| default | **27332/131072 = 20.853%** | **4389.000** | fails **both**, by ~3–4 orders |
| `JAX_DEFAULT_MATMUL_PRECISION=highest` + `--xla_gpu_deterministic_ops --xla_gpu_autotune_level=0` | ⭐ **1264/131072 = 0.964%** | **53.000** | ⭐ **PASSES the fraction bound**, fails ULP by **53×** |

⭐ Forcing precision removes **95%** of the differing elements and **99%** of the ULP excess — the
bulk of the disagreement is reduced-precision arithmetic, exactly as the ~1e-3 relative signature
says. ⛔ **But it does not clear the gate**: a residual handful of elements (just under 1%) are still
tens of ULP out, and those are elements whose *own* cached magnitude is small (53 ULP at
`max|delta| = 0.00195312` puts the offending element's magnitude near 0.04), where any absolute error
is enormous in ULP terms. The matmul knob does not govern the cuDNN **convolution** algorithm, which
is the remaining suspect.

⚠ **The consequence the Hub most needs.** The newly-landed one-ULP tolerance was derived from a CSF3
**L40S** incident of **1 element in 131072 (7.6e-6)** — a genuine rounding flip. ⛔ **It does not
admit the L4 case and should not**: 20.853% of elements at 4389 ULP is exactly what that gate exists
to convict. ⇒ **`gen_cell` / `gen_store` still do not run on an L4**, and anyone reading
`gen-reencode-tolerance` as "GPU gen runs are unblocked" would be over-reading it. **The gate is now
right; the L4 is still refused, for a different and better-evidenced reason than I first gave.**
⭐ **The cheap next step is named by the table**: the residual is convolution precision, so the
experiment to run is a conv-precision control (or a CPU-only encode step) — **not** a wider bound.

⛔ **I changed nothing** — `chlu/**` is out of my scope. What the owner needs to decide is whether the
L4's convolution precision can be forced (a cuDNN/`jax.lax.conv` precision control, not the matmul
knob), or whether the encode step runs on CPU, or whether the cache is rebuilt per device.

---

## 6. Defects found and fixed in this spoke's own work

| what | how it was caught | fix |
|---|---|---|
| ⭐ **A refused launch was charged its wall clock, and a quota refusal was retried.** Three attempts spent reprinting one refusal. | running the supervisor against the **live** launcher instead of the stub | spend basis is now *whether `== created chlu-` ever appeared*, which covers stockout, tag-pin and every refuse-before-create path at once; a quota refusal stops the loop (`6c148a9`) |
| ⭐⭐ **§6.1 An attempt inherited the previous attempt's verdict.** `out/EXIT_CODE`/`DONE`/`FAILED`/`PREEMPTED` describe ONE attempt and the sync-back never deletes, so `--wait` returned on a **stale** `EXIT_CODE` 62 s after create — before the VM booted — and the supervisor read a stale `FAILED` and scored a fresh attempt a payload failure. ⚠ **`pilot1` only escaped it because the VM I destroyed never got its `EXIT_CODE` off the machine.** | `gencell2` attempt 2 "finishing" in 62 s | `launch.sh` clears the four per-attempt sentinels **before** it creates anything — the only place that wins the race against a `--wait` poll. ⛔ `result.json` untouched: it describes the job (`c7a6980`… `f0…`) |
| ⭐ **…and that sweep then killed the launcher it protects.** `gcloud storage rm` on an absent object exits non-zero, and on a job's *first* attempt none of them exist ⇒ `set -e` ended `launch.sh` 4 s in, having created nothing. | the supervisor said "created NO VM" three times and charged $0 — **the honest accounting is what made it visible** | `\|\| true` + a test that asserts it |
| ⭐ **An empty artefact root passed.** `startup.sh` makes the dir before fetching into it, so `-e` asked a question it had already answered. | the `gate1` VM named four roots, not five | a directory root must be non-empty (`8980072`) |
| **I broke two of the peer's `test_cloud_launch.py` invariants** — the out/ restore had no `ckpt` exclusion, and a *comment* saying `docker run` defeated the textual "sync loop starts first" proxy. | running their file, not just mine | both fixed in my code, ⛔ **neither test relaxed**; the artefact fetch is deliberately a `cp` so it does **not** inherit the ckpt filter, which would silently drop a staged file named `*ckpt*` (`c7a6980`) |

---

## 7. Git footprint
- **Branch** `agent/experiment-engineer/gen-cloud-runner`, task base `main @ 9bf6422`, worktree
  `../CHLU-runner` (left in place for Hub review). ⚠ **`main` moved 3 commits under me near the end**
  (`gen-reencode-tolerance` — the very check in §5.1), so the branch is **rebased onto `main @
  dcd667e`** per protocol §3.5. The rebase was conflict-free; verified **from the main repo**:
  `git log --oneline main..agent/experiment-engineer/gen-cloud-runner` → **12**, and
  `git diff --stat main..` → **6 files, +1587 / −38** (the deletions are all inside the four files I
  own). ⭐ The `--check` self-test made re-verifying the whole gate set after the rebase free.
- **Commits (12):** `97c567c` the real entry point · `edcbf33` launcher selection + refusals ·
  `934ac8f` the supervisor · `5ed3471` tests + runbook · `6c148a9` spend basis + quota stop ·
  `8980072` empty root · `c7a6980` the peer's two invariants · `e81f720` h5py + payload-failure stop ·
  `21c8660` the VM-side dataset fetch · `2ccdd70` stale per-attempt sentinels · `7536bd3` `|| true` ·
  `043f206` runbook.
- **Files touched — only the six I own:** `scripts/cloud/{startup.sh,launch.sh,Dockerfile,README.md}`,
  `scripts/cloud/run_until_done.sh` (new), `tests/test_cloud_runner.py` (new).
- ⛔ **Untouched:** `chlu/**` · `chlu/config.py` · `pyproject.toml` · `uv.lock` · `scripts/cloud/ckpt.py`
  (a test diffs it against `main` to keep it that way) · `stage_gcs.sh` / `verify_gcs.sh` (the bridge's) ·
  `scripts/csf3/**` · **`scripts/cloud/IMAGE_DIGEST`** (see reconciliation 1). No push, no PR.
- **GCP:** 13 VMs created, **all `chlu-*`, all labelled `owner=chlu`, all gone**
  (`--filter="labels.owner=chlu"` → empty). ⛔ **The four other-team instances (`tsfm-pair-a/b`,
  `jonas-vm-a100`, `a100-dev-machine`) were never touched and are still RUNNING.** Three new images in
  Artifact Registry (~7 GB each, ~$2.10/mo) — §8 says which to keep. ⛔ No project-level setting,
  quota, bucket or registry repo was created or changed; the bridge spoke's `artefacts/` prefix was
  read, never written.
- **Tests:** `tests/test_cloud_runner.py` **93 passed, 0 failed, 0 skipped** *together with* the peer's
  `tests/test_cloud_launch.py` (36 of those 93 are theirs, all still green). 17 s, main venv, no GPU,
  no network, no JAX, no docker. `ruff check scripts/cloud tests/test_cloud_runner.py` → clean.
  ⛔ **The full suite was not run:** nothing added is on any `chlu/**` import path (one new test file
  plus shell scripts), so the suite count is unchanged by construction.

**COST — from the GCE operations log (insert→delete per VM), list price, an estimate not an invoice:**

| instance | sec | hours | $/h | USD |
|---|---|---|---|---|
| `pilot1` att.1 (L4 spot, **destroyed mid-sweep**) | 503 | 0.1397 | 0.343 | 0.05 |
| `pilot1` att.2 (L4 spot, **finished the job itself**) | 581 | 0.1614 | 0.343 | 0.06 |
| `gate1` (L4 spot, the roots refusal on a VM) | 52 | 0.0145 | 0.343 | 0.01 |
| `gencell1` · `gencell2` · `gencell2` att.2 (found: h5py, dataset, stale sentinel) | 338 · 272 · 16 | 0.1738 | 0.343 | 0.06 |
| `gencell4` · `gencell5` (old gate: 2⁻⁸ / 2⁻⁹) | 284 · 290 | 0.1594 | 0.343 | 0.05 |
| ⭐ `gencell6` · `gencell7` (**new gate: 20.853%/4389 ULP vs 0.964%/53 ULP**) | 284 · 278 | 0.1561 | 0.343 | 0.05 |
| 3 × `chlu-builder-*` (`e2-standard-8` image builds) | 3070 | 0.8528 | 0.268 | **0.23** |
| **TOTAL — 13 VMs** | | **1.658 h** | | **$0.50** |

GPU VMs **$0.28** (10), image builders **$0.23** (3). ⛔ **$0.00** was spent by either cap
demonstration, and $0.00 by the ~20 region-wide stockouts across the day — a stockout creates no VM.
Standing cost added: **3 images × ~7 GB ≈ $2.10/month**; §8 item 2 deletes the two superseded ones.

---

## 8. ⛔ [HUB] THE ONE ACTION THIS SPOKE COULD NOT TAKE

⛔⛔ **`scripts/cloud/IMAGE_DIGEST` still points at an image that cannot run gen work.** It is not in
my file list and it is shared state — a peer spoke was launching VMs through this path while I
worked — so I **built and verified the replacement and left the pin alone.** The Hub's action is one
line; ⭐ **no rebuild is needed, the image exists and has run.**

```bash
echo "us-central1-docker.pkg.dev/forgisprova/chlu/chlu-cuda@sha256:3f9d8ec681e2179ff563ac534d95bb3da976cd542803b8ba1626bd7def3c739e" \
  > scripts/cloud/IMAGE_DIGEST
```

| | digest | `chlu` at | h5py | new re-encode tolerance | verdict |
|---|---|---|---|---|---|
| currently pinned | `@sha256:01043565…5c44c` | `8a404de`+ — ⚠ **10 files stale vs `main`** incl. `config.py`, `exp_gen_cell.py`, `exp_gen_store.py` | ⛔ no | ⛔ no | ⛔ cannot run gen; runs last week's code |
| interim | `@sha256:16334204…12676` | `6c148a9` | ⛔ no | ⛔ no | superseded — **delete** |
| interim | `@sha256:496b765c…86186` | `c7a6980` | ✅ yes | ⛔ no | superseded — **delete** |
| ⭐ **the one to pin** | `@sha256:3f9d8ec6…c739e` | **`dcd667e` = `main`** | ✅ **yes** | ✅ **yes** | ⭐ **built, pulled and run on an L4 today** |

Until it is pinned, every `launch.sh` / `run_until_done.sh` call must pass
`--image us-central1-docker.pkg.dev/forgisprova/chlu/chlu-cuda@sha256:3f9d8ec6…c739e` — which is how
`gencell6`/`gencell7` were run, so the escape hatch is exercised, not assumed. Digests in full:
`.claude/outputs/gen-cloud-runner/IMAGE_DIGESTS.txt`.

2. **Delete the three superseded images** (`@sha256:01043565…`, `@sha256:16334204…`,
   `@sha256:496b765c…`) once the pin moves: `gcloud artifacts docker images delete …` — ~$2.10/month.
   ⚠ **Not before**: `@sha256:01043565…` is the currently-pinned one and the peer spoke's acceptance
   evidence refers to it.

---

## 9. Open questions / follow-ups / risks
1. ⛔⛔ **An L4 cannot re-encode the gen feature cache, even under the new tolerance** (§5.1):
   20.853% of elements at max 4389 ULP. ⭐ **This is the single thing standing between GCP and a gen
   run** and it is a `chlu/**` decision, so I changed nothing. ⚠ **It also carries a retraction of my
   own first reading** ("exactly one float16 ULP") — see §5.1 and 7.36; that wrong number must not
   reach the log. The open question is whether the L4's *convolution* precision can be forced.
2. ⭐ **`tests/test_gen_data_splits.py` hides a missing `h5py` as 8 SKIPs.** The pyproject comment
   predicted it; it has now cost a GPU-minute and two image builds. Any environment shipping gen work
   should assert the import. The image does now; the laptop and CSF3 do not.
3. ⭐ **`us-central1` L4 was stocked out region-wide — spot *and* on-demand — for ~35 minutes**, and
   `us-east4` with it. ~14 stockouts across the day. ⭐ **`us-east4` also has
   `PREEMPTIBLE_NVIDIA_L4_GPUS = 8, all free`** (the peer measured only `us-central1`), so `--region
   us-east4` is a real second pool — ⚠ at cross-region egress on the 7 GB pull and the 2 GB artefact
   fetch, both of which are free in-region. A second Artifact Registry repo + bucket there would fix
   that, and is a [HEAD] cost decision, not an engineering one.
4. **A stockout consumes an attempt.** Deliberate — the cap must bind — but it means a long stockout
   can exhaust the budget before any real work. If that bites, the fix is a separate
   `--max-stockouts`, not an unbounded loop.
5. ⚠ **Do not edit `startup.sh` while a supervisor is running.** `launch.sh` uploads it as metadata at
   create time, so attempt *N+1* runs a different launcher than attempt *N*. I did this to myself
   mid-`gencell2`; it is now in the runbook.
6. **`3dshapes.h5` is re-downloaded once per VM** (255 MiB, ~20 s, $0.00 — owner-paid egress). Across
   a large sweep that is minutes, not dollars. Staging it would cost storage *and* the egress the
   bridge measured away; the cache lives on the host mount so it is once per VM, not once per unit.
7. **The `--quick`/`toy` scale is the only thing that has run.** Nothing here says a *pilot*-scale job
   fits an L4's 23 GB or finishes inside `--max-hours`. The first real arm should be launched with
   `--wait` and watched once.
8. **`out/job.log_.gstmp`** — a 0-byte artefact of a `gcloud storage cp` interrupted by a destroyed
   VM. Harmless; `fetch_results.sh` copies it down. Cosmetic.

---

## Proposed handover updates (for the Hub)

**§7-CURRENT additions:**
- **7.36 [cloud, standing] ⭐⭐ `gen_reencode`'s NEW TOLERANCE DOES NOT UNBLOCK AN L4, AND SHOULD
  NOT.** `gen-reencode-tolerance` (merged `7a1c4ff`) replaced bit-identity with *one ULP of the cache
  dtype + <1% of elements differing*, derived from a CSF3 **L40S** incident of **1 element in 131072
  (7.6e-6)**. ⭐ **Measured on an L4 today under that new gate** (`gen-cloud-runner` §5.1):
  **27332/131072 elements differ — 20.853% — at max 4389.000 float16 ULP** at default precision. ⛔
  Three to four orders outside the bound on both axes; the gate convicts it correctly. ⭐ **With
  `JAX_DEFAULT_MATMUL_PRECISION=highest` + deterministic XLA, same card and rows: 0.964% and 53.000
  ULP — it PASSES the 1% fraction bound and fails the 1-ULP bound by 53×.** So precision is ~95% of
  the answer and the residual is not matmul: the knob does not govern the cuDNN **convolution**
  algorithm. ⇒ **`gen_cell` / `gen_store` still do not run on an L4.** ⚠ **Do not read
  `gen-reencode-tolerance` as "GPU gen runs are unblocked".** The named next step is a conv-precision
  control or a CPU-only encode step — ⛔ **not** a wider bound.
  ⚠ **RETRACTION, same spoke, same day:** `gen-cloud-runner` first reported this as *"exactly one
  float16 ULP"*, inferred from the old gate's single scalar by matching the max absolute delta to the
  ULP of the **largest value in the block**. **That inference does not follow** — nothing identified
  which element moved — and the per-element gate refutes it. ⛔ **Never quote "one ULP" for the L4.**
- **7.37 [ENV, standing] The cloud image shipped without `h5py` and therefore could not open
  Shapes3D at all** — no gen experiment could have run on GCP. Fixed (`Dockerfile --extra gen`, import
  asserted at build). ⭐ **The class of bug matters more than the instance:**
  `tests/test_gen_data_splits.py` `importorskip`s h5py, so its 8 real-data tests pass as **SKIPs** and
  nothing goes red. Any environment that ships gen work must assert the import, not skip it.
- **7.38 [cloud, standing] ⛔ `scripts/cloud/IMAGE_DIGEST` is stale on two axes** — its `chlu` predates
  7 `chlu/` files on `main` (incl. `config.py`, `exp_gen_store.py` — i.e. **the old bit-identity
  re-encode gate** — and a +592-line `exp_gen_cell.py`) and it has no `h5py`. The
  replacement `@sha256:3f9d8ec6…c739e` (`chlu` = `main @ dcd667e`, h5py, new tolerance) is **built and
  has run on an L4**; the pin move is one line
  (`gen-cloud-runner` §8) and was deliberately left to the Hub as shared state.
- **7.39 [cloud, standing] `us-east4` also has `PREEMPTIBLE_NVIDIA_L4_GPUS = 8, all free.** A second
  L4 pool matters: on 2026-09-15 `us-central1` L4 was stocked out **region-wide in both pools** for
  ~35 minutes, and `us-east4` briefly with it. ⚠ Cross-region costs egress on the 7 GB image pull and
  the 2 GB artefact fetch, which are free in-region.
- **7.40 [cloud, standing] A GCS output prefix is additive, so per-attempt sentinels leak across
  attempts.** `out/EXIT_CODE`/`DONE`/`FAILED`/`PREEMPTED` describe one attempt; `launch.sh` now clears
  them before creating a VM. Anything else that polls a job's bucket state must assume staleness.

**New capability for the running log:**
⭐ **GCP now runs CHLU experiments unattended.** `scripts/cloud/run_until_done.sh --job <j> --card l4
--spot --exp <exp> --units '0 1 2'` drives a real arm to completion across pre-emption with **no
human in the loop**, under a hard attempt cap and a hard spend cap; `startup.sh` holds the single
`EXP` registry (adding an arm = one row), asserts the five `CHLU_*` roots and the decoder recipe by
name **and** hash before the GPU, fetches the dataset the bridge chose not to stage, and banks
completed units in a `ckpt.py`-backed journal so a relaunch resumes instead of restarting.
**Demonstrated:** a spot L4 destroyed mid-sweep, relaunched by itself, skipped the banked unit, and
finished. ⛔ **Gen arms are blocked by 7.36, not by the cloud path** — every artefact, root, decoder
and dataset check passes on a VM and the arm reaches its own preflight before the cache gate stops
it. ⛔ **And `scripts/cloud/IMAGE_DIGEST` must be repointed before any of this is usable by default**
(7.38).
