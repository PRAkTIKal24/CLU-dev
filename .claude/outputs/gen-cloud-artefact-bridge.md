# gen-cloud-artefact-bridge — experiment-engineer report

**Task + acceptance criterion:** put the gen artefact set in GCS and prove it OPENS there, by
porting `scripts/csf3/stage_artefacts.sh` + `verify_artefacts.sh` — **transport only**.

**Status: done.** All six acceptance items met. The 44-file set (2,097,001,955 B) is in
`gs://chlu-cloud-forgisprova/artefacts/`, byte counts matching the manifest exactly, and
**`verify_gcs.sh` ran ON A VM green: 0 missing, 0 checksum mismatches, 0 failed to load.** The
verifier was **demonstrated failing both ways on that same VM** — a flipped byte (1 mismatch,
0 load failures) and a byte-perfect-per-manifest-but-truncated pickle (0 mismatches, 1 load
failure). `3dshapes.h5` is **VM-downloaded, not staged**, sha256 asserted by the production path
and the file opened through `open_images()`. **Total spend: $0.02 one-off, $0.042/month recurring.**

## ⚠ RECONCILIATION LIST (first-10-lines rule, protocol §5) — **two sites, an owner is needed**
1. ⭐⭐ **The shipped image cannot open either image dataset.** `h5py` is in the **`gen` extra**
   (`pyproject.toml:60`), and `scripts/cloud/Dockerfile` builds `uv sync --frozen --no-dev
   --extra cuda`. **Measured on the VM:** `ImportError: Shapes3D is a single HDF5 file and requires
   h5py`. The 44 manifest artefacts are unaffected (every h5py import in `chlu/` is lazy), but
   **any gen job that touches shapes3d or dSprites dies on the first image read.** Owner: whoever
   holds `scripts/cloud/Dockerfile` (`gen-cloud-runner` this wave) — one word, `--extra gen`,
   and a rebuild+re-digest. ⛔ **Not mine to edit** (§"Files you own").
2. **`scripts/csf3/verify_artefacts.sh`'s `SKIP_LOAD` path yields `status="failed"` even when every
   checksum matches** (`all_ld = False` by construction). Ported verbatim, deliberately — it is
   arguably right ("checksums only is NOT a pass") but it is a *silent* right, and a reader of
   `STAGED.json` cannot tell that kind of `failed` from a real one without reading `skip_load`.
   Cosmetic, cross-platform, needs one owner or neither file should change.

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** ⛔ **none — instrument/infrastructure.** No arm, no mechanism, no number here enters
  any table. ⛔ **No PREREG:** the acceptance is a mechanism, not a measured ratio/exponent/law.
- **Laundering control:** n/a for a dial. The claim "the verifier works" has the only control that
  matters and it ran: **the verifier demonstrated FAILING, twice, on the same VM, in both of the
  two independent ways it can fail.** A checker that cannot fail proves nothing.
- **Falsifies the claim:** a green verify that a real gen job then contradicts by failing to open a
  staged artefact; or a file set here that differs from the CSF3 manifest.
- **Does NOT falsify:** GCS list-price drift, zone stockouts, and the image's missing `gen` extra
  (that is a finding about the *image*, not about the bridge — reported, not papered over).

---

## 0. FLAG PROVENANCE (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/gen-cloud-artefact-bridge` @ **`02919ef`, `772cfce`** (2). Task base `main @ 9bf6422`; **`main` advanced to `7a1c4ff` during the work** (another Hub merged `gen-reencode-tolerance`), so the branch is **rebased onto `main @ 7a1c4ff`** per protocol §3.5 — conflict-free, and the diff vs `main` is **purely additive (3 files, 1139 insertions, 0 deletions)**. |
| worktree | `../CHLU-bridge` (as tasked). `gen-cloud-runner` ran `launch.sh` concurrently; **file sets disjoint, no collision** (`git status` clean throughout). |
| project / region / zone | `forgisprova` / `us-central1` / `us-central1-a` |
| bucket | `gs://chlu-cloud-forgisprova` — **US-CENTRAL1, region (not multi-region), STANDARD** |
| **prefix** | **`artefacts/`** — i.e. `gs://chlu-cloud-forgisprova/artefacts/<relpath>`, disjoint from the runner's `jobs/` |
| manifest | `chlu-gen-stage-manifest/1`, generated `2026-09-15T17:34:53Z` by **`scripts/csf3/stage_artefacts.sh`** (⛔ not re-derived here), **44 files, 2,097,001,955 B (1.95 GiB)**, sha256 of the manifest `6d02d325…75485e` — **byte-identical in the bucket** |
| axes | `datasets=[shapes3d] pretexts=[recon] seeds=[0,1,2]` |
| `open_with` histogram | `eqx_pickle` 18 · `json` 14 · `npy` 6 · `feature_cache` 3 · `c_invalid` 1 · `winner_hashes` 1 · `store_plan` 1 |
| verify VM | `chlu-verify-gcs`, **`e2-standard-4` (4 vCPU, 16 GB, NO GPU)**, `us-central1-a`, Ubuntu 22.04 LTS, 60 GB pd-balanced, `--scopes=cloud-platform`, `labels=owner=chlu,job=verify-gcs` |
| run image | `us-central1-docker.pkg.dev/forgisprova/chlu/chlu-cuda@sha256:0104356…5c44c` (**the shipped digest, unmodified**), python 3.11.16 in-container |
| VM lifetime | created `20:03:14Z`, phases done `20:09:20Z`, **deleted `20:11:04Z`** — 7 min 50 s. ⭐ `gcloud compute instances list --filter="labels.owner=chlu"` → **0 items**. No pre-existing instance touched. |
| local env | **main venv reused** (protocol §4); no worktree `uv sync`; gcloud CLI 585.0.0 |
| ⛔ NOT done | no edit to `scripts/csf3/**` (**byte-unchanged**) · no edit to `launch.sh`, `startup.sh`, `run_until_done.sh`, `Dockerfile`, `ckpt.py` · no `chlu/**`, no `config.py` · no second launch path · no second checkpoint mechanism · no science |

---

## 1. ⭐ ACCEPTANCE 2 — THE SET IS ACTUALLY IN GCS

```
gs://chlu-cloud-forgisprova/artefacts/   TOTAL: 46 objects, 2,097,038,481 bytes (1.95 GiB)
   = 44 artefacts  2,097,001,955  ⭐ EXACTLY the manifest's total_bytes
   +  1 manifest       19,062     (stage_manifest.json, sha256-identical to the laptop's)
   +  1 verify script  17,464     (_verify/verify_gcs.sh — how it reached the VM)
```
Upload 19:57:30Z → 20:02:5xZ (~5.4 min, ~6 MiB/s home uplink). The VM read the same set back at
**364.0 MiB/s** (`gcloud storage rsync -r`, ~6 s for 1.95 GiB) — same-region GCS→GCE.

⭐ **Idempotent, proven**: an immediate second run skips everything.
```
[stage-gcs] already there : 44 files (2097001955 bytes) — skipped
[stage-gcs] to upload     : 0 files (0 bytes, 0.00 GiB)
```
⚠ **Why the skip key is sha256-in-custom-metadata and not a GCS hash.** `features.npy` (1.83 GiB)
was uploaded as a **parallel composite object**, and composite objects **carry no `md5Hash`**. An
md5-based idempotence check would have failed on exactly the one file where re-uploading actually
costs something. Each object therefore carries `metadata.sha256 = <manifest sha256>`, and one
`gcloud storage ls --recursive --json` decides the whole plan in a single API call.

## 2. ⭐⭐ ACCEPTANCE 3 — `verify_gcs.sh` RAN ON A VM, GREEN (⛔ not a laptop simulation)

`STAGED-GCS.json`, written on `chlu-verify-gcs` and copied back to the bucket:
```
schema               chlu-gen-staged/1        host                 chlu-verify-gcs
status               complete                 artefact_root        /var/chlu/artefacts
checksum_match       True                     n_files              44
loaders_all_opened   True                     n_missing            0
n_checksum_mismatch  0                        n_load_failed        0
total_bytes_on_disk  2097001955               verified_at          2026-09-15T20:08:30Z
```
All 44 opened **through their real production loaders** on that machine — `load_model` (18
equinox pickles, jax-backed), `load_feature_cache` (which runs its OWN `features_sha256` gate, i.e.
the production checksum, not ours), `np.load`, `load_store_plan`, `load_winner_hashes`,
`load_c_requests`. The five `CHLU_*` roots were exported by the script itself, so the run proves
**the same resolution a real gen job on that VM will get**.

## 3. ⭐ ACCEPTANCE 4 — THE VERIFIER DEMONSTRATED FAILING, BOTH WAYS, ON THE SAME VM

| demo | what was done | result | `exit` |
|---|---|---|---|
| **checksum gate** | flipped **one byte** at offset 2000 of `…/seed_1/feature_rows.npy` (it still `np.load`s fine) | `n_checksum_mismatch=1`, `n_load_failed=0`, `loaders_all_opened=True`, `status=failed`<br>`sha256 MISMATCH: ce9a066b19554866 != d7177d099dc76fde (expected)` | **1** |
| ⭐⭐ **loader gate** | truncated `…/seed_0/store_s0.pkl` to 268,464 B **and re-hashed it IN the manifest** — so the checksum gate *passes it* | `n_checksum_mismatch=0`, **`checksum_match=True`**, `n_load_failed=1`, `status=failed`<br>`UnpicklingError: pickle data was truncated` | **1** |

⭐ The second row is the whole reason the verify opens files: **a byte-perfect artefact that the
loader rejects is still a dead run**, and a pure checksum gate reports it as healthy. Evidence:
`VM-FAIL-corrupt-byte.json`, `VM-FAIL-unloadable.json` (both in the output dir and in
`gs://…/artefacts/_verify/`). ⛔ **The GCS objects were never mutated** — both corruptions were on
the VM's local copy (`--no-fetch`), and the bucket's manifest still hashes to `6d02d325…75485e`
with `store_s0.pkl` at its full 536,929 B.

## 4. ⭐ ACCEPTANCE 5 — `3dshapes.h5`: **VM-DOWNLOADED, NOT STAGED**

**Decision: the VM downloads it**, exactly as a CSF3 compute node does. Three reasons, in order:

1. ⭐ **The production download path asserts the checksum itself.** `download_shapes3d()` →
   `download_file(url, dest, sha256=H5_SHA256)` **plus** an exact `H5_BYTES` check. A truncated
   fetch **cannot** look fine. Without that property this would be a shortcut; with it, it is a
   decision.
2. **It is free and leaves nothing behind.** Upstream is a **public bucket** (`H5_URL =
   storage.googleapis.com/3d-shapes/3dshapes.h5`) whose egress its owner pays, and GCS→GCE inside
   one region is free. Staging it would buy **255 MiB × $0.020/GiB-month of our storage, forever**,
   plus a 255 MiB laptop upload, to avoid a download that costs us $0.00.
3. ⭐ **It keeps the platforms comparable:** neither stages a dataset, so the CSF3 manifest's
   `not_staged` clause stays true on both.

**Measured on the VM** (`STAGED-GCS.json → dataset`):
```
sha256_expected 0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785
sha256_actual   0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785   ✅ match
bytes 267,573,662 = expected · downloaded True · opened True
detail: labels_shape [480000, 6] · images_shape [480000, 64, 64, 3]
root:   /var/chlu/artefacts/data/gen/shapes3d
```
(The same sha256 is asserted against the laptop copy, and a test pins it to
`chlu/data/shapes3d.py::H5_SHA256` so the bridge's documented hash cannot drift from the loader's.)

⚠ **The open needed a runtime `h5py` install** — see reconciliation item 1. `--allow-pip`
installed `h5py==3.16.0` (`uv pip`, 4.8 MiB, ~0.5 s) and the file then opened; the JSON records
`h5py_installed_at_runtime: true` and keeps the original `ImportError` in `error` so the gap is
visible rather than laundered.

## 5. ACCEPTANCE 1 — the refusals, demonstrated against the REAL manifest

| case | exit | output |
|---|---|---|
| **partial** (one required file removed) | **3** | `⛔ outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/encoder.pkl` / `loader: chlu.utils.checkpoints.load_model via exp_gen_store.fit_seed / exp_gen_cell.run_seed / exp_gen_certificate (param_hash asserted …)` — ⭐ **the file AND its loader** |
| **empty** (a root holding only the manifest — the worktree case) | **4** | `the artefact set is EMPTY … ⚠ Not one of the 44 expected artefacts exists there.` |
| **stale** (a file edited after hashing) | **5** | `1 file(s) no longer match the manifest — it is STALE … ⛔ Re-derive it: --refresh-manifest` |
| `--dry-run` | 0 | 44-line plan + cost, **zero `gcloud storage cp` calls** (asserted in tests) |

⚠ **One bug this port had and no longer has:** the first version treated "empty" as *an empty
manifest*, so a wrong `--claude-root` produced a 44-line missing-file dump instead of the
one-line wrong-root diagnosis the CSF3 script is careful to give. Fixed in `772cfce`; emptiness is
now "not one of them is here", the CSF3 semantics. Logs: `refusals.log`, `dry-run.log`.

## 6. ⭐ THE CONTRACT WITH `gen-cloud-runner` — exact prefixes to export

⛔ I did **not** edit `startup.sh`. These are the five variable names and the exact values, also
emitted by **`scripts/cloud/stage_gcs.sh --print-env`** (and pinned by two tests, one of which
asserts the names match `stage_artefacts.sh`'s `env_for_the_job` block):

```sh
# bucket gs://chlu-cloud-forgisprova   prefix artefacts/
gcloud storage rsync -r gs://chlu-cloud-forgisprova/artefacts /var/chlu/artefacts
export CHLU_CLAUDE_ROOT="/var/chlu/artefacts"
export CHLU_GEN_DATA_ROOT="/var/chlu/artefacts/data/gen"
export CHLU_GEN_BRIDGE_ROOT="/var/chlu/artefacts/outputs/gen-feature-bridge/runs"
export CHLU_GEN_STORE_RUNS="/var/chlu/artefacts/outputs/gen-store-grouping/runs"
export CHLU_GEN_DECLIFT_DIR="/var/chlu/artefacts/outputs/gen-decoder-lift/winner"
```
`JOB_ART_ROOT` overrides the root if `/var/chlu/artefacts` is inconvenient (it is the path
`startup.sh` already bind-mounts as `/var/chlu`, which is why it was chosen). ⭐ The **names and
the relative layout are CSF3's verbatim**; only the root moves. The dataset lands under
`$CHLU_GEN_DATA_ROOT/shapes3d/` via `chlu data shapes3d --download` (or
`verify_gcs.sh --with-dataset`), **which is the only place the two platforms differ.**

## 7. ⭐ COST

| line | amount |
|---|---|
| **GCS storage, recurring** | 2.097 GB × $0.020/GB-mo = **$0.0419/month** ($0.50/year) |
| upload (ingress) | **$0.00** |
| GCS → GCE read, same region (the VM's 1.95 GiB fetch) | **$0.00** |
| `3dshapes.h5` (255 MiB, public bucket → our VM) | **$0.00** — owner-paid egress; ⭐ this is the staging we avoided |
| class-A ops (46 writes) | $0.0002 |
| verify VM: `e2-standard-4` 0.1306 h × $0.134/h | $0.0175 |
| verify VM boot disk: 60 GB pd-balanced × 0.1306 h | $0.0011 |
| **one-off total** | **≈ $0.02** |

⚠ List prices (us-central1, 2026-09-15), estimates and not an invoice — the same caveat
`scripts/cloud/config.sh` carries.

## 8. ACCEPTANCE 6 — tests

`tests/test_stage_gcs.py` — **20 passed, 0 skipped, 0 failed** (4.8 s, main venv,
`PYTHONPATH=<worktree>`; ⛔ no jax import by design — a cold JAX import is ~20 min here and none of
these assertions need it). Run together with the peer's suite: **56 passed** (20 + 36
`test_cloud_launch.py`, unchanged and still green after the rebase).

They drive the real shell script through a **stub `gcloud`** that records every invocation:
dry-run transfers nothing; the three refusals with their exact exit codes; the partial refusal
names the loader; idempotent skip on a matching sha256; re-upload on a mismatching one; the object
layout is `gs://<bucket>/<prefix>/<relpath>`; `--print-env` emits exactly the five roots and they
match `stage_artefacts.sh`'s block; the 3dshapes sha256 equals `chlu/data/shapes3d.py`'s;
**`verify_gcs.sh`'s `LOADERS` keys equal `verify_artefacts.sh`'s key-for-key**; and `STAGED-GCS.json`
is `STAGED.json`'s key set plus exactly `{gcs_root, vm, dataset}`.

---

## 9. What I did — and what I refused to do

| # | deliverable | where | status |
|---|---|---|---|
| 1 | `stage_gcs.sh` — manifest-driven upload, idempotent/resumable, `--dry-run`, 3 refusals | `scripts/cloud/stage_gcs.sh` (new, 401 ln) | ✅ |
| 2 | `verify_gcs.sh` — fetch + re-checksum + **OPEN**, `STAGED-GCS.json`, runs on the VM | `scripts/cloud/verify_gcs.sh` (new, 383 ln) | ✅ |
| 3 | tests | `tests/test_stage_gcs.py` (new, 355 ln) | ✅ |
| 4 | the set in GCS + a green on-VM verify + both failure demos | bucket + `.claude/outputs/gen-cloud-artefact-bridge/**` | ✅ |

⛔ **What I did NOT build, on purpose.** No launcher: `verify_gcs.sh` documents the one-off
`gcloud compute instances create` in its header (the analogue of the CSF3 file's `#SBATCH` block)
and the actual boot script lives in the **output dir as scratch** (`vm_startup.sh`), not in
`scripts/cloud/`. A second launch path would diverge from `launch.sh` silently. ⛔ No second
checkpoint mechanism — nothing here touches `ckpt.py`. ⛔ The artefact set is **never re-derived**:
`--refresh-manifest` shells out to `scripts/csf3/stage_artefacts.sh --manifest-only`, so a refresh
cannot drift from the cluster's, and its exit codes (3/4) are inherited as ours.

## 10. Findings

1. ⭐⭐ **The shipped image has no `h5py`** (reconciliation 1). Measured, not inferred.
2. ⭐ **Composite objects have no md5** — the reason the idempotence key is sha256-in-metadata
   (§1). This bites any future GCS tooling in this repo that reaches for `md5Hash`.
3. **A GPU is not needed to verify.** Every loader is numpy/pickle/json on the CPU; the whole
   verify is 44 files in ~21 s on 4 vCPU (three passes, 20:08:30 / 20:08:52 / 20:09:13 Z). An L4 would have cost 6× for the same answer. ⚠ Worth
   remembering the next time "we need a GPU VM to check something" comes up.
4. **CSF3's `verify_artefacts.sh` `SKIP_LOAD` reports `failed` on a clean set** (reconciliation 2).
5. ⚠ **The bridge is `shapes3d_recon` seeds 0–2 only** — that is what the manifest's axes say. A
   dSprites or `cls` gen job on GCP needs its own `--refresh-manifest --datasets/--pretexts` run
   *and* a re-stage; the stager will not silently ship a set that does not exist.

## 11. Git footprint
- **Branch:** `agent/experiment-engineer/gen-cloud-artefact-bridge` (worktree `../CHLU-bridge`)
- **Commits:** `02919ef` (the port), `772cfce` (empty-vs-partial semantics), **rebased onto
  `main @ 7a1c4ff`**, conflict-free.
- **Files touched:** `scripts/cloud/stage_gcs.sh` (new), `scripts/cloud/verify_gcs.sh` (new),
  `tests/test_stage_gcs.py` (new). **+1139 / −0.** Nothing else in the repo is modified.
- **Commands run:** `stage_gcs.sh --dry-run` / (real) / `--print-env`; `gcloud compute instances
  create|delete chlu-verify-gcs`; `verify_gcs.sh` ×3 on the VM; `pytest tests/test_stage_gcs.py
  tests/test_cloud_launch.py` → 56 passed.
- **Unresolved conflicts:** none. **Left running:** nothing (`labels.owner=chlu` → 0 instances).
- **Artefacts:** `.claude/outputs/gen-cloud-artefact-bridge/` — `STAGED-GCS.json`,
  `VM-FAIL-corrupt-byte.json`, `VM-FAIL-unloadable.json`, `vm-verify.log`, `vm_startup.sh`,
  `dry-run.log`, `stage-upload.log`, `idempotence.log`, `refusals.log`. Same files (bar the logs)
  under `gs://…/artefacts/_verify/`.

## 12. Open questions / follow-ups / risks
- ⭐ **Blocking for any gen job on GCP: the image needs `--extra gen`** (finding 1). The bridge is
  green without it; the *science* is not runnable with it missing.
- The `_verify/` prefix (17 KB of script + 3 JSONs + a log) is evidence, left in place. Delete it
  whenever the Hub is done reading it; `artefacts/` itself should stay.
- ⚠ **A re-stage after re-deriving the manifest is not free of thought:** `--refresh-manifest`
  re-derives from the *current* `.claude/`, so if a bridge run has been re-generated locally the
  stager will (correctly) refuse as STALE and then upload the new bytes. The VM must re-verify
  after any such re-stage — `STAGED-GCS.json` records `manifest_generated_at` precisely so a stale
  verdict is detectable.
- Not tested: a **spot** VM for the verify (irrelevant — 8 minutes) and a **second region**.

## Proposed handover updates (for the Hub)

**§7-CURRENT — new item:**
- **7.30 [OPEN, blocking for GCP gen runs — owner needed]** **the shipped cloud image cannot open
  either image dataset.** `h5py` lives in the `gen` extra; `scripts/cloud/Dockerfile` builds
  `uv sync --frozen --no-dev --extra cuda`. Measured on a VM (2026-09-15): `ImportError: Shapes3D
  is a single HDF5 file and requires h5py`. Staged artefacts are unaffected (all h5py imports in
  `chlu/` are lazy) — **a gen job is not**. Fix is `--extra gen` in the Dockerfile plus a rebuild
  and a new `IMAGE_DIGEST`; ⛔ owned by whoever holds the Dockerfile, not by this spoke. Evidence:
  `.claude/outputs/gen-cloud-artefact-bridge/STAGED-GCS.json → dataset.error`.

**§ cloud path — additions:**
- **The gen artefact set now has a cloud home.** `gs://chlu-cloud-forgisprova/artefacts/` (44
  files, 1.95 GiB, `$0.042/month`), staged by `scripts/cloud/stage_gcs.sh`, verified on a VM by
  `scripts/cloud/verify_gcs.sh`. The file set is still **derived by `scripts/csf3/stage_artefacts.sh`**
  — ⛔ never re-derived — so CSF3 and GCP stage the identical 44 files and a result can move
  between them. `jobs/` (runner) and `artefacts/` (bridge) are disjoint prefixes of one bucket.
- **The five roots a gen job must export on GCP** are in §6 of this report and in
  `stage_gcs.sh --print-env`. ⛔ `3dshapes.h5` is downloaded on the VM, never staged; the
  production loader asserts its sha256 and byte count.
