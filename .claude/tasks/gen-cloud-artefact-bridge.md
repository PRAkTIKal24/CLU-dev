# Task: gen-cloud-artefact-bridge — ⭐ the gen artefacts have NO path to GCP. Build it. Transport only.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-cloud-artefact-bridge.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ **`scripts/csf3/stage_artefacts.sh` + `verify_artefacts.sh` — you are PORTING these, not redesigning them** · `.claude/outputs/gen-cloud-enablement.md`
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** the Hub's tip **after `gen-cloud-enablement` merges** · **Branch:** `agent/experiment-engineer/gen-cloud-artefact-bridge` · **Worktree:** `../CHLU-bridge`
- ⚠ **Budget ≈ 0.5 day.** ⛔ **No sub-agents.** ⛔ No science, no arm, no number that enters any table.

## ⛔ MECHANICAL PRECONDITION — check FIRST, stop if absent
⭐ **SATISFIED 2026-09-15:** `gen-cloud-enablement` is merged (`main @ 9bf6422`); `scripts/cloud/launch.sh`,
`ckpt.py`, `startup.sh` and a digest-pinned image are on your base. ⭐ **Reuse `ckpt.py`; ⛔ never a second
checkpoint mechanism.** ⛔ If either is missing, stop and report `NOT RUN — precondition absent`. ⛔ **Do not invent a second launch
path**; a peer built that one and two will diverge silently.

## Why this exists (measured, 2026-09-15)
⭐ **CSF3 `gpuA` is GPU-saturated and it is STRUCTURAL, not a backlog.** `AllocTRES gres/gpu=4` of `CfgTRES
gres/gpu=4` on every `mix` node — the idle CPUs are a mirage — with **8 of 76 GPUs hard-down** (`node860` system
board fault since 07 Aug; `node847` GPU0 VRAM error since 10 Sep) and **an empty pending queue**. Estimated start:
**3 days**, unmoved by shortening `-t` from 4 h to 40 min. ⇒ ⛔ **Nothing we do to a job changes it.**
⛔ **And GCP cannot take the work:** `stage_artefacts.sh` is rsync-over-ssh to CSF3 scratch and nothing else.
**The 1.95 GiB gen artefact set and the 255 MB `3dshapes.h5` have no cloud path.** ⭐ The cloud spoke built
checkpoint/output sync **outbound**; ⛔ nothing carries the **inputs**.
⇒ ⭐⭐ **This is not a route-B workaround. EVERY gen job on GCP needs these artefacts** — the iteration ladder for
target 1 included. It is on the headline's critical path.

## The one job
**Put the gen artefact set in GCS and prove it opens there, reusing the CSF3 logic.**

## ⭐ PORT, DO NOT REDESIGN
⛔⛔ **The hard part is already built and proven.** `stage_artefacts.sh` derives the 44-file set **from the loaders**
(⛔ never a directory listing), checksums it, and writes `stage_manifest.json`; `verify_artefacts.sh` re-checksums
**and OPENS every file through its real production loader**. ⭐ **Both are merged, and the verify demonstrably
catches a byte-perfect-but-unloadable file** — which a checksum gate passes.
⭐ **Change the TRANSPORT and nothing else:** `gcloud storage` in place of `rsync`, the verify running on the VM
instead of a compute node. ⛔ **Do not re-derive the file set.** ⛔ **Do not "improve" the manifest format** — CSF3
and GCP must stay comparable.

## Scope
1. **`scripts/cloud/stage_gcs.sh`** — push the manifest's file set to `gs://<bucket>/artefacts/`, same relative
   layout the loaders expect. Idempotent, resumable, `--dry-run`, refuses an empty **and** a partial set.
2. ⭐⭐ **`3dshapes.h5` IS IN SCOPE HERE and was NOT in the CSF3 set** (255 MB, sha256
   `0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785` — **assert it**). ⚠ On CSF3 it is downloaded
   on a compute node because login nodes have no internet; ⛔ **on GCP decide explicitly**: stage it, or download it
   on the VM. **Say which and why.** ⚠ A VM has internet — ⭐ downloading is likely cheaper than egress.
3. **`scripts/cloud/verify_gcs.sh`** — runs **on the VM**, re-checksums **and opens** every artefact, writes
   `STAGED-GCS.json` with the same schema as `STAGED.json`.
4. ⛔ **The five `CHLU_*` roots are NOT yours** — `gen-cloud-runner` owns `startup.sh` this wave and wires them.
   ⭐ **Publish the exact prefixes it must export** (bucket, prefix, and the five variable names) in your report, so
   the two halves agree. ⛔ Do not edit `startup.sh` or `launch.sh`.

## Files you own — and nothing else
- `scripts/cloud/stage_gcs.sh` **(new)** · `scripts/cloud/verify_gcs.sh` **(new)** · `tests/test_stage_gcs.py` **(new)**
- `.claude/outputs/gen-cloud-artefact-bridge/**`
⛔ Not `scripts/csf3/**` (⭐ **read it, port from it, leave it alone** — it is a live path for a cluster user),
⛔ **not `scripts/cloud/launch.sh`, `startup.sh`, `run_until_done.sh`, `Dockerfile` — `gen-cloud-runner` holds those**,
⛔ not `chlu/**`, ⛔ not `chlu/config.py`.
⭐ **You may run in parallel with `gen-cloud-runner`: you add files, it edits files, and the sets are disjoint.**

## Acceptance
1. `stage_gcs.sh` with a working `--dry-run`; refuses empty and partial sets, naming the missing file **and its loader**.
2. ⭐ **The set actually in GCS**, byte counts matching the manifest. **Report the bucket, the prefix and the cost.**
3. ⭐⭐ **`verify_gcs.sh` run ON A VM, green**: 0 missing, 0 checksum mismatches, **0 failed to load**.
   ⛔ **Not a laptop simulation** — the point is that the loaders work *there*.
4. ⭐ **The verifier demonstrated FAILING**, both ways, as the CSF3 one was: a flipped byte **and** a
   byte-perfect-but-unloadable file. ⛔ A checker that cannot fail proves nothing.
5. `3dshapes.h5` resolved — staged or VM-downloaded, decision stated, **sha256 asserted either way**.
6. Targeted tests green; passed/skipped/failed reported. A **cost line** (storage + egress + VM).

## What would prove this spoke wrong
- ⭐ **You re-derived the file set** and it differs from the CSF3 manifest ⇒ the two platforms are no longer
  comparable and a result cannot be moved between them.
- ⭐ **The verifier checksums but never opens.** A byte-perfect artefact the loader rejects is still a dead run,
  and that is the exact failure the CSF3 verify caught.
- ⭐ **You staged the dataset at egress cost when the VM could have downloaded it free**, or downloaded it without
  asserting the checksum so a truncated fetch looks fine.
- You edited a CSF3 path and broke a live cluster workflow to save yourself a few lines.
