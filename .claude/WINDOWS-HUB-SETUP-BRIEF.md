# BRIEF — bring up a CHLU hub workspace on this Windows machine

**You are a Claude agent on a Windows box. Nobody here has context. This file is all of it.**
**Your job is ENVIRONMENT ONLY. ⛔ You are not doing science and ⛔ you are not fixing the bundle.**

---

## 1. What this machine is for
A research programme (CLU — an addressable memory store) is preparing an ICLR submission. Long-running
**autonomous research agents ("hubs")** are to run here, so that the main machine stays free for other work.
Each hub gets one research question and ~10 hours.

⭐ **Before any of that can happen, this machine has to be able to open the workspace at all.** That is the
only thing being asked of you.

## 2. What you have
A self-contained directory, extracted from `chlu-hub-bundle.tar.gz`, which **IS** the workspace root:
```
<bundle>/                     <- this whole directory is $CHLU_ROOT
  chlu/ scripts/ tests/       <- the Python package and tooling (416 files, fixed commit)
  .claude/                    <- charters, agent profiles, frozen artefacts (~1.95 GB)
  env_hub.sh                  <- sets CHLU_ROOT and every derived CHLU_* path
  BUNDLE_MANIFEST.json        <- every file with its size + sha256
  README-HUB-BUNDLE.md        <- read this too
```
⭐ **The code came in via `git archive`, not a checkout** — so nothing has been rewritten in transit.

## 3. ⭐⭐ YOUR ONE JOB
```bash
. ./env_hub.sh
env | grep ^CHLU_                      # every path MUST be under this directory
bash scripts/verify_hub_workspace.sh
cat WORKSPACE-READY.json
```
**Make that pass by fixing the ENVIRONMENT.** Then send `WORKSPACE-READY.json` back. That is the deliverable.

## 4. ⛔⛔ THE LINE YOU DO NOT CROSS
⭐ **Fix the environment. ⛔ NEVER the bundle.**
- ⛔ **Do not edit, move, regenerate or delete ANY file listed in `BUNDLE_MANIFEST.json`.** Every one is
  sha256-pinned and several are *frozen scientific assets*. Changing one silently invalidates results that
  other machines have already produced.
- ⛔⛔ **Do not weaken, skip, `--force`, or edit `verify_hub_workspace.sh` to make it pass.** ⭐ **A failing
  verifier is INFORMATION, not an obstacle.** It exists to catch exactly the problems you are looking for. If
  you cannot make it pass honestly, **report that** — that is a useful answer and the expected one if
  something real is wrong.
- ⛔ **Do not modify anything under `chlu/`** (the Python package) or `.claude/`. If the code looks wrong,
  it is almost certainly the environment. **Report it, do not patch it.**
- ⭐ **You MAY freely:** install Python, `uv`, packages, `rsync`, coreutils; set env vars; adjust shell
  config; install WSL2; change git config. **That is the whole permitted surface.**

## 5. Environment, in the expected order
1. **Git config first**, before anything git touches:
   `git config --global core.autocrlf false`
   ⚠ Otherwise a later `git` operation rewrites `*.sh` with CRLF and every script dies with
   `bad interpreter: no such file or directory` — an error that tells you nothing about the real cause.
2. **Python 3.11**, and `uv` (`pip install uv`). Then in the bundle root: `uv sync` (or
   `pip install -e .`). ⭐ **CPU only. ⛔ Do NOT install CUDA/GPU JAX** — GPU is Linux-only by declaration
   (`pyproject.toml`: `cuda = [... ; sys_platform == 'linux']`) and no GPU work happens on this machine.
3. `gcloud` CLI, authenticated. Heavy compute runs on Google Cloud, ⛔ never here.
4. Re-run §3.

## 6. ⚠ Three failure modes we predicted — check these before inventing a new theory
1. ⭐ **A stale `CHLU_*` variable already exported in your shell** silently overrides the bundle's derived
   root, because `env_hub.sh` deliberately honours an explicit export. ⭐ **This has already caused one false
   failure on the main machine.** If `env | grep ^CHLU_` shows a path *outside* the bundle: `unset` it and
   re-source. ⛔ Do not "fix" `env_hub.sh`.
2. ⚠ **`rsync` is not in Git Bash** and some scripts call it. ⛔ Not needed for §3, but it *is* needed later
   for moving results. If it is missing, say so.
3. ⚠ **MSYS path mangling** — Git Bash may convert `/c/Users/...` to Windows form when calling native
   binaries, so a POSIX path exported in bash can arrive at Python unusable. ⭐ **This is the one we could
   not predict from the other machine. If you hit it, that is the most valuable thing you can report.**

⭐ **JAX does publish Windows CPU wheels** — native Windows may work. ⚠ **But the whole toolchain is bash
(33 `.sh` scripts, zero `.ps1`/`.bat`).** If Git Bash fights you, **WSL2 is the sanctioned fallback** and you
may install it. ⛔ Neither route is proven; you are the first to try either. **Say which you used.**

## 7. What success looks like
`WORKSPACE-READY.json` exists and reports: every artefact present, checksums matching, **every file opened
through its real loader**, `chlu` importable, one targeted test green. It also records
`platform.system/release/machine` and every `CHLU_*`, so it is self-documenting.

## 8. What to send back — even if it failed
1. ⭐ **`WORKSPACE-READY.json`** (or the error if it was never written).
2. The exact failing command and its **full** output. ⚠ A `tail` throws away the part that matters.
3. Which route you used — **Git Bash or WSL2** — and what you installed.
4. ⭐ **Anything you had to work around**, however small. ⛔ Especially anything you could not fix honestly.

## 9. If you are stuck
⭐ **Stop and report.** ⛔ Do not improvise around a failing check, and ⛔ do not make it pass by lowering it.
⭐ **"This machine cannot run the workspace because X" is a complete and useful answer.** The people on the
other end would far rather have an honest blocker than a green tick that is not true.
