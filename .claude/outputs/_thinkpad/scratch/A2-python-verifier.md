# SPOKE A2 — Install Python and drive `verify_hub_workspace.sh` to a verdict
**You are the critical path. You are the only spoke that installs anything.**

Read `.temp_cl/CONTEXT.md` first.
Work in: `/c/Users/PratikJawahar/chlu/chlu-hub-bundle`
Report to: **`C:\Users\PratikJawahar\OneDrive - Forgis\Desktop\CLU-dev\.temp_cl\reports\A2-python-verifier.md`**

## The goal
Get `bash scripts/verify_hub_workspace.sh` to write a `WORKSPACE-READY.json` with `"status": "ready"` —
**honestly** — or produce the definitive account of why this machine cannot. Both are acceptable
deliverables. A fake green is not.

## Your write surface
System installs · the virtualenv · `WORKSPACE-READY.json` and `.workspace-verify-test.log` (the verifier
writes these itself) · your report. Nothing else in the bundle. In particular do NOT touch
`pyproject.toml` or `uv.lock`; if `uv` wants to rewrite the lock, use `--frozen`/`--locked` and **report
the conflict** instead of letting it rewrite.

## Step 1 — Python 3.12, and kill the Store stub
`pyproject.toml` says `requires-python = ">=3.10, <3.13"`. **Install 3.12.** (3.13 is out of range.)

The Microsoft Store app-execution-alias stub is actively harmful here: it satisfies the verifier's
`command -v python3` guard and then fails at every real call. Deal with it explicitly — either turn the
aliases off (Settings, Apps, Advanced app settings, App execution aliases: off for `python.exe` and
`python3.exe`) or ensure the real interpreter precedes `WindowsApps` on PATH. **Prove the stub is gone:**

    type -a python python3
    python -c "import sys; print(sys.executable, sys.version)"

Note there is normally **no `python3.exe`** on Windows, and the verifier defaults to `python3`. It exposes
a knob: `VERIFY_PYTHON`. Decide whether to (a) set `VERIFY_PYTHON=python`, or (b) put a `python3` shim on
PATH. **Say which and why.** Both are environment changes, which is permitted; editing the script is not.

## Step 2 — dependencies
Install `uv`, then from the bundle root:

    uv sync --extra gen --group test

NEVER `--extra cuda` (Linux-only marker; this box has an Intel Arc GPU). The `eval` and `cafe` extras are
probably not needed for the verifier — add them only if a loader demands it, and say so.

**Venv location:** the bundle is NOT inside OneDrive (it lives at `~/chlu/`), so a normal in-tree `.venv/`
is fine. Confirm the bundle's `.gitignore` covers `.venv/`; if it does not, use
`UV_PROJECT_ENVIRONMENT=/c/clu-venv/hub` instead and say so loudly, because every later session must then
set that variable.

Prove JAX is CPU and healthy — quote the exact output:

    python -c "import jax; print(jax.__version__); print(jax.devices())"

## Step 3 — THE PATH QUESTION. This is the part that matters most.
`env_hub.sh` does `export PYTHONPATH="$CHLU_ROOT${PYTHONPATH:+:$PYTHONPATH}"` where `$CHLU_ROOT` is
`/c/Users/PratikJawahar/chlu/chlu-hub-bundle`. Native Windows Python wants `C:\...` and a **semicolon**
separator. Separately, the verifier passes `VF_ROOT="$CHLU_ROOT"` into Python, which calls
`Path(VF_ROOT).resolve()` — on Windows that turns `/c/Users/...` into the **nonexistent** `C:\c\Users\...`.

**Establish empirically what actually happens.** Do not assume; MSYS may or may not convert it.

    cd /c/Users/PratikJawahar/chlu/chlu-hub-bundle && . ./env_hub.sh
    python -c "import os,sys; from pathlib import Path; print(repr(os.environ['CHLU_ROOT'])); print(repr(os.environ.get('PYTHONPATH'))); print(Path(os.environ['CHLU_ROOT']).resolve()); print(Path(os.environ['CHLU_ROOT']).resolve().is_dir()); print(sys.path[:4])"

Report that output **verbatim** — it is the single most valuable paragraph in your report.

If it is broken, the **sanctioned** fix is an environment one: `env_hub.sh` deliberately honours an
explicit export (`${CHLU_ROOT:-...}`), so you may pre-export a Windows-form `CHLU_ROOT` **before** sourcing
it. Do NOT edit `env_hub.sh`. But note the verifier then re-runs `CHLU_ROOT="$(cd "$CHLU_ROOT" && pwd)"`,
which in Git Bash converts it **back** to POSIX form — check whether your fix actually survives that, and
report the answer either way. If it cannot survive, **say so plainly: that is the finding that decides
whether this machine needs WSL2.**

## Step 4 — run the verifier and read the result properly

    cd /c/Users/PratikJawahar/chlu/chlu-hub-bundle && . ./env_hub.sh
    env | grep ^CHLU_          # every path must sit under the bundle
    bash scripts/verify_hub_workspace.sh

It checks six things. **Report the verdict of each separately**, because one "NOT READY" hides which:

1. sha256 of all 57 manifest files
2. every artefact **opened through its real production loader** (not a re-implemented parse)
3. doc anchors present (a truncated charter checksums fine and is still useless)
4. `chlu.__file__` is **under `$CHLU_ROOT`** — the portability assertion; if this fails the run proves
   nothing about the bundle
5. the targeted test `tests/test_hub_workspace.py`
6. git root tree == the manifest's `pin_tree`

Then `cat WORKSPACE-READY.json` and **paste it whole into your report**. If it still does not exist, say
so and give the full console output instead.

**Expect specific trouble at these points, and report each:**
- `_load_code()` runs `subprocess.run(["bash","-n",...])` **from inside Python** — is `bash` on the
  *Windows* PATH that Python sees? Probably not.
- `_load_code()` asserts `os.access(p, os.X_OK)` ("lost its executable bit in transit"). Windows semantics
  differ; report what it actually does.
- The `eqx_pickle`, `feature_cache`, `npy`, `store_plan`, `winner_hashes`, `c_invalid` loaders import
  `chlu.utils.checkpoints`, `chlu.core.gen_bridge`, `chlu.core.gen_store`, `chlu.core.gen_dream`,
  `chlu.core.gen_certificate` and pull in **jax**. Any one may fail for reasons unrelated to the file.
- The hub write-probe writes `$CHLU_HUB_OUTPUTS/<hub_slug>/.workspace-write-probe.json`.

If something fails, **diagnose and report — do not route around it.** You may use `SKIP_LOAD=1` /
`SKIP_TEST=1` to *isolate* which stage breaks, but the script itself states these are **NOT a pass**;
never report a run that used them as success.

## Step 5 — unblock `gcloud` (cheap, and it is already 90% there)
gcloud is installed at `~/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin` with credentials already in
`~/AppData/Roaming/gcloud/`, and fails only for want of Python. Set `CLOUDSDK_PYTHON` to your new
interpreter and report `gcloud --version`, `gcloud auth list`, `gcloud config list`.
**Read-only gcloud calls only.** Nothing that creates a resource, submits a job, or spends money.

## Step 6 — persistence
State exactly how you made each of `VERIFY_PYTHON`, `CLOUDSDK_PYTHON`, `UV_PROJECT_ENVIRONMENT` (if used)
and any PATH change persist — user-level `setx`, PowerShell profile, `~/.bashrc` — or that they are
session-only. The human needs a copy-pasteable "how to open a working shell tomorrow" block.
**Put that block at the very top of your report.**
