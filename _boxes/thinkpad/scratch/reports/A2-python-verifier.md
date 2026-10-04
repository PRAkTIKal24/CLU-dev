# SPOKE A2 — Install Python and drive `verify_hub_workspace.sh` to a verdict

## ⭐ VERDICT: `WORKSPACE-READY.json` says **`"status": "ready"`**, verifier exit code **0**

Earned honestly: **no `SKIP_LOAD`, no `SKIP_TEST`, no edit to any script, no edit to any manifest file,
`uv.lock` byte-identical before and after.** 57/57 checksums matched, 57/57 artefacts opened through their
real production loaders, `chlu` imported from the bundle, git root tree == pin tree, hub write-probe
written, targeted test `7 passed, 19 skipped`.

⚠ **One caveat you must read before trusting the green tick** — verifier check #3 (doc anchors) is
**silently vacuous on Windows** and checked *zero* anchors. I performed that check by hand instead and all
11 anchors pass. Details in §5.3. **This machine does not need WSL2.**

---

# 1. HOW TO OPEN A WORKING SHELL TOMORROW

## Git Bash — the shell the verifier expects

```bash
export CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-hub-bundle
cd "$CHLU_ROOT"
source ./env_hub.sh
export PATH="$CHLU_ROOT/.venv/Scripts:$PATH"
export PYTHONUTF8=1
export VERIFY_PYTHON="$CHLU_ROOT/.venv/Scripts/python.exe"
export CLOUDSDK_PYTHON="C:/Users/PratikJawahar/AppData/Local/Programs/Python/Python312/python.exe"

# prove it:
bash scripts/verify_hub_workspace.sh      # -> exit 0, status=ready
```

That block is **not illustrative — it is the exact block I ran verbatim as my final clean-room run**, and
it produced `status=ready`, exit 0, with a fresh `verified_at` timestamp.

### Which lines are already persistent and which you must still type

| Line | Persistent? | How |
|---|---|---|
| `PYTHONUTF8=1` | ✅ **persisted** | user-scope env var (registry), set by me |
| `CLOUDSDK_PYTHON` | ✅ **persisted** | user-scope env var (registry), set by me |
| `python` / `python3` / `uv` on PATH | ✅ **already persistent** | user PATH already had `Python312\` and `Python312\Scripts\` *ahead of* `WindowsApps` |
| `CHLU_ROOT` + `source ./env_hub.sh` | ❌ session-only | by design — `env_hub.sh` is the single source of truth |
| `PATH` += `.venv/Scripts` | ❌ session-only | **deliberately not persisted** (see §6) |
| `VERIFY_PYTHON` | ❌ session-only | **deliberately not persisted** (see §6) |

Because `PYTHONUTF8` and `CLOUDSDK_PYTHON` are persistent, the lines you genuinely *must* type in a fresh
shell are the `CHLU_ROOT`/`source` pair and the `.venv/Scripts` PATH prepend. I kept those explicit rather
than persisting them so the bundle stays relocatable.

> ⚠ The two persisted variables live in the **registry**, so they reach **newly launched** shells only.
> The session that spawned this work has a stale environment block and will not see them until restarted.

## PowerShell equivalent (if you ever need it)

```powershell
$env:CHLU_ROOT   = 'C:\Users\PratikJawahar\chlu\chlu-hub-bundle'
$env:PATH        = "$env:CHLU_ROOT\.venv\Scripts;$env:PATH"
$env:PYTHONUTF8  = '1'
# NOTE: env_hub.sh is a bash script; PowerShell cannot source it. Use Git Bash to run the verifier.
```

---

# 2. WHAT I DID

1. Established that this session's `python`/`python3` were **only** the Microsoft Store
   app-execution-alias stub — trap #1, exactly as predicted.
2. **Discovered a real CPython 3.12.10 was already installed** at
   `%LOCALAPPDATA%\Programs\Python\Python312` (dated 14 Sep), merely invisible to this session.
   **`CONTEXT.md`'s "No Python" is inaccurate** — the cause was a stale PATH, not a missing interpreter.
3. Installed `uv` 0.12.15 into that interpreter's `Scripts\` (already on the persistent user PATH).
4. `uv lock --check` → lock already current. `uv sync --extra gen --group test --frozen` → clean, lock
   unchanged.
5. Ran the Step-3 path experiment. **MSYS2 auto-converts POSIX env-var paths for native Windows
   processes** — trap #2 does not bite on this machine.
6. Ran the verifier. Hit 5 failures, all one root cause (cp1252 vs UTF-8). Fixed with `PYTHONUTF8=1`, an
   **environment** change rather than a script edit. Re-ran → `ready`.
7. Unblocked `gcloud` with `CLOUDSDK_PYTHON`.
8. Killed the Store stub permanently by placing a real `python3.exe` beside the base interpreter.
9. Audited that I damaged nothing: `uv.lock` sha unchanged, zero content drift in git.

---

# 3. WHAT WORKS

## 3.1 The interpreter

```
C:\Users\PratikJawahar\chlu\chlu-hub-bundle\.venv\Scripts\python.exe
3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
```

`pyproject.toml` requires `>=3.10, <3.13`; 3.12.10 sits inside that window. **No new Python needed to be
installed** — the pre-existing one was already the right version.

## 3.2 JAX is CPU and healthy — exact output

```
$ python -c "import jax; print(jax.__version__); print(jax.devices())"
0.9.0
[CpuDevice(id=0)]
```

and from the verifier's own import check: `"jax_devices": ["TFRT_CPU_0"]`. No CUDA anywhere; `--extra
cuda` was never used. Companions: `jaxlib 0.9.0`, `numpy 2.4.1`, `equinox 0.13.4`, `h5py 3.16.0`.

## 3.3 Dependencies

```
$ uv lock --check
Resolved 123 packages in 1ms                      # lock was already current
$ uv sync --extra gen --group test --frozen
... exit 0
$ sha256sum uv.lock
3438aa07d5c99b3341da630a8ae0eab0cc6c4606c42c815c475b180e1581cbd2   (identical before and after)
```

- `--frozen` was used as instructed. **uv never attempted to rewrite the lock** — there is no conflict to
  report. `pyproject.toml` untouched (sha `bb278675c3c46f99338f7cfdded36a0bcac35cd3fc2ec205f25e923bc4a4533b`).
- **The `eval` and `cafe` extras were NOT installed and were NOT needed.** All 57 artefacts opened without
  them, including every `gen_*` loader. Do not add them speculatively.
- One harmless pre-existing deprecation warning:
  `warning: The 'tool.uv.dev-dependencies' field (used in 'pyproject.toml') is deprecated`.

## 3.4 Venv location

In-tree `.venv/` — correct per the brief. **Confirmed `.gitignore` line 5 is `.venv/`**, so it is ignored.
`UV_PROJECT_ENVIRONMENT` was **not** needed and is **not** set — no later session has to set it. The
bundle lives at `~/chlu/`, outside OneDrive, so nothing is being sync-mangled.

## 3.5 ⭐ THE PATH QUESTION (Step 3) — verbatim output

This is the paragraph the brief asked for. Command, then unmodified output:

```
$ cd /c/Users/PratikJawahar/chlu/chlu-hub-bundle && . ./env_hub.sh
$ python -c "import os,sys; from pathlib import Path; print(repr(os.environ['CHLU_ROOT'])); print(repr(os.environ.get('PYTHONPATH'))); print(Path(os.environ['CHLU_ROOT']).resolve()); print(Path(os.environ['CHLU_ROOT']).resolve().is_dir()); print(sys.path[:4])"

'C:/Users/PratikJawahar/chlu/chlu-hub-bundle'
'C:/Users/PratikJawahar/chlu/chlu-hub-bundle'
C:\Users\PratikJawahar\chlu\chlu-hub-bundle
True
['', 'C:\\Users\\PratikJawahar\\chlu\\chlu-hub-bundle', 'C:\\Users\\PratikJawahar\\AppData\\Local\\Programs\\Python\\Python312\\python312.zip', 'C:\\Users\\PratikJawahar\\AppData\\Local\\Programs\\Python\\Python312\\DLLs']
```

**Reading.** Bash holds `CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-hub-bundle`, but native Windows Python
receives `'C:/Users/PratikJawahar/chlu/chlu-hub-bundle'`. **The MSYS2 runtime rewrites POSIX paths in
environment variables when it spawns a native Windows process.** Therefore:

- `Path(VF_ROOT).resolve()` → `C:\Users\PratikJawahar\chlu\chlu-hub-bundle`, and **`.is_dir()` is `True`**.
- The feared `C:\c\Users\...` **never materialises.**
- The colon separator is handled too. With a genuinely multi-entry `PYTHONPATH`:
  ```
  $ PYTHONPATH="/c/Users/PratikJawahar/chlu/chlu-hub-bundle:/tmp/second" python -c "import os;print(repr(os.environ['PYTHONPATH']))"
  'C:\\Users\\PratikJawahar\\chlu\\chlu-hub-bundle;C:\\Users\\PRATIK~1\\AppData\\Local\\Temp\\second'
  ```
  Colon → **semicolon**, both entries → Windows form, and both land on `sys.path`.

**Conclusion: trap #2 is a non-issue on this machine, and no `CHLU_ROOT` pre-export workaround is needed.**
I never had to pre-export a Windows-form `CHLU_ROOT`, so whether such a fix would survive the verifier's
`CHLU_ROOT="$(cd "$CHLU_ROOT" && pwd)"` re-normalisation never became load-bearing. For the record it
would **not** survive — Git Bash's `pwd` always emits POSIX form — but because the POSIX form *works*,
that round-trip is harmless rather than fatal. **This machine does not need WSL2.**

## 3.6 The `python3` decision — I chose (b), a shim, and here is why

The brief offered (a) `VERIFY_PYTHON=python` or (b) a `python3` shim on PATH. **I did (b)**, in two places:

```bash
cp "$PYBASE/python.exe"        "$PYBASE/python3.exe"          # base interpreter, machine-wide fix
cp ".venv/Scripts/python.exe"  ".venv/Scripts/python3.exe"    # the venv (uv does not create one)
```

**Why (b) over (a):** `VERIFY_PYTHON=python` fixes exactly one script and leaves the stub live for
everything else — the `Makefile`, any hub-spawned tooling, any future script that says `python3`. The
stub's real danger is that it *satisfies* `command -v python3` and then dies at the first real call, so
every such caller inherits the same booby trap. A real `python3.exe` earlier on PATH than `WindowsApps`
defeats the stub **for every caller at once** and makes `command -v python3` tell the truth. It remains
purely an environment change; no script was edited.

**Proof the stub is defeated**, in the working shell:

```
$ type -a python python3
python is /c/Users/PratikJawahar/chlu/chlu-hub-bundle/.venv/Scripts/python
python is /c/Users/PratikJawahar/AppData/Local/Microsoft/WindowsApps/python
python3 is /c/Users/PratikJawahar/chlu/chlu-hub-bundle/.venv/Scripts/python3
python3 is /c/Users/PratikJawahar/AppData/Local/Microsoft/WindowsApps/python3

$ python3 -c "import sys,jax; print(sys.executable); print(jax.devices())"
C:\Users\PratikJawahar\chlu\chlu-hub-bundle\.venv\Scripts\python3.exe
[CpuDevice(id=0)]
```

The stub is still *present* but now permanently *shadowed*. And in a **brand-new shell** with no bundle
activation at all, resolving against the persistent Machine+User PATH:

```
python.exe     -> C:\Users\PratikJawahar\AppData\Local\Programs\Python\Python312\python.exe
python3.exe    -> C:\Users\PratikJawahar\AppData\Local\Programs\Python\Python312\python3.exe
uv.exe         -> C:\Users\PratikJawahar\AppData\Local\Programs\Python\Python312\Scripts\uv.exe
gcloud.cmd     -> C:\Users\PratikJawahar\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd
```

`WindowsApps` appears **only** in the User PATH and **only after** `Python312`, and not at all in the
Machine PATH — so the shadowing is structural, not luck. I did **not** disable the app-execution aliases
in Settings; that is now optional belt-and-braces rather than necessary.

## 3.7 The three traps the brief told me to expect at `_load_code()`

| Trap | Predicted | Actual |
|---|---|---|
| `subprocess.run(["bash","-n",...])` — is `bash` on the *Windows* PATH Python sees? | "probably not" | ✅ **Yes.** `shutil.which('bash')` → `C:\Program Files\Git\usr\bin\bash.EXE`; `bash -n scripts/verify_hub_workspace.sh` → rc 0. MSYS translates PATH on spawn, so Git's own bash is reachable. |
| `os.access(p, os.X_OK)` — "lost its executable bit in transit" | semantics differ | ⚠ **Returns `True`, but vacuously.** `os.stat` mode is `0o100666` — *no execute bits at all* — yet `os.access(X_OK)` is `True`, because Windows reports X_OK for any existing readable file. **The assertion cannot fail on Windows.** Harmless here, but it is not evidence. |
| gen/jax loaders (`eqx_pickle`, `feature_cache`, `npy`, `store_plan`, `winner_hashes`, `c_invalid`) | any may fail | ✅ **All opened.** No failure attributable to jax, equinox, or the gen modules. |
| hub write-probe | — | ✅ Written to `...\.claude\outputs\hub-composition\.workspace-write-probe.json`, `writable=true`. |

---

# 4. THE SIX CHECKS — VERDICT OF EACH, SEPARATELY

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | sha256 of all 57 manifest files | ✅ **PASS** | `files=57 missing=0 mismatched=0` |
| 2 | every artefact opened through its real production loader | ✅ **PASS** | `unopenable=0`, `loaders_all_opened: true` |
| 3 | doc anchors present | ⚠ **VACUOUS — 0 anchors checked.** Verified by hand instead: **all 11 pass** | **see §5.3** |
| 4 | `chlu.__file__` under `$CHLU_ROOT` | ✅ **PASS** | `chlu=C:\Users\PratikJawahar\chlu\chlu-hub-bundle\chlu\__init__.py under_root=True` |
| 5 | targeted test `tests/test_hub_workspace.py` | ✅ **PASS** | `7 passed, 19 skipped in 10.79s` |
| 6 | git root tree == manifest `pin_tree` | ✅ **PASS** | `root_tree=12831fe1... matches_pin=True` |

### On check 5's 19 skips — by design, not Windows damage

All 19 carry the identical reason, and it is the script's own documented recursion guard:

```
SKIPPED [1] tests\test_hub_workspace.py:293: nested under verify_hub_workspace.sh (recursion guard)
... (19 occurrences, lines 293 through 661)
```

`verify_hub_workspace.sh` exports `CHLU_WORKSPACE_VERIFY_DEPTH=1` to its pytest child precisely so the
round-trip bundle-building tests do not fork — the incident the script's comments describe, where 28
processes had to be killed by hand. Run standalone at depth 0 those 19 would execute, but they build
bundles and the brief forbids heavy compute, so I did not force them. **Nothing skipped for a Windows
reason.**

---

# 5. WHAT WAS BROKEN — full output

## 5.1 The original failure (pre-existing) — trap #1, confirmed and resolved

`.workspace-verify-test.log` as I found it, in full:

```
Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Apps > Advanced app settings > App execution aliases.
```

Root cause as `CONTEXT.md` predicted, **except** for one correction: a perfectly good CPython 3.12.10 was
already on disk *and already on the persistent user PATH*. The session that ran the verifier held a stale
environment block predating the 14 Sep Python install. Resolved as in §3.6.

## 5.2 The real Windows defect: cp1252 vs UTF-8 — 5 failures, one cause

First honest run: `files=57 missing=0 mismatched=0 unopenable=3`, test `2 failed, 5 passed, 19 skipped`,
`status=NOT-READY`. **All five failures were the same bug.** Full output:

```
[verify] FAIL scripts/bundle_hub_workspace.sh   UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 120: character maps to <undefined>
[verify] FAIL scripts/verify_hub_workspace.sh   UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 121: character maps to <undefined>
[verify] FAIL tests/test_hub_workspace.py       UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 1125: character maps to <undefined>
```
```
___________________ test_verify_needs_no_gpu_and_no_cluster ___________________
    def test_verify_needs_no_gpu_and_no_cluster():
        """⛔ This is plumbing. No GPU, no sbatch, no ssh, no transfer."""
>       text = VERIFY_SH.read_text()
               ^^^^^^^^^^^^^^^^^^^^^
tests\test_hub_workspace.py:254:
..\..\AppData\Local\Programs\Python\Python312\Lib\pathlib.py:1028: in read_text
    return f.read()
self = <encodings.cp1252.IncrementalDecoder object at 0x000002D457C0D760>
input = b'#!/usr/bin/env bash\n# scripts/verify_hub_workspace.sh \xe2\x80\x94 prove a hub workspace bundle is USABLE, ON THE F... say "=== \xe2\x9b\x94 NOT READY \xe2\x80\x94 see $READY_OUT. \xe2\x9b\x94 Do not launch a hub. ==="\nfi\nexit "$rc"\n'
    def decode(self, input, final=False):
>       return codecs.charmap_decode(input,self.errors,decoding_table)[0]
E       UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 121: character maps to <undefined>
..\..\AppData\Local\Programs\Python\Python312\Lib\encodings\cp1252.py:23: UnicodeDecodeError

__________ test_the_legacy_absolute_path_survives_only_as_a_default ___________
>       hits = [ln for ln in BUNDLE_SH.read_text().splitlines()
                             ^^^^^^^^^^^^^^^^^^^^^
tests\test_hub_workspace.py:244:
E       UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 120: character maps to <undefined>

FAILED tests/test_hub_workspace.py::test_the_legacy_absolute_path_survives_only_as_a_default
FAILED tests/test_hub_workspace.py::test_verify_needs_no_gpu_and_no_cluster
[verify] targeted test: FAILED (rc=1): 2 failed, 5 passed, 19 skipped in 11.50s
[verify] === ⛔ NOT READY — see .../WORKSPACE-READY.json. ⛔ Do not launch a hub. ===
```

**Diagnosis.** Not corruption — those three files' **sha256 matched the manifest exactly**.
`Path.read_text()` with no `encoding=` uses the locale encoding, which on this machine is **cp1252**. The
bundle's shell scripts and tests are UTF-8 and full of `⛔ ⭐ —`. `\xe2\x9b\x94` is `⛔`; `0x90` is an
unmapped cp1252 byte. `_load_code()` and both failing tests all call bare `read_text()`.

**Fix — environment, not source.** PEP 540 UTF-8 Mode:

```bash
export PYTHONUTF8=1
```

This makes `read_text()` default to UTF-8 process-wide. **No file was edited.** Re-run → `unopenable=0`,
`7 passed, 19 skipped`, `status=ready`, exit 0. I persisted `PYTHONUTF8=1` at user scope because *any*
Python touching this UTF-8-heavy repo on this machine hits the same wall.

> This is a genuine cross-platform bug in the bundle, not a local misconfiguration. On macOS/Linux the
> locale encoding is already UTF-8, so it never fires there. **Reported, not patched**, per the rules.

## 5.3 ⚠ THE ONE CHECK THE VERIFIER DID NOT ACTUALLY PERFORM

Even in the green run, every doc reports **`"anchors_checked": 0`** — including the charter, which
`DOC_ANCHORS` says must carry 4. Cause, proven directly:

```
$ python3 -c "... rel = str(Path(p).resolve().relative_to(root)) ..."
rel as computed by _load_doc : '.claude\\autonomous-hub-charter.md'
DOC_ANCHORS key expected     : '.claude/autonomous-hub-charter.md'
dict lookup hits?            : False
```

`_load_doc()` builds `rel` with `Path.relative_to()`, which on Windows yields **backslash** separators,
while `DOC_ANCHORS` is keyed with **forward slashes**. `DOC_ANCHORS.get(rel, [])` therefore **always
misses**, `missing` is always `[]`, and the function returns `"loaded": true` having inspected nothing.

**On Windows, check #3 silently degrades to "the file is non-empty".** A truncated charter — the precise
failure the check exists to catch, per its own comment *"a truncated charter checksums fine and is
useless"* — **would sail straight through.** It is a one-call fix upstream (`.as_posix()`), Windows-only,
and it affects the 4 docs that have anchors: `AGENT_PROTOCOL.md`, `autonomous-hub-charter.md`,
`agents/autonomous-hub.md`, `handover_context.md`. (The three `.claude/tasks/*.md` docs legitimately have
no anchors registered, so `0` is correct for those.)

**I could not fix this without editing the script, which is forbidden, and no environment variable can
change `relative_to()`'s separator. So I performed the check by hand:**

```
PASS .claude/AGENT_PROTOCOL.md          (2 anchors)
PASS .claude/autonomous-hub-charter.md  (4 anchors)  <- incl. pin ae907466..., "MATCHED-BYTES LOOKUP", "core philosophy"
PASS .claude/agents/autonomous-hub.md   (4 anchors)
PASS .claude/handover_context.md        (1 anchor)
```

**All 11 anchors are present; the docs are genuinely intact** — but that assurance comes from me, not from
the verifier. This is open question #1.

## 5.4 Git noise — non-fatal, and not mine

```
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
```

One of the 588 macOS AppleDouble `._*` files (trap #4) landed *inside* `.git/objects/pack/`, and git tries
to parse it as a pack index. It is **noise, not damage**: `git rev-parse HEAD` → `5c8bfece...`,
`git rev-list --max-parents=0 HEAD` → `172fa35d...`, and the verifier's check #6 resolved the root tree
correctly and matched the pin. All 588 `._*` files were still present when I finished — **that cleanup is
another spoke's brief and I deliberately did not touch it.**

Also, `git status` reports **479 modified files**. This is *not* content drift:

```
$ git diff --stat | tail -1
 36 files changed, 0 insertions(+), 0 deletions(-)
$ git diff --summary | head -1
 mode change 100755 => 100644 scripts/bundle_hub_workspace.sh
$ git config core.filemode
true
```

Every entry is `mode change 100755 => 100644` — the POSIX executable bit, which NTFS does not carry, with
`core.filemode=true`. **Zero bytes changed**, and the 57/57 sha256 match is the independent proof.
Pre-existing from extraction; not caused by me. It does mean a hub that inspects `os.access(X_OK)` on
Windows is, per §3.7, asking a question Windows cannot answer.

---

# 6. PERSISTENCE — exactly what I changed, and where

| Setting | Value | Mechanism | Scope |
|---|---|---|---|
| `PYTHONUTF8` | `1` | `[Environment]::SetEnvironmentVariable(...,'User')` → registry | **persistent, user-level** |
| `CLOUDSDK_PYTHON` | `C:\Users\PratikJawahar\AppData\Local\Programs\Python\Python312\python.exe` | same | **persistent, user-level** |
| `python3.exe` (base) | copy of `Python312\python.exe` | file copy | **persistent, on-disk** |
| `python3.exe` (venv) | copy of `.venv\Scripts\python.exe` | file copy | persistent until the venv is rebuilt |
| `uv` 0.12.15 | `Python312\Scripts\uv.exe` | `python -m pip install uv` | **persistent, already on user PATH** |
| PATH | **unchanged** | — | the user PATH already had Python312 ahead of WindowsApps |
| `VERIFY_PYTHON` | not set | — | **session-only**, in the §1 block |
| `UV_PROJECT_ENVIRONMENT` | **not used** | — | in-tree `.venv/` is gitignored, so it was unnecessary |

**Verified by reading back from the registry:**
```
CLOUDSDK_PYTHON = C:\Users\PratikJawahar\AppData\Local\Programs\Python\Python312\python.exe
PYTHONUTF8      = 1
```

**Why `VERIFY_PYTHON` and the `.venv` PATH entry are deliberately NOT persisted:** both hard-code the
bundle's current location. Persisting them would make a moved or rebuilt bundle fail confusingly, and
would silently point *other* checkouts at *this* bundle's venv — the very class of error check #4 exists
to catch. They belong in the per-session block, where they are visible.

> ⚠ **Footgun.** In a fresh shell that skips the PATH line, `python3` is the **base** 3.12.10, which has
> **no jax**. The verifier would then fail loudly at the import check — noisy rather than silent, so it is
> survivable, but the `VERIFY_PYTHON` line in the §1 block exists to make it impossible.

**Reversing `PYTHONUTF8=1`**, should it ever disturb another project on this box:
`[Environment]::SetEnvironmentVariable('PYTHONUTF8',$null,'User')`. I judged it safe — UTF-8 mode is the
direction CPython is moving — but it is a machine-wide behaviour change and you should know it is there.
Note there is also a PyTorch install in that base interpreter (`torchrun.exe`, `numpy`, `sympy` present in
`Python312\Scripts`), untouched by any of this; the bundle runs entirely out of its own `.venv`.

---

# 7. gcloud — unblocked (Step 5)

`CLOUDSDK_PYTHON` was the only thing missing. **Read-only calls only; nothing created, submitted, or
billed.**

```
$ gcloud --version
Google Cloud SDK 584.0.0
bq 2.1.38
core 2026.09.04
gcloud-crc32c 1.0.0
gsutil 5.37
Updates are available for some Google Cloud CLI components.  To install them,
please run:
  $ gcloud components update

$ gcloud auth list
      Credentialed Accounts
ACTIVE  ACCOUNT
*       pratik.jawahar@forgis.com

To set the active account, run:
    $ gcloud config set account `ACCOUNT`

$ gcloud config list
[accessibility]
screen_reader = False
[core]
account = pratik.jawahar@forgis.com
disable_usage_reporting = True
project = forgisprova

Your active configuration is: [default]
```

Account authenticated, project `forgisprova` — the same project as the handoff bucket
`gs://chlu-cloud-forgisprova/`. I did **not** run `gcloud components update` (it writes to the SDK), and
did not list or touch the bucket (outside my brief).

---

# 8. `WORKSPACE-READY.json`

Live at `C:\Users\PratikJawahar\chlu\chlu-hub-bundle\WORKSPACE-READY.json`. The complete top-level object
is reproduced verbatim below. The `"files"` array (57 entries × ~18 lines of JSON) is rendered as a table
in §8.2 rather than raw JSON — it carries the same information in readable form, and the raw file is on
disk if you need it byte-for-byte.

## 8.1 Verbatim, with the `files` array elided

```json
{
  "artefact_axes": {
    "datasets": [ "shapes3d" ],
    "pretexts": [ "recon" ],
    "seeds": [ 0, 1, 2 ]
  },
  "checksum_match": true,
  "chlu_root": "C:\\Users\\PratikJawahar\\chlu\\chlu-hub-bundle",
  "code_is_the_pin": true,
  "env": {
    "CHLU_CLAUDE_ROOT": "C:/Users/PratikJawahar/chlu/chlu-hub-bundle/.claude",
    "CHLU_GEN_BRIDGE_ROOT": "C:/Users/PratikJawahar/chlu/chlu-hub-bundle/.claude/outputs/gen-feature-bridge/runs",
    "CHLU_GEN_DATA_ROOT": "C:/Users/PratikJawahar/chlu/chlu-hub-bundle/.claude/data/gen",
    "CHLU_GEN_DECLIFT_DIR": "C:/Users/PratikJawahar/chlu/chlu-hub-bundle/.claude/outputs/gen-decoder-lift/winner",
    "CHLU_GEN_STORE_RUNS": "C:/Users/PratikJawahar/chlu/chlu-hub-bundle/.claude/outputs/gen-store-grouping/runs",
    "CHLU_GEN_WIDE_BRIDGE_ROOT": "C:/Users/PratikJawahar/chlu/chlu-hub-bundle/.claude/outputs/gen-bridge-wide/runs",
    "CHLU_HUB_OUTPUTS": "C:/Users/PratikJawahar/chlu/chlu-hub-bundle/.claude/outputs",
    "CHLU_ROOT": "C:/Users/PratikJawahar/chlu/chlu-hub-bundle"
  },
  "files": [ ... 57 entries, all "checksum_match": true and "loaded": true — see §8.2 ... ],
  "git_check": {
    "branch": "main",
    "error": null,
    "head": "5c8bfec",
    "matches_pin": true,
    "pin_tree": "12831fe1be17d04da027dcbb6e45b4dff6f1ef95",
    "repo": true,
    "root_commit": "172fa35d8c2b5480f4b00ba3ac814532875b001b",
    "root_tree": "12831fe1be17d04da027dcbb6e45b4dff6f1ef95",
    "worktree_possible": true
  },
  "host": "FORGIS_14",
  "hub_output_path_writable": true,
  "hubs": [ "hub-composition", "hub-refusal", "hub-deletion" ],
  "import_check": {
    "chlu_file": "C:\\Users\\PratikJawahar\\chlu\\chlu-hub-bundle\\chlu\\__init__.py",
    "chlu_version": "0.2.4",
    "error": null,
    "jax": "0.9.0",
    "jax_devices": [ "TFRT_CPU_0" ],
    "ok": true,
    "sys_path_0": "C:\\Users\\PratikJawahar\\chlu\\chlu-hub-bundle",
    "under_root": true
  },
  "import_under_root": true,
  "loaders_all_opened": true,
  "manifest": "C:/Users/PratikJawahar/chlu/chlu-hub-bundle/BUNDLE_MANIFEST.json",
  "manifest_generated_at": "2026-09-16T14:33:46.645892+00:00",
  "n_checksum_mismatch": 0,
  "n_files": 57,
  "n_load_failed": 0,
  "n_missing": 0,
  "pin": "ae907466ba2b5fbec124e8995cf4d7c32a1ab30e",
  "platform": {
    "machine": "AMD64",
    "python": "3.12.10",
    "release": "11",
    "system": "Windows"
  },
  "schema": "chlu-hub-workspace-ready/1",
  "skip_load": false,
  "status": "ready",
  "targeted_test": "passed: 7 passed, 19 skipped in 10.91s",
  "total_bytes_on_disk": 2102084732,
  "verified_at": "2026-09-16T15:52:34.783372+00:00",
  "worktree_possible": true,
  "write_check": {
    "error": null,
    "hub_slug": "hub-composition",
    "path": "C:\\Users\\PratikJawahar\\chlu\\chlu-hub-bundle\\.claude\\outputs\\hub-composition",
    "probe": "C:\\Users\\PratikJawahar\\chlu\\chlu-hub-bundle\\.claude\\outputs\\hub-composition\\.workspace-write-probe.json",
    "writable": true
  }
}
```

Key lines: **`"status": "ready"`**, `"skip_load": false`, `"n_missing": 0`, `"n_checksum_mismatch": 0`,
`"n_load_failed": 0`, `"import_under_root": true`, `"code_is_the_pin": true`,
`"hub_output_path_writable": true`, `"pin": "ae907466ba2b5fbec124e8995cf4d7c32a1ab30e"`, 2.10 GB verified.

## 8.2 All 57 files — checksum and loader verdict

*(`anchors_checked: 0` on rows 1, 2, 3 and 8 is the §5.3 defect, not a property of the files.)*

| # | path | open_with | sha | loaded | detail |
|---|------|-----------|-----|--------|--------|
| 1 | `.claude/AGENT_PROTOCOL.md` | doc | ok | yes | `{"anchors_checked":0,"chars":9713,"lines":72}` |
| 2 | `.claude/agents/autonomous-hub.md` | doc | ok | yes | `{"anchors_checked":0,"chars":10433,"lines":158}` |
| 3 | `.claude/autonomous-hub-charter.md` | doc | ok | yes | `{"anchors_checked":0,"chars":6532,"lines":81}` |
| 4 | `.claude/data/gen/shapes3d/a7_redundant.json` | control_set | ok | yes | `{"n_records":72,"name":"a7_redundant"}` |
| 5 | `.claude/data/gen/shapes3d/c_invalid.json` | c_invalid | ok | yes | `{"factors":["floor_hue","wall_hue","object_hue","scale","shape","orie...` |
| 6 | `.claude/data/gen/shapes3d/factor_index.npz` | npz_index | ok | yes | `{"factors":["floor_hue","wall_hue","object_hue","scale","shape","orie...` |
| 7 | `.claude/data/gen/shapes3d/query_sets.json` | query_registry | ok | yes | `{"dataset":"shapes3d","sets":{"a7_redundant":72,"base":2048,"c_invali...` |
| 8 | `.claude/handover_context.md` | doc | ok | yes | `{"anchors_checked":0,"chars":1001293,"lines":3845}` |
| 9 | `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` | json | ok | yes | `{"keys":9}` |
| 10 | `.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/seed_0/baselines_manifest.json` | json | ok | yes | `{"keys":14}` |
| 11 | `.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/seed_1/baselines_manifest.json` | json | ok | yes | `{"keys":14}` |
| 12 | `.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/seed_2/baselines_manifest.json` | json | ok | yes | `{"keys":14}` |
| 13 | `.claude/outputs/gen-decoder-lift/winner/decoder_v1_s0.pkl` | eqx_pickle | ok | yes | `{"n_leaves":40}` |
| 14 | `.claude/outputs/gen-decoder-lift/winner/decoder_v1_s1.pkl` | eqx_pickle | ok | yes | `{"n_leaves":40}` |
| 15 | `.claude/outputs/gen-decoder-lift/winner/decoder_v1_s2.pkl` | eqx_pickle | ok | yes | `{"n_leaves":40}` |
| 16 | `.claude/outputs/gen-decoder-lift/winner/winner_hashes.json` | winner_hashes | ok | yes | `{"seeds":["seed_0","seed_1","seed_2"]}` |
| 17 | `.claude/outputs/gen-external-classifier/classifier_gate.json` | json | ok | yes | `{"keys":15}` |
| 18 | `.claude/outputs/gen-external-classifier/seed_0/classifier_manifest.json` | json | ok | yes | `{"keys":12}` |
| 19 | `.claude/outputs/gen-external-classifier/seed_0/external_classifier.pkl` | eqx_pickle | ok | yes | `{"n_leaves":30}` |
| 20 | `.claude/outputs/gen-external-classifier/seed_1/classifier_manifest.json` | json | ok | yes | `{"keys":12}` |
| 21 | `.claude/outputs/gen-external-classifier/seed_1/external_classifier.pkl` | eqx_pickle | ok | yes | `{"n_leaves":30}` |
| 22 | `.claude/outputs/gen-external-classifier/seed_2/classifier_manifest.json` | json | ok | yes | `{"keys":12}` |
| 23 | `.claude/outputs/gen-external-classifier/seed_2/external_classifier.pkl` | eqx_pickle | ok | yes | `{"n_leaves":30}` |
| 24 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/bridge_manifest.json` | json | ok | yes | `{"keys":17}` |
| 25 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/decoder.pkl` | eqx_pickle | ok | yes | `{"n_leaves":24}` |
| 26 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/encoder.pkl` | eqx_pickle | ok | yes | `{"n_leaves":22}` |
| 27 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/feature_cache.json` | feature_cache | ok | yes | `{"dtype":"float16","feat_shape":[128,4,4],"n_rows":480000,"scope":"fu...` |
| 28 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/feature_rows.npy` | npy | ok | yes | `{"dtype":"int64","shape":[480000]}` |
| 29 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/features.npy` | npy | ok | yes | `{"dtype":"float16","shape":[480000,128,4,4]}` |
| 30 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/probe.pkl` | eqx_pickle | ok | yes | `{"n_leaves":24}` |
| 31 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/bridge_manifest.json` | json | ok | yes | `{"keys":17}` |
| 32 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/decoder.pkl` | eqx_pickle | ok | yes | `{"n_leaves":24}` |
| 33 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/encoder.pkl` | eqx_pickle | ok | yes | `{"n_leaves":22}` |
| 34 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/feature_cache.json` | feature_cache | ok | yes | `{"dtype":"float16","feat_shape":[128,4,4],"n_rows":9288,"scope":"used"}` |
| 35 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/feature_rows.npy` | npy | ok | yes | `{"dtype":"int64","shape":[9288]}` |
| 36 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/features.npy` | npy | ok | yes | `{"dtype":"float16","shape":[9288,128,4,4]}` |
| 37 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/probe.pkl` | eqx_pickle | ok | yes | `{"n_leaves":24}` |
| 38 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/bridge_manifest.json` | json | ok | yes | `{"keys":17}` |
| 39 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/decoder.pkl` | eqx_pickle | ok | yes | `{"n_leaves":24}` |
| 40 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/encoder.pkl` | eqx_pickle | ok | yes | `{"n_leaves":22}` |
| 41 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/feature_cache.json` | feature_cache | ok | yes | `{"dtype":"float16","feat_shape":[128,4,4],"n_rows":9288,"scope":"used"}` |
| 42 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/feature_rows.npy` | npy | ok | yes | `{"dtype":"int64","shape":[9288]}` |
| 43 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/features.npy` | npy | ok | yes | `{"dtype":"float16","shape":[9288,128,4,4]}` |
| 44 | `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/probe.pkl` | eqx_pickle | ok | yes | `{"n_leaves":24}` |
| 45 | `.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_0/store_s0.pkl` | eqx_pickle | ok | yes | `{"n_leaves":5}` |
| 46 | `.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_1/store_s1.pkl` | eqx_pickle | ok | yes | `{"n_leaves":5}` |
| 47 | `.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_2/store_s2.pkl` | eqx_pickle | ok | yes | `{"n_leaves":5}` |
| 48 | `.claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_0/store_manifest.json` | json | ok | yes | `{"keys":20}` |
| 49 | `.claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_1/store_manifest.json` | json | ok | yes | `{"keys":23}` |
| 50 | `.claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_2/store_manifest.json` | json | ok | yes | `{"keys":23}` |
| 51 | `.claude/outputs/gen-store-grouping/runs/shapes3d_recon/store_plan.json` | store_plan | ok | yes | `{"D":2048,"K":5,"n_modes":64}` |
| 52 | `.claude/tasks/hub-composition.md` | doc | ok | yes | `{"anchors_checked":0,"chars":4658,"lines":66}` |
| 53 | `.claude/tasks/hub-deletion.md` | doc | ok | yes | `{"anchors_checked":0,"chars":4414,"lines":59}` |
| 54 | `.claude/tasks/hub-refusal.md` | doc | ok | yes | `{"anchors_checked":0,"chars":4723,"lines":65}` |
| 55 | `scripts/bundle_hub_workspace.sh` | code | ok | yes | `{"lines":943,"parsed":"bash -n"}` |
| 56 | `scripts/verify_hub_workspace.sh` | code | ok | yes | `{"lines":485,"parsed":"bash -n"}` |
| 57 | `tests/test_hub_workspace.py` | code | ok | yes | `{"lines":720,"parsed":"python"}` |

---

# 9. WHAT I COULD NOT RESOLVE HONESTLY

1. **The doc-anchor check (#3) cannot be made to work on Windows from the environment.** It needs a
   one-call source change (`.as_posix()`), which I am forbidden to make. I substituted a manual check and
   it passes, but the verifier's own green tick on those four docs is not evidence. §5.3.
2. **The `os.access(X_OK)` assertion in `_load_code()` is unfalsifiable on Windows.** It returns `True` for
   files whose mode is `0o100666`. I cannot make it meaningful; I can only tell you it proved nothing. §3.7.
3. **The 19 recursion-guarded tests were not exercised at depth 0.** They build bundles and run the
   verifier recursively; running them is real compute and risks the 28-process fork incident the script
   documents. So the round-trip bundle/rebuild path is **untested on Windows** — the *bundle* is verified,
   the *bundling machinery* is not.
4. **The 588 AppleDouble `._*` files are untouched**, including the one inside `.git/objects/pack/` that
   makes every `git` call emit four error lines. It is another spoke's brief and I stayed out. Git works
   correctly despite it, but it will be noise in every hub log until cleaned.
5. **I did not verify the full test suite** — only `tests/test_hub_workspace.py`, as the brief specified.
   The suite is reportedly green at 354 on macOS; whether it is green on Windows is **unknown**, and given
   that two of the seven targeted tests failed on a Windows-specific encoding issue, I would not assume it.

---

# 10. NUMBERED OPEN QUESTIONS FOR THE HUMAN

1. **Do you want the two Windows portability bugs fixed upstream, at the pin?** Both are one-liners in
   `scripts/verify_hub_workspace.sh`: (a) `_load_doc` should key `DOC_ANCHORS` off
   `Path(p).resolve().relative_to(root).as_posix()`; (b) `_load_code` and the two tests should pass
   `encoding="utf-8"` to `read_text()`. Fixing them changes files that are **sha256-pinned in
   `BUNDLE_MANIFEST.json`** and would move the pin tree, so this is your call, not mine. Until (a) is
   fixed, **every Windows verify run silently skips the truncation check on the charter.**
2. **Is `PYTHONUTF8=1` acceptable as a permanent machine-wide setting?** It is currently required for the
   verifier to pass. If you would rather not have it global, the alternative is putting it in the §1 block
   only — but then anyone who forgets it gets a confusing `UnicodeDecodeError` on a byte-perfect file.
3. **Should the full test suite be run on Windows before a hub is launched?** My brief scoped me to the
   targeted test. Given that the encoding bug broke 2 of 7 targeted tests, I would expect more cp1252
   casualties across 354 — cheap to find out now, expensive to find out at hour 6. Say the word and it is
   one command.
4. **Is `tests/test_hub_workspace.py`'s round-trip bundle-rebuild path expected to work on Windows at
   all?** Those are the 19 guarded tests. `bundle_hub_workspace.sh` is 943 lines of bash that tars,
   checksums and re-verifies; whether it survives NTFS (no exec bit, no symlinks by default) is untested
   and is the obvious next unknown.
5. **Do you want the Microsoft Store app-execution aliases actually turned off** (Settings → Apps →
   Advanced app settings → App execution aliases), or is shadowing them via PATH sufficient? I chose
   shadowing because it is reversible and needs no GUI, but the stubs are still physically present and
   will resurface for anyone who puts `WindowsApps` early on PATH.
6. **Confirm the `._*` cleanup is owned by another spoke.** If it is not, the one inside
   `.git/objects/pack/` should be removed before a hub runs — not for correctness, but because four
   `error:` lines on every git call will pollute ten hours of autonomous logs and may be misread by the
   hub itself as a repository fault.

---

*Artefacts I left behind, all outside the bundle's protected surface: `WORKSPACE-READY.json` and
`.workspace-verify-test.log` (written by the verifier itself), `coverage.xml` / `.coverage` (pytest, both
gitignored), `.venv/` (gitignored), the write-probe at
`.claude/outputs/hub-composition/.workspace-write-probe.json` (the verifier's, left in place per §0.1 —
never delete). Raw console logs of all three verifier runs are alongside this report as
`_A2-verify-run1.log` (the cp1252 failure), `_A2-verify-run2.log` (first green) and `_A2-verify-run3.log`
(clean-room reproduction using the §1 block verbatim).*
