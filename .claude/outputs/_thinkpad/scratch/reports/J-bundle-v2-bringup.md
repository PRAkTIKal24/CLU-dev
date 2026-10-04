# SPOKE J — bundle v2 bring-up (native Windows, FORGIS_14)

Run date 2026-09-16 · reporter: Spoke J · workspace `C:\Users\PratikJawahar\chlu\chlu-bundle-v2`

---

## THE THREE LINES

1. **`status` = `NOT-READY`.** Exactly **one** thing is red: the targeted test file reports
   `1 failed, 9 passed, 27 skipped`. **All six verifier checks that inspect the bundle itself are
   green** — 146/146 checksums, 146/146 loaders, 0 missing, 0 mismatched, 0 unopenable, import under
   root, root tree == pin tree, write probe writable. ⭐ The single failure is in the **test harness's
   Windows mock**, not in the bundle, not in the shipped verifier, and not in `chlu/` — proof in §6.2.
2. **`anchors_checked` NON-ZERO: ⭐⭐ YES.** All **4** docs that have registered anchors report
   non-zero (`2 / 4 / 4 / 1`), exactly matching their `DOC_ANCHORS` entry counts. On v1 all four were
   `0`. **The `.as_posix()` fix works on Windows.** The other 8 `open_with: "doc"` entries report `0`
   because they have **no** `DOC_ANCHORS` entry — correct by design, not a regression.
3. **Frozen-artefact continuity v1→v2: 56 overlap · 52 matched · 4 differ · ⭐ 0 of the differences is
   a frozen artefact.** All 43 overlapping `.claude/outputs/**` artefacts and all 4 `.claude/data/**`
   files are **bit-identical across the two pins**. The 4 that differ are the charter (expected,
   6610→8672 B) and the 3 tooling files (expected — they are *at* the pin in v2).

---

## 1. Extract

```
cd /c/Users/PratikJawahar/chlu
tar xzf chlu-bundle-v2.tar.gz -C /c/Users/PratikJawahar/chlu     # exit 0
```

- Tarball re-verified before extracting:
  `7b4f120f73c2c8636525e0a98b3182e980aef4aab861cbab9e9a163b2998872b`, 1,412,267,343 bytes.
  **Matches.** Not re-downloaded.
- Top-level members inspected first: a single directory `chlu-bundle-v2` — extracted as a **sibling**
  of `chlu-hub-bundle`, as instructed.
- stderr was **143 × `tar: Ignoring unknown extended header keyword 'SCHILY.fflags'`** and nothing
  else. Benign (macOS bsdtar file-flags headers GNU tar does not consume); exit 0.
- ⭐ **The tarball is NOT inside the extracted tree** — a `find` for `*.tar.gz` / `*.tgz` under
  `chlu-bundle-v2` returns nothing. The v1 `cp .` accident did not recur.
- ⭐ **0 AppleDouble `._*` and 0 `.DS_Store`** inside the tree, confirming the hub's read-only finding
  on disk. 703 files outside `.git/` and `.venv/`.
- ⚠ A stray `._chlu-hub-bundle` (163 B) sits in the **parent** `chlu/` directory. It predates this
  run, is a v1-adjacent artefact, is outside both bundles, and I did not touch it.

⚠ **Process note, in the spirit of the brief's warning:** my *first* extract attempt mistyped the
working directory (`cd /c/Users/PratikJawahar` instead of `…/chlu`). `tar` failed closed —
`Cannot open: No such file or directory`, exit 2, **nothing was written anywhere**. I re-ran with an
explicit `-C` absolute path. Recording it because it is exactly the hazard §1 calls out, and it cost
nothing only because `tar` could not find its input.

## 2. Stale-environment kill — the env proof

This machine's Git Bash gives a **fresh shell per invocation**, so nothing was inherited to begin with:
`env | grep ^CHLU_` before sourcing returned `(none inherited)`, and no `chlu` path was on the inbound
`PATH`. I ran the §2 procedure anyway. Output of `env | grep ^CHLU_` after `. ./env_hub.sh`:

```
CHLU_CLAUDE_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude
CHLU_GEN_BRIDGE_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-feature-bridge/runs
CHLU_GEN_DATA_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/data/gen
CHLU_GEN_DECLIFT_DIR=/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-decoder-lift/winner
CHLU_GEN_STORE_RUNS=/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-store-grouping/runs
CHLU_GEN_WIDE_BRIDGE_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-bridge-wide/runs
CHLU_HUB_OUTPUTS=/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs
CHLU_ROOT=/c/Users/PratikJawahar/chlu/chlu-bundle-v2
```

⭐ **All eight contain `chlu-bundle-v2`. `env | grep -i chlu-hub-bundle` → NONE.** `PYTHONPATH` is the
v2 root alone. No v1 `.venv/Scripts` on `PATH` — independently corroborated in §5 by the verifier's own
line `import: chlu=C:\Users\PratikJawahar\chlu\chlu-bundle-v2\chlu\__init__.py under_root=True`.

## 3. Venv

⚠ **`uv` is not on `PATH` in any shell on this box.** It is not in `~/.local/bin`, `~/.cargo/bin`, or
`%LOCALAPPDATA%\Programs\uv`. It lives at
`C:\Users\PratikJawahar\AppData\Local\Programs\Python\Python312\Scripts\uv.exe` (a `pip install uv`
into the base 3.12.10). ⭐ Likewise, bare `python` / `python3` resolve to the **Microsoft Store alias
stub**, not a real interpreter. Anyone scripting v2 bring-up must prepend that Scripts directory.

```
uv --version      -> uv 0.12.15 (x86_64-pc-windows-msvc)
uv sync --locked  -> exit 0
```

⭐ **`--locked` was accepted; `uv.lock` was NOT rewritten, and there was no conflict to report.**
I used `--locked` rather than a bare `uv sync` precisely so a silent lock rewrite was impossible.

⭐ **Neither `--extra gen` nor `--group test` proved necessary** — I checked rather than assumed:

- `--group test` is unneeded because uv installs the **`dev`** dependency-group by default, which
  already carries `pytest 8.4.2` + `pytest-cov 5.0.0`. Both §5's targeted test and §8b's precision
  tests ran on it.
- `--extra gen` is unneeded because **`h5py` is absent and all 146 loaders still opened**
  (`unopenable=0`). No bundled artefact takes the Shapes3D HDF5 path; the raw dataset is deliberately
  not bundled.
- ⛔ `--extra cuda` was never used.

| | version |
|---|---|
| python | **3.11.16** (uv-managed, `AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none`) |
| jax | **0.9.0** |
| jaxlib | **0.9.0** |
| numpy | **2.4.1** |
| `jax.devices()` | `[CpuDevice(id=0)]` (`TFRT_CPU_0`), `default_backend='cpu'` |
| `chlu.__file__` | `C:\Users\PratikJawahar\chlu\chlu-bundle-v2\chlu\__init__.py` |

⭐⭐ **Unprompted but load-bearing for §8: `uv` resolved Python 3.11.16 on its own, and jax/numpy came
out at 0.9.0 / 2.4.1 — *identical* to the Mac reference's `0.9.0 / 2.4.1`, on a 3.11.13 Mac.** The two
boxes are therefore on the same jax and the same numpy, and the same Python minor. Python version is
not a free variable in the §8 comparison.

## 4. Git hygiene

Both set **before** any git operation touched the tree:

```
git config --local core.autocrlf false     -> file:.git/config  false
git config --local core.filemode false     -> file:.git/config  false
```

- `git fsck --no-progress` → **clean, zero output, exit 0.** (As predicted — no AppleDouble this time.)
- `git status --porcelain | wc -l` → **0.** ⭐ Confirmed *after* the full verify run, so the write
  probe, `WORKSPACE-READY.json`, `coverage.xml` and `.venv/` are all correctly gitignored.
- `HEAD` = `8e36bc94893130d444a4f237d49bd5a94c573cdd`, a single commit:
  `[bundle] the CHLU tree at pin 4125fce70b60ba1d9627ddc352d25bad5a08974a, verbatim`
- Root tree = **`345244bbd40a29a7842357342e32e528890e0670`** = the brief's expected `pin_tree`. ✅
- `git ls-files | wc -l` = **419**. ✅ (matches the brief's v2 row)
- ⚠ The upstream pin commit `4125fce7…` is **not an object in this repo** (`git cat-file -t` fails).
  Expected: the bundle re-commits a `git archive` of that pin, so the *tree* is the pin's and the
  *commit* is the bundle's own. Same shape as v1. The verifier checks the tree, which is the right check.

## 5. Verify — the six checks, separately

Command (run from `$CHLU_ROOT`):

```
VERIFY_PYTHON="$CHLU_ROOT/.venv/Scripts/python.exe" bash scripts/verify_hub_workspace.sh
```

Exit **1**. Reproduced identically on two separate runs (`1 failed, 9 passed, 27 skipped`).

| # | check | result |
|---|---|---|
| 1 | re-checksum every manifest file (sha256) | ✅ **PASS** — `files=146 missing=0 mismatched=0`, `checksum_match: true` |
| 2 | open every artefact through its REAL production loader | ✅ **PASS** — `unopenable=0`, `loaders_all_opened: true`, `skip_load: false` |
| 3 | read the DOCS and assert each one's anchors | ✅ **PASS** — 12/12 docs loaded; **anchors now actually inspected** (see §6) |
| 4 | import `chlu`, assert `__file__` under `$CHLU_ROOT` | ✅ **PASS** — `under_root: true`, `sys_path_0` = v2 root, jax 0.9.0 |
| 5 | run ONE targeted test (`tests/test_hub_workspace.py`) | ⛔ **FAIL** — `1 failed, 9 passed, 27 skipped`. **The only red.** See §6.2 |
| 6 | assert from GIT that root tree IS the pin's tree | ✅ **PASS** — `matches_pin: true`, `worktree_possible: true` |
| + | hub write-path probe (not numbered in the header) | ✅ **PASS** — `hub-refusal` → `writable: true` |

⛔ **No `SKIP_LOAD`, no `SKIP_TEST`, no `--force`, nothing edited to make a check pass.**
`WORKSPACE-READY.json` records `skip_load: false`.

### 5.1 Is `PYTHONUTF8=1` still load-bearing? — ⭐ YES, but only for the TEST FILE

I tested it, once, both ways (cheap — ~15 s of test time each):

| | checks 1/2/3/4/6 + write probe | check 5 (targeted test) |
|---|---|---|
| **with** `PYTHONUTF8=1` | all green | `1 failed, 9 passed, 27 skipped` |
| **without** `PYTHONUTF8` (locale = **cp1252**) | **all still green** | ⛔ `3 failed, 7 passed, 27 skipped` |

⭐ **The shipped verifier no longer needs it.** `grep -n 'read_text()' scripts/verify_hub_workspace.sh`
returns **zero** bare calls, and `encoding="utf-8"` appears **14×** — the fix is complete in the
shipped code, and checks 1–4 and 6 pass under cp1252.

⛔ **But `tests/test_hub_workspace.py` still needs it.** The two extra failures are
`test_the_legacy_absolute_path_survives_only_as_a_default` and `test_verify_needs_no_gpu_and_no_cluster`,
both `UnicodeDecodeError: 'charmap' codec can't decode byte 0x90`. Cause: the **test file** calls
`VERIFY_SH.read_text()` with no `encoding=` at **lines 247 and 254**, and the verifier it is reading is
full of `⭐⛔⚠`. There are 21 bare `read_text()` calls in that test file overall.

⇒ **Net: `PYTHONUTF8=1` remains required to reach `ready` on this box, but the dependency has moved
out of the product and into the test.** ⛔ I did not patch it — the file is manifest-pinned.

### 5.2 `WORKSPACE-READY.json` — top level (146-entry `files` array elided)

Copied verbatim to `_hub_scratch/reports/_J-workspace-ready.json` (from the canonical `PYTHONUTF8=1` run).

```json
"schema": "chlu-hub-workspace-ready/1",
"status": "NOT-READY",
"pin": "4125fce70b60ba1d9627ddc352d25bad5a08974a",
"n_files": 146, "n_missing": 0, "n_checksum_mismatch": 0, "n_load_failed": 0,
"checksum_match": true, "loaders_all_opened": true, "skip_load": false,
"import_under_root": true, "code_is_the_pin": true, "hub_output_path_writable": true,
"worktree_possible": true,
"targeted_test": "FAILED (rc=1): 1 failed, 9 passed, 27 skipped in 13.36s",
"host": "FORGIS_14",
"platform": {"machine": "AMD64", "python": "3.11.16", "release": "10", "system": "Windows"},
"chlu_root": "C:\\Users\\PratikJawahar\\chlu\\chlu-bundle-v2",
"manifest_generated_at": "2026-09-16T17:59:35.593667+00:00",
"total_bytes_on_disk": 2321832945,
"hubs": ["hub-refusal", "hub-generation-table", "hub-composition"],
"artefact_axes": {"datasets": ["shapes3d"], "pretexts": ["recon"], "seeds": [0, 1, 2]},
"supplements": [],
"git_check": {"repo": true, "branch": "main",
  "root_commit": "8e36bc94893130d444a4f237d49bd5a94c573cdd", "head": "8e36bc9",
  "root_tree": "345244bbd40a29a7842357342e32e528890e0670",
  "pin_tree":  "345244bbd40a29a7842357342e32e528890e0670",
  "matches_pin": true, "worktree_possible": true, "error": null},
"import_check": {"ok": true, "under_root": true, "chlu_version": "0.2.4", "jax": "0.9.0",
  "jax_devices": ["TFRT_CPU_0"], "error": null},
"write_check": {"hub_slug": "hub-refusal", "writable": true, "error": null},
"hub_requirements": {"hub-composition": {"n_files": 3},
                     "hub-generation-table": {"n_files": 3},
                     "hub-refusal": {"n_files": 92,
                       "excluded": [".claude/outputs/gen-arms-a7-wiring/refusal_scores.json"]}}
```

⭐ Confirms the brief's v2 table from the artefact itself: `hubs` = composition / refusal /
**generation-table**, **no `hub-deletion`**. `supplements: []` — the refusal artefacts are native, and
⛔ I did **not** apply `hub-refusal.supplement.tar.gz`.

⭐ Note `hub_requirements["hub-refusal"].excluded` lists `gen-arms-a7-wiring/refusal_scores.json` with
the reason *"⛔ marked superseded/never-ship in its own clause of the charter"* — **the quarantine is
now enforced by the manifest itself**, not just by convention. Good.

---

## 6. ⭐⭐ THE ACCEPTANCE CRITERION — `anchors_checked`

### 6.1 All 12 `open_with: "doc"` entries

| path | `anchors_checked` | chars | lines | loaded |
|---|---:|---:|---:|---|
| `.claude/AGENT_PROTOCOL.md` | **2** | 9,713 | 72 | true |
| `.claude/agents/autonomous-hub.md` | **4** | 10,433 | 158 | true |
| `.claude/autonomous-hub-charter.md` | **4** | 8,533 | 105 | true |
| `.claude/handover_context.md` | **1** | 1,001,293 | 3,845 | true |
| `.claude/outputs/gen-read-cell-arm/runs/dynread_seed0.log` | 0 | 9,495 | 100 | true |
| `.claude/outputs/gen-read-cell-arm/runs/dynread_seed1.log` | 0 | 9,871 | 103 | true |
| `.claude/outputs/gen-read-cell-arm/runs/dynread_seed2.log` | 0 | 9,904 | 103 | true |
| `.claude/outputs/gen-read-cell-arm/runs/progress.txt` | 0 | 414 | 12 | true |
| `.claude/outputs/gen-read-cell-arm/runs/run_scored.out` | 0 | 414 | 12 | true |
| `.claude/tasks/hub-composition.md` | 0 | 4,658 | 66 | true |
| `.claude/tasks/hub-generation-table.md` | 0 | 5,750 | 76 | true |
| `.claude/tasks/hub-refusal.md` | 0 | 4,723 | 65 | true |

### ⭐⭐ Plainly: THE FIX WORKS ON WINDOWS.

`DOC_ANCHORS` in `verify_hub_workspace.sh` registers anchors for **exactly four** paths, and the counts
above match its entries **one for one**: `AGENT_PROTOCOL.md` → 2, `agents/autonomous-hub.md` → 4,
`autonomous-hub-charter.md` → 4, `handover_context.md` → 1. **On v1 every one of these was `0`.**

**No anchored doc reports `0`.** The eight zeros are files with no `DOC_ANCHORS` entry at all —
`len(DOC_ANCHORS.get(rel, []))` is legitimately `0` for them; they are still opened, decoded and
non-empty-checked. That is the designed behaviour, not the bug.

Check #3 is therefore **live again on Windows**: a truncated charter would now be refused here. I
confirmed the mechanism directly — running the shipped `_load_doc` against a real `pathlib.Path` on the
real charter returns `anchors_checked: 4`, and against a **truncated** charter it raises
`required anchor(s) absent: …`.

### 6.2 ⛔ The one failing test — it is the harness's mock, NOT the bundle

`tests/test_hub_workspace.py::test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`

```
E  AssertionError: the document was decoded as cp1252, not utf-8: 245 chars read for a 234-char file
E                  (each ⭐/⛔/⚠ became its three raw utf-8 bytes)
E  assert 245 == 234
tests\test_hub_workspace.py:641: AssertionError
```

⭐ **The stated diagnosis is wrong, and it is arithmetically impossible.** The delta is **11**. The
test's text contains exactly **three** non-ASCII characters (`⭐⛔⚠`); a cp1252 misdecode would add
2 chars each = **+6**, not +11. Worse, cp1252 **cannot decode these bytes at all** — `0x90` is
undefined in cp1252 and raises `UnicodeDecodeError` under the mock's `errors="strict"`. So the code
path the message describes could not have produced this number.

**The real cause: CR bytes.** Measured directly:

```
len(text)                        : 234
newlines in text                 : 11
bytes on disk after write_text   : 251
CR bytes on disk                 : 11     <-- exactly the delta
utf-8 decode of raw bytes        : 245    <-- what the mock returns
real Path.read_text(utf-8)       : 234    <-- what actually runs on this box
```

`Path.write_text(text, encoding="utf-8")` opens in **text mode** (`newline=None`), so on Windows every
`\n` is written as `\r\n`. The test's `_WindowsFlavourPath.read_text` is implemented as
`read_bytes().decode(...)`, which **skips the universal-newline translation that real
`pathlib.Path.read_text` performs**. The mock reproduces cp1252 and backslashed `relative_to`, but it
does **not** reproduce text-mode newline translation — and on Windows that is a third divergence.

Decisive side-by-side, running **the shipped `_load_doc` source** both ways:

| `Path` class | `chars` | vs `len(text)` = 234 | `anchors_checked` |
|---|---:|---:|---:|
| `_WindowsFlavourPath` (the mock) | 245 | **+11** (= CR count) | **4** |
| real `pathlib.Path` (**what actually runs here**) | **234** | **0** ✅ | **4** |

⭐ Three conclusions: (a) the shipped `_load_doc` decodes UTF-8 **correctly** on this machine — the
`encoding="utf-8"` fix is genuine and present; (b) `anchors_checked == 4` under **both** paths, so the
`.as_posix()` fix is confirmed independently of this failure; (c) the test can only fail where
`os.linesep == "\r\n"`, i.e. it was authored and green on macOS and is **red on every Windows box**.

⛔ **I did not patch it.** `tests/test_hub_workspace.py` is one of the 146 manifest-pinned files, and
CONTEXT forbids editing a test to make a check pass. This needs an upstream fix — see Q1.

---

## 7. Scientific continuity v1 → v2 (read-only; v1 untouched)

Compared `BUNDLE_MANIFEST.json` sha256 by path: v1 (57 entries, pin `ae907466…`) vs v2 (146 entries,
pin `4125fce7…`). Both manifests carry `schema: chlu-hub-workspace-bundle/1`.

*Method note:* comparing manifests is equivalent to comparing bytes on disk here, because each
manifest was independently proven to match its own tree — v1 reached `status: ready` (57/57) and v2
re-checksummed 146/146 with `n_checksum_mismatch: 0` in §5. So a manifest-to-manifest match is a
disk-to-disk match. Nothing in v1 was read except its manifest, and nothing in v1 was written.

**Overlap: 56 paths · matched: 52 · differ: 4**

| category | matched | differ |
|---|---:|---:|
| ⭐ `.claude/outputs/**` (frozen scientific artefacts) | **43** | **0** |
| `.claude/data/**` | **4** | **0** |
| `.claude/` docs + tasks | 5 | 1 |
| code / tooling | 0 | 3 |
| **TOTAL** | **52** | **4** |

### Every file that differs — named

| path | v1 → v2 | verdict |
|---|---|---|
| `.claude/autonomous-hub-charter.md` | `41b6fcbe…` 6,610 B → `0204f799…` 8,672 B | ✅ **Expected.** The brief predicts exactly this. |
| `scripts/bundle_hub_workspace.sh` | `d6e3d3ef…` 48,051 B → `607b9dab…` 77,124 B | ✅ **Expected** — tooling, now *at* the pin. |
| `scripts/verify_hub_workspace.sh` | `d98fbb46…` 22,458 B → `e9fa770a…` 25,876 B | ✅ **Expected** — carries the `.as_posix()` + `encoding` fixes. |
| `tests/test_hub_workspace.py` | `eca5c97e…` 32,144 B → `bed159d5…` 56,531 B | ✅ **Expected** — +24 KB of new Windows-flavour tests. |

### ⭐⭐ THE RESULT THE HUMANS SHOULD NOTE

**Not one frozen scientific artefact changed between the two pins.** All 43 overlapping
`.claude/outputs/**` files and all 4 `.claude/data/gen/**` files are **bit-identical**, including the
three `decoder_v1_s{0,1,2}.pkl` winners, the `gen-feature-bridge` decoders, `winner_hashes.json`, the
`gen-arms-a7-wiring` baselines manifests and the quarantined `refusal_scores.json`. **The re-pin did
not disturb the corpus.** Every difference is a document or a tool — exactly as the charter claims
("Every other harness file is byte-identical to the old pin").

Also as predicted:

- **v1-only, dropped in v2 (1):** `.claude/tasks/hub-deletion.md`. ✅ `hub-deletion` is gone.
- **v2-only, new in v2 (90):** overwhelmingly the natively-shipped `hub/refusal` artefacts
  (`gen-read-cell-arm/runs/**`, `gen-val-split-fix/refusal_scores.json`) plus
  `.claude/tasks/hub-generation-table.md`.

---

## 8. Encode-precision parity

### 8a. `encode_precision_record()` — side by side

| key | **Mac (Darwin arm64, py3.11.13)** | **This box (Windows AMD64, py3.11.16)** | |
|---|---|---|---|
| `requested` | `highest` | `'highest'` | ✅ |
| `resolved` | `HIGHEST` | `'HIGHEST'` | ✅ |
| `backend` | `cpu` | `'cpu'` | ✅ |
| `device_kind` | *(not reported)* | `'cpu'` | ⭐ new |
| `n_devices` | *(not reported)* | `1` | ⭐ new |
| `jax_version` | *(0.9.0, from the header)* | `'0.9.0'` | ✅ |
| numpy | 2.4.1 | **2.4.1** | ✅ |

⭐ **All three comparable keys match the Mac exactly, and jax/numpy match too.** The two boxes resolve
`ENCODE_PRECISION` identically.

### 8b. The precision tests

```
python -m pytest tests/test_gen_bridge.py -k "precision" -q --no-cov -p no:cacheprovider
4 passed, 35 deselected, 1 warning in 3.84s
```

⚠ The brief says three; `-k precision` in fact selects **four**. All four pass:

| test | result |
|---|---|
| ⭐⭐ `test_encode_precision_does_not_move_the_numbers_on_cpu` | ✅ **PASSED** |
| `test_encode_path_convolutions_carry_the_requested_precision` | ✅ PASSED |
| `test_encode_precision_record_names_the_precision_backend_and_device` | ✅ PASSED |
| `test_encode_images_announces_the_precision_once_per_process` | ✅ PASSED |

⭐⭐ **Reporting the load-bearing one explicitly, as asked:
`test_encode_precision_does_not_move_the_numbers_on_cpu` PASSES on native Windows.** Its docstring's
condition — *"if this ever fails on the laptop, the fix has changed the science"* — **is not
triggered.** `Precision` remains a no-op on this CPU backend, so the assumption every
`features_sha256` in the programme rests on holds here.

### 8c. The sum/max/SHA256 probe — ⛔ **NOT RUN, pending the script**

No probe script was supplied to me, and **I did not invent one.** I independently confirmed it is
absent rather than taking it on trust: `8821b833` (the reference SHA) appears in **0** files anywhere
in the bundle. `51849` and `4.8279` do hit 2–3 files, but those are **incidental substring matches
inside JSON score arrays** (`gen-arms-a7-wiring/refusal_scores.json`,
`gen-val-split-fix/refusal_scores.json`) and `uv.lock` — not a probe, and not its input. The probe's
input therefore remains unknown, so a reconstruction would compare a different quantity and any
mismatch would be misread as platform divergence. ⇒ **Third comparison: NOT RUN.** Supply the script
and it is a two-minute job.

### 8d. Reading the above correctly

Per the brief's own framing: `ENCODE_PRECISION` is a **no-op on both machines** (both CPU); it exists
to stop XLA:GPU using TF32. Nothing in §8a/§8b speaks to Mac-CPU vs Windows-CPU arithmetic, and
nothing here should be read as establishing cross-machine parity. **No ULP-level disagreement was
observed at all**, because no cross-machine numeric comparison was possible without the §8c probe.
⛔ **Cross-machine parity remains UNSETTLED, and the Head's ruling that no hub starts on this box until
it is settled is untouched by this report.**

---

## 9. What is broken

1. ⛔ **`status` is `NOT-READY`, so no hub may launch on v2 yet** — the workspace gate is binary and it
   is red. The cause is §6.2's harness bug, but **the gate does not care why**, and I will not pretend
   it is green.
2. ⛔ **`test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale` fails on any Windows
   box.** Mock fidelity, not product. Full analysis and full output in §6.2. The fix is upstream and
   one line: give `_WindowsFlavourPath.read_text` the newline translation real `Path.read_text` does
   (or assert against a CR-normalised length). ⛔ Not patched here — manifest-pinned.
3. ⚠ **`PYTHONUTF8=1` is still required to pass check #5**, now because of 21 bare `read_text()` calls
   in the *test file* (lines 247 and 254 are the two that bite). The shipped verifier is clean.

## 10. What I could not resolve honestly

- **§8c's third comparison.** No script, unknown input, so no number. Guessing it would have
  manufactured exactly the false alarm the brief exists to prevent.
- **Whether v2 would reach `ready` with the test fixed.** Every other check is green and the failure is
  provably confined to the mock, so I believe it would — but **I did not and could not demonstrate it**
  without editing a pinned file, which is forbidden. Treat "would be ready" as my reasoned expectation,
  ⛔ not as a verified result.
- **Cross-machine numeric parity.** Out of scope for this brief and untouched by it (§8d).

## 11. Housekeeping confirmations

- ⭐ **No sub-agents.** I launched none at any point, so none can still be running.
- ⭐ **v1 is untouched:** `git -C /c/Users/PratikJawahar/chlu/chlu-hub-bundle status --porcelain | wc -l`
  → **0**. Its `WORKSPACE-READY.json` still reads `"status": "ready"`, and its newest file mtime
  (19:44) predates this session (20:26+). Nothing in v1 was read except `BUNDLE_MANIFEST.json`;
  nothing was written, moved or deleted.
- ⛔ `hub-refusal.supplement.tar.gz` **not applied, not moved, not touched** (`supplements: []`).
- ⛔ No manifest-listed file edited anywhere. No `git commit` / `push` / `reset` / `checkout --`.
  No scored runs, no hub launched, no GPU, no cloud spend.
- ⭐ `OMP_NUM_THREADS=MKL_NUM_THREADS=4` throughout (the brief named no figure; capped to 4 of 16 cores
  to stay a good citizen alongside parallel spokes).
- The OneDrive checkout at `Desktop\CLU-dev` was never worked in.
- Artefact delivered: `_hub_scratch/reports/_J-workspace-ready.json` (canonical `PYTHONUTF8=1` run).

---

## 12. Numbered open questions

1. ⭐⭐ **Will you fix `_WindowsFlavourPath.read_text` upstream and re-cut the manifest?** It is the
   only thing between v2 and `ready` on this box. The mock needs to reproduce **three** Windows
   behaviours, not two: backslashed `relative_to`, cp1252 default decode, **and text-mode `\n`→`\r\n`
   translation**. The assertion's failure message also misdiagnoses the cause and should be corrected,
   or the next person will chase cp1252 for an hour as I nearly did.
2. ⚠ **Should `tests/test_hub_workspace.py`'s 21 bare `read_text()` calls get `encoding="utf-8"`?**
   The shipped verifier was fixed but the test file was not, so `PYTHONUTF8=1` stays load-bearing for
   the wrong reason. The test file itself argues (line 630) that *"the encoding belongs in the CODE"* —
   it does not yet take its own advice.
3. ⛔ **`DOC_ANCHORS` still pins the OLD pin.** The charter anchor list requires the literal string
   `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`. v2's charter passes **only because line 25 retains a
   `### (superseded) ae907466…` marker**. So the anchor whose stated purpose is *"a charter that does
   not carry it cannot pin a hub"* is now verifying a **superseded** pin, and does **not** assert v2's
   real pin `4125fce7…` (which appears at line 15 in short form as `4125fce`). Two consequences: the
   guard is weaker than it reads, and it will fail spuriously the day someone tidies the superseded
   marker out. Should the anchor move to `4125fce`?
4. ⚠ **Is `uv`'s invisibility intended?** `uv`, `python` and `python3` are all unusable from a default
   shell on this box; `uv` only exists inside the base 3.12.10's `Scripts` directory. Should the
   bring-up instructions name that path, or should `uv` go on the user `PATH` persistently?
5. ⭐ **Confirm the venv policy.** `uv sync --locked` alone was sufficient — `dev` (hence pytest) is
   installed by default and no loader needed `h5py`. Do you want `--extra gen` installed anyway so the
   8 Shapes3D tests in `tests/test_gen_data_splits.py` stop silently `importorskip`-ing? They are
   currently invisible skips, which is the exact failure mode `pyproject.toml` warns about.
6. ⚠ **The brief's "×8 `--supplement-apply`" count does not match the tree:** I find **11** occurrences
   across 10 lines (`bundle_hub_workspace.sh` 6, `verify_hub_workspace.sh` 1, `test_hub_workspace.py` 3).
   Harmless, but if that count was a verification signal, re-derive it.
7. ⭐ **v2 revives the `chlu` CLI on Windows — confirmed, and it changes what this box can do.**
   `import chlu.cli` and `import chlu.chlu` both succeed and `python -m chlu.chlu --help` prints the
   full ~60-command surface. The `import resource` at `train_cluformer.py:829` is genuinely lazy (inside
   the non-Linux fallback branch, not a swallowed `try/except`). CONTEXT's ⛔ *"the entire `chlu` CLI is
   dead on native Windows"* is **obsolete at this pin**, and with it the
   `python -m chlu.experiments.exp_gen_cell` workaround. Does this unblock `chlu data shapes3d --download`
   here — i.e. is a download now sanctioned, given the hard constraint still forbids any **re-encode**
   that feeds a scored comparison?
8. ⚠ **Which hub is intended first on v2?** The write probe defaulted to `hub-refusal`, whose 92
   required files are now all present natively. But ⛔ the scored-run budget is three **per target
   across every machine**, and this box cannot see the Mac's count. Confirm the remaining budget for
   `hub/refusal` and `hub/generation-table` before anything is launched here.
