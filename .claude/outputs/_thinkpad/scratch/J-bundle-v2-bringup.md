# SPOKE J — Bring up bundle v2 and drive it to `status: ready` with **non-zero `anchors_checked`**

Read `_hub_scratch/CONTEXT.md` first — **but note §"STATUS" in it describes v1.** This brief supersedes it
wherever they disagree.
Report to: **`/c/Users/PratikJawahar/chlu/_hub_scratch/reports/J-bundle-v2-bringup.md`**

## What is already done — do not repeat it
The tarball is **downloaded and checksum-verified** at `/c/Users/PratikJawahar/chlu/chlu-bundle-v2.tar.gz`
(`7b4f120f73c2c8636525e0a98b3182e980aef4aab861cbab9e9a163b2998872b`, 1,412,267,343 bytes).
The hub has already inspected it, read-only:
- **1328 members · 0 AppleDouble `._*` · 0 `.DS_Store`** — the bundler is fixed; there is **no cleanup to do.**
- All four promised fixes are present: `.as_posix()` (`verify_hub_workspace.sh:254`), `encoding="utf-8"`
  ×14, `--supplement-apply` ×8, and `import resource` **moved inside a function**
  (`train_cluformer.py:829`, lazy — not a swallowed `try/except`).
- **The refusal artefacts ship natively**: `gen-read-cell-arm` (102 members) and `gen-val-split-fix`.
  ⛔ **Do not apply `hub-refusal.supplement.tar.gz` to v2.** It is already inside.
- `env_hub.sh` is byte-identical to v1's.

## ⚠ v2 is not a patch release — it is a new pin
| | v1 | **v2** |
|---|---|---|
| pin | `ae907466…` | **`4125fce70b60ba1d9627ddc352d25bad5a08974a`** |
| pin_tree | `12831fe1…` | **`345244bbd40a29a7842357342e32e528890e0670`** |
| pinned files | 57 | **146** |
| tracked files | 416 | **419** (the 3 tooling files are now *at* the pin, not a later overlay) |
| charter | 6610 B | **8672 B** |
| hub tasks | composition, **deletion**, refusal | composition, refusal, **generation-table** |

⭐ **`hub-deletion` is gone from v2** — absent from `.claude/tasks/` and from the manifest's `hubs` list.
⭐ **`hub-generation-table` has arrived** — target 1, the headline, which v1 could not run at all.
Schema stays `chlu-hub-workspace-bundle/1`. Manifest now carries `hub_requirements` and `not_bundled` keys.

## ⛔ Rules
- ⛔ **Do not touch, move or delete `/c/Users/PratikJawahar/chlu/chlu-hub-bundle` (v1).** It is the known-good
  fallback and its `.claude/outputs/hub-composition/` holds the verifier's write-probe. **v1 stays until a
  human retires it.**
- ⛔ Never edit a manifest-listed file, never edit a script to make a check pass. `SKIP_LOAD=1`/`SKIP_TEST=1`
  may isolate a failure; they are **⛔ NOT a pass**.
- ⛔ No scored runs, no hub, no GPU, no cloud spend.

## Procedure

### 1. Extract — mind the working directory
⚠ **The last time this command block was run, a `cp .` landed a 206 MB tarball *inside* the bundle**
because the shell's cwd was not where the author assumed. Be explicit:
```bash
cd /c/Users/PratikJawahar/chlu
tar xzf chlu-bundle-v2.tar.gz          # creates ./chlu-bundle-v2/ — a sibling of chlu-hub-bundle/
cd chlu-bundle-v2
```
Confirm the tarball itself is **not** inside the extracted tree.

### 2. Kill the stale environment — this is a real hazard here, not a precaution
Your shell almost certainly has `CHLU_*` exported and pointing at **v1**, and `env_hub.sh` deliberately
honours an explicit export, so a stale value **silently overrides the derived root** and you would verify
v1 while believing you verified v2.
```bash
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
. ./env_hub.sh
env | grep ^CHLU_        # ⭐ EVERY path must contain chlu-bundle-v2
```
⭐ **Paste that `env | grep` output into your report.** If any path says `chlu-hub-bundle`, stop and fix it.
Also ensure **no v1 `.venv/Scripts` is on your `PATH`** — same failure, different mechanism.

### 3. Venv — v2 ships none by design
```bash
uv sync                 # add --extra gen --group test if a loader or the targeted test needs it; say which
```
⛔ Never `--extra cuda`. If `uv` wants to rewrite `uv.lock`, use `--frozen`/`--locked` and **report the
conflict** rather than letting it rewrite. Report jax/jaxlib/numpy versions and `jax.devices()`.

### 4. Git hygiene — set these *before* any git operation touches the tree
```bash
git config --local core.autocrlf false    # system gitconfig sets it TRUE; a flip CRLFs every .sh
git config --local core.filemode false    # macOS-written filemode=true vs NTFS ⇒ phantom "modified" rows
```
Then report `git fsck --no-progress` (expect clean — there are no AppleDouble files this time) and
`git status --porcelain | wc -l` (expect **0**).

### 5. Verify
```bash
VERIFY_PYTHON="$CHLU_ROOT/.venv/Scripts/python.exe" bash scripts/verify_hub_workspace.sh
```
`PYTHONUTF8=1` is already persisted at user scope; v2 also passes `encoding="utf-8"` explicitly, so it
should no longer be load-bearing — **say whether it still is**, by testing once without it if cheap.

**Report each of the six checks separately**, then paste `WORKSPACE-READY.json`'s top level (elide the
146-entry `files` array) and copy the file to `_hub_scratch/reports/_J-workspace-ready.json`.

### 6. ⭐⭐ THE ACCEPTANCE CRITERION
**`anchors_checked` must now be NON-ZERO** for every doc that has registered anchors. On v1 it was `0` for
all four — the `_load_doc()` backslash bug — and that made the charter-truncation guard silently inert.
v2 carries the `.as_posix()` fix at line 254.

⭐ **Report `anchors_checked` for all 12 `open_with: "doc"` entries in a table.** State plainly whether the
fix works on Windows. **This is the single result the humans are waiting for.** If any anchored doc still
reports `0`, that is the headline of your report.

### 7. Scientific continuity — ⭐ do this, nobody has asked for it and it matters
v2 is a **new pin**. Frozen artefacts are supposed to be frozen *across* pins. Verify that:
compare the sha256 of every file that appears in **both** v1's `BUNDLE_MANIFEST.json` (57 entries, at
`/c/Users/PratikJawahar/chlu/chlu-hub-bundle/BUNDLE_MANIFEST.json`) and v2's (146), by path.
- Report **how many overlap**, **how many match**, and **name every one that differs.**
- ⭐ A changed `.claude/outputs/**` artefact between pins would be a serious finding — those are the frozen
  scientific assets other machines' results already rest on. A changed **doc** (e.g. the charter, 6610 →
  8672 B) is expected and fine.
Do this read-only, against v1's files in place. ⛔ Do not modify v1.

## Report shape
Lead with three lines: **`status`**, **`anchors_checked` non-zero yes/no**, and **frozen-artefact
continuity v1→v2 (n matched / n differ)**. Then the six checks, the env proof from §2, the version table,
and §7's comparison. End with numbered open questions.

⭐ Before you finish, confirm no sub-agent of yours is still running, and confirm v1 is untouched
(`git -C /c/Users/PratikJawahar/chlu/chlu-hub-bundle status --porcelain | wc -l` should still be 0).

---

# §8 — ADDED: the encode-precision parity check. Run after §5 passes.

The Mac has produced a reference. **Reproduce the comparable parts exactly and report them side by side.**

```
MAC REFERENCE (Darwin arm64, py3.11.13)
  platform        : Darwin arm64 py3.11.13
  jax/numpy       : 0.9.0 / 2.4.1
  ENCODE_PRECISION: highest   (resolved=HIGHEST backend=cpu)
  sum(float64)    : 51849.2300147639
  max             : 4.8279623985
  SHA256          : 8821b833154f41a99f7ef7b515250c7a6ac58229bb08b2bde894ad243cb47dec
```

## 8a. `encode_precision_record()` — directly comparable, do this one
`chlu/core/gen_bridge.py:881`. Its output IS the Mac's third line. Run it in the v2 venv and report all
six keys:
```python
from chlu.core.gen_bridge import encode_precision_record
print(encode_precision_record())
```
Expect `requested='highest'`, `resolved='HIGHEST'`, `backend='cpu'`. ⭐ Report `device_kind`, `n_devices`
and `jax_version` too — the Mac's one-liner omits them and they are worth having on record.

## 8b. The three precision tests — canonical, deterministic, and already in the bundle
```bash
python -m pytest tests/test_gen_bridge.py -k "precision" -q --no-cov -p no:cacheprovider
```
⭐⭐ **`test_encode_precision_does_not_move_the_numbers_on_cpu` is the load-bearing one.** Its own docstring
says: *"`Precision` is a no-op on CPU backends — and it must stay one, because the whole feature cache was
written on CPU and every `features_sha256` in the programme depends on it. ⛔ If this ever fails on the
laptop, the fix has changed the science."* **Report its result explicitly, pass or fail.**

## 8c. ⛔ Do NOT invent the sum/max/SHA256 probe
The script that produced `sum=51849.2300147639`, `max=4.8279623985`,
`SHA256=8821b833…` is **not in the bundle** (grep confirms). Its input is therefore unknown.
⛔ **Do not guess it.** A probe with a different input yields a different hash, and the mismatch would be
misread as a platform divergence — the exact false alarm this whole line of work exists to avoid.
⭐ If — and only if — a probe script is supplied to you, run it unmodified and report all six lines beside
the Mac's. Otherwise record §8a and §8b and state plainly that the third comparison is **NOT RUN, pending
the script**.

## 8d. Context you need to read the result correctly
⚠ **`ENCODE_PRECISION` does not address Mac-CPU vs Windows-CPU.** `gen_bridge.py:866-873` states
`Precision` is *"documented as having no effect on CPU backends, which is exactly what this fix needs"* —
it exists to stop XLA:GPU running float32 convolutions through **TF32** on Ampere/Ada (the L40S defect:
`max|delta| = 2^-8 = 4181 float16 ULP` on 20.85 % of elements, **with the encoder parameter hash
matching**). Both machines here are CPU, so this fix is a no-op on both.
⚠ The charter's declared property #3 already concedes: **"CPU/GPU feature agreement is ~53 ULP on ~1 % of
elements, not bit-exact, after the precision fix (20.85 % → 0.96 %)."** ⭐ So the programme has *already*
accepted non-bit-exactness as the standard. Do not report ULP-level disagreement as a failure; report its
**magnitude** and let the humans judge it against that stated bar.
