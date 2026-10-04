# gen-bundle-repairs — experiment-engineer report
Task + acceptance criterion: repair the three bundle/verify defects the Windows hub found; **each fix demonstrated FAILING before it is demonstrated passing.**
Status: **done**

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** ⛔ **none — correctness repairs to tooling.** **Falsifies:** nothing.
- **Laundering control:** ⭐ each fix demonstrated failing first. **Honoured for all three** (3+2+6 = 11 tests red before, green after).
- ⛔ No PREREG (no measured ratio/exponent/law). ⛔ No sub-agents. ⛔ No number enters any table. ⛔ No GPU.

## ⚠ DOWNSTREAM RECONCILIATION LIST — needs an owner (Hub, at the review that accepts this)
1. ⛔ **`gen-arms-a7-wiring/refusal_scores.json` IS ALREADY ON THE FAR SIDE, and it must stay.** It is not an omission to correct — `scripts/csf3/stage_artefacts.sh:240` ships it as `exp_gen_cell.check_preconditions`' own precondition. ⛔ Dropping it would break a *different* consumer, so it now travels **flagged**: `BUNDLE_MANIFEST.json` carries `superseded_for_hub: "hub-refusal"` and a `superseded_note` on that row. ⭐ **The Hub should still tell the far side in words** — a manifest field is not a charter.
2. ⚠ **The 2.1 GB bundle already on the far side was built by the pre-fix tar.** If it was extracted with GNU tar it has ~700 `._*` files inside `.git/` and `git fsck` fails there *now*. ⭐ The repair is to **rebuild+resend, or apply the supplement and re-extract `.git`** — ⛔ the fix stops creation, it cannot retro-clean a tree already on disk.
3. ⚠ **Every `WORKSPACE-READY.json` produced on Windows before this commit is void as evidence for check #3.** It reported `loaded: true, anchors_checked: 0` for all four documents. ⛔ Nothing was ever anchor-checked there.

---

## What I did
- **Defect 1** — `verify_hub_workspace.sh`: `_load_doc()` key normalised with `.as_posix()`; `encoding="utf-8"` pinned **in the code** on every `read_text`/`open` in the script (11 call sites), ⛔ not delegated to `PYTHONUTF8`.
- **Defect 2** — `bundle_hub_workspace.sh`: every `tar` now goes through `chlu_tar` (`COPYFILE_DISABLE=1` + `COPY_EXTENDED_ATTRIBUTES_DISABLE=1` + probe-gated `--no-mac-metadata`/`--no-xattrs` + `._*`/`.DS_Store` excludes), and `assert_no_apple_double` reads every archive back **with python** and refuses if any survive. The return-pack `find` excludes the same two patterns so its manifest and its tar agree.
- **Defect 3** — a **per-hub artefact requirement** derived from the hub's own charter, plus `--print-requirements`, `--require`/`--exclude-require`/`--no-charter-requires`, `--supplement` and `--supplement-apply`. Verifier gained a generic `npz` loader and now also opens whatever a supplement topped the workspace up with.
- 11 new tests in `tests/test_hub_workspace.py` (25 → 36 test functions).

## How I verified — ⭐ every fix RED before, GREEN after

### ⛔⛔ Defect 1 — the anchor check that inspected zero anchors
The tests `exec` the **shipped source** of `DOC_ANCHORS` and `_load_doc` out of `verify_hub_workspace.sh` under a `Path` class reproducing the two Windows behaviours (backslashed `relative_to`, cp1252 default decode). ⛔ Nothing is re-implemented.

**BEFORE (verify script unmodified at `main`) — `3 failed`:**
```
>       with pytest.raises(ValueError, match="required anchor"):
E       Failed: DID NOT RAISE <class 'ValueError'>          <- the truncated charter PASSED
E       AssertionError: check #3 inspected 0 of 4 anchors — it degraded to 'the file is non-empty'
E       AssertionError: the document was decoded as cp1252, not utf-8: 240 chars read for a 234-char file
```
**AFTER — `3 passed, 26 deselected in 0.13s`.**
A POSIX positive control asserts the same refusal fires on this platform, so the Windows leg cannot pass for the wrong reason.

### ⛔ Defect 2 — AppleDouble into `.git/`, and `git fsck`
⭐ **The finding that explains why this was never caught here.** macOS `tar` writes one `._<name>` member **per entry** and **hides them again on `tar -t` / `tar -x`** — it knows they are its own metadata. GNU tar on the far side has no such knowledge and writes every one out as an ordinary file. Same tarball, two readers:

| reader | members | AppleDouble |
|---|---|---|
| `tar -tzf` (bsdtar 3.5.3) | 1193 | **0** |
| python `tarfile` | 2386 | **1193** |

⛔ So *"our local `.git` has ZERO such files"* was true and irrelevant: they are created **in transit, by the tar**, exactly as reported. My own first probe reproduced 0 — because I listed with bsdtar. ⚠ **Any future check of this must read with python.**

**BEFORE — `2 failed`**, and the full chain on the retained pre-fix tarball:
```
AssertionError: 1193 AppleDouble member(s) in the tar: ['._b', 'b/._LICENSE', 'b/._uv.lock', ...]
PRE-FIX: AppleDouble files extracted to a fresh dir: 1192   (697 of them inside .git/)
         e.g. ['.git/._objects', '.git/._logs', '.git/._description']
PRE-FIX: git fsck rc=8
  error: refs/heads/._main: badRefName: invalid refname format
  error: refs/heads/._main: badRefContent:
AssertionError: the resource fork survived the archive
```
**AFTER — `2 passed`:** tar member list 0 AppleDouble, fresh extraction 0 on disk, **`git fsck` rc=0**, and the planted resource fork does not survive the archive. Extraction in the test uses python `tarfile`, ⛔ not macOS `tar`, because macOS `tar` is the unreliable witness.

### ⛔ Defect 3 — the omitted artefacts, and the class
⭐ **The charter is the declaration.** Every backticked token in a hub's task file that contains `/`, is relative, and ends in a file extension or `/` becomes a requirement, resolved under `$CHLU_CLAUDE_ROOT` (or `.../outputs/` for a bare `slug/...`). ⛔ The hub's own write path is never required. ⛔ A token whose ⛔/⚠/⭐-delimited **clause** marks it superseded is an exclusion — **clause, not line**, which is what separates the two `refusal_scores.json` paths that sit on one line of the real charter.

**Derived from the real `hub-refusal.md`, unmodified:**
```
  files to ship : 92
  ⭐ REQUIRED  .claude/outputs/gen-read-cell-arm/runs                     (88 files)
  ⭐ REQUIRED  .claude/outputs/gen-val-split-fix/refusal_scores.json
  ⭐ REQUIRED  .claude/AGENT_PROTOCOL.md / autonomous-hub-charter.md / handover_context.md
  ⛔ EXCLUDED  .claude/outputs/gen-arms-a7-wiring/refusal_scores.json   (superseded, same line as the one above)
     ignored   .claude/outputs/hub-refusal.md, .../hub-refusal/LOG.md   (the hub's OWN write path)
```
⛔ `hub/refusal` (a branch), `../CHLU-hub-refusal` (a worktree) and `exp_gen_baselines.py:1562` (a call site) are correctly not candidates.

**Acceptance 3 — `hub-refusal`'s real bundle built (2.172 GiB, 144 `.claude` files):**
```
⭐ PRESENT .claude/outputs/gen-val-split-fix/refusal_scores.json  2899315 bytes  open_with=json  on-disk=True
⭐ PRESENT .claude/outputs/gen-read-cell-arm/runs/  -> 88 files, 216.8 MB, all on disk=True
⚠  1 shipped file flagged SUPERSEDED by a hub charter
```

**Acceptance 4 — the refusal (rc=2, names the paths, writes nothing):**
```
FATAL: ⛔ hub-broken.md declares 2 artefact requirement(s) that do not exist — ⛔ REFUSING
  ⛔ MISSING  .claude/data/gen/nope/query_sets.json
  ⛔ MISSING  .claude/outputs/gen-nonexistent-slug/refusal_scores.json
  ⭐ Fix the charter, stage the artefact, or pass --exclude-require <path>.
$ ls /tmp/reqdemo/out  ->  No such file or directory
```
⭐ **Pre-fix the same case returned `rc=0` and built the bundle.** That rc=0 *is* the defect.

**The supplement, on the real 206 MB artefact set, against a workspace with a hub at work in it:**
```
[bundle] supplement: 92 file(s), 0.206 GiB   (⛔ not a 2.1 GB rebuild)
[bundle] ⛔ NOT overwriting .../.claude/AGENT_PROTOCOL.md (differs) -> wrote AGENT_PROTOCOL.md.supplement-20260916T162905Z
[bundle]   2 new · 89 already identical (no-op) · 1 differing, KEPT and written beside
```
The hub's edit survived verbatim. **Applied a second time:** `0 new · 92 already identical · 0 differing`, and still exactly **1** stamped copy — ⛔ a persistently-differing file does not accrue one sidecar per apply.

**BEFORE (bundler+verify reverted to `main`, new tests) — `6 failed`** (`unknown flag: --print-requirements`; `rc=0 (expected 2)`). **AFTER — `6 passed`.**

### ⭐⭐ END TO END, on the real 2.172 GiB bundle
`verify_hub_workspace.sh` run inside the built `hub-refusal` workspace, **rc=0**:
```
[verify] supplements: SUPPLEMENT_MANIFEST-20260916T162914Z.json, SUPPLEMENT_MANIFEST.json
[verify] import: chlu=/private/tmp/reqdemo/hubrefusal/chlu/__init__.py under_root=True jax=0.9.0
[verify] git   : root_tree=12831fe1… matches_pin=True
[verify] files=144 missing=0 mismatched=0 unopenable=0 import_ok=True
[verify] targeted test: passed: 10 passed, 21 skipped in 15.19s
[verify] status=ready
```
`WORKSPACE-READY.json`: `status: ready`, `code_is_the_pin: true`, `loaders_all_opened: true`,
`n_load_failed: 0` — **including all 78 required `.npz` traces opened through the new loader** — and
the charter row reports **`anchors_checked: 4`**, ⛔ not 0. The 21 skips inside the bundle are the
recursion guard doing its job (`CHLU_WORKSPACE_VERIFY_DEPTH`), ⛔ not failures.

## Targeted tests — final
```
tests/test_hub_workspace.py:  35 passed, 2 skipped, 0 failed  (340.20s)
```
⚠ **Both skips are environment gates, not defects:** `test_chlu_is_imported_from_chlu_root` (needs `$CHLU_ROOT` set) and `test_live_bundle_manifest_is_internally_consistent` (only runs *inside* an extracted bundle). `ruff check tests/test_hub_workspace.py` → **All checks passed**.
⛔ `ruff format --check` reports this file unformatted — ⚠ **it was already unformatted at `main`** (verified by stashing). ⛔ I did not reformat: §3.3 forbids opportunistic reformatting of a shared file.

## Findings/results
1. ⭐⭐ **macOS `tar` hides the AppleDouble members it writes.** This is the root reason a real, `git fsck`-breaking defect looked absent on this machine. ⭐ **Any check for this class must read the archive with python, ⛔ never with `tar -t`.** `assert_no_apple_double` now does exactly that on every archive the bundler writes.
2. ⛔ **Check #3 has never run on Windows.** Every document reported `loaded: true, anchors_checked: 0`; on that platform the check was `len(text.strip()) > 0`. The two `errors="replace"` sister reads meant the corruption was silent rather than loud.
3. ⭐ **The superseded a7 file cannot simply be dropped** — the launcher table opens it for `check_preconditions`. It ships, flagged. See reconciliation item 1.
4. ⚠ **`.DS_Store` litter is already inside the artefact tree** (`.claude/outputs/.DS_Store`, `.claude/outputs/cluformer-pilot/csf3_outs/.DS_Store`). Now excluded from every archive.
5. ⭐ Honouring `hub-refusal`'s own promise adds **+219.7 MB** to its bundle (216.8 MB of `gen-read-cell-arm/runs/` + 2.9 MB of `gen-val-split-fix/refusal_scores.json`); the built total is **2.172 GiB / 144 `.claude` files**.

## Git footprint
- **Worktree:** `../CHLU-bundlefix` · **Branch:** `agent/experiment-engineer/gen-bundle-repairs` · **Base:** `main` @ `a8acf7a`
- **Commit:** `d812e2f` — `[experiment-engineer] repair three hub-workspace bundle defects the Windows hub found`
- **Files touched (3, all owned):** `scripts/bundle_hub_workspace.sh` (+617/−…), `scripts/verify_hub_workspace.sh` (+81), `tests/test_hub_workspace.py` (+493). **1154 insertions, 37 deletions.**
- ⭐ Ref confirmed **from the main repo**: `git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/gen-bundle-repairs` → `d812e2f`. ⛔ Worktree left in place for Hub review; not removed, not pushed, no PR.
- ⚠ **ONE commit, not three.** All three defects live in the same two scripts plus one test file and interleave hunk-by-hunk; splitting them would have meant hand-reconstructing intermediate file states. §3.4 deviation, stated rather than hidden; the message enumerates the three defects separately.
- ⛔ No rebase was needed (base `main` did not move during the task). ⛔ No conflicts. ⛔ Nothing outside the four owned paths was touched — ⛔ not `chlu/**`, ⛔ not `scripts/csf3/**`, ⛔ not any charter.

## Cost
≈ 2 h wall-clock against a ≈ 4 h budget. CPU only, ⛔ no GPU, ⛔ no sub-agents. Dominant costs: 3 × full targeted-suite runs (≈ 5 m 40 s each, each building several real bundles), one real 2.172 GiB `hub-refusal` bundle build, one 206 MB supplement build + two applies.

## Open questions / follow-ups / risks
1. ⚠ **The requirement derivation is a heuristic over prose, and it is deliberately conservative.** A bare `slug/file.ext` is only picked up when its first segment is an existing outputs slug **or** has a hyphenated slug shape; anything else is ignored **silently**. ⭐ `--print-requirements` makes the verdict checkable in one command — ⭐ **the Hub should run it on each hub charter at the review that accepts it**, rather than trusting the derivation unseen.
2. ⚠ The negation vocabulary is fixed (`SUPERSEDED|never quote|never ship|do not ship|do not quote|not shipping|DEPRECATED`). A charter that bans an artefact in different words would ship it. ⛔ A missed exclusion ships a file that should not be quoted; it does not break the build.
3. ⚠ An unmapped file suffix in a required directory is a **refusal**, not a silent skip. Correct (an artefact that cannot be opened cannot be verified) but it will bite the first charter that names a `.h5` or a `.parquet` — the fix is one line in `BY_SUFFIX` plus a loader in the verifier.
4. ⛔ **Unverified:** `--no-mac-metadata`/`--no-xattrs` were probe-gated and confirmed present on this bsdtar 3.5.3; I could not test the GNU-tar path, where the probe should select `--no-xattrs` only. The `--exclude` and `COPYFILE_DISABLE` legs are platform-independent.
5. ⚠ The verifier now materialises every array of every required `.npz`. For `hub-refusal` that is 78 files / ~207 MB — correct (laziness is what lets a truncated member reach hour six) but it makes the far side's verify measurably slower.

## Proposed handover updates (for the Hub)
- `scripts/bundle_hub_workspace.sh` gained a **per-hub artefact requirement derived from the hub's own charter**. ⛔ **A hub charter that names an artefact in backticks now dictates the bundle's contents** — writing `.claude/outputs/<slug>/...` into a charter is a build-time commitment, and a path that does not exist **refuses the build**. ⭐ Run `--print-requirements --hub <slug>` when accepting any new hub charter.
- **Known Issues candidate:** ⛔ *"macOS `tar` writes `._*` AppleDouble members and hides them from `tar -t`/`tar -x`. Never verify an archive with macOS `tar`; read it with python. Any bundle built before `d812e2f` and extracted with GNU tar has ~700 `._*` files in `.git/` and fails `git fsck`."*
- **Known Issues candidate:** ⛔ *"Every `WORKSPACE-READY.json` produced on Windows before `d812e2f` reports `anchors_checked: 0` — the charter-anchor check never ran there."*
