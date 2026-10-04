# gen-hub-workspace-bundle — experiment-engineer report

**Task + acceptance criterion:** produce a portable hub workspace bundle and prove it works from a
different root — extracted elsewhere, `CHLU_ROOT` repointed, a hub-shaped smoke green there.
**Status: done.**

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** ⛔ **none — infrastructure.** **Falsifies:** nothing. ⛔ No arm, no science, no number that
  enters any table.
- **Laundering control:** extraction to a **different absolute path on this machine**. Executed:
  `/tmp/chlu-hub-test`, `CHLU_ROOT` set to it, `chlu` imported **from there** (asserted, not assumed).

## ⚠ RECONCILIATION LIST — items with no owner but mine (first 10 lines, per protocol §5)
1. ⭐ **The charter needs NO path edit** — it contains **zero** absolute paths. The allow-listed output
   path the brief attributes to it lives in `.claude/agents/autonomous-hub.md` §0.3, which is mine and is
   fixed. The proposed diff for files I do not own covers **3 lines in 2 files**, not the charter.
2. ⛔ **The brief's premise "JAX HAS NO NATIVE WINDOWS SUPPORT" is FALSE as measured today.** `jaxlib`
   publishes `win_amd64` CPU wheels for **every** version this repo can resolve. WSL2 is still the right
   call, but for **different, testable reasons**. §4 below. **The Hub must re-decide on the evidence.**
3. ⚠ **A frozen artefact changed on disk mid-session** (`classifier_gate.json`, 1297→1296 B, 13:02).
   Someone or something rewrote a charter-frozen asset while I was bundling it. §7.
4. ⛔ **`git bundle create … main..<branch>` cannot be fetched here** — the code return path fails *on
   arrival*. Fixed and measured (§5b). A hub told the wrong form loses its CLU changes.

---

## What I did

1. **Determined the minimum set and derived it from the LOADERS** — by *invoking*
   `scripts/csf3/stage_artefacts.sh --manifest-only`, not by re-listing anything. That script is
   unedited and uncopied. Added a small **hub addendum** (§2) that is also loader-derived.
2. **`scripts/bundle_hub_workspace.sh` (new)** — builds a bundle that *is* a complete `$CHLU_ROOT`;
   manifest + sha256 per file; idempotent; `--dry-run`; and the **return path** in both directions.
3. **`$CHLU_ROOT` as the single source of truth**, with the macOS checkout as its default, so ⛔ nothing
   on this machine changes behaviour. ⛔ No Python touched — the production code already reads every
   `CHLU_*` root; this supplies them from one variable.
4. **`scripts/verify_hub_workspace.sh` (new)** — re-checksums, **opens every artefact through its real
   loader**, checks each document still carries its own anchor, **asserts `chlu.__file__` is under
   `$CHLU_ROOT`**, asserts from git that **the code IS the pin**, writes the hub's output path, runs one
   targeted test, emits `WORKSPACE-READY.json`.
5. **`tests/test_hub_workspace.py` (new)** — 25 test functions (21 of them bundle-building, guarded by `@needs_bundle`), including every failure mode.
6. **Made the bundle a git repository** (§6) — because the hub's sandbox is a *worktree* and
   `git archive` yields a directory that can hold none.
7. **Edited `.claude/agents/autonomous-hub.md`** — path portability only, §0.2/§0.3. ⛔ No rule changed.
8. **Proposed (not applied) the diff** for `AGENT_PROTOCOL.md` and `handover_context.md`:
   `.claude/outputs/gen-hub-workspace-bundle/path-portability.diff`.

---

## 1. ⭐⭐ ACCEPTANCE #1 — the bundle works from a DIFFERENT root

Built from `/Users/user/Desktop/CHLU`, extracted at **`/tmp/chlu-hub-test`** (a different absolute path,
resolving to `/private/tmp/...`), `CHLU_ROOT` repointed by `source ./env_hub.sh`. ⛔ Not a dry run.

```
[verify] import: chlu=/private/tmp/chlu-hub-test/chlu/__init__.py under_root=True jax=0.9.0
[verify] git   : root_tree=12831fe1be17d04da027dcbb6e45b4dff6f1ef95 matches_pin=True head=0734733
[verify] write : /tmp/chlu-hub-test/.claude/outputs/hub-smoke-bundle writable=True
[verify] files=55 missing=0 mismatched=0 unopenable=0 import_ok=True
[verify] targeted test: passed: 7 passed, 19 skipped in 12.81s
[verify] status=ready                                                     ← rc=0
```

And a **hub-shaped boot** (`logs/hub_boot_smoke.log`) — the six things a hub does in its first five
minutes, in order, from the bundle root:

| step | result |
|---|---|
| 1. read `AGENT_PROTOCOL.md` | 9 713 chars, DIAL DECLARATION present |
| 2. read `handover_context.md` §7 | 1 001 293 chars, §7 located |
| 3. read the charter + **the pin** | 6 532 chars, `ae907466…1ab30e` present, floor clause present |
| 4. read its **own** task file | 1 037 chars |
| 5. **write** to `$CHLU_HUB_OUTPUTS/<hub>/LOG.md` | `/private/tmp/chlu-hub-test/.claude/outputs/hub-smoke-bundle/LOG.md`, inside root |
| 6. `import chlu` + a real gen loader | `chlu=/private/tmp/chlu-hub-test/chlu/__init__.py` v0.2.4; `load_store_plan` → **K=5, n_modes=64, D=2048** |

**Verdict: `HUB BOOT OK`.**

### ⛔⛔ The trap this exposed, and why the import assertion exists
`python -c` puts the **CWD at `sys.path[0]`, ahead of `PYTHONPATH`**. Measured on this machine:

```
cwd=/Users/user/Desktop/CHLU  PYTHONPATH=/Users/user/Desktop/CHLU-bundle
  -> chlu /Users/user/Desktop/CHLU/chlu/__init__.py        ⛔ PYTHONPATH LOST
cwd=/Users/user/Desktop/CHLU-bundle  (same PYTHONPATH)
  -> chlu /Users/user/Desktop/CHLU-bundle/chlu/__init__.py ✔
```
⚠ Without `import_under_root`, a verify run from the wrong directory passes **every** other check while
silently exercising the original machine's checkout — a portability proof that proves nothing.
`env_hub.sh` documents it; `verify_hub_workspace.sh` asserts it; `WORKSPACE-READY.json` records it.

---

## 2. ACCEPTANCE #2 — the minimum set, justified, with its size

**1.967 GiB total · 55 `.claude` files + 416 tracked code files.** Under the ~2.5 GB bar. (Source
`.claude/` is 31 GB: `data` 11 G, `outputs` 9 G, `scratch` 11 G.)

| kind | n | what |
|---|---|---|
| `artefact` | 47 | the frozen gen set, **derived from `stage_artefacts.sh`'s loader table** (44) + the hub addendum (3) |
| `doc` | 5 | the four files the agent profile names + the hub's own task file |
| `tooling` | 3 | the bundle/verify scripts + their test, **declared** as an overlay (§6) |

`open_with` dispatch: `eqx_pickle` 18 · `json` 14 · `npy` 6 · `doc` 5 · `feature_cache` 3 · `code` 3 ·
`store_plan` · `winner_hashes` · `c_invalid` · `query_registry` · `control_set` · `npz_index`.

**Where the 1.967 GiB is:**

| share | bytes | file |
|---|---|---|
| **93.08 %** | 1 966 080 128 | `outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/features.npy` |
| 1.80 % | 38 043 776 | …`seed_1/features.npy` |
| 1.80 % | 38 043 776 | …`seed_2/features.npy` |
| 2.84 % | — | all 52 remaining `.claude` files |
| 0.48 % | 10 076 160 | the entire code tree at the pin |

⭐ **Nine-tenths of the bundle is one file, and it is asymmetric on purpose.** Seed 0's feature cache is
`scope="full"` (480 000 rows); seeds 1 and 2 are `scope="used"` (9 288 rows) — a **51.7×** difference,
read out of the caches' own manifests:

| seed | scope | n_rows | features_bytes |
|---|---|---|---|
| 0 | `full` | 480 000 | 1 966 080 128 |
| 1 | `used` | 9 288 | 38 043 776 |
| 2 | `used` | 9 288 | 38 043 776 |

`full` is a superset of `used`, so it is correct to ship — but if a hub only ever reads the used rows,
re-encoding seed 0 at `scope="used"` would take the bundle from **1.967 GiB to ~0.13 GiB**. ⛔ I did not
do it: that is a **re-encode**, which is exactly what the pin's hazard gate
(`exp_gen_store._assert_cache_reencodes(will_extend=…)`) governs, and it is not an infrastructure
spoke's call. **Flagged for the Hub, not acted on.**

**⛔ Deliberately NOT in the bundle**, each with a reason (recorded in `BUNDLE_MANIFEST.json.not_bundled`):
- `.claude/scratch/` (11 GB) — no loader names any of it.
- Every other slug's `.claude/outputs/**` — the agent profile says *"⛔ Read nothing else from the campaign
  archive"*; shipping it would ship the distraction the profile exists to prevent.
- The campaign ledgers (`advisor-*`, `claims_matrix`, `negative_results`, `research_roadmap`, `papers/`).
- The dataset blobs (§3).

### The hub addendum — also loader-derived, ⛔ not a directory listing
`stage_artefacts.sh`'s table is scoped to the **five CSF3 launcher stages**. A hub aiming at a **refusal**
win runs `exp_gen_baselines`, which at `exp_gen_baselines.py:1547` opens the query-set registry and every
control set the registry *names*. Those rows are not in the launcher's table. The bundle **reads the
registry to name its own files** — the same discipline `stage_artefacts.sh` uses for `feature_cache.json`'s
`rows_file`/`features_file`:

| file | loader | size |
|---|---|---|
| `data/gen/shapes3d/query_sets.json` | `gen_splits.load_registered_query_sets` via `exp_gen_baselines.py:1547` | 1 616 |
| `data/gen/shapes3d/a7_redundant.json` | `gen_request.load_registered_control_requests` (`:1562`) — named by `query_sets.json['sets']['a7_redundant']['file']` | 69 217 |
| `data/gen/shapes3d/factor_index.npz` | `gen_splits.load_index` — the `base` set declares `derived_from: factor_index.npz` | 3 844 150 |

⚠ Without these three (3.7 MB), `require_registered_controls` **raises** on the far side and the refusal
column cannot be scored — discovered at hour 6, not hour 0.
`c_invalid.json` is named by *two* real loaders and is shipped **once** (first row wins, tested).

---

## 3. ACCEPTANCE #6 — the dataset is not bundled, and its checksum is asserted

`3dshapes.h5` is **267 573 662 B (255 MiB)** and re-downloadable. **I verified the declared sha256
against the file on this machine** — it matches exactly:

```
$ shasum -a 256 .claude/data/gen/shapes3d/3dshapes.h5
0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785   ✔ == the value in the task brief
```

`README-HUB-BUNDLE.md` (generated into every bundle) gives the far-side procedure:
`chlu data shapes3d --download`, then `sha256sum "$CHLU_GEN_DATA_ROOT/shapes3d/3dshapes.h5"` **must**
equal that value — a different sha is a different grid, and every request index in the C set is then
wrong. ⭐ Most hub work needs no image blob at all: the feature caches are pre-encoded.

---

## 4. ⭐ ACCEPTANCE #4 — the WSL2 finding, stated plainly. **The brief's premise does not hold.**

**⛔ "JAX HAS NO NATIVE WINDOWS SUPPORT" is false as measured 2026-09-16.** From the PyPI JSON API
(`logs/wsl2_jax_platform_evidence.log`):

| jaxlib | platform wheel tags | `win_amd64`? | pythons |
|---|---|---|---|
| **0.6.2** (this repo, py<3.11) | macosx_11_0_arm64, manylinux2014_{aarch64,x86_64}, **win_amd64** | **YES** | cp310–cp313 |
| **0.9.0** (this repo, py≥3.11) | macosx_11_0_arm64, manylinux_2_27_{aarch64,x86_64}, **win_amd64** | **YES** | cp311–cp314 |
| 0.10.2 / 0.11.0 / 0.11.1 | same | **YES** | cp311+/cp312+ |

`jax` itself is a pure-Python wheel (`any`). ⇒ **`pip install jax jaxlib` on native Windows CPU resolves
and installs** for exactly the versions `uv.lock` pins. A day spent discovering "it does not work" would
have been a day spent discovering the opposite.

**⭐ I still recommend WSL2 — on grounds I could test:**
1. **The tooling is bash, end to end.** 33 `.sh` scripts; **zero** `.ps1`/`.bat`/`.cmd` exist anywhere in
   the repo. `bundle_hub_workspace.sh`, `verify_hub_workspace.sh`, `env_hub.sh`, all of `scripts/cloud/`
   (the hub's **default** compute per its profile §4) and all of `scripts/csf3/` are bash, and use
   `mktemp -t`, `shasum`/`sha256sum`, `rsync`, `find -print`, POSIX paths. There is **no** native-Windows
   entry point to the workspace, and writing one is not this spoke's job.
2. **GPU is Linux-only by declaration** — `pyproject.toml:62`: `cuda = ["jax[cuda12]; sys_platform == 'linux'"]`.
3. Confirmed POSIX-only imports in `chlu/training/train_cluformer.py` and across `scripts/`.

**⛔ NOT RUN — and I could not run it:** I have no Windows machine and no WSL2. **Untested:**
`wsl --install`; Zed's WSL2 remote attach; `uv sync` / `pip install -e .` resolution under WSL2; whether
`/mnt/c` vs the ext4 home actually degrades the 1.9 GB `features.npy` mmap (the README's `/mnt/c` warning
is **received wisdom, ⛔ not measured here**). ⚠ The first thing to do on that machine is run
`verify_hub_workspace.sh` and read `WORKSPACE-READY.json` — it records `platform.system/release/machine`
and every `CHLU_*`, so the first far-side run is self-documenting.

---

## 5. ⭐ ACCEPTANCE #6 — THE RETURN PATH, documented and demonstrated (safe direction)

### 5a. Results — `--return-pack` there, `--return-ingest` here
`logs/return_roundtrip.log`. ⚠ A result that is not synced back does not exist.

```
FAR SIDE : scripts/bundle_hub_workspace.sh --return-pack <hub-slug> --return-out ~/<slug>.return.tar.gz
           -> packed 4 file(s), 205 bytes; sha256 ea9113c9…46470
              (LOG.md, PREREG-hub-smoke-bundle-1.md, runs/results.csv, <slug>.md)
HERE     : scripts/bundle_hub_workspace.sh --return-ingest <tar> [--dest DIR]
           -> [ingest] audit ok: 4 file(s), 205 bytes, hub=hub-smoke-bundle
              ingested -> …/return-demo/hub-smoke-bundle
SAME TAR : -> ⚠ already exists — ⛔ NOT overwriting.
              diverting the whole return to …/hub-smoke-bundle.returned-20260916T114952Z
              ⭐ DIVERTED: the pre-existing … was left untouched. Reconcile by hand.
```
The first result's `LOG.md` was **byte-identical afterwards** (sha re-checked). ⛔ **It is structurally
incapable of overwriting**, and every guard is tested:
- every member path must be **relative**, `..`-free, not a symlink, and prefixed with `<hub-slug>`;
- every sha must match `RETURN_MANIFEST.json`;
- **a member in the tar that the manifest does not declare aborts the whole ingest** — ⛔ nothing is
  placed, not even the good files;
- an existing destination diverts the **entire** set to `<slug>.returned-<utc>/` and says so.
- The verifier's own write probe is **excluded by name** (⛔ never by a blanket dotfile filter, which
  would silently drop a hub's own `.PREREG`). ⚠ Found by observation: timestamped probes were
  accumulating inside the hub's results directory and riding home in the tar. Fixed: one fixed path.

### 5b. ⛔ Code — and the form that silently fails
A hub **may modify the CLU**; that work is on its branch and `git push` is forbidden to it.
**Measured failure:**

```
FAR SIDE : git bundle create f main..agent/autonomous-hub/<slug>      # 671 bytes — looks fine
HERE     : git fetch f 'refs/heads/…:refs/remotes/farside/<slug>'
           ⛔ error: Repository lacks these prerequisite commits: cba9732…
```
A range bundle is **thin**; its prerequisite is the workspace bundle's own synthetic base commit, which
this machine has never seen. ⚠ It fails **on arrival** — after the hub is done and the second machine may
be gone. **The fix, measured and demonstrated:**

```
FAR SIDE : git bundle create f agent/autonomous-hub/<slug>   # 3 140 878 B, "records a complete history"
HERE     : git fetch f 'refs/heads/…:refs/remotes/farside/<slug>'   ✔
           git diff --stat <pin> refs/remotes/farside/<slug>
             chlu/core/gen_store.py | 2 +   (+ the 3 declared tooling files)
           git merge-base --is-ancestor refs/remotes/farside/<slug> main -> NOT an ancestor
```
⛔ `refs/remotes/farside/*` shares no ancestor with `main`, so it **cannot fast-forward anything**: worst
case an unreferenced ref, never a lost commit. The Hub reviews, then cherry-picks or merges deliberately.
*(The demo ref, worktree and branch were created and removed; `git for-each-ref refs/remotes/` is back to
`origin`/`clu`/`clu-dev` only.)*

---

## 6. ⭐⭐ The bundle is a git repository — and the code IS the pin, provably

⚠ **`git archive` gives a plain directory.** The hub's profile §0.2 makes a **worktree** its sandbox and
says *"The Hub merges your branch after review."* In a plain directory a hub can create **no branch, no
worktree, no commit** — its one isolation mechanism does not exist, and its CLU changes have no way home.
So the bundle is `git init`'d, in **two commits on purpose**:

```
root commit  = the pin's tree, VERBATIM
  bundle : 12831fe1be17d04da027dcbb6e45b4dff6f1ef95
  source : git rev-parse ae907466…^{tree}  ->  12831fe1be17d04da027dcbb6e45b4dff6f1ef95   ✔ identical
HEAD         = + the declared tooling overlay, and nothing else:
  git diff --stat $(git rev-list --max-parents=0 HEAD) HEAD
    scripts/bundle_hub_workspace.sh | 942 ++++
    scripts/verify_hub_workspace.sh | 484 ++++
    tests/test_hub_workspace.py     | 719 ++++
    3 files changed, 2145 insertions(+)
```
⭐ *"The code is the pin"* is now **one `rev-parse`**, recorded in the manifest
(`code.pin_tree`, `code.bundle_base_tree_matches_pin: true`) and **asserted by the verifier** — which
refuses `ready` if the root tree differs or if there is no repository at all.

**Why an overlay at all, and why it is safe:** the pin **predates** this tooling, so a bundle built from
it alone could neither verify itself nor return a result. The three files are pure plumbing: ⛔ none is
imported by `chlu`, ⛔ none is part of the scoring harness the charter §A.4 freezes (classifier, splits,
rival rows, byte ledger, bypass assertions). It is **declared in the manifest and visible in one git
command** — ⛔ not silent. A worktree was created and removed in the bundle to prove §0.2 is satisfiable.

⚠ **Ordering hazard, found and fixed:** a rebuild over a previous build's overlay would fold the overlay
into the base commit and "the code is the pin" would quietly stop being true. The base commit now
excludes exactly the tooling paths **the pin does not contain** (computed from the pin, ⛔ not hardcoded),
with the tree assertion as the backstop. Tested.

---

## 7. ACCEPTANCE #3 — the checker **demonstrated failing**, three ways

⛔ A checker that cannot fail proves nothing. All three run against the **real** 1.967 GiB bundle
(`logs/verify_fail_*.log`), and each is also a test.

**(a) A file that never arrived** — `classifier_gate.json` moved aside (⛔ moved, never deleted):
```
status=NOT-READY  checksum_match=False  n_missing=1  n_checksum_mismatch=0  n_load_failed=0   rc=1
  .claude/outputs/gen-external-classifier/classifier_gate.json : MISSING on the far side
```

**(b) ⭐⭐ A BYTE-PERFECT file its loader rejects** — `store_plan.json` replaced with valid JSON carrying
the right `schema` string, **and the manifest's sha256 updated to match**, so the checksum leg passes
cleanly:
```
status=NOT-READY  checksum_match=TRUE  n_checksum_mismatch=0  loaders_all_opened=False  n_load_failed=1
  store_plan.json : checksum_match=True loaded=False | KeyError: 'n_modes'
```
⭐ **`checksum_match` stayed `True` while the verdict went red** — exactly the failure a checksum alone
cannot see, and the `verify_artefacts.sh` lesson honoured.

**(c) A document that lost its own anchor** — a charter truncated to its title line, manifest sha again
updated to match:
```
status=NOT-READY  checksum_match=TRUE  n_checksum_mismatch=0  n_load_failed=1               rc=1
  autonomous-hub-charter.md : checksum_match=True loaded=False
    ValueError: required anchor(s) absent: 'ae907466ba2b5fbec124e8995cf4d7c32a1ab30e',
                                           'MATCHED-BYTES LOOKUP', 'core philosophy'
```
⚠ A charter truncated above the pin checksums perfectly and **pins nothing**. Each document is checked
for its own anchors: the charter for the **pin**, §C's floor and §B's philosophy; the profile for §0,
"YOU NEVER DELETE" and pre-registration; the protocol for the dial declaration; the handover for §7.

*(The three demos ran with `SKIP_TEST=1` so the verdict isolates the file-level check. An earlier pass
without it also returned rc=1, with the targeted test additionally red on the missing file —
`1 failed, 6 passed, 15 skipped`.)*

**(d) also asserted, in tests:** an import resolving **outside** `$CHLU_ROOT`; and a bundle with **no git
repository** (`--no-git`) → `NOT-READY`, `worktree_possible: false`.

**Repair:** each victim was restored in place and the re-verify returned `status=ready`,
`files=55 missing=0 mismatched=0 unopenable=0`. ⚠ ⛔ A *rebuild* no longer repairs the **code** half —
see §8.8; it still re-checks and repairs every `.claude` file.

---

## 8. ⚠ FINDINGS THE HUB SHOULD ACT ON

1. **⛔ The brief's JAX/Windows premise is wrong today** (§4). Re-decide WSL2 on the shell-tooling and
   GPU-policy grounds, ⛔ not on "JAX won't install".
2. **⚠ A charter-frozen artefact changed on disk while I was bundling it.** `classifier_gate.json`:
   `1297 B / fbd2c55a…` at 13:02, `1296 B / 88079d5f…` on re-derivation minutes later; mtime `Sep 16
   13:02`. Content still reads `"gate": "PASS"`. ⛔ I did not write it and cannot say who did (§A.4 calls
   this file frozen and read-only for hubs). ⭐ **This is the argument for the manifest**: the bundle
   records exactly which bytes shipped, so the two machines can be compared later. **Recommend the Hub
   establish which process rewrites it**, and build the three hubs' bundles from one stable snapshot —
   the "stable HEAD" discipline, applied to artefacts.
3. **⭐ 93 % of the bundle is seed 0's `scope="full"` cache** (§2). A `scope="used"` re-encode would take
   it to ~0.13 GiB, but that is a re-encode under the hazard gate. **The Hub's call, not mine.**
4. **⛔ `git bundle create main..<branch>` fails only on arrival** (§5b). Tell the hubs the whole-branch
   form, or their CLU changes do not come home.
5. **46, not 45, task files hardcode the path** — `gen-hub-replication.md` was written after the count.
   ⛔ None is reachable by a hub (the profile forbids the archive; the bundle ships one task file), so I
   left all 46 alone; rewriting live spoke charters in bulk is how a base branch changes silently.
6. **⛔ No `.claude/tasks/hub-*.md` exists at the pin.** The bundle **refuses** to build without one:
   *"A hub with no charter of its own is not a hub — it is ten unsupervised hours with no question."*
   The acceptance run used a clearly-labelled stand-in via `--hub-task` (§10). **The Hub must write
   `hub-generation-table.md` / `hub-composition.md` / `hub-refusal.md` / `hub-deletion.md` before any
   real bundle can be built.** Then: `scripts/bundle_hub_workspace.sh --hub <slug> --out <dir> --tar`.
7. **⛔ A rebuild used to overwrite a hub's own code — found and fixed.** The builder re-extracted
   `git archive <pin>` over the existing bundle on every run. Once a hub has edits in `chlu/**` — which
   it is *expected* to have, §1.1 says it may modify the CLU — that silently replaced them with the pin's
   versions, leaving no diff to notice. ⭐ The code is now extracted **only into a fresh bundle**; the
   `.claude` set is still re-checked and repaired every run, and the code is under git so `git status`
   in the bundle shows any corruption. **⚠ To rebuild the code clean, delete the bundle.** Tested.
8. **⚠ I forked my own test run.** The recursion guard was implicit ("a bundle has no `.git`, so the
   round-trip tests skip there") and stopped holding the moment I made the bundle a git repo: verify →
   pytest → verify forked and **28 processes had to be killed by hand**. Now an explicit exported
   `$CHLU_WORKSPACE_VERIFY_DEPTH`, with a hard refusal at depth ≥ 2. ⭐ Recorded because the class of
   error — *a guard that works by side effect* — is the one worth remembering, not this instance.

---

## 9. ACCEPTANCE #5 — the path-portability diff (copy-able, for the Hub to apply)

**`.claude/outputs/gen-hub-workspace-bundle/path-portability.diff`** — three one-line edits in two files.

⭐⭐ **The charter's diff is EMPTY.** `.claude/autonomous-hub-charter.md` contains **zero** absolute paths
(`grep -n '/Users/\|/home/\|~/\|Desktop'` → no match). The allow-listed output path the brief attributes
to it is in `.claude/agents/autonomous-hub.md` §0.3 — **mine**, and already fixed. ⛔ I did not touch the
charter.

| file | line | change |
|---|---|---|
| `.claude/AGENT_PROTOCOL.md` | 37 | `/Users/…/.venv/bin/python` → `"$CHLU_ROOT/.venv/bin/python"`, **plus** a warning that on a second machine the venv is *that machine's own* |
| `.claude/handover_context.md` | 28 | **Repo:** `/Users/…/CHLU` → `$CHLU_ROOT` (default stated) |
| `.claude/handover_context.md` | 507 | "in the main repo dir, `/Users/…/CHLU`" → `$CHLU_ROOT` |

**What I DID change in the file I own** (`.claude/agents/autonomous-hub.md`, ⭐ **paths only — ⛔ not one
rule**): §0.2 opens with `$CHLU_ROOT` as the named workspace root (default = this machine, so ⛔ nothing
changes here) and *"⚠ if `$CHLU_ROOT` is unset, stop and say so"*; the worktree and sibling-checkout paths
become `$CHLU_ROOT-<slug>` / `$CHLU_ROOT-merge`; §0.3's allow-list becomes
`$CHLU_HUB_OUTPUTS/<hub-slug>/**`, with its expansion spelled out on both machines.

---

## 10. How I verified — commands and observed output

| # | command | result |
|---|---|---|
| 1 | `bash scripts/bundle_hub_workspace.sh --src … --hub hub-smoke-bundle --hub-task … --out /tmp/chlu-hub-test` | rc=0, 55+416 files, 1.967 GiB, base tree == pin tree, **11.6 s wall** |
| 2 | `… --out /tmp/chlu-hub-test --dry-run` | rc=0, full plan, **nothing written** |
| 3 | `… --hub hub-generation-table` (no task file) | rc=2, *"no task file at …"* — the refusal |
| 4 | `cd /tmp/chlu-hub-test && source ./env_hub.sh && bash scripts/verify_hub_workspace.sh` | **rc=0, `status=ready`**, 55/0/0/0, `import_under_root=True`, `matches_pin=True` |
| 5 | `hub_boot_smoke.py hub-smoke-bundle` (from the bundle root) | **`HUB BOOT OK`**, K=5/n_modes=64/D=2048 |
| 6 | failure demos (a)(b)(c) | rc=1 each; **(b) and (c) with `checksum_match=True`** |
| 7 | restore + re-verify | `status=ready`, 55/0/0/0 |
| 8 | return pack → ingest → ingest again | ingested, then **diverted**; first result byte-identical |
| 9 | `git bundle` code return | `main..` **fails**; whole-branch 3 140 878 B **succeeds** |
| 10 | `uv`-venv `pytest -q tests/test_hub_workspace.py tests/test_stage_artefacts.py` | see below |
| 11 | `ruff check tests/test_hub_workspace.py` | **All checks passed!** |

**Targeted tests — `PASSED / SKIPPED / FAILED`:**

| run | passed | skipped | failed |
|---|---|---|---|
| `tests/test_hub_workspace.py` + `tests/test_stage_artefacts.py` (this machine, main venv) | **40** | **2** | **0** |
| `tests/test_hub_workspace.py` nested **inside the bundle** at `/tmp/chlu-hub-test` | **7** | **19** | **0** |

The 2 skips on this machine are the two environment-conditional tests (`CHLU_ROOT` unset outside a
bundle; no live `BUNDLE_MANIFEST.json`). The 19 skips inside the bundle are `@needs_bundle` — the
recursion guard (§8.7); the 7 that run there are exactly the ones that matter on the far side: both
scripts parse, the verifier names no machine, no GPU/cluster reference, `$CHLU_GEN_DATA_ROOT` follows its
env var, **`chlu` is imported from `$CHLU_ROOT`**, and the live manifest is internally consistent.

⛔ **`ruff format` NOT applied**: 109 of 145 files in `tests/` are already non-conforming, so the new file
matches the prevailing style. `ruff check` is clean.

**Cost line: CPU only, laptop only. ⛔ No GPU, no cluster, no cloud, $0.00.** ~4 h wall, of which the
dominant costs were five full `pytest` passes (2–4 min each; the round-trip tests build real bundles) and
four 2 GB bundle builds at ~8–12 s each (APFS, same volume). JAX import was warm at ~3.5 s, ⛔ not the
20 min §4 of the protocol warns about. **⚠ One self-inflicted cost:** the forked test run (§8.7) burned
~10 min and 28 processes before I killed it.

---

## Git footprint

- **Worktree:** `/Users/user/Desktop/CHLU-bundle` (left in place for review). **Branch:**
  `agent/experiment-engineer/gen-hub-workspace-bundle`, off the **pin** `ae907466…1ab30e`.
- **Commits** (verified visible from the MAIN repo — `git -C /Users/user/Desktop/CHLU log --oneline
  ae907466..agent/experiment-engineer/gen-hub-workspace-bundle`, per protocol §3.2):

| hash | subject |
|---|---|
| `0e604ed` | a portable hub workspace: `$CHLU_ROOT`, and a bundle that proves itself |
| `6e47ce9` | tests: the bundle is the pin, the checker can fail, the return cannot overwrite |
| `c5650c2` | the code return path: bundle the branch WHOLE, not `main..branch` |
| `29bbc60` | never re-extract the code over a bundle a hub may be working in |

- **Tracked files touched — 3, all new, all owned:** `scripts/bundle_hub_workspace.sh`,
  `scripts/verify_hub_workspace.sh`, `tests/test_hub_workspace.py`.
- **Untracked (`.claude/`) touched:** `.claude/agents/autonomous-hub.md` (mine, paths only) and
  `.claude/outputs/gen-hub-workspace-bundle/**` (mine).
- ⛔ **NOT touched:** `.claude/autonomous-hub-charter.md` · `scripts/csf3/**` · `chlu/**` ·
  `chlu/config.py` · `.gitignore` · `.claude/tasks/**` (read only) · any other slug's outputs.
- **No rebase needed** — the branch base (the pin) has not moved. ⛔ Not pushed. No conflicts.
- **Left on disk for review:** `/tmp/chlu-hub-test` (2.0 GB, the verified bundle) and
  `/tmp/hub-smoke-bundle.return.tar.gz`. ⚠ `/tmp` is volatile; rebuild in 12 s if it is gone.
- **Removed (all mine, all created this session):** the demo worktree `/tmp/chlu-hub-wt`, its branch, and
  `refs/remotes/farside/hub-smoke-bundle`. ⭐ `git for-each-ref refs/remotes/` is back to
  `origin`/`clu`/`clu-dev` only.

## Output-directory contents
```
.claude/outputs/gen-hub-workspace-bundle/
  path-portability.diff          ⭐ §9 — for the Hub to apply
  hub-smoke-bundle.md            the STAND-IN task file (⛔ labelled as such in its first line)
  hub_boot_smoke.py              the six-step hub boot, run from the bundle root
  stage_manifest_probe.json      the first artefact-set derivation (13:02) — kept as the §8.2 evidence
  return-demo/                   §5a: the clean landing + the diverted second arrival
  return-demo.superseded-…/  x2  earlier runs, superseded (⛔ renamed, not deleted)
  logs/  bundle_build.log · verify_pass.log · verify_fail_missing.log ·
         verify_fail_unloadable.log · verify_fail_anchor.log · hub_boot_smoke.log ·
         return_roundtrip.log · return_code_gitbundle.log ·
         wsl2_jax_platform_evidence.log · pytest_targeted.log
```

## Open questions / follow-ups / risks
1. **Which of the four hub targets go to the second machine?** Three of `hub-generation-table`,
   `hub-composition`, `hub-refusal`, `hub-deletion`. One bundle can carry several (`--hub` repeats) — but
   ⚠ they would then **share one `.claude/outputs` and one git repo**, which is the very collision §0.2
   exists to prevent. ⭐ **Recommend one bundle per hub** (1.967 GiB each, ~6 GB for three) unless the
   Head wants them coordinating.
2. **Does the second machine have `rsync`/`scp` to here, or is transfer by hand?** `--return-pack`
   produces one tar and prints its sha256; nothing in the design assumes a network.
3. **The `scope="full"` cache (§8.3)** — worth a re-encode, or ship 2 GB three times?
4. ⛔ **Untested and unbudgeted:** everything WSL2 (§4). ⭐ The first far-side action should be
   `verify_hub_workspace.sh`; `WORKSPACE-READY.json` records the platform, so that run documents itself.
5. **The `--hub-task` escape hatch is a loaded gun.** It exists only because no `hub-*.md` exists yet.
   ⭐ Once the real charters land, build with `--hub <slug>` alone so the refusal can do its job.

---

## Proposed handover updates (for the Hub)

**§7-CURRENT — new entries:**

- **7.30 [ENV, standing — ⛔ corrects a premise in circulation]** **JAX *does* ship native-Windows CPU
  wheels.** `jaxlib` publishes `win_amd64` for 0.6.2, 0.9.0, 0.10.2, 0.11.x — i.e. every version
  `uv.lock` can resolve; `jax` is pure Python. ⛔ "JAX has no Windows support" is **false as of
  2026-09-16** and must not be repeated. ⭐ WSL2 is still the recommendation for the second machine, on
  two testable grounds instead: **(i)** the entire workspace toolchain is bash — 33 `.sh`, **zero**
  `.ps1`/`.bat`/`.cmd`, including `scripts/cloud/` which the hub profile §4 makes the *default* compute;
  **(ii)** `pyproject.toml:62` gates the GPU extra `sys_platform == 'linux'`. ⛔ WSL2 itself is **NOT RUN**
  — no Windows machine exists here. Evidence: `gen-hub-workspace-bundle` §4.

- **7.31 [OPEN, provenance — needs an owner]** **A charter-frozen artefact was rewritten on disk during a
  session.** `.claude/outputs/gen-external-classifier/classifier_gate.json` went `1297 B / fbd2c55a…` →
  `1296 B / 88079d5f…`, mtime `2026-09-16 13:02`, content still `"gate": "PASS"`. §A.4 names this file a
  frozen asset, read-only for hubs. ⛔ The rewriting process is unidentified. ⭐ Consequence for the
  second machine: **build all hub bundles from one stable snapshot and compare
  `BUNDLE_MANIFEST.json` shas across them** — the "suite runs need a stable HEAD" discipline, applied to
  artefacts rather than commits. Evidence: `gen-hub-workspace-bundle` §8.2 + `stage_manifest_probe.json`.

- **7.32 [ENV, standing — the portability trap]** **`python -c` puts the CWD at `sys.path[0]`, ahead of
  `PYTHONPATH`.** Measured: with `PYTHONPATH=<worktree>` but cwd in the primary checkout, `chlu` resolves
  to the **primary checkout**. ⚠ Every `PYTHONPATH=<worktree> …/.venv/bin/python` invocation in
  `AGENT_PROTOCOL.md` §4 is therefore **cwd-dependent**, and a portability or bit-identity check run from
  the wrong directory silently tests the wrong tree. ⭐ Assert `chlu.__file__` before trusting any such
  run; `verify_hub_workspace.sh` does, and refuses `ready` otherwise.

- **7.33 [infra, new — the second machine]** **A portable hub workspace exists.**
  `scripts/bundle_hub_workspace.sh` (build · `--return-pack` · `--return-ingest`) and
  `scripts/verify_hub_workspace.sh`, on branch `agent/experiment-engineer/gen-hub-workspace-bundle` off
  the pin. **1.967 GiB** (55 `.claude` files + 416 code files), artefact set **derived by invoking**
  `scripts/csf3/stage_artefacts.sh`. `$CHLU_ROOT` is the single source of truth, defaulting to
  `/Users/user/Desktop/CHLU` — ⛔ nothing on this machine changes. The bundle is a **git repository whose
  root commit carries the pin's tree verbatim** (`12831fe1…`), so the hub's mandatory worktree exists and
  "the code is the pin" is one `rev-parse`. ⛔ **No `.claude/tasks/hub-*.md` exists yet and the builder
  refuses without one** — write the three charters first. Return path: results by `--return-pack` (⛔
  cannot overwrite; diverts to `<slug>.returned-<utc>/`), code by **whole-branch** `git bundle` (⛔ *not*
  `main..<branch>` — that fails on arrival).

**§0 / agent roster:** `.claude/agents/autonomous-hub.md` §0.2/§0.3 now speak in `$CHLU_ROOT` and
`$CHLU_HUB_OUTPUTS`. ⛔ No rule changed; the macOS defaults are unchanged and stated inline.

**Apply when convenient:** `.claude/outputs/gen-hub-workspace-bundle/path-portability.diff` — three
one-line edits (`AGENT_PROTOCOL.md:37`, `handover_context.md:28`, `:507`). ⭐ **The charter needs no
change**; its diff is empty.
