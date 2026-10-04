# SPOKE E — AppleDouble cleanup, then the two untested git legs

Host `FORGIS_14` · Windows 11 Pro 26200 · Git Bash (MSYS2) · GNU tar 1.35 · gzip 1.14
Workspace `C:/Users/PratikJawahar/chlu/chlu-hub-bundle` · run 2026-09-16

---

## ⭐ THE TWO VERDICTS

**(1) Is `git fsck` clean now? — YES.** `git fsck --no-progress` produces **zero output, exit 0**.
Before the cleanup it emitted 22 error lines and exited **76**. All 588 AppleDouble files are gone,
all 57 sha256-pinned manifest files are byte-identical to how I found them, and the pin tree is
still `12831fe1be17d04da027dcbb6e45b4dff6f1ef95`.

**(2) Can a hub create a worktree and get its results home on this machine? — YES, both legs.**
`git worktree add` works (NTFS `.git`-file pointer mechanism confirmed, 419 files checked out,
33/33 `.sh` files LF-clean, worktree status clean), and both return legs work:
`--return-pack` produced a well-formed `RETURN_MANIFEST.json` whose 6 sha256s I verified
independently, and `git bundle create main` → 3.0 MB, `bundle verify` = *"records a complete
history"*, and I **cloned from it** to prove it arrives with the correct root tree.

⭐ **Third result (§3): `anchors_checked` is `0` for ALL FOUR docs that have registered anchors.**
The hub's analysis is correct and is now measured, not argued. Details and proof in §3.

---

# §1 — The cleanup

## What I did

### Step 1 — integrity snapshot BEFORE

Computed sha256 of all 57 `BUNDLE_MANIFEST.json` files into `/tmp/spokeE/before.sha`.

| check | result |
|---|---|
| manifest files present | 57 / 57 |
| sha256 matches manifest | **57 / 57 OK**, 0 mismatch, 0 missing |
| `git rev-parse $(git rev-list --max-parents=0 HEAD)^{tree}` | `12831fe1be17d04da027dcbb6e45b4dff6f1ef95` ✅ == pin |

`git status --porcelain` before: **480 rows** — 36 `M` (mode-only) + 444 `??`.
Every git invocation also printed to stderr:

```
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
```

`git fsck --no-progress` BEFORE — **exit 76**, full output:

```
error: refs/._heads: badRefName: invalid refname format
error: refs/._heads: badRefContent:
error: refs/._tags: badRefName: invalid refname format
error: refs/._tags: badRefContent:
error: refs/heads/._main: badRefName: invalid refname format
error: refs/heads/._main: badRefContent:
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
bad sha1 file: .git/objects/21/._72c509348ccf0db2a48281ba0200c787b1ffba
bad sha1 file: .git/objects/5c/._8bfece366bdbefb7b43625d90d38974c4e5c85
bad sha1 file: .git/objects/67/._67a6040cddac17c99b8789ad3fa3e127585914
bad sha1 file: .git/objects/6f/._7c61504138d64295d348c464579aa47ec45192
bad sha1 file: .git/objects/70/._1e437e8ab4575966dada7d8afd605ed8393cea
bad sha1 file: .git/objects/86/._dfd811ddc84a1c4ed74a88fa808bbfc123fd7b
bad sha1 file: .git/objects/8c/._44661f227ef242236779c5d24ac7d1f7523ac3
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
error: packfile .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.pack index not opened
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
error: unable to load rev-index for pack '.git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.pack'
```

### Step 2 — enumerate to a list

`find . -name '._*' -type f | sed 's#^\./##' | sort > /tmp/spokeE/appledouble.list`

**count = 588** — matches the brief exactly. Distribution matches the brief exactly:

| dir | n |
|---|---|
| `chlu/` | 232 |
| `tests/` | 145 |
| `.claude/` | 90 |
| `.git/` | 54 |
| `scripts/` | 48 |
| `<root>` | 16 |
| `projects/` | 2 |
| `logs/` | 1 |

Size histogram — **582 × 163 bytes, 6 × 263 bytes**, exactly as the brief's corroborating evidence
predicted. Total ≈ 96 KB on disk.

### Step 3 — HARD GUARD (re-verified myself, two independent ways)

**Guard A — `grep -Ff /tmp/spokeE/appledouble.list BUNDLE_MANIFEST.json` → zero hits (exit 1).**

**Guard B — structured set comparison in Python:**

```
manifest paths: 57 appledouble: 588
exact path intersection: EMPTY
manifest entries with ._ basename: NONE
manifest entries with any ._ path component: NONE
appledouble paths that are ancestors of manifest paths: NONE
quarantined file in manifest: True
quarantined file in delete list: False
list entries NOT matching '._' basename rule: NONE
```

⭐ Guard B also checked two things the brief's `grep` recipe alone would not catch: that no manifest
path has a `._` **directory component**, and that no listed path is an **ancestor directory** of a
manifest path. Both NONE.

⛔ **On the quarantined pinned file specifically.** There are two adjacent names and only one is a
sidecar. I checked them side by side before deleting anything:

```
-rw-r--r--  163      .claude/outputs/gen-arms-a7-wiring/._refusal_scores.json   <- AppleDouble, DELETED
-rw-r--r--  2660729  .claude/outputs/gen-arms-a7-wiring/refusal_scores.json     <- PINNED, UNTOUCHED
```

The pinned file's sha256 is identical in `before.sha` and `final.sha`. It was never moved, renamed
or opened for writing.

### Step 4 — deletion, in two separated passes

Both passes ran through a loop that **re-asserts the guard on every single path immediately before
`rm`** (basename must start with `._`; pass A additionally requires the `.git/` prefix, pass B
refuses it). Nothing was piped from `find` into a mutation.

**Pass A — the 54 under `.git/`: deleted 54, 0 failures.**
`git fsck` immediately afterwards: **exit 0, no output.** The tool was fixed by pass A alone.

**Pass B — the remaining 534: deleted 534, 0 failures.**

**Total deleted: 588. Remaining `._*` files anywhere in the bundle: 0.**

## ⭐ Full accounting of everything deleted OUTSIDE `.git/` (534 files)

The complete path list is saved beside this report at
**`_hub_scratch/reports/_E-deleted-non-git.txt`** (534 lines). Grouped by containing directory:

| directory | n | | directory | n |
|---|---|---|---|---|
| `<root>` | 16 | | `chlu/experiments/` | 82 |
| `logs/` | 1 | | `chlu/experiments/kt/` | 5 |
| `projects/` | 2 | | `chlu/training/` | 9 |
| `scripts/` | 10 | | `chlu/utils/` | 4 |
| `scripts/cafe/` | 1 | | `chlu/` | 12 |
| `scripts/cloud/` | 18 | | `chlu/cli/` | 7 |
| `scripts/csf3/` | 19 | | `chlu/core/` | 45 |
| `tests/` | 145 | | `chlu/data/` | 17 |
| `chlu/eval/` | 21 | | `chlu/data/industrial/` | 7 |
| `chlu/eval/_tsb_vendor/` | 6 | | `chlu/eval/_tsb_vendor/affiliation/` | 6 |
| `chlu/eval/rivals/` | 11 | | **`.claude/` (all depths)** | **90** |

### The 90 under `.claude/`, listed in full as the Head asked

```
.claude/._AGENT_PROTOCOL.md
.claude/._agents
.claude/._autonomous-hub-charter.md
.claude/._data
.claude/._handover_context.md
.claude/._outputs
.claude/._tasks
.claude/agents/._autonomous-hub.md
.claude/data/._gen
.claude/data/gen/._shapes3d
.claude/data/gen/shapes3d/._a7_redundant.json
.claude/data/gen/shapes3d/._c_invalid.json
.claude/data/gen/shapes3d/._factor_index.npz
.claude/data/gen/shapes3d/._query_sets.json
.claude/outputs/._gen-arms-a7-wiring
.claude/outputs/._gen-decoder-lift
.claude/outputs/._gen-external-classifier
.claude/outputs/._gen-feature-bridge
.claude/outputs/._gen-read-compose
.claude/outputs/._gen-store-grouping
.claude/outputs/gen-arms-a7-wiring/._refusal_scores.json
.claude/outputs/gen-arms-a7-wiring/._runs
.claude/outputs/gen-arms-a7-wiring/runs/._historical_688128
.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/._seed_0
.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/._seed_1
.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/._seed_2
.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/seed_0/._baselines_manifest.json
.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/seed_1/._baselines_manifest.json
.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/seed_2/._baselines_manifest.json
.claude/outputs/gen-decoder-lift/._winner
.claude/outputs/gen-decoder-lift/winner/._decoder_v1_s0.pkl
.claude/outputs/gen-decoder-lift/winner/._decoder_v1_s1.pkl
.claude/outputs/gen-decoder-lift/winner/._decoder_v1_s2.pkl
.claude/outputs/gen-decoder-lift/winner/._winner_hashes.json
.claude/outputs/gen-external-classifier/._classifier_gate.json
.claude/outputs/gen-external-classifier/._seed_0
.claude/outputs/gen-external-classifier/._seed_1
.claude/outputs/gen-external-classifier/._seed_2
.claude/outputs/gen-external-classifier/seed_0/._classifier_manifest.json
.claude/outputs/gen-external-classifier/seed_0/._external_classifier.pkl
.claude/outputs/gen-external-classifier/seed_1/._classifier_manifest.json
.claude/outputs/gen-external-classifier/seed_1/._external_classifier.pkl
.claude/outputs/gen-external-classifier/seed_2/._classifier_manifest.json
.claude/outputs/gen-external-classifier/seed_2/._external_classifier.pkl
.claude/outputs/gen-feature-bridge/._runs
.claude/outputs/gen-feature-bridge/runs/._shapes3d_recon
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/._seed_0
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/._seed_1
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/._seed_2
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/._bridge_manifest.json
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/._decoder.pkl
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/._encoder.pkl
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/._feature_cache.json
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/._feature_rows.npy
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/._features.npy
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/._probe.pkl
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/._bridge_manifest.json
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/._decoder.pkl
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/._encoder.pkl
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/._feature_cache.json
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/._feature_rows.npy
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/._features.npy
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/._probe.pkl
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/._bridge_manifest.json
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/._decoder.pkl
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/._encoder.pkl
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/._feature_cache.json
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/._feature_rows.npy
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/._features.npy
.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/._probe.pkl
.claude/outputs/gen-read-compose/._runs
.claude/outputs/gen-read-compose/runs/._shapes3d_recon
.claude/outputs/gen-read-compose/runs/shapes3d_recon/._seed_0
.claude/outputs/gen-read-compose/runs/shapes3d_recon/._seed_1
.claude/outputs/gen-read-compose/runs/shapes3d_recon/._seed_2
.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_0/._store_s0.pkl
.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_1/._store_s1.pkl
.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_2/._store_s2.pkl
.claude/outputs/gen-store-grouping/._runs
.claude/outputs/gen-store-grouping/runs/._shapes3d_recon
.claude/outputs/gen-store-grouping/runs/shapes3d_recon/._seed_0
.claude/outputs/gen-store-grouping/runs/shapes3d_recon/._seed_1
.claude/outputs/gen-store-grouping/runs/shapes3d_recon/._seed_2
.claude/outputs/gen-store-grouping/runs/shapes3d_recon/._store_plan.json
.claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_0/._store_manifest.json
.claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_1/._store_manifest.json
.claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_2/._store_manifest.json
```

⭐ Note the shape of that list: every entry is a `._<name>` sidecar sitting **next to** a real file or
directory of the same name. Not one of them is itself an artefact. The 16 at `<root>` are the same
pattern (`._.git`, `._BUNDLE_MANIFEST.json`, `._env_hub.sh`, `._chlu`, …).

### Step 5 — PROOF that nothing else changed

```
diff /tmp/spokeE/before.sha /tmp/spokeE/after.sha   ->  EMPTY
```

57/57 still `OK` against the manifest's recorded sha256. 0 mismatch, 0 missing.
Root tree after deletion: `12831fe1be17d04da027dcbb6e45b4dff6f1ef95` ✅ unchanged.

(I re-ran this a **third** time at the very end of all of §1–§3 — see "Final state" below. Still
byte-identical to `before.sha`.)

### Step 6 — the acceptance condition

```
$ git fsck --no-progress
[exit=0]
```

**No output at all.** ⭐ The condition is met.

`git status --porcelain` at this point: **37 rows — 36 mode-only `M`, 1 untracked.**
The brief expected 0 untracked. The single untracked row is **not** an AppleDouble leftover:

```
?? hub-refusal.supplement.tar.gz
```

See open question 1 — someone has placed a **second, 206 MB copy of the supplement tarball inside
the bundle**. It is a distinct file (different inode) from the approved copy at
`/c/Users/PratikJawahar/chlu/hub-refusal.supplement.tar.gz`, not a hardlink. It is outside my
deletion scope (`._*` only) and adjacent to a gated decision I was told to leave alone, so **I did
not touch it**. With that one file discounted, untracked is 0 as predicted.

### The `core.filemode` change

```
git config --local core.filemode false
```

| | before | after |
|---|---|---|
| `git status` total rows | 480 (then 37 post-cleanup) | **1** |
| mode-only `M` rows | **36** | **0** |
| untracked `??` rows | 444 (then 1) | 1 (the supplement tarball) |

Confirmed it changed only the comparison, not the recorded data:

```
$ git ls-files -s scripts/verify_hub_workspace.sh
100755 2172c509348ccf0db2a48281ba0200c787b1ffba 0   scripts/verify_hub_workspace.sh   <- still 100755
$ git rev-parse $(git rev-list --max-parents=0 HEAD)^{tree}
12831fe1be17d04da027dcbb6e45b4dff6f1ef95                                             <- pin unchanged
$ git fsck --no-progress
[exit=0]
```

`core.autocrlf` is still pinned `false` repo-locally — I checked, I did not touch it.

---

# §2 — The two untested legs

## 2a. `git worktree add` — ✅ WORKS

```
$ git worktree add /c/Users/PratikJawahar/chlu/_wt_probe -b probe/windows-worktree-test
Preparing worktree (new branch 'probe/windows-worktree-test')
HEAD is now at 5c8bfec [bundle] the workspace tooling, overlaid on the pin
[exit=0]
```

**No warnings.** Probe placed outside the bundle, as instructed.

**⚠ The NTFS mechanism question — answered directly.** The worktree root's `.git` is a **regular
ASCII file, not a symlink**, so the absence of NTFS symlinks is a non-issue:

```
$ ls -l .git   ->  -rw-r--r-- 77 .git       (not l-------)
$ file .git    ->  ASCII text
$ [ -L .git ]  ->  NOT a symlink
$ cat .git
gitdir: C:/Users/PratikJawahar/chlu/chlu-hub-bundle/.git/worktrees/_wt_probe
```

The admin dir `.git/worktrees/_wt_probe/` (HEAD, ORIG_HEAD, commondir, gitdir, index, logs/, refs/)
was created correctly in the main repo. Git resolves the pointer in both directions:

```
$ git rev-parse --show-toplevel   ->  C:/Users/PratikJawahar/chlu/_wt_probe
$ git rev-parse --abbrev-ref HEAD ->  probe/windows-worktree-test
$ git status --porcelain          ->  0 rows (clean)
```

⭐ That the worktree is **clean with 0 rows** is itself a result: the repo-local `core.filemode
false` from §1 propagates to worktrees, so a hub's sandbox starts clean rather than 36-files dirty.

**Content check:** 419 tracked files, `chlu/` `tests/` `scripts/` all present, 218 `chlu/**/*.py`.

**LF check — I checked all 33 `.sh` files, not a sample:**

```
files containing CR: 0
$ head -1 scripts/verify_hub_workspace.sh | od -c
0000000   # ! / u s r / b i n / e n v   b a s h \n     <- bare LF, no \r
$ bash -n scripts/verify_hub_workspace.sh   ->  OK
```

**Teardown, and repo health after it:**

```
$ git worktree remove /c/Users/PratikJawahar/chlu/_wt_probe   [exit=0]
$ git branch -D probe/windows-worktree-test
Deleted branch probe/windows-worktree-test (was 5c8bfec).
$ git worktree list
C:/Users/PratikJawahar/chlu/chlu-hub-bundle  5c8bfec [main]
$ ls .git/worktrees   ->  gone
$ git fsck --no-progress   ->  [exit=0], no output
$ git branch -a   ->  * main
```

**Still fsck-clean after removal.** No stale admin dir, no leftover probe directory, no extra branch.

## 2b. The return path

### Results leg — `scripts/bundle_hub_workspace.sh --return-pack` — ✅ WORKS

**I read the `--return-pack` section first (lines 157–246).** What I found that mattered:

- **It DOES offer `--dry-run`** (`DRY_RUN=1` at line 139, honoured at line 226). The brief asked me
  to check — the answer is yes, and it exits before creating any tarball.
- `PY_BIN` defaults to **`python3`**, overridable via `$BUNDLE_PYTHON`. ⚠ On this box bare `python3`
  is the base 3.12.10 (adequate here — the manifest step is stdlib `json` only — but I set
  `BUNDLE_PYTHON` to the venv interpreter to match how a hub will actually be launched).
- `CHLU_HUB_OUTPUTS` is **individually overridable** inside `chlu_derive_roots` (line 78ff).
  ⭐ I used that to run the real code path against a **throwaway payload staged outside the bundle**,
  so exercising this leg required **zero writes into `.claude/`** and zero risk to a pinned artefact.
  Slug: `probe-windows-returnpack`. No real hub slug was used at any point.

**The payload I staged** (7 files, deliberately adversarial): a top-level `<slug>.md`; a nested
`runs/shapes3d_recon/seed_0/` three levels deep; a **256 KB binary** (`blob.bin`); a **real 131 KB
artefact copied out of the bundle** (`store_manifest.json`) to exercise sha256 over genuine content;
a **UTF-8 file containing `⭐ ⛔ —`**; a **`.PREREG` dotfile** (must survive); and a
**`.workspace-write-probe.json`** (must be dropped).

**Dry run:**

```
$ bash scripts/bundle_hub_workspace.sh --return-pack probe-windows-returnpack \
       --return-out /c/Users/PratikJawahar/probe.return.tar.gz --dry-run
[bundle] DRY RUN — would pack 6 file(s), 393806 bytes -> /c/Users/PratikJawahar/probe.return.tar.gz
[exit=0]
```

6 of 7 — the write-probe was excluded, the `.PREREG` dotfile was not. **No tarball was created.**

**Real pack:**

```
[bundle] packed 6 file(s), 393806 bytes -> /c/Users/PratikJawahar/probe.return.tar.gz
[bundle] sha256(probe.return.tar.gz) = 09cf28b4a60d70447abf56ba617b6e7f721bf6c65e9348472e56cfc193f7bf56
[exit=0]
-rw-r--r--  278848  /c/Users/PratikJawahar/probe.return.tar.gz
```

**⚠ The `mktemp -t` / two-pass-tar question — answered.** This box has **GNU tar 1.35, not bsdtar**.
The two-pass construction is therefore not *needed* here, but it is **harmless and correct** under
GNU tar: `tar -cf` then `tar -rf` appended `RETURN_MANIFEST.json` cleanly as the last member.
`mktemp -t NAME.XXXXXX` and `mktemp -d -t NAME.XXXXXX` both behave GNU-style under MSYS2
(`/tmp/chlu_probe.OxDnQ8`), the `trap ... EXIT` cleanup fired, and `gzip 1.14` is present.
⭐ **The script's macOS-compatibility workaround is portable to Windows unchanged.** Archive order:

```
probe-windows-returnpack.md
probe-windows-returnpack/.PREREG
probe-windows-returnpack/runs/shapes3d_recon/seed_0/blob.bin
probe-windows-returnpack/runs/shapes3d_recon/seed_0/notes.md
probe-windows-returnpack/runs/shapes3d_recon/seed_0/store_manifest.json
probe-windows-returnpack/summary.json
RETURN_MANIFEST.json
```

**Is the manifest well-formed?** Yes — valid JSON, `schema: chlu-hub-return/1`, all required keys,
paths sorted, forward-slash separators (⭐ **no Windows backslash contamination** — unlike the
verifier bug in §3, this code path is clean), and `source_chlu_root` correctly recorded as
`C:/Users/PratikJawahar/chlu/chlu-hub-bundle`, `source_host: FORGIS_14`.

**Do the sha256s verify?** I extracted the tarball to a temp dir and re-hashed every file with an
independent script (I did **not** run `--return-ingest`):

```
schema: chlu-hub-return/1 | slug: probe-windows-returnpack | n_files: 6 | total_bytes: 393806
OK   probe-windows-returnpack.md                                        64      fb4041f83149dfb4
OK   probe-windows-returnpack/.PREREG                                   35      67fd24944a271d08
OK   probe-windows-returnpack/runs/.../blob.bin                         262144  28d08929bcada60d
OK   probe-windows-returnpack/runs/.../notes.md                         36      725f0a92d2d4a257
OK   probe-windows-returnpack/runs/.../store_manifest.json              131506  580dbdfdba2c202a
OK   probe-windows-returnpack/summary.json                              21      6773ea527545de04
--- verified 6 OK, 0 BAD; bytes on disk 393806 vs manifest 393806
in tar but NOT declared: NONE
declared but NOT in tar: NONE
write-probe present in tar: False
.PREREG dotfile preserved: True
```

**6/6 sha256 verify, byte counts exact, no undeclared members, no missing members.** The by-name
exclusion works and the blanket-dotfile hazard the script's comment warns about does not bite.

### Code leg — `git bundle` — ✅ WORKS

Whole branch, **not** a range, exactly as instructed:

```
$ git bundle create /c/Users/PratikJawahar/probe.gitbundle main
[exit=0]
-rw-r--r--  3141469  probe.gitbundle           (3.0 MB)

$ git bundle verify /c/Users/PratikJawahar/probe.gitbundle
C:/Users/PratikJawahar/probe.gitbundle is okay
The bundle contains this ref:
5c8bfece366bdbefb7b43625d90d38974c4e5c85 refs/heads/main
The bundle records a complete history.
The bundle uses this hash algorithm: sha1
[exit=0]
```

⭐ **"The bundle records a complete history"** is the line that matters — no prerequisite, so the
arrival-side failure mode the brief warns about cannot occur. **Size: 3.0 MB (3,141,469 bytes)**,
right at the ~3 MB the brief predicted.

**I went one step further and actually cloned from it**, because `verify` only checks self-consistency:

```
$ git clone -b main probe.gitbundle $TMP/arrived        [exit=0]
$ git rev-parse $(git rev-list --max-parents=0 HEAD)^{tree}
12831fe1be17d04da027dcbb6e45b4dff6f1ef95                 <- the pin, intact after the round trip
$ git rev-parse HEAD  ->  5c8bfece366bdbefb7b43625d90d38974c4e5c85
$ git ls-files | wc -l  ->  419
$ git fsck --no-progress  ->  [exit=0]
```

**The code leg round-trips with the pin tree intact.**

### Probe artefacts — deleted

⭐ **I deleted both probe artefacts and confirmed it.** `probe.return.tar.gz` and `probe.gitbundle`
are gone from `~`; the staged payload dir `_hub_scratch/_E_return_probe/` is gone; the `_wt_probe`
worktree dir is gone; both extraction temp dirs are gone. ⛔ `--return-ingest` was **never run**.

---

# §3 — The verifier re-run and the `anchors_checked` measurement

## Did the cleanup disturb anything? No.

Run verbatim per `CONTEXT.md` §"How to open a working shell", then `bash scripts/verify_hub_workspace.sh`.
**Exit 0.** Fresh `WORKSPACE-READY.json` copied to `_hub_scratch/reports/_E-workspace-ready.json`.

| field | value | required |
|---|---|---|
| `status` | **`ready`** | ✅ |
| `n_checksum_mismatch` | **0** | ✅ |
| `n_missing` | **0** | ✅ |
| `n_load_failed` | **0** | ✅ |
| `code_is_the_pin` | **true** | ✅ |

Supporting: `skip_load: false`, `loaders_all_opened: true`, `checksum_match: true`,
`import_under_root: true`, `hub_output_path_writable`/`write_check.writable: true`,
`targeted_test: "passed: 7 passed, 19 skipped in 25.78s"`, `verified_at: 2026-09-16T17:23:44Z`,
host `FORGIS_14`, platform `Windows 11 / AMD64 / Python 3.12.10`.

⭐ And notably, `git_check` now reads:
`{"repo": true, "branch": "main", "head": "5c8bfec", "root_tree": "12831fe1…", "matches_pin": true, "worktree_possible": true, "error": null}`
— **`worktree_possible: true`**, which §2a independently confirmed by actually doing it.

**The cleanup and the `core.filemode` change disturbed nothing.** `status` is still `ready`.

## ⭐ 4. Is `anchors_checked` non-zero for the four docs with registered anchors? — **NO.**

**First, a correction to the brief: there are SEVEN `open_with: "doc"` entries, not eight.** I counted
independently in both `BUNDLE_MANIFEST.json` and `WORKSPACE-READY.json` — both say 7 (the 4 anchored
docs + the 3 `.claude/tasks/*.md`). The brief's own breakdown (4 + 3) also sums to 7.

| `open_with: "doc"` path | `loaded` | **`anchors_checked`** | anchors registered in `DOC_ANCHORS` | expected |
|---|---|---|---|---|
| `.claude/AGENT_PROTOCOL.md` | true | **0** | 2 | ⛔ should be 2 |
| `.claude/agents/autonomous-hub.md` | true | **0** | 4 | ⛔ should be 4 |
| `.claude/autonomous-hub-charter.md` | true | **0** | 4 | ⛔ should be 4 |
| `.claude/handover_context.md` | true | **0** | 1 | ⛔ should be 1 |
| `.claude/tasks/hub-composition.md` | true | 0 | 0 | ✅ 0 is correct |
| `.claude/tasks/hub-deletion.md` | true | 0 | 0 | ✅ 0 is correct |
| `.claude/tasks/hub-refusal.md` | true | 0 | 0 | ✅ 0 is correct |

Raw `detail` objects, quoted from the fresh `WORKSPACE-READY.json`:

```json
".claude/AGENT_PROTOCOL.md"          {"anchors_checked": 0, "chars": 9713,    "lines": 72}
".claude/agents/autonomous-hub.md"   {"anchors_checked": 0, "chars": 10433,   "lines": 158}
".claude/autonomous-hub-charter.md"  {"anchors_checked": 0, "chars": 6532,    "lines": 81}
".claude/handover_context.md"        {"anchors_checked": 0, "chars": 1001293, "lines": 3845}
".claude/tasks/hub-composition.md"   {"anchors_checked": 0, "chars": 4658,    "lines": 66}
".claude/tasks/hub-deletion.md"      {"anchors_checked": 0, "chars": 4414,    "lines": 59}
".claude/tasks/hub-refusal.md"       {"anchors_checked": 0, "chars": 4723,    "lines": 65}
```

**`anchors_checked` is 0 for all four docs that have registered anchors. Check #3 inspects
0 of 11 registered anchors on this machine.** The Mac's plan requirement — *"`anchors_checked` must
be non-zero"* — **cannot be satisfied by the currently-shipped script on Windows.** The hub's
analysis was right.

### The mechanism, measured

`scripts/verify_hub_workspace.sh:209`:

```python
rel = str(Path(p).resolve().relative_to(root))
```

`DOC_ANCHORS` (line 191) is keyed with **forward** slashes. `str()` on a Windows `PurePath` emits
**backslashes**. I reproduced the exact lookup with the real table and the real files:

```
built key : '.claude\\AGENT_PROTOCOL.md'          table key : '.claude/AGENT_PROTOCOL.md'
  lookup hit: False  -> anchors_checked = 0
  with .as_posix(): '.claude/AGENT_PROTOCOL.md' -> hit=True, would be 2
  anchors ACTUALLY present in the file: 2/2   missing=NONE

built key : '.claude\\autonomous-hub-charter.md'  table key : '.claude/autonomous-hub-charter.md'
  lookup hit: False  -> anchors_checked = 0
  with .as_posix(): hit=True, would be 4
  anchors ACTUALLY present in the file: 4/4   missing=NONE

built key : '.claude\\agents\\autonomous-hub.md'  table key : '.claude/agents/autonomous-hub.md'
  lookup hit: False  -> anchors_checked = 0
  with .as_posix(): hit=True, would be 4
  anchors ACTUALLY present in the file: 4/4   missing=NONE

built key : '.claude\\handover_context.md'        table key : '.claude/handover_context.md'
  lookup hit: False  -> anchors_checked = 0
  with .as_posix(): hit=True, would be 1
  anchors ACTUALLY present in the file: 1/1   missing=NONE
```

⭐ **The last line of each block is the reassuring half: all 11 anchors ARE present in the files,
including the pin `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e` in the charter and `Known Issues` in
`handover_context.md`.** Nothing is truncated. This is a pure dict-lookup miss on the path
separator, and `.as_posix()` on line 209 makes every one of them hit — the exact one-call fix
already identified. ⛔ **I did not touch the file.** It is sha256-pinned; its sha256 is byte-identical
in `before.sha` and `final.sha`.

⭐ **A silent-failure note for the Head.** `DOC_ANCHORS.get(rel, [])` degrades to an *empty* check,
not an error. On Windows this check passes loudly and inspects nothing. `anchors_checked: 0` was the
only symptom — it is worth making the fix also **assert** that a doc in the table yields a non-zero
count, so a future separator regression fails instead of going quiet.

---

# Final state of the workspace

| | |
|---|---|
| `git fsck --no-progress` | **exit 0, no output** |
| 57 manifest sha256 vs. my pre-cleanup snapshot | **byte-identical (`diff` empty), 3rd re-check** |
| root tree | `12831fe1be17d04da027dcbb6e45b4dff6f1ef95` ✅ |
| `git status --porcelain` | 1 row: `?? hub-refusal.supplement.tar.gz` (see Q1) |
| `git worktree list` | main worktree only |
| `git branch -a` | `* main` only |
| `._*` files anywhere | **0** |
| `core.filemode` / `core.autocrlf` (local) | `false` / `false` |
| `WORKSPACE-READY.json` | `status: ready`, exit 0 |
| `hub-refusal.supplement.tar.gz` | ⛔ **NOT applied**, untouched at 216,089,526 bytes |

**Total changes I made to the workspace: 588 files deleted (all `._*`), and one git config line
(`core.filemode false`).** No commit, no push, no reset, no checkout. Nothing in `chlu/`, `tests/`
or `.claude/` was edited or moved. `--return-ingest` was never run.

⭐ **No sub-agent of mine is running. I spawned none at any point.**

---

# What I could not resolve honestly

- **I cannot prove the 588 sidecars were content-free by inspecting them post hoc** — they are
  deleted. I verified the *guard* (zero manifest overlap, 582×163 + 6×263 byte sizes matching the
  established evidence, every one a `._`-prefixed twin of a real neighbour) and I proved the
  *consequence* (all 57 pinned sha256s unchanged, pin tree unchanged, verifier still `ready`). If a
  scientific asset had been inside one of them, it was not one of the 57 pinned files and not
  anything `verify_hub_workspace.sh` or `tests/test_hub_workspace.py` can see. ⭐ They are not
  recoverable from this machine.
- **The results leg was exercised against a staged payload under an overridden
  `CHLU_HUB_OUTPUTS`, not against `.claude/outputs/` itself.** The script code path is identical —
  only the root differs — but I have not measured what `--return-pack` does on a multi-GB real slug
  (walk time, tar size, whether `gzip -c` on a several-GB `pack.tar` is acceptable here). Using a
  real slug was explicitly out of scope, and `.claude/` is write-protected by the spoke rules.
- **`--return-ingest` remains completely untested**, by instruction. Its path-prefix / `..` /
  symlink assertions are the half that runs on the Mac. ⚠ I did notice it reads
  `RETURN_MANIFEST.json` with a bare `.read_text()` (line 273) — the same encoding hazard the
  `CONTEXT.md` notes flag elsewhere — but that runs on macOS, where it is harmless.
- **I could not fix the `anchors_checked` bug** and did not try. Correct per §3.5 — the file is
  sha256-pinned and the humans own the fix.

---

# Numbered open questions

1. ⚠ **There is a second 206 MB copy of `hub-refusal.supplement.tar.gz` INSIDE the bundle**, at
   `chlu-hub-bundle/hub-refusal.supplement.tar.gz`. It is a separate file, not a hardlink, from the
   approved copy at `chlu/hub-refusal.supplement.tar.gz`, and it is the **only** untracked row in
   `git status`. I did not touch it (outside my `._*` scope, and adjacent to the gated Spoke H
   decision). **Who put it there, and should it be removed?** It costs 206 MB, it will be swept into
   any tarring of the workspace, and it is the one thing standing between this repo and a genuinely
   0-untracked `git status`.
2. **The brief says 8 `open_with: "doc"` entries; there are 7.** Both `BUNDLE_MANIFEST.json` and
   `WORKSPACE-READY.json` agree on 7, and the brief's own 4+3 breakdown sums to 7. Is an **eighth
   doc missing from the manifest**, or is the 8 simply a miscount? I assumed a miscount.
3. **Should the `.as_posix()` fix also add an assertion?** `DOC_ANCHORS.get(rel, [])` fails *open* —
   an unmatched key means "check nothing" and still reports success. A one-line assert that a doc
   listed in `DOC_ANCHORS` yields `anchors_checked > 0` would turn the next separator regression
   into a failure instead of a silent zero. That is a second upstream change, not the one already
   scoped.
4. **Does the Mac side need re-pinning after this cleanup?** All 57 sha256s and the pin tree are
   unchanged, so `BUNDLE_MANIFEST.json` is still correct. But if the Mac ever rebuilds a bundle from
   its own tree, it will reintroduce all 588 sidecars unless `COPYFILE_DISABLE=1` is set on the tar.
   **Is that now set in the build path?** Nothing here can enforce it.
5. **Is `core.filemode false` wanted permanently, or only for this workspace?** It is set
   `--local`, so it dies with this bundle and cannot follow anything home. Flagging only because the
   Mac's own checkout still has `core.filemode true` and will keep producing the 100755/100644
   asymmetry for any future bundle.
6. **GNU tar, not bsdtar.** The two-pass `-C` construction works fine here, so no change is needed —
   but it means **this machine cannot regression-test the bsdtar behaviour the workaround exists
   for.** Only the Mac can. Worth knowing before anyone "simplifies" that code on the basis of a
   Windows green tick.
