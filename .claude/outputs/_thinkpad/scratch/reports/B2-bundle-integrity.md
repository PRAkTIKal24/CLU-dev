# SPOKE B2 — Bundle integrity and macOS contamination

**Verdict: YES — the bundle is byte-intact and safe to build on.**
All 57 sha256-pinned files are present, correct-sized and checksum-clean. The git root tree equals the
pin. The transfer is **not** the problem; A2 is chasing the right problem.

**But three findings change the picture, and one contradicts the brief:**

1. ⚠ **The brief's §3 precondition is false.** `CONTEXT.md` and the B2 brief both state *"none are inside
   `.claude/`"*. **90 of the 588 AppleDouble files ARE under `.claude/`**, sitting directly beside frozen
   artefacts (`._features.npy` next to `features.npy`, `._decoder_v1_s0.pkl` next to the real decoder).
   Because that precondition fails, **I did not execute the quarantine.** The plan is in §3 below, unrun.
2. 🔴 **AppleDouble files are inside `.git/` and they break git.** `.git/objects/pack/._pack-*.idx`,
   seven `.git/objects/XX/._<sha>` and `.git/refs/heads/._main` make **every** git command emit errors
   and `git fsck` fail. Git still resolves correctly, but this is live corruption of the object store's
   namespace, and the hub's sandbox is a **git worktree**.
3. ⚠ **`git status` is not clean: 36 phantom "modified" files**, caused by `core.filemode = true`
   (inherited from the Mac) meeting a filesystem with no exec bit. Zero content changed.

*Method note: everything below was done with Git Bash coreutils and `git` only — no Python, as briefed.*

---

## What I did

Worked read-only in `/c/Users/PratikJawahar/chlu/chlu-hub-bundle`. `jq` is **not** present in this Git
Bash, so I parsed `BUNDLE_MANIFEST.json` with an `awk` record scanner (`/tmp/manifest.tsv`, 57 records —
matching the manifest's own count) and verified each with `sha256sum -b` and `stat -c %s`.

I made **exactly one** mutation, the one §4 explicitly permits: `git config --local core.autocrlf false`
(see §4). Nothing else was written, moved or deleted anywhere in the bundle.

---

## 1. Independent checksum of all 57 manifest files

**All 57 survived the transfer. Yes.** 0 missing · 0 byte-size mismatches · 0 sha256 mismatches.

| path | bytes expected / actual | present | sha256 |
|---|---|---|---|
| `.claude/AGENT_PROTOCOL.md` | 9802 / 9802 | ✅ | match |
| `.claude/agents/autonomous-hub.md` | 10694 / 10694 | ✅ | match |
| `.claude/autonomous-hub-charter.md` | 6610 / 6610 | ✅ | match |
| `.claude/data/gen/shapes3d/a7_redundant.json` | 69217 / 69217 | ✅ | match |
| `.claude/data/gen/shapes3d/c_invalid.json` | 72041 / 72041 | ✅ | match |
| `.claude/data/gen/shapes3d/factor_index.npz` | 3844150 / 3844150 | ✅ | match |
| `.claude/data/gen/shapes3d/query_sets.json` | 1616 / 1616 | ✅ | match |
| `.claude/handover_context.md` | 1023731 / 1023731 | ✅ | match |
| `.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` | 2660729 / 2660729 | ✅ | match |
| `.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/seed_0/baselines_manifest.json` | 107880 / 107880 | ✅ | match |
| `.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/seed_1/baselines_manifest.json` | 94556 / 94556 | ✅ | match |
| `.claude/outputs/gen-arms-a7-wiring/runs/historical_688128/seed_2/baselines_manifest.json` | 114245 / 114245 | ✅ | match |
| `.claude/outputs/gen-decoder-lift/winner/decoder_v1_s0.pkl` | 6069142 / 6069142 | ✅ | match |
| `.claude/outputs/gen-decoder-lift/winner/decoder_v1_s1.pkl` | 6069142 / 6069142 | ✅ | match |
| `.claude/outputs/gen-decoder-lift/winner/decoder_v1_s2.pkl` | 6069142 / 6069142 | ✅ | match |
| `.claude/outputs/gen-decoder-lift/winner/winner_hashes.json` | 2157 / 2157 | ✅ | match |
| `.claude/outputs/gen-external-classifier/classifier_gate.json` | 1296 / 1296 | ✅ | match |
| `.claude/outputs/gen-external-classifier/seed_0/classifier_manifest.json` | 8131 / 8131 | ✅ | match |
| `.claude/outputs/gen-external-classifier/seed_0/external_classifier.pkl` | 3918847 / 3918847 | ✅ | match |
| `.claude/outputs/gen-external-classifier/seed_1/classifier_manifest.json` | 8092 / 8092 | ✅ | match |
| `.claude/outputs/gen-external-classifier/seed_1/external_classifier.pkl` | 3918847 / 3918847 | ✅ | match |
| `.claude/outputs/gen-external-classifier/seed_2/classifier_manifest.json` | 7997 / 7997 | ✅ | match |
| `.claude/outputs/gen-external-classifier/seed_2/external_classifier.pkl` | 3918847 / 3918847 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/bridge_manifest.json` | 4128 / 4128 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/decoder.pkl` | 1004401 / 1004401 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/encoder.pkl` | 1716565 / 1716565 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/feature_cache.json` | 555 / 555 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/feature_rows.npy` | 3840128 / 3840128 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/features.npy` | 1966080128 / 1966080128 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/probe.pkl` | 2534999 / 2534999 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/bridge_manifest.json` | 4142 / 4142 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/decoder.pkl` | 1004401 / 1004401 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/encoder.pkl` | 1716565 / 1716565 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/feature_cache.json` | 551 / 551 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/feature_rows.npy` | 74432 / 74432 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/features.npy` | 38043776 / 38043776 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/probe.pkl` | 2534999 / 2534999 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/bridge_manifest.json` | 4121 / 4121 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/decoder.pkl` | 1004401 / 1004401 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/encoder.pkl` | 1716565 / 1716565 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/feature_cache.json` | 551 / 551 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/feature_rows.npy` | 74432 / 74432 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/features.npy` | 38043776 / 38043776 | ✅ | match |
| `.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_2/probe.pkl` | 2534999 / 2534999 | ✅ | match |
| `.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_0/store_s0.pkl` | 536929 / 536929 | ✅ | match |
| `.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_1/store_s1.pkl` | 536929 / 536929 | ✅ | match |
| `.claude/outputs/gen-read-compose/runs/shapes3d_recon/seed_2/store_s2.pkl` | 536929 / 536929 | ✅ | match |
| `.claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_0/store_manifest.json` | 131506 / 131506 | ✅ | match |
| `.claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_1/store_manifest.json` | 131335 / 131335 | ✅ | match |
| `.claude/outputs/gen-store-grouping/runs/shapes3d_recon/seed_2/store_manifest.json` | 130926 / 130926 | ✅ | match |
| `.claude/outputs/gen-store-grouping/runs/shapes3d_recon/store_plan.json` | 17695 / 17695 | ✅ | match |
| `.claude/tasks/hub-composition.md` | 4838 / 4838 | ✅ | match |
| `.claude/tasks/hub-deletion.md` | 4577 / 4577 | ✅ | match |
| `.claude/tasks/hub-refusal.md` | 4889 / 4889 | ✅ | match |
| `scripts/bundle_hub_workspace.sh` | 48051 / 48051 | ✅ | match |
| `scripts/verify_hub_workspace.sh` | 22458 / 22458 | ✅ | match |
| `tests/test_hub_workspace.py` | 32144 / 32144 | ✅ | match |

**Corroboration (unplanned, valuable).** While I worked, a parallel spoke got Python running and the
verifier wrote `WORKSPACE-READY.json` at 17:52 — so `CONTEXT.md`'s "it was never written" is now stale.
Its Python checksum stage independently reports:

```json
"checksum_match": true,
"code_is_the_pin": true,
"status": "NOT-READY",
"python": "3.12.10"
```

Two independent implementations (its Python `hashlib`, my `sha256sum`) agree on all 57. The residual
`NOT-READY` comes from the targeted pytest run (`.workspace-verify-test.log` = `..FF`, 2 passed 2
failed), **not** from integrity. I re-ran my full 57-file verification afterwards: **still byte-identical**,
and the root tree still `12831fe1…`. Nothing that spoke did disturbed the pinned set.

---

## 2. Executable bits

### The three manifest code files

| file | `ls -l` | `stat` octal | `test -x` | git index mode |
|---|---|---|---|---|
| `scripts/bundle_hub_workspace.sh` | `-rwxr-xr-x` | 755 | TRUE | 100755 |
| `scripts/verify_hub_workspace.sh` | `-rwxr-xr-x` | 755 | TRUE | 100755 |
| `tests/test_hub_workspace.py` | `-rw-r--r--` | 644 | FALSE | 100644 |

### All 34 `.sh` files

31 report 755; three report 644: `env_hub.sh`, `scripts/cloud/config.sh`, `scripts/csf3/env.sh`
(the latter two are sourced fragments, not entry points). Full listing captured; the split is the point.

### Does NTFS via Git Bash preserve a meaningful exec bit? **No.**

`test -x` agrees with `ls -l` on all 34 — but that is not evidence of preservation, because **both read
the same MSYS heuristic**. The decisive experiment is comparing MSYS's view against git's recorded mode:

| file | MSYS `ls` | git index | has `#!` shebang? |
|---|---|---|---|
| `scripts/verify_hub_workspace.sh` | **755** | 100755 | yes — `#!/usr/bin/env bash` |
| `scripts/cloud/config.sh` | **644** | **100755** | **no** — starts `# scripts/cloud/conf…` |
| `scripts/csf3/env.sh` | 644 | 100644 | no |

`scripts/cloud/config.sh` is recorded `100755` by git yet MSYS reports `644`. The mode was **not** carried
across; **MSYS is synthesising the exec bit from the `#!` magic number.** Real POSIX permissions did not
survive tar → NTFS. `ls -l` here renders file *content*, not stored metadata.

This is also the exact cause of finding 3 — see §4.

### Will `_load_code()`'s `os.access(p, os.X_OK)` assertion fail? **No, I expect it to pass.** Two reasons:

**(a) It never runs on the `.py` file.** `scripts/verify_hub_workspace.sh:220-234` returns early:

```python
def _load_code(p):
    p = Path(p)
    if p.suffix == ".py":
        src = p.read_text()
        compile(src, str(p), "exec")
        return {"lines": ..., "parsed": "python"}     # <- returns BEFORE the X_OK check
    ...
    if not os.access(p, os.X_OK):
        raise PermissionError("%s lost its executable bit in transit" % p.name)
```

So `tests/test_hub_workspace.py` — the one file that genuinely reports non-executable — is never tested
for X_OK. Only the two `.sh` files reach line 232.

**(b) Both `.sh` files pass under either interpreter.** They have shebangs, so MSYS reports them 755; and
on native Windows CPython `os.access()` is implemented over `GetFileAttributesW`, which has no execute
concept — `X_OK` is effectively ignored and the call returns `True` for any existing file. The assertion
is a no-op on Windows rather than a hazard.

⚠ **The real risk on that line is the one above it, not the X_OK.** `_load_code` does
`subprocess.run(["bash", "-n", str(p)])`. Launched from *native* Windows Python, that requires `bash` to
be resolvable on `PATH` — a POSIX assumption crossing into Windows Python, and a plausible failure that
would surface as a confusing `SyntaxError`. I confirmed the scripts themselves are fine: `bash -n` returns
0 on both, and both are **LF-clean (CR count = 0)**. So if that check ever fails, suspect the launcher,
not the script. **That belongs to A2 — flagging, not chasing.**

---

## 3. The 588 AppleDouble `._*` files

### Distribution — the brief's figures are close, but the `.claude` line is wrong

| area | actual count | brief said |
|---|---|---|
| `chlu/` | 232 | ~160 |
| `tests/` | 145 | ~145 ✅ |
| **`.claude/`** | **90** | **"none" ❌** |
| `.git/` | 54 | "some" ✅ |
| `scripts/` | 48 | ~47 ✅ |
| root | 16 | 16 ✅ |
| `projects/` | 2 | — |
| `logs/` | 1 | — |
| **total** | **588** | 588 ✅ |

I re-derived this rather than trusting it, which is how the `.claude/` error surfaced. Examples of the 90:
`.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/._features.npy`,
`.claude/outputs/gen-decoder-lift/winner/._decoder_v1_s0.pkl`,
`.claude/data/gen/shapes3d/._factor_index.npz`, `.claude/._autonomous-hub-charter.md`.

**I did independently re-confirm the other precondition: no `._*` path appears in `BUNDLE_MANIFEST.json`
(grep count 0).** And the manifest's 54 `.claude` entries are *exactly* the 54 real files on disk —
`diff` of the two sorted lists is empty. So the AppleDouble files are pure additions; nothing pinned is
shadowed *out*, only shadowed *beside*.

### Are they zero-byte, or real resource-fork data? **Neither — they are xattr-only, and inert.**

None is zero-byte. 582 are exactly **163 bytes and byte-identical to each other**
(sha256 `1565991b…`); 6 are 263 bytes. Hexdump shows the AppleDouble v2 magic `00051607`, `Mac OS X`,
an `ATTR` block, and a single `com.apple.provenance` xattr (the 263-byte ones add `com.apple.macl`).
**Entry ID 1 (the resource fork) is absent — total payload is ~15 KB of Finder provenance metadata across
all 588.** No CHLU data is inside them, so quarantining loses nothing scientific.

### Does `pytest` collect any? **No.**

`pyproject.toml:94-95` sets `testpaths = ["tests"]`, `python_files = "test_*.py"`. **I reasoned from the
glob** — Python has only just appeared via a parallel spoke, and I did not run `--collect-only`, to stay
read-only and out of A2's way. pytest fnmatches the *basename* against `test_*.py`; `._test_foo.py` begins
with `.` so it cannot match a pattern anchored on the literal `test`. Same for `conftest.py`, matched by
exact name. **Stronger still:** the verifier does not rely on collection at all — line 453 is
`"$PY_BIN" -m pytest -q "$TEST_FILE"`, naming the file explicitly.

### Does any code path glob a directory rather than name a file?

I grepped `chlu/` for `glob(`, `rglob(`, `iterdir()`, `listdir(`. **Important: `pathlib`'s `*` does NOT
exclude dotfiles** (unlike the shell), so `*.npz` genuinely matches `._factor_index.npz`. Results:

**Safe — prefix-anchored patterns (`._` cannot match):**
`gen_wells.py:681` `ckpt_{tag}_step*.json` · `exp_anti_erosion.py:1178` `erosion_*_records.json` ·
`exp_cluformer_pilot.py:1316` `pilot_*` · `kt/postproc.py:48` `{mode}_task*.json` ·
`exp_gen_atom_mean.py:1234` and `exp_gen_cell.py:1988` `seed_*` (also `is_dir()`-guarded).

**Safe — exact filename:** `gen_splits.py:985` `root.glob("**/query_sets.json")`.

**Safe — guard filters it out:** `project.py:102` `base_dir.iterdir()` *does* enumerate
`projects/._README.md`, but the next line requires `is_dir() and (d/"project.json").exists()`.

**⚠ LOW — `exp_gen_encoder.py:182`:**
`if c.is_dir() and any(c.glob("*.h5")) or (c.is_dir() and any(c.glob("*.npz")))`.
This is the gen-data-root detector, and `.claude/data/gen/shapes3d/._factor_index.npz` **does** match
`*.npz`. Harmless *today* because the real `factor_index.npz` is there too, so `any()` is `True`
regardless. It would only bite in a directory containing **only** `._*.npz` — which would make an empty
directory look like a valid data root, precisely the "silently-skipped data set" that line's own R26.7
comment warns against. Worth knowing; not currently triggered.

**⚠ LOW — `chlu/utils/checkpoints.py:124`:** `sorted(directory.glob(pattern))`, with
`get_latest_checkpoint(..., pattern="*.pkl")` above it. `*.pkl` would match
`.claude/outputs/gen-decoder-lift/winner/._decoder_v1_s0.pkl`. Mitigated twice over: selection is
`max(key=st_mtime)` not index-based, and **grep finds no caller of either function** in `chlu/` or
`scripts/` — it is currently dead code.

**Not a risk at all — the four loaders the brief flagged.** I read them in `verify_hub_workspace.sh` and
the premise does not hold: `query_registry`, `control_set`, `feature_cache` and `npz_index` **do not scan
directories.** `_load_query_registry` and `_load_npz_index` take a file path; `_load_control_set` is
registry-driven (it looks the set up *by name* in `query_sets.json` and raises `KeyError` otherwise);
`_load_feature_cache` calls `gen_bridge.py:1032`, which reads `feature_cache.json` and opens
`manifest["features_file"]` **by name, with its own sha256 verification**. `grep glob|iterdir|listdir`
over the whole verifier returns nothing. Good design; the contamination cannot reach them.

**Dataset loaders (not exercised by this bundle, flagged for completeness):**
`industrial/mimii.py:46,59` `rglob("*.wav")`, `industrial/skab.py:84` `glob("*.csv")`,
`industrial/smd_tsb.py:63` `rglob("*_SMD_*.csv")` — all three **would** pick up `._` siblings, and
`._foo.wav` is not a valid WAV. No such data is in the bundle, but if industrial datasets are ever
unpacked from a Mac-made archive, these three break.

### 🔴 Do the ones in `.git/` affect git? **Yes — materially. This is the finding I would act on first.**

Every single git invocation now prints to stderr:

```
error: index file .git/objects/pack/._pack-e1cc3bb553b933f50a1ed4be1608848514687219.idx is too small
```

(repeated 2–7× per command, on `rev-parse`, `status`, `log`, `rev-list` — everything). `git fsck` fails:

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

Read it precisely: git is **correct but noisy**. It tries to open `._pack-….idx` *as a pack index*, finds
163 bytes, rejects it, and falls through to the real `.idx`. Every object still resolves; the pin
verifies. But:

- **`git fsck` exits non-zero.** Any gate that treats fsck as a health check reports a corrupt repo.
- **`.git/refs/heads/._main` is a malformed ref**, and `.git/objects/pack/._multi-pack-index` shadows the
  midx. **`git worktree add` is the hub's very first move** (per the charter, the sandbox *is* a worktree)
  and it writes refs and reads the object store.
- The 14 `.git/hooks/._*.sample` files are inert — git runs hooks only by exact name, and `.sample` never
  runs regardless.
- Anything capturing git's **stderr** — or running `set -e` over a pipeline — can misread this as failure.

This alone justifies quarantine, independently of the `.claude/` question.

### Quarantine plan — ⛔ **WRITTEN, NOT EXECUTED**

**Why not executed:** the brief conditions execution on *"none is under `.claude/`"*. **90 are.** The
precondition fails, so per the brief I stopped. I also think that is right on the merits: these sit
against ~2.0 GB of frozen scientific artefacts, and a bulk move there deserves a human's signature — not
a spoke's. The other two preconditions **do** hold (none in the manifest; the move is reversible).

The plan, ready to run verbatim:

```bash
# 1. Quarantine root; relative paths preserved; MOVE not rm -- fully reversible.
Q=/c/Users/PratikJawahar/chlu/_appledouble_quarantine
cd /c/Users/PratikJawahar/chlu/chlu-hub-bundle

# 2. Snapshot manifest state BEFORE (manifest.tsv = path/bytes/sha256, parsed by awk).
while IFS=$'\t' read -r p b s; do sha256sum -b "$p"; done < manifest.tsv > /tmp/before.sha

# 3. Enumerate to a list first -- never pipe find straight into a mutation.
find . -name '._*' -type f | sed 's#^\./##' | sort > /tmp/appledouble.list
wc -l /tmp/appledouble.list          # must read exactly 588

# 4. HARD GUARD: abort if any listed path appears in BUNDLE_MANIFEST.json.
grep -Ff /tmp/appledouble.list BUNDLE_MANIFEST.json && { echo "ABORT: manifest hit"; exit 1; }

# 5. Move, preserving relative paths.
while read -r f; do mkdir -p "$Q/$(dirname "$f")"; mv "$f" "$Q/$f"; done < /tmp/appledouble.list

# 6. PROVE nothing else changed.
while IFS=$'\t' read -r p b s; do sha256sum -b "$p"; done < manifest.tsv > /tmp/after.sha
diff /tmp/before.sha /tmp/after.sha && echo "all 57 unchanged"
git status --porcelain | wc -l       # expect 36 (the mode rows), 0 untracked
git fsck --no-progress && echo "fsck clean"
git rev-parse $(git rev-list --max-parents=0 HEAD)^{tree}   # expect 12831fe1...
```

**Rollback** is the same loop with source and destination swapped.

**Sequencing recommendation:** `.git/` (54 files) is the highest-value, lowest-risk slice — nothing there
is a scientific artefact, and it is what is actually breaking a tool today. It could be done first and
separately from the 90 under `.claude/`, which are the ones needing sign-off.

---

## 4. Git state — the pin

**The tree matches the pin. ✅**

```
root commit : 172fa35d8c2b5480f4b00ba3ac814532875b001b
its tree    : 12831fe1be17d04da027dcbb6e45b4dff6f1ef95
manifest    : pin_tree = 12831fe1be17d04da027dcbb6e45b4dff6f1ef95   <- EXACT MATCH
HEAD        : 5c8bfec   branch: main
```

The two-commit history is exactly as documented — `172fa35 [bundle] the CHLU tree at pin ae907466…,
verbatim`, then `5c8bfec [bundle] the workspace tooling, overlaid on the pin`.

**File-count reconciliation is exact:** `git ls-tree -r` on the root commit = **416 files**, matching the
manifest's `n_tracked_files: 416`. HEAD = **419**. The diff between them is precisely the three
`tooling_overlay` files (`scripts/bundle_hub_workspace.sh`, `scripts/verify_hub_workspace.sh`,
`tests/test_hub_workspace.py`). Every file accounted for.

### `core.autocrlf` — checked, and **I set it**

`git config --get core.autocrlf` returned `false`, but tracing the origin showed *why*, and it was luck:

```
file:C:/Program Files/Git/etc/gitconfig     true      <- system default, the dangerous one
file:C:/Users/PratikJawahar/.gitconfig      false     <- global, happens to override it
(no repo-local value)
```

There was **no repo-local setting**. It resolved to `false` only because this user's global config
overrides the system default — so a different user, a different shell, or an edit to `~/.gitconfig` would
silently flip the repo to `autocrlf=true`, rewrite all 34 `.sh` files to CRLF, and produce exactly the
`bad interpreter: no such file or directory` that tells you nothing. §4 explicitly permits setting this,
so I pinned it where it cannot drift:

```
git config --local core.autocrlf false      # <- THE ONE MUTATION I MADE
```

Verified after: repo-local value `false`, resolved value `false`. I also confirmed the files are currently
**LF-clean (CR count = 0** on `verify_hub_workspace.sh`, `bundle_hub_workspace.sh`, `env_hub.sh`), so
nothing had already been damaged. `core.eol` and `core.safecrlf` are unset; `ignorecase=true` and
`precomposeunicode=true` are macOS leftovers and both are harmless on NTFS.

### ⚠ `git status` is NOT clean — 36 phantom modifications

```
36 M      443 ??
```

**The 36 `M` entries are mode-only; zero bytes differ.** `git diff --stat` reports
`36 files changed, 0 insertions(+), 0 deletions(-)`, and `git diff --summary` is 36 lines of:

```
 mode change 100755 => 100644 scripts/verify_hub_workspace.sh
 mode change 100755 => 100644 scripts/cloud/config.sh
 mode change 100755 => 100644 scripts/c3_geometry_sweep.py
 ...
```

**Cause:** `.git/config` carries `filemode = true`, written on macOS. Git-for-Windows cannot read an exec
bit from NTFS and reports `0644` for every regular file, so all 36 files stored as `100755` look modified
forever. This is the same root cause as §2 seen from the git side — and it is the cleanest proof that the
exec bit did not survive the transfer.

**Recommended fix (one line), which I deliberately did NOT apply:**

```bash
git config --local core.filemode false
```

It is safe — it changes only what git *compares*, never the `100755` modes recorded in the index or the
pin tree. **I left it alone on purpose:** other spokes are running in parallel right now and may be
reading `git status` output, and flipping it underneath them would change what they see mid-flight. §4
sanctioned `autocrlf` by name; `filemode` it did not. **Hub's call — but it should be made, because a
permanently-dirty tree can block `git worktree add` and trains everyone to ignore `git status`.**

**The 443 `??` are the AppleDouble files, and `.gitignore` does NOT cover them.** `git check-ignore -v`
returns no match for `._LICENSE`. (443 rather than 588 because `.gitignore` has `.claude/**`, so the 90
there are excluded as part of an already-ignored tree, and `.git/`'s 54 are internal.) `.gitignore`
contains `.DS_Store` but **no `._*` rule** — the Mac-hygiene line that would have prevented all of this is
the one that is missing. Adding `._*` would silence the untracked noise, but ⚠ **it would hide the problem
rather than fix it**, and `.gitignore` is a tracked file at the pin, so I did not touch it.

---

## 5. Tar and transfer provenance

**The original tarball is present and intact.**

```
/c/Users/PratikJawahar/chlu/chlu-hub-bundle.tar.gz
size   : 1,196,136,468 bytes
gzip -t: OK  (CRC/length valid over the whole stream)
sha256 : c3cac259a18d50bbaf0eb2c0c9aac4e43cf955348a7aa7d7fd126856c39a8375
```

⚠ **Do not compare that 1.196 GB against the manifest's `tar_bytes: 10076160`.** They are different
objects: `tar_bytes` is nested *inside* the `"code"` block and refers to the ~10 MB `git archive` of the
code tree, not the 2 GB bundle. There is no discrepancy here — I record the distinction because the brief
phrases it as one comparison and it is easy to misread as a 100× shortfall.

**`gcloud` cross-check: UNVERIFIED, not guessed.** The published checksum in
`gs://chlu-cloud-forgisprova/handoff/` could not be fetched — `gcloud` is broken for want of a Python
interpreter. The sha256 above is recorded so the human can compare it against the bucket.

### Byte accounting for `.claude` — reconciled, benign

| measure | bytes |
|---|---|
| manifest `claude_bytes` | 2,102,084,732 |
| on disk, apparent (`du -sb`) | 2,101,997,349 |
| on disk, real files only (54, excl. AppleDouble) | 2,101,982,079 |
| the 90 AppleDouble files | 15,270 |
| on disk, allocated (`du -sk` × 1024) | 2,102,264,832 |

The apparent-size total is 87,383 bytes (0.004%) below the manifest. **This is a measurement-convention
difference, not missing data**, and I can say so rather than assume it, on three grounds:

1. The manifest figure **falls between** the apparent size (2,101,997,349) and the allocated size
   (2,102,264,832). macOS `du` on APFS counts directory inodes and xattr storage that NTFS reports as
   zero; the bundle is bracketed by the two conventions exactly as you would expect.
2. **The file sets are identical.** `diff` between the 54 real files on disk and the 54 `.claude` entries
   in the manifest is **empty** — nothing extra, nothing absent.
3. **Decisively: all 54 sha256 values match.** A sha256 cannot survive a missing or truncated byte. A `du`
   total is a weak proxy for integrity; 54 matching digests are a strong one, and they agree.

No `.DS_Store` files were found anywhere (0), so that Mac-hygiene vector is clean — only `._*` came across.

---

## What works

- All **57** pinned files: present, correct size, correct sha256 — verified twice, before and after a
  parallel spoke ran the verifier.
- Root-commit tree **equals the pin**; the 416/419 file counts reconcile exactly against the overlay.
- `.claude` file set is **exactly** the manifest's 54 — no additions, no omissions.
- Shell layer: all 34 `.sh` files **LF-clean**; `bash -n` clean on both manifest shell scripts.
- `core.autocrlf` now pinned `false` repo-locally.
- The verifier's four set-opening loaders are **name-driven and immune** to `._*` contamination.
- `pytest` cannot collect AppleDouble files, and the verifier names its test file explicitly anyway.
- Original tarball present, `gzip -t` clean.
- AppleDouble payload is **xattr-only** — ~15 KB of Finder provenance in total, **no resource-fork data** —
  so quarantine destroys nothing.

## What is broken

1. 🔴 **54 AppleDouble files inside `.git/`** — `git fsck` fails; every git command emits errors. Full
   output in §3. The hub's sandbox is a worktree, so this is on the critical path.
2. ⚠ **90 AppleDouble files under `.claude/`** — contradicting the brief and `CONTEXT.md`. Currently
   inert, but it invalidates a stated precondition and blocked the quarantine.
3. ⚠ **`git status` permanently dirty: 36 mode-only rows** from `core.filemode=true`. One-line fix, not
   applied; reasons in §4.
4. ⚠ **`.gitignore` has no `._*` rule** — 443 untracked files.
5. ⚪ Two low-severity `._`-matching globs (`exp_gen_encoder.py:182`, `checkpoints.py:124`) plus three in
   the unused industrial dataset loaders. None currently triggered.

## What I could not resolve honestly

- **The GCS published checksum is unverified.** `gcloud` cannot run without Python. I recorded the local
  sha256 rather than guessing. **The Mac→GCS leg is therefore confirmed only as far as "internally
  consistent and gzip-valid", not as "matches what was uploaded."** My 57 sha256 matches make corruption
  vanishingly unlikely — the manifest was built before transfer — but that is inference, not measurement.
- **The 87,383-byte `claude_bytes` gap** is explained to my satisfaction (§5) but not proven to the byte;
  that would need `du` on `batMac.local`. It is bounded, benign, and cannot coexist with 54 matching
  digests.
- **I did not execute the quarantine** — a stated precondition was false. Deliberate, per the brief.
- **I could not run the `os.access`/`bash -n` assertion end-to-end** — that is A2's Python work and I
  stayed out of it. §2 is reasoned from the source plus CPython's documented Windows behaviour, and is
  labelled as prediction, not measurement.
- **`git worktree add` is untested.** It writes, so it was out of scope. Given the malformed
  `.git/refs/heads/._main`, **this is the single highest-value next test** — and it should be run *after*
  the `.git/` quarantine, not before.

---

## Open questions for the human

1. **May I quarantine the 54 `._*` files inside `.git/`?** Lowest risk, highest value, nothing scientific
   — and it is what is breaking `git fsck` today. I would do this slice first and alone.
2. **May I quarantine the 90 under `.claude/`?** This is the one the brief's precondition blocked. They
   are inert today and contain no CHLU data, but they sit against frozen artefacts. **Your call, not a
   spoke's.**
3. **Should `core.filemode` be set to `false`?** It removes 36 phantom modifications and cannot alter the
   pin. I held off only because parallel spokes may be reading `git status`.
4. **Does `git worktree add` succeed on this repo?** The hub's first action depends on it, and I would not
   run a writing command. Please have someone test it after (1).
5. **Can someone with a working `gcloud` compare the tarball's sha256
   (`c3cac259a18d50bbaf0eb2c0c9aac4e43cf955348a7aa7d7fd126856c39a8375`) against the published checksum in
   `gs://chlu-cloud-forgisprova/handoff/`?** This is the last unclosed link in the provenance chain.
6. **Should the bundler be fixed at source?** `COPYFILE_DISABLE=1 tar …` (or `--no-xattrs`) on
   `batMac.local` prevents all 588 for every future bundle. A `._*` line in `.gitignore` would also help,
   but `.gitignore` is tracked at the pin, so that needs a deliberate decision.
7. **`CONTEXT.md` is now stale on two points** — `WORKSPACE-READY.json` *has* been written (17:52,
   `status: NOT-READY`, Python 3.12.10 found), and "none are inside `.claude/`" is wrong. Worth correcting
   before other spokes read it as fact, as I nearly did.
