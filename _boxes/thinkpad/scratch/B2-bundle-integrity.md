# SPOKE B2 — Bundle integrity and macOS contamination  **(needs no Python — start immediately)**

Read `.temp_cl/CONTEXT.md` first.
Work in: `/c/Users/PratikJawahar/chlu/chlu-hub-bundle`
Report to: **`C:\Users\PratikJawahar\OneDrive - Forgis\Desktop\CLU-dev\.temp_cl\reports\B2-bundle-integrity.md`**

## Why you exist
Spoke A2 is installing Python so the verifier can run. But the verifier's checksum stage is *itself*
Python, so **nothing about this 2.0 GB Mac→GCS→Windows transfer has been validated yet**. You can validate
it right now with Git Bash coreutils (`sha256sum`, `stat`, `find`) and `git` — no Python required. If the
download is corrupt, A2 is chasing the wrong problem, and you will know first.

## Read-only by default
You verify and you plan. You do **not** delete, move or repair anything in the bundle — with one
tightly-scoped exception in §3, and only after its precondition is satisfied and written down.
No `git commit`, no `git reset`, no `git checkout --`. Nothing listed in `BUNDLE_MANIFEST.json` is ever
touched.

## 1. Independently checksum all 57 manifest files
`BUNDLE_MANIFEST.json` has `schema: "chlu-hub-workspace-bundle/1"` and a `files[]` array whose records
carry `path`, `bytes`, `sha256`, `loader`, `open_with`, `kind`. Parse it **without Python** — `jq` if Git
Bash has it, otherwise `grep`/`sed`/`awk` — and check every record for **presence, byte size, and sha256**
using `sha256sum`.

Report a table: `path | bytes expected/actual | sha256 match | notes`. Then state plainly: **did all 57
survive the transfer, yes or no?** If any failed, that is the headline of your report, and it means the
transfer — not the environment — is the problem.

## 2. Executable bits
The verifier's `_load_code()` asserts `os.access(p, os.X_OK)` and fails with *"lost its executable bit in
transit"*. Three manifest entries are code: `scripts/bundle_hub_workspace.sh`,
`scripts/verify_hub_workspace.sh`, `tests/test_hub_workspace.py`. Report `ls -l` and `stat` for each, and
for all 34 `.sh` files in the bundle. Does NTFS via Git Bash preserve a meaningful exec bit at all? Does
`test -x` agree with `ls -l`? Say whether you expect that assertion to pass or fail, and why.

## 3. The 588 AppleDouble `._*` files
The tar was built on macOS without `COPYFILE_DISABLE=1`, so 588 AppleDouble metadata files rode along:
~145 in `tests/`, ~160 under `chlu/`, ~47 under `scripts/`, some in `.git/`, 16 at the root.

Already established — do not redo: **none appear in `BUNDLE_MANIFEST.json`**, and **none are inside
`.claude/`**. So they are not frozen artefacts. Assess the actual blast radius:

- Does `pytest` collect any? `pyproject.toml` sets `python_files = "test_*.py"`, and `._test_x.py` does not
  match that glob — **verify** rather than trust (reason from the glob, or `--collect-only` if Python has
  appeared), and say which you did.
- Does any code path **glob a directory** rather than name a file? The `query_registry`, `control_set`,
  `feature_cache` and `npz_index` loaders open sets by scanning. Grep `chlu/` for `glob`, `iterdir`,
  `listdir`, `rglob` over artefact directories and report whether a `._*` sibling could be picked up.
- Do any in `.git/hooks` or `.git/objects` affect git? (`git fsck`, `git status` — read-only.)
- Are they zero-byte, or do they carry real resource-fork data?

**Then produce a quarantine plan. Execute it only if ALL of these hold:** you have confirmed none is in the
manifest, none is under `.claude/`, and you **move** them to
`/c/Users/PratikJawahar/chlu/_appledouble_quarantine/` preserving relative paths (never `rm`, so it stays
reversible). Re-run the `git status` and checksum checks afterwards to prove nothing else changed. If any
precondition fails, **do not execute — just write the plan.** Report which you did.

## 4. Git state — the pin
The verifier asserts the bundle's **root commit's tree** equals `code.pin_tree` in the manifest
(`12831fe1be17d04da027dcbb6e45b4dff6f1ef95`), because the hub's sandbox is a git worktree and a plain
`git archive` directory cannot hold one. Check it yourself, read-only:

    git -C . rev-list --max-parents=0 HEAD
    git -C . rev-parse <that>^{tree}
    git -C . rev-parse --short HEAD ; git -C . rev-parse --abbrev-ref HEAD
    git -C . status --porcelain | head -40
    git -C . config --get core.autocrlf

Report whether the tree matches the pin. **Also check `core.autocrlf` inside this repo specifically** — if
it is not `false`, a future git operation will rewrite the `.sh` files to CRLF and every script will die
with `bad interpreter: no such file or directory`, an error that tells you nothing about the real cause.
Setting a git config value is an environment change and **is permitted**; say if you set it. Also report
whether `git status` is clean, whether the AppleDouble files show as untracked, and whether `.gitignore`
covers them.

## 5. Tar and transfer provenance
The manifest records `built_on_host: batMac.local`, `built_from_root: /Users/user/Desktop/CHLU`,
`claude_bytes: 2102084732`, `tar_bytes: 10076160`, `code.source: git archive ae907466...`,
`n_tracked_files: 416`. Compare against what is actually on disk (`du -sb .claude`, and a count of
non-AppleDouble tracked files). Does the extraction account for every byte? Is the original tarball at
`/c/Users/PratikJawahar/chlu/chlu-hub-bundle.tar.gz` still present and intact? Note that a published
checksum in `gs://chlu-cloud-forgisprova/handoff/` cannot be fetched right now because `gcloud` is broken
for want of Python — record that as unverified rather than guessing.

## Report shape
Lead with a one-line verdict: **is this bundle byte-intact and safe to build on, yes or no?** Then the
sections above. End with numbered open questions for the human.
