# Task: gen-bundle-repairs — ⛔ three defects found by the Windows hub. One is a safety check that does NOTHING.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-bundle-repairs.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `scripts/bundle_hub_workspace.sh` · `scripts/verify_hub_workspace.sh` · `.claude/tasks/hub-refusal.md` (the charter whose artefacts are missing)
- **Base:** the Hub's tip · **Branch:** `agent/experiment-engineer/gen-bundle-repairs` · **Worktree:** `../CHLU-bundlefix`
- **Env:** main venv. ⚠ **Budget ≈ 4 hours. LEAN.** ⛔ No PREREG. ⛔ No sub-agents. ⛔ No number enters any table.

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — correctness repairs to tooling.** **Falsifies:** nothing.
- **Laundering control:** ⭐ **each fix is demonstrated FAILING before it is demonstrated passing.**

## ⛔⛔ DEFECT 1 — A SAFETY CHECK THAT SILENTLY DOES NOTHING (fix this first)
`verify_hub_workspace.sh`'s `_load_doc()` builds its lookup key with `Path.relative_to()`, which yields
**backslashes on Windows**, while `DOC_ANCHORS` is keyed with **forward slashes**. ⇒ ⛔⛔ **the dict lookup ALWAYS
misses, `missing` is ALWAYS empty, and every doc returns `loaded: true` having inspected ZERO anchors.**
⭐ **On Windows, check #3 degrades to "the file is non-empty."** ⛔ **A truncated charter — the exact thing the
check exists to catch — sails through reporting success.**
⭐ **Fix:** normalise the key (`.as_posix()`), ⛔ not by loosening the check.
⛔⛔ **ACCEPTANCE IS A TEST THAT FAILS ON A WINDOWS-STYLE PATH BEFORE THE FIX AND PASSES AFTER.** ⛔ Fixing it by
inspection is precisely what this defect proves we cannot rely on.
⚠ **Sister bug:** bare `read_text()` hits **cp1252** on Windows. ⭐ Set `encoding="utf-8"` **in the code** —
`PYTHONUTF8=1` is a good environment workaround but ⛔ the code should not depend on it.

## ⛔ DEFECT 2 — THE BUNDLE SHIPS macOS RESOURCE FORKS INTO `.git/`, AND `git fsck` FAILS
54 AppleDouble (`._*`) files reached the far side: `refs/heads/._main` is a **malformed ref**, seven
`.git/objects/XX/._<sha>` are **bad sha1 files**, and `._pack-*.idx` **shadows the real pack index on every git
call.** ⚠ **Measured: our local `.git` has ZERO such files** ⇒ ⛔ **they are created in transit by macOS `tar`,
which bundles resource forks unless told not to.**
⭐ **Fix:** `COPYFILE_DISABLE=1` on the tar path (and `--no-xattrs` where supported). ⭐ **Verify by building a
tar, extracting it to a fresh dir, and running `git fsck` — it must pass and the `._` count must be 0.**
⚠ **This is on the hub's critical path:** `git worktree add` is a hub's literal first move and
`git bundle create` is how its code comes home; both write refs and read the object store.

## ⛔ DEFECT 3 — THE BUNDLE OMITS THE ARTEFACTS `hub-refusal`'s OWN CHARTER DEPENDS ON
`.claude/tasks/hub-refusal.md` says it is "the cheapest unclaimed target" because it scores off artefacts that
already exist. ⛔ **Neither is in the bundle**, so on the far side it is the **most expensive**, not the cheapest:
| needed | size |
|---|---|
| `.claude/outputs/gen-read-cell-arm/runs/` (energy traces, settled states, decoded tuples) | **207 MB** |
| `.claude/outputs/gen-val-split-fix/refusal_scores.json` (the repaired-split rivals) | **2.8 MB** |
⛔ **`gen-arms-a7-wiring/refusal_scores.json` is SUPERSEDED — ⛔ never ship it as the rival row**, and if it is
already on the far side, say so in your report so the Hub can have it quarantined.
⭐ **Fix:** a **per-hub artefact requirement** the bundler honours, so a hub charter cannot again promise inputs the
bundle does not carry. ⛔ Do not hardcode two paths and move on — ⭐ **make the omission impossible to repeat.**
⭐ **Also produce a SUPPLEMENT tarball** (`--supplement`) carrying just these, so a working far-side workspace is
topped up ⛔ **without rebuilding 2.1 GB and without disturbing a hub that may already be running in it.**

## Files you own — and nothing else
- `scripts/bundle_hub_workspace.sh` · `scripts/verify_hub_workspace.sh` · `tests/test_hub_workspace.py`
- `.claude/outputs/gen-bundle-repairs/**`
⛔ Not `.claude/autonomous-hub-charter.md`, ⛔ not `.claude/tasks/hub-*.md` (⭐ **read `hub-refusal.md` to derive
its requirement; the Hub edits charters**), ⛔ not `chlu/**`, ⛔ not `chlu/config.py`, ⛔ not `scripts/csf3/**`.

## Acceptance
1. ⭐⭐ **Defect 1: a test that FAILS on a Windows-style path before the fix and PASSES after.** Show both.
   ⛔ Not "I checked the anchors by hand."
2. ⭐ **Defect 2: tar → extract to a fresh dir → `git fsck` passes, `._` count is 0.** Demonstrated, ⛔ not argued.
3. ⭐ **Defect 3: a per-hub artefact requirement**, demonstrated by building `hub-refusal`'s bundle and showing both
   paths present — **plus a `--supplement` tarball** that tops up an existing workspace ⛔ without overwriting.
4. ⭐ **The bundler REFUSES to build a hub whose declared artefact requirement is unmet**, naming the missing path.
   ⛔ Silent omission is the defect; a loud refusal is the fix.
5. Targeted tests green, **passed / skipped / failed**. A **cost line**. ⛔ No GPU.

## What would prove this spoke wrong
- ⭐⭐ **You fixed defect 1 without a failing test first.** The whole defect is that inspection reported success
  while the check did nothing — repeating that method would be the same mistake in a new place.
- ⭐ **You hardcoded the two refusal paths** instead of making the class of omission impossible.
- ⭐ **The supplement overwrites a running hub's workspace.** ⛔ It tops up; it never replaces.
- ⭐ You "fixed" the AppleDouble problem by deleting files on the far side rather than stopping the tar creating them.
