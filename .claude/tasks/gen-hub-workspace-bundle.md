# Task: gen-hub-workspace-bundle — ⭐ three hubs must run on a SECOND MACHINE. `git push` does not get them there.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-hub-workspace-bundle.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ **`scripts/csf3/stage_artefacts.sh` — it already derives the artefact set FROM THE LOADERS and checksums it; you are REUSING that, not rewriting it** · `.claude/autonomous-hub-charter.md` §A.4 (the pin) · `.claude/agents/autonomous-hub.md` §0 (the sandbox)
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** ⭐ the PIN `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e` · **Branch:** `agent/experiment-engineer/gen-hub-workspace-bundle` · **Worktree:** `../CHLU-bundle`
- ⚠ **Budget ≈ 1 day.** ⛔ No sub-agents. ⛔ No science, no arm, ⛔ no number that enters any table.

## ⭐ DIAL DECLARATION
- **Dial:** ⛔ **none — infrastructure.** **Falsifies:** nothing.
- **Laundering control:** ⭐ **extraction to a DIFFERENT absolute path on this machine** — the only way to prove
  portability without the second machine.

## The measured problem
Three autonomous hubs are to run on a **second machine (Windows + Zed)**. ⛔ **Pushing the repo does not carry
what a hub reads.**
1. ⛔ **`.gitignore:22` excludes `.claude/**`.** A push carries **no** charter, **no** agent profile, **no** hub
   task files, **no** artefacts. ⚠ The one existing mechanism, `clu-dev/internal-state`, is a snapshot from
   **2026-09-14** — it predates this entire week.
2. ⛔⛔ **JAX HAS NO NATIVE WINDOWS SUPPORT.** A hub imports `chlu` locally even when its compute is on GCloud.
   ⇒ ⭐ **WSL2, not native Windows.** Establish and document this; ⛔ do not assume it works.
3. ⛔ **45 task files hardcode `/Users/user/Desktop/CHLU`**, and the charter's allow-listed output path is an
   absolute macOS path. On the second machine **every one of them is wrong.**
4. ⚠ Sizes: `.claude/data` **11 G**, `.claude/outputs` **9 G** — ⛔ **most of which a hub does not need.**
   The gen dataset is **325 M and re-downloadable**.

## The one job
**Produce a portable hub workspace bundle, and prove it works from a different root.**

## Scope
1. ⭐⭐ **Determine the MINIMUM set.** ⛔ Not 20 GB. A hub needs: the code at the pin · `.claude/autonomous-hub-charter.md`
   · `.claude/agents/autonomous-hub.md` · `.claude/AGENT_PROTOCOL.md` · `.claude/handover_context.md` ·
   its own `.claude/tasks/hub-<target>.md` · and the **frozen artefact set**.
   ⭐ **Derive the artefact set with `stage_artefacts.sh`'s existing manifest logic — from the LOADERS, ⛔ never a
   directory listing.** ⛔ Do not ship `.claude/scratch/`, ⛔ not other slugs' outputs, ⛔ not `.claude/data/` bulk.
   ⭐ **Report the total size.** If it exceeds ~2.5 GB, say what is in it and why.
2. ⭐ **`scripts/bundle_hub_workspace.sh` (new)** — builds the bundle with a **manifest + sha256 per file**,
   idempotent, `--dry-run`. ⭐ Reuse the manifest format so the two platforms stay comparable.
3. ⭐⭐ **PATH PORTABILITY.** Introduce **`$CHLU_ROOT`** (and the derived `CHLU_*` roots) as the single source of
   truth, with the macOS path as its default so ⛔ **nothing on this machine changes behaviour.**
   ⛔ **You do NOT edit `.claude/autonomous-hub-charter.md`** — it is Head/Advisor-confirmed. ⭐ **Instead: emit the
   exact list of absolute-path lines in files you do not own, as a copy-able diff for the Hub to apply.**
   ⛔ You also do not fix the 45 legacy task files — ⭐ **only the files a hub actually reads** are in scope.
4. ⭐ **`scripts/verify_hub_workspace.sh` (new)** — run on the far side: re-checksums, **and OPENS every artefact
   through its real loader** (the `verify_artefacts.sh` lesson: a byte-perfect file that will not load is still a
   dead run), plus a `chlu` import and one targeted test. Writes `WORKSPACE-READY.json`.
5. ⭐⭐ **THE RETURN PATH — ⛔ do not forget this.** Three hubs will produce results **on that machine**. State
   exactly how `.claude/outputs/<hub-slug>/**` gets back here, ⛔ and never by a path that can overwrite anything.
   ⚠ **A result that is not synced back does not exist.**
6. **Dataset:** ⛔ not bundled. ⭐ Document the re-download + **sha256 assertion**
   (`0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785`).

## ⛔ You cannot test the thing that matters — so test the thing you can
⛔ **There is no second machine here.** ⭐ **Extract the bundle to a DIFFERENT ABSOLUTE PATH on this machine**
(e.g. `/tmp/chlu-hub-test/`), set `CHLU_ROOT` to it, and boot a **hub-shaped smoke**: read the charter, read a hub
task file, **write to the allow-listed output path**, import `chlu`, run one targeted test.
⭐ **That is the real acceptance.** ⛔ If it only works from `/Users/user/Desktop/CHLU`, it is not portable and the
bundle has failed.

## Files you own — and nothing else
- `scripts/bundle_hub_workspace.sh` **(new)** · `scripts/verify_hub_workspace.sh` **(new)** ·
  `tests/test_hub_workspace.py` **(new)** · `.claude/agents/autonomous-hub.md` (⭐ **path portability only** —
  ⛔ do not touch its rules) · `.claude/outputs/gen-hub-workspace-bundle/**`
⛔ **Not `.claude/autonomous-hub-charter.md`** (propose a diff, §3), ⛔ not `.claude/tasks/**` other than reading,
⛔ not `scripts/csf3/**` (⭐ read it, reuse its logic, leave it alone), ⛔ not `chlu/**`, ⛔ not `chlu/config.py`,
⛔ not `.gitignore`.

## Acceptance
1. ⭐⭐ **The bundle extracted to a DIFFERENT root, and a hub-shaped smoke passed there** — charter read, task file
   read, output written to the allow-listed path, `chlu` imported, one targeted test green. ⛔ Not a dry run.
2. The minimum set justified, **with its total size**. ⛔ No `scratch/`, no foreign slugs, no dataset bulk.
3. `verify_hub_workspace.sh` **demonstrated FAILING** two ways: a missing file, and a **byte-perfect file that will
   not load**. ⛔ A checker that cannot fail proves nothing.
4. ⭐ **The WSL2 finding stated plainly** — tested how far you could, and ⛔ explicitly `NOT RUN` where you could not.
5. The path-portability diff for the charter, copy-able, for the Hub to apply.
6. ⭐ **The return path documented and demonstrated** in the safe direction (no overwrite).
7. Targeted tests green, **passed / skipped / failed**. A **cost line**. ⛔ No GPU — this is plumbing.

## What would prove this spoke wrong
- ⭐⭐ **It only works from the original path.** Then nothing was solved and three hubs fail on their first write.
- ⭐ **You shipped 20 GB** because it was easier than deriving the minimum.
- ⭐ **You edited the charter.** It is Head/Advisor-confirmed; you propose, the Hub applies.
- ⭐ **No return path**, so three hubs produce results that never leave that machine.
- ⭐ **You assumed native Windows JAX works.** It does not, and discovering that on the second machine costs a day.
