# SPOKE E — AppleDouble cleanup, then the two untested git legs a hub depends on
**Sequential by design: §1 must complete and pass before §2 begins.** Core budget: negligible.

Read `_hub_scratch/CONTEXT.md` first.
Work in: `/c/Users/PratikJawahar/chlu/chlu-hub-bundle`
Report to: **`/c/Users/PratikJawahar/chlu/_hub_scratch/reports/E-cleanup-gitpreflight.md`**

## Why you exist
588 macOS AppleDouble `._*` files rode along in the tar (built on a Mac without `COPYFILE_DISABLE=1`).
54 are inside `.git/` and they are **live breakage**, not cosmetic: `git fsck` fails, and every git
command prints `error: index file .git/objects/pack/._pack-….idx is too small`. That matters because
**`git worktree add` is the hub's first action** (the sandbox *is* a worktree) and **`git bundle create`
is how code comes home**. Neither has ever been run on this machine. You clean up, then test both.

---

# §1 — The cleanup. APPROVED by the Head, with an exact scope.

## The approved scope, verbatim
> Delete any file whose **basename starts with `._`** AND which **does not appear in
> `BUNDLE_MANIFEST.json`**. Nothing in `.git/` is a scientific artefact, and the source repo was verified
> to contain ZERO such files — they were created **in transit** by macOS tar, so they carry no content.
> **Condition: `git fsck` must pass afterwards.** Report anything you delete outside `.git/`.

Corroborating evidence already established — you may rely on it, but re-verify the guard yourself:
582 of the 588 are exactly **163 bytes and byte-identical**; 6 are 263 bytes. They are AppleDouble v2
xattr-only records (`com.apple.provenance`, `com.apple.macl`) — **entry ID 1, the resource fork, is
absent**. Total payload ~15 KB across all 588. Nothing scientific is inside them.

Distribution: `chlu/` 232 · `tests/` 145 · **`.claude/` 90** · `.git/` 54 · `scripts/` 48 · root 16 ·
`projects/` 2 · `logs/` 1.

## ⛔ The one thing that must not go wrong
`.claude/outputs/gen-arms-a7-wiring/refusal_scores.json` is **quarantined in the scientific sense — never
quote it as evidence.** It is **one of the 57 sha256-pinned manifest files.** ⛔ **Do NOT move, rename or
delete it.** Deleting a pinned artefact is the single worst outcome available to you. The manifest guard
below exists to make that impossible — do not skip it.

## Procedure — enumerate, guard, snapshot, then delete
Deletion is now authorised, so `rm` is permitted. Even so, **never pipe `find` straight into a mutation.**

1. **Snapshot integrity BEFORE.** Compute sha256 of all 57 manifest files into `/tmp/before.sha`.
   Record `git rev-parse $(git rev-list --max-parents=0 HEAD)^{tree}` (must be `12831fe1…`).
2. **Enumerate to a list**: `find . -name '._*' -type f | sed 's#^\./##' | sort > /tmp/appledouble.list`.
   It should read **588**. If it does not, **stop and report** — the count changing means something else
   moved.
3. **HARD GUARD — abort on any manifest hit.** Check every listed path against `BUNDLE_MANIFEST.json`
   (e.g. `grep -Ff /tmp/appledouble.list BUNDLE_MANIFEST.json`). Any hit at all ⇒ **abort, delete
   nothing, report.** Expected result: zero hits.
4. **Delete**, in two clearly separated passes so the report can distinguish them:
   **(a) the 54 under `.git/`** — the approved, zero-risk slice that fixes a broken tool today;
   **(b) the remaining 534** — `chlu/`, `tests/`, `.claude/`, `scripts/`, root, `projects/`, `logs/`.
   ⭐ **Report the full list of everything deleted outside `.git/`**, grouped by directory, as the Head
   asked. A count is not enough for the `.claude/` ones — list them.
5. **PROVE nothing else changed.** Re-checksum all 57 into `/tmp/after.sha` and `diff` against
   `/tmp/before.sha` — must be empty. Re-check the root tree still equals the pin.
6. **The acceptance condition: `git fsck` must pass.** Run `git fsck --no-progress` and paste the full
   output. If it still fails, **say so plainly and stop** — do not start §2 on a repo that fails fsck.
   Also confirm `git status --porcelain` now shows **36 mode-only rows and 0 untracked**.

## Also do this — it removes 36 phantom modifications
```bash
git config --local core.filemode false
```
This changes only what git **compares**; it cannot alter the `100755` modes recorded in the index or the
pin tree. A permanently-dirty tree can block `git worktree add` and trains everyone to ignore `git
status`. Confirm the pin tree is unchanged afterwards and report the before/after `git status` counts.

---

# §2 — The two untested legs. ⛔ Only if §1's `git fsck` passed.

Both have never run on Windows. A hub will depend on both, one at hour 0 and one at hour 10.

## 2a. `git worktree add` — the hub's first move
The agent profile §0.2 says the hub's sandbox **is** a git worktree. Until §1, `.git/refs/heads/._main`
was a malformed ref. Test it for real, then clean up after yourself:
```bash
git worktree add /c/Users/PratikJawahar/chlu/_wt_probe -b probe/windows-worktree-test
# ... confirm the tree is populated, chlu/ present, a .sh file is LF-clean ...
git worktree remove /c/Users/PratikJawahar/chlu/_wt_probe
git branch -D probe/windows-worktree-test
git worktree list && git fsck --no-progress
```
Put the probe **outside** the bundle. Report whether it succeeded, any warnings, and whether the repo is
still fsck-clean after removal. ⚠ NTFS has no symlinks by default and git worktrees use a `.git` **file**
pointer rather than a link — confirm that mechanism works here.

## 2b. The return path — how ten hours of work comes home
Two independent legs, both untested on Windows.

**Results leg** — `scripts/bundle_hub_workspace.sh --return-pack`. It shells out to Python to build
`RETURN_MANIFEST.json` (a sha256 per file), excludes the verifier's `.workspace-write-probe.json` **by
name**, and builds the tar in two `-C` passes because bsdtar stops parsing options at the first file
operand. **Read the `--return-pack` section of that script before running anything**, and check whether it
offers a `--dry-run`. Use a **throwaway slug**, never a real hub slug. Write the output tarball **outside
the bundle** (e.g. `~/probe.return.tar.gz`). Report: does it run, is the manifest well-formed, do the
sha256s verify, and does Git Bash's `mktemp -t` / two-pass `tar` behave like macOS bsdtar?

**Code leg** — whole-branch bundle, ⛔ **never a range**:
```bash
git bundle create ~/probe.gitbundle <branch>
git bundle verify ~/probe.gitbundle
```
⚠ `git bundle create … main..<branch>` is documented to fail **only on arrival** — a thin bundle's
prerequisite is this repo's synthetic base commit, which the Mac has never seen. Take the ~3 MB. Confirm
`bundle verify` passes and report the size. Delete both probe artefacts when done and say that you did.

⛔ **Do not run `--return-ingest`.** That is the Mac's half of the round trip.

---

## Report shape
Lead with two one-line verdicts: **(1) is `git fsck` clean now?** and **(2) can a hub create a worktree
and get its results home on this machine?** Then §1's deletion accounting (with the full non-`.git/`
list), then §2's two legs with full output. End with numbered open questions.

⭐ Before you finish, **confirm no sub-agent of yours is still running.**

---

# §3 — ADDED. Re-run the verifier and measure `anchors_checked`. ⛔ Only after §1 passed fsck.

The Mac's plan says: *"Re-run the verifier. `anchors_checked` must be non-zero."* The hub's analysis says
that **cannot** happen on Windows with the currently-shipped `verify_hub_workspace.sh`, because
`_load_doc()` builds its `DOC_ANCHORS` key with `Path.relative_to()`, which yields **backslashes** here,
so the dict lookup always misses. **Your job is to settle that with a measurement, not an argument.**

1. Re-run the verifier exactly as `CONTEXT.md` §"How to open a working shell" specifies:
   ```bash
   bash scripts/verify_hub_workspace.sh
   ```
2. Confirm the cleanup and the `core.filemode` change did **not** disturb anything: `status` must still be
   `ready`, `n_checksum_mismatch: 0`, `n_missing: 0`, `n_load_failed: 0`, `code_is_the_pin: true`.
   ⭐ If `status` is no longer `ready`, that is the headline of your whole report — say so first.
3. ⭐ **Extract `anchors_checked` for every one of the 8 `open_with: "doc"` entries** in
   `WORKSPACE-READY.json` and report them in a table. Four of them (`AGENT_PROTOCOL.md`,
   `autonomous-hub-charter.md`, `agents/autonomous-hub.md`, `handover_context.md`) have anchors registered
   in `DOC_ANCHORS` and **should** report non-zero. The three `.claude/tasks/*.md` legitimately have none,
   so `0` is correct for those.
4. State plainly: **is `anchors_checked` non-zero for the four docs that have registered anchors — yes or
   no?** If no, your report is the evidence the Mac needs. Quote the numbers.
5. ⛔ **Do not attempt to fix it.** `verify_hub_workspace.sh` is sha256-pinned; editing it would break the
   manifest and the pin. The fix is a one-call upstream change (`.as_posix()`) and the humans own it.

Copy the fresh `WORKSPACE-READY.json` to `_hub_scratch/reports/_E-workspace-ready.json` so there is a
record independent of the bundle.

⛔ **Do NOT apply `hub-refusal.supplement.tar.gz`.** It is downloaded and checksum-verified at
`/c/Users/PratikJawahar/chlu/hub-refusal.supplement.tar.gz`, but applying it is a **separate, gated
decision** (Spoke H) that the human has not yet made. Leave it alone.
