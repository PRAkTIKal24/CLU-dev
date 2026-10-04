# TWO-MACHINE SETUP — Mac + Windows laptop

**Scoped 2026-09-12 at the Head's direction. Self-contained: you need no other document.**

⛔ **Make no changes until the Head approves a plan.** This thread is for deciding, then executing.

---

## 1. The goal (Head)

Work on CHLU from **this Mac and a separate Zed instance on a Windows laptop**, switching between them freely.

- ✅ **Code and program state must sync both ways.**
- ⛔ **Claude threads must NOT transfer.** Fresh threads on each machine.
- ⭐ **Custom user-level skills should be available on Windows.**
- Zed's native SSH is an option the Head is open to.

## 2. What is on each side of the line

| state | lives in | sync? |
|---|---|---|
| Code + `uv.lock` | the git repo | ✅ already solved — plain git |
| **The program brain** — charters, handovers, outputs, tasks, papers, submissions (~250 MB, 3035 files) | `.claude/` in the repo, **gitignored** | ✅ **this is the actual gap** |
| Datasets | `.claude/data/` — 10 GB, re-downloadable | ❌ never in git |
| Scratch | `.claude/scratch/` — 4.5 GB, regenerable | ❌ never |
| Claude threads, sessions, auto-memory | `~/.claude/` (outside the repo) | ❌ deliberately not — this is the Head's ask |
| Custom skills | `~/.claude/skills/` — 5 of them | ✅ wanted on Windows |

The clean split: **repo-`.claude/` is program state, `~/.claude/` is personal state.** Syncing the first without the second gives exactly "same project, fresh threads."

## 3. Verified state (2026-09-12 — re-verify before acting, it moves daily)

```
local main        9a4457d      5 commits AHEAD of clu-dev — unpushed
clu-dev/main      e4f3d71      code + uv.lock
clu-dev/internal-state  97f259e   brain snapshot, 2026-09-11, one-way force-pushed
clu/main          b0257bc      39 commits BEHIND clu-dev
origin/main       40c2f310     FROZEN public repo — never push
```

- `.gitignore` has `.claude/**` and `docs/**`; `uv.lock` **is** tracked.
- Three remotes: `clu` (code-only, becomes public at paper acceptance), `clu-dev` (daily dev + brain), `origin` (frozen).
- Backups are working: yesterday's snapshot used the corrected exclusions (`minus scratch/data`).

## 4. ⛔ Already analysed — do not re-propose

**The Head asked: "can we just remove `.claude/` from gitignore, and add it back when pushing to `clu`?"** Four independent reasons that fails:

1. **It was tried on 2026-07-20 and broke the workflow.** `.claude` was briefly tracked on `clu-dev/main`; pulling that onto the local code-only main polluted it with tracked `.claude`, fixed by `git reset --mixed`. The current code-on-main / brain-on-`internal-state` split is the result.
2. **`.gitignore` does not untrack.** Once tracked, re-ignoring does nothing. Cleaning it for `clu` needs `git rm --cached -r .claude` — a deletion commit — so the two repos' histories diverge permanently and every later push needs another toggle commit.
3. **Asymmetric, expensive failure.** `clu` goes public at acceptance. One push in the wrong toggle state publishes the internal brain. `push_repo.sh` has a hard leak guard specifically for this; a toggle routes around it.
4. **CSF3 pulls `clu-dev/main`.** Tracking `.claude` there drags 250 MB onto the cluster on every pull.

## 5. The recommended plan (Advisor's, for the Head to accept or replace)

Keep `.claude` gitignored in the code repo — leak-proof by construction. Put the brain on **its own orphan branch of `clu-dev`**, call it `brain`, whose *root is the contents of `.claude/`*. Clone that branch into the `.claude/` directory as a **nested repo**:

```
CHLU/            <- clu-dev, branch main    (code; .gitignore hides .claude/)
├── .git/
└── .claude/     <- clu-dev, branch brain   (own .git; parent never sees it)
    └── .gitignore   <- scratch/  data/  .DS_Store
```

Why this shape:
- **Still "clu-dev carries the brain"** — same repo, same credentials, one place.
- **`clu` never needs toggling.** Its tree stays permanently clean.
- **Two-way sync works.** Ordinary pull/commit/push; markdown and JSON merge fine.
- **It removes a real hazard.** `internal-state` today is a one-way force-pushed snapshot built with a temp-index trick, because checking that branch out normally once **deleted the whole brain from disk**. A nested repo has no such risk — its worktree *is* `.claude/`.

⚠ Setup asymmetry: the Mac's `.claude/` already has content, so it is converted in place (`git init` + remote + fetch + `reset --mixed`, which does not touch files). Windows is a plain clone into an empty directory.

### The alternative: Zed native SSH
Run a headless Zed server on the Mac, Windows Zed as a thin client. **Nothing syncs at all** — one copy, one venv, one VPN setup, agents run in the Mac's terminal. Needs Remote Login on the Mac and reachability (Tailscale makes it work from anywhere).

**Decision rule:** if the Mac is usually on and reachable when the laptop is used, **SSH is simpler and safer**. If the laptop must work with the Mac off (travel), the two-repo setup is required. Both can coexist — the brain branch costs nothing to have.

## 6. Skills

Five user-level: `create-spokes`, `full-sitrep`, `hub-handover`, `kill-stale-proc`, `quick-sitrep`. Four are CHLU-specific; `kill-stale-proc` is general.

Proposal: **canonical copies in `.claude/skills/`** (so they version and sync with the brain), **symlinked** to `~/.claude/skills/<name>` on each machine so they stay globally available. WSL symlinks behave like Linux — identical on both machines.

⚠ Cleanup while there: `.claude/skills/` holds both `hub-handover` and a `hub-handover 2` duplicate (a macOS artifact). Delete the duplicate.

## 7. Windows-side constraints

- **Use WSL2, not native Windows.** The stack is JAX/uv with Linux lockfile markers plus bash scripts (`#!/bin/bash --login`, rsync to CSF3, the backup runbook). Zed opens WSL projects natively. Clone **inside** WSL, not on the Windows filesystem — cross-filesystem I/O is slow and reintroduces CRLF and permission problems.
- `git config core.autocrlf false` — **CRLF silently breaks every bash script the moment it reaches CSF3.**
- `git config core.longpaths true` — some `.claude/outputs/` paths are deep.
- Add **only `clu-dev`** as a remote, named `clu-dev` (identical commands on both machines). ⛔ Do not add `origin` — nothing to gain, one more way to push to the frozen public repo by accident. Keep `clu` Mac-only; releases happen from one place.
- GlobalProtect VPN works on Windows; run `ssh`/`rsync` to CSF3 from inside WSL.

## 8. Switching discipline

Leaving a machine: **commit and push both repos** (code → `clu-dev main`; brain → `clu-dev brain`). Arriving: pull both. Conflicts only arise if both machines hold uncommitted edits to the same brain file — always pushing before switching avoids it entirely.

Agent worktrees (`CHLU-merge`, `CHLU-wt2`, …) are local scaffolding and need not exist on the laptop.

## 9. Open decisions for the Head

1. Primary mode — **two-repo sync**, **Zed SSH**, or both?
2. Brain as an orphan branch of `clu-dev`, or a separate `CLU-brain` repo? (Advisor: the branch — fewer moving parts, matches the Head's model.)
3. Skills: symlink from the brain repo, or plain copies kept in step by hand?
4. Retire the one-way `internal-state` snapshot once the brain branch works, or keep it as a second backup? (Advisor: keep — costs nothing.)

⚠ Independently of all four: **local `main` is 5 commits unpushed and `clu` is 39 behind.** Worth clearing before any of this starts, so both machines begin from the same place.

## 10. How to work in this thread

Answer, then stop. Simple language, no document citations at the Head. Numbered options with a recommendation when a decision is needed. ⛔ No changes to any repo, remote, or config until the Head approves.
