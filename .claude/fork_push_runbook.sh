#!/bin/bash --login
# =====================================================================
# fork_push_runbook.sh — push local work to the two private forks.
# SELF-GUARDING + WORKING-TREE-SAFE. Prepared by Hub 2026-07-19.
#
#   Repo A  clu      git@github.com:PRAkTIKal24/CLU.git      — CODE ONLY, anonymized later
#   Repo B  clu-dev  git@github.com:PRAkTIKal24/CLU-dev.git  — internal: code + .claude/docs backup
#   origin           git@github.com:PRAkTIKal24/CHLU.git     — PUBLIC, FROZEN, never pushed
#
# ⚠⚠ THE BUG THIS VERSION FIXES (hit 2026-07-19, fully recovered):
# The first version did `git checkout -b internal-state; git add -f .claude docs;
# git commit; git checkout main`. Because .claude/docs are GITIGNORED on main,
# `git checkout main` DELETED them from the working tree (tracked on
# internal-state, absent on main => checkout removes them). It wiped the 43MB
# program brain from the working tree (recovered from the branch afterwards with
# `git checkout internal-state -- .claude docs && git reset HEAD -- .claude docs`).
# FIX: never check the branch out in the main worktree. Build the internal-state
# commit with a TEMPORARY INDEX via plumbing (read-tree/write-tree/commit-tree).
# The working tree is NEVER touched.
# =====================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --------------------------------------------------------- 0. preconditions
BR=$(git rev-parse --abbrev-ref HEAD)
[ "$BR" = "main" ] || { echo "ABORT: not on main (on '$BR')."; exit 1; }
git diff --quiet && git diff --cached --quiet || { echo "ABORT: working tree dirty."; exit 1; }
echo "on main @ $(git rev-parse --short HEAD), clean."

# --------------------------------------------------------- 1. LEAK GUARD (hard)
LEAK=$(git ls-files .claude docs 'projects/*/' | grep -v -E '^projects/(README.md|.gitkeep)$' || true)
[ -z "$LEAK" ] || { echo "ABORT — would leak to the CODE repo:"; echo "$LEAK"; exit 1; }
echo "leak guard passed."

# --------------------------------------------------------- 2. remotes (idempotent)
git remote get-url clu     >/dev/null 2>&1 || git remote add clu     git@github.com:PRAkTIKal24/CLU.git
git remote get-url clu-dev >/dev/null 2>&1 || git remote add clu-dev git@github.com:PRAkTIKal24/CLU-dev.git

# --------------------------------------------------------- 3. push CODE to both
# .claude/docs/checkpoints are gitignored => not in the tree => not pushed.
git push -u clu     main
git push -u clu-dev main

# --------------------------------------------------------- 4. internal-state -> clu-dev ONLY (plumbing; worktree-safe)
# Build the snapshot commit off `main` using a THROWAWAY index. EXCLUDE
# .claude/scratch (regenerable) AND .claude/data (re-downloadable public datasets:
# CAMELS 7.1G, N-CMAPSS 2.3G, C2W10 streams 700M). Sizes as of 2026-08-26:
# .claude = 15G total, scratch 4.5G, data 10G, irreplaceable brain ~250MB. THE MAIN WORKING TREE IS NEVER CHECKED OUT AWAY
# FROM — no deletion risk. Also drop >90MB stragglers outside scratch (GitHub 100MB limit).
TMPIDX=$(mktemp)
export GIT_INDEX_FILE="$TMPIDX"
git read-tree main                                              # start from main's code tree
BIG=(); while IFS= read -r f; do BIG+=(":(exclude)$f"); done < <(find .claude docs -type f -size +90M 2>/dev/null | grep -v '^\.claude/scratch/' || true)
git add -f .claude docs ':(exclude).claude/scratch' ':(exclude).claude/scratch/**' ':(exclude).claude/data' ':(exclude).claude/data/**' "${BIG[@]}"
TREE=$(git write-tree)
COMMIT=$(git commit-tree "$TREE" -p main -m "internal state snapshot $(date -u +%Y-%m-%dT%H:%MZ) @ $(git rev-parse --short main) (brain: .claude minus scratch + docs)")
unset GIT_INDEX_FILE; rm -f "$TMPIDX"
git push -f clu-dev "$COMMIT:refs/heads/internal-state"          # -f: rolling snapshot; clu-dev only
git update-ref refs/heads/internal-state "$COMMIT"              # move local ref WITHOUT checking it out

echo "DONE (working tree untouched)."
echo "  clu     = code(main)"
echo "  clu-dev = code(main) + internal-state(code + .claude/docs minus scratch)"
echo "NEVER: git push clu internal-state   |   git checkout internal-state  (in the main worktree)"
