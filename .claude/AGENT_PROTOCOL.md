# CHLU Spoke-Agent Protocol (shared)

**Every spoke agent reads this file first, then `.claude/handover_context.md`, then its task file.** It defines how the hub-and-spoke program runs, where artifacts go, and — critically — the **git discipline that lets multiple agents run on separate threads in parallel without clobbering each other.**

---

## 1. The workflow you sit inside
- The **Hub** (research-lead agent) writes you a **task file** at `.claude/tasks/<slug>.md` and tells the **Head** (human) to spawn you.
- The **Head** spawns you on a fresh thread, points you at your task file, and chooses your model.
- You execute, then write your results to **`.claude/outputs/<slug>.md`** (and, if you edit code, to a git branch — see §3).
- The Head tells the Hub you're done; the Hub reviews your output file + branch and plans the next step.
- **You report to the Hub, not the Head.** Your output file is consumed by another agent — write dense, factual, structured notes, not human-facing prose.

## 2. Where artifacts go (all of `.claude/**` is gitignored)
- **Read:** `.claude/AGENT_PROTOCOL.md` (this), `.claude/handover_context.md` (living context), `.claude/tasks/<slug>.md` (your job).
- **Write your report:** `.claude/outputs/<slug>.md`. Create it; overwrite freely.
- **Scratch** (scripts, intermediate data, figures): under `.claude/scratch/<slug>/` or `.claude/outputs/<slug>/`. Never in the repo root or inside `chlu/`.
- **Rule: every artifact/report/note/scratch file you create lives under `.claude/`** so it stays untracked. The *only* things that belong in tracked files are actual production code changes (§3), and only if your task is code.
- **Do not edit `.claude/handover_context.md` directly** — that's the Hub's. Put any proposed handover updates in a `## Proposed handover updates` section of your output file.

## 3. Git & parallel-work discipline (assume you are NOT alone)
Other spokes may be editing the repo on other threads at the same time, possibly touching the same files. Discipline is mandatory for anyone who edits **tracked code** (research-only agents that just write to `.claude/` can skip this — nothing to commit).

1. **Never commit to `main`.** First action if you'll edit code: `git fetch`, then create a dedicated branch **`agent/<type>/<slug>`** off the base named in your task file (default `main`).
2. **Isolate overlap with a worktree.** If your task file flags that you share files with a concurrent agent — or if `git status` shows uncommitted changes you didn't make, or another agent's branch is checked out — work in a dedicated worktree so there's no filesystem-level collision:
   `git worktree add ../CHLU-<slug> -b agent/<type>/<slug>` → work there → **before removing the worktree, verify from the MAIN repo that your branch ref shows your commits** (`git -C <main-repo> log --oneline main..agent/<type>/<slug>`) — a worktree can be removed with the shared ref never having advanced (this lost 8 commits in wave-4, recovered only because the report logged the hashes) → `git worktree remove ../CHLU-<slug>`.
   **Never edit the shared main checkout while another agent's uncommitted work is present in it** — if you find yourself there, stop, create a worktree, and note the collision in your report (w4 precedent: a `git add` swept ~90 foreign lines into another agent's commit).
3. **Stay strictly in scope.** Edit only the files/functions your task names. Do **not** opportunistically reformat, rename, or refactor shared files — that is how parallel agents collide. Touch the minimal hunk; if you must edit a shared file, note exactly which lines and why.
4. **Atomic, labeled commits.** One logical change per commit. Imperative subject **prefixed with your agent tag**, e.g. `[experiment-engineer] fix broken chlu data figure8/sine import`. End every commit message with:
   `Co-Authored-By: Claude <noreply@anthropic.com>`
5. **Rebase before you finish, never clobber.** Rebase onto **your NAMED BASE (local `main`), NOT `origin/main`.** ⚠ **This repo's `origin/main` is stale** (frozen at `40c2f31`, 2026-07-02) while local `main` is ~80+ commits ahead and **deliberately unpushed** (anonymity strategy) — `git rebase origin/main` will try to replay dozens of foreign commits and conflict in files outside your scope. If you hit that, `git rebase --abort` (your true base hasn't moved, so rebase-onto-`main` is a no-op) and report it. Resolve in-scope conflicts only; STOP and report anything outside your scope — do not guess.
6. **Forbidden:** force-pushing a shared branch; `git reset --hard` / `git checkout .` over anyone's uncommitted work; committing secrets or large binaries; editing `.git` config; pushing or opening PRs unless the task explicitly says so. Leave your branch for Hub/Head review.
7. **Report your git footprint** in your output file: branch name, commit hashes, files touched, tests/commands run, and any unresolved conflicts.

## 4. Environment facts (this machine)
- Env/deps via **uv**: `uv sync`; run things with `uv run ...`. Tests `uv run pytest -q`; lint/format `uv run ruff check` / `ruff format`.
- **⚠ Worktree venvs (w6 lesson):** a fresh `uv sync` inside a worktree can resolve NEWER packages than the main venv (w6: JAX 0.9.0 → 0.10.2 flipped one bit-level lattice test). **Preferred: reuse the main venv** — run `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …` with cwd in the worktree (minus-the-physics precedent). If you must sync, use `uv sync --frozen` and report the resolved JAX version in your flag-provenance table.
- **JAX cold-start is pathologically slow here (~20+ min for even `chlu --help`/pytest collect).** Budget for it, keep the session warm, use `--quick` for smoke runs, and don't mistake a slow import for a hang.
- No `timeout` binary on this macOS. `projects/*/` and `.claude/**` and `docs/**` are gitignored.

## 5. Output-file format (write this to `.claude/outputs/<slug>.md`)
```
# <slug> — <agent-type> report
Task + acceptance criterion: <one line>
Status: done | partial | blocked
What I did: <bullets>
How I verified: <commands + observed output; real numbers, not claims>
Findings/results: <the substance>
Git footprint: branch, commits, files touched  (omit if no code changes)
Open questions / follow-ups / risks:
## Proposed handover updates (for the Hub)
```

**Flag-provenance rule (mandatory since 2026-07-07, critique M4):** every quantitative result in your report carries a **flag-provenance table** — commit hash, seed(s), and ALL non-default config flags in effect (lyapunov mode, langevin_noise, anchor, epochs, sleep_frequency/steps, kinetic mode, γ, dt, …). Training-config-sensitive results without this table will be sent back. The papers inherit these tables; reviewers reproducing across sections must not find apparent contradictions.

**Pre-registration rule (mandatory since 2026-07-10, w14 review; Head-approved).** If your task's acceptance criterion is a **measured ratio, exponent, slope, or law**, you must write `PREREG.md` into your output directory — stating the predicted values and *how they were derived* — **before running the harness that measures them.** Commit to a number, then measure it.

*Why this exists (the failure it prevents).* The flag-provenance rule governs **tables**. It does not govern **predictions and parenthetical asides** — and that is exactly where a wrong number lived and propagated for two waves: `t-lever-forgetting` §7 predicted a "≈13.9× memory vault" from an assumption (a locally-thermalized friction hole) that the shipped code does not implement. It reached the running log and `future_work.md` as fact. `v5-gate` pre-registered **both** competing hypotheses, measured `107.77 ± 4.78×`, and **rejected the 13.9× by a factor 8.11**. In the same wave `rb-bound-trained`'s prereg caught a mislabelled residual coefficient (`((2−γ)/2)²(εμ/γ)²`, not bare `(εμ/γ)²`). A pre-registered prediction that survives is evidence; one that fails is a *finding*. An un-pre-registered agreement is neither.

*Corollary — reconciliation lists need an owner.* If your report contains a "downstream reconciliation list" (sites that must change because of what you found), say so in your report's **first 10 lines**. The Hub converts it into an explicit curator or engineer task at the review that accepts your report. The "2.6" retraction sat live for two waves because it was buried inside one agent's report with no owner and no wave-boundary check.

## 6. Honesty
If tests fail, a run diverges/NaNs, or you couldn't finish — say so, with the evidence. A truthful partial result beats a confident wrong one. Thread seeds and report the exact config you used; research must be reproducible.

## 7. Dial declaration (w24 Head ruling — binding on every task from w25 on)
Every task file the Hub writes opens with a **DIAL DECLARATION** block, and every spoke report echoes it before its first result:
- **Dial:** which of the four dials the claim addresses (admission · lifetimes · isolation · compute-adaptive reads) — or "none: instrument/recon/theory."
- **Laundering control:** the trivial-substitute control that must run alongside every performance number.
- **Falsifies the claim:** the measured outcome that kills it.
- **Does NOT falsify the claim:** the outcomes that are known confirmations of standing theorems, stated in advance — e.g. *losing to an oracle or classical method on a metric-native protocol is the metric-native-ceiling theorem, not news, and does not falsify a dial claim.*
The claim rule is applied when the task is **scoped**, not at review. A verdict written against an axis the declaration excluded is non-compliant (w24 precedent: the R3 "leaderboard NO" mis-score).
