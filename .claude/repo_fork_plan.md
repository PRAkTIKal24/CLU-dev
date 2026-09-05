# Private-repo fork + anonymization plan

> **Status:** PLAN — awaiting Head decisions (§4) and the private-repo URL. Nothing pushed yet.
> **Author:** Hub, 2026-07-19, on the Head's request to "plan how to push all staged/committed code to a fresh private repo, easier to anonymize than the public CHLU repo."

## The problem, stated precisely
- `origin` = `git@github.com:PRAkTIKal24/CHLU.git` — **the public repo from the ICLR AI&PDE paper.** Frozen at `40c2f31`. Carries the Head's name in the URL, the history, and the published-paper association.
- Local `main` is **100 commits ahead, unpushed**, **all authored `Pratik Jawahar <…@users.noreply.github.com>`**, with `[experiment-engineer]`/`Co-Authored-By: Claude` in every message.
- **Two independent needs, currently unmet:**
  1. **De-risk.** 100 commits + the entire research program (`.claude/**`) live on **one laptop, unpushed.** Laptop loss = total loss. This is the urgent one.
  2. **Anonymizable submissions.** Anonymizing the *public* CHLU repo is hard (name in URL/history/paper). A **fresh private repo** we control from commit 1 is the clean base for per-submission anonymized snapshots.

## What this is NOT
- **Not a CSF3 blocker.** CSF3 uses `scripts/csf3/push_repo.sh` = **rsync to `~/scratch`**, not a git remote. Tomorrow's runs do not need the fork. (The fork's urgency is de-risk, not CSF.)
- **Not the public CLU release.** That still happens at acceptance/arXiv (repo public, full history, Zenodo v2). This is the *private dev origin* that precedes it.

## The plan (three layers)

### Layer 1 — the private dev origin (do ASAP; de-risk)
1. **Head creates a private GitHub repo** (their account). Name candidate: `CLU` or `clu-dev` (package stays `chlu` inside — no rename, no pickle breakage; the PyPI-`clu`-collision only matters at public-package time, not for a private dev repo).
2. Add it as a **second remote** (keep `origin`=public CHLU frozen; add `private`):
   ```
   git remote add private git@github.com:<HEAD>/<CLU-private>.git
   git push private main          # ships all 100 commits, history intact
   ```
   History keeping the Head's name is **fine here — the repo is private.** Anonymization happens only at snapshot time (Layer 3).
3. **Thereafter push to `private` routinely** (every wave-integration ff). Never push to `origin`.
4. ⚠ **`.claude/**` and `docs/**` are gitignored — they do NOT travel with this push.** That means the program brain (handover, ledger, matrix, all agent outputs, the paper drafts) is **still on one laptop only.** See §4 decision (b) — this must be solved separately or the de-risk is half-done.

### Layer 2 — research-state backup (decide in §4)
The tracked code is only half the program. `.claude/**` (handover, `claims_matrix.md`, `philosophy-synthesis.md`, `negative_results.md`, all `outputs/*`, the paper drafts under `.claude/papers/`) is the irreplaceable part and it is gitignored. Options (Head picks):
- **(a) A second private repo** `clu-research-state` tracking `.claude/**` + `docs/**`. Cleanest; versioned; never mixed with the anonymizable code repo.
- **(b) An orphan branch** `research-state` in the private dev repo holding those dirs. One repo, but risks accidental inclusion in a snapshot.
- **(c) Periodic tarball to Drive/rds.** Simplest, least versioned.
- **Hub recommendation: (a).** It keeps the anonymizable code repo pristine and the research state fully versioned. Two private repos, clean separation.

### Layer 3 — anonymized snapshots (build ~Aug, per submission; not now)
For each double-blind submission, generate a clean artifact **from the private dev repo**, never linking to it:
- **Squash to a single commit** (kills the 100-commit `[experiment-engineer]`/`Co-Authored-By` trail and the author history).
- **Strip author identity:** commit as an anonymous identity; scrub name/affiliation strings from tracked code. **Known identifier sites (keep current):** `pyproject.toml` (authors), `scripts/csf3/README.md` + `push_repo.sh` + `sync_project.sh` (Manchester/CSF3/username refs), and **⚠ `scripts/csf3/{job_gpu_eval,job_gpu_single,job_gpu_array_seeds,job_cpu_smoke,setup_env_job}.sh` — hardcoded `--mail-user=pratik.jawahar@postgrad.manchester.ac.uk`** (added 2026-07-20 at the Head's request for email-on-completion; each line is tagged `# (identifier - strip for anon submissions)`). The anon-snapshot script must blank the `--mail-user=` line (or the whole `#SBATCH --mail-*` pair). `git grep -iE "pratik|jawahar|pierini|manchester|cern|forgis|postgrad|<username>"` must return clean on the snapshot. **Cleaner alternative if wanted:** drop the hardcoded email and set `export SBATCH_MAIL_USER=…` in the CSF3 shell profile (Slurm honors it; zero email in tracked files).
- **Exclude** `.claude/**`, `docs/**`, `projects/**` (checkpoints may carry provenance), CSF3 usernames.
- **Delivery** per the venue: anonymous.4open.science proxy, or a squashed-history throwaway repo, or an OpenReview supplementary zip. **No real-repo link in the submission; self-citation in third person.**
- This is well-scoped engineering (a `make anon-snapshot` script + a deanonymization checklist test) → a spoke task **when submission is near (~Aug)**, not now.

## §4 — Head decisions needed
1. **Private repo host + name** — GitHub private under your account? Name (`CLU` / `clu-dev` / other)? Provide the URL and I'll prepare the exact `remote add` + `push` (I will not push until you confirm).
2. **Research-state backup** — Layer 2 option (a)/(b)/(c)? (Hub rec: (a), a second private repo for `.claude/**`+`docs/**`.)
3. **Cutover timing** — push `main` to `private` now (recommended: de-risk before scaling real-data results this week), keeping `origin`/public-CHLU frozen and untouched?
4. **Dev-commit identity** — keep authoring dev commits as-is (your name; private repo) and anonymize only at snapshot time? (Recommended — rewriting author on every dev commit is friction for no private-repo benefit.)

## Hub note on execution
Creating the repo needs your account (I can't). Once you give the URL and the go-ahead, the `remote add` + first `push` is three commands I can run — but **pushing is outward-facing and publishes, so I will not run it until you explicitly confirm the URL and say go.** The anonymization tooling (Layer 3) is deferred to ~Aug.
