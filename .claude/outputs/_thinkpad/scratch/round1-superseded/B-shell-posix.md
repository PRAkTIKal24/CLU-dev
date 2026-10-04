# SPOKE B — POSIX/shell viability on Windows  **(READ-ONLY: diagnose, do not remediate)**

Read `.temp_cl/CONTEXT.md` first.
Report to: **`.temp_cl/reports/B-shell-posix.md`**

## Why you exist
The entire tooling surface of this project is bash. There are **zero `.ps1` and zero `.bat` files** in the
repo. Meanwhile the worktree is CRLF, `rsync` is absent, and WSL is not installed. Before the human
commits this machine to a route — **native Git Bash vs WSL2** — somebody has to establish what actually
breaks. That is you.

## ⛔ Hard constraint: you change NOTHING
You are a diagnostician. You produce a **plan with exact commands**, you do not execute the remediation.
- ⛔ No `git add --renormalize`, no `git reset --hard`, no `git checkout -- .`, no `dos2unix` on a tracked
  file, no writing `.gitattributes` into the repo.
- ⛔ No installing WSL. No installing anything at all. (Spoke A owns installs and is running in parallel.)
- ✅ You may copy tracked files into `.temp_cl/scratch/` and mangle **the copies** freely. That is how you
  test a fix without applying it.
- ✅ Read-only git is fine: `git ls-files --eol`, `git diff --stat`, `git check-attr`, `git config --get`.

## Investigate and report

### 1. The CRLF situation — scope it precisely
- Confirm the 333/334 dirty files are **purely** line-ending noise. Prove it, e.g. per-file
  `git diff --ignore-cr-at-eol --stat` or `git diff -w`, and report **any file with a real content diff**.
  If even one file has substantive changes, that is the single most important line in your report.
- Are there tracked **binary** or binary-ish files that a naive renormalise would corrupt? Enumerate them
  (`git ls-files` + check for `-text` / actual binary content). Note anything under `chlu/eval/_tsb_vendor`
  (vendored third-party, pinned) and any `.npz`/`.npy`/`.pkl`/`.h5`/data fixtures.
- Is there a `.gitattributes` anywhere? (Check repo root and subdirs.) Report present/absent.
- Write the **exact remediation command sequence** you would run, and the exact `.gitattributes` content
  you would add — as a fenced block **in your report and in `.temp_cl/scratch/proposed.gitattributes`**.
  Not in the repo.
- State the blast radius: after remediation, how many files change, and does it produce a commit the Mac
  side would have to absorb?

### 2. Do the `.sh` scripts actually fail?
There are 16: 13 under `scripts/csf3/`, 3 at `scripts/smoke_c3_*.sh`.
- Copy them to `.temp_cl/scratch/` and demonstrate the failure mode on the **copy** (`bash scripts/...` vs
  `bash <lf-converted copy>`). Quote the real error.
- Note: `bash script.sh` and `./script.sh` behave differently w.r.t. CRLF shebangs — test both, report both.
- Read each script and classify: which are **CSF3/Slurm-only** (irrelevant on Windows), which are
  **genuinely needed locally**. Give the table.

### 3. MSYS path mangling — the flagged unknown
Git Bash converts POSIX paths to Windows form when invoking native binaries. The hub wants a definitive
answer on whether this breaks the project.
- Demonstrate it concretely (e.g. `MSYS_NO_PATHCONV`, `MSYS2_ARG_CONV_EXCL`, a `/c/...` path passed to a
  native `.exe`).
- Grep the repo for places a path crosses the bash→Python boundary: env vars, `argparse` paths, CLI
  invocations in `Makefile` and `scripts/`. Which of those would be corrupted?
- Note: the workspace path contains **a space and a hyphen** (`OneDrive - Forgis`). Test quoting behaviour
  through that path specifically — it is a likely failure source independent of MSYS.

### 4. `rsync` and `make`
- `rsync` is absent. What in the repo calls it? (`grep -rn rsync`.) What is the Windows-native substitute
  (`robocopy`? `scp`? Git Bash `cp -a`?) and does it preserve what the callers need?
- Is `make` available in Git Bash? Read the `Makefile` and report which targets would work here and which
  would not.

### 5. The route recommendation
Give a **clear recommendation: native Git Bash or WSL2**, with the reasoning and the cost of each. Include:
- What WSL2 would cost (install, disk, whether the repo would need to live inside the WSL filesystem for
  acceptable I/O — note the repo is currently on a OneDrive Files-On-Demand path, which WSL accesses over
  the slow `/mnt/c` 9p bridge).
- What native Git Bash leaves permanently broken, if anything.
Do not hedge into "it depends" — state a recommendation and what would change your mind.
