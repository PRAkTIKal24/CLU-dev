# csf3-download-race-and-sbatch — experiment-engineer report
Task + acceptance criterion: make `download_file` concurrency-safe (unique-temp + atomic-rename + check-final-first, no shared `.part`) with a local test that fails on `main`/passes on branch; add a fetch-only download-once path + corrected `job_gpu_eval.sh` recipe; align all 5 sbatch scripts (`-n 1 -c 8`, separate `-e`, array throttle, parameterized mail — no hardcoded email); full suite green.
Status: **done**

> **⚠ FIRST-10-LINES OWNER FLAG (for the Hub):** during my rebase onto local `main` I discovered the **Head pushed a live commit `393800f` to `main`** ("csf3: align sbatch to HEPA conventions") that does the same sbatch alignment as my Part B **but hardcodes `--mail-user=pratik.jawahar@postgrad.manchester.ac.uk`** into all 5 scripts — the exact `clu`-anonymization leak the task forbids. I resolved the rebase conflict by keeping my parameterized (`$CLU_MAIL`, no in-repo address) side. **The Head must confirm they are OK submitting via `sbatch --mail-user=$CLU_MAIL …`** rather than the hardcoded address they just added. If this branch is NOT merged, `main` currently carries the leaked email (see Proposed handover updates).

## What I did
- **A (core fix).** Rewrote `download_file` in `chlu/data/industrial/base.py`:
  1. **check-final-first** — if `dest` exists and verifies, return it (short-circuit; the common case once one job has published). On a checksum mismatch of an existing final, fall through and re-download (handles a corrupt/legacy file).
  2. Download to a **process-unique** `tempfile.mkstemp(dir=dest.parent, suffix=".<pid>.part")` (same filesystem as `dest`), verify sha256 on *that* temp, then **`os.replace(temp, dest)`** (atomic same-fs rename; last-writer-wins is safe because every temp is an identical *verified* file).
  3. Own temp is `unlink`-ed on any failure; **removed the shared `.part` path entirely** — no peer-readable/clobberable partial. No `flock` (unreliable on the networked home FS, per task).
- **A test.** New `tests/test_download_concurrency.py`: a throttled localhost HTTP server + `multiprocessing` (spawn) pool of 4 workers downloading the same URL into one shared cache; asserts all succeed, final verifies, no stray temp/`.part` left. Plus a short-circuit test (verified final ⇒ no network touch).
- **A2 (fetch-only).** Added `chlu eval --fetch-only` (`chlu/cli/eval_cmd.py`): forces `--download`, fetches+verifies the cache serially, prints the verified path, exits 0 — no scoring. Rewrote `job_gpu_eval.sh`'s flagship recipe to the **download-once** pattern: `--fetch-only` ONCE serially, THEN the parallel eval jobs **without `--download`**, with a plain-language warning about the 6-parallel-`--download` mistake.
- **B (sbatch alignment).** All 5 scripts: GPU jobs → `-n 1 -c 8` (was `-n 8` = 8 tasks); added separate `-e logs/%x-%j.err` (array `%x-%A_%a.err`) alongside `-o`; `job_gpu_array_seeds.sh` throttled to `-a 0-4%4` (≤4 concurrent, overridable `sbatch -a 0-N%M`); mail **parameterized** via submit-time `--mail-user=$CLU_MAIL` documented in every header + `#SBATCH --mail-type=END,FAIL` (no address in any tracked file). Serial scripts (`job_cpu_smoke`, `setup_env_job`) get `-e` + mail-doc only (they stay single-core `serial`, no `-n 8` present).

## How I verified (real output)
- **Test fails on pre-fix code, passes on branch** — restored the original `base.py`, ran the concurrency test → reproduced the *exact* CSF3 failure:
  `AssertionError: ["FileNotFoundError: … voraus-fixture.parquet.part -> … voraus-fixture.parquet"]` → `1 failed`. Restored fix → `2 passed`.
- **fetch-only path** (direct call against a synthetic parquet fixture, network not needed): `fetch-only return code: 0`, printed `fetched + verified voraus -> …/voraus-ad-dataset-100hz.parquet`.
- **Targeted tests:** `tests/test_download_concurrency.py tests/test_industrial_loaders.py tests/test_cli_data.py tests/test_data.py` → **16 passed**; post-rebase `download+industrial` → **9 passed**.
- **Full suite:** `uv`-venv `pytest -q` → **328 passed, 7 warnings in 345s** (exit 0).
- **Lint:** `ruff check` on the 3 changed py files → *All checks passed*. (Skipped `ruff format`: those files were never ruff-formatted in-repo — format would expand pre-existing compact `add_argument` blocks and sweep foreign lines into my commit, §3.3; my additions match the surrounding compact style.)
- **Scripts:** `bash -n` on all 8 `scripts/csf3/*.sh` → ok; directive grep confirms `-n 1`/`-c 8`/`-a 0-4%4`/`--mail-type` present, no `--mail-user=<addr>`.
- **Anonymization grep:** `git grep -i 'pratik|@postgrad'` over `scripts/ chlu/ tests/` → **CLEAN**. (`manchester` still appears **only** as the pre-existing cluster hostname `csf3.itservices.manchester.ac.uk` in `README.md`/`push_repo.sh`/`sync_project.sh` — institutional infra required to connect, NOT introduced by me and out of scope to change; flagged below.)

## Findings/results
- Root cause confirmed exactly as the task described: pre-fix `download_file` wrote a **shared** `dest.parent/<name>.part`; N parallel jobs corrupted it (`sha256 mismatch`) or one renamed it out from under the rest (`FileNotFoundError: …parquet.part` / `… .part -> .parquet`). The new unique-temp+atomic-replace removes the shared mutable state entirely.
- **Durability, not a blocker:** the Head's operational workaround (drop `--download`, use the now-cached parquet) already unblocks the current re-runs. This branch prevents recurrence for TEP and any future parallel launch.
- A stray `--download` on a parallel job can **no longer** corrupt the cache after this fix — but the download-once recipe still avoids N redundant ~1 GB pulls.

## Git footprint
- **Branch:** `agent/experiment-engineer/csf3-download-race-and-sbatch` (off local `main`), rebased onto local `main` (now `393800f`), **not pushed, not merged** — left for Hub/Head review.
- **Commits (post-rebase hashes):**
  - `a42e9f0` make download_file concurrency-safe (unique-temp + atomic-rename) — `chlu/data/industrial/base.py`, `tests/test_download_concurrency.py`
  - `c1e1daf` add chlu eval --fetch-only (download-once path) — `chlu/cli/eval_cmd.py`
  - `a5978f6` csf3 sbatch: -n1 -c8, separate -e, array throttle, param mail — 5× `scripts/csf3/*.sh`
- **Files touched:** `chlu/data/industrial/base.py`, `chlu/cli/eval_cmd.py`, `tests/test_download_concurrency.py`, `scripts/csf3/{job_gpu_eval,job_gpu_single,job_gpu_array_seeds,job_cpu_smoke,setup_env_job}.sh`.
- **Rebase conflict:** yes, in-scope — the 5 sbatch `#SBATCH` blocks collided with the Head's live `393800f`. Resolved keeping my parameterized-mail side (removes the hardcoded email); non-conflicting regions auto-merged. No out-of-scope edits.
- **Worktree** `../CHLU-csf3-download` used per §3.2; branch ref verified from the MAIN repo (`git log main..agent/…` showed all 3 commits) **before** `git worktree remove`. Branch persists in the main repo.
- Used the main venv (`/Users/user/Desktop/CHLU/.venv`, JAX 0.9.0) per §4, not a fresh worktree sync.

## Open questions / follow-ups / risks
1. **Head decision needed:** is the hardcoded `--mail-user` in `main`'s `393800f` intentional? My branch removes it (per the task's explicit acceptance criterion). If the Head wants email, use `sbatch --mail-user=$CLU_MAIL …` (works with my `--mail-type=END,FAIL`).
2. **Pre-existing `manchester` hostname** in `scripts/csf3/README.md`, `push_repo.sh`, `sync_project.sh` — required to connect to CSF3, but is a Layer-3 identifier for `clu` anonymization. Out of my scope; needs a curator/Head call on whether to parameterize the host too.
3. `--fetch-only` reuses the existing `--download`/`--root`/`--variant` plumbing; for a non-voraus dataset it fetches whatever that loader's `fetch()` pulls. Not exercised on TEP/SKAB here (voraus was the failing case).

## Proposed handover updates (for the Hub)
- **New §7-CURRENT resolved item:** `download_file` was **not concurrency-safe** (shared `.part`) — caused the CSF3 flagship 5/6 `sha256 mismatch`/`FileNotFoundError`. **RESOLVED** on `agent/experiment-engineer/csf3-download-race-and-sbatch` (unique-temp + atomic-rename + check-final-first; test `tests/test_download_concurrency.py` fails on old code, passes now). Durable fix; matters for TEP + any future parallel launch.
- **New CLI knob:** `chlu eval --fetch-only` (download+verify+exit). Correct CSF3 pattern = fetch-once serially, then parallel eval jobs **without** `--download` (now documented in `job_gpu_eval.sh` header).
- **sbatch convention now:** GPU jobs `-n 1 -c 8`; separate `-o`/`-e`; `job_gpu_array_seeds` throttled `-a 0-4%4`; **mail parameterized** via `sbatch --mail-user=$CLU_MAIL` (no in-repo address).
- **⚠ ANONYMIZATION LEAK ON `main`:** commit `393800f` (Head, live) hardcodes `pratik.jawahar@postgrad.manchester.ac.uk` in all 5 sbatch scripts. My branch removes it; **if this branch is not merged, `main` ships the leaked email.** Recommend merging this branch (or cherry-picking `a5978f6`) to clear it. Also note the pre-existing cluster-hostname `manchester` refs in README/push_repo/sync_project (separate curator call).
