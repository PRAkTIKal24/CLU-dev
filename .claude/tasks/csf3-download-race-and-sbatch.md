# Task: csf3-download-race-and-sbatch — concurrency-safe dataset download + sbatch convention alignment (engineer)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/csf3-download-race-and-sbatch.md`
- **Read first:** protocol (**§3.5 rebase local `main`; §5 flag-provenance**) · `.claude/csf_outs1.txt` (the actual failure logs — read them) · `chlu/data/industrial/base.py` (`download_file`, `_verify`, the `.part` handling) · `chlu/data/industrial/voraus_ad.py` (`fetch`) · `scripts/csf3/job_gpu_eval.sh` (the flagship recipe in its header) + `job_gpu_single.sh` + `job_gpu_array_seeds.sh` + `job_cpu_smoke.sh` + `setup_env_job.sh` · the HEPA sbatch example in this task (§B).
- **Git:** branch off local `main` (currently `ef944be`, which already added `#SBATCH -o logs/…`). Worktree per §3.2.
- **Context:** the first CSF3 flagship launch (6 parallel jobs, all with `--download`) failed 5/6 with **`sha256 mismatch`** and **`FileNotFoundError: …parquet.part`**. Root cause: `download_file` is **not concurrency-safe** — all jobs wrote to the *same* shared cache path `~/.cache/chlu/datasets/voraus_ad/…parquet.part`, corrupting each other; one job won the race (downloaded, renamed `.part→.parquet`), the rest died on the corrupted or moved partial. The Head's operational workaround (drop `--download`, use the now-cached parquet) unblocks the current runs — **this task is the durable fix so it can't recur (matters for TEP and any future parallel launch).**

## A. Concurrency-safe download (the core fix)
Make `download_file` (and thus every industrial loader) safe under N parallel `--download` processes sharing one cache. **Do NOT rely on `flock`** — the CSF3 home cache is on a networked filesystem where advisory locks are unreliable. Use **unique-temp + atomic-rename + check-final-first**:
1. **If the final verified file already exists, use it** (short-circuit — the common case once one job has downloaded). Verify its sha256; if it matches, return immediately, no download.
2. Else download to a **process-unique temp** (`mkstemp` / `…parquet.<pid>.<rand>.part`, never the shared `.part`), verify sha256 on *that* temp, then **atomically `os.replace(temp, final)`** (same-filesystem rename is atomic). Concurrent processes each write their own temp; the renames are last-writer-wins but every temp is an identical verified file, so the final is always valid. Clean up your own temp on failure.
3. Never leave a shared `.part` that another process can read or clobber. Remove the old shared-`.part` code path.
4. **Local test (do this — it's reproducible without CSF):** spawn ≥4 processes calling `download_file` on a small test URL (or a `file://`/local-HTTP fixture) into one shared cache dir concurrently; assert all succeed, the final file verifies, and no `sha256 mismatch`/`FileNotFoundError`. Add it as a test (mark network-dependent ones skippable; the concurrency logic can be tested with a local fixture server or a monkeypatched fetch).

## A2. Pre-fetch pattern (so we don't even download redundantly)
Even with A, N concurrent first-downloads each pull ~1 GB. Give the recipe a **download-once** path:
- Add a tiny **fetch-only** entry — e.g. `chlu eval --dataset voraus --fetch-only` (fetch + verify + exit, no scoring) or a `chlu data fetch <dataset>` subcommand. It populates the cache serially.
- **Update `job_gpu_eval.sh`'s header recipe** to the correct pattern: run the fetch-only once (or in `setup_env_job.sh`, which already runs on an internet-capable compute node), THEN launch the parallel eval jobs **without `--download`**. State this plainly so nobody repeats the 6-parallel-`--download` mistake.

## B. Sbatch convention alignment (fold in — same CSF3-infra area)
The Head's proven HEPA header (adopt the patterns, not the HEPA specifics):
```
#SBATCH -J msl_event
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 1                    # 1 TASK ...
#SBATCH -c 8                    # ... with 8 cores  (<=12/GPU on CSF3)
#SBATCH -t 0-06:00
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err
#SBATCH -a 1-24%2              # %2 = concurrency throttle
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<addr>
```
Apply to `job_gpu_eval.sh`, `job_gpu_single.sh`, `job_gpu_array_seeds.sh`, `job_cpu_smoke.sh`, `setup_env_job.sh`:
1. **⭐ `-n 8` → `-n 1 -c 8`** (1 task, 8 cores) — the real correctness fix; our current `-n 8` requests 8 tasks. Keep `-G 1` on GPU jobs, drop `-G` on the serial ones (already so).
2. **Add `-e logs/%x-%j.err`** (array: `%x-%A_%a.err`) alongside the existing `-o` — separate stderr. (Keep the `-o` I already added.)
3. **`-a 0-4` → `-a 0-4%4`** on `job_gpu_array_seeds.sh` — throttle to ≤4 concurrent (the CSF3 ≤4-GPU/user cap). Make the `%N` overridable.
4. **Email: PARAMETERIZE, do NOT hardcode.** ⚠ `pratik.jawahar@postgrad.manchester.ac.uk` is a name+affiliation identifier — hardcoding it leaks into `clu` (the anonymization base; see `repo_fork_plan.md` Layer-3). Add `--mail-type=END,FAIL` but set the address from an env var with an empty default (e.g. `#SBATCH --mail-user=` is invalid, so instead: only emit mail flags when `$CLU_MAIL` is set, or document `sbatch --mail-user=$CLU_MAIL …` at submit). Whatever you choose, **no email string in a tracked file.** Verify with `git grep -i 'pratik\|manchester\|@postgrad'` → clean.

## Acceptance
`download_file` concurrency-safe (unique-temp + atomic-rename + check-final-first, no shared `.part`), with a local concurrency test that fails on `main` and passes on the branch; a fetch-only path + the corrected download-once recipe in `job_gpu_eval.sh`; all sbatch scripts on `-n 1 -c 8` with separate `-e`, the array throttle, and mail parameterized (no hardcoded email — `git grep` clean); full suite green. Flag-provenance per §5. **Note in the report that the current re-runs (drop-`--download`) already work — this is durability, not a blocker.**
