# gen-cluster-stage-artefacts — experiment-engineer report

Task + acceptance criterion: build and prove a verified staging path that puts the gen artefact set on
CSF3 scratch, checksummed end to end — ⛔ the MECHANISM + the PROOF THE SET IS COMPLETE, **not** the transfer.
Status: **done** (laptop legs). ⭐ **Every cluster-side leg is `NOT RUN` — see §5; CSF3 needs 2FA and I cannot reach it.**

## ⭐ DIAL DECLARATION (echoed)
- **Dial:** ⛔ **none — plumbing.** No arm, no mechanism, no verdict, no number that enters any table.
- **Laundering control:** n/a — nothing is scored here. **Falsifies:** nothing.
- ⚠ It *did* reveal that the artefact set we believe is complete is **not reachable** as shipped: see the
  three-item **DOWNSTREAM RECONCILIATION LIST (§6) — it needs an owner at the review that accepts this report.**
- ⛔ No PREREG: the acceptance criterion is a mechanism, not a measured ratio/exponent/slope/law.

## What I did
- **Derived** the artefact set from the loaders (`exp_gen_*` resolvers + the bridge/store/decoder/classifier
  loaders), ⛔ not from a directory listing. Loader map in §2; three files that *look* like artefacts and are
  **deliberately excluded because nothing opens them** are named in §2.4.
- **`scripts/csf3/stage_artefacts.sh` (new)** — enumerates the set, refuses an empty **and** a partial one,
  is idempotent + resumable, has a working `--dry-run`, writes the manifest.
- **`scripts/csf3/verify_artefacts.sh` (new)** — GPU-less cluster job (`-p multicore`, ⛔ never `-p serial`)
  that re-checksums **and OPENS** every file through its real production loader, writing `STAGED.json`.
- **`tests/test_stage_artefacts.py` (new)** — 16 tests, including the two failure demonstrations.
- **`.claude/outputs/gen-cluster-stage-artefacts/stage_manifest.json`** — 44 files, **2,097,001,955 B (1.953 GiB)**.
- ⛔ **`push_repo.sh` NOT edited.** Its `--exclude '.claude/'` (line 42) is reported as a finding (§6.1) and left
  alone: changing a live sync path under a cluster user is how a working sync silently stops working.

## How I verified (commands + observed output)
```
bash -n scripts/csf3/stage_artefacts.sh ; bash -n scripts/csf3/verify_artefacts.sh        -> clean
CLAUDE_ROOT=…/.claude bash scripts/csf3/stage_artefacts.sh --dry-run
    -> 44 files, total 2097001955 bytes (1.95 GiB); nothing transferred
CLAUDE_ROOT=/tmp/emptyroot … --dry-run
    -> "the artefact set is EMPTY — refusing to run", exit 4, no manifest written
(seed-1 subset, decoder_v1_s1.pkl removed) … --dry-run
    -> names the file AND its loader, "Refusing to stage a partial set", exit 3
```
Verify job, run locally (`VERIFY_LOCAL=1`) against **real shipped artefacts**, main venv, `PYTHONPATH=<worktree>`:

| run | root | files | result | exit |
|---|---|---|---|---|
| clean | the real `.claude` (44 files, 1.95 GiB) | 44 | `status=complete checksum_match=True`, 0 missing / 0 mismatched / **0 failed to load** | 0 |
| ⭐ one byte flipped | seed-1 copy, 1 byte in `decoder_v1_s1.pkl` + 1 byte in `features.npy` | 18 | `status=failed checksum_match=False`, **2 mismatched, 1 failed to load** | 1 |
| ⭐⭐ byte-perfect, unloadable | seed-1 copy, `store_plan.json` schema `…/1`→`…/2` | 18 | `checksum_match=**True**`, `status=failed`, **n_load_failed=1**: `ValueError: not a store plan: 'chlu-gen-store-plan/2'` | 1 |

Evidence JSON kept at `.claude/outputs/gen-cluster-stage-artefacts/LOCAL-VERIFY-{clean,corrupt-byte,byte-perfect-unloadable}.json`.
⭐ The third row is the point: **a pure checksum gate calls that a complete stage.** Opening it is what catches it.
⚠ Note from the second row: the flipped byte in the *real* 6 MB `decoder_v1_s1.pkl` landed in an array payload and
`load_model` **still unpickled it happily** — only the sha256 caught that one. Neither check subsumes the other,
which is why `STAGED.json` carries `checksum_error` and `load_error` as **separate, non-clobbering** fields.

Tests (main venv, `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`, `PYTHONPATH=<worktree>`):
```
pytest tests/test_stage_artefacts.py tests/test_gen_launcher.py -q
    -> 55 passed / 0 skipped / 0 failed   (16 new + 39 pre-existing launcher tests)
pytest tests/test_stage_artefacts.py -q            (default random ordering, and again under -p no:randomly)
    -> 16 passed / 0 skipped / 0 failed  in both orderings
ruff check tests/test_stage_artefacts.py           -> All checks passed!
```
⚠ The new tests use a **synthetic** `.claude` mirror with real schemas, so the suite never depends on the
multi-GB shipped artefacts being present. ⛔ No test touches CSF3 and none can: `CSF3_USER` is stripped from
the test environment so a stray code path cannot attempt a transfer.

---

## 1. The defect, re-measured
`scripts/csf3/push_repo.sh:42` carries `--exclude '.claude/'`; `sync_project.sh` moves only `projects/<name>/`,
which no gen artefact is. ⇒ **Confirmed: before this branch there was no path that put a gen artefact on CSF3
scratch.** The green cluster smoke ran `SMOKE=1` with a trivial payload and proves the launcher starts, ⛔ not
that a stage module could load anything.

## 2. ⭐ THE ARTEFACT SET, DERIVED FROM THE LOADERS

### 2.1 Which of the launcher's five stages open what
`job_gpu_gen.sh` accepts `STAGE=encoder|decoder|store|cell|certificate`.

| STAGE | module | opens under `.claude/` |
|---|---|---|
| `encoder` | `chlu.experiments.exp_gen_encoder` | ⭐ **nothing.** It needs only the dataset root; `build_warm_start` fits its own encoder/store/decoder. Not a consumer. |
| `decoder` | `chlu.experiments.exp_gen_decoder` | ⛔ **THE MODULE DOES NOT EXIST** (§6.2). |
| `store` | `chlu.experiments.exp_gen_store` | the frozen bridge + the C set |
| `cell` | `chlu.experiments.exp_gen_cell` | the bridge, the **winner decoder**, the **external classifier** (+ its gate), the **A7 refusal scores**, LACE's inherited manifest, the store plan, the composition store |
| `certificate` | `chlu.experiments.exp_gen_certificate` | the bridge, the store plan (+ the shipped `store_manifest.json`), the C set |

### 2.2 The file → loader map (this is `stage_manifest.json`'s `loader` field, verbatim in spirit)
| file (relative to the staged root) | the module that opens it |
|---|---|
| `outputs/gen-feature-bridge/runs/<ds>_<px>/seed_<s>/bridge_manifest.json` | `exp_gen_store.bridge_dir` · `exp_gen_cell.run_seed` · `exp_gen_certificate.bridge_dir` |
| … `/encoder.pkl` | `chlu.utils.checkpoints.load_model`; `param_hash` asserted against the bridge manifest |
| … `/decoder.pkl` | `load_model` via `exp_gen_store.fit_seed` (`assert_decoder_hash`); also `resolve_decoder` when `decoder_source='bridge'` |
| … `/probe.pkl` | `load_model` via `exp_gen_store.fit_seed` / `exp_gen_cell.run_seed` |
| … `/feature_cache.json` | `chlu.core.gen_bridge.load_feature_cache` |
| … `/feature_rows.npy`, `/features.npy` | ⭐ **named BY `feature_cache.json`** (`rows_file` / `features_file`), read off disk by the stage script — that is how the set stays derived from the loader rather than from a guess about filenames |
| `outputs/gen-store-grouping/runs/<ds>_<px>/store_plan.json` | `chlu.core.gen_store.load_store_plan` via `exp_gen_cell.store_plan_path` / `exp_gen_certificate.store_runs_dir` |
| `outputs/gen-store-grouping/runs/<ds>_<px>/seed_<s>/store_manifest.json` | `exp_gen_certificate.run_seed` (the "shipped" record; **optional** — absent just means no bit-identity assertion) |
| `outputs/gen-read-compose/runs/<ds>_<px>/seed_<s>/store_s<s>.pkl` | `exp_gen_cell.resolve_store`. ⚠ **Absent ⇒ the cell silently RE-TRAINS a store** (`allow_store_training=True`) and its row stops being comparable to `gen-read-compose`'s. Staged as **required** for that reason. |
| `outputs/gen-decoder-lift/winner/winner_hashes.json` | `chlu.core.gen_dream.load_winner_hashes` via `exp_gen_compose.resolve_decoder` |
| `outputs/gen-decoder-lift/winner/decoder_v1_s<s>.pkl` | `load_model` via `resolve_decoder`; recipe **and** param hash asserted (`dec_recipe` still defaults to the FAILED `baseline`) |
| `outputs/gen-external-classifier/classifier_gate.json` | `exp_gen_cell.check_preconditions` — ⛔ `"gate"` must read `"PASS"` |
| `outputs/gen-external-classifier/seed_<s>/external_classifier.pkl` | `load_model` via `exp_gen_cell.load_external_classifier` |
| `outputs/gen-external-classifier/seed_<s>/classifier_manifest.json` | `load_external_classifier` (supplies the asserted `param_hash`) |
| `outputs/gen-arms-a7-wiring/refusal_scores.json` | `exp_gen_cell.check_preconditions` — ⛔ `"status"` must read `"complete"` |
| `outputs/gen-arms-a7-wiring/runs/historical_688128/seed_<s>/baselines_manifest.json` | `exp_gen_cell.lace_selection` — the INHERITED LACE configuration |
| `data/gen/<ds>/c_invalid.json` | `chlu.core.gen_certificate.load_c_requests` via `exp_gen_certificate.c_invalid_path`; `exp_gen_store.load_c_set` |

### 2.3 ⚠ TOTAL BYTES — and where they are
Axes `shapes3d × recon × seeds {0,1,2}` (the default; the script takes `--datasets/--pretexts/--seeds`):

| group | files | bytes |
|---|---:|---:|
| `gen-feature-bridge` | 21 | 2,061,938,615 (1966.4 MiB) |
| `gen-decoder-lift` | 4 | 18,209,583 |
| `gen-external-classifier` | 7 | 11,782,057 |
| `gen-arms-a7-wiring` | 4 | 2,977,410 |
| `gen-read-compose` | 3 | 1,610,787 |
| `gen-store-grouping` | 4 | 411,462 |
| `data/gen/shapes3d/c_invalid.json` | 1 | 72,041 |
| **TOTAL** | **44** | **2,097,001,955 B = 1.953 GiB** |

⭐ **98.3 % of it is three files**: `seed_0/features.npy` alone is **1,966,080,128 B** (`scope="full"`, 480,000 rows);
seeds 1 and 2 were written at `scope="used"` and are 38 MB each. ⚠ **Adding `--pretexts "recon cls"` roughly
doubles the total** (the `shapes3d_cls` bridge exists locally; its `store_plan.json` does **not**, so a `cls`
`cell`/`certificate` run would be refused by this script — correctly — until that plan is fitted).
⚠ **CSF3 scratch has no backup and a 3-month auto-cleanup.** Everything staged is re-stageable by re-running
the script (idempotent) or it is lost; nothing here is the only copy.

### 2.4 ⛔ Deliberately NOT staged (and why)
- **Image blobs** — `3dshapes.h5` (255 MB), the dSprites `.npz`/`.hdf5` (45 MB). Re-downloadable on a **compute**
  node (`chlu data <ds> --download`; login nodes have no internet). ⛔ Never stage a dataset.
  ⚠ *Correction to the task's premise:* `.claude/data/` is **11 GB total but `.claude/data/gen/` is only 325 MB**
  — the 10 GB lives elsewhere under `.claude/data/`. The exclusion still stands on the "re-downloadable" ground.
- **`factor_index.npz`, `query_sets.json`, `a7_redundant.json`** — derived, but ⛔ **no loader among the five
  stages opens them** (`query_sets.json`/`a7_redundant.json` are opened by `exp_gen_baselines`, which is not a
  launcher stage; `factor_index.npz` only by the `gen_splits` audit path). An artefact nobody can name a loader
  for does not go. ⚠ If a wave runs `PAYLOAD_MODULE=chlu.experiments.exp_gen_baselines`, those two rejoin the set.
- **`encoder_manifest.json`, `shared_decoder_hash.json`** (bridge seed dirs), **`ceiling.json` / `gate.json`**
  (bridge tag dirs), **`winning_recipe.yaml`**, every `.png` and `.log` — written, never read back by any stage.
  `encoder_manifest.json` is read only by `exp_gen_bridge` itself, which ⛔ never runs on the cluster (the bridge
  is consumed frozen and never rebuilt).

### 2.5 The layout, and the one variable that makes it work
Every gen loader's default is `<repo>/.claude/<…>`, so the set is staged as a **verbatim mirror of `.claude/`**
at `~/scratch/CHLU/artefacts/`, and one export makes every default resolve:
```
export CHLU_CLAUDE_ROOT=$HOME/scratch/CHLU/artefacts
export CHLU_GEN_DATA_ROOT=$CHLU_CLAUDE_ROOT/data/gen
export CHLU_GEN_BRIDGE_ROOT=$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs
export CHLU_GEN_STORE_RUNS=$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs
export CHLU_GEN_DECLIFT_DIR=$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner
```
(the manifest carries these in `env_for_the_job`; `verify_artefacts.sh` exports them itself so the verify job
proves the **same resolution** a real stage run gets).

## 3. ⭐ ONE COPY-ABLE BLOCK FOR THE HEAD
```bash
# ═══ 0. LAPTOP — on campus or GlobalProtect VPN, in /Users/user/Desktop/CHLU ═══
export CSF3_USER=<uom-it-username>                   # form mabcxyz1, NOT the email
git rev-parse HEAD                                   # note this hash — step 4 needs it
#   -> prints a 40-char commit

# ═══ 1. code + uv.lock (unchanged path; ⛔ it carries NO artefacts) ═══
scripts/csf3/push_repo.sh
#   -> rsync file list ending: done. uv.lock included: yes

# ═══ 2. the artefacts: plan first, then go ═══
scripts/csf3/stage_artefacts.sh --dry-run
#   -> 44 lines "<bytes>  <path>", then:
#      [stage] total: 2097001955 bytes (1.95 GiB)
scripts/csf3/stage_artefacts.sh
#   -> rsync of 45 files (44 + stage_manifest.json) into scratch/CHLU/artefacts/
#   -> ends "[stage] staged 2097001955 bytes. ⭐ NOW RUN THE VERIFY JOB"

# ═══ 3. CSF3 LOGIN NODE ═══
ssh $CSF3_USER@csf3.itservices.manchester.ac.uk
cd ~/scratch/CHLU && mkdir -p logs
sbatch scripts/csf3/setup_env_job.sh                 # skip if the venv is already built
#   -> slurm-<id>.out ends "[setup] DONE" with "jax 0.9.0 | chlu 0.2.4"

# ═══ 3b. the DATASET — ⛔ never staged; compute nodes have internet, login nodes do not ═══
sbatch -p multicore -n 1 -c 8 -t 2:00:00 -o logs/getdata-%j.out --wrap \
  'cd ~/scratch/CHLU && source scripts/csf3/env.sh && \
   chlu data shapes3d --download --data-root $HOME/scratch/CHLU/artefacts/data/gen/shapes3d'
#   -> "3dshapes.h5: 267573662 bytes sha256=…" then the split report
#   ⚠ THIS REWRITES c_invalid.json. That is why step 3c exists.

# ═══ 3c. LAPTOP again — re-stage (idempotent): restores the single-sourced C set ═══
scripts/csf3/stage_artefacts.sh
#   -> rsync sends c_invalid.json only if the cluster's copy now differs

# ═══ 4. VERIFY — ⭐ checksums AND opens. Nothing launches until this is green. ═══
#      (login node) — GPU-less, ⛔ never -p serial
sbatch scripts/csf3/verify_artefacts.sh
squeue --me
cat ~/scratch/CHLU/artefacts/outputs/gen-cluster-stage-artefacts/STAGED.json | head -20
#   -> "status": "complete", "checksum_match": true, "loaders_all_opened": true,
#      "n_files": 44, "n_missing": 0, "n_checksum_mismatch": 0, "n_load_failed": 0
#   ⛔ Anything else: re-stage the files STAGED.json names. Do NOT launch a stage run.

# ═══ 5. only now, a real stage run (⭐ GEN_GIT_COMMIT is exported HERE, on the login node;
#         ⛔ `git pull` on the cluster serves a stale fork — push_repo.sh is the only code path) ═══
export GEN_GIT_COMMIT=<the hash from step 0>
export ART=$HOME/scratch/CHLU/artefacts
DATASET=shapes3d PRETEXT=recon STAGE=cell PLAN=1 bash scripts/csf3/job_gpu_gen.sh
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=cell,ARMS=clu,SEEDS=0 1 2,\
GEN_GIT_COMMIT=$GEN_GIT_COMMIT,CHLU_CLAUDE_ROOT=$ART,CHLU_GEN_DATA_ROOT=$ART/data/gen,\
CHLU_GEN_BRIDGE_ROOT=$ART/outputs/gen-feature-bridge/runs,\
CHLU_GEN_STORE_RUNS=$ART/outputs/gen-store-grouping/runs,\
CHLU_GEN_DECLIFT_DIR=$ART/outputs/gen-decoder-lift/winner" \
  -a 0-2%2 -t 04:00:00 scripts/csf3/job_gpu_gen.sh
#   ⚠ The five CHLU_* exports are carried by hand because neither env.sh nor job_gpu_gen.sh
#     knows about the artefact root yet — see §6.3, which needs an owner.
```

## 4. Git footprint
- **Branch:** `agent/experiment-engineer/gen-cluster-stage-artefacts` (worktree `../CHLU-stage`, base `main @ 3eb9eb0`)
- **Commit:** `98c8534` — *"a verified staging path for the gen artefact set on CSF3"*
- **Files (all new, all mine):** `scripts/csf3/stage_artefacts.sh`, `scripts/csf3/verify_artefacts.sh`,
  `tests/test_stage_artefacts.py`. ⛔ **Zero edits** to `chlu/**`, `push_repo.sh`, `sync_project.sh`, or any
  existing job script.
- Branch ref confirmed from the MAIN repo (`git -C /Users/user/Desktop/CHLU log main..agent/…` shows `98c8534`).
  Base `main` still `3eb9eb0`, so the rebase is a no-op. No conflicts. Worktree left in place for review.
- ⚠ Five other worktrees are live (`CHLU-bprime-family2`, `CHLU-inv-base`, `CHLU-merge`, `CHLU-wt2`, `CHLU-wt3`)
  — I stayed entirely inside `../CHLU-stage` and never touched the shared checkout.

## 5. ⛔⛔ EVERY CLUSTER-SIDE ITEM IS `NOT RUN`
CSF3 requires 2FA. **I cannot reach it and I did not try.** ⛔ Not a zero, not a failure, not a pass:

| leg | status |
|---|---|
| rsync of the 44 files to `~/scratch/CHLU/artefacts/` | **NOT RUN** — needs `CSF3_USER` + VPN (Head, §3 step 2) |
| the dataset download job on a compute node | **NOT RUN** — Head, §3 step 3b |
| `sbatch scripts/csf3/verify_artefacts.sh` | **NOT RUN** — Head, §3 step 4 |
| `STAGED.json` with `"status":"complete"`, `"checksum_match":true` | **NOT RUN** — ⛔ **written only by the Head's verify job.** It is the mechanical precondition for `gen-wells-route-b`. |
| any `STAGE=cell/store/certificate` run | **NOT RUN** |
| the `-p multicore` partition choice, the `-t 1:00:00` budget, the far-side sha256 throughput | **NOT RUN / UNTESTED-BY-AGENT** — pattern-based from the official CSF3 docs cited in `scripts/csf3/README.md §1` |

⭐ The local `LOCAL-VERIFY-*.json` files are **laptop** runs of the same script against the same bytes. They are
evidence that the mechanism works; they are ⛔ **not** `STAGED.json` and are named so they can never be mistaken for it.

## 6. ⭐ DOWNSTREAM RECONCILIATION LIST — needs an owner
1. **`push_repo.sh:42` `--exclude '.claude/'` — reported, NOT edited** (task-forbidden, and correctly so: it is
   the live code path for a cluster user). ⭐ It is now *correct by design*: code goes by `push_repo.sh`,
   artefacts by `stage_artefacts.sh`, and the two never fight over the same tree. The only debt is documentation:
   `scripts/csf3/README.md §2`'s file table lists neither new script. **Owner: whoever owns the README.**
2. ⛔ **`chlu.experiments.exp_gen_decoder` DOES NOT EXIST**, yet `job_gpu_gen.sh` maps `STAGE=decoder` to it and
   the stage DAG (`encoder → decoder → store`) requires it. The launcher's §7b precondition check catches it and
   dies loudly (`"stage module … does not exist yet"`), so this is not silent — but **`STAGE=decoder` cannot be
   run**, and `store`'s upstream-manifest gate will therefore refuse unless `ALLOW_MISSING_UPSTREAM=1`.
   ⚠ In practice the shipped decoder comes from `gen-decoder-lift` and is *staged*, not fitted on the cluster, so
   the DAG edge may simply be stale. **Owner: the Hub — decide whether `decoder` is retired from the DAG or written.**
3. ⚠ **Neither `env.sh` nor `job_gpu_gen.sh` knows about the artefact root.** Until one of them exports the five
   `CHLU_*` variables (§2.5), every submit must carry them by hand in `--export` (§3 step 5), and an omission
   fails *late* — inside the loader, after the GPU is allocated. ⛔ I do not own either file (a peer spoke holds
   config this wave). **Owner: the engineer who next touches `scripts/csf3/env.sh`.** The one-liner is
   `export CHLU_CLAUDE_ROOT="${CHLU_CLAUDE_ROOT:-$CLU_REPO/artefacts}"` plus the four derived paths.

## 7. Open questions / risks
- ⚠ **`chlu data <ds> --download` rewrites `c_invalid.json` unconditionally** (`_build_gen_artifacts` runs on every
  invocation, not only on download). The C set is nominally deterministic (`c_invalid_seed=0`), but the standing
  rule is that it is single-sourced and ⛔ never regenerated. §3's ordering (download → **re-stage** → verify)
  makes the laptop's copy authoritative and makes any divergence a **loud** verify failure rather than a silent
  substitution. Worth a cheap `--no-artifacts` flag on the CLI; ⛔ not made here (out of scope, not my file).
- ⚠ **Seeds 1 and 2 ship a `scope="used"` feature cache (9,288 rows).** `exp_gen_cell` re-encodes any uncovered
  row on the fly — which needs the **encoder and the dataset**, on the GPU. That is a wallclock cost on those two
  seeds that the `scope="full"` seed 0 does not pay, and it is another reason the dataset must be on the cluster.
- ⚠ `verify_artefacts.sh` opens `c_invalid.json` through `load_c_requests` with the factor names taken from the
  **file's own header** (the dataset is deliberately not staged). That exercises the schema check, the
  factor-index/name agreement and the record copy; ⛔ the one thing it cannot catch is a C set built against a
  *different grid* — the stage run itself asserts that against the loaded split. Declared, not hidden.
- The manifest pins the axes it was generated for. A run on `dsprites`, on `cls`, or on the
  `recomb2element_wide` condition (which suffixes the artifacts tag) needs its own `stage_artefacts.sh` invocation
  **and** the corresponding artefacts to exist locally — the script refuses rather than staging a partial set.

## Proposed handover updates (for the Hub)
- **§7-CURRENT, new item — `[RESOLVED by gen-cluster-stage-artefacts]` the artefact-transfer hole.** `push_repo.sh`
  excludes `.claude/` and `sync_project.sh` moves only `projects/`, so no gen artefact could reach CSF3. There is
  now a third, orthogonal path: `scripts/csf3/stage_artefacts.sh` (laptop → `~/scratch/CHLU/artefacts/`, a verbatim
  `.claude/` mirror, 44 files / 1.953 GiB for `shapes3d×recon×seeds 0-2`) + `scripts/csf3/verify_artefacts.sh`
  (re-checksums **and opens** every file, writes `STAGED.json`). ⛔ `push_repo.sh` unchanged by design.
- **§7-CURRENT, new item — `[OPEN, launcher]` `STAGE=decoder` has no module.** `chlu.experiments.exp_gen_decoder`
  does not exist; `job_gpu_gen.sh`'s DAG names it as `store`'s upstream. Fails loudly at the §7b check, never
  silently. Decide: retire the edge, or write the module.
- **§7-CURRENT, new item — `[OPEN, env]` the artefact root is not exported by `env.sh`.** Until it is, every gen
  submit must carry `CHLU_CLAUDE_ROOT` + four derived paths in `--export`; an omission fails inside the loader
  after the A100 is allocated.
- **Gate status for `gen-wells-route-b`:** its mechanical precondition — `STAGED.json` with `"status":"complete"`
  and `"checksum_match":true` — is ⛔ **NOT RUN**. The script that writes it exists and is proven (including
  proven *failing*, twice, two different ways). **The gate now turns on one Head action**, not on more engineering.
