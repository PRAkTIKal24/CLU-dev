# D2 — What work should move to this machine

**Spoke:** D2 (compute split) · read-only, no experiments run, nothing installed, nothing modified.
**Workspace read:** `/c/Users/PratikJawahar/chlu/chlu-hub-bundle` (2.6 GB, git `main`, 2 commits:
`172fa35` = the pin's tree verbatim, `5c8bfec` = the bundle overlay).
**Machine, re-measured by me:** Intel Core Ultra 9 285H · **16 cores / 16 logical** · **63.4 GB RAM** ·
C: 833.6 GB free of 951.5 GB.

---

## ⭐ RECOMMENDATION — five bullets

1. **⭐⭐ Run `hub/composition` here, and run it first.** It is the only one of the three chartered hubs
   whose *every input is already inside the bundle* — 3 seeds of feature cache, the three `store_s*.pkl`,
   the frozen `v1_capacity_schedule` decoder, the gated hash-asserted external classifier, and the
   `historical_688128` LACE selection source that `GenCellConfig.lace_selection_source` points at. It is
   CPU-declared by its own charter, and its work decomposes into ~72 independent units
   (6 arms × 3 seeds × the pre-registered 4-point temperature grid), which is exactly the shape 16 cores
   and 63 GB beats an 8-core/16 GB Mac at. Its risk is **scientific** (the bar is a ~4× improvement on a
   criterion `knn_proto` itself misses — a well-controlled null is the likely and charter-acceptable
   outcome), **not environmental**.

2. **⛔ Do NOT run `hub/refusal` here — it is BLOCKED on a missing artefact, not on compute.** Its charter
   calls it "the cheapest unclaimed target" because it is "scored off artefacts that already exist" under
   `.claude/outputs/gen-read-cell-arm/runs/` and `gen-val-split-fix/refusal_scores.json`. **Neither
   directory is in the bundle.** `.claude/outputs/` contains exactly six slugs and neither is one of them.
   The *only* `refusal_scores.json` present is `gen-arms-a7-wiring/refusal_scores.json`, which the same
   charter marks **SUPERSEDED — never quote it**. Refusal here would have to regenerate the cell-arm
   energy traces *and* re-score the rivals on the repaired split, i.e. it is the most expensive of the
   three, not the cheapest. **This looks like a bundling omission and is my #1 question for the human.**

3. **⭐ Move the pure-CPU gen sweeps here permanently, hub or no hub.** `scripts/csf3/job_cpu_gen.sh`
   exists *because* `exp_gen_atom_mean` and `exp_gen_baselines` "were being run on the laptop, where a
   four-hour `exp_gen_atom_mean` sweep held a 16 GB machine at 570 % CPU all evening and **BLOCKED the
   scored cell arm**". That is the exact contention this box removes: 16 cores and 63 GB against the Mac's
   8 cores and 16 GB. The script's own default is `-c 16` — this machine's core count, exactly.

4. **⭐⭐ The bundle README's headline WSL2 mandate is FALSE, and that changes the plan.**
   `README-HUB-BUNDLE.md` says "⛔⛔ JAX HAS NO NATIVE WINDOWS SUPPORT — use WSL2 … `jaxlib` publishes **no
   `win_amd64` wheel**". The bundle's own `uv.lock` contains `jaxlib-0.9.0-cp311-cp311-win_amd64.whl` and
   `jaxlib-0.9.0-cp312-cp312-win_amd64.whl` (and three more at 0.6.2), and its `resolution-markers`
   explicitly enumerate `sys_platform == 'win32'`. **Spoke A2's own log confirms it empirically**:
   `platform win32, python 3.12.10-final-0`, `jax=0.9.0`, `chlu=C:\…\chlu-hub-bundle\chlu\__init__.py
   under_root=True`, `[verify] status=ready`. Native Windows is viable for the whole generative line.
   WSL2 is still needed for the *streaming/C3* line (§Blockers B1) — but that line is not what any of the
   three hubs is scored on.

5. **⛔ Do NOT plan on "GCloud CPU" as this box's escape hatch — there is no CPU path in `scripts/cloud/`.**
   The agent profile §4 says "⭐⭐ **GCloud CPU is your default. Always.**" but `scripts/cloud/cards.json`
   contains **four cards and all four are GPU** (`l4`, `t4`, `a100-40`, `a100-80`); `launch.sh` has no
   `--card cpu`. Combined with "⚠ CSF3 needs 2FA and you cannot reach it", **an agent's CPU default has no
   reachable remote implementation at all.** That is the single strongest argument for moving CPU work
   onto this box: it is currently the only uncontended CPU resource an agent can actually reach.

---

## What I did

Pure read-and-reason over the bundle. I read, in full: the autonomous hub charter, the `autonomous-hub`
agent profile, `AGENT_PROTOCOL.md`, `README-HUB-BUNDLE.md`, `env_hub.sh`, `pyproject.toml`, the `Makefile`,
all three hub task charters, `handover_context.md` §7 (Known Issues), `scripts/csf3/job_cpu_smoke.sh`,
`scripts/csf3/job_cpu_gen.sh` (header + guards), `scripts/cloud/README.md` §1–5, `cards.json`, `cost.sh`,
and the `--return-pack` half of `scripts/bundle_hub_workspace.sh`. I extracted every `#SBATCH` directive
from all ten CSF3 job scripts, enumerated the CLI, inventoried `.claude/`, read the shipped manifests for
recorded wall-clock, and ran static greps for GPU-only and POSIX-only constructs. I ran **no** experiment,
installed **nothing**, and wrote **nothing** inside the bundle.

One cheap, non-founding check: `command -v python` → the Microsoft Store stub, still broken in *my* shell.
I then read Spoke A2's `_A2-verify-run2.log` (already in `reports/`) and it shows the verifier **green** on
native Windows. Where I lean on that, I say so — it is A2's evidence, not mine.

---

## The three chartered hubs, assessed separately (this is the actual decision)

Two sentences each, per the brief:

- **`hub/composition`** — Target: held-out factor combinations rendered *correctly* and read by the frozen
  external classifier, beating every rival including `knn_proto` on the compositional factor, **with ≥ 60
  of 72 distinct VALID NOVEL tuples**, and with dreaming inside the target (valid novel tuples at a declared
  finite temperature, or reported DEAD in those words). The inherited facts make the difficulty explicit:
  the `clu` codebook read renders **9.00 ± 0.82** distinct tuples, `knn_proto` — the ceiling — renders
  **56.00 ± 3.74** and does not itself meet the bar, so the hub is asked for a ~4× improvement on our best.

- **`hub/deletion`** — Target: exact structural deletion of one factor value or one atom, shown on all four
  conditions (unrecoverable from *every composition*, every other factor preserved, byte-exact provenance,
  **and the rivals given the same deletion**). Its own charter names the trap — "store load-bearing BY
  CONSTRUCTION is a METHOD-SECTION fact, never a headline" — so a win that follows from the rival simply
  having no deletion operator is worth nothing and must be reported as by-construction.

- **`hub/refusal`** — Target: **one** landscape-derived criterion (a function of settled energy and
  convergence, nothing else) that refuses both `out_of_range` and `duplicate_factor` while accepting
  unseen-valid, at a single threshold chosen on valid data only, with **zero invalid supervision**, beating
  kNN (0.8457 / 0.9583) and LACE (0.8097) on both kinds over 3 seeds. It is charted as the cheapest because
  it should need no new scored generation run — **but the artefacts that claim rests on are absent from this
  bundle** (see Blockers, C1).

### Why composition beats the other two *on this box*

| | composition | deletion | refusal |
|---|---|---|---|
| inputs present in bundle | ⭐ **all of them** | mostly | ⛔ **the two it names are missing** |
| net-new mechanism code | moderate (temperature grid already pre-registered) | ⛔ **most** — see C2 | moderate |
| shape of the work | ⭐ **~72 independent units** — fans out across 16 cores | sequential-ish: build operator, then measure | one cell run, then cheap scoring |
| failure mode if it fails | ⭐ a well-controlled null, explicitly a success under the charter | ⛔ risk of a by-construction non-result | ⛔ risk of an unusable run (superseded rivals) |
| needs GPU | no (charter: CPU default) | no | no |

Deletion's extra cost is concrete: on the **generative** store the only deletion primitive that exists is
`chlu/core/gen_store.delete_store(store, mode="zero"|"mean"|"scramble")`, and its own docstring says it is
the **laundering control** — "the point of this function is to be *measured against*". It removes the whole
landscape. The scoped, per-factor-value / per-atom exact deletion the charter targets **does not exist on
the gen path** and must be written, *and* written again for each rival's table so they get the same
deletion. `store_lifecycle.py` / `well_lifecycle.py` do implement exact lifetimes and trash — but on the
*streaming* CLU path, not the gen store the hubs are scored on.

⚠ One thing that helps deletion: `gen-store-grouping/runs/shapes3d_recon/seed_*/store_manifest.json`
already carries a `deletion` block with `delete_zero_*`, `delete_scramble_*` and `delete_mean_*` deltas on
both aggregate and compositional. The pre-deletion control the charter demands is therefore partly in hand.

---

## The classification table

Cost figures marked **[measured]** come from wall-clock recorded inside the bundle's own shipped manifests
(built on `batMac.local`, an 8-core/16 GB machine); **[stated]** from a comment in the repo;
**[est.]** is my inference and is flagged as such. Nothing here was timed by me.

| work class | entry point | needs GPU? | est. cost (wall / RAM) | runnable here? | verdict |
|---|---|---|---|---|---|
| **hub/composition** (6 arms × 3 seeds × 4 temps) | `python -m chlu.experiments.exp_gen_cell --stage cell --seed N` + `--stage aggregate` | ⛔ no — charter §D: CPU default, GPU only for a headline | **[est.]** ~1–3 h/seed on 16 cores; ≤ 8 GB RSS (features are `mmap_mode="r"`) | ✅ **yes** | ⭐⭐ **RUN HERE** |
| **hub/deletion** | same cell harness + a **new** scoped deletion operator | ⛔ no | **[est.]** dominated by build time, not compute | ✅ yes, after the operator exists | **RUN HERE (second)** |
| **hub/refusal** | `exp_gen_cell` traces → `gen_certificate` scoring | ⛔ no | cheap *if* the traces existed | ⛔ not as chartered | ⛔ **BLOCKED** (on `gen-read-cell-arm/runs/` + `gen-val-split-fix/refusal_scores.json`) |
| CPU gen sweeps — atom-mean | `python -m chlu.experiments.exp_gen_atom_mean` (via `EXP=atom_mean`) | ⛔ no (`JAX_PLATFORMS=cpu` asserted) | **[stated]** ~4 h wall at 570 % CPU on 8 cores ⇒ ~23 CPU-h | ✅ yes — **better here than on the Mac** | ⭐ **RUN HERE** |
| CPU gen sweeps — baselines (knn / lace) | `python -m chlu.experiments.exp_gen_baselines` | ⛔ no | **[measured]** 3 legs/seed summing 3060 s / 2261 s / 3227 s ⇒ **~0.85 h wall per seed** | ✅ yes | ⭐ **RUN HERE** |
| **test suite** (2,711 tests in 145 files) | `pytest -q` (addopts force `--cov=chlu`) | no | **[measured, A2]** a 7-test targeted slice = 10.9 s; full suite unmeasured. ⚠ `AGENT_PROTOCOL` §4: JAX cold start ~20 min on the Mac | ✅ mostly — see B2 for the ~12 files that will fail | ⭐ **RUN HERE** (`pytest -n auto`, xdist is in the `test` group) |
| lint / format | `ruff check` · `ruff format` · `black` · `pylint` | no | seconds–minutes | ✅ yes (all have win wheels) | ⭐ **RUN HERE** |
| eval / metrics over the **frozen bundled artefacts** | `chlu/eval/*`, `chlu/eval/_tsb_vendor` (vendored, ruff-excluded) | no | minutes | ✅ yes | ⭐ **RUN HERE** |
| plotting / ICLR figures | `chlu/utils/plotting.py` (1,048 stmts), `exp_*/plot_*.py`, matplotlib `Agg` | no | minutes | ✅ yes | ⭐ **RUN HERE** |
| data prep / dataset download | `chlu data shapes3d --download`, `dsprites`, `enwik8`, `wikitext`, `chlu/data/industrial/*` | no | I/O-bound; Shapes3D = 255 MiB, one sha256 to assert | ✅ yes — 833 GB free | ⭐ **RUN HERE** (needs `h5py`, the `gen` extra) |
| small smoke runs / ablations | `--quick` on the exp CLIs; `scripts/smoke_c3_local.sh` | no | minutes | ✅ mostly | **RUN HERE** |
| **full training runs** (cluformer / C3 ladder) | `chlu/training/train_cluformer.py`, `scripts/csf3/job_gpu_c3_seeds.sh` | ⭐ **yes — A100-80** | `-c 12` ≈ 120 GB host RAM, `-t 1-0`, array `0-14%4` (5 arms × 3 seeds) | ⛔ **no** (no CUDA; `import resource` at module scope) | ⛔ **CSF3 or GCP ONLY** |
| KT tranche | `scripts/csf3/job_gpu_kt.sh` | mixed — `winding2d`/`reduced`/`postproc` run `-G 0 -p serial` | 2–3 h per seed CPU-mode legs; `-t 8:00:00` | ⚠ the `-G 0` legs only | **RUN HERE (CPU legs) / CSF3 (GPU legs)** |
| industrial anomaly eval (voraus / SKAB / SMD / TEP / MIMII) | `chlu eval …`, `scripts/csf3/job_gpu_eval.sh` | GPU per the job script | `-c 8`, `-t 4:00:00` | ⚠ datasets absent; CPU-slow | **CSF3 or GCP ONLY** |
| CAFE benchmark | `scripts/cafe/run_clu_cafe.py`, `scripts/csf3/job_gpu_cafe.sh` | GPU (`-c 8 -t 2:00:00`) | — | ⛔ no | ⛔ **BLOCKED** — `cafe_bench` is "a separate checkout kept outside this repo" and is **not in the bundle** |
| cloud orchestration (submit + collect) | `scripts/cloud/launch.sh` · `run_until_done.sh` · `status.sh` · `cost.sh` · `fetch_results.sh` | n/a (dispatches) | ~$0.343/h L4 spot → $2.04/h A100-80 on-demand | ⚠ **see C3** | **BLOCKED (on `gcloud`+Python)**, then ⭐ RUN HERE |
| CSF3 submission | `scripts/csf3/push_repo.sh`, `sbatch` | n/a | — | ⛔ no | ⛔ **BLOCKED** — profile §4: "CSF3 needs 2FA and you cannot reach it" |

### The CSF3 job scripts, decoded (they encode the real resource envelope)

| script | part. | GPU | cores | walltime | array | what it runs |
|---|---|---|---|---|---|---|
| `job_cpu_smoke.sh` | `serial` | **none** | **1** | `1:00:00` | — | `chlu --version`, `chlu info`, headless-Agg check, `pytest tests/test_core.py tests/test_data.py`. ⭐ "~5 min warm". **This is the CPU-viability signal and it is tiny.** |
| `job_cpu_gen.sh` | `multicore` `-A fatpou01` | **none** (`JAX_PLATFORMS=cpu` asserted) | **`-c 16`**, hard cap 84 | `1-0`, max `7-0` | seed-unit ledger, `--requeue`, `USR1@300` | `exp_gen_atom_mean` \| `exp_gen_baselines` |
| `job_gpu_array_seeds.sh` | `gpuA` | 1 × A100-80 | 8 | `4:00:00` | `0-4%4` | seed sweep |
| `job_gpu_c3_seeds.sh` | `gpuA` | 1 | **12** (≈120 GB, the 1-GPU ceiling) | `1-00:00:00` | `0-14%4` (5 arms × 3 seeds) | C3 ladder |
| `job_gpu_cluformer.sh` | `gpuA` | 1 | 8 | `12:00:00` | — | tier-3 cluformer pilot |
| `job_gpu_gen.sh` | `gpuA` | 1 | 8 | `4:00:00` | `0-11%2` | the GPU gen stages |
| `job_gpu_wells.sh` | `gpuA` | 1 | 8 | `2-0` | — | wells route-B, resumable |
| `job_gpu_kt.sh` | `gpuA` (CPU modes override `-p serial -G 0 -c 1`) | 1 or **0** | 8 | `4:00:00` | `0-2%3` | KT tranche |
| `job_gpu_eval.sh` / `job_gpu_cafe.sh` / `job_gpu_single.sh` | `gpuA` | 1 | 8 | `4:00:00` / `2:00:00` / `4:00:00` | — | eval · CAFE · single exp-c |

**Read:** every GPU script is 1 GPU / 8–12 cores — i.e. *single-device*, never multi-GPU. Nothing in this
programme needs a GPU *cluster*; it needs **one big card** (A100-80, or the 120 GB of host RAM that
`-c 12` buys). This box has neither, and cannot get either. But the CPU side (`multicore`, 16–84 cores,
7-day walltime) is exactly this box's shape at 16 cores — minus the 7-day walltime, plus the unit ledger
that makes a kill cost one seed.

---

## Blockers, called out explicitly

### A. CPU-only blockers — surprisingly few

**There is no hard CUDA dependency anywhere in `chlu/`.** I grepped for `jax.devices("gpu")`, `pmap`,
`device_put`, `NamedSharding`, `Mesh(`, and `cuda`. The only hits are:

- `sharding` — every occurrence is the project's own **store-sharding** vocabulary
  (`chlu/core/shard_store.py`, `exp_sharded_store.py`, `config.py`), not JAX device sharding.
- `chlu/experiments/kt/runner.py:2` — "array-job **sharding**", i.e. Slurm array splitting.
- `chlu/training/train_cluformer.py:687–692` — sets `JAX_PLATFORMS: "cpu"` for its worker processes.
- `cuda` — confined to `scripts/cloud/` and the `cuda` extra, which is marked
  `sys_platform == 'linux'` in `pyproject.toml` and is therefore a **no-op off Linux by design**.

⇒ The code is device-agnostic JAX. **CPU-only is a throughput constraint here, not a correctness one.**
The hard NOs are the *scale* ones: anything sized for an 80 GB A100 or for 120 GB of host RAM
(`job_gpu_c3_seeds.sh`, `job_gpu_cluformer.sh`, `job_gpu_gen.sh`) does not fit 63 GB and has no card.

### B. Native-Windows blockers (distinct from CPU-only, and the more interesting axis)

**B1 — `chlu/training/train_cluformer.py:42: import resource`.** `resource` is Unix-only stdlib; this is a
module-scope import and it will raise `ModuleNotFoundError` on native Windows. It is **not** on the
`import chlu` path (`chlu/__init__.py` imports `chlu.training.train`, not `train_cluformer`), which is why
A2's verifier imported `chlu` cleanly and why A2's coverage report shows `train_cluformer.py 820 0%`. It is
reached by `exp_anti_erosion.py` (module scope) and lazily by `chlu/eval/text_slices.py`. ⇒ **The
cluformer / C3 / streaming line is dead on native Windows and needs WSL2.** The generative line — every one
of the three hubs — does not touch it.

**B2 — ~12 test files carry POSIX assumptions.** 11 shell out via `subprocess.run(["bash", …])`
(`test_c3_csf3_harness`, `test_c3_gb_geometry`, `test_c3_run3_budget_exemption`, `test_cloud_launch`,
`test_cloud_runner`, `test_gen_launcher`, `test_hub_workspace`, `test_job_cpu_gen`, `test_stage_artefacts`,
`test_stage_gcs`, `test_ttt_stability_and_d5_wiring`); these *may* work through Git Bash's `bash.exe` but
will hit the `/c/…` vs `C:\…` translation problem. `test_job_cpu_gen.py` parametrises over
`signal.SIGKILL` and sends `signal.SIGUSR1` — **neither exists on Windows**; that file will error at
collection or at the test. Four files assert `os.access(p, os.X_OK)`, which on Windows is satisfied by any
readable file, so those pass *vacuously* rather than failing — worth knowing before anyone quotes them.
Several hard-code `/tmp` paths. **There is not a single `sys.platform` skipif in the entire suite** — it
has never been run on Windows and nothing is guarded for it.

**B3 — `env_hub.sh` exports a POSIX `PYTHONPATH` with a colon.** `export PYTHONPATH="$CHLU_ROOT${PYTHONPATH:+:$PYTHONPATH}"`
— colon-separated, and `$CHLU_ROOT` in Git Bash form is `/c/Users/…`. Native Windows Python wants `C:\…`
and `;`. A2's run came out green, so the single-element case evidently survives; a *second* element on
`PYTHONPATH` would silently break it. Flagged for A2/A1, not mine to resolve.

**B4 — 588 AppleDouble `._*` files, 54 of them inside `.git/`.** Every `git` invocation in the bundle now
prints `error: index file .git/objects/pack/._pack-….idx is too small` — because
`._pack-e1cc3bb….idx/.pack/.rev` (163 bytes each) sit beside the real 13 KB `.idx` and 3.1 MB `.pack` and
git tries to parse them as packs. Git still works (`git rev-list --count HEAD` → 2, `git branch` → `main`),
so this is currently noise — **but the code-return path is `git bundle create`**, and a script that
inspects git's stderr, or `set -e`-style plumbing, can trip on it. ⭐ **Good news on the pytest side:**
pytest's `python_files = "test_*.py"` does **not** match `._test_core.py`, so the 145 AppleDouble files in
`tests/` do **not** pollute collection, and `._*.py` under `chlu/` is not an importable module name. ⚠ The
fix is to delete three files in `.git/objects/pack/` — **they are not in `BUNDLE_MANIFEST.json`**, but §0.1
of the agent profile forbids any agent from deleting anything. **That is a human decision** (Q4).

### C. Absent-dependency blockers — what dies on a missing file rather than on compute

**C1 — ⛔⛔ `hub/refusal`'s scoring artefacts are not in the bundle.** `.claude/outputs/` contains exactly:
`gen-arms-a7-wiring` (2.9 M) · `gen-decoder-lift` (18 M) · `gen-external-classifier` (12 M) ·
`gen-feature-bridge` (2.0 G) · `gen-read-compose` (1.6 M) · `gen-store-grouping` (425 K).
Both `gen-read-cell-arm/` and `gen-val-split-fix/` are **absent**, and `BUNDLE_MANIFEST.json`'s 57 pinned
paths do not mention them. **This is the highest-value finding in this report** and it inverts the
charter's own cost ranking of the three hubs.

**C1b — `env_hub.sh` also exports `CHLU_GEN_WIDE_BRIDGE_ROOT=…/gen-bridge-wide/runs`, which is likewise
absent.** Anything on the *wide* split (`chlu/data/shapes3d.build_wide_split`, `test_gen_bridge_wide.py`)
has no artefacts here.

**C2 — Shapes3D HDF5: NOT shipped, and correctly so.** `.claude/data/gen/shapes3d/` holds only
`factor_index.npz` (3.8 MB), `c_invalid.json`, `a7_redundant.json`, `query_sets.json` — 3.9 MB total. The
255 MiB `3dshapes.h5` is absent by design and re-downloadable from
`https://storage.googleapis.com/3d-shapes/3dshapes.h5`, with
`sha256 = 0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785` to assert. **Crucially, the hub
work does not need it:** `exp_gen_cell` imports no `h5py` and never calls `open_images` — it reads the
pre-encoded `features.npy`, renders through the frozen decoder, and takes its grid/splits from
`factor_index.npz` (`gen_splits.py:1011` — `derived_from: "factor_index.npz"`). ⇒ **The generative line
runs here with no image blobs at all**, exactly as the README promises. `h5py` (the `gen` extra) has
Windows wheels in `uv.lock`; without it the 8 real-data tests in `test_gen_data_splits.py` `importorskip`
and pass as **silent skips** — a gap `pyproject.toml` itself warns is "invisible unless you read the skip
count".

**C3 — the cloud orchestrator role.** Every `scripts/cloud/` entry point needs `gcloud` **and** a Python
(`PY="${PYTHON:-python3}"` in `cost.sh`, `chlu_card_spec`/`chlu_price` in `config.sh`, `quota.py` as the
preflight). CONTEXT records `gcloud` failing for want of an interpreter. Once Python is on PATH this
should unblock — **and then this box is a genuinely good orchestrator**: it submits, polls
`run_until_done.sh` under its attempt and spend caps, and `fetch_results.sh` pulls into `.claude/outputs/`.
⚠ But `fetch_results.sh` is `gcloud storage rsync` — **gcloud's own rsync, not the `rsync` binary** — so
the *absence of `rsync` on this machine does not block it*. See the return-path section.

**C4 — `cafe_bench` is absent and is supposed to be.** `pyproject.toml`: "cafe_bench itself is NOT declared
here — it is a separate checkout kept outside this repo". `scripts/cafe/run_clu_cafe.py:143` guards the
import and exits with "ERROR: cafe_bench not importable". Anything CAFE is **BLOCKED** here.

**C5 — all other datasets are absent** (enwik8, wikitext, dSprites, MNIST, and the five industrial
loaders). They are downloads, not bundle content. 833 GB free makes this box a fine place to fetch them.

---

## The parallelism argument — which shapes exploit 16 cores, and which don't

**Exploits it (move here):**

- **Seed arrays.** Every gen sweep's unit is a seed and `SEEDS="0 1 2"` is the default; `job_cpu_gen.sh`'s
  unit ledger makes each seed independently resumable ("a kill costs ONE UNIT, never the run"). Three
  seeds × six arms is 18 independent jobs before the temperature grid is applied.
- **Arm sweeps.** `exp_gen_cell.ARMS` = `clu, store_removed, knn_proto, lace, ebm_compose, dynread` — six
  arms, `--arm` selectable, mutually independent.
- **The composition temperature grid.** `temp ∈ {0.0, 0.005, 0.05, 0.5}` is already pre-registered for that
  cell. 6 × 3 × 4 = **72 independent units.**
- **`pytest -n auto`.** `pytest-xdist>=3.5.0,<4` is in the `test` group. 2,711 tests across 145 files; the
  cold-JAX import is paid once *per worker*, which argues for `-n 8` or so rather than `-n 16` given the
  ~20-min cold-start warning in `AGENT_PROTOCOL` §4. ⚠ `addopts` force `--cov=chlu --cov-report=xml`, which
  costs real time under xdist — consider `-p no:cacheprovider --no-cov` for iteration loops.
- **Any `--quick` ablation grid.**

**Does not exploit it (leave on the Mac or send to a cluster):**

- **One long single-device train.** `job_gpu_cluformer.sh` (`-t 12:00:00`, one GPU) and
  `job_gpu_c3_seeds.sh` (`-t 1-0`, `-c 12` ≈ 120 GB). These are single-stream, memory-hungry, and
  GPU-shaped. A 16-core CPU is not "2× a Mac" here; it is simply the wrong machine.
- **Interactive iteration on the Mac's warm venv.** `AGENT_PROTOCOL` §4 says keep the session warm because
  a cold JAX import is ~20 min. Anything conversational stays where the venv is already hot.

⭐ **The RAM argument is the underrated one.** The three feature caches are 1.83 GiB each (float16, 480,000 ×
2048) = **5.5 GiB for all three seeds**, read via `mmap_mode="r"` (`gen_bridge.py:1046`). On a 16 GB Mac
that is a real fraction of memory and it is why one sweep "BLOCKED the scored cell arm". On 63 GB it is
noise — **all three seeds can be page-cache resident simultaneously.** This box does not just have more
cores; it removes the specific contention the project has already been bitten by.

⚠ **Sizing caveat:** JAX's CPU backend and the BLAS pools will each try to take all 16 cores. `job_cpu_gen.sh`
already handles this — it pins `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to the core grant because "an unbound
numpy/sklearn … oversubscribes its cgroup and runs SLOWER". Running N units concurrently here needs the
same discipline (e.g. 4 units × `OMP_NUM_THREADS=4`), not 16 units at 16 threads each. **The agent profile
caps sub-agents at 3 concurrent — written for a 16 GB Mac that "has thermalled before". That cap is
arguably too conservative for this box; it is the human's to relax, not an agent's** (Q5).

---

## ⭐⭐ The split hazard — how results come home, and what collides

### The round trip, as designed

**Results** (`.claude/outputs/**`) come home by `--return-pack` / `--return-ingest`:

```
# far side (this box):
./scripts/bundle_hub_workspace.sh --return-pack <slug> --return-out ~/<slug>.return.tar.gz
# primary (Mac):
scripts/bundle_hub_workspace.sh --return-ingest /path/<slug>.return.tar.gz
```

It packs exactly `<slug>/**` plus `<slug>.md`, excludes the verifier's own `.workspace-write-probe.json`
**by name** (deliberately, so a hub's own `.PREREG` is not swept up by a blanket dotfile filter), writes a
`RETURN_MANIFEST.json` with a sha256 per file, and builds the tar in two `-C` passes because bsdtar stops
parsing options at the first file operand.

**Code** comes home separately, by `git bundle` — *whole branch, never a range*:

```
git -C "$CHLU_ROOT" bundle create ~/<slug>.gitbundle agent/autonomous-hub/<slug>
git fetch /path/<slug>.gitbundle 'refs/heads/…:refs/remotes/farside/<slug>'
```

⛔ `git bundle create … main..<branch>` is documented to fail **only on arrival** — a range bundle is thin
and its prerequisite is this repo's synthetic base commit, which the Mac has never seen. ⭐ Take the 3 MB.

### What collides, and what does not

- ⭐ **Nothing, by construction — and I checked the mechanism, not the claim.** `--return-ingest` refuses
  absolute paths, `..`, symlinks, and any member not prefixed with `<slug>`; refuses any sha mismatch;
  **aborts the whole ingest if the tar holds a file the manifest does not declare**; and if
  `.claude/outputs/<slug>` already exists it diverts the *entire* set to `<slug>.returned-<utc>/` and prints
  the divert. Nothing existing is ever touched. Combined with **YOU NEVER DELETE** (§0.1) and the single
  writable path `$CHLU_HUB_OUTPUTS/<slug>/**` (§0.3), a Mac-side hub and a Windows-side hub on **different
  slugs** cannot collide at all.
- ⛔ **The one real collision: two machines running the same slug.** The ingest will divert rather than
  overwrite — so nothing is lost — but you end up with `hub-composition/` and `hub-composition.returned-<utc>/`
  and **no mechanical statement of which is authoritative**. ⭐ **Mitigation: give the far side its own slug
  suffix from the start** (e.g. `hub-composition-win`), so the two never contend for one directory.
  ⚠ Note the charter-level consequence: §A.2 caps a hub at **3 scored runs against the held-out sets**. If
  the same target is pursued on two machines, *the scored-run budget is shared and neither machine can see
  the other's count.* That is a scientific-integrity hazard, not a filesystem one, and it is the strongest
  reason **not** to split a single hub target across both machines.
- ⭐ **`rsync` being absent here does NOT break the return path.** `--return-pack` uses `tar`, `gzip`,
  `sha256sum`/`shasum` (with a documented fallback), `find`, `mktemp`, `hostname` — all present in Git Bash.
  Every `rsync` in the tree is either `gcloud storage rsync` (gcloud's own subcommand, not the binary) or a
  *printed suggestion* in a CSF3 job script's closing "pull to laptop with:" line. **No code path on this
  machine calls the `rsync` binary.**
- ⚠ **But `--return-pack` needs Python.** It shells `"$PY_BIN"` to build `RETURN_MANIFEST.json`. Since
  Python is a prerequisite for running a hub at all, this is not a new blocker — but it means **the return
  path cannot be exercised as a dry run before Python is settled**, and it should be smoke-tested
  (`--dry-run`) on a throwaway slug *before* ten hours of work depends on it (Q3).
- ⚠ **`git bundle create` is the untested leg**, and it runs against a `.git` that currently errors on every
  invocation (B4). Also untested: whether Git Bash's `mktemp -t` template handling and the two-pass `tar`
  behave identically to macOS bsdtar here.

---

## What I could not resolve honestly

1. **I have no timing for `exp_gen_cell`.** No cell run is in the bundle (that is C1), and I ran nothing.
   My "~1–3 h/seed on 16 cores" is an inference anchored on `exp_gen_baselines`' measured ~0.85 h/seed and
   the stated 4 h/23 CPU-h for `exp_gen_atom_mean`. **Treat it as an order of magnitude, not a budget.**
   One `--quick --stage cell --seed 0` run would replace this guess with a number; the brief correctly
   forbade me from doing it.
2. **I cannot say how much *faster* this box is than the Mac per core.** Apple Silicon vs Core Ultra 9
   single-thread on JAX's CPU backend is a benchmark question, and the honest claim is only the structural
   one: **2× the cores and 4× the RAM**, which matters for fan-out and for not blocking a scored run.
3. **I did not verify that a full `uv sync` resolves on Windows.** I verified that every non-CUDA package in
   `uv.lock` has a `win_amd64` or pure-Python wheel (the *only* gaps are the 17 `nvidia-*` / `jax-cuda12-*`
   packages, which are `sys_platform == 'linux'`-gated anyway), and that A2 got a working native-Windows
   `jax=0.9.0` on Python 3.12.10. Those are strong but they are not "`uv sync --frozen` succeeded".
4. **I have not read `.claude/c3-handover.md`** because it is not in the bundle. `handover_context.md` is
   the **archived Campaign-2** document, explicitly closed 2026-08-12 with "⛔ NO 'NEXT C2 HUB': C3 LIVES IN
   `.claude/c3-handover.md`". §7 Known Issues (which the hub profile mandates) *is* present and I read it.
   Whether the C3 handover was meant to travel, I cannot say.
5. **I could not measure how long the full test suite takes**, nor exactly how many of the 2,711 tests fail
   on native Windows. My B2 list is static analysis: ~12 files at risk, of which I am confident only about
   `test_job_cpu_gen.py` (SIGKILL/SIGUSR1 do not exist on Windows).

---

## Numbered open questions for the human

1. **⛔⛔ Was `gen-read-cell-arm/` and `gen-val-split-fix/refusal_scores.json` omitted from the bundle by
   accident?** `hub/refusal` is chartered as the cheapest target *entirely because* those artefacts exist,
   and they are not here. If they can be added (they should be small — `gen-read-compose` is 1.6 MB and
   `gen-store-grouping` 425 KB), refusal becomes a genuinely cheap, high-value run for this box and my
   recommendation changes. If they cannot, **the refusal charter's cost claim needs a correction**, because
   the next agent that reads it will plan against a premise that is false on this machine.

2. **Should `README-HUB-BUNDLE.md`'s WSL2 section be corrected?** Its stated reason — "`jaxlib` publishes no
   `win_amd64` wheel" — is contradicted by the bundle's own `uv.lock` and by A2's green native-Windows
   verifier run. WSL2 is still the right answer for the *cluformer/C3/streaming* line (`import resource`)
   and for the POSIX-shelling tests. ⛔ I did not edit it — it is frozen bundle content — but the file is
   the first thing the next person reads, and it is currently steering them wrong. **Do you want native
   Windows for the gen line (simpler, and now demonstrated), or WSL2 for everything (one environment, and
   the README's `/mnt/c` warning then applies to where the 1.9 GB `features.npy` lives)?**

3. **May someone smoke-test the return path before a hub commits ten hours to it?**
   `--return-pack --dry-run` on a throwaway slug, plus one `git bundle create` + `git bundle verify`,
   costs minutes and tests the two legs (Python-built manifest; git-with-a-broken-idx) that have never run
   on Windows. ⛔ I did not do it — it writes files, and I am read-only.

4. **May the 54 AppleDouble files under `.git/` be deleted?** They are **not** in `BUNDLE_MANIFEST.json`,
   they make every `git` call emit `error: index file … is too small`, and `git bundle` is the code-return
   mechanism. §0.1 forbids every agent from deleting anything, so this needs you. (The other 534 `._*`
   files are harmless: pytest's `test_*.py` glob does not match `._test_*.py`, and `._*.py` is not an
   importable module name.)

5. **Should the 3-concurrent-sub-agent cap be raised on this box?** Agent profile §3 caps it at 3 with the
   reason "This is a 16 GB laptop and it has thermalled before." That reason does not hold at 16 cores and
   63 GB. **A hub cannot relax its own charter** — if you want the fan-out this box is good for, the cap
   needs raising explicitly, with the thread-pinning discipline from `job_cpu_gen.sh`
   (`OMP_NUM_THREADS`/`MKL_NUM_THREADS` = the per-unit grant, not 16).

6. **If a hub runs here, what slug does it use?** I recommend a machine-suffixed slug
   (`hub-composition-win`) so a Mac-side run of the same target can never contend for
   `.claude/outputs/hub-composition/`. ⚠ And please confirm the §A.2 ruling: **is the 3-scored-run budget
   per hub, or per target across machines?** Two machines pursuing one target cannot see each other's
   count, and that is a route to an un-retractable over-selection.

7. **Is there any intent to add a CPU card to `scripts/cloud/`?** The agent profile says "GCloud CPU is your
   default. Always", but `cards.json` is four GPU cards and `launch.sh` has no CPU path, while CSF3 is
   2FA-blocked to agents. As it stands, **an agent's CPU default is unimplemented** — which is precisely the
   gap this machine fills, but it also means the profile's §4 text is currently unachievable as written.

8. **Does `cafe_bench` need to travel to this box?** It is deliberately outside the repo. If any ICLR figure
   depends on a CAFE number, that leg is **NOT RUN** here and will stay that way until the checkout arrives.
