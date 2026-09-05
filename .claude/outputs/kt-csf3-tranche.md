# kt-csf3-tranche — experiment-engineer report

**Task + acceptance criterion:** make the KT/A100 confirmation-at-scale run actually launchable — promote the
gitignored `kt-2d-csf3` scratch scripts into the tracked tree (bit-exact round-trip vs the committed laptop
JSONs as the gate), write a conventions-compliant `scripts/csf3/job_gpu_kt.sh` with a sized launch recipe,
local smoke first, suite green, and end with a copy-pasteable `sbatch` block.

**Status: DONE** — with one **material finding that changes what the tranche should run** (§4). Everything
packaged, verified bit-exact, suite green. Soft exponent (a) is ready and worth launching *today*. Soft
exponent (b) **should not be launched as scoped**; I have measured evidence that the scoped fix cannot work,
and the sbatch header says so in the recipe itself.

**⚠ RECONCILIATION LIST (owner needed — Hub, assign at review), 2 items, both in §4:**
1. **`kt-2d-csf3` §5/§7's explanation of the 1-D `-0.7` slope is probably wrong.** It is ascribed to `xi~1.2`
   at `T/J=1.0`; measurement says it is a **fit-window saturation artifact**. The recommended remedy
   ("rerun at `T/J=0.5`") makes the slope **flatter, not steeper**. Sites: `kt-2d-csf3.md` §5, §8, §9 item 2;
   any handover copy; `future_work.md` if it inherited the "clean -1 needs a lower-T run" line.
2. **The 2-D `L>=32` sign change is already half-visible in a 4-walker probe** (`tau_med` at `T/J=1.10`:
   45 → 89 → 97 → 85 for L=16/32/48/64, slope `~-0.06` over L>=32). Not a result — sizing evidence — but it
   means (a) is a high-value, cheap launch.

---

## 1. Flag provenance (mandatory)

| item | value |
|---|---|
| base commit | `a5978f6` (local `main`); branch `agent/experiment-engineer/kt-csf3-tranche`, 5 commits, HEAD `d5cac9b` |
| worktree | `../CHLU-kt-csf3-tranche` (isolated — a concurrent agent's branch `clu-cafe-integration` was checked out in the shared repo and it committed `5043362` mid-session). Ref verified from main repo, then removed. |
| env | **main venv reused** per §4 (`/Users/user/Desktop/CHLU/.venv`, `PYTHONPATH=<worktree>`); **jax 0.9.0**, equinox 0.13.4, numpy 2.4.1, scipy 1.17.0; **CPU**. No `uv sync` in the worktree ⇒ no w6-style version drift. |
| precision | **float64** (`jax_enable_x64=True`) everywhere on the CLU path; numpy float64 for reduced-MC |
| langevin_noise | **`"fdt"`** (asserted in-process; repo default `legacy` rejected — §7.22) |
| kinetic / governor | **`newtonian_learned`** / governor **OFF** (both asserted) |
| designed unit | `CHLU(dim=2, hidden=4)`, `MexicanHatPotential(lam=1, f=1, k_spec=None)`, `log_mass_for_inertia([1,1])` |
| coupling | `channel_spring_coupling(2,2,kappa=0.05, channel=(0,1))`; `J = 2 kappa r*^2 = 0.10`, `k_r = 8` |
| **T_KT** | **`1.786 kappa r*^2 = 0.0893` CLU units `= 0.8929 J`** at kappa=0.05. **Never `0.1786`** — grepped: every occurrence in my code is an explicit retraction warning, none asserts it. |
| sampler | `langevin_step` vmapped over walkers, `dt=0.02`, `gamma=0.10`, `m_eff=lat.effective_mass()`, `noise_mode="fdt"` |
| seeds | round-trips at the laptop seeds: 700 (2-D winding), 100/101/102 (reduced rho_s), 31 (1-D MSD), 7 + equil 1234 (bridge) |
| suite | **341 passed** in 285 s (328 baseline + 13 new), `--no-cov`. `ruff check chlu/ tests/test_kt.py` → **All checks passed**. |

`ruff format` NOT applied: the repo does not enforce it (37 pre-existing files would reformat), so applying it
would have inflated the diff and churned verbatim-preserved physics.

---

## 2. Round-trip parity — the acceptance gate [PASS, BIT-EXACT]

Re-ran the *promoted* code at the laptop seeds/sizings against `.claude/outputs/kt-2d-csf3/*.json`.
Pre-registered (PREREG P1) as **absdiff 0, not "within error"**:

| check | reference | reproduced | absdiff |
|---|---|---|---|
| 2-D winding `tau_med/tau_mean`, `T/J=1.30`, L=8/12/16, seed 700, nwalk 24 | 16/18.8333, 27/28.9167, 29/31.0 | identical | **0** |
| reduced-XY `rho_s`, L=8, `T/J=0.60`, seeds 100–102 | `0.8244828012941149` | `0.8244828012941149` | **0.00e+00** |
| reduced-XY `rho_s` SEM (same cell) | `0.00041260235470954687` | identical | **0.00e+00** |
| 1-D winding MSD rate, N=8/16, `T/J=1.0`, seed 31, NW=256 | `6.54966062238462e-05`, `7.850910166334073e-05` | identical | **0.00e+00** |
| **E-bridge kill criterion**, L=8, `T/J=0.70/0.85/1.00` — all 7 fields/row (`rho_clu`, `rho_reduced`, `rho_ratio`, `cos_clu`, `cos_ratio`, `drift`, `u_xy`) | ratios 0.980/0.957/0.931 | identical to 12 dp | **0.00e+00 on all 21** |

Re-verified **after** the ruff-driven variable renames inside the physics (`I`→`Ix` etc.) — 13/13 still pass, so
the renames are provably numerics-neutral. **Nothing about the physics was lost in packaging.**

## 3. What I built

**Layout decision — `chlu/experiments/kt/` (a package) + a thin CLI, not `scripts/kt/`.** Justification: these
are not standalone one-shot scripts, they are a *sweep* that must be driven by a Slurm array with per-cell
provenance, and they import the real `chlu.core` stack (`CLULattice`, `langevin_step`). Putting them under
`chlu/experiments/` gets them config-driven knobs, the `chlu` console-script entry (so sbatch calls
`chlu exp-kt`, exactly like `chlu eval`), and pytest coverage. `scripts/kt/` would have re-created a second,
untested execution path.

- `chlu/experiments/kt/{reduced_xy,clu_path,postproc,runner}.py` — the five scratch scripts, physics verbatim.
- `ExperimentKTConfig` in `chlu/config.py` (+ `CHLUConfig`/`get_default_config`/`load`/`save` wiring; YAML
  round-trip verified). **Defaults reproduce the laptop run exactly** so parity is the out-of-the-box
  behaviour; the tranche overrides on the sbatch line, keeping provenance attached to the run.
- `chlu exp-kt` in `chlu/cli/experiment_cmd.py` — `--mode`, `--task-id` (array cell sharding), `--out`,
  `--quick`, plus tranche overrides.
- `scripts/csf3/job_gpu_kt.sh` — `-n 1 -c 8`, separate `-e`, `logs/%x-%A_%a.{out,err}`, `-a 0-2%3` throttle,
  `--mail-user=$CLU_MAIL` **parameterized** (no address in-repo, per the Head's 2026-07-20 decision).
- `tests/test_kt.py` — 13 tests (parity, guards, grids, shard-merge, the 0.1786 guard).

**Settings discipline (item 4).** `assert_kt_settings()` raises on: x64 off, `langevin_noise != "fdt"`,
`kinetic_mode != "newtonian_learned"`, governor on. It runs per-cell *and* as a cheap preflight in the sbatch
script before any wallclock is spent. Verified firing by test.

**No dataset download (item 2, confirmed).** Every KT mode is synthetic/self-generated — the lattice is built
in-process and initial conditions come from an in-process reduced-XY warm start. No cache, no `--download`, so
**zero exposure to the shared-cache race that killed 5/6 of the first voraus launch**. Stated in the header.

**Local smoke (item 5).** All five modes end-to-end on `--quick`: `winding1d`, `winding2d`, `bridge`,
`reduced` each execute and write JSON; array shards (`--task-id 0,1`) write `winding2d_task{0,1}.json`;
`postproc` merges them into `reduced_xy.json`, computes the slope, and skips absent sections without crashing.

## 4. ⚠ The finding: soft exponent (b) should not be launched as scoped

Sizing the 1-D recipe required knowing the actual slip rates, and the probes contradict the task's premise.

**(i) The MSD estimator is saturation-dominated.** The through-origin fit `rate = <t·msd>/<t·t>` is only valid
while the walk is diffusive. It is not. Same run, same seed 31, N=8, `T/J=1.0`, varying only the fit window:

| window | `t <=` | `msd_end` | fitted rate |
|---|---|---|---|
| first 5% | 2 500 | 0.68 | **2.48e-4** |
| first 10% | 5 000 | 1.07 | 2.35e-4 |
| first 25% | 12 500 | 1.46 | 1.42e-4 |
| full | 50 000 | 1.32 | **4.04e-5** |

A factor **6** purely from window choice. Independent confirmation: the laptop's own 30 000-step N=8 run gives
`6.55e-5`, my identical-seed 50 000-step run gives `4.04e-5` — the "rate" decays as you watch longer. And in
the earliest window N=8 and N=32 have **near-identical rates** (2.48e-4 vs 2.43e-4, slope ≈ 0), i.e. the
reported `N`-scaling largely reflects *how fast different N saturate*, not a slip rate.

**(ii) Lowering T makes it worse, not better.** The scoped remedy was `T/J=0.5`. Measured (N=8, NW=256, 50k):

| `T/J` | 1.0 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|
| rate (N=8) | 4.04e-5 | 3.61e-5 | 3.42e-5 | 3.15e-5 |
| slope over N∈{8,32} | **0.39** | — | — | **0.15** |

The rate barely moves (−22%) and the N-slope **flattens toward 0**, the opposite of the hoped-for +1.

**(iii) Root cause — the winding is barely metastable.** `E_wind(N=8,w=1) = N·J·(1−cos 2π/N) = 0.234` vs
`T = 0.10` at `T/J=1.0`, so `E/T = 2.3`. The ring simply relaxes in ~10³ steps; there is no long-lived winding
whose lifetime could scale as `1/N`. A well-posed (unsaturated) window opens only at `T/J <= 0.2` (`E/T >= 11.7`):

| `T/J` | 1.0 | 0.5 | 0.3 | 0.2 | 0.1 |
|---|---|---|---|---|---|
| `E/T` | 2.34 | 4.69 | 7.81 | 11.72 | 23.43 |
| msd @2.5k | 0.680 | 0.199 | 0.027 | 0.000 | 0.000 |
| msd @50k | 1.316 | 1.059 | 0.977 | 0.797 | **0.148** |

**Consequence.** (b) is an **estimator problem first, a compute problem second**. Throwing A100 hours at
`T/J=0.5` will return a slope around 0.1–0.5 that must *not* be read as "the exponent is soft". Mitigations
shipped: an optional `--msd-fit-max` diffusive-window fit (default `None` keeps parity), and a runtime warning
when `msd_final > 1` with no window cut. **My recommendation to the Hub: don't spend GPU on (b) yet — switch
the 1-D arm to first-passage `tau` (the same estimator as the 2-D arm), which also makes the headline
1-D-degrades-vs-2-D-improves contrast apples-to-apples instead of comparing two different statistics.**
That is a small analyst/engineer task, not a cluster job.

**Soft exponent (a) is in good shape and cheap.** A 4-walker sizing probe already shows the `L^2` masking
dying out on schedule: `tau_med` at `T/J=1.10` goes 45 (L=16, laptop) → **89 → 97 → 85** (L=32/48/64), i.e.
the apparent `+1.1` slope collapses to `~-0.06` by `L>=32`. Pre-registered predictions in `PREREG.md`
(P2a slope < +0.5 at `T/J=1.10`; P2b slope < 0 at `T/J=1.30`; P2c slope > +2 below `T_KT`). Cost is minutes
per above-`T_KT` cell; the wallclock is dominated by below-`T_KT` cells censoring at `n_max` (measured
1.8/3.6/5.8 min per cell at L=32/48/64 for nwalk=24).

## 5. Git footprint

Branch **`agent/experiment-engineer/kt-csf3-tranche`** off `main` `a5978f6`; rebase onto local `main` = no-op
(base unmoved). **Not pushed, no PR** — left for review.

| commit | subject | files |
|---|---|---|
| `13b78df` | config: add ExperimentKTConfig | `chlu/config.py` |
| `1138a00` | promote the validated KT scripts into the tracked tree | `chlu/experiments/kt/{__init__,reduced_xy,clu_path,postproc,runner}.py` |
| `a74e5c4` | cli: add `chlu exp-kt` hook | `chlu/cli/experiment_cmd.py` |
| `de5aa78` | csf3: job_gpu_kt.sh | `scripts/csf3/job_gpu_kt.sh` |
| `d5cac9b` | tests: pin round-trip parity + settings guards | `tests/test_kt.py` |

No shared-file collisions: the concurrent `clu-cafe-integration` agent touched `scripts/cafe/`, disjoint from
everything above. Main checkout never edited by me. No unresolved conflicts.

## 6. Limitations / honesty

- **Round-trip is exact but the reference is single-seed-per-cell** on the CLU arms (256 walkers). Parity
  proves I preserved the code, not that the laptop numbers are statistically converged.
- **The 2-D `L>=32` probe is 4 walkers, one seed.** Suggestive, not a result. The launch exists to settle it.
- The `--msd-fit-max` window fit is **new code, not laptop-validated physics** — it is off by default and its
  only test is that it restricts the fit and raises the rate. Its *scientific* validity is the Hub's call.
- `postproc` figures live in `kt/postproc.py`, not `utils/plotting.py`. Deliberate: they are bespoke KT
  diagnostics (universal-jump line, WM extrapolation), and `plotting.py` is a heavily-shared 1048-line file I
  did not want to touch while another agent was live in the tree. Flagging as a small future consolidation.
- I have **no CSF access**: the sbatch script is syntax-checked (`bash -n`) and its logic exercised locally via
  the same `chlu exp-kt` calls, but it has never run under Slurm. First launch should be the cheap one below.
- `shellcheck` is not installed on this machine, so the script was not shellcheck-linted (directives kept
  consistent with the existing `job_gpu_*.sh` which were).

---

## 7. LAUNCH BLOCK — copy-paste for the Head

**Launch (a) now. Hold (b)** pending the Hub's call on §4.

```bash
# ---- on CSF3, once ----
cd ~/scratch/CHLU
git pull                          # carries chlu/experiments/kt/ + scripts/csf3/job_gpu_kt.sh
mkdir -p logs
export CLU_MAIL=<your address>    # parameterized on purpose; no address in the repo

# ---- (a) 2-D winding survival at L>=32: the sign-change run. CPU, ~2-3 h/seed ----
sbatch -p serial -G 0 -c 1 -t 8:00:00 -a 0-2 --mail-user=$CLU_MAIL \
       --export=ALL,MODE=winding2d,SEED_BASE=700,OUT=$HOME/scratch/clu_kt/w2d,\
EXTRA_ARGS='--l-values 16 24 32 48 64 --tj-values 0.60 0.70 1.00 1.10 1.20 1.30 --nwalk-2d 96' \
       scripts/csf3/job_gpu_kt.sh

# ---- collect: merge the 3 seed shards + write summary.json/figures (~1 min) ----
# substitute the array job id printed above for <JOBID>
sbatch -p serial -G 0 -c 1 -t 0:30:00 --dependency=afterany:<JOBID> --mail-user=$CLU_MAIL \
       --export=ALL,MODE=postproc,OUT=$HOME/scratch/clu_kt/w2d \
       scripts/csf3/job_gpu_kt.sh

# ---- pull results back ----
rsync -avz csf3:$HOME/scratch/clu_kt/w2d/ ./kt_w2d/
```

Optional, independent of (a) — hardens the kill criterion from L=8 to L=16 (**GPU**, A100):

```bash
sbatch -t 8:00:00 --mail-user=$CLU_MAIL \
       --export=ALL,MODE=bridge,OUT=$HOME/scratch/clu_kt/bridge,\
EXTRA_ARGS='--l-values 16 --tj-values 0.70 0.85 1.00' \
       scripts/csf3/job_gpu_kt.sh
```

**Do not run the (b) recipe** in the script header until the Hub rules on §4; it is written out there, with its
warning attached, so it is ready the moment that call is made.

---

## Proposed handover updates (for the Hub)

### For §7 (Known Issues) — NEW: the 1-D winding MSD estimator saturates
> **7.23 [OPEN, methodology] The 1-D winding MSD slip-rate estimator is saturation-dominated at the settings
> used.** `kt-2d-csf3`'s soft `-0.7` slope is ascribed there to `xi~1.2` at `T/J=1.0`; measurement
> (`kt-csf3-tranche`, `d5cac9b`) shows it is largely a **fit-window artifact**: the same run/seed fits
> `2.5e-4` over `t<=2500` vs `4.0e-5` over `t<=50000` (6x), and in the earliest window the N-scaling vanishes
> (N=8 and N=32 rates equal to 2%). Root cause: `E_wind(N=8,w=1)/T = 2.3` — the ring winding is barely
> metastable and relaxes in ~1e3 steps. **Lowering `T/J` 1.0→0.5 flattens the slope (0.39→0.15), so the
> recommended remedy is counter-productive.** A well-posed window needs `T/J <= 0.2`. Mitigation shipped:
> `run_winding_msd(msd_fit_max=...)` + a runtime warning; default unchanged (parity preserved).
> **Recommended fix: replace the 1-D MSD arm with first-passage `tau`**, matching the 2-D estimator and making
> the flagship 1-D-vs-2-D memory contrast apples-to-apples.

### For §5 / wherever the KT tranche is described — the scripts are now tracked
> **The `kt-2d-csf3` A100 tranche is launchable** (`kt-csf3-tranche`, branch
> `agent/experiment-engineer/kt-csf3-tranche`, 5 commits off `a5978f6`, suite **341 green**). The five scratch
> scripts are promoted to **`chlu/experiments/kt/`** behind **`chlu exp-kt --mode {winding1d,winding2d,bridge,
> reduced,postproc}`**, config-driven via **`ExperimentKTConfig`**, with **`scripts/csf3/job_gpu_kt.sh`**
> (`-n 1 -c 8`, separate `-e`, `-a 0-2%3`, `--mail-user=$CLU_MAIL`, sized recipes in the header, no dataset
> download). Promotion verified **bit-exact** against the committed laptop JSONs (absdiff `0.00e+00` incl. all
> 21 E-bridge kill-criterion fields). `float64` + `langevin_noise="fdt"` + `newtonian_learned` + no-governor
> are **asserted in-process and in an sbatch preflight**, so a §7.22 misconfiguration fails loudly.
> **Soft exponent (a) is ready to launch; (b) is blocked on §7.23, not on compute.**

### For §5 provenance — sizing evidence for (a)
> A 4-walker probe (not a result) shows the `L^2` masking dying out as predicted: 2-D winding `tau_med` at
> `T/J=1.10` goes **45 (L=16) → 89 → 97 → 85 (L=32/48/64)**, slope `~-0.06` over `L>=32` vs the laptop's
> apparent `+1.1`. Predictions pre-registered in `.claude/outputs/kt-csf3-tranche/PREREG.md` before launch.

### For §3 (config defaults)
> New `experiment_kt` block in `CHLUConfig`. Note it deliberately **does not** inherit the repo-wide
> `langevin_noise="legacy"` default — it pins `"fdt"` and asserts it (§7.22).
