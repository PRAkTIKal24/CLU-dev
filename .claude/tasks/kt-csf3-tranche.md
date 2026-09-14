# Task: kt-csf3-tranche — make the KT/A100 confirmation-at-scale run actually launchable (experiment-engineer)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/kt-csf3-tranche.md`
- **Read first:** protocol · `.claude/outputs/kt-2d-csf3.md` (**§7 "outstanding A100 tranche"** + the PREREG + the two soft exponents in §5) · `.claude/scratch/kt-2d-csf3/{kt_clu.py, reduced_xy.py, kt_winding1d.py, kt_winding_msd.py, postproc.py}` (the analyst's laptop scripts — the physics is DONE and validated, this task is packaging) · `scripts/csf3/job_gpu_single.sh` + `job_gpu_array_seeds.sh` (the sbatch conventions) · the handover's CSF3 block (`-n 1 -c 8`, separate `-e`, `logs/`, array throttle, `$CLU_MAIL`).
- **Git:** branch off local `main`. Do NOT rebase onto `origin` (frozen — handover §7.21).

## Why this task exists (the blocker)
The KT physics is **confirmed** on the real CLU 2-D path, but the A100 tranche is **not launchable as-is**: the scripts live in **`.claude/scratch/`, which is gitignored**, so the Head's workflow (commit → push `clu-dev/main` → `git pull` on CSF3) will never carry them to the cluster. There is also **no sbatch wrapper** — `kt-2d-csf3.md` says "CSF3-ready (`job_gpu_single.sh` *pattern*)", i.e. patterned after, not written. Fix both.

## Items
1. **Promote the scripts into the tracked tree.** Move/adapt the five scratch scripts into a tracked home consistent with repo conventions (suggest `chlu/experiments/kt/` with a thin CLI entry, or `scripts/kt/` if they are genuinely standalone research scripts — pick one, justify it, don't scatter). Preserve the validated physics **exactly** — this is a packaging task, not a rewrite. If you refactor for the CLI, a numerical round-trip check against the committed `.claude/outputs/kt-2d-csf3/*.json` is the acceptance gate.
2. **Write the sbatch wrapper** (`scripts/csf3/job_gpu_kt.sh`), mirroring the current conventions: `-n 1 -c 8`, separate `-e`, `#SBATCH -o logs/%x-%j.out`, array throttle for the seed/L sweep, `--mail-user=$CLU_MAIL` (**parameterized — Head confirmed 2026-07-20, do NOT hardcode an address**). No dataset download needed (this is synthetic/self-generated — confirm and say so).
3. **Target the two soft exponents specifically** (from `kt-2d-csf3.md` §5/§7 — these are the *only* reason the run exists):
   (a) **2-D winding survival at `L≥32`** — resolve the sign change of `τ ∝ L^{πρ_s/T−2}` above `T_KT`, which was unresolved at `L≤16` because vortex-diffusion traversal (`∝L²`) masks the negative Arrhenius exponent at small `L`.
   (b) **1-D lower-`T` run** — recover the clean `τ ∝ 1/N` slope `−1` (laptop gave `−0.7`, `ξ≈1.2` at `T/J=1.0`).
   Size both (walltime/seeds/L values) and put the launch recipe in the job-script header, as `job_gpu_eval.sh` does.
4. **Settings discipline:** `float64`, `langevin_noise="fdt"` (⚠ handover §7.22 — `legacy` is the repo default and T is **not** in energy units there; none of this physics holds under it), no scorer. Assert the flags in-script so a misconfigured run fails loudly instead of silently producing garbage.
5. **Local smoke first** (Head's standing rule: never spend CSF to find a local bug) — a tiny-`L`, few-step run proving the path executes end-to-end and `postproc.py` consumes the output.
6. **Carry the number correction:** `T_KT = 1.786κr*²` at `κ=0.05` is **0.0893** (`= 0.8929 J`), **NOT "0.1786"** — the slip is annotated in `xy-1d-control.md`. Make sure nothing you write inherits it.

## Acceptance
Scripts tracked (so `git pull` on CSF3 carries them) + a conventions-compliant `job_gpu_kt.sh` with a sized launch recipe in its header + local smoke passing + numerical round-trip vs the committed laptop JSONs + suite green. **Deliver the exact `sbatch` command line for the Head to run** — the engineer has no CSF access, so the report must end with a copy-pasteable launch block. Report anything about the physics you could not preserve faithfully.
