# Task: csf3-runbook — cluster execution path for CLU experiments (2×A100, Manchester CSF3)

- **Agent:** `experiment-engineer` · **Base:** `main` · **Branch:** only if committing scripts (recommended: `agent/experiment-engineer/csf3-runbook`; scripts under `scripts/csf3/`) · **Output:** `.claude/outputs/csf3-runbook.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, handover §6 (env caveats), roadmap D3.
- **Hard constraint:** you (the agent) run on the Head's laptop and likely have **no CSF3 credentials/SSH access**. This task PREPARES everything so the Head's first real submission is copy-paste; anything you cannot execute, you mark UNTESTED with the exact command for the Head to run. Do not guess-and-assert cluster facts — label pattern-based assumptions.

## Deliverables
1. **Runbook doc** (in the output file + `scripts/csf3/README.md`): CSF3 access pattern (login nodes, filesystem layout, batch system — **verify which scheduler CSF3 currently uses (SGE/qsub heritage vs SLURM) from the official Manchester CSF3 docs on the web**; cite the doc page), GPU partition request syntax for A100s, module/env strategy.
2. **Environment recipe:** uv-based or conda-based env build for jax[cuda] on A100 (pin jax/jaxlib CUDA versions compatible with the cluster's CUDA driver per docs; provide both a `uv sync --extra cuda`-style path if we add an extra, and a plain pip fallback). Include the JAX persistent-compilation-cache setup (big win given our cold-start pain) and `XLA_PYTHON_CLIENT_MEM_FRACTION` guidance for shared nodes.
3. **Batch templates** (`scripts/csf3/`): (a) single-GPU training job template with our CLI (`chlu exp-… --project …`) or python -m entry, (b) array-job template for seed sweeps, (c) rsync push/pull helpers for `projects/<name>/` artifacts (results/models/plots) between laptop and cluster — mind that `projects/*/` is gitignored, so artifact sync is rsync, not git.
4. **Repo touch-ups if needed:** anything small that blocks headless cluster runs (e.g., matplotlib Agg backend guard, `fetch_openml` cache dir env var for offline nodes, dataset path overrides via env/config). Keep minimal; commit on the branch.
5. **Smoke-test plan for the Head:** an ordered checklist — (1) env build command, (2) 5-min CPU smoke job, (3) single-A100 exp-a quick job, (4) artifact pull — each with expected output. The Head executes; you specify.

**Verify what you can locally:** templates lint/bash -n clean; the headless-matplotlib path actually works (`MPLBACKEND=Agg uv run --no-sync python -c "...plot smoke..."`); document everything else as UNTESTED-BY-AGENT.
