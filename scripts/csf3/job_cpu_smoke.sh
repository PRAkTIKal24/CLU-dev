#!/bin/bash --login
# scripts/csf3/job_cpu_smoke.sh — cheap CPU-only smoke test of the CLU stack.
#
# Verifies: venv activates, chlu imports & CLI runs, core physics tests pass
# (symplecticity/energy/mass/grads), matplotlib is headless-safe.
# Expected runtime: ~5 min warm; first-ever run may take longer while XLA
# compiles cold (populates $JAX_COMPILATION_CACHE_DIR for later jobs).
#
# SUBMIT:  cd ~/scratch/CHLU && sbatch scripts/csf3/job_cpu_smoke.sh
#
#SBATCH -p serial            # 1-core Intel CPU partition
#SBATCH -t 1:00:00           # generous buffer for a cold XLA cache
#SBATCH --job-name=clu-smoke-cpu

module purge
set -eo pipefail

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
# Non-GPU node: force the CPU backend so the CUDA plugin doesn't probe/warn.
export JAX_PLATFORMS=cpu
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"

echo "=== 1. CLI + version ==="
chlu --version
chlu info

echo "=== 2. headless matplotlib (Agg) ==="
python - <<'PY'
import matplotlib
assert matplotlib.get_backend().lower() == "agg", matplotlib.get_backend()
import matplotlib.pyplot as plt
fig = plt.figure(); plt.plot([0, 1], [0, 1]); fig.savefig("/tmp/clu_agg_smoke.png")
print("Agg backend OK, wrote /tmp/clu_agg_smoke.png")
PY

echo "=== 3. core physics tests ==="
python -m pytest -q tests/test_core.py tests/test_data.py

echo "=== CPU smoke PASSED ==="
