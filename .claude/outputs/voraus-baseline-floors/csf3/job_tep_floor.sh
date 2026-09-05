#!/bin/bash --login
# TEP-Rieth baseline floor on CSF3 — CPU only. The scale/CC0 set (10 000 test
# units) — the budget stressor. Submit from ~/scratch/CHLU.
#SBATCH -p multicore
#SBATCH -n 24
#SBATCH -t 1-0
#SBATCH -J tep_floor
#SBATCH -o slurm-tep-%j.out
set -euo pipefail
module purge
cd "${CLU_REPO:-$HOME/scratch/CHLU}"
source scripts/csf3/env.sh          # .venv MUST be built with --extra eval
export CHLU_DATA_ROOT="$HOME/scratch/chlu_data"

# Download the 3 needed RData on THIS compute node (internet). Faulty-training
# (471 MB) is NOT needed — canonical train = fault-free-training only.
python - <<'PY'
import os
from chlu.data.industrial.tep_rieth import TEPRieth
root=os.environ["CHLU_DATA_ROOT"]+"/tep_rieth"
# fetch fault-free pair (default) then the faulty-testing set:
try:
    ds=TEPRieth(root=root)  # errors if nothing present
except Exception:
    pass
import chlu.data.industrial.tep_rieth as T
d=T.TEPRieth.__new__(T.TEPRieth)
from pathlib import Path; d.root=Path(root)
d.fetch(keys=("fault_free_training","fault_free_testing","faulty_testing"))
print("fetched:", sorted(p.name for p in Path(root).glob("*.RData")))
PY

python .claude/scratch/voraus-baseline-floors/tep_floor.py \
  --root "$CHLU_DATA_ROOT/tep_rieth" \
  --window 100 --train-stride 2 --max-train-windows 100000 --test-stride 1 \
  --metrics-mode fast \
  --out projects/tep_floor/results
echo "=== TEP floor DONE ==="
