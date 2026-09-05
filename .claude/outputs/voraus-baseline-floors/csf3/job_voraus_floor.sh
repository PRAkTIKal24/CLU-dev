#!/bin/bash --login
# voraus-AD baseline floor on CSF3 — CPU only (sklearn/numpy), NO GPU needed.
# Submit from ~/scratch/CHLU. Partition: multicore (AMD Genoa, 7-day max).
#SBATCH -p multicore
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH -J voraus_floor
#SBATCH -o slurm-voraus-%j.out
set -euo pipefail
module purge
cd "${CLU_REPO:-$HOME/scratch/CHLU}"
# env.sh activates .venv (must have been built WITH --extra eval, see README)
source scripts/csf3/env.sh
export CHLU_DATA_ROOT="$HOME/scratch/chlu_data"

# Data: rsync'd parquet preferred (sha c90ab1c7...); else download on this
# compute node (has internet). Uncomment to auto-fetch:
# python -c "from chlu.data.industrial.voraus_ad import VorausAD; VorausAD(root='$CHLU_DATA_ROOT/voraus_ad', download=True)"

python .claude/scratch/voraus-baseline-floors/voraus_floor.py \
  --root "$CHLU_DATA_ROOT/voraus_ad" \
  --window 100 --train-stride 10 --max-train-windows 100000 --test-stride 5 \
  --columns-limit 0 \
  `# test_stride=5: KNN/LOF scoring dominates wall-time; episode mean-reduce is`\
  `# near-insensitive to test stride (measured), so this is ~5x faster, ~lossless.`\
  --out projects/voraus_floor/results
echo "=== voraus floor DONE ==="
