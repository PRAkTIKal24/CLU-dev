cd /c/chlu/worktrees/CHLU-w13-seeds-and-tables
source /c/chlu/env_w3.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-w13-seeds-and-tables.txt)"
