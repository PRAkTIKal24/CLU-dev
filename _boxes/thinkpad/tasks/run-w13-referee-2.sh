cd /c/chlu/worktrees/CHLU-w13-referee-2
source /c/chlu/env_w3.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-w13-referee-2.txt)"
