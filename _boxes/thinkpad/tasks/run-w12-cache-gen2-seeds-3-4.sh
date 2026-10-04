cd /c/chlu/worktrees/CHLU-w12-cache-gen2-seeds-3-4
source /c/chlu/env_w3.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-w12-cache-gen2-seeds-3-4.txt)"
