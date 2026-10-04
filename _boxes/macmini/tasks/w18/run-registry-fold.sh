cd /Users/batmac/clu/worktrees/CHLU-w18-registry-fold
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w18/launch-registry-fold-pass4.txt)"
