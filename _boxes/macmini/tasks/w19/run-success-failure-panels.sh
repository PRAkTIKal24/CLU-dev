cd /Users/batmac/clu/worktrees/CHLU-w19-success-failure-panels
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w19/launch-success-failure-panels.txt)"
