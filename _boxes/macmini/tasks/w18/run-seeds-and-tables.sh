cd /Users/batmac/clu/worktrees/CHLU-w18-seeds-and-tables
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w18/launch-seeds-and-tables-pass4.txt)"
