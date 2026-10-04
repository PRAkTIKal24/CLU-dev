cd /Users/batmac/clu/worktrees/CHLU-w15-table-fixes
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w15/launch-table-fixes.txt)"
