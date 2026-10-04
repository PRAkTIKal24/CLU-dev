cd /Users/batmac/clu/worktrees/CHLU-w18-referee-fixes
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w18/launch-referee-fixes.txt)"
