cd /Users/batmac/clu/worktrees/CHLU-w20-referee4-fixes
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w20/launch-referee4-fixes.txt)"
