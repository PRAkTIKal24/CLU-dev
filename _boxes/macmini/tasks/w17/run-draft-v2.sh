cd /Users/batmac/clu/worktrees/CHLU-w17-draft-v2
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w17/launch-draft-v2.txt)"
