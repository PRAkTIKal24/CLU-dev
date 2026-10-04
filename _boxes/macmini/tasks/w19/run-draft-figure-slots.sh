cd /Users/batmac/clu/worktrees/CHLU-w19-draft-figure-slots
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w19/launch-draft-figure-slots.txt)"
