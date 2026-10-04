cd /Users/batmac/clu/worktrees/CHLU-w20-citation-prune
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w20/launch-citation-prune.txt)"
