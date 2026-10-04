cd /Users/batmac/clu/worktrees/CHLU-w17-venue-scout
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w17/launch-venue-scout.txt)"
