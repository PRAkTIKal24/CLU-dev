cd /c/chlu/worktrees/CHLU-w17-venue-scout
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w17/launch-venue-scout.txt)"
