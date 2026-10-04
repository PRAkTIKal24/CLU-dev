cd /c/chlu/worktrees/CHLU-w8-env-scout
source /c/chlu/env_w8.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w8/launch-env-scout.txt)"
