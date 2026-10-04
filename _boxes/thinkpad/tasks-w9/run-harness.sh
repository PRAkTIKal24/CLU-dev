cd /c/chlu/worktrees/CHLU-w9-harness
source /c/chlu/env_w9.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks-w9/launch-harness.txt)"
