cd /c/chlu/worktrees/CHLU-w15-cmaze-t-harness
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w15/launch-cmaze-t-harness.txt)"
