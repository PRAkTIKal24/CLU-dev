cd /c/chlu/worktrees/CHLU-w17-draft-v2
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w17/launch-draft-v2.txt)"
