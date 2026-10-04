cd /c/chlu/worktrees/CHLU-w10-draft-v0
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-w10-draft-v0.txt)"
