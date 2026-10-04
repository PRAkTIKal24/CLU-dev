cd /c/chlu/worktrees/CHLU-hub-unseen-valid
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-unseen-valid.txt)"
