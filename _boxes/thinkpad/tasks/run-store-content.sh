cd /c/chlu/worktrees/CHLU-hub-store-content
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-store-content.txt)"
