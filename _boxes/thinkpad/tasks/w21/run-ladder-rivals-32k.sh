cd /c/chlu/worktrees/CHLU-w21-ladder-rivals-32k
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w21/launch-ladder-rivals-32k.txt)"
