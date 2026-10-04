cd /c/chlu/worktrees/CHLU-w21-rel-newseeds
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w21/launch-rel-newseeds.txt)"
