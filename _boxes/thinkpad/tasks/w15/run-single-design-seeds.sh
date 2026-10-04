cd /c/chlu/worktrees/CHLU-w15-single-design-seeds
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/w15/launch-single-design-seeds.txt)"
