cd /Users/batmac/clu/worktrees/CHLU-w21-registry-fold
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w21/launch-registry-fold.txt)"
