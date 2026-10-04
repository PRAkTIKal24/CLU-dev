cd /Users/batmac/clu/worktrees/CHLU-w22-draft-rulings
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w22/launch-draft-rulings.txt)"
