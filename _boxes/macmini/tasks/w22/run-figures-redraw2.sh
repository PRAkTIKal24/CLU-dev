cd /Users/batmac/clu/worktrees/CHLU-w22-figures-redraw2
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/w22/launch-figures-redraw2.txt)"
