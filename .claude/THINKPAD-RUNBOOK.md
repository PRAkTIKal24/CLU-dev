# THINKPAD RUNBOOK — launching and watching autonomous hubs from the Mac
**Every command here was executed and verified 2026-09-18. ⛔ Do not rediscover this.**

> ⚠⚠ **SUPERSEDED IN PART — read `.claude/AUTOHUB-SSH-RECIPE.md` first** (verified 2026-09-19 by
> actually launching wave 3). This file is still correct on **method**; two of its **facts** are wrong:
> 1. ⛔ §3's *"`sshkey` CANNOT read `C:\Users\PratikJawahar\**`"* is **FALSE** — it reads it fine once
>    `git config --global --add safe.directory …` is set. ⭐ The 2 GB frozen instrument never needs copying.
> 2. ⚠ §0's *"the remote default shell is `cmd`"* — it is **PowerShell 5.1**. The heredoc-pipe pattern
>    §0 prescribes is still exactly right, and bypasses the question entirely.
>
> ⛔ The recipe also documents **the two interactive dialogs that silently stall every hub** (folder-trust
> and bypass-permissions acceptance) — not mentioned here, and each one looks like a hub that is merely
> thinking.

---

## 0. ⛔⛔ THE QUOTING RULE — read this first or you will waste an hour
**The remote default shell is `cmd`, not bash.** Three layers of quoting (your shell → ssh → cmd → bash) eat
`$vars`, backslashes and `;`. ⭐ **ALWAYS pipe a heredoc'd script. ⛔ NEVER inline-quote a remote command.**

```bash
# ⭐ THE ONLY PATTERN THAT RELIABLY WORKS
cat > /tmp/job.sh <<'EOF'          # ⛔ QUOTED 'EOF' — or your Mac's bash expands $vars before sending
  <commands, plain bash, $HOME etc. resolve REMOTELY>
EOF
ssh -o BatchMode=yes forgis-14 'C:\msys64\usr\bin\bash.exe -l' < /tmp/job.sh
```
⛔ **Failures this prevents, each of which actually happened:**
- `ssh host "bash -lc \"...\""` → nested quotes collapse, `unexpected EOF while looking for matching '"'`.
- `powershell -Command "... $env:USERPROFILE ..."` → **your Mac's bash ate `$env`**, leaving `:USERPROFILE\...`.
- `cmd.exe /c "installer.exe"` inside a piped script → **cmd consumed the rest of the script from stdin.**
  ⭐ Add `< /dev/null` to any Windows `.exe` you invoke.

## 1. Reaching it
```bash
ssh forgis-14                      # user `sshkey`, key-based, no password
```
⛔ **`AzureAD\PratikJawahar` over SSH is a DEAD END** — an Azure AD account has no local credential for sshd to
build a logon session from, so key auth succeeds and the session dies (`Connection reset`). ⛔ **Do not retry it.**

## 2. What is installed (verified on a fresh login shell)
`claude 2.1.276` · `python 3.11.9` **64-bit AMD64** (⭐ the build JAX's `win_amd64` wheels need) · `uv 0.12.16` ·
`node v22.20.0` · `git 2.55.0` · `rsync 3.5.0` · `tmux 3.7c`.
⭐ **PATH is persistent** in `~/.bash_profile` — a fresh login shell has all of it. ⛔ Don't re-export it.

## 3. ⭐ The shared directory
**`C:\chlu`** (`/c/chlu` from bash) has dual ACLs — **both `sshkey` and `AzureAD\PratikJawahar` have full control.**
⭐ **Hubs write here, so nothing strands on a private profile.** ⚠ `sshkey` CANNOT read
`C:\Users\PratikJawahar\**` — anything there needs the Head to copy it out.
```
/c/chlu/worktrees/   hub worktrees (code)
/c/chlu/outputs/     hub outputs — reports, PREREGs, LOG.md
/c/chlu/returns/     return packs + git bundles
/c/chlu/scratch/     launch directives and addenda
```

## 4. ⭐⭐ LAUNCHING A HUB
```bash
cat > /tmp/launch.sh <<'EOF'
cd /c/chlu/workspace
git worktree add ../CHLU-hub-<target> -b hub/<target> <PIN>    # ⛔ NOT `git checkout -b` — see below
tmux kill-session -t hubs 2>/dev/null
tmux new-session -d -s hubs -n <target>
tmux send-keys -t hubs:<target> 'cd /c/chlu/CHLU-hub-<target> && claude "<the full hub launch line>"' Enter
tmux ls
EOF
ssh -o BatchMode=yes forgis-14 'C:\msys64\usr\bin\bash.exe -l' < /tmp/launch.sh
```
⛔⛔ **ONE WORKTREE PER HUB.** `git checkout -b` in a shared directory means agent 2 yanks the files out from under
agent 1 mid-edit. ⭐ Additional hubs: `tmux new-window -t hubs -n <target>`.
⭐ **Proven:** tmux survives disconnect, and `claude` inside it runs to completion while you are gone.
⭐ **`claude` works NON-INTERACTIVELY** (`claude -p "..."`), auth persists across fresh connections.

## 5. Watching, without disturbing anything
```bash
cat > /tmp/peek.sh <<'EOF'
tmux list-windows -t hubs
echo "--- log ---";    tail -40 /c/chlu/outputs/hub-<target>/LOG.md
echo "--- prereg ---"; ls -la /c/chlu/outputs/hub-<target>/ | grep -i prereg
EOF
ssh -o BatchMode=yes forgis-14 'C:\msys64\usr\bin\bash.exe -l' < /tmp/peek.sh
```
⭐ **`LOG.md` beats scraping panes** — structured, written as the hub goes, survives a pane scrolling past its limit.
⭐ **A new PREREG appearing = a scored run is imminent.** That is the one moment worth watching.
`tmux capture-pane -p -t hubs:<target> -S -100` shows liveness when a hub has gone quiet.

## 6. ⛔⛔ COLLECTING — rsync fails SILENTLY without this flag
```bash
rsync -az --rsync-path='C:\msys64\usr\bin\rsync.exe' forgis-14:/c/chlu/ <dest>/
```
⛔ **Without `--rsync-path` the remote shell is `cmd`, rsync is not found, and it transfers NOTHING and exits 0.**
⚠ A "successful" backup that moved zero bytes. ⭐ **Always check the file count afterwards.**
Filter to the scientific record (the rest is ~9.9 GB of `.npz` dev caches):
`--include='*/' --include='*.md' --include='*.json' --include='*.py' --include='*.sh' --include='*.log' --include='*.csv' --include='*.bundle' --exclude='*'`

## 7. ⛔ A hub's return needs BOTH halves
| mechanism | carries | misses |
|---|---|---|
| `git bundle create <f> --all` | tracked code | **everything in `.claude/`** (gitignored) |
| `--return-pack <slug>` | `.claude/outputs/<slug>/**` | **all code** |
⛔⛔ **`git status` does NOT list gitignored files** — a worktree holding an entire unsaved outputs tree reports
clean. ⭐ **Check with a directory listing, never with git.** This nearly stranded five hubs' work.

## 8. Gotchas worth knowing before you hit them
- **Windows installers take minutes.** Python showed `python.exe` present while still installing; `pip` only
  appeared at the end. ⭐ Poll for the process to exit, don't conclude from a partial tree.
- **`robocopy` exit codes 0–7 are SUCCESS** (8+ is failure). Counterintuitive.
- **Windows globs are case-insensitive** — `CHLU-hub-*` also matched `chlu-hub-bundle`.
- **`.claude/` lives only in the main checkout**, never in a worktree. A hub writing "notes" from inside a worktree
  silently creates a second `.claude/` nobody reads. ⭐ Write to `$CHLU_HUB_OUTPUTS/<slug>/`.
- ⭐ **A stale `CHLU_*` in your shell silently overrides the derived root.** `unset` before sourcing `env_hub.sh`.
