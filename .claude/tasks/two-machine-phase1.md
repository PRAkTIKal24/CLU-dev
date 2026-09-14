# PHASE 1 — prep you can do NOW
Touches no repo, no remote, no running wave. Phases 2–4 stay gated on the Head's go.

## A. ThinkPad — WSL2 base
In **PowerShell as Administrator**:
```
wsl --install -d Ubuntu
wsl --set-default-version 2
wsl -l -v          # confirm VERSION = 2, not 1
```
Then **inside the Ubuntu shell**, one block:
```
sudo apt update && sudo apt install -y git tmux build-essential curl
curl -LsSf https://astral.sh/uv/install.sh | sh
git config --global core.autocrlf false
git config --global core.longpaths true
git config --global user.name "Pratik Jawahar"
git config --global user.email "info@forgis.com"
ssh-keygen -t ed25519 -C "thinkpad-wsl2" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```
⚠ `core.autocrlf false` is not optional — CRLF silently breaks every bash script the moment it reaches CSF3.
Add that printed public key to GitHub (it needs access to `CLU-dev` only).

## B. Tailscale — both machines
ThinkPad: install **inside WSL2**, not on Windows. WSL2 then gets its own tailnet address and no Windows→WSL port-proxy is needed.
```
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```
Mac: install the app (not currently present), sign in to the same tailnet.
Verify **both directions**: `tailscale ping <other-host>`.

## C. SSH server inside WSL2
```
sudo apt install -y openssh-server
sudo ssh-keygen -A
sudo service ssh start
```
Append your Mac's public key to `~/.ssh/authorized_keys`. Test from the Mac over the tailnet.
Make it survive reboot (WSL2 has no systemd by default for this) — add to `/etc/wsl.conf`:
```
[boot]
command = "service ssh start"
```

## D. Windows must not nap
Settings → Power: **plugged in → never sleep**, and lid-close → do nothing. Otherwise the "stationary server" naps mid-sweep.
Also confirm: **Windows Pro or Home?** (`winver`) — Pro has RDP built in; Home needs RustDesk instead.

## E. Zed on Windows
Install, sign in to the same account, confirm settings + keybindings arrive. Open the WSL2 project natively.

## F. ⭐ De-risk the remote path BEFORE betting CHLU on it
Zed SSH from the Mac into WSL2 on a **throwaway repo**. Run one ACP thread. Pull the network / close the lid.
Observe what actually happens to the thread. This is the known-rough part of the whole plan — find out now, on something disposable.

## G. Mac — the one untested interaction
Once Tailscale is installed: connect GlobalProtect, then check whether Tailscale still routes.
Full-tunnel GlobalProtect can capture everything and break the tailnet. If it does → split-tunnel, or raise the VPN only for the CSF3 rsync window.

## H. Mac — enable inbound access (for the reverse direction)
System Settings → General → Sharing → **Remote Login** on, and **Screen Sharing** on. Both reachable over Tailscale.
