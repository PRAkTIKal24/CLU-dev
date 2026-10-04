# PRE-FLIGHT for a 12-hour unattended run on FORGIS_14
**Run these before launching a hub. Not one of them is optional.**

Measured state as of 2026-09-16, before any change:

| setting | value | verdict |
|---|---|---|
| last boot | 2026-09-12 00:02 | 4 days uptime |
| CBS RebootPending | False | ok |
| WU RebootRequired | False | ok |
| **PendingFileRenameOperations** | **True** | ⚠ something is staged for the next boot |
| **sleep on AC (`STANDBYIDLE`)** | **0x12c = 300 s = 5 MINUTES** | 🔴 **the run-killer** |
| hibernate on AC (`HIBERNATEIDLE`) | 0 (disabled) | ok |
| disk timeout AC | 30 s | harmless |
| power scheme | Balanced | see step 2 |
| power source | AC (BatteryStatus = 2) | ok |
| Active Hours | not configured | ⚠ no protection against an auto-reboot |

---

## 🔴 Step 1 — the real hazard is not the update popup, it is SLEEP
**This machine is set to sleep after 5 minutes idle on AC.** Windows idle detection keys off *user input*,
not CPU load — a long-running CPU-bound Python process does **not** by itself keep the machine awake. An
unattended 12-hour run on these settings is at serious risk of being suspended minutes after you walk away.

```powershell
powercfg /change standby-timeout-ac 0      # never sleep on AC
powercfg /change hibernate-timeout-ac 0    # belt and braces (already 0)
powercfg /change monitor-timeout-ac 10     # screen may still blank; that is fine and saves power
```
Verify afterwards — `0x00000000` is what you want:
```powershell
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String 'Current AC Power Setting'
```
⭐ **Remember to put it back after the campaign** (`powercfg /change standby-timeout-ac 30`).

## Step 2 — Windows Update, and then stop it reaching in mid-run
Do the update and reboot **first**, as planned. `PendingFileRenameOperations = True` says something is
already staged for the next boot, so a reboot is warranted regardless.

⚠ But the popup was never the real problem — **an automatic restart is.** After updating, close the door:
```powershell
# Pause further updates for the campaign window (Settings > Windows Update > Pause updates) —
# or set Active Hours to span the run so Windows will not auto-restart inside it:
Set-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' ActiveHoursStart 0
Set-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' ActiveHoursEnd   23
```
(Active Hours caps at an 18-hour span on most builds; pausing updates is the stronger guarantee.)

## Step 3 — confirm the reboot actually cleared the staged work
After the update reboot, re-check. All three should be `False`:
```powershell
Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending'
Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired'
[bool](Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' -Name PendingFileRenameOperations -EA SilentlyContinue)
```

## Step 4 — re-verify the workspace after the reboot
A reboot clears the shell environment, and the registry-scoped vars (`PYTHONUTF8`, `CLOUDSDK_PYTHON`) only
reach **newly launched** shells — which is exactly what you will have. Confirm the workspace still stands:
```bash
cd /c/Users/PratikJawahar/chlu/chlu-bundle-v2
unset $(env | grep ^CHLU_ | cut -d= -f1) 2>/dev/null
. ./env_hub.sh
env | grep ^CHLU_        # every path must contain chlu-bundle-v2
export PYTHONUTF8=1      # ⚠ STILL LOAD-BEARING — see below
VERIFY_PYTHON="$CHLU_ROOT/.venv/Scripts/python.exe" bash scripts/verify_hub_workspace.sh
```

### ⚠ `PYTHONUTF8=1` is still load-bearing on v2, for a NEW reason
v2 fixed the *read* side (`encoding="utf-8"` on file reads) but **stdout is still cp1252**. Without UTF-8
mode, the first encode dies here:
```
gen_bridge.py:915  _announce_encode_precision -> log(...)
UnicodeEncodeError: 'charmap' codec can't encode character '⭐' in position 2
```
⛔ That is on the **encode path**, not in a diagnostic corner. It is persisted at user scope so a fresh
shell inherits it — but **prove it is set in the hub's actual shell before launching**, because a 12-hour
run that dies at minute one on a star character is a stupid way to lose a day.

## Step 5 — survive losing the agent thread
⚠ On GCP, a lost connection did not kill the job; **here the compute is local**, so if the launching
process dies the work dies with it unless it was written to be resumable.
- ⭐ Structure the run as **resumable units with a ledger**, the way `scripts/csf3/job_cpu_gen.sh` does —
  its design note is that *"a kill costs ONE UNIT, never the run."*
- Pin threads per unit (`OMP_NUM_THREADS` / `MKL_NUM_THREADS`) rather than letting JAX and BLAS each grab
  all 16 cores; an unbound numpy/sklearn oversubscribes and runs **slower**.
- Write progress to `$CHLU_HUB_OUTPUTS/<slug>/` as you go. §0.1 **YOU NEVER DELETE** applies throughout.

## Step 6 — record the start stamp
The 12-hour budget is **true wall clock**, so it needs an anchor that is not the agent's own sense of time:
```bash
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$CHLU_HUB_OUTPUTS/<slug>/START_UTC.txt"
```
Check elapsed against that file, never against recollection.
