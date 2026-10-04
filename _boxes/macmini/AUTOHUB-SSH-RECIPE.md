# AUTO-HUB SSH LAUNCH RECIPE — the process that actually worked
**Executed end to end and verified 2026-09-19, launching wave 3's four hubs on FORGIS_14.
⛔ Do not rediscover any of this.** Companion to `THINKPAD-RUNBOOK.md`, which is still correct on
method — ⚠ but wrong on two facts. See §1.

---

## 0. ⭐⭐ THE ONE PATTERN — everything else is detail
**Never inline-quote a remote command. Always pipe a heredoc'd script into msys64 bash.**

```bash
cat > /tmp/job.sh <<'EOF'        # ⛔ QUOTED 'EOF' or your Mac's bash expands $vars before sending
  <plain bash; $HOME, $PWD etc. resolve REMOTELY>
EOF
ssh -o BatchMode=yes forgis-14 'C:\msys64\usr\bin\bash.exe -l' < /tmp/job.sh
```
⭐ This bypasses the remote default shell entirely, so **you never care what it is.**
⛔ Add `< /dev/null` to any Windows `.exe` you invoke inside a piped script — it will otherwise
consume the rest of your script from stdin.

## 1. ⚠ TWO CORRECTIONS TO `THINKPAD-RUNBOOK.md` — both cost time on 2026-09-19
1. ⛔ **"`sshkey` CANNOT read `C:\Users\PratikJawahar\**`" is FALSE.** It reads it fine. The only
   obstacle is git's ownership guard, cleared once:
   ```bash
   git config --global --add safe.directory C:/Users/PratikJawahar/chlu/chlu-bundle-v2
   ```
   ⭐⭐ **This matters enormously: the 2 GB frozen instrument never has to be copied.** Read it in place.
2. ⚠ **The remote default shell is PowerShell 5.1, not `cmd`.** Irrelevant if you follow §0, but it
   means inline `ssh host "..."` fails in *different* ways than the runbook describes (`&` is a
   reserved operator; `\"` nesting dies). ⭐ Also stale: the bundle README's **"JAX has no Windows
   wheel, use WSL2"** — JAX runs natively, and WSL is not even installed (`wsl -l -q` is empty).

## 2. Where everything lives
| what | path | writable by `sshkey`? |
|---|---|---|
| ⭐ frozen instrument (2 GB cache, decoder-lift winner, store grouping) | `/c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/**` | read-only — **leave it there** |
| the wave-2 venv's site-packages (jax 0.9.0, numpy 2.4.1, eqx 0.13.4) | `…/chlu-bundle-v2/.venv/Lib/site-packages` | read-only |
| wave-N code workspace | `/c/chlu/workspace-iterN` | ✅ dual ACL |
| hub worktrees | `/c/chlu/worktrees/CHLU-hub-<target>` | ✅ |
| **the only place a hub writes** | `/c/chlu/outputs/hub-<target>/` | ✅ |
| data (h5 + montero + value_holdout) | `/c/chlu/data/gen/shapes3d` | ✅ |
| task files, launch prompts, runners | `/c/chlu/tasks/` | ✅ |
| scored-run ledger | `/c/chlu/SCORED-LEDGER.md` | ✅ |

⛔ **Never write into the Head's profile.** Read from it; write to `/c/chlu`.

## 3. ⭐ The environment file — `/c/chlu/env_w3.sh`
Sourced from **inside** a worktree, so `CHLU_ROOT` is that worktree.

```bash
for v in $(env | grep -o '^CHLU_[A-Z_]*'); do unset "$v"; done   # a stale CHLU_* silently overrides
export CHLU_ROOT="$PWD"
_H=/c/Users/PratikJawahar/chlu/chlu-bundle-v2
export CHLU_CLAUDE_ROOT="$_H/.claude"
export CHLU_GEN_BRIDGE_ROOT="$_H/.claude/outputs/gen-feature-bridge/runs"
export CHLU_GEN_WIDE_BRIDGE_ROOT="$_H/.claude/outputs/gen-bridge-wide/runs"
export CHLU_GEN_STORE_RUNS="$_H/.claude/outputs/gen-store-grouping/runs"
export CHLU_GEN_DECLIFT_DIR="$_H/.claude/outputs/gen-decoder-lift/winner"
export CHLU_GEN_DATA_ROOT=/c/chlu/data/gen     # WRITABLE
export CHLU_HUB_OUTPUTS=/c/chlu/outputs        # WRITABLE — the only place a hub writes
export PYTHONPATH="$CHLU_ROOT:$_H/.venv/Lib/site-packages"   # ⛔ ORDER IS LOAD-BEARING — see §4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset _H
```

⭐⭐ **Do NOT build a venv and do NOT `uv sync`.** The system python (3.11.9, sshkey's own) imports the
wave-2 site-packages directly and gets **jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4 — wave 2's exact
versions.** No install, no network, ⭐ **and no version drift against the numbers you are comparing to.**
⚠ The box has **16 cores**; 4 hubs × 4 threads saturates it exactly.

## 4. ⛔⛔ THE IMPORT TRAP — verify it, never reason about it
`…/.venv/Lib/site-packages/_editable_impl_chlu.pth` contains
`C:\Users\PratikJawahar\chlu\chlu-bundle-v2`. **If `chlu` resolves there, every hub silently runs the
Head's checkout instead of its own worktree and every number is void.**
`$CHLU_ROOT` first in `PYTHONPATH` prevents it. **Prove it every time:**

```python
import os, pathlib, chlu
cf = pathlib.Path(chlu.__file__).resolve()
assert str(cf).lower().startswith(str(pathlib.Path(os.environ["CHLU_ROOT"]).resolve()).lower()), cf
```

## 5. Shipping code and data
```bash
# Mac: bundle the branch WHOLE (⛔ never `main..branch` — a thin bundle dies on arrival). ~4 MB.
git bundle create /tmp/base-iterN.bundle base/iterN
git bundle verify /tmp/base-iterN.bundle          # "records a complete history"

# stage payload: bundle + tasks/*.md + the new data artefacts, then
rsync -az --rsync-path='C:\msys64\usr\bin\rsync.exe' /tmp/ship/ forgis-14:/c/chlu/ship/
```
⛔⛔ **Without `--rsync-path` rsync transfers NOTHING and exits 0.** ⭐ **Exit 0 proves nothing —
always verify by file count AND checksum:**
```bash
find /c/chlu/ship -type f | wc -l      # must equal what you staged
sha256sum <the artefacts>              # must equal the Mac's
```

Then, on the box:
```bash
WS=/c/chlu/workspace-iterN
git init -q "$WS" && cd "$WS"
git fetch -q /c/chlu/ship/base-iterN.bundle 'refs/heads/base/iterN:refs/heads/base/iterN'
git checkout -q base/iterN
git diff --shortstat <PIN>..HEAD       # ⭐ assert additive: N insertions, 0 deletions
```

## 6. One worktree per hub — ⛔ never `git checkout -b` in a shared dir
```bash
cd /c/chlu/workspace-iterN
for t in <targets>; do
  git worktree add -q /c/chlu/worktrees/CHLU-wN-$t -b wN/$t base/iterN
  mkdir -p /c/chlu/outputs/wN-$t                  # ⭐ NAMESPACE BY WAVE — see below
  rm -f /c/chlu/outputs/wN-$t/START_UTC.txt       # a stale one makes the hub refuse to start
done
```
⛔⛔ **NAMESPACE EVERY PER-WAVE PATH BY THE WAVE NUMBER — worktree, branch, tmux session AND OUTPUT
DIRECTORY.** ⚠ **This bit us on 2026-09-19:** wave 4 reused `outputs/hub-unseen-valid/`, which wave 3's
hub of the same target already owned, and the new hub began **appending to the old hub's `LOG.md`**
(79 KB of wave-3 content) with wave 3's `PREREG` sitting there to be overwritten. ⭐ The wave-3 record
survived only because it had already been pulled to the Mac. **`wN-<target>` everywhere costs nothing
and makes the collision impossible.**
⚠ `.claude/` must NOT exist in a worktree — a hub writing "notes" there creates a second one nobody
reads. Hubs write to `$CHLU_HUB_OUTPUTS/hub-<target>/`.

## 7. ⛔⛔ THE TWO INTERACTIVE PROMPTS THAT SILENTLY STALL EVERY HUB
**A freshly launched `claude` blocks on two dialogs. Neither produces an error; the hub just sits
there and you think it is thinking.** ⭐ Check with `capture-pane`, not by waiting.

**(a) "Is this a project you trust?"** — kill it in advance by pre-seeding the config:
```python
import json, os, shutil
p = os.path.expanduser('~/.claude.json'); shutil.copy(p, p + '.bak')
d = json.load(open(p)); pr = d.setdefault('projects', {})
tmpl = {"allowedTools": [], "history": [], "mcpContextUris": [], "mcpServers": {},
        "enabledMcpjsonServers": [], "disabledMcpjsonServers": [], "hasTrustDialogAccepted": True,
        "projectOnboardingSeenCount": 1, "hasClaudeMdExternalIncludesApproved": False,
        "hasClaudeMdExternalIncludesWarningShown": False}
for t in [<targets>]:
    for k in (f"C:\\chlu\\worktrees\\CHLU-hub-{t}", f"C:/chlu/worktrees/CHLU-hub-{t}"):
        pr.setdefault(k, dict(tmpl))["hasTrustDialogAccepted"] = True   # ⭐ BOTH slash styles
json.dump(d, open(p, 'w'), indent=2)
```
**(b) "You accept all responsibility … Bypass Permissions mode"** — ⛔ not pre-seedable; answer it:
```bash
tmux send-keys -t hub-$t Down ; sleep 1 ; tmux send-keys -t hub-$t Enter
```
⚠ The default selection is **"No, exit"** — ⛔ a bare `Enter` kills the hub. **`Down` first.**

## 8. ⭐ Launch — a runner per hub, a tmux SESSION per hub
A session (not a window) per hub is what lets the Head attach to one without touching the others.

```bash
# /c/chlu/tasks/run-<t>.sh
cd /c/chlu/worktrees/CHLU-hub-<t>
source /c/chlu/env_w3.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /c/chlu/tasks/launch-<t>.txt)"
```
⭐ **Put the prompt in a file and `$(cat …)` it.** The prompt is ~2.7 KB with punctuation that does
not survive `tmux send-keys` quoting.

```bash
for t in <targets>; do
  tmux has-session -t hub-$t 2>/dev/null && { echo "REFUSING hub-$t: exists"; continue; }
  tmux new-session -d -s hub-$t -c /c/chlu/worktrees/CHLU-hub-$t
  tmux send-keys -t hub-$t "bash /c/chlu/tasks/run-$t.sh" Enter
done
```
⛔ **The `has-session` guard is the fix for wave 2's two-agents-on-one-hub incident. Keep it.**

## 9. Watching, without disturbing anything
```bash
tmux ls
tmux capture-pane -p -t hub-<t> -S -25        # liveness, no attach needed
tail -40 /c/chlu/outputs/hub-<t>/LOG.md       # ⭐ the real record
ls /c/chlu/outputs/hub-<t>/ | grep -i prereg  # ⭐ a new PREREG = a scored run is imminent
```
⭐ Healthy pane looks like `✶ Beaming… (1m 23s · ↓ 8.6k tokens · thinking)`.
⚠ An empty `>` prompt with no spinner means the prompt was never submitted — resend it.

## 9b. ⭐⭐ WAKING UP — the Hub has NO timer of its own. Arm one.
⛔⛔ **A Hub that says "I will check in 2 hours" will not.** There is no autonomous clock; without a
mechanism the Hub only wakes when the Head types. ⚠ **This cost wave 4 and wave 5 real idle hours.**

⭐ **The mechanism: a background Bash `until`-loop that EXITS when something happens.** A backgrounded
command keeps running across turns and **re-invokes the Hub when it exits** — that is the wake-up.
⛔ `Monitor` is the wrong tool here: it caps at 30 min and is built for *streams* of events, not one wake.

**The detector (runs on the box).** ⛔⛔ **THREE states, not two — "not working" does NOT mean
"finished".** A hub can be **BLOCKED** on a confirmation dialog:
⚠⚠ **`--permission-mode bypassPermissions` does NOT cover the dangerous-`rm` guard.** A hub that tries
to `rm -rf` a path the client judges critical stalls on *"Do you want to proceed?"* indefinitely, and to
a two-state detector it looks exactly like a finished hub. **This happened in wave 5 within 8 minutes
of launch.** ⭐ Detect BLOCK separately and wake on it — a blocked hub needs the Hub *now*.
```bash
for t in <targets>; do
  tmux has-session -t w5-$t 2>/dev/null || { echo "GONE $t"; continue; }
  tmux capture-pane -p -t w5-$t -S -6 | grep -q 'esc to interrupt' && echo "WORK $t" || echo "IDLE $t"
done
```

**The watcher (runs on the Mac, backgrounded).**
```bash
START=$(date +%s); MAX=7200; FAILS=0
while :; do
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=20 forgis-14 'C:\msys64\usr\bin\bash.exe -l' < /tmp/detect.sh 2>/dev/null)
  if [ -z "$OUT" ]; then FAILS=$((FAILS+1)); [ $FAILS -ge 5 ] && { echo "WAKE reason=UNREACHABLE"; exit 0; }
  else FAILS=0
    echo "$OUT" | grep -q '^IDLE ' && { echo "WAKE reason=HUB-FINISHED"; echo "$OUT"; exit 0; }
    echo "$OUT" | grep -q '^GONE ' && { echo "WAKE reason=SESSION-GONE"; echo "$OUT"; exit 0; }
  fi
  [ $(( $(date +%s) - START )) -ge $MAX ] && { echo "WAKE reason=HEARTBEAT-2H"; exit 0; }
  sleep 180
done
```
⭐⭐ **FOUR states, and the middle two are why a naive detector lies.** Measured on live hubs:
| state | pane signature | wake? |
|---|---|---|
| **BLOCK** | `Do you want to proceed?` / `Esc to cancel · Tab to amend` | ⭐ **YES, fastest** |
| **WORK** | `esc to interrupt` | no |
| **BUSYBG** | footer shows `N shells` / `N monitors` — between turns, background work pending | no, it self-resumes |
| **IDLE** | none of the above | ⭐ yes — finished, or stranded |

⭐ **Text sitting in a hub's input box is a QUEUED MESSAGE, not stuck input** (`Press up to edit queued
messages`). It is consumed at the next turn boundary — so it strands only when the agent never takes
another turn. ⛔ Sending `Enter` does NOT flush it; type a fresh message instead, which queues behind.

### ⭐⭐ Unsticking a stranded agent — and why they strand
⭐ **To unstick: TYPE A FRESH MESSAGE, then Enter.** That starts a new turn and the queued message is
consumed with it. ⛔ **Bare `Enter` does nothing** — verified on stranded hubs in waves 4 and 5.

⚠ **The commonest cause of stranding: `nohup`'d detached jobs.** A hub launches background compute,
ends its turn to wait, and **nothing ever re-invokes it** — detached output does not wake the session,
and the work finishes into silence. ⚠ Such a hub shows **no** `shells`/`monitors` footer, so it reads
**IDLE** even though its compute is alive. ⭐ **Check `PY <n>` and the LOG mtime before believing IDLE.**
⭐ **Prevention, put it in the launch prompt:** *"poll your background jobs to completion INSIDE a turn
(loop on their output files with sleep); do not end a turn while detached work is running."*

⛔⛔ **SILENCE IS NOT SUCCESS — the exit condition must cover every terminal state**, not just
"a hub finished": a vanished session, an unreachable box, and a plain time-based heartbeat. ⚠ A watcher
that only fires on the happy path sleeps straight through a crash and looks identical to "still running."
⚠ **Drop finished agents from the detector as you harvest them.** A finished agent is permanently
IDLE, so leaving it in the list makes the watcher fire on it every poll and starves the real signals.
⭐ **Re-arm it after every wake.** One wake = one notification = one armed watcher consumed.

## 9c. ⭐⭐ CLOUD / GPU LAUNCHES — the failure, and the protocol that follows from it
**Learned 2026-09-21, on the programme's first real GPU spend. Cost of the lesson: $0.57.**

### What happened
The w8 `encoder-loop` job died twice within 12 s of reaching the payload:
`FileNotFoundError: /var/chlu/artefacts/data/gen/shapes3d/3dshapes.h5 not found`, preceded by
`[dataset] ⚠ no --dataset in the experiment args; nothing fetched`. The artefacts bucket held a
**72 KB JSON** where a **267 MB dataset** was required.

⛔⛔ **THE ROOT CAUSE IS A STUB.** The hub validated its launcher end-to-end **against a labelled
stub** and reported `launch.sh accepts this spec, preflight passes`. ⭐ **A stub validates plumbing,
never inputs.** It cannot know an artefact is absent, because it never reads one.
⚠ **And the Hub skipped a precondition the spec itself listed** — `stage_gcs.sh --dry-run`, which
exists precisely to fail loudly on an empty artefact set (*"an empty stage is a silent no-op that
looks like a success on the far side"*).

### ⭐ Why it was cheap, and the thing to preserve
`run_until_done.sh` stopped after **two consecutive payload failures with no pre-emption**:
> *"That is a defect, not a lost VM, and a relaunch reproduces it."*

⭐⭐ **That rule is the difference between $0.57 and $12.** ⛔ Never raise `--max-attempts` to push a
job past a repeating payload failure — the supervisor is right and the payload is wrong.

### ⛔ THE PRE-LAUNCH CHECKLIST — run ALL of it, not the parts that look relevant
1. ⭐ **Run every precondition the spec lists, including the boring one.** If a spec names
   `stage_gcs.sh --dry-run`, run it. ⛔ The Hub's judgement about which checks matter is exactly
   what the checklist exists to replace.
2. **Enumerate the artefacts the payload READS and verify each is present and non-trivial in size.**
   For a gen job that is: the external classifier · the decoder-lift winner · ⭐ **the dataset** ·
   the feature cache if the job reads one. ⚠ A directory that exists but holds one small file is
   the failure mode — **check sizes, not existence.**
3. **Verify the module accepts every flag the launcher will forward.** `startup.sh` extracts
   `--dataset` from `--exp-args` **and still passes the whole string through verbatim**, so adding a
   flag the experiment's `argparse` does not define fails a *second* way. ⛔ Grep the module's
   `add_argument` list before adding anything to `--exp-args`.
4. **Launch ONE unit first.** Concurrency here is one anyway (`A2_CPUS` is exactly full at 12, so the
   GPU metric never binds), so sequencing costs nothing and buys a real end-to-end check.

### ⭐ THE FIX PROTOCOL, in order
1. ⭐⭐ **Read `gs://<bucket>/jobs/<job>/out/job.log` first.** It names the exact missing path and the
   exact traceback. ⛔ Do not theorise before reading it.
2. **Classify the failure.** The supervisor already separates **pre-emption** (`out/PREEMPTED`, relaunch
   is correct) from **payload failure** (`out/FAILED`, a relaunch reproduces it). Trust that split.
3. **For a missing input, decide which mechanism owns it** — is the VM *supposed* to fetch it
   (a `--dataset`-style path) or is it *supposed* to be staged? ⚠ `3dshapes.h5` is deliberately
   **not staged** by default: it comes from a public bucket whose owner pays. But that path only
   fires if the module accepts the flag — and here it did not.
4. ⭐⭐ **Prefer a server-side copy to a laptop round-trip.** The public source was itself GCS, so:
   ```bash
   gcloud storage cp gs://3d-shapes/3dshapes.h5 \
       gs://<bucket>/artefacts/data/gen/shapes3d/3dshapes.h5   # 61.7 MiB/s, no local transfer
   ```
   ⭐ Then verify: size against a known-good copy, and the sha256 against the runbook's pinned value.
5. **Re-run the spec's preconditions, then relaunch the single unit.** ⛔ Do not relaunch the full
   sweep until one unit has produced a `result.json`.
6. ⭐ **A staged input is a one-time fix, not a patch** — record it, because every later job benefits.

### Cost discipline
⭐ Report a cost line every time (`scripts/cloud/cost.sh`); a GPU run with no cost line in the wave
report is not finished. The watchdog (`--max-hours`) should bind **before** the supervisor's
`--max-usd`, so the cheaper limit is the one that fires.

## 9d. ⭐⭐ BATMAC — THE SECOND BOX. Everything here was learned by getting it wrong (2026-09-23)
`ssh batmac@batmac` over Tailscale. **Apple M4, 10 cores, 16 GB, ~389 GB free.** ⛔ **Apple Metal — it NEVER uses GCloud; only `sshkey`/forgis-14 does.**

### ⛔⛔ THE ONE THING THAT WILL BREAK EVERYTHING: THE KEYCHAIN
Claude Code's OAuth token lives in the **macOS login Keychain**, and **an SSH session cannot unlock it**. A spoke launched from a plain SSH-started tmux prints
`Not logged in · Please run /login · Run in another terminal: security unlock-keychain` and sits there looking like it is thinking.
- ⭐ **The fix, and the ONLY one that worked:** on batmac's **physical console or via Screen Sharing** (⛔ **not over SSH**), run
  `tmux kill-server 2>/dev/null; tmux new-session -d -s hubhost`
  That starts the tmux **server** inside the unlocked GUI security session. Every session created inside it afterwards — including ones you create over SSH — inherits keychain access.
- ⛔⛔ **NEVER run `tmux kill-server` on batmac again.** That server IS the keychain access. Killing it un-authenticates every agent on the machine. Put this line in **every** batmac launch prompt.
- ⛔ `launchctl asuser 501 …` does **NOT** work around it — it needs root (`Could not switch to audit session: Operation not permitted`).
- ⛔ Starting tmux **over SSH** does not work either, even after the GUI login: the server inherits the SSH audit session. ⚠ We lost time to this — a `hubhost` started from an SSH terminal looked identical and still failed.
- ⭐ **Verify, never assume:** create a session inside the server and run `security show-keychain-info ~/Library/Keychains/login.keychain-db`. **`RC=0` + `no-timeout` = good.**

### ⚠ `claude` is not on PATH in a non-interactive SSH shell
`~/.local/bin` is added by **`.zshrc`**, which a non-interactive/login `zsh -l` does **not** source. So `command -v claude` returns nothing and the tool looks missing when it is installed and logged in.
⭐ Export it in the runner: `export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"`.
⚠ `~/.zshrc` also contains a stale `export PATH="$PATH:/Users/user/.local/bin"` — the **Mac's** user, not batmac's. Harmless, but do not trust that line.

### Where everything lives on batmac
| what | path |
|---|---|
| repo (worktree source) | `~/clu/workspace` — keep at the base the wave scored on |
| per-spoke worktrees | `~/clu/worktrees/CHLU-<wave>-<slug>` |
| **the only place a spoke writes** | `~/clu/outputs/<wave>-<slug>/` |
| shared draft dir | `~/clu/outputs/w17-paper/` (both `.tex`, PDFs, `references.bib`, `figures/`, `iclr2027/`) |
| evidence base (read-only) | `~/clu/evidence/outputs/` |
| staged binary artefacts | `~/clu/stage/outputs/` (decoder-lift winner, external classifiers, feature cache) |
| dataset | `~/clu/data/gen/shapes3d/` — sha256 **0a0f6ed9…**, matches the runbook pin |
| env contract | `~/clu/env_bm.sh` — exports `CHLU_GEN_DATA_ROOT`, `CHLU_GEN_DECLIFT_DIR`, `CHLU_GEN_FEATURES_ROOT`, `CHLU_GEN_CLASSIFIER_ROOT`, and puts the venv first on PATH |
| tasks + launch prompts | `~/clu/tasks/<wave>/` |

⭐ **venv pins match forgis-14 exactly** — jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4 — so numbers are comparable across machines. `uv sync` in `~/clu/workspace` reproduces it.
⚠ **The feature cache is `seed_0` ONLY** (1.97 GB). A spoke needing seeds 1–2 must write `NEEDS-ARTEFACTS.json`; ⛔ never substitute.

### ⭐⭐ WHAT BELONGS HERE vs ON forgis-14
- ⛔ **Scored runs stay on forgis-14.** It owns `/c/chlu/SCORED-LEDGER.md` and ⛔ **the ledger must never be duplicated.**
- ⭐ **Anything that must COMPILE LaTeX belongs on batmac.** **TeX Live 2026 is installed here; ⛔ forgis-14 has NO LaTeX at all.** This is not a preference — a `cite-tagger`-class spoke simply cannot finish on the box.
- ⭐ Everything zero-scored — drafting, figures, tables, registries, referees — and Shapes/maze compute that fits `seed_0` or generates its own data.
- ⛔ **Do not migrate a RUNNING spoke to rebalance.** Restarting costs its accumulated hours; place the *next* one instead.
- ⚠ **Measure before rebalancing.** A single `Win32_Processor LoadPercentage` sample is worthless: forgis-14 read **0 / 30 / 0 / 51 / 1** across five samples 4 s apart. Take ≥ 5.

### Detector differences from the box — these bite silently
- ⭐ **`stat -f %m`** on batmac (BSD) vs **`stat -c %Y`** on msys. A detector copied between machines reports `99999` for every log and reads everything as WORK.
- ⭐ **`rsync` needs NO `--rsync-path` here.** That flag is a forgis-14 (msys) requirement only.
- ⛔⛔ **Session-prefix regexes rot.** A reaper written as `grep -E '^w1[789]-'` **silently skipped every `w20-` session**, so finished spokes stayed alive and blocked the next launch's concurrency guard. ⭐ Use `^w[12][0-9]-`, and re-read the regex whenever the wave number rolls over.
- ⛔ A detector loop written for one wave (`S=w19-$t`) will watch **nothing** when reused for another wave's spokes (`w18-…`). ⭐ Address sessions by **full name**, not prefix + slug.

### ⭐⭐ THE `.tex` LOCK — batmac's own concurrency rule
Multiple spokes want `DRAFT-v2.tex` and `DRAFT-v2-style.tex`. **Two agents on one file corrupts it.**
⭐ **Serialise them, and make each launch script REFUSE if a peer session is live:**
```bash
for g in w18-placeholder-fill w19-draft-figure-slots w20-referee3-fixes w20-citation-prune; do
  tmux has-session -t $g 2>/dev/null && { echo "ABORT: $g is live"; exit 1; }
done
```
⚠ **That guard is only as good as your reaping** — a finished-but-unreaped session blocks the next launch as a false positive. ⭐ **Reap on harvest**, not later.
⭐ **Order matters, not just exclusion:** `citation-prune` must run **after** `referee3-fixes`, because the positioning rewrite changes which citations are needed. Pruning first prunes against text that is about to change.

### Launch on batmac — the differences from §8
```bash
cat > ~/clu/tasks/<wave>/run-<slug>.sh <<'EOF'
cd /Users/batmac/clu/worktrees/CHLU-<wave>-<slug>
source /Users/batmac/clu/env_bm.sh
exec claude --permission-mode bypassPermissions --model opus "$(cat /Users/batmac/clu/tasks/<wave>/launch-<slug>.txt)"
EOF
tmux new-session -d -s <wave>-<slug> -c ~/clu/worktrees/CHLU-<wave>-<slug>   # joins the hubhost server
tmux send-keys -t <wave>-<slug> "bash ~/clu/tasks/<wave>/run-<slug>.sh" Enter
```
⭐ Trust dialog: pre-seed `~/.claude.json` exactly as §7a, with the **batmac** worktree paths.
⭐ Bypass dialog: still **`Down` then `Enter`** — ⛔ the default is "No, exit".

### ⛔ Put these lines in EVERY batmac launch prompt
1. *"Never run `tmux kill-server` on this machine — that server holds this machine's keychain access and killing it breaks authentication for every agent here."*
2. *"Avoid `rm <glob>` — the dangerous-`rm` guard is NOT covered by `bypassPermissions` and will stall you indefinitely; write into a fresh timestamped subdir instead."*
3. The `.tex` lock statement, if the spoke touches either draft.
4. ⚠ *"The feature cache here is seed_0 only; if you need seeds 1–2 write `NEEDS-ARTEFACTS.json` rather than substituting."*

---

## 9e. ⭐⭐ WATCHER DISCIPLINE — the rules that cost the most to learn
- ⛔⛔ **ARM THE WATCHER ALONE.** ⚠ Chaining it (`… && bash /tmp/rearm.sh`) makes the background task ambiguous: its output holds the chained commands' text, so you cannot tell "armed" from "fired", and one chained arm exited immediately without anyone noticing.
- ⛔⛔ **RE-ARM ATOMICALLY.** Arming without killing the previous one leaves **two** watchers that both fire on the same event and each consume a wake. Use a script that kills, verifies, then `exec`s:
```bash
pkill -x -f 'bash /tmp/watch3.sh'    # -x = EXACT full command line
sleep 2
[ "$(ps -axo args | grep -c '^bash /tmp/watch3.sh$')" -ne 0 ] && { echo "REFUSING"; exit 1; }
exec bash /tmp/watch3.sh
```
- ⛔⛔ **`pkill -f '/tmp/watch3.sh'` KILLS YOUR OWN SHELL.** Your command line *contains* that string, so `-f` matches the caller. ⭐ **`pkill -x -f 'bash /tmp/watch3.sh'`.** Same trap makes `ps | grep -c 'watch3'` count itself — ⭐ match `^bash /tmp/watch3.sh$` exactly.
- ⛔ **A watcher NEVER survives a client restart usefully.** The process may live, but the wake notification goes to the thread that armed it. ⭐ After any Zed/CLI restart: **kill it and arm a fresh one.** *(Remote tmux work is unaffected — that is the whole reason spokes run in tmux.)*
- ⛔⛔ **PRUNE ON HARVEST — in BOTH places.** Remove the spoke from the **detector** *and* **kill its tmux session**. Leaving it in the detector re-fires the watcher every poll; leaving the session alive blocks the next launch's concurrency guard and holds ~150 MB on a 16 GB box. ⚠ **14 dead sessions** accumulated on batmac and **10** on forgis-14 before this was noticed.
- ⛔⛔ **A GATE ON `[ -f X ]` FIRES EARLY.** An agent's own `END_UTC.txt` is **not** proof it finished — twice a gate fired while the spoke still read ACTIVE. ⭐ **Triple check: `END_UTC` exists AND the pane is not working AND the log has been quiet ≥ 10 min.** For a JSON gate, require it to **parse** and carry the right status, not merely exist.

---

## 9f. ⭐ TALKING TO A RUNNING HUB
⭐ **Short pointer + durable file.** Write the instruction as `AMENDMENT-N.md` **into the hub's own output dir**, then `tmux send-keys` **one short line** pointing at it.
⛔ **Never send a long message.** If the composer already holds queued text a long one appends, the composer goes multi-line, and **`Enter` inserts a newline instead of submitting** — the agent stays idle and the queue is now worse.
⭐ **Read the pane first** and refuse to send if it shows `Press up to edit queued messages` / `ctrl+x ctrl+s to send now`.
⭐ Queued text is consumed at the next turn boundary, so it is safe **while the agent is working** (`esc to interrupt`). It only strands if the agent never takes another turn.
⭐ **The file is the real channel** — if a queued message is ever lost, the instruction still exists where the hub will look.

---

## 10. Collecting — ⛔ a hub's return needs BOTH halves
| mechanism | carries | misses |
|---|---|---|
| `git bundle create <f> hub/<t>` | tracked code | **everything in `.claude/`** |
| the outputs tarball / rsync | reports, LOG.md, PREREGs | **all code** |
⛔⛔ **`git status` does NOT list gitignored files** — a worktree holding an entire unsaved outputs
tree reports clean. ⭐ **Verify with a directory listing, never with git.**
```bash
rsync -az --rsync-path='C:\msys64\usr\bin\rsync.exe' \
  --include='*/' --include='*.md' --include='*.json' --include='*.py' --include='*.sh' \
  --include='*.log' --include='*.csv' --include='*.bundle' --exclude='*' \
  forgis-14:/c/chlu/outputs/ .claude/outputs/_thinkpad/outputs/
```
⭐ Filter, or you pull ~9.9 GB of `.npz` dev caches. **Check the file count afterwards.**

## 11a. ⭐ Pre-launch checklist — BATMAC
- [ ] ⛔⛔ `hubhost` tmux server is **alive** and was started from the **GUI console**, not SSH (`tmux ls` shows it; a session inside it gets `RC=0` from `security show-keychain-info`)
- [ ] `export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"` in the runner — else `claude` is "missing"
- [ ] `source ~/clu/env_bm.sh`, then assert `chlu.__file__` resolves **inside the worktree**
- [ ] worktree branched from **the base that wave actually scored on**, not reflexively `base/iter8`
- [ ] trust pre-seeded in `~/.claude.json` for the **batmac** paths, both slash styles
- [ ] `.tex` lock: no peer session live, and the launch script **refuses** if one is
- [ ] finished sessions **reaped** (`^w[12][0-9]-` + `END_UTC` + not working) — a stale one blocks the guard
- [ ] launch prompt carries: never `tmux kill-server`; avoid `rm <glob>`; seed_0-only cache; the `.tex` lock
- [ ] after launch: past the bypass dialog (**`Down` then `Enter`**) and `START_UTC.txt` + `LOG.md` exist

## 11. Pre-launch checklist — FORGIS-14, all of it, every time
- [ ] `git config --global --add safe.directory …` applied (§1)
- [ ] payload verified by **file count and checksum**, not by rsync's exit code (§5)
- [ ] workspace diff vs the pin is **additive, 0 deletions** (§5)
- [ ] `chlu.__file__` resolves **inside the worktree** (§4)
- [ ] feature cache opens: `(480000, 128, 4, 4) float16`, 1,966,080,128 bytes (§3)
- [ ] dataset sha256 `0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785`
- [ ] one worktree per hub, no stale `START_UTC.txt` (§6)
- [ ] trust pre-seeded, **both slash styles** (§7a)
- [ ] `has-session` guard in the launch loop (§8)
- [ ] after launch: every pane past **both** dialogs and showing a spinner (§7, §9)
- [ ] after ~2 min: `START_UTC.txt` and `LOG.md` exist for every hub (§9)
