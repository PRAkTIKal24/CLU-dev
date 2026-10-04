# HUB OPERATING POLICY — how this Hub runs the ICLR programme
**Written 2026-09-20 after waves 3–8. ⭐⭐ A fresh Hub reads THIS, then
`AUTOHUB-SSH-RECIPE.md` for command-level detail. Everything here was learned by getting it wrong.**

---

## 1. The loop, in one paragraph
The Head and the Advisor pre-scope waves. **The Hub executes them back to back, self-authorised**, and
does not wait for approval between waves. Per wave: write the task files → build a `base/iterN` →
ship to the ThinkPad → launch one agent per target in its own tmux session → **arm a watcher** →
on each wake, triage · summarise finished agents · fire retries where permitted → at the wave
boundary, harvest, compile `w<N>-advisor-report.md`, and **launch the next wave immediately**.
⛔ The only cause to hold a launch is the box being unreachable — or §4's premise check.

## 2. ⛔⛔ THE HUB HAS NO CLOCK. ARM A WATCHER OR NOTHING HAPPENS.
**A Hub that says "I will check in two hours" will not.** There is no autonomous timer; without a
mechanism the Hub wakes only when the Head types. ⚠ **This cost waves 4 and 5 real idle hours.**

⭐ **The mechanism: a backgrounded Bash `until`-loop that EXITS when something happens** — a
backgrounded command keeps running across turns and **re-invokes the Hub when it exits**.
⛔ `Monitor` is the wrong tool: it caps at 30 minutes and is built for event *streams*, not one wake.
⭐ **Re-arm after every wake.** One wake consumes one watcher.
⛔⛔ **Wake on every terminal state, not just the happy one** — finished, **blocked**, session gone,
box unreachable, and a 2 h heartbeat. A watcher that only fires on "a hub finished" sleeps through a
crash, and silence looks identical to "still running".
⚠ **Prune finished agents from the detector as you harvest them** — a finished agent is permanently
IDLE and will re-fire the watcher every poll, starving the real signals.

## 3. ⭐⭐ THE FIVE AGENT STATES — "not working" NEVER means "finished"
Measured on live hubs. A two-state detector lies, and lied to this Hub four separate times.

| state | pane signature | wake? | what it means |
|---|---|---|---|
| **BLOCK** | `Do you want to proceed?` / `Esc to cancel · Tab to amend` | ⭐ **yes, fastest** | stalled on a dialog |
| **WORK** | `esc to interrupt` | no | running this turn |
| **SUBAG** | `to see subagents` · `( ) general-purpose` · `● main` | no | subagents running |
| **BUSYBG** | footer shows `N shells` / `N monitors` | no | between turns, self-resumes |
| **IDLE** | none of the above | ⭐ yes | finished, **or stranded** |

⛔⛔ **`--permission-mode bypassPermissions` does NOT cover the dangerous-`rm` guard.** A hub that
tries to `rm -rf` a path the client judges critical stalls forever and looks exactly like a finished
hub. **This fired within 8 minutes of a wave-5 launch.**
⚠ **Before believing IDLE, check the python count and the LOG mtime.**

⭐⭐ **A SIXTH CHECK, AND IT IS THE ONE THAT STOPS THE CHURN: a LOG-RECENCY GUARD.**
Pane state alone is not enough. A long-running hub sits between turns for seconds at a time and reads
**IDLE** while its compute is alive — in wave 8 one hub did this three times in twenty minutes and woke
the watcher each time. ⛔ **Every false wake costs a watcher and starves the real signals.**
⭐ **Rule: if the agent's `LOG.md` was written in the last 10 minutes, it is ACTIVE regardless of pane
state.** Only an agent that is pane-idle **and** log-quiet is genuinely finished or stranded.
```bash
AGE=$(( $(date +%s) - $(stat -c %Y "$OD/LOG.md") ))
[ "$AGE" -lt 600 ] && echo "ACTIVE $t (log ${AGE}s)" || echo "IDLE $t (log ${AGE}s)"
```
⚠ Print the age in the state line — on a real wake it tells you at a glance whether the agent stopped
one minute ago or two hours ago.

### Stranding, and how to unstick
⭐ Text in an agent's input box is a **queued message**, not stuck input (`Press up to edit queued
messages`). It is consumed at the next turn boundary — so **it strands only when the agent never takes
another turn.** ⛔ **Bare `Enter` does not flush it. Type a FRESH message, then Enter** — that starts a
turn and the queue goes with it.
⛔⛔ **AND THE TRAP THAT BEAT THE FIX: a LONG message APPENDS to an existing queued one.** If the
composer already holds queued text, typing more makes it multi-line and **`Enter` inserts a newline
instead of submitting** — the agent stays idle and you have made the queue worse. ⚠ This wasted two
wakes in wave 8 on agents that had **finished and were asking a question**.
⭐⭐ **The reliable unstick, in three keys:** `Escape` (clears the composer) → **one SHORT single-line
message** → `Enter`. ⛔ Never send a paragraph to a session that already shows queued text.
⭐ **Read the pane before answering** — a finished agent often ends with an explicit question
(*"say so and I'll do it inside the turn"*). Answer THAT, briefly; do not restate the brief.

⚠ **The commonest cause of stranding is `nohup`'d work.** An agent launches detached compute, ends its
turn to wait, and nothing ever re-invokes it. Such an agent shows no shells/monitors footer, so it
reads IDLE while its compute is alive. ⭐ **Prevention, put it in every launch prompt:** *"poll your
background jobs to completion INSIDE a turn; do not end a turn while detached work is running."*

## 4. ⭐⭐ THE PREMISE CHECK — the one gate on autonomous chaining
A pre-approved wave was scoped **before** the previous wave reported. **Wave 3 killed two premises;
wave 5 killed wave 7's entire instrument; wave 6's headline law turned out not to exist.**

Before launching wave N+1, check each agent's stated premise against wave N's findings:
- ⭐ **premise survives ⇒ launch it unchanged, immediately. No approval, no delay.**
- ⛔ **premise contradicted BY A MEASUREMENT ⇒ do not launch that one.** Launch **every other agent in
  the wave**, and report the held one with the number that killed it and the smallest amendment that
  would revive it. ⛔ **The Hub does not redesign it** — that is the Head's and the Advisor's.

⭐ **This is not licence to hesitate. The default is launch.** Holding requires a measurement you can
name, never a reservation or a judgement call. ⚠ If the check ever holds more than one agent in a
wave, the waves are being scoped too far ahead of the evidence — tell the Head **immediately**, not at
the boundary.

## 5. The retry policy (auto-hub retry)
Full charter: `autonomous-hub-retry-charter.md`. The essentials:
- ⭐⭐ **Fire on TIME REMAINING, ⛔ never on the verdict.** In wave 3 the *winning* hub had exhausted its
  target's budget while the hub that returned **0 of 3 runs intact** was a clean negative that exited
  at 13 % of its allowance. **Negatives free the most time and hand over the sharpest next question.**
- **Fire iff** the parent is genuinely finished **and ≥ 3 h remain measured from DISCOVERY**, not from
  the parent's exit. With 2 h polling that gives three live windows. ⭐ Below the floor, **bank it**.
- **Allowance = 10 h − elapsed at discovery.** ⛔ Not a fresh 10 h.
- ⛔⛔ **A winning parent has usually SPENT the budget a retry would need.** Three kinds, named in the
  charter: **R-A robustness** (the default after a win; 0 scored runs), **R-B mechanism** (0 scored
  runs), **R-C adjacent target** (inherits that target's balance).
- **One retry per parent, capped at `.1`.** Branch from the parent's tip only as a same-target
  continuation; ⛔ never from a peer's branch.

## 5b. ⭐⭐ THE NOVELTY VERDICT IS A FRAMING INPUT, NOT A GATE (Head, 2026-09-20)
⛔⛔ **A novelty result never stops an experiment.** It decides **how the paper is framed** — which
sentence may be claimed as the contribution — and nothing else. ⭐ **Even a zero-novelty verdict, the
worst case, does not change what we run: waves 8–10 execute as planned.**
⚠ The Hub over-read this once, treating "STOP that claim" as though it could halt downstream work.
It cannot. **Stop the CLAIM, never the WORK.**
⭐ Where it *does* bind: wave 10's framing spokes — the registry fold (which records the composition
result **with** its novelty verdict) and the draft skeleton. Carry the verdict into those two, and
into no others.

## 5c. ⭐⭐ CLOUD SPEND — AUTHORISED, AND BOUNDED (Head, 2026-09-20)
⭐ **Project `forgisprova` has a billing method and the Hub is CLEARED to spend on it** without
asking each time. ⛔ **The authorisation is bounded and the Hub does not stretch it:**
- ⭐ **IN SCOPE:** runs belonging to **waves 8–10**, and **obvious retries that are likely to win**.
- ⛔ **OUT OF SCOPE:** everything else. Exploratory GPU work, speculative sweeps, a third attempt at
  something that has already failed twice — ⛔ **ask first.**

**The mechanics, because they do not change:**
- ⚠ `gcloud` is authenticated on the **Mac** (`pratik.jawahar@forgis.com`), **absent from the
  ThinkPad**. A hub builds and verifies the job and **emits a launch spec**; ⭐ **the Hub executes the
  authenticated step from the Mac** and returns results to the box.
- ⭐ **Card:** A100-40GB **SPOT** (measured quota: limit 16, 0 in use) is the training path. ⛔ A100-40GB
  on-demand has **0 free**. L4 spot (8 free) is the cheap interruption-test card.
- ⭐⭐ **Checkpoint-first is mandatory** — spot VMs are taken away. Checkpoint to Cloud Storage on a
  wall-clock timer, verify a config fingerprint before resuming, sync outputs while running, and
  **prove resumability with `kill -9` before spending anything.** The VM deletes itself on every exit path.
- ⭐ **Report a cost line every time** (`scripts/cloud/cost.sh`). ⛔ A GPU run with no cost line in the
  wave report is not finished.
- ⛔ **Verify the pipeline end to end on CPU at small scale first.** A spec that has only been reasoned
  about is not ready to spend money.

## 5d. ⭐⭐ THE GYM IS NEVER KILLED (Head, 2026-09-20)
⛔⛔ **Wave 9's maze runs REGARDLESS, on local CPU.** There is no screening outcome that cancels it.
⭐ The Head has checked the maze design with the Advisor and it should be winnable.
- ⛔ **Instrument checks are reported columns, never abort conditions** — and never grounds for
  rejecting a candidate environment. Report the number; the Head weighs it.
- ⭐ **The env spoke's job is ADOPT-OR-SPEC, never adopt-or-abandon.** If nothing fits, it specs the
  build to be **built and run**.
- ⭐ **The only thing that can stop GPU spend is a hub concluding the maze is UNWINNABLE BY DESIGN**
  — and even then **the CPU experiment still runs.**
- ⭐ The two-gym law (a rule must render a route **illegal**, not merely slower; and the rules'
  interaction must be worth enough that a composition-blind policy cannot score well) is now a
  **design constraint to satisfy**, ⛔ not a gate that rejects.
⚠ The Hub previously recommended holding wave 7 on premise-death grounds. ⭐ **Under this ruling that
reasoning does not apply to the maze:** a design weakness is written as a **correction to the spec**,
never as a reason to stop.

## 6. ⛔ WHERE THE HUB STOPS
⭐ **Rescoping inside a target or a dial is scoping — the Hub does it.**
⛔ **Choosing a new target is research direction — the Hub does not.** If the obvious next move is a
new target, **bank the time, write the question down, report it.** ⛔ Do not pick.

⭐⭐ **WIDENED 2026-09-20 (Head): WHERE THERE IS RETRY BUDGET, THE HUB DECIDES AND DOES NOT WAIT.**
Design calls **inside a target** — what the stored object is, which of two measured remedies to take,
whether to hold or spend a run — are the Hub's **when budget exists to absorb being wrong.**
⚠ The precedent: wave 8's exactness was falsified on one seed of four, and the remedy (declare the
**subspace**, not the eigenvector basis, to be the object) changed what the store IS. ⭐ The Hub
recommended and the Head cleared it — and then instructed that such calls not come back.
**So: decide, log the decision and the reasoning in the wave report, and move.** ⛔ Still escalate only
a genuinely NEW TARGET, or a call with no budget left to absorb a wrong answer.

## 7. Scored-run discipline
**3 per TARGET, across every machine and attempt.** Replications of the same PREREG are **free**
(Advisor ruling, wave 4). ⭐ **The ledger row is written BEFORE the run** — a hub that cannot write
`/c/chlu/SCORED-LEDGER.md` cannot spend. All scored runs reported; ⛔ the best never alone.
⚠ **A hub can write its ledger row to the wrong path.** One did, into a stray `C:\c\chlu\` tree, and
the row would have been lost with the cleanup. ⭐ **Before approving any deletion, diff the stray
ledger against the real one and append what is missing.**

## 8. ⛔ THE MISTAKES REGISTER — every one of these actually happened
1. ⛔ **Output directories not namespaced by wave.** Wave 4 reused `outputs/hub-unseen-valid/` and a
   new hub began appending to wave 3's 79 KB `LOG.md`, with wave 3's PREREG there to be overwritten.
   ⭐ **Namespace worktree, branch, tmux session AND output dir: `wN-<target>` everywhere.**
2. ⛔ **A spoke finished with 12 files untracked and no bundle.** Its own queued `commit this` never
   fired; the Hub had to commit and bundle by hand at the boundary. ⭐ **Put "commit and bundle the
   branch WHOLE before you finish" in every launch prompt**, and verify: outputs by **directory
   listing** (⛔ `git status` cannot see gitignored outputs), code by `git status` + `git bundle verify`.
3. ⛔⛔ **A hub ran `taskkill /F /IM python.exe` twice**, killing every python process on a four-agent
   box — it destroyed another hub's training run and a third's deletion seeds. ⭐ **"Never run a
   box-wide kill; kill only your own PIDs, captured at launch" goes in every launch prompt.**
4. ⛔ **Bundles made with `--all` contain every branch at different times.** The same branch name has
   different hashes in different bundles. ⭐ **Fetch each hub's branch from ITS OWN bundle's HEAD.**
5. ⛔ **zsh ate `:r` in a refspec** (`$t:refs/...` → history modifier). ⭐ **Brace it: `${t}:refs/...`.**
6. ⛔ **`set -e` plus a no-op `tmux kill-session` aborts the whole setup script** before anything runs.
   ⭐ Guard cleanup with `|| true`.
7. ⛔ **A fixed spinner word-list gave a false negative on all five hubs at once.** ⭐ Match pane
   *content* per §3, never a keyword list.
8. ⛔ **The Hub's own retry hypothesis was false by construction** (settling cannot leave an invariant
   subspace). ⭐ Scope retries so the hypothesis can be **killed**, and expect it to be.
9. ⚠ **A long foreground SSH breaks** (`Read from remote host: Operation timed out`). Harmless —
   tmux survives — but ⭐ never hold an SSH open with long sleeps; poll in short calls.
10. ⚠ **`rsync` with multiple sources into one destination collides** — four `LOG.md` overwrote each
    other. ⭐ **One rsync per agent, into its own directory.**

## 9. The non-obvious mechanics (detail in `AUTOHUB-SSH-RECIPE.md`)
- ⭐ **Never inline-quote a remote command. Pipe a heredoc'd script into msys64 bash** — that bypasses
  the remote shell entirely, so you never care what it is (it is PowerShell, not `cmd`).
- ⛔⛔ **`rsync` needs `--rsync-path='C:\msys64\usr\bin\rsync.exe'` or it transfers NOTHING and exits 0.**
  ⭐ **Verify every transfer by file count AND checksum.** Exit 0 proves nothing.
- ⛔⛔ **The import trap:** the shared site-packages carries an editable `.pth` pointing at the Head's
  own checkout. ⭐ Put `$CHLU_ROOT` first in `PYTHONPATH` and **assert `chlu.__file__` resolves inside
  the worktree** every time.
- ⭐ **Do not build a venv.** The system python plus the existing site-packages gives wave-2's exact
  versions (jax 0.9.0 / numpy 2.4.1 / equinox 0.13.4) — no install, no network, **no version drift**.
- ⭐ `sshkey` **can** read the Head's profile (one `git config --global --add safe.directory`), so the
  2 GB frozen instrument is read in place and never copied.
- ⭐ **Pre-seed folder trust** in `~/.claude.json` (**both** slash styles) or every agent stalls on the
  trust dialog; the bypass-permissions dialog is answered with **`Down` then `Enter`** — ⛔ the default
  selection is "No, exit", so a bare `Enter` kills the agent.
- ⚠ **GPU/cloud:** `gcloud` is authenticated on the **Mac** (project `forgisprova`) and **absent from
  the ThinkPad**. A hub cannot launch, poll or fetch a cloud job. ⭐ **The hub emits a ready-to-run
  launch spec; the Hub executes the authenticated step from the Mac** and returns results to the box.

## 10. Boot sequence for a fresh Hub
1. `.claude/iclr-long-hub.md` — the role.
2. `.claude/iclr-hub-handover.md` — where things stand.
3. **this file** — how the machinery runs.
4. `.claude/AUTOHUB-SSH-RECIPE.md` — before touching the ThinkPad.
5. `.claude/autonomous-hub-retry-charter.md` — the retry type and the chaining model.
6. `.claude/autonomous-hub-charter.md` — only when scoping a hub.
7. The **latest** `w<N>-advisor-report.md` — the programme's state. ⛔ Not the older ones.
⛔ **Do not read the campaign ledgers or `handover_context.md`.** Every agent that booted on them
optimised for the paperwork instead of the paper.
