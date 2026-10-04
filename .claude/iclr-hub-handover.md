# ICLR LONG — EXPERIMENT HUB HANDOVER. Rewritten 2026-09-25 ~12:0xZ. ⛔ Lean by design.
**You are the EXPERIMENT Hub. ⛔ A separate PAPER Hub owns the draft and is RUNNING RIGHT NOW.**

## BOOT — in this order, those five and nothing else
1. `.claude/iclr-long-hub.md` — the role.
2. **this file** — live state and what you owe.
3. ⭐⭐ `.claude/HUB-OPERATING-POLICY.md` — how this Hub runs.
4. ⭐⭐ `.claude/AUTOHUB-SSH-RECIPE.md` — **before touching either box.** §9d batmac · §9e watchers · §9f talking to a running spoke · §11a.
5. ⭐ `.claude/outputs/w23-advisor-report.md` — the newest evidence, and it **corrects three claims in w22's**.
⛔ **Do NOT read at boot:** the campaign ledgers · `handover_context.md` · w9–w22 reports (pull one row when a decision needs it) · `.claude/paper-lead-hub.md` (⛔ that is the OTHER Hub's brief) · `iclr-hub-handover.PREV-265L.md` (superseded, kept only for history).

## ⭐⭐ WHAT IS LIVE RIGHT NOW — checked 2026-09-25 ~12:0xZ, not remembered
- **forgis-14: 0 sessions. IDLE.** No compute anywhere in the programme.
- **batmac:** `hubhost` + `authtest` (infrastructure — ⛔ never kill) **and `pw1-draft-figures`.**
  ⛔⛔ **`pw1-*` IS THE PAPER HUB'S. Not yours. Do not touch, harvest, reap, monitor or report on it.** ⭐ It holds the `.tex` lock.
- **Your watcher:** ⛔ **DISARMED** (W24 harvested 2026-09-26 ~02:30Z; Head's standing order: no watcher while nothing is queued). Heartbeat is now 2 h; the detector ages a spoke by the newest file in its output dir and wakes on a busy footer with 45 min of quiet. ⛔ `/tmp/pw-watch.sh` is the PAPER Hub's — never kill it.
- ⭐ **W24 CLOSED.** The package is at the Mac's `/Users/user/Desktop/nclu-iclr-code/` (196 files, 48.0 MB, no `.git`), with the report at `.claude/outputs/w24-advisor-report.md` and the full harvest at `.claude/outputs/w24-repro-package/`. The batmac session is reaped.
  - The Hub's re-scan caught a `PW-1`/scrub-scar leak that the spoke's scanner had missed; it was fixed through AMENDMENT-3 and re-proved.
  - The Head ACCEPTED all four open items as-is and wants no action on them: the licence stub, Tab./Fig. adapt left out, CMAZE-T baselines not reproducing across machines (LSTM 0.800 vs 0.627), and the across-wells control at 0.1273 vs 0.1065. ⭐ Scripts at `/tmp/{watch3,detect-box,detect-bm}.sh` — ⚠ `/tmp` does not survive a Mac reboot; recreate from §9e.
- **All budgets EXHAUSTED:** `shapes-matched` 3/3 · `maze-t-symmetric` 9/9 · `maze-t-seed4` 3/3 · `maze-t-rel-newseeds` 3/3. ⛔ **Nothing may be spent without the Head opening a new target.**
- **Waves 15–23 ALL CLOSED**, each with an advisor report on disk.

## ⭐⭐ WHAT SURVIVES A THREAD CHANGE, AND WHAT DOES NOT
- ⛔⛔ **The watcher does NOT.** The process may live, but its wake goes to the thread that armed it. **At boot: `pkill -x -f 'bash /tmp/watch3.sh'`, then background `bash /tmp/watch3.sh` DIRECTLY, ALONE.**
- ⛔⛔ **VERIFY IT IS TRACKED, EVERY TIME:** `ps -axo pid,ppid,args | awk '$NF=="/tmp/watch3.sh"'` — ⭐⭐ **`PPID 1` MEANS ORPHANED** (wakes nobody while looking alive). Causes: `rearm.sh`'s `exec`, or `&` inside the background task. Both reproduced 2026-09-24/25.
- ⛔⛔ **ONE EXPERIMENT HUB AT A TIME.** `/tmp/watch3.sh` is a single global resource; two Hubs re-arming it blind each other. ⭐ `ListAgents` at boot; retire peers.
- ⭐ **tmux work on both boxes survives.** Nothing to re-arm remotely.
- ⛔⛔ **NEVER `tmux kill-server` on batmac** — `hubhost` holds its keychain; recovery is physical console only (§9d).
- ⛔ **Never a box-wide kill** (`taskkill /F /IM python.exe`) — it once destroyed two agents' scored runs.

## WHAT YOU OWE, in order
1. ⛔ **Do NOT arm a watcher while holding** (Head, 2026-09-25) — arm only once the Head gives a wave.
2. ⛔ **Nothing is queued.** All waves closed; both boxes idle apart from the paper Hub. ⭐ **Do not invent a wave** — the Head scopes them with the Advisor.
3. ⛔ **HELD BY THE HEAD — do not act without an explicit instruction:**
   - ⭐⭐ **The negative-test sweep** (the Hub's strongest recommendation, w23 §6.3): **478 of 507 manifest cells are never read**, and an unmutated build into a missing dir exits 1 — so a large share of the register's tamper tests **could not have caught anything**. The register is the paper's verification story.
   - **The two w16 CMAZE-T per-arm placeholders** — ⛔ unbuildable from any artefact (w16 persisted no arm object). **Cut the panels, or re-run w16.** Nothing else clears them.
   - Referee **M3** · **MF-11** (length) · **MF-12** (venue items) · the abstract's **maze sentence** (`\TODO`).
4. ⛔ **`w17-paper/`, `figures/`, `paper-visuals/` are NOT YOURS.** Read for evidence; ⛔ never write.

## STATE THAT CANNOT BE RECONSTRUCTED
**⛔⛔ STANDING RULE (Head, 2026-09-24, all waves): PRE-REGISTRATIONS ARE BANNED.** No bars, labels or predictions before a number. ⭐ **Produce and print everything; the Head decides what is quotable.** ⚠ Earlier waves' `PREREG-*.md` are an **anti-pattern** — say so in every launch prompt or spokes copy them. ⭐ **The ledger row survives** (a spend record, not a prediction).
**⭐ The Head's LABEL RULE** replaces the old margin bars: **WIN** (ahead on mean + majority of seeds, gate holds) · **COMPARABLE** (within one seed's spread) · **LOSS**, mechanism named, ⛔ no numeric bar.
**⛔ Caveats that MUST travel with their claim** (full evidence in the w20/w21/w23 reports):
- **Shapes headline is ≈ +7**, not the draft's +39…+49 — and the asymmetry was **training steps, not data**. The old comparator ran **1,280 steps**.
- ⚠ **A single 2,560-step rival fit varies ±4.1 novel/corner on the PRNG draw alone** — against that ≈+7. ⛔ Print fit-level variance beside any single-fit margin. ⚠ The PRNG-replicates at the *carried* budgets were **declared NOT-RUN** (Head: "not cheap enough").
- **Deletion:** exact on store **and** book, but **conjunction deletion removes red-with-other-shapes (−6.1…−6.4, 3/3 seeds)**; a random-1 % control moves −0.02 ⇒ **content, not instability**. *"Reads no training rows"* must print the book's **0.97 GB** statistic and refit time beside it.
- **The maze contribution is an INPUT REPRESENTATION, not a memory result** — `+rel` recovers 97–100 % of the margin; a control ties the CLU on seed 4 (0.9004, exact on all 16 tuples); a rival ties it on seed 5.
- ⭐⭐ **w23: the seed-5 misses are CAPACITY, not generalisation** — the store fails that outcome code on **211/317 of its own FIT rows**. And **"eq-columns all-zero" is FALSE**: relations reach the store, **identity does not**.
- ⚠ **w23 corrected w22:** the w15 register closure covers **212 of 2,293** `% SRC` tokens — the three w15 spokes only, ⛔ **not the paper**. The drafts print **27.135×**.
- ⛔ **C-512 is unverified under symmetric inputs** — may not be promoted in A1's place.
**⚠ Known traps:** ⛔ Windows python cannot open an msys `/c/...` path → `cygpath -m`. ⛔ `env_w3.sh` joins `PYTHONPATH` with `:` → re-export with `;`. ⛔ `[ -f END_UTC ]` fires EARLY — use the triple check. ⛔ **Prune on harvest in BOTH places** (detector *and* `tmux kill-session`). ⛔ **Namespace session AND output dir identically** — a mismatch made a finished spoke read as stranded (2026-09-25). ⛔ **A paraphrase is not a source** — fold from JSONs; the artefact wins.
