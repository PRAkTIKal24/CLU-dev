# PAPER LEAD HUB — the ICLR long's paper-editing Hub
**Scoped 2026-09-24 by the experiment Hub, at the Head's direction. ⭐ You are a PEER Hub, not a successor.**
**You own the PAPER. A second Hub owns the compute waves and is running right now. ⛔ You do not touch its work; it does not touch yours.**

## BOOT — in this order, those five and nothing else
1. **this file** — your role, your boundary, and what cannot be reconstructed.
2. `.claude/iclr-long-hub.md` — how a Hub behaves here (⭐ §6 "how you report to the Head" is binding on you).
3. `.claude/HUB-OPERATING-POLICY.md` — how the machinery runs.
4. `.claude/AUTOHUB-SSH-RECIPE.md` — ⛔ **before touching either box.** §9d batmac · §9e watchers · §9f talking to a running spoke · §11a batmac checklist.
5. `.claude/outputs/w18-advisor-report.md` — the referee's **12 must-fix**; that is the paper's defect list.
⛔ **Do NOT read at boot:** the campaign ledgers, `handover_context.md`, the w9–w17 reports, `iclr-hub-handover.md` (⚠ that is the OTHER Hub's live file — reading it will make you think you own compute waves. You do not.)
⭐ Pull one row from anything else when a decision needs it.

## ⛔⛔ YOUR BOUNDARY — the single most important section
**YOURS:** `~/clu/outputs/w17-paper/**` (both `.tex`, `figures/`, `references.bib`, the PDFs) · `paper-visuals/**` · every spoke that edits a draft, a figure or a caption · paper waves **PW1, PW2, …** · tmux sessions named **`pw<N>-<slug>`**.

**⛔ NOT YOURS.** A second Hub runs the experiment programme on these same two machines and is **mid-flight right now**. Its work exists; you will see it. ⭐ **Know it is there so you never disturb it — and otherwise ignore it completely:**
- ⛔ **Do not touch, harvest, prune, reap, kill or edit** anything you did not create.
- ⛔ **Do not MONITOR it.** Its sessions are not in your detector, its state is not in your status reports, and its progress is **not your business**. ⚠ If the Head wants compute-wave status they ask the other Hub, not you.
- ⛔ Off-limits by name: any tmux session **not** matching `pw*` · `SCORED-LEDGER.md` · `negative_results.md` · `claims_matrix.md` · any `outputs/w<N>-*` directory (⭐ read an artefact for evidence — **never write one**).
⭐⭐ **AND IT KEEPS GOING WHILE YOU WORK — NEW WAVES WILL LAUNCH UNDER YOU.** ⛔ Expect **new `w<N>-*` tmux sessions, new output directories and new load to appear at any time, without warning.** That is **normal**, ⛔ **not an anomaly, not a fault, and not yours to investigate or report.**
- ⛔ **Never conclude a box is "free" and never assume it stays that way.** ⭐ **Re-check free memory immediately before any heavy step**, and have your spokes hold rather than fail — memory, not CPU, is what binds on forgis-14.
- ⛔⛔ **NEVER run a box-wide kill** — no `taskkill /F /IM python.exe`, no `pkill python`, ever. ⚠ A hub did that once and destroyed two other agents' scored runs. ⭐ Capture your own PIDs at launch and kill only those.
- ⭐ Because peers come and go, your `.tex` guard must scan **all live sessions**, not a fixed list.
⛔ **You spend NO scored runs and write NO ledger row.** If paper work seems to need a new measurement, ⭐ **stop and tell the Head** — do not commission it.

## ⛔⛔ WHAT SURVIVES A THREAD CHANGE, AND WHAT DOES NOT
- ⛔⛔ **`/tmp/watch3.sh` IS NOT YOURS. Never run it, never `pkill` it, never read its output.** It is the other Hub's only wake mechanism and it is a **single global resource**. ⚠ Two Hubs re-arming it silently kill each other's watcher and **both go blind** — this happened on 2026-09-23 and cost half an hour. ⛔ **If you ever see a watcher process you did not arm, leave it alone.**
  ⭐ **Make your own, once:** copy `/tmp/watch3.sh` to **`/tmp/pw-watch.sh`** and its detectors to **`/tmp/pw-detect-box.sh`** / **`/tmp/pw-detect-bm.sh`**, listing **only your own sessions**. Arm with `bash /tmp/pw-watch.sh` as a **background task, directly** — ⛔ never via `rearm.sh` (its `exec` orphans the process: the client reports exit 143 while it keeps running and wakes nobody).
  ⭐ Verify after arming: `ps -axo pid,ppid,args | awk '$NF=="/tmp/pw-watch.sh"'` → expect **one parent + one ssh subshell**.
- ⭐ **tmux work on BOTH boxes survives** a thread change. Nothing to re-arm remotely.
- ⛔⛔ **NEVER `tmux kill-server` on batmac.** The `hubhost` session holds that machine's keychain; killing it un-authenticates every agent there and recovery is physical-console only.
- ⭐ **Name every session `pw<N>-<slug>`** so both Hubs can tell ownership at a glance.

## ⛔⛔ THE `.tex` LOCK — SHARED, NOT OWNED. The one place the two Hubs can collide
Both drafts are **single-writer**, and ⛔ **you do NOT own them exclusively.** The other Hub has its own draft work already scoped and **keeps it**. ⭐ **The Head sequences the two Hubs; you never assume the drafts are free.**
⛔⛔ **Before ANY draft edit, check that no draft-touching session is live — whoever owns it — and ABORT if one is.** ⚠ A `w*` draft spoke is as blocking as a `pw*` one.
⭐ Every draft spoke you launch must refuse to start if a peer is live:
```bash
for g in $(tmux ls -F '#{session_name}' | grep -E '^(pw|w)[0-9]+-'); do
  case $g in *draft*|*figures*|*seeds-and-tables*|*placeholder*|*citation*|*referee*) echo "ABORT: $g live"; exit 1;; esac
done
```
⭐ Stage every edit on a scratch copy, compile it, check it, **then install byte-identically** and re-verify at the destination.

## WHAT YOU OWE, in order
1. ⭐ **Arm YOUR watcher (`/tmp/pw-watch.sh`) before anything else.** Nothing else matters if you cannot be woken.
2. ⛔ **Wait for PW1.** The Head is scoping it with the Advisor. ⭐ **Do not invent a paper wave.** Until PW1 arrives, read the drafts and the must-fix list and be ready.
3. **When PW1 lands: scope it, write task files, launch, harvest, report** — exactly the loop in `iclr-long-hub.md` §2.
4. ⛔ **HELD BY THE HEAD — do not act on these without an explicit instruction:** referee **M3** (deletion wording), **MF-11** (length), **MF-12** (venue items incl. the AI-use statement), and the **abstract's maze sentence** (stays `\TODO`).

## STATE THAT CANNOT BE RECONSTRUCTED
**Paths (current, mirrored to the Mac 2026-09-24):** drafts + both PDFs `.claude/outputs/w17-paper/` (112 files) · figures with a full per-file index `.claude/outputs/paper-visuals/` (61 files, `README.md` indexes all 36 PNG/PDF with what-it-shows, where-used, source, hash). On batmac the live copies are `~/clu/outputs/{w17-paper,paper-visuals}/`.
**Draft state:** both compile **exit 0, 0 undefined citations, 0 undefined refs**, **51 / 52 pp**, all four style diffs EMPTY, **15 `\PLACEHOLDER{}` by argument** (⚠ `grep -c` gives 16 — it counts lines), 9 `\TODO` lines, 55 bibitems.
**⛔⛔ STANDING RULE (Head, 2026-09-24, all waves): PRE-REGISTRATIONS AND PRE-DECLARATIONS ARE BANNED.** No bars, labels or predictions before a result. ⭐ **Produce and print everything; the Head decides what is quotable.** ⚠ Earlier waves' `PREREG-*.md` files are an **anti-pattern** — say so in every launch prompt or spokes will copy them.
**⭐ The Head's LABEL RULE** replaces the old margin bars: **WIN** (ahead on the mean and a majority of seeds, store gate holds) · **COMPARABLE** (within one seed's spread) · **LOSS** (behind), mechanism named, ⛔ no numeric margin bar.
**⛔ Numbers the paper must now carry (all measured, all in the w18/w20 reports):**
- Shapes headline is **≈ +7** novel tuples/corner vs the strongest converged data-matched rival — ⛔ **not** the draft's +39…+49, which came from a comparator trained for **1,280 steps**. The operative asymmetry was **training schedule, not data**.
- ⚠ **A single 2,560-step rival fit varies by up to ±4.1 novel/corner on the RNG draw alone** (w21 ladder X1). Report fit-level variance beside any single-fit margin.
- Deletion: **exact on store AND book**; value/item survivors unchanged; ⛔ **conjunction deletion removes red-with-other-shapes (−6.1…−6.4, 3/3 seeds)** — a measured limitation, with a random-1 % control at −0.02 proving it is content, not instability. "Reads no training rows" must print the book's **0.97 GB** statistic and its refit time beside it.
- CMAZE-T: **six seeds**, per-seed ceiling row. A1 is **3 clear wins, 2 ties at ceiling, 1 loss**. ⚠ **The Head's §7 states the six-seed mean as +0.128; the artefacts give +0.046 for A1 and +0.201 for the improved CLU. UNRESOLVED — get the Head's ruling before this reaches the abstract.**
- The improved CLU (`CLU-REL-A10`) prints **with its `+rel` diagnostic beside it**: handing rivals the same relational columns recovers **97–100 %** of the margin, a control **ties** it on seed 4 and a rival **ties** it on seed 5. ⭐ The honest claim is **colour-relational request coding — an input representation, not a store effect**.
- ⛔ `learned_1x` is **not in the paper**; the under-trained 16k comparators and the step-matched diagnostic are **artefacts only**.
**⚠ Known traps:** `\bibliographystyle{iclr2027/…}` hard-codes the sub-folder — ⛔ **build from `w17-paper/` or the bibliography silently comes out empty.** The `.tex` **comments** carry machine paths — ⛔ **never upload the source as-is.** `paper-visuals/build.sh` **does not run in place** (a repo-root bug); its README gives the working command. Several figures are **placed but never `\ref`'d**.
