# ICLR LONG — HUB HANDOVER. Rewritten 2026-09-23 ~21:4xZ. ⛔ Lean by design. Read, then act.

## BOOT — in this order, those five and nothing else
1. `.claude/iclr-long-hub.md` — the role.
2. **this file** — live state and what you owe.
3. ⭐⭐ `.claude/HUB-OPERATING-POLICY.md` — how this Hub runs.
4. ⭐⭐ `.claude/AUTOHUB-SSH-RECIPE.md` — **before touching either box.** ⚠ Rewritten 2026-09-23: **§9d batmac**, **§9e watcher discipline**, **§9f talking to a running hub**, **§11a batmac checklist**. Every line was learned by getting it wrong.
5. `.claude/outputs/w17-advisor-report.md` — the paper's state.
⛔ **Do NOT read at boot:** the campaign ledgers, `handover_context.md`, or the w9–w16 reports. ⭐ Pull one row when a decision needs it. The w15/w16 reports are short and are cited below where they matter.

## ⭐⭐ WHAT IS LIVE RIGHT NOW — verified 2026-09-23 ~21:4xZ
**forgis-14 — 2 scored hubs, 17 python procs:**
| session | state | budget |
|---|---|---|
| `w19-maze-t-symmetric` | WORK | **3/9**. CMAZE-T under symmetric inputs. Latest decision line: **WORDS-MAIN and WORDS-FULL DECLARED, 0 further runs; OBS pending.** ⚠ Its OBS word-extractor was under-fit (train tuple-acc 0.404/0.683/0.767) → `PREREG AMENDMENT 1`, 60k-step diagnostic. |
| `w20-shapes-data-matched` | ACTIVE | **0/3**. Spend plan (`PREREG AMENDMENT 3`): run 1 = M-131k rows 3–4 **incl. LACE/PoE@131k**; run 2 = M-16k; run 3 = **matched deletion**. |

**batmac — 0 live spokes.** ⭐ `hubhost` ALIVE (that is the keychain — see §9d). The whole paper chain is **harvested and reaped**: placeholder-fill → draft-figure-slots → referee3-fixes → citation-prune → referee-4 → referee4-fixes.

⭐ **Waves 15, 16, 17 CLOSED** with advisor reports written. **19 and 20 in flight.**
⚠⚠ **WAVE 18 IS *NOT* CLOSED** — an earlier version of this file said it was; that was wrong.
**4 of 5 spokes finished** (`placeholder-fill` pass 2 COMPLETE · `referee-4` · `referee-fixes` · `seeds-and-tables` pass 2). ⛔ **TWO PASSES ARE OUTSTANDING, both blocked on w19 landing:**
- `registry-fold` **pass 4** — for **w19 `maze-t-symmetric` only** (pass 3 folded everything else).
- `seeds-and-tables` **pass 3** — fold w19/w20 rows, **plus two of its own open findings** (below).

## WHAT SURVIVES A THREAD CHANGE, AND WHAT DOES NOT
- ⛔⛔ **The watcher does NOT.** The process may live, but its wake goes to the thread that armed it. ⭐ **At boot: `pkill -x -f 'bash /tmp/watch3.sh'`, then arm a fresh one ALONE (`bash /tmp/rearm.sh`, never chained).** Scripts live at `/tmp/{watch3,rearm,detect-box,detect-bm,reap}.sh` — ⚠ **`/tmp` does not survive a Mac reboot; recreate from §9e if gone.**
- ⭐ **tmux work on BOTH boxes survives.** Nothing to re-arm remotely.
- ⛔⛔ **NEVER `tmux kill-server` on batmac.** `hubhost` holds its keychain; killing it un-authenticates every agent there. Recovery is **physical console / Screen Sharing only** (§9d).
- ⭐ Check at boot: `ssh forgis-14 … 'tmux ls'`, `ssh batmac@batmac 'tmux ls'` (expect `hubhost`), and `bash /tmp/detect-box.sh` via ssh.

## ⭐⭐ STANDING AUTHORISATION (Head, 2026-09-23) — DO NOT WAIT TO BE TOLD
1. ⭐ **When w19 closes: LAUNCH THE w18 RE-PASSES IMMEDIATELY.** `registry-fold` pass 4 and `seeds-and-tables` pass 3. **Fully authorised; no approval needed.**
2. ⭐ **When w20 closes: harvest and COMPILE ITS REPORT without waiting for instructions.**
3. ⭐ **Resolve minor ambiguities yourself.** Pre-authorised. Record the choice and the reasoning in the wave report and move on. ⛔ Still escalate only a genuinely NEW TARGET, or a call with no budget left to absorb a wrong answer.

## WHAT YOU OWE, in order
1. ⭐ **Arm a watcher before anything else.** Nothing else matters if you cannot be woken.
2. **Harvest w19 and w20 when they finish**, then **write `w19-advisor-report.md` and `w20-advisor-report.md`**. Both are owed; w15/w16/w17 are already written.
3. **The two w18 re-passes** (above) the moment w19 closes — you are authorised, do not ask. Fold into their briefs these **two open findings `seeds-and-tables` raised and nobody has actioned**:
   - ⛔ **MF2's rounding slip is still in the draft:** `23.5117` at `DRAFT-v2.tex:1317` and `:2262`. w15 `table-fixes` corrected it to **23.5121** in DRAFT-v1 and the fix never propagated to v2.
   - ⚠ The ladder's **n ≥ 3 TRAINING-DATA note** travels with `tab:ladder` but **not with `tab:k5`**, which has the same rungs.
   ⭐ The **11 `\PLACEHOLDER{}`** (10 w19 + 1 w20) are filled by a `placeholder-fill` pass 3 once both hubs land.
4. ⛔ **HELD FOR THE HEAD — do not touch:** referee **M3** (deletion wording), **MF-11** (main text ≈**18 pp** vs ICLR 2027's **9 pp strict desk-reject** limit), **MF-12** (venue items incl. AI-use statement).
5. ⭐ Waves are **PRE-AUTHORISED** in `advisor-wave-plan-w15-w18.md` (§1–§6, now includes w19 and w20). ⚠ **That file is LIVE — the Head appends sections mid-run. Re-read it; do not work from memory.**

## STATE THAT CANNOT BE RECONSTRUCTED
**Bases:** `base/iter8 = 148f3b7` **on the box only** (the Mac's same-named branch is stale `fa60b15`). w19 branched from **`w16/maze-wm = 9f80f59`** (what w16 scored on). w20 from `base/iter8`.
**Budgets:** `maze-t-symmetric` 3/9 · `shapes-matched` 0/3. ⛔ **EXHAUSTED: single-design, maze, read-verb, composition, generation-table, encoder-loop.**
**Paper:** 55 bib entries, every one cited, **0 undefined citations**. Four style diffs EMPTY. **Referee-v3 = 4/10, "READY for the Head's editing pass, NOT ready for submission."**

**⛔ Caveats that MUST travel with their claim:**
- **Deletion is NOT capability erasure** — after deleting every red row the 20 deleted corners still compose red, 3 seeds (w15).
- **K\* is not resolvable by its own rule** (tolerance 0.02; seed sd exceeds it at K=10,16,20,24,28,32). ⭐ The finding is **the plateau**, ledger minimum at K=24.
- **CPU BAR D is UNMEASURED, not VOID** (registry `N359`: the w8 scorer loaded a rival from the wrong feature space). **GPU BAR D = MIXED, no claim either way.**
- **`landscape_shuffled_across_wells` must be a DERANGEMENT** (`N350`: `rng.permutation(K)` at K=2 is the identity half the time). ⚠ **w15's harness control columns used the old control and were never re-measured.**
- **LACE/PoE**: as instantiated **in this feature space at their own budgets** — ⛔ not a refutation of LACE in general.
- ⛔ CMAZE-1/2/2b/T/O never share a table. Seed count is a column. Every byte figure names its convention.

**⚠ Traps that cost this Hub time — all real, all in §9d–§9f:**
- ⛔ Windows python cannot open an msys `/c/...` path → `cygpath -m` first. This silently broke a gate for hours.
- ⛔ `[ -f END_UTC ]` fires EARLY — an agent writes it before finishing. Use the **triple check**.
- ⛔ **Prune on harvest in BOTH places** — detector *and* `tmux kill-session`. 24 dead sessions accumulated before this was noticed; one blocked a launch.
- ⛔ `pkill -f '/tmp/watch3.sh'` kills your own shell. Use `pkill -x -f 'bash /tmp/watch3.sh'`.
- ⛔ **A paraphrase is not a source** (`N351`): a wave report written from a summary disagreed with its primary artefact in **8 places**. ⭐ Ship primary artefacts to whichever machine does the folding.
- ⛔ **Your LOCAL copy of a spoke's `FINDINGS.md` goes stale the moment that spoke runs another pass.** ⚠ This Hub answered "is w18 closed?" from a stale pull and got it wrong. ⭐ **Re-pull before reading any multi-pass spoke's findings.**
- ⚠ **Do not over-claim a compile defect:** the "226 undefined citations" figure was the w17 logs; `registry-fold` measured the w19 **baseline** at **61**. The rule stands — *an exit code is not a resolved bibliography* — the number did not.
