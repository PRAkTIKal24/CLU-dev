# ICLR LONG — HUB HANDOVER. Rewritten 2026-09-23 ~21:4xZ. ⛔ Lean by design. Read, then act.

## BOOT — in this order, those five and nothing else
1. `.claude/iclr-long-hub.md` — the role.
2. **this file** — live state and what you owe.
3. ⭐⭐ `.claude/HUB-OPERATING-POLICY.md` — how this Hub runs.
4. ⭐⭐ `.claude/AUTOHUB-SSH-RECIPE.md` — **before touching either box.** ⚠ Rewritten 2026-09-23: **§9d batmac**, **§9e watcher discipline**, **§9f talking to a running hub**, **§11a batmac checklist**. Every line was learned by getting it wrong.
5. `.claude/outputs/w17-advisor-report.md` — the paper's state.
⛔ **Do NOT read at boot:** the campaign ledgers, `handover_context.md`, or the w9–w16 reports. ⭐ Pull one row when a decision needs it. The w15/w16 reports are short and are cited below where they matter.

## ⭐⭐ WHAT IS LIVE RIGHT NOW — verified 2026-09-24 ~00:45Z by the lead Hub
**forgis-14 — 3 agents. ⚠ At the 3-worktree cap; ⛔ launch nothing more there.**
| session | state | budget |
|---|---|---|
| `w20-shapes-data-matched` | ACTIVE | ⭐ **3/3 — BUDGET FULLY SPENT** (row 3 written 23:36:23Z, before the run; it reads `3/3 (LAST)`). Run 3 = matched deletion under `AMENDMENT-2`; its rival scoring finished 36/36 at 04:02Z. Finishing analysis/report. |
✅ **`w21-maze-t-clu-improve` DONE, HARVESTED, REAPED (11h58m). Runs 7–9/9 spent. Code `w21/maze-t-clu-improve = f491a04` (868 insertions, 14 deletions — it edits the CLU, so ⛔ NOT purely additive).**
⭐⭐ **RESULT: WIN by the Head's rule.** `CLU-REL-A10` BOTH **TEST 0.929** vs best matched rival **0.685** (transformer 80k), ahead **3/3** seeds (0.908/0.894/0.984 vs 0.803/0.567/0.685). BOTH VAL 0.937.
⛔⛔ **BUT THE MECHANISM IS AN INPUT REPRESENTATION, NOT THE STORE — and the spoke says so itself.** Give the **same relational columns to every rival** (`+rel` diagnostic, matched floats, frozen recipe): the best rival goes **0.685 → 0.862**. ⇒ **~0.18 of the 0.24 margin is the CODING, which any learner can be handed.**
- The CLU keeps **+0.066** mean over the best `+rel` rival, and is ahead on only **2 of 3** seeds — ⛔ **behind** mlp+rel 80k on seed 0 (0.908 vs 0.915).
- ⛔⛔ **On WORDS-FULL VAL the store's contribution is ZERO:** matched transformer+rel **0.941** vs CLU **0.937**.
⭐ **Mechanism, named and measured:** A1's held-out failures are all-or-nothing per tuple — for a colour `c`, `(pocket=c, tint=c)` never occurs in train, so an **identity**-coded design (one-hots, pairs, triples) has nothing to read, and neither does a rival given the same input (every matched rival sits at exactly **0.5674** on TEST seed 1). The new CLU adds `eq(x,y)` — "these two colour words name the same colour" — as ONE column, learnable from other colours. Those rows go **0 → 1.000**. ⛔ The `(c,c,c)` interaction family is **still 0 on every seed**.
⚠ The **store gate holds on means but is weaker than A1's** (controls climb with atom count).
⭐ **The honest claim: colour-relational request coding — an input representation carried through the CLU's closed-form head. It is NOT a store effect.**
⛔ DID NOT RUN (declared): WORDS-FULL TEST with the new CLU, OBS / WORDS-MAIN with the new CLU, dyn-interactions, two-stage fit, item-store, read-verb variants.

✅ **`w21-words-full-test` DONE, HARVESTED, REAPED 2026-09-24.** Runs **4–6/9 spent**. **WORDS-FULL TEST = LOSS** by the Head's rule: CLU **0.7246** vs matched transformer-8k **0.7421**, Δ̄ **−0.0175**, σ 0.0140, CLU ahead on **0/3** seeds; **store gate holds**. ⭐ TEST reproduces VAL almost exactly (−0.0175 vs −0.0173) and **nothing was re-tuned**. Code `w21/words-full-test = 6a3db43` on the Mac (3 files, 358 insertions, **0 deletions**).
⭐⭐ **Its integrity checks are the reason to trust it:** the VAL instrument reproduction was **bit-identical** to w19 (max |Δ| = 0 over CLU + 4 controls × 4 rows × 3 seeds), and WORDS-FULL TEST CLU + controls == w19 BOTH TEST CLU + controls **bit-for-bit**.
⚠ **It declared the label's sensitivity rather than hiding it:** a σ taken from the CLU's own per-seed spread (0.137) or R\*'s (0.151) would read **COMPARABLE**; the PREREG fixed the **paired-difference** spread before any TEST number, so the label stays **LOSS**. ⭐ That spread is almost entirely seed 1's shared data ceiling (0.5674, where every strong arm sits).
⚠ **Correction it made on itself:** rival rollout validity is **not** exactly 0 — max matched **0.111**, diagnostic **0.289**. TABLES/FINDINGS print the maxima; "no arm validates a rollout" stays as the limitation, with those numbers.

**batmac — 1 live spoke:** `w21-figures-redraw` (wave-plan §6a; `dataviz`; worktree from `w19/maze-t-symmetric = 0455961`). ⭐ `hubhost` ALIVE.
⚠ **Staged for it:** w19's **BOTH** persisted objects at `~/clu/stage/w19-persist/BOTH_seed{0,1,2}/test/` (270 files, 59 MB, hashes verified on arrival), plus `w21-words-full-test` and a refreshed `w19-maze-t-symmetric` in `~/clu/evidence/outputs/`.
⛔⛔ **§6a calls figures-redraw "zero compute" but its maze part is NOT** — the run JSONs carry **no per-item results**, so per-arm panels require loading each persisted object and re-reading on a machine that did not score it. ⭐ The brief therefore gates Part C on reproducing w19's BOTH TEST n-way (**0.800 / 0.566 / 0.808**, controls 0.000/0.225/0.244/0.166); **if it does not reproduce, Part C stops and Parts A/B/D still ship.**

⭐⭐ **`spoke/ladder-rivals-32k` is STAGED, NOT LAUNCHED** (`/c/chlu/tasks/w21/{ladder-rivals-32k.md,launch-*.txt,run-*.sh}`). ⛔ **Mechanical precondition: `/c/chlu/outputs/w20-shapes-data-matched/END_UTC.txt` must EXIST** — it must never run while that hub is live. ⭐ Launch it on the w20-DONE wake. It is a **FREE replication — no ledger row, no scored run.**

⭐ **batmac keychain:** verified `no-timeout`, RC=0 from a session created **inside** the `hubhost` server; ⛔ RC=36 from a plain SSH shell is the wrong test, not a fault.

⭐ **Waves 15, 16, 17 CLOSED** with advisor reports. ⭐⭐ **WAVE 19 CLOSED — `w19-advisor-report.md` WRITTEN** (3/9 spent, ALL FOUR CONDITIONS LOST; code `w19/maze-t-symmetric = 0455961` on the Mac, 1,144 insertions / 0 deletions).
⭐⭐ **WAVE 18 CLOSED 2026-09-23.** Both re-passes harvested, reaped, code landed from **their own** bundles: `w18/seeds-and-tables = f52e794`, `w18/registry-fold = 9fc536b`. ⚠ Each bundle carries **five** w18 branches at different times — ⛔ never fetch a spoke's branch from a peer's bundle.
⭐⭐ **WAVE 21 LAUNCHED 2026-09-24 00:42Z** per wave-plan **§5b** (Head-ruled; ⛔ **§5b supersedes §5a's "no margin claim" line**).

## ⛔⛔ STANDING RULE — PRE-REGISTRATIONS ARE BANNED (Head, 2026-09-24, ALL WAVES)
**No pre-declarations. No bars, labels, thresholds or predictions before the number exists.** ⛔ Write no PREREG. ⚠ **This reverses years of practice in this repo — every earlier wave's `PREREG-*.md` is a pattern to NOT copy**, and every launch prompt must say so explicitly or the spoke will imitate what it finds on disk.
⭐ **PRODUCE AND PRINT EVERYTHING** — every arm, seed, row and candidate, including the bad ones. **The Head decides afterwards what is quotable.**
⭐ **Provenance continues unchanged:** config, seed, param + file hashes, in `LOG.md`.
⭐ **The LEDGER ROW SURVIVES** — it is a SPEND record, not a prediction. Row before each scored run, append-only, seed + condition, ⛔ no expected outcome. It is the only thing stopping a target being overspent.
⛔⛔ **THE CONSEQUENCE THE HUB MUST POLICE: completeness is now the ONLY protection against post-hoc selection.** A number measured and not printed is a defect. ⚠ A referee will attack "labels chosen after seeing the results"; the only defence is that nothing was withheld.

## ⭐⭐ THE FOURTH-SEED DESIGN — AMENDED BY THE HEAD 2026-09-24 (⛔ screening CANCELLED)
⛔ **The original §5c accept/reject screening is DEAD.** No ±0.07 window, no stopping at the first passing seed, ⛔ no seed discarded.
⭐ **Run seeds 3, 4 AND 5, keep all three:**
- **Winner row — words-plus-observation on TEST, CLU A1 + ALL RIVALS** = the **3 scored runs** of target `maze-t-seed4`.
- **Comparable row — observation (OBS) on VAL, CLU + ALL RIVALS** = ⭐ **FREE** (w19's OBS was VAL-only, so its COMPARABLE label is a VAL label; this is the like-for-like extension).
- ⛔ **`words-full` gets NO new seeds** (a LOSS row; the Head did not include it). ⚠ Its row shows **3 seeds** against the others' **6** — print the seed count per row, ⛔ never average across different seed sets.
⭐⭐ **Presentation: EASY seeds → MAIN TEXT, HARD seeds → APPENDIX, both printed in full.** The finding to support with numbers: **on a hard seed every model struggles, ours included** ⇒ that seed's held-out data is **inherently hard**, not a CLU failure.
⭐⭐ **The per-seed CEILING ROW (best score ANY arm reaches, arm named) is now the LOAD-BEARING evidence for that claim.** ⛔ Seeds are classified easy/hard by **measured ceiling**, never by the CLU's own score — classifying by our own result would be choosing seeds by our own answer. The cut goes in the caption.
⭐ The zero-run "unreachable held-out combinations" diagnosis extends to **every** hard seed, not just seed 1.
⚠ Recorded as `AMENDMENT-1.md` in `/c/chlu/outputs/w21-maze-t-seed4/`; the spoke was pointed at it mid-run.

## ⛔⛔ "INHERENTLY HARD" IS NOT SAFE AS STATED — seed 1 is hard for a REPRESENTATION, not in itself (w21 seed4 diagnosis v2, 2026-09-24 08:29Z)
The Head's intended appendix claim is *"at hard seeds all models including ours struggle ⇒ that seed's held-out data is inherently hard."* ⛔ **The zero-fit diagnosis says that is false for seed 1 as stated.**
⭐ **Two different counts of "unreachable", and the gap between them IS the finding:**
| seed | unidentified for **colour-IDENTITY** designs | ⭐ also unreachable by **colour RELATION** |
|---|---|---|
| 0 | 118/761 | **0** |
| **1** | **403/742 (54 %)** | ⭐ **79 rows, 4 interaction tuples (≈11 %)** |
| 2 | 215/686 | **0** |
| 3 | 339/675 | 4 rows, 2 tuples |
| 4 | 305/723 | **0** |
| 5 | 40/660 | 40 rows, 4 tuples |
⛔⛔ **The measured refutation:** `CLU-REL-A10` (the improved, colour-**relational** CLU) reads **VAL seed 1 = 0.9141**, against an **A1-family ceiling of 0.5689** — the ceiling over every arm using colour-**identity** designs, rivals included. **Our own improved CLU clears the "ceiling" of the seed that was going to be called inherently hard.**
⭐ Mechanism: every class-D tuple has a colour pair (pocket×door or pocket×tint) that is a held-out 2-way pair ⇒ 0 train rows by construction **under identity coding**. Under relational coding that block mostly disappears; only the `(c,c,c)` signature `pocket==door==tint` is genuinely absent under **any** colour (seeds 1 and 5).
⚠⚠ **CONSEQUENCES THE HUB MUST ENFORCE:**
1. ⛔ The appendix sentence cannot say "inherently hard" unqualified. ⭐ The defensible claim is: **"hard for colour-identity representations — every rival and A1 — and largely solved by relational coding."**
2. ⛔ **The CEILING ROW must name its representation family.** Seed 1's ceiling is **0.5689 over the A1/identity family** but **0.9141 once CLU-REL-A10 is in the table.** A ceiling row that silently mixes families is wrong.
3. ⭐ This is a **better** result than the one it replaces: it is a mechanism, not an excuse. The improved CLU is not merely scoring higher — it is reaching a class of held-out rows that identity-coded arms cannot reach at all.
⚠ `0.9141` is a **VAL** number; the improved config's TEST legs (`maze-t-symmetric` 7–9) were still running when this was written.

## ⭐⭐ THE HEAD'S LABEL RULE (§5b) — it REPLACES the w16/w19 PREREG margin bars
Every condition prints in full with per-seed numbers. The label is the **measured relation to the best matched rival over BOTH budgets**: **WIN** (ahead on the mean and a majority of seeds, store gate holds) · **COMPARABLE** (within one seed's spread) · **LOSS** (behind) — with the mechanism named. ⛔ **No numeric margin bar.**
⭐ Under it, **words-plus-observation TEST (0.725 vs 0.626, store gate holds) = a WIN row.** ⭐ The **w16 pixel-rivals row prints beside** the symmetric rows with its asymmetry in the caption (Advisor default; the Head may move it to the appendix).

## WHAT SURVIVES A THREAD CHANGE, AND WHAT DOES NOT
- ⛔⛔ **The watcher does NOT.** The process may live, but its wake goes to the thread that armed it. ⭐ **At boot: `pkill -x -f 'bash /tmp/watch3.sh'`, then arm a fresh one ALONE — background `bash /tmp/watch3.sh` DIRECTLY.**
- ⛔⛔ **ARM IT ALONE, AND NEVER WITH `&` OR CHAINED TO ANOTHER COMMAND.** ⚠ Putting `&` inside the background task **detaches it**: the client reports the task *completed* — replaying the PREVIOUS wake's text, which looks like a fresh result — while the watcher runs on **orphaned** and its wake reaches nobody. ⭐⭐ **THE TELL, CHECK IT EVERY TIME: `ps -axo pid,ppid,args | awk '$NF=="/tmp/watch3.sh"'` — `PPID 1` MEANS ORPHANED.** A healthy watcher's PPID is the client's shell wrapper. ⚠ Reproduced 2026-09-25.
- ⛔⛔ **DO NOT ARM VIA `/tmp/rearm.sh` IN THIS CLIENT.** Its `exec` breaks the harness's background-task tracking: the client reports the task *failed, exit 143* while the process **keeps running orphaned**, and its wake reaches nobody. Two orphans were found this way on 2026-09-23. ⭐ Verify after arming with `ps -axo pid,ppid,args | awk '$NF=="/tmp/watch3.sh"'` and expect **one parent + one ssh subshell**; a second *parent* (different PPID) is a second watcher.
- ⛔⛔ **ONE HUB THREAD AT A TIME.** `/tmp/watch3.sh` is a single global resource. Two Hub threads each re-arm with the same `pkill` and silently kill each other's watcher, so neither wakes reliably. ⭐ Check `ListAgents` at boot; retire peers before arming. Scripts live at `/tmp/{watch3,rearm,detect-box,detect-bm,reap}.sh` — ⚠ **`/tmp` does not survive a Mac reboot; recreate from §9e if gone.**
- ⭐ **tmux work on BOTH boxes survives.** Nothing to re-arm remotely.
- ⛔⛔ **NEVER `tmux kill-server` on batmac.** `hubhost` holds its keychain; killing it un-authenticates every agent there. Recovery is **physical console / Screen Sharing only** (§9d).
- ⭐ Check at boot: `ssh forgis-14 … 'tmux ls'`, `ssh batmac@batmac 'tmux ls'` (expect `hubhost`), and `bash /tmp/detect-box.sh` via ssh.

## ⭐⭐ STANDING AUTHORISATION (Head, 2026-09-23) — DO NOT WAIT TO BE TOLD
1. ⭐ **When w19 closes: LAUNCH THE w18 RE-PASSES IMMEDIATELY.** `registry-fold` pass 4 and `seeds-and-tables` pass 3. **Fully authorised; no approval needed.**
2. ⭐ **When w20 closes: harvest and COMPILE ITS REPORT without waiting for instructions.**
3. ⭐ **Resolve minor ambiguities yourself.** Pre-authorised. Record the choice and the reasoning in the wave report and move on. ⛔ Still escalate only a genuinely NEW TARGET, or a call with no budget left to absorb a wrong answer.

## ✅ `w21/registry-fold` PASS 5 DONE (batmac, harvested, reaped, landed `f1ca948`)
Folded **w20 + all four w21 units** into `negative_results.md` (1.79 MB) and `claims_matrix.md` (567 KB). ⭐ It **declared what it lacked instead of substituting** (`NEEDS-ARTEFACTS.json`, 4 items).
⭐⭐ **RESOLVED — `learned_1x` is NOT missing, and nobody should chase it again.** w20's FINDINGS say it was "kept in artefacts"; the curator could not find it in w20's 1,481 staged files and asked. ⛔ **w20 never fitted new `learned_1x` rivals** — it referenced the **existing w12 rows** (`/c/chlu/outputs/w12-composition-ladder/runs/*/rows_learned_1x.json`, 20 files incl. `LADDER.json`) and declined to print them. The curator **did** fold the w12 value. ⭐ Those 20 files are now staged at `~/clu/evidence/outputs/w12-composition-ladder-learned1x/` on batmac. ⚠ **The ambiguity is w20's wording, not a lost artefact.**
⚠ **Three curator requests remain OPEN and are genuine gaps, not oversights:**
1. **The rows-3–4 corner → requested-content map** — needed to re-derive the seed-0 **−32.1 / −1.47 / +0.51** deletion breakdown and to compute it for **seeds 1–2**. ⛔ Not in any staged JSON (`N374`). It lives in w20's `deletion_store/` per-corner lists on the ThinkPad.
2. **Persisted objects** (w20/w21 rivals, stores, books) for hash re-verification on batmac — only JSONs and manifests were staged.
3. ⭐⭐ **`N380` compares UNPAIRED means from two instruments.** A paired `K*=32 − learned_1x` estimand on the shipped leg **does not exist**; producing one needs a free rescore, which the curator correctly did **not** run (it is zero-compute by charter). ⛔ **Do not quote N380 as a paired comparison.**

## ⭐⭐⭐ CLU-REL-A10 ON ALL SIX SEEDS — the improvement is REAL and LARGE; the STORE still is not doing it (`w21/rel-newseeds = 31eeadd`, 3/3 spent, 2026-09-24)
| seed | A1 CLU | **CLU-REL-A10** | REL best matched rival | REL best `+rel` | REL max control |
|---|---|---|---|---|---|
| 0 | 0.800 | 0.908 | 0.803 | 0.915 | 0.778 |
| 1 | **0.566** | **0.894** | 0.567 | 0.869 | 0.563 |
| 2 | 0.808 | 0.984 | 0.685 | 0.819 | 0.477 |
| 3 | 0.674 | **0.927** | 0.723 | 0.921 | 0.698 |
| 4 | 0.533 | **0.900** | 0.624 | ⛔ **0.900 (control `random_atoms`)** | ⛔ **0.900** |
| 5 | 0.682 | **0.939** | ⛔ **0.939 (transformer 80k)** | 0.939 | 0.827 |
| **mean** | **0.677** | ⭐ **0.925** | 0.724 | 0.894 | 0.707 |
⭐⭐ **It generalises, and it fixes exactly A1's two weak seeds** — seed 1 **0.566 → 0.894**, seed 5 **0.682 → 0.939**. ⭐ The CLU reaches the **ceiling on all six seeds**. ⭐ **These margins do NOT rest on failed rivals:** on seeds 3–5 there is **no WORSE@80k**, and the best matched rival (transformer) is **unflagged** on every one.
⛔⛔ **THREE MEASUREMENTS SAY THE STORE IS NOT THE CAUSE:**
1. **`+rel` erases the margin on the new seeds.** CLU − best `+rel`: −0.007, +0.024, +0.165, **+0.006, +0.000, +0.000** ⇒ six-seed mean **+0.031**. Share of margin recovered by handing rivals the columns: **0.97 / 1.00 / n.a.** ⇒ on seeds 3–5 the coding reproduces **essentially the whole margin**.
2. ⛔⛔ **The seed-4 control tie REPEATS ON TEST:** `control: random_atoms` **= CLU = 0.9004**, and per tuple it matches the CLU **exactly on all 16 held-out tuples**. The store is **not load-bearing** there — confirmed, not a VAL fluke.
3. ⚠ **Even where the CLU "wins" on seed 3**, on the four tuples where it beats every matched rival the **controls are just as exact** (70/70, 86/86).
⛔ **Seed 5: CLU 0.939 == best matched rival 0.939 — no margin at all.**
⭐ **The defensible claim: colour-relational request CODING is a large, robust, six-seed improvement in an INPUT REPRESENTATION. ⛔ It is not a memory/store result, and three independent measurements now say so.**

## ⛔⛔ THE WIN ROW DOES NOT REPLICATE ON SIX SEEDS (w21 `maze-t-seed4`, CLOSED 2026-09-24)
**BOTH TEST n-way, CLU A1 unchanged, seeds 0–5. Code `w21/maze-t-seed4 = bd2468f`, 3/3 spent.**
| | s0 | s1 | s2 | s3 | s4 | s5 | **mean (6)** |
|---|---|---|---|---|---|---|---|
| **CLU (A1)** | 0.800 | 0.566 | 0.808 | 0.674 | 0.533 | 0.682 | **0.677** |
| best matched rival | 0.627 | 0.567 | 0.784 | 0.412 | 0.531 | **0.867** | — |
| **CLU − best** | +0.173 | −0.001 | +0.023 | **+0.262** | +0.001 | ⛔ **−0.185** | ⛔ **+0.046** |
⛔ **The 3-seed margin of +0.099 becomes +0.046 over six seeds.**
⛔⛔ **On seed 5 — the EASIEST seed (ceiling 0.939) — the CLU LOSES by 0.185** (0.682 vs mlp-80k 0.867, lstm-80k 0.865). ⭐ Cause is identified per-tuple: the CLU scores **0.000 on two fully-supported class-A tuples** (`(2,0,1,2)` n=90, `(2,0,3,2)` n=80) where **8 trained arms score 1.000**. ⛔ **A CLU-specific miss on reachable data — not a hard-seed effect.**
⚠ **Seed 3's +0.262 is inflated:** its matched MLP did **not converge at either budget**, so that rival field is artificially weak. The 10×-floats transformer (0.714) still beats the CLU (0.674) there.
⭐ **Easy/hard classified by measured CEILING, largest-gap cut at 0.641** (sorted ceilings 0.535, 0.567, 0.714, 0.808, 0.830, 0.939) ⇒ **easy {0,2,3,5}, hard {1,4}**. ⛔ The CLU's own score played no part. ⚠ Sensitivity printed: on **matched-arms-only** ceilings the rule gives hard {1,3,4}.
⛔⛔ **"Easy seeds in the main text" does NOT rescue the row** — the easy set **includes seed 5**, the worst loss. Easy-set CLU mean **0.741**, hard-set **0.549**, original {0,1,2} **0.725**.
⭐ **On hard seeds the CLU IS within 0.003 of the best any arm reaches** (s1 ceiling 0.567 / CLU 0.566; s4 ceiling 0.535 / CLU 0.533) — so the "everything struggles" claim holds **for those two seeds**, ⛔ but see the representational caveat above (CLU-REL-A10 reaches 0.914 on seed 1 VAL).
⚠ **Other rows, all printed:** `value` CLU 0.478 vs arm-ceiling 0.804 · `pair` CLU 0.384 vs 0.550, with learned BLIND-SEQUENTIAL **beating** the CLU on several seeds · `seen` 0.965–0.973 vs 1.000 · **rollout validity 0.000 on all six seeds** (rivals reach 0.267).
⭐ **Deletion holds on all 6 seeds:** store-scoped exactness ≤1.94e-15, deployed-atom Δ = 0, downdate == fitted-without (fraction 1.000), adaptation 0.09–0.88 s and **0 gradient steps** vs the rival's 71–140 s / 8,000 steps / 1,920,000 examples.
⭐ **Zero-fit replay:** every persisted object's param hash matches and every number reproduces bit-exactly — for w19 seeds 0–2, `words-full-test` seeds 0–2, w19 OBS VAL, and its own seeds 3–5.

## ⛔⛔⛔ THE STORE GATE FAILED OUTRIGHT FOR CLU-REL-A10 ON SEED 4 (w21 seed4, 2026-09-24 13:37Z)
**`CLU-REL-A10` BOTH VAL seed 4: a charter-C CONTROL EQUALS THE CLU — both 0.8994.** ⛔ The control is supposed to collapse. It did not; it tied.
⭐ The spoke's own words: **"on this seed the store is NOT load-bearing for CLU-REL-A10"** — and it **printed it** rather than dropping the row.
⛔⛔ **This is now the THIRD independent demonstration that the improved CLU's gain is the RELATIONAL CODING and not the store:**
1. **WORDS-FULL VAL** — matched transformer **+rel 0.941** vs CLU **0.937** ⇒ store contribution **zero**.
2. **BOTH VAL seed 3** — the `+rel` rival **0.9236** *beats* CLU **0.9181**.
3. **BOTH VAL seed 4** — a **control ties the CLU at 0.8994** ⇒ the store is doing no work at all.
⚠ Against **frozen-input** rivals the improved CLU still wins decisively (seed 3: 0.918 vs 0.750/0.711; seed 4: 0.899 vs 0.657/0.563). ⭐ **Both halves must be printed together:** *"beats every rival that is denied the coding; ties or loses to any rival given it; and on one seed its own control matches it."*
⛔ **The paper cannot claim this as a memory/store result.** ⭐ It is an **input-representation** result — which is still a real finding, and it is the one the evidence supports.

## ⛔⛔ THE MATCHED-FLOAT RULE PRODUCES RIVALS THAT CANNOT BE OPTIMISED — measured (w21 seed4, 2026-09-24 12:18Z)
⚠ The w19 report raised this as an **open Advisor question**; it is now a **measurement**.
Matching on read-path floats gives a **954-dim-input** rival a **4-unit hidden layer**. On seed 3 the matched `mlp` **did not converge at EITHER budget**: final loss **0.180 @8k** and **0.205 @80k** — ⛔ **worse with ten times the training** — n-way 0.003 / 0.107, seen 0.483 / 0.846. Seed 4's matched `lstm` @80k shows the same shape (final loss 0.137, n-way 0.107).
⭐ The rivals **ran their full declared budgets**; this is a **capacity / optimisation pathology of the 4-unit matched rival**, and the spoke **printed it in the training table** rather than reporting only the score.
⛔⛔ **CONSEQUENCE: some "matched rival" columns across this programme are not measuring what a matched rival can do — they measure an optimiser failing on an under-width network.** ⚠ It cuts BOTH ways: it can flatter the store (rival too weak to train) and it undercuts any claim that the store beat a fair opponent. ⭐ This belongs in the paper as a **methodological limitation of the matching rule**, ⛔ not a footnote — and it is a rule question for the Head/Advisor, not the Hub's to change.

## ⛔⛔ A CLU-SPECIFIC FAILURE ON REACHABLE DATA — seed 5 (w21 seed4, 2026-09-24 10:47Z)
⛔ **Not a hard-seed story. The opposite.** On seed 5, the CLU scores **0.000** on two **class-A "free"** tuple classes — `(2,0,1,2)` n=90 and `(2,0,3,2)` n=80, **170 rows** — where **8 trained arms reach 1.000**. The data is **reachable**, the rivals get it perfectly, and the CLU gets none of it.
⭐ This is a **CLU-specific miss**, printed as such, and it is part of why seed 5 reads CLU **0.6818** against best-rival-80k **0.8667**.
⚠ Contrast with seed 4, whose hard tuples are class **D** — blocked by a held-out pair (pocket 2 × tint 2) — and are **all reachable by colour relation**. ⇒ ⛔ **Two different failure modes must not be merged in the write-up:** seed 4 = representational blockage (relational coding fixes it); **seed 5 = the CLU failing on data every rival handles.**
⭐ Verified: seed-5 persisted objects replay with hashes OK and `reproduce-all True`, so this is not a scoring artefact.

## ⭐⭐ THE HEAD'S REPORTING ORDER (2026-09-24) — REPORT EVERYTHING, OMIT NOTHING
**"Report all run experiments and results; do not omit any due to pre-reg issues from the past."**
⛔ Anything previously suppressed, downgraded or hidden **because a pre-registration did not cover it** is now **REPORTABLE and must be printed** with its status. ⭐ Sweep the record for these markers and print each with what it is and why it was withheld:
`DECLARED NOT-RUN` · `VOID` / `VOIDED` · `UNMEASURED` · `never a bar` / `not a bar` / `printed, never the label` · `not reported` · `informational only` · `out of brief` · `DECLARED` (as a substitute for a run) · `withdrawn`.
⚠ **Known members of this set (non-exhaustive, check each before quoting):**
- w19: **6 of 9 runs DECLARED NOT-RUN** (WORDS-MAIN / WORDS-FULL / OBS failed VAL) — their **VAL numbers exist and print**; **OBS-X10** diagnostic (*"never a bar"*); **C-512 not re-run**.
- w20: **VOIDED seed-1/2 rung stores** (per-seed dead-mask defect — a REAL defect, not a prereg artefact: report the defect AND what was voided); **LACE/PoE @131k on ladder rungs DECLARED NOT-RUN** (budget, cost measured); **w12comp rivals on val marked "informational only"**; the **VOID_seenmask_all_seen** first val pass.
- w15/w8: **CPU BAR D UNMEASURED, not void** (`N359`); **GPU BAR D MIXED**.
- Everywhere: the **10×-read-floats** and **10×-steps** diagnostic columns (*"never a bar"*), the **planner columns** excluded from ceilings (ORACLE = 1.000 by construction), and **smoke runs** (*"plumbing, not reported"*).
- `words-full-test`: the **σ-sensitivity** it printed but did not label with (COMPARABLE under two other defensible spreads).
- ⚠⚠ **w20 REMOVED `learned_1x` from the printed tables** (2026-09-24 09:14Z) — the 536k and ~3.2M-param rivals matched to the store's four-part MEMORY ledger — on the COMMON rule *"rivals matched on read-path parameters; the inflated-rival row does not exist."* ⭐ **Kept in artefacts.** ⛔ Under the Head's omit-nothing order this must be **printed with its status and the reason**, not silently absent: it is the strongest-capacity rival anyone fitted.
- ⚠ **w20 deletion, `red ∧ sphere`: NOT significant on 3 of 12 legs** (vs mlp s1 +2.49 [−0.94, 6.02], s2 +2.60 [−0.50, 5.67]; vs lstm s2 +2.48 [−0.87, 5.82]). ⭐ `value`/`item` legs are all clear of 0 (smallest +6.95). ⛔ Print the CIs, not just the point estimates.
- ⭐ **w20 (c) refusal measurement:** `value` scope — refusal **1.000** on red-corner requests, false refusal **0.000**. `conj` — refusal **undefined** (no rows-3–4 corner request is red∧sphere), false refusal 0.000. `item` — N/A. ⛔ "Undefined" is printed as undefined, never as a zero.
- ⭐ **`clu-improve` self-correction:** candidates tried = **68** (+ A1-repro), **57 gate-passers** — its own PREREG/ledger rows say 69 and were **not edited** (hashed / append-only). ⛔ Print 68 with the correction noted.
⭐ **Print the number, its status, and why it was withheld.** ⛔ A result withheld for a reason that no longer exists is exactly what the Head has now banned.

## ✅ THE SIX-SEED MEAN IS RULED AND CORRECTED (Head, 2026-09-24)
⛔ **§7's `+0.128` was WRONG and is corrected in the wave plan to `+0.046`**, with the recomputation recorded inline. ⭐ Per-seed CLU−best-matched-rival = **+0.173 / −0.001 / +0.023 / +0.262 / +0.001 / −0.185**, mean **+0.0457** by both routes, from `w19-maze-t-symmetric/test/BOTH_test_seed{0,1,2}.json` and `w21-maze-t-seed4/test/BOTH_test_seed{3,4,5}.json`. ⚠ For reference the **improved** CLU's six-seed mean is **+0.201** — ⛔ do not conflate them.
⭐ `w22-draft-rulings` launched on batmac 2026-09-24 with the corrected figure and instructions to re-verify it against the JSONs itself.

## ⛔⛔ A SECOND HUB NOW OWNS THE PAPER (scoped 2026-09-24) — `paper_lead_hub`
Its brief is `.claude/paper-lead-hub.md`. ⭐ **It is a PEER, not a successor.**
**IT owns:** paper waves **PW1+** only — plot regenerations, `.tex` edits, bib/cite tags — **each scoped by the Head before it starts.**
**YOU KEEP EVERYTHING ALREADY GIVEN TO YOU** (Head, 2026-09-24): the compute waves · `w21-ladder-rivals-32k` · `registry-fold` / `seeds-and-tables` · **`draft-rulings`** · the registry + claims matrix · `SCORED-LEDGER.md` · `w21-` and `w22-advisor-report.md`. ⛔ **Hand nothing over.**
⛔⛔ **THE `.tex` LOCK IS SHARED, NOT TRANSFERRED.** Both Hubs have draft work. ⭐ **The Head sequences us.** Before launching any draft spoke, check that **no** draft-touching session is live — `w*` or `pw*` — and abort if one is.
⛔⛔ **THE WATCHER IS THE OTHER COLLISION.** `/tmp/watch3.sh` stays **yours alone**; the paper Hub uses **`/tmp/pw-watch.sh`** with its own detectors. ⚠ If you ever see a watcher you did not arm, ⛔ **do not `pkill`** — check `ListAgents` first.
⭐ **Session naming:** yours `w<N>-<slug>`, theirs `pw<N>-<slug>`.

## ⛔ OPEN FIGURE DEFECT — CMAZE-T panels hide the legality fact (Head, 2026-09-24)
Full write-up + worked example: **`.claude/outputs/FIGURE-DEFECT-cmazet-legality.md`**. ⭐ **No number affected; the judge is correct.** The panel draws the resulting STATE while the judge scores the FACT VECTOR, so **legality is invisible whenever displacement = 0** — **3,740** arm-pairs render identically with opposite marks. ⭐ Fix = show predicted legality per panel; test against **seed 2 item 482** (CLU `[1,0,0,3]` CROSS vs mlp `[0,0,0,3]` TICK, identical pictures). ⚠ ⛔ Discard the Hub's first diagnosis blaming the undrawn pocket-next field — it was wrong.

## ⭐⭐ W21 COMPUTE IS COMPLETE — `ladder-rivals-32k` CLOSED (2026-09-25, `w21/ladder-rivals-32k = 3cd0110`)
**MF-2 is real on paper and IMMATERIAL in measurement.** Rivals fitted on the store's **own 32,768 content rows** land within **±1.5 novel/corner** of the 131,072-row set at every rung, and the difference **goes both ways**.
K\*=32 minus the strongest rival, mean over rungs: **A (32,256 rows, 5,040 steps) +17.53 · B (131,072 rows) +9.97 · C (32,768 rows) +10.28 · C step-matched +10.75.**
⭐ **Every bootstrap CI95 in sets B, C and C-sm has a lower bound above 0** (lowest **+1.98**, n=2 seed 2 vs the step-matched mlp).
⭐⭐ **It confirms w20 independently: at matched steps the POOL SIZE barely moves these rivals; the STEP BUDGET is what moves them.** ⛔ The ladder shrink from set A to B/C is real and **survives the data match**.
⭐ Pool identity proven on **15/15** rung×seed legs: `n_fit_rows = n_store_train = 32768`, and the rung pool is identical to `w12-composition-ladder` partitions — so set C read **exactly the store's content rows, which are also `learned_ref_ownpool`'s rows**.
⛔ **forgis-14 is now IDLE. All W21 compute is done.**

## ⭐⭐⭐ WAVE 21 IS CLOSED (2026-09-25) — `w21-advisor-report.md` WRITTEN
**All 7 units harvested, reaped, landed. 12 scored runs. Every closure criterion met and checked, not remembered.**
`words-full-test 6a3db43` · `maze-t-clu-improve f491a04` · `maze-t-seed4 bd2468f` · `rel-newseeds 31eeadd` · `ladder-rivals-32k 3cd0110` · `figures-redraw cc17f15` · `registry-fold f1ca948`. ⛔ No w21 session remains on either box.
**Budgets:** `maze-t-symmetric` **9/9 EXHAUSTED** · `maze-t-seed4` **3/3** · `maze-t-rel-newseeds` **3/3**.
⭐⭐ **The headline for the Advisor: the maze contribution is an INPUT-REPRESENTATION result, not a memory result** — `+rel` recovers 97–100 % of the margin, a control ties the CLU on seed 4 (0.9004, exact on all 16 tuples) and a rival ties it on seed 5.
⚠⚠ **The finding with the widest blast radius: a single 2,560-step rival fit varies by up to ±4.1 novel/corner on the PRNG draw alone** (ladder X1: rows, steps, width, params, trainer, schedule all identical) — against a headline margin of ≈ +7. ⛔ **Fit-level variance must print beside any single-fit margin.**

## ⭐⭐⭐ WAVE 22 IS CLOSED (2026-09-25) — `w22-advisor-report.md` WRITTEN
**All 7 spoke-passes harvested, reaped, landed. ZERO scored runs.** `paper-visuals b871a4d` · `figures-redraw2 102d1de` · `figures3 cb5e0ac` · `draft-rulings e9370a2` · `registry6 8db7d33` · `seeds5 3bb83da`. ⛔ No w21/w22 session on either box; both idle.
⭐⭐ **Three things this wave achieved:** the paper's headline is now the **measured ≈ +7** (the +39…+49 comparator numbers proven gone from rendered text, **with a positive control**) · the Head's **figure defect is fixed (33,412 → 0)** · the **w15 verification gap is CLOSED, 199/199**, after two passes had declined it.
⭐ Draft: **48/49 pp** (was 51/52), **3 placeholders** left.
⛔⛔ **THE ONE BLOCKER FOR A PLACEHOLDER-FREE DRAFT:** the two **w16 CMAZE-T per-arm** placeholders **cannot be built from any artefact** — w16 persisted no arm object and no per-transition prediction. ⭐ **Cut the panels, or re-run w16.** The Head must rule.
⚠ **Two Hub errors on the record, both caught by spokes:** the figure defect was first misdiagnosed as the undrawn `pocket-next` field (it was **legality**), then sized at 3,740 when the true figure was **33,412**.

## ⭐⭐⭐ WAVE 23 IS CLOSED (2026-09-25) — `w23-advisor-report.md` WRITTEN
Both spokes harvested, reaped, landed: `w23/seed5-diagnosis 677e4e3` · `w23/fold 6199b40`. ⛔ Zero scored runs; nothing written to `w17-paper/`, `figures/` or `paper-visuals/`. Both boxes idle.
⭐⭐ **Q1: the seed-5 misses are CONTENT — and NOT a hold-out effect.** The store fails that outcome code on **211/317 of its own FIT rows**. Capacity, not generalisation.
⭐⭐ **Q2: the "eq-columns all-zero" premise is FALSE.** They are colour-**anonymous** and fire normally; what is absent is every column that **names** the colour. ⇒ **relations reach the store, identity does not.**
⚠⚠⚠ **THREE CORRECTIONS TO THE HUB'S OWN w22 REPORT:** (1) the w15 closure covers **212 of 2,293** `% SRC` tokens — the three w15 spokes only, ⛔ **not the paper**; (2) the drafts print **27.135×**, not 27.13 or 27.14; (3) ⛔⛔ **the negative-test defect is a CLASS** — an unmutated build into a missing dir exits 1, and **478 of 507 manifest cells are never read**, so many tamper tests could not have caught anything.

## WHAT YOU OWE, in order
1. ⭐ **Arm a watcher before anything else.** Nothing else matters if you cannot be woken.
2. ✅ **`w19-advisor-report.md` WRITTEN** (⚠ and CORRECTED 2026-09-24 — its store-gate claim was wrong; see `N371`). ✅ **`w18-advisor-report.md` WRITTEN.** ✅ **`w20-advisor-report.md` WRITTEN 2026-09-24** (headline: data-matched margin **+7.03** vs the draft's +39.7…+49.1 — and the cause is **training SCHEDULE, not data**). ⛔ **STILL OWED: `w21-advisor-report.md`.** ⭐ w21's re-prints the w19 condition table **under the label rule** beside the new rows.
3. **The w18 re-passes.** ✅ `registry-fold` pass 4 DONE, harvested, reaped. ⛔ `seeds-and-tables` **pass 4 still RUNNING** — harvest it.
   ✅⚠ **Both previously-flagged findings are ALREADY ACTIONED — do NOT re-raise or re-launch them.** `seeds-and-tables` **pass 3** (20:31:15Z) fixed MF2 `23.5117 → 23.5121` (now at `v2:1726` / `style:1713`; the `:2262` block moved to `CORRECTIONS-ABSORBED.tex`) **and** added the K5 inference note to the `tab:k5` caption. ⛔ An earlier version of this file called them unactioned — that was read off a **stale** pass-2 `FINDINGS.md`. ⭐ **Re-pull a multi-pass spoke's LOG before believing any "outstanding" item.**
   ⭐ Pass 3 also put **7 w19 qualifications** of the w16 CMAZE-T margin into both drafts (abstract, intro, §maze-wm, summary, limitations, `tab:cmazet` caption + n-way row label), replacing every "symmetric-input test is pending". ⛔ **No w19 number entered the draft** — those are still `\PLACEHOLDER{}`.
   ⭐⭐ **ONE FOLD IS STILL OWED, and it is ONE spoke not two — `seeds-and-tables` PASS 5, launched when w20 lands.** Scope it to both at once:
   - the **w20 cells** (the 4 named `tab:del` placeholders pass 4 created), and
   - the **w15 `bar-d-score` / `coverage-control` / `lace-poe-rows` fold into `TABLES.md`** — these landed in evidence and `placeholder-fill` already wrote them into the draft, but pass 3 and pass 4 both declared them out of brief, so **the draft carries numbers that TABLES.md does not yet verify.**
   ⚠ **The placeholder count MOVED in pass 4:** 12 pre-existing + **4 NEW named** (`tab:del` (b) ×3, (c) value-red). ⛔ The old "11 (10 w19 + 1 w20)" figure is stale — re-count before quoting, and quote it with a timestamp.
3b. ⭐⭐⭐ **M3 IS NOW MEASURED — the evidence the Head's held ruling was waiting for (w20, 2026-09-24 01:22Z).** Under the Head's own AMENDMENT-2 definition (address book **REFIT**, not carried), deleting `red ∧ sphere` (5,511 rows, seed 0):
- (a) store exactness **5.38e-07**; (b) book-statistic downdate **exact** (E 1.9e-11, W 2.9e-9, rendered 3.3e-8) — ⭐ the *mechanism* works.
- ⛔⛔ **(d) FAILS.** 109 survivors go **47.495 → 41.367 (Δ −6.128)**, matching the refit-without pipeline to **+0.000**. The damage is concentrated exactly where M3 predicted: **20 red + non-sphere corners fall Δ −32.1** (min −36, max −20), against −1.47 for non-red+sphere and **+0.51** for non-red+non-sphere.
- ⭐⭐ **The spoke's own conclusion: "T7's 'unchanged' row held ONLY because the book was CARRIED."** ⛔ With the book refit, deleting `red ∧ sphere` **removes red composition with other shapes**. That is M3's objection, measured, not argued.
⭐⭐ **IT REPLICATES, AND THE RIVALS DO NOT SUFFER IT (04:22Z).** Seed 2 `conjunction_live`: **49.119 → 42.817 (Δ −6.303)**, again = refit-without. So the collapse is **−6.13 (seed 0) and −6.30 (seed 2)**.
⛔⛔ **The comparison that hurts most: under `red ∧ sphere` the RIVALS lose < 1 novel/corner on the survivors; the store loses 6.13.** The rivals' crude retrain-without is **cleaner collateral-wise** than the store's exact deletion.
⭐ **What still favours the store:** after deletion it is **ahead of every rival's retrain-without on survivor corners** — seed 0, value +11.74/+13.67/+23.07/+30.93 (mlp/lstm/transformer/cont), `red∧sphere` +5.26/+8.30/+16.64/+23.67, item +11.60/+14.23/+22.42/+28.34 — and does it **data-free in under a second** against **548–1,563 s** and 120–130 k rows re-read. ⚠ Rival retrains are at the DECLARED schedule, so the **mlp/cont_mlp P1 under-training flags travel with these numbers.**
⚠ Note this is the OPPOSITE failure from w15's caveat. w15 found deletion **under**-reaches (deleted corners still compose red, book carried); this finds that once the book is refit it **over**-reaches (collateral damage far outside the deleted scope). ⛔ Both readings must travel together; neither alone supports a clean "exact, scoped deletion" claim.
⭐⭐ **THE INSTABILITY CONTROL IS IN, AND IT CLOSES THE ESCAPE HATCH (02:16Z).** Random **item deletion (1 %, 1,311 rows, seed 2)** moves the survivors by **Δ −0.009** (49.119 → 49.110), again matching refit-without. ⇒ **the book refit is STABLE under random deletion**, so the `red ∧ sphere` collapse is **CONTENT, not ALS instability.** ⛔ The obvious rebuttal — "refitting the address book is just numerically unstable" — is measured and dead. **The RULING remains the Head's.**

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
