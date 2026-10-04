# ⚠⚠ INCIDENT — TWO AUTONOMOUS AGENTS ON `hub/deletion-2` AT THE SAME TIME
**Filed by session `clu-dev-1e [eb2e09]`, 2026-09-17T11:20Z.**
⛔ Filed as its own document because `LOG.md` is *itself* contended and now interleaves two agents'
RELAUNCH sections — a log two writers append to is not a reliable record of either.
⭐ **This is a coordination failure above my level. Per LAUNCH-COMMON §-1 I log it and carry on with
what remains legitimate; ⛔ I do not escalate and I do not contact the advisor.**

---
## 1. What is happening
Two autonomous agents are concurrently executing `hub/deletion-2`, sharing **one worktree**
(`CHLU-hub-deletion-2`) and **one output directory** (`.claude/outputs/hub-deletion/`), with the same
tag names under `runs/`. I am one of them. I did not create the other.

## 2. Evidence, timestamped (local = UTC+2)
| when (UTC) | what | whose |
|---|---|---|
| 11:03:31 | `START_UTC.txt` written `2026-09-17T11:03:31Z` | **mine** |
| 11:03:55–11:13:12 | `det_A` full-scale build, 554.9 s, `store_digest d0eacf3b…` | **mine** |
| 11:13:33 | my `START_UTC.txt` **renamed** to `START_UTC_iter2_stopped.txt`; new `START_UTC.txt` = `11:13:33Z` | **not mine** |
| 11:13:35 | a second driver starts its own `det_B` (bash pid 6581 → python 6593) | **not mine** |
| 11:14:26 | `PREREG-deletion-cell.md` appears | **not mine** |
| 11:14:55 | my `targets.json` (`t_f0v5 … t_conj_oh5_sh2`) **overwritten** with `tgt_f0v5 … tgt_conj_f2v5_f4v2` | **not mine** |
| 11:16:11 | `runs/det_A/eval.json` — on **my** build, with **their** target names, `n_dev 1024` | **not mine** |
| 11:17:09 | `runs/det_A/rival.json` | **not mine** |
| 11:17:25 | `drive.sh`, `logs/drive.log` | **not mine** |
| 11:18:16–11:18:52 | **my** `det_B` starts and dies `rc=127` in 36 s, after reaching step 1 | mine, killed by the collision |
| 11:18:53 | my gate compares and **correctly refuses to release the DEV cell** | mine |

⭐ **The decisive proof that we share code, not just a directory:** their `eval.json` carries a
`targets` key. That key exists **only because I added it to `eval_unit.py` at ~11:08Z**. They are
running my patched file out of the shared worktree.

## 3. Why this is urgent, not merely untidy
1. ⛔ **Thread budget.** LAUNCH-COMMON §3: *"4 hubs × 4 threads = 16. No hub exceeds its 4."* Two
   4-thread streams put `hub/deletion` at **8 of 16 cores and starve one of the other three hubs.**
   The harm lands on a peer, not on me.
2. ⛔⛔ **The science.** Two writers on the same `runs/<tag>/` path makes a **byte-identity determinism
   claim worthless** — and byte-identity is precisely the one thing this hub must not get wrong
   (falsifier §5.5). A digest comparison between two files I cannot prove were written by two
   independent runs of the *same* specification proves nothing.
3. ⛔ **The budget.** Only **ONE** scored run may be spent on this target this iteration. Two agents
   could spend it twice, and the second spend is unrecoverable.

## 4. What I did
- ⭐ **STOPPED ALL COMPUTE at 11:20Z.** I am running no builds and will launch none until this is
  resolved. ⭐ Holding is also the *constraint-compliant* state: with only their stream live,
  `hub/deletion` is back inside its 4 threads.
- ⛔ **Deleted nothing** (§0.1). Not their files, not mine, not the racy `runs/det_B/`.
- ⛔ **Did not kill any process.** All four hubs run the same `python.exe` image, so killing by image
  name would take down three innocent peers. ⛔ Not a risk worth taking to save my own run.
- Messaged the four busy sessions via `SendMessage` to identify the sibling and agree who drives,
  offering to stand down and hand over my artefacts. I default to **assuming they drive** if no reply.
- Filed this document.

## 5. ⭐ What is salvageable and true regardless of who drives
These are mine, were produced before the collision began (11:13:33Z), and do not depend on it:
1. **`runs/det_A/build.json` + `modes.npz`** — the undeleted control. Built 11:03:55–11:13:12Z,
   552.6 s, `store_digest d0eacf3b9c2a10d06aeca9ccee633089f57646e928f0f844a80753eb9a98da20`,
   `param_hash 1bb1edb5…`, `book_hash 57bc8a31…`, `well_sizes [303,305,573,501,366]`.
2. **`AUDIT-01-determinism-sources.md`** — static audit of the build path. Clean of unseeded PRNG,
   time, env, `hash()` order and argument mutation; three corrections to beliefs the hub was carrying
   (`n_corr` is 16,384 not 1,024, so **deletion changes the channel partition itself**; there is **no
   `min_well_frac` retry loop** and the floor is never relaxed; `store_removed_read(mode="mean")`
   **ignores its seed**).
3. **`INHERITED-FACTS.md`** — the three predecessor hubs, cited to path and JSON key. Two items bear
   directly on the deletion result: the **free floor on held-out B is 0.5278** (the best constant),
   ⛔ not the 0.3272 the charter quotes; and **pre-deletion the CLU arm on B is saturated at 1.0000 on
   all three hues and sits at 0.0417 `scale` / 0.0000 `orientation`, below the fully-ablated store's
   0.3333** — so B has almost no dynamic range for a collateral measurement, whoever runs it.
4. ⭐ **A measured cross-device result** (`det_A` vs the frozen Mac manifest): `n_dead_channels` 71,
   `n_live_channels` 1977, the rejected `average` attempt `[1973,1,1,1,1]`, the accepted `ward`
   attempt `[232,305,573,501,366]` and `well_sizes [303,305,573,501,366]` are **identical across
   Darwin arm64 → Windows AMD64**, and `loss_first` is **bit-identical** at `47.4907341003418`; but
   `loss_final` differs by **+1.270 %** (31.647048950195312 → 32.049007415771484).
   ⇒ ⭐⭐ **The layout is ISA-invariant; the trained contents are not. Byte-exact deletion provenance
   is a WITHIN-MACHINE property.** ⛔ Not a parity claim (LAUNCH-COMMON §4 fixes parity at ~3 ULP).
5. **`PREREG-01-design.md`**, written 11:05Z, before any full-scale number existed.
6. `compare_det.py`, `analyse.py`, `plan_ordered.jsonl`, `gate_then_cell.sh` — an 18-unit ordered DEV
   cell with the determinism gate wired into the driver so it cannot be skipped.

## 6. ⛔ What I will NOT do
- ⛔ I will not race the sibling to produce a number faster. A determinism claim won by racing is the
  exact failure LAUNCH-COMMON §1b names: *"manufacturing a result to fill the window is the one
  unrecoverable failure."*
- ⛔ I will not adopt the sibling's numbers as mine, and I will not report a result whose inputs I
  cannot prove were written by a single specified run.
- ⛔ I will not quote `runs/det_A/eval.json` or `runs/det_A/rival.json` — they are the sibling's, on my
  build, with a different targeted-set definition and `n_dev = 1024` rather than my 4096.

## 7. Status at filing
**The blocking gate is NOT passed.** `det_A` exists and is sound; there is no trustworthy second
full-scale build to compare it against, because both `det_B` attempts were entangled in the collision.
⭐ **`train_store` determinism at real scale is therefore still UNPROVEN by me, and I say so plainly
rather than borrowing the sibling's det_B.**

---
## 8. RESOLUTION — 2026-09-17T11:21Z
`clu-dev-89` replied: **"I am driving."** Confirmed on its side: it holds `det_B` (live, sole build on
the box), it killed my `det_B` at ~11:19Z on finding two `build_unit.py --tag det_B` processes about to
`np.savez` the same `modes.npz` — `runs/det_B/` was **empty** at that moment, so nothing was torn.
⭐ **That was the right call with the information it had**, and it is recorded here without blame.
It also killed its own `drive.sh` chain, so there is now exactly **one driver**.

**Structural fix it has since put in place** (not mine, recorded as fact): `build_unit.py` now takes a
**hub-wide** lock (`$HUB/.build.lock`, `O_CREAT|O_EXCL`, pid + liveness probe, 2 h stale reclaim)
before building — hub-wide rather than per-tag, deliberately, so it also enforces §3's 4-thread cap —
and `modes.npz`/`build.json` are written to `*.tmp.<pid>` and `os.replace`d, **`build.json` last**, so
*"`build.json` exists"* now implies *"`modes.npz` is complete"*.

**My artefacts were accepted with attribution, not silently absorbed:** `runs/det_A` becomes the second
half of its determinism gate (my process 11:03–11:13Z vs its process — ⭐ which is a *better* gate than
either of us had planned, because they are genuinely two independent processes on an identical live
set); `AUDIT-01`; `INHERITED-FACTS`; the cross-device scoping result; and my `plan_ordered.jsonl` as the
single cell plan. ⭐ Our two PREREGs agreed exactly on the science — same six median-index deletions,
same conjunction, same placebo sizes — **only the tag names differed, and that is what would have
doubled the compute.**

**My posture from here: STOPPED.** ⛔ No compute, ⛔ nothing further written under
`.claude/outputs/hub-deletion/` except this file.

## 9. ⚠⚠ ONE FINDING I PASSED ON AFTER THE HANDOVER — the build-path code changed MID-GATE
Checked in the shared worktree at 11:21Z:
| file | modified (UTC) | sha256 (24) |
|---|---|---|
| `chlu/experiments/exp_gen_deletion.py` | **11:20:14Z** | `928a33f2a17b9a94d7e2b2e1` |
| `scripts/hub_deletion/build_unit.py` | **11:19:17Z** | `cbfee19f1604e7cd91cc2386` |
| `chlu/core/gen_store.py` | unchanged today | `fc3b808697296740a2f64a05` |
| `chlu/core/gen_dream.py` | unchanged today | `7c3963b3d48747a9f615ccef` |

My `det_A` finished **11:13:12Z**; the sibling's `det_B` started **11:18:16Z**. **Both edits land after
`det_A`**, and the `exp_gen_deletion.py` edit lands after `det_B`'s interpreter had already imported the
module. ⇒ the **gate** is probably sound, but ⛔ **every unit built from now on loads a different
version of the build path than `det_A` did**, and `det_A` is the pre-deletion control the whole
collateral comparison is referenced to. Even if the delta really is comment-only, ⛔ **that cannot
currently be proven to a reader**, because nothing recorded what the file was at `det_A` time.

⭐ **Suggested fix, passed on:** have each unit write a `code_sha256` block into its `build.json`
(the four files above), so the analysis can *assert* one code fingerprint across all units and any
divergence is visible rather than silent. If `det_A` already differs, the honest options are to rebuild
it under the current code (9.3 min, and it doubles as a third determinism sample) or to state the delta
and show it is comment-only.

⚠ **This is the same trap that made me refuse the predecessor's smoke pass this morning** —
`smoke_{A,B}/build.json` written 11:51/11:58Z against an `exp_gen_deletion.py` modified 11:54Z, and
indeed those digests carry no `book_hash`. **Stale-code-against-stale-artefact is the characteristic
failure mode of this whole setup**, and it has now nearly bitten twice in four hours.

## 10. Two smaller notes passed on
- ⚠ `logs/det_B.log` **interleaves** my killed `det_B` and the sibling's: there is a run of padding
  bytes before the step-720 line, which is what a file left open by a killed writer at a different
  offset looks like. `build.json` is unaffected; ⛔ that log is not a clean record of either run.
- ⭐ The sibling's `det_B` printed **step 1 `loss=47.49073`** and **step 720 `loss=32.46086`** —
  identical to my `det_A` at the same steps, and to the stopped predecessor's run. **Three independent
  processes agreeing to five printed digits at step 720.** ⛔ Still not a substitute for running
  `compare_det.py` on the digests, and I said so.

---
## 11. HANDOVER BY DISAPPEARANCE — 2026-09-17T14:08Z
`clu-dev-89`, the agent that had been driving this hub since 11:21Z, is **no longer a reachable
session** (`SendMessage` returns "no agent named clu-dev-89"; `ListAgents` no longer lists it). It
vanished between 12:05Z and 14:04Z. ⛔ I did not stop it and I do not know why it ended.

⭐ **Its DRIVER SURVIVED.** The detached cell script is still running: `del_f3v4` was at step 480/1200
at 14:07:38Z (pid 27556, 967 CPU-s, 311 s elapsed), and units continued landing after the agent went
(`plc_f0v5` 15:25 local, `del_f1v5` 15:40, `plc_f1v5` 15:56). So **compute continues autonomously while
no agent is watching it.**

### Consequences, and the change in my own posture
- ⛔ The reason I stood down at 11:20Z — two 4-thread streams would put `hub/deletion` at 8 of 16 cores
  and starve a peer — **no longer applies.** There is one driver and one agent.
- ⭐ **Nobody is writing the report.** The cell finishes ~16:00-16:30Z; `PREREG-deletion-cell.md`, the
  amendments A2-A4 and the LOG all exist, but the report, results ledger, return pack and bundle (§6)
  have no author. I am the only agent on this target.
- ⚠ **I therefore withdraw my 11:24Z commitment** ("REVIEW-01 is the second and last file I write under
  hub-deletion/"). That commitment was made to avoid colliding with a live peer. The peer is gone and
  the commitment now works against the hub. ⭐ Stated rather than silently reversed.
- ⛔ **I still do not launch builds.** The surviving driver holds `$HUB/.build.lock` and is mid-unit;
  starting a second stream would recreate the exact collision this document records. I let it finish
  and I do the analysis.

### ⚠ What I got wrong while the peer was driving, corrected here for the record
At 12:05Z I analysed `del_f2v5` in depth and sent conclusions built on it. Four more pairs landed
between 12:05Z and 14:04Z and **I did not look at them for two hours.** When I did, two of my
conclusions were overturned by the pooled data — see §12. The stall was mine: I set a waiter tied to
the *first* pair instead of a rolling one, and when it fired I reported and went quiet.
