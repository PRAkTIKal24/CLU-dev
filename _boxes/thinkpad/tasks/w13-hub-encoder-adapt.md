> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **Evidence base: `/c/chlu/evidence/outputs/`** · matrix + registry at `/c/chlu/evidence/`.
>   ⭐ Current state = **w9-, w10- and w12-** reports there. ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** —
>   frozen, older, every number stale. ⛔ `/c/chlu/evidence/` is READ-ONLY.
> - ⭐ **Write ONLY to `/c/chlu/outputs/w13-w13-hub-encoder-adapt/`** and your own worktree (branch `w13/w13-hub-encoder-adapt`).
> - ⭐ **START_UTC.txt first · LOG.md as you go · END_UTC.txt last.**
> - ⭐ **Assert `chlu.__file__` resolves inside YOUR worktree** before any number — the shared
>   site-packages has an editable `.pth` to the Head's checkout; if it wins, every number is void.
> - ⭐ **Poll background jobs to completion INSIDE a turn.** ⛔ Never end a turn with detached work
>   running — nothing will re-invoke you. ⚠ A w12 hub stalled exactly this way at a rung boundary.
> - ⛔⛔ **Never a box-wide kill.** Capture your PIDs at launch; kill only those. ⚠ **Other agents are
>   running on this box** — never touch their sessions, worktrees, output dirs or processes.
> - ⭐ **Commit and bundle your branch WHOLE:** `git bundle create <f> w13/w13-hub-encoder-adapt` — ⛔ that branch
>   only, **never `--all`**. Verify, and check `list-heads` shows exactly ONE ref.
> - ⭐ **A published file is not a finished file.** ⚠ Three times this programme, a READY json appeared
>   while its author was still writing. If you gate on someone's artefact, check it is COMPLETE
>   (non-null fields, expected counts), not merely present.
> - ⭐ **Anything not run is a DECLARED NOT-RUN with a reason** — ⛔ never a zero, a blank or silence.
> - ⭐ **Every number names the file it came from.** ⛔ No sub-agents for long work.

# w13 hub/encoder-adapt — what does LIGHT encoder adaptation actually buy?

- **Agent:** autonomous hub. ⭐⭐ **TARGET `encoder-adapt`, 0/3 — ITS OWN BUDGET.**
  ⛔ **It does NOT draw on `encoder-loop`.** (Head 2026-09-21; the plan's heading still says
  *"target encoder-loop, 2 left"* — ⛔ **that line is STALE. The ledger is authoritative.**)
- **Output:** `/c/chlu/outputs/w13-encoder-adapt/`. **Branch:** `w13/encoder-adapt`.
- **CPU default;** GPU only if the timed budget demands it. ⭐ **`kill -9` resumability must be PROVED
  before any spend** — checkpoint on a wall-clock timer, verify a config fingerprint before resuming.

## THE ONE QUESTION
**The frozen harness is the primary object. Light adaptation is optional — what does it buy?**

## THE ARMS — equal steps and images, one bit-identical warm start per seed, 3 seeds
| arm | encoder | role |
|---|---|---|
| **`frozen`** | frozen | ⭐ **REFERENCE** |
| **`adapter`** | small low-rank / last-block adapter, **encoder weights frozen** | treatment |
| **`anchor`** | full unfreeze **+ penalty on distance from the frozen weights/features** | treatment |

**Loss for both treatments = store read error on fitting rows + the SWAP LOSS**, reconstruction kept
as in w8. ⭐ The swap loss is in because w8 measured it repairing what unfreezing breaks — positive on
all 3 GPU seeds (+0.0100 / +0.0129 / +0.0075) at 10× steps and 4× data.

**Store design FIXED at K\* = 32** (wave 12, `CROSSOVER-val.json`; rank/width 0.9844, selection rule
frozen and timestamped before any val number existed).

## ⭐⭐ THE BAR — it is the MARGIN, and it starts NEGATIVE
Every baseline (MLP, LSTM, transformer, continuous MLP, LACE) is **refit on each arm's features**.
> **BAR: `single design − best learned baseline` must GROW from `frozen` to `adapted`.**

⚠ **Read this before you interpret anything.** Wave 12 measured that margin as **NEGATIVE** at K=5 —
the store loses to a same-bytes MLP by **−0.054 to −0.077** of ceiling at n = 2, 4, 5, 6. ⭐ So the
question here is **"does adaptation close a deficit?"**, ⛔ not "does it extend a lead." **A margin
that grows and stays negative is a real and reportable result** — say so in those words.
⛔⛔ **A feature gain shared by ALL arms is a finding about the ENCODER, not about the store**, and it
is reported as such. That is the trap this bar exists to catch.

⭐ **Factor probe printed as a MARGIN over random-init.** ⛔ No absolute probe number is a claim —
w8 measured a random-init encoder probing Shapes3D at ~0.97, and `alone`'s margin went NEGATIVE on
2 of 3 seeds. ⚠ **A single training run of this encoder is not reliable** — w9's harness independently
found the same instability. **Report per-seed; ⛔ never pool.**

## ⛔ NON-NEGOTIABLE
- ⭐ **Write your ledger row in `/c/chlu/SCORED-LEDGER.md` BEFORE each scored run** — your OWN row
  under target `encoder-adapt`. ⛔ Another hub's row does not authorise you.
- ⭐ **Equal steps AND equal images across arms**, verified and printed. ⛔ If `anchor` needs more
  wall clock, print a seconds-matched control too and say which decides.
- ⭐ **All 3 seeds or none.** ⛔ The best is never reported alone.
- ⛔ **Never "9/9 seeds"** — report seed count and configuration count separately.
- ⛔ **Never "no changes after deletion"** — that phrasing is banned programme-wide.

## ⛔ WHAT WOULD PROVE THIS HUB WRONG
- **The margin does not grow on ≥2 of 3 seeds** ⇒ ⭐ adaptation does not buy a store result. Report it;
  that is the secondary paragraph either way.
- **Every arm improves together** ⇒ ⭐ it is an encoder gain, ⛔ not a store gain — say which.
- **`anchor` diverges from the frozen features** ⇒ the penalty is mis-scaled; report the distance,
  ⛔ do not tune until it looks better.
