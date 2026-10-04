> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **Evidence base: `/c/chlu/evidence/outputs/`** · claims matrix + negatives registry at
>   `/c/chlu/evidence/`. ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — it is a FROZEN, OLDER snapshot
>   and every number in it is stale. Reading it silently produces stale work.
> - ⭐ **Write ONLY to `/c/chlu/outputs/w12-w12-spoke-clue-theorist/`** and your own worktree. ⛔ Never another
>   spoke's tree, ⛔ never the Head's profile, ⛔ never create a `.claude/` dir in your worktree.
> - ⭐ **Write `START_UTC.txt` first, keep `LOG.md` as you go, `END_UTC.txt` last.**
> - ⭐ **Assert `chlu.__file__` resolves inside YOUR worktree** before any number. The shared
>   site-packages has an editable `.pth` to the Head's checkout; if it wins, your numbers are void.
> - ⭐ **Poll background jobs to completion INSIDE a turn.** ⛔ Never end a turn with detached work
>   running — nothing will re-invoke you.
> - ⛔⛔ **Never a box-wide kill.** Capture your PIDs at launch and kill only those. A hub once
>   destroyed two other agents' runs this way.
> - ⭐ **Commit and bundle your branch WHOLE before you finish:** `git bundle create <f> <your
>   branch>` — ⛔ **that branch only, never `--all`** (an `--all` bundle mixes other agents' work in).
>   Verify it, and check `list-heads` shows exactly one ref.
> - ⭐ **Anything not run is a DECLARED NOT-RUN with a reason** — ⛔ never a zero, a blank, or silence.
> - ⭐ **A result exists only if it is on disk.** Every number you report names the file it came from.
> - ⛔ **No sub-agents for long work.** If the job is too big, say so.

# w12 spoke/clue-theorist — is the subspace store the CLU with settling switched off?

- **Agent:** `physics-theorist`.
- **Output:** `/c/chlu/outputs/w12-clue-theorist/CLUE.md` + LOG.md + `checks/`.
- **Owns:** its output dir + branch `w12/clue-theorist`. ⛔ **No production `chlu/` code.**
- **Scored runs:** ⛔ none. Small numerical sanity checks only.

## THE ONE JOB
Write **the tightest HONEST statement** that the subspace store is **the CLU with settling switched
off (the zero-dynamics limit)** — and what the shorts' retention law (**CRR**) predicts when settling
is switched **on**. ⭐ **Deliverable = the paragraph the paper can carry, PLUS what it may NOT say.**

## ⭐ THE MEASUREMENT THAT ALREADY BEARS ON THIS — start here, do not rediscover it
Wave 6 established that an **in-well settle on an orthonormal basis is an ALGEBRAIC IDENTITY**:
`TensorAddress.read` renders through the span's own orthonormal basis, so the addressed point
**already sits at the joint minimum**. ⭐ The zero-step arm was asserted **BIT-IDENTICAL** to the
static read by `np.array_equal` on 256 requests, **and it passed.**
⇒ ⭐ **The zero-dynamics claim is not a conjecture; it is measured.** Your job is to state its scope
exactly, and to say where dynamics can still act (wave 6's answer: **across wells, not within one**).

## ⭐ AND THE HALF THAT IS A PREDICTION, NOT A MEASUREMENT
CRR predicts the field store's **settle half-life, parameter-free, to a factor 1.10** at the median
held-out corner; derived overdamped slope **+1.1385** against measured **+1.1368**.
⛔ **It predicts a HALF-LIFE, not a taxonomy row. It supports NO scoreboard claim.**
⚠ **And a caution the referee raised: this sharpest number has the programme's WEAKEST artefact
trail.** ⭐ **Check the provenance yourself and say how strong it actually is.** ⛔ Do not write D18 up
as a confirmation; `d14_confinement` is tautological — quote the Hessian check instead.

## ACCEPTANCE CRITERION
⭐ One paragraph the paper can carry verbatim, plus an explicit **"what this may NOT say"** list, plus
**a numerical sanity check on the store itself** that a reader could re-run from your `checks/`.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- The zero-dynamics limit holds only under conditions we do not meet ⇒ ⭐ **name the conditions.**
  That is the paragraph, and it is a better one than an unqualified claim.
- CRR's provenance does not survive your check ⇒ ⛔ **say so.** The number leaves the main text.
