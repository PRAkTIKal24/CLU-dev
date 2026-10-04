> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **Evidence base: `/c/chlu/evidence/outputs/`** · claims matrix + negatives registry at
>   `/c/chlu/evidence/`. ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — it is a FROZEN, OLDER snapshot
>   and every number in it is stale. Reading it silently produces stale work.
> - ⭐ **Write ONLY to `/c/chlu/outputs/w12-w12-spoke-cache-gen2-seeds-3-4/`** and your own worktree. ⛔ Never another
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

# w12 spoke/cache-gen2-seeds-3-4 — generation-2 caches for seeds 3 and 4

- **Agent:** `experiment-engineer`. **Runs FIRST** — it gates whether this wave can report 5 seeds.
- **Output:** `/c/chlu/outputs/w12-cache-gen2-seeds-3-4/` (LOG.md, FINDINGS.md, CACHE-READY.json).
- **Owns:** its output dir + branch `w12/cache-gen2-seeds-3-4`. ⛔ No other spoke's tree.
- **Scored runs:** ⛔ **NONE.** This builds caches; it measures nothing.

## THE ONE JOB
Build **generation-2 feature caches for seeds 3 and 4**, with the **re-encode gate applied**, in the
same layout every downstream stage reads.

## ACCEPTANCE CRITERION
⭐ **Publish `CACHE-READY.json`** listing, per seed, the cache path, the row count, the dtype/shape,
and the per-block sha256. ⛔ A cache that exists but is not listed there does not count — downstream
hubs gate on this file, so it is the deliverable.

## ⛔ NON-NEGOTIABLE
- ⭐ **Build on ONE CPU device.** `w4_cache2.device_guard` refuses otherwise, and every stage
  downstream of the cache reads it.
- ⭐ **Assert `chlu.__file__` resolves inside YOUR worktree** before you build anything. The shared
  site-packages carries an editable `.pth` pointing at the Head's checkout; if `chlu` resolves there
  every number built on this cache is void.
- ⭐ Match seeds 0–2's cache **shape and dtype exactly** and say so in `CACHE-READY.json`.
- ⛔ **Do not touch any seed 0–2 cache.** They are live inputs to other rows.
- ⭐ Poll your own jobs to completion **inside a turn**; ⛔ never end a turn with detached work running.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- The re-encode gate rejects seeds 3/4 ⇒ **say so and stop.** ⭐ A 3-seed wave declared is correct;
  a 5-seed claim on a cache that failed its gate is not.
- The caches do not match seeds 0–2 in shape/dtype ⇒ they are **not poolable**, and mixing them into
  one plot without a printed `n` is the failure this spoke exists to prevent.
