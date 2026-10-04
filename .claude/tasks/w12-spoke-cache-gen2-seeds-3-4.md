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
