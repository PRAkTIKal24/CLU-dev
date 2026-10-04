> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **Evidence base: `/c/chlu/evidence/outputs/`** · claims matrix + negatives registry at
>   `/c/chlu/evidence/`. ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — it is a FROZEN, OLDER snapshot
>   and every number in it is stale. Reading it silently produces stale work.
> - ⭐ **Write ONLY to `/c/chlu/outputs/w12-w12-spoke-kway-ladder-harness/`** and your own worktree. ⛔ Never another
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

# w12 spoke/kway-ladder-harness — the k-way composition families, k = 3, 4, 5

- **Agent:** `experiment-engineer`.
- **Output:** `/c/chlu/outputs/w12-kway-ladder-harness/` (LOG.md, FINDINGS.md, **LADDER-READY.json**).
- **Owns:** its output dir + branch `w12/kway-ladder-harness`. ⛔ No other spoke's tree.
- **Scored runs:** ⛔ **NONE.** You build the instrument; `hub/composition-ladder` runs it.

## THE ONE JOB
Extend the value-holdout condition with `kway-*` families for **k = 3, 4, 5**: k factors jointly
pinned to a held-out box (the same widths the n-way builder uses), the other 6−k free.
⭐ **Row 3 is k=2 and row 4 is k=6 — the existing ends.** You are building the middle.

## ACCEPTANCE CRITERION
⭐ **≥ 49 pairwise-disjoint corners per k**, the **same paired-corner estimand and bar as rows 3–4**,
and `LADDER-READY.json` naming each family, its corner count, and its disjointness check.
⛔ **PRE-DECLARE THE BAR BEFORE ANY NUMBER EXISTS**, in a file, dated. Charter A.1.

## ⛔ NON-NEGOTIABLE
- ⭐ **Scale and orientation stay INELIGIBLE for VALUE hold-outs**, but are **eligible inside k-way
  boxes**, exactly as in row 4. ⛔ Do not quietly widen this.
- ⭐ **Build the check that can KILL the ladder before the ladder.** If the corners are not disjoint,
  or a k-family cannot reach 49, the ladder measures nothing — so write that check first and run it.
- ⛔ The k=2 and k=6 ends must reproduce the EXISTING row-3/row-4 corner construction. If your k=2
  disagrees with row 3, **your builder is wrong, not row 3.** Assert it.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
- A k for which ≥49 disjoint corners do not exist ⇒ ⭐ **declare that k NOT-RUN with the count you
  reached.** ⛔ Never pad with overlapping corners; the paired estimand assumes disjointness.
- Your k=2 not matching row 3's corners ⇒ stop and report; every ladder point would be off-instrument.
