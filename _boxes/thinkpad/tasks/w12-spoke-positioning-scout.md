> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **Evidence base: `/c/chlu/evidence/outputs/`** · claims matrix + negatives registry at
>   `/c/chlu/evidence/`. ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — it is a FROZEN, OLDER snapshot
>   and every number in it is stale. Reading it silently produces stale work.
> - ⭐ **Write ONLY to `/c/chlu/outputs/w12-w12-spoke-positioning-scout/`** and your own worktree. ⛔ Never another
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

# w12 spoke/positioning-scout — how a new primitive gets positioned, and what evidence it needs

- **Agent:** `web-scout`. Read-only.
- **Output:** `/c/chlu/outputs/w12-positioning-scout/POSITIONING.md` (cited brief) + LOG.md.
- **Owns:** its output dir only. ⛔ No code, no other tree.
- **Scored runs:** ⛔ none.

## ⛔ FIRST, A MECHANICAL CHECK — do it before anything else
⭐ **Confirm you can reach the web from this box.** Fetch one page and print its title.
⛔ **If you cannot, write `NO-WEB-ACCESS.txt` into your output dir, say so, and STOP.**
⚠ Do not substitute recalled knowledge for a citation — an uncited brief is worse than none here.

## THE ONE JOB
**How did papers that introduced a NEW PRIMITIVE position it, and what minimal dynamics evidence did
they show?** Cover **modern Hopfield networks · S4 / Mamba · neural ODEs · energy transformers**.

Answer three things, with citations:
1. **The positioning sentence** each used — what did they claim to BE, and what did they not claim?
2. ⭐ **The minimum evidence** they showed for the dynamics claim. ⛔ Not their best result — their
   *minimum*: what would a reviewer have accepted as sufficient?
3. **What ICLR reviewers accept as MOTIVATION versus as CLAIM.** These are different bars and the
   distinction is the whole point of this brief.

## THE DELIVERABLE
⭐ **A recommended shape for our latent-dynamics outlook section** — what we may say given that our
dynamics evidence is a **settle half-life prediction (CRR)** and ⛔ **not a benchmark result.**

## ⛔ CONSTRAINTS FROM THE PROGRAMME — these bind your recommendation
- ⛔ **Never** *"multilinear/bilinear factored generation with a request-indexed tensor head"* —
  Tenenbaum & Freeman (2000) and TensorFaces (2002) own it.
- ⭐ Our claim ladder is: **(1) exact deletion on a compositional generative store · (2) the interface
  split · (3) wells as a use.**
- ⭐ The deletion sentence is **"downdate the statistic, not the truncated factors."**

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
The honest answer is *"reviewers require a benchmark win for a primitive claim"* ⇒ ⭐ **say that
plainly.** It is a finding about our positioning, and it is more useful than a comfortable one.
