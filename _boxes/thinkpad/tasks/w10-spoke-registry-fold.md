> ## ⛔ STANDING RULES FOR THIS BOX — read before the job below
> - ⭐ **The evidence base is `/c/chlu/evidence/outputs/`** (live, shipped 2026-09-21).
>   ⛔⛔ **DO NOT read `$CHLU_CLAUDE_ROOT`** — it points at a FROZEN, OLDER snapshot of the same
>   tree, and every number in it is out of date. Reading it silently produces a stale paper.
> - ⭐ **You write ONLY to `/c/chlu/outputs/w10-registry-fold/`.** ⛔ No other spoke's tree. ⛔ Never
>   into the Head's profile. ⛔ Never create a `.claude/` directory in your worktree.
> - ⭐ **Poll any background job to completion INSIDE a turn** (loop on its output file with sleep).
>   ⛔ Do not end a turn while detached work is running — nothing will re-invoke you.
> - ⛔⛔ **Never run a box-wide kill** (`taskkill /F /IM python.exe` or equivalent). Kill only PIDs
>   you captured at launch. Another agent's training run has been destroyed this way before.
> - ⭐ **Commit and bundle your branch WHOLE before you finish** — `git bundle create <f> <your
>   branch>`, ⛔ not `--all`, which mixes other agents' branches into your file.
> - ⭐ Write a `LOG.md` in your output dir as you go. It is the record, and the Hub reads it.
> - ⭐ If something is not run, it is a **DECLARED NOT-RUN with a reason**, ⛔ never a zero, a blank
>   or a silent omission.

# w10 spoke/registry-fold — fold every wave-3..9 verdict into the claims matrix + negatives registry

- **Agent:** `doc-curator`.
- **Output:** updates to the claims matrix and `/c/chlu/evidence/negative_results.md` (⟲ dated addenda only).
- **Owns:** `/c/chlu/outputs/w10-registry-fold/**` ONLY. It writes UPDATED COPIES of `claims_matrix.md` and `negative_results.md` there; ⛔ `/c/chlu/evidence/` is READ-ONLY. The Hub folds them back.
- **Scored runs:** ⛔ none.

## THE ONE JOB
Every verdict below is recorded **with the number that decided it and the file that holds it** —
and, where one exists, **with its novelty verdict attached to the same row**.

## THE ITEM LIST — each is a row, none may be dropped silently
1. ⭐ **Exact deletion by downdate.** Downdate == refit: 1e-16…1e-14 on Shapes3D (9/9 seeds, two
   independent implementations) **and** 7.5e-16 / 8.8e-16 / 8.2e-16 on CMAZE-1, subspace projector
   0.000e+00, 0 passes, items never kept. ⭐ Independently replicated from a clean checkout with
   **zero numeric differences**. ⭐ **Unclaimed in the literature** — and ⛔ do NOT concede exactness
   to the decremental-PCA line: Hall–Marshall–Martin is approximate *in its own words*.
2. ⛔ **The composition result does NOT transfer to a route task** (w9, all 3 seeds): matched
   transformer/LSTM beat the CLU on unseen pairs; a matched-bytes lookup ties or beats it; and
   `landscape_shuffled` / `random_atoms` **do not collapse**. No row claimed for the arm.
   ⭐ Record the isolated mechanism: plan-space overlay of the two halves scores **1.0000**,
   latent-space averaging scores **0.13** — the read is linear in the request design, so it composes
   latents where the task composes plans. **The lever is the read verb, not the partition.**
3. ⛔ **The store does not shape the representation** (w8 BAR B fired) — ⭐ but record the
   decomposition, not the wash: unfreezing HURTS, the swap loss REPAIRS it, and the net is two real
   effects cancelling.
4. ⛔ **Compose and forget are incompatible at matched bytes** — both hold at **7.48×** the byte pin
   and fail together at 1.0×; the covariance forces K ≥ 36 and composition dies above K ≈ 5.
5. ⛔ **Symbolic refusal closed** · ⛔ **lifetimes dominated** · ⛔ **placement closed** · ⛔ **row 2
   lost** · ⚠ the **Mexican-hat/interpolation absence** (an Advisor error, recorded as such).
6. ⚠ **gen-1 / gen-2 hygiene** and the **two-gym laws** — including w9's own correction: CMAZE-1's
   rule B cannot trap a route that uses **one key**, and 1242 of 1449 routes do. That is a **spec
   weakness logged against the environment**, not a result.
7. ⛔ **The novelty STOP:** never claim *"multilinear/bilinear factored generation with a
   request-indexed tensor head"* (Tenenbaum & Freeman 2000; TensorFaces 2002; MLC-VAE 2020).

## ⛔ READ BOTH, NOT ONE
The novelty close-out is in **`/c/chlu/evidence/outputs/_thinkpad/w8/w8-novelty/NOVELTY.md` AND its
`LOG.md`** — the LOG carries the corrections that give ground back. Reading only the first
produces a more pessimistic registry than the evidence supports.

## ACCEPTANCE CRITERION
Every row above appears with **a number and a path**. ⭐ A verdict recorded without the measurement
that decided it is not a registry entry.

## ⛔ WHAT WOULD PROVE THIS SPOKE WRONG
A row you cannot source ⇒ say so in the row rather than paraphrasing a report.
