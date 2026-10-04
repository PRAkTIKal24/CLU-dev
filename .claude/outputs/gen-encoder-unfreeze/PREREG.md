# PREREG — gen-encoder-unfreeze (written BEFORE the harness was run)

**Written:** 2026-09-14, before `exp_gen_encoder.py` had ever executed.
**Base:** `main @ 994ca9b`. **Branch:** `agent/experiment-engineer/gen-encoder-unfreeze`.
**Scale it applies to:** the **LOCAL SMOKE** ordered by the Head's 2026-09-14 scope change. ⛔ Every number below is a
plumbing-grade directional prediction at toy scale. It is **not** a phase-1 result and does not license any claim.

## DIAL DECLARATION
- **Dial:** *the encoder* — frozen in every shipped phase-1 artefact; here it trains with the store in the loop.
- **Laundering control:** the **frozen-encoder arm at matched compute** — identical stage-2 loop, identical batches,
  identical step count, identical loss; the *only* difference is whether the encoder's leaves are in the optimiser.
- **Falsifies the claim:** unfreezing does not raise the distinct-code count on B.
- **Does NOT falsify:** absolute distinct-code counts at smoke scale being far below 72 (the harness runs ~10^2
  gradient steps, not ~10^3–10^4); losing on reconstruction MSE to the shipped frozen pipeline (different loop).

## The predictions, and how they were derived

### P1 (primary, directional) — unfreezing raises the distinct-code count on B
`n_distinct(B, unfrozen) >= n_distinct(B, frozen)`, and I predict a **small strict lift**, `+1 … +4` codes.
*Derivation:* the published collapse is 9 distinct codes over 72 requests (R29 Q2) with `30 bits >> 72`, so capacity is
not binding — the binding thing is that the frozen features put the whole colour corner inside a handful of
nearest-mode cells, so the address book's naive-Bayes posterior argmaxes onto the same tuple. A straight-through
gradient lets the reconstruction loss push same-code features apart, which is the *only* mechanism by which unfreezing
could act here. At ~60 joint steps that mechanism is barely engaged ⇒ a small lift, not a fix.
⭐ **I am explicitly pre-registering that a NULL result is the likely outcome at this scale** and that a null is a
finding (it narrows the problem to "not the freezing"), not a failure of the build.

### P2 (companion) — B and MS move together
`sign(Δ_unfrozen-frozen on B) == sign(Δ on MS)` and `|Δ_B − Δ_MS| <= 3` codes.
*Derivation:* R29's framing — the collapse is a property of the **colour corner**, not of held-out-ness. MS lives in the
same corner with a *seen* shape. If B moved and MS did not, the mechanism would be about held-out-ness and that would
itself be the news.

### P3 (structural, must hold exactly) — the channel does not widen
Realised channel width `= K·log2(M)` bits **exactly**, identical for both arms, and the number of distinct decoder
inputs over any request set is `<= M^K`. ⭐ Any widening is a declared change to the claim (R20), not an improvement.

### P4 (the test must have teeth) — the bypass check fails on a deliberately bypassed variant
`assert_no_feature_bypass(..., bypass_alpha=a)` **raises** for every `a > 0` and **passes** at `a == 0`.
*Derivation:* the check's discriminating leg is value-level: two feature vectors with the SAME K codes must produce a
**bit-identical** decoder input. A bypass `q = read + a·z` breaks that by exactly `a·(z1 − z2) != 0`.

### P5 (frozen-arm integrity) — the control really is frozen
The frozen arm's encoder `param_hash` is **bit-identical** before and after its stage-2 loop, and both arms' stage-2
loops consume the **same number of optimiser steps on the same batches**.

### P6 (no collateral damage) — the frozen PATH is untouched
Every fingerprint in `.claude/scratch/gen-encoder-unfreeze/baseline_fingerprint.json`, measured on the **unmodified**
tree at `main @ 994ca9b` before any edit, reproduces **bit-identically** after the edit.

## What would make me report this spoke as failed
- P4 fails (the bypass test passes on the broken variant) ⇒ the guarantee is unprotected and nothing else matters.
- P6 fails ⇒ I changed the frozen path.
- The unfrozen arm "wins" on P1 while the step/batch budgets are not matched ⇒ I measured compute, not the encoder.
