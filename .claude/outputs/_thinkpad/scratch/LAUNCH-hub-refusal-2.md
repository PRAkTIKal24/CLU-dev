# ADDENDUM — `hub/refusal-2`
**Read `LAUNCH-COMMON.md` first** (gate ruling, budget, R1–R5). Then `.claude/tasks/hub-refusal.md` is
your task — **re-posed below on your predecessor's structural result.**

- **Slug:** `hub-refusal` · **Branch:** `hub/refusal-2` · already created at `945a3e9`
- **Worktree:** `/c/Users/PratikJawahar/chlu/CHLU-hub-refusal-2` · already created
- **Threads: 4.** Sub-agents: ≤ 2, inside those 4, **every one on `model: "opus"`**. You run on Opus.

## ⚠ Relaunch (COMMON §-0)
Your stopped predecessor wrote **nothing** — `hub-refusal/` holds only the *original* overnight hub's
work, which remains your inherited fact. Worktree is clean. New `START_UTC.txt`, new `RELAUNCH` log
section. ⭐ The §5.1 falsifier is CLEAR (generation-table's file).

## ⛔ The old criterion is DEAD. Do not re-derive it.
Your predecessor measured the original criterion (settled energy of the argmax/MALA read) to be
**structurally blind**: the address book is a simplex over stored atoms and **"a simplex has no outside"**.
`out_of_range` has the **sign backwards**, monotonically, over 13× extrapolation — and the store is
**decorative** for it (`store_removed` 0.7809 ≥ 0.7511). ⛔ That line is closed. Its analysis is inherited
fact at `.claude/outputs/hub-refusal/REPORT-hub-refusal.md`.

## ⭐ THE NEW CRITERION — pre-register exactly this, ONE, landscape-derived, threshold on `val` only
**Coercive-component screening.** `S = E_store(x_read)` — the store's own energy at the **linear** read
point. (Vocabulary and precedent: *Certified Access*, BIBO screen — *"refuse the jump if the exit lies
outside the coercive component"*.)

⭐ **Why this is not the dead criterion:** under the **argmax** read this carries **zero** information —
the read is always *on* an atom, distance 0.0000. Under the **linear** read the weights are unconstrained,
so an invalid request can put `x_read` **outside every basin**. The read change is what makes refusal
possible at all.

**Rule:** refuse iff `S > quantile_0.95(S on val)` — using the harness's **own shipped**
`refusal_valid_quantile`. ⛔ **Not swept.** ⛔ Threshold chosen on `val` only.

**Companion, REJECTED IN ADVANCE:** `d(x_read, nearest atom) / atom spread` (the §C.4 statistic) —
**computed and logged, ⛔ never adopted.** Declaring this now is what stops it becoming a second bite.

## ⛔ MANDATORY DISCLOSURE — put this in PREREG-1, verbatim
> *"This criterion was chosen on a structural argument formed after reading `hub/refusal`'s analysis of
> invalid examples on a fit-derived synthetic analogue. No benchmark invalid row was seen. The
> zero-invalid-supervision claim is therefore 'structural, not blind', and the paper says so."*

⭐ This is not a formality. A refusal threshold that saw the data it refuses looks excellent and means
nothing; the honest framing is what makes the result usable.

## Mandatory controls — all three, on every row
1. **`store_removed` must collapse `S`'s AUROC to ~0.5.** ⛔ Otherwise it is the address book again and
   **the result is void.**
2. **A physics-free readout-margin control.** *Certified Access* §4.1 measured the relaxation residual at
   ≤ +0.024 AUROC over a plain margin — **a reviewer will ask.**
3. ⭐ **`accept A7` printed beside `accept B` on every row.** Three of five rivals falsely refuse ~60 % of
   the valid A7 control, and **the published table hides it.**

## Your bar
`min(AUROC_oor, AUROC_dup) ≥ 0.75` at the single val-only threshold · `accept_A7 ≥ accept_B − 0.05` ·
and **beating every rival on the PAIR** (best rival pair is kNN's **(0.8426, 0.3519)**; ⭐ **no rival
refuses both kinds**).
⛔ Beating kNN on `out_of_range` alone (0.97) is **not required and not expected** — do not chase it.

## ⛔ GO / NO-GO before any scored spend
Run `S` once on your predecessor's synthetic analogue (`scorer/refusal_lib.py`, `term_experts` —
inherited) at rung A and B, seed 0.
⛔ **If `min(oor, dup) < 0.6` there: DO NOT SPEND. Report the number and stop the refusal line for this
iteration.** That is a complete and acceptable outcome.

## Your one scored run
C-set scoring on **seed 0** traces from the **linear** read. 1 spend max, and only after the go/no-go
clears and PREREG names object, bar and controls.

⚠ Your predecessor concluded the 3-seed refusal win is unreachable here (seeds 1–2 need ~23,480 rows
re-encoded). ⭐ That remains true — which is why your target this iteration is **seed 0 only**, reported
as *"scored 1/3 — seed 0 only, 3-seed pending full caches"*. ⛔ Do not attempt seeds 1 or 2.
