# w15 spoke/cmaze-t-harness — BUILD CMAZE-T, THE TRANSITION INSTRUMENT
**Read `COMMON.md` first.** Slug `cmaze-t-harness`. Agent type: experiment-engineer.
⭐⭐ **WAVE 16 IS GATED ON YOUR `HARNESS-READY.json`. It is the critical path of this campaign.**

## The one job
Build **CMAZE-T**, the compositional *transition* instrument, and ship it with a seeded VAL smoke so wave 16 opens with numbers.

**The task:** given current grid state `s_t` (layout, positions, pocket) and action `a_t`, generate next state `s_{t+1}`. The **ENGINE judges the transition exactly**: legal under rules A and B, correct displacement, correct pocket. Rollouts of length L chain generated states; a rollout is valid iff **every** step is legal and the goal is reached within budget.

⛔ **Engine, solver and exact judge are IMPORTED UNMODIFIED from CMAZE-1.** You are building a request/transition layer and a harness around them — not a new engine. If you must touch engine code, stop and write why in `LOG.md` instead.

## Factors and hold-outs
- Factors of the transition: agent colour × key colour × door colour × tinted-floor colour × direction (× carry state).
- **VALUE hold-outs:** one colour never seen in a rule.
- **COMBINATION hold-outs:** 2-way and n-way — colour × door × tile × direction combinations never seen together.
- ⭐⭐ **On RANDOM mazes never seen at test**, so a within-maze lookup cannot transfer. **Cross-maze fragment-lookup is the floor and must be a printed column.**

## Rows the harness must produce
`seen` · `unseen 2-way` · `unseen n-way` · `rollout validity` · `value hold-out` (never-seen colour; **scored, no refusal display** — refusal displays are retired).

## Deletion rows
Delete every transition carrying colour c (**value**), and colour c ∧ tinted floor (**conjunction**). Measure post-deletion composition on the *other* colours vs rival **retrain-without** and rival **as-trained**. **Adaptation cost printed.**

## Arms
CLU single design · MLP / LSTM / transformer transition models at **matched params on the same frozen state encoder** · **lookup floor** (state,action → next state; within-maze AND cross-maze columns) · charter-C controls · **BLIND-COLOUR** and **BLIND-SEQUENTIAL** columns printed.

## Acceptance criterion — you are done when ALL of these hold
`$CHLU_HUB_OUTPUTS/w15-cmaze-t-harness/HARNESS-READY.json` exists with `"status": "ready"` and contains, each measured not asserted:
1. **Independent replay:** every generated transition re-judged by a second, independently-driven stepper — **0 disagreements**. Print the count of transitions replayed.
2. **No impossible example** is ever shown to a non-refuser arm. Print the count checked.
3. The **hold-out sets are disjoint from train** by construction, with the check printed (value hold-out colour never appears in a train rule; n-way combos never co-occur in train).
4. **Rule B bites:** report the fraction of transitions where the rule renders a route/action *illegal* rather than merely slower. ⚠ This is a **reported column, never an abort condition** — report the number whatever it is.
5. A **seeded VAL smoke** of the Config-A single design vs **every** rival, 3 seeds, all rows above, written to `VAL-SMOKE.json` + a readable table in `FINDINGS.md`.
6. `PREREG.md` with the bars pre-declared **before** any number.

## What would prove this wrong
⭐ If a **composition-blind** policy (BLIND-COLOUR / BLIND-SEQUENTIAL) scores near the CLU on `unseen n-way`, the instrument does not measure composition and you must say so plainly in `HARNESS-READY.json` under `"instrument_warnings"`. ⛔ That is a **reported finding, not a reason to abandon** — ship the harness with the warning. The two-gym law (a rule must render a route ILLEGAL, not merely slower; and the rules' interaction must be worth enough that a composition-blind policy cannot score well) is a **design constraint to satisfy**, ⛔ not a gate that rejects.

## Files you own
`chlu/exp_w15_cmaze_t*.py`, `scripts/w15_cmaze_t_*.py`, your worktree, your output dir. ⛔ You do NOT own anything under `chlu/exp_w8_envscout*` or `chlu/exp_w9_maze*` — import them, never edit them.

## Scored runs
**ZERO.** This is instrument construction. Val smoke is free.
