# w15 spoke/cmaze-o-harness — BUILD CMAZE-O, THE OBSERVATION INSTRUMENT
**Read `COMMON.md` first.** Slug `cmaze-o-harness`. Agent type: experiment-engineer.
⭐⭐ **WAVE 16 IS GATED ON YOUR `HARNESS-READY.json`. It is the critical path of this campaign.**

## The one job
Build **CMAZE-O**, the compositional **observation-generation** instrument — the world model's *observation model* — and ship it with a seeded VAL smoke.

**Request vocabulary:** agent colour × key colour × door colour × tinted-floor colour × agent cell × key cell × door cell × goal cell × layout id (a **fixed bank of layouts seen in training**).
**Output:** the rendered grid state.
**Judge:** exact placement of every object AND legality of the state under the engine's rules (a key never on a wall, a door on a corridor, colours consistent). ⛔ **No tolerance.** `tolerance_used` must be `null` on every row.

⛔ **Engine and judge imported unmodified from CMAZE-1.**

## Hold-outs
**VALUE** (one colour never seen) and **COMBINATION** (2-way, n-way over the vocabulary).

## Rows
`seen` · `unseen 2-way` · `unseen n-way` · `value hold-out`.

## Deletion rows
Delete every state carrying colour c (**value**), and colour c ∧ tinted floor (**conjunction**). Post-deletion composition on the other colours vs rival **retrain-without** and **as-trained**. **Adaptation cost printed.**

## Arms
CLU single design · MLP / LSTM / transformer / **continuous MLP** at matched params **on the same frozen encoder and decoder** · **lookup floor** · charter-C controls · **BLIND columns**.

## Acceptance criterion — ALL of these
`$CHLU_HUB_OUTPUTS/w15-cmaze-o-harness/HARNESS-READY.json` with `"status": "ready"` containing:
1. **Independent replay** of every judged state by a second independently-driven checker — **0 disagreements**, count printed.
2. **No impossible request** shown to a non-refuser arm; count checked printed.
3. Hold-out disjointness from train proven by construction, check printed.
4. `tolerance_used: null` asserted on every row.
5. **Seeded VAL smoke** of Config-A single design vs **every** rival, 3 seeds, all rows, in `VAL-SMOKE.json` + a table in `FINDINGS.md`.
6. `PREREG.md` with bars pre-declared before any number.

## What would prove this wrong
⭐ If the **lookup floor** reaches the CLU on `unseen n-way`, the task is memorisable and the instrument cannot support a store claim — record that in `"instrument_warnings"` and **ship anyway**. ⛔ A weakness is written as a correction to the spec, never as a reason to stop.

## Files you own
`chlu/exp_w15_cmaze_o*.py`, `scripts/w15_cmaze_o_*.py`, your worktree, your output dir. ⛔ Import `exp_w8_envscout` / `exp_w9_maze`; never edit them. ⛔ You do NOT share any file with `cmaze-t-harness` — it owns `*_cmaze_t*`, you own `*_cmaze_o*`.

## Scored runs
**ZERO.**
