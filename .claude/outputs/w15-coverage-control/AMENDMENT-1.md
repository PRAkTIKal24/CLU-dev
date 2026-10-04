# AMENDMENT 1 — the val geometry (written after a FAILED val launch; no valid number existed)

**What happened.** The PREREG's val geometry (middle scale band = indices 3 and 4, width 2) was launched at 07:14Z. Every cell of every seed read `distinct_valid_novel_tuples = 0.0000` with `judge_accuracy = 0.0000` — store and controls alike (`logs/amend0/val_seed*.log`, rows kept in `val_INVALID_amend0_width2/`).

**Cause (measured, `logs/amend1_dryrun.txt` and the debug run).** The test k=3 cells hold out the ladder's object hues (0.1, 0.2, 0.8, 0.9) × shapes 1–3 at scale indices 0–2 and 5–7; the width-2 val cells removed indices 3–4 of the same hue × shape. Shape 0 is the value-hold-out shape. So **no row of those four object hues survives in train_3val**; `value_presence_bitmap` marks them absent and `accept_rule` refuses every val request. That leg measured refusal, not composition. ⛔ Those zeros are an instrument fault and are NOT reported as a result.

**Change (the only one).** `MID_SCALE_IDX = (3,)` — the val cells pin scale 0.964 alone, leaving 1.036 in train. A new hard assertion refuses any val rung in which a value requested by a val corner is absent from the train index. Dry run: accept fraction 1.0 on both val rungs.

**Consequence, DECLARED.** val k=3: 6 cells / 18,000 rows, train 162,000; val k=6: 60 cells / 1,440 rows, train 277,920; both probe the identical 60 corners × **24 rows** (test: 120 × 72). Everything else in PREREG.md — arms, cells A/B/C/D, controls, learned_1x, seeds 0/1/2, every bar — is unchanged.

The bars were fixed before any valid number; the failed launch produced no informative number (all zeros incl. controls), so nothing was seen that could steer this amendment.
