# figures/rendered/ — w19 `success-failure-panels` additions (files `*w19*` only)

w17's files in this directory (`rc_set1_*`, `rc_set2_*`, `rc_set3_*`, `rc_deletion_*`, `REQUESTS.json`,
`SAME-DECODER.json`, `STATUS.json`, `README.md`) are **untouched**. Sets 1 and 3 stay exactly as w17 built them.
Builder: branch `w19/success-failure-panels`, `w19_success_failure_panels/build_twice.sh`.

## Status per panel set
| file | task | status |
|---|---|---|
| `rc_w19_t7_per_arm_success_failure` | T7 Shapes3D, K\*=32, seed 0 | **RENDERED.** One success row and one failure row each for the CLU, MLP, LSTM, Transformer and Continuous MLP. All six columns are rendered on every row. The subject arm is the bracket at the left and the coloured frame on its panel |
| `rc_w19_t7_rival_succeeds_clu_fails` | T7, seed 0 | **RENDERED.** A rival ✓ and the CLU ✗ on the same held-out request: MLP, LSTM, Transformer |
| `rc_w19_t7_rival_succeeds_clu_fails_seed1` | T7, seed 1 | **RENDERED.** The continuous MLP's only such request at any seed (seed 1, rendered through seed 1's own decoder) |
| `rc_w19_cmazet_families` | CMAZE-T | **ENGINE-ONLY.** Shows $s_t$ + action and the engine's true $s_{t+1}$ for one held-out TEST n-way transition from each of the five named families. The arm columns are hatched: **DECLARED NOT BUILDABLE** |
| `rc_w19_cmazet_door_then_tint` | CMAZE-T | **ENGINE-ONLY.** Door-then-tint (`interaction`) transitions, one per move direction present (N, S, W; the seed-0 hold-out has no E interaction tuple). Arm columns DECLARED |
| `rc_w19_cmazet_deletion` | CMAZE-T | **ENGINE-ONLY.** One transition on deleted colour c = 1. The columns CLU before/after (downdate), rival as trained and rival retrained without are all DECLARED |

## "Rival succeeds, CLU fails": such rows EXIST. Counts are in `REQUESTS-w19.json` `t7.counts_per_seed`
Counted over **every** held-out T7 test request (5,688 per seed), rival valid and CLU invalid:

| seed | MLP | LSTM | Transformer | Continuous MLP | any rival | (lookup floor, not a rival) |
|---|---|---|---|---|---|---|
| 0 | 26 | 13 | 2 | 0 | 30 | 67 |
| 1 | 11 | 13 | 3 | 1 | 24 | 26 |
| 2 | 3 | 25 | 6 | 0 | 30 | 46 |

For context, the CLU is valid on 5,177 / 5,167 / 5,354 of those requests. ⚠ These counts come from this spoke's re-render of the w12/w12b T7 legs, which reproduced every per-corner n_valid in the three `scored.json`. They are w≤14 evidence.

## Same frozen decoder: ASSERTED in code (`SAME-DECODER-w19.json`)
The path is w17's `OneDecoder` / `assert_same_decoder`, unchanged. Seed 0 has 78 panels, 1 decoder object, and hash `01ef91f7…` = `winner_hashes.json` seed_0. Seed 1 has 6 panels and hash `001011bc…` = seed_1. The positive control (render + classify = `Judge.read_back`) is identical on both. Each selected request is re-rendered alone, and its six verdicts must equal the full-corner scan's, or the build stops.
CMAZE-T renders no decoder or codec output at all. Its panels are integer engine state.

## Selection: by rule, seeded (`REQUESTS-w19.json` holds the rules and exact tuples)
* T7: the scan covers all 109 corners × 5,688 requests × 3 seeds. Each per-corner per-arm n_valid reproduces `scored.json` (0 mismatches). Each row is a uniform draw with `default_rng(19)` from all qualifying requests.
* CMAZE-T: the transitions are the scored TEST leg's own rows, re-derived from the deterministic engine. Checked against `w16-maze-wm/t1/A1_test_seed{0,1,2}.json`: bank meta, hold-outs and per-class row counts are all equal. Rows are taken first-in-scorer-order.

## DECLARED
* **CMAZE-T arm columns, and "one success + one failure per arm" on CMAZE-T:** not buildable. w16 never persisted a codec, CLU, control or rival object, nor any per-transition prediction. Re-training would be a new TEST-leg run with a new codec, which is forbidden. What would unblock it: `outputs/w19-success-failure-panels/NEEDS-ARTEFACTS.json`.
* **CMAZE-T family `rule_a`:** there are 0, 0 and 1 TEST n-way rows at seeds 0, 1 and 2. Because p_door_tint = 1.0, a closed door is untinted only where its tint was reverted. The `rule_a` row is the one seed-2 row, labelled "(seed 2)".
* ⛔ AMENDMENT-1: every CMAZE-T figure is **w16 material**. It carries no w16 number, but it belongs in a `\PLACEHOLDER` slot until the Head and the Advisor have discussed w16. The T7 figures are w≤14 evidence.

## Rebuild proof
`SHA256-w19-build1.txt` = `SHA256-w19-build2.txt`: two separate processes, 15 files, byte-identical.

## Caption drafts (for `draft-figure-slots`)
% SOURCE: figures/rendered/{REQUESTS-w19,SAME-DECODER-w19}.json ; evidence w12-single-design-crossover + w12b-nr6-and-kstar scored.json
* per-arm: "Held-out requests (seed 0, K* = 32) chosen at random, by a fixed seed, among those each arm renders validly (upper row of its pair) and invalidly (lower row). Every arm is rendered on every request through one frozen decoder. The mark is the judge's verdict."
* rival-succeeds-CLU-fails: "Held-out requests on which a rival renders validly and the CLU does not. Such requests exist at every seed (24–30 of 5,688); one per rival is shown."
* CMAZE-T (placeholder slot): "Held-out CMAZE-T transitions, one per rule family: $s_t$ with the action, and the engine's $s_{t+1}$. \PLACEHOLDER{w16 CMAZE-T per-arm generated $s_{t+1}$ with exact-judge marks; not buildable from persisted artefacts — may remain undrawn}"
