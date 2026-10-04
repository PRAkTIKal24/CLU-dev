# PREREG — w15/cmaze-o-harness (CMAZE-O, the observation instrument)

**Spoke:** `w15/cmaze-o-harness`. **Scored runs planned: ZERO.** Everything below is
measured on **VAL**. No `/c/chlu/SCORED-LEDGER.md` row is written because no scored run
is spent (COMMON.md, "Scored-run discipline (only spokes that spend a scored run)").

**Written before any arm number exists.** Timestamp in
`PREREG.timestamp.txt` beside this file.

## What preceded this file, and why it is not a result

Two **instrument-construction probes** were run before this PREREG. Neither measures an
arm; both are properties of the environment, and both are reported in FINDINGS.md with
their artefact path:

1. **braid probe** — how often a door can be load-bearing at each braid fraction. It
   fixed `BRAID = 0.08` and the balanced-articulation door placement.
   (`artefacts/braid_probe.json`)
2. **AE loss probe** — the autoencoder's own ceiling under per-cell BCE on the object
   channels was `exact_objects = 0.0025`, which fixed the decision to train the four
   object channels with a **spatial softmax over the 225 cells**.
   (`artefacts/ae_loss_probe.json`)

No arm, no control, no rival and no held-out row was scored before this file.

## The instrument

* Request = **nine words**: layout id × agent colour × key colour × door colour ×
  tinted-floor colour × agent zone × key zone × door zone × goal zone. **n = 9** factors
  jointly composed (COMMON.md notation). **K = 8** wells.
* Answer = the **rendered grid state**.
* Engine, solver and legality decided by `w8_envscout.maze_colour`, **imported
  unmodified**.
* Judge = `chlu.exp_w15_cmaze_o_env.judge_state`: `valid = exact_objects AND legal`.
  `exact_objects` is the eight request-borne facts (four object cells, four colour
  words). `legal` is the engine's own clause set. **`tolerance_used` is `null` on every
  row.**
* Layout bank: **12 layouts**, fixed per seed, all seen in training.
* Rows: `seen`, `unseen_2way`, `unseen_nway` (n = 4 over layout × agent zone × key
  colour × tint colour), `value_holdout` (colour 5 never seen anywhere),
  `value_slot_holdout` (colour 4 never on an agent, but on keys/doors/tints).
* **3 seeds, all or none: 0, 1, 2.** No "9/9 seeds".

## Bars, declared now

| # | bar | what it decides | if it fails |
|---|---|---|---|
| B1 | `AE-CEILING` **valid ≥ 0.50** on `seen`, mean over the 3 seeds | whether any arm can be measured at all on this pipeline | the instrument is DECLARED not viable at this encoder budget and the number is reported as such — not as an arm result |
| B2 | every one of the four charter-C controls at **valid ≤ 0.10 × CLU valid** on `seen` | whether a store claim is available | it is an **interface** claim or **nothing**, and FINDINGS says which |
| B3 | the **lookup floor** on `unseen_nway` at **valid < CLU valid − 0.02** | whether the task is memorisable | recorded in `HARNESS-READY.json → instrument_warnings` and **shipped anyway** (task file) |
| B4 | every **BLIND** column strictly below the CLU on `seen` | whether the composition is load-bearing at all | recorded as an instrument warning and shipped |
| B5 | independent-replay disagreements **= 0**; impossible requests shown to a non-refuser arm **= 0**; hold-out disjointness check **passes**; `tolerance_used` **null** on every row | whether `HARNESS-READY.json` may say `"status": "ready"` | status is **not** `ready` and the failure is named |
| B6 | store-scoped deletion exactness: `max abs diff ≤ 1e-6` between the downdated store and a store fitted from scratch without the deleted rows | whether the deletion row is store-scoped exact | the deletion row is reported as approximate with the measured gap |

## What is NOT a bar

* **The CLU vs the four learned rivals.** No bar is set, because the honest outcome of a
  first VAL smoke is the table. The pre-declared reporting rule is: the CLU is said to
  **separate** from a rival only if it is ahead on **≥ 2 of 3 seeds** on that row;
  otherwise the recorded outcome is **"no separation"**. All four rivals are reported on
  all rows; ⛔ the best is never reported alone.
* **`value_holdout` (strict).** It is expected to be **0.0000 for every arm including the
  ceiling**: a colour that was never once rendered cannot be rendered. It is shipped as a
  no-tolerance integrity check on the judge, not as a comparison.

## Matching, declared

* Rivals are matched to the CLU's **deployed read floats** = atom floats + address-book
  floats, to within **2 %**, and the realised count is printed on every row.
* The **float64 sufficient statistic** (`S, t, n` and the head's `A, Mx`) is what buys
  exact deletion. It is declared **beside** the deployed bytes, never netted in, and
  **never matched to a rival** — the rivals are not asked to delete.
* The **lookup table is a printed floor, never a rival.**
* Rivals receive the same interpolated one-hots the CLU's design is built from but
  **without the pair-interaction block**.
* The frozen encoder and decoder are shared by every arm and are declared **outside** the
  deletable object.

## Deletion, declared

* **Value deletion:** every training state carrying colour `c` in any slot.
* **Conjunction deletion:** colour `c` **∧** tinted floor — i.e. every training state
  whose *tinted-floor* colour is `c`.
* The CLU deletes by exact downdate of `S, t, n, A, Mx`; post-deletion adaptation is
  **light and data-free** (one closed-form re-solve of the head against the already
  retained statistics) — ⛔ never "no changes". Its wall-clock is printed.
* The rival deletes only by **retrain-without**, and its cost is printed. The
  **as-trained** rival (no deletion at all) is printed beside it.

## What would prove this wrong

* the lookup floor reaching the CLU on `unseen_nway` → the task is memorisable;
* the charter-C controls not collapsing → there is no store claim here, only an
  interface claim;
* a BLIND column matching the CLU → the composition the instrument is built around is
  not load-bearing;
* the AE ceiling near zero → the pipeline, not the read, is what the table measures.
