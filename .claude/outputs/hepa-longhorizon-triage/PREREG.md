# PREREG — hepa-longhorizon-triage
**Authored by the Hub, 2026-09-07, BEFORE any dataset was acquired and BEFORE any run.**
Frozen at repo HEAD `7fcef50`. No spoke may edit this file. Spokes read it and are graded against it.

Rationale for Hub authorship: the agent that runs the experiment must not author the prediction it is
graded on. Predictions below are the Hub's, from the label-construction code documented in
`.claude/outputs/hepa-benchmark-scout.md` §4 only — no data was inspected.

---

## 0 · The one question
Which HEPA-suite datasets genuinely require entity history older than 512 steps? Only those can carry
a long-horizon section. Everything else is decoration.

## 1 · THE PROTOCOL FILE — named once, used everywhere
**`hepa/utils/config.py` + `REPRODUCIBILITY.md` (the "clone-actual" protocol).**
NOT `papers/hepa-2026/{PROTOCOL.md,config_snapshot.py}`.

Reason: clone-actual is what a fresh clone runs and is internally consistent; the paper snapshot's own
`results_table1.csv` disagrees with paper Table 1 in >=3 cells (scout D2), and the snapshot defines no
per-dataset normalisation policy — it omits `norm_mode="none"` for C-MAPSS, without which C-MAPSS
h-AUROC goes to chance. Deltas we deliberately impose on clone-actual are listed in §2 and nowhere else.

Frozen settings inherited verbatim: AdamW; lr 3e-4 pretrain / 1e-3 finetune; wd 1e-2; batch 64;
alpha=0.1; n_cuts=40; dropout 0.1; target_mode="joint_train"; pretrain 50 ep / finetune 30 ep,
patience 10; C-MAPSS finetune = fixed epochs, all others = val-loss early stopping; per-dataset
NORM_POLICY as shipped (RevIN off for C-MAPSS, on elsewhere).
**Seeds: 42, 123, 456** (first three of clone-actual's {42,123,456,789,1337}). Every quoted number is
mean +/- std over exactly these three. A number from one seed is not a number.

## 2 · The only permitted deltas from clone-actual
1. `max_context` is swept. All else held.
2. Patch size stays P=16 at every rung. Token count therefore scales with context; that is intended
   and must be reported alongside every h-AUROC (a longer context that also buys more tokens is not a
   clean test of *history*, and the report must say so).
3. Horizon set per dataset unchanged (K=150 C-MAPSS/TEP, K=200 elsewhere).
4. **Horizon conditioning is HEPA's scalar-concat mechanism (`Linear(257,256)`, raw dt in native
   units) for EVERY arm that produces an h-AUROC in this task, including the tripwires.** The suite's
   published baselines use a 264K learned horizon embedding instead (scout D8); that mismatch is one
   reason the published margins are hard to read, and we do not reproduce it. Any arm compared here
   shares one conditioning mechanism or it is not compared.

## 3 · Context rungs
Reference rung for the kill rule is **512 on all nine datasets**, without exception.
Rungs: {shipped-if-not-512} u {512, 2048, 8192-or-max-available}.
`max_available` is computed per dataset by the audit spoke and is a hard ceiling; where the ceiling
falls below a rung, that rung is **NOT RUN** and is reported as not-run, never as a null or a zero.
Pre-registered expectation: BATADAL (43ch @ 1/hour, ~8.7k rows total) cannot reach 8192; C-MAPSS
cannot reach 2048.

## 4 · KILL RULE (verbatim from the Head, not reinterpreted)
A dataset is **KEEP** only if BOTH:
- (a) >=3-seed h-AUROC at the longest available context beats the 512 rung **by more than the seed
  spread**, AND
- (b) the matched-byte exemplar store at the same state-byte budget **does not close that gap to
  within 0.02 h-AUROC**.

Otherwise **DROP** — or **HOLD** if the data audit is the *only* failure.
Operational reading of "seed spread": max minus min across the three seeds at the 512 rung. The gap
must exceed that raw range, not a standard error. This is the conservative reading and it is chosen
deliberately.

## 5 · POINT PREDICTIONS — per dataset, made blind
The governing hypothesis: **a label defined relative to a local rolling statistic cannot reward long
history, because the target has already been normalised against recent history by construction.**
Labels that are absolute-threshold or externally supplied can reward it. This is the falsifiable idea;
if the sweep contradicts the split below, the idea is wrong and the report says so.

| Dataset | Label mechanism (scout §4) | Prediction | Confidence | Why |
|---|---|---|---|---|
| C-MAPSS FD001 | run-to-failure, last cycle | **DROP on audit** | high | engine lives 128-368 cycles; entities never reach 512 |
| C-MAPSS FD002 | as FD001 | **DROP on audit** | high | as above |
| C-MAPSS FD003 | as FD001 | **DROP on audit** | high | as above |
| C-MAPSS FD004 | as FD001 | **DROP on audit** | high | as above |
| PSM | supplied test_label | **DROP** | med | ships at context 100 and wins by +.01; server faults are locally-precursored |
| MBA | annotation dilated +/-w | **DROP** | med | 8192 steps @275Hz = 30 s; arrhythmia precursor is seconds, not minutes |
| GECCO | supplied EVENT column | **KEEP** | med | external ground truth, not locally normalised; 8192 min = 5.7 d spans the diurnal baseline |
| BATADAL | supplied ATT_FLAG | **HOLD** | med | right label type, but hourly data caps context near 2048; audit decides |
| ETTm1 | (OT - local_mean)/local_std > 2sigma | **DROP** | high | label is *defined* against a rolling window; long history is normalised away by construction |
| Weather | (T - rolling_mean)/rolling_std > 3sigma | **DROP** | high | same construction as ETTm1 |
| Beijing-AQ | pm25 > 250 ug/m3, ABSOLUTE | **KEEP** | med-high | only absolute threshold in the suite; 8192 h ~ 11 months carries the heating-season regime |

**Pre-registered aggregate:** 2 KEEP (GECCO, Beijing-AQ), 1 HOLD (BATADAL), 8 DROP.
**What would falsify the governing hypothesis:** ETTm1 or Weather passing the kill rule, or Beijing-AQ
failing it while ETTm1 passes. Either outcome means label-locality does not predict history-need, and
the section must be rebuilt on a different criterion.
**What would kill the long-horizon section entirely:** fewer than 2 KEEP datasets.

## 6 · Scope bans (from the Head, binding)
- **No CLU arm. No CLU number.** This task prices the datasets, not our model. A spoke that reports a
  CLU number has failed the task.
- SMAP, TEP, VIX are **out of scope**: VIX excluded (its label threshold is fit on the test split);
  SMAP and TEP carry event definitions their own authors call placeholder. They are neither run nor
  reported here.
- Work that did not run is reported as **NOT RUN**, with the reason. Never as a null, a zero, or a
  failure.
- HEPA is **CC BY-NC-SA 4.0** (non-commercial, share-alike). It is cloned into `.claude/` only. No
  HEPA source enters a tracked file — share-alike would propagate into our repo.
