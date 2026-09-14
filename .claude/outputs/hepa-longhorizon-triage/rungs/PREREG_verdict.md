# PREREG — hepa-lh-verdict (spoke-level, protocol §5 pre-registration rule)
Written by **experiment-engineer** on **2026-09-07 12:51 CEST**, **before a single wave-2 tripwire run
was executed**. It does NOT amend the Hub's frozen `PREREG.md` or `PREREG_AMENDMENT.md`; those govern.
This file only records *my* predictions so the measurements can be graded rather than narrated.

Inputs I had when writing this: `PREREG.md`, `PREREG_AMENDMENT.md`, `hepa-lh-tripwire.md` (wave-1
numbers for FD001@512/@2048 and GECCO@512), `hepa-lh-audit.md` Tables A/B, and the harness source.
No wave-2 number existed.

## P0 · Structural predictions (mechanism, not fit) — near-certain
1. **`prevalence` = 0.5000 exactly, on every dataset at every rung.** Constant scores ⇒ every
   non-degenerate per-horizon AUROC is 0.5. (Confirmed on FD001 in wave 1; asserted for all.)
2. **The three trivial baselines are RUNG-INVARIANT.** `EventDataset.starts` depends on
   `min_context` and `stride` only, never on `max_context`; `prevalence`/`seasonal`/`last_event_age`
   are functions of the cut index `t` and of `y_train`, not of the context slice. Therefore for a
   given dataset the three trivial numbers must be **bit-identical at 512, 2048 and 8192**.
   *Falsified if any trivial number moves between rungs on the same dataset.*
3. **The exemplar store rises weakly and monotonically with the rung, from BYTES not history.**
   Wave 1 measured +0.0039 on FD001 512→2048 with 0 % of windows reaching either rung. I predict a
   positive 512→8192 store delta on ≥4 of the 6 datasets, median delta in **[0.00, +0.05]**.
4. **`raw_window` collapses to ≈0.50 at 2048 and 8192 everywhere** (1–5 exemplars fit the budget).

## P1 · Point predictions — tripwire h-AUROC at 512 (primary budget `residual_stream`)
Central value with an 80 % interval. "store" = best PREREG-compliant kNN arm; "tripwire" = max(store,
best trivial). Reasoning is one line each and is the thing being graded, not the digit.

| Dataset | store @512 | last_event_age | seasonal | **tripwire @512** | reasoning |
|---|---|---|---|---|---|
| PSM       | 0.70 [0.60–0.82] | 0.65 [0.52–0.78] | 0.60 [0.50–0.72] | **0.72 [0.62–0.85]** | 25 ch, supplied labels, ttprev median 481 |
| GECCO     | 0.60 (measured w1) | 0.5182 (measured w1) | 0.8181 (measured w1) | **0.8181 `deterministic`** | rung-invariant per P0.2 |
| ETTm1     | 0.85 [0.75–0.95] | 0.80 [0.65–0.92] | 0.60 [0.50–0.72] | **0.87 [0.78–0.96]** | label = (OT−rolling_mean)/rolling_std; `chan_stats` literally stores mean/std/last ⇒ the descriptor contains the label's own ingredients |
| Weather   | 0.85 [0.72–0.95] | 0.70 [0.55–0.85] | 0.60 [0.50–0.75] | **0.86 [0.74–0.96]** | same construction as ETTm1 |
| BeijingAQ | 0.88 [0.78–0.96] | 0.80 [0.65–0.92] | 0.62 [0.50–0.75] | **0.89 [0.80–0.97]** | absolute pm25>250 threshold; the last channel value is nearly the label |
| MBA       | 0.75 [0.60–0.90] `deterministic` | 0.75 [0.60–0.88] | NOT_APPLICABLE | **0.80 [0.65–0.93]** | 275 Hz ECG, ttprev median 118 ⇒ age is strong; store is deterministic at every rung |

**Which arm wins the tripwire:** exemplar store on ETTm1 / Weather / BeijingAQ / PSM; `seasonal` on
GECCO; `last_event_age` on MBA. (6 calls, graded individually.)

## P2 · The verdict prediction (§6, gated — recorded now so it cannot be retrofitted)
Under `PREREG_AMENDMENT.md` clause (b2) — a tie at any rung is a DROP — and with tripwire values as
above, I predict **0 KEEP**: every swept dataset DROPs, most of them on (b2) rather than on (a).
By PREREG §5 that means **"fewer than 2 KEEP ⇒ the long-horizon section dies"** fires.
Specifically I predict clause (b2) fails (tripwire ≥ HEPA − 0.02 at some rung) on **ETTm1, Weather,
BeijingAQ and GECCO**, and that PSM fails clause (a) (its curve is flat because it ships at 100 and
its time-to-last-similar-regime is 53 steps).
*Falsified by:* any dataset where HEPA beats the tripwire by > 0.02 at every rung AND the longest
rung beats 512 by more than the 512 seed spread.

## P3 · The task-§8 whole-premise falsification
Task §8: the premise dies if the tripwire tracks or beats HEPA **at every rung on every dataset**.
I predict this **fires on 4–6 of the 6 swept datasets but NOT on all of them** — I expect PSM and/or
GECCO to leave HEPA a > 0.02 margin somewhere. Recorded as a prediction that can fail either way.

## P4 · The information-age slice (§5, gated)
I predict the old-precursor slice (decisive precursor > 512 steps) is **underpowered (n < 10 events)
on Weather and MBA**, and that on the datasets where it is powered the 512→longest gain is **equal or
larger in the YOUNG slice than in the old slice** — i.e. a capacity effect, not a memory effect.
*Falsified by:* an old-slice gain that exceeds the young-slice gain by more than the seed spread on
any dataset.
