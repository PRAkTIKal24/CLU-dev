# PREREG — `bprime-referee-closures` (CLU n=9 column + 5 figure renders)

Filed **before** running the n=9 CLU aggregation (protocol §5 pre-registration rule; the acceptance
criterion contains a measured **verdict** — the rescue gate — and measured margins, so predictions are
committed here first).

Timestamp: filed 2026-08-01, before executing `.claude/scratch/bprime-referee-closures/n9_clu_column.py`.

---

## 0. DISCLOSURE — what I had already seen when writing this

Full honesty about information leakage into the predictions:

1. **Seeds 0/1/2 CLU values are public** — they are the banked column already printed in `draft-r2.md`
   §4.1.1 (`full −0.5261 ± 0.0863`, `launder −0.4472`, `blank −0.4221`, `+0 B margin −0.3180 ± 0.0804`,
   `same-keys null −0.8175`), and `BANKED_CLU` in `chlu/experiments/exp_bprime_rivals.py` carries the
   per-seed triples: `full = [−0.682608, −0.384693, −0.511032]`,
   `launder = [−0.496261, −0.413103, −0.432255]`, `blank = [−0.438906, −0.404201, −0.423079]`.
2. ⚠ **I saw ONE unpublished cell while inspecting the artifact schema**: seed 3's
   `clu_reproduction = {full −0.4215518, launder −0.2504672, blank −0.3951167}` (printed by a
   `list(keys)` probe on `seeds3to8/exp_bprime_rivals_metrics.json`), and the seed-0 launder-set keys
   (`settle_deleted`, `same_keys_null`, `knn2_mean_+0B`, `knn2_idw_+0B`, `raw_table_mean_+0B`).
   **Seeds 4–8 are unseen at filing time.** Predictions below use seeds 0–3 only; 6 of 9 cells are blind.
3. **No CLU number is re-measured by this task.** The seeds 3–8 CLU cells were *already computed* by the
   F3 rider (`run_rivals_cell` runs the shipped CLU write/read path on every cell regardless of the rival
   grid) and merely never aggregated. This task is a **re-aggregation**, so the prereg is about what the
   pooled statistics and the gate verdict will say, not about a fresh run.

## 1. Aggregation rule I commit to (fixed before looking)

- Cells: `.claude/outputs/bprime-rivals-f3/{run400,seeds3to8}` (F3 primary path) and
  `{repro_c2w4,repro_c2w4_s3to8}` (first-pass code path), `family=aggregate`, `arm=base`,
  non-degenerate, seeds 0–8 ⇒ **n = 9**.
- `full`, `launder`, `dividend` come from the **shipped `audit_table`** (`clu_reproduced` block) so the
  rule is byte-identical to the published one; every other column uses the *same* convention as rider 1
  (`n9_aggregate.py`): sample sd `ddof=1`, `SE = sd/√n`, and every *margin/lift* is computed **paired per
  seed** and then averaged (never as a difference of column means).
- Rescue gate (B.5, shipped form): `lift = full − blank` per seed; **RESCUED iff `mean(lift) > 2·SE(lift)`**.
- `+0 B margin`: **primary** = per-seed arg-max over the *exclusive* `+0 B` reader set
  (`{knn2_mean_+0B, knn2_idw_+0B, raw_table_mean_+0B}`), margin = `full − best`, paired — this is the
  rivals' own per-cell rule (`zero_byte_margin.signed_margin_full_minus_sub`). **Secondary** = the banked
  convention (one reader name fixed at the column level).
- `raw-table margin`: `full − max(exclusive +0 B set ∪ {settle_deleted})` per seed, paired — the exact
  candidate set the rivals' `raw_table_control` uses (for the CLU these ARE its own raw readers).

## 2. Registered predictions (committed numbers)

| # | quantity at n = 9 | prediction | derivation |
|---|---|---|---|
| **P1** | **rescue-gate verdict** | ⛔ **NOT RESCUED** (probability of RESCUED ≤ 10 %) | both lifts I can see are **negative**: n=3 gives `−0.5261 − (−0.4221) = −0.1040`; seed 3 gives `−0.4216 − (−0.3951) = −0.0265`. A negative mean can never clear `> 2·SE`. For RESCUED, seeds 4–8 would have to average `lift ≳ +0.10` and reverse the sign of the pooled mean. |
| **P2** | `mean(lift = full − blank)` | **−0.09 ± 0.06**, i.e. in **[−0.20, +0.02]**; and `|mean| < 2·SE` OR mean < 0 | inverse-variance-free average of the 4 visible lifts (−0.2437, +0.0195, −0.0880, −0.0265 ⇒ mean −0.0847) extended with regression toward 0 for the unseen 5 |
| **P3** | `dividend = full − launder` | **negative**, mean in **[−0.22, −0.03]**, point estimate **−0.11** | n=3 −0.0789; seed 3 −0.1711 (paired) |
| **P4** | `+0 B margin` (primary rule) | **negative and ≥ 2 SE below zero**, mean in **[−0.40, −0.18]**, point estimate **−0.28** | n=3 −0.3180 ± 0.0804 (≈4 SE); seed 3's `full` is 0.10 *better* than the n=3 mean, so the pooled margin should shrink slightly toward zero |
| **P5** | `raw-table margin` vs `+0 B margin` | **identical on ≥ 8 of 9 seeds** (so the two column means agree to ≤ 0.01) | the raw candidate set adds only `settle_deleted` (≈ −0.45), which is far below the `knn2` readers (≈ −0.21); the arg-max is therefore unchanged unless a seed's launder beats its knn2 readers |
| **P6** | `full − same_keys_null` | **positive and > 2 SE** (the CLU sits **above** its same-keys null, unlike both TTT arms — pilot-placement-probe R5) | n=3: `−0.5261 − (−0.8175) = +0.2914` |
| **P7** | `full` | mean in **[−0.56, −0.42]**, point estimate **−0.50** | seeds 0–3 mean = −0.4999 |
| **P8** | CLU per-seed values are **bit-identical between the two code paths** on all 9 seeds | ✅ identical to full float precision | the CLU write/read path in `run_rivals_cell` is keyed on `PRNGKey(seed)`/`PRNGKey(seed+1)` and never touches the rival tuning grid; seeds 0 and 3 already match digit-for-digit across `run400`/`repro_c2w4` |
| **P9** | byte ledger (`5456 B / 100 B / 54.56×`) is **constant across all 9 seeds** | ✅ constant (unlike the TTT arms' state bytes, R4) | the CLU's `n_live`/`A` are set by the family's admission, which was 5/8 on seeds 0–2; ⚠ if admission varies by seed this prediction FAILS and the ledger must ship as a per-seed set, exactly as R4 forced for the TTT arms |

**Falsifiers / what would make me report a finding rather than a confirmation:** P1 flipping to RESCUED
(would mean the n=3 "✅" was right for the wrong reason and MF-1's premise partly dissolves); P3 turning
positive (the store *would* be paying a dividend over its own launder at power — a materially different
paper sentence); P5 failing (the launder becomes the binding raw reader on some seed); P9 failing (the CLU
byte column becomes seed-dependent and the paper's `5456/100/54.56×` must be re-labelled).

**Not falsified by anything here:** the CLU losing to a raw table read at its own bytes — that is the
metric-native-ceiling theorem on a metric-native protocol and is the paper's own headline (§4.2).

## 3. Figures (no prediction; provenance discipline instead)

The five App-K renders make **no new number**. Registered constraint: every plotted value must trace to a
named JSON field, tabulated figure-by-figure in the report; any value I cannot trace is **omitted from the
figure and declared**, never approximated or re-derived by eye.
