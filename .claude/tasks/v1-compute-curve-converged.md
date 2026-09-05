# V1 — the gated-compute-allocation figure AT CONVERGED ACCURACY (a re-run, not a replot)

**Scoped by the V1 Shorts Advisor on the Head's instruction, 2026-08-29.** Charter basis: Add.102 (the 400-epoch figure banked) + this addendum.

**Agent:** `results-analyst` (Bash-capable; runs experiments and produces paper figures).
**Writes:** `.claude/outputs/v1-compute-curve-converged/**` and, on acceptance only, one PNG into `.claude/NIPSsubmission/v1-ttcl/figs/`.
**Report:** `.claude/outputs/v1-compute-curve-converged.md`

---

## 0. ⛔ WHY THIS IS A RUN AND NOT A REPLOT — verify this yourself before starting

The shipped figure (`figs/fig_compute_allocation.png`) is the **400-epoch** `v1-pivot/full` run: accuracies **0.85 / 0.53 / 0.28** at kv16/24/32. The Head wants the same axes where accuracy is **≥ 0.8**.

⭐ **The converged data exists but CANNOT produce this figure, and the reason is specific:** `scratch/regime-remap-2000ep/runs/*.json` holds **198 cells at 500/1000/2000/4000 epochs** — and **every value in them is a SCALAR** (`clu_gate_acc`, `clu_full_acc`, `savings`, `clu_gate_cost`, `fidelity`, …). ⛔ **Advisor-verified: zero list-valued keys across the sample.** A compute-allocation curve needs the **per-query × per-stage** arrays (`score_stages`, `correct_stages`, `cost`, shape `(T,S)`) that `_simulate_tau_policy` sweeps. Those were never banked for the converged cells. ⇒ **two points per cell are recoverable, a curve is not.**
**Confirm this yourself in one command before running anything** (if it returns list-valued keys, STOP and report — the run may be unnecessary):
```
python -c "import json,glob;print(sorted({k for f in glob.glob('.claude/scratch/regime-remap-2000ep/runs/*.json') for k,v in json.load(open(f)).items() if isinstance(v,list)}))"
```

## 1. The job

Re-run **`exp_v1_calibration`** — the experiment that *does* save the stage arrays — at **converged training**, then render the same figure from the new artifacts.

- **Cells (Head-chosen, all ≥0.8 at 2000 ep in the banked regime table):** `(N=128, kv=32)` · `(N=256, kv=64)` · `(N=384, kv=96)`. Their 2000-ep gate accuracies are **1.00 / 0.99 / 0.91** with savings **9.9× / 9.5× / 6.2×** — ⭐ **exactly the three step-reduction numbers §4.1 already prints**, which is what makes this figure the plot of the paper's own claim.
  ⚠ If the Head instead names `kv ∈ {16,32,64}`, that is `(128,16)`, `(128,32)`, `(256,64)` — same procedure, and `(128,16)` should be run additionally rather than as a substitute if time allows.
- **Epochs:** `cfg.train_epochs = 2000` (the calibration experiment reads `cfg.train_epochs`; the regime-remap `driver.py` is the working precedent for driving epochs per cell). ⚠ **`exp_v1_calibration.py:408` and `:1262` clamp `train_epochs` to ≤120 on a `quick` path — ⛔ do NOT run quick, and verify the clamp is not in force on your path before trusting any result.**
- **Seeds:** 5 (matching the shipped figure). **Levels** go in `cfg.calib_difficulty_levels`; `calib_n_seeds=5`.
- ⛔ **Change nothing else.** `calib_n_policy_taus` stays **25**, `calib_features` stays `r_margin`, `calib_p_exit` stays `0.5` — the shipped figure's settings.

**Cost estimate (measured, not guessed):** the regime runs at 2000 ep took a **median 95–108 s per cell** on laptop CPU (N128/kv32 95 s · N256/kv64 106 s · N384/kv96 108 s). `exp_v1_calibration` does strictly more per cell (calibration heads, probes, LTT, sweeps), so budget **several minutes per cell-seed**, i.e. **~1–3 h** for 3 cells × 5 seeds. ⚠ Report actual wall time; if a cell exceeds ~30 min, stop and report rather than burning the budget.

## 2. The figure

Render with the **same code path as the shipped figure** — `.claude/scratch/v1-compute-fig/render.py`, which copies `_simulate_tau_policy` **verbatim** from `chlu/experiments/exp_v1_gate.py:321` and the panel construction from `exp_v1_calibration.py:878–928`, and takes only `summary.json` + `metrics.npz`. Point it at the new results.
⛔ **NO Hopfield line — Head ruling 2026-08-29: Hopfield is cut from V1's text, so it does not appear on any V1 figure.** `render.py`'s `render(..., hopfield=False)` already does this.

## 3. ⛔ Acceptance criteria

1. **Every panel's gate accuracy ≥ 0.8.** ⚠ If a cell lands below, that is a **finding, not a failure** — report it and do not quietly swap the cell for an easier one.
2. ⭐ **The learned operating points must reproduce the banked regime numbers** (savings ≈ 9.9× / 9.5× / 6.2×, accuracy ≈ 1.00 / 0.99 / 0.91). ⛔ **A material disagreement is a STOP-and-report, never something to tune away** — two independent runs of the same configuration disagreeing is the finding.
3. **A data tap**: peak accuracy, the cost at peak, the operating point, and always-full for every panel — the numbers any caption will quote.
4. Flag-provenance table (commit, seeds, every non-default flag, JAX version) per protocol §5; PREREG per §5 is **not** required (no ratio/law is being measured — this is a re-measurement at a new epoch budget).

## 4. ⚠ REPORT THIS EXPLICITLY — it decided the last figure's reading

On the 400-epoch figure the **naive margin-gated arm peaks ABOVE the calibrated learned gate** at the kv16 knee (0.9168 vs 0.8988 at the same compute), and edges it at kv24/kv32. **Measure and report the same comparison at convergence.** ⭐ If the learned gate leads at convergence, that is a materially better figure for the paper; if the naive arm still leads, the Head needs to know before it ships. ⛔ Report it either way — do not select the cells or the axes to make it come out.

## 5. Caption fences (binding, carry them whatever the numbers do)
⛔ **CM-23(b) is shape-only; the absolute-dominance reading is RETRACTED (N90); N95 confirms the loss survives with headroom deliberately built in ⇒ saturation is NOT an available explanation.** ⛔ Claim form is **"same accuracy at ~1/Nth the steps," NEVER "more compute buys more accuracy."** ⚠ At 2000 ep the paper's own grid has gate accuracy **equal to** full-budget accuracy — the claim is the step reduction, not an accuracy gain. Report honestly if any panel is non-monotonic.

## 6. ⚠ Hazards
⛔ `grep` here is `ugrep`: use `/usr/bin/grep`; count with `grep -o … | wc -l`, never `grep -c`. ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — per-file only. ⚠ JAX cold start on this machine is pathologically slow (~20 min); keep the session warm and do not mistake it for a hang. ⚠ Positive-control every negative, **both polarities**.

## DIAL DECLARATION
**Dial:** compute-adaptive reads. **Laundering control:** the naive margin-gated and raw-R-gated arms are on the same axes and must be plotted, never dropped. **Falsifies:** the gate showing no step reduction at converged accuracy, or a naive arm dominating it at equal compute. **Does NOT falsify:** losing to an oracle or to a one-shot associative memory on a metric-native protocol — that is the metric-native-ceiling theorem, not news.
