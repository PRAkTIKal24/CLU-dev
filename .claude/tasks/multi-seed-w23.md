# Task: multi-seed-w23 — error bars on every w23 headline before any of it ships (w24)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/multi-seed-w23.md` · **Branch:** none expected (analysis + re-runs; flags defects, does not fix them)
- **Read first:** `.claude/AGENT_PROTOCOL.md` · the four w23 engineer reports (`phi-read-in.md`, `retry-compute-study.md`, `dimension-aware-budget.md`, `controller-mvp.md`) and their `PREREG.md` files · `.claude/negative_results.md` **N89–N94** · `.claude/claims_matrix.md` v2.2
- **All four experiments are merged and on `main` @ `5e466c0` (612 green) with CLI entry points — re-run them, do not rebuild them.**

## Why
**[HEAD RULING, direction queue point 7] Multi-seed before any paper number.** Nearly every w23 headline is **seed 0**. The margins are large and the verdicts are probably safe — but four tier-A registry entries and a claims-matrix version bump now rest on single-seed runs, and referees will ask. This task converts w23's verdicts into quotable numbers with error bars, or tells us which ones soften.

## Item 1 — `phi-read-in` (currently seed 0) → **≥5 seeds**
`chlu exp-phi-read-in`. Re-run the capacity + noise sweeps, both φ arms, both datasets. Report with CIs:
- ⭐ **the laundering verdict** (N89, tier A): does "CLU-in-φ never beats kNN-in-φ, max margin 0.000" hold across seeds? **This is the wave's headline negative — it must be seed-robust.**
- the **CIFAR** CLU-vs-Hopfield-in-φ margin ⚠ **and the MNIST high-load cell where Hopfield is ahead** (0.957 vs 0.871) — the CIFAR scoping is now binding wording (CM-23 amendment 1); confirm or correct it.
- the retry-confidence AUROC (0.845–0.988).

## Item 2 — `retry-compute-study` (currently seed 0) → **≥5 seeds**
`chlu exp-retry-compute`. The 8-cell grid. Report with CIs:
- ⭐ **the mechanism attribution** (N90's surviving positive): are random-kick and ensemble-of-k **still dead flat** in all cells? The ±3pp falsification bars are pre-registered — apply them per seed, not just on the mean.
- the gated lift (+6.6…+76.2 pp) and the saturation multiplier (×1.2–1.8).
- the **NN gap** (−3.5…−42.2 pp). ⚠ The Head+Advisor direction doc quotes *"within 3–13pp of the NN ceiling"*; the measured range is wider. **Pin the true range with CIs** — this number is headed for an appendix.

## Item 3 — `dimension-aware-budget` frontier (currently 2-seed re-checks) → **≥3 seeds at 2× budget**
`chlu exp-designed-mechanism`. The budget-adequate walls **d=4 = 16** and **d=5 ≥ 32** rest on 2-seed re-checks, and the write is **seed-fragile exactly at the 0.9 rung** (3-seed 0.876 vs 2-seed 0.93–0.98). These two cells carry the "`2^d`, base 2 — geometry vindicated" half of N92 (tier A). Re-run them at 2× atoms, ≥3 seeds. Also confirm the ceiling cells (d=6 K=64, d=8 K=64) hold.

## Item 4 — `controller-mvp`: **verify, do not re-run**
Already 5 seeds, paired proposal sequences. Confirm the reported ±values are seed std over those 5, and that the headline cells (per-admitted 1.000; per-offered 0.081 fixed / 0.669 sized) carry them. If so, mark it multi-seed-clean and spend the compute elsewhere.

## Acceptance
A table of **every w23 headline number with mean ± CI and seed count**, and an explicit three-way triage:
1. **SURVIVES** — quotable as-is;
2. **SOFTENS** — direction holds, magnitude moves (state the corrected number);
3. **⛔ FLIPS** — the verdict does not survive multi-seed (escalate immediately; this would touch a tier-A registry entry and the matrix).
Propose registry wording for anything in (2) or (3); **the curator applies it, you do not edit `negative_results.md`.** State wall-clock cost per item — if the full ≥5-seed φ+retry grid is prohibitive, prioritise **Item 1's laundering verdict and Item 2's mechanism controls** (the two tier-A load-bearing results) and say plainly what you dropped.

## ⚠ Notes
- Re-use the main venv; `results/` is not committed (repo precedent).
- Report any **defect** you find to the Hub for the engineer — do not fix `chlu/` code yourself.
- Standing caveat N94: state the epoch count of every diagnostic; under-trained fits are their own regime.
