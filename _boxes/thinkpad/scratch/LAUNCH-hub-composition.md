# ADDENDUM — `hub/composition`
**Read `LAUNCH-COMMON.md` first.** Then `.claude/tasks/hub-composition.md` is your task.

- **Slug:** `hub-composition` · **Branch:** `hub/composition`
- **Worktree:** `git worktree add /c/Users/PratikJawahar/chlu/CHLU-hub-composition -b hub/composition HEAD`
  ⛔ NOT `ae907466` — see COMMON §1a.
- **Core budget: `OMP_NUM_THREADS=5` / `MKL_NUM_THREADS=5` / `OPENBLAS_NUM_THREADS=5`.**

⚠ **`.claude/outputs/hub-composition/` ALREADY EXISTS.** It holds `.workspace-write-probe.json`, written
by the **verifier**, not by a previous hub. ⛔ Do not delete it (§0.1). **There is no prior run to
resume** — do not mistake it for one.

## ⛔⛔ YOU ARE THE HUB MOST LIKELY TO BREAK THE HARD CONSTRAINT — read COMMON §4 twice
Your work naturally invites rebuilding an input: a re-encode, or a rebuilt store. **The rival rows are
frozen and Mac-produced.** Rebuilding an input that feeds a scored comparison breaks comparability with
them **even if your own arithmetic is perfectly self-consistent**.

⭐ **Load frozen artefacts and render. Do not rebuild.**
⚠ If a line of attack requires **extending the feature cache**, ⛔ **STOP and report it as a blocker**
rather than proceeding. That is a decision for the humans, not for you.
(`exp_gen_store._assert_cache_reencodes(will_extend=…)` is the gate; it binds when a run will EXTEND the
cache. It is there to catch exactly this.)

## The shape of the work — and the timing trap
~**72 independent units**: 6 arms × 3 seeds × the pre-registered temperature grid
`temp ∈ {0.0, 0.005, 0.05, 0.5}`. That fan-out is exactly what this box's cores are for.
⭐ Run several units concurrently **within your 5-core budget** (e.g. 2 units × 2–3 threads each).
⛔ Never 5 units × 5 threads — oversubscription runs **slower**, not faster.

⚠⚠ **Nobody has ever timed `exp_gen_cell` on this machine.** The only anchors are `exp_gen_baselines` at
~0.85 h/seed measured on an **8-core Mac**, and `exp_gen_atom_mean` at a stated ~4 h / ~23 CPU-h.
⭐ **Time ONE `--quick` unit first and extrapolate before committing to a grid.** A 12-hour budget spent
discovering the unit cost at hour 9 is a wasted run, and this is the most likely way you waste it.

CLI surface, confirmed working on this box:
```
--dataset --pretext --seed --arm --out --quick --config
--stage {cell,aggregate,all,encoder,decoder,store,dream,certificate,baselines,compose}
```

⚠ **Declared property of the pin, ⛔ not a bug for you to fix:** `GenCellConfig.arms` defaults to **five**
arms while the module constant `ARMS` has **six**. `dynread` must be selected explicitly with
`--arm dynread`. It was left that way deliberately — adding an expensive, *losing* arm to every default
cell run imposes compute nobody asked for.

⚠ Also declared and ⛔ not yours to touch: `GenCellConfig.lace_selection_source` still points at
`gen-arms-a7-wiring/runs/historical_688128`, the **broken-split** LACE selection. Re-pointing it would
move a rival's numbers and needs its own scoped re-score.
