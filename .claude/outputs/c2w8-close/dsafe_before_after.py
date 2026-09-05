"""Item (v) before/after: refusal rate under the SIZING vs the STORE population.

⛔ A regression cell, NOT a science cell. Toy stream (the census's own pytest
rig), 3 seeds, everything else identical. The refusal rate is REPORTED, never
tuned to a target.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/user/Desktop/CHLU-c2w8close")
from chlu.config import get_default_config                      # noqa: E402
from chlu.experiments.exp_well_lifecycle import (               # noqa: E402
    apply_quick, run_census_cell,
)


def toy_data(n_per_class=30, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(10, dim)) * 3.0
    X = np.concatenate([centers[c] + rng.normal(size=(n_per_class, dim)) * 0.35
                        for c in range(10)]).astype(np.float32)
    y = np.concatenate([np.full(n_per_class, c) for c in range(10)])
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(X))
    return (X[:cut], y[:cut]), (X[cut:], y[cut:])


def cfg_for(pop):
    cfg = get_default_config()
    apply_quick(cfg)
    w = cfg.experiment_well_lifecycle
    w.addr_dim, w.capacity, w.well_budget = 3, 6, 3
    w.n_offer_per_task, w.write_steps, w.read_steps = 6, 20, 60
    w.address_steps, w.read_batch = 40, 4
    w.capture_dirs, w.capture_bisect_steps = 4, 3
    w.d_safe_population = pop
    cl = cfg.experiment_cl_entry
    cl.n_tasks, cl.n_train_per_task, cl.n_test_per_task = 3, 20, 10
    cl.n_fit_region, cl.n_fit_pool = 150, 80
    return cfg


rows = {}
data = toy_data()
for pop in ("sizing", "store"):
    cells = [run_census_cell(cfg_for(pop), seed=s, data=data, verbose=False)
             for s in (0, 1, 2)]
    rows[pop] = {
        "d_safe": [c["geometry"]["d_safe"] for c in cells],
        "median_nn_sizing": [c["geometry"]["median_nn_task1"] for c in cells],
        "median_nn_store_population": [
            c["geometry"]["median_nn_store_population"] for c in cells],
        "ratio_pop_over_sizing": [
            c["geometry"]["spacing_population_block"]["ratio_population_over_sizing"]
            for c in cells],
        "refusal_rate": [c["stream"]["refusal_rate"] for c in cells],
        "n_offered": [c["stream"]["n_offered"] for c in cells],
        "n_refused": [c["stream"]["n_offered"] - c["stream"]["n_admitted"]
                      for c in cells],
        "n_live_end": [c["stream"]["n_live_end"] for c in cells],
        "G_DRIFT_two_sided_ratio": [
            c["census"]["G_DRIFT_two_sided"]["ratio"] for c in cells],
        "G_DRIFT_two_sided_pass": [
            c["census"]["G_DRIFT_two_sided"]["pass"] for c in cells],
        "cue_sigma_over_codebook_spacing": [
            c["g_addr"].get("cue_sigma_over_codebook_spacing") for c in cells],
        "cue_sigma_over_sizing_spacing": [
            c["g_addr"].get("cue_sigma_over_sizing_spacing") for c in cells],
        "frac_never_read_settle_gated": [
            c["usage"]["frac_never_read"] for c in cells],
    }
print(json.dumps(rows, indent=1, default=float))
with open("/Users/user/Desktop/CHLU/.claude/outputs/c2w8-close/"
          "dsafe_before_after.json", "w") as f:
    json.dump(rows, f, indent=1, default=float)
