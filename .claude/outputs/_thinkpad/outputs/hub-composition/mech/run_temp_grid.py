"""⭐⭐ TARGET 3 — DREAMING AT THE PRE-REGISTERED FINITE-TEMPERATURE GRID.

⛔ The grid is `temp ∈ {0.0, 0.005, 0.05, 0.5}` EXACTLY, as pre-registered for this cell.
⛔ It may not be widened after a result is seen, and it is not widened here.
β (`request_strength`) = 0.0 is DREAMING; β = 1.0 runs beside it as the steered control
on the SAME chain, so any difference isolates the request energy.

⛔ `val` requests only. ⛔ B never enters. Resumable: one JSON per unit.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np
from pathlib import Path
from chlu.config import gen_dynamical_read_config
from chlu.experiments.exp_gen_dynamical_read import build_landscape
from mech_driver import build_context, arm_settle, score

TEMPS = (0.0, 0.005, 0.05, 0.5)          # ⛔ pre-registered. Not widened.
BETAS = (0.0, 1.0)                        # 0.0 = DREAMING; 1.0 = the steered control
BUDGET = os.environ.get("TG_BUDGET", "A")
STEPS = int(os.environ.get("TG_STEPS", "1000"))
WHICH = os.environ.get("TG_WHICH", "val")

TAG = os.environ.get("TG_TAG", "")
led = Path(os.environ['OUT']) / "mech" / f"temp_grid_{BUDGET}_{WHICH}_s{STEPS}{TAG}"
led.mkdir(parents=True, exist_ok=True)
ctx = build_context(0)
c = gen_dynamical_read_config(ctx["cfg"])
t = time.time()
built = build_landscape(ctx["cfg"], BUDGET, ctx["X_store"], ctx["lab_store"], ctx["wc"],
                        ctx["feat_shape"], ctx["cards"], 0, log=lambda *a: None)
print(f"### landscape {BUDGET} in {time.time()-t:.0f}s", flush=True)
land, book = built["land"], built["book"]
atoms = np.asarray(ctx["lab_store"], np.int64)[np.asarray(built["store"].atom_rows, np.int64)]
rows, X, lab = ctx["sets"][WHICH]

for beta in BETAS:
    for temp in TEMPS:
        unit = led / f"b{beta:g}_T{temp:g}.json"
        if unit.exists():
            print(f"[skip] {unit.name}", flush=True); continue
        t = time.time()
        r = arm_settle(ctx, land, book, WHICH, temp, STEPS, beta, 0, c)
        rep = score(ctx, r["feats"], lab, atoms)
        rep0 = score(ctx, r["x0"], lab, atoms)      # the launch, same chain
        rec = {"budget": BUDGET, "which": WHICH, "beta": beta, "temp": temp,
               "steps": STEPS, "n": rep["n"],
               "settled": rep, "launch": rep0,
               "convergence": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in r["convergence"].items()
                               if not isinstance(v, np.ndarray)},
               "separation": {k: v for k, v in r["separation"].items() if not isinstance(v, np.ndarray)},
               "separation_launch": {k: v for k, v in r["separation_launch"].items() if not isinstance(v, np.ndarray)},
               "settle_cfg": r["settle_cfg"], "wall_s": time.time() - t}
        json.dump(rec, open(unit, "w"), indent=2, default=str)
        print(f"beta={beta:g} T={temp:<6g} SETTLED distinct={rep['distinct']:4d} "
              f"novel={rep['distinct_novel']:4d} valid={rep['distinct_valid']:4d} "
              f"V^N={rep['distinct_valid_novel']:4d} | LAUNCH distinct={rep0['distinct']:4d} "
              f"valid={rep0['distinct_valid']:4d} | sep={rec['separation']['mean_pairwise_rms_distance']:.4f} "
              f"(launch {rec['separation_launch']['mean_pairwise_rms_distance']:.4f}) "
              f"collapsed={rec['separation']['frac_pairs_collapsed']:.3f} "
              f"identical={rec['separation']['all_states_identical']} "
              f"({rec['wall_s']:.0f}s)", flush=True)
print("done")
