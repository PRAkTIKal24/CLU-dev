"""Price the encoder fit BEFORE the sweep (task §Budget).

Times a small number of fit steps per arm on the real CIFAR task-1 pool and
extrapolates to the shipped ``enc_steps``. Also prints the *measured* φ parameter
count per arm (the §A4.3 ledger term) and the store's memory floats, so the arm
cut can be declared on numbers rather than taste.
"""
import json
import os
import time

import numpy as np

from chlu.config import get_default_config
from chlu.experiments.exp_cl_entry import apply_cifar10, build_cl_stream
from chlu.experiments.exp_phi_read_in import build_read_in, read_in_param_floats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

config = get_default_config()
apply_cifar10(config)
cfg = config.experiment_cl_entry
cfg.n_fit_region = 25000
cfg.n_fit_pool = 6000
cfg.phi_regimes = ["task1_only"]

t0 = time.time()
stream = build_cl_stream(cfg, 0)
print(f"[price] stream built in {time.time() - t0:.1f}s; "
      f"fit pool {stream['fit_pool_task1_only'].shape}", flush=True)

rows = []
PROBE_STEPS = 50
for arm, phi_dim in (("pca", 32), ("randconv", 256), ("convae", 256), ("simclr", 256)):
    c = cfg
    c.phi_arm, c.phi_dim = arm, phi_dim
    old_steps = c.enc_steps
    c.enc_steps = PROBE_STEPS if arm in ("convae", "simclr") else old_steps
    t = time.time()
    phi, prov = build_read_in(arm, c.dataset, stream["train_X"][0],
                              stream["fit_pool_task1_only"], c, 0)
    fit_s = time.time() - t
    t = time.time()
    _ = np.asarray(phi(stream["test_X"][0]))
    read_s = time.time() - t
    pf = read_in_param_floats(phi)
    per_step = (fit_s / PROBE_STEPS) if arm in ("convae", "simclr") else 0.0
    rows.append({
        "arm": arm, "phi_dim": phi_dim, "probe_steps": c.enc_steps,
        "probe_fit_seconds": fit_s, "seconds_per_step": per_step,
        "projected_8000_min": per_step * 8000 / 60.0,
        "projected_20000_min": per_step * 20000 / 60.0,
        "read_500_seconds": read_s,
        "phi_param_floats": pf, "provenance": prov,
    })
    print(json.dumps(rows[-1], default=float), flush=True)
    c.enc_steps = old_steps

with open(os.path.join(OUT, "encoder_price.json"), "w") as f:
    json.dump({"rows": rows, "probe_steps": PROBE_STEPS}, f, indent=2, default=float)
print("[price] done", flush=True)
