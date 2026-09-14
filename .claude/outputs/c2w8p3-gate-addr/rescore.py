"""Re-score a banked C2W8 pass-2 arm on G-ADDR. One (arm, seed, scale) per run.

usage: rescore.py <armA|armB> <seed> <addr_scale_mult> <out.json>
"""
import json
import sys
import time

from chlu.config import get_default_config
from chlu.experiments import exp_well_lifecycle as ewl

arm, seed, mult, out_path = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
width_frac = float(sys.argv[5]) if len(sys.argv) > 5 else None
payload_scale = float(sys.argv[6]) if len(sys.argv) > 6 else None

cfg = get_default_config()
cfg.experiment_well_lifecycle.addr_scale_mult = mult
if width_frac is not None:
    # ⚠ the BANKED arm-A census ran at 1.5; the config DEFAULT is 0.5 (the pilot
    # cell). Not setting this silently re-scores a different store.
    cfg.experiment_capture_arm_a.atom_width_frac_spacing = width_frac
if payload_scale is not None:
    # ERRATA §3 cell S-pay: co-scale the PAYLOAD channel with the address scale,
    # which is the one absolute quantity the "scale-only" control leaves fixed.
    cfg.experiment_well_lifecycle.payload_scale = payload_scale
t0 = time.time()

if arm == "armA":
    from chlu.experiments.exp_capture_armA import gate_legs, own_foreign, run_cell

    cell = run_cell(cfg, seed, verbose=True)
    legs = gate_legs(cell)
    of = own_foreign(cell)
elif arm == "armB":
    from chlu.experiments.exp_capture_armB import gate_legs, own_foreign, run_arm_b_cell

    cell = run_arm_b_cell(cfg, seed, verbose=True)
    legs = gate_legs(cell)
    of = own_foreign(cell)
else:
    raise SystemExit(f"unknown arm {arm}")

res = {
    "arm": arm, "seed": seed, "addr_scale_mult": mult,
    "atom_width_frac_spacing": width_frac,
    "payload_scale_override": payload_scale,
    "g_addr": cell["g_addr"],
    "pass2_gate_legs": legs,
    "own_foreign_repaired_estimator": of,
    "census_P": cell["census"]["P"], "census_M": cell["census"]["M"],
    "theta_att": cell["census"]["theta_att_block"],
    "usage": cell["usage"],
    "self_probe": cell["self_probe"],
    "geometry": cell["geometry"],
    "flags": cell["flags"],
    "bytes": cell["bytes"],
    "depth_raw_median": cell["census"]["depth_raw_median"],
    "wall_s": float(time.time() - t0),
    "cell": cell,
}
with open(out_path, "w") as f:
    json.dump(ewl._jsonable(res), f, indent=2)
g = cell["g_addr"]
print(f"[{arm} seed {seed} scale x{mult}] A1={g['A1']['correct_basin_rate']:.4f} "
      f"thr={g['A1']['threshold']:.4f} A2={g['A2']['never_addressed_frac']:.4f} "
      f"A3a={g['A3']['A3a_cue_margin']:+.4f} A3b={g['A3']['A3b_stream_margin']} "
      f"PASS={g['gate_addr_pass']} -> {out_path}", flush=True)
