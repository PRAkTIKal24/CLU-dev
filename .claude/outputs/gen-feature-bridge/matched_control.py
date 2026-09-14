"""POST-HOC (not pre-registered) diagnostic: is the B drop COMPOSITIONAL or PHOTOMETRIC?

The Montero held-out cell is `all three hues >= 0.8 AND scale >= 1.1 AND
shape == cylinder AND orientation > 20`. All three hues >= 0.8 means floor, wall
and object are all in the same magenta band, so object/background CONTRAST is
near-minimal there. A photometric explanation of the ceiling_B drop is therefore
confounded with the compositional one.

Control: A_matched = the SAME photometric corner with shape != cylinder (seen
combinations, and any row used in fitting is removed). If ceiling_A_matched is
close to ceiling_B, the drop is photometric, not compositional.
"""
import json, sys
import numpy as np

sys.path.insert(0, "/Users/user/Desktop/CHLU-gen-bridge")
from chlu.config import get_default_config
from chlu.core.gen_bridge import features_for_rows, load_feature_cache, stream_images
from chlu.data import shapes3d as s3d
from chlu.eval.gen_probe import factor_labels, measure_ceiling, probe_accuracy
from chlu.experiments.exp_gen_bridge import select_rows
from chlu.utils.checkpoints import load_model

RUN = "/Users/user/Desktop/CHLU/.claude/outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0"
ROOT = None  # -> $CHLU_GEN_DATA_ROOT
cfg = get_default_config().gen_bridge

split, labels = s3d.build_split(ROOT)
grid = split.grid
rows = select_rows(split, cfg, 0)
v = labels  # (480000, 6) realised factor values, file order == grid row order

photometric = (v[:, 0] >= 0.8) & (v[:, 1] >= 0.8) & (v[:, 2] >= 0.8) & (v[:, 3] >= 1.1) & (v[:, 5] > 20)
matched = np.where(photometric & (v[:, 4] != 1.0))[0]
matched = np.setdiff1d(matched, rows["train"])           # never seen in fitting
assert np.intersect1d(matched, split.test_rows).size == 0  # and never held out
print(f"photometric corner rows       : {int(photometric.sum())}")
print(f"  of which held out (B)       : {split.test_rows.size}")
print(f"  A_matched (shape != cylinder, minus fitted rows): {matched.size}")

dec = load_model(f"{RUN}/decoder.pkl")
probe = load_model(f"{RUN}/probe.pkl")
crows, feats, man = load_feature_cache(RUN, verify=True)
print(f"feature cache scope={man['scope']} n_rows={man['n_rows']} (full grid: {split.n_total})")

out = {}
for tag, r in (("A_matched", matched), ("B", rows["eval_b"]), ("A_random", rows["eval_a"])):
    f = features_for_rows(crows, feats, r)
    lab = factor_labels(grid, r)
    real = stream_images("shapes3d", r, ROOT, int(cfg.cache_block))
    ceil = measure_ceiling(dec, probe, f, lab, grid.names, int(cfg.eval_batch))
    pr = probe_accuracy(probe, real, lab, grid.names, int(cfg.eval_batch))
    out[tag] = {"n": int(r.size), "ceiling": ceil, "probe_on_real": pr,
                "object_bg_contrast": float(np.mean(np.abs(real - real.mean(axis=(2, 3), keepdims=True))))}
    print(f"\n{tag}: n={r.size}")
    print(f"  ceiling aggregate = {ceil['aggregate']:.4f}   per-factor "
          + ", ".join(f"{k}={vv:.3f}" for k, vv in ceil['per_factor'].items()))
    print(f"  probe on REAL      = {pr['aggregate']:.4f}")

json.dump(out, open("/Users/user/Desktop/CHLU/.claude/outputs/gen-feature-bridge/matched_control.json", "w"),
          indent=2, sort_keys=True, default=float)
print("\nratio B / A_random  =", out["B"]["ceiling"]["aggregate"] / out["A_random"]["ceiling"]["aggregate"])
print("ratio B / A_matched =", out["B"]["ceiling"]["aggregate"] / out["A_matched"]["ceiling"]["aggregate"])
