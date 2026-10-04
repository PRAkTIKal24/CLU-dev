"""⭐ THE LAUNDERING CONTROL for the movement claim.

`knn_proto` has NO tuning axis, so it cannot be moved by a selection rule — it moves
only because repairing `val` necessarily swaps 1.55 % of the `fit` rows its prototype
table is built from.  ⛔ To claim that is *resampling noise* rather than *our repair
weakening a rival*, the noise has to be MEASURED: here are three further, equally valid
stratified splits (different salts, same rule, same seed 0) and what `knn_proto` does
under each.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WT"])
from chlu.config import gen_row_split_config, get_default_config
from chlu.experiments.exp_gen_baselines import run_experiment_gen_baselines

OUT = Path(os.environ["OUT"]) / "runs/split_resample_null"
SALTS = {"salt_a": 20_260_915, "salt_b": 20_260_916, "salt_c": 20_260_917}

for tag, salt in SALTS.items():
    d = OUT / tag
    man = d / "shapes3d_recon_recomb2element" / "seed_0" / "baselines_manifest.json"
    if man.exists():
        print(f"SKIP {tag} (manifest exists)"); continue
    cfg = get_default_config()
    rs = gen_row_split_config(cfg)
    rs.seed_offset = int(salt)
    print(f"=== {tag}: seed_offset={salt} ===")
    run_experiment_gen_baselines(config=cfg, dataset="shapes3d", pretext="recon",
                                 out_dir=str(d), seeds=[0], arm="knn")

print("\n### knn_proto under four INDEPENDENT stratified splits (seed 0, deployed budget)")
rows = [("published (broken head/tail split)",
         Path(os.environ["REPO"]) / ".claude/outputs/gen-arms-a7-wiring/runs/deployed_524288/seed_0/baselines_manifest.json"),
        ("the shipped repair (seed_offset=20260914)",
         Path(os.environ["OUT"]) / "runs/deployed_524288/seed_0/baselines_manifest.json")]
rows += [(f"alternative stratified split {t} (seed_offset={s})",
          OUT / t / "shapes3d_recon_recomb2element" / "seed_0" / "baselines_manifest.json")
         for t, s in SALTS.items()]
print("| split | knn_proto in-corner balacc | knn_proto composition B agg | knn_proto composition A agg |")
print("|---|---|---|---|")
for lbl, p in rows:
    if not Path(p).exists():
        print(f"| {lbl} | MISSING | | |"); continue
    v = json.load(open(p))["arms"]["knn"]["variants"]["knn_proto"]
    print(f"| {lbl} | {v['in_corner_discrimination']['balanced_accuracy']:.4f} | "
          f"{v['composition_B']['aggregate']:.4f} | {v['composition_A']['aggregate']:.4f} |")
