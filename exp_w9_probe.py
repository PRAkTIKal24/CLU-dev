"""What the shared encoder's features actually carry, per seed.

Run after `exp_w9_harness.py`.  The encoder is a deterministic function of
(seed, training images), and the training images are a deterministic function of the
seed, so this rebuilds the SAME encoder each seed report used and probes it properly:
threshold-free AUC and balanced accuracy at the best threshold, instead of a fixed
0.5 cut that is meaningless on a plane which is 0.4% positive.

This is the evidence for "the shared encoder does not advantage one arm": whatever is
linearly present here is present for every arm, because every arm reads these arrays.
It is also how an arm's failures get attributed -- a policy that emits a pickup on an
empty cell while `key*` probes near 1.0 is failing to USE the feature, not failing to
have it.
"""

from __future__ import annotations

import argparse
import json
import os

from w9_harness import data, encoder, render
from exp_w9_harness import FEAT_DIM, SCALES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--scale", default="std")
    ap.add_argument("--out", default="/c/chlu/outputs/w9-harness")
    a = ap.parse_args()
    cfg = SCALES[a.scale]
    allout = {}
    for seed in [int(x) for x in a.seeds.split(",")]:
        base = 1_000_000 + 5_000_000 * seed
        tr, _ = data.make_instances(cfg["n_train"], base)
        ev, _ = data.make_instances(cfg["n_eval"], base + 2_000_000)
        tr_imgs = render.render_batch(tr)
        ev_imgs = render.render_batch(ev)
        enc, erep = encoder.train(tr_imgs, feat_dim=FEAT_DIM, steps=cfg["enc_steps"],
                                  seed=seed, verbose=False)
        k = int(tr[0].n_colours)
        print("seed %d  (encoder BCE %.5f, pixel acc %.5f, %d params)"
              % (seed, erep["final_bce"], erep["pixel_accuracy"],
                 erep["params_encoder"]))
        pr0 = encoder.probe_report(enc, ev_imgs, encoder.plane_names(k), radius=0)
        pr2 = encoder.probe_report(enc, ev_imgs, encoder.plane_names(k), radius=2)
        allout[seed] = {"encoder": {kk: vv for kk, vv in erep.items()
                                    if kk != "loss_curve"},
                        "probe_cell": pr0, "probe_patch5x5": pr2}
    path = os.path.join(a.out, "encoder_probe.json")
    with open(path, "w") as f:
        json.dump(allout, f, indent=1)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
