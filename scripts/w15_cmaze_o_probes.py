"""The two instrument-construction probes, saved as artefacts.

Neither measures an arm.  Both are properties of the ENVIRONMENT and of the shared
autoencoder, both were run before ``PREREG.md``, and both are declared as such.

1. ``braid_probe.json``   -- how often a door can be LOAD-BEARING at each braid
   fraction.  This is what fixed ``BRAID = 0.08`` and the balanced-articulation door.
2. ``ae_loss_probe.json`` -- the autoencoder's own ceiling under each of the three loss
   shapes tried.  This is what fixed the spatial-softmax object loss and the masked
   colour softmax.  Its numbers are transcribed from the probe logs named inside it.

Usage:  python scripts/w15_cmaze_o_probes.py <out_dir>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chlu.exp_w15_cmaze_o_env import (N_ZONES, ZONE_SIDE, _zone_centre,   # noqa: E402
                                      cut_balance, floor_components, floor_degree)
from w8_envscout.maze_colour import FLOOR, _braid, _carve_maze            # noqa: E402


def braid_probe(seeds=range(90_000, 90_012),
                braids=(0.40, 0.25, 0.15, 0.08)) -> dict:
    rows = []
    for braid in braids:
        arts, seps = [], []
        for s in seeds:
            rng = np.random.default_rng(s)
            g = _carve_maze(7, 7, rng)
            _braid(g, rng, braid)
            deg = floor_degree(g)
            door, anchors = {}, {}
            for z in range(N_ZONES):
                r0, c0 = ZONE_SIDE * (z // 3), ZONE_SIDE * (z % 3)
                cr, cc = _zone_centre(z)
                cells = [(r, c) for r in range(r0, r0 + ZONE_SIDE)
                         for c in range(c0, c0 + ZONE_SIDE) if g[r, c] == FLOOR]
                cells.sort(key=lambda p: (abs(p[0] - cr) + abs(p[1] - cc), p[0], p[1]))
                anchors[z] = cells[:4]
                corr = [p for p in cells if deg[p] >= 2]
                if corr:
                    door[z] = max(corr, key=lambda p: (cut_balance(g, p),
                                                       -(abs(p[0] - cr)
                                                         + abs(p[1] - cc))))
            arts.append(sum(1 for z in door if cut_balance(g, door[z]) > 0) / N_ZONES)
            tot = sep = 0
            for zd in door:
                cs = floor_components(g, door[zd])
                lab = {}
                for i, comp in enumerate(cs):
                    for p in comp:
                        lab[p] = i
                for za in range(N_ZONES):
                    for zg in range(N_ZONES):
                        if len(anchors[za]) < 4 or len(anchors[zg]) < 4:
                            continue
                        a, b = anchors[za][0], anchors[zg][3]
                        if a not in lab or b not in lab:
                            continue
                        tot += 1
                        sep += int(lab[a] != lab[b])
            seps.append(sep / max(1, tot))
        rows.append({"braid": braid,
                     "zones_with_a_balanced_articulation_door": float(np.mean(arts)),
                     "agent_goal_separated_by_the_door": float(np.mean(seps))})
    return {
        "probe": "braid_probe",
        "what_it_measures": ("how often the door's designated cell can actually be "
                             "LOAD-BEARING -- i.e. how often removing it separates the "
                             "agent's cell from the goal's cell"),
        "is_an_arm_number": False,
        "n_layout_seeds": len(list(seeds)),
        "rows": rows,
        "decision": ("BRAID = 0.08 and the door on the balanced articulation point.  "
                     "CMAZE-1 adopted braid 0.40 because its task emits a ROUTE and its "
                     "adversarial tint placement needs route MULTIPLICITY; CMAZE-O "
                     "emits a STATE and uses no adversarial placement.  At 0.40 RULES A "
                     "and B are wallpaper on this instrument."),
        "engine_touched": False,
    }


AE_LOSS_PROBE = {
    "probe": "ae_loss_probe",
    "what_it_measures": ("the SHARED autoencoder's own ceiling: encode the TRUE "
                         "picture, decode it, judge it.  No arm is involved."),
    "is_an_arm_number": False,
    "protocol": "2,000 admissible requests, 12 layouts, latent D = 128, batch 64",
    "rows": [
        {"loss_on_object_and_colour_channels": "per-cell BCE everywhere",
         "item_tensor": "22 channels, per-slot colour block",
         "ae_steps": 1500, "exact_objects": 0.0025, "exact_layout": None,
         "legal": 0.8275,
         "log": "artefacts/ae_probe_v1.log"},
        {"loss_on_object_and_colour_channels":
             "spatial softmax over the 225 cells for the four objects; colour still BCE",
         "item_tensor": "11 channels, shared colour field",
         "ae_steps": 2500, "exact_objects": 0.0200, "exact_layout": 0.9850,
         "legal": 0.9725,
         "object_cells_exact": {"agent": 1.0, "key": 1.0, "door": 1.0, "goal": 1.0},
         "object_colours_exact": {"agent": 0.3625, "key": 0.3575, "door": 0.3400,
                                  "tint": 0.4425},
         "log": "artefacts/ae_probe_v2.log"},
        {"loss_on_object_and_colour_channels":
             "spatial softmax for the four objects + MASKED per-cell softmax over the "
             "5 colours wherever a colour lives",
         "item_tensor": "11 channels, shared colour field",
         "ae_steps": 1500, "exact_objects": 0.3375, "exact_layout": 0.9825,
         "legal": 0.8400,
         "object_cells_exact": {"agent": 0.5325, "key": 0.9950, "door": 0.9225,
                                "goal": 0.9950},
         "object_colours_exact": {"agent": 0.6600, "key": 0.9950, "door": 0.9350,
                                  "tint": 0.7250},
         "log": "artefacts/ae_probe_v3.log",
         "note": ("still improving at 1,500 steps -- the VAL smoke trains the shared "
                  "autoencoder for 6,000 steps and reports the realised ceiling on "
                  "every row")},
    ],
    "decision": ("the judge takes an argmax over 225 cells and an argmax over 5 "
                 "colours, so every piece of the loss is the cross-entropy of exactly "
                 "that argmax.  A per-cell BCE never makes the correct cell compete "
                 "against the other 224."),
}


def main(argv):
    out = argv[1]
    os.makedirs(out, exist_ok=True)
    b = braid_probe()
    with open(os.path.join(out, "braid_probe.json"), "w", encoding="utf-8") as f:
        json.dump(b, f, indent=1)
    with open(os.path.join(out, "ae_loss_probe.json"), "w", encoding="utf-8") as f:
        json.dump(AE_LOSS_PROBE, f, indent=1)
    for r in b["rows"]:
        print("braid %.2f  balanced-articulation zones %.4f  separated %.4f"
              % (r["braid"], r["zones_with_a_balanced_articulation_door"],
                 r["agent_goal_separated_by_the_door"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
