#!/usr/bin/env python3
"""Merge fig_task_cmazet's provenance (make_task_figures.py --only cmazet) and the glyph-identity
check into the build's PROVENANCE.json. usage: merge_task_prov.py <PROVENANCE.json> <task prov> <glyph json>"""
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = ["w19_task_figures/make_task_figures.py", "w21_figures/cmazet_panels.py",
       "w8_envscout/maze_colour.py", "w9_harness/dynamics.py", "chlu/exp_w15_cmaze_t.py"]


def main(prov_p, task_p, glyph_p):
    P = json.load(open(prov_p))
    t = json.load(open(task_p))["fig_task_cmazet"]
    g = json.load(open(glyph_p))
    assert g["identical"], "glyph identity gate failed"
    assert t["a"].get("budget_not_drawn") is not None
    P["figures"]["fig_task_cmazet"] = dict(
        t, built_by="w19_task_figures/make_task_figures.py --only cmazet (w22 figures3, AMENDMENT-1)",
        banner=False, placeholder=False, awaiting=None,
        sources=SRC, source_sha256={s: hashlib.sha256(open(os.path.join(REPO, s), "rb").read())
                                    .hexdigest() for s in SRC},
        notes="Task definition, no result. Every maze cell is drawn by the result panels' renderer "
              "cmazet_panels.draw_state (agent = black disc, centre = pocket colour, white = empty); "
              "the time/budget line is removed; c1..c4 are the panels' MAZE_RGB. fig_task_shapes is "
              "NOT rebuilt (w19 build, needs 3dshapes.h5).",
        glyph_identity=g)
    P["figures_redraw"]["stems_redrawn_or_added"] = sorted(
        set(P["figures_redraw"]["stems_redrawn_or_added"]) | {"fig_task_cmazet"})
    with open(prov_p, "w", encoding="utf-8") as f:
        json.dump(P, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
