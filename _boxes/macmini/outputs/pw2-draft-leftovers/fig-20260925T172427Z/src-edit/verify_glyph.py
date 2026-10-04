"""AMENDMENT-1 check: the agent glyph in fig_task_cmazet IS the result panels' agent glyph.

1. Source level: every agent the task figure draws comes from cmazet_panels.draw_state (via
   draw_grid). For each pocket state the task figure shows (c3 in (a); c1 and empty in (b)-(d)),
   the 10x10 agent-cell pixels of the task figure's raster are compared with the agent-cell
   pixels of a real result-panel raster (the per-item re-read, first item whose drawn arm state
   has that pocket; for c3: seed 2 item 482, the Head's test case). Compared on the disc mask
   (rad <= 4.2, the glyph itself) and on the whole cell when both cells are plain floor.
2. Output level: the ring and centre colours are present in the embedded image palettes of
   BOTH PDFs (fig_task_cmazet.pdf and fig_cmazet_families.pdf) as the same bytes.
usage: python -m w19_task_figures.verify_glyph <task_fig_dir> <panels_fig_dir> <out_json>
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# PW-1 item 4: repo layout (w19_task_figures/ next to w21_figures/) or a flat figure folder
FIGS = os.path.join(os.path.dirname(HERE), "w21_figures")
if not os.path.isfile(os.path.join(FIGS, "figroot.py")):
    FIGS = HERE
sys.path.insert(0, FIGS)
import figroot  # noqa: E402
figroot.add_code_to_path()
import cmazet_panels as CP  # noqa: E402
if FIGS == HERE:
    import make_task_figures as TF  # noqa: E402
else:
    from w19_task_figures import make_task_figures as TF  # noqa: E402
from w9_harness.dynamics import State  # noqa: E402
figroot.check_loaded()

C = CP.CELL
yy, xx = np.mgrid[0:C, 0:C]
DISC = np.hypot(yy - (C - 1) / 2.0, xx - (C - 1) / 2.0) <= 4.2


def cell(img, r, c, r0=0, c0=0):
    return img[(r - r0) * C:(r - r0 + 1) * C, (c - c0) * C:(c - c0 + 1) * C]


def panel_agent_block(pocket_want):
    d = json.load(open(os.path.join(FIGS, "data", "cmazet_items.json")))
    order = sorted(d["seeds"], key=lambda s: (s["seed"] != 2, s["seed"]))
    for srec in order:
        mz = CP.Maze(srec)
        idx = [482] + list(range(len(srec["items"]))) if srec["seed"] == 2 else \
            range(len(srec["items"]))
        for i in idx:
            it = srec["items"][i]
            arm = next((a for a in ["CLU"] + sorted(srec["verdicts"])
                        if State(mz.states(it, srec["verdicts"][a][i][2])[2].pos,
                                 mz.states(it, srec["verdicts"][a][i][2])[2].ks,
                                 0).pocket == pocket_want), None)
            if arm is None:
                continue
            inst, st, gen = mz.states(it, srec["verdicts"][arm][i][2])
            img = CP.draw_state(inst, gen.pos, gen.ks, st.pos, 5)
            h = 2
            r, c = gen.pos[0] - st.pos[0] + h, gen.pos[1] - st.pos[1] + h
            if not (0 <= r < 5 and 0 <= c < 5):
                continue
            floor_plain = inst.tint[gen.pos] == 0 and inst.door[gen.pos] == 0
            return img[r * C:(r + 1) * C, c * C:(c + 1) * C], dict(
                seed=srec["seed"], i=i, arm=arm, pocket=pocket_want,
                plain_floor=bool(floor_plain))
    raise AssertionError(pocket_want)


def task_blocks():
    inst, st, *_ = TF.pick_example_state()
    out = [("(a) grid", st.pocket, cell(CP.draw_grid(inst, st.pos, st.ks), *st.pos),
            bool(inst.tint[st.pos] == 0 and inst.door[st.pos] == 0))]
    for tint_c, door_c, pk in ((2, 0, 1), (1, 0, 1), (1, 1, 1)):
        im, s0, s1, _w, _a = TF.rule_case(tint_c, door_c, pk)
        for tag, s in (("before", s0), ("after", s1)):
            out.append(("(%d,%d) %s" % (tint_c, door_c, tag), s.pocket,
                        cell(CP.draw_grid(im, s.pos, s.ks), *s.pos),
                        bool(im.tint[s.pos] == 0 and im.door[s.pos] == 0)))
    return out


def pdf_palettes(path):
    b = open(path, "rb").read()
    cols = set()
    for m in re.finditer(rb"/Indexed /DeviceRGB (\d+) \((.*?)\) \]", b, re.S):
        raw = m.group(2).replace(b"\\(", b"(").replace(b"\\)", b")").replace(b"\\\\", b"\\")
        n = int(m.group(1)) + 1
        if len(raw) == 3 * n:
            cols |= {raw[3 * j:3 * j + 3].hex() for j in range(n)}
    return cols


def main(task_dir, panels_dir, out):
    res = {"blocks": []}
    ok_all = True
    cache = {}
    for name, pocket, blk, plain in task_blocks():
        if pocket not in cache:
            cache[pocket] = panel_agent_block(pocket)
        pblk, prec = cache[pocket]
        disc_eq = bool(np.array_equal(blk[DISC], pblk[DISC]))
        cell_eq = bool(np.array_equal(blk, pblk)) if (plain and prec["plain_floor"]) else None
        ok_all &= disc_eq and cell_eq is not False
        res["blocks"].append(dict(task_panel=name, pocket=int(pocket), panel_item=prec,
                                  disc_pixels_equal=disc_eq, whole_cell_equal=cell_eq,
                                  n_disc_pixels=int(DISC.sum())))
    # float RGB -> the byte matplotlib embeds (truncation, as its float->uint8 image path does)
    hx = lambda rgb: bytes((np.asarray(rgb, np.float32) * 255).astype(np.uint8)).hex()  # noqa
    ring = hx((0.05, 0.05, 0.05))
    centres = {str(c): hx(rgb) for c, rgb in CP.MAZE_RGB.items()}
    centres["empty"] = "ffffff"
    pt = pdf_palettes(os.path.join(task_dir, "fig_task_cmazet.pdf"))
    pp = pdf_palettes(os.path.join(panels_dir, "fig_cmazet_families.pdf"))
    shown = {"ring": ring, **{k: v for k, v in centres.items() if k in ("1", "3", "empty")}}
    res["pdf_palette"] = {k: dict(hex=v, in_task_pdf=v in pt, in_panels_pdf=v in pp)
                          for k, v in shown.items()}
    ok_all &= all(v["in_task_pdf"] and v["in_panels_pdf"] for v in res["pdf_palette"].values())
    res["identical"] = bool(ok_all)
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
