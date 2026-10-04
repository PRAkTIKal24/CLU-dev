"""w19 task-figures -- WHAT THE TASKS ARE, drawn.  Two task-definition figures, no result.

fig_task_shapes   Shapes3D: the six factors with example values (dataset images), one example
                  request with its rendered answer and the judge's read-back, and which cells
                  the 2-way (pair) and six-factor (n-way) hold-outs remove from the fit pool.
fig_task_cmazet   CMAZE-T: one annotated grid (engine-generated), the two rules and the
                  door-then-tint interaction as three before/after panels, each with its five
                  request words and its four-fact answer.

Inputs:  <assets>/shapes_images.npz  (extract_images.py, Stage A)
         <assets>/task_assets.npz     (extract_assets.py, Stage B: frozen-decoder render + fit pool)
         the CMAZE-T engine itself (w8_envscout.maze_colour, w9_harness.dynamics,
         chlu.exp_w15_cmaze_t), run here -- integers only, no randomness beyond pinned seeds.
Every CMAZE-T verdict drawn is asserted twice: by the engine stepper and by the independent
``step_ref``; the build stops if they disagree or if a panel's verdict is not the one its
title names.  No w15/w16 NUMBER (rate, score, seed count) appears (AMENDMENT 1).

Determinism: Agg/PDF pinned, bundled DejaVu Sans, PDF/PNG metadata stripped,
SOURCE_DATE_EPOCH=0, no unseeded randomness.   usage: make_task_figures.py --assets D --out D
"""
import argparse
import json
import os

os.environ["SOURCE_DATE_EPOCH"] = "0"
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import (ConnectionPatch, FancyArrowPatch, Patch,  # noqa: E402
                                Rectangle)

from w8_envscout.maze_colour import (CARRIED, FLOOR, UNTAKEN, USED, WALL, Instance,  # noqa: E402
                                     generate, solve)
from w9_harness.dynamics import State, initial_state, key_homes, step  # noqa: E402
from chlu import exp_w15_cmaze_t as T  # noqa: E402

# w22 figures3 (AMENDMENT-1): every maze cell in fig_task_cmazet is drawn by the RESULT PANELS'
# own renderer, cmazet_panels.draw_state, so the legend glyphs ARE the panel glyphs (same
# function, same MAZE_RGB, pocket from State(pos, ks, 0).pocket), not an imitation of them.
import sys as _sys  # noqa: E402
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "w21_figures"))
import cmazet_panels as CP  # noqa: E402

# ---------------------------------------------------------------- style (w17/plots, verbatim)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
PRESENT = "#d9d8d3"          # "in the fit pool": neutral, never an arm colour
WALLC = "#3d3c39"
SURFACE = "#ffffff"
TEXT_W = 5.5
# the four CMAZE-T key/door/tint colours c1..c4.  Categorical slots 1, 2, 3, 7 of the dataviz
# reference palette: this 4-set passes the validator ALL-PAIRS (worst CVD dE 9.2, normal 16.3);
# slot 3 (aqua) is below 3:1 on white, so every coloured mark carries its digit (relief rule).
# w22 figures3: replaced by the result panels' MAZE_RGB (cmazet_panels), so c1..c4 are the same
# colours in the task figure and in every CMAZE-T result panel. (Was: dataviz slots 1, 2, 3, 7.)
MAZE_COL = {c: "#%02x%02x%02x" % tuple(int(round(255 * x)) for x in rgb)
            for c, rgb in CP.MAZE_RGB.items()}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "pdf.compression": 6,
    "svg.hashsalt": "w19-task-figures",
    "path.simplify": False,
    "hatch.linewidth": 0.5,
})

PROV = {}
FACTOR_LABEL = {"floor_hue": "floor hue", "wall_hue": "wall hue", "object_hue": "object hue",
                "scale": "scale", "shape": "shape", "orientation": "orientation"}
SHAPE_NAME = {0: "cube", 1: "cylinder", 2: "sphere", 3: "capsule"}


def fmt_val(name, v):
    if name == "shape":
        return SHAPE_NAME[int(round(v))]
    if name == "orientation":
        return "%+.0f°" % v if abs(v) > 1e-9 else "0°"
    if name == "scale":
        return "%.2f" % v
    return "%.1f" % v


def save(fig, out, stem):
    fig.savefig(os.path.join(out, stem + ".pdf"),
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(os.path.join(out, stem + ".png"), metadata={"Software": None})
    plt.close(fig)


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ================================================================== (a) SHAPES
def presence_grid(ax, M, xlabels, ylabels, xname, yname, fs=5.5):
    """Binary presence: grey = in the fit pool, white = missing.  x = columns of M."""
    ny, nx = M.shape
    for i in range(ny):
        for j in range(nx):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=PRESENT if M[i, j] > 0 else SURFACE,
                                   edgecolor=SURFACE if M[i, j] > 0 else GRID, lw=0.4))
    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(nx) + 0.5)
    ax.set_xticklabels(xlabels, fontsize=fs, rotation=90 if max(len(s) for s in xlabels) > 3 else 0)
    ax.set_yticks(np.arange(ny) + 0.5)
    ax.set_yticklabels(ylabels, fontsize=fs)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlabel(xname, fontsize=6, labelpad=2)
    ax.set_ylabel(yname, fontsize=6, labelpad=2)
    for s in ax.spines.values():
        s.set_visible(False)


def outline(ax, x0, y0, w, h, lw=1.4, ls="-", color=INK):
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color, lw=lw, ls=ls, zorder=5))


def fig_task_shapes(assets, out):
    S = np.load(os.path.join(assets, "shapes_images.npz"))
    A = np.load(os.path.join(assets, "task_assets.npz"))
    with open(os.path.join(assets, "ASSETS.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    strip = json.loads(str(S["strip"]))
    rows = [int(r) for r in S["rows"]]
    img_of = {r: S["images"][i] for i, r in enumerate(rows)}
    lab_of = {r: S["labels"][i] for i, r in enumerate(rows)}
    gv = meta["grid_values"]
    names = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
    req_idx = [int(x) for x in A["request_indices"]]
    rb_idx = [int(x) for x in A["read_back"][0]]
    assert int(S["request_row"]) == meta["request"]["dataset_row"]
    assert meta["render"]["judge_valid"] is True and rb_idx == req_idx

    FW, FH = TEXT_W, 4.9
    fig = plt.figure(figsize=(FW, FH))

    def ax_in(x, y, w, h):
        return fig.add_axes((x / FW, y / FH, w / FW, h / FH))

    def ftext(x, y, s, **kw):
        fig.text(x / FW, y / FH, s, **kw)

    def farrow(x0, y0, x1, y1):
        fig.add_artist(FancyArrowPatch((x0 / FW, y0 / FH), (x1 / FW, y1 / FH),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, lw=0.8, color=INK2))

    # ---- (a) six factors, example values: dataset images, one factor varied at a time
    ftext(0.02, 4.86, "(a) Six factors", fontsize=8, ha="left", va="top", color=INK)
    x_img0, pitch_x, iw = 0.72, 0.40, 0.29
    top, pitch_y = 4.62, 0.40
    for f, name in enumerate(names):
        y = top - iw - f * pitch_y
        ftext(x_img0 - 0.06, y + iw / 2, FACTOR_LABEL[name], fontsize=6.5, color=INK,
              ha="right", va="center")
        rr = strip["strip_rows"][name]
        for j, r in enumerate(rr):
            ax = ax_in(x_img0 + j * pitch_x, y, iw, iw)
            off(ax)
            ax.imshow(img_of[r], interpolation="nearest")
            v = float(lab_of[r][f])
            assert abs(v - gv[name][strip["strip_indices"][name][j]]) < 1e-9
            ftext(x_img0 + j * pitch_x + iw / 2, y + iw + 0.015, fmt_val(name, v),
                  fontsize=5.3, color=INK2, ha="center", va="bottom")

    # ---- (b) request -> rendered answer -> judge read-back
    ftext(2.86, 4.86, "(b) Request → rendered answer", fontsize=8, ha="left", va="top", color=INK)
    req_vals = meta["request"]["factor_values"]
    ys = np.linspace(4.22, 2.62, 6)
    xr0, xr1 = 2.88, 3.84          # request column
    xi0, iw2 = 4.06, 0.80          # image
    xj0, xj1 = 5.10, 5.49          # read-back column
    hy = 4.52
    ftext((xr0 + xr1) / 2, hy, "request", fontsize=6.8, ha="center", va="center", color=INK,
          weight="bold")
    ftext(xi0 + iw2 / 2, hy, "answer", fontsize=6.8, ha="center", va="center", color=INK,
          weight="bold")
    ftext((xj0 + xj1) / 2, hy, "judge", fontsize=6.8, ha="center", va="center", color=INK,
          weight="bold")
    for f, name in enumerate(names):
        ftext(xr0, ys[f], FACTOR_LABEL[name], fontsize=5.8, color=INK2, ha="left", va="center")
        ftext(xr1, ys[f], fmt_val(name, req_vals[f]), fontsize=6, color=INK, ha="right",
              va="center")
        ftext(xj0, ys[f], fmt_val(name, gv[name][rb_idx[f]]), fontsize=6, color=INK, ha="left",
              va="center")
        ftext(xj1, ys[f], "✓" if rb_idx[f] == req_idx[f] else "✗", fontsize=7, color=INK,
              ha="right", va="center")
    yi0 = (ys[0] + ys[-1]) / 2 - iw2 / 2
    axi = ax_in(xi0, yi0, iw2, iw2)
    off(axi)
    img = np.transpose(A["render"][0], (1, 2, 0)).clip(0, 1)
    axi.imshow(img, interpolation="nearest")
    ftext(xi0 + iw2 / 2, yi0 - 0.07, "frozen decoder", fontsize=5.5, color=INK2, ha="center",
          va="top")
    ymid = yi0 + iw2 / 2
    farrow(xr1 + 0.04, ymid, xi0 - 0.04, ymid)
    farrow(xi0 + iw2 + 0.03, ymid, xj0 - 0.04, ymid)

    # ---- (c) 2-way hold-out: object hue x shape, fit-pool presence
    ftext(0.02, 2.14, "(c) 2-way hold-out", fontsize=8, ha="left", va="top", color=INK)
    cell = 0.205
    axc = ax_in(0.62, 0.86, 10 * cell, 4 * cell)
    P = A["pair_pool"].T  # rows = shape (4), cols = object hue (10)
    presence_grid(axc, P, ["%.1f" % v for v in gv["object_hue"]],
                  [SHAPE_NAME[int(v)] for v in gv["shape"]], "object hue", "shape")
    vh = meta["fit_pool"]["value_holdout"]
    for i in range(4):
        axc.add_patch(Rectangle((vh["object_hue_index"], i), 1, 1, fill=False, hatch="////",
                                edgecolor=INK2, lw=0, zorder=4))
    for j in range(10):
        axc.add_patch(Rectangle((j, vh["shape_index"]), 1, 1, fill=False, hatch="////",
                                edgecolor=INK2, lw=0, zorder=4))
    for (i, j, _cid) in meta["fit_pool"]["pair_cells"]:
        assert P[j, i] == 0
        outline(axc, i + 0.08, j + 0.08, 0.84, 0.84, lw=1.3)

    # ---- (d) six-factor hold-out: projections present, the conditional slice missing
    ftext(2.98, 2.14, "(d) Six-factor hold-out", fontsize=8, ha="left", va="top", color=INK)
    box = meta["fit_pool"]["nway_box_indices"]
    c2 = 0.074
    ax1 = ax_in(3.20, 0.74, 10 * c2, 10 * c2)
    presence_grid(ax1, A["marg_fw"],
                  ["%.1f" % v if k % 3 == 0 else "" for k, v in enumerate(gv["wall_hue"])],
                  ["%.1f" % v if k % 3 == 0 else "" for k, v in enumerate(gv["floor_hue"])],
                  "wall hue", "floor hue", fs=5)
    outline(ax1, box["wall_hue"][0], box["floor_hue"][0], len(box["wall_hue"]),
            len(box["floor_hue"]), lw=1.1, ls="--")
    ax1.set_title("all rows", fontsize=5.8, pad=2, color=INK2)
    orient_lab = [("%+.0f" % v).replace("+0", "0") if k % 7 == 0 else ""
                  for k, v in enumerate(gv["orientation"])]
    scale_lab = ["%.2f" % v if k % 7 == 0 else "" for k, v in enumerate(gv["scale"])]
    for ax, M, ttl, sl in ((ax_in(4.36, 1.36, 15 * c2, 8 * c2), A["marg_so"], "all rows", False),
                           (ax_in(4.36, 0.40, 15 * c2, 8 * c2), A["slice_so"],
                            "floor, wall, object hue, shape\nin the corner", True)):
        presence_grid(ax, M, orient_lab if sl else [""] * 15, scale_lab, "orientation" if sl else "",
                      "scale", fs=5)
        ax.set_title(ttl, fontsize=5.8, pad=2, color=INK2, linespacing=1.0)
        # nway-034 in both; in the slice also the three six-factor corners sharing it
        for sc0 in (0, len(gv["scale"]) - 3):
            for or0 in (0, len(gv["orientation"]) - 3):
                is_034 = (sc0 == box["scale"][0] and or0 == box["orientation"][0])
                if sl or is_034:
                    outline(ax, or0, sc0, 3, 3, lw=1.1, ls="--")
    assert int((A["slice_so"][np.ix_(box["scale"], box["orientation"])] > 0).sum()) == 0
    assert int((A["marg_so"][np.ix_(box["scale"], box["orientation"])] == 0).sum()) == 0
    assert int((A["marg_fw"][np.ix_(box["floor_hue"], box["wall_hue"])] == 0).sum()) == 0

    # ---- one legend for (c) and (d)
    axl = ax_in(0.02, 0.0, FW - 0.04, 0.22)
    off(axl)
    handles = [Patch(facecolor=PRESENT, edgecolor=PRESENT, label="in fit pool"),
               Patch(facecolor=SURFACE, edgecolor=GRID, lw=0.6, label="not in fit pool"),
               Patch(facecolor=SURFACE, edgecolor=INK, lw=1.3, label="pair corner (2-way)"),
               Patch(facecolor=SURFACE, edgecolor=INK, lw=1.1, ls="--",
                     label="six-factor corner (n-way)"),
               Patch(facecolor=SURFACE, edgecolor=INK2, hatch="////", lw=0.4,
                     label="value hold-out")]
    axl.legend(handles=handles, loc="center", ncol=5, fontsize=5.8, handlelength=1.3,
               handleheight=1.1, columnspacing=1.1, borderaxespad=0, handletextpad=0.5)
    save(fig, out, "fig_task_shapes")
    PROV["fig_task_shapes"] = {
        "a": "3dshapes.h5 rows " + json.dumps(strip["strip_rows"], sort_keys=True),
        "b": {"request": meta["request"], "render": meta["render"]},
        "c": "fit pool (condition test, n_pool %d, pool_rng_seed %d): object_hue x shape presence;"
             " pair cells %s; value hold-out %s" % (meta["fit_pool"]["n_pool"],
                                                    meta["fit_pool"]["pool_rng_seed"],
                                                    meta["fit_pool"]["pair_cells"],
                                                    meta["fit_pool"]["value_holdout"]),
        "d": {"corner": meta["fit_pool"]["nway_corner"], "box": box,
              "rows_in_pool": meta["fit_pool"]["nway_rows_in_pool"],
              "min_pairwise_projection_count": min(
                  meta["fit_pool"]["nway_min_count_over_all_15_pairwise_projections_of_the_box"].values()),
              "slice_shares_corners": ["nway-028", "nway-034", "nway-040", "nway-046"]},
    }


# ================================================================== (b) CMAZE-T
def digits(ax, inst, ks, pos, x0=0.0, y0=0.0, rows=None, fs=4.2):
    """Colour digits over the draw_state raster (relief rule: identity never by colour alone):
    one small ink digit in the lower-right of every tinted floor, door and untaken-key cell."""
    H, W = inst.grid.shape
    homes = {tuple(inst.key_at[i]): i + 1 for i in range(int(inst.n_colours))}
    for r in (range(H) if rows is None else rows):
        for c in range(W):
            if inst.grid[r, c] != FLOOR or (r, c) == tuple(pos):
                continue
            kc = homes.get((r, c), 0)
            d = int(inst.door[r, c])
            v = (kc if kc and ks[kc - 1] == UNTAKEN else 0) or d or int(inst.tint[r, c])
            if v:
                ax.text(x0 + c + 0.80, y0 + r + 0.78, str(v), ha="center", va="center",
                        fontsize=fs, color=INK, zorder=5)


def pick_example_state():
    """RULE (written after seeing only the bank's FIRST maze, never the chosen one): walk the seed-0 TRAIN bank's generator sequence exactly
    as ``build_bank`` does (base 2_000_000, rng seed base + 5551, ``tinted_door_instance``); take
    the FIRST maze whose optimal route picks up >= 2 keys; show the state right after its
    second pickup (one door already opened, one key in the pocket)."""
    base = 2_000_000
    rng = np.random.default_rng(base + 5551)
    s = base
    for _ in range(200):
        b = generate(s)
        s += 1
        if b is None:
            continue
        inst, rec = T.tinted_door_instance(b, rng)
        if inst is None:
            continue
        # build_bank also draws transitions from the same rng; replay that so the stream is
        # the bank's own and the maze is the bank's maze
        m_idx = 0
        T.transitions_for_maze(inst, m_idx, rng, per_tuple=2)
        T.natural_rows_for_maze(inst, m_idx, rng, n=150)
        L, route = solve(inst)
        if route.count("P") < 2:
            continue
        homes = key_homes(inst)
        st = initial_state(inst)
        n_p = 0
        for t, a in enumerate(route):
            st, legal, _ = step(inst, st, a, homes)
            assert legal
            if a == "P":
                n_p += 1
                if n_p == 2:
                    return inst, st, t + 1, route, s - 1, rec
    raise AssertionError("no maze with >= 2 pickups in 200 generator seeds")


def mini_instance(tint_c, door_c):
    """A 1 x 2 strip: agent cell (0,0), target cell (0,1).  Engine-typed Instance."""
    g = np.full((1, 2), FLOOR, np.uint8)
    tint = np.zeros((1, 2), np.int8)
    door = np.zeros((1, 2), np.int8)
    tint[0, 1] = tint_c
    door[0, 1] = door_c
    keys = tuple((9, 9 + i) for i in range(4))   # off-strip homes: no key on the agent's cell
    return Instance(0, g, tint, door, keys, (0, 0), (9, 0), 4, 99, 1, 1)


def rule_case(tint_c, door_c, pocket):
    inst = mini_instance(tint_c, door_c)
    ks = [UNTAKEN] * 4
    ks[pocket - 1] = CARRIED
    st = State((0, 0), tuple(ks), 0)
    homes = key_homes(inst)
    words = T.words_of(inst, st, "E", homes)
    legal, disp, ev, pk, nxt = T.outcome_of(inst, st, "E", homes)
    carried = pocket
    used = frozenset(i + 1 for i, v in enumerate(ks) if v == USED)
    taken = frozenset(i + 1 for i, v in enumerate(ks) if v != UNTAKEN)
    ok_ref, pos_ref, car_ref, _u, _t = T.step_ref(inst, (0, 0), carried, used, taken, "E")
    assert bool(ok_ref) == bool(legal), "engine stepper and step_ref disagree"
    if legal:
        assert pos_ref == nxt.pos and car_ref == nxt.pocket
    return inst, st, nxt, words, (legal, disp, ev, pk)


WORD_FMT = ("w_agent", "w_key", "w_door", "w_tint", "w_dir")


def word_str(i, v, k=4):
    if i in (0, 1, 3):
        return "c%d" % v if v else "none"
    if i == 2:
        return {0: "none", k + 1: "open", k + 2: "wall"}.get(v, "c%d closed" % v)
    return "NESWP"[v]


def answer_str(legal, disp, ev, pk, k=4):
    e = "none" if ev == 0 else ("pick c%d" % ev if ev <= k else "spend c%d" % (ev - k))
    return [("legal", "1" if legal else "0"),
            ("disp", ["stay", "N", "E", "S", "W"][disp]),
            ("event", e),
            ("pocket", "c%d" % pk if pk else "empty")]


def fig_task_cmazet(out):
    inst, st, t_now, route, gen_seed, rec = pick_example_state()
    H, W = inst.grid.shape
    fig = plt.figure(figsize=(TEXT_W, 4.85))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], left=0.01, right=0.99, top=0.965,
                          bottom=0.01, hspace=0.12)
    gtop = gs[0].subgridspec(1, 2, width_ratios=[1.0, 0.62], wspace=0.02)
    ax = fig.add_subplot(gtop[0, 0])
    off(ax)
    grid_img = CP.draw_grid(inst, st.pos, st.ks)
    ax.imshow(grid_img, extent=(0, W, H, 0), interpolation="nearest")
    digits(ax, inst, st.ks, st.pos)
    assert State(st.pos, st.ks, 0).pocket == st.pocket
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_title("(a) CMAZE-T grid", loc="left", fontsize=8, pad=3)

    # --- annotations: one exemplar of each element (the RIGHT-most one, so leader lines stay
    # short), labels in the right-hand column level with their target where space allows
    def rightmost(pred):
        best = None
        for r in range(H):
            for c in range(W):
                if pred(r, c) and (best is None or (c, -r) > (best[1], -best[0])):
                    best = (r, c)
        return best
    closed = rightmost(lambda r, c: inst.door[r, c] and st.ks[inst.door[r, c] - 1] != USED)
    opened = rightmost(lambda r, c: inst.door[r, c] and st.ks[inst.door[r, c] - 1] == USED)
    tinted = rightmost(lambda r, c: inst.grid[r, c] == FLOOR and inst.tint[r, c]
                       and not inst.door[r, c])
    untaken = [inst.key_at[i] for i in range(4) if st.ks[i] == UNTAKEN]
    key_cell = max(untaken, key=lambda p: (p[1], -p[0])) if untaken else None
    assert closed and opened and tinted and key_cell and st.pocket
    targets = [("agent", (st.pos[0] - 0.36, st.pos[1])),
               ("pocket (disc centre)", (st.pos[0], st.pos[1])),
               ("key", key_cell), ("closed door", closed), ("open door", opened),
               ("tinted floor", tinted), ("goal", inst.goal)]
    targets.sort(key=lambda x: (x[1][0], x[0]))
    axn = fig.add_subplot(gtop[0, 1])
    off(axn)
    axn.set_xlim(0, 1)
    axn.set_ylim(H, 0)
    ax.apply_aspect()
    axn.apply_aspect()

    def levels(order):
        ly, prev = [], -9.0
        for _lab, (r, _c) in order:          # level with the target, pushed down to clear
            y = max(r + 0.5, prev + 1.05, 0.6)
            ly.append(y)
            prev = y
        over = ly[-1] - (H - 1.4)            # keep clear of the colour row
        return [y - over for y in ly] if over > 0 else ly

    def crossings(order):
        segs = [(axn.transData.transform((0.28, y)), ax.transData.transform((c + 0.5, r + 0.5)))
                for (_l, (r, c)), y in zip(order, levels(order))]

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
        n = 0
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                (p1, p2), (p3, p4) = segs[i], segs[j]
                if ccw(p1, p3, p4) * ccw(p2, p3, p4) < 0 and ccw(p1, p2, p3) * ccw(p1, p2, p4) < 0:
                    n += 1
        return n
    # label order: the fewest leader-line crossings over all orders; ties -> row order
    import itertools
    best = min(itertools.permutations(targets),
               key=lambda o: (crossings(o), [targets.index(t) for t in o]))
    LEADER_CROSSINGS = crossings(best)
    targets = list(best)
    ly = levels(targets)
    for (lab, (r, c)), y in zip(targets, ly):
        axn.text(0.30, y, lab, ha="left", va="center", fontsize=6.5, color=INK)
        fig.add_artist(ConnectionPatch(xyA=(0.28, y), coordsA=axn.transData,
                                       xyB=(c + 0.5, r + 0.5), coordsB=ax.transData,
                                       arrowstyle="-", lw=0.5, color=INK2, shrinkA=0,
                                       shrinkB=1.5, zorder=20))
    for i in range(4):   # the colour key: square swatches (marker size in points stays square)
        axn.scatter([0.33 + 0.09 * i], [H - 0.35], s=36, marker="s", color=MAZE_COL[i + 1],
                    edgecolors="none", zorder=6, clip_on=False)
        axn.text(0.33 + 0.09 * i, H - 0.34, str(i + 1), ha="center", va="center", fontsize=4.6,
                 color="white", weight="bold", zorder=7)
    axn.text(0.33 + 0.09 * 3.7, H - 0.35, "colours", ha="left", va="center", fontsize=6.5,
             color=INK)

    # --- (b) the two rules and their interaction: three before/after panels
    cases = [("legal", dict(tint_c=2, door_c=0, pocket=1), True),
             ("illegal (rule B)", dict(tint_c=1, door_c=0, pocket=1), False),
             ("legal: key spent at door", dict(tint_c=1, door_c=1, pocket=1), True)]
    gb = gs[1].subgridspec(1, 3, wspace=0.10)
    prov_cases = []
    for j, (ttl, kw, want) in enumerate(cases):
        inst_m, s0, s1, words, ans = rule_case(**kw)
        assert bool(ans[0]) == want, (ttl, ans)
        axp = fig.add_subplot(gb[0, j])
        off(axp)
        axp.set_xlim(-0.1, 5.9)
        axp.set_ylim(5.55, -0.55)
        axp.set_aspect("equal")
        # before (x 0..2) -> after (x 3.6..5.6)
        for x0, s in ((0.0, s0), (3.6, s1)):
            axp.imshow(CP.draw_grid(inst_m, s.pos, s.ks), extent=(x0, x0 + 2, 1, 0),
                       interpolation="nearest")
            digits(axp, inst_m, s.ks, s.pos, x0=x0, fs=5.2)
            for c in range(2):
                axp.add_patch(Rectangle((x0 + c, 0), 1, 1, fill=False, edgecolor=INK2, lw=0.5))
        axp.add_patch(FancyArrowPatch((2.15, 0.5), (3.45, 0.5), arrowstyle="-|>",
                                      mutation_scale=7, lw=0.8, color=INK2))
        axp.text(2.8, 0.18, "E", ha="center", va="center", fontsize=6, color=INK2)
        axp.text(2.8, 1.05, "✓" if want else "✗", ha="center", va="center", fontsize=9,
                 color=INK, weight="bold")
        axp.text(0.0, -0.35, "(%s) %s" % ("bcd"[j], ttl), ha="left", va="bottom", fontsize=7,
                 color=INK)
        # the five request words and the four-fact answer
        y = 1.9
        axp.text(0.0, y, "words", fontsize=5.8, color=INK2, weight="bold", va="center")
        axp.text(3.6, y, "answer", fontsize=5.8, color=INK2, weight="bold", va="center")
        for i, v in enumerate(words):
            axp.text(0.0, y + 0.68 * (i + 1), WORD_FMT[i], fontsize=5.4, color=INK2, va="center",
                     family="DejaVu Sans Mono")
            axp.text(3.35, y + 0.68 * (i + 1), word_str(i, v), fontsize=5.6, color=INK, va="center",
                     ha="right")
        for i, (nm, v) in enumerate(answer_str(*ans)):
            axp.text(3.6, y + 0.68 * (i + 1), nm, fontsize=5.4, color=INK2, va="center",
                     family="DejaVu Sans Mono")
            axp.text(5.9, y + 0.68 * (i + 1), v, fontsize=5.6, color=INK, va="center", ha="right",
                     weight="bold" if nm == "legal" else "normal")
        prov_cases.append({"panel": ttl, "tint": kw["tint_c"], "door": kw["door_c"],
                           "pocket": kw["pocket"], "action": "E",
                           "words": [int(w) for w in words], "answer": [int(a) for a in ans],
                           "engine_step_and_step_ref_agree": True})
    save(fig, out, "fig_task_cmazet")
    PROV["fig_task_cmazet"] = {
        "a": {"rule": pick_example_state.__doc__.strip(), "generator_seed": int(gen_seed),
              "tint_receipt": rec, "optimal_route": route, "shown_after_step": int(t_now),
              "budget_not_drawn": int(inst.budget), "opt_len": int(inst.opt_len),
              "pocket": int(st.pocket), "keystate": [int(x) for x in st.ks],
              "grid_shape": [int(H), int(W)], "n_colours": int(inst.n_colours)},
        "b_c_d": prov_cases,
        "maze_colours": MAZE_COL,
        "leader_crossings": int(LEADER_CROSSINGS),
        "glyph_source": "every maze cell drawn by w21_figures/cmazet_panels.draw_state (via "
                        "draw_grid), the result panels' own renderer; digits overlaid as text",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=("all", "cmazet"), default="all",
                    help="cmazet: rebuild fig_task_cmazet alone (fig_task_shapes needs 3dshapes.h5 "
                         "assets and is not rebuilt)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.only == "all":
        fig_task_shapes(a.assets, a.out)
    fig_task_cmazet(a.out)
    with open(os.path.join(a.out, "PROVENANCE-task-figures.json"), "w", encoding="utf-8") as f:
        json.dump(PROV, f, indent=1, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
