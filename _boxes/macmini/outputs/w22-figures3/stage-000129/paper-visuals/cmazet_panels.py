"""Part C: CMAZE-T per-arm panels, drawn from the verified per-item re-read (cmazet_extract.py).

Nothing here loads a model. Each arm's GENERATED s_{t+1} is built engine-free from the arm's own
emitted outcome by the engine's apply_outcome (the judge's own constructor); the mark is the
exact judge's verdict recorded in the per-item file. Maze content is re-derived from the
deterministic TEST bank and every drawn row is asserted equal to the item record.

Panel sets are DATA (data/cmazet_panels.json): each set names its stem, its row-selection rule and
its column list. A second pass (an improved CLU config) adds a column/set entry there, plus that
config's per-item file, not code.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

VALID = ("✓", "#1f7a32")
INVALID = ("✗", "#c0392b")
MAZE_RGB = {1: (0.80, 0.19, 0.17), 2: (0.20, 0.40, 0.80), 3: (0.88, 0.66, 0.0),
            4: (0.48, 0.25, 0.63)}
DIRS = {"N": "north", "E": "east", "S": "south", "W": "west", "P": "pick up"}
CELL = 10


def _engine():
    for p in (REPO, os.path.join(REPO, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import chlu
    assert os.path.dirname(os.path.dirname(os.path.abspath(chlu.__file__))) == REPO, chlu.__file__
    import chlu.exp_w15_cmaze_t as T
    import w16_cmaze_t as W16
    return T, W16


def _mix(rgb, a):
    return tuple(1.0 - a * (1.0 - x) for x in rgb)


def draw_state(inst, pos, ks, centre, win):
    """RGB crop (win x win cells) around ``centre``: walls dark, tint pale, doors framed in
    their colour (closed: bar across), untaken keys as diamonds, goal as a cross, the agent as a
    black disc whose centre shows the pocket's colour (white = empty)."""
    from w8_envscout.maze_colour import FLOOR, UNTAKEN, USED
    from w9_harness.dynamics import State
    H, W = inst.grid.shape
    img = np.ones((win * CELL, win * CELL, 3), np.float32)
    h = win // 2
    homes = {tuple(inst.key_at[i]): i + 1 for i in range(int(inst.n_colours))}
    pocket = State(pos, ks, 0).pocket
    yy, xx = np.mgrid[0:CELL, 0:CELL]
    ctr = (CELL - 1) / 2.0
    rad = np.hypot(yy - ctr, xx - ctr)
    diamond = (np.abs(yy - ctr) + np.abs(xx - ctr)) <= 2.6
    for i in range(win):
        for j in range(win):
            r, c = centre[0] - h + i, centre[1] - h + j
            blk = img[i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL]
            if not (0 <= r < H and 0 <= c < W) or inst.grid[r, c] != FLOOR:
                blk[:] = (0.23, 0.23, 0.23)
                continue
            t = int(inst.tint[r, c])
            blk[:] = _mix(MAZE_RGB[t], 0.38) if t else (0.96, 0.955, 0.94)
            d = int(inst.door[r, c])
            if d:
                w = 2 if ks[d - 1] != USED else 1
                blk[:w] = MAZE_RGB[d]
                blk[-w:] = MAZE_RGB[d]
                blk[:, :w] = MAZE_RGB[d]
                blk[:, -w:] = MAZE_RGB[d]
                if ks[d - 1] != USED:
                    blk[4:6, 2:-2] = MAZE_RGB[d]
            kc = homes.get((r, c), 0)
            if kc and ks[kc - 1] == UNTAKEN:
                blk[diamond] = MAZE_RGB[kc]
            if (r, c) == tuple(inst.goal):
                blk[(np.abs(yy - xx) <= 0.6) | (np.abs(yy + xx - (CELL - 1)) <= 0.6)] = (0.1, 0.1, 0.1)
            if (r, c) == tuple(pos):
                blk[rad <= 4.2] = (0.05, 0.05, 0.05)
                blk[rad <= 2.4] = MAZE_RGB[pocket] if pocket else (1.0, 1.0, 1.0)
    return img


def draw_grid(inst, pos, ks):
    """The WHOLE maze drawn by draw_state (same function, same glyphs): a window wide enough to
    hold the grid, centred so that window row/column i is maze row/column i, then cropped."""
    H, W = inst.grid.shape
    win = 2 * max(H, W) + 1
    h = win // 2
    return draw_state(inst, pos, ks, (h, h), win)[:H * CELL, :W * CELL]


def arrow(ax, action, win):
    import matplotlib
    h = win // 2
    cx = cy = h * CELL + (CELL - 1) / 2.0
    if action == "P":
        ax.add_patch(matplotlib.patches.Circle((cx, cy), CELL * 0.72, fill=False, ec="#111111",
                                               lw=1.0))
        return
    dy, dx = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}[action]
    ax.annotate("", xy=(cx + dx * CELL * 1.05, cy + dy * CELL * 1.05),
                xytext=(cx + dx * CELL * 0.45, cy + dy * CELL * 0.45),
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.1, mutation_scale=6,
                                shrinkA=0, shrinkB=0))


#: glyphs of the emitted displacement block {stay, N, E, S, W} (exp_w15_cmaze_t.MOVE_ORDER)
KEY_LINE = ("Under each $s_{t+1}$: the four facts the judge scores, as emitted (engine column: the "
            "truth). Line 1: legal / illegal. Line 2: move (↑ → ↓ ←, · = stay), key event (+ colour = "
            "picked up, − colour = used at a door, blank = none), pocket after the move (disc centre; "
            "white = empty). An illegal emission leaves the agent in place and is scored on its pocket "
            "alone, so only the pocket is shown. ✓/✗ is the judge's verdict on all four facts.")
DISP_GLYPH = {0: "·", 1: "↑", 2: "→", 3: "↓", 4: "←"}
INK, INK2 = "#0b0b0b", "#52514e"


def facts_shown(em, k=4):
    """The facts the strip DISPLAYS for an emission ``(legal, disp, ev, pocket_next)``.

    An illegal emission is scored on its pocket alone (judge_transition: apply_outcome ignores
    displacement and event when legal = 0), so only the pocket is shown. A legal emission shows
    all of displacement, key event and pocket. The judge's verdict on an item is a function of
    this tuple, so two panels showing the same tuple cannot carry opposite marks."""
    legal, disp, ev, pn = (int(x) for x in em)
    return (0, pn) if legal == 0 else (1, disp, ev, pn)


def fact_strip(fig, rect, em, k=4):
    """Draw the emitted fact vector under a state panel: line 1 the legality word, line 2
    move / key event / pocket-next. Deterministic; no data-dependent layout."""
    import matplotlib
    f = facts_shown(em, k)
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97, "legal" if f[0] else "illegal", ha="center", va="top", fontsize=4.5,
            color=INK if not f[0] else INK2, fontweight="bold" if not f[0] else "normal")
    yc = 0.27
    if f[0]:
        ax.text(0.17, yc, DISP_GLYPH[f[1]], ha="center", va="center", fontsize=5.4, color=INK)
        ev = f[2]
        if ev:
            c = ev if ev <= k else ev - k
            ax.text(0.40, yc, "+" if ev <= k else "−", ha="center", va="center", fontsize=5.0,
                    color=INK)
            ax.add_patch(matplotlib.patches.Rectangle((0.47, yc - 0.17), 0.13, 0.34,
                                                      fc=MAZE_RGB[c], ec="none"))
    pn = f[-1]
    ax.add_patch(matplotlib.patches.Ellipse((0.83, yc), 0.20, 0.52, fc="#0d0d0d", ec="none"))
    ax.add_patch(matplotlib.patches.Ellipse((0.83, yc), 0.11, 0.29,
                                            fc=MAZE_RGB[pn] if pn else (1.0, 1.0, 1.0),
                                            ec="none"))
    return f


class Maze:
    """TEST bank of one seed, re-derived; rows asserted against the per-item record."""
    _cache = {}

    def __init__(self, seed_rec):
        T, W16 = _engine()
        s = (seed_rec["seed"], json.dumps(seed_rec["bank"], sort_keys=True))
        if s not in Maze._cache:
            insts, rows, _nat, meta = T.build_bank(seed_rec["bank"]["ev_base"], 45, per_tuple=2,
                                                   n_natural=150)
            assert meta == seed_rec["bank"]["meta"], "TEST bank re-derivation differs"
            Maze._cache[s] = insts
        self.T, self.insts = T, Maze._cache[s]

    def states(self, item, emitted=None):
        from w9_harness.dynamics import State
        T = self.T
        inst = self.insts[item["maze"]]
        assert int(inst.seed) == item["maze_seed"]
        st = State(tuple(item["pos"]), tuple(item["ks"]), 0)
        legal, disp, ev, pn, nxt = T.outcome_of(inst, st, item["action"])
        assert [int(legal), int(disp), int(ev), int(pn)] == item["truth"], "row differs from item"
        if emitted is None:
            return inst, st, nxt
        gen, _coherent = T.apply_outcome(inst, st, *emitted[:3])
        return inst, st, gen


def row_label(item, seed):
    w = item["words"]
    k = 4
    door = {0: "no door", k + 1: "open door", k + 2: "wall"}.get(w[2], f"door {w[2]}")
    out = "legal" if item["truth"][0] else "illegal"
    fam = item["family"].replace("_", " ")
    s = "" if seed == 0 else f" (seed {seed})"
    return (f"{fam}{s}\npocket {w[0] or 'empty'} · {door}\n"
            f"tint {w[3] or 'none'} · {DIRS[item['action']]} → {out}")


def grid(plt, out_stem, cols, rows, save_fn, sources, notes, extra, win=5, panel=0.40,
         left=1.35, title=None, col_groups=None, blocks=1, strip=0.18, key_line=True):
    """cols: [(header, colour or None, key)], key None = s_t, 'engine' = true s_{t+1}, else arm.
    rows: [(label, seed_rec, item, {arm: (verdict, emitted)})]."""
    import matplotlib
    nr, nc = len(rows), len(cols)
    per = -(-nr // blocks)
    top = 0.42 + (0.2 if col_groups else 0.0)
    bw = left + nc * panel + 0.08
    fw = blocks * bw
    pitch = panel + strip
    import textwrap
    key_lines = textwrap.wrap(KEY_LINE, max(40, int(fw * 72 / 2.45))) if key_line else []
    foot = 0.05 + 0.078 * len(key_lines) if key_lines else 0.0
    fh = top + per * pitch + 0.08 + foot
    fig = plt.figure(figsize=(fw, fh))
    gap = 0.04
    shown = []
    for rr, (lab, srec, item, arms) in enumerate(rows):
        mz = Maze(srec)
        bx, r = (rr // per) * bw, rr % per
        for c, (hdr, colr, key) in enumerate(cols):
            x0 = (bx + left + c * panel + gap / 2) / fw
            y0 = 1.0 - (top + r * pitch + panel - gap / 2) / fh
            ax = fig.add_axes([x0, y0, (panel - gap) / fw, (panel - gap) / fh])
            ax.set_xticks([])
            ax.set_yticks([])
            if key is None:
                inst, st, nxt = mz.states(item)
                ax.imshow(draw_state(inst, st.pos, st.ks, st.pos, win), interpolation="nearest")
                arrow(ax, item["action"], win)
                v = None
            elif key == "engine":
                inst, st, nxt = mz.states(item)
                ax.imshow(draw_state(inst, nxt.pos, nxt.ks, st.pos, win), interpolation="nearest")
                v = None
                em = item["truth"]
            else:
                ok, em = arms[key]
                inst, st, gen = mz.states(item, em)
                ax.imshow(draw_state(inst, gen.pos, gen.ks, st.pos, win), interpolation="nearest")
                v = ok
            for s in ax.spines.values():
                s.set_edgecolor("#9a9994")
                s.set_linewidth(0.5)
            if key is not None:
                srect = [x0, 1.0 - (top + r * pitch + panel + strip - 0.045) / fh,
                         (panel - gap) / fw, (strip - 0.045) / fh]
                shown.append(dict(row=rr, col=hdr.replace("\n", " "), key=key,
                                  emitted=[int(x) for x in em],
                                  facts_shown=list(fact_strip(fig, srect, em)),
                                  mark=None if v is None else bool(v)))
            if v is not None:
                ch, fc = VALID if v else INVALID
                ax.text(0.96, 0.96, ch, transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.5, fontweight="bold", color="white",
                        bbox=dict(boxstyle="circle,pad=0.1", fc=fc, ec="white", lw=0.5))
        fig.text((bx + left - 0.06) / fw, 1.0 - (top + r * pitch + 0.5 * panel) / fh, lab, ha="right",
                 va="center", fontsize=4.9, linespacing=1.2, color="#0b0b0b")
    for bx, (c, (hdr, colr, key)) in [(b * bw, cc) for b in range(blocks) for cc in enumerate(cols)]:
        xc = (bx + left + (c + 0.5) * panel) / fw
        fig.text(xc, 1.0 - (top - 0.07) / fh, hdr, ha="center", va="bottom", fontsize=4.8,
                 color="#0b0b0b", linespacing=1.1)
        if colr:
            fig.add_artist(matplotlib.lines.Line2D(
                [(bx + left + c * panel + 0.06) / fw, (bx + left + (c + 1) * panel - 0.06) / fw],
                [1.0 - (top - 0.035) / fh] * 2, color=colr, lw=2.2, solid_capstyle="butt"))
    if col_groups:
        for lab, a, b in col_groups:
            if not lab:
                continue
            xa = (left + a * panel + 0.04) / fw
            xb = (left + (b + 1) * panel - 0.04) / fw
            y = 1.0 - (top - 0.40) / fh
            fig.add_artist(matplotlib.lines.Line2D([xa, xb], [y, y], color="#55544f", lw=0.6))
            fig.text((xa + xb) / 2, y + 0.03 / fh, lab, ha="center", va="bottom", fontsize=5.0,
                     color="#333333")
    if title:
        fig.text(0.005, 0.995, title, ha="left", va="top", fontsize=6.2, color="#0b0b0b")
    if key_lines:
        fig.text(0.005, 0.04 / fh, "\n".join(key_lines), ha="left", va="bottom", fontsize=4.9,
                 color=INK2, linespacing=1.25)
    extra = dict(extra, facts_shown=shown)
    save_fn(fig, os.path.dirname(out_stem), os.path.basename(out_stem), sources, notes, extra)


def build(out, items_path, PROV, save_fn):
    import make_figures as M
    plt = M.plt
    spec = json.load(open(os.path.join(HERE, "data", "cmazet_panels.json")))
    data = json.load(open(items_path))
    assert data["summary"]["all_arms_equal_artefact"], "re-read gate did not pass"
    seeds = {r["seed"]: r for r in data["seeds"]}
    arms = spec["arms"]
    colour = {a["key"]: (M.ARMS[a["arm_style"]]["color"] if a.get("arm_style") else M.CONTROL)
              for a in arms}
    srcs = sorted({r["artefact"] for r in data["seeds"]})
    src_notes = ("per-item re-read of the persisted w19 BOTH TEST objects (hash-verified; every "
                 "arm's n-way score equals the scored artefact, asserted in cmazet_extract.py); "
                 "items file sha256 recorded under items_sha256")
    import hashlib
    ish = hashlib.sha256(open(items_path, "rb").read()).hexdigest()

    def verdicts(srec, i, keys, table=None):
        V = table if table is not None else srec["verdicts"]
        return {k: (bool(V[k][i][0]), V[k][i][2]) for k in keys}

    arm_cols = [(a["header"], colour[a["key"]], a["key"]) for a in arms]
    groups, pos = [], 2
    for g in spec["groups"]:
        n = sum(1 for a in arms if a["group"] == g)
        groups.append((g, pos, pos + n - 1))
        pos += n
    base_cols = [("$s_t$ + action", None, None), ("engine\n$s_{t+1}$", None, "engine")]
    fams = spec["families_order"]
    base = dict(seeds=seeds, arms=arms, srcs=srcs, src_notes=src_notes, ish=ish)
    for ps in spec["panel_sets"]:
        if ps.get("status", "landed") != "landed":
            continue
        if ps.get("items"):
            # a panel set with its OWN gate-passed per-item file (e.g. CLU-REL-A10)
            ip = os.path.join(HERE, "data", ps["items"])
            d2 = json.load(open(ip))
            assert d2["summary"]["all_arms_equal_artefact"], "re-read gate did not pass: " + ip
            assert d2["summary"]["config"] == ps["config"], (d2["summary"]["config"], ps)
            seeds = {r["seed"]: r for r in d2["seeds"]}
            arms = base["arms"] + ps.get("extra_arms", [])
            srcs = sorted({r["artefact"] for r in d2["seeds"]})
            src_notes = ps["gate_note"] + " items file sha256 recorded under items_sha256"
            ish = hashlib.sha256(open(ip, "rb").read()).hexdigest()
        else:
            seeds, arms, srcs = base["seeds"], base["arms"], base["srcs"]
            src_notes, ish = base["src_notes"], base["ish"]
        rows, chosen = [], []
        if ps["select"] == "first_per_family":
            for f in fams:
                for sd in ps["seed_order"]:
                    srec = seeds[sd]
                    hit = next((it for it in srec["items"] if it["family"] == f), None)
                    if hit:
                        rows.append((row_label(hit, sd), srec, hit,
                                     verdicts(srec, hit["i"], [a["key"] for a in arms])))
                        chosen.append(dict(seed=sd, i=hit["i"], family=f))
                        break
            cols = base_cols + arm_cols
            cg = groups
        elif ps["select"] == "interaction_per_direction":
            srec = seeds[ps["seed_order"][0]]
            for d in "NESW":
                hit = next((it for it in srec["items"]
                            if it["family"] == "interaction" and it["action"] == d), None)
                if hit:
                    rows.append((row_label(hit, srec["seed"]), srec, hit,
                                 verdicts(srec, hit["i"], [a["key"] for a in arms])))
                    chosen.append(dict(seed=srec["seed"], i=hit["i"], direction=d))
            cols = base_cols + arm_cols
            cg = groups
        elif ps["select"] == "success_failure_per_arm":
            srec = seeds[ps["seed_order"][0]]
            rng = np.random.default_rng(ps["rng_seed"])
            cols = base_cols + [("the arm's\n$s_{t+1}$", None, "subject")]
            for a in arms:
                V = srec["verdicts"][a["key"]]
                for want, tag in ((1, "succeeds"), (0, "fails")):
                    cand = [j for j, v in enumerate(V) if v[0] == want]
                    if not cand:
                        chosen.append(dict(arm=a["key"], want=tag, none_exists=True))
                        continue
                    j = int(cand[int(rng.integers(len(cand)))])
                    it = srec["items"][j]
                    lab = f"{a['name']}\n{tag} · " + row_label(it, srec["seed"])
                    rows.append((lab, srec, it, {"subject": (bool(V[j][0]), V[j][2])}))
                    chosen.append(dict(arm=a["key"], want=tag, i=j, n_candidates=len(cand)))
            cg = None
        elif ps["select"] == "deletion_first_per_family":
            srec = seeds[ps["seed_order"][0]]
            D = srec["deletion"][ps["scope"]]
            dcols = ps["columns"]
            cols = base_cols + [(c["header"], M.ARMS[c["arm_style"]]["color"], c["key"])
                                for c in dcols]
            for f in fams:
                pos_ = next((p for p, i in enumerate(D["row_index_in_nway"])
                             if srec["items"][i]["family"] == f), None)
                if pos_ is None:
                    continue
                it = srec["items"][D["row_index_in_nway"][pos_]]
                rows.append((row_label(it, srec["seed"]), srec, it,
                             verdicts(srec, pos_, [c["key"] for c in dcols], D["verdicts"])))
                chosen.append(dict(seed=srec["seed"], i=it["i"], family=f))
            cg = [(g["label"], g["from"], g["to"]) for g in ps.get("groups", [])]
        else:
            raise ValueError(ps["select"])
        grid(plt, os.path.join(out, ps["stem"]), cols, rows, save_fn, srcs,
             ps["notes"] + " " + src_notes,
             {"items_sha256": ish, "selection": chosen, "select_rule": ps["select"],
              "part": "C"},
             win=spec["window_cells"], panel=ps.get("panel_in", 0.40),
             left=ps.get("left_in", 1.35), col_groups=cg, blocks=ps.get("blocks", 1))
