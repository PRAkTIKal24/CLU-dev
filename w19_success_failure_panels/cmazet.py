"""CMAZE-T panels: s_t + action, the ENGINE's true s_{t+1}, and one column per arm.

WHAT IS BUILT FROM WHAT
* The transitions are the scored TEST leg's own rows, re-derived from the deterministic
  engine exactly as ``scripts/w16_cmaze_t.py::full`` derives them (seed 0: TEST bank
  generator block 2,300,000.., ``make_holdout(4, seed=0)``, ``split_rows``, ``cap_split``).
  The re-derivation is ASSERTED against the primary record ``w16-maze-wm/t1/A1_test_seed0.json``
  (bank meta, hold-out tuples, per-class row counts) before anything is drawn.
* s_t and the true s_{t+1} come from the engine's own stepper (``outcome_of`` ->
  ``w9_harness.dynamics.step``); legality is the engine's flag.  No arm is involved.
* Every ARM column is DECLARED NOT BUILDABLE and drawn hatched: w16 persisted no arm
  object (no frozen codec, no CLU store, no rival weights) and no per-transition
  prediction -- ``A1_test_seed*.json`` holds aggregate scores only.  Re-training them here
  would be a new TEST-leg run with a NEW codec, which the task forbids (zero scored runs;
  never substitute a codec).  NEEDS-ARTEFACTS.json names what would unblock them.

SELECTION RULES (fixed before drawing; nothing about any arm enters)
* families      the first TEST n-way row, in the scorer's (capped) order, of each of the
                five named families ``NWAY_FAMILIES``;
* door-then-tint  the first TEST n-way row of family ``interaction`` for each move
                direction that has one (N, E, S, W);
* deletion      the deleted colour c is the primary record's ``DELETION.deleted_colour``;
                the transition is the first TEST n-way ``interaction`` row whose pocket is
                c (else the first n-way row mentioning c).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EVID_T = "/Users/batmac/clu/evidence/outputs/w16-maze-wm/t1/A1_test_seed{}.json"
EVID = Path(EVID_T.format(0))
FIGS = {"families": "rc_w19_cmazet_families",
        "door_then_tint": "rc_w19_cmazet_door_then_tint",
        "deletion": "rc_w19_cmazet_deletion"}
#: CMAZE-T arm order, names and colours as fig_maze_wm (make_maze_figures.py) and FIGURES.md
ARMS_T = (("CLU", "CLU", "#2a78d6"), ("rival: transformer", "Transformer", "#e87ba4"),
          ("rival: mlp", "MLP", "#1baf7a"), ("rival: lstm", "LSTM", "#eda100"))
#: maze content colours (colour index 1..4); content, not arms
MAZE_RGB = {1: (0.80, 0.19, 0.17), 2: (0.20, 0.40, 0.80), 3: (0.88, 0.66, 0.0),
            4: (0.48, 0.25, 0.63)}
DIR_NAMES = ("N", "E", "S", "W", "P")
WIN, CELL = 7, 10


def _paths():
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO))


def rebuild_test_rows(seed=0):
    """The TEST leg's scored rows for one seed, asserted against the primary record."""
    _paths()
    import chlu.exp_w15_cmaze_t as T
    import w16_cmaze_t as W16
    evp = Path(EVID_T.format(seed))
    ev = json.loads(evp.read_text(encoding="utf-8"))
    assert ev["seed_index"] == seed and ev["leg"] == "test" and ev["scored_run"] is True
    base = 2_000_000 + 400_000 * seed
    ev_base = base + 300_000
    insts, rows, _nat, meta = T.build_bank(ev_base, 45, per_tuple=2, n_natural=150)
    want = {k: v for k, v in ev["banks"]["test"].items() if k != "seed_block"}
    if meta != want:
        raise SystemExit(f"TEST bank re-derivation differs from the primary record: {meta} != {want}")
    ho = T.make_holdout(W16.K_COLOURS, seed=seed)
    if ([list(p) for p in ho.pairs] != ev["holdout"]["pairs"]
            or [list(c) for c in ho.nway] != ev["holdout"]["nway"]
            or ho.value_colour != ev["holdout"]["value_colour"]):
        raise SystemExit("hold-out re-derivation differs from the primary record")
    split = W16.cap_split(T.split_rows(rows, ho, W16.K_COLOURS), seed)
    got = {k: len(v) for k, v in split.items()}
    if got != ev["holdout"]["n_eval_rows_scored"]:
        raise SystemExit(f"scored row counts differ: {got} != {ev['holdout']['n_eval_rows_scored']}")
    check = {"bank_meta_equal": True, "holdout_equal": True, "n_eval_rows_scored": got,
             "against": str(evp)}
    return T, W16, insts, split, ho, ev, check


def family_of(T, k, value_colour, key4):
    fam = T._nway_candidates(k, value_colour)
    hits = [f for f in T.NWAY_FAMILIES if tuple(key4) in fam[f]]
    assert len(hits) == 1, (key4, hits)
    return hits[0]


def select(T, W16, split, ho, ev):
    k = W16.K_COLOURS
    nway = split["nway"]
    fams = [family_of(T, k, ho.value_colour, (r.words[0], r.words[2], r.words[3], r.words[4]))
            for r in nway]
    sel = {"families": [], "door_then_tint": [], "deletion": []}
    for f in T.NWAY_FAMILIES:
        i = next((j for j, g in enumerate(fams) if g == f), None)
        if i is not None:
            sel["families"].append(i)
    for d in range(4):
        i = next((j for j, (g, r) in enumerate(zip(fams, nway))
                  if g == "interaction" and r.words[4] == d), None)
        if i is not None:
            sel["door_then_tint"].append(i)
    c = int(ev["DELETION"]["deleted_colour"])
    i = next((j for j, (g, r) in enumerate(zip(fams, nway))
              if g == "interaction" and r.words[0] == c), None)
    if i is None:
        i = next(j for j, r in enumerate(nway) if c in (r.words[0], r.words[1], r.words[2], r.words[3]))
    sel["deletion"].append(i)
    return sel, fams, c


def _mix(rgb, a):
    return tuple(1.0 - a * (1.0 - x) for x in rgb)


def draw_state(T, inst, pos, ks, centre):
    """RGB crop (WIN x WIN cells) of the maze around ``centre`` at keystate ``ks``."""
    from w8_envscout.maze_colour import FLOOR, UNTAKEN, USED
    from w9_harness.dynamics import State
    H, W = inst.grid.shape
    img = np.ones((WIN * CELL, WIN * CELL, 3), np.float32)
    h = WIN // 2
    homes = {tuple(inst.key_at[i]): i + 1 for i in range(int(inst.n_colours))}
    pocket = State(pos, ks, 0).pocket
    yy, xx = np.mgrid[0:CELL, 0:CELL]
    ctr = (CELL - 1) / 2.0
    rad = np.hypot(yy - ctr, xx - ctr)
    diamond = (np.abs(yy - ctr) + np.abs(xx - ctr)) <= 2.6
    for i in range(WIN):
        for j in range(WIN):
            r, c = centre[0] - h + i, centre[1] - h + j
            blk = img[i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL]
            if not (0 <= r < H and 0 <= c < W) or inst.grid[r, c] != FLOOR:
                blk[:] = (0.23, 0.23, 0.23)
                continue
            t = int(inst.tint[r, c])
            blk[:] = _mix(MAZE_RGB[t], 0.38) if t else (0.96, 0.955, 0.94)
            d = int(inst.door[r, c])
            if d:                                            # the tint stays visible inside
                w = 2 if ks[d - 1] != USED else 1            # closed: thick frame; open: thin
                blk[:w] = MAZE_RGB[d]
                blk[-w:] = MAZE_RGB[d]
                blk[:, :w] = MAZE_RGB[d]
                blk[:, -w:] = MAZE_RGB[d]
                if ks[d - 1] != USED:                        # closed: a bar across the doorway
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


def arrow_overlay(action):
    """The action on s_t: an arrow from the agent toward the target cell (P: a ring)."""
    h = WIN // 2
    cx = cy = h * CELL + (CELL - 1) / 2.0

    def fn(ax):
        import matplotlib
        if action == "P":
            ax.add_patch(matplotlib.patches.Circle((cx, cy), CELL * 0.72, fill=False,
                                                   ec="#111111", lw=1.2))
            return
        dy, dx = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}[action]
        ax.annotate("", xy=(cx + dx * CELL * 1.05, cy + dy * CELL * 1.05),
                    xytext=(cx + dx * CELL * 0.45, cy + dy * CELL * 0.45),
                    arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.3,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
    return fn


def words_label(T, row, fam):
    w = row.words
    k = 4
    door = {0: "none", k + 1: "open", k + 2: "wall"}.get(w[2], str(w[2]))
    out = "legal" if row.legal else "illegal"
    return (f"{fam.replace('_', ' ')}\npocket {w[0] or 'empty'} · door {door} · "
            f"tint {w[3] or 'none'}\nmove {DIR_NAMES[w[4]]} \u2192 {out}")


def build(out_dir: Path) -> dict:
    from .figure import draw_grid
    T, W16, insts, split, ho, ev, check = rebuild_test_rows(0)
    sel, fams, c_del = select(T, W16, split, ho, ev)
    nway = split["nway"]

    def truth(row, ins=insts):
        inst = ins[row.maze]
        from w9_harness.dynamics import State
        st = State(tuple(row.pos), tuple(row.ks), 0)
        legal, disp, evn, pn, nxt = T.outcome_of(inst, st, row.action)
        assert (legal, disp, evn, pn) == (row.legal, row.disp, row.ev, row.pocket_next)
        a = draw_state(T, inst, st.pos, st.ks, st.pos)
        b = draw_state(T, inst, nxt.pos, nxt.ks, st.pos)
        return a, b, nxt

    records, figs = {}, {}
    arm_labels = [l for _, l, _ in ARMS_T]
    arm_cols = [c for _, _, c in ARMS_T]
    # the families set: seed 0 rows; a family with NO seed-0 TEST n-way row is taken from
    # the first seed that has one (labelled with its seed) -- declared in the record
    fam_rows = [(0, insts, nway[i], fams[i], i) for i in sel["families"]]
    have = {fams[i] for i in sel["families"]}
    fallback = {}
    for f in T.NWAY_FAMILIES:
        if f in have:
            continue
        for sd in (1, 2):
            T2, W2, insts2, split2, ho2, ev2, _c = rebuild_test_rows(sd)
            fam2 = [family_of(T2, W2.K_COLOURS, ho2.value_colour,
                              (r.words[0], r.words[2], r.words[3], r.words[4]))
                    for r in split2["nway"]]
            j = next((q for q, g in enumerate(fam2) if g == f), None)
            if j is not None:
                fam_rows.append((sd, insts2, split2["nway"][j], f, j))
                fallback[f] = {"seed_0_rows": 0, "taken_from_seed": sd, "index_in_split": j}
                break
        else:
            fallback[f] = "DECLARED: no TEST n-way row of this family at any seed"
    order = {f: n for n, f in enumerate(T.NWAY_FAMILIES)}
    fam_rows.sort(key=lambda t: order[t[3]])
    sets = {"families": fam_rows,
            "door_then_tint": [(0, insts, nway[i], fams[i], i) for i in sel["door_then_tint"]]}
    for key, rows_ in sets.items():
        imgs, verd, rl, ov, rec = [], [], [], {}, []
        for r_i, (sd, ins, row, fam, i) in enumerate(rows_):
            a, b, nxt = truth(row, ins)
            imgs.append([a, b] + [None] * len(ARMS_T))
            verd.append([None] * (2 + len(ARMS_T)))
            lab = words_label(T, row, fam)
            if sd != 0:
                lab = lab.replace("\n", f" (seed {sd})\n", 1)
            rl.append(lab)
            ov[(r_i, 0)] = arrow_overlay(row.action)
            rec.append(dict(_record(T, ins, row, fam, i, nxt), seed=sd))
        figs[key] = draw_grid(out_dir / FIGS[key], ["$s_t$ + action", "engine $s_{t+1}$"] + arm_labels,
                              ["#333333", "#333333"] + arm_cols, rl, imgs, verd, overlays=ov,
                              left=2.3)
        records[key] = rec
    records["families_fallback"] = fallback
    # deletion
    i = sel["deletion"][0]
    row = nway[i]
    a, b, nxt = truth(row)
    cols = ["$s_t$ + action", "engine $s_{t+1}$", "CLU\nbefore", "CLU after\n(downdate)"]
    ccol = ["#333333", "#333333", "#2a78d6", "#2a78d6"]
    groups = [("CLU", 2, 3)]
    for j, (_k, lab, colr) in enumerate(ARMS_T[1:]):
        cols += [f"{lab}\nas trained", f"{lab}\nretrained\nwithout"]
        ccol += [colr, colr]
        groups.append((lab, 4 + 2 * j, 5 + 2 * j))
    figs["deletion"] = draw_grid(out_dir / FIGS["deletion"], cols, ccol,
                                 [words_label(T, row, fams[i])],
                                 [[a, b] + [None] * (len(cols) - 2)], [[None] * len(cols)],
                                 col_groups=groups, overlays={(0, 0): arrow_overlay(row.action)},
                                 left=2.3)
    records["deletion"] = [dict(_record(T, insts, row, fams[i], i, nxt), seed=0, deleted_colour=c_del,
                                deleted_colour_source=f"{EVID}::DELETION.deleted_colour")]
    return {"figures": figs, "transitions": records, "selection": sel,
            "re_derivation_check": check}


def _record(T, insts, row, fam, idx, nxt):
    inst = insts[row.maze]
    return {"split": "nway (TEST, capped scorer order)", "index_in_split": int(idx),
            "family": fam, "maze_index_in_test_bank": int(row.maze),
            "maze_generator_seed": int(inst.seed), "prefix_from_start": row.prefix,
            "s_t": {"pos": list(map(int, row.pos)), "keystate": list(map(int, row.ks))},
            "action": row.action,
            "words": dict(zip(T.WORD_NAMES, map(int, row.words))),
            "true_outcome": {"legal": int(row.legal), "displacement": int(row.disp),
                             "key_event": int(row.ev), "pocket_next": int(row.pocket_next),
                             "rule_b_bites": bool(row.bites_b)},
            "s_t+1": {"pos": list(map(int, nxt.pos)), "keystate": list(map(int, nxt.ks))},
            "arm_outputs": "DECLARED NOT BUILDABLE (no persisted arm object or prediction)"}


def family_counts(seeds=(0, 1, 2)) -> dict:
    """TEST n-way rows per named family, per seed (re-derived + asserted as above)."""
    out = {}
    for sd in seeds:
        T, W16, _i, split, ho, _ev, chk = rebuild_test_rows(sd)
        cnt = {f: 0 for f in T.NWAY_FAMILIES}
        held = {f: 0 for f in T.NWAY_FAMILIES}
        for r in split["nway"]:
            cnt[family_of(T, W16.K_COLOURS, ho.value_colour,
                          (r.words[0], r.words[2], r.words[3], r.words[4]))] += 1
        for c in ho.nway:
            held[family_of(T, W16.K_COLOURS, ho.value_colour, c)] += 1
        out[str(sd)] = {"test_nway_rows_per_family": cnt,
                        "held_out_tuples_per_family": held, "re_derivation_check": chk}
    return out
