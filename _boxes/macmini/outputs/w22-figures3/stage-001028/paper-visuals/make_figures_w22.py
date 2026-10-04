"""w22 figures-redraw pass 2 -- the figures brought up to the wave-20/21 evidence.

Built on top of make_figures_w21.py (imported; its style, save(), provenance and forbidden-word
check are used unchanged). Stems drawn here (each overwrites the pass-1 file of the same stem):
  fig_crossover         unchanged from pass 1 (make_figures_w21.fig_crossover)
  fig_ladder            + the w20 131,072-row data-matched rival set; POOL SIZE AND STEP BUDGET in
                        every panel title; a 32,768-row slot left in data/ladder_rival_sets.json
  fig_headline_steps    NEW: rows 3-4 TEST, rival score against optimiser steps at three pool sizes
                        (w20: the operative variable was steps, not data)
  fig_maze_conditions   (a) the condition table (+ the CLU-REL-A10 row); (b) six seeds x two configs,
                        per-seed ceilings, hard/easy marking, the '+rel' series, ties marked
  fig_maze_rel_tuples   NEW: per-tuple held-out n-way, CLU-REL-A10, six seeds (hash-linked replay)
  fig_deletion          the four-measurement rule (w20 matched deletion, 131,072-row pool)
  fig_cmazet_*          pass-1 panels, rebuilt from the verified per-item file; pass 3: under every
                        s_{t+1} a strip shows the facts the judge scores (legality, move, key event,
                        pocket-next), so no two panels with the same picture carry opposite marks
  fig_cmazet_rel_per_arm  pass 3: CLU-REL-A10 per-arm panels from the gate-2 re-read
                        (cmazet_extract_rel.py -> data/cmazet_items_rel.json)

No number is new evidence: every value is read from an artefact named in PROVENANCE.json.
usage: python make_figures_w22.py --out <dir> [--panels <items json>] --merge <PROVENANCE.json>
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures_w21 as W  # noqa: E402
import maze_six  # noqa: E402

M = W.M
plt = W.plt
Line2D, Patch, FuncFormatter = W.Line2D, W.Patch, W.FuncFormatter
ARMS = W.ARMS
INK, INK2, GRID, FLOOR, CONTROL = W.INK, W.INK2, W.GRID, W.FLOOR, W.CONTROL
TEXT_W = W.TEXT_W
load, sha, data, jpath, mean, save = W.load, W.sha, W.data, W.jpath, W.mean, W.save
CLU_C = ARMS["clu"]["color"]
RIVAL_C = "#eb6834"  # categorical slot 2: "best matched rival" as a role (families vary by seed)
HARD_BG = "#efeee9"
SHORT = {"mlp": "MLP", "lstm": "LSTM", "transformer": "Transf.", "cont_mlp": "Cont. MLP"}
FAMS = ["mlp", "lstm", "transformer", "cont_mlp"]
W20 = "w20-shapes-data-matched/SHAPES-MATCHED.json"


def fmt_int(v):
    return f"{v:,}"


# ---------------------------------------------------------------- 1. ladder
def _w20_carried(d, n, fam, seed, store_key):
    pm = d["ladder"][str(n)]["P3_margin"][store_key]["per_family"][fam][str(seed)]
    carried = pm["carried_rival"]
    arm = d["ladder"][str(n)]["arms"][carried]
    leg = carried.split("__")[1].split("@")[0]  # 'declared' | 'tenx'
    return arm["all"]["per_seed"][seed], carried, leg


def read_set(s, key, rungs, seeds):
    """per-rung list of per-seed values (+ per-point step budget) for one arm of one set."""
    if s["reader"] == "w20_carried":
        d = load(s["source"])
        vals, steps = [], []
        for n in rungs:
            rv, rs = [], []
            for sd in seeds:
                v, carried, leg = _w20_carried(d, n, key, sd, s["store_key"])
                ce = d["ladder"][str(n)]["convergence_evidence"][key][str(sd)][leg]
                assert ce["n_fit_rows"] == s["pool_rows"], (n, key, sd)
                rv.append(v)
                rs.append(ce["optimizer_steps"])
            vals.append(rv)
            steps.append(rs)
        return vals, steps
    ser = W.read_series(s, s["arms"][key], rungs, seeds)
    return ser, [[s["steps"]] * len(seeds) for _ in rungs]


def check_steps(s, rungs, seeds):
    c = s["steps_check"]
    srcs = []
    if c["reader"] == "json_path_each":
        for n in rungs:
            for sd in seeds:
                for a in c["arms"]:
                    rel = c["source_template"].format(n=n, seed=sd, arm=a)
                    assert jpath(load(rel), c["path"]) == s["steps"], (s["id"], rel)
                    srcs.append(rel)
    return sorted(set(srcs))


def fig_ladder(out):
    man = data("ladder_rival_sets.json")
    rungs, seeds = man["rungs"], man["seeds"]
    st = man["store"]
    store = W.read_series(st, None, rungs, seeds)
    dstore = load(st["source"])
    for n, ps in zip(rungs, store):
        assert abs(mean(ps) - jpath(dstore, st["mean_template"].format(n=n))) < 1e-9
    fl = man["floor"]
    floor = W.read_series(fl, fl["arm"], rungs, seeds)
    sets = [s for s in man["sets"] if s["status"] == "landed"]
    sources = [st["source"], fl["source"]]
    step_sources = {}
    for s in sets:
        assert jpath(load(s["pool_source"]), s["pool_path"]) == s["pool_rows"], s["id"]
        sources += [s["source"], s["pool_source"]]
        step_sources[s["id"]] = check_steps(s, rungs, seeds)
        sources += step_sources[s["id"]]
        if s["reader"] == "w20_carried":  # the w20 store is the same store, seed by seed
            d = load(s["source"])
            for n, ps in zip(rungs, store):
                assert d["ladder"][str(n)]["arms"][s["store_key"]]["all"]["per_seed"] == ps

    ncol = min(len(sets), 3)
    nrow = (len(sets) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(TEXT_W, 2.75 * nrow), squeeze=False,
                            sharey=True, layout="constrained")
    x = rungs
    set_notes, below = [], []
    any_tenx = False
    for i, s in enumerate(sets):
        ax = axs[i // ncol][i % ncol]
        end_labels = []

        def draw(series, color, marker, ls="-", lw=1.2, label=None, z=3, steps=None):
            mu = [mean(v) for v in series]
            ax.plot(x, mu, color=color, marker=marker, ms=3.4, lw=lw, ls=ls, zorder=z)
            base = min(s["steps"]) if isinstance(s.get("steps"), list) else s.get("steps")
            for j, (n, v) in enumerate(zip(x, series)):
                big = [steps is not None and steps[j][k] > base for k in range(len(v))]
                for k, vv in enumerate(v):
                    ax.scatter([n], [vv], s=7, lw=0.6, zorder=z + 1, edgecolor=color,
                               facecolor=color if big[k] else "white")
            if label:
                end_labels.append([mu[-1], label])
            return mu

        draw(store, CLU_C, ARMS["clu"]["marker"], lw=1.8, label="CLU", z=5)
        draw(floor, FLOOR, ARMS["table"]["marker"], ls="--", lw=1.0, label="table")
        for key in FAMS:
            if key not in s["arms"]:
                continue
            ser, steps = read_set(s, key, rungs, seeds)
            if isinstance(s.get("steps"), list):
                any_tenx |= any(v > min(s["steps"]) for r in steps for v in r)
            if s.get("constant_for_n_ge_3"):
                for v in ser[1:]:
                    assert v == ser[1], f"{s['id']}:{key} not constant on n >= 3"
            a = ARMS[key]
            draw(ser, a["color"], a["marker"], label=SHORT[key], steps=steps)
            # recorded, never a build gate: rung/seed cells where the rival is at or above the store
            for n, sv, rv in zip(x, store, ser):
                for sd, a_, b_ in zip(seeds, sv, rv):
                    if b_ >= a_:
                        below.append(dict(set=s["id"], arm=key, n=n, seed=sd, store=a_, rival=b_))
        end_labels.sort()
        for j in range(1, len(end_labels)):
            end_labels[j][0] = max(end_labels[j][0], end_labels[j - 1][0] + 4.2)
        for yv, lab in end_labels:
            ax.text(x[-1] + 0.18, yv, lab, va="center", ha="left", fontsize=5.8, color=INK)
        ttl = s["title"].split(",")[0]
        ax.set_title(f"({'abcdef'[i]}) {ttl}\n{fmt_int(s['pool_rows'])} rows · {s['steps_label']}",
                     loc="left", fontsize=6.2)
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 0.3, x[-1] + 1.25)
        ax.set_xlabel("n, factors jointly held out\n(n ≥ 3: the same 120 corners)", fontsize=6.4)
        set_notes.append(f"{s['id']}: {s['title']}, {fmt_int(s['pool_rows'])} rows, "
                         f"{s['steps_label']} ({s['pool_note']}); source {s['source']}")
    for j in range(len(sets), nrow * ncol):
        axs[j // ncol][j % ncol].set_visible(False)
    axs[0][0].set_ylabel("distinct valid novel tuples / corner")
    axs[0][0].set_ylim(0, 75)
    handles = [Line2D([0], [0], color=CLU_C, marker="o", lw=1.8, label="CLU, K* = 32")]
    handles += [Line2D([0], [0], color=ARMS[k]["color"], marker=ARMS[k]["marker"], lw=1.2,
                       label=ARMS[k]["name"]) for k in FAMS if any(k in s["arms"] for s in sets)]
    handles += [Line2D([0], [0], color=FLOOR, marker="x", ls="--", lw=1.0,
                       label="Lookup table (floor)"),
                Line2D([0], [0], color=INK2, marker="o", lw=0, markerfacecolor="white", ms=3,
                       label="each seed; line = mean")]
    if any_tenx:
        handles.append(Line2D([0], [0], color=INK2, marker="o", lw=0, ms=3,
                              label="seed carried at 204,800 steps"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6)
    pending = [f"{s['id']} ({fmt_int(s['pool_rows'])} rows): {s['awaiting']}"
               for s in man["sets"] if s["status"] != "landed"]
    save(fig, out, "fig_ladder", sources,
         "3 seeds (0, 1, 2) on every rung; line = mean of the three per-seed values, dots = seeds. "
         "One panel per landed rival column set (data/ladder_rival_sets.json), each beside the "
         "same store (KSTAR-LADDER.json by_rung.*.kstar_K32.store; for the w20 set asserted equal, "
         "seed by seed, to SHAPES-MATCHED.json ladder.*.arms.store_K32_n131072) and table floor. "
         "Pool size and optimiser-step budget in every panel title, each asserted from its "
         "artefact. Sets drawn: " + "; ".join(set_notes) + ". w20 set: the P1-carried fit per "
         "family and seed (P3_margin.store_K32_n131072.per_family.*.*.carried_rival); a filled "
         "dot is a seed carried at its 204,800-step (10x) fit, an open dot the 20,480-step fit. "
         "For n >= 3 the rungs probe the identical 120 corners.",
         extra={"rival_sets_drawn": [s["id"] for s in sets], "rival_sets_not_landed": pending,
                "rival_at_or_above_store": below, "step_sources": step_sources})
    return below


# ---------------------------------------------------------------- 2. headline: steps, not data
def _steps_of(rel_tmpl, seeds, fam=None):
    vals = set()
    srcs = []
    for s in seeds:
        rel = rel_tmpl.format(seed=s, fam=fam)
        d = load(rel)
        vals.add((d["optimizer_steps"], d["n_fit_rows"]))
        srcs.append(rel)
    assert len(vals) == 1, (rel_tmpl, vals)
    return vals.pop(), srcs


def fig_headline_steps(out):
    d = load(W20)
    A = d["rows34"]["arms"]
    seeds = [0, 1, 2]
    R = "w20-shapes-data-matched/rivals/"
    sources = [W20]
    pts = {f: [] for f in FAMS}  # (steps, rows, per_seed, arm key, kind)

    def add(fam, arm, tmpl, kind):
        (st, rows), srcs = _steps_of(R + tmpl, seeds, fam)
        sources.extend(srcs)
        pts[fam].append((st, rows, A[arm]["all"]["per_seed"], arm, kind))

    for f in FAMS:
        add(f, f"{f}__declared@test", "test/seed{seed}/{fam}/declared.json", "declared")
        add(f, f"{f}__tenx@test", "test/seed{seed}/{fam}/tenx.json", "tenx")
    for f in ("mlp", "cont_mlp"):
        add(f, f"{f}__declared@test30x", "test30x/seed{seed}/{fam}/declared.json", "30x")
    add("mlp", "mlp__declared@test16k_stepmatched", "test16k_stepmatched/seed{seed}/{fam}/declared.json",
        "stepmatched")
    add("mlp", "mlp__declared@test32k_stepmatched", "test32k_stepmatched/seed{seed}/{fam}/declared.json",
        "stepmatched")
    # the draft's comparators and the data-matched MLP: their steps from their own artefacts
    lr_src = [f"w12-composition-ladder/runs/shipped_seed{s}/rows_learned_ref_ownpool.json" for s in seeds]
    lr = [load(x)["summary"] for x in lr_src]
    assert {(x["train_info"]["steps"], x["n_fit_rows"]) for x in lr} == {(2560, 32768)}
    assert [x["distinct_valid_novel_tuples_mean"] for x in lr] == \
        A["EXISTING_learned_ref_ownpool@130514_32768rows_2560steps"]["all"]["per_seed"]
    sources += lr_src
    pts["mlp"].append((2560, 32768, A["EXISTING_learned_ref_ownpool@130514_32768rows_2560steps"]
                       ["all"]["per_seed"], "EXISTING_learned_ref_ownpool", "ownpool"))
    bl_src = [f"w12-deletion-baselines/BASELINES-READY.seed_{s}.json" for s in seeds]
    bl = [load(x) for x in bl_src]
    t7 = load("w15-single-design-seeds/T7-3SEEDS.json")
    t7rows = {t7["rival_deletion_column"]["scopes"]["value"]["per_seed"][f"seed{s}"]["families"][f]
              ["as_trained_cost"]["data_rows_read"] for s in seeds for f in FAMS}
    assert t7rows == {16384}, t7rows
    ep = {b["build_scale"]["epochs"] for b in bl}
    bt = {b["pins"]["batch"] for b in bl}
    assert ep == {20} and bt == {256}
    comp_steps = 20 * math.ceil(16384 / 256)
    sources += bl_src + ["w15-single-design-seeds/T7-3SEEDS.json"]
    for f in FAMS:
        pts[f].append((comp_steps, 16384, A[f"w12comp_{f}__as_trained"]["all"]["per_seed"],
                       f"w12comp_{f}__as_trained", "draft"))
    store = {n: A[f"store_K32_n{n}"]["all"]["per_seed"] for n in (16384, 32768, 131072)}
    floor = A["table_FLOOR"]["all"]["per_seed"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.7), sharey=True, layout="constrained",
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    POOL_M = {16384: "s", 32768: "^", 131072: "o"}
    xt = [1280, 2560, 20480, 204800, 614400]
    for ax in (a1, a2):
        lo, hi = min(mean(v) for v in store.values()), max(mean(v) for v in store.values())
        ax.axhspan(lo - 0.15, hi + 0.15, color=CLU_C, alpha=0.18, lw=0, zorder=1)
        ax.axhline(mean(store[131072]), color=CLU_C, lw=1.4, zorder=2)
        ax.axhline(mean(floor), color=FLOOR, lw=0.9, ls="--", zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(xt)
        ax.set_xticklabels(["1,280", "\n2,560", "20,480", "204,800", "\n614,400"], fontsize=5.6)
        ax.minorticks_off()
        ax.set_xlim(900, 900000)
        ax.set_xlabel("rival optimiser steps (log)")
    a1.text(1000, mean(store[131072]) + 1.2,
            "CLU, K* = 32: fit on 16,384 / 32,768 / 131,072 rows\n"
            f"{mean(store[16384]):.1f} / {mean(store[32768]):.1f} / {mean(store[131072]):.1f}",
            fontsize=5.4, color=INK, va="bottom")
    a1.text(30000, mean(floor) + 0.8, f"lookup table (floor) {mean(floor):.1f}", fontsize=5.4,
            color=INK2, va="bottom")
    # (a) the MLP: every (steps, rows) cell that exists
    c = ARMS["mlp"]["color"]
    line131 = sorted((p[0], mean(p[2])) for p in pts["mlp"] if p[1] == 131072)
    a1.plot([p[0] for p in line131], [p[1] for p in line131], color=c, lw=1.0, zorder=3)
    for st, rows, ps, arm, kind in sorted(pts["mlp"]):
        m = POOL_M[rows]
        a1.scatter([st], [mean(ps)], marker=m, s=26, color=c, edgecolor="white", lw=0.6, zorder=5)
        a1.scatter([st] * 3, ps, marker=m, s=7, facecolor="white", edgecolor=c, lw=0.6, zorder=4)
    a1.set_title("(a) MLP rival: every pool × step cell", loc="left", fontsize=7)
    a1.set_ylabel("distinct valid novel tuples / corner\n(rows 3–4, TEST, 109 corners)")
    a1.set_ylim(0, 58)
    a1.legend(handles=[Line2D([0], [0], marker=POOL_M[r], color=c, lw=0, ms=4.5,
                              label=f"{fmt_int(r)} rows") for r in (16384, 32768, 131072)],
              loc="center right", fontsize=5.8, title="fit pool", title_fontsize=5.8)
    # (b) every family: the draft's budget, then the 131,072-row fits at 1x / 10x / 30x
    for f in FAMS:
        a = ARMS[f]
        srt = sorted(pts[f])
        cells = [p for p in srt if p[4] in ("draft", "declared", "tenx", "30x")]
        assert cells[0][4] == "draft" and all(p[1] == 131072 for p in cells[1:])
        # the draft comparator sits on another pool: its segment is dotted, never a solid line
        a2.plot([p[0] for p in cells[:2]], [mean(p[2]) for p in cells[:2]], color=a["color"],
                lw=0.9, ls=":", zorder=3)
        a2.plot([p[0] for p in cells[1:]], [mean(p[2]) for p in cells[1:]], color=a["color"],
                lw=1.0, zorder=3)
        for st, rows, ps, arm, kind in cells:
            a2.scatter([st], [mean(ps)], marker=a["marker"], s=20, color=a["color"],
                       edgecolor="white", lw=0.5, zorder=5)
            a2.scatter([st] * 3, ps, marker=a["marker"], s=6, facecolor="white",
                       edgecolor=a["color"], lw=0.5, zorder=4)
        last = cells[-1]
        a2.text(last[0] * 1.15, mean(last[2]), SHORT[f], fontsize=5.6, va="center", color=INK)
    a2.set_title("(b) Four families: 1,280 steps on 16,384 rows,\nthen 131,072 rows at "
                 "20,480 / 204,800 / 614,400", loc="left", fontsize=6.6)
    save(fig, out, "fig_headline_steps", sources,
         "Rows 3-4 TEST, all 109 corners, 3 seeds (0, 1, 2); big marker = mean, small open = "
         "seeds. Scores: SHAPES-MATCHED.json rows34.arms.*.all.per_seed. Steps and fit rows per "
         "fit from each fit's own declared.json / tenx.json (optimizer_steps, n_fit_rows), "
         "asserted identical across seeds; data-matched MLP 2,560 steps on 32,768 rows from "
         "rows_learned_ref_ownpool.json train_info; the draft's 16,384-row comparators: 1,280 "
         "steps = 20 epochs (BASELINES-READY build_scale.epochs) x ceil(16,384 / 256) batches "
         "(pins.batch 256; 16,384 = T7 as_trained_cost.data_rows_read) - derived, no artefact "
         "stores that step count as a field. Blue line = CLU K*=32 on 131,072 rows; band = its "
         "16,384 / 32,768 / 131,072-row fits. 30x fits exist only for mlp and cont_mlp.",
         extra={"cells": {f: [dict(steps=p[0], rows=p[1], per_seed=p[2], arm=p[3], kind=p[4])
                              for p in sorted(pts[f])] for f in FAMS},
                "store_per_seed": {str(k): v for k, v in store.items()},
                "draft_comparator_steps_derived": comp_steps})


# ---------------------------------------------------------------- 3. maze condition table + six seeds
def _ps_runs(r):
    ps = {}
    for rel in r["runs"]:
        nw = load(rel)["ROWS"]["nway"]
        for k, v in nw.items():
            if k != "n":
                ps.setdefault(k, []).append(v)
    return ps


def fig_maze_conditions(out):
    man = data("maze_conditions.json")
    rows = [r for r in man["rows"] if r["status"] == "landed"]
    sources = [man["rule_source"]]
    comp = []
    for r in rows:
        if r["schema"] == "runs":
            ps = _ps_runs(r)
            sources += r["runs"] + [r["check_source"]]
            for i, rel in enumerate(r["runs"]):
                dd = load(rel)
                assert dd["leg"].upper() == r["leg"] and dd["seed_index"] == i
        else:
            d = load(r["source"])
            sources.append(r["source"])
            assert d["leg"].upper() == r["leg"], (r["id"], d["leg"])
            ps = W.maze_per_seed(d, r["schema"], man["primary_row"])
            if "HEAD_LABEL" in d:
                pass
        sources += r.get("note_sources", [])
        assert len(ps["CLU"]) == 3
        h = W.head_label(ps)
        if r["schema"] != "runs":
            d = load(r["source"])
            if "HEAD_LABEL" in d:
                assert d["HEAD_LABEL"]["PRIMARY"] == h["label"], (r["id"], h["label"])
                assert d["HEAD_LABEL"]["rows"][man["primary_row"]]["R_star"] == h["R_star"]
        else:
            chk = load(r["check_source"])["rows"]["nway"]
            assert chk["R_star"] == h["R_star"] and chk["CLU_per_seed"] == ps["CLU"]
            assert chk["R_star_per_seed"] == h["R_star_per_seed"]
        shown = "ASYMMETRIC" if r.get("asymmetric") else h["label"]
        comp.append((r, ps, h, shown))
    six = maze_six.read()
    sources += six["sources"]

    n = len(comp)
    fig = plt.figure(figsize=(TEXT_W, 0.37 * n + 3.0), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[0.37 * n + 0.72, 2.28])
    gs = top.add_gridspec(1, 3, width_ratios=[1.55, 1.9, 0.62])
    tl, ax, tr = top.add_subplot(gs[0]), top.add_subplot(gs[1]), top.add_subplot(gs[2])
    for a in (tl, tr):
        a.set_axis_off()
    for a in (tl, ax, tr):
        a.set_ylim(n - 0.5, -0.75)
    used = set()
    for i, (r, ps, h, shown) in enumerate(comp):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f3ef", lw=0, zorder=0)
        rname, fam = W.pretty_arm(h["R_star"])
        used.add(fam)
        rc = ARMS[fam]
        dy = 0.14
        ax.scatter(ps["CLU"], [i - dy] * 3, s=16, marker="o", color=CLU_C, zorder=4,
                   edgecolor="white", lw=0.4)
        ax.plot([mean(ps["CLU"])] * 2, [i - dy - 0.13, i - dy + 0.13], color=CLU_C, lw=1.4,
                zorder=3)
        ax.scatter(h["R_star_per_seed"], [i + dy] * 3, s=16, marker=rc["marker"],
                   color=rc["color"], zorder=4, edgecolor="white", lw=0.4)
        ax.plot([mean(h["R_star_per_seed"])] * 2, [i + dy - 0.13, i + dy + 0.13],
                color=rc["color"], lw=1.4, zorder=3)
        muted = r.get("asymmetric")
        yl, ys = (i - 0.30, i - 0.03) if muted else (i - 0.12, i + 0.17)
        tl.text(0.0, yl, r["label"], fontsize=6.4, color=INK2 if muted else INK,
                va="center", ha="left", fontweight="bold")
        sub = f"{r['leg']} · seeds 0–2 · best matched rival: {rname}"
        tl.text(0.0, ys, sub, fontsize=5.2, color=INK2, va="center", ha="left")
        if muted:
            tl.text(0.0, i + 0.30, r["note"].replace(", the CLU", ",\nthe CLU"), fontsize=4.9,
                    color=INK2, va="center", ha="left", style="italic", linespacing=1.1)
        tr.text(0.04, i - 0.08, shown, fontsize=6.4, va="center", ha="left",
                color=INK2 if muted else INK, fontweight="bold")
        tr.text(0.04, i + 0.24, "store gate " + ("holds" if h["store_gate"] else "fails"),
                fontsize=5.0, va="center", ha="left", color=INK2)
    ax.set_xlim(-0.02, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("unseen n-way validity, exact judge (fraction)", fontsize=6.4)
    ax.set_title("(a) CLU (●) and best matched rival, seeds 0–2", loc="left", fontsize=6.8)
    tl.text(0.0, -0.62, "condition", fontsize=6.2, color=INK2, va="center")
    tr.text(0.04, -0.62, "label", fontsize=6.2, color=INK2, va="center")
    handles = [Line2D([0], [0], color=CLU_C, marker="o", lw=0, label="CLU, each seed")]
    handles += [Line2D([0], [0], color=ARMS[f]["color"], marker=ARMS[f]["marker"], lw=0,
                       label=f"{ARMS[f]['name']} rival") for f in FAMS if f in used]
    handles += [Line2D([0], [0], color=INK2, lw=1.4, label="mean of 3 seeds")]
    top.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=5.6)

    # (b) six seeds x two configs
    b1, b2 = bot.subplots(1, 2, sharey=True)
    marks = []
    off = {"clu": -0.21, "rival": -0.07, "rel": 0.07, "ctrl": 0.21}
    for bx, cfg, ttl in ((b1, "A1", "(b) A1 (no relational request coding)"),
                         (b2, "REL", "(c) CLU-REL-A10 (colour-relational request coding)")):
        P = six["config"][cfg]
        filled = cfg == "REL"
        for s in six["seeds"]:
            if six["class"][s] == "hard":
                bx.axvspan(s - 0.45, s + 0.45, color=HARD_BG, lw=0, zorder=0)
            p = P[s]
            ceil = p["ceiling_incl_rel"] if cfg == "REL" else p["ceiling"]
            bx.plot([s - 0.34, s + 0.34], [ceil] * 2, color=INK, lw=1.1, zorder=2,
                    solid_capstyle="butt")
            bx.scatter([s + off["ctrl"]], [p["max_control"]], marker="x", s=18, color=FLOOR,
                       lw=1.0, zorder=4)
            bx.scatter([s + off["rival"]], [p["best_matched"]], marker="^", s=22,
                       facecolor="white", edgecolor=RIVAL_C, lw=1.0, zorder=4)
            if cfg == "REL":
                bx.scatter([s + off["rel"]], [p["best_rel"]], marker="^", s=22, color=RIVAL_C,
                           edgecolor="white", lw=0.4, zorder=5)
            bx.scatter([s + off["clu"]], [p["CLU"]], marker="o", s=24,
                       facecolor=CLU_C if filled else "white", edgecolor=CLU_C, lw=1.0, zorder=6)
            # every seed where a rival or a control reaches the CLU is marked, by rule
            notes = []
            if p["best_matched"] > p["CLU"]:
                notes.append("rival > CLU")
            elif p["best_matched"] == p["CLU"]:
                notes.append("rival = CLU")
            if p["max_control"] >= p["CLU"]:
                notes.append("control " + ("=" if p["max_control"] == p["CLU"] else ">") + " CLU")
            if notes:
                marks.append(dict(config=cfg, seed=s, marks=notes, CLU=p["CLU"],
                                  best_matched=p["best_matched"],
                                  best_matched_arms=p["best_matched_arms"],
                                  max_control=p["max_control"],
                                  max_control_arms=p["max_control_arms"]))
                top_y = max(p["CLU"], p["best_matched"], p["max_control"],
                            p["ceiling_incl_rel"] if cfg == "REL" else p["ceiling"])
                bx.text(s, top_y + 0.03, "\n".join(n_.replace(" CLU", "\nCLU") for n_ in notes),
                        ha="center", va="bottom", fontsize=4.9, color=INK, fontweight="bold",
                        linespacing=1.0)
        bx.set_xticks(six["seeds"])
        bx.set_xticklabels([f"{s}\n{six['class'][s]}" for s in six["seeds"]], fontsize=5.8)
        bx.set_xlim(-0.55, 5.55)
        bx.set_ylim(0, 1.2)
        bx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bx.grid(axis="x", visible=False)
        bx.set_title(ttl, loc="left", fontsize=6.6)
        bx.set_xlabel("seed (shaded: hard, by A1's ceiling)", fontsize=6.2)
    b1.axhline(six["cut"], color=INK2, lw=0.7, ls=":", zorder=1)
    b1.text(5.5, six["cut"] - 0.02, f"hard/easy cut {six['cut']:.3f}", fontsize=5.0,
            color=INK2, ha="right", va="top")
    b1.set_ylabel("unseen n-way, TEST, words + observation", fontsize=6.2)
    h2 = [Line2D([0], [0], marker="o", lw=0, color=CLU_C, markerfacecolor="white", ms=4.5,
                 label="CLU, A1"),
          Line2D([0], [0], marker="o", lw=0, color=CLU_C, ms=4.5, label="CLU, CLU-REL-A10"),
          Line2D([0], [0], marker="^", lw=0, color=RIVAL_C, markerfacecolor="white", ms=4.5,
                 label="best matched rival (of 6)"),
          Line2D([0], [0], marker="^", lw=0, color=RIVAL_C, ms=4.5,
                 label="best rival + relational columns"),
          Line2D([0], [0], marker="x", lw=0, color=FLOOR, ms=4.5, label="highest control"),
          Line2D([0], [0], color=INK, lw=1.1, label="ceiling (best train-fitted arm)")]
    bot.legend(handles=h2, loc="outside lower center", ncol=3, fontsize=5.6)
    labels = {r["id"]: dict(shown=shown, head_rule=h["label"], R_star=h["R_star"],
                            R_star_per_seed=h["R_star_per_seed"], CLU_per_seed=ps["CLU"],
                            diff_per_seed=h["diff_per_seed"], mean_diff=h["mean_diff"],
                            sigma_paired=h["sigma"], seeds_ahead=h["seeds_ahead"],
                            store_gate=h["store_gate"], leg=r["leg"])
              for r, ps, h, shown in comp}
    six_rec = {cfg: {str(s): {k: v for k, v in six["config"][cfg][s].items() if k != "arms"}
                     for s in six["seeds"]} for cfg in ("A1", "REL")}
    save(fig, out, "fig_maze_conditions", sources,
         "(a) CMAZE-T condition table, primary row unseen n-way (exact judge), seeds 0-2; label = "
         "the Head's rule (head_label(), PREREG-w21-words-full-test.md), asserted equal to the "
         "artefact's own R* (and HEAD_LABEL where carried). (b), (c) words + observation TEST, "
         "six seeds, both configs, read from each seed's scored run JSON (ROWS.nway) and asserted "
         "float-equal to REL-NEWSEEDS.json nway_summary; ceilings per REL-NEWSEEDS "
         "ceiling_definition ((c): incl. the +rel arms); hard = A1 ceiling below the cut of "
         "MAZE-T-SEED4.json (largest gap; seeds 1, 4) - never by the CLU's own score. Best "
         "matched rival = max over the six matched arms (3 families x 8k/80k steps) on that seed. "
         "No claim label on the six-seed panels (the rule is written for 3 seeds). Every seed "
         "where a rival or control reaches the CLU is annotated by rule.",
         extra={"labels": labels, "six_seed": six_rec, "marked": marks,
                "class": six["class"], "cut": six["cut"]})
    return marks


# ---------------------------------------------------------------- 4. per-tuple, CLU-REL-A10
def arm_short(arm):
    """artefact arm name -> a short tick label (family, budget); '' for the CLU."""
    if arm == "CLU":
        return ""
    fam = {"mlp": "MLP", "lstm": "LSTM", "transformer": "Transf."}
    if arm.startswith("control: "):
        c = arm.split(": ", 1)[1]
        return " (" + {"landscape_removed": "removed", "landscape_shuffled": "shuffled",
                       "landscape_shuffled_across_wells": "across wells",
                       "random_atoms": "random atoms"}[c] + ")"
    if arm.startswith("diagnostic+rel@"):
        b, f = arm[len("diagnostic+rel@"):].split(": ")
        return f" ({fam[f.split()[0]]}, {b})"
    if arm.startswith("rival@10x-steps: "):
        return f" ({fam[arm.split(': ')[1]]}, 80k)"
    if arm.startswith("rival: "):
        return f" ({fam[arm.split(': ')[1]]}, 8k)"
    raise ValueError(arm)


def fig_maze_rel_tuples(out):
    import matplotlib.colors as mcolors
    six = maze_six.read()
    P = six["config"]["REL"]
    cmap = mcolors.LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fc", "#b7d3f2", "#2a78d6",
                                                                  "#0d3f7e"])
    fig, axs = plt.subplots(3, 2, figsize=(TEXT_W, 4.6), layout="constrained")
    sources = [maze_six.SUMMARY]
    rec = {}
    for s, ax in zip(six["seeds"], axs.flat):
        tt, src = maze_six.pertuple(s)
        sources.append(src)
        p = P[s]
        rows = [("CLU", "CLU")]
        rows.append(("rival", p["best_matched_arms"][0]))
        rows.append(("rival +rel", p["best_rel_arms"][0]))
        rows.append(("control", p["max_control_arms"][0]))
        tuples = list(tt["CLU"].keys())
        n_tot = sum(v[1] for v in tt["CLU"].values())
        # the aggregate of the per-tuple counts is the scored n-way (asserted, every drawn row)
        for _, arm in rows:
            assert list(tt[arm].keys()) == tuples
            assert sum(v[0] for v in tt[arm].values()) / n_tot == p["arms"][arm], (s, arm)
        mat = [[tt[arm][t][0] / tt[arm][t][1] for t in tuples] for _, arm in rows]
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        eq = []
        for j, (lab, arm) in enumerate(rows):
            k = sum(tt[arm][t] == tt["CLU"][t] for t in tuples)
            eq.append(k)
            if j:
                ax.text(len(tuples) - 0.4, j, f" = CLU on {k}/{len(tuples)}", va="center",
                        ha="left", fontsize=5.0, color=INK,
                        fontweight="bold" if k == len(tuples) else "normal")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{lab}{arm_short(arm)} {p['arms'][arm]:.3f}" for lab, arm in rows],
                           fontsize=5.0)
        ax.set_xticks(range(len(tuples)))
        ax.set_xticklabels([str(i + 1) for i in range(len(tuples))], fontsize=4.6)
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"seed {s} ({six['class'][s]}), {n_tot} rows", loc="left", fontsize=6.4)
        ax.set_xlim(-0.5, 18 + 3.6)
        rec[s] = dict(tuples=tuples, rows={lab: arm for lab, arm in rows}, equal_to_clu=eq,
                      ties={lab: p[k] for lab, k in (("rival", "best_matched_arms"),
                                                     ("rival +rel", "best_rel_arms"),
                                                     ("control", "max_control_arms"))})
    for ax in axs[-1]:
        ax.set_xlabel("held-out n-way tuple (artefact order)", fontsize=5.8)
    sm = M.matplotlib.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axs, location="bottom", shrink=0.45, aspect=40, pad=0.02)
    cb.set_label("fraction of the tuple's rows valid (exact judge)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.4)
    save(fig, out, "fig_maze_rel_tuples", sources,
         "CLU-REL-A10, words + observation TEST, seeds 0-5: per held-out n-way tuple, the "
         "fraction of its rows judged valid, for the CLU, the seed's best matched rival, best "
         "+rel diagnostic and highest control (first-listed arm where several tie; all tied arms "
         "in 'ties'). Counts: the zero-fit replay replay/BOTH_test_seed*_PERTUPLE.json "
         "(PER_TUPLE_NWAY, stored [n, correct]; n sums to n_nway_rows, asserted); its objects are hash-identical to the staged objects "
         "(w22 HASH-CROSSCHECK-REL-vs-replay.json) and the per-tuple sums reproduce the scored "
         "n-way of every drawn row (asserted). '= CLU on k/16' counts tuples with identical "
         "(correct, n).",
         extra={"per_seed": {str(k): v for k, v in rec.items()}})
    return rec


# ---------------------------------------------------------------- 5. deletion, four measurements
DSCOPES = [("value", "red value", "value"), ("conjunction_live", "red ∧ sphere", "conjunction"),
           ("item", "1 % items (control)", "item")]


def fig_deletion(out):
    d = load(W20)["deletion_matched"]
    PS = d["per_scope"]
    seeds = ["0", "1", "2"]
    fig = plt.figure(figsize=(TEXT_W, 5.6), layout="constrained")
    r1, r2, r3 = fig.subfigures(3, 1, height_ratios=[1.0, 1.05, 1.1])
    aa, ab, ac = r1.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    # (a) store == refit without; (b) book statistic / rendered features
    xs = range(len(DSCOPES))
    for i, (sc, lab, _) in enumerate(DSCOPES):
        st = PS[sc]["store"]
        a_dev = [st[s]["a"]["max_abs_mode_dev"] for s in seeds]
        b_dev = [st[s]["b"]["b2_book"]["rendered_features_rel_dev_on_survivor_requests"]
                 for s in seeds]
        b1 = [(st[s]["b"]["b1_statistic"]["max_abs_count_dev"],
               st[s]["b"]["b1_statistic"]["max_abs_first_moment_dev"]) for s in seeds]
        assert all(st[s]["a"]["RESULT"] and st[s]["b"]["RESULT"] for s in seeds)
        assert all(x == (0, 0.0) for x in b1)
        aa.scatter([i + (k - 1) * 0.12 for k in range(3)], a_dev, s=12, color=CLU_C, zorder=4,
                   edgecolor="white", lw=0.4)
        ab.scatter([i + (k - 1) * 0.12 for k in range(3)], b_dev, s=12, color=CLU_C, zorder=4,
                   edgecolor="white", lw=0.4)
    for ax, ttl, yl in ((aa, "(a) store = refit without", "max |mode deviation|"),
                        (ab, "(b) book = refit without",
                         "rendered-feature rel. deviation\n(book statistic: 0 on 9/9)")):
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1e-5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(["value", "red ∧\nsphere", "1 %\nitems"], fontsize=5.8)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(ttl, loc="left", fontsize=6.6)
        ax.set_ylabel(yl, fontsize=5.8)
        ax.grid(axis="x", visible=False)
    # (c) display, never a result
    ac.set_axis_off()
    ac.set_title("(c) refusal (a display only)", loc="left", fontsize=6.6)
    lines = []
    for sc, lab, _ in DSCOPES:
        c = [PS[sc]["store"][s]["c_DISPLAY"] for s in seeds]
        assert all(x["DISPLAY_NEVER_A_RESULT"] for x in c)
        if "DECLARED_NA" in c[0]:
            txt = "no hole exists; nothing to refuse"
        else:
            rr = {x["refusal_rate_on_deleted_corner_requests"] for x in c}
            fr = {x["false_refusal_rate_on_survivor_requests"] for x in c}
            nd = {x["n_deleted_corner_requests"] for x in c}
            assert len(rr) == len(fr) == len(nd) == 1
            rr, fr, nd = rr.pop(), fr.pop(), nd.pop()
            txt = (f"refused {rr:.3f} of {nd:,} requests" if rr is not None
                   else "no corner requests it") + f";\nfalse refusal {fr:.3f}"
        lines.append((lab.split("\n")[0], txt))
    for j, (lab, txt) in enumerate(lines):
        ac.text(0.0, 0.80 - j * 0.28, lab, fontsize=6.0, fontweight="bold", color=INK,
                transform=ac.transAxes)
        ac.text(0.0, 0.80 - j * 0.28 - 0.03, txt, fontsize=5.4, color=INK2, transform=ac.transAxes,
                va="top", linespacing=1.15)
    ac.text(0.0, -0.05, "every seed identical", fontsize=5.2, color=INK2, style="italic",
            transform=ac.transAxes)
    # (d) survivors before -> after, three seeds, the random-1 % control beside the conjunction
    dax = r2.subplots(1, 3, sharey=True)
    for ax, (sc, lab, short) in zip(dax, DSCOPES):
        st = PS[sc]["store"]
        for k, s in enumerate(seeds):
            b, a = st[s]["d"]["before_mean"], st[s]["d"]["after_mean"]
            ax.plot([0, 1], [b, a], color=CLU_C, lw=1.0, zorder=3)
            ax.scatter([0], [b], s=14, facecolor="white", edgecolor=CLU_C, lw=0.9, zorder=4)
            ax.scatter([1], [a], s=14, color=CLU_C, zorder=4)
        dl = " / ".join(f"{st[s]['d']['paired_after_minus_before']['mean_paired_delta']:+.2f}"
                        for s in seeds)
        nc = {st[s]["d"]["n_survivor_corners"] for s in seeds}
        assert len(nc) == 1
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["before", "after"], fontsize=6)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"{lab.replace(chr(10), ' ')}, {nc.pop()} corners\nΔ seeds 0/1/2: {dl}",
                     loc="left", fontsize=6.0)
        ax.grid(axis="x", visible=False)
    dax[0].set_ylabel("CLU novel tuples / survivor\ncorner (d)", fontsize=6)
    dax[0].set_ylim(38, 55)
    r2.suptitle("(d) Composition on the corners that survive, before → after",
                x=0.01, ha="left", fontsize=7)
    # (e) paired delta with CI;  (f) CLU after - rival retrained without, same rows
    e, f = r3.subplots(1, 2, gridspec_kw={"width_ratios": [0.8, 1.4]})
    for i, (sc, lab, short) in enumerate(DSCOPES):
        st = PS[sc]["store"]
        for k, s in enumerate(seeds):
            dd = st[s]["d"]["paired_after_minus_before"]
            lo, hi = dd["bootstrap_ci95"]
            x = i + (k - 1) * 0.16
            e.plot([x, x], [lo, hi], color=CLU_C, lw=1.0)
            e.scatter([x], [dd["mean_paired_delta"]], s=12, color=CLU_C, zorder=4)
            z = st[s]["d"]["downdate_minus_refit_without_pipeline"]
            assert z["mean_paired_delta"] == 0.0 and z["corners_tied"] == st[s]["d"]["n_survivor_corners"]
    e.axhline(0, color=INK2, lw=0.7)
    e.set_xticks(range(3))
    e.set_xticklabels(["value", "red ∧\nsphere", "1 % items\n(control)"], fontsize=5.8)
    e.set_ylabel("after − before, paired\n(95 % bootstrap CI)", fontsize=5.8)
    e.set_title("(e) (d) as a paired change, 3 seeds", loc="left", fontsize=6.4)
    e.grid(axis="x", visible=False)
    for i, (sc, lab, short) in enumerate(DSCOPES):
        for j, fam in enumerate(FAMS):
            rv = PS[sc]["rivals"][fam]
            for k, s in enumerate(seeds):
                r_ = rv[s]
                lo, hi = r_["ci95"]
                x = i * 5 + j + (k - 1) * 0.22
                col = ARMS[fam]["color"]
                f.plot([x, x], [lo, hi], color=col, lw=0.9)
                f.scatter([x], [r_["ours_minus_rival_retrain_without"]], s=10, color=col,
                          marker=ARMS[fam]["marker"], zorder=4, edgecolor="white", lw=0.3)
    f.axhline(0, color=INK2, lw=0.7)
    f.set_xticks([1.5, 6.5, 11.5])
    f.set_xticklabels(["value", "red ∧ sphere", "1 % items"], fontsize=5.8)
    f.set_ylabel("CLU after − rival retrained\nwithout, same rows (95 % CI)", fontsize=5.8)
    f.set_title("(f) against each rival retrained without the rows", loc="left", fontsize=6.4)
    f.grid(axis="x", visible=False)
    f.legend(handles=[Line2D([0], [0], marker=ARMS[k]["marker"], color=ARMS[k]["color"], lw=0,
                             ms=4, label=ARMS[k]["name"]) for k in FAMS],
             loc="upper right", fontsize=5.4, ncol=2)
    rows_read = {sc: {fam: sorted({PS[sc]["rivals"][fam][s]["rival_cost_line"]["data_rows_read"]
                                   for s in seeds}) for fam in FAMS} for sc, _, _ in DSCOPES}
    save(fig, out, "fig_deletion", [W20],
         "Rows 3-4 TEST, K* = 32, fit pool 131,072 rows, seeds 0, 1, 2; all from SHAPES-MATCHED.json "
         "deletion_matched.per_scope.<scope> (definition: forget c = downdate the store, downdate "
         "the address book's statistic and refit the book from it, punch the hole). (a) "
         "store.<seed>.a.max_abs_mode_dev; (b) b.b2_book.rendered_features_rel_dev_on_survivor_"
         "requests (b1 count and first-moment deviations are 0 on every leg, asserted); (c) "
         "c_DISPLAY, a display never a result; (d) d.before_mean -> d.after_mean with "
         "paired_after_minus_before.mean_paired_delta; the random 1 % item deletion is the "
         "refit-instability control beside the conjunction; (e) the paired delta with its "
         "bootstrap_ci95 (downdate minus refit-without = 0.000, all corners tied, asserted); (f) "
         "rivals.<fam>.<seed>.ours_minus_rival_retrain_without with ci95, rivals retrained on the "
         "same surviving rows of the same pool at their declared schedule.",
         extra={"rival_rows_read": rows_read})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", default=None)
    ap.add_argument("--merge", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W.fig_crossover(args.out)
    fig_ladder(args.out)
    fig_headline_steps(args.out)
    fig_maze_conditions(args.out)
    fig_maze_rel_tuples(args.out)
    fig_deletion(args.out)
    if args.panels:
        import cmazet_panels  # noqa: E402
        cmazet_panels.build(args.out, args.panels, W.PROV, save_fn=save)
    with open(args.merge, "r", encoding="utf-8") as f:
        base = json.load(f)
    for stem, e in base["figures"].items():
        if stem not in W.PROV:
            e["banner"] = False
            if e.get("placeholder"):
                e["placeholder"] = False
                e["awaiting"] = None
    for stem in W.PROV:
        W.PROV[stem]["built_by"] = "make_figures_w22.py" if stem in (
            "fig_ladder", "fig_headline_steps", "fig_maze_conditions", "fig_maze_rel_tuples",
            "fig_deletion") else W.PROV[stem]["built_by"]
    base["figures"].update(W.PROV)
    base.pop("w21_figures_redraw", None)
    base["figures_redraw"] = {
        "built_by": "make_figures.py then make_figures_w22.py (build.sh)",
        "stems_redrawn_or_added": sorted(W.PROV),
        "cmazet_rel_panels": "drawn (pass 3): gate 2 passed, 132/132 arms float== the scored run "
                             "JSONs (data/cmazet_items_rel.REPRO.json)",
        "cmazet_fact_strip": "pass 3: every CMAZE-T s_{t+1} panel carries the emitted facts the "
                             "judge scores; same-picture opposite-mark pairs 0 "
                             "(cmazet_ambiguity.py)"}
    with open(args.merge, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
