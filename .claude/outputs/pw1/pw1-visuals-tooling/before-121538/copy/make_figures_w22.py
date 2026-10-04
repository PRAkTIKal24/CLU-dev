"""w22 figures-redraw pass 2 -- the figures brought up to the wave-20/21 evidence.

Built on top of make_figures_w21.py (imported; its style, save(), provenance and forbidden-word
check are used unchanged). Stems drawn here (each overwrites the pass-1 file of the same stem):
  fig_crossover         unchanged from pass 1 (make_figures_w21.fig_crossover)
  fig_ladder            PW-1 item 10: ONE panel, the w20 131,072-row data-matched set; no title
                        (pool size and steps in the caption), legend at the right, no end labels
  fig_headline_steps    NEW: rows 3-4 TEST, rival score against optimiser steps at three pool sizes
                        (w20: the operative variable was steps, not data)
  fig_maze_conditions   PW-1 item 1: (a) four input conditions, every seed; (b) / (c) six seeds
                        without / with relational coding; no labels, ceiling, hard/easy or overlay
  fig_maze_rel_tuples   retired from the build (PW-1 item 15); the function is kept, not called
  fig_deletion          PW-1 item 12: two panels (former (e), (f)); the four measurements' other
                        values are asserted and recorded in PROVENANCE, no longer drawn
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

    # PW-1 item 10 (Head, 2026-09-25): ONE panel, the data-matched 131,072-row set; no title (pool
    # and steps go to the caption), x label only "n, factors jointly held out", legend at the right
    # arranged vertically, no line-end labels, legend names per item 10.6, y label per item 10.7
    assert [s["id"] for s in sets] == ["data_matched_131072"], [s["id"] for s in sets]
    s = sets[0]
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.6), layout="constrained")
    x = rungs
    set_notes, below = [], []
    any_tenx = False
    base = min(s["steps"])
    tenx = max(s["steps"])
    assert s["steps"] == [base, tenx] and tenx == 10 * base, s["steps"]  # the "10x" in the legend
    tenx_seen = []

    def draw(series, color, marker, ls="-", lw=1.2, z=3, steps=None):
        mu = [mean(v) for v in series]
        ax.plot(x, mu, color=color, marker=marker, ms=3.4, lw=lw, ls=ls, zorder=z)
        for j, (n, v) in enumerate(zip(x, series)):
            big = [steps is not None and steps[j][k] > base for k in range(len(v))]
            for k, vv in enumerate(v):
                ax.scatter([n], [vv], s=7, lw=0.6, zorder=z + 1, edgecolor=color,
                           facecolor=color if big[k] else "white")
        return mu

    draw(store, CLU_C, ARMS["clu"]["marker"], lw=1.8, z=5)
    draw(floor, FLOOR, ARMS["table"]["marker"], ls="--", lw=1.0)
    for key in FAMS:
        if key not in s["arms"]:
            continue
        ser, steps = read_set(s, key, rungs, seeds)
        assert all(v in (base, tenx) for r in steps for v in r), (key, steps)
        tenx_seen += [dict(arm=key, n=n, seed=sd) for n, r in zip(x, steps)
                      for sd, v in zip(seeds, r) if v == tenx]
        any_tenx |= any(v > base for r in steps for v in r)
        a = ARMS[key]
        draw(ser, a["color"], a["marker"], steps=steps)
        # recorded, never a build condition: rung/seed cells where the baseline is at or above
        for n, sv, rv in zip(x, store, ser):
            for sd, a_, b_ in zip(seeds, sv, rv):
                if b_ >= a_:
                    below.append(dict(set=s["id"], arm=key, n=n, seed=sd, store=a_, rival=b_))
    ax.set_xticks(x)
    ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)
    ax.set_xlabel("n, factors jointly held out")
    ax.set_ylabel("distinct valid novel tuples\nper held-out combination")
    ax.set_ylim(0, 75)
    set_notes.append(f"{s['id']}: {s['title']}, {fmt_int(s['pool_rows'])} rows, "
                     f"{s['steps_label']} ({s['pool_note']}); source {s['source']}")
    handles = [Line2D([0], [0], color=CLU_C, marker=ARMS["clu"]["marker"], lw=1.8, label="n-CLU")]
    handles += [Line2D([0], [0], color=ARMS[k]["color"], marker=ARMS[k]["marker"], lw=1.2,
                       label=ARMS[k]["name"]) for k in FAMS if k in s["arms"]]
    handles += [Line2D([0], [0], color=FLOOR, marker=ARMS["table"]["marker"], ls="--", lw=1.0,
                       label="Lookup table (floor)"),
                Line2D([0], [0], color=INK2, marker="o", lw=0, markerfacecolor="white", ms=3,
                       label="each seed; line = mean")]
    if any_tenx:
        handles.append(Line2D([0], [0], color=INK2, marker="o", lw=0, ms=3,
                              label=r"seed at $10{\times}$ steps"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
              fontsize=6.5, borderaxespad=0.0)
    # every legend entry is a drawn series (item 10: no legend entry without its series)
    assert [h.get_label() for h in handles[1:1 + sum(k in s["arms"] for k in FAMS)]] == \
        [ARMS[k]["name"] for k in FAMS if k in s["arms"]]
    assert [ARMS[k]["name"] for k in FAMS] == ["MLP", "LSTM", "Transformer", "Continuous MLP"]
    rendered = M.check_vocab_pw1(fig)
    pending = [f"{s['id']} ({fmt_int(s['pool_rows'])} rows): {s['awaiting']}"
               for s in man["sets"] if s["status"] == "not_landed"]
    save(fig, out, "fig_ladder", sources,
         "PW-1 item 10: one panel, no title, legend at the right. "
         "3 seeds (0, 1, 2) on every rung; line = mean of the three per-seed values, dots = seeds. "
         "The one landed rival column set (data/ladder_rival_sets.json; the 32,256-row and the "
         "32,768-row data-matched MLP sets are retired by item 10.1), beside the store "
         "(KSTAR-LADDER.json by_rung.*.kstar_K32.store; asserted equal, "
         "seed by seed, to SHAPES-MATCHED.json ladder.*.arms.store_K32_n131072) and table floor. "
         "Pool size and optimiser steps are asserted from the artefact and printed in the "
         "caption. Set drawn: " + "; ".join(set_notes) + ". w20 set: the P1-carried fit per "
         "family and seed (P3_margin.store_K32_n131072.per_family.*.*.carried_rival); a filled "
         "dot is a seed carried at its 204,800-step (10x) fit, an open dot the 20,480-step fit. "
         "For n >= 3 the rungs probe the identical 120 corners.",
         extra={"rival_sets_drawn": [s["id"] for s in sets], "rival_sets_not_landed": pending,
                "rival_sets_retired": [s_["id"] for s_ in man["sets"] if s_["status"] == "retired"],
                "rival_at_or_above_store": below, "step_sources": step_sources,
                "tenx_points": tenx_seen, "rendered_text": rendered})
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
        ax.set_xlabel("baseline optimiser steps (log)")
    a1.text(1000, mean(store[131072]) + 1.2,
            "n-CLU, K* = 32: fit on 16,384 / 32,768 / 131,072 rows\n"
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
    a1.set_title("(a) MLP baseline: every pool × step cell", loc="left", fontsize=7)
    a1.set_ylabel("distinct valid novel tuples\nper held-out combination\n(rows 3–4, test, 109 combinations)",
                  fontsize=6.6)
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
        a2.text(last[0] * 1.15, mean(last[2]), ARMS[f]["name"], fontsize=5.6, va="center", color=INK)
    a2.set_title("(b) Four families: 1,280 steps on 16,384 rows,\nthen 131,072 rows at "
                 "20,480 / 204,800 / 614,400", loc="left", fontsize=6.6)
    a2.set_xlim(900, 2600000)  # room for the full family names (PW-1 item 2: no "Transf.")
    M.check_vocab_pw1(fig)
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
MATCHED6 = ("rival: mlp", "rival: lstm", "rival: transformer",
            "rival@10x-steps: mlp", "rival@10x-steps: lstm", "rival@10x-steps: transformer")
TREL = ("diagnostic+rel@8k: transformer +rel", "diagnostic+rel@80k: transformer +rel")


def _budget(k):
    """PW-1 item 1.8: the baseline's step limit, in the paper's words."""
    return r"$10\times$ training steps" if ("10x-steps" in k or "@80k" in k) \
        else "declared training steps"


def _fam(k):
    return k.split(":", 1)[1].strip().split(" ")[0]


def fig_maze_conditions(out):
    """PW-1 item 1 (Head-ruled 2026-09-25): (a) four input conditions, every seed drawn, seed count
    printed; (b) / (c) six seeds without / with relational coding. No claim labels, no ceiling,
    no hard/easy marking, no text over the data."""
    import cmazet_panels  # the rendered-string vocabulary and overlap checks
    man = data("maze_conditions.json")
    rows = [r for r in man["rows"] if r["status"] == "landed"]
    six = maze_six.read()
    sources = list(six["sources"])
    S = six["seeds"]
    comp = []
    for r in rows:
        if r["schema"] == "w19":
            d = load(r["source"])
            sources.append(r["source"])
            assert d["leg"].upper() == r["leg"], (r["id"], d["leg"])
            ps = W.maze_per_seed(d, r["schema"], man["primary_row"])
            rstar = max((k for k in ps if k.startswith("rival")), key=lambda k: mean(ps[k]))
            assert rstar == W.head_label(ps)["R_star"]  # the same comparator the draft's table uses
            clu, base, seeds = ps["CLU"], ps[rstar], [0, 1, 2]
            assert len(clu) == len(base) == 3
        else:
            P = six["config"][r["config"]]
            clu = [P[s]["CLU"] for s in S]
            keys = MATCHED6 if r["comparator"] == "best_matched" else TREL
            rstar = max(keys, key=lambda k: mean([P[s]["arms"][k] for s in S]))
            base, seeds = [P[s]["arms"][rstar] for s in S], S
            if r.get("check_source"):
                c3 = W.maze_per_seed(load(r["check_source"]), "w19", man["primary_row"])
                sources.append(r["check_source"])
                assert c3["CLU"] == clu[:3], (r["id"], c3["CLU"], clu[:3])
                for k in MATCHED6:
                    assert c3[k] == [P[s]["arms"][k] for s in (0, 1, 2)], (r["id"], k)
        comp.append(dict(row=r, clu=clu, base=base, seeds=seeds, rstar=rstar))
    rel_arm = next(c["rstar"] for c in comp if c["row"].get("comparator") == "transformer_rel")
    a1_arm = next(c["rstar"] for c in comp if c["row"].get("comparator") == "best_matched")
    assert _fam(a1_arm) == "transformer" and _fam(rel_arm) == "transformer", (a1_arm, rel_arm)

    TC = ARMS["transformer"]
    fams_used = {_fam(c["rstar"]) for c in comp}
    n = len(comp)
    fig = plt.figure(figsize=(TEXT_W, 0.40 * n + 3.05), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[0.40 * n + 0.80, 2.25])
    gs = top.add_gridspec(1, 2, width_ratios=[1.45, 2.6])
    tl, ax = top.add_subplot(gs[0]), top.add_subplot(gs[1])
    tl.set_axis_off()
    for a in (tl, ax):
        a.set_ylim(n - 0.5, -0.5)
    texts = []
    for i, c in enumerate(comp):
        r = c["row"]
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f3ef", lw=0, zorder=0)
        fam = _fam(c["rstar"])
        rc = ARMS[fam]
        is_rel = r.get("comparator") == "transformer_rel"
        dy = 0.15
        ax.scatter(c["clu"], [i - dy] * len(c["clu"]), s=16, marker="o", color=CLU_C, zorder=4,
                   edgecolor="white", lw=0.4)
        ax.plot([mean(c["clu"])] * 2, [i - dy - 0.13, i - dy + 0.13], color=CLU_C, lw=1.4,
                zorder=3)
        ax.scatter(c["base"], [i + dy] * len(c["base"]), s=16, marker=rc["marker"], zorder=4,
                   facecolor="white" if is_rel else rc["color"],
                   edgecolor=rc["color"], lw=0.9 if is_rel else 0.4)
        ax.plot([mean(c["base"])] * 2, [i + dy - 0.13, i + dy + 0.13], color=rc["color"],
                lw=1.4, zorder=3)
        texts.append(tl.text(0.0, i - 0.13, r["label"], fontsize=6.4, color=INK, va="center",
                             ha="left", fontweight="bold"))
        leg = {"TEST": "test", "VAL": "validation"}[r["leg"]]
        seeds = (f"seeds 0–{c['seeds'][-1]}" if c["seeds"] == list(range(len(c["seeds"])))
                 else "seeds " + ", ".join(map(str, c["seeds"])))
        name = ARMS[fam]["name"] + ("-REL" if is_rel else "")
        sub = (f"{leg} · {len(c['seeds'])} seeds ({seeds})\n"
               f"{'baseline' if is_rel else 'best baseline'}: {name}, {_budget(c['rstar'])}")
        texts.append(tl.text(0.0, i + 0.20, sub, fontsize=5.2, color=INK2, va="center",
                             ha="left", linespacing=1.15))
    ax.set_xlim(-0.02, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("unseen n-way validity, the engine's judge (fraction)", fontsize=6.4)
    ax.set_title("(a) n-CLU vs Best Matched Baseline for Different Input Choices", loc="left",
                 fontsize=6.8)
    ns = sorted({len(c["seeds"]) for c in comp})
    handles = [Line2D([0], [0], color=CLU_C, marker="o", lw=0, label="n-CLU, each seed")]
    handles += [Line2D([0], [0], color=ARMS[f]["color"], marker=ARMS[f]["marker"], lw=0,
                       label=f"{ARMS[f]['name']}, each seed")
                for f in ("mlp", "lstm", "transformer", "cont_mlp") if f in fams_used]
    handles += [Line2D([0], [0], color=TC["color"], marker=TC["marker"], lw=0,
                       markerfacecolor="white", label="Transformer-REL, each seed")]
    handles += [Line2D([0], [0], color=INK2, lw=1.4,
                       label="mean of " + " or ".join(map(str, ns)) + " seeds")]
    top.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=5.6)

    # (b) / (c): the six seeds of the two words-plus-observation rows, x = seed number only
    sb1, sb2 = bot.subfigures(1, 2)
    b1, b2 = sb1.subplots(), sb2.subplots()
    per_seed = {}
    off = {"clu": -0.18, "base": 0.0, "ctrl": 0.18}
    for bx, sb, cfg, arm, ttl, lab_c, lab_b in (
            (b1, sb1, "A1", a1_arm, "(b) Test Results w/o Relational Coding Across Seeds", "n-CLU",
             "Transformer (best baseline)"),
            (b2, sb2, "REL", rel_arm, "(c) Test Results with Relational Coding Across Seeds",
             "n-CLU-REL", "Transformer-REL (best baseline)")):
        P = six["config"][cfg]
        rel = cfg == "REL"
        per_seed[cfg] = {}
        for s in S:
            p = P[s]
            per_seed[cfg][str(s)] = dict(n_CLU=p["CLU"], baseline=p["arms"][arm], baseline_arm=arm,
                                         highest_control=p["max_control"],
                                         highest_control_arms=p["max_control_arms"],
                                         source=p["source"])
            bx.scatter([s + off["ctrl"]], [p["max_control"]], marker="x", s=18, color=FLOOR,
                       lw=1.0, zorder=4)
            bx.scatter([s + off["base"]], [p["arms"][arm]], marker=TC["marker"], s=22, zorder=5,
                       facecolor="white" if rel else TC["color"],
                       edgecolor=TC["color"] if rel else "white", lw=1.0 if rel else 0.4)
            bx.scatter([s + off["clu"]], [p["CLU"]], marker="o", s=24, color=CLU_C,
                       edgecolor="white", lw=0.4, zorder=6)
        bx.set_xticks(S)
        bx.set_xticklabels([str(s) for s in S], fontsize=5.8)
        bx.set_xlim(-0.55, 5.55)
        bx.set_ylim(0, 1.02)
        bx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bx.grid(axis="x", visible=False)
        bx.set_title(ttl, loc="left", fontsize=6.6)
        bx.set_xlabel("seed", fontsize=6.2)
        h = [Line2D([0], [0], marker="o", lw=0, color=CLU_C, ms=4.5, label=lab_c),
             Line2D([0], [0], marker=TC["marker"], lw=0, color=TC["color"], ms=4.5,
                    markerfacecolor="white" if rel else TC["color"], label=lab_b),
             Line2D([0], [0], marker="x", lw=0, color=FLOOR, ms=4.5, label="Highest control")]
        sb.legend(handles=h, loc="outside lower center", ncol=2, fontsize=5.4)
    b1.set_ylabel("unseen n-way validity (fraction)", fontsize=6.2)
    cmazet_panels.check_vocab(fig)
    cmazet_panels.check_no_overlap(fig, texts)
    rows_rec = {c["row"]["id"]: dict(label=c["row"]["label"], leg=c["row"]["leg"],
                                     seeds=c["seeds"], n_CLU_per_seed=c["clu"],
                                     comparator_arm=c["rstar"], comparator_per_seed=c["base"],
                                     n_CLU_mean=mean(c["clu"]), comparator_mean=mean(c["base"]),
                                     source=c["row"].get("source", "maze_six.py (six-seed read)"))
                for c in comp}
    save(fig, out, "fig_maze_conditions", sources,
         "PW-1 item 1. (a) four input conditions, primary row unseen n-way (the engine's judge); "
         "every seed a row has is drawn (3 for the words-full and observation rows, from their "
         "BARS artefacts; 6 for the two words-plus-observation rows, from each seed's run JSON "
         "via maze_six.py, asserted float-equal to REL-NEWSEEDS.json). Comparator = the matched "
         "baseline with the highest mean over the row's seeds (every kind, both step limits); for "
         "the n-CLU-REL row, Transformer-REL at the step limit with the higher six-seed mean. "
         "(b), (c) per seed: n-CLU / n-CLU-REL, the (a) comparator of that row, and the highest "
         "control (max over the four controls on that seed). No claim label, no ceiling, no "
         "hard/easy class, no text over the data.",
         extra={"rows": rows_rec, "six_seed": per_seed})
    return rows_rec


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
    """PW-1 item 12 (Head, 2026-09-25): former panels (a)-(d) removed; former (e), (f) relettered
    (a), (b). The removed panels' numbers are still read and asserted here and written to
    PROVENANCE (deletion_facts) for the deletion table; they are no longer drawn."""
    d = load(W20)["deletion_matched"]
    PS = d["per_scope"]
    seeds = ["0", "1", "2"]
    # the removed panels' measurements: asserted and recorded, not drawn
    facts = {}
    for sc, lab, _ in DSCOPES:
        st = PS[sc]["store"]
        assert all(st[s]["a"]["RESULT"] and st[s]["b"]["RESULT"] for s in seeds)
        b1 = [(st[s]["b"]["b1_statistic"]["max_abs_count_dev"],
               st[s]["b"]["b1_statistic"]["max_abs_first_moment_dev"]) for s in seeds]
        assert all(x == (0, 0.0) for x in b1)
        c = [st[s]["c_DISPLAY"] for s in seeds]
        assert all(x["DISPLAY_NEVER_A_RESULT"] for x in c)
        nc = {st[s]["d"]["n_survivor_corners"] for s in seeds}
        assert len(nc) == 1
        facts[sc] = {s: dict(
            max_abs_mode_dev=st[s]["a"]["max_abs_mode_dev"],
            book_statistic_count_dev=b1[k][0], book_statistic_first_moment_dev=b1[k][1],
            book_rendered_features_rel_dev=st[s]["b"]["b2_book"][
                "rendered_features_rel_dev_on_survivor_requests"],
            refusal={x: c[k].get(x) for x in (
                "refusal_rate_on_deleted_corner_requests", "n_deleted_corner_requests",
                "false_refusal_rate_on_survivor_requests", "n_survivor_requests", "DECLARED_NA")
                if x in c[k]},
            before_mean=st[s]["d"]["before_mean"], after_mean=st[s]["d"]["after_mean"],
            n_survivor_combinations=st[s]["d"]["n_survivor_corners"]) for k, s in enumerate(seeds)}
    fig, (e, f) = plt.subplots(1, 2, figsize=(TEXT_W, 2.45), layout="constrained",
                               gridspec_kw={"width_ratios": [0.8, 1.4]})
    ticks = ["value", "red sphere", "1 % items\n(control)"]  # item 12.5: "red sphere"
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
    e.set_xticklabels(ticks, fontsize=5.8)
    e.set_ylabel("after − before, novel tuples per held-out\ncombination (95 % bootstrap CI)",
                 fontsize=5.8)
    e.set_title("(a) Composition on the surviving\ncombinations, after − before\n(3 seeds)",
                loc="left", fontsize=6.4)
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
    f.set_xticklabels([t.replace("\n", " ") for t in ticks], fontsize=5.8)
    f.set_ylabel("n-CLU after − baseline retrained\nwithout, same rows (95 % CI)", fontsize=5.8)
    f.set_title("(b) n-CLU after deletion − baseline\nretrained without the rows", loc="left",
                fontsize=6.4)
    f.grid(axis="x", visible=False)
    # item 12.6: the legend moves to the right of (b), vertical, clear of every interval
    f.legend(handles=[Line2D([0], [0], marker=ARMS[k]["marker"], color=ARMS[k]["color"], lw=0,
                             ms=4, label=ARMS[k]["name"]) for k in FAMS],
             loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=5.8, ncol=1,
             borderaxespad=0.0)
    rendered = M.check_vocab_pw1(fig)
    rows_read = {sc: {fam: sorted({PS[sc]["rivals"][fam][s]["rival_cost_line"]["data_rows_read"]
                                   for s in seeds}) for fam in FAMS} for sc, _, _ in DSCOPES}
    save(fig, out, "fig_deletion", [W20],
         "PW-1 item 12: two panels, former (e) and (f) relettered (a) and (b); former (a)-(d) "
         "removed and their values moved to the deletion table (deletion_facts below, asserted "
         "and recorded, not drawn). "
         "Rows 3-4 TEST, K* = 32, fit pool 131,072 rows, seeds 0, 1, 2; all from SHAPES-MATCHED.json "
         "deletion_matched.per_scope.<scope> (definition: forget c = downdate the store, downdate "
         "the address book's statistic and refit the book from it, punch the hole). "
         "(a) store.<seed>.d.paired_after_minus_before.mean_paired_delta with its "
         "bootstrap_ci95 (downdate minus refit-without = 0.000, all corners tied, asserted); (b) "
         "rivals.<fam>.<seed>.ours_minus_rival_retrain_without with ci95, rivals retrained on the "
         "same surviving rows of the same pool at their declared schedule.",
         extra={"rival_rows_read": rows_read, "deletion_facts": facts, "rendered_text": rendered})

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
