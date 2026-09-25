"""w21 figures-redraw -- the paper's redrawn figures, every value read from an evidence artefact.

Figures built here (each overwrites any earlier file of the same stem in --out):
  fig_crossover         K-curve, validation, mean +- seed sd at EVERY K, n = 3 (w15 crossover-seeds)
  fig_ladder            K* ladder: store beside each rival column set (data/ladder_rival_sets.json)
  fig_deletion          deletion at K*, 3 seeds, store and every rival (w15 single-design-seeds)
  fig_maze_conditions   CMAZE-T condition table, per-seed markers, Head's-rule labels
                        (data/maze_conditions.json)
  fig_cmazet_*          per-arm CMAZE-T panels (Part C), only if --panels points at a verified
                        per-item extraction (see cmazet_extract.py); data/cmazet_panels.json

Style, arm colours and names are the w17 module's (make_figures.py, imported, unchanged). No banner,
no pipeline language in any rendered string. Every source file's SHA-256 is written to
PROVENANCE.json beside the figure it feeds. Determinism: Agg, bundled DejaVu Sans, no timestamps,
sorted inputs, no randomness.

usage: python make_figures_w21.py --out <dir> [--panels <cmazet per-item json>]
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as M  # noqa: E402  (sets rcParams, SOURCE_DATE_EPOCH, Agg)

plt = M.plt
Line2D = M.Line2D
Patch = M.Patch
FuncFormatter = M.FuncFormatter
ARMS = M.ARMS
INK, INK2, GRID, FLOOR, CONTROL = M.INK, M.INK2, M.GRID, M.FLOOR, M.CONTROL
TEXT_W = M.TEXT_W
DATA = os.path.join(HERE, "data")
PROV = {}

# words a rendered string must never contain (Part D: no pipeline language in the image)
FORBIDDEN = ("PLACEHOLDER", "placeholder", "wave", "w1", "w2", "spoke", "regenerat", "pending",
             "Hub", "Head", "TODO")


# ---------------------------------------------------------------- helpers
def ev(rel):
    return os.path.join(M.EVIDENCE, rel)


def load(rel):
    with open(ev(rel), "r", encoding="utf-8") as f:
        return json.load(f)


def sha(rel):
    h = hashlib.sha256()
    with open(ev(rel), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def data(name):
    with open(os.path.join(DATA, name), "r", encoding="utf-8") as f:
        return json.load(f)


def jpath(d, path):
    for k in path.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def mean(v):
    return sum(v) / len(v)


def check_text(fig):
    """Assert no rendered string carries banner or pipeline language."""
    for t in fig.findobj(M.matplotlib.text.Text):
        s = t.get_text()
        for w in FORBIDDEN:
            assert w not in s, f"forbidden word {w!r} in rendered text {s!r}"


def save(fig, out, stem, sources, notes, extra=None):
    check_text(fig)
    fig.savefig(os.path.join(out, stem + ".pdf"),
                metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                          "Creator": None})
    fig.savefig(os.path.join(out, stem + ".png"), metadata={"Software": None})
    plt.close(fig)
    srcs = sorted(set(sources))
    PROV[stem] = dict(sources=srcs, source_sha256={s: sha(s) for s in srcs}, placeholder=False,
                      awaiting=None, banner=False, notes=notes, built_by="make_figures_w21.py")
    if extra:
        PROV[stem].update(extra)


def seed_dots(ax, x, vals, color, marker="o", size=11, filled=True, z=4, jitter=0.0):
    xs = [x + (i - 1) * jitter for i in range(len(vals))]
    ax.scatter(xs, vals, s=size, marker=marker, zorder=z, linewidths=0.8,
               facecolor=color if filled else "white", edgecolor=color if filled else color)


# ---------------------------------------------------------------- 1. crossover (Part A)
def fig_crossover(out):
    src = "w15-crossover-seeds/CROSSOVER-3SEED.json"
    d = load(src)
    Ks = list(d["sweep"])
    assert Ks == sorted(int(k) for k in d["per_K"]) and d["seeds"] == [0, 1, 2]
    rows = [d["per_K"][str(k)] for k in Ks]
    for r in rows:  # n = 3 at every K, or the figure refuses to build
        assert r["n_seeds"] == 3 and r["val_composition"]["n"] == 3
    per_seed = [[r["per_seed"][str(s)]["val_composition"] for s in (0, 1, 2)] for r in rows]
    mu = [r["val_composition"]["mean"] for r in rows]
    sd = [r["val_composition"]["sd"] for r in rows]
    for ps, m, s in zip(per_seed, mu, sd):  # the artefact's own mean/sd are what is drawn
        assert abs(mean(ps) - m) < 1e-12 and abs(statistics.stdev(ps) - s) < 1e-12

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.35), constrained_layout=True)
    clu = ARMS["clu"]
    a1.fill_between(Ks, [m - s for m, s in zip(mu, sd)], [m + s for m, s in zip(mu, sd)],
                    color=clu["color"], alpha=0.16, lw=0, zorder=1)
    a1.plot(Ks, mu, color=clu["color"], marker="o", ms=3.5, lw=1.6, zorder=3)
    for k, ps in zip(Ks, per_seed):
        a1.scatter([k] * 3, ps, s=9, facecolor="white", edgecolor=clu["color"], lw=0.7, zorder=4)
    # the four charter-C controls, mean over seeds and corners, at every K
    cmax = [max(r["control_gate"][c]["mean_over_seeds_and_corners"] for c in r["control_gate"])
            for r in rows]
    a1.scatter(Ks, cmax, marker="x", s=14, color=CONTROL, lw=1.0, zorder=3)
    a1.set_xticks(Ks)
    a1.set_xlabel("K, wells")
    a1.set_ylabel("validation composition")  # PW-1 item 11.1
    a1.set_title("(a) Composition vs K, 3 seeds", loc="left")
    a1.set_ylim(-0.03, 1.0)
    a1.legend(handles=[
        Line2D([0], [0], color=clu["color"], marker="o", ms=3.5, label="n-CLU, mean of 3 seeds"),  # PW-1 item 11.4
        Patch(facecolor=clu["color"], alpha=0.16, label="± 1 seed sd"),
        Line2D([0], [0], color=clu["color"], marker="o", ms=3.5, lw=0, markerfacecolor="white",
               label="each seed"),
        Line2D([0], [0], color=CONTROL, marker="x", lw=0, label="highest of 4 controls"),
    ], loc="lower right", ncol=1)

    bottom = [0.0] * len(Ks)
    for key, name, col in M.PARTS:
        vals = [r["parts_across_seeds"][key]["mean"] for r in rows]
        a2.bar(Ks, vals, width=2.0, bottom=bottom, color=col, edgecolor="white", lw=0.5,
               label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    tot = [r["honest_total_bytes"]["mean"] for r in rows]
    tsd = [r["honest_total_bytes"]["sd"] for r in rows]
    assert all(abs(b - t) < 0.5 for b, t in zip(bottom, tot)), "four parts != honest total"
    a2.errorbar(Ks, tot, yerr=tsd, fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5, zorder=4)
    ref = d["per_K"]["32"]["per_seed"]["0"]["ledger_four_part"]["reference_pin_bytes"]
    a2.axhline(ref, color=INK2, lw=0.8, ls="--", label=f"{ref:,} B reference")
    a2.set_xticks(Ks)
    a2.set_xlabel("K, wells")
    a2.set_ylabel(r"Bytes ($\times 10^6$)")  # PW-1 item 11.2
    a2.set_title("(b) Memory footprint vs K", loc="left")  # PW-1 item 11.3
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}"))
    a2.legend(loc="upper right", ncol=1)
    M.check_vocab_pw1(fig)
    save(fig, out, "fig_crossover", [src],
         "n = 3 seeds (0, 1, 2) at EVERY K = 5..32; no single-seed line. (a) line = the "
         "artefact's val_composition.mean, band = +-1 val_composition.sd (sample sd, asserted "
         "equal to the per-seed values), open dots = per_seed.*.val_composition; grey x = the "
         "highest of the four charter-C controls' mean_over_seeds_and_corners (0 at every K "
         "except K = 20, landscape_shuffled_across_wells 0.000193). (b) stacked "
         "parts_across_seeds.*.mean, error bar = honest_total_bytes.sd; 524,288 B is a reference "
         "line only. The artefact is SMOKE_NOT_A_CLAIM (validation).")


# ---------------------------------------------------------------- 2. ladder (Part A)
def read_ladder_csv(rel):
    outd = {}
    with open(ev(rel), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["rung"] == "shipped":
                continue
            outd[(r["arm"], int(r["rung"]), int(r["seed"]))] = float(
                r["mean_distinct_valid_novel_tuples"])
    return outd


def read_series(spec, arm, rungs, seeds):
    """per-rung list of per-seed values for one arm of one set (reader named in the manifest)."""
    if spec["reader"] == "ladder_csv":
        t = read_ladder_csv(spec["source"])
        return [[t[(arm, n, s)] for s in seeds] for n in rungs]
    if spec["reader"] == "json_path":
        d = load(spec["source"])
        return [[jpath(d, spec["path_template"].format(n=n, seed=s, arm=arm)) for s in seeds]
                for n in rungs]
    raise ValueError("unknown reader " + spec["reader"])


LADDER_ORDER = ["mlp", "lstm", "transformer", "cont_mlp"]


def fig_ladder(out):
    man = data("ladder_rival_sets.json")
    rungs, seeds = man["rungs"], man["seeds"]
    st = man["store"]
    store = read_series(st, None, rungs, seeds)
    dstore = load(st["source"])
    for n, ps in zip(rungs, store):  # the artefact's own mean
        assert abs(mean(ps) - jpath(dstore, st["mean_template"].format(n=n))) < 1e-9
    fl = man["floor"]
    floor = read_series(fl, fl["arm"], rungs, seeds)
    sets = [s for s in man["sets"] if s["status"] == "landed"]
    sources = [st["source"], fl["source"]]
    for s in sets:  # pool size printed on the panel is asserted from its artefact
        assert jpath(load(s["pool_source"]), s["pool_path"]) == s["pool_rows"], s["id"]
        sources += [s["source"], s["pool_source"]]

    ncol = min(len(sets), 3)
    nrow = (len(sets) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(TEXT_W, 2.55 * nrow), squeeze=False,
                            sharey=True, constrained_layout=True)
    clu = ARMS["clu"]
    x = rungs
    set_notes = []
    for i, s in enumerate(sets):
        ax = axs[i // ncol][i % ncol]

        def draw(series, color, marker, ls="-", lw=1.2, label=None, z=3):
            mu = [mean(v) for v in series]
            ax.plot(x, mu, color=color, marker=marker, ms=3.4, lw=lw, ls=ls, zorder=z)
            for n, v in zip(x, series):
                ax.scatter([n] * len(v), v, s=6, facecolor="white", edgecolor=color, lw=0.6,
                           zorder=z + 1)
            if label:
                end_labels.append([mu[-1], label])
            return mu

        end_labels = []
        draw(store, clu["color"], clu["marker"], lw=1.8, label="CLU", z=5)
        draw(floor, FLOOR, ARMS["table"]["marker"], ls="--", lw=1.0, label="table")
        arm_means = {}
        for key in LADDER_ORDER:
            if key not in s["arms"]:
                continue
            ser = read_series(s, s["arms"][key], rungs, seeds)
            if s.get("constant_for_n_ge_3"):
                for v in ser[1:]:
                    assert v == ser[1], f"{s['id']}:{key} not constant on n >= 3"
            a = ARMS[key]
            short = {"mlp": "MLP", "lstm": "LSTM", "transformer": "Transf.",
                     "cont_mlp": "Cont. MLP"}[key]
            arm_means[key] = draw(ser, a["color"], a["marker"], label=short)
            # the store is above this arm on every rung and seed (the draft's reading)
            for sv, rv in zip(store, ser):
                assert all(a_ > b_ for a_, b_ in zip(sv, rv))
        # direct labels at the line ends, pushed apart so none overlaps (min gap 4.2 units)
        end_labels.sort()
        for j in range(1, len(end_labels)):
            end_labels[j][0] = max(end_labels[j][0], end_labels[j - 1][0] + 4.2)
        for yv, lab in end_labels:
            ax.text(x[-1] + 0.18, yv, lab, va="center", ha="left", fontsize=5.8, color=INK)
        ax.set_title(f"({'abcdef'[i]}) {s['title']}", loc="left", fontsize=7)
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 0.3, x[-1] + 1.15)
        ax.set_xlabel("n, factors jointly held out\n(n ≥ 3: the same 120 corners)")
        set_notes.append(f"{s['id']}: {s['title']} ({s['pool_note']}); source {s['source']}")
    for j in range(len(sets), nrow * ncol):
        axs[j // ncol][j % ncol].set_visible(False)
    axs[0][0].set_ylabel("distinct valid novel tuples / corner")
    axs[0][0].set_ylim(0, 75)
    handles = [Line2D([0], [0], color=clu["color"], marker="o", lw=1.8, label="CLU, K* = 32")]
    handles += [Line2D([0], [0], color=ARMS[k]["color"], marker=ARMS[k]["marker"], lw=1.2,
                       label=ARMS[k]["name"]) for k in LADDER_ORDER
                if any(k in s["arms"] for s in sets)]
    handles += [Line2D([0], [0], color=FLOOR, marker="x", ls="--", lw=1.0,
                       label="Lookup table (floor)"),
                Line2D([0], [0], color=INK2, marker="o", lw=0, markerfacecolor="white", ms=3,
                       label="each seed; line = mean")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6)
    pending = [f"{s['id']} ({s['pool_rows']:,} rows): {s['awaiting']}"
               for s in man["sets"] if s["status"] != "landed"]
    save(fig, out, "fig_ladder", sources,
         "3 seeds (0, 1, 2) on every rung; line = mean of the three per-seed values, open dots = "
         "seeds. One panel per landed rival column set (data/ladder_rival_sets.json), each "
         "beside the same store (KSTAR-LADDER.json by_rung.*.kstar_K32.store) and table floor "
         "(LADDER.csv rival_table). Sets drawn: " + "; ".join(set_notes) + ". Pool sizes "
         "asserted from each set's own artefact. For n >= 3 the rungs probe the identical 120 "
         "corners: a fit-pool coverage axis, not a depth axis. The Learned-1x stress opponent "
         "is not drawn (it is not the paper's matching rule). Asserted: the store exceeds every "
         "drawn rival on every rung and seed.",
         extra={"rival_sets_drawn": [s["id"] for s in sets],
                "rival_sets_not_landed": pending})


# ---------------------------------------------------------------- 3. deletion (Part A)
DEL_SCOPES = [("value", "red value"), ("conjunction_live", "red sphere"), ("item", "random 1 % of items")]
DEL_RIVALS = ["mlp", "lstm", "transformer", "cont_mlp"]


def fig_deletion(out):
    src = "w15-single-design-seeds/T7-3SEEDS.json"
    d = load(src)
    assert d["n_seeds"] == 3 and d["K_star"] == 32 and d["condition"] == "test"
    seeds = ["seed0", "seed1", "seed2"]
    DR = d["deletion_rows"]["scopes"]
    RC = d["rival_deletion_column"]["scopes"]

    fig = plt.figure(figsize=(TEXT_W, 4.3), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1.05, 1])
    atop = top.subplots(1, 3, sharey=True)
    a2, a3 = bot.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 1]})
    clu = ARMS["clu"]
    arms = ["clu"] + DEL_RIVALS
    for ax, (scope, title) in zip(atop, DEL_SCOPES):
        dr, rc = DR[scope]["per_seed"], RC[scope]["per_seed"]
        ncorn = {dr[s]["n_survivor_corners"] for s in seeds}
        assert len(ncorn) == 1
        ncorn = ncorn.pop()
        for i, key in enumerate(arms):
            if key == "clu":
                before = [dr[s]["survivors_before"] for s in seeds]
                after = [dr[s]["survivors_after"] for s in seeds]
                for s in seeds:  # the store's after equals the value the rival table prints
                    assert rc[s]["ours_after_deletion_on_the_same_survivors"] == dr[s]["survivors_after"]
            else:
                fam = [rc[s]["families"][key] for s in seeds]
                for f in fam:
                    assert f["n_survivor_corners"] == ncorn
                before = [f["as_trained_novel_per_corner_on_the_scopes_survivors"] for f in fam]
                after = [f["retrain_without_novel_per_corner"] for f in fam]
            col = ARMS[key]["color"]
            for off, vals, filled in ((-0.19, before, False), (0.19, after, True)):
                ax.bar(i + off, mean(vals), 0.34, color=col if filled else "white",
                       edgecolor=col, lw=0.9, zorder=2)
                ax.scatter([i + off] * 3, vals, s=5, color=INK, zorder=4, lw=0)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(["CLU", "MLP", "LSTM", "Transf.", "Cont.\nMLP"], fontsize=6)
        ax.set_title(f"{title}\n({ncorn} survivor corners)", loc="left", fontsize=6.8)
    atop[0].set_ylabel("distinct valid novel tuples\nper survivor corner")
    atop[0].set_ylim(0, 62)
    top.suptitle("(a) Composition on the corners that survive deletion, before (open) and after "
                 "(filled)", x=0.01, ha="left", fontsize=7.5)

    # (b) adaptation seconds, (c) data rows read -- value scope, 3 seeds
    dv, rv = DR["value"]["per_seed"], RC["value"]["per_seed"]
    lines = []
    ours = [dv[s]["OUR_ADAPTATION_COST"] for s in seeds]
    refit = [dv[s]["REFIT_WITHOUT_COST"] for s in seeds]
    lines.append(("CLU downdate", clu["color"], True, [o["seconds"] for o in ours],
                  {o["data_rows_read"] for o in ours}))
    lines.append(("CLU refit without (reference)", clu["color"], False,
                  [r["seconds"] for r in refit], {r["data_rows_read"] for r in refit}))
    rpool = set()
    for key in DEL_RIVALS:
        fam = [rv[s]["families"][key] for s in seeds]
        rpool |= {f["as_trained_cost"]["data_rows_read"] for f in fam}
        lines.append((f"{ARMS[key]['name']} retrain without", ARMS[key]["color"], True,
                      [f["retrain_without_cost"]["seconds"] for f in fam],
                      {f["retrain_without_cost"]["data_rows_read"] for f in fam}))
    assert all(len(r) == 1 for *_, r in lines), "rows read differ across seeds"
    assert len(rpool) == 1
    rpool = rpool.pop()
    y = list(range(len(lines)))[::-1]
    for yi, (lab, col, filled, secs, _) in zip(y, lines):
        a2.barh(yi, mean(secs), 0.66, color=col if filled else "white", edgecolor=col, lw=0.9)
        a2.scatter(secs, [yi] * 3, s=6, color=INK, zorder=4, lw=0)
    a2.set_xscale("log")
    a2.set_yticks(y)
    a2.set_yticklabels([lab for lab, *_ in lines], fontsize=6)
    a2.set_xlabel("seconds (log); bar = mean, dots = seeds")
    a2.set_title("(b) Adaptation time, red value", loc="left")
    for yi, (lab, col, filled, _, rr) in zip(y, lines):
        v = rr.pop()
        a3.barh(yi, v, 0.66, color=col if filled else "white", edgecolor=col, lw=0.9)
        a3.text(v, yi, f" {v:,}", va="center", fontsize=5.8, color=INK)
    a3.set_yticks(y)
    a3.set_yticklabels([])
    a3.set_xlabel("data rows read (every seed)")
    a3.set_title("(c) Data read, red value", loc="left")
    a3.set_xlim(0, 131072 * 1.45)
    a3.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    save(fig, out, "fig_deletion", [src],
         "TEST, K* = 32, 3 seeds (0, 1, 2), all from T7-3SEEDS.json. (a) survivor corners of each "
         "scope: store survivors_before (open) / survivors_after (filled, the downdated store); "
         "each rival as_trained_novel_per_corner_on_the_scopes_survivors (open) / "
         "retrain_without_novel_per_corner (filled); bar = mean of 3 seeds, dots = seeds. "
         "The vacuous red-AND-cube scope (0 rows) is not drawn. (b)/(c) value scope: store "
         "OUR_ADAPTATION_COST, the store's refit-without reference, and each rival's "
         f"retrain_without_cost; rows read are identical across seeds (asserted). Rivals are "
         f"fit on {rpool:,} rows; the store on its 131,072-row pool. Exactness is store-scoped; "
         "adaptation is light and data-free, never 'no change'.",
         extra={"rival_pool_rows": rpool})


# ---------------------------------------------------------------- 4. maze condition table (Part B)
def maze_per_seed(d, schema, row):
    r = d["rows"][row]
    if schema == "w19":
        return r["per_seed"]
    if schema == "w16":
        return {k: v["per_seed"] for k, v in r.items() if isinstance(v, dict) and "per_seed" in v}
    raise ValueError(schema)


def head_label(ps):
    """The Head's rule, as written in PREREG-w21-words-full-test.md (THE LABEL RULE)."""
    clu = ps["CLU"]
    matched = [k for k in ps if k.startswith("rival")]
    rstar = max(matched, key=lambda k: mean(ps[k]))
    ctrls = [k for k in ps if k.startswith("control")]
    gate = all(mean(ps[c]) <= mean(clu) - 0.15 for c in ctrls)
    diff = [c - r for c, r in zip(clu, ps[rstar])]
    dbar, sig = mean(diff), statistics.stdev(diff)
    ahead = sum(x > 0 for x in diff)
    if dbar > 0 and ahead >= 2 and gate:
        lab = "WIN"
    elif abs(dbar) <= sig:
        lab = "COMPARABLE"
    elif dbar < 0:
        lab = "LOSS"
    else:
        lab = "COMPARABLE"
    return dict(label=lab, R_star=rstar, R_star_per_seed=ps[rstar], diff_per_seed=diff,
                mean_diff=dbar, sigma=sig, seeds_ahead=ahead, store_gate=gate)


def pretty_arm(k):
    fam = k.split(":", 1)[1].strip().split(" ")[0]
    name = {"mlp": "MLP", "lstm": "LSTM", "transformer": "transformer",
            "cont_mlp": "continuous MLP"}[fam]
    extra = []
    if "10x-steps" in k:
        extra.append("10× steps")
    if "+extractor" in k:
        extra.append("+ extractor")
    return name + (f" ({', '.join(extra)})" if extra else ""), fam


def fig_maze_conditions(out):
    man = data("maze_conditions.json")
    rows = [r for r in man["rows"] if r["status"] == "landed"]
    sources = [man["rule_source"]]
    comp = []
    for r in rows:
        d = load(r["source"])
        sources.append(r["source"])
        sources += r.get("note_sources", [])
        assert d["leg"].upper() == r["leg"], (r["id"], d["leg"])
        ps = maze_per_seed(d, r["schema"], man["primary_row"])
        assert len(ps["CLU"]) == 3
        h = head_label(ps)
        if "HEAD_LABEL" in d:  # the artefact's own label must agree with this function
            assert d["HEAD_LABEL"]["PRIMARY"] == h["label"], (r["id"], h["label"])
            assert d["HEAD_LABEL"]["rows"][man["primary_row"]]["R_star"] == h["R_star"]
        shown = "ASYMMETRIC" if r.get("asymmetric") else h["label"]
        comp.append((r, ps, h, shown))

    n = len(comp)
    fig = plt.figure(figsize=(TEXT_W, 0.42 * n + 0.95), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.9, 0.62])
    tl, ax, tr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for a in (tl, tr):
        a.set_axis_off()
    for a in (tl, ax, tr):
        a.set_ylim(n - 0.5, -0.75)
    clu = ARMS["clu"]
    fam_color = {k: ARMS[k] for k in ("mlp", "lstm", "transformer", "cont_mlp")}
    used = set()
    for i, (r, ps, h, shown) in enumerate(comp):
        if i % 2 == 0:
            for a in (ax,):
                a.axhspan(i - 0.5, i + 0.5, color="#f4f3ef", lw=0, zorder=0)
        rname, fam = pretty_arm(h["R_star"])
        used.add(fam)
        rc = fam_color[fam]
        dy = 0.14
        ax.scatter(ps["CLU"], [i - dy] * 3, s=16, marker="o", color=clu["color"], zorder=4,
                   edgecolor="white", lw=0.4)
        ax.plot([mean(ps["CLU"])] * 2, [i - dy - 0.13, i - dy + 0.13], color=clu["color"],
                lw=1.4, zorder=3)
        ax.scatter(h["R_star_per_seed"], [i + dy] * 3, s=16, marker=rc["marker"],
                   color=rc["color"], zorder=4, edgecolor="white", lw=0.4)
        ax.plot([mean(h["R_star_per_seed"])] * 2, [i + dy - 0.13, i + dy + 0.13],
                color=rc["color"], lw=1.4, zorder=3)
        muted = r.get("asymmetric")
        tl.text(0.0, i - 0.12, r["label"], fontsize=6.6, color=INK2 if muted else INK,
                va="center", ha="left", fontweight="bold")
        sub = f"{r['leg']} · best matched rival: {rname}"
        tl.text(0.0, i + 0.17, sub, fontsize=5.4, color=INK2, va="center", ha="left")
        if muted:
            tl.text(0.0, i + 0.43, r["note"].replace(", the CLU", ",\nthe CLU"), fontsize=5.0,
                    color=INK2, va="center", ha="left", style="italic", linespacing=1.1)
        tr.text(0.04, i - 0.08, shown, fontsize=6.6, va="center", ha="left",
                color=INK2 if muted else INK, fontweight="bold")
        tr.text(0.04, i + 0.24, "store gate " + ("holds" if h["store_gate"] else "fails"),
                fontsize=5.2, va="center", ha="left", color=INK2)
    ax.set_xlim(-0.02, 1.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("unseen n-way validity, exact judge (fraction)")
    ax.set_title("CLU (●) and best matched rival, each seed", loc="left", fontsize=6.8)
    tl.text(0.0, -0.62, "condition", fontsize=6.4, color=INK2, va="center")
    tr.text(0.04, -0.62, "label", fontsize=6.4, color=INK2, va="center")
    handles = [Line2D([0], [0], color=clu["color"], marker="o", lw=0, label="CLU, each seed")]
    handles += [Line2D([0], [0], color=fam_color[f]["color"], marker=fam_color[f]["marker"], lw=0,
                       label=f"{fam_color[f]['name']} rival, each seed")
                for f in ("mlp", "lstm", "transformer", "cont_mlp") if f in used]
    handles += [Line2D([0], [0], color=INK2, lw=1.4, label="mean of 3 seeds")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=5.8)
    labels = {r["id"]: dict(shown=shown, head_rule=h["label"], R_star=h["R_star"],
                            R_star_per_seed=h["R_star_per_seed"], CLU_per_seed=ps["CLU"],
                            diff_per_seed=h["diff_per_seed"], mean_diff=h["mean_diff"],
                            sigma_paired=h["sigma"], seeds_ahead=h["seeds_ahead"],
                            store_gate=h["store_gate"], leg=r["leg"], source=r["source"])
              for r, ps, h, shown in comp}
    pending = [f"{r['id']}: {r['awaiting']}" for r in man["rows"] if r["status"] != "landed"]
    save(fig, out, "fig_maze_conditions", sources,
         "CMAZE-T condition table, primary row unseen n-way (exact judge), n = 3 seeds per row. "
         "Per-seed markers for the CLU and for R* (the matched rival with the highest 3-seed "
         "mean over every kind and both budgets); tick = 3-seed mean. Label = the Head's rule "
         "(PREREG-w21-words-full-test.md), computed by head_label() from each row's BARS "
         "artefact; asserted equal to HEAD_LABEL where the artefact carries one (WORDS-FULL "
         "TEST). No numeric margin bar is drawn. The observation-rival row is labelled "
         "ASYMMETRIC (rivals read a 636-dim observation, the CLU the 318-wide word design) and "
         "carries no claim label. Rows are read from data/maze_conditions.json in order.",
         extra={"labels": labels, "rows_not_landed": pending})
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", default=None,
                    help="verified per-item CMAZE-T extraction (Part C); omitted = no panels")
    ap.add_argument("--prov-name", default="PROVENANCE-w21.json")
    ap.add_argument("--merge", default=None,
                    help="an existing PROVENANCE.json: override its entries per stem, keep the rest")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_crossover(args.out)
    fig_ladder(args.out)
    fig_deletion(args.out)
    fig_maze_conditions(args.out)
    if args.panels:
        import cmazet_panels  # noqa: E402  (Part C; only with a verified extraction)
        cmazet_panels.build(args.out, args.panels, PROV, save_fn=save)
    with open(os.path.join(args.out, args.prov_name), "w", encoding="utf-8") as f:
        json.dump({"evidence_root": M.EVIDENCE, "matplotlib": M.matplotlib.__version__,
                   "backend": M.matplotlib.get_backend(), "figures": PROV}, f, indent=1,
                  sort_keys=True)
        f.write("\n")
    if args.merge:
        with open(args.merge, "r", encoding="utf-8") as f:
            base = json.load(f)
        for stem, e in base["figures"].items():
            if stem not in PROV:  # built by make_figures.py with the banner switched off
                e["banner"] = False
                if e.get("placeholder"):
                    e["placeholder"] = False
                    e["awaiting"] = None
                    e["w21"] = ("no in-image banner (w21 Part D); the figure's status lives in the "
                                "draft's placeholder index, never in the image")
        base["figures"].update(PROV)
        base["w21_figures_redraw"] = {
            "built_by": "make_figures.py then make_figures_w21.py (build.sh)",
            "stems_redrawn_or_added": sorted(PROV)}
        with open(args.merge, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=1, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
