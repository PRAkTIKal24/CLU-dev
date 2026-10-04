"""w17/plots -- the paper's figures, built from w<=14 evidence artefacts only.

Every plotted value is read from a file under EVIDENCE (read-only). Nothing is typed in by
hand, rounded or recalled; where a figure shows a mean it is either the artefact's own mean or
is checked against it. Provenance for each figure is written to <out>/PROVENANCE.json.

AMENDMENT 1 (Head, 2026-09-22): no w15 or w16 artefact is read. Figures that wave 18 must
regenerate carry a PLACEHOLDER tag and are listed in PROVENANCE.json.

Determinism: Agg/PDF backends pinned, bundled DejaVu Sans, no timestamps in PDF/PNG metadata,
inputs read in sorted order, no randomness. Build twice and compare shasums (see FINDINGS.md).

usage: python w17_plots/make_figures.py --out <dir>
"""
import argparse
import csv
import json
import os

os.environ["SOURCE_DATE_EPOCH"] = "0"
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from chlu.paths import out_root as _out_root
EVIDENCE = str(_out_root())

# ---------------------------------------------------------------- style
# One arm = one name = one colour, in every figure. Categorical slots 1..6 of the dataviz
# reference palette in its validated order; floor and controls are neutral greys.
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
FLOOR = "#6b6a66"
CONTROL = "#b5b4ae"
ARMS = {
    "clu":         dict(name="CLU", color="#2a78d6", marker="o"),
    "learned_1x":  dict(name="Learned 1× (four-part total)", color="#eb6834", marker="s"),
    "mlp":         dict(name="MLP", color="#1baf7a", marker="^"),
    "lstm":        dict(name="LSTM", color="#eda100", marker="v"),
    "transformer": dict(name="Transformer", color="#e87ba4", marker="D"),
    "cont_mlp":    dict(name="Continuous MLP", color="#008300", marker="P"),
    "table":       dict(name="Lookup table (floor)", color=FLOOR, marker="x"),
    "controls":    dict(name="Charter-C controls", color=CONTROL, marker="."),
    # encoder-adapt configurations of the CLU (slots 7, 8)
    "adapter":     dict(name="CLU + adapter", color="#4a3aa7", marker="s"),
    "anchor":      dict(name="CLU + anchor", color="#e34948", marker="^"),
}
# the four byte-ledger parts of the CLU: one hue (the CLU's), light -> dark
PARTS = [
    ("p1_atom_bytes", "atoms", "#b7d3f2"),
    ("p2_covariance_plus_count_bytes", "covariance + count", "#6fa6e6"),
    ("p3_address_book_bytes", "address book", "#2a78d6"),
    ("p4_coupling_hole_table_bytes", "coupling / hole table", "#0d3f7e"),
]

TEXT_W = 5.5  # ICLR text width, inches
PLACEHOLDER_TAG = "PLACEHOLDER · w≤14 evidence · wave 18 regenerates"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "pdf.compression": 6,
    "svg.hashsalt": "w17-plots",
    "path.simplify": False,
})

PROV = {}


def ev(rel):
    return os.path.join(EVIDENCE, rel)


def load(rel):
    with open(ev(rel), "r", encoding="utf-8") as f:
        return json.load(f)


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def tag_placeholder(fig):
    # w21 figures-redraw (Part D): no in-image banner on any figure. The placeholder status of a
    # figure lives in PROVENANCE.json and the draft's placeholder index, never in the image.
    # fig_crossover / fig_ladder / fig_deletion are redrawn by make_figures_w21.py (build.sh).
    return None


def save(fig, out, stem, sources, placeholder, awaiting, notes):
    fig.savefig(os.path.join(out, stem + ".pdf"),
                metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                          "Creator": None})
    fig.savefig(os.path.join(out, stem + ".png"), metadata={"Software": None})
    plt.close(fig)
    PROV[stem] = dict(sources=sorted(set(sources)), placeholder=placeholder,
                      awaiting=awaiting, notes=notes)


# PW-1 items 2, 3, 13: words no rendered string of the figures this module family owns may carry
# (fig_ladder, fig_crossover, fig_deletion, fig_headline_steps, fig_encoder_adapt). The maze
# figures have their own check (cmazet_panels.check_vocab). A bare "CLU" is banned: n-CLU only.
import re  # noqa: E402

PW1_BANNED = re.compile(
    r"(?<!n-)\bCLU\b|rival|corner|\bstore\b|\bgate\b|artefact|artifact|pre-?regist|\bA1\b|"
    r"\bA10\b|C-512|improved|config|spoke|\bhub\b|\bwave\b|charter|ledger|oblong|exact|"
    r"budget|Transf\.|Cont\. MLP|∧|10x|\bpric|\bbuys?\b|\bpaid\b|\bpays?\b|scored run|"
    r"landscape_|random_atoms|composition \(|held-out composition", re.IGNORECASE)


PW1_TEXT = []  # rendered strings per checked figure, in call order (read by build_shapes_only)


def check_vocab_pw1(fig):
    """Assert every rendered string obeys the PW-1 vocabulary; returns the strings checked."""
    fig.canvas.draw()  # tick labels are only set on draw
    seen = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s:
            continue
        m = PW1_BANNED.search(s)
        assert m is None, f"PW-1 vocabulary: {m.group(0)!r} in rendered text {s!r}"
        seen.append(s)
    PW1_TEXT.append(sorted(set(seen)))
    return sorted(set(seen))


def legend_handle(key, line=True):
    a = ARMS[key]
    return Line2D([0], [0], color=a["color"], marker=a["marker"],
                  lw=1.5 if line else 0, label=a["name"])


# ---------------------------------------------------------------- 1. crossover
def fig_crossover(out):
    src = "w12-single-design-crossover/CROSSOVER-val.json"
    d = load(src)
    Ks = sorted(int(k) for k in d["per_K"])
    rows = [d["per_K"][str(k)] for k in Ks]
    seed0 = [r["val_composition_per_seed"]["0"] for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.3), constrained_layout=True)

    clu = ARMS["clu"]
    a1.plot(Ks, seed0, color=clu["color"], marker=clu["marker"], lw=1.5, ms=3.5)
    multi = [(k, r) for k, r in zip(Ks, rows) if r["n_seeds"] >= 3]
    for k, r in multi:
        vals = [r["val_composition_per_seed"][s] for s in sorted(r["val_composition_per_seed"])]
        a1.vlines(k, min(vals), max(vals), color=clu["color"], lw=3, alpha=0.35)
        a1.scatter([k] * len(vals), vals, s=10, facecolor="white", edgecolor=clu["color"],
                   lw=0.8, zorder=3)
        a1.scatter([k], [r["val_composition_mean"]], marker="_", s=60, color=clu["color"],
                   lw=1.5, zorder=4)
    # the four charter-C controls at K*: all read exactly 0 in the artefact
    gate = d["control_gate_at_K_star"]
    ctrl_means = [gate[c]["mean"] for c in sorted(gate)]
    a1.scatter([d["K_star"]] * len(ctrl_means), ctrl_means, color=CONTROL, marker="x", s=14,
               zorder=3)
    a1.set_xlabel("K, wells")
    a1.set_ylabel("val composition (valid fraction)")
    a1.set_title("(a) Composition vs K", loc="left")
    a1.set_xticks(Ks)
    a1.set_ylim(-0.03, 1.0)
    a1.legend(handles=[
        Line2D([0], [0], color=clu["color"], marker="o", ms=3.5, label="CLU, seed 0 (n = 1)"),
        Line2D([0], [0], color=clu["color"], lw=3, alpha=0.35,
               label="CLU, seeds 0–2, range (n = 3)"),
        Line2D([0], [0], color=CONTROL, marker="x", lw=0, label="controls at K* (4)"),
    ], loc="lower right")

    bottom = [0] * len(Ks)
    for key, name, col in PARTS:
        vals = [r["ledger_four_part"][key] for r in rows]
        a2.bar(Ks, vals, width=2.0, bottom=bottom, color=col, edgecolor="white", lw=0.5,
               label=name)
        bottom = [b + v for b, v in zip(bottom, vals)]
    totals = [r["ledger_four_part"]["honest_total_bytes"] for r in rows]
    assert all(close(b, t, 0.5) for b, t in zip(bottom, totals)), "four parts != honest total"
    ref = rows[0]["ledger_four_part"]["reference_pin_bytes"]
    a2.axhline(ref, color=INK2, lw=0.8, ls="--", label=f"{ref:,} B reference")
    a2.set_xlabel("K, wells")
    a2.set_ylabel("bytes (\u00d710\u2076), four parts,\nsummed, never netted")
    a2.set_title("(b) Four-part byte ledger vs K", loc="left")
    a2.set_xticks(Ks)
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}"))
    a2.legend(loc="upper right", ncol=1)
    tag_placeholder(fig)
    save(fig, out, "fig_crossover", [src], True,
         "w15 crossover-seeds (3 seeds at every K) -- may confirm, narrow, or overturn the "
         "no-crossover reading",
         "n = 1 (seed 0) at K = 8..28; n = 3 (seeds 0,1,2) at K = 5 and K = 32. Byte "
         "convention: the crossover hub's four-part honest ledger (atoms + covariance/count + "
         "address book + coupling/hole), summed and never netted; 524,288 B drawn as a "
         "reference line only. The artefact's first key is SMOKE_NOT_A_CLAIM: true; the "
         "honest_total_bytes column (not the parts-1-2-4 column) is the one plotted, per the "
         "w13 advisor report section 5.")


# ---------------------------------------------------------------- 2. ladder
def ladder_rival_rows():
    src = "w12-composition-ladder/LADDER.csv"
    out = {}
    with open(ev(src), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["rung"] == "shipped":
                continue
            out[(r["arm"], int(r["k"]), int(r["seed"]))] = float(
                r["mean_distinct_valid_novel_tuples"])
    return src, out


def fig_ladder(out):
    src = "w12b-nr6-and-kstar/KSTAR-LADDER.json"
    d = load(src)
    rsrc, riv = ladder_rival_rows()
    ns = sorted(int(n) for n in d["by_rung"])
    seeds = [0, 1, 2]

    def per_seed(n, what):
        b = d["by_rung"][str(n)]["kstar_K32"][what]
        assert b["n"] == 3
        return b["per_seed"], b["mean"]

    series = {}
    for key, what in (("clu", "store"), ("learned_1x", "learned_1x")):
        series[key] = [per_seed(n, what) for n in ns]
    for key, arm in (("mlp", "rival_mlp"), ("lstm", "rival_lstm"),
                     ("transformer", "rival_transformer"), ("cont_mlp", "rival_cont_mlp"),
                     ("table", "rival_table")):
        rows = []
        for n in ns:
            ps = [riv[(arm, n, s)] for s in seeds]
            rows.append((ps, sum(ps) / len(ps)))
        series[key] = rows
    # "best learned" = the strongest learned arm per (rung, seed); verify it is learned_1x
    best_is_1x = True
    for i, n in enumerate(ns):
        for j, s in enumerate(seeds):
            one_x = series["learned_1x"][i][0][j]
            for key in ("mlp", "lstm", "transformer", "cont_mlp"):
                if series[key][i][0][j] > one_x:
                    best_is_1x = False
    ceilings = []
    for n in ns:
        c = {d["legs"][str(n)]["kstar_K32"][str(s)]["ceiling_per_corner"] for s in seeds}
        assert len(c) == 1
        ceilings.append(c.pop())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 3.0), constrained_layout=True)
    a1.step(ns, ceilings, where="mid", color=INK2, lw=0.8, ls=":")
    order = ["clu", "learned_1x", "mlp", "lstm", "transformer", "cont_mlp", "table"]
    for key in order:
        a = ARMS[key]
        lo = [min(ps) for ps, _ in series[key]]
        hi = [max(ps) for ps, _ in series[key]]
        mu = [m for _, m in series[key]]
        ls = "--" if key == "table" else "-"
        a1.fill_between(ns, lo, hi, color=a["color"], alpha=0.15, lw=0)
        a1.plot(ns, mu, color=a["color"], marker=a["marker"], ls=ls,
                lw=1.8 if key == "clu" else 1.2, ms=3.5)
    a1.set_xticks(ns)
    a1.set_xlabel("n, factors jointly composed\n(n ≥ 3: the same 120 corners)")
    a1.set_ylabel("distinct valid novel tuples / corner")
    a1.set_title("(a) Composition vs n, K* = 32", loc="left")
    handles = [legend_handle(k) for k in order]
    for h in handles:
        if h.get_label() in ("MLP", "LSTM", "Transformer", "Continuous MLP"):
            h.set_label(h.get_label() + " (524,288 B)")
    handles[-1].set_linestyle("--")
    handles.append(Line2D([0], [0], color=INK2, ls=":", lw=0.8, label="per-corner ceiling"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6)
    a1.set_ylim(0, 80)

    m = [per_seed(n, "margin") for n in ns]
    clu = ARMS["clu"]
    lo = [min(ps) for ps, _ in m]
    hi = [max(ps) for ps, _ in m]
    a2.fill_between(ns, lo, hi, color=clu["color"], alpha=0.2, lw=0)
    for j in range(3):
        a2.scatter(ns, [ps[j] for ps, _ in m], s=9, facecolor="white",
                   edgecolor=clu["color"], lw=0.8, zorder=3)
    a2.plot(ns, [mu for _, mu in m], color=clu["color"], marker="o", ms=3.5)
    a2.axhline(0, color=INK2, lw=0.7)
    a2.axhline(1.0, color=INK2, lw=0.7, ls="--")
    a2.text(ns[-1], 1.0, "declared +1.0", fontsize=6, color=INK2, ha="right", va="bottom")
    a2.set_xticks(ns)
    a2.set_xlabel("n, factors jointly composed")
    a2.set_ylabel("CLU − opponent (tuples / corner)")
    title = "(b) Margin vs best learned arm" if best_is_1x else "(b) Margin vs learned 1×"
    a2.set_title(title, loc="left")
    tag_placeholder(fig)
    save(fig, out, "fig_ladder", [src, rsrc], True,
         "wave 18 regeneration (Amendment 1 lists the ladder); no w15/w16 row is read",
         f"3 seeds (0,1,2) on every rung; mean line, min-max seed band. Learned 1x uses the "
         f"crossover hub's FOUR-PART convention at K* (the one convention used in every "
         f"figure); the K = 5 full-square-priced opponent (the inflated row) is NOT drawn. "
         f"MLP/LSTM/Transformer/continuous MLP/table are the 524,288 B reference-pin rivals "
         f"from the w12 ladder (K-independent). Panel (b) plots KSTAR-LADDER.json "
         f"by_rung.*.kstar_K32.margin per seed. learned_1x is the strongest learned arm at "
         f"every (rung, seed): {best_is_1x}. For n >= 3 the axis is a TRAINING-DATA axis "
         f"(identical 120 corners); only n = 2 vs n = 6 is a depth contrast.")


# ---------------------------------------------------------------- 3. deletion
def fig_deletion(out):
    src = "w12-single-design-crossover/deletion/test_seed0_K32/deletion.json"
    d = load(src)
    assert d["K"] == 32 and d["seed"] == 0 and d["condition"] == "test"
    sc = d["scopes"]
    groups = [
        ("value", "survivors", "red value:\nsurvivors"),
        # w18 seeds-and-tables pass 4 (Head DELETION-RULE-AMENDMENT item 2): the deleted-corner
        # ("red value: red corners") bars are RETIRED from every table and figure; the evidence stays
        # in the artefact (scopes.value.{before_same_corners,after}.deleted). Survivors only: measurement (d).
        ("conjunction_live", "survivors", "red sphere:\nsurvivors"),
        ("item", "survivors", "item 1 %:\nsurvivors"),
    ]
    before, after, ncs = [], [], []
    for scope, part, _ in groups:
        b = sc[scope]["before_same_corners"][part]
        a = sc[scope]["after"][part]
        assert b["n_corners"] == a["n_corners"]
        before.append(b["novel_per_corner_mean"])
        after.append(a["novel_per_corner_mean"])
        ncs.append(a["n_corners"])

    rsrcs = []
    riv = []
    for key in ("mlp", "cont_mlp", "transformer", "lstm"):
        rel = f"w12-deletion-baselines/comparators/seed_0/{key}__retrain_without__value-object_hue-0.rival.json"
        r = load(rel)
        rsrcs.append(rel)
        cl = r["cost_line"]
        riv.append((key, cl["seconds"], cl["data_rows_read"], r["deletion_counts"]["n_fit_before"]))
    ours = sc["value"]["our_cost_line"]
    ref = sc["value"]["exactness_store_scoped"]["refit_without_cost_line"]
    pool = json.loads(sc["value"]["receipt"]["n_before"])[0]

    fig = plt.figure(figsize=(TEXT_W, 4.1), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 1.05])
    a1 = top.subplots()
    a2, a3 = bot.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 1]})
    clu = ARMS["clu"]
    x = list(range(len(groups)))
    w = 0.38
    a1.bar([i - w / 2 for i in x], before, w, color="white", edgecolor=clu["color"], lw=1.0,
           hatch="////", label="before deletion")
    a1.bar([i + w / 2 for i in x], after, w, color=clu["color"], edgecolor="white", lw=0.5,
           label="after deletion")
    a1.set_xticks(x)
    a1.set_xticklabels([f"{g[2]}\n({n} corners)" for g, n in zip(groups, ncs)], fontsize=6.5)
    a1.set_ylabel("distinct valid novel tuples / corner")
    a1.set_title("(a) CLU composition, K* = 32", loc="left")
    a1.legend(loc="upper right", fontsize=6.5, ncol=2)
    a1.set_ylim(0, 65)

    labels = ["CLU downdate", "CLU refit-without"] + [f"{ARMS[k]['name']} retrain" for k, *_ in riv]
    colors = [clu["color"], clu["color"]] + [ARMS[k]["color"] for k, *_ in riv]
    hatches = ["", "////"] + [""] * len(riv)
    secs = [ours["seconds"], ref["seconds"]] + [s for _, s, _, _ in riv]
    rows = [ours["data_rows_read"], ref["data_rows_read"]] + [r for _, _, r, _ in riv]
    pools = [pool, pool] + [p for *_, p in riv]
    y = list(range(len(labels)))[::-1]
    for yi, v, c, h in zip(y, secs, colors, hatches):
        a2.barh(yi, v, 0.7, color="white" if h else c, edgecolor=c, hatch=h, lw=0.8)
    a2.set_xscale("log")
    a2.set_yticks(y)
    a2.set_yticklabels([f"{lab} ({p:,}-row pool)" for lab, p in zip(labels, pools)], fontsize=5.8)
    a2.set_xlabel("seconds (log)")
    a2.set_title("(b) Adaptation time", loc="left")
    for yi, v, c, h in zip(y, rows, colors, hatches):
        a3.barh(yi, v, 0.7, color="white" if h else c, edgecolor=c, hatch=h, lw=0.8)
        a3.text(v, yi, f" {v:,}", va="center", fontsize=5.8, color=INK)
    a3.set_yticks(y)
    a3.set_yticklabels([])
    a3.set_xlabel("data rows read")
    a3.set_title("(c) Data read", loc="left")
    a3.set_xlim(0, max(rows) * 1.45)
    a3.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a3.tick_params(axis="x", labelsize=6)
    tag_placeholder(fig)
    save(fig, out, "fig_deletion", [src] + rsrcs, True,
         "w15 single-design-seeds (deletion rows at seeds 1-2) -- may reproduce or break the "
         "seed-0 reading",
         "CLU: 1 seed (0), TEST, K = 32, value scope = every fit row with object_hue index 0 "
         "(11,044 rows). Exactness is STORE-SCOPED; the frozen encoder/decoder and the "
         "address book are outside the deletable object (the book is carried). Post-deletion "
         "adaptation is light and data-free, not 'no change'. Rivals delete only by "
         "retrain-without on their OWN 16,384-row fit pool (seed 0 cost line; the pools "
         "differ and are printed on the axis). Rival post-deletion composition on the scored "
         "corners was never measured (DECLARED NOT-RUN in the evidence), so panel (a) is "
         "CLU-only. Panel (a) prints survivor corners only (measurement (d)); the deleted-corner "
         "bars were retired by the Head's deletion-rule amendment (w18 seeds-and-tables pass 4).")


# ---------------------------------------------------------------- 4. encoder-adapt
def fig_encoder_adapt(out):
    src = "w13-encoder-adapt/CROSS-ARM.json"
    d = load(src)
    assert d["seed_count"] == 3
    M = d["MARGIN"]
    seeds = sorted(M["adapter"]["per_seed"], key=int)
    cfgs = ["frozen", "adapter", "anchor"]
    fig, ax = plt.subplots(figsize=(TEXT_W * 0.55, 2.2), constrained_layout=True)
    for s in seeds:
        fz = M["adapter"]["per_seed"][s]["margin_frozen"]
        assert close(fz, M["anchor"]["per_seed"][s]["margin_frozen"])
        vals = [fz, M["adapter"]["per_seed"][s]["margin_adapter"],
                M["anchor"]["per_seed"][s]["margin_anchor"]]
        ax.plot([0, 1], vals[:2], color=CONTROL, lw=0.8, zorder=1)
        ax.plot([0, 2], [vals[0], vals[2]], color=CONTROL, lw=0.8, zorder=1)
        for i, (c, v) in enumerate(zip(cfgs, vals)):
            a = ARMS["clu" if c == "frozen" else c]
            ax.scatter([i], [v], color=a["color"], marker=a["marker"], s=22, zorder=3,
                       edgecolor="white", lw=0.5)
        ax.text(-0.1, vals[0], f"seed {s}", fontsize=6, color=INK2, va="center", ha="right")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["frozen encoder", "adapter", "anchor"])
    ax.set_xlim(-0.55, 2.25)
    # PW-1 items 2-3: no internal arm name, "held-out combination" (never corner)
    ax.set_ylabel("n-CLU margin over the 180k-parameter\nMLP baseline, novel tuples\nper held-out combination",
                  fontsize=6.2)
    ax.set_title("Encoder adaptation, K* = 32, 3 seeds", loc="left")
    check_vocab_pw1(fig)
    save(fig, out, "fig_encoder_adapt", [src], False, None,
         "3 seeds x 3 configurations = 9 arm-runs (3 SEEDS). Margin = single design - best "
         "learned baseline (mlp_180k, chosen per arm); a DIFFERENT opponent from the ladder's "
         "learned 1x, so this figure is never merged with fig_ladder. Equal compute: 400 "
         "steps / 25,600 images per arm-run (printed in CROSS-ARM.md).")


# ---------------------------------------------------------------- 5. maze (placeholders)
MAZE_ARMS = [("clu", "clu"), ("lookup_matched", "table"), ("mlp", "mlp"), ("lstm", "lstm"),
             ("transformer", "transformer")]
MAZE_CTRLS = ["landscape_shuffled", "random_atoms", "landscape_removed"]


def fig_maze_readverb(out):
    runs = ["w14-read-verb/runs/scored1_seed0_test.json",
            "w14-read-verb/runs/scored2_seed1_test.json",
            "w14-read-verb/runs/scored3_seed2_test.json"]
    ds = [load(r) for r in runs]
    assert [x["seed"] for x in ds] == [0, 1, 2]
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_W, 2.4), constrained_layout=True, sharey=True)
    for ax, row, title in zip(axs, ["unseen_pair", "nway"], ["(a) Unseen pair", "(b) n-way"]):
        ceil = [x["pipeline_ceiling_rows"][row]["validity"] for x in ds]
        ax.axhspan(min(ceil), max(ceil), color=GRID, alpha=0.9, lw=0, zorder=0)
        labels = []
        allarms = [(k, ARMS[a]) for k, a in MAZE_ARMS] + [(c, ARMS["controls"]) for c in MAZE_CTRLS]
        for i, (k, a) in enumerate(allarms):
            avg = [x["table"][row][k]["AVG"]["validity"] for x in ds]
            mrg = [x["table"][row][k]["MERGE_target_wins"]["validity"] for x in ds]
            for off, vals, filled in ((-0.18, avg, False), (0.18, mrg, True)):
                mu = sum(vals) / 3
                ax.bar(i + off, mu, 0.34, color=a["color"] if filled else "white",
                       edgecolor=a["color"], lw=0.8, hatch="" if filled else "////")
                ax.scatter([i + off] * 3, vals, s=5, color=INK, zorder=3, lw=0)
            labels.append(a["name"] if k not in MAZE_CTRLS else k.replace("_", " "))
        ax.set_xticks(range(len(allarms)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=5.8)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("validity, exact judge (fraction)")
    axs[1].legend(handles=[
        Patch(facecolor="white", edgecolor=INK2, hatch="////", label="AVG (latent mean)"),
        Patch(facecolor=INK2, edgecolor=INK2, label="MERGE (plan-space verb)"),
        Patch(facecolor=GRID, label="pipeline ceiling, seed range"),
    ], loc="upper right", fontsize=6)
    tag_placeholder(fig)
    save(fig, out, "fig_maze_readverb", runs, True,
         "w16 maze-wm rows (composition, rollout validity, deletion) -- each may land as a "
         "declared negative",
         "CMAZE-2b only (w14 request layer, GRID_REBASED budgets), TEST, 3 seeds; bar = mean "
         "of the three per-seed values, dots = seeds. MERGE = MERGE_target_wins (the arm, per "
         "w14-read-verb/TABLES.md). This is an INTERFACE (verb) result: the same merge lifts "
         "the matched-bytes lookup floor more than the CLU and the charter-C controls do NOT "
         "collapse, so no store row is claimed. The lookup floor's ~0.97 is the WITHIN-MAZE "
         "floor that cannot transfer across mazes.")


# environment audit, not an arm: neutral shades, never an arm colour
ENV_SHADES = ["#3d3c39", "#7a7974", "#b0afa9"]


def fig_maze_requestlayer(out):
    b_rel = "w14-request-layer-fix/artefacts/measure_before_0.json"
    a_rels = [f"w14-request-layer-fix/artefacts/measure_after_{i}.json" for i in range(3)]
    b = load(b_rel)
    As = [load(r) for r in a_rels]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.6), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1.6]})
    keys = ["1", "2", "3", "4"]
    x = list(range(len(keys)))
    bh = [b["key_count_histogram"].get(k, 0) for k in keys]
    ah = [[A["key_count_histogram"].get(k, 0) for k in keys] for A in As]
    w = 0.2
    a1.bar([i - 1.5 * w for i in x], bh, w, color="white", edgecolor=INK2, hatch="////", lw=0.8,
           label="CMAZE-2 layer (w9), seed 0")
    for j, h in enumerate(ah):
        a1.bar([i + (j - 0.5) * w for i in x], h, w, color=ENV_SHADES[j],
               edgecolor="white", lw=0.4,
               label=f"CMAZE-2b layer (w14), seed {j}")
    a1.set_xticks(x)
    a1.set_xticklabels(keys)
    a1.set_xlabel("keys on the optimal route")
    a1.set_ylabel("routes (count)")
    a1.set_title("(a) Keys per route", loc="left")
    fig.legend(loc="outside lower center", ncol=2, fontsize=6)

    planners = list(b["all_pairs_slack10"]["blind"].keys())
    short = [p.split("(")[0].strip() for p in planners]
    x = list(range(len(planners)))
    bv = [b["all_pairs_slack10"]["blind"][p]["validity"] for p in planners]
    a2.bar([i - 1.5 * w for i in x], bv, w, color="white", edgecolor=INK2, hatch="////", lw=0.8)
    for j, A in enumerate(As):
        assert list(A["all_pairs_slack10"]["blind"].keys()) == planners
        av = [A["all_pairs_slack10"]["blind"][p]["validity"] for p in planners]
        a2.bar([i + (j - 0.5) * w for i in x], av, w, color=ENV_SHADES[j],
               edgecolor="white", lw=0.4)
    a2.set_xticks(x)
    a2.set_xticklabels(short, rotation=30, ha="right", fontsize=6)
    a2.set_ylabel("validity at opt + 10 % budget")
    a2.set_ylim(0, 1.05)
    a2.set_title("(b) Scripted planners, all pairs", loc="left")
    tag_placeholder(fig)
    save(fig, out, "fig_maze_requestlayer", [b_rel] + a_rels, True,
         "w16 maze-wm (CMAZE-T) rows supersede the maze section -- APPENDIX figure; may be "
         "dropped or re-scoped",
         "Environment audit, not a scoreboard: the request layer before (CMAZE-2, w9) and "
         "after (CMAZE-2b, w14) the one-rule fix. No learned arm appears. BEFORE is seed 0 "
         "only (the only before-artefact on disk); AFTER is seeds 0,1,2. SLACK10 = CMAZE-1's "
         "own opt + ceil(0.10 opt) budget rule. Two environments share this figure ONLY "
         "because the figure is the fix itself; no score from one is compared with a score "
         "from the other.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_crossover(args.out)
    fig_ladder(args.out)
    fig_deletion(args.out)
    fig_encoder_adapt(args.out)
    # PW-1 item 14 retires fig_maze_readverb and fig_maze_requestlayer: the functions are kept,
    # they are no longer built
    with open(os.path.join(args.out, "PROVENANCE.json"), "w", encoding="utf-8") as f:
        json.dump({"evidence_root": EVIDENCE, "matplotlib": matplotlib.__version__,
                   "backend": matplotlib.get_backend(), "figures": PROV}, f, indent=1,
                  sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
