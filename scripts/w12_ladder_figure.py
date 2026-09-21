"""w12 hub/composition-ladder -- THE FIGURE.

Panel A  x = number of jointly composed factors k, y = score, one line per arm, seed band.
Panel B  x = k, y = the PAIRED margin against the BEST LEARNED BASELINE at that (k, seed)
         -- the rule was fixed in PREREG section 5 before any arm number existed, and it
         picks the STRONGEST learner at each rung, not the mean and not a post-hoc choice.

House style is `scripts/w10_build_tables.py`'s, unchanged, so this figure sits beside the
paper's others.  The categorical palette was validated: lightness band PASS, chroma floor
PASS, CVD separation PASS (worst adjacent dE 9.1 protan), normal-vision floor PASS
(worst 19.6), contrast WARN -> every series is DIRECTLY LABELLED and a full table view
ships beside the figure (LADDER.csv / LADDER.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8b8a85"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d2", "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.dpi": 160,
})

RUNGS = ["2", "3", "4", "5", "6"]
#: hue is assigned to the ENTITY in fixed order and never cycled or re-assigned.
SERIES = [
    ("single_design_K5", "single design K=5", CAT[0], "-"),
    ("pca_tensor", "pca_tensor (w5 store)", CAT[1], "-"),
    ("rival_mlp", "MLP", CAT[2], "-"),
    ("rival_lstm", "LSTM", CAT[3], "-"),
    ("rival_transformer", "transformer", CAT[4], "-"),
    ("rival_cont_mlp", "continuous MLP", CAT[5], "-"),
    ("rival_table", "table floor", CAT[6], "-"),
]
CTRL = [
    ("single_design_K5__landscape_shuffled", "sd - shuffled", CAT[0], (0, (4, 2))),
    ("single_design_K5__random_atoms", "sd - random atoms", CAT[0], (0, (1, 2))),
    ("pca_tensor__landscape_shuffled", "pca - shuffled", CAT[1], (0, (4, 2))),
    ("pca_tensor__random_atoms", "pca - random atoms", CAT[1], (0, (1, 2))),
]


def series_xy(rep, arm, field):
    """(ks, mean, lo, hi, n_seeds_per_k) -- only rungs where the arm actually ran."""
    ks, mu, lo, hi, ns = [], [], [], [], []
    for r in RUNGS:
        A = rep["rungs"].get(r, {}).get("arms", {}).get(arm, {})
        if A.get("status") != "RUN":
            continue
        if field == "mean":
            v = [A["per_seed"][s]["mean"] for s in sorted(A["per_seed"])]
        else:
            v = [A["per_seed"][s]["paired_vs_best_learned"]["mean_paired_delta"]
                 for s in sorted(A["per_seed"])
                 if A["per_seed"][s].get("paired_vs_best_learned")]
        if not v:
            continue
        v = np.asarray(v, float)
        ks.append(int(r)); mu.append(v.mean()); lo.append(v.min()); hi.append(v.max())
        ns.append(v.size)
    return np.asarray(ks), np.asarray(mu), np.asarray(lo), np.asarray(hi), ns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.ladder, encoding="utf-8") as fh:
        rep = json.load(fh)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    nseeds = sorted({n for arm, _, _, _ in SERIES
                     for n in series_xy(rep, arm, "mean")[4]})
    nstr = ("n = %d seeds" % nseeds[0] if len(nseeds) == 1
            else "n = %s seeds (varies -- see the table)" % nseeds)
    ncorn = {r: rep["rungs"][r]["n_corners_per_seed"] for r in RUNGS
             if rep["rungs"].get(r, {}).get("n_corners_per_seed")}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # ---- Panel A: the ladder -------------------------------------------
    ax = axes[0]
    for arm, lab, col, ls in SERIES:
        ks, mu, lo, hi, _ = series_xy(rep, arm, "mean")
        if not ks.size:
            continue
        ax.fill_between(ks, lo, hi, color=col, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(ks, mu, color=col, linestyle=ls, zorder=4,
                marker="o", markersize=5.0, markerfacecolor=col,
                markeredgecolor=SURFACE, markeredgewidth=2.0)
        ax.annotate(" " + lab, (ks[-1], mu[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.4, color=INK2, va="center", zorder=6)
    for arm, lab, col, ls in CTRL:
        ks, mu, lo, hi, _ = series_xy(rep, arm, "mean")
        if not ks.size:
            continue
        ax.plot(ks, mu, color=col, linestyle=ls, linewidth=1.3, alpha=0.75, zorder=3)
    ax.set_xlabel("k  -  number of factors jointly composed in the held-out box")
    ax.set_ylabel("distinct valid NOVEL tuples per corner")
    ax.set_title("A. the composition ladder", fontsize=9.5)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    # ---- Panel B: the paired margin vs the BEST learned baseline --------
    ax = axes[1]
    ax.axhline(0.0, color=INK3, linewidth=1.2, zorder=1)
    ax.axhline(rep["margin"], color=INK3, linewidth=1.2, linestyle=(0, (5, 3)), zorder=1)
    ax.annotate("declared margin +%.1f" % rep["margin"], (6.02, rep["margin"]),
                xytext=(2, 2), textcoords="offset points", fontsize=7.0, color=INK2)
    for arm, lab, col, ls in SERIES:
        ks, mu, lo, hi, _ = series_xy(rep, arm, "delta")
        if not ks.size:
            continue
        ax.fill_between(ks, lo, hi, color=col, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(ks, mu, color=col, linestyle=ls, zorder=4,
                marker="o", markersize=5.0, markerfacecolor=col,
                markeredgecolor=SURFACE, markeredgewidth=2.0)
        ax.annotate(" " + lab, (ks[-1], mu[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.4, color=INK2, va="center", zorder=6)
    for arm, lab, col, ls in CTRL:
        ks, mu, lo, hi, _ = series_xy(rep, arm, "delta")
        if not ks.size:
            continue
        ax.plot(ks, mu, color=col, linestyle=ls, linewidth=1.3, alpha=0.75, zorder=3)
    ax.set_xlabel("k  -  number of factors jointly composed in the held-out box")
    ax.set_ylabel("paired per-corner margin vs the BEST learned baseline")
    ax.set_title("B. margin against the strongest learner at each rung", fontsize=9.5)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], color=c, linestyle=s, label=l) for _, l, c, s in SERIES]
    handles += [plt.Line2D([], [], color=INK3, linestyle=(0, (4, 2)), linewidth=1.3,
                           label="control: landscape_shuffled"),
                plt.Line2D([], [], color=INK3, linestyle=(0, (1, 2)), linewidth=1.3,
                           label="control: random_atoms")]
    fig.legend(handles=handles, frameon=False, ncol=5, fontsize=7.6,
               loc="lower center", bbox_to_anchor=(0.5, -0.10))

    bl = {r: sorted({rep["rungs"][r]["best_learned_per_seed"][s]["arm"]
                     for s in rep["rungs"][r]["best_learned_per_seed"]})
          for r in RUNGS if rep["rungs"].get(r, {}).get("best_learned_per_seed")}
    fig.suptitle("Shapes3D composition ladder - how far composition stretches as k grows",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.5, -0.20,
             ("Band = min..max over seeds; line = seed mean.  %s; corners per rung per seed: %s.  "
              "Metric and margin are the harness's (LADDER-READY.json:estimand); "
              "unit = corner.\nBest learned baseline per (k, seed) = the learned arm with the "
              "largest mean, a rule fixed in PREREG section 5 before any arm number existed: %s.  "
              "Learned arms sit at the 524,288 B REFERENCE pin, not at 1x.")
             % (nstr, ncorn, bl),
             ha="center", fontsize=7.0, color=INK2)
    fig.tight_layout()
    p = out / "fig_composition_ladder.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
