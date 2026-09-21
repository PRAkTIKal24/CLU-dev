"""w12 hub/composition-ladder -- THE FIGURE.

Panel A  x = number of jointly composed factors k, y = score, ONE LINE PER ARM, seed band.
Panel B  x = k, y = the PAIRED margin against the BEST LEARNED BASELINE at that (k, seed)
         -- the rule was fixed before any arm number existed (PREREG section 5, amended by
         AMENDMENT-3 so the opponent can only get STRONGER), and it picks the strongest
         learner at each rung, never the mean and never a post-hoc choice.

COLOUR (dataviz skill, procedure followed in order; colour chosen LAST).
The programme's house palette is the skill's own reference instance
(`scripts/w10_build_tables.py:CAT` == `references/palette.md`'s 8 categorical slots). It is
validated for ADJACENT pairs. A line chart lets any two series be compared, so this figure
was validated at `--pairs all`, where the full 8 slots FAIL (worst normal-vision dE 7.1,
`#e34948` vs `#eb6834`). The skill's remedy for that is to cut series or facet rather than
generate a hue, so the arms are grouped by BYTE PIN + POPULATION -- four entities, four
hues -- and the arm inside a group is carried by line style plus a direct label. Measured:

  node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a,#4a3aa7" --mode light --pairs all
    [PASS] lightness band   [PASS] chroma floor
    [PASS] CVD separation   worst all-pairs #1baf7a<->#eb6834 dE 9.2 (deutan) - tritan 9.6
    [PASS] normal-vision    worst all-pairs #4a3aa7<->#2a78d6 dE 16.3
    [WARN] contrast         #1baf7a 2.74:1 -> relief taken as DIRECT LABELS + a table view
                            (LADDER.md / LADDER.csv ship beside this figure)
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
#: the four validated hues, assigned to the four ENTITY GROUPS in fixed order.
H_SD, H_PCA, H_REF, H_OWN = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d2", "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.dpi": 160,
})

RUNGS = ["2", "3", "4", "5", "6"]

SOLID = "-"
DASH = (0, (5, 2))
DOT = (0, (1, 2))
DASHDOT = (0, (5, 2, 1, 2))
LONG = (0, (8, 2))
FINE = (0, (2, 1))

#: (arm, label, hue, style, linewidth, is_primary)
SERIES = [
    ("single_design_K5", "single design K=5", H_SD, SOLID, 2.2, True),
    ("single_design_K5__landscape_shuffled", "· landscape_shuffled", H_SD, DASH, 1.3, False),
    ("single_design_K5__random_atoms", "· random_atoms", H_SD, DOT, 1.3, False),
    ("pca_tensor", "pca_tensor (w5 store)", H_PCA, SOLID, 2.2, True),
    ("pca_tensor__landscape_shuffled", "· landscape_shuffled", H_PCA, DASH, 1.3, False),
    ("pca_tensor__random_atoms", "· random_atoms", H_PCA, DOT, 1.3, False),
    ("rival_mlp", "MLP", H_REF, SOLID, 1.5, False),
    ("rival_lstm", "LSTM", H_REF, DASH, 1.5, False),
    ("rival_transformer", "transformer", H_REF, DOT, 1.5, False),
    ("rival_cont_mlp", "continuous MLP", H_REF, DASHDOT, 1.5, False),
    ("rival_table", "table floor", H_REF, FINE, 1.5, False),
    ("learned_ref_ownpool", "MLP own-pool @ ref pin", H_OWN, SOLID, 2.0, True),
    ("learned_1x", "MLP own-pool @ 1x", H_OWN, LONG, 2.0, True),
]
GROUPS = [
    (H_SD, "single design K=5 (DECLARED) + its two charter-C.2 controls"),
    (H_PCA, "pca_tensor, the wave-5 subspace store + its two charter-C.2 controls"),
    (H_REF, "spoke A's rival set @ the 524,288 B REFERENCE pin (montero_combination pool)"),
    (H_OWN, "this hub's own fits, on each rung's OWN pool (AMENDMENT-1)"),
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
        ks.append(int(r))
        mu.append(v.mean())
        lo.append(v.min())
        hi.append(v.max())
        ns.append(v.size)
    return np.asarray(ks), np.asarray(mu), np.asarray(lo), np.asarray(hi), ns


def draw(ax, rep, field):
    for arm, lab, col, ls, lw, primary in SERIES:
        ks, mu, lo, hi, _ = series_xy(rep, arm, field)
        if not ks.size:
            continue
        ax.fill_between(ks, lo, hi, color=col, alpha=0.11, linewidth=0, zorder=2)
        ax.plot(ks, mu, color=col, linestyle=ls, linewidth=lw, zorder=4 if primary else 3,
                marker="o" if primary else None, markersize=5.0,
                markerfacecolor=col, markeredgecolor=SURFACE, markeredgewidth=2.0)
        if primary:
            ax.annotate(" " + lab, (ks[-1], mu[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7.2, color=INK2,
                        va="center", zorder=6)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xlim(1.85, 6.9)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.ladder, encoding="utf-8") as fh:
        rep = json.load(fh)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    nseeds = sorted({n for s in SERIES for n in series_xy(rep, s[0], "mean")[4]})
    nstr = ("n = %d seeds" % nseeds[0] if len(nseeds) == 1
            else "n = %s seeds -- NOT uniform, see the table" % nseeds)
    ncorn = {("k=" + r): sorted(set(rep["rungs"][r]["n_corners_per_seed"].values()))
             for r in RUNGS if rep["rungs"].get(r, {}).get("n_corners_per_seed")}

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    ax = axes[0]
    draw(ax, rep, "mean")
    for r in RUNGS:
        c = rep["rungs"].get(r, {}).get("ceiling_per_corner")
        if not c:
            continue
        ax.plot([int(r) - 0.40, int(r) + 0.40], [c, c], color=INK3, linewidth=1.5,
                linestyle=(0, (4, 2)), zorder=6)
    cs = [rep["rungs"][r].get("ceiling_per_corner") for r in RUNGS
          if rep["rungs"].get(r, {}).get("ceiling_per_corner")]
    if cs:
        ax.annotate("ceiling = the corner's own row count (%s)"
                    % " / ".join(str(c) for c in sorted(set(cs))),
                    (2.0, max(cs)), xytext=(0, 5), textcoords="offset points",
                    fontsize=7.0, color=INK2, ha="left")
    ax.set_xlabel("k  —  number of factors jointly composed in the held-out box")
    ax.set_ylabel("distinct valid NOVEL tuples per corner")
    ax.set_title("A. the composition ladder", fontsize=9.5)

    ax = axes[1]
    ax.axhline(0.0, color=INK3, linewidth=1.2, zorder=1)
    ax.axhline(rep["margin"], color=INK3, linewidth=1.2, linestyle=(0, (5, 3)), zorder=1)
    draw(ax, rep, "delta")
    ax.annotate("declared margin +%.1f" % rep["margin"], (1.9, rep["margin"]),
                xytext=(0, 3), textcoords="offset points", fontsize=7.0, color=INK2)
    ax.set_xlabel("k  —  number of factors jointly composed in the held-out box")
    ax.set_ylabel("paired per-corner margin vs the BEST learned baseline")
    ax.set_title("B. margin against the strongest learner at each rung", fontsize=9.5)

    handles = [plt.Line2D([], [], color=c, linestyle=SOLID, linewidth=2.2, label=l)
               for c, l in GROUPS]
    handles += [
        plt.Line2D([], [], color=INK3, linestyle=SOLID, linewidth=2.0, label="the arm"),
        plt.Line2D([], [], color=INK3, linestyle=DASH, linewidth=1.3,
                   label="control landscape_shuffled / LSTM / 1x"),
        plt.Line2D([], [], color=INK3, linestyle=DOT, linewidth=1.3,
                   label="control random_atoms / transformer"),
        plt.Line2D([], [], color=INK3, linestyle=DASHDOT, linewidth=1.3,
                   label="continuous MLP"),
        plt.Line2D([], [], color=INK3, linestyle=FINE, linewidth=1.3, label="table floor"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, fontsize=7.4,
               loc="lower center", bbox_to_anchor=(0.5, -0.19))

    bl = {}
    for r in RUNGS:
        R = rep["rungs"].get(r, {})
        b = R.get("best_learned_per_seed") or {}
        who = sorted({b[s]["arm"] for s in b if b[s].get("arm")})
        if who:
            bl["k=" + r] = who
    fig.suptitle("Shapes3D composition ladder — how far composition stretches as k grows",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.5, -0.30,
             ("Band = min..max over seeds; line = seed mean; marker on the four primary "
              "series only. %s. Corners per rung per seed: %s.\n"
              "The metric and the +%.1f margin are the harness's "
              "(LADDER-READY.json:estimand); the unit is the CORNER. The ceiling is the "
              "corner's own row count and it CHANGES at k=3 (36 -> 72), so a k=2 absolute "
              "and a k>=3 absolute are not on one scale — the ceiling is drawn for that "
              "reason.\nBest learned baseline per (k, seed) = the learned arm with the "
              "largest mean, fixed before any arm number existed (PREREG §5, widened by "
              "AMENDMENT-3 so the opponent can only get stronger): %s.\n"
              "Hues are the entity's BYTE PIN + POPULATION group; the arm inside a group "
              "is the line style. Palette validated at --pairs all (see the module "
              "docstring). Every arm also appears as its own row in LADDER.md / LADDER.csv.")
             % (nstr, ncorn, rep["margin"], bl),
             ha="center", fontsize=7.0, color=INK2)
    fig.tight_layout()
    p = out / "fig_composition_ladder.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
