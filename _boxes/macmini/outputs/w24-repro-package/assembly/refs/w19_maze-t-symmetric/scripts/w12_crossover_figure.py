"""w12 -- THE PAPER FIGURE: composition and ledger bytes vs K, two y-axes.

The task file asks for a dual axis by name, so this figure has one.  The repo's own
figure convention (`scripts/w10_build_tables.py`) otherwise prefers small multiples,
and that preference is honoured in the SECOND panel, which is the one that carries the
controls and rank/width -- the quantities that decide whether a row may be claimed.

Everything drawn is read from `CROSSOVER-val.json`, which is written by
`w12_crossover_report.py` from the per-row JSONs.  Nothing is recomputed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("C:/chlu/outputs/w12-single-design-crossover")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8b8a85"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
       "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d2", "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.dpi": 160,
})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="val")
    a = ap.parse_args(argv)
    doc = json.loads((OUT / f"CROSSOVER-{a.condition}.json").read_text(encoding="utf-8"))
    per = doc["per_K"]
    Ks = sorted(int(k) for k in per)
    if not Ks:
        print("no rows -- nothing to draw")
        return 1
    comp = np.array([per[str(k)]["val_composition_mean"] for k in Ks])
    sd = np.array([per[str(k)]["val_composition_sd"] for k in Ks])
    byt = np.array([per[str(k)]["ledger_four_part"]["honest_total_bytes"]
                    for k in Ks], float)
    rw = np.array([per[str(k)]["rank_over_width_max"] for k in Ks])
    ns = sorted({per[str(k)]["n_seeds"] for k in Ks})
    if len(ns) == 1:
        nlab = f"n = {ns[0]} seed" + ("" if ns[0] == 1 else "s")
    else:
        nlab = f"n = {min(ns)}-{max(ns)} seeds, printed per K"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- panel A: the dual axis the task asks for -------------------------
    ax = axes[0]
    ax.errorbar(Ks, comp, yerr=np.nan_to_num(sd), marker="o", markersize=5,
                color=CAT[0], capsize=3, zorder=4, label="val composition")
    v5 = per["5"]["val_composition_mean"] if "5" in per else None
    if v5 is not None:
        ax.axhline(v5, color=INK3, lw=1.2, ls=(0, (4, 2)), zorder=2)
        ax.axhline(v5 - 0.02, color=CAT[2], lw=1.2, ls=(0, (2, 2)), zorder=2)
        ax.axhline(v5 - 0.10, color=CAT[7], lw=1.2, ls=(0, (1, 2)), zorder=2)
        ax.annotate("K=5 reference %.3f" % v5, (Ks[-1], v5), xytext=(-2, 4),
                    textcoords="offset points", ha="right", fontsize=7.2, color=INK2)
        ax.annotate("selection band  -0.02", (Ks[0], v5 - 0.02), xytext=(2, -10),
                    textcoords="offset points", fontsize=7.2, color=CAT[2])
        ax.annotate("crossover  -0.10", (Ks[0], v5 - 0.10), xytext=(2, -10),
                    textcoords="offset points", fontsize=7.2, color=CAT[7])
    ax.set_xlabel("K  (number of wells; the partition gets finer to the right)")
    ax.set_ylabel("val composition (mean valid over 24 corners)", color=CAT[0])
    ax.tick_params(axis="y", labelcolor=CAT[0])
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(Ks, byt / 1e6, marker="s", markersize=4.6, color=CAT[1], lw=1.8,
             ls=(0, (5, 2)), zorder=3, label="four-part ledger")
    ax2.set_ylabel("four-part ledger, MB (summed, never netted)", color=CAT[1])
    ax2.tick_params(axis="y", labelcolor=CAT[1])
    ax2.axhline(524_288 / 1e6, color=INK3, lw=1.0, ls=":", zorder=1)
    ax2.annotate("524,288 B REFERENCE (not a budget here)",
                 (Ks[0], 524_288 / 1e6), xytext=(4, 4), textcoords="offset points",
                 ha="left", fontsize=7.0, color=INK3)
    ks = doc.get("K_star")
    kc = doc.get("K_crossover")
    top = ax.get_ylim()[1]
    if ks:
        ax.axvline(ks, color=CAT[2], lw=1.0, alpha=0.5, zorder=1)
        ax.annotate(f"K* = {ks}", (ks, top), xytext=(-6, -12), ha="right",
                    textcoords="offset points", fontsize=8.5, color=CAT[2],
                    fontweight="bold", va="top")
    if not kc:
        ax.annotate("NO CROSSOVER inside the swept range", (Ks[len(Ks) // 2], top),
                    xytext=(0, -12), textcoords="offset points", fontsize=8.5,
                    color=CAT[7], fontweight="bold", va="top", ha="center")
    if kc:
        ax.axvline(kc, color=CAT[7], lw=1.0, alpha=0.5, zorder=1)
        ax.annotate(f"crossover K = {kc}", (kc, top), xytext=(-6, -26), ha="right",
                    textcoords="offset points", fontsize=8.5, color=CAT[7],
                    fontweight="bold", va="top")
    ax.set_xticks(Ks)
    ax.set_title("composition and ledger bytes vs K   (%s)" % nlab, fontsize=9.8)

    # ---- panel B: the controls and rank/width -----------------------------
    ax3 = axes[1]
    ax3.plot(Ks, rw, marker="^", markersize=5, color=INK, label="rank / well width")
    names = [("landscape_shuffled", CAT[3]), ("random_atoms", CAT[7]),
             ("landscape_removed", INK3),
             ("landscape_shuffled_across_wells", CAT[4])]
    for name, col in names:
        ys = []
        for k in Ks:
            cs = per[str(k)]["controls"]
            vals = [c[name]["valid_mean"] for c in cs.values()
                    if c and name in c]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        if not np.all(np.isnan(ys)):
            ax3.plot(Ks, ys, marker="o", markersize=4, color=col, lw=1.6,
                     ls=(0, (4, 2)), label=name)
    ax3.plot(Ks, comp, marker="o", markersize=5, color=CAT[0], label="CLU (arm)")
    ax3.axhline(0.938, color=CAT[6], lw=1.0, ls=":", zorder=1)
    ax3.annotate("w9 saturation trap sat at rank/d_k = 0.938", (Ks[0], 0.938),
                 xytext=(2, 4), textcoords="offset points", fontsize=7.0, color=CAT[6])
    ax3.set_xticks(Ks)
    ax3.set_xlabel("K")
    ax3.set_ylabel("val composition  /  rank per well width")
    ax3.set_title("the controls, and the saturation tell-tale", fontsize=9.8)
    ax3.grid(True, axis="y", zorder=0)
    ax3.set_axisbelow(True)
    ax3.legend(frameon=False, fontsize=7.2, loc="center left")

    fig.text(0.5, -0.09,
             "LEFT uses two y-axes, as the task file asks. Composition is the existing "
             "24-corner val read; error bars are the seed sd. The ledger is the four "
             "parts summed\n(atoms | covariance+count | address book | coupling/hole "
             "table) and is NEVER netted. RIGHT: a control that does NOT collapse means "
             "the store is DECORATIVE at that K\nand no row may be claimed for it "
             "-- the trap that wave 9's maze arm fell into at rank/d_k = 0.938. "
             "VAL condition: a selection quantity, not a claim.",
             ha="center", fontsize=7.0, color=INK2)
    fig.tight_layout()
    p = OUT / f"fig_crossover_{a.condition}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
