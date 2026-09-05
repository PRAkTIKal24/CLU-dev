"""Two figures for the c3-trackb-tripwire report."""
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

O = Path("/Users/user/Desktop/CHLU/.claude/outputs/c3-trackb-tripwire")
BUDGET = 1_966_080


def load(fn):
    p = O / fn
    return [json.loads(l) for l in open(p)] if p.exists() else []


def fig1():
    rows = load("arms.jsonl") + load("arms_distfix.jsonl")
    lad = collections.defaultdict(lambda: -9)
    for r in rows:
        sel = "k-means" if "kmeans" in r["tag"] else "random"
        lad[(r["window"], r["scaling"], sel, r["L"], r["in_budget"])] = max(
            lad[(r["window"], r["scaling"], sel, r["L"], r["in_budget"])],
            r["median_nse_447"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, win in zip(axes, (30, 365)):
        for sc, col in (("perbasin", "tab:blue"), ("global", "tab:orange"),
                        ("raw", "tab:green")):
            for sel, ls in (("k-means", "-"), ("random", "--")):
                pts = sorted((L, v, inb) for (w, s, se, L, inb), v
                             in lad.items() if w == win and s == sc
                             and se == sel)
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls,
                        color=col, marker="o", ms=4,
                        label=f"{sc} / {sel}")
                for L, v, inb in pts:
                    if not inb:
                        ax.plot([L], [v], "x", color="k", ms=7)
        ax.set_xscale("log")
        tk = [250, 500, 1000, 2000, 5000] + ([2761] if win == 30 else [265])
        ax.set_xticks(sorted(tk))
        ax.set_xticklabels([f"{t:,}" for t in sorted(tk)], fontsize=7,
                           rotation=45)
        ax.minorticks_off()
        ax.axhline(0.7580, color="crimson", lw=2)
        ax.axhline(0.6028, color="grey", lw=1.2, ls=":")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_title(f"{win}-day window  (at-budget L = "
                     f"{'2,761' if win == 30 else '265'})")
        ax.set_xlabel("store size L (exemplars)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("median NSE, 447 basins")
    axes[0].text(300, 0.775, "LSTM(static) ensemble 0.758", color="crimson",
                 fontsize=8)
    axes[0].text(300, 0.615, "SAC-SMA 0.603", color="grey", fontsize=8)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("CAMELS-US criterion-4 tripwire: exemplar-store ladder "
                 "(x = over budget)", fontsize=11)
    fig.tight_layout()
    fig.savefig(O / "fig1_camels_ladder.png", dpi=150)
    print("wrote fig1")


def fig2():
    rows = [r for r in load("ncmapss_arms.jsonl")
            if r.get("arm") in ("knn", "traj_similarity")]
    lad = collections.defaultdict(lambda: 9e9)
    for r in rows:
        key = (r.get("feats", r.get("rep")),
               r.get("L", r.get("n_library")), r["in_budget"])
        lad[key] = min(lad[key], r["rmse"])
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for f, col in (("resid", "tab:blue"), ("w_xs", "tab:red"),
                   ("w_xs_resid", "tab:purple"),
                   ("resid_cycle", "tab:green"),
                   ("hi14", "tab:brown"), ("hi1d", "tab:olive")):
        pts = sorted((L, v) for (ff, L, inb), v in lad.items() if ff == f)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=4,
                    color=col, label=f)
    for y, lab, c in ((4.14, "CNN hybrid 4.14", "crimson"),
                      (4.95, "CNN data-driven 4.95", "darkorange"),
                      (7.89, "FNN data-driven 7.89", "grey"),
                      (12.39, "affine-in-cycle 12.39", "black"),
                      (19.90, "mean-RUL 19.90", "black")):
        ax.axhline(y, color=c, lw=1.2, ls="--" if y > 8 else "-")
        ax.text(260, y + 0.25, lab, fontsize=7, color=c)
    ax.set_xscale("log")
    ax.set_xlabel("store size L (exemplars / library segments)")
    ax.set_ylabel("per-sample RMSE [cycles], DS02 test units {11,14,15}")
    ax.set_title("N-CMAPSS DS02 criterion-4 tripwire + the missing "
                 "criterion-2 rows", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(O / "fig2_ncmapss_ladder.png", dpi=150)
    print("wrote fig2")


if __name__ == "__main__":
    fig1()
    fig2()
