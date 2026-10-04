"""shapes sweep k figure."""

from __future__ import annotations
from nclu.paths import run_dir
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(str(run_dir("shapes3d/k_sweep_seeds")))
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8b8a85"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PART_SPEC = (
    ("p1_atom_bytes", "atoms (float32)", "#1b4f8f"),
    ("p2_covariance_plus_count_bytes", "covariance + count (float64)", "#2a78d6"),
    ("p3_address_book_bytes", "address book", "#7fb2e8"),
    ("p4_coupling_hole_table_bytes", "coupling / hole table", "#c8ddf5"),
)
SEED_MARK = {0: "o", 1: "s", 2: "^"}
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": "#d8d7d2",
        "grid.color": "#e8e7e2",
        "grid.linewidth": 0.8,
        "font.size": 9,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 160,
        "pdf.fonttype": 42,
        "svg.hashsalt": "shapes3d/k_sweep_seeds",
    }
)
PDF_META = {
    "Title": "CROSSOVER-3SEED",
    "Author": "NCLU an earlier study",
    "Subject": "val K-sweep, n = 3 seeds",
    "Creator": "matplotlib",
    "Producer": "matplotlib",
    "CreationDate": None,
}
PNG_META = {"Software": None}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    doc = json.loads((out / "CROSSOVER-3SEED.json").read_text(encoding="utf-8"))
    per = doc["per_K"]
    Ks = sorted((int(k) for k in per))
    if not Ks:
        print("no rows -- nothing to draw")
        return 1
    u = doc["u_shape"]
    seeds = sorted(u["seeds"])
    comp = np.array([per[str(k)]["val_composition"]["mean"] for k in Ks])
    clo = np.array([per[str(k)]["val_composition"]["min"] for k in Ks])
    chi = np.array([per[str(k)]["val_composition"]["max"] for k in Ks])
    tot = np.array([per[str(k)]["honest_total_bytes"]["mean"] for k in Ks])
    tlo = np.array([per[str(k)]["honest_total_bytes"]["min"] for k in Ks])
    thi = np.array([per[str(k)]["honest_total_bytes"]["max"] for k in Ks])
    ns = sorted({per[str(k)]["n_seeds"] for k in Ks})
    nlab = (
        f"n = {ns[0]} seeds {seeds}"
        if len(ns) == 1
        else f"n = {min(ns)}-{max(ns)} seeds, printed per K"
    )
    fig = plt.figure(figsize=(15.4, 5.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.1, 1.0], wspace=0.34, top=0.84, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(Ks, clo, chi, color=CAT[0], alpha=0.16, lw=0, zorder=2)
    ax.plot(
        Ks,
        comp,
        marker="o",
        markersize=5,
        color=CAT[0],
        zorder=4,
        label="val composition (mean of 3 seeds)",
    )
    for s in seeds:
        ys = [
            per[str(k)]["per_seed"][str(s)]["val_composition"]
            if str(s) in per[str(k)]["per_seed"]
            else np.nan
            for k in Ks
        ]
        ax.plot(
            Ks,
            ys,
            ls="none",
            marker=SEED_MARK.get(s, "x"),
            markersize=3.6,
            mfc="none",
            mew=0.9,
            color=CAT[0],
            alpha=0.85,
            zorder=5,
            label=f"seed {s}",
        )
    v5 = per["5"]["val_composition"]["mean"] if "5" in per else None
    if v5 is not None:
        ax.axhline(v5, color=INK3, lw=1.1, ls=(0, (4, 2)), zorder=1)
        ax.axhline(v5 - 0.02, color=CAT[2], lw=1.1, ls=(0, (2, 2)), zorder=1)
        ax.axhline(v5 - 0.1, color=CAT[7], lw=1.1, ls=(0, (1, 2)), zorder=1)
        ax.annotate(
            "K=5 reference %.3f" % v5,
            (Ks[-1], v5),
            xytext=(-2, 4),
            textcoords="offset points",
            ha="right",
            fontsize=7.0,
            color=INK2,
        )
        ax.annotate(
            "selection band  -0.02",
            (Ks[0], v5 - 0.02),
            xytext=(2, -10),
            textcoords="offset points",
            fontsize=7.0,
            color=CAT[2],
        )
        ax.annotate(
            "crossover  -0.10",
            (Ks[0], v5 - 0.1),
            xytext=(2, -10),
            textcoords="offset points",
            fontsize=7.0,
            color=CAT[7],
        )
    ax.set_xlabel("K  (wells; the partition gets finer to the right)")
    ax.set_ylabel("val composition (mean valid, 24 corners)", color=CAT[0])
    ax.tick_params(axis="y", labelcolor=CAT[0])
    ax.set_xticks(Ks)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.fill_between(Ks, tlo / 1000000.0, thi / 1000000.0, color=CAT[1], alpha=0.16, lw=0, zorder=1)
    ax2.plot(
        Ks,
        tot / 1000000.0,
        marker="s",
        markersize=4.4,
        color=CAT[1],
        lw=1.8,
        ls=(0, (5, 2)),
        zorder=3,
        label="honest four-part total",
    )
    ax2.set_ylabel("honest four-part total, MB (summed, NEVER netted)", color=CAT[1])
    ax2.tick_params(axis="y", labelcolor=CAT[1])
    ax2.axhline(524288 / 1000000.0, color=INK3, lw=1.0, ls=":", zorder=0)
    ax2.annotate(
        "524,288 B REFERENCE column (not a budget here)",
        (Ks[-1], 524288 / 1000000.0),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        fontsize=6.8,
        color=INK3,
    )
    ks, kc = (doc.get("K_star"), doc.get("K_crossover"))
    top = ax.get_ylim()[1]
    if ks:
        ax.axvline(ks, color=CAT[2], lw=1.0, alpha=0.5, zorder=1)
        ax.annotate(
            f"K* = {ks}",
            (ks, top),
            xytext=(-6, -10),
            ha="right",
            textcoords="offset points",
            fontsize=8.2,
            color=CAT[2],
            fontweight="bold",
            va="top",
        )
    if not kc:
        ax.annotate(
            "NO CROSSOVER inside the swept range",
            (Ks[max(0, len(Ks) // 2 - 1)], top),
            xytext=(0, -24),
            textcoords="offset points",
            fontsize=8.2,
            color=CAT[7],
            fontweight="bold",
            va="top",
            ha="center",
        )
    else:
        ax.axvline(kc, color=CAT[7], lw=1.0, alpha=0.5, zorder=1)
        ax.annotate(
            f"crossover K = {kc}",
            (kc, top),
            xytext=(-6, -26),
            ha="right",
            textcoords="offset points",
            fontsize=8.2,
            color=CAT[7],
            fontweight="bold",
            va="top",
        )
    ax.set_title("A. the honest four-part total BESIDE composition\n%s" % nlab, fontsize=9.2, pad=7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        frameon=False,
        fontsize=6.4,
        ncol=2,
        loc="center left",
        bbox_to_anchor=(0.01, 0.2),
        handlelength=1.6,
        columnspacing=1.0,
        labelspacing=0.32,
    )
    axb = fig.add_subplot(gs[0, 1])
    idx = np.arange(len(Ks), dtype=float)
    bottom = np.zeros(len(Ks))
    for key, lab, col in PART_SPEC:
        v = np.array([per[str(k)]["parts_across_seeds"][key]["mean"] for k in Ks]) / 1000000.0
        axb.bar(
            idx,
            v,
            bottom=bottom,
            width=0.66,
            color=col,
            edgecolor=SURFACE,
            linewidth=0.6,
            label=lab,
            zorder=2,
        )
        bottom = bottom + v
    seed_tot = {}
    for s in seeds:
        ys = [
            per[str(k)]["per_seed"][str(s)]["footprint_four_part"]["honest_total_bytes"] / 1000000.0
            if str(s) in per[str(k)]["per_seed"]
            else np.nan
            for k in Ks
        ]
        seed_tot[s] = ys
        axb.plot(
            idx,
            ys,
            ls="none",
            marker=SEED_MARK.get(s, "x"),
            markersize=4.4,
            mfc="none",
            mew=1.1,
            color=INK,
            zorder=5,
            label=f"seed {s} total",
        )
    axb.set_xticks(idx)
    axb.set_xticklabels([str(k) for k in Ks])
    axb.set_xlabel("K  (wells)")
    axb.set_ylabel("honest four-part total, MB")
    verdict = u["S3_verdict"]
    axb.set_title(
        f"B. the footprint in four parts, stacked -- never netted\nU-shape, min near K=24: {verdict}   per-seed minima K = "
        + ", ".join((str(u["per_seed_minima"][str(s)]) for s in seeds)),
        fontsize=9.0,
    )
    axb.grid(True, axis="y", zorder=0)
    axb.set_axisbelow(True)
    axb.legend(frameon=False, fontsize=6.4, loc="upper right", handlelength=1.4, labelspacing=0.32)
    tail = [k for k in Ks if k >= 16]
    if len(tail) >= 3:
        axi = axb.inset_axes([0.4, 0.335, 0.575, 0.375], zorder=9)
        axi.set_facecolor(SURFACE)
        axi.patch.set_alpha(1.0)
        for sp in axi.spines.values():
            sp.set_edgecolor("#d8d7d2")
        ti = np.array([per[str(k)]["honest_total_bytes"]["mean"] for k in tail]) / 1000.0
        lo = np.array([per[str(k)]["honest_total_bytes"]["min"] for k in tail]) / 1000.0
        hi = np.array([per[str(k)]["honest_total_bytes"]["max"] for k in tail]) / 1000.0
        axi.fill_between(tail, lo, hi, color=CAT[1], alpha=0.2, lw=0, zorder=1)
        axi.plot(tail, ti, marker="s", markersize=3.6, color=CAT[1], lw=1.5, zorder=3)
        for s in seeds:
            axi.plot(
                tail,
                [seed_tot[s][Ks.index(k)] * 1000.0 for k in tail],
                ls="none",
                marker=SEED_MARK.get(s, "x"),
                markersize=3.4,
                mfc="none",
                mew=0.9,
                color=INK,
                zorder=4,
            )
        kms = sorted({u["per_seed"][str(s)]["K_argmin"] for s in seeds})
        for km in kms:
            if km in tail:
                axi.axvline(km, color=CAT[7], lw=1.0, alpha=0.6, zorder=2)
        lab = (
            f"min: K = {kms[0]} on all {len(seeds)} seeds"
            if len(kms) == 1
            else "per-seed minima K = "
            + ", ".join((str(u["per_seed_minima"][str(s)]) for s in seeds))
        )
        axi.annotate(
            lab,
            (kms[0], axi.get_ylim()[1]),
            xytext=(4, -3),
            textcoords="offset points",
            fontsize=6.2,
            color=CAT[7],
            fontweight="bold",
            va="top",
            ha="left",
        )
        axi.set_xticks(tail)
        axi.tick_params(labelsize=6.0, length=2.5, pad=1.5)
        for lbl in axi.get_xticklabels():
            lbl.set_bbox(dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
        axi.set_title(
            "zoom on the flat tail -- kB, seed band shaded",
            fontsize=6.4,
            fontweight="normal",
            color=INK2,
            pad=2.5,
        )
        axi.grid(True, axis="y", zorder=0)
        axi.set_axisbelow(True)
        axi.spines["top"].set_visible(True)
        axi.spines["right"].set_visible(True)
    axc = fig.add_subplot(gs[0, 2])
    rw = np.array(
        [
            max(
                (
                    per[str(k)]["per_seed"][str(s)]["rank_over_width_max"]
                    for s in per[str(k)]["seeds"]
                )
            )
            for k in Ks
        ]
    )
    axc.plot(Ks, rw, marker="^", markersize=4.6, color=INK, label="rank / well width")
    axc.fill_between(Ks, clo, chi, color=CAT[0], alpha=0.16, lw=0, zorder=2)
    axc.plot(
        Ks,
        comp,
        marker="o",
        markersize=4.6,
        color=CAT[0],
        zorder=4,
        label="CLU arm (seed band shaded)",
    )
    names = [
        ("landscape_shuffled", CAT[3]),
        ("random_atoms", CAT[7]),
        ("landscape_removed", INK3),
        ("landscape_shuffled_across_wells", CAT[4]),
    ]
    for name, col in names:
        ys = [
            per[str(k)]["control_gate"][name]["worst_over_seeds_and_corners"]
            if name in per[str(k)]["control_gate"]
            else np.nan
            for k in Ks
        ]
        if not np.all(np.isnan(ys)):
            axc.plot(
                Ks,
                ys,
                marker="o",
                markersize=3.4,
                color=col,
                lw=1.5,
                ls=(0, (4, 2)),
                label=f"{name} (worst)",
            )
    axc.axhline(0.938, color=CAT[6], lw=1.0, ls=":", zorder=1)
    axc.annotate(
        "saturation trap sat at rank/d_k = 0.938",
        (Ks[0], 0.938),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=6.8,
        color=CAT[6],
    )
    axc.set_xticks(Ks)
    axc.set_xlabel("K  (wells)")
    axc.set_ylabel("val composition  /  rank per well width")
    axc.set_title(
        "C. composition on 3 seeds with the band,\nthe landscape controls, the saturation tell-tale",
        fontsize=9.2,
        pad=7,
    )
    axc.grid(True, axis="y", zorder=0)
    axc.set_axisbelow(True)
    axc.legend(frameon=False, fontsize=6.4, loc="center left", handlelength=1.4, labelspacing=0.32)
    fig.text(
        0.5,
        -0.085,
        "BYTE CONVENTION (the same one the JSON carries): honest four-part total, uncompressed raw array bytes, summed and NEVER netted -- (1) atoms = sum_k modes_k.size x 4 B, float32; (2) covariance + count = sum_k stat_floats_k x 8 B, float64,\nthe precision actually retained, plus 11,488 B of count table; (3) address book = TensorAddress.byte_footprint['address_bytes']; (4) coupling/hole table = 5,288 + 1,322 = 6,610 B. No entropy coding, no deduplication, no part subtracted from any other.\nThe 524,288 B pin is a REFERENCE column for continuity with round 5-8 and is NOT a budget in this sweep. Condition VAL -- a free selection quantity, SMOKE_NOT_A_CLAIM, zero runs. Composition is the 24-corner val read; the band is the seed min-max over "
        + ", ".join((f"seed {s}" for s in seeds))
        + ".\nA control that does not collapse means the store is DECORATIVE at that K and no row may be claimed for it -- the trap round 9's maze arm fell into at rank/d_k = 0.938.",
        ha="center",
        fontsize=6.8,
        color=INK2,
    )
    pdf = out / "fig_crossover_3seed.pdf"
    png = out / "fig_crossover_3seed.png"
    fig.savefig(pdf, bbox_inches="tight", metadata=PDF_META)
    fig.savefig(png, bbox_inches="tight", metadata=PNG_META)
    plt.close(fig)
    print(f"wrote {pdf}  sha256 {_sha(pdf)}")
    print(f"wrote {png}  sha256 {_sha(png)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
