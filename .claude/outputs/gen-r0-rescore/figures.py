"""Figures for gen-r0-rescore. Scratch analysis script."""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(os.environ["RESCORE_OUT"])
S = json.load(open(OUT / "summary.json"))
cells = json.load(open(OUT / "leakage_cells.json"))["cells"]
old_all = json.load(open(OUT / "old_leakage_cells.json"))
seeds = S["seeds"]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) non-degeneracy: old vs corrected cells
o = np.array([c["old"] if c["old"] == c["old"] else np.nan for c in cells], float)
n = np.array([c["corrected"] for c in cells], float)
nan_mask = ~np.isfinite(o)
ax[0].axhline(0, color="k", lw=.5); ax[0].axvline(0, color="k", lw=.5)
ax[0].axhspan(-1, 1, color="tab:green", alpha=.08)
finite = np.isfinite(o)
ax[0].scatter(np.clip(o[finite], -3, 3), n[finite], s=22, c="tab:blue", label="old finite (x clipped to ±3)")
ax[0].scatter(np.full(nan_mask.sum(), 3.4), n[nan_mask], s=26, marker="x", c="tab:red",
              label=f"old = NaN ({int(nan_mask.sum())}/{len(cells)} cells)")
ax[0].set_xlabel("OLD leakage  $1-\\mathrm{acc_{held}}/\\mathrm{acc_{matched}}$  (clipped)")
ax[0].set_ylabel("CORRECTED leakage (balanced-accuracy transfer)")
ax[0].set_title(f"(a) every (well,factor) cell, both arms, seeds {seeds}\n"
                f"corrected range [{np.min(n):+.3f}, {np.max(n):+.3f}], 0 NaN, 0 outside ±1")
ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_ylim(-1.15, 1.15)

# (b) read re-score
r = S["read_rescored"]
labels = ["correlation", "random"]
x = np.arange(2); w = 0.35
for i, (key, nm, col) in enumerate([("old_decoder", "shipped (superseded) decoder", "tab:gray"),
                                    ("v1_decoder", "V1 capacity_schedule decoder", "tab:blue")]):
    m = [r[a][key]["aggregate"]["mean"] for a in labels]
    e = [r[a][key]["aggregate"]["std"] for a in labels]
    ax[1].bar(x + (i - .5) * w, m, w, yerr=e, capsize=3, label=nm, color=col)
ax[1].axhline(r["ceiling_no_store"]["old_decoder"]["mean"], ls="--", c="tab:gray", lw=1)
ax[1].axhline(r["ceiling_no_store"]["v1_decoder"]["mean"], ls="--", c="tab:blue", lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
ax[1].set_ylabel("read aggregate (raw multi-class top-1)")
mo = r["margin_corr_minus_rand"]["old_decoder"]["aggregate"]
mn = r["margin_corr_minus_rand"]["v1_decoder"]["aggregate"]
ax[1].set_title(f"(b) R0's read, old vs new decoder\nmargin {mo['mean']:+.4f} -> {mn['mean']:+.4f}")
ax[1].legend(fontsize=7, loc="upper right"); ax[1].set_ylim(0, 1.05)

# (c) kill rule (i) corrected
g = S["kill_rule_i_corrected_real_features"]
c2 = S["corner_metric_on_store_read"]
names = ["real feat.\nbaseline dec.", "real feat.\nV1 dec.",
         "store read\ncorr · V1", "store read\nrand · V1"]
vals = [g["baseline_decoder"]["balanced_accuracy"], g["v1_decoder"]["balanced_accuracy"],
        c2["correlation"]["v1_decoder"]["balanced_accuracy"], c2["random"]["v1_decoder"]["balanced_accuracy"]]
au = [g["baseline_decoder"]["auroc"], g["v1_decoder"]["auroc"],
      c2["correlation"]["v1_decoder"]["auroc"], c2["random"]["v1_decoder"]["auroc"]]
xx = np.arange(4)
ax[2].bar(xx - .2, [v["mean"] for v in vals], .4, yerr=[v["std"] for v in vals], capsize=3,
          label="balanced accuracy", color="tab:purple")
ax[2].bar(xx + .2, [v["mean"] for v in au], .4, yerr=[v["std"] for v in au], capsize=3,
          label="AUROC", color="tab:orange")
ax[2].axhline(0.5, color="k", ls=":", lw=1); ax[2].text(3.35, .51, "chance", fontsize=7, ha="right")
ax[2].axhline(0.8, color="tab:red", ls="--", lw=1); ax[2].text(3.35, .81, "R3.1 floor 0.8", fontsize=7, ha="right")
ax[2].set_xticks(xx); ax[2].set_xticklabels(names, fontsize=7.5)
ax[2].set_ylabel("cylinder vs not, WITHIN the corner")
ax[2].set_title("(c) kill rule (i), corrected metric\n(n_pos = 72, n_neg = 198/199)")
ax[2].legend(fontsize=7, loc="lower left"); ax[2].set_ylim(0, 1.05)

fig.suptitle(f"gen-r0-rescore — R0 under R20's corrected metrics (seeds {seeds}, main @ e4f3d71)", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "rescore_summary.png", dpi=140, bbox_inches="tight")
print("wrote", OUT / "rescore_summary.png")
