"""⭐ Two panels: the laundering control (B vs MS) and the noise trap (distinct vs accuracy)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent
RUNS = R / "runs" / "shapes3d_recon"
S = [0, 1, 2]
swp = {s: json.load(open(RUNS / f"seed_{s}" / f"sweep_s{s}.json"))["sweep"] for s in S}
sco = {s: json.load(open(RUNS / f"seed_{s}" / f"scores_s{s}.json")) for s in S}
COL = {"argmax": "k", "sample": "crimson", "margin": "tab:orange", "topk": "tab:blue"}

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
a = ax[0]
for i, r in enumerate(swp[0]):
    b = np.mean([swp[s][i]["sets"]["B"]["n_distinct"] / 72 for s in S])
    m = np.mean([swp[s][i]["sets"]["MS"]["n_distinct"] / swp[s][i]["sets"]["MS"]["n_requests"]
                 for s in S])
    a.scatter(m, b, c=COL[r["rule"]], s=26, alpha=0.8,
              label=r["rule"] if all(swp[0][j]["rule"] != r["rule"] for j in range(i)) else None)
a.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
a.set_xlabel("distinct fraction — MS (matched-seen, SAME corner)")
a.set_ylabel("distinct fraction — B (held-out)")
a.set_title("⭐ the laundering control: every knob moves BOTH\n(a point far above the diagonal = a B-specific artefact)")
a.legend(fontsize=7)

a = ax[1]
for key in sco[0]["configs"]:
    c = [sco[s]["configs"][key] for s in S]
    x = np.mean([v["n_distinct"]["B"] for v in c])
    y = np.mean([v["in_corner_discrimination"]["balanced_accuracy"] for v in c])
    e = np.std([v["in_corner_discrimination"]["balanced_accuracy"] for v in c])
    rule = key.split("|")[0]
    a.errorbar(x, y, yerr=e, fmt="o", c=COL[rule], capsize=3)
    a.annotate(key.replace("|margin=0|k=1", "").replace("|", " "), (x, y),
               textcoords="offset points", xytext=(6, 5), fontsize=6.5)
a.axhline(sco[0]["pass_target"]["balacc"], ls="-.", c="crimson", lw=1,
          label="knn_proto/proto target 0.8817")
a.axvline(sco[0]["pass_target"]["distinct_floor"], ls="--", c="k", lw=1, label="PASS floor 60/72")
a.axhline(0.5, ls="-", c="grey", lw=0.8, label="chance")
a.set_xlabel("distinct codes on B (of 72)")
a.set_ylabel("in-corner shape balanced accuracy")
a.set_title("⛔ both numbers or neither: the PASS corner is top-right\n(⛔ nothing reaches it)")
a.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(R / "collapse.png", dpi=150)
print("wrote", R / "collapse.png")
