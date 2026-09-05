"""Figures for c2w10-i2 (banked artifacts only)."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/user/Desktop/CHLU/.claude/outputs/c2w10-i2")
from i2_analysis import SRC, OUT, analyse_seed  # noqa: E402

tel = json.load(open(SRC))
per = [analyse_seed(s) for s in tel["per_seed"]]
V = json.load(open(f"{OUT}/I2-VERDICT.json"))

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

for j, (s, p) in enumerate(zip(tel["per_seed"], per)):
    # --- row 0 col j: the WHOLE curve, raw vs netted (all items, median overlaid)
    a = ax[0, j]
    curves = s["depth_curves_raw_and_netted"]
    for iid, cv in curves.items():
        a.plot(cv["chunk"], cv["depth_raw"], color="C0", alpha=0.15, lw=0.6)
        a.plot(cv["chunk"], np.array(cv["depth_netted"]), color="C3", alpha=0.15, lw=0.6)
    a.set_yscale("log")
    a.set_title(f"seed {s['seed']}: depth curves (blue raw, red netted)\n"
                f"raw median log-drop {p['validity_E_netted']['median_raw_total_log_drop']:.3f} "
                f"nats vs netted {p['validity_E_netted']['median_netted_total_log_drop']:.1e}")
    a.set_xlabel("chunk")
    a.set_ylabel("depth (log)")

# --- row 1 col 0: U vs E_netted
a = ax[1, 0]
for p in per:
    U = [r["U"] for r in p["_rows"]]
    E = [r["E_net"] for r in p["_rows"]]
    a.scatter(U, E, s=12, alpha=0.7,
              label=f"seed {p['seed']}: rho={p['rho_U_E']['rho']:+.3f}")
a.set_xlabel("U = read_hits (item-id keyed)")
a.set_ylabel("E netted (nats/chunk)")
a.set_title("I2-c: U vs netted erosion rate\n(note the 1e-8 y-scale = float32 round-off)")
a.legend(fontsize=8)

# --- row 1 col 1: U vs depth_netted
a = ax[1, 1]
for p in per:
    U = [r["U"] for r in p["_rows"]]
    D = [r["depth_net_last"] for r in p["_rows"]]
    a.scatter(U, D, s=12, alpha=0.7,
              label=f"seed {p['seed']}: rho={p['rho_U_depth']['rho']:+.3f}")
a.set_xlabel("U = read_hits")
a.set_ylabel("netted depth at the measurement point")
a.set_title("I2-d: U vs netted depth")
a.legend(fontsize=8)

# --- row 1 col 2: forest plot of the two legs with the registered thresholds
a = ax[1, 2]
ys, labs = [], []
y = 0
for key, name in (("rho_U_E_by_seed", "rho(U,E)"), ("rho_U_depth_by_seed", "rho(U,depth)")):
    for sd, r in V[key].items():
        a.plot([r["lo_2se"], r["hi_2se"]], [y, y], color="k", lw=1.5)
        a.plot([r["rho"]], [y], "o", color="C0" if "E_by" in key else "C1")
        labs.append(f"{name} s{sd}")
        ys.append(y)
        y += 1
a.axvline(-0.10, color="C3", ls="--", lw=1, label="leg 1 bar: lower > -0.10")
a.axvline(+0.30, color="C2", ls="--", lw=1, label="leg 2 bar: lower >= +0.30")
a.axvline(-0.20, color="C4", ls=":", lw=1, label="CONFIRM bar: upper < -0.20")
a.axvline(0.0, color="grey", lw=0.6)
a.set_yticks(ys)
a.set_yticklabels(labs, fontsize=8)
a.set_xlabel("Spearman rho with 2-SE Fisher-z bounds")
a.set_title(f"registered legs -> branch = {V['branch']}\n"
            f"detectable |rho| ~ 0.26-0.27 at n = 57-60")
a.legend(fontsize=7, loc="lower right")

fig.tight_layout()
fig.savefig(f"{OUT}/i2_figs.png", dpi=140)
print("wrote", f"{OUT}/i2_figs.png")
