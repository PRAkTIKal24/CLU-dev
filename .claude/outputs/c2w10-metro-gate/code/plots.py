import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
FIG = "/Users/user/Desktop/CHLU/.claude/outputs/c2w10-metro-gate"
p = np.load(os.path.join(OUT, "pairs.npz")); Y = p["y"].astype(float); X = p["X"].astype(float)
g = np.load(os.path.join(OUT, "grid.npz")); t0 = datetime.fromtimestamp(g["t0"][0])
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(OUT, "res", "*.json"))}

# ---- fig1: window ladder
Ls = [250, 500, 1000, 1007, 2000, 5000, 5037, 10000, 14894, 34847]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for sc, mk in (("raw", "o-"), ("std", "s--")):
    xs, ys = [], []
    for L in Ls:
        k = f"knnwin_{L}_{sc}"
        if k in R: xs.append(L); ys.append(R[k]["mae"])
    ax.plot(xs, ys, mk, label=f"k-NN sliding window, k=5, {sc}")
xs = [L for L in (1007, 5037, 14894) if f"knnk_{L}_10_raw" in R]
ax.plot(xs, [R[f"knnk_{L}_10_raw"]["mae"] for L in xs], "^:", color="tab:red", label="k-NN, k=10, raw")
for nm, c in (("gbdt_tuned", "k"), ("gbdt", "0.35"), ("gru_big", "tab:purple"), ("mlp", "tab:brown"), ("ridge_batch", "tab:gray")):
    if nm in R: ax.axhline(R[nm]["mae"], color=c, ls="-", lw=1.2, alpha=.8, label=f"{nm} (strong)")
ax.axhline(float(np.abs(X[:, 25] - Y).mean()), color="tab:green", ls=":", lw=1.5,
           label="seasonal-naive t-168h")
ax.axhline(float(np.abs(X[:, 0] - Y).mean()), color="tab:orange", ls=":", lw=1.5,
           label="persistence / seasonal-naive t-24h")
for L, lb in ((5037, "0.634 MiB"), (1007, "133 kB"), (14894, "CLU d=12 1.875 MiB")):
    ax.axvline(L, color="0.8", lw=1); ax.text(L, ax.get_ylim()[1] * .995, lb, rotation=90,
                                              fontsize=6, va="top", ha="right", color="0.4")
ax.set_xscale("log"); ax.set_xlabel("exemplar store size L (pairs)"); ax.set_ylabel("prequential MAE")
ax.set_title("Metro, 24-h horizon, hidden clock: exemplar-store byte frontier (OURS)")
ax.legend(fontsize=6.5, ncol=2); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_window_ladder.png"), dpi=160); plt.close(fig)

# ---- fig2: drift-magnitude heatmaps
D = np.load(os.path.join(OUT, "driftmaps.npz"))
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
for a, key, ttl in ((axs[0], "D_primary", "day windows, TV on binned y"),
                    (axs[1], "W_season", "week windows, TV on joint(y,temp,clouds)")):
    TV = D[f"TV_{key}"]
    im = a.imshow(TV, cmap="magma", aspect="auto")
    a.set_title(f"Webb drift magnitude: {ttl}", fontsize=9)
    a.set_xlabel("window index (time order)"); a.set_ylabel("window index (time order)")
    plt.colorbar(im, ax=a, fraction=.046)
fig.suptitle("Metro drift map -- THIS ANNOTATION IS OURS, NOT THE LITERATURE'S", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_drift_magnitude.png"), dpi=160); plt.close(fig)

# ---- fig3: regime timeline (the revisit schedule)
fig, axs = plt.subplots(3, 1, figsize=(11, 5.4), sharex=False)
for a, key, ttl in ((axs[0], "D_primary", "D_primary (K=2, silhouette): day-type"),
                    (axs[1], "D_fine4", "D_fine4 (K=4 forced)"),
                    (axs[2], "D_joint", "D_joint (K=3): season/weather")):
    lab = D[f"lab_{key}"]; keys = D[f"key_{key}"]
    dates = [datetime.fromordinal(int(k)) for k in keys]
    a.scatter(dates, lab, s=2, c=lab, cmap="tab10")
    a.set_ylabel("regime"); a.set_title(ttl, fontsize=8); a.set_yticks(sorted(set(lab.tolist())))
fig.suptitle("Metro regime revisit schedule (ours)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_regime_timeline.png"), dpi=160); plt.close(fig)

# ---- fig4: prequential curves
def roll(pr, w=720):
    e = np.abs(pr - Y); c = np.concatenate([[0], np.cumsum(e)])
    return (c[w:] - c[:-w]) / w
fig, ax = plt.subplots(figsize=(9.5, 4.2))
show = [("persistence", X[:, 0]), ("seasonal_naive_168", X[:, 25])]
for nm in ("knnwin_5037_raw", "knnwin_1007_raw", "gbdt_tuned", "gru_big", "mlp"):
    f = os.path.join(OUT, "res", nm + ".npz")
    if os.path.exists(f): show.append((nm, np.load(f)["pred"].astype(float)))
for nm, pr in show:
    ax.plot(np.arange(len(roll(pr))), roll(pr), lw=.9, label=nm)
ax.set_xlabel("pair index"); ax.set_ylabel("rolling MAE (window 720)")
ax.set_title("Metro prequential MAE, 24-h horizon, hidden clock"); ax.legend(fontsize=7, ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_prequential.png"), dpi=160); plt.close(fig)
print("figures written")
