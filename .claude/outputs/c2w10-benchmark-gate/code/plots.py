import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/user/Desktop/CHLU/.claude/outputs/c2w10-benchmark-gate"
M = json.load(open("metrics.json"))
S = json.load(open("structure.json"))
CP = S["change_points"]

# ---- Fig 1: prequential curves (window 1000) ----
fig, ax = plt.subplots(figsize=(11, 5))
show = [("arf100_s1", "ARF (100 trees)", "k", 1.6),
        ("samknn_1000_std", "SAM-kNN L=1000 (std)", "tab:red", 1.2),
        ("samknn_5000_std", "SAM-kNN L=5000 (std, published default)", "tab:orange", 1.2),
        ("knns_1000_std", "kNN_S L=1000 (std)", "tab:blue", 1.0),
        ("knns_5000_std", "kNN_S L=5000 (std)", "tab:cyan", 1.0),
        ("knns_14782_std", "kNN_S L=14782 (std, CLU-byte-matched)", "tab:purple", 1.0),
        ("nochange", "No-Change (persistence)", "0.6", 1.0)]
for tag, lab, c, lw in show:
    if not os.path.exists(f"preds/{tag}_curve.npy"):
        continue
    cur = np.load(f"preds/{tag}_curve.npy") * 100
    x = np.arange(len(cur)) + 1000
    k = 201
    sm = np.convolve(cur, np.ones(k) / k, mode="valid")
    ax.plot(x[k // 2:k // 2 + len(sm)], sm, color=c, lw=lw, label=lab)
for cp in CP:
    ax.axvline(cp, color="g", ls="--", lw=1)
ax.text(CP[0], 96, " cp1 26,568", color="g", fontsize=8, va="top")
ax.text(CP[1], 96, " cp2 53,364", color="g", fontsize=8, va="top")
for i, (a, b) in enumerate(S["cycles"]):
    ax.text((a + b) / 2, 22, f"cycle {i+1}\n{'20→40°C' if i!=1 else '40→20°C'}",
            ha="center", fontsize=9, color="g")
ax.set_xlabel("stream position"); ax.set_ylabel("prequential accuracy % (window 1000, smoothed 201)")
ax.set_title("INSECTS incremental-reoccurring (balanced) — B1/B2 arms")
ax.legend(fontsize=8, loc="lower right", ncol=2); ax.grid(alpha=.3); ax.set_ylim(15, 100)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_prequential_curves.png", dpi=150); plt.close(fig)

# ---- Fig 2: the window ladder ----
fig, ax = plt.subplots(figsize=(7, 4.6))
Ls, accs = [], []
for t in M:
    if t.startswith("knns_") and t.endswith("_std") and not t.startswith("ABR_"):
        Ls.append(int(t.split("_")[1])); accs.append(M[t]["acc"])
o = np.argsort(Ls); Ls = np.array(Ls)[o]; accs = np.array(accs)[o]
ax.plot(Ls, accs, "o-", color="tab:blue", label="kNN_S (sliding window, std)")
for L, a in zip(Ls, accs):
    ax.annotate(f"{a:.1f}", (L, a), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
arf = np.mean([M[t]["acc"] for t in M if t.startswith("arf100_s")])
ax.axhline(arf, color="k", ls="-", lw=1.4, label=f"ARF 100 trees = {arf:.2f}")
ax.axhline(arf - 2.0, color="r", ls=":", lw=1.4, label="criterion-4 tripwire (ARF − 2.0)")
ax.axhline(77.13, color="0.5", ls="--", lw=1, label="Souza published ARF = 77.13")
for t, m_, c in [("samknn_1000_std", "s", "tab:red"), ("samknn_5000_std", "D", "tab:orange")]:
    ax.plot([int(t.split("_")[1])], [M[t]["acc"]], m_, color=c, ms=9,
            label=f"SAM-kNN L_max={t.split('_')[1]} = {M[t]['acc']:.2f}")
ax.set_xscale("log"); ax.set_xlabel("stored exemplars L (log)"); ax.set_ylabel("prequential accuracy %")
ax.set_title("Accuracy DECREASES with store size: recency is the hidden regime variable")
ax.legend(fontsize=7.5, loc="lower left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_window_ladder.png", dpi=150); plt.close(fig)
print("figures written")
