"""RECONSTRUCTION of the lost generator for fig_frontier_clean.png (banked
md5 bcc5f32dcd85e01740638c6608f26320)."""
import sys, os
S = "/Users/user/Desktop/CHLU/.claude/scratch/v1-replot-pass"
sys.path.insert(0, S)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frontier_data import load, EPOCHS

D, (hlo, hhi) = load()
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[1].axhspan(hlo, hhi, color="k", alpha=0.12, label="Hopfield band")
for kv in (32, 64, 96, 128):
    d = D[kv]
    ax[0].errorbar(EPOCHS, d["fid"], yerr=d["fid_err"], marker="o", capsize=3, label=f"kv{kv}")
    ax[1].errorbar(EPOCHS, d["gate"], yerr=d["gate_err"], marker="s", capsize=3, label=f"kv{kv}")
ax[0].set(xlabel="train epochs", ylabel="CLU-EBM storage fidelity",
          title="Fidelity vs epochs (corr=0, ne3, 3 seeds)")
ax[1].set(xlabel="train epochs", ylabel="CLU gated accuracy",
          title="Gated acc vs epochs (Hopfield band shaded)")
for a in ax:
    a.grid(alpha=0.3); a.legend(loc="lower right"); a.set_ylim(0, 1.05); a.set_xticks(EPOCHS)
fig.tight_layout()
fig.savefig(os.path.join(S, "orig_renders", "fig_frontier_clean.png"), dpi=120)
print("wrote reconstruction; hop band", hlo, hhi)
