"""Figures + tables for mia-decay-measurement. All numbers re-derived from the JSON."""
import json
import os

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = ".claude/outputs/mia-decay-measurement"
R = json.load(open(os.path.join(OUT, "mia_metrics.json")))
PE = R["per_example"]
M = R["meta"]
A_GRID = M["A_grid"]
TAU = [-np.log(a) for a in A_GRID]
TAU_EVICT = float(np.log(1.0 / M["amp_floor"]))


def col(metric, field="auc"):
    return np.array([PE[metric][f"{a:g}"][field][0] for a in A_GRID]), \
           np.array([PE[metric][f"{a:g}"][field][1] for a in A_GRID])


def crossing(x, y, level, decreasing=True):
    """First x where y crosses `level` (linear interp). None if it never does."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(x) - 1):
        a, b = y[i], y[i + 1]
        if (a >= level >= b) or (a <= level <= b):
            if b == a:
                return float(x[i])
            return float(x[i] + (level - a) * (x[i + 1] - x[i]) / (b - a))
    return None


tab = {}

# ---------------------------------------------------------------- FIGURE 1
ret, ret_s = col("retention")
p_s1, p_s1s = col("paired/s1")
p_s2, _ = col("paired/s2")
p_s4, _ = col("paired/s4")
p_s5, _ = col("paired/s5")
h_s1, _ = col("history/s1")
tpr_s1 = np.array([PE["paired/s1"][f"{a:g}"]["tpr@fpr0.05"][0] for a in A_GRID])

pe_ret = R["postevict"]["retention_mean"][0]
pe_pair = PE["postevict_paired/s1"]["evicted"]["auc"]
pe_hist = PE["postevict_history/s1"]["evicted"]["auc"]
pe_hole = PE["postevict_history/hole"]["evicted"]["auc"]
pe_nliv = PE["postevict_history/n_live"]["evicted"]["auc"]

fig, ax = plt.subplots(figsize=(7.6, 5.0))
tau_x = TAU + [TAU_EVICT + 0.35]
ax.plot(tau_x, list(ret) + [pe_ret], "o-", color="C0", lw=2.2, ms=5,
        label="retention (shipped value-recovery)")
ax.plot(tau_x, list(p_s1) + [pe_pair[0]], "s-", color="C3", lw=2.2, ms=4.5,
        label="MIA AUC — TM-1 query, paired-placement")
ax.plot(tau_x, list(p_s4) + [pe_pair[0]], "^-", color="C1", lw=1.6, ms=4,
        label="MIA AUC — TM-2a white-box (address channel)")
ax.plot(tau_x, list(h_s1) + [pe_hist[0]], "d--", color="C2", lw=1.6, ms=4,
        label="MIA AUC — TM-1 query, history OUT (allocator trace)")
ax.axvline(TAU_EVICT, color="0.4", ls=":", lw=1.4)
ax.text(TAU_EVICT - 0.05, 0.28, "self-evict  (amp < floor 0.05)", rotation=90,
        ha="right", va="bottom", fontsize=8, color="0.35")
ax.axhline(0.5, color="0.7", lw=0.9, ls="-.")
ax.text(0.02, 0.515, "chance (AUC 0.5)", fontsize=7.5, color="0.5")
# TTL laundering line (CLU-with-a-TTL-flag: amp == 1 until expiry)
ax.plot(tau_x, [p_s1[0]] * len(TAU) + [pe_pair[0]], color="k", lw=1.2, ls=(0, (6, 2)),
        label="laundering control: TTL flag (amp$\\equiv$1 till expiry)")
ax.set_xlabel(r"$\tau = \mathrm{leak}\cdot t \;=\; -\ln A$   (amplitude $A=e^{-\tau}$)")
ax.set_ylabel("retention  /  per-example MIA AUC")
ax.set_ylim(-0.04, 1.08)
ax.set_title("CLU store: what an adversary sees as an item decays\n"
             f"{M['n_targets']}targets x {len(M['seeds'])}seeds x {M['n_worlds']} paired worlds, "
             f"{M['n_query']} queries/world", fontsize=10)
sec = ax.secondary_xaxis("top", functions=(lambda t: np.exp(-t), lambda a: -np.log(np.maximum(a, 1e-9))))
sec.set_xlabel("amplitude A", fontsize=9)
ax.legend(fontsize=7.6, loc="center left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_retention_vs_mia.png"), dpi=160)
plt.close(fig)

tab["fig1"] = {
    "tau": TAU, "A": A_GRID, "retention": ret.tolist(),
    "mia_paired_s1": p_s1.tolist(), "mia_paired_s2": p_s2.tolist(),
    "mia_paired_s4": p_s4.tolist(), "mia_paired_s5": p_s5.tolist(),
    "mia_history_s1": h_s1.tolist(), "tpr_fpr05_s1": tpr_s1.tolist(),
    "postevict": {"retention": pe_ret, "paired_s1_auc": pe_pair,
                  "history_s1_auc": pe_hist, "history_hole_auc": pe_hole,
                  "history_nlive_auc": pe_nliv},
    "tau_evict": TAU_EVICT,
}

# ---------------------------------------------------------------- FIGURE 2 (TM-3)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
ax = axes[0]
A = np.array(A_GRID)
tm3 = {}
for sg in M["sigmas"]:
    y = np.array([PE[f"tm3/{sg}"][f"{a:g}"]["auc"][0] for a in A_GRID])
    ys = np.array([PE[f"tm3/{sg}"][f"{a:g}"]["auc"][1] for a in A_GRID])
    ax.errorbar(A, y, yerr=ys, marker="o", ms=3.5, capsize=2, label=f"$\\sigma_{{obs}}$={sg}")
    pred = 0.5 * (1 + np.vectorize(lambda z: math.erf(z / np.sqrt(2)))(0.3935 * A / (1.5 * sg)))
    ax.plot(A, pred, ls=":", color=ax.lines[-1].get_color(), lw=1.0)
    a75 = crossing(A[::-1], y[::-1], 0.75)
    tm3[str(sg)] = {"A": A_GRID, "auc": y.tolist(), "auc_std": ys.tolist(),
                    "A75_measured": a75, "A75_predicted": 2.57 * sg,
                    "prereg_curve": pred.tolist()}
ax.axhline(0.5, color="0.7", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("amplitude A"); ax.set_ylabel("per-example MIA AUC")
ax.set_title("TM-3: resolution-limited white-box probe\n(dotted = pre-registered $\\Phi(0.3935A/1.5\\sigma)$)",
             fontsize=9.5)
ax.legend(fontsize=8)
# laundering control on the same panel
for sg in M["sigmas"]:
    ax.plot([A.min(), A.max()], [tm3[str(sg)]["auc"][0]] * 2, ls="--", lw=0.8, color="0.6")
ax.text(0.06, 1.01, "dashed grey = TTL-flag control (amp$\\equiv$1)", fontsize=7, color="0.4")

ax2 = axes[1]
gap = np.array([PE["stat_s4"][f"{a:g}"]["gap"][0] for a in A_GRID])
dp = np.array([PE["stat_s4"][f"{a:g}"]["dprime"][0] for a in A_GRID])
ax2.loglog(A, gap, "o-", label=r"white-box depth gap $s_4^{IN}-s_4^{OUT}$")
ax2.loglog(A, 0.3935 * A, ls=":", color="k", label=r"pre-registered $0.3935\,A$")
ax2.loglog(A, dp / dp[0], "s-", color="C3", label="effect size $d'$ (normalised)")
ax2.set_xlabel("amplitude A"); ax2.set_ylabel("gap / normalised $d'$")
ax2.set_title("the decaying quantity is the EFFECT SIZE, not the AUC", fontsize=9.5)
ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_resolution.png"), dpi=160)
plt.close(fig)
tab["tm3"] = tm3
tab["s4_gap"] = {"A": A_GRID, "gap": gap.tolist(), "dprime": dp.tolist(),
                 "gap_over_A": (gap / A).tolist()}

# ---------------------------------------------------------------- FIGURE 3 (radius)
rg = M["R_grid"]; ar = M["A_radius"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
rad = {"r_grid": rg, "retention": {}, "mia_s1": {}, "mia_s2": {},
       "R50_retention": {}, "R75_mia": {}}
for j, Aa in enumerate(ar):
    yr = np.array([PE["radius/retention"][f"A{Aa:g}|r{r:g}"]["auc"][0] for r in rg])
    ym = np.array([PE["radius/mia_s1"][f"A{Aa:g}|r{r:g}"]["auc"][0] for r in rg])
    y2 = np.array([PE["radius/mia_s2"][f"A{Aa:g}|r{r:g}"]["auc"][0] for r in rg])
    axes[0].plot(rg, yr, "o-", ms=4, label=f"A={Aa:g}")
    axes[1].plot(rg, ym, "o-", ms=4, label=f"A={Aa:g}")
    rad["retention"][f"{Aa:g}"] = yr.tolist()
    rad["mia_s1"][f"{Aa:g}"] = ym.tolist()
    rad["mia_s2"][f"{Aa:g}"] = y2.tolist()
    rad["R50_retention"][f"{Aa:g}"] = crossing(rg, yr, 0.5)
    rad["R75_mia"][f"{Aa:g}"] = crossing(rg, ym, 0.75)
ttl_ret = R["ttl"]["present1_ret"]; ttl_s1 = R["ttl"]["present1_s1"]
axes[0].plot(rg, ttl_ret, "k--", lw=1.4, label="TTL vector-store (row live)")
axes[0].axvline(R["ttl"]["R_lookup"], color="k", ls=":", lw=0.9)
for a in axes:
    a.axhline(0.5, color="0.75", lw=0.8)
axes[1].axhline(0.75, color="0.75", lw=0.8, ls=":")
axes[0].set_xlabel("adversary/user launch radius r from $c_i$"); axes[0].set_ylabel("retention")
axes[1].set_xlabel("adversary launch radius r from $c_i$"); axes[1].set_ylabel("per-example MIA AUC (TM-1)")
axes[0].set_title("basin shrinks with amplitude: retention vs read radius", fontsize=9.5)
axes[1].set_title("MIA outlives retention at every radius", fontsize=9.5)
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_radius.png"), dpi=160)
plt.close(fig)
rad["ttl_retention"] = ttl_ret
rad["ttl_s1"] = ttl_s1
rad["ttl_absent_s1"] = R["ttl"]["present0_s1"]
rad["ttl_R50"] = crossing(rg, np.array(ttl_ret), 0.5)
tab["radius"] = rad

# ---------------------------------------------------------------- tables
tab["postevict_table"] = {
    k: {"auc": PE[k]["evicted"]["auc"], "tpr@fpr0.05": PE[k]["evicted"]["tpr@fpr0.05"],
        "tpr@fpr0.01": PE[k]["evicted"]["tpr@fpr0.01"],
        "n_examples": PE[k]["evicted"]["n_examples"]}
    for k in PE if k.startswith("postevict_")
}
tab["dump"] = {k: [float(np.max(v)), float(np.mean(v))] for k, v in R["dump"].items()}
tab["decay_law"] = R["decay_law"]
tab["meta"] = M
tab["p4_payload_channel"] = {
    "s1_in_A1": PE["stat_s1"]["1"]["in_mean"][0],
    "s1_in_Amin": PE["stat_s1"][f"{A_GRID[-1]:g}"]["in_mean"][0],
    "s5_in_A1": PE["stat_s5"]["1"]["in_mean"][0],
    "s5_in_Amin": PE["stat_s5"][f"{A_GRID[-1]:g}"]["in_mean"][0],
}
json.dump(tab, open(os.path.join(OUT, "tables.json"), "w"), indent=2)

print(json.dumps({k: tab[k] for k in ("p4_payload_channel", "postevict_table", "dump")}, indent=2))
print("\nfig1 retention:", np.round(ret, 4).tolist())
print("fig1 mia paired s1:", np.round(p_s1, 4).tolist())
print("fig1 mia paired s4:", np.round(p_s4, 4).tolist())
print("fig1 mia history s1:", np.round(h_s1, 4).tolist())
print("tpr@5%:", np.round(tpr_s1, 3).tolist())
print("\nR50 retention:", rad["R50_retention"], "\nR75 mia:", rad["R75_mia"],
      "\nTTL R50:", rad["ttl_R50"])
print("\nA75 tm3:", {k: (v["A75_measured"], v["A75_predicted"]) for k, v in tm3.items()})
print("\ns4 gap/A:", np.round(gap / A, 4).tolist())
