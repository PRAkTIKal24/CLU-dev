import json, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT=".claude/outputs/mia-decay-measurement"
R=json.load(open(os.path.join(OUT,"mia_metrics.json"))); PE=R["per_example"]; M=R["meta"]
A=M["A_grid"]; TAU=[-np.log(a) for a in A]; TE=float(np.log(1/M["amp_floor"]))
def cal(m):
    o=[]
    for a in A:
        v=np.array(PE[m][f"{a:g}"]["auc_all"]); v=np.maximum(v,1-v); o.append(v.mean())
    return np.array(o)
def calpe(m):
    v=np.array(PE[m]["evicted"]["auc_all"]); v=np.maximum(v,1-v); return float(v.mean())
ret=np.array([PE["retention"][f"{a:g}"]["auc"][0] for a in A])
tpr1=np.array([PE["paired/s1"][f"{a:g}"]["tpr@fpr0.01"][0] for a in A])
s1,s4,h1,h4=cal("paired/s1"),cal("paired/s4"),cal("history/s1"),cal("history/s4")
pe_ret=R["postevict"]["retention_mean"][0]
X=TAU+[TE+0.30]
fig,axes=plt.subplots(1,2,figsize=(13.2,5.0),gridspec_kw={"width_ratios":[1.55,1]})
ax=axes[0]
ax.plot(X,list(ret)+[pe_ret],"o-",c="C0",lw=2.6,ms=6,zorder=5,label="RETENTION (shipped value-recovery)")
ax.plot(X,list(s4)+[calpe("postevict_paired/s4")],"^-",c="C1",lw=2.0,ms=5,label="MIA TM-2a white-box, paired OUT")
ax.plot(X,list(s1)+[calpe("postevict_paired/s1")],"s-",c="C3",lw=2.0,ms=5,label="MIA TM-1 query, paired OUT")
ax.plot(X,list(h4)+[calpe("postevict_history/s4")],"v--",c="C2",lw=1.8,ms=5,label="MIA TM-2a white-box, history OUT")
ax.plot(X,list(h1)+[calpe("postevict_history/s1")],"d--",c="C4",lw=1.6,ms=4.5,label="MIA TM-1 query, history OUT")
ax.plot(X,[1.0]*len(A)+[calpe("postevict_history/hole")],":",c="k",lw=2.0,
        label="MIA allocator trace (hole stat), history OUT")
ax.plot(X,[s1[0]]*len(A)+[calpe("postevict_paired/s1")],c="0.35",lw=1.4,ls=(0,(7,3)),
        label="LAUNDERING CONTROL: TTL flag (amp$\\equiv$1 till expiry)")
ax.axvline(TE,c="0.4",ls=":",lw=1.4); ax.axhline(0.5,c="0.75",lw=1.0,ls="-.")
ax.text(TE-0.06,0.20,"self-evict (amp < floor 0.05)",rotation=90,ha="right",fontsize=8,c="0.35")
ax.text(0.03,0.515,"chance (AUC 0.5)",fontsize=7.5,c="0.55")
ax.set_xlabel(r"$\tau=\mathrm{leak}\cdot t=-\ln A$   (amplitude $A=e^{-\tau}$)")
ax.set_ylabel("retention  /  per-example MIA AUC (direction-calibrated)")
ax.set_ylim(-0.04,1.09); ax.legend(fontsize=7.4,loc="center left")
sec=ax.secondary_xaxis("top",functions=(lambda t:np.exp(-t),lambda a:-np.log(np.maximum(a,1e-9))))
sec.set_xlabel("amplitude A",fontsize=9)
ax.set_title("the store stops ANSWERING before it stops LEAKING",fontsize=11)
ax2=axes[1]
ax2.plot(TAU,ret,"o-",c="C0",lw=2.6,ms=6,label="retention")
ax2.plot(TAU,s1,"s-",c="C3",lw=2.0,ms=5,label="MIA AUC (TM-1 query)")
ax2.plot(TAU,tpr1,"*-",c="C5",lw=1.8,ms=8,label="MIA TPR @ FPR 1% (TM-1, LiRA)")
ax2.plot(TAU,s4,"^-",c="C1",lw=2.0,ms=5,label="MIA AUC (TM-2a white-box)")
ax2.axvline(TE,c="0.4",ls=":",lw=1.4)
ax2.set_xlim(1.5,3.1); ax2.set_ylim(0.78,1.02)
ax2.set_xlabel(r"$\tau=\mathrm{leak}\cdot t$"); ax2.set_ylabel("score")
ax2.set_title("zoom: the 'neither present nor absent' band\n"
              "at the floor retention = 0.832, MIA AUC = 0.983, TPR@1%FPR = 0.858",fontsize=9.5)
ax2.legend(fontsize=8,loc="lower left")
fig.suptitle(f"CLU designed store, MVC-0 decay: {M['n_targets']} targets x {len(M['seeds'])} seeds x "
             f"{M['n_worlds']} paired worlds x {M['n_query']} queries  (per-example U-LiRA)",fontsize=10)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(os.path.join(OUT,"fig1_retention_vs_mia.png"),dpi=160); plt.close(fig)
json.dump({"tau":TAU,"A":A,"retention":ret.tolist(),"cal_paired_s1":s1.tolist(),
           "cal_paired_s4":s4.tolist(),"cal_history_s1":h1.tolist(),"cal_history_s4":h4.tolist(),
           "tpr_fpr01_paired_s1":tpr1.tolist(),
           "postevict":{"retention":pe_ret,"paired_s1":calpe("postevict_paired/s1"),
                        "paired_s4":calpe("postevict_paired/s4"),
                        "history_s1":calpe("postevict_history/s1"),
                        "history_s2":calpe("postevict_history/s2"),
                        "history_s4":calpe("postevict_history/s4"),
                        "history_s5":calpe("postevict_history/s5"),
                        "history_hole":calpe("postevict_history/hole"),
                        "history_nlive":calpe("postevict_history/n_live")}},
          open(os.path.join(OUT,"fig1_data.json"),"w"),indent=2)
print("ok")
