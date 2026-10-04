import json, numpy as np
from pathlib import Path
R=Path("/Users/user/Desktop/CHLU/.claude/outputs")
old={s:json.load(open(R/f"gen-cell-harness/runs/shapes3d_recon/seed_{s}/cell_manifest.json")) for s in (0,1,2)}
new={s:json.load(open(R/f"gen-val-split-fix/runs/cell/seed_{s}/seed_{s}/cell_manifest.json")) for s in (0,1,2)}
def dig(d,p):
    for k in p.split("|"):
        d=d.get(k) if isinstance(d,dict) else None
    return d
ARMS=["clu","ebm_compose","knn_proto","lace","lace/ld_published","lace/ld_corrected","store_removed"]
print("### CELL HARNESS — MOVED / UNCHANGED (external instrument)")
print("| arm | statistic | broken split | repaired split | Δmean | per-seed |")
print("|---|---|---|---|---|---|")
for arm in ARMS:
    for lbl,p in [("in-corner shape balacc", f"arms|{arm}|external|in_corner|balanced_accuracy"),
                  ("in-corner AUROC",        f"arms|{arm}|external|in_corner|auroc"),
                  ("composition B aggregate",f"arms|{arm}|external|composition_B|aggregate"),
                  ("distinct factor tuples on B", f"arms|{arm}|distinct|B|factor_tuples")]:
        a=[dig(old[s],p) for s in (0,1,2)]; b=[dig(new[s],p) for s in (0,1,2)]
        if any(x is None for x in a+b):
            continue
        a=[float(x) for x in a]; b=[float(x) for x in b]
        print(f"| `{arm}` | {lbl} | {np.mean(a):.4f} ± {np.std(a):.4f} | {np.mean(b):.4f} ± {np.std(b):.4f} | "
              f"{'**bit-identical**' if a==b else f'{np.mean(b)-np.mean(a):+.4f}'} | "
              + " · ".join(f"{x:.4f}→{y:.4f}" for x,y in zip(a,b)) + " |")
print("\n### cell: which selections moved")
for s in (0,1,2):
    eo=dig(old[s],"arms|ebm_compose|tuning_detail|selected") or dig(old[s],"arms|ebm_compose|tuning|selected")
    en=dig(new[s],"arms|ebm_compose|tuning_detail|selected") or dig(new[s],"arms|ebm_compose|tuning|selected")
    ks=("step_size","prior_sigma","n_steps","clip")
    if eo and en:
        so={k:eo.get(k) for k in ks}; sn={k:en.get(k) for k in ks}
        print(f"  seed {s} ebm_compose: {'MOVED' if so!=sn else 'same '} {so} -> {sn}")
    lo=dig(old[s],"arms|lace|lace_selection|selected") or dig(old[s],"lace_selection|selected")
    ln=dig(new[s],"arms|lace|lace_selection|selected") or dig(new[s],"lace_selection|selected")
    print(f"  seed {s} lace (INHERITED): {lo} -> {ln}")
print("\n### cell: structural invariants")
for lbl,p in [("store deployed bytes","arms|clu|byte_ledger|deployed_bytes"),
              ("clu channel bits","arms|clu|channel_bits"),
              ("decoder param hash","arms|clu|decoder_param_hash"),
              ("K","K"), ("D","D"),
              ("rows.B","rows|B"), ("rows.MS","rows|MS"), ("rows.fit","rows|fit"), ("rows.val","rows|val")]:
    a=[dig(old[s],p) for s in (0,1,2)]; b=[dig(new[s],p) for s in (0,1,2)]
    print(f"  {lbl:24s} {'IDENTICAL' if a==b else 'MOVED'}  old={a} new={b}")
