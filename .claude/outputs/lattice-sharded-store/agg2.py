import json, glob, os
from collections import defaultdict
import numpy as np
rows=[]
for f in glob.glob("w*.jsonl")+glob.glob("u_*.jsonl"):
    for ln in open(f):
        ln=ln.strip()
        if ln: rows.append(json.loads(ln))
# dedupe: prefer the record with the most diagnostics (oracle/by_router/blank)
best={}
def rank(r):
    w=r["written"]
    return (("by_router" in w), (w.get("strict_oracle_route") is not None), ("value_blank_ok" in r))
for r in rows:
    k=(r["cell"],r["arm"],r["seed"],bool(r.get("atom_init_local")))
    if k not in best or rank(r)>rank(best[k]): best[k]=r
g=defaultdict(list)
for k,r in best.items(): g[(k[0],k[1],k[3])].append(r)
print(f"{'cell':>9} {'arm':>18} {'lc':>2} {'n':>1} {'N':>1} {'atoms':>6} {'strict(R2)':>16} {'oracle=RG':>16} {'route':>6} {'payerr':>7} {'blank':>5} {'sec':>5}")
print("-"*115)
out={}
for k in sorted(g, key=lambda x:(x[0],x[1],x[2])):
    rs=sorted(g[k], key=lambda r:r["seed"])
    st=np.array([r["written"]["strict_success_rate"] for r in rs])
    orc=[r["written"].get("strict_oracle_route") for r in rs]
    orc=np.array([o for o in orc if o is not None])
    ro=np.array([r["written"]["route_accuracy"] for r in rs])
    pe=np.array([r["written"]["payload_abs_err_mean"] for r in rs])
    bl=[r["value_blank_ok"] for r in rs if "value_blank_ok" in r]
    out[k]=dict(strict=st.mean(), sd=st.std(), n=len(rs),
                oracle=(orc.mean() if len(orc) else None), osd=(orc.std() if len(orc) else None),
                nblank=len(bl), blankok=all(bl) if bl else None)
    os_ = f"{orc.mean():.4f}+-{orc.std():.4f}" if len(orc) else "-"
    print(f"{k[0]:>9} {k[1]:>18} {str(k[2])[0]:>2} {len(rs):>1} {rs[0]['n_shards']:>1} "
          f"{rs[0]['atoms_total']:>6} {st.mean():.4f}+-{st.std():.4f} {os_:>16} {ro.mean():>6.3f} "
          f"{pe.mean():>7.4f} {(f'{sum(bl)}/{len(bl)}' if bl else '-'):>5} "
          f"{np.mean([r['seconds_total'] for r in rs]):>5.0f}"
          + ("  PASS(R2)" if st.mean()>=0.9 else "")
          + ("  PASS(RG)" if len(orc) and orc.mean()>=0.9 else ""))
json.dump({f"{k[0]}|{k[1]}|{k[2]}":v for k,v in out.items()}, open("agg.json","w"), indent=1)
