import numpy as np, json
S=json.load(open("structure.json")); M=json.load(open("metrics.json"))
z=np.load("preds/nochange.npz"); y=z["y"]
arms=["arf100_s1","samknn_1000_std","samknn_5000_std","knns_1000_std","knns_5000_std","nochange"]
P={a:np.load(f"preds/{a}.npz")["pred"] for a in arms}
print(f"{'cyc/band':10s} {'range':>16s} {'H(y) bits':>9s} {'majclass%':>9s} " + " ".join(f"{a[:13]:>13s}" for a in arms))
rows=[]
for b in S["bands"]:
    a,e=max(b["start"],1),b["end"]
    yy=y[a:e]; _,c=np.unique(yy,return_counts=True); p=c/c.sum()
    H=-(p*np.log2(p)).sum(); maj=100*p.max()
    accs=[100*(P[k][a:e]==yy).mean() for k in arms]
    print(f"c{b['cycle']}b{b['band']:<7d} [{a:6d},{e:6d}) {H:9.3f} {maj:9.1f} " + " ".join(f"{v:13.2f}" for v in accs))
    rows.append(dict(cycle=b['cycle'],band=b['band'],start=a,end=e,entropy_bits=float(H),
                     majority_pct=float(maj),**{k:float(v) for k,v in zip(arms,accs)}))
json.dump(rows,open("bandstats.json","w"),indent=1)
