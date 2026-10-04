import json, numpy as np
OUT=".claude/outputs/mia-decay-measurement"
r=json.load(open(OUT+"/mia_metrics.json")); pe=r['per_example']
CB=np.array(r['meta']['codebook']); RD=r['meta']['disk_radius']; A=r['meta']['A_grid']
rows=[]
for seed in [0,1,2]:
    rng=np.random.default_rng(4242+seed); ang=rng.random(8)*2*np.pi
    rad=RD*np.sqrt(rng.random(8))*0.8
    for ti in range(8): rows.append((seed,ti,float(rad[ti]),float(CB[ti])))
rows=np.array(rows)
out={"A_grid":A,"per_target":{}, "corr":{}}
for a in A:
    y=np.array(pe['retention'][f'{a:g}']['auc_all'])
    out["corr"][f"{a:g}"]={"a2":float(np.corrcoef(y,rows[:,3]**2)[0,1]),
                            "cnorm":float(np.corrcoef(y,rows[:,2])[0,1])}
for ti in range(8):
    m=rows[:,1]==ti
    out["per_target"][str(ti)]={"payload":float(CB[ti]),"a2":float(CB[ti]**2),
        "mean_c_norm":float(rows[m,2].mean()),
        "retention":[float(np.array(pe['retention'][f'{a:g}']['auc_all'])[m].mean()) for a in A],
        "mia_s1":[float(np.array(pe['paired/s1'][f'{a:g}']['auc_all'])[m].mean()) for a in A]}
    ret=np.array(out["per_target"][str(ti)]["retention"]); tau=np.array([-np.log(x) for x in A])
    t50=None
    for i in range(len(A)-1):
        if ret[i]>=0.5>=ret[i+1]:
            t50=float(tau[i]+(0.5-ret[i])*(tau[i+1]-tau[i])/(ret[i+1]-ret[i])); break
    out["per_target"][str(ti)]["tau50_retention"]=t50
json.dump(out,open(OUT+"/payload_dependence.json","w"),indent=2)
print(json.dumps({k:out["per_target"][k]["tau50_retention"] for k in out["per_target"]},indent=1))
print("corr a2 at floor:",out["corr"]["0.051"])
