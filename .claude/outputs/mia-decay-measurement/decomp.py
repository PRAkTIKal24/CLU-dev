"""Supplementary: decompose retention into basin vs value criterion, and reproduce
the controller-mvp decay-demo scoring (single-site basin => vacuous)."""
import json, sys
import numpy as np, jax
sys.path.insert(0,'.claude/scratch/mia-decay-measurement')
import mia_harness as H

A_LIST=[1.0,0.2,0.1,0.08,0.0608,0.051]
rows={}
for seed in [0,1,2]:
    rng=np.random.default_rng(4242+seed)
    ang=rng.random(H.N_TARGETS)*2*np.pi; rad=H.R_DISK*np.sqrt(rng.random(H.N_TARGETS))*0.8
    for ti in range(4):
        c=np.array([rad[ti]*np.cos(ang[ti]),rad[ti]*np.sin(ang[ti])]); a=float(H.CODEBOOK[ti%H.CAP])
        bg=np.array([H.CODEBOOK[(ti+1+j)%H.CAP] for j in range(H.N_BG)])
        W=H.build_worlds(seed,c,a,bg,32); base=H._pack(W,"in"); ts=W["tslot"]
        Q,P=H.queries_native(seed,c,32)
        for A in A_LIST:
            pk=H.set_amp(base,ts,A); addr,val=H.read_batch(pk,Q,P)
            d=np.linalg.norm(addr[:,:,None,:2]-pk["centers"][:,None,:,:],axis=-1)
            d=np.where(pk["active"][:,None,:]>0,d,np.inf)
            basin=(d.argmin(-1)==ts[:,None]).mean()
            value=(np.abs(val-a)<H.TOL).mean()
            strict=((d.argmin(-1)==ts[:,None])&(np.abs(val-a)<H.TOL)).mean()
            rows.setdefault(f"{A:g}",[]).append([float(basin),float(value),float(strict)])
out={k:np.array(v).mean(0).tolist() for k,v in rows.items()}
print(json.dumps({"A -> [basin_only, value_only(=controller-mvp decay-demo scoring), strict]":out},indent=2))
json.dump(out,open('.claude/outputs/mia-decay-measurement/retention_decomposition.json','w'),indent=2)
