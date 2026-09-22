"""Diagnose set 2: per-arm validity on the candidate held-out corner and on SEEN rows."""
import json, os, sys, numpy as np
from w17_rendered_compositions import render as RR, selection as S
RR._setenv(True)
from chlu.experiments import exp_gen_factor_wells as fw
fw.PINS["condition"]="test"; fw.PINS["seed"]=0
cond=fw.load_condition(); grid=cond["grid"]
bitmap=fw.value_presence_bitmap(grid, cond["train_rows"])
judge=fw.Judge(0, log=print)
od=RR.OneDecoder.from_judge(judge,0,S.EVIDENCE/S.WINNER_HASHES)
objs,_=RR._load_arms(0,grid,None)
corners={c["corner_id"]:c for c in fw.pair_subcorners(cond)+fw.nway_corners(cond)}
order=[a for a,_,_ in S.ARMS]
out={}
vals,res=RR._read_corner(objs,od,corners["nway-029"],grid,bitmap)
V=np.stack([res[a]["valid"] for a in order],1)
out["nway-029"]={"per_arm":dict(zip(order,V.sum(0).tolist())),
  "rows_where_cont_mlp_valid":{str(i):dict(zip(order,V[i].tolist())) for i in np.flatnonzero(V[:,order.index("cont_mlp__as_trained")])},
  "max_arms_co_valid":int(V.sum(1).max()),
  "n_rows_all_but_cont_mlp":int(V[:,[j for j,a in enumerate(order) if a!="cont_mlp__as_trained"]].all(1).sum()),
  "n_rows_all_learned_rivals_and_clu":int(V[:,[j for j,a in enumerate(order) if a not in("table",)]].all(1).sum())}
train=np.asarray(cond["train_rows"],np.int64); rng=np.random.default_rng(20260923)
probe={"corner_id":"SEEN","family":"seen","rows":np.sort(rng.choice(train,512,replace=False))}
vals,res=RR._read_corner(objs,od,probe,grid,bitmap)
V=np.stack([res[a]["valid"] for a in order],1)
A=np.stack([res[a]["accept"] for a in order],1)
out["SEEN512"]={"valid_per_arm":dict(zip(order,V.sum(0).tolist())),"accept_per_arm":dict(zip(order,A.sum(0).tolist())),
  "max_arms_co_valid":int(V.sum(1).max()),
  "n_all_but_cont_mlp":int(V[:,[j for j,a in enumerate(order) if a!="cont_mlp__as_trained"]].all(1).sum())}
print(json.dumps(out,indent=1))
json.dump(out,open(os.environ["O"]+"/diag/set2_diag.json","w"),indent=1)
