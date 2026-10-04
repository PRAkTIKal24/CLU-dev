"""Scan EVERY held-out test corner at seeds 0,1,2: per request, which of the six columns
are valid through that seed's one frozen decoder.  Decides whether set 2 (all arms
succeed) exists anywhere at request level, and what the nearest pattern is if not."""
import json, os, numpy as np
from w17_rendered_compositions import render as RR, selection as S
RR._setenv(True)
from chlu.experiments import exp_gen_factor_wells as fw
order=[a for a,_,_ in S.ARMS]
res_all={}
for seed in (0,1,2):
    fw.PINS["condition"]="test"; fw.PINS["seed"]=seed
    cond=fw.load_condition(); grid=cond["grid"]
    bitmap=fw.value_presence_bitmap(grid, cond["train_rows"])
    judge=fw.Judge(seed, log=print)
    od=RR.OneDecoder.from_judge(judge,seed,S.EVIDENCE/S.WINNER_HASHES)
    objs,_=RR._load_arms(seed,grid,None)
    corners=fw.pair_subcorners(cond)+fw.nway_corners(cond)
    hist={}; best=[]; allsix=[]; fiveplus=[]
    pc=S.per_corner(seed); mism=[]
    for c in corners:
        vals,res=RR._read_corner(objs,od,c,grid,bitmap)
        V=np.stack([res[a]["valid"] for a in order],1)
        got={a:int(V[:,j].sum()) for j,a in enumerate(order)}
        want={a:int(pc[a][c["corner_id"]]["n_valid"]) for a in order}
        if got!=want: mism.append([c["corner_id"],got,want])
        k=V.sum(1)
        for n in k: hist[int(n)]=hist.get(int(n),0)+1
        for i in np.flatnonzero(k==6): allsix.append([c["corner_id"],int(c["rows"][i])])
        for i in np.flatnonzero(k>=5): fiveplus.append([c["corner_id"],int(c["rows"][i]),dict(zip(order,V[i].tolist()))])
    res_all[seed]={"n_corners":len(corners),"n_requests":int(sum(hist.values())),
       "hist_n_arms_co_valid":dict(sorted(hist.items())),"all_six":allsix,
       "five_or_more":fiveplus,"reproduction_mismatches":mism}
    print(seed, res_all[seed]["hist_n_arms_co_valid"], len(allsix), len(fiveplus), "mism", len(mism), flush=True)
json.dump(res_all,open(os.environ["O"]+"/diag/covalid_scan.json","w"),indent=1)
