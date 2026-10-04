import os,sys,time,json
sys.path.insert(0, os.path.join(os.environ['OUT'],'mech'))
import numpy as np
from chlu.config import gen_dynamical_read_config
from chlu.experiments.exp_gen_dynamical_read import build_landscape
from mech_driver import build_context, arm_clu, arm_x0, score
ctx=build_context(0)
c=gen_dynamical_read_config(ctx["cfg"])
res={}
for budget in ("A","B"):
    t=time.time()
    built=build_landscape(ctx["cfg"],budget,ctx["X_store"],ctx["lab_store"],ctx["wc"],
                          ctx["feat_shape"],ctx["cards"],0,log=lambda *a: None)
    bt=time.time()-t
    print(f"### LANDSCAPE {budget} built in {bt:.1f}s", flush=True)
    land,book=built["land"],built["book"]
    atoms=np.asarray(ctx["lab_store"],np.int64)[np.asarray(built["store"].atom_rows,np.int64)]
    print(f"    atoms: {atoms.shape[0]} rows, {np.unique(atoms,axis=0).shape[0]} distinct tuples", flush=True)
    for which in ("val","MS"):
        rows,X,lab=ctx["sets"][which]
        t=time.time(); lau=arm_x0(ctx,land,book,which); dt=time.time()-t
        rep=score(ctx,lau["x0"],lab,atoms)
        res[f"x0_{budget}_{which}"]=rep
        print(f"x0[{budget}]/{which:3s} n={rep['n']:4d} distinct={rep['distinct']:4d} "
              f"valid={rep['distinct_valid']:4d} VALIDxNOVEL={rep['distinct_valid_novel']:4d} "
              f"exact={rep['exact_match_frac']:.4f} idx_tuples={int(np.unique(lau['atom_index'],axis=0).shape[0])} "
              f"conf={lau['mean_confidence']:.3f} ({dt:.1f}s)", flush=True)
    if budget=="A":
        for which in ("val","MS"):
            rows,X,lab=ctx["sets"][which]
            f=arm_clu(ctx,which); rep=score(ctx,f,lab,atoms); res[f"clu_{which}"]=rep
            print(f"clu /{which:3s}     n={rep['n']:4d} distinct={rep['distinct']:4d} "
                  f"valid={rep['distinct_valid']:4d} VALIDxNOVEL={rep['distinct_valid_novel']:4d} "
                  f"exact={rep['exact_match_frac']:.4f}", flush=True)
        # ⭐ THE CEILING on the SAME rows: real cached features through the same decoder
        for which in ("val","MS"):
            rows,X,lab=ctx["sets"][which]
            rep=score(ctx,X,lab,atoms); res[f"ceiling_{which}"]=rep
            print(f"CEIL/{which:3s}     n={rep['n']:4d} distinct={rep['distinct']:4d} "
                  f"valid={rep['distinct_valid']:4d} exact={rep['exact_match_frac']:.4f}", flush=True)
json.dump(res,open(os.path.join(os.environ['OUT'],'mech','x0_probe.json'),'w'),indent=2)
print("done")
