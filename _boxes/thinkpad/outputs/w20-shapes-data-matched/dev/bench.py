import os, sys, time, numpy as np
sys.path.insert(0, "C:/chlu/worktrees/CHLU-w20-shapes-data-matched")
import jax, equinox as eqx
from chlu.experiments import exp_gen_factor_wells as fw
from chlu.eval.rivals import w12_deletion_baselines as W12B, w4_learners as L
fw.PINS["condition"]="test"; fw.PINS["seed"]=0
cond=fw.load_condition(); grid=cond["grid"]
z=np.load("C:/chlu/outputs/hub-factor-wells/run/partitions.npz",allow_pickle=True); pool=np.asarray(z["pool"],np.int64)
fp=fw.fit_pool(cond["train_rows"], int(fw.PINS["n_pool"]), int(fw.PINS["pool_rng_seed"]))
print("pool==fit_pool:", np.array_equal(pool, fp), "train_rows", cond["train_rows"].size, flush=True)
F=fw.features_mmap(0)
rows=pool[:8192]; t=time.time(); Y=np.asarray(fw.gather(F, rows),np.float32); print("gather 8192", time.time()-t, flush=True)
cards=[int(c) for c in grid.cardinalities]
for fam in sys.argv[1:]:
    arm=W12B.matched_arm(fam,cards,2048,jax.random.PRNGKey(0))
    X=W12B._encode(arm,grid,rows,cards)
    t=time.time(); m,_=L.train(arm["build"](arm["width"]),X,Y,jax.random.PRNGKey(1),1,1e-3,0.0); t1=time.time()-t
    t=time.time(); m,_=L.train(arm["build"](arm["width"]),X,Y,jax.random.PRNGKey(1),6,1e-3,0.0); t2=time.time()-t
    print(fam, arm["param_count"], "1ep(8192) %.1fs  2ep %.1fs -> per-epoch@131k ~ %.1fs" % (t1,t2,(t2-t1)/5*16), flush=True)
