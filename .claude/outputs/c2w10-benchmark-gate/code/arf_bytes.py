import numpy as np, pickle, sys, time, json
from river import forest
D=np.load("incremental_reoccurring_balanced.npy"); X=D[:,:-1]; y=D[:,-1].astype(int)
cols=[f"f{j}" for j in range(33)]
nm=int(sys.argv[1]); seed=int(sys.argv[2])
m=forest.ARFClassifier(n_models=nm,seed=seed); t=time.time()
for i in range(len(y)):
    d=dict(zip(cols,X[i])); m.predict_one(d); m.learn_one(d,y[i])
blob=pickle.dumps(m,protocol=5)
def nodes(t):
    try: return t._root.n_nodes if t._root is not None else 0
    except Exception: return None
nn=[]
for mem in m.models:
    tr=getattr(mem,'model',mem)
    try: nn.append(tr.n_nodes)
    except Exception: nn.append(None)
out=dict(n_models=nm,seed=seed,pickle_bytes=len(blob),wall_s=time.time()-t,
         n_nodes_per_tree=nn, total_nodes=sum(x for x in nn if x), 
         n_bkg=sum(1 for mem in m.models if getattr(mem,'_background_model',None) is not None))
print(json.dumps(out))
json.dump(out,open(f"preds/arfbytes_{nm}_{seed}.json","w"),indent=1)
