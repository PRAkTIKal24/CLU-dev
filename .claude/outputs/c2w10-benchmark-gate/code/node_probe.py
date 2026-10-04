import numpy as np
from river import forest
D=np.load("incremental_reoccurring_balanced.npy"); X=D[:,:-1]; y=D[:,-1].astype(int)
cols=[f"f{j}" for j in range(33)]
m=forest.ARFClassifier(n_models=10,seed=1)
shown=False
for i in range(40000):
    d=dict(zip(cols,X[i])); m.predict_one(d); m.learn_one(d,y[i])
    if not shown and len(m.models): print("member type:",type(m.models[0]).__name__, [a for a in dir(m.models[0]) if 'node' in a.lower() or 'height' in a.lower()]); shown=True
    if (i+1)%5000==0:
        nn=[mem.n_nodes for mem in m.models]
        print(i+1,"n_nodes:",nn,"sum",sum(nn),flush=True)
