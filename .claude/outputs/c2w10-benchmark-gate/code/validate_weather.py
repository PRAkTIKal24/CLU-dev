import numpy as np, time
from samknn_port import SAMKNN
X=np.loadtxt("ne_data.csv",delimiter=","); y=np.loadtxt("ne_class.csv",dtype=np.int32)
print("weather",X.shape,np.bincount(y))
def zscore_global(X): 
    s=X.std(0); s[s<1e-12]=1; return (X-X.mean(0))/s
for scal,XX in [("raw",X),("globalz",zscore_global(X))]:
    for name,kw in [("SAM(5000)",dict(maxSize=5000,useLTM=True,recalculateSTMError=False)),
                    ("kNN_S(5000)",dict(maxSize=5000,useLTM=False,recalculateSTMError=None))]:
        t=time.time(); m=SAMKNN(n_neighbors=5,knnWeights='distance',**kw)
        p=m.alternateFitPredict(XX,y)
        err=100*(1-np.mean(p==y))
        print(f"  {scal:8s} {name:12s} interleaved test-train ERROR = {err:.2f}%   ({time.time()-t:.0f}s)")
