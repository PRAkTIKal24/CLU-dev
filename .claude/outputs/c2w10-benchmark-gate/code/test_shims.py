import numpy as np
from samknn_port import _nn, SAMKNN
rng=np.random.default_rng(0); ok=True
# get1ToNDistances = squared euclidean
for _ in range(200):
    s=rng.normal(size=7); S=rng.normal(size=(13,7))
    ref=np.array([((s-S[i])**2).sum() for i in range(13)])
    assert np.allclose(_nn.get1ToNDistances(s,S),ref,atol=1e-12)
# nArgMin: n smallest, lowest-index tie-break (C: scan ascending, strict <)
def ref_nargmin(n,v):
    idx=[];
    for i in range(n):
        best=None;bv=np.inf
        for j in range(len(v)):
            if v[j]<bv and j not in idx: bv=v[j];best=j
        idx.append(best)
    return idx
for _ in range(300):
    v=rng.integers(0,5,size=12).astype(float)  # forces ties
    n=int(rng.integers(1,6))
    a=_nn.nArgMin(n,v)[0].tolist(); b=ref_nargmin(n,v)
    if sorted(a)!=sorted(b) or not np.allclose(v[a],v[b]): ok=False;print("nArgMin MISMATCH",v,n,a,b)
    # exact tie-break identity too:
    if a!=b: ok=False; print("nArgMin ORDER MISMATCH",v,n,a,b)
# mostCommon: ties -> lowest label
assert _nn.mostCommon(np.array([[1,1,2,2,3]]))[0]==1
assert _nn.mostCommon(np.array([[5,5,2,2,3]]))[0]==2
assert _nn.mostCommon(np.array([[7,7,7]]))[0]==7
# linear weighted: w=1/max(d,1e-9), ties->lowest label
l=np.array([[3,3,9]]); d=np.array([[1.0,1.0,0.4]])
# 3: 1+1=2 ; 9: 2.5 -> 9
assert _nn.getLinearWeightedLabels(l,d)[0]==9
l=np.array([[3,9]]); d=np.array([[0.5,0.5]])   # tie -> lowest label 3
assert _nn.getLinearWeightedLabels(l,d)[0]==3
l=np.array([[3,9]]); d=np.array([[0.0,0.5]])   # d=0 -> 1e9 dominates
assert _nn.getLinearWeightedLabels(l,d)[0]==3
# end-to-end: kNN_S(useLTM=False, no STM adaption) == brute-force distance-weighted sliding kNN
def brute(X,y,L,k=5):
    out=[]
    for i in range(len(y)):
        lo=max(0,i-L); Xs=X[lo:i]; ys=y[lo:i]
        if len(ys)==0: out.append(0); continue
        d=np.sqrt(((Xs-X[i])**2).sum(1)); n=min(k,len(ys))
        idx=np.argsort(d,kind='stable')[:n]; w=1/np.maximum(d[idx],1e-9)
        lab=np.unique(ys[idx]); s=np.array([w[ys[idx]==t].sum() for t in lab])
        out.append(lab[np.argmax(s)])
    return np.array(out)
X=rng.normal(size=(400,4)); y=rng.integers(2,8,size=400)
m=SAMKNN(maxSize=60,useLTM=False,recalculateSTMError=None)
p=m.alternateFitPredict(X,y)
b=brute(X,y,60)
agree=(p==b).mean()
print("kNN_S port vs brute force agreement: %.6f (n=400, L=60)"%agree)
ok = ok and agree==1.0
print("SHIM TESTS:", "PASS" if ok else "FAIL")
