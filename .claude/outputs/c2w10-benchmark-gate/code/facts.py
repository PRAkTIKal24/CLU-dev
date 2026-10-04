import numpy as np, hashlib, json, collections, os
BASE="/Users/user/Desktop/CHLU/.claude/data/c2w10-streams"
out={}
for name in ["incremental_reoccurring_balanced","incremental_abrupt_balanced"]:
    p=os.path.join(BASE,name+".csv")
    h=hashlib.sha256(open(p,'rb').read()).hexdigest()
    D=np.loadtxt(p,delimiter=",",dtype=np.float64)
    X=D[:,:-1]; y=D[:,-1].astype(np.int64)
    hist=collections.Counter(y.tolist())
    nochange=float(np.mean(y[1:]==y[:-1]))*100
    # persistence over all n (first prediction undefined) -> also report over n-1
    print(name, D.shape, "sha256",h)
    print("  classes",sorted(hist.items()), "n_classes",len(hist))
    print("  NoChange acc (t=2..n) %.4f%%"%nochange)
    print("  X min %.4f max %.4f  any nan %s"%(X.min(),X.max(),np.isnan(D).any()))
    out[name]=dict(sha256=h,n=int(D.shape[0]),f=int(X.shape[1]),k=len(hist),
                   hist={int(a):int(b) for a,b in sorted(hist.items())},nochange=nochange,
                   bytes=os.path.getsize(p))
    np.save(name+".npy", D.astype(np.float64))
json.dump(out,open("facts.json","w"),indent=1)
