import numpy as np, json
from metrics import all_metrics
S=json.load(open("structure.json"))
z=np.load("preds/nochange.npz"); pred,y=z["pred"],z["y"]
cur=np.load("preds/nochange_curve.npy")*100
print("No-Change windowed(1000) curve: min %.1f  median %.1f  max %.1f"%(cur.min(),np.median(cur),cur.max()))
for th in (80,90,95):
    frac=(cur>th).mean()*100
    print("  fraction of stream positions with No-Change window acc > %d%% : %.2f%% (%d of %d)"%(th,frac,(cur>th).sum(),len(cur)))
# where
hi=np.where(cur>90)[0]+1000
if len(hi):
    # contiguous runs
    runs=[];st=hi[0];pv=hi[0]
    for v in hi[1:]:
        if v>pv+1: runs.append((st,pv)); st=v
        pv=v
    runs.append((st,pv))
    print("  contiguous >90%% regions (stream positions):",[(int(a),int(b)) for a,b in runs])
print()
print("per-band No-Change accuracy (OUR band map, B=5):")
for b in S["bands"]:
    a,e=max(b["start"],1),b["end"]
    print("  cycle %d band %d  [%6d,%6d)  n=%5d  No-Change=%6.2f%%"%(b["cycle"],b["band"],a,e,e-a,100*(pred[a:e]==y[a:e]).mean()))
# max run-length of identical consecutive labels
d=np.diff(y); brk=np.flatnonzero(d)+1
runlen=np.diff(np.concatenate([[0],brk,[len(y)]]))
print("\nlabel run-lengths: max %d, mean %.2f, #runs %d"%(runlen.max(),runlen.mean(),len(runlen)))
i=int(np.argmax(runlen)); start=int(np.concatenate([[0],brk])[i])
print("  longest run starts at index %d, label %d"%(start,y[start]))
