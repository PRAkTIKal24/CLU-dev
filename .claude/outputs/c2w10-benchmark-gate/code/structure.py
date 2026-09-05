import numpy as np, json, math
D=np.load("incremental_reoccurring_balanced.npy"); X=D[:,:-1]; y=D[:,-1].astype(int); n=len(y)
CP=[26568,53364]
cyc=[(0,CP[0]),(CP[0],CP[1]),(CP[1],n)]
print("n=",n,"cycle sizes",[b-a for a,b in cyc],"sum",sum(b-a for a,b in cyc))
# empirical verification of the schedule: smoothed trace of a global feature statistic
w=2000
f=X[:, :7].mean(1)      # the 7 large-scale (wingbeat) features
sm=np.convolve(f,np.ones(w)/w,mode='valid')
pos=np.arange(len(sm))+w//2
print("smoothed trace at cycle-relative positions (mean of features 1-7, window 2000):")
for a,b in cyc:
    idx=[p for p in np.linspace(a,b-1,6).astype(int) if 0<=p-w//2<len(sm)]
    print("  cycle %6d-%6d :"%(a,b), " ".join("%.1f"%sm[p-w//2] for p in idx))
# turning points of the smoothed trace
tp1=int(np.argmax(sm[:CP[1]-w//2])+w//2); tp2=int(np.argmin(sm[CP[0]-w//2:]) + CP[0])
print("argmax of smoothed trace before cp2: %d   (published cp1=%d)"%(tp1,CP[0]))
print("argmin of smoothed trace after cp1 : %d   (published cp2=%d)"%(tp2,CP[1]))

# ---- band map (OUR construction) ----
B=5; bands=[]
for ci,(a,b) in enumerate(cyc):
    edges=np.linspace(a,b,B+1).round().astype(int)
    for bi in range(B):
        bands.append(dict(cycle=ci+1,band=bi,start=int(edges[bi]),end=int(edges[bi+1]),
                          n=int(edges[bi+1]-edges[bi])))
pair=[dict(c1_band=b,c2_band=B-1-b,c3_band=b) for b in range(B)]
# ---- decimation ladder ----
lad={}
for m in [1,2,5,10]:
    keep=np.arange(0,n,m)
    nm=len(keep)
    cps=[int(math.ceil(c/m)) for c in CP]
    counts=[int(((keep>=a)&(keep<b)).sum()) for a,b in cyc]
    lad[str(m)]=dict(n_instances=nm,change_points=cps,per_cycle_counts=counts,
                     all_three_cycles=all(c>0 for c in counts),
                     both_change_points=(0<cps[0]<cps[1]<nm),
                     sum_check=(sum(counts)==nm),
                     band_sizes=[int(((keep>=bd['start'])&(keep<bd['end'])).sum()) for bd in bands])
    print("m=%2d n=%6d cps=%s per-cycle=%s ok=%s"%(m,nm,cps,counts,lad[str(m)]['all_three_cycles'] and lad[str(m)]['both_change_points']))
json.dump(dict(change_points=CP,cycles=[[a,b] for a,b in cyc],bands=bands,revisit_pairing=pair,
               decimation=lad, turning_points_empirical=[tp1,tp2]),
          open("structure.json","w"),indent=1)
print("band sizes m=1:",lad['1']['band_sizes'])
