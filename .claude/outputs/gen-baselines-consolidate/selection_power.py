"""Does any VALID-ONLY selection statistic predict refusal power? (R27.2's premise, measured.)
READ-ONLY over gen-baselines artefacts."""
import json
from statistics import mean, stdev

RUNS=".claude/outputs/gen-baselines/runs/shapes3d_recon"
def spearman(a,b):
    def rank(x):
        o=sorted(range(len(x)),key=lambda i:x[i]); r=[0]*len(x)
        i=0
        while i<len(o):
            j=i
            while j+1<len(o) and x[o[j+1]]==x[o[i]]: j+=1
            avg=(i+j)/2+1
            for kk in range(i,j+1): r[o[kk]]=avg
            i=j+1
        return r
    ra,rb=rank(a),rank(b); n=len(a)
    ma,mb=mean(ra),mean(rb)
    num=sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    den=(sum((x-ma)**2 for x in ra)*sum((x-mb)**2 for x in rb))**0.5
    return num/den if den else float('nan')

out=[]
out.append("# Selection-power diagnostic — can a valid-only criterion pick a refusal configuration?\n")
out.append("Per arm-family, over the 36 tuning trials: Spearman rho between the selection criterion "
           "(validation aggregate / validation `shape`, both valid-only) and the ORACLE-ONLY AUROC(B vs C).\n")
out.append("| family | seed | rho(val_aggregate, AUROC(B vs C)) | rho(val_shape, AUROC(B vs C)) | selected val_agg | selected AUROC | best-AUROC trial's val_agg rank |")
out.append("|---|---|---|---|---|---|---|")
res={}
for fam in ("knn","lace"):
    for s in (0,1,2):
        d=json.load(open(f"{RUNS}/seed_{s}/baselines_manifest.json"))
        t=d["arms"][fam]["tuning"]["trials"]; sel=d["arms"][fam]["tuning"]["selected"]
        va=[x["val_aggregate"] for x in t]; vc=[x["val_compositional"] for x in t]
        au=[x["auroc_c_vs_b_ORACLE_ONLY"] for x in t]
        r1,r2=spearman(va,au),spearman(vc,au)
        best=max(range(len(t)),key=lambda i:au[i])
        rank_va=sorted(va,reverse=True).index(va[best])+1
        res[(fam,s)]=(r1,r2)
        out.append(f"| `{fam}` | {s} | {r1:+.3f} | {r2:+.3f} | {sel['val_aggregate']:.4f} | "
                   f"{sel['auroc_c_vs_b_ORACLE_ONLY']:.4f} | {rank_va}/36 |")
for fam in ("knn","lace"):
    v1=[res[(fam,s)][0] for s in (0,1,2)]; v2=[res[(fam,s)][1] for s in (0,1,2)]
    out.append(f"| `{fam}` | **mean** | **{mean(v1):+.3f} ± {stdev(v1):.3f}** | **{mean(v2):+.3f} ± {stdev(v2):.3f}** | — | — | — |")

out.append("\n## Near-tie sensitivity: top-2 trials by the pre-registered criterion\n")
out.append("| family | seed | rank | val_agg | AUROC(B vs C) ORACLE-ONLY | config |")
out.append("|---|---|---|---|---|---|")
for fam in ("knn","lace"):
    for s in (0,1,2):
        d=json.load(open(f"{RUNS}/seed_{s}/baselines_manifest.json"))
        t=sorted(d["arms"][fam]["tuning"]["trials"],key=lambda x:-x["val_aggregate"])
        for r in (0,1):
            x=t[r]
            cfg={k:v for k,v in x.items() if k in ("k","metric","normaliser","deployment","sigma","form","shrinkage")}
            out.append(f"| `{fam}` | {s} | {r+1} | {x['val_aggregate']:.4f} | {x['auroc_c_vs_b_ORACLE_ONLY']:.4f} | `{cfg}` |")
open(".claude/outputs/gen-baselines-consolidate/selection_power.md","w").write("\n".join(out)+"\n")
print("\n".join(out))
