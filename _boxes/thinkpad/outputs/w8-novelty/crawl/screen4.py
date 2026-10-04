import json,re,sys
rows=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8")]
R=lambda p: re.compile(p,re.I)
TEN=R(r"\b(tensor|multilinear|bilinear|Tucker|CP decomposition|outer product|core tensor|tensor[- ]product)\b")
PCA=R(r"\b(principal component|principal subspace|PCA|eigen(space|vector|decomposition|faces)|SVD|covariance)\b")
GEN=R(r"\b(generative|generator|GAN\b|VAE|diffusion|synthes\w+|decoder|image generation)\b")
COMPO=R(r"\b(compositional|unseen combination|unseen attribute|novel combination|combinatorial general\w+|attribute combination)\b")
FROZ=R(r"\b(frozen|freeze|pre[- ]?trained|off[- ]the[- ]shelf)\b")
def t(r): return r["title"]+" . "+r["abs"]
print("TOTAL SCREENED (deletion-side citing works):",len(rows))
for name,f in [("TENSOR/MULTILINEAR",TEN),("PCA/EIGEN/COVARIANCE",PCA),("GENERATIVE",GEN),("COMPOSITIONAL",COMPO)]:
    h=[r for r in rows if f.search(t(r))]
    print("\n=== %s : %d ==="%(name,len(h)))
    for r in sorted(h,key=lambda r:-(r["year"] or 0))[:18]:
        print("  [%s][%s] %s"%(r["year"],r["anchor"][:14],r["title"][:98]))
print("\n=== JOINT: (TENSOR or COMPOSITIONAL) and GENERATIVE ===")
h=[r for r in rows if (TEN.search(t(r)) or COMPO.search(t(r))) and GEN.search(t(r))]
print("count:",len(h))
for r in sorted(h,key=lambda r:-(r["year"] or 0))[:25]:
    print("  [%s][%s] %s"%(r["year"],r["anchor"][:14],r["title"][:98]))
print("\n=== JOINT: PCA and GENERATIVE and FROZEN ===")
h=[r for r in rows if PCA.search(t(r)) and GEN.search(t(r)) and FROZ.search(t(r))]
print("count:",len(h))
for r in sorted(h,key=lambda r:-(r["year"] or 0))[:25]:
    print("  [%s][%s] %s"%(r["year"],r["anchor"][:14],r["title"][:98]))
