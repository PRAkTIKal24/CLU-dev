import json,re,sys
rows=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8")]
R=lambda p: re.compile(p,re.I)
A6=R(r"\b(unlearn\w*|data deletion|deletion|delete|erasure|forget\w*|downdat\w+|decremental)\b")
GEN=R(r"\b(generative|generator|GAN|VAE|diffusion|synthes\w+|decoder)\b")
TEN=R(r"\b(tensor|multilinear|bilinear|Tucker|CP decomposition|outer product|core tensor)\b")
PCA=R(r"\b(principal component|principal subspace|PCA|eigen\w+|SVD|covariance)\b")
EXACT=R(r"\b(exact\w*|provabl\w+|equivalent to retrain\w*|retrain\w* from scratch|certified)\b")
COMPO=R(r"\b(compositional|unseen combination|unseen attribute|novel combination|combinatorial general\w+)\b")
FROZ=R(r"\b(frozen|freeze|pre[- ]?trained|off[- ]the[- ]shelf)\b")
def t(r): return r["title"]+" . "+r["abs"]
print("=== A6 + GENERATIVE + EXACT (the joint claim) ===")
h=[r for r in rows if A6.search(t(r)) and GEN.search(t(r)) and EXACT.search(t(r))]
print("count:",len(h))
for r in sorted(h,key=lambda r:-(r["year"] or 0))[:30]:
    print("  [%s] %s"%(r["year"],r["title"][:112]))
print()
print("=== A6 + (TENSOR or PCA) ===")
h=[r for r in rows if A6.search(t(r)) and (TEN.search(t(r)) or PCA.search(t(r)))]
print("count:",len(h))
for r in sorted(h,key=lambda r:-(r["year"] or 0))[:30]:
    which="TEN" if TEN.search(t(r)) else "PCA"
    print("  [%s][%s] %s"%(r["year"],which,r["title"][:105]))
print()
print("=== COMPO + FROZEN (+ tensor?) ===")
h=[r for r in rows if COMPO.search(t(r)) and FROZ.search(t(r))]
print("count:",len(h))
for r in sorted(h,key=lambda r:-(r["year"] or 0))[:25]:
    print("  [%s]%s %s"%(r["year"]," TEN" if TEN.search(t(r)) else "    ",r["title"][:105]))
print()
print("=== 'unseen attribute combinations' family ===")
for r in sorted([r for r in rows if "unseen attribute" in t(r).lower()],key=lambda r:-(r["year"] or 0))[:20]:
    print("  [%s] %s"%(r["year"],r["title"][:112]))
