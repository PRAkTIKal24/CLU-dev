import json,re,sys
ATOMS={
 "A1_wells": r"\b(block[- ]?(wise|diagonal|term)|part[- ]based|partition(ed|ing)?|group[- ]?wise|per[- ]group|slot|well)\b",
 "A2_pcasub": r"\b(principal (component|subspace|direction)|PCA|eigen(space|vector|decomposition|faces)|SVD|leading subspace|unsupervised(ly)? (discover|learn|found))\b",
 "A3A4_tensorhead": r"\b(tensor[- ]product|multilinear|bilinear|Tucker|CP decomposition|core tensor|outer product|multiplicative interaction|Khatri|mode[- ]n product|address book|look[- ]?up table|codebook|dictionary)\b",
 "A3A4_labelidx": r"\b(attribute|label|factor|condition(al|ing|ed)|class[- ]conditional|one[- ]hot|request|query)\b",
 "A5_frozen": r"\b(frozen|freeze|pre[- ]?trained (generator|decoder|GAN|model|diffusion)|off[- ]the[- ]shelf|without retraining|training[- ]free)\b",
 "A6_delete": r"\b(unlearn\w*|machine unlearning|data deletion|deleting|delete|deletion|forget(ting|ful)?|erasure|erase|right to be forgotten|downdat\w+|decremental|GDPR)\b",
}
C={k:re.compile(v,re.I) for k,v in ATOMS.items()}
rows=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8")]
def score(r):
    t=(r["title"]+" . "+r["abs"])
    return {k:bool(c.search(t)) for k,c in C.items()}
a6=[];strong=[]
for r in rows:
    s=score(r); r["_s"]=s
    if s["A6_delete"]: a6.append(r)
    n=sum(s.values())
    if s["A5_frozen"] and s["A3A4_tensorhead"] and s["A3A4_labelidx"]: strong.append((n,r))
print("=== TOTAL SCREENED:",len(rows))
print("=== A6 (deletion/unlearning language) HITS:",len(a6))
for r in sorted(a6,key=lambda r:-(r["year"] or 0)):
    print("  [%s] %s | %s | %s" % (r["year"], r["title"][:105], r["venue"][:32], r["anchor"][:22]))
print()
print("=== STRONG (frozen + tensor-head + label-index) HITS:",len(strong))
for n,r in sorted(strong,key=lambda x:-(x[1]["year"] or 0))[:40]:
    flags="".join(k[0:2] for k,v in r["_s"].items() if v)
    print("  [%s] %s | %s | %s" % (r["year"], r["title"][:100], r["venue"][:30], r["anchor"][:20]))
