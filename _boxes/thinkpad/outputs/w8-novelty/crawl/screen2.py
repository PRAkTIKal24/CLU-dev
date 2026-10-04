import json,re,sys,collections
ATOMS={
 "A1": r"\b(block[- ]?(wise|diagonal|term)|part[- ]based|partition(ed|ing)?|group[- ]?wise|per[- ]group|slot)\b",
 "A2": r"\b(principal (component|subspace|direction)|PCA|eigen(space|vector|decomposition|faces)|SVD)\b",
 "A4": r"\b(tensor[- ]product|multilinear|bilinear|Tucker|CP decomposition|core tensor|outer product|multiplicative interaction|Khatri|address book|look[- ]?up table|codebook)\b",
 "A3": r"\b(attribute|label|factor|conditional|conditioning|one[- ]hot|class[- ]conditional)\b",
 "A5": r"\b(frozen|freeze|pre[- ]?trained|off[- ]the[- ]shelf|without retraining|training[- ]free)\b",
 "A6": r"\b(unlearn\w*|data deletion|deletion|erasure|downdat\w+|decremental|right to be forgotten|GDPR|redact\w+)\b",
 "COMPO": r"\b(compositional|unseen combination|novel combination|combinatorial generalis|combinatorial generaliz|zero[- ]shot)\b",
}
C={k:re.compile(v,re.I) for k,v in ATOMS.items()}
rows=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8")]
cnt=collections.Counter()
for r in rows:
    t=r["title"]+" . "+r["abs"]
    r["_s"]={k:bool(c.search(t)) for k,c in C.items()}
    for k,v in r["_s"].items():
        if v: cnt[k]+=1
    r["_n"]=sum(1 for k in ("A1","A2","A3","A4","A5","COMPO") if r["_s"][k])
print("atom prevalence over %d works:"%len(rows), dict(cnt))
print()
print("=== A4 (tensor head) AND A5 (frozen/pretrained) ===")
h=[r for r in rows if r["_s"]["A4"] and r["_s"]["A5"]]
print("count:",len(h))
for r in sorted(h,key=lambda r:-(r["year"] or 0))[:25]:
    print("  [%s] %s"%(r["year"], r["title"][:110]))
print()
print("=== A4 AND COMPO (tensor head + compositional/unseen) ===")
h2=[r for r in rows if r["_s"]["A4"] and r["_s"]["COMPO"]]
print("count:",len(h2))
for r in sorted(h2,key=lambda r:-(r["year"] or 0))[:25]:
    print("  [%s] %s"%(r["year"], r["title"][:110]))
print()
print("=== TOP by atom count (A1,A2,A3,A4,A5,COMPO) ===")
for r in sorted(rows,key=lambda r:(-r["_n"],-(r["year"] or 0)))[:25]:
    fl="".join(k for k in ("A1","A2","A3","A4","A5","COMPO") if r["_s"][k])
    print("  n=%d [%s] %-95s {%s}"%(r["_n"],r["year"],r["title"][:95],fl))
