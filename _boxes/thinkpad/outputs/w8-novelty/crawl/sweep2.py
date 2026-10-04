import json,sys,time,urllib.request,urllib.parse
Q=['"sufficient statistic" "unlearning"','"sufficient statistics" "deletion" "model"',
   '"count table" "unlearning"','"summation form" "unlearning"',
   '"exact unlearning" "subspace"','"exact unlearning" "frozen"',
   '"unlearning" "conditional generation" "attribute"',
   '"deletion" "principal component" "exact"','"forgetting" "eigenmodel"',
   '"compositional" "generative" "exact deletion"','"tensor" "right to be forgotten"']
SEL="id,display_name,publication_year,abstract_inverted_index,cited_by_count"
def ab(inv):
    if not inv: return ""
    p={}
    for w,ix in inv.items():
        for i in ix: p[i]=w
    return " ".join(p[k] for k in sorted(p))
seen=set();out=[]
for q in Q:
    u=("https://api.openalex.org/works?filter=title_and_abstract.search:%s&per-page=100&select=%s&mailto=info@forgis.com"
       %(urllib.parse.quote(q),SEL))
    d=None
    for a in range(5):
        try:
            with urllib.request.urlopen(u,timeout=90) as r: d=json.loads(r.read().decode());break
        except Exception as e:
            if a==4: d={"results":[]}
            else: time.sleep(3*(a+1))
    res=d.get("results",[])
    print("%-48s -> %d"%(q[:48],len(res)))
    for w in res:
        k=w["id"].split("/")[-1]
        if k in seen: continue
        seen.add(k); out.append((q,w.get("publication_year"),w.get("display_name") or "",ab(w.get("abstract_inverted_index"))))
import re
KEY=re.compile(r"\b(tensor|multilinear|bilinear|principal component|PCA|eigen|subspace|compositional|generative|generator|decoder|frozen)\b",re.I)
EX=re.compile(r"\b(exact|provabl|equivalent to retrain|sufficient statistic)\b",re.I)
print("\n=== candidates with (structure-word AND exactness-word) ===")
hits=[(y,t) for q,y,t,a in out if KEY.search(t+" "+a) and EX.search(t+" "+a)]
print("count:",len(hits))
for y,t in sorted(hits,key=lambda x:-(x[0] or 0))[:30]: print("  [%s] %s"%(y,t[:105]))
