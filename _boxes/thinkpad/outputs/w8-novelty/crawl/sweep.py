import json,sys,time,urllib.request,urllib.parse
QUERIES=[
 '"exact unlearning"','"machine unlearning" "generative model"','"machine unlearning" "principal component"',
 '"unlearning" "covariance"','"unlearning" "tensor decomposition"','"data deletion" "generative"',
 '"exact deletion" "model"','"compositional generation" "frozen"','"unseen attribute combinations"',
 '"tensor product" "compositional generation"','"multilinear" "generative model" "attribute"',
 '"address book" "generative"','"look-up table" "attribute" "generation"',
 '"factorized conditional" "generation"','"principal subspace" "decoder" "attribute"',
 '"style and content" "tensor" "generation"','"compositional" "frozen decoder"',
 '"downdating" "covariance" "eigen"','"decremental" "principal component analysis"',
 '"unlearning" "diffusion model" "exact"','"retrain" "equivalent" "unlearning" "closed form"',
]
SEL="id,display_name,publication_year,abstract_inverted_index,cited_by_count"
def abstract(inv):
    if not inv: return ""
    pos={}
    for w,ix in inv.items():
        for i in ix: pos[i]=w
    return " ".join(pos[k] for k in sorted(pos))
seen=set();n=0
with open(sys.argv[1],"w",encoding="utf-8") as f:
    for q in QUERIES:
        cursor="*";got=0;pages=0
        while cursor and pages<3:
            url=("https://api.openalex.org/works?filter=title_and_abstract.search:%s&per-page=200&cursor=%s&select=%s&mailto=info@forgis.com"
                 %(urllib.parse.quote(q),urllib.parse.quote(cursor),SEL))
            d=None
            for a in range(5):
                try:
                    with urllib.request.urlopen(url,timeout=90) as r:
                        d=json.loads(r.read().decode());break
                except Exception as e:
                    if a==4: print("FAIL",q,e,file=sys.stderr);d={"results":[],"meta":{}}
                    else: time.sleep(3*(a+1))
            res=d.get("results",[])
            for w in res:
                k=w["id"].split("/")[-1]
                if k in seen: continue
                seen.add(k)
                f.write(json.dumps({"id":k,"anchor":"sweep:"+q,"year":w.get("publication_year"),
                  "title":w.get("display_name") or "","venue":"","cites":w.get("cited_by_count"),
                  "abs":abstract(w.get("abstract_inverted_index"))})+"\n")
                n+=1
            got+=len(res);pages+=1
            cursor=d.get("meta",{}).get("next_cursor")
            if not res: break
        print("%-52s -> %d" % (q[:52],got), file=sys.stderr)
print("TOTAL NEW UNIQUE:",n,file=sys.stderr)
