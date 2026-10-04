import json, sys, time, urllib.request, urllib.parse, os
OUT = sys.argv[1]
ANCHORS = {
 "W2113055885":"TensorFaces (ECCV02)",
 "W2160551639":"Multilinear subspace analysis of image ensembles (03)",
 "W2131691213":"Multilinear image analysis for facial recognition (03)",
 "W2170653751":"Tenenbaum&Freeman bilinear (00)",
 "W2135666716":"Face Transfer multilinear (05)",
 "W3035589463":"MLC-VAE (ICML20)",
 "W3084065867":"MLC-VAE alt record",
 "W3131867783":"CausalX block multilinear",
 "W4281710985":"PandA (ICLR23)",
 "W4322420602":"PandA alt record",
 "W3014852036":"GANSpace",
 "W4287817254":"GANSpace alt record",
}
SEL="id,display_name,publication_year,abstract_inverted_index,doi,primary_location"
def abstract(inv):
    if not inv: return ""
    pos={}
    for w,ix in inv.items():
        for i in ix: pos[i]=w
    return " ".join(pos[k] for k in sorted(pos))
seen=set(); n=0
with open(OUT,"w",encoding="utf-8") as f:
    for wid,label in ANCHORS.items():
        cursor="*"; got=0
        while cursor:
            url=("https://api.openalex.org/works?filter=cites:%s&per-page=200&cursor=%s&select=%s&mailto=info@forgis.com"
                 % (wid, urllib.parse.quote(cursor), SEL))
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(url, timeout=90) as r:
                        d=json.loads(r.read().decode()); break
                except Exception as e:
                    if attempt==5: print("FAIL",wid,e,file=sys.stderr); d={"results":[],"meta":{}}
                    else: time.sleep(3*(attempt+1))
            res=d.get("results",[])
            for w in res:
                k=w["id"].split("/")[-1]
                if k in seen: continue
                seen.add(k)
                loc=(w.get("primary_location") or {}).get("source") or {}
                f.write(json.dumps({"id":k,"anchor":label,"year":w.get("publication_year"),
                    "title":w.get("display_name") or "","venue":loc.get("display_name") or "",
                    "abs":abstract(w.get("abstract_inverted_index"))})+"\n")
                n+=1
            got+=len(res)
            cursor=d.get("meta",{}).get("next_cursor")
            if not res: break
        print("anchor %-45s citing=%d" % (label, got), file=sys.stderr)
print("TOTAL UNIQUE CITING WORKS:", n, file=sys.stderr)
