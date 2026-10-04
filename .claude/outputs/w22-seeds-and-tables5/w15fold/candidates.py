import json, sys
from pathlib import Path
from jnum import flatten, matches, norm
EV = Path("/Users/batmac/clu/evidence/outputs")
FILES = {
 "w15-coverage-control": ["COVERAGE-CONTROL.json"],
 "w15-lace-poe-rows": ["GENTABLE-KSTAR.json"],
 "w15-bar-d-score": ["BAR-D.json", "CROSS-ARM-GPU-test.json", "CPU-LEG-BAR-D-under-w15-rules.json", "CPU-LEG-CROSS-ARM-recomputed.json"],
}
cache = {}
toks = json.load(open(sys.argv[1]))
for t in toks:
    art = t["srcs"][0].split("/")[0]
    if " of " in t["token"] or "/" in t["token"]: 
        print(f"L{t['line']:5d} {t['token']:>14}  COUNT"); continue
    hits = []
    for f in FILES[art]:
        if f not in cache: cache[f] = list(flatten(json.load(open(EV/art/f))))
        for p, v in cache[f]:
            r = matches(t["token"], v)
            if r and (norm(t["token"]).startswith("-") == (v < 0) or v == 0): hits.append(f"{f}#{p} ({r})")
    print(f"L{t['line']:5d} {t['token']:>14}  n={len(hits):3d}  " + " | ".join(hits[:4]))
