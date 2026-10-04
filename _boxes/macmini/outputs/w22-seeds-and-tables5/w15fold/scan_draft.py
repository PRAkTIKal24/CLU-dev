"""List every number token the draft prints under a w15-{bar-d-score,coverage-control,lace-poe-rows} SRC scope.
Scope = same-line `% SRC:` or an enclosing `% SRCBLOCK:` ... `% ENDSRC`. Tokens: decimals (signed), comma-grouped
integers, integers of 4+ digits, and `a of b` / `a/b` counts. Output JSON list of {line, token, srcs, text}."""
import json, re, sys
from pathlib import Path
W15 = ("w15-bar-d-score", "w15-coverage-control", "w15-lace-poe-rows")
TOK = re.compile(r"(?<![\w.\\])([+\-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+\-−]?\d+\.\d+|\d{4,}|\d+ of \d+|\d+/\d+)(?![\w])")
def scan(path):
    out, block = [], None
    for i, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if s.startswith("% SRCBLOCK:"):
            block = [p.strip() for p in s[len("% SRCBLOCK:"):].split(";")]; continue
        if s.startswith("% ENDSRC"):
            block = None; continue
        body, _, com = raw.partition("% SRC:")
        # strip other latex comments
        body = re.split(r"(?<!\\)%", body)[0].replace("--", " – ")
        srcs = [p.strip() for p in com.split(";")] if com else (block or [])
        if not any(p.split("/")[0] in W15 for p in srcs):
            continue
        for m in TOK.finditer(body):
            out.append({"line": i, "token": m.group(1), "srcs": srcs, "text": body.strip()[:200]})
    return out
if __name__ == "__main__":
    res = scan(sys.argv[1])
    json.dump(res, sys.stdout, indent=1, ensure_ascii=False)
