"""The w15 three-way check as a library: run(draft_path) -> {"rows", "extras", "problems"}.
Every w15-scoped number -> (draft line, cited FINDINGS line, JSON value). See w15_map.py."""
import json, re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from scan_draft import scan
from jnum import resolve, matches
from w15_map import MAP, EXTRA
EV = Path("/Users/batmac/clu/evidence/outputs")
_j, _t = {}, {}
def J(f):
    if f not in _j: _j[f] = json.loads((EV / f).read_text(encoding="utf-8"))
    return _j[f]
def T(f):
    if f not in _t: _t[f] = (EV / f).read_text(encoding="utf-8").splitlines()
    return _t[f]
def variants(tok):
    t = tok.replace("$", "")
    vs = [t]
    if t.startswith("-"): vs += ["\u2212" + t[1:]]
    if t.startswith("+"): vs += [t[1:]]
    if "," in t: vs += [t.replace(",", "")]
    m = re.match(r"(\d+) of (\d+)$", t)
    if m: vs += [f"{m.group(1)}/{m.group(2)}"]
    return vs
def find_line(path, tok):
    for i, ln in enumerate(T(path), 1):
        for v in variants(tok):
            if re.search(r"(?<![\d.])" + re.escape(v) + r"(?![\d])", ln):
                return i, v
    return None, None
def jcheck(tok, f, ptr):
    v = resolve(J(f), ptr)
    m = re.match(r"(\d+)(?: of |/)(\d+)$", tok)
    if m:
        return v, ("count-equal" if v == int(m.group(1)) else None)
    return v, matches(tok, v)
def fmt_leaf(v):
    if isinstance(v, bool) or isinstance(v, str): return str(v)
    if isinstance(v, int): return str(v)
    return str(Decimal(repr(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
def run(draft):
    draft = Path(draft); lines = draft.read_text(encoding="utf-8").splitlines()
    toks = scan(draft)
    byline = {}
    for t in toks: byline.setdefault(t["line"], []).append(t)
    anchor_line = {a: [i for i, ln in enumerate(lines, 1) if a in ln] for a in list(MAP) + [e[0] for e in EXTRA]}
    rows, problems, extras = [], [], []
    mapped = set()
    for a, spec in MAP.items():
        hits = anchor_line[a]
        if len(hits) != 1: problems.append(f"ANCHOR {a!r}: {len(hits)} lines"); continue
        L = hits[0]; tl = byline.get(L, []); mapped.add(L)
        if len(tl) != len(spec): problems.append(f"L{L} {a!r}: {len(tl)} tokens vs {len(spec)} map entries"); continue
        for t, sp in zip(tl, spec):
            fl, fv = find_line(t["srcs"][0], t["token"])
            row = {"draft_line": L, "token": t["token"], "findings": t["srcs"][0], "findings_line": fl, "findings_verbatim": fv}
            if sp[0] == "TXT":
                _, path, verb = sp
                hit = [i for i, ln in enumerate(T(path), 1) if verb in ln]
                row.update(artefact=path, artefact_line=hit[0] if hit else None, artefact_verbatim=verb, artefact_ok=bool(hit))
            elif sp[0] == "NOJSON":
                row.update(artefact=None, artefact_ok=None, nojson=sp[1])
            else:
                f, ptr = sp; v, r = jcheck(t["token"], f, ptr)
                row.update(artefact=f, pointer=ptr, json_value=v, rounding=r, artefact_ok=r is not None)
            rows.append(row)
    unmapped = sorted(set(byline) - mapped)
    for L in unmapped: problems.append(f"L{L}: w15-scoped tokens with no map entry: {[t['token'] for t in byline[L]]}")
    for a, verb, f, ptrs in EXTRA:
        hits = anchor_line[a]
        if len(hits) != 1: problems.append(f"EXTRA ANCHOR {a!r}: {len(hits)}"); continue
        L = hits[0]
        extras.append({"draft_line": L, "draft_verbatim": verb, "on_line": verb in lines[L - 1], "artefact": f,
                       "pointers": ptrs, "values": [resolve(J(f), p) for p in ptrs]})
        if verb not in lines[L - 1]: problems.append(f"EXTRA {verb!r} not on L{L}")
    return {"draft": str(draft), "n_tokens": len(toks), "rows": rows, "extras": extras, "problems": problems}
