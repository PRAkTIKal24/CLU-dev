"""Shared: flatten JSON numeric leaves to RFC-6901 pointers; test whether a printed token is the JSON value
at the printed precision (ROUND_HALF_UP or ROUND_HALF_EVEN on the float's repr). Nothing else is computed."""
import json
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
def esc(k): return str(k).replace("~", "~0").replace("/", "~1")
def flatten(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flatten(v, p + "/" + esc(k))
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flatten(v, p + "/" + str(i))
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield p, o
def resolve(o, ptr):
    for part in ptr.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        o = o[int(part)] if isinstance(o, list) else o[part]
    return o
def norm(tok):
    t = tok.replace("−", "-").replace(",", "").replace("$", "").strip()
    return t[1:] if t.startswith("+") else t
def matches(tok, val):
    """printed token == JSON value at printed precision. Returns the rounding rule that matched, or None."""
    t = norm(tok)
    try: d = Decimal(t)
    except Exception: return None
    places = -d.as_tuple().exponent if d.as_tuple().exponent < 0 else 0
    q = Decimal(1).scaleb(-places)
    v = Decimal(repr(float(val))) if isinstance(val, float) else Decimal(val)
    if v == d: return "exact"
    for name, rule in (("half-up", ROUND_HALF_UP), ("half-even", ROUND_HALF_EVEN)):
        r = v.quantize(q, rounding=rule)
        if r == d or (r == 0 and d == 0): return name
    return None
