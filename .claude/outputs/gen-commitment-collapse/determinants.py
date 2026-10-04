"""⭐ From the persisted code tuples alone: WHICH factors the committed code is a function of."""
import json, itertools
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
out = {}
for s in (0, 1, 2):
    for k in ("A", "B", "MS"):
        d = json.load(open(R / f"codes_{k}_seed{s}.json"))
        Rq, C, names = np.array(d["requests"]), np.array(d["codes"]), d["factor_names"]
        varying = [i for i in range(Rq.shape[1]) if len(np.unique(Rq[:, i])) > 1]
        rec = {"n_requests": len(Rq), "n_distinct_codes": d["n_distinct_code_configs"],
               "varying_factors": [names[i] for i in varying],
               "n_values": {names[i]: int(len(np.unique(Rq[:, i]))) for i in varying}}
        # the SMALLEST factor subset the code is a deterministic function of
        best = None
        for r in range(1, len(varying) + 1):
            for sub in itertools.combinations(varying, r):
                g = {}
                for row, c in zip(Rq, C):
                    g.setdefault(tuple(row[list(sub)]), set()).add(tuple(c))
                if all(len(v) == 1 for v in g.values()):
                    best = [names[i] for i in sub]
                    break
            if best:
                break
        rec["code_is_a_function_of"] = best
        rec["ignored_factors"] = [names[i] for i in varying
                                  if best is not None and names[i] not in best]
        out[f"{k}_seed{s}"] = rec
json.dump(out, open(R / "determinants.json", "w"), indent=2, sort_keys=True)
for k, v in out.items():
    print(f"{k:8s} {v['n_distinct_codes']:4d} codes / {v['n_requests']:4d} requests · "
          f"code = f({', '.join(v['code_is_a_function_of'] or ['?'])}) · "
          f"IGNORED: {', '.join(v['ignored_factors']) or '—'}")
