"""Cache COVERAGE only: a set-membership count on ROW INDICES.
⛔ Not a scored run: no feature of eval_a/eval_b is read, nothing is rendered or scored.
It answers one question — can a scored run happen on this box without extending the cache?
"""
import json, sys, os
import numpy as np
import ctx
c = ctx.build_context(0)
crows = np.asarray(c["crows"], np.int64)
rows = c["rows"]
out = {"cache": {"n_rows": int(c["cman"]["n_rows"]), "scope": str(c["cman"]["scope"]),
                 "D": int(c["cman"]["D"])}, "sets": {}}
need = [np.asarray(c["pool"], np.int64)]
for k, v in rows.items():
    v = np.asarray(v, np.int64)
    need.append(v)
    miss = int((~np.isin(v, crows)).sum())
    out["sets"][k] = {"n": int(v.size), "uncached": miss}
allneed = np.unique(np.concatenate(need))
out["run_seed_shortfall"] = int((~np.isin(allneed, crows)).sum())
out["n_needed_total"] = int(allneed.size)
out["gate_would_be_enforced"] = bool(out["run_seed_shortfall"] > 0)
print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(os.environ["CHLU_OUT"].replace("/c/","C:/",1),
                                 "dev", "coverage_seed0.json"), "w"), indent=2)
