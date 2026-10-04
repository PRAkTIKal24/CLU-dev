"""⭐⭐ WHY is the store's residual ~39x more damaging than isotropic noise of the same norm?

Three controls, all at the SAME residual magnitude, all on `val`.  Let e_i = roundtrip_i - x_i.
  same       x_i + e_i          the store's actual output (the reference)
  permuted   x_i + e_pi(i)      ⭐ SAME residual set, SAME norms, ⛔ decoupled from the item
  negated    x_i - e_i          same magnitude, opposite direction
  gaussian   x_i + N(0, .) at matched per-item relative norm
⭐ If `permuted` is much less damaging than `same`, the damage is that each item's residual points
along ITS OWN distinguishing content (a pull toward its codeword) — ⛔ not a property of the
feature space at large.  That distinguishes "wrong metric" from "quantisation collapses items".
⛔ Diagnostic only. ⛔ `val` only; B never enters.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np
from chlu.core.gen_store import address_codes
from chlu.core.gen_compose import read_compose
from mech_driver import build_context, score

ctx = build_context(0)
rows, X, lab = ctx["sets"]["val"]
Xf = np.asarray(X, np.float32).reshape((X.shape[0], -1))
nrm = np.linalg.norm(Xf, axis=1)
code = address_codes(ctx["store"], Xf)
rt = np.asarray(read_compose(ctx["store"], code), np.float32).reshape(Xf.shape)
e = rt - Xf
rng = np.random.default_rng(20260916)
perm = rng.permutation(Xf.shape[0])
g = rng.standard_normal(Xf.shape).astype(np.float32)
g *= (np.linalg.norm(e, axis=1, keepdims=True) /
      np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9))
cands = {"ceiling": Xf, "same (the store)": rt, "permuted residual": Xf + e[perm],
         "negated residual": Xf - e, "gaussian, matched norm": Xf + g}
out = {}
for name, Z in cands.items():
    rep = score(ctx, Z.astype(np.float32), lab, None)
    rel = float(np.mean(np.linalg.norm(Z - Xf, axis=1) / nrm))
    out[name] = {"exact": rep["exact_match_frac"], "valid": rep["distinct_valid"],
                 "n": rep["n"], "relerr": rel, "distinct": rep["distinct"]}
    print(f"{name:24s} relerr={rel:.4f}  exact={rep['exact_match_frac']:.4f}  "
          f"valid={rep['distinct_valid']:4d}/{rep['n']}  distinct={rep['distinct']:4d}", flush=True)
json.dump(out, open(os.path.join(os.environ['OUT'], 'mech', 'residual.json'), 'w'), indent=2)
print("done")
