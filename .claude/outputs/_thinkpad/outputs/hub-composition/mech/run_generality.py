"""⭐ IS THE SELF-ALIGNMENT DIAGNOSIS GENERAL, OR ONLY TRUE OF THE CODEBOOK?

Finding 10 was measured on the 524,288 B CODEBOOK.  The dynamical/addressed read uses a
completely different object — stored ATOMS, per-well argmax, no k-means centres at all.
If self-alignment is structural to "read out a stored object", it must appear there too.
⛔ If it does NOT, my Finding-11 conclusion is over-generalised and I have to say so.

Same ladder, at each arm's OWN residual magnitude.  ⛔ `val` only; B never enters.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np
from chlu.experiments.exp_gen_dynamical_read import build_landscape
from mech_driver import build_context, arm_x0, score

ctx = build_context(0)
rows, X, lab = ctx["sets"]["val"]
Xf = np.asarray(X, np.float32).reshape((X.shape[0], -1))
nrm = np.linalg.norm(Xf, axis=1)
rng = np.random.default_rng(20260916)
out = {}
for budget in ("A", "B"):
    built = build_landscape(ctx["cfg"], budget, ctx["X_store"], ctx["lab_store"], ctx["wc"],
                            ctx["feat_shape"], ctx["cards"], 0, log=lambda *a: None)
    atoms = np.asarray(ctx["lab_store"], np.int64)[np.asarray(built["store"].atom_rows, np.int64)]
    x0 = np.asarray(arm_x0(ctx, built["land"], built["book"], "val")["x0"], np.float32)
    e = x0 - Xf
    perm = rng.permutation(Xf.shape[0])
    g = rng.standard_normal(Xf.shape).astype(np.float32)
    g *= (np.linalg.norm(e, axis=1, keepdims=True) /
          np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9))
    print(f"\n=== ATOM read, rung {budget} (atoms are TRAIN ROWS, ⛔ not centres) ===", flush=True)
    out[budget] = {}
    for name, Z in (("same (the x0 arm)", x0), ("negated residual", Xf - e),
                    ("permuted residual", Xf + e[perm]), ("gaussian, matched norm", Xf + g),
                    ("ceiling", Xf)):
        rep = score(ctx, Z.astype(np.float32), lab, atoms)
        rel = float(np.mean(np.linalg.norm(Z - Xf, axis=1) / nrm))
        out[budget][name] = {"relerr": rel, "exact": rep["exact_match_frac"],
                             "valid": rep["distinct_valid"], "n": rep["n"]}
        print(f"  {name:24s} relerr={rel:.4f}  exact={rep['exact_match_frac']:.4f}  "
              f"valid={rep['distinct_valid']:4d}/{rep['n']}", flush=True)
json.dump(out, open(os.path.join(os.environ['OUT'], 'mech', 'generality.json'), 'w'), indent=2)
print("done")
