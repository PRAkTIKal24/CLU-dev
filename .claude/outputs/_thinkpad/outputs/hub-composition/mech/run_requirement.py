"""⭐⭐ HOW MUCH SMALLER WOULD THE STORE'S RESIDUAL HAVE TO BE TO REACH THE BAR?

Scale the store's OWN residual: x_i + alpha * e_i, alpha in [0,1].  ⛔ Not a heuristic
conversion through the isotropic-noise curve — the residual keeps its own (self-aligned)
structure, which is the thing that does the damage.  Read off alpha* at exact_match = 0.8333
(= 60/72, my target's bar) and at knn_proto's 14.33/72 = 0.1991.

Then: how does the realised residual scale with codewords M?  Measured on the store family,
so the alpha* can be turned into a BYTE requirement instead of an adjective.
⛔ Diagnostic. ⛔ `val` only; B never enters.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np, jax
from chlu.core.gen_store import (address_codes, channel_correlation, dead_channels,
                                 grouping_labels, init_store, store_bytes,
                                 well_channels_from_labels)
from chlu.core.gen_compose import read_compose
from mech_driver import build_context, score

ctx = build_context(0)
rows, X, lab = ctx["sets"]["val"]
Xf = np.asarray(X, np.float32).reshape((X.shape[0], -1))
nrm = np.linalg.norm(Xf, axis=1)
code = address_codes(ctx["store"], Xf)
rt = np.asarray(read_compose(ctx["store"], code), np.float32).reshape(Xf.shape)
e = rt - Xf
out = {"alpha": {}, "scaling": {}}
print("--- alpha sweep on the store's OWN residual (self-aligned structure preserved) ---", flush=True)
for a in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.0):
    Z = Xf + a * e
    rep = score(ctx, Z.astype(np.float32), lab, None)
    rel = float(np.mean(np.linalg.norm(Z - Xf, axis=1) / nrm))
    out["alpha"][f"{a:g}"] = {"alpha": a, "relerr": rel, "exact": rep["exact_match_frac"],
                              "valid": rep["distinct_valid"], "n": rep["n"]}
    print(f"alpha={a:<5g} relerr={rel:.4f}  exact={rep['exact_match_frac']:.4f}  "
          f"valid={rep['distinct_valid']:4d}/512", flush=True)

print("--- how the residual shrinks with codewords M (K = 5, the shipped partition) ---", flush=True)
gs = ctx["cfg"].gen_store
Xtr = np.asarray(ctx["X_store"], np.float32)
Xc = Xtr[: int(gs.n_corr)]
dead = dead_channels(Xc); corr = channel_correlation(Xc)
labels, _ = grouping_labels(corr, 5, "correlation", 0, gs.grouping_method,
                            dead=dead, min_well_frac=float(gs.min_well_frac))
wc = well_channels_from_labels(labels, 5)
for M in (16, 32, 64, 128, 256, 512, 1024, 2048):
    st = init_store(Xtr, wc, int(M), ctx["feat_shape"],
                    jax.random.PRNGKey(20_260_916), seed=0)
    c = address_codes(st, Xf)
    r = np.asarray(read_compose(st, c), np.float32).reshape(Xf.shape)
    rel = float(np.mean(np.linalg.norm(r - Xf, axis=1) / nrm))
    sb = int(store_bytes(st)["store_bytes"])
    rep = score(ctx, r, lab, None)
    out["scaling"][str(M)] = {"M": M, "relerr": rel, "store_bytes": sb,
                              "exact": rep["exact_match_frac"], "valid": rep["distinct_valid"]}
    print(f"M={M:<5d} store={sb:10d} B  relerr={rel:.4f}  exact={rep['exact_match_frac']:.4f} "
          f"valid={rep['distinct_valid']:4d}/512", flush=True)
json.dump(out, open(os.path.join(os.environ['OUT'], 'mech', 'requirement.json'), 'w'), indent=2)
print("done")
