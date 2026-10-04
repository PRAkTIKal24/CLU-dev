"""⭐⭐ HOW MUCH FEATURE FIDELITY DOES THE DECODER ACTUALLY NEED?

Two curves, both on `val`, both diagnostics (they touch true features, so ⛔ neither is
an arm):
  (a) NOISE TOLERANCE  — x = X_true + eps, ||eps||/||X|| = r.  exact_match vs r.
  (b) PCA TRUNCATION   — X projected on the top-r principal directions fitted on the
      store's own TRAIN rows (label-free).  exact_match vs r, plus the rel-err it costs.

⭐ Purpose: convert "the store cannot round-trip" into a REQUIREMENT with a number on it —
the rel-err a read must reach before the bar is even physically in play.
⛔ `val` only. ⛔ B never enters.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np
from mech_driver import build_context, score

ctx = build_context(0)
rows, X, lab = ctx["sets"]["val"]
Xf = np.asarray(X, np.float32).reshape((X.shape[0], -1))
nrm = np.linalg.norm(Xf, axis=1, keepdims=True)
res = {"noise": {}, "pca": {}}

base = score(ctx, Xf, lab, None)
print(f"CEILING  exact={base['exact_match_frac']:.4f} valid={base['distinct_valid']}/{base['n']}",
      flush=True)
res["ceiling"] = base

rng = np.random.default_rng(20260916)
for r in (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
    E = rng.standard_normal(Xf.shape).astype(np.float32)
    E *= (r * nrm / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9))
    Xn = Xf + E
    rep = score(ctx, Xn, lab, None)
    got = float(np.mean(np.linalg.norm(Xn - Xf, axis=1) / nrm.ravel()))
    res["noise"][f"{r:g}"] = {"requested_relerr": r, "realised_relerr": got,
                              "exact": rep["exact_match_frac"], "valid": rep["distinct_valid"]}
    print(f"NOISE  r={r:<5g} relerr={got:.4f}  exact={rep['exact_match_frac']:.4f} "
          f"valid={rep['distinct_valid']:4d}/{rep['n']}", flush=True)

Xtr = np.asarray(ctx["X_store"], np.float32)
mu = Xtr.mean(0, keepdims=True)
# ⭐ SVD on the store's own train rows — label-free, and the OPTIMAL linear compressor
t = time.time()
U, S, Vt = np.linalg.svd(Xtr[:8192] - mu, full_matrices=False)
print(f"(svd {time.time()-t:.0f}s)", flush=True)
for r in (8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024):
    V = Vt[:r]
    Xp = (Xf - mu) @ V.T @ V + mu
    rep = score(ctx, Xp.astype(np.float32), lab, None)
    got = float(np.mean(np.linalg.norm(Xp - Xf, axis=1) / nrm.ravel()))
    res["pca"][str(r)] = {"r": r, "relerr": got, "exact": rep["exact_match_frac"],
                          "valid": rep["distinct_valid"],
                          "float_bytes_per_read": r * 4,
                          "basis_bytes": int(r * Xf.shape[1] * 4)}
    print(f"PCA    r={r:<5d} relerr={got:.4f}  exact={rep['exact_match_frac']:.4f} "
          f"valid={rep['distinct_valid']:4d}/{rep['n']}  basis={r*Xf.shape[1]*4:9d} B", flush=True)
json.dump(res, open(os.path.join(os.environ['OUT'], 'mech', 'capacity.json'), 'w'), indent=2)
print("done")
