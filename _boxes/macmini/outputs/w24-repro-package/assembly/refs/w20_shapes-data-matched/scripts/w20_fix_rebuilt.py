"""Re-derive a REBUILT ladder K*=32 store's basis at its REUSED address book's widths.

`pinned_basis` admits the mean's perpendicular direction only when its relative norm
clears 1e-4.  On a well where that norm sits at the threshold, a rebuild in a different
process can land on the other side of it (BLAS rounding) and give r columns where the
w12b book was fitted against r+1 -> a shape error in `render_span`.  This rewrites the
basis with the perpendicular column FORCED exactly where the book expects it, records
every adjustment, and leaves everything else untouched.  Whether the rebuild is faithful
is then decided by the integrity check: the rescored store must reproduce w12b's
per-corner rows (w20_tables.py).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w20_lib import OUT, jdump, log  # noqa: E402


def forced_basis(store, widths):
    out, adj = [], []
    for k, (M, r) in enumerate(zip(store.modes, store.deployed_rank)):
        V = np.asarray(M[1:1 + r], np.float64).T
        mu = np.asarray(M[0], np.float64)
        perp = mu - V @ (V.T @ mu) if r > 0 else mu
        nrm = float(np.linalg.norm(perp))
        ref = float(np.linalg.norm(mu)) or 1.0
        w = int(widths[k])
        if w == V.shape[1] + 1:
            V = np.hstack([V, (perp / max(nrm, 1e-300))[:, None]])
            if nrm / ref <= 1e-4:
                adj.append({"well": k, "rel_perp_norm": nrm / ref, "action": "perp column FORCED (book width %d)" % w})
        elif w != V.shape[1]:
            raise SystemExit("well %d: book width %d vs rank %d -- not a threshold case" % (k, w, V.shape[1]))
        out.append(np.ascontiguousarray(V))
    return out, adj


def main():
    for p in sorted((OUT / "stores").glob("rung*/seed*/K32_n131072/arm.pkl")):
        with open(p, "rb") as fh:
            a = pickle.load(fh)
        widths = [int(np.asarray(W).shape[1]) for W in a["address"].W]
        cur = [int(b.shape[1]) for b in a["basis"]]
        if cur == widths:
            log("%s: widths already agree" % p.parent)
            continue
        basis, adj = forced_basis(a["store"], widths)
        a["basis"] = basis
        a["basis_adjustments"] = adj
        with open(p, "wb") as fh:
            pickle.dump(a, fh)
        jdump({"arm": str(p), "adjustments": adj}, p.parent / "basis_adjustments.json")
        log("%s: %d well(s) adjusted %s" % (p.parent, len(adj), adj))


if __name__ == "__main__":
    main()
