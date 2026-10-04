"""THE MECHANISM — a continuous request coordinate settled in ONE valley.

⛔ NOT a blend weight.  The pinned interface (``one_hot_interpolated``) already makes a
between-values request a LINEAR BLEND of the two bracketing one-hots, and the pinned
span read is then ``out = (Phi @ W) @ G`` -- i.e. the CHORD MIDPOINT of the two atoms.
That chord point is OFF the feature manifold, which is exactly why the judge reads it
as a scattered, not-quite-requested hue.  This hub's dial is whether a basin carries a
CONTINUOUS COORDINATE instead:

  * every atom of a well carries a KERNEL of width sigma;
  * sigma is set so neighbouring kernels OVERLAP, and the two bracketing atoms stop
    being two basins and become ONE VALLEY;
  * the read SETTLES WITH FRICTION down that valley from a request-only launch point.

The settled point is pulled toward regions of high atom support, i.e. back ONTO the
manifold, while the request coordinate decides WHERE along the valley it lands.

⛔ The coefficients stay SIGNED and unconstrained -- no simplex is imposed anywhere.
⛔ No feature tensor enters launch, settle or read (charter section B.3): the only
inputs are the request design ``Phi``, the head and the store's own atoms.
"""
from __future__ import annotations

import numpy as np


def kernel_sigma(G: np.ndarray, rho: float) -> float:
    """sigma = rho x (median nearest-neighbour distance between the well's atoms).

    ⭐ ``rho`` is THE dial.  rho -> 0 gives disjoint basins and the settle snaps to one
    atom (a lookup, charter section C.4).  rho -> inf gives one quadratic bowl whose
    minimum is the weighted mean (the chord point again).  The claim of this hub lives
    strictly between the two, and ``rho`` is selected on ``val`` only.
    """
    A = np.asarray(G, np.float64)
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(1))
    return float(rho * np.median(nn))


def settle_well(c: np.ndarray, G: np.ndarray, z0: np.ndarray, sigma: float,
                steps: int, dt: float, gamma: float, lam: float) -> np.ndarray:
    """Damped descent of U(z) = -sum_m c_m K_sigma(z - g_m) + (lam/2)||z - z0||^2.

    ``c`` is (N, M) SIGNED; ``G`` is (M, d); ``z0`` is (N, d), the request-only launch.
    Friction ``gamma`` is what makes this a settle and not a jump: the state loses
    energy until it rests at the bottom of the valley the request selected.
    """
    N, M = c.shape
    d = G.shape[1]
    Gd = np.asarray(G, np.float64)
    cd = np.asarray(c, np.float64)
    z = np.asarray(z0, np.float64).copy()
    v = np.zeros((N, d), np.float64)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for _ in range(int(steps)):
        # K (N, M) and its gradient, in one pass, chunk-free (M is 64, d <= 2048)
        Dz = z[:, None, :] - Gd[None, :, :]                           # (N, M, d)
        diff2 = np.einsum("nmd,nmd->nm", Dz, Dz)                      # (N, M)
        K = np.exp(-diff2 * inv2s2)                                   # (N, M)
        wgt = cd * K * (2.0 * inv2s2)                                 # (N, M)
        # grad of -sum_m c_m K_m  =  +sum_m wgt_m (z - g_m)
        grad = np.einsum("nm,nmd->nd", wgt, Dz)
        grad += lam * (z - np.asarray(z0, np.float64))
        v = (1.0 - gamma * dt) * v - dt * grad
        dz = dt * v
        z = z + dz
    return z.astype(np.float32), float(np.abs(dz).mean())


def valley_read(heads, store, Phi: np.ndarray, z0: np.ndarray, rho: float,
                steps: int = 60, dt: float = 0.15, gamma: float = 3.0,
                lam: float = 0.02, chunk: int = 512) -> np.ndarray:
    """(N, P) request design -> (N, D) settled read.  ⛔ No feature argument."""
    P = np.asarray(Phi, np.float32)
    N = int(P.shape[0])
    out = np.zeros((N, int(store.dim)), np.float32)
    resid = []
    for W, w, G in zip(heads, store.well_channels, store.modes):
        ch = np.asarray(w, np.int64)
        Gm = np.asarray(G, np.float64)
        sig = kernel_sigma(Gm, rho)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            c = P[s:e] @ np.asarray(W, np.float32)
            zz, res = settle_well(c, Gm, np.asarray(z0[s:e, ch], np.float64),
                                  sig, steps, dt, gamma, lam)
            out[s:e, ch] = zz
            resid.append(res)
    return out, float(np.mean(resid))


def overlap_report(store, rho: float) -> dict:
    """Proof that the kernels OVERLAP -- the premise of 'one valley', measured."""
    rows = []
    for k, G in enumerate(store.modes):
        A = np.asarray(G, np.float64)
        d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(d2.min(1))
        sig = kernel_sigma(A, rho)
        rows.append({
            "well": k, "n_atoms": int(A.shape[0]), "dim": int(A.shape[1]),
            "median_nn": float(np.median(nn)), "sigma": sig,
            # two kernels overlap usefully when sigma >= half the spacing
            "overlap_ratio": float(sig / (0.5 * np.median(nn))),
            "frac_pairs_within_2sigma": float((np.sqrt(d2) < 2 * sig).mean()),
        })
    return {"rho": rho, "wells": rows,
            "min_overlap_ratio": float(min(r["overlap_ratio"] for r in rows))}


def settle_diagnostics(store, Phi, heads, z0: np.ndarray, z: np.ndarray,
                       requested_hue, bracket_hue) -> dict:
    """Charter section C.4: is the settled state just sitting on its nearest atom?

    ``d_near`` is the distance from the settled point to the NEAREST atom of the well;
    ``d_req`` is the distance the request itself asks it to travel (half the spacing
    between the two bracketing atoms).  An arm whose settled state has d_near -> 0 is a
    lookup table and is reported as one, whatever its score.
    """
    rows = []
    for k, (w, G) in enumerate(zip(store.well_channels, store.modes)):
        ch = np.asarray(w, np.int64)
        A = np.asarray(G, np.float64)
        Z = np.asarray(z[:, ch], np.float64)
        Z0 = np.asarray(z0[:, ch], np.float64)
        # ⛔ CHUNKED: the unchunked form materialises (N, M, d) float64 -- ~8.6 GB at
        # N=4096 -- which is the one allocation on this hub big enough to be killed on
        # a shared box. Chunking changes no number; it only bounds the peak.
        nn = np.empty(Z.shape[0], np.float64)
        for s0 in range(0, Z.shape[0], 256):
            e0 = min(s0 + 256, Z.shape[0])
            blk = Z[s0:e0, None, :] - A[None, :, :]
            nn[s0:e0] = np.sqrt(np.einsum("nmd,nmd->nm", blk, blk)).min(1)
        d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        spacing = float(np.median(np.sqrt(d2.min(1))))
        rows.append({
            "well": k,
            "d_near_mean": float(nn.mean()),
            "atom_spacing_median": spacing,
            "d_near_over_spacing": float(nn.mean() / spacing),
            "moved_from_launch_mean": float(np.linalg.norm(Z - Z0, axis=1).mean()),
            "moved_over_spacing": float(
                np.linalg.norm(Z - Z0, axis=1).mean() / spacing),
            "frac_snapped_within_5pct": float((nn < 0.05 * spacing).mean()),
        })
    return {"wells": rows,
            "max_frac_snapped": float(max(r["frac_snapped_within_5pct"] for r in rows)),
            "min_d_near_over_spacing": float(min(r["d_near_over_spacing"] for r in rows)),
            "note": ("d_near -> 0 means the settled state IS an atom, i.e. a lookup "
                     "(charter section C.4), and is reported as one whatever it scores")}
