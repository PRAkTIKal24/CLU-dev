"""The zero-dynamics identity: the zero-step settle equals the static read (synthetic
checks of conditions C1-C4 and the coercivity certificate).

    python scripts/zero_dynamics_check.py
"""

import json
import os
import types
import numpy as np
from nclu.core.gen_read import TensorAddress, render_span, span_basis

RES = {}


def _banner(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def build_store(seed=0):
    rng = np.random.default_rng(seed)
    dims = [40, 30, 20]
    n_atoms = [16, 12, 20]
    idx, o = ([], 0)
    for d in dims:
        idx.append(np.arange(o, o + d))
        o += d
    modes = [rng.normal(size=(n, d)).astype(np.float32) for n, d in zip(n_atoms, dims)]
    return types.SimpleNamespace(modes=modes, well_channels=idx, dim=int(sum(dims)))


def build_address(store, U, seed=1, rank=8, card=(6, 5, 4)):
    rng = np.random.default_rng(seed)
    ta = TensorAddress(list(card), rank=rank, seed=seed)
    ta.E = [rng.normal(1.0, 0.3, (int(c), rank)) for c in card]
    widths = [int(u.shape[1]) for u in U]
    ta.W = [rng.normal(0.0, 0.2, (rank, r)) for r in widths]
    ta._widths = widths
    tot = int(sum(widths))
    ta.center = rng.normal(0.0, 0.1, tot)
    ta.scale = np.full(tot, 0.5)
    gv = [np.arange(int(c), dtype=float) for c in card]
    return (ta, gv)


def probe_A(store, U, ta, gv, vals):
    X, oor = ta.read(store, vals, gv, U)
    offs = []
    for u, ix in zip(U, store.well_channels):
        xk = X[:, ix].astype(np.float64)
        uk = u.astype(np.float64)
        res = xk - xk @ uk @ uk.T
        nrm = np.linalg.norm(xk, axis=1)
        offs.append(np.linalg.norm(res, axis=1) / np.maximum(nrm, 1e-30))
    off = np.concatenate(offs)
    RES["A_n_requests"] = int(X.shape[0])
    RES["A_out_of_range"] = int(oor.sum())
    RES["A_offspan_median"] = float(np.median(off))
    RES["A_offspan_max"] = float(off.max())
    print("requests                     %d  (out-of-range %d)" % (X.shape[0], oor.sum()))
    print("off-span fraction of STATE   median %.3e   max %.3e" % (np.median(off), off.max()))
    print("  round 6 measured on the real store: median 0.0 / max 1.96e-07")
    print("  (compose-physics/LOG.md 08:30Z, pca_tensor row)")
    ok = bool(off.max() < 1e-05)
    print("VERDICT  the addressed point is in span(G_k) to float32 rounding: %s" % ok)
    RES["A_pass"] = ok
    return X


def settle_span_coords(Z, Zreq, beta, steps, eta):
    """The DEPLOYED parameterisation: x_k = U_k z_k, so dist(x, span) == 0 by algebra.

    Energy in z: E(z) = (beta/2) ||z - z_req||^2.  The gradient at z_req is exactly zero
    in floating point, so the state does not move for ANY number of steps.
    """
    Z = [z.copy() for z in Z]
    for _ in range(int(steps)):
        Z = [z - eta * beta * (z - zr) for z, zr in zip(Z, Zreq)]
    return Z


def settle_channel_coords(X, A, P, beta, steps, eta):
    """The SAME energy, state carried in CHANNEL coordinates (the parameterisation the
    identity does NOT need, kept here to show what it costs)."""
    X = X.copy()
    for _ in range(int(steps)):
        G = X - X @ P + beta * (X - A)
        X = X - eta * G
    return X


def probe_B(store, U, ta, gv, vals, X, beta=1.0, eta=0.3):
    coeffs, _ = ta.coefficients(vals, gv)
    Zreq = [np.asarray(c, np.float32) for c in coeffs]
    static = render_span(store, Zreq, U)
    same = bool(np.array_equal(static, X))
    print(
        "static read == TensorAddress.read : %s   (np.array_equal, %d requests)"
        % (same, X.shape[0])
    )
    RES["B_static_matches_read"] = same
    rows = []
    for steps in (0, 1, 10, 1000):
        Z = settle_span_coords(Zreq, Zreq, beta, steps, eta)
        Xs = render_span(store, Z, U)
        eq = bool(np.array_equal(Xs, static))
        rows.append((steps, eq, float(np.abs(Xs - static).max())))
        print(
            "  span-coord settle steps=%5d   np.array_equal=%s   max|diff|=%.3e"
            % (steps, eq, rows[-1][2])
        )
    RES["B_span_coord"] = [{"steps": s, "array_equal": e, "max_abs_diff": d} for s, e, d in rows]
    RES["B_pass"] = bool(all((e for _, e, _ in rows)) and same)
    print(
        "VERDICT  in-well settle in span coordinates is a NO-OP at every step count: %s"
        % RES["B_pass"]
    )
    print("  round 6 asserted this in code at steps=0 only, on 256 requests, np.array_equal")
    print("  (compose-physics/LOG.md 08:32Z; arm.json zerostep_bit_identical_to_pca_tensor=true)")
    P = np.zeros((store.dim, store.dim), np.float32)
    for u, ix in zip(U, store.well_channels):
        P[np.ix_(ix, ix)] = (u @ u.T).astype(np.float32)
    rows2 = []
    for steps in (1, 10, 1000):
        Xc = settle_channel_coords(static, static, P, beta, steps, eta)
        d = float(np.abs(Xc - static).max() / max(np.abs(static).max(), 1e-30))
        rows2.append((steps, bool(np.array_equal(Xc, static)), d))
        print(
            "  channel-coord settle steps=%5d   np.array_equal=%s   rel max|diff|=%.3e"
            % (steps, rows2[-1][1], d)
        )
    RES["B2_channel_coord"] = [
        {"steps": s, "array_equal": e, "rel_max_abs_diff": d} for s, e, d in rows2
    ]
    print("VERDICT  in EXACT arithmetic both are identities; in float32 only the span")
    print("         parameterisation is BIT-exact.  The residual is the same 1e-7 the")
    print("         incumbent carries standing still, not motion.")
    return (static, P)


def probe_C(static, P, beta=1.0, eta=0.3, steps=4000, delta=0.1, seed=7):
    """Falsifier 1: break 'a in span(G_k)' and the settle starts to move."""
    rng = np.random.default_rng(seed)
    n, D = static.shape
    Q = rng.normal(size=(n, D)).astype(np.float32)
    Q = Q - Q @ P
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    A = static + np.float32(delta) * Q
    Xs = settle_channel_coords(A, A, P, beta, steps, eta)
    moved = np.linalg.norm(Xs - A, axis=1)
    pred = delta / (1.0 + beta)
    print("address pushed off-span by delta = %s" % delta)
    print("  measured settle displacement  median %.6f  max %.6f" % (np.median(moved), moved.max()))
    print("  closed-form prediction delta/(1+beta) = %.6f" % pred)
    RES["C_delta"] = delta
    RES["C_displacement_median"] = float(np.median(moved))
    RES["C_prediction"] = float(pred)
    RES["C_rel_err"] = float(abs(np.median(moved) - pred) / pred)
    ok = bool(RES["C_rel_err"] < 0.001)
    RES["C_pass"] = ok
    print("  relative error %.3e" % RES["C_rel_err"])
    print("VERDICT  the no-op is bought by 'a in span', NOT by settling being weak: %s" % ok)


def probe_D(store, U, static, beta=1.0, eta=0.05, steps=2000, sigma=1.0):
    disp = []
    for G, ix in zip(store.modes, store.well_channels):
        Ak = np.asarray(G, np.float64)
        x0 = static[:, ix].astype(np.float64)
        x = x0.copy()
        for _ in range(int(steps)):
            d2 = ((x[:, None, :] - Ak[None, :, :]) ** 2).sum(-1)
            w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * sigma**2))
            w = w / w.sum(1, keepdims=True)
            g_store = x - w @ Ak
            x = x - eta * (g_store + beta * (x - x0))
        disp.append(np.linalg.norm(x - x0, axis=1) / np.maximum(np.linalg.norm(x0, axis=1), 1e-30))
    d = np.concatenate(disp)
    RES["D_rel_displacement_median"] = float(np.median(d))
    RES["D_rel_displacement_max"] = float(d.max())
    print("atoms-as-POINTS energy, same request well, same beta:")
    print("  relative displacement  median %.4f   max %.4f" % (np.median(d), d.max()))
    ok = bool(np.median(d) > 0.001)
    RES["D_pass"] = ok
    print("VERDICT  the identity is a property of the BASIS energy, not of settling: %s" % ok)


def probe_E(ta, gv, vals, beta=1.0, seed=13):
    """Where dynamics CAN act: across wells.  z* = (beta I - lambda W)^-1 beta z_req."""
    rng = np.random.default_rng(seed)
    coeffs, _ = ta.coefficients(vals, gv)
    widths = [int(c.shape[1]) for c in coeffs]
    tot = int(sum(widths))
    Wb = np.zeros((tot, tot))
    off = np.cumsum([0] + widths)
    for k in range(len(widths)):
        for l in range(k + 1, len(widths)):
            B = rng.normal(size=(widths[k], widths[l]))
            B = B / np.linalg.norm(B, 2)
            Wb[off[k] : off[k + 1], off[l] : off[l + 1]] = B
            Wb[off[l] : off[l + 1], off[k] : off[k + 1]] = B.T
    Zreq = np.concatenate([np.asarray(c, np.float64) for c in coeffs], axis=1)
    out = []
    for lam in (0.0, 0.0001, 0.01, 0.05, 0.2):
        H = beta * np.eye(tot) - lam * Wb
        Zstar = beta * Zreq @ np.linalg.inv(H).T
        eig = np.linalg.eigvalsh(H)
        eta = 1.0 / max(eig.max(), 1e-12)
        Z = Zreq.copy()
        for _ in range(20000):
            Z = Z - eta * (Z @ H.T - beta * Zreq)
        gap = float(np.abs(Z - Zstar).max())
        rel = float(np.linalg.norm(Zstar - Zreq) / np.linalg.norm(Zreq))
        exact = bool(np.array_equal(Zstar, Zreq)) if lam == 0.0 else None
        out.append(
            {
                "lambda": lam,
                "min_eig_H": float(eig.min()),
                "rel_displacement": rel,
                "iter_vs_closed_form_max_diff": gap,
                "lambda0_exact_identity": exact,
            }
        )
        print(
            "  lambda=%-8g minEig(H)=%.4f  rel displacement=%.3e  |iter - closed form|=%.2e%s"
            % (lam, eig.min(), rel, gap, "   array_equal(z*, z_req)=True" if exact else "")
        )
    RES["E_rows"] = out
    RES["E_pass"] = bool(out[0]["rel_displacement"] == 0.0 and out[0]["lambda0_exact_identity"])
    print("VERDICT  at lambda = 0 the settled state IS the static read, exactly;")
    print("         every unit of motion in this store is CROSS-WELL.")


def probe_F(K=5, beta=1.0, lam_declared=1.0):
    S = (np.ones((K, K)) - np.eye(K)) * 0.5
    D = np.eye(K) * (beta / 2.0)
    Dm = np.linalg.inv(np.sqrt(D))
    M = Dm @ S @ Dm
    lam_star = 1.0 / max(abs(np.linalg.eigvalsh(M)).max(), 1e-300)
    min_eig = float(np.linalg.eigvalsh(D - lam_declared * S).min())
    print(
        "wells K=%d   pairs=%d   beta=%s   declared lambda=%s"
        % (K, K * (K - 1) // 2, beta, lam_declared)
    )
    print("  lambda*                   = %.8f   (arm.json: 0.24999999969559716)" % lam_star)
    print("  certificate min eigenvalue= %.8f   (arm.json: -1.5000000024352236)" % min_eig)
    print(
        "  lambda actually used      = 0.9 * lambda* = %.6f   (LOG 08:33Z: 0.225000)"
        % (0.9 * lam_star)
    )
    RES["F_lambda_star"] = float(lam_star)
    RES["F_min_eig"] = min_eig
    RES["F_pass"] = bool(abs(lam_star - 0.25) < 1e-09 and abs(min_eig + 1.5) < 1e-09)
    print("VERDICT  certificate reproduced from first principles: %s" % RES["F_pass"])


def main():
    store = build_store()
    U = span_basis(store)
    ranks = [int(u.shape[1]) for u in U]
    dims = [len(ix) for ix in store.well_channels]
    orth = max(
        (
            float(np.abs(u.T.astype(np.float64) @ u.astype(np.float64) - np.eye(u.shape[1])).max())
            for u in U
        )
    )
    RES["store"] = {
        "dims": dims,
        "span_ranks": ranks,
        "full_rank_wells": [i for i, (r, d) in enumerate(zip(ranks, dims)) if r == d],
        "max_orthonormality_residual": orth,
    }
    _banner("store")
    print(
        "well dims %s   span ranks %s   full-rank wells %s"
        % (dims, ranks, RES["store"]["full_rank_wells"])
    )
    print("max |U^T U - I| = %.3e   (span_basis returns float32)" % orth)
    ta, gv = build_address(store, U)
    rng = np.random.default_rng(3)
    vals = np.stack([rng.integers(0, c, 256) for c in ta.cardinalities], axis=1).astype(float)
    _banner("A -- the premise: the addressed point already lies in span(G_k)")
    X = probe_A(store, U, ta, gv, vals)
    _banner("B -- the claim: an in-well settle is a NO-OP at any step count")
    static, P = probe_B(store, U, ta, gv, vals, X)
    _banner("C -- falsifier 1: put the address off the span and the settle moves")
    probe_C(static, P)
    _banner("D -- falsifier 2: atoms as POINTS instead of a basis, and the settle moves")
    probe_D(store, U, static)
    _banner("E -- where dynamics CAN act: across wells")
    probe_E(ta, gv, vals)
    _banner("F -- the coercivity certificate, from first principles")
    probe_F()
    _banner("SUMMARY")
    keys = [k for k in RES if k.endswith("_pass")]
    for k in keys:
        print("  %-12s %s" % (k, RES[k]))
    allp = all((RES[k] for k in keys))
    print("  ALL PASS   %s" % allp)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_01_zero_dynamics.json")
    with open(out, "w") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True)
    print("  wrote %s" % out)
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
