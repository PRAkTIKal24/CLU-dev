"""w6 hub/compose-physics -- THE DIAL: does SETTLING add anything to composition?

What this module is, in one sentence
------------------------------------
It takes ``pca_tensor`` -- wave 5's result, the per-well leading principal subspace read
by the product address book, which beats ``mlp_180k`` on rows 3 and 4 on all three seeds
-- **holds its content and its address book fixed, byte for byte**, and replaces only
the last step of the read: instead of *emitting* the addressed point, it **settles** a
state in the landscape those same atoms shape and reads the settled state.

Why the arm is shaped the way it is
-----------------------------------
Two measurements made before any code was written (LOG 06:53Z, 06:56Z) fix the design.

1. ⭐ **The incumbent's atoms are a BASIS, not a point cloud.**  Measured: per well, one
   mean vector (norm 7.9-18.2) plus 63 mutually **orthonormal** directions -- the
   nearest-neighbour atom distance is 1.4142 = sqrt(2) in every well, the signature of
   orthonormal unit vectors.
2. ⛔ **An in-well settle on such content is an ALGEBRAIC IDENTITY.**  The honest store
   energy for a basis is distance to its span; the request enters as a basin at the
   address book's point; and ``TensorAddress.read`` renders through the orthonormal basis
   of that very span, so the addressed point **already lies at the joint minimum**.  The
   settled state equals the static read for reasons of geometry, not of measurement.

So the only place dynamics can act on this content is **across wells**, and the arm is a
cross-well relaxation:

    E(z) = (beta/2) sum_k || z~_k - z~_k^req ||^2  -  lambda sum_{k<l} z~_k^T W_kl z~_l

with the state parameterised **in the span** (``x_k = U_k z_k``), which *is* the store's
own energy enforced exactly rather than penalised, and which is also what makes 49
corners times S settle steps affordable: the state is ~312-dimensional, not 2048.

``W_kl`` is the programme's own bilinear low-rank coupling
(:func:`chlu.core.gen_landscape.fit_low_rank_coupling`) re-derived in coefficient space:
the standardised cross-covariance of the **fit pool's** span coefficients, top-``r`` SVD,
each pair normalised by its own leading singular value.  ⛔ Label-free, pool-only, and
no held-out row enters it.

⚠ The bilinear form is **indefinite**, so the coercivity certificate is mandatory and not
optional -- ``gen_landscape.coupling_stability`` records a real uncertified chain running
to ``E = -1.6e28``.  In coefficient space ``D = (beta/2) I`` and ``S_kl = ||W_kl||_2 / 2``,
so ``lambda* = 1 / rho(D^-1/2 S D^-1/2)``; the strength actually used is
``min(declared, 0.9 lambda*)`` and the clip is **reported whenever it binds**.

⭐⭐ How the dial is isolated so the answer cannot be confounded by parameters
-----------------------------------------------------------------------------
The coupling is fitted, so it carries declared read-side bytes beside the address book --
the same rule w5 applied to ``address_bytes``.  ⛔ **The store stays at 524,288 B and is
not touched.**  But a margin of the settled arm over ``pca_tensor`` alone would still be
ambiguous: coupling parameters, or dynamics?  So the hub's dial is measured by a
**parameter-identical** contrast, three legs off ONE fitted coupling:

* ``pca_settle_zerostep`` -- 0 settle steps.  ⭐ Must be **bit-identical** to
  ``pca_tensor``; asserted in code, not hoped for.  It proves the harness change is a
  no-op so any difference elsewhere is the dynamics.
* ``pca_settle_onestep``  -- **one** gradient step, same ``(beta, lambda, W)``.
* ``pca_settle``          -- settled to convergence, same ``(beta, lambda, W)``.

⭐ ``pca_settle`` minus ``pca_settle_onestep`` is **the relaxation**, at identical
parameters and identical bytes.  That contrast *is* this hub's dial and nothing else can
explain it.

⚠ A property of this landscape, declared in advance because it is a RESULT
--------------------------------------------------------------------------
A quadratic request basin plus a bilinear coupling is a quadratic energy, so its settled
state is the closed-form linear map ``z~* = (beta I - lambda W)^-1 beta z~^req``.
**On content that is a basis, the landscape is necessarily quadratic, and "settling" is
therefore an iterative solver for a linear operator.**  ⛔ That is not a strawman choice;
it is what the store *is*.  Any non-convexity would have to come from somewhere other
than the deployed atoms, which would change the content -- and wave 4 already spent a hub
learning not to move two dials at once.

The one non-convex, genuinely store-grounded alternative is the atom-**kernel** energy in
channel space (``AtomLandscape``), and charter section E.4 has **already banked its
null** (density-over-atoms descent, launch 0.685 -> settled 0.211).  ⛔ No scored run is
spent on it.  ⭐ ``pca_settle_kernel`` runs it on **val only**, so the banked null is
confirmed on *this* content rather than asserted.

Nothing is edited on disk
-------------------------
``exp_gen_read`` and ``exp_w5_compose_field`` are extended at runtime exactly as w5
extended wave 3: this module appends to ``ARMS``, updates the per-arm dicts and wraps
``build_arm``/``arm_features``.  ``install()`` asserts every pre-existing arm's store,
verb and class are unchanged, so no rival's and no control's code path moves.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import exp_gen_factor_wells as fw
from . import exp_gen_read as R
from . import exp_w5_compose_field as W5
from ..core.gen_read import render_span, span_basis

# ---------------------------------------------------------------------------
# 0. The pins this hub owns.  Selection values are chosen on `val` and nowhere else.
# ---------------------------------------------------------------------------

W6_PINS = {
    # ⭐ THE CONTENT IS NOT A DIAL HERE.  It is the incumbent's own object, built by the
    # incumbent's own function, and the atom block is fingerprinted against it.
    "content_from": "exp_w5_compose_field.build_pca_content (the incumbent's own object)",
    "address_from": ("the incumbent's own cached TensorAddress pickle -- the SAME cache "
                     "key, so the file w5 fitted is reused, not a refit of it"),
    "store_bytes": int(W5.W5_PINS["store_bytes"]),
    # -- the settle, selected on val --------------------------------------
    "coupling_rank": 16,          # r in W_kl; the coupling's whole capacity dial
    "request_strength": 1.0,      # beta -- the request basin's stiffness
    "coupling_strength": 1.0,     # lambda DECLARED; the certificate may clip it
    "safety_frac": 0.9,           # the certificate's margin, as gen_landscape uses
    "settle_steps": 200,          # S -- "to convergence"; convergence is REPORTED
    "step_size": 0.25,            # eta
    "settle_tol": 1e-6,           # relative displacement at which the chain is settled
    # -- the val-only kernel leg (the banked null, confirmed not asserted) --
    "kernel_steps": 200,
    "kernel_step_size": 0.05,
    "kernel_sigma_rule": "nn_median over the atom rows (gen_landscape.atom_bandwidths)",
}

#: ⭐ THE ARM and the two legs that isolate its dial, plus the charter-C family.
PHYS_ARMS = (
    "pca_settle",              # THE ARM: cross-well relaxation to convergence
    "pca_settle_onestep",      # ABLATION: ONE step, identical (beta, lambda, W)
    "pca_settle_zerostep",     # ABLATION: 0 steps -- asserted BIT-IDENTICAL to pca_tensor
    "pca_settle_removed",      # CONTROL (C.2): atoms zeroed, must collapse
    "pca_settle_shuffled",     # CONTROL (C.2): atom VALUES permuted, must collapse
    "pca_settle_lookup",       # FLOOR (C.1/C.4): nearest atom to the SETTLED state
    "pca_settle_kernel",       # val-only: the banked density-descent null, on THIS content
)

#: ⭐ VAL-ONLY SELECTION LEGS.  ``arm -> (fraction of the certified-safe lambda, steps)``.
#:
#: ⛔ Why these exist, stated so nobody mistakes them for results.  The first val run
#: showed the settled read LOSING monotonically in the amount of relaxation.  A negative
#: result on a badly-tuned arm is not a measurement of the dynamics, it is a measurement
#: of my parameter choice -- so the whole ``(lambda, steps)`` family is swept **on val**,
#: which charter A.2 makes free, and the winner is what the PREREG then freezes.
#: ⭐ The family is continuous in ``lambda``: at ``lambda = 0`` the settle is the identity
#: and the arm IS ``pca_tensor``, so this sweep contains the incumbent as its own limit.
#: ⛔ None of these is ever scored or quoted on the held-out set.
SWEEP_ARMS = {
    "pca_settle_f01_conv": (0.01, "converge"),
    "pca_settle_f05_conv": (0.05, "converge"),
    "pca_settle_f10_conv": (0.10, "converge"),
    "pca_settle_f25_conv": (0.25, "converge"),
    "pca_settle_f50_conv": (0.50, "converge"),
    "pca_settle_f100_s3": (1.00, 3),
    "pca_settle_f100_s10": (1.00, 10),
    "pca_settle_f25_s3": (0.25, 3),
}

#: ⭐ VAL-ONLY: the coupling's CAPACITY, swept separately.  ``arm -> coupling rank``.
#: ⛔ This exists to remove the last "you built the coupling wrong" objection: if the
#: settled read loses at rank 4, 16 and 64 alike, the loss is not a capacity artefact.
RANK_SWEEP = {"pca_settle_r4": 4, "pca_settle_r64": 64}

PHYS_ARMS = PHYS_ARMS + tuple(SWEEP_ARMS) + tuple(RANK_SWEEP)

PHYS_ARM_CLASS = {
    "pca_settle": "ARM (the PHYSICS read: settled state, cross-well relaxation)",
    "pca_settle_onestep": "ABLATION (ONE settle step; identical beta, lambda, W)",
    "pca_settle_zerostep": "ABLATION (ZERO settle steps; asserted identical to pca_tensor)",
    "pca_settle_removed": "CONTROL (charter C.2, atoms zeroed)",
    "pca_settle_shuffled": "CONTROL (charter C.2, atom values permuted)",
    "pca_settle_lookup": "FLOOR (charter C.1/C.4, nearest atom to the settled state)",
    "pca_settle_kernel": "VAL-ONLY (the BANKED density-over-atoms null, charter E.4)",
}

#: ``(perturbation, n_settle_steps)``.  ``None`` = the arm's own unperturbed atoms.
SETTLE_SPEC = {
    "pca_settle": (None, "converge"),
    "pca_settle_onestep": (None, 1),
    "pca_settle_zerostep": (None, 0),
    "pca_settle_removed": ("zero", "converge"),
    "pca_settle_shuffled": ("value_permute", "converge"),
    "pca_settle_lookup": (None, "converge"),
    "pca_settle_kernel": (None, "kernel"),
}
SETTLE_SPEC.update({a: (None, s) for a, (_f, s) in SWEEP_ARMS.items()})
SETTLE_SPEC.update({a: (None, "converge") for a in RANK_SWEEP})
PHYS_ARM_CLASS.update({
    a: (f"VAL-ONLY SELECTION (lambda = {f:g} x certified-safe, steps = {s})")
    for a, (f, s) in SWEEP_ARMS.items()})
#: ⚠ ONE sweep leg is PROMOTED into the scored ladder by
#: PREREG-w6-compose-physics-1.md section 2: the gentlest NON-TRIVIAL dynamics, which is
#: the family's strongest case for a TIE. ⛔ Relabelled so no scored row is ever
#: printed carrying the words "VAL-ONLY".
PHYS_ARM_CLASS["pca_settle_f05_conv"] = (
    "ABLATION (the gentlest dynamics: lambda = 0.05 x certified-safe, converged)")
PHYS_ARM_CLASS.update({
    a: f"VAL-ONLY SELECTION (coupling rank {r}, lambda = certified-safe, converged)"
    for a, r in RANK_SWEEP.items()})


# ---------------------------------------------------------------------------
# 1. The coupling, in COEFFICIENT space
# ---------------------------------------------------------------------------


class SpanCoupling:
    """``E_couple(z) = - sum_{k<l} z~_k^T W_kl z~_l`` at rank ``r``, in span coordinates.

    ⭐ The same estimator ``gen_landscape.fit_low_rank_coupling`` uses, moved from
    channel space to the span's coordinates: for each pair the standardised
    cross-covariance ``C = (1/n) Z~_k^T Z~_l``, its top-``r`` SVD ``C ~= U S V^T``, stored
    as ``U_kl = S^{1/2} U^T`` and ``V_kl = S^{1/2} V^T`` with each pair divided by its own
    leading singular value so ``lambda`` is dimensionless and one wide well cannot
    silently own the whole term.

    ⛔ **Fitted on the fit pool's coefficients only.**  No label, no held-out row.
    """

    def __init__(self, Z: Sequence[np.ndarray], rank: int):
        self.rank = int(rank)
        self.K = len(Z)
        self.centres = [np.asarray(z, np.float64).mean(axis=0) for z in Z]
        self.radii = [
            np.maximum(np.asarray(z, np.float64).std(axis=0), 1e-6) for z in Z]
        B = [((np.asarray(z, np.float64) - c[None, :]) / s[None, :])
             for z, c, s in zip(Z, self.centres, self.radii)]
        n = max(int(B[0].shape[0]), 1)
        self.pairs, self.u, self.v = [], [], []
        for k in range(self.K):
            for lo in range(k + 1, self.K):
                C = (B[k].T @ B[lo]) / n
                rr = int(min(self.rank, min(C.shape)))
                U, S, Vt = np.linalg.svd(C, full_matrices=False)
                U, S, Vt = U[:, :rr], S[:rr], Vt[:rr]
                s0 = float(S[0]) if S.size and S[0] > 0 else 1.0
                sc = np.sqrt(S / s0)
                Uk = (U * sc[None, :]).T                 # (rr, r_k)
                Vl = (Vt * sc[:, None])                  # (rr, r_l)
                if rr < self.rank:                       # pad so the ledger is exact
                    Uk = np.pad(Uk, ((0, self.rank - rr), (0, 0)))
                    Vl = np.pad(Vl, ((0, self.rank - rr), (0, 0)))
                self.pairs.append((k, lo))
                self.u.append(np.ascontiguousarray(Uk.astype(np.float32)))
                self.v.append(np.ascontiguousarray(Vl.astype(np.float32)))

    # -- the ledger -------------------------------------------------------
    def byte_ledger(self) -> dict:
        b = int(sum(x.size for x in self.u + self.v) * 4)
        std = int(sum(c.size + s.size
                      for c, s in zip(self.centres, self.radii)) * 4)
        return {
            "coupling_bytes": b,
            "rank": int(self.rank),
            "n_pairs": len(self.pairs),
            "standardisation_bytes": std,
            "total_read_side_bytes_added": b + std,
            "dtype": "float32",
            "bytes_formula": "4 * sum_{k<l} r * (r_k + r_l)",
            "note": ("EXTRA to the store's 524,288 B and reported BESIDE it, never "
                     "netted into it -- the same rule w5 stated for address_bytes. "
                     "⛔ The standardisation is NOT declared free: it is counted and "
                     "printed, so a reader can add it without recomputing anything."),
        }

    # -- the physics ------------------------------------------------------
    def spectral_norms(self) -> np.ndarray:
        """``||W_kl||_2`` per pair, on ``r x r`` matrices (``gen_landscape``'s trick)."""
        out = []
        for U, V in zip(self.u, self.v):
            Un, Vn = np.asarray(U, np.float64), np.asarray(V, np.float64)
            ev = np.linalg.eigvals((Un @ Un.T) @ (Vn @ Vn.T))
            out.append(float(np.sqrt(max(float(np.max(ev.real)), 0.0))))
        return np.asarray(out, np.float64)

    def certificate(self, beta: float, lam: float) -> dict:
        """⛔⭐ Is ``E`` bounded below?  The same argument, in span coordinates.

        ``E >= u^T C u - const`` with ``C_kk = beta/2`` and ``C_kl = -lambda ||W_kl||_2/2``
        where ``u_k = ||z~_k||``.  Coercive iff ``C >= 0``; with ``D = (beta/2) I`` the
        largest safe strength is ``lambda* = 1/rho(D^-1/2 S D^-1/2)`` in closed form.
        ⛔ A diverged chain's energy trace is not a measurement of anything.
        """
        norms = self.spectral_norms()
        S = np.zeros((self.K, self.K), np.float64)
        for (k, lo), w in zip(self.pairs, norms):
            S[k, lo] = S[lo, k] = float(w) / 2.0
        d = np.full(self.K, float(beta) / 2.0)
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-300))
        rho = float(np.max(np.abs(np.linalg.eigvalsh(dinv[:, None] * S * dinv[None, :]))))
        lam_star = float("inf") if rho <= 0 else 1.0 / rho
        C = np.diag(d) - float(lam) * S
        min_eig = float(np.min(np.linalg.eigvalsh(C)))
        return {
            "request_strength_beta": float(beta),
            "coupling_strength_declared": float(lam),
            "pair_spectral_norms": [float(x) for x in norms],
            "max_safe_coupling_strength": lam_star,
            "certificate_min_eigenvalue": min_eig,
            "bounded_below": bool(min_eig >= 0.0),
            "rule": ("E >= u^T C u - const with C_kk = beta/2 and "
                     "C_kl = -lambda ||W_kl||_2 / 2; coercive iff C >= 0"),
        }

    def safe_strength(self, beta: float, declared: float, frac: float) -> dict:
        cert = self.certificate(beta, declared)
        lam_star = float(cert["max_safe_coupling_strength"])
        cap = lam_star * float(frac)
        used = float(min(float(declared), cap)) if np.isfinite(cap) else float(declared)
        return {"coupling_strength_used": used,
                "certificate_bound_the_strength": bool(used < float(declared)),
                "safety_frac": float(frac), "certificate": cert}

    def field(self, Zt: Sequence[np.ndarray]) -> list:
        """``[ sum_{l != k} W_kl z~_l ]`` -- minus the gradient of ``E_couple``."""
        out = [np.zeros_like(z) for z in Zt]
        for (k, lo), U, V in zip(self.pairs, self.u, self.v):
            Un = np.asarray(U, np.float32)
            Vn = np.asarray(V, np.float32)
            out[k] += (Zt[lo] @ Vn.T) @ Un          # W_kl  z~_l
            out[lo] += (Zt[k] @ Un.T) @ Vn          # W_kl^T z~_k
        return out


# ---------------------------------------------------------------------------
# 2. The settle
# ---------------------------------------------------------------------------


def settle_coefficients(cpl: SpanCoupling, Zreq: Sequence[np.ndarray], *,
                        beta: float, lam: float, steps, step_size: float,
                        tol: float) -> tuple:
    """``z^req -> z*`` by gradient descent on ``E``.  Returns ``(z*, trace)``.

    ⛔ **There is no feature argument and no row index on this path.**  Its inputs are the
    address book's own coefficients, the coupling and three declared floats.

    ``steps = 0`` returns the launch **unchanged and by identity** -- that is what makes
    ``pca_settle_zerostep`` a bit-identical reproduction of ``pca_tensor`` rather than a
    numerically-close one.
    """
    cen = [np.asarray(c, np.float32) for c in cpl.centres]
    rad = [np.asarray(s, np.float32) for s in cpl.radii]
    n_steps = 0 if steps == 0 else (int(W6_PINS["settle_steps"])
                                    if steps == "converge" else int(steps))
    trace = {"n_steps_requested": n_steps, "displacement": [], "energy": []}
    if n_steps == 0:
        trace.update(converged=True, n_steps_taken=0, max_rel_displacement=0.0,
                     final_rel_displacement=0.0)
        return [np.asarray(z, np.float32) for z in Zreq], trace

    Zt0 = [((np.asarray(z, np.float32) - c[None, :]) / s[None, :])
           for z, c, s in zip(Zreq, cen, rad)]
    Zt = [z.copy() for z in Zt0]
    scale0 = float(max(np.sqrt(sum(float((z ** 2).sum()) for z in Zt0)), 1e-12))
    taken = 0
    for i in range(n_steps):
        fld = cpl.field(Zt)
        grad = [beta * (z - z0) - lam * f for z, z0, f in zip(Zt, Zt0, fld)]
        Zn = [z - float(step_size) * g for z, g in zip(Zt, grad)]
        disp = float(np.sqrt(sum(float(((a - b) ** 2).sum())
                                 for a, b in zip(Zn, Zt)))) / scale0
        e_req = 0.5 * beta * sum(float(((z - z0) ** 2).sum())
                                 for z, z0 in zip(Zn, Zt0))
        e_cpl = -lam * 0.5 * sum(float((z * f).sum())
                                 for z, f in zip(Zn, cpl.field(Zn)))
        Zt = Zn
        taken = i + 1
        trace["displacement"].append(disp)
        trace["energy"].append(float(e_req + e_cpl) / max(int(Zt[0].shape[0]), 1))
        if not np.isfinite(trace["energy"][-1]):
            trace["diverged"] = True
            break
        if disp < float(tol):
            break
    trace["n_steps_taken"] = taken
    trace["converged"] = bool(trace["displacement"]
                              and trace["displacement"][-1] < float(tol))
    trace["max_rel_displacement"] = float(max(trace["displacement"])
                                          if trace["displacement"] else 0.0)
    trace["final_rel_displacement"] = float(trace["displacement"][-1]
                                            if trace["displacement"] else 0.0)
    # keep the trace small on disk: the shape of the descent, not every step
    for key in ("displacement", "energy"):
        t = trace[key]
        trace[key] = [float(x) for x in (t[:20] + t[-20:] if len(t) > 40 else t)]
    return [(z * s[None, :] + c[None, :]).astype(np.float32)
            for z, c, s in zip(Zt, cen, rad)], trace


def settle_kernel_channels(store, X0: np.ndarray, *, steps: int,
                           step_size: float, sigma) -> tuple:
    """⛔ The BANKED null (charter E.4), run on THIS content for confirmation only.

    Descent on the programme's own atom-kernel energy
    ``E_k(x) = -sigma_k^2 logsumexp_i(-||x_k - a_ki||^2 / 2 sigma_k^2)`` in CHANNEL space,
    launched from the incumbent's static read.  ⛔ Never scored; ``val`` only.
    """
    X = np.asarray(X0, np.float32).copy()
    idxs = [np.asarray(w, np.int64) for w in store.well_channels]
    trace = {"displacement": []}
    for _ in range(int(steps)):
        step_disp = 0.0
        for idx, G, s in zip(idxs, store.modes, np.asarray(sigma, np.float64)):
            A = np.asarray(G, np.float32)
            z = X[:, idx]
            d2 = (np.sum(z * z, 1)[:, None] - 2.0 * (z @ A.T)
                  + np.sum(A * A, 1)[None, :])
            lk = -np.maximum(d2, 0.0) / (2.0 * float(s) ** 2)
            p = np.exp(lk - lk.max(1, keepdims=True))
            p /= np.maximum(p.sum(1, keepdims=True), 1e-30)
            grad = z - (p @ A)                      # d/dx of the kernel free energy
            X[:, idx] = z - float(step_size) * grad
            step_disp += float((grad ** 2).sum())
        trace["displacement"].append(float(np.sqrt(step_disp)))
    trace["displacement"] = [float(x) for x in trace["displacement"][:40]]
    return X, trace


# ---------------------------------------------------------------------------
# 3. Diagnostics the charter asks for by name
# ---------------------------------------------------------------------------


def atom_nn_spacing(store) -> list:
    """Median nearest-neighbour distance between the DEPLOYED atoms, per well."""
    out = []
    for G in store.modes:
        A = np.asarray(G, np.float64)
        d2 = (np.sum(A * A, 1)[:, None] - 2.0 * (A @ A.T) + np.sum(A * A, 1)[None, :])
        np.fill_diagonal(d2, np.inf)
        out.append(float(np.median(np.sqrt(np.maximum(d2.min(1), 0.0)))))
    return out


def nearest_atom_report(store, coeffs: Sequence[np.ndarray], basis) -> dict:
    """⭐ charter C.4: **is the settled state sitting on its nearest atom?**

    Reported per well as the distance from the settled point to the nearest atom, and --
    the number that actually decides it -- that distance **relative to the spacing of the
    atoms themselves**.  A ratio near 0 is a lookup table and is reported as one.
    """
    out = {}
    spacings = atom_nn_spacing(store)
    for k, (C, Uk, G) in enumerate(zip(coeffs, basis, store.modes)):
        pts = np.asarray(C, np.float32) @ np.asarray(Uk, np.float32).T   # (N, d_k)
        A = np.asarray(G, np.float32)
        d2 = (np.sum(pts * pts, 1)[:, None] - 2.0 * (pts @ A.T)
              + np.sum(A * A, 1)[None, :])
        dmin = np.sqrt(np.maximum(d2.min(1), 0.0))
        out[f"well_{k}"] = {
            "median_distance_to_nearest_atom": float(np.median(dmin)),
            "median_atom_nn_spacing": float(spacings[k]),
            "ratio": float(np.median(dmin) / max(float(spacings[k]), 1e-12)),
        }
    r = [v["ratio"] for v in out.values()]
    out["min_ratio_over_wells"] = float(min(r))
    out["verdict"] = ("LOOKUP (charter C.4): the settled state sits on its nearest atom"
                      if min(r) < 0.05 else
                      "NOT a lookup: the settled state sits strictly between atoms")
    return out


# ---------------------------------------------------------------------------
# 4. Building the arm
# ---------------------------------------------------------------------------


def content_sha(store) -> str:
    """sha256 of the DEPLOYED atom block -- the fingerprint that proves content is held."""
    import hashlib
    h = hashlib.sha256()
    for M in store.modes:
        h.update(np.ascontiguousarray(np.asarray(M, np.float32)).tobytes())
    return h.hexdigest()


def build_coupling(out_dir: Path, store, basis, log: Callable = print,
                   rank: int = None) -> dict:
    """Fit the cross-well coupling on the FIT POOL's span coefficients.  Resumable."""
    out_dir = Path(out_dir)
    rank = int(W6_PINS["coupling_rank"]) if rank is None else int(rank)
    stem = f"span_coupling_r{rank}"
    cache = out_dir / f"{stem}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            obj = pickle.load(fh)
        log(f"  coupling: reused {cache.name}")
        return obj
    pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    pool = pool[: int(R.READ_PINS["n_addr_train"])]
    t0 = time.time()
    # ⚠ span_coefficients TAKES FEATURES.  That is FIT TIME, exactly as the address
    # book's own targets are built from features (exp_gen_read._pool_span_targets).
    # ⛔ It is never called on the read path; assert_read_request_only proves that.
    _Y, T = R._pool_span_targets(store, pool, basis, log=log)
    cpl = SpanCoupling(T, rank)
    safe = cpl.safe_strength(float(W6_PINS["request_strength"]),
                             float(W6_PINS["coupling_strength"]),
                             float(W6_PINS["safety_frac"]))
    obj = {"coupling": cpl, "byte_ledger": cpl.byte_ledger(), "safe": safe,
           "n_pool_rows": int(pool.size), "span_ranks": [int(t.shape[1]) for t in T],
           "fit_seconds": time.time() - t0,
           "label_use": ("NONE -- the coupling is an unsupervised second moment of the "
                         "fit pool's own span coefficients")}
    with open(cache, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump({k: v for k, v in obj.items() if k != "coupling"}, out_dir / f"{stem}.json")
    log(f"  coupling: rank {cpl.rank}, {len(cpl.pairs)} pairs, "
        f"{obj['byte_ledger']['coupling_bytes']:,} B, lambda_used="
        f"{safe['coupling_strength_used']:.6f} "
        f"(declared {W6_PINS['coupling_strength']}, lambda*="
        f"{safe['certificate']['max_safe_coupling_strength']:.6f}, "
        f"clipped={safe['certificate_bound_the_strength']}) in {obj['fit_seconds']:.1f}s")
    return obj


def build_arm(arm: str, out_dir: Path, log: Callable = print) -> dict:
    """Dispatch: this hub's settle legs here, everything else down the w5/w3 chain."""
    if arm not in PHYS_ARMS:
        return _ORIG_BUILD_ARM(arm, out_dir, log=log)

    out_dir = Path(out_dir)
    arm_dir = out_dir / "arms" / arm
    done = arm_dir / "arm.pkl"
    if done.exists():
        log(f"  arm {arm}: already fitted, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    t0 = time.time()
    from ..core.gen_read import TensorAddress, assert_read_request_only

    cond = fw.load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]

    # ⭐ THE CONTENT, UNCHANGED.  The incumbent's own builder, the incumbent's own object.
    base = W5.build_pca_content(out_dir, log=log)
    store = base["store"]
    basis = span_basis(store)
    sha = content_sha(store)

    # ⭐ THE ADDRESS BOOK, UNCHANGED.  The cache key is byte-for-byte the one
    # _build_pca_arm writes, so the file w5 fitted is REUSED and not refitted.
    cache = out_dir / (
        f"address_pca_prod_R{int(R.READ_PINS['rank'])}_lam{R.READ_PINS['ridge']}"
        f"_sw{int(R.READ_PINS['sweeps'])}_n{int(R.READ_PINS['n_addr_train'])}.pkl")
    if cache.exists():
        with open(cache, "rb") as fh:
            addr = pickle.load(fh)
        log(f"  address book: reused the INCUMBENT's own {cache.name}")
        reused = True
    else:
        pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
        pool = pool[: int(R.READ_PINS["n_addr_train"])]
        Y, T = R._pool_span_targets(store, pool, basis, log=log)
        ntr = int(0.9 * Y.shape[0])
        addr = TensorAddress(cards, rank=int(R.READ_PINS["rank"]),
                             ridge=float(R.READ_PINS["ridge"]),
                             seed=int(R.READ_PINS["address_seed"]))
        addr.multiplicative = True
        addr.fit(Y[:ntr], [t[:ntr] for t in T], sweeps=int(R.READ_PINS["sweeps"]),
                 log=log, Y_val=Y[ntr:], targets_val=[t[ntr:] for t in T])
        with open(cache, "wb") as fh:
            pickle.dump(addr, fh)
        reused = False

    kind, steps = SETTLE_SPEC[arm]
    cpl_obj = build_coupling(out_dir, store, basis, log=log,
                             rank=RANK_SWEEP.get(arm))

    obj = {
        "arm": arm, "read": "settle", "kind": "span",
        "store": store, "basis": basis, "address": addr,
        "coupling": cpl_obj["coupling"],
        "settle_steps": steps,
        "beta": float(W6_PINS["request_strength"]),
        # ⭐ The certified-safe strength, scaled by this leg's declared fraction.  A
        # sweep leg only ever scales DOWN, so the coercivity certificate still holds.
        "lam": float(cpl_obj["safe"]["coupling_strength_used"])
               * float(SWEEP_ARMS.get(arm, (1.0, None))[0]),
        "lam_fraction_of_certified_safe": float(SWEEP_ARMS.get(arm, (1.0, None))[0]),
        "step_size": float(W6_PINS["step_size"]),
        "tol": float(W6_PINS["settle_tol"]),
        "byte_ledger": dict(base["byte_ledger"]),
        "address_bytes": addr.byte_ledger(),
        "coupling_bytes": dict(cpl_obj["byte_ledger"]),
        "coupling_certificate": cpl_obj["safe"],
        "head_bytes": {"head_bytes": 0,
                       "note": ("this read's address book IS the tensor book; the "
                                "coupling is declared beside it and never netted in")},
        "partition": base["partition"], "label_use": base["label_use"],
        "content": ("per-well principal subspace of the fit pool (unsupervised) -- "
                    "the INCUMBENT's content, unchanged"),
        "content_sha256": sha,
        "address_book_reused_from_incumbent": bool(reused),
        "atom_nn_spacing": atom_nn_spacing(store),
        "protos": R._FitFingerprint(list(store.modes)),
    }

    # charter B.3, on the deployed object: the read is reachable from the REQUEST alone.
    pool0 = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    vals0 = grid.values_of(pool0[:64])
    obj["request_only"] = assert_read_request_only(
        addr, store, vals0, grid.values, basis, rng_seed=int(fw.PINS["seed"]))

    if kind is not None:
        obj["store"] = W5.perturb_modes(store, kind, int(fw.PINS["seed"]))
        obj["perturbation"] = {"kind": kind, "when": "after the address book was fitted"}
        if kind == "zero":
            obj["zeroed"] = True
            obj["basis"] = basis
        else:
            obj["basis"] = W5.basis_at_fixed_rank(
                obj["store"], [np.asarray(U).shape[1] for U in basis])

    led = obj["store"].byte_ledger() if hasattr(obj["store"], "byte_ledger") else None
    if led is not None and not led["matches_budget"]:
        raise AssertionError(f"{arm} deploys {led['deployed_bytes']} B, not the pinned "
                             f"{W6_PINS['store_bytes']} B")

    # ⭐⭐ THE ASSERTION THAT MAKES THE DIAL CLEAN.  Zero settle steps must reproduce the
    # incumbent read BIT-IDENTICALLY -- not "closely".  If it does not, the harness moved
    # something other than the dynamics and every margin on this hub is uninterpretable.
    if arm == "pca_settle_zerostep":
        vchk = grid.values_of(pool0[:256])
        got, _ = _settled_features(obj, vchk, grid)
        want, _ = addr.read(store, vchk, grid.values, basis)
        if not np.array_equal(np.asarray(got, np.float32),
                              np.asarray(want, np.float32)):
            raise AssertionError(
                "pca_settle_zerostep is NOT bit-identical to the incumbent read -- "
                "the harness moved something other than the settle")
        obj["zerostep_bit_identical_to_pca_tensor"] = True
        log("  arm pca_settle_zerostep: BIT-IDENTICAL to the incumbent read on 256 "
            "requests -- the harness change is a proven no-op")

    obj["fit_seconds"] = time.time() - t0
    arm_dir.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    fw._jdump({k: v for k, v in obj.items()
               if k not in ("store", "basis", "address", "coupling", "protos")},
              arm_dir / "arm.json")
    log(f"  arm {arm}: fitted in {obj['fit_seconds']:.1f}s; store "
        f"{obj['byte_ledger']['deployed_bytes']:,} B + address "
        f"{obj['address_bytes'].get('address_bytes', 0):,} B + coupling "
        f"{obj['coupling_bytes']['coupling_bytes']:,} B")
    return obj


# ---------------------------------------------------------------------------
# 5. The read
# ---------------------------------------------------------------------------

#: Filled by the read so the launcher can print the diagnostics the charter asks for
#: without a second pass over the corners.
DIAG: dict = {}


def _settled_features(obj: dict, values: np.ndarray, grid) -> tuple:
    arm = obj["arm"]
    store, basis, addr = obj["store"], obj["basis"], obj["address"]
    if obj.get("zeroed"):
        n = int(np.asarray(values).shape[0])
        return (np.zeros((n, int(store.dim)), np.float32), np.zeros(n, bool))
    Zreq, oor = addr.coefficients(values, grid.values)
    _kind, steps = SETTLE_SPEC[arm]

    if steps == "kernel":
        X0 = render_span(store, Zreq, basis)
        sig = atom_nn_spacing(store)
        X, tr = settle_kernel_channels(
            store, X0, steps=int(W6_PINS["kernel_steps"]),
            step_size=float(W6_PINS["kernel_step_size"]), sigma=np.asarray(sig))
        d = DIAG.setdefault(arm, {"n_calls": 0, "trace": tr, "sigma": sig,
                                  "launch_to_settled_l2": []})
        d["n_calls"] += 1
        d["launch_to_settled_l2"].append(
            float(np.median(np.linalg.norm(X - X0, axis=1))))
        return X, oor

    Zs, tr = settle_coefficients(
        obj["coupling"], Zreq, beta=float(obj["beta"]), lam=float(obj["lam"]),
        steps=steps, step_size=float(obj["step_size"]), tol=float(obj["tol"]))

    d = DIAG.setdefault(arm, {"n_calls": 0, "traces": [], "nearest_atom": None,
                              "launch_to_settled_rel": []})
    d["n_calls"] += 1
    if len(d["traces"]) < 3:
        d["traces"].append(tr)
    num = float(np.sqrt(sum(float(((a - b) ** 2).sum()) for a, b in zip(Zs, Zreq))))
    den = float(max(np.sqrt(sum(float((b ** 2).sum()) for b in Zreq)), 1e-12))
    d["launch_to_settled_rel"].append(num / den)
    if d["nearest_atom"] is None:
        d["nearest_atom"] = nearest_atom_report(store, Zs, basis)

    if arm == "pca_settle_lookup":
        # charter C.1/C.4: the matched-bytes lookup over the SAME atoms, driven by the
        # SETTLED state.  If this matches the arm, the settled state IS an atom.
        out = np.zeros((np.asarray(values).shape[0], int(store.dim)), np.float32)
        for C, w, G, Uk in zip(Zs, store.well_channels, store.modes, basis):
            A = np.asarray(G, np.float32)
            pts = np.asarray(C, np.float32) @ np.asarray(Uk, np.float32).T
            d2 = (np.sum(pts * pts, 1)[:, None] - 2.0 * (pts @ A.T)
                  + np.sum(A * A, 1)[None, :])
            out[:, np.asarray(w, np.int64)] = A[np.argmin(d2, axis=1)]
        return out, oor
    return render_span(store, Zs, basis), oor


def arm_features(obj: dict, values: np.ndarray, grid) -> tuple:
    """Dispatch: the settle verb here, everything else down the w5/w3 chain."""
    if obj.get("read") == "settle":
        return _settled_features(obj, values, grid)
    return _ORIG_ARM_FEATURES(obj, values, grid)


# ---------------------------------------------------------------------------
# 6. The runtime extension.  Additive, and asserted so.
# ---------------------------------------------------------------------------

_ORIG_BUILD_ARM = None
_ORIG_ARM_FEATURES = None
_ORIG_ARMS = None


def install() -> dict:
    """Append this hub's legs on top of w5's, without editing either module."""
    global _ORIG_BUILD_ARM, _ORIG_ARM_FEATURES, _ORIG_ARMS
    w5 = W5.install()                       # w5's legs first: this hub sits ON them
    _ORIG_BUILD_ARM = R.build_arm
    _ORIG_ARM_FEATURES = R.arm_features
    _ORIG_ARMS = tuple(R.ARMS)
    before = {"arms": tuple(R.ARMS), "store": dict(R.ARM_STORE),
              "read": dict(R.ARM_READ), "cls": dict(R.ARM_CLASS)}
    R.ARMS = tuple(_ORIG_ARMS) + PHYS_ARMS
    R.ARM_STORE.update({a: "pca" for a in PHYS_ARMS})
    R.ARM_READ.update({a: "settle" for a in PHYS_ARMS})
    R.ARM_CLASS.update(PHYS_ARM_CLASS)
    R.build_arm = build_arm
    R.arm_features = arm_features
    for a in before["arms"]:
        if (R.ARM_STORE.get(a) != before["store"].get(a)
                or R.ARM_READ.get(a) != before["read"].get(a)
                or R.ARM_CLASS.get(a) != before["cls"].get(a)):
            raise AssertionError(f"installing this hub moved pre-existing arm {a!r}")
    if tuple(R.ARMS)[:len(_ORIG_ARMS)] != _ORIG_ARMS:
        raise AssertionError("installing this hub reordered exp_gen_read.ARMS")
    return {"w5": w5, "added": list(PHYS_ARMS), "n_arms": len(R.ARMS),
            "pre_existing_unchanged": True}


def _make_artifacts_portable() -> dict:
    """Same defect, same fix as w5's: exactly ONE copy of every class in the process.

    The launcher runs ``python -m``, so this file's classes would live in ``__main__``
    and every pickled artifact would name ``__main__.SpanCoupling`` -- unreadable to a
    replicator loading them from an ordinary import.  The ``__main__`` entry point
    delegates to the package module and this overwrites the ``__main__`` copies.
    """
    import sys
    names = ("SpanCoupling",)
    g = globals()
    m = sys.modules.get("__main__")
    if __name__ != "__main__" and m is not None and m is not sys.modules.get(__name__):
        for n in names:
            setattr(m, n, g[n])
        return {"mode": "package classes installed into __main__", "classes": names}
    return {"mode": "no action needed"}


_PORTABILITY = _make_artifacts_portable()


def _dump_diag(out: Path, arm: str, log: Callable) -> None:
    if arm not in DIAG:
        return
    d = dict(DIAG[arm])
    rel = d.pop("launch_to_settled_rel", None)
    if rel:
        d["launch_to_settled_rel_median"] = float(np.median(rel))
        d["launch_to_settled_rel_max"] = float(np.max(rel))
    l2 = d.pop("launch_to_settled_l2", None)
    if l2:
        d["launch_to_settled_l2_median"] = float(np.median(l2))
    (out / "arms" / arm).mkdir(parents=True, exist_ok=True)
    fw._jdump(d, out / "arms" / arm / "settle_diagnostics.json")
    na = d.get("nearest_atom") or {}
    log(f"  arm {arm}: launch->settled relative displacement median="
        f"{d.get('launch_to_settled_rel_median', float('nan')):.6f}; "
        f"C.4 {na.get('verdict', 'n/a')} "
        f"(min ratio {na.get('min_ratio_over_wells', float('nan')):.4f})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=("pins", "prepare", "fit", "run", "table"))
    p.add_argument("--arm", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", default="val", choices=("val", "test"))
    p.add_argument("--max-seconds", type=float, default=2700.0)
    p.add_argument("--primary", default="pca_settle")
    p.add_argument("--margin", type=float, default=1.0)
    a = p.parse_args(argv)

    inst = install()
    R.set_condition(a.seed, a.condition)
    _bi = R.bind_instrument(log=lambda m: print(m, flush=True))
    out = R.out_leg(Path(a.out), a.seed, a.condition)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"stage={a.stage} arm={a.arm} out={out} seed={a.seed} cond={a.condition}")

    if a.stage == "pins":
        fw._jdump({"w6_pins": W6_PINS, "w5_pins": W5.W5_PINS, "read_pins": R.READ_PINS,
                   "fw_pins": fw.PINS, "install": inst, "instrument": _bi,
                   "bridge_root": os.environ.get("CHLU_GEN_BRIDGE_ROOT"),
                   "feature_cache": R._cache_manifest(int(a.seed))}, out / "pins.json")
        print(json.dumps({"w6_pins": W6_PINS, "install": inst}, indent=1, default=str))
        return 0
    if a.stage == "prepare":
        log(json.dumps(R.stage_prepare(out, log=log)))
        return 0
    if a.stage == "fit":
        log(json.dumps(R.stage_fit(a.arm, out, log=log), default=str))
        return 0
    if a.stage == "run":
        res = R.stage_run(a.arm, out, a.max_seconds, log=log)
        _dump_diag(out, a.arm, log)
        log(json.dumps(res, default=str))
        return 0
    if a.stage == "table":
        with R.read_dial():
            t = R.build_table(out, primary=a.primary, margin=a.margin, log=log)
        for name, blk in t["arms"].items():
            r4 = (blk.get("r4_comp_nway") or {})
            r3 = (blk.get("r3_comp_2way") or {})
            log(f"    {name:24s} {blk['class']:46s} "
                f"r4={r4.get('novel_tuples_per_corner', float('nan')):7.3f} "
                f"r3={r3.get('novel_tuples_per_corner', float('nan')):7.3f}")
        return 0
    raise SystemExit(f"stage {a.stage} not implemented")


if __name__ == "__main__":
    from chlu.experiments.exp_w6_compose_physics import main as _pkg_main

    raise SystemExit(_pkg_main())
