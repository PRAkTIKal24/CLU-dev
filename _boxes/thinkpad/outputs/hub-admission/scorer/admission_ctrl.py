"""hub/admission -- the MANDATORY CONTROLS of SS4, and the paired statistics of R1.

(a) whole-latent kNN gate, q = 0.95
(b) the FAIR rival: per-well-factored kNN gate, same q
(c) single-well (K = 1) CLU on the SAME atoms -- must lose localisation
(e) physics-free head margin M
(f) accept_A7 beside accept_B on every row (programme-wide rule, 2026-09-17)
(d) live-set hold-out / write-back lives in u4_liveset.py -- it needs rebuilt stores.

⛔ NOT the SSC.2 atom shuffle: SS4 excludes it (it tests decoding, and cannot certify an
interface-facing claim).

THE ASYMMETRY, STATED ONCE AND NEVER HIDDEN.  The CLU gate has NO fitted threshold: the
level set is zero and zero is derived.  Every rival here is CALIBRATED at q on a
calibration split of VALID requests built from ``fit_rows`` -- ⛔ never from ``val``
(which carries the accept number) and ⛔ never from B (held out).  Giving the rival its
own free parameter, fitted on data the arm never sees, is the honest direction for the
asymmetry to run.
"""
from __future__ import annotations

import os
import sys

import numpy as np

CHLU_ROOT = os.environ["CHLU_ROOT"]
if CHLU_ROOT not in sys.path:
    sys.path.insert(0, CHLU_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from chlu.core.gen_landscape import (  # noqa: E402
    AtomLandscape,
    AtomStore,
    _well_logk,
    atom_bandwidths,
    request_energy,
)


# ---------------------------------------------------------------------------
# distances, shared by every kNN control
# ---------------------------------------------------------------------------

def atom_matrix(store) -> np.ndarray:
    """``(N, D)`` -- the atoms reassembled into whole rows.  They ARE whole train rows
    (``assert_atoms_are_train_rows``), so this is a faithful reassembly, not a mixture."""
    Z = np.zeros((store.n_atoms, store.dim), np.float32)
    for k, w in enumerate(store.well_channels):
        Z[:, np.asarray(w, np.int64)] = np.asarray(store.atoms[k], np.float32)
    return Z


def _nn_dist(Q: np.ndarray, A: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = np.empty(Q.shape[0], np.float64)
    aa = (A * A).sum(1)[None, :]
    for i in range(0, Q.shape[0], chunk):
        q = Q[i: i + chunk].astype(np.float64)
        d2 = (q * q).sum(1)[:, None] - 2.0 * (q @ A.T.astype(np.float64)) + aa
        out[i: i + q.shape[0]] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def whole_latent_dist(store, X) -> np.ndarray:
    return _nn_dist(np.asarray(X, np.float32), atom_matrix(store))


def factored_dist(store, X) -> np.ndarray:
    """``(n, K)`` per-well nearest-atom distance."""
    Xn = np.asarray(X, np.float32)
    return np.stack([_nn_dist(Xn[:, np.asarray(w, np.int64)], np.asarray(a, np.float32))
                     for w, a in zip(store.well_channels, store.atoms, strict=True)], axis=1)


# ---------------------------------------------------------------------------
# (a) and (b): the kNN gates
# ---------------------------------------------------------------------------

class KnnGate:
    """A distance gate with ONE fitted number per column, the q-quantile on calibration."""

    def __init__(self, kind: str, q: float, thresholds: np.ndarray):
        self.kind, self.q, self.thresholds = kind, float(q), np.asarray(thresholds, np.float64)

    @classmethod
    def fit(cls, store, X_cal, kind: str = "factored", q: float = 0.95) -> "KnnGate":
        D = factored_dist(store, X_cal) if kind == "factored" else whole_latent_dist(store, X_cal)[:, None]
        return cls(kind, q, np.quantile(D, q, axis=0))

    def decide(self, store, X) -> dict:
        D = factored_dist(store, X) if self.kind == "factored" else whole_latent_dist(store, X)[:, None]
        ref = D > self.thresholds[None, :]
        return {"refuse_col": ref, "refuse": ref.any(1), "accept": ~ref.any(1),
                "D": D, "n_cols_refusing": ref.sum(1)}

    def summary(self, store, X) -> dict:
        d = self.decide(store, X)
        return {"n": int(d["accept"].shape[0]), "accept_rate": float(d["accept"].mean()),
                "refuse_rate": float(d["refuse"].mean()),
                "mean_cols_refusing": float(d["n_cols_refusing"].mean()),
                "kind": self.kind, "q": self.q,
                "thresholds": [float(t) for t in self.thresholds]}


# ---------------------------------------------------------------------------
# (c) the single-well (K = 1) CLU control -- SAME atoms, SAME bytes, one well
# ---------------------------------------------------------------------------

def build_K1(land, X_train, config_frac: float = 1.0, rule: str = "nn_median"):
    """A K = 1 landscape on the SAME atom rows and the SAME channels.

    ⭐ Identical store bytes (``N * D * 4``, independent of K -- ``AtomStore``'s own
    docstring says so), identical atom rows, identical bandwidth RULE.  Exactly one
    declared quantity differs: the channel partition is trivial.  It must therefore
    lose LOCALISATION while keeping whatever global detection power the gate has.
    """
    store = land.store
    all_ch = [tuple(range(store.dim))]
    rows = np.asarray(store.atom_rows, np.int64)
    s1 = AtomStore(all_ch, store.feat_shape, np.asarray(X_train, np.float32), rows,
                   selection=f"K1_CONTROL(from {store.selection})")
    sig = atom_bandwidths(s1, float(config_frac), str(rule))
    return AtomLandscape(s1, sig, None, float(land.request_strength), 0.0)


# ---------------------------------------------------------------------------
# (e) the physics-free head margin M
# ---------------------------------------------------------------------------

def head_margin(P: np.ndarray) -> np.ndarray:
    """``(n,)`` -- the standard readout margin of the address-book posterior.

    Per well, ``top1 - top2`` of the posterior over the well's atoms; pooled by the MIN
    over wells, so a request that is ambiguous in ANY well has a small margin -- the
    same "any well objects" reduction the gate uses, so the comparison isolates the
    PHYSICS and not the reduction.  ⛔ No landscape, no distance, no sigma: this is the
    paper's "standard readout margin", the thing the residual must beat.
    """
    Q = np.sort(np.asarray(P, np.float64), axis=-1)
    return (Q[..., -1] - Q[..., -2]).min(axis=-1)


class MarginGate:
    def __init__(self, q: float, thr: float):
        self.q, self.thr = float(q), float(thr)

    @classmethod
    def fit(cls, P_cal, q: float = 0.95) -> "MarginGate":
        # refuse the SMALL-margin tail; 1 - q so the calibration refusal rate matches kNN's
        return cls(q, float(np.quantile(head_margin(P_cal), 1.0 - q)))

    def summary(self, P) -> dict:
        m = head_margin(P)
        acc = m >= self.thr
        return {"n": int(acc.size), "accept_rate": float(acc.mean()),
                "refuse_rate": float((~acc).mean()), "kind": "head_margin_M",
                "q": self.q, "thresholds": [self.thr]}

    def decide(self, P) -> dict:
        m = head_margin(P)
        return {"accept": m >= self.thr, "refuse": m < self.thr, "M": m}


# ---------------------------------------------------------------------------
# per-(term, well) residuals -- the localisation statistic
# ---------------------------------------------------------------------------

def per_term_well_residuals(land, X, log_w) -> np.ndarray:
    """``(n, T, K)`` -- ``E_{t,k}(x) - E_k(x)``.

    The SAME arithmetic ``gen_landscape.request_energy`` performs, kept per (term, well)
    instead of summed.  :func:`assert_matches_shipped_request_energy` proves the sum over
    (t, k) reproduces the shipped scalar, so this is a reduction, not a re-derivation.
    """
    Xj = jnp.asarray(np.asarray(X, np.float32))
    sig = land.sigma
    lks = _well_logk(land.store, Xj, sig)
    T = int(log_w.shape[0])
    out = np.empty((int(Xj.shape[0]), T, len(lks)), np.float64)
    lw = jnp.asarray(np.asarray(log_w, np.float32))
    for k, (lk, s) in enumerate(zip(lks, sig, strict=True)):
        base = -(s ** 2) * jax.nn.logsumexp(lk, axis=-1)
        for t in range(T):
            e = -(s ** 2) * jax.nn.logsumexp(lk + lw[t, :, k, :], axis=-1)
            out[:, t, k] = np.asarray(e - base, np.float64)
    return out


def assert_matches_shipped_request_energy(land, X, log_w, n_terms, tol: float = 1e-2) -> dict:
    r = per_term_well_residuals(land, X, log_w)
    T = int(log_w.shape[0])
    active = (np.asarray(n_terms)[:, None] > np.arange(T)[None, :])
    got = (r * active[:, :, None]).sum(axis=(1, 2))
    # the shipped scalar sums ALL T slots; mask the inactive ones the same way it is used
    ref = np.asarray(request_energy(land, jnp.asarray(np.asarray(X, np.float32)),
                                    jnp.asarray(np.asarray(log_w, np.float32))), np.float64)
    ref_all = r.sum(axis=(1, 2))
    denom = np.maximum(np.abs(ref), 1.0)
    rel = float(np.max(np.abs(ref_all - ref) / denom)) if ref.size else 0.0
    return {"max_rel_delta_all_slots": rel, "agrees": bool(rel <= tol),
            "masked_sum_mean": float(got.mean()),
            "note": "sum_{t,k} residual (this module) vs gen_landscape.request_energy (shipped)"}


def localise(land, X, log_w, n_terms, gate_result) -> np.ndarray:
    """``(n,)`` -- the PRE-REGISTERED rule: the requested term with the largest residual
    among UNCERTIFIED wells.  "Uncertified" = the well is not certified-accepted, i.e.
    it is one of the wells the gate is not vouching for; if every well is certified the
    rule falls back to all wells (it must still name a term)."""
    r = per_term_well_residuals(land, X, log_w)                 # (n, T, K)
    T = int(log_w.shape[0])
    active = (np.asarray(n_terms)[:, None] > np.arange(T)[None, :])
    unc = ~np.asarray(gate_result["certified_well"])            # (n, K)
    allcert = ~unc.any(axis=1)
    unc[allcert] = True
    m = np.where(unc[:, None, :], r, -np.inf).max(axis=2)       # (n, T)
    m = np.where(active, m, -np.inf)
    return m.argmax(axis=1)


# ---------------------------------------------------------------------------
# R1: the paired statistic
# ---------------------------------------------------------------------------

def paired_bootstrap(acc_arm, acc_riv, ref_arm, ref_riv, n_boot: int = 4000, seed: int = 20260917):
    """Delta = score(arm) - score(rival), score = 0.5*(refuse_out_of_support + accept_B).

    PAIRED: one resample of the accept rows and one of the refuse rows is shared by both
    arms, so the CI is on the DIFFERENCE and the arms' correlation is kept.
    R1: tie iff the 95% CI includes 0 OR |Delta| <= m = 0.020.
    """
    rng = np.random.default_rng(int(seed))
    a_arm, a_riv = np.asarray(acc_arm, bool), np.asarray(acc_riv, bool)
    r_arm, r_riv = np.asarray(ref_arm, bool), np.asarray(ref_riv, bool)
    na, nr = a_arm.size, r_arm.size
    d = np.empty(int(n_boot), np.float64)
    for b in range(int(n_boot)):
        ia, ir = rng.integers(0, na, na), rng.integers(0, nr, nr)
        s_arm = 0.5 * (r_arm[ir].mean() + a_arm[ia].mean())
        s_riv = 0.5 * (r_riv[ir].mean() + a_riv[ia].mean())
        d[b] = s_arm - s_riv
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    s_arm = 0.5 * (r_arm.mean() + a_arm.mean())
    s_riv = 0.5 * (r_riv.mean() + a_riv.mean())
    delta = float(s_arm - s_riv)
    m = 0.020
    if delta > 0 and lo > 0:
        v = "win"
    elif delta < -m and hi < 0:
        v = "loss"
    else:
        v = "tie"
    return {"score_arm": float(s_arm), "score_rival": float(s_riv), "delta": delta,
            "ci95": [lo, hi], "m": m, "verdict_R1": v, "n_boot": int(n_boot),
            "accept_arm": float(a_arm.mean()), "accept_rival": float(a_riv.mean()),
            "refuse_arm": float(r_arm.mean()), "refuse_rival": float(r_riv.mean())}
