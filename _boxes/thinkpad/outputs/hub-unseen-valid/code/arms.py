"""The row-2 arms and controls. Every one turns a REQUEST into a feature vector.

  knn_proto      FactorPrototypes.embed -- the pin's rival, R8's declared embedding,
                 which ALREADY linearly interpolates off-grid values. Second row.
  basis_head     THE BAR (w3-common s1): a fitted 64-atom dictionary per well (the
                 k-means++ atoms, landscape training NOT run) + a request-conditioned
                 linear head, signed, fitted by ridge. The head's request featurisation
                 is the TENT basis, so the head interpolates between prototypes --
                 which is what makes it the sharp control for THIS row.
  clu_span       THE ARM: the SAME head over the LANDSCAPE's trained modes. Byte- and
                 structure-identical to basis_head; the only difference is whether the
                 atoms were shaped by the landscape objective.
  clu_discrete   the shipped 30-bit read: interpolated address -> argmax code per well
                 -> gen_store.read_from_codes. This is the one the SNAP is measured on.
  store_removed_* the laundering control (charter C.2): store content deleted, head
                 REFITTED against the deleted store (the strongest form of the control).

Request-only: no feature tensor enters any read here. The heads are fitted on train
features offline; at read time an arm sees only phi(request) and its own atoms.
"""
from __future__ import annotations

import numpy as np

TENT_EPS = 0.0


def _tent_index(values, x):
    """(i, j, w): x = (1-w)*values[i] + w*values[j]. Interpolates inside, extrapolates
    outside -- the same rule gen_request._extrapolating_index declares for R8."""
    v = np.asarray(values, np.float64)
    x = float(x)
    if v.size == 1:
        return 0, 0, 0.0
    if x <= v[0]:
        i, j = 0, 1
    elif x >= v[-1]:
        i, j = v.size - 2, v.size - 1
    else:
        j = int(np.searchsorted(v, x, side="right"))
        j = min(max(j, 1), v.size - 1)
        i = j - 1
    w = (x - v[i]) / (v[j] - v[i])
    return i, j, float(w)


class Tent:
    """phi(request): per-factor tent weights, + a PAIR-INTERACTION block, + a bias.

    The tent (hat) weights make phi a continuous, piecewise-linear function of every
    requested value, so an OFF-GRID value is expressed as the interpolation of its two
    bracketing grid slots -- the same rule R8 declares for the prototype table.

    ⭐ The pair block is not optional decoration: w3-common s1 records that on the
    wave-2 basis+head object it is **the pair-interaction block, not the store, that
    beats knn_proto**. A control built without it is not the wave-2 object and would be
    a strawman. Pair features are products of tent weights, so they interpolate
    bilinearly and an off-grid request stays continuous through them.
    """

    def __init__(self, grid, pairs=True):
        self.values = [np.asarray(v, np.float64) for v in grid.values]
        self.cards = [int(v.size) for v in self.values]
        self.nf = len(self.cards)
        self.off = np.cumsum([0] + self.cards)[:-1]
        self.main = int(sum(self.cards))
        self.pairs = bool(pairs)
        self.pair_off = {}
        o = self.main
        if self.pairs:
            for k in range(self.nf):
                for l in range(k + 1, self.nf):
                    self.pair_off[(k, l)] = o
                    o += self.cards[k] * self.cards[l]
        self.dim = o + 1

    def _slots(self, terms):
        """per factor: list of (index, weight); a duplicated factor contributes twice."""
        sl = [[] for _ in range(self.nf)]
        for t in terms:
            k = int(t["factor_index"])
            i, j, w = _tent_index(self.values[k], float(t["value"]))
            sl[k].append((i, 1.0 - w))
            if j != i:
                sl[k].append((j, w))
        return sl

    def of_terms(self, terms):
        p = np.zeros(self.dim, np.float64)
        p[-1] = 1.0
        sl = self._slots(terms)
        for k in range(self.nf):
            for i, w in sl[k]:
                p[self.off[k] + i] += w
        if self.pairs:
            for (k, l), base in self.pair_off.items():
                for i, wi in sl[k]:
                    for j, wj in sl[l]:
                        p[base + i * self.cards[l] + j] += wi * wj
        return p

    def many(self, requests):
        out = np.zeros((len(requests), self.dim), np.float64)
        for n, r in enumerate(requests):
            out[n] = self.of_terms(r["request"] if isinstance(r, dict) else r)
        return out

    def of_labels(self, Y):
        """Train rows: exact grid values -> one-hot tents (w = 0) + their pair products."""
        Y = np.asarray(Y, np.int64)
        n = Y.shape[0]
        out = np.zeros((n, self.dim), np.float64)
        out[:, -1] = 1.0
        ar = np.arange(n)
        for k in range(self.nf):
            out[ar, self.off[k] + Y[:, k]] = 1.0
        if self.pairs:
            for (k, l), base in self.pair_off.items():
                out[ar, base + Y[:, k] * self.cards[l] + Y[:, l]] = 1.0
        return out


# ---------------------------------------------------------------------------
# the request-conditioned linear head, factored through a per-well basis
# ---------------------------------------------------------------------------
class SpanHead:
    """x_k = (phi @ W_k) @ A_k. Signed: nothing constrains the coefficients to a
    simplex (w3-common s4.5 forbids imposing one)."""

    def __init__(self, atoms, well_channels, dim, feat_shape):
        self.A = [np.asarray(a, np.float64) for a in atoms]
        self.wc = [np.asarray(w, np.int64) for w in well_channels]
        self.dim = int(dim)
        self.feat_shape = tuple(feat_shape)
        self.W = None

    def fit(self, Phi, X, ridge_phi=1e-2, ridge_atom=1e-6):
        """Two-sided ridge: the best linear map phi -> x_k, expressed in the basis."""
        Phi = np.asarray(Phi, np.float64)
        G = Phi.T @ Phi + ridge_phi * np.eye(Phi.shape[1])
        Gi = np.linalg.inv(G)
        self.W = []
        for A, w in zip(self.A, self.wc):
            Y = np.asarray(X[:, w], np.float64)
            R = Gi @ (Phi.T @ Y)                       # (P, d_k) unconstrained head
            AA = A @ A.T + ridge_atom * np.eye(A.shape[0])
            self.W.append(R @ A.T @ np.linalg.inv(AA))  # (P, M) coefficients
        return self

    def coeffs(self, Phi):
        return [np.asarray(Phi, np.float64) @ W for W in self.W]

    def read(self, Phi):
        Phi = np.asarray(Phi, np.float64)
        out = np.zeros((Phi.shape[0], self.dim), np.float32)
        for W, A, w in zip(self.W, self.A, self.wc):
            out[:, w] = ((Phi @ W) @ A).astype(np.float32)
        return out.reshape((Phi.shape[0],) + self.feat_shape)

    def coeff_report(self, Phi):
        B = self.coeffs(Phi)
        allc = np.concatenate([b.ravel() for b in B])
        return {
            "frac_negative": float((allc < 0).mean()),
            "mean_L1_per_well": float(np.mean([np.abs(b).sum(1).mean() for b in B])),
            "frac_wells_forming_a_probability_vector": float(
                np.mean([
                    ((b >= 0).all(1) & (np.abs(b.sum(1) - 1.0) < 1e-6)).mean() for b in B
                ])
            ),
        }


# ---------------------------------------------------------------------------
# the CLU addressing head, made CONTINUOUS in the requested value
# ---------------------------------------------------------------------------
def interp_log_scores(book, tent, requests):
    """(N, K, M) log-scores for requests that may name OFF-GRID values.

    The shipped gen_dream.address_requests indexes log_cond[v] by an integer class,
    so an off-grid value is INEXPRESSIBLE to it. This replaces that lookup with the
    tent-weighted combination of the two bracketing slices -- the same rule R8 already
    declares for the prototype table -- so the posterior is a continuous function of
    the requested value. Nothing else about the addressing head changes, and no
    feature tensor enters (charter B.3).
    """
    lp = np.asarray(book.log_prior, np.float64)          # (K, M)
    out = np.broadcast_to(lp, (len(requests),) + lp.shape).copy()
    for n, r in enumerate(requests):
        for t in (r["request"] if isinstance(r, dict) else r):
            k = int(t["factor_index"])
            lc = np.asarray(book.log_cond[k], np.float64)  # (V, K, M)
            i, j, w = _tent_index(tent.values[k], float(t["value"]))
            slab = (1.0 - w) * lc[i] + w * lc[j] if j != i else lc[i]
            out[n] += slab - lp
    return out


def posterior_from_scores(S):
    S = np.asarray(S, np.float64)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=-1, keepdims=True)
    return P


def codes_from_posterior(P):
    return np.argmax(np.asarray(P), axis=-1).astype(np.int64)
