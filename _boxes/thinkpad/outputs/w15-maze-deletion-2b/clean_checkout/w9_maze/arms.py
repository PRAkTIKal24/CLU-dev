"""The arms: the CLU ported onto CMAZE-2, its charter-C controls, and the floor.

Every arm in this module answers the SAME question -- given four words
``(maze, agent, target, budget)``, produce a point in the shared encoder's latent
space -- and every arm is rendered by the SAME frozen decoder and run by the SAME
parameterless executor.  So a margin between two arms is a margin on the read.

The CLU
-------
It is the wave-5 incumbent's content carried as the wave-8 single design's statistic,
imported, not re-implemented:

* ``exp_w8_single_design.CovContentStore`` -- per well ``k``, the sufficient statistic
  ``S_k = sum x x^T, t_k = sum x, n_k``; the deployed atoms are ``[mu ; leading
  eigenvectors]``; deleting an item is a rank-m subtract with no refit and no pass over
  the data, and the items are never kept;
* ``gen_request_head`` -- the read ``x_hat_k = Phi(request) W_k G_k`` with the
  factorial design ``[bias | main | pair]`` and the closed-form ridge.  There is no
  feature argument on ``head_read``: charter B.3 holds by signature, and
  ``assert_head_request_only`` is run on every arm of every row.

The partition is ``ward_balanced``: ward linkage on the absolute correlation of the
live latent channels, cut into equal wells along the dendrogram's leaf order.  It is
the recipe at ``exp_gen_placement.py:317-334``, and it sees no label at all.

THE IDENTITY THAT GOVERNS THIS PORT, measured on val and stated once:

    span fraction = K * n_atoms / D = n_atoms / d_k = THE SATURATION RATIO

A store cannot span more of the latent space without moving toward saturation, where
``rank == d_k``, the projection is the identity, the content carries no information
and the random-atom control stops collapsing.  On Shapes3D the incumbent lives at
~1-3 % span; maze-route latents need >= ~72 % to render even a training item.

The controls (charter C)
------------------------
* ``landscape_removed``  -- the atoms are zeroed.  Must collapse.
* ``landscape_shuffled`` -- the atom rows are permuted across wells, so the same bytes
  are deployed in the wrong basins.  Must collapse.
* ``random_atoms``       -- atoms drawn from a Gaussian, same shape, same head fit.
* ``MATCHED-BYTES LOOKUP`` -- the mean of the nearest stored items by request
  agreement, holding as many items as the CLU's read holds floats.  Charter C.1: an arm
  that does not beat this is reported as decorative.
"""

from __future__ import annotations

import numpy as np

from chlu.core.gen_request_head import (HeadSpec, accumulate_gram,
                                        assert_head_request_only, head_read,
                                        request_design, solve_head)
from chlu.experiments.exp_w8_single_design import CovContentStore

__all__ = ["ward_balanced_partition", "CluRead", "LookupRead", "make_grids",
           "design_of", "store_floats", "head_floats", "nearest_item_audit",
           "HeadSpec"]


def ward_balanced_partition(Z: np.ndarray, K: int, seed: int = 0):
    """Label-free equal-width wells from the correlation dendrogram's leaf order.

    Transcribed from ``exp_gen_placement.py`` (the ``ward_balanced`` arm) so the
    partition this hub deploys is the one the programme already ships.  A channel with
    zero variance is DEAD; dead channels are dealt to the smallest well, exactly as
    there.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage

    X = np.asarray(Z, np.float64)
    D = int(X.shape[1])
    sd = X.std(axis=0)
    dead = sd <= 0
    live = np.flatnonzero(~dead)
    corr = np.corrcoef(X[:, live], rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    Zl = linkage(np.abs(corr), method="ward")
    order = live[leaves_list(Zl)]
    lab = np.zeros(D, np.int64)
    base = order.size // K
    cap = np.full(K, base, np.int64)
    cap[: order.size - base * K] += 1
    pos = 0
    for j, c in enumerate(cap):
        lab[order[pos:pos + int(c)]] = j
        pos += int(c)
    cnt = np.bincount(lab[live], minlength=K).astype(np.int64)
    for c in np.flatnonzero(dead):
        j = int(np.argmin(cnt))
        lab[c] = j
        cnt[j] += 1
    return [np.flatnonzero(lab == j).astype(np.int64) for j in range(K)], int(dead.sum())


def make_grids(n_mazes: int, n_agents: int, n_targets: int, budgets):
    """The address book's words, as the grids ``request_design`` interpolates on."""
    return [np.arange(n_mazes, dtype=np.float64),
            np.arange(n_agents, dtype=np.float64),
            np.arange(n_targets, dtype=np.float64),
            np.asarray(sorted(budgets), dtype=np.float64)]


def design_of(requests, grids, spec):
    """(N, 4) request values -> the design matrix.  No feature can enter here."""
    V = np.array([r.values() for r in requests], dtype=np.float64)
    return request_design(V, grids, spec)


def store_floats(store) -> int:
    return int(sum(np.asarray(M).size for M in store.modes))


def head_floats(heads) -> int:
    return int(sum(np.asarray(W).size for W in heads))


def nearest_item_audit(Z_hat, Z_train):
    """Charter C.4: is the read a lookup table in disguise?

    For each read, the distance to the NEAREST stored training latent, divided by the
    typical distance between two stored latents.  A ratio near 0 means the read has
    settled onto a written item and the arm is reported as a lookup table.
    """
    Zh = np.asarray(Z_hat, np.float64)
    Zt = np.asarray(Z_train, np.float64)
    d2 = np.empty((Zh.shape[0], Zt.shape[0]))
    for s in range(0, Zh.shape[0], 64):
        d2[s:s + 64] = ((Zh[s:s + 64, None, :] - Zt[None, :, :]) ** 2).sum(-1)
    nearest = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    idx = np.random.default_rng(0).integers(0, Zt.shape[0], size=(2, 512))
    typ = float(np.sqrt(((Zt[idx[0]] - Zt[idx[1]]) ** 2).sum(-1)).mean())
    ratio = float(nearest.mean() / max(typ, 1e-9))
    return {"nearest_atom_distance_mean": float(nearest.mean()),
            "typical_inter_item_distance": typ, "ratio": ratio,
            "verdict": ("LOOKUP TABLE (charter C.4): the read settles on a written item"
                        if ratio < 0.05 else "not a lookup table by C.4")}


class CluRead:
    """The CLU: a landscape of atoms, addressed by a request.  Nothing else reads it."""

    def __init__(self, Z_train, requests_train, grids, K=8, n_atoms=8, ridge=1.0,
                 pair_block=True, control=None, seed=0):
        self.grids = grids
        self.K, self.n_atoms, self.control = int(K), int(n_atoms), control
        self.spec = HeadSpec([int(g.size) for g in grids], pair_block=pair_block,
                             ridge=float(ridge))
        Z = np.asarray(Z_train, np.float32)
        self.dim = int(Z.shape[1])
        self.well_channels, self.n_dead = ward_balanced_partition(Z, self.K, seed)
        self.store = CovContentStore(Z, self.well_channels, self.n_atoms, self.dim)
        rng = np.random.default_rng(seed + 991)
        if control == "landscape_removed":
            self.store.modes = [np.zeros_like(M) for M in self.store.modes]
        elif control == "landscape_shuffled":
            #: the same bytes, deployed in the wrong basins: every well's atom block is
            #: dealt to a different well (truncated or padded to that well's width).
            pool = [np.asarray(M) for M in self.store.modes]
            perm = rng.permutation(self.K)
            new = []
            for k, ch in enumerate(self.well_channels):
                src = pool[perm[k]]
                d = int(ch.size)
                M = np.zeros((self.n_atoms, d), np.float32)
                w = min(d, src.shape[1])
                M[:, :w] = src[:, :w]
                new.append(M)
            self.store.modes = new
        elif control == "random_atoms":
            self.store.modes = [rng.normal(size=M.shape).astype(np.float32) /
                                np.sqrt(M.shape[1]) for M in self.store.modes]
        self.fit_head(Z, requests_train)

    def fit_head(self, Z, requests):
        Phi, _ = design_of(requests, self.grids, self.spec)
        A, Mx, _n = accumulate_gram([Phi], [np.asarray(Z, np.float64)],
                                    self.spec.width, self.dim)
        self.A, self.Mx = A, Mx
        self.heads = solve_head(A, Mx, self.store, self.spec.ridge)

    def refit_head_to_store(self):
        """Re-solve the head against the CURRENT atoms, reusing the retained Gram."""
        self.heads = solve_head(self.A, self.Mx, self.store, self.spec.ridge)

    def delete_rows_exact(self, X_rows, requests_deleted):
        """THE DELETION, applied to EVERY retained statistic, not only the store's.

        The wave-8 object carries the address book across the basis change with
        ``W_k <- W_k R_k``.  That carry is stated for a read expressed in the
        ORTHONORMAL basis ``U_k``; this hub's read is expressed in the deployed MODES
        ``G_k``, so it is not the right change of coordinates here -- measured on the
        scored leg it cost 0.1569 validity on the surviving targets.

        The repair is not a refit.  The head's own retained state is a pair of
        sufficient statistics, ``A = Phi^T Phi`` and ``Mx = Phi^T X``, additive in
        exactly the way the store's ``S, t, n`` are.  So the deletion subtracts the
        deleted rows' contribution from them too, and the same closed form is
        re-solved::

            S_k -= X_d^T X_d ;  t_k -= sum X_d ;  n_k -= m
            A   -= Phi_d^T Phi_d ;  Mx -= Phi_d^T X_d

        No pass over the retained pool, no items kept, nothing read that the caller did
        not hand in -- the same standard the store's downdate meets.
        """
        Xd = np.asarray(X_rows, np.float32)
        Phi_d, _ = design_of(requests_deleted, self.grids, self.spec)
        rec = self.store.delete_rows(Xd)
        P = np.asarray(Phi_d, np.float64)
        self.A = self.A - P.T @ P
        self.Mx = self.Mx - P.T @ np.asarray(Xd, np.float64)
        self.heads = solve_head(self.A, self.Mx, self.store, self.spec.ridge)
        rec["head_statistic_downdated"] = True
        rec["head_mechanism"] = ("A -= Phi_d^T Phi_d, Mx -= Phi_d^T X_d, then the same "
                                 "closed form is re-solved; the pool is never touched")
        return rec

    def carry_basis_change(self):
        """``W_k <- W_k R_k`` after a deletion -- exact, data-free, no pool row seen.

        This is the wave-8 mechanism, kept so the as-pre-registered number can be
        reproduced; :meth:`delete_rows_exact` is the corrected path for this read.
        """
        rot = getattr(self.store, "last_basis_change", None)
        if rot is None:
            raise AssertionError("no deletion has been performed")
        out = []
        for W, R in zip(self.heads, rot):
            if R.size == 0:
                out.append(np.zeros_like(W))
                continue
            r = R.shape[0]
            Wn = np.zeros_like(W)
            Wn[:, :R.shape[1]] = (np.asarray(W[:, :r], np.float32)
                                  @ np.asarray(R, np.float32))
            out.append(Wn)
        self.heads = out

    def read(self, requests) -> np.ndarray:
        Phi, oor = design_of(requests, self.grids, self.spec)
        self.last_out_of_range = oor
        return head_read(self.heads, self.store, Phi)

    def assert_request_only(self, requests):
        Phi, _ = design_of(requests[:8], self.grids, self.spec)
        return assert_head_request_only(self.heads, self.store, Phi)

    # -- ledgers -----------------------------------------------------------
    def ledger(self) -> dict:
        sf = store_floats(self.store)
        hf = head_floats(self.heads)
        stat = int(sum(w.stat_floats() for w in self.store.stats))
        return {
            "read_floats": sf + hf,
            "store_atom_floats": sf,
            "head_floats": hf,
            "deployed_bytes": 4 * (sf + hf),
            "statistic_floats_float64": stat,
            "statistic_bytes": 8 * stat,
            "note": ("deployed bytes are the atoms plus the address book, which is what "
                     "produces a read; the float64 sufficient statistic is what buys "
                     "EXACT deletion and is declared beside them, never netted in"),
            "K": self.K, "n_atoms": self.n_atoms, "dim": self.dim,
            "design_width": int(self.spec.width), "dead_channels": int(self.n_dead),
            "deployed_rank_per_well": [int(r) for r in self.store.deployed_rank],
            "rank_over_width": [float(r) / float(c.size) for r, c in
                                zip(self.store.deployed_rank, self.well_channels)],
        }


class LookupRead:
    """The MATCHED-BYTES LOOKUP floor (charter C.1).

    It holds whole stored latents -- as many as the CLU's read holds floats -- and
    answers a request with the mean of the stored items whose request words agree most.
    It is the object the CLU must beat: if a landscape of atoms cannot do better than
    keeping the items themselves at the same byte count, the landscape is decorative.
    """

    def __init__(self, Z_train, requests_train, budget_floats=None, seed=0):
        Z = np.asarray(Z_train, np.float32)
        n, d = Z.shape
        keep = n if budget_floats is None else max(1, int(budget_floats) // int(d))
        rng = np.random.default_rng(seed + 7)
        idx = np.arange(n) if keep >= n else rng.choice(n, size=keep, replace=False)
        self.idx = np.sort(idx)
        self.Z = Z[self.idx]
        self.keys = np.array([[r.m, r.a, r.t, r.b] for r in requests_train],
                             np.int64)[self.idx]
        self.n_stored = int(self.Z.shape[0])
        self.floats = int(self.Z.size)

    def read(self, requests) -> np.ndarray:
        Q = np.array([[r.m, r.a, r.t, r.b] for r in requests], np.int64)
        out = np.zeros((Q.shape[0], self.Z.shape[1]), np.float32)
        #: the maze word is weighted above the others: two routes in different mazes
        #: are not neighbours in any sense, which is the most generous reading of
        #: "nearest" for a lookup on this task.
        w = np.array([4.0, 1.0, 1.0, 0.25], np.float64)
        for i in range(Q.shape[0]):
            agree = ((self.keys == Q[i]) * w).sum(axis=1)
            best = agree.max()
            sel = np.flatnonzero(agree >= best - 1e-9)
            out[i] = self.Z[sel].mean(axis=0)
        return out
