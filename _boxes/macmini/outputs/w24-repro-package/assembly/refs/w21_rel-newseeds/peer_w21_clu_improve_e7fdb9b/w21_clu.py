"""w21 hub/maze-t-clu-improve -- the CLU candidates.  ONLY the CLU changes.

``TransitionCluX`` is ``A.TransitionClu`` (same store, same partition, same closed-form
end-to-end ridge ``solve_head``, same ``head_read`` with no feature argument, same
exact deletion by downdating A, Mx, S_k, t_k, n_k) with a WIDER / RE-WEIGHTED request
design.  The design is still a pure function of the request (the five words, plus -- only
where a candidate says so -- the observation's dynamics scalars, which every rival already
receives inside the observation).  Candidate knobs:

  pair_scale   s_p   multiplies every pair column.  Because the ridge penalises ||W||^2,
                     scaling a column block by s is EXACTLY a per-block ridge of ridge/s^2
                     on that block (W = s W'), i.e. hierarchical shrinkage of the pairs.
  triples      tuple of factor-index triples whose one-hot product block is appended
               ("all" = all 10), scaled by triple_scale.
  dyn          "none" | "main" (the 31 dyn scalars as columns) | "x<f>" (dyn x word-f
               one-hots too); scaled by dyn_scale.  Dyn scalars are standardised with the
               TRAIN mean/sd, which are part of the object (printed in the ledger count).
  ridge, K, n_atoms   as A1.

Ledger: read floats = atoms + address book (design width x n_atoms x K) -- the rivals are
re-matched to THIS number by the standing rule.
"""
from __future__ import annotations

import itertools
import numpy as np

import chlu.exp_w15_cmaze_t_arms as A
from chlu.core.gen_request_head import HeadSpec, request_design


def onehots(W, cards):
    W = np.asarray(W, np.int64)
    return [np.eye(c, dtype=np.float32)[np.clip(W[:, f], 0, c - 1)] for f, c in enumerate(cards)]


class DesignX:
    def __init__(self, cards, pair_block=True, pair_scale=1.0, main_scale=1.0,
                 triples=(), triple_scale=1.0, dyn="none", dyn_scale=1.0,
                 dyn_mu=None, dyn_sd=None, rel="none", rel_scale=1.0, rel_x=(), k_colours=4):
        self.cards = [int(c) for c in cards]
        self.pair_block, self.pair_scale, self.main_scale = bool(pair_block), float(pair_scale), float(main_scale)
        F = len(self.cards)
        if triples == "all":
            triples = tuple(itertools.combinations(range(F), 3))
        self.triples = tuple(tuple(int(x) for x in t) for t in triples)
        self.triple_scale, self.dyn, self.dyn_scale = float(triple_scale), str(dyn), float(dyn_scale)
        self.dyn_mu, self.dyn_sd = dyn_mu, dyn_sd
        self.spec = HeadSpec(self.cards, pair_block=pair_block, ridge=1.0)
        self.grids = [np.arange(c, dtype=np.float64) for c in self.cards]
        n_main = int(self.spec.layout["n_main"])
        self.width_base = int(self.spec.width)
        self.n_main = n_main
        self.width_tri = int(sum(np.prod([self.cards[i] for i in t]) for t in self.triples))
        self.width_dyn = 0
        if self.dyn != "none":
            nd = 31 if dyn_mu is None else int(np.size(dyn_mu))
            self.width_dyn = nd
            if self.dyn.startswith("x"):
                for f in [int(c) for c in self.dyn[1:]]:
                    self.width_dyn += nd * self.cards[f]
        #: colour-relational block ("colour coding"): the colour-bearing words share ONE
        #: colour alphabet 1..k; eq(x, y) = both name a colour and it is the same colour.
        self.rel, self.rel_scale, self.k = str(rel), float(rel_scale), int(k_colours)
        self.rel_x = tuple(int(f) for f in rel_x)
        self.rel_pairs = []
        if self.rel != "none":
            cw = (0, 1, 2, 3)                      # agent/pocket, key, door, tint
            self.rel_pairs = [p for p in itertools.combinations(cw, 2)]
            if self.rel == "eq3":
                self.rel_pairs = self.rel_pairs + [(0, 2, 3)]
        n_rel = len(self.rel_pairs)
        self.width_rel = n_rel + n_rel * int(sum(self.cards[f] for f in self.rel_x))
        self.width = self.width_base + self.width_tri + self.width_dyn + self.width_rel

    def to_dict(self):
        return {"cards": self.cards, "pair_block": self.pair_block, "pair_scale": self.pair_scale,
                "main_scale": self.main_scale, "triples": [list(t) for t in self.triples],
                "triple_scale": self.triple_scale, "dyn": self.dyn, "dyn_scale": self.dyn_scale,
                "width": self.width, "width_base": self.width_base, "width_triples": self.width_tri,
                "width_dyn": self.width_dyn, "rel": self.rel, "rel_pairs": [list(p) for p in self.rel_pairs],
                "rel_x": list(self.rel_x), "rel_scale": self.rel_scale, "width_rel": self.width_rel}

    def relations(self, W):
        W = np.asarray(W, np.int64)
        k = self.k
        col = lambda f: np.where((W[:, f] >= 1) & (W[:, f] <= k), W[:, f], -1 - f)
        R = []
        for p in self.rel_pairs:
            e = np.ones(len(W), bool)
            for a, b in zip(p[:-1], p[1:]):
                e &= col(a) == col(b)
            R.append(e.astype(np.float32))
        return np.stack(R, 1) if R else np.zeros((len(W), 0), np.float32)

    def __call__(self, W, D=None):
        W = np.asarray(W, np.float64)
        Phi, oor = request_design(W, self.grids, self.spec)
        Phi = np.asarray(Phi, np.float32)
        Phi[:, 1:1 + self.n_main] *= self.main_scale
        Phi[:, 1 + self.n_main:] *= self.pair_scale
        blocks = [Phi]
        if self.triples:
            oh = onehots(W, self.cards)
            N = W.shape[0]
            for (i, j, k) in self.triples:
                b = oh[i][:, :, None, None] * oh[j][:, None, :, None] * oh[k][:, None, None, :]
                blocks.append(self.triple_scale * b.reshape(N, -1))
        if self.dyn != "none":
            assert D is not None, "this candidate consumes the dyn scalars"
            Dz = (np.asarray(D, np.float32) - self.dyn_mu) / self.dyn_sd
            blocks.append(self.dyn_scale * Dz)
            if self.dyn.startswith("x"):
                oh = onehots(W, self.cards)
                for f in [int(c) for c in self.dyn[1:]]:
                    blocks.append(self.dyn_scale * (Dz[:, :, None] * oh[f][:, None, :]).reshape(len(Dz), -1))
        if self.rel != "none":
            R = self.relations(W)
            blocks.append(self.rel_scale * R)
            if self.rel_x:
                oh = onehots(W, self.cards)
                for f in self.rel_x:
                    blocks.append(self.rel_scale * (R[:, :, None] * oh[f][:, None, :]).reshape(len(R), -1))
        out = np.concatenate(blocks, 1).astype(np.float32)
        assert out.shape[1] == self.width, (out.shape, self.width)
        return out, oor


class TransitionCluX(A.TransitionClu):
    """A.TransitionClu with ``DesignX``.  Request = (words, dyn) pair; words-only when
    dyn == 'none'.  Controls are inherited verbatim (same constructor code path)."""

    def __init__(self, Z_train, words_train, grids, dyn_train=None, K=2, n_atoms=6, ridge=1.0,
                 design_kw=None, control=None, seed=0, well_channels=None, pair_block=True):
        kw = dict(design_kw or {})
        kw.setdefault("pair_block", pair_block)
        if kw.get("dyn", "none") != "none" and kw.get("dyn_mu") is None:
            D = np.asarray(dyn_train, np.float32)
            mu, sd = D.mean(0), D.std(0)
            sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
            kw["dyn_mu"], kw["dyn_sd"] = mu.astype(np.float32), sd
        self.dx = DesignX([int(g.size) for g in grids], **kw)
        self._dyn_train = dyn_train
        super().__init__(Z_train, (words_train, dyn_train), grids, K=K, n_atoms=n_atoms,
                         ridge=ridge, pair_block=kw["pair_block"], control=control, seed=seed,
                         well_channels=well_channels)
        self.spec.layout = dict(self.spec.layout, width=self.dx.width)

    def design(self, req):
        if isinstance(req, tuple):
            W, D = req
        else:
            W, D = req, None
        return self.dx(W, D)

    def fit_head(self, Z, req):
        from chlu.core.gen_request_head import accumulate_gram, solve_head
        Phi, _ = self.design(req)
        A_, Mx, _n = accumulate_gram([Phi], [np.asarray(Z, np.float64)], self.dx.width, self.dim)
        self.A, self.Mx = A_, Mx
        self.heads = solve_head(A_, Mx, self.store, self.spec.ridge)

    def ledger(self):
        led = super().ledger()
        extra = 0
        if self.dx.dyn != "none":
            extra = 2 * int(np.size(self.dx.dyn_mu))      # train mean + sd travel with the object
            led["read_floats"] += extra
            led["deployed_bytes_float32"] = 4 * led["read_floats"]
        led["design"] = self.dx.to_dict()
        led["dyn_standardiser_floats"] = extra
        return led


# ================================================== runner hooks (w19 runner, CLU swapped)
class CluFactory:
    """Drop-in for ``A.TransitionClu`` inside the w19 runner: same signature, the frozen
    candidate design baked in.  Words-only designs (dyn == 'none') -- asserted."""

    CONTROLS = A.TransitionClu.CONTROLS

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.design_kw = dict(cfg.get("design_kw", {}))
        assert self.design_kw.get("dyn", "none") == "none", "runner hook is words-only"

    def __call__(self, Z, W, grids, K=2, n_atoms=6, ridge=1.0, pair_block=True, control=None,
                 seed=0, well_channels=None):
        #: the frozen cfg's K / n_atoms / ridge win over the runner's A1 values (which the
        #: FROZEN blind arms keep).
        K = int(self.cfg.get("K", K))
        n_atoms = int(self.cfg.get("n_atoms", n_atoms))
        ridge = float(self.cfg.get("ridge", ridge))
        return TransitionCluX(Z, W, grids, K=K, n_atoms=n_atoms, ridge=ridge,
                              design_kw=self.design_kw, control=control, seed=seed,
                              well_channels=well_channels, pair_block=pair_block)

    def make_control(self, Ztr, Wtr, grids, K, n_atoms, ridge, control, seed, pair_block=True):
        """``W16.make_control`` verbatim, with the candidate CLU: across-wells is a
        DERANGEMENT (W16.derangement), asserted by the runner."""
        import w16_cmaze_t as W16
        if control != "landscape_shuffled_across_wells":
            return self(Ztr, Wtr, grids, K=K, n_atoms=n_atoms, ridge=ridge, control=control,
                        seed=seed, pair_block=pair_block)
        obj = self(Ztr, Wtr, grids, K=K, n_atoms=n_atoms, ridge=ridge, seed=seed,
                   pair_block=pair_block)
        rng = np.random.default_rng(seed + 991)
        pool = [np.asarray(M) for M in obj.store.modes]
        perm = W16.derangement(obj.K, rng)
        new = []
        for j, ch in enumerate(obj.well_channels):
            src = pool[perm[j]]
            d = int(ch.size)
            M = np.zeros((obj.n_atoms, d), np.float32)
            w = min(d, src.shape[1])
            M[:, :w] = src[:, :w]
            new.append(M)
        obj.store.modes = new
        obj.control = control + " (deranged)"
        obj.fit_head(np.asarray(Ztr, np.float32), Wtr)
        obj.derangement = [int(x) for x in perm]
        return obj

    def relations_block(self, W):
        """The candidate's colour-relational columns (rel + rel_x crosses) as a plain
        array -- used ONLY by the printed '+rel' rival DIAGNOSTIC, never a bar."""
        dx = DesignX(T_cards(), **self.design_kw)
        full, _ = dx(W)
        return full[:, dx.width - dx.width_rel:]


def T_cards():
    import chlu.exp_w15_cmaze_t as T
    return T.alphabets(4)
