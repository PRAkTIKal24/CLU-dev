"""The CMAZE-O arms: the CLU single design, its charter-C controls, the matched-bytes
lookup floor, the matched-parameter learned rivals, and the BLIND columns.

Every latent-emitting arm in this module answers the SAME question -- given nine words,
produce a point in the shared encoder's latent space -- and every one of them is
rendered by the SAME frozen decoder and read back by the SAME parameterless state
decoder.  So a margin between two of them is a margin **on the read**.

The CLU (Config-A single design)
--------------------------------
The wave-5 incumbent's content carried as the wave-8 single design's statistic,
imported, never re-implemented:

* ``chlu.experiments.exp_w8_single_design.CovContentStore`` -- per well ``k`` the
  sufficient statistic ``S_k = sum x x^T, t_k = sum x, n_k``; the deployed atoms are
  ``[mu ; leading eigenvectors]``; deleting rows is a rank-m subtract with no refit, no
  pass over the data, and the items are never kept;
* ``chlu.core.gen_request_head`` -- the read ``x_hat_k = Phi(request) W_k G_k`` with the
  factorial design ``[bias | main | pair]`` and the closed-form ridge.  There is no
  feature argument on ``head_read``: charter B.3 holds **by signature**, and
  ``assert_head_request_only`` is run on every arm of every row.

The partition is ``ward_balanced``, imported from ``w9_maze.arms`` so this hub deploys
the partition the programme already ships.  It sees no label at all.

The controls (charter C)
-------------------------
* ``landscape_removed``                -- the atoms are zeroed.  Must collapse.
* ``landscape_shuffled``               -- each well's atom ROWS are permuted inside that
                                          well.  **MEASURED TO BE A PROVED NULL on this
                                          read** -- see below.  Kept, because a control
                                          that cannot move is worth stating once.
* ``landscape_shuffled_channels``      -- each well's atom block has its CHANNELS
                                          permuted inside that well.  Same bytes, same
                                          basin, wrong channels.  This is the
                                          within-well control that can actually move.
* ``landscape_shuffled_across_wells``  -- each well's atom BLOCK is dealt to a different
                                          well.  Same bytes, wrong basins.
* ``random_atoms``                     -- atoms from a Gaussian, same shape, same head
                                          fit.

WHY ``landscape_shuffled`` IS A PROVED NULL FOR THIS READ
----------------------------------------------------------
The read is ``x_hat_k = Phi W_k G_k`` and ``W_k`` is **re-solved against the deployed
atoms** by ``solve_head``.  Permuting the ROWS of ``G_k`` by ``P`` replaces ``G_k`` with
``P G_k``; the ridge objective ``|| Phi W (P G) - X ||^2 + lam ||W||^2`` is minimised by
``W* P^T`` where ``W*`` minimises the unpermuted objective, because ``|| W P^T ||_F =
|| W ||_F``.  So ``Phi (W* P^T)(P G) = Phi W* G``: the read is **bit-identical**.  The
CMAZE-O VAL smoke measured exactly that -- the arm and this control agree to every
printed digit on all five rows, all three seeds
(``valsmoke/VAL-SMOKE.json``), and ``scripts/w15_cmaze_o_span.py`` checks
``max |read_CLU - read_ctrl|`` directly.

That is a statement about the READ VERB, not about CMAZE-O: any signed-span read whose
address book is re-solved against the atoms is invariant to atom ordering.  It is
reported so that the row-permutation control is never mistaken for evidence.

The floor and the rivals
------------------------
* ``OLookupRead`` -- the MATCHED-BYTES LOOKUP.  It holds whole stored latents, as many as
  the CLU's read holds floats, and answers with the mean of the stored items whose words
  agree most.  **It is a printed floor, never a rival** (COMMON.md).
* ``ORival`` -- MLP / LSTM / transformer / **continuous MLP** at MATCHED PARAMETERS, on
  the SAME frozen encoder and decoder.  Matched to the CLU's DEPLOYED READ floats (atoms
  plus address book).  The float64 sufficient statistic -- what buys exact deletion -- is
  declared separately and is NOT charged to them, because they are not asked to delete.
  Naive vs naive: they get the same interpolated one-hots the CLU's design is built from
  but **without the pair-interaction block**, because handing a learned model the CLU's
  factorial design would be handing it the CLU's inductive bias.

The BLIND columns
-----------------
A blind arm is the CMAZE-O analogue of ``w8_envscout.screen``'s blind planner: an arm
that composes only PART of the request and is graded by the unrelaxed exact judge.  They
are parameterless, they emit a state directly, and they are printed beside every row.

* ``BLIND-layout-free-cells``  -- the right layout picture and the right colours, but the
  zone words resolved by a LAYOUT-INDEPENDENT table.  Blind to ``layout x zone``, which
  is the one interaction CMAZE-O is built around.
* ``BLIND-colour-marginal``    -- the right layout and the right cells, colours from the
  training marginal.  Blind to the four colour words.
* ``BLIND-marginal``           -- both: everything from its own marginal.  The
  fully factorised generator.
* ``BLIND-no-layout``          -- the modal layout's picture, zones resolved in it,
  colours right.  Blind to the layout word.
* ``RANDOM``                   -- a legal state for a uniformly random request.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.core.gen_request_head import (HeadSpec, accumulate_gram,
                                        assert_head_request_only, head_read,
                                        request_design, solve_head)
from chlu.experiments.exp_w8_single_design import CovContentStore
from chlu.exp_w15_cmaze_o_env import (N_COLOURS, N_ZONES, N_SLOTS, OState,
                                      ORequest, render_state)
from w9_maze.arms import (head_floats, nearest_item_audit, store_floats,
                          ward_balanced_partition)

MATCH_TOL = 0.02
#: the controls the VAL smoke ran, in the order it ran them
CONTROLS = ("landscape_removed", "landscape_shuffled",
            "landscape_shuffled_across_wells", "random_atoms")
#: the full shipped set, including the within-well CHANNEL permutation added after the
#: VAL smoke proved the ROW permutation is a null.  ``scripts/w15_cmaze_o_span.py``
#: measures this one on all three seeds.
CONTROLS_SHIPPED = ("landscape_removed", "landscape_shuffled",
                    "landscape_shuffled_channels",
                    "landscape_shuffled_across_wells", "random_atoms")


# ------------------------------------------------------------------ the design
def make_grids(n_layouts: int):
    """The address book's words, as the grids ``request_design`` interpolates on.

    The colour grids run over ALL ``N_COLOURS`` words, including the VALUE hold-out, so
    an unseen colour is a real unseen WORD with an unsupported design column rather than
    a value clamped onto the nearest trained one.  ``out_of_range`` is therefore 0 on
    every row, and is printed.
    """
    return [np.arange(n_layouts, dtype=np.float64)] + \
           [np.arange(1, N_COLOURS + 1, dtype=np.float64) for _ in range(4)] + \
           [np.arange(N_ZONES, dtype=np.float64) for _ in range(4)]


def design_of(requests, grids, spec):
    """(N, 9) request values -> the design matrix.  No feature can enter here."""
    V = np.array([r.values() for r in requests], dtype=np.float64)
    return request_design(V, grids, spec)


# ------------------------------------------------------------------ the CLU
class OCluRead:
    """The CLU: a landscape of atoms addressed by nine words.  Nothing else reads it."""

    def __init__(self, Z_train, requests_train, grids, K=8, n_atoms=12, ridge=1.0,
                 pair_block=True, control=None, seed=0, well_channels=None):
        self.grids = grids
        self.K, self.n_atoms, self.control = int(K), int(n_atoms), control
        self.spec = HeadSpec([int(g.size) for g in grids], pair_block=pair_block,
                             ridge=float(ridge))
        Z = np.asarray(Z_train, np.float32)
        self.dim = int(Z.shape[1])
        if well_channels is None:
            self.well_channels, self.n_dead = ward_balanced_partition(Z, self.K, seed)
        else:
            #: a FORCED partition.  Used only by the deletion study, where the
            #: refit-without reference object must occupy the SAME wells for
            #: "store after deletion == store fitted without the rows" to be a
            #: statement about the rows rather than about the partition.
            self.well_channels, self.n_dead = list(well_channels), 0
        self.store = CovContentStore(Z, self.well_channels, self.n_atoms, self.dim)
        rng = np.random.default_rng(seed + 991)
        if control == "landscape_removed":
            self.store.modes = [np.zeros_like(M) for M in self.store.modes]
        elif control == "landscape_shuffled":
            #: the same bytes in the same basin, in the wrong ORDER.  Proved null for
            #: this read -- see the module docstring; kept and reported, never used as
            #: evidence.
            self.store.modes = [np.asarray(M)[rng.permutation(M.shape[0])]
                                for M in self.store.modes]
        elif control == "landscape_shuffled_channels":
            #: the same bytes in the same basin, on the wrong CHANNELS.  This one moves.
            self.store.modes = [np.asarray(M)[:, rng.permutation(M.shape[1])]
                                for M in self.store.modes]
        elif control == "landscape_shuffled_across_wells":
            #: the same bytes deployed in the WRONG basins
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
            self.store.modes = [rng.normal(size=M.shape).astype(np.float32)
                                / np.sqrt(M.shape[1]) for M in self.store.modes]
        elif control is not None:
            raise KeyError(control)
        self.fit_head(Z, requests_train)

    # -- fitting ----------------------------------------------------------
    def fit_head(self, Z, requests):
        Phi, _ = design_of(requests, self.grids, self.spec)
        A, Mx, _n = accumulate_gram([Phi], [np.asarray(Z, np.float64)],
                                    self.spec.width, self.dim)
        self.A, self.Mx = A, Mx
        self.heads = solve_head(A, Mx, self.store, self.spec.ridge)

    def refit_head_to_store(self):
        self.heads = solve_head(self.A, self.Mx, self.store, self.spec.ridge)

    def delete_rows_exact(self, X_rows, requests_deleted):
        """THE DELETION, applied to EVERY retained statistic, not only the store's.

        ``S, t, n`` and the head's own ``A = Phi^T Phi``, ``Mx = Phi^T X`` are all
        additive in the rows, so the deletion subtracts the deleted rows' contribution
        from all of them and the same closed form is re-solved.  No pass over the
        retained pool, no items kept, nothing read that the caller did not hand in.
        This is ``w9_maze.arms.CluRead.delete_rows_exact``'s mechanism, on nine words.
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

    # -- reading ----------------------------------------------------------
    def read(self, requests) -> np.ndarray:
        Phi, oor = design_of(requests, self.grids, self.spec)
        self.last_out_of_range = int(np.sum(oor))
        return head_read(self.heads, self.store, Phi)

    def assert_request_only(self, requests):
        Phi, _ = design_of(requests[:8], self.grids, self.spec)
        return assert_head_request_only(self.heads, self.store, Phi)

    # -- ledgers ----------------------------------------------------------
    def ledger(self) -> dict:
        sf = store_floats(self.store)
        hf = head_floats(self.heads)
        stat = int(sum(w.stat_floats() for w in self.store.stats))
        return {
            "read_floats": sf + hf,
            "store_atom_floats": sf,
            "head_floats": hf,
            "deployed_bytes_float32": 4 * (sf + hf),
            "byte_convention": ("DEPLOYED BYTES = 4 * (atom floats + address-book "
                                "floats), float32; this is what turns nine words into "
                                "a latent"),
            "statistic_floats_float64": stat,
            "statistic_bytes_float64": 8 * stat,
            "statistic_convention": ("the float64 sufficient statistic S,t,n plus the "
                                     "head's A,Mx is what buys EXACT deletion; it is "
                                     "declared BESIDE the deployed bytes, never netted "
                                     "into them, and is never matched to a rival"),
            "K": self.K, "n_atoms": self.n_atoms, "dim": self.dim,
            "design_width": int(self.spec.width), "dead_channels": int(self.n_dead),
            "deployed_rank_per_well": [int(r) for r in self.store.deployed_rank],
            "span_fraction": float(np.mean(
                [float(r) / float(c.size) for r, c in
                 zip(self.store.deployed_rank, self.well_channels)])),
        }


# ------------------------------------------------------------------ the floor
class OLookupRead:
    """The MATCHED-BYTES LOOKUP FLOOR.  A printed floor, never a rival."""

    def __init__(self, Z_train, requests_train, budget_floats=None, seed=0):
        Z = np.asarray(Z_train, np.float32)
        n, d = Z.shape
        keep = n if budget_floats is None else max(1, int(budget_floats) // int(d))
        rng = np.random.default_rng(seed + 7)
        idx = np.arange(n) if keep >= n else rng.choice(n, size=keep, replace=False)
        self.idx = np.sort(idx)
        self.Z = Z[self.idx]
        self.keys = np.array([r.words() for r in requests_train], np.int64)[self.idx]
        self.n_stored = int(self.Z.shape[0])
        self.floats = int(self.Z.size)
        #: the layout word is weighted above the others: two states in different mazes
        #: are not neighbours in any sense, which is the most generous reading of
        #: "nearest" for a lookup on this task.
        self.w = np.array([4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], np.float64)

    def read(self, requests) -> np.ndarray:
        Q = np.array([r.words() for r in requests], np.int64)
        out = np.zeros((Q.shape[0], self.Z.shape[1]), np.float32)
        for i in range(Q.shape[0]):
            agree = ((self.keys == Q[i]) * self.w).sum(axis=1)
            best = agree.max()
            sel = np.flatnonzero(agree >= best - 1e-9)
            out[i] = self.Z[sel].mean(axis=0)
        return out

    def ledger(self) -> dict:
        return {"items_held": self.n_stored, "read_floats": self.floats,
                "deployed_bytes_float32": 4 * self.floats,
                "byte_convention": "4 * (items held x latent width), float32"}


# ------------------------------------------------------------------ the rivals
def _count(model):
    return int(sum(v.size for v in jax.tree.leaves(eqx.filter(model, eqx.is_array))))


class MLP(eqx.Module):
    layers: list

    def __init__(self, key, n_in, h, d_out, depth=2):
        ks = jax.random.split(key, depth + 1)
        dims = [n_in] + [h] * depth + [d_out]
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=ks[i])
                       for i in range(depth + 1)]

    def __call__(self, x):
        for lyr in self.layers[:-1]:
            x = jax.nn.gelu(lyr(x))
        return self.layers[-1](x)


class LSTMHead(eqx.Module):
    emb: eqx.nn.Linear
    cell: eqx.nn.LSTMCell
    out: eqx.nn.Linear
    h: int

    def __init__(self, key, n_tok_in, h, d_out):
        k1, k2, k3 = jax.random.split(key, 3)
        self.emb = eqx.nn.Linear(n_tok_in, h, key=k1)
        self.cell = eqx.nn.LSTMCell(h, h, key=k2)
        self.out = eqx.nn.Linear(h, d_out, key=k3)
        self.h = int(h)

    def __call__(self, toks):
        state = (jnp.zeros(self.h), jnp.zeros(self.h))
        for t in range(toks.shape[0]):
            state = self.cell(jax.nn.gelu(self.emb(toks[t])), state)
        return self.out(state[0])


class TransformerHead(eqx.Module):
    """``w9_maze.baselines.TransformerHead`` with the token count as a parameter.

    The wave-9 class hard-codes four tokens; CMAZE-O has nine words.  Same block, same
    pre-norm residual arrangement, same mean-pool readout.
    """

    emb: eqx.nn.Linear
    pos: jax.Array
    attn: list
    ff: list
    ln: list
    out: eqx.nn.Linear

    def __init__(self, key, n_tok_in, h, d_out, n_tokens=9, layers=2, heads=4):
        ks = jax.random.split(key, 3 + 3 * layers)
        self.emb = eqx.nn.Linear(n_tok_in, h, key=ks[0])
        self.pos = jax.random.normal(ks[1], (n_tokens, h)) * 0.02
        self.attn = [eqx.nn.MultiheadAttention(heads, h, key=ks[3 + 3 * i])
                     for i in range(layers)]
        self.ff = [[eqx.nn.Linear(h, 2 * h, key=ks[4 + 3 * i]),
                    eqx.nn.Linear(2 * h, h, key=ks[5 + 3 * i])] for i in range(layers)]
        self.ln = [eqx.nn.LayerNorm(h) for _ in range(2 * layers)]
        self.out = eqx.nn.Linear(h, d_out, key=ks[2])

    def __call__(self, toks):
        x = jax.vmap(self.emb)(toks) + self.pos
        for i, at in enumerate(self.attn):
            y = jax.vmap(self.ln[2 * i])(x)
            x = x + at(y, y, y)
            y = jax.vmap(self.ln[2 * i + 1])(x)
            x = x + jax.vmap(lambda v: self.ff[i][1](jax.nn.gelu(self.ff[i][0](v))))(y)
        return self.out(x.mean(axis=0))


def _fit_width(build, target, lo=4, hi=1024):
    """Smallest width whose parameter count is nearest the target from either side."""
    best, best_gap = lo, None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = _count(build(mid))
        gap = abs(n - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = mid, gap
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best, _count(build(best))


def main_effect_design(requests, grids):
    """``[bias | main effects]`` only -- the rivals' input.  No pair block."""
    spec = HeadSpec([int(g.size) for g in grids], pair_block=False, ridge=1.0)
    Phi, _ = request_design(np.array([r.values() for r in requests], np.float64),
                            grids, spec)
    return np.asarray(Phi, np.float32), spec


def token_inputs(Phi_main, spec):
    """One token per word, zero-padded to a common width, plus a per-token bias."""
    lay = spec.layout
    cards, offs = lay["cardinalities"], lay["main_offsets"]
    F = len(cards)
    w = max(cards)
    out = np.zeros((Phi_main.shape[0], F, w + 1), np.float32)
    for f in range(F):
        out[:, f, :cards[f]] = Phi_main[:, offs[f]:offs[f] + cards[f]]
        out[:, f, w] = 1.0
    return out


def continuous_inputs(requests, grids):
    """The nine words as NUMBERS: min-max normalised value + a requested flag each.

    The ``cont_mlp`` convention of ``chlu.eval.rivals.w4_learners.ContMLPRival``: the
    input is the value, not a class index, so a word outside the trained range is simply
    a number no training row carried.
    """
    V = np.array([r.values() for r in requests], np.float64)
    out = np.zeros((V.shape[0], 2 * V.shape[1]), np.float32)
    for f, g in enumerate(grids):
        lo, hi = float(g[0]), float(g[-1])
        out[:, f] = (V[:, f] - lo) / max(hi - lo, 1e-9)
        out[:, V.shape[1] + f] = 1.0
    return out


class ORival:
    """One learned rival, trained to the item latents at MATCHED PARAMETERS."""

    KINDS = ("mlp", "cont_mlp", "lstm", "transformer")

    def __init__(self, kind, key, n_in, d_out, target_params, seq_in=None, n_tokens=9):
        self.kind = kind
        if kind in ("mlp", "cont_mlp"):
            build = (lambda h: MLP(key, n_in, h, d_out))
        elif kind == "lstm":
            build = (lambda h: LSTMHead(key, seq_in, h, d_out))
        elif kind == "transformer":
            #: the width must be a multiple of the head count, so the search runs over
            #: BLOCKS of 4 -- searching over h directly is non-monotonic in steps of 4
            #: and lands outside the 2 % match band.
            build4 = (lambda q: TransformerHead(key, seq_in, 4 * max(1, q), d_out,
                                                n_tokens=n_tokens))
            q, self.n_params = _fit_width(build4, int(target_params), lo=1, hi=256)
            self.width = 4 * max(1, q)
            self.model = build4(q)
            self.target_params = int(target_params)
            self.matched = bool(abs(self.n_params - target_params)
                                <= MATCH_TOL * target_params)
            return
        else:
            raise KeyError(kind)
        self.width, self.n_params = _fit_width(build, int(target_params))
        self.model = build(self.width)
        self.target_params = int(target_params)
        self.matched = bool(abs(self.n_params - target_params)
                            <= MATCH_TOL * target_params)

    def fit(self, X, Y, steps=3000, batch=64, lr=1e-3, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        n = int(X.shape[0])
        perm = rng.permutation(n)
        n_val = max(8, int(0.10 * n))
        vi, ti = perm[:n_val], perm[n_val:]
        Xt, Yt, Xv, Yv = X[ti], Y[ti], X[vi], Y[vi]

        def loss_fn(model, xb, yb):
            return jnp.mean((jax.vmap(model)(xb) - yb) ** 2)

        sched = optax.warmup_cosine_decay_schedule(1e-4, lr, int(0.05 * steps),
                                                   steps, 1e-5)
        opt = optax.adam(sched)
        state = opt.init(eqx.filter(self.model, eqx.is_array))

        @eqx.filter_jit
        def step(model, state, xb, yb):
            lo, gr = eqx.filter_value_and_grad(loss_fn)(model, xb, yb)
            upd, state = opt.update(gr, state, eqx.filter(model, eqx.is_array))
            return eqx.apply_updates(model, upd), state, lo

        @eqx.filter_jit
        def vloss(model):
            return loss_fn(model, jnp.asarray(Xv), jnp.asarray(Yv))

        model = self.model
        best, best_v, self.hist = model, np.inf, []
        for t in range(steps):
            idx = rng.integers(0, Xt.shape[0], size=min(batch, Xt.shape[0]))
            model, state, lo = step(model, state, jnp.asarray(Xt[idx]),
                                    jnp.asarray(Yt[idx]))
            if (t + 1) % 250 == 0:
                v = float(vloss(model))
                self.hist.append((t + 1, float(lo), v))
                if v < best_v:
                    best_v, best = v, model
                if verbose:
                    print("    %-12s step %5d train %.5f heldin-val %.5f"
                          % (self.kind, t + 1, float(lo), v), flush=True)
        self.model, self.val_mse = best, float(best_v)
        return self

    def read(self, X) -> np.ndarray:
        out = []
        for s in range(0, int(X.shape[0]), 256):
            out.append(np.asarray(jax.vmap(self.model)(jnp.asarray(X[s:s + 256]))))
        return np.concatenate(out, 0).astype(np.float32)

    def ledger(self) -> dict:
        return {"kind": self.kind, "width": int(self.width),
                "params": int(self.n_params), "target_params": self.target_params,
                "matched_within_2pct": self.matched,
                "byte_convention": "parameters x 4 bytes, float32",
                "deployed_bytes_float32": 4 * int(self.n_params)}


# ------------------------------------------------------------------ BLIND columns
class BlindArms:
    """Parameterless generators that compose only PART of the request.

    Each one emits a STATE directly and is graded by the same unrelaxed exact judge.
    Their tables are read off the TRAIN requests only.
    """

    def __init__(self, bank, train_requests):
        self.bank = bank
        W = np.array([r.words() for r in train_requests], np.int64)
        self.modal_layout = int(np.bincount(W[:, 0], minlength=len(bank)).argmax())
        self.modal_colour = [int(np.bincount(W[:, 1 + i]).argmax()) for i in range(4)]
        #: the LAYOUT-INDEPENDENT zone -> cell table: for every (slot, zone) the cell
        #: that is most often the answer across the layouts train actually saw.
        tab = np.zeros((N_SLOTS, N_ZONES), np.int64)
        for s in range(N_SLOTS):
            for z in range(N_ZONES):
                cnt = {}
                for lid in np.unique(W[:, 0]):
                    v = int(bank[int(lid)].slot_cell[s, z])
                    cnt[v] = cnt.get(v, 0) + int((W[:, 0] == lid).sum())
                tab[s, z] = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        self.free_cell = tab

    def _cells_from_table(self, req):
        return [(int(self.free_cell[s, z]) // 15, int(self.free_cell[s, z]) % 15)
                for s, z in enumerate((req.za, req.zk, req.zd, req.zg))]

    def state(self, name: str, req: ORequest, rng=None) -> OState:
        lay = self.bank[req.layout]
        true_cells = [lay.cell(s, z) for s, z in
                      enumerate((req.za, req.zk, req.zd, req.zg))]
        cols = [req.ca, req.ck, req.cd, req.ct]
        if name == "BLIND-layout-free-cells":
            cells, pic = self._cells_from_table(req), lay
        elif name == "BLIND-colour-marginal":
            cells, pic = true_cells, lay
            cols = list(self.modal_colour)
        elif name == "BLIND-marginal":
            cells, pic = self._cells_from_table(req), lay
            cols = list(self.modal_colour)
        elif name == "BLIND-no-layout":
            pic = self.bank[self.modal_layout]
            cells = [pic.cell(s, z) for s, z in
                     enumerate((req.za, req.zk, req.zd, req.zg))]
        elif name == "RANDOM":
            r = rng
            pic = self.bank[int(r.integers(len(self.bank)))]
            zs = [int(r.integers(N_ZONES)) for _ in range(4)]
            cells = [pic.cell(s, zs[s]) for s in range(4)]
            cols = [int(r.integers(1, N_COLOURS + 1)) for _ in range(4)]
        else:
            raise KeyError(name)
        return OState(grid=pic.grid, tint_mask=pic.tint_mask, ct=int(cols[3]),
                      agent=cells[0], ca=int(cols[0]), key=cells[1], ck=int(cols[1]),
                      door=cells[2], cd=int(cols[2]), goal=cells[3])


BLIND_NAMES = ("BLIND-layout-free-cells", "BLIND-colour-marginal", "BLIND-marginal",
               "BLIND-no-layout", "RANDOM")

__all__ = ["OCluRead", "OLookupRead", "ORival", "BlindArms", "BLIND_NAMES", "CONTROLS",
           "make_grids", "design_of", "main_effect_design", "token_inputs",
           "continuous_inputs", "nearest_item_audit", "MATCH_TOL"]
