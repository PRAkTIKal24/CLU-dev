"""CMAZE-T arms: the CLU single design, the matched-parameter rivals, the floors.

Every arm here answers the SAME question -- given a transition request, emit the code of
the next state -- and every arm is graded by the SAME exact transition judge
(``exp_w15_cmaze_t.judge_transition``).  So a margin between two arms is a margin on
the transition, not on a featurisation.

WHAT EACH ARM IS ALLOWED TO SEE
    CLU                 the five REQUEST WORDS and nothing else.  ``head_read`` has no
                        feature argument; charter B.3 holds by signature.
    MLP / LSTM / TFM    the FROZEN state encoder's features around the agent, the global
                        pooled map, the agent's own dynamic state, and the action.
                        One observation function, shared; the trunks differ only in how
                        they combine observations across time.
    lookup floors       a raw discrete key, uncanonicalised (``lookup-raw``), the five
                        words (``lookup-words``), or env-scout's colour-canonicalised
                        fragment (``frag``).  Printed floors, never rivals.

TWO FROZEN OBJECTS, BOTH DECLARED OUTSIDE THE DELETABLE OBJECT
    ``Codec``           next-state code (21 dims at k=4) <-> latent (D dims).  Trained by
                        RECONSTRUCTION on TRAIN codes only and frozen before any arm
                        exists.  Its own encode->decode->judge score on the train rows is
                        the PIPELINE CEILING and is printed beside every row.
    state encoder       ``w9_harness.encoder``, a conv autoencoder on maze images,
                        reconstruction only, frozen before any arm exists.
    ⛔ Neither is downdated, refitted or adapted when a deletion happens.  Deletion is
    STORE-SCOPED and the post-deletion adaptation is data-free.
"""
from __future__ import annotations
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chlu.core.gen_request_head import (HeadSpec, accumulate_gram,
                                        assert_head_request_only, head_read,
                                        request_design, solve_head)
from chlu.experiments.exp_w8_single_design import CovContentStore
from w9_harness import encoder as w9enc
from w9_harness import features as w9feat
from w9_harness import render as w9render
from w9_harness.arms import D_MODEL, HEAD_KEY, TRUNKS, ActionHead, n_params
import chlu.exp_w15_cmaze_t as T
class Codec(eqx.Module):
    """Next-state code <-> latent.  Reconstruction only; frozen before any arm exists."""

    e1: eqx.nn.Linear
    e2: eqx.nn.Linear
    d1: eqx.nn.Linear
    d2: eqx.nn.Linear

    def __init__(self, key, cdim, latent, width=96):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.e1 = eqx.nn.Linear(cdim, width, key=k1)
        self.e2 = eqx.nn.Linear(width, latent, key=k2)
        self.d1 = eqx.nn.Linear(latent, width, key=k3)
        self.d2 = eqx.nn.Linear(width, cdim, key=k4)

    def enc(self, v):
        return self.e2(jax.nn.gelu(self.e1(v)))

    def dec(self, z):
        return self.d2(jax.nn.gelu(self.d1(z)))


def _block_ce(logits, target, blocks):
    loss = 0.0
    for name in ("legal", "disp", "event", "pocket"):
        o, c = blocks[name]
        loss = loss + optax.softmax_cross_entropy(logits[:, o:o + c],
                                                  target[:, o:o + c]).mean()
    return loss


def train_codec(codes, k, latent=128, steps=2500, batch=128, lr=3e-3, seed=0,
                verbose=True):
    """Train the codec by per-block softmax reconstruction on TRAIN codes only."""
    t0 = time.time()
    blocks = T.code_blocks(k)
    cdim = T.code_dim(k)
    model = Codec(jax.random.PRNGKey(7000 + seed), cdim, latent)
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, 1e-2))
    st = opt.init(eqx.filter(model, eqx.is_array))

    def loss_fn(m, xb):
        z = jax.vmap(m.enc)(xb)
        lg = jax.vmap(m.dec)(z)
        return _block_ce(lg, xb, blocks)

    @eqx.filter_jit
    def upd(m, st, xb):
        l, g = eqx.filter_value_and_grad(loss_fn)(m, xb)
        u, st = opt.update(g, st, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, u), st, l

    X = np.asarray(codes, np.float32)
    rng = np.random.default_rng(seed)
    hist = []
    for s in range(steps):
        idx = rng.integers(0, X.shape[0], size=min(batch, X.shape[0]))
        model, st, l = upd(model, st, jnp.asarray(X[idx]))
        if s % 500 == 0 or s == steps - 1:
            hist.append((s, float(l)))
    rep = {"steps": steps, "latent": int(latent), "code_dim": int(cdim),
           "final_loss": hist[-1][1], "loss_curve": hist,
           "params": n_params(model), "seconds": time.time() - t0,
           "objective": "per-block softmax cross-entropy on the next-state code; "
                        "no request, no arm, no judge enters the loss"}
    if verbose:
        print("  CODEC latent %d  loss %.5f  [%.1fs]"
              % (latent, rep["final_loss"], rep["seconds"]), flush=True)
    return model, rep


@eqx.filter_jit
def _cenc(model, X):
    return jax.vmap(model.enc)(X)


@eqx.filter_jit
def _cdec(model, Z):
    return jax.vmap(model.dec)(Z)


def _pad_to(X, batch):
    """Pad the leading axis up to a multiple of ``batch``.

    ⚠ Not cosmetic: ``jax.jit`` retraces on every new leading dimension, and a rollout
    whose live set shrinks by one each step would recompile at every step.  Padding
    pins the traced shape.  The padding rows are sliced off before anything is judged.
    """
    n = int(X.shape[0])
    m = ((n + batch - 1) // batch) * batch
    if m == n:
        return X, n
    return np.concatenate([X, np.zeros((m - n,) + X.shape[1:], X.dtype)], 0), n


def codec_encode(model, codes, batch=512):
    X, n = _pad_to(np.asarray(codes, np.float32), batch)
    out = [np.asarray(_cenc(model, jnp.asarray(X[s:s + batch])))
           for s in range(0, X.shape[0], batch)]
    return np.concatenate(out, 0)[:n].astype(np.float32)


def codec_decode(model, Z, batch=512):
    Za, n = _pad_to(np.asarray(Z, np.float32), batch)
    out = [np.asarray(_cdec(model, jnp.asarray(Za[s:s + batch])))
           for s in range(0, Za.shape[0], batch)]
    return np.concatenate(out, 0)[:n]


def maze_images(insts):
    return np.stack([w9render.render(i) for i in insts]).astype(np.float32)


def train_state_encoder(insts, feat_dim=12, width=32, steps=1500, seed=0, verbose=True,
                        restarts=1, accept_bce=None):
    """``w9_harness.encoder`` on TRAIN maze images.  Reconstruction only, then frozen.

    ``restarts``/``accept_bce`` are w9's own reliability guard: a single fit of this
    autoencoder is not reliable, and the restart is selected by RECONSTRUCTION LOSS AND
    NOTHING ELSE, so no arm, action or judge takes part in choosing it.
    """
    X = maze_images(insts)
    enc, rep = w9enc.train(X, feat_dim=feat_dim, width=width, steps=steps, seed=seed,
                           verbose=verbose, restarts=restarts, accept_bce=accept_bce)
    return enc, rep


def feature_maps(enc, insts):
    return w9enc.features(enc, maze_images(insts))


def obs_of(inst, S, st, action, gsum=None):
    """The ONE observation every learned rival consumes: w9's shared vector + action."""
    a = np.zeros(5, np.float32)
    a[T.ACTIONS.index(action)] = 1.0
    return np.concatenate([w9feat.obs(inst, S, st, gsum), a])


def obs_dim(feat_dim, k):
    return w9feat.obs_dim(feat_dim, k) + 5


def build_obs(rows, insts, feats, gsums):
    return np.stack([obs_of(insts[r.maze], feats[r.maze], r.state(), r.action,
                            gsums[r.maze]) for r in rows]).astype(np.float32)


def build_codes(rows, k):
    return np.stack([T.encode_outcome(k, r.legal, r.disp, r.ev, r.pocket_next)
                     for r in rows]).astype(np.float32)


def build_words(rows):
    return np.array([r.words for r in rows], np.float64)


def colour_blind_alphabets(k):
    """``(carrying?, key-here?, door kind, tint?, direction)`` -- colour identity gone."""
    return [2, 2, 4, 2, 5]


def colour_blind_words(words, k):
    """Collapse colour IDENTITY out of the request, keeping everything else.

    ⭐ This is the LEARNED BLIND-COLOUR arm's request: it still sees that there is a
    door, that the door is closed, that the floor is tinted and that the pocket is full
    -- it simply cannot tell whether the colours MATCH.  Since RULE A is
    ``pocket == door colour`` and RULE B is ``pocket == tint colour``, a colour-blind
    request cannot express either rule, which is exactly what makes it the falsifier.
    """
    W = np.asarray(words, np.int64)
    out = np.zeros_like(W)
    out[:, 0] = (W[:, 0] > 0).astype(np.int64)
    out[:, 1] = (W[:, 1] > 0).astype(np.int64)
    d = W[:, 2]
    out[:, 2] = np.where(d == 0, 0, np.where(d <= k, 1, np.where(d == k + 1, 2, 3)))
    out[:, 3] = (W[:, 3] > 0).astype(np.int64)
    out[:, 4] = W[:, 4]
    return out.astype(np.float64)


class TransitionClu:
    """The CLU single design on CMAZE-T.

    ``request_design`` over the five words gives ``[bias | main | pair]``;
    ``CovContentStore`` carries the landscape as the sufficient statistic
    ``(S_k, t_k, n_k)`` per well; ``solve_head`` is the closed-form end-to-end ridge.
    ⛔ The read is ``head_read(heads, store, Phi)`` -- there is no feature argument, so
    charter B.3 holds by signature and not by inspection.
    """

    CONTROLS = ("landscape_removed", "landscape_shuffled",
                "landscape_shuffled_across_wells", "random_atoms")

    def __init__(self, Z_train, words_train, grids, K=8, n_atoms=6, ridge=1.0,
                 pair_block=True, control=None, seed=0, well_channels=None):
        self.grids = grids
        self.K, self.n_atoms, self.control = int(K), int(n_atoms), control
        self.spec = HeadSpec([int(g.size) for g in grids], pair_block=pair_block,
                             ridge=float(ridge))
        Z = np.asarray(Z_train, np.float32)
        self.dim = int(Z.shape[1])
        if well_channels is not None:
            #: ⭐ used ONLY by the deletion's "fitted without the rows" reference.  The
            #: store-scoped exactness claim is about the retained statistic GIVEN the
            #: partition; recomputing ``ward_balanced_partition`` on the surviving rows
            #: would put different CHANNELS in each well and compare two different
            #: objects.  Measured before this was fixed: max relative delta 1.26 with
            #: identical per-well counts -- the counts matched and the statistics could
            #: not, which is what gave the bug away.
            self.well_channels = [np.asarray(c, np.int64) for c in well_channels]
            self.n_dead = 0
        else:
            self.well_channels, self.n_dead = ward_balanced_partition(Z, self.K, seed)
        self.store = CovContentStore(Z, self.well_channels, self.n_atoms, self.dim)
        rng = np.random.default_rng(seed + 991)
        if control == "landscape_removed":
            self.store.modes = [np.zeros_like(M) for M in self.store.modes]
        elif control == "landscape_shuffled":
            #: the same bytes, in the RIGHT basin, on the WRONG CHANNELS -- each well's
            #: atom block has its CHANNEL (column) order permuted.
            #: ⚠ The first version of this control permuted the atom ROWS instead.  That
            #: was measured and it is a NO-OP BY CONSTRUCTION: the address book is fitted
            #: AFTER the control is applied, and ``head_read`` computes ``(Phi W_k) G_k``,
            #: so a row permutation of ``G_k`` is absorbed exactly by the corresponding
            #: column permutation of ``W_k``.  Measured on the 12-maze dry run: the
            #: row-permuted control scored 0.8195 on `unseen n-way`, identical to the
            #: CLU's 0.8195, to every digit.  A column permutation moves the DEPLOYED
            #: SPAN and cannot be absorbed.  Recorded rather than quietly fixed.
            self.store.modes = [np.asarray(M)[:, rng.permutation(M.shape[1])]
                                for M in self.store.modes]
        elif control == "landscape_shuffled_across_wells":
            #: the same bytes, deployed in the WRONG basins -- every well's atom block
            #: is dealt to a different well, truncated or padded to that well's width.
            pool = [np.asarray(M) for M in self.store.modes]
            perm = rng.permutation(self.K)
            new = []
            for j, ch in enumerate(self.well_channels):
                src = pool[perm[j]]
                d = int(ch.size)
                M = np.zeros((self.n_atoms, d), np.float32)
                w = min(d, src.shape[1])
                M[:, :w] = src[:, :w]
                new.append(M)
            self.store.modes = new
        elif control == "random_atoms":
            self.store.modes = [rng.normal(size=M.shape).astype(np.float32)
                                / np.sqrt(M.shape[1]) for M in self.store.modes]
        self.fit_head(Z, words_train)

    # -- fit ---------------------------------------------------------------
    def design(self, words):
        return request_design(np.asarray(words, np.float64), self.grids, self.spec)

    def fit_head(self, Z, words):
        Phi, _ = self.design(words)
        A, Mx, _n = accumulate_gram([Phi], [np.asarray(Z, np.float64)],
                                    self.spec.width, self.dim)
        self.A, self.Mx = A, Mx
        self.heads = solve_head(A, Mx, self.store, self.spec.ridge)

    def read(self, words):
        Phi, oor = self.design(words)
        self.last_out_of_range = oor
        return head_read(self.heads, self.store, Phi)

    def assert_request_only(self, words):
        Phi, _ = self.design(np.asarray(words)[:8])
        return assert_head_request_only(self.heads, self.store, Phi)

    # -- deletion ----------------------------------------------------------
    def delete_rows_exact(self, X_rows, words_deleted):
        """THE DELETION, applied to EVERY retained statistic, not only the store's.

        ``S_k -= X_d^T X_d ; t_k -= sum X_d ; n_k -= m`` and
        ``A -= Phi_d^T Phi_d ; Mx -= Phi_d^T X_d``, then the SAME closed form is
        re-solved.  ⭐ No pass over the retained pool, no items kept, nothing read that
        the caller did not hand in.  ⛔ The codec and the state encoder are outside this
        object and are not touched.
        """
        t0 = time.time()
        Xd = np.asarray(X_rows, np.float32)
        Phi_d, _ = self.design(words_deleted)
        rec = self.store.delete_rows(Xd)
        P = np.asarray(Phi_d, np.float64)
        self.A = self.A - P.T @ P
        self.Mx = self.Mx - P.T @ np.asarray(Xd, np.float64)
        self.heads = solve_head(self.A, self.Mx, self.store, self.spec.ridge)
        rec["head_statistic_downdated"] = True
        rec["seconds"] = time.time() - t0
        rec["mechanism_head"] = ("A -= Phi_d^T Phi_d, Mx -= Phi_d^T X_d, then the same "
                                 "closed form is re-solved; the pool is never touched")
        rec["frozen_objects_touched"] = []
        return rec

    # -- ledger ------------------------------------------------------------
    def ledger(self):
        sf = int(sum(np.asarray(M).size for M in self.store.modes))
        hf = int(sum(np.asarray(W).size for W in self.heads))
        stat = int(sum(w.stat_floats() for w in self.store.stats))
        dk = [int(c.size) for c in self.well_channels]
        return {
            "read_floats": sf + hf, "store_atom_floats": sf, "head_floats": hf,
            "deployed_bytes_float32": 4 * (sf + hf),
            "statistic_floats_float64": stat, "statistic_bytes_float64": 8 * stat,
            "byte_convention": ("deployed bytes = 4 x (atom floats + address-book "
                                "floats), the object that produces a read; the float64 "
                                "sufficient statistic is what buys EXACT deletion and is "
                                "declared beside it, never netted in"),
            "K": self.K, "n_atoms": self.n_atoms, "dim": self.dim,
            "design_width": int(self.spec.width), "dead_channels": int(self.n_dead),
            "well_widths": dk,
            "span_fraction_n_atoms_over_d_k": [self.n_atoms / max(d, 1) for d in dk],
            "deployed_rank_per_well": [int(r) for r in self.store.deployed_rank],
        }


def ward_balanced_partition(Z, K, seed=0):
    """Label-free equal-width wells from the correlation dendrogram's leaf order.

    Transcribed from ``w9_maze.arms`` (itself ``exp_gen_placement.py``'s
    ``ward_balanced`` arm), so the partition CMAZE-T deploys is the one the programme
    already ships.  A channel with zero variance is DEAD and is dealt to the smallest
    well.  ⛔ It sees no label.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage

    X = np.asarray(Z, np.float64)
    D = int(X.shape[1])
    sd = X.std(axis=0)
    dead = sd <= 0
    live = np.flatnonzero(~dead)
    corr = np.nan_to_num(np.corrcoef(X[:, live], rowvar=False), nan=0.0)
    order = live[leaves_list(linkage(np.abs(corr), method="ward"))]
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


class TransitionPolicy(eqx.Module):
    """One trunk + the ONE shared head shape, emitting the next-state code logits."""

    trunk: eqx.Module
    head: ActionHead

    def __init__(self, kind, obs_d, width, n_out, key, d_model=D_MODEL, **kw):
        self.trunk = TRUNKS[kind](obs_d, d_model, width, key, **kw)
        self.head = ActionHead(d_model, n_out, HEAD_KEY)      # same key for every arm

    def __call__(self, seq):
        return jax.vmap(self.head)(self.trunk(seq))


def build_policy(kind, obs_d, width, n_out, seed=0, **kw):
    return TransitionPolicy(kind, obs_d, width, n_out,
                            jax.random.PRNGKey(1000 + seed), **kw)


def match_widths(obs_d, n_out, target, kinds=("mlp", "lstm", "transformer"),
                 lo=4, hi=200, tmax=16, verbose=True):
    """Closest achievable parameter count per trunk against a COMMON target.

    ⭐ The target is the CLU's READ-PATH float count (atoms + address book).  ⛔ The
    memory ledger is printed separately and is never matched: naive vs naive is matched
    on the read path only.  Integer layer shapes cannot hit a target exactly, so the
    count ACHIEVED and the residual are reported rather than a match claimed.
    """
    def _scan(kind, widths, kw):
        """Coarse-to-fine so the search costs tens of model builds, not thousands.
        The count is monotone in the width, so a coarse pass plus a +-step refinement
        finds the same argmin a full sweep would."""
        best = None
        for w in widths:
            if w < lo or w > hi:
                continue
            p = n_params(build_policy(kind, obs_d, w, n_out, **kw))
            if best is None or abs(p - target) < abs(best[1] - target):
                best = (w, p, kw)
        return best

    out = {}
    for kind in kinds:
        best = None
        cfgs = ([dict(n_layers=l, heads=h, tmax=tmax) for l in (1, 2) for h in (2, 4)]
                if kind == "transformer" else [{}])
        for kw in cfgs:
            stride = kw.get("heads", 1) * 4 if kind == "transformer" else 8
            c = _scan(kind, range(lo, hi + 1, stride), kw)
            if c is None:
                continue
            step = kw.get("heads", 1) if kind == "transformer" else 1
            f = _scan(kind, range(c[0] - stride, c[0] + stride + 1, step), kw)
            c = f if (f is not None and abs(f[1] - target) < abs(c[1] - target)) else c
            if best is None or abs(c[1] - target) < abs(best[1] - target):
                best = c
        out[kind] = best
    if verbose:
        print("  MATCHED PARAMETERS (target %d = CLU read floats, obs %d, out %d)"
              % (target, obs_d, n_out))
        for kd, (w, p, kw) in out.items():
            extra = "" if not kw else "  " + " ".join("%s=%s" % (a, b)
                                                      for a, b in kw.items()
                                                      if a != "tmax")
            print("    %-12s width %3d  params %7d  (%+.2f%%)%s"
                  % (kd, w, p, 100.0 * (p - target) / max(target, 1), extra))
    return out


def _seq_loss(policy, X, Y, M, blocks):
    """Masked per-block softmax cross-entropy over padded sequences."""
    logits = jax.vmap(policy)(X)                              # (B, T, cdim)
    tot = 0.0
    for name in ("legal", "disp", "event", "pocket"):
        o, c = blocks[name]
        ce = optax.softmax_cross_entropy(logits[:, :, o:o + c], Y[:, :, o:o + c])
        tot = tot + (ce * M).sum() / jnp.maximum(M.sum(), 1.0)
    return tot


@eqx.filter_jit
def _pforward(policy, X):
    return jax.vmap(policy)(X)


def policy_predict(policy, Obs, k, batch=256):
    """Single transitions, presented as length-1 sequences (position 0 of a causal
    forward pass is exactly the prefix of length 1 for every trunk)."""
    O, n = _pad_to(np.asarray(Obs, np.float32)[:, None, :], batch)
    out = []
    for s in range(0, O.shape[0], batch):
        lg = np.asarray(_pforward(policy, jnp.asarray(O[s:s + batch])))[:, 0, :]
        out.append(T.decode_codes(k, lg))
    return np.concatenate(out, 0)[:n]


def majority_code(rows, k):
    from collections import Counter as C
    c = C((r.legal, r.disp, r.ev, r.pocket_next) for r in rows)
    return c.most_common(1)[0][0]


def build_lookup(rows, keyfn):
    from collections import Counter as C, defaultdict as dd
    tab = dd(C)
    for r in rows:
        tab[keyfn(r)][(r.legal, r.disp, r.ev, r.pocket_next)] += 1
    return {kk: v.most_common(1)[0][0] for kk, v in tab.items()}


def lookup_predict(rows, keyfn, table, fallback):
    out, miss = [], 0
    for r in rows:
        v = table.get(keyfn(r))
        if v is None:
            miss += 1
            v = fallback
        out.append(v)
    return np.array(out, np.int64), {"coverage": 1.0 - miss / max(len(rows), 1),
                                     "table_size": len(table)}


def words_key(r):
    return r.words


def make_raw_key(insts, radius=1):
    """Uncanonicalised local code patch + pocket + action.  ⛔ No colour canonicalisation
    -- that is what separates this floor from the fragment lookup."""
    from w8_envscout.maze_colour import FLOOR, UNTAKEN, USED
    from w9_harness.dynamics import key_homes

    homes = {i: key_homes(inst) for i, inst in enumerate(insts)}

    def key(r):
        inst = insts[r.maze]
        g, tint, door = inst.grid, inst.tint, inst.door
        H, W = g.shape
        k = int(inst.n_colours)
        patch = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r.pos[0] + dr, r.pos[1] + dc
                if not (0 <= rr < H and 0 <= cc < W) or g[rr, cc] != FLOOR:
                    patch.append(0)
                    continue
                dcol = int(door[rr, cc])
                if dcol:
                    patch.append(2 + 2 * k if r.ks[dcol - 1] == USED else 1 + k + dcol)
                    continue
                kc = homes[r.maze].get((rr, cc), 0)
                if kc and r.ks[kc - 1] == UNTAKEN:
                    patch.append(3 + 2 * k + kc)
                    continue
                tcol = int(tint[rr, cc])
                patch.append(1 + tcol if tcol else 1)
        pocket = 0
        for i, v in enumerate(r.ks):
            if v == 1:
                pocket = i + 1
        return (tuple(patch), pocket, r.action)
    return key


def make_frag_key(insts, radius=1):
    """env-scout's COLOUR-CANONICALISED fragment, reused unchanged from
    ``w8_envscout.screen.canon_features`` -- the most generous featurisation a retrieval
    policy can be given.  REPORTED COLUMN; never an abort condition."""
    from w8_envscout.screen import canon_features

    def key(r):
        return (canon_features(insts[r.maze], r.pos, list(r.ks), radius), r.action)
    return key


def score_rows(insts, rows, emissions, label=""):
    """The exact transition judge, and only it.

    ``emissions`` is (N, 4) integer ``(legal, disp, ev, pocket_next)``.
    """
    from collections import Counter as C
    ok, reasons = 0, C()
    for r, e in zip(rows, emissions):
        good, why, _g = T.judge_transition(insts[r.maze], r.state(), r.action, e)
        ok += good
        if not good:
            reasons[why] += 1
    n = max(len(rows), 1)
    return {"label": label, "n": len(rows), "exact": ok / n,
            "failure_modes": dict(reasons.most_common(5))}


def rollout_validity(insts, rollouts, predict_batch, label=""):
    """Chain GENERATED states, LOCKSTEP across rollouts.

    A rollout is valid iff **every** step is exactly right and the goal is entered
    within the instance's budget.  ⭐ The state fed to step ``t+1`` is the arm's OWN
    generated state, never the engine's; the moment a step is wrong the rollout is dead
    and nothing re-synchronises it.

    ``predict_batch(maze_idx_list, states, actions) -> (N, 4)``.  All live rollouts
    advance together so an arm pays ONE forward pass per step instead of one per
    rollout -- the same lockstep discipline as ``w9_harness.rollout``.
    """
    n = len(rollouts)
    if n == 0:
        return {"label": label, "n_rollouts": 0,
                "DECLARED_NOT_RUN": "no rollouts in this bank"}
    states = [T.initial_state(insts[ro["maze"]]) for ro in rollouts]
    alive = np.ones(n, bool)
    good = np.ones(n, bool)
    steps_ok, steps_tot = 0, 0
    Tmax = max(len(ro["actions"]) for ro in rollouts)
    first_error = np.full(n, -1, np.int64)
    for t in range(Tmax):
        idx = [i for i in range(n) if alive[i] and t < len(rollouts[i]["actions"])]
        if not idx:
            break
        mz = [rollouts[i]["maze"] for i in idx]
        sts = [states[i] for i in idx]
        acts = [rollouts[i]["actions"][t] for i in idx]
        em = predict_batch(mz, sts, acts)
        for j, i in enumerate(idx):
            inst = insts[rollouts[i]["maze"]]
            okj, _why, gen = T.judge_transition(inst, states[i], acts[j], em[j])
            steps_tot += 1
            steps_ok += bool(okj)
            if not okj:
                good[i] = False
                alive[i] = False
                first_error[i] = t
                continue
            states[i] = gen
    valid = reached = 0
    for i, ro in enumerate(rollouts):
        inst = insts[ro["maze"]]
        hit_goal = bool(good[i] and states[i].pos == inst.goal)
        reached += hit_goal
        valid += bool(hit_goal and len(ro["actions"]) <= int(inst.budget))
    fe = first_error[first_error >= 0]
    return {"label": label, "n_rollouts": n, "validity": valid / n,
            "goal_reached_rate": reached / n,
            "all_steps_exact_rate": float(good.mean()),
            "per_step_exactness": steps_ok / max(steps_tot, 1),
            "steps_attempted": int(steps_tot),
            "mean_rollout_length": float(np.mean([len(r["actions"])
                                                  for r in rollouts])),
            "mean_first_error_step": float(fe.mean()) if fe.size else None}


