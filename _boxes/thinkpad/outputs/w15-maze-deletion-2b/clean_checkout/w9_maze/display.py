"""The DISPLAY rows: refusal, deletion, and composition after deletion.

These are inductive-bias rows.  Wave-9 rule 3 is binding: a DISPLAY row is NEVER
quoted as a RESULT, its only comparator is a supervised refuser trained on impossible
examples, and our line is that no impossible example was ever seen.

Two kinds of deletion, and they are not the same kind, so they are never presented as
if they were:

* CONTENT DELETION (the green target).  Every item written under the word is removed
  from the sufficient statistic by a rank-m subtract -- ``S_k -= X^T X``,
  ``t_k -= sum x``, ``n_k -= m`` -- the atoms are re-derived, and the address book is
  carried across the basis change by ``W_k <- W_k R_k``.  No refit, no pass over the
  pool, the items were never kept.  ``downdate_vs_refit`` measures the residual against
  a from-scratch refit on the surviving rows, which is the wave-8 property this hub is
  asking to transfer.
* INTERFACE DELETION (move-left).  The action word is removed from the emission
  interface, so the executor can no longer emit it and the certificate can no longer
  plan with it.  NO ATOM IS TOUCHED.  It therefore carries none of the exactness claim
  above, and it is labelled as an interface deletion everywhere it appears.
"""

from __future__ import annotations

import numpy as np

from chlu.experiments.exp_w8_single_design import CovContentStore
from w9_maze import evaluate
from w9_maze.encoder import decode_plans


def downdate_vs_refit(Z_all, keep_mask, well_channels, n_atoms, dim):
    """Exactness of the deletion: downdate the statistic vs refit from the survivors.

    Returns the residuals on the retained statistic and on the deployed atoms.  The
    atom comparison is made on the SUBSPACE (the principal angles), because an
    eigenvector's sign and any rotation inside a degenerate eigenspace are gauge, not
    content -- the wave-8 module makes the same distinction and for the same reason.
    """
    Z = np.asarray(Z_all, np.float32)
    full = CovContentStore(Z, well_channels, n_atoms, dim)
    drop = Z[~np.asarray(keep_mask, bool)]
    receipt = full.delete_rows(drop)
    refit = CovContentStore(Z[np.asarray(keep_mask, bool)], well_channels, n_atoms, dim)

    s_res, sub_res = [], []
    for wd, wr in zip(full.stats, refit.stats):
        den = max(1.0, float(np.abs(wr.S).max()))
        s_res.append(float(np.abs(wd.S - wr.S).max() / den))
    for Ud, Ur in zip(full.span_bases(), refit.span_bases()):
        r = min(Ud.shape[1], Ur.shape[1])
        if r == 0:
            sub_res.append(0.0)
            continue
        P = Ud[:, :r] @ Ud[:, :r].T - Ur[:, :r] @ Ur[:, :r].T
        sub_res.append(float(np.abs(P).max()))
    return {
        "receipt": receipt,
        "statistic_max_rel_residual": float(max(s_res)),
        "subspace_max_projector_residual": float(max(sub_res)),
        "n_rows_removed": int((~np.asarray(keep_mask, bool)).sum()),
        "n_rows_kept": int(np.asarray(keep_mask, bool).sum()),
        "refit": False, "passes_over_retained_data": 0, "items_kept": False,
    }


def delete_word_content(clu, Z_train, requests_train, factor, value,
                        exact_head=False):
    """Remove every item written under one word, exactly, and carry the address book.

    The word's own address-book columns are zeroed as well: the atoms that grounded it
    are gone, so leaving the interface pointing at them would be the address book
    answering for content that no longer exists.
    """
    keys = np.array([[r.m, r.a, r.t, r.b] for r in requests_train], np.int64)
    col = {"maze": 0, "agent": 1, "target": 2, "budget": 3}[factor]
    drop = keys[:, col] == int(value)
    Zd = np.asarray(Z_train, np.float32)[drop]
    if exact_head:
        rec = clu.delete_rows_exact(
            Zd, [r for r, d in zip(requests_train, drop) if d])
    else:
        rec = clu.store.delete_rows(Zd)
        clu.carry_basis_change()
    lay = clu.spec.layout
    off = lay["main_offsets"][col]
    dead_cols = [off + int(value)]
    for (i, j), o in zip(lay["pairs"], lay["pair_offsets"]):
        cards = lay["cardinalities"]
        if i == col:
            for b in range(cards[j]):
                dead_cols.append(o + int(value) * cards[j] + b)
        elif j == col:
            for a in range(cards[i]):
                dead_cols.append(o + a * cards[j] + int(value))
    for W in clu.heads:
        W[np.asarray(dead_cols, np.int64), :] = 0.0
    rec["n_items_deleted"] = int(drop.sum())
    rec["address_book_columns_cleared"] = len(dead_cols)
    rec["word"] = "%s=%d" % (factor, value)
    return rec


def route_uses(bank, rows, action="W"):
    return np.array([action in bank[m].route[(i, j)] for (_r, m, i, j) in rows], bool)


def refusal_row(bank, ae, reader, feasible_rows, infeasible_rows, forbid=(),
                subset=None):
    """Refusal with accept-unseen-valid, the only way a refusal number may be quoted.

    ``refusal`` on the impossible requests is worthless without ``false_refusal`` on
    the possible ones: an object that refuses everything scores 1.0 on the first.
    """
    out = {}
    for tag, rows in (("impossible", infeasible_rows), ("possible", feasible_rows)):
        if subset is not None:
            rows = [r for k, r in enumerate(rows) if subset[tag][k]]
        if not rows:
            out[tag] = {"n": 0}
            continue
        Zh = reader.read([r for (r, _m, _i, _j) in rows])
        plans, scenes = decode_plans(ae, Zh)
        recs = evaluate.answers(bank, rows, plans, scenes, use_certificate=True,
                                forbid=forbid)
        agg = evaluate.aggregate(recs)
        out[tag] = agg
    if out["impossible"].get("n") and out["possible"].get("n"):
        out["summary"] = {
            "refusal_on_impossible": out["impossible"]["refusal"],
            "false_refusal_on_possible": out["possible"]["refusal"],
            "validity_on_possible_after_certificate": out["possible"]["validity"],
        }
    return out


class SupervisedRefuser:
    """THE ONLY comparator for the DISPLAY rows -- and it HAS seen impossible examples.

    A transformer over the same four words, at the same matched parameter count, with
    one extra output: refuse or answer.  It is trained on a labelled mixture of
    possible and impossible requests drawn from the TRAINING mazes, which is precisely
    the supervision our object is not given.
    """

    def __init__(self, key, seq_in, target_params, seed=0):
        from w9_maze.baselines import LearnedArm
        self.arm = LearnedArm("transformer", key, seq_in, 1, target_params,
                              seq_in=seq_in)
        self.seed = seed

    def fit(self, Tok, y, steps=3000, lr=1e-3):
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import optax

        model = self.arm.model
        Y = np.asarray(y, np.float32)[:, None]

        def loss_fn(m, xb, yb):
            p = jax.vmap(m)(xb)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(p, yb))

        opt = optax.adam(optax.warmup_cosine_decay_schedule(
            1e-4, lr, int(0.05 * steps), steps, 1e-5))
        state = opt.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def step(m, st, xb, yb):
            lo, gr = eqx.filter_value_and_grad(loss_fn)(m, xb, yb)
            up, st = opt.update(gr, st, eqx.filter(m, eqx.is_array))
            return eqx.apply_updates(m, up), st, lo

        rng = np.random.default_rng(self.seed)
        for t in range(steps):
            b = rng.integers(0, Tok.shape[0], size=min(64, Tok.shape[0]))
            model, state, lo = step(model, state, jnp.asarray(Tok[b]),
                                    jnp.asarray(Y[b]))
        self.arm.model = model
        self.train_loss = float(lo)
        return self

    def refuses(self, Tok):
        import jax
        import jax.numpy as jnp
        p = np.asarray(jax.vmap(self.arm.model)(jnp.asarray(Tok)))[:, 0]
        return p > 0.0                      # logit > 0 == P(refuse) > 0.5
