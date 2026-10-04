"""hub/store-content — scoreboard rows 5, 6 and 7.

⛔ NOT this hub's target rows.  They are filled because w3-hub-common §2 says every
hub fills all seven so the four hubs are comparable, and they are **reported, never
claimed**.  ⛔ w3-hub-common §7 is binding here: no row-5 or row-6 number from this
hub is evidence for the store, and the laundering control (`grid_syntax_rule`) is
printed beside every refusal number.

THE ACCEPTANCE SCORE, declared
------------------------------
An arm must answer "do I hold this?" from the request alone.  For the addressed read
the natural quantity already exists: the address book's **total log-evidence** of the
request, summed over wells,

    s(request) = Σ_k  logsumexp_i  atom_posterior(book, request)[k, i]

⭐ It is request-only — ``atom_posterior`` takes no feature tensor — and it is a
function of what the store holds, so deleting atoms moves it, which is what row 7
needs.  ⛔ It is NOT derived from the spacing of the stored items, so it does not
repeat the admission gate's death (w3-hub-common §4.8).

THE THRESHOLD
-------------
One scalar, chosen on the **fit** rows as the ``q``-quantile of their own scores, and
⛔ used unchanged on rows 1, 5 and 6 — ``score_refusal_row`` checks that identity and
scores the row 0 if it does not hold.
"""
from __future__ import annotations

import numpy as np

from chlu.core.gen_landscape import atom_posterior

#: ⭐ PINNED IN ADVANCE.  The fit rows' own 5th percentile: the store claims to hold
#: what looks like the bottom 95 % of what was written into it.
THRESHOLD_QUANTILE = 0.05


def evidence(book, requests: np.ndarray) -> np.ndarray:
    """(n, F) requests -> (n,) total log-evidence.  ⛔ Request-only."""
    S = np.asarray(atom_posterior(book, np.asarray(requests, np.int64)), np.float64)
    m = S.max(axis=-1, keepdims=True)
    lse = (m[..., 0] + np.log(np.exp(S - m).sum(axis=-1)))      # (n, K)
    return lse.sum(axis=1)


def fit_threshold(book, fit_requests: np.ndarray, q: float = THRESHOLD_QUANTILE) -> dict:
    s = evidence(book, fit_requests)
    return {"threshold": float(np.quantile(s, float(q))), "quantile": float(q),
            "n_fit": int(s.size), "fit_score_mean": float(s.mean()),
            "fit_score_std": float(s.std()),
            "rule": "sum_k logsumexp_i atom_posterior(book, request)[k, i]",
            "note": "request-only; NOT derived from the spacing of the stored items"}


def accept_mask(book, requests: np.ndarray, threshold: float) -> np.ndarray:
    return evidence(book, requests) >= float(threshold)


# ---------------------------------------------------------------------------
# row 7 — deletion, with NO retraining
# ---------------------------------------------------------------------------

def delete_value(arm, grid, pool_rows: np.ndarray, factor: str, value_index: int):
    """Delete every atom carrying ``factor = value_index``, **in place**.

    ⛔ **No retraining of anything.**  The atom's coordinates are ZEROED and its
    address-book columns are driven to ``-inf`` in log space, so the atom can never
    be addressed and contributes nothing to any read.  The partition, the
    bandwidths, the grouping and the read verb are untouched, which is what
    "deletion is an operation on atoms" has to mean.

    ⭐ **Zeroed in place, NOT dropped — and the harness is why.**  My first version
    removed the atom ROWS, which shrank the deployed object, and
    ``gen_scoreboard.assert_byte_ledger`` refused the table: *"deployed_bytes is not
    constant across rows - the same deployed object must answer every row, or the
    table is seven experiments wearing one arm's name."*  That is the right rule and
    zeroing is the pin's own convention (``gen_store.delete_store(mode="zero")``).
    The deployed ledger therefore stays at exactly the budget on all seven rows, and
    the row measures whether the OUTPUT changed rather than whether the array did.
    """
    import copy
    import jax.numpy as jnp
    from chlu.core.gen_landscape import AtomAddressBook

    f = grid.index_of(factor)
    # ``arm.atom_rows_global`` already holds the dataset rows the atoms were taken
    # from, so the deletion predicate is read straight off the grid.
    global_rows = np.asarray(arm.atom_rows_global, np.int64)
    fi = grid.factor_indices_of(global_rows)
    keep = np.flatnonzero(fi[:, f] != int(value_index))
    dropped = int(global_rows.size - keep.size)

    gone = np.setdiff1d(np.arange(global_rows.size), keep)
    NEG = -1e9        # finite, so log-sums stay well defined; 1e9 nats is never won

    store = copy.copy(arm.store)
    zeroed = []
    for a in arm.store.atoms:
        A = np.asarray(a).copy()
        A[gone] = 0.0
        zeroed.append(jnp.asarray(A))
    object.__setattr__(store, "atoms", zeroed)

    lp = np.asarray(arm.book.log_prior).copy()
    lp[:, gone] = NEG
    lc = []
    for c in arm.book.log_cond:
        x = np.asarray(c).copy()
        x[:, :, gone] = NEG
        lc.append(x)
    book = AtomAddressBook(lp, lc)

    new = copy.copy(arm)
    new.store, new.book = store, book
    d = int(sum(len(w) for w in store.well_channels))
    b = int(store.atoms[0].shape[0] * d * 4)
    new.ledger = {**arm.ledger, "deployed_bytes": b,
                  "what": f"post-deletion (zeroed in place): {store.atoms[0].shape[0]} "
                          f"atoms x {d} float32, {dropped} of them erased",
                  "deleted_atoms": int(dropped)}
    return new, {"factor": factor, "value_index": int(value_index),
                 "n_atoms_total": int(global_rows.size),
                 "n_atoms_erased": dropped,
                 "n_atoms_surviving": int(keep.size),
                 "mode": "zeroed in place; address-book columns driven to -1e9",
                 "deployed_bytes_unchanged": True, "retrained": False}




