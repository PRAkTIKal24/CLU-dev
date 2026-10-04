"""⭐⭐ hub/composition — THE VALID-AND-NOVEL TUPLE COUNT.

⛔ One function, used for every arm and every rung, so no arm can be counted by a
different rule than another.  ⛔ Nothing here re-defines "valid" or "novel" — the two
definitions are fixed in `LOG.md` (2026-09-16T19:20Z), declared BEFORE any novelty
number was measured, and restated in the docstring below.

    VALID(i) := the external classifier reads render i as the tuple that was REQUESTED,
                on ALL SIX factors.  ⭐ Identical to `gen_probe.probe_accuracy`'s own
                per-item `exact_match` — the same arithmetic, not a re-spelling.
    NOVEL(t) := t is not the factor tuple of ANY stored atom.
    HEADLINE := |{ distinct t : produced, VALID and NOVEL }|.

⭐ The three weaker readings are computed too and returned BESIDE the headline, each
named, so a reader can see exactly what the strict rule costs and ⛔ nobody has to take
my word that the reachable one was not chosen:
    distinct                 — every distinct produced tuple (the shipped column)
    distinct_novel           — distinct, novel, ⛔ correctness not required
    distinct_valid           — distinct, correct, ⛔ novelty not required
    distinct_valid_novel     — ⭐ THE HEADLINE
⚠ `n_values_realised` per factor is emitted with every per-factor number, because
Montero's corner pins `shape` to 1 of 4 values and a per-factor accuracy without it has
produced a wrong headline in this programme once already.
"""
from __future__ import annotations
import numpy as np


def _as_tuples(a):
    return {tuple(int(x) for x in r) for r in np.asarray(a, np.int64).reshape((-1, np.asarray(a).shape[-1]))}


def valid_novel_report(pred_tuples, target_tuples, atom_tuples, factor_names) -> dict:
    """``(N,F) classifier reads · (N,F) requested · (M,F) stored atoms -> the counts``."""
    P = np.asarray(pred_tuples, np.int64)
    T = np.asarray(target_tuples, np.int64)
    if P.shape != T.shape:
        raise AssertionError(f"⛔ predictions {P.shape} and targets {T.shape} differ — "
                             "these are not the same render set")
    A = _as_tuples(atom_tuples) if atom_tuples is not None else set()
    n, F = P.shape
    valid_i = (P == T).all(axis=1)                       # ⭐ per-item exact match
    novel_i = np.array([tuple(r) not in A for r in P.tolist()], bool)

    def dcount(mask):
        return int(np.unique(P[mask], axis=0).shape[0]) if mask.any() else 0

    all_m = np.ones(n, bool)
    # ⚠ the per-factor column is meaningless without how many values the set realises
    per_factor = {}
    for j, nm in enumerate(factor_names):
        per_factor[str(nm)] = {
            "accuracy": float((P[:, j] == T[:, j]).mean()),
            "n_values_realised_in_targets": int(np.unique(T[:, j]).size),
            "n_values_realised_in_predictions": int(np.unique(P[:, j]).size),
        }
    return {
        "n": int(n),
        "n_factors": int(F),
        # ⭐ THE HEADLINE
        "distinct_valid_novel": dcount(valid_i & novel_i),
        # the ladder down to the shipped column, so the cost of each clause is visible
        "distinct_valid": dcount(valid_i),
        "distinct_novel": dcount(novel_i),
        "distinct": dcount(all_m),
        # item-level counts (⛔ not the headline: distinctness is the headline's unit)
        "n_valid_items": int(valid_i.sum()),
        "n_novel_items": int(novel_i.sum()),
        "n_valid_and_novel_items": int((valid_i & novel_i).sum()),
        "exact_match_frac": float(valid_i.mean()),
        "n_stored_atom_tuples_distinct": int(len(A)),
        "n_targets_distinct": int(np.unique(T, axis=0).shape[0]),
        "n_targets_that_are_novel": int(sum(1 for r in np.unique(T, axis=0).tolist()
                                            if tuple(r) not in A)),
        "per_factor": per_factor,
        "definitions": {
            "VALID": "external classifier reads ALL SIX factors as the REQUESTED tuple "
                     "(== gen_probe.probe_accuracy per-item exact_match)",
            "NOVEL": "the read tuple is not the factor tuple of any stored atom",
            "HEADLINE": "distinct_valid_novel",
            "declared": "LOG.md 2026-09-16T19:20Z, before any novelty number existed",
        },
    }
