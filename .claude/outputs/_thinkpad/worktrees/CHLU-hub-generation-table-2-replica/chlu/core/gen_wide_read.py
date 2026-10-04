"""⭐ The wide (span) read, for a possibly NON-UNIFORM per-well atom allocation.

⛔ **This module does not replace the shipped read verb.**  When the allocation is
uniform — which every SCORED configuration in ``hub/generation-table-2`` is — this is a
**one-line pass-through** of :func:`chlu.core.gen_compose.clu_wide_vector`, the pinned
harness's own function, so the scored path is the shipped path and
``assert_request_only``'s signature guarantee applies verbatim.

⚠ The pinned ``clu_wide_vector`` requires a rectangular ``(N, K, M)`` posterior and so
refuses a store whose wells hold different numbers of atoms, even though ``GenStore``
itself supports one (``n_modes_per_well``) and ``store_bytes`` already prices it
correctly.  ⛔ ``gen_compose.py`` is a **pinned file** (charter §A.4) and is not edited.
:func:`wide_vector` therefore dispatches: uniform → the shipped function, bit-identical;
ragged → the same arithmetic written out, used for **DEV lever isolation only** and
declared as such wherever a ragged number is quoted.

The equivalence is not asserted, it is TESTED: ``tests/test_gen_wide_read.py`` checks
that on a uniform store the two paths are bit-identical on random inputs.

⛔ No feature tensor enters here, by signature (charter §B.3).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["is_uniform", "stack_coefficients", "wide_vector"]


def is_uniform(coeffs: Sequence[np.ndarray]) -> bool:
    """Do all wells carry the same number of atoms?"""
    return len({int(np.asarray(a).shape[1]) for a in coeffs}) == 1


def stack_coefficients(coeffs: Sequence[np.ndarray]) -> np.ndarray:
    """``[ (N, M) ] * K  ->  (N, K, M)`` — the layout ``clu_wide_vector`` expects.

    ⛔ Refuses a ragged allocation rather than padding it: a zero-padded atom is still
    an atom in ``store_bytes``' eyes and padding would silently misprice the store.
    """
    if not is_uniform(coeffs):
        raise ValueError(
            "stack_coefficients needs a uniform allocation; got "
            f"{[int(np.asarray(a).shape[1]) for a in coeffs]}"
        )
    return np.stack([np.asarray(a, np.float32) for a in coeffs], axis=1)


def wide_vector(store, coeffs: Sequence[np.ndarray]) -> np.ndarray:
    """``[ (N, M_k) ] -> (N,) + feat_shape``.  ⭐ Uniform => the SHIPPED read verb."""
    from .gen_compose import clu_wide_vector

    if is_uniform(coeffs):
        return clu_wide_vector(store, stack_coefficients(coeffs))

    # ⚠ ragged — DEV only.  Same arithmetic, written out because the pinned function
    # takes a rectangular array and ⛔ pinned files are not edited.
    dt = np.asarray(store.modes[0]).dtype
    n = int(np.asarray(coeffs[0]).shape[0])
    out = np.zeros((n, int(store.dim)), dt)
    for w, m, a in zip(store.well_channels, store.modes, coeffs, strict=True):
        out[:, np.asarray(w, np.int64)] = np.asarray(a, dt) @ np.asarray(m, dt)
    return out.reshape((n,) + tuple(int(x) for x in store.feat_shape))
