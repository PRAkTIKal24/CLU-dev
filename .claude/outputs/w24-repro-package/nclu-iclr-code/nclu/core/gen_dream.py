"""gen dream."""

from __future__ import annotations
import json
from typing import Optional, Sequence
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from nclu.core.gen_bridge import param_hash

REQUEST_PROTOCOLS = ("full", "partial")
WINNING_DECODER_RECIPE = "v1_capacity_schedule"


def load_winner_hashes(path) -> dict:
    """``gen-decoder-lift``'s per-seed winner hashes (``winner/winner_hashes.json``)."""
    with open(path) as fh:
        return json.load(fh)


def assert_dream_decoder(
    decoder,
    expected_hash: str,
    recipe_name: Optional[str] = None,
    expected_recipe: str = WINNING_DECODER_RECIPE,
) -> dict:
    got = param_hash(decoder)
    if got != str(expected_hash):
        raise AssertionError(
            f"decoder parameter hash mismatch — this is NOT the shared frozen decoder (expected {expected_hash}, got {got}). Every arm renders through the same decoder; substituting one voids every cross-arm comparison in the phase."
        )
    if recipe_name is not None and str(recipe_name) != str(expected_recipe):
        raise AssertionError(
            f"decoder recipe is {recipe_name!r}, not {expected_recipe!r} — the 'baseline' recipe is the SUPERSEDED decoder that failed the gate (0.7191 balanced accuracy, 0/3 seeds). A run through it looks entirely normal in the logs and is void."
        )
    return {
        "decoder_param_hash": got,
        "decoder_recipe": None if recipe_name is None else str(recipe_name),
        "expected_recipe": str(expected_recipe),
        "checked": True,
    }


class AddressBook(eqx.Module):
    log_prior: jnp.ndarray
    log_cond: list

    def __init__(self, log_prior, log_cond):
        self.log_prior = jnp.asarray(log_prior, jnp.float32)
        self.log_cond = [jnp.asarray(x, jnp.float32) for x in log_cond]

    @property
    def n_wells(self) -> int:
        return int(self.log_prior.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.log_prior.shape[1])

    @property
    def n_factors(self) -> int:
        return len(self.log_cond)


def fit_address_book(
    codes: np.ndarray, labels: np.ndarray, cardinalities: Sequence[int], smoothing: float = 1.0
) -> AddressBook:
    """Fit :class:`AddressBook` from train codes + train factor labels."""
    C = np.asarray(codes, np.int64)
    Y = np.asarray(labels, np.int64)
    n, K = C.shape
    M = int(C.max()) + 1 if C.size else 1
    a = float(smoothing)
    prior = np.zeros((K, M), np.float64)
    for k in range(K):
        prior[k] = (np.bincount(C[:, k], minlength=M) + a) / (n + a * M)
    conds = []
    for f, card in enumerate(cardinalities):
        V = int(card)
        cond = np.zeros((V, K, M), np.float64)
        for v in range(V):
            sel = Y[:, f] == v
            nv = int(sel.sum())
            for k in range(K):
                cnt = np.bincount(C[sel, k], minlength=M).astype(np.float64)
                cond[v, k] = (cnt + a) / (nv + a * M)
        conds.append(np.log(cond))
    return AddressBook(np.log(prior), conds)


def address_requests(book: AddressBook, requests: np.ndarray) -> np.ndarray:
    R = np.asarray(requests, np.int64)
    lp = np.asarray(book.log_prior)
    out = np.broadcast_to(lp, (R.shape[0],) + lp.shape).copy()
    for f, lc in enumerate(book.log_cond):
        lc = np.asarray(lc)
        asked = R[:, f] >= 0
        if not asked.any():
            continue
        v = R[asked, f]
        out[asked] += lc[v] - lp[None, :, :]
    return out.astype(np.float32)


def requests_from_labels(labels: np.ndarray, protocol: str, n_drop: int, seed: int) -> np.ndarray:
    Y = np.asarray(labels, np.int64).copy()
    if protocol == "full":
        return Y
    if protocol != "partial":
        raise ValueError(f"protocol must be one of {REQUEST_PROTOCOLS}, got {protocol!r}")
    rng = np.random.default_rng(int(seed) * 6364136 + 20260911)
    F = Y.shape[1]
    k = int(min(max(0, n_drop), F - 1))
    for i in range(Y.shape[0]):
        drop = rng.choice(F, size=k, replace=False)
        Y[i, drop] = -1
    return Y
