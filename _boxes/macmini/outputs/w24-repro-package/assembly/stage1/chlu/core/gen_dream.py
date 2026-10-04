"""**Conditional dreaming** on the factored store's landscape: clamp the wells a
request addresses, relax the rest, run Langevin, decode through the ONE shared
frozen decoder — and emit an audit trail that lets a referee convict the sampler.

⭐ DIAL DECLARATION — *isolation*.  The question is whether clamping the wells a
request addresses **actually steers what comes out**.
- **Laundering control:** ⭐ the **zero-step rollout** — decode the clamp directly,
  no Langevin, same clamp, same seed.  It runs on every reported number.
- **Falsifies:** the zero-step control matches the full rollout ⇒ the sampler is
  inert and we are decoding the clamp, not sampling the landscape.  Also
  falsifying: a flat energy trace, or a chain that never leaves its init.
- **Does NOT falsify:** blurry or low-fidelity samples in absolute terms.  The
  frozen decoder's measured ceiling bounds that and it is read as a ratio (R3.2).

⛔ **THE SEPARABILITY TRAP, AND WHY THIS MODULE HAS A COUPLING TERM.**
A store energy built only from per-well mixtures,

    ``E(x) = sum_k E_k(x_k)``,

is **separable**, and under a separable energy the conditional law of the relaxed
wells given the clamped wells **does not depend on the clamp at all**.  No
sampler, however good, can steer an unaddressed well from a clamped one: Langevin
would merely re-find whatever basin its initialisation sits in and the zero-step
control would match the full rollout **by construction**.  The landscape here
therefore carries two terms:

1. the **within-well** mixture free energy over the store's own modes, and
2. ⭐ an **inter-well compatibility field** ``J[k, l][m, m']`` — the pointwise
   mutual information of the empirical code co-occurrence on the **train** split,
   i.e. a Potts/MRF over the ``K`` code variables, **learned from data,
   label-free**, and never a function of the features at read time.

⚠ This is a **declared addition to the store**, not a silent one.  It costs
``K(K-1)/2 * M^2 * 4`` bytes on top of ``store_bytes`` (163,840 B at the shipped
``K=5, M=64``), which a byte-matched baseline must be told about.  ⛔ It does
**not** widen the read channel: the decode still goes
``x -> address_codes -> read_from_codes -> decoder``, so the channel is the same
``K*log2(M) = 30`` bits R20 declared a design constant, and
:func:`chlu.core.gen_store.read_from_codes` still never sees the features.
``coupling_strength = 0`` is run as a full arm so the separability argument above
is **measured**, not asserted.

⭐ **THE SAMPLER, DECLARED (the Head's sampler ruling).**
- Family: **overdamped Langevin** on ``x in R^D`` with the clamped coordinates
  frozen (the drift and the noise are masked to the free coordinates, so the
  clamp is exact and clamp *drift* is 0 by construction; the informative audit
  quantity is the **clamp pull**, the displacement the landscape would have
  applied to the clamp, which is logged).
- ⭐ **A Metropolis-Hastings correction IS applied** in the primary sampler
  (``"mala"``).  Du et al. (ICML 2023, arXiv:2302.11552) argue that for composed
  energies *the sampler, not the model, is the failure point*, and their critique
  bites hardest on **uncorrected** samplers; the cheapest way to be immune to it
  is to not be uncorrected.
- ``"ula"`` is the **same proposal with the accept/reject step removed** — one
  bit of difference — so the cost of skipping Metropolis is *measured*, not
  argued.  ``"uld"`` is the repo's own underdamped
  :func:`chlu.core.integrators.langevin_step` at ``noise_mode="fdt"`` (§7.22: T is
  in energy units only under ``fdt``; ⛔ ``legacy`` is never used for a reported
  number).
- **Acceptance is logged every step in every sampler**, including ``ula``, where
  the acceptance probability is computed **and then ignored** — so the size of the
  correction the unadjusted arm skips is on disk and auditable by someone who does
  not trust us.
- ``tau`` (the soft-assignment sharpness) and ``T`` (the sampling temperature) are
  both annealed geometrically.  ⚠ Declared, because at the sharp ``tau`` the store
  was trained at the basins are deep and separated, and an unannealed chain cannot
  leave them — which is exactly the mixing failure Du et al. are about.

⛔ **Standing convention (inherited from** :mod:`chlu.core.gen_store` **):** wells
are never named semantically.  Integer indices only.

Runnable through :mod:`chlu.experiments.exp_gen_dream`.
"""
from __future__ import annotations
import json
from typing import Optional, Sequence
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from .gen_bridge import param_hash
REQUEST_PROTOCOLS = ("full", "partial")
WINNING_DECODER_RECIPE = "v1_capacity_schedule"
def load_winner_hashes(path) -> dict:
    """``gen-decoder-lift``'s per-seed winner hashes (``winner/winner_hashes.json``)."""
    with open(path) as fh:
        return json.load(fh)


def assert_dream_decoder(
    decoder, expected_hash: str, recipe_name: Optional[str] = None,
    expected_recipe: str = WINNING_DECODER_RECIPE,
) -> dict:
    """⛔⭐ Refuse to score anything through the wrong decoder.

    Two independent checks, because either alone is weak:

    1. **the parameter hash** — the arm decodes through the *same* frozen renderer
       every other arm uses (:func:`chlu.core.gen_bridge.assert_decoder_hash` is
       the same check; this wrapper adds the second one and the recipe name to the
       manifest); and
    2. ⭐ **the recipe name** — ``chlu/config.py``'s ``dec_recipe`` defaults to
       ``"baseline"``, the decoder that **FAILED** the gate.  A run decoded through
       it produces perfectly normal-looking logs and is **void**.  If the recipe is
       known and is not :data:`WINNING_DECODER_RECIPE`, this raises.
    """
    got = param_hash(decoder)
    if got != str(expected_hash):
        raise AssertionError(
            "decoder parameter hash mismatch — this is NOT the shared frozen "
            f"decoder (expected {expected_hash}, got {got}). Every arm renders "
            "through the same decoder; substituting one voids every cross-arm "
            "comparison in the phase."
        )
    if recipe_name is not None and str(recipe_name) != str(expected_recipe):
        raise AssertionError(
            f"decoder recipe is {recipe_name!r}, not {expected_recipe!r} — the "
            "'baseline' recipe is the SUPERSEDED decoder that failed the gate "
            "(0.7191 balanced accuracy, 0/3 seeds). A run through it looks "
            "entirely normal in the logs and is void."
        )
    return {
        "decoder_param_hash": got,
        "decoder_recipe": None if recipe_name is None else str(recipe_name),
        "expected_recipe": str(expected_recipe),
        "checked": True,
    }


class AddressBook(eqx.Module):
    """Maps a **request** (a factor assignment) to a per-well distribution over modes.

    ⚠ **Declared supervision.**  The store, the coupling and the read are
    label-free (R1).  Grounding the *words* of a request in the store is not: a
    request says "shape = 2" and something must connect that symbol to the store.
    Every arm pays this — R8 gives the kNN arm composed per-factor *prototype
    features*, which are built from the same labels — so the asymmetry is declared
    and identical, and it enters at the interface, never in the generative path.

    ``log_prior``  ``(K, M)`` — ``log P(code_k = m)`` on the train split.
    ``log_cond``   list over factors, ``(V_f, K, M)`` — ``log P(code_k = m | f = v)``.

    Composition is naive Bayes / a product of experts in log space:

        ``s_k(m) = log P(m) + sum_{f in request} [ log P(m | f = v_f) - log P(m) ]``
    """

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
    codes: np.ndarray, labels: np.ndarray, cardinalities: Sequence[int],
    smoothing: float = 1.0,
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
    """``(N, F) requests -> (N, K, M)`` log-scores.  ``-1`` means "not requested".

    ⛔ A request is a *factor assignment*, not an image: nothing here touches the
    feature cache, which is what makes the same call work for A, B and C.
    """
    R = np.asarray(requests, np.int64)
    lp = np.asarray(book.log_prior)  # (K, M)
    out = np.broadcast_to(lp, (R.shape[0],) + lp.shape).copy()
    for f, lc in enumerate(book.log_cond):
        lc = np.asarray(lc)  # (V, K, M)
        asked = R[:, f] >= 0
        if not asked.any():
            continue
        v = R[asked, f]
        out[asked] += lc[v] - lp[None, :, :]
    return out.astype(np.float32)


def requests_from_labels(
    labels: np.ndarray, protocol: str, n_drop: int, seed: int
) -> np.ndarray:
    """Turn factor labels ``(N, F)`` into requests ``(N, F)`` with ``-1`` = not asked.

    ``full``    every factor is requested (cell-native: A/B/C records are full
                assignments).
    ``partial`` ``n_drop`` factors per request are dropped, chosen by a seeded RNG
                and fixed per request index.  ⭐ This is what *conditional*
                generation means, and it guarantees the relaxed set is non-empty
                regardless of how sharp the address book turns out to be.
    """
    Y = np.asarray(labels, np.int64).copy()
    if protocol == "full":
        return Y
    if protocol != "partial":
        raise ValueError(f"protocol must be one of {REQUEST_PROTOCOLS}, got {protocol!r}")
    rng = np.random.default_rng(int(seed) * 6_364_136 + 20_260_911)
    F = Y.shape[1]
    k = int(min(max(0, n_drop), F - 1))
    for i in range(Y.shape[0]):
        drop = rng.choice(F, size=k, replace=False)
        Y[i, drop] = -1
    return Y


