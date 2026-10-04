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
from pathlib import Path
from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .gen_bridge import param_hash
from .gen_store import GenStore, address_codes, read_from_codes
from .integrators import langevin_step

#: Samplers this module can run.  ``mala`` is the primary and IS
#: Metropolis-corrected; ``ula`` is the identical proposal with the correction
#: removed; ``uld`` is the repo's underdamped integrator at ``noise_mode="fdt"``.
SAMPLERS = ("mala", "ula", "uld")
#: Arms this module owns.  ``store_removed`` is the ablation the task assigns to
#: this spoke; ``no_coupling`` is the separability control (``lambda = 0``).
ARMS = ("clu", "store_removed", "no_coupling")
#: Rollout variants scored on every arm.  ⭐ ``zero_step`` is the laundering
#: control and is never optional.
VARIANTS = ("full", "zero_step", "prior_mode")
#: How the relaxed wells are initialised.
INIT_MODES = ("centroid", "prior_sample", "prior_mode")
#: Request protocols (see the module's PREREG §4).  ``full`` is cell-native;
#: ``partial`` drops factors so the relaxed set cannot be empty.
REQUEST_PROTOCOLS = ("full", "partial")
#: ⭐ The decoder recipe that WON ``gen-decoder-lift`` (0.9414 balanced accuracy,
#: 3/3 seeds).  ⛔ ``chlu/config.py``'s ``dec_recipe`` still defaults to
#: ``"baseline"`` — the recipe that FAILED the gate (0.7191, 0/3) — deliberately,
#: so Montero's history reproduces.  An arm decoded through the baseline is void
#: and looks entirely normal in the logs, which is why
#: :func:`assert_dream_decoder` exists.
WINNING_DECODER_RECIPE = "v1_capacity_schedule"

__all__ = [
    "ARMS",
    "INIT_MODES",
    "REQUEST_PROTOCOLS",
    "SAMPLERS",
    "VARIANTS",
    "WINNING_DECODER_RECIPE",
    "AddressBook",
    "DreamLandscape",
    "address_requests",
    "assert_dream_decoder",
    "clamp_from_posterior",
    "RolloutCfg",
    "coupling_bytes",
    "dream_rollout",
    "energy",
    "fit_address_book",
    "fit_code_coupling",
    "init_state",
    "load_winner_hashes",
    "mode_distance_scale",
    "occupancy_saturation",
    "quantise",
    "resolve_rollout_cfg",
    "requests_from_labels",
    "rollout_read",
    "save_trace",
    "well_energies",
]


# ---------------------------------------------------------------------------
# 1. ⭐ The decoder guard — ⛔ check this FIRST, not last
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 2. The inter-well compatibility field (learned from the TRAIN codes)
# ---------------------------------------------------------------------------


def fit_code_coupling(codes: np.ndarray, n_modes: int, smoothing: float = 1.0) -> np.ndarray:
    """``J[k, l][m, m'] = PMI(code_k = m ; code_l = m')`` in nats, ``(K, K, M, M)``.

    Fitted on the **train** codes only.  ⛔ No labels, no held-out row, no decoder.
    Symmetric in ``(k, m) <-> (l, m')``, with the diagonal blocks ``J[k, k]`` set to
    zero (a well never couples to itself — that is what ``E_k`` is for).

    ⚠ PMI, not raw log-joint, on purpose: the marginal term is already carried by
    the within-well energy, so coupling on the *joint* would double-count the
    marginals and quietly re-weight the wells by how peaked their own usage is.
    """
    C = np.asarray(codes, np.int64)
    n, K = C.shape
    M = int(n_modes)
    a = float(smoothing)
    J = np.zeros((K, K, M, M), np.float64)
    marg = np.zeros((K, M), np.float64)
    for k in range(K):
        marg[k] = (np.bincount(C[:, k], minlength=M).astype(np.float64) + a) / (n + a * M)
    for k in range(K):
        for lo in range(k + 1, K):
            joint = np.zeros((M, M), np.float64)
            np.add.at(joint, (C[:, k], C[:, lo]), 1.0)
            joint = (joint + a) / (n + a * M * M)
            pmi = np.log(joint) - np.log(marg[k])[:, None] - np.log(marg[lo])[None, :]
            J[k, lo] = pmi
            J[lo, k] = pmi.T
    return J.astype(np.float32)


def coupling_bytes(k_wells: int, n_modes: int) -> dict:
    """⚠ Byte accounting for the coupling field — it is **extra** to ``store_bytes``.

    ``gen-baselines`` byte-matches to ``store_bytes`` (524,288 B at the shipped
    cell).  The coupling is a second structure and must be declared separately or
    the byte match is wrong.  ⛔ It does not widen the read channel.
    """
    k, m = int(k_wells), int(n_modes)
    n_pairs = k * (k - 1) // 2
    return {
        "coupling_bytes": int(n_pairs * m * m * 4),
        "n_pairs": int(n_pairs),
        "n_modes": m,
        "dtype": "float32",
        "bytes_formula": "K(K-1)/2 * M^2 * 4",
        "bits_per_read_unchanged": float(k * np.log2(m)),
        "note": (
            "⚠ EXTRA to gen_store.store_bytes; a byte-matched baseline must be "
            "told about it. ⛔ The read channel is unchanged at K*log2(M) bits."
        ),
    }


# ---------------------------------------------------------------------------
# 3. ⭐ The request addressing head (R8: the CLU arm's own; ⛔ never a baseline's)
# ---------------------------------------------------------------------------


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


def clamp_from_posterior(log_scores: np.ndarray, threshold: float) -> dict:
    """``(N, K, M)`` log-scores -> the clamp: which wells, which mode, how sure.

    ⭐ A well is **clamped** iff the request concentrates its posterior on one mode
    above ``threshold``; otherwise the request does not address it and it is
    **relaxed**.  ⛔ The rule is a property of the request and the address book
    alone — no image, no held-out quantity and no arm outcome enters it.
    """
    S = np.asarray(log_scores, np.float64)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=-1, keepdims=True)
    code = np.argmax(P, axis=-1).astype(np.int64)          # (N, K)
    conf = np.max(P, axis=-1)                              # (N, K)
    clamped = conf >= float(threshold)
    return {
        "code": code,
        "confidence": conf,
        "clamped": clamped,
        "posterior": P.astype(np.float32),
        "threshold": float(threshold),
        "mean_n_clamped": float(clamped.sum(axis=1).mean()) if clamped.size else 0.0,
        "mean_n_relaxed": float((~clamped).sum(axis=1).mean()) if clamped.size else 0.0,
        "mean_confidence": float(conf.mean()) if conf.size else float("nan"),
    }


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


# ---------------------------------------------------------------------------
# 4. The landscape
# ---------------------------------------------------------------------------


class DreamLandscape(eqx.Module):
    """The store's landscape: per-well mixture wells + the inter-well PMI field.

    ``E(x) = sum_k [ -tau^2 logsumexp_m(-d2_km / (2 tau^2)) ]
             - lam * sum_{k<l} pi_k^T J[k,l] pi_l``

    with ``pi_k = softmax(-d2_k / (2 tau^2))`` the same soft occupancy the store was
    trained with.  The first term's minima are exactly the store's modes; the
    second is what makes the wells **conditionally dependent** — ⛔ without it the
    energy is separable and conditional dreaming is provably inert (module
    docstring).
    """

    store: GenStore
    coupling: jnp.ndarray
    coupling_strength: float = eqx.field(static=True)

    def __init__(self, store: GenStore, coupling, coupling_strength: float = 1.0):
        self.store = store
        K, M = store.n_wells, store.n_modes
        if coupling is None:
            coupling = np.zeros((K, K, M, M), np.float32)
        J = jnp.asarray(coupling, store.modes[0].dtype)
        if J.shape != (K, K, M, M):
            raise ValueError(
                f"coupling must be (K, K, M, M) = {(K, M)} -> {(K, K, M, M)}, got {J.shape}"
            )
        self.coupling = J
        self.coupling_strength = float(coupling_strength)

    @property
    def n_wells(self) -> int:
        return int(self.store.n_wells)

    @property
    def n_modes(self) -> int:
        return int(self.store.n_modes)


def _well_d2(store: GenStore, X, dt):
    """Per-well squared distances to every mode, ``[ (N, M) ]``.  Gram form.

    ⚠ Gram form, not the ``(N, M, d)`` broadcast: at the shipped cell the broadcast
    is hundreds of MB per well and the run dies on a laptop
    (:func:`chlu.core.gen_store.address_codes` carries the same note).
    """
    Z = jnp.asarray(X, dt).reshape((X.shape[0], -1))
    out = []
    for w, m in zip(store.well_channels, store.modes, strict=True):
        z = Z[:, jnp.asarray(w, jnp.int32)]
        out.append(
            jnp.sum(z * z, axis=-1)[:, None]
            - 2.0 * (z @ m.T)
            + jnp.sum(m * m, axis=-1)[None, :]
        )
    return out


def well_energies(land: DreamLandscape, X, tau):
    """``(E_within (N,), pi [ (N, M) ])`` — the separable part and the occupancies."""
    dt = land.store.modes[0].dtype
    tau = jnp.asarray(tau, dt)
    d2 = _well_d2(land.store, X, dt)
    e = jnp.zeros((X.shape[0],), dt)
    pis = []
    for d in d2:
        s = -d / (2.0 * tau**2)
        e = e - (tau**2) * jax.nn.logsumexp(s, axis=-1)
        pis.append(jax.nn.softmax(s, axis=-1))
    return e, pis


def energy(land: DreamLandscape, X, tau):
    """``(N,)`` total energy.  ⛔ Traceable: no Python branch on a traced value."""
    e, pis = well_energies(land, X, tau)
    lam = land.coupling_strength
    if lam != 0.0:
        K = land.n_wells
        for k in range(K):
            for lo in range(k + 1, K):
                e = e - lam * jnp.einsum(
                    "nm,mp,np->n", pis[k], land.coupling[k, lo], pis[lo]
                )
    return e


def _energy_sum(land, X, tau):
    return jnp.sum(energy(land, X, tau))


# ---------------------------------------------------------------------------
# 5. Clamp / initialisation
# ---------------------------------------------------------------------------


def _coord_masks(store: GenStore, well_mask: np.ndarray) -> np.ndarray:
    """``(N, K)`` well mask -> ``(N, D)`` coordinate mask."""
    wm = np.asarray(well_mask, bool)
    out = np.zeros((int(wm.shape[0]), store.dim), bool)
    for k, w in enumerate(store.well_channels):
        rows = np.flatnonzero(wm[:, k])
        if rows.size:
            out[np.ix_(rows, np.asarray(w, np.int64))] = True
    return out


def _codes_to_x(store: GenStore, codes: np.ndarray) -> np.ndarray:
    """``(N, K)`` codes -> ``(N, D)`` concatenated mode vectors (the store read, flat)."""
    return np.asarray(
        read_from_codes(store, np.asarray(codes, np.int64))
    ).reshape((np.asarray(codes).shape[0], -1))


def init_state(
    store: GenStore, clamp: dict, prior: np.ndarray, init_mode: str, seed: int
) -> dict:
    """Build ``x0``, the clamp value vector and the coordinate mask.

    Clamped wells sit exactly on their clamped mode.  Relaxed wells are
    initialised with **no information about the request**:

    ``centroid``     the well's mode centroid — the least-committed point, and the
                     one that gives the drift the most freedom (default).
    ``prior_sample`` a mode drawn from the empirical marginal ``P(code_k)``.
    ``prior_mode``   the marginal's argmax (deterministic).

    ⭐ Whatever the choice, the **zero-step** control decodes exactly this state,
    so the contrast is "the clamp plus an uninformed guess" vs "the clamp plus
    whatever the landscape infers".
    """
    if init_mode not in INIT_MODES:
        raise ValueError(f"init_mode must be one of {INIT_MODES}, got {init_mode!r}")
    code = np.asarray(clamp["code"], np.int64).copy()
    clamped = np.asarray(clamp["clamped"], bool)
    n, K = code.shape
    rng = np.random.default_rng(int(seed) * 2_246_822_519 % (2**32))
    P = np.asarray(prior, np.float64)
    P = P / P.sum(axis=1, keepdims=True)

    x_clamp = _codes_to_x(store, code)
    x0 = x_clamp.copy()
    init_code = code.copy()
    for k in range(K):
        free = np.flatnonzero(~clamped[:, k])
        if not free.size:
            continue
        cols = np.asarray(store.well_channels[k], np.int64)
        modes = np.asarray(store.modes[k])
        if init_mode == "centroid":
            x0[np.ix_(free, cols)] = modes.mean(axis=0)[None, :]
            init_code[free, k] = int(
                np.argmin(((modes - modes.mean(axis=0)) ** 2).sum(axis=1))
            )
        elif init_mode == "prior_sample":
            m = rng.choice(modes.shape[0], size=free.size, p=P[k])
            x0[np.ix_(free, cols)] = modes[m]
            init_code[free, k] = m
        else:  # prior_mode
            m = int(np.argmax(P[k]))
            x0[np.ix_(free, cols)] = modes[m][None, :]
            init_code[free, k] = m
    return {
        "x0": x0.astype(np.float32),
        "x_clamp": x_clamp.astype(np.float32),
        "coord_clamped": _coord_masks(store, clamped),
        "init_code": init_code,
        "init_mode": str(init_mode),
    }


def quantise(store: GenStore, X) -> np.ndarray:
    """``x -> codes``.  ⭐ The ONLY channel from the rollout to the decoder."""
    return address_codes(store, np.asarray(X, np.float32).reshape((np.asarray(X).shape[0], -1)))


def rollout_read(store: GenStore, X) -> tuple:
    """``x -> codes -> read_from_codes``.  ⛔ Preserves the 30-bit channel exactly.

    ⚠ **The read never sees ``x``.**  A dream that handed its continuous state to
    the decoder would carry ``D`` floats across the encoder->decoder boundary and
    destroy R20's declared design constant.  The rollout is a *search over code
    configurations*, and the codes are all that crosses.
    """
    codes = quantise(store, X)
    return codes, np.asarray(read_from_codes(store, codes), np.float32)


# ---------------------------------------------------------------------------
# 6. ⭐ The samplers, and the audit trail
# ---------------------------------------------------------------------------


class RolloutCfg:
    """The rollout knobs in **ABSOLUTE units**, resolved from
    :class:`~chlu.config.GenDreamConfig`'s fractions by :func:`resolve_rollout_cfg`.

    ⚠ Kept separate from the config dataclass on purpose: ``tau`` and ``T`` in the
    config are *fractions* of quantities measured from the store, and shipping the
    fractions into the sampler was exactly the defect that froze the first run.
    """

    __slots__ = (
        "sampler", "steps", "step_size", "tau_start", "tau_end", "temp_start",
        "temp_end", "uld_dt", "uld_gamma", "calibration",
    )

    def __init__(self, sampler, steps, step_size, tau_start, tau_end, temp_start,
                 temp_end, uld_dt, uld_gamma, calibration=None):
        self.sampler = str(sampler)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.uld_dt = float(uld_dt)
        self.uld_gamma = float(uld_gamma)
        self.calibration = dict(calibration or {})

    def replace(self, **kw) -> "RolloutCfg":
        d = {f: getattr(self, f) for f in self.__slots__}
        d.update(kw)
        return RolloutCfg(**d)

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__slots__}


def mode_distance_scale(store: GenStore) -> dict:
    """⭐ The store's own geometry: the median distance **between modes**, per well.

    ⛔ Computed from the train-fitted modes alone — no eval row, no probe reading,
    no accuracy.  It is the natural unit for ``tau``: ``exp(-d^2 / 2 tau^2)`` is
    saturated one-hot whenever ``tau`` is small against this scale, and a saturated
    soft occupancy has vanishing gradient in the coupling term **and** isolated
    quadratic basins with no communication.  Specifying ``tau`` in absolute units
    (the store's own *training* schedule) is therefore a sampler defect, and it is
    the one that froze this module's first run (``moved = 0.05``).
    """
    per_well = []
    for m in store.modes:
        M = np.asarray(m, np.float64)
        g = (M * M).sum(axis=1)
        d2 = np.maximum(g[:, None] + g[None, :] - 2.0 * (M @ M.T), 0.0)
        iu = np.triu_indices(M.shape[0], 1)
        per_well.append(float(np.median(np.sqrt(d2[iu]))) if iu[0].size else 0.0)
    return {
        "per_well_median_mode_distance": per_well,
        "scale": float(np.mean(per_well)) if per_well else 1.0,
        "rule": "mean over wells of the median pairwise distance between that "
                "well's modes (train-fitted modes only)",
    }


def occupancy_saturation(land: "DreamLandscape", X, tau) -> dict:
    """⭐ The audit column that convicts a frozen chain: mean top-1 occupancy mass.

    ``mean_max_pi ~ 1.0`` means the soft assignment is one-hot, the coupling term's
    gradient has vanished and the chain is sitting in an isolated basin.  ⛔ Logged,
    never inferred from behaviour.
    """
    _, pis = well_energies(land, jnp.asarray(np.asarray(X, np.float32)), tau)
    mx = [float(jnp.mean(jnp.max(p, axis=-1))) for p in pis]
    return {"mean_max_pi": float(np.mean(mx)), "per_well_max_pi": mx,
            "tau": float(tau)}


def resolve_rollout_cfg(cfg, store: GenStore, land: "DreamLandscape", x0,
                        step_size: Optional[float] = None) -> RolloutCfg:
    """Turn ``GenDreamConfig``'s fractions into a sampler config in absolute units.

    ``tau_calibration="mode_distance"`` -> ``tau = frac * mode_distance_scale``.
    ``temp_calibration="energy_scale"`` -> ``T = frac * median |E(x0)|`` at the
    reference sharpness, i.e. the temperature is expressed **in the landscape's own
    energy units**, which is the only way ``exp(-dE/T)`` means anything after the
    ``tau`` change.  ``"absolute"`` keeps the literal numbers (the pre-fix
    behaviour, retained so the first run stays reproducible).
    """
    tau_scale = 1.0
    tau_info = {"tau_calibration": str(cfg.tau_calibration)}
    if str(cfg.tau_calibration) == "mode_distance":
        md = mode_distance_scale(store)
        tau_scale = md["scale"]
        tau_info["mode_distance"] = md
    elif str(cfg.tau_calibration) != "absolute":
        raise ValueError(
            f"tau_calibration must be 'mode_distance' or 'absolute', got "
            f"{cfg.tau_calibration!r}"
        )
    tau_start = float(cfg.tau_start) * tau_scale
    tau_end = float(cfg.tau_end) * tau_scale

    temp_scale = 1.0
    tau_info["temp_calibration"] = str(cfg.temp_calibration)
    if str(cfg.temp_calibration) == "energy_scale":
        e0 = np.asarray(energy(land, jnp.asarray(np.asarray(x0, np.float32)),
                               jnp.float32(tau_end)))
        temp_scale = float(np.median(np.abs(e0))) if e0.size else 1.0
        temp_scale = temp_scale if temp_scale > 0 else 1.0
        tau_info["energy_scale"] = temp_scale
    elif str(cfg.temp_calibration) != "absolute":
        raise ValueError(
            f"temp_calibration must be 'energy_scale' or 'absolute', got "
            f"{cfg.temp_calibration!r}"
        )
    tau_info["tau_start_abs"] = tau_start
    tau_info["tau_end_abs"] = tau_end
    tau_info["temp_start_abs"] = float(cfg.temp_start) * temp_scale
    tau_info["temp_end_abs"] = float(cfg.temp_end) * temp_scale
    tau_info["saturation_at_tau_start"] = occupancy_saturation(land, x0, tau_start)
    tau_info["saturation_at_tau_end"] = occupancy_saturation(land, x0, tau_end)
    return RolloutCfg(
        sampler=cfg.sampler,
        steps=int(cfg.steps),
        step_size=float(step_size if step_size is not None else (cfg.step_size or cfg.step_size_init)),
        tau_start=tau_start,
        tau_end=tau_end,
        temp_start=tau_info["temp_start_abs"],
        temp_end=tau_info["temp_end_abs"],
        uld_dt=float(cfg.uld_dt),
        uld_gamma=float(cfg.uld_gamma),
        calibration=tau_info,
    )


def _geom(a: float, b: float, i: int, n: int) -> float:
    a, b = float(a), float(b)
    if n <= 1:
        return b
    return float(a * (b / a) ** (i / (n - 1)))


def _mala_step(land, x, free, key, eps, temp, tau):
    """One MALA/ULA proposal.  Returns ``(x_prop, log_alpha, accepted, clamp_pull, key)``.

    The drift and the noise are **masked to the free coordinates**, so the clamped
    coordinates are bit-identical in ``x`` and ``x'`` (clamp drift is 0 by
    construction) and the two Gaussian proposal densities differ only on the free
    block — which is what makes the MH ratio below exact for the *conditional*
    target ``p(x_free | x_clamped)``.
    """
    dt = x.dtype
    eps = jnp.asarray(eps, dt)
    temp = jnp.asarray(temp, dt)
    gfun = jax.grad(lambda z: _energy_sum(land, z, tau))
    g = gfun(x)
    # ⭐ the audit quantity: the displacement the landscape WOULD apply to the
    # clamped coordinates if they were free.  0 here means the coupling is not
    # acting on the clamp at all.
    n_clamped = jnp.maximum(jnp.sum(1.0 - free, axis=-1), 1.0)
    clamp_pull = jnp.sqrt(jnp.sum(((eps * g) * (1.0 - free)) ** 2, axis=-1) / n_clamped)
    mean_f = x - eps * g * free
    key, nk, uk = jax.random.split(key, 3)
    noise = jax.random.normal(nk, x.shape, dt) * jnp.sqrt(2.0 * eps * temp)
    xp = mean_f + noise * free
    gp = gfun(xp)
    mean_r = xp - eps * gp * free
    denom = 4.0 * eps * temp
    logq_fwd = -jnp.sum(((xp - mean_f) * free) ** 2, axis=-1) / denom
    logq_rev = -jnp.sum(((x - mean_r) * free) ** 2, axis=-1) / denom
    de = energy(land, xp, tau) - energy(land, x, tau)
    log_alpha = -de / temp + logq_rev - logq_fwd
    u = jax.random.uniform(uk, (x.shape[0],), dt)
    accepted = jnp.log(jnp.maximum(u, jnp.asarray(1e-30, dt))) < log_alpha
    return xp, log_alpha, accepted, clamp_pull, key


@eqx.filter_jit
def _mala_step_jit(land, x, free, key, eps, temp, tau, tau_ref, correct):
    """⚠ ``eps``/``temp``/``tau`` arrive as **jnp scalars** so they are traced, not
    static: annealing them as Python floats would retrace the whole step at every
    step of the rollout.  ``correct`` is a Python bool (2 values, static).

    ⚠ Two energies come back.  ``E`` is the energy of the *annealed* landscape the
    chain is actually on at this step; ``E_ref`` is the SAME state scored at the
    fixed reference sharpness ``tau_ref``.  ⛔ Only ``E_ref`` may be compared
    across steps — with ``tau`` annealing, ``E[0]`` and ``E[-1]`` are values of two
    different functions and the trace can rise while the state strictly improves
    (the same trap ``gen-store-grouping`` filed against annealed training losses)."""
    xp, log_alpha, accepted, clamp_pull, key = _mala_step(land, x, free, key, eps, temp, tau)
    take = accepted if correct else jnp.ones_like(accepted)
    x_new = jnp.where(take[:, None], xp, x)
    return (
        x_new, jnp.exp(jnp.minimum(log_alpha, 0.0)), accepted, clamp_pull, key,
        energy(land, x_new, tau), energy(land, x_new, tau_ref),
    )


@eqx.filter_jit
def _uld_step_jit(land, x, p, free, key, dt_step, gamma, temp, tau, tau_ref, m_eff):
    """One step of the REPO's underdamped integrator, vmapped over the batch.

    ``H(q, p) = E(q) + 0.5 |p|^2`` at unit mass, ``noise_mode="fdt"`` — so T is in
    energy units (§7.22: ⛔ ``legacy`` is never used for a reported number) and the
    stationary law is ``exp(-H/T)``.  The clamp is enforced by holding ``q`` and
    zeroing ``p`` on the clamped coordinates after the step.
    """

    def h_fn(q, mom):
        return energy(land, q[None, :], tau)[0] + 0.5 * jnp.sum(mom * mom)

    def one(q, mom, k):
        return langevin_step(
            h_fn, q, mom, dt_step, gamma, temp, k, noise_mode="fdt", m_eff=m_eff
        )

    keys = jax.random.split(key, x.shape[0] + 1)
    q, mom, _ = jax.vmap(one)(x, p, keys[1:])
    q = jnp.where(free > 0, q, x)
    mom = mom * free
    return q, mom, keys[0], energy(land, q, tau), energy(land, q, tau_ref)


def dream_rollout(
    land: DreamLandscape,
    x0: np.ndarray,
    coord_clamped: np.ndarray,
    cfg,
    key,
    steps: Optional[int] = None,
    sampler: Optional[str] = None,
) -> dict:
    """⭐ The clamped-well Langevin rollout, with its audit trail.

    ``steps = 0`` is the **zero-step control**: no gradient, no noise, no
    acceptance — the state is returned exactly as handed in, so the caller decodes
    the clamp itself.  ⛔ That control is this spoke's acceptance criterion and it
    is never optional.

    Returns ``{"x": final state, "trace": {...}}``; every array in ``trace`` is
    written to disk by :func:`save_trace`, never summarised only in prose.
    """
    sampler = str(sampler or cfg.sampler)
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
    n_steps = int(cfg.steps if steps is None else steps)
    dt = land.store.modes[0].dtype
    x = jnp.asarray(np.asarray(x0, np.float32), dt)
    free = jnp.asarray((~np.asarray(coord_clamped, bool)).astype(np.float32), dt)
    n = int(x.shape[0])

    tau0, tau1 = float(cfg.tau_start), float(cfg.tau_end)
    t0, t1 = float(cfg.temp_start), float(cfg.temp_end)
    eps = jnp.asarray(float(cfg.step_size), dt)
    # ⭐ the FIXED reference sharpness every cross-step energy comparison uses
    tau_ref = jnp.asarray(tau1, dt)
    e_hist, er_hist = [], []
    a_hist, acc_hist, pull_hist, tau_hist, temp_hist = [], [], [], [], []

    tau_now = jnp.asarray(_geom(tau0, tau1, 0, max(n_steps, 1)), dt)
    e_hist.append(np.asarray(energy(land, x, tau_now), np.float64))
    er_hist.append(np.asarray(energy(land, x, tau_ref), np.float64))
    code0 = quantise(land.store, np.asarray(x))

    p = jnp.zeros_like(x)
    m_eff = jnp.ones((int(x.shape[1]),), dt)
    uld_dt = jnp.asarray(float(cfg.uld_dt), dt)
    uld_gamma = jnp.asarray(float(cfg.uld_gamma), dt)
    for i in range(n_steps):
        tau_now = jnp.asarray(_geom(tau0, tau1, i, n_steps), dt)
        temp_now = jnp.asarray(_geom(t0, t1, i, n_steps), dt)
        key, sk = jax.random.split(key)
        if sampler == "uld":
            # ⭐ the acceptance the unadjusted underdamped chain is SKIPPING is
            # still computed, on the same state, so the trace can convict it.
            _xp, aprob, acc, pull, _k, _e, _er = _mala_step_jit(
                land, x, free, sk, eps, temp_now, tau_now, tau_ref, False
            )
            key, sk2 = jax.random.split(key)
            x, p, _kk, e_new, er_new = _uld_step_jit(
                land, x, p, free, sk2, uld_dt, uld_gamma, temp_now, tau_now, tau_ref, m_eff
            )
        else:
            x, aprob, acc, pull, _k, e_new, er_new = _mala_step_jit(
                land, x, free, sk, eps, temp_now, tau_now, tau_ref, sampler == "mala"
            )
        a_hist.append(np.asarray(aprob, np.float64))
        acc_hist.append(np.asarray(acc, bool))
        pull_hist.append(np.asarray(pull, np.float64))
        tau_hist.append(float(tau_now))
        temp_hist.append(float(temp_now))
        e_hist.append(np.asarray(e_new, np.float64))
        er_hist.append(np.asarray(er_new, np.float64))

    x_out = np.asarray(x, np.float32)
    code_t = quantise(land.store, x_out)
    sat_first = occupancy_saturation(land, x0, jnp.asarray(tau0, dt))
    sat_final = occupancy_saturation(land, x_out, jnp.asarray(tau1, dt))
    trace = {
        "energy": np.stack(e_hist) if e_hist else np.zeros((0, n)),   # (steps+1, N)
        "energy_ref": np.stack(er_hist) if er_hist else np.zeros((0, n)),
        "accept_prob": np.stack(a_hist) if a_hist else np.zeros((0, n)),
        "accepted": np.stack(acc_hist) if acc_hist else np.zeros((0, n), bool),
        "clamp_pull": np.stack(pull_hist) if pull_hist else np.zeros((0, n)),
        "tau": np.asarray(tau_hist, np.float64),
        "temperature": np.asarray(temp_hist, np.float64),
        "code_init": code0,
        "code_final": code_t,
    }
    trace["summary"] = rollout_summary(trace, sampler, n_steps, cfg)
    trace["summary"]["mean_max_pi_first"] = sat_first["mean_max_pi"]
    trace["summary"]["mean_max_pi_final"] = sat_final["mean_max_pi"]
    trace["summary"]["occupancy_is_saturated"] = bool(sat_first["mean_max_pi"] > 0.99)
    return {"x": x_out, "trace": trace}


def rollout_summary(trace: dict, sampler: str, n_steps: int, cfg) -> dict:
    """⭐ The numbers a referee reads first: is the chain moving, and is it accepting?"""
    E = np.asarray(trace["energy_ref"])   # ⛔ never trace["energy"]: tau is annealed
    code0 = np.asarray(trace["code_init"])
    codeT = np.asarray(trace["code_final"])
    changed = (code0 != codeT)
    e0 = E[0] if E.shape[0] else np.zeros(0)
    eT = E[-1] if E.shape[0] else np.zeros(0)
    # ⚠ a RATIO OF MEANS over the LARGER of the two scales, not a mean of
    # per-sample ratios: a clamped well sits exactly on its mode and contributes
    # ~0, so an |E_0| denominator can be ~0 (measured: -149011.5 on one arm of the
    # first run, and -2.4e7 on a one-mode-per-well fixture where E_0 is exactly 0).
    scale = max(
        float(np.mean(np.abs(e0))) if e0.size else 0.0,
        float(np.mean(np.abs(eT))) if eT.size else 0.0,
    )
    denom = scale if scale > 1e-8 else 1.0
    acc = np.asarray(trace["accept_prob"])
    took = np.asarray(trace["accepted"])
    return {
        "sampler": str(sampler),
        "steps": int(n_steps),
        "step_size": float(cfg.step_size),
        "tau_start": float(cfg.tau_start),
        "tau_end": float(cfg.tau_end),
        "temp_start": float(cfg.temp_start),
        "temp_end": float(cfg.temp_end),
        "metropolis_correction_applied": bool(sampler == "mala"),
        "noise_convention": (
            "fdt (chlu.core.integrators.langevin_step; T in energy units, §7.22)"
            if sampler == "uld"
            else "overdamped: x' = x - eps*grad E + sqrt(2*eps*T)*xi (T in energy units)"
        ),
        "mean_accept_prob": float(acc.mean()) if acc.size else float("nan"),
        "final_accept_prob": float(acc[-1].mean()) if acc.size else float("nan"),
        "accepted_fraction": float(took.mean()) if took.size else float("nan"),
        "energy_first": float(np.mean(e0)) if e0.size else float("nan"),
        "energy_final": float(np.mean(eT)) if eT.size else float("nan"),
        "energy_note": (
            "energy_first/final/relative_drop are read off energy_ref (the state "
            "scored at the FIXED tau_end). ⛔ trace['energy'] is the annealed "
            "landscape's own energy and its endpoints are values of two different "
            "functions — it is logged for the audit, never differenced."
        ),
        "energy_absolute_change": float(np.mean(eT) - np.mean(e0)) if e0.size else float("nan"),
        "energy_scale_for_the_ratio": float(denom),
        "energy_relative_drop": float((np.mean(e0) - np.mean(eT)) / denom)
        if e0.size else float("nan"),
        "energy_is_flat": bool(
            e0.size and float(np.mean(np.abs(e0 - eT)) / denom) < 1e-6
        ),
        "mean_clamp_pull": float(np.asarray(trace["clamp_pull"]).mean())
        if np.asarray(trace["clamp_pull"]).size else float("nan"),
        "mean_codes_changed": float(changed.sum(axis=1).mean()) if changed.size else 0.0,
        "frac_rollouts_that_moved": float((changed.sum(axis=1) > 0).mean())
        if changed.size else 0.0,
    }


def save_trace(trace: dict, path) -> str:
    """⭐ Write the audit trail to disk.  ⛔ Not summarised in prose — on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        energy=np.asarray(trace["energy"], np.float32),
        energy_ref=np.asarray(trace["energy_ref"], np.float32),
        accept_prob=np.asarray(trace["accept_prob"], np.float32),
        accepted=np.asarray(trace["accepted"], np.bool_),
        clamp_pull=np.asarray(trace["clamp_pull"], np.float32),
        tau=np.asarray(trace["tau"], np.float32),
        temperature=np.asarray(trace["temperature"], np.float32),
        code_init=np.asarray(trace["code_init"], np.int16),
        code_final=np.asarray(trace["code_final"], np.int16),
    )
    return str(path)
