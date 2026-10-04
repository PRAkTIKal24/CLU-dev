"""⭐ The matched-capacity **NULL ARMS** N1–N5 (C2W5 ``orgdiv-null-arms``).

Built against ``.claude/outputs/orgdiv-cat-test/FROZEN-interfaces.md`` — **the same
frozen ``phi`` instance, the same launch protocol, the same reader class, the same
split** as :mod:`chlu.core.factored_store`'s physics arm. Only the *organizer*
varies, which is ``PREREG-TierII.md`` §0's registered control (the organizer swap).

⚠ **What this module is for, after the C2W5 re-scope.** ``orgdiv-cat-test`` died at
K5 (the physics arm reads ``0.0008`` vs chance ``0.0004`` on unseen queries), so
there is no physics arm to swap against and no ``OD`` to compute. The Hub's
re-scoped question is the **family-solvability audit**: *does ANY matched-capacity
organizer clear ``chance + 0.05`` on the rule-4-valid unseen split?* These are the
arms that answer it, each in its strongest registered form (prereg §4.2).

**The five arms** (prereg §4.2; each is a map ``launch points -> z``, where ``z`` is
the ``(P, d+m)`` object the FROZEN reader class consumes — exactly the shape the
physics arm's settled read produces):

* **N1** :func:`n1_gradient_placed` — ⭐ the cleanest, most damaging swap.
  **Identical store parameterisation** (``centers``/``log_width``/``amp`` of the
  very same :class:`~chlu.core.factored_store.FactoredStore`), trained by plain
  Adam on the read objective, read by a **static assignment rule**. No rollout
  anywhere, DOF and bytes identical to the physics arm by construction.
* **N2** :func:`n2_vq` — k-means++ / VQ-STE(+EMA) / product-VQ codebooks.
* **N3** :func:`n3_static_geometric` — the fitted power/Apollonius rule
  ``argmin_j[||z-c_j||^2/2 sigma_j^2 - b_j]`` (F5's null), ``(c, sigma, b)`` fitted
  jointly, plus the **oracle-imitation** variant fitted on the physics arm's own
  assignments (T5.2 rider (i)).
* **N4** :func:`n4_knn` — no training; raw keys + payloads, ``k`` and weighting
  swept (the C2W1 ``knn2_idw`` substitute was the arm that beat us).
* **N5** :func:`n5_titans` — surprise-gated fast-weight write with momentum and
  weight decay; **init = PARAMETERS, per-stream deviation = STATE** (declared).

⛔ **Two rules this module enforces mechanically, because the wave dies if either
slips.** (i) Nothing here ever touches ``Q_unseen``: every fit, every selection and
every hyperparameter is a function of the SEEN split only (the harness holds out a
validation slice *from within SEEN*). (ii) ``phi`` is never re-drawn, never
re-scaled and never re-keyed — :func:`launch_points` reproduces
:func:`~chlu.core.factored_store.multi_particle_read`'s launch stage **exactly**,
including its key-folding, so the arms and the physics arm see bit-identical
launch points (asserted in ``tests/test_null_arms.py``).

⛔ **Claim form (prereg §2.6, inherited verbatim):** no well, code or atom is ever
named semantically here or in any artifact this module produces.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.factored_store import (
    CatFamily,
    CatTestConfig,
    FactoredStore,
    FrozenPhi,
    exact_set_accuracy,
    multi_particle_read,
    occupancy,
    store_population_spacing,
    write_store,
)

__all__ = [
    "NullArmGrid",
    "ARMS",
    "launch_points",
    "store_codebook",
    "n1_gradient_placed",
    "n2_vq",
    "n3_static_geometric",
    "n4_knn",
    "n5_titans",
    "phi_decodability_ceiling",
    "arm_ledger",
    "read_flops",
    "fit_code_payloads",
    "shuffle_launches",
    # -- the ZERO-PARAMETER identity readers (C2W7 reconciliation 1) ---------
    "READERS_IDENTITY",
    "READERS_PLUS_IDENTITY",
    "well_identity_fit",
    "well_identity_apply",
    "sum_identity_fit",
    "sum_identity_apply",
    "fit_readers_plus_identity",
    "apply_reader_plus_identity",
    "score_readers_plus_identity",
    "shrinkage_report",
    # -- ⭐ C2W11 (spoke C): the organizer swap's null side -------------------
    "feature_launch_states",
    "feature_keys",
    "n1_confidence",
    "n2_confidence",
    "n3_confidence",
    "n4_confidence",
    "n5_confidence",
    "novelty_auroc",
    "expected_calibration_error",
    "instantiate_landscape",
    "anytime_read",
    "feature_decodability_ceiling",
]


# ==========================================================================
# the registered grid (prereg §4.3 + the F3/A17.4 tuning standard)
# ==========================================================================
@dataclass
class NullArmGrid:
    """The **registered** tuning budget, committed before the grid ran.

    prereg §4.3: *"each arm gets >= 5 learning-rate points x 3 capacity points x 3
    seeds on the SEEN split, selected by a held-out-from-seen validation split
    (never on ``Q_unseen``)"*. ⚠ For the **gradient-free** arms the learning-rate
    axis is declared **substituted**, never dropped: N2/k-means gets 10 k-means++
    restarts x 5 commitment costs, N4 gets 5 ``k`` x 2 weightings, N3 is fitted by
    Adam on a soft surrogate and therefore keeps a genuine learning-rate axis.
    """

    # -- shared -------------------------------------------------------------
    lrs: Tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)  # >= 5 points
    tune_seeds: Tuple[int, ...] = (0, 1, 2)
    score_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    n_val: int = 32  # held out FROM SEEN (128 -> 96 train / 32 val)
    steps: int = 400

    # -- N1: capacity = atoms per well; the matched point is the physics arm's
    # ⚠ ``a`` covers BOTH readings of the FROZEN ledger (its table says
    # ``a = 12``/21 504 B, the cat-test's registered deviation D2 ran ``a = 32``/
    # 57 344 B). The ambiguity is resolved by MEASURING both, never by improvising.
    n1_atoms_per_well: Tuple[int, ...] = (12, 32, 64)
    n1_taus: Tuple[float, ...] = (0.05, 0.2, 1.0)
    n1_inits: Tuple[str, ...] = ("shell", "written")

    # -- N2 ------------------------------------------------------------------
    n2_variants: Tuple[str, ...] = ("kmeans", "vq_ste", "product_vq")
    n2_codes: Tuple[int, ...] = (32, 64, 128)  # capacity points (N_a, 2N_a, 4N_a)
    n2_commitments: Tuple[float, ...] = (0.0, 0.1, 0.25, 1.0, 4.0)  # >= 5 points
    n2_restarts: int = 10

    # -- N3: capacity = which of (c, sigma, b) is free ----------------------
    n3_levels: Tuple[str, ...] = ("b", "sb", "csb")
    n3_payloads: Tuple[str, ...] = ("written", "fitted")

    # -- N4 ------------------------------------------------------------------
    n4_ks: Tuple[int, ...] = (1, 2, 3, 5, 10)
    n4_weights: Tuple[str, ...] = ("uniform", "idw")
    n4_keys: Tuple[str, ...] = ("set_code", "launch_mean")

    # -- N5: capacity = fast-weight hidden width ----------------------------
    n5_hidden: Tuple[int, ...] = (64, 413, 1024)  # 413 ~= matched param count
    n5_momentum: Tuple[float, ...] = (0.0, 0.9)
    n5_decay: Tuple[float, ...] = (0.0, 0.01)
    n5_gate: Tuple[str, ...] = ("none", "surprise")
    n5_chunk: Tuple[int, ...] = (1, 32)
    n5_passes: int = 4
    n5_pretrain_steps: int = 400

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


ARMS = ("N1", "N2", "N3", "N4", "N5")


# ==========================================================================
# the launch protocol — byte-identical to the physics arm's
# ==========================================================================
def launch_points(phi: FrozenPhi, cfg: CatTestConfig, indicators: np.ndarray,
                  key, batch: int = 256) -> np.ndarray:
    """``(B, N_a) -> (B, P, dim)`` the launch states, **exactly** as the physics
    read produces them.

    ⛔ This deliberately reproduces
    :func:`~chlu.core.factored_store.multi_particle_read`'s launch stage
    line-for-line, *including its ``jax.random.fold_in(key, lo)`` batching*, so
    that with a zero-step settle the two are bit-identical. That equality is the
    ``PhiMismatchError`` guard for this wave and it is asserted in the tests.
    """
    ind = jnp.asarray(indicators, dtype=jnp.float32)
    B = int(ind.shape[0])
    out = []

    @eqx.filter_jit
    def launch_all(ind_b, keys_b):
        return jax.vmap(lambda i, k: phi.launch(i, k, float(cfg.query_sigma)))(
            ind_b, keys_b)

    for lo in range(0, B, int(batch)):
        hi = min(lo + int(batch), B)
        keys_b = jax.random.split(jax.random.fold_in(key, lo), hi - lo)
        out.append(np.asarray(launch_all(ind[lo:hi], keys_b)))
    return np.concatenate(out, axis=0)


def shuffle_launches(q0: np.ndarray, seed: int) -> np.ndarray:
    """⭐ The **laundering control** every arm runs beside its score.

    Permutes the launch blocks across queries, so the query carries **no**
    information about ``A(x)`` while every byte of capacity, every fitted
    parameter and every training step is preserved. An arm that still scores here
    is scoring on a fitting artifact, not on organization. (Registered prediction:
    ``<= chance + 0.005`` on every arm.)
    """
    rng = np.random.default_rng(int(seed) + 7717)
    return np.asarray(q0)[rng.permutation(len(q0))]


def store_codebook(V, cfg: CatTestConfig, n_wells: Optional[int] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """The **arm's own** ``(anchors, payloads)`` codebook, for reader R2.

    FROZEN §(ii): *"[R2's] null-arm form reads the null arm's own codebook
    payloads."* For an atom-parameterised arm the codebook is the per-group mean
    atom, weighted by depth — the arm's own answer to "where is well ``j`` and what
    does it carry", with no reference to the physics arm or to the true ``v_j``.
    """
    n_wells = int(cfg.n_wells if n_wells is None else n_wells)
    c = np.asarray(V.centers, dtype=np.float64)
    w = np.asarray(V.amp, dtype=np.float64) ** 2 + 1e-12
    n_atoms = c.shape[0]
    idx = (np.arange(n_atoms) * n_wells) // n_atoms
    anchors = np.zeros((n_wells, cfg.addr_dim))
    payloads = np.zeros((n_wells, cfg.payload_dim))
    for j in range(n_wells):
        m = idx == j
        if not m.any():
            continue
        ww = w[m] / w[m].sum()
        anchors[j] = (ww[:, None] * c[m, : cfg.addr_dim]).sum(0)
        payloads[j] = (ww[:, None] * c[m, cfg.addr_dim:]).sum(0)
    return anchors, payloads


# ==========================================================================
# ledgers (prereg §1: bytes, capacity and per-query read compute, per arm)
# ==========================================================================
def arm_ledger(arm: str, cfg: CatTestConfig, *, n_params: int,
               n_state: int = 0, phi_bytes: int = 576, **extra) -> Dict[str, Any]:
    """The two-sided ledger row an arm may not be scored without.

    ``learned-initial-state rule`` (prereg §1): an initialisation is
    **PARAMETERS**; only the per-sequence deviation is **STATE**. Both are declared
    here for every arm, which is what makes N5 (a fast-weight arm) comparable at
    all.
    """
    return {"arm": arm, "n_params": int(n_params), "param_bytes": int(n_params) * 4,
            "n_state": int(n_state), "state_bytes": int(n_state) * 4,
            "phi_bytes": int(phi_bytes),
            "total_bytes": int(n_params) * 4 + int(n_state) * 4 + int(phi_bytes),
            **extra}


def read_flops(arm: str, cfg: CatTestConfig, *, n_units: int = 0,
               hidden: int = 0) -> int:
    """Per-**query** read cost, in multiply-adds (prereg §1's compute rule).

    ⚠ The rule ("if arms differ by > 2x, the cheaper arm additionally runs at the
    richer arm's budget, else the comparison is void") binds a *comparison*. After
    the re-scope there is no physics arm in this report's comparison, so the column
    is **reported**, and the physics arm's cost is quoted from the cat-test's
    config for context: ``P * (400 + 800)`` Verlet steps, each a full
    ``n_atoms x dim`` gradient.
    """
    P, d, m = int(cfg.n_particles), int(cfg.addr_dim), int(cfg.payload_dim)
    if arm in ("N1", "N2", "N3"):
        return int(P * n_units * (d + 1))
    if arm == "N4":
        return int(cfg.n_items * (d + m))
    if arm == "N5":
        return int(d * hidden + hidden * m)
    if arm == "physics":
        return int(P * (cfg.address_steps + cfg.read_steps) * cfg.n_atoms
                   * (cfg.dim + 1))
    return 0


# ==========================================================================
# N1 — gradient-placed atoms (⭐ the cleanest swap)
# ==========================================================================
def _n1_logits(V, q0: jnp.ndarray, addr_dim: int) -> jnp.ndarray:
    """``log`` of the atom's own well contribution ``A_i exp(-r^2/2 s_i^2)``.

    Identical functional form to
    :class:`~chlu.core.memory_potentials.AtomDictionaryPotential`'s summand — the
    arm reads the **same landscape family** the physics arm settles in; it simply
    assigns statically instead of relaxing.
    """
    s = jnp.exp(V.log_width)
    d2 = jnp.sum((q0[:, None, :addr_dim] - V.centers[None, :, :addr_dim]) ** 2, -1)
    return 2.0 * jnp.log(jnp.abs(V.amp) + 1e-12) - d2 / (2.0 * s ** 2 + 1e-9)


def _n1_read(V, q0: jnp.ndarray, addr_dim: int, tau, hard: bool = False):
    lg = _n1_logits(V, q0, addr_dim)
    if hard:
        return V.centers[jnp.argmax(lg, axis=-1)]
    return jax.nn.softmax(lg / tau, axis=-1) @ V.centers


@eqx.filter_jit
def _n1_train_core(params, static, Q, Y, tau, lr, *, steps: int, addr_dim: int,
                   payload_dim: int, dim: int, n_particles: int, batch: int):
    """⚠ Module-level and jitted ONCE per (shape, static) tuple.

    A closure defined inside the arm builder would recompile on **every grid
    point** (90 configs x 3 seeds), which is how a "5 lr x 3 capacity" budget turns
    into an afternoon of XLA. ``lr`` and ``tau`` are traced
    (``optax.inject_hyperparams``), so only the atom count and the batch size
    trigger a recompile.
    """
    import optax

    d, m, P = int(addr_dim), int(payload_dim), int(n_particles)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=lr)

    def loss(p):
        V = eqx.combine(p, static)
        z = _n1_read(V, Q, d, tau).reshape(int(batch), P, int(dim))
        return jnp.mean(jnp.sum((z[:, :, d:d + m].sum(1) - Y) ** 2, -1))

    st = opt.init(params)

    def body(carry, _):
        p, st = carry
        v, g = eqx.filter_value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return (eqx.apply_updates(p, u), st), v

    (p, _), hist = jax.lax.scan(body, (params, st), None, length=int(steps))
    return p, hist


def n1_gradient_placed(cfg: CatTestConfig, family: CatFamily, anchors: np.ndarray,
                       q0_tr: np.ndarray, y_tr: np.ndarray, *, lr: float,
                       tau: float, init: str = "shell", steps: int = 400,
                       seed: int = 0, atoms_per_well: Optional[int] = None
                       ) -> Dict[str, Any]:
    """⭐ **N1.** Same store parameterisation, plain Adam, **static** read.

    ``init="written"`` starts each group's atom payload block at that well's ``v_j``
    (the state a *successful* physics write would have produced) — a declared
    designed mechanism costing zero parameters, included so that N1 cannot be
    accused of being handicapped relative to a written physics store.
    ``init="shell"`` is the physics arm's own init, bit-identical.

    ⛔ No rollout anywhere: the read is one softmax (or argmax) over atoms, so any
    margin N1 shows is attributable to *gradient placement*, and any margin the
    physics arm shows over N1 is attributable to *the dynamics as a training
    signal* and to nothing else.
    """
    c = cfg if atoms_per_well is None else _with(cfg, atoms_per_well=int(atoms_per_well))
    d, m, P = int(c.addr_dim), int(c.payload_dim), int(c.n_particles)
    store = FactoredStore(c, anchors, jax.random.PRNGKey(int(seed)))
    V0 = store.V
    if init == "written":
        n_atoms = int(V0.centers.shape[0])
        grp = (np.arange(n_atoms) * c.n_wells) // n_atoms
        cc = np.array(V0.centers, dtype=np.float32, copy=True)
        cc[:, d:d + m] = np.asarray(family.payloads)[grp]
        V0 = eqx.tree_at(lambda t: t.centers, V0, jnp.asarray(cc, jnp.float32))

    Q = jnp.asarray(np.asarray(q0_tr).reshape(-1, c.dim), jnp.float32)
    Y = jnp.asarray(y_tr, jnp.float32)
    B = int(np.asarray(q0_tr).shape[0])
    spec = jax.tree_util.tree_map(eqx.is_inexact_array, V0)
    params, static = eqx.partition(V0, spec)
    p, hist = _n1_train_core(params, static, Q, Y,
                             jnp.asarray(tau, jnp.float32),
                             jnp.asarray(lr, jnp.float32),
                             steps=int(steps), addr_dim=d, payload_dim=m,
                             dim=int(c.dim), n_particles=P, batch=B)
    V = eqx.combine(p, static)
    n_params = int(sum(np.asarray(x).size
                       for x in jax.tree_util.tree_leaves(
                           eqx.filter(V, eqx.is_inexact_array))))
    anc, pay = store_codebook(V, c)
    return {"V": V, "cfg": c, "loss_first": float(hist[0]), "loss_last": float(hist[-1]),
            "codebook": (anc, pay), "tau": float(tau),
            "ledger": arm_ledger("N1", c, n_params=n_params,
                                 n_atoms=int(V.centers.shape[0]),
                                 read_flops=read_flops("N1", c,
                                                       n_units=int(V.centers.shape[0])),
                                 atoms_per_well=int(c.atoms_per_well), init=init)}


def n1_apply(fit: Dict[str, Any], q0: np.ndarray, hard: bool = False) -> np.ndarray:
    c = fit["cfg"]
    B, P = np.asarray(q0).shape[0], int(c.n_particles)
    Q = jnp.asarray(np.asarray(q0).reshape(-1, c.dim), jnp.float32)
    z = _n1_read(fit["V"], Q, int(c.addr_dim),
                 jnp.asarray(fit["tau"], jnp.float32), hard=hard)
    return np.asarray(z).reshape(B, P, c.dim)


# ==========================================================================
# N2 — VQ (k-means++ / VQ-STE+EMA / product-VQ)
# ==========================================================================
def _kmeanspp(X: np.ndarray, k: int, rng, restarts: int = 10, iters: int = 50):
    """k-means++ with ``restarts`` restarts; returns the lowest-inertia codebook."""
    best, best_in = None, np.inf
    for _ in range(int(restarts)):
        C = np.empty((k, X.shape[1]))
        C[0] = X[rng.integers(len(X))]
        d2 = ((X - C[0]) ** 2).sum(-1)
        for i in range(1, k):
            p = d2 / max(d2.sum(), 1e-12)
            C[i] = X[rng.choice(len(X), p=p)]
            d2 = np.minimum(d2, ((X - C[i]) ** 2).sum(-1))
        for _ in range(int(iters)):
            a = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            for i in range(k):
                mm = a == i
                if mm.any():
                    C[i] = X[mm].mean(0)
        a = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        inertia = float(((X - C[a]) ** 2).sum())
        if inertia < best_in:
            best, best_in = C.copy(), inertia
    return best, best_in


def fit_code_payloads(assign: np.ndarray, Y: np.ndarray, n_codes: int,
                      ridge: float = 1e-6) -> np.ndarray:
    """Least-squares per-code payloads from the SEEN split.

    ``assign`` is ``(B, P)``; the design matrix is the per-query **code count**
    vector, so this solves ``min_V || M V - Y ||^2`` with ``M[b, j] = #{p :
    assign[b,p] = j}``. ⚠ **Ledgered as ``n_codes * m`` ARM parameters** (prereg
    §4.2 registers N2's DOF as ``N_a(d+m)``), not as reader parameters — the FROZEN
    reader-capacity bound (``< N_a m = 256`` *fitted reader* params) is a bound on
    the decoder, and it is respected separately. The adjacency to SP-1 is real and
    is reported: SP-1 fits on the TRUE indicator, this fits on an *organizer's
    recovered* assignment, which is exactly the thing under test.
    """
    B = len(Y)
    M = np.zeros((B, int(n_codes)))
    np.add.at(M, (np.repeat(np.arange(B), assign.shape[1]), assign.ravel()), 1.0)
    A = M.T @ M + float(ridge) * np.eye(int(n_codes))
    return np.linalg.solve(A, M.T @ np.asarray(Y))


@eqx.filter_jit
def _n2_ste_core(par, X, Y, beta, lr, *, steps: int, n_particles: int,
                 payload_dim: int):
    """VQ-STE with the published commitment cost, trained on the READ objective."""
    import optax

    P, m = int(n_particles), int(payload_dim)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=lr)

    def loss(par):
        Cc, Vv = par
        d2 = jnp.sum((X[:, None, :] - Cc[None]) ** 2, -1)
        j = jnp.argmin(d2, -1)
        q = Cc[j]
        read = jnp.mean(jnp.sum((Vv[j].reshape(-1, P, m).sum(1) - Y) ** 2, -1))
        commit = beta * jnp.mean(
            jnp.sum((X - jax.lax.stop_gradient(q)) ** 2, -1)
            + jnp.sum((jax.lax.stop_gradient(X) - q) ** 2, -1))
        return read + commit

    st = opt.init(par)

    def body(carry, _):
        par, st = carry
        v, g = jax.value_and_grad(loss)(par)
        u, st = opt.update(g, st, par)
        return (optax.apply_updates(par, u), st), v

    (par, _), hist = jax.lax.scan(body, (par, st), None, length=int(steps))
    return par, hist


def n2_vq(cfg: CatTestConfig, family: CatFamily, q0_tr: np.ndarray,
          y_tr: np.ndarray, *, variant: str = "kmeans", n_codes: int = 32,
          commitment: float = 0.25, lr: float = 1e-2, steps: int = 400,
          seed: int = 0, restarts: int = 10, payload_source: str = "fitted",
          anchors: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """**N2.** Best of {k-means++ x10 restarts, VQ-STE with EMA, product-VQ}.

    The codebook lives in the **address** block (that is where the launch points
    live and where the physics arm's wells sit); the per-code payload is the arm's
    stored content. ``payload_source="written"`` maps each code to its nearest well
    anchor and takes that well's written ``v_j`` — the *same* store content the
    physics arm carries; ``"fitted"`` solves for it on SEEN.
    """

    d, m, P = int(cfg.addr_dim), int(cfg.payload_dim), int(cfg.n_particles)
    rng = np.random.default_rng(int(seed) + 991)
    X = np.asarray(q0_tr)[..., :d].reshape(-1, d)
    Y = np.asarray(y_tr)
    groups = 1
    extra: Dict[str, Any] = {}

    if variant == "kmeans":
        C, inertia = _kmeanspp(X, int(n_codes), rng, restarts=int(restarts))
        extra["inertia"] = inertia
    elif variant == "product_vq":
        groups = 2
        half = d // 2
        Cs = []
        for g in range(groups):
            sl = slice(g * half, (g + 1) * half) if g == 0 else slice(half, d)
            Cg, _ = _kmeanspp(X[:, sl], int(n_codes), rng, restarts=int(restarts))
            Cs.append(Cg)
        C = Cs
    elif variant == "vq_ste":
        # straight-through VQ trained on the READ objective + commitment cost,
        # with an EMA codebook update (the published rule).
        C0, _ = _kmeanspp(X, int(n_codes), rng, restarts=2, iters=10)
        Cj = jnp.asarray(C0, jnp.float32)
        Vp = jnp.asarray(rng.normal(size=(int(n_codes), m)) * 0.1, jnp.float32)
        (Cj, Vp), hist = _n2_ste_core(
            (Cj, Vp), jnp.asarray(X, jnp.float32), jnp.asarray(Y, jnp.float32),
            jnp.asarray(commitment, jnp.float32), jnp.asarray(lr, jnp.float32),
            steps=int(steps), n_particles=P, payload_dim=m)
        C = np.asarray(Cj)
        extra["loss_first"], extra["loss_last"] = float(hist[0]), float(hist[-1])
        extra["ema_payloads"] = np.asarray(Vp)
    else:
        raise ValueError(f"unknown N2 variant {variant!r}")

    def assign(q0):
        Z = np.asarray(q0)[..., :d]
        if groups == 1:
            return ((Z[:, :, None, :] - C[None, None]) ** 2).sum(-1).argmin(-1)
        half = d // 2
        out = []
        for g, Cg in enumerate(C):
            sl = slice(0, half) if g == 0 else slice(half, d)
            out.append(((Z[:, :, None, sl] - Cg[None, None]) ** 2).sum(-1).argmin(-1))
        return np.stack(out, -1)  # (B, P, groups)

    a_tr = assign(q0_tr)
    if groups == 1:
        if payload_source == "written" and anchors is not None:
            near = ((C[:, None, :] - np.asarray(anchors)[None]) ** 2).sum(-1).argmin(1)
            Pay = np.asarray(family.payloads)[near]
        else:
            Pay = fit_code_payloads(a_tr, Y, int(n_codes))
        n_params = int(C.size + Pay.size)
        codebook = (C, Pay)
    else:
        Pay = [fit_code_payloads(a_tr[..., g], Y * 0.5, int(n_codes))
               for g in range(groups)]
        n_params = int(sum(c.size for c in C) + sum(p.size for p in Pay))
        codebook = (np.concatenate(C, axis=1), Pay[0] + Pay[1])

    def apply(q0):
        a = assign(q0)
        Bq = np.asarray(q0).shape[0]
        z = np.zeros((Bq, P, cfg.dim), dtype=np.float32)
        if groups == 1:
            z[:, :, :d] = C[a]
            z[:, :, d:] = Pay[a]
        else:
            half = d // 2
            z[:, :, :half] = C[0][a[..., 0]]
            z[:, :, half:d] = C[1][a[..., 1]]
            z[:, :, d:] = Pay[0][a[..., 0]] + Pay[1][a[..., 1]]
        return z

    return {"apply": apply, "assign": assign, "codebook": codebook, "cfg": cfg,
            "variant": variant,
            "ledger": arm_ledger("N2", cfg, n_params=n_params, n_codes=int(n_codes),
                                 variant=variant, commitment=float(commitment),
                                 payload_source=payload_source,
                                 read_flops=read_flops("N2", cfg,
                                                       n_units=int(n_codes) * groups)),
            **extra}


# ==========================================================================
# N3 — the fitted static-geometric rule (F5's null) + the oracle imitation
# ==========================================================================
@eqx.filter_jit
def _n3_train_core(pfree, pfix, L, Y, pay_tbl, tgt, tau, lr, *, steps: int,
                   batch: int, n_particles: int, payload_dim: int,
                   pay_fixed: bool, use_target: bool):
    """Adam on the softmin surrogate of the power/Apollonius rule (module-level:
    one compile per (free-parameter structure, shape), not one per grid point)."""
    import optax

    P, m = int(n_particles), int(payload_dim)
    opt = optax.inject_hyperparams(optax.adam)(learning_rate=lr)

    def loss(pf):
        p = dict(pfix)
        p.update(pf)
        s2 = 2.0 * jnp.exp(p["log_sigma"]) ** 2 + 1e-9
        d2 = jnp.sum((L[:, None, :] - p["c"][None]) ** 2, -1)
        lg = -d2 / s2[None, :] + p["b"][None, :]
        if use_target:
            return -jnp.mean(jax.nn.log_softmax(lg / tau, -1)[
                jnp.arange(lg.shape[0]), tgt])
        w = jax.nn.softmax(lg / tau, -1)
        tbl = pay_tbl if pay_fixed else p["pay"]
        return jnp.mean(jnp.sum(((w @ tbl).reshape(int(batch), P, m).sum(1) - Y) ** 2,
                                -1))

    st = opt.init(pfree)

    def body(carry, _):
        pf, st = carry
        v, g = jax.value_and_grad(loss)(pf)
        u, st = opt.update(g, st, pf)
        return (optax.apply_updates(pf, u), st), v

    (pf, _), hist = jax.lax.scan(body, (pfree, st), None, length=int(steps))
    return pf, hist


def n3_static_geometric(cfg: CatTestConfig, family: CatFamily, anchors: np.ndarray,
                        q0_tr: np.ndarray, y_tr: np.ndarray, *, level: str = "sb",
                        lr: float = 1e-2, tau: float = 0.2, steps: int = 400,
                        seed: int = 0, payload_source: str = "written",
                        target_assign: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """**N3.** ``argmin_j [ ||z-c_j||^2 / 2 sigma_j^2 - b_j ]``, fitted on SEEN.

    ⭐ This is **F5's null** and — per prereg §7 Thm O2 — the decision rule the
    physics arm provably has to leading order. ``level`` is the capacity axis:
    ``"b"`` fits the power weights only, ``"sb"`` fits ``(sigma, b)``, ``"csb"``
    fits the centers too.

    ⚠ Fitted by **Adam on a softmin surrogate** (temperature ``tau``), evaluated
    with the hard rule. This is a declared strengthening of the cat-test's
    Powell/coordinate search: it makes the "5 learning-rate points" axis genuine
    for this arm and it reaches lower training loss.

    ``target_assign`` (``(B, P)`` ints) switches the arm to the **oracle-imitation**
    form of T5.2 rider (i): the objective becomes cross-entropy against the physics
    arm's OWN assignments. *A physics arm that cannot beat an imitation of itself
    has no organization claim.*
    """
    d, m, P = int(cfg.addr_dim), int(cfg.payload_dim), int(cfg.n_particles)
    N = int(cfg.n_wells)
    B = np.asarray(q0_tr).shape[0]
    L = jnp.asarray(np.asarray(q0_tr)[..., :d].reshape(-1, d), jnp.float32)
    Y = jnp.asarray(y_tr, jnp.float32)
    ruler = cfg.s_measured if cfg.s_measured is not None else cfg.atom_width
    par = {"b": jnp.zeros((N,)),
           "log_sigma": jnp.full((N,), float(np.log(ruler))),
           "c": jnp.asarray(anchors[:, :d], jnp.float32)}
    if payload_source == "written":
        pay_tbl = jnp.asarray(family.payloads, jnp.float32)
    else:
        par["pay"] = jnp.asarray(np.zeros((N, m)), jnp.float32)
        pay_tbl = None
    free = {"b": ("b",), "sb": ("b", "log_sigma"),
            "csb": ("b", "log_sigma", "c")}[level]
    if payload_source == "fitted":
        free = free + ("pay",)
    tgt = None if target_assign is None else jnp.asarray(
        np.asarray(target_assign).reshape(-1), jnp.int32)

    pfree = {k: par[k] for k in free}
    pfix = {k: v for k, v in par.items() if k not in free}
    pfree, hist = _n3_train_core(
        pfree, pfix, L, Y,
        pay_tbl if pay_tbl is not None else jnp.zeros((N, m), jnp.float32),
        tgt if tgt is not None else jnp.zeros((L.shape[0],), jnp.int32),
        jnp.asarray(tau, jnp.float32), jnp.asarray(lr, jnp.float32),
        steps=int(steps), batch=B, n_particles=P, payload_dim=m,
        pay_fixed=bool(pay_tbl is not None), use_target=bool(tgt is not None))
    p = dict(pfix)
    p.update(pfree)
    c_np = np.asarray(p["c"])
    sig = np.asarray(jnp.exp(p["log_sigma"]))
    b_np = np.asarray(p["b"])
    # ⭐ C2W11 (spoke C): the fitted rule is returned explicitly so the arm's
    # DECISION MARGIN is available as its confidence channel (VALUE leg ii's
    # null side). Additive: an extra key, no existing key changes.
    rule = {"c": c_np, "sigma": sig, "b": b_np}
    pay_np = (np.asarray(family.payloads) if pay_tbl is not None
              else np.asarray(p["pay"]))

    def assign(q0):
        Z = np.asarray(q0)[..., :d]
        d2 = ((Z[:, :, None, :] - c_np[None, None]) ** 2).sum(-1)
        return np.argmin(d2 / (2.0 * sig[None, None, :] ** 2 + 1e-9)
                         - b_np[None, None, :], -1)

    def apply(q0):
        a = assign(q0)
        z = np.zeros((np.asarray(q0).shape[0], P, cfg.dim), dtype=np.float32)
        z[:, :, :d] = c_np[a]
        z[:, :, d:] = pay_np[a]
        return z

    n_params = int(sum(np.asarray(par[k]).size for k in free))
    if payload_source == "written":
        n_params += int(np.asarray(family.payloads).size)  # store content, ledgered
    return {"apply": apply, "assign": assign, "cfg": cfg, "level": level,
            "rule": rule,
            "codebook": (c_np, pay_np), "loss_first": float(hist[0]),
            "loss_last": float(hist[-1]), "oracle_imitation": tgt is not None,
            "ledger": arm_ledger("N3", cfg, n_params=n_params, level=level,
                                 payload_source=payload_source,
                                 read_flops=read_flops("N3", cfg, n_units=N))}


# ==========================================================================
# N4 — kNN (no training; the C2W1 substitute that beat us)
# ==========================================================================
def n4_knn(cfg: CatTestConfig, key_tr: np.ndarray, y_tr: np.ndarray,
           key_q: np.ndarray, *, k: int = 1, weight: str = "idw") -> np.ndarray:
    """**N4.** ``k`` nearest SEEN keys, uniform or inverse-distance weighted.

    ⛔ A *direct predictor*: it has no ``z`` and therefore no reader-class column —
    its decoder is the weighting itself (0 fitted parameters, ``K(d+m)`` bytes of
    **state**). Reported that way rather than dressed up as a latent map.
    """
    D = np.linalg.norm(np.asarray(key_q)[:, None, :] - np.asarray(key_tr)[None],
                       axis=-1)
    kk = min(int(k), D.shape[1])
    idx = np.argsort(D, axis=1)[:, :kk]
    dd = np.take_along_axis(D, idx, 1)
    w = np.ones_like(dd) if weight == "uniform" else 1.0 / (dd + 1e-9)
    w = w / w.sum(1, keepdims=True)
    return (w[..., None] * np.asarray(y_tr)[idx]).sum(1)


def n4_keys(kind: str, phi: FrozenPhi, cfg: CatTestConfig, indicators: np.ndarray,
            q0: np.ndarray) -> np.ndarray:
    """N4's key space: the exact set code, or the arm-visible launch mean."""
    if kind == "set_code":
        return np.asarray(phi.set_code(jnp.asarray(indicators, jnp.float32)))
    o = np.asarray(phi.offsets)
    return (np.asarray(q0)[..., : cfg.addr_dim] - o[None]).mean(1)


# ==========================================================================
# N5 — Titans-style surprise-gated fast weights
# ==========================================================================
def _mlp_init(d: int, h: int, m: int, key):
    """⚠ Pinned to float32 **deliberately**, not incidentally.

    Every other array in this module is float32 (the repo's convention), but
    ``jax.random.normal``/``jnp.zeros`` follow the *global* x64 flag — and some
    tests in the suite enable it process-wide. The fast-weight pass then carries
    float64 ``M`` against float32 data, the surprise becomes float64, and
    ``lax.scan``'s carry types stop matching (caught by the full suite, invisible
    to this file's tests run alone). Pinning here keeps the arm bit-identical
    whatever the ambient flag is.
    """
    k1, k2 = jax.random.split(key)
    f32 = jnp.float32
    return {"W1": (jax.random.normal(k1, (d, h)) / np.sqrt(d)).astype(f32),
            "b1": jnp.zeros((h,), f32),
            "W2": (jax.random.normal(k2, (h, m)) / np.sqrt(h)).astype(f32),
            "b2": jnp.zeros((m,), f32)}


def _mlp(p, x):
    return jnp.tanh(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


@eqx.filter_jit
def _n5_pretrain_core(p, K0, Y0, lr, *, steps: int):
    """Meta-learn ``M_0`` (the arm's PARAMETERS) on the SEEN stream."""
    import optax

    opt = optax.inject_hyperparams(optax.adam)(learning_rate=lr)

    def loss(p):
        return jnp.mean(jnp.sum((_mlp(p, K0) - Y0) ** 2, -1))

    st = opt.init(p)

    def body(carry, _):
        p, st = carry
        v, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return (optax.apply_updates(p, u), st), v

    (p, _), hist = jax.lax.scan(body, (p, st), None, length=int(steps))
    return p, hist


@eqx.filter_jit
def _n5_stream_core(M, K0, Y0, batches, lr, momentum, decay, *, gated: bool):
    """The online surprise-gated fast-weight pass (the arm's STATE)."""
    def surprise(p, ib):
        return jnp.mean(jnp.sum((_mlp(p, K0[ib]) - Y0[ib]) ** 2, -1))

    S = jax.tree_util.tree_map(jnp.zeros_like, M)

    dt = jax.eval_shape(surprise, M, batches[0]).dtype  # never assume float32

    def body(carry, ib):
        M, S, lbar = carry
        ll, g = jax.value_and_grad(surprise)(M, ib)
        th = lr * jax.nn.sigmoid(ll - lbar) if gated else lr
        S = jax.tree_util.tree_map(lambda s, gg: momentum * s - th * gg, S, g)
        M = jax.tree_util.tree_map(lambda mm, s: (1.0 - decay) * mm + s, M, S)
        return (M, S, (0.9 * lbar + 0.1 * ll).astype(dt)), ll

    (M, _, _), ls = jax.lax.scan(body, (M, S, jnp.asarray(1.0, dt)), batches)
    return M, ls


def n5_titans(cfg: CatTestConfig, key_tr: np.ndarray, y_tr: np.ndarray, *,
              hidden: int = 413, lr: float = 1e-2, momentum: float = 0.9,
              decay: float = 0.01, gate: str = "surprise", chunk: int = 1,
              passes: int = 4, pretrain_steps: int = 400, order=None,
              seed: int = 0) -> Dict[str, Any]:
    """**N5.** ``M_t`` updated online by the published surprise-gated rule.

    ``S_t = eta S_{t-1} - theta_t grad_M l(M_{t-1}; k_t, v_t)`` ,
    ``M_t = (1 - alpha) M_{t-1} + S_t`` , with ``l = ||M(k) - v||^2`` (the surprise)
    and ``theta_t = theta * sigmoid(l_t - l_bar)`` when ``gate="surprise"``.

    ⭐ **Learned-initial-state rule, declared (prereg §1):** ``M_0`` is meta-learned
    on the SEEN stream by Adam and is counted as **PARAMETERS**; the per-stream
    deviation ``M_t - M_0`` is counted as **STATE**. Both appear in the ledger.
    Chunk granularity ``chunk`` is matched to the physics arm's write/organize
    granularity (1 item, or the organizer's batch of 32).
    """
    d = int(np.asarray(key_tr).shape[-1])
    m = int(np.asarray(y_tr).shape[-1])
    K0 = jnp.asarray(key_tr, jnp.float32)
    Y0 = jnp.asarray(y_tr, jnp.float32)
    M0 = _mlp_init(d, int(hidden), m, jax.random.PRNGKey(int(seed) + 5))
    lr_j = jnp.asarray(lr, jnp.float32)

    # -- meta-learn the initialisation (PARAMETERS) -------------------------
    M0, hist = _n5_pretrain_core(M0, K0, Y0, lr_j, steps=int(pretrain_steps))

    # -- the online surprise-gated pass over the write stream (STATE) -------
    n = int(K0.shape[0])
    order = np.arange(n) if order is None else np.asarray(order)[:n]
    idx = np.concatenate([np.asarray(order, dtype=np.int32)] * int(passes))
    nb = int(np.ceil(len(idx) / max(int(chunk), 1)))
    pad = nb * int(chunk) - len(idx)
    if pad:
        idx = np.concatenate([idx, idx[:pad]])
    batches = jnp.asarray(idx.reshape(nb, int(chunk)), jnp.int32)

    M, ls = _n5_stream_core(M0, K0, Y0, batches, lr_j,
                            jnp.asarray(momentum, jnp.float32),
                            jnp.asarray(decay, jnp.float32),
                            gated=bool(gate != "none"))
    n_params = int(sum(np.asarray(v).size for v in M0.values()))
    return {"M": M, "M0": M0, "apply": lambda k: np.asarray(
        _mlp(M, jnp.asarray(k, jnp.float32))),
        "pre_loss_first": float(hist[0]), "pre_loss_last": float(hist[-1]),
        "stream_loss_first": float(ls[0]), "stream_loss_last": float(ls[-1]),
        "ledger": arm_ledger("N5", cfg, n_params=n_params, n_state=n_params,
                             hidden=int(hidden), gate=gate, chunk=int(chunk),
                             momentum=float(momentum), decay=float(decay),
                             init_is="PARAMETERS", deviation_is="STATE",
                             read_flops=read_flops("N5", cfg, hidden=int(hidden)))}


# ==========================================================================
# ⛔ DECLARED OUT-OF-CLASS DIAGNOSTIC — the phi-decodability ceiling
# ==========================================================================
def phi_decodability_ceiling(phi: FrozenPhi, cfg: CatTestConfig, family: CatFamily,
                             q0_unseen: Optional[np.ndarray] = None,
                             chunk: int = 64) -> Dict[str, float]:
    """How much of ``A(x)`` survives the frozen ``phi`` **at all**?

    A combinatorial matched filter that enumerates all ``C(N_a, F)`` set codes and
    returns the nearest one, then reads the *true* ``v_j``. Two conditions:
    **noiseless** (the exact ``phi(x)``) and **as-launched** (the ``P``-launch mean
    ``mean_p(q0_p - o_p)``, which is the sufficient statistic every arm sees).

    ⛔ **Reported, NEVER scored as an arm** — the SP-1 precedent
    (``orgdiv-cat-test`` §7.3). It enumerates 35 960 combinations and consults the
    written payload table, so it is not matched-capacity to anything. Its job is to
    separate *"the arms are weak"* from *"the frozen ``phi`` destroyed the set"* —
    a distinction the audit's verdict depends on, and one that no arm's score can
    make on its own.
    """
    N_a, F = int(cfg.n_wells), int(cfg.f_subset)
    combos = np.array(list(itertools.combinations(range(N_a), F)), dtype=np.int32)
    ind = np.zeros((len(combos), N_a), dtype=np.float32)
    np.put_along_axis(ind, combos, 1.0, axis=1)
    codes = np.asarray(phi.set_code(jnp.asarray(ind)), dtype=np.float64)
    pay = np.asarray(family.payloads)
    y_tab = ind.astype(np.float64) @ pay

    ind_u = family.indicator(family.unseen, N_a)
    c_exact = np.asarray(phi.set_code(jnp.asarray(ind_u)), dtype=np.float64)
    out: Dict[str, float] = {}
    for name, C in (("noiseless", c_exact),
                    ("as_launched", None if q0_unseen is None else
                     (np.asarray(q0_unseen)[..., : cfg.addr_dim]
                      - np.asarray(phi.offsets)[None]).mean(1))):
        if C is None:
            continue
        hit, dec = [], []
        for lo in range(0, len(C), int(chunk)):
            D = ((C[lo:lo + chunk, None, :] - codes[None]) ** 2).sum(-1)
            j = D.argmin(1)
            dec.append(j)
            hit.append((combos[j] == family.unseen[lo:lo + chunk]).all(1))
        j = np.concatenate(dec)
        out[f"{name}_combo_exact"] = float(np.concatenate(hit).mean())
        out[f"{name}_accuracy"] = exact_set_accuracy(y_tab[j], family.y_unseen,
                                                     family.tol)
    out["n_combos"] = int(len(combos))
    return out


def _with(cfg: CatTestConfig, **kw) -> CatTestConfig:
    d = asdict(cfg)
    d.update(kw)
    return CatTestConfig(**d)


# ==========================================================================
# ⭐ THE ZERO-PARAMETER IDENTITY READERS (C2W7 reconciliation 1, `AMENDMENT-C2W7`)
# ==========================================================================
#
# ⛔ **Why these exist.** Every fitted member of the frozen reader class is fitted
# by **least squares** while the metric, :func:`exact_set_accuracy`, is a
# **thresholded** all-or-nothing accuracy. ``c2w7-read-cardinality`` §4 measured the
# consequence on one latent: lstsq shrinks ``diag(W)`` to ~0.40, which pushes the
# residual of the queries whose asserted set is *exactly right* from 0.006 to 0.537
# against ``tol = 0.234`` ⇒ the 72-parameter reader scores **0.0000** where a
# **zero-parameter** identity reader scores **0.0539 +/- 0.0207**.
#
# ⛔ **They are ADDED to the class, never substituted for it.** The default
# ``which=`` of every shipped ``fit_readers*`` is unchanged, so every prior code
# path stays bit-identical; a caller that wants the audit asks for it explicitly
# and reports fitted **and** identity columns side by side.
#
# ⚠ **The assumption they make, stated where it can be checked.** An identity
# reader assumes the latent is **already in the target's units and scale** — it has
# no gain and no bias with which to correct a mis-scaled code. Where that is false
# (a latent whose payload table is scaled by ``alpha != 1``, a code that sums to
# ``k`` rather than to ``F``, a store whose payloads were never written at the
# family's radius) the identity member scores 0 while the fitted member is exact.
# ``tests/test_reader_identity.py`` asserts both directions.

#: the zero-parameter identity twins, one per fitted member that has a natural one.
READERS_IDENTITY = ("well_identity", "sum_identity")

#: the frozen class **plus** its identity twins (the audit's scored class).
READERS_PLUS_IDENTITY = ("sum_linear", "well_table", "knn", "mlp",
                         "well_identity", "sum_identity")


def well_identity_fit(z, y, *, anchors, well_payloads) -> Dict[str, Any]:
    """⭐ ``yhat = sum_{j in set(occ(z))} v_j`` — **0 fitted parameters**.

    The unfitted twin of :func:`~chlu.core.factored_store._well_table_fit`: the
    same hard nearest-well assignment, the same payload table read from the store,
    but no ``W`` and no bias. ``z`` and ``y`` are accepted and **ignored** so the
    signature matches the fitted members (nothing is fitted; there is nothing to
    select on and nothing to overfit).

    ⭐ Note the ``set(...)``: the occupied wells are **deduplicated**, because the
    target is a sum over ``F`` *distinct* wells. This is the only choice in the
    reader and it is the generous one — the multiset variant is strictly worse
    whenever two particles share a well.
    """
    del z, y
    return {"kind": "well_identity", "anchors": np.asarray(anchors),
            "well_payloads": np.asarray(well_payloads), "n_params": 0}


def well_identity_apply(mdl, z) -> np.ndarray:
    occ = occupancy(z, mdl["anchors"])
    pay = np.asarray(mdl["well_payloads"])
    n_wells = int(pay.shape[0])
    ind = np.zeros((len(occ), n_wells), dtype=np.float64)
    np.put_along_axis(ind, np.asarray(occ), 1.0, axis=1)  # dedup: set, not multiset
    return ind @ pay


def sum_identity_fit(z, y, *, addr_dim: int) -> Dict[str, Any]:
    """⭐ ``yhat = sum_p payload_block(z_p)`` — **0 fitted parameters**.

    The unfitted twin of :func:`~chlu.core.factored_store._sum_linear_fit`, and
    **the objective every organizer in this module is trained on**
    (:func:`~chlu.experiments.exp_null_arms._z_native`), so on the arms that carry
    a latent it reproduces the published ``native`` column exactly. It is scored as
    a reader here so the identity column is complete rather than implicit.
    """
    del z, y
    return {"kind": "sum_identity", "addr_dim": int(addr_dim), "n_params": 0}


def sum_identity_apply(mdl, z) -> np.ndarray:
    return np.asarray(z)[:, :, int(mdl["addr_dim"]):].sum(1)


def fit_readers_plus_identity(z_seen, y_seen, *, anchors=None, well_payloads=None,
                              addr_dim: Optional[int] = None, seed: int = 0,
                              which: Tuple[str, ...] = READERS_PLUS_IDENTITY
                              ) -> Dict[str, Any]:
    """The frozen class **plus** the identity twins, fitted on the SEEN split only.

    ⛔ The identity members ignore ``y_seen`` entirely — pass it anyway so a
    reviewer can see that no branch of this function selects on ``Q_unseen``.
    """
    from chlu.core.factored_store import fit_readers

    base = [w for w in which if w not in READERS_IDENTITY]
    out = fit_readers(z_seen, y_seen, anchors=anchors,
                      well_payloads=well_payloads, seed=seed, which=base)
    if "well_identity" in which and anchors is not None \
            and well_payloads is not None:
        out["well_identity"] = well_identity_fit(z_seen, y_seen, anchors=anchors,
                                                 well_payloads=well_payloads)
    if "sum_identity" in which:
        d = int(addr_dim if addr_dim is not None
                else np.asarray(anchors).shape[1])
        out["sum_identity"] = sum_identity_fit(z_seen, y_seen, addr_dim=d)
    return out


def apply_reader_plus_identity(mdl, z) -> np.ndarray:
    from chlu.core.factored_store import apply_reader

    if mdl["kind"] == "well_identity":
        return well_identity_apply(mdl, z)
    if mdl["kind"] == "sum_identity":
        return sum_identity_apply(mdl, z)
    return apply_reader(mdl, z)


def score_readers_plus_identity(readers, z, y, tol) -> Dict[str, float]:
    return {k: exact_set_accuracy(apply_reader_plus_identity(v, z), y, float(tol))
            for k, v in readers.items()}


def shrinkage_report(w, pred_fitted, pred_identity, y, tol, *,
                     asserted_sets=None, subsets=None) -> Dict[str, Any]:
    """⭐ THE MECHANISM, MEASURED — the two named suspects of C2W7 reconciliation 1.

    Suspect 1, **``diag(W)`` shrinkage**: a least-squares fit of ``yhat = W [f, 1]``
    against an all-or-nothing metric minimises MSE over *all* queries, so it trades
    the queries it already gets right for the many it does not. Reported as
    ``diag_W_mean`` (an unshrunk reader has ``diag(W) == 1``).

    Suspect 2, **the ``tol`` crossing**: on the sub-population whose asserted set is
    **exactly right**, the identity residual is ~0 while the shrunk fitted residual
    can exceed ``tol``. That population's SIZE is what decides whether the pathology
    can move a published number at all — with zero such queries there is nothing to
    destroy, however hard ``W`` is shrunk.

    ``asserted_sets``/``subsets`` are optional; without them only the ``W``
    statistics and the two global accuracies are returned.
    """
    out: Dict[str, Any] = {"tol": float(tol)}
    if w is not None:
        W = np.asarray(w)
        k = min(W.shape[0], W.shape[1])
        dg = np.diag(W[:k, :k])
        off = W[:k, :k] - np.diag(dg)
        out.update({"diag_W": [float(v) for v in dg],
                    "diag_W_mean": float(np.mean(dg)),
                    "diag_W_abs_mean": float(np.mean(np.abs(dg))),
                    "offdiag_rms": float(np.sqrt(np.mean(off ** 2))),
                    "bias_norm": float(np.linalg.norm(W[k:]))
                    if W.shape[0] > k else 0.0})
    y = np.asarray(y)
    rf = np.linalg.norm(np.asarray(pred_fitted) - y, axis=-1)
    ri = np.linalg.norm(np.asarray(pred_identity) - y, axis=-1)
    out.update({"n_queries": int(len(y)),
                "acc_fitted": float((rf <= tol).mean()),
                "acc_identity": float((ri <= tol).mean()),
                "resid_fitted_mean": float(rf.mean()),
                "resid_identity_mean": float(ri.mean())})
    if asserted_sets is not None and subsets is not None:
        ok = np.array([set(np.asarray(s).tolist())
                       == set(np.asarray(a).tolist())
                       for s, a in zip(asserted_sets, subsets, strict=True)])
        out["n_set_exactly_right"] = int(ok.sum())
        out["frac_set_exactly_right"] = float(ok.mean())
        if ok.any():
            out.update({
                "resid_identity_on_exact_mean": float(ri[ok].mean()),
                "resid_fitted_on_exact_mean": float(rf[ok].mean()),
                "identity_within_tol_on_exact": int((ri[ok] <= tol).sum()),
                "fitted_within_tol_on_exact": int((rf[ok] <= tol).sum()),
                # ⭐ the C2W7 crossing, as a boolean: identity keeps them, the
                # fitted reader loses them.
                "c2w7_crossing_fires": bool((ri[ok] <= tol).sum()
                                            > (rf[ok] <= tol).sum())})
        else:
            out["c2w7_crossing_fires"] = False
            out["note"] = ("no query has an exactly-right asserted set ⇒ the "
                           "shrinkage has nothing to destroy at this cell")
    return out


# ==========================================================================
# ⭐⭐ C2W11 (spoke C) — THE ORGANIZER SWAP'S NULL SIDE, on the REPAIRED
# substrate: feature-factored launches, a matched confidence channel on every
# arm, and a LANDSCAPE INSTANTIATION so the static arms have an anytime curve.
# ==========================================================================
#
# ⛔ Everything below is **additive**. No function above it changes behaviour,
# so every C2W5 number this module produced is reproducible bit-for-bit.
#
# ⚠ The three things that MOVED between C2W5 and C2W11, and why each needs new
# code rather than a new flag:
#   (i)   the launch is `R*e_{j_c} + sigma_q xi` from
#         :class:`~chlu.core.feature_launch.FeatureLaunchHead`, not
#         `set_code + o_p`, so ``phi.offsets`` no longer exists on the object an
#         arm launches from and every "de-offset the launch" line is invalid;
#   (ii)  VALUE leg ii scores a graded-novelty read, so every arm must emit a
#         **principled** confidence — an arm without one is a declared NOT-RUN,
#         never a scored "uncalibrated" arm;
#   (iii) VALUE leg iii is the SWAP-DIFFERENCED anytime curve, and the organizer
#         swap hands the null arm the SAME reader class and the SAME k-particle
#         read. ⇒ the null's organization must be **instantiated as a
#         landscape** and read with the identical read. It therefore HAS a
#         curve; it is not a flat line by fiat.
#
# ⛔ Wells, codes, channels and atoms carry integer indices and nothing else
# (``PREREG-TierII.md`` §2.6).


def feature_launch_states(head, cfg: CatTestConfig, indicators: np.ndarray, key,
                          batch: int = 256) -> np.ndarray:
    """``(B, N_a) -> (B, k, dim)`` the C2W11 launch states, **bit-identically**.

    ⛔ This reproduces :func:`~chlu.core.factored_store.multi_particle_read`'s
    launch stage line-for-line — *including its ``fold_in(key, lo)`` chunking*,
    which :func:`chlu.core.feature_launch.launch_points` does **not** use (it
    splits over the whole batch at once, so the two disagree for ``B > batch``).
    The physics arm scores through ``multi_particle_read``; therefore so must
    the launch points every null arm sees, or the arms are not on the same
    queries. Asserted against a zero-step ``multi_particle_read`` in
    ``tests/test_c2w11_nulls.py``.
    """
    ind = jnp.asarray(indicators, dtype=jnp.float32)
    B = int(ind.shape[0])
    out = []

    @eqx.filter_jit
    def launch_all(ind_b, keys_b):
        return jax.vmap(lambda i, k: head.launch(i, k, float(cfg.query_sigma)))(
            ind_b, keys_b)

    for lo in range(0, B, int(batch)):
        hi = min(lo + int(batch), B)
        keys_b = jax.random.split(jax.random.fold_in(key, lo), hi - lo)
        out.append(np.asarray(launch_all(ind[lo:hi], keys_b)))
    return np.concatenate(out, axis=0)


def feature_keys(kind: str, head, cfg: CatTestConfig, indicators: np.ndarray,
                 q0: np.ndarray) -> np.ndarray:
    """N4/N5's key space under feature-factored launches.

    ⚠ ``n4_keys``'s ``"launch_mean"`` subtracts ``phi.offsets`` — a designed
    offset that **does not exist** here. The three registered key spaces are:

    * ``"set_code"``   — the exact ``phi`` set-code (the *noiseless-key* variant,
      flagged as such wherever it wins, exactly as C2W5 flagged it);
    * ``"launch_mean"`` — the mean of the ``k`` launch points (the arm-visible,
      permutation-invariant statistic);
    * ⭐ ``"launch_flat"`` — the ``k`` launch points concatenated. **Strictly
      richer than the mean** and registered *for* the nulls: a mean over
      channels destroys precisely the per-channel structure the C2W11 launch was
      built to create, and scoring the nulls on the destroyed statistic would be
      hobbling them. It is the F3-grade form.
    """
    d = int(cfg.addr_dim)
    if kind == "set_code":
        return np.asarray(head.set_code(jnp.asarray(indicators, jnp.float32)))
    z = np.asarray(q0)[..., :d]
    if kind == "launch_mean":
        return z.mean(1)
    if kind == "launch_flat":
        return z.reshape(len(z), -1)
    raise ValueError(f"unknown C2W11 key space {kind!r}")


# --------------------------------------------------------------------------
# ⭐ THE MATCHED CONFIDENCE CHANNEL — one per arm, per PARTICLE (VALUE leg ii)
# --------------------------------------------------------------------------
# ⛔ Every channel below is the arm's OWN native quantity, not a bolted-on
# detector: N1's is the read objective's own weight on the winning atom, N2's is
# the quantisation error it minimises, N3's is the margin of the decision rule
# it fits, N4's is the neighbour distance its prediction is a function of, and
# N5's is the surprise its gate is a function of. An arm with no such quantity
# is a **declared NOT-RUN for V2**, never a scored "uncalibrated" arm.
#
# Sign convention, fixed once: **higher = more confident / more familiar**, so a
# novelty score is ``-confidence`` and the AUROC below is computed against the
# novel label with that sign already applied.


def n1_confidence(fit: Dict[str, Any], q0: np.ndarray) -> np.ndarray:
    """``(B, k)`` N1's confidence: the winning atom's read-objective weight.

    ``max_i [2 log|A_i| - ||q-c_i||^2 / 2 s_i^2]`` — the log of the very term
    the softmax read normalises, i.e. the read objective's own evidence that
    *some* stored unit explains this launch point. (The set-level residual is
    reported beside it by the harness; per-feature scoring needs a per-particle
    quantity and this is the arm's.)
    """
    c = fit["cfg"]
    B, P = np.asarray(q0).shape[0], int(c.n_particles)
    Q = jnp.asarray(np.asarray(q0).reshape(-1, c.dim), jnp.float32)
    lg = _n1_logits(fit["V"], Q, int(c.addr_dim))
    return np.asarray(jnp.max(lg, axis=-1)).reshape(B, P)


def n2_confidence(fit: Dict[str, Any], q0: np.ndarray) -> np.ndarray:
    """``(B, k)`` N2's confidence: **negative distance-to-codebook** (registered)."""
    C = np.asarray(fit["codebook"][0], dtype=np.float64)
    d = int(fit["cfg"].addr_dim)
    Z = np.asarray(q0)[..., :C.shape[1] if C.shape[1] < d else d]
    d2 = ((Z[:, :, None, :] - C[None, None, :, :Z.shape[-1]]) ** 2).sum(-1)
    return -np.sqrt(d2.min(-1))


def n3_confidence(fit: Dict[str, Any], q0: np.ndarray, kind: str = "evidence"
                  ) -> np.ndarray:
    """``(B, k)`` N3's confidence from its own fitted rule.

    ``score_j = -||z-c_j||^2 / 2 sigma_j^2 + b_j``. Two readings, both emitted:

    * ⭐ ``"evidence"`` (**default, and the arm's F3-grade form**) —
      ``max_j score_j``, the fitted rule's own evidence that *some* cell explains
      this point. It is the exact analogue of N1's max-logit and N2's negative
      distance-to-codebook, so the five arms carry one consistent channel family.
    * ``"margin"`` — ``score_(1) - score_(2)``, the form named in the task.

    ⚠⚠ **Why the default is not the margin, decided by a TEST and not by a
    score.** The margin is a legitimate *assignment*-uncertainty statistic and a
    **broken novelty statistic**: as ``z`` leaves the codebook the two leading
    ``-d^2/2 sigma^2`` terms separate without bound, so the margin *grows* with
    distance and reports maximal confidence exactly where the arm knows least
    (measured on the fixture: margin 6.98 at the fitted points vs **280.99** at
    ``z + 50``; ``tests/test_c2w11_nulls.py::test_confidence_is_higher_where_the_
    arm_was_fitted``). Hobbling the null with a channel that is inverted by
    construction is the same referee attack as hobbling it anywhere else, so the
    arm gets the sound channel and the registered one is reported beside it.
    """
    r = fit["rule"]
    d = int(fit["cfg"].addr_dim)
    Z = np.asarray(q0)[..., :d]
    d2 = ((Z[:, :, None, :] - np.asarray(r["c"])[None, None]) ** 2).sum(-1)
    sc = -d2 / (2.0 * np.asarray(r["sigma"])[None, None, :] ** 2 + 1e-9) \
        + np.asarray(r["b"])[None, None, :]
    if kind == "margin":
        top2 = np.sort(sc, axis=-1)[..., -2:]
        return top2[..., 1] - top2[..., 0]
    if kind != "evidence":
        raise ValueError(f"unknown N3 confidence kind {kind!r}")
    return sc.max(-1)


def n4_confidence(key_tr: np.ndarray, key_q: np.ndarray) -> np.ndarray:
    """N4's confidence: **negative nearest-neighbour distance** in its key space."""
    D = np.linalg.norm(np.asarray(key_q)[:, None, :] - np.asarray(key_tr)[None],
                       axis=-1)
    return -D.min(1)


def n5_confidence(fit: Dict[str, Any], keys: np.ndarray) -> np.ndarray:
    """N5's confidence: ``-||M(k) - M_0(k)||^2`` — the **surprise gate**, at read time.

    ⚠ **Stated explicitly because it is the one place the published rule does not
    hand you a read-time quantity.** Titans' gate is
    ``theta_t = theta * sigmoid(l_t - l_bar)`` with ``l_t = ||M(k_t) - v_t||^2``,
    which needs the **target** ``v_t`` and therefore does not exist at read time.
    The target-free form of the same signal is the deviation the online pass
    actually accumulated at this key: how far the fast weights ``M`` have been
    driven from their meta-learned initialisation ``M_0`` where this query lands.
    It is the integral of the gate along the write stream, evaluated at ``k``.
    ⛔ Declared as a *derived* channel, not as the published gate itself.
    """
    K = jnp.asarray(keys, jnp.float32)
    a = np.asarray(_mlp(fit["M"], K))
    b = np.asarray(_mlp(fit["M0"], K))
    return -((a - b) ** 2).sum(-1)


def novelty_auroc(scores: np.ndarray, novel: np.ndarray) -> float:
    """AUROC of ``-confidence`` (i.e. the novelty score) against the novel label.

    Rank-based (ties averaged), so it is exact rather than trapezoid-approximate.
    Returns ``nan`` when either class is empty — ⛔ a missing class is a NOT-RUN,
    never a 0.5.
    """
    s = -np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(novel).ravel().astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks over ties
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> Dict[str, Any]:
    """Equal-width-bin ECE, with the pieces that make a degenerate ECE legible.

    ⚠ **The degeneracy is registered, not discovered:** when accuracy is ~0
    everywhere, ECE collapses to *mean confidence* and the most under-confident
    arm "wins". ``acc``, ``mean_conf`` and ``degenerate`` are therefore returned
    beside ``ece`` and are reported together, always.
    """
    c = np.clip(np.asarray(conf, dtype=np.float64).ravel(), 0.0, 1.0)
    a = np.asarray(correct).ravel().astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if not m.any():
            continue
        w = float(m.mean())
        gap = abs(float(a[m].mean()) - float(c[m].mean()))
        ece += w * gap
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "acc": float(a[m].mean()), "conf": float(c[m].mean())})
    return {"ece": float(ece), "acc": float(a.mean()),
            "mean_conf": float(c.mean()), "n_bins": int(n_bins), "bins": bins,
            "degenerate": bool(a.mean() <= 0.01),
            "degenerate_note": ("accuracy <= 0.01: ECE is mean confidence and "
                                "the most under-confident arm wins by "
                                "construction. Reported WITH the accuracy.")}


# --------------------------------------------------------------------------
# ⭐⭐ V3's null side — INSTANTIATING A CODEBOOK AS A LANDSCAPE
# --------------------------------------------------------------------------
def instantiate_landscape(cfg: CatTestConfig, centers: np.ndarray,
                          payloads: np.ndarray, *, seed: int = 0,
                          total_atoms: Optional[int] = None,
                          width_frac: Optional[float] = None,
                          place_depth: Optional[float] = None,
                          depth_scale: Optional[np.ndarray] = None
                          ) -> Tuple[FactoredStore, Dict[str, Any]]:
    """Turn a null arm's ``(centers, payloads)`` codebook into a **real store**.

    ⭐ **Why this exists.** The organizer swap changes *who decides where the
    wells go*, not *how they are read* (``PREREG-TierII.md`` §1). So the null's
    answer to "where do the wells go" is instantiated with the **same placing
    write, the same atom budget and the same width law** the physics arm uses,
    and is then read with the identical k-particle anytime read. It is
    :func:`~chlu.core.factored_store.write_store` that does the writing — this
    function chooses nothing the physics arm does not also choose.

    ⚠⚠ **The mirror-image referee attack is closed here, by measurement.**
    *"You gave the null a badly-instantiated landscape"* is the same attack as
    *"you hobbled the competition"*. The three knobs that could hobble it —
    ``total_atoms`` (the atom budget), ``width_frac`` (the co-scaled width) and
    ``place_depth`` (the amplitude) — are therefore **swept and selected on a
    held-out-from-seen split**, exactly like every other null hyperparameter.
    The defaults are the physics arm's own values.

    ``total_atoms`` is the budget in atoms, split evenly over the codebook, so a
    64-code arm gets 6 atoms/code where a 32-code arm gets 12 — **matched total
    capacity**, not matched per-unit capacity.
    """
    C = np.asarray(centers, dtype=np.float64)[:, : int(cfg.addr_dim)]
    Pay = np.asarray(payloads, dtype=np.float64)
    n_codes = int(C.shape[0])
    tot = int(cfg.n_atoms if total_atoms is None else total_atoms)
    a = max(1, int(tot // n_codes))
    c2 = _with(cfg, n_wells=n_codes, atoms_per_well=a,
               # ⛔ the guard is for the CLAIM cell's width; this is a declared
               # instantiation sweep and passes its width explicitly below.
               atom_width_frac_spacing=None, atom_width_selected_frac=None,
               width_guard=False,
               place_depth=float(cfg.place_depth if place_depth is None
                                 else place_depth))
    sp = store_population_spacing(C)
    wf = float(cfg.atom_width_frac_spacing if width_frac is None
               else width_frac) if (cfg.atom_width_frac_spacing is not None
                                    or width_frac is not None) else None
    width = float(cfg.atom_width) if wf is None else float(wf * sp["median_nn"])
    anchors = np.zeros((n_codes, cfg.dim), dtype=np.float64)
    anchors[:, : int(cfg.addr_dim)] = C
    key = jax.random.PRNGKey(int(seed) + 31337)
    k_i, k_w = jax.random.split(key, 2)
    store = FactoredStore(c2, anchors, k_i, atom_width=width)
    if depth_scale is None:
        r = float(cfg.depth_ratio)
        depth_scale = np.where(np.arange(n_codes) % 2 == 0, 1.0, r)
    store, wrep = write_store(store, c2, anchors, Pay, k_w,
                              depth_scale=np.asarray(depth_scale),
                              atom_width=width)
    info = {"n_codes": n_codes, "atoms_per_well": a, "total_atoms": n_codes * a,
            "atom_width": width, "width_frac": wf,
            "codebook_spacing": sp, "place_depth": float(c2.place_depth),
            "store_bytes": int(store.n_bytes()), "write": wrep,
            "cfg": c2}
    return store, info


def anytime_read(store: FactoredStore, head, cfg: CatTestConfig,
                 indicators: np.ndarray, key, *, budget: int,
                 batch: int = 256) -> np.ndarray:
    """The **identical** k-particle anytime read, at a total Verlet budget.

    ⛔ The split rule is quoted from ``FROZEN-INTERFACES-C2W11.json``'s
    ``v3_budget_grid`` and is not re-derived: ``address = round(b/3)``,
    ``read = b - round(b/3)``, ``gamma_address`` then ``gamma_read``, ``dt``
    from the config. The read itself is
    :func:`~chlu.core.factored_store.multi_particle_read` — the physics arm's
    own function, called with the null's store. **A mismatched axis voids VALUE
    leg iii**, so there is exactly one implementation and both arms use it.
    """
    b = int(budget)
    a = int(round(b / 3.0))
    c = _with(cfg, address_steps=a, read_steps=b - a)
    return multi_particle_read(store, head, c, indicators, key, batch=batch)


# --------------------------------------------------------------------------
# ⛔ THE DECODABILITY CEILING, RECOMPUTED ON THE FEATURE-FACTORED LAUNCHES
# --------------------------------------------------------------------------
def feature_decodability_ceiling(head, cfg: CatTestConfig, family: CatFamily,
                                 q0_unseen: Optional[np.ndarray] = None,
                                 chunk: int = 32) -> Dict[str, Any]:
    """How much of ``A(x)`` survives to the launch points **every arm sees**?

    ⛔ **DECLARED OUT-OF-CLASS. Never an arm, never a score.** It enumerates all
    ``C(N_a, F)`` combinations and consults the written payload table, so it is
    matched to nothing. Its job is the one no arm's score can do: separate *"the
    arms are weak"* from *"the launch destroyed the set"*.

    ⚠ :func:`phi_decodability_ceiling`'s as-launched branch **cannot be reused**:
    it de-offsets by ``phi.offsets``, and the C2W11 launch has no offsets. The
    matched filter here compares the observed ``(k, d)`` launch block against
    each candidate combination's own **noiseless launch block** ``R e_{j_c}``,
    in the deflation's deterministic channel order.

    Two conditions, both reported:

    * **noiseless** — nearest set-code over all combinations (unchanged by the
      launch swap; it measures ``phi`` at ``d``, not the launch);
    * **as-launched** — nearest *launch block*, i.e. what a decoder that is
      allowed to know the launch head can still recover from the very points the
      arms consume.

    Plus ``asserted_set_exact``: the fraction of queries whose per-channel
    nearest-code decode is already exactly ``A(x)`` — the K6 statistic, recomputed
    here so the ceiling and the *arms'* ceiling can be read in one table.
    """
    N_a, F, R = int(cfg.n_wells), int(cfg.f_subset), float(cfg.ball_radius)
    d = int(cfg.addr_dim)
    combos = np.array(list(itertools.combinations(range(N_a), F)), dtype=np.int32)
    ind = np.zeros((len(combos), N_a), dtype=np.float32)
    np.put_along_axis(ind, combos, 1.0, axis=1)
    codes_all = np.asarray(head.set_code(jnp.asarray(ind)), dtype=np.float64)
    pay = np.asarray(family.payloads)
    y_tab = ind.astype(np.float64) @ pay

    ind_u = family.indicator(family.unseen, N_a)
    c_exact = np.asarray(head.set_code(jnp.asarray(ind_u)), dtype=np.float64)
    out: Dict[str, Any] = {"n_combos": int(len(combos))}

    # -- noiseless: nearest set-code ---------------------------------------
    hit, dec = [], []
    for lo in range(0, len(c_exact), 64):
        D = ((c_exact[lo:lo + 64, None, :] - codes_all[None]) ** 2).sum(-1)
        j = D.argmin(1)
        dec.append(j)
        hit.append((combos[j] == family.unseen[lo:lo + 64]).all(1))
    j = np.concatenate(dec)
    out["noiseless_combo_exact"] = float(np.concatenate(hit).mean())
    out["noiseless_accuracy"] = exact_set_accuracy(y_tab[j], family.y_unseen,
                                                   family.tol)

    if q0_unseen is None:
        return out

    # -- as-launched: nearest NOISELESS LAUNCH BLOCK -----------------------
    chan = jax.jit(jax.vmap(head.channels))
    picks = np.asarray(chan(jnp.asarray(codes_all, jnp.float32)))  # (n_combos, k)
    ref = R * np.asarray(head.codes, dtype=np.float64)[picks]      # (n_combos,k,d)
    ref_f = ref.reshape(len(combos), -1)
    obs = np.asarray(q0_unseen, dtype=np.float64)[..., :d].reshape(
        len(family.unseen), -1)
    hit, dec = [], []
    for lo in range(0, len(obs), int(chunk)):
        D = ((obs[lo:lo + chunk, None, :] - ref_f[None]) ** 2).sum(-1)
        jj = D.argmin(1)
        dec.append(jj)
        hit.append((combos[jj] == family.unseen[lo:lo + chunk]).all(1))
    jj = np.concatenate(dec)
    out["as_launched_combo_exact"] = float(np.concatenate(hit).mean())
    out["as_launched_accuracy"] = exact_set_accuracy(y_tab[jj], family.y_unseen,
                                                     family.tol)

    # -- the per-channel decode (K6's statistic), for the same table --------
    zc = np.asarray(q0_unseen, dtype=np.float64)[..., :d]
    cd = np.asarray(head.codes, dtype=np.float64) * R
    a_hat = ((zc[:, :, None, :] - cd[None, None]) ** 2).sum(-1).argmin(-1)
    exact = np.array([set(a_hat[i].tolist()) == set(family.unseen[i].tolist())
                      for i in range(len(a_hat))])
    out["asserted_set_exact"] = float(exact.mean())
    out["asserted_set_precision"] = float(np.mean(
        [np.isin(a_hat[i], family.unseen[i]).mean() for i in range(len(a_hat))]))
    return out
