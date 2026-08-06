"""⭐ **C2W8 PASS 2, ARM B** — a standard MLP-class head on ``phi`` that **emits
the well parameters** (center, width, depth, payload), with the well's functional
form left designed exactly as it is now.

**What this replaces.** The shipped write is 300 Adam steps of
:func:`~chlu.training.train_memory.train_memory_landscape` per item. Here the
write is **one forward pass**: ``h_theta(phi(x), a) -> (c, s, D, alpha)``, and
the item's atom block is *set* to that Gaussian well. The atoms, and therefore
the landscape, are the same object
(:class:`~chlu.core.memory_potentials.AtomDictionaryPotential`:
``V = confine*|q|^2 - sum_j A_j exp(-|q - c_j|^2 / 2 s_j^2)``); only the
*mechanism that chooses the parameters* changes.

⛔⛔ **THE DECLARED TRAP (PREREG-C2W8-PASS2 §2, ERRATA §1 Q4).** An emission head
that produces **one private well per item** restores explicit per-item store
parameters and is **laundered by construction**. That is the shipped
configuration here, and it is labelled ``NO_TIER_II_CLAIM`` by
:func:`emission_ledger`, mechanically, in every artifact. The configuration that
could ever carry a tier-ii claim is **per-item COEFFICIENTS OVER A SHARED WELL
VOCABULARY**, which this module *specifies* (:class:`WellVocabulary`,
:func:`compose_wells`) and does **not** build: the private-well case is exactly
the degenerate ``vocabulary_size = n_items, coefficients = one-hot`` special
case, and ``tests/test_emission_head.py`` asserts the two paths agree bitwise.

⛔⛔ **THE HEAD'S BINDING PROHIBITION (PREREG-C2W8-PASS2 §0).** The emitted center
is a **learned, continuous** function of ``phi`` and is **never** pinned,
snapped, or regularized to ``phi(item)``. What stops it from being a pinned
anchor is stated precisely, because emitting a center *is* nearly the pinning
operation:

* the objective contains **no term of the form** ``|c - phi|^2`` (or any other
  strictly-positive function of that displacement). The only term that mentions
  ``phi`` at all is :func:`reach_penalty`, a **hinge on reachability with a free
  width**;
* the hinge is **exactly zero — bitwise, with an exactly zero gradient in the
  center** — as soon as the launch point ``(phi, 0)`` lies inside ``rho * s`` of
  the well. A pinning term has a *unique* minimizer ``c = phi``, is never zero,
  and pulls at every step;
* the hinge is satisfiable **without moving the center at all**, by growing the
  emitted width ``s``. Its zero set is a manifold, not a point;
* ``|c - phi|`` is consequently a **measured, reported, non-zero** quantity of
  the arm, never a target.

Both properties are pytest-asserted (``test_reach_penalty_is_not_a_pin_*``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chlu.core.memory_potentials import AtomDictionaryPotential

#: The label a private-well arm carries in every artifact it touches (K8).
NO_TIER_II_CLAIM = "NO_TIER_II_CLAIM"


# ==========================================================================
# the emission interface: well parameters, and the vocabulary they may share
# ==========================================================================
class WellParams(eqx.Module):
    """A set of ``n`` designed wells, in the store's own parameterisation.

    ``centers`` is ``(n, dim)`` in the FULL store space (address block followed
    by the payload block), so a well is one point, not two half-objects.
    """

    centers: jnp.ndarray  # (n, dim)  -- address block + payload block
    log_widths: jnp.ndarray  # (n,)
    log_depths: jnp.ndarray  # (n,)

    @property
    def n(self) -> int:
        return int(self.centers.shape[0])

    @property
    def widths(self) -> jnp.ndarray:
        return jnp.exp(self.log_widths)

    @property
    def depths(self) -> jnp.ndarray:
        return jnp.exp(self.log_depths)


class WellVocabulary(eqx.Module):
    """⭐ **The interface the factored store needs, specified but NOT built.**

    A store holds ``V`` well templates; an item is written as a **coefficient
    vector** ``w_i in R^V`` over them, and its contribution to the landscape is
    ``sum_v w_iv * well(theta_v)``.

    * ``vocabulary_shared = True`` (``V < n_items``, coefficients dense) is the
      configuration that can carry a tier-ii claim. **It is not built this
      wave** (PREREG-C2W8-PASS2 §6) and nothing here trains it.
    * ``vocabulary_shared = False`` is the **degenerate special case**
      ``V = n_items``, ``w_i = one-hot(i)`` — i.e. exactly the private-well
      configuration this arm ships, reached through the *same* interface. That
      equivalence is asserted bitwise in ``tests/test_emission_head.py``, which
      is what makes *"the arm does not foreclose the factored store"* a
      mechanical statement rather than a promise.
    """

    templates: WellParams
    shared: bool = eqx.field(static=True)

    @property
    def size(self) -> int:
        return int(self.templates.n)


def compose_wells(vocab: WellVocabulary, coefficients: jnp.ndarray) -> WellParams:
    """``(vocabulary, coefficients) -> per-item wells`` — the factored read of
    the emission interface.

    ``coefficients`` is ``(n_items, V)``. With ``V = n_items`` and a one-hot
    coefficient matrix this returns the templates unchanged (bitwise: ``0 * x``
    is ``0`` and ``x + 0`` is ``x`` in IEEE-754), which is the degenerate
    private-well case.

    ⛔ This function is the **interface**, not the factored store: nothing in
    this module trains a shared vocabulary, and the shipped arm only ever calls
    it through :func:`private_vocabulary`.
    """
    w = jnp.asarray(coefficients, dtype=vocab.templates.centers.dtype)
    return WellParams(
        centers=w @ vocab.templates.centers,
        log_widths=w @ vocab.templates.log_widths,
        log_depths=w @ vocab.templates.log_depths,
    )


def private_vocabulary(params: WellParams) -> Tuple[WellVocabulary, jnp.ndarray]:
    """The degenerate embedding of ``n`` private wells into the shared interface.

    Returns ``(vocabulary of size n, identity coefficients)``, so
    ``compose_wells(*private_vocabulary(p))`` is ``p``.
    """
    return (WellVocabulary(templates=params, shared=False),
            jnp.eye(params.n, dtype=params.centers.dtype))


# ==========================================================================
# the head
# ==========================================================================
class EmissionHead(eqx.Module):
    """MLP-class head ``(phi, a) -> (center, width, depth, payload)``.

    **Inputs.** ``phi`` is the read-in's address-space vector for the item (the
    same object the shipped write receives as ``address``) and ``a`` is the
    item's payload — *the content being stored*.

    ⚠ **Why ``a`` is an INPUT and not an output of ``phi`` alone.** A head that
    emitted the payload coordinate from ``phi`` by itself would make the read a
    ``phi -> value`` feed-forward classifier: the store would "retrieve" content
    it was never given, which is engineered separability wearing a memory
    costume (intervention §8 prohibition 2). Here the head *encodes* the content
    it is handed; the emitted payload coordinate is ``a`` plus a **bounded**
    learned correction (``|delta| <= payload_delta_max``, default strictly below
    the read's own ``payload_tol``), so it is emitted, continuous, learned — and
    cannot invent a value.

    **Outputs.** The center is raw MLP output: there is deliberately **no
    residual/identity path from ``phi`` to the center**, because such a path is
    a soft pin. Width and depth are squashed into declared bands (a designed
    choice: an unbounded emitted depth makes the landscape scale-free and an
    unbounded width dissolves every basin).
    """

    mlp: eqx.nn.MLP
    addr_dim: int = eqx.field(static=True)
    payload_dim: int = eqx.field(static=True)
    width_min: float = eqx.field(static=True)
    width_max: float = eqx.field(static=True)
    depth_min: float = eqx.field(static=True)
    depth_max: float = eqx.field(static=True)
    payload_delta_max: float = eqx.field(static=True)

    def __init__(self, addr_dim: int, payload_dim: int, key, *,
                 hidden: int = 64, layers: int = 2,
                 width_min: float = 0.15, width_max: float = 0.80,
                 depth_min: float = 0.05, depth_max: float = 3.0,
                 payload_delta_max: float = 0.05):
        self.addr_dim = int(addr_dim)
        self.payload_dim = int(payload_dim)
        self.width_min = float(width_min)
        self.width_max = float(width_max)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.payload_delta_max = float(payload_delta_max)
        self.mlp = eqx.nn.MLP(
            in_size=int(addr_dim + payload_dim),
            out_size=int(addr_dim + 2 + payload_dim),
            width_size=int(hidden), depth=int(layers),
            activation=jax.nn.tanh, key=key,
        )

    # -- emission ---------------------------------------------------------
    def __call__(self, phi: jnp.ndarray, payload: jnp.ndarray):
        """One item: ``-> (center_addr, log_width, log_depth, payload_coord)``."""
        phi = jnp.asarray(phi, dtype=jnp.float32).reshape(-1)[: self.addr_dim]
        a = jnp.asarray(payload, dtype=jnp.float32).reshape(-1)[: self.payload_dim]
        out = self.mlp(jnp.concatenate([phi, a]))
        d, m = self.addr_dim, self.payload_dim
        center = out[:d]
        w = self.width_min + (self.width_max - self.width_min) * jax.nn.sigmoid(out[d])
        dep = self.depth_min + (self.depth_max - self.depth_min) * jax.nn.sigmoid(out[d + 1])
        pay = a + self.payload_delta_max * jnp.tanh(out[d + 2: d + 2 + m])
        return center, jnp.log(w), jnp.log(dep), pay

    def emit(self, phi: jnp.ndarray, payloads: jnp.ndarray) -> WellParams:
        """Batched emission -> :class:`WellParams` in FULL store coordinates."""
        phi = jnp.atleast_2d(jnp.asarray(phi, dtype=jnp.float32))
        a = jnp.asarray(payloads, dtype=jnp.float32).reshape(phi.shape[0], -1)
        c, lw, ld, pay = jax.vmap(self.__call__)(phi, a)
        return WellParams(centers=jnp.concatenate([c, pay], axis=-1),
                          log_widths=lw, log_depths=ld)

    def emit_center(self, phi, payload) -> np.ndarray:
        """The **address block only** — the derived address the controller places.

        ⚠ This is the write's placement decision and it is where the arm is most
        at risk of becoming D2a. See the module docstring: nothing pins it to
        ``phi``; the coupling is a hinge with a free width, and ``|c - phi|`` is
        reported, not targeted.
        """
        c, _, _, _ = self(jnp.asarray(phi), jnp.asarray(payload))
        return np.asarray(c, dtype=float)

    # -- the ledger's own numbers ------------------------------------------
    def n_params(self) -> int:
        return int(sum(int(np.asarray(x).size) for x in
                       jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_inexact_array))))

    def n_bytes(self) -> int:
        """float32 parameter bytes — this arm's sharpest byte-ledger column."""
        return int(self.n_params() * 4)


# ==========================================================================
# the designed well form, built from emitted parameters
# ==========================================================================
def emitted_potential(params: WellParams, *, dim: int, confine: float
                      ) -> AtomDictionaryPotential:
    """The emitted wells AS THE SHIPPED POTENTIAL — one atom per well.

    ⭐ Constructed by :func:`equinox.tree_at` on a real
    :class:`~chlu.core.memory_potentials.AtomDictionaryPotential`, never
    re-implemented, so *"the well's functional form is left designed exactly as
    it is now"* is true by construction rather than by inspection.
    """
    n = params.n
    base = AtomDictionaryPotential(dim=int(dim), n_atoms=int(n),
                                   key=jax.random.PRNGKey(0), confine=float(confine),
                                   n_groups=int(n))
    return eqx.tree_at(
        lambda t: [t.centers, t.log_width, t.amp], base,
        replace=[params.centers.astype(base.centers.dtype),
                 params.log_widths.astype(base.log_width.dtype),
                 jnp.sqrt(params.depths).astype(base.amp.dtype)],
    )


def apply_emitted_well(store, slot: int, center, width: float, depth: float):
    """Set slot ``slot``'s atom block to one designed well — the WRITE.

    Every atom the slot owns is placed at the site with amplitude
    ``sqrt(depth / n_rows)``, so the block's atom-sum at the site is exactly
    ``depth`` (the census's ``own_atom_depth``). This is the same atom surgery
    :func:`chlu.core.well_lifecycle.plant_item` performs — that module is
    read-only this wave, so the arithmetic is repeated here rather than imported.

    Returns the new store (Equinox modules are immutable).
    """
    rows = np.asarray(store.group_rows(int(slot)), dtype=bool)
    n = int(rows.sum())
    if n == 0:
        raise RuntimeError(f"apply_emitted_well: slot {slot} owns no atom rows")
    idx = jnp.asarray(np.nonzero(rows)[0])
    z = np.asarray(center, dtype=float).reshape(-1)[: store.dim]
    atoms = store.V.learned
    c = jnp.asarray(np.repeat(z[None, :], n, axis=0), dtype=atoms.centers.dtype)
    a = jnp.full((n,), float(np.sqrt(max(float(depth), 0.0) / n)), dtype=atoms.amp.dtype)
    w = jnp.full((n,), float(np.log(max(float(width), 1e-6))), dtype=atoms.log_width.dtype)
    V = eqx.tree_at(
        lambda t: [t.learned.centers, t.learned.amp, t.learned.log_width],
        store.V,
        replace=[atoms.centers.at[idx].set(c), atoms.amp.at[idx].set(a),
                 atoms.log_width.at[idx].set(w)],
    )
    return eqx.tree_at(lambda s: s.V, store, V)


# ==========================================================================
# the DESIGNED write objective the head is trained through (charter §A28.1)
# ==========================================================================
def reach_penalty(params: WellParams, phi: jnp.ndarray, *, addr_dim: int,
                  rho: float = 2.0) -> jnp.ndarray:
    """⭐ The **reach hinge** — the only term that mentions ``phi``.

    ``L = mean_i relu(|q_launch,i - z_i| - rho * s_i)^2`` with
    ``q_launch = (phi_i, 0)``, i.e. exactly the point the read launches from
    (:meth:`chlu.core.clu_system.CluSystem.read` zeroes the payload channels).
    This is monitor #11's saddle-reach criterion moved to write time.

    ⛔ **It is not a pin, and this is checkable rather than asserted:** the term
    is bitwise ``0`` with a bitwise ``0`` gradient in the center whenever the
    launch already lies inside ``rho * s``, and it can be driven to zero by
    growing ``s`` alone at a fixed center. A pin (``|c - phi|^2``) has neither
    property. Both are pytest-asserted.
    """
    z = params.centers
    q = jnp.zeros_like(z).at[:, :int(addr_dim)].set(
        jnp.asarray(phi, dtype=z.dtype)[:, :int(addr_dim)])
    d = jnp.linalg.norm(q - z, axis=-1)
    return jnp.mean(jax.nn.relu(d - float(rho) * params.widths) ** 2)


def emission_write_loss(head: EmissionHead, phi: jnp.ndarray, payloads: jnp.ndarray,
                        key, *, dim: int, confine: float, addr_dim: int,
                        reach_weight: float = 1.0, reach_rho: float = 2.0,
                        loss_kwargs: Optional[Dict[str, Any]] = None) -> jnp.ndarray:
    """The head's training objective = **the shipped write objective** + reach.

    ⭐ ``chlu.training.train_memory.write_loss`` is imported and called
    unmodified on the landscape the head emits, with the same ``loss_kwargs``
    the streaming write uses (``barrier_pairs="nn"``, ``crowd_targets`` = the
    batch). The organization pressure — separation, minimality, the explicit
    ``payload = 0`` query manifold — is therefore *the designed write objective*,
    not a bespoke surrogate; this is charter §A28.1's *"routed through the write
    objective"*, and it is what the arm's ledger declares as a designed
    mechanism.
    """
    from chlu.training.train_memory import write_loss

    params = head.emit(phi, payloads)
    V = emitted_potential(params, dim=int(dim), confine=float(confine))
    kw = dict(payload_index=int(addr_dim), barrier_pairs="nn")
    kw.update(dict(loss_kwargs or {}))
    total = write_loss(V, params.centers, key, crowd_targets=params.centers, **kw)
    if reach_weight > 0.0:
        total = total + float(reach_weight) * reach_penalty(
            params, phi, addr_dim=int(addr_dim), rho=float(reach_rho))
    return total


def pretrain_emission_head(head: EmissionHead, phi_pool, payload_pool, key, *,
                           dim: int, confine: float, addr_dim: int,
                           steps: int = 400, batch: int = 16, lr: float = 3e-3,
                           weight_decay: float = 1e-4, reach_weight: float = 1.0,
                           reach_rho: float = 2.0,
                           loss_kwargs: Optional[Dict[str, Any]] = None,
                           callback: Optional[Callable] = None
                           ) -> Tuple[EmissionHead, Dict[str, Any]]:
    """Amortise the write: Adam on the head through :func:`emission_write_loss`.

    ⚠ **This is the arm's amortised cost and it is ledgered, not hidden.** The
    per-item write becomes one forward pass, but the head is paid for once, here.
    The report carries ``steps x batch`` write-objective evaluations against the
    ``300 x n_items`` Adam steps the shipped write spends.

    ⚠ ``phi_pool`` must come from the ``phi`` fit pool (task-1 only on this rig),
    so the head sees **no stream item and no test item** — the same discipline
    ``phi`` itself is held to. ``payload_pool`` is drawn synthetically over the
    payload range: the head is never shown a (phi, label) pair, so no label can
    leak into placement.
    """
    opt = optax.adamw(learning_rate=float(lr), weight_decay=float(weight_decay))
    params, static = eqx.partition(head, eqx.is_inexact_array)
    state = opt.init(params)
    P = jnp.asarray(np.asarray(phi_pool, dtype=np.float32))
    A = jnp.asarray(np.asarray(payload_pool, dtype=np.float32).reshape(P.shape[0], -1))
    n = int(P.shape[0])
    b = int(min(max(batch, 2), n))

    @eqx.filter_jit
    def step(params, state, idx, k):
        def loss_fn(p):
            h = eqx.combine(p, static)
            return emission_write_loss(h, P[idx], A[idx], k, dim=dim, confine=confine,
                                       addr_dim=addr_dim, reach_weight=reach_weight,
                                       reach_rho=reach_rho, loss_kwargs=loss_kwargs)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        upd, state = opt.update(grads, state, params)
        return eqx.apply_updates(params, upd), state, loss

    hist = []
    for t in range(int(steps)):
        key, k_b, k_l = jax.random.split(key, 3)
        idx = jax.random.choice(k_b, n, shape=(b,), replace=False)
        params, state, loss = step(params, state, idx, k_l)
        hist.append(float(loss))
        if callback is not None:
            callback(t, float(loss))
    trained = eqx.combine(params, static)
    return trained, {"loss_first": hist[0] if hist else float("nan"),
                     "loss_last": hist[-1] if hist else float("nan"),
                     "steps": int(steps), "batch": int(b), "pool_size": n,
                     "objective_evals": int(steps) * int(b),
                     "history": hist}


# ==========================================================================
# K8 — the trap, made machine-checkable
# ==========================================================================
def emission_ledger(head: Optional[EmissionHead], *, n_items: int,
                    vocabulary_shared: bool = False,
                    vocabulary_size: Optional[int] = None,
                    wells_per_item: int = 1) -> Dict[str, Any]:
    """⭐ **K8**: the declaration every arm-B artifact must carry.

    ``wells_per_item`` and ``vocabulary_shared`` are **always present** (asserted
    in ``tests/test_emission_head.py``), and a private-well arm is labelled
    :data:`NO_TIER_II_CLAIM` **here**, mechanically, so the label travels with
    every number computed from the same dict.
    """
    private = (not bool(vocabulary_shared))
    return {
        "wells_per_item": int(wells_per_item),
        "vocabulary_shared": bool(vocabulary_shared),
        "vocabulary_size": int(n_items if vocabulary_size is None else vocabulary_size),
        "coefficients": ("one-hot (degenerate)" if private else "dense"),
        "n_items": int(n_items),
        "tier_ii_status": (NO_TIER_II_CLAIM if private else "not-barred-by-K8"),
        "why": (
            "private wells per item = explicit per-item store parameters = "
            "laundered by construction (intervention §8 prohibition 2); the "
            "label travels with every number from this arm"
            if private else
            "coefficients over a shared well vocabulary (NOT built in C2W8)"
        ),
        "head_param_count": (0 if head is None else int(head.n_params())),
        "head_bytes": (0 if head is None else int(head.n_bytes())),
        "factored_store_built": False,
    }


__all__ = [
    "NO_TIER_II_CLAIM", "WellParams", "WellVocabulary", "compose_wells",
    "private_vocabulary", "EmissionHead", "emitted_potential",
    "apply_emitted_well", "reach_penalty", "emission_write_loss",
    "pretrain_emission_head", "emission_ledger",
]
