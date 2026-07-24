"""Write objective for the addressable-memory landscape (w20).

**A THIRD training path — do not conflate it with the other two.** ``train.py`` is
Wake-Sleep dynamics fitting (MSE + Lyapunov, Exp A/B); ``train_generative.py`` is
pure EBM PCD (persistent buffer, Exp C). This module is neither: it *writes K
items into a landscape* by making each target state a local minimum of ``V``,
with a barrier between neighbouring items. There is no data, no replay buffer and
no rollout in the loss.

Why a static (no-BPTT) objective. The retrieval loop is
``query -> [gamma>0 relaxation] -> address -> [rollout] -> read``. Training
through the damped relaxation is possible in theta (unlike in the *address*,
which Prop 5 shows is gradient-opaque), but it costs a full BPTT per step and
buys nothing here: what makes retrieval work is precisely that ``z_i`` is an
attracting minimum, which is exactly what this loss states. Keeping the write
objective free of the rollout also keeps the w20 question clean — any failure is
a property of the learned landscape, not of an optimizer fighting a chaotic
horizon.

The loss (per item ``i``, target ``z_i = (c_i, a_i)``):

.. code-block:: text

    L_grad  = |grad V(z_i)|^2                          # stationary
    L_min   = mean_delta relu(V(z_i) - V(z_i+delta) + margin)   # and a MINIMUM
    L_bar   = sum_{i<j} relu(barrier + max(V(z_i),V(z_j)) - V(midpoint_ij))^2
    L_reg   = weight decay on the learned parameters

``delta`` is sampled anisotropically: ``sigma_addr`` on the address plane and
``sigma_pay`` on the payload channel, and the q2=0 **query manifold** is included
explicitly in the perturbation set, because that is the point retrieval actually
starts from (a landscape that is a minimum only for infinitesimal delta will not
pull a q2=0 query up to a_i).
"""

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


def trainable_filter(V: eqx.Module):
    """Filter spec selecting ONLY ``V.learned`` — the designed part stays fixed.

    Returns ``None`` when there is nothing to train (the ``designed`` rung), so
    callers can skip training entirely rather than optimizing an empty tree.
    """
    if getattr(V, "learned", None) is None:
        return None
    # NOTE: the spec must consist of BOOLEANS, not of the arrays themselves —
    # ``eqx.filter(..., is_inexact_array)`` returns the arrays and equinox
    # rejects it here ("filter_spec must consist of booleans and callables").
    spec = jax.tree_util.tree_map(lambda _: False, V)
    return eqx.tree_at(
        lambda t: t.learned,
        spec,
        replace=jax.tree_util.tree_map(eqx.is_inexact_array, V.learned),
    )


def _atom_owner(n_atoms: int, n_groups: int):
    """Owner group index per atom row, matching ``AtomDictionaryPotential.group_rows``.

    Blocks are contiguous with boundaries ``round(g*n/G)``; this is the vectorized
    inverse of that partition (static, so it is a trace-time constant).
    """
    import numpy as _np

    bounds = _np.array(
        [round(g * n_atoms / n_groups) for g in range(n_groups + 1)], dtype=int
    )
    rows = _np.arange(n_atoms)
    return _np.clip(_np.searchsorted(bounds, rows, side="right") - 1, 0, n_groups - 1)


def atom_crowding_penalty(V: eqx.Module, sites: jnp.ndarray, d_safe: float):
    """⭐ w24 CROWDING term: penalize an atom that lands within ``d_safe`` of a
    site it does **not** own, weighted by that atom's depth.

    ``L_crowd = mean_a  A_a * sum_{j != owner(a)} relu(d_safe - |c_a - z_j|)^2``

    Why this exists (w24 ``write-ceiling-break``): the w23 tell is that the write
    loss reaches ~0 while retrieval already fails at ``K > K_ceiling`` — i.e. the
    objective is **blind to crowding**. The standard terms only look at the stored
    points and pairwise midpoints; nothing in them notices that item ``i``'s own
    atoms have drifted into item ``j``'s basin (which is exactly what ruins the
    relaxation that retrieval runs). This is the write-time analogue of the MVC-0
    spacing gate (:func:`chlu.core.admission.admit_site`, ``D_SAFE_MULT``): the
    gate refuses a *site* too close to a stored one; this term refuses a *write*
    whose support encroaches on one.

    Returns ``0.0`` (no-op, no gradient) unless ``V`` carries a grouped atom
    dictionary whose group count equals the number of sites — for any other
    potential family the notion of "the atoms this item owns" is undefined.
    """
    atoms = getattr(V, "learned", None)
    if atoms is None:
        atoms = V
    if not all(hasattr(atoms, a) for a in ("centers", "amp", "n_groups")):
        return jnp.asarray(0.0)
    K = int(sites.shape[0])
    n = int(atoms.centers.shape[0])
    if int(atoms.n_groups) != K or K < 2:
        return jnp.asarray(0.0)
    owner = jnp.asarray(_atom_owner(n, K))
    foreign = (owner[:, None] != jnp.arange(K)[None, :]).astype(jnp.float32)
    c, s = atoms.centers, jnp.asarray(sites, dtype=atoms.centers.dtype)
    d2 = (
        jnp.sum(c**2, axis=-1)[:, None]
        + jnp.sum(s**2, axis=-1)[None, :]
        - 2.0 * (c @ s.T)
    )
    dist = jnp.sqrt(jnp.maximum(d2, 0.0) + 1e-12)
    pen = jax.nn.relu(d_safe - dist) ** 2 * foreign
    depth = atoms.amp**2
    return jnp.mean(depth * jnp.sum(pen, axis=-1))


def write_loss(
    V: eqx.Module,
    targets: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_perturb: int = 32,
    sigma_addr: float = 0.25,
    sigma_pay: float = 0.6,
    margin: float = 0.15,
    barrier: float = 0.2,
    w_grad: float = 1.0,
    w_min: float = 1.0,
    w_barrier: float = 1.0,
    payload_index: int = 2,
    min_agg: str = "mean",
    barrier_pairs: str = "all",
    item_agg: str = "mean",
    crowd_weight: float = 0.0,
    crowd_d_safe: float = 0.0,
    crowd_targets: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Scalar write loss (see module docstring). ``targets`` is (K, dim).

    ``payload_index`` is the coordinate that carries the stored value (the read-out
    channel launched at zero by the anti-decoration guard). It defaults to ``2``,
    the ring/w20 convention (address plane ``q0,q1`` + payload ``q2``); a
    ``d``-dimensional address ball uses ``payload_index = d`` (address ``q[:d]`` +
    payload ``q[d]``). All coordinates get ``sigma_addr`` jitter except the payload
    channel, which gets ``sigma_pay`` and is pinned to 0 on the query manifold.

    **w24 levers (all default to the w20-w23 behaviour bit-for-bit).**

    ``min_agg``
        ``"mean"`` (default) averages the minimum-violation over the perturbation
        set; ``"max"`` takes the WORST direction per item. Averaging is the
        crowding blindness in miniature: the few perturbation directions that
        point at a crowded neighbour are outvoted by the many that do not, so the
        loss can reach ~0 with the item's basin already broken on one side.
    ``barrier_pairs``
        ``"all"`` (default) averages the barrier over all ``K(K-1)/2`` pairs, so a
        violated pair carries weight ``~1/K^2`` while the number of *violated*
        pairs stays ``O(K)``: the crowding signal is DILUTED as ``1/K``. ``"nn"``
        keeps only each item's NEAREST other site (``K`` pairs), which both
        un-dilutes it and drops the cost from ``O(K^2)`` to ``O(K)``.
    ``item_agg``
        ``"mean"`` (default) or ``"sum"`` over items: with ``"sum"`` the gradient
        each item contributes does not shrink as ``K`` grows (the scale-invariance
        arm; note Adam is invariant to a *global* rescale, so this only bites
        through decoupled weight decay and the relative weight of the terms).
    ``crowd_weight`` / ``crowd_d_safe`` / ``crowd_targets``
        Add ``crowd_weight * atom_crowding_penalty(V, crowd_targets or targets,
        crowd_d_safe)``. ``crowd_targets`` is the FULL stored set, which matters
        for a sequential write where ``targets`` is a single item.
    """
    K, dim = targets.shape
    gradV = jax.grad(lambda q: V(q))
    red = jnp.sum if item_agg == "sum" else jnp.mean

    # --- stationarity at each target ---
    l_grad = red(jax.vmap(lambda z: jnp.sum(gradV(z) ** 2))(targets))

    # --- each target must be a MINIMUM over a finite neighbourhood ---
    k_p, k_q = jax.random.split(key, 2)
    scale = jnp.full((dim,), sigma_addr).at[payload_index].set(sigma_pay)
    delta = jax.random.normal(k_p, (K, n_perturb, dim)) * scale
    # ...and explicitly at the query manifold (payload channel launched at zero),
    # jittered on the address plane exactly as the queries will be.
    q_jit = jax.random.normal(k_q, (K, n_perturb, dim)) * scale
    q_jit = q_jit.at[:, :, payload_index].set(0.0)
    query_pts = targets[:, None, :] + q_jit
    query_pts = query_pts.at[:, :, payload_index].set(0.0)

    pts = jnp.concatenate([targets[:, None, :] + delta, query_pts], axis=1)
    v_t = jax.vmap(V)(targets)  # (K,)
    v_p = jax.vmap(jax.vmap(V))(pts)  # (K, 2*n_perturb)
    viol = jax.nn.relu(v_t[:, None] - v_p + margin)  # (K, 2*n_perturb)
    per_item_min = jnp.max(viol, axis=1) if min_agg == "max" else jnp.mean(viol, axis=1)
    l_min = red(per_item_min)

    # --- a barrier between neighbouring items (otherwise the wells merge) ---
    l_bar = jnp.asarray(0.0)
    if barrier_pairs == "all" and crowd_targets is None:
        # unchanged w20 path (kept bit-identical under the default item_agg:
        # every prior result used it)
        if K > 1:
            i, j = jnp.triu_indices(K, k=1)
            mids = 0.5 * (targets[i] + targets[j])
            v_m = jax.vmap(V)(mids)
            v_hi = jnp.maximum(v_t[i], v_t[j])
            l_bar = jnp.mean(jax.nn.relu(barrier + v_hi - v_m) ** 2)
            if item_agg == "sum":
                # a per-item aggregation must scale EVERY term the same way, or
                # the scale-invariant arm silently down-weights the barrier by K
                l_bar = l_bar * K
    else:
        others = targets if crowd_targets is None else jnp.asarray(crowd_targets)
        if others.shape[0] > 1 or crowd_targets is not None:
            d2 = jnp.sum((targets[:, None, :] - others[None, :, :]) ** 2, axis=-1)
            self_pair = d2 < 1e-12
            if barrier_pairs == "nn":
                jj = jnp.argmin(jnp.where(self_pair, jnp.inf, d2), axis=1)  # (K,)
                nb = others[jj]
                mids = 0.5 * (targets + nb)
                v_m = jax.vmap(V)(mids)
                v_o = jax.vmap(V)(nb)
                v_hi = jnp.maximum(v_t, v_o)
                l_bar = red(jax.nn.relu(barrier + v_hi - v_m) ** 2)
            else:
                mids = 0.5 * (targets[:, None, :] + others[None, :, :])
                v_m = jax.vmap(jax.vmap(V))(mids)
                v_o = jax.vmap(V)(others)
                v_hi = jnp.maximum(v_t[:, None], v_o[None, :])
                pen = jax.nn.relu(barrier + v_hi - v_m) ** 2
                pen = jnp.where(self_pair, 0.0, pen)
                l_bar = red(jnp.mean(pen, axis=1))

    total = w_grad * l_grad + w_min * l_min + w_barrier * l_bar
    if crowd_weight > 0.0 and crowd_d_safe > 0.0:
        sites = targets if crowd_targets is None else jnp.asarray(crowd_targets)
        total = total + crowd_weight * atom_crowding_penalty(V, sites, crowd_d_safe)
    return total


def train_memory_landscape(
    V: eqx.Module,
    targets: jnp.ndarray,
    key: jax.random.PRNGKey,
    steps: int = 400,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    loss_kwargs: Optional[dict] = None,
    callback: Optional[Callable] = None,
    update_mask_fn: Optional[Callable] = None,
):
    """Write ``targets`` into ``V`` by Adam on the learned part only.

    Args:
        V: a ``DesignFreedomPotential`` (or any module with a ``.learned``
            subtree; ``.learned is None`` => returned unchanged).
        targets: (K, dim) target states ``z_i = (c_i, a_i)``.
        key: PRNG key, split per step (never reused).
        steps, lr, weight_decay: optimizer settings.
        loss_kwargs: forwarded to :func:`write_loss`.
        update_mask_fn: optional ``updates -> updates`` map applied to the
            optimizer's update tree **before** it is added to the parameters.
            This is how a write is made **local in parameter space** (w21): with
            :func:`chlu.core.memory_potentials.atom_write_mask_fn` every atom
            outside the written item's block comes out of the write
            bit-identical. It must mask the *updates*, not the gradients,
            because ``optax.adamw``'s decoupled weight decay is applied to the
            update and would otherwise still shrink the frozen parameters.

    Returns:
        ``(V_trained, history)`` where history is a list of scalar losses.
    """
    spec = trainable_filter(V)
    if spec is None:
        return V, []

    loss_kwargs = dict(loss_kwargs or {})
    params, static = eqx.partition(V, spec)
    opt = optax.adamw(lr, weight_decay=weight_decay)
    state = opt.init(params)

    @eqx.filter_jit
    def step_fn(params, static, state, k):
        def loss_fn(p):
            return write_loss(eqx.combine(p, static), targets, k, **loss_kwargs)

        val, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, state = opt.update(grads, state, params)
        if update_mask_fn is not None:
            updates = update_mask_fn(updates)
        params = eqx.apply_updates(params, updates)
        return params, state, val

    history = []
    for i in range(steps):
        key, k = jax.random.split(key)
        params, state, val = step_fn(params, static, state, k)
        history.append(float(val))
        if callback is not None:
            callback(i, float(val))
    return eqx.combine(params, static), history
