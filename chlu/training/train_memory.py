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
) -> jnp.ndarray:
    """Scalar write loss (see module docstring). ``targets`` is (K, dim).

    ``payload_index`` is the coordinate that carries the stored value (the read-out
    channel launched at zero by the anti-decoration guard). It defaults to ``2``,
    the ring/w20 convention (address plane ``q0,q1`` + payload ``q2``); a
    ``d``-dimensional address ball uses ``payload_index = d`` (address ``q[:d]`` +
    payload ``q[d]``). All coordinates get ``sigma_addr`` jitter except the payload
    channel, which gets ``sigma_pay`` and is pinned to 0 on the query manifold.
    """
    K, dim = targets.shape
    gradV = jax.grad(lambda q: V(q))

    # --- stationarity at each target ---
    l_grad = jnp.mean(jax.vmap(lambda z: jnp.sum(gradV(z) ** 2))(targets))

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
    l_min = jnp.mean(jax.nn.relu(v_t[:, None] - v_p + margin))

    # --- a barrier between neighbouring items (otherwise the wells merge) ---
    l_bar = jnp.asarray(0.0)
    if K > 1:
        i, j = jnp.triu_indices(K, k=1)
        mids = 0.5 * (targets[i] + targets[j])
        v_m = jax.vmap(V)(mids)
        v_hi = jnp.maximum(v_t[i], v_t[j])
        l_bar = jnp.mean(jax.nn.relu(barrier + v_hi - v_m) ** 2)

    return w_grad * l_grad + w_min * l_min + w_barrier * l_bar


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
