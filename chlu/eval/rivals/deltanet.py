"""⭐ **Delta-rule memories** — DeltaNet, Gated DeltaNet, **Gated DeltaNet-2**.

Minimal *faithful* reimplementations on the gym harness. ⛔ §A14.2's ruling:
**Gated DeltaNet-2 (arXiv:2605.22791) REPLACES GDN as the delta-rule reference
arm**; GDN(-1) is retained only as the ablation that isolates what the -2 revision
changed.

**Equations implemented — verified from the GDN-2 paper's own HTML this session,
not inherited from GDN(-1)'s accounting** (the task requires saying which):

=========  ===================================================================
Eq. 1      ``S_t = S_{t-1} + k_t v_t^T``, ``o_t = S_t^T q_t`` (linear attention)
Eq. 5      **DeltaNet**: ``S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T``
Eq. 6      **Gated DeltaNet**: ``S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1}
           + beta_t k_t v_t^T`` (scalar ``alpha_t``)
Eq. 8      **GDN-2** gated directions: ``e_t = b_t (*) k_t``, ``z_t = w_t (*) v_t``
Eq. 9      ``Sbar_t = D_t S_{t-1}``; ``r_t = Sbar_t^T e_t``;
           ``S_t = Sbar_t + k_t (z_t - r_t)^T``
Eq. 10     **GDN-2, the boxed recurrence**:
           ``S_t = (I - k_t (b_t (*) k_t)^T) D_t S_{t-1} + k_t (w_t (*) v_t)^T``
Eq. 11     ``b_t = sigma(W_b x_t)``, ``w_t = sigma(W_w x_t)``  (channel-wise
           **erase** and **write** gates — the whole point of the -2 revision)
Eq. 12     ``g_t = -exp(a) (*) softplus(W_f x_t + delta)``, ``alpha_t = exp(g_t)``
§3.1       optional negative-eigenvalue variant: the **erase** gate is scaled to
           ``[0,2]^{d_k}`` while the **write** gate stays in ``[0,1]^{d_v}``
§3.5       block design: q and k paths use L2 normalisation, v uses SiLU
Eq. 90     ⭐ **state-size convention, VERIFIED not inherited**: *"a per-layer
           recurrent state of ``H d_k d_v = 16 * 128 * 128 = 262,144`` floats per
           batch element"* — the -2 revision **preserves** DeltaNet/GDN's
           ``n_head * d_k * d_v`` accounting (`rival-recon` F2), so the ledger row
           in `PREREG-Bprime.md` §2 stands unchanged for GDN-2.
=========  ===================================================================

⚠ **Minimal, and captioned as such in every table.** Faithful to the update
equation and the state size; minimal in everything else — ``n_head = 1``, no short
convolution, no SWA hybrid block, no chunkwise WY kernel (we run the sequential
recurrence, which is the same function). Reference implementations exist and are
portable (FLA's 41-model table); **this is not a vendored training stack.**

⭐ **Learned-initial-state rule** (`PREREG-Bprime.md` §4.1): ``S_0`` is
**PARAMETERS** (it is shared across streams); only the per-stream deviation is
**STATE**.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.eval.rivals.ledger import TwoSidedLedger

#: The three delta-rule variants. ``gdn2`` is the §A14.2 **reference** arm.
DELTA_VARIANTS = ("deltanet", "gdn", "gdn2")


def _l2(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


class DeltaMemory(eqx.Module):
    """DeltaNet / Gated DeltaNet / Gated DeltaNet-2 as a gym-harness memory.

    State ``S`` is ``(d_k, d_v)``; the read is ``o = S^T q`` (Eq. 1), followed by
    the shared output head ``theta_O``.
    """

    theta_K: jnp.ndarray
    theta_Q: jnp.ndarray
    theta_V: jnp.ndarray
    theta_O: jnp.ndarray
    S0: jnp.ndarray
    w_beta: jnp.ndarray       # (d_in,)      beta_t = sigmoid(w_beta . x)
    W_b: jnp.ndarray          # (d_k, d_in)  Eq. 11 erase gate
    W_w: jnp.ndarray          # (d_v, d_in)  Eq. 11 write gate
    W_f: jnp.ndarray          # (d_k, d_in)  Eq. 12 log-decay projection
    a_log: jnp.ndarray        # (d_k,)       Eq. 12 `a`
    delta: jnp.ndarray        # (d_k,)       Eq. 12 `delta`

    variant: str = eqx.field(static=True)
    d_in: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    n_head: int = eqx.field(static=True)
    erase_scale: float = eqx.field(static=True)

    def __init__(self, d_in: int, d_head: int, m: int, *, key,
                 variant: str = "gdn2", n_head: int = 1,
                 erase_scale: float = 2.0, init_scale: float = 0.5):
        if variant not in DELTA_VARIANTS:
            raise ValueError(f"unknown delta variant {variant!r}; "
                             f"known: {DELTA_VARIANTS}")
        self.variant = str(variant)
        self.d_in, self.d_head, self.m = int(d_in), int(d_head), int(m)
        self.n_head = int(n_head)
        # §3.1: the negative-eigenvalue variant scales ONLY the erase gate to
        # [0,2]^{d_k}; the write gate stays in [0,1]^{d_v}.
        self.erase_scale = float(erase_scale)
        ks = jax.random.split(key, 8)
        s = float(init_scale)
        self.theta_K = jax.random.normal(ks[0], (d_head, d_in)) * s / np.sqrt(d_in)
        self.theta_Q = jax.random.normal(ks[1], (d_head, d_in)) * s / np.sqrt(d_in)
        self.theta_V = jax.random.normal(ks[2], (d_head, d_in)) * s / np.sqrt(d_in)
        self.theta_O = jax.random.normal(ks[3], (m, d_head)) * s / np.sqrt(d_head)
        self.S0 = jnp.zeros((d_head, d_head))
        self.w_beta = jnp.zeros((d_in,))
        self.W_b = jax.random.normal(ks[4], (d_head, d_in)) * s / np.sqrt(d_in)
        self.W_w = jax.random.normal(ks[5], (d_head, d_in)) * s / np.sqrt(d_in)
        self.W_f = jax.random.normal(ks[6], (d_head, d_in)) * s / np.sqrt(d_in)
        self.a_log = jnp.zeros((d_head,))
        self.delta = jnp.zeros((d_head,))

    # -- the projections (§3.5 block design) --------------------------------
    def _kv(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return _l2(self.theta_K @ x), jax.nn.silu(self.theta_V @ x)

    def init_state(self) -> jnp.ndarray:
        """``S_0`` — **parameters** under the learned-initial-state rule."""
        return self.S0

    # -- the recurrence ------------------------------------------------------
    def write(self, xs: jnp.ndarray, mask: Optional[jnp.ndarray] = None
              ) -> jnp.ndarray:
        """Run the delta rule over the write stream.

        ``mask`` (1 = a real token) makes a padded row a **no-op**: the state is
        interpolated back to its previous value, so padding can neither erase nor
        write.
        """
        xs = jnp.asarray(xs, dtype=jnp.float32)
        msk = (jnp.ones((xs.shape[0],)) if mask is None
               else jnp.asarray(mask, dtype=jnp.float32))
        variant = self.variant

        def step(S, carry):
            x, mm = carry
            k, v = self._kv(x)
            if variant == "gdn2":
                # Eq. 11 / §3.1
                b = self.erase_scale * jax.nn.sigmoid(self.W_b @ x)   # (d_k,)
                w = jax.nn.sigmoid(self.W_w @ x)                      # (d_v,)
                # Eq. 12: channel-wise decay, alpha in (0,1]^{d_k}
                g = -jnp.exp(self.a_log) * jax.nn.softplus(self.W_f @ x + self.delta)
                alpha = jnp.exp(g)
                Sbar = alpha[:, None] * S                             # Eq. 9 D_t S
                e = b * k                                             # Eq. 8
                z = w * v                                             # Eq. 8
                r = Sbar.T @ e                                        # Eq. 9
                Snew = Sbar + jnp.outer(k, z - r)                     # Eq. 9/10
            else:
                beta = jax.nn.sigmoid(jnp.dot(self.w_beta, x))
                if variant == "gdn":
                    # Eq. 6 with the Eq. 12 parameterisation collapsed to a scalar
                    g = -jnp.exp(jnp.mean(self.a_log)) * jax.nn.softplus(
                        jnp.mean(self.W_f @ x) + jnp.mean(self.delta))
                    alpha = jnp.exp(g)
                else:  # Eq. 5, DeltaNet
                    alpha = 1.0
                Sbar = alpha * S
                Snew = Sbar + jnp.outer(k, beta * (v - Sbar.T @ k))
            return S + mm * (Snew - S), None

        S, _ = jax.lax.scan(step, self.init_state(), (xs, msk))
        return S

    def read(self, S: jnp.ndarray, xq: jnp.ndarray) -> jnp.ndarray:
        """Eq. 1's output rule ``o = S^T q`` plus the shared output head."""
        q = jax.vmap(lambda x: _l2(self.theta_Q @ x))(jnp.asarray(xq, jnp.float32))
        return (q @ S) @ self.theta_O.T

    # -- the byte-matched table (P5's construction) --------------------------
    def kv_table(self, xs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """The ``(theta_K x_t, theta_V x_t)`` pairs **as the recurrence consumes
        them** (post-L2 / post-SiLU) — the table stores exactly what the state is
        trying to store."""
        return jax.vmap(self._kv)(jnp.asarray(xs, dtype=jnp.float32))

    def query_keys(self, xq: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda x: _l2(self.theta_Q @ x))(jnp.asarray(xq, jnp.float32))

    def decode_values(self, vals: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(vals, dtype=jnp.float32) @ self.theta_O.T

    # -- the ledger ----------------------------------------------------------
    @property
    def d_k(self) -> int:
        return int(self.d_head)

    @property
    def d_v(self) -> int:
        return int(self.d_head)

    def declared_state_floats(self) -> int:
        return int(self.n_head * self.d_head * self.d_head)

    def ledger(self, moved: Optional[int] = None) -> TwoSidedLedger:
        d, di, m = self.d_head, self.d_in, self.m
        pb = {"theta_K": d * di, "theta_Q": d * di, "theta_V": d * di,
              "theta_O": m * d, "S0_init": d * d}
        if self.variant == "gdn2":
            pb.update({"W_b(Eq11)": d * di, "W_w(Eq11)": d * di,
                       "W_f(Eq12)": d * di, "a(Eq12)": d, "delta(Eq12)": d,
                       "w_beta(unused)": di})
        else:
            pb.update({"w_beta": di, "W_f(Eq12)": d * di, "a(Eq12)": d,
                       "delta(Eq12)": d, "W_b(unused)": d * di,
                       "W_w(unused)": d * di})
        sb = {"S_deviation": int(self.n_head * d * d)}
        return TwoSidedLedger(
            arm=self.variant, param_floats=int(sum(pb.values())),
            state_floats=int(sum(sb.values())), param_breakdown=pb,
            state_breakdown=sb,
            state_convention=("n_head*d_k*d_v — VERIFIED for GDN-2 from its own "
                              "Eq. 90 (H*d_k*d_v = 16*128*128 = 262,144 floats "
                              "per layer), i.e. the -2 revision preserves the "
                              "DeltaNet/GDN accounting"),
            note=("S_0 is PARAMETERS (shared across streams); only the per-stream "
                  "deviation is STATE"
                  + (f"; measured moved floats = {int(moved)}" if moved is not None
                     else "")),
        ).check()


def metric_native_verdict(variant: str) -> Dict[str, Any]:
    """The equation-level argument (D3.6), stated before it is measured."""
    common = ("the read is o = S^T q (Eq. 1). Every variant's S is a sum of outer "
              "products k_s z_s^T with data-dependent left/right factors, so "
              "o = sum_s z_s (k_s . q) up to the erase corrections: a **linear "
              "kernel smoother over the stored (k, v) pairs under the dot-product "
              "metric**, with q and k L2-normalised (§3.5) so arg-min ||q - k|| "
              "and arg-max q.k coincide EXACTLY. This is `rival-recon` F9's "
              "'the rivals' reads are metric-native too', at equation level.")
    extra = {
        "deltanet": "Eq. 5: the only non-metric ingredient is beta_t, a scalar.",
        "gdn": "Eq. 6 adds a scalar decay alpha_t; still a scalar reweighting.",
        "gdn2": ("Eq. 10 makes erase (b_t, on the key side) and write (w_t, on the "
                 "value side) CHANNEL-WISE, so the effective metric is a learned "
                 "diagonal reweighting of the key space rather than the identity. "
                 "It is still a metric — a Mahalanobis one with a diagonal, "
                 "token-dependent shape — so criterion 4 still closes, and the "
                 "byte-matched table it is audited against is entitled to the same "
                 "shape (which is why the +0 B readers are run on the SAME "
                 "projected keys)."),
    }[variant]
    return {"verdict": "metric-native", "argument": common + " " + extra,
            "measured_against": "its own byte-matched (theta_K x, theta_V x) table"}


__all__ = ["DELTA_VARIANTS", "DeltaMemory", "metric_native_verdict"]
