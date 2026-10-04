"""⭐ **TTT-class memory** — a minimal *faithful* reimplementation on the gym harness.

Sun, Li, Dalal et al., *"Learning to (Learn at Test Time): RNNs with Expressive
Hidden States"*, arXiv:2407.04620. **Equations implemented (read off the paper's
HTML this session, not inherited):**

===========  =====================================================================
Eq. 1        ``z_t = f(x_t; W_t)`` — the output rule
Eq. 2        ``W_t = W_{t-1} - eta * grad l(W_{t-1}; x_t)`` — the update rule
Eq. 4        ``l(W; x_t) = || f(theta_K x_t; W) - theta_V x_t ||^2`` — the learned
             self-supervised inner task (§2.3)
Eq. 5        ``z_t = f(theta_Q x_t; W_t)`` — the read
§2.4         mini-batch TTT: ``G_t = grad l(W_{t'}; x_t)`` with
             ``t' = t - mod(t, b)``, ``b = 16`` *"for all experiments in this paper"*
§2.7         ``f_res(x) = x + LN(f(x))``; **learnable ``W_0``, shared across all
             sequences**; learnable ``eta``; ``f_lin(x) = Wx`` (square ``W``),
             ``f_MLP`` = two layers, hidden ``4x``, GELU
===========  =====================================================================

⭐ **The learned-initial-state rule** (`PREREG-Bprime.md` §4.1) is the audit's
sharpest edge and the paper hands it to us verbatim: *"the TTT initialization
``W_0`` is shared between all sequences, even though subsequent weights
``W_1..W_T`` are different for each input sequence"* ⇒ **``W_0`` is PARAMETERS
(F1); only the per-stream deviation is STATE (F2).** State ledger
``d_head^2`` (Linear) / ``8 d_head^2`` (MLP) **+ the ``b = 16`` in-flight buffer**
(`PREREG-Bprime.md` §2), and the *measured* moved-float count is reported beside
the declared convention.

⚠ **Minimal, and captioned as such in every table.** Faithful to the update
equation, the inner task and the state size; minimal in everything else — one
head, no backbone, no convolution branch, no gating over a residual stream. This
is **not** a vendored training stack.

⚠ **Declared regime caveat.** The gym's write stream is ~10-19 tokens, so at the
paper's own ``b = 16`` the whole stream is one or two mini-batches — i.e. batch
GD, the regime the paper's Table 1 ablation shows is **1.70 ppl worse** than
``b = 16`` at ``T = 2048``. ``b`` is therefore part of the arm's tuning grid
(``b in {1, 16}``, best-of-grid on the fit split, **reported**), so the rival is
audited at its own best rather than at a setting our stream length degrades.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.eval.rivals.ledger import TTT_MINI_BATCH, TwoSidedLedger


def _layer_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                eps: float = 1e-6) -> jnp.ndarray:
    mu = jnp.mean(x)
    var = jnp.mean((x - mu) ** 2)
    return gamma * (x - mu) / jnp.sqrt(var + eps) + beta


class TTTMemory(eqx.Module):
    """TTT-Linear / TTT-MLP as an addressable memory on the gym stream.

    ``kind = "linear"`` ⇒ ``f_lin(x) = W x`` with ``W`` square (state ``d_head^2``);
    ``kind = "mlp"`` ⇒ two layers with ``4x`` hidden and GELU (state ``8 d_head^2``).
    """

    theta_K: jnp.ndarray
    theta_Q: jnp.ndarray
    theta_V: jnp.ndarray
    theta_O: jnp.ndarray
    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: Optional[jnp.ndarray]
    b2: Optional[jnp.ndarray]
    ln_gamma: jnp.ndarray
    ln_beta: jnp.ndarray
    theta_lr: jnp.ndarray
    log_eta: jnp.ndarray

    kind: str = eqx.field(static=True)
    d_in: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    mini_batch: int = eqx.field(static=True)

    def __init__(self, d_in: int, d_head: int, m: int, *, key,
                 kind: str = "linear", mini_batch: int = TTT_MINI_BATCH,
                 init_scale: float = 0.5):
        if kind not in ("linear", "mlp"):
            raise ValueError(f"unknown TTT kind {kind!r}")
        self.kind = str(kind)
        self.d_in, self.d_head, self.m = int(d_in), int(d_head), int(m)
        self.mini_batch = int(mini_batch)
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        s = float(init_scale)
        self.theta_K = jax.random.normal(k1, (d_head, d_in)) * s / np.sqrt(d_in)
        self.theta_Q = jax.random.normal(k2, (d_head, d_in)) * s / np.sqrt(d_in)
        self.theta_V = jax.random.normal(k3, (d_head, d_in)) * s / np.sqrt(d_in)
        self.theta_O = jax.random.normal(k4, (m, d_head)) * s / np.sqrt(d_head)
        if kind == "linear":
            self.W1 = jax.random.normal(k5, (d_head, d_head)) * s / np.sqrt(d_head)
            self.b1 = jnp.zeros((d_head,))
            self.W2 = None
            self.b2 = None
        else:
            h = 4 * d_head
            self.W1 = jax.random.normal(k5, (h, d_head)) * s / np.sqrt(d_head)
            self.b1 = jnp.zeros((h,))
            self.W2 = jax.random.normal(k6, (d_head, h)) * s / np.sqrt(h)
            self.b2 = jnp.zeros((d_head,))
        self.ln_gamma = jnp.ones((d_head,))
        self.ln_beta = jnp.zeros((d_head,))
        self.theta_lr = jnp.zeros((d_in,))
        self.log_eta = jnp.log(jnp.asarray(1.0))

    # -- the inner model f, with §2.7's LN + residual -----------------------
    def _f_res(self, W: Tuple[jnp.ndarray, ...], x: jnp.ndarray) -> jnp.ndarray:
        if self.kind == "linear":
            W1, b1 = W
            y = W1 @ x + b1
        else:
            W1, b1, W2, b2 = W
            y = W2 @ jax.nn.gelu(W1 @ x + b1) + b2
        return x + _layer_norm(y, self.ln_gamma, self.ln_beta)

    def _loss(self, W: Tuple[jnp.ndarray, ...], x: jnp.ndarray) -> jnp.ndarray:
        """Eq. 4: ``|| f(theta_K x; W) - theta_V x ||^2``."""
        r = self._f_res(W, self.theta_K @ x) - self.theta_V @ x
        return jnp.sum(r * r)

    # -- the state -----------------------------------------------------------
    def init_state(self) -> Tuple[jnp.ndarray, ...]:
        """``W_0`` — **parameters** under the learned-initial-state rule; the
        *deviation* from this is what the state ledger counts."""
        if self.kind == "linear":
            return (self.W1, self.b1)
        return (self.W1, self.b1, self.W2, self.b2)

    def write(self, xs: jnp.ndarray, mask: Optional[jnp.ndarray] = None
              ) -> Tuple[jnp.ndarray, ...]:
        """Run the inner loop over the stream (Eq. 2 + §2.4 mini-batching).

        ``mask`` (``(T,)``, 1 = a real token) exists so a padded chunk contributes
        exactly zero gradient — the padding must not be able to write.
        """
        xs = jnp.asarray(xs, dtype=jnp.float32)
        T = int(xs.shape[0])
        msk = jnp.ones((T,)) if mask is None else jnp.asarray(mask, dtype=jnp.float32)
        b = max(1, min(int(self.mini_batch), max(T, 1)))
        n_chunk = int(np.ceil(T / b))
        pad = n_chunk * b - T
        if pad:
            xs = jnp.concatenate([xs, jnp.zeros((pad, xs.shape[1]))], axis=0)
            msk = jnp.concatenate([msk, jnp.zeros((pad,))], axis=0)
        xs = xs.reshape(n_chunk, b, -1)
        msk = msk.reshape(n_chunk, b)
        grad_fn = jax.grad(self._loss)
        eta0 = jnp.exp(self.log_eta)

        def step(W, carry):
            xc, mc = carry
            # per-token learning rate (§2.7 "learnable eta"), gated by the mask
            eta = eta0 * jax.nn.sigmoid(xc @ self.theta_lr) * mc  # (b,)
            # ⭐ §2.4: EVERY gradient in the mini-batch is taken at W_{t'} (=W)
            gs = jax.vmap(lambda x: grad_fn(W, x))(xc)
            upd = tuple(jnp.tensordot(eta, g, axes=(0, 0)) for g in gs)
            return tuple(w - u for w, u in zip(W, upd, strict=True)), None

        W, _ = jax.lax.scan(step, self.init_state(), (xs, msk))
        return W

    def read(self, W: Tuple[jnp.ndarray, ...], xq: jnp.ndarray) -> jnp.ndarray:
        """Eq. 5 + the output projection: ``theta_O f(theta_Q x_q; W_T)``."""
        xq = jnp.asarray(xq, dtype=jnp.float32)
        z = jax.vmap(lambda x: self._f_res(W, self.theta_Q @ x))(xq)
        return z @ self.theta_O.T

    # -- the byte-matched table (P5's construction) --------------------------
    def kv_table(self, xs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """The ``(theta_K x_t, theta_V x_t)`` pairs — the *same* key/value the
        inner loss writes, so the table is literally what the memory is trying to
        store."""
        xs = jnp.asarray(xs, dtype=jnp.float32)
        return xs @ self.theta_K.T, xs @ self.theta_V.T

    def query_keys(self, xq: jnp.ndarray) -> jnp.ndarray:
        """``theta_Q x_q`` — the address the table launder matches on."""
        return jnp.asarray(xq, dtype=jnp.float32) @ self.theta_Q.T

    def decode_values(self, vals: jnp.ndarray) -> jnp.ndarray:
        """The **shared** output head, applied to whatever the arm retrieved."""
        return jnp.asarray(vals, dtype=jnp.float32) @ self.theta_O.T

    # -- the ledger ----------------------------------------------------------
    @property
    def d_k(self) -> int:
        return int(self.d_head)

    @property
    def d_v(self) -> int:
        return int(self.d_head)

    def declared_state_floats(self) -> int:
        core = self.d_head ** 2 if self.kind == "linear" else 8 * self.d_head ** 2
        return int(core + self.mini_batch * self.d_head)

    def ledger(self, moved: Optional[int] = None) -> TwoSidedLedger:
        """F1/F2 with the learned-initial-state rule applied.

        ``moved`` is the *measured* number of state floats the stream changed; the
        declared convention (`PREREG-Bprime.md` §2) is reported as the primary and
        the measurement as its check.
        """
        d, di, m = self.d_head, self.d_in, self.m
        pb = {"theta_K": d * di, "theta_Q": d * di, "theta_V": d * di,
              "theta_O": m * d, "layer_norm": 2 * d, "theta_lr": di, "log_eta": 1}
        if self.kind == "linear":
            pb["W0_init"] = d * d + d
        else:
            pb["W0_init"] = 4 * d * d + 4 * d + d * 4 * d + d
        core = d * d if self.kind == "linear" else 8 * d * d
        sb = {"W_deviation": int(core), "mini_batch_buffer": self.mini_batch * d}
        return TwoSidedLedger(
            arm=f"ttt_{self.kind}", param_floats=int(sum(pb.values())),
            state_floats=int(sum(sb.values())), param_breakdown=pb,
            state_breakdown=sb,
            state_convention=("d_head^2 (Linear) / 8*d_head^2 (MLP) + b*d_head "
                              f"in-flight buffer at b={self.mini_batch} "
                              "(PREREG-Bprime §2; Sun et al. §2.4/§2.7)"),
            note=("W_0 is PARAMETERS (paper: shared between all sequences); only "
                  "the per-stream deviation is STATE"
                  + (f"; measured moved floats = {int(moved)}" if moved is not None
                     else "")),
        ).check()


def measured_state_floats(W0, W1) -> int:
    """How many state floats the stream actually moved (the rivals get the same
    measured check the CLU gets in :func:`~chlu.eval.rivals.ledger.clu_two_sided_ledger`)."""
    a = [np.asarray(x) for x in jax.tree_util.tree_leaves(W0)]
    b = [np.asarray(x) for x in jax.tree_util.tree_leaves(W1)]
    return int(sum(int(np.sum(x != y)) for x, y in zip(a, b, strict=True)))


def metric_native_verdict(kind: str) -> Dict[str, Any]:
    """The equation-level argument, stated before it is measured (D3.6)."""
    if kind == "linear":
        return {
            "verdict": "metric-native",
            "argument": ("with f_lin and gradients taken at W_0 (§2.6 equivalence) "
                         "the read is z_t = W_0 q - 2 eta sum_s (W_0 k_s - v_s)"
                         "(k_s . q): a LINEAR kernel smoother over the stored "
                         "(k, v) pairs. Theorem 2 of the paper makes the same point "
                         "in general: the nonparametric TTT learner IS the "
                         "Nadaraya-Watson estimator with kernel exp((theta_K x)^T "
                         "theta_Q x'). A read that is a kernel average of stored "
                         "values is a metric operation on its own state."),
            "measured_against": "its own byte-matched (theta_K x, theta_V x) table",
        }
    return {
        "verdict": "weakly metric-native",
        "argument": ("f_MLP inserts a GELU between the two layers, so the read is "
                     "not a kernel average of stored values and the metric-native "
                     "ceiling argument (intervention §6 criterion 4) does not close "
                     "at the equation level. This is the ONLY arm in this task for "
                     "which it does not, and it is why P3 was written about "
                     "function-valued memories."),
        "measured_against": "its own byte-matched (theta_K x, theta_V x) table",
    }


__all__ = ["TTTMemory", "measured_state_floats", "metric_native_verdict"]
