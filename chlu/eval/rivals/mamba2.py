"""⭐ **Mamba-2 (SSD)** — the selective state-space rival arm.

Dao & Gu, *"Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality"*, **ICML 2024**, arXiv:2405.21060.
Reference implementation ``state-spaces/mamba`` (``mamba_ssm/modules/mamba2.py``).
⚠ **Citation provenance:** venue/year/id and the reference-implementation defaults
below are taken from `rival-recon`'s pinned record (`.claude/outputs/rival-recon.md`
§1.4/§1.5), which is a prior `web-scout` verification; they were **not re-verified
against an authoritative record in this session** (no web tool in this agent's kit)
— flagged per `bprime-cite-check`'s single-sourcing pattern.

**Why this arm exists.** The B′ survey sentence names *"SSMs (Mamba-1/2/3)"* among
the families it positions against, and until now **none was measured** — the
referee's missing-experiment 5. This is the minimal faithful arm that closes it.

**The equations implemented** (this rig's notation, ``n_head = 1``, ``ngroups = 1``):

=========  ===================================================================
SSM        ``h_t = A_t h_{t-1} + B_t x_t``, ``y_t = C_t^T h_t`` (+ ``D x_t``)
**SSD**    the paper's structural restriction: ``A_t = a_t I`` — a **scalar**
           times the identity ⇒ the recurrence is a 1-semiseparable masked
           attention, which is the *duality* the paper is named for
selection  ``Delta_t = softplus(w_Delta . x_t + Delta_bias)``,
           ``B_t = theta_K x_t``, ``C_t = theta_Q x_t`` (all input-dependent —
           Mamba-1 §3.2's selection mechanism, carried over unchanged)
discretise ``a_t = exp(Delta_t * A)`` with ``A = -exp(A_log) < 0``; the input is
           ZOH-scaled, ``x_t -> Delta_t x_t``
chunked    the **SSD block/chunk algorithm** (§6): within a chunk the state pass
           is one matmul ``B_c^T diag(exp(cum_end - cum_j) Delta_j) V_c``; across
           chunks a scalar decay. ⭐ Unlike TTT's mini-batch ``b``, the SSD chunk
           length is **exact** — it is a re-association of the same sum, so it is
           provably inert and is NOT a tuning axis (asserted in the tests).
init       reference-implementation init: ``A ~ U(1, 16)`` (``A_init_range``) and
           ``Delta ~ exp(U(log 1e-3, log 1e-1))`` inverse-softplused into
           ``Delta_bias`` (``dt_min``/``dt_max``)
=========  ===================================================================

⭐ **Mamba-2 IS the delta arms' degenerate case, and the rival authors say so, not
us:** Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025) presents *"Mamba2 as
``S_t = alpha_t S_{t-1} + v_t k_t^T``"* — i.e. Eq. 6 with the delta-erase term
deleted. That is exactly what this module computes, and it is why this arm's row is
the cleanest available isolation of *what the erase term buys*: at the shipped
budget Mamba-2 and the three delta arms land on **byte-identical state** (5184 B).

⚠ **Minimal, and captioned as such in every table** — the same caption the TTT and
delta arms carry, because protocol uniformity across families *is* the deliverable:
faithful to the update equation and the state size, minimal in everything else
(``n_head = 1``, **no short convolution branch**, no gated-RMSNorm/``z`` branch by
default, no backbone, no Triton kernel — we run the chunked recurrence, which is
the same function). **Not** a vendored training stack.

⚠ **The dropped conv branch is a declared deviation, and it is IN THE RIVAL'S
FAVOUR under an iso-state budget.** The reference implementation's inference cache
is *two* tensors — ``conv_state`` ``(d_inner + 2 ngroups d_state) x d_conv`` and
``ssm_state`` ``n_head x head_dim x d_state`` (`rival-recon` §1.4). Dropping the
conv means **every one of the budget's bytes goes to the SSM state** instead of
9/16 of them going to a 4-tap window, i.e. the arm gets a *larger* recurrent state
than a faithful block would at the same byte budget. The ``d_conv`` window is the
Mamba analogue of TTT's in-flight mini-batch buffer, which this rig *does* count;
we state the asymmetry rather than leave it for a referee.

⭐ **Learned-initial-state rule** (`PREREG-Bprime.md` §4.1): ``S_0`` is
**PARAMETERS** (shared across streams); only the per-stream deviation is **STATE**.
Mamba-2's published ``h_0`` is the zero state, so ``S_0`` is initialised to zeros
here — learnable, exactly as the delta arms' ``S_0`` is, so the rule bites the same
way for both.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.eval.rivals.ledger import TwoSidedLedger

#: The SSD chunk length. ⭐ The reference implementation's default is 256
#: (`rival-recon` §1.4/§1.5), which is >> this rig's ~10-19-token stream, so the
#: chunk granularity is **matched to the rig's** (the same 16 the TTT arm's
#: mini-batch uses) and the equivalence is asserted in the tests. Because SSD
#: chunking is an exact re-association, the choice cannot change a single output
#: bit beyond fp32 rounding — which is why ``Q`` is not in the tuning grid while
#: TTT's ``b`` is.
SSD_CHUNK = 16

#: Reference-implementation init ranges (``mamba_ssm``): ``A_init_range=(1,16)``,
#: ``dt_min=1e-3``, ``dt_max=1e-1``.
A_INIT_RANGE = (1.0, 16.0)
DT_MIN, DT_MAX = 1e-3, 1e-1


def _inverse_softplus(x: jnp.ndarray) -> jnp.ndarray:
    return x + jnp.log(-jnp.expm1(-x))


class Mamba2Memory(eqx.Module):
    """Mamba-2's SSD recurrence as a gym-harness memory.

    State ``S`` is ``(d_state, d_head) = (N, P)`` — the same ``(d_k, d_v)`` shape
    the delta arms carry, so the two are byte-comparable by construction. The read
    is ``o = S^T C_q`` followed by the shared output head ``theta_O``.

    ``use_D`` / ``gate_z`` expose the block-level skip and gate that the default
    (minimal) configuration drops; they exist so the "you hobbled Mamba-2" attack
    can be answered by measurement rather than by assertion. ⛔ Both default to
    ``False`` — the same minimality every other arm in this rig is held to.
    """

    theta_K: jnp.ndarray      # (N, d_in)   B_t = theta_K x_t   (selective)
    theta_Q: jnp.ndarray      # (N, d_in)   C_t = theta_Q x_t   (selective)
    theta_V: jnp.ndarray      # (P, d_in)   the SSM input path
    theta_O: jnp.ndarray      # (m, P)      shared output head
    S0: jnp.ndarray           # (N, P)      h_0 — PARAMETERS (learned-init rule)
    w_dt: jnp.ndarray         # (d_in,)     Delta_t = softplus(w_dt.x + dt_bias)
    dt_bias: jnp.ndarray      # ()          reference init: inv_softplus(U[1e-3,1e-1])
    A_log: jnp.ndarray        # ()          A = -exp(A_log), A ~ U(1, 16)
    D: jnp.ndarray            # (P,)        skip connection (off by default)
    W_z: jnp.ndarray          # (P, d_in)   gate branch (off by default)

    d_in: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    n_head: int = eqx.field(static=True)
    chunk: int = eqx.field(static=True)
    use_D: bool = eqx.field(static=True)
    gate_z: bool = eqx.field(static=True)

    def __init__(self, d_in: int, d_head: int, m: int, *, key,
                 d_state: Optional[int] = None, n_head: int = 1,
                 chunk: int = SSD_CHUNK, use_D: bool = False,
                 gate_z: bool = False, init_scale: float = 0.5):
        self.d_in, self.d_head, self.m = int(d_in), int(d_head), int(m)
        # ⭐ the iso-state sizing choice, declared: N = P = d_head, so the state is
        # n_head*d_state*d_head = d^2 — byte-identical to the delta arms'
        # n_head*d_k*d_v at the same budget. (The paper keeps N and P independent;
        # the rig has one width knob, and this is the choice that makes the
        # cross-family comparison exact.)
        self.d_state = int(d_state) if d_state is not None else int(d_head)
        self.n_head = int(n_head)
        self.chunk = int(chunk)
        self.use_D, self.gate_z = bool(use_D), bool(gate_z)
        ks = jax.random.split(key, 8)
        s, N, P = float(init_scale), self.d_state, self.d_head
        self.theta_K = jax.random.normal(ks[0], (N, d_in)) * s / np.sqrt(d_in)
        self.theta_Q = jax.random.normal(ks[1], (N, d_in)) * s / np.sqrt(d_in)
        self.theta_V = jax.random.normal(ks[2], (P, d_in)) * s / np.sqrt(d_in)
        self.theta_O = jax.random.normal(ks[3], (m, P)) * s / np.sqrt(P)
        self.S0 = jnp.zeros((N, P))
        self.w_dt = jnp.zeros((d_in,))
        # reference init: dt ~ exp(U(log dt_min, log dt_max)), bias = inv_softplus
        dt = jnp.exp(jax.random.uniform(
            ks[4], (), minval=float(np.log(DT_MIN)), maxval=float(np.log(DT_MAX))))
        self.dt_bias = _inverse_softplus(dt)
        # reference init: A ~ U(A_init_range), stored as log A (A_real = -A)
        a = jax.random.uniform(ks[5], (), minval=A_INIT_RANGE[0],
                               maxval=A_INIT_RANGE[1])
        self.A_log = jnp.log(a)
        self.D = jnp.ones((P,))
        self.W_z = jax.random.normal(ks[6], (P, d_in)) * s / np.sqrt(d_in)

    # -- the selective projections -------------------------------------------
    def _kv(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """``(B_t, v_t)`` — the pair the state is trying to store.

        ⚠ **No L2 normalisation.** GDN-2's §3.5 block design normalises the
        ``q``/``k`` paths; Mamba-2 does **not**, and we do not add it — the SiLU on
        the value path is Mamba's own (``x = silu(conv1d(x))``, minus the conv).
        The consequence is a *finding*, not an oversight: in Mamba-2's own key
        space ``arg-min ||q - k||`` and ``arg-max q.k`` do **not** coincide, so its
        metric-native verdict is strictly weaker than the delta arms' (see
        :func:`metric_native_verdict`).
        """
        return self.theta_K @ x, jax.nn.silu(self.theta_V @ x)

    def _dt(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.softplus(jnp.dot(self.w_dt, x) + self.dt_bias)

    def init_state(self) -> jnp.ndarray:
        """``S_0`` — **parameters** under the learned-initial-state rule."""
        return self.S0

    # -- the recurrence, three ways (all the same function) -------------------
    def _stream(self, xs: jnp.ndarray, mask: Optional[jnp.ndarray]
                ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """``(B, v, Delta, log a)`` per token, with the mask folded into ``Delta``.

        A masked token gets ``Delta_t = 0`` ⇒ ``a_t = exp(0) = 1`` and input scale
        ``0`` ⇒ ``h_t = h_{t-1}`` **exactly**. Padding can neither decay nor write.
        """
        xs = jnp.asarray(xs, dtype=jnp.float32)
        msk = (jnp.ones((xs.shape[0],)) if mask is None
               else jnp.asarray(mask, dtype=jnp.float32))
        B, v = jax.vmap(self._kv)(xs)                       # (T,N), (T,P)
        dt = jax.vmap(self._dt)(xs) * msk                   # (T,)
        log_a = -dt * jnp.exp(self.A_log)                   # (T,), <= 0
        return B, v, dt, log_a

    def write(self, xs: jnp.ndarray, mask: Optional[jnp.ndarray] = None
              ) -> jnp.ndarray:
        """The **chunked SSD** state pass (§6) — the shipped path.

        Per chunk, with ``cum_j`` the inclusive cumulative log-decay from the chunk
        start and ``tot = cum_{Q-1}``::

            S <- exp(tot) * S  +  B_c^T diag(exp(tot - cum_j) * Delta_j) V_c

        which is one matmul instead of ``Q`` rank-1 updates, and is *exactly* the
        sequential recurrence re-associated (asserted in the tests).
        """
        xs = jnp.asarray(xs, dtype=jnp.float32)
        T = int(xs.shape[0])
        if T == 0:
            return self.init_state()
        B, v, dt, log_a = self._stream(xs, mask)
        Q = max(1, min(int(self.chunk), T))
        n_chunk = int(np.ceil(T / Q))
        pad = n_chunk * Q - T
        if pad:                                   # padded rows: dt = 0 ⇒ no-op
            B = jnp.concatenate([B, jnp.zeros((pad, B.shape[1]))], axis=0)
            v = jnp.concatenate([v, jnp.zeros((pad, v.shape[1]))], axis=0)
            dt = jnp.concatenate([dt, jnp.zeros((pad,))], axis=0)
            log_a = jnp.concatenate([log_a, jnp.zeros((pad,))], axis=0)
        B = B.reshape(n_chunk, Q, -1)
        v = v.reshape(n_chunk, Q, -1)
        dt = dt.reshape(n_chunk, Q)
        log_a = log_a.reshape(n_chunk, Q)

        def step(S, c):
            Bc, vc, dtc, lac = c
            cum = jnp.cumsum(lac)                 # (Q,), inclusive
            tot = cum[-1]
            w = jnp.exp(tot - cum) * dtc          # (Q,)
            return jnp.exp(tot) * S + Bc.T @ (w[:, None] * vc), None

        S, _ = jax.lax.scan(step, self.init_state(), (B, v, dt, log_a))
        return S

    def write_sequential(self, xs: jnp.ndarray,
                         mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """The naive per-token recurrence ``h_t = a_t h_{t-1} + B_t (Delta_t v_t)^T``.

        Kept as the **reference** the chunked path is tested against — the SSD
        claim is that the two are the same function, so we assert it rather than
        assume it.
        """
        xs = jnp.asarray(xs, dtype=jnp.float32)
        if int(xs.shape[0]) == 0:
            return self.init_state()
        B, v, dt, log_a = self._stream(xs, mask)

        def step(S, c):
            b, vv, d, la = c
            return jnp.exp(la) * S + jnp.outer(b, d * vv), None

        S, _ = jax.lax.scan(step, self.init_state(), (B, v, dt, log_a))
        return S

    def read(self, S: jnp.ndarray, xq: jnp.ndarray) -> jnp.ndarray:
        """``y = C_q^T h_T`` (+ the optional ``D`` skip / ``z`` gate) then ``theta_O``."""
        xq = jnp.asarray(xq, dtype=jnp.float32)
        C = xq @ self.theta_Q.T                              # (n, N)
        y = C @ S                                            # (n, P)
        if self.use_D:
            y = y + self.D * jax.vmap(lambda x: jax.nn.silu(self.theta_V @ x))(xq)
        if self.gate_z:
            y = y * jax.nn.silu(xq @ self.W_z.T)
        return y @ self.theta_O.T

    def read_dual(self, xs: jnp.ndarray, xq: jnp.ndarray,
                  mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """⭐ **The duality the paper is named for**, computed directly.

        Expanding the recurrence, the read is a *masked attention* over the stream::

            o_q = sum_j exp(cum_T - cum_j) (C_q . B_j) Delta_j v_j   (+ S_0 term)

        i.e. a dot-product kernel smoother with an exponential recency weighting —
        the quadratic form of the same linear recurrence. Asserted equal to
        :meth:`read` ∘ :meth:`write` in the tests; that assertion *is* the SSD
        property, verified rather than cited.
        """
        xs = jnp.asarray(xs, dtype=jnp.float32)
        xq = jnp.asarray(xq, dtype=jnp.float32)
        C = xq @ self.theta_Q.T
        if int(xs.shape[0]) == 0:
            return (C @ self.init_state()) @ self.theta_O.T
        B, v, dt, log_a = self._stream(xs, mask)
        cum = jnp.cumsum(log_a)
        w = jnp.exp(cum[-1] - cum) * dt                      # (T,)
        y = (C @ B.T * w[None, :]) @ v                       # (n, P)
        y = y + jnp.exp(cum[-1]) * (C @ self.init_state())
        if self.use_D:
            y = y + self.D * jax.vmap(lambda x: jax.nn.silu(self.theta_V @ x))(xq)
        if self.gate_z:
            y = y * jax.nn.silu(xq @ self.W_z.T)
        return y @ self.theta_O.T

    # -- the byte-matched table (P5's construction) --------------------------
    def kv_table(self, xs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """The ``(B_t, v_t) = (theta_K x_t, silu(theta_V x_t))`` pairs **as the
        recurrence consumes them** — the table stores exactly what the state is
        trying to store. ``Delta_t`` is the step size, not part of the value, so it
        is no more in the table than ``beta_t`` is in the delta arms'."""
        return jax.vmap(self._kv)(jnp.asarray(xs, dtype=jnp.float32))

    def query_keys(self, xq: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(xq, dtype=jnp.float32) @ self.theta_Q.T

    def decode_values(self, vals: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(vals, dtype=jnp.float32) @ self.theta_O.T

    # -- the ledger ----------------------------------------------------------
    @property
    def d_k(self) -> int:
        return int(self.d_state)

    @property
    def d_v(self) -> int:
        return int(self.d_head)

    def declared_state_floats(self) -> int:
        """``n_head * head_dim * d_state`` — the reference implementation's own
        ``ssm_state`` shape ``(B, nheads, headdim, d_state)`` (`rival-recon` §1.4).
        ⚠ The reference's ``conv_state`` is **excluded**: this arm has no conv
        branch, a deviation declared in the module docstring and in the ledger's
        ``state_convention``, and one that is in the rival's favour under an
        iso-state budget."""
        return int(self.n_head * self.d_head * self.d_state)

    def ledger(self, moved: Optional[int] = None) -> TwoSidedLedger:
        N, P, di, m = self.d_state, self.d_head, self.d_in, self.m
        pb = {"theta_K(B_t)": N * di, "theta_Q(C_t)": N * di,
              "theta_V(x_t)": P * di, "theta_O": m * P,
              "S0_init": N * P, "w_dt": di, "dt_bias": 1, "A_log": 1,
              "D(unused)" if not self.use_D else "D": P,
              "W_z(unused)" if not self.gate_z else "W_z": P * di}
        sb = {"ssm_state_deviation": int(self.n_head * P * N)}
        return TwoSidedLedger(
            arm="mamba2", param_floats=int(sum(pb.values())),
            state_floats=int(sum(sb.values())), param_breakdown=pb,
            state_breakdown=sb,
            state_convention=(
                "n_head*head_dim*d_state — the reference implementation's own "
                "ssm_state shape (B, nheads, headdim, d_state) with "
                "n_head = ngroups = 1 and d_state = head_dim = d, so the state is "
                "d^2, BYTE-IDENTICAL to the delta arms' n_head*d_k*d_v. "
                "⚠ conv_state ((d_inner + 2*ngroups*d_state)*d_conv) is EXCLUDED "
                "because this minimal arm has no conv branch — a declared "
                "deviation that is IN THE RIVAL'S FAVOUR under an iso-state "
                "budget (every byte goes to the SSM state)."),
            note=("S_0 is PARAMETERS (Mamba-2's published h_0 is the zero state; "
                  "it is learnable here exactly as the delta arms' S_0 is); only "
                  "the per-stream deviation is STATE"
                  + (f"; measured moved floats = {int(moved)}" if moved is not None
                     else "")),
        ).check()


def metric_native_verdict() -> Dict[str, Any]:
    """The equation-level argument (D3.6), stated before it is measured.

    ⭐ **Weaker than the delta arms', and the difference is the whole point of
    running this arm.**
    """
    return {
        "verdict": "metric-native (unnormalised)",
        "argument": (
            "SSD's scalar-identity restriction A_t = a_t I makes the read a "
            "1-semiseparable masked attention: expanding h_t = a_t h_{t-1} + "
            "B_t (Delta_t v_t)^T gives o_q = sum_j gamma_j (C_q . B_j) Delta_j v_j "
            "with gamma_j = prod_{r>j} a_r in (0,1] — a DOT-PRODUCT kernel smoother "
            "over the stored (B, v) pairs with an exponential recency weighting. "
            "That is a metric operation on its own state, so criterion 4 closes in "
            "the same sense it does for DeltaNet (Eq. 5) and Gated DeltaNet "
            "(Eq. 6) — of which Mamba-2 IS the erase-free degenerate case, per "
            "the Gated DeltaNet paper's own presentation of it as "
            "S_t = alpha_t S_{t-1} + v_t k_t^T. ⚠ BUT it closes WEAKLY: unlike "
            "GDN-2 §3.5, Mamba-2 does NOT L2-normalise its B/C paths, so "
            "arg-min ||q - k|| and arg-max q.k do NOT coincide in its own key "
            "space — the key-norm term survives. Its byte-matched table is "
            "therefore read by a WORSE-MATCHED arg-min reader than the delta "
            "arms', which inflates its dividend and its +0 B margin against that "
            "table while leaving the RAW-metric control (which does not use its "
            "projections at all) untouched. This is exactly the §4 "
            "projected-versus-raw finding, appearing again on a new family."),
        "measured_against": "its own byte-matched (theta_K x, theta_V x) table",
        "citation": ("Dao & Gu, ICML 2024, arXiv:2405.21060 (SSD); the "
                     "'Mamba2 as S_t = alpha_t S_{t-1} + v_t k_t^T' reading is "
                     "Gated DeltaNet's own, Yang, Kautz & Hatamizadeh, ICLR 2025, "
                     "arXiv:2412.06464"),
    }


__all__ = ["SSD_CHUNK", "A_INIT_RANGE", "DT_MIN", "DT_MAX", "Mamba2Memory",
           "metric_native_verdict"]
