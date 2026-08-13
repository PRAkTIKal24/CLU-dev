"""⭐ **Mamba-2 as a C3 tier-iii ladder arm** — the tuned SSM rival.

Dao & Gu, *"Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality"*, **ICML 2024, arXiv:2405.21060**;
official implementation ``state-spaces/mamba``, ``mamba_ssm/modules/mamba2.py``.

⛔ **This is NOT** :mod:`chlu.eval.rivals.mamba2`. That module is B′'s *memory-gym*
arm: a deliberately minimal single-head SSD recurrence with **no conv branch and
no gate**, sized to the gym's iso-state budget. This module is the **language-model
ladder arm**: the official block's inference recurrence — in-projection, causal
depthwise **conv branch**, selective ``(Delta, B, C)``, the SSD state update, the
``D`` skip, the **gated RMSNorm** and the out-projection — carrying the official
implementation's own two-tensor inference cache (``conv_state`` + ``ssm_state``),
which is the cache :data:`chlu.eval.byte_ledger.RIVAL_SPECS` ledgers.

⭐ **The anti-hobbling rule is this module's governing discipline** (task §0). A
rival that loses because we implemented it badly is worthless as a control, and
this program has already had one C2W10 verdict inverted by exactly that. So the
arm gets **every branch the official block has**, at the official defaults, and
the two places where the shell constrains it are named below rather than left for
a referee.

**Declared deviations, both structural to the swap and identical for every arm**

1. ⚠ **The seam is the shell's, not Mamba-2's.** The block hands every cell a
   pooled per-**chunk** latent ``z_c`` of width ``dim = addr_dim + payload_dim``
   and takes back a vector of the same width
   (:class:`~chlu.core.blocks.StreamBlock`). So this arm's in/out projections are
   ``dim -> d_in_proj`` and ``d_inner -> dim`` rather than ``d_model -> ...``,
   and one "token" of its recurrence is one chunk. ⛔ This is the *same* seam the
   CLU store, the GRU control and the TTT arm are given — changing it for one arm
   would end the swap. **Everything inside the cell is Mamba-2's own geometry.**
2. ⚠ **Read-before-write.** The shell reads the state that holds chunks
   ``0..c-2`` and *then* writes chunk ``c-1``, because a store that is read after
   being handed the current chunk is an echo (block docstring, step 3-before-4).
   Mamba-2's published output is ``y_t = C_t^T h_t`` with ``h_t`` *after* the
   update; here it is ``C_t^T h_{t-1}``. ⭐ The current chunk still reaches the
   read through the ``D`` skip and the ``z`` gate, exactly as in the official
   block, so the deviation costs the arm only the current-step SSM term — and it
   costs the CLU arm the same one.

**Provenance discipline (⛔ TRAP 2, scout §1.5).** Every state-bearing
hyperparameter is pinned in :class:`Mamba2ArmConfig` and carries a per-number
string in :data:`MAMBA2_PROVENANCE`. ⚠ On the record, and unlike GDN-2's row:
these numbers come from the **official implementation**, *not* a paper table —
the scout could not obtain arXiv:2405.21060's per-size appendix (the PDF would
not parse; ar5iv returned front matter only), so every string here begins
``OFFICIAL IMPLEMENTATION:`` and none begins ``PAPER:``. If that appendix is ever
reached, reconcile and report the disagreement as a finding.

**The shrink is solved, not chosen.** The pinned cell is **3.09x** the ruled
2 MiB ceiling, and :func:`chlu.eval.byte_ledger.shrink_to_budget` solves the
declared knob **down**: ``d_state 128 -> 39``. ⛔ That value is imported, never
re-derived here (task §1.4).

**Dtype.** ⛔ TOTAL state bytes **as deployed, with no dtype normalisation**
(Head+Advisor). Mamba-2 deploys its inference cache in the model dtype (bf16 in
the official released models) while our store is float32; the asymmetry is real,
it favours the rival, and it stays.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from chlu.eval.byte_ledger import (
    MATCHED_STATE_BYTE_BUDGET,
    RIVAL_SPECS,
    shrink_to_budget,
)
from chlu.eval.rivals.c3_registry import C3RivalArm, register_c3_rival

#: The key this arm must reproduce **to the byte**.
SPEC_NAME = "mamba2"

#: ⭐ The knob solved DOWN by the harness, imported rather than re-solved.
_SHRINK = shrink_to_budget(SPEC_NAME, MATCHED_STATE_BYTE_BUDGET)
PUBLISHED_D_STATE: int = int(RIVAL_SPECS[SPEC_NAME].params["d_state"])
SHRUNK_D_STATE: int = int(_SHRINK["shrunk_value"])

#: Official-implementation init ranges (``mamba_ssm/modules/mamba2.py``).
A_INIT_RANGE = (1.0, 16.0)
DT_MIN, DT_MAX, DT_INIT_FLOOR = 1e-3, 1e-1, 1e-4
NORM_EPS = 1e-5

#: ⛔ Per-number provenance. A test asserts every string starts with ``PAPER:``
#: or ``OFFICIAL IMPLEMENTATION:`` — the same assertion ``RIVAL_SPECS`` carries.
MAMBA2_PROVENANCE: Dict[str, str] = {
    "d_model": (
        "OFFICIAL IMPLEMENTATION: the rival's own reference width for the "
        "26-47 M recurrent class (c3-benchmark-scout §1.5: 24 L x d_model 512, "
        "~6 d_model^2 params/layer => 37.75 M). ⛔ NOT inherited from the shell's "
        "d_model: the arm's internal geometry is Mamba-2's, the seam is the "
        "shell's."),
    "n_layers": (
        "OFFICIAL IMPLEMENTATION: 24 layers is the scout §1.5 recurrent-class "
        "reference this arm's pinned byte count is quoted at. ⚠ The DEPLOYED "
        "layer count is the shell's (PilotConfig.n_layers) and is ledgered "
        "separately — see reference_row()."),
    "d_state": (
        "OFFICIAL IMPLEMENTATION: state-spaces/mamba, mamba_ssm/modules/mamba2.py "
        "default d_state=128; allocate_inference_cache allocates "
        "(batch, nheads, headdim, d_state). ⭐ SHRUNK to fit the ruled 2 MiB "
        "ceiling by chlu.eval.byte_ledger.shrink_to_budget('mamba2') "
        f"({PUBLISHED_D_STATE} -> {SHRUNK_D_STATE}); ⛔ solved, never hand-chosen."),
    "d_conv": (
        "OFFICIAL IMPLEMENTATION: mamba2.py default d_conv=4 (the causal "
        "depthwise conv width). conv_state holds the d_conv-1 PAST taps; the "
        "current input is not state."),
    "expand": "OFFICIAL IMPLEMENTATION: mamba2.py default expand=2 => d_inner = 2*d_model.",
    "headdim": "OFFICIAL IMPLEMENTATION: mamba2.py default headdim=64 => nheads = d_inner/64.",
    "ngroups": (
        "OFFICIAL IMPLEMENTATION: mamba2.py default ngroups=1 (one shared (B,C) "
        "group) => conv_dim = d_inner + 2*ngroups*d_state."),
    "dtype_bytes": (
        "OFFICIAL IMPLEMENTATION: the inference cache is allocated in the model "
        "dtype (bf16 in the released models), and c3-benchmark-scout §1.5 quotes "
        "the rival table at bf16 = 2 B/element. ⛔ Convention (Head+Advisor): "
        "TOTAL state bytes AS DEPLOYED, no dtype normalisation — our store's "
        "float32 vs the rival's bf16 is a real asymmetry in the RIVAL's favour "
        "and it stays."),
    "conv_bias": "OFFICIAL IMPLEMENTATION: mamba2.py default conv_bias=True.",
    "proj_bias": "OFFICIAL IMPLEMENTATION: mamba2.py default bias=False on in_proj/out_proj.",
    "rmsnorm": (
        "OFFICIAL IMPLEMENTATION: mamba2.py default rmsnorm=True — RMSNormGated("
        "d_inner, norm_before_gate=False, group_size=d_inner//ngroups)."),
    "use_D": "OFFICIAL IMPLEMENTATION: mamba2.py allocates D (nheads,) with D_has_hdim=False.",
    "learnable_init_state": (
        "OFFICIAL IMPLEMENTATION: Mamba-2's published h_0 is the zero state; it "
        "is a learnable PARAMETER here exactly as the GRU control's h0, the TTT "
        "arm's W0 and the CLU store's V0 are (PREREG-Bprime §4.1: an "
        "initialisation is PARAMETERS, only the per-stream deviation is STATE)."),
}


class Mamba2ArmConfig(NamedTuple):
    """The pinned, state-bearing geometry. ⛔ No field here has a library default
    that was not read off ``mamba_ssm/modules/mamba2.py`` — see
    :data:`MAMBA2_PROVENANCE`, one string per number."""

    d_model: int = 512
    n_layers: int = 24
    d_state: int = SHRUNK_D_STATE
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    dtype_bytes: int = 2
    conv_bias: bool = True
    proj_bias: bool = False
    rmsnorm: bool = True
    use_D: bool = True

    # -- derived geometry ---------------------------------------------------
    @property
    def d_inner(self) -> int:
        return int(self.expand) * int(self.d_model)

    @property
    def n_heads(self) -> int:
        return self.d_inner // int(self.headdim)

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * int(self.ngroups) * int(self.d_state)

    @property
    def d_in_proj(self) -> int:
        return 2 * self.d_inner + 2 * int(self.ngroups) * int(self.d_state) \
            + self.n_heads

    def state_elements_per_layer(self) -> int:
        """``nheads*headdim*d_state + conv_dim*(d_conv-1)`` — the official
        implementation's own two-tensor inference cache, and **exactly** the
        formula :data:`chlu.eval.byte_ledger.RIVAL_SPECS` uses."""
        return int(self.n_heads * int(self.headdim) * int(self.d_state)
                   + self.conv_dim * (int(self.d_conv) - 1))

    def state_bytes_per_layer(self) -> int:
        return int(self.state_elements_per_layer() * int(self.dtype_bytes))

    def total_state_bytes(self, n_layers: Optional[int] = None) -> int:
        n = int(self.n_layers if n_layers is None else n_layers)
        return int(n * self.state_bytes_per_layer())


def resolve_config(overrides: Any = None) -> Mamba2ArmConfig:
    """The pinned config with declared overrides applied, **validated loudly**.

    ⛔ An unknown key is an error: a typo'd override that silently did nothing
    would be a config claimed but not run.
    """
    ov = dict(overrides or {})
    if isinstance(ov.get(SPEC_NAME), dict):      # a per-arm block is allowed
        ov = dict(ov[SPEC_NAME])
    cfg = Mamba2ArmConfig()
    unknown = set(ov) - set(cfg._fields)
    if unknown:
        raise ValueError(
            f"unknown mamba2 arm config key(s) {sorted(unknown)}; "
            f"known: {list(cfg._fields)}")
    cfg = cfg._replace(**{k: (bool(v) if isinstance(getattr(cfg, k), bool)
                              else int(v)) for k, v in ov.items()})
    if cfg.d_inner % int(cfg.headdim):
        raise ValueError(
            f"mamba2: d_inner = expand*d_model = {cfg.d_inner} is not divisible "
            f"by headdim={cfg.headdim} — the official block requires "
            "d_inner % headdim == 0; refusing to silently round.")
    if cfg.n_heads < 1:
        raise ValueError(
            f"mamba2: nheads = d_inner/headdim = {cfg.n_heads} < 1 at "
            f"d_model={cfg.d_model}, expand={cfg.expand}, headdim={cfg.headdim}.")
    if cfg.n_heads % int(cfg.ngroups):
        raise ValueError(
            f"mamba2: nheads={cfg.n_heads} is not divisible by "
            f"ngroups={cfg.ngroups}.")
    if int(cfg.d_conv) < 1 or int(cfg.d_state) < 1 or int(cfg.dtype_bytes) < 1:
        raise ValueError(f"mamba2: non-positive geometry in {cfg}")
    return cfg


class Mamba2CellState(NamedTuple):
    """The official implementation's **two-tensor inference cache**.

    ``ssm``  ``(nheads, headdim, d_state)`` — ``allocate_inference_cache``'s
    ``ssm_state``; ``conv`` ``(d_conv-1, conv_dim)`` — the retained window of the
    causal depthwise conv.

    ⚠ The reference caches ``d_conv`` columns because it writes the *current*
    input into the last slot before convolving; the genuinely **retained** state
    is the ``d_conv-1`` past taps, which is what ``RIVAL_SPECS``' formula
    ``(d_inner + 2*ngroups*d_state)*(d_conv-1)`` counts and what is held here.
    """

    ssm: jnp.ndarray
    conv: jnp.ndarray


class Mamba2Cell(eqx.Module):
    """Mamba-2's block, as a stream-block memory cell.

    One ``read``/``write`` pair per chunk **is** the official ``step()``: roll the
    conv window, ``silu`` the depthwise conv, split ``(x, B, C)``, discretise with
    ``Delta_t = softplus(w_dt . z + dt_bias)`` and ``a_t = exp(Delta_t A)``,
    ``A = -exp(A_log)``, update ``h <- a h + (Delta x) B^T``, read
    ``y = h C + D x``, gate-and-RMSNorm with ``z``, project out.
    """

    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    conv_w: jnp.ndarray
    conv_b: jnp.ndarray
    A_log: jnp.ndarray
    dt_bias: jnp.ndarray
    D: jnp.ndarray
    norm_w: jnp.ndarray
    ssm0: jnp.ndarray
    conv0: jnp.ndarray

    cfg: Mamba2ArmConfig = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, cfg: Mamba2ArmConfig, *, latent_dim: int, key):
        ks = jax.random.split(key, 6)
        self.cfg = cfg
        self.latent_dim = int(latent_dim)
        H, P, N = cfg.n_heads, int(cfg.headdim), int(cfg.d_state)
        # in/out projections: torch nn.Linear's default init is
        # U(-1/sqrt(fan_in), 1/sqrt(fan_in)), which is eqx.nn.Linear's default.
        self.in_proj = eqx.nn.Linear(self.latent_dim, cfg.d_in_proj,
                                     use_bias=bool(cfg.proj_bias), key=ks[0])
        self.out_proj = eqx.nn.Linear(cfg.d_inner, self.latent_dim,
                                      use_bias=bool(cfg.proj_bias), key=ks[1])
        # depthwise causal conv over the (x, B, C) channels, torch Conv1d init.
        s = 1.0 / float(int(cfg.d_conv)) ** 0.5
        self.conv_w = jax.random.uniform(
            ks[2], (cfg.conv_dim, int(cfg.d_conv)), minval=-s, maxval=s)
        self.conv_b = (jax.random.uniform(ks[3], (cfg.conv_dim,), minval=-s,
                                          maxval=s)
                       if cfg.conv_bias else jnp.zeros((cfg.conv_dim,)))
        # A ~ U(1, 16) per head, stored as log A (A_real = -A).
        a = jax.random.uniform(ks[4], (H,), minval=A_INIT_RANGE[0],
                               maxval=A_INIT_RANGE[1])
        self.A_log = jnp.log(a)
        # dt ~ exp(U(log dt_min, log dt_max)), floored, inverse-softplussed.
        u = jax.random.uniform(ks[5], (H,))
        dt = jnp.exp(u * (jnp.log(DT_MAX) - jnp.log(DT_MIN)) + jnp.log(DT_MIN))
        dt = jnp.clip(dt, DT_INIT_FLOOR, None)
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))          # inverse softplus
        self.D = jnp.ones((H,))
        self.norm_w = jnp.ones((cfg.d_inner,))
        # ⭐ the learned initial state: PARAMETERS (Mamba-2's published h_0 is
        # the zero state), only the per-stream deviation is STATE.
        self.ssm0 = jnp.zeros((H, P, N))
        self.conv0 = jnp.zeros((int(cfg.d_conv) - 1, cfg.conv_dim))

    # -- the official step, split across the shell's read/write ---------------
    def init_state(self) -> Mamba2CellState:
        return Mamba2CellState(ssm=self.ssm0, conv=self.conv0)

    def _project(self, z):
        """``in_proj`` then the official split ``[z_gate, xBC, dt]``."""
        zxbcdt = self.in_proj(jnp.asarray(z, jnp.float32))
        di, cd = self.cfg.d_inner, self.cfg.conv_dim
        return zxbcdt[:di], zxbcdt[di: di + cd], zxbcdt[di + cd:]

    def _conv(self, conv_state, xBC):
        """``silu(depthwise_conv([past taps ; current]))`` + the rolled window."""
        win = jnp.concatenate([conv_state, xBC[None, :]], axis=0)   # (d_conv, cd)
        y = jnp.sum(win * self.conv_w.T, axis=0) + self.conv_b
        return jax.nn.silu(y), win[1:]

    def _split_xbc(self, xbc):
        di, gn = self.cfg.d_inner, int(self.cfg.ngroups) * int(self.cfg.d_state)
        return xbc[:di], xbc[di: di + gn], xbc[di + gn:]

    def _per_head(self, g):
        """``(ngroups*d_state,) -> (nheads, d_state)`` — heads share their group's
        ``B``/``C``, which is what ``ngroups < nheads`` means."""
        G, H, N = int(self.cfg.ngroups), self.cfg.n_heads, int(self.cfg.d_state)
        return jnp.repeat(g.reshape(G, N), H // G, axis=0)

    def _gated_norm(self, y, z_gate):
        """``RMSNormGated(norm_before_gate=False, group_size=d_inner//ngroups)``."""
        if not self.cfg.rmsnorm:
            return y * jax.nn.silu(z_gate)
        h = y * jax.nn.silu(z_gate)
        G = int(self.cfg.ngroups)
        hg = h.reshape(G, self.cfg.d_inner // G)
        hg = hg * jax.lax.rsqrt(jnp.mean(hg * hg, axis=-1, keepdims=True) + NORM_EPS)
        return hg.reshape(self.cfg.d_inner) * self.norm_w

    def read(self, state: Mamba2CellState, z, plan_c=None, read_mode=None,
             verlet=None):
        """``y = C_t^T h + D x_t``, gated-RMSNormed, projected back to the seam.

        ⚠ ``h`` is the state **before** this chunk's write (the shell's
        read-before-write contract, identical for every arm); the current chunk
        still enters through ``D x_t`` and the ``z`` gate, as in the official
        block.
        """
        z_gate, xBC, _ = self._project(z)
        xbc, _ = self._conv(state.conv, xBC)
        x, _, C = self._split_xbc(xbc)
        H, P = self.cfg.n_heads, int(self.cfg.headdim)
        xh = x.reshape(H, P)
        y = jnp.einsum("hpn,hn->hp", state.ssm, self._per_head(C))
        if self.cfg.use_D:
            y = y + self.D[:, None] * xh
        return self.out_proj(self._gated_norm(y.reshape(self.cfg.d_inner), z_gate))

    def write(self, state: Mamba2CellState, z, plan_c=None) -> Mamba2CellState:
        """The SSD update ``h <- exp(Delta A) h + (Delta x) B^T`` + the conv roll.

        ``plan_c`` is the CLU controller's :class:`~chlu.core.blocks.WritePlan`
        and is **ignored** here, exactly as the GRU and TTT arms ignore it: a
        rival's write policy is its own.
        """
        _, xBC, dt_raw = self._project(z)
        xbc, conv_next = self._conv(state.conv, xBC)
        x, B, _ = self._split_xbc(xbc)
        H, P = self.cfg.n_heads, int(self.cfg.headdim)
        dt = jax.nn.softplus(dt_raw + self.dt_bias)                  # (H,)
        dA = jnp.exp(dt * (-jnp.exp(self.A_log)))                    # (H,)
        xh = x.reshape(H, P) * dt[:, None]                           # (H,P)
        ssm = state.ssm * dA[:, None, None] \
            + xh[:, :, None] * self._per_head(B)[:, None, :]
        return Mamba2CellState(ssm=ssm, conv=conv_next)

    # -- the ledger ----------------------------------------------------------
    def cell_ledger(self) -> Dict[str, Any]:
        """⭐ The row the arm enters the ladder through.

        ``state_bytes`` is ``dtype_bytes * state_floats`` **as deployed** (bf16),
        NOT float32: the harness's :func:`chlu.eval.byte_ledger.arm_ledger`
        multiplies this by ``n_layers`` to get the total, so the deployed dtype
        must be applied here or the rival would be charged our store's width.
        """
        cfg = self.cfg
        p = int(sum(x.size for x in jax.tree_util.tree_leaves(
            eqx.filter(self, eqx.is_inexact_array))))
        H, P, N = cfg.n_heads, int(cfg.headdim), int(cfg.d_state)
        return {
            "params": p,
            "state_floats": cfg.state_elements_per_layer(),
            "state_bytes": cfg.state_bytes_per_layer(),
            "dtype_bytes": int(cfg.dtype_bytes),
            "state_breakdown": {"ssm_state": int(H * P * N),
                                "conv_state": int(cfg.conv_dim
                                                  * (int(cfg.d_conv) - 1))},
            "state_convention": (
                "nheads*headdim*d_state + (d_inner + 2*ngroups*d_state)*(d_conv-1)"
                " — the official implementation's own two-tensor inference cache "
                "(mamba_ssm/modules/mamba2.py: allocate_inference_cache). "
                "dtype = bf16 AS DEPLOYED (no dtype normalisation)."),
            "rival_spec": SPEC_NAME,
            "rival_config": dict(cfg._asdict()),
            "provenance": dict(MAMBA2_PROVENANCE),
            "learned_init_is_params": True,
        }


# ==========================================================================
# the reference row — the pinned arithmetic, reproduced from OUR config
# ==========================================================================
def reference_row(budget: int = MATCHED_STATE_BYTE_BUDGET) -> Dict[str, Any]:
    """Published vs shrunk, computed from :class:`Mamba2ArmConfig` itself.

    ⭐ The point of computing it here rather than reading ``RIVAL_SPECS`` is that
    the two must **agree to the byte**; a test asserts it, so a drift in either
    place is caught rather than reported.
    """
    pinned = Mamba2ArmConfig()
    published = pinned._replace(d_state=PUBLISHED_D_STATE)
    return {
        "rival": SPEC_NAME,
        "citation": "Dao & Gu, ICML 2024, arXiv:2405.21060; state-spaces/mamba",
        "provenance_kind": "OFFICIAL IMPLEMENTATION",
        "paper_appendix_obtained": False,
        "reference_n_layers": int(pinned.n_layers),
        "dtype_bytes": int(pinned.dtype_bytes),
        "formula": ("n_L * (nheads*headdim*d_state + "
                    "(d_inner + 2*ngroups*d_state)*(d_conv-1))"),
        "published": {
            "config": dict(published._asdict()),
            "state_bytes_per_layer": published.state_bytes_per_layer(),
            "total_state_bytes": published.total_state_bytes(),
            "occupancy": published.total_state_bytes() / float(budget),
        },
        "shrunk": {
            "config": dict(pinned._asdict()),
            "knob": _SHRINK["knob"],
            "published_value": PUBLISHED_D_STATE,
            "shrunk_value": SHRUNK_D_STATE,
            "state_bytes_per_layer": pinned.state_bytes_per_layer(),
            "total_state_bytes": pinned.total_state_bytes(),
            "occupancy": pinned.total_state_bytes() / float(budget),
        },
        "shrink_solution": dict(_SHRINK),
        "budget_bytes": int(budget),
        "provenance": dict(MAMBA2_PROVENANCE),
    }


def deployed_row(cfg: Mamba2ArmConfig, n_layers: int,
                 budget: int = MATCHED_STATE_BYTE_BUDGET) -> Dict[str, Any]:
    """This arm's row **as the shell actually deploys it** (``n_layers`` cells).

    ⚠ The shell is the scout's *attention-class* reference (12 L), while the
    shrink knob was solved on the rival's own *recurrent-class* reference (24 L),
    so the deployed occupancy is about half the ceiling. Reported, never
    silently re-solved (task §1.4).
    """
    tot = cfg.total_state_bytes(n_layers)
    return {
        "rival": SPEC_NAME, "deployed_n_layers": int(n_layers),
        "state_bytes_per_layer": cfg.state_bytes_per_layer(),
        "total_state_bytes": tot, "budget_bytes": int(budget),
        "occupancy": tot / float(budget), "within_budget": tot <= int(budget),
        "reference_n_layers": int(cfg.n_layers),
        "reference_total_state_bytes": cfg.total_state_bytes(),
    }


def build_cell(cfg: Mamba2ArmConfig, *, latent_dim: int, key) -> Mamba2Cell:
    return Mamba2Cell(cfg, latent_dim=int(latent_dim), key=key)


MAMBA2_ARM = register_c3_rival(C3RivalArm(
    name="mamba2", spec_name=SPEC_NAME, resolve=resolve_config,
    build=build_cell, reference_row=reference_row, deployed_row=deployed_row))


__all__ = [
    "SPEC_NAME", "PUBLISHED_D_STATE", "SHRUNK_D_STATE", "MAMBA2_PROVENANCE",
    "Mamba2ArmConfig", "Mamba2CellState", "Mamba2Cell", "resolve_config",
    "build_cell", "reference_row", "deployed_row", "MAMBA2_ARM",
]
