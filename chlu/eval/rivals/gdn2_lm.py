"""⭐⭐ **Gated DeltaNet-2 as a C3 Track-A LANGUAGE-MODEL ARM** (arXiv:2605.22791).

`c3-rival-arms` §B, phase 2. This is **not** the B′ gym memory in
:mod:`chlu.eval.rivals.deltanet` (``DeltaMemory(variant="gdn2")``, one head, no
convolution, a toy-harness ``read``/``write`` pair). This is the **rival LM**: a
stack of GDN-2 token mixers that trains on a real byte stream and produces a byte
ledger the C3 matched-state-byte control can consume.

⛔ **Not GDN v1** (arXiv:2412.06464). The ids are a known confusion; the ``2605``
id is carried on every provenance string below.

**Anti-hobbling is the governing discipline** (task §0): a rival that loses
because we implemented it badly is worthless as a control. So the port follows
the authors' own artefacts, obtained and read directly:

=================================  =========================================
`lit_gpt/gdn2.py`                  ``GatedDeltaNet2.__init__`` / ``forward`` —
                                   projections, short convolutions, gate
                                   construction, output gate, init
`lit_gpt/gdn2_ops/`                ``fused_recurrent_gdn2`` — the **reference
`fused_recurrent_gdn2.py`**        recurrence**, ported statement for
                                   statement into :func:`GDN2Layer._recur`
`lit_gpt/model.py`                 ``Block`` (pre-norm, non-parallel residual),
                                   ``LLaMAMLP`` (SwiGLU), ``_init_weights``
`lit_gpt/config.py`                ``gdn2_1.3B`` — the released geometry
paper §3.1 Eqs. 8-12, §3.5,        the equations and the block design
App. C.1, D.2, D.5, E.1 Eq. 90
=================================  =========================================

⚠ **What is a shim and what is faithful.** The Triton kernels are replaced by a
``jax.lax.scan`` over the *same* recurrence — the kernel's own statements, in the
kernel's own order, including its ``1e-6`` L2 epsilon and its ``d_k**-0.5`` query
scale (:func:`GDN2Layer._recur`; the equivalence is pinned by tests against the
paper's Eq. 9/Eq. 10 forms *and* against the kernel's line-by-line arithmetic).
The chunkwise WY algorithm (§3.3) is a **training-throughput** device that
computes the same function as the sequential recurrence, so not porting it costs
speed, never fidelity. ⛔ Grouped value attention (``num_v_heads > num_heads``)
is **not implemented** and is refused loudly rather than silently ignored; the
official default is ``num_v_heads = num_heads``.

⛔⛔ **THE `flash-linear-attention` 3x TRAP** (task §1.2). FLA's
``fla/layers/gated_deltanet.py`` defaults are ``hidden_size=2048, expand_v=2,
head_dim=256, num_heads=6`` ⇒ ``H*d_k*d_v = 6*256*512 = 786,432`` floats per
layer = **3.000x** the GDN-2 paper's own 262,144. **No default in this module is
inherited from any library.** Every state-bearing number is pinned in
:class:`GDN2Config` and carries a provenance string in :data:`GDN2_PROVENANCE`;
:func:`fla_trap_check` re-derives the 3x rather than quoting it.

⭐ **The arm enters the ladder only through the byte ledger**
(:func:`gdn2_ledger_row`), which reproduces
``chlu.eval.byte_ledger.RIVAL_SPECS["gated_deltanet2"]`` **to the byte** at the
published pin and uses :func:`~chlu.eval.byte_ledger.shrink_to_budget`'s solved
knob — never a hand-solved one — when shrinking to the ceiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.eval.byte_ledger import (
    BF16_BYTES,
    MATCHED_STATE_BYTE_BUDGET,
    RIVAL_SPECS,
    LedgerError,
    shrink_to_budget,
)

#: The paper. Carried on every provenance string so the v1/-2 confusion cannot
#: survive a grep.
GDN2_ARXIV = "arXiv:2605.22791"
GDN2_CITATION = (
    "Hatamizadeh, Choi, Kautz (2026), *Gated DeltaNet-2: Decoupling Erase and "
    "Write in Linear Attention*, arXiv:2605.22791 (21 May 2026, NVIDIA); code "
    "https://github.com/NVlabs/GatedDeltaNet-2. ⛔ NOT arXiv:2412.06464 (GDN v1)."
)

#: ⛔ The name in :data:`chlu.eval.byte_ledger.RIVAL_SPECS` this arm must match.
RIVAL_SPEC_NAME = "gated_deltanet2"

#: Admissible provenance prefixes. ``PAPER:`` and ``OFFICIAL IMPLEMENTATION:``
#: are the task's two (§1.2). ⚠ ``HARNESS LEDGER:`` is a **third, declared**
#: prefix for the two numbers that are *ours*, not the rival's — ``n_layers`` and
#: ``d_model`` are the weight class WE chose (scout §1.5's recurrent-class
#: reference, pinned into ``RIVAL_SPECS``), and labelling them ``PAPER:`` would
#: be a false citation. They are state-bearing, so they are pinned and sourced
#: like everything else; they are simply sourced to us.
PROVENANCE_PREFIXES = ("PAPER:", "OFFICIAL IMPLEMENTATION:", "HARNESS LEDGER:")

#: ⭐ Per-number provenance for every pinned hyperparameter. A test asserts that
#: every field of :class:`GDN2Config` appears here and that every string starts
#: with one of :data:`PROVENANCE_PREFIXES`.
GDN2_PROVENANCE: Dict[str, str] = {
    "n_layers": (
        "HARNESS LEDGER: chlu.eval.byte_ledger.RIVAL_SPECS['gated_deltanet2']"
        ".n_layers = 24 — the recurrent-class reference config of "
        "`c3-benchmark-scout` §1.5 (24 L, d_model 512), which is OUR weight-class "
        "choice, not a number from arXiv:2605.22791. ⚠ See GDN2_PARAM_CLASS_NOTE."
    ),
    "d_model": (
        "HARNESS LEDGER: RIVAL_SPECS['gated_deltanet2'].d_model = 512, same "
        "source and same caveat as n_layers. (The paper's own setting is "
        "d_model 2048 at 1.3 B; the released config gdn2_1.3B is n_embd 2304.)"
    ),
    "n_heads": (
        "PAPER: arXiv:2605.22791 §E.1 Eq. 90 — 'Gated DeltaNet, KDA, and Gated "
        "DeltaNet-2 use H=16 heads with d_k=128 and d_v=128, giving a per-layer "
        "recurrent state of H d_k d_v = 16*128*128 = 262,144 floats per batch "
        "element. Since d_model=2048, this equals 128 d_model.' Scaled to "
        "d_model 512 at H=4, which holds d_k=d_v=128 exactly. ⛔ NOT the "
        "flash-linear-attention default num_heads=6 (see fla_trap_check)."
    ),
    "head_dim": (
        "PAPER: arXiv:2605.22791 §E.1 Eq. 90 — d_k = d_v = 128. ``None`` means "
        "'derive as d_model // n_heads', which is RIVAL_SPECS' own tie "
        "(formula 'n_L * d_model^2 / H') and reproduces the paper's 128 exactly "
        "at d_model 512 / H 4. ⛔ NOT the flash-linear-attention default 256."
    ),
    "expand_v": (
        "OFFICIAL IMPLEMENTATION: NVlabs/GatedDeltaNet-2 lit_gpt/gdn2.py, "
        "GatedDeltaNet2.__init__ default expand_v=1 (=> head_v_dim == head_dim, "
        "d_v = d_k = 128, matching PAPER Eq. 90). ⛔ NOT the "
        "flash-linear-attention GatedDeltaNet default expand_v=2, which is the "
        "other half of the 3x trap."
    ),
    "num_v_heads": (
        "OFFICIAL IMPLEMENTATION: lit_gpt/gdn2.py — "
        "`num_v_heads = num_v_heads if num_v_heads is not None else num_heads`. "
        "``None`` = no grouped value attention, the shipped default."
    ),
    "use_short_conv": (
        "OFFICIAL IMPLEMENTATION: lit_gpt/gdn2.py default use_short_conv=True; "
        "PAPER §3.5 / Fig. 1 — 'query and key paths use linear projection, short "
        "convolution, SiLU, and L2 normalization. The value path uses linear "
        "projection, short convolution, and SiLU.'"
    ),
    "conv_size": (
        "OFFICIAL IMPLEMENTATION: lit_gpt/gdn2.py default conv_size=4."
    ),
    "conv_bias": (
        "OFFICIAL IMPLEMENTATION: lit_gpt/gdn2.py default conv_bias=False."
    ),
    "allow_neg_eigval": (
        "OFFICIAL IMPLEMENTATION: lit_gpt/gdn2.py default allow_neg_eigval=False; "
        "PAPER §3.1 / App. C.1 — 'We also support the negative-eigenvalue variant "
        "of [20] by scaling only the erase gate to [0,2]^{d_k}. The write gate "
        "remains in [0,1]^{d_v}.' Byte-inert (it scales a gate, not a state)."
    ),
    "use_mlp": (
        "PAPER: arXiv:2605.22791 §3.5 — 'The recurrent model stacks GDN-2 mixers "
        "and MLPs under the standard residual block'; OFFICIAL IMPLEMENTATION: "
        "lit_gpt/config.py gdn2_1.3B sets _mlp_class='LLaMAMLP'. Byte-inert "
        "(the MLP holds no inference state). ⚠ GDN2_PARAM_CLASS_NOTE."
    ),
    "mlp_intermediate": (
        "OFFICIAL IMPLEMENTATION: lit_gpt/config.py gdn2_1.3B — n_embd=2304, "
        "intermediate_size=6208 => ratio 2.69444…; ``None`` derives "
        "round(d_model * 6208/2304). Byte-inert."
    ),
    "norm_eps": (
        "OFFICIAL IMPLEMENTATION: lit_gpt/config.py gdn2_1.3B norm_eps=1e-5 "
        "(and lit_gpt/gdn2.py GatedDeltaNet2 norm_eps=1e-5)."
    ),
    "init_gain": (
        "PAPER: arXiv:2605.22791 App. D.5 — 'All linear layers are initialized "
        "with Xavier uniform "
        "weights and gain 2^{-2.5}'; OFFICIAL IMPLEMENTATION: lit_gpt/gdn2.py "
        "GatedDeltaNet2._initialize_weights."
    ),
    "embed_init_std": (
        "OFFICIAL IMPLEMENTATION: lit_gpt/model.py GPT._init_weights with "
        "mamba_init=True (set by config gdn2_1.3B) — "
        "`torch.nn.init.normal_(module.weight, std=0.02)` for nn.Embedding."
    ),
    "vocab_size": (
        "HARNESS LEDGER: the Track-A byte stream's vocabulary "
        "(chlu.data.enwik8.VOCAB_SIZE = 256). Byte-inert."
    ),
    "dtype_bytes": (
        "HARNESS LEDGER: task §1.5 — 'TOTAL state bytes, AS DEPLOYED, no dtype "
        "normalisation'. GDN-2 deploys at bf16 (2 B/element), which is what "
        "RIVAL_SPECS records (BF16_BYTES) and what the scout's table is quoted "
        "in. ⛔ Our own store is fp32; that asymmetry is real and stays."
    ),
}

#: ⚠⚠ **A conflict the ladder must resolve; NOT resolvable here.** See the module
#: report §Findings and :func:`gdn2_param_class_table`.
GDN2_PARAM_CLASS_NOTE = (
    "The 24-layer pin and the 26-47 M weight class are jointly infeasible with "
    "the paper's own recurrent block. A GDN-2 token mixer is ~7.03*d_model^2 "
    "params (6*d^2 of q/k/v/b/w/o plus f_proj/g_proj/conv/A_log/dt_bias); adding "
    "the LLaMAMLP at the released ratio makes a block ~15.1*d_model^2, so 24 L at "
    "d_model 512 is ~95.4 M — 2.0x the class ceiling. The scout's "
    "'recurrent class ~= 6*d_model^2 per layer => 37.75 M at 24 L' is only "
    "reachable with an MLP-FREE stack (44.5 M here). Both resolutions move a "
    "claims-relevant axis (params, or n_layers and hence state bytes), so this "
    "module ships `use_mlp` as a knob at the FAITHFUL default and reports the "
    "arithmetic instead of choosing."
)


# ==========================================================================
# the pinned config
# ==========================================================================
@dataclass(frozen=True)
class GDN2Config:
    """Every state-bearing hyperparameter of the GDN-2 arm, pinned.

    ⛔ No field here may be filled from a library default. :data:`GDN2_PROVENANCE`
    carries one string per field and a test asserts the coverage and the prefix.
    """

    # -- state-bearing ------------------------------------------------------
    n_layers: int = 24
    d_model: int = 512
    n_heads: int = 4
    #: ``None`` => ``d_model // n_heads`` (RIVAL_SPECS' own ``d_model^2/H`` tie).
    head_dim: Optional[int] = None
    expand_v: float = 1.0
    num_v_heads: Optional[int] = None
    use_short_conv: bool = True
    conv_size: int = 4
    dtype_bytes: int = BF16_BYTES
    # -- byte-inert ---------------------------------------------------------
    conv_bias: bool = False
    allow_neg_eigval: bool = False
    use_mlp: bool = True
    mlp_intermediate: Optional[int] = None
    norm_eps: float = 1e-5
    init_gain: float = 2.0 ** -2.5
    embed_init_std: float = 0.02
    vocab_size: int = 256

    # -- derived ------------------------------------------------------------
    @property
    def head_k_dim(self) -> int:
        if self.head_dim is not None:
            return int(self.head_dim)
        if self.d_model % self.n_heads:
            raise LedgerError(
                f"⛔ d_model={self.d_model} is not divisible by n_heads="
                f"{self.n_heads}, so the RIVAL_SPECS tie d_k = d_model/H has no "
                f"integer realisation. floor() would silently change the byte "
                f"ledger; pass head_dim explicitly and declare it instead."
            )
        return int(self.d_model // self.n_heads)

    @property
    def head_v_dim(self) -> int:
        hv = self.head_k_dim * self.expand_v
        if abs(hv - round(hv)) > 1e-9:
            raise LedgerError(
                f"⛔ expand_v={self.expand_v} x head_dim={self.head_k_dim} is not "
                f"an integer (official lit_gpt/gdn2.py raises the same way)."
            )
        return int(round(hv))

    @property
    def n_v_heads(self) -> int:
        return int(self.num_v_heads) if self.num_v_heads is not None else int(self.n_heads)

    @property
    def key_dim(self) -> int:
        return int(self.n_heads * self.head_k_dim)

    @property
    def value_dim(self) -> int:
        return int(self.n_v_heads * self.head_v_dim)

    @property
    def mlp_hidden(self) -> int:
        if self.mlp_intermediate is not None:
            return int(self.mlp_intermediate)
        # OFFICIAL IMPLEMENTATION: gdn2_1.3B n_embd 2304 -> intermediate 6208.
        return int(round(self.d_model * 6208 / 2304))

    def validate(self) -> "GDN2Config":
        if self.n_v_heads != self.n_heads:
            raise NotImplementedError(
                "⛔ grouped value attention (num_v_heads > num_heads) is NOT "
                "implemented in this port and is refused rather than silently "
                "ignored. The official default is num_v_heads = num_heads."
            )
        _ = (self.head_k_dim, self.head_v_dim)   # raise early on bad geometry
        return self

    def as_flag_table(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ==========================================================================
# the pinned / shrunk reference configs — the ONLY two the ladder may use
# ==========================================================================
def gdn2_published_config(**overrides: Any) -> GDN2Config:
    """The **published pin** — reproduces ``RIVAL_SPECS`` to the byte (P1)."""
    spec = RIVAL_SPECS[RIVAL_SPEC_NAME]
    cfg = GDN2Config(
        n_layers=int(spec.n_layers), d_model=int(spec.d_model),
        n_heads=int(spec.params["n_heads"]),
        head_dim=int(spec.params["d_k"]),
        **overrides,
    )
    return cfg.validate()


def gdn2_shrunk_config(budget: int = MATCHED_STATE_BYTE_BUDGET,
                       **overrides: Any) -> Tuple[GDN2Config, Dict[str, Any]]:
    """⭐ The **shrink-to-match** config, using the solved knob, never a hand one.

    :func:`chlu.eval.byte_ledger.shrink_to_budget` owns the solve (task §1.4). We
    take its ``shrunk_value`` for ``n_heads`` and re-derive ``head_dim`` from
    ``RIVAL_SPECS``' own tie ``d_k = d_v = d_model / H``.

    ⚠⚠ **P3, disclosed and returned, never smoothed over.** At the solved
    ``H = 6`` that tie has **no integer realisation** (``512/6 = 85.33…``). The
    floor ``head_dim = 85`` lands the arm at ``24*6*85*85*2 = 2,080,800 B``, i.e.
    **16,352 B (0.780 %) UNDER** the ledger's ideal ``2,097,152``. The shrink
    direction is preserved (an arm never grows), and the deployed number is the
    realized one. The returned dict carries both.
    """
    solved = shrink_to_budget(RIVAL_SPEC_NAME, budget)
    if solved["knob"] != "n_heads":                     # pragma: no cover
        raise LedgerError(f"⛔ unexpected shrink knob {solved['knob']!r}")
    h = int(solved["shrunk_value"])
    spec = RIVAL_SPECS[RIVAL_SPEC_NAME]
    d_model = int(spec.d_model)
    head_dim = d_model // h                              # the tie, floored
    cfg = GDN2Config(n_layers=int(spec.n_layers), d_model=d_model,
                     n_heads=h, head_dim=head_dim, **overrides).validate()
    ideal = int(solved["state_bytes_shrunk"])
    realized = main_recurrent_state_bytes(cfg)
    solved = dict(solved)
    solved.update({
        "ledger_ideal_bytes": ideal,
        "realized_bytes": realized,
        "realized_head_dim": head_dim,
        "integer_geometry_exact": (d_model % h == 0),
        "realized_minus_ideal_bytes": realized - ideal,
        "realized_occupancy": realized / float(budget),
        "note": (
            "P3: RIVAL_SPECS ties d_k = d_v = d_model/H; at the solved H the tie "
            "has no integer realisation, so the realized geometry is the floor "
            "and lands UNDER the ledger's ideal. The shrink never grows the arm."
            if d_model % h else
            "the tie is exact at this H; realized == ideal."
        ),
    })
    return cfg, solved


# ==========================================================================
# the byte ledger — the ONLY door into the ladder
# ==========================================================================
def main_recurrent_state_elements(cfg: GDN2Config) -> int:
    """``n_L * H * d_k * d_v`` — the paper's own quantity (§E.1 Eq. 90)."""
    return int(cfg.n_layers * cfg.n_heads * cfg.head_k_dim * cfg.head_v_dim)


def main_recurrent_state_bytes(cfg: GDN2Config) -> int:
    return int(main_recurrent_state_elements(cfg) * cfg.dtype_bytes)


def conv_state_elements(cfg: GDN2Config) -> int:
    """The short convolutions' cached state — ``(2*key_dim + value_dim)*(K-1)``.

    ⚠ **This is real, deployed state that ``RIVAL_SPECS`` does not count.** The
    official layer caches ``(conv_state_q, conv_state_k, conv_state_v)`` beside
    ``recurrent_state`` in ``update_layer_cache``. The paper's matched quantity is
    explicitly the **"main recurrent state size"** (§E.1), which is what
    ``RIVAL_SPECS`` implements — so the two conventions genuinely differ and both
    are reported. ⛔ Choosing between them is `c3-gb-landing`'s, not ours.
    """
    if not cfg.use_short_conv:
        return 0
    return int((2 * cfg.key_dim + cfg.value_dim) * (cfg.conv_size - 1)
               * cfg.n_layers)


def conv_state_bytes(cfg: GDN2Config) -> int:
    return int(conv_state_elements(cfg) * cfg.dtype_bytes)


def gdn2_param_count(cfg: GDN2Config) -> Dict[str, int]:
    """Parameter arithmetic, from the config alone (no model built)."""
    d, hv = cfg.d_model, cfg.head_v_dim
    kd, vd = cfg.key_dim, cfg.value_dim
    mixer = 2 * d * kd + d * vd                             # q_proj, k_proj, v_proj
    mixer += d * kd + d * vd                                # b_proj, w_proj
    mixer += vd * d                                         # o_proj
    mixer += d * hv + hv * kd                               # f_proj (2 linears)
    mixer += d * hv + hv * vd + vd                          # g_proj (+bias)
    mixer += cfg.n_heads + kd                               # A_log, dt_bias
    mixer += hv                                             # o_norm weight
    if cfg.use_short_conv:
        mixer += (2 * kd + vd) * cfg.conv_size
        if cfg.conv_bias:
            mixer += 2 * kd + vd
    mlp = 3 * d * cfg.mlp_hidden if cfg.use_mlp else 0
    norms = d * (2 if cfg.use_mlp else 1)
    block = mixer + mlp + norms
    total = block * cfg.n_layers + cfg.vocab_size * d + d + d * cfg.vocab_size
    return {"mixer_per_layer": int(mixer), "mlp_per_layer": int(mlp),
            "norms_per_layer": int(norms), "block": int(block),
            "embedding": int(cfg.vocab_size * d),
            "head": int(d * cfg.vocab_size), "final_norm": int(d),
            "total": int(total),
            "mixer_over_d_model_squared": float(mixer) / float(d * d)}


def gdn2_ledger_row(cfg: GDN2Config, *,
                    budget: int = MATCHED_STATE_BYTE_BUDGET,
                    label: str = "gdn2") -> Dict[str, Any]:
    """⭐ The ledger row. An arm that cannot produce one **fails loudly**.

    Columns, all declared, none inferred:

    * ``main_recurrent_state_bytes`` — the paper's matched quantity and the one
      ``RIVAL_SPECS`` implements. ``rival_specs_delta_bytes`` is asserted to be 0
      at the published pin.
    * ``conv_state_bytes`` — real deployed state the ledger's formula omits (P4).
    * ``total_state_bytes_as_deployed`` — task §1.5's convention: the sum.
    * ``phi_*`` — 0, and that is a **statement**: this arm is a standalone LM and
      does not use the shared ``StreamPhi`` read-in at all, so there is no φ to
      hide budget in. ``phi_accounted`` is True for the same reason
      ``chlu.eval.byte_ledger`` insists a zero-state arm still ledgers a 0.
    """
    cfg.validate()
    spec = RIVAL_SPECS[RIVAL_SPEC_NAME]
    main_b = main_recurrent_state_bytes(cfg)
    conv_b = conv_state_bytes(cfg)
    total_b = main_b + conv_b
    # RIVAL_SPECS at the SAME n_heads, so the row is comparable cell for cell.
    ref_b = int(spec.state_bytes(n_heads=cfg.n_heads))
    params = gdn2_param_count(cfg)
    return {
        "arm": label,
        "rival": RIVAL_SPEC_NAME,
        "citation": GDN2_CITATION,
        "arxiv": GDN2_ARXIV,
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "head_k_dim": cfg.head_k_dim,
        "head_v_dim": cfg.head_v_dim,
        "expand_v": cfg.expand_v,
        "dtype_bytes": cfg.dtype_bytes,
        "formula": "main = n_L * H * d_k * d_v ; conv = n_L*(2*key_dim+value_dim)*(K-1)",
        "arithmetic": (
            f"main = {cfg.n_layers} * {cfg.n_heads} * {cfg.head_k_dim} * "
            f"{cfg.head_v_dim} * {cfg.dtype_bytes} B = {main_b:,} B; "
            f"conv = {cfg.n_layers} * (2*{cfg.key_dim} + {cfg.value_dim}) * "
            f"({cfg.conv_size}-1) * {cfg.dtype_bytes} B = {conv_b:,} B; "
            f"total as deployed = {total_b:,} B"
        ),
        "main_recurrent_state_elements": main_recurrent_state_elements(cfg),
        "main_recurrent_state_bytes": main_b,
        "conv_state_elements": conv_state_elements(cfg),
        "conv_state_bytes": conv_b,
        "total_state_bytes_as_deployed": total_b,
        "rival_specs_bytes": ref_b,
        "rival_specs_delta_bytes": main_b - ref_b,
        "reproduces_rival_specs": main_b == ref_b,
        "phi_params_bytes": 0,
        "phi_state_bytes": 0,
        "phi_accounted": True,
        "phi_note": ("standalone rival LM — it does not use the shared "
                     "chlu.core.blocks.StreamPhi read-in, so φ is 0 by "
                     "construction and that is asserted, not inferred."),
        "budget_bytes": int(budget),
        "occupancy_main": main_b / float(budget),
        "occupancy_total_as_deployed": total_b / float(budget),
        "within_budget_main": main_b <= budget,
        "within_budget_total_as_deployed": total_b <= budget,
        "params_total": params["total"],
        "params_breakdown": params,
        "provenance": dict(GDN2_PROVENANCE),
        "pinned_config": cfg.as_flag_table(),
    }


class UnledgeredGDN2Error(LedgerError):
    """The arm could not reproduce ``RIVAL_SPECS``. Loud, never a warning."""


def assert_reproduces_rival_specs(cfg: GDN2Config) -> Dict[str, Any]:
    """⛔ Fail loudly if the arm's main recurrent state ≠ ``RIVAL_SPECS``."""
    row = gdn2_ledger_row(cfg)
    if not row["reproduces_rival_specs"]:
        raise UnledgeredGDN2Error(
            f"⛔ the GDN-2 arm's main recurrent state is "
            f"{row['main_recurrent_state_bytes']:,} B but RIVAL_SPECS"
            f"['{RIVAL_SPEC_NAME}'] at n_heads={cfg.n_heads} says "
            f"{row['rival_specs_bytes']:,} B "
            f"(delta {row['rival_specs_delta_bytes']:+,} B). An arm whose bytes "
            f"disagree with the pinned table cannot enter the ladder. "
            f"Arithmetic: {row['arithmetic']}"
        )
    return row


def fla_trap_check() -> Dict[str, Any]:
    """⛔⛔ TRAP 2, **re-derived here rather than quoted** (task §1.2).

    ``fla/layers/gated_deltanet.py`` (checked this session) defaults
    ``hidden_size=2048, expand_v=2, head_dim=256, num_heads=6``; the official
    NVlabs GDN-2 layer defaults ``head_dim=128, num_heads=16, expand_v=1``.
    """
    fla = 6 * 256 * int(256 * 2)
    official = 16 * 128 * int(128 * 1)
    return {
        "fla_defaults": {"hidden_size": 2048, "expand_v": 2, "head_dim": 256,
                         "num_heads": 6,
                         "source": "OFFICIAL IMPLEMENTATION: fla-org/"
                                   "flash-linear-attention fla/layers/"
                                   "gated_deltanet.py (GDN v1 layer)"},
        "official_gdn2_defaults": {"hidden_size": 2048, "expand_v": 1,
                                   "head_dim": 128, "num_heads": 16,
                                   "source": "OFFICIAL IMPLEMENTATION: NVlabs/"
                                             "GatedDeltaNet-2 lit_gpt/gdn2.py"},
        "paper_state_floats_per_layer": 262_144,
        "fla_state_floats_per_layer": int(fla),
        "official_state_floats_per_layer": int(official),
        "fla_over_paper": fla / 262_144.0,
        "official_over_paper": official / 262_144.0,
        "verdict": (
            "the 3x trap is the flash-linear-attention GatedDeltaNet (v1) layer's "
            "defaults; NVlabs' own GDN-2 layer defaults reproduce the paper "
            "exactly. Neither is inherited here."
        ),
    }


def gdn2_param_class_table(cfg: Optional[GDN2Config] = None) -> Dict[str, Any]:
    """The P6 arithmetic: what lands in 26-47 M and what does not."""
    base = cfg if cfg is not None else gdn2_published_config()
    with_mlp = gdn2_param_count(replace(base, use_mlp=True))
    no_mlp = gdn2_param_count(replace(base, use_mlp=False))
    half = replace(base, n_layers=max(1, base.n_layers // 2))
    return {
        "note": GDN2_PARAM_CLASS_NOTE,
        "weight_class": [26_000_000, 47_000_000],
        "with_mlp_24L": with_mlp["total"],
        "mlp_free_24L": no_mlp["total"],
        "with_mlp_half_layers": gdn2_param_count(half)["total"],
        "half_layers_state_bytes": main_recurrent_state_bytes(half),
        "mixer_over_d_model_squared": with_mlp["mixer_over_d_model_squared"],
        "in_class": {
            "with_mlp_24L": 26e6 <= with_mlp["total"] <= 47e6,
            "mlp_free_24L": 26e6 <= no_mlp["total"] <= 47e6,
        },
    }


# ==========================================================================
# the model
# ==========================================================================
def _xavier_uniform(key, shape: Tuple[int, ...], gain: float) -> jnp.ndarray:
    """PAPER App. D.5 / official ``_initialize_weights``: Xavier uniform, gain."""
    fan_out, fan_in = int(shape[0]), int(np.prod(shape[1:]))
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-a, maxval=a)


def _kaiming_linear(key, shape: Tuple[int, ...]) -> jnp.ndarray:
    """torch ``nn.Linear`` default: ``kaiming_uniform_(a=sqrt(5))`` = U(±1/√fan_in)."""
    fan_in = int(np.prod(shape[1:]))
    a = 1.0 / math.sqrt(fan_in)
    return jax.random.uniform(key, shape, minval=-a, maxval=a)


def _rms_norm(x: jnp.ndarray, w: jnp.ndarray, eps: float) -> jnp.ndarray:
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _l2_kernel(x: jnp.ndarray) -> jnp.ndarray:
    """The kernel's own L2, including its epsilon **inside** the sqrt.

    ``fused_recurrent_gdn2_fwd_kernel``: ``b_k = b_k / tl.sqrt(tl.sum(b_k*b_k)
    + 1e-6)``.
    """
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-6)


class GDN2Layer(eqx.Module):
    """One Gated DeltaNet-2 token mixer (PAPER §3.5 / official ``gdn2.py``)."""

    q_proj: jnp.ndarray
    k_proj: jnp.ndarray
    v_proj: jnp.ndarray
    b_proj: jnp.ndarray
    w_proj: jnp.ndarray
    o_proj: jnp.ndarray
    f_proj1: jnp.ndarray
    f_proj2: jnp.ndarray
    g_proj1: jnp.ndarray
    g_proj2: jnp.ndarray
    g_bias: jnp.ndarray
    A_log: jnp.ndarray
    dt_bias: jnp.ndarray
    o_norm_w: jnp.ndarray
    q_conv: Optional[jnp.ndarray]
    k_conv: Optional[jnp.ndarray]
    v_conv: Optional[jnp.ndarray]
    cfg: GDN2Config = eqx.field(static=True)

    def __init__(self, cfg: GDN2Config, *, key):
        cfg.validate()
        self.cfg = cfg
        d, kd, vd, hv = cfg.d_model, cfg.key_dim, cfg.value_dim, cfg.head_v_dim
        ks = jax.random.split(key, 13)
        g = float(cfg.init_gain)
        self.q_proj = _xavier_uniform(ks[0], (kd, d), g)
        self.k_proj = _xavier_uniform(ks[1], (kd, d), g)
        self.v_proj = _xavier_uniform(ks[2], (vd, d), g)
        self.b_proj = _xavier_uniform(ks[3], (kd, d), g)
        self.w_proj = _xavier_uniform(ks[4], (vd, d), g)
        self.o_proj = _xavier_uniform(ks[5], (d, vd), g)
        self.f_proj1 = _xavier_uniform(ks[6], (hv, d), g)
        self.f_proj2 = _xavier_uniform(ks[7], (kd, hv), g)
        self.g_proj1 = _xavier_uniform(ks[8], (hv, d), g)
        self.g_proj2 = _xavier_uniform(ks[9], (vd, hv), g)
        self.g_bias = jnp.zeros((vd,))
        # official: A_log = log(U(1,16)) per KEY head; dt_bias per key channel.
        self.A_log = jnp.log(jax.random.uniform(ks[10], (cfg.n_heads,),
                                                minval=1.0, maxval=16.0))
        dt = jnp.exp(jax.random.uniform(ks[11], (kd,))
                     * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = jnp.maximum(dt, 1e-4)
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))
        self.o_norm_w = jnp.ones((hv,))
        if cfg.use_short_conv:
            # torch Conv1d(groups=C) default init: U(±1/sqrt(kernel_size)).
            ck = jax.random.split(ks[12], 3)
            a = 1.0 / math.sqrt(cfg.conv_size)
            self.q_conv = jax.random.uniform(ck[0], (kd, cfg.conv_size),
                                             minval=-a, maxval=a)
            self.k_conv = jax.random.uniform(ck[1], (kd, cfg.conv_size),
                                             minval=-a, maxval=a)
            self.v_conv = jax.random.uniform(ck[2], (vd, cfg.conv_size),
                                             minval=-a, maxval=a)
        else:
            self.q_conv = self.k_conv = self.v_conv = None

    # -- the block design (PAPER §3.5, Fig. 1) -------------------------------
    def _short_conv(self, x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """Causal depthwise conv then SiLU — FLA ``ShortConvolution(activation='silu')``."""
        k = int(self.cfg.conv_size)
        pad = jnp.zeros((k - 1, x.shape[-1]), dtype=x.dtype)
        xp = jnp.concatenate([pad, x], axis=0)
        wins = jnp.stack([xp[i: i + x.shape[0]] for i in range(k)], axis=0)
        return jax.nn.silu(jnp.einsum("ktc,ck->tc", wins, w))

    def _recur(self, q, k, v, g, b, w):
        """⭐ The reference recurrence, ported from ``fused_recurrent_gdn2``.

        Statement order is the kernel's: L2-normalise q and k, scale q by
        ``d_k**-0.5``, decay the state, project it onto ``b ⊙ k`` to get the
        erase term, form ``v_new = w ⊙ v − erase``, write the rank-one update,
        then read with q from the **updated** state (PAPER Eq. 9/10, "The output
        is o_t = S_t^T q_t").
        """
        scale = float(self.cfg.head_k_dim) ** -0.5

        def step(S, inp):
            q_t, k_t, v_t, g_t, b_t, w_t = inp          # (H,dk) / (H,dv)
            k_t = _l2_kernel(k_t)
            q_t = _l2_kernel(q_t) * scale
            S = S * jnp.exp(g_t)[..., None]             # (H,dk,dv)
            e = b_t * k_t                               # (H,dk)
            erase = jnp.einsum("hkv,hk->hv", S, e)
            v_new = w_t * v_t - erase
            S = S + k_t[..., None] * v_new[:, None, :]
            o = jnp.einsum("hkv,hk->hv", S, q_t)
            return S, o

        S0 = jnp.zeros((self.cfg.n_heads, self.cfg.head_k_dim,
                        self.cfg.head_v_dim))
        _, o = jax.lax.scan(step, S0, (q, k, v, g, b, w))
        return o

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """``x`` is ``(T, d_model)`` for ONE sequence; returns ``(T, d_model)``."""
        cfg = self.cfg
        T = x.shape[0]
        q = x @ self.q_proj.T
        k = x @ self.k_proj.T
        v = x @ self.v_proj.T
        if cfg.use_short_conv:
            q = self._short_conv(q, self.q_conv)
            k = self._short_conv(k, self.k_conv)
            v = self._short_conv(v, self.v_conv)
        else:
            q, k, v = jax.nn.silu(q), jax.nn.silu(k), jax.nn.silu(v)
        # Eq. 12 / App. C.1: A_log is per key head, broadcast over its channels;
        # dt_bias is per key channel; the activation is computed in fp32.
        f = (x @ self.f_proj1.T) @ self.f_proj2.T
        a = jnp.repeat(jnp.exp(self.A_log), cfg.head_k_dim)
        g = -a * jax.nn.softplus(f + self.dt_bias)
        # Eq. 11: channel-wise erase (key axis) and write (value axis) gates.
        b = jax.nn.sigmoid(x @ self.b_proj.T)
        w = jax.nn.sigmoid(x @ self.w_proj.T)
        if cfg.allow_neg_eigval:                         # §3.1: erase gate only
            b = b * 2.0
        H, dk, dv = cfg.n_heads, cfg.head_k_dim, cfg.head_v_dim
        o = self._recur(q.reshape(T, H, dk), k.reshape(T, H, dk),
                        v.reshape(T, H, dv), g.reshape(T, H, dk),
                        b.reshape(T, H, dk), w.reshape(T, H, dv))
        # App. D.5: RMSNorm on the recurrent output, times a SiLU output gate.
        gate = ((x @ self.g_proj1.T) @ self.g_proj2.T + self.g_bias
                ).reshape(T, H, dv)
        o = _rms_norm(o, self.o_norm_w, cfg.norm_eps) * jax.nn.silu(gate)
        return o.reshape(T, cfg.value_dim) @ self.o_proj.T


class _SwiGLU(eqx.Module):
    """``lit_gpt`` ``LLaMAMLP`` = ``xformers`` SwiGLU: ``w3(silu(w1 x) * w2 x)``."""

    w1: jnp.ndarray
    w2: jnp.ndarray
    w3: jnp.ndarray

    def __init__(self, d: int, hidden: int, n_layers: int, *, key):
        ks = jax.random.split(key, 3)
        self.w1 = _kaiming_linear(ks[0], (hidden, d))
        self.w2 = _kaiming_linear(ks[1], (hidden, d))
        # mamba_init: the residual-output matrix is rescaled by 1/sqrt(2*n_layer).
        self.w3 = _kaiming_linear(ks[2], (d, hidden)) / math.sqrt(2 * n_layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return (jax.nn.silu(x @ self.w1.T) * (x @ self.w2.T)) @ self.w3.T


class GDN2Block(eqx.Module):
    """``lit_gpt`` ``Block`` at ``parallel_residual=False``: pre-norm, two residuals."""

    norm1_w: jnp.ndarray
    norm2_w: Optional[jnp.ndarray]
    mixer: GDN2Layer
    mlp: Optional[_SwiGLU]
    cfg: GDN2Config = eqx.field(static=True)

    def __init__(self, cfg: GDN2Config, *, key):
        self.cfg = cfg
        ks = jax.random.split(key, 2)
        self.norm1_w = jnp.ones((cfg.d_model,))
        self.mixer = GDN2Layer(cfg, key=ks[0])
        if cfg.use_mlp:
            self.norm2_w = jnp.ones((cfg.d_model,))
            self.mlp = _SwiGLU(cfg.d_model, cfg.mlp_hidden, cfg.n_layers, key=ks[1])
        else:
            self.norm2_w, self.mlp = None, None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.mixer(_rms_norm(x, self.norm1_w, self.cfg.norm_eps))
        if self.mlp is not None:
            x = x + self.mlp(_rms_norm(x, self.norm2_w, self.cfg.norm_eps))
        return x


class GDN2LM(eqx.Module):
    """The GDN-2 rival arm: embedding -> N x :class:`GDN2Block` -> norm -> head.

    ⭐ The call signature is ``(tokens, plans=None, read_mode=None, verlet=None)``
    so the pilot's :func:`chlu.training.train_cluformer.loss_fn` /
    :func:`~chlu.training.train_cluformer.token_nll` (and therefore the retention
    slices) work on this arm **unchanged**. ``plans`` is a CLU write plan and is
    meaningless here; it is accepted and ignored so no caller needs a branch.
    """

    embed: jnp.ndarray
    blocks: Tuple[GDN2Block, ...]
    final_norm_w: jnp.ndarray
    head: jnp.ndarray
    cfg: GDN2Config = eqx.field(static=True)

    def __init__(self, cfg: GDN2Config, *, key):
        cfg.validate()
        self.cfg = cfg
        ks = jax.random.split(key, cfg.n_layers + 2)
        self.embed = jax.random.normal(ks[0], (cfg.vocab_size, cfg.d_model)) \
            * float(cfg.embed_init_std)
        self.blocks = tuple(GDN2Block(cfg, key=ks[1 + i])
                            for i in range(cfg.n_layers))
        self.final_norm_w = jnp.ones((cfg.d_model,))
        self.head = _kaiming_linear(ks[-1], (cfg.vocab_size, cfg.d_model))

    def __call__(self, tokens: jnp.ndarray, plans=None, read_mode=None,
                 verlet=None) -> jnp.ndarray:
        del plans, read_mode, verlet          # CLU-only; accepted and ignored
        x = self.embed[jnp.asarray(tokens, dtype=jnp.int32)]
        for blk in self.blocks:
            x = blk(x)
        x = _rms_norm(x, self.final_norm_w, self.cfg.norm_eps)
        return x @ self.head.T

    # -- the ledger, on the built object -------------------------------------
    def cell_ledger(self) -> Dict[str, int]:
        """The ``cell_ledger()`` surface every C3 arm exposes."""
        p = int(sum(a.size for a in jax.tree_util.tree_leaves(
            eqx.filter(self, eqx.is_inexact_array))))
        el = main_recurrent_state_elements(self.cfg)
        return {"params": p, "state_floats": el,
                "state_bytes": int(el * self.cfg.dtype_bytes)}

    def ledger_row(self, *, budget: int = MATCHED_STATE_BYTE_BUDGET,
                   label: str = "gdn2") -> Dict[str, Any]:
        row = gdn2_ledger_row(self.cfg, budget=budget, label=label)
        row["params_measured"] = self.cell_ledger()["params"]
        row["params_measured_matches_arithmetic"] = (
            row["params_measured"] == row["params_total"])
        return row


# ==========================================================================
# the registry entry (§2: "its registry/config entry")
# ==========================================================================
def build_gdn2_arm(cfg: Optional[GDN2Config] = None, *, key,
                   check_ledger: bool = True) -> GDN2LM:
    """Construct the arm, refusing to return one that cannot be ledgered."""
    cfg = (cfg if cfg is not None else gdn2_published_config()).validate()
    model = GDN2LM(cfg, key=key)
    if check_ledger:
        row = model.ledger_row()
        if not row["reproduces_rival_specs"]:
            raise UnledgeredGDN2Error(
                f"⛔ built GDN-2 arm does not reproduce RIVAL_SPECS: "
                f"{row['main_recurrent_state_bytes']:,} B vs "
                f"{row['rival_specs_bytes']:,} B. {row['arithmetic']}"
            )
    return model


#: ⭐ The C3 rival-arm registry entry for GDN-2. Each rival spoke owns exactly
#: one such record, in its own module, so three concurrent engineers cannot
#: collide in a shared table.
GDN2_ARM: Dict[str, Any] = {
    "name": "gdn2",
    "rival_spec": RIVAL_SPEC_NAME,
    "arxiv": GDN2_ARXIV,
    "citation": GDN2_CITATION,
    "build": build_gdn2_arm,
    "published_config": gdn2_published_config,
    "shrunk_config": gdn2_shrunk_config,
    "ledger_row": gdn2_ledger_row,
    "provenance": GDN2_PROVENANCE,
    "trained_by_this_spoke": False,
    "sanity_anchor": (
        "⛔ NONE that is protocol-comparable. arXiv:2605.22791's own numbers are "
        "zero-shot WikiText perplexity (15.90) for a 1.3 B model with a subword "
        "tokenizer after 100 B FineWeb-Edu tokens at 4 k context — a different "
        "corpus, tokenizer, param count and protocol on all four axes, i.e. a "
        "rival-vs-rival number, not a venue number (scout §1.1.1). The nearest "
        "in-class byte-level anchor for 'our implementation is not broken' is the "
        "recurrent-class enwik8 band around 1.0-1.06 bpc at 39-41 M "
        "(Transformer-XL 12L 1.06, Adaptive-Span 1.02, Mega 1.02, Longformer "
        "small 1.00) — a SANITY BAND, never a matched baseline: every one of "
        "those is quoted at an eval context larger than its train context."
    ),
}


__all__ = [
    "GDN2_ARM", "GDN2_ARXIV", "GDN2_CITATION", "GDN2_PROVENANCE",
    "GDN2_PARAM_CLASS_NOTE", "PROVENANCE_PREFIXES", "RIVAL_SPEC_NAME",
    "GDN2Config", "GDN2Layer", "GDN2Block", "GDN2LM", "UnledgeredGDN2Error",
    "build_gdn2_arm", "gdn2_published_config", "gdn2_shrunk_config",
    "gdn2_ledger_row", "gdn2_param_count", "gdn2_param_class_table",
    "assert_reproduces_rival_specs", "fla_trap_check",
    "main_recurrent_state_bytes", "main_recurrent_state_elements",
    "conv_state_bytes", "conv_state_elements",
]
