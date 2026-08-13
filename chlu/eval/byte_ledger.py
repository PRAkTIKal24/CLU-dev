"""⭐ The BYTE LEDGER — the invariant the whole tier-iii control rests on.

The primary claim (charter §2) is at **matched params AND matched state-bytes**
against the TTT-class system swap. "Matched state bytes" is therefore not a
remark in a paper's methods paragraph; it is a quantity every arm must be able to
produce, from its config, with the arithmetic shown. So:

* every run emits a ledger artifact — per arm, the **inference-time state in
  bytes, including φ**;
* an arm that cannot produce its ledger makes the run **fail loudly**
  (:class:`UnledgeredArmError`), never warn;
* an arm that does not fit the pre-registered budget also fails loudly
  (:class:`StateByteBudgetError`), because a harness that lets an over-budget arm
  through has quietly decided the control.

⛔ **THE BUDGET IS NOT THIS MODULE'S TO CHOOSE.** `c3-benchmark-scout` §1.5
established with the arithmetic that at a fixed ≈38 M params the natural
inference state spans **1.60 MB (TTT-Linear) → 100.7 MB (sliding-window @4 k)** —
a **63× range** — and whichever number is picked advantages some rivals and
cripples others. It is a **decision**, taken by the Head and Advisor
(2026-08-13), and it lives here as **one named constant**
(:data:`MATCHED_STATE_BYTE_BUDGET`) so that confirming its last digit is a
one-line edit rather than a hunt through call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# ==========================================================================
# ⭐ THE RULED BUDGET — one constant, one edit
# ==========================================================================
#: ⭐ **RULED (Advisor + Head, 2026-08-13): the matched-state-byte budget is
#: pre-registered at ≈2 MB**, and implemented as the **2 MiB ceiling**
#: ``2,097,152 B``.
#:
#: *Rationale, recorded here because a bare number invites re-litigation.* At the
#: 26–47 M weight class the CLU store at ``d=12`` (**1,966,080 B**, 0.94× of this
#: ceiling) and TTT-Linear (**1,597,440 B**, 0.76×) land in the same place
#: naturally, so **the two-sided system swap is byte-honest by construction**;
#: every other rival — Mamba-2 3.29×, sliding-window 6.40×, TXL-at-3800 23.7× —
#: is **shrunk to match rather than grown**, which is the defensible direction of
#: a matched-bytes control. A *ceiling that no arm sits exactly on* cannot be
#: accused of having been tuned to any one arm, which is why 2 MiB was preferred
#: over setting the budget equal to our own store's 1,966,080 B.
#:
#: ⚠ **Provenance of this exact figure**: the ruling says "≈2 MB"; the Hub
#: recommended 2 MiB and a one-line confirmation of the final digit was still
#: outstanding when this module was written. Per the task, the harness is built
#: against the constant and defaults to 2,097,152 B. ⛔ Do not introduce a third
#: value; change THIS line if the confirmation differs.
MATCHED_STATE_BYTE_BUDGET: int = 2_097_152

#: Provenance string carried into every artifact beside the number, so a reader
#: of the JSON never has to come back to this file to learn where it came from.
BUDGET_PROVENANCE = (
    "RULED Advisor+Head 2026-08-13 as '≈2 MB'; implemented as the 2 MiB ceiling "
    "2,097,152 B on the Hub's recommendation (CLU d=12 1,966,080 B = 0.94x; "
    "TTT-Linear 1,597,440 B = 0.76x). Last-digit confirmation was OUTSTANDING at "
    "build time — see chlu/eval/byte_ledger.py MATCHED_STATE_BYTE_BUDGET."
)

#: Our arms run in **float32**; the scout's published-rival table is quoted in
#: **bf16**. A ledger that mixes them silently is a 2× lie, so the element width
#: is a declared field of every row, never an assumption.
FP32_BYTES = 4
BF16_BYTES = 2


class LedgerError(RuntimeError):
    """Base class for the two loud failures."""


class UnledgeredArmError(LedgerError):
    """An arm's cell cannot state its inference-time state in bytes."""


class StateByteBudgetError(LedgerError):
    """An arm's ledgered state does not fit the pre-registered budget."""


# ==========================================================================
# ⚠⚠ TRAP 2 — rival state configs are PINNED here, never inherited
# ==========================================================================
#: ⛔ **Never inherit a library default and then claim byte-matching.** The
#: `flash-linear-attention` defaults for GDN/GDN-2 (``head_dim=256,
#: num_heads=6, expand_v=2``) give **3× the state the GDN-2 paper's own numbers
#: imply**; a ledger built on them would silently void the entire control. Every
#: state-bearing hyperparameter below is pinned explicitly, and every one carries
#: the provenance of *where the number came from* — a paper table or an official
#: implementation — because those are not the same evidence.
#:
#: Reference configs (scout §1.5), both landing in the 26–47 M class:
#:   * *attention class*  — 12 L, d_model 512, 8 heads × 64, d_ff 2048 ⇒ 37.88 M
#:   * *recurrent class*  — 24 L, d_model 512 (≈6·d_model² params/layer) ⇒ 37.75 M


@dataclass(frozen=True)
class RivalStateSpec:
    """One rival's inference state, from ITS OWN pinned hyperparameters.

    ``state_elements`` is computed by :meth:`elements`; nothing here is measured
    from a built model, because we do not build the rivals — we ledger them so the
    matched-bytes table can be assembled and the shrink-to-match direction can be
    solved before any arm trains.
    """

    name: str
    n_layers: int
    d_model: int
    formula: str
    params: Dict[str, int]
    provenance: str
    dtype_bytes: int = BF16_BYTES
    #: The single hyperparameter the shrink-to-match control is allowed to move.
    #: ⭐ Declared per rival so "shrink to match" is a mechanical solve on a named
    #: knob rather than an unrecorded judgement call.
    shrink_knob: Optional[str] = None

    def elements(self, **overrides: int) -> int:
        p = dict(self.params, **overrides)
        return int(_RIVAL_FORMULAS[self.name](self.n_layers, self.d_model, p))

    def state_bytes(self, **overrides: int) -> int:
        return int(self.elements(**overrides) * self.dtype_bytes)


def _f_ttt_linear(n_L, d_model, p):
    # W1 (H, d_h, d_h) + b1 (H, 1, d_h) per layer.
    return n_L * (p["n_heads"] * p["head_dim"] ** 2 + p["n_heads"] * p["head_dim"])


def _f_ttt_mlp(n_L, d_model, p):
    # TTT-MLP adds W2 (H, 4 d_h, d_h) + b2, and W1 at 4x width.
    return n_L * p["n_heads"] * (8 * p["head_dim"] ** 2 + 5 * p["head_dim"])


def _f_gdn(n_L, d_model, p):
    # The paper states "262,144 floats per batch element" per layer at
    # d_model 2048 / H 16 => state/layer = d_model^2 / H exactly.
    return n_L * (d_model ** 2) // p["n_heads"]


def _f_mamba2(n_L, d_model, p):
    d_inner = d_model * p["expand"]
    n_heads = d_inner // p["headdim"]
    ssm_state = n_heads * p["headdim"] * p["d_state"]
    conv_dim = d_inner + 2 * p["ngroups"] * p["d_state"]
    conv_state = conv_dim * (p["d_conv"] - 1)
    return n_L * (ssm_state + conv_state)


def _f_txl(n_L, d_model, p):
    # Transformer-XL caches previous-segment HIDDEN STATES (d_model per position
    # per layer), not separate K and V — hence 1x, not 2x.
    return n_L * p["mem_len"] * d_model


def _f_sliding(n_L, d_model, p):
    # Sliding-window attention caches K AND V for the last w positions — hence 2x.
    return 2 * n_L * p["window"] * d_model


_RIVAL_FORMULAS = {
    "ttt_linear": _f_ttt_linear,
    "ttt_mlp": _f_ttt_mlp,
    "gated_deltanet2": _f_gdn,
    "mamba2": _f_mamba2,
    "transformer_xl": _f_txl,
    "sliding_window": _f_sliding,
}

#: ⭐ The pinned rival table. Every ``provenance`` distinguishes a **paper table**
#: from an **official implementation**; the scout could not obtain Mamba-2's
#: appendix table (the arXiv PDF would not parse), so ours comes from the code and
#: says so — that difference must survive into our flag-provenance table.
RIVAL_SPECS: Dict[str, RivalStateSpec] = {
    s.name: s for s in (
        RivalStateSpec(
            name="ttt_linear", n_layers=24, d_model=512,
            formula="n_L * (H*d_h^2 + H*d_h)",
            params={"n_heads": 8, "head_dim": 64},
            provenance="OFFICIAL IMPLEMENTATION: test-time-training/ttt-lm-pytorch/"
                       "ttt.py — W1 (H,d_h,d_h), b1 (H,1,d_h); head_dim=width//n_heads. "
                       "arXiv:2407.04620 publishes no enwik8/WT-103/PG-19 cell.",
            shrink_knob="head_dim",
        ),
        RivalStateSpec(
            name="ttt_mlp", n_layers=24, d_model=512,
            formula="n_L * H * (8*d_h^2 + 5*d_h)",
            params={"n_heads": 8, "head_dim": 64},
            provenance="OFFICIAL IMPLEMENTATION: ttt-lm-pytorch/ttt.py (W2 at 4x "
                       "width + b2, W1 symmetric).",
            shrink_knob="head_dim",
        ),
        RivalStateSpec(
            name="gated_deltanet2", n_layers=24, d_model=512,
            formula="n_L * d_model^2 / H",
            # ⛔ H=4 keeps d_k=d_v=128 as the PAPER states, at d_model 512.
            params={"n_heads": 4, "d_k": 128, "d_v": 128},
            provenance="PAPER: arXiv:2605.22791 (Hatamizadeh, Choi, Kautz, NVIDIA, "
                       "2026-05-21) states H=16, d_k=d_v=128, d_model=2048 and "
                       "'262,144 floats per batch element' per layer => "
                       "state/layer = d_model^2/H. Scaled to d_model 512 at H=4 to "
                       "hold d_k=d_v=128. ⛔ NOT the flash-linear-attention default "
                       "(head_dim=256, num_heads=6, expand_v=2), which is 3x the "
                       "paper's state — TRAP 2.",
            shrink_knob="n_heads",
        ),
        RivalStateSpec(
            name="mamba2", n_layers=24, d_model=512,
            formula="n_L * (n_heads*headdim*d_state + (d_inner + 2*ngroups*d_state)*(d_conv-1))",
            params={"d_state": 128, "d_conv": 4, "expand": 2, "headdim": 64,
                    "ngroups": 1},
            provenance="OFFICIAL IMPLEMENTATION: state-spaces/mamba, "
                       "mamba_ssm/modules/mamba2.py defaults; allocate_inference_cache "
                       "allocates (batch, nheads, headdim, d_state). ⚠ The paper's "
                       "per-size appendix table was NOT OBTAINED (arXiv:2405.21060) — "
                       "this row's provenance is CODE, not a paper table.",
            shrink_knob="d_state",
        ),
        RivalStateSpec(
            name="transformer_xl", n_layers=12, d_model=512,
            formula="n_L * mem_len * d_model",
            params={"mem_len": 512},
            provenance="PAPER: arXiv:1901.02860 Table 2 + §3 (segment 784, eval "
                       "attention length 3800). Caches hidden states, not K/V.",
            shrink_knob="mem_len",
        ),
        RivalStateSpec(
            name="sliding_window", n_layers=12, d_model=512,
            formula="2 * n_L * window * d_model",
            params={"window": 512},
            provenance="PAPER: arXiv:2004.05150 (Longformer small, 41 M => 1.00 bpc); "
                       "per-layer windows 32->512 staged. Caches K AND V.",
            shrink_knob="window",
        ),
    )
}


def shrink_to_budget(name: str, budget: int = MATCHED_STATE_BYTE_BUDGET
                     ) -> Dict[str, Any]:
    """⭐ Solve a rival's declared knob **down** so it fits ``budget``.

    This is the mechanical half of the ruling: the budget was chosen so that
    over-budget rivals are **shrunk to match rather than grown**, and a control
    that direction only counts if the shrink is a solved, reported number rather
    than a hand-tuned one. Returns the original and shrunk knob value, the bytes
    on both sides, and the occupancy after shrinking.
    """
    spec = RIVAL_SPECS[name]
    if spec.shrink_knob is None:
        raise LedgerError(f"rival {name!r} declares no shrink knob")
    knob = spec.shrink_knob
    v0 = int(spec.params[knob])
    before = spec.state_bytes()
    # ⭐ monotone in every declared knob (state is increasing in head_dim,
    # d_state, mem_len, window; DEcreasing in n_heads for GDN), so a scan from the
    # published value is exact and needs no solver.
    increasing = knob != "n_heads"
    v, best = v0, None
    for _ in range(4096):
        if spec.state_bytes(**{knob: v}) <= budget:
            best = v
            break
        v = (v - 1) if increasing else (v + 1)
        if v < 1:
            break
    if best is None:
        raise LedgerError(f"rival {name!r} cannot reach {budget} B via {knob}")
    after = spec.state_bytes(**{knob: best})
    return {
        "rival": name, "knob": knob, "published_value": v0, "shrunk_value": best,
        "state_bytes_published": before, "state_bytes_shrunk": after,
        "budget_bytes": int(budget),
        "occupancy_published": before / float(budget),
        "occupancy_shrunk": after / float(budget),
        "shrink_factor": (before / after) if after else float("inf"),
        "formula": spec.formula, "provenance": spec.provenance,
        "dtype_bytes": spec.dtype_bytes,
    }


def rival_reference_table(budget: int = MATCHED_STATE_BYTE_BUDGET
                          ) -> Dict[str, Any]:
    """Every pinned rival's published bytes, occupancy, and shrink solution."""
    rows = {}
    for name, spec in RIVAL_SPECS.items():
        b = spec.state_bytes()
        row: Dict[str, Any] = {
            "state_bytes": b, "elements": spec.elements(),
            "dtype_bytes": spec.dtype_bytes, "n_layers": spec.n_layers,
            "d_model": spec.d_model, "formula": spec.formula,
            "pinned_params": dict(spec.params), "provenance": spec.provenance,
            "occupancy": b / float(budget), "within_budget": b <= budget,
        }
        if spec.shrink_knob is not None and b > budget:
            row["shrink_to_match"] = shrink_to_budget(name, budget)
        rows[name] = row
    return {"budget_bytes": int(budget), "budget_provenance": BUDGET_PROVENANCE,
            "rivals": rows}


# ==========================================================================
# our arms: the ledger computed FROM THE CONFIG, φ included
# ==========================================================================
@dataclass
class ArmLedger:
    """One arm's inference-time byte ledger, with the arithmetic in the object."""

    arm: str
    n_layers: int
    cell_state_bytes_per_layer: int
    phi_params_bytes_per_layer: int
    phi_state_bytes_per_layer: int
    cell_params: int
    dtype_bytes: int = FP32_BYTES
    arithmetic: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_state_bytes(self) -> int:
        """⭐ What actually occupies memory at inference: EVERY layer's cell state.

        ⚠ Per-layer is the seductive number and it is the wrong one — a 12-layer
        model holds 12 cell states at once. The scout's rival table is likewise
        summed over ``n_L``, so this is the like-for-like column.
        """
        return int(self.n_layers * (self.cell_state_bytes_per_layer
                                    + self.phi_state_bytes_per_layer))

    @property
    def total_phi_params_bytes(self) -> int:
        return int(self.n_layers * self.phi_params_bytes_per_layer)

    def as_row(self, budget: int) -> Dict[str, Any]:
        tot = self.total_state_bytes
        return {
            "arm": self.arm,
            "n_layers": self.n_layers,
            "dtype_bytes": self.dtype_bytes,
            "cell_params": self.cell_params,
            "cell_state_bytes_per_layer": self.cell_state_bytes_per_layer,
            # ⛔ φ is ledgered EXPLICITLY on every arm, with both of its columns,
            # because "byte ledgers on every arm incl. φ" is a charter §5
            # invariant and an omitted φ row is indistinguishable from a
            # forgotten one. φ is the SHARED read-in (chlu.core.blocks.StreamPhi)
            # and is bit-identical across arms by assert_shared_shell_identical.
            "phi_params_bytes_per_layer": self.phi_params_bytes_per_layer,
            "phi_params_bytes_total": self.total_phi_params_bytes,
            "phi_state_bytes_per_layer": self.phi_state_bytes_per_layer,
            "phi_state_bytes_total": int(self.n_layers
                                         * self.phi_state_bytes_per_layer),
            "phi_accounted": True,
            "total_state_bytes": tot,
            "budget_bytes": int(budget),
            # ⭐ occupancy, not just compliance: the table must show how much of
            # the envelope each arm truly uses.
            "occupancy": tot / float(budget),
            "within_budget": tot <= budget,
            "arithmetic": self.arithmetic,
            **self.extra,
        }


def phi_bytes(pcfg) -> Tuple[int, int]:
    """φ's (params_bytes, state_bytes) **per layer**, from the config alone.

    :class:`~chlu.core.blocks.StreamPhi` is an ``eqx.nn.MLP(d_model -> dim)`` with
    one hidden layer of width ``psi_hidden``:
    ``d_model*H + H`` then ``H*H + H`` (MLP depth 1 => one hidden-to-hidden) then
    ``H*dim + dim``.

    ⭐ **φ's inference STATE is zero and that is a statement, not an omission.**
    φ is a feed-forward map applied per chunk; it retains nothing across chunks.
    Its *parameters* are ledgered (and are identical on every arm, which is what
    makes the swap a swap). If φ ever gains recurrent state, this function is the
    one place that must change, and :meth:`ArmLedger.as_row` will carry it
    automatically.
    """
    mcfg = pcfg.memory_cfg()
    scfg = pcfg.store_cfg()
    d_model, h, dim = int(pcfg.d_model), int(mcfg.psi_hidden), int(scfg.dim)
    n_params = (d_model * h + h) + (h * h + h) + (h * dim + dim)
    return int(FP32_BYTES * n_params), 0


def arm_ledger(arm: str, pcfg, swap_ledger: Dict[str, Any],
               *, budget: int = MATCHED_STATE_BYTE_BUDGET) -> Dict[str, Any]:
    """Ledger one arm, or raise :class:`UnledgeredArmError`.

    ``swap_ledger`` is :func:`chlu.training.train_cluformer.solve_arms`'s second
    return value — the per-cell ledger measured off the constructed cells.

    ⛔ A missing/malformed row raises. A **zero-state** row does not: ``none`` and
    ``echo`` are the null arms and genuinely hold no state, which is a ledger, not
    a missing ledger. The distinction is the whole point — "0" must be asserted,
    never inferred from absence.
    """
    row = swap_ledger.get(arm)
    if not isinstance(row, dict) or "state_bytes" not in row:
        raise UnledgeredArmError(
            f"⛔ arm {arm!r} produced no byte ledger (swap_ledger row={row!r}). "
            f"The tier-iii claim is at matched params AND matched state-bytes, so "
            f"an unledgered arm cannot be reported at all. Give its cell a "
            f"cell_ledger() returning {{'params','state_floats','state_bytes'}}."
        )
    ph_p, ph_s = phi_bytes(pcfg)
    n_L = int(pcfg.n_layers)
    cell_b = int(row["state_bytes"])
    led = ArmLedger(
        arm=arm, n_layers=n_L,
        cell_state_bytes_per_layer=cell_b,
        phi_params_bytes_per_layer=ph_p, phi_state_bytes_per_layer=ph_s,
        cell_params=int(row.get("params", 0)),
        arithmetic=(
            f"total_state_bytes = n_layers * (cell_state_bytes + phi_state_bytes) "
            f"= {n_L} * ({cell_b} + {ph_s}) = {n_L * (cell_b + ph_s)} B; "
            f"phi_params_bytes = {n_L} * {ph_p} = {n_L * ph_p} B (SHARED, "
            f"bit-identical across arms); dtype = float32 ({FP32_BYTES} B/elt)."
        ),
        extra={"state_floats_per_layer": int(row.get("state_floats", 0))},
    )
    return led.as_row(budget)


def build_byte_ledger(pcfg, swap_ledger: Dict[str, Any], arms,
                      *, budget: int = MATCHED_STATE_BYTE_BUDGET,
                      enforce: bool = True) -> Dict[str, Any]:
    """⭐ The per-run ledger artifact. Fails loudly; never warns.

    Args:
        enforce: when ``True`` (the ruling's posture) an arm whose
            ``total_state_bytes`` exceeds ``budget`` raises
            :class:`StateByteBudgetError`. Turning it off is a declared,
            recorded act — the artifact carries ``enforced: false`` — because a
            silently unenforced budget is exactly the failure mode the ruling was
            written to prevent.
    """
    rows = {a: arm_ledger(a, pcfg, swap_ledger, budget=budget) for a in arms}
    over = {a: r for a, r in rows.items() if not r["within_budget"]}
    art: Dict[str, Any] = {
        "budget_bytes": int(budget),
        "budget_provenance": BUDGET_PROVENANCE,
        "enforced": bool(enforce),
        "arms": rows,
        "over_budget": sorted(over),
        "phi_accounted_on_every_arm": all(r["phi_accounted"] for r in rows.values()),
        "rival_reference": rival_reference_table(budget),
    }
    if enforce and over:
        raise StateByteBudgetError(
            "⛔ matched-state-byte budget violated — the tier-iii control is not "
            "byte-honest at this config, so the run stops before it trains:\n"
            + "".join(
                f"    {a}: {r['total_state_bytes']:,} B = {r['occupancy']:.2f}x the "
                f"{budget:,} B budget  ({r['arithmetic']})\n"
                for a, r in sorted(over.items()))
            + "  Shrink the store (capacity / atoms_per_item / addr_dim+payload_dim "
              "/ n_layers) until it fits, or set enforce_state_byte_budget=false "
              "to record a DECLARED, non-compliant run.\n"
              f"  Budget provenance: {BUDGET_PROVENANCE}"
        )
    return art
