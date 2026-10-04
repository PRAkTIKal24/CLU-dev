"""⭐ **The C3 rival-arm registry** — how a tuned rival enters the tier-iii ladder.

`c3-benchmark-scout` established that **every modern rival's 26-47 M cell is NOT
PUBLISHED**, so every rival row of the tier-iii table is a from-scratch run of
ours. A rival therefore needs three things, and this module is where the first
two meet:

1. a **cell** implementing the stream block's memory interface
   (``init_state`` / ``read`` / ``write`` / ``cell_ledger``), identical in every
   respect to what the CLU and TTT arms are given — same shell, same chunk
   granularity, same optimiser, same data order (only ``block.cell`` changes);
2. a **pinned config** whose every state-bearing hyperparameter carries a
   ``PAPER:``/``OFFICIAL IMPLEMENTATION:`` provenance string, so no library
   default can be inherited and then reported as byte-matched (⛔ the
   ``flash-linear-attention`` 3x trap, scout §1.5);
3. a **byte-ledger row** reproducing :data:`chlu.eval.byte_ledger.RIVAL_SPECS`'
   pinned value **to the byte**.

⭐ **One generic seam, one line per rival.** Each rival lives in its own module
and calls :func:`register_c3_rival` at import; :data:`C3_RIVAL_MODULES` is the
single list the trainer consults. Three engineers built the three rivals in
parallel, so the *only* shared surface is this list — a rival that needs more
than one line here is doing something the seam should have absorbed.

⛔ **This registry does not decide the budget, the shrink, or the comparison.**
It constructs cells and reports what they cost.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, NamedTuple, Tuple


class C3RivalArm(NamedTuple):
    """One tuned rival, as the ladder sees it.

    Attributes:
        name: the arm name used in ``PilotConfig.arms`` and in every artifact.
        spec_name: the key in :data:`chlu.eval.byte_ledger.RIVAL_SPECS` this arm
            must reproduce to the byte. ⛔ Not optional: an arm with no pinned
            reference cannot be ledgered against the scout's table.
        resolve: ``(overrides: dict) -> config`` — the pinned config, with the
            ``shrink_to_budget``-solved knob already applied.
        build: ``(config, *, latent_dim, key) -> eqx.Module`` — one cell.
        reference_row: ``() -> dict`` — the pinned/published/shrunk arithmetic,
            for the artifact and for the reproduction test.
        deployed_row: ``(config, n_layers, budget) -> dict`` — what the SHELL
            actually deploys, which is not the rival's own reference geometry.
    """

    name: str
    spec_name: str
    resolve: Callable[..., Any]
    build: Callable[..., Any]
    reference_row: Callable[[], Dict[str, Any]]
    deployed_row: Callable[..., Dict[str, Any]]


#: ⭐ The one shared line per rival. Imported lazily (see :func:`_ensure_loaded`)
#: so that importing the registry cannot drag JAX in before the caller wants it.
C3_RIVAL_MODULES: Tuple[str, ...] = (
    "chlu.eval.rivals.c3_mamba2",
)

_REGISTRY: Dict[str, C3RivalArm] = {}
_LOADED = False


def register_c3_rival(arm: C3RivalArm) -> C3RivalArm:
    """Register one rival arm. ⛔ Re-registering under a different definition is
    an error, not a silent overwrite — two modules claiming one arm name is
    exactly the parallel-engineer collision this registry exists to make loud."""
    prev = _REGISTRY.get(arm.name)
    if prev is not None and prev is not arm and prev.build is not arm.build:
        raise ValueError(
            f"C3 rival arm {arm.name!r} is already registered by another module; "
            "two definitions of one arm name would silently decide which rival "
            "the ladder ran.")
    _REGISTRY[arm.name] = arm
    return arm


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True          # set first: a rival module importing this one is fine
    for mod in C3_RIVAL_MODULES:
        importlib.import_module(mod)


def c3_rival_names() -> Tuple[str, ...]:
    """Every registered rival arm name, sorted."""
    _ensure_loaded()
    return tuple(sorted(_REGISTRY))


def is_c3_rival(name: str) -> bool:
    _ensure_loaded()
    return str(name) in _REGISTRY


def get_c3_rival(name: str) -> C3RivalArm:
    _ensure_loaded()
    try:
        return _REGISTRY[str(name)]
    except KeyError:
        raise KeyError(
            f"unknown C3 rival arm {name!r}; registered: {c3_rival_names()}"
        ) from None


def resolve_c3_rival(name: str, overrides: Any = None):
    """The rival's pinned config, with any declared override applied."""
    return get_c3_rival(name).resolve(overrides or {})


def make_c3_rival_cell(name: str, *, latent_dim: int, overrides: Any = None,
                       config: Any = None, key):
    """Construct one rival cell. ``config`` (pre-resolved) wins over ``overrides``."""
    arm = get_c3_rival(name)
    cfg = config if config is not None else arm.resolve(overrides or {})
    return arm.build(cfg, latent_dim=int(latent_dim), key=key)


def c3_rival_reference_rows() -> Dict[str, Dict[str, Any]]:
    """Every registered rival's published/shrunk arithmetic, for the artifact."""
    return {n: get_c3_rival(n).reference_row() for n in c3_rival_names()}


def c3_rival_deployed_row(name: str, config: Any, n_layers: int,
                          budget: int) -> Dict[str, Any]:
    """One rival's AS-DEPLOYED byte row (the shell's layer count, not its own)."""
    return get_c3_rival(name).deployed_row(config, int(n_layers), int(budget))


__all__ = [
    "C3RivalArm", "C3_RIVAL_MODULES", "register_c3_rival", "c3_rival_names",
    "is_c3_rival", "get_c3_rival", "resolve_c3_rival", "make_c3_rival_cell",
    "c3_rival_reference_rows", "c3_rival_deployed_row",
]
