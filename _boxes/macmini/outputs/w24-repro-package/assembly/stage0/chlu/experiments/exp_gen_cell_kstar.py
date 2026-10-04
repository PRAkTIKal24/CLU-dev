"""⭐ THE K* SINGLE DESIGN AS AN ARM OF THE GENERATION TABLE (w15 spoke/lace-poe-rows).

``exp_gen_cell`` registers this arm as ``kstar`` (⛔ NOT on ``ARMS``: that tuple is the
pin's declared rival-row list, exactly as ``x0`` and ``dynread`` are kept off it).  This
module OWNS the arm's construction; ``exp_gen_cell`` owns the instrument (the cell's
rows, the frozen decoder, the hash-asserted external classifier) and calls in here.

WHAT THE ARM IS — nothing re-selected, nothing new invented
-----------------------------------------------------------
The main text's single design, as `w12 hub/single-design-crossover` froze it:

* ``CovContentStore`` (``exp_w8_single_design``) — atoms derived from a per-well
  covariance statistic, so deletion is an exact downdate;
* the LABEL-FREE ``ward_balanced`` cut of the channel-correlation dendrogram
  (``scripts/w8_matched_bytes_recut.ward_balanced_labels``) at **K* = 32**,
  pinned by ``SELECTION-RULE.md`` (2026-09-21T11:10:24Z) — ⛔ K is NOT re-chosen here;
* A = min(64, min_k d_k − 1) (``scripts/w12_crossover_lib.atoms_for``);
* the span read through ``pinned_basis`` and a ``TensorAddress`` at ``exp_gen_read.
  READ_PINS`` (rank 128, ridge 1e-2, 40 sweeps, multiplicative, 90/10 fit/early-val);
* the four charter-C controls in the crossover hub's own form (``perturb_modes`` applied
  AFTER the address book is fitted, read through ``qr_basis`` at the book's own widths).

WHAT IS DIFFERENT, DECLARED
---------------------------
The store is fitted on THIS CELL's store pool (the generation table's own rows and its
frozen bridge cache), because an arm scored on rows of its own is not on this table.
The partition therefore differs from the crossover hub's, by the same rule.

⛔ Byte ledger: the crossover hub's FOUR-PART convention (atoms | covariance+count |
address book | coupling/hole table), summed and NEVER netted.  The object is several
times the 524,288 B the table's store column is held to; that multiple travels with
every number and ⛔ no K* row is called matched-bytes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

#: ⭐ The arm's name on the table and its control rows.  ``kstar`` is the arm.
KSTAR_ARM = "kstar"
#: ⭐ K* — READ from the crossover hub's frozen rule's published answer; not a knob.
KSTAR_K = 32
KSTAR_SOURCE = ("C:/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json "
                "(K_star = 32 under SELECTION-RULE.md, 2026-09-21T11:10:24Z)")
#: ⭐ the four charter-C controls, in COMMON.md's names, mapped to the crossover hub's
#: ``perturb_modes`` kinds.
KSTAR_CONTROLS = (
    ("landscape_removed", "zero"),
    ("landscape_shuffled", "value_permute"),
    ("landscape_shuffled_across_wells", "well_permute"),
    ("random_atoms", "random_span"),
)
KSTAR_ROWS = ("full",) + tuple(c for c, _ in KSTAR_CONTROLS)
#: ⭐ VAL BARS — declared in PREREG-w15-lace-poe-rows-1.md BEFORE any val number existed.
#: VB1: the arm's val aggregate (external classifier, mean per-factor accuracy over the
#:      cell's 512 stratified val rows) is at least this.
KSTAR_VAL_BAR_ABS = 0.50
#: VB2: EVERY control's val aggregate sits at least this far below the arm's.
KSTAR_VAL_BAR_COLLAPSE = 0.10
N_CORR = 16_384


def kstar_label(row: str) -> str:
    return KSTAR_ARM if row == "full" else f"{KSTAR_ARM}/{row}"


def _scripts():
    """The crossover hub's library, imported from THIS worktree's ``scripts/``.

    ⛔ One implementation, not two: the partition, the atom rule, the basis, the
    controls and the ledger are the crossover hub's own functions, byte for byte.
    """
    root = Path(__file__).resolve().parents[2]
    sd = root / "scripts"
    if str(sd) not in sys.path:
        sys.path.insert(0, str(sd))
    import w12_crossover_lib as XL                                 # noqa: E402
    from w8_matched_bytes_recut import ward_balanced_labels         # noqa: E402
    if Path(XL.__file__).resolve().parents[1] != root:
        raise AssertionError(f"⛔ w12_crossover_lib resolves OUTSIDE {root}: {XL.__file__}")
    return XL, ward_balanced_labels


def request_values(grid_values, lab: np.ndarray) -> np.ndarray:
    """(N, F) factor INDICES -> (N, F) requested VALUES, the TensorAddress contract."""
    lab = np.asarray(lab, np.int64)
    return np.stack([np.asarray(grid_values[k], np.float64)[lab[:, k]]
                     for k in range(lab.shape[1])], axis=1)


def build(X_store: np.ndarray, lab_store: np.ndarray, cards, seed: int,
          log: Callable = print) -> dict:
    """Fit the K* object on the cell's store pool.  Returns the object and its ledger."""
    from ..core.gen_read import TensorAddress, span_coefficients
    from ..core.gen_store import channel_correlation, dead_channels
    from ..experiments.exp_gen_read import READ_PINS
    from ..experiments.exp_w8_single_design import CovContentStore

    XL, ward = _scripts()
    t0 = time.time()
    X = np.asarray(X_store, np.float32).reshape((X_store.shape[0], -1))
    D = int(X.shape[1])
    if D != int(XL.D):
        raise AssertionError(f"⛔ D={D}, the crossover object is D={XL.D}")
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_923)
    sub = np.sort(rng.choice(X.shape[0], size=min(N_CORR, X.shape[0]), replace=False))
    corr = channel_correlation(X[sub])
    dead = np.asarray(dead_channels(X), bool).ravel()
    lab = ward(corr, KSTAR_K, dead)
    wc = [np.flatnonzero(lab == k).astype(np.int64) for k in range(KSTAR_K)]
    sizes = [int(c.size) for c in wc]
    A = XL.atoms_for(sizes)
    st = CovContentStore(X, wc, A, D)
    basis = XL.pinned_basis(st)
    rw = [round(int(b.shape[1]) / s, 4) for b, s in zip(basis, sizes)]
    log(f"  ⭐ [kstar] CovContentStore K={KSTAR_K} A={A} on {X.shape[0]} store rows: "
        f"wells {min(sizes)}..{max(sizes)} · dead {int(dead.sum())} · rank/width "
        f"{min(rw):.4f}..{max(rw):.4f} ({time.time() - t0:.1f}s)")

    t1 = time.time()
    Y = np.asarray(lab_store, np.int64)
    T = span_coefficients(st, X, basis)
    ntr = int(0.9 * Y.shape[0])
    addr = TensorAddress([int(c) for c in cards], rank=int(READ_PINS["rank"]),
                         ridge=float(READ_PINS["ridge"]),
                         seed=int(READ_PINS["address_seed"]))
    addr.multiplicative = True
    addr.fit(Y[:ntr], [t_[:ntr] for t_ in T], sweeps=int(READ_PINS["sweeps"]),
             log=lambda *_: None, Y_val=Y[ntr:], targets_val=[t_[ntr:] for t_ in T])
    addr_s = time.time() - t1
    abytes = int(addr.byte_ledger()["address_bytes"])
    led = XL.four_part_ledger(st, abytes)
    led["address_fit_seconds"] = round(addr_s, 1)
    led["convention"] = ("FOUR-PART (w12 single-design-crossover): atoms + covariance/"
                         "count + address book + coupling/hole table, summed, NEVER "
                         "netted; 524,288 B is a REFERENCE column")
    log(f"  ⭐ [kstar] address book fitted in {addr_s:.1f}s · FOUR-PART LEDGER atoms "
        f"{led['p1_atom_bytes']:,} | cov+count {led['p2_covariance_plus_count_bytes']:,} | "
        f"address {abytes:,} | coupling {led['p4_coupling_hole_table_bytes']:,} = "
        f"{led['honest_total_bytes']:,} B = {led['times_reference_pin']}x the 524,288 B "
        "REFERENCE")
    info = {"K": KSTAR_K, "K_source": KSTAR_SOURCE, "A": int(A), "well_sizes": sizes,
            "n_dead": int(dead.sum()), "deployed_rank": [int(r) for r in st.deployed_rank],
            "rank_over_width": rw, "rank_over_width_max": float(max(rw)),
            "n_store_rows": int(X.shape[0]), "n_corr_rows": int(sub.size),
            "address_pins": {k: READ_PINS[k] for k in ("rank", "ridge", "sweeps",
                                                        "address_seed")},
            "build_seconds": round(time.time() - t0, 1)}
    return {"store": st, "basis": basis, "address": addr, "ledger": led, "info": info,
            "XL": XL}


def objects(built: dict, seed: int) -> dict:
    """``row -> (store, basis, zeroed)``: the arm and its four charter-C controls."""
    XL = built["XL"]
    st, basis = built["store"], built["basis"]
    out = {"full": (st, basis, False)}
    widths = [b.shape[1] for b in basis]
    for name, kind in KSTAR_CONTROLS:
        mods = XL.perturb_modes(st.modes, kind, int(seed))
        sh = XL.ShimStore(mods, st.well_channels, st.dim, st.deployed_rank)
        if kind == "zero":
            out[name] = (sh, basis, True)
        else:
            out[name] = (sh, XL.qr_basis(mods, st.deployed_rank, widths=widths), False)
    return out


def read(built: dict, obj, values: np.ndarray, grid_values) -> tuple:
    """(N, F) VALUES -> (N, D) features, request-only.  ⛔ No feature enters."""
    store, basis, zeroed = obj
    n = int(np.asarray(values).shape[0])
    if zeroed:
        return np.zeros((n, int(store.dim)), np.float32), np.zeros(n, bool)
    f, oor = built["address"].read(store, values, grid_values, basis)
    return np.asarray(f, np.float32), np.asarray(oor, bool)


def val_gate(val_scores: dict) -> dict:
    """VB1/VB2 on the cell's VAL rows — evaluated BEFORE any B number is recorded."""
    arm = float(val_scores["full"])
    vb1 = bool(arm >= KSTAR_VAL_BAR_ABS)
    ctl = {c: float(val_scores[c]) for c, _ in KSTAR_CONTROLS}
    vb2 = {c: bool(arm - v >= KSTAR_VAL_BAR_COLLAPSE) for c, v in ctl.items()}
    return {"arm_val_aggregate": arm, "controls_val_aggregate": ctl,
            "VB1_abs_bar": KSTAR_VAL_BAR_ABS, "VB1_pass": vb1,
            "VB2_collapse_bar": KSTAR_VAL_BAR_COLLAPSE, "VB2_pass_per_control": vb2,
            "PASS": bool(vb1 and all(vb2.values())),
            "source": "PREREG-w15-lace-poe-rows-1.md §3 (declared before any val number)"}


def row_extra(built: dict, row: str, per_set: Optional[dict] = None) -> dict:
    led = dict(built["ledger"])
    ledger = {"arm": kstar_label(row), "deployed_bytes": int(led["honest_total_bytes"]),
              "budget_bytes": 524_288, "multiple_of_budget": led["times_reference_pin"],
              "four_part": led,
              "budget_note": (f"⛔ NOT MATCHED: {led['times_reference_pin']}x the 524,288 B "
                              "store column, FOUR-PART convention, never netted. Quoted "
                              "ONLY with this multiple on the same line.")}
    return {
        "uses_store": True,
        "byte_ledger": ledger,
        "channel_bits": None,
        "kstar": {"row": row, "is_the_arm": row == "full",
                  "role_on_the_table": ("⭐ THE ARM — the K* single design" if row == "full"
                                        else "⭐ charter-C control — MUST COLLAPSE"),
                  "info": built["info"], "request_sets": per_set or {}},
        "mechanism": ("request -> TensorAddress span coefficients -> the K*=32 "
                      "CovContentStore's per-well span -> decode. ⛔ No feature, row "
                      "index or label enters the read."
                      if row == "full" else
                      f"charter-C control `{row}` of the K* arm: the atom block perturbed "
                      "AFTER the address book was fitted (crossover hub's form)."),
    }
