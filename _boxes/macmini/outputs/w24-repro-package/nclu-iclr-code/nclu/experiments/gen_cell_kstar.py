"""gen cell kstar."""

from __future__ import annotations
from nclu.paths import run_dir
import sys
import time
from pathlib import Path
from typing import Callable, Optional
import numpy as np

KSTAR_ARM = "kstar"
KSTAR_K = 32
KSTAR_SOURCE = str(
    run_dir("shapes3d/k_sweep/CROSSOVER-val.json (K_star = 32 under SELECTION-RULE.md)")
)
KSTAR_CONTROLS = (
    ("landscape_removed", "zero"),
    ("landscape_shuffled", "value_permute"),
    ("landscape_shuffled_across_wells", "well_permute"),
    ("random_atoms", "random_span"),
)
KSTAR_ROWS = ("full",) + tuple((c for c, _ in KSTAR_CONTROLS))
KSTAR_VAL_BAR_ABS = 0.5
KSTAR_VAL_BAR_COLLAPSE = 0.1
N_CORR = 16384


def kstar_label(row: str) -> str:
    return KSTAR_ARM if row == "full" else f"{KSTAR_ARM}/{row}"


def _scripts():
    root = Path(__file__).resolve().parents[2]
    sd = root / "scripts"
    if str(sd) not in sys.path:
        sys.path.insert(0, str(sd))
    import shapes_nclu_lib as XL
    from shapes_matched_bytes import ward_balanced_labels

    if Path(XL.__file__).resolve().parents[1] != root:
        raise AssertionError(f"⛔ crossover_lib resolves OUTSIDE {root}: {XL.__file__}")
    return (XL, ward_balanced_labels)


def request_values(grid_values, lab: np.ndarray) -> np.ndarray:
    """(N, F) factor INDICES -> (N, F) requested VALUES, the TensorAddress contract."""
    lab = np.asarray(lab, np.int64)
    return np.stack(
        [np.asarray(grid_values[k], np.float64)[lab[:, k]] for k in range(lab.shape[1])], axis=1
    )


def build(
    X_store: np.ndarray, lab_store: np.ndarray, cards, seed: int, log: Callable = print
) -> dict:
    from nclu.core.gen_read import TensorAddress, span_coefficients
    from nclu.core.gen_store import channel_correlation, dead_channels
    from nclu.experiments.exp_gen_read import READ_PINS
    from nclu.experiments.covariance_store import CovContentStore

    XL, ward = _scripts()
    t0 = time.time()
    X = np.asarray(X_store, np.float32).reshape((X_store.shape[0], -1))
    D = int(X.shape[1])
    if D != int(XL.D):
        raise AssertionError(f"⛔ D={D}, the crossover object is D={XL.D}")
    rng = np.random.default_rng(int(seed) * 7919 + 20260923)
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
    log(
        f"  ⭐ [kstar] CovContentStore K={KSTAR_K} A={A} on {X.shape[0]} store rows: wells {min(sizes)}..{max(sizes)} · dead {int(dead.sum())} · rank/width {min(rw):.4f}..{max(rw):.4f} ({time.time() - t0:.1f}s)"
    )
    t1 = time.time()
    Y = np.asarray(lab_store, np.int64)
    T = span_coefficients(st, X, basis)
    ntr = int(0.9 * Y.shape[0])
    addr = TensorAddress(
        [int(c) for c in cards],
        rank=int(READ_PINS["rank"]),
        ridge=float(READ_PINS["ridge"]),
        seed=int(READ_PINS["address_seed"]),
    )
    addr.multiplicative = True
    addr.fit(
        Y[:ntr],
        [t_[:ntr] for t_ in T],
        sweeps=int(READ_PINS["sweeps"]),
        log=lambda *_: None,
        Y_val=Y[ntr:],
        targets_val=[t_[ntr:] for t_ in T],
    )
    addr_s = time.time() - t1
    abytes = int(addr.byte_footprint()["address_bytes"])
    led = XL.four_part_footprint(st, abytes)
    led["address_fit_seconds"] = round(addr_s, 1)
    led["convention"] = (
        "FOUR-PART (single-design-crossover): atoms + covariance/count + address book + coupling/hole table, summed, NEVER netted; 524,288 B is a REFERENCE column"
    )
    log(
        f"  ⭐ [kstar] address book fitted in {addr_s:.1f}s · FOUR-PART FOOTPRINT atoms {led['p1_atom_bytes']:,} | cov+count {led['p2_covariance_plus_count_bytes']:,} | address {abytes:,} | coupling {led['p4_coupling_hole_table_bytes']:,} = {led['honest_total_bytes']:,} B = {led['times_reference_pin']}x the 524,288 B REFERENCE"
    )
    info = {
        "K": KSTAR_K,
        "K_source": KSTAR_SOURCE,
        "A": int(A),
        "well_sizes": sizes,
        "n_dead": int(dead.sum()),
        "deployed_rank": [int(r) for r in st.deployed_rank],
        "rank_over_width": rw,
        "rank_over_width_max": float(max(rw)),
        "n_store_rows": int(X.shape[0]),
        "n_corr_rows": int(sub.size),
        "address_pins": {k: READ_PINS[k] for k in ("rank", "ridge", "sweeps", "address_seed")},
        "build_seconds": round(time.time() - t0, 1),
    }
    return {"store": st, "basis": basis, "address": addr, "footprint": led, "info": info, "XL": XL}


def objects(built: dict, seed: int) -> dict:
    XL = built["XL"]
    st, basis = (built["store"], built["basis"])
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
    store, basis, zeroed = obj
    n = int(np.asarray(values).shape[0])
    if zeroed:
        return (np.zeros((n, int(store.dim)), np.float32), np.zeros(n, bool))
    f, oor = built["address"].read(store, values, grid_values, basis)
    return (np.asarray(f, np.float32), np.asarray(oor, bool))


def val_gate(val_scores: dict) -> dict:
    """VB1/VB2 on the cell's VAL rows — evaluated BEFORE any B number is recorded."""
    arm = float(val_scores["full"])
    vb1 = bool(arm >= KSTAR_VAL_BAR_ABS)
    ctl = {c: float(val_scores[c]) for c, _ in KSTAR_CONTROLS}
    vb2 = {c: bool(arm - v >= KSTAR_VAL_BAR_COLLAPSE) for c, v in ctl.items()}
    return {
        "arm_val_aggregate": arm,
        "controls_val_aggregate": ctl,
        "VB1_abs_bar": KSTAR_VAL_BAR_ABS,
        "VB1_pass": vb1,
        "VB2_collapse_bar": KSTAR_VAL_BAR_COLLAPSE,
        "VB2_pass_per_control": vb2,
        "PASS": bool(vb1 and all(vb2.values())),
        "source": "PROTOCOL-shapes3d/lace_poe_32k-1.md §3 (declared before any val number)",
    }


def row_extra(built: dict, row: str, per_set: Optional[dict] = None) -> dict:
    led = dict(built["footprint"])
    footprint = {
        "arm": kstar_label(row),
        "deployed_bytes": int(led["honest_total_bytes"]),
        "budget_bytes": 524288,
        "multiple_of_budget": led["times_reference_pin"],
        "four_part": led,
        "budget_note": f"⛔ NOT MATCHED: {led['times_reference_pin']}x the 524,288 B store column, FOUR-PART convention, never netted. Quoted ONLY with this multiple on the same line.",
    }
    return {
        "uses_store": True,
        "byte_footprint": footprint,
        "channel_bits": None,
        "kstar": {
            "row": row,
            "is_the_arm": row == "full",
            "role_on_the_table": "⭐ THE ARM — the K* single design"
            if row == "full"
            else "⭐ landscape control — MUST COLLAPSE",
            "info": built["info"],
            "request_sets": per_set or {},
        },
        "mechanism": "request -> TensorAddress span coefficients -> the K*=32 CovContentStore's per-well span -> decode. ⛔ No feature, row index or label enters the read."
        if row == "full"
        else f"landscape control `{row}` of the K* arm: the atom block perturbed AFTER the address book was fitted (crossover study's form).",
    }
