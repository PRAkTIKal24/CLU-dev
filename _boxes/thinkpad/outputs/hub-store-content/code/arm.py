"""hub/store-content — build one content arm and read it.

⛔ Nothing here is a new mechanism.  The grouping rule, the atom selection, the
bandwidth rule, the address book, the read verb and the four charter rows are the
PINNED objects, called (never re-implemented) from ``chlu.core.gen_store``,
``chlu.core.gen_landscape`` and the pin's ``x0`` arm.  The ONLY thing this file
varies is the matrix ``Z`` those objects are handed — which is this hub's dial.

THE READ, spelled out because every number depends on it::

    x0 = concat_k  atoms[k][ argmax_i P(atom_i | request, well k) ]

``X0_SETTLING_STEPS = 0`` and ``X0_READ_MODE = "state"``: the launch IS the decoder's
input.  The decoder is handed ``x0[:, :n_penult]`` reshaped to (128, 4, 4); the
earlier-layer coordinates are what the read is ADDRESSED THROUGH and are never
rendered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from chlu.core.gen_store import (
    channel_correlation, correlation_grouping, dead_channels,
    well_channels_from_labels,
)
from chlu.core.gen_landscape import (
    AtomStore, atom_bandwidths, assert_atoms_are_train_rows, atom_posterior,
    fit_atom_address_book, launch_state, select_atom_rows, store_byte_ledger,
)

from content import BUDGET_BYTES, build_Z

#: pinned from the frozen store plan
K_WELLS = 5
#: pinned from the pin's x0 arm
LAUNCH_MODE = "clamp_atom"
LOOKUP_LAUNCH_MODE = "posterior_mean"
SETTLING_STEPS = 0
READ_MODE = "state"
#: pinned bandwidth rule (gen_landscape default)
SIGMA_RULE = "nn_median"
SIGMA_FRAC = 1.0
#: pinned address-book Laplace smoothing (gen_landscape default)
BOOK_SMOOTHING = 1.0
#: pinned atom-selection rule
ATOM_SELECTION = "kmeanspp"

ROWS = ("full", "atom_mean_lookup", "row_lookup", "atoms_shuffled")


@dataclass
class Arm:
    spec: object
    store: AtomStore
    book: object
    sigma: np.ndarray
    scales: dict
    labels_vec: np.ndarray
    atom_rows_global: np.ndarray
    ledger: dict
    grouping: dict
    provenance: dict

    @property
    def n_penult(self) -> int:
        return int(self.spec.n_penult)


def _grouping(Z: np.ndarray, seed: int, log=print) -> tuple:
    """Correlation grouping over the JOINT coordinates - the pinned rule, wider input.

    The rule is ``gen_store.correlation_grouping`` verbatim: dead channels excluded,
    |corr| affinity, agglomerative clustering, a minimum well size.  K is pinned at 5
    (the frozen plan's choice) rather than re-chosen, because re-choosing K on a wider
    matrix would move two dials at once.
    """
    corr = channel_correlation(Z)
    dead = dead_channels(Z)
    lab, info = correlation_grouping(corr, K_WELLS, method="auto", dead=dead,
                                     return_info=True)
    return np.asarray(lab, np.int64), info


def well_composition(spec, well_channels) -> dict:
    """⭐ Does the earlier-layer content actually MIX into the wells, or does it sit
    in wells of its own?

    The joint coordinates are laid out block by block, so a well's composition is
    read straight off the index ranges.  ⛔ This matters and is not cosmetic: if the
    b3 coordinates form their own wells, the arm is two stores side by side and the
    earlier layer cannot influence the routing of the wells the decoder reads from.
    ``mixing`` is 1.0 when every well holds the global block mixture and 0.0 when
    every well is pure.
    """
    bounds, off = [], 0
    for b in spec.blocks:
        d = spec.dims_per_block[b]
        bounds.append((b, off, off + d))
        off += d
    per_well, purity = [], []
    for w in well_channels:
        idx = np.asarray(w, np.int64)
        comp = {b: int(((idx >= lo) & (idx < hi)).sum()) for b, lo, hi in bounds}
        per_well.append(comp)
        tot = max(sum(comp.values()), 1)
        purity.append(max(comp.values()) / tot)
    glob = {b: (hi - lo) / spec.D for b, lo, hi in bounds}
    base = max(glob.values())
    mixing = float(np.clip((1.0 - np.mean(purity)) / max(1.0 - base, 1e-9), 0.0, 1.0))
    return {"well_block_composition": per_well,
            "mean_well_purity": float(np.mean(purity)),
            "global_block_fractions": {k: float(v) for k, v in glob.items()},
            "mixing": mixing,
            "mixing_note": ("1.0 = every well holds the global block mixture; "
                            "0.0 = every well is a single block. A low value means "
                            "the joint store is two stores side by side.")}


def build_arm(spec, pool_rows: np.ndarray, labels_pool: np.ndarray,
              cardinalities, seed: int = 0, src=None, log=print) -> Arm:
    """Fit one content arm on the store's pool.  Every knob above is pinned."""
    info = spec.assert_matched()
    N = spec.n_atoms
    log(f"  [{spec.name}] D={spec.D} N={N} bytes={info['store_bytes']:,}")

    # ---- 1. content, standardised by scales measured on the ATOM rows ----
    # ⛔ Two passes: the atom rows are chosen on a provisionally scaled Z, then the
    # scales are RE-MEASURED on exactly those atom rows and everything downstream
    # uses those.  That makes the standardisation a function of the stored atoms,
    # which is the only thing that makes it free.
    Z0, _ = build_Z(spec, pool_rows, seed=seed, src=src)
    a_idx = select_atom_rows(Z0, N, ATOM_SELECTION, int(seed))
    _, scales = build_Z(spec, pool_rows[a_idx], seed=seed, src=src)
    Z, _ = build_Z(spec, pool_rows, seed=seed, src=src, scales=scales)
    del Z0

    # ---- 2. the partition ----
    lab, ginfo = _grouping(Z, seed, log=log)
    wc = well_channels_from_labels(lab, K_WELLS)
    well_sizes = [len(w) for w in wc]
    log(f"  [{spec.name}] wells {well_sizes} (sum {sum(well_sizes)})")

    # ---- 3. the atoms: genuine stored train rows ----
    store = AtomStore(wc, (spec.D,), Z, a_idx, selection=ATOM_SELECTION)
    assert_atoms_are_train_rows(store, Z)
    led = store_byte_ledger(store.n_atoms, well_sizes)
    if int(led["store_bytes"]) != BUDGET_BYTES:
        raise AssertionError(
            f"{spec.name}: measured ledger {led['store_bytes']:,} B != "
            f"{BUDGET_BYTES:,} B budget")
    measured = store.n_bytes()
    if measured != BUDGET_BYTES:
        raise AssertionError(
            f"{spec.name}: the LIVE object measures {measured:,} B, the ledger says "
            f"{led['store_bytes']:,} B - a byte claim may not be made from either")

    # ---- 4. bandwidth and address book ----
    sigma = atom_bandwidths(store, SIGMA_FRAC, SIGMA_RULE)
    book = fit_atom_address_book(store, Z, labels_pool, cardinalities, sigma,
                                 smoothing=BOOK_SMOOTHING)

    return Arm(
        spec=spec, store=store, book=book, sigma=np.asarray(sigma),
        scales=scales, labels_vec=lab, atom_rows_global=np.asarray(pool_rows)[a_idx],
        ledger={**led, "deployed_bytes": int(led["store_bytes"]),
                "budget_bytes": BUDGET_BYTES,
                "what": f"{N} atoms x {spec.D} float32 = the store, content={info['blocks']}"},
        grouping={"well_sizes": well_sizes, "K": K_WELLS,
                  "method_used": ginfo.get("method_used"),
                  "n_dead_channels": int(ginfo.get("n_dead_channels", -1)),
                  **well_composition(spec, wc)},
        provenance={"spec": info, "scales": {k: float(v) for k, v in scales.items()},
                    "sigma": [float(x) for x in np.asarray(sigma)],
                    "atom_selection": ATOM_SELECTION, "seed": int(seed),
                    "launch_mode": LAUNCH_MODE, "settling_steps": SETTLING_STEPS,
                    "read_mode": READ_MODE, "sigma_rule": SIGMA_RULE,
                    "book_smoothing": BOOK_SMOOTHING},
    )


def read_arm(arm: Arm, requests: np.ndarray, row: str = "full", seed: int = 0) -> dict:
    """``(n, F) request index matrix -> the decoder's input``, for one charter row.

    ``full``              the arm: per-well posterior-argmax atom.
    ``atom_mean_lookup``  §C.1 floor: the posterior-weighted MEAN of the same atoms.
    ``row_lookup``        §C.1 harder floor: the single best-scoring WHOLE stored row.
    ``atoms_shuffled``    §C.2 landscape-removed: same indices, permuted atom table.
    """
    if row not in ROWS:
        raise ValueError(row)
    store, book = arm.store, arm.book
    R = np.asarray(requests, np.int64)
    if row in ("full", "atom_mean_lookup"):
        mode = LAUNCH_MODE if row == "full" else LOOKUP_LAUNCH_MODE
        lau = launch_state(_LandShim(store), book, R, mode, 0.5)
        return {"x": np.asarray(lau["x0"], np.float32),
                "atom_index": np.asarray(lau["atom_index"], np.int64),
                "confidence": np.asarray(lau["confidence"], np.float32),
                "launch_mode": mode}
    S = atom_posterior(book, R)
    x = np.zeros((int(R.shape[0]), int(store.dim)), np.float32)
    if row == "row_lookup":
        idx = np.asarray(S, np.float64).sum(axis=1).argmax(axis=-1)
        for k, w in enumerate(store.well_channels):
            x[:, np.asarray(w, np.int64)] = np.asarray(store.atoms[k], np.float32)[idx]
        return {"x": x, "atom_index_shared": idx.astype(np.int64),
                "launch_mode": "row_lookup"}
    idx = np.asarray(S, np.float64).argmax(axis=-1)
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_916)
    for k, w in enumerate(store.well_channels):
        A = np.asarray(store.atoms[k], np.float32)
        x[:, np.asarray(w, np.int64)] = A[rng.permutation(A.shape[0])][idx[:, k]]
    return {"x": x, "atom_index": idx.astype(np.int64), "launch_mode": "atoms_shuffled"}


class _LandShim:
    """``launch_state`` only ever touches ``land.store``; this hands it one without
    fitting a coupling the zero-step read provably does not use (the pin's own
    ``x0_coupling_is_not_needed``)."""

    def __init__(self, store):
        self.store = store


def penult_of(arm: Arm, x: np.ndarray) -> np.ndarray:
    """The frozen decoder's input: the leading penultimate slice, as (n,128,4,4).

    ⛔ The RMS standardisation is UNDONE here.  The store's coordinates are
    ``p / scales['p']``; the decoder was fitted on raw penultimate features and a
    globally rescaled input is a different image.  For the ``P`` arm this multiply
    is the exact inverse of the divide, so ``P`` reproduces the incumbent content
    bit-for-bit up to one float32 round-trip - which is checked, not assumed, by
    ``check_P_reproduces_cache``.
    """
    n = int(np.asarray(x).shape[0])
    z = np.asarray(x, np.float32)[:, : arm.n_penult] * np.float32(arm.scales["p"])
    return z.reshape((n, 128, 4, 4))


def check_P_reproduces_cache(arm: Arm, pool_rows, src, tol: float = 1e-4) -> dict:
    """⭐ The provenance check for the whole hub: the ``P`` arm's atoms, after the
    scale round-trip, must be the FROZEN CACHE's own rows.

    If this drifts, every ``WIDEN`` vs ``P`` comparison is against a baseline that is
    not the incumbent, and nothing downstream would notice.
    """
    rows = np.asarray(arm.atom_rows_global, np.int64)
    ref = np.asarray(src["p"][rows], np.float32).reshape(rows.size, -1)
    got = np.zeros((rows.size, arm.store.dim), np.float32)
    for k, w in enumerate(arm.store.well_channels):
        got[:, np.asarray(w, np.int64)] = np.asarray(arm.store.atoms[k], np.float32)
    got = got[:, : arm.n_penult] * np.float32(arm.scales["p"])
    err = float(np.max(np.abs(got - ref))) if ref.size else 0.0
    rel = err / max(float(np.abs(ref).max()), 1e-9)
    return {"max_abs": err, "max_rel": rel, "n_atoms": int(rows.size),
            "passed": bool(rel <= tol)}
