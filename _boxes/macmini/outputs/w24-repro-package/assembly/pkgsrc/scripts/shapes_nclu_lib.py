"""shapes nclu lib."""

from __future__ import annotations
from pathlib import Path
import numpy as np

D = 2048
REFERENCE_PIN = 524288
RANK_TARGET = 64
COUNTS_BYTES = 11488
COUPLING_BYTES = 5288 + 1322
from nclu.paths import partition_file as _partition_file

PART_VAL = str(_partition_file("val"))
PART_TEST = str(_partition_file("test"))


def log(m=""):
    print(m, flush=True)


def atoms_for(well_sizes) -> int:
    """A = min(64, min_k d_k - 1).  STRICTLY below the narrowest well width."""
    return int(min(RANK_TARGET, min((int(d) for d in well_sizes)) - 1))


def pinned_basis(store):
    out = []
    for M, r in zip(store.modes, store.deployed_rank):
        V = np.asarray(M[1 : 1 + r], np.float64).T
        mu = np.asarray(M[0], np.float64)
        perp = mu - V @ (V.T @ mu) if r > 0 else mu
        nrm = float(np.linalg.norm(perp))
        ref = float(np.linalg.norm(mu)) or 1.0
        if nrm / ref > 0.0001:
            V = np.hstack([V, (perp / nrm)[:, None]])
        out.append(np.ascontiguousarray(V))
    return out


def qr_basis(modes, deployed_rank, widths=None):
    out = []
    for i, (M, r) in enumerate(zip(modes, deployed_rank)):
        d = int(M.shape[1])
        w = int(widths[i]) if widths is not None else min(d, 1 + int(r))
        if w <= 0:
            out.append(np.zeros((d, 0), np.float64))
            continue
        blk = np.asarray(M[: max(1, 1 + int(r))], np.float64).T
        Q = np.linalg.qr(blk, mode="complete")[0]
        out.append(np.ascontiguousarray(Q[:, : min(w, d)]))
    return out


class ShimStore:
    """The three attributes the span read touches: modes, well_channels, dim.

    Used only to carry a PERTURBED atom block through the unchanged read path.
    It holds no statistic and can neither be written to nor deleted from -- it is a
    control, not a store.
    """

    def __init__(self, modes, well_channels, dim, deployed_rank):
        self.modes = list(modes)
        self.well_channels = [np.asarray(c, np.int64) for c in well_channels]
        self.dim = int(dim)
        self.deployed_rank = list(deployed_rank)
        self.n_atoms = int(modes[0].shape[0]) if modes else 0

    @property
    def K(self):
        return len(self.modes)


def perturb_modes(modes, kind: str, seed: int):
    rng = np.random.default_rng(int(seed) * 7919 + 20260921)
    Gs = [np.asarray(G, np.float32) for G in modes]
    if kind == "zero":
        return [np.zeros_like(G) for G in Gs]
    if kind == "value_permute":
        return [G.reshape(-1)[rng.permutation(G.size)].reshape(G.shape) for G in Gs]
    if kind == "random_span":
        out = []
        for G in Gs:
            R = rng.normal(size=G.shape).astype(np.float32)
            R *= float(np.linalg.norm(G)) / (float(np.linalg.norm(R)) + 1e-12)
            out.append(R)
        return out
    if kind == "well_permute":
        K = len(Gs)
        perm = rng.permutation(K)
        while K > 1 and np.all(perm == np.arange(K)):
            perm = rng.permutation(K)
        out = []
        for k in range(K):
            src = Gs[int(perm[k])]
            dst_w = Gs[k].shape[1]
            B = np.zeros((src.shape[0], dst_w), np.float32)
            m = min(dst_w, src.shape[1])
            B[:, :m] = src[:, :m]
            out.append(B)
        return out
    raise ValueError(kind)


def four_part_footprint(store, address_bytes: int) -> dict:
    """SUMMED AND PRINTED IN FOUR PARTS, NEVER NETTED.

    1. atoms                      the deployed content
    2. covariance + count         the statistic that makes deletion exact
    3. address book               the read's parameters
    4. coupling / hole table      the request counts that carry the refusal holes
    """
    atoms = int(sum((M.size for M in store.modes))) * 4
    cov = int(sum((w.stat_bytes() for w in store.stats)))
    total = atoms + cov + int(address_bytes) + COUPLING_BYTES + COUNTS_BYTES
    return {
        "p1_atom_bytes": atoms,
        "p2_covariance_plus_count_bytes": cov + COUNTS_BYTES,
        "p2_covariance_bytes": cov,
        "p2_count_bytes": COUNTS_BYTES,
        "p3_address_book_bytes": int(address_bytes),
        "p4_coupling_hole_table_bytes": COUPLING_BYTES,
        "honest_total_bytes": total,
        "reference_pin_bytes": REFERENCE_PIN,
        "times_reference_pin": round(total / REFERENCE_PIN, 4),
        "netted": False,
        "note": "four parts summed, never netted; the 524,288 B column is a REFERENCE only this round -- the single design is 1x its own total",
    }


def load_partition(condition: str):
    z = dict(np.load(PART_VAL if condition == "val" else PART_TEST, allow_pickle=True))
    dead = np.asarray(z["dead"]).ravel().astype(bool)
    pool = np.asarray(z["pool"], np.int64)
    return (z, dead, pool)


def wells_for(corr, K, dead):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from shapes_matched_bytes import ward_balanced_labels

    lab = ward_balanced_labels(corr, K, dead)
    return [np.flatnonzero(lab == k).astype(np.int64) for k in range(K)]
