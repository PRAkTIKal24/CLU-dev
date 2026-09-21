"""w12 hub/single-design-crossover -- shared library for the K-crossover sweep.

NOTHING IN HERE SELECTS K.  The selection rule is declared, dated and frozen in
`/c/chlu/outputs/w12-single-design-crossover/SELECTION-RULE.md`, written at
2026-09-21T11:10:24Z, before any number in this module was produced.

What this module is
-------------------
The w8 single-design pipeline (`scripts/w8_build_matched_bytes.py` +
`scripts/w8_read_matched_bytes.py` + `scripts/w8_score_corners.py`), factored so the
same code path runs at every K in the sweep and so the charter-C controls can be run
against a fitted arm without refitting the store.

THE PIN CHANGED, AND IT IS SAID HERE RATHER THAN IMPLIED.  w8 pinned the FOUR-PART
honest ledger at 524,288 B and let the atom block shrink to fit.  This wave does not:
the rank is 64 per well, capped STRICTLY below the well width, and the object's own
four-part total is what the learned baselines are matched to (1x).  The 524,288 B
number is carried as a REFERENCE COLUMN on every row and is never a budget here.

The four parts -- atoms | covariance+count | address book | coupling/hole table --
are printed separately on every row and are NEVER netted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

D = 2048
REFERENCE_PIN = 524_288          # REFERENCE ONLY this wave, not a budget
RANK_TARGET = 64                 # "rank 64 per well, capped strictly below well width"

#: The two ledger parts that depend on the factor cardinalities and NOT on K, taken from
#: the w8 realised object so the two waves' ledgers compare line by line
#: (`scripts/w8_build_matched_bytes.py:COUNTS_BYTES/COUPLING_BYTES`, themselves the
#: spoke's realised `DowndatableStore.count_ledger()/coupling_ledger()`).
COUNTS_BYTES = 11_488
COUPLING_BYTES = 5_288 + 1_322

PART_VAL = "C:/chlu/outputs/hub-factor-wells/run-val/partitions.npz"
PART_TEST = "C:/chlu/outputs/hub-factor-wells/run/partitions.npz"


def log(m=""):
    print(m, flush=True)


def atoms_for(well_sizes) -> int:
    """A = min(64, min_k d_k - 1).  STRICTLY below the narrowest well width."""
    return int(min(RANK_TARGET, min(int(d) for d in well_sizes) - 1))


def pinned_basis(store):
    """The store's own eigen-ordered direction rows as the basis, plus the mean's
    perpendicular part.  Verbatim from `scripts/w8_read_matched_bytes.py:pinned_basis`
    (itself the spoke's `pinned_span_basis` threshold, 1e-4 relative).  Zero bytes.
    """
    out = []
    for M, r in zip(store.modes, store.deployed_rank):
        V = np.asarray(M[1:1 + r], np.float64).T
        mu = np.asarray(M[0], np.float64)
        perp = mu - V @ (V.T @ mu) if r > 0 else mu
        nrm = float(np.linalg.norm(perp))
        ref = float(np.linalg.norm(mu)) or 1.0
        if nrm / ref > 1e-4:
            V = np.hstack([V, (perp / nrm)[:, None]])
        out.append(np.ascontiguousarray(V))
    return out


def qr_basis(modes, deployed_rank):
    """An orthonormal basis of a PERTURBED atom block.

    `pinned_basis` assumes the direction rows are already orthonormal, which is true
    of the real store and FALSE of a shuffled or random block.  A control must be read
    through a basis of ITS OWN atoms, or the perturbation is silently discarded and the
    control proves nothing.  This is the `CovContentStore.span_bases()` construction
    (QR of the deployed rows), applied to whatever block it is given.
    """
    out = []
    for M, r in zip(modes, deployed_rank):
        if r <= 0:
            out.append(np.zeros((M.shape[1], 0), np.float64))
            continue
        Q = np.linalg.qr(np.asarray(M[:1 + r], np.float64).T)[0]
        out.append(np.ascontiguousarray(Q))
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
    """charter C.2 perturbations of the atom block.  Same shapes, same byte ledger.

    ``zero``            atoms hold nothing -> `landscape_removed`
    ``value_permute``   the SHIPPED C.2 definition (`exp_gen_read.perturb_store`):
                        the atom VALUES are permuted inside each well's mode matrix.
                        Wave 3's mode-INDEX permutation is a no-op for a span read
                        (permuting rows of G does not move its left singular vectors),
                        which is why the shipped definition permutes values.
    ``well_permute``    the task file's literal wording -- atoms SHUFFLED ACROSS WELLS.
                        Well j's block is deployed into well k.  ward_balanced wells
                        differ in width by at most 1, so a block is cropped or
                        zero-padded to the destination width; that is recorded on the row.
                        MEASURED PROPERTY OF THIS IMPLEMENTATION, stated rather than
                        left for a reader to discover: the permutation is uniform with
                        only the IDENTITY rejected, not a derangement, so a well may
                        keep its own block.  A uniform permutation has one fixed point in
                        expectation at any K, i.e. about 1/K of wells; at K=5 seed 0 one
                        of the five wells did keep its own atoms.  That makes this
                        control slightly WEAKER than a derangement would be, which is the
                        conservative direction for a control that must COLLAPSE.  The
                        primary C.2 control is ``value_permute``, which has no such
                        property, and the verdict is taken from it.  The definition is
                        held FIXED across every row this wave rather than improved
                        mid-sweep, so val rows and scored rows carry the same control.
    ``random_span``     atoms replaced by a random Gaussian block of the same shape,
                        rescaled to the same Frobenius norm -> `random_atoms`.
    """
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_921)
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


def four_part_ledger(store, address_bytes: int) -> dict:
    """SUMMED AND PRINTED IN FOUR PARTS, NEVER NETTED.

    1. atoms                      the deployed content
    2. covariance + count         the statistic that makes deletion exact
    3. address book               the read's parameters
    4. coupling / hole table      the request counts that carry the refusal holes
    """
    atoms = int(sum(M.size for M in store.modes)) * 4
    cov = int(sum(w.stat_bytes() for w in store.stats))
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
        "note": ("four parts summed, never netted; the 524,288 B column is a REFERENCE "
                 "only this wave -- the single design is 1x its own total"),
    }


def load_partition(condition: str):
    z = dict(np.load(PART_VAL if condition == "val" else PART_TEST, allow_pickle=True))
    dead = np.asarray(z["dead"]).ravel().astype(bool)
    pool = np.asarray(z["pool"], np.int64)
    return z, dead, pool


def wells_for(corr, K, dead):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from w8_matched_bytes_recut import ward_balanced_labels
    lab = ward_balanced_labels(corr, K, dead)
    return [np.flatnonzero(lab == k).astype(np.int64) for k in range(K)]
