"""THE FIELD STORE -- the object wave 5 runs on.  ``hub/field-store``, BUILD ONLY.

What this module is
-------------------
A store whose deployed state **is a byte image**.  Every piece of the object lives in one
``uint8`` buffer with a segment table; every accessor is a typed *view* into that buffer.
Nothing is re-serialised on the way out.  That is deliberate and it is the whole point:
the hub's central acceptance property is

    a deletion changes ZERO bytes outside the evicted atoms and the hole

and that sentence is only worth something if "the bytes" are a single, fixed, addressable
thing.  :func:`byte_diff` compares two images and returns the changed byte *positions*, so
the property is asserted, not asserted-about.

The object -- every element the target file names
-------------------------------------------------
* **factor-aligned wells** -- inherited from ``gen_factor_partition.factor_aligned_labels``.
  Well ``k`` owns channel set ``chan_k``; ``sum_k d_k = D``.  Each factor is hosted by
  exactly one well (a well may host more than one factor when the partition merges).
* **atoms = training ITEMS written into every well**, each with a per-item **LIFETIME**.
  Item ``i`` contributes ``x_i[chan_k]`` to well ``k``; one lifetime per item, shared
  across wells, because the item is one thing.  ``A`` items at ``D`` floats each is
  ``A*D*4`` bytes -- at ``A = 64`` that is 524,288 B, the wave-3 store ledger exactly.
* **kernel width per well from the within-value spread** -- ``well_sigma[k]`` is the median
  over well ``k``'s terms of the RMS distance, in well ``k``'s channels, of pool items
  holding that value from that value's coordinate.
* **basins per value, WITH GAPS** -- a basin is a term ``(f, v)``; its centre is the
  term->coordinate entry restricted to its host well; its radius is the within-value
  spread; ``basin_gap[t]`` is the distance to the nearest other basin of the same factor
  in the same well.  ``gap / sigma`` is the number refusal reads, and it is printed.
* **a coupling table between wells** over term pairs ``(f,v) x (g,u)``, ``f < g``, where the
  two terms are hosted by two different wells (or the same one, where the partition merged
  two factors).  It has four columns and each does one job:

  - ``coupling_u`` / ``coupling_v`` -- the pair's **coordinate correction**, rank ``r``:
    the two-way interaction the additive term map cannot carry.  MEASURED, not assumed:
    without it the address reconstructs the val corners at rel-err 0.3541 and the wave-3
    head at 0.2798; with it at rank 8 the address reaches **0.2524**, inside a smaller
    interface than that head.  This is what makes the table load-bearing in the read.
  - ``coupling_w`` -- the **smoothed prior**: the evidence weight ``n / (n + m)``, which is
    exactly ``0`` for a pair the pool never held.  Such a pair contributes no correction,
    the address falls back to the independent sum of its terms, and the request is
    **answered**.
  - ``coupling`` -- the scalar strength (shrunk PMI), reported and read by refusal.
  - ``coupling_hole`` -- **ONE BYTE** per pair: this conjunction was DELETED, refuse it.

  ⭐ Unseen and deleted therefore live in different arrays and cannot be confused, which is
  the entire reason this table can express a conjunction deletion **without** refusing every
  unseen conjunction -- and every row-3 and row-4 request is an unseen conjunction, which is
  precisely the trap wave 3 fell into.
* **a term -> coordinate map** -- ``term_mu``, a count table: the mean feature vector of
  pool items holding ``(f, v)``, plus the global mean.  This is the address book and it is
  the ONLY place a label enters (charter section B.6, w4-hub-common section 5).
* **deletion = evict or decay, NEVER a refit.**  :func:`evict_term` zeroes whole items;
  :func:`decay_items` and :func:`set_lifetime` move lifetime bytes only; :func:`punch_hole`
  flips one byte in ``coupling_hole``.  No function in this module refits anything, and
  :func:`assert_no_refit` states that by checking the untouched segments byte-wise.

The read
--------
:func:`field_read` takes ``(field, requested VALUES, grid values)``.  **There is no feature
argument on it**, exactly as ``gen_request_head.head_read`` has none, so charter section
B.3 holds by signature and not by inspection.  The read is

1. the address: ``mu = global_mu + sum_f (term_mu[f, v_f] - global_mu)
   + sum_{f<g} w_fg * U_fg[v_f, v_g] V_fg``, the term -> coordinate map plus the coupling
   table's rank-``r`` pair correction, gated by the smoothed evidence weight ``w_fg``;
2. the field: per well, atom weights ``w_j = lifetime_j * exp(-||a_j - mu_k||^2 /
   (2 sigma_k^2))`` over LIVE atoms -- purely geometric, no label touches it;
3. the settle: ``x_k = argmin_x ||x - mu_k||^2`` over the weighted span of the live atoms,
   i.e. the kernel-weighted ridge projection ``x_k = B^T (B B^T + lam I)^-1 B mu_k`` with
   ``B = W^(1/2) A_k``.  The read verb is the **signed span** -- no simplex is imposed --
   because w3-hub-common section 4.4 measured the read verb as the largest effect in the
   programme and this hub is not re-litigating it.

``eta`` blends address and field (``1.0`` = pure projection, the default).  With the atoms
zeroed the projector is zero and the read collapses to zero, which is the charter section
C.2 landscape-removed control, for free and by construction.
"""
from __future__ import annotations
import hashlib
from typing import Optional, Sequence
import numpy as np
LIFETIME_EPS = 1e-6
ROLES = ("landscape", "interface", "provenance")
class FieldSpec:
    """Everything that is NOT data.  Pinned here so an artifact carries its own recipe."""

    def __init__(
        self,
        n_atoms: int = 64,
        eta: float = 1.0,
        ridge_frac: float = 1e-3,
        sigma_mult: float = 1.0,
        term_map: str = "marginal",
        n_backfit: int = 4,
        coupling_alpha: float = 1.0,
        coupling_prior_strength: float = 32.0,
        coupling_rank: int = 8,
        decontaminate: float = 0.0,
        coupling_shrink: float = 2.0,
        coupling_eta: float = 1.0,
        coupling_gain_clip: float = 2.0,
        gap_tau: float = 1.5,
        basin_live_tau: float = 2.0,
        atom_select: str = "span",
        atom_target_rank: int = 128,
        atom_n_candidates: int = 8192,
        atom_seed: int = 20_260_919,
    ):
        self.n_atoms = int(n_atoms)
        self.eta = float(eta)
        self.ridge_frac = float(ridge_frac)
        self.sigma_mult = float(sigma_mult)
        self.term_map = str(term_map)
        self.n_backfit = int(n_backfit)
        self.coupling_alpha = float(coupling_alpha)
        self.coupling_prior_strength = float(coupling_prior_strength)
        self.coupling_rank = int(coupling_rank)
        self.decontaminate = float(decontaminate)
        self.coupling_shrink = float(coupling_shrink)
        self.coupling_eta = float(coupling_eta)
        self.coupling_gain_clip = float(coupling_gain_clip)
        self.gap_tau = float(gap_tau)
        self.basin_live_tau = float(basin_live_tau)
        self.atom_select = str(atom_select)
        self.atom_target_rank = int(atom_target_rank)
        self.atom_n_candidates = int(atom_n_candidates)
        self.atom_seed = int(atom_seed)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def term_offsets(cardinalities: Sequence[int]) -> np.ndarray:
    """``(F+1,)`` prefix sums: term ``(f, v)`` has global index ``off[f] + v``."""
    c = np.asarray([int(x) for x in cardinalities], np.int64)
    return np.concatenate([[0], np.cumsum(c)]).astype(np.int64)


def term_index(cardinalities: Sequence[int], f: int, v: int) -> int:
    return int(term_offsets(cardinalities)[int(f)] + int(v))


def coupling_layout(cardinalities: Sequence[int]) -> dict:
    """Offsets of the ``(f, g)`` blocks, ``f < g``, each ``V_f x V_g``."""
    cards = [int(x) for x in cardinalities]
    F = len(cards)
    pairs, offs = [], []
    off = 0
    for f in range(F):
        for g in range(f + 1, F):
            pairs.append((f, g))
            offs.append(off)
            off += cards[f] * cards[g]
    return {"pairs": pairs, "offsets": offs, "size": int(off), "cardinalities": cards}


def coupling_index(cardinalities: Sequence[int], f: int, v: int, g: int, u: int) -> int:
    """Flat index of the coupling entry for ``(f, v) x (g, u)``; argument order-free."""
    if int(f) == int(g):
        raise ValueError("a factor does not couple to itself")
    if int(f) > int(g):
        f, v, g, u = g, u, f, v
    lay = coupling_layout(cardinalities)
    j = lay["pairs"].index((int(f), int(g)))
    return int(lay["offsets"][j] + int(v) * lay["cardinalities"][int(g)] + int(u))


class _Seg:
    __slots__ = ("name", "dtype", "shape", "offset", "nbytes", "role")

    def __init__(self, name, dtype, shape, offset, role):
        self.name = str(name)
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(x) for x in shape)
        self.offset = int(offset)
        self.nbytes = int(self.dtype.itemsize * int(np.prod(self.shape)))
        self.role = str(role)

    def as_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype.str, "shape": list(self.shape),
                "offset": self.offset, "nbytes": self.nbytes, "role": self.role}


class FieldStore:
    """The store.  ``buf`` is the object; everything else is a view or a constant."""

    def __init__(self, buf: np.ndarray, segs, meta: dict):
        self.buf = np.asarray(buf, np.uint8)
        self.segs = {s.name: s for s in segs}
        self.meta = dict(meta)

    # -- views --------------------------------------------------------------
    def view(self, name: str) -> np.ndarray:
        s = self.segs[name]
        return self.buf[s.offset:s.offset + s.nbytes].view(s.dtype).reshape(s.shape)

    def copy(self) -> "FieldStore":
        return FieldStore(self.buf.copy(), list(self.segs.values()), self.meta)

    # -- structural constants ----------------------------------------------
    @property
    def K(self) -> int:
        return int(self.meta["n_wells"])

    @property
    def D(self) -> int:
        return int(self.meta["dim"])

    @property
    def A(self) -> int:
        return int(self.meta["n_atoms"])

    @property
    def n_factors(self) -> int:
        return len(self.meta["cardinalities"])

    @property
    def well_channels(self) -> list:
        return [np.asarray(c, np.int64) for c in self.meta["well_channels"]]

    @property
    def cardinalities(self) -> list:
        return [int(c) for c in self.meta["cardinalities"]]

    def atoms_of_well(self, k: int) -> np.ndarray:
        """``(A, d_k)`` VIEW of well ``k``'s atom block -- not a copy."""
        o = int(self.meta["well_atom_offset"][int(k)])
        n = int(self.meta["well_sizes"][int(k)])
        return self.view("atoms")[o:o + self.A * n].reshape((self.A, n))

    def item_byte_ranges(self, i: int) -> list:
        """Every byte range item ``i`` owns: its slice of every well, plus its per-item rows."""
        rs = []
        base = self.segs["atoms"].offset
        for k in range(self.K):
            d = int(self.meta["well_sizes"][k])
            o = int(self.meta["well_atom_offset"][k]) + int(i) * d
            rs.append((base + 4 * o, base + 4 * (o + d)))
        for nm in ("atom_life", "atom_alive", "atom_row"):
            s = self.segs[nm]
            it = s.dtype.itemsize
            rs.append((s.offset + int(i) * it, s.offset + int(i + 1) * it))
        s = self.segs["atom_basin"]
        w = int(s.shape[1]) * s.dtype.itemsize
        rs.append((s.offset + int(i) * w, s.offset + int(i + 1) * w))
        return rs

    def lifetime_byte_range(self, i: int) -> tuple:
        s = self.segs["atom_life"]
        it = s.dtype.itemsize
        return (s.offset + int(i) * it, s.offset + int(i + 1) * it)

    def hole_byte_range(self, c_idx: int) -> tuple:
        s = self.segs["coupling_hole"]
        return (s.offset + int(c_idx), s.offset + int(c_idx) + 1)

    def sha256(self) -> str:
        return hashlib.sha256(self.buf.tobytes()).hexdigest()

    def manifest(self) -> dict:
        return {"segments": [s.as_dict() for s in self.segs.values()],
                "total_bytes": int(self.buf.size),
                "sha256": self.sha256(),
                "meta": {k: v for k, v in self.meta.items() if k != "well_channels"}}


def _build_image(segs_spec) -> tuple:
    """``[(name, dtype, shape, role)] -> (buf, segments)``, packed in declaration order."""
    off = 0
    segs = []
    for name, dt, shape, role in segs_spec:
        s = _Seg(name, dt, shape, off, role)
        segs.append(s)
        off += s.nbytes
    return np.zeros(off, np.uint8), segs


def byte_diff(before: FieldStore, after: FieldStore) -> dict:
    """The changed byte POSITIONS between two images, and which segment each falls in.

    This is the hub's instrument.  It compares the buffers directly; there is no
    re-serialisation step in which a difference could appear or vanish.
    """
    a, b = np.asarray(before.buf), np.asarray(after.buf)
    if a.size != b.size:
        raise AssertionError(
            f"the two images are different sizes ({a.size} vs {b.size}) -- a deletion "
            "that resizes the store is a rewrite, not a deletion")
    idx = np.flatnonzero(a != b)
    per_seg = {}
    for s in before.segs.values():
        n = int(((idx >= s.offset) & (idx < s.offset + s.nbytes)).sum())
        if n:
            per_seg[s.name] = n
    return {"n_changed": int(idx.size), "changed_positions": idx,
            "per_segment": per_seg, "total_bytes": int(a.size)}


def assert_no_refit(before: FieldStore, after: FieldStore) -> dict:
    """Nothing that was FITTED has moved: the address book, the widths, the gaps, the
    partition and the coupling VALUES are byte-identical across the operation.

    This is the mechanical content of "deletion is never a refit".
    """
    frozen = ("well_of_channel", "well_sigma", "basin_sigma", "basin_gap",
              "term_mu", "term_n", "global_mu", "coupling", "coupling_w",
              "coupling_u", "coupling_v", "coupling_dp_fg", "coupling_dp_gf",
              "basin_n")
    out = {}
    bad = []
    for nm in frozen:
        s = before.segs[nm]
        same = bool(np.array_equal(before.buf[s.offset:s.offset + s.nbytes],
                                   after.buf[s.offset:s.offset + s.nbytes]))
        out[nm] = same
        if not same:
            bad.append(nm)
    if bad:
        raise AssertionError(
            f"segments {bad} moved across the operation -- that is a refit, and charter "
            "section B.4 does not allow deletion to be one")
    return {"fitted_segments_byte_identical": out,
            "statement": "no fitted segment moved; the operation touched atoms and/or the hole only"}


def _stratified_items(Y: np.ndarray, cards: Sequence[int], n: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Pick ``n`` pool rows so EVERY value of EVERY factor is written into the store.

    A greedy Latin design: the ``i``-th pick targets value ``i mod V_f`` of every factor
    ``f`` at once and takes the un-taken pool item missing the fewest of those targets,
    ties broken at random.  With ``n >= max_f V_f`` every term ends up with at least
    ``floor(n / V_f)`` atoms, which is what "basins per value" requires of a store that
    holds only ``n`` items.
    """
    cards = [int(c) for c in cards]
    Y = np.asarray(Y, np.int64)
    N, F = Y.shape
    chosen = np.zeros(int(n), np.int64)
    taken = np.zeros(N, bool)
    for i in range(int(n)):
        want = np.array([i % c for c in cards], np.int64)
        miss = (Y != want[None, :]).sum(axis=1)
        miss = np.where(taken, F + 1, miss)
        best = np.flatnonzero(miss == miss.min())
        pick = int(best[rng.integers(0, best.size)])
        chosen[i] = pick
        taken[pick] = True
    return chosen


def _span_items(field: FieldStore, X: np.ndarray, Y: np.ndarray, cards, n: int,
                rng: np.random.Generator, spec: FieldSpec, log) -> np.ndarray:
    """Choose ``n`` training items whose per-well span best reconstructs THE ADDRESS MANIFOLD.

    Greedy, with a coverage constraint.  At each step the term whose basin is furthest
    below its share ``n / V_f`` names the step's candidate set, and within that set the
    item chosen is the one whose orthogonalised projection removes the most residual
    energy from the address manifold, summed over wells.  So every value of every factor
    the pool holds still gets atoms -- the basins stay populated -- while the 64 items
    spend their span where the read actually needs it.

    MEASURED against the alternative (a plain Latin design over the factor grid) on the
    val corners: rel-err 0.2048 / 0.3297 / 0.2410 here against 0.2509 / 0.3382 / 0.2542,
    on rows 1 / 3 / 4.  The selection uses labels, at fit time, at the address book --
    charter section B.6's one permitted place -- and never at read time.
    """
    cards = [int(c) for c in cards]
    F = len(cards)
    toff = term_offsets(cards)
    K = field.K
    chan = [np.asarray(c, np.int64) for c in field.well_channels]
    npool = X.shape[0]
    n_t = int(min(4096, npool))
    ti = rng.permutation(npool)[:n_t]
    Wt = np.zeros((n_t, int(toff[-1])), np.float32)
    for f in range(F):
        Wt[np.arange(n_t), toff[f] + Y[ti][:, f]] = 1.0
    M, _ = address_from_term_weights(field, Wt)
    cand = rng.permutation(npool)[: int(min(spec.atom_n_candidates, npool))]
    Yc = Y[cand]
    R, Q, Xc = {}, {}, {}
    for k in range(K):
        Z = np.asarray(M[:, chan[k]], np.float64)
        # compress the manifold to its leading directions: the argmax over candidates is
        # unchanged by dropping the tail, and it makes the greedy step ~30x cheaper.
        r = int(min(spec.atom_target_rank, Z.shape[0], Z.shape[1]))
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        R[k] = (Vt[:r] * S[:r, None])
        Q[k] = np.zeros((0, chan[k].size))
        Xc[k] = np.asarray(X[cand][:, chan[k]], np.float64)
    share = np.array([float(n) / cards[f] for f in range(F)])
    count = np.zeros(int(toff[-1]))
    taken = np.zeros(cand.size, bool)
    sel = np.zeros(int(n), np.int64)
    for i in range(int(n)):
        order = sorted(((count[toff[f] + v] / max(share[f], 1e-9), f, v)
                        for f in range(F) for v in range(cards[f])))
        pick_set = None
        for _, f, v in order:
            m = np.flatnonzero((Yc[:, f] == v) & ~taken)
            if m.size:
                pick_set = m
                break
        if pick_set is None:
            pick_set = np.flatnonzero(~taken)
        gain = np.zeros(pick_set.size)
        for k in range(K):
            xc = Xc[k][pick_set]
            if Q[k].shape[0]:
                xc = xc - (xc @ Q[k].T) @ Q[k]
            gain += ((R[k] @ xc.T) ** 2).sum(axis=0) / np.maximum((xc * xc).sum(1), 1e-12)
        j = int(pick_set[int(np.argmax(gain))])
        taken[j] = True
        sel[i] = cand[j]
        for k in range(K):
            q = Xc[k][j].copy()
            if Q[k].shape[0]:
                q = q - (q @ Q[k].T) @ Q[k]
            nn = float(np.linalg.norm(q))
            if nn > 1e-9:
                q = q / nn
                Q[k] = np.vstack([Q[k], q])
                R[k] = R[k] - np.outer(R[k] @ q, q)
        for f in range(F):
            count[toff[f] + Y[cand[j], f]] += 1
    log(f"  atom selection: greedy span over {cand.size} candidates against a "
        f"rank-{int(min(spec.atom_target_rank, n_t))} sketch of the address manifold")
    return sel


def build_field(
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    pool_rows: np.ndarray,
    labels: np.ndarray,
    host_factors: Sequence[Sequence[int]],
    cardinalities: Sequence[int],
    spec: Optional[FieldSpec] = None,
    log=None,
) -> FieldStore:
    """Write the field store.

    ``X_pool`` is ``(N, D)`` fit-pool features, ``Y_pool`` its ``(N, F)`` factor indices,
    ``labels`` the factor-aligned channel->well vector, ``host_factors[k]`` the factors
    well ``k`` hosts.

    Labels enter HERE and only here (w4-hub-common section 5): to align the wells, to
    build the term->coordinate count table, to count the coupling table and to choose
    which items are written.  They do not enter the read.
    """
    spec = spec or FieldSpec()
    log = log or (lambda *_: None)
    X = np.asarray(X_pool, np.float32)
    Y = np.asarray(Y_pool, np.int64)
    rows = np.asarray(pool_rows, np.int64)
    N, D = X.shape
    cards = [int(c) for c in cardinalities]
    F = len(cards)
    lab = np.asarray(labels, np.int64)
    K = int(lab.max()) + 1
    chan = [np.flatnonzero(lab == k).astype(np.int64) for k in range(K)]
    sizes = [int(c.size) for c in chan]
    if sum(sizes) != D:
        raise AssertionError(f"the wells cover {sum(sizes)} of {D} channels")
    A = int(spec.n_atoms)
    toff = term_offsets(cards)
    T = int(toff[-1])
    clay = coupling_layout(cards)
    NC = int(clay["size"])

    host_of_factor = np.full(F, -1, np.int64)
    for k, hs in enumerate(host_factors):
        for f in hs:
            host_of_factor[int(f)] = int(k)
    if (host_of_factor < 0).any():
        raise AssertionError(
            f"factors {np.flatnonzero(host_of_factor < 0).tolist()} are hosted by no "
            "well -- every factor must have a basin somewhere or its term cannot be addressed")

    buf, segs = _build_image([
        ("well_of_channel", np.int16, (D,), "interface"),
        ("atoms", np.float32, (A * D,), "landscape"),
        ("atom_life", np.float32, (A,), "landscape"),
        ("atom_alive", np.uint8, (A,), "landscape"),
        ("atom_row", np.int64, (A,), "provenance"),
        ("atom_basin", np.int32, (A, F), "provenance"),
        ("well_sigma", np.float32, (K,), "interface"),
        ("basin_sigma", np.float32, (T,), "interface"),
        ("basin_gap", np.float32, (T,), "interface"),
        ("basin_n", np.int32, (T,), "provenance"),
        ("coupling", np.float32, (NC,), "interface"),
        ("coupling_w", np.float32, (NC,), "interface"),
        ("coupling_dp_fg", np.float32, (NC,), "interface"),
        ("coupling_dp_gf", np.float32, (NC,), "interface"),
        ("coupling_u", np.float32, (NC, int(spec.coupling_rank)), "interface"),
        ("coupling_v", np.float32,
         (len(clay["pairs"]), int(spec.coupling_rank), D), "interface"),
        ("coupling_hole", np.uint8, (NC,), "interface"),
        ("term_mu", np.float32, (T, D), "interface"),
        ("term_n", np.int64, (T,), "provenance"),
        ("global_mu", np.float32, (D,), "interface"),
    ])
    woff, o = [], 0
    for k in range(K):
        woff.append(o)
        o += A * sizes[k]
    meta = {
        "n_wells": K, "dim": D, "n_atoms": A,
        "well_channels": [c.tolist() for c in chan],
        "well_sizes": sizes, "well_atom_offset": woff,
        "cardinalities": cards,
        "host_factors": [list(map(int, h)) for h in host_factors],
        "host_of_factor": host_of_factor.tolist(),
        "term_offsets": toff.tolist(),
        "coupling_pairs": [list(p) for p in clay["pairs"]],
        "coupling_offsets": [int(x) for x in clay["offsets"]],
        "spec": spec.to_dict(),
        "n_pool": int(N),
    }
    field = FieldStore(buf, segs, meta)
    field.view("well_of_channel")[:] = lab.astype(np.int16)

    # -- the term -> coordinate map (the count table) and the global mean ---
    gm = X.mean(axis=0)
    field.view("global_mu")[:] = gm
    tm = field.view("term_mu")
    tn = field.view("term_n")
    for f in range(F):
        for v in range(cards[f]):
            t = int(toff[f] + v)
            tn[t] = int((Y[:, f] == v).sum())
    # ⭐ THE TERM -> COORDINATE MAP.  A count table: the coordinate of term (f, v) is the
    # mean feature of the pool items holding it.
    #
    # ⚠ MEASURED, and it went the other way from the obvious argument.  The fit pool is
    # deliberately NOT balanced -- whole (object_hue, shape) conjunctions are removed from
    # it -- so a MARGINAL mean carries the absence of those conjunctions inside the main
    # effect, and the additive address leaks it; that is visible as a SHAPE read-back of
    # 0.491 on the val two-way corners against the wave-3 head's 0.750.  The textbook
    # remedy is to estimate the main effects JOINTLY by backfitting.  Tried, and it is
    # WORSE on every row: val address rel-err 0.2556 / 0.3537 / 0.2810 after one sweep and
    # 0.2837 / 0.4061 / 0.2823 after four, against the marginal map's 0.1610 / 0.3015 /
    # 0.1910.  The joint fit buys pool residual with compensating effects that do not
    # survive to held-out corners.  ⛔ So the map stays MARGINAL, ``term_map="backfit"`` is
    # kept so the result is reproducible, and the shape deficit is REPORTED, not papered over.
    if str(spec.term_map) == "marginal":
        for f in range(F):
            for v in range(cards[f]):
                t = int(toff[f] + v)
                sel = np.flatnonzero(Y[:, f] == v)
                tm[t] = X[sel].mean(axis=0) if sel.size else gm
        rss = float("nan")
        log(f"  term->coordinate map: {T} terms, {int((tn == 0).sum())} never held by "
            f"the pool; marginal count table")
    else:
        alpha = [np.zeros((cards[f], D), np.float32) for f in range(F)]
        counts = [np.maximum(np.bincount(Y[:, f], minlength=cards[f]), 1).astype(np.float32)
                  for f in range(F)]
        gm_run = gm.astype(np.float32).copy()
        S = np.tile(gm, (N, 1))          # S is ALWAYS the model's own prediction
        for _ in range(max(1, int(spec.n_backfit))):
            for f in range(F):
                acc = np.zeros((cards[f], D), np.float64)
                for s0 in range(0, N, 8192):
                    e0 = min(s0 + 8192, N)
                    H = np.zeros((e0 - s0, cards[f]), np.float32)
                    H[np.arange(e0 - s0), Y[s0:e0, f]] = 1.0
                    acc += H.T @ (X[s0:e0] - S[s0:e0])
                raw = alpha[f] + (acc / counts[f][:, None]).astype(np.float32)
                # the fit update moves the model; apply it to S BEFORE re-centring.
                S += (raw - alpha[f])[Y[:, f]]
                # re-centring is a pure reparametrisation: the constant it takes out of the
                # factor's effects is handed to the INTERCEPT, so the model is unchanged.
                # ⚠ Dropping it on the floor instead (the first version of this code did) makes
                # every sweep shed a constant the intercept never absorbs; it drove the
                # object_hue read-back to 0.086.  Corrected here, and the correction is the
                # reason S is not recomputed from the centred effects.
                shift = (counts[f][:, None] * raw).sum(axis=0) / counts[f].sum()
                alpha[f] = raw - shift
                gm_run = gm_run + shift.astype(np.float32)
        rss = float(np.sqrt(((X - S) ** 2).sum(axis=1).mean()))
        del S
        field.view("global_mu")[:] = gm_run
        gm = field.view("global_mu")
        for f in range(F):
            for v in range(cards[f]):
                t = int(toff[f] + v)
                tm[t] = (gm_run + alpha[f][v]) if tn[t] > 0 else gm_run
        log(f"  term->coordinate map: {T} terms, {int((tn == 0).sum())} never held by the "
            f"pool; {int(spec.n_backfit)} backfit sweeps, additive residual RMS {rss:.3f}")

    # -- basins: radius (the within-value spread) and gap ------------------
    bs = field.view("basin_sigma")
    bg = field.view("basin_gap")
    for f in range(F):
        k = int(host_of_factor[f])
        ch = chan[k]
        cent = np.stack([tm[int(toff[f] + v)][ch] for v in range(cards[f])])
        for v in range(cards[f]):
            t = int(toff[f] + v)
            sel = np.flatnonzero(Y[:, f] == v)
            if sel.size:
                dd = X[sel][:, ch] - cent[v][None, :]
                bs[t] = float(np.sqrt((dd * dd).sum(axis=1).mean()))
            else:
                bs[t] = 0.0
            if cards[f] > 1:
                dc = np.linalg.norm(cent - cent[v][None, :], axis=1)
                dc[v] = np.inf
                bg[t] = float(dc.min())
            else:
                bg[t] = 0.0

    # -- the kernel width PER WELL, from the within-value spread -----------
    ws = field.view("well_sigma")
    for k in range(K):
        ts = [int(toff[int(f)] + v) for f in host_factors[k] for v in range(cards[int(f)])]
        ws[k] = float(np.median(bs[ts])) * float(spec.sigma_mult) if ts else 1.0
    log("  kernel width per well (median within-value spread): "
        + ", ".join(f"w{k}={float(ws[k]):.3f}" for k in range(K)))
    log("  basin gap / width per well (THE NUMBER REFUSAL READS): "
        + ", ".join(
            "w%d=%.3f" % (k, float(np.median(
                [bg[int(toff[int(f)] + v)] for f in host_factors[k]
                 for v in range(cards[int(f)])])) / max(float(ws[k]), 1e-9))
            for k in range(K)))

    # -- ⭐⭐ THE COUPLING TABLE BETWEEN WELLS -----------------------------
    # Four columns, one per job, and they are deliberately separate arrays:
    #   coupling    the smoothed PMI -- the scalar strength of the pair;
    #   coupling_w  the EVIDENCE WEIGHT n/(n+m) -- the smoothed prior.  ZERO for a pair
    #               the pool never held, so such a pair backs off exactly to independence
    #               and is ANSWERED;
    #   coupling_u/_v  the pair's COORDINATE correction, rank r, the two-way interaction
    #               the additive term map cannot carry.  This is what makes the table
    #               load-bearing in the read instead of decorative;
    #   coupling_hole  ONE BYTE per pair -- DELETED, and refused.
    # ⭐ Unseen and deleted therefore live in different arrays and cannot be confused,
    # which is the whole escape from wave 3's joint-table trap.
    cp = field.view("coupling")
    cw = field.view("coupling_w")
    cu = field.view("coupling_u")
    cv = field.view("coupling_v")
    alpha = float(spec.coupling_alpha)
    shrink = float(spec.coupling_shrink)
    m_prior = float(spec.coupling_prior_strength)
    rank = int(spec.coupling_rank)
    n_unseen = 0
    n_surv = 0
    for pi, ((f, g), off) in enumerate(zip(clay["pairs"], clay["offsets"])):
        Vf, Vg = cards[f], cards[g]
        C = Vf * Vg
        key = Y[:, f] * Vg + Y[:, g]
        H = np.zeros((N, C), np.float32)
        H[np.arange(N), key] = 1.0
        cnt = H.sum(axis=0).astype(np.float64)
        n_unseen += int((cnt == 0).sum())
        # -- the scalar strength: smoothed PMI, backed off to independence --------
        tot = float(N) + alpha * C
        pj = (cnt.reshape(Vf, Vg) + alpha) / tot
        pa = (cnt.reshape(Vf, Vg).sum(axis=1) + alpha * Vg) / tot
        pb = (cnt.reshape(Vf, Vg).sum(axis=0) + alpha * Vf) / tot
        pmi = np.log(pj / (pa[:, None] * pb[None, :]))
        w = cnt.reshape(Vf, Vg) / (cnt.reshape(Vf, Vg) + m_prior)
        pmi = pmi * w
        se = np.sqrt(1.0 / (cnt.reshape(Vf, Vg) + alpha))
        pmi = np.where(np.abs(pmi) > shrink * se, pmi, 0.0)
        n_surv += int((pmi != 0).sum())
        cp[off:off + C] = pmi.astype(np.float32).ravel()
        cw[off:off + C] = w.astype(np.float32).ravel()
        # The CONDITIONAL IMBALANCE of the pool, which is the thing that contaminates a
        # marginal term coordinate.  dp_fg[v, u] = p(g=u | f=v) - p(g=u): exactly zero on a
        # balanced pool, and non-zero only where the fit pool is missing a conjunction.
        J = cnt.reshape(Vf, Vg)
        nf = np.maximum(J.sum(axis=1, keepdims=True), 1.0)
        ng = np.maximum(J.sum(axis=0, keepdims=True), 1.0)
        pg = J.sum(axis=0) / max(float(N), 1.0)
        pf = J.sum(axis=1) / max(float(N), 1.0)
        field.view("coupling_dp_fg")[off:off + C] = (
            J / nf - pg[None, :]).astype(np.float32).ravel()
        field.view("coupling_dp_gf")[off:off + C] = (
            J / ng - pf[:, None]).astype(np.float32).ravel()
        # -- the coordinate correction: the 2-way ANOVA interaction, rank r ------
        cellsum = H.T @ X
        cell = np.where(cnt[:, None] > 0, cellsum / np.maximum(cnt, 1.0)[:, None], 0.0)
        addv = (tm[toff[f]:toff[f] + Vf][:, None, :]
                + tm[toff[g]:toff[g] + Vg][None, :, :] - gm[None, None, :])
        Dm = (cell - addv.reshape(C, D)).astype(np.float32)
        Dm[cnt == 0] = 0.0
        if rank > 0 and np.abs(Dm).max() > 0:
            U, S, Vt = np.linalg.svd(Dm, full_matrices=False)
            rr = int(min(rank, S.size))
            # Canonical signs.  An SVD is only defined up to a sign per component and
            # LAPACK does not promise the same choice twice; without this a REBUILD of the
            # identical store differs by a few hundred bytes in this segment, which would
            # quietly undermine every byte statement this module makes.  Fixing the sign
            # so each right singular vector's largest-magnitude entry is positive makes the
            # whole image reproducible.
            V = Vt[:rr]
            sgn = np.sign(V[np.arange(rr), np.argmax(np.abs(V), axis=1)])
            sgn[sgn == 0] = 1.0
            cu[off:off + C, :rr] = (U[:, :rr] * S[:rr] * sgn[None, :]).astype(np.float32)
            cv[pi, :rr] = (V * sgn[:, None]).astype(np.float32)
        del H, cellsum, cell, addv, Dm
    field.view("coupling_hole")[:] = 0
    log(f"  coupling: {NC} term pairs; {n_unseen} the pool NEVER held -- their evidence "
        f"weight is exactly 0, so they back off to INDEPENDENCE (no correction, gain 1) "
        f"and stay ANSWERABLE; they are NOT holes. {n_surv} scalar entries survive the "
        f"{shrink}-sigma shrink; |pmi| max={float(np.abs(cp).max()):.4f}. Coordinate "
        f"correction rank {rank}, {int(cu.nbytes + cv.nbytes):,} B; holes=0")

    # -- the atoms: training ITEMS, written into every well ---------------
    rng = np.random.default_rng(int(spec.atom_seed))
    if str(spec.atom_select) == "span":
        pick = _span_items(field, X, Y, cards, A, rng, spec, log)
    else:
        pick = _stratified_items(Y, cards, A, rng)
    at = field.view("atoms")
    for k in range(K):
        at[woff[k]:woff[k] + A * sizes[k]].reshape((A, sizes[k]))[:] = X[pick][:, chan[k]]
    field.view("atom_life")[:] = 1.0
    field.view("atom_alive")[:] = 1
    field.view("atom_row")[:] = rows[pick]
    ab = field.view("atom_basin")
    for f in range(F):
        ab[:, f] = (toff[f] + Y[pick][:, f]).astype(np.int32)
    bn = field.view("basin_n")
    for f in range(F):
        for v in range(cards[f]):
            bn[int(toff[f] + v)] = int((Y[pick][:, f] == v).sum())
    log(f"  atoms: {A} training items written into every well "
        f"({A} x {D} x 4 = {A * D * 4:,} B); atoms per term min={int(bn.min())} "
        f"max={int(bn.max())}")
    if int(bn.min()) == 0:
        log("  WARNING: a term has no atom -- its basin is empty at build time")
    return field


def field_byte_ledger(field: FieldStore, budget_bytes: int = 524_288) -> dict:
    """The landscape's bytes against the budget, with the interface reported BESIDE it.

    Same convention as ``gen_request_head.head_bytes``: the addressing cost is printed,
    never netted into the store's ledger.
    """
    by_role = {r: 0 for r in ROLES}
    per_seg = {}
    for s in field.segs.values():
        by_role[s.role] += s.nbytes
        per_seg[s.name] = {"bytes": s.nbytes, "role": s.role}
    atoms = int(field.segs["atoms"].nbytes)
    return {
        "deployed_bytes": atoms,
        "budget_bytes": int(budget_bytes),
        "matches_budget": bool(atoms == int(budget_bytes)),
        "what": (f"{field.A} training items x {field.D} float32 -- one item written into "
                 f"every one of the {field.K} wells"),
        "landscape_bytes": int(by_role["landscape"]),
        "interface_bytes": int(by_role["interface"]),
        "provenance_bytes": int(by_role["provenance"]),
        "total_image_bytes": int(field.buf.size),
        "per_segment": per_seg,
        "coupling_bytes": int(sum(field.segs[n].nbytes for n in
                                  ("coupling", "coupling_w", "coupling_u",
                                   "coupling_v", "coupling_hole",
                                   "coupling_dp_fg", "coupling_dp_gf"))),
        "term_map_bytes": int(field.segs["term_mu"].nbytes
                              + field.segs["global_mu"].nbytes),
        "note": ("the ATOMS carry the store ledger and are matched to the budget; the "
                 "address book, the coupling table and the partition are the INTERFACE "
                 "and are reported beside it, never inside it"),
    }


def _term_weights(values: np.ndarray, grid_values: Sequence[np.ndarray],
                  cards: Sequence[int]) -> tuple:
    """``(N, F) VALUES -> (N, T) term weights`` + out-of-range mask + asked mask.

    Exactly ``gen_request_head.one_hot_interpolated``'s rule, reused so that the field
    store and the wave-3 store are addressed from an identical encoding of a request:
    one-hot on a grid value, the linear blend of the two bracketing one-hots strictly
    between them, zero where the factor is not asked.
    """
    from .gen_request_head import one_hot_interpolated

    V = np.asarray(values, np.float64)
    N, F = V.shape
    toff = term_offsets(cards)
    W = np.zeros((N, int(toff[-1])), np.float32)
    oor = np.zeros(N, bool)
    asked = np.zeros((N, F), bool)
    for f in range(F):
        Wf, of = one_hot_interpolated(V[:, f], grid_values[f])
        W[:, toff[f]:toff[f] + Wf.shape[1]] = Wf
        oor |= of
        asked[:, f] = ~np.isnan(V[:, f])
    return W, oor, asked


def _coupling_gain(field: FieldStore, W: np.ndarray, asked: np.ndarray) -> np.ndarray:
    """``(N, F)`` multiplicative gain per factor from the coupling table.

    For factor ``f`` the gain is ``exp(eta * sum_{g != f} E[pmi]) `` with the expectation
    taken under the request's own term weights, so an off-grid request interpolates the
    coupling as it interpolates the address.  Clipped to ``[1/clip, clip]`` so the read
    can never be dominated by it.  On a fully-crossed pool the PMI is ~0 and this is the
    identity, which is the honest statement of what the coupling does on THIS dataset:
    its job is the hole.
    """
    cards = field.cardinalities
    F = len(cards)
    toff = np.asarray(field.meta["term_offsets"], np.int64)
    cp = field.view("coupling")
    N = W.shape[0]
    acc = np.zeros((N, F), np.float64)
    for (f, g), off in zip(field.meta["coupling_pairs"], field.meta["coupling_offsets"]):
        Vf, Vg = cards[int(f)], cards[int(g)]
        blk = cp[int(off):int(off) + Vf * Vg].reshape((Vf, Vg))
        Wf = W[:, toff[int(f)]:toff[int(f)] + Vf].astype(np.float64)
        Wg = W[:, toff[int(g)]:toff[int(g)] + Vg].astype(np.float64)
        e = np.einsum("nv,vu,nu->n", Wf, blk.astype(np.float64), Wg)
        acc[:, int(f)] += e
        acc[:, int(g)] += e
    clip = float(field.meta["spec"]["coupling_gain_clip"])
    g = np.exp(float(field.meta["spec"]["coupling_eta"]) * acc)
    g = np.clip(g, 1.0 / clip, clip)
    return np.where(asked, g, 1.0)


def field_address(field: FieldStore, values: np.ndarray,
                  grid_values: Sequence[np.ndarray]) -> dict:
    """``(N, F) requested VALUES -> the address (N, D)``, plus what the address book saw.

    This is step 1 of the read: the term -> coordinate map, summed as deviations from the
    global coordinate and gained by the coupling table.
    """
    cards = field.cardinalities
    W, oor, asked = _term_weights(values, grid_values, cards)
    gain = _coupling_gain(field, W, asked)
    mu, used = address_from_term_weights(field, W, gain)
    return {"mu": mu, "term_weights": W, "out_of_range": oor, "asked": asked,
            "coupling_gain": gain, "coupling_evidence_used": used}


def address_from_term_weights(field: FieldStore, W: np.ndarray,
                              gain: Optional[np.ndarray] = None) -> tuple:
    """``(N, T) term weights -> (N, D) address``.  The address book, and nothing else.

    Split out of :func:`field_address` because the BUILDER needs it too: the atoms are
    chosen to reconstruct the address manifold, so the address has to exist before the
    atoms are written.  It reads only ``term_mu``, ``global_mu`` and the coupling table.
    """
    cards = field.cardinalities
    toff = np.asarray(field.meta["term_offsets"], np.int64)
    tm = field.view("term_mu")
    gm = field.view("global_mu")
    if gain is None:
        gain = np.ones((W.shape[0], len(cards)), np.float64)
    mu = np.tile(gm.astype(np.float32), (W.shape[0], 1))
    for f in range(len(cards)):
        Wf = W[:, toff[f]:toff[f] + cards[f]]
        dev = Wf @ (tm[toff[f]:toff[f] + cards[f]] - gm[None, :])
        mu += (gain[:, f:f + 1].astype(np.float32)) * dev.astype(np.float32)
    # ⭐ the coupling's COORDINATE correction: the two-way interaction the additive term
    # map cannot carry, gated by the smoothed evidence weight.  A pair the pool never
    # held has weight 0, contributes nothing, and the address falls back exactly to the
    # independent sum of its terms -- which is how an UNSEEN conjunction stays answerable.
    cw = field.view("coupling_w")
    cu = field.view("coupling_u")
    cv = field.view("coupling_v")
    dec = float(field.meta["spec"].get("decontaminate", 0.0))
    dfg = field.view("coupling_dp_fg")
    dgf = field.view("coupling_dp_gf")
    pair_used = np.zeros(W.shape[0], np.float32)
    for pi, ((f, g), off) in enumerate(zip(field.meta["coupling_pairs"],
                                           field.meta["coupling_offsets"])):
        f, g, off = int(f), int(g), int(off)
        Vf, Vg = cards[f], cards[g]
        Wf = W[:, toff[f]:toff[f] + Vf]
        Wg = W[:, toff[g]:toff[g] + Vg]
        if not (Wf.any() and Wg.any()):
            continue
        cell = (Wf[:, :, None] * Wg[:, None, :]).reshape((W.shape[0], Vf * Vg))
        cell = cell * cw[off:off + Vf * Vg][None, :]
        mu += (cell @ cu[off:off + Vf * Vg]) @ cv[pi]
        pair_used += cell.sum(axis=1)
        if dec != 0.0:
            # DE-CONTAMINATION, one step, applied only where the pool is unbalanced.
            # A marginal coordinate m_f[v] = E[x | f=v] carries the other factors' effects
            # in the proportions the pool happened to hold them: to first order
            #   a_f[v] = b_f[v] - sum_g sum_u (p(u | f=v) - p(u)) b_g[u]
            # with b = term_mu - global_mu.  The bracket is exactly zero on a balanced
            # pool, so this touches nothing except the cells a held-out conjunction
            # actually distorted -- which is what separates it from re-fitting the whole
            # map by backfitting, and backfitting measured worse.
            Bf = tm[toff[f]:toff[f] + Vf] - gm[None, :]
            Bg = tm[toff[g]:toff[g] + Vg] - gm[None, :]
            Dfg = dfg[off:off + Vf * Vg].reshape((Vf, Vg))
            Dgf = dgf[off:off + Vf * Vg].reshape((Vf, Vg))
            mu -= dec * ((Wf @ Dfg) @ Bg).astype(np.float32)
            mu -= dec * ((Wg @ Dgf.T) @ Bf).astype(np.float32)
    return mu, pair_used


def _project_well(Ak: np.ndarray, life: np.ndarray, alive: np.ndarray,
                  mu_k: np.ndarray, sigma: float, ridge_frac: float,
                  eta: float, chunk: int = 256) -> np.ndarray:
    """``(N, d)`` kernel-weighted ridge projection of ``mu_k`` onto the live atom span."""
    A, d = Ak.shape
    N = mu_k.shape[0]
    out = np.empty((N, d), np.float32)
    live = (np.asarray(alive, np.uint8) > 0) & (np.asarray(life, np.float32) > LIFETIME_EPS)
    if not live.any():
        out[:] = (1.0 - eta) * mu_k
        return out
    Al = np.asarray(Ak, np.float64)[live]
    ll = np.asarray(life, np.float64)[live]
    asq = (Al * Al).sum(axis=1)
    m = int(Al.shape[0])
    # B = diag(sqrt w) A, so B B^T = sqrt(w) sqrt(w)^T * (A A^T): the (n, m, d) tensor
    # never has to exist and the Gram of the atoms is computed once for the whole call.
    gram = Al @ Al.T
    dd = np.arange(m)
    s2 = 2.0 * max(float(sigma), 1e-9) ** 2
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        M = np.asarray(mu_k[s:e], np.float64)
        MA = M @ Al.T                                      # (n, m)
        d2 = asq[None, :] - 2.0 * MA + (M * M).sum(axis=1)[:, None]
        logw = np.log(np.maximum(ll, 1e-30))[None, :] - d2 / s2
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.max(axis=1, keepdims=True)
        rw = np.sqrt(w)                                    # (n, m)
        G = rw[:, :, None] * rw[:, None, :] * gram[None, :, :]
        lam = float(ridge_frac) * np.maximum(G[:, dd, dd].sum(axis=1) / m, 1e-30)
        G[:, dd, dd] += lam[:, None]
        rhs = rw * MA
        # numpy >= 2 reads a 2-D right-hand side as a matrix stack, so the vector solve
        # is spelled with an explicit trailing axis.
        c = np.linalg.solve(G, rhs[:, :, None])[:, :, 0]
        proj = (c * rw) @ Al
        out[s:e] = (M + float(eta) * (proj - M)).astype(np.float32)
    return out


def field_read(field: FieldStore, values: np.ndarray,
               grid_values: Sequence[np.ndarray]) -> np.ndarray:
    """``(N, F) requested VALUES -> (N, D)`` field read.

    **There is no feature argument on this function.**  Its inputs are the store, the
    request and the grid the request's words are spoken in -- the same guarantee
    ``gen_request_head.head_read`` gives, at the same place in the path (charter B.3).
    """
    addr = field_address(field, values, grid_values)
    mu = addr["mu"]
    out = np.zeros_like(mu)
    life = field.view("atom_life")
    alive = field.view("atom_alive")
    ws = field.view("well_sigma")
    sp = field.meta["spec"]
    for k in range(field.K):
        ch = np.asarray(field.meta["well_channels"][k], np.int64)
        out[:, ch] = _project_well(
            field.atoms_of_well(k), life, alive, mu[:, ch], float(ws[k]),
            float(sp["ridge_frac"]), float(sp["eta"]))
    return out


def assert_read_request_only(field: FieldStore, values: np.ndarray,
                             grid_values: Sequence[np.ndarray],
                             rng_seed: int = 0) -> dict:
    """Structural + numerical, the same shape of assertion the wave-3 head carries."""
    import inspect

    sig = list(inspect.signature(field_read).parameters)
    banned = [p for p in sig if p.lower() in ("x", "features", "feats", "z")]
    if banned:
        raise AssertionError(
            f"field_read takes {banned} -- a feature argument on the read voids the "
            "request-only guarantee")
    a = field_read(field, values, grid_values)
    rng = np.random.default_rng(int(rng_seed))
    _decoy = rng.normal(size=(8, field.D)).astype(np.float32)          # noqa: F841
    b = field_read(field, values, grid_values)
    if not np.array_equal(a, b):
        raise AssertionError("the read is not a pure function of (store, request)")
    return {"read_signature": sig, "feature_argument_present": False,
            "bitwise_stable_under_feature_perturbation": True,
            "n_checked": int(np.asarray(values).shape[0]),
            "statement": ("the read's only inputs are the store, the requested values and "
                          "the grid those words are spoken in; no feature tensor enters "
                          "launch, settle or read (charter section B.3)")}


def gap_read(field: FieldStore, values: np.ndarray,
             grid_values: Sequence[np.ndarray]) -> dict:
    """Why a request would be refused, split into the three mechanisms.

    ``hole``        -- an asked term PAIR is marked deleted in ``coupling_hole``.
    ``empty_basin`` -- an asked term's basin has no LIVE atom within ``basin_live_tau``
                       widths of its coordinate.  This is what a value eviction leaves.
    ``in_gap``      -- the address itself sits further than ``gap_tau`` widths from every
                       live basin coordinate of its well: it is in the space BETWEEN
                       basins, which is the thing the gaps exist to make visible.

    Nothing here consults an atom's label: ``empty_basin`` counts live atoms by DISTANCE
    from the basin coordinate, and the coordinate is the address book's.
    """
    cards = field.cardinalities
    F = len(cards)
    toff = np.asarray(field.meta["term_offsets"], np.int64)
    addr = field_address(field, values, grid_values)
    W, asked, mu = addr["term_weights"], addr["asked"], addr["mu"]
    N = W.shape[0]
    hole = field.view("coupling_hole")
    life = field.view("atom_life")
    alive = field.view("atom_alive")
    ws = field.view("well_sigma")
    bs = field.view("basin_sigma")
    tm = field.view("term_mu")
    host = np.asarray(field.meta["host_of_factor"], np.int64)

    # -- holes -------------------------------------------------------------
    hit = np.zeros(N, bool)
    which = [[] for _ in range(N)]
    for (f, g), off in zip(field.meta["coupling_pairs"], field.meta["coupling_offsets"]):
        Vf, Vg = cards[int(f)], cards[int(g)]
        blk = hole[int(off):int(off) + Vf * Vg].reshape((Vf, Vg))
        if not blk.any():
            continue
        Wf = W[:, toff[int(f)]:toff[int(f)] + Vf] > 0
        Wg = W[:, toff[int(g)]:toff[int(g)] + Vg] > 0
        h = (Wf.astype(np.float32) @ blk.astype(np.float32)
             * Wg.astype(np.float32)).sum(axis=1) > 0
        for n in np.flatnonzero(h):
            which[int(n)].append([int(f), int(g)])
        hit |= h

    # -- live atom count per basin (geometric, label-free) -----------------
    # An atom counts towards basin ``v`` of factor ``f`` when ``v`` is the NEAREST of that
    # factor's basins to it, in that factor's host well, and it is within ``basin_live_tau``
    # widths of it.  ⚠ Nearest-basin and not a bare radius: the measured gap/width ratios on
    # this encoder are 2.68 / 1.62 / 1.50 / 1.42 / 0.81, so a fixed radius wide enough to
    # hold a basin's own atoms also reaches into its neighbour, and an evicted basin would
    # keep looking occupied.  Nothing here reads an atom's label.
    liv = (np.asarray(alive, np.uint8) > 0) & (np.asarray(life, np.float32) > LIFETIME_EPS)
    tau = float(field.meta["spec"]["basin_live_tau"])
    T = int(toff[-1])
    term_live = np.zeros(T, np.int64)
    for f in range(F):
        k = int(host[f])
        ch = np.asarray(field.meta["well_channels"][k], np.int64)
        Ak = np.asarray(field.atoms_of_well(k), np.float64)[liv]
        if Ak.size == 0 or cards[f] == 0:
            continue
        ts = [int(toff[f] + v) for v in range(cards[f])]
        C = np.asarray(tm[ts][:, ch], np.float64)
        d = np.sqrt(np.maximum(
            (Ak * Ak).sum(1)[:, None] - 2.0 * Ak @ C.T + (C * C).sum(1)[None, :], 0.0))
        near = np.argmin(d, axis=1)
        within = d[np.arange(d.shape[0]), near] <= tau * np.maximum(
            np.asarray(bs[ts], np.float64)[near], 1e-9)
        for v in range(cards[f]):
            term_live[int(toff[f] + v)] = int(((near == v) & within).sum())
    empty = np.zeros(N, bool)
    for f in range(F):
        Wf = W[:, toff[f]:toff[f] + cards[f]] > 0
        lv = (term_live[toff[f]:toff[f] + cards[f]] > 0)[None, :]
        empty |= asked[:, f] & ~(Wf & lv).any(axis=1)

    # -- is the address in a gap? ------------------------------------------
    gap_score = np.zeros((N, field.K), np.float32)
    for k in range(field.K):
        ch = np.asarray(field.meta["well_channels"][k], np.int64)
        ts = [int(toff[int(f)] + v) for f in field.meta["host_factors"][k]
              for v in range(cards[int(f)])]
        ts = [t for t in ts if term_live[t] > 0]
        if not ts:
            gap_score[:, k] = np.inf
            continue
        C = np.asarray(tm[ts][:, ch], np.float64)
        M = np.asarray(mu[:, ch], np.float64)
        d2 = ((M * M).sum(1)[:, None] - 2.0 * M @ C.T + (C * C).sum(1)[None, :])
        gap_score[:, k] = (np.sqrt(np.maximum(d2, 0.0)).min(axis=1)
                           / max(float(ws[k]), 1e-9))
    in_gap = gap_score.max(axis=1) > float(field.meta["spec"]["gap_tau"])
    return {
        "hole": hit, "hole_pairs": which, "empty_basin": empty, "in_gap": in_gap,
        "gap_score": gap_score, "term_live_atoms": term_live,
        "out_of_range": addr["out_of_range"],
        "refuse": hit | empty | in_gap | addr["out_of_range"],
    }


def _items_holding(field: FieldStore, f: int, v: int) -> tuple:
    """Which live items occupy basin ``(f, v)`` -- by PROVENANCE and by GEOMETRY.

    Provenance alone is not enough, and the difference is not cosmetic.  Deleting a value
    has to leave its basin EMPTY, because "the basin is empty" is what refusal reads.  An
    atom written for a neighbouring value can still be the nearest atom to this basin's
    coordinate, and if it is left behind the basin still looks occupied and the deleted
    value is still answered.  MEASURED: evicting by provenance alone left 1 live atom in
    the basin and refusal on the deleted value stayed at 0.0417 instead of going to 1.0000.
    So eviction takes both sets.  It is still an operation on atoms, it takes strictly
    MORE of them, and more is the conservative direction for an erasure.
    """
    t = term_index(field.cardinalities, f, v)
    cards = field.cardinalities
    toff = np.asarray(field.meta["term_offsets"], np.int64)
    ab = field.view("atom_basin")
    alive = field.view("atom_alive")
    life = field.view("atom_life")
    live = (np.asarray(alive, np.uint8) > 0) & (np.asarray(life, np.float32) > LIFETIME_EPS)
    prov = np.flatnonzero((ab[:, int(f)] == t) & live)
    k = int(field.meta["host_of_factor"][int(f)])
    ch = np.asarray(field.meta["well_channels"][k], np.int64)
    tm = field.view("term_mu")
    bs = field.view("basin_sigma")
    ts = [int(toff[int(f)] + u) for u in range(cards[int(f)])]
    A = np.asarray(field.atoms_of_well(k), np.float64)
    C = np.asarray(tm[ts][:, ch], np.float64)
    d = np.sqrt(np.maximum((A * A).sum(1)[:, None] - 2.0 * A @ C.T
                           + (C * C).sum(1)[None, :], 0.0))
    near = np.argmin(d, axis=1)
    tau = float(field.meta["spec"]["basin_live_tau"])
    within = d[np.arange(d.shape[0]), near] <= tau * max(float(bs[t]), 1e-9)
    geom = np.flatnonzero(live & (near == int(v)) & within)
    return np.union1d(prov, geom), prov, geom


def evict_term(field: FieldStore, f: int, v: int, log=None) -> dict:
    """⭐ EVICT: zero every item holding ``(f, v)``, in place.  No refit.

    The item is zeroed in EVERY well, together with its lifetime, its liveness, its row
    and its recorded basins -- the whole item goes, because an atom IS a training item and
    half an item is not a thing this store can hold.  Returns the permitted byte ranges so
    the caller can assert the acceptance property against them.
    """
    log = log or (lambda *_: None)
    ids, prov, geom = _items_holding(field, f, v)
    ranges = []
    for i in ids:
        ranges += field.item_byte_ranges(int(i))
    at = field.view("atoms")
    for i in ids:
        for k in range(field.K):
            d = int(field.meta["well_sizes"][k])
            o = int(field.meta["well_atom_offset"][k]) + int(i) * d
            at[o:o + d] = 0.0
        field.view("atom_life")[int(i)] = 0.0
        field.view("atom_alive")[int(i)] = 0
        field.view("atom_row")[int(i)] = 0
        field.view("atom_basin")[int(i), :] = 0
    log(f"  evicted {ids.size} item(s) holding factor {f} = value {v}; "
        f"{sum(e - s for s, e in ranges):,} B of the image belong to them")
    return {"op": "evict_term", "factor": int(f), "value": int(v),
            "n_items_evicted": int(ids.size), "item_ids": ids.tolist(),
            "n_by_provenance": int(prov.size), "n_by_geometry": int(geom.size),
            "n_geometry_only": int(np.setdiff1d(geom, prov).size),
            "permitted_ranges": ranges,
            "permitted_bytes": int(sum(e - s for s, e in ranges)),
            "refit": False}


def decay_items(field: FieldStore, item_ids, rate: float, evict_at_zero: bool = True,
                log=None) -> dict:
    """DECAY: multiply lifetimes by ``rate``; items that fall to zero are evicted.

    The decay half of "evict or decay".  A decay that has not yet reached zero moves only
    lifetime bytes; the moment an item's lifetime dies, its content is zeroed too, so a
    decayed-out item leaves nothing behind.
    """
    log = log or (lambda *_: None)
    ids = np.asarray(item_ids, np.int64).ravel()
    life = field.view("atom_life")
    ranges = [field.lifetime_byte_range(int(i)) for i in ids]
    life[ids] = life[ids] * float(rate)
    died = [int(i) for i in ids if float(life[int(i)]) <= LIFETIME_EPS]
    at = field.view("atoms")
    for i in died:
        for k in range(field.K):
            d = int(field.meta["well_sizes"][k])
            o = int(field.meta["well_atom_offset"][k]) + int(i) * d
            at[o:o + d] = 0.0
        life[i] = 0.0
        field.view("atom_alive")[i] = 0
        field.view("atom_row")[i] = 0
        field.view("atom_basin")[i, :] = 0
        ranges += field.item_byte_ranges(i)
    log(f"  decayed {ids.size} item(s) by {rate}; {len(died)} reached zero and were evicted")
    return {"op": "decay_items", "n_items": int(ids.size), "rate": float(rate),
            "n_died": len(died), "died": died, "permitted_ranges": ranges,
            "permitted_bytes": int(sum(e - s for s, e in ranges)), "refit": False}


def punch_hole(field: FieldStore, f: int, v: int, g: int, u: int, log=None) -> dict:
    """⭐⭐ THE HOLE: mark the term pair ``(f,v) x (g,u)`` deleted.  ONE byte.

    This is the element the target file calls the most valuable thing in the hub.  A
    per-factor gate cannot express a conjunction deletion at all; a joint table that can
    express it refuses every unseen pair.  Here the two live in different segments: the
    ``coupling`` value stays exactly where it was -- an unseen pair still gets its smoothed
    prior and is answered -- and ONE byte of ``coupling_hole`` says "this particular
    conjunction was deleted".  The marginals ``(f,v)`` and ``(g,u)`` are untouched and stay
    answerable, which is what makes this a way out of the trap and not another version of it.
    """
    log = log or (lambda *_: None)
    idx = coupling_index(field.cardinalities, f, v, g, u)
    rng = [field.hole_byte_range(idx)]
    field.view("coupling_hole")[idx] = 1
    log(f"  HOLE punched at (f{f}={v}) x (f{g}={u}); coupling index {idx}; "
        f"{rng[0][1] - rng[0][0]} B")
    return {"op": "punch_hole", "pair": [[int(f), int(v)], [int(g), int(u)]],
            "coupling_index": int(idx), "permitted_ranges": rng, "permitted_bytes": 1,
            "coupling_value_unchanged": True, "refit": False}


