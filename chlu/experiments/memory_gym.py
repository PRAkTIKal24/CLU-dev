"""The internal **memory gym** — Track 1 task generators (C2W1, charter §2.2).

The gym is *development currency* and is **never the paper's primary claim**
(charter §2.2). Its sole KPI is the **dynamics dividend**

    dividend = (full CLU) - (its own settle-deleted / matched-bytes launder)

on the harness that ``full-clu-harness`` landed, with the three harness-native
controls firing on **every** cell and a byte ledger published beside every
number. ⭐ *A dividend of ≈0 or negative at v0 is the charter's own stated
expectation and falsifies nothing; a POSITIVE cell is suspicious and goes through
every control plus a seed re-run before it is written down.*

**One family per structural opening** (charter §2.1), so the gym spans the
hypothesis space rather than sampling it — and **each family carries a written
metric-native argument**, because intervention §6 criterion 4 (*if the query lives
in the same metric space as the stored keys, a classical method is the provable
ceiling*) is a theorem about our situation, confirmed four times:

``overload``  (a) beyond-capacity compression
    ~3x more items than the **atom budget** was sized for. ⚠ The overload is in
    atoms-per-item, **not** in slots: the harness's eviction *re-draws* the freed
    atom group (verbatim erasure), so a slot-overloaded CLU evicts exactly as
    verbatim as a table does and opening (a) would be untestable.
    *Metric-native argument:* the byte-matched table is **never** budget-limited
    here (see :func:`byte_ratio_law`), so F1 fails criterion 4 as a dividend task
    and is retained as the **byte-frontier instrument** — that is declared in
    advance, not discovered at review.

``aggregate`` (b) non-metric-native queries
    the answer is a convex combination of **two** stored payloads, computed
    *between* basins; no stored payload is within ``payload_tol`` of it (queries
    that violate that are dropped at construction).
    *Metric-native argument:* an arg-min lookup returns a *stored* payload, so its
    error is bounded below by a positive constant. Criterion 4 holds against
    arg-min but **not** against aggregation-augmented classical baselines, so the
    family ships with its own strong control: a 2-NN mean at **+0 B**
    (:func:`chlu.eval.dividend.knn_mean_launder`), uniform *and* inverse-distance
    weighted. That is the honest ceiling and it is expected to win.

``recency``   (c) trajectory information
    the target is *which of two items is more recent* — not a stored value, and
    not a function of the query-key metric (recency lives in the landscape,
    because a per-item lifetime physically shallows that item's own atom rows).
    *Metric-native argument:* no arg-min over ``(keys, payloads)`` can exceed
    chance ⇒ criterion 4 holds against the frozen launder. ⛔ **But a table's row
    order already encodes insertion order**, so ``order_aware_launder`` answers the
    family exactly at **+0 B** — it travels with every F3 number, and a positive
    F3 dividend against the frozen control is therefore a laundering artefact of
    that control's byte allocation, not a dynamics dividend.

``manifold``  (d) manifold-valued memories via flat directions
    a scoped, honest **stub with its blocker named** (the task file's own
    acceptance for this opening): the store is given a spectator axis the write
    objective never constrains, and we measure whether reads retain a *set* of
    settled states along it. ⛔ Blocker, named in advance:
    ``train_memory_landscape`` digs point wells, not valleys — a genuine
    manifold-valued memory needs a **ridge write** (multi-row collinear targets
    for one item) and the controller has no verb for it. The optional
    ``ridge`` arm measures the blocker instead of asserting it.
    *Laundering:* the ``echo_launder`` (return the launch coordinate) scores 1.000
    at +0 B, so any positive F4 dividend is **by-construction** and is reported as
    a capability measurement, never as a dividend (intervention §8.3).

**Cross-cutting stream properties every family carries** (charter §2.2): capacity
pressure · interference (crowd targets in the write loss) · a **deletion demand**
· a **revisit** of an earlier address (regime re-identification) · at least four
**consolidation windows**, which is what finally makes monitor #6 applicable.

⚠ **Config lives here, not in ``chlu/config.py``** (C2W1 file-ownership rule:
C1W27 owns two blocks of that file this wave). :class:`GymConfig` is a plain
dataclass with a ``from_mapping`` override path and a ``clu_overrides`` passthrough
onto :class:`~chlu.core.clu_system.CluSystemConfig`, so the gym stays
config-driven without touching the shared config module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chlu.core.clu_system import CluSystemConfig
from chlu.core.memory_potentials import designed_payloads, designed_sites

#: The four gym families, one per charter §2.1 opening.
FAMILIES = ("overload", "aggregate", "recency", "manifold")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class GymConfig:
    """Gym-level knobs. Store/read knobs go through :attr:`clu_overrides`.

    Every default here is either the harness's shipped band or a value whose
    provenance is in the docstring; the per-family deviations live in
    :data:`FAMILY_DEFAULTS` so a cell's non-default set is inspectable
    (:meth:`as_flag_table`).
    """

    family: str = "overload"
    arm: str = "base"
    seed: int = 0

    # -- the stream --------------------------------------------------------
    n_offer: int = 8
    budget: Optional[int] = 6
    #: capacity the ATOM budget is sized for (the overload denominator)
    reference_capacity: int = 6
    #: capacity of the store; None => n_offer (every offer gets a slot)
    capacity: Optional[int] = None
    deletion: bool = True
    revisit: bool = True
    collision_offer: bool = True
    #: |a| of a deliberately over-excursion item, 0 => none (monitor #11 probe)
    reach_stress_amp: float = 0.0
    payload_lo: float = -1.0
    payload_hi: float = 1.0

    # -- consolidation (monitor #6 needs >= 4 windows) ---------------------
    consolidate_every: int = 3
    min_consolidations: int = 5

    # -- the query law -----------------------------------------------------
    n_query_per_item: int = 8
    n_query_per_pair: int = 8
    #: pairs are used when their address distance is <= this multiple of ``sep``
    max_pair_dist_mult: float = 1.7
    #: interpolation window for the aggregate family (0.5 = exact midpoint)
    lam_lo: float = 0.35
    lam_hi: float = 0.65
    #: occupancy kernel radius as a fraction of ``sep`` (trajectory read-out)
    occupancy_radius_frac: float = 0.5
    #: ⛔ **C2W2 D4 — the recency-family harness DEFECT fix** (default ``False``
    #: = shipped behaviour, bit-for-bit; a regression test asserts it).
    #:
    #: ``queries_recency`` asks *"which of THESE TWO items was written more
    #: recently"* and :func:`score_index` grades it against a **2-way chance of
    #: 0.5** — but every CLU-side arm (``argmax`` occupancy, ``point_assign``)
    #: and the frozen settle-deleted launder answer an unrestricted ``K``-way
    #: question over all live sites, while only ``order_aware_launder(k=2)`` is
    #: restricted to the pair. Measured (seed 0, K=5, 9 pairs, 72 queries): the
    #: CLU's answer falls **outside its own pair 19.4 %** of the time, and
    #: restricting it lifts trajectory accuracy **0.4306 -> 0.5556** and the
    #: launder **0.4861 -> 0.5139**. The gym's sub-chance ``0.3019`` was that
    #: category error, not a null.
    #:
    #: With this ``True``, every index-family arm chooses **between the query's
    #: own two candidates**, which is the question the labels and the chance
    #: rate both assume.
    restrict_index_to_pair: bool = False
    #: manifold family: spectator launch grid
    n_manifold_launch: int = 12
    manifold_launch_span: float = 0.6
    ridge_write: bool = False
    ridge_targets: int = 5

    # -- passthrough to CluSystemConfig -----------------------------------
    clu_overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, overrides: Optional[dict] = None) -> "GymConfig":
        """Build from a YAML/JSON mapping, ignoring unknown keys (as
        ``chlu.config.load_config`` does, so an old project file cannot crash a
        new schema)."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in dict(overrides or {}).items() if k in known}
        return cls(**kw)

    def as_flag_table(self) -> Dict[str, Any]:
        """Every non-default gym flag in effect — the flag-provenance table."""
        base = GymConfig()
        out: Dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v != getattr(base, f.name):
                out[f.name] = v
        return out

    @property
    def overload_factor(self) -> float:
        """Items per item-worth of atom budget (``3.0`` = the charter's ~3x)."""
        k = int(self.capacity if self.capacity is not None else self.n_offer)
        return float(k) / float(max(self.reference_capacity, 1))

    def build_clu(self) -> CluSystemConfig:
        """The :class:`CluSystemConfig` this gym cell runs on.

        ⭐ The atom budget is pinned to :attr:`reference_capacity`, **not** to the
        store's capacity, via ``atoms_per_item`` — that is what makes the overload
        family an overload of *atoms per item* rather than of slots.
        """
        cap = int(self.capacity if self.capacity is not None else self.n_offer)
        over = dict(self.clu_overrides)
        atoms_per_item = int(over.pop("atoms_per_item", 32))
        # pin the TOTAL atom budget to reference_capacity * atoms_per_item, then
        # express it as a per-slot number for the actual capacity.
        total = int(atoms_per_item * self.reference_capacity)
        kw = dict(
            capacity=cap,
            budget=(int(self.budget) if self.budget is not None else None),
            atoms_per_item=max(1, total // max(cap, 1)),
            min_atoms=max(total, cap),
            min_atoms_base=max(total, cap),
            min_atoms_c=1.0,
            n_query_per_item=int(self.n_query_per_item),
            seed=int(self.seed),
        )
        kw.update(over)
        return CluSystemConfig(**kw)


#: Per-family deviations from :class:`GymConfig`'s defaults. Anything not listed
#: is the shipped harness band (declared in the report's provenance table).
FAMILY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # (a) beyond-capacity compression: 3x the atom budget's reference load, and
    # NO eviction (capacity = n_offer), because eviction is verbatim erasure.
    # ⚠ `d_safe_override`: the derived radius is 2*0.3 + 2.576*0.15 = 0.9864 while
    # 18 farthest-point sites in the unit 4-ball achieve only ~2*18^(-1/4) = 0.971,
    # so the store would fail its OWN admission gate (doctrine I-13) and refuse
    # every offer after the first — a spotless, empty store, i.e. the degenerate
    # configuration the task forbids settling into. 0.58 ~ 0.6*sep_expected, the
    # harness's own S4 convention, keeps the stream admissible while `sep/sigma_q`
    # stays IN band (~6.5 vs the 5.15 bar), so monitor #8 is not the thing under
    # test here.
    "overload": dict(
        n_offer=18, capacity=18, budget=18, reference_capacity=6,
        deletion=True, revisit=True, collision_offer=False,
        n_query_per_item=4, consolidate_every=4,
        clu_overrides=dict(stage_admission=True, d_safe_override=0.58),
    ),
    # (b) non-metric-native queries: real capacity pressure, in-band separation.
    "aggregate": dict(
        n_offer=8, capacity=6, budget=6, reference_capacity=6,
        n_query_per_pair=8, consolidate_every=2,
        clu_overrides=dict(stage_admission=True),
    ),
    # (c) trajectory information: lifetimes ON, so recency is in the landscape.
    "recency": dict(
        n_offer=8, capacity=6, budget=6, reference_capacity=6,
        n_query_per_pair=8, consolidate_every=2,
        clu_overrides=dict(stage_admission=True, stage_lifetimes=True, leak=0.06),
    ),
    # (d) flat directions: one unconstrained spectator axis.
    "manifold": dict(
        n_offer=6, capacity=6, budget=6, reference_capacity=6,
        deletion=False, revisit=False, collision_offer=False,
        n_manifold_launch=12,
        clu_overrides=dict(stage_admission=True, n_spectator=1),
    ),
}

#: Named arms (a deviation from the family default, run as its own cell).
ARMS: Dict[str, Dict[str, Any]] = {
    # F2's mechanism arm: overlapping basins, where a settle CAN stop between
    # wells. ⚠ This is the harness's collapsed band (sep/sigma_q ~ 3.1), and the
    # admission radius must be taken deliberately out of band for the stage to
    # exist at all (permitted basin interaction and the merge certificate are
    # mutually exclusive by construction — the harness's own finding).
    "aggregate/tight": dict(
        clu_overrides=dict(ball_radius=0.45, d_safe_override=0.32)),
    # F4's blocker arm: a ridge write (multi-row collinear targets).
    "manifold/ridge": dict(ridge_write=True),
    # ⭐ F1's BYTE-PARITY FRONTIER: the atom budget is the swept axis. The label is
    # the *reference* per-item budget; at 3x overload the atoms actually owned per
    # live item is `ref * reference_capacity / capacity`, and the reported x-axis
    # is always the MEASURED byte ratio. `ref3` is the architectural floor: one
    # atom group per item cannot go below one atom.
    "overload/ref3": dict(clu_overrides=dict(atoms_per_item=3)),
    "overload/ref8": dict(clu_overrides=dict(atoms_per_item=8)),
    "overload/ref16": dict(clu_overrides=dict(atoms_per_item=16)),
    # ⭐ THE 1x-LOAD REFERENCE. Without it the frontier is confounded: a flat
    # accuracy-vs-bytes curve at 3x overload cannot distinguish "the atom budget
    # is too small" from "18 items at d=4 is simply past `K_learned(4)`". These
    # arms differ from `base`/`ref*` in the LOAD ONLY (same override radius, same
    # stream construction), so the difference is attributable.
    "overload/load1x": dict(n_offer=6, capacity=6, budget=6),
    "overload/load1x_ref8": dict(n_offer=6, capacity=6, budget=6,
                                 clu_overrides=dict(atoms_per_item=8)),
    "overload/load1x_ref3": dict(n_offer=6, capacity=6, budget=6,
                                 clu_overrides=dict(atoms_per_item=3)),
    # ⭐ the SHIPPED-BUDGET anchor: 341*6 = 2046 atoms ~ the harness's own 2048, so
    # the frontier curve connects to `full-clu-harness`'s measured point (byte
    # ratio 478.7x at K=6 / 359.2x at K=8, decode 0.906) on the gym's own stream.
    "overload/load1x_shipped": dict(n_offer=6, capacity=6, budget=6,
                                    clu_overrides=dict(atoms_per_item=341)),
    # #11 reach probe: an over-excursion item WITH the write-time gate disabled,
    # so the certificate's runtime form gets something to fire on. With the gate
    # on, `admit.reach` refuses the item instead and #11 stays clear — both
    # outcomes are reportable and neither is a failure.
    "overload/reach_free": dict(
        reach_stress_amp=1.6, clu_overrides=dict(stage_admission=False)),
}


def gym_config(family: str, arm: str = "base", seed: int = 0,
               **overrides) -> GymConfig:
    """A gym config for ``family``/``arm``, with the family defaults applied.

    ``clu_overrides`` merge (dict-update) rather than replace, so an arm may
    change one store knob without restating the family's.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown gym family {family!r}; known: {FAMILIES}")
    kw: Dict[str, Any] = dict(FAMILY_DEFAULTS.get(family, {}))
    arm_kw = dict(ARMS.get(f"{family}/{arm}", {})) if arm != "base" else {}
    if arm != "base" and f"{family}/{arm}" not in ARMS:
        raise ValueError(f"unknown arm {family}/{arm}")
    clu = dict(kw.pop("clu_overrides", {}))
    clu.update(arm_kw.pop("clu_overrides", {}))
    clu.update(dict(overrides.pop("clu_overrides", {})))
    kw.update(arm_kw)
    kw.update(overrides)
    return GymConfig(family=family, arm=arm, seed=int(seed),
                     clu_overrides=clu, **kw)


def byte_ratio_law(atoms_per_item: float, addr_dim: int = 4,
                   payload_dim: int = 1, n_spectator: int = 0) -> float:
    """``full/launder`` byte ratio, in closed form and **independent of K**.

    With one atom group per item (which is what makes the write masked/C3-local)
    each item costs ``atoms_per_item * (dim + 2)`` floats of ``V_theta`` against
    the launder's ``dim`` floats per row, plus the retained address codebook::

        ratio = atoms_per_item * (dim + 2)/dim + addr_dim/dim

    ⛔ **Hence ``ratio >= atoms_per_item``, and matched bytes requires
    ``atoms_per_item < 1`` — atoms shared between items, which the masked write
    forbids by construction.** Matched bytes is therefore *unreachable* at v0,
    not merely unachieved, and a byte-matched table always holds MORE items than
    the CLU does, so it is never budget-limited. That is why the overload family
    reports a frontier and not a dividend.
    """
    dim = float(addr_dim + payload_dim + n_spectator)
    return float(atoms_per_item) * (dim + 2.0) / dim + float(addr_dim) / dim


# --------------------------------------------------------------------------
# the write stream
# --------------------------------------------------------------------------
@dataclass
class GymStream:
    """A gym write stream plus the ground truth the scorers need.

    Attributes:
        items: the write-stream rows handed to
            :meth:`~chlu.core.clu_system.CluSystem.write_stream`.
        offered: one row per *distinct offered item* with ``item_id``, ``address``,
            ``payload`` and ``order`` (insertion index — the recency ground truth).
        chunks: item-index boundaries at which a consolidation is run.
    """

    items: List[dict]
    offered: List[dict]
    chunks: List[int]

    @property
    def addresses(self) -> np.ndarray:
        return np.stack([np.asarray(o["address"], dtype=float) for o in self.offered])

    @property
    def payloads(self) -> np.ndarray:
        return np.asarray([[float(o["payload"])] for o in self.offered], dtype=float)

    @property
    def order(self) -> np.ndarray:
        return np.asarray([int(o["order"]) for o in self.offered], dtype=int)


def make_gym_stream(gcfg: GymConfig, ccfg: Optional[CluSystemConfig] = None
                    ) -> GymStream:
    """Build the family's write stream (farthest-point sites, designed payloads).

    Carries every cross-cutting property the charter asks for: capacity pressure
    (``n_offer > budget``), a **deletion demand** naming an item that is still
    live, a **revisit** of an earlier address (regime re-identification), an
    optional near-duplicate **collision offer** so the admission gate has
    something it can refuse (without one a farthest-point stream legitimately
    gives monitor #3 a fire rate of 0), and an optional over-excursion item that
    probes the reach certificate.
    """
    ccfg = ccfg or gcfg.build_clu()
    n = int(gcfg.n_offer)
    sites = np.asarray(designed_sites(ccfg.addr_dim, n, R=ccfg.ball_radius,
                                      seed=gcfg.seed))
    pays = np.asarray(designed_payloads(n, seed=gcfg.seed, lo=gcfg.payload_lo,
                                        hi=gcfg.payload_hi))
    items: List[dict] = []
    offered: List[dict] = []
    chunks: List[int] = []
    order = 0
    for i in range(n):
        items.append({"item_id": i, "address": sites[i], "payload": float(pays[i])})
        offered.append({"item_id": i, "address": sites[i], "payload": float(pays[i]),
                        "order": order})
        order += 1
        if gcfg.collision_offer and i == n // 2:
            items.append({"item_id": 1000 + i, "address": sites[i] * 1.005,
                          "payload": float(-pays[i])})
        # the deletion demand must name an item that is still LIVE (a delete of an
        # absent id is a silent no-op that empties the stage of its demand).
        if gcfg.deletion and i == max(2, n - 3):
            items.append({"item_id": i - 1, "delete": True})
        # the REVISIT: the same address offered again later, so the store has to
        # re-identify a regime it has already seen rather than allocate a new one.
        if gcfg.revisit and i == n - 1:
            j = max(0, n // 3)
            items.append({"item_id": 2000 + j, "address": sites[j] * 1.0,
                          "payload": float(pays[j])})
        if gcfg.reach_stress_amp and i == n - 2:
            items.append({"item_id": 3000, "address": sites[(i + 1) % n] * 0.5,
                          "payload": float(gcfg.reach_stress_amp)})
        if gcfg.consolidate_every and (i + 1) % gcfg.consolidate_every == 0:
            chunks.append(len(items))
    if len(items) not in chunks:
        chunks.append(len(items))
    # monitor #6 needs >= window+1 = 4 (loss, acq) pairs; pad by re-consolidating
    while len(chunks) < int(gcfg.min_consolidations):
        chunks.append(len(items))
    return GymStream(items=items, offered=offered, chunks=chunks)


# --------------------------------------------------------------------------
# the query sets
# --------------------------------------------------------------------------
@dataclass
class QuerySet:
    """A family's queries plus everything a scorer or a launder needs.

    Attributes:
        q0: ``(n, dim)`` launch points handed to :meth:`CluSystem.read`.
        keys: ``(n, addr_dim)`` the address part — what a launder is given.
        target: ``(n, ...)`` the family's target (a value, an index, a coordinate).
        label: ``(n,)`` integer label into :attr:`alphabet` (``-1`` if none).
        alphabet: ``(A, m)`` decode alphabet, or ``None``.
        kind: ``"value" | "index" | "coord"`` — which scorer applies.
        meta: family extras (pairs, launch coordinates, live ids, ...).
    """

    q0: np.ndarray
    keys: np.ndarray
    target: np.ndarray
    label: np.ndarray
    alphabet: Optional[np.ndarray]
    kind: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.q0.shape[0])


def _jitter(rng: np.random.Generator, n: int, d: int, sigma: float) -> np.ndarray:
    return rng.normal(size=(n, d)) * float(sigma)


def _launch(ccfg: CluSystemConfig, addr: np.ndarray,
            spectator: Optional[np.ndarray] = None) -> np.ndarray:
    """A launch buffer: address block filled, payload block zeroed (the read
    zeroes it anyway), spectator block optional."""
    q0 = np.zeros((addr.shape[0], ccfg.dim), dtype=np.float32)
    q0[:, : ccfg.addr_dim] = addr
    if spectator is not None and ccfg.n_spectator > 0:
        j = ccfg.addr_dim + ccfg.payload_dim
        q0[:, j: j + ccfg.n_spectator] = np.asarray(spectator).reshape(-1, ccfg.n_spectator)
    return q0


def queries_overload(gcfg: GymConfig, ccfg: CluSystemConfig, stream: GymStream,
                     rng: np.random.Generator) -> QuerySet:
    """F1: every **offered** item is queried, live or not (that is the point)."""
    addr = stream.addresses
    pays = stream.payloads
    n_per = int(gcfg.n_query_per_item)
    lab = np.repeat(np.arange(addr.shape[0]), n_per)
    q_addr = addr[lab] + _jitter(rng, lab.size, ccfg.addr_dim, ccfg.query_sigma)
    return QuerySet(
        q0=_launch(ccfg, q_addr), keys=q_addr, target=pays[lab], label=lab,
        alphabet=pays, kind="value",
        meta={"item_ids": [o["item_id"] for o in stream.offered],
              "offer_order": stream.order[lab]},
    )


def _pairs_within(centers: np.ndarray, mult: float) -> List[Tuple[int, int]]:
    k = centers.shape[0]
    if k < 2:
        return []
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    sep = float(np.min(d))
    out = [(i, j) for i in range(k) for j in range(i + 1, k)
           if d[i, j] <= mult * sep]
    if not out:  # always keep the closest pair
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        out = [(int(min(i, j)), int(max(i, j)))]
    return out


def queries_aggregate(gcfg: GymConfig, ccfg: CluSystemConfig, centers: np.ndarray,
                      pays: np.ndarray, rng: np.random.Generator) -> QuerySet:
    """F2: a convex combination of two neighbouring items, computed *between* basins.

    The target ``(1-lam) a_i + lam a_j`` is not any stored payload; queries whose
    target lands within ``payload_tol`` of a stored payload are **dropped at
    construction**, so the arg-min launder cannot be accidentally right.
    """
    pairs = _pairs_within(centers, gcfg.max_pair_dist_mult)
    rows_q, rows_t, rows_pair, rows_lam = [], [], [], []
    tol = float(ccfg.payload_tol)
    for (i, j) in pairs:
        for _ in range(int(gcfg.n_query_per_pair)):
            lam = float(rng.uniform(gcfg.lam_lo, gcfg.lam_hi))
            tgt = (1.0 - lam) * pays[i] + lam * pays[j]
            if np.min(np.abs(pays - tgt[None, :])) < tol:
                continue  # a stored payload would answer it: not a (b)-query
            pos = (1.0 - lam) * centers[i] + lam * centers[j]
            rows_q.append(pos + _jitter(rng, 1, ccfg.addr_dim, ccfg.query_sigma)[0])
            rows_t.append(tgt)
            rows_pair.append((i, j))
            rows_lam.append(lam)
    if not rows_q:
        raise RuntimeError("aggregate family produced no admissible queries")
    q_addr = np.stack(rows_q)
    tgt = np.stack(rows_t)
    return QuerySet(
        q0=_launch(ccfg, q_addr), keys=q_addr, target=tgt,
        label=np.full((tgt.shape[0],), -1, dtype=int), alphabet=None, kind="value",
        meta={"pairs": np.asarray(rows_pair, dtype=int),
              "lam": np.asarray(rows_lam, dtype=float),
              "n_pairs": len(pairs), "stored_payloads": pays},
    )


def queries_recency(gcfg: GymConfig, ccfg: CluSystemConfig, centers: np.ndarray,
                    born: np.ndarray, rng: np.random.Generator) -> QuerySet:
    """F3: at the midpoint of two live items — **which was written more recently?**

    Not a stored value and not a function of the query-key metric.
    """
    pairs = _pairs_within(centers, gcfg.max_pair_dist_mult)
    pairs = [(i, j) for (i, j) in pairs if born[i] != born[j]]
    if not pairs:
        raise RuntimeError("recency family produced no admissible pairs")
    rows_q, rows_lab, rows_pair = [], [], []
    for (i, j) in pairs:
        later = int(i if born[i] > born[j] else j)
        for _ in range(int(gcfg.n_query_per_pair)):
            pos = 0.5 * (centers[i] + centers[j])
            rows_q.append(pos + _jitter(rng, 1, ccfg.addr_dim, ccfg.query_sigma)[0])
            rows_lab.append(later)
            rows_pair.append((i, j))
    q_addr = np.stack(rows_q)
    lab = np.asarray(rows_lab, dtype=int)
    return QuerySet(
        q0=_launch(ccfg, q_addr), keys=q_addr, target=lab, label=lab,
        alphabet=None, kind="index",
        meta={"pairs": np.asarray(rows_pair, dtype=int), "born": born,
              "n_pairs": len(pairs)},
    )


def queries_manifold(gcfg: GymConfig, ccfg: CluSystemConfig, centers: np.ndarray,
                     rng: np.random.Generator) -> QuerySet:
    """F4: the spectator axis is swept at each item's own address.

    The target is the **launch** spectator coordinate: a store that holds a
    *manifold* of settled states returns it, a store that holds a point collapses
    it. ⚠ The ``echo_launder`` returns it exactly at +0 B — which is why this is
    a capability measurement and not a dividend.
    """
    if ccfg.n_spectator < 1:
        raise RuntimeError("manifold family needs n_spectator >= 1")
    grid = np.linspace(-gcfg.manifold_launch_span, gcfg.manifold_launch_span,
                       int(gcfg.n_manifold_launch))
    k = centers.shape[0]
    lab = np.repeat(np.arange(k), grid.size)
    spec = np.tile(grid, k)
    q_addr = centers[lab] + _jitter(rng, lab.size, ccfg.addr_dim,
                                    0.25 * ccfg.query_sigma)
    return QuerySet(
        q0=_launch(ccfg, q_addr, spectator=spec), keys=q_addr,
        target=spec.reshape(-1, 1), label=lab, alphabet=None, kind="coord",
        meta={"grid": grid, "spectator_index": ccfg.addr_dim + ccfg.payload_dim},
    )


# --------------------------------------------------------------------------
# gym-side read-outs (the system's own psi stays payload-shaped; these are the
# gym's, and they are what makes the point-vs-trajectory ablation possible)
# --------------------------------------------------------------------------
def readout_settled(res, ccfg: CluSystemConfig) -> np.ndarray:
    """The payload channels of the settled point — the v0 read 26 waves used."""
    j = ccfg.addr_dim
    return np.asarray(res.state.q_star)[:, j: j + ccfg.payload_dim]


def readout_tail_mean(res, ccfg: CluSystemConfig, tail_frac: float = 0.5
                      ) -> np.ndarray:
    """Mean payload channel over the phase-2 tail (a trajectory read-out)."""
    traj = np.asarray(res.traj)
    ph = np.asarray(res.phase)
    idx = np.where(ph == 2)[0]
    if idx.size == 0:
        idx = np.arange(traj.shape[1])
    i0 = idx[max(0, idx.size - max(1, int(tail_frac * idx.size)))]
    j = ccfg.addr_dim
    return traj[:, i0:, j: j + ccfg.payload_dim].mean(axis=1)


def readout_spectator(res, ccfg: CluSystemConfig) -> np.ndarray:
    """The settled spectator coordinate (the manifold family's read-out)."""
    j = ccfg.addr_dim + ccfg.payload_dim
    return np.asarray(res.state.q_star)[:, j: j + ccfg.n_spectator]


def readout_occupancy(res, centers: np.ndarray, radius: float,
                      phase: int = 2) -> np.ndarray:
    """⭐ **The trajectory read-out**: soft time-occupancy near each stored site.

    ``occ[q, i] = mean_t softmax_i(-|q_t - c_i|^2 / 2 radius^2)`` over the
    phase-``phase`` trajectory points. A settled point can name one item; a
    trajectory that passes near competing wells encodes a *distribution* over
    answers (charter §2.1(c)) — this is the cheapest handcrafted ψ that reads it,
    and scoring pillar (c) with a handcrafted ψ is the declared v0 limitation
    (the learned read-out is ``trainability-spike``'s).
    """
    traj = np.asarray(res.traj)
    ph = np.asarray(res.phase)
    sel = np.where(ph == phase)[0]
    if sel.size == 0:
        sel = np.arange(traj.shape[1])
    d = int(centers.shape[1])
    pos = traj[:, sel, :d]
    dist2 = np.sum((pos[:, :, None, :] - centers[None, None, :, :]) ** 2, axis=-1)
    r = max(float(radius), 1e-6)
    w = np.exp(-dist2 / (2.0 * r * r))
    w = w / np.maximum(np.sum(w, axis=-1, keepdims=True), 1e-30)
    return np.asarray(np.mean(w, axis=1))


def restrict_to_pair(scores: np.ndarray, pairs: np.ndarray, *,
                     higher_is_better: bool = True) -> np.ndarray:
    """⛔ **C2W2 D4 harness fix**: choose between the query's OWN two candidates.

    ``scores`` is ``(n_queries, K)`` (occupancy, or negated distance); ``pairs``
    is ``(n_queries, 2)`` from ``QuerySet.meta``. Returns the chosen **global**
    item index per query, so the result is scored by the unchanged
    :func:`score_index` against the unchanged 2-way chance of 0.5.

    Why this is a fix and not a thumb on the scale: the recency question, its
    labels and its chance rate are all 2-way, and the +0 B ``order_aware``
    substitute was **already** restricted this way (``k=2``). Leaving the CLU
    arms unrestricted graded a 6-way answer on a 2-way curve — which is how a
    working store measured *below* chance.
    """
    sc = np.asarray(scores, dtype=float)
    pr = np.asarray(pairs, dtype=int)
    take = np.take_along_axis(sc, pr, axis=1)
    pick = np.argmax(take, axis=1) if higher_is_better else np.argmin(take, axis=1)
    return np.asarray(np.take_along_axis(pr, pick[:, None], axis=1).ravel())


def readout_point_assign(res, centers: np.ndarray) -> np.ndarray:
    """The **point** arm of the ablation: nearest stored site to ``q*``."""
    d = int(centers.shape[1])
    q = np.asarray(res.state.q_star)[:, :d]
    return np.argmin(np.linalg.norm(q[:, None, :] - centers[None, :, :], axis=-1),
                     axis=1)


# --------------------------------------------------------------------------
# scorers — one per QuerySet.kind, so the CLU and every launder are scored by
# EXACTLY the same function (this is what keeps a control honest)
# --------------------------------------------------------------------------
def score_value(pred: np.ndarray, qs: QuerySet) -> Dict[str, float]:
    """Value predictions: MAE/RMSE plus decode over the family's alphabet."""
    pred = np.asarray(pred, dtype=float).reshape(len(qs), -1)
    tgt = np.asarray(qs.target, dtype=float).reshape(len(qs), -1)
    mae = float(np.mean(np.abs(pred - tgt)))
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    out = {"mae": mae, "neg_mae": -mae, "rmse": rmse}
    if qs.alphabet is not None and qs.label is not None and np.all(qs.label >= 0):
        alpha = np.asarray(qs.alphabet, dtype=float)
        dec = np.argmin(np.linalg.norm(pred[:, None, :] - alpha[None, :, :], axis=-1),
                        axis=1)
        out["decode"] = float(np.mean(dec == qs.label))
        out["chance"] = 1.0 / float(alpha.shape[0])
    else:
        # no per-query alphabet: decode against the *stored* payloads is the
        # honest legibility check (did we return SOME stored value)
        stored = qs.meta.get("stored_payloads")
        if stored is not None:
            stored = np.asarray(stored, dtype=float)
            d_stored = np.min(np.abs(pred[:, None, :] - stored[None, :, :]).sum(-1),
                              axis=1)
            d_tgt = np.abs(pred - tgt).sum(-1)
            out["beats_nearest_stored"] = float(np.mean(d_tgt < d_stored))
    return out


def score_index(pred: np.ndarray, qs: QuerySet) -> Dict[str, float]:
    """Index predictions (which item): accuracy against a 2-way chance of 0.5."""
    pred = np.asarray(pred, dtype=int).reshape(-1)
    lab = np.asarray(qs.label, dtype=int).reshape(-1)
    acc = float(np.mean(pred == lab))
    pairs = qs.meta.get("pairs")
    chance = 0.5 if pairs is not None else 1.0 / max(len(set(lab.tolist())), 1)
    return {"acc": acc, "chance": float(chance), "neg_mae": -(1.0 - acc)}


def score_coord(pred: np.ndarray, qs: QuerySet) -> Dict[str, float]:
    """Coordinate predictions: R² against the launch coordinate + spread ratio."""
    pred = np.asarray(pred, dtype=float).reshape(len(qs), -1)[:, :1].ravel()
    tgt = np.asarray(qs.target, dtype=float).reshape(len(qs), -1)[:, :1].ravel()
    ss_tot = float(np.sum((tgt - np.mean(tgt)) ** 2))
    ss_res = float(np.sum((tgt - pred) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    spread = float(np.std(pred) / max(np.std(tgt), 1e-12))
    return {"r2": r2, "spread_ratio": spread, "mae": float(np.mean(np.abs(pred - tgt))),
            "neg_mae": -float(np.mean(np.abs(pred - tgt)))}


SCORERS = {"value": score_value, "index": score_index, "coord": score_coord}

#: Primary (higher-is-better) metric per family — the dividend's own metric.
PRIMARY_METRIC = {"overload": "decode", "aggregate": "neg_mae",
                  "recency": "acc", "manifold": "r2"}


def score(qs: QuerySet, pred: np.ndarray) -> Dict[str, float]:
    """Score ``pred`` for ``qs`` with the scorer its ``kind`` selects."""
    return SCORERS[qs.kind](pred, qs)


__all__ = [
    "FAMILIES", "FAMILY_DEFAULTS", "ARMS", "GymConfig", "GymStream", "QuerySet",
    "gym_config", "byte_ratio_law", "make_gym_stream",
    "queries_overload", "queries_aggregate", "queries_recency", "queries_manifold",
    "readout_settled", "readout_tail_mean", "readout_spectator",
    "readout_occupancy", "readout_point_assign", "restrict_to_pair",
    "score", "score_value", "score_index", "score_coord", "SCORERS",
    "PRIMARY_METRIC",
]
