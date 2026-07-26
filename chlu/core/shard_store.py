"""The N-unit **sharded store** (w25): Prop L2 made flesh.

`lattice-capacity-theory` (w24) proved **Prop L2** — a masked write on disjoint atom
groups *is* N independent optimizers, because AdamW's state factorizes elementwise
over parameter blocks — and **Theorem L1**

.. code-block:: text

    K_total = min( K_addr(Q, s, sigma_q) ,  sum_r K_write(unit r) )

i.e. sharding relocates the binding constraint from the write operator onto
geometry, and can never exceed geometry. Nobody had built the N-unit store. This
module is that container, built to the theorist's five-point list, with every trap
from his §2.3 code-path audit closed **in code**, not in prose:

1. **The shard container is a** :class:`~chlu.core.lattice.CLULattice` **with
   ``edges=()`` and ``couplings=()``** — a *product store*, `V(q) = sum_r V_r(q_r)`,
   whose kinetic term is exactly separable per unit. It is **never** one wide
   relativistic CHLU: a single relativistic unit of dimension `N*d` carries ONE
   global Lorentz factor that couples every block (measured off-block
   ``d2T/dp_i dp_j = 8.4e-2`` against a diagonal of 0.81; block-0 speed falls 11x
   when its neighbours are hot), and ``mass_parameterization="*_zeromean"`` centres
   ``log_mass`` over *all* coordinates. :func:`build_sharded_store` refuses that
   configuration (§2.3 #6, #8).
2. **Localized atom initialisation** per group — the N98 fix, implemented in
   :class:`~chlu.core.memory_potentials.AtomDictionaryPotential`
   (``group_centers`` + ``local_radius``) and reached from here through
   :func:`shard_partition`'s per-shard site sets.
3. **A global address allocator** — :class:`ShardedRegistry`, N MVC-0
   :class:`~chlu.core.controller.Controller` s whose admission spacing test runs
   against the **union** of all shards' live addresses. This is the ONLY global
   object in the design, and it is a *registry, not an optimizer*.
4. **Routers R2 and R3 only.** ``R2 = argmin_r V_r(q)`` (pre-settle energy, i.e. the
   nearest-well score, evaluable WITHOUT running the dynamics) and
   ``R3 = argmin_r ||x_final - q||`` (settling displacement). ⛔ **R1 (post-settle
   energy) is deliberately NOT implemented** — N97 measured it at or below chance
   (0.031 vs chance 0.25 at N=4) because at equal well depth the relaxed energy
   records which well the query fell into, not how well it matched.
5. **Per-shard query noise.** Each shard is its own ``d``-dimensional unit, so the
   ``fixed_norm`` jitter is ``sigma_q/sqrt(d)`` with ``d`` the SHARD address
   dimension. Implementing shards as coordinate blocks of one wide unit would make
   it ``sigma_q/sqrt(N*d)`` and hand the sharded arm an unearned noise advantage
   (§2.3 #10). :func:`assert_per_shard_query_noise` states this as an executable
   check.

⚠ **The honest scope caveat, stated first (N89 discipline).** ``R2`` is, at the read
moment, a nearest-neighbour score over stored addresses — classical indexing, the
same object that laundered the w23 φ result. The defensible claim is therefore
narrow and true: *the write is additive at zero optimizer cost, and the read stays
O(1) in depth, because a classical O(N) score suffices to route.* **"Capacity
multiplies by sharding" must never be presented as a dynamical result.**

⚠ Heterogeneous per-unit ``gamma`` breaks conformal symplecticity and the Prop-4
singular-value pairing (§2.3 #11): every rollout here uses a **uniform scalar
gamma**.
"""

from typing import Callable, List, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.core.controller import Controller
from chlu.core.lattice import CLULattice
from chlu.core.memory_potentials import designed_sites, site_separation

#: The routers this module ships. ⛔ R1 (post-settle energy) is ABSENT BY DESIGN
#: (N97: at or below chance at equal well depth). Do not add it back without
#: re-opening N97.
#:
#: ``RG`` is the **registry** router — distance to the nearest address the writer
#: recorded — which the theorist's §3 lists alongside pre-settle energy as an
#: admissible O(1) statistic ("pre-settle energy, distance to the registry, or an
#: explicit tag"). It is **flagrantly classical nearest-neighbour indexing** and is
#: declared as such (N89): the CLU dynamics contribute the settle-and-read, never
#: the routing.
ROUTERS = ("R2", "R3", "RG")

#: Allocation strategies for the union of item addresses across shards.
ALLOCATIONS = ("global", "local")

#: How globally-allocated sites are dealt out to shards.
PARTITIONS = ("spread", "contiguous", "roundrobin")


# ---------------------------------------------------------------------------
# Allocation: where the items live (the ONLY global object)
# ---------------------------------------------------------------------------


def _greedy_partition(c: np.ndarray, n: int, balance_first: bool) -> List[np.ndarray]:
    """Greedy maximin assignment in allocator order, balanced to ``ceil(K/n)``.

    ``balance_first`` decides the lexicographic key: fill the emptiest shard first
    (round-robin-like, good when the site order is already space-filling) or take
    the farthest shard first (good at low ``d``, where the order matters less).
    """
    K = c.shape[0]
    cap = int(np.ceil(K / n))
    groups: List[List[int]] = [[] for _ in range(n)]
    for i in range(K):
        best, best_key = None, None
        for r in range(n):
            if len(groups[r]) >= cap:
                continue
            score = (
                np.inf
                if not groups[r]
                else float(np.min(np.linalg.norm(c[groups[r]] - c[i], axis=1)))
            )
            key = (-len(groups[r]), score, -r) if balance_first else (
                score, -len(groups[r]), -r
            )
            if best is None or key > best_key:
                best, best_key = r, key
        groups[best].append(i)
    return [np.asarray(sorted(g), dtype=int) for g in groups]


def _min_within(c: np.ndarray, groups) -> float:
    return min(
        (float(site_separation(c[g])) if len(g) > 1 else float("inf")) for g in groups
    )


def spread_partition(centers, n_shards: int, return_name: bool = False):
    """Deal globally-allocated sites to shards so each shard is *well separated*.

    Within-shard separation is the quantity sharding is supposed to buy on the WRITE
    side (the union separation — the read-side quantity — is conserved by Prop L4,
    whatever the partition). So the allocator takes the **best of four deterministic
    partitions** by ``min_r sep(shard_r)``: two greedy maximin passes (farthest-first
    and balance-first) plus contiguous and round-robin chunking of the allocator's
    farthest-point order. No single rule dominates — measured at d=2 K=16 N=4 the
    greedy is 0.790 vs 0.441 contiguous, while at d=4 K=32 N=4 contiguous is 0.834
    vs 0.710 greedy — and in the high-``d`` cells that actually decide the additivity
    question they agree to <1% (concentration of measure).

    This is the same posture ``designed_sites`` already takes ("the best packing this
    design can do", an upper envelope rather than a typical draw), applied one level
    up; the monolithic arm gets the identical farthest-point placement, so nothing
    here is a sharded-only advantage.

    Deterministic. Returns a list of ``n_shards`` int arrays of item indices (and,
    with ``return_name``, the name of the winning rule).
    """
    c = np.asarray(centers, dtype=float)
    K = c.shape[0]
    n = max(1, int(n_shards))
    idx = np.arange(K)
    bounds = [round(r * K / n) for r in range(n + 1)]
    cands = {
        "greedy_far": _greedy_partition(c, n, balance_first=False),
        "greedy_balanced": _greedy_partition(c, n, balance_first=True),
        "contiguous": [idx[bounds[r] : bounds[r + 1]] for r in range(n)],
        "roundrobin": [idx[r::n] for r in range(n)],
    }
    name = max(sorted(cands), key=lambda k: _min_within(c, cands[k]))
    return (cands[name], name) if return_name else cands[name]


def shard_partition(centers, n_shards: int, strategy: str = "spread"):
    """Partition item indices across shards (:data:`PARTITIONS`)."""
    if strategy not in PARTITIONS:
        raise ValueError(f"strategy must be one of {PARTITIONS}, got {strategy!r}")
    K = int(np.asarray(centers).shape[0])
    n = max(1, int(n_shards))
    if strategy == "spread":
        return spread_partition(centers, n)
    idx = np.arange(K)
    if strategy == "roundrobin":
        return [idx[r::n] for r in range(n)]
    bounds = [round(r * K / n) for r in range(n + 1)]
    return [idx[bounds[r] : bounds[r + 1]] for r in range(n)]


def allocate_sites(
    d: int,
    K: int,
    n_shards: int,
    R: float = 1.0,
    seed: int = 0,
    allocation: str = "global",
    partition: str = "spread",
):
    """Item sites for a sharded store, GLOBAL or LOCAL allocation.

    ``"global"`` — one farthest-point packing of the **union** (the registry knows
    every shard's addresses), then dealt to shards by ``partition``. This is the
    allocator Theorem L1's condition **W4** requires.

    ``"local"`` — each shard packs its own address set *without knowing the others*
    (a different allocator seed per shard). Within-shard separation looks great and
    the union collapses; this is the control that shows the global allocator is
    load-bearing (theorist §2.1: union read 0.917 -> 0.552 as N goes 2 -> 8).

    Returns ``(centers (K,d), groups list[np.ndarray], diagnostics dict)``.
    """
    if allocation not in ALLOCATIONS:
        raise ValueError(f"allocation must be one of {ALLOCATIONS}, got {allocation!r}")
    n = max(1, int(n_shards))
    rule = partition
    if allocation == "global":
        centers = np.asarray(designed_sites(d, K, R=R, seed=seed))
        if partition == "spread" and n > 1:
            groups, rule = spread_partition(centers, n, return_name=True)
        else:
            groups = shard_partition(centers, n, strategy=partition)
    else:
        per = [round((r + 1) * K / n) - round(r * K / n) for r in range(n)]
        blocks, groups, at = [], [], 0
        for r, m in enumerate(per):
            blocks.append(np.asarray(designed_sites(d, m, R=R, seed=seed + 1000 * r)))
            groups.append(np.arange(at, at + m, dtype=int))
            at += m
        centers = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, d))
    within = [
        float(site_separation(centers[g])) if len(g) > 1 else float("inf")
        for g in groups
    ]
    return (
        centers.astype(np.float32),
        groups,
        {
            "allocation": allocation,
            "partition": rule if allocation == "global" else "n/a",
            "union_separation": float(site_separation(centers)),
            "within_shard_separation_min": float(min(within)) if within else float("inf"),
            "within_shard_separation": within,
            "shard_sizes": [int(len(g)) for g in groups],
        },
    )


class ShardedRegistry:
    """⭐ Build item 3 — the **global address allocator**, as a registry.

    N MVC-0 :class:`~chlu.core.controller.Controller` s over N shards, wired so that
    every controller's admission spacing test (``admit_site``, refuse-and-relocate)
    runs against the **union** of all shards' live addresses. Nothing is optimized
    across shards and no gradient crosses a shard boundary: the only global object
    in the whole design is this registry of *where things were written*.

    ``global_alloc=False`` gives the control: each controller sees only its own
    addresses (per-shard allocation), which is what collapses the union separation.
    """

    def __init__(self, controllers: Sequence[Controller], global_alloc: bool = True):
        self.controllers = list(controllers)
        self.global_alloc = bool(global_alloc)
        if self.global_alloc:
            for r, ctl in enumerate(self.controllers):
                ctl.peer_addresses_fn = self._peers_of(r)

    def _peers_of(self, r: int) -> Callable[[], np.ndarray]:
        def peers():
            blocks = [
                c.stored_addresses()
                for i, c in enumerate(self.controllers)
                if i != r and c.n_live
            ]
            return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 2))

        return peers

    @property
    def n_shards(self) -> int:
        return len(self.controllers)

    def stored_addresses(self) -> np.ndarray:
        """The union registry — every live address in every shard."""
        blocks = [c.stored_addresses() for c in self.controllers if c.n_live]
        return np.concatenate(blocks, axis=0) if blocks else np.zeros((0, 2))

    def union_separation(self) -> float:
        return float(site_separation(self.stored_addresses()))

    def offer(self, shard: int, *args, **kwargs) -> dict:
        """Offer an item to shard ``shard``; admission sees the union."""
        row = dict(self.controllers[int(shard)].offer(*args, **kwargs))
        row["shard"] = int(shard)
        return row

    def offer_round_robin(self, item_id: int, *args, **kwargs) -> dict:
        return self.offer(int(item_id) % self.n_shards, item_id, *args, **kwargs)

    def stats(self) -> dict:
        out = {}
        for k in self.controllers[0].stats:
            out[k] = int(sum(c.stats[k] for c in self.controllers))
        out["n_live"] = int(sum(c.n_live for c in self.controllers))
        return out


# ---------------------------------------------------------------------------
# The container: N units, edges = (), couplings = ()
# ---------------------------------------------------------------------------


class ShardedStore(eqx.Module):
    """A product store of ``N`` CLU shards — the first real N-unit CLU store.

    ``lattice`` holds the units (``edges=()``, ``couplings=()`` — asserted at
    construction), ``shard_items`` records which global item indices each shard
    owns, and ``d`` is the per-shard ADDRESS dimension (each unit has latent
    dimension ``d+1``: address ``q[:d]`` + payload ``q[d]``).

    The read is ONE joint Verlet rollout of the lattice with the query broadcast
    into every block: because ``V`` is separable the blocks evolve independently, so
    all ``N`` shards settle **simultaneously** — the read is O(1) in rollout depth
    and O(N) only in pointwise FLOPs.
    """

    lattice: CLULattice
    shard_items: tuple = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, lattice: CLULattice, shard_items, d: int):
        if lattice.edges != () or lattice.couplings != ():
            raise ValueError(
                "a sharded store must be a PRODUCT store: CLULattice(units, "
                f"edges=(), couplings=()) — got {len(lattice.edges)} edge(s). "
                "Coupled units are not shards (theorist §2.3 #7)."
            )
        self.lattice = lattice
        self.shard_items = tuple(tuple(int(i) for i in g) for g in shard_items)
        self.d = int(d)

    @property
    def n_shards(self) -> int:
        return self.lattice.n_units

    @property
    def unit_dim(self) -> int:
        return self.d + 1

    def shard_of(self, item: int) -> int:
        for r, g in enumerate(self.shard_items):
            if item in g:
                return r
        raise KeyError(f"item {item} is in no shard")

    def item_shard_map(self, K: int) -> np.ndarray:
        """(K,) shard index per global item id."""
        out = np.full((K,), -1, dtype=int)
        for r, g in enumerate(self.shard_items):
            for i in g:
                out[i] = r
        return out

    def broadcast(self, q: jnp.ndarray) -> jnp.ndarray:
        """Replicate one ``(d+1,)`` latent into the joint ``(N*(d+1),)`` state."""
        return jnp.tile(q, self.n_shards)

    def V_per_shard(self, q: jnp.ndarray) -> jnp.ndarray:
        """``(N,)`` per-shard potential at ONE latent point — the R2 score.

        Evaluated **without running the dynamics**: this is the whole basis of the
        O(1)-in-depth read claim.
        """
        return jnp.stack([u.potential_net(q) for u in self.lattice.units])


def build_sharded_store(
    potentials: Sequence[eqx.Module],
    shard_items,
    d: int,
    kinetic_mode: str = "newtonian_learned",
    inertia=None,
) -> ShardedStore:
    """Wrap ``N`` per-shard potentials into a :class:`ShardedStore`.

    ⛔ ``kinetic_mode="relativistic"`` is **refused**. A relativistic kinetic term is
    non-separable *within* a unit, which is fine, but the reason shards must be
    lattice units at all is that a single relativistic unit of dimension ``N*d``
    couples every block through one global Lorentz factor (§2.3 #6). Allowing a
    relativistic shard here would invite exactly the wide-unit implementation the
    build list forbids, and its per-unit speed limit would additionally make the two
    arms' read dynamics incomparable. Newtonian modes only; stated as a hard error
    rather than a docstring warning.
    """
    from chlu.experiments.goldstone_harness import clu_with_potential

    if kinetic_mode == "relativistic":
        raise ValueError(
            "relativistic shards are refused: one global Lorentz factor couples "
            "every block in a wide unit (off-block d2T/dp_i dp_j = 8.4e-2, block "
            "speed falls 11x with hot neighbours; theorist §2.3 #6). Use "
            "'newtonian_identity' or 'newtonian_learned' and keep shards as "
            "separate CLULattice units."
        )
    dim = int(d) + 1
    inert = jnp.ones(dim) if inertia is None else inertia
    units = [
        clu_with_potential(V, dim=dim, kinetic_mode=kinetic_mode, inertia=inert)
        for V in potentials
    ]
    lattice = CLULattice(units, edges=(), couplings=())
    return ShardedStore(lattice, shard_items, d=int(d))


def assert_per_shard_query_noise(query_sigma: float, d_shard: int, n_shards: int,
                                 scale: float, tol: float = 1e-6) -> None:
    """Executable form of the §2.3 #10 fairness trap.

    The ``fixed_norm`` query jitter is ``sigma/sqrt(d)`` **per axis with ``d`` the
    SHARD address dimension**. Raises if the caller has (accidentally) scaled with
    the joint dimension ``N*d``, which would silently shrink the per-shard query
    noise as ``N`` grows and hand the sharded arm an unearned advantage.
    """
    want = float(query_sigma) / np.sqrt(float(d_shard))
    if abs(float(scale) - want) > tol:
        wide = float(query_sigma) / np.sqrt(float(d_shard) * max(n_shards, 1))
        raise AssertionError(
            f"per-shard query noise must be sigma/sqrt(d_shard) = {want:.6f}, got "
            f"{float(scale):.6f}"
            + (
                "  <-- this is sigma/sqrt(N*d): the fairness trap (theorist §2.3 #10)"
                if abs(float(scale) - wide) <= tol
                else ""
            )
        )


# ---------------------------------------------------------------------------
# Routing (R2 / R3 only) + the abstention deadband
# ---------------------------------------------------------------------------


def route_from_scores(scores: np.ndarray, deadband: float = 0.0):
    """Top-1 route + a **top-2 abstention deadband**.

    ``scores`` is ``(n_queries, N)``, lower = better. Returns
    ``(route (n,), margin (n,), abstain (n,) bool)`` where ``margin`` is the gap
    between the best and second-best shard score. A product-store routing miss is
    **total** — the wrong shard's payload comes back with full confidence, value
    accuracy tracks route accuracy 1:1 — so an abstention rule is not optional in a
    deployed design (theorist §2.2). It is nonetheless OFF by default here
    (``deadband=0``) so the headline numbers stay per-offered and comparable to the
    monolithic arm (N91 discipline: no abstention credit in a headline).
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"scores must be (n, N), got {s.shape}")
    order = np.argsort(s, axis=1)
    route = order[:, 0]
    if s.shape[1] < 2:
        margin = np.full((s.shape[0],), np.inf)
    else:
        best = s[np.arange(s.shape[0]), order[:, 0]]
        second = s[np.arange(s.shape[0]), order[:, 1]]
        margin = second - best
    abstain = margin < float(deadband)
    return route, margin, abstain


@eqx.filter_jit
def _r2_apply(store: "ShardedStore", Q0: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(store.V_per_shard)(Q0)


def r2_scores(store: ShardedStore, Q0: jnp.ndarray) -> np.ndarray:
    """``R2 = argmin_r V_r(q)`` — pre-settle energy, **no dynamics run**.

    ``Q0`` is ``(n, d+1)`` (payload channel launched at 0, as the read does).
    Returns ``(n, N)`` scores. This is the routing statistic the O(1)-in-depth claim
    rests on: it is a classical nearest-well score over the stored addresses (N89).

    The jit lives at module level (not in a per-call closure) so repeated calls at
    the same shape REUSE the compiled kernel — otherwise a routing "measurement"
    times the XLA compiler, not the router.
    """
    return np.asarray(_r2_apply(store, jnp.asarray(Q0)))


def r3_scores(addr_x: np.ndarray, Q0: np.ndarray, d: int) -> np.ndarray:
    """``R3 = argmin_r ||x_final_r - q||`` — the settling displacement.

    The physically native score: the work the dynamics had to do to accept the
    query. It needs the settle, but the settle is run anyway for the read (one joint
    rollout for all shards), so R3 costs nothing beyond it. ``addr_x`` is
    ``(n, N, d)``, ``Q0`` is ``(n, d+1)``.
    """
    q = np.asarray(Q0)[:, None, :d]
    return np.linalg.norm(np.asarray(addr_x) - q, axis=-1)


def rg_scores(store: ShardedStore, Q0, centers) -> np.ndarray:
    """``RG = argmin_r min_{c in shard r} ||q_addr - c||`` — the REGISTRY router.

    Distance from the raw query to the nearest address **the writer recorded**
    ("the writer records where it wrote" — the MVC-0 placement rule, C1). No
    dynamics, no learned parameters, ``O(K_total * d)`` flops.

    ⚠ **This is classical nearest-neighbour indexing and is declared as such**
    (N89). It is included because the theorist's own O(1) condition names it
    ("pre-settle energy, distance to the registry, or an explicit tag"), and
    because it separates two very different failures: a *store* that cannot hold
    its items, versus a *score* that cannot tell which unit holds them. Any number
    it produces belongs to the classical index, not to the CLU dynamics.
    """
    q = np.asarray(Q0)[:, : store.d]
    c = np.asarray(centers)
    out = np.empty((q.shape[0], store.n_shards), dtype=float)
    for r, g in enumerate(store.shard_items):
        gi = np.asarray(g, dtype=int)
        d2 = ((q[:, None, :] - c[gi][None, :, :]) ** 2).sum(-1)
        out[:, r] = np.sqrt(d2.min(axis=1))
    return out


def router_scores(
    router: str, store: ShardedStore, Q0, addr_x=None, centers=None
) -> np.ndarray:
    """Dispatch a router by name. ⛔ ``"R1"`` raises (N97)."""
    if router == "R1":
        raise ValueError(
            "R1 (post-settle energy) is not implemented and must not be: N97 "
            "measured it AT OR BELOW CHANCE (0.031 vs chance 0.25 at N=4) — at "
            "equal well depth the relaxed energy records which well the query fell "
            "into, not how well it matched. Use R2 (pre-settle) or R3 (displacement)."
        )
    if router not in ROUTERS:
        raise ValueError(f"router must be one of {ROUTERS}, got {router!r}")
    if router == "R2":
        return r2_scores(store, Q0)
    if router == "RG":
        if centers is None:
            raise ValueError("RG needs the registry (centers)")
        return rg_scores(store, Q0, centers)
    if addr_x is None:
        raise ValueError("R3 needs the settled addresses (addr_x)")
    return r3_scores(addr_x, np.asarray(Q0), store.d)


# ---------------------------------------------------------------------------
# The read: ONE joint rollout settles every shard
# ---------------------------------------------------------------------------


def sharded_two_phase(
    store: ShardedStore,
    Q0: jnp.ndarray,
    P0: jnp.ndarray,
    cfg,
    chunk: Optional[int] = None,
):
    """query -> [gamma_address relax] -> address -> [gamma_read rollout] -> read.

    The query is broadcast into every shard block and ONE joint ``CLULattice``
    rollout is run: ``V`` is separable, so the blocks evolve independently and all
    ``N`` shards settle simultaneously. Returns
    ``(addr_x (n, N, d), payload_tail (n, N, n_subsample))``.

    ``cfg`` supplies ``dt``, ``gamma_address``, ``address_steps``, ``gamma_read``,
    ``read_steps``, ``tail_frac``, ``n_subsample``, ``rollout_chunk`` — read from
    ``experiment_designed_mechanism`` so the sharded read is the w23 read, verbatim.
    ``gamma`` is a uniform scalar (§2.3 #11: heterogeneous per-unit gamma breaks
    conformal symplecticity).
    """
    N = store.n_shards
    steps = int(cfg.read_steps)
    start = int((1.0 - cfg.tail_frac) * steps)
    tail_idx = jnp.asarray(np.linspace(start, steps - 1, cfg.n_subsample).astype(int))

    n = int(Q0.shape[0])
    base = int(cfg.rollout_chunk if chunk is None else chunk)
    # the joint state is N x wider than a monolithic one: shrink the vmap chunk so
    # peak memory is comparable (the trajectory is (steps, 2*N*dim) per query).
    c = max(8, min(n, base // max(1, N)))
    xs, feats = [], []
    for i in range(0, n, c):
        q, p = jnp.asarray(Q0[i : i + c]), jnp.asarray(P0[i : i + c])
        pad = c - q.shape[0]
        if pad > 0:
            q = jnp.concatenate([q, jnp.zeros((pad,) + q.shape[1:])], axis=0)
            p = jnp.concatenate([p, jnp.zeros((pad,) + p.shape[1:])], axis=0)
        x, f = _two_phase_chunk(
            store, q, p, tail_idx,
            int(cfg.address_steps), steps,
            float(cfg.dt), float(cfg.gamma_address), float(cfg.gamma_read),
        )
        x, f = np.asarray(x), np.asarray(f)
        if pad > 0:
            x, f = x[: c - pad], f[: c - pad]
        xs.append(x)
        feats.append(f)
    return np.concatenate(xs, axis=0), np.concatenate(feats, axis=0)


@eqx.filter_jit
def _two_phase_chunk(store, Q, P, tail_idx, address_steps, read_steps,
                     dt, gamma_address, gamma_read):
    """One compiled two-phase chunk. Module level so the kernel is REUSED across
    chunks and cells (a closure would recompile on every call)."""
    lat = store.lattice
    N, D = store.n_shards, store.lattice.dim
    d, dim = store.d, store.unit_dim

    def one(q, p):
        qj, pj = jnp.tile(q, N), jnp.tile(p, N)
        tr1 = lat(qj, pj, address_steps, dt, gamma_address)
        aq, ap = tr1[-1, :D], tr1[-1, D:]
        tr2 = lat(aq, ap, read_steps, dt, gamma_read)
        q_fin = tr2[-1, :D].reshape(N, dim)
        pay = tr2[:, :D].reshape(read_steps, N, dim)[tail_idx][:, :, d]  # (n_sub, N)
        return q_fin[:, :d], pay.T  # (N, d), (N, n_sub)

    return jax.vmap(one)(Q, P)


__all__ = [
    "ALLOCATIONS",
    "PARTITIONS",
    "ROUTERS",
    "ShardedRegistry",
    "ShardedStore",
    "allocate_sites",
    "assert_per_shard_query_noise",
    "build_sharded_store",
    "r2_scores",
    "r3_scores",
    "route_from_scores",
    "router_scores",
    "shard_partition",
    "sharded_two_phase",
    "spread_partition",
]
