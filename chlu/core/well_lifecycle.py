"""⭐ **C2W8 stage 1** — the well census: what a live well *is*, and which live
wells a lifecycle verb could act on.

This module is the **kill-condition instrument, built before the verbs it can
kill** (`PREREG-C2W8.md` §5 K1, standing doctrine §A12). It measures, on a real
over-dug :class:`~chlu.core.clu_system.CluSystem`:

* **well states** (§3.1) — fitted depth at the item's own site (**raw AND
  decay-netted**, build requirement B1), the item's own-vs-foreign atom-sum
  decomposition at that site, ``lambda_min`` at the relaxed site, and the
  **measured** capture radius;
* **``theta_att``**, the capture floor — *measured* by SC-6's direction
  bisection on this rig, **never a guessed constant**, because mechanic 1 says an
  eroded well has already ceased to exist as an attractor and must be counted
  separately from a live-but-unread one;
* **``P``** (prunable) and **``M``** (mergeable) per §3.3, and the mechanical
  **stage-2 unlock** rule (§5 K1: ``UNLOCK iff P >= 0.05 or M >= 0.05`` on the
  seed mean; ``KILL iff both < 0.05 on every seed``).

Two design rules are load-bearing and are asserted in
``tests/test_well_lifecycle.py``:

⛔ **Depth is not usage.** ``P``'s usage leg is ``read_hits`` from
:mod:`chlu.experiments.usage_telemetry` (item-id-keyed), never depth. Depth
enters only through ``is_attractor`` — i.e. through *"does this well still
exist"*, never through *"is this well useful"* (mechanic 2, §A23.5 ACTIVE).

⛔ **The two populations are reported separately.** ``{eroded, not attractor}``
is a bookkeeping problem; only ``{live attractor, never read}`` is prunable. A
census that merges them would license a prune verb on wells that are already
gone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: K1's registered threshold (prereg §5). Not a tunable: it is the number the
#: unlock verdict was registered against before any cell ran.
UNLOCK_THRESHOLD = 0.05


@dataclass(frozen=True)
class WellState:
    """One live well, as the census sees it (prereg §3.1)."""

    item_id: int
    slot: int
    depth_raw: float  # fitted D at the item's own site, on the learned V
    depth_netted: float  # depth_raw / designed decay applied since the write (B1)
    decay_factor: float  # the designed factor itself (1.0 => nothing to net)
    s_eff: float
    own_atom_depth: float  # the item's OWN atoms' contribution at the site
    foreign_atom_depth: float  # every other atom's contribution at the site
    lambda_min: float  # at the RELAXED site (never at the recorded site)
    capture_radius: float  # SC-6 bisection; NaN if not measured
    site_drift: float  # |relaxed site - recorded site| (slot != well)
    is_attractor: bool
    read_hits: int
    protected: bool  # leak == 0 / permanent cohort — excluded from P
    born_t: int

    @property
    def eroded(self) -> bool:
        """Mechanic 1: below the measured floor => already gone, not prunable."""
        return not self.is_attractor

    @property
    def prunable(self) -> bool:
        return bool(self.is_attractor and self.read_hits == 0 and not self.protected)


# --------------------------------------------------------------------------
# B1 — the designed-decay netting
# --------------------------------------------------------------------------
def designed_decay_factors(controller) -> Dict[int, float]:
    """Cumulative **designed** depth-decay applied to each item since its write.

    Replayed from the controller's own verb log (every ``decay`` record carries
    ``factors[item_id]``), so it is exact rather than reconstructed from a rate:
    the product over the ticks an item was live IS the decay the landscape got.

    ⚠ **B1 (§A27.1).** A depth curve quoted raw silently credits the arm for the
    decay it was *designed* to suffer. Netting moved C2W6's E1 seed 0 by −34 %,
    and this wave's P2 leg is depth *restoration* — exactly the quantity netting
    corrects. Every curve in this wave is reported raw AND netted.
    """
    out: Dict[int, float] = {}
    for rec in getattr(controller, "log", []):
        if getattr(rec, "verb", "") != "decay" or not getattr(rec, "applied", False):
            continue
        for iid, f in dict(rec.detail.get("factors", {})).items():
            f = float(f)
            if f <= 0.0:
                continue
            out[int(iid)] = out.get(int(iid), 1.0) * f
    return out


def own_foreign_site_depth(store, slot: int, site) -> Tuple[float, float]:
    """``(own, foreign)`` atom-sum depth at ``site`` — the interference split.

    ``own`` is the contribution of the atoms the slot owns (the only rows a
    masked write may move, hence the only rows the designed decay scales);
    ``foreign`` is every other atom's contribution at the same point. The pair is
    what :func:`chlu.experiments.exp_anti_erosion._interference_audit` consumes,
    which is how this wave reuses C2W6's residual-vs-decay-law instrument
    **without editing that file**.
    """
    atoms = store.atoms
    A = np.asarray(atoms.amp, dtype=float) ** 2
    s = np.exp(np.asarray(atoms.log_width, dtype=float))
    c = np.asarray(atoms.centers, dtype=float)
    z = np.asarray(site, dtype=float).reshape(1, -1)[:, : c.shape[1]]
    d2 = np.sum((c - z) ** 2, axis=-1)
    w = A * np.exp(-d2 / (2.0 * s**2 + 1e-12))
    m = np.asarray(store.group_rows(int(slot)), dtype=bool)
    return float(np.sum(w[m])), float(np.sum(w[~m]))


# --------------------------------------------------------------------------
# theta_att — the MEASURED capture floor (§3.1; never a guessed constant)
# --------------------------------------------------------------------------
def capture_radii(system, sites, *, n_dirs: int = 16, steps: int = 8,
                  tol: Optional[float] = None, seed: int = 0) -> np.ndarray:
    """SC-6's direction bisection at every given site -> ``(n_sites,)`` radii.

    ``lambda_min > 0`` certifies a *local minimum*, not a nonempty basin (the
    theorist measured a basin of 0.000 at ``lambda_min = +0.910``), so the floor
    is measured here rather than inferred.
    """
    from chlu.core.soft_certificate import capture_radius

    sites = np.atleast_2d(np.asarray(sites, dtype=float))
    tol = float(system.cfg.query_sigma if tol is None else tol)
    r_hi = float(system.cfg.ball_radius)
    out = np.full((sites.shape[0],), np.nan, dtype=float)
    for i, z in enumerate(sites):
        r = capture_radius(system._relax_points, z, n_dirs=int(n_dirs),
                           r_hi=r_hi, steps=int(steps), tol=tol,
                           seed=int(seed) + i)
        out[i] = float(r["capture_radius"])
    return out


def measure_theta_att(depths, capture, sigma_q: float) -> Dict[str, Any]:
    """The **measured** capture floor ``theta_att`` on this rig.

    Registered rule (ERRATA-C2W8, filed before the census ran):

    > A well *captures* iff its measured capture radius is at least the
    > operating query jitter ``sigma_q`` — the read's own queries are drawn at
    > that scale, so a basin narrower than the jitter cannot be addressed.
    > ``theta_att`` := the **largest fitted depth among the wells that did NOT
    > capture** (0.0 when every well captured). It is therefore a floor at which
    > capture was *observed to fail on this rig*, not a constant.

    Returned alongside the ingredients, so the number is auditable.
    """
    d = np.asarray(depths, dtype=float)
    r = np.asarray(capture, dtype=float)
    ok = np.isfinite(r) & (r >= float(sigma_q))
    bad = np.isfinite(r) & ~ok
    theta = float(np.max(d[bad])) if np.any(bad) else 0.0
    return {
        "theta_att": theta,
        "rule": ("max fitted depth among wells whose measured capture radius < "
                 "sigma_q; 0.0 if every well captured (ERRATA-C2W8 §1)"),
        "sigma_q": float(sigma_q),
        "n_capturing": int(np.sum(ok)),
        "n_non_capturing": int(np.sum(bad)),
        "n_unmeasured": int(np.sum(~np.isfinite(r))),
        "capture_radius_median": (float(np.median(r[np.isfinite(r)]))
                                  if np.any(np.isfinite(r)) else float("nan")),
        "captures": ok.tolist(),
    }


# --------------------------------------------------------------------------
# the census
# --------------------------------------------------------------------------
def well_states(system, telemetry=None, *, n_dirs: int = 16,
                bisect_steps: int = 8, measure_capture: bool = True,
                seed: int = 0) -> Tuple[List[WellState], Dict[str, Any]]:
    """Every live well's state, plus the ``theta_att`` block it was scored on."""
    ids, centers, pays = system.codebook()
    if len(ids) == 0:
        return [], measure_theta_att([], [], system.cfg.query_sigma)
    addr_dim, pay_dim = system.store.addr_dim, system.store.payload_dim
    sites = np.zeros((len(ids), system.store.dim), dtype=np.float32)
    sites[:, :addr_dim] = centers
    sites[:, addr_dim: addr_dim + pay_dim] = pays

    depths, widths = system.well_fits()
    relaxed = system._relaxed_sites()
    lam = system._lambda_min_per_point(relaxed)
    drift = np.linalg.norm(np.asarray(relaxed) - sites, axis=-1)
    cap = (capture_radii(system, sites, n_dirs=n_dirs, steps=bisect_steps,
                         seed=seed)
           if measure_capture else np.full((len(ids),), np.nan))
    theta = measure_theta_att(depths, cap, float(system.cfg.query_sigma))
    factors = designed_decay_factors(system.controller)

    recs = {int(r.item_id): r for r in system.controller.allocator.records.values()}
    states: List[WellState] = []
    for i, iid in enumerate(ids):
        iid = int(iid)
        rec = recs.get(iid)
        slot = int(rec.slot) if rec is not None else -1
        own, foreign = own_foreign_site_depth(system.store, slot, sites[i])
        f = float(factors.get(iid, 1.0))
        captures = bool(theta["captures"][i]) if measure_capture else True
        is_att = bool(float(lam[i]) > 0.0
                      and float(depths[i]) > float(theta["theta_att"])
                      and captures)
        states.append(WellState(
            item_id=iid, slot=slot,
            depth_raw=float(depths[i]),
            depth_netted=float(depths[i] / f) if f > 0 else float("nan"),
            decay_factor=f, s_eff=float(widths[i]),
            own_atom_depth=own, foreign_atom_depth=foreign,
            lambda_min=float(lam[i]),
            capture_radius=float(cap[i]), site_drift=float(drift[i]),
            is_attractor=is_att,
            read_hits=(int(telemetry.hits(iid)) if telemetry is not None else 0),
            protected=bool(rec.permanent or float(rec.leak) == 0.0) if rec else False,
            born_t=int(rec.born) if rec is not None else -1,
        ))
    return states, theta


def mergeable_pairs(system, states: Sequence[WellState], *,
                    payload_thresh: Optional[float] = None,
                    r_cert: Optional[float] = None) -> Tuple[List[dict], Dict[str, float]]:
    """Live-well PAIRS admissible for merge on **mechanical** criteria (§3.3).

    A pair is admissible iff **both** hold:

    * ``|a_i - a_j| <= payload_thresh`` — registered as ``cfg.payload_tol``
      (ERRATA-C2W8 §2): two payloads closer than the read's own tolerance are
      indistinguishable at read-out, so the pair carries one value, not two;
    * ``|c_i - c_j| <= R_cert = 2 s_max + kappa' sigma_q`` — the SC-1/SC-2
      certificate radius, i.e. the pair is exactly the near-duplicate that
      over-digging is *supposed* to produce and that the certificate wanted
      separated.

    ⛔ Depth does not enter. A spurious shallow well is **trash**, not merge
    material (§A20.3(d)) — that separation is stage 2's, and this function
    reports the raw pair population it will act on.
    """
    from chlu.core.soft_certificate import cert_radius

    ids, centers, pays = system.codebook()
    thr = float(system.cfg.payload_tol if payload_thresh is None else payload_thresh)
    r = float(cert_radius(system._s_max(), float(system.cfg.query_sigma),
                          float(system.cfg.d_safe_kappa_prime))
              if r_cert is None else r_cert)
    pos = {int(s.item_id): k for k, s in enumerate(states)}
    out: List[dict] = []
    n = len(ids)
    for a in range(n):
        for b in range(a + 1, n):
            dc = float(np.linalg.norm(centers[a] - centers[b]))
            dp = float(np.linalg.norm(np.asarray(pays[a]) - np.asarray(pays[b])))
            if dp <= thr and dc <= r:
                out.append({
                    "item_i": int(ids[a]), "item_j": int(ids[b]),
                    "center_sep": dc, "payload_dist": dp,
                    "depth_i": states[pos[int(ids[a])]].depth_raw if int(ids[a]) in pos else float("nan"),
                    "depth_j": states[pos[int(ids[b])]].depth_raw if int(ids[b]) in pos else float("nan"),
                })
    return out, {"payload_thresh": thr, "r_cert": r,
                 "n_pairs": int(n * (n - 1) / 2)}


def census(system, telemetry=None, *, well_budget: Optional[int] = None,
           n_admitted: Optional[int] = None, seed: int = 0,
           n_dirs: int = 16, bisect_steps: int = 8,
           measure_capture: bool = True) -> Dict[str, Any]:
    """The K1 instrument: ``P``, ``M``, both well populations, ``theta_att``.

    ``P`` = fraction of **live wells** that are live attractors, never read, and
    not protected. ``M`` = fraction of **live-well pairs** admissible for merge.
    Denominators are stated in the result (``n_live`` and ``n_pairs``) because
    the designed negatives are expressed against them.
    """
    states, theta = well_states(system, telemetry, n_dirs=n_dirs,
                                bisect_steps=bisect_steps,
                                measure_capture=measure_capture, seed=seed)
    pairs, pair_meta = mergeable_pairs(system, states)
    n_live = len(states)
    n_pairs = int(pair_meta["n_pairs"])
    prunable = [s for s in states if s.prunable]
    eroded = [s for s in states if s.eroded]
    live_unread = [s for s in states if s.is_attractor and s.read_hits == 0]
    protected = [s for s in states if s.protected]
    P = float(len(prunable) / n_live) if n_live else 0.0
    M = float(len(pairs) / n_pairs) if n_pairs else 0.0
    wb = int(well_budget if well_budget is not None else system.cfg.capacity)
    n_adm = int(n_admitted if n_admitted is not None else n_live)
    depths = np.asarray([s.depth_raw for s in states], dtype=float)
    netted = np.asarray([s.depth_netted for s in states], dtype=float)
    return {
        "P": P,
        "M": M,
        "n_live": n_live,
        "n_pairs": n_pairs,
        "n_prunable": len(prunable),
        "n_mergeable_pairs": len(pairs),
        "overdig": float(n_adm / wb) if wb else float("nan"),
        "well_budget": wb,
        "n_admitted": n_adm,
        "theta_att_block": theta,
        "pair_criteria": pair_meta,
        # ⭐ the two populations, SEPARATE (mechanic 1)
        "population_eroded_not_attractor": [s.item_id for s in eroded],
        "population_live_attractor_never_read": [s.item_id for s in live_unread],
        "population_protected": [s.item_id for s in protected],
        "depth_raw_median": float(np.median(depths)) if depths.size else float("nan"),
        "depth_netted_median": float(np.median(netted)) if netted.size else float("nan"),
        "depth_raw_geomean": _geomean(depths),
        "depth_netted_geomean": _geomean(netted),
        "mergeable_pairs": pairs,
        "wells": [asdict(s) for s in states],
    }


def _geomean(x) -> float:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.mean(np.log(v)))) if v.size else float("nan")


def unlock_verdict(P_by_seed: Sequence[float], M_by_seed: Sequence[float],
                   threshold: float = UNLOCK_THRESHOLD) -> Dict[str, Any]:
    """K1, computed **mechanically** (prereg §5) — no judgement call.

    ``UNLOCK iff mean(P) >= thr or mean(M) >= thr``;
    ``KILL iff P < thr and M < thr on EVERY seed``.
    """
    P = np.asarray(list(P_by_seed), dtype=float)
    M = np.asarray(list(M_by_seed), dtype=float)
    thr = float(threshold)
    unlock = bool(P.size and (float(np.mean(P)) >= thr or float(np.mean(M)) >= thr))
    kill = bool(P.size and np.all(P < thr) and np.all(M < thr))
    return {
        "stage2_unlock": unlock,
        "kill": kill,
        "threshold": thr,
        "P_mean": float(np.mean(P)) if P.size else float("nan"),
        "M_mean": float(np.mean(M)) if M.size else float("nan"),
        "P_by_seed": [float(x) for x in P],
        "M_by_seed": [float(x) for x in M],
        "rule": ("UNLOCK iff mean(P) >= 0.05 or mean(M) >= 0.05; "
                 "KILL iff both < 0.05 on every seed (PREREG-C2W8 §5 K1)"),
    }


# --------------------------------------------------------------------------
# planting — the designed negatives' construction kit
# --------------------------------------------------------------------------
def plant_item(system, item_id: int, address, payload: float, *,
               depth: float, width: Optional[float] = None,
               leak: float = 0.0, permanent: bool = False,
               jitter: float = 0.0, seed: int = 0) -> int:
    """Hand-build one well of known depth at a known site, **without a write**.

    The designed negatives (§5 K1) need a store whose ground truth is known by
    construction: *"a census instrument that cannot see a planted population
    cannot license a kill."* Planting sets the slot's own atoms directly (total
    ``sum A_j = depth`` at the site), which is the same object the writer digs,
    so the census reads a planted well through exactly the shipped estimator.

    Returns the slot the item was admitted into.
    """
    import equinox as eqx
    import jax.numpy as jnp

    addr_dim, pay_dim = system.store.addr_dim, system.store.payload_dim
    address = np.asarray(address, dtype=float).reshape(-1)[:addr_dim]
    res = system.controller.admit(int(item_id), address, float(payload),
                                  permanent=bool(permanent), leak=float(leak))
    if not res.applied:
        raise RuntimeError(f"plant_item: admission refused for {item_id}: {res.reason}")
    system._payloads[int(item_id)] = np.atleast_1d(np.asarray(payload, dtype=float))
    system._born[int(item_id)] = system._t
    slot = system._slot_of(int(item_id))

    rows = np.asarray(system.store.group_rows(slot), dtype=bool)
    n = int(rows.sum())
    if n == 0:
        raise RuntimeError(f"plant_item: slot {slot} owns no atom rows")
    z = np.zeros((system.store.dim,), dtype=float)
    z[:addr_dim] = address
    z[addr_dim: addr_dim + pay_dim] = float(payload)
    rng = np.random.default_rng(int(seed) + int(item_id))
    c = np.repeat(z[None, :], n, axis=0)
    if jitter > 0:
        c = c + rng.normal(size=c.shape) * float(jitter)
    w = float(system.cfg.atom_width if width is None else width)
    idx = np.nonzero(rows)[0]
    atoms = system.store.V.learned
    new_c = atoms.centers.at[jnp.asarray(idx)].set(jnp.asarray(c, dtype=atoms.centers.dtype))
    new_a = atoms.amp.at[jnp.asarray(idx)].set(
        jnp.asarray(np.sqrt(max(float(depth), 0.0) / n), dtype=atoms.amp.dtype))
    new_w = atoms.log_width.at[jnp.asarray(idx)].set(
        jnp.asarray(np.log(w), dtype=atoms.log_width.dtype))
    V = eqx.tree_at(
        lambda t: [t.learned.centers, t.learned.amp, t.learned.log_width],
        system.store.V, replace=[new_c, new_a, new_w])
    system.store = eqx.tree_at(lambda s: s.V, system.store, V)
    return int(slot)


def flatten_unused_groups(system) -> None:
    """Drive every **unallocated** slot's atoms to zero depth.

    A planted store still carries the init scatter (``amp = sqrt(1e-4)`` per
    atom) in its free groups; summed over thousands of atoms that is a real,
    non-designed background. Flattening it makes a planted census exactly the
    population that was planted — the designed negatives are then statements
    about the instrument, not about the init distribution.
    """
    import equinox as eqx
    import jax.numpy as jnp

    used = {int(r.slot) for r in system.controller.allocator.records.values()}
    keep = np.zeros((system.store.V.learned.n_atoms,), dtype=bool)
    for s in used:
        keep |= np.asarray(system.store.group_rows(int(s)), dtype=bool)
    # ⚠ `np.array(..., copy=True)`, never `np.asarray`: under `jax_enable_x64` a
    # float64 JAX array converts zero-copy and `asarray` hands back a READ-ONLY
    # view, so the in-place write below raises — but only when an x64-enabling
    # module ran first (§7.23's ordering hazard, which is how this was found: the
    # test passed alone and failed in the full suite).
    amp = np.array(system.store.V.learned.amp, dtype=float, copy=True)
    amp[~keep] = 0.0
    V = eqx.tree_at(lambda t: t.learned.amp, system.store.V,
                    jnp.asarray(amp, dtype=system.store.V.learned.amp.dtype))
    system.store = eqx.tree_at(lambda s: s.V, system.store, V)


__all__ = [
    "UNLOCK_THRESHOLD", "WellState", "designed_decay_factors",
    "own_foreign_site_depth", "capture_radii", "measure_theta_att",
    "well_states", "mergeable_pairs", "census", "unlock_verdict",
    "plant_item", "flatten_unused_groups",
]
