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

---

⭐⭐ **C2W8 PASS 3 — this module also carries `G-ADDR` (:func:`gate_addr`), the
ADDRESSABILITY leg** (charter §A30.1, ``PREREG-C2W8-PASS3.md`` §2), plus Ruling
3's compliance counterfactual (:func:`displaced_write_counterfactual`).

*Why.* Three times in one wave this programme shipped **a gate that cannot fail
on the thing that matters**: pass 1's vacuous ``M`` (geometric leg ~10x the
address resolution, refused nothing), monitor #3's admission gate (refusal rate
0.000, still open), and the pass-2 capture gate, which passed **arm B on 3/3
seeds with 16/16 items never read**. G-CAP/G-DEC/G-DRIFT all measure
retrievability *at the sites*; **nothing measured whether a query reaches its
site.** G-ADDR does, with ground truth by construction, designed negatives AND a
designed positive, all pytest-asserted in ``tests/test_gate_addr.py``.

Three instrument findings this module now documents rather than carrying
silently: the SC-6 capture instrument's ``tol/expansion_rate`` floor and its
confinement-minimum false positive (:func:`capture_radii`), ``theta_att``'s
arm-dependent dynamic range (:func:`measure_theta_att`), and the kernel-mismatch
repair of :func:`own_foreign_site_depth`.
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


def _atom_profile_np(d2, s, kernel: str = "gaussian", cutoff: float = 2.5):
    """float64 mirror of :func:`chlu.core.memory_potentials.atom_profile`.

    The shipped profile is written in ``jnp`` and therefore evaluates in
    **float32**; this census estimator is a float64 diagnostic and must not
    inherit a 1e-7 relative wobble into a banked number. The two are pinned
    against each other for **all three kernels** in
    ``tests/test_gate_addr.py::test_numpy_atom_profile_mirrors_the_shipped_one``.
    ⛔ If a kernel is added to ``ATOM_KERNELS``, that test fails until it is
    mirrored here — which is the point.
    """
    from chlu.core.memory_potentials import ATOM_KERNELS

    if kernel not in ATOM_KERNELS:
        raise ValueError(f"atom kernel must be one of {ATOM_KERNELS}, got {kernel!r}")
    d2 = np.asarray(d2, dtype=float)
    s = np.asarray(s, dtype=float)
    if kernel == "gaussian":
        return np.exp(-d2 / (2.0 * s**2 + 1e-9))
    R = np.maximum(float(cutoff) * s, 1e-12)
    if kernel == "wendland":
        t = np.clip(np.sqrt(np.maximum(d2, 0.0)) / R, 0.0, 1.0)
        return (1.0 - t) ** 4 * (1.0 + 4.0 * t)
    g = np.exp(-d2 / (2.0 * s**2 + 1e-9))
    gR = np.exp(-(R**2) / (2.0 * s**2 + 1e-9))
    return np.where(d2 <= R**2, (g - gR) / np.maximum(1.0 - gR, 1e-12), 0.0)


def own_foreign_site_depth(store, slot: int, site) -> Tuple[float, float]:
    """``(own, foreign)`` atom-sum depth at ``site`` — the interference split.

    ``own`` is the contribution of the atoms the slot owns (the only rows a
    masked write may move, hence the only rows the designed decay scales);
    ``foreign`` is every other atom's contribution at the same point. The pair is
    what :func:`chlu.experiments.exp_anti_erosion._interference_audit` consumes,
    which is how this wave reuses C2W6's residual-vs-decay-law instrument
    **without editing that file**.

    ⭐ **C2W8 pass-3 fix (pass-2 reconciliation item 1).** This estimator used to
    hard-code the **Gaussian** atom profile ``exp(-d2 / (2 s^2 + 1e-12))``. Under
    any compact-atom arm (``atom_kernel = "wendland" | "truncated_gaussian"``,
    C2W8 pass 2 arm A) that is **kernel-mismatched**: it credits every atom with a
    tail the landscape does not have, so **both legs are over-read**, the foreign
    leg worst (foreign atoms are the far ones). Arm A's own/foreign was reported
    *through* the mismatched form, labelled. It now reads the store's **own**
    profile (:func:`chlu.core.memory_potentials.atom_profile`) with the store's
    own ``kernel``/``kernel_cutoff`` and, if present, its ``axis_width_scale`` —
    i.e. exactly the expression ``AtomDictionaryStore.__call__`` sums.

    ⛔ **Continuity with every banked number is a designed invariant.** Under
    ``kernel = "gaussian"`` the only difference from the pre-pass-3 form is the
    epsilon in the denominator (``1e-9``, the store's own, vs the ``1e-12`` this
    function used), a **3e-8 relative** change — six orders below the least
    significant digit ever quoted (4 dp). It is pinned numerically in
    ``tests/test_gate_addr.py::test_own_foreign_matches_the_legacy_gaussian_form``.
    Every pass-1/pass-2 own/foreign reading was taken on a Gaussian store or was
    explicitly labelled kernel-mismatched, so no banked number moves silently.
    """
    atoms = store.atoms
    A = np.asarray(atoms.amp, dtype=float) ** 2
    s = np.asarray(np.exp(np.asarray(atoms.log_width, dtype=float)), dtype=float)
    c = np.asarray(atoms.centers, dtype=float)
    z = np.asarray(site, dtype=float).reshape(1, -1)[:, : c.shape[1]]
    diff = c - z
    axis = getattr(atoms, "axis_width_scale", None)
    if axis is not None:
        diff = diff / np.asarray(axis, dtype=float)[None, : diff.shape[1]]
    d2 = np.sum(diff**2, axis=-1)
    w = A * _atom_profile_np(d2, s, str(getattr(atoms, "kernel", "gaussian")),
                             float(getattr(atoms, "kernel_cutoff", 2.5)))
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

    ⚠⚠ **TWO PROPERTIES OF THIS INSTRUMENT, MEASURED BY K7 (C2W8 pass 2, arm A
    §2) AND DOCUMENTED HERE BECAUSE THEY CHANGE HOW A POSITIVE READING MUST BE
    READ.** They are pytest-pinned in ``tests/test_compact_atoms.py`` (the arm's
    file); the census is the instrument, so the census documents them itself
    (pass-2 reconciliation item 2).

    1. **There is a POSITIVE FLOOR of ``tol / expansion_rate``, not 0.** The
       bisection only asks whether the relaxed point lands within ``tol`` of the
       site, so **a site whose relaxation barely moves reports a positive radius
       with no basin at all**. K7-2 measured ``0.001953`` on a deliberately
       *expanding* map (``z + 5(x-z)``, ``tol = 0.01``) — exactly ``tol/λ`` to one
       bisection cell — where the registered prediction was exactly 0.0.
       ⇒ a majority-positive G-CAP is **never** read alone: quote it beside
       ``lambda_min`` at the relaxed site **and the magnitude** of the radius
       (at the census operating point ``tol = sigma_q``, so the floor is
       ``sigma_q / expansion_rate``).
    2. **A flat site at the CONFINEMENT MINIMUM is a false positive of ~``r_hi``.**
       K7-5 planted a depth-``1e-9`` site at the origin, where ``V = alpha|q|^2``
       only, and measured ``0.99902`` at ``r_hi = 1.0``: everything relaxes back to
       the bowl's bottom, so the bisection saturates. Benign for this census only
       because real ``phi`` sites sit at ``|z| ~ 0.5-1.0``; ⛔ **any rig whose
       sites approach the origin must re-check this before quoting G-CAP.**
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

    ⚠⚠ **``theta_att``'s DYNAMIC RANGE IS ARM-DEPENDENT, AND IT DEGENERATES TO
    EXACTLY 0.0000 WHEN EVERY WELL CAPTURES** (C2W8 pass 2 housekeeping; §A29.7).
    The rule above returns ``0.0`` by construction when ``n_non_capturing == 0``,
    which is a *floor that was never exercised*, not a measurement that the floor
    is low. Because ``is_attractor`` tests ``depth > theta_att``, an arm in which
    everything captures admits every well as an attractor on a **vacuous**
    comparison. ⛔ **Consequence, binding on every cross-arm reading: ``P`` (whose
    numerator counts live attractors) is NOT comparable across arms without
    quoting ``n_non_capturing`` beside it.** Compare ``P`` only between arms with
    a non-degenerate ``theta_att``, or state the degeneracy on the number.
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
# ⭐⭐ G-ADDR — the ADDRESSABILITY leg (C2W8 PASS 3, prereg §2; charter §A30.1)
# --------------------------------------------------------------------------
#: The cue jitter, **as a dimensionless multiple of the measured key spacing**
#: (prereg-pass-3 §4). ⛔ Never an absolute sigma: the rig normalises addresses to
#: unit radius, so an absolute jitter makes `sigma / spacing` movable by
#: rescaling `phi` alone, with zero information gain — a leg expressed in
#: absolute units measures the SCALE, not the memory. 1.0 is the census's own
#: operating point to within 7 % (`sigma_q / spacing = 0.15 / 0.138..0.147`).
GADDR_KAPPA_Q = 1.0

#: A1 must clear BOTH `4 x chance` and `chance + 2 SE` (registered before the
#: first cell; see `.claude/outputs/c2w8p3-gate-addr/PREREG.md` §1).
GADDR_A1_CHANCE_MULT = 4.0
#: A2 (never-addressed fraction) ceiling.
GADDR_A2_MAX = 0.5


def _median_nn_dist(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=float)
    if p.shape[0] < 2:
        return float("nan")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(np.min(d, axis=1)))


def _sites_of(system) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ids, centers, sites)`` — sites are full store-space ``(c_i | a_i | 0)``."""
    ids, centers, pays = system.codebook()
    addr_dim, pay_dim = system.store.addr_dim, system.store.payload_dim
    sites = np.zeros((len(ids), system.store.dim), dtype=float)
    if len(ids):
        sites[:, :addr_dim] = centers
        sites[:, addr_dim: addr_dim + pay_dim] = pays
    return ids, np.asarray(centers, dtype=float), sites


def cue_queries(centers: np.ndarray, dim: int, *, spacing: float,
                kappa_q: float = GADDR_KAPPA_Q, n_per_item: int = 8,
                seed: int = 0, permute: bool = False
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The G-ADDR cue set: ``(q0, targets, declared_targets)``.

    For each live item ``i``, ``n_per_item`` queries ``c_i + kappa_q * spacing *
    eps`` (per-coordinate Gaussian) in the ADDRESS channels; payload channels are
    zero, which is what the read path enforces anyway.

    ``targets[k]`` is the index of the item the query was **generated from** —
    the ground truth that makes "correct basin" meaningful and that the pass-2
    instrument never had (it only ever asked "*some* basin").

    ``permute=True`` is **designed negative 2**: the *declared* target of each
    query is rotated to a different item (``i -> (i+1) % n``) while the queries
    themselves are unchanged, so a leg that cannot tell right from wrong scores
    the same on both. Returns ``declared_targets`` (what the leg is scored
    against) separately from ``targets`` (where the query really came from).
    """
    c = np.asarray(centers, dtype=float)
    n = c.shape[0]
    rng = np.random.default_rng(int(seed) + 60_000)
    tgt = np.repeat(np.arange(n), int(n_per_item))
    sig = float(kappa_q) * float(spacing)
    q0 = np.zeros((tgt.size, int(dim)), dtype=np.float32)
    q0[:, : c.shape[1]] = c[tgt] + rng.normal(size=(tgt.size, c.shape[1])) * sig
    declared = ((tgt + 1) % max(n, 1)) if permute else tgt
    return q0, tgt, declared


def gate_addr(system, *, spacing: Optional[float] = None,
              kappa_q: float = GADDR_KAPPA_Q, n_query_per_item: Optional[int] = None,
              seed: int = 0, capture: Optional[Sequence[float]] = None,
              stream_margin: Optional[float] = None,
              stream_margin_se: Optional[float] = None,
              launder_bytes: Optional[int] = None,
              permute: bool = False,
              n_dirs: int = 16, bisect_steps: int = 8) -> Dict[str, Any]:
    """⭐⭐ **G-ADDR** — *does a query reach the well of the item it asked for?*

    The leg the C2W8 gate was missing, and the reason it was missing is a defect
    class this wave caught three times: **a gate that cannot fail on the thing
    that matters.** Pass 1's ``M`` was vacuous (geometric leg ~10x the address
    resolution); monitor #3's admission gate refuses nothing (rate 0.000, still
    open); and the pass-2 capture gate passed **arm B on 3/3 seeds with 16/16
    items never read** — because G-CAP/G-DEC/G-DRIFT all measure retrievability
    *at the sites*, and **nothing measured whether a query reaches its site.**

    Three sub-legs, all two-sided, thresholds registered before the first cell
    (``.claude/outputs/c2w8p3-gate-addr/PREREG.md`` §1):

    * **A1 — correct-basin rate.** A cue query is correct iff its settled point
      ``q*`` (i) resolves to the **QUERIED** item in the address channels
      (``argmin_j |q*_addr - c_j| == i``, the shipped ``_assign`` rule) **and**
      (ii) lies inside that item's **measured** SC-6 capture radius in full store
      space (``|q* - z_i| <= rho_i``; ``rho_i`` NaN or 0 => there is no basin, so
      no query can be correct). ⛔ *"Some basin"* is reported (``any_basin_rate``)
      and is **not** the leg.
    * **A2 — never-addressed fraction:** live items with **zero** correct cue
      reads. ⚠ Deliberately **not** the banked telemetry ``n_never_read``: that
      counter credits a read only when ``covered = True``, and ``covered`` is
      computed on the **launch point** (``min_j |q0 - c_j| <= 1/2 min-sep``), so it
      is a property of the query distribution against the codebook and is very
      nearly independent of the store. The banked figure is reported beside this
      one by the caller.
    * **A3 — launder margin, at the DECLARED matching.** ``A3a`` compares the
      store and a **kNN-in-phi launder** on the *same queries under the same
      decision rule* (which item did you resolve to). ``A3b`` is the census
      stream's held-out class-accuracy margin (``read_acc - knn_acc``), supplied
      by the caller as ``stream_margin``; where no stream exists it is a declared
      **NOT-APPLICABLE**, never a null. ⛔ Both are **matched-ITEMS**, not
      matched-bytes: the ratio is reported and travels with every quotation.

      ⚠⚠ **The criterion is "does NOT LOSE beyond 2 SE", not "beats".**
      (``ERRATA`` §1, filed before any arm was scored.) The cue set draws
      ``q = c_i + sigma * eps`` with equal priors, under which **1-NN over the
      stored keys IS the Bayes-optimal decoder** — so requiring the physics to
      beat it on its own metric-native protocol is the metric-native-ceiling
      theorem written as a gate leg, and it makes G-ADDR **unpassable**. The
      designed positive control measured exactly that: A1 = 1.000, launder
      1.000, margin **0.000**. Whether daylight exists above the launder is the
      spine's question (PREREG-C2W8-PASS3 §6), not this gate's; both margins are
      reported **two-sided**.

    ``permute=True`` runs designed negative 2 in place (same store, same queries,
    wrong declared targets).
    """
    ids, centers, sites = _sites_of(system)
    n = int(len(ids))
    n_per = int(system.cfg.n_query_per_item if n_query_per_item is None
                else n_query_per_item)
    addr_dim = int(system.store.addr_dim)
    if n == 0:
        return {"status": "NOT RUN — empty codebook", "n_items": 0,
                "gate_addr_pass": False}
    sp = float(_median_nn_dist(centers) if spacing is None else spacing)
    rho = (np.asarray(capture, dtype=float) if capture is not None
           else capture_radii(system, sites, n_dirs=int(n_dirs),
                              steps=int(bisect_steps), seed=int(seed)))
    rho = np.where(np.isfinite(rho), rho, 0.0)

    q0, true_tgt, tgt = cue_queries(centers, int(system.store.dim), spacing=sp,
                                    kappa_q=float(kappa_q), n_per_item=n_per,
                                    seed=int(seed), permute=bool(permute))
    res = system.read(q0)
    q_star = np.asarray(res.state.q_star, dtype=float)
    a_star = q_star[:, :addr_dim]

    # -- leg (i): which item did the settle resolve to (shipped _assign rule) --
    d_addr = np.linalg.norm(a_star[:, None, :] - centers[None, :, :], axis=-1)
    resolved = np.argmin(d_addr, axis=1)
    voronoi_ok = resolved == tgt
    # -- leg (ii): is the settle INSIDE the queried item's measured basin -----
    d_full = np.linalg.norm(q_star - sites[tgt], axis=-1)
    in_own_basin = (rho[tgt] > 0.0) & (d_full <= rho[tgt])
    correct = voronoi_ok & in_own_basin
    # "some basin" — the pass-2-style question, reported and NOT the leg
    d_any = np.linalg.norm(q_star[:, None, :] - sites[None, :, :], axis=-1)
    any_basin = np.any(d_any <= np.maximum(rho, 0.0)[None, :], axis=1)

    # -- the launder: 1-NN in phi over the live keys, SAME queries ------------
    d_l = np.linalg.norm(q0[:, None, :addr_dim] - centers[None, :, :], axis=-1)
    launder_ok = np.argmin(d_l, axis=1) == tgt

    N = int(tgt.size)
    chance = 1.0 / float(n)
    se = float(np.sqrt(max(chance * (1.0 - chance), 0.0) / max(N, 1)))
    A1 = float(np.mean(correct))
    A1_vor = float(np.mean(voronoi_ok))
    A1_launder = float(np.mean(launder_ok))
    hits = np.array([int(np.sum(correct[tgt == i])) for i in range(n)], dtype=int)
    A2 = float(np.mean(hits == 0))
    A3a = float(A1_vor - A1_launder)
    # McNemar SE of a PAIRED difference in proportions (the two decoders see the
    # same queries), so "does not lose beyond 2 SE" is a paired statement.
    b = int(np.sum(voronoi_ok & ~launder_ok))
    c = int(np.sum(~voronoi_ok & launder_ok))
    se_a3a = float(np.sqrt(b + c) / max(N, 1))
    thr_a1 = float(max(GADDR_A1_CHANCE_MULT * chance, chance + 2.0 * se))
    a1_pass = bool(A1 >= thr_a1)
    a2_pass = bool(A2 <= GADDR_A2_MAX)
    a3a_pass = bool(A3a >= -2.0 * se_a3a)
    a3b_applicable = stream_margin is not None and np.isfinite(
        float(stream_margin if stream_margin is not None else np.nan))
    se_a3b = float(stream_margin_se) if stream_margin_se is not None else 0.0
    a3b_pass = (bool(float(stream_margin) >= -2.0 * se_a3b) if a3b_applicable
                else True)
    clu_bytes = int(system.n_bytes())
    knn_bytes = int(n * (addr_dim + int(system.store.payload_dim)) * 4
                    if launder_bytes is None else launder_bytes)
    return {
        "n_items": n, "n_queries": N,
        "kappa_q": float(kappa_q), "spacing_ref": sp,
        "cue_sigma": float(kappa_q) * sp,
        # ⭐ §4's guard: every geometric quantity as a DIMENSIONLESS RATIO with
        # the scale stated. `spacing_ref` is the rig's DECLARED measured key
        # spacing (the census hands in its own `median_nn_task1`, the number
        # G-DRIFT is scored against); `codebook_spacing` is the live store's own
        # median-NN, which is the resolution the read must actually beat.
        "codebook_spacing": float(_median_nn_dist(centers)),
        "cue_sigma_over_spacing_ref": float(kappa_q),
        "cue_sigma_over_codebook_spacing": float(
            (float(kappa_q) * sp) / max(_median_nn_dist(centers), 1e-12)),
        "cue_displacement_over_codebook_spacing": float(
            (float(kappa_q) * sp * np.sqrt(addr_dim))
            / max(_median_nn_dist(centers), 1e-12)),
        "permuted_targets": bool(permute),
        "A1": {
            "correct_basin_rate": A1, "chance": chance, "se": se,
            "threshold": thr_a1,
            "rule": ("settle resolves to the QUERIED item (address argmin) AND "
                     "lies inside that item's MEASURED capture radius in full "
                     "store space; threshold max(4 x chance, chance + 2 SE)"),
            "margin_in_se": float((A1 - chance) / se) if se > 0 else float("nan"),
            "pass": a1_pass,
            # reported, never the leg
            "voronoi_only_rate": A1_vor,
            "any_basin_rate": float(np.mean(any_basin)),
            "in_own_basin_rate": float(np.mean(in_own_basin)),
            "n_items_with_zero_basin": int(np.sum(rho <= 0.0)),
            "capture_radius_median": float(np.median(rho)),
        },
        "A2": {
            "never_addressed_frac": A2,
            "n_never_addressed": int(np.sum(hits == 0)),
            "threshold": float(GADDR_A2_MAX), "pass": a2_pass,
            "correct_hits_by_item": {int(ids[i]): int(hits[i]) for i in range(n)},
            "rule": ("live items with ZERO correct cue reads; NOT the banked "
                     "telemetry n_never_read (which is launch-point coverage)"),
        },
        "A3": {
            "matching": "matched-ITEMS (same keys, same queries) — NOT matched-bytes",
            "clu_total_bytes": clu_bytes, "knn_launder_bytes": knn_bytes,
            "byte_ratio_clu_over_launder": float(clu_bytes / max(knn_bytes, 1)),
            "A3a_cue_margin": A3a,
            "A3a_store_rate": A1_vor, "A3a_launder_rate": A1_launder,
            "A3a_se_paired": se_a3a, "A3a_threshold": float(-2.0 * se_a3a),
            "A3a_discordant_store_only": b, "A3a_discordant_launder_only": c,
            "A3a_pass": a3a_pass,
            "A3a_strict_margin": float(A1 - A1_launder),
            "A3b_stream_margin": (float(stream_margin) if a3b_applicable else None),
            "A3b_se_pooled": (se_a3b if a3b_applicable else None),
            "A3b_threshold": (float(-2.0 * se_a3b) if a3b_applicable else None),
            "A3b_applicable": bool(a3b_applicable),
            "A3b_status": ("measured" if a3b_applicable else
                           "NOT-APPLICABLE (declared): no stream on this rig — "
                           "never reported as a null"),
            "A3b_pass": a3b_pass,
            "pass": bool(a3a_pass and a3b_pass),
        },
        "gate_addr_pass": bool(a1_pass and a2_pass and a3a_pass and a3b_pass),
        "rule": ("G-ADDR (PREREG-C2W8-PASS3 §2, thresholds in this spoke's "
                 "PREREG.md §1 as amended by its ERRATA §1): A1 correct-basin "
                 ">= max(4 x chance, chance + 2 SE) AND A2 never-addressed "
                 "<= 0.5 AND the store does NOT LOSE to its own kNN-in-phi "
                 "launder beyond 2 SE on the cue set (A3a) nor on the stream "
                 "(A3b, where applicable). ⛔ 'beats the launder' is NOT the "
                 "criterion: 1-NN is Bayes-optimal on a metric-native cue."),
    }


def gate_addr_verdict(legs_by_seed: Sequence[Dict[str, Any]],
                      *, min_seeds: int = 3) -> Dict[str, Any]:
    """The arm-level G-ADDR verdict — mechanical, never a judgement call.

    ``pass`` iff every seed passes every leg and at least ``min_seeds`` were run.
    Single-cell rigs (the designed controls) call it with ``min_seeds = 1``.
    """
    legs = list(legs_by_seed)
    ok = [bool(g.get("gate_addr_pass", False)) for g in legs]
    return {
        "n_seeds": len(legs),
        "min_seeds": int(min_seeds),
        "A1_pass_seeds": int(sum(bool(g["A1"]["pass"]) for g in legs)),
        "A2_pass_seeds": int(sum(bool(g["A2"]["pass"]) for g in legs)),
        "A3_pass_seeds": int(sum(bool(g["A3"]["pass"]) for g in legs)),
        "all_legs_same_seed": int(sum(ok)),
        "gate_addr_pass": bool(len(legs) >= int(min_seeds) and all(ok) and legs),
        "A1_by_seed": [float(g["A1"]["correct_basin_rate"]) for g in legs],
        "A2_by_seed": [float(g["A2"]["never_addressed_frac"]) for g in legs],
        "A3a_by_seed": [float(g["A3"]["A3a_cue_margin"]) for g in legs],
        "A3b_by_seed": [g["A3"]["A3b_stream_margin"] for g in legs],
        "rule": ("every leg on every seed, >= min_seeds seeds "
                 "(PREREG-C2W8-PASS3 §2)"),
    }


# --------------------------------------------------------------------------
# RULING 3's counterfactual — outcome, not identity (prereg-pass-3 §7)
# --------------------------------------------------------------------------
def displaced_write_counterfactual(system, item_id: int, address, payload: float, *,
                                   delta, seed: int = 0) -> Dict[str, Any]:
    """⭐ Can the attractor MOVE OFF the stored key when the write objective asks?

    ``atom_site_local_init`` (C2W8 pass 2 arm A) is ruled COMPLIANT **conditional
    on this check** (charter §A30.3): initialising the admitted slot's atoms at
    the item's own site is a designed *starting point*, not a constraint on where
    the attractor ends up — **unless it is**, in which case the near-zero site
    drift arm A measured is an algebraic identity, capture was bought by pinning,
    and the ruling REVERSES.

    The counterfactual is deliberately the harshest honest one:

    1. admit the item at ``address`` (the codebook records ``address``);
    2. run the **site-local init at ``address``** — atoms placed exactly at the
       stored key, i.e. the lever under scrutiny, at full strength;
    3. run the shipped learned write with its target **displaced to
       ``address + delta``** — the write objective now prefers a minimum that is
       NOT the stored key;
    4. relax from the stored key and see where the attractor is.

    Returns ``follow = |q* - address| / |delta|`` (0 = provably pinned, 1 = the
    attractor went exactly where the objective asked). ⛔ ``follow < 0.5`` is the
    reversal condition; the caller escalates rather than shipping.
    """
    import jax

    d = int(system.store.addr_dim)
    m = int(system.store.payload_dim)
    a = np.asarray(address, dtype=float).reshape(-1)[:d]
    dl = np.asarray(delta, dtype=float).reshape(-1)
    if dl.size < d:
        dl = np.pad(dl, (0, d - dl.size))
    dl = dl[:d]
    pay = np.atleast_1d(np.asarray(payload, dtype=float))

    res = system.controller.admit(int(item_id), a, float(pay[0]))
    if not res.applied:
        raise RuntimeError(f"counterfactual: admission refused: {res.reason}")
    system._payloads[int(item_id)] = pay
    system._born[int(item_id)] = system._t
    slot = system._slot_of(int(item_id))
    key = jax.random.PRNGKey(int(seed) + 5171)

    site = np.zeros((system.store.dim,), dtype=float)
    site[:d] = a
    site[d: d + m] = pay
    target = site.copy()
    target[:d] = a + dl

    # (2) the lever at full strength: atoms initialised AT the stored key
    if system.cfg.atom_site_local_init:
        system._localize_slot_atoms(slot, a, pay, jax.random.fold_in(key, int(slot)))
    # (3) the write objective prefers the DISPLACED minimum
    loss = system._write_item(slot, a + dl, pay, key)

    settled = np.asarray(system._relax_points(site[None, :]))[0]
    move = float(np.linalg.norm(settled[:d] - a))
    norm_delta = float(np.linalg.norm(dl))
    to_target = float(np.linalg.norm(settled[:d] - (a + dl)))
    follow = float(move / norm_delta) if norm_delta > 0 else float("nan")
    return {
        "item_id": int(item_id), "slot": int(slot),
        "delta_norm": norm_delta,
        "moved_off_key": move,
        "residual_to_displaced_target": to_target,
        "follow_fraction": follow,
        "attractor_can_move": bool(follow >= 0.5),
        "write_loss": float(loss),
        "atom_site_local_init": bool(system.cfg.atom_site_local_init),
        "settled_point": settled.tolist(),
        "rule": ("follow = |q* - stored key| / |delta|; PASS iff >= 0.5 "
                 "(PREREG c2w8p3-gate-addr §3 P6). follow < 0.5 => the attractor "
                 "is algebraically pinned to the key => §A30.3 REVERSES the "
                 "atom_site_local_init compliance ruling => ESCALATE"),
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
    # ⭐ C2W8 pass 3 — the addressability leg and Ruling 3's counterfactual
    "GADDR_KAPPA_Q", "GADDR_A1_CHANCE_MULT", "GADDR_A2_MAX",
    "cue_queries", "gate_addr", "gate_addr_verdict",
    "displaced_write_counterfactual",
]
