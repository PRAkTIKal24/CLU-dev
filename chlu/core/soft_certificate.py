"""⭐ **SC-1…SC-7 — the merge certificate as a MONITORED SOFT CONSTRAINT.**

Standing harness doctrine (Head ruling §A9.8: *"it is what makes any
non-separable measurement legal"*), implemented **verbatim** from
`doctrine-repairs.md` §4.4. This module does not redesign the spec; it lands it.

**The structural finding it exists to repair.** The shipped harness sets

    d_safe := 2 s_max + kappa' sigma_q            (`derived_d_safe`)

so the **admission radius IS the certificate radius** — the two "mutually
exclusive" bands (#3's fire-rate band and the merge certificate) are **one
object**. Driving the certificate into violation therefore drives the gate's fire
rate to ``f = 1.000`` (it refuses everything), and ``f in {0, 1}`` is monitor
#3's trip. That is *why* charter §A2.5 was confirmed **7/7** under the shipped
rule and **refuted 11/12** under the soft certificate: separating the two makes
both satisfiable at once, and the non-separable region the campaign has never had
access to opens up.

**The spec.**

``SC-1`` — break the identification. ``d_safe = zeta * sep_expected`` (``zeta =
    0.6``, the harness's own S4 convention) is an **independent, declared**
    admission radius. ``R_cert = 2 s_max + kappa' sigma_q`` is **still computed
    and still reported** — it is no longer the gate.
    *(This retires the ``d_safe_override`` hack: the gym needed it twice and the
    harness once, always "deliberately out of band" — **the override WAS the soft
    certificate, undeclared.**)*
``SC-2`` — every admitted write logs ``cert_margin = sep_after - R_cert`` and
    ``deficit_rel = max(0, -cert_margin) / sep_after``.
``SC-3`` — the violation budget, declared per run: ``deficit_rel <= B`` with
    **``B = 0.33``** and ``mean(deficit_rel) <= B/2``. ⛔ **Exceeding it is a TRIP
    of monitor #3, never a refusal** — a soft constraint that refuses is a hard
    constraint with extra steps.
``SC-4`` — what replaces the guarantee, three runtime legs, all already computed:
    (i) ``lambda_min,i > lambda_floor`` at every live site (#8-N3) — the *exact*
    non-merger condition the certificate approximates; (ii) the **C3 first-order
    calibration leg** (§1.3, :func:`c3_calibration`) — the certificate's job was
    to bound crowding drift, so bound it *measured*; (iii) monitor #2 with the
    corrected radius, INAPPLICABLE where (i) fails.
``SC-5`` — what is given up, stated in the artifact: :data:`SC5_STATEMENT`.
``SC-6`` — the hard floor that does **not** relax: ``lambda_min,i > 0`` at every
    live site, **plus a measured capture radius** (:func:`capture_radius`) —
    ``lambda_min > 0`` is necessary but **not sufficient** for a nonempty basin
    (measured basin **0.000** at ``lambda_min = +0.910``), so a site whose
    measured basin is below ``sigma_q`` is refused certification.
``SC-7`` — the falsifier, recorded rather than coded (:data:`SC7_FALSIFIER`).

⚠ **The price, carried unsoftened** (`doctrine-repairs.md` §4.3): decoupling buys
``rho_ex`` up to **6.3x** at a ``lambda_min`` cost of **2.2-6.0x**, and **the
dividend in that region stays ~0** (+0.0043 … -0.0067). The relaxation is a
**precondition, not a result** — §A2.1 predicted exactly this.
⚠ **The budget's outer edge was located by a broken proxy.** R1's corrected
inradius is a *prerequisite* for setting ``B``, not an independent cleanup;
``sep/2`` is **never-quote** as a certified inradius and the corrected proxy is
valid only inside ``s/sep in [0.15, 0.30]``. That domain travels with ``B = 0.33``
(:data:`BUDGET_DOMAIN`).

⛔ **DEFAULT-OFF and additive.** With ``CluSystemConfig.soft_certificate = False``
the harness is **bit-identical to `6ff4c1d`**; a blocking regression test asserts
it (the C1W27 ``payload_gate`` precedent).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: SC-1: the harness's own S4 convention, ``d_safe = zeta * sep_expected``.
SC1_ZETA = 0.6

#: SC-3: the violation budget. **Measured edge** (`doctrine-repairs.md` §4.3):
#: deficits 0 / 11.6 / 32.9 / 54.2 % of ``sep`` give ``rho_ex`` 0.127 / 0.183 /
#: 0.294 / 7.94, the last being ``D > U`` (the R1 estimator breaking down).
SC3_BUDGET_B = 0.33

#: ⚠ The domain in which ``B = 0.33`` was located, and outside which the
#: corrected inradius proxy is invalid. **State it wherever B appears.**
BUDGET_DOMAIN = "s/sep in [0.15, 0.30]; the edge was located by the (broken) sep/2 proxy"

#: §1.3, derived not tuned: ``kappa = 3`` from the pooled ``q95(Delta/B) = 4.20``
#: rounded inward; ``eta = 0.10`` from the ungated ``P[Delta > 3B] = 0.062`` with
#: a 1.6x margin; the calibration band ``[1/3, 3]`` contains N74's shipped gated
#: band ``[0.73, 1.63]``, so a healthy store passes with room.
C3_KAPPA = 3.0
C3_ETA = 0.10
C3_RHO_BAND: Tuple[float, float] = (1.0 / 3.0, 3.0)

#: The settle's numerical floor at 900 steps — below it a "drift" is arithmetic.
C3_DELTA_NUM = 1e-6

#: SC-4(i)/SC-6: the hard floor. ``lambda_min > 0`` does **not** certify a
#: nonempty basin (measured basin 0.000 at ``lambda_min = +0.910``), which is why
#: SC-6 adds the measured capture radius.
SC6_LAMBDA_FLOOR = 0.0

SC5_STATEMENT = (
    "Prop D1's guarantee (settle = argmin inside every certified ball, hence "
    "D <= U) NO LONGER HOLDS A PRIORI. It is replaced by a measured pair "
    "(lambda_min, rho_C3) and a declared budget B. The measured price on the "
    "theorist's grid: lambda_min / 2.2-6.0, implicit-gradient conditioning and "
    "truncation depth ~ 1/lambda_min, rho_ex x1.4-6.3, and the dividend in that "
    "region stays ~0 (+0.0043 ... -0.0067) — a PRECONDITION, not a result."
)

SC7_FALSIFIER = (
    "SC-7 (falsifier, not code): if a shared/factored store's wells cannot hold "
    "lambda_min > lambda_floor at ANY admissible B, then basin interaction and "
    "non-degeneracy are genuinely disjoint and that is a Head escalation. On the "
    "theorist's grid they are NOT disjoint: at B = 0.33, lambda_min = +3.19 with "
    "rho_ex = 0.294."
)


# --------------------------------------------------------------------------
# the config object (kept HERE, not in `chlu/config.py`, per the C2 rule that a
# C2 config object lives in the C2-owned module that uses it)
# --------------------------------------------------------------------------
@dataclass
class SoftCertificateConfig:
    """Every SC knob, with the spec's own constants as defaults.

    ``enabled = False`` is the shipped harness, bit-identical.
    """

    enabled: bool = False
    zeta: float = SC1_ZETA                  # SC-1
    budget_B: float = SC3_BUDGET_B          # SC-3
    lambda_floor: float = SC6_LAMBDA_FLOOR  # SC-4(i)/SC-6
    kappa: float = C3_KAPPA                 # SC-4(ii)
    eta: float = C3_ETA
    rho_band: Tuple[float, float] = C3_RHO_BAND
    delta_num: float = C3_DELTA_NUM
    #: SC-6: 32-direction bisection at one site per consolidation (~6 s in the
    #: toy). 0 disables the measurement and the SC-6 leg reports INAPPLICABLE —
    #: it never silently passes.
    capture_dirs: int = 32
    capture_bisect_steps: int = 12
    #: ``None`` => derive ``sep_expected`` from the designed site geometry.
    sep_expected: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["budget_domain"] = BUDGET_DOMAIN
        return d


# --------------------------------------------------------------------------
# SC-1 / SC-2 — the two radii, now two objects
# --------------------------------------------------------------------------
def expected_separation(sites: np.ndarray) -> float:
    """``sep_expected``: the min pairwise separation of the DESIGNED site set.

    Independent of ``s_max`` and ``sigma_q`` by construction — which is the whole
    point of SC-1. It is a property of the *address geometry* (how many items in
    what ball), not of the write's width or the query law.
    """
    s = np.atleast_2d(np.asarray(sites, dtype=float))
    if s.shape[0] < 2:
        return float("nan")
    d = np.linalg.norm(s[:, None, :] - s[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))


def soft_d_safe(sep_expected: float, zeta: float = SC1_ZETA) -> float:
    """⭐ **SC-1**: the declared admission radius ``zeta * sep_expected``."""
    return float(zeta) * float(sep_expected)


def cert_radius(s_max: float, sigma_q: float, kappa_prime: float = 2.576) -> float:
    """``R_cert = 2 s_max + kappa' sigma_q`` — **still computed, no longer the gate.**

    Numerically identical to
    :func:`~chlu.core.clu_controller.derived_d_safe`; it is given its own name
    here because SC-1's entire content is that these are **two different
    objects** that the shipped code identified.
    """
    return float(2.0 * float(s_max) + float(kappa_prime) * float(sigma_q))


def cert_margin(sep_after: float, r_cert: float) -> Dict[str, float]:
    """⭐ **SC-2**: ``cert_margin`` and ``deficit_rel`` for one write."""
    sep = float(sep_after)
    m = sep - float(r_cert)
    return {"cert_margin": float(m), "sep_after": sep, "R_cert": float(r_cert),
            "deficit_rel": float(max(0.0, -m) / max(sep, 1e-12)),
            "violated": bool(m < 0.0)}


def budget_state(deficits: Sequence[float], B: float = SC3_BUDGET_B) -> Dict[str, Any]:
    """⭐ **SC-3**: the violation budget, evaluated. ⛔ **A trip, never a refusal.**"""
    d = np.asarray([x for x in np.asarray(deficits, dtype=float).ravel()
                    if np.isfinite(x)], dtype=float)
    if d.size == 0:
        return {"n": 0, "max_deficit_rel": float("nan"),
                "mean_deficit_rel": float("nan"), "B": float(B),
                "within_budget": True, "applicable": False,
                "budget_domain": BUDGET_DOMAIN}
    mx, mn = float(np.max(d)), float(np.mean(d))
    return {"n": int(d.size), "max_deficit_rel": mx, "mean_deficit_rel": mn,
            "B": float(B), "B_half": float(B) / 2.0,
            "leg_max_ok": bool(mx <= float(B)),
            "leg_mean_ok": bool(mn <= float(B) / 2.0),
            "within_budget": bool(mx <= float(B) and mn <= float(B) / 2.0),
            "applicable": True, "budget_domain": BUDGET_DOMAIN,
            "note": "exceeding the budget TRIPS monitor #3; it never refuses a write"}


# --------------------------------------------------------------------------
# SC-4(ii) / D3 — the C3 first-order calibration leg (monitor #3, leg ii')
# --------------------------------------------------------------------------
def c3_calibration(pairs: Sequence[Dict[str, float]], *,
                   kappa: float = C3_KAPPA, eta: float = C3_ETA,
                   rho_band: Tuple[float, float] = C3_RHO_BAND,
                   lambda_floor: float = SC6_LAMBDA_FLOOR,
                   delta_num: float = C3_DELTA_NUM) -> Dict[str, Any]:
    """⭐ **D3 / SC-4(ii)** — the leg that REPLACES ``corr(margin, drift)``.

    Each pair is ``{"B": predicted drift ||grad dV_j(q_i*)|| / lambda_min,i,
    "delta": measured drift at the RELAXED fixed point, "lambda_min": ...}``.

    * **Qualifying** iff ``lambda_min > lambda_floor`` **and** ``delta >
      delta_num``. ⛔ **No ``max(lambda, 1e-9)`` clamp** (fix S3): the clamp sends
      ``B -> inf`` and ``rho_C3 -> 0`` at any non-minimum, i.e. it reports a
      *perfect* certificate precisely where there is none.
    * ``rho_C3 = median(delta / B)`` (calibration), ``v = P[delta > kappa B]``
      (soundness).
    * **Trip iff** ``rho_C3 not in rho_band`` **or** ``v > eta``.
    * **INAPPLICABLE** (never "pass") with fewer than **3** qualifying pairs —
      which is what the 18/28 gym cells whose sites are not minima become, and
      that is #8-N3's trip, not this one's (no double-counting).

    Measured head-to-head in the gym-like regime (12 seeds): shipped leg
    ``+0.412`` mean with **1/12 sign flips** and 4/12 below its bar; this leg
    **+0.914** mean, **0/12** sign flips, **0/12** below bar — **at zero extra
    cost**, because ``CluSystem._c3_check`` already computes the ratio.
    """
    rows = [dict(p) for p in (pairs or [])]
    qual = [p for p in rows
            if np.isfinite(p.get("B", np.nan)) and np.isfinite(p.get("delta", np.nan))
            and p.get("B", 0.0) > 0.0
            and float(p.get("lambda_min", -np.inf)) > float(lambda_floor)
            and float(p["delta"]) > float(delta_num)]
    out: Dict[str, Any] = {
        "n_pairs": len(rows), "n_qualifying": len(qual),
        "lambda_floor": float(lambda_floor), "delta_num": float(delta_num),
        "kappa": float(kappa), "eta": float(eta),
        "rho_band": [float(rho_band[0]), float(rho_band[1])],
        "leg": "c3_first_order_calibration (D3: replaces corr(margin, drift))",
    }
    if len(qual) < 3:
        out.update({"applicable": False, "tripped": False,
                    "rho_c3": float("nan"), "violation_rate": float("nan"),
                    "why": ("fewer than 3 qualifying pairs (lambda_min <= floor "
                            "=> the site is not a minimum => #8-N3's trip, not "
                            "this one's)")})
        return out
    r = np.asarray([float(p["delta"]) / float(p["B"]) for p in qual], dtype=float)
    rho = float(np.median(r))
    v = float(np.mean(r > float(kappa)))
    out.update({
        "applicable": True, "rho_c3": rho, "violation_rate": v,
        "median_ratio": rho, "q95_ratio": float(np.quantile(r, 0.95)),
        "tripped": bool(rho < rho_band[0] or rho > rho_band[1] or v > float(eta)),
    })
    return out


# --------------------------------------------------------------------------
# SC-6 — the hard floor that does not relax
# --------------------------------------------------------------------------
def capture_radius(relax_fn: Callable[[np.ndarray], np.ndarray], site: np.ndarray, *,
                   n_dirs: int = 32, r_hi: float = 1.0, steps: int = 12,
                   tol: float = 0.1, seed: int = 0) -> Dict[str, float]:
    """⭐ **SC-6**: the **measured** basin radius at one site (32-direction bisection).

    ``lambda_min > 0`` certifies a *local* minimum, **not a nonempty basin** — the
    theorist measured a basin of **0.000** at ``lambda_min = +0.910``. So the
    non-merger floor is measured, not inferred: along each of ``n_dirs`` random
    unit directions, bisect for the largest displacement that still relaxes back
    to within ``tol`` of the site; the capture radius is the **minimum** over
    directions (a basin is only as wide as its narrowest direction).

    ``relax_fn`` maps ``(n, dim) -> (n, dim)`` (the post-write relaxation).
    """
    z = np.asarray(site, dtype=float).reshape(-1)
    rng = np.random.default_rng(int(seed))
    dirs = rng.normal(size=(int(n_dirs), z.size))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    lo = np.zeros((int(n_dirs),), dtype=float)
    hi = np.full((int(n_dirs),), float(r_hi), dtype=float)
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        pts = z[None, :] + dirs * mid[:, None]
        end = np.asarray(relax_fn(pts.astype(np.float32)))
        back = np.linalg.norm(end - z[None, :], axis=1) <= float(tol)
        lo = np.where(back, mid, lo)
        hi = np.where(back, hi, mid)
    return {"capture_radius": float(np.min(lo)), "median_radius": float(np.median(lo)),
            "max_radius": float(np.max(lo)), "n_dirs": int(n_dirs),
            "r_hi": float(r_hi), "tol": float(tol),
            "note": ("SC-6: lambda_min > 0 is NECESSARY but NOT SUFFICIENT for a "
                     "nonempty basin (measured basin 0.000 at lambda_min = +0.910)")}


def sc6_state(lambda_mins: Sequence[float], *, sigma_q: float,
              capture: Optional[Dict[str, float]] = None,
              lambda_floor: float = SC6_LAMBDA_FLOOR) -> Dict[str, Any]:
    """⭐ **SC-6**: the floor's verdict — ``lambda_min > 0`` **and** basin >= ``sigma_q``."""
    lam = np.asarray([x for x in np.asarray(lambda_mins, dtype=float).ravel()
                      if np.isfinite(x)], dtype=float)
    lam_ok = bool(lam.size > 0 and float(np.min(lam)) > float(lambda_floor))
    out: Dict[str, Any] = {
        "lambda_min_over_sites": (float(np.min(lam)) if lam.size else float("nan")),
        "lambda_floor": float(lambda_floor), "leg_lambda_ok": lam_ok,
        "n_sites": int(lam.size),
    }
    if capture is None:
        out.update({"leg_capture_applicable": False, "certified": False,
                    "why": ("SC-6's capture radius was NOT measured => the floor "
                            "is INAPPLICABLE, never 'passed'")})
        return out
    r = float(capture.get("capture_radius", float("nan")))
    ok = bool(np.isfinite(r) and r >= float(sigma_q))
    out.update({"leg_capture_applicable": True, "capture_radius": r,
                "sigma_q": float(sigma_q), "leg_capture_ok": ok,
                "certified": bool(lam_ok and ok)})
    return out


# --------------------------------------------------------------------------
# ⭐ K9 — the merge criterion, RE-REGISTERED (C2W8 pass 2, rider 2)
#
# `ERRATA-C2W8-PASS2.md` §2 is the registration; this is its implementation.
# ⛔ **This is a PREDICATE, not a merge verb.** Nothing here merges, prunes or
# writes; merge stays unbuilt until the capture gate passes
# (`PREREG-C2W8-PASS2.md` §6). `well_lifecycle.mergeable_pairs` — pass 1's
# criterion — is untouched and still the harness's own rule.
#
# **Why pass 1's criterion had to be retired.** It admitted a pair iff
# ``payload_dist <= 0.1`` and ``center_sep <= R_cert``. On the frozen census
# ``R_cert`` was **10.31-11.23x the measured key spacing**, so the geometric leg
# refused **nothing** (monitor #3 ``vacuous_gate`` tripped 3/3 at refusal rate
# **0.000**), and every admitted pair's ``payload_dist`` was **exactly 0.0** —
# true by construction in a class-incremental stream. ``M`` measured *"the stream
# contains two items of the same class"*.
# --------------------------------------------------------------------------

#: K9: ``r_merge = rho_geom * key_spacing``. **1.0 = one address resolution** —
#: two centres are one well only when the store cannot tell them apart.
MERGE_RHO_GEOM = 1.0

#: K9: the payload leg's tolerance as a fraction of the **measured** payload
#: spread. An absolute tolerance is what made pass 1 vacuous.
MERGE_TAU_PAYLOAD = 0.25

#: K9 anti-vacuity clause: a payload spread at or below this carries no
#: discriminative content ⇒ the leg is INAPPLICABLE and **refuses**.
MERGE_PAYLOAD_DEGENERATE = 1e-9


@dataclass
class MergeCriterionConfig:
    """The re-registered merge criterion's knobs (defaults = the registration)."""

    rho_geom: float = MERGE_RHO_GEOM
    tau_payload: float = MERGE_TAU_PAYLOAD
    payload_degenerate: float = MERGE_PAYLOAD_DEGENERATE

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def payload_scale_from_pairs(payload_dists: Sequence[float]) -> float:
    """The payload channel's **measured** spread: median pairwise ``||a_i - a_j||``.

    ⚠ Measured over the **whole** live-well pair population, never over the
    admitted subset — scaling the tolerance by the spread of the pairs the
    tolerance already selected is circular, and would reproduce pass 1's vacuity
    with extra steps. Returns 0.0 for an empty population (⇒ degenerate ⇒ the
    payload leg refuses, never silently passes).
    """
    d = np.asarray([x for x in np.asarray(payload_dists, dtype=float).ravel()
                    if np.isfinite(x)], dtype=float)
    return float(np.median(d)) if d.size else 0.0


def merge_admissible(center_sep: float, payload_dist: float, *,
                     key_spacing: float, payload_scale: float,
                     cfg: Optional[MergeCriterionConfig] = None) -> Dict[str, Any]:
    """⭐ **K9**: is one pair admissible for merge? Two legs, both able to refuse.

    * **geometry** — ``center_sep <= rho_geom * key_spacing`` with ``key_spacing``
      the **measured** ``median_nn_task1``. INAPPLICABLE (⇒ **refuse**) if the
      spacing is missing/non-finite/non-positive: an unmeasured ruler certifies
      nothing.
    * **payload** — ``payload_dist <= tau_payload * payload_scale`` with
      ``payload_scale`` **measured** (:func:`payload_scale_from_pairs`).
      INAPPLICABLE (⇒ **refuse**) when the spread is degenerate — the clause that
      kills pass 1's "all payloads are 0.0, so all payloads match" vacuity.

    Returns the verdict **with the leg that refused it** (``refused_on``), so a
    refusal is never anonymous.
    """
    c = cfg or MergeCriterionConfig()
    ks, ps = float(key_spacing), float(payload_scale)
    dc, dp = float(center_sep), float(payload_dist)

    geom_applicable = bool(np.isfinite(ks) and ks > 0.0)
    r_merge = float(c.rho_geom) * ks if geom_applicable else float("nan")
    geom_ok = bool(geom_applicable and np.isfinite(dc) and dc <= r_merge)

    pay_applicable = bool(np.isfinite(ps) and ps > float(c.payload_degenerate))
    pay_tol = float(c.tau_payload) * ps if pay_applicable else float("nan")
    pay_ok = bool(pay_applicable and np.isfinite(dp) and dp <= pay_tol)

    refused_on = []
    if not geom_ok:
        refused_on.append("geometry_inapplicable" if not geom_applicable else "geometry")
    if not pay_ok:
        refused_on.append("payload_degenerate" if not pay_applicable else "payload")
    return {
        "admitted": bool(geom_ok and pay_ok),
        "refused_on": refused_on,
        "center_sep": dc, "payload_dist": dp,
        "key_spacing": ks, "payload_scale": ps,
        "r_merge": r_merge, "payload_tol": pay_tol,
        "rho_geom": float(c.rho_geom), "tau_payload": float(c.tau_payload),
        "geometry_applicable": geom_applicable, "geometry_ok": geom_ok,
        "payload_applicable": pay_applicable, "payload_ok": pay_ok,
        #: commensurability with the address resolution: `rho_geom` by
        #: construction (pass 1's identified radius was 10.31-11.23x)
        "commensurability": (float(r_merge / ks) if geom_applicable else float("nan")),
    }


def merge_criterion_report(pairs: Sequence[Dict[str, float]], *, key_spacing: float,
                           payload_scale: Optional[float] = None,
                           cfg: Optional[MergeCriterionConfig] = None) -> Dict[str, Any]:
    """⭐ **K9** over a pair population: refusal rate + the two-sided vacuity check.

    ``pairs`` are dicts with ``center_sep`` and ``payload_dist``. If
    ``payload_scale`` is None it is measured from the population itself
    (:func:`payload_scale_from_pairs`) — so pass **all** live-well pairs, not a
    pre-selected subset.

    ⚠ ``vacuous_gate_would_trip`` applies monitor #3's own ``f in {0, 1}``
    convention **in both directions**: a criterion that refuses *everything* is as
    uninformative as one that refuses *nothing*, and is reported as such.
    """
    c = cfg or MergeCriterionConfig()
    rows = [dict(p) for p in (pairs or [])]
    scale = (payload_scale_from_pairs([p.get("payload_dist", np.nan) for p in rows])
             if payload_scale is None else float(payload_scale))
    verdicts = [merge_admissible(p.get("center_sep", np.nan),
                                 p.get("payload_dist", np.nan),
                                 key_spacing=key_spacing, payload_scale=scale, cfg=c)
                for p in rows]
    n = len(verdicts)
    n_adm = sum(1 for v in verdicts if v["admitted"])
    rate = float((n - n_adm) / n) if n else float("nan")
    return {
        "n_pairs": n, "n_admitted": n_adm,
        "M_prime": float(n_adm / n) if n else float("nan"),
        "refusal_rate": rate,
        "n_refused_geometry": sum(1 for v in verdicts if "geometry" in v["refused_on"]),
        "n_refused_payload": sum(1 for v in verdicts if "payload" in v["refused_on"]),
        "n_refused_payload_degenerate": sum(
            1 for v in verdicts if "payload_degenerate" in v["refused_on"]),
        "n_refused_geometry_inapplicable": sum(
            1 for v in verdicts if "geometry_inapplicable" in v["refused_on"]),
        "key_spacing": float(key_spacing), "payload_scale": scale,
        "r_merge": float(c.rho_geom) * float(key_spacing),
        "commensurability": float(c.rho_geom),
        "criterion": c.as_dict(),
        "vacuous_gate_would_trip": bool(n > 0 and rate in (0.0, 1.0)),
        "verdicts": verdicts,
        "note": ("K9 (ERRATA-C2W8-PASS2 §2): a PREDICATE, not a merge verb; "
                 "pass 1's criterion refused 0.000 at R_cert = 10.31-11.23x the "
                 "key spacing with every payload_dist exactly 0.0"),
    }


# --------------------------------------------------------------------------
# the whole report
# --------------------------------------------------------------------------
def soft_certificate_report(cfg: SoftCertificateConfig, *, sep_expected: float,
                            sep_after: float, s_max: float, sigma_q: float,
                            kappa_prime: float = 2.576,
                            deficits: Optional[Sequence[float]] = None,
                            c3_pairs: Optional[Sequence[dict]] = None,
                            lambda_mins: Optional[Sequence[float]] = None,
                            capture: Optional[Dict[str, float]] = None
                            ) -> Dict[str, Any]:
    """SC-1…SC-6 in one dict — what the monitors read and the artifact records."""
    r_cert = cert_radius(s_max, sigma_q, kappa_prime)
    d_safe = soft_d_safe(sep_expected, cfg.zeta)
    margin = cert_margin(sep_after, r_cert)
    defs = list(deficits) if deficits is not None else [margin["deficit_rel"]]
    return {
        "enabled": bool(cfg.enabled),
        "SC1": {"d_safe": d_safe, "zeta": float(cfg.zeta),
                "sep_expected": float(sep_expected), "R_cert": r_cert,
                "identified_with_R_cert": False,
                "note": ("the admission radius and the certificate radius are "
                         "TWO OBJECTS; the shipped harness identified them")},
        "SC2": margin,
        "SC3": budget_state(defs, cfg.budget_B),
        "SC4": {"c3_calibration": c3_calibration(
            c3_pairs or [], kappa=cfg.kappa, eta=cfg.eta, rho_band=cfg.rho_band,
            lambda_floor=cfg.lambda_floor, delta_num=cfg.delta_num)},
        "SC5": SC5_STATEMENT,
        "SC6": sc6_state([] if lambda_mins is None else lambda_mins,
                         sigma_q=sigma_q, capture=capture,
                         lambda_floor=cfg.lambda_floor),
        "SC7": SC7_FALSIFIER,
        "price": ("rho_ex up to 6.3x at a lambda_min cost of 2.2-6.0x, and the "
                  "dividend in that region stays ~0 (+0.0043 ... -0.0067): a "
                  "PRECONDITION, not a result"),
    }


__all__ = [
    "SC1_ZETA", "SC3_BUDGET_B", "BUDGET_DOMAIN", "C3_KAPPA", "C3_ETA",
    "C3_RHO_BAND", "C3_DELTA_NUM", "SC6_LAMBDA_FLOOR", "SC5_STATEMENT",
    "SC7_FALSIFIER", "SoftCertificateConfig", "expected_separation",
    "soft_d_safe", "cert_radius", "cert_margin", "budget_state",
    "c3_calibration", "capture_radius", "sc6_state", "soft_certificate_report",
    # K9 — the re-registered merge criterion (a predicate; NOT a merge verb)
    "MERGE_RHO_GEOM", "MERGE_TAU_PAYLOAD", "MERGE_PAYLOAD_DEGENERATE",
    "MergeCriterionConfig", "payload_scale_from_pairs", "merge_admissible",
    "merge_criterion_report",
]
