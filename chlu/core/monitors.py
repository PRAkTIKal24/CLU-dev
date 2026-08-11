"""The 13 anti-collapse monitors — **guards, never losses** (C2W1).

Twenty-six waves produced a map of the *specific* ways a full CLU collapses
(``advisor-head-intervention.md`` §5). This module turns each measured failure
mode into a **runtime-computable invariant with a trip predicate**, so the mode
fails LOUDLY instead of silently improving a metric.

**API FREEZE (C2W1).** ``memory-gym-v0`` and ``trainability-spike`` branch off
this surface. The public objects are :class:`MonitorReading`, :class:`Monitor`,
:class:`MonitorContext`, :class:`MonitorRegistry`, :func:`default_registry` and
the thirteen concrete monitor classes.

Two rules that are not negotiable and are the reason this file exists:

1. **No monitor quantity may enter any objective.** Modes #6 and #8 are *caused*
   by optimising against a proxy the objective can satisfy without the goal
   (`controller-doctrine` §6). Monitors observe; the controller acts through
   designed verbs; nothing here is differentiated.
2. **A monitor that never fires on any configuration you ran is UNTESTED, not
   green.** :meth:`MonitorRegistry.summary` reports ``untested`` explicitly.

The table implemented here is `controller-doctrine`'s (2026-07-29), which
supersedes the Hub's provisional table in the task file. The diff it applies,
carried here so the code and the doctrine cannot drift apart:

===  ==========================================================================
 #   change from the provisional table
===  ==========================================================================
 1   **REPLACED**. ``corr(q*, q_launch) > 0.90`` false-trips on a healthy store
     (measured 0.973-0.978 healthy vs 0.993-1.000 unconverged: everything is
     above the threshold). Trip on the *residual* ``rho_conv`` and the
     *displacement* ``delta`` instead; ``corr`` is reported, never tripped.
 2   **SHARPENED**. The inline launder needs labels => not runtime. The runtime
     form is the label-free disagreement mass ``D``, the uncovered mass ``U``
     and the exploitation ratio ``rho_ex = D/U`` (Prop D1: ``D <= U``).
     ``U < u_floor`` makes the monitor **inapplicable**, not passing.
 3   **SHARPENED**. Fire-rate alone is necessary-not-sufficient (N74: a gate can
     fire and certify nothing). Adds the validity leg and packing utilisation.
 4   confirmed; chance comes from the empirical marginal, not ``1/K``.
 5   confirmed (self-probe, label-free).
 6   **SHARPENED**: retrieval leg = the self-probe acquisition rate. **+ the
     TWO-SIDED DEAD-BAND** — ``slope_loss < -eps and slope_acq <= +eps_acq``,
     ``eps = 1e-9 * max|loss|``, ``eps_acq = 1e-9 * max|acq|``. The loss half
     landed in C2W2 (gym R2): 31 of the monitor's 58 first-ever trips fired at
     ``slope_write_loss = -5.2e-17`` — a monitor that trips on the
     floating-point floor is a monitor that gets disabled. **The acq half
     landed in C2W4**: it is the same band on the other leg, and without it a
     ``slope_acq = +1e-17`` counted as "acquisition is rising" and *suppressed*
     a genuine trip (a false NEGATIVE). One leg with a dead-band and one
     without is a half-repair that moves the trip count in one direction only.
 7   **SHARPENED (scope)**: a ``pytest`` gauge over the *trajectory*, and the
     gauge is **Newtonian-only** (relativistic breaks as O(1/c^2)). C2W2:
     :data:`GAUGE_SCOPE` names the scope per ``kinetic_mode`` and ``pytest``
     is parameterised over all three.
 8   **SHARPENED**: ``kappa = 5`` indexes ``spacing/sigma``, not ``margin/sigma``
     (99% needs ``margin >= 2.576 sigma``); ``delta_read`` is basin-conditioned.
 9   **REPLACED**: ``|corr(retention, |a|)| > 0.30`` trips at *every* excursion
     tested. Trip on the retention **effect size** instead; corr is a direction
     indicator only.
10   **SHARPENED**: an O(1) plumbing tier (declared-but-never-read config field)
     plus the expensive semantic sweep. **C2W2 REPAIR: tier (a) is now
     IMPLEMENTED** — :class:`ConfigAccessProxy` + :func:`assert_knobs_live`
     fail at startup on a declared-but-never-read field (doctrine I-8; C1W27
     declared ``knob_tier_a_implemented: false`` rather than silently passing).
11   confirmed — the saddle criterion on ``L = sqrt(|c|^2 + a^2)``, zero free
     parameters.
12   **SHARPENED**: C3 ratio on the k nearest items, plus allocation fairness
     and oldest-item self-probe retention.
13   confirmed — a **provenance field, not a trip**.
===  ==========================================================================

Plus **M14 guard-liveness** (`controller-doctrine` §6): mode #3 applied to the
controller itself — on a canary stream constructed to require intervention,
every designed guard must fire at least once.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Trip severity classes — `controller-doctrine` §5 (P3, trigger ordering).
# Ordering is by TRIP-IMPLICATION: if X's trip is implied by Y's, Y acts first.
# --------------------------------------------------------------------------
#: I = instrument validity. A reading taken while one of these is tripped
#: carries ZERO information about the store, so **no memory-mutating verb may
#: fire in the same step** (axiom A1 + consequence 1).
CLASS_I = ("blank", "dead_axis", "maturity", "mass_gauge", "gate_validity")
#: II = structural integrity (admission/reach/starvation/certificates).
CLASS_II = ("vacuous_gate", "reach", "starvation", "certificates")
#: III = dynamics regime (settle convergence, addressing, objective).
CLASS_III = ("overdamping", "addressing", "objective_divergence")
#: IV = policy/economics. #2 **escalates, never acts** (a controller able to act
#: on the dividend would learn to suppress the settle — w20 one level up).
CLASS_IV = ("lifetimes", "settle_argmin")
#: V = the only irreversible verbs.
CLASS_V = ("eviction", "deletion")

#: name -> trigger-ordering class (the controller consults this before acting).
SEVERITY = {
    "blank": "I", "dead_axis": "I", "maturity": "I", "mass_gauge": "I",
    "vacuous_gate": "II", "reach": "II", "starvation": "II",
    "certificates": "II",
    "overdamping": "III", "addressing": "III", "objective_divergence": "III",
    "lifetimes": "IV", "settle_argmin": "IV",
    "guard_liveness": "I",
    # C2W7 (charter §A21): the learned launch head's codebook-collapse row.
    "launch_collapse": "I",
    # --- BEGIN c2w10-lifecycle (additive: one row, class II) ---
    # C2W10 L4: the protected fraction hit its bound and promotions are being
    # REFUSED. Structural (it changes what the store will admit into protection),
    # and it fails loudly at runtime — it is never a loss term. The monitor
    # itself lives in `chlu.core.store_lifecycle.ProtectedSaturationMonitor` and
    # is attached through `MonitorRegistry.register`.
    "protected_saturation": "II",
    # --- END c2w10-lifecycle ---
}


@dataclass(frozen=True)
class MonitorReading:
    """One observation of one monitor.

    Attributes:
        name: monitor name (stable string key).
        mode: which of the 13 anti-collapse modes this is (1..13; 14 = M14).
        value: the scalar invariant (``nan`` when inapplicable).
        band: human-readable productive band, with provenance.
        tripped: did the trip predicate fire.
        cost_ms: wall-clock cost of this observation.
        applicable: ``False`` marks *inapplicable*, which is NOT passing
            (monitor #2 at ``U < u_floor`` is the canonical case: 0/0).
        severity_class: ``"I".."V"`` — the trigger-ordering class.
        lever: which lever a trip is attributed to (for the trip log).
        verb: the restoring verb the controller should consider.
        stage: harness stage label at observation time.
        detail: everything else, including the reported-but-never-tripped
            diagnostics (e.g. ``corr(q*, q0)`` for mode 1).
        provenance: maturity fields (mode 13): epochs/write-steps/wall-clock.
    """

    name: str
    mode: int
    value: float
    band: str
    tripped: bool
    cost_ms: float = 0.0
    applicable: bool = True
    severity_class: str = "III"
    lever: str = ""
    verb: str = ""
    stage: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-safe dict (the reported artifact)."""
        return {
            "name": self.name,
            "mode": self.mode,
            "value": _jsonable(self.value),
            "band": self.band,
            "tripped": bool(self.tripped),
            "applicable": bool(self.applicable),
            "severity_class": self.severity_class,
            "cost_ms": round(float(self.cost_ms), 4),
            "lever": self.lever,
            "verb": self.verb,
            "stage": self.stage,
            "detail": {k: _jsonable(v) for k, v in self.detail.items()},
            "provenance": {k: _jsonable(v) for k, v in self.provenance.items()},
        }


def _jsonable(x):
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if math.isnan(v) else v
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


@dataclass
class MonitorContext:
    """Everything a monitor may look at. Filled by the harness, never by a loss.

    Fields are all optional: a monitor whose inputs are absent returns an
    **inapplicable** reading rather than crashing, so a partial stage can still
    run the registry.
    """

    stage: str = ""
    t: int = 0
    system: Any = None  # CluSystem (avoid the import cycle)
    reads: Optional[dict] = None  # batched read diagnostics
    self_probe: Optional[dict] = None  # label-free store-re-reads-itself pass
    blank: Optional[dict] = None  # blank/empty-store control read
    write_log: Optional[Sequence[dict]] = None
    controller: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def get(self, section: str, key: str, default=None):
        """``ctx.get("reads", "rho_conv")`` with None-safe sections."""
        sec = getattr(self, section, None)
        if sec is None:
            return default
        return sec.get(key, default)


class Monitor(Protocol):
    """The monitor protocol: ``observe(ctx) -> MonitorReading``."""

    name: str
    mode: int

    def observe(self, ctx: MonitorContext) -> MonitorReading:  # pragma: no cover
        ...


@dataclass
class TripRecord:
    """A trip, timestamped and attributed (the reported artifact)."""

    wall_clock: float
    t: int
    stage: str
    reading: MonitorReading


class MonitorRegistry:
    """Runs every monitor, logs trips loudly, and reports what was never tested.

    Args:
        monitors: the monitors to run.
        loud: when ``True`` (default) each trip is printed the moment it fires.
        logger: optional ``str -> None`` sink for the loud line.
    """

    def __init__(
        self,
        monitors: Sequence[Monitor],
        loud: bool = True,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.monitors: List[Monitor] = list(monitors)
        self.loud = bool(loud)
        self.logger = logger or print
        self.readings: List[MonitorReading] = []
        self._trips: List[TripRecord] = []
        self._last: List[MonitorReading] = []

    def register(self, monitor: Monitor) -> None:
        """Add a monitor after construction."""
        self.monitors.append(monitor)

    def observe(self, ctx: MonitorContext) -> List[MonitorReading]:
        """Run every monitor once; log and return the readings."""
        out: List[MonitorReading] = []
        for m in self.monitors:
            t0 = time.perf_counter()
            try:
                r = m.observe(ctx)
            except Exception as exc:  # a broken monitor is loud, never silent
                r = MonitorReading(
                    name=getattr(m, "name", type(m).__name__),
                    mode=getattr(m, "mode", -1),
                    value=float("nan"), band="n/a", tripped=False,
                    applicable=False, detail={"error": repr(exc)},
                )
            r = replace(
                r,
                cost_ms=(time.perf_counter() - t0) * 1e3,
                stage=ctx.stage,
                severity_class=SEVERITY.get(r.name, r.severity_class),
            )
            out.append(r)
            self.readings.append(r)
            if r.tripped:
                rec = TripRecord(time.time(), ctx.t, ctx.stage, r)
                self._trips.append(rec)
                if self.loud:
                    self.logger(
                        f"⛔ MONITOR TRIP [#{r.mode} {r.name}] stage={ctx.stage!r} "
                        f"t={ctx.t} value={r.value:.6g} band={r.band} "
                        f"lever={r.lever!r} verb={r.verb!r}"
                    )
        self._last = out
        return out

    @property
    def trips(self) -> List[TripRecord]:
        """Every trip so far, in order."""
        return list(self._trips)

    def class_i_tripped(self, window: int = 1) -> List[str]:
        """Names of class-I monitors tripped in the last ``window`` observations.

        The controller consults this before any memory-mutating verb
        (`controller-doctrine` §5, consequence 1: ``evict`` may not fire in the
        same step as a class-I trip).
        """
        names: List[str] = []
        n_mon = max(1, len(self.monitors))
        tail = self.readings[-window * n_mon:] if self.readings else []
        for r in tail:
            if r.tripped and SEVERITY.get(r.name) == "I":
                names.append(r.name)
        return sorted(set(names))

    def summary(self) -> Dict[str, dict]:
        """Per-monitor: observations, trips, ever-tripped, and **untested**.

        ``untested = (n_observations == 0) or (never applicable)``. A monitor
        that never fired on any configuration is labelled untested, never green.
        """
        out: Dict[str, dict] = {}
        for m in self.monitors:
            name = m.name
            rs = [r for r in self.readings if r.name == name]
            applic = [r for r in rs if r.applicable]
            trips = [r for r in rs if r.tripped]
            out[name] = {
                "mode": getattr(m, "mode", -1),
                "severity_class": SEVERITY.get(name, "III"),
                "n_observations": len(rs),
                "n_applicable": len(applic),
                "n_trips": len(trips),
                "ever_tripped": bool(trips),
                # a monitor that never fired on any configuration is UNTESTED,
                # not green (task file, acceptance criterion).
                "never_trips_by_design": bool(getattr(m, "never_trips", False)),
                "untested": (not getattr(m, "never_trips", False))
                and ((len(applic) == 0) or (len(trips) == 0)),
                "stages_tripped": sorted({r.stage for r in trips}),
                "last_value": (rs[-1].value if rs else float("nan")),
                "band": (rs[-1].band if rs else ""),
            }
        return out

    def to_markdown(self) -> str:
        """The trip-state table (a reported artifact of every run)."""
        rows = [
            "| # | monitor | class | obs | applic | trips | state | stages tripped |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, s in sorted(self.summary().items(), key=lambda kv: kv[1]["mode"]):
            if s.get("never_trips_by_design"):
                state = "n/a by design (not a runtime trip)"
            elif s["n_applicable"] == 0:
                state = "⚠ UNTESTED (never applicable)"
            elif s["n_trips"] == 0:
                state = "⚠ UNTESTED (never fired)"
            else:
                state = "⛔ TRIPPED"
            rows.append(
                f"| {s['mode']} | {name} | {s['severity_class']} | "
                f"{s['n_observations']} | {s['n_applicable']} | {s['n_trips']} | "
                f"{state} | {', '.join(s['stages_tripped']) or '—'} |"
            )
        return "\n".join(rows)


# ==========================================================================
# The thirteen. Each carries its band + provenance in the docstring, and its
# FALSE-TRIP MODE (the benign situation that fires it) — an uncharacterised
# monitor gets disabled by the next engineer, and then it is not a guard.
# ==========================================================================


def _inapplicable(name, mode, band, why, verb="", lever="") -> MonitorReading:
    return MonitorReading(
        name=name, mode=mode, value=float("nan"), band=band, tripped=False,
        applicable=False, verb=verb, lever=lever, detail={"why": why},
    )


class OverdampingMonitor:
    """#1 overdamping -> "the last observation" (`controller-doctrine` row 1).

    Invariant: ``rho_conv = med|grad V(q*)| / med|grad V(q0)|`` (did the settle
    actually settle) and ``delta = med|q* - q0| / sep`` (did it actually move).
    **Trip if ``rho_conv > rho_max`` OR ``delta < delta_min``.**
    ``corr(q*, q0)`` is REPORTED, never tripped — it measures 0.973-0.978 on a
    healthy store and 0.993-1.000 on an unconverged one, i.e. the Hub's
    provisional ``> 0.90`` predicate fires on both.

    Band: ``gamma*N`` such that ``rho_conv < 1e-6``; measured ``gamma in
    [0.05, 0.5]`` at ``N=400, eps=0.05``.  Verb: ``anneal`` / ``retry`` / ``stop``.

    False-trip mode: a store whose items sit at very flat minima has small
    ``|grad V(q0)|`` so the ratio is noise. Guarded by flooring the denominator.
    """

    name = "overdamping"
    mode = 1
    false_trip = "flat minima => tiny |grad V(q0)| => rho_conv is noise (denominator floored)"

    def __init__(self, rho_max: float = 1e-6, delta_min: float = 0.02,
                 grad_floor: float = 1e-9):
        self.rho_max = float(rho_max)
        self.delta_min = float(delta_min)
        self.grad_floor = float(grad_floor)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        g0 = ctx.get("reads", "grad_norm_q0")
        gs = ctx.get("reads", "grad_norm_qstar")
        disp = ctx.get("reads", "displacement")
        sep = ctx.extras.get("sep", float("nan"))
        band = f"rho_conv <= {self.rho_max:g} and delta >= {self.delta_min:g} (doctrine S2b)"
        if g0 is None or gs is None:
            return _inapplicable(self.name, self.mode, band, "no read diagnostics")
        g0 = np.asarray(g0, dtype=float)
        gs = np.asarray(gs, dtype=float)
        den = max(float(np.median(g0)), self.grad_floor)
        rho = float(np.median(gs)) / den
        delta = (
            float(np.median(np.asarray(disp, dtype=float)) / sep)
            if disp is not None and np.isfinite(sep) and sep > 0
            else float("nan")
        )
        tripped = bool(rho > self.rho_max) or bool(
            np.isfinite(delta) and delta < self.delta_min
        )
        return MonitorReading(
            name=self.name, mode=self.mode, value=rho, band=band, tripped=tripped,
            lever="gamma/steps", verb="anneal|retry|stop",
            detail={
                "rho_conv": rho, "delta": delta,
                # REPORTED, never tripped (the provisional predicate's statistic)
                "corr_q0_qstar": ctx.get("reads", "corr_q0_qstar", float("nan")),
                "grad_norm_q0_med": float(np.median(g0)),
                "grad_norm_qstar_med": float(np.median(gs)),
                "false_trip_mode": self.false_trip,
            },
        )


class SettleArgminMonitor:
    """#2 settle -> arg-min. **This monitor IS the dividend** (charter §2.1).

    Runtime, label-free (Prop D1/D2):

    * ``D  = P_q[argmin_i|q* - c_i| != argmin_i|q0 - c_i|]`` — the settle/arg-min
      disagreement mass, i.e. the only place a dividend can come from;
    * ``U  = P_q[q0 not in union ball(c_i, r_i)]`` — the uncovered mass;
    * ``rho_ex = D / U`` — exploitation.

    **Trip if ``rho_ex < rho_min``** (default 0.10). **Inapplicable (not passing)
    when ``U < u_floor``** — with an excellent phi all mass lands inside the
    certified balls and the ratio is 0/0.

    Prop D1 gives ``D <= U`` and ``D = 0 => dividend <= 0``; Corollary D2a gives
    ``D = 0`` *exactly* for equal-depth symmetric stores — which is why w26's
    same-keys launder beat CLU 6/6 structurally, not accidentally.

    ⭐ **This monitor has NO restoring verb and must never fire one.** Its fix is
    a *configuration* change (heterogeneity, non-metric-native queries, a
    trajectory read). A controller able to act on it would learn to act by
    suppressing the settle.
    """

    name = "settle_argmin"
    mode = 2
    false_trip = "a genuinely tight query law puts all mass inside the balls => U -> 0 => 0/0"

    def __init__(self, rho_min: float = 0.10, u_floor: float = 0.01):
        self.rho_min = float(rho_min)
        self.u_floor = float(u_floor)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = f"rho_ex >= {self.rho_min:g}; D <= U by Prop D1; INAPPLICABLE if U < {self.u_floor:g}"
        a_settle = ctx.get("reads", "assign_settle")
        a_argmin = ctx.get("reads", "assign_argmin")
        covered = ctx.get("reads", "covered")
        if a_settle is None or a_argmin is None or covered is None:
            return _inapplicable(self.name, self.mode, band, "no assignment diagnostics",
                                 verb="ESCALATE (no verb by design)")
        a_settle = np.asarray(a_settle)
        a_argmin = np.asarray(a_argmin)
        covered = np.asarray(covered, dtype=bool)
        D = float(np.mean(a_settle != a_argmin))
        U = float(np.mean(~covered))
        detail = {
            "D": D, "U": U, "prop_D1_holds": bool(D <= U + 1e-12),
            "false_trip_mode": self.false_trip,
            "note": "no restoring verb: this monitor ESCALATES, never acts",
        }
        if U < self.u_floor:
            r = _inapplicable(self.name, self.mode, band,
                              f"U={U:.4f} < u_floor => rho_ex is 0/0",
                              verb="ESCALATE (no verb by design)")
            return replace(r, detail={**r.detail, **detail})
        rho_ex = D / U
        return MonitorReading(
            name=self.name, mode=self.mode, value=rho_ex, band=band,
            tripped=bool(rho_ex < self.rho_min),
            lever="store geometry / query law (NOT a controller verb)",
            verb="ESCALATE (no verb by design)",
            detail={**detail, "rho_ex": rho_ex},
        )


class VacuousGateMonitor:
    """#3 vacuous gate (N74: spacing 1.4142 vs ``d_safe`` 1.10 => the gate could
    not fire arithmetically; N91: the *address space* was binding, not the gate).

    Legs: (i) fire-rate ``f`` over the stream — **trip if f in {0, 1}**;
    ⭐ **(ii′) the C3 first-order calibration leg** (C2W3 D3, see below);
    (iii) packing utilisation ``n_live / N_pack`` — trip above 0.95;
    ⭐ **(iv) the SC-3 violation budget** when the soft certificate is on.

    ⛔ **C2W3 D3 — leg (ii) is RETIRED as a correlation and REPLACED.** The
    shipped leg was ``validity = -corr(gate_margin, post_write_drift)``, tripping
    below 0.30. It is **sign-unstable on a learned V_theta** for four named
    causes (`doctrine-repairs.md` §1.1): (1) 18 of 28 gym cells have
    ``lambda_min <= 0``, where "drift of the fixed point" has no first-order
    theory at all; (2) the ``(A_j, s_j)`` heterogeneity a learned write produces
    moves ``||grad dV||`` by **1823x** at fixed distance, and none of it enters a
    pure distance margin; (3) ``d exp(-d^2/2s^2)`` peaks at ``d = s`` — closer is
    not monotonically worse; (4) a pairing bug — the margin is logged *before*
    relocation while the drift is caused by the site written *after*.
    Its validity domain is ``d/s >~ 4``; the gym runs at ``d/s ~ 1.9-2``.

    The replacement bounds the same harm **measured**: with
    ``B_ij = ||grad dV_j(q_i*)|| / lambda_min,i`` (predicted, first order) and
    ``Delta_ij`` measured at the relaxed fixed point, trip iff
    ``rho_C3 = median(Delta/B)`` leaves ``[1/3, 3]`` or ``P[Delta > 3B] > 0.10``;
    **INAPPLICABLE** below 3 qualifying pairs. Head-to-head, 12 seeds, gym-like:
    shipped **+0.412** mean with **1/12 sign flips**, replacement spearman
    **+0.914** with **0/12** — **at zero extra cost**, since
    ``CluSystem._c3_check`` already computes every term.
    **The old leg survives as a REPORTED diagnostic only and may not trip.**

    Verb: ``admit`` (recalibrate) / ``expand`` / ``stop``.
    False-trip mode: a stream of genuinely well-separated proposals gives
    ``f = 0`` legitimately — which is why leg (ii′) must agree before acting.
    """

    name = "vacuous_gate"
    mode = 3
    false_trip = "genuinely well-separated proposals give f=0 legitimately (leg ii must agree)"

    def __init__(self, validity_min: float = 0.30, utilisation_max: float = 0.95,
                 c3_leg: bool = True):
        self.validity_min = float(validity_min)
        self.utilisation_max = float(utilisation_max)
        #: ⛔ C2W3 D3. ``False`` restores the retired correlation leg as the
        #: tripping leg — kept only so the repair's own effect is measurable as a
        #: controlled diff, never as a recommended configuration.
        self.c3_leg = bool(c3_leg)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = (
            "fire-rate f in (0,1) strictly; "
            + ("C3 calibration rho_C3 in [1/3,3] and P[Delta>3B] <= 0.10 "
               "(C2W3 D3; the corr leg is RETIRED to a diagnostic)"
               if self.c3_leg else
               f"validity corr >= {self.validity_min:g} (RETIRED leg, forced on)")
            + f"; n_live/N_pack <= {self.utilisation_max:g} (N74/N91)"
            + "; SC-3 deficit_rel <= B when the soft certificate is on")
        log = list(ctx.write_log or [])
        offers = [r for r in log if r.get("decision") in
                  ("admit", "relocate", "refuse_spacing", "refuse_full", "refuse_reach")]
        if not offers:
            return _inapplicable(self.name, self.mode, band, "no offers yet",
                                 verb="admit|expand|stop")
        refused = [r for r in offers if str(r.get("decision", "")).startswith("refuse")]
        f = len(refused) / len(offers)
        util = float(ctx.extras.get("utilisation", float("nan")))
        # validity leg: does the gate margin predict the measured post-write drift?
        pairs = [(r.get("gate_margin"), r.get("post_write_drift")) for r in log
                 if r.get("gate_margin") is not None
                 and r.get("post_write_drift") is not None]
        pairs = [(a, b) for a, b in pairs
                 if np.isfinite(float(a)) and np.isfinite(float(b))]
        validity = float("nan")
        if len(pairs) >= 3:
            m = np.asarray([a for a, _ in pairs], dtype=float)
            d = np.asarray([b for _, b in pairs], dtype=float)
            if np.std(m) > 1e-12 and np.std(d) > 1e-12:
                # a valid certificate predicts LESS drift at LARGER margin
                validity = float(-np.corrcoef(m, d)[0, 1])
        # -- ⭐ leg (ii′): the C3 first-order calibration test (C2W3 D3) --------
        from chlu.core.soft_certificate import c3_calibration

        soft = ctx.extras.get("soft_certificate") or {}
        sc_kwargs = {}
        if soft:
            sc1 = soft.get("SC4", {}).get("c3_calibration", {})
            sc_kwargs = {k: sc1[k] for k in ("kappa", "eta", "lambda_floor",
                                             "delta_num") if k in sc1}
            if "rho_band" in sc1:
                sc_kwargs["rho_band"] = tuple(sc1["rho_band"])
        c3 = c3_calibration(ctx.extras.get("c3_pairs") or [], **sc_kwargs)
        # -- ⭐ leg (iv): SC-3's violation budget (soft certificate only) -------
        budget = soft.get("SC3") if soft else None

        tripped = bool(f <= 0.0 or f >= 1.0)
        if self.c3_leg:
            tripped = tripped or bool(c3.get("applicable") and c3.get("tripped"))
        else:  # the retired path, kept only as a controlled diff
            tripped = tripped or bool(np.isfinite(validity)
                                      and validity < self.validity_min)
        tripped = tripped or bool(np.isfinite(util) and util > self.utilisation_max)
        # ⛔ SC-3: exceeding the budget is a TRIP, never a refusal — a soft
        # constraint that refuses is a hard constraint with extra steps.
        if budget and budget.get("applicable") and not budget.get("within_budget"):
            tripped = True
        return MonitorReading(
            name=self.name, mode=self.mode, value=f, band=band, tripped=tripped,
            lever="d_safe / address-space volume", verb="admit|expand|stop",
            detail={"fire_rate": f, "n_offers": len(offers), "n_refused": len(refused),
                    # the RETIRED leg, reported and never tripping (C2W3 D3)
                    "validity_corr_RETIRED_DIAGNOSTIC": validity,
                    "validity_corr_domain": ("valid only for d/s >~ 4; the gym runs "
                                             "at d/s ~ 1.9-2"),
                    "c3_calibration_leg": c3,
                    "soft_certificate_budget": budget,
                    "utilisation": util,
                    "false_trip_mode": self.false_trip},
        )


class BlankControlMonitor:
    """#4 blank controls passing (N68: blanks 0.992-1.000; a 1e-4 address leak
    makes classification perfect).

    Blank/empty-store read on the **strongest read in use**, scored by decode
    (never ``tol``: N110 — the ``tol`` metric is vacuous at m>1).
    **Trip if ``acc_blank >= chance + 3*se``**, with chance taken from the
    **empirical marginal**, not ``1/K``.

    ⚠ With a trajectory read the blank must be run through the *same* psi on the
    *same* representation, because the trajectory **contains ``q0 = phi(x)``**:
    a blank-store psi read is then exactly "a classifier on phi(x)".
    Verb: ``stop`` — a blank-passing instrument invalidates every other reading.
    """

    name = "blank"
    mode = 4
    false_trip = "a skewed marginal label distribution makes 1/K the wrong bar (chance is empirical)"

    def __init__(self, sigma_mult: float = 3.0):
        self.sigma_mult = float(sigma_mult)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = f"acc_blank < chance + {self.sigma_mult:g}*se, chance = empirical marginal (N68/N110)"
        if ctx.blank is None or "score" not in ctx.blank:
            return _inapplicable(self.name, self.mode, band, "no blank control run",
                                 verb="stop")
        score = float(ctx.blank["score"])
        chance = float(ctx.blank.get("chance", float("nan")))
        se = float(ctx.blank.get("se", 0.0))
        bar = chance + self.sigma_mult * se
        tripped = bool(np.isfinite(bar) and score >= bar)
        return MonitorReading(
            name=self.name, mode=self.mode, value=score, band=band, tripped=tripped,
            lever="read-out / instrument", verb="stop",
            detail={"score": score, "chance": chance, "se": se, "bar": bar,
                    "metric": ctx.blank.get("metric", "decode"),
                    "representation": ctx.blank.get("representation", "settled_point"),
                    "false_trip_mode": self.false_trip},
        )


class AddressingMonitor:
    """#5 learned addressing dies (w19: 0/18 = 4.2% = chance).

    **Self-probe** (label-free: the store knows what it wrote) acquisition rate
    ``acq = P[basin(relax(phi(x_i))) == i]`` over live items. **Trip if
    ``acq < acq_min``** (default 0.90; chance is ``1/K``). ``basin`` and
    ``strict`` are reported separately — w26 showed the annealed-read gain *is*
    address acquisition (``readonly`` reproduces the baseline to 4 dp).

    Verb: ``anneal`` -> ``route`` -> ``place``.
    False-trip mode: an item deliberately admitted-then-decayed self-probes
    badly and that is correct behaviour — items below the decay floor are
    excluded.
    """

    name = "addressing"
    mode = 5
    false_trip = "an item deliberately decayed below the amp floor self-probes badly (excluded)"

    def __init__(self, acq_min: float = 0.90):
        self.acq_min = float(acq_min)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = f"acq >= {self.acq_min:g} (chance 1/K); w19 violation 0.042"
        acq = ctx.get("self_probe", "acq")
        if acq is None:
            return _inapplicable(self.name, self.mode, band, "no self-probe pass",
                                 verb="anneal|route|place")
        acq = float(acq)
        return MonitorReading(
            name=self.name, mode=self.mode, value=acq, band=band,
            tripped=bool(acq < self.acq_min),
            lever="addressing / read schedule", verb="anneal|route|place",
            detail={"acq_basin": acq,
                    "acq_strict": ctx.get("self_probe", "strict", float("nan")),
                    "chance": ctx.get("self_probe", "chance", float("nan")),
                    "n_probed": ctx.get("self_probe", "n_probed", 0),
                    "false_trip_mode": self.false_trip},
        )


class ObjectiveDivergenceMonitor:
    """#6 objective/goal divergence (w25/w26: write loss -> 0 while retrieval
    fails — the objective stops seeing crowding).

    Rolling window over ``slope(write-loss)`` and ``slope(acq)`` from #5.
    **Trip if the loss falls while acquisition is flat/declining for ``window``
    consecutive windows.** Leading indicator (curriculum-blind):
    ``min_i (sep_i - 2 s_i)`` trending down while the loss falls.
    Verb: ``stop`` the write objective -> ``consolidate`` -> resume.

    ⭐ **THE TWO-SIDED DEAD-BAND (C2W2 loss half + C2W4 acq half).** The
    originally shipped predicate was ``slope_loss < 0 and slope_acq <= 0``, with
    **no dead-band on either leg**, so a converged write whose loss is flat to
    round-off trips: **31 of the monitor's 58 first-ever trips fired at
    ``slope_write_loss = -5.2e-17``, ``slope_acq = -5.9e-17``** — over half its
    trips were an epsilon artefact. The predicate is now

        ``slope_loss < -eps and slope_acq <= +eps_acq``

    with ``eps = eps_rel * scale``, ``scale = max(|loss|)`` over the window, and
    symmetrically ``eps_acq = eps_acq_rel * scale_acq``,
    ``scale_acq = max(|acq|)`` over the window (both bands are relative to their
    own leg's magnitude, not to an absolute number that would be wrong the
    moment the quantity is rescaled), ``eps_rel = eps_acq_rel = 1e-9``.

    ⛔ **``eps_rel = 0.0`` restores the pre-C2W2 predicate exactly and
    ``eps_acq_rel = 0.0`` restores the C2W2-C2W3 (loss-half-only) predicate
    exactly** — that is how the before/after re-scores are done without
    re-running the store, and it is why the repair is auditable: a repair that
    cannot be turned off is not auditable.

    **Why the acq half is load-bearing (C2W4, `bprime-fb4-gate` R4).** The loss
    half only ever *removes* trips (false positives). The missing acq half is
    the other error: ``slope_acq <= 0.0`` means a ``+1e-17`` round-off slope
    counts as "acquisition is rising" and **suppresses** a genuine trip — a
    false NEGATIVE. Landing one leg and not the other moves the trip count in
    one direction only, which is not a repair.

    ⚠ **This band is a ROUND-OFF floor, not a RESOLUTION floor.** Both legs are
    ``1e-9 * (the leg's own window scale)``, i.e. ~7 orders above the float64
    ulp and ~6 orders below any real slope in the C2W1 record. `doctrine-repairs`
    §2.3 additionally proposed a *resolution* floor on the acq leg
    (``1/(n_probed * window)`` ~ 4e-2, "a slope whose extrapolated change over
    the window is below the quantity's own quantum is not a measurement") and
    predicted 2 recovered false negatives from it. That is a **larger and
    different** claim — it would trip on genuine sub-quantum acquisition slopes
    — and it is deliberately NOT shipped here: `eps_acq_rel` is the knob a
    future task would raise to test it, and the C2W4 re-score reports the flip
    count as a function of the band (see `.claude/outputs/harness-debt.md`).

    A "loss fell by 1e-17" is not divergence; it is the floating-point floor.
    And an "acquisition rose by 1e-17" is not acquisition.
    """

    name = "objective_divergence"
    mode = 6
    false_trip = ("a curriculum change (harder items) legitimately flattens retrieval "
                  "as loss falls; and (pre-C2W2) a converged write whose slope is "
                  "numerically zero — the dead-band closes the second one")

    def __init__(self, window: int = 3, eps_rel: float = 1e-9,
                 eps_floor: float = 1e-30, eps_acq_rel: float = 1e-9,
                 eps_acq_floor: float = 1e-30):
        self.window = int(window)
        self.eps_rel = float(eps_rel)
        self.eps_floor = float(eps_floor)
        self.eps_acq_rel = float(eps_acq_rel)
        self.eps_acq_floor = float(eps_acq_floor)
        self._hist: List[tuple] = []

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = (f"sign agreement over {self.window} windows (w25/w26 violation), "
                f"dead-band eps = {self.eps_rel:g} * max|loss| (gym R2), "
                f"eps_acq = {self.eps_acq_rel:g} * max|acq| (fb4 R4)")
        loss = ctx.get("self_probe", "write_loss")
        acq = ctx.get("self_probe", "acq")
        if loss is None or acq is None:
            return _inapplicable(self.name, self.mode, band, "no (loss, acq) pair",
                                 verb="stop|consolidate")
        self._hist.append((float(loss), float(acq)))
        n = self.window + 1
        if len(self._hist) < n:
            return _inapplicable(self.name, self.mode, band,
                                 f"history {len(self._hist)}/{n}", verb="stop|consolidate")
        h = np.asarray(self._hist[-n:], dtype=float)
        x = np.arange(n, dtype=float)
        slope_loss = float(np.polyfit(x, h[:, 0], 1)[0])
        slope_acq = float(np.polyfit(x, h[:, 1], 1)[0])
        scale = float(np.max(np.abs(h[:, 0])))
        scale_acq = float(np.max(np.abs(h[:, 1])))
        eps = max(self.eps_rel * scale, self.eps_floor if self.eps_rel > 0 else 0.0)
        eps_acq = max(self.eps_acq_rel * scale_acq,
                      self.eps_acq_floor if self.eps_acq_rel > 0 else 0.0)
        tripped = objective_divergence_predicate(slope_loss, slope_acq, eps, eps_acq)
        return MonitorReading(
            name=self.name, mode=self.mode, value=slope_acq, band=band, tripped=tripped,
            lever="write objective", verb="stop|consolidate",
            detail={"slope_write_loss": slope_loss, "slope_acq": slope_acq,
                    "eps_dead_band": eps, "loss_scale": scale,
                    "eps_rel": self.eps_rel,
                    "eps_acq_dead_band": eps_acq, "acq_scale": scale_acq,
                    "eps_acq_rel": self.eps_acq_rel,
                    # the pre-repair predicate, carried so a trip-state diff can
                    # be re-scored offline from the artifact (gym R2)
                    "tripped_pre_repair": objective_divergence_predicate(
                        slope_loss, slope_acq, 0.0, 0.0),
                    # ...and the loss-half-only predicate (the C2W2-C2W3 shipped
                    # state, which produced the published "27"), so the C2W4
                    # acq-half diff is re-scorable offline too (fb4 R4)
                    "tripped_loss_half_only": objective_divergence_predicate(
                        slope_loss, slope_acq, eps, 0.0),
                    "leading_indicator_min_sep_minus_2s":
                        ctx.extras.get("min_sep_minus_2s", float("nan")),
                    "false_trip_mode": self.false_trip},
        )


class MassGaugeMonitor:
    """#7 mass stores nothing — **a gauge TEST, not a runtime monitor**.

    ``(M, V, p0) -> (lambda M, lambda V, lambda p0)`` must leave the whole
    **trajectory** invariant (endpoint-only comparison passes vacuously once
    both runs settle into the same minimum: measured 9.1e-2 -> 3.6e-3 by
    doubling the step budget alone).

    Scope, and it is new: the gauge is **Newtonian-only**. Under the
    relativistic kinetic term it breaks as O(1/c^2) — 9.1e-2 relative at the
    paper's ``c = 5``. So "mass may not be used as an information channel" holds
    in Newtonian mode; in relativistic mode N76 does **not** forbid it.

    At runtime this monitor reports ``applicable=False`` and points at
    :func:`gauge_orbit_residual`, which ``pytest`` asserts on.
    """

    name = "mass_gauge"
    mode = 7
    never_trips = True
    false_trip = "endpoint-only comparison passes vacuously once both runs settle (compare trajectories)"

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        r = _inapplicable(
            self.name, self.mode,
            "pytest gauge: trajectory-wise invariance, Newtonian exact; "
            "relativistic breaks as O(1/c^2) (SCOPE, not a pass)",
            "gauge test runs in pytest (test_monitors.py), not in the stream",
        )
        return replace(r, detail={**r.detail,
                                  "kinetic_mode": ctx.extras.get("kinetic_mode", "?"),
                                  "false_trip_mode": self.false_trip})


class CertificateMonitor:
    """#8 learning erases design (w20: free ``V_theta`` destroys designed
    structure). The C1-C5/N1-N4 certificates, re-checked each consolidation:

    * **N1** injectivity of item -> site;
    * **N2** ``sep / sigma_q >= 5.15`` — ⚠ ``kappa = 5`` indexes **spacing**/sigma,
      not margin/sigma: the law is ``acc ~ erf(margin / sqrt(2) sigma)`` and 99%
      needs ``margin >= 2.576 sigma`` (`controller-doctrine` R1);
    * **N3** ``lambda_min(Hess V(c_i)) > 0`` at every site;
    * **N4** ``min_{i!=j} |a_i - a_j| >= 2 delta_read``, with ``delta_read``
      measured **conditioned on correct basin assignment** (otherwise N4
      double-counts monitor #5's address errors).

    **Trip on any one.** Verb: ``place`` / ``admit`` / ``evict``.
    ⚠ On a **global-support** learned write the spacing certificate certifies
    nothing (N75) — then the only honest verb is ``stop`` + report.
    """

    name = "certificates"
    mode = 8
    false_trip = "a declared register/coset shares a site by design => N1 fires (exempt declared cosets)"

    def __init__(self, spacing_sigma_min: float = 5.15, n4_ratio_min: float = 2.0):
        self.spacing_sigma_min = float(spacing_sigma_min)
        self.n4_ratio_min = float(n4_ratio_min)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = (f"N1 injective; N2 sep/sigma_q >= {self.spacing_sigma_min:g}; "
                f"N3 lambda_min > 0; N4 gap/(2*delta_read) >= 1 (doctrine R1)")
        c = ctx.extras.get("certificates")
        if not c:
            return _inapplicable(self.name, self.mode, band, "no certificate pass",
                                 verb="place|admit|evict")
        n1 = bool(c.get("injective", True))
        sep_sigma = float(c.get("sep_over_sigma_q", float("nan")))
        lam_min = float(c.get("lambda_min", float("nan")))
        payload_gap = float(c.get("payload_gap", float("nan")))
        delta_read = float(c.get("delta_read_basin_conditioned", float("nan")))
        n4_ratio = (payload_gap / (2.0 * delta_read)
                    if np.isfinite(payload_gap) and np.isfinite(delta_read) and delta_read > 0
                    else float("nan"))
        fails = []
        if not n1:
            fails.append("N1")
        if np.isfinite(sep_sigma) and sep_sigma < self.spacing_sigma_min:
            fails.append("N2")
        if np.isfinite(lam_min) and lam_min <= 0.0:
            fails.append("N3")
        if np.isfinite(n4_ratio) and n4_ratio < 1.0:
            fails.append("N4")
        return MonitorReading(
            name=self.name, mode=self.mode, value=sep_sigma, band=band,
            tripped=bool(fails), lever="write operator / placement",
            verb="place|admit|evict (stop under a global-support write)",
            detail={"failed": fails, "N1_injective": n1, "N2_sep_over_sigma_q": sep_sigma,
                    "N3_lambda_min": lam_min, "N4_ratio": n4_ratio,
                    "payload_gap": payload_gap, "delta_read": delta_read,
                    "erf_accuracy_at_margin": c.get("erf_accuracy", float("nan")),
                    "false_trip_mode": self.false_trip},
        )


class LifetimeMonitor:
    """#9 payload-dependent lifetimes (w25: r = -0.85 — "lifetime is a dial you
    set" is false without a fix).

    **Effect size, not correlation**: ``Delta_ret = max_g ret_g - min_g ret_g``
    over ``|a|`` groups from the #5 self-probe pass. **Trip if
    ``Delta_ret > 0.10``.** ``corr(ret, |a|)`` is reported as a *direction*
    indicator only — it exceeds 0.30 at every excursion measured, so a
    correlation predicate is always on, and a monitor that is always on is a
    monitor that gets disabled.

    ⚠ Known-uncleanable-by-any-verb at large excursion: the fix is C1W27's
    option-(d) gated stiffness, which C2 must **not** build. A trip here at
    large excursion is a *reported scope*, not a harness failure.
    """

    name = "lifetimes"
    mode = 9
    false_trip = "items with deliberately different leak_i show intended spread (group within a leak cohort)"

    def __init__(self, spread_max: float = 0.10):
        self.spread_max = float(spread_max)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = (f"Delta_ret <= {self.spread_max:g} (effect size, doctrine R4); "
                "corr reported as direction only")
        ret = ctx.get("self_probe", "retention")
        amp = ctx.get("self_probe", "payload_abs")
        if ret is None or amp is None:
            return _inapplicable(self.name, self.mode, band, "no retention/payload pair",
                                 verb="decay")
        ret = np.asarray(ret, dtype=float)
        amp = np.asarray(amp, dtype=float)
        ok = np.isfinite(ret) & np.isfinite(amp)
        ret, amp = ret[ok], amp[ok]
        if ret.size < 2:
            return _inapplicable(self.name, self.mode, band, "fewer than 2 live items",
                                 verb="decay")
        # two |a| groups (below/above the median) => the effect size
        med = float(np.median(amp))
        lo = ret[amp <= med]
        hi = ret[amp > med]
        if lo.size == 0 or hi.size == 0:
            spread = float(np.max(ret) - np.min(ret))
        else:
            spread = float(abs(np.mean(lo) - np.mean(hi)))
        corr = float("nan")
        if np.std(ret) > 1e-12 and np.std(amp) > 1e-12:
            corr = float(np.corrcoef(ret, amp)[0, 1])
        return MonitorReading(
            name=self.name, mode=self.mode, value=spread, band=band,
            tripped=bool(spread > self.spread_max),
            lever="payload excursion / decay law", verb="decay (uncleanable at large excursion)",
            detail={"delta_retention": spread, "corr_direction_only": corr,
                    "n_items": int(ret.size),
                    "known_uncleanable": "at large excursion the fix is C1W27 gated stiffness",
                    "false_trip_mode": self.false_trip},
        )


class DeadAxisMonitor:
    """#10 degenerate axes / silent knobs (N19/N58/N20: mis-wired fields that are
    never read; read-mode axis dead at ``clu_steps=1``).

    Two tiers. **(a) plumbing, O(1)**: every declared config field is wrapped in
    an access-counting proxy — trip **at startup** if a declared knob is never
    read. **(b) semantic**: perturb each knob +/- and trip if no declared
    observable moves by more than ``3 * noise``.

    Verb: ``stop`` at startup. A dead axis makes every band statement about that
    axis vacuous.
    """

    name = "dead_axis"
    mode = 10
    false_trip = "a knob that is live but currently at a no-op value is not dead (only 'never read' trips)"

    def __init__(self, noise_mult: float = 3.0):
        self.noise_mult = float(noise_mult)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = "every DECLARED knob is read at least once; each swept knob moves an observable"
        reads = ctx.extras.get("knob_reads")
        declared = ctx.extras.get("knobs_declared")
        if reads is None or declared is None:
            return _inapplicable(self.name, self.mode, band, "no config access counts",
                                 verb="stop")
        never_read = sorted(k for k in declared if int(reads.get(k, 0)) == 0)
        sweep = ctx.extras.get("knob_sweep") or {}  # knob -> observable delta / noise
        dead_semantic = sorted(k for k, v in sweep.items()
                               if float(v) <= self.noise_mult)
        tripped = bool(never_read) or bool(dead_semantic)
        return MonitorReading(
            name=self.name, mode=self.mode, value=float(len(never_read) + len(dead_semantic)),
            band=band, tripped=tripped, lever="config plumbing", verb="stop",
            detail={"never_read": never_read, "dead_semantic": dead_semantic,
                    "n_declared": len(declared), "sweep": sweep,
                    "false_trip_mode": self.false_trip},
        )


class ReachMonitor:
    """#11 reach failure — **the saddle criterion, zero free parameters**.

    Per item, with ``beta_i = D_i / (2 alpha s_i^2)`` and
    ``L_i = sqrt(|c_i|^2 + a_i^2)``: the launch manifold captures item ``i`` iff
    ``L_i / s_i < kappa_stat(beta_i)``, i.e. iff ``|a_i| < a_U`` where ``a_U`` is
    set by the middle root ``R_2`` of ``h(v) = v (1 + beta e^{-v^2/2}) = L/s``.
    **Trip at write time on any item with margin ``a_U - |a_i| <= 0``.**

    Verified 31/32 on the trained shipped ``V`` with zero free parameters; reach
    is **logarithmically un-buyable** (kappa 4 -> 5 costs 55x depth).
    Verb: ``place`` (move the item inward) / ``anneal`` / ``route`` / ``admit``.
    False-trip mode: the criterion is single-well and has no neighbour term, so
    it mis-classifies crowded high-K cells — pair it with #12 above ~80%
    packing utilisation.
    """

    name = "reach"
    mode = 11
    false_trip = "single-well criterion: no neighbour term => mis-flags crowded cells (pair with #12 above 80% utilisation)"

    def __init__(self, margin_min: float = 0.0):
        self.margin_min = float(margin_min)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = f"a_U - |a_i| > {self.margin_min:g} for every live item (saddle criterion, 0 free params)"
        margins = ctx.extras.get("reach_margins")
        if margins is None or len(margins) == 0:
            return _inapplicable(self.name, self.mode, band, "no per-item reach margins",
                                 verb="place|anneal|route|admit")
        m = np.asarray(margins, dtype=float)
        worst = float(np.min(m))
        n_bad = int(np.sum(m <= self.margin_min))
        return MonitorReading(
            name=self.name, mode=self.mode, value=worst, band=band,
            tripped=bool(n_bad > 0), lever="payload excursion / |c| / width",
            verb="place|anneal|route|admit",
            detail={"worst_margin": worst, "n_unreachable": n_bad,
                    "n_items": int(m.size),
                    "unreachable_ids": _jsonable(ctx.extras.get("unreachable_ids", [])),
                    "a_U": _jsonable(ctx.extras.get("a_U", [])),
                    "false_trip_mode": self.false_trip},
        )


class StarvationMonitor:
    """#12 starve-and-overwrite (w26: naive sequential/masked writes give each
    item ``atoms/K`` and later writes bury earlier ones).

    Three legs: (a) allocation fairness ``min_i D_i / max_i D_i`` — trip below
    0.5; (b) the **C3 first-order bound** ``|grad dV(q*)| / lambda_min`` on the
    ``k`` nearest stored items — trip above ``2x`` the bound (C1 gated: median
    ratio 1.0002); (c) self-probe retention of the OLDEST items after each write
    — trip on a drop above ``eps`` per write.
    ⚠ Evaluate at the **relaxed fixed point**, not the launch point (N74).
    """

    name = "starvation"
    mode = 12
    false_trip = "an intentional update of an already-live item looks like starvation (exempt same-id writes)"

    def __init__(self, fairness_min: float = 0.5, c3_ratio_max: float = 2.0,
                 retention_drop_max: float = 0.10):
        self.fairness_min = float(fairness_min)
        self.c3_ratio_max = float(c3_ratio_max)
        self.retention_drop_max = float(retention_drop_max)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = (f"fairness >= {self.fairness_min:g}; C3 ratio <= {self.c3_ratio_max:g}x "
                f"the first-order bound; oldest-item retention drop <= {self.retention_drop_max:g}")
        fair = ctx.extras.get("fairness")
        if fair is None:
            return _inapplicable(self.name, self.mode, band, "no allocation stats",
                                 verb="admit|place|expand|evict")
        fair = float(fair)
        c3 = float(ctx.extras.get("c3_ratio", float("nan")))
        drop = float(ctx.extras.get("oldest_retention_drop", float("nan")))
        tripped = bool(fair < self.fairness_min)
        tripped = tripped or bool(np.isfinite(c3) and c3 > self.c3_ratio_max)
        tripped = tripped or bool(np.isfinite(drop) and drop > self.retention_drop_max)
        return MonitorReading(
            name=self.name, mode=self.mode, value=fair, band=band, tripped=tripped,
            lever="write operator / allocation", verb="admit|place|expand|evict",
            detail={"fairness": fair, "c3_ratio": c3, "oldest_retention_drop": drop,
                    "false_trip_mode": self.false_trip},
        )


class MaturityMonitor:
    """#13 under-trained artefacts — **a provenance field, NOT a trip** (N94).

    Every reading carries ``{epochs, write_steps, wall_clock}``; a maturity gate
    refuses to *promote* a sub-threshold reading to a band statement. It never
    stops a run: it stops a claim.
    """

    name = "maturity"
    mode = 13
    never_trips = True
    false_trip = "n/a — not a trip"

    def __init__(self, min_write_steps: int = 40):
        self.min_write_steps = int(min_write_steps)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        steps = int(ctx.extras.get("write_steps", 0))
        mature = steps >= self.min_write_steps
        return MonitorReading(
            name=self.name, mode=self.mode, value=float(steps),
            band=f"write_steps >= {self.min_write_steps} to PROMOTE a reading (N94)",
            tripped=False,  # never a trip, by construction
            applicable=True, lever="", verb="",
            detail={"promotable": mature, "false_trip_mode": self.false_trip},
            provenance={"write_steps": steps,
                        "epochs": ctx.extras.get("epochs"),
                        "wall_clock_s": ctx.extras.get("wall_clock_s"),
                        "note": ("readings below the maturity threshold may be logged, "
                                 "never promoted to a band statement")},
        )


class GuardLivenessMonitor:
    """M14 — mode #3 applied to the **controller itself** (`controller-doctrine`
    §6): on a canary stream constructed to require intervention, every designed
    guard must fire at least once. **Trip if any guard's canary firing rate is 0.**

    This is the only check that catches "the policy has learned a parameter
    setting that makes a guard arithmetically unable to fire" — N74's failure
    transplanted to a learned policy. Without it, "no monitor quantity in the
    objective" is insufficient: the policy cannot weaken a guard through the
    loss, but it *can* drive the store into a regime where the guard is vacuous.
    """

    name = "guard_liveness"
    mode = 14
    false_trip = "a guard that the canary stream does not exercise (canary must be constructed to require each)"

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = "every designed guard fires >= once on the canary stream"
        counts = ctx.extras.get("canary_guard_counts")
        if counts is None:
            return _inapplicable(self.name, self.mode, band, "no canary stream run",
                                 verb="stop")
        dead = sorted(k for k, v in counts.items() if int(v) == 0)
        return MonitorReading(
            name=self.name, mode=self.mode, value=float(len(dead)), band=band,
            tripped=bool(dead), lever="controller policy / guard calibration", verb="stop",
            detail={"never_fired": dead, "counts": _jsonable(counts),
                    "false_trip_mode": self.false_trip},
        )


class LaunchCollapseMonitor:
    """#15 **launch collapse** — the codebook-collapse mode of a *learned* launch
    head (charter §A21, C2W7). ⛔ The anti-collapse row iteration 1 did not need
    because its head was DESIGNED, not learned.

    **Statistic: the MARGINAL usage perplexity** ``S_marg = exp(H(p_bar))`` where
    ``p_bar`` is the across-inputs mean of the read's per-well importance code,
    normalised. It is the *effective number of wells the head uses over the whole
    batch*: ``N_a`` = a perfectly uniform marginal, ``1`` = every query is sent to
    the same well. **Trip if ``S_marg < band_lo * N_a``** (default ``0.5 N_a``).

    ⛔⛔ **Per-query concentration is CONFIDENCE and is NEVER what this row
    watches.** A query that puts all ``k`` of its particles into its ``F`` wells
    is the design working; the failure being monitored is the *marginal* — the
    head learning to ignore its input. The reported ``mean_per_query_perplexity``
    is a **diagnostic only** and is never the trip predicate.

    Verb: ``regularize`` (the batch-level anti-collapse penalty of
    :func:`chlu.core.multiplicity_read.anticollapse_penalty`, which ships **OFF**
    — doctrine §3.3's monitored-first/regularized-second order) -> ``place``.

    **Designed negative (N74: a guard that cannot fire is vacuous):** an
    input-independent allocation (every query's code identical) gives
    ``S_marg = F`` or less; a one-well head gives ``S_marg = 1``. Both are
    asserted in ``tests/test_multiplicity_read.py``.
    """

    name = "launch_collapse"
    mode = 15
    false_trip = ("a genuinely low-entropy TASK (few distinct answers in the "
                  "batch) has a low marginal by construction — the band is "
                  "scoped to the family's own S_eff, not to N_a, when declared")

    def __init__(self, band_lo: float = 0.5):
        self.band_lo = float(band_lo)

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        band = (f"S_marg = exp(H(p_bar)) >= {self.band_lo:g} * N_a "
                "(marginal well usage across the batch)")
        st = ctx.extras.get("launch_usage")
        if st is None:
            return _inapplicable(self.name, self.mode, band,
                                 "no launch-usage pass", verb="regularize|place")
        n_w = float(st.get("n_wells", 0) or 0)
        s_marg = float(st.get("marginal_perplexity", float("nan")))
        if n_w <= 0 or math.isnan(s_marg):
            return _inapplicable(self.name, self.mode, band,
                                 "launch-usage pass carried no N_a/perplexity",
                                 verb="regularize|place")
        return MonitorReading(
            name=self.name, mode=self.mode, value=s_marg, band=band,
            tripped=bool(s_marg < self.band_lo * n_w),
            lever="launch head / allocation", verb="regularize|place",
            detail={"marginal_perplexity": s_marg, "n_wells": n_w,
                    "floor": self.band_lo * n_w,
                    "marginal_max": st.get("marginal_max", float("nan")),
                    "marginal_entropy": st.get("marginal_entropy", float("nan")),
                    "S_eff_marginal": st.get("S_eff_marginal", float("nan")),
                    "mean_per_query_perplexity": st.get("per_query_perplexity",
                                                        float("nan")),
                    "per_query_is_diagnostic_only": True,
                    "false_trip_mode": self.false_trip},
        )


# --------------------------------------------------------------------------
# Free functions used by monitors and by ``pytest``
# --------------------------------------------------------------------------
def objective_divergence_predicate(slope_loss: float, slope_acq: float,
                                   eps: float = 0.0,
                                   eps_acq: float = 0.0) -> bool:
    """Monitor #6's trip predicate, as a free function so a recorded reading can
    be **re-scored offline** at any ``(eps, eps_acq)`` (the C2W2 and C2W4
    before/after diffs — the store is never re-run).

    Three settings span the monitor's whole history:

    * ``eps = eps_acq = 0`` — the **pre-repair** predicate (``slope_loss < 0 and
      slope_acq <= 0``), 58 trips on the C2W1 gym;
    * ``eps > 0, eps_acq = 0`` — the **C2W2 loss half only** (the C2W2-C2W3
      shipped state), 27 trips;
    * ``eps > 0, eps_acq > 0`` — the **two-sided** band shipped in C2W4.

    ⚠ The predicate is **monotone non-decreasing in ``eps_acq``**: raising the
    acq band can only *add* trips. A ``TRIP -> no-trip`` flip attributed to
    ``eps_acq`` is therefore a contradiction and means something else moved.
    """
    return bool(float(slope_loss) < -float(eps)
                and float(slope_acq) <= +float(eps_acq))


class ConfigAccessProxy:
    """⭐ Monitor #10 **tier (a)**: the O(1) access-counting config proxy
    (doctrine I-8; C2W1 shipped ``knob_tier_a_implemented: false``).

    Wraps a dataclass config and counts every attribute read, so a knob that is
    **declared but never read** is caught at startup — before any expensive
    semantic sweep. It catches N19, N20 and N58 exactly, all three of which are
    "the field is wired to nothing".

    Usage::

        cfg = ConfigAccessProxy(CluSystemConfig(addr_dim=4))
        system = build_system(cfg.unwrap(), ...)   # or pass the proxy itself
        ...
        assert_knobs_live(cfg, exempt={"quick"})   # raises at startup
        ctx.extras.update(cfg.knob_extras())       # feeds monitor #10

    Notes and limits, stated rather than discovered later:

    * attribute **reads** are counted (``__getattr__``/``__getattribute__``),
      including reads made by ``dataclasses.fields``-driven code such as
      ``as_flag_table``; call :meth:`reset` after any such bulk pass or every
      knob will look live;
    * ``dataclasses.replace(proxy, ...)`` does **not** work (it dispatches on
      ``type``) — use ``replace(proxy.unwrap(), ...)``;
    * it is a guard, never a loss: nothing here may enter an objective.
    """

    __slots__ = ("_obj", "_counts", "_declared")

    def __init__(self, obj, declared: Optional[Sequence[str]] = None):
        object.__setattr__(self, "_obj", obj)
        if declared is None:
            try:
                from dataclasses import fields as _fields

                declared = [f.name for f in _fields(obj)]
            except Exception:
                declared = [k for k in vars(obj) if not k.startswith("_")]
        object.__setattr__(self, "_declared", tuple(str(k) for k in declared))
        object.__setattr__(self, "_counts", {k: 0 for k in declared})

    # -- delegation ---------------------------------------------------------
    def __getattr__(self, name):
        obj = object.__getattribute__(self, "_obj")
        counts = object.__getattribute__(self, "_counts")
        if name in counts:
            counts[name] += 1
        return getattr(obj, name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)

    def __repr__(self):
        return f"ConfigAccessProxy({object.__getattribute__(self, '_obj')!r})"

    # -- the tier-(a) API ---------------------------------------------------
    def unwrap(self):
        """The wrapped config (no counting)."""
        return object.__getattribute__(self, "_obj")

    @property
    def counts(self) -> Dict[str, int]:
        return dict(object.__getattribute__(self, "_counts"))

    @property
    def declared(self) -> List[str]:
        return list(object.__getattribute__(self, "_declared"))

    def reset(self) -> None:
        """Zero every counter (call after a bulk ``fields()``-driven pass)."""
        counts = object.__getattribute__(self, "_counts")
        for k in counts:
            counts[k] = 0

    def never_read(self, exempt: Sequence[str] = ()) -> List[str]:
        counts = object.__getattribute__(self, "_counts")
        ex = set(exempt)
        return sorted(k for k, v in counts.items() if v == 0 and k not in ex)

    def knob_extras(self, exempt: Sequence[str] = ()) -> Dict[str, Any]:
        """The two ``ctx.extras`` keys :class:`DeadAxisMonitor` reads."""
        counts = self.counts
        declared = [k for k in self.declared if k not in set(exempt)]
        return {"knob_reads": {k: counts.get(k, 0) for k in declared},
                "knobs_declared": declared,
                "knob_tier_a_implemented": True}


class DeadKnobError(RuntimeError):
    """A declared config field was never read — monitor #10 tier (a), at startup."""


def assert_knobs_live(proxy: "ConfigAccessProxy", exempt: Sequence[str] = (),
                      raise_on_dead: bool = True) -> List[str]:
    """**Fail at startup** on any declared-but-never-read field (doctrine I-8).

    Returns the dead list; raises :class:`DeadKnobError` unless
    ``raise_on_dead=False``. A dead axis makes every band statement about that
    axis vacuous, so this is a *stop*, not a warning.
    """
    dead = proxy.never_read(exempt)
    if dead and raise_on_dead:
        raise DeadKnobError(
            f"{len(dead)} declared config field(s) never read: {dead}. "
            "Monitor #10 tier (a) (doctrine I-8): a knob wired to nothing makes "
            "every band statement about that knob vacuous (N19/N20/N58)."
        )
    return dead


#: Kinetic modes the #7 gauge is *stated* for, and its scope in each.
#: `controller-doctrine` I-7 / R2: the gauge is **Newtonian-only**; under the
#: relativistic kinetic it breaks as O(1/c^2) (9.1e-2 relative at the paper's
#: c = 5), so a relativistic residual is a reported SCOPE, never a pass/fail.
#: ⚠ C2W2 measurement (this is a SHARPENING of I-7's "the gauge is
#: Newtonian-only"): under ``newtonian_identity`` the transformation is **not a
#: gauge at all** — ``T = 0.5 p^2`` ignores ``M``, so scaling ``(M, V, p0)``
#: rescales ``V`` and ``p0`` with nothing to compensate them and the trajectory
#: moves (measured residual **0.25** on the atom store). The gauge is exact in
#: ``newtonian_learned`` only. N76's "mass stores nothing" still holds trivially
#: in identity mode, because the mass is not in the dynamics at all.
GAUGE_SCOPE = {
    "newtonian_identity": ("N/A — M does not enter T, so (M,V,p0)->(lam M, lam V, "
                           "lam p0) is NOT a gauge orbit here (measured residual "
                           "0.25); mass stores nothing trivially"),
    "newtonian_learned": "exact (the gauge the theorem is stated for)",
    "relativistic": "BREAKS as O(1/c^2) — SCOPE, not a pass (doctrine R2)",
}


def gauge_orbit_residual(model, q0, p0, steps: int, dt: float, gamma: float,
                         lam: float = 2.0) -> float:
    """Monitor #7's gauge test: max relative **trajectory** deviation under
    ``(M, V, p0) -> (lambda M, lambda V, lambda p0)``.

    Newtonian: exactly zero (measured 0.0 / 2.8e-16). Relativistic: O(1/c^2),
    which is a **scope**, not a pass.
    """
    import equinox as eqx
    import jax.numpy as jnp

    from chlu.experiments.goldstone_harness import log_mass_for_inertia

    lam = float(lam)
    base = np.asarray(model(q0, p0, steps, dt, gamma))

    inertia = np.asarray(model.effective_inertia())
    scaled = eqx.tree_at(
        lambda m: m.log_mass, model,
        replace=log_mass_for_inertia(jnp.asarray(lam * inertia)),
    )
    pot = model.potential_net

    class _Scaled(eqx.Module):
        inner: eqx.Module
        lam: float = eqx.field(static=True)

        def __call__(self, q):
            return self.lam * self.inner(q)

    scaled = eqx.tree_at(lambda m: m.potential_net, scaled, replace=_Scaled(pot, lam))
    other = np.asarray(scaled(q0, lam * jnp.asarray(p0), steps, dt, gamma))

    dim = np.asarray(q0).shape[-1]
    # q must be invariant; p is rescaled by lambda (it is the gauge direction).
    dq = np.abs(other[..., :dim] - base[..., :dim])
    scale = max(float(np.max(np.abs(base[..., :dim]))), 1e-12)
    return float(np.max(dq) / scale)


def _h(v: float, beta: float) -> float:
    return v * (1.0 + beta * math.exp(-0.5 * v * v))


def _hprime(v: float, beta: float) -> float:
    return 1.0 + beta * math.exp(-0.5 * v * v) * (1.0 - v * v)


def _bisect(f, lo: float, hi: float, n: int = 200) -> float:
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _stationarity_extrema(beta: float):
    """``(v_a, v_b)`` — the local max and local min of ``h``, or ``None``."""
    if beta <= math.exp(1.5) / 2.0:  # 2.2408: no spurious minimum can ever exist
        return None
    # h' = 1 + beta e^{-v^2/2}(1 - v^2): negative somewhere in (1, inf)
    grid = np.linspace(1.0, 12.0, 2000)
    vals = np.array([_hprime(float(v), beta) for v in grid])
    neg = np.where(vals < 0)[0]
    if neg.size == 0:
        return None
    i0, i1 = int(neg[0]), int(neg[-1])
    v_a = _bisect(lambda v: _hprime(v, beta), float(grid[max(i0 - 1, 0)]), float(grid[i0]))
    v_b = _bisect(lambda v: _hprime(v, beta), float(grid[i1]), float(grid[min(i1 + 1, len(grid) - 1)]))
    return v_a, v_b


def kappa_stat(beta: float) -> float:
    """``kappa_stat(beta) = h(v_b)`` — below it the well is the only attractor.

    Measured anchors (`readout-channel-theory` §1.0): 3.33 / 4.08 / 4.67 / 6.06
    at beta = 1e1 / 1e2 / 1e3 / 1e6. Returns ``inf`` when no spurious minimum can
    exist (``beta <= e^{3/2}/2 = 2.2408``) — reach is then unconditional.
    """
    ext = _stationarity_extrema(float(beta))
    if ext is None:
        return float("inf")
    return _h(ext[1], float(beta))


def _middle_root(A: float, beta: float) -> Optional[float]:
    """The saddle root ``v_2`` of ``h(v) = A``, or ``None`` if there is no saddle."""
    ext = _stationarity_extrema(beta)
    if ext is None:
        return None
    v_a, v_b = ext
    h_a, h_b = _h(v_a, beta), _h(v_b, beta)
    if not (h_b < A < h_a):
        return None
    return _bisect(lambda v: _h(v, beta) - A, v_a, v_b)


def saddle_reach_threshold(depth: float, width: float, alpha: float,
                           c_norm: float, a_max: float = 8.0) -> float:
    """Monitor #11's ``a_U``: the largest payload excursion still reachable.

    Zero free parameters; solves the middle root of
    ``h(v) = v (1 + beta e^{-v^2/2}) = L / s`` with ``beta = D / (2 alpha s^2)``
    and ``L = sqrt(c_norm^2 + a^2)``. Returns ``inf`` when no saddle exists (the
    item is reachable at any excursion).

    Criterion (U) (`readout-channel-theory` §1.0): the query is captured by item
    ``i`` iff ``L/s < kappa_stat(beta)`` **or** ``|a_i| < R_2``. ``a_U`` is the
    self-consistent crossing ``a = s * v_2(sqrt(c^2 + a^2)/s)``.
    """
    D, s, al, c = float(depth), float(width), float(alpha), float(c_norm)
    if s <= 0 or D <= 0 or al <= 0:
        return float("inf")
    # ⚠ C2W5: the three positivity guards above are NOT sufficient. A group whose
    # wells were never dug has a depth-weighted width that underflows, and
    # `s * s` flushes to 0.0 while `s > 0` still holds (measured: s = 1e-200
    # raises, s = 1e-154 does not) — so the division below raised
    # ZeroDivisionError and killed seed 2 of the C2W4 pilot's shipped 3-seed run.
    # `cluformer-pilot` guarded its own CALLER (commit 7bc166a, in
    # `train_cluformer.py`); the guard belongs here, where the division is.
    # An item with no well is ABSENT, not "unreachable": `inf` is the same answer
    # the D <= 0 branch already gives it.
    den = 2.0 * al * s * s
    if not (den > 0.0) or not math.isfinite(den):
        return float("inf")
    beta = D / den
    ks = kappa_stat(beta)

    def captured(a: float) -> bool:
        A = math.sqrt(c * c + a * a) / s
        if A < ks:
            return True
        v2 = _middle_root(A, beta)
        if v2 is None:
            # no saddle at this A: the well is the only attractor on the ray
            return True
        return a < s * v2

    if captured(a_max):
        return float("inf")
    lo, hi = 0.0, a_max
    if not captured(lo):
        return 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if captured(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def erf_margin_accuracy(margin: float, sigma: float) -> float:
    """``acc ~ erf(margin / (sqrt(2) sigma))`` — the N2 law.

    ⚠ ``kappa = 5`` in `clu-controller-spec` §2 indexes **spacing**/sigma, not
    margin/sigma: 99% needs ``margin >= 2.576 sigma`` <=> ``spacing >= 5.15
    sigma`` (`controller-doctrine` R1). This function is the corrected form.
    """
    if sigma <= 0:
        return 1.0
    return float(math.erf(float(margin) / (math.sqrt(2.0) * float(sigma))))


def default_registry(loud: bool = True, **kwargs) -> MonitorRegistry:
    """The 13 monitors + M14 + the C2W7 launch-collapse row (#15)."""
    monitors = [
        OverdampingMonitor(**kwargs.get("overdamping", {})),
        SettleArgminMonitor(**kwargs.get("settle_argmin", {})),
        VacuousGateMonitor(**kwargs.get("vacuous_gate", {})),
        BlankControlMonitor(**kwargs.get("blank", {})),
        AddressingMonitor(**kwargs.get("addressing", {})),
        ObjectiveDivergenceMonitor(**kwargs.get("objective_divergence", {})),
        MassGaugeMonitor(),
        CertificateMonitor(**kwargs.get("certificates", {})),
        LifetimeMonitor(**kwargs.get("lifetimes", {})),
        DeadAxisMonitor(**kwargs.get("dead_axis", {})),
        ReachMonitor(**kwargs.get("reach", {})),
        StarvationMonitor(**kwargs.get("starvation", {})),
        MaturityMonitor(**kwargs.get("maturity", {})),
        GuardLivenessMonitor(),
        LaunchCollapseMonitor(**kwargs.get("launch_collapse", {})),
    ]
    return MonitorRegistry(monitors, loud=loud)


__all__ = [
    "CLASS_I", "CLASS_II", "CLASS_III", "CLASS_IV", "CLASS_V", "SEVERITY",
    "MonitorReading", "MonitorContext", "Monitor", "TripRecord",
    "MonitorRegistry", "default_registry",
    "OverdampingMonitor", "SettleArgminMonitor", "VacuousGateMonitor",
    "BlankControlMonitor", "AddressingMonitor", "ObjectiveDivergenceMonitor",
    "MassGaugeMonitor", "CertificateMonitor", "LifetimeMonitor",
    "DeadAxisMonitor", "ReachMonitor", "StarvationMonitor", "MaturityMonitor",
    "GuardLivenessMonitor", "LaunchCollapseMonitor",
    "gauge_orbit_residual", "saddle_reach_threshold", "kappa_stat",
    "erf_margin_accuracy",
    # C2W2 repairs (#6 dead-band, #10 tier (a), #7 scope)
    "objective_divergence_predicate", "ConfigAccessProxy", "DeadKnobError",
    "assert_knobs_live", "GAUGE_SCOPE",
]
