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
 6   **SHARPENED**: retrieval leg = the self-probe acquisition rate.
 7   **SHARPENED (scope)**: a ``pytest`` gauge over the *trajectory*, and the
     gauge is **Newtonian-only** (relativistic breaks as O(1/c^2)).
 8   **SHARPENED**: ``kappa = 5`` indexes ``spacing/sigma``, not ``margin/sigma``
     (99% needs ``margin >= 2.576 sigma``); ``delta_read`` is basin-conditioned.
 9   **REPLACED**: ``|corr(retention, |a|)| > 0.30`` trips at *every* excursion
     tested. Trip on the retention **effect size** instead; corr is a direction
     indicator only.
10   **SHARPENED**: an O(1) plumbing tier (declared-but-never-read config field)
     plus the expensive semantic sweep.
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

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence


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
        raise NotImplementedError


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
        raise NotImplementedError

    def register(self, monitor: Monitor) -> None:
        """Add a monitor after construction."""
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> List[MonitorReading]:
        """Run every monitor once; log and return the readings."""
        raise NotImplementedError

    @property
    def trips(self) -> List[TripRecord]:
        """Every trip so far, in order."""
        raise NotImplementedError

    def class_i_tripped(self, window: int = 1) -> List[str]:
        """Names of class-I monitors tripped in the last ``window`` observations.

        The controller consults this before any memory-mutating verb
        (`controller-doctrine` §5, consequence 1: ``evict`` may not fire in the
        same step as a class-I trip).
        """
        raise NotImplementedError

    def summary(self) -> Dict[str, dict]:
        """Per-monitor: observations, trips, ever-tripped, and **untested**.

        ``untested = (n_observations == 0) or (never applicable)``. A monitor
        that never fired on any configuration is labelled untested, never green.
        """
        raise NotImplementedError

    def to_markdown(self) -> str:
        """The trip-state table (a reported artifact of every run)."""
        raise NotImplementedError


# ==========================================================================
# The thirteen. Each carries its band + provenance in the docstring, and its
# FALSE-TRIP MODE (the benign situation that fires it) — an uncharacterised
# monitor gets disabled by the next engineer, and then it is not a guard.
# ==========================================================================


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

    def __init__(self, rho_max: float = 1e-6, delta_min: float = 0.02,
                 grad_floor: float = 1e-9):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def __init__(self, rho_min: float = 0.10, u_floor: float = 0.01):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


class VacuousGateMonitor:
    """#3 vacuous gate (N74: spacing 1.4142 vs ``d_safe`` 1.10 => the gate could
    not fire arithmetically; N91: the *address space* was binding, not the gate).

    Three legs: (i) fire-rate ``f`` over the stream — **trip if f in {0, 1}**;
    (ii) **validity** ``corr(gate margin, measured post-write drift)`` — trip
    below 0.3 (a certificate that does not predict drift is not a certificate);
    (iii) packing utilisation ``n_live / N_pack`` — trip above 0.95.

    Verb: ``admit`` (recalibrate) / ``expand`` / ``stop``.
    False-trip mode: a stream of genuinely well-separated proposals gives
    ``f = 0`` legitimately — which is why leg (ii) must agree before acting.
    """

    name = "vacuous_gate"
    mode = 3

    def __init__(self, validity_min: float = 0.30, utilisation_max: float = 0.95):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def __init__(self, sigma_mult: float = 3.0):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def __init__(self, acq_min: float = 0.90):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


class ObjectiveDivergenceMonitor:
    """#6 objective/goal divergence (w25/w26: write loss -> 0 while retrieval
    fails — the objective stops seeing crowding).

    Rolling window over ``slope(write-loss)`` and ``slope(acq)`` from #5.
    **Trip if the loss falls while acquisition is flat/declining for ``window``
    consecutive windows.** Leading indicator (curriculum-blind):
    ``min_i (sep_i - 2 s_i)`` trending down while the loss falls.
    Verb: ``stop`` the write objective -> ``consolidate`` -> resume.
    """

    name = "objective_divergence"
    mode = 6

    def __init__(self, window: int = 3):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def __init__(self, spacing_sigma_min: float = 5.15, n4_ratio_min: float = 2.0):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def __init__(self, spread_max: float = 0.10):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def __init__(self, noise_mult: float = 3.0):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def __init__(self, margin_min: float = 0.0):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def __init__(self, fairness_min: float = 0.5, c3_ratio_max: float = 2.0,
                 retention_drop_max: float = 0.10):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


class MaturityMonitor:
    """#13 under-trained artefacts — **a provenance field, NOT a trip** (N94).

    Every reading carries ``{epochs, write_steps, wall_clock}``; a maturity gate
    refuses to *promote* a sub-threshold reading to a band statement. It never
    stops a run: it stops a claim.
    """

    name = "maturity"
    mode = 13

    def __init__(self, min_write_steps: int = 40):
        raise NotImplementedError

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


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

    def observe(self, ctx: MonitorContext) -> MonitorReading:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Free functions used by monitors and by ``pytest``
# --------------------------------------------------------------------------
def gauge_orbit_residual(model, q0, p0, steps: int, dt: float, gamma: float,
                         lam: float = 2.0) -> float:
    """Monitor #7's gauge test: max relative **trajectory** deviation under
    ``(M, V, p0) -> (lambda M, lambda V, lambda p0)``.

    Newtonian: exactly zero (measured 0.0 / 2.8e-16). Relativistic: O(1/c^2),
    which is a **scope**, not a pass.
    """
    raise NotImplementedError


def saddle_reach_threshold(depth: float, width: float, alpha: float,
                           c_norm: float) -> float:
    """Monitor #11's ``a_U``: the largest payload excursion still reachable.

    Zero free parameters; solves the middle root of
    ``h(v) = v (1 + beta e^{-v^2/2}) = L / s`` with ``beta = D / (2 alpha s^2)``
    and ``L = sqrt(c_norm^2 + a^2)``. Returns ``inf`` when no saddle exists (the
    item is reachable at any excursion).
    """
    raise NotImplementedError


def erf_margin_accuracy(margin: float, sigma: float) -> float:
    """``acc ~ erf(margin / (sqrt(2) sigma))`` — the N2 law.

    ⚠ ``kappa = 5`` in `clu-controller-spec` §2 indexes **spacing**/sigma, not
    margin/sigma: 99% needs ``margin >= 2.576 sigma`` <=> ``spacing >= 5.15
    sigma`` (`controller-doctrine` R1). This function is the corrected form.
    """
    raise NotImplementedError


def default_registry(loud: bool = True, **kwargs) -> MonitorRegistry:
    """The 13 monitors + M14, at their doctrine-default bands."""
    raise NotImplementedError


__all__ = [
    "CLASS_I", "CLASS_II", "CLASS_III", "CLASS_IV", "CLASS_V",
    "MonitorReading", "MonitorContext", "Monitor", "TripRecord",
    "MonitorRegistry", "default_registry",
    "OverdampingMonitor", "SettleArgminMonitor", "VacuousGateMonitor",
    "BlankControlMonitor", "AddressingMonitor", "ObjectiveDivergenceMonitor",
    "MassGaugeMonitor", "CertificateMonitor", "LifetimeMonitor",
    "DeadAxisMonitor", "ReachMonitor", "StarvationMonitor", "MaturityMonitor",
    "GuardLivenessMonitor",
    "gauge_orbit_residual", "saddle_reach_threshold", "erf_margin_accuracy",
]
