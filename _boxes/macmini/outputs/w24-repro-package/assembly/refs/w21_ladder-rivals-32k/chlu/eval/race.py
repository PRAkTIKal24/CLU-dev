"""The **C2W2 race card**: the one cell schema both routes emit, and its scorer.

⭐ **This module is the wave's public surface and it is FROZEN once landed.**
``traj-write-objective`` (Route 1, the write objective) and ``ssb-shell-atoms``
(Route 2, the shell-atom store) run on different branches and touch disjoint
files; the *only* thing that makes their numbers comparable — and the only thing
the C2W2 gate is evaluated on — is that both emit :class:`RaceCell` records with
identical semantics. **Emit into this schema; do not fork it.**

**What a cell is.** One ``(route, arm, family, seed)`` measurement of the dividend

    ``dividend = (full CLU) - (its own settle-deleted launder)``

on one family's primary metric, carrying *every* control that the charter binds
to it — because a dividend without its controls is not a result:

* the **four mandatory launders** — settle-deleted (the dividend's own
  denominator), same-keys null, blank/empty store, and ⭐ the **trajectory
  launder** (``full`` / ``q0_only`` / ``endpoints`` / ``blank_store``) on every
  psi that can see the address block (charter §A5/C2W2.1, no exceptions);
* the family's strongest **+0 B substitute** *and its signed margin* — the
  charter §A6 "weak proceed" caveat is graded on that margin, so it is recorded
  beside the dividend rather than argued about at review;
* the **two-sided byte ledger** with its ``matched`` flag (⛔ never quote a cell
  as a byte-matched dividend when ``ratio >= 2.20`` — that ratio is
  architectural, gym PREREG-B1);
* the **write record** (``steps``, ``final_loss``, ``lambda_min_min``,
  ``converged``) and ``gate_admissible``, because the gym measured unconverged
  writes at sub-shipped budgets (final loss 0.20-0.24, ``lambda_min`` -0.21…-1.20)
  and *letting an unwritten store cast a <=0 vote would fire B' on noise*;
* the **term-liveness record**, because a cell may only vote as "asked and did
  not deliver" if the objective term/dial was actually live, and the grid must
  carry a **perturbing anchor** — *"a term that never moves anything at any
  tested setting hasn't been asked; it's been whispered at"* (Head, verbatim).

**Two counterweights are implemented here, not left to prose.**

1. :func:`score_family` reports ``admissible_coverage`` as a first-class number
   and returns **every excluded cell with its reason**. Silent filtering is
   forbidden — the named failure mode is admissibility filtering quietly gutting
   coverage until B' can never fire.
2. A family with **zero admissible cells** grades ``ABSTAIN`` (after its one
   bounded budget escalation), which neither blocks B' nor supports "proceed";
   and an arm whose grid carried **no perturbing anchor** grades
   ``UNDER_POWERED_GRID``, never ``<=0 vote``.

⚠ **The gate itself is applied by the Hub**, not here. :func:`gate_summary`
computes the *arithmetic* the charter specifies ("clears 0 beyond 2 SE", sample
sd with ``ddof=1``, ``SE = sd/sqrt(n)``) and grades each family; it deliberately
does not print a verdict on the program.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

#: Bump when a field's *meaning* changes. Both routes must agree on this string.
RACE_SCHEMA_VERSION = "c2w2.1"

#: The two C2W2 routes. ``route1`` = the write objective (trajectory/path terms);
#: ``route2`` = the shell-atom store (+ tilt). Aliases are normalised, so an arm
#: labelled ``"route2_shell_atoms"`` still lands in ``route2``.
ROUTES = ("route1", "route2")

#: Grades :func:`score_family` can return. Only ``LE_ZERO_VOTE`` casts the
#: gate's <=0 vote; ``ABSTAIN`` and ``UNDER_POWERED_GRID`` cast **no** vote.
GRADE_PROCEED = "proceed"
GRADE_WEAK_PROCEED = "weak_proceed"
GRADE_LE_ZERO_VOTE = "le_zero_vote"
GRADE_ABSTAIN = "abstain"
GRADE_UNDER_POWERED = "under_powered_grid"

#: ⛔ Above this byte ratio a cell is architectural, never a byte-matched
#: dividend (gym PREREG-B1). Recorded on the ledger so the never-quote rule is
#: machine-checkable rather than a footnote.
BYTE_RATIO_ARCHITECTURAL = 2.20

_NAN = float("nan")


# ==========================================================================
# the sub-records a cell carries
# ==========================================================================
@dataclass
class TrajectoryLaunder:
    """⭐ The mandatory trajectory launder (doctrine I-2, charter §A5/C2W2.1).

    The trajectory buffer **contains** ``q0 = phi(x)``, so a psi over the raw
    buffer can be a classifier on the query embedding and nothing else. Every
    number is the family's primary metric under a different view of the buffer:

    ``full``
        psi on the real trajectory.
    ``q0_only``
        the buffer replaced everywhere by the launch point — what survives is
        ``phi(x)``, not the store. C2W1 measured **0.129 vs chance 0.125** for
        the DeepSets psi, which *refuted* the predicted leak.
    ``endpoints``
        the **capacity-matched** baseline ``[q0, q_addr, q*, p*]`` — never
        ``q0_only``; a trajectory term that only beats ``q0_only`` has beaten a
        strictly smaller read, not a point read.
    ``blank_store``
        the identical system with nothing written (spike §4.3: 31-63% of the
        only replicating v0 effect was reproduced by an empty store).

    ``bar`` is the leak bar (``chance + 3 SE``). :meth:`fired` is the report's
    falsifier: **if it fires, no psi number in the report is quotable** until it
    is re-run store-relative.
    """

    full: float = _NAN
    q0_only: float = _NAN
    endpoints: float = _NAN
    blank_store: float = _NAN
    chance: float = _NAN
    bar: float = _NAN

    @property
    def leak(self) -> float:
        """``max(q0_only, blank_store) - bar``. > 0 means the launder fired."""
        vals = [v for v in (self.q0_only, self.blank_store) if _finite(v)]
        if not vals or not _finite(self.bar):
            return _NAN
        return float(max(vals) - self.bar)

    def fired(self) -> bool:
        lk = self.leak
        return bool(_finite(lk) and lk > 0.0)

    @property
    def over_endpoints(self) -> float:
        """``full - endpoints`` — the honest 'the trajectory bought something'
        margin, against the capacity-matched baseline."""
        return _sub(self.full, self.endpoints)

    def as_dict(self) -> dict:
        return {"full": _j(self.full), "q0_only": _j(self.q0_only),
                "endpoints": _j(self.endpoints), "blank_store": _j(self.blank_store),
                "chance": _j(self.chance), "bar": _j(self.bar),
                "leak": _j(self.leak), "fired": self.fired(),
                "over_endpoints": _j(self.over_endpoints)}


@dataclass
class WriteRecord:
    """Write convergence — gate ruling (i). A cell may only **vote** if its
    write converged.

    ``lambda_min_min`` is the minimum over **every recorded site** of
    ``lambda_min(Hess V)``; ``< 0`` means at least one item sits on a saddle (the
    gym's multi-target ridge write measured **-0.5946**, spectator participation
    1.000 — a saddle, not a valley) and the cell is inadmissible.
    """

    steps: int = 0
    final_loss: float = _NAN
    lambda_min_min: float = _NAN
    converged: bool = False
    plateaued: bool = False
    reason: str = ""

    def admissible(self) -> bool:
        if not self.converged or self.plateaued:
            return False
        return not (_finite(self.lambda_min_min) and self.lambda_min_min < 0.0)

    def failure_reason(self) -> str:
        if self.reason:
            return self.reason
        if not self.converged:
            return "write_not_converged"
        if self.plateaued:
            return "write_loss_plateaued"
        if _finite(self.lambda_min_min) and self.lambda_min_min < 0.0:
            return f"lambda_min<0 ({self.lambda_min_min:+.4f})"
        return ""

    def as_dict(self) -> dict:
        return {"steps": int(self.steps), "final_loss": _j(self.final_loss),
                "lambda_min_min": _j(self.lambda_min_min),
                "converged": bool(self.converged), "plateaued": bool(self.plateaued),
                "admissible": self.admissible(), "reason": self.failure_reason()}


@dataclass
class Liveness:
    """Term/dial liveness — gate ruling (ii), with its counterweight.

    ``value`` vs ``baseline`` against ``bar`` is the registered liveness test
    (Route 1: does the trajectory-written store's trajectory carry more decodable
    information than the endpoint-written store's, against the **capacity-matched**
    ``endpoints`` baseline *and* the blank-store probe). ``coefficient`` is the
    setting it was measured at.

    ⭐ ``perturbing_anchor`` marks the grid's liveness anchor: at least one
    coefficient at which the term **visibly perturbs the write, even
    destructively**. Without it, an inert-everywhere result is an **under-powered
    grid**, not a legitimate <=0 vote — and :func:`score_family` enforces that.
    """

    passed: bool = False
    coefficient: float = _NAN
    value: float = _NAN
    baseline: float = _NAN
    bar: float = _NAN
    perturbing_anchor: bool = False
    grid: Sequence[float] = field(default_factory=tuple)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        return _sub(self.value, self.bar)

    def as_dict(self) -> dict:
        return {"passed": bool(self.passed), "coefficient": _j(self.coefficient),
                "value": _j(self.value), "baseline": _j(self.baseline),
                "bar": _j(self.bar), "margin": _j(self.margin),
                "perturbing_anchor": bool(self.perturbing_anchor),
                "grid": [_j(float(g)) for g in self.grid],
                "detail": _json_safe(self.detail)}


@dataclass
class ByteLedger:
    """The two-sided byte ledger. Mirrors
    :class:`chlu.eval.dividend.ByteAccount` but is JSON-flat and carries the
    ⛔ architectural flag."""

    full: int = 0
    launder: int = 0
    tol: float = 0.05
    breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        return float(self.full) / max(float(self.launder), 1.0)

    @property
    def matched(self) -> bool:
        return abs(self.ratio - 1.0) <= float(self.tol)

    @property
    def architectural(self) -> bool:
        """⛔ ``ratio >= 2.20`` => never quotable as a byte-matched dividend."""
        return self.ratio >= BYTE_RATIO_ARCHITECTURAL

    def as_dict(self) -> dict:
        return {"full": int(self.full), "launder": int(self.launder),
                "ratio": _j(self.ratio), "matched": self.matched,
                "architectural": self.architectural,
                "breakdown": {k: int(v) for k, v in self.breakdown.items()}}

    @classmethod
    def from_account(cls, acct) -> "ByteLedger":
        """Build from a :class:`chlu.eval.dividend.ByteAccount`."""
        return cls(full=int(acct.full_bytes), launder=int(acct.launder_bytes),
                   breakdown={k: int(v) for k, v in dict(acct.breakdown).items()})


# ==========================================================================
# the cell
# ==========================================================================
@dataclass
class RaceCell:
    """One ``(route, arm, family, seed)`` record. **The unit of the gate.**

    Higher-is-better metrics only (``metric_name`` says which; the gym's
    ``PRIMARY_METRIC`` already sign-corrects error-like metrics as ``neg_mae``).

    ``gate_admissible`` is *derived* by :meth:`resolve_admissibility` from the
    write record and the trajectory launder unless it is set explicitly — so an
    engineer cannot mark a cell admissible by forgetting to check.
    """

    route: str
    arm: str
    family: str
    seed: int
    metric_name: str

    # -- the dividend and its four launders -------------------------------
    full: float = _NAN
    settle_deleted_launder: float = _NAN
    same_keys_null: float = _NAN
    blank: float = _NAN
    plus_zero_byte_substitute: float = _NAN
    trajectory_launder: TrajectoryLaunder = field(default_factory=TrajectoryLaunder)

    # -- provenance / accounting ------------------------------------------
    bytes: ByteLedger = field(default_factory=ByteLedger)
    phi_id: str = ""
    phi_bytes: int = 0
    write: WriteRecord = field(default_factory=WriteRecord)
    liveness: Liveness = field(default_factory=Liveness)
    monitors: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)

    # -- verdict bookkeeping ----------------------------------------------
    gate_admissible: Optional[bool] = None
    exclusion_reason: str = ""
    seeds_n: int = 1
    dividend_se: float = _NAN
    notes: str = ""
    schema_version: str = RACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.route = _norm_route(self.route)

    # -- derived ----------------------------------------------------------
    @property
    def dividend(self) -> float:
        """``full - settle_deleted_launder`` — the KPI, computed, never stored,
        so a cell can never carry a dividend inconsistent with its own arms."""
        return _sub(self.full, self.settle_deleted_launder)

    @property
    def substitute_margin(self) -> float:
        """⭐ Signed **+0 B substitute margin** (charter §A6): ``full - substitute``.
        Negative => a family clearing 2 SE is a **weak proceed**, not a proceed."""
        return _sub(self.full, self.plus_zero_byte_substitute)

    def resolve_admissibility(self) -> bool:
        """Derive ``gate_admissible`` (and its reason) if it was not set.

        Inadmissible when the write did not converge / sits on a saddle, or when
        the **trajectory launder fired** (the psi is reading ``phi(x)``, not the
        store — the report's hard falsifier).
        """
        if self.gate_admissible is not None:
            if not self.gate_admissible and not self.exclusion_reason:
                self.exclusion_reason = "marked_inadmissible"
            return bool(self.gate_admissible)
        reasons: List[str] = []
        if not self.write.admissible():
            reasons.append(self.write.failure_reason() or "write_inadmissible")
        if self.trajectory_launder.fired():
            reasons.append(f"trajectory_launder_fired (leak {self.trajectory_launder.leak:+.4f})")
        if not _finite(self.dividend):
            reasons.append("dividend_not_finite")
        self.gate_admissible = not reasons
        self.exclusion_reason = "; ".join(reasons)
        return bool(self.gate_admissible)

    def as_dict(self) -> dict:
        self.resolve_admissibility()
        return {
            "schema_version": self.schema_version,
            "route": self.route, "arm": self.arm, "family": self.family,
            "seed": int(self.seed), "metric_name": self.metric_name,
            "full": _j(self.full),
            "settle_deleted_launder": _j(self.settle_deleted_launder),
            "same_keys_null": _j(self.same_keys_null),
            "blank": _j(self.blank),
            "plus_zero_byte_substitute": _j(self.plus_zero_byte_substitute),
            "plus_zero_byte_substitute_margin": _j(self.substitute_margin),
            "dividend": _j(self.dividend),
            "dividend_se": _j(self.dividend_se),
            "trajectory_launder": self.trajectory_launder.as_dict(),
            "bytes": self.bytes.as_dict(),
            "phi_id": self.phi_id, "phi_bytes": int(self.phi_bytes),
            "write": self.write.as_dict(),
            "liveness": self.liveness.as_dict(),
            "monitors": _json_safe(self.monitors),
            "gate_admissible": bool(self.gate_admissible),
            "exclusion_reason": self.exclusion_reason,
            "seeds_n": int(self.seeds_n),
            "flags": _json_safe(self.flags),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RaceCell":
        d = dict(d)
        tl = dict(d.pop("trajectory_launder", {}) or {})
        wr = dict(d.pop("write", {}) or {})
        lv = dict(d.pop("liveness", {}) or {})
        by = dict(d.pop("bytes", {}) or {})
        d.pop("dividend", None)
        d.pop("plus_zero_byte_substitute_margin", None)
        known = {f.name for f in fields(cls)}
        kw = {k: (_NAN if v is None and k in _FLOAT_FIELDS else v)
              for k, v in d.items() if k in known}
        return cls(
            trajectory_launder=TrajectoryLaunder(**_only(TrajectoryLaunder, tl)),
            write=WriteRecord(**_only(WriteRecord, wr)),
            liveness=Liveness(**_only(Liveness, lv)),
            bytes=ByteLedger(**_only(ByteLedger, by)),
            **kw,
        )


_FLOAT_FIELDS = {"full", "settle_deleted_launder", "same_keys_null", "blank",
                 "plus_zero_byte_substitute", "dividend_se"}


def make_cell(route: str, arm: str, family: str, seed: int, metric_name: str,
              **kw) -> RaceCell:
    """Convenience constructor: sub-records may be passed as plain dicts.

    ``make_cell("route1", "traj_write", "aggregate", 0, "neg_mae",
    full=..., settle_deleted_launder=..., write={"steps": 300, ...})``
    """
    for name, klass in (("trajectory_launder", TrajectoryLaunder),
                        ("write", WriteRecord), ("liveness", Liveness),
                        ("bytes", ByteLedger)):
        v = kw.get(name)
        if isinstance(v, dict):
            kw[name] = klass(**_only(klass, v))
    cell = RaceCell(route=route, arm=arm, family=family, seed=int(seed),
                    metric_name=metric_name, **kw)
    cell.resolve_admissibility()
    return cell


# ==========================================================================
# the scorer
# ==========================================================================
@dataclass
class FamilyVerdict:
    """The aggregate over seeds for one ``(route, arm, family)`` — and the
    grade the charter's decision rule is read off.

    ⛔ ``admissible_coverage`` is a **first-class reported number**, at the top
    of the results, never buried: admissibility filtering quietly gutting
    coverage until B' can never fire is the named failure mode.
    """

    route: str
    arm: str
    family: str
    metric_name: str
    n_cells: int
    n_admissible: int
    seeds: List[int]
    dividend_mean: float
    dividend_sd: float
    dividend_se: float
    clears_two_se: bool
    substitute_margin_mean: float
    liveness_passed: bool
    perturbing_anchor: bool
    grade: str
    excluded: List[Dict[str, Any]] = field(default_factory=list)
    bytes_matched: Optional[bool] = None
    bytes_ratio: float = _NAN
    trajectory_launder_fired: bool = False
    escalated: bool = False

    @property
    def admissible_coverage(self) -> float:
        return float(self.n_admissible) / max(int(self.n_cells), 1)

    @property
    def votes_le_zero(self) -> bool:
        """Does this family/arm cast the gate's <=0 vote."""
        return self.grade == GRADE_LE_ZERO_VOTE

    def as_dict(self) -> dict:
        return {"route": self.route, "arm": self.arm, "family": self.family,
                "metric_name": self.metric_name,
                "n_cells": self.n_cells, "n_admissible": self.n_admissible,
                "admissible_coverage": _j(self.admissible_coverage),
                "seeds": list(self.seeds),
                "dividend_mean": _j(self.dividend_mean),
                "dividend_sd": _j(self.dividend_sd),
                "dividend_se": _j(self.dividend_se),
                "clears_two_se": bool(self.clears_two_se),
                "substitute_margin_mean": _j(self.substitute_margin_mean),
                "liveness_passed": bool(self.liveness_passed),
                "perturbing_anchor": bool(self.perturbing_anchor),
                "grade": self.grade, "excluded": _json_safe(self.excluded),
                "bytes_matched": self.bytes_matched, "bytes_ratio": _j(self.bytes_ratio),
                "trajectory_launder_fired": bool(self.trajectory_launder_fired),
                "escalated": bool(self.escalated),
                "votes_le_zero": self.votes_le_zero}


def score_family(cells: Sequence[RaceCell], *, escalated: bool = False
                 ) -> FamilyVerdict:
    """Aggregate one ``(route, arm, family)`` group over seeds and grade it.

    The arithmetic is the charter's, verbatim: **sample sd (``ddof=1``),
    ``SE = sd/sqrt(n)``, "clears 0 beyond 2 SE" means ``mean - 2*SE > 0``.**

    Grading:

    * **zero admissible cells** -> :data:`GRADE_ABSTAIN` (after the one bounded
      escalation; ``escalated=True`` records that it was taken). Abstain casts
      **no** vote: it neither blocks B' nor supports "proceed".
    * ``clears_two_se`` and substitute margin ``> 0`` -> :data:`GRADE_PROCEED`.
    * ``clears_two_se`` but substitute margin ``<= 0`` -> :data:`GRADE_WEAK_PROCEED`
      (charter §A6, pre-registered before adjudication).
    * does not clear, **and the term was inert with no perturbing anchor in the
      grid** -> :data:`GRADE_UNDER_POWERED` — *not* a <=0 vote.
    * otherwise -> :data:`GRADE_LE_ZERO_VOTE`.
    """
    cells = list(cells)
    if not cells:
        raise ValueError("score_family needs at least one cell")
    route = cells[0].route
    arm = cells[0].arm
    family = cells[0].family
    metric = cells[0].metric_name
    for c in cells:
        c.resolve_admissibility()

    ok = [c for c in cells if c.gate_admissible]
    excluded = [{"seed": int(c.seed), "arm": c.arm, "family": c.family,
                 "reason": c.exclusion_reason or "unspecified",
                 "dividend": _j(c.dividend)}
                for c in cells if not c.gate_admissible]

    divs = np.asarray([c.dividend for c in ok], dtype=float)
    divs = divs[np.isfinite(divs)]
    n = int(divs.size)
    mean = float(np.mean(divs)) if n else _NAN
    sd = float(np.std(divs, ddof=1)) if n > 1 else _NAN
    se = float(sd / math.sqrt(n)) if n > 1 and _finite(sd) else _NAN
    clears = bool(_finite(mean) and _finite(se) and (mean - 2.0 * se) > 0.0)

    margins = np.asarray([c.substitute_margin for c in ok], dtype=float)
    margins = margins[np.isfinite(margins)]
    margin_mean = float(np.mean(margins)) if margins.size else _NAN

    live = bool(ok) and any(c.liveness.passed for c in ok)
    anchor = any(c.liveness.perturbing_anchor for c in cells)
    fired = any(c.trajectory_launder.fired() for c in cells)

    if not ok:
        grade = GRADE_ABSTAIN
    elif clears:
        grade = (GRADE_PROCEED if (_finite(margin_mean) and margin_mean > 0.0)
                 else GRADE_WEAK_PROCEED)
    elif not live and not anchor:
        grade = GRADE_UNDER_POWERED
    else:
        grade = GRADE_LE_ZERO_VOTE

    ratios = [c.bytes.ratio for c in cells if c.bytes.launder > 0]
    return FamilyVerdict(
        route=route, arm=arm, family=family, metric_name=metric,
        n_cells=len(cells), n_admissible=len(ok),
        seeds=sorted(int(c.seed) for c in ok),
        dividend_mean=mean, dividend_sd=sd, dividend_se=se, clears_two_se=clears,
        substitute_margin_mean=margin_mean, liveness_passed=live,
        perturbing_anchor=anchor, grade=grade, excluded=excluded,
        bytes_matched=(all(c.bytes.matched for c in cells if c.bytes.launder > 0)
                       if ratios else None),
        bytes_ratio=(float(np.median(ratios)) if ratios else _NAN),
        trajectory_launder_fired=fired, escalated=bool(escalated),
    )


def score_card(cells: Iterable[RaceCell], *,
               escalated: Optional[Sequence[str]] = None) -> List[FamilyVerdict]:
    """Group cells by ``(route, arm, family)`` and score each group.

    ``escalated`` is the list of ``"family"`` (or ``"route/arm/family"``) keys
    that took the one bounded budget escalation (gate ruling (i)).
    """
    esc = set(escalated or ())
    groups: Dict[tuple, List[RaceCell]] = {}
    for c in cells:
        groups.setdefault((c.route, c.arm, c.family), []).append(c)
    out = []
    for k in sorted(groups):
        key_forms = {k[2], "/".join(k)}
        out.append(score_family(groups[k], escalated=bool(key_forms & esc)))
    return out


def coverage_table(verdicts: Sequence[FamilyVerdict]) -> Dict[str, dict]:
    """⛔ **Admissible-cell coverage per family** — report this at the TOP of the
    results, not inside a JSON field."""
    out: Dict[str, dict] = {}
    for v in verdicts:
        e = out.setdefault(v.family, {"n_cells": 0, "n_admissible": 0,
                                      "arms": [], "reasons": []})
        e["n_cells"] += v.n_cells
        e["n_admissible"] += v.n_admissible
        e["arms"].append(v.arm)
        e["reasons"].extend(x["reason"] for x in v.excluded)
    for e in out.values():
        e["coverage"] = float(e["n_admissible"]) / max(e["n_cells"], 1)
        e["arms"] = sorted(set(e["arms"]))
        e["reasons"] = sorted(set(e["reasons"]))
    return out


def gate_summary(verdicts: Sequence[FamilyVerdict]) -> dict:
    """The C2W2 gate's **arithmetic** (the verdict itself is the Hub's).

    >  if writes that explicitly ask trajectories/paths to carry information
    >  still give dividend <= 0 on every family (multi-seed, substitute-audited,
    >  both routes) => B' activates [...]. If any family clears 0 beyond 2 SE =>
    >  C2W3 proceeds as planned.

    Returns the families that cleared, those casting a <=0 vote, and — the part
    that keeps the gate honest — those that **abstain** or ran an
    **under-powered grid** and therefore cast no vote at all.
    """
    cleared = [v for v in verdicts if v.grade in (GRADE_PROCEED, GRADE_WEAK_PROCEED)]
    le_zero = [v for v in verdicts if v.grade == GRADE_LE_ZERO_VOTE]
    abstain = [v for v in verdicts if v.grade == GRADE_ABSTAIN]
    weak = [v for v in verdicts if v.grade == GRADE_WEAK_PROCEED]
    under = [v for v in verdicts if v.grade == GRADE_UNDER_POWERED]
    return {
        "schema_version": RACE_SCHEMA_VERSION,
        "n_verdicts": len(verdicts),
        "cleared_two_se": [f"{v.route}/{v.arm}/{v.family}" for v in cleared],
        "weak_proceed": [f"{v.route}/{v.arm}/{v.family}" for v in weak],
        "le_zero_votes": [f"{v.route}/{v.arm}/{v.family}" for v in le_zero],
        "abstained": [f"{v.route}/{v.arm}/{v.family}" for v in abstain],
        "under_powered_grids": [f"{v.route}/{v.arm}/{v.family}" for v in under],
        "any_family_clears": bool(cleared),
        "routes_present": sorted({v.route for v in verdicts}),
        "coverage": coverage_table(verdicts),
        # deliberately NOT a verdict on the program: the Hub applies the gate.
        "note": "arithmetic only; the C2W2 gate is applied by the Hub",
    }


# ==========================================================================
# reporting / io
# ==========================================================================
def verdicts_to_markdown(verdicts: Sequence[FamilyVerdict]) -> str:
    """The race card as the table that goes in the report."""
    rows = [
        "| route | arm | family | metric | dividend ± SE | 2SE? | +0 B margin | "
        "admissible | live | anchor | bytes | grade |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        d = (f"{v.dividend_mean:+.4f} ± {v.dividend_se:.4f}"
             if _finite(v.dividend_se) else
             (f"{v.dividend_mean:+.4f} (1 seed)" if _finite(v.dividend_mean) else "n/a"))
        b = ("n/a" if not _finite(v.bytes_ratio)
             else f"{v.bytes_ratio:.2f}x{'' if v.bytes_matched else ' (unmatched)'}")
        rows.append(
            f"| {v.route} | {v.arm} | {v.family} | {v.metric_name} | {d} | "
            f"{'yes' if v.clears_two_se else 'no'} | "
            f"{v.substitute_margin_mean:+.4f} | "
            f"{v.n_admissible}/{v.n_cells} ({v.admissible_coverage:.0%}) | "
            f"{'yes' if v.liveness_passed else 'no'} | "
            f"{'yes' if v.perturbing_anchor else 'NO'} | {b} | **{v.grade}** |"
        )
    excl = [(v, x) for v in verdicts for x in v.excluded]
    if excl:
        rows += ["", "**Excluded cells (every one, with its reason — silent "
                 "filtering is forbidden):**", "",
                 "| route/arm/family | seed | reason |", "|---|---|---|"]
        for v, x in excl:
            rows.append(f"| {v.route}/{v.arm}/{v.family} | {x['seed']} | {x['reason']} |")
    return "\n".join(rows)


def save_cells(path, cells: Iterable[RaceCell]) -> Path:
    """Write the race card as a JSON list (``allow_nan=False``: NaN -> null)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([c.as_dict() for c in cells], indent=2,
                            allow_nan=False))
    return p


def load_cells(path) -> List[RaceCell]:
    """Read a race card written by :func:`save_cells` (or JSONL)."""
    txt = Path(path).read_text().strip()
    if txt.startswith("["):
        recs = json.loads(txt)
    else:
        recs = [json.loads(ln) for ln in txt.splitlines() if ln.strip()]
    return [RaceCell.from_dict(r) for r in recs]


# ==========================================================================
# helpers
# ==========================================================================
def _norm_route(route: str) -> str:
    r = str(route).strip().lower()
    for known in ROUTES:
        if r == known or r.startswith(known + "_") or r.startswith(known + "-"):
            return known
    if r in ("r1", "1", "write_objective", "traj_write_objective"):
        return "route1"
    if r in ("r2", "2", "shell_atoms", "ssb_shell_atoms"):
        return "route2"
    raise ValueError(
        f"unknown route {route!r}; both C2W2 branches must emit one of {ROUTES} "
        "(aliases 'route2_shell_atoms', 'r2', ... are normalised) so the two "
        "race cards are comparable"
    )


def _finite(x) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _sub(a, b) -> float:
    return float(a) - float(b) if (_finite(a) and _finite(b)) else _NAN


def _j(x):
    """JSON-safe scalar: NaN/inf -> ``None`` (``json`` has no NaN literal)."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return v if math.isfinite(v) else None


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (str, bool, np.bool_)) or x is None:
        return bool(x) if isinstance(x, np.bool_) else x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _j(x)
    return str(x)


def _only(klass, d: dict) -> dict:
    known = {f.name for f in fields(klass)}
    return {k: v for k, v in d.items() if k in known}


__all__ = [
    "RACE_SCHEMA_VERSION", "ROUTES", "BYTE_RATIO_ARCHITECTURAL",
    "GRADE_PROCEED", "GRADE_WEAK_PROCEED", "GRADE_LE_ZERO_VOTE",
    "GRADE_ABSTAIN", "GRADE_UNDER_POWERED",
    "TrajectoryLaunder", "WriteRecord", "Liveness", "ByteLedger",
    "RaceCell", "make_cell", "FamilyVerdict",
    "score_family", "score_card", "coverage_table", "gate_summary",
    "verdicts_to_markdown", "save_cells", "load_cells",
]
