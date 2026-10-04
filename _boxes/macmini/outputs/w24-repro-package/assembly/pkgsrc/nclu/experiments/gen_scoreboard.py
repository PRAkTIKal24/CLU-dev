"""gen scoreboard."""

from __future__ import annotations
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import numpy as np
from nclu.data.gen_splits import write_json_with_checksum
from nclu.data.gen_value_holdout import ROW_EVIDENCE_RANKING, VALUE_HOLDOUT_SCOPE


class ScoreboardError(AssertionError):
    pass


ROW_IDS = (
    "r1_seen",
    "r2_unseen_valid",
    "r3_comp_2way",
    "r4_comp_nway",
    "r5_comp_refusal",
    "r6_invalid_refusal",
    "r7_deletion",
)
ROW_SPEC: dict[str, dict] = {
    "r1_seen": {
        "index": 1,
        "requests": "a stored value-combination (ON-grid, every value in the fit pool)",
        "scored_as": "judge accuracy + valid tuples",
        "metric": "accuracy",
        "query_set": "row1_seen",
        "note": "⭐⭐: also the ACCEPT-LEG of every refusal row (5, 6 and 7's deleted half). It is structurally matched to row5_held_out — same record-key set, term count, id shape and per-kind count — so the pair differs only in whether the store holds the value.",
    },
    "r2_unseen_valid": {
        "index": 2,
        "requests": "a hue BETWEEN two stored values (in-basin interpolation)",
        "scored_as": "judge accuracy + valid tuples",
        "metric": "accuracy",
        "query_set": "row2_unseen",
        "note": "⛔: ACCEPT-ONLY. Row 2 is `an earlier study`'s target and keeps its own scoring; it is ⛔ consumed by NO refusal row. Pairing it against row 5 was the leak: off-grid by definition vs on-grid by definition.",
    },
    "r3_comp_2way": {
        "index": 3,
        "requests": "seen values, unseen PAIR",
        "scored_as": "distinct valid novel tuples",
        "metric": "novelty",
        "query_set": "pair-* cells",
    },
    "r4_comp_nway": {
        "index": 4,
        "requests": "seen values, unseen n-way conjunction",
        "scored_as": "distinct valid novel tuples",
        "metric": "novelty",
        "query_set": "nway-* cells",
    },
    "r5_comp_refusal": {
        "index": 5,
        "requests": "a value held out ENTIRELY (on-grid, never stored)",
        "scored_as": "the refusal pair (reject_invalid, accept_unseen_valid)",
        "metric": "refusal",
        "query_set": "row5_held_out",
    },
    "r6_invalid_refusal": {
        "index": 6,
        "requests": "a value OUTSIDE the grid's range (or a duplicated factor)",
        "scored_as": "the refusal pair (reject_invalid, accept_unseen_valid)",
        "metric": "refusal",
        "query_set": "c_invalid",
        "note": "⛔ build_c_invalid's format. There is no second invalid format. ⛔⛔: SANITY CHECK ONLY (`evidential: false`) — out-of-range is decidable from the grid alone, so this row can never be non-free.",
    },
    "r7_deletion": {
        "index": 7,
        "requests": "post-deletion: rows 3-4 for survivors, row 5 for the deleted value",
        "scored_as": "both, ⛔ with NO retraining",
        "metric": "deletion",
        "query_set": "nway-* (survivors) + row5_held_out (deleted)",
    },
}
REFUSAL_ROWS = ("r5_comp_refusal", "r6_invalid_refusal")
ACCEPT_LEG_ROW = "r1_seen"
NON_EVIDENTIAL_ROWS = tuple(
    sorted((k for k, v in ROW_EVIDENCE_RANKING.items() if not v["evidential"]))
)
for _rid, _rank in ROW_EVIDENCE_RANKING.items():
    ROW_SPEC[_rid]["evidential"] = bool(_rank["evidential"])
    ROW_SPEC[_rid]["evidence_rank"] = int(_rank["rank"])
    ROW_SPEC[_rid]["evidence_note"] = _rank["why"]
for _rid in ROW_IDS:
    ROW_SPEC[_rid].setdefault("evidential", True)
del _rid
PRIMARY_CONTROL = "basis_head"
CONTROL_ARMS = (PRIMARY_CONTROL, "knn_proto")
LAUNDERING_ARM = "grid_syntax_rule"


@dataclass(frozen=True)
class ScoreboardConfig:
    byte_budget: int = 524288
    byte_match_tolerance: int = 0
    required_arms: tuple = CONTROL_ARMS
    require_laundering_control: bool = True
    decoration_margin: float = 0.0
    refusal_rows: tuple = REFUSAL_ROWS
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping] = None) -> "ScoreboardConfig":
        mapping = dict(mapping or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"ScoreboardConfig: unknown key(s) {unknown}; known: {sorted(known)}")
        if "required_arms" in mapping:
            mapping["required_arms"] = tuple(mapping["required_arms"])
        if "refusal_rows" in mapping:
            mapping["refusal_rows"] = tuple(mapping["refusal_rows"])
        return cls(**mapping)

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["required_arms"] = list(out["required_arms"])
        out["refusal_rows"] = list(out["refusal_rows"])
        return out


def make_outcome(
    accept: Sequence[bool],
    valid: Sequence[bool],
    tuples: Optional[np.ndarray],
    threshold: float,
    setting: Any,
    byte_footprint: Mapping,
    request_ids: Optional[Sequence[str]] = None,
) -> dict:
    accept = np.asarray(accept, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if accept.shape != valid.shape:
        raise ScoreboardError(
            f"accept {accept.shape} and valid {valid.shape} disagree on the request count"
        )
    if np.any(valid & ~accept):
        raise ScoreboardError(
            f"{int(np.sum(valid & ~accept))} request(s) are marked VALID while refused — a refused request produces no output for the judge to score"
        )
    if tuples is not None:
        tuples = np.asarray(tuples)
        if tuples.ndim != 2 or len(tuples) != accept.size:
            raise ScoreboardError(f"tuples must be (N, F) with N={accept.size}, got {tuples.shape}")
    led = dict(byte_footprint)
    if not isinstance(led.get("deployed_bytes"), (int, np.integer)):
        raise ScoreboardError(
            "byte_footprint needs an integer 'deployed_bytes' — ⛔ a row without a measured footprint is not a table row"
        )
    return {
        "accept": accept,
        "valid": valid,
        "tuples": tuples,
        "threshold": float(threshold),
        "setting": setting,
        "byte_footprint": {**led, "deployed_bytes": int(led["deployed_bytes"])},
        "request_ids": list(request_ids) if request_ids is not None else None,
        "n_requests": int(accept.size),
    }


def _distinct(tuples: np.ndarray, keep: np.ndarray) -> set:
    if tuples is None:
        return set()
    sel = tuples[np.asarray(keep, dtype=bool)]
    return {tuple((int(x) for x in row)) for row in sel}


def score_accuracy_row(outcome: Mapping) -> dict:
    accept, valid = (outcome["accept"], outcome["valid"])
    n = int(accept.size)
    distinct = _distinct(outcome["tuples"], valid)
    return {
        "metric": "accuracy",
        "n_requests": n,
        "n_accepted": int(accept.sum()),
        "n_valid": int(valid.sum()),
        "judge_accuracy": float(valid.sum() / n) if n else 0.0,
        "accuracy_given_accepted": float(valid.sum() / accept.sum()) if accept.sum() else 0.0,
        "distinct_valid_tuples": len(distinct),
        "score": float(valid.sum() / n) if n else 0.0,
    }


def score_novelty_row(outcome: Mapping, seen_tuples: Sequence[Sequence[int]]) -> dict:
    seen = {tuple((int(x) for x in t)) for t in seen_tuples}
    valid = outcome["valid"]
    produced = _distinct(outcome["tuples"], valid)
    novel = produced - seen
    n = int(valid.size)
    return {
        "metric": "novelty",
        "n_requests": n,
        "n_accepted": int(outcome["accept"].sum()),
        "n_valid": int(valid.sum()),
        "distinct_valid_tuples": len(produced),
        "distinct_valid_novel_tuples": len(novel),
        "n_seen_tuples_supplied": len(seen),
        "replayed_seen_tuples": len(produced & seen),
        "score": float(len(novel)),
    }


def score_refusal_row(
    outcome: Mapping,
    accept_leg_outcome: Mapping,
    row_id: str = "",
    accept_leg_row: str = ACCEPT_LEG_ROW,
) -> dict:
    reject = float(1.0 - outcome["accept"].mean()) if outcome["accept"].size else 0.0
    a2 = accept_leg_outcome["accept"]
    accept_unseen = float(a2.mean()) if a2.size else 0.0
    leg = accept_leg_row
    faults: list[str] = []
    if outcome["threshold"] != accept_leg_outcome["threshold"]:
        faults.append(
            f"threshold {outcome['threshold']!r} != {leg}'s {accept_leg_outcome['threshold']!r} — accept_unseen_valid was NOT measured at the same threshold, so the pair is not a pair"
        )
    if json.dumps(outcome["setting"], sort_keys=True, default=str) != json.dumps(
        accept_leg_outcome["setting"], sort_keys=True, default=str
    ):
        faults.append(
            f"setting fingerprint differs from {leg}'s — the two halves of the pair come from different configurations"
        )
    if (
        outcome["byte_footprint"]["deployed_bytes"]
        != accept_leg_outcome["byte_footprint"]["deployed_bytes"]
    ):
        faults.append(
            f"deployed_bytes {outcome['byte_footprint']['deployed_bytes']} != {leg}'s {accept_leg_outcome['byte_footprint']['deployed_bytes']} — a different object answered the two halves of the pair"
        )
    gmean = float(np.sqrt(max(reject, 0.0) * max(accept_unseen, 0.0)))
    return {
        "metric": "refusal",
        "row": row_id,
        "n_requests": int(outcome["accept"].size),
        "reject_invalid": reject,
        "accept_unseen_valid": accept_unseen,
        "accept_unseen_valid_source": leg,
        "accept_leg_row": leg,
        "accept_leg_note": "⭐: the accept-leg is ROW 1 — a STORED, on-grid request. Both halves are well-formed on-grid requests; they differ only in whether the store holds the value. ⛔ It was row 2 (off-grid by definition), and that pairing was measured free at (1.0000, 1.0000) with zero deployed bytes.",
        "n_accept_leg_requests": int(a2.size),
        "pair": [reject, accept_unseen],
        "refusal_gmean": gmean,
        "same_setting": not faults,
        "faults": faults,
        "score": 0.0 if faults else gmean,
        "note": "⛔ the PAIR, never one number: an arm that refuses everything scores 0 (reject=1, accept=0) and an arm that accepts everything scores 0",
    }


def score_deletion_row(
    survivor_outcome: Mapping,
    deleted_outcome: Mapping,
    accept_leg_outcome: Mapping,
    seen_tuples: Sequence[Sequence[int]],
    pre_deletion_refusal: Optional[Mapping] = None,
    fit_fingerprint: Optional[tuple] = None,
) -> dict:
    survivors = score_novelty_row(survivor_outcome, seen_tuples)
    deleted = score_refusal_row(deleted_outcome, accept_leg_outcome, row_id="r7_deletion")
    faults = list(deleted["faults"])
    if fit_fingerprint is not None:
        pre, post = fit_fingerprint
        if pre != post:
            faults.append(
                f"fit fingerprint moved ({pre!r} -> {post!r}) — the arm RETRAINED, and row 7 explicitly forbids it"
            )
    delta = None
    if pre_deletion_refusal is not None:
        before = float(1.0 - np.asarray(pre_deletion_refusal["accept"]).mean())
        delta = float(deleted["reject_invalid"] - before)
    return {
        "metric": "deletion",
        "survivors": survivors,
        "deleted": deleted,
        "delta_reject": delta,
        "no_retraining_verified": fit_fingerprint is not None and (not faults),
        "faults": faults,
        "both_hold": bool(
            not faults and survivors["distinct_valid_novel_tuples"] > 0 and (deleted["score"] > 0.0)
        ),
        "score": 0.0
        if faults or survivors["distinct_valid_novel_tuples"] == 0
        else float(deleted["score"]),
        "note": "⭐ delta_reject is the non-free half of the refusal claim: deleting a value from the store does not move the factor grid, so the §4 no-model rule's delta_reject is exactly 0.0 by construction.",
    }


def assert_byte_footprint(
    outcomes: Mapping[str, Mapping[str, Mapping]], cfg: ScoreboardConfig, primary_arm: str
) -> dict:
    table: dict[str, dict] = {}
    problems: list[str] = []
    for arm, rows in outcomes.items():
        per_row = {}
        seen_bytes = set()
        for row_id, out in rows.items():
            led = dict(out["byte_footprint"])
            b = int(led["deployed_bytes"])
            seen_bytes.add(b)
            matched = abs(b - int(cfg.byte_budget)) <= int(cfg.byte_match_tolerance)
            note = str(led.get("budget_note") or "").strip()
            if not matched and (not note):
                problems.append(
                    f"{arm}/{row_id}: deployed {b:,} B against a {cfg.byte_budget:,} B budget and carries NO budget_note — an unmatched footprint must be DECLARED, not silent"
                )
            if not matched and arm in (primary_arm, PRIMARY_CONTROL):
                problems.append(
                    f"{arm}/{row_id}: deployed {b:,} B != the {cfg.byte_budget:,} B budget (tolerance {cfg.byte_match_tolerance} B). ⭐ {primary_arm} vs {PRIMARY_CONTROL} is THE like-for-like; an unmatched footprint there is a different experiment, not a footnote"
                )
            per_row[row_id] = {
                "deployed_bytes": b,
                "budget_bytes": int(led.get("budget_bytes", cfg.byte_budget)),
                "byte_matched": bool(matched),
                "multiple_of_budget": float(b / cfg.byte_budget) if cfg.byte_budget else None,
                "what": led.get("what"),
                "budget_note": led.get("budget_note"),
            }
        if len(seen_bytes) > 1:
            problems.append(
                f"{arm}: deployed_bytes is not constant across rows ({sorted(seen_bytes)}) — the same deployed object must answer every row, or the table is seven experiments wearing one arm's name"
            )
        table[arm] = per_row
    if problems:
        raise ScoreboardError("byte-footprint audit FAILED:\n  " + "\n  ".join(problems))
    return table


def _stub_footprint(cfg: ScoreboardConfig, what: str) -> dict:
    return {
        "deployed_bytes": int(cfg.byte_budget),
        "budget_bytes": int(cfg.byte_budget),
        "what": what,
    }


def refuse_everything_outcome(
    n: int, cfg: Optional[ScoreboardConfig] = None, n_factors: int = 6
) -> dict:
    cfg = cfg or ScoreboardConfig()
    return make_outcome(
        accept=np.zeros(n, bool),
        valid=np.zeros(n, bool),
        tuples=-np.ones((n, n_factors), np.int64),
        threshold=0.0,
        setting={"arm": "refuse_everything"},
        byte_footprint=_stub_footprint(cfg, "⛔ nothing — the stub deploys no store"),
    )


def accept_everything_outcome(
    n: int, cfg: Optional[ScoreboardConfig] = None, n_factors: int = 6
) -> dict:
    """The mirror falsifier: accepts every request, so the pair is ``(0.0, 1.0)`` ⇒ 0."""
    cfg = cfg or ScoreboardConfig()
    return make_outcome(
        accept=np.ones(n, bool),
        valid=np.zeros(n, bool),
        tuples=-np.ones((n, n_factors), np.int64),
        threshold=0.0,
        setting={"arm": "accept_everything"},
        byte_footprint=_stub_footprint(cfg, "⛔ nothing — the stub deploys no store"),
    )


def grid_syntax_accept(request: Sequence[Mapping], grid) -> bool:
    by_factor: dict[int, set] = {}
    any_interior_offgrid = False
    for term in request:
        k = int(term["factor_index"])
        v = float(term["value"])
        by_factor.setdefault(k, set()).add(v)
        vals = grid.values[k]
        if v < vals[0] or v > vals[-1]:
            return False
        if not np.any(vals == v):
            any_interior_offgrid = True
    if any((len(s) > 1 for s in by_factor.values())):
        return False
    return any_interior_offgrid


def grid_syntax_outcome(
    records: Sequence[Mapping], grid, cfg: Optional[ScoreboardConfig] = None
) -> dict:
    cfg = cfg or ScoreboardConfig()
    accept = np.array([grid_syntax_accept(r["request"], grid) for r in records], dtype=bool)
    return make_outcome(
        accept=accept,
        valid=np.zeros(accept.size, bool),
        tuples=-np.ones((accept.size, grid.n_factors), np.int64),
        threshold=0.0,
        setting={"arm": LAUNDERING_ARM, "rule": "offgrid-in-range & no-duplicate"},
        byte_footprint={
            "deployed_bytes": 0,
            "budget_bytes": int(cfg.byte_budget),
            "what": "⛔ the factor grid's realised ranges. No store, no features.",
            "budget_note": "⛔ DECLARED: this arm deploys ZERO bytes of store. It is not a competitor for rows 1-4 (it produces no image); it exists so no refusal-row number ships without the no-model rule's score beside it.",
        },
        request_ids=[r.get("id") for r in records],
    )


def verdict_vs_control(arm_score: float, control_score: float, margin: float) -> dict:
    delta = float(arm_score - control_score)
    beats = delta > float(margin)
    return {
        "delta_vs_basis_head": delta,
        "margin_declared_in_advance": float(margin),
        "beats_control": bool(beats),
        "verdict": "beats basis+head beyond the declared margin"
        if beats
        else "⛔ A CLU mechanism that ties basis+head is decoration and is reported as such, in those words.",
    }


def sanity_check_verdict(
    row_id: str, arm_score: Optional[float], control_score: Optional[float]
) -> dict:
    rank = ROW_EVIDENCE_RANKING.get(row_id, {})
    return {
        "evidential": False,
        "verdict": f"⛔ SANITY CHECK ONLY — {row_id} is NOT evidence for the store. "
        + str(rank.get("why", "")),
        "delta_vs_basis_head": float(arm_score - control_score)
        if arm_score is not None and control_score is not None
        else None,
        "delta_note": "⚠ Reported so the degenerate-arm guard is visible (refuse-everything → 0, accept-everything → 0). ⛔ It is NOT a margin over a control and must not be quoted as one.",
        "quotable_as": "degenerate-arm guard; scoreboard plumbing check",
        "not_quotable_as": "evidence that the store computes a refusal barrier, at any margin",
    }


def build_scoreboard(
    primary_arm: str,
    outcomes: Mapping[str, Mapping[str, Mapping]],
    seen_tuples: Sequence[Sequence[int]],
    cfg: Optional[ScoreboardConfig] = None,
    deletion: Optional[Mapping[str, Mapping]] = None,
    provenance: Optional[Mapping] = None,
) -> dict:
    cfg = cfg or ScoreboardConfig()
    required = list(dict.fromkeys([primary_arm, *cfg.required_arms]))
    if cfg.require_laundering_control and LAUNDERING_ARM not in outcomes:
        raise ScoreboardError(
            f"⛔ the laundering control {LAUNDERING_ARM!r} is missing. The refusal rows are winnable by a no-model syntactic rule (see the module docstring §4); a refusal number without that rule's score beside it means nothing. Set require_laundering_control=False only to say so out loud."
        )
    missing = [a for a in required if a not in outcomes]
    if missing:
        raise ScoreboardError(
            f"missing arm(s) {missing}. ⭐ {PRIMARY_CONTROL} is the like-for-like and knn_proto is the second row; both are required on EVERY row."
        )
    for arm in outcomes:
        gaps = [r for r in ROW_IDS if r not in outcomes[arm]]
        if gaps and arm in required:
            raise ScoreboardError(f"{arm}: missing row(s) {gaps} of the seven")
    footprint = assert_byte_footprint(
        {a: {r: o for r, o in rows.items() if r in ROW_IDS} for a, rows in outcomes.items()},
        cfg,
        primary_arm,
    )
    rows_out: dict[str, dict] = {}
    for row_id in ROW_IDS:
        spec = ROW_SPEC[row_id]
        per_arm: dict[str, dict] = {}
        for arm, rows in outcomes.items():
            out = rows.get(row_id)
            if out is None:
                per_arm[arm] = {"score": None, "note": "⛔ NOT RUN"}
                continue
            if spec["metric"] == "accuracy":
                per_arm[arm] = score_accuracy_row(out)
            elif spec["metric"] == "novelty":
                per_arm[arm] = score_novelty_row(out, seen_tuples)
            elif spec["metric"] == "refusal":
                per_arm[arm] = score_refusal_row(out, rows[ACCEPT_LEG_ROW], row_id=row_id)
            elif spec["metric"] == "deletion":
                d = (deletion or {}).get(arm)
                if d is None:
                    per_arm[arm] = {
                        "metric": "deletion",
                        "score": None,
                        "note": "⛔ NOT RUN — row 7 needs a post-deletion measurement and the pre-deletion refusal it is compared against",
                    }
                else:
                    per_arm[arm] = score_deletion_row(
                        d["survivor"],
                        d["deleted"],
                        rows[ACCEPT_LEG_ROW],
                        seen_tuples,
                        pre_deletion_refusal=d.get("pre"),
                        fit_fingerprint=d.get("fit_fingerprint"),
                    )
            per_arm[arm]["byte_footprint"] = footprint[arm][row_id]
        control = per_arm.get(PRIMARY_CONTROL, {}).get("score")
        arm_score = per_arm.get(primary_arm, {}).get("score")
        if row_id in NON_EVIDENTIAL_ROWS:
            verdict = sanity_check_verdict(row_id, arm_score, control)
        elif arm_score is not None and control is not None:
            verdict = verdict_vs_control(arm_score, control, cfg.decoration_margin)
        else:
            verdict = {"verdict": "⛔ NOT SCORED — arm or control row missing"}
        rows_out[row_id] = {
            **spec,
            "row_id": row_id,
            "value_holdout_scope": VALUE_HOLDOUT_SCOPE,
            "arms": per_arm,
            "verdict": verdict,
        }
    return {
        "schema": {
            "version": 1,
            "description": "The seven-row taxonomy as columns of one table, every arm and both controls on every row, byte footprint asserted per row.",
        },
        "value_holdout_scope": VALUE_HOLDOUT_SCOPE,
        "primary_arm": primary_arm,
        "row_evidence_ranking": ROW_EVIDENCE_RANKING,
        "accept_leg_row": ACCEPT_LEG_ROW,
        "change_1": "⭐ Row 5's accept-leg is ROW 1 (a stored, on-grid value), ⛔ not row 2. Row 2 is accept-only and is consumed by no refusal row. Row 6 is `evidential: false` — a sanity check, never evidence for the store. Row 7's delta_reject is the programme's strongest instrument: deleting a value does not move the factor grid, so a no-model rule's delta is 0.0.",
        "controls": {
            "like_for_like": PRIMARY_CONTROL,
            "second_row": "knn_proto",
            "laundering_control": LAUNDERING_ARM,
            "note": "⭐ basis+head (the round-2 object) is the FIRST control and the like-for-like; knn_proto is the second row; the grid-syntax rule is the no-model laundering control on the refusal rows.",
        },
        "config": cfg.to_dict(),
        "arms": sorted(outcomes),
        "rows": rows_out,
        "byte_footprint": footprint,
        "provenance": dict(provenance or {}),
    }


def write_scoreboard(table: Mapping, path: str | Path) -> tuple[Path, str]:
    """Write the table as checksummed JSON; return ``(path, sha256)``."""
    return write_json_with_checksum(dict(table), path)
