"""Assemble `verdicts.json` -- the acceptance artifact of task `hepa-lh-verdict`.

One record per HEPA-suite dataset row carrying, per task §2:
  verdict | the audit's four measurements | the tripwire number at every rung |
  the information-age slice | the PREREG §5 prediction and whether it held |
  the artifact path behind every number.

GATES (task §3), checked on disk, never judged:
  A  PREREG_AMENDMENT.md  -> the reading of kill-rule clause (b). §6 needs it.
  B  sweep.json (+ preds) -> HEPA's own reproduction. §5's h-AUROC and §6 need it.
Whatever is gated is emitted as an explicit "GATED_ON_B" record with the reason,
never as a null, a zero or a guess (PREREG §6).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

OUT = Path(os.environ.get("TRIPWIRE_OUT",
                          Path(__file__).resolve().parents[1]))
RUNGS_DIR = OUT / "rungs"

# Rung sets. Sweep-covered rows first, then the two priced-but-not-swept rows.
RUNG_SET: Dict[str, List[int]] = {
    "ETTm1": [512, 2048, 8192],
    "BeijingAQ": [512, 2048],
    "PSM": [100, 512, 2048, 8192],
    "Weather": [512, 2048],
    "GECCO": [512, 2048, 8192],
    "MBA": [512],
}
SWEPT = ["ETTm1", "BeijingAQ", "PSM", "Weather"]      # task §4 / §5 / §6
ALL_ROWS = ["FD001", "FD002", "FD003", "FD004", "PSM", "MBA", "GECCO",
            "BATADAL", "ETTm1", "Weather", "BeijingAQ"]

# Verdicts the sweep never gets to touch (task §6, wave-1 reasons carried forward)
CARRY_FORWARD = {
    "FD001": ("DROP_ON_AUDIT", "cov_512 = 0.000 vs the 0.10 bar (audit criterion 1)"),
    "FD002": ("DROP_ON_AUDIT", "cov_512 = 0.000 vs the 0.10 bar (audit criterion 1)"),
    "FD003": ("DROP_ON_AUDIT", "cov_512 = 0.010 vs the 0.10 bar (audit criterion 1)"),
    "FD004": ("DROP_ON_AUDIT", "cov_512 = 0.004 vs the 0.10 bar (audit criterion 1)"),
    "MBA": ("DROP", "max_available_context = 512 = the reference rung; no rung above "
                    "the reference exists, so kill-rule clause (a) has nothing to compare"),
    "BATADAL": ("DROP", "max_available_context = 100 < the 512 reference rung; no rung "
                        "above the reference exists (and n = 5 events, 2 in ft_test)"),
    "GECCO": ("HOLD", "ft_test contains exactly ONE event onset: clause (a) is not "
                      "measurable. A power failure, not a negative result."),
}
PREREG_S5 = {
    "FD001": "DROP on audit", "FD002": "DROP on audit", "FD003": "DROP on audit",
    "FD004": "DROP on audit", "PSM": "DROP", "MBA": "DROP", "GECCO": "KEEP",
    "BATADAL": "HOLD", "ETTm1": "DROP", "Weather": "DROP", "BeijingAQ": "KEEP",
}
TOL = 0.02          # PREREG §4(b1) and AMENDMENT (b2) share one tolerance
SCR = OUT.parents[1] / "scratch" / "hepa-lh-verdict"


def _tw_number(res: Dict, budget: str) -> Optional[Dict]:
    """The tripwire number at one byte-ledger reading (default = the primary)."""
    if budget == res["tripwire"]["budget_used"]:
        t = res["tripwire"]
        return {"tripwire_h_auroc": t["tripwire_h_auroc"],
                "source": t["tripwire_source"],
                "best_exemplar_arm": t["best_exemplar_arm"],
                "best_exemplar_h_auroc": t["best_exemplar_h_auroc_test"],
                "best_trivial_arm": t["best_trivial_arm"],
                "best_trivial_h_auroc": t["best_trivial_h_auroc_test"],
                "budget_bytes": t["budget_bytes"]}
    arms = res["exemplar_store"][budget]["arms"]
    best, bv, bt = None, -1.0, float("nan")
    for name, e in arms.items():
        if e.get("status") != "OK" or not e.get("prereg_compliant"):
            continue
        v = e["val"]["h_auroc"] if e["val"]["regime"] == "deterministic" else e["val"]["mean"]
        if v > bv:
            best, bv = name, v
            bt = e["test"]["h_auroc"] if e["test"]["regime"] == "deterministic" else e["test"]["mean"]
    triv = {n: e["h_auroc"] for n, e in res["trivial_baselines"].items()
            if e.get("status") == "OK"}
    bn = max(triv, key=lambda n: triv[n]) if triv else None
    tv = triv[bn] if bn else float("nan")
    return {"tripwire_h_auroc": float(max(bt, tv)),
            "source": "exemplar_store" if bt >= tv else "trivial",
            "best_exemplar_arm": best, "best_exemplar_h_auroc": bt,
            "best_trivial_arm": bn, "best_trivial_h_auroc": tv,
            "budget_bytes": res["exemplar_store"][budget]["budget_bytes"]}


_SIGNED = None


def signed_trivials(ds: str) -> Optional[Dict]:
    """DIAGNOSTIC: the trivial arms with a direction bit chosen on ft_val.

    Not the frozen wave-1 tripwire number -- reported alongside it. See
    `.claude/scratch/hepa-lh-verdict/trivials_signed.py` for why suppressing it
    would be suppressing a result (PSM's last-event-age arm is h-AUROC 0.0766
    under the frozen sign and 0.9234 under the val-selected one).
    """
    global _SIGNED
    if _SIGNED is None:
        f = SCR / "trivials_signed.json"
        _SIGNED = json.load(open(f)) if f.exists() else {}
    return _SIGNED.get(ds)


def rung_record(ds: str, rung: int) -> Dict:
    path = RUNGS_DIR / f"{ds}_L{rung}.json"
    if not path.exists():
        return {"status": "NOT_RUN", "reason": f"{path.name} absent", "artifact": str(path)}
    res = json.load(open(path))
    st = res["exemplar_store"]
    primary = res["tripwire"]["budget_used"]
    arms = st[primary]["arms"]
    per_arm = {}
    for name, e in arms.items():
        if e.get("status") != "OK":
            per_arm[name] = {"status": e.get("status"), "reason": e.get("reason")}
            continue
        t = e["test"]
        per_arm[name] = {
            "prereg_compliant": e["prereg_compliant"],
            "regime": t["regime"],
            "h_auroc": t.get("h_auroc", t.get("mean")),
            "std": t.get("std"), "range": t.get("range"), "per_seed": t.get("per_seed"),
            "n_exemplars": e["store"]["n_exemplars_realised"],
            "n_pool": e["store"]["n_pool"],
            "feature_dim": e["store"]["feature_dim"],
            "budget_utilisation": e["store"]["budget_utilisation"],
            "k_selected_per_seed": e["k_selected_per_seed"],
        }
    return {
        "status": "OK",
        "artifact": str(path),
        "context_len": res["context_len"],
        "n_tokens": res["byte_ledger"]["n_tokens"],
        "budget_bytes_all_readings": res["byte_ledger"]["budget_bytes"],
        "primary_budget": primary,
        "tripwire": {b: _tw_number(res, b) for b in st.keys()},
        "tripwire_signed_diagnostic": _signed_tw(res, primary, ds),
        "trivial_baselines": {
            n: ({"status": e["status"], "reason": e.get("reason")}
                if e.get("status") != "OK"
                else {"h_auroc": e["h_auroc"], "regime": e["regime"],
                      "h_auroc_if_direction_flipped": 1.0 - e["h_auroc"]})
            for n, e in res["trivial_baselines"].items()},
        "exemplar_arms_primary_budget": per_arm,
        "effective_context_test": res["splits"]["effective_context_test"],
        "n_windows": res["splits"]["n_windows"],
        "n_nondegenerate_horizons_test": res["splits"]["n_nondegenerate_horizons_test"],
        "runtime_seconds": res.get("runtime_seconds"),
        "seeds": res["seeds"],
        "norm_mode": res["protocol"]["norm_mode"],
    }


def _signed_tw(res: Dict, primary: str, ds: str) -> Dict:
    """max(best PREREG-compliant store at this rung, best val-sign-selected trivial)."""
    sg = signed_trivials(ds)
    frozen = _tw_number(res, primary)
    if not sg:
        return {"status": "NOT_COMPUTED"}
    best_triv = sg["best_trivial_val_selected_sign_test"]
    store = frozen["best_exemplar_h_auroc"]
    return {
        "status": "OK",
        "note": "DIAGNOSTIC, not the frozen wave-1 tripwire. One bit (the ranking "
                "direction) selected on ft_val, never on test; zero parameters, no "
                "window, no encoder. It can only make the tripwire stronger and a "
                "DROP easier, which is the admissible direction under the AMENDMENT.",
        "best_trivial_val_selected_sign": best_triv,
        "best_exemplar_h_auroc": store,
        "tripwire_signed_h_auroc": float(max(store, best_triv)),
        "arms": sg["arms"],
    }


def audit_four_measurements(a: Dict) -> Dict:
    return {
        "m1_history_available": {
            "segment_len": a["segment_len"], "event_history": a["event_history"],
            "coverage_gt": a["coverage_gt"],
            "coverage_gt_conservative_minusK": a["coverage_gt_conservative_minusK"],
            "max_available_context": a["max_available_context"],
            "max_useful_context": a["max_useful_context"],
            "full_context_frac_of_positive_cuts": a["full_context_frac_of_positive_cuts"],
        },
        "m2_time_to_previous_event": a["time_to_prev_event"],
        "m3_time_to_last_similar_regime": a["time_to_last_similar_regime"],
        "m4_event_counts": {
            "n_events_onsets": a["n_events_onsets"],
            "n_positive_steps": a["n_positive_steps"],
            "n_events_ft_test": a["by_split"]["ft_test"]["n_events"],
            "by_split": a["by_split"],
        },
        "audit_verdict": a["verdict"],
        "artifact": "audit.json (results-analyst, task hepa-lh-audit)",
    }


def kill_rule(ds: str, rungs: Dict[str, Dict], sweep: Optional[Dict]) -> Dict:
    """PREREG §4 with clause (b) as ruled in PREREG_AMENDMENT.md."""
    if sweep is None:
        return {"status": "GATED_ON_B",
                "reason": "sweep.json absent: HEPA's own reproduction does not exist, "
                          "and the AMENDMENT binds clause (b2) to the reproduction, "
                          "never to the README. No clause is scored.",
                "clause_a": "GATED_ON_B", "clause_b1": "GATED_ON_B",
                "clause_b2": "GATED_ON_B"}
    return score_kill_rule(ds, rungs, sweep)


DEVICE_PRIMARY = "cpu"      # sweep.json _meta: the reference family. `mps` uses
                            # different float kernels; families are never mixed inside
                            # one gap, but a dataset measured wholly on one family is
                            # internally comparable and is scored within it.
_RECOMP = None


def _recomputed(ds: str, rung: int, device: str) -> Optional[Dict]:
    """HEPA h-AUROC re-scored by us from `preds/*.npz`.

    Used ONLY where `sweep.json`'s published snapshot predates a cell that the
    sweep has already written predictions for. It is HEPA's own dumped CDF
    surface scored with HEPA's own metric under a key-based alignment, and it
    agrees with `sweep.json`'s float32 value to <= 1.5e-6 on every cell where
    both exist (the sweep's own stated fp16 tolerance is 2e-4). Every such cell
    is labelled `source = "recomputed_from_preds"` so no reader mistakes it for
    a number the sweep published.
    """
    global _RECOMP
    if _RECOMP is None:
        f = SCR / "hepa_slice_eval.json"
        _RECOMP = json.load(open(f)) if f.exists() else {}
    cell = (_RECOMP.get(ds) or {}).get(f"{rung}_{device}")
    if not cell or cell.get("all", {}).get("status"):
        return None
    a = cell["all"]
    return {"mean": a["mean"], "spread": a["range"], "n_seeds": a["n_seeds"],
            "per_seed": a["per_seed"]}


def _families_at(sweep: Dict, ds: str, rung: int) -> Dict[str, Dict]:
    """Every device family the sweep has for this cell, plus preds-recomputed fills."""
    row = (sweep.get("sweep") or {}).get(ds) or {}
    cell = row.get(str(rung)) if isinstance(row.get(str(rung)), dict) else {}
    out: Dict[str, Dict] = {}
    for fam in ("cpu", "mps"):
        c = cell.get(fam)
        if isinstance(c, dict) and c.get("h_auroc_mean") is not None:
            out[fam] = {"mean": float(c["h_auroc_mean"]),
                        "spread": c.get("h_auroc_spread_max_minus_min"),
                        "n_seeds": int(c.get("n_seeds") or 0),
                        "status": c.get("status"), "device": fam,
                        "source": "sweep.json",
                        "realised_context_mean": c.get("realised_context_mean"),
                        "frac_windows_at_full_rung": c.get("frac_windows_at_full_rung")}
        rc = _recomputed(ds, rung, fam)
        if rc and (fam not in out or rc["n_seeds"] > out[fam]["n_seeds"]):
            out[fam] = {**rc, "device": fam, "status": "RUN",
                        "source": "recomputed_from_preds",
                        "note": "sweep.json's snapshot lagged this cell; re-scored from "
                                "the sweep's own preds/*.npz (agrees to <=1.5e-6 wherever "
                                "both exist)"}
    for v in out.values():
        v["usable_for_scoring"] = bool(v["n_seeds"] >= 3)
    return out


def _hepa_at(sweep: Dict, ds: str, rung: int) -> Dict:
    """The HEPA value used for the per-rung level test, clause (b2).

    (b2) is a level test at one rung, so families are not mixed inside it. Where
    more than one family has a full 3 seeds, we take **the most favourable value
    to HEPA** -- the conservative direction for an instrument built to kill a
    KEEP, since it makes the tripwire's job harder, not easier.
    """
    fams = _families_at(sweep, ds, rung)
    ok = {f: v for f, v in fams.items() if v["usable_for_scoring"]}
    pool = ok or fams
    if not pool:
        return {"mean": None, "n_seeds": 0, "device": None, "source": None,
                "status": "NOT RUN", "usable_for_scoring": False, "families": {}}
    best = max(pool.values(), key=lambda v: v["mean"])
    return {**best, "families": {f: {k: v[k] for k in ("mean", "n_seeds", "status", "source")}
                                 for f, v in fams.items()},
            "selection": ("best-of-family at 3 seeds (most favourable to HEPA)"
                          if len(ok) > 1 else "only family with 3 seeds"
                          if ok else "no family reached 3 seeds -- reported, not scored")}


def score_kill_rule(ds: str, rungs: Dict[str, Dict], sweep: Dict) -> Dict:
    rlist = list(RUNG_SET[ds])
    hepa = {r: _hepa_at(sweep, ds, r) for r in rlist}
    hepa = {r: (v or {"mean": None, "status": "ABSENT_FROM_SWEEP_JSON",
                      "n_seeds": 0, "usable_for_scoring": False}) for r, v in hepa.items()}
    # Clause (a) compares two rungs, so both come from ONE float family. PREREG
    # §4(a) says "the LONGEST available context", so we pick the family that
    # REACHES FURTHEST (cpu breaks ties) rather than defaulting to cpu and
    # silently truncating the sweep at a shorter rung.
    per_fam: Dict[str, List[int]] = {}
    for r in rlist:
        for f, v in _families_at(sweep, ds, r).items():
            if v["usable_for_scoring"]:
                per_fam.setdefault(f, []).append(r)
    cand = {f: sorted(rs) for f, rs in per_fam.items() if 512 in rs and len(rs) > 1}
    scoring_family = (max(cand, key=lambda f: (max(cand[f]), f == DEVICE_PRIMARY))
                      if cand else None)
    usable = cand.get(scoring_family, [])
    fam_vals = {r: _families_at(sweep, ds, r).get(scoring_family) for r in usable}
    reported = [r for r in rlist if hepa[r].get("mean") is not None]

    tw = {r: (rungs[str(r)]["tripwire"][rungs[str(r)]["primary_budget"]]["tripwire_h_auroc"]
              if rungs[str(r)].get("status") == "OK" else None) for r in rlist}
    tws = {r: (rungs[str(r)].get("tripwire_signed_diagnostic", {}).get("tripwire_signed_h_auroc")
               if rungs[str(r)].get("status") == "OK" else None) for r in rlist}

    base = {
        "status": "SCORED",
        "hepa_source": "sweep.json reproduction (AMENDMENT: never REPRODUCIBILITY.md)",
        "hepa_by_rung": {str(r): hepa[r] for r in rlist},
        "tripwire_by_rung": {str(r): tw[r] for r in rlist},
        "tripwire_signed_by_rung": {str(r): tws[r] for r in rlist},
        "scoring_family": scoring_family,
        "scoring_family_values": {str(r): fam_vals[r]["mean"] for r in usable},
        "scoring_family_rule": "clause (a) compares two rungs, so both come from ONE "
                               "float family (cpu preferred); clause (b2) is a per-rung "
                               "level test and uses every reported rung, family labelled",
        "rungs_usable_for_scoring": [str(r) for r in usable],
        "rungs_reported_but_not_scored": [
            str(r) for r in reported if r not in usable],
    }

    if 512 not in usable or len(usable) < 2:
        base.update({
            "status": "NOT_SCORABLE",
            "reason": ("the HEPA reproduction has fewer than two scorable rungs on the "
                       "primary (cpu) family at 3 seeds; PREREG §1 forbids scoring a "
                       "one-seed cell and the sweep's own _meta forbids mixing the mps "
                       "float family into one gap. Reported, never nulled."),
            "clause_a": "NOT_MEASURABLE", "clause_b1": "NOT_MEASURABLE",
            "clause_b2": "NOT_MEASURABLE",
            "verdict": "GATED_ON_B",
        })
        return base

    longest = max(usable)
    h512, hL = fam_vals[512]["mean"], fam_vals[longest]["mean"]
    spread = fam_vals[512]["spread"]
    gain = hL - h512
    # If the only scorable rung IS the reference, clause (a) has nothing to
    # compare and a "gain of 0.0 fails" would be a misleading way to say so.
    a_measurable = longest > 512
    a_pass = bool(a_measurable and spread is not None and gain > spread)

    b1_pass = None
    tw_gain = None
    if tw.get(512) is not None and tw.get(longest) is not None:
        tw_gain = tw[longest] - tw[512]
        b1_pass = bool(not (tw_gain >= gain - TOL))

    margins = {r: (None if (tw[r] is None or hepa[r].get("mean") is None)
                   else hepa[r]["mean"] - tw[r]) for r in reported}
    b2_fail = [r for r, m in margins.items() if m is not None and not (m > TOL)]
    b2_pass = bool(len(b2_fail) == 0 and margins and all(m is not None for m in margins.values()))

    margins_s = {r: (None if (tws[r] is None or hepa[r].get("mean") is None)
                     else hepa[r]["mean"] - tws[r]) for r in reported}
    b2s_fail = [r for r, m in margins_s.items() if m is not None and not (m > TOL)]

    # Does the verdict depend on WHICH byte-ledger reading the Hub calls primary?
    b2_by_budget = {}
    for bud in ("token_only", "residual_stream", "kv_cache"):
        mm = {}
        for r in reported:
            rr = rungs[str(r)]
            cell = (rr.get("tripwire") or {}).get(bud)
            if rr.get("status") != "OK" or cell is None or hepa[r].get("mean") is None:
                mm[r] = None
            else:
                mm[r] = hepa[r]["mean"] - cell["tripwire_h_auroc"]
        fails = [r for r, m in mm.items() if m is not None and not (m > TOL)]
        b2_by_budget[bud] = {
            "pass": bool(not fails and mm and all(m is not None for m in mm.values())),
            "margins_by_rung": {str(r): mm[r] for r in reported},
            "fails_at_rungs": [str(r) for r in fails]}

    # AMENDMENT: a tripwire tie-or-better at ANY rung is a DROP, "regardless of the
    # shape of the context curve" -- so a b2 failure decides even when (a) cannot be
    # measured. Only if b2 holds and (a) is unmeasurable is the row a power failure.
    if not a_measurable:
        keep = False
        outcome = "DROP" if not b2_pass else "HOLD"
    else:
        keep = bool(a_pass and b1_pass and b2_pass)
        outcome = "KEEP" if keep else "DROP"
    base.update({
        "clause_b2_budget_robustness": {
            "note": "clause (b2) recomputed at each of the three byte-ledger readings, "
                    "so the verdict can be checked against a Hub re-ruling of the "
                    "primary budget without a re-run",
            "by_budget": b2_by_budget,
            "verdict_invariant_to_budget_ruling": len(
                {v["pass"] for v in b2_by_budget.values()}) == 1},
        "longest_scorable_rung": longest,
        "seed_spread_at_512_max_minus_min": spread,
        "hepa_gain_longest_minus_512": gain,
        "clause_a": ({"pass": a_pass,
                      "test": "hepa(longest scorable rung) - hepa(512) > raw seed spread at 512",
                      "value": gain, "threshold": spread}
                     if a_measurable else
                     {"pass": None, "status": "NOT_MEASURABLE",
                      "reason": "the longest SCORABLE rung is the 512 reference itself "
                                "(higher rungs are NOT RUN, mps-only, or fewer than 3 "
                                "seeds); there is no gain to compare to the seed spread. "
                                "A power failure, not a zero gain.",
                      "seed_spread_at_512": spread}),
        "clause_b1": {"pass": b1_pass,
                      "test": "the same-byte store does NOT reproduce HEPA's 512->longest "
                              "gain to within 0.02 (a comparison of GAINS)",
                      "tripwire_gain": tw_gain, "hepa_gain": gain, "tolerance": TOL},
        "clause_b2": {"pass": b2_pass,
                      "test": "at EVERY reported rung, hepa_mean - tripwire_mean > 0.02; "
                              "a tie fails (PREREG_AMENDMENT.md)",
                      "margins_by_rung": {str(r): margins[r] for r in reported},
                      "fails_at_rungs": [str(r) for r in b2_fail], "tolerance": TOL},
        "clause_b2_signed_sensitivity": {
            "pass": bool(len(b2s_fail) == 0 and margins_s
                         and all(m is not None for m in margins_s.values())),
            "test": "DIAGNOSTIC: b2 against the val-sign-selected trivial arms; the "
                    "frozen b2 above is the scored clause",
            "margins_by_rung": {str(r): margins_s[r] for r in reported},
            "fails_at_rungs": [str(r) for r in b2s_fail]},
        "verdict": outcome,
    })
    return base


def build() -> Dict:
    audit = json.load(open(OUT / "audit.json"))
    sweep_path = OUT / "sweep.json"
    sweep = json.load(open(sweep_path)) if sweep_path.exists() else None
    slices = json.load(open(OUT.parents[1] / "scratch" / "hepa-lh-verdict"
                            / "slice_report.json"))
    gate_a = (OUT / "PREREG_AMENDMENT.md").exists()

    rows = {}
    for ds in ALL_ROWS:
        rung_recs = {str(r): rung_record(ds, r) for r in RUNG_SET.get(ds, [])}
        rec: Dict[str, object] = {
            "dataset": ds,
            "swept_by_hepa_lh_sweep": ds in SWEPT,
            "rung_set": RUNG_SET.get(ds, []),
            "audit_measurements": audit_four_measurements(audit[ds]),
            "rungs": rung_recs,
            "information_age_slice": slices.get(ds, {
                "status": "NOT_COMPUTED",
                "reason": "row is DROP_ON_AUDIT / no rung above the reference; the "
                          "slice would price a dataset that is already out"}),
            "prereg_s5": {"prediction": PREREG_S5[ds]},
        }
        if ds in CARRY_FORWARD:
            v, why = CARRY_FORWARD[ds]
            rec["verdict"] = v
            rec["verdict_basis"] = why
            rec["kill_rule"] = {
                "status": "NOT_APPLICABLE",
                "reason": why,
                "clause_a": "NOT_MEASURABLE", "clause_b1": "NOT_MEASURABLE",
                "clause_b2": ("NOT_SCORED: needs the HEPA reproduction (sweep.json), "
                              "which does not cover this row"),
            }
        else:
            kr = kill_rule(ds, rung_recs, sweep)
            rec["kill_rule"] = kr
            rec["verdict"] = kr.get("verdict", "GATED_ON_B")
            rec["verdict_basis"] = kr.get("reason", "PREREG §4 + AMENDMENT clause (b)")
        pred = PREREG_S5[ds]
        v = rec["verdict"]
        if v == "GATED_ON_B":
            held = "NOT_SETTLED"
        else:
            held = (pred.upper().startswith(v.split("_")[0])
                    or (pred == "DROP on audit" and v == "DROP_ON_AUDIT"))
        rec["prereg_s5"]["held"] = held
        rec["prereg_s5"]["verdict"] = v
        rows[ds] = rec

    n_keep = sum(1 for r in rows.values() if r["verdict"] == "KEEP")
    return {
        "schema": "hepa-lh-verdict/1",
        "task": "hepa-lh-verdict",
        "gates": {
            "A_PREREG_AMENDMENT.md": "PRESENT" if gate_a else "ABSENT",
            "B_sweep.json": "PRESENT" if sweep is not None else "ABSENT",
            "B_preds_dir": "PRESENT" if (OUT / "preds").exists() else "ABSENT",
        },
        "kill_rule_reading": {
            "source": "PREREG.md §4 as amended by PREREG_AMENDMENT.md",
            "clause_a": "h-AUROC at the longest available rung beats the 512 rung by "
                        "more than the raw seed spread (max - min) at 512",
            "clause_b1": "the matched-byte store does NOT close the longest-minus-512 "
                         "gap to within 0.02 h-AUROC (read as a comparison of GAINS: "
                         "the store fails the clause if its own 512->longest gain "
                         "reaches within 0.02 of HEPA's)",
            "clause_b2": "at EVERY rung, HEPA_mean - tripwire_mean > 0.02; a tie fails",
            "keep_iff": "a AND b1 AND b2",
            "hepa_number": "the reproduction in sweep.json, never REPRODUCIBILITY.md",
        },
        "primary_byte_budget": "residual_stream (all three readings reported at every rung)",
        "aggregate": {
            "n_KEEP": n_keep,
            "n_DROP": sum(1 for r in rows.values() if r["verdict"].startswith("DROP")),
            "n_HOLD": sum(1 for r in rows.values() if r["verdict"] == "HOLD"),
            "n_GATED": sum(1 for r in rows.values() if r["verdict"] == "GATED_ON_B"),
            "prereg_s5_clause": ("PREREG §5: fewer than 2 KEEP kills the long-horizon "
                                 "section entirely"),
        },
        "rows": rows,
    }


if __name__ == "__main__":
    v = build()
    p = OUT / "verdicts.json"
    json.dump(v, open(p, "w"), indent=1)
    print(f"wrote {p}")
    print("gates:", v["gates"])
    for ds, r in v["rows"].items():
        tw = {k: (None if x.get("status") != "OK" else
                  round(x["tripwire"][x["primary_budget"]]["tripwire_h_auroc"], 4))
              for k, x in r["rungs"].items()}
        print(f"  {ds:10s} {r['verdict']:14s} tripwire={tw}")
    print("aggregate:", v["aggregate"])
