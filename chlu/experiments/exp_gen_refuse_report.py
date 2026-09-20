"""w5 ``hub/refuse`` -- the report stage.  Scores what ``exp_gen_refuse`` measured.

⛔ Every refusal number in here is printed with **the 8-byte rule's number beside it**, on
every row, without exception, because the target file requires exactly that and because a
refusal number quoted alone is not evidence about a store.

The scoring is the frozen harness's, not this module's: ``gen_scoreboard.score_refusal_row``
for the pair (accept-leg = ROW 1, faults to 0 unless threshold + setting + deployed_bytes
match across the pair) and ``gen_scoreboard.score_novelty_row`` for the survivor half of the
deletion row.  This module assembles outcomes and tabulates; ⛔ it does not define a metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .exp_gen_factor_wells import _jdump, load_condition, seen_tuple_mask
from .exp_gen_refuse import (
    PINS, QUERY_ROWS, RULES_GAP, RULES_PRIMARY, SEEDS, bitmap8_bytes,
    bitmap8_bits_changed_by_a_conjunction_deletion, pair_table_bytes, scored_corners,
    seed_dir,
)
from .gen_scoreboard import (
    make_outcome, score_novelty_row, score_refusal_row,
)

#: Deployed bytes per refuser.  ⭐ The refusal pair FAULTS TO 0 unless the two halves agree
#: on this, so it is declared once and used everywhere.
LEDGER = {
    "field_geom": 524_288,
    "field_geom_gap": 524_288,
    "field_landscape_removed": 524_288,
    "field_landscape_shuffled": 524_288,
    "bitmap8": 8,
    "pair_presence": 166,
    "pair_deletion": 174,
}

BUDGET_NOTE = {
    "field_geom": "the ATOMS, matched to the 524,288 B pin. The address book, the coupling "
                  "table and the partition are the INTERFACE and are printed beside it, "
                  "never netted into it (gen_field.field_byte_ledger).",
    "field_geom_gap": "as field_geom.",
    "field_landscape_removed": "⛔ charter §C.2 control: the same bytes, zeroed.",
    "field_landscape_shuffled": "⛔ charter §C.2 control: the same bytes, permuted.",
    "bitmap8": "⭐ THE BAR. One bit per (factor, value); 57 terms -> 8 bytes packed. "
               "⛔ It deploys no store and produces no image.",
    "pair_presence": "the joint pair table, 1,322 cells at one bit -> 166 B packed. "
                     "⛔ It deploys no store and produces no image.",
    "pair_deletion": "⚠ THE DECLARED RESIDUAL: the 8-byte rule + a SEPARATE 166 B "
                     "pair-deletion table. ⛔ It deploys no store and produces no image.",
}

REFUSERS = tuple(LEDGER)


def _setting(name: str) -> dict:
    rule = {"field_geom": list(RULES_PRIMARY), "field_geom_gap": list(RULES_GAP),
            "field_landscape_removed": list(RULES_PRIMARY),
            "field_landscape_shuffled": list(RULES_PRIMARY),
            "bitmap8": ["value-presence bitmap", "bracket check"],
            "pair_presence": ["joint pair-presence table", "bracket check"],
            "pair_deletion": ["value-presence bitmap", "separate pair-deletion table",
                              "bracket check"]}[name]
    return {"arm": name, "rule": rule,
            "basin_live_tau": float(PINS["field_basin_live_tau"]),
            "gap_tau": (float(PINS["field_gap_tau"]) if name == "field_geom_gap" else None)}


def _outcome(name: str, blk: dict, ids=None) -> dict:
    return make_outcome(
        accept=np.asarray(blk["accept"], bool), valid=np.asarray(blk["valid"], bool),
        tuples=np.asarray(blk["tuples"]), threshold=0.0, setting=_setting(name),
        byte_ledger={"deployed_bytes": LEDGER[name], "budget_bytes": 524_288,
                     "what": BUDGET_NOTE[name]},
        request_ids=ids)


# ---------------------------------------------------------------------------
# 1. The query rows
# ---------------------------------------------------------------------------


def score_queries(out: Path, log: Callable = print) -> dict:
    """Rows 1, 2, 5 and 6, every refuser, every seed, with the 8-byte rule beside."""
    per_seed = {}
    for seed in SEEDS:
        p = seed_dir(out, seed) / "queries.npz"
        if not p.exists():
            log(f"  ⛔ seed {seed}: queries.npz absent -- DECLARED NOT RUN, never a null")
            continue
        arms = np.load(p, allow_pickle=True)["arms"].item()
        meta = json.loads((seed_dir(out, seed) / "queries.json").read_text("utf-8"))
        rows = {}
        for name in REFUSERS:
            if name not in arms:
                continue
            oc = {rid: _outcome(name, arms[name][rid], arms[name][rid].get("ids"))
                  for rid, _ in QUERY_ROWS}
            leg = oc["r1_seen"]
            rows[name] = {
                "r1_seen_accept": float(oc["r1_seen"]["accept"].mean()),
                "r1_seen_valid": float(oc["r1_seen"]["valid"].mean()),
                "r2_accept": float(oc["r2_unseen_valid"]["accept"].mean()),
                "r2_valid": float(oc["r2_unseen_valid"]["valid"].mean()),
            }
            for rid in ("r5_comp_refusal", "r6_invalid_refusal"):
                rows[name][rid] = score_refusal_row(oc[rid], leg, row_id=rid)
        # the degenerate guards, on the same request counts
        n5 = int(arms["bitmap8"]["r5_comp_refusal"]["accept"].size)
        n1 = int(arms["bitmap8"]["r1_seen"]["accept"].size)
        rows["refuse_everything"] = {"r5_comp_refusal": _degenerate(n5, n1, False),
                                     "r6_invalid_refusal": _degenerate(n5, n1, False)}
        rows["accept_everything"] = {"r5_comp_refusal": _degenerate(n5, n1, True),
                                     "r6_invalid_refusal": _degenerate(n5, n1, True)}
        per_seed[seed] = {"rows": rows, "meta": meta}
        log(f"  seed {seed}: scored {len(rows)} refusers on rows 1/2/5/6")
    return per_seed


def _degenerate(n_row: int, n_leg: int, accept_all: bool) -> dict:
    """⛔ The falsifiers, scored by the same function: both MUST come out 0."""
    st = {"arm": "accept_everything" if accept_all else "refuse_everything"}
    led = {"deployed_bytes": 0, "budget_bytes": 524_288,
           "what": "⛔ nothing -- the stub deploys no store"}
    mk = lambda n: make_outcome(
        accept=(np.ones if accept_all else np.zeros)(n, bool), valid=np.zeros(n, bool),
        tuples=-np.ones((n, 6), np.int64), threshold=0.0, setting=st, byte_ledger=led)
    return score_refusal_row(mk(n_row), mk(n_leg), row_id="degenerate")


# ---------------------------------------------------------------------------
# 2. The deletion rows -- ⭐ where the claim lives
# ---------------------------------------------------------------------------


def _pair(accept_on_row: np.ndarray, accept_on_leg: np.ndarray) -> dict:
    r = float(1.0 - accept_on_row.mean()) if accept_on_row.size else 0.0
    a = float(accept_on_leg.mean()) if accept_on_leg.size else 0.0
    return {"reject": r, "accept_leg": a, "gmean": float(np.sqrt(max(r, 0.0) * max(a, 0.0)))}


def score_deletion(out: Path, log: Callable = print) -> dict:
    """The HOLE (a conjunction deletion) and the EVICTION (a value deletion)."""
    cond = load_condition()
    grid = cond["grid"]
    seen = [tuple(int(x) for x in r)
            for r in grid.factor_indices_of(cond["train_rows"])]
    per_seed = {}
    for seed in SEEDS:
        p = seed_dir(out, seed) / "deletion.npz"
        if not p.exists():
            log(f"  ⛔ seed {seed}: deletion.npz absent -- DECLARED NOT RUN, never a null")
            continue
        z = np.load(p, allow_pickle=True)
        H, E = z["hole"].item(), z["evict"].item()
        meta = json.loads((seed_dir(out, seed) / "deletion.json").read_text("utf-8"))
        out_seed = {"meta": meta, "hole": {}, "evict": {}}

        # ---- A. the HOLE -------------------------------------------------
        nh = np.asarray(H["names_hole"], bool)
        uc = np.asarray(H["unseen_conj"], bool)
        TC, CV = np.asarray(H["corner_tuples"]), np.asarray(H["corner_values"])
        jv = _judge_valid(CV, TC, grid)
        for name in REFUSERS:
            if name not in H["post"]["corners"]:
                continue
            pre_c = np.asarray(H["pre"]["corners"][name], bool)
            post_c = np.asarray(H["post"]["corners"][name], bool)
            post_1 = np.asarray(H["post"]["row1"][name], bool)
            pre_1 = np.asarray(H["pre"]["row1"][name], bool)
            surv = ~nh
            sv = make_outcome(
                accept=post_c[surv], valid=(jv & post_c)[surv],
                tuples=np.where(post_c[surv][:, None], TC[surv], -1),
                threshold=0.0, setting=_setting(name),
                byte_ledger={"deployed_bytes": LEDGER[name], "budget_bytes": 524_288,
                             "what": BUDGET_NOTE[name]})
            out_seed["hole"][name] = {
                "reject_deleted_pre": float(1.0 - pre_c[nh].mean()),
                "reject_deleted_post": float(1.0 - post_c[nh].mean()),
                "delta_reject": float(pre_c[nh].mean() - post_c[nh].mean()),
                "accept_unseen_valid_pre": float(pre_1.mean()),
                "accept_unseen_valid_post": float(post_1.mean()),
                "accept_unseen_conjunction_pre": float(pre_c[uc].mean()),
                "accept_unseen_conjunction_post": float(post_c[uc].mean()),
                "accept_survivors_post": float(post_c[surv].mean()),
                "pair_frozen": _pair(post_c[nh], post_1),
                "pair_conjunction": _pair(post_c[nh], post_c[uc]),
                "survivors": score_novelty_row(sv, seen),
                "n_deleted": int(nh.sum()), "n_unseen_conj": int(uc.sum()),
                "n_survivors": int(surv.sum()),
            }
        out_seed["hole"]["_proof"] = {
            "byte_assertion": H["byte_assertion"], "no_refit": H["no_refit"],
            "rendered_features_bit_identical": bool(H["rendered_features_bit_identical"]),
            "bitmap8_bits_changed": int(H["bitmap8_bits_changed"]),
            "pair": H["pair"], "coupling_index": int(H["coupling_index"]),
        }
        out_seed["hole"]["_groups"] = {
            k: {nm: {"pre": float(np.asarray(g["pre"][nm], bool).mean()),
                     "post": float(np.asarray(g["post"][nm], bool).mean())}
                for nm in REFUSERS if nm in g["post"]}
            for k, g in H["groups"].items()}

        # ---- B. the EVICTION ---------------------------------------------
        ne = np.asarray(E["names_evict"], bool)
        TCe = np.asarray(E["corner_tuples_post"])
        jve = _judge_valid(CV, TCe, grid)
        for name in REFUSERS:
            if name not in E["post"]["corners"]:
                continue
            pre_c = np.asarray(E["pre"]["corners"][name], bool)
            post_c = np.asarray(E["post"]["corners"][name], bool)
            post_1 = np.asarray(E["post"]["row1"][name], bool)
            surv = ~ne
            sv = make_outcome(
                accept=post_c[surv], valid=(jve & post_c)[surv],
                tuples=np.where(post_c[surv][:, None], TCe[surv], -1),
                threshold=0.0, setting=_setting(name),
                byte_ledger={"deployed_bytes": LEDGER[name], "budget_bytes": 524_288,
                             "what": BUDGET_NOTE[name]})
            out_seed["evict"][name] = {
                "reject_deleted_pre": float(1.0 - pre_c[ne].mean()),
                "reject_deleted_post": float(1.0 - post_c[ne].mean()),
                "delta_reject": float(pre_c[ne].mean() - post_c[ne].mean()),
                "accept_unseen_valid_post": float(post_1.mean()),
                "accept_survivors_post": float(post_c[surv].mean()),
                "pair_frozen": _pair(post_c[ne], post_1),
                "survivors": score_novelty_row(sv, seen),
                "n_deleted": int(ne.sum()), "n_survivors": int(surv.sum()),
            }
        out_seed["evict"]["_proof"] = {
            "byte_assertion": E["byte_assertion"], "no_refit": E["no_refit"],
            "n_items_evicted": int(E["n_items_evicted"]),
            "bitmap8_bits_changed": int(E["bitmap8_bits_changed"]),
            "bitmap8_note": E["bitmap8_note"],
            "read_changed_on_surviving_requests_frac": float(
                E["read_changed_on_surviving_requests_frac"]),
            "read_rel_change_on_surviving_requests": float(
                E["read_rel_change_on_surviving_requests"]),
        }
        per_seed[seed] = out_seed
        log(f"  seed {seed}: deletion scored")
    return per_seed


def _judge_valid(values, tuples, grid) -> np.ndarray:
    from .exp_gen_factor_wells import judge_valid
    return judge_valid(values, tuples, grid)


# ---------------------------------------------------------------------------
# 3. Aggregation and the written table
# ---------------------------------------------------------------------------


def _agg(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None], float)
    if a.size == 0:
        return {"mean": None, "spread": None, "n_seeds": 0, "per_seed": []}
    return {"mean": float(a.mean()), "spread": float(a.max() - a.min()),
            "n_seeds": int(a.size), "per_seed": [float(x) for x in a]}


def score_rivals(out: Path, Q: dict, log: Callable = print) -> dict:
    """Run 3: the prototype table refusing by absence, and kNN density.

    ⛔ Neither can win: rows 5/6 already tie the 8-byte rule at a perfect pair. They are
    here because the target file names them, and because ⭐ both have ``delta_reject =
    0.0000`` on a conjunction deletion for the same exact reason the 8-byte rule does --
    which widens the run-2 claim from "a per-factor gate cannot follow" to "no
    per-factor and no atom-density rule can follow".
    """
    from .exp_gen_refuse import bitmap8_bytes, table_absence_bytes

    per_seed = {}
    for seed in SEEDS:
        p = seed_dir(out, seed) / "rivals.npz"
        if not p.exists():
            log(f"  ⛔ seed {seed}: rivals.npz absent -- DECLARED NOT RUN, never a null")
            continue
        arms = np.load(p, allow_pickle=True)["arms"].item()
        meta = json.loads((seed_dir(out, seed) / "rivals.json").read_text("utf-8"))
        rows = {}
        for name in ("table_absence", "knn_density"):
            if name not in arms:
                continue
            led = {"table_absence": meta["table_absence_bytes"],
                   "knn_density": 524_288}[name]
            oc = {}
            for rid, _ in QUERY_ROWS:
                a = np.asarray(arms[name][rid]["accept"], bool)
                oc[rid] = make_outcome(
                    accept=a, valid=np.zeros(a.size, bool),
                    tuples=-np.ones((a.size, 6), np.int64), threshold=0.0,
                    setting={"arm": name}, byte_ledger={"deployed_bytes": led})
            rows[name] = {
                "deployed_bytes": led,
                "r1_seen_accept": float(oc["r1_seen"]["accept"].mean()),
                "r2_accept": float(oc["r2_unseen_valid"]["accept"].mean()),
            }
            for rid in ("r5_comp_refusal", "r6_invalid_refusal"):
                rows[name][rid] = score_refusal_row(oc[rid], oc["r1_seen"], row_id=rid)
        per_seed[seed] = {"rows": rows, "meta": meta}
        log(f"  seed {seed}: rivals scored")

    # ⭐ the identity check declared in PREREG-3 §4.1, run rather than assumed
    identical = None
    if per_seed and Q:
        checks = []
        for s in per_seed:
            if s not in Q:
                continue
            for rid in ("r5_comp_refusal", "r6_invalid_refusal"):
                a = per_seed[s]["rows"].get("table_absence", {}).get(rid)
                b = Q[s]["rows"].get("bitmap8", {}).get(rid)
                if a and b:
                    checks.append(abs(a["reject_invalid"] - b["reject_invalid"]) < 1e-12)
            a1 = per_seed[s]["rows"].get("table_absence", {}).get("r1_seen_accept")
            b1 = Q[s]["rows"].get("bitmap8", {}).get("r1_seen_accept")
            if a1 is not None and b1 is not None:
                checks.append(abs(a1 - b1) < 1e-12)
        identical = bool(checks) and all(checks)
    return {"per_seed": per_seed, "table_absence_equals_bitmap8": identical}


def build_report(out: Path, log: Callable = print) -> dict:
    cond = load_condition()
    grid = cond["grid"]
    c3, c4 = scored_corners(cond)
    Q = score_queries(out, log=log)
    D = score_deletion(out, log=log)
    R = score_rivals(out, Q, log=log)

    seeds_q = sorted(Q)
    seeds_d = sorted(D)
    rep = {
        "hub": "w5 hub/refuse",
        "condition": PINS["condition"],
        "seeds_queries": seeds_q, "seeds_deletion": seeds_d,
        "seeds_NOT_RUN": {
            "queries": [s for s in SEEDS if s not in seeds_q],
            "deletion": [s for s in SEEDS if s not in seeds_d]},
        "n_corners": {"row3": len(c3), "row4": len(c4), "total": len(c3) + len(c4)},
        "bitmap8_bytes": bitmap8_bytes(grid),
        "pair_table": pair_table_bytes(grid),
        "bitmap8_bits_changed_by_a_conjunction_deletion":
            bitmap8_bits_changed_by_a_conjunction_deletion(),
        "thresholds": {
            "basin_live_tau": float(PINS["field_basin_live_tau"]),
            "gap_tau": float(PINS["field_gap_tau"]),
            "set_on": "the val condition, by the w4 field-store hub, before this hub "
                      "existed and before any invalid row was scored",
            "units": "within-value spread (basin_sigma / well_sigma) -- ⛔ NOT the "
                     "spacing of stored items",
            "plateau": "at tau in {2, 4, 8} the empty basins are EXACTLY the two "
                       "never-held values, on all three seeds",
        },
        "labels": "used ONCE at fit time to align wells and build the address book, and "
                  "at the interface. ⛔ Never in the landscape, the read or the write.",
        "population_footnote": (
            "Every rival in RIVALS-READY.json is fitted on the montero_combination pool, "
            "which CONTAINS both held-out values (3,157 rows at object_hue 0.5; 7,972 at "
            "shape 0.0). On rows 5 and 7 a rival saw the value the field store never did. "
            "⛔ Mechanism can enter the row; the fit cannot. Not counted in either "
            "direction."),
        "per_seed_queries": Q, "per_seed_deletion": D,
        "rivals": R,
    }
    # -- the rival refusers, as a table beside everything else -------------
    rt = {}
    for row in ("r5_comp_refusal", "r6_invalid_refusal"):
        rt[row] = {}
        for name in ("table_absence", "knn_density"):
            got = [R["per_seed"][s]["rows"].get(name, {}).get(row)
                   for s in sorted(R["per_seed"])]
            got = [g for g in got if g]
            if not got:
                continue
            first = sorted(R["per_seed"])[0]
            rt[row][name] = {
                "deployed_bytes": R["per_seed"][first]["rows"][name]["deployed_bytes"],
                "reject_invalid": _agg([g["reject_invalid"] for g in got]),
                "accept_unseen_valid": _agg([g["accept_unseen_valid"] for g in got]),
                "score_gmean": _agg([g["score"] for g in got]),
            }
    rt["r1_seen_accept"] = {
        n: _agg([R["per_seed"][s]["rows"][n]["r1_seen_accept"]
                 for s in sorted(R["per_seed"]) if n in R["per_seed"][s]["rows"]])
        for n in ("table_absence", "knn_density")}
    rt["hole_delta_reject"] = {
        n: _agg([R["per_seed"][s]["meta"]["hole"][n]["delta_reject"]
                 for s in sorted(R["per_seed"])
                 if n in R["per_seed"][s]["meta"].get("hole", {})])
        for n in ("table_absence", "knn_density")}
    rt["hole_accept_unseen_conjunction"] = {
        n: _agg([R["per_seed"][s]["meta"]["hole"][n]["accept_unseen_conjunction_post"]
                 for s in sorted(R["per_seed"])
                 if n in R["per_seed"][s]["meta"].get("hole", {})])
        for n in ("table_absence", "knn_density")}
    rep["TABLE_rivals"] = rt
    rep["lace_energy"] = {
        "NOT_RUN": True,
        "why": "⛔ DECLARED NOT RUN, never a null. Spoke A does not expose LACE as a "
               "callable: its output is a stochastic draw, not a function of the "
               "request, so an energy cannot be evaluated on my requests. Established "
               "mechanically (`load_rival` raises `RivalRefusal`), not asserted.",
        "byte_multiple_over_the_store": 95.1,
    }
    rep["full_rival_set_on_a_refusal_row"] = (
        "⛔ DASH + FOOTNOTE, established mechanically and not asserted. Every learner in "
        "spoke A's set is a GENERATOR: `r(values) -> (features, out_of_range)`. It "
        "answers every in-range request and has no mechanism by which it could decline, "
        "so on a refusal row it IS the `accept_everything` degenerate arm and scores 0 "
        "by construction. Separately, every one of them is fitted on "
        "`montero_combination`, which holds BOTH held-out values, so on rows 5 and 7 it "
        "saw what the field store never did. ⛔ A dash is never a win for us.")

    # -- the aggregate refusal table, 8-byte rule BESIDE every number -------
    tab = {}
    for row in ("r5_comp_refusal", "r6_invalid_refusal"):
        tab[row] = {}
        for name in list(REFUSERS) + ["refuse_everything", "accept_everything"]:
            got = [Q[s]["rows"].get(name, {}).get(row) for s in seeds_q]
            got = [g for g in got if g]
            if not got:
                continue
            tab[row][name] = {
                "deployed_bytes": LEDGER.get(name, 0),
                "reject_invalid": _agg([g["reject_invalid"] for g in got]),
                "accept_unseen_valid": _agg([g["accept_unseen_valid"] for g in got]),
                "score_gmean": _agg([g["score"] for g in got]),
                "faults": sorted({f for g in got for f in g["faults"]}),
            }
    for row in ("r1_seen_accept", "r2_accept"):
        tab[row] = {n: _agg([Q[s]["rows"][n][row] for s in seeds_q
                             if row in Q[s]["rows"].get(n, {})])
                    for n in REFUSERS if any(row in Q[s]["rows"].get(n, {})
                                             for s in seeds_q)}
    rep["TABLE_rows_5_6"] = tab

    dtab = {}
    for which in ("hole", "evict"):
        dtab[which] = {}
        for name in REFUSERS:
            got = [D[s][which].get(name) for s in seeds_d]
            got = [g for g in got if g]
            if not got:
                continue
            dtab[which][name] = {
                "deployed_bytes": LEDGER[name],
                "reject_deleted_post": _agg([g["reject_deleted_post"] for g in got]),
                "delta_reject": _agg([g["delta_reject"] for g in got]),
                "accept_unseen_valid_post": _agg(
                    [g["accept_unseen_valid_post"] for g in got]),
                "accept_survivors_post": _agg([g["accept_survivors_post"] for g in got]),
                "survivor_distinct_valid_novel_tuples": _agg(
                    [g["survivors"]["distinct_valid_novel_tuples"] for g in got]),
                "pair_frozen_gmean": _agg([g["pair_frozen"]["gmean"] for g in got]),
            }
            if which == "hole":
                dtab[which][name]["accept_unseen_conjunction_post"] = _agg(
                    [g["accept_unseen_conjunction_post"] for g in got])
                dtab[which][name]["pair_conjunction_gmean"] = _agg(
                    [g["pair_conjunction"]["gmean"] for g in got])
    rep["TABLE_deletion"] = dtab

    # -- the verdicts, in the target file's own words ----------------------
    rep["VERDICTS"] = _verdicts(rep)
    _jdump(rep, out / "report.json")
    _write_markdown(rep, out / "REPORT.md")
    log(f"  report -> {out / 'report.json'} and {out / 'REPORT.md'}")
    return rep


def _verdicts(rep: dict) -> dict:
    v = {}
    t = rep.get("TABLE_rows_5_6", {})
    for row in ("r5_comp_refusal", "r6_invalid_refusal"):
        f = t.get(row, {}).get("field_geom")
        b = t.get(row, {}).get("bitmap8")
        if not (f and b):
            continue
        fm, bm = f["score_gmean"]["mean"], b["score_gmean"]["mean"]
        tie = fm is not None and bm is not None and abs(fm - bm) < 1e-9
        v[row] = {
            "field_geom_pair": [f["reject_invalid"]["mean"],
                                f["accept_unseen_valid"]["mean"]],
            "bitmap8_pair": [b["reject_invalid"]["mean"], b["accept_unseen_valid"]["mean"]],
            "verdict": (
                "⛔ THE 8-BYTE RULE MATCHES MY PAIR ON THIS ROW ⇒ NO CLAIM. The dial "
                "declaration is falsified here: what was measured is the factor grid, "
                "not the store." if tie else
                ("the field store is ABOVE the 8-byte rule on this row"
                 if (fm or 0) > (bm or 0) else
                 "⛔ the field store is BELOW the 8-byte rule on this row")),
            "evidential": row != "r6_invalid_refusal",
            "note": ("⛔ row 6 is `evidential: false` in "
                     "gen_value_holdout.ROW_EVIDENCE_RANKING and can never be anything "
                     "else: out-of-range is decidable from the factor grid alone."
                     if row == "r6_invalid_refusal" else
                     "row 5's accept-leg is ROW 1 (AMENDMENT 1)."),
        }
    d = rep.get("TABLE_deletion", {}).get("hole", {})
    if d.get("field_geom") and d.get("bitmap8"):
        fg, b8 = d["field_geom"], d["bitmap8"]
        pp = d.get("pair_presence", {})
        pd = d.get("pair_deletion", {})
        v["r7_hole"] = {
            "field_geom": [fg["reject_deleted_post"]["mean"],
                           fg["accept_unseen_conjunction_post"]["mean"]],
            "bitmap8": [b8["reject_deleted_post"]["mean"],
                        b8["accept_unseen_conjunction_post"]["mean"]],
            "bitmap8_delta_reject": b8["delta_reject"]["mean"],
            "bitmap8_bits_changed": 0,
            "pair_presence_accept_unseen_conjunction":
                (pp.get("accept_unseen_conjunction_post") or {}).get("mean"),
            "pair_deletion_reject_deleted":
                (pd.get("reject_deleted_post") or {}).get("mean"),
            "note": (
                "⭐ The contrast is EXACT on one side and MEASURED on the other: the "
                "per-factor gate's delta_reject is 0.0000 because ZERO of its bits change "
                "under a conjunction deletion, and the joint pair-PRESENCE table that can "
                "express the deletion refuses the never-held conjunctions. ⚠ The declared "
                "residual -- a SEPARATE pair-deletion table -- does follow, and is priced "
                "on the table."),
        }
    return v


def _fmt(a: Optional[dict]) -> str:
    if not a or a.get("mean") is None:
        return "—"
    return f"{a['mean']:.4f} ± {a['spread']:.4f}"


def _write_markdown(rep: dict, path: Path) -> None:
    L = []
    A = L.append
    A("# w5 hub/refuse — results")
    A("")
    A(f"Condition **{rep['condition']}** · seeds queries {rep['seeds_queries']}, "
      f"deletion {rep['seeds_deletion']} · cache **generation 2** · "
      f"{rep['n_corners']['total']} scored corners "
      f"({rep['n_corners']['row3']} row-3 + {rep['n_corners']['row4']} row-4).")
    A("")
    A(f"⭐ **Labels:** {rep['labels']}")
    A("")
    A("⛔ **The bar is the 8-byte rule, not the table.** Every number below has "
      f"`bitmap8` ({rep['bitmap8_bytes']} B) printed beside it.")
    A("")
    A("## Rows 5 and 6 — the pair (reject_invalid, accept_unseen_valid), mean ± seed spread")
    A("")
    A("| row | refuser | bytes | reject_invalid | accept_unseen_valid | gmean |")
    A("|---|---|---|---|---|---|")
    for row in ("r5_comp_refusal", "r6_invalid_refusal"):
        for name, r in rep["TABLE_rows_5_6"].get(row, {}).items():
            A(f"| `{row}` | `{name}` | {r['deployed_bytes']:,} | "
              f"{_fmt(r['reject_invalid'])} | {_fmt(r['accept_unseen_valid'])} | "
              f"{_fmt(r['score_gmean'])} |")
    A("")
    A("## Refuse-after-deletion")
    A("")
    for which, title in (("hole", "A. CONJUNCTION deletion (`punch_hole`) — ⭐ THE row"),
                         ("evict", "B. VALUE deletion (`evict_term`)")):
        A(f"### {title}")
        A("")
        cols = ("| refuser | bytes | reject_deleted | Δreject | accept unseen-conj | "
                "accept row1 | survivors: distinct valid novel tuples |")
        if which == "evict":
            cols = ("| refuser | bytes | reject_deleted | Δreject | accept survivors | "
                    "accept row1 | survivors: distinct valid novel tuples |")
        A(cols)
        A("|---|---|---|---|---|---|---|")
        for name, r in rep["TABLE_deletion"].get(which, {}).items():
            third = (r.get("accept_unseen_conjunction_post") if which == "hole"
                     else r.get("accept_survivors_post"))
            A(f"| `{name}` | {r['deployed_bytes']:,} | "
              f"{_fmt(r['reject_deleted_post'])} | {_fmt(r['delta_reject'])} | "
              f"{_fmt(third)} | {_fmt(r['accept_unseen_valid_post'])} | "
              f"{_fmt(r['survivor_distinct_valid_novel_tuples'])} |")
        A("")
    A("## Verdicts")
    A("")
    for k, v in rep["VERDICTS"].items():
        A(f"- **{k}** — {v.get('verdict') or v.get('note')}")
    A("")
    A(f"⚠ **Population footnote (binding):** {rep['population_footnote']}")
    path.write_text("\n".join(L), encoding="utf-8")
