"""w21 spoke/ladder-rivals-32k -- BUILD LADDER-RIVALS-32K.json / TABLES.md / CAPTION.* from disk.

Reads only per-corner score files and fit artefacts.  Every number carries its source file.
The w20 131,072-row set and the EXISTING ladder arms are READ, never refitted or re-scored.

Three rival column sets, each labelled by POOL *and* STEPS (Hub AMENDMENT 2026-09-24):
  A  spoke-A `@ref`     32,256 Montero combination-split rows (holds the held-out values),
                        ~118k params, 5,040 steps              (w12-composition-ladder rows)
  B  w20 data-matched   the rung's 131,072-row pool; DECLARED 20,480 / TENX 204,800 steps
  C  w21 (this spoke)   the store's own 32,768 content rows; DECLARED 5,120 / TENX 51,200
                        steps, + a step-matched column at set B's carried budget
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w21L_lib import FAMILIES, OUT, RUNGS, W20_OUT, jdump, store_pool_n  # noqa: E402

SEEDS = (0, 1, 2)
MARGIN = 1.0
P1_BAR = 1.0
KL = Path("C:/chlu/outputs/w12b-nr6-and-kstar/kstar_ladder")
CL = Path("C:/chlu/outputs/w12-composition-ladder/runs")
SPA = Path("C:/chlu/outputs/spoke-a-rivals/rivals")
STEPMATCH = {"mlp": 204_800, "cont_mlp": 204_800, "lstm": 20_480, "transformer": 20_480}
W20_CARRIED_STAGE = {"mlp": "tenx", "cont_mlp": "tenx", "lstm": "declared", "transformer": "declared"}


def rd(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def rows_of(p):
    j = rd(p)
    if j is None:
        return None
    rows = j["per_corner"]
    return sorted(rows, key=lambda r: r["corner_id"])


def mean_rows(rows):
    return None if rows is None else float(np.mean([r["distinct_valid_novel_tuples"] for r in rows]))


def paired(a_rows, b_rows):
    from chlu.experiments import exp_gen_factor_wells as fw
    return fw.paired_estimand(a_rows, b_rows, "distinct_valid_novel_tuples", MARGIN, "a", "b")


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None,
            "per_seed": xs, "n_seeds": len(xs)}


def pe_short(pe):
    return {"delta": pe["mean_paired_delta"], "ci95": pe["bootstrap_ci95"],
            "won_tied_lost": [pe["corners_arm_wins"], pe["corners_tied"], pe["corners_control_wins"]],
            "sign_test_p": pe["sign_test_p_two_sided"]}


def fit_meta(rung, seed, fam, stage):
    p = OUT / "rivals" / ("rung%s_n32768" % rung) / ("seed%d" % seed) / fam / ("%s.json" % stage)
    j = rd(p)
    if j is None:
        return None
    c = j["curve"]
    step = max(1, len(c) // 20)
    return {"source": str(p), "n_fit_rows": j["n_fit_rows"], "param_count": j["param_count"],
            "epochs": j["epochs"], "optimizer_steps": j["optimizer_steps"], "batch": j["batch"],
            "lr": j.get("stage_lr", j["selected"]["lr"]),
            "weight_decay": j.get("stage_weight_decay", j["selected"]["weight_decay"]),
            "train_mse_first_epoch": c[0]["train_mse"], "train_mse_final_epoch": c[-1]["train_mse"],
            "last_decile_rel_change": j["last_decile_rel_change"],
            "train_mse_curve_every_%d_epochs" % step:
                [round(r["train_mse"], 6) for r in c[::step]] + [round(c[-1]["train_mse"], 6)],
            "fit_seconds": j["train_info"]["seconds"]}


def selection_meta(rung, seed, fam):
    p = OUT / "rivals" / ("rung%s_n32768" % rung) / ("seed%d" % seed) / fam / "selection.json"
    j = rd(p)
    if j is None:
        return None
    return {"source": str(p), "selected": j["selected"], "split": j.get("selection_split"),
            "short_trials": [{"lr": s["lr"], "wd": s["weight_decay"], "val": s["short_val_mse"]}
                             for s in j.get("short", [])],
            "heldout_curve_by_epochs": {str(f["epochs"]): [round(r["val_mse"], 6) for r in f["curve"]
                                                           if "val_mse" in r] for f in j.get("full", [])}}


def my_score(rung, seed, fam, stage):
    p = OUT / "scores" / ("rung%s" % rung) / ("seed%d" % seed) / ("%s__%s@rung%s_n32768.json" % (fam, stage, rung))
    return rows_of(p), str(p)


def w20_score(rung, seed, fam, stage):
    p = W20_OUT / "scores" / ("rung%s" % rung) / ("seed%d" % seed) / ("%s__%s@rung%s.json" % (fam, stage, rung))
    return rows_of(p), str(p)


def spokea_steps(seed, fam):
    j = rd(SPA / ("seed_%d" % seed) / ("%s.json" % fam))
    return None if j is None else int(j["fit"]["steps"])


def rung_block(rung):
    out = {"rung": rung, "store_pool": {}, "store": {}, "setA_spokeA_32256": {}, "setB_w20_131072": {},
           "setC_w21_32768": {}, "crosscheck": {}, "convergence_evidence": {}}
    store = {}
    for s in SEEDS:
        out["store_pool"][s] = store_pool_n(rung, s)
        p = KL / ("%s_seed%d_K32" % (rung, s)) / "rows_single_design_K32.json"
        store[s] = rows_of(p)
        out["store"][s] = {"mean": mean_rows(store[s]), "source": str(p)}
    out["store"]["summary"] = ms([out["store"][s]["mean"] for s in SEEDS])

    # ---- set A: spoke-A @ref, 32,256 Montero rows ------------------------------------
    A = {}
    for fam in FAMILIES:
        per = {}
        for s in SEEDS:
            p = CL / ("%s_seed%d" % (rung, s)) / ("rows_rival_%s.json" % fam)
            r = rows_of(p)
            A[(fam, s)] = r
            per[s] = {"mean": mean_rows(r), "steps": spokea_steps(s, fam), "source": str(p)}
        per["summary"] = ms([per[s]["mean"] for s in SEEDS])
        out["setA_spokeA_32256"][fam] = per

    # ---- set B: w20, 131,072 rows (READ) --------------------------------------------
    B = {}
    for fam in FAMILIES:
        per = {}
        for st in ("declared", "tenx"):
            for s in SEEDS:
                r, p = w20_score(rung, s, fam, st)
                B[(fam, st, s)] = r
                per.setdefault(st, {})[s] = {"mean": mean_rows(r), "source": p if r else None}
            per[st]["summary"] = ms([per[st][s]["mean"] for s in SEEDS])
        per["w20_carried_stage"] = W20_CARRIED_STAGE[fam]
        per["w20_carried_steps"] = 204_800 if W20_CARRIED_STAGE[fam] == "tenx" else 20_480
        out["setB_w20_131072"][fam] = per

    # ---- set C: this spoke, 32,768 store-content rows -------------------------------
    C = {}
    for fam in FAMILIES:
        per = {}
        stages = ["declared", "tenx", "steps%d" % STEPMATCH[fam]] + (["ownpoolsched"] if fam == "mlp" else [])
        for st in stages:
            for s in SEEDS:
                r, p = my_score(rung, s, fam, st)
                C[(fam, st, s)] = r
                fm = fit_meta(rung, s, fam, st)
                per.setdefault(st, {})[s] = {"mean": mean_rows(r), "score_source": p if r else None,
                                            "optimizer_steps": fm and fm["optimizer_steps"],
                                            "n_fit_rows": fm and fm["n_fit_rows"],
                                            "param_count": fm and fm["param_count"]}
                out["convergence_evidence"].setdefault(fam, {}).setdefault(str(s), {})[st] = fm
            per[st]["summary"] = ms([per[st][s]["mean"] for s in SEEDS])
        for s in SEEDS:
            out["convergence_evidence"][fam][str(s)]["selection"] = selection_meta(rung, s, fam)
        # P1 carry rule (w20's bar)
        p1 = {}
        for s in SEEDS:
            d, t = per["declared"][s]["mean"], per["tenx"][s]["mean"]
            if d is None or t is None:
                p1[s] = {"declared": d, "tenx": t, "carried": None, "P1": "NOT RUN"}
                continue
            under = (t - d) > P1_BAR
            p1[s] = {"declared": d, "tenx": t, "tenx_minus_declared": t - d,
                     "P1": "UNDER-TRAINED: TENX carries" if under else "PASS",
                     "carried": "tenx" if under else "declared",
                     "carried_score": t if under else d}
            sm = per["steps%d" % STEPMATCH[fam]][s]["mean"]
            if sm is not None:
                p1[s]["stepmatched"] = sm
                p1[s]["stepmatched_minus_tenx"] = sm - t
        per["P1"] = p1
        per["carried_summary"] = ms([p1[s].get("carried_score") for s in SEEDS])
        out["setC_w21_32768"][fam] = per

    # ---- margins ---------------------------------------------------------------------
    def strongest(means_by_fam):
        ok = {f: v for f, v in means_by_fam.items() if v is not None}
        return max(ok, key=ok.get) if ok else None

    marg = {}

    def vs_per_seed_strongest(cands_by_seed, note):
        """w20's ladder convention (w20_tables.summarise_block 'vs_strongest'): per seed, the
        rival = the max over the candidate fits of their mean score; paired per corner."""
        per = {}
        for s in SEEDS:
            cands = [(lab, r) for lab, r in cands_by_seed.get(s, []) if r is not None]
            if not cands or not store[s]:
                continue
            lab, r = max(cands, key=lambda kv: mean_rows(kv[1]))
            per[s] = dict(pe_short(paired(store[s], r)), rival=lab, rival_mean=mean_rows(r),
                          store_mean=mean_rows(store[s]))
        return {"convention": note, "per_seed": per,
                "mean_delta": ms([per[s]["delta"] for s in per]),
                "complete_3_seeds": len(per) == 3}

    W20C = "per seed: max over families of max(DECLARED, TENX) (= w20's ladder headline convention)"
    marg["setC_vs_strongest_declared_or_tenx"] = vs_per_seed_strongest(
        {s: [("%s__%s" % (f, st), C[(f, st, s)]) for f in FAMILIES for st in ("declared", "tenx")]
         for s in SEEDS}, W20C + " -- set C, 32,768 rows, <=51,200 steps")
    marg["setC_vs_strongest_stepmatched"] = vs_per_seed_strongest(
        {s: [("%s__steps%d" % (f, STEPMATCH[f]), C[(f, "steps%d" % STEPMATCH[f], s)]) for f in FAMILIES]
         for s in SEEDS}, "per seed: max over families of the STEP-MATCHED fit (mlp/cont_mlp 204,800; "
                          "lstm/transformer 20,480 steps = w20's carried budgets) -- set C")
    marg["setC_vs_best_of_every_fit"] = vs_per_seed_strongest(
        {s: [("%s__%s" % (f, st), C[(f, st, s)]) for (f, st, ss) in C if ss == s and st != "ownpoolsched"]
         for s in SEEDS}, "per seed: max over EVERY set-C fit (declared, tenx, step-matched) -- the "
                          "rival's best case")
    marg["setB_vs_strongest_declared_or_tenx_RECOMPUTED"] = vs_per_seed_strongest(
        {s: [("%s__%s" % (f, st), B[(f, st, s)]) for f in FAMILIES for st in ("declared", "tenx")]
         for s in SEEDS}, W20C + " -- set B (w20), recomputed from w20's score files; must equal w20's "
                               "printed ladder P3")
    marg["setA_vs_strongest"] = vs_per_seed_strongest(
        {s: [("spokeA_%s" % f, A[(f, s)]) for f in FAMILIES] for s in SEEDS},
        "per seed: max over the four spoke-A @ref families -- set A")
    # w20's printed value, READ, for the integrity check of the recomputation
    w20j = rd(W20_OUT / "SHAPES-MATCHED.json")
    try:
        vs = w20j["ladder"][rung]["P3_margin"]["store_K32_n131072"]["vs_strongest"]
        pr = {int(k): v["delta"] for k, v in vs.items()}
        rc = marg["setB_vs_strongest_declared_or_tenx_RECOMPUTED"]["per_seed"]
        marg["setB_integrity_vs_w20_printed"] = {
            "w20_printed": pr, "recomputed": {s: rc[s]["delta"] for s in rc},
            "identical": all(abs(pr[s] - rc[s]["delta"]) < 1e-9 for s in rc if s in pr)}
    except Exception as e:  # noqa: BLE001
        marg["setB_integrity_vs_w20_printed"] = {"error": repr(e)}
    out["margins_K32_store_minus_rival"] = marg

    # ---- cross-check vs learned_ref_ownpool --------------------------------------------
    ref, mine, per = {}, {}, {}
    for s in SEEDS:
        p = CL / ("%s_seed%d" % (rung, s)) / "rows_learned_ref_ownpool.json"
        ref[s] = rows_of(p)
        j = rd(p)
        mine[s], mp = my_score(rung, s, "mlp", "ownpoolsched")
        ti = j["summary"].get("train_info", {}) if j else {}
        row = {"ref_mean": mean_rows(ref[s]), "ref_source": str(p), "ref_steps": ti.get("steps"),
               "ref_params": j["summary"].get("param_count") if j else None,
               "ref_n_fit_rows": j["summary"].get("n_fit_rows") if j else None,
               "mine_mean": mean_rows(mine[s]), "mine_source": mp}
        fm = fit_meta(rung, s, "mlp", "ownpoolsched")
        if fm:
            row.update(mine_steps=fm["optimizer_steps"], mine_params=fm["param_count"],
                       mine_n_fit_rows=fm["n_fit_rows"])
        if ref[s] and mine[s]:
            row["paired_mine_minus_ref"] = pe_short(paired(mine[s], ref[s]))
        per[s] = row
    rs = ms([per[s]["ref_mean"] for s in SEEDS])
    msd = ms([per[s]["mine_mean"] for s in SEEDS])
    X1 = {"per_seed": per, "ref": rs, "mine": msd}
    if rs and msd and rs["n_seeds"] == 3 and msd["n_seeds"] == 3:
        d = msd["mean"] - rs["mean"]
        spread = max(rs["sd"], msd["sd"])
        X1.update(delta_mean_mine_minus_ref=d, seed_spread_bar=spread,
                  verdict="REPRODUCES within seed spread" if abs(d) <= spread
                  else "DISCREPANCY (reported, not resolved)")
    out["crosscheck"]["X1_same_rows_same_steps"] = X1
    # DIAGNOSTIC (not the verdict): the ref's own recipe replayed with its own PRNG keys
    dg = {}
    for s in SEEDS:
        rr, rp = my_score(rung, s, "mlp", "refreplay")
        if rr is None or ref[s] is None:
            dg[s] = {"status": "NOT RUN"}
            continue
        A_ = {r["corner_id"]: r["distinct_valid_novel_tuples"] for r in rr}
        B_ = {r["corner_id"]: r["distinct_valid_novel_tuples"] for r in ref[s]}
        dg[s] = {"replay_mean": mean_rows(rr), "ref_mean": mean_rows(ref[s]), "source": rp,
                 "n_corners": len(B_), "n_corners_differing": sum(A_.get(c) != v for c, v in B_.items()),
                 "identical_per_corner": A_ == B_}
    out["crosscheck"]["X1_diagnostic_ref_recipe_replay"] = {
        "what": ("learned_ref_ownpool's recipe (w12_ladder_1x.fit_one) replayed with ITS OWN PRNG keys "
                 "and trainer (scripts/w21_x1_replay.py). A check on what differs between the X1 arms; "
                 "the X1 verdict is computed from mlp__ownpoolsched only."),
        "per_seed": dg}
    X2 = {"ref_2560_steps": rs}
    for st in ("declared", "tenx", "steps204800"):
        X2["mine_%s" % st] = out["setC_w21_32768"]["mlp"][st]["summary"]
        X2["mine_%s_steps" % st] = sorted({out["setC_w21_32768"]["mlp"][st][s]["optimizer_steps"]
                                           for s in SEEDS} - {None})
    X2["reading"] = ("STEP-MISMATCHED (expected to differ): same 32,768 rows, different optimiser-step "
                     "budgets. Reported as the measured steps effect (Hub AMENDMENT).")
    out["crosscheck"]["X2_same_rows_different_steps"] = X2
    return out


def main():
    J = {"schema": "w21 ladder-rivals-32k v1", "free_replication": True, "ledger_row": None,
         "scored_runs_spent": 0, "seeds": list(SEEDS), "rungs": {}}
    for k in RUNGS:
        J["rungs"][k] = rung_block(k)
    jdump(J, OUT / "LADDER-RIVALS-32K.json")
    print("wrote", OUT / "LADDER-RIVALS-32K.json")


if __name__ == "__main__":
    main()
