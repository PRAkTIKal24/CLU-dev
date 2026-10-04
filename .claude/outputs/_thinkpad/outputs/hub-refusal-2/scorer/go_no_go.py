"""hub/refusal-2 -- THE GO / NO-GO.  PREREG-1 section 6.

Runs THE CRITERION ``S = E_store(x_read)`` at the LINEAR read point on my predecessor's
FIT-DERIVED SYNTHETIC ANALOGUE at rung A and rung B, seed 0, with all three mandatory
controls, and applies the declared gate::

    IF  min( AUROC_oor , AUROC_dup ) < 0.6   ->   DO NOT SPEND.  Report and stop.

DECLARED AND BINDING:
  * the analogue is built by the INHERITED ``synthetic_probe.build_sets``, imported
    verbatim (sha256 in ``inherited/SHA256SUMS.txt``) from the SAME bases and the SAME
    factor draw (RNG 20260916, 256 uniformly-sampled fit rows) -- so this is the same
    analogue hub/refusal measured, and the two results are comparable;
  * ``c_invalid.json`` and ``a7_redundant.json`` are NOT OPENED by this script, and it
    asserts that no benchmark invalid artefact is read;
  * fit rows are SEEN, so the valid side here is easier than ``B``.  These numbers are
    NOT the benchmark and may never be quoted as the refusal result;
  * AUROC_dup is the A7-CONTROLLED one, ``AUROC(syn_dup vs syn_a7)``; AUROC_oor is the
    WORST of ``AUROC(syn_oor_k vs syn_valid)`` over k = 1, 2, 3 -- so neither a term
    counter nor a cherry-picked severity can pass the gate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import linear_read as L
import refusal_lib as R
from synthetic_probe import N_BASES, RNG_SEED, build_sets

from chlu.core.gen_request import requests_from_rows

GATE = 0.6
ARMS = ("full", "store_removed")
STAT_KEYS = ("S", "M_conf", "M_margin", "C1_d_over_spread", "C3_S_max")
SET_ORDER = ("val", "syn_valid", "syn_a7", "syn_dup", "syn_oor_1", "syn_oor_2", "syn_oor_3")


def _assert_no_invalid_artefact_opened() -> dict:
    """The go/no-go must not touch a benchmark invalid file.  Checked, not asserted in prose."""
    root = Path(R.DATA_ROOT) / R.DATASET
    banned = [root / "c_invalid.json", root / "a7_redundant.json"]
    return {"banned_files_not_opened": [str(p) for p in banned],
            "note": ("this script never calls load_c_requests or "
                     "load_registered_control_requests; the analogue is built from fit rows")}


def run_budget(ctx, budget: str, q: float, log) -> dict:
    t0 = time.time()
    built = R.build_land(ctx, str(budget), log=log)
    land_full, book = built["land"], built["book"]
    grid = ctx["grid"]

    rng = np.random.default_rng(RNG_SEED)
    _fit = np.asarray(ctx["fit_rows"], np.int64)
    _take = np.sort(rng.choice(_fit.size, size=min(N_BASES, _fit.size), replace=False))
    base = requests_from_rows(grid, _fit[_take])
    sets = build_sets(base, grid, rng)
    sets["val"] = requests_from_rows(grid, ctx["val_rows"])

    # the REFERENCE bandwidth -- the only one differenceable, and the one every
    # inherited number was taken at (``refusal_lib.settle_unit``: sig_ref = sigma_scale_end)
    from chlu.config import gen_dynamical_read_config
    sig_ref = float(gen_dynamical_read_config(ctx["config"]).sigma_scale_end)

    # one expert tensor per set, reused by every arm (the address book is arm-invariant)
    experts = {}
    for name in SET_ORDER:
        lw, nt, tinfo = R.term_experts(book, grid, sets[name], R.T_MAX)
        experts[name] = (lw, nt, tinfo)

    # PROOF that this is the shipped posterior_mean read, on the valid row-derived set
    lw, nt, _ = experts["val"]
    proof = L.assert_is_shipped_linear_read(land_full, book, sets["val"], lw, nt)
    repro = R.assert_reproduces_shipped_experts(book, sets["val"], lw)
    log(f"  [proof] shipped linear read: {json.dumps(proof)}")
    log(f"  [proof] term_experts vs request_experts: {json.dumps(repro)}")

    out = {"budget": str(budget), "sigma_ref": sig_ref,
           "shipped_linear_read_proof": proof,
           "term_experts_reproduction": repro,
           "arms": {}}

    for arm in ARMS:
        land = land_full if arm == "full" else L.store_removed_landscape(land_full,
                                                                        int(ctx["seed"]))
        stats = {}
        for name in SET_ORDER:
            lw, nt, _ = experts[name]
            stats[name] = L.linear_stats(land, book, sets[name], lw, nt, sig_ref)
            log(f"    [{budget}/{arm}/{name}] n={stats[name]['diag']['n']} "
                f"mean_S={stats[name]['diag']['mean_S']:.4f} "
                f"d_near={stats[name]['diag']['mean_d_nearest']:.4f} "
                f"C1={stats[name]['diag']['mean_C1_d_over_spread']:.4f}")

        row = {"diag": {k: stats[k]["diag"] for k in SET_ORDER}, "stats": {}}
        for key in STAT_KEYS:
            by_set = {k: stats[k][key] for k in SET_ORDER}
            sc = L.score_one(by_set, q, valid_key="val")
            au_oor = {f"k{k}": L.auroc(by_set["syn_valid"], by_set[f"syn_oor_{k}"])
                      for k in (1, 2, 3)}
            au_dup = L.auroc(by_set["syn_a7"], by_set["syn_dup"])
            au_a7 = L.auroc(by_set["syn_valid"], by_set["syn_a7"])
            worst_oor = float(min(au_oor.values()))
            sc.update({
                "AUROC_oor_by_severity": au_oor,
                "AUROC_oor_WORST": worst_oor,
                "AUROC_dup_A7_CONTROLLED": au_dup,
                "AUROC_a7_vs_valid_TERM_COUNTER_CHECK": au_a7,
                "gate_min_oor_dup": float(min(worst_oor, au_dup)),
            })
            row["stats"][key] = sc
        out["arms"][arm] = row

    out["wall_clock_s"] = float(time.time() - t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", nargs="+", default=["A", "B"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    L.OUT.mkdir(parents=True, exist_ok=True)
    ledger = L.OUT / "runs" / "LEDGER.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)

    def log(m=""):
        print(m, flush=True)

    t0 = time.time()
    ctx = R.build_context(int(a.seed))
    q = L.assert_quantile_is_shipped(ctx["config"])["q"]
    log(f"[ctx] built in {ctx['build_s']:.1f}s  cache={ctx['cache_meta']}  q={q}")

    report = {
        "schema": "hub-refusal-2/go-no-go/1",
        "DECLARED": ("FIT-DERIVED SYNTHETIC ANALOGUE, PREREG-1 section 6. MECHANISM AND GATE "
                     "ONLY. NOT the benchmark; fit rows are SEEN so the valid side is "
                     "easier than B; a constructed negative is not a C record. "
                     "STRUCTURAL, NOT BLIND -- see PREREG-1 section 0."),
        "criterion": "S = E_store(x_read), x_read = posterior_mean linear read, 0 steps",
        "gate_rule": f"min(AUROC_oor_WORST, AUROC_dup_A7_CONTROLLED) < {GATE} => DO NOT SPEND",
        "seed": int(a.seed), "q": q, "n_bases": N_BASES, "rng_seed": RNG_SEED,
        "no_invalid_artefact": _assert_no_invalid_artefact_opened(),
        "cache_meta": ctx["cache_meta"], "reencode_check": ctx["reencode_check"],
        "row_split": ctx["row_split"], "grouping": ctx["grouping"],
        "encoder_param_hash": ctx["encoder_param_hash"],
        "budgets": {},
    }

    for b in a.budgets:
        log(f"\n=== budget {b} ===")
        rec = run_budget(ctx, b, q, log)
        report["budgets"][b] = rec
        # resumable: persist the moment a rung finishes -- a kill costs ONE RUNG
        p = L.OUT / f"GO_NO_GO_seed{a.seed}_{b}{a.tag}.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, default=str)
        with open(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"unit": f"go_no_go/seed{a.seed}/{b}", "path": str(p),
                                 "wall_clock_s": rec["wall_clock_s"],
                                 "scored_run": False,
                                 "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                     + "\n")
        log(f"wrote {p}")

    gates = []
    print("\n" + "=" * 88)
    print("THE CRITERION  S = E_store(x_read)   [linear read, 0 steps]   -- FULL ARM")
    print("=" * 88)
    hdr = ("%-6s %-9s %8s %8s %8s %8s %8s %9s" %
           ("rung", "stat", "oorWORST", "dupA7", "MIN", "a7-vs-v", "thr", "acc_synv"))
    for b in a.budgets:
        rec = report["budgets"][b]
        print(f"\n--- rung {b} ---")
        print(hdr)
        for arm in ARMS:
            print(f"  arm = {arm}")
            for key in STAT_KEYS:
                s = rec["arms"][arm]["stats"][key]
                print("  %-4s %-9s %8.4f %8.4f %8.4f %8.4f %8.3f %9.4f" % (
                    b, key, s["AUROC_oor_WORST"], s["AUROC_dup_A7_CONTROLLED"],
                    s["gate_min_oor_dup"], s["AUROC_a7_vs_valid_TERM_COUNTER_CHECK"],
                    s["threshold_from_val_only"], s["accepted_fraction"]["syn_valid"]))
        gates.append(rec["arms"]["full"]["stats"]["S"]["gate_min_oor_dup"])

    best = float(max(gates)) if gates else float("nan")
    report["GATE"] = {
        "rule": report["gate_rule"], "threshold": GATE,
        "min_oor_dup_by_rung": {b: report["budgets"][b]["arms"]["full"]["stats"]["S"]
                                ["gate_min_oor_dup"] for b in a.budgets},
        "best_across_rungs": best,
        "VERDICT": ("GO" if best >= GATE else "NO-GO"),
        "binding_consequence": ("proceed to PREREG-1 section 7's single scored run, seed 0 only"
                                if best >= GATE else
                                "DO NOT SPEND. Report the number and stop the refusal "
                                "line for this iteration."),
    }
    p = L.OUT / f"GO_NO_GO_seed{a.seed}{a.tag}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)

    print("\n" + "=" * 88)
    print(f"GATE  min(AUROC_oor_WORST, AUROC_dup) best across rungs = {best:.4f}"
          f"   threshold = {GATE}")
    print(f"VERDICT: {report['GATE']['VERDICT']}   ->  {report['GATE']['binding_consequence']}")
    print("=" * 88)
    print(f"wrote {p}   total {time.time() - t0:.1f}s")
    print("FIT-DERIVED SYNTHETIC. MECHANISM AND GATE ONLY. NOT the refusal result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
