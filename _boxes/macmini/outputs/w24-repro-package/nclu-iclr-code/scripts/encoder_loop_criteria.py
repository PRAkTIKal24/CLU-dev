"""encoder loop criteria."""

from __future__ import annotations
from nclu.paths import run_dir
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("w8rep", HERE / "encoder_loop_report.py")
w8rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w8rep)
METRIC = w8rep.METRIC
ROWS = list(w8rep.ROWS)
MARGIN = 1.0
CONTROLS = ("pca_landscape_removed", "pca_landscape_shuffled", "pca_random_refit")
FLOOR = "pca_lookup_argmax"
PROTOCOL_FILE = str(run_dir("encoder_loop/criteria/protocol.md"))
CPU_CROSS_ARM = str(run_dir("encoder_loop/CROSS-ARM.json"))
CPU_PINS = {"joint_steps": 400, "n_train": 8192, "dec_refit_steps": 400, "warm_steps": 1500}
MANIFESTS = Path(str(run_dir("encoder_loop/cloud/manifests")))
NEVER_MERGE = "This table is scored at the GPU leg's pins (joint_steps 4000 / n_train 32768 / dec_refit_steps 1800). It NEVER shares a table with the CPU leg (400 / 8192 / 400): a run at one budget and a run at another do not share a table (exp_encoder_loop.apply_compute_overrides). The CPU cells below are a separate block for the DIRECTION comparison only; nothing is pooled."


def mean(rows):
    return float(np.mean([r[METRIC] for r in rows])) if rows else float("nan")


def growth_stat(p_swap, r_swap, p_alone, r_alone, seed=20260923):
    ids = [r["corner_id"] for r in p_swap]
    for other in (r_swap, p_alone, r_alone):
        if [r["corner_id"] for r in other] != ids:
            raise AssertionError("growth needs the same corners, in order, in all four cells")
    v = lambda rs: np.array([float(r[METRIC]) for r in rs])
    g = v(p_swap) - v(r_swap) - (v(p_alone) - v(r_alone))
    rng = np.random.default_rng(seed)
    n = g.size
    boot = np.array([g[rng.integers(0, n, n)].mean() for _ in range(10000)])
    return {
        "growth": float(g.mean()),
        "n_corners": int(n),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "corners_growth_pos": int((g > 0).sum()),
        "corners_growth_neg": int((g < 0).sum()),
    }


def read_pins(run_base: Path, seeds):
    pins = {}
    for s in seeds:
        for arm in ("alone", "loop_nosw", "loop_swap"):
            p = run_base / f"seed_{s}" / arm / "arm.json"
            if p.exists():
                j = json.loads(p.read_text())
                c = j["compute"]
                pins[f"s{s}:{arm}"] = {
                    "joint_steps": int(c["gradient_steps"]),
                    "images_seen": int(c["images_seen"]),
                    "n_train": int(c["n_train_rows"]),
                    "dec_refit_steps": int(j["decoder_refit"]["recipe"]["steps"]),
                    "decoder_recipe": j["decoder_refit"]["recipe"]["name"],
                    "encoder_param_hash": c["encoder_param_hash_after"],
                    "source": str(p),
                }
    return pins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-base", required=True)
    ap.add_argument("--run-base", required=True)
    ap.add_argument("--condition", default="test")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gate", action="store_true", help="evaluate the VAL gate V1-V3 only")
    a = ap.parse_args(argv)
    sb, rb, od = (Path(a.score_base), Path(a.run_base), Path(a.out_dir))
    od.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for s in [] if a.gate else a.seeds:
        for enc in ("alone", "loop_nosw", "loop_swap"):
            leg = w8rep.leg_dir(sb, enc, a.condition, s)
            if not leg.exists():
                continue
            for sa in ("pca_tensor", "rival_mlp_180k", FLOOR, *CONTROLS):
                got = w8rep.load_corner_rows(leg, sa, s, a.condition)
                if got is not None:
                    loaded[s, enc, sa] = got

    def cell(s, enc, sa, row):
        k = (s, enc, sa)
        return loaded[k][row]["rows"] if k in loaded else None

    pins = read_pins(rb, a.seeds)
    pin_set = {(v["joint_steps"], v["n_train"], v["dec_refit_steps"]) for v in pins.values()}
    if a.gate:
        s = a.seeds[0]
        row = "r4_comp_nway"
        g = {}
        for enc in ("alone", "loop_swap"):
            t = json.loads((w8rep.leg_dir(sb, enc, a.condition, s) / "READ-TABLE.json").read_text())
            g[enc] = {
                sa: t["arms"].get(sa, {}).get(row, {}).get("novel_tuples_per_corner")
                for sa in ("pca_tensor", "rival_mlp_180k", "pca_landscape_removed")
            }
            g[enc]["n_corners"] = t["arms"]["pca_tensor"][row]["n_corners"]
        ok = lambda x: x is not None
        v1 = all(
            (
                ok(g[e]["pca_landscape_removed"]) and g[e]["pca_landscape_removed"] <= MARGIN
                for e in g
            )
        )
        v2 = any((ok(g[e]["rival_mlp_180k"]) and g[e]["rival_mlp_180k"] > 0 for e in g))
        v3 = all(
            (
                ok(g[e]["pca_tensor"])
                and ok(g[e]["pca_landscape_removed"])
                and (g[e]["pca_tensor"] > g[e]["pca_landscape_removed"] + MARGIN)
                for e in g
            )
        )
        out = {
            "schema": "nclu-loop-valgate/1",
            "condition": a.condition,
            "seed": s,
            "row": row,
            "means": g,
            "V1_C2_collapses": v1,
            "V2_baseline_not_degenerate": v2,
            "V3_arm_renders": v3,
            "GATE_PASS": bool(v1 and v2 and v3),
            "pins": pins,
            "protocol": PROTOCOL_FILE,
        }
        (od / "VAL-GATE.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
        print(
            json.dumps(
                {
                    k: out[k]
                    for k in (
                        "means",
                        "V1_C2_collapses",
                        "V2_baseline_not_degenerate",
                        "V3_arm_renders",
                        "GATE_PASS",
                    )
                },
                indent=2,
                default=float,
            )
        )
        return 0
    cross_p = od / f"CROSS-ARM-GPU-{a.condition}.json"
    w8rep.main(
        [
            "--score-base",
            str(sb),
            "--run-base",
            str(rb),
            "--condition",
            a.condition,
            "--seeds",
            *map(str, a.seeds),
            "--out",
            str(cross_p),
        ]
    )
    cross = json.loads(cross_p.read_text())
    cross["protocol"] = PROTOCOL_FILE
    cross["protocol_note"] = (
        "encoder_loop_report.py writes its own PROTOCOL name into this field; this leg is scored under protocol.md, overwritten here"
    )
    cross["budget_pins"] = pins
    cross["NEVER_SHARES_A_TABLE_WITH_CPU_LEG"] = NEVER_MERGE
    cross_p.write_text(json.dumps(cross, indent=2, sort_keys=True, default=float) + "\n")
    bar = {}
    for row in ROWS:
        per = {}
        for s in a.seeds:
            P_s, R_s = (
                cell(s, "loop_swap", "pca_tensor", row),
                cell(s, "loop_swap", "rival_mlp_180k", row),
            )
            P_a, R_a = (
                cell(s, "alone", "pca_tensor", row),
                cell(s, "alone", "rival_mlp_180k", row),
            )
            if None in (P_s, R_s, P_a, R_a):
                per[str(s)] = {
                    "status": "MISSING",
                    "cells_present": [
                        x
                        for x, c in (
                            ("P_swap", P_s),
                            ("R_swap", R_s),
                            ("P_alone", P_a),
                            ("R_alone", R_a),
                        )
                        if c is not None
                    ],
                }
                continue
            gs = growth_stat(P_s, R_s, P_a, R_a)
            void = mean(R_s) == 0.0 and mean(R_a) == 0.0
            ctl = {}
            for enc in ("alone", "loop_swap"):
                ctl[enc] = {
                    c: mean(cell(s, enc, c, row)) if cell(s, enc, c, row) is not None else "NOT-RUN"
                    for c in (*CONTROLS, FLOOR)
                }
            collapsed = all(
                (
                    isinstance(ctl[e][c], float) and ctl[e][c] <= MARGIN
                    for e in ctl
                    for c in CONTROLS
                )
            )
            per[str(s)] = {
                "status": "VOID" if void else "SCORED",
                "pca_tensor_mean": {"alone": mean(P_a), "loop_swap": mean(P_s)},
                "rival_mlp_180k_mean": {"alone": mean(R_a), "loop_swap": mean(R_s)},
                "M_alone": mean(P_a) - mean(R_a),
                "M_loop_swap": mean(P_s) - mean(R_s),
                "within_arm_paired": {
                    e: w8rep.pair(
                        cell(s, e, "pca_tensor", row),
                        cell(s, e, "rival_mlp_180k", row),
                        f"pca_tensor@{e}",
                        f"rival_mlp_180k@{e}",
                        MARGIN,
                    )
                    for e in ("alone", "loop_swap")
                },
                **gs,
                "controls_and_floor_mean": ctl,
                "landscape_C_all_three_controls_collapse": collapsed,
            }
        scored = [v for v in per.values() if v.get("status") == "SCORED"]
        n_seeds = len(a.seeds)
        grows = (
            len(scored) == n_seeds == 3
            and all((v["growth"] > 0 for v in scored))
            and (sum((v["ci95"][0] > 0 for v in scored)) >= 2)
        )
        nogrow = sum((v["growth"] <= 0 for v in scored)) >= 2
        verdict = "GROWS" if grows else "DOES NOT GROW" if nogrow else "MIXED"
        if len(scored) < 2 and (not grows):
            verdict = "VOID/INSUFFICIENT"
        bar[row] = {
            "per_seed": per,
            "n_seeds_scored": len(scored),
            "n_seeds_void": sum((v.get("status") == "VOID" for v in per.values())),
            "growths": [v["growth"] for v in scored],
            "growth_ci95_lows": [v["ci95"][0] for v in scored],
            "VERDICT": verdict,
            "landscape_C_collapse_all_seeds": bool(scored)
            and all((v["landscape_C_all_three_controls_collapse"] for v in scored)),
        }

    def ordering_cells(ps_getter, seeds):
        cells = {}
        for row in ROWS:
            for s in seeds:
                a_, n_, w_ = (ps_getter(s, e, row) for e in ("alone", "loop_nosw", "loop_swap"))
                if None in (a_, n_, w_):
                    continue
                cells[f"s{s}:{row}"] = {
                    "Aprime_nosw_minus_alone": n_ - a_,
                    "Adprime_swap_minus_nosw": w_ - n_,
                    "A_swap_minus_alone": w_ - a_,
                }
        return cells

    def gpu_mean(s, e, row):
        c = cell(s, e, "pca_tensor", row)
        return mean(c) if c is not None else None

    cpu = json.loads(Path(CPU_CROSS_ARM).read_text())

    def cpu_mean(s, e, row):
        return cpu["per_seed"][str(s)][row]["mean_novel_per_corner"].get(f"{e}:pca_tensor")

    def summarise(cells):
        n = len(cells)
        neg_ap = sum((v["Aprime_nosw_minus_alone"] < 0 for v in cells.values()))
        pos_adp = sum((v["Adprime_swap_minus_nosw"] > 0 for v in cells.values()))
        return {"n_cells": n, "Aprime_negative": neg_ap, "Adprime_positive": pos_adp}

    gpu_cells, cpu_cells = (ordering_cells(gpu_mean, a.seeds), ordering_cells(cpu_mean, [0, 1, 2]))
    gsum, csum = (summarise(gpu_cells), summarise(cpu_cells))
    reproduced = (
        gsum["n_cells"] == 6 and gsum["Aprime_negative"] >= 4 and (gsum["Adprime_positive"] >= 4)
    )
    out = {
        "schema": "nclu-loop-criteria/1",
        "protocol": PROTOCOL_FILE,
        "condition": a.condition,
        "seeds": a.seeds,
        "seed_count": len(a.seeds),
        "metric": METRIC,
        "margin": MARGIN,
        "BUDGET_PINS_OF_THIS_ROW": {
            "joint_steps": 4000,
            "n_train": 32768,
            "dec_refit_steps": 1800,
            "warm_steps": 4000,
            "as_read_from_arm_manifests": pins,
            "all_arms_share_one_pin_set": len(pin_set) == 1,
            "pin_sets_seen": sorted(pin_set),
        },
        "NEVER_SHARES_A_TABLE_WITH_CPU_LEG": NEVER_MERGE,
        "manifest_count_measured": {
            "path": str(MANIFESTS),
            "n_files": sum((1 for p in MANIFESTS.rglob("*") if p.is_file()))
            if MANIFESTS.exists()
            else 0,
        },
        "BAR_D_GPU": bar,
        "BAR_D_GPU_VERDICT_row4_PRIMARY": bar["r4_comp_nway"]["VERDICT"],
        "ORDERING": {
            "rule": "REPRODUCED iff on GPU A' < 0 in >= 4/6 cells AND A'' > 0 in >= 4/6 cells",
            "GPU_block_pins": "4000 / 32768 / dec 1800",
            "GPU_cells": gpu_cells,
            "GPU_summary": gsum,
            "CPU_block_pins_SEPARATE_TABLE": CPU_PINS,
            "CPU_source": CPU_CROSS_ARM,
            "CPU_cells": cpu_cells,
            "CPU_summary": csum,
            "REPRODUCED": reproduced,
        },
        "cross_arm_table": str(cross_p),
    }
    bp = od / ("BAR-D.json" if a.condition == "test" else f"BAR-D-{a.condition}.json")
    bp.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(
        json.dumps(
            {
                "BAR_D_row4": bar["r4_comp_nway"]["VERDICT"],
                "BAR_D_row3": bar["r3_comp_2way"]["VERDICT"],
                "ORDERING_REPRODUCED": reproduced,
                "GPU": gsum,
                "CPU": csum,
            },
            indent=2,
            default=float,
        )
    )
    print(f"-> {bp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
