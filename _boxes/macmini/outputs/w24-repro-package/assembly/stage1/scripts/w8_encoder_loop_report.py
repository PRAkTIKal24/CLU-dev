"""w8 hub/encoder-loop -- THE CROSS-ARM TABLE.  This is the script that decides the bars.

The w5 leg pairs arms WITHIN one leg (pca_tensor vs a rival, same features).  This hub's
question needs the other pairing: **the same corner, the same store object, the same
judge -- and a DIFFERENT ENCODER**.  So this script reads each encoder arm's leg and
pairs `pca_tensor@loop_swap` against `pca_tensor@alone` corner by corner, using the
harness's own `paired_estimand` so the statistic is byte-identical to the one every other
number in this programme is decided by.

It emits, per seed and per row:

  BAR A   pca_tensor@loop_swap  -  pca_tensor@alone            (THE HUB'S QUESTION)
  BAR A'  pca_tensor@loop_nosw  -  pca_tensor@alone            (unfreezing WITHOUT the swap loss)
  BAR A'' pca_tensor@loop_swap  -  pca_tensor@loop_nosw        (what the SWAP LOSS alone bought)
  BAR D   (pca_tensor - rival_mlp_180k) within each arm, and whether that margin GROWS
          from `alone` to `loop_swap`.  If it does not, the features are better for
          EVERYONE -- a finding about the encoder, NOT about the store, and it says so.

Every number is per seed.  Nothing is pooled across seeds into a single figure: three
seeds are three measurements, and the programme reports all three or none.

    python scripts/w8_encoder_loop_report.py --score-base DIR --run-base DIR \
        --condition test --seeds 0 1 2 --out DIR/CROSS-ARM.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
ROWS = {
    "r4_comp_nway": ("nway", 72),     # 49 corners, ceiling 72 tuples
    "r3_comp_2way": ("pair", 36),     # 60 sub-corners, ceiling 36 (object_hue AND shape fixed)
}
METRIC = "distinct_valid_novel_tuples"
ENCODER_ARMS = ("alone", "frozen", "loop_nosw", "loop_swap", "alone_wc")
SCORE_ARMS = ("pca_tensor", "rival_mlp_180k", "rival_cont_mlp", "rival_transformer",
              "rival_lstm", "pca_lookup_argmax", "pca_landscape_removed")
def leg_dir(score_base: Path, enc_arm: str, condition: str, seed: int) -> Path:
    return Path(score_base) / enc_arm / f"{condition}_seed{int(seed)}"


def load_corner_rows(leg: Path, score_arm: str, seed: int, condition: str):
    """Per-corner rows for one (encoder arm, score arm), through the HARNESS's own code."""
    from chlu.experiments import exp_gen_factor_wells as fw
    from chlu.experiments import exp_gen_read as R

    fw.PINS["seed"] = int(seed)
    fw.PINS["condition"] = str(condition)
    cond = fw.load_condition()
    grid = cond["grid"]
    seen = fw.seen_tuple_mask(grid, cond["train_rows"])
    out = {}
    for row_id, (family, ceiling) in ROWS.items():
        corners = (fw.nway_corners(cond) if family == "nway"
                   else fw.pair_subcorners(cond))
        adir = leg / "arms" / score_arm
        if not (adir / "rows.npz").exists():
            return None
        rows = R._corner_rows(adir, corners, grid, seen)
        out[row_id] = {"rows": rows, "ceiling": ceiling, "n_corners": len(rows)}
    return out


def pair(a_rows, b_rows, a_name, b_name, margin=1.0):
    from chlu.experiments import exp_gen_factor_wells as fw
    return fw.paired_estimand(a_rows, b_rows, METRIC, float(margin), a_name, b_name)


def mean_metric(rows):
    return float(np.mean([r[METRIC] for r in rows])) if rows else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--score-base", required=True)
    p.add_argument("--run-base", required=True)
    p.add_argument("--condition", default="test")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    sb, rb = Path(a.score_base), Path(a.run_base)
    report = {
        "schema": "chlu-w8-encoder-loop-cross-arm/1",
        "condition": a.condition, "seeds": list(a.seeds),
        "metric": METRIC, "margin_declared_in_advance": float(a.margin),
        "prereg": "PREREG-w8-encoder-loop-1.md",
        "how_to_read_this": (
            "The unit is the CORNER, paired: the same corner, the same store object "
            "(pca_tensor, rebuilt from that arm's own features by the unmodified w5 code), "
            "the same INVARIANT judge -- and a different ENCODER. Three seeds are three "
            "measurements and all three are reported. Row 3's ceiling is 36 tuples/corner "
            "and row 4's is 72, so the two rows are NOT on one absolute scale."),
        "per_seed": {},
        "compute_match": {},
        "probe_margin_over_random_init": {},
    }

    for seed in a.seeds:
        sk = str(seed)
        report["per_seed"][sk] = {}
        loaded = {}
        for enc in ENCODER_ARMS:
            leg = leg_dir(sb, enc, a.condition, seed)
            if not leg.exists():
                continue
            for sa in SCORE_ARMS:
                got = load_corner_rows(leg, sa, seed, a.condition)
                if got is not None:
                    loaded[(enc, sa)] = got

        present = sorted({e for (e, _s) in loaded})
        report["per_seed"][sk]["encoder_arms_present"] = present
        report["per_seed"][sk]["score_arms_present"] = sorted({s for (_e, s) in loaded})

        for row_id in ROWS:
            blk = {}
            # --- the absolute mean per (encoder arm, score arm) -----------------
            blk["mean_novel_per_corner"] = {
                f"{e}:{s}": mean_metric(v[row_id]["rows"]) for (e, s), v in loaded.items()}
            blk["frac_of_ceiling"] = {
                f"{e}:{s}": mean_metric(v[row_id]["rows"]) / float(v[row_id]["ceiling"])
                for (e, s), v in loaded.items()}

            # --- BAR A / A' / A'': the encoder comparison, on pca_tensor --------
            bars = {}
            for name, (x, y) in {
                "BAR_A_loop_swap_minus_alone": ("loop_swap", "alone"),
                "BAR_Aprime_loop_nosw_minus_alone": ("loop_nosw", "alone"),
                "BAR_Adprime_swap_loss_alone": ("loop_swap", "loop_nosw"),
                "COMPUTE_CONTROL_loop_swap_minus_alone_wc": ("loop_swap", "alone_wc"),
            }.items():
                kx, ky = (x, "pca_tensor"), (y, "pca_tensor")
                if kx in loaded and ky in loaded:
                    bars[name] = pair(loaded[kx][row_id]["rows"],
                                      loaded[ky][row_id]["rows"],
                                      f"pca_tensor@{x}", f"pca_tensor@{y}", a.margin)
            blk["encoder_comparison"] = bars

            # --- BAR D: does the baseline close the gap, WITHIN each arm? -------
            within, growth = {}, {}
            for e in present:
                ka, kb = (e, "pca_tensor"), (e, "rival_mlp_180k")
                if ka in loaded and kb in loaded:
                    within[e] = pair(loaded[ka][row_id]["rows"], loaded[kb][row_id]["rows"],
                                     f"pca_tensor@{e}", f"rival_mlp_180k@{e}", a.margin)
            if "alone" in within and "loop_swap" in within:
                d0 = within["alone"]["mean_paired_delta"]
                d1 = within["loop_swap"]["mean_paired_delta"]
                growth = {
                    "clu_margin_over_mlp180k_at_alone": d0,
                    "clu_margin_over_mlp180k_at_loop_swap": d1,
                    "growth": float(d1 - d0),
                    "verdict": (
                        "the CLU's margin over the matched-parameter learner GROWS with the "
                        "store in the loop -- consistent with the store shaping the "
                        "representation in a way the store itself exploits"
                        if d1 > d0 else
                        "the CLU's margin over the matched-parameter learner DOES NOT GROW. "
                        "The features are better (or worse) FOR EVERYONE: this is a finding "
                        "ABOUT THE ENCODER, NOT ABOUT THE STORE."),
                }
            blk["baseline_within_arm"] = within
            blk["BAR_D_does_the_baseline_close_the_gap"] = growth
            report["per_seed"][sk][row_id] = blk

        # --- BAR C: equal compute, read off the training manifests --------------
        cm = {}
        for enc in ENCODER_ARMS:
            man = rb / f"seed_{seed}" / enc / "arm.json"
            if man.exists():
                c = json.loads(man.read_text())["compute"]
                cm[enc] = {k: c[k] for k in
                           ("gradient_steps", "images_seen", "train_seconds",
                            "seconds_per_step", "n_params_optimised", "encoder_changed",
                            "resumed_from_step")}
        if {"alone", "loop_nosw", "loop_swap"} <= set(cm):
            steps = {cm[e]["gradient_steps"] for e in ("alone", "loop_nosw", "loop_swap")}
            imgs = {cm[e]["images_seen"] for e in ("alone", "loop_nosw", "loop_swap")}
            cm["BAR_C_equal_steps"] = {
                "gradient_steps_identical": len(steps) == 1,
                "images_seen_identical": len(imgs) == 1,
                "values": {"gradient_steps": sorted(steps), "images_seen": sorted(imgs)},
            }
        if "alone_wc" in cm and "loop_swap" in cm:
            cm["BAR_C_equal_seconds"] = {
                "alone_wc_seconds": cm["alone_wc"]["train_seconds"],
                "loop_swap_seconds": cm["loop_swap"]["train_seconds"],
                "alone_wc_got_at_least_the_treatments_seconds":
                    cm["alone_wc"]["train_seconds"] >= cm["loop_swap"]["train_seconds"],
                "note": ("if the treatment wins on equal STEPS but not against alone_wc on "
                         "equal SECONDS, it is a COMPUTE WIN and not a representation win"),
            }
        report["compute_match"][sk] = cm

        # --- BAR E: the probe, as a MARGIN, never an absolute --------------------
        pm = {}
        for enc in ENCODER_ARMS:
            pp = rb / f"seed_{seed}" / enc / "probe.json"
            if pp.exists():
                j = json.loads(pp.read_text())
                pm[enc] = {
                    "MARGIN_aggregate": j["MARGIN_OVER_RANDOM_INIT"]["aggregate"],
                    "MARGIN_exact_match": j["MARGIN_OVER_RANDOM_INIT"]["exact_match"],
                    "MARGIN_per_factor": j["MARGIN_OVER_RANDOM_INIT"]["per_factor"],
                    "absolute_DO_NOT_QUOTE": j["arm_probe"]["aggregate"],
                    "random_init_floor_DO_NOT_QUOTE": j["random_init_floor"]["aggregate"],
                }
        report["probe_margin_over_random_init"][sk] = pm

    # ---- the verdict across seeds, stated by the rule declared in the PREREG ----
    verdicts = {}
    for row_id in ROWS:
        deltas, ci_lows = [], []
        for sk in report["per_seed"]:
            b = (report["per_seed"][sk].get(row_id, {})
                 .get("encoder_comparison", {}).get("BAR_A_loop_swap_minus_alone"))
            if b:
                deltas.append(b["mean_paired_delta"])
                ci_lows.append(b["bootstrap_ci95"][0])
        if deltas:
            verdicts[row_id] = {
                "n_seeds": len(deltas),
                "mean_paired_deltas": deltas,
                "ci95_lows": ci_lows,
                "BAR_A_PASS": bool(len(deltas) == 3 and all(d > a.margin for d in deltas)
                                   and all(c > a.margin for c in ci_lows)),
                "BAR_B_FALSIFIED": bool(sum(1 for d in deltas if d <= 0) >= 2),
                "SECONDARY_all_positive": bool(deltas and all(d > 0 for d in deltas)),
            }
    report["VERDICT"] = verdicts
    report["verdict_rule"] = (
        "BAR A (PRIMARY): mean paired delta > margin AND bootstrap CI95 low > margin on "
        "ALL THREE seeds, row 4. BAR B (THE FALSIFIER): mean paired delta <= 0 on two or "
        "more seeds => store-in-the-loop features do NOT beat encoder-alone at equal "
        "compute, and that is the reported result. SECONDARY (declared in advance as "
        "secondary and never quotable as BAR A): all three deltas > 0.")

    out = Path(a.out) if a.out else (sb / "CROSS-ARM.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps(report.get("VERDICT", {}), indent=2, default=float))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
