"""w13 hub/encoder-adapt -- the CROSS-ARM report.

Reads every arm manifest, probe, rivals and score artefact this hub wrote and produces
`CROSS-ARM.json` + `CROSS-ARM.md`.  It computes NOTHING that is not already on disk; it
only lines the arms up so the three questions can be answered:

  1. does the MARGIN grow from `frozen` to an adapted arm, on >= 2 of 3 seeds?
  2. do ALL arms move together -- in which case it is an ENCODER finding, not a store one?
  3. how far did `anchor` travel from the frozen features and weights?

Anything absent is written as a DECLARED NOT-RUN with the reason.  Never a zero, never a
blank.  Seed count and configuration count are printed SEPARATELY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("frozen", "adapter", "anchor")
REFERENCE = "frozen"


def log(m=""):
    print(m, flush=True)


def jload(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as exc:                                        # noqa: BLE001
        return {"DECLARED_NOT_RUN": f"{type(exc).__name__}: {exc}", "path": str(p)}


def not_run(d) -> bool:
    return isinstance(d, dict) and "DECLARED_NOT_RUN" in d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-base", required=True)
    ap.add_argument("--score-base", required=True)
    ap.add_argument("--condition", default="test")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    run, sc = Path(a.run_base), Path(a.score_base)
    doc = {
        "schema": "chlu-w13-encoder-adapt-crossarm/1",
        "hub": "w13-encoder-adapt",
        "condition": a.condition, "K_star": a.K,
        "seed_count": len(seeds), "seeds": seeds,
        "configuration_count": len(ARMS),
        "arm_runs": len(seeds) * len(ARMS),
        "counting_note": ("SEED COUNT and CONFIGURATION COUNT are separate numbers. "
                          f"{len(seeds)} seeds x {len(ARMS)} arms = "
                          f"{len(seeds) * len(ARMS)} arm-runs. That is "
                          f"{len(seeds)} SEEDS, never '{len(seeds) * len(ARMS)} seeds'."),
        "per_seed": {}, "compute_match": {}, "displacement": {},
        "probe_margin": {}, "MARGIN": {}, "VERDICT": {},
    }

    for s in seeds:
        doc["per_seed"][str(s)] = {}
        doc["compute_match"][str(s)] = {}
        for arm in ARMS:
            ad = run / f"seed_{s}" / arm
            man = jload(ad / "arm.json")
            prb = jload(ad / "probe.json")
            riv = jload(ad / "rivals.json")
            scd = jload(sc / f"{a.condition}_seed{s}_{arm}_K{a.K}" / "score.json")
            entry = {"arm.json": ("ok" if not not_run(man) else man),
                     "probe.json": ("ok" if not not_run(prb) else prb),
                     "rivals.json": ("ok" if not not_run(riv) else riv),
                     "score.json": ("ok" if not not_run(scd) else scd)}
            if not not_run(man):
                c = man["compute"]
                entry["compute"] = {
                    "gradient_steps": c["gradient_steps"],
                    "images_seen": c["images_seen"],
                    "train_seconds": c["train_seconds"],
                    "seconds_per_step": c["seconds_per_step"],
                    "n_params_optimised": c["n_params_optimised"],
                    "n_encoder_params_optimised": c["n_encoder_params_optimised"],
                    "encoder_changed": c["encoder_changed"],
                    "base_encoder_changed": c["base_encoder_changed"],
                    "beta_swap": c["beta_swap"],
                    "anchor_lambda": c["anchor_lambda"],
                    "swap_identity_abs_delta":
                        (c.get("swap_identity_vs_frozen_swap_loss") or {})
                        .get("abs_delta_loss"),
                }
                entry["displacement"] = c.get("displacement_final")
                entry["adapter"] = man.get("adapter")
                doc["compute_match"][str(s)][arm] = entry["compute"]
            if not not_run(prb):
                entry["probe_margin_over_random_init"] = prb.get(
                    "MARGIN_OVER_RANDOM_INIT")
                entry["probe_absolute_NOT_A_CLAIM"] = {
                    "arm": (prb.get("arm_probe") or {}).get("aggregate"),
                    "random_init_floor": (prb.get("random_init_floor") or {}
                                          ).get("aggregate"),
                    "note": ("an absolute probe number says how easy Shapes3D is under a "
                             "linear readout; only the margin says anything about the arm"),
                }
            if not not_run(scd):
                entry["MARGIN"] = scd.get("MARGIN")
                entry["single_design_novel_per_corner"] = (
                    (scd.get("per_arm") or {}).get("cov_tensor", {})
                    .get("all", {}).get("novel_per_corner_mean"))
                entry["controls"] = {
                    k: (scd["per_arm"][k]["all"].get("novel_per_corner_mean")
                        if k in (scd.get("per_arm") or {}) else "DECLARED NOT-RUN")
                    for k in ("landscape_removed", "landscape_shuffled",
                              "landscape_shuffled_across_wells", "random_atoms")}
                entry["rank_over_width_max"] = scd.get("rank_over_width_max")
                entry["ledger_four_part"] = scd.get("ledger_four_part")
            doc["per_seed"][str(s)][arm] = entry

    # ---- the bar --------------------------------------------------------
    def margin_of(s, arm):
        e = doc["per_seed"][str(s)].get(arm, {})
        m = e.get("MARGIN")
        if isinstance(m, dict) and "MARGIN_all_corners" in m:
            return float(m["MARGIN_all_corners"])
        return None

    for arm in ARMS:
        if arm == REFERENCE:
            continue
        rows, grew = {}, []
        for s in seeds:
            mr, ma = margin_of(s, REFERENCE), margin_of(s, arm)
            if mr is None or ma is None:
                rows[str(s)] = {"DECLARED_NOT_RUN":
                                f"margin missing (reference={mr}, {arm}={ma})"}
                continue
            rows[str(s)] = {"margin_frozen": mr, "margin_" + arm: ma,
                            "delta": ma - mr, "grew": bool(ma > mr),
                            "both_still_negative": bool(ma < 0 and mr < 0)}
            grew.append(ma > mr)
        doc["MARGIN"][arm] = {
            "per_seed": rows,
            "n_seeds_with_both_margins": len(grew),
            "n_seeds_margin_grew": int(sum(grew)),
            "BAR_margin_grows_on_at_least_2_of_3": (
                bool(sum(grew) >= 2) if len(grew) == len(seeds) else
                "DECLARED NOT-RUN -- not every seed has both margins"),
        }

    # ---- is it an ENCODER gain rather than a STORE gain? -----------------
    lvl = {}
    for arm in ARMS:
        vals = [doc["per_seed"][str(s)][arm].get("single_design_novel_per_corner")
                for s in seeds]
        lvl[arm] = [v for v in vals if v is not None]
    doc["VERDICT"]["arm_level_composition"] = {
        "single_design_novel_per_corner_per_seed": lvl,
        "how_to_read": ("if EVERY arm's level moves together while the MARGIN does not, "
                        "the gain is in the ENCODER's features, which every baseline also "
                        "gets -- it is NOT a store result and is reported as such"),
    }
    doc["VERDICT"]["displacement_anchor"] = {
        str(s): doc["per_seed"][str(s)].get("anchor", {}).get("displacement")
        for s in seeds}
    doc["VERDICT"]["probe_is_per_seed_never_pooled"] = {
        arm: {str(s): doc["per_seed"][str(s)][arm].get(
            "probe_margin_over_random_init") for s in seeds} for arm in ARMS}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CROSS-ARM.json").write_text(json.dumps(doc, indent=2, default=str),
                                        encoding="utf-8")
    log(f"wrote {out / 'CROSS-ARM.json'}")

    # ---- a readable table ------------------------------------------------
    L = ["# w13 hub/encoder-adapt — CROSS-ARM", "",
         f"**{len(seeds)} seeds ({', '.join(map(str, seeds))}) × {len(ARMS)} "
         f"configurations = {len(seeds) * len(ARMS)} arm-runs.** "
         f"⛔ That is {len(seeds)} SEEDS, never “{len(seeds) * len(ARMS)} seeds”.", "",
         f"Condition `{a.condition}`, K\\* = {a.K}.", "",
         "## Equal steps and equal images — printed, not claimed", "",
         "| seed | arm | steps | images | seconds | s/step | enc params in optimiser |",
         "|---|---|---|---|---|---|---|"]
    for s in seeds:
        for arm in ARMS:
            c = doc["compute_match"].get(str(s), {}).get(arm)
            if not c:
                L.append(f"| {s} | {arm} | — | — | — | — | DECLARED NOT-RUN |")
                continue
            L.append(f"| {s} | {arm} | {c['gradient_steps']} | {c['images_seen']} | "
                     f"{c['train_seconds']:.1f} | {c['seconds_per_step']:.3f} | "
                     f"{c['n_encoder_params_optimised']:,} |")
    L += ["", "## THE MARGIN — single design − best learned baseline, per seed", "",
          "| seed | arm | single design | best learned | margin | Δ vs frozen |",
          "|---|---|---|---|---|---|"]
    for s in seeds:
        for arm in ARMS:
            m = doc["per_seed"][str(s)][arm].get("MARGIN")
            if not isinstance(m, dict) or "MARGIN_all_corners" not in m:
                L.append(f"| {s} | {arm} | — | — | DECLARED NOT-RUN | — |")
                continue
            d = ""
            if arm != REFERENCE:
                r = doc["MARGIN"].get(arm, {}).get("per_seed", {}).get(str(s), {})
                d = (f"{r['delta']:+.4f}" if "delta" in r else "—")
            L.append(f"| {s} | {arm} | {m['single_design_mean']:.4f} | "
                     f"{m['best_learned_baseline']} {m['best_learned_mean']:.4f} | "
                     f"**{m['MARGIN_all_corners']:+.4f}** | {d} |")
    L += ["", "## Factor probe — MARGIN over random init, PER SEED, never pooled", "",
          "| seed | arm | aggregate margin | exact-match margin |", "|---|---|---|---|"]
    for s in seeds:
        for arm in ARMS:
            p = doc["per_seed"][str(s)][arm].get("probe_margin_over_random_init")
            if not isinstance(p, dict):
                L.append(f"| {s} | {arm} | DECLARED NOT-RUN | — |")
                continue
            L.append(f"| {s} | {arm} | {p.get('aggregate'):+.4f} | "
                     f"{p.get('exact_match'):+.4f} |")
    L += ["", "## `anchor` — distance from the frozen encoder (REPORTED, never tuned)", "",
          "| seed | feature rel-L2² | weight rel-L2² |", "|---|---|---|"]
    for s in seeds:
        d = doc["per_seed"][str(s)].get("anchor", {}).get("displacement")
        if not isinstance(d, dict):
            L.append(f"| {s} | DECLARED NOT-RUN | — |")
            continue
        L.append(f"| {s} | {d.get('feature_relative_l2_squared')} | "
                 f"{d.get('weight_relative_l2_squared')} |")
    (out / "CROSS-ARM.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote {out / 'CROSS-ARM.md'}")
    log("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
