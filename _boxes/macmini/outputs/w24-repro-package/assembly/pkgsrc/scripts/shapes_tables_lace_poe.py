"""shapes tables lace poe."""

from __future__ import annotations
from nclu.paths import run_dir
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nclu.experiments.factor_wells import DECORATION_MARGIN_NOVELTY, paired_estimand

OUT = Path(str(run_dir("shapes3d/lace_poe_32k")))
KSTAR_SCORE_SEED0 = Path(str(run_dir("shapes3d/k_sweep/scored/run1_seed0_K32/scored.json")))
RCB = Path(str(run_dir("shapes3d/ladder_nclu")))
LAD = Path(str(run_dir("shapes3d/ladder_earlier")))
SEEDS = (0, 1, 2)
RUNGS = ("2", "3", "4", "5", "6")
METRIC = "distinct_valid_novel_tuples"
LEARNED = ("mlp", "lstm", "transformer", "cont_mlp")
KCTL_LADDER = ("landscape_removed", "landscape_shuffled", "random_atoms")
KCTL_SHIPPED = (
    "landscape_removed",
    "landscape_shuffled",
    "landscape_shuffled_across_wells",
    "random_atoms",
)


def jload(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def rel(p) -> str:
    return str(Path(p)).replace("\\", "/")


def mean(rows):
    return float(np.mean([r[METRIC] for r in rows])) if rows else None


def shipped_sources(seed: int) -> dict:
    """arm -> (per_corner rows, source path) on rows 3+4 at one seed, from this package's
    scoring outputs (scripts/shapes_score_arms.py --corners rows34)."""
    from nclu.paths import run_dir

    d = run_dir("shapes3d", "scores", "rows34", "seed%d" % seed)
    out = {}
    p = d / "store_K32_n131072.json"
    if p.exists():
        out["kstar_store"] = (jload(p)["per_corner"], rel(p))
    for c in KCTL_SHIPPED:
        p = d / ("store_K32_n131072__%s.json" % c)
        if p.exists():
            out[f"kstar_store__{c}"] = (jload(p)["per_corner"], rel(p))
    p = d / "table_FLOOR.json"
    if p.exists():
        out["table_FLOOR"] = (jload(p)["per_corner"], rel(p))
    return out


def rung_sources(rung: str, seed: int) -> dict:
    kd = RCB / "kstar_ladder" / f"{rung}_seed{seed}_K32"
    out = {}
    p = kd / "rows_single_design_K32.json"
    if p.exists():
        out["kstar_store"] = (jload(p)["per_corner"], rel(p))
    for c in KCTL_LADDER:
        p = kd / f"rows_single_design_K32__{c}.json"
        if p.exists():
            out[f"kstar_store__{c}"] = (jload(p)["per_corner"], rel(p))
    p = kd / "rows_learned_1x.json"
    if p.exists():
        out["learned_1x@kstar_total"] = (jload(p)["per_corner"], rel(p))
    ld = LAD / f"{rung}_seed{seed}"
    for f in LEARNED:
        p = ld / f"rows_rival_{f}.json"
        if p.exists():
            out[f"{f}@524288ref"] = (jload(p)["per_corner"], rel(p))
    p = ld / "rows_rival_table.json"
    if p.exists():
        out["table_FLOOR"] = (jload(p)["per_corner"], rel(p))
    return out


def mine(rung: str, seed: int) -> dict:
    out = {}
    for arm in ("lace", "poe"):
        p = OUT / "ladder" / f"{rung}_seed{seed}" / f"rows_{arm}.json"
        if p.exists():
            d = jload(p)
            out[arm.upper() if arm == "lace" else "PoE"] = (d["per_corner"], rel(p), d["summary"])
    return out


def order_like(ref, rows):
    ix = {r["corner_id"]: r for r in rows}
    return [ix[r["corner_id"]] for r in ref]


def rung_block(rung: str) -> dict:
    per_seed = {}
    for s in SEEDS:
        src = shipped_sources(s) if rung == "shipped" else rung_sources(rung, s)
        my = mine(rung, s)
        per_seed[s] = {"src": src, "mine": my}
    arms = {}
    names = []
    for s in SEEDS:
        for a in list(per_seed[s]["src"]) + list(per_seed[s]["mine"]):
            if a not in names:
                names.append(a)
    for a in names:
        row = {"per_seed": {}, "sources": {}}
        for s in SEEDS:
            got = per_seed[s]["src"].get(a) or per_seed[s]["mine"].get(a)
            if got is None:
                row["per_seed"][s] = None
                continue
            rows = got[0]
            ent = {"all": mean(rows), "n_corners": len(rows)}
            if rung == "shipped":
                ent["row3_pair"] = mean([r for r in rows if r["family"] == "pair"])
                ent["row4_nway"] = mean([r for r in rows if r["family"] == "nway"])
            row["per_seed"][s] = ent
            row["sources"][s] = got[1]
            if len(got) > 2:
                smry = got[2]
                row.setdefault("budget_per_seed", {})[s] = smry.get("budget")
                row.setdefault("selected_per_seed", {})[s] = smry.get("selected")
        vals = [row["per_seed"][s]["all"] for s in SEEDS if row["per_seed"][s]]
        row["n_seeds"] = len(vals)
        row["mean_over_seeds"] = float(np.mean(vals)) if vals else None
        row["seeds_present"] = [s for s in SEEDS if row["per_seed"][s]]
        arms[a] = row
    paired = {}
    for riv in ("LACE", "PoE"):
        paired[riv] = {}
        for s in SEEDS:
            st = per_seed[s]["src"].get("kstar_store")
            rv = per_seed[s]["mine"].get(riv)
            if st is None or rv is None:
                paired[riv][s] = None
                continue
            a = order_like(st[0], rv[0])
            paired[riv][s] = paired_estimand(
                a, st[0], METRIC, DECORATION_MARGIN_NOVELTY, riv, "kstar_store"
            )
    verdict = {}
    for riv, ps in paired.items():
        ds = [ps[s]["mean_paired_delta"] for s in SEEDS if ps.get(s)]
        if len(ds) < len(SEEDS):
            verdict[riv] = {
                "verdict": "DECLARED NOT-RUN / INCOMPLETE",
                "seeds_present": [s for s in SEEDS if ps.get(s)],
            }
            continue
        beats = all((d >= DECORATION_MARGIN_NOVELTY for d in ds))
        ahead = float(np.mean(ds)) > 0
        verdict[riv] = {
            "mean_paired_delta_rival_minus_kstar_per_seed": ds,
            "mean_over_seeds": float(np.mean(ds)),
            "rule": "rival BEATS the CLU iff mean paired delta >= +1.0 novel/corner on EVERY seed (PROTOCOL 2a)",
            "verdict": "⛔⛔ RIVAL BEATS THE CLU -- HEADLINE NEGATIVE"
            if beats
            else "rival ahead, within margin"
            if ahead
            else "CLU ahead",
            "fw_strict_beats_beyond_margin_every_seed": all(
                (ps[s]["beats_beyond_margin"] for s in SEEDS)
            ),
        }
    return {"arms": arms, "paired_vs_kstar": paired, "verdict": verdict}


def part_a() -> dict:
    d = OUT / "scored" / "cell"
    res = d / "cell_results.json"
    if not res.exists():
        return {"status": "DECLARED NOT-RUN", "why": f"{rel(res)} absent"}
    agg = jload(res)
    mans = {
        s: jload(d / f"seed_{s}" / "cell_manifest.json")
        for s in SEEDS
        if (d / f"seed_{s}" / "cell_manifest.json").exists()
    }
    table = {}
    for arm, r in agg["arms"].items():
        per = {}
        for s, m in mans.items():
            a = m["arms"].get(arm)
            if a is None:
                continue
            per[s] = {
                "B_aggregate": a["external"]["composition_B"]["aggregate"],
                "in_corner_balacc": a["external"]["in_corner"]["balanced_accuracy"],
                "distinct_tuples_B": a["distinct"]["B"]["factor_tuples"],
                "deployed_bytes": (a.get("byte_footprint") or {}).get("deployed_bytes"),
            }
        table[arm] = {
            "per_seed": per,
            "n_seeds": len(per),
            "B_aggregate": r["external_B_aggregate"],
            "in_corner_balacc": r["external_in_corner_balacc"],
            "distinct_tuples_B": r["distinct_factor_tuples_B"],
            "byte_footprint": r["byte_footprint"],
            "source": rel(res) + f":arms.{arm}",
        }

    def per_seed(arm):
        return [
            mans[s]["arms"][arm]["external"]["composition_B"]["aggregate"]
            if arm in mans[s]["arms"]
            else None
            for s in SEEDS
            if s in mans
        ]

    k = per_seed("kstar")
    bars = {}
    if all((v is not None for v in k)) and len(k) == 3:
        s1 = {c: [kk - v for kk, v in zip(k, per_seed(f"kstar/{c}"))] for c in KCTL_SHIPPED}
        bars["S1_controls_collapse"] = {
            "delta_kstar_minus_control_per_seed": s1,
            "PASS": all((all((x >= 0.1 for x in v)) for v in s1.values())),
        }
        for nm, arm, rule in (
            ("S2_vs_clu", "clu", "every"),
            ("S3_vs_knn_proto", "knn_proto", "mean"),
            ("S4_vs_lace", "lace", "mean"),
            ("S4_vs_ebm_compose", "ebm_compose", "mean"),
        ):
            o = per_seed(arm)
            if any((v is None for v in o)):
                bars[nm] = {"status": "NOT RUN for some seed", "per_seed": o}
                continue
            ds = [a - b for a, b in zip(k, o)]
            if rule == "every":
                v = (
                    "K* BEATS"
                    if all((x >= 0.02 for x in ds))
                    else "TIE"
                    if all((abs(x) < 0.02 for x in ds))
                    else "LOSES / MIXED"
                )
            else:
                m = float(np.mean(ds))
                v = "WIN" if m >= 0.02 else "TIE" if abs(m) <= 0.02 else "LOSE"
            bars[nm] = {
                "delta_per_seed": ds,
                "mean_delta": float(np.mean(ds)),
                "margin": 0.02,
                "rule": rule,
                "verdict": v,
            }
    gates = {s: m.get("kstar_val") for s, m in mans.items()}
    return {
        "status": "RUN",
        "seeds_present": sorted(mans),
        "table": table,
        "bars": bars,
        "kstar_val_gates": gates,
        "not_run": {s: m.get("NOT_RUN") for s, m in mans.items()},
        "source": rel(res),
    }


def main() -> int:
    out = {
        "schema": "shapes3d/lace_poe_32k/GENTABLE-KSTAR/1",
        "metric_partB": METRIC,
        "margin_partB": DECORATION_MARGIN_NOVELTY,
        "seeds": list(SEEDS),
        "PART_A_generation_table": part_a(),
        "PART_B_rows_3_4": rung_block("shipped"),
        "PART_B_ladder": {f"k={r}": rung_block(r) for r in RUNGS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "GENTABLE-KSTAR.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", OUT / "GENTABLE-KSTAR.json")
    with open(OUT / "GENTABLE-KSTAR.md", "w", encoding="utf-8") as fh:
        fh.write(markdown(out))
    print("wrote", OUT / "GENTABLE-KSTAR.md")
    return 0


def _f(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def budget_cell(row) -> str:
    b = row.get("budget_per_seed") or {}
    b = next((v for v in b.values() if v), None)
    if not b:
        return ""
    if "classifier_params" in b:
        return f"{b['classifier_params']:,} clf params = {b['x_store_reference_classifier_only']}x ref ({b['x_store_reference_with_whitening']}x with whitening); fit {b['fit_rows']:,} rows, {b['epochs']} ep, {b['tuning_trials']} trials, ode"
    return f"{b['expert_params']:,} expert params = {b['x_store_reference']}x ref; fit {b['fit_rows']:,} rows, {b['epochs']} ep, {b['tuning_trials']} trials, SGLD"


def markdown(out) -> str:
    L = [
        "# GENTABLE-KSTAR — an earlier study",
        "",
        "Every number below is read from the artefact named in `GENTABLE-KSTAR.json` beside it. Seed count is a column. ⛔ LACE and PoE are ADDED rows; the learned-rival columns are intact.",
        "",
    ]
    A = out["PART_A_generation_table"]
    L += ["## PART A — the generation table (`exp_gen_cell`), run 3/3, generation-2 cache", ""]
    if A.get("status") != "RUN":
        L += [f"DECLARED NOT-RUN: {A.get('why')}", ""]
    else:
        L += [
            "| arm | n seeds | composition-B aggregate (s0 / s1 / s2) | mean ± sd | in-corner balacc mean | distinct tuples on B (mean) | deployed bytes (convention) |",
            "|---|---|---|---|---|---|---|",
        ]
        for arm, r in A["table"].items():
            ps = r["per_seed"]
            per = " / ".join((_f(ps[s]["B_aggregate"]) if s in ps else "—" for s in SEEDS))
            bl = r["byte_footprint"] or {}
            conv = (
                "four-part, never netted; " + str(bl.get("multiple_of_budget")) + "x ref"
                if arm.startswith("kstar")
                else str(bl.get("what", ""))[:40]
            )
            L.append(
                f"| `{arm}` | {r['n_seeds']} | {per} | {_f(r['B_aggregate']['mean'])} ± {_f(r['B_aggregate']['std'])} | {_f(r['in_corner_balacc']['mean'])} | {_f(r['distinct_tuples_B']['mean'], 1)} | {bl.get('deployed_bytes')} ({conv}) |"
            )
        L += [
            "",
            "### Pre-declared bars (PROTOCOL §1b)",
            "",
            "```",
            json.dumps(A["bars"], indent=1),
            "```",
            "",
        ]
    for title, blk in [("Shapes rows 3–4 (60 pair + 49 n-way corners)", out["PART_B_rows_3_4"])] + [
        (f"ladder rung {k}", v) for k, v in out["PART_B_ladder"].items()
    ]:
        L += [
            f"## PART B — {title}",
            "",
            "`distinct_valid_novel_tuples` per corner, mean over corners; seeds 0/1/2.",
            "",
            "| arm | n seeds | s0 | s1 | s2 | mean | own budget (LACE/PoE) |",
            "|---|---|---|---|---|---|---|",
        ]
        for arm, r in blk["arms"].items():
            ps = r["per_seed"]
            cells = [_f(ps[s]["all"], 3) if ps.get(s) else "—" for s in SEEDS]
            L.append(
                f"| `{arm}` | {r['n_seeds']} | "
                + " | ".join(cells)
                + f" | {_f(r['mean_over_seeds'], 3)} | {budget_cell(r)} |"
            )
        L += ["", "Verdict (PROTOCOL §2a):", ""]
        for riv, v in blk["verdict"].items():
            L.append(
                f"- **{riv}**: {v.get('verdict')} — per-seed paired Δ (rival − K\\*) {[round(x, 3) for x in v.get('mean_paired_delta_rival_minus_kstar_per_seed', [])]}"
            )
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
