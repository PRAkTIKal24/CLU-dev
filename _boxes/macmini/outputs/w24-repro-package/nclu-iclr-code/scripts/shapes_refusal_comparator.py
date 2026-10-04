"""Read-path-matched comparators and the refuse-token transformer for the refusal display rows.

python scripts/shapes_refusal_comparator.py --seed 0
"""

from __future__ import annotations
from nclu.paths import run_dir
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nclu.eval.baselines import refusal_comparator as RC

LOG_LINES = []


def log(msg=""):
    LOG_LINES.append(str(msg))
    print(msg, flush=True)


def _jdump(obj, path):
    def _d(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_d)
        fh.write("\n")


def merge_ready(out: Path) -> dict:
    out = Path(out)
    per = sorted(out.glob("BASELINES-READY.seed_*.json"))
    seeds, by_seed = ([], {})
    for p in per:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        by_seed[str(d["seed"])] = d
        seeds.append(int(d["seed"]))
    seeds = sorted(seeds)
    any_one = by_seed[str(seeds[0])]
    merged = {
        "study": any_one["study"],
        "branch": any_one["branch"],
        "scored_runs_spent": 0,
        "NOT_A_SCORED_RUN": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds_present": seeds,
        "n_records_total": sum((len(d["records"]) for d in by_seed.values())),
        "pins": any_one["pins"],
        "comparators": any_one["comparators"],
        "cost_fields": any_one["cost_fields"],
        "how_the_scripts_call_them": any_one["how_the_scripts_call_them"],
        "per_seed": by_seed,
        "SEED_COVERAGE": "seeds present: %s. Any seed NOT listed here is a DECLARED NOT-RUN for this study, never a zero: re-run scripts/deletion_baselines.py with --seed S and this file merges it in."
        % seeds,
    }
    _jdump(merged, out / "BASELINES-READY.json")
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n-fit",
        type=int,
        default=16384,
        help="fit-pool rows used for this BUILD pass (declared, not tuned)",
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--arms", default=",".join(RC.LEARNED_ARMS))
    ap.add_argument("--specs", default="value,conjunction,item")
    ap.add_argument("--n-refuse-per-class", type=int, default=4096)
    ap.add_argument("--out", default=str(run_dir("shapes3d/refusal_comparator")))
    a = ap.parse_args(argv)
    t_start = time.time()
    out = Path(a.out)
    cmp_root = out / "comparators"
    smoke_dir = out / "smoke" / ("seed_%d" % a.seed)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    wt = RC.assert_package_in_root()
    log("nclu.__file__ = %s  (inside %s)" % (wt["package_file"], wt["expected_root"]))
    import jax
    from nclu.experiments.factor_wells import (
        PINS,
        feature_cache_manifest,
        features_mmap,
        fit_pool,
        gather,
        load_condition,
        value_presence_bitmap,
    )

    cond = load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    F = features_mmap(a.seed)
    cman = feature_cache_manifest(a.seed)
    if int(cman.get("generation", -1)) != 2:
        raise AssertionError("feature cache is generation %r, not 2" % cman.get("generation"))
    feat_shape = tuple((int(x) for x in F.shape[1:]))
    D = int(np.prod(feat_shape))
    log(
        "feature cache: generation %d, scope=%s, sha256=%s..., feat_shape=%s, D=%d"
        % (cman["generation"], cman.get("scope"), cman["features_sha256"][:16], feat_shape, D)
    )
    pool = fit_pool(cond["train_rows"], PINS["n_pool"], PINS["pool_rng_seed"])
    fit_rows = np.sort(pool[: int(a.n_fit)])
    log("fit pool: %d drawn, %d used for this BUILD pass" % (pool.size, fit_rows.size))

    def gfn(rows):
        return gather(F, np.asarray(rows, np.int64))

    specs = {}
    if "value" in a.specs:
        specs["value"] = RC.value_spec(grid)
    if "conjunction" in a.specs:
        specs["conjunction"] = RC.conjunction_spec(grid)
    if "item" in a.specs:
        specs["item"] = RC.item_spec()
    arms = [s for s in a.arms.split(",") if s]
    key0 = jax.random.PRNGKey(int(a.seed) * 7717 + 20260921)
    log("")
    log(
        "== matched arms at the %d B pin (%d parameters) =="
        % (RC.PINS_REFUSAL["byte_pin"], RC.PINS_REFUSAL["param_target"])
    )
    armspecs, base = ({}, {})
    Y_fit = gfn(fit_rows)
    for fam in arms:
        key0, k = jax.random.split(key0)
        armspecs[fam] = RC.matched_arm(fam, cards, D, k, log=log)
    log("")
    log("== as-trained (the arm BEFORE any deletion) ==")
    for fam in arms:
        key0, k = jax.random.split(key0)
        base[fam] = RC.fit_as_trained(
            armspecs[fam], grid, fit_rows, Y_fit, cards, k, epochs=a.epochs, log=log
        )
        RC.save_comparator(base[fam], a.seed, grid, cman, cmp_root, "", part=None)
    records, budgets_by_spec, not_run = ({}, {}, {})
    for sname, spec in specs.items():
        log("")
        log("=" * 72)
        log("DELETION %s -- %s (%s)" % (sname.upper(), spec.name(), spec.human))
        part = RC.partition_fit_rows(grid, fit_rows, spec)
        log(
            "  rows: %d fit -> %d kept, %d deleted (%.3f%%)"
            % (
                part["n_fit_before"],
                part["n_fit_after"],
                part["n_deleted"],
                100.0 * part["deleted_fraction"],
            )
        )
        kept, deleted = (part["kept_rows"], part["deleted_rows"])
        bitmap = value_presence_bitmap(grid, kept)
        rng = np.random.default_rng(7717 + {"value": 1, "conjunction": 2, "item": 3}[sname])
        retain = kept[rng.choice(kept.size, size=min(deleted.size, kept.size), replace=False)]
        V_forget = RC.values_from_rows(grid, deleted)
        V_retain = RC.values_from_rows(grid, retain)
        T_forget, T_retain = (gfn(deleted), gfn(retain))

        def smoke(read, who):
            f, oor_f = read(V_forget)
            r, oor_r = read(V_retain)
            rec = {
                "forget_feature_mse": float(np.mean((f - T_forget) ** 2)),
                "retain_feature_mse": float(np.mean((r - T_retain) ** 2)),
                "n_forget": int(V_forget.shape[0]),
                "n_retain": int(V_retain.shape[0]),
                "out_of_range_forget": int(np.asarray(oor_f).sum()),
                "out_of_range_retain": int(np.asarray(oor_r).sum()),
            }
            rec["forget_minus_retain"] = rec["forget_feature_mse"] - rec["retain_feature_mse"]
            log(
                "    SMOKE %-52s forget_mse %.5f  retain_mse %.5f  (delta %+.5f)"
                % (
                    who,
                    rec["forget_feature_mse"],
                    rec["retain_feature_mse"],
                    rec["forget_minus_retain"],
                )
            )
            return rec

        log("")
        log("  -- the AS-TRAINED column (baselines that deleted nothing) --")
        for fam in arms:
            nm = RC.comparator_name(fam, "as_trained", "")
            rd = RC.load_comparator(a.seed, nm, cmp_root)
            records["%s@%s" % (nm, spec.name())] = {
                "kind": "as_trained",
                "arm": fam,
                "spec": part["spec"],
                "name_on_disk": nm,
                "cost": base[fam]["cost"],
                "param_count": armspecs[fam]["param_count"],
                "param_error": armspecs[fam]["param_error"],
                "checkpoint": str(cmp_root / ("seed_%d" % a.seed) / ("%s.rival.json" % nm)),
                "smoke": smoke(rd, nm),
            }
        spec_budgets = {}
        log("")
        log("  -- COMPARATOR 1: retrain from scratch WITHOUT --")
        for fam in arms:
            key0, k = jax.random.split(key0)
            res = RC.retrain_without(
                armspecs[fam], grid, part, gfn, cards, k, epochs=a.epochs, log=log
            )
            p = RC.save_comparator(res, a.seed, grid, cman, cmp_root, spec.name(), part=part)
            name = RC.comparator_name(fam, "retrain_without", spec.name())
            rd = RC.load_comparator(a.seed, name, cmp_root)
            records[name] = {
                "kind": "retrain_without",
                "arm": fam,
                "spec": part["spec"],
                "cost": res["cost"],
                "param_count": armspecs[fam]["param_count"],
                "param_error": armspecs[fam]["param_error"],
                "compute": res["compute"],
                "checkpoint": str(p),
                "fit": res["fit"],
                "smoke": smoke(rd, name),
            }
        log("")
        log("  -- COMPARATOR 2: approximate unlearning at MATCHED COMPUTE --")
        for method in ("neggrad", "finetune"):
            for fam in arms:
                key0, k = jax.random.split(key0)
                res = RC.approximate_unlearn(
                    base[fam],
                    armspecs[fam],
                    grid,
                    part,
                    gfn,
                    cards,
                    k,
                    method=method,
                    epochs=a.epochs,
                    log=log,
                )
                p = RC.save_comparator(res, a.seed, grid, cman, cmp_root, spec.name(), part=part)
                name = RC.comparator_name(fam, "unlearn_%s" % method, spec.name())
                rd = RC.load_comparator(a.seed, name, cmp_root)
                spec_budgets["%s/%s" % (fam, method)] = res["compute"]
                records[name] = {
                    "kind": "unlearn_%s" % method,
                    "arm": fam,
                    "spec": part["spec"],
                    "cost": res["cost"],
                    "param_count": armspecs[fam]["param_count"],
                    "param_error": armspecs[fam]["param_error"],
                    "compute": res["compute"],
                    "neggrad_ceiling_measured": res["neggrad_ceiling_measured"],
                    "checkpoint": str(p),
                    "fit": res["fit"],
                    "smoke": smoke(rd, name),
                }
        budgets_by_spec[sname] = RC.assert_compute_matched(spec_budgets)
        log(
            "  MATCHED COMPUTE CHECKED: every unlearner spent %s gradient row-visits"
            % format(budgets_by_spec[sname]["gradient_row_visits"], ",")
        )
        log("")
        log("  -- COMPARATOR 3: the refuse-token transformer --")
        applies = RC.refusal_row_applies(grid, part, bitmap)
        log(
            "    applicability (MEASURED): %d of %d requests built from the deleted rows are certified impossible%s"
            % (
                applies["n_certified_impossible"],
                applies["n_deleted_requests_tested"],
                " [%s]" % ",".join(applies["reasons"]) if applies["reasons"] else "",
            )
        )
        if not applies["applies"]:
            log("    DECLARED NOT-RUN: %s" % applies["why"])
            not_run["refuse_token@%s" % spec.name()] = {
                "comparator": "refuse_token",
                "spec": part["spec"],
                "applicability": applies,
                "reason": applies["why"],
                "NOT_a_zero": "there is no refusal DISPLAY row for this deletion at all, so a refusal number would be a number about nothing. Reported as a declared NOT-RUN.",
            }
            _jdump(
                {
                    "spec": part["spec"],
                    "counts": {k2: v for k2, v in part.items() if isinstance(v, (int, float))},
                    "matched_compute": budgets_by_spec[sname],
                    "refuse_token_NOT_RUN": not_run["refuse_token@%s" % spec.name()],
                    "records": {
                        k2: v
                        for k2, v in records.items()
                        if v["spec"]["label"] == part["spec"]["label"]
                    },
                },
                smoke_dir / ("%s.json" % sname),
            )
            continue
        key0, k = jax.random.split(key0)
        res = RC.fit_refuse_token_transformer(
            grid,
            part,
            gfn,
            cards,
            k,
            bitmap,
            D,
            seed=a.seed,
            epochs=a.epochs,
            n_per_class=a.n_refuse_per_class,
            include_out_of_range=False,
            log=log,
        )
        p = RC.save_comparator(res, a.seed, grid, cman, cmp_root, spec.name(), part=part)
        name = RC.comparator_name("transformer", "refuse_token", spec.name())
        rd = RC.load_comparator(a.seed, name, cmp_root)
        sm = smoke(rd, name)
        sm["refuse_score_on_deleted_requests"] = float(rd.refuse_score(V_forget).mean())
        sm["refuse_score_on_surviving_requests"] = float(rd.refuse_score(V_retain).mean())
        sm["refuse_rate_on_deleted_requests"] = float(rd.refuses(V_forget).mean())
        sm["false_refusal_on_surviving_requests"] = float(rd.refuses(V_retain).mean())
        log(
            "    SMOKE refuse: deleted-request refuse rate %.4f  false refusal %.4f"
            % (sm["refuse_rate_on_deleted_requests"], sm["false_refusal_on_surviving_requests"])
        )
        records[name] = {
            "kind": "refuse_token",
            "arm": "transformer",
            "applicability": applies,
            "spec": part["spec"],
            "cost": res["cost"],
            "param_count": res["arm"]["param_count"],
            "param_error": res["arm"]["param_error"],
            "separation": res["separation"],
            "sets": res["sets"],
            "checkpoint": str(p),
            "fit": res["fit"],
            "smoke": sm,
        }
        key0, k = jax.random.split(key0)
        diag = RC.fit_refuse_token_transformer(
            grid,
            part,
            gfn,
            cards,
            k,
            bitmap,
            D,
            seed=a.seed,
            epochs=a.epochs,
            n_per_class=max(a.n_refuse_per_class // 4, 256),
            include_out_of_range=True,
            log=log,
        )
        records[name]["out_of_range_diagnostic"] = {
            "separation": diag["separation"],
            "sets": diag["sets"],
            "why": "DECLARED DIAGNOSTIC, not the primary. A `tokens` input is a closed vocabulary of grid classes, so an off-grid request is encoded as its nearest grid class and this family CANNOT see out-of-range-ness. The number is printed so the structural limit is a measurement rather than a paragraph.",
        }
        _jdump(
            {
                "spec": part["spec"],
                "counts": {k2: v for k2, v in part.items() if isinstance(v, (int, float))},
                "matched_compute": budgets_by_spec[sname],
                "records": {
                    k2: v
                    for k2, v in records.items()
                    if v["spec"]["label"] == part["spec"]["label"]
                },
            },
            smoke_dir / ("%s.json" % sname),
        )
    log("")
    log("== BASELINES-READY ==")
    ready = {
        "study": "refusal comparator",
        "branch": "refusal comparator",
        "scored_runs_spent": 0,
        "NOT_A_SCORED_RUN": True,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "worktree_assertion": wt,
        "seed": int(a.seed),
        "build_scale": {
            "n_fit_rows_used": int(fit_rows.size),
            "n_pool_drawn": int(pool.size),
            "epochs": int(a.epochs or RC.PINS_REFUSAL["retrain_epochs"]),
            "DECLARED": "this is the BUILD/SMOKE scale. The study's scored leg sets --n-fit to the pool size it scores on; nothing in the comparators changes with it.",
        },
        "pins": RC.PINS_REFUSAL,
        "feature_cache": {
            "generation": int(cman["generation"]),
            "scope": cman.get("scope"),
            "features_sha256": cman["features_sha256"],
            "root": os.environ.get("NCLU_FEATURES_ROOT"),
        },
        "comparators": RC.COMPARATORS,
        "matched_compute": budgets_by_spec,
        "cost_fields": list(RC.COST_FIELDS),
        "how_the_scripts_call_them": {
            "load": "from nclu.eval.baselines.refusal_comparator import load_comparator; r = load_comparator(seed, name, root)",
            "load_without_comparator": "from nclu.eval.baselines.api import load_rival; r = load_rival(seed, name, root=<comparators dir>)",
            "contract": "values (N, F) float, nan = not requested -> (features (N, D) float32, out_of_range (N,) bool) -- the same contract exp_gen_factor_wells.arm_features uses",
            "refusal": "a refuse_token comparator comes back as RefuseTokenRead: r(values) gives features, r.refuse_score(values) gives the refuse token in [0,1], r.refuses(values) applies the declared threshold",
            "comparators_root": str(cmp_root),
        },
        "records": records,
        "DECLARED_NOT_RUN": {
            "measured_not_applicable": not_run,
            "our_own_arm_cost_line": "the single design's own deletion cost line (seconds / bytes rewritten / data rows read = 0) is NOT measured here. It belongs to the object the crossover study builds at K*, which does not exist yet; measuring it against a K=5 stand-in would print a cost for an object no table carries. NOT-RUN with that reason, never a zero.",
            "scored_corners": "no held-out test corner, query set or scored request is read by this study. Scored budget spent: 0, by design.",
            "seeds_1_and_2": "this pass builds seed %d. The comparators are seed-parameterised and the study runs the seeds it scores; re-run with --seed 1 / --seed 2."
            % a.seed,
            "external_judge": "no judge accuracy is reported. These comparators emit FEATURES; the crossover study renders and judges them with its own frozen decoder and hash-asserted external classifier. The smoke number is feature MSE against the cached ground truth, which needs no instrument.",
        },
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    _jdump(ready, out / ("BASELINES-READY.seed_%d.json" % a.seed))
    merged = merge_ready(out)
    log(
        "wrote %s  (%d comparator records, %.1f s); merged BASELINES-READY.json now carries seeds %s and %d records in total"
        % (
            out / ("BASELINES-READY.seed_%d.json" % a.seed),
            len(records),
            time.time() - t_start,
            merged["seeds_present"],
            merged["n_records_total"],
        )
    )
    with open(smoke_dir / "run.log", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
