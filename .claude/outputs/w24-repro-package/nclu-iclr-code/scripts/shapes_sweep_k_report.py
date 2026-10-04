"""shapes sweep k report."""

from __future__ import annotations
from nclu.paths import run_dir
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

RC = Path(str(run_dir("shapes3d/k_sweep")))
OUT = Path(str(run_dir("shapes3d/k_sweep_seeds")))
SWEEP = (5, 8, 10, 12, 16, 20, 24, 28, 32)
SEEDS = (0, 1, 2)
WITHIN = 0.02
CROSS = 0.1
BYTE_CONVENTION = "HONEST FOUR-PART TOTAL, uncompressed raw array bytes, SUMMED AND NEVER NETTED: (1) atoms = sum_k modes_k.size x 4 B (float32, the deployed content); (2) covariance + count = sum_k stat_floats_k x 8 B (float64 -- the precision actually retained; counting it at 4 would be netting by dtype) + 11,488 B of count table; (3) address book = TensorAddress.byte_footprint['address_bytes']; (4) coupling/hole table = 5,288 + 1,322 = 6,610 B. No entropy coding, no deduplication, no subtraction of any part from any other. The 524,288 B pin is a REFERENCE COLUMN for continuity with round 5-8, NOT a budget in this sweep."
NEAR_24 = (20, 24, 28)
PARTS = (
    "p1_atom_bytes",
    "p2_covariance_plus_count_bytes",
    "p3_address_book_bytes",
    "p4_coupling_hole_table_bytes",
)
CONTROL_NAMES = (
    "landscape_removed",
    "landscape_shuffled",
    "landscape_shuffled_across_wells",
    "random_atoms",
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_rows():
    rows, prov = ({}, {})
    for K in SWEEP:
        rows[K] = {}
        for s in SEEDS:
            for tree, tag in ((OUT, "shapes3d/k_sweep_seeds"), (RC, "shapes3d/k_sweep")):
                p = tree / "val_sweep" / f"val_K{K}_seed{s}" / "row.json"
                if p.exists():
                    rows[K][s] = json.loads(p.read_text(encoding="utf-8"))
                    prov[f"K{K:02d}_seed{s}"] = {
                        "source_round": tag,
                        "path": str(p).replace("\\", "/"),
                        "sha256": _sha(p),
                    }
                    break
    return (rows, prov)


def _stats(xs):
    a = np.asarray(xs, float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else None,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def build(rows, prov):
    doc = {
        "SMOKE_NOT_A_CLAIM": True,
        "artefact": "CROSSOVER-3SEED",
        "round": "an earlier study",
        "condition": "val",
        "scored_runs_spent": 0,
        "notation": "K = wells.  n in this artefact is the SEED count and it is a column on every table.  (n = factors jointly composed is a different quantity and is not used here.)",
        "seeds": list(SEEDS),
        "sweep": list(SWEEP),
        "byte_convention": BYTE_CONVENTION,
        "instrument": {
            "runner": "scripts/crossover_seeds.py -> scripts/val_sweep.py main verbatim, module-level OUT redirected and nothing else",
            "partition": "ward_balanced_labels (label-free correlation-dendrogram cut)",
            "rank_rule": "A = min(64, min_k d_k - 1), strictly below the narrowest well",
            "read": "TensorAddress product address, R.READ_PINS rank/ridge/sweeps, 90/10 split, pinned_basis",
            "metric": "24-corner val read, mean `valid`, score_corners code path",
            "feature_cache": "GENERATION 2 (the earlier baseline set cache_generation2), asserted in-process per seed by assert_generation2",
        },
        "frozen_rule": {
            "file": str(RC / "SELECTION-RULE.md").replace("\\", "/"),
            "declared_at": "not recorded",
            "within": WITHIN,
            "crossover": CROSS,
            "K_star": "max { K in SWEEP : v(5) - mean_seeds v(K) <= 0.02 }",
            "K_cross": "min { K in SWEEP : v(5) - mean_seeds v(K) > 0.10 }",
            "note": "the rule is the earlier study's, applied here unchanged at n = 3; this study does not move it and does not re-derive it",
        },
        "u_shape_test_declared_in": "LOG.md section 5, written before the seed-1 and seed-2 totals existed",
        "provenance": dict(sorted(prov.items())),
        "per_K": {},
        "not_run": [],
    }
    v5 = None
    for K in SWEEP:
        per = rows[K]
        if not per:
            doc["not_run"].append(
                {"K": K, "reason": "DECLARED NOT-RUN -- no row.json on disk in either tree"}
            )
            continue
        ss = sorted(per)
        ps = {}
        for s in ss:
            r = per[s]
            led = r["footprint_four_part"]
            ex = r.get("exactness")
            ps[str(s)] = {
                "seed": s,
                "A_requested": int(r["A_requested"]),
                "well_width_min": int(min(r["well_sizes"])),
                "well_width_max": int(max(r["well_sizes"])),
                "rank_over_width_min": float(r["rank_over_width_min"]),
                "rank_over_width_max": float(r["rank_over_width_max"]),
                "val_composition": float(r["composition_val"]["valid_mean"]),
                "val_composition_corner_sd": float(r["composition_val"]["valid_sd"]),
                "val_composition_n_corners": int(r["composition_val"]["n_corners"]),
                "footprint_four_part": {
                    k: led[k]
                    for k in (
                        "p1_atom_bytes",
                        "p2_covariance_plus_count_bytes",
                        "p2_covariance_bytes",
                        "p2_count_bytes",
                        "p3_address_book_bytes",
                        "p4_coupling_hole_table_bytes",
                        "honest_total_bytes",
                        "reference_pin_bytes",
                        "times_reference_pin",
                        "netted",
                    )
                },
                "controls": {
                    n: {
                        "valid_mean": c["valid_mean"],
                        "valid_max": c["valid_max"],
                        "drop_from_clu": c["drop_from_clu"],
                    }
                    for n, c in sorted((r["controls"] or {}).items())
                },
                "exactness": {
                    k: ex[k]
                    for k in (
                        "definition",
                        "n_rows_deleted",
                        "max_abs_mode_dev",
                        "max_subspace_projector_dist",
                        "ranks_agree",
                        "rows_refitted",
                        "data_passes_during_deletion",
                        "items_kept",
                    )
                }
                if ex
                else None,
            }
        comp = [ps[str(s)]["val_composition"] for s in ss]
        tot = [ps[str(s)]["footprint_four_part"]["honest_total_bytes"] for s in ss]
        m = float(np.mean(comp))
        if K == 5:
            v5 = m
        row = {
            "K": K,
            "n_seeds": len(ss),
            "seeds": ss,
            "per_seed": ps,
            "val_composition": _stats(comp),
            "drop_from_K5": float(v5 - m) if v5 is not None else None,
            "honest_total_bytes": _stats(tot),
            "parts_across_seeds": {
                p: _stats([ps[str(s)]["footprint_four_part"][p] for s in ss]) for p in PARTS
            },
            "netted": False,
            "byte_convention": "see the document-level `byte_convention` field",
        }
        gate = {}
        for s in ss:
            for n, c in ps[str(s)]["controls"].items():
                gate.setdefault(n, []).append(c["valid_max"])
        gmean = {}
        for s in ss:
            for n, c in ps[str(s)]["controls"].items():
                gmean.setdefault(n, []).append(c["valid_mean"])
        row["control_gate"] = {
            n: {
                "per_seed_valid_max": v,
                "per_seed_valid_mean": gmean[n],
                "worst_over_seeds_and_corners": float(np.max(v)),
                "mean_over_seeds_and_corners": float(np.mean(gmean[n])),
                "collapses": bool(np.max(v) <= 1e-09),
            }
            for n, v in sorted(gate.items())
        }
        row["all_landscape_C_controls_collapse"] = bool(
            len(row["control_gate"]) == len(CONTROL_NAMES)
            and all((g["collapses"] for g in row["control_gate"].values()))
        )
        row["primary_C2_control_collapses"] = bool(
            "landscape_shuffled" in row["control_gate"]
            and row["control_gate"]["landscape_shuffled"]["collapses"]
        )
        row["controls_not_collapsing"] = [
            {
                "control": n,
                "seed": ss[i],
                "valid_max_over_corners": float(g["per_seed_valid_max"][i]),
                "valid_mean_over_corners": float(g["per_seed_valid_mean"][i]),
                "arm_valid_mean": ps[str(ss[i])]["val_composition"],
            }
            for n, g in row["control_gate"].items()
            for i in range(len(ss))
            if g["per_seed_valid_max"][i] > 1e-09
        ]
        row["selection_resolvability"] = {
            "tolerance": WITHIN,
            "composition_seed_sd": row["val_composition"]["sd"],
            "seed_sd_exceeds_tolerance": None
            if row["val_composition"]["sd"] is None
            else bool(row["val_composition"]["sd"] > WITHIN),
        }
        ex = [ps[str(s)]["exactness"] for s in ss if ps[str(s)]["exactness"]]
        sub = [
            e["max_subspace_projector_dist"]
            for e in ex
            if e["max_subspace_projector_dist"] is not None
        ]
        row["exactness_worst"] = (
            {
                "n_seeds": len(ex),
                "max_abs_mode_dev": max((e["max_abs_mode_dev"] for e in ex)),
                "max_subspace_projector_dist": max(sub) if sub else None,
                "ranks_agree_all_seeds": all((e["ranks_agree"] for e in ex)),
                "rows_refitted": 0,
                "data_passes_during_deletion": 0,
            }
            if ex
            else None
        )
        doc["per_K"][str(K)] = row
    ran = [K for K in SWEEP if str(K) in doc["per_K"]]
    if v5 is None:
        doc["K_star"] = None
        doc["K_star_reason"] = "K=5 reference row absent; the rule is not applicable"
        doc["K_crossover"] = None
        doc["K_crossover_note"] = "not applicable without the K=5 reference row"
    else:
        ok = [K for K in ran if v5 - doc["per_K"][str(K)]["val_composition"]["mean"] <= WITHIN]
        cross = [K for K in ran if v5 - doc["per_K"][str(K)]["val_composition"]["mean"] > CROSS]
        doc["K_star"] = int(max(ok)) if ok else None
        doc["K_star_candidates"] = ok
        doc["K_crossover"] = int(min(cross)) if cross else None
        doc["K_crossover_note"] = (
            None
            if cross
            else f"no K in the swept range drops more than {CROSS} below K=5 at n = 3; the range is NOT extended to manufacture a crossover"
        )
    doc["u_shape"] = u_shape(doc)
    return doc


def u_shape(doc):
    """S1-S4 of LOG.md section 5, applied literally."""
    Ks = [K for K in SWEEP if str(K) in doc["per_K"]]
    seeds = sorted({s for K in Ks for s in doc["per_K"][str(K)]["seeds"]})
    full = [K for K in Ks if all((str(s) in doc["per_K"][str(K)]["per_seed"] for s in seeds))]
    per_seed = {}
    for s in seeds:
        T = {
            K: doc["per_K"][str(K)]["per_seed"][str(s)]["footprint_four_part"]["honest_total_bytes"]
            for K in full
        }
        kmin = min(full, key=lambda K: T[K])
        i = full.index(kmin)
        dec = [full[j] for j in range(1, i + 1) if T[full[j]] >= T[full[j - 1]]]
        inc = [full[j] for j in range(i + 1, len(full)) if T[full[j]] <= T[full[j - 1]]]
        per_seed[str(s)] = {
            "totals_by_K": {str(K): int(T[K]) for K in full},
            "K_argmin": int(kmin),
            "min_bytes": int(T[kmin]),
            "S1_violations_before_min": dec,
            "S1_violations_after_min": inc,
            "S1_u_shaped": not dec and (not inc),
            "S2_min_near_K24": int(kmin) in NEAR_24,
        }
    minima = sorted({per_seed[str(s)]["K_argmin"] for s in seeds})
    s1 = all((per_seed[str(s)]["S1_u_shaped"] for s in seeds))
    s2 = all((per_seed[str(s)]["S2_min_near_K24"] for s in seeds))
    res = None
    if s2 and minima:
        kstar = min(
            NEAR_24,
            key=lambda k: (
                abs(k - float(np.mean([per_seed[str(s)]["K_argmin"] for s in seeds]))),
                k,
            ),
        )
        mk = doc["per_K"][str(kstar)]["honest_total_bytes"]
        band = mk["sd"] or 0.0
        nb = {}
        for k in NEAR_24:
            if k == kstar or str(k) not in doc["per_K"]:
                continue
            g = doc["per_K"][str(k)]["honest_total_bytes"]["mean"] - mk["mean"]
            nb[str(k)] = {"gap_bytes": float(g), "gap_exceeds_seed_band": bool(g > band)}
        res = {
            "K_used": kstar,
            "seed_band_sd_bytes": float(band),
            "neighbours": nb,
            "minimum_resolvable": bool(
                nb and all((v["gap_exceeds_seed_band"] for v in nb.values()))
            ),
        }
    return {
        "claim": "the honest four-part total is U-shaped in K with its minimum near K = 24",
        "test_declared_in": "LOG.md section 5 (S1-S4), written before the seed-1 and seed-2 totals existed",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "K_with_all_seeds": full,
        "per_seed": per_seed,
        "per_seed_minima": {str(s): per_seed[str(s)]["K_argmin"] for s in seeds},
        "distinct_minima": minima,
        "S1_all_seeds_u_shaped": bool(s1),
        "S2_all_minima_near_K24": bool(s2),
        "S3_verdict": "CONFIRMED" if s1 and s2 else "REFUTED",
        "S4_resolvability": res,
        "note": "S3 REFUTED is a RESULT, not a failure: the per-seed minima are then the reportable quantity and the sentence is not claimed.",
    }


def _opt(x):
    return "n/a (K=5 not present)" if x is None else "%+.4f" % x


def _sd(x):
    return f"{x:,.0f}" if x is not None else "--"


def fmt_md(doc) -> str:
    L = []
    A = L.append
    u = doc["u_shape"]
    A("# CROSSOVER-3SEED -- the K-crossover at n = 3 seeds")
    A("")
    A(
        "**Condition `val`, the free selection condition (protocol A.2). `SMOKE_NOT_A_CLAIM`. runs spent by this study: ZERO.**"
    )
    A("")
    A(
        "Every number below is read from a `row.json` listed in the `provenance` block of `CROSSOVER-3SEED.json` and in section 6. No number in this file lives only in prose."
    )
    A("")
    A(
        "**Notation.** `K` = wells. `n` in this artefact is the SEED count and it is a column on every table. (`n` = factors jointly composed is a different quantity and is not used here.)"
    )
    A("")
    A("## Byte convention")
    A("")
    A("> " + BYTE_CONVENTION.replace("; ", ";  \n> "))
    A("")
    A("## 1. Composition and the four-part total vs K, n = 3 at every K")
    A("")
    A(
        "| K | n | A | well width | rank/width | val comp mean | seed sd | seed band | drop vs K=5 | 4-part total mean B | seed sd B | seed band B |"
    )
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for K in SWEEP:
        r = doc["per_K"].get(str(K))
        if not r:
            A(f"| {K} | -- | -- | -- | -- | DECLARED NOT-RUN | | | | | | |")
            continue
        p0 = r["per_seed"][str(r["seeds"][0])]
        c, t = (r["val_composition"], r["honest_total_bytes"])
        rwlo = min((r["per_seed"][str(s)]["rank_over_width_min"] for s in r["seeds"]))
        rwhi = max((r["per_seed"][str(s)]["rank_over_width_max"] for s in r["seeds"]))
        csd = f"{c['sd']:.4f}" if c["sd"] is not None else "--"
        A(
            f"| {K} | {r['n_seeds']} | {p0['A_requested']} | {p0['well_width_min']}..{p0['well_width_max']} | {rwlo:.4f}..{rwhi:.4f} | {c['mean']:.4f} | {csd} | [{c['min']:.4f}, {c['max']:.4f}] | {_opt(r['drop_from_K5'])} | {t['mean']:,.0f} | {_sd(t['sd'])} | [{t['min']:,.0f}, {t['max']:,.0f}] |"
        )
    A("")
    A(
        f"Seeds: {doc['seeds']}. Frozen rule (applied here unchanged at n = 3): **K\\* = {doc.get('K_star')}**, **K_crossover = {doc.get('K_crossover')}**"
        + (f" -- {doc['K_crossover_note']}" if doc.get("K_crossover_note") else "")
        + "."
    )
    A("")
    A("### 1b. Is the frozen rule resolvable at n = 3?")
    A("")
    A(
        "The frozen tolerance is 0.02 in composition. The table below puts the SEED sd beside it at every K. Where the seed sd is wider than the tolerance the rule cannot separate two K on three seeds, and the honest statement is that it cannot -- not that it did."
    )
    A("")
    A("| K | n | composition seed sd | tolerance | seed sd wider than tolerance |")
    A("|---|---|---|---|---|")
    for K in SWEEP:
        r = doc["per_K"].get(str(K))
        if not r:
            A(f"| {K} | -- | DECLARED NOT-RUN | | |")
            continue
        q = r["selection_resolvability"]
        sdv = f"{q['composition_seed_sd']:.4f}" if q["composition_seed_sd"] is not None else "--"
        A(
            f"| {K} | {r['n_seeds']} | {sdv} | {q['tolerance']:.2f} | {('YES' if q['seed_sd_exceeds_tolerance'] else 'no')} |"
        )
    A("")
    A("## 2. The honest four-part footprint, per K per seed -- never netted")
    A("")
    A(
        "| K | seed | atoms B | cov+count B | address B | coupling/hole B | **honest total B** | x 524,288 B REFERENCE |"
    )
    A("|---|---|---|---|---|---|---|---|")
    for K in SWEEP:
        r = doc["per_K"].get(str(K))
        if not r:
            A(f"| {K} | -- | DECLARED NOT-RUN | | | | | |")
            continue
        for s in r["seeds"]:
            g = r["per_seed"][str(s)]["footprint_four_part"]
            A(
                f"| {K} | {s} | {g['p1_atom_bytes']:,} | {g['p2_covariance_plus_count_bytes']:,} | {g['p3_address_book_bytes']:,} | {g['p4_coupling_hole_table_bytes']:,} | **{g['honest_total_bytes']:,}** | {g['times_reference_pin']:.4f} |"
            )
    A("")
    A(
        "The four parts are printed separately and summed. No part is subtracted from any other and no part is dropped. The 524,288 B column is a REFERENCE for continuity with round 5-8, not a budget in this sweep."
    )
    A("")
    A("## 3. The U-shape, decided by the test declared before the numbers")
    A("")
    A(f"**Claim under test:** {u['claim']}.")
    A("")
    A(
        f"**Test:** {u['test_declared_in']}. S1 = strictly decreasing to the minimum and strictly increasing after it. S2 = the minimum lies in {list(NEAR_24)}. S3 = S1 and S2 on all three seeds."
    )
    A("")
    A(
        "| seed | "
        + " | ".join((f"K={K}" for K in u["K_with_all_seeds"]))
        + " | argmin K | S1 U-shaped | S2 near K=24 |"
    )
    A("|---|" + "---|" * (len(u["K_with_all_seeds"]) + 3))
    for s in u["seeds"]:
        d = u["per_seed"][str(s)]
        cells = " | ".join(
            (
                f"**{d['totals_by_K'][str(K)]:,}**"
                if K == d["K_argmin"]
                else f"{d['totals_by_K'][str(K)]:,}"
                for K in u["K_with_all_seeds"]
            )
        )
        A(
            f"| {s} | {cells} | **{d['K_argmin']}** | {('yes' if d['S1_u_shaped'] else 'NO')} | {('yes' if d['S2_min_near_K24'] else 'NO')} |"
        )
    A("")
    A(
        "Bold cell = that seed's minimum. Bytes are the honest four-part total under the convention named above."
    )
    A("")
    A(
        f"- per-seed minima: `{u['per_seed_minima']}`; distinct minima across seeds: `{u['distinct_minima']}`"
    )
    A(f"- S1 (every seed U-shaped): **{u['S1_all_seeds_u_shaped']}**")
    for s in u["seeds"]:
        d = u["per_seed"][str(s)]
        if not d["S1_u_shaped"]:
            A(
                f"  - seed {s}: violations before the minimum at K {d['S1_violations_before_min']}, after the minimum at K {d['S1_violations_after_min']}"
            )
    A(f"- S2 (every minimum in {list(NEAR_24)}): **{u['S2_all_minima_near_K24']}**")
    A(f"- **S3 VERDICT: {u['S3_verdict']}**. {u['note']}")
    if u["S4_resolvability"]:
        r4 = u["S4_resolvability"]
        A(
            f"- S4 resolvability at K = {r4['K_used']}: seed band sd {r4['seed_band_sd_bytes']:,.0f} B; neighbour gaps "
            + ", ".join(
                (
                    f"K={k} {v['gap_bytes']:,.0f} B ({('clears' if v['gap_exceeds_seed_band'] else 'INSIDE')} the band)"
                    for k, v in sorted(r4["neighbours"].items(), key=lambda kv: int(kv[0]))
                )
            )
            + f"; minimum resolvable: **{r4['minimum_resolvable']}**"
        )
    else:
        A(
            "- S4 resolvability: not evaluated -- S2 did not hold, so there is no single minimum to resolve."
        )
    A("")
    A("## 4. The landscape controls, every K, every seed")
    A("")
    A(
        "| K | n | landscape_removed | landscape_shuffled | landscape_shuffled_across_wells | random_atoms | all four collapse (strict) |"
    )
    A("|---|---|---|---|---|---|---|")
    for K in SWEEP:
        r = doc["per_K"].get(str(K))
        if not r:
            A(f"| {K} | -- | DECLARED NOT-RUN | | | | |")
            continue
        g = r["control_gate"]
        cells = [
            f"{g[n]['worst_over_seeds_and_corners']:.4f} (mean {g[n]['mean_over_seeds_and_corners']:.6f})"
            if n in g
            else "--"
            for n in CONTROL_NAMES
        ]
        A(
            f"| {K} | {r['n_seeds']} | "
            + " | ".join(cells)
            + f" | {('YES' if r['all_landscape_C_controls_collapse'] else 'NO')} |"
        )
    A("")
    A(
        "Each cell is `worst` = max over the 24 val corners then max over the 3 seeds, with the mean over corners and seeds in brackets. A control that does not collapse means the store is DECORATIVE at that K and no row may be claimed for it. The gate above is STRICT: it asks for exactly zero on the worst of the 24 corners on the worst of the 3 seeds."
    )
    A("")
    bad = [
        (K, c)
        for K in SWEEP
        for c in (doc["per_K"].get(str(K)) or {}).get("controls_not_collapsing", [])
    ]
    if bad:
        A("Cells that did NOT collapse under the strict gate, named rather than averaged away:")
        A("")
        A(
            "| K | seed | control | valid max over 24 corners | valid mean over 24 corners | the arm at that cell |"
        )
        A("|---|---|---|---|---|---|")
        for K, c in bad:
            A(
                f"| {K} | {c['seed']} | `{c['control']}` | {c['valid_max_over_corners']:.4f} | {c['valid_mean_over_corners']:.6f} | {c['arm_valid_mean']:.4f} |"
            )
        A("")
    else:
        A("No cell failed the strict gate.")
        A("")
    A(
        "`landscape_shuffled` (the SHIPPED protocol C.2 definition -- atom VALUES permuted inside each well) is the control declared the verdict is taken from, and it is reported beside the strict gate, never instead of it. `landscape_shuffled_across_wells` is the task file's literal wording and is weaker by construction: the earlier study's own `scripts/crossover_lib.py` records that its permutation is uniform with only the IDENTITY rejected, so about one well in K keeps its own atoms. Both are reported; neither is dropped."
    )
    A("")
    A("| K | n | strict gate (all four) | primary C.2 control `landscape_shuffled` |")
    A("|---|---|---|---|")
    for K in SWEEP:
        r = doc["per_K"].get(str(K))
        if not r:
            A(f"| {K} | -- | DECLARED NOT-RUN | |")
            continue
        A(
            f"| {K} | {r['n_seeds']} | {('PASS' if r['all_landscape_C_controls_collapse'] else 'FAIL')} | {('collapses' if r['primary_C2_control_collapses'] else 'DOES NOT COLLAPSE')} |"
        )
    A("")
    A("## 5. Store-scoped exactness, worst case over seeds")
    A("")
    A(
        'Definition: the frozen encoder and decoder are declared OUTSIDE the deletable object and are NOT refit. Exactness is store-scoped -- `downdate(delete the red rows)` against a store fitted from scratch without those rows, same partition, same rank. Post-deletion adaptation is light and data-free; it is not "no changes". A rival deletes only by retrain-without, and that cost is printed in the earlier study\'s own deletion artefacts, not here.'
    )
    A("")
    A(
        "| K | n | max abs mode deviation | max subspace projector distance | ranks agree | rows refitted | data passes during deletion |"
    )
    A("|---|---|---|---|---|---|---|")
    for K in SWEEP:
        r = doc["per_K"].get(str(K))
        if not r or not r["exactness_worst"]:
            A(f"| {K} | -- | DECLARED NOT-RUN | | | | |")
            continue
        e = r["exactness_worst"]
        sub = (
            f"{e['max_subspace_projector_dist']:.3e}"
            if e["max_subspace_projector_dist"] is not None
            else "--"
        )
        A(
            f"| {K} | {e['n_seeds']} | {e['max_abs_mode_dev']:.3e} | {sub} | {e['ranks_agree_all_seeds']} | {e['rows_refitted']} | {e['data_passes_during_deletion']} |"
        )
    A("")
    A("## 6. Provenance -- every cell, its round and its row hash")
    A("")
    A("| cell | source round | row.json sha256 (first 16) |")
    A("|---|---|---|")
    for k, v in doc["provenance"].items():
        A(f"| {k} | {v['source_round']} | `{v['sha256'][:16]}` |")
    A("")
    if doc["not_run"]:
        A("## 7. DECLARED NOT-RUN")
        A("")
        for nr in doc["not_run"]:
            A(f"- K = {nr['K']}: {nr['reason']}")
        A("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    rows, prov = load_rows()
    doc = build(rows, prov)
    jp = out / "CROSSOVER-3SEED.json"
    jp.write_text(
        json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    mp = out / "CROSSOVER-3SEED.md"
    mp.write_text(fmt_md(doc), encoding="utf-8", newline="\n")
    print(f"wrote {jp}  sha256 {_sha(jp)}")
    print(f"wrote {mp}  sha256 {_sha(mp)}")
    print("n_seeds per K: " + str({K: doc["per_K"][K]["n_seeds"] for K in doc["per_K"]}))
    print(f"K* = {doc.get('K_star')}   K_crossover = {doc.get('K_crossover')}")
    print(
        f"U-shape S3 verdict: {doc['u_shape']['S3_verdict']}   per-seed minima {doc['u_shape']['per_seed_minima']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
