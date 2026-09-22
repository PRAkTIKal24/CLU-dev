"""w15 -- the markdown tables, rendered from T7-3SEEDS.json and nothing else.

Every table prints its seed count as a column or in its caption, every byte figure
travels with its convention named, and a missing leg prints DECLARED NOT-RUN with the
absent path.  Nothing is computed here that is not already in the JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("C:/chlu/outputs/w15-single-design-seeds")
SEEDS = [0, 1, 2]
CONTROLS = ("landscape_removed", "landscape_shuffled",
            "landscape_shuffled_across_wells", "random_atoms")
RIVALS = ("mlp", "lstm", "transformer", "cont_mlp")
SCOPES = ("value", "conjunction", "conjunction_live", "item")


def f(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "**YES**" if x else "**NO**"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    d = json.loads((OUT / "T7-3SEEDS.json").read_text(encoding="utf-8"))
    L = []
    A = L.append

    A("# w15 — T7 AT THREE SEEDS, AND THE DELETION ROWS")
    A("")
    A(f"Rendered from `{OUT / 'T7-3SEEDS.json'}` by `scripts/w15_findings_tables.py`. "
      f"Built {d['built_utc']}.")
    A("")
    A(f"**Condition TEST · K\\* = {d['K_star']} · n = {d['n_seeds']} seeds "
      f"{d['seeds']} · scored runs spent by this hub: "
      f"{d['SCORED_RUNS_SPENT_BY_THIS_HUB']}**")
    A("")
    A(f"Notation: {d['notation']}")
    A("")

    # --- T7 -----------------------------------------------------------------
    t7 = d["T7_rows_3_and_4"]
    A("## T7 — rows 3 and 4 at K\\* = 32 on TEST, **n = 3, seeds [0, 1, 2]**")
    A("")
    A("Metric: `distinct_valid_novel_tuples` **per corner**; the corner is the unit. "
      "Row 3 = 60 tiled `pair-*` sub-corners, row 4 = 49 Montero `nway-*` corners.")
    A("")
    A("| arm | role | row | seed 0 | seed 1 | seed 2 | mean ± sd | n seeds |")
    A("|---|---|---|---|---|---|---|---|")
    for arm, rec in t7["rows"].items():
        for row, key in (("row 3 (pair)", "row3_pair"),
                         ("row 4 (n-way)", "row4_nway")):
            cells = []
            for s in SEEDS:
                v = rec["per_seed"][f"seed{s}"]
                cells.append("**DECLARED NOT-RUN**" if "DECLARED_NOT_RUN" in v
                             else f(v[key], 3))
            agg = rec[f"{key}_over_seeds"]
            sd = f(agg.get("sd"), 3) if agg.get("sd") is not None else "n/a"
            A(f"| `{arm}` | {rec['role']} | {row} | " + " | ".join(cells)
              + f" | {f(agg.get('mean'), 3)} ± {sd} | {agg.get('n')} |")
    A("")

    # --- control gate --------------------------------------------------------
    g = d["control_gate"]
    A("## The charter-C control gate — measured at EVERY seed, never assumed")
    A("")
    A(f"> {g['rule']}")
    A("")
    A("| seed | arm (all 109 corners) | `landscape_removed` | `landscape_shuffled` | "
      "`landscape_shuffled_across_wells` | `random_atoms` | all four collapse? |")
    A("|---|---|---|---|---|---|---|")
    for s in SEEDS:
        r = g["per_seed"][f"seed{s}"]
        if "DECLARED_NOT_RUN" in r:
            A(f"| {s} | **DECLARED NOT-RUN** | — | — | — | — | — |")
            continue
        cs = [f(r["controls"][c]["novel_per_corner_all"], 4) for c in CONTROLS]
        A(f"| {s} | {f(r['arm_novel_per_corner_all'], 4)} | " + " | ".join(cs)
          + f" | {f(r['ALL_FOUR_COLLAPSE'])} |")
    A("")
    A(f"**ALL FOUR COLLAPSE ON EVERY SEED: {f(g['ALL_FOUR_COLLAPSE_ON_EVERY_SEED'])}** "
      f"⇒ {g['consequence']}")
    A("")
    A("Paired estimand against each control, margin 1.0, per seed — the corner is the "
      "unit:")
    A("")
    A("| seed | control | mean paired delta | CI95 | won / tied / lost | "
      "beats the 1.0 margin? |")
    A("|---|---|---|---|---|---|")
    for s in SEEDS:
        r = g["per_seed"][f"seed{s}"]
        if "DECLARED_NOT_RUN" in r:
            continue
        for c in CONTROLS:
            v = r["controls"][c]
            ci = v.get("paired_ci95") or [None, None]
            A(f"| {s} | `{c}` | {f(v['paired_mean_delta'], 3)} | "
              f"[{f(ci[0], 2)}, {f(ci[1], 2)}] | {v.get('corners_won')} / "
              f"{v.get('corners_tied')} / {v.get('corners_lost')} | "
              f"{f(v['beats_the_1.0_margin'])} |")
    A("")

    # --- byte ledger ---------------------------------------------------------
    b = d["byte_ledger_per_seed"]
    A("## The four-part byte ledger, per seed — ⛔ NEVER NETTED")
    A("")
    A(f"**Convention:** {b['convention']}")
    A("")
    A("| seed | 1. atoms | 2. covariance + count | 3. address book | "
      "4. coupling / hole | **four-part total** | × the 524,288 B REFERENCE | n seeds |")
    A("|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        r = b["per_seed"][f"seed{s}"]
        if "DECLARED_NOT_RUN" in r:
            A(f"| {s} | **DECLARED NOT-RUN** | | | | | | 1 |")
            continue
        A(f"| {s} | {r['p1_atom_bytes']:,} | {r['p2_covariance_plus_count_bytes']:,} | "
          f"{r['p3_address_book_bytes']:,} | {r['p4_coupling_hole_table_bytes']:,} | "
          f"**{r['honest_total_bytes']:,} B** | {r['times_reference_pin']}× | 1 |")
    A("")
    A("⛔ The memory ledger is printed here and is **never matched** between arms. "
      "The 524,288 B column is a REFERENCE only; the object is 1× its own total.")
    A("")

    # --- deletion ------------------------------------------------------------
    dl = d["deletion_rows"]
    A("## The deletion rows — three scopes, one object, every cost line printed")
    A("")
    A(f"**Exactness:** {dl['exactness_definition']}")
    A("")
    A(f"**Adaptation:** {dl['adaptation_language']}")
    A("")
    A("| scope | seed | rows deleted | survivor corners | novel/corner before → after | "
      "paired after − before | CI95 | store-scoped exactness max&nbsp;\\|mode dev\\| | "
      "ranks agree | OUR seconds | OUR bytes rewritten | OUR data rows read |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in SCOPES:
        rec = dl["scopes"][sc]
        for s in SEEDS:
            v = rec["per_seed"][f"seed{s}"]
            if "DECLARED_NOT_RUN" in v:
                A(f"| {sc} | {s} | **DECLARED NOT-RUN** — {v['DECLARED_NOT_RUN']} "
                  "| | | | | | | | |")
                continue
            ci = v.get("paired_ci95") or [None, None]
            c = v["OUR_ADAPTATION_COST"]
            A(f"| `{sc}` | {s} | {v['n_rows_deleted']:,} | {v['n_survivor_corners']} | "
              f"{f(v['survivors_before'], 3)} → {f(v['survivors_after'], 3)} | "
              f"{f(v['paired_after_minus_before'], 4)} | "
              f"[{f(ci[0], 3)}, {f(ci[1], 3)}] | "
              f"{v['exactness_max_abs_mode_dev']:.3e} | "
              f"{f(v['ranks_agree_with_refit_without'])} | {c['seconds']} | "
              f"{c['bytes_rewritten']:,} | **{c['data_rows_read']}** |")
    A("")
    A("### The rival's own cost line on the same scope — refit-from-scratch-without")
    A("")
    A("| scope | seed | arm | seconds | data rows read | bytes rewritten | refit? |")
    A("|---|---|---|---|---|---|---|")
    for sc in SCOPES:
        rec = dl["scopes"][sc]
        for s in SEEDS:
            v = rec["per_seed"][f"seed{s}"]
            if "DECLARED_NOT_RUN" in v:
                continue
            o, r = v["OUR_ADAPTATION_COST"], v["REFIT_WITHOUT_COST"]
            A(f"| `{sc}` | {s} | ours (`cov_tensor`) | {o['seconds']} | "
              f"**{o['data_rows_read']}** | {o['bytes_rewritten']:,} | no |")
            A(f"| `{sc}` | {s} | refit-without (the store reference) | {r['seconds']} | "
              f"{r['data_rows_read']:,} | {r['bytes_rewritten']:,} | yes |")
    A("")

    # --- rival deletion column ----------------------------------------------
    rv = d["rival_deletion_column"]
    A("## The RIVAL deletion column — a rival deletes ONLY by retrain-without")
    A("")
    A(f"> {rv['rule']}")
    A("")
    A("Every figure on a row is measured on the **same survivor corners** as the arm, "
      "so the rival's as-trained → retrain-without step is a before/after on exactly "
      "the corners our before/after uses.")
    A("")
    A("| scope | seed | survivors | family | rival **as-trained** (survivors) | rival "
      "**retrain-without** (survivors) | rival after − before | ours after deletion | "
      "paired ours − rival | CI95 | won / tied / lost | beats 1.0? | rival retrain "
      "seconds | rival rows read |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in SCOPES:
        rec = rv["scopes"][sc]
        for s in SEEDS:
            v = rec["per_seed"][f"seed{s}"]
            if "DECLARED_NOT_RUN" in v:
                A(f"| `{sc}` | {s} | — | — | **DECLARED NOT-RUN** — "
                  f"{v['DECLARED_NOT_RUN']} | | | | | | | | | |")
                continue
            for fam in RIVALS:
                x = v["families"][fam]
                rw = x["retrain_without_novel_per_corner"]
                pm = x["paired_ours_minus_rival"]
                cost = x.get("retrain_without_cost") or {}
                ci = pm.get("ci95") or [None, None]
                rws = ("**DECLARED NOT-RUN**" if isinstance(rw, dict) else f(rw, 3))
                rp = x.get("rival_after_minus_before_on_survivors") or {}
                A(f"| `{sc}` | {s} | {x.get('n_survivor_corners')} | `{fam}` | "
                  f"{f(x['as_trained_novel_per_corner_on_the_scopes_survivors'], 3)} | "
                  f"{rws} | {f(rp.get('mean_paired_delta'), 3)} | "
                  f"{f(v['ours_after_deletion_on_the_same_survivors'], 3)} | "
                  f"{f(pm.get('mean_paired_delta'), 3)} | "
                  f"[{f(ci[0], 2)}, {f(ci[1], 2)}] | {pm.get('corners_won')} / "
                  f"{pm.get('corners_tied')} / {pm.get('corners_lost')} | "
                  f"{f(pm.get('beats_the_1.0_margin'))} | "
                  f"{f(cost.get('seconds'), 2)} | "
                  f"{f(cost.get('data_rows_read'))} |")
    A("")

    A("## What did NOT run")
    A("")
    for k, v in d["WHAT_DID_NOT_RUN"].items():
        A(f"- **`{k}`** — {v}")
    A("")

    p = OUT / "T7-3SEEDS.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", p, len(L), "lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
