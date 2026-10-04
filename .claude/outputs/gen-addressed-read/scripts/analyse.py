"""⭐⭐ The report tables, computed from the manifests. ⛔ No number is typed by hand.

⛔⛔ The MATCHED-BYTES row leads. The 8x row is printed only with its byte multiple on
the same line. That is enforced here, not remembered.
"""
import json
from pathlib import Path

import numpy as np

R = Path("/Users/user/Desktop/CHLU/.claude/outputs/gen-addressed-read/runs")
agg = json.loads((R / "cell" / "cell_results.json").read_text())
mans = [json.loads((R / "cell" / f"seed_{s}" / "cell_manifest.json").read_text())
        for s in (0, 1, 2)]
V = agg["x0_verdict"]
BUDGET = 524_288


def per_seed(arm):
    return [m["arms"][arm]["external"]["composition_B"]["aggregate"]
            for m in mans if arm in m.get("arms", {})]


def row(label, arm, note=""):
    v = per_seed(arm)
    if not v:
        return f"| {label} | ⛔ NOT RUN | — | — | — | {note} |"
    a = agg["arms"][arm]
    led = a["byte_ledger"]
    b = led.get("deployed_bytes")
    head = led.get("addressing_head_bytes")
    bytestr = "not byte-matched" if not b else (
        f"{b:,}" + (f" + {head:,} head" if head else "")
        + f" = {b / BUDGET:.2f}×")
    return (f"| {label} | **{np.mean(v):.4f}** ± {np.std(v):.4f} | "
            f"{' / '.join(f'{x:.4f}'[1:] for x in v)} | "
            f"{a['distinct_factor_tuples_B']['mean']:.2f} ± "
            f"{a['distinct_factor_tuples_B']['std']:.2f} | {bytestr} | {note} |")


print("\n## ⭐⭐ THE GENERATION TABLE — composition-B aggregate, external classifier, "
      "3 seeds\n")
print("| arm | composition-B | per-seed | distinct tuples / 72 | deployed bytes "
      "(× the 524,288 B budget) | role |")
print("|---|---|---|---|---|---|")
print(row("`store_removed` (**the null**)", "store_removed", "laundering control"))
print(row("`ebm_compose`", "ebm_compose", "like-for-like"))
print(row("`lace`", "lace", "like-for-like (cell harness)"))
print(row("⭐ `clu` (**the codebook read**)", "clu",
          "⭐⭐ **the matched-bytes rival**"))
print(row("⭐⭐ **`x0` (THE ADDRESSED READ, MATCHED BYTES)**", "x0",
          "⭐⭐ **THE HEADLINE ROW**"))
print(row("`x0/A_atom_mean_lookup`", "x0/A_atom_mean_lookup",
          "⭐ §C.1 floor (the Advisor's named rival)"))
print(row("`x0/A_row_lookup`", "x0/A_row_lookup", "⭐ §C.1 floor (whole stored row)"))
print(row("`x0/A_atoms_shuffled`", "x0/A_atoms_shuffled", "⭐ §C.2 control"))
print(row("`dynread` (the SETTLED read, same atoms)", "dynread",
          "⛔ another spoke's row, shown for contrast"))
print(row("`knn_proto` (**ceiling**)", "knn_proto", "⛔ unbounded real-valued channel"))

print("\n### ⭐ The 8× companion (⛔ never quoted without the multiple)\n")
print("| row | composition-B | per-seed | distinct tuples / 72 | deployed bytes |")
print("|---|---|---|---|---|")
for lbl, arm in (("`x0/B`", "x0/B"),
                 ("`x0/B_atom_mean_lookup`", "x0/B_atom_mean_lookup"),
                 ("`x0/B_row_lookup`", "x0/B_row_lookup"),
                 ("`x0/B_atoms_shuffled`", "x0/B_atoms_shuffled"),
                 ("`dynread/B` (settled)", "dynread/B")):
    print(row(lbl, arm, "")[:-2].replace("| ⭐⭐ ", "| "))

print("\n### ⭐⭐ THE VERDICT\n")
print(V["headline"])
for budget in ("A", "B"):
    r = V["per_rung"].get(budget, {})
    if not r.get("ours"):
        continue
    mult = (r.get("bytes") or {}).get("multiple_of_budget", float("nan"))
    print(f"\n**Rung {budget} — `{r['label']}`, {mult:.2f}× the budget**")
    print("| rival | role | Δ | paired per-seed | 3/3 same sign | intervals overlap "
          "| verdict |")
    print("|---|---|---|---|---|---|---|")
    items = [(k, v, v.get("role", "")) for k, v in r["against"].items()]
    items += [(k, v, "⭐ §C.1 matched-bytes LOOKUP FLOOR")
              for k, v in r["charter_C1_floors"].items()]
    items += [(r["charter_C2_control"]["label"], r["charter_C2_control"],
               "⭐ §C.2 landscape-removed control")]
    for k, v, role in items:
        if v.get("verdict") == "NOT RUN":
            print(f"| `{k}` | {role} | — | — | — | — | ⛔ NOT RUN |")
            continue
        ps = " / ".join(f"{x:+.4f}" for x in v["paired_per_seed"].values())
        print(f"| `{k}` | {role.split('(')[0].strip()} | **{v['delta']:+.4f}** | {ps} | "
              f"{'yes' if v['same_sign_3_of_3'] else '**no**'} | "
              f"{'yes' if v['intervals_overlap_mean_pm_sd'] else 'no'} | "
              f"**{v['verdict']}** |")
    print(f"\n- ceiling verdict: **{r['verdict_vs_ceiling']}** · like-for-like all "
          f"beaten: **{'YES' if r['like_for_like_all_beaten'] else 'NO'}** · "
          f"⭐ §C.1 floor: **{'PASS' if r['charter_C1_floor_PASS'] else '⛔ FAIL'}** · "
          f"§C.2 control collapses: "
          f"**{'YES' if r['charter_C2_control_COLLAPSES'] else '⛔ NO'}** · "
          f"**PASS: {'YES' if r['PASS'] else 'NO'}**")
    print(f"- {r['charter_C1_floor_statement']}")

print("\n### ⭐⭐ THE BYTE LEDGER — MEASURED\n")
led = mans[0]["arms"]["x0"]["byte_ledger"]
clu = mans[0]["arms"]["clu"]["byte_ledger"]
print("| | `x0` rung A (measured) | `clu` (published) | `dynread` rung A |")
print("|---|---|---|---|")
dled = mans[0]["arms"]["dynread"]["byte_ledger"]
print(f"| store / atoms | **{led['store_bytes']:,}** | "
      f"**{clu['deployed_bytes']:,}** | {dled['store_bytes']:,} |")
print(f"| addressing head | **{led['addressing_head_bytes']:,}** | "
      f"**{clu['addressing_head_bytes']:,}** | 0 |")
print(f"| coupling, DEPLOYED | **{led['coupling_bytes']:,}** | 0 | "
      f"{dled['coupling_bytes']:,} |")
print(f"| × the 524,288 B budget | **{led['multiple_of_budget']:.2f}×** | 1.00× | "
      f"{dled['multiple_of_budget']:.2f}× |")
print(f"\n- byte-matched to `clu` field for field: "
      f"**{led['byte_matched_to_clu_field_for_field']}**")
p = led["coupling_not_needed_proof"]
print(f"- coupling needed: **{p['coupling_needed']}** — bit-identical `x0` across "
      f"{', '.join('`' + k + '`' for k in p['bit_identical'])} "
      f"(max |Δ| = {max(p['max_abs_delta'].values()):.1f} on all four)")
print(f"- coupling bytes that LEAVE the ledger: **{led['coupling_bytes_excluded']:,} B**")

print("\n### zero-step proof + §C.4\n")
for s, m in zip((0, 1, 2), mans):
    z = m["arms"]["x0"]["addressed_read"]["request_sets"]["B"]["zero_step_proof"]
    g = m["arms"]["x0"]["addressed_read"]["request_sets"]["B"]
    print(f"- seed {s}: steps={z['settling_steps']} · "
          f"displacement from launch = {z['displacement_from_launch_rms']} · "
          f"decoder input IS the launch bitwise = "
          f"{z['decoder_input_is_the_launch_bitwise']} · energy trace = "
          f"{z['energy_trace']} · returns a stored atom = "
          f"{g.get('returns_a_stored_atom')} · distinct index tuples = "
          f"{g.get('n_distinct_index_tuples')}/72 · mean confidence = "
          f"{g.get('mean_confidence'):.4f}")
rl = [m["arms"]["x0/A_row_lookup"]["addressed_read"]["request_sets"]["B"]
      for m in mans]
print(f"- `row_lookup` returns {np.mean([r['n_distinct_rows_returned'] for r in rl]):.2f}"
      f" distinct whole train rows of 72 requests")
print("\n### instrument-ordering check\n")
ri = agg["instrument_invariance"]["_ranking"]
print(f"- by external: {ri['by_external'][:4]} …")
print(f"- by probe:    {ri['by_probe'][:4]} …")
print(f"- identical ordering: **{ri['identical']}**")
