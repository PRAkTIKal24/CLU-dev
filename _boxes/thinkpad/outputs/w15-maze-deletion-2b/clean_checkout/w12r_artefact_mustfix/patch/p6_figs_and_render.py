"""w12r patch 6 -- figures follow the relabel and the corrected field; TABLES.md renders every
block w12r added."""
import sys, pathlib
P = pathlib.Path(sys.argv[1])
s = P.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), repr(old[:90]))
    s = s.replace(old, new, n)


# ---------------- figures follow MF10 and MF8 -----------------------------------------
rep('    fig.suptitle("CMAZE-1 main scoreboard - 3 scored seeds, held-out TEST", fontsize=11,',
    '    fig.suptitle("CMAZE-2 main scoreboard - 3 scored seeds, held-out TEST", fontsize=11,')
rep('    """The caveat column that must travel with every CMAZE-1 composition row."""',
    '    """The caveat column that must travel with every CMAZE-2 composition row."""')
rep('    ax.set_title("Every CMAZE-1 composition row carries BLIND-SEQUENTIAL measured on ITS OWN split")',
    '    ax.set_title("Every CMAZE-2 composition row carries BLIND-SEQUENTIAL measured on ITS OWN split")')

rep('''    dd_abs = [max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS]''',
    '''    # w12r/MF8: the field the SOURCE SPOKE's own FINAL REPORT prints - worst over the pinned
    # 11,024-row deletion AND the random-64 deletion - not the narrower per-well-by-seed field.
    dd_abs = [ptr(dd, f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev") for s in SEEDS]
    dd_floor = [ptr(dd, f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev") for s in SEEDS]''')
rep('''           label="w8/downdate  max ABSOLUTE deviation, worst well")''',
    '''           label="w8/downdate  max ABSOLUTE deviation (pinned + random-64)")
    ax.scatter(xs + 0.18, dd_floor, marker="_", s=260, color=INK1, zorder=6, linewidths=1.8,
               label="w8/downdate  its own reproducibility floor")''')
rep('''    ax.set_title("Exact deletion - two independent implementations, 3 seeds each, TEST")''',
    '''    ax.set_title("Exact deletion (STORE-SCOPED) - two implementations, 3 seeds each, TEST")''')
rep('''    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.text(0.5, -0.04,
             "Different units: w9/maze reports a relative residual on the sufficient statistic "
             "(projector residual 0.0 on every seed);\\nw8/downdate reports an absolute max deviation per well "
             "(worst = 0.0094 float32 ulp at unit scale). Both: 0 refits, 0 passes over the data, items never kept.",
             ha="center", fontsize=7.0, color=INK2)''',
    '''    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.06))
    fig.text(0.5, -0.06,
             "EXACT = STORE-SCOPED: the store content equals the content fitted from the retained rows alone, "
             "holding the FROZEN encoder and decoder fixed. Neither is refit.\\n"
             "Different units: w9/maze reports a relative residual on the sufficient statistic "
             "(projector residual 0.0 on every seed); w8/downdate reports an absolute max deviation.\\n"
             "On seeds 1 and 2 the w8 deviation sits BELOW its own reproducibility floor (the dash), so there the "
             "residual is not resolvable at all. Both: 0 refits, 0 passes over the data, items never kept.",
             ha="center", fontsize=6.8, color=INK2)''')

# ---------------- renderers for the new blocks -----------------------------------------
RENDER_HELPERS = '''def _ledger_md(led):
    """Render a four-part byte ledger block: every part on its own line, never netted."""
    out = []
    if "status" in led:
        out.append(f"> **BYTE LEDGER: {led['status']}**")
        out.append("")
        out.append(f"> {led['why']}")
        return out
    out.append(f"**Byte ledger — {led['rule']}**")
    out.append("")
    rows = []
    for k, v in led.get("parts", {}).items():
        if isinstance(v, dict) and "status" in v:
            rows.append([k, "**" + v["status"] + "**", v.get("why", "")])
        else:
            rows.append([k, fmt_any(v), v.get("formula") or v.get("what") or
                         ("stored field" if "provenance" in v else "")])
    for k, v in led.get("totals", {}).items():
        rows.append(["**" + k + "**", "**" + fmt_any(v) + "**",
                     v.get("formula") or ("stored field" if "provenance" in v else "")])
    out.append(md_table(["part (seed 0 / seed 1 / seed 2)", "bytes", "how it is obtained"], rows))
    out.append("")
    for key in ("pin_status", "THE HEADLINE THE DRAFT OMITTED", "no_row_here_is_under_the_524288_B_pin"):
        if key in led:
            out.append(f"> ⛔ **{key}:** {led[key]}")
            out.append("")
    if "the_sentence_the_artefact_itself_writes" in led:
        out.append("> The artefact's own words: *\\"" +
                   led["the_sentence_the_artefact_itself_writes"]["text"] + "\\"*")
        out.append("")
    return out


def _kv_md(d, skip=()):
    rows = []
    for k, v in d.items():
        if k in skip or k.startswith("_"):
            continue
        if isinstance(v, dict) and ("values" in v or "derived" in v or "printed" in v):
            rows.append([k, fmt_any(v)])
        elif isinstance(v, (list, tuple)):
            rows.append([k, "; ".join(str(x) for x in v)])
        elif isinstance(v, dict):
            continue
        else:
            rows.append([k, str(v)])
    return md_table(["field", "value"], rows)


'''
rep("def render_md(manifest):", RENDER_HELPERS + "def render_md(manifest):")

# ---- definitions block, printed before TABLE 1 ----------------------------------------
rep('''    A("---")
    A("")

    # ---- T1 -------------------------------------------------------------------------''',
    '''    A("---")
    A("")

    # ---- w12r: the three definitions the prose must cite ------------------------------
    ed = manifest["exactness_definition"]
    A("## ⭐ DEFINITION — what `exact` means in this paper (`TABLES.json` `/exactness_definition`)")
    A("")
    A(f"> **{ed['scope']}.** {ed['definition']}")
    A("")
    A("**What it is NOT:**")
    for x in ed["what_it_is_not"]:
        A(f"- ⛔ {x}")
    A("")
    A(f"**Why only this scope is supported.** {ed['why_only_this_scope_is_supported']}")
    A("")
    A(md_table(["evidence", "path", "pointer / locator"],
               [[e["what"], f"`{e['path']}`", f"`{e.get('pointer') or e.get('locator')}`"]
                for e in ed["evidence"]]))
    A("")
    A("⛔ **Every cell whose label carries the word `exact` is scoped by this definition.** In this "
      "file that is: T1's RESULT rows and every ceiling row (`validity under the exact judge` — a "
      "different sense: the JUDGE is exact, see 1e); T1 §1d's post-deletion rows; all of TABLE 4; "
      "all of TABLE 5. The two senses are kept apart on purpose: **an exact JUDGE** is integer "
      "arithmetic with no tolerance (`/judge_audit/exact` = True), while **exact DELETION** is the "
      "store-scoped definition above.")
    A("")
    env = manifest["environments"]
    A("## ⭐ DEFINITION — CMAZE-1 and CMAZE-2 are two environments, not one")
    A("")
    A(md_table(["environment", "what it is", "tables"],
               [["**CMAZE-1**", env["CMAZE-1"]["what"], ", ".join(env["CMAZE-1"]["tables_that_use_it"])],
                ["**CMAZE-2**", env["CMAZE-2"]["what"], ", ".join(env["CMAZE-2"]["tables_that_use_it"])]]))
    A("")
    A(f"> ⛔⛔ **{env['RULE']}**")
    A("")
    A("**The three numbers only read coherently once they are separated:**")
    A("")
    A(md_table(["BLIND-SEQUENTIAL", "value", "where"],
               [["published CMAZE-1", "**0.4980**", "`/c/chlu/outputs/w8-env-scout/ENV.md` lines 42, 49, 358"],
                ["CMAZE-1 at the w9/harness tier", "**0.5317**  (0.5550 / 0.5300 / 0.5100)",
                 "`/c/chlu/outputs/w9-harness/harness_seed0.json` `/columns/blind-sequential (key order from A)/validity` — TABLE 3"],
                ["CMAZE-2", "**0.9037**  (0.9077 / 0.8905 / 0.9128)",
                 "`/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity` — TABLE 1"]]))
    A("")
    sc = manifest["seed_and_configuration_counts"]
    A("## ⛔ NEVER `9/9 SEEDS` — seed count and configuration count are separate fields")
    A("")
    A(f"> {sc['why_this_field_exists']}")
    A("")
    A(md_table(["implementation", "seeds", "configurations", "legs", "scored runs", "configuration"],
               [[k, v["seeds"], v["configurations"], v["legs"], v["scored_runs"], v["configuration"]]
                for k, v in sc["implementations"].items()]))
    A("")
    A("**Correct phrasings:** " + " · ".join(sc["correct_phrasings"]))
    A("")
    A("---")
    A("")

    # ---- T1 -------------------------------------------------------------------------''')

# ---- T1: ledger + fragment + declared gap, printed after 1c ---------------------------
rep('''    A("### 1d. DISPLAY rows — inductive bias; the only comparator is the supervised refuser")''',
    '''    A("### 1c-bis. ⛔ FRAGMENT-LOOKUP — same-maze and cross-maze are DIFFERENT QUANTITIES")
    A("")
    fr = t1["fragment_columns"]
    A(f"> {fr['why_this_block_exists']}")
    A("")
    A(md_table(["column", "r=1", "r=2"],
               [[k, fmt_cell(v["r=1"]), fmt_cell(v["r=2"])] for k, v in fr["rows"].items()]))
    A("")
    A(f"> ⛔ **Not a like-for-like arm.** {fr['not_like_for_like']}")
    A("")
    A(f"> **Why the same-maze number is high.** {fr['why_same-maze_is_high']}")
    A("")
    A(f"> ⭐ **{fr['the_number_the_draft_left_on_the_floor']}**")
    A("")
    A("### 1c-ter. ⛔ DECLARED NOT MEASURED — the charter rule that is unsatisfiable for row 4")
    A("")
    A(md_table(["row", "column", "status", "why"],
               [[d["row"], d["column"], "**" + d["status"] + "**", d["why"]]
                for d in t1["declared_not_measured"]]))
    A("")
    A("### 1c-quater. ⛔ THE BYTE LEDGER — four parts, printed, never netted")
    A("")
    for line in _ledger_md(t1["byte_ledger"]):
        A(line)
    A("")
    A("### 1d. DISPLAY rows — inductive bias; the only comparator is the supervised refuser")''')

# ---- T2 ledger -------------------------------------------------------------------------
rep('''    A("> **The reading.** The net is a wash because two real effects cancel:''',
    '''    for line in _ledger_md(t2["byte_ledger"]):
        A(line)
    A("")
    A("> **The reading.** The net is a wash because two real effects cancel:''')

# ---- T3 ledger status --------------------------------------------------------------------
rep('''    A("**DISPLAY comparator at this tier**")''',
    '''    for line in _ledger_md(t3["byte_ledger"]):
        A(line)
    A("")
    A("**DISPLAY comparator at this tier**")''')

# ---- T4: cells render (mixed cell / derived), caveats, ledger ------------------------------
rep('''        A(md_table(["property", "seed 0 / seed 1 / seed 2"],
                   [[k, " / ".join(v["format"].format(x) for x in v["values"])] for k, v in r["cells"].items()]))
        if "summary_from_artefact" in r:''',
    '''        A(md_table(["property", "seed 0 / seed 1 / seed 2"],
                   [[k, fmt_any(v)] for k, v in r["cells"].items()]))
        if "row_class_w12r_note" in r:
            A("")
            A(f"> ⛔ **Row class.** {r['row_class_w12r_note']}")
        if "caveats_in_the_gate_file_itself" in r:
            cav = r["caveats_in_the_gate_file_itself"]
            A("")
            A(f"> ⛔ **Caveats that live inside the cited gate file.** {cav['w12r_note']}")
            A("")
            a = cav["(a) admission cut is MARGINAL on every seed"]
            A(md_table(["caveat", "value", "what it means"],
                       [["`cut_is_marginal` per seed", fmt_any(a), a["what_it_means"]]]))
            A("")
            b = cav["(b) the SUBSPACE-level assertion a1_subspace FAILS on seed 2"]
            A(md_table(["field", "value"],
                       [["`a1_subspace.passed_every_seed`", f"**{b['passed_every_seed']}**"],
                        ["`a1_subspace.seeds_failing`", ", ".join(b["seeds_failing"])],
                        ["`a1_subspace.worst_in_float32_ulp`", f"**{b['worst_in_float32_ulp']:.5f}**"],
                        ["`a1_subspace.worst_floor_subspace_in_float32_ulp`",
                         f"**{b['worst_floor_subspace_in_float32_ulp']:.5f}**"]]))
            A("")
            A(f"> {b['reading']}")
            A("")
            A(f"> ⛔ {b['watch_out']}")
        if "summary_from_artefact" in r:''')

rep('''    A("See figure 4.")
    A("")''',
    '''    A("### TABLE 4 — THE BYTE LEDGER, four parts, printed, never netted (w12r/MF3)")
    A("")
    led = t4["byte_ledger"]
    A("**w9/maze leg (CMAZE-2)**")
    A("")
    w9 = led["w9/maze leg (CMAZE-2)"]
    A(md_table(["part (seed 0 / seed 1 / seed 2)", "bytes", "how it is obtained"],
               [[k, ("**" + v["status"] + "**" if isinstance(v, dict) and "status" in v else fmt_any(v)),
                 (v.get("why", "") if isinstance(v, dict) and "status" in v
                  else (v.get("formula") or "stored field"))]
                for k, v in w9.items() if isinstance(v, dict)]))
    A("")
    A(f"> {w9['pin_status']}")
    A("")
    A("**w8/downdate leg (Shapes3D, the 524,288 B pin at tolerance 0)**")
    A("")
    w8 = led["w8/downdate leg (Shapes3D, the 524,288 B pin at tolerance 0)"]
    A(md_table(["part (seed 0 / seed 1 / seed 2)", "bytes", "how it is obtained"],
               [[k, fmt_any(v), (v.get("formula") or v.get("what") or "stored field")]
                for k, v in w8.items() if isinstance(v, dict) and
                ("values" in v or "derived" in v or "printed" in v)]))
    A("")
    A(f"> ⛔⛔ **{w8['THE HEADLINE THE DRAFT OMITTED']}**")
    A("")
    A("> **The residual, declared rather than netted away.** The source spoke's own four-part table "
      "(`/c/chlu/outputs/w8-downdate/LOG.md` lines 414-422 and 1036-1044) names store, covariance, "
      "count table and coupling+holes. Those five numbers leave **exactly 16,424 B** unaccounted on "
      "every one of the three TEST seeds. w12r identifies that residual as the running MEANS "
      "(`sum_k d_k x 8 = 16,384 B`, and `sum_k d_k = 2,048 = D` on every seed) plus the per-well "
      "COUNTS (`5 x 8 = 40 B`); with it the ledger closes to the byte against `honest_total_bytes`. "
      "⚠ The identification is w12r's inference from `d_k`; the residual itself is arithmetic. "
      "Separately, `items_kept_detail/object_bytes_full` exceeds `honest_total_bytes` by **exactly "
      "16,384 B on every seed** — w12r could not attribute that to a named field and reports it as a "
      "DECLARED RESIDUAL rather than netting it away.")
    A("")
    A("> ⭐ **Exact is not free.** On TEST the same deletion takes surviving-target composition from "
      + fmt_cell(t4["exact_is_not_free"]["surviving-target composition on TEST, BEFORE"]) + " to "
      + fmt_cell(t4["exact_is_not_free"]["surviving-target composition on TEST, AFTER"])
      + ". The claim is exactness, never cost-free erasure.")
    A("")
    A("See figure 4.")
    A("")

    # ---- T5 ---------------------------------------------------------------------------
    t5 = manifest["tables"]["T5"]
    A("---")
    A("")
    A(f"## TABLE 5 — {t5['title']}")
    A("")
    A(f"*{t5['caption']}*")
    A("")
    A(md_table(["implementation", "class", "seeds", "configs", "legs", "scored runs", "configuration", "band"],
               [[r["implementation"], "**" + r["row_class"] + "**", r["seeds"], r["configurations"],
                 r["legs"], r["scored_runs"], r["configuration"], "**" + r["band"] + "**"]
                for r in t5["rows"]]))
    A("")
    for r in t5["rows"]:
        A(f"**{r['implementation']}**  ·  metric: {r['metric']}")
        A("")
        cc = r.get("cells") or r.get("cells_markdown_only")
        A(md_table(["quantity", "value (seed 0 / seed 1 / seed 2)", "traced to"],
                   [[k, fmt_any(v),
                     (f"`{v['path']}` {v['lines']}" if "printed" in v
                      else f"`{v['provenance'][0]['path']}` `{v['provenance'][0]['pointer']}`")]
                    for k, v in cc.items()]))
        A("")
        A(f"> {r['note']}")
        A("")
    fa = t5["failure_agreement"]
    A(f"### ⭐⭐ {fa['title']}")
    A("")
    A(f"> {fa['what_happened']}")
    A("")
    A(md_table(["leg", "float32 ulp at unit scale", "traced to"],
               [[k, fmt_any(v),
                 (f"`{v['path']}` {v['lines']}" if "printed" in v
                  else f"`{v['provenance'][0]['path']}` `{v['provenance'][0]['pointer']}`")]
                for k, v in fa["legs"].items()]))
    A("")
    A(f"> ⭐⭐ **The citable sentence.** {fa['the_citable_sentence']}")
    A("")
    A(f"> **How it was closed.** {fa['how_it_was_closed']['fix']}, "
      f"`tau` = {fa['how_it_was_closed']['tau']:.6e} "
      f"({fa['how_it_was_closed']['tau_derivation']}).")
    A("")
    A(f"> ⛔⛔ **What must never be printed.** {fa['what_must_never_be_printed']}")
    A("")''')

P.write_text(s, encoding="utf-8")
print("patch 6 applied")
