#!/usr/bin/env python
"""w13 patch 4/4 -- figures 6/7/8, the renderer for T7..T12, the new DECLARED NOT-RUNs
(NR6..NR10), the new CORRECTIONS (C15..C19) and the manifest blocks.
Exact-string patch; every anchor must occur EXACTLY once."""
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "build_tables.py"
t = SRC.read_text(encoding="utf-8")


def sub(anchor, replacement):
    global t
    n = t.count(anchor)
    assert n == 1, "anchor occurs %d times, expected 1: %r" % (n, anchor[:110])
    t = t.replace(anchor, replacement)


# ---------------------------------------------------------------- 0. arithmetic fix
sub("[-21.0816, -16.5307, -18.0816],", "[-21.0816, -16.5306, -18.0816],")

# ---------------------------------------------------------------- 1. figures 6, 7, 8
FIGS = r'''
# ---- w13 figures ----------------------------------------------------------------------
def fig6_composition_ladder():
    """The n-way ladder.  x = n (the RUNGS), NOT K.  Every point is 3 seeds and the seed
    count is printed on the panel.  Panel B carries BOTH byte matchings, because they tell
    opposite stories about the n-trend."""
    L = DOCS["lad_json"]
    ns = [2, 3, 4, 5, 6]
    series = [("single_design_K5", "single design (arm), K=5", CAT[0]),
              ("learned_1x", "learned_1x (matched to the arm's OWN bytes)", CAT[1]),
              ("rival_mlp", "rival_mlp @524,288 B", CAT[2]),
              ("rival_lstm", "rival_lstm @524,288 B", CAT[3]),
              ("rival_table", "rival_table (floor)", CAT[4])]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    for arm, lab, col in series:
        per = [[ptr(L, "/rungs/%d/arms/%s/per_seed/%d/mean" % (n, arm, s)) for s in range(3)] for n in ns]
        ax.plot(ns, [sum(p) / 3 for p in per], marker="o", color=col, label=lab, zorder=4)
        ax.fill_between(ns, [min(p) for p in per], [max(p) for p in per], color=col, alpha=0.14, zorder=2)
    ax.set_xticks(ns)
    ax.set_xlabel("n  =  number of jointly composed factors  (the RUNG)")
    ax.set_ylabel("distinct valid novel tuples / corner")
    ax.set_title("A.  score vs n   -   K = 5 wells on every point, n_seeds = 3", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    recessive(ax)
    ax.annotate("ceiling 36\n(60 corners)", (2, 36), xytext=(0, -2), textcoords="offset points",
                ha="center", va="top", fontsize=7.0, color=INK3)
    ax.annotate("ceiling 72  (120 corners, and for n>=3 the SAME 120)", (4, 72.6),
                ha="center", fontsize=7.0, color=INK3)

    ax = axes[1]
    for key, lab, col in [("paired_vs_best_learned_5", "vs best @524,288 B rival  -  margin GROWS with n", CAT[0]),
                          ("paired_vs_best_learned", "vs learned_1x (the arm's OWN bytes)  -  NEGATIVE", CAT[7])]:
        per = [[ptr(L, "/rungs/%d/arms/single_design_K5/per_seed/%d/%s/mean_paired_delta" % (n, s, key))
                for s in range(3)] for n in ns]
        ax.plot(ns, [sum(p) / 3 for p in per], marker="o", color=col, label=lab, zorder=4)
        ax.fill_between(ns, [min(p) for p in per], [max(p) for p in per], color=col, alpha=0.16, zorder=2)
        for n, p in zip(ns, per):
            ax.plot([n] * 3, p, marker="o", markersize=3.0, linestyle="none", markerfacecolor=SURFACE,
                    markeredgecolor=INK, markeredgewidth=0.8, zorder=6)
    ax.axhline(0.0, color=INK, linewidth=1.0, zorder=3)
    ax.axhline(1.0, color=INK3, linewidth=1.2, linestyle=(0, (4, 2)), zorder=3)
    ax.annotate("the +1.0 bar, declared in advance", (6, 1.0), xytext=(-4, 4), textcoords="offset points",
                ha="right", fontsize=7.0, color=INK2)
    ax.set_xticks(ns)
    ax.set_xlabel("n  =  number of jointly composed factors  (the RUNG)")
    ax.set_ylabel("paired margin, the corner as the unit")
    ax.set_title("B.  THE SIGN FLIP  -  the same rows under the two byte matchings", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.6, loc="center left")
    recessive(ax)

    fig.suptitle("APPENDIX  -  the n-way ladder at K = 5 wells, 3 seeds on every point",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.text(0.5, -0.13,
             "n is the RUNG (jointly composed factors); K is the WELL COUNT and is 5 on every point of "
             "both panels.  Bands are the seed min-max over 3 seeds.\n"
             "PANEL B IS THE CAPTION: against the 524,288 B rivals the margin grows +7.08..+9.65 (n=2) to "
             "+14.99..+22.43 (n=6); against a learner matched to the arm's own\n24.4x total it is NEGATIVE "
             "at n = 2, 4, 5 and 6.  A reader shown one line alone draws the opposite conclusion.  "
             "learned_1x is ONE family (MLP); the other three 1x rivals are NR8.",
             ha="center", fontsize=7.0, color=INK2)
    fig.tight_layout()
    p = FIG / "fig6_composition_ladder_n.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig7_k_curve():
    """The K-curve.  x = K (the WELLS), NOT n.  Two stacked panels rather than two y-axes,
    so neither curve borrows the other's scale.  n is printed at every point."""
    C, Lg = DOCS["x_cross_val"], DOCS["x_ledger_K"]
    Ks = [5, 8, 10, 12, 16, 20, 24, 28, 32]
    comp = [ptr(C, "/per_K/%d/val_composition_mean" % k) for k in Ks]
    nseed = [ptr(C, "/per_K/%d/n_seeds" % k) for k in Ks]
    rw = [ptr(Lg, "/rows/%d/rank_over_width_max" % i) for i in range(9)]
    t124 = [ptr(Lg, "/rows/%d/times_reference_pin_parts_1_2_4" % i) for i in range(9)]
    t4 = [ptr(C, "/per_K/%d/ledger_four_part/times_reference_pin" % k) for k in Ks]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.15]})
    ax = axes[0]
    ax.plot(Ks, comp, marker="o", color=CAT[0], zorder=4, label="val composition (validity)")
    for k, c, nn in zip(Ks, comp, nseed):
        ax.annotate("n=%d" % nn, (k, c), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=6.8, color=(INK if nn == 3 else "#b04a20"))
    ax.set_ylim(0.85, 0.96)
    ax.set_ylabel("val composition")
    ax.set_title("A.  composition vs K   -   n_seeds IS PRINTED AT EVERY POINT (3 at the two ends, 1 in between)",
                 fontsize=9.3)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    recessive(ax)

    ax = axes[1]
    ax.plot(Ks, t124, marker="s", color=CAT[1], zorder=4,
            label="parts 1+2+4 ONLY (no address book)  -  FALLS 7.4815x -> 2.0663x")
    ax.plot(Ks, t4, marker="o", color=CAT[0], zorder=4,
            label="the HONEST FOUR-PART total  -  U-SHAPED, min 3.9942x at K = 24")
    ax.axhline(1.0, color=INK3, linewidth=1.2, linestyle=(0, (4, 2)), zorder=3)
    ax.annotate("the retired 524,288 B pin  -  the object never reaches it at rank 64",
                (32, 1.0), xytext=(-4, 6), textcoords="offset points", ha="right",
                fontsize=7.0, color=INK2)
    ax2 = ax.twinx()
    ax2.plot(Ks, rw, marker="^", color=INK3, linewidth=1.2, zorder=3, label="rank / width (max)")
    ax2.set_ylabel("rank / width (max)", color=INK3)
    ax2.tick_params(axis="y", colors=INK3)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, 1.05)
    ax.set_xticks(Ks)
    ax.set_xlabel("K  =  number of WELLS  (the partition).  n is not varied anywhere in this figure.")
    ax.set_ylabel("x the 524,288 B reference")
    ax.set_title("B.  the byte ledger vs K   -   TWO TOTALS, AND THEY MOVE IN DIFFERENT DIRECTIONS", fontsize=9.3)
    ax.legend(frameon=False, fontsize=7.4, loc="upper center")
    recessive(ax)

    fig.suptitle("The K-curve  -  K is WELLS, not the ladder's n", fontsize=11, fontweight="bold", y=0.99)
    fig.text(0.5, -0.06,
             "Panel B is the caption the draft needs: quoting '7.48x -> 2.07x' quotes the ORANGE curve, "
             "which has NO address book in it.  With the measured address book\nthe total is U-shaped and "
             "rises again after K = 24, because the book grows 5.4x (195,584 -> 1,059,824 B) while the "
             "covariance falls 6.0x (3,391,552 -> 560,608 B).\nSeven of the nine points carry ONE seed; "
             "the two ends carry three.  The line between them is drawn only because n is printed on it.",
             ha="center", fontsize=7.0, color=INK2)
    fig.tight_layout()
    p = FIG / "fig7_k_curve_wells.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig8_four_part_ledger():
    """The four-part ledger as four stacked parts across K -- never netted, never a single
    number.  This is the row the referee said Table 1 did not print."""
    C = DOCS["x_cross_val"]
    Ks = [5, 8, 10, 12, 16, 20, 24, 28, 32]
    parts = [("p1_atom_bytes", "1. atoms", CAT[0]),
             ("p2_covariance_plus_count_bytes", "2. covariance + count ledger", CAT[1]),
             ("p3_address_book_bytes", "3. address book", CAT[2]),
             ("p4_coupling_hole_table_bytes", "4. coupling / hole table", CAT[3])]
    vals = {k: [ptr(C, "/per_K/%d/ledger_four_part/%s" % (K, k)) for K in Ks] for k, _l, _c in parts}
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = np.arange(len(Ks))
    bottom = np.zeros(len(Ks))
    for key, lab, col in parts:
        v = np.array(vals[key], dtype=float) / 524288.0
        ax.bar(x, v, bottom=bottom, color=col, label=lab, zorder=3, edgecolor=SURFACE, linewidth=0.8)
        bottom = bottom + v
    ax.plot(x, bottom, marker="o", color=INK, linewidth=1.4, zorder=5, label="honest total (the SUM)")
    for xi, b in zip(x, bottom):
        ax.annotate("%.3fx" % b, (xi, b), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=6.8, color=INK)
    ax.axhline(1.0, color=INK3, linewidth=1.2, linestyle=(0, (4, 2)), zorder=4)
    ax.annotate("524,288 B  =  1x  (the RETIRED pin, a reference column this wave)", (len(Ks) - 1, 1.0),
                xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=7.0, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(["K=%d%s" % (K, "\n(K*)" if K == 32 else "") for K in Ks])
    ax.set_ylabel("x the 524,288 B reference")
    ax.set_title("The FOUR-PART LEDGER, printed and never netted  -  across K = 5..32 WELLS", fontsize=10)
    ax.legend(frameon=False, fontsize=7.6, ncol=3, loc="upper center")
    recessive(ax)
    fig.text(0.5, -0.10,
             "The lead object is 3.99x-7.85x the 524,288 B reference and the referee rejected the draft "
             "partly because Table 1 printed no byte row at all.\nPart 2 falls 6.0x as K rises and part 3 "
             "grows 5.4x; the two nearly cancel after K = 24.  Part 4 (6,610 B) is visible only as the "
             "hairline it is.\nThe ladder's own legs (T8) carry parts 1-3 only: part 4 is a DECLARED "
             "ABSENCE there, never a zero.",
             ha="center", fontsize=7.0, color=INK2)
    fig.tight_layout()
    p = FIG / "fig8_four_part_ledger_across_K.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


'''
sub("FIG_SPECS = [\n", FIGS + "FIG_SPECS = [\n")

sub('''     ["/env_columns/{row}/blind/BLIND-SEQUENTIAL(key order from A)/validity"]),
]''',
    '''     ["/env_columns/{row}/blind/BLIND-SEQUENTIAL(key order from A)/validity"]),
    ("fig6", fig6_composition_ladder,
     "T8: the n-way ladder (n = 2..6) at K = 5 wells, 3 seeds; panel B carries BOTH byte matchings "
     "and is where the sign flips",
     ["lad_json"],
     ["/rungs/{n}/arms/{arm}/per_seed/{seed}/mean",
      "/rungs/{n}/arms/single_design_K5/per_seed/{seed}/paired_vs_best_learned/mean_paired_delta",
      "/rungs/{n}/arms/single_design_K5/per_seed/{seed}/paired_vs_best_learned_5/mean_paired_delta"]),
    ("fig7", fig7_k_curve,
     "T12: val composition and BOTH byte totals against K = 5..32 wells, with n_seeds printed at every "
     "point and rank/width beside them",
     ["x_cross_val", "x_ledger_K"],
     ["/per_K/{K}/val_composition_mean", "/per_K/{K}/n_seeds",
      "/per_K/{K}/ledger_four_part/times_reference_pin",
      "/rows/{i}/times_reference_pin_parts_1_2_4", "/rows/{i}/rank_over_width_max"]),
    ("fig8", fig8_four_part_ledger,
     "T12: the FOUR-PART byte ledger stacked across K, never netted, against the 524,288 B reference",
     ["x_cross_val"],
     ["/per_K/{K}/ledger_four_part/p1_atom_bytes",
      "/per_K/{K}/ledger_four_part/p2_covariance_plus_count_bytes",
      "/per_K/{K}/ledger_four_part/p3_address_book_bytes",
      "/per_K/{K}/ledger_four_part/p4_coupling_hole_table_bytes"]),
]''')

# ---------------------------------------------------------------- 2. the renderer
RENDER = r'''
def _w13_cellfmt(c):
    """Render a w13 cell.  A 1-seed cell prints ONE number and never `mean (v)`."""
    if isinstance(c, dict) and "values" in c and "provenance" in c:
        f = c.get("format", "{:.4f}")
        vals = c["values"]
        txt = []
        for v in vals:
            if isinstance(v, bool):
                txt.append("true" if v else "false")
            elif isinstance(v, (int, float)):
                txt.append(f.format(v))
            else:
                txt.append(str(v))
        return " / ".join(txt)
    return fmt_any(c)


def _w13_rows_md(rows):
    cols = []
    for r in rows:
        for c in r["cells"]:
            if c not in cols:
                cols.append(c)
    head = ["row", "class", "seeds (n)"] + cols
    body = []
    for r in rows:
        ns = r.get("n_seeds_printed")
        line = [r["label"], r["row_class"], ("—" if ns is None else str(ns))]
        for c in cols:
            line.append(_w13_cellfmt(r["cells"][c]) if c in r["cells"] else "—")
        body.append(line)
    return md_table(head, body)


def _w13_block_md(A, title, d):
    """Render a dict of cells / derived numbers / plain strings as a field table."""
    rows = []
    for k, v in d.items():
        if isinstance(v, dict) and ("values" in v or "derived" in v or "printed" in v):
            rows.append([k, _w13_cellfmt(v), _traced(v)])
        elif isinstance(v, dict) and "status" in v:
            rows.append([k, "**" + v["status"] + "**", v.get("why", "")])
        elif isinstance(v, dict):
            continue
        elif isinstance(v, (list, tuple)):
            rows.append([k, "; ".join(str(x) for x in v), ""])
        else:
            rows.append([k, str(v), ""])
    if not rows:
        return
    A("**" + title + "**")
    A("")
    A(md_table(["field", "value", "traced to"], rows))
    A("")


def _w13_tables_md(A, manifest):
    A("---")
    A("")
    A("# ⭐ WAVE 12's ROWS, FOLDED IN — TABLES 7 to 12")
    A("")
    A("**Added by `w13 spoke/seeds-and-tables`. ZERO scored runs: every number below is re-read "
      "from a wave-12 artefact at the path and pointer printed beside it.**")
    A("")
    nt = manifest["notation"]
    A("## ⛔⛔ THE TWO NOTATIONS THAT MUST NEVER COLLIDE")
    A("")
    A(md_table(["symbol", "means", "range measured", "where it lives"],
               [["**n**", nt["n"]["meaning"], nt["n"]["range_measured"], "T8 (and fig6)"],
                ["**K**", nt["K"]["meaning"], nt["K"]["range_measured"],
                 "T7, T9, T9b, T12 (and fig7, fig8)"]]))
    A("")
    A(f"> ⛔ **{nt['RULE']}**")
    A("")
    A(f"> {nt['the_collision_in_one_line']}")
    A("")
    A("## ⭐ SEED COUNTS ARE A COLUMN, NOT A FOOTNOTE")
    A("")
    sl = manifest["seed_ledger"]
    A(md_table(["table", "what it is", "seeds (n)", "seed ids", "why not more"],
               [[r["table"], r["what"], r["n_seeds"], r["seed_ids"], r["why_not_more"]]
                for r in sl["per_table"]]))
    A("")
    A(f"> ⛔ **{sl['NEVER_MIX']}**")
    A("")
    A(f"> ⭐ **The generation-2 caches for seeds 3 and 4 DO exist and are verified** — "
      f"{sl['caches_for_seeds_3_and_4']}")
    A("")
    A(f"> ⛔ **And they still buy no row.** {sl['why_5_seed_rows_are_impossible_anyway']}")
    A("")

    for tid in ("T7", "T8", "T9", "T9b", "T10", "T11", "T12"):
        t = manifest["tables"][tid]
        A("---")
        A("")
        A("## " + t["title"])
        A("")
        A(f"**Class:** {t['table_class']} · **Placement:** {_w13_cellfmt(t['placement']) if isinstance(t.get('placement'), dict) else t.get('placement', '—')}")
        A("")
        if "notation" in t:
            A("> **Notation on this table.** " + " · ".join(
                f"**{k}**: {v}" for k, v in t["notation"].items()))
            A("")
        A("**Pins.** " + _pin_line(t["pins"]))
        A("")
        A("*Pins traced to:* " + "; ".join(
            f"`{p['path']}` `{p['pointer']}`" for p in t.get("pin_provenance", [])[:4])
          + (" …" if len(t.get("pin_provenance", [])) > 4 else ""))
        A("")
        if "SMOKE_NOT_A_CLAIM_as_the_artefact_records_it" in t:
            A("> ⚠ **The artefact's own `SMOKE_NOT_A_CLAIM` field reads "
              + _w13_cellfmt(t["SMOKE_NOT_A_CLAIM_as_the_artefact_records_it"]) + ".**")
            A("")
        if "SMOKE_NOT_A_CLAIM_as_the_artefacts_record_it" in t:
            A("> ⚠ **The two source artefacts' own `SMOKE_NOT_A_CLAIM` fields read "
              + _w13_cellfmt(t["SMOKE_NOT_A_CLAIM_as_the_artefacts_record_it"])
              + " (CROSSOVER-val.json / LEDGER-ACROSS-K.json).**")
            A("")
        _w13_block_md(A, "Seed count", t.get("seed_count", {}))
        A(_w13_rows_md(t["rows"]))
        A("")
        for key, heading in (("rank_width", "rank / width — the decorative warning"),
                             ("byte_matching", "⛔ BOTH BYTE MATCHINGS, PRINTED"),
                             ("byte_matching_flip", "⛔ BOTH BYTE MATCHINGS, PRINTED — AND THE FLIP"),
                             ("instrument", "Instrument — what this ladder actually varies"),
                             ("the_price_of_keeping_items", "The price of keeping items"),
                             ("the_item_store_forgets_exactly_and_loses_composition",
                              "The other half of the item-store ablation — it forgets, and loses composition"),
                             ("sphere_caveat", "The sphere caveat, MEASURED"),
                             ("control gate at K*", "The control gate at K\\*")):
            if key in t:
                _w13_block_md(A, heading, t[key])
        if "byte_ledger" in t:
            for line in _ledger_md(t["byte_ledger"]):
                A(line)
            A("")
        if "the_probe_that_bounds_the_claim" in t:
            pb = t["the_probe_that_bounds_the_claim"]
            A("**⛔⛔ THE PROBE THAT BOUNDS THE WHOLE DELETION CLAIM**")
            A("")
            A("> " + pb["what"])
            A("")
            A(_w13_rows_md(pb["rows"]))
            A("")
            _w13_block_md(A, "the carry", {k: v for k, v in pb.items()
                                           if isinstance(v, dict) and "values" in v})
            A("> ⛔⛔ **" + pb["THE BOUND, STATED WITHOUT SPIN"] + "**")
            A("")
            A("> ⭐ " + pb["the_contrast_with_wave_8"])
            A("")
        for key in ("THE CAPTION THAT MUST TRAVEL WITH THIS TABLE - TWO CURVES, NOT ONE",
                    "WHAT THE TWO SIDES OF THIS TABLE ARE, AND ARE NOT",
                    "THE POINT THE HOLES MAKE",
                    "language_rule_the_head_set",
                    "the_object_never_reaches_the_retired_pin",
                    "cross_reference"):
            if key in t and isinstance(t[key], str):
                A("> ⛔ **" + key.replace("_", " ") + ":** " + t[key])
                A("")
        if "verdict" in t:
            v = t["verdict"]
            A("> ⭐ **VERDICT.** " + v["statement"])
            A("")
            extra = {k: x for k, x in v.items()
                     if isinstance(x, dict) and "values" in x and "provenance" in x}
            if extra:
                _w13_block_md(A, "verdict, traced", extra)
        for cav in t.get("caveats", []):
            A("> ⚠ " + cav)
            A("")

    gc = manifest["gate_checks"]
    A("---")
    A("")
    A("## ⭐ COMPLETENESS GATES — a published file is not a finished file")
    A("")
    A("> " + gc["why"])
    A("")
    A(md_table(["artefact", "check", "expected", "found", "verdict"],
               [[g["artefact"], g["check"], g["expected"], g["found"], g["verdict"]]
                for g in gc["checks"]]))
    A("")


'''
sub("def render_md(manifest):", RENDER + "def render_md(manifest):")

sub('''    # ---- NOT RUN ----------------------------------------------------------------------
    A("---")''',
    '''    _w13_tables_md(A, manifest)

    # ---- NOT RUN ----------------------------------------------------------------------
    A("---")''')

# ---------------------------------------------------------------- 3. NR6..NR10, C15..C19
EXTRA = r'''
# --------------------------------------------------------------------------------------
# 7w. w13 -- the DECLARED NOT-RUNs wave 12 left behind.  Never a blank, never a zero.
# --------------------------------------------------------------------------------------
NOT_RUN += [
    {
        "id": "NR6",
        "row": "single-design rows 3 and 4 at K* = 32 on TEST at SEEDS 1 AND 2 (T7, T9, T10 our rows)",
        "status": "NOT RUN - 1 of 3 scored runs spent, 2 remain",
        "what_exists": ("run 1/3, seed 0, at `scored/run1_seed0_K32/scored.json`: 60 pair corners + "
                        "49 n-way corners, 10 arms, control gate passed, and the deletion legs beside it."),
        "why": ("wall clock. Run 1/3 cost 1,242.3 s of address fit plus 10 arm passes over 109 corners "
                "plus four deletion legs. Budget after wave 12: `single-design` 1 of 3 spent, 2 remain. "
                "The FIRST thing run 2/3 should buy is a TEST number at K = 5, because the third "
                "pre-declared falsification - 'composition at K* not within 0.02 of K = 5 on TEST though "
                "it was on val' - is NOT directly testable while only K* has a TEST row."),
        "evidence": [{"source": "x_scored1", "path": SOURCES["x_scored1"], "pointer": "/run_n"},
                     {"source": "x_scored1", "path": SOURCES["x_scored1"], "pointer": "/seed"},
                     {"source": "x_findings", "path": SOURCES["x_findings"], "locator": "FINDINGS.md section 11"}],
        "launch_line": (
            "# from the crossover hub's worktree, one scored run per seed\n"
            "for SEED in 1 2 ; do\n"
            "  python scripts/w12_scored_leg.py --seed $SEED --K 32 --condition test \\\n"
            "    --out /c/chlu/outputs/w12-single-design-crossover/scored/run${SEED}_seed${SEED}_K32\n"
            "done"),
        "cost_note": ("about 1,242 s of address fit + about 10 arm passes over 109 corners per seed on "
                      "FORGIS_14 CPU, and each consumes one of the two remaining `single-design` scored runs."),
    },
    {
        "id": "NR7",
        "row": "ANY row at 5 seeds - seeds 3 and 4 on the ladder (T8) and on the deletion rows (T9)",
        "status": ("NOT RUN. ⭐ The generation-2 CACHES for seeds 3 and 4 DO exist, are complete and are "
                   "independently re-hashed - and they still buy no row."),
        "what_exists": ("`CACHE-READY.json` (produced 2026-09-21T15:27:15Z) carries BOTH seeds at "
                        "`status: READY`, `generation: 2`, `scope: full`, 480,000 rows, "
                        "`array_shape [480000, 128, 4, 4]`, 59 block hashes each, "
                        "`shape_dtype_matches_seeds_0_2: true`, bridge complete, bridge gate PROCEED. "
                        "`VERIFY.json` re-hashes every block from the bytes on disk from a process that "
                        "does not import chlu: `all_pass: true`. The SHIPPED re-encode gate passes on "
                        "both seeds at `frac_differing 0.0`, `max_ulp_delta 0.0`."),
        "why": ("the cache is not the binding constraint - the JUDGE is. Every composition number in "
                "wave 12 is produced by `fw.Judge(seed)`, which loads a frozen decoder and a "
                "hash-asserted external classifier. Measured by both wave-12 hubs independently: "
                "`fw.classifier_dir(3)` and `fw.classifier_dir(4)` do NOT exist, `decoder_v1_s3.pkl` "
                "does not exist, and `/c/chlu/outputs/spoke-a-rivals` holds `seed_0`, `seed_1` and "
                "`seed_2` only, so the rivals are missing too. `Judge(3)` raises FileNotFoundError. "
                "SEPARATELY: the caches landed AFTER both scoring hubs closed their seed gates - the "
                "ladder gated at 11:31:33Z when the file was absent, and the crossover read a 12:27:01Z "
                "version in which seed 4 still read `status: NOT_RUN`, 0 blocks. Neither hub could have "
                "used them even if the judge had existed."),
        "evidence": [{"source": "cache_ready", "path": SOURCES["cache_ready"], "pointer": "/seeds/3/status"},
                     {"source": "cache_ready", "path": SOURCES["cache_ready"], "pointer": "/seeds/4/status"},
                     {"source": "cache_ready", "path": SOURCES["cache_ready"], "pointer": "/produced_utc"},
                     {"source": "cache_verify", "path": SOURCES["cache_verify"], "pointer": "/all_pass"},
                     {"source": "cache_gate", "path": SOURCES["cache_gate"], "pointer": "/verdict/passes"},
                     {"source": "lad_findings", "path": SOURCES["lad_findings"], "locator": "FINDINGS.md section 1"},
                     {"source": "x_findings", "path": SOURCES["x_findings"], "locator": "FINDINGS.md section 4"}],
        "launch_line": (
            "# the missing pieces are the JUDGE's, not the cache's: for SEED in 3 4\n"
            "python -m chlu.experiments.exp_gen_bridge --stage all --seed $SEED     # decoder_v1_s$SEED.pkl\n"
            "python -m chlu.experiments.exp_gen_external_classifier --seed $SEED    # classifier_dir($SEED)\n"
            "python -m chlu.eval.rivals.w4_api --fit --seed $SEED                   # spoke-a-rivals/seed_$SEED\n"
            "# only then re-run the ladder and the deletion legs at 5 seeds"),
        "cost_note": ("a decoder + external classifier + five fitted rival families per seed, then the "
                      "18-leg ladder and the deletion legs again at 5 seeds. The caches themselves cost "
                      "396.6 s (seed 3) and 340.9 s (seed 4) and are already paid."),
    },
    {
        "id": "NR8",
        "row": "the 1x BYTE-MATCHED COMPARATOR COLUMN on T7, and three of the four 1x families on T8",
        "status": "NOT RUN on T7 entirely; on T8 ONE family (an MLP, `learned_1x`) exists and three do not",
        "what_exists": ("T8's `learned_1x` is an MLP fitted at the arm's own measured total. That is the "
                        "whole of the 1x opponent set anywhere in wave 12."),
        "why": ("`BASELINES-READY.json` pins `param_target = 131,072` (= 524,288 B / 4) and names no "
                "1x-matched entry point; refitting four families at 535,784 parameters was not "
                "affordable beside the sweep. ⛔ The direction of the error runs AGAINST the programme: "
                "a 1x-matched rival would carry about 4.09x these parameters and would be STRONGER, so "
                "the available comparison FLATTERS the arm and no row is reported as a 1x result. On T8 "
                "the single measured 1x family is therefore the WEAKEST available 1x opponent, and the "
                "arm is already NEGATIVE against it at n = 2, 4, 5 and 6."),
        "evidence": [{"source": "x_scored1", "path": SOURCES["x_scored1"], "pointer": "/baseline_matching/1x_column"},
                     {"source": "x_scored1", "path": SOURCES["x_scored1"], "pointer": "/baseline_matching/1x_reason"},
                     {"source": "base_ready", "path": SOURCES["base_ready"], "pointer": "/pins/param_target"},
                     {"source": "x_cross_val", "path": SOURCES["x_cross_val"],
                      "pointer": "/per_K/32/baseline_match_1x/target_params_float32"},
                     {"source": "lad_findings", "path": SOURCES["lad_findings"], "locator": "FINDINGS.md section 9"}],
        "launch_line": (
            "# fit the four families at THIS object's own total, not at the retired pin\n"
            "python -m chlu.eval.rivals.w12_deletion_baselines --fit \\\n"
            "  --arms mlp,lstm,transformer,cont_mlp --param-target 535784 --seed 0 \\\n"
            "  --out /c/chlu/outputs/w12-deletion-baselines/comparators_1x\n"
            "# then re-score T7's rows 3 and 4 against them"),
        "cost_note": ("four families at about 4.1x the parameter count of the fitted set, plus one "
                      "re-score of 109 corners per family. No new scored run is consumed if it rides on "
                      "the same run as NR6."),
    },
    {
        "id": "NR9",
        "row": "the REFUSAL DISPLAY HEAD-TO-HEAD (T10) - our refusal rate against the refuse-token "
               "transformer's render rate on the SAME post-deletion requests",
        "status": "NOT RUN",
        "what_exists": ("BOTH sides exist separately and are printed in T10. Ours: 2 holes, 1 byte each, "
                        "0 data rows read, no refit, layout asserted equal to `gen_field`'s over 1,322 "
                        "pair slots. Theirs: the refuse-token transformer at 3 seeds with a measured "
                        "separation (AUROC 0.8133-0.8755, false refusal on possible 7.8-12.7 %). ⛔ They "
                        "are NOT the same quantity and T10 does not difference them."),
        "why": ("the crossover hub's wall clock ran out after the scored rows and the deletion legs. "
                "⚠ Carried regardless: wave 9's refuser separated possible from impossible at only "
                "+0.047 and a ridge probe on the same observation scored AUC 0.48-0.55, i.e. chance, so "
                "a DISPLAY comparison against a refuser of that quality is not yet informative - which "
                "is an argument for printing the comparator's FALSE-REFUSAL column beside any such row, "
                "not for skipping the row."),
        "evidence": [{"source": "x_del_test", "path": SOURCES["x_del_test"],
                      "pointer": "/display_refusal/comparator_status"},
                     {"source": "x_del_test", "path": SOURCES["x_del_test"],
                      "pointer": "/display_refusal/carried_caveat"},
                     {"source": "base_ready", "path": SOURCES["base_ready"],
                      "pointer": "/comparators/refuse_token/entry_point"}],
        "launch_line": (
            "# both sides are already on disk; this is one scoring pass, not a fit\n"
            "python scripts/w12_refusal_display.py --seed 0 --K 32 --condition test \\\n"
            "  --ours /c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32 \\\n"
            "  --comparator transformer__refuse_token__value-object_hue-0 \\\n"
            "  --out /c/chlu/outputs/w12-single-design-crossover/display/refusal_test_seed0_K32"),
        "cost_note": "one pass over the post-deletion request set; no fit and no scored run is consumed.",
    },
    {
        "id": "NR10",
        "row": "`generation-table` run 3 on the K* object against the full rival set",
        "status": "NOT RUN - ⭐ the run remains UNSPENT",
        "what_exists": "nothing on this object; `exp_gen_cell.py` carries its own arm registry.",
        "why": ("`exp_gen_cell.py`'s registry (`clu`, `store_removed`, `knn_proto`, `lace`, `ebm_compose`, "
                "`dynread`) would have to be wired to accept a K* single-design object. That is a build, "
                "and wave 12 had no wall clock for it beside the sweep. It is carried here because the "
                "wave plan commissions it in the same clause as T7's rows."),
        "evidence": [{"source": "x_findings", "path": SOURCES["x_findings"], "locator": "FINDINGS.md section 11"},
                     {"source": "wave_plan_12", "path": SOURCES["wave_plan_12"],
                      "locator": "section 1, hub/single-design-crossover item 2"}],
        "launch_line": (
            "# wire the K* object into the cell's arm registry, then\n"
            "python -m chlu.experiments.exp_gen_cell --stage cell --seed 0 --arm single_design_K32\n"
            "python /c/chlu/outputs/hub-generation-table/relaunch/scored/run_arm.py "
            "<out_dir>/seed0 <spec>.json"),
        "cost_note": "a build plus one leg; consumes the last `generation-table` scored run.",
    },
]

CORRECTIONS += [
    {
        "id": "C15 (w13)",
        "claim_as_written": ("the wave-13 brief states that the crossover's scored row 1/3 was spent at "
                             "4 SEEDS, declared, and that the ladder ran 3 seeds."),
        "artefact_says": ("the ladder's 3 seeds are right. The scored crossover row is at ONE seed. "
                          "`scored/` holds exactly one directory, `run1_seed0_K32`, its `/seed` is 0 and "
                          "its `/run_n` is 1. The crossover's own FINDINGS section 9 is headed "
                          "'SCORED RUN 1/3 - rows 3 and 4 on TEST at K* = 32, seed 0'. What IS at 3 seeds "
                          "in that hub is the VAL sweep at K = 5 and K = 32 (`/per_K/5/n_seeds` = 3, "
                          "`/per_K/32/n_seeds` = 3) - and the val sweep labels itself "
                          "`SMOKE_NOT_A_CLAIM: true`."),
        "effect": ("a four-fold overstatement of the evidence behind the main scored row. T7 prints "
                   "`seeds (n)` = 1 in a COLUMN and says so in its title."),
        "action": "T7's seed column and seed_count block; NR6 carries the two unspent runs.",
        "provenance": [{"source": "x_scored1", "path": SOURCES["x_scored1"], "pointer": "/seed"},
                       {"source": "x_scored1", "path": SOURCES["x_scored1"], "pointer": "/run_n"},
                       {"source": "x_cross_val", "path": SOURCES["x_cross_val"], "pointer": "/per_K/32/n_seeds"},
                       {"source": "x_cross_val", "path": SOURCES["x_cross_val"], "pointer": "/SMOKE_NOT_A_CLAIM"}],
    },
    {
        "id": "C16 (w13)",
        "claim_as_written": ("'the honest total FALLS 7.48x -> 2.07x from K = 5 to K = 32, because the "
                             "covariance shrinks faster than the address book grows.'"),
        "artefact_says": ("7.4815x -> 2.0663x is `total_bytes_parts_1_2_4_only` / "
                          "`times_reference_pin_parts_1_2_4` - the total with the ADDRESS BOOK LEFT OUT. "
                          "The HONEST four-part total, `honest_total_bytes`, runs 4,118,034 -> 2,143,138 B, "
                          "i.e. 7.8545x -> 4.0877x, and it is U-SHAPED with a minimum of 3.9942x at "
                          "K = 24. The address book GROWS 195,584 -> 1,059,824 B (5.4x) while the "
                          "covariance falls 3,391,552 -> 560,608 B (6.0x), and past K = 24 the book grows "
                          "FASTER than the covariance falls - the crossover hub's own FINDINGS section 5 "
                          "says exactly that."),
        "effect": ("the stated mechanism is inverted and the quoted curve is a three-part total, not the "
                   "four-part one the referee asked for. A reader would conclude the object gets "
                   "monotonically cheaper as K rises; it does not."),
        "action": ("T12 prints BOTH totals as separate columns with the address-book column between "
                   "them, and the caption names which curve each figure is quoting. fig7 panel B draws "
                   "both; fig8 stacks the four parts."),
        "provenance": [{"source": "x_ledger_K", "path": SOURCES["x_ledger_K"],
                        "pointer": "/rows/0/times_reference_pin_parts_1_2_4"},
                       {"source": "x_ledger_K", "path": SOURCES["x_ledger_K"],
                        "pointer": "/rows/8/times_reference_pin_parts_1_2_4"},
                       {"source": "x_cross_val", "path": SOURCES["x_cross_val"],
                        "pointer": "/per_K/5/ledger_four_part/times_reference_pin"},
                       {"source": "x_cross_val", "path": SOURCES["x_cross_val"],
                        "pointer": "/per_K/24/ledger_four_part/times_reference_pin"},
                       {"source": "x_cross_val", "path": SOURCES["x_cross_val"],
                        "pointer": "/per_K/32/ledger_four_part/times_reference_pin"}],
    },
    {
        "id": "C17 (w13)",
        "claim_as_written": ("the crossover FINDINGS section 7 prints the VAL deletion cost line as "
                             "1.45 / 0.32 / 0.72 / 0.28 s for our adaptation and 13.2 / 14.7 / 12.5 / "
                             "13.7 s for the refit-without reference."),
        "artefact_says": ("`deletion/val_seed0_K5/deletion.json` reads 2.4543 / 0.1874 / 0.6205 / 0.2554 s "
                          "for our adaptation and 23.486 / 11.752 / 10.843 / 11.446 s for refit-without. "
                          "Every other column in that section - rows deleted, bytes rewritten, data rows "
                          "read, paired delta, store-scoped exactness - matches the artefact exactly."),
        "effect": ("the prose understates our own adaptation time and understates the reference's, in "
                   "the same direction, so the RATIO the sentence rests on is roughly preserved - but "
                   "the printed seconds are not the measured seconds. T9b prints the artefact's."),
        "action": "T9b reads every cost figure straight from the artefact; the FINDINGS table is not used.",
        "provenance": [{"source": "x_del_val", "path": SOURCES["x_del_val"],
                        "pointer": "/scopes/value/our_cost_line/seconds"},
                       {"source": "x_del_val", "path": SOURCES["x_del_val"],
                        "pointer": "/scopes/value/exactness_store_scoped/refit_without_cost_line/seconds"},
                       {"source": "x_findings", "path": SOURCES["x_findings"], "locator": "FINDINGS.md section 7"}],
    },
    {
        "id": "C18 (w13)",
        "claim_as_written": ("the crossover FINDINGS section 5 prints the val sweep at K = 5 as 0.8918 "
                             "and at K = 32 as 0.9016, with 'n = 1 seed at every K'."),
        "artefact_says": ("the final `CROSSOVER-val.json` carries THREE seeds at both ends: "
                          "`/per_K/5/val_composition_mean` = 0.895255 over seeds [0, 1, 2] and "
                          "`/per_K/32/val_composition_mean` = 0.929012 over seeds [0, 1, 2]. The seven "
                          "middle K still carry one seed each. The prose was written against an earlier "
                          "state of the same file."),
        "effect": ("the K* row is understated by 0.0274 and the drop from K = 5 is -0.0338, not -0.0098. "
                   "More importantly the seed count differs BETWEEN ROWS OF ONE TABLE, which is exactly "
                   "the hazard this spoke exists to print."),
        "action": ("T12 prints `seeds (n) at this K` as a COLUMN - 3 at K = 5 and K = 32, 1 at the seven "
                   "middle points - and fig7 annotates n at every point."),
        "provenance": [{"source": "x_cross_val", "path": SOURCES["x_cross_val"], "pointer": "/per_K/5/n_seeds"},
                       {"source": "x_cross_val", "path": SOURCES["x_cross_val"], "pointer": "/per_K/8/n_seeds"},
                       {"source": "x_cross_val", "path": SOURCES["x_cross_val"],
                        "pointer": "/per_K/32/val_composition_mean"},
                       {"source": "x_cross_val", "path": SOURCES["x_cross_val"], "pointer": "/per_K/32/drop_from_K5"}],
    },
    {
        "id": "C19 (w13)",
        "claim_as_written": ("the ladder artefact writes its rung as `k` (`/rungs/{k}`) and the crossover "
                             "writes its well count as `K` (`/per_K/{K}`). Read together, the two files "
                             "put a 'k=5' and a 'K=5' on the same page meaning different things - and "
                             "`LADDER.json` also carries `K_star: null`, written when the crossover's "
                             "`K_star` had not yet been decided."),
        "artefact_says": ("`LADDER.json` `/K_of_the_store_arms` = 5 (WELLS) while `/rungs/5` is a rung of "
                          "FIVE JOINTLY COMPOSED FACTORS at that same K = 5. `CROSSOVER-val.json` "
                          "`/K_star` now reads 32. `LADDER.json` `/K_star_source` records the stale read "
                          "in its own words: 'K_star=null, K=5 reference row absent; rule not applicable'."),
        "effect": ("a collision the Hub has made once already. Anyone merging the two tables on 'k' "
                   "produces a table in which one axis is two different quantities."),
        "action": ("`TABLES.json` `/notation` defines both symbols; T8 relabels the rung as **n** and "
                   "prints K = 5 as a COLUMN on every row; T7/T9/T9b/T12 print K and state that n is not "
                   "varied. No figure mixes the two axes."),
        "provenance": [{"source": "lad_json", "path": SOURCES["lad_json"], "pointer": "/K_of_the_store_arms"},
                       {"source": "lad_json", "path": SOURCES["lad_json"], "pointer": "/K_star"},
                       {"source": "lad_json", "path": SOURCES["lad_json"], "pointer": "/K_star_source"},
                       {"source": "x_cross_val", "path": SOURCES["x_cross_val"], "pointer": "/K_star"}],
    },
]

# --------------------------------------------------------------------------------------
# 7x. w13 -- the SEED LEDGER and the COMPLETENESS GATES.
# --------------------------------------------------------------------------------------
SEED_LEDGER = {
    "rule": "the seed count is a COLUMN of every table in this file, never a footnote.",
    "per_table": [
        {"table": "T1 / T2 / T3 / T4 / T5 / T6", "what": "waves 5-9, carried unchanged from w12r",
         "n_seeds": 3, "seed_ids": "0, 1, 2", "why_not_more": "wave-5..9 scored legs; not re-run"},
        {"table": "**T7**", "what": "scored rows 3 and 4 at K* = 32, TEST", "n_seeds": 1, "seed_ids": "0",
         "why_not_more": "NR6 (2 of 3 scored runs unspent) and NR7 (seeds 3-4 unjudgeable)"},
        {"table": "**T8**", "what": "the n-way ladder, n = 2..6, K = 5", "n_seeds": 3, "seed_ids": "0, 1, 2",
         "why_not_more": "NR7"},
        {"table": "**T9**", "what": "deletion, TEST, K* = 32", "n_seeds": 1, "seed_ids": "0",
         "why_not_more": "NR6, NR7"},
        {"table": "**T9b**", "what": "deletion, VAL, K = 5 (SMOKE)", "n_seeds": 1, "seed_ids": "0",
         "why_not_more": "NR6, NR7"},
        {"table": "**T10**", "what": "refusal DISPLAY - ours / the comparator", "n_seeds": "1 / 3",
         "seed_ids": "0  /  0, 1, 2",
         "why_not_more": "our rows NR6; the comparator is already at 3 seeds. The head-to-head is NR9"},
        {"table": "**T11**", "what": "the two ablations, w5 TEST", "n_seeds": 3, "seed_ids": "0, 1, 2",
         "why_not_more": "wave-5 scored legs; never re-run at 5"},
        {"table": "**T12**", "what": "the K-curve, VAL (SMOKE)", "n_seeds": "3 / 1",
         "seed_ids": "0, 1, 2 at K = 5 and K = 32;  0 only at K = 8..28",
         "why_not_more": "wall clock at the seven middle K; NR7 beyond 3"},
    ],
    "NEVER_MIX": ("3-, 4- and 5-seed rows are NEVER plotted together without a printed n per row. In this "
                  "file no row is at 4 or 5 seeds; the counts actually present are 1 and 3, and they are "
                  "printed as a column on every table and annotated on every figure that mixes them."),
    "caches_for_seeds_3_and_4": ("both READY, generation 2, scope full, 480,000 rows, 59 block hashes "
                                 "each, shape and dtype matching seeds 0-2, bridge complete and its gate "
                                 "PROCEED; independently re-hashed from the bytes on disk with "
                                 "`all_pass: true`; the SHIPPED re-encode gate passes on both at "
                                 "`frac_differing 0.0` and `max_ulp_delta 0.0`."),
    "why_5_seed_rows_are_impossible_anyway": (
        "no frozen decoder (`decoder_v1_s3.pkl` absent), no external classifier "
        "(`fw.classifier_dir(3)` and `(4)` absent) and no fitted rivals "
        "(`/c/chlu/outputs/spoke-a-rivals` holds seed_0, seed_1 and seed_2 only) exist for either seed, "
        "so `fw.Judge(3)` raises and no corner of seeds 3 or 4 can be judged. The caches also landed "
        "AFTER both scoring hubs closed their seed gates. See NR7 for the launch line."),
}

GATE_CHECKS = {
    "why": ("a published file is not a finished file. Three times this programme a READY json appeared "
            "while its author was still writing. Every wave-12 artefact this file gates on was checked "
            "COMPLETE - expected counts and non-null fields, not merely present - before a cell was "
            "emitted from it."),
    "checks": [
        {"artefact": "`w12-single-design-crossover`", "check": "END_UTC.txt present",
         "expected": "a UTC stamp", "found": "2026-09-21T14:28:13Z", "verdict": "CLOSED"},
        {"artefact": "`w12-composition-ladder`", "check": "END_UTC.txt present",
         "expected": "a UTC stamp", "found": "2026-09-21T16:54:33Z", "verdict": "CLOSED"},
        {"artefact": "`w12-deletion-baselines`", "check": "END_UTC.txt present",
         "expected": "a UTC stamp", "found": "2026-09-21T12:16:00Z", "verdict": "CLOSED"},
        {"artefact": "`w12-cache-gen2-seeds-3-4`", "check": "END_UTC.txt present",
         "expected": "a UTC stamp", "found": "2026-09-21T15:28:41Z", "verdict": "CLOSED"},
        {"artefact": "`LADDER.json`", "check": "rungs present and no leg failed",
         "expected": "rungs {2,3,4,5,6,shipped}, `not_run` = []",
         "found": "%d rungs, not_run = %r" % (len(DOCS["lad_json"]["rungs"]), DOCS["lad_json"]["not_run"]),
         "verdict": "COMPLETE"},
        {"artefact": "`LADDER.json`", "check": "arms per rung x seeds per arm",
         "expected": "13 arms x 3 seeds at every rung",
         "found": "%d arms, %d seeds at n = 2" % (
             len(DOCS["lad_json"]["rungs"]["2"]["arms"]),
             DOCS["lad_json"]["rungs"]["2"]["arms"]["single_design_K5"]["n_seeds"]),
         "verdict": "COMPLETE"},
        {"artefact": "`BYTES.json`", "check": "fitted legs with a measured ledger",
         "expected": "18 legs per store arm",
         "found": "%d / %d" % (DOCS["lad_bytes"]["summary"]["single_design_K5"]["n_legs"],
                               DOCS["lad_bytes"]["summary"]["pca_tensor"]["n_legs"]),
         "verdict": "COMPLETE"},
        {"artefact": "`scored/run1_seed0_K32/scored.json`", "check": "it is a scored run and its corners add up",
         "expected": "`SCORED` true, 60 + 49 = 109 corners",
         "found": "SCORED = %r, %d + %d = %d" % (
             DOCS["x_scored1"]["SCORED"], DOCS["x_scored1"]["n_corners"]["row3_pair"],
             DOCS["x_scored1"]["n_corners"]["row4_nway"],
             DOCS["x_scored1"]["per_arm"]["cov_tensor"]["all"]["n_corners"]),
         "verdict": "COMPLETE"},
        {"artefact": "`CROSSOVER-val.json`", "check": "every swept K carries a ledger and a mean",
         "expected": "9 K, each with `ledger_four_part` and `val_composition_mean`",
         "found": "%d K, all non-null" % len(DOCS["x_cross_val"]["per_K"]), "verdict": "COMPLETE"},
        {"artefact": "`CACHE-READY.json`", "check": "seeds 3 and 4 READY with 59 block hashes each",
         "expected": "status READY, n_blocks 59, len(block_sha256) 59, shape match true",
         "found": "seed 3: %s / %d / %d / %r ; seed 4: %s / %d / %d / %r" % (
             DOCS["cache_ready"]["seeds"]["3"]["status"], DOCS["cache_ready"]["seeds"]["3"]["n_blocks"],
             len(DOCS["cache_ready"]["seeds"]["3"]["block_sha256"]),
             DOCS["cache_ready"]["seeds"]["3"]["shape_dtype_matches_seeds_0_2"],
             DOCS["cache_ready"]["seeds"]["4"]["status"], DOCS["cache_ready"]["seeds"]["4"]["n_blocks"],
             len(DOCS["cache_ready"]["seeds"]["4"]["block_sha256"]),
             DOCS["cache_ready"]["seeds"]["4"]["shape_dtype_matches_seeds_0_2"]),
         "verdict": "COMPLETE - and it STILL buys no row; see NR7"},
        {"artefact": "`BASELINES-READY.json`", "check": "records present on every seed",
         "expected": "150 records, seeds [0, 1, 2]",
         "found": "%d records, seeds %r" % (DOCS["base_ready"]["n_records_total"],
                                            DOCS["base_ready"]["seeds_present"]),
         "verdict": "COMPLETE"},
    ],
}

'''
_FA = "# 9.  FIGURES -- regenerated here, from the same pointers the tables use."
sub(_FA, EXTRA + _FA)

# ---------------------------------------------------------------- 4. manifest fields
sub('''        "exactness_definition": EXACTNESS_DEFINITION,''',
    '''        "exactness_definition": EXACTNESS_DEFINITION,
        "notation": NOTATION,
        "seed_ledger": SEED_LEDGER,
        "gate_checks": GATE_CHECKS,
        "w13_fold": {
            "spoke": "w13 spoke/seeds-and-tables",
            "starts_from": ["/c/chlu/outputs/w12r-artefact-mustfix/tables-r1/build_tables.py",
                            "/c/chlu/evidence/outputs/w10-seeds-and-tables/"],
            "adds": ["T7 scored rows 3-4 at K* = 32", "T8 the n-way ladder n = 2..6 at K = 5",
                     "T9 deletion on TEST", "T9b deletion on VAL (SMOKE)",
                     "T10 refusal DISPLAY", "T11 the two ablations", "T12 the K-curve",
                     "fig6, fig7, fig8", "NR6..NR10", "C15..C19",
                     "/notation, /seed_ledger, /gate_checks"],
            "scored_runs_spent": 0,
            "verify_tables_py": "BYTE-IDENTICAL to scripts/w10_verify_tables.py - it is the acceptance "
                                "check and was not edited.",
        },''')

sub('''            "A row that was not run is a DECLARED NOT-RUN with a reason and a launch line, never a blank or a zero.",''',
    '''            "A row that was not run is a DECLARED NOT-RUN with a reason and a launch line, never a blank or a zero.",
            "w13: n (jointly composed factors, the ladder rungs) and K (wells, the partition) are NEVER both written k. Every ladder row states its K; every wells row states its n. See /notation.",
            "w13: the seed count is a COLUMN of every table, never a footnote, and rows at different seed counts are never plotted together without it. See /seed_ledger.",
            "w13: BOTH byte matchings are printed on every row that has them - the object's own measured total (1x) and the 524,288 B reference - because they tell opposite stories about the n-trend.",
            "w13: T7 (K = 32, 1 seed, TEST) and T8 (K = 5, 3 seeds) are different objects at different pins and never share a table or a figure axis.",
            "w13: an artefact that labels itself SMOKE_NOT_A_CLAIM is rendered as SMOKE (T9b, T12), never as a RESULT.",''')

sub('''    A("# TABLES.md — the paper's scoreboard, every row at 3 seeds on generation 2")''',
    '''    A("# TABLES.md — the paper's scoreboard, with the seed count printed on every row")''')

SRC.write_text(t, encoding="utf-8")
print("p4 OK ->", SRC)
