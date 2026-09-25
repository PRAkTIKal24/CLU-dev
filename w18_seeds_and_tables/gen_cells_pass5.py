"""PASS 5 generator (run ONCE; its outputs are static and every build re-verifies them).

Writes cells_pass5.py (the register: SRC5, CELLS5, JCELLS5, JEQ5) and TABLES.pass5.frag.md (the template sections
§13-§16, spliced into TABLES.template.md). Nothing is computed: a CELLS5 value is copied from its artefact line, a
JCELLS5 value is a printed number checked against a JSON leaf at the printed precision, and a JEQ5 entry asserts
that two JSON leaves are EQUAL (the two ties). usage: python gen_cells_pass5.py"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = Path("/Users/batmac/clu/evidence/outputs")
sys.path.insert(0, str(HERE.parent / "w22_seeds5"))
from jnum import resolve, matches  # noqa: E402
import w15_check_lib as W  # noqa: E402

D5V = sorted((HERE / "draft_pass5").glob("DRAFT-v2.pass5-*.tex"))[0]
D5S = sorted((HERE / "draft_pass5").glob("DRAFT-v2-style.pass5-*.tex"))[0]

SRC5 = {
    "D5V": "@here/draft_pass5/" + D5V.name, "D5S": "@here/draft_pass5/" + D5S.name,
    "BDF5": "w15-bar-d-score/FINDINGS.md", "CCF": "w15-coverage-control/FINDINGS.md", "LPF": "w15-lace-poe-rows/FINDINGS.md",
    "IDC": "w15-coverage-control/IDENTITY-CHECK.txt",
    "JCC": "w15-coverage-control/COVERAGE-CONTROL.json", "JG": "w15-lace-poe-rows/GENTABLE-KSTAR.json",
    "JBD": "w15-bar-d-score/BAR-D.json", "JXG": "w15-bar-d-score/CROSS-ARM-GPU-test.json",
    "JRT": "w15-bar-d-score/score/alone/test_seed0/READ-TABLE.json",
    "JP0": "w8-encoder-loop/cloud/manifests/seed_0/alone/probe.json",
    "JP1": "w8-encoder-loop/cloud/manifests/seed_1/alone/probe.json",
    "JP2": "w8-encoder-loop/cloud/manifests/seed_2/alone/probe.json",
    "W20F": "w20-shapes-data-matched/FINDINGS.md", "W20M": "w20-shapes-data-matched/SHAPES-MATCHED.md",
    "JW20": "w20-shapes-data-matched/SHAPES-MATCHED.json",
    "LRF": "w21-ladder-rivals-32k/FINDINGS.md", "LRT": "w21-ladder-rivals-32k/TABLES.md", "LRC": "w21-ladder-rivals-32k/CAPTION.md",
    "JLR": "w21-ladder-rivals-32k/LADDER-RIVALS-32K.json",
    "WFT": "w21-words-full-test/TABLES.md", "JWF": "w21-words-full-test/test/BARS-WORDS-FULL-test.json",
    "S4T": "w21-maze-t-seed4/TABLES.md", "S4F": "w21-maze-t-seed4/FINDINGS.md",
    "RNT": "w21-rel-newseeds/TABLES.md", "RNF": "w21-rel-newseeds/FINDINGS.md",
    "JR4": "w21-rel-newseeds/test/BOTH_test_seed4.json", "JR5": "w21-rel-newseeds/test/BOTH_test_seed5.json",
    "JR3": "w21-rel-newseeds/test/BOTH_test_seed3.json", "JR4P": "w21-rel-newseeds/replay/BOTH_test_seed4_PERTUPLE.json",
    "JC0": "w21-maze-t-clu-improve/test/BOTH_test_seed0.json", "JC1": "w21-maze-t-clu-improve/test/BOTH_test_seed1.json",
    "JC2": "w21-maze-t-clu-improve/test/BOTH_test_seed2.json",
    "JA0": "w19-maze-t-symmetric/test/BOTH_test_seed0.json", "JA1": "w19-maze-t-symmetric/test/BOTH_test_seed1.json",
    "JA2": "w19-maze-t-symmetric/test/BOTH_test_seed2.json", "JA3": "w21-maze-t-seed4/test/BOTH_test_seed3.json",
    "JA4": "w21-maze-t-seed4/test/BOTH_test_seed4.json", "JA5": "w21-maze-t-seed4/test/BOTH_test_seed5.json",
    "SFP": "@outputs/w19-success-failure-panels/LOG.md",
}
ALIAS_OF = {v: k for k, v in SRC5.items()}
CELLS5, JCELLS5, JEQ5 = {}, {}, {}
_n = {}


def path_of(alias):
    rel = SRC5[alias]
    if rel.startswith("@here/"):
        return HERE / rel[6:]
    if rel.startswith("@outputs/"):
        return Path("/Users/batmac/clu/outputs") / rel[9:]
    return EV / rel


_lines = {}


def lines(alias):
    if alias not in _lines:
        _lines[alias] = path_of(alias).read_text(encoding="utf-8").splitlines()
    return _lines[alias]


def nid(prefix):
    _n[prefix] = _n.get(prefix, 0) + 1
    return f"p5_{prefix}_{_n[prefix]}"


def cell(prefix, alias, line, value, v=False):
    assert value in lines(alias)[line - 1], (alias, line, value)
    cid = nid(prefix)
    CELLS5[cid] = (alias, line, value)
    return "{{" + cid + (":v}}" if v else "}}")


def find(alias, needle, nth=1, after=None):
    """the unique line holding `needle`; with `after`, the first such line after the unique line holding `after`."""
    start = find(alias, after) if after else 0
    hits = [i for i, ln in enumerate(lines(alias), 1) if needle in ln and i > start]
    assert hits, (alias, needle, after)
    if after is None:
        assert len(hits) == 1, (alias, needle, hits)
    return hits[nth - 1] if after is None else hits[0]


def cf(prefix, alias, needle, value=None):
    """cell on the unique line holding `needle`; value defaults to needle."""
    return cell(prefix, alias, find(alias, needle), value or needle)


def jcell(prefix, alias, ptr, printed):
    val = resolve(json.loads(path_of(alias).read_text(encoding="utf-8")), ptr)
    if isinstance(val, str):
        assert val == printed, (alias, ptr, val, printed)
    else:
        m = re.fullmatch(r"(\d+)(?: of |/)(\d+)", printed)
        assert (val == int(m.group(1))) if m else matches(printed, val), (alias, ptr, val, printed)
    cid = nid(prefix)
    JCELLS5[cid] = (alias, ptr, printed)
    return "{{" + cid + "}}"


def jeq(prefix, alias_a, ptr_a, alias_b, ptr_b, printed, n_items=None):
    a = resolve(json.loads(path_of(alias_a).read_text(encoding="utf-8")), ptr_a)
    b = resolve(json.loads(path_of(alias_b).read_text(encoding="utf-8")), ptr_b)
    assert a == b, (ptr_a, ptr_b)
    if n_items is not None:
        assert len(a) == n_items
    cid = nid(prefix)
    JEQ5[cid] = (alias_a, ptr_a, alias_b, ptr_b, printed, n_items)
    return "{{" + cid + "}}"


NUMTOK = re.compile(r"(?<![\w.§])([+\-−]?(?:\d+(?:\.\d+)?[eE][-+]?\d+|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+))(?![\w])")


def rowcopy(prefix, alias, line):
    """A source table row, copied verbatim; each decimal / comma-grouped / scientific token becomes a value-only
    cell, and the row's pointer is appended as a last column."""
    src = lines(alias)[line - 1]
    out, pos = [], 0
    for m in NUMTOK.finditer(src):
        out.append(src[pos:m.start()])
        out.append(cell(prefix, alias, line, m.group(1), v=True))
        pos = m.end()
    out.append(src[pos:])
    row = "".join(out).rstrip()
    ptr = "{{@" + alias + ":" + str(line) + "}}"
    return row + " " + ptr + " |" if row.endswith("|") else row + " " + ptr


def rowcopy_find(prefix, alias, needle, after=None):
    return rowcopy(prefix, alias, find(alias, needle, after=after))


def header(alias, needle, after=None):
    """A header row (copied verbatim, numbers as cells) plus its separator, with a pointer column."""
    ln = find(alias, needle, after=after)
    h = rowcopy("hdr", alias, ln)
    ncol = lines(alias)[ln - 1].count("|") - 1
    return h + "\n" + "|---" * (ncol + 1) + "|"


out = []
P = out.append

# =====================================================================================================
# §13 the w15 fold
# =====================================================================================================
P("## 13. ⛔⛔ PASS 5: THE w15 FOLD, CLOSED. EVERY w15 NUMBER THE DRAFTS PRINT, CHECKED THREE WAYS")
P("")
P("Passes 3 and 4 declared this fold out of brief (§6.1 stayed STALE). It is in brief now, and it is **closed**: the "
  "drafts print these numbers, and from this pass the register checks every one of them.")
P("")
P("**Method (`w22_seeds5/w15_check.py`, `w15_map.py`).** Every number token that either draft prints under a "
  "`% SRC:` line or `% SRCBLOCK:` naming `w15-bar-d-score`, `w15-coverage-control` or `w15-lace-poe-rows` is found by "
  "`scan_draft.py` (decimals, signed decimals, comma-grouped integers, integers of four or more digits, and `a of b` / "
  "`a/b` counts). Each gets three cells in the rows below:")
P("1. **the draft**: the value, verbatim on its line of the pinned pass-5 copy of each draft (`draft_pass5/`, the "
  "installed bytes);")
P("2. **the record the draft cites**: verbatim on a line of that spoke's `FINDINGS.md` (the provenance comment's own "
  "claim);")
P("3. **the artefact** (a paraphrase is not a source): the JSON leaf named by an RFC-6901 pointer, and the build "
  "re-checks that the printed value **is** that leaf at the printed precision (exact, or rounded half-up / half-even "
  "from the float's repr). A count is checked as integer equality. Where the artefact is plain text "
  "(`IDENTITY-CHECK.txt`), the cell is verbatim on its line.")
P("")
P("Numbers the token scanner cannot see (integer zeros inside a table cell, counts that wrap a line) are listed "
  "after each spoke's table, each against its JSON leaves.")
P("")

check_v = W.run(D5V)
check_s = W.run(D5S)
assert not check_v["problems"] and not check_s["problems"], (check_v["problems"], check_s["problems"])
assert len(check_v["rows"]) == len(check_s["rows"])
FALIAS = {"w15-bar-d-score/FINDINGS.md": "BDF5", "w15-coverage-control/FINDINGS.md": "CCF", "w15-lace-poe-rows/FINDINGS.md": "LPF"}
spokes = [("w15-coverage-control", "13.1", "`coverage-control` → `tab:ladder`'s coverage row and the coverage-control paragraph"),
          ("w15-lace-poe-rows", "13.2", "`lace-poe-rows` → `tab:gentable-kstar`, `tab:lace-poe` (both blocks), the LACE/PoE paragraphs"),
          ("w15-bar-d-score", "13.3", "`bar-d-score` → the Criterion D line, `tab:bard-gpu`, the decomposition paragraph")]
n_ok = {"rows": 0, "json": 0}
disagree = []
for spoke, sec, title in spokes:
    rows = [(a, b) for a, b in zip(check_v["rows"], check_s["rows"]) if a["findings"].startswith(spoke)]
    P(f"### {sec} {title}: {len(rows)} numbers")
    P("")
    P("| # | `DRAFT-v2.tex` | `DRAFT-v2-style.tex` | the cited `FINDINGS.md` | the artefact (JSON leaf at the printed precision, or text line) |")
    P("|---|---|---|---|---|")
    for k, (rv, rs) in enumerate(rows, 1):
        assert rv["token"] == rs["token"], (rv, rs)
        tok = rv["token"]
        c1 = cell("w15d", "D5V", rv["draft_line"], tok)
        c2 = cell("w15s", "D5S", rs["draft_line"], tok)
        if rv["findings_line"] is None:
            assert rv["artefact_ok"] and tok == "27.135", rv
            c3 = "not in FINDINGS, which prints the truncated value; the draft line now also cites the JSON (§13.4)"
        else:
            c3 = cell("w15f", FALIAS[rv["findings"]], rv["findings_line"], rv["findings_verbatim"])
        if rv.get("pointer"):
            al = ALIAS_OF[rv["artefact"]]
            if rv["artefact_ok"]:
                c4 = jcell("w15j", al, rv["pointer"], tok) + f" ({rv['rounding']})"
                n_ok["json"] += 1
            else:
                c4 = "⛔ **DISAGREES**: the leaf is " + jcell("w15x", al, rv["pointer"], str(rv["json_value"]))
                disagree.append((rv, rs))
        elif rv["artefact"] is None:
            c4 = "a count: checked against its JSON leaves in the list below this table"
            n_ok["json"] += 1
        else:
            c4 = cell("w15t", ALIAS_OF[rv["artefact"]], rv["artefact_line"], rv["artefact_verbatim"])
            n_ok["json"] += 1
        n_ok["rows"] += 1
        P(f"| {k} | {c1} | {c2} | {c3} | {c4} |")
    P("")
    ex = [(a, b) for a, b in zip(check_v["extras"], check_s["extras"]) if a["artefact"].startswith(spoke)]
    if ex:
        P(f"Numbers the scanner cannot see ({spoke}):")
        P("")
        for ev, es in ex:
            al = ALIAS_OF[ev["artefact"]]
            leaves = " · ".join(jcell("w15e", al, p, _fmt) for p, _fmt in
                                ((p, W.fmt_leaf(resolve(json.loads(path_of(al).read_text(encoding='utf-8')), p))) for p in ev["pointers"]))
            P(f"- {cell('w15ed', 'D5V', ev['draft_line'], ev['draft_verbatim'])} and {cell('w15es', 'D5S', es['draft_line'], es['draft_verbatim'])}: "
              f"the JSON leaves it reads ({len(ev['pointers'])}): {leaves}")
        P("")
P("### 13.4 ⛔ The one disagreement, and what was done")
P("")
assert len(disagree) == 0, "the installed drafts must carry the corrected value"
P("The pre-pass-5 drafts printed PoE's budget as " + cell("w15q", "LPF", find("LPF", "3,556,665 parameters = **27.13×"), "27.13×")
  + " the reference, in two places per draft, copying the spoke's `FINDINGS.md`. The acceptance JSON holds "
  + jcell("w15q", "JG", "/PART_B_rows_3_4/arms/PoE/budget_per_seed/0/x_store_reference", "27.135")
  + ", which is " + jcell("w15q", "JG", "/PART_B_rows_3_4/arms/PoE/budget_per_seed/0/deployed_bytes", "14226660")
  + " B over the " + jcell("w15q", "JG", "/PART_A_generation_table/table/clu/byte_ledger/budget_bytes", "524288")
  + " B reference. Rounded to two decimals, that ratio is one hundredth above what FINDINGS prints: FINDINGS truncated "
  "where it should have rounded. **The artefact wins.** Both drafts now print the JSON's own value, "
  + jcell("w15q", "JG", "/PART_B_rows_3_4/arms/PoE/budget_per_seed/0/x_store_reference", "27.135")
  + "×, with `GENTABLE-KSTAR.json` added to the line's `% SRC:` (`w22_seeds5/apply_pass5_edits.py`, edit E7). The rows "
  "above check the corrected value.")
P("")
P(f"**Result.** {n_ok['rows']} numbers per draft, each verbatim in both drafts and in its cited `FINDINGS.md`, and each "
  f"equal to its artefact ({n_ok['json']} of {n_ok['rows']}; the extra cells cover the counts and zeros). "
  "**Nothing in the three spokes is left unfolded.** One value quoted by `bar-d-score` is not its own: the `alone` probe "
  "margins are w8's, and they are checked against the w8 probe JSON that `bar-d-score`'s `FINDINGS.md` names. Where a "
  "draft line cites FINDINGS for a number the JSON holds at more digits, the draft's rounding is the one printed in "
  "the rows above (half-up or half-even).")
P("")

# =====================================================================================================
# §14 w20 shapes-data-matched
# =====================================================================================================
P("## 14. WAVE 20 `hub/shapes-data-matched` (3 of 3 scored runs spent). FOLDED from `SHAPES-MATCHED.json` and its rendered tables")
P("")
P("§11 said \"awaiting\". It has landed: `evidence/outputs/w20-shapes-data-matched/`. Rows are copied verbatim from "
  "`SHAPES-MATCHED.md` (rendered from the JSON by the hub); the headline is also checked against the JSON leaf.")
P("")
P("### 14.1 The headline: the data-matched margin on rows 3–4, Shapes3D, K\\* = 32, TEST, 3 seeds")
P("")
hl = "/headline/data_matched_margin_rows34_K32"
P("- margin, mean: " + jcell("w20h", "JW20", hl + "/mean", "7.03") + "; per seed "
  + jcell("w20h", "JW20", hl + "/per_seed/0", "7.49") + " / " + jcell("w20h", "JW20", hl + "/per_seed/1", "8.60") + " / "
  + jcell("w20h", "JW20", hl + "/per_seed/2", "5.02") + "; verdict " + jcell("w20h", "JW20", hl + "/verdict", "SHRINKS"))
P("- against the rival at its best over every fit that ran (10× and 30× included): "
  + jcell("w20h", "JW20", "/headline/vs_best_over_all_fits_incl_30x/mean", "6.54"))
P("- the draft's old margin against the 16k-row comparators, recomputed from the same score files: "
  + jcell("w20h", "JW20", "/headline/draft_margin_recomputed/min", "39.7") + " … "
  + jcell("w20h", "JW20", "/headline/draft_margin_recomputed/max", "49.1"))
P("")
P(header("W20M", "| store | seed | strongest rival | store | rival | Δ | CI95 | won/tied/lost |", after="### P3 — THE DATA-MATCHED MARGIN"))
for s in ("0", "1", "2"):
    P(rowcopy_find("w20p3", "W20M", f"| store_K32_n131072 | {s} | mlp__tenx@test |"))
P(rowcopy_find("w20p3", "W20M", "| store_K32_n131072 | **mean** |"))
P("")
P("### 14.2 P1 convergence: DECLARED against 10× the steps, per family and seed (the carried fit is the last column)")
P("")
W1 = "### P1 — convergence"
P(header("W20M", "| family | seed | DECLARED | TENX | TENX − DECLARED | P1 | carried |", after=W1))
for fam in ("mlp", "lstm", "transformer", "cont_mlp"):
    for s in "012":
        P(rowcopy_find("w20p1", "W20M", f"| {fam} | {s} | ", after=W1))
P("")
P("### 14.3 Table M-131k: the arms (rows 3–4, TEST; fit rows in the arm label)")
P("")
P(header("W20M", "| arm | role | all corners mean ± sd | per seed |", after="## Table M-131k"))
for needle in ("| store K*=32, fit 131,072 rows |", "| store K=5, fit 131,072 rows |", "| store K*=32, fit 16,384 rows |",
               "| store K*=32, fit 32,768 rows |", "| prototype table |", "| mlp, 131,072 rows, DECLARED",
               "| mlp, 131,072 rows, 10x steps |", "| lstm, 131,072 rows, DECLARED", "| transformer, 131,072 rows, DECLARED",
               "| cont_mlp, 131,072 rows, 10x steps |", "| MLP 130,514 p, 32,768 rows", "| mlp__declared@test30x |"):
    P(rowcopy_find("w20a", "W20M", needle, after="## Table M-131k"))
P("")
P("### 14.4 Matched deletion: the four measurements, every scope and seed (rows 3–4 survivor corners, the store's full pool)")
P("")
P(header("W20M", "| scope | seed | deleted rows | (a) store max|mode dev|"))
for sc in ("value", "conjunction_live", "item"):
    for s in "012":
        P(rowcopy_find("w20d", "W20M", f"| {sc} | {s} | " + {"value": "11044", "conjunction_live": "5511", "item": "1311"}[sc]))
P("")
P("Statistic size and book refit, from the JSON: the book's exact statistic is "
  + jcell("w20d", "JW20", "/deletion_matched/per_scope/conjunction_live/store/0/b/b1_statistic/statistic_bytes", "967304800")
  + " B (the fit data itself); (d) on the conjunction at seed 0 is "
  + jcell("w20d", "JW20", "/deletion_matched/per_scope/conjunction_live/store/0/d/paired_after_minus_before/mean_paired_delta", "-6.128")
  + ", and downdate − refit-without is "
  + jcell("w20d", "JW20", "/deletion_matched/per_scope/conjunction_live/store/0/d/downdate_minus_refit_without_pipeline/mean_paired_delta", "0.0") + ".")
P("")
P("Rivals' retrain-without on the same rows (the conjunction scope, every family and seed):")
P("")
P(header("W20M", "| scope | seed | family | rival as-trained → retrain-without (survivors) |"))
for fam in ("mlp", "lstm", "transformer", "cont_mlp"):
    for s in "012":
        P(rowcopy_find("w20r", "W20M", f"| conjunction_live | {s} | {fam} | "))
P("")
P("⛔ **A report-vs-artefact disagreement in w20's own `FINDINGS.md`.** It says: "
  + cf("w20x", "W20F", "the rivals lose < 1 novel/corner when retrained without red∧sphere")
  + ". The rows above say otherwise at seed 2: the MLP goes " + cf("w20x", "W20M", "| conjunction_live | 2 | mlp | 43.174 → 40.220", "43.174 → 40.220")
  + " and the continuous MLP " + cf("w20x", "W20M", "| conjunction_live | 2 | cont_mlp | 26.844 → 17.275", "26.844 → 17.275")
  + ". **The artefact wins.** (w22 draft-rulings found the same; the drafts print each rival's before → after.)")
P("")
P("### 14.5 The ladder, data-matched set (each rung's own pool): store minus the strongest data-matched rival")
P("")
for k in range(2, 7):
    ln = [i for i, t in enumerate(lines("W20M"), 1) if t.startswith("store_K32_n131072 − strongest DATA-MATCHED rival")]
    P("- n = " + str(k) + ": " + rowcopy("w20l", "W20M", ln[k - 2]))
P("")

# =====================================================================================================
# §15 w21
# =====================================================================================================
P("## 15. WAVE 21. FOLDED from each unit's own TABLES.md and run JSONs")
P("")
P("### 15.1 `words-full-test`: CMAZE-T, condition WORDS-FULL (the 318-wide design given to EVERY arm), TEST, 3 seeds. **LOSS**")
P("")
P(header("WFT", "| leg | row | n_seeds | CLU (s0 / s1 / s2) |", after="## 1. The label"))
P(rowcopy_find("wf", "WFT", "| TEST | `nway` | 3 |"))
P(rowcopy_find("wf", "WFT", "| VAL (w19) | `nway` | 3 |"))
P("")
P("From the label JSON: " + jcell("wf", "JWF", "/HEAD_LABEL/PRIMARY", "LOSS") + "; mean paired difference "
  + jcell("wf", "JWF", "/HEAD_LABEL/rows/nway/mean_diff", "-0.0175") + "; seeds with the CLU ahead "
  + jcell("wf", "JWF", "/HEAD_LABEL/rows/nway/seeds_CLU_ahead", "0") + " of 3.")
P("")
P("### 15.2 `maze-t-seed4`: the SIX-seed table (BOTH, TEST, n-way), the per-seed CEILING row, and the easy/hard split")
P("")
P("The cut: " + cf("s4", "S4T", "cut = 0.6407 (gap 0.1467 between 0.5674 and 0.7141)") + "; easy "
  + cf("s4", "S4T", "Easy = [0, 2, 3, 5]; hard = [1, 4]") + ". Six seeds tried, six kept, none discarded.")
P("")
A0 = "## 0. Seed classification by measured CEILING"
P(header("S4T", "| seed | origin | BOTH TEST ceiling | reached by | CLU |", after=A0))
for s in range(6):
    P(rowcopy_find("s4c", "S4T", f"| {s} | {'original' if s < 3 else 'added'} | ", after=A0))
P("")
A1 = "row `nway` (exact) · seeds = 6 [0, 1, 2, 3, 4, 5] · rows/seed [761, 742, 686, 675, 723, 660]"
P(header("S4T", "| arm | mean all (6) | mean easy [0, 2, 3, 5] |", after=A1))
for needle in ("| CLU | 0.677 |", "| rival (matched, 80,000 steps): lstm |", "| rival (matched, 80,000 steps): mlp |",
               "| rival (matched, 80,000 steps): transformer |", "| **PER-SEED CEILING** (best trained arm) |",
               "| best MATCHED RIVAL per seed", "| CLU − best matched rival (per seed) |", "| CLU − ceiling |"):
    P(rowcopy_find("s4n", "S4T", needle, after=A1))
P("")
P("Per-seed CLU (A1) from the run JSONs: " + " / ".join(
    jcell("s4j", f"JA{s}", "/ROWS/nway/CLU", v) for s, v in enumerate(("0.800", "0.566", "0.808", "0.674", "0.533", "0.682"))))
P("")
P("Sensitivity of the split (every possible cut):")
P("")
P(header("S4T", "| cut between | gap | hard seeds |", after=A0))
for needle in ("| 0.5353 – 0.5674 |", "| 0.5674 – 0.7141 |", "| 0.7141 – 0.8076 |", "| 0.8076 – 0.8305 |", "| 0.8305 – 0.9394 |"):
    P(rowcopy_find("s4s", "S4T", needle, after=A0))
P("")
P("### 15.3 `maze-t-clu-improve` + `rel-newseeds`: CLU-REL-A10 at six seeds, with the `+rel` diagnostic beside it")
P("")
R0 = "## 0. Six-seed side-by-side"
P(header("RNT", "| quantity | all (6) [0, 1, 2, 3, 4, 5] | orig 0-2 |", after=R0))
for needle in ("| A1 CLU |", "| A1 best matched rival |", "| REL CLU |", "| REL ceiling |", "| REL best matched rival |",
               "| REL best +rel |", "| REL max control |"):
    P(rowcopy_find("rn", "RNT", needle, after=R0))
P("")
P("Per-seed REL CLU from the run JSONs: "
  + " / ".join(jcell("rnj", a, "/ROWS/nway/CLU", v) for a, v in (("JC0", "0.908"), ("JC1", "0.894"), ("JC2", "0.984"),
                                                                    ("JR3", "0.927"), ("JR4", "0.900"), ("JR5", "0.939"))))
P("")
P("The `+rel` diagnostic (rivals handed the CLU's relational columns; never a rival row, never a label), n-way, TEST:")
P("")
R2 = "## 2. ⭐⭐ The `+rel` DIAGNOSTIC"
P(header("RNT", "| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean (6) |", after=R2))
for needle in ("| CLU | 0.908 |", "| rival (matched, 80,000 steps): transformer |", "| diagnostic+rel@80k: transformer +rel |",
               "| **best matched rival (frozen input, 8k/80k)** |", "| **best +rel diagnostic (8k/80k)** |",
               "| +rel lift = best +rel − best matched rival |", "| CLU − best matched rival | +0.105 |", "| CLU − best +rel |",
               "| share of the CLU's margin over the best matched rival"):
    P(rowcopy_find("rnr", "RNT", needle, after=R2))
P("")
P("### 15.4 ⛔⛔ THE TWO TIES, PRINTED AS CELLS. They are measurements, as much as the wins are")
P("")
P("**Seed 4: a charter-C control ties the improved CLU.** `control: random_atoms` equals CLU-REL-A10 on n-way:")
P("")
R3 = "## 3. ⭐⭐ The four charter-C controls against the CLU"
P(header("RNT", "| seed | leg | config | n | CLU | landscape_removed |", after=R3))
P(rowcopy_find("tie4", "RNT", "| 4 | TEST | CLU-REL-A10 | 723 |", after=R3))
P(rowcopy_find("tie4", "RNT", "| 4 | VAL (free) | CLU-REL-A10 | 696 |", after=R3))
P("")
P("- run JSON, n-way: " + jeq("tie4", "JR4", "/ROWS/nway/CLU", "JR4", "/ROWS/nway/control: random_atoms",
                               "CLU == control: random_atoms == 0.9004149377593361") )
P("- per held-out tuple (zero-fit replay, every object hash-checked before loading): "
  + jeq("tie4", "JR4P", "/PER_TUPLE_NWAY/CLU", "JR4P", "/PER_TUPLE_NWAY/control: random_atoms",
        "the CLU's and random_atoms' per-tuple counts are identical on all 16 held-out tuples", n_items=16)
  + "; objects checked " + jcell("tie4", "JR4P", "/n_objects_checked", "24"))
P("- the record's own words: " + cf("tie4", "RNF", "`control: random_atoms` = CLU = **0.9004** on n-way"))
P("")
P("**Seed 5: the best matched rival ties the improved CLU.** The matched transformer at ten times its steps reads what the CLU reads:")
P("")
P("- run JSON, n-way: " + jeq("tie5", "JR5", "/ROWS/nway/CLU", "JR5", "/ROWS/nway/rival@10x-steps: transformer",
                               "CLU == rival@10x-steps: transformer == 0.9393939393939394"))
P("- the table row: " + rowcopy_find("tie5", "RNT", "| CLU − best matched rival | +0.105 |", after=R2))
P("- the record's own words: " + cf("tie5", "RNF", "(transformer @80k, 0.939 = 0.939)"))
P("")
P("Where both drafts print them, as measurements: seed 4 in the text, "
  + cf("dtie", "D5V", "(random atoms, 0.9004, equal on every held-out tuple)") + " / "
  + cf("dtie", "D5S", "(random atoms, 0.9004, equal on every held-out tuple)") + ", and in `tab:cmazet-sym`, "
  + cf("dtie", "D5V", "\\textbf{0.900} / 0.827 (seed 4: random atoms equals the store)") + " / "
  + cf("dtie", "D5S", "\\textbf{0.900} / 0.827 (seed 4: random atoms equals the store)") + "; seed 5 in `tab:cmazet-sym`, "
  + cf("dtie", "D5V", "best matched rival 0.803 / 0.567 / 0.685 / 0.723 / 0.624 / 0.939") + " / "
  + cf("dtie", "D5S", "best matched rival 0.803 / 0.567 / 0.685 / 0.723 / 0.624 / 0.939") + ", which equals the store's own seed-5 value in the same row.")
P("")
P("So on seed 4 the store is not load-bearing for CLU-REL-A10 (a random-atoms control reads the same on every held-out "
  "tuple), and on seed 5 CLU-REL-A10 has no margin over a matched rival. Both drafts print both ties (`tab:cmazet-sym` "
  "and the CLU-REL-A10 paragraphs).")
P("")
P("### 15.5 `ladder-rivals-32k`: the THIRD rival set (the store-content set). Pool AND steps in every header")
P("")
P("The caption the unit ships (every pool and step budget), verbatim: " + cf("lr", "LRC", "**Set A** (the draft's ladder rivals, spoke-A `@ref`): **32,256**", "**32,256**")
  + " rows / " + cf("lr", "LRC", "**5,040** steps. **Set B**", "**5,040**") + " steps (A); "
  + cf("lr", "LRC", "the rung's **131,072**-row pool", "**131,072**") + " rows / "
  + cf("lr", "LRC", "DECLARED **20,480** steps, TENX **204,800** steps (mlp", "DECLARED **20,480** steps, TENX **204,800** steps") + " (B); "
  + cf("lr", "LRC", "the store's own **32,768** content rows", "**32,768**") + " rows / "
  + cf("lr", "LRC", "DECLARED **5,120** steps, TENX **51,200** steps", "DECLARED **5,120** steps, TENX **51,200** steps")
  + " plus step-matched (C).")
P("")
P(header("LRT", "| rung | store K\\*=32 (content 32,768 rows) |"))
for k in range(2, 7):
    P(rowcopy_find("lr3", "LRT", f"| n={k} | " + {2: "32.64", 3: "61.61", 4: "66.09", 5: "66.58", 6: "66.60"}[k] + " |"))
P("")
P("Margins, 3-seed mean per rung, from the JSON:")
P("")
P("| set | n = 2 | n = 3 | n = 4 | n = 5 | n = 6 | mean over rungs (the unit's FINDINGS; arithmetic of the five JSON leaves) |")
P("|---|---|---|---|---|---|---|")
sets = [("A (spoke-A `@ref`)", "setA_vs_strongest", ("10.76", "15.62", "20.09", "20.58", "20.61"), "A +17.53"),
        ("B (w20, data-matched)", "setB_vs_strongest_declared_or_tenx_RECOMPUTED", ("5.13", "11.18", "12.55", "10.69", "10.30"), "B +9.97"),
        ("C (w21, store-content)", "setC_vs_strongest_declared_or_tenx", ("6.08", "11.48", "14.00", "9.47", "10.36"), "C +10.28"),
        ("C-sm (w21, step-matched to B)", "setC_vs_strongest_stepmatched", ("5.87", "13.40", "14.54", "10.62", "9.30"), "C-sm +10.75")]
for label, key, vals, mean in sets:
    row = [jcell("lrm", "JLR", f"/rungs/{k}/margins_K32_store_minus_rival/{key}/mean_delta/mean", v) for k, v in zip(range(2, 7), vals)]
    P(f"| {label} | " + " | ".join(row) + " | " + cf("lrmean", "LRF", mean) + " |")
P("")
P("Every bootstrap interval of sets B, C and C-sm lies above zero; the lowest lower bound is "
  + jcell("lr", "JLR", "/rungs/2/margins_K32_store_minus_rival/setC_vs_strongest_stepmatched/per_seed/2/ci95/0", "1.98")
  + " (n = 2, seed 2, step-matched MLP).")
P("")
P("**X1, the single-fit variance (it bears on every single-fit margin in the paper).** Two MLP fits with the same rows, "
  "width, schedule and steps, differing only in the random draw:")
P("")
X1 = "## Table X1"
P(header("LRT", "| rung | ref s0/s1/s2 [mean ± sd] |", after=X1))
for k in range(2, 7):
    P(rowcopy_find("x1", "LRT", f"| n={k} | " + {2: "5.57", 3: "8.22", 4: "8.54", 5: "9.87", 6: "10.67"}[k] + " / ", after=X1))
P("")
P("- the record: " + cf("x1", "LRF", "Per-seed paired Δ ranges from −2.15 to +4.13") + "; "
  + cf("x1", "LRF", "per-corner CI95 excludes 0 on **12 of 15** legs", "**12 of 15**") + ".")
P("")

# =====================================================================================================
# §16 placeholders
# =====================================================================================================
P("## 16. PLACEHOLDERS AFTER PASS 5 (both drafts, installed bytes)")
P("")
P("| placeholder | state | reason |")
P("|---|---|---|")
P("| the ladder's store-content rival set (`tab:ladder`) | ⭐ **FILLED** in both drafts from §15.5 (edit E3). The block prints "
  "all eight family fits (selected and ten-times steps), the paired margin row, and the step-matched margin row. Pool and "
  "steps are in every block header and in the caption (edits E1, E2) | `w21/ladder-rivals-32k` landed |")
for key, fig in (("w16 CMAZE-T per-arm generated $s_{t+1}$ (store, transformer, MLP, LSTM) with", "fig:cmazet-families"),
                 ("w16 CMAZE-T per-arm", "fig:cmazet-door-tint"),
                 ("w16 CMAZE-T post-deletion per-arm generated", "fig:cmazet-deletion")):
    ln = [i for i, t in enumerate(lines("D5V"), 1) if "\\PLACEHOLDER{" + key in t]
    ln = ln[0] if key != "w16 CMAZE-T per-arm" else [i for i, t in enumerate(lines("D5V"), 1) if t.rstrip() == "\\PLACEHOLDER{w16 CMAZE-T per-arm"][0]
    P(f"| {cell('ph', 'D5V', ln, chr(92) + 'PLACEHOLDER{' + key)} (`{fig}`) | ⛔ **PERMANENTLY BLOCKED**, left in place | "
      + cf("ph", "SFP", "w16 persisted NO arm object and NO per-transition prediction") + ". The placeholder index "
      "(`app:placeholders`) now says so in both drafts (edit E5) |")
P("")
P("Counts in the installed drafts: `\\PLACEHOLDER{` in the text, 3 each (was 4); `\\TODO{` in the text, 4 each "
  "(unchanged). The four TODOs are the Head's abstract CMAZE-T sentence, the `pca_tensor` byte ratio, the AI-use "
  "statement and the code archive: none is a table cell, and none is this spoke's to fill.")
P("")

(HERE / "TABLES.pass5.frag.md").write_text("\n".join(out) + "\n", encoding="utf-8")
reg = ['"""Pass-5 cell register (static; generated once by gen_cells_pass5.py, then re-verified by every build).',
       "",
       "CELLS5: (alias, line, value) verbatim on the line. JCELLS5: (alias, JSON pointer, printed value), the printed value",
       "must be the JSON leaf at its printed precision (strings: equal). JEQ5: (alias A, pointer A, alias B, pointer B,",
       'printed text, n items or None): the two JSON values must be equal (and hold n items)."""', "",
       "SRC5 = " + repr(SRC5), "", "CELLS5 = " + repr(CELLS5), "", "JCELLS5 = " + repr(JCELLS5), "", "JEQ5 = " + repr(JEQ5), ""]
(HERE / "cells_pass5.py").write_text("\n".join(reg), encoding="utf-8")
print(f"cells {len(CELLS5)}, json cells {len(JCELLS5)}, json equalities {len(JEQ5)}; w15 rows {n_ok}")
