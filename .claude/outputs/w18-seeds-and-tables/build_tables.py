"""w18 spoke/seeds-and-tables: render TABLES.md from TABLES.template.md and the cell register.

Every `{{id}}` token in the template is replaced by a registered cell's value plus its pointer.
Pass 2 adds two token forms for dense tables: `{{@ALIAS:LINE}}` prints a row's source pointer,
and `{{id:v}}` prints a cell's value alone. The build FAILS unless a `{{id:v}}` token shares its
template line with the `{{@ALIAS:LINE}}` pointer of that exact cell, so every number still
travels with its file and line.
A cell is registered as (source file under the evidence root, 1-indexed line, value) and the
build FAILS unless the value occurs verbatim on that line. The build also fails if the template,
outside tokens, carries a decimal number or a comma-grouped number: those may only enter
TABLES.md through a checked cell. Nothing is computed; nothing is rounded.

Usage: python build_tables.py [EVIDENCE_ROOT] [OUT_DIR]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/batmac/clu/evidence/outputs")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else HERE)

SRC = {
    "W16": "w16-advisor-report.md",
    "W17": "w17-advisor-report.md",
    "W13A": "w13-advisor-report.md",
    "REF2": "w13-referee-2/REFEREE-v1.md",
    "KL": "w12b-nr6-and-kstar/KSTAR-LADDER.md",
    "KF": "w12b-nr6-and-kstar/FINDINGS.md",
    "DB": "w12-deletion-baselines/FINDINGS.md",
    "PLAN": "advisor-wave-plan-w15-w18.md",
    "W13T": "w13-seeds-and-tables/TABLES.md",
}
# Pass 2 sources (the six landed w15 spokes, the w16 primary record, and the draft itself).
from cells_pass2 import SRC2, CELLS2  # noqa: E402
SRC.update(SRC2)
# Pass 3: w19 `maze-t-symmetric`, the drafts after pass 3, the Hub's corrected report.
from cells_pass3 import SRC3, CELLS3  # noqa: E402
SRC.update(SRC3)
# Pass 4: the Head's DELETION-RULE-AMENDMENT (2026-09-23 22:05) and the drafts after pass 4. The pass-3 draft
# aliases (DNOW, DSTY) are pinned to exact copies of the pass-3 text in draft_pass4/.
from cells_pass4 import SRC4, CELLS4  # noqa: E402
SRC.update(SRC4)
# The pass-2 draft cells describe DRAFT-v2.tex as it was at pass 2 (md5 e7745f2c). The live file has
# since moved, so they are pinned to that exact text, reconstructed from the pass-2 patched copy
# (caption_conventions/DRAFT-v2.captions.tex + `patch -R CAPTIONS.diff`; sha256 equals the pass-2 register).
SRC["DRAFT"] = "@here/draft_pass3/DRAFT-v2.pass2-e7745f2c.tex"
# The Hub CORRECTED w16-advisor-report.md after pass 2, from pass 2's own section 4b. Cells quoting the
# report's earlier text can no longer be checked on the live file: they are checked against the committed
# pass-2 manifest instead (same value, same line, verbatim_on_line true, at the sha256 recorded there),
# and are rendered as HISTORICAL.
PASS2 = json.loads((HERE / "draft_pass3" / "pass2-manifest.json").read_text(encoding="utf-8"))
HISTORICAL = {"W16"}

# id: (source alias, line, value). The value must occur verbatim on that line.
CELLS = {
    # ---- wave 16, CMAZE-T, Config A1 scored TEST leg (w16 report section 1)
    "t_a1_cfg": ("W16", 13, "latent 128, K 2, 6 atoms, rank/width 0.094"),
    "t_a1_runs": ("W16", 13, "3 (1–3)"),
    "t_a1_surv": ("W16", 13, "survives** rivals trained 10× longer"),
    "t_a1_row": ("W16", 21, "a STORE row on CMAZE-T unseen n-way"),
    "t_clu_mean": ("W16", 26, "0.7246"),
    "t_clu_seeds": ("W16", 26, "0.800 / 0.566 / 0.808"),
    "t_best_mean": ("W16", 27, "0.4035"),
    "t_best_seeds": ("W16", 27, "0.394 / 0.395 / 0.421"),
    "t_mlp_lstm": ("W16", 28, "0.2194 / 0.1413"),
    "t_lrem": ("W16", 29, "0.0000"),
    "t_lrem_seeds": ("W16", 29, "0 / 0 / 0"),
    "t_lshuf": ("W16", 30, "0.2248"),
    "t_across": ("W16", 31, "0.2435"),
    "t_ratoms": ("W16", 32, "0.1655"),
    "t_bseq": ("W16", 33, "0.3418"),
    "t_bcol": ("W16", 34, "0.0716"),
    "t_xmaze": ("W16", 35, "0.0000"),
    "t_xmaze_seeds": ("W16", 35, "0 / 0 / 0"),
    "t_sb1": ("W16", 37, "margin 0.321, ahead on every seed"),
    "t_sb2": ("W16", 37, "every control ≤ CLU − 0.15"),
    "t_sb3": ("W16", 37, "learned blind max 0.342 vs bar 0.675"),
    "t_audit": ("W16", 37, "Replay **0 disagreements**, **0 impossible rows**, disjointness passes on all three TEST banks"),
    "t_harness_warn": ("W16", 22, "The w15 harness shipped ready **with a warning** that a charter-C control did not collapse on the pair row"),
    # ---- CMAZE-T against longer-trained rivals
    "t_80k": ("W16", 44, "re-tested Task 1 against **80,000-step** rivals"),
    "t_80k_mlp": ("W16", 44, "mlp 0.392/0.000/0.369"),
    "t_80k_lstm": ("W16", 44, "lstm 0.001/0.010/0.352"),
    "t_80k_tf": ("W16", 44, "transformer 0.418/0.388/0.389"),
    "t_80k_clu": ("W16", 44, "all far below the CLU"),
    "t_c512_cfg": ("W16", 15, "latent 512, K 2, 8 atoms"),
    "t_c512_runs": ("W16", 15, "3 (7–9)"),
    "t_c512_val": ("W16", 15, "0.725 vs 0.453"),
    "t_c512_riv": ("W16", 15, "vs 10×-trained rivals"),
    "t_c512_cav": ("W16", 51, "Config C-512 seed 1: `random_atoms` 0.464 vs CLU"),
    # ---- CMAZE-T negatives and deletion
    "t_roll": ("W16", 48, "CMAZE-T rollout validity = 0.000 for every learned arm"),
    "t_roll_step": ("W16", 48, "Per-step 0.925"),
    "t_roll_len": ("W16", 48, "compounds over 43 steps (0.93^43 ≈ 0.04)"),
    "t_roll_link": ("W16", 48, "Named link: READ"),
    "t_roll_q": ("W16", 65, "Rollout validity is 0.000 for every arm"),
    "t_del": ("W16", 52, "relative ΔS = 0.0 on every seed; after-deletion == refit-without"),
    "t_del_on": ("W16", 52, "Exact deletion holds on the winner"),
    "t_cfgb": ("W16", 16, "CMAZE-T Config B (match-bit interface)"),
    "t_cfgb_val": ("W16", 16, "FAILED VAL (VB3)"),
    "t_cfgb_why": ("W16", 50, "Config B's task-shaped interface lifts the RIVAL twice as much as the CLU"),
    "plan_del_winner": ("PLAN", 48, "On the winning task/config: deletion rows (value, conjunction)"),
    # ---- CMAZE-O
    "o_c_cfg": ("W16", 14, "CMAZE-O · Config C (C-field)** (K 8, 64 atoms)"),
    "o_c_runs": ("W16", 14, "3 (4–6)"),
    "o_c_val": ("W16", 14, "0.693 vs 0.167"),
    "o_c_wd": ("W16", 14, "rival margin WITHDRAWN"),
    "o_c_budget": ("W16", 41, "against rivals at the recipe's training budget"),
    "o_conv": ("W16", 42, "With converged (30k-step) rivals"),
    "o_conv_clu": ("W16", 42, "CLU n-way 0.088 / 0.150 / 0.173"),
    "o_conv_riv": ("W16", 42, "best rival 0.315 / 0.413 / 0.463"),
    "o_field": ("W16", 42, "field CLU 0.70 < field MLP 0.99"),
    "o_latent": ("W16", 42, "latent CLU 0.14 < latent rival ~0.40"),
    "o_amend": ("W16", 44, "`PREREG-T2 AMENDMENT 4` now *requires* 30k-step rivals"),
    "o_cost": ("W16", 45, "It cost **three scored runs on a claim that does not stand**"),
    "o_neg": ("W16", 49, "CMAZE-O yields no store row at any config"),
    "o_neg_link": ("W16", 49, "Named link: INTERFACE"),
    "o_a": ("W16", 17, "CMAZE-O Config A (rank/width ≤ 0.3)"),
    "o_a_val": ("W16", 17, "FAILED VAL (VB0)"),
    "o_a_why": ("W16", 17, "the AE latent cannot be carried by a ≤ 0.3 store"),
    "o_c2": ("W16", 18, "CMAZE-O Config C2 (rank/width 0.5)"),
    "o_c2_val": ("W16", 18, "FAILED VAL (VB1, 10×-trained rivals)"),
    "o_1012": ("W16", 58, "any field config would need n-way ≥ 1.02 to clear the 0.99 MLP"),
    # ---- budget and fallback
    "b_spend": ("W16", 4, "Scored spend: 9 of 18"),
    "b_spent": ("W16", 57, "1–3, 4–6, 7–9"),
    "b_1012": ("W16", 58, "10–12 (T2 cap)"),
    "b_1318": ("W16", 59, "13–18 (Task 3)"),
    "b_t3why": ("W16", 59, "the brief opens Task 3 *\"ONLY if Tasks 1–2 yield no win\"*; Task 1 won"),
    "b_record": ("W16", 3, "`/c/chlu/outputs/w16-maze-wm/`"),
    "b_merge": ("W16", 3, "the two w15 harness branches merged (`b1ca534`)"),
    # ---- wave 15, the only quoted figure on batmac
    "x_tol": ("W17", 199, "tolerance is **0.02**"),
    "x_ks": ("W17", 199, "K = 10, 16, 20, 24, 28 **and at K\\* = 32** (0.0259)"),
    # ---- the 1x bracket
    "br_bracket": ("W13A", 85, "938,413–1,710,324"),
    "br_lo": ("REF2", 185, "938,413 params"),
    "br_lo_conv": ("REF2", 186, "`p2_if_symmetric_triangle_float32`"),
    "br_hi": ("REF2", 186, "1,710,324 params"),
    "br_hi_conv": ("REF2", 186, "(float64 triangle)"),
    "br_used": ("W13A", 86, "3,241,076"),
    "br_n1": ("W13A", 73, "1,029,776-param"),
    "br_ladder_fit": ("KF", 119, "3,131,564 – 3,260,030"),
    "br_kstar": ("KF", 33, "536,270–536,972 params"),
    "br_n1_rows": ("REF2", 192, "labelled **1 seed · replication · not scored**"),
    # ---- byte-convention sources for the audit
    "bc_blob": ("DB", 122, "`bytes_rewritten` is the arm's whole"),
    "bc_blob2": ("DB", 123, "parameter blob"),
    # ---- conventions quoted in the audit
    "mf2": ("PLAN", 34, "23.51× convention"),
    "pin": ("W13T", 623, "524,288 B reference"),
    # ---- the ladder note
    "ladder_note": ("PLAN", 26, "we beat the naive fair rivals at every n; the CLU's own 2-way→6-way rise is not to be inferred because rungs 3–6 share corners and differ in coverage"),
}

for _reg in (CELLS2, CELLS3, CELLS4):
    for _cid in _reg:
        if _cid in CELLS:
            raise SystemExit(f"duplicate cell id {_cid}")
    CELLS.update(_reg)
HIST_USED = set()

NUM_RE = re.compile(r"(?<![\w.§])(\d+(?:\.\d+)?[eE][-+]?\d+|\d+\.\d+|\d{1,3}(?:,\d{3})+)(?![\w])")
TOKEN_RE = re.compile(r"\{\{([a-z0-9_]+)(:v)?\}\}")
PTR_RE = re.compile(r"\{\{@([A-Z0-9]+):(\d+)\}\}")
ANY_RE = re.compile(r"\{\{[^}]*\}\}")
HEADING_RE = re.compile(r"^#+\s+\d+(?:\.\d+)*")  # section numbers are not data


def src_path(alias):
    rel = SRC[alias]
    if rel.startswith("@here/"):
        return HERE / rel[len("@here/"):]
    return Path(rel) if rel.startswith("/") else EVIDENCE / rel


def verify_cells():
    manifest, failures = {}, []
    for cid, (alias, line, value) in CELLS.items():
        path = src_path(alias)
        lines = path.read_text(encoding="utf-8").splitlines()
        text = lines[line - 1] if 0 < line <= len(lines) else ""
        ok = value in text
        entry = {"value": value, "source": SRC[alias], "line": line, "verbatim_on_line": ok}
        if not ok and alias in HISTORICAL:
            old = PASS2["cells"].get(cid, {})
            if (old.get("value") == value and old.get("line") == line and old.get("verbatim_on_line")
                    and old.get("source") == SRC[alias]):
                ok = True
                HIST_USED.add(cid)
                entry.update(verbatim_on_line=False, historical=True,
                             verified_against="pass-2 manifest", pass2_source_sha256=PASS2["source_sha256"][SRC[alias]])
        if not ok:
            failures.append(f"{cid}: {value!r} not on {SRC[alias]}:{line}")
        manifest[cid] = entry
    return manifest, failures


def main():
    manifest, failures = verify_cells()
    template = (HERE / "TABLES.template.md").read_text(encoding="utf-8")
    used = {m[0] for m in TOKEN_RE.findall(template)}
    unknown = sorted(used - set(CELLS))
    unused = sorted(set(CELLS) - used)
    tlines = template.splitlines()
    bare = [(i + 1, m.group(0)) for i, ln in enumerate(tlines)
            for m in NUM_RE.finditer(HEADING_RE.sub("", ANY_RE.sub("", ln)))]
    # a value-only token must share its line with the pointer of that exact cell
    orphans, bad_ptr = [], []
    for i, ln in enumerate(tlines):
        ptrs = {(a, int(n)) for a, n in PTR_RE.findall(ln)}
        for a, n in ptrs:
            if a not in SRC:
                bad_ptr.append(f"template l.{i + 1}: unknown alias {a}")
            elif not 0 < n <= len(src_path(a).read_text(encoding="utf-8").splitlines()):
                bad_ptr.append(f"template l.{i + 1}: {a}:{n} out of range")
        for cid, v in TOKEN_RE.findall(ln):
            if v and cid in CELLS and (CELLS[cid][0], CELLS[cid][1]) not in ptrs:
                orphans.append(f"template l.{i + 1}: {{{{{cid}:v}}}} has no pointer {CELLS[cid][0]}:{CELLS[cid][1]} on its line")
    if failures or unknown or bare or orphans or bad_ptr:
        for f in failures:
            print("CELL FAIL", f)
        for u in unknown:
            print("UNKNOWN TOKEN", u)
        for ln, num in bare:
            print(f"UNSOURCED NUMBER template l.{ln}: {num}")
        for o in orphans + bad_ptr:
            print("POINTER FAIL", o)
        sys.exit(1)

    def disp(alias):
        rel = SRC[alias]
        return rel[len("@here/"):] if rel.startswith("@here/") else rel

    def render(m):
        cid = m.group(1)
        alias, line, value = CELLS[cid]
        if cid in HIST_USED:
            sha = PASS2["source_sha256"][SRC[alias]][:8]
            return value if m.group(2) else f"{value} `[{disp(alias)}:{line}, HISTORICAL: pass-2 text, sha256 {sha}…; since corrected by the Hub]`"
        return value if m.group(2) else f"{value} `[{disp(alias)}:{line}]`"

    out = PTR_RE.sub(lambda m: f"`[{disp(m.group(1))}:{m.group(2)}]`", TOKEN_RE.sub(render, template))
    leftover = [(i + 1, t) for i, ln in enumerate(out.splitlines()) for t in re.findall(r"\{\{[^}]*\}\}", ln)]
    bad_ids = [cid for cid in CELLS if not re.fullmatch(r"[a-z0-9_]+", cid)]
    if leftover or bad_ids:
        for ln, t in leftover:
            print(f"UNRENDERED TOKEN TABLES.md l.{ln}: {t}")
        for cid in bad_ids:
            print(f"BAD CELL ID (must be [a-z0-9_]+): {cid}")
        sys.exit(1)
    src_hashes = {SRC[a]: hashlib.sha256(src_path(a).read_bytes()).hexdigest() for a in SRC}
    (OUT / "TABLES.md").write_text(out, encoding="utf-8")
    (OUT / "TABLES-w18.json").write_text(json.dumps({
        "schema": "chlu-w18-seeds-and-tables/1",
        "evidence_root": str(EVIDENCE),
        "source_sha256": src_hashes,
        "cells": manifest,
        "cells_unused_in_template": unused,
        "cells_historical": sorted(HIST_USED),
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: {len(CELLS)} cells verbatim-verified on their lines; "
          f"{len(used)} used in TABLES.md; {len(HIST_USED)} historical (pass-2 manifest); "
          f"0 unsourced numbers in the template; unused={len(unused)}")


if __name__ == "__main__":
    main()
