"""w18 spoke/seeds-and-tables: render TABLES.md from TABLES.template.md and the cell register.

Every `{{id}}` token in the template is replaced by a registered cell's value plus its pointer.
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

NUM_RE = re.compile(r"(?<![\w.§])(\d+\.\d+|\d{1,3}(?:,\d{3})+)(?![\w])")
TOKEN_RE = re.compile(r"\{\{([a-z0-9_]+)\}\}")
HEADING_RE = re.compile(r"^#+\s+\d+(?:\.\d+)*")  # section numbers are not data


def verify_cells():
    manifest, failures = {}, []
    for cid, (alias, line, value) in CELLS.items():
        path = EVIDENCE / SRC[alias]
        lines = path.read_text(encoding="utf-8").splitlines()
        text = lines[line - 1] if 0 < line <= len(lines) else ""
        ok = value in text
        if not ok:
            failures.append(f"{cid}: {value!r} not on {SRC[alias]}:{line}")
        manifest[cid] = {"value": value, "source": SRC[alias], "line": line, "verbatim_on_line": ok}
    return manifest, failures


def main():
    manifest, failures = verify_cells()
    template = (HERE / "TABLES.template.md").read_text(encoding="utf-8")
    used = set(TOKEN_RE.findall(template))
    unknown = sorted(used - set(CELLS))
    unused = sorted(set(CELLS) - used)
    bare = [(i + 1, m.group(0)) for i, ln in enumerate(template.splitlines())
            for m in NUM_RE.finditer(HEADING_RE.sub("", TOKEN_RE.sub("", ln)))]
    if failures or unknown or bare:
        for f in failures:
            print("CELL FAIL", f)
        for u in unknown:
            print("UNKNOWN TOKEN", u)
        for ln, num in bare:
            print(f"UNSOURCED NUMBER template l.{ln}: {num}")
        sys.exit(1)

    def render(m):
        alias, line, value = CELLS[m.group(1)]
        return f"{value} `[{SRC[alias]}:{line}]`"

    out = TOKEN_RE.sub(render, template)
    src_hashes = {SRC[a]: hashlib.sha256((EVIDENCE / SRC[a]).read_bytes()).hexdigest() for a in SRC}
    (OUT / "TABLES.md").write_text(out, encoding="utf-8")
    (OUT / "TABLES-w18.json").write_text(json.dumps({
        "schema": "chlu-w18-seeds-and-tables/1",
        "evidence_root": str(EVIDENCE),
        "source_sha256": src_hashes,
        "cells": manifest,
        "cells_unused_in_template": unused,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: {len(CELLS)} cells verbatim-verified on their lines; "
          f"{len(used)} used in TABLES.md; 0 unsourced numbers in the template; unused={unused}")


if __name__ == "__main__":
    main()
