#!/usr/bin/env python
"""w10 spoke/seeds-and-tables -- build TABLES.json, TABLES.md and figures/ from artefacts only.

Reads nothing but the artefact files registered in SOURCES below.  Writes only into
/c/chlu/outputs/w10-seeds-and-tables/.  Every cell emitted into TABLES.json carries the
source id and the JSON pointer it was read from, so a reader can regenerate every number
and every figure from the manifest alone.

    python build_tables.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def _default_out() -> Path:
    """Run from this directory, or from anywhere with W10_OUT_DIR / argv[1] pointing at it."""
    here = Path(__file__).resolve().parent
    return here if (here / "START_UTC.txt").exists() else Path("C:/chlu/outputs/w10-seeds-and-tables")


OUT = Path(os.environ.get("W10_OUT_DIR") or
           (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "")
           or _default_out())
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# 0.  THE SOURCE REGISTER.  Nothing outside this dict is read.
# --------------------------------------------------------------------------------------
SOURCES = {
    # --- T1: CMAZE-2 (request layer over CMAZE-1), hub/maze, TEST, 3 scored seeds --------
    "maze_s0": "/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json",
    "maze_s1": "/c/chlu/outputs/w9-maze/runs/scored2_seed1_test.json",
    "maze_s2": "/c/chlu/outputs/w9-maze/runs/scored3_seed2_test.json",
    "maze_amend_val_s0": "/c/chlu/outputs/w9-maze/runs/amend_exactdelete_seed0_val.json",
    "maze_replica_s0": "/c/chlu/outputs/w9-maze/replication/replica_seed0_test.json",
    "maze_replication_md": "/c/chlu/outputs/w9-maze/replication/REPLICATION.md",
    "maze_ledger": "/c/chlu/outputs/w9-maze/LEDGER.md",
    "maze_log": "/c/chlu/outputs/w9-maze/LOG.md",
    "maze_prereg1": "/c/chlu/outputs/w9-maze/PREREG-w9-maze-1.md",
    # --- T2: Shapes3D, w8/encoder-loop, 3 seeds ------------------------------------------
    "enc_crossarm": "/c/chlu/outputs/w8-encoder-loop/CROSS-ARM.json",
    "enc_report": "/c/chlu/outputs/w8-encoder-loop/run/encoder_loop_report.json",
    "enc_log": "/c/chlu/outputs/w8-encoder-loop/LOG.md",
    "enc_launchspec": "/c/chlu/outputs/w8-encoder-loop/cloud/LAUNCH-SPEC.md",
    # --- T3: CMAZE-1 PROPER, w9/harness tier, 3 seeds ------------------------------------
    "harness_s0": "/c/chlu/outputs/w9-harness/harness_seed0.json",
    "harness_s1": "/c/chlu/outputs/w9-harness/harness_seed1.json",
    "harness_s2": "/c/chlu/outputs/w9-harness/harness_seed2.json",
    "harness_results": "/c/chlu/outputs/w9-harness/RESULTS.md",
    # --- T4: exact deletion, second implementation ---------------------------------------
    "downdate_test": "/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json",
    "downdate_val": "/c/chlu/outputs/w8-downdate/DOWNDATE-READY.json",
    "downdate_log": "/c/chlu/outputs/w8-downdate/LOG.md",   # w12r/MF3: the four-part ledger table
    # --- context / declared not-runs ------------------------------------------------------
    "single_design_log": "/c/chlu/outputs/w8-single-design/LOG.md",
    # --- w12r/MF12+MF13: the w5 compose-field byte and parameter ledger ---------------
    "w5_read_s0": "/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json",
    "w5_read_s1": "/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json",
    "w5_read_s2": "/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json",
    "w5_log": "/c/chlu/outputs/w5-compose-field/LOG.md",
    "gentable_spec": "/c/chlu/outputs/hub-generation-table/relaunch/scored/scored_spec.json",
    "gentable_log": "/c/chlu/outputs/hub-generation-table/LOG-iter2.md",
    "gentable_runarm": "/c/chlu/outputs/hub-generation-table/relaunch/scored/run_arm.py",
    "envscout_env": "/c/chlu/outputs/w8-env-scout/ENV.md",
    "envscout_defaults": "/c/chlu/outputs/w8-env-scout/screen_results_DEFAULTS.txt",
    "wave_plan": "/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md",
    "w8_advisor": "/c/chlu/evidence/outputs/w8-advisor-report.md",
}

SEEDS = (0, 1, 2)

# --------------------------------------------------------------------------------------
# 0b. w12r -- THE THREE DEFINITIONS THE PROSE MUST CITE.  Added by w12r spoke/artefact-
#     mustfix to close referee MF4 (define "exact"), MF10 (CMAZE-1 vs CMAZE-2) and the
#     "9/9 seeds" phrasing that three spokes independently flagged.
# --------------------------------------------------------------------------------------
EXACTNESS_DEFINITION = {
    "id": "EXACT-STORE-SCOPED",
    "term": "exact",
    "scope": "STORE-SCOPED",
    "definition": (
        "EXACT means: after deleting a set of rows, the STORE CONTENT equals, to float roundoff, "
        "the content fitted from the retained rows alone, HOLDING THE FROZEN ENCODER AND DECODER "
        "FIXED. Neither encoder nor decoder is refit at any point. It is NOT a claim that the "
        "renderer forgets, and it is NOT retrain-equivalence of the generative model."
    ),
    "what_it_is_not": [
        "NOT retrain-equivalence of the generative model - the renderer is never refit",
        "NOT cost-free - on TEST the same deletion takes surviving-target composition from 0.1024 to 0.0000",
        "NOT a byte-level claim about the address book - the basis-change carry rewrites the book and is declared separately",
    ],
    "why_only_this_scope_is_supported": (
        "The frozen autoencoder's own training set contains eval-bank items, so the encoder and "
        "decoder weights fitted with the deleted rows present are untouched by the deletion."
    ),
    "evidence": [
        {"source": "maze_s0", "path": SOURCES["maze_s0"], "pointer": "/ae_train_items",
         "what": "frozen autoencoder training inventory, seed 0: eval_bank_train 543 / pool 4860 / pool_sites 270"},
        {"source": "maze_s1", "path": SOURCES["maze_s1"], "pointer": "/ae_train_items",
         "what": "seed 1: eval_bank_train 529"},
        {"source": "maze_s2", "path": SOURCES["maze_s2"], "pointer": "/ae_train_items",
         "what": "seed 2: eval_bank_train 552"},
        {"source": "maze_log", "path": SOURCES["maze_log"], "locator": "LOG.md:376",
         "what": "every arm is rendered by the same FROZEN DECODER through the same latent standardisation"},
        {"source": "maze_s0", "path": SOURCES["maze_s0"],
         "pointer": "/display/delete_green_exactness/refit",
         "what": "refit = False - the deletion refits nothing"},
    ],
    "cite_as": "TABLES.json /exactness_definition",
}

ENVIRONMENTS = {
    "CMAZE-1": {
        "what": "the env-scout adopted colour-maze generator `w8_envscout.maze_colour`, called with its own defaults",
        "tables_that_use_it": ["T3"],
        "blind_sequential_published": 0.4980,
        "blind_sequential_at_the_w9_harness_tier": 0.5317,
        "evidence": [
            {"source": "envscout_env", "path": SOURCES["envscout_env"], "locator": "ENV.md:42,49,358",
             "what": "published CMAZE-1 composition-blind (BLIND-SEQUENTIAL) score 0.4980"},
            {"source": "harness_s0", "path": SOURCES["harness_s0"],
             "pointer": "/columns/blind-sequential (key order from A)/validity",
             "what": "CMAZE-1 proper at the w9/harness tier, seed 0 = 0.5550; seeds 1,2 = 0.5300 / 0.5100; mean 0.5317"},
        ],
    },
    "CMAZE-2": {
        "what": ("a REQUEST LAYER over CMAZE-1: the engine, the solver and the exact integer judge are "
                 "`w8_envscout.maze_colour`, IMPORTED UNMODIFIED; the layer adds named agents and named "
                 "targets inside one maze, placed across two door-free components, and seals one target "
                 "behind walls per maze"),
        "tables_that_use_it": ["T1", "T4 (the w9/maze leg)", "T5 (implementation 1)"],
        "blind_sequential_on_this_split": 0.9037,
        "evidence": [
            {"source": "maze_log", "path": SOURCES["maze_log"], "locator": "LOG.md:366-372",
             "what": "the w9/maze FINAL REPORT's own one-paragraph definition of CMAZE-2"},
            {"source": "maze_s0", "path": SOURCES["maze_s0"],
             "pointer": "/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity",
             "what": "0.9077 on seed 0; 0.8905 / 0.9128 on seeds 1,2; mean 0.9037"},
        ],
    },
    "THE_THREE_NUMBERS_ONLY_READ_COHERENTLY_ONCE_SEPARATED":
        "0.4980 published CMAZE-1 | 0.5317 CMAZE-1 at the w9/harness tier (T3) | 0.9037 CMAZE-2 (T1)",
    "RULE": ("T1 (CMAZE-2) and T3 (CMAZE-1) MUST NEVER SHARE A TABLE. They are a DIFFERENT "
             "ENVIRONMENT, not merely a different budget."),
}

SEED_AND_CONFIGURATION_COUNTS = {
    "why_this_field_exists": (
        "The phrase '9/9 seeds' is wrong as written and must never be printed. The nine legs are "
        "3 CONFIGURATIONS x 3 SEEDS inside ONE implementation. Seed count and configuration count "
        "are recorded here as SEPARATE fields so no sentence can multiply them into a seed count."
    ),
    "never_print": "9/9 seeds",
    "implementations": {
        "w9/maze CluRead.delete_rows_exact": {
            "seeds": 3, "configurations": 1, "legs": 3, "scored_runs": 3,
            "configuration": "K=8, n_atoms=31 - the one pinned CMAZE-2 configuration",
        },
        "w8 spoke/downdate-build exp_w8_downdate": {
            "seeds": 3, "configurations": 1, "legs": 3, "scored_runs": 0,
            "configuration": "factor partition K=5, A=64 (the incumbent's grain) on Shapes3D features",
        },
        "w8 hub/single-design matched-bytes candidates": {
            "seeds": 3, "configurations": 3, "legs": 9, "scored_runs": 0,
            "configuration": "ward_balanced K=40 / K=44 / K=48 - THIS is where the phrase '9/9' comes from",
        },
    },
    "correct_phrasings": [
        "3 seeds x 3 configurations inside ONE implementation (hub/single-design) = 9 legs",
        "3 seeds at ONE configuration in each of the two other implementations",
        "THREE implementations in total, printed as three rows - see T5",
    ],
}


def native(p: str) -> Path:
    """The manifest records MSYS-style paths (/c/chlu/...); Python needs C:/chlu/... on Windows."""
    m = re.match(r"^/([a-zA-Z])/(.*)$", p)
    return Path(f"{m.group(1).upper()}:/{m.group(2)}") if m else Path(p)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_json(sid: str):
    return json.loads(native(SOURCES[sid]).read_text(encoding="utf-8"))


def ptr(obj, pointer: str):
    """RFC-6901-ish JSON pointer read; raises if the path is absent."""
    cur = obj
    for tok in pointer.strip("/").split("/"):
        tok = tok.replace("~1", "/").replace("~0", "~")
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


# --------------------------------------------------------------------------------------
# 1.  Cell helpers.  A cell is ALWAYS (values-per-seed, source id, pointer-per-seed).
# --------------------------------------------------------------------------------------
def cell(src_ids, pointer, docs, fmt="{:.4f}"):
    """src_ids: list of 3 source ids (one per seed); pointer: identical pointer in each."""
    vals, prov = [], []
    for sid in src_ids:
        v = ptr(docs[sid], pointer)
        vals.append(v)
        prov.append({"source": sid, "path": SOURCES[sid], "pointer": pointer})
    numeric = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = {"seeds": list(SEEDS)[: len(vals)], "values": vals, "provenance": prov, "format": fmt}
    if len(numeric) == len(vals) and numeric:
        out["mean"] = sum(numeric) / len(numeric)
        out["min"] = min(numeric)
        out["max"] = max(numeric)
    return out


# --- w12r additions: two NON-cell carriers, so verify_tables.py never mis-asserts them ----
def derived(values, froms, formula, fmt="{:,.0f}", note=None):
    """A number COMPUTED from registered pointers.  Deliberately NOT shaped like a `cell`
    (it carries `derived`/`from`, never `values`+`provenance`), because verify_tables.py
    asserts a cell equals the raw field at its pointer and a derived number does not."""
    return {"derived": values, "from": froms, "formula": formula, "format": fmt,
            "machine_checked_by_verify_tables": False,
            "note": note or "DERIVED from the pointers in `from` by `formula` - recompute to check."}


def dsrc(sid, pointer):
    return {"source": sid, "path": SOURCES[sid], "pointer": pointer}


def locator(values, sid, lines, what, fmt="{:,.0f}"):
    """A number that exists ONLY in a markdown artefact.  Carries its registered path and the
    line numbers that hold it, and declares that verify_tables.py cannot check it."""
    return {"printed": values, "source": sid, "path": SOURCES[sid], "lines": lines,
            "what": what, "format": fmt, "machine_checked_by_verify_tables": False}


def fmt_any(c):
    """Render a cell, a derived number or a markdown locator, whichever it is."""
    if "values" in c and "provenance" in c:
        if "mean" in c:
            return fmt_cell(c)
        f = c.get("format", "{:,.0f}")
        return " / ".join(f.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                          else str(v) for v in c["values"])
    f = c.get("format", "{:,.0f}")
    vals = c.get("derived", c.get("printed"))
    if isinstance(vals, (int, float, str, bool)):
        vals = [vals]
    return " / ".join(f.format(v) if isinstance(v, (int, float)) else str(v) for v in vals)


def fmt_cell(c, mean_only=False):
    f = c.get("format", "{:.4f}")
    vals = c["values"]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        if mean_only:
            return f.format(c["mean"])
        return f.format(c["mean"]) + " (" + " / ".join(f.format(v) for v in vals) + ")"
    return " / ".join(str(v) for v in vals)


# --------------------------------------------------------------------------------------
# 2.  Load everything once.
# --------------------------------------------------------------------------------------
DOCS = {}
for sid, p in SOURCES.items():
    if p.endswith(".json"):
        DOCS[sid] = load_json(sid)

MAZE = ["maze_s0", "maze_s1", "maze_s2"]
HARN = ["harness_s0", "harness_s1", "harness_s2"]

TABLES = {}
FIGURES = {}
NOT_RUN = []
CORRECTIONS = []

# --------------------------------------------------------------------------------------
# 3.  TABLE 1 -- THE MAIN SCOREBOARD.  CMAZE-2, hub/maze, TEST, 3/3 scored seeds.
#     w12r/MF10: this table is CMAZE-2, a REQUEST LAYER over CMAZE-1, and never CMAZE-1.
# --------------------------------------------------------------------------------------
m0 = DOCS["maze_s0"]
T1_PINS = {
    "cell": "CMAZE-2 (a REQUEST LAYER over CMAZE-1: engine, solver and exact integer judge imported unmodified from `w8_envscout.maze_colour`), hub/maze",
    "environment": "CMAZE-2",
    "environment_is_NOT": "CMAZE-1 - see TABLES.json /environments; T3 is CMAZE-1 proper",
    "condition": "test (held out)",
    "generation": 2,
    "seeds": [0, 1, 2],
    "scored_runs": "3 of 3 (S1/S2/S3, pre-registered before execution)",
    "K": ptr(m0, "/K"),
    "n_atoms": ptr(m0, "/n_atoms"),
    "ridge": ptr(m0, "/ridge"),
    "ae_steps": ptr(m0, "/ae_steps"),
    "baseline_steps": ptr(m0, "/bl_steps"),
    "latent_d": ptr(m0, "/latent_d"),
    "read_floats": ptr(m0, "/ledger/read_floats"),
    "deployed_bytes": ptr(m0, "/ledger/deployed_bytes"),
    "statistic_bytes_float64_declared_beside": ptr(m0, "/ledger/statistic_bytes"),
    "rank_over_width": ptr(m0, "/ledger/rank_over_width/0"),
    "matched_params": {k: ptr(m0, f"/baseline_params/{k}/params") for k in ("mlp", "lstm", "transformer")},
    "n_mazes": ptr(m0, "/n_mazes"),
}
T1_PIN_SRC = [{"source": "maze_s0", "path": SOURCES["maze_s0"], "pointer": p}
              for p in ("/K", "/n_atoms", "/ridge", "/ae_steps", "/bl_steps", "/ledger", "/baseline_params")]

ROW_KEYS = [
    ("seen", "row 1 - seen request", "RESULT"),
    ("unseen_pair", "row 2 - unseen agent-target pair, seen maze", "RESULT"),
    ("nway", "row 3 - n-way (CONJUNCTION)", "RESULT"),
    ("nway_route", "row 3 - n-way (PER ROUTE)", "RESULT"),
    ("budget_interp", "row 4 - budget interpolation", "RESULT"),
]
ARM_ORDER = [
    ("clu", "CLU (the arm)", "arm"),
    ("mlp", "MLP (matched params)", "baseline"),
    ("lstm", "LSTM (matched params)", "baseline"),
    ("transformer", "Transformer (matched params)", "baseline"),
    ("lookup_matched", "lookup, matched bytes (C.1 floor)", "ablation"),
    ("clu_lowrank", "clu_lowrank (n_atoms=8)", "ablation"),
    ("clu_nopair", "clu_nopair (no pair term)", "ablation"),
    ("landscape_shuffled", "landscape_shuffled (C.2)", "ablation"),
    ("random_atoms", "random_atoms (C.2)", "ablation"),
    ("landscape_removed", "landscape_removed (C.2)", "ablation"),
]


def maze_cell(row_key, arm):
    if row_key == "nway_route":
        return cell(MAZE, f"/table/nway/{arm}/per_route/validity", DOCS)
    return cell(MAZE, f"/table/{row_key}/{arm}/validity", DOCS)


t1_rows = []
for rk, rlabel, rclass in ROW_KEYS:
    base = "nway" if rk == "nway_route" else rk
    n_ptr = "/table/nway/clu/per_route/n" if rk == "nway_route" else f"/table/{base}/clu/n"
    row = {
        "label": rlabel,
        "row_class": rclass,
        "n_scored": cell(MAZE, n_ptr, DOCS, fmt="{:.0f}"),
        "cells": {label: maze_cell(rk, arm) for arm, label, _ in ARM_ORDER},
        "ceilings": {
            "representation": cell(MAZE, f"/representation_ceiling/{base}/validity", DOCS),
            "pipeline (encode->decode->execute->judge)": cell(MAZE, f"/pipeline_ceiling_rows/{base}/validity", DOCS),
            "store span": cell(MAZE, f"/span_ceiling_rows/{base}/validity", DOCS),
            "_ceiling_n": cell(MAZE, f"/pipeline_ceiling_rows/{base}/n", DOCS, fmt="{:.0f}"),
        },
    }
    if base in ("unseen_pair", "nway"):
        row["caveat_columns_measured_on_THIS_split"] = {
            "BLIND-SEQUENTIAL": cell(MAZE, f"/env_columns/{base}/blind/BLIND-SEQUENTIAL(key order from A)/validity", DOCS),
            "BLIND-FACTOR": cell(MAZE, f"/env_columns/{base}/blind/BLIND-FACTOR    (A x B local only)/validity", DOCS),
            "BLIND-A": cell(MAZE, f"/env_columns/{base}/blind/BLIND-A         (rule B dropped)/validity", DOCS),
            "BLIND-COLOUR": cell(MAZE, f"/env_columns/{base}/blind/BLIND-COLOUR    (colours collapsed)/validity", DOCS),
            "BLIND-B": cell(MAZE, f"/env_columns/{base}/blind/BLIND-B         (rule A dropped)/validity", DOCS),
            "ORACLE": cell(MAZE, f"/env_columns/{base}/blind/ORACLE          (full composition)/validity", DOCS),
            "RANDOM": cell(MAZE, f"/env_columns/{base}/random/validity", DOCS),
            "rule B bends the route": cell(MAZE, f"/env_columns/{base}/bend/bend_fraction", DOCS, fmt="{:.3f}"),
        }
    t1_rows.append(row)

# DISPLAY rows -- refusal, measured on held-out mazes 48..95 where the supervised refuser is scored.
DISPLAY_CONDS = [
    ("budget_infeasible", "budget below the shortest legal route"),
    ("walled", "target walled off"),
    ("mt_unseen", "unseen (agent,target) novelty control"),
    ("grounded_possible", "grounded-possible  (FALSE refusals - lower is better)"),
]
t1_display = []
for ck, clabel in DISPLAY_CONDS:
    t1_display.append({
        "label": f"refusal - {clabel}",
        "row_class": "DISPLAY",
        "cells": {
            "CLU refusal rate (all mazes)": cell(MAZE, f"/display/refusal/{ck}/refusal", DOCS),
            "CLU refusal rate (held-out mazes 48..95)": cell(MAZE, f"/display/clu_refusal_rate_heldout_mazes/{ck}/refusal", DOCS),
            "supervised refuser (the only comparator)": cell(MAZE, f"/display/supervised_refuser/refusal_rate/{ck}", DOCS),
            "CLU validity when it acts (all mazes)": cell(MAZE, f"/display/refusal/{ck}/validity", DOCS),
        },
        "n_scored": cell(MAZE, f"/display/refusal/{ck}/n", DOCS, fmt="{:.0f}"),
    })
t1_display.append({
    "label": "post-deletion refusal - delete the green target (CONTENT deletion)",
    "row_class": "DISPLAY",
    "cells": {
        "refusal on green requests BEFORE": cell(MAZE, "/display/delete_green/green_before/refusal", DOCS),
        "refusal on green requests AFTER": cell(MAZE, "/display/delete_green/green_after/refusal", DOCS),
        "validity on green requests AFTER": cell(MAZE, "/display/delete_green/green_after/validity", DOCS),
    },
    "n_scored": cell(MAZE, "/display/delete_green/green_before/n", DOCS, fmt="{:.0f}"),
})
t1_display.append({
    "label": "post-deletion refusal - delete the move-left action (INTERFACE deletion, no atom touched)",
    "row_class": "DISPLAY",
    "cells": {
        "refusal BEFORE (grounded-possible)": cell(MAZE, "/display/delete_move_left/before/refusal", DOCS),
        "refusal AFTER, routes that NEED move-left": cell(MAZE, "/display/delete_move_left/after_needs_W/refusal", DOCS),
        "validity AFTER, routes that NEED move-left": cell(MAZE, "/display/delete_move_left/after_needs_W/validity", DOCS),
        "refusal AFTER, routes that do NOT need it": cell(MAZE, "/display/delete_move_left/after_no_W/refusal", DOCS),
    },
    "n_scored": cell(MAZE, "/display/delete_move_left/after_needs_W/n", DOCS, fmt="{:.0f}"),
})
t1_display.append({
    "label": "post-deletion composition WITHOUT retraining - surviving targets (as pre-registered)",
    "row_class": "DISPLAY",
    "cells": {
        "validity on surviving targets BEFORE": cell(MAZE, "/display/delete_green/other_before/validity", DOCS),
        "validity on surviving targets AFTER": cell(MAZE, "/display/delete_green/other_after/validity", DOCS),
        "delta": cell(MAZE, "/display/delete_green/composition_without_retraining/validity_delta_on_other_targets", DOCS, fmt="{:+.4f}"),
    },
    "n_scored": cell(MAZE, "/display/delete_green/other_before/n", DOCS, fmt="{:.0f}"),
})


# --------------------------------------------------------------------------------------
# 3b. w12r/MF3 -- THE FOUR-PART BYTE LEDGER FOR T1's STORE, PRINTED, NEVER NETTED.
#     Referee MF3: "Table 1 (the lead) prints no byte row."  It does now.
# --------------------------------------------------------------------------------------
T1_BYTE_LEDGER = {
    "rule": "FOUR PARTS, PRINTED SEPARATELY, NEVER NETTED INTO ONE NUMBER.",
    "dtype_note": "the deployed object is float32 (4 B/float); the sufficient statistic is float64 (8 B/float)",
    "parts": {
        "1. atoms (the store)": derived(
            [ptr(DOCS[sid], "/ledger/store_atom_floats") * 4 for sid in MAZE],
            [dsrc(sid, "/ledger/store_atom_floats") for sid in MAZE],
            "store_atom_floats x 4 (float32)"),
        "2. sufficient statistic (covariance / count ledger), float64": cell(
            MAZE, "/ledger/statistic_bytes", DOCS, fmt="{:,.0f}"),
        "3. address book (the head)": derived(
            [ptr(DOCS[sid], "/ledger/head_floats") * 4 for sid in MAZE],
            [dsrc(sid, "/ledger/head_floats") for sid in MAZE],
            "head_floats x 4 (float32)"),
        "4. coupling / hole table": {
            "status": "DECLARED NOT PRESENT - not a zero and not a blank",
            "why": ("no coupling table and no hole table exists in the w9/maze artefacts: the three "
                    "scored run files carry no field matching 'hole' or 'coupl'. The w9 leg's "
                    "deletion is a downdate of (S, t, n) and of the head statistics (A, Mx); it has "
                    "no interface hole table of the kind the w8 leg carries."),
            "checked_in": [SOURCES[sid] for sid in MAZE],
            "machine_checked_by_verify_tables": False,
        },
    },
    "totals": {
        "deployed (atoms + address book) - what produces a read": cell(
            MAZE, "/ledger/deployed_bytes", DOCS, fmt="{:,.0f}"),
        "honest total (deployed + statistic)": derived(
            [ptr(DOCS[sid], "/ledger/deployed_bytes") + ptr(DOCS[sid], "/ledger/statistic_bytes")
             for sid in MAZE],
            [dsrc(sid, "/ledger/deployed_bytes") for sid in MAZE]
            + [dsrc(sid, "/ledger/statistic_bytes") for sid in MAZE],
            "deployed_bytes + statistic_bytes"),
        "statistic as a share of deployed": derived(
            [ptr(DOCS[sid], "/ledger/statistic_bytes") / ptr(DOCS[sid], "/ledger/deployed_bytes")
             for sid in MAZE],
            [dsrc(sid, "/ledger/statistic_bytes") for sid in MAZE],
            "statistic_bytes / deployed_bytes", fmt="{:.2%}"),
    },
    "pin_status": (
        "T1 IS NOT UNDER THE 524,288 B PIN. That pin belongs to the Shapes3D arms (T2, T4's w8 leg). "
        "T1's budget is the MATCHED-PARAMETER budget: mlp 410,920 / lstm 409,846 / transformer 405,776 "
        "against the CLU's read of 410,688 floats. Do not print a `x the pin` figure for this table."
    ),
    "the_sentence_the_artefact_itself_writes": {
        "text": ("deployed bytes are the atoms plus the address book, which is what produces a read; "
                 "the float64 sufficient statistic is what buys EXACT deletion and is declared beside "
                 "them, never netted in"),
        "provenance": [dsrc("maze_s0", "/ledger/note")],
    },
}

# --------------------------------------------------------------------------------------
# 3c. w12r/MF11 -- FRAGMENT-LOOKUP: same-maze and cross-maze are DIFFERENT QUANTITIES.
#     Referee MF11: "0.0000 everywhere" is false on w9's own split.  Both are printed.
# --------------------------------------------------------------------------------------
T1_FRAGMENT = {
    "why_this_block_exists": (
        "'FRAGMENT-LOOKUP r=1/r=2 at 0.0000 everywhere' is TRUE of the CROSS-MAZE quantity and FALSE "
        "of the SAME-MAZE quantity on this hub's own split. Both are printed here, labelled, so no "
        "sentence can quote one as the other."
    ),
    "rows": {
        "FRAGMENT-LOOKUP, CROSS-MAZE (the published CMAZE-1 tripwire; the split does NOT leak)": {
            "r=1": cell(MAZE, "/env_columns/fragment_cross_maze/FRAG-cross-r1/validity", DOCS),
            "r=2": cell(MAZE, "/env_columns/fragment_cross_maze/FRAG-cross-r2/validity", DOCS),
        },
        "FRAGMENT-LOOKUP, SAME-MAZE (a DIFFERENT quantity, and NOT a like-for-like arm)": {
            "r=1": cell(MAZE, "/env_columns/fragment/FRAG-r1/validity", DOCS),
            "r=2": cell(MAZE, "/env_columns/fragment/FRAG-r2/validity", DOCS),
        },
    },
    "not_like_for_like": (
        "FRAG reads the TRUE MAZE at every step, which charter B.3 forbids the CLU. The same-maze "
        "table is therefore a floor on what a stored-route table achieves when it may look, not a "
        "comparable arm."
    ),
    "why_same-maze_is_high": (
        "this hub's rows are unseen PAIRS on SEEN mazes, so a fragment table contains routes through "
        "the very same maze, and a 5x5 colour-canonicalised patch is close to a position index in a "
        "15x15 grid"
    ),
    "which_mean": (
        "the three per-seed r=2 validities are 0.5342 / 0.5364 / 0.4636 at n = 584 / 604 / 632. This "
        "file prints the UNWEIGHTED seed mean, 0.5114. The 0.5105 quoted by `w9-maze/LOG.md` and by "
        "REFEREE-v0.md is the n-WEIGHTED (pooled) mean, 0.51044. See correction C11."
    ),
    "the_number_the_draft_left_on_the_floor": (
        "same-maze r=2 reaches 0.5105 against row 2's pipeline ceiling of 0.6319 - 81% of the "
        "pipeline ceiling, where the CLU reaches 20% and the best learned baseline 29%. It is the "
        "strongest single exhibit for 'every piece of every answer is already written'."
    ),
    "provenance_of_the_caveat_text": [
        {"source": "maze_log", "path": SOURCES["maze_log"], "locator": "LOG.md:404-409,587-588"},
    ],
}

# --------------------------------------------------------------------------------------
# 3d. w12r/MF14 -- the charter rule "every composition row carries BLIND-SEQUENTIAL on its
#     own split" is UNSATISFIABLE for row 4.  Declared, not left blank.
# --------------------------------------------------------------------------------------
T1_DECLARED_NOT_MEASURED = [
    {
        "row": "row 4 - budget interpolation",
        "column": "BLIND-SEQUENTIAL (and every other BLIND column)",
        "status": "DECLARED NOT MEASURED - not a blank, not a zero",
        "why": ("`/env_columns` carries exactly four keys - `unseen_pair`, `nway`, `fragment`, "
                "`fragment_cross_maze` - on every scored seed. There is no blind block for the "
                "budget-interpolation row on any seed, so the charter rule is unsatisfiable for "
                "row 4 and is declared rather than silently skipped."),
        "checked_in": [SOURCES[sid] for sid in MAZE],
        "pointer_checked": "/env_columns",
        "provenance": [dsrc(sid, "/env_columns") for sid in MAZE],
    },
]

T1_COLUMN_CLASSES = {
    "CLU (the arm)": {"class": "arm", "what": "the CLU read: atoms + request-addressed address book"},
    "MLP (matched params)": {"class": "RESULT baseline", "what": "learned baseline, 410,920 params"},
    "LSTM (matched params)": {"class": "RESULT baseline", "what": "learned baseline, 409,846 params"},
    "Transformer (matched params)": {"class": "RESULT baseline", "what": "learned baseline, 405,776 params"},
    "lookup, matched bytes (C.1 floor)": {"class": "ablation (charter C.1 floor)",
        "what": "a printed floor: the best a table of stored routes can do at the arm's byte budget"},
    "clu_lowrank (n_atoms=8)": {"class": "ablation",
        "what": "the same read with 8 atoms per well instead of 31 - the saturation-ratio lever"},
    "clu_nopair (no pair term)": {"class": "ablation", "what": "the read with the pair interaction removed"},
    "landscape_shuffled (C.2)": {"class": "ablation (charter C.2 control)",
        "what": "atoms permuted across wells - if the row survives, the store is decorative"},
    "random_atoms (C.2)": {"class": "ablation (charter C.2 control)",
        "what": "atoms replaced by random directions at the same rank"},
    "landscape_removed (C.2)": {"class": "ablation (charter C.2 control)",
        "what": "atoms zeroed - the collapse control"},
}

TABLES["T1"] = {
    "title": "MAIN SCOREBOARD - CMAZE-2 (w9 hub/maze), held-out TEST, 3 of 3 scored seeds",
    "environment": "CMAZE-2",
    "column_classes": T1_COLUMN_CLASSES,
    "instrument": "exact integer judge, no tolerance anywhere; validity under the exact judge",
    "pins": T1_PINS,
    "pin_provenance": T1_PIN_SRC,
    "caption": (
        "THE ENVIRONMENT IS CMAZE-2: a REQUEST LAYER over CMAZE-1, with the engine, the solver and the "
        "exact integer judge imported unmodified from `w8_envscout.maze_colour`. It is NOT CMAZE-1, and it "
        "must never share a table with T3, which is CMAZE-1 proper - a different environment, not merely a "
        "different budget. "
        "Every cell is the mean of scored seeds 0/1/2 with the three per-seed values in brackets. "
        "Pins: K=8, n_atoms=31, ridge=0.01, AE 12,000 steps, baselines 4,000 steps, read = 410,688 floats "
        "(1,642,752 B deployed) with the float64 sufficient statistic (35,904 B) declared beside it and never "
        "netted in; matched parameters mlp 410,920 / lstm 409,846 / transformer 405,776. "
        "The four-part byte ledger is printed as its own block below and is never netted into one number. "
        "Row 3 is reported BOTH as the scored CONJUNCTION and PER ROUTE, because the ceilings are per route."
    ),
    "result_rows": t1_rows,
    "display_rows": t1_display,
    "byte_ledger": T1_BYTE_LEDGER,
    "fragment_columns": T1_FRAGMENT,
    "declared_not_measured": T1_DECLARED_NOT_MEASURED,
}

# --------------------------------------------------------------------------------------
# 4.  TABLE 2 -- Shapes3D generation cell, w8/encoder-loop instrument, 3 seeds.
#     SEPARATE TABLE: different compute pin AND a shortened decoder recipe.
# --------------------------------------------------------------------------------------
ca = DOCS["enc_crossarm"]
er = DOCS["enc_report"]
T2_PINS = {
    "cell": "Shapes3D generation cell, w8 hub/encoder-loop",
    "condition": ptr(ca, "/condition"),
    "generation": 2,
    "seeds": ptr(ca, "/seeds"),
    "scored_runs": "1 of 3 (2 returned unspent)",
    "decoder_recipe": ptr(er, "/pins/dec_recipe"),
    "joint_steps": ptr(er, "/pins/joint_steps"),
    "n_train": ptr(er, "/pins/n_train"),
    "dec_refit_steps": ptr(er, "/pins/dec_refit_steps"),
    "warm_steps": ptr(er, "/pins/warm_steps"),
    "k_wells": ptr(er, "/pins/k_wells"),
    "n_modes": ptr(er, "/pins/n_modes"),
    "metric": ptr(ca, "/metric"),
    "margin_declared_in_advance": ptr(ca, "/margin_declared_in_advance"),
}
T2_PIN_SRC = [
    {"source": "enc_report", "path": SOURCES["enc_report"], "pointer": "/pins"},
    {"source": "enc_crossarm", "path": SOURCES["enc_crossarm"], "pointer": "/metric"},
]
ENC = ["enc_crossarm"] * 3


def enc_cell(pointer_tmpl, fmt="{:.3f}"):
    """CROSS-ARM.json holds all three seeds in one file: build the per-seed pointers by hand."""
    vals, prov = [], []
    for s in SEEDS:
        p = pointer_tmpl.format(seed=s)
        vals.append(ptr(ca, p))
        prov.append({"source": "enc_crossarm", "path": SOURCES["enc_crossarm"], "pointer": p})
    return {"seeds": list(SEEDS), "values": vals, "provenance": prov, "format": fmt,
            "mean": sum(vals) / 3, "min": min(vals), "max": max(vals)}


t2_rows = []
for rk, rlabel in (("r3_comp_2way", "row 3 - 2-way composition"), ("r4_comp_nway", "row 4 - n-way composition")):
    t2_rows.append({
        "label": rlabel,
        "row_class": "RESULT",
        "cells": {
            "pca_tensor @ encoder ALONE": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_Aprime_loop_nosw_minus_alone/control_mean"),
            "pca_tensor @ loop_nosw (unfrozen, NO swap loss)": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_Aprime_loop_nosw_minus_alone/arm_mean"),
            "pca_tensor @ loop_swap (THE ARM)": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_A_loop_swap_minus_alone/arm_mean"),
            "A' = loop_nosw - alone": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_Aprime_loop_nosw_minus_alone/mean_paired_delta", "{:+.3f}"),
            "A'' = loop_swap - loop_nosw (what the SWAP LOSS bought)": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_Adprime_swap_loss_alone/mean_paired_delta", "{:+.3f}"),
            "A = loop_swap - alone (the NET)": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_A_loop_swap_minus_alone/mean_paired_delta", "{:+.3f}"),
            "A: bootstrap CI95 low": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_A_loop_swap_minus_alone/bootstrap_ci95/0", "{:+.3f}"),
            "A: n corners (paired)": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_A_loop_swap_minus_alone/n_corners", "{:.0f}"),
            "A'': sign-test p (two-sided)": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_Adprime_swap_loss_alone/sign_test_p_two_sided", "{:.4g}"),
            "A': sign-test p (two-sided)": enc_cell("/per_seed/{seed}/" + rk + "/encoder_comparison/BAR_Aprime_loop_nosw_minus_alone/sign_test_p_two_sided", "{:.4g}"),
        },
    })
    t2_rows.append({
        "label": rlabel + " - ablation: matched-parameter learner inside each encoder arm",
        "row_class": "ablation",
        "cells": {
            "rival_mlp_180k @ alone": enc_cell("/per_seed/{seed}/" + rk + "/baseline_within_arm/alone/control_mean"),
            "rival_mlp_180k @ loop_swap": enc_cell("/per_seed/{seed}/" + rk + "/baseline_within_arm/loop_swap/control_mean"),
            "CLU margin over rival_mlp_180k @ alone": enc_cell("/per_seed/{seed}/" + rk + "/BAR_D_does_the_baseline_close_the_gap/clu_margin_over_mlp180k_at_alone", "{:+.3f}"),
            "CLU margin over rival_mlp_180k @ loop_swap": enc_cell("/per_seed/{seed}/" + rk + "/BAR_D_does_the_baseline_close_the_gap/clu_margin_over_mlp180k_at_loop_swap", "{:+.3f}"),
        },
        "note_in_artefact": ptr(ca, "/per_seed/0/" + rk + "/BAR_D_does_the_baseline_close_the_gap/verdict"),
    })

# --- w12r/MF3: the four-part byte ledger for T2's store -------------------------------
T2_ARM_KEYS = ["seed0:loop_swap", "seed1:loop_swap", "seed2:loop_swap"]
T2_BYTE_LEDGER = {
    "rule": "FOUR PARTS, PRINTED SEPARATELY, NEVER NETTED INTO ONE NUMBER.",
    "arm_rows_read": T2_ARM_KEYS,
    "parts": {
        "1. atoms (the store - 'the modes ARE the store')": {
            "seeds": list(SEEDS),
            "values": [ptr(er, f"/rows/{k}/store_bytes/store_bytes") for k in T2_ARM_KEYS],
            "provenance": [{"source": "enc_report", "path": SOURCES["enc_report"],
                            "pointer": f"/rows/{k}/store_bytes/store_bytes"} for k in T2_ARM_KEYS],
            "format": "{:,.0f}",
        },
        "2. sufficient statistic (covariance / count ledger)": {
            "status": "DECLARED NOT PRESENT - not a zero and not a blank",
            "why": ("this instrument retains no per-well covariance and makes NO deletion claim. "
                    "The exact-deletion claim lives in T4 and T5, not here."),
            "machine_checked_by_verify_tables": False,
        },
        "3. address book": locator(
            [195064, 195064, 195064], "enc_log", "LOG.md:702,871",
            "byte-ledger assertion, per arm per seed: store 524,288 B exact, address book 195,064 B "
            "declared BESIDE it and never netted in"),
        "4. coupling / hole table": {
            "status": "DECLARED NOT PRESENT",
            "why": "no coupling or hole table exists on this instrument; there is no deletion interface here.",
            "machine_checked_by_verify_tables": False,
        },
    },
    "totals": {
        "honest total (atoms + address book)": derived(
            [ptr(er, f"/rows/{k}/store_bytes/store_bytes") + 195064 for k in T2_ARM_KEYS],
            [{"source": "enc_report", "path": SOURCES["enc_report"],
              "pointer": f"/rows/{k}/store_bytes/store_bytes"} for k in T2_ARM_KEYS]
            + [{"source": "enc_log", "path": SOURCES["enc_log"], "locator": "LOG.md:702,871"}],
            "store_bytes + address_book_bytes(195,064, markdown locator)"),
        "x the 524,288 B pin, store only": derived(
            [ptr(er, f"/rows/{k}/store_bytes/store_bytes") / 524288 for k in T2_ARM_KEYS],
            [{"source": "enc_report", "path": SOURCES["enc_report"],
              "pointer": f"/rows/{k}/store_bytes/store_bytes"} for k in T2_ARM_KEYS],
            "store_bytes / 524288", fmt="{:.4f}x"),
    },
    "the_sentence_the_artefact_itself_writes": {
        "text": ("The modes ARE the store. The addressing head is reported separately and is not "
                 "part of this count; a byte-matched baseline matches store_bytes."),
        "provenance": [{"source": "enc_report", "path": SOURCES["enc_report"],
                        "pointer": "/rows/seed0:loop_swap/store_bytes/note"}],
    },
}

TABLES["T2"] = {
    "title": "Shapes3D generation cell - w8 hub/encoder-loop instrument, 3 seeds",
    "instrument": ptr(ca, "/metric") + "; external image classifier, identical across arms",
    "pins": T2_PINS,
    "pin_provenance": T2_PIN_SRC,
    "caption": (
        "SEPARATE TABLE BY CHARTER. These absolutes were produced by the shortened `w8_v1_short` decoder at "
        "joint_steps=400 / n_train=8192 and are NOT comparable with the incumbent's row-3/row-4 values "
        "(a shortened decoder recipe is a different instrument, not a different setting). The metric is a COUNT "
        "of distinct valid novel factor tuples per corner, not a validity rate: row 3's ceiling is 36 tuples/corner "
        "and row 4's is 72, so the two rows are not on one absolute scale either."
    ),
    "rows": t2_rows,
    "byte_ledger": T2_BYTE_LEDGER,
    "verdict": {
        "BAR_A_PASS": {"r3": ptr(ca, "/VERDICT/r3_comp_2way/BAR_A_PASS"), "r4": ptr(ca, "/VERDICT/r4_comp_nway/BAR_A_PASS")},
        "BAR_B_FALSIFIED": {"r3": ptr(ca, "/VERDICT/r3_comp_2way/BAR_B_FALSIFIED"), "r4": ptr(ca, "/VERDICT/r4_comp_nway/BAR_B_FALSIFIED")},
        "provenance": [{"source": "enc_crossarm", "path": SOURCES["enc_crossarm"], "pointer": "/VERDICT"}],
    },
}

# --------------------------------------------------------------------------------------
# 5.  TABLE 3 -- CMAZE-1 PROPER at the w9/harness tier.  DIFFERENT PIN: 60k params, 500/200.
#     w12r/MF10: DIFFERENT ENVIRONMENT from T1, not merely a different budget.
# --------------------------------------------------------------------------------------
h0 = DOCS["harness_s0"]
T3_PINS = {
    "cell": "CMAZE-1 PROPER (env-scout adopted defaults), w9 harness tier `cmaze1`",
    "environment": "CMAZE-1",
    "condition": "eval split, 200 instances/seed",
    "generation": 2,
    "seeds": [0, 1, 2],
    "n_train": ptr(h0, "/config/n_train"),
    "n_eval": ptr(h0, "/config/n_eval"),
    "enc_steps": ptr(h0, "/config/enc_steps"),
    "bc_steps": ptr(h0, "/config/bc_steps"),
    "target_params": ptr(h0, "/config/target_params"),
    "matched_params": {"mlp": 60047, "lstm": 59965, "transformer": 59819},
    "judge_exact": ptr(h0, "/judge_audit/exact"),
}
T3_PIN_SRC = [{"source": "harness_s0", "path": SOURCES["harness_s0"], "pointer": p}
              for p in ("/config", "/judge_audit/exact", "/tier")]

T3_ARMS = [
    ("results/mlp", "MLP (matched ~60k)", "baseline"),
    ("results/lstm", "LSTM (matched ~60k)", "baseline"),
    ("results/transformer", "Transformer (matched ~60k)", "baseline"),
    ("results/table", "Table floor (raw state)", "ablation"),
    ("results/table_enc", "Table-enc (encoder bits)", "ablation"),
    ("columns/oracle (full composition)", "ORACLE (full composition)", "reference"),
    ("columns/blind-A (rule B dropped)", "BLIND-A", "reference"),
    ("columns/blind-B (rule A dropped)", "BLIND-B", "reference"),
    ("columns/blind-colour (colours collapsed)", "BLIND-COLOUR", "reference"),
    ("columns/blind-factor (A x B local only)", "BLIND-FACTOR", "reference"),
    ("columns/blind-sequential (key order from A)", "BLIND-SEQUENTIAL", "reference"),
    ("columns/fragment-lookup r=1", "FRAGMENT-LOOKUP r=1 (CMAZE-1 tier column)", "reference"),
    ("columns/fragment-lookup r=2", "FRAGMENT-LOOKUP r=2 (CMAZE-1 tier column)", "reference"),
]
t3_rows = []
for pth, label, rclass in T3_ARMS:
    t3_rows.append({
        "label": label,
        "row_class": rclass,
        "cells": {
            "validity (exact judge)": cell(HARN, f"/{pth}/validity", DOCS),
            "score": cell(HARN, f"/{pth}/score", DOCS),
            "goal rate, budget lifted": cell(HARN, f"/{pth}/goal_rate_budget_lifted", DOCS),
            "legal prefix / optimal": cell(HARN, f"/{pth}/mean_legal_prefix_over_opt", DOCS),
        },
        "n_scored": cell(HARN, f"/{pth}/n", DOCS, fmt="{:.0f}"),
    })
t3_display = [{
    "label": "supervised refuser (the only DISPLAY comparator), decision at t=0",
    "row_class": "DISPLAY",
    "cells": {
        "refusal on IMPOSSIBLE (correct)": cell(HARN, "/results/refuser_impossible_decision_at_t=0/refusal_rate", DOCS),
        "refusal on POSSIBLE (false)": cell(HARN, "/results/refuser_possible_decision_at_t=0/refusal_rate", DOCS),
    },
}]
# --- w12r/MF3: T3 carries no store.  Declared, not omitted. ---------------------------
T3_BYTE_LEDGER = {
    "status": "NOT APPLICABLE - THIS TABLE CARRIES NO STORE. Declared, not omitted.",
    "why": ("every arm at this tier is a learned parametric baseline matched at ~60,000 parameters "
            "(mlp 60,047 / lstm 59,965 / transformer 59,819) or one of the environment's own "
            "reference columns. There is no atom block, no sufficient statistic, no address book "
            "and no coupling table, so there is nothing to print a four-part ledger for."),
    "no_row_here_is_under_the_524288_B_pin": True,
    "provenance": [{"source": "harness_s0", "path": SOURCES["harness_s0"], "pointer": "/matched_widths"},
                   {"source": "harness_s0", "path": SOURCES["harness_s0"], "pointer": "/config/target_params"}],
}

TABLES["T3"] = {
    "title": "CMAZE-1 PROPER at the w9/harness tier - learned baselines and the environment's own columns, 3 seeds",
    "environment": "CMAZE-1",
    "instrument": "w9_harness.judge_maze.JUDGE, exact integer judge, no tolerance",
    "pins": T3_PINS,
    "pin_provenance": T3_PIN_SRC,
    "caption": (
        "SEPARATE TABLE BY CHARTER, AND ON TWO GROUNDS, NOT ONE. (i) ENVIRONMENT: this table is CMAZE-1 "
        "PROPER; T1 is CMAZE-2, a REQUEST LAYER over it. (ii) BUDGET: this tier trains ~60k-parameter arms "
        "on 500 instances and evaluates 200, while T1's arms are matched at ~410k parameters over 96 mazes / "
        "1,208 requests. w12r/MF10: the earlier wording of this caption said `the same environment`, which "
        "was wrong - T1 and T3 are a DIFFERENT ENVIRONMENT, not merely a different budget, and they must "
        "never share a table."
    ),
    "rows": t3_rows,
    "display_rows": t3_display,
    "byte_ledger": T3_BYTE_LEDGER,
}

# --------------------------------------------------------------------------------------
# 6.  TABLE 4 -- exact deletion, two independent implementations, 3 seeds each.
# --------------------------------------------------------------------------------------
dd = DOCS["downdate_test"]
t4_rows = [
    {
        "label": "w9/maze `CluRead.delete_rows_exact` - delete the green target, TEST leg (CMAZE-2)",
        "row_class": "RESULT",
        "row_class_w12r_note": ("w12r/MF7: this leg's RESULT label is CORRECT - it sits inside the "
                                "pre-registered scored runs S1/S2/S3, 3 of 3 spent."),
        "scored_runs": 3,
        "environment": "CMAZE-2",
        "implementation": "w9 hub/maze",
        "cells": {
            "statistic max relative residual (downdate vs refit)": cell(MAZE, "/display/delete_green_exactness/statistic_max_rel_residual", DOCS, fmt="{:.3e}"),
            "subspace max projector residual": cell(MAZE, "/display/delete_green_exactness/subspace_max_projector_residual", DOCS, fmt="{:.3e}"),
            "rows removed": cell(MAZE, "/display/delete_green_exactness/n_rows_removed", DOCS, fmt="{:.0f}"),
            "rows kept": cell(MAZE, "/display/delete_green_exactness/n_rows_kept", DOCS, fmt="{:.0f}"),
            "refit performed": cell(MAZE, "/display/delete_green_exactness/refit", DOCS, fmt="{}"),
            "passes over retained data": cell(MAZE, "/display/delete_green_exactness/passes_over_retained_data", DOCS, fmt="{:.0f}"),
            "items kept": cell(MAZE, "/display/delete_green_exactness/items_kept", DOCS, fmt="{}"),
            "address-book bytes deployed by the basis-change carry": cell(MAZE, "/display/delete_green_exactness/receipt/basis_change_bytes_deployed", DOCS, fmt="{:.0f}"),
        },
    },
    {
        "label": "w8/downdate `exp_w8_downdate` - covariance downdate == refit-without-those-rows, TEST leg",
        "row_class": "ASSERTION BATTERY (object property) - 3 seeds - TEST condition - 0 SCORED RUNS",
        "row_class_w12r_note": (
            "w12r/MF7. This row was labelled RESULT. Its own artefact says otherwise, verbatim: "
            "`/note` = 'No scored run and no claim: these are properties of the object (deviations, "
            "byte counts, row counts), not outcomes. hub/single-design scores the object.' The hub "
            "commissioned to score it - `hub/single-design` - scored 0 of 3 and is DECLARED NOT-RUN "
            "NR3, with its full budget intact. Nobody scored this leg. The artefact was NOT relabelled "
            "to match the table; the table was relabelled to match the artefact."),
        "row_class_provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/note"}],
        "scored_runs": 0,
        "implementation": "w8 spoke/downdate-build (different code, different author)",
        "cells": {
            "max abs deviation, worst well AND worst deletion (pinned 11,024 rows + random 64) - THE SPOKE'S OWN ROW": {
                "seeds": list(SEEDS),
                "values": [ptr(dd, f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev") for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev"} for s in SEEDS],
                "format": "{:.4e}",
            },
            "reproducibility floor for that leg (same rows downdated in two orders)": {
                "seeds": list(SEEDS),
                "values": [ptr(dd, f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev") for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev"} for s in SEEDS],
                "format": "{:.4e}",
            },
            "deviation ABOVE its own floor, i.e. resolvable at all?": derived(
                [ptr(dd, f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev")
                 > ptr(dd, f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev") for s in SEEDS],
                [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                  "pointer": f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev"} for s in SEEDS]
                + [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                    "pointer": f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev"} for s in SEEDS],
                "worst_max_abs_dev > floor_max_abs_dev, per seed", fmt="{}",
                note=("w12r/MF9: TWO of the three TEST seeds sit BELOW their own measured "
                      "reproducibility floor (seed 1: 1.126e-09 vs 4.282e-09; seed 2: 2.842e-14 vs "
                      "5.684e-14), so on those legs the residual is NOT RESOLVABLE and the claim "
                      "rests on the floor rather than on the deviation. The draft declared one.")),
            "max abs deviation over the PINNED deletion only (w12r/MF8: the narrower field the draft printed)": {
                "seeds": list(SEEDS),
                "values": [max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}",
                                "reduction": "max over the 5 wells"} for s in SEEDS],
                "format": "{:.3e}",
                "mean": sum(max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS) / 3,
                "w12r_note": ("MF8. This field covers the PINNED 11,024-row deletion only. The source "
                              "spoke's own FINAL REPORT prints the worst over the pinned deletion AND "
                              "the random-64 deletion, which is 19x larger on seed 0 and 4x larger on "
                              "seed 2. Both are printed; the spoke's own row is the one to quote."),
            },
            "negative control fired on every seed": {
                "seeds": list(SEEDS), "values": [ptr(dd, "/downdate_equals_refit/negative_control_fired_every_seed")] * 3,
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": "/downdate_equals_refit/negative_control_fired_every_seed"}] * 3,
                "format": "{}",
            },
            "cov bytes (statistic, declared beside the pin)": {
                "seeds": list(SEEDS),
                "values": [ptr(dd, f"/cov_bytes_per_seed/{s}") for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/cov_bytes_per_seed/{s}"} for s in SEEDS],
                "format": "{:,.0f}",
            },
            "no pass over data": {
                "seeds": list(SEEDS), "values": [ptr(dd, "/no_pass_over_data")] * 3,
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/no_pass_over_data"}] * 3,
                "format": "{}",
            },
            "items kept": {
                "seeds": list(SEEDS), "values": [ptr(dd, "/items_kept")] * 3,
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/items_kept"}] * 3,
                "format": "{}",
            },
        },
        "caveats_in_the_gate_file_itself": {
            "w12r_note": "MF6. Both are leaf fields of the very file this row cites, and neither was carried.",
            "(a) admission cut is MARGINAL on every seed": {
                "seeds": list(SEEDS),
                "values": [ptr(dd, f"/per_seed/{s}/cut_is_marginal") for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/per_seed/{s}/cut_is_marginal"} for s in SEEDS],
                "format": "{}",
                "what_it_means": ("a direction sits near the threshold, so a small perturbation could "
                                  "change WHICH directions are kept - the same instability class as the "
                                  "original seed-2 failure. It does not block the gate; it belongs in "
                                  "the caption of anything scored on this object."),
                "flagged_by": {"source": "single_design_log", "path": SOURCES["single_design_log"],
                               "locator": "LOG.md, 2026-09-20T22:45Z entry"},
            },
            "(b) the SUBSPACE-level assertion a1_subspace FAILS on seed 2": {
                "passed_every_seed": ptr(dd, "/a1_subspace/passed_every_seed"),
                "seeds_failing": ptr(dd, "/a1_subspace/seeds_failing"),
                "worst_in_float32_ulp": ptr(dd, "/a1_subspace/worst_in_float32_ulp"),
                "worst_floor_subspace_in_float32_ulp": ptr(dd, "/a1_subspace/worst_floor_subspace_in_float32_ulp"),
                "reading": ("the byte-level assertion `downdate_equals_refit` passes on all three seeds; "
                            "the separate subspace-level assertion `a1_subspace` on the PRE-CUT store "
                            "fails on seed 2 at 60.36 ulp - BELOW its own 66.11 ulp floor - so that "
                            "failure is a property of the CONTENT and not of the downdate, which is what "
                            "the artefact's own `reproducibility_floor.what` field states."),
                "watch_out": ("the top-level `all_three_assertions_passed = true` EXCLUDES `a1_subspace`. "
                              "A referee opening the cited JSON reads `passed: false` inside the lead "
                              "result's own artefact."),
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/a1_subspace"},
                               {"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": "/reproducibility_floor/what"}],
            },
        },
        "summary_from_artefact": {
            "worst_max_abs_dev_over_seeds": ptr(dd, "/downdate_equals_refit/worst_max_abs_dev_over_seeds"),
            "worst_in_float32_ulp_at_unit_scale": ptr(dd, "/downdate_equals_refit/worst_in_float32_ulp_at_unit_scale"),
            "passed_every_seed": ptr(dd, "/downdate_equals_refit/passed_every_seed"),
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/downdate_equals_refit"}],
        },
    },
]
# --- w12r/MF3: the four-part byte ledger on BOTH legs of T4 ---------------------------
# The residual below is the one number the source LOG's own four-part table does not name.
# store + covariance + count table + coupling + holes leaves exactly 16,424 B unaccounted on
# EVERY test seed.  16,424 = sum_k d_k x 8 (= 16,384 B, the running MEANS, float64, and
# sum_k d_k = 2,048 = D on every seed) + 5 x 8 (= 40 B, the per-well COUNTS).  With that, the
# ledger closes EXACTLY against honest_total_bytes on all three seeds.  The identification of
# the residual is w12r's inference from `d_k`; the residual itself is arithmetic.
_DK = [[w["d_k"] for w in ptr(dd, f"/per_seed/{s}/content_cost/per_well")] for s in SEEDS]
_COUNT_TABLE_B = 11488
_COUPLING_B, _HOLES_B = 5288, 1322
_MEANS_AND_COUNTS_B = [sum(dk) * 8 + len(dk) * 8 for dk in _DK]

T4_BYTE_LEDGER = {
    "rule": "FOUR PARTS, PRINTED SEPARATELY, NEVER NETTED INTO ONE NUMBER.",
    "w9/maze leg (CMAZE-2)": {
        "see": "T1.byte_ledger - the same object, the same fields, printed in full there",
        "1. atoms (the store)": derived([ptr(DOCS[sid], "/ledger/store_atom_floats") * 4 for sid in MAZE],
                                        [dsrc(sid, "/ledger/store_atom_floats") for sid in MAZE],
                                        "store_atom_floats x 4 (float32)"),
        "2. sufficient statistic, float64": cell(MAZE, "/ledger/statistic_bytes", DOCS, fmt="{:,.0f}"),
        "3. address book (the head)": derived([ptr(DOCS[sid], "/ledger/head_floats") * 4 for sid in MAZE],
                                              [dsrc(sid, "/ledger/head_floats") for sid in MAZE],
                                              "head_floats x 4 (float32)"),
        "4. coupling / hole table": {
            "status": "DECLARED NOT PRESENT in any w9/maze artefact",
            "why": ("the three scored run files carry no field matching 'hole' or 'coupl'. This leg's "
                    "deletion downdates (S, t, n) and the head statistics (A, Mx); it has no interface "
                    "hole table of the kind the w8 leg carries."),
        },
        "deployed (atoms + address book)": cell(MAZE, "/ledger/deployed_bytes", DOCS, fmt="{:,.0f}"),
        "honest total (deployed + statistic)": derived(
            [ptr(DOCS[sid], "/ledger/deployed_bytes") + ptr(DOCS[sid], "/ledger/statistic_bytes")
             for sid in MAZE],
            [dsrc(sid, "/ledger/deployed_bytes") for sid in MAZE]
            + [dsrc(sid, "/ledger/statistic_bytes") for sid in MAZE],
            "deployed_bytes + statistic_bytes"),
        "statistic as a share of deployed": derived(
            [ptr(DOCS[sid], "/ledger/statistic_bytes") / ptr(DOCS[sid], "/ledger/deployed_bytes") for sid in MAZE],
            [dsrc(sid, "/ledger/statistic_bytes") for sid in MAZE],
            "statistic_bytes / deployed_bytes", fmt="{:.2%}"),
        "bytes the deletion's basis-change carry deploys": cell(
            MAZE, "/display/delete_green_exactness/receipt/basis_change_bytes_deployed", DOCS, fmt="{:,.0f}"),
        "pin_status": "NOT under the 524,288 B pin - this leg is matched on PARAMETERS, not on store bytes.",
    },
    "w8/downdate leg (Shapes3D, the 524,288 B pin at tolerance 0)": {
        "1. atoms - principal subspaces (THE STORE, and it matches the pin exactly)": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/per_seed/{s}/store_bytes") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/store_bytes"} for s in SEEDS],
            "format": "{:,.0f}",
        },
        "2a. running covariance, float64 sum_k d_k^2": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/cov_bytes_per_seed/{s}") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/cov_bytes_per_seed/{s}"} for s in SEEDS],
            "format": "{:,.0f}",
        },
        "2b. running means + per-well counts (the residual the source table does not name)": derived(
            _MEANS_AND_COUNTS_B,
            [{"source": "downdate_test", "path": SOURCES["downdate_test"],
              "pointer": f"/per_seed/{s}/content_cost/per_well"} for s in SEEDS],
            "sum_k d_k x 8 + n_wells x 8 (float64 means, float64 counts)",
            note=("16,424 B on every seed. It is exactly the residual left by honest_total_bytes minus "
                  "store minus covariance minus the count table minus coupling minus holes, so the "
                  "four-part ledger closes to the byte once it is printed.")),
        "2. covariance / count ledger, TOTAL": derived(
            [ptr(dd, f"/cov_bytes_per_seed/{s}") + _MEANS_AND_COUNTS_B[s] for s in SEEDS],
            [{"source": "downdate_test", "path": SOURCES["downdate_test"],
              "pointer": f"/cov_bytes_per_seed/{s}"} for s in SEEDS],
            "cov_bytes + means_and_counts"),
        "3. request count table": locator(
            [_COUNT_TABLE_B] * 3, "downdate_log", "LOG.md:420 and :1041",
            "part 3 of the source spoke's own four-part ledger table", fmt="{:,.0f}"),
        "4. coupling + holes": locator(
            [f"{_COUPLING_B:,} + {_HOLES_B:,}"] * 3, "downdate_log", "LOG.md:421 and :1042",
            "part 4 of the source spoke's own four-part ledger table: coupling table + one byte per hole",
            fmt="{}"),
        "honest total (all four parts)": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/per_seed/{s}/honest_total_bytes") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/honest_total_bytes"} for s in SEEDS],
            "format": "{:,.0f}",
        },
        "x THE 524,288 B PIN": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/per_seed/{s}/times_over_budget") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/times_over_budget"} for s in SEEDS],
            "format": "{:.4f}x",
        },
        "ledger closes to the byte?": derived(
            [ptr(dd, f"/per_seed/{s}/store_bytes") + ptr(dd, f"/cov_bytes_per_seed/{s}")
             + _MEANS_AND_COUNTS_B[s] + _COUNT_TABLE_B + _COUPLING_B + _HOLES_B
             == ptr(dd, f"/per_seed/{s}/honest_total_bytes") for s in SEEDS],
            [{"source": "downdate_test", "path": SOURCES["downdate_test"],
              "pointer": f"/per_seed/{s}/honest_total_bytes"} for s in SEEDS],
            "store + cov + means_and_counts + 11,488 + 5,288 + 1,322 == honest_total_bytes", fmt="{}"),
        "serialised object bytes, for comparison": {
            "seeds": list(SEEDS),
            "values": [ptr(dd, f"/per_seed/{s}/items_kept_detail/object_bytes_full") for s in SEEDS],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/items_kept_detail/object_bytes_full"} for s in SEEDS],
            "format": "{:,.0f}",
        },
        "DECLARED RESIDUAL, unexplained": derived(
            [ptr(dd, f"/per_seed/{s}/items_kept_detail/object_bytes_full")
             - ptr(dd, f"/per_seed/{s}/honest_total_bytes") for s in SEEDS],
            [{"source": "downdate_test", "path": SOURCES["downdate_test"],
              "pointer": f"/per_seed/{s}/items_kept_detail/object_bytes_full"} for s in SEEDS],
            "object_bytes_full - honest_total_bytes",
            note=("exactly 16,384 B on every one of the three seeds. w12r could not attribute it to a "
                  "named field and does NOT net it away: it is reported as a declared residual. "
                  "It does not move `times_over_budget`, which the artefact computes from "
                  "honest_total_bytes.")),
        "THE HEADLINE THE DRAFT OMITTED": (
            "the object that forgets exactly is itself 24.55x / 23.84x / 23.67x the 524,288 B pin. "
            "The paper penalises the compose-and-forget object of section 5 at 7.48x; its own lead "
            "result is three times worse and printed no byte row at all."),
        "provenance_of_the_four_part_split": [
            {"source": "downdate_log", "path": SOURCES["downdate_log"],
             "locator": "LOG.md:414-422 (section 3, THE LEDGER) and LOG.md:1036-1044 (FINAL REPORT "
                        "section 3) - the source spoke's own four-part table. Parts 3 and 4 exist only "
                        "in that markdown, so they are carried as locators and flagged as not "
                        "machine-checkable by verify_tables.py."},
        ],
    },
}

TABLES["T4"] = {
    "title": "Exact deletion (STORE-SCOPED) - two implementations side by side, 3 seeds each, TEST condition. THE THREE implementations, with their bands kept apart, are TABLE 5.",
    "instrument": "downdate vs refit-without-those-rows, compared directly; no tolerance band claimed",
    "pins": {
        "w9/maze leg": "K=8, n_atoms=31, CluRead sufficient statistics (S,t,n) + head statistics (A, Mx)",
        "w8/downdate leg": "K=5 wells on Shapes3D features, per-well covariance + running mean + count",
        "generation": 2,
        "seeds": [0, 1, 2],
    },
    "pin_provenance": [
        {"source": "maze_s0", "path": SOURCES["maze_s0"], "pointer": "/display/delete_green_exactness/receipt/mechanism"},
        {"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/note"},
    ],
    "caption": (
        "EXACT HERE MEANS STORE-SCOPED (TABLES.json /exactness_definition): after deleting a set of rows the "
        "STORE CONTENT equals, to float roundoff, the content fitted from the retained rows alone, HOLDING THE "
        "FROZEN ENCODER AND DECODER FIXED. Neither is refit. It is not a claim that the renderer forgets. "
        "Both legs are 3 seeds on generation 2 on their own TEST condition. They are reported side by side "
        "as two implementations of one claim, not as one instrument: the deviations are in different units "
        "(w9/maze reports a RELATIVE residual on the statistic and a projector residual; w8/downdate reports "
        "an ABSOLUTE max deviation per well). "
        "CAVEATS THAT TRAVEL WITH THE w8 LEG (w12r/MF6): the store is built under a variance-share admission "
        "cut that is MARGINAL on every seed (`cut_is_marginal_any_seed = true`); the byte-level assertion "
        "`downdate_equals_refit` passes on all three seeds, while the separate subspace-level assertion "
        "`a1_subspace` on the PRE-CUT store FAILS on seed 2 at 60.36 ulp - below its own 66.11 ulp floor - so "
        "that failure is a property of the content, not of the downdate; and the top-level "
        "`all_three_assertions_passed = true` EXCLUDES `a1_subspace`. "
        "THE w8 LEG IS NOT A SCORED ROW (w12r/MF7): it is an assertion battery on the object at 0 scored runs; "
        "the hub commissioned to score it returned its full budget (DECLARED NOT-RUN NR3). "
        "THE BYTE LEDGER IS PRINTED BELOW AND IS NEVER NETTED (w12r/MF3): the w8 object's honest total is "
        "24.55x / 23.84x / 23.67x the 524,288 B pin."
    ),
    "rows": t4_rows,
    "byte_ledger": T4_BYTE_LEDGER,
    "exactness_definition_ref": "TABLES.json /exactness_definition (EXACT-STORE-SCOPED)",
    "exact_is_not_free": {
        "w12r_note": "MF4/MF5: exactness is claimed; cost-free erasure is not.",
        "surviving-target composition on TEST, BEFORE": cell(
            MAZE, "/display/delete_green/other_before/validity", DOCS),
        "surviving-target composition on TEST, AFTER": cell(
            MAZE, "/display/delete_green/other_after/validity", DOCS),
    },
}


# --------------------------------------------------------------------------------------
# 6b. w12r -- TABLE 5.  THREE IMPLEMENTATIONS, THREE ROWS.
#     "9/9 seeds, two independent implementations" is wrong in three ways at once:
#     the nine legs are 3 CONFIGURATIONS x 3 seeds inside ONE implementation; the band is
#     not shared; and it was not unconditional - it FAILED first.  There are THREE
#     implementations, and the strongest form of the claim is the FAILURE AGREEMENT.
# --------------------------------------------------------------------------------------
sd = "single_design_log"
t5_rows = [
    {
        "implementation": "1. w9 hub/maze - `CluRead.delete_rows_exact` (CMAZE-2, TEST)",
        "row_class": "RESULT - 3 of 3 scored runs (S1/S2/S3)",
        "seeds": 3, "configurations": 1, "legs": 3, "scored_runs": 3,
        "configuration": "K=8, n_atoms=31 - one pinned configuration",
        "metric": "max RELATIVE residual on the sufficient statistic; max projector residual on the subspace",
        "cells": {
            "statistic max relative residual": cell(
                MAZE, "/display/delete_green_exactness/statistic_max_rel_residual", DOCS, fmt="{:.4e}"),
            "subspace max projector residual": cell(
                MAZE, "/display/delete_green_exactness/subspace_max_projector_residual", DOCS, fmt="{:.4e}"),
            "refit performed": cell(MAZE, "/display/delete_green_exactness/refit", DOCS, fmt="{}"),
            "passes over retained data": cell(
                MAZE, "/display/delete_green_exactness/passes_over_retained_data", DOCS, fmt="{:.0f}"),
        },
        "band": "7.522e-16 .. 8.826e-16 on the statistic; EXACTLY 0.000e+00 on the subspace projector",
        "note": ("this is the leg that is unambiguously in the 1e-16 band, it is the only one inside "
                 "scored runs, and it replicates from a clean checkout with ZERO numeric differences "
                 "(`/c/chlu/outputs/w9-maze/replication/REPLICATION.md`)"),
    },
    {
        "implementation": "2. w8 spoke/downdate-build - `exp_w8_downdate` (Shapes3D, TEST)",
        "row_class": "ASSERTION BATTERY (object property) - 0 SCORED RUNS",
        "seeds": 3, "configurations": 1, "legs": 3, "scored_runs": 0,
        "configuration": "factor partition K=5, A=64 - the incumbent's grain",
        "metric": "max ABSOLUTE deviation per well, downdate vs refit-without-those-rows",
        "cells": {
            "worst max abs deviation (pinned 11,024 rows + random 64)": {
                "seeds": list(SEEDS),
                "values": [ptr(dd, f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev") for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev"}
                               for s in SEEDS],
                "format": "{:.4e}",
            },
            "its own reproducibility floor": {
                "seeds": list(SEEDS),
                "values": [ptr(dd, f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev") for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev"}
                               for s in SEEDS],
                "format": "{:.4e}",
            },
        },
        "band": "2.842e-14 .. 1.126e-09 - FIVE ORDERS COARSER than implementation 1",
        "note": ("the worst leg (TEST seed 1, 1.126e-09) sits BELOW its own reproducibility floor of "
                 "4.282e-09, and so does seed 2 (2.842e-14 vs 5.684e-14): on TWO of three legs the "
                 "residual is not resolvable at all. Printed that way it is STRONGER, not weaker - an "
                 "unresolvable residual is the best available outcome for an exactness claim. "
                 "The two additional val legs the advisor quotes are 4.547e-13 and 1.164e-10."),
    },
    {
        "implementation": "3. w8 hub/single-design - matched-bytes candidates (Shapes3D, TEST)",
        "row_class": "ASSERTION BATTERY (object property) - 0 SCORED RUNS",
        "seeds": 3, "configurations": 3, "legs": 9, "scored_runs": 0,
        "configuration": "ward_balanced K=40 / K=44 / K=48 - THREE configurations, three seeds each",
        "metric": "max atom deviation in float32 ulp at unit scale; and the same at subspace level",
        "cells_markdown_only": {
            "atom-level ulp, worst over the 9 legs": locator(
                [0.094], sd, "LOG.md:476-481", "9/9 PASS, worst observed 0.0938 ulp", fmt="{:.4f}"),
            "subspace-level ulp, worst over the 9 legs": locator(
                [0.0251], sd, "LOG.md:495-503", "matched-bytes 9/9 PASS, worst subspace 0.0251 ulp",
                fmt="{:.4f}"),
            "absolute deviation, seed 0, K=40 / K=44 / K=48": locator(
                [5.551e-16, 1.421e-14, 1.490e-13], sd, "LOG.md:472",
                "after the sign-convention fix, test leg seed 0", fmt="{:.4e}"),
        },
        "band": "5.551e-16 .. 1.490e-13 on seed 0; worst ulp over all nine legs 0.094 (atom) / 0.0251 (subspace)",
        "note": ("THIS is the implementation the phrase '9/9' came from, and the nine are 3 "
                 "CONFIGURATIONS x 3 SEEDS, not nine seeds. It passes under BOTH the atom definition "
                 "and the subspace definition, so it does not need the definition change to survive. "
                 "Every number in this row lives in a MARKDOWN artefact and is carried as a locator: "
                 "verify_tables.py re-hashes the file but cannot re-resolve these values."),
        "machine_checked_by_verify_tables": False,
    },
]

T5_FAILURE_AGREEMENT = {
    "title": "THE STRONGEST FORM OF THE TWO-IMPLEMENTATION CLAIM IS THE FAILURE AGREEMENT, NOT THE SUCCESSES",
    "what_happened": (
        "Two implementations, different code, different authors, different sign conventions, "
        "reproduced THE SAME failure on THE SAME seed in THE SAME well: the incumbent's coarse, "
        "imbalanced `factor K=5, A=64` partition on TEST seed 2."
    ),
    "legs": {
        "w8 spoke/downdate-build, atom level": locator(
            [32.90], "downdate_log", "LOG.md:625-640 (3.921e-06 = 32.90 ulp) and LOG.md:728,794,905 (4.227e-06 = 35.46 ulp)",
            "assertion 1 FAILS on test seed 2 against a declared criterion of <= 1 float32 ulp", fmt="{:.2f}"),
        "w8 spoke/downdate-build, subspace level (a stored field)": {
            "seeds": [2], "values": [ptr(dd, "/a1_subspace/worst_in_float32_ulp")],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": "/a1_subspace/worst_in_float32_ulp"}],
            "format": "{:.5f}",
        },
        "w8 spoke/downdate-build, its own subspace floor (a stored field)": {
            "seeds": [2], "values": [ptr(dd, "/a1_subspace/worst_floor_subspace_in_float32_ulp")],
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": "/a1_subspace/worst_floor_subspace_in_float32_ulp"}],
            "format": "{:.5f}",
        },
        "w8 hub/single-design, atom level": locator(
            [14.377], sd, "LOG.md:479,481", "the same seed-2 failure, reproduced independently", fmt="{:.3f}"),
        "w8 hub/single-design, subspace level": locator(
            [14.4518], sd, "LOG.md:501", "the subspace definition does NOT rescue it", fmt="{:.4f}"),
        "w8 hub/single-design, its own two-order floor": locator(
            [36.9305], sd, "LOG.md:501,506",
            "the floor EXCEEDS the observed deviation: seed 2's subspace is not a function of its data",
            fmt="{:.4f}"),
    },
    "the_citable_sentence": (
        "The second implementation reproduced the same seed-2 failure with different code and its own "
        "sign convention - 14.377 ulp against 32.90 - and both implementations' own reproducibility "
        "floors EXCEED their observed deviations there, so both independently conclude that seed 2's "
        "content is not a function of its data by any method. That is a stronger corroboration than "
        "either implementation's successes."
    ),
    "how_it_was_closed": {
        "fix": "a variance-share admission cut with NO FREE PARAMETER, declared before the re-run and not moved",
        "tau": ptr(dd, "/per_seed/0/content_cost/var_share_tol"),
        "tau_provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                            "pointer": f"/per_seed/{s}/content_cost/var_share_tol"} for s in SEEDS],
        "tau_derivation": "4.941056e-16 / 1.1920929e-07 = 4.144858e-09",
    },
    "what_must_never_be_printed": (
        "'1e-16..1e-14 on Shapes3D, 9/9 seeds, two independent implementations'. Wrong three ways: "
        "the nine legs are 3 configurations x 3 seeds inside ONE implementation; the band is not "
        "shared (the peer runs 2.842e-14 .. 1.126e-09, five orders coarser, its worst leg below its "
        "own 4.282e-09 reproducibility floor); and it was not unconditional - it FAILED first at "
        "32.90 ulp."
    ),
}

TABLES["T5"] = {
    "title": "Exact deletion - THE THREE IMPLEMENTATIONS AS THREE ROWS, and the failure they agree on",
    "w12r_note": ("Added by w12r spoke/artefact-mustfix. There are THREE implementations, not two, and "
                  "their bands are NOT shared. Seed count and configuration count are separate fields "
                  "on every row so that no sentence can multiply them into a seed count."),
    "instrument": ("three different codebases, three different authors, compared only on the QUESTION "
                   "(does the downdate equal the refit?), never merged into one band"),
    "pins": {
        "generation": 2,
        "condition": "TEST on all three implementations",
        "exactness_scope": "STORE-SCOPED - see TABLES.json /exactness_definition",
    },
    "pin_provenance": [
        {"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/condition"},
        {"source": "maze_s0", "path": SOURCES["maze_s0"], "pointer": "/which"},
    ],
    "caption": (
        "THE BANDS ARE NOT SHARED AND ARE NOT MERGED. Implementation 1 (w9/maze, CMAZE-2) is at "
        "7.5e-16..8.8e-16 with a projector residual of exactly 0.000e+00 and is the only one inside "
        "scored runs. Implementation 2 (w8 spoke/downdate-build) runs 2.842e-14..1.126e-09 - five "
        "orders coarser - with TWO of its three TEST legs below their own reproducibility floors. "
        "Implementation 3 (w8 hub/single-design) is 3 CONFIGURATIONS x 3 SEEDS = 9 legs, worst 0.094 "
        "atom ulp / 0.0251 subspace ulp. Nine legs are NOT nine seeds."
    ),
    "rows": t5_rows,
    "failure_agreement": T5_FAILURE_AGREEMENT,
    "seed_and_configuration_counts": SEED_AND_CONFIGURATION_COUNTS,
}


# --------------------------------------------------------------------------------------
# 6c. w12r -- TABLE 6.  The w5 compose-field arm's BYTE and PARAMETER ledger.
#     Closes referee MF12 (the parameter match is quoted from a figure the source spoke
#     CORRECTED AWAY), MF13 ("at 524,288 B" invites a byte comparison the arm does not win
#     as stated) and SF2 (the ceiling ladder's rungs differ 9x in address-book size).
# --------------------------------------------------------------------------------------
W5 = ["w5_read_s0", "w5_read_s1", "w5_read_s2"]
_W5_RIVAL_PARAMS = 178952


def w5_cell(arm, field, fmt="{:,.0f}"):
    return cell(W5, f"/arms/{arm}/{field}", DOCS, fmt=fmt)


_arm_total_params = [ptr(DOCS[sid], "/arms/pca_tensor/param_count")
                     + ptr(DOCS[sid], "/arms/pca_tensor/address_bytes") // 4 for sid in W5]
_arm_total_bytes = [ptr(DOCS[sid], "/arms/pca_tensor/store_bytes")
                    + ptr(DOCS[sid], "/arms/pca_tensor/address_bytes") for sid in W5]
_rival_bytes = _W5_RIVAL_PARAMS * 4

T6_LADDER_RUNGS = ["oracle_content_fine", "oracle_content_items", "oracle_content_pool",
                   "oracle_span", "oracle_span_field", "span_head"]

TABLES["T6"] = {
    "title": "w5 compose-field - the BYTE and PARAMETER ledger of `pca_tensor` against `rival_mlp_180k`, TEST, 3 seeds",
    "w12r_note": ("Added by w12r spoke/artefact-mustfix. The draft quoted a -0.01% parameter match "
                  "from the VAL leg, in the very place the source spoke wrote a correction bullet to "
                  "prevent it, and quoted `at 524,288 B` in a way that reads as 27% cheaper than the "
                  "rival. Both are fixed here with the scored TEST numbers, and both come out as a "
                  "STRONGER claim than the draft's: the comparison is matched to within +-0.5%."),
    "instrument": "byte and parameter counts recorded by the scoring run itself; no measurement is re-derived",
    "pins": {"condition": "test", "seeds": [0, 1, 2], "store_pin_B": 524288, "tolerance": 0},
    "pin_provenance": [{"source": "w5_read_s0", "path": SOURCES["w5_read_s0"], "pointer": "/pins"},
                       {"source": "w5_read_s0", "path": SOURCES["w5_read_s0"], "pointer": "/condition"}],
    "caption": (
        "THE STORE IS PINNED AT 524,288 B AT TOLERANCE 0; THE ADDRESS BOOK IS DECLARED BESIDE IT AND "
        "NEVER NETTED. Total deployed bytes are 713,632 / 719,352 / 719,352 against the rival's "
        "715,808 - matched to within +-0.5%, which is what the artefacts prove and is a stronger "
        "statement than `at 524,288 B`. On parameters the arm carries 178,408 / 179,838 / 179,838 "
        "against 178,952: -0.304% / +0.495% / +0.495%, so ON TWO OF THREE SEEDS THE ARM CARRIES "
        "SLIGHTLY MORE PARAMETERS THAN THE BAR. It is stated that way round rather than quoting the "
        "seed that flatters the arm."
    ),
    "rows": [
        {
            "label": "`pca_tensor` (THE ARM)",
            "row_class": "arm",
            "cells": {
                "store bytes (the pin, tolerance 0)": w5_cell("pca_tensor", "store_bytes"),
                "address book bytes (declared beside, never netted)": w5_cell("pca_tensor", "address_bytes"),
                "store parameters": w5_cell("pca_tensor", "param_count"),
                "TOTAL deployed bytes (store + address book)": derived(
                    _arm_total_bytes,
                    [dsrc(sid, "/arms/pca_tensor/store_bytes") for sid in W5]
                    + [dsrc(sid, "/arms/pca_tensor/address_bytes") for sid in W5],
                    "store_bytes + address_bytes"),
                "TOTAL parameters (store + address book / 4)": derived(
                    _arm_total_params,
                    [dsrc(sid, "/arms/pca_tensor/param_count") for sid in W5]
                    + [dsrc(sid, "/arms/pca_tensor/address_bytes") for sid in W5],
                    "param_count + address_bytes // 4 (float32 book)"),
            },
        },
        {
            "label": "`rival_mlp_180k` (the parameter bar)",
            "row_class": "RESULT baseline",
            "cells": {
                "store bytes": w5_cell("rival_mlp_180k", "store_bytes"),
                "address book bytes": w5_cell("rival_mlp_180k", "address_bytes"),
                "parameters": w5_cell("rival_mlp_180k", "param_count"),
                "TOTAL deployed bytes": derived(
                    [_rival_bytes] * 3,
                    [dsrc(sid, "/arms/rival_mlp_180k/param_count") for sid in W5],
                    "param_count x 4 (float32); the rival has no store and no address book"),
            },
        },
        {
            "label": "THE COMPARISON, as the artefacts support it",
            "row_class": "derived",
            "cells": {
                "arm bytes as a share of the rival's": derived(
                    [b / _rival_bytes for b in _arm_total_bytes],
                    [dsrc(sid, "/arms/pca_tensor/address_bytes") for sid in W5],
                    "(store_bytes + address_bytes) / (rival param_count x 4)", fmt="{:.2%}"),
                "arm parameters against the bar": derived(
                    [(p - _W5_RIVAL_PARAMS) / _W5_RIVAL_PARAMS for p in _arm_total_params],
                    [dsrc(sid, "/arms/pca_tensor/param_count") for sid in W5]
                    + [dsrc(sid, "/arms/rival_mlp_180k/param_count") for sid in W5],
                    "(arm_total_params - 178,952) / 178,952", fmt="{:+.3%}"),
            },
        },
    ],
    "the_correction_this_table_carries": {
        "id": "MF12",
        "what_the_draft_printed": "178,928 vs 178,952 = -0.01%, pointed at `w5-compose-field/LOG.md` 02:45Z",
        "why_it_is_wrong": ("178,928 is the VAL leg's figure. The source spoke wrote a correction bullet "
                            "at 01:58 UTC headed 'CORRECTION to a number I put in the PREREG' saying so, "
                            "and the 02:45Z entry the draft points at is the val arm-clears-the-bar "
                            "bullet (+16.875), not a byte or parameter entry."),
        "what_the_test_legs_say": "178,408 / 179,838 / 179,838 = -0.304% / +0.495% / +0.495%",
        "the_spokes_own_sentence": ("the secondary parameter match is within +-0.5% rather than the "
                                    "+-0.013% I wrote, and on two of three seeds the arm carries "
                                    "slightly MORE parameters than the bar. I state it that way round "
                                    "rather than quoting the seed that flatters me."),
        "provenance": [{"source": "w5_log", "path": SOURCES["w5_log"], "locator": "LOG.md, the 01:58 UTC entry"}]
                      + [dsrc(sid, "/arms/pca_tensor/param_count") for sid in W5],
    },
    "ceiling_ladder_address_books": {
        "w12r_note": ("SF2. A byte-ledger paper cannot ship a ceiling ladder whose rungs differ 9x in "
                      "address-book size silently. `oracle_span` and `span_head` carry an address book "
                      "of 1,766,400 B - 9.3x the arm's own book and 3.4x the 524,288 B store pin - "
                      "while every other rung carries 0."),
        "rungs": {r: {"address_bytes": w5_cell(r, "address_bytes"),
                      "store_bytes": (w5_cell(r, "store_bytes") if
                                      ptr(DOCS["w5_read_s0"], f"/arms/{r}/store_bytes") is not None else None),
                      "class": ptr(DOCS["w5_read_s0"], f"/arms/{r}/class")}
                  for r in T6_LADDER_RUNGS},
    },
}

# --------------------------------------------------------------------------------------
# 7.  DECLARED NOT-RUNS -- never a blank, never a zero, each with its launch line.
# --------------------------------------------------------------------------------------
NOT_RUN = [
    {
        "id": "NR1",
        "row": "Shapes3D generation cell (incumbent instrument: K=64, A=1,216 NMF atoms, shared-basis head, "
               "eval_b, composition-B validity) at 3 seeds",
        "status": "NOT RUN at 3 seeds - SEED 0 ONLY EXISTS",
        "what_exists": "one scored cell, seed 0: arm 0.8495 / 51 tuples of 72 at 454,336 B against knn_proto "
                       "0.8588 / 57 at 475,136 B; delta -0.0093, TIE under R1",
        "why": "hub/generation-table-2 spent scored run 1 of 3 on seed 0; run 3 is still unspent. A 1-seed row "
               "cannot enter a 3-seed table and is not averaged into one.",
        "evidence": [{"source": "gentable_log", "path": SOURCES["gentable_log"], "locator": "16:12Z scored-run table"},
                     {"source": "gentable_spec", "path": SOURCES["gentable_spec"], "pointer": "/seed"},
                     {"source": "w8_advisor", "path": SOURCES["w8_advisor"], "locator": "Budget after wave 8"}],
        "launch_line": (
            "# for SEED in 1 2 ; two legs per seed, from $CHLU_ROOT\n"
            "python -m chlu.experiments.exp_gen_cell --stage cell --seed $SEED\n"
            "jq '.seed=$SEED' /c/chlu/outputs/hub-generation-table/relaunch/scored/scored_spec.json "
            "> /tmp/scored_spec_s$SEED.json\n"
            "python /c/chlu/outputs/hub-generation-table/relaunch/scored/run_arm.py "
            "<out_dir>/seed$SEED /tmp/scored_spec_s$SEED.json"
        ),
        "cost_note": "leg 1 814 s + leg 2 277 s on FORGIS_14 CPU for seed 0; two more seeds ~ 40 min CPU total, "
                     "but each consumes one of the remaining `generation-table` scored runs (1 left).",
    },
    {
        "id": "NR2",
        "row": "Shapes3D generation cell at the GPU compute pin (joint_steps=4000, n_train=32768, dec_steps=1800)",
        "status": "NOT RUN - deferred, spec ready, blocked on the Docker daemon on the Head's Mac",
        "why": "the CPU leg was forced to joint_steps=400 and the shortened `w8_v1_short` decoder; at that pin "
               "BAR D is VOID (the 400-step decoder pins the MSE-regression baseline to exactly 0.000 where the "
               "incumbent's 1800-step decoder gives it 53-60) and C.1 is weak evidence only. This is a SECOND "
               "TABLE at a different budget, not extra rows for T2.",
        "evidence": [{"source": "enc_launchspec", "path": SOURCES["enc_launchspec"], "locator": "SS1 launch block, SS5 cost table"},
                     {"source": "enc_log", "path": SOURCES["enc_log"], "locator": "FINAL REPORT / DECLARED NOT-RUN"}],
        "launch_line": (
            "export CHLU_W8_CKPT_REMOTE=gs://chlu-cloud-forgisprova/jobs\n"
            "for SEED in 0 1 2; do\n"
            "  scripts/cloud/run_until_done.sh --job \"w8enc-s${SEED}\" --card a100-40 --spot \\\n"
            "      --exp w8_encoder_loop --units \"${SEED}\" \\\n"
            "      --artefacts gs://chlu-cloud-forgisprova/artefacts \\\n"
            "      --exp-args \"--stage cloud --ckpt-every-s 300 --joint-steps 4000 --dec-steps 1800 "
            "--warm-steps 4000 --n-train 32768\" \\\n"
            "      --max-hours 6 --max-attempts 8 --max-usd 12.00\n"
            "done"
        ),
        "cost_note": "$26.46 for 3 seeds at A100-40 SPOT $1.47/h; A100 concurrency is ONE, so the seeds are "
                     "sequential. `kill -9` resumability proved (resumed leg reached bit-identical parameters).",
    },
    {
        "id": "NR3",
        "row": "w8 single-design object (covariance store that composes AND forgets) - RESULT rows 3/4 and "
               "post-deletion composition/refusal at 3 seeds on TEST",
        "status": "NOT RUN - 0 of 3 scored, full budget intact",
        "why_w12r_qualifier": (
            "w12r/MF16. `at K >= 36 composition collapses` is TRUE OF THE ONE CONFIGURATION AT K >= 36 "
            "WHOSE COMPOSITION WAS MEASURED, and must be stated that way. The same hub's 18:32Z "
            "`ward_balanced` sweep lists FIVE configurations that fit the 524,288 B pin at float64 - "
            "K=36 (span 144, 523,680 B, 0.999x), K=38 (span 266, 523,744 B, 0.999x), K=40 (span 360, "
            "518,080 B, 0.988x), K=42 (span 504, 522,704 B, 0.997x), K=48 (span 864, 521,984 B, "
            "0.996x) - and the composition of FOUR of them was never measured. The one that was "
            "measured is K=40 with A=7, span 280, 519,794 B = 0.991x (the 23:22Z entry), which sits "
            "BELOW the sweep's own K=40 row and below the composing object's realised span of 308: "
            "three of the five that fit the pin carry MORE deployed span than the object that composes "
            "at 0.8918. The threshold is also DTYPE-CONDITIONAL: the same hub's earlier sweep prints a "
            "float32 matched-bytes point at K=21, A=94, 517,952 B = 0.988x the pin, span 336, excluded "
            "by ARGUMENT (the peer's measured 1e-10..5e-10 downdate residuals are held not to survive "
            "float32) and NEVER MEASURED. The supported form is: NO OBJECT WITH A FLOAT64 SUFFICIENT "
            "STATISTIC composes and forgets inside 524,288 B, on the one such configuration measured."),
        "why_w12r_qualifier_provenance": [
            {"source": "single_design_log", "path": SOURCES["single_design_log"],
             "locator": "LOG.md, the 18:32Z ward_balanced sweep; the 23:22Z K=40/A=7 entry; LOG.md:99 "
                        "for the float32 matched-bytes point"}],
        "why": "the object was falsified on `val` before a scored run was justified: the covariance costs "
               "sum_k d_k^2, fitting it inside the 524,288 B pin forces K >= 36, and at K=40/A=7 "
               "(519,794 B = 0.991x the pin) composition collapses to valid mean 0.0648. At K=5 composition "
               "holds at 0.8918 (incumbent 0.8941) but the honest four-part ledger is 3,922,450 B = 7.48x "
               "the pin. Byte pin is primary at tolerance 0.",
        "evidence": [{"source": "single_design_log", "path": SOURCES["single_design_log"],
                      "locator": "FINAL REPORT (REVISED) SS1-SS3"}],
        "launch_line": "# would require a content whose exact deletion does NOT need a dxd statistic; no such\n"
                       "# object exists on disk. The nearest runnable leg is the ward_balanced K=5 object at\n"
                       "# 7.48x the byte pin, which cannot carry a RESULT row under the pin-primary rule:\n"
                       "python -m chlu.experiments.exp_w8_single_design --stage score --partition ward_balanced "
                       "--k 5 --condition test --seed {0,1,2}",
        "cost_note": "declared NOT-RUN is not a null: the val falsification is the wave-8 answer.",
    },
    {
        "id": "NR4",
        "row": "hub/maze-encoder-loop - the w8 encoder-loop comparison repeated on CMAZE-1",
        "status": "NOT RUN - contingent, and the contingency did not fire",
        "why": "the wave plan commissions it only `if w8's result was positive; otherwise a declared NOT-RUN`. "
               "w8/encoder-loop's BAR B falsifier FIRED on row 3 and the net deltas straddle zero on row 4, "
               "so the contingency is unmet.",
        "evidence": [{"source": "wave_plan", "path": SOURCES["wave_plan"], "locator": "SS2 hub/maze-encoder-loop"},
                     {"source": "enc_crossarm", "path": SOURCES["enc_crossarm"], "pointer": "/VERDICT"}],
        "launch_line": "# only if a positive encoder-loop result appears at the GPU pin (NR2):\n"
                       "python -m chlu.experiments.exp_w9_maze --stage score --encoder-arm {alone,loop_swap} "
                       "--condition test --seed {0,1,2}",
        "cost_note": "no maze-encoder-loop budget was drawn.",
    },
    {
        "id": "NR5",
        "row": "w9/maze post-deletion composition with the REPAIRED deletion, on TEST at 3 seeds",
        "status": "NOT RUN on TEST - measured on VAL, seed 0 only (Amendment 1)",
        "why": "the scored budget was 3/3 spent before the repair existed. The as-pre-registered scored-leg "
               "DISPLAY row stands (surviving targets fall to 0.0000 on all three seeds); the repaired "
               "mechanism was re-measured on VAL seed 0 only, at -0.0120, inside BAR 6's +-0.05.",
        "evidence": [{"source": "maze_amend_val_s0", "path": SOURCES["maze_amend_val_s0"],
                      "pointer": "/display/delete_green/composition_without_retraining/validity_delta_on_other_targets"},
                     {"source": "maze_log", "path": SOURCES["maze_log"], "locator": "09:55Z AMENDMENT 1"},
                     {"source": "maze_ledger", "path": SOURCES["maze_ledger"], "locator": "row A1"}],
        "launch_line": "python -m chlu.experiments.exp_w9_maze --stage score --exact-head-delete "
                       "--condition test --seed {0,1,2}   # needs a 4th scored run; budget is 3/3 spent",
        "cost_note": "~240 s/seed on FORGIS_14 CPU, but it is a HELD-OUT number and the maze budget is exhausted.",
    },
]

# --------------------------------------------------------------------------------------
# 8.  CORRECTIONS -- narrative vs artefact, found while tracing every cell.
# --------------------------------------------------------------------------------------
CORRECTIONS = [
    {
        "id": "C1",
        "claim_as_written": "BLIND-SEQUENTIAL on this hub's split is `0.9037-0.9130` (w10 task file; "
                            "w9-maze LOG.md 09:40Z bullet quotes 0.9130 for seed 2)",
        "artefact_says": "unseen-pair BLIND-SEQUENTIAL = 0.9077 / 0.8905 / 0.9128 for seeds 0/1/2, mean 0.9037; "
                         "n-way BLIND-SEQUENTIAL = 0.9229 / 0.8933 / 0.8841, mean 0.9001",
        "effect": "the quoted band's upper end is 0.9128 not 0.9130, and its lower end is the SEED MINIMUM 0.8905, "
                  "not the 3-seed mean 0.9037. The band as written mixes a mean with a per-seed value.",
        "provenance": [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                        "pointer": "/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity"} for s in SEEDS],
        "action": "T1 prints the three per-seed values; the mean is labelled as a mean.",
    },
    {
        "id": "C2",
        "claim_as_written": "w8 advisor report: exact deletion `downdate == refit ... to 1e-16...1e-14`, 9/9 seeds",
        "artefact_says": "DOWNDATE-READY-test.json worst_max_abs_dev_over_seeds = 1.1255e-09 "
                         "(seed 1, well 3); DOWNDATE-READY.json (val) worst_max_abs_dev = 1.1642e-10",
        "effect": "the stated range understates the worst observed deviation by ~5 orders of magnitude. The "
                  "claim still holds in the units the artefact itself uses for the bar "
                  "(worst_in_float32_ulp_at_unit_scale = 0.0094, i.e. under one float32 ulp), but the paper must "
                  "quote the ulp figure or the true absolute, not `1e-14`.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/downdate_equals_refit"},
                       {"source": "downdate_val", "path": SOURCES["downdate_val"], "pointer": "/downdate_equals_refit"}],
        "action": "T4 prints per-seed absolutes and the ulp figure side by side.",
    },
    {
        "id": "C3",
        "claim_as_written": "the n-way row and its ceiling (w9 charter warning)",
        "artefact_says": "the scored n-way row has n = 183 / 192 / 191 (a CONJUNCTION over the routes of one "
                         "request), while pipeline_ceiling_rows.nway has n = 389 / 403 / 414 (PER ROUTE)",
        "effect": "the row and its printed ceiling are computed over different denominators, so "
                  "0.0284 vs 0.5255 is not a like-for-like fraction-of-ceiling.",
        "provenance": ([{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"], "pointer": "/table/nway/clu/n"} for s in SEEDS]
                       + [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"], "pointer": "/pipeline_ceiling_rows/nway/n"} for s in SEEDS]),
        "action": "T1 carries row 3 TWICE - conjunction and per route - and prints n on both.",
    },
    {
        "id": "C4",
        "claim_as_written": "w8 advisor report SS5.5: `Wave 9 inherits the 0.4980 blind-sequential column as a caveat`",
        "artefact_says": "BLIND-SEQUENTIAL is 0.5317 (0.5550 / 0.5300 / 0.5100) at the w9/harness tier - which "
                         "DOES reproduce the published 0.4980 - and 0.9037 (0.9077 / 0.8905 / 0.9128) on the "
                         "w9/maze hub's unseen-pair split",
        "effect": "the caveat is not a property of CMAZE-1 that wave 9 `inherits`; it is a property of the "
                  "REQUEST LAYER. The published tripwire survives intact where the harness draws fresh "
                  "instances, and is destroyed only by the hub's `unseen pair on a SEEN maze` request layer, "
                  "where 1,242 of 1,449 optimal routes pick up exactly one key and there is no key ORDER for "
                  "rule B to trap. Two different numbers, two different splits, and the paper must not print "
                  "one where the other belongs.",
        "provenance": ([{"source": f"harness_s{s}", "path": SOURCES[f"harness_s{s}"],
                         "pointer": "/columns/blind-sequential (key order from A)/validity"} for s in SEEDS]
                       + [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                           "pointer": "/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity"} for s in SEEDS]),
        "action": "T1 and T3 each print the value measured on their own split; figure 5 draws both against "
                  "the published 0.4980.",
    },
]

# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# 9b. w12r/artefact-mustfix -- the corrections THIS pass made, each against REFEREE-v0.md.
# --------------------------------------------------------------------------------------
CORRECTIONS += [
    {
        "id": "C5 (w12r/MF3)",
        "claim_as_written": "TABLE 4 (the lead result) printed EIGHT property rows on the w9 leg and "
                            "FIVE on the w8 leg, and NO BYTE ROW on either.",
        "artefact_says": "the w8 gate file records the overrun as named fields: `honest_total_bytes` "
                         "= 12,869,482 / 12,500,170 / 12,409,722 B and `times_over_budget` = 24.5466 / "
                         "23.8422 / 23.6697 against the 524,288 B pin. The w9 ledger records "
                         "`deployed_bytes` = 1,642,752 with `statistic_bytes` = 35,904 (2.19%) "
                         "declared beside it.",
        "effect": "the paper penalised the compose-and-forget object of section 5 at 7.48x the pin and "
                  "exempted its own lead result at 24.55x - three times worse - one section later. "
                  "A referee who opens the cited JSON reads `times_over_budget: 24.5466` in the file "
                  "cited for the headline.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": f"/per_seed/{s}/times_over_budget"} for s in SEEDS]
                      + [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                          "pointer": f"/per_seed/{s}/honest_total_bytes"} for s in SEEDS]
                      + [{"source": "maze_s0", "path": SOURCES["maze_s0"], "pointer": "/ledger"}],
        "action": "every table that carries a store now prints its FOUR-PART byte ledger as its own "
                  "block, never netted: T1 section 1c-quater, T2, T4. T3 carries no store and says so.",
    },
    {
        "id": "C6 (w12r/MF4)",
        "claim_as_written": "`exact` was used without a definition anywhere in the tables.",
        "artefact_says": "the frozen autoencoder's training inventory includes eval-bank items "
                         "(`ae_train_items`: eval_bank_train 543 / 529 / 552), and `refit` = False: "
                         "the encoder and decoder fitted with the deleted rows present are untouched.",
        "effect": "without the definition the claim reads as retrain-equivalence of the generative "
                  "model, which is not what was measured, and is the standard unlearning objection.",
        "provenance": [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"], "pointer": "/ae_train_items"}
                       for s in SEEDS]
                      + [{"source": "maze_s0", "path": SOURCES["maze_s0"],
                          "pointer": "/display/delete_green_exactness/refit"}],
        "action": "`/exactness_definition` is a top-level citable field (STORE-SCOPED, frozen encoder "
                  "and decoder never refit); T4's caption carries it; every place the word appears is "
                  "flagged, and the two senses (exact JUDGE vs exact DELETION) are separated.",
    },
    {
        "id": "C7 (w12r/MF7)",
        "claim_as_written": "TABLE 4's w8/downdate leg was `class RESULT`.",
        "artefact_says": "`/note`: \"No scored run and no claim: these are properties of the object "
                         "(deviations, byte counts, row counts), not outcomes. hub/single-design scores "
                         "the object.\" The hub commissioned to score it scored 0 of 3 (DECLARED "
                         "NOT-RUN NR3, full budget intact).",
        "effect": "a RESULT label on a leg nobody scored is the one thing in this table a referee can "
                  "call inflation. It costs the paper nothing to drop: two independent implementations "
                  "is still the claim.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/note"}],
        "action": "relabelled `ASSERTION BATTERY (object property) - 3 seeds - TEST condition - 0 "
                  "SCORED RUNS`. The artefact was NOT relabelled to match the table.",
    },
    {
        "id": "C8 (w12r/MF8 + MF9)",
        "claim_as_written": "T4 printed `/downdate_equals_refit/max_abs_dev_per_well_by_seed` "
                            "(2.365e-14 / 1.126e-09 / 7.105e-15) and declared ONE unresolvable leg.",
        "artefact_says": "`/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev` - the field the source "
                         "spoke's own FINAL REPORT prints, worst over the pinned 11,024-row deletion AND "
                         "the random-64 deletion - is 4.547e-13 / 1.126e-09 / 2.842e-14: 19x larger on "
                         "seed 0 and 4x larger on seed 2. And TWO legs sit below their own "
                         "reproducibility floor (seed 1: 1.126e-09 vs 4.282e-09; seed 2: 2.842e-14 vs "
                         "5.684e-14), not one.",
        "effect": "a correction whose whole point was to stop an understated figure being printed was "
                  "printing a different understated figure; and the paper was hiding two thirds of the "
                  "strongest fact it has, since an unresolvable residual is the best available outcome "
                  "for an exactness claim.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev"} for s in SEEDS]
                      + [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                          "pointer": f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev"} for s in SEEDS],
        "action": "T4 and T5 print the spoke's own row first, the narrower field beside it and labelled, "
                  "each leg's own floor, and a resolvable? row. Figure 4 plots the corrected field and "
                  "the floors.",
    },
    {
        "id": "C9 (w12r/MF10)",
        "claim_as_written": "TABLE 1 was labelled CMAZE-1, the same label as TABLE 3.",
        "artefact_says": "the w9/maze FINAL REPORT defines the scored environment as CMAZE-2, \"a "
                         "REQUEST layer over CMAZE-1. The engine, the solver and the exact integer judge "
                         "are `w8_envscout.maze_colour`, imported unmodified\". TABLE 3 is CMAZE-1 proper.",
        "effect": "under one label the triple 0.4980 (published CMAZE-1) / 0.5317 (CMAZE-1 at the "
                  "harness tier) / 0.9037 (CMAZE-2) is unreadable, and correction C4 - which says 0.4980 "
                  "is a property of the REQUEST LAYER, not of CMAZE-1 - contradicts itself on the page.",
        "provenance": [{"source": "maze_log", "path": SOURCES["maze_log"], "locator": "LOG.md:366-372"},
                       {"source": "maze_s0", "path": SOURCES["maze_s0"],
                        "pointer": "/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity"},
                       {"source": "harness_s0", "path": SOURCES["harness_s0"],
                        "pointer": "/columns/blind-sequential (key order from A)/validity"}],
        "action": "T1, its caption, its pins, figures 1 and 5 and every w9 maze row are relabelled "
                  "CMAZE-2; T3 is labelled CMAZE-1 PROPER; `/environments` defines both in one line "
                  "each; T3's caption no longer says `the same environment`; the two never share a table.",
    },
    {
        "id": "C10 (w12r/MF11)",
        "claim_as_written": "\"FRAGMENT-LOOKUP r=1/r=2 at 0.0000 everywhere\".",
        "artefact_says": "true of the CROSS-MAZE quantity (0.0006 / 0.0000) and false of the SAME-MAZE "
                         "quantity on w9's own split (0.0741 / 0.5105 pooled, 0.5114 as the unweighted "
                         "seed mean).",
        "effect": "the false half is also the paper's strongest exhibit: same-maze r=2 reaches 81% of "
                  "row 2's pipeline ceiling of 0.6319 where the CLU reaches 20%.",
        "provenance": [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                        "pointer": "/env_columns/fragment/FRAG-r2/validity"} for s in SEEDS]
                      + [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                          "pointer": "/env_columns/fragment_cross_maze/FRAG-cross-r2/validity"} for s in SEEDS],
        "action": "T1 section 1c-bis prints both, labelled, with the charter-B.3 not-like-for-like "
                  "caveat. T3's fragment rows are labelled CMAZE-1 tier columns.",
    },
    {
        "id": "C11 (w12r) - which mean is 0.5105?",
        "claim_as_written": "the same-maze FRAGMENT-LOOKUP r=2 figure is quoted as 0.5105 by "
                            "`w9-maze/LOG.md` and by REFEREE-v0.md MF11.",
        "artefact_says": "the three per-seed validities are 0.5342 / 0.5364 / 0.4636 at n = 584 / 604 / "
                         "632. The UNWEIGHTED seed mean is 0.51143; the n-WEIGHTED (pooled) mean is "
                         "0.51044, which rounds to the quoted 0.5105.",
        "effect": "0.0010 absolute. It changes nothing in the reading, but this file's convention "
                  "everywhere else is the unweighted per-seed mean, so the two numbers must not be "
                  "used interchangeably without saying which is which.",
        "provenance": [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                        "pointer": "/env_columns/fragment/FRAG-r2/validity"} for s in SEEDS]
                      + [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"],
                          "pointer": "/env_columns/fragment/FRAG-r2/n"} for s in SEEDS],
        "action": "T1 section 1c-bis prints the three per-seed values and the unweighted mean 0.5114; "
                  "the pooled 0.5105 is named here so the prose can cite either knowingly.",
    },
    {
        "id": "C12 (w12r) - the '9/9 seeds' phrasing",
        "claim_as_written": "\"1e-16...1e-14 on Shapes3D, 9/9 seeds, two independent implementations\".",
        "artefact_says": "the nine legs are 3 CONFIGURATIONS (ward_balanced K=40/44/48) x 3 SEEDS inside "
                         "ONE implementation. There are THREE implementations, and their bands are not "
                         "shared: 7.5e-16..8.8e-16 (w9/maze), 2.842e-14..1.126e-09 (w8 downdate-build), "
                         "5.551e-16..1.490e-13 (w8 single-design). And it was not unconditional: it "
                         "FAILED first on TEST seed 2 at 32.90 float32 ulp.",
        "effect": "three defects in one sentence. The strongest form of the two-implementation claim is "
                  "the FAILURE AGREEMENT - the second implementation reproduced the same seed-2 failure "
                  "with different code at 14.377 ulp against 32.90.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": "/a1_subspace/worst_in_float32_ulp"},
                       {"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": "/a1_subspace/worst_floor_subspace_in_float32_ulp"},
                       {"source": "single_design_log", "path": SOURCES["single_design_log"],
                        "locator": "LOG.md:472,479,481,495-503"},
                       {"source": "downdate_log", "path": SOURCES["downdate_log"],
                        "locator": "LOG.md:625-640,728,794,905"}],
        "action": "TABLE 5 prints the three implementations as three rows with seed count and "
                  "configuration count as SEPARATE fields, and makes the failure agreement a citable "
                  "row. `/seed_and_configuration_counts` is a top-level field. The phrase is on the "
                  "never-print list.",
    },
    {
        "id": "C13 (w12r/MF14)",
        "claim_as_written": "the charter rule \"every composition row carries BLIND-SEQUENTIAL measured "
                            "on its own split\", applied to all four rows.",
        "artefact_says": "`/env_columns` carries exactly four keys - `unseen_pair`, `nway`, `fragment`, "
                         "`fragment_cross_maze` - on every scored seed. There is no blind block for row "
                         "4 (budget interpolation) on any seed.",
        "effect": "the rule is unsatisfiable for row 4, and neither the tables nor the draft said so.",
        "provenance": [{"source": f"maze_s{s}", "path": SOURCES[f"maze_s{s}"], "pointer": "/env_columns"}
                       for s in SEEDS],
        "action": "T1 section 1c-ter declares it NOT MEASURED - not a blank and not a zero - and the "
                  "charter rule in `rules_applied` now carries the exception.",
    },
    {
        "id": "C14 (w12r/MF6)",
        "claim_as_written": "TABLE 4 carried no caption caveats.",
        "artefact_says": "two leaf fields of the cited gate file: `cut_is_marginal` = true on ALL THREE "
                         "seeds, and `a1_subspace.passed_every_seed` = false with `seeds_failing` = "
                         "[\"2\"] at 60.36282 float32 ulp against its own 66.11072 ulp floor. The "
                         "top-level `all_three_assertions_passed` = true EXCLUDES `a1_subspace`.",
        "effect": "a referee opening the paper's lead-result artefact reads `passed: false` inside it "
                  "and the paper says nothing. Stated with the floor beside it, the failure is a "
                  "property of the CONTENT and not of the downdate - which is what the artefact's own "
                  "`reproducibility_floor.what` field says.",
        "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/a1_subspace"},
                       {"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": "/cut_is_marginal_any_seed"},
                       {"source": "downdate_test", "path": SOURCES["downdate_test"],
                        "pointer": "/reproducibility_floor/what"}],
        "action": "both are printed under T4's w8 row and both are in T4's caption.",
    },
]

# 9.  FIGURES -- regenerated here, from the same pointers the tables use.
# --------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8b8a85"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d8d7d2", "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "figure.dpi": 160,
})


def recessive(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def fig1_main_scoreboard():
    """Two panels (small multiples, never a dual axis): row 1 sits near the ceiling and rows 2-4
    live below 0.2, so they are faceted onto their own scales.  Three seeds as dots on every bar."""
    arms = [("clu", "CLU (arm)", CAT[0]), ("mlp", "MLP", CAT[1]), ("lstm", "LSTM", CAT[2]),
            ("transformer", "Transformer", CAT[3]), ("lookup_matched", "lookup (matched bytes)", CAT[4])]
    panels = [([("row 1\nseen", "seen")], 1.12, "row 1 - seen request"),
              ([("row 2\nunseen pair", "unseen_pair"), ("row 3\nn-way (conj.)", "nway"),
                ("row 4\nbudget interp.", "budget_interp")], 0.70, "rows 2-4 - composition rows")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1, 2.6]})
    for ax, (rows, ymax, title) in zip(axes, panels):
        n = len(arms)
        w = 0.82 / n
        for j, (arm, label, colour) in enumerate(arms):
            means, per = [], []
            for _, rk in rows:
                v = [ptr(DOCS[sx], "/table/" + rk + "/" + arm + "/validity") for sx in MAZE]
                means.append(sum(v) / 3)
                per.append(v)
            xs = np.arange(len(rows)) + (j - (n - 1) / 2) * w
            ax.bar(xs, means, width=w * 0.86, color=colour,
                   label=label if title.startswith("row 1") else None,
                   zorder=3, edgecolor=SURFACE, linewidth=1.0)
            for x, vv in zip(xs, per):
                ax.plot([x] * 3, vv, marker="o", markersize=3.4, linestyle="none",
                        markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=0.9, zorder=5)
        for i, (_, rk) in enumerate(rows):
            c = sum(ptr(DOCS[sx], "/pipeline_ceiling_rows/" + rk + "/validity") for sx in MAZE) / 3
            ax.plot([i - 0.46, i + 0.46], [c, c], color=INK3, linewidth=1.6, linestyle=(0, (4, 2)), zorder=6)
            ax.annotate("ceiling %.3f" % c, (i, c), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7.2, color=INK2)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels([r[0] for r in rows])
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=9.5)
        recessive(ax)
    axes[0].set_ylabel("validity under the exact judge")
    axes[0].legend(frameon=False, ncol=5, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, -0.34))
    fig.suptitle("CMAZE-2 main scoreboard - 3 scored seeds, held-out TEST", fontsize=11,
                 fontweight="bold", y=1.02)
    fig.text(0.5, -0.12, "Dots are the three seeds; bars are their mean. Dashed line is the pipeline ceiling "
                         "(encode -> decode -> execute -> judge) on that row.\nK=8, A=31, ridge=0.01, 410,688 read "
                         "floats; baselines matched at 405,776-410,920 params. Panels have DIFFERENT y-scales.",
             ha="center", fontsize=7.2, color=INK2)
    fig.tight_layout()
    pth = FIG / "fig1_cmaze2_main_scoreboard.png"
    fig.savefig(pth, bbox_inches="tight")
    plt.close(fig)
    return pth


def fig2_nway_conjunction_vs_route():
    """The charter's comparability warning, drawn: the n-way row under both readings, each against
    the ceiling that is actually computed on its own denominator."""
    arms = [("clu", "CLU"), ("mlp", "MLP"), ("lstm", "LSTM"), ("transformer", "Transformer"),
            ("lookup_matched", "lookup\n(matched B)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), sharey=False)
    for ax, (mode, title, getter) in zip(axes, [
        ("conj", "scored as a CONJUNCTION", lambda s, a: ptr(DOCS[s], f"/table/nway/{a}/validity")),
        ("route", "scored PER ROUTE", lambda s, a: ptr(DOCS[s], f"/table/nway/{a}/per_route/validity")),
    ]):
        means = [sum(getter(s, a) for s in MAZE) / 3 for a, _ in arms]
        xs = np.arange(len(arms))
        ax.bar(xs, means, width=0.6, color=[CAT[0], CAT[1], CAT[1], CAT[1], CAT[4]], zorder=3,
               edgecolor=SURFACE, linewidth=1.0)
        for x, (a, _) in zip(xs, arms):
            ax.plot([x] * 3, [getter(s, a) for s in MAZE], marker="o", markersize=3.4, linestyle="none",
                    markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=0.9, zorder=5)
            ax.annotate(f"{sum(getter(s, a) for s in MAZE)/3:.4f}", (x, means[xs.tolist().index(x)]),
                        xytext=(0, 12), textcoords="offset points", ha="center", fontsize=7, color=INK2)
        ceil = sum(ptr(DOCS[s], "/pipeline_ceiling_rows/nway/validity") for s in MAZE) / 3
        n_row = [ptr(DOCS[s], "/table/nway/clu/n") for s in MAZE]
        n_ceil = [ptr(DOCS[s], "/pipeline_ceiling_rows/nway/n") for s in MAZE]
        ax.axhline(ceil, color=INK3, linewidth=1.6, linestyle=(0, (4, 2)), zorder=6)
        ax.annotate("pipeline ceiling %.4f   (n=%d/%d/%d, PER ROUTE)" % (ceil, n_ceil[0], n_ceil[1], n_ceil[2]),
                    (-0.45, ceil), xytext=(0, 4), textcoords="offset points", ha="left", fontsize=7.2, color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, l in arms], fontsize=8)
        ax.set_ylim(0, 0.60)
        nn = n_row if mode == "conj" else n_ceil
        ax.set_title("%s   (n = %d / %d / %d)" % (title, nn[0], nn[1], nn[2]), fontsize=9.5)
        recessive(ax)
    axes[0].set_ylabel("validity under the exact judge")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CAT[0], CAT[1], CAT[4])]
    axes[0].legend(handles, ["CLU (arm)", "learned baselines (matched params)", "lookup (matched bytes)"],
                   frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.90))
    fig.suptitle("Row 3 is scored as a conjunction; its ceiling is per route - both readings, or the row\n"
                 "is not comparable with its own ceiling", fontsize=10, fontweight="bold", y=1.06)
    fig.tight_layout()
    p = FIG / "fig2_nway_conjunction_vs_route.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig3_encoder_loop_decomposition():
    """A' / A'' / A per seed, both rows. The finding is that two real effects cancel."""
    cols = [("BAR_Aprime_loop_nosw_minus_alone", "A'  unfreezing\n(no swap loss)", CAT[1]),
            ("BAR_Adprime_swap_loss_alone", "A''  what the\nswap loss bought", CAT[2]),
            ("BAR_A_loop_swap_minus_alone", "A  the net", CAT[0])]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    for ax, (rk, rlabel) in zip(axes, [("r3_comp_2way", "row 3 - 2-way"), ("r4_comp_nway", "row 4 - n-way")]):
        for j, (key, label, colour) in enumerate(cols):
            vals = [ptr(ca, f"/per_seed/{s}/{rk}/encoder_comparison/{key}/mean_paired_delta") for s in SEEDS]
            xs = np.arange(3) + (j - 1) * 0.26
            ax.bar(xs, vals, width=0.23, color=colour, label=label if rk == "r3_comp_2way" else None,
                   zorder=3, edgecolor=SURFACE, linewidth=1.0)
            for x, v, s in zip(xs, vals, SEEDS):
                pv = ptr(ca, f"/per_seed/{s}/{rk}/encoder_comparison/{key}/sign_test_p_two_sided")
                if pv <= 0.0013:
                    ax.annotate("*", (x, v), xytext=(0, 4 if v >= 0 else -11), textcoords="offset points",
                                ha="center", fontsize=11, color=INK)
        ax.axhline(0, color=INK, linewidth=1.0, zorder=4)
        ax.axhline(1.0, color=INK3, linewidth=1.4, linestyle=(0, (4, 2)), zorder=4)
        ax.annotate("BAR A margin +1.0", (-0.45, 1.0), xytext=(0, 4), textcoords="offset points",
                    ha="left", fontsize=7.2, color=INK2)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels([f"seed {s}" for s in SEEDS])
        ax.set_title(rlabel)
        recessive(ax)
    axes[0].set_ylabel("paired delta, distinct valid novel tuples")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.12))
    fig.suptitle("The store does not force better features - and the reason is that two real effects cancel\n"
                 "Shapes3D, w8_v1_short decoder, joint_steps=400, n_train=8192.  * = sign-test p <= 0.0013",
                 fontsize=9.5, fontweight="bold", y=1.10)
    fig.tight_layout()
    p = FIG / "fig3_encoder_loop_decomposition.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig4_exact_deletion():
    """Both implementations, log scale, against the float64 eps reference line."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    maze_stat = [ptr(DOCS[s], "/display/delete_green_exactness/statistic_max_rel_residual") for s in MAZE]
    # w12r/MF8: the field the SOURCE SPOKE's own FINAL REPORT prints - worst over the pinned
    # 11,024-row deletion AND the random-64 deletion - not the narrower per-well-by-seed field.
    dd_abs = [ptr(dd, f"/per_seed/{s}/downdate_equals_refit/worst_max_abs_dev") for s in SEEDS]
    dd_floor = [ptr(dd, f"/per_seed/{s}/reproducibility_floor/floor_max_abs_dev") for s in SEEDS]
    xs = np.arange(3)
    ax.bar(xs - 0.18, maze_stat, width=0.33, color=CAT[0], zorder=3, edgecolor=SURFACE, linewidth=1.0,
           label="w9/maze  statistic max RELATIVE residual")
    ax.bar(xs + 0.18, dd_abs, width=0.33, color=CAT[1], zorder=3, edgecolor=SURFACE, linewidth=1.0,
           label="w8/downdate  max ABSOLUTE deviation (pinned + random-64)")
    ax.scatter(xs + 0.18, dd_floor, marker="_", s=260, color=INK, zorder=6, linewidths=1.8,
               label="w8/downdate  its own reproducibility floor")
    for x, v in zip(xs - 0.18, maze_stat):
        ax.annotate(f"{v:.2e}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.8, color=INK2)
    for x, v in zip(xs + 0.18, dd_abs):
        ax.annotate(f"{v:.2e}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.8, color=INK2)
    ax.axhline(2.220446049250313e-16, color=INK3, linewidth=1.4, linestyle=(0, (4, 2)), zorder=4)
    ax.annotate("float64 eps = 2.22e-16", (-0.45, 2.220446049250313e-16), xytext=(0, -11),
                textcoords="offset points", ha="left", fontsize=7.2, color=INK2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6))
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("deviation, downdate vs refit-without-those-rows")
    ax.set_title("Exact deletion (STORE-SCOPED) - two implementations, 3 seeds each, TEST")
    recessive(ax)
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.06))
    fig.text(0.5, -0.06,
             "EXACT = STORE-SCOPED: the store content equals the content fitted from the retained rows alone, "
             "holding the FROZEN encoder and decoder fixed. Neither is refit.\n"
             "Different units: w9/maze reports a relative residual on the sufficient statistic "
             "(projector residual 0.0 on every seed); w8/downdate reports an absolute max deviation.\n"
             "On seeds 1 and 2 the w8 deviation sits BELOW its own reproducibility floor (the dash), so there the "
             "residual is not resolvable at all. Both: 0 refits, 0 passes over the data, items never kept.",
             ha="center", fontsize=6.8, color=INK2)
    fig.tight_layout()
    p = FIG / "fig4_exact_deletion.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig5_blind_sequential():
    """The caveat column that must travel with every CMAZE-2 composition row."""
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    up = [ptr(DOCS[s], "/env_columns/unseen_pair/blind/BLIND-SEQUENTIAL(key order from A)/validity") for s in MAZE]
    nw = [ptr(DOCS[s], "/env_columns/nway/blind/BLIND-SEQUENTIAL(key order from A)/validity") for s in MAZE]
    xs = np.arange(3)
    ax.bar(xs - 0.18, up, width=0.33, color=CAT[0], zorder=3, edgecolor=SURFACE, linewidth=1.0,
           label="row 2 - unseen pair split")
    ax.bar(xs + 0.18, nw, width=0.33, color=CAT[1], zorder=3, edgecolor=SURFACE, linewidth=1.0,
           label="row 3 - n-way split")
    for x, v in list(zip(xs - 0.18, up)) + list(zip(xs + 0.18, nw)):
        ax.annotate(f"{v:.4f}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7.2, color=INK2)
    ax.axhline(0.4980, color=CAT[7], linewidth=1.8, linestyle=(0, (4, 2)), zorder=6)
    ax.annotate("published CMAZE-1 BLIND-SEQUENTIAL 0.4980 - NOT the value on this split",
                (-0.45, 0.4980), xytext=(0, 5), textcoords="offset points", ha="left", fontsize=7.4,
                color=CAT[7], bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("composition-blind validity")
    ax.set_title("Every CMAZE-2 composition row carries BLIND-SEQUENTIAL measured on ITS OWN split")
    recessive(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.0, -0.12))
    fig.text(0.5, -0.19,
             "Measured cause (w9/maze LOG SS3): 1,242 of 1,449 optimal routes pick up exactly ONE key, so there is "
             "no key ORDER for rule B to trap.\nThis is a weakness of the REQUEST LAYER, not of CMAZE-1.",
             ha="center", fontsize=7.0, color=INK2)
    fig.tight_layout()
    p = FIG / "fig5_blind_sequential_on_this_split.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


FIG_SPECS = [
    ("fig1", fig1_main_scoreboard, "T1 RESULT rows 1-4: arm, learned baselines and the matched-bytes floor, "
                                   "3 seeds as dots, pipeline ceiling per row",
     ["maze_s0", "maze_s1", "maze_s2"],
     ["/table/{row}/{arm}/validity", "/pipeline_ceiling_rows/{row}/validity"]),
    ("fig2", fig2_nway_conjunction_vs_route, "Row 3 under both readings against the per-route ceiling",
     ["maze_s0", "maze_s1", "maze_s2"],
     ["/table/nway/{arm}/validity", "/table/nway/{arm}/per_route/validity",
      "/pipeline_ceiling_rows/nway/validity", "/pipeline_ceiling_rows/nway/n", "/table/nway/clu/n"]),
    ("fig3", fig3_encoder_loop_decomposition, "T2: A' / A'' / A per seed on rows 3 and 4",
     ["enc_crossarm"],
     ["/per_seed/{seed}/{row}/encoder_comparison/{bar}/mean_paired_delta",
      "/per_seed/{seed}/{row}/encoder_comparison/{bar}/sign_test_p_two_sided"]),
    ("fig4", fig4_exact_deletion, "T4: both deletion implementations, log scale",
     ["maze_s0", "maze_s1", "maze_s2", "downdate_test"],
     ["/display/delete_green_exactness/statistic_max_rel_residual",
      "/downdate_equals_refit/max_abs_dev_per_well_by_seed/{seed}"]),
    ("fig5", fig5_blind_sequential, "T1 caveat column: BLIND-SEQUENTIAL on this split vs the published 0.4980",
     ["maze_s0", "maze_s1", "maze_s2", "envscout_defaults"],
     ["/env_columns/{row}/blind/BLIND-SEQUENTIAL(key order from A)/validity"]),
]

# --------------------------------------------------------------------------------------
# 10. Render.
# --------------------------------------------------------------------------------------
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)



# --------------------------------------------------------------------------------------
# 10b. TABLES.md -- rendered from the SAME manifest, so the prose and the JSON cannot drift.
# --------------------------------------------------------------------------------------
def _pin_line(pins):
    parts = []
    for k, v in pins.items():
        if isinstance(v, dict):
            v = ", ".join(f"{a} {b:,}" if isinstance(b, int) else f"{a} {b}" for a, b in v.items())
        elif isinstance(v, int) and abs(v) >= 1000:
            v = f"{v:,}"
        parts.append(f"`{k}` = {v}")
    return " · ".join(parts)


def _cells_table(rows, cell_key="cells", extra_first=None):
    cols = []
    for r in rows:
        for c in r[cell_key]:
            if c not in cols and not c.startswith("_"):
                cols.append(c)
    head = ["row", "class"] + (["n (s0/s1/s2)"] if extra_first else []) + cols
    body = []
    for r in rows:
        line = [r["label"], "**" + r["row_class"] + "**"]
        if extra_first:
            n = r.get("n_scored")
            line.append(" / ".join(f"{v:.0f}" for v in n["values"]) if n else "—")
        for c in cols:
            line.append(fmt_cell(r[cell_key][c]) if c in r[cell_key] else "—")
        body.append(line)
    return md_table(head, body)


def _ledger_md(led):
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
        out.append("> The artefact's own words: *\"" +
                   led["the_sentence_the_artefact_itself_writes"]["text"] + "\"*")
        out.append("")
    return out


def _traced(v):
    """One `traced to` string for a cell, a derived number or a markdown locator."""
    if "printed" in v:
        return f"`{v['path']}` {v['lines']}"
    prov = v.get("provenance") or v.get("from") or []
    if not prov:
        return ""
    paths = sorted({e["path"] for e in prov})
    ptrs = []
    for e in prov:
        q = e.get("pointer") or e.get("locator", "")
        if q not in ptrs:
            ptrs.append(q)
    shown = ptrs[0] if len(ptrs) == 1 else ptrs[0] + " … one entry per seed"
    return "; ".join(f"`{q}`" for q in paths) + f" `{shown}`"


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


def render_md(manifest):
    L = []
    A = L.append
    A("# TABLES.md — the paper's scoreboard, every row at 3 seeds on generation 2")
    A("")
    A(f"**Built** {manifest['generated_utc']} by `build_tables.py` in this directory. "
      "**Scored runs spent by this spoke: ZERO** — it re-reads artefacts only.")
    A("")
    A("**How to trace any cell.** Every number below is written `mean (seed0 / seed1 / seed2)`. "
      "`TABLES.json` carries the same cell with a `provenance` list — one `{source, path, pointer}` "
      "entry per seed — and `sources` maps every source id to its absolute path, byte count and "
      "sha256. `python build_tables.py` regenerates this file, `TABLES.json` and every figure from "
      "those paths alone.")
    A("")
    A("**The rules this file is built under** (from the w10 charter):")
    for r in manifest["rules_applied"]:
        A(f"- {r}")
    A("")
    A("---")
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

    # ---- T1 -------------------------------------------------------------------------
    t1 = manifest["tables"]["T1"]
    A(f"## TABLE 1 (MAIN) — {t1['title']}")
    A("")
    A(f"**Pins.** {_pin_line(t1['pins'])}")
    A("")
    A(f"*{t1['caption']}*")
    A("")
    A("### 1a. RESULT rows — validity under the exact judge")
    A("")
    A(_cells_table(t1["result_rows"], extra_first=True))
    A("")
    A("**Column classes.** Every column above is one of three things, and the charter requires the "
      "label to be printed, not inferred:")
    A("")
    A(md_table(["column", "class", "what it is"],
               [[k, "**" + v["class"] + "**", v["what"]] for k, v in t1["column_classes"].items()]))
    A("")
    A("> ⛔ **BAR 3 (the C.2 controls) FAILS and that is the mechanism of this table.** On row 1 "
      "`landscape_shuffled` reads 0.9509 and `random_atoms` 0.9601 against the CLU's 1.0000; on row 2 "
      "`random_atoms` (0.1414) is inside the seed spread of the CLU (0.1279). Only `landscape_removed` "
      "collapses to 0.0000. ⛔ **Under this hub's own pre-registration no RESULT row is claimed for the "
      "CLU arm on CMAZE-2.** BAR 1 also fails on all three composition rows: against the best learned "
      "baseline the CLU is behind by -0.0560 (row 2), -0.0209 (row 3 conjunction) and -0.0434 (row 4). "
      "Traced to `/c/chlu/outputs/w9-maze/LOG.md` §5.")
    A("")
    A("### 1b. Ceilings printed beside every RESULT row")
    A("")
    ceil_cols = ["representation", "pipeline (encode->decode->execute->judge)", "store span"]
    rows = []
    for r in t1["result_rows"]:
        rows.append([r["label"]]
                    + [fmt_cell(r["ceilings"][c]) for c in ceil_cols]
                    + [" / ".join(f"{v:.0f}" for v in r["ceilings"]["_ceiling_n"]["values"])]
                    + [" / ".join(f"{v:.0f}" for v in r["n_scored"]["values"])])
    A(md_table(["row"] + ceil_cols + ["ceiling n (s0/s1/s2)", "row n (s0/s1/s2)"], rows))
    A("")
    A("> ⚠ **Row 3's denominators differ.** The scored n-way row is a CONJUNCTION over the routes of "
      "one request (n = 183 / 192 / 191); its ceilings are computed PER ROUTE (n = 389 / 403 / 414). "
      "The conjunction reading and the per-route reading are both printed above, and only the "
      "per-route reading is comparable with the printed ceiling. See figure 2.")
    A("")
    A("### 1c. Caveat columns, measured on THIS split — they travel with every composition row")
    A("")
    cav_rows = []
    for r in t1["result_rows"]:
        cav = r.get("caveat_columns_measured_on_THIS_split")
        if not cav:
            continue
        cav_rows.append([r["label"]] + [fmt_cell(cav[c]) for c in cav])
    cav_cols = list(t1["result_rows"][1]["caveat_columns_measured_on_THIS_split"].keys())
    A(md_table(["row"] + cav_cols, cav_rows))
    A("")
    A("> ⛔ **BLIND-SEQUENTIAL on this split is 0.9077 / 0.8905 / 0.9128 (mean 0.9037) on row 2 and "
      "0.9229 / 0.8933 / 0.8841 (mean 0.9001) on row 3 — NOT the published CMAZE-1 0.4980.** "
      "Measured cause (w9/maze LOG §3): 1,242 of 1,449 optimal routes pick up exactly one key, so "
      "there is no key ORDER for rule B to trap. This is a property of the REQUEST LAYER, not of "
      "CMAZE-1. Published-value provenance: `/c/chlu/outputs/w8-env-scout/screen_results_DEFAULTS.txt`. "
      "See figure 5.")
    A("")
    A("### 1c-bis. ⛔ FRAGMENT-LOOKUP — same-maze and cross-maze are DIFFERENT QUANTITIES")
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
    A(f"> ⚠ **Which mean?** {fr['which_mean']}")
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
    A("### 1d. DISPLAY rows — inductive bias; the only comparator is the supervised refuser")
    A("")
    A("**Refusal.** The CLU has never seen an impossible example; the supervised refuser is trained on "
      "552 / 562 / 522 of them (seeds 0/1/2) and is evaluated on held-out mazes 48..95 only, so the like-for-like CLU column is "
      "the held-out one. Lower is better on the grounded-possible row (those are FALSE refusals).")
    A("")
    ref_rows = [r for r in t1["display_rows"] if r["label"].startswith("refusal - ")]
    A(_cells_table(ref_rows, extra_first=True))
    A("")
    A("> ⛔ **THE READING OF THESE THREE ROWS, which the registry filed and the draft dropped "
      "(w12r/MF15, registry `N-I8-19`, tier A).** *Refusal is NOVELTY DETECTION, not wall detection.* "
      "The store refuses the **reachable-but-unwritten** combination (`mt_unseen`, **0.9781** = "
      "0.9721 / 0.9794 / 0.9829) as often as the **walled** one (`walled`, **0.9801** = 0.9606 / "
      "0.9846 / 0.9952) — both rows are in the table above — and its false-refusal rate on "
      "grounded-possible requests is **0.8182 against a pre-registered 0.20**, a 4x miss. The "
      "supervised refuser, which HAS seen impossible examples, separates better. ⛔ **No refusal win "
      "is claimed.** Printing the three DISPLAY rows without this reading is what the registry asked "
      "the paper not to do. *Traced to:* "
      "`/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/display/refusal/*/refusal` and "
      "`/display/clu_refusal_rate_heldout_mazes/*/refusal`.")
    A("")
    A("**Post-deletion refusal.** Two different operations, kept apart: the CONTENT deletion removes "
      "every item written under the word from the sufficient statistic (this is the one that carries "
      "the exactness claim, Table 4); the INTERFACE deletion removes the action word from the emission "
      "interface and touches no atom.")
    A("")
    for r in t1["display_rows"]:
        if not r["label"].startswith("post-deletion refusal"):
            continue
        A(f"*{r['label']}*  ·  class **{r['row_class']}**  ·  n = "
          + " / ".join(f"{v:.0f}" for v in r["n_scored"]["values"]))
        A("")
        A(md_table(["column", "mean (seed0 / seed1 / seed2)"],
                   [[k, fmt_cell(v)] for k, v in r["cells"].items()]))
        A("")
    A("**Post-deletion composition without retraining.**")
    A("")
    for r in t1["display_rows"]:
        if not r["label"].startswith("post-deletion composition"):
            continue
        A(f"*{r['label']}*  ·  class **{r['row_class']}**  ·  n = "
          + " / ".join(f"{v:.0f}" for v in r["n_scored"]["values"]))
        A("")
        A(md_table(["column", "mean (seed0 / seed1 / seed2)"],
                   [[k, fmt_cell(v)] for k, v in r["cells"].items()]))
        A("")
    A("> ⚠ **This row is the AS-PRE-REGISTERED number and it FAILS.** The w8 basis-change carry "
      "`W_k <- W_k R_k` is stated for a read expressed in the orthonormal basis `U_k`, while this "
      "hub's read is expressed in the deployed modes `G_k`, so the carry is the wrong change of "
      "coordinates here. Re-measured on VAL seed 0 with the repaired mechanism "
      "(`arms.CluRead.delete_rows_exact`, which downdates the head's own sufficient statistics "
      "`A = Phi^T Phi` and `Mx = Phi^T X`): surviving targets **0.0944 -> 0.0823, delta -0.0120**, "
      "inside BAR 6's +-0.05. ⛔ That is VAL, seed 0, ONE measurement — the TEST version at 3 seeds "
      "is **DECLARED NOT-RUN NR5**, not a row. Provenance: "
      "`/c/chlu/outputs/w9-maze/runs/amend_exactdelete_seed0_val.json` "
      "`/display/delete_green/composition_without_retraining/validity_delta_on_other_targets`.")
    A("")

    A("### 1e. Instrument audits that make the rows readable at all")
    A("")
    A(md_table(["audit", "result", "traced to"], [
        ["judge exact in all four parts (legal moves / reaches target / within budget / pickup rules)",
         "**True**", "`/c/chlu/outputs/w9-harness/harness_seed0.json` `/judge_audit/exact`"],
        ["tolerance used anywhere in scoring", "**None**",
         "`/c/chlu/outputs/w9-harness/RESULTS.md` audit table"],
        ["stepper disagreements with the judge", "**0 / 750 routes**",
         "`/c/chlu/outputs/w9-harness/harness_seed{0,1,2}.json` `/stepper_audit/disagreements`"],
        ["impossible examples seen by a non-refuser arm", "**0**",
         "`/c/chlu/outputs/w9-harness/RESULTS.md` audit table"],
        ["read is request-only (no feature argument; bitwise stable under feature perturbation)",
         "**True, 8 checked per seed**",
         "`/c/chlu/outputs/w9-maze/runs/scored1_seed0_test.json` `/request_only_assertion`"],
        ["plan-map collisions (representation ceiling)", "**0 on every row, every seed**",
         "`/c/chlu/outputs/w9-maze/runs/scored{1_seed0,2_seed1,3_seed2}_test.json` `/representation_ceiling/*/collisions`"],
        ["independent replication of scored run 1 (clean checkout, encoder retrained from scratch)",
         "**zero numeric differences**",
         "`/c/chlu/outputs/w9-maze/replication/REPLICATION.md`; `/c/chlu/outputs/w9-maze/LEDGER.md` row R1"],
    ]))
    A("")

    # ---- T2 -------------------------------------------------------------------------
    t2 = manifest["tables"]["T2"]
    A("---")
    A("")
    A(f"## TABLE 2 — {t2['title']}")
    A("")
    A(f"**Pins.** {_pin_line(t2['pins'])}")
    A("")
    A(f"*{t2['caption']}*")
    A("")
    A("> ⛔ **This table does not share a table with Table 1, nor with the incumbent's row-3/row-4 "
      "values.** Different cell, different metric, different compute pin, and a shortened decoder "
      "recipe (`w8_v1_short`) — a different instrument, not a different setting.")
    A("")
    for r in t2["rows"]:
        A(f"**{r['label']}**  ·  class **{r['row_class']}**")
        A("")
        A(md_table(["column", "mean (seed0 / seed1 / seed2)"],
                   [[k, fmt_cell(v)] for k, v in r["cells"].items()]))
        if "note_in_artefact" in r:
            A("")
            A(f"> {r['note_in_artefact']}")
        A("")
    v = t2["verdict"]
    A(md_table(["bar", "row 3", "row 4"],
               [["BAR A passed (mean paired delta > +1.0 and CI95 low > +1.0, all three seeds)",
                 v["BAR_A_PASS"]["r3"], v["BAR_A_PASS"]["r4"]],
                ["BAR B falsifier fired (delta <= 0 on two or more seeds)",
                 v["BAR_B_FALSIFIED"]["r3"], v["BAR_B_FALSIFIED"]["r4"]]]))
    A("")
    for line in _ledger_md(t2["byte_ledger"]):
        A(line)
    A("")
    A("> **The reading.** The net is a wash because two real effects cancel: unfreezing the encoder "
      "with the store on the gradient path HURTS (A′ negative on 5 of 6 seed-rows, two at p ≤ 0.0013) "
      "and the swap loss REPAIRS that damage (A″ positive on 4 of 6 seed-rows, two at p ≤ 0.0003). "
      "See figure 3.")
    A("")

    # ---- T3 -------------------------------------------------------------------------
    t3 = manifest["tables"]["T3"]
    A("---")
    A("")
    A(f"## TABLE 3 — {t3['title']}")
    A("")
    A(f"**Pins.** {_pin_line(t3['pins'])}")
    A("")
    A(f"*{t3['caption']}*")
    A("")
    A(_cells_table(t3["rows"], extra_first=True))
    A("")
    A("> ⭐ **The published tripwire reproduces HERE.** BLIND-SEQUENTIAL is **0.5317 (0.5550 / 0.5300 / "
      "0.5100)** at this tier against the published CMAZE-1 **0.4980**, and FRAGMENT-LOOKUP r=1 and r=2 are "
      "both **0.0000** on every seed. Compare Table 1, where the same column reads **0.9037** on the hub's "
      "unseen-pair split (CMAZE-2). ⛔ The difference is the REQUEST LAYER — which IS the difference "
      "between the two environments, CMAZE-1 here and CMAZE-2 there. See correction C4 and C9, and "
      "⛔ never merge these two tables.")
    A("")
    for line in _ledger_md(t3["byte_ledger"]):
        A(line)
    A("")
    A("**DISPLAY comparator at this tier**")
    A("")
    A(_cells_table(t3["display_rows"]))
    A("")

    # ---- T4 -------------------------------------------------------------------------
    t4 = manifest["tables"]["T4"]
    A("---")
    A("")
    A(f"## TABLE 4 — {t4['title']}")
    A("")
    A(f"*{t4['caption']}*")
    A("")
    for r in t4["rows"]:
        A(f"**{r['label']}**  ·  class **{r['row_class']}**  ·  implementation: {r['implementation']}")
        A("")
        A(md_table(["property", "seed 0 / seed 1 / seed 2"],
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
        if "summary_from_artefact" in r:
            sm = r["summary_from_artefact"]
            A("")
            A(f"> Artefact's own summary: `passed_every_seed` = {sm['passed_every_seed']} · "
              f"`worst_max_abs_dev_over_seeds` = {sm['worst_max_abs_dev_over_seeds']:.4e} · "
              f"`worst_in_float32_ulp_at_unit_scale` = {sm['worst_in_float32_ulp_at_unit_scale']:.6f}")
        A("")
    A("### TABLE 4 — THE BYTE LEDGER, four parts, printed, never netted (w12r/MF3)")
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
                   [[k, fmt_any(v), _traced(v)] for k, v in cc.items()]))
        A("")
        A(f"> {r['note']}")
        A("")
    fa = t5["failure_agreement"]
    A(f"### ⭐⭐ {fa['title']}")
    A("")
    A(f"> {fa['what_happened']}")
    A("")
    A(md_table(["leg", "float32 ulp at unit scale", "traced to"],
               [[k, fmt_any(v), _traced(v)] for k, v in fa["legs"].items()]))
    A("")
    A(f"> ⭐⭐ **The citable sentence.** {fa['the_citable_sentence']}")
    A("")
    A(f"> **How it was closed.** {fa['how_it_was_closed']['fix']}, "
      f"`tau` = {fa['how_it_was_closed']['tau']:.6e} "
      f"({fa['how_it_was_closed']['tau_derivation']}).")
    A("")
    A(f"> ⛔⛔ **What must never be printed.** {fa['what_must_never_be_printed']}")
    A("")

    # ---- T6 ---------------------------------------------------------------------------
    t6 = manifest["tables"]["T6"]
    A("---")
    A("")
    A(f"## TABLE 6 — {t6['title']}")
    A("")
    A(f"**Pins.** {_pin_line(t6['pins'])}")
    A("")
    A(f"*{t6['caption']}*")
    A("")
    for r in t6["rows"]:
        A(f"**{r['label']}**  ·  class **{r['row_class']}**")
        A("")
        A(md_table(["quantity", "seed 0 / seed 1 / seed 2", "traced to"],
                   [[k, fmt_any(v), _traced(v)] for k, v in r["cells"].items()]))
        A("")
    c = t6["the_correction_this_table_carries"]
    A(f"> ⛔ **{c['id']}.** The draft printed *{c['what_the_draft_printed']}*. {c['why_it_is_wrong']} "
      f"The scored TEST legs say **{c['what_the_test_legs_say']}**.")
    A("")
    A(f"> The source spoke's own sentence: *\"{c['the_spokes_own_sentence']}\"*")
    A("")
    lad = t6["ceiling_ladder_address_books"]
    A("**Ceiling ladder — address-book bytes per rung (SF2)**")
    A("")
    A(md_table(["rung", "class", "address book bytes", "store bytes"],
               [[k, v["class"], fmt_any(v["address_bytes"]),
                 (fmt_any(v["store_bytes"]) if v["store_bytes"] else "—")]
                for k, v in lad["rungs"].items()]))
    A("")
    A(f"> ⚠ {lad['w12r_note']}")
    A("")

    # ---- NOT RUN ----------------------------------------------------------------------
    A("---")
    A("")
    A("## DECLARED NOT-RUN register — never a blank, never a zero")
    A("")
    for nr in manifest["declared_not_run"]:
        A(f"### {nr['id']} — {nr['row']}")
        A("")
        A(f"**Status:** {nr['status']}")
        A("")
        if "what_exists" in nr:
            A(f"**What does exist:** {nr['what_exists']}")
            A("")
        A(f"**Why:** {nr['why']}")
        A("")
        if "why_w12r_qualifier" in nr:
            A(f"> ⛔ **w12r qualifier.** {nr['why_w12r_qualifier']}")
            A("")
            A("> *Traced to:* " + "; ".join(
                f"`{e['path']}` {e.get('locator') or e.get('pointer')}"
                for e in nr["why_w12r_qualifier_provenance"]))
            A("")
        A("**Evidence:**")
        for e in nr["evidence"]:
            loc = e.get("pointer") or e.get("locator", "")
            A(f"- `{e['path']}` — {loc}")
        A("")
        A("**Launch line:**")
        A("")
        A("```bash")
        A(nr["launch_line"])
        A("```")
        A("")
        A(f"**Cost:** {nr['cost_note']}")
        A("")

    # ---- CORRECTIONS ------------------------------------------------------------------
    A("---")
    A("")
    A("## CORRECTIONS found while tracing every cell")
    A("")
    A("Each is a place where a narrative number in a LOG, a report or the task file does not match "
      "the artefact it is drawn from. None of them changes a verdict; all of them change a number "
      "the paper would otherwise print.")
    A("")
    for c in manifest["corrections"]:
        A(f"### {c['id']}")
        A("")
        A(f"- **As written:** {c['claim_as_written']}")
        A(f"- **Artefact says:** {c['artefact_says']}")
        A(f"- **Effect:** {c['effect']}")
        A(f"- **What this file does:** {c['action']}")
        A(f"- **Traced to:** " + "; ".join(
            f"`{pv['path']}` `{pv.get('pointer') or pv.get('locator')}`" for pv in c["provenance"][:3])
          + (" …" if len(c["provenance"]) > 3 else ""))
        A("")

    # ---- FIGURES ----------------------------------------------------------------------
    A("---")
    A("")
    A("## FIGURES — all regenerated from the manifest by `python build_tables.py`")
    A("")
    A(md_table(["figure", "file", "what it shows", "sources"],
               [[fid, f"`{f['file']}`", f["description"],
                 "<br>".join(f"`{x['path']}`" for x in f["sources"])]
                for fid, f in manifest["figures"].items()]))
    A("")

    # ---- SOURCES ----------------------------------------------------------------------
    A("---")
    A("")
    A("## SOURCE REGISTER — every file this spoke read")
    A("")
    A(md_table(["id", "path", "bytes", "sha256 (first 16)"],
               [[sid, f"`{v['path']}`", f"{v['bytes']:,}" if v["bytes"] else "MISSING",
                 (v["sha256"] or "")[:16]] for sid, v in manifest["sources"].items()]))
    A("")
    A("⛔ **Nothing outside this register was read**, and in particular `$CHLU_CLAUDE_ROOT` was not "
      "read at any point — it is a frozen older snapshot.")
    A("")
    return "\n".join(L) + "\n"


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # figures first, so the manifest can carry their hashes
    fig_manifest = {}
    for fid, fn, desc, srcs, pointers in FIG_SPECS:
        p = fn()
        fig_manifest[fid] = {
            "file": f"figures/{p.name}",
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
            "description": desc,
            "sources": [{"source": s, "path": SOURCES[s]} for s in srcs],
            "pointers_used": pointers,
            "regenerate": f"python build_tables.py    # rebuilds {p.name} from the sources above",
        }
        print(f"[fig] {p}")

    src_manifest = {}
    for sid, path in SOURCES.items():
        pp = native(path)
        src_manifest[sid] = {
            "path": path,
            "exists": pp.exists(),
            "sha256": sha256(pp) if pp.exists() else None,
            "bytes": pp.stat().st_size if pp.exists() else None,
        }

    manifest = {
        "schema": "chlu-w10-seeds-and-tables/1",
        "spoke": "w10 spoke/seeds-and-tables",
        "generated_utc": now,
        "builder": "build_tables.py (in this directory)",
        "acceptance_criterion": (
            "Every cell in TABLES.md traces to a file path recorded here: each cell carries `provenance`, a list "
            "of {source, path, pointer} - one entry per seed. A reader can regenerate every figure by running "
            "`python build_tables.py` against the paths in `sources`."
        ),
        "rules_applied": [
            "All three seeds or none - no row reports a best-of.",
            "Runs at different compute budgets do not share a table - T1/T2/T3/T4 are four pins.",
            "w8/encoder-loop absolutes (shortened `w8_v1_short` decoder) do not share a table with the incumbent's.",
            "CMAZE-2's n-way row is reported as a CONJUNCTION and PER ROUTE, because its ceilings are per route.",
            "Every CMAZE-2 composition row carries BLIND-SEQUENTIAL measured on its own split, not the published 0.4980; where no BLIND block exists (row 4) that is DECLARED, never blank.",
            "w12r/MF10: T1 is CMAZE-2 (a request layer over CMAZE-1) and T3 is CMAZE-1 proper. They are a DIFFERENT ENVIRONMENT, not merely a different budget, and they never share a table.",
            "w12r/MF3: every table that carries a store prints its FOUR-PART byte ledger - atoms, covariance/count ledger, address book, coupling/hole table - separately, NEVER netted into one number.",
            "w12r/MF4: `exact` is STORE-SCOPED and the frozen encoder and decoder are never refit; the definition is a citable field at TABLES.json /exactness_definition.",
            "w12r: never `9/9 seeds`. Seed count and configuration count are separate fields, and the three implementations are three rows (T5).",
            "A row that was not run is a DECLARED NOT-RUN with a reason and a launch line, never a blank or a zero.",
        ],
        "exactness_definition": EXACTNESS_DEFINITION,
        "environments": ENVIRONMENTS,
        "seed_and_configuration_counts": SEED_AND_CONFIGURATION_COUNTS,
        "w12r_revision": {
            "spoke": "w12r spoke/artefact-mustfix",
            "closes": ["MF3 (the byte row)", "MF4 (define exact)", "MF6 (the two gate caveats)",
                       "MF7 (the mislabelled row)", "MF8 (the field the source spoke reports)",
                       "MF9 (both unresolvable legs)", "MF10 (CMAZE-1 vs CMAZE-2)",
                       "MF11 (same-maze vs cross-maze fragment)", "MF14 (row 4's declared BLIND gap)",
                       "the '9/9 seeds' phrasing (T5)"],
            "scored_runs_spent": 0,
            "referee": "/c/chlu/evidence/outputs/w10-referee/REFEREE-v0.md",
        },
        "sources": src_manifest,
        "tables": TABLES,
        "figures": fig_manifest,
        "declared_not_run": NOT_RUN,
        "corrections": CORRECTIONS,
    }
    (OUT / "TABLES.json").write_text(json.dumps(manifest, indent=1, sort_keys=False), encoding="utf-8")
    print(f"[json] {OUT/'TABLES.json'}")
    (OUT / "TABLES.md").write_text(render_md(manifest), encoding="utf-8")
    print(f"[md]   {OUT/'TABLES.md'}")
    return manifest


if __name__ == "__main__":
    main()
