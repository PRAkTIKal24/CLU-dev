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
    # --- w13: WAVE 12's rows.  Added by w13 spoke/seeds-and-tables. -------------------
    # T7 / T9 / T10 / T12: hub/single-design-crossover (K = WELLS; K* = 32)
    "x_cross_val": "/c/chlu/outputs/w12-single-design-crossover/CROSSOVER-val.json",
    "x_ledger_K": "/c/chlu/outputs/w12-single-design-crossover/LEDGER-ACROSS-K.json",
    "x_scored1": "/c/chlu/outputs/w12-single-design-crossover/scored/run1_seed0_K32/scored.json",
    "x_del_test": "/c/chlu/outputs/w12-single-design-crossover/deletion/test_seed0_K32/deletion.json",
    "x_del_val": "/c/chlu/outputs/w12-single-design-crossover/deletion/val_seed0_K5/deletion.json",
    "x_probe": "/c/chlu/outputs/w12-single-design-crossover/deletion/probe_test_seed0_K32.json",
    "x_findings": "/c/chlu/outputs/w12-single-design-crossover/FINDINGS.md",
    "x_selection_rule": "/c/chlu/outputs/w12-single-design-crossover/SELECTION-RULE.md",
    # T8: hub/composition-ladder (n = RUNGS, n = 2..6; its store arms sit at K = 5)
    "lad_json": "/c/chlu/outputs/w12-composition-ladder/LADDER.json",
    "lad_bytes": "/c/chlu/outputs/w12-composition-ladder/BYTES.json",
    "lad_findings": "/c/chlu/outputs/w12-composition-ladder/FINDINGS.md",
    "lad_ruling": "/c/chlu/outputs/w12-composition-ladder/HEAD-RULING-K.md",
    "harness_ready": "/c/chlu/outputs/w12-kway-ladder-harness/LADDER-READY.json",
    # T10: spoke/deletion-baselines -- the only permitted refusal comparator
    "base_ready": "/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json",
    # the seed gate: generation-2 caches for seeds 3 and 4
    "cache_ready": "/c/chlu/outputs/w12-cache-gen2-seeds-3-4/CACHE-READY.json",
    "cache_verify": "/c/chlu/outputs/w12-cache-gen2-seeds-3-4/VERIFY.json",
    "cache_gate": "/c/chlu/outputs/w12-cache-gen2-seeds-3-4/reencode_gate_seeds_3_4.json",
    # T11 ablation 2: the wave-4/5 ITEM store -- forgets, loses composition
    "w5_delete_report": "/c/chlu/outputs/w5-delete/run1/report.json",
    # which seeds actually have FITTED RIVALS - the field NR7 turns on
    "rivals_ready": "/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json",
    # the plan this fold is executed against
    "wave_plan_12": "/c/chlu/evidence/outputs/advisor-wave-plan-w12-w14.md",
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


# --------------------------------------------------------------------------------------
# 0c. w13 -- THE TWO NOTATIONS THAT MUST NEVER COLLIDE.
# --------------------------------------------------------------------------------------
NOTATION = {
    "why_this_field_exists": (
        "n and K were both written 'k' in the wave-12 artefacts and the Hub collided them once. "
        "They are different axes of different tables and this file never writes 'k' for either."
    ),
    "n": {
        "meaning": "the number of jointly composed factors -- the LADDER RUNGS",
        "range_measured": "n = 2, 3, 4, 5, 6",
        "table": "T8",
        "source_calls_it": ("`LADDER.json` /rungs/{2,3,4,5,6} writes this as 'k' and as 'rung'. "
                            "T8 relabels it n and states K on every row."),
    },
    "K": {
        "meaning": "the number of WELLS -- the partition of the feature space",
        "range_measured": "K = 5, 8, 10, 12, 16, 20, 24, 28, 32; K* = 32",
        "tables": ["T7", "T9", "T12"],
        "source_calls_it": "`CROSSOVER-val.json` /per_K/{K} and /K_star.",
    },
    "RULE": ("EVERY LADDER ROW STATES ITS K. EVERY WELLS ROW STATES ITS n. "
             "A row that states neither is not a row."),
    "the_collision_in_one_line": ("T8's five rows are n = 2..6 at ONE K (K = 5). "
                                  "T7's one row is ONE n-set (60 pair corners + 49 n-way corners) "
                                  "at ONE K (K* = 32). They are not the same axis and never share a table."),
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


# --- w13 addition: a cell whose entries are (source, pointer) pairs, one per seed. --------
def wcell(entries, fmt="{:.4f}", seeds=None, stats=True):
    """Wave-12's artefacts hold every seed inside ONE json, so a cell is a list of
    (source_id, pointer) pairs -- one per seed -- rather than one pointer across N files.
    The shape emitted is exactly what verify_tables.py asserts: values[i] is the value on
    disk at provenance[i]'s path and pointer.  `seeds` is the PRINTED seed label list and
    is never inferred: a 1-seed row says so."""
    vals, prov = [], []
    for sid, pointer in entries:
        vals.append(ptr(DOCS[sid], pointer))
        prov.append({"source": sid, "path": SOURCES[sid], "pointer": pointer})
    out = {"seeds": list(seeds) if seeds is not None else list(range(len(vals))),
           "n_seeds": len(vals), "values": vals, "provenance": prov, "format": fmt}
    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if stats and nums and len(nums) == len(vals) and len(vals) > 1:
        out["mean"] = sum(nums) / len(nums)
        out["min"] = min(nums)
        out["max"] = max(nums)
    return out


def w1(sid, pointer, fmt="{:.4f}", seed=0):
    """A ONE-SEED cell.  Exists so that no 1-seed number can be printed without its seed."""
    return wcell([(sid, pointer)], fmt=fmt, seeds=[seed])


def w3s(arm, tmpl, fmt="{:.4f}"):
    """A THREE-SEED cell across the three w5 READ-TABLE.json files, one per seed; `tmpl`
    carries one `%s` for the arm name."""
    return wcell([(sid, tmpl % arm) for sid in W5], fmt=fmt, seeds=[0, 1, 2])


def w3p(sid, tmpl, fmt="{:.4f}", seeds=(0, 1, 2)):
    """Alias of w3 kept separate so a per-seed pointer inside ONE json reads differently
    from a per-seed pointer across three."""
    return w3(sid, tmpl, fmt=fmt, seeds=seeds)


def w3(sid, tmpl, fmt="{:.4f}", seeds=(0, 1, 2)):
    """A THREE-SEED cell out of one json: `tmpl` carries one `{s}` placeholder."""
    return wcell([(sid, tmpl.format(s=s)) for s in seeds], fmt=fmt, seeds=list(seeds))


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
# 8w. w13 -- WAVE 12'S ROWS.  T7..T12.  Every row states BOTH n and K.
#     ZERO scored runs here: these re-read wave 12's artefacts.
# --------------------------------------------------------------------------------------
X_ARMS = [
    ("cov_tensor", "**`cov_tensor` -- THE ARM** (single design, K = 32)", "RESULT"),
    ("landscape_removed", "`landscape_removed`", "control"),
    ("landscape_shuffled", "`landscape_shuffled`", "control"),
    ("landscape_shuffled_across_wells", "`landscape_shuffled_across_wells`", "control"),
    ("random_atoms", "`random_atoms`", "control"),
    ("mlp__as_trained", "`mlp` (524,288 B reference)", "comparator @524,288 B"),
    ("lstm__as_trained", "`lstm` (524,288 B reference)", "comparator @524,288 B"),
    ("transformer__as_trained", "`transformer` (524,288 B reference)", "comparator @524,288 B"),
    ("cont_mlp__as_trained", "`cont_mlp` (524,288 B reference)", "comparator @524,288 B"),
    ("table", "`table` -- the FLOOR, never a rival", "floor"),
]

_t7_rows = []
for _arm, _label, _cls in X_ARMS:
    _cells = {
        "row 3 - 60 pair corners": w1("x_scored1", "/per_arm/%s/row3_pair/novel_per_corner_mean" % _arm, "{:.3f}"),
        "row 4 - 49 n-way corners": w1("x_scored1", "/per_arm/%s/row4_nway/novel_per_corner_mean" % _arm, "{:.3f}"),
        "all 109 corners": w1("x_scored1", "/per_arm/%s/all/novel_per_corner_mean" % _arm, "{:.3f}"),
        "judge accuracy": w1("x_scored1", "/per_arm/%s/all/judge_accuracy_mean" % _arm, "{:.4f}"),
    }
    if _arm == "cov_tensor":
        _cells["paired delta vs the arm"] = derived(
            ["-- (this IS the arm)"], [dsrc("x_scored1", "/per_arm/cov_tensor/all/novel_per_corner_mean")],
            "the arm is not compared with itself", fmt="{}")
        _cells["CI95 of that delta"] = derived(
            ["--"], [dsrc("x_scored1", "/per_arm/cov_tensor/all/novel_per_corner_mean")], "n/a", fmt="{}")
        _cells["corners the arm wins"] = derived(
            ["--"], [dsrc("x_scored1", "/n_corners/row3_pair")], "n/a", fmt="{}")
        _cells["params @ the 524,288 B pin"] = derived(
            ["NOT A PARAMETER MODEL - see the byte ledger"],
            [dsrc("x_scored1", "/ledger_four_part/honest_total_bytes")],
            "the arm is a store; its size is the four-part ledger, not a parameter count", fmt="{}")
    else:
        _cells["paired delta vs the arm"] = w1(
            "x_scored1", "/paired_vs_cov_tensor/%s/mean_paired_delta" % _arm, "{:+.4f}")
        _cells["CI95 of that delta"] = wcell(
            [("x_scored1", "/paired_vs_cov_tensor/%s/bootstrap_ci95/0" % _arm),
             ("x_scored1", "/paired_vs_cov_tensor/%s/bootstrap_ci95/1" % _arm)],
            fmt="{:+.3f}", seeds=["lo", "hi"], stats=False)
        _cells["corners the arm wins"] = w1(
            "x_scored1", "/paired_vs_cov_tensor/%s/corners_arm_wins" % _arm, "{:.0f}")
        if _arm.endswith("__as_trained"):
            _cells["params @ the 524,288 B pin"] = w1(
                "x_scored1", "/rival_meta/%s/param_count" % _arm, "{:,.0f}")
        else:
            _cells["params @ the 524,288 B pin"] = derived(
                ["--"], [dsrc("x_scored1", "/baseline_matching/reference_column_bytes")],
                "a control or a lookup floor carries no matched parameter count", fmt="{}")
    _t7_rows.append({"label": _label, "row_class": _cls, "seeds_printed": [0], "n_seeds_printed": 1,
                     "cells": _cells})

TABLES["T7"] = {
    "title": ("TABLE 7 (MAIN) - Shapes3D value-holdout, the SINGLE DESIGN at K* = 32, TEST: "
              "rows 3 and 4. 1 scored seed, DECLARED."),
    "table_class": "RESULT",
    "placement": "MAIN TEXT",
    "environment": "Shapes3D value-holdout condition (NOT a maze: never in a table with T1 or T3)",
    "notation": {
        "K_wells": 32,
        "n_factors": ("row 3 = pair corners (2 factors pinned); row 4 = n-way corners (all six pinned). "
                      "This table varies neither: it is ONE n-set at ONE K."),
        "rule": ("T7 states K = 32 on every row; the ladder that varies n is T8 and every one of its "
                 "rows sits at K = 5."),
    },
    "pins": {"condition": "test", "K (wells)": 32, "A_requested": 63, "seed": 0,
             "metric": "distinct_valid_novel_tuples per corner",
             "estimand": "fw.paired_estimand, the CORNER as the unit",
             "margin declared in advance": 1.0,
             "feature cache": "generation 2, scope full, 480,000 rows",
             "address fit seconds": 1242.3163497447968},
    "pin_provenance": [
        dsrc("x_scored1", "/K_star"), dsrc("x_scored1", "/A_requested"), dsrc("x_scored1", "/seed"),
        dsrc("x_scored1", "/condition"), dsrc("x_scored1", "/n_corners/row3_pair"),
        dsrc("x_scored1", "/n_corners/row4_nway"), dsrc("x_scored1", "/address_fit_seconds"),
        dsrc("x_scored1", "/feature_cache/generation"), dsrc("x_scored1", "/rank_over_width_max"),
        dsrc("x_scored1", "/SCORED"),
    ],
    "seed_count": {
        "scored seed id": w1("x_scored1", "/seed", "{:.0f}"),
        "PRINTED AS A COLUMN, NOT A FOOTNOTE": "n = 1 (seed 0). Every row of T7 is a 1-seed row.",
        "why not 3": ("the crossover hub spent run 1 of its 3 `single-design` scored runs on seed 0 at "
                      "K* = 32; runs 2/3 and 3/3 (seeds 1 and 2) are NR6, wall clock."),
        "why not 5": ("seeds 3 and 4 have generation-2 caches (NR7) but no frozen decoder, no external "
                      "classifier and no fitted rivals, so no corner of them can be judged."),
        "NEVER MIX": ("T7 (1 seed) must not share a plot with T8 (3 seeds) or T1/T2/T3 (3 seeds) unless "
                      "the seed count is printed per row."),
    },
    "rows": _t7_rows,
    "rank_width": {
        "rank/width max on the scored object": w1("x_scored1", "/rank_over_width_max", "{:.4f}"),
        "wave 9's decorative threshold": 0.938,
        "reading": ("0.9688 is ABOVE the 0.938 at which wave 9's maze arm was decorative, and the four "
                    "controls still read exactly 0.000. Only the control decides; rank/width is printed "
                    "because it is the warning it is."),
    },
    "byte_ledger": {
        "rule": "FOUR PARTS, PRINTED, NEVER NETTED (w12r/MF3). Seed 0, K* = 32, TEST.",
        "seed_label": "part (ONE seed, id 0)",
        "parts": {
            "1. atoms": w1("x_scored1", "/ledger_four_part/p1_atom_bytes", "{:,.0f}"),
            "2. covariance + count ledger": w1("x_scored1", "/ledger_four_part/p2_covariance_plus_count_bytes", "{:,.0f}"),
            "2a. of which covariance": w1("x_scored1", "/ledger_four_part/p2_covariance_bytes", "{:,.0f}"),
            "2b. of which count": w1("x_scored1", "/ledger_four_part/p2_count_bytes", "{:,.0f}"),
            "3. address book": w1("x_scored1", "/ledger_four_part/p3_address_book_bytes", "{:,.0f}"),
            "4. coupling / hole table": w1("x_scored1", "/ledger_four_part/p4_coupling_hole_table_bytes", "{:,.0f}"),
        },
        "totals": {
            "honest total (the sum of the four)": w1("x_scored1", "/ledger_four_part/honest_total_bytes", "{:,.0f}"),
            "x the 524,288 B reference": w1("x_scored1", "/ledger_four_part/times_reference_pin", "{:.4f}"),
        },
        "netted": False,
        "pin_status": ("the object is 4.0897x the 524,288 B figure. That figure is a REFERENCE COLUMN in "
                       "this table, never a budget the object met."),
    },
    "byte_matching": {
        "WHY BOTH COLUMNS ARE PRINTED": (
            "the two matchings tell OPPOSITE stories and a reader who sees one draws the opposite "
            "conclusion. See T8's caption for the sign flip on the ladder."),
        "1x = this object's own measured total, bytes": w1(
            "x_scored1", "/ledger_four_part/honest_total_bytes", "{:,.0f}"),
        "1x = float32 parameters a 1x-matched rival would carry": w1(
            "x_cross_val", "/per_K/32/baseline_match_1x/target_params_float32", "{:,.0f}"),
        "1x column status": "DECLARED NOT-RUN - no rival was fitted at this size (NR8).",
        "524,288 B reference, bytes": w1("x_scored1", "/ledger_four_part/reference_pin_bytes", "{:,.0f}"),
        "524,288 B reference, float32 parameters": w1(
            "x_cross_val", "/per_K/32/baseline_match_1x/reference_pin_params_float32", "{:,.0f}"),
        "rivals actually fitted, params": "130,433 - 131,891, i.e. within +/-0.63 % of 131,072",
        "524,288 B column status": "MEASURED - every comparator row above is against this column.",
        "DIRECTION OF THE ERROR": (
            "AGAINST the arm. A 1x-matched rival would carry about 4.09x these parameters and would be "
            "STRONGER, so the printed comparison FLATTERS the arm and no row above is a 1x result."),
    },
    "verdict": {
        "statement": ("the control gate PASSES - all four controls read exactly 0.000 on both rows at "
                      "rank/width 0.9688 - so the row may be claimed, at 1 seed, against the 524,288 B "
                      "reference column and NOT at 1x."),
        "charter C.1 vs the prototype table floor": w1(
            "x_scored1", "/paired_vs_cov_tensor/table/mean_paired_delta", "{:+.4f}"),
        "its CI95 lower bound": w1("x_scored1", "/paired_vs_cov_tensor/table/bootstrap_ci95/0", "{:+.4f}"),
        "margin declared in advance": w1(
            "x_scored1", "/paired_vs_cov_tensor/table/margin_declared_in_advance", "{:+.1f}"),
        "provenance": [dsrc("x_scored1", "/paired_vs_cov_tensor/table/mean_paired_delta"),
                       dsrc("x_scored1", "/paired_vs_cov_tensor/landscape_removed/mean_paired_delta")],
    },
    "caveats": [
        ("`cont_mlp` scores 0.000 / 0.061 - at the floor. That is a fact about that rival's fit on this "
         "population, not a demonstration about the arm, and it is printed rather than leaned on."),
        ("The deletion legs of this same object are T9 and the refusal DISPLAY is T10. Same object, same "
         "seed, still separate tables, because a DISPLAY is not a RESULT."),
        ("The request-only property is asserted in the artefact: no feature tensor enters launch, settle "
         "or read (charter B.3), bitwise stable under feature perturbation over 64 checks."),
    ],
}

# --------------------------------------------------------------------------------------
# 8x. TABLE 8 -- THE n-WAY LADDER.  n = 2..6.  K = 5 ON EVERY ROW.  3 seeds.
# --------------------------------------------------------------------------------------
LAD_RUNGS = ["2", "3", "4", "5", "6"]
LAD_ARMS = [
    ("single_design_K5", "single design (the arm), K = 5"),
    ("pca_tensor", "pca_tensor - composition-only store (ABLATION)"),
    ("learned_1x", "learned_1x - MLP matched to the arm's OWN total"),
    ("rival_mlp", "rival_mlp @524,288 B"),
    ("rival_lstm", "rival_lstm @524,288 B"),
    ("rival_transformer", "rival_transformer @524,288 B"),
    ("rival_cont_mlp", "rival_cont_mlp @524,288 B"),
    ("rival_table", "rival_table - the FLOOR"),
    ("learned_ref_ownpool", "learned_ref_ownpool"),
]

_t8_rows = []
for _rg in LAD_RUNGS + ["shipped"]:
    _cells = {
        "K (wells)": derived([5], [dsrc("lad_json", "/K_of_the_store_arms")],
                             "every leg of this ladder sits at K = 5", fmt="{:.0f}"),
        "seeds (n), traced": w1("lad_json", "/n_seeds_declared", "{:.0f}"),
        "corners": w3("lad_json", "/rungs/" + _rg + "/n_corners_per_seed/{s}", "{:.0f}"),
        "ceiling / corner": w1("lad_json", "/rungs/" + _rg + "/ceiling_per_corner", "{:.0f}"),
    }
    for _arm, _lab in LAD_ARMS:
        _cells[_lab] = w3("lad_json", "/rungs/" + _rg + "/arms/" + _arm + "/per_seed/{s}/mean", "{:.4f}")
    _cells["delta vs 1x (learned_1x)"] = w3(
        "lad_json",
        "/rungs/" + _rg + "/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned/mean_paired_delta",
        "{:+.4f}")
    _cells["delta vs best @524,288 B"] = w3(
        "lad_json",
        "/rungs/" + _rg + "/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned_5/mean_paired_delta",
        "{:+.4f}")
    _cells["rank/width max (arm)"] = w3(
        "lad_json",
        "/rungs/" + _rg + "/arms/single_design_K5/per_seed/{s}/rank_width/rank_over_width_max", "{:.4f}")
    if _rg != "shipped":
        _cells["control landscape_shuffled, paired delta"] = wcell(
            [("lad_json", "/control_verdicts/" + _rg
              + "/single_design_K5/landscape_shuffled/per_seed/%d/mean_paired_delta" % i) for i in range(3)],
            fmt="{:+.4f}", seeds=[0, 1, 2])
        _cells["control random_atoms, paired delta"] = wcell(
            [("lad_json", "/control_verdicts/" + _rg
              + "/single_design_K5/random_atoms/per_seed/%d/mean_paired_delta" % i) for i in range(3)],
            fmt="{:+.4f}", seeds=[0, 1, 2])
        _lbl, _cl = "**n = " + _rg + "**", "RESULT (APPENDIX placement)"
    else:
        _lbl = "`shipped` - the value-holdout condition as it exists on disk (reproduction leg)"
        _cl = "instrument check, NOT a rung"
    _t8_rows.append({"label": _lbl, "row_class": _cl, "seeds_printed": [0, 1, 2],
                     "n_seeds_printed": 3, "cells": _cells})

TABLES["T8"] = {
    "title": ("TABLE 8 - the n-WAY LADDER on Shapes3D: n = 2..6 jointly composed factors, "
              "K = 5 WELLS on every row, 3 seeds on every row."),
    "table_class": "RESULT",
    "placement": w1("lad_json", "/placement", "{}"),
    "notation": {
        "n": "the ladder rung = the number of jointly composed factors. n = 2, 3, 4, 5, 6.",
        "K": ("5 on EVERY row of this table. The source writes the rung as `k`; this table writes the "
              "rung as n and prints K separately, because K is the WELL COUNT and the two were "
              "collided once."),
        "RULE": "T8 (K = 5) and T7 (K = 32) are two different objects and never share a table or an axis.",
    },
    "pins": {"condition": "Shapes3D value-holdout, k-way families from the w12 harness",
             "K (wells)": 5, "metric": "distinct_valid_novel_tuples per corner",
             "estimand": "paired, the corner as the unit", "margin": 1.0,
             "arms per leg": 13, "legs": "18 = (5 rungs + `shipped`) x 3 seeds",
             "scored arm-runs": 234},
    "pin_provenance": [dsrc("lad_json", "/K_of_the_store_arms"), dsrc("lad_json", "/metric"),
                       dsrc("lad_json", "/margin"), dsrc("lad_json", "/n_seeds_declared"),
                       dsrc("lad_json", "/seeds"), dsrc("lad_json", "/not_run"),
                       dsrc("lad_json", "/best_learned_rule"), dsrc("lad_json", "/K_star_source")],
    "seed_count": {
        "seeds on every row": w1("lad_json", "/n_seeds_declared", "{:.0f}"),
        "PRINTED AS A COLUMN": "`seeds (n)` is a column of this table, not a footnote.",
        "why not 5": ("seed 3 passes the generation-2 cache gate and still cannot be scored - `Judge(3)` "
                      "raises FileNotFoundError on `decoder_v1_s3.pkl`; seed 4's cache landed at "
                      "2026-09-21T15:27:15Z, after this hub had closed its seed gate. Both are NR7."),
    },
    "rows": _t8_rows,
    "byte_ledger": {
        "seed_label": "part (seed 0 / seed 1 / seed 2 unless the row says otherwise)",
        "rule": ("FOUR PARTS, NEVER NETTED. The ladder's ledger file carries parts 1-3 measured on all 18 "
                 "fitted legs; part 4 (coupling / hole table) is NOT in that file and is a DECLARED "
                 "ABSENCE below, never a zero."),
        "parts": {
            "1. atoms (both store arms, every leg)": w1(
                "lad_bytes", "/legs/2_seed0/single_design_K5/p1_atom_bytes", "{:,.0f}"),
            "2. sufficient statistics - single design, n = 2, seeds 0/1/2": wcell(
                [("lad_bytes", "/legs/2_seed%d/single_design_K5/p2_statistics_bytes" % s) for s in range(3)],
                fmt="{:,.0f}", seeds=[0, 1, 2]),
            "2. sufficient statistics - pca_tensor (the ablation)": wcell(
                [("lad_bytes", "/legs/2_seed%d/pca_tensor/p2_statistics_bytes" % s) for s in range(3)],
                fmt="{:,.0f}", seeds=[0, 1, 2]),
            "3. address book - single design, n = 2, seeds 0/1/2": wcell(
                [("lad_bytes", "/per_arm/single_design_K5/%d/p3_address_book" % i) for i in range(3)],
                fmt="{:,.0f}", seeds=[0, 1, 2]),
            "4. coupling / hole table": {
                "status": "DECLARED ABSENT FROM THE SOURCE",
                "why": ("`BYTES.json` carries three parts per leg. The coupling/hole table is measured on "
                        "the K* object in T7 (6,610 B) and across the K-sweep in T12; it is NOT measured "
                        "on the ladder's legs and is therefore not printed as a zero here.")},
        },
        "totals": {
            "single design total - min over 18 legs": w1(
                "lad_bytes", "/summary/single_design_K5/total_bytes_min", "{:,.0f}"),
            "single design total - max over 18 legs": w1(
                "lad_bytes", "/summary/single_design_K5/total_bytes_max", "{:,.0f}"),
            "x the 524,288 B reference - min": w1(
                "lad_bytes", "/summary/single_design_K5/times_reference_pin_min", "{:.4f}"),
            "x the 524,288 B reference - max": w1(
                "lad_bytes", "/summary/single_design_K5/times_reference_pin_max", "{:.4f}"),
            "pca_tensor total - min / max": wcell(
                [("lad_bytes", "/summary/pca_tensor/total_bytes_min"),
                 ("lad_bytes", "/summary/pca_tensor/total_bytes_max")],
                fmt="{:,.0f}", seeds=["min", "max"], stats=False),
            "pca_tensor x reference - min / max": wcell(
                [("lad_bytes", "/summary/pca_tensor/times_reference_pin_min"),
                 ("lad_bytes", "/summary/pca_tensor/times_reference_pin_max")],
                fmt="{:.4f}", seeds=["min", "max"], stats=False),
        },
        "netted": False,
        "THE HEADLINE THE DRAFT OMITTED": (
            "the single design costs 23.886x-24.875x the 524,288 B reference and `pca_tensor` costs "
            "1.358x-1.372x - about 18x less - for composition scores that are IDENTICAL at every rung "
            "and every seed. The single design's bytes buy EXACT DELETION, not composition."),
    },
    "byte_matching_flip": {
        "THE CAPTION THAT MUST TRAVEL WITH THIS TABLE": (
            "THE TWO BYTE MATCHINGS TELL OPPOSITE STORIES ABOUT THE n-TREND. Against the 524,288 B rivals "
            "the arm's margin GROWS with n: +7.083..+9.650 at n = 2 rising to +14.992..+22.425 at n = 6. "
            "Against a learner matched to the arm's OWN total (1x) the margin is NEGATIVE at n = 2, 4, 5 "
            "and 6. A reader shown one column alone draws the opposite conclusion, so both columns are "
            "printed on every row of this table."),
        "delta vs best @524,288 B - n = 2, per seed": w3(
            "lad_json",
            "/rungs/2/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned_5/mean_paired_delta", "{:+.4f}"),
        "delta vs best @524,288 B - n = 6, per seed": w3(
            "lad_json",
            "/rungs/6/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned_5/mean_paired_delta", "{:+.4f}"),
        "delta vs 1x - n = 2, per seed": w3(
            "lad_json",
            "/rungs/2/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned/mean_paired_delta", "{:+.4f}"),
        "delta vs 1x - n = 6, per seed": w3(
            "lad_json",
            "/rungs/6/arms/single_design_K5/per_seed/{s}/paired_vs_best_learned/mean_paired_delta", "{:+.4f}"),
        "delta vs 1x, ceiling-normalised, n = 2/3/4/5/6": derived(
            [-0.077315, 0.058912, -0.073380, -0.054167, -0.056404],
            [dsrc("lad_json", "/rungs/%s/arms/single_design_K5/per_seed/%d/"
                              "paired_vs_best_learned/mean_paired_delta" % (rg, s))
             for rg in LAD_RUNGS for s in range(3)]
            + [dsrc("lad_json", "/rungs/%s/ceiling_per_corner" % rg) for rg in LAD_RUNGS],
            "mean over the 3 seeds of the paired delta, divided by that rung's ceiling per corner",
            fmt="{:+.6f}",
            note=("DERIVED. Negative at n = 2, 4, 5, 6 (-0.0773 .. -0.0542) and positive only at n = 3. "
                  "Recompute from the 20 pointers in `from`.")),
        "the_1x_opponent": (
            "`learned_1x` is ONE family (an MLP) fitted at the arm's own total. A 1x-matched LSTM, "
            "transformer, continuous MLP and table are a DECLARED NOT-RUN (NR8), so the 1x column is the "
            "WEAKEST available 1x opponent and the true 1x margin can only be more negative."),
    },
    "instrument": {
        "FOR n >= 3 THIS LADDER IS A TRAINING-DATA AXIS, NOT A COMPOSITION-DEPTH AXIS": (
            "n = 3, 4, 5 and 6 probe the IDENTICAL 120 corners; only how coarsely the hold-out is carved "
            "changes, and with it the held-out row count (300,000 / 235,200 / 213,600 / 200,640). Every "
            "arm fitted on the rung's own pool tracks that count at r = -0.96 .. -0.98; every arm not "
            "fitted on it is EXACTLY constant across those four rungs. The n = 2 -> n >= 3 step is the "
            "only genuine change of probe on this ladder."),
        "corners at n = 2 (disjoint from n >= 3)": w3("lad_json", "/rungs/2/n_corners_per_seed/{s}", "{:.0f}"),
        "corners at n = 6": w3("lad_json", "/rungs/6/n_corners_per_seed/{s}", "{:.0f}"),
        "no arm has a breaking point": w1("lad_json", "/same_k_for_every_arm/statement", "{}"),
        "legs that failed": derived(["0 - the source's own `not_run` field is the empty list"],
                                    [dsrc("lad_json", "/not_run")], "stored field", fmt="{}"),
        "ceiling-normalised arm, n = 2 -> n = 6": wcell(
            [("lad_json", "/ladder_verdicts/single_design_K5/per_rung/%s/normalised_mean" % rg)
             for rg in LAD_RUNGS], fmt="{:.4f}", seeds=["n=2", "n=3", "n=4", "n=5", "n=6"], stats=False),
    },
    "verdict": {
        "statement": ("the store NEVER CLEARS THE BAR against a learner matched to its own bytes - at "
                      "n = 2, 4, 5 and 6 on every seed, and at 2 of 3 seeds at n = 3, so under the "
                      "pre-declared every-seed rule it clears at NO rung. Nothing deteriorates as n "
                      "grows; ceiling-normalised the arm goes UP, 0.8483 -> 0.8966. Both statements are "
                      "the ladder's result and neither of them is a curve."),
        "provenance": [dsrc("lad_json", "/ladder_verdicts/single_design_K5/per_rung/2/clears_bar"),
                       dsrc("lad_json", "/ladder_verdicts/single_design_K5/per_rung/6/clears_bar"),
                       dsrc("lad_json", "/ladder_verdicts/single_design_K5/per_rung/2/normalised_mean"),
                       dsrc("lad_json", "/ladder_verdicts/single_design_K5/per_rung/6/normalised_mean"),
                       dsrc("lad_json", "/same_k_for_every_arm/statement")],
    },
}


# --------------------------------------------------------------------------------------
# 8y. TABLE 9 / 9b -- DELETION.  value / conjunction / conjunction-live / item.
#     T9 = TEST, seed 0, K* = 32 (the source's own SMOKE_NOT_A_CLAIM is false).
#     T9b = VAL, seed 0, K = 5   (the source's own SMOKE_NOT_A_CLAIM is TRUE -> labelled SMOKE).
#     Two conditions and two K: two tables, never one.
# --------------------------------------------------------------------------------------
DEL_SCOPES = [
    ("value", "(a) VALUE - every fit row carrying red (`object_hue` index 0)"),
    ("conjunction", "(b) CONJUNCTION - red AND cube, AS THE TASK SPECIFIES IT -- **VACUOUS**"),
    ("conjunction_live", "(b') RED SPHERE - the DECLARED non-vacuous substitute"),
    ("item", "(c) ITEM - a random 1 % of fit rows"),
]


def _deletion_table(sid, tid, cond, kk, seed, smoke):
    rows = []
    for sc, label in DEL_SCOPES:
        b = "/scopes/" + sc
        cells = {
            "K (wells)": derived([kk], [dsrc(sid, "/K")], "stored field /K", fmt="{:.0f}"),
            "seeds (n), traced": derived([1], [dsrc(sid, "/seed")], "one seed, id " + str(seed),
                                         fmt="{:.0f}"),
            "rows deleted": w1(sid, b + "/n_rows_deleted", "{:,.0f}", seed=seed),
            "fraction of the fit pool": w1(sid, b + "/frac_rows_deleted", "{:.6f}", seed=seed),
            "OUR adaptation, seconds": w1(sid, b + "/our_cost_line/seconds", "{:.4f}", seed=seed),
            "OUR bytes rewritten": w1(sid, b + "/our_cost_line/bytes_rewritten", "{:,.0f}", seed=seed),
            "OUR data rows read": w1(sid, b + "/our_cost_line/data_rows_read", "{:,.0f}", seed=seed),
            "OUR refit?": w1(sid, b + "/our_cost_line/refit", "{}", seed=seed),
            "REFERENCE refit-without, seconds": w1(
                sid, b + "/exactness_store_scoped/refit_without_cost_line/seconds", "{:.3f}", seed=seed),
            "REFERENCE refit-without, data rows read": w1(
                sid, b + "/exactness_store_scoped/refit_without_cost_line/data_rows_read", "{:,.0f}", seed=seed),
            "store-scoped exactness, max abs mode dev": w1(
                sid, b + "/exactness_store_scoped/max_abs_mode_dev", "{:.3e}", seed=seed),
            "ranks agree": w1(sid, b + "/exactness_store_scoped/ranks_agree", "{}", seed=seed),
            "survivor corners": w1(sid, b + "/n_survivor_corners", "{:.0f}", seed=seed),
            "paired delta on survivors": w1(
                sid, b + "/paired_after_minus_before_on_survivors/mean_paired_delta", "{:+.4f}", seed=seed),
            "DELETED corners": w1(sid, b + "/n_deleted_corners", "{:.0f}", seed=seed),
            "principal-angle cos_min (the basis DID move)": w1(
                sid, b + "/basis_change_audit/principal_angle_cos_min", "{:.6f}", seed=seed),
        }
        if sc == "value" and cond == "test":
            cells["DELETED corners, novel/corner BEFORE"] = w1(
                sid, b + "/before_same_corners/deleted/novel_per_corner_mean", "{:.4f}", seed=seed)
            cells["DELETED corners, novel/corner AFTER"] = w1(
                sid, b + "/after/deleted/novel_per_corner_mean", "{:.4f}", seed=seed)
            cells["DELETED corners, judge acc BEFORE"] = w1(
                sid, b + "/before_same_corners/deleted/judge_accuracy_mean", "{:.4f}", seed=seed)
            cells["DELETED corners, judge acc AFTER"] = w1(
                sid, b + "/after/deleted/judge_accuracy_mean", "{:.4f}", seed=seed)
        else:
            cells["DELETED corners, novel/corner BEFORE"] = derived(
                ["DECLARED: this scope deletes no corner"], [dsrc(sid, b + "/n_deleted_corners")],
                "the source writes DECLARED_NOT_RUN in place of a number", fmt="{}")
            cells["DELETED corners, novel/corner AFTER"] = derived(
                ["DECLARED: this scope deletes no corner"], [dsrc(sid, b + "/n_deleted_corners")],
                "the source writes DECLARED_NOT_RUN in place of a number", fmt="{}")
        rows.append({"label": label, "row_class": ("RESULT" if not smoke else "SMOKE - NOT A CLAIM"),
                     "seeds_printed": [seed], "n_seeds_printed": 1, "cells": cells})

    return {
        "title": ("TABLE %s - EXACT DELETION (STORE-SCOPED) on the single design, %s condition, "
                  "K = %d, 1 seed (id %d).%s" % (tid[1:], cond.upper(), kk, seed,
                                                 "  SMOKE - NOT A CLAIM." if smoke else "")),
        "table_class": "RESULT" if not smoke else "SMOKE (the artefact labels itself so)",
        "placement": "MAIN TEXT" if not smoke else "APPENDIX",
        "SMOKE_NOT_A_CLAIM_as_the_artefact_records_it": w1(sid, "/SMOKE_NOT_A_CLAIM", "{}", seed=seed),
        "notation": {"K_wells": kk,
                     "n_factors": "not varied in this table; the n-ladder is T8",
                     "rule": "every row of T%s states K = %d and its seed count." % (tid[1:], kk)},
        "pins": {"condition": cond, "K (wells)": kk, "seed": seed,
                 "definition of exact": "STORE-SCOPED - see TABLES.json /exactness_definition",
                 "feature cache": "generation 2, scope full, 480,000 rows"},
        "pin_provenance": [dsrc(sid, "/K"), dsrc(sid, "/seed"), dsrc(sid, "/condition"),
                           dsrc(sid, "/A_requested"), dsrc(sid, "/feature_cache/generation"),
                           dsrc(sid, "/SMOKE_NOT_A_CLAIM")],
        "seed_count": {"seeds (n)": 1,
                       "PRINTED AS A COLUMN": "`seeds (n)` is a column of this table.",
                       "why not 3 or 5": "NR6 (wall clock, seeds 1-2) and NR7 (seeds 3-4 unjudgeable)."},
        "rows": rows,
        "byte_ledger": {
            "rule": "FOUR PARTS, NEVER NETTED. Measured on the object this deletion ran on.",
            "seed_label": "part (ONE seed, id %d)" % seed,
            "parts": {
                "1. atoms": w1(sid, "/ledger_four_part/p1_atom_bytes", "{:,.0f}", seed=seed),
                "2. covariance + count ledger": w1(
                    sid, "/ledger_four_part/p2_covariance_plus_count_bytes", "{:,.0f}", seed=seed),
                "3. address book": w1(sid, "/ledger_four_part/p3_address_book_bytes", "{:,.0f}", seed=seed),
                "4. coupling / hole table": w1(
                    sid, "/ledger_four_part/p4_coupling_hole_table_bytes", "{:,.0f}", seed=seed),
            },
            "totals": {
                "honest total": w1(sid, "/ledger_four_part/honest_total_bytes", "{:,.0f}", seed=seed),
                "x the 524,288 B reference": w1(sid, "/ledger_four_part/times_reference_pin", "{:.4f}", seed=seed),
            },
            "netted": False,
        },
        "language_rule_the_head_set": (
            "NEVER 'no changes after deletion'. The sentence is: post-deletion adaptation is LIGHT and "
            "DATA-FREE - the covariance and count are downdated by subtraction, the atoms are "
            "re-eigendecomposed from the downdated statistic and the address book is carried across the "
            "basis change, with NO row of data read - against a reference that can only delete by "
            "refitting from scratch without the deleted rows, reading 120,028-131,072 rows to do it."),
        "vacuous_row_is_printed_not_hidden": w1(
            sid, "/scopes/conjunction/vacuous_reason", "{}", seed=seed),
    }


TABLES["T9"] = _deletion_table("x_del_test", "T9", "test", 32, 0, False)
TABLES["T9b"] = _deletion_table("x_del_val", "T9b", "val", 5, 0, True)

# ---- the finding that bounds the whole deletion claim, on T9's own object ---------------
TABLES["T9"]["the_probe_that_bounds_the_claim"] = {
    "what": ("how far the RENDERED answer moved on exactly the requests whose evidence was deleted, "
             "in the store's own units. No judge, no decoder. 5,688 requests."),
    "rows": [
        {"label": "corners REQUESTING red", "row_class": "probe", "seeds_printed": [0], "n_seeds_printed": 1,
         "cells": {
             "requests": w1("x_probe", "/per_group/corners REQUESTING red/n_requests", "{:,.0f}"),
             "rel. RMS change, address book CARRIED": w1(
                 "x_probe", "/per_group/corners REQUESTING red/relative_rms_change_carried", "{:.3e}"),
             "rel. RMS change, address book NOT carried": w1(
                 "x_probe", "/per_group/corners REQUESTING red/relative_rms_change_uncarried", "{:.6f}"),
             "max abs change, carried": w1(
                 "x_probe", "/per_group/corners REQUESTING red/max_abs_change_carried", "{:.3e}"),
         }},
        {"label": "corners NOT requesting red", "row_class": "probe", "seeds_printed": [0], "n_seeds_printed": 1,
         "cells": {
             "requests": w1("x_probe", "/per_group/corners NOT requesting red/n_requests", "{:,.0f}"),
             "rel. RMS change, address book CARRIED": w1(
                 "x_probe", "/per_group/corners NOT requesting red/relative_rms_change_carried", "{:.3e}"),
             "rel. RMS change, address book NOT carried": w1(
                 "x_probe", "/per_group/corners NOT requesting red/relative_rms_change_uncarried", "{:.6f}"),
             "max abs change, carried": w1(
                 "x_probe", "/per_group/corners NOT requesting red/max_abs_change_carried", "{:.3e}"),
         }},
    ],
    "address book bytes changed by the carry": w1("x_probe", "/carry/address_book_bytes_changed", "{:,.0f}"),
    "address book bytes total": w1("x_probe", "/carry/address_book_bytes_total", "{:,.0f}"),
    "data rows touched by the carry": w1("x_probe", "/carry/data_rows_touched", "{:,.0f}"),
    "THE BOUND, STATED WITHOUT SPIN": (
        "deleting all 11,044 red rows changes the 20 red-requesting corners by EXACTLY ZERO "
        "(32.25 -> 32.25 novel valid tuples per corner, judge accuracy 0.8958 -> 0.8958). The store-scoped "
        "exactness is real - the uncarried read moves by 0.61 relative RMS - but the basis change is "
        "GENERIC, not targeted: 0.6101 on red-requesting corners against 0.6143 on the others. At "
        "K* = 32 (rank/width 0.9531-0.9688) the per-well content carries no value-specific information; "
        "the value identity lives in the address book, which the deletion CARRIES and does not refit. "
        "'The store has forgotten red' and 'the object no longer answers red' are different statements "
        "and only the first is demonstrated."),
    "the_contrast_with_wave_8": (
        "at K = 5, A = 64 (rank/width 0.156) wave 8 found the basis change CONCENTRATED: cos_min 0.1999 "
        "against 0.9976 for a size-matched random deletion. The targeting of a deletion is a function of "
        "rank/width and it disappears exactly where the byte ledger is cheapest."),
}

TABLES["T9"]["sphere_caveat"] = {
    "HEAD_RULING": w1("x_del_test", "/sphere_caveat/HEAD_RULING", "{}"),
    "source": w1("x_del_test", "/sphere_caveat/source", "{}"),
    "what_it_threatens": w1("x_del_test", "/sphere_caveat/what_it_threatens", "{}"),
    "judged-sphere excess, `landscape_removed` (degraded bound)": w1(
        "x_del_test", "/sphere_caveat/measured_upper_bound_degraded_arms/landscape_removed/judged_sphere_excess",
        "{:+.4f}"),
    "judged-sphere excess, `random_atoms` (degraded bound)": w1(
        "x_del_test", "/sphere_caveat/measured_upper_bound_degraded_arms/random_atoms/judged_sphere_excess",
        "{:+.4f}"),
    "judged-sphere excess AFTER the red-sphere deletion": w1(
        "x_del_test", "/scopes/conjunction_live/shape_report_after/judged_sphere_excess", "{:+.4f}"),
    "verdict": ("the hazard does NOT reproduce on this instrument: a degraded render here is judged CUBE, "
                "the shape held out of the split entirely, so the sphere excess is NEGATIVE for both "
                "maximally degraded arms. A degrading baseline cannot score spuriously as sphere here. "
                "Scope: seed 0, TEST corners, K* = 32, this decoder and this classifier."),
}

# --------------------------------------------------------------------------------------
# 8z. TABLE 10 -- REFUSAL DISPLAY ROWS.  A DISPLAY IS NEVER A RESULT.
# --------------------------------------------------------------------------------------
_t10_rows = [
    {"label": "**OURS - the coupling/hole table**, TEST, K* = 32, seed 0",
     "row_class": "DISPLAY - never a result", "seeds_printed": [0], "n_seeds_printed": 1,
     "cells": {
         "K (wells)": derived([32], [dsrc("x_del_test", "/K")], "stored field /K", fmt="{:.0f}"),
         "seeds (n), traced": derived([1], [dsrc("x_del_test", "/seed")], "one seed, id 0", fmt="{:.0f}"),
         "holes punched": w1("x_del_test", "/display_refusal/n_holes_after", "{:.0f}"),
         "bytes rewritten": w1("x_del_test", "/display_refusal/cost_line/bytes_rewritten", "{:,.0f}"),
         "data rows read": w1("x_del_test", "/display_refusal/cost_line/data_rows_read", "{:,.0f}"),
         "seconds": w1("x_del_test", "/display_refusal/cost_line/seconds", "{:.4f}"),
         "refit?": w1("x_del_test", "/display_refusal/cost_line/refit", "{}"),
         "coupling table bytes": w1("x_del_test", "/display_refusal/coupling_table_bytes", "{:,.0f}"),
         "layout asserted equal to `gen_field`'s": w1(
             "x_del_test", "/display_refusal/layout_is_the_field_store's/layout_equals_gen_field", "{}"),
         "pair slots / indices checked": wcell(
             [("x_del_test", "/display_refusal/layout_is_the_field_store's/n_pairs"),
              ("x_del_test", "/display_refusal/layout_is_the_field_store's/n_indices_checked")],
             fmt="{:,.0f}", seeds=["pairs", "checked"], stats=False),
     }},
    {"label": "OURS - the coupling/hole table, VAL, K = 5, seed 0 (SMOKE)",
     "row_class": "DISPLAY - SMOKE, never a result", "seeds_printed": [0], "n_seeds_printed": 1,
     "cells": {
         "K (wells)": derived([5], [dsrc("x_del_val", "/K")], "stored field /K", fmt="{:.0f}"),
         "seeds (n), traced": derived([1], [dsrc("x_del_val", "/seed")], "one seed, id 0", fmt="{:.0f}"),
         "holes punched": w1("x_del_val", "/display_refusal/n_holes_after", "{:.0f}"),
         "bytes rewritten": w1("x_del_val", "/display_refusal/cost_line/bytes_rewritten", "{:,.0f}"),
         "data rows read": w1("x_del_val", "/display_refusal/cost_line/data_rows_read", "{:,.0f}"),
         "seconds": w1("x_del_val", "/display_refusal/cost_line/seconds", "{:.4f}"),
         "refit?": w1("x_del_val", "/display_refusal/cost_line/refit", "{}"),
         "coupling table bytes": w1("x_del_val", "/display_refusal/coupling_table_bytes", "{:,.0f}"),
         "layout asserted equal to `gen_field`'s": w1(
             "x_del_val", "/display_refusal/layout_is_the_field_store's/layout_equals_gen_field", "{}"),
     }},
]

_RT = "/per_seed/{s}/records/transformer__refuse_token__%s"
for _lbl, _rec in [("VALUE deletion (red)", "value-object_hue-0"),
                   ("CONJUNCTION deletion (red AND sphere)", "conj-object_hue0-shape2")]:
    _b = _RT % _rec
    _t10_rows.append({
        "label": "COMPARATOR - refuse-token transformer, " + _lbl,
        "row_class": "DISPLAY comparator - 3 seeds, FEATURE-LEVEL, off the hub's instrument",
        "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
        "cells": {
            "K (wells)": derived(["n/a - a learned rival has no wells"],
                                 [dsrc("base_ready", "/pins/byte_pin")], "the rival is byte-matched, not partitioned",
                                 fmt="{}"),
            "seeds (n), traced": derived([3], [dsrc("base_ready", "/seeds_present")], "stored field",
                                         fmt="{:.0f}"),
            "requests certified impossible": w3("base_ready", _b + "/applicability/n_certified_impossible", "{:,.0f}"),
            "AUROC, impossible vs possible": w3(
                "base_ready", _b + "/out_of_range_diagnostic/separation/auroc", "{:.4f}"),
            "separation (mean score gap)": w3(
                "base_ready", _b + "/out_of_range_diagnostic/separation/separation", "{:.4f}"),
            "refuse rate on impossible": w3(
                "base_ready", _b + "/out_of_range_diagnostic/separation/refuse_rate_on_impossible", "{:.4f}"),
            "FALSE refusal on possible": w3(
                "base_ready", _b + "/out_of_range_diagnostic/separation/false_refusal_on_possible", "{:.4f}"),
            "bytes rewritten": w3("base_ready", _b + "/cost/bytes_rewritten", "{:,.0f}"),
            "data rows read": w3("base_ready", _b + "/cost/data_rows_read", "{:,.0f}"),
            "seconds": w3("base_ready", _b + "/cost/seconds", "{:.4f}"),
            "params": w3("base_ready", _b + "/param_count", "{:,.0f}"),
        }})

_t10_rows.append({
    "label": "ITEM deletion (random 1 %) - THERE IS NO REFUSAL ROW AT ALL",
    "row_class": "DECLARED NOT-APPLICABLE, measured - never a zero",
    "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
    "cells": {
        "requests built from deleted rows, tested": w3(
            "base_ready",
            "/per_seed/{s}/DECLARED_NOT_RUN/measured_not_applicable/refuse_token@item-0.01/"
            "applicability/n_deleted_requests_tested", "{:,.0f}"),
        "of those, CERTIFIED impossible": w3(
            "base_ready",
            "/per_seed/{s}/DECLARED_NOT_RUN/measured_not_applicable/refuse_token@item-0.01/"
            "applicability/n_certified_impossible", "{:,.0f}"),
        "a refusal row applies?": w3(
            "base_ready",
            "/per_seed/{s}/DECLARED_NOT_RUN/measured_not_applicable/refuse_token@item-0.01/"
            "applicability/applies", "{}"),
    }})

TABLES["T10"] = {
    "title": "TABLE 10 - REFUSAL DISPLAY ROWS.  A DISPLAY IS NEVER A RESULT.",
    "table_class": "DISPLAY",
    "placement": "MAIN TEXT, labelled DISPLAY",
    "DISPLAY_NEVER_A_RESULT": w1("x_del_test", "/display_refusal/DISPLAY_NEVER_A_RESULT", "{}"),
    "notation": {"K_wells": "32 on our TEST row, 5 on our VAL row; n/a for a byte-matched rival",
                 "rule": "our two rows are at different K and are labelled as such."},
    "pins": {"our object": "single design, Shapes3D value-holdout",
             "comparator": "refuse-token transformer, the ONLY permitted comparator for a refusal DISPLAY row",
             "comparator byte pin": 524288},
    "pin_provenance": [dsrc("base_ready", "/pins/byte_pin"), dsrc("base_ready", "/pins/refuse_threshold"),
                       dsrc("base_ready", "/seeds_present"), dsrc("base_ready", "/NOT_A_SCORED_RUN"),
                       dsrc("x_del_test", "/display_refusal/comparator_status")],
    "seed_count": {"our rows": 1, "the comparator's rows": 3,
                   "NEVER MIX": ("our 1-seed rows and the comparator's 3-seed rows are printed in one "
                                 "table with the seed count AS A COLUMN, and are NOT plotted together "
                                 "without it.")},
    "rows": _t10_rows,
    "WHAT THE TWO SIDES OF THIS TABLE ARE, AND ARE NOT": (
        "OUR side and the COMPARATOR side are NOT the same quantity and are not differenced. Ours is a "
        "cost line: 2 bytes, 0 data rows, no refit, on the object T9 deleted from. The comparator's is a "
        "SEPARATION diagnostic measured at the FEATURE level by spoke/deletion-baselines on its own "
        "16,384-row fit pool, with no judge and no decoder. The head-to-head DISPLAY leg - our refusal "
        "rate against the refuse-token transformer's render rate on the same post-deletion requests - is "
        "NR9, a DECLARED NOT-RUN: the crossover hub's wall clock ran out after the scored rows and the "
        "deletion legs."),
    "the_comparator_status_as_the_artefact_records_it": w1(
        "x_del_test", "/display_refusal/comparator_status", "{}"),
    "why_this_row_exists": w1("x_del_test", "/display_refusal/what_this_row_is_for", "{}"),
    "carried_caveat": w1("x_del_test", "/display_refusal/carried_caveat", "{}"),
    "THE POINT THE HOLES MAKE": (
        "one hole (index 920, red AND cube) is refusable although NO RED CUBE WAS EVER WRITTEN - cube is "
        "held out of the split entirely. Refusal is a SEPARATE mechanism from forgetting and does not "
        "need a deleted row to have existed. T9's probe makes that separation load-bearing rather than "
        "rhetorical: since the downdate does not stop the object answering red, the hole is the only "
        "mechanism here that can."),
    "verdict": {
        "statement": ("our side of every refusal row is MEASURED (2 bytes, 0 rows read, no refit, layout "
                      "asserted equal to the field store's over 1,322 pair slots). The head-to-head "
                      "against the refuse-token transformer is NR9. The comparator itself is measured at "
                      "3 seeds and separates at AUROC 0.8133-0.8755 with a 7.8-12.7 % FALSE refusal rate "
                      "on possible requests, which is why a DISPLAY comparison against it would have to "
                      "print that false-refusal column too."),
        "provenance": [dsrc("x_del_test", "/display_refusal/cost_line/bytes_rewritten"),
                       dsrc("x_del_test", "/display_refusal/comparator_status"),
                       dsrc("base_ready", "/per_seed/0/records/transformer__refuse_token__value-object_hue-0/"
                                          "out_of_range_diagnostic/separation/auroc")],
    },
}

# --------------------------------------------------------------------------------------
# 8aa. TABLE 11 -- THE TWO ABLATIONS.  Composition-only store, and the item store.
#      Both at the 524,288 B pin, TEST, 3 seeds -- ONE pin, so they may share a table.
# --------------------------------------------------------------------------------------
_ABL_R3 = "/arms/%s/r3_comp_2way/novel_tuples_per_corner"
_ABL_R4 = "/arms/%s/r4_comp_nway/novel_tuples_per_corner"


def _abl_cells(arm):
    return {
        "pin (store bytes)": w3s(arm, "/arms/%s/store_bytes", "{:,.0f}"),
        "address-book bytes": w3s(arm, "/arms/%s/address_bytes", "{:,.0f}"),
        "params @ the pin": w3s(arm, "/arms/%s/param_count", "{}"),
        "row 3 (2-way), 60 corners, ceiling 36": w3s(arm, _ABL_R3.replace("%s", "%s"), "{:.4f}"),
        "row 3, fraction of ceiling": w3s(arm, "/arms/%s/r3_comp_2way/frac_of_ceiling", "{:.4f}"),
        "row 4 (n-way), 49 corners, ceiling 72": w3s(arm, _ABL_R4.replace("%s", "%s"), "{:.4f}"),
        "row 4, fraction of ceiling": w3s(arm, "/arms/%s/r4_comp_nway/frac_of_ceiling", "{:.4f}"),
        "class as the artefact records it": w3s(arm, "/arms/%s/class", "{}"),
    }


_t11_rows = [
    {"label": "**`pca_tensor` - the COMPOSITION-ONLY store** (no sufficient statistics, so no exact deletion)",
     "row_class": "ABLATION - composes, cannot forget", "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
     "cells": _abl_cells("pca_tensor")},
    {"label": "**`oracle_content_items` - the ITEM store**, 64 whole written rows read by an ORACLE",
     "row_class": "ABLATION - forgets, loses composition (and it is a CEILING, not an arm)",
     "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
     "cells": _abl_cells("oracle_content_items")},
    {"label": "`rival_mlp_180k` - the learned reference these two are read against",
     "row_class": "comparator", "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
     "cells": _abl_cells("rival_mlp_180k")},
    {"label": "`oracle_content_fine` - the ABSOLUTE content ceiling at the same pin",
     "row_class": "ceiling, NOT AN ARM", "seeds_printed": [0, 1, 2], "n_seeds_printed": 3,
     "cells": _abl_cells("oracle_content_fine")},
]

TABLES["T11"] = {
    "title": ("TABLE 11 - THE TWO ABLATIONS: the composition-only store (composes, cannot forget) and "
              "the ITEM store (forgets, loses composition). w5 compose-field, TEST, 3 seeds, one pin."),
    "table_class": "ablation",
    "placement": "MAIN TEXT",
    "notation": {"K_wells": 5, "n_factors": "row 3 = 2-way, row 4 = n-way; this table does not vary n",
                 "rule": "one pin (524,288 B store bytes), one condition (TEST), 3 seeds on every row."},
    "pins": {"condition": "test", "store bytes": 524288, "K (wells)": 5,
             "metric": "distinct novel valid tuples per corner",
             "corners": "row 3: 60, ceiling 36 / row 4: 49, ceiling 72"},
    "pin_provenance": [dsrc("w5_read_s0", "/arms/pca_tensor/store_bytes"),
                       dsrc("w5_read_s0", "/corner_counts/r3_comp_2way"),
                       dsrc("w5_read_s0", "/corner_counts/r4_comp_nway"),
                       dsrc("w5_read_s0", "/corner_ceilings/r3_comp_2way"),
                       dsrc("w5_read_s0", "/corner_ceilings/r4_comp_nway"),
                       dsrc("w5_read_s0", "/primary")],
    "seed_count": {"seeds on every row": 3,
                   "PRINTED AS A COLUMN": "`seeds (n)` = 3 on every row of T11.",
                   "note": "these are wave-5 scored TEST legs and were never re-run at 5 seeds."},
    "rows": _t11_rows,
    "the_price_of_keeping_items": {
        "item store minus the learned reference, row 4, per seed": derived(
            [-21.0816, -16.5306, -18.0816],
            [dsrc(sid, "/arms/oracle_content_items/r4_comp_nway/novel_tuples_per_corner") for sid in W5]
            + [dsrc(sid, "/arms/rival_mlp_180k/r4_comp_nway/novel_tuples_per_corner") for sid in W5],
            "oracle_content_items r4 minus rival_mlp_180k r4, per seed", fmt="{:+.4f}",
            note=("DERIVED. Their 3-seed mean is -18.5646, which is the programme's '18.6 tuples/corner "
                  "price of keeping items'. Recompute from the six pointers in `from`.")),
        "the composition-only store minus the item store, row 4, per seed": derived(
            [31.2245, 25.2041, 25.8776],
            [dsrc(sid, "/arms/pca_tensor/r4_comp_nway/novel_tuples_per_corner") for sid in W5]
            + [dsrc(sid, "/arms/oracle_content_items/r4_comp_nway/novel_tuples_per_corner") for sid in W5],
            "pca_tensor r4 minus oracle_content_items r4, per seed", fmt="{:+.4f}",
            note="DERIVED. The item store is handed the TRUE FEATURE and the arm is not."),
        "reading": ("the content that WINS composition forfeits exact per-item deletion by construction, "
                    "and that forfeit is priced here at 18.56 tuples/corner on row 4 (3-seed mean). "
                    "The item store is a CEILING handed the true feature, not an arm, so the price is a "
                    "lower bound on what an item store would actually cost."),
    },
    "the_item_store_forgets_exactly_and_loses_composition": {
        "what": ("the other half of the item-store ablation, measured by w5 hub/delete on its own object: "
                 "EVICTION refuses perfectly on the deleted set and FAILS the composition bar."),
        "eviction: refusal on the deleted set": w3p("w5_delete_report",
                                                    "/across_seeds/evict_refuse_on_deleted_set/per_seed/{s}", "{:.4f}"),
        "eviction: composition mean delta (bar is +/-1.0)": w3p(
            "w5_delete_report", "/across_seeds/evict_composition_mean_delta/per_seed/{s}", "{:+.4f}"),
        "the HOLE instead: flag on the deleted set": w3p(
            "w5_delete_report", "/across_seeds/hole_flag_on_deleted_set/per_seed/{s}", "{:.4f}"),
        "the HOLE instead: composition mean delta": w3p(
            "w5_delete_report", "/across_seeds/hole_composition_mean_delta/per_seed/{s}", "{:+.4f}"),
        "row-removal seconds": w3p("w5_delete_report", "/across_seeds/row_removal_s/per_seed/{s}", "{:.4f}"),
        "retrain-from-scratch seconds": w3p(
            "w5_delete_report", "/across_seeds/retrain_from_scratch_s/per_seed/{s}", "{:.4f}"),
        "verdict": ("the operation that is exact on bytes AND free on behaviour is the HOLE, not deletion "
                    "in general: evicting `red` refuses at 1.0000 and costs -2.0899 / -0.9101 / -6.7416 "
                    "composition against a declared +/-1.0 bar, while the hole refuses at 1.0000 and "
                    "costs exactly 0.0000."),
    },
    "cross_reference": ("T8's byte ledger prices the same composition-only ablation on the LADDER's own "
                        "fitted objects: `pca_tensor` at 1.358x-1.372x the 524,288 B reference against "
                        "the single design's 23.886x-24.875x, for composition scores identical at every "
                        "rung and every seed."),
}

# --------------------------------------------------------------------------------------
# 8bb. TABLE 12 -- THE K-CURVE.  K = 5..32 WELLS.  n is not varied here.
# --------------------------------------------------------------------------------------
K_ROWS = [(0, 5), (1, 8), (2, 10), (3, 12), (4, 16), (5, 20), (6, 24), (7, 28), (8, 32)]
_t12_rows = []
for _i, _K in K_ROWS:
    _L = "/rows/%d" % _i
    _P = "/per_K/%d" % _K
    _t12_rows.append({
        "label": "**K = %d**%s" % (_K, "  (= K\\*)" if _K == 32 else ""),
        "row_class": "SMOKE (val sweep) + exact byte arithmetic",
        "seeds_printed": ptr(DOCS["x_cross_val"], _P + "/seeds"),
        "n_seeds_printed": ptr(DOCS["x_cross_val"], _P + "/n_seeds"),
        "cells": {
            "K (wells)": w1("x_ledger_K", _L + "/K", "{:.0f}"),
            "seeds (n) at this K": w1("x_cross_val", _P + "/n_seeds", "{:.0f}"),
            "well width min / max": wcell([("x_ledger_K", _L + "/well_width_min"),
                                           ("x_ledger_K", _L + "/well_width_max")],
                                          fmt="{:.0f}", seeds=["min", "max"], stats=False),
            "rank/width max": w1("x_ledger_K", _L + "/rank_over_width_max", "{:.4f}"),
            "val composition, mean over the seeds run at this K": w1(
                "x_cross_val", _P + "/val_composition_mean", "{:.6f}"),
            "drop vs K = 5 (the frozen rule)": w1("x_cross_val", _P + "/drop_from_K5", "{:+.6f}"),
            "1. atoms B": w1("x_ledger_K", _L + "/p1_atom_bytes", "{:,.0f}"),
            "2. covariance + count B": w1("x_ledger_K", _L + "/p2_covariance_plus_count_bytes", "{:,.0f}"),
            "3. address book B": w1("x_ledger_K", _L + "/p3_address_book_bytes", "{:,.0f}"),
            "3. address book status": w1("x_ledger_K", _L + "/p3_status", "{}"),
            "4. coupling / hole table B": w1("x_ledger_K", _L + "/p4_coupling_hole_table_bytes", "{:,.0f}"),
            "TOTAL, parts 1+2+4 ONLY (no address book)": w1(
                "x_ledger_K", _L + "/total_bytes_parts_1_2_4_only", "{:,.0f}"),
            "x reference, parts 1+2+4": w1("x_ledger_K", _L + "/times_reference_pin_parts_1_2_4", "{:.4f}"),
            "TOTAL, the HONEST FOUR-PART sum": w1("x_ledger_K", _L + "/honest_total_bytes", "{:,.0f}"),
            "x reference, four-part": w1("x_cross_val", _P + "/ledger_four_part/times_reference_pin", "{:.4f}"),
            "1x params a matched rival would carry": w1(
                "x_ledger_K", _L + "/baseline_match_1x_params_float32", "{:,.0f}"),
        }})

TABLES["T12"] = {
    "title": ("TABLE 12 - THE K-CURVE: the four-part byte ledger and val composition against K = 5..32 "
              "WELLS. n is NOT varied in this table."),
    "table_class": "SMOKE (the val sweep labels itself so) + exact arithmetic on the partition shape",
    "placement": "APPENDIX, and the figure is a main-text candidate",
    "SMOKE_NOT_A_CLAIM_as_the_artefacts_record_it": wcell(
        [("x_cross_val", "/SMOKE_NOT_A_CLAIM"), ("x_ledger_K", "/SMOKE_NOT_A_CLAIM")],
        fmt="{}", seeds=["CROSSOVER-val.json", "LEDGER-ACROSS-K.json"], stats=False),
    "notation": {"K": "the number of WELLS - this is the axis of T12",
                 "n": ("NOT VARIED. Every row of T12 reads the same 24 val corners. The n-ladder is T8 "
                       "and it sits at one K."),
                 "rule": "T12 is a K-axis; T8 is an n-axis; they are never the same plot."},
    "pins": {"condition": "val", "A_requested": "64 (63 at K = 32, where A < well width binds)",
             "rank per well": "capped strictly below the well width",
             "selection rule": "K* = max{K : v(5) - mean_seeds v(K) <= 0.02}, frozen 2026-09-21T11:10:24Z",
             "crossover rule": "min{K : drop > 0.10}"},
    "pin_provenance": [dsrc("x_cross_val", "/rule/within"), dsrc("x_cross_val", "/rule/crossover"),
                       dsrc("x_cross_val", "/rule/declared_at"), dsrc("x_cross_val", "/K_star"),
                       dsrc("x_cross_val", "/K_crossover"), dsrc("x_cross_val", "/condition"),
                       dsrc("x_ledger_K", "/reference_pin_bytes")],
    "seed_count": {
        "PRINTED AS A COLUMN": "`seeds (n) at this K` is a column: 3 at K = 5 and K = 32, 1 at every other K.",
        "NEVER MIX": ("a plot of this table MUST print n per point - the two ends carry 3 seeds and the "
                      "seven middle points carry 1. A line drawn through them without n is a lie about "
                      "the evidence."),
    },
    "rows": _t12_rows,
    "K_star": w1("x_cross_val", "/K_star", "{:.0f}"),
    "K_crossover": derived(["NONE INSIDE THE SWEPT RANGE"], [dsrc("x_cross_val", "/K_crossover")],
                           "the stored field is null", fmt="{}"),
    "K_crossover_note": w1("x_cross_val", "/K_crossover_note", "{}"),
    "control gate at K*": {
        "landscape_removed, per seed": w3("x_cross_val", "/control_gate_at_K_star/landscape_removed/per_seed/{s}",
                                          "{:.4f}"),
        "landscape_shuffled, per seed": w3("x_cross_val", "/control_gate_at_K_star/landscape_shuffled/per_seed/{s}",
                                           "{:.4f}"),
        "landscape_shuffled_across_wells, per seed": w3(
            "x_cross_val", "/control_gate_at_K_star/landscape_shuffled_across_wells/per_seed/{s}", "{:.4f}"),
        "random_atoms, per seed": w3("x_cross_val", "/control_gate_at_K_star/random_atoms/per_seed/{s}", "{:.4f}"),
    },
    "THE CAPTION THAT MUST TRAVEL WITH THIS TABLE - TWO CURVES, NOT ONE": (
        "TWO DIFFERENT TOTALS ARE PRINTED AND THEY MOVE IN DIFFERENT DIRECTIONS. "
        "(1) PARTS 1+2+4 ONLY - atoms, covariance/count and the coupling table, with NO address book - "
        "FALLS MONOTONICALLY, 7.4815x -> 2.0663x from K = 5 to K = 32, because the covariance shrinks "
        "6.0x (3,391,552 -> 560,608 B). "
        "(2) THE HONEST FOUR-PART TOTAL - the same three parts PLUS the measured address book - is "
        "U-SHAPED: 7.8545x at K = 5, a minimum of 3.9942x at K = 24, and 4.0877x at K* = 32, because the "
        "ADDRESS BOOK GROWS 5.4x (195,584 -> 1,059,824 B) and at the fine end it grows FASTER than the "
        "covariance falls. "
        "ANY SENTENCE THAT QUOTES '7.48x -> 2.07x' IS QUOTING CURVE (1) AND HAS DROPPED PART 3. The "
        "mechanism for curve (1) is not 'the covariance shrinks faster than the address book grows' - "
        "there is no address book in it. Curve (2), which does contain one, says the opposite at the "
        "fine end."),
    "the_object_never_reaches_the_retired_pin": (
        "the minimum honest four-part total over the whole sweep is 3.9942x 524,288 B, at K = 24. "
        "'Matched bytes at 524,288 B' is not available to this object at rank 64 under any K, which is "
        "why this wave's 1x pin is the object's own total and 524,288 B is a REFERENCE column."),
    "verdict": {
        "statement": ("every K in {5, 8, 10, 12, 16, 20, 24, 28, 32} satisfies the frozen rule, so "
                      "K* = 32 - the finest partition swept. The crossover threshold of 0.10 is never "
                      "reached; the largest drop anywhere is +0.012731 at K = 10. The range was NOT "
                      "extended to manufacture a crossover."),
        "provenance": [dsrc("x_cross_val", "/K_star"), dsrc("x_cross_val", "/K_star_candidates"),
                       dsrc("x_cross_val", "/K_crossover"), dsrc("x_cross_val", "/per_K/10/drop_from_K5")],
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
                "does not exist, and spoke A's own readiness manifest records "
                "`rivals_seeds_present` = [0, 1, 2] for every one of its seven rival families, so the "
                "rivals are missing for seeds 3 and 4 too. `Judge(3)` raises FileNotFoundError. "
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
                     {"source": "x_findings", "path": SOURCES["x_findings"], "locator": "FINDINGS.md section 4"},
                     {"source": "rivals_ready", "path": SOURCES["rivals_ready"],
                      "pointer": "/rivals_seeds_present/mlp"},
                     {"source": "rivals_ready", "path": SOURCES["rivals_ready"], "pointer": "/seeds_complete"}],
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
        "(`/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` records `rivals_seeds_present` = [0, 1, 2] "
        "for all seven families) exist for either seed, "
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
        {"artefact": "`RIVALS-READY.json`", "check": "which seeds have FITTED rivals",
         "expected": "status ready, a seed list per family",
         "found": "status %r, seeds_complete %r, rivals_seeds_present[mlp] %r" % (
             DOCS["rivals_ready"]["status"], DOCS["rivals_ready"]["seeds_complete"],
             DOCS["rivals_ready"]["rivals_seeds_present"]["mlp"]),
         "verdict": "COMPLETE - and it names 3 seeds, not 5"},
        {"artefact": "`BASELINES-READY.json`", "check": "records present on every seed",
         "expected": "150 records, seeds [0, 1, 2]",
         "found": "%d records, seeds %r" % (DOCS["base_ready"]["n_records_total"],
                                            DOCS["base_ready"]["seeds_present"]),
         "verdict": "COMPLETE"},
    ],
}

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
    pc = [[ptr(L, "/rungs/%d/arms/pca_tensor/per_seed/%d/mean" % (n, s)) for s in range(3)] for n in ns]
    ax.plot(ns, [sum(p) / 3 for p in pc], color=SURFACE, linewidth=1.1, linestyle=(0, (2, 2)),
            zorder=6, label="pca_tensor  -  BIT-IDENTICAL to the arm, at 1/18th the bytes")
    ax.plot([1.72, 2.28], [36, 36], color=INK3, linewidth=1.4, linestyle=(0, (4, 2)), zorder=5)
    ax.plot([2.72, 6.28], [72, 72], color=INK3, linewidth=1.4, linestyle=(0, (4, 2)), zorder=5)
    ax.annotate("ceiling 36 (60 corners)", (2, 36), xytext=(6, 3), textcoords="offset points",
                ha="left", fontsize=6.8, color=INK3)
    ax.annotate("ceiling 72 (120 corners, and for n>=3 the SAME 120)", (4.5, 72),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6.8, color=INK3)
    ax.set_ylim(0, 80)
    ax.set_xticks(ns)
    ax.set_xlabel("n  =  number of jointly composed factors  (the RUNG)")
    ax.set_ylabel("distinct valid novel tuples / corner")
    ax.set_title("A.  score vs n   -   K = 5 wells on every point, n_seeds = 3", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    recessive(ax)

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
    ax.annotate("524,288 B  =  1x  (the RETIRED pin, a reference column this wave)", (-0.45, 1.0),
                xytext=(0, -12), textcoords="offset points", ha="left", fontsize=7.0, color=INK2)
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
    out.append(md_table([led.get("seed_label", "part (seed 0 / seed 1 / seed 2)"),
                         "bytes", "how it is obtained"], rows))
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


def render_md(manifest):
    L = []
    A = L.append
    A("# TABLES.md — the paper's scoreboard, with the seed count printed on every row")
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

    _w13_tables_md(A, manifest)

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
            "w13: n (jointly composed factors, the ladder rungs) and K (wells, the partition) are NEVER both written k. Every ladder row states its K; every wells row states its n. See /notation.",
            "w13: the seed count is a COLUMN of every table, never a footnote, and rows at different seed counts are never plotted together without it. See /seed_ledger.",
            "w13: BOTH byte matchings are printed on every row that has them - the object's own measured total (1x) and the 524,288 B reference - because they tell opposite stories about the n-trend.",
            "w13: T7 (K = 32, 1 seed, TEST) and T8 (K = 5, 3 seeds) are different objects at different pins and never share a table or a figure axis.",
            "w13: an artefact that labels itself SMOKE_NOT_A_CLAIM is rendered as SMOKE (T9b, T12), never as a RESULT.",
        ],
        "exactness_definition": EXACTNESS_DEFINITION,
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
        },
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
