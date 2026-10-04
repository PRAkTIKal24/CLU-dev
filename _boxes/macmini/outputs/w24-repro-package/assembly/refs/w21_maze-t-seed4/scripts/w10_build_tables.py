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
    # --- T1: CMAZE-1, hub/maze, TEST, 3 scored seeds ------------------------------------
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
    # --- T3: CMAZE-1, w9/harness tier, 3 seeds -------------------------------------------
    "harness_s0": "/c/chlu/outputs/w9-harness/harness_seed0.json",
    "harness_s1": "/c/chlu/outputs/w9-harness/harness_seed1.json",
    "harness_s2": "/c/chlu/outputs/w9-harness/harness_seed2.json",
    "harness_results": "/c/chlu/outputs/w9-harness/RESULTS.md",
    # --- T4: exact deletion, second implementation ---------------------------------------
    "downdate_test": "/c/chlu/outputs/w8-downdate/DOWNDATE-READY-test.json",
    "downdate_val": "/c/chlu/outputs/w8-downdate/DOWNDATE-READY.json",
    # --- context / declared not-runs ------------------------------------------------------
    "single_design_log": "/c/chlu/outputs/w8-single-design/LOG.md",
    "gentable_spec": "/c/chlu/outputs/hub-generation-table/relaunch/scored/scored_spec.json",
    "gentable_log": "/c/chlu/outputs/hub-generation-table/LOG-iter2.md",
    "gentable_runarm": "/c/chlu/outputs/hub-generation-table/relaunch/scored/run_arm.py",
    "envscout_env": "/c/chlu/outputs/w8-env-scout/ENV.md",
    "envscout_defaults": "/c/chlu/outputs/w8-env-scout/screen_results_DEFAULTS.txt",
    "wave_plan": "/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md",
    "w8_advisor": "/c/chlu/evidence/outputs/w8-advisor-report.md",
}

SEEDS = (0, 1, 2)


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
    numeric = [v for v in vals if isinstance(v, (int, float))]
    out = {"seeds": list(SEEDS)[: len(vals)], "values": vals, "provenance": prov, "format": fmt}
    if len(numeric) == len(vals) and numeric:
        out["mean"] = sum(numeric) / len(numeric)
        out["min"] = min(numeric)
        out["max"] = max(numeric)
    return out


def fmt_cell(c, mean_only=False):
    f = c.get("format", "{:.4f}")
    vals = c["values"]
    if all(isinstance(v, (int, float)) for v in vals):
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
# 3.  TABLE 1 -- THE MAIN SCOREBOARD.  CMAZE-1, hub/maze, TEST, 3/3 scored seeds.
# --------------------------------------------------------------------------------------
m0 = DOCS["maze_s0"]
T1_PINS = {
    "cell": "CMAZE-1 (env-scout adopted generator), hub/maze",
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
    "title": "MAIN SCOREBOARD - CMAZE-1 (w9 hub/maze), held-out TEST, 3 of 3 scored seeds",
    "column_classes": T1_COLUMN_CLASSES,
    "instrument": "exact integer judge, no tolerance anywhere; validity under the exact judge",
    "pins": T1_PINS,
    "pin_provenance": T1_PIN_SRC,
    "caption": (
        "Every cell is the mean of scored seeds 0/1/2 with the three per-seed values in brackets. "
        "Pins: K=8, n_atoms=31, ridge=0.01, AE 12,000 steps, baselines 4,000 steps, read = 410,688 floats "
        "(1,642,752 B deployed) with the float64 sufficient statistic (35,904 B) declared beside it and never "
        "netted in; matched parameters mlp 410,920 / lstm 409,846 / transformer 405,776. "
        "Row 3 is reported BOTH as the scored CONJUNCTION and PER ROUTE, because the ceilings are per route."
    ),
    "result_rows": t1_rows,
    "display_rows": t1_display,
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
    "verdict": {
        "BAR_A_PASS": {"r3": ptr(ca, "/VERDICT/r3_comp_2way/BAR_A_PASS"), "r4": ptr(ca, "/VERDICT/r4_comp_nway/BAR_A_PASS")},
        "BAR_B_FALSIFIED": {"r3": ptr(ca, "/VERDICT/r3_comp_2way/BAR_B_FALSIFIED"), "r4": ptr(ca, "/VERDICT/r4_comp_nway/BAR_B_FALSIFIED")},
        "provenance": [{"source": "enc_crossarm", "path": SOURCES["enc_crossarm"], "pointer": "/VERDICT"}],
    },
}

# --------------------------------------------------------------------------------------
# 5.  TABLE 3 -- CMAZE-1 at the w9/harness tier.  DIFFERENT PIN: 60k params, 500/200.
# --------------------------------------------------------------------------------------
h0 = DOCS["harness_s0"]
T3_PINS = {
    "cell": "CMAZE-1 (env-scout adopted defaults), w9 harness tier `cmaze1`",
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
    ("columns/fragment-lookup r=1", "FRAGMENT-LOOKUP r=1", "reference"),
    ("columns/fragment-lookup r=2", "FRAGMENT-LOOKUP r=2", "reference"),
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
TABLES["T3"] = {
    "title": "CMAZE-1 at the w9/harness tier - learned baselines and the environment's own columns, 3 seeds",
    "instrument": "w9_harness.judge_maze.JUDGE, exact integer judge, no tolerance",
    "pins": T3_PINS,
    "pin_provenance": T3_PIN_SRC,
    "caption": (
        "SEPARATE TABLE BY CHARTER: this tier trains ~60k-parameter arms on 500 instances and evaluates 200, "
        "while T1's arms are matched at ~410k parameters over 96 mazes / 1,208 requests. The two are different "
        "compute budgets on the same environment and do not share a table."
    ),
    "rows": t3_rows,
    "display_rows": t3_display,
}

# --------------------------------------------------------------------------------------
# 6.  TABLE 4 -- exact deletion, two independent implementations, 3 seeds each.
# --------------------------------------------------------------------------------------
dd = DOCS["downdate_test"]
t4_rows = [
    {
        "label": "w9/maze `CluRead.delete_rows_exact` - delete the green target, TEST leg",
        "row_class": "RESULT",
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
        "row_class": "RESULT",
        "implementation": "w8 spoke/downdate-build (different code, different author)",
        "cells": {
            "max abs deviation, worst well": {
                "seeds": list(SEEDS),
                "values": [max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS],
                "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"],
                                "pointer": f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}",
                                "reduction": "max over the 5 wells"} for s in SEEDS],
                "format": "{:.3e}",
                "mean": sum(max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS) / 3,
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
        "summary_from_artefact": {
            "worst_max_abs_dev_over_seeds": ptr(dd, "/downdate_equals_refit/worst_max_abs_dev_over_seeds"),
            "worst_in_float32_ulp_at_unit_scale": ptr(dd, "/downdate_equals_refit/worst_in_float32_ulp_at_unit_scale"),
            "passed_every_seed": ptr(dd, "/downdate_equals_refit/passed_every_seed"),
            "provenance": [{"source": "downdate_test", "path": SOURCES["downdate_test"], "pointer": "/downdate_equals_refit"}],
        },
    },
]
TABLES["T4"] = {
    "title": "Exact deletion - two independent implementations, 3 seeds each, TEST condition",
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
        "Both legs are 3 seeds on generation 2 on their own TEST condition. They are reported side by side "
        "as two implementations of one claim, not as one instrument: the deviations are in different units "
        "(w9/maze reports a RELATIVE residual on the statistic and a projector residual; w8/downdate reports "
        "an ABSOLUTE max deviation per well)."
    ),
    "rows": t4_rows,
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
    fig.suptitle("CMAZE-1 main scoreboard - 3 scored seeds, held-out TEST", fontsize=11,
                 fontweight="bold", y=1.02)
    fig.text(0.5, -0.12, "Dots are the three seeds; bars are their mean. Dashed line is the pipeline ceiling "
                         "(encode -> decode -> execute -> judge) on that row.\nK=8, A=31, ridge=0.01, 410,688 read "
                         "floats; baselines matched at 405,776-410,920 params. Panels have DIFFERENT y-scales.",
             ha="center", fontsize=7.2, color=INK2)
    fig.tight_layout()
    pth = FIG / "fig1_cmaze1_main_scoreboard.png"
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
    dd_abs = [max(ptr(dd, f"/downdate_equals_refit/max_abs_dev_per_well_by_seed/{s}")) for s in SEEDS]
    xs = np.arange(3)
    ax.bar(xs - 0.18, maze_stat, width=0.33, color=CAT[0], zorder=3, edgecolor=SURFACE, linewidth=1.0,
           label="w9/maze  statistic max RELATIVE residual")
    ax.bar(xs + 0.18, dd_abs, width=0.33, color=CAT[1], zorder=3, edgecolor=SURFACE, linewidth=1.0,
           label="w8/downdate  max ABSOLUTE deviation, worst well")
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
    ax.set_title("Exact deletion - two independent implementations, 3 seeds each, TEST")
    recessive(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.text(0.5, -0.04,
             "Different units: w9/maze reports a relative residual on the sufficient statistic "
             "(projector residual 0.0 on every seed);\nw8/downdate reports an absolute max deviation per well "
             "(worst = 0.0094 float32 ulp at unit scale). Both: 0 refits, 0 passes over the data, items never kept.",
             ha="center", fontsize=7.0, color=INK2)
    fig.tight_layout()
    p = FIG / "fig4_exact_deletion.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig5_blind_sequential():
    """The caveat column that must travel with every CMAZE-1 composition row."""
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
    ax.set_title("Every CMAZE-1 composition row carries BLIND-SEQUENTIAL measured on ITS OWN split")
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
      "CLU arm on CMAZE-1.** BAR 1 also fails on all three composition rows: against the best learned "
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
    A("### 1d. DISPLAY rows — inductive bias; the only comparator is the supervised refuser")
    A("")
    A("**Refusal.** The CLU has never seen an impossible example; the supervised refuser is trained on "
      "552 / 562 / 522 of them (seeds 0/1/2) and is evaluated on held-out mazes 48..95 only, so the like-for-like CLU column is "
      "the held-out one. Lower is better on the grounded-possible row (those are FALSE refusals).")
    A("")
    ref_rows = [r for r in t1["display_rows"] if r["label"].startswith("refusal - ")]
    A(_cells_table(ref_rows, extra_first=True))
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
      "unseen-pair split. ⛔ The difference is the REQUEST LAYER, not the environment — see correction C4.")
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
                   [[k, " / ".join(v["format"].format(x) for x in v["values"])] for k, v in r["cells"].items()]))
        if "summary_from_artefact" in r:
            sm = r["summary_from_artefact"]
            A("")
            A(f"> Artefact's own summary: `passed_every_seed` = {sm['passed_every_seed']} · "
              f"`worst_max_abs_dev_over_seeds` = {sm['worst_max_abs_dev_over_seeds']:.4e} · "
              f"`worst_in_float32_ulp_at_unit_scale` = {sm['worst_in_float32_ulp_at_unit_scale']:.6f}")
        A("")
    A("See figure 4.")
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
        A(f"- **Traced to:** " + "; ".join(f"`{pv['path']}` `{pv['pointer']}`" for pv in c["provenance"][:3])
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
            "CMAZE-1's n-way row is reported as a CONJUNCTION and PER ROUTE, because its ceilings are per route.",
            "Every CMAZE-1 composition row carries BLIND-SEQUENTIAL measured on its own split, not the published 0.4980.",
            "A row that was not run is a DECLARED NOT-RUN with a reason and a launch line, never a blank or a zero.",
        ],
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
