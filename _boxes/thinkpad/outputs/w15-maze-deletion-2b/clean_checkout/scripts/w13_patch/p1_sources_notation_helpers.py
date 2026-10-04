#!/usr/bin/env python
"""w13 patch 1/4 -- register wave-12's artefacts, fix the n/K notation, and add the
n-seed cell helper.  Exact-string patch; every anchor must occur EXACTLY once."""
import io, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "build_tables.py"
t = SRC.read_text(encoding="utf-8")

def sub(anchor, replacement):
    n = t.count(anchor)
    assert n == 1, f"anchor occurs {n} times, expected 1:\n{anchor[:120]!r}"
    return t.replace(anchor, replacement)

# ---------------------------------------------------------------- 1. the source register
A = '''    "wave_plan": "/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md",
    "w8_advisor": "/c/chlu/evidence/outputs/w8-advisor-report.md",
}
'''
B = '''    "wave_plan": "/c/chlu/evidence/outputs/advisor-wave-plan-w8-w10.md",
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
    # the plan this fold is executed against
    "wave_plan_12": "/c/chlu/evidence/outputs/advisor-wave-plan-w12-w14.md",
}
'''
t = sub(A, B)

# ---------------------------------------------------------------- 2. notation + seed ledger
A = '''def native(p: str) -> Path:
    """The manifest records MSYS-style paths (/c/chlu/...); Python needs C:/chlu/... on Windows."""'''
B = '''# --------------------------------------------------------------------------------------
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
    """The manifest records MSYS-style paths (/c/chlu/...); Python needs C:/chlu/... on Windows."""'''
t = sub(A, B)

# ---------------------------------------------------------------- 3. the n-seed cell helper
A = '''# --- w12r additions: two NON-cell carriers, so verify_tables.py never mis-asserts them ----'''
B = '''# --- w13 addition: a cell whose entries are (source, pointer) pairs, one per seed. --------
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
    if stats and nums and len(nums) == len(vals):
        out["mean"] = sum(nums) / len(nums)
        out["min"] = min(nums)
        out["max"] = max(nums)
    return out


def w1(sid, pointer, fmt="{:.4f}", seed=0):
    """A ONE-SEED cell.  Exists so that no 1-seed number can be printed without its seed."""
    return wcell([(sid, pointer)], fmt=fmt, seeds=[seed])


def w3(sid, tmpl, fmt="{:.4f}", seeds=(0, 1, 2)):
    """A THREE-SEED cell out of one json: `tmpl` carries one `{s}` placeholder."""
    return wcell([(sid, tmpl.format(s=s)) for s in seeds], fmt=fmt, seeds=list(seeds))


# --- w12r additions: two NON-cell carriers, so verify_tables.py never mis-asserts them ----'''
t = sub(A, B)

SRC.write_text(t, encoding="utf-8")
print("p1 OK ->", SRC)
