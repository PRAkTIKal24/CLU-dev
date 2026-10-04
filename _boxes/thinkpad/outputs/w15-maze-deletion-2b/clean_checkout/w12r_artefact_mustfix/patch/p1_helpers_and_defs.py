"""w12r patch 1 -- helpers, exactness definition, environment definitions, seed/config counts."""
import sys, pathlib
P = pathlib.Path(sys.argv[1])
s = P.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new, n)


HELPERS = '''# --- w12r additions: two NON-cell carriers, so verify_tables.py never mis-asserts them ----
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
        return fmt_cell(c)
    f = c.get("format", "{:,.0f}")
    vals = c.get("derived", c.get("printed"))
    if isinstance(vals, (int, float, str, bool)):
        vals = [vals]
    return " / ".join(f.format(v) if isinstance(v, (int, float)) else str(v) for v in vals)


def fmt_cell(c, mean_only=False):'''

rep("def fmt_cell(c, mean_only=False):", HELPERS)

DEFS = '''SEEDS = (0, 1, 2)

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
}'''

rep("SEEDS = (0, 1, 2)", DEFS)

P.write_text(s, encoding="utf-8")
print("patch 1 applied")
