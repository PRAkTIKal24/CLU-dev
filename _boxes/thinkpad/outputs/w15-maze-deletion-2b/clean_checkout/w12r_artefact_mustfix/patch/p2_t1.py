"""w12r patch 2 -- T1: MF10 relabel to CMAZE-2, MF3 byte ledger, MF11 fragment split,
MF14 declared not-measured BLIND column on row 4."""
import sys, pathlib
P = pathlib.Path(sys.argv[1])
s = P.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), repr(old[:90]))
    s = s.replace(old, new, n)


# ---- MF10: source-register comments --------------------------------------------------
rep("    # --- T1: CMAZE-1, hub/maze, TEST, 3 scored seeds ------------------------------------",
    "    # --- T1: CMAZE-2 (request layer over CMAZE-1), hub/maze, TEST, 3 scored seeds --------")
rep("    # --- T3: CMAZE-1, w9/harness tier, 3 seeds -------------------------------------------",
    "    # --- T3: CMAZE-1 PROPER, w9/harness tier, 3 seeds ------------------------------------")
rep("# 3.  TABLE 1 -- THE MAIN SCOREBOARD.  CMAZE-1, hub/maze, TEST, 3/3 scored seeds.",
    "# 3.  TABLE 1 -- THE MAIN SCOREBOARD.  CMAZE-2, hub/maze, TEST, 3/3 scored seeds.\n"
    "#     w12r/MF10: this table is CMAZE-2, a REQUEST LAYER over CMAZE-1, and never CMAZE-1.")
rep('# 5.  TABLE 3 -- CMAZE-1 at the w9/harness tier.  DIFFERENT PIN: 60k params, 500/200.',
    '# 5.  TABLE 3 -- CMAZE-1 PROPER at the w9/harness tier.  DIFFERENT PIN: 60k params, 500/200.\n'
    '#     w12r/MF10: DIFFERENT ENVIRONMENT from T1, not merely a different budget.')

# ---- MF10: T1 pins + title -----------------------------------------------------------
rep('    "cell": "CMAZE-1 (env-scout adopted generator), hub/maze",',
    '    "cell": "CMAZE-2 (a REQUEST LAYER over CMAZE-1: engine, solver and exact integer judge '
    'imported unmodified from `w8_envscout.maze_colour`), hub/maze",\n'
    '    "environment": "CMAZE-2",\n'
    '    "environment_is_NOT": "CMAZE-1 - see TABLES.json /environments; T3 is CMAZE-1 proper",')
rep('    "title": "MAIN SCOREBOARD - CMAZE-1 (w9 hub/maze), held-out TEST, 3 of 3 scored seeds",',
    '    "title": "MAIN SCOREBOARD - CMAZE-2 (w9 hub/maze), held-out TEST, 3 of 3 scored seeds",\n'
    '    "environment": "CMAZE-2",')
rep('    "cell": "CMAZE-1 (env-scout adopted defaults), w9 harness tier `cmaze1`",',
    '    "cell": "CMAZE-1 PROPER (env-scout adopted defaults), w9 harness tier `cmaze1`",\n'
    '    "environment": "CMAZE-1",')
rep('    "title": "CMAZE-1 at the w9/harness tier - learned baselines and the environment\'s own columns, 3 seeds",',
    '    "title": "CMAZE-1 PROPER at the w9/harness tier - learned baselines and the environment\'s own columns, 3 seeds",\n'
    '    "environment": "CMAZE-1",')

# ---- the T1 block: ledger, fragment columns, declared gap, caption ---------------------
T1_EXTRA = '''
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

'''
rep("T1_COLUMN_CLASSES = {", T1_EXTRA + "T1_COLUMN_CLASSES = {")

# ---- attach to the T1 dict + rewrite its caption --------------------------------------
rep('''    "result_rows": t1_rows,
    "display_rows": t1_display,
}''',
    '''    "result_rows": t1_rows,
    "display_rows": t1_display,
    "byte_ledger": T1_BYTE_LEDGER,
    "fragment_columns": T1_FRAGMENT,
    "declared_not_measured": T1_DECLARED_NOT_MEASURED,
}''')

rep('''        "Every cell is the mean of scored seeds 0/1/2 with the three per-seed values in brackets. "
        "Pins: K=8, n_atoms=31, ridge=0.01, AE 12,000 steps, baselines 4,000 steps, read = 410,688 floats "
        "(1,642,752 B deployed) with the float64 sufficient statistic (35,904 B) declared beside it and never "
        "netted in; matched parameters mlp 410,920 / lstm 409,846 / transformer 405,776. "
        "Row 3 is reported BOTH as the scored CONJUNCTION and PER ROUTE, because the ceilings are per route."''',
    '''        "THE ENVIRONMENT IS CMAZE-2: a REQUEST LAYER over CMAZE-1, with the engine, the solver and the "
        "exact integer judge imported unmodified from `w8_envscout.maze_colour`. It is NOT CMAZE-1, and it "
        "must never share a table with T3, which is CMAZE-1 proper - a different environment, not merely a "
        "different budget. "
        "Every cell is the mean of scored seeds 0/1/2 with the three per-seed values in brackets. "
        "Pins: K=8, n_atoms=31, ridge=0.01, AE 12,000 steps, baselines 4,000 steps, read = 410,688 floats "
        "(1,642,752 B deployed) with the float64 sufficient statistic (35,904 B) declared beside it and never "
        "netted in; matched parameters mlp 410,920 / lstm 409,846 / transformer 405,776. "
        "The four-part byte ledger is printed as its own block below and is never netted into one number. "
        "Row 3 is reported BOTH as the scored CONJUNCTION and PER ROUTE, because the ceilings are per route."''')

P.write_text(s, encoding="utf-8")
print("patch 2 applied")
