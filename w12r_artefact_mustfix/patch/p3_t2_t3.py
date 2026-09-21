"""w12r patch 3 -- T2 byte ledger (MF3) and T3 (CMAZE-1 proper) ledger status + caption."""
import sys, pathlib
P = pathlib.Path(sys.argv[1])
s = P.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), repr(old[:90]))
    s = s.replace(old, new, n)


T2_LEDGER = '''# --- w12r/MF3: the four-part byte ledger for T2's store -------------------------------
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

TABLES["T2"] = {'''
rep('TABLES["T2"] = {', T2_LEDGER)
rep('''    "rows": t2_rows,
    "verdict": {''',
    '''    "rows": t2_rows,
    "byte_ledger": T2_BYTE_LEDGER,
    "verdict": {''')

T3_LEDGER = '''# --- w12r/MF3: T3 carries no store.  Declared, not omitted. ---------------------------
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

TABLES["T3"] = {'''
rep('TABLES["T3"] = {', T3_LEDGER)
rep('''    "rows": t3_rows,
    "display_rows": t3_display,
}''',
    '''    "rows": t3_rows,
    "display_rows": t3_display,
    "byte_ledger": T3_BYTE_LEDGER,
}''')

# T3 caption: it said "the same environment".  MF10 / SF19: different ENVIRONMENT too.
rep('''        "SEPARATE TABLE BY CHARTER: this tier trains ~60k-parameter arms on 500 instances and evaluates 200, "
        "while T1's arms are matched at ~410k parameters over 96 mazes / 1,208 requests. The two are different "
        "compute budgets on the same environment and do not share a table."''',
    '''        "SEPARATE TABLE BY CHARTER, AND ON TWO GROUNDS, NOT ONE. (i) ENVIRONMENT: this table is CMAZE-1 "
        "PROPER; T1 is CMAZE-2, a REQUEST LAYER over it. (ii) BUDGET: this tier trains ~60k-parameter arms "
        "on 500 instances and evaluates 200, while T1's arms are matched at ~410k parameters over 96 mazes / "
        "1,208 requests. w12r/MF10: the earlier wording of this caption said `the same environment`, which "
        "was wrong - T1 and T3 are a DIFFERENT ENVIRONMENT, not merely a different budget, and they must "
        "never share a table."''')

# T3's fragment rows: label them as the CMAZE-1 tier columns, not as T1's same-maze quantity.
rep('''    ("columns/fragment-lookup r=1", "FRAGMENT-LOOKUP r=1", "reference"),
    ("columns/fragment-lookup r=2", "FRAGMENT-LOOKUP r=2", "reference"),''',
    '''    ("columns/fragment-lookup r=1", "FRAGMENT-LOOKUP r=1 (CMAZE-1 tier column)", "reference"),
    ("columns/fragment-lookup r=2", "FRAGMENT-LOOKUP r=2 (CMAZE-1 tier column)", "reference"),''')

P.write_text(s, encoding="utf-8")
print("patch 3 applied")
