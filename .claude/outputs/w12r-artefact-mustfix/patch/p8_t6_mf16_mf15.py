"""w12r patch 8 -- TABLE 6 (the w5 compose-field byte and parameter ledger: MF12, MF13, SF2),
the MF16 qualifier on NR3, and the MF15 refusal reading on T1."""
import sys, pathlib
P = pathlib.Path(sys.argv[1])
s = P.read_text(encoding="utf-8")


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), repr(old[:90]))
    s = s.replace(old, new, n)


# ---- register the three w5 scored READ-TABLEs ----------------------------------------
OLD = '    "single_design_log": "/c/chlu/outputs/w8-single-design/LOG.md",'
NEW = (OLD + "\n"
       '    # --- w12r/MF12+MF13: the w5 compose-field byte and parameter ledger ---------------\n'
       '    "w5_read_s0": "/c/chlu/outputs/w5-compose-field/scored/test_seed0/READ-TABLE.json",\n'
       '    "w5_read_s1": "/c/chlu/outputs/w5-compose-field/scored/test_seed1/READ-TABLE.json",\n'
       '    "w5_read_s2": "/c/chlu/outputs/w5-compose-field/scored/test_seed2/READ-TABLE.json",\n'
       '    "w5_log": "/c/chlu/outputs/w5-compose-field/LOG.md",')
rep(OLD, NEW)

T6 = '''
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

'''
rep("# --------------------------------------------------------------------------------------\n"
    "# 7.  DECLARED NOT-RUNS -- never a blank, never a zero, each with its launch line.",
    T6 + "# --------------------------------------------------------------------------------------\n"
         "# 7.  DECLARED NOT-RUNS -- never a blank, never a zero, each with its launch line.")

# ---- MF16: NR3's unconditional K>=36 statement is conditional -------------------------
ANCHOR_NR3 = ('        "why": "the object was falsified on `val` before a scored run was justified: '
              'the covariance costs "')
rep(ANCHOR_NR3,
    '''        "why_w12r_qualifier": (
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
''' + ANCHOR_NR3)

P.write_text(s, encoding="utf-8")
print("patch 8 applied")
