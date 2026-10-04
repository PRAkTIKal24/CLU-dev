"""Render w6 hub/compose-physics' scored legs into the table this hub reports.

⛔ It renders; it does not decide.  Every verdict printed comes from
``exp_gen_read.build_table``'s paired estimand against the margin declared in
`PREREG-w6-compose-physics-1.md`, and every row carries its own class, so w6-common §1's
"a win over a RIVAL is a RESULT, a win over an earlier CLU configuration is an ABLATION"
is a property of the table rather than of the prose around it.

⭐ It also evaluates the PREREG's bars **including the TIE TEST**, which needs BOTH ends
of the bootstrap interval and not only the 2.5 % quantile that ``beats_beyond_margin``
uses.  The tie test is the one that decides "decoration", so it is computed here and
printed with its inputs rather than asserted in prose.

    python scripts/w6_compose_physics_report.py LEG [LEG ...] > SCORED-TABLE.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MARGIN = 1.0

#: Print order: the arm, then the parameter-identical ladder it is read against, then the
#: bar, then the rest of the rival set, then ablations, floors, controls, and the ceilings
#: LAST -- a ceiling is not a result and must not be the first number a reader's eye hits.
ORDER = [
    "pca_settle",
    "pca_settle_onestep", "pca_settle_f05_conv", "pca_settle_zerostep",
    "pca_tensor",
    "rival_mlp_180k",
    "rival_table", "rival_mlp", "rival_cont_mlp", "rival_transformer", "rival_lstm",
    "rival_lace", "rival_poe_ebm",
    "span_head", "pca_tensor_sum",
    "pca_settle_lookup", "pca_lookup_argmax", "pca_lookup_convex",
    "pca_settle_removed", "pca_settle_shuffled",
    "pca_landscape_removed", "pca_landscape_shuffled", "pca_random_refit",
    "oracle_content_pool", "oracle_span_field", "oracle_span",
    "oracle_content_items", "oracle_content_fine", "oracle_true_feature",
]

#: The byte multiple w6-common requires beside these two wherever they appear.
BYTE_MULTIPLE = {"rival_lace": "95.1x the store budget",
                 "rival_poe_ebm": "27.1x the store budget"}

#: The PREREG's bars.  ``(id, control, what it would mean)`` -- all against the primary.
BARS = [
    ("B1", "pca_settle_zerostep",
     "the DIAL: dynamics ON vs OFF at identical parameters and identical bytes"),
    ("B2", "pca_tensor", "the INCUMBENT and the bar"),
    ("B3", "pca_settle_onestep", "the RELAXATION: converged vs one step"),
    ("C1", "pca_settle_lookup", "charter C.1/C.4: the matched-bytes lookup"),
]
ROWS = [("r4_comp_nway", "row 4"), ("r3_comp_2way", "row 3")]


def fmt(x, n=3):
    if x is None:
        return "—"
    return f"{x:.{n}f}" if isinstance(x, float) else str(x)


def load(leg: Path, primary: str = None) -> dict:
    p = (leg / (f"READ-TABLE-primary-{primary}.json" if primary
                else "READ-TABLE.json"))
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv) -> int:
    legs = [Path(a) for a in argv]
    tabs = {p.name: load(p) for p in legs}
    names = list(tabs)

    print("## Rows 3 and 4 — every leg, every seed  (scored 1/2)\n")
    print("**Labels are used ONCE at fit time to align the wells and at the address "
          "book, and nowhere else. The CONTENT is unsupervised, and so is the "
          "coupling.** The store's byte ledger, the address book's and the coupling's "
          "are printed separately and are **never netted together**. Metric = distinct "
          "valid novel tuples per corner; row 4 = 49 n-way corners (per-corner ceiling "
          "72), row 3 = 60 pair sub-corners (ceiling 36) — ⚠ the two rows are not on "
          "one scale. Cache **generation 2**.\n")
    print("⭐ `pca_tensor`, every rival and every ceiling are wave 5's own scored "
          "objects, copied bit-for-bit, **not re-run** — so this table is continuous "
          "with the one wave 5 published.\n")

    head = ("| leg | class | store B | address B | params | "
            + " | ".join(f"r4 {n}" for n in names) + " | "
            + " | ".join(f"r3 {n}" for n in names) + " |")
    print(head)
    print("|" + "---|" * (5 + 2 * len(names)))

    present = [a for a in ORDER
               if any(a in t["arms"] or a in t["not_run"] for t in tabs.values())]
    present += sorted({a for t in tabs.values() for a in t["arms"]} - set(present))

    for a in present:
        any_t = next((t for t in tabs.values() if a in t["arms"]), None)
        if any_t is None:
            reason = next(t["not_run"][a] for t in tabs.values() if a in t["not_run"])
            print(f"| `{a}` | **DECLARED NOT-RUN** | — | — | — |"
                  + " — |" * (2 * len(names)) + f" {reason}")
            continue
        b = any_t["arms"][a]
        note = BYTE_MULTIPLE.get(a)
        r4 = [fmt((t["arms"].get(a, {}).get("r4_comp_nway") or {})
                  .get("novel_tuples_per_corner")) for t in tabs.values()]
        r3 = [fmt((t["arms"].get(a, {}).get("r3_comp_2way") or {})
                  .get("novel_tuples_per_corner")) for t in tabs.values()]
        label = f"`{a}`" + (f" ⚠ {note}" if note else "")
        print(f"| {label} | {b['class']} | {fmt(b.get('store_bytes'), 0)} | "
              f"{fmt(b.get('address_bytes'), 0)} | {fmt(b.get('param_count'), 0)} | "
              + " | ".join(r4) + " | " + " | ".join(r3) + " |")

    # ------------------------------------------------------------------
    print("\n## The PREREG's bars — evaluated, with both ends of the interval\n")
    prim = str(next(iter(tabs.values()))["primary"])
    print(f"Primary = `{prim}`, margin **±{MARGIN}** novel tuple/corner, declared in "
          "`PREREG-w6-compose-physics-1.md` before the first held-out number.\n")
    print("`beats` requires the mean paired delta to clear **+" f"{MARGIN}" "** AND the "
          "bootstrap 2.5 % quantile to clear it too, on **all three seeds**.\n")
    print("| bar | comparator | row | " + " | ".join(f"delta {n}" for n in names)
          + " | " + " | ".join(f"95% CI {n}" for n in names) + " | verdict |")
    print("|" + "---|" * (4 + 2 * len(names)))
    for bid, ctrl, _what in BARS:
        for rkey, rlab in ROWS:
            k = f"{rkey}:{prim}_vs_{ctrl}"
            rows = [t["estimands"].get(k) for t in tabs.values()]
            if not any(rows):
                continue
            d = [(r or {}).get("mean_paired_delta") for r in rows]
            ci = [((r or {}).get("bootstrap_ci95") or [None, None]) for r in rows]
            met = all(x is not None and x > MARGIN for x in d) and all(
                c[0] is not None and c[0] > MARGIN for c in ci)
            verdict = "**MET**" if met else (
                "**NOT met — NEGATIVE**" if all(x is not None and x < 0 for x in d)
                else "**NOT met**")
            print(f"| {bid} | `{ctrl}` | {rlab} | "
                  + " | ".join(fmt(x) for x in d) + " | "
                  + " | ".join(f"[{fmt(c[0])}, {fmt(c[1])}]" for c in ci)
                  + f" | {verdict} |")

    # ------------------------------------------------------------------
    print("\n### The TIE TEST\n")
    print("⭐ A tie is what decides **decoration**, so it is computed from BOTH ends of "
          "the interval: |delta| < " f"{MARGIN}" " **and** the whole 95 % interval "
          "inside ±" f"{MARGIN}" ", on both rows and all three seeds.\n")
    print("| comparator | row | " + " | ".join(f"delta {n}" for n in names) + " | "
          + " | ".join(f"95% CI {n}" for n in names) + " | tie? |")
    print("|" + "---|" * (3 + 2 * len(names)))
    for ctrl in ("pca_settle_zerostep", "pca_tensor", "pca_settle_onestep"):
        for rkey, rlab in ROWS:
            k = f"{rkey}:{prim}_vs_{ctrl}"
            rows = [t["estimands"].get(k) for t in tabs.values()]
            if not any(rows):
                continue
            d = [(r or {}).get("mean_paired_delta") for r in rows]
            ci = [((r or {}).get("bootstrap_ci95") or [None, None]) for r in rows]
            tie = (all(x is not None and abs(x) < MARGIN for x in d)
                   and all(c[0] is not None and c[0] > -MARGIN and c[1] < MARGIN
                           for c in ci))
            print(f"| `{ctrl}` | {rlab} | " + " | ".join(fmt(x) for x in d) + " | "
                  + " | ".join(f"[{fmt(c[0])}, {fmt(c[1])}]" for c in ci)
                  + f" | {'**TIE**' if tie else '**NOT a tie**'} |")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # ⭐ THE LADDER'S OWN TIE TESTS.  ``build_table`` computes every estimand against ONE
    # primary, so the gentler rungs' comparisons against the incumbent need their own
    # table.  They are produced by re-running ONLY the table stage with a different
    # --primary (no rescoring of anything) and saved beside the canonical one.
    print("\n### The ladder's own tie tests against the incumbent\n")
    print("⭐ This is the row that decides **decoration**: does a rung where the "
          "dynamics barely move the state tie `pca_tensor`?\n")
    print("| rung | vs | row | " + " | ".join(f"delta {n}" for n in names) + " | "
          + " | ".join(f"95% CI {n}" for n in names) + " | tie? |")
    print("|" + "---|" * (4 + 2 * len(names)))
    for rung in ("pca_settle_f05_conv", "pca_settle_onestep"):
        try:
            alt = {p.name: load(p, rung) for p in legs}
        except FileNotFoundError:
            continue
        for rkey, rlab in ROWS:
            k = f"{rkey}:{rung}_vs_pca_tensor"
            rows = [t["estimands"].get(k) for t in alt.values()]
            if not any(rows):
                continue
            d = [(r or {}).get("mean_paired_delta") for r in rows]
            ci = [((r or {}).get("bootstrap_ci95") or [None, None]) for r in rows]
            tie = (all(x is not None and abs(x) < MARGIN for x in d)
                   and all(c[0] is not None and c[0] > -MARGIN and c[1] < MARGIN
                           for c in ci))
            print(f"| `{rung}` | `pca_tensor` | {rlab} | "
                  + " | ".join(fmt(x) for x in d) + " | "
                  + " | ".join(f"[{fmt(c[0])}, {fmt(c[1])}]" for c in ci)
                  + f" | {'**TIE**' if tie else '**NOT a tie**'} |")

    # ------------------------------------------------------------------
    print("\n### The read-side byte ledger, per leg (never netted into the store)\n")
    print("| leg | store B | address B | coupling B | standardisation B | "
          "read-side added vs the incumbent |")
    print("|---|---|---|---|---|---|")
    leg0 = legs[0]
    for a in ("pca_tensor", "pca_settle_zerostep", "pca_settle_f05_conv",
              "pca_settle_onestep", "pca_settle"):
        p = leg0 / "arms" / a / "arm.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        cb = m.get("coupling_bytes") or {}
        c = int(cb.get("coupling_bytes", 0))
        sd = int(cb.get("standardisation_bytes", 0))
        print(f"| `{a}` | {int(m['byte_ledger']['deployed_bytes'])} | "
              f"{int((m.get('address_bytes') or {}).get('address_bytes', 0))} | "
              f"{c if c else '—'} | {sd if sd else '—'} | "
              f"{(c + sd) if (c or sd) else 0} |")
    print("\n⚠ Every settle leg pays the SAME added read-side bytes, so the ladder "
          "B1/B3 is parameter-identical and byte-identical end to end. ⛔ The store is "
          "524,288 B on every row, tolerance 0.\n")

    print("\n## Every paired estimand against the primary\n")
    print("| comparison | class of comparator | " + " | ".join(
        f"delta {n}" for n in names) + " | " + " | ".join(
        f"ci95_low {n}" for n in names) + " | " + " | ".join(
        f"beats {n}" for n in names) + " |")
    print("|" + "---|" * (2 + 3 * len(names)))
    keys = []
    for t in tabs.values():
        for k in t["estimands"]:
            if k not in keys:
                keys.append(k)
    for k in sorted(keys):
        rows = [t["estimands"].get(k) for t in tabs.values()]
        first = next((r for r in rows if r), None)
        if first is None:
            continue
        print(f"| `{k}` | {first.get('comparison_class', '?')} | "
              + " | ".join(fmt((r or {}).get("mean_paired_delta")) for r in rows)
              + " | " + " | ".join(
                  fmt(((r or {}).get("bootstrap_ci95") or [None, None])[0])
                  for r in rows)
              + " | " + " | ".join(
                  str((r or {}).get("beats_beyond_margin")) for r in rows) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
