"""Render one or more legs' READ-TABLE.json into the table this hub reports.

⛔ It renders; it does not decide.  Every verdict printed comes from
``exp_gen_read.build_table``'s paired estimand against the margin declared in
`PREREG-w5-compose-field-1.md`, and every row carries its own class, so w5-common §1's
"a win over a RIVAL is a RESULT, a win over anything else is an ABLATION" is a property
of the table rather than of the prose around it.

    python scripts/w5_compose_field_report.py LEG [LEG ...] > table.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

#: Print order: the arm, the bar, the rest of the rival set, the ablations, the floors,
#: the controls, then the ceilings LAST -- a ceiling is not a result and must not be the
#: first number a reader's eye lands on.
ORDER = [
    "pca_tensor",
    "rival_mlp_180k",
    "rival_table", "rival_mlp", "rival_cont_mlp", "rival_transformer", "rival_lstm",
    "rival_lace", "rival_poe_ebm",
    "span_head", "pca_tensor_sum",
    "pca_lookup_argmax", "pca_lookup_convex", "field_lookup_argmax",
    "pca_landscape_removed", "pca_landscape_shuffled", "pca_random_refit",
    "pca_span_rotated",
    "oracle_content_pool", "oracle_span_field", "oracle_span",
    "oracle_content_items", "oracle_content_oracle", "oracle_content_fine",
    "oracle_true_feature",
]

#: The byte multiple w5-common §3.9 requires beside these two wherever they appear.
BYTE_MULTIPLE = {"rival_lace": "95.1x the store budget",
                 "rival_poe_ebm": "27.1x the store budget"}


def fmt(x, n=3):
    if x is None:
        return "—"
    return f"{x:.{n}f}" if isinstance(x, float) else str(x)


def load(leg: Path) -> dict:
    with open(leg / "READ-TABLE.json", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv) -> int:
    legs = [Path(a) for a in argv]
    tabs = {p.name: load(p) for p in legs}
    names = list(tabs)

    print("## Rows 3 and 4 — every leg, every seed\n")
    print("**Labels are used ONCE at fit time to align the wells and at the address "
          "book, and nowhere else.** The store's byte ledger and the address book's are "
          "printed separately and are **never netted together**. Metric = distinct "
          "valid novel tuples per corner; row 4 = 49 n-way corners (per-corner ceiling "
          "72), row 3 = 60 pair sub-corners (ceiling 36) — ⚠ the two rows are not on "
          "one scale. Cache **generation 2**.\n")

    head = ("| leg | class | store B | address B | params | "
            + " | ".join(f"r4 {n}" for n in names) + " | "
            + " | ".join(f"r3 {n}" for n in names) + " |")
    print(head)
    print("|" + "---|" * (5 + 2 * len(names)))

    present = [a for a in ORDER
               if any(a in t["arms"] or a in t["not_run"] for t in tabs.values())]
    # anything the runner produced that ORDER forgot still gets printed, at the end
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

    print("\n## The paired estimands — the only thing that decides a row\n")
    print("Primary = `" + str(next(iter(tabs.values()))["primary"]) + "`, margin **+"
          + str(next(iter(tabs.values()))["margin_declared"])
          + "** novel tuple/corner, declared in the PREREG before the first held-out "
            "number. `beats` requires the mean paired delta to clear the margin AND "
            "the bootstrap 2.5 % quantile to clear it too.\n")
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
                  fmt(((r or {}).get("bootstrap_ci95") or [None, None])[0]) for r in rows)
              + " | " + " | ".join(
                  str((r or {}).get("beats_beyond_margin")) for r in rows) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
