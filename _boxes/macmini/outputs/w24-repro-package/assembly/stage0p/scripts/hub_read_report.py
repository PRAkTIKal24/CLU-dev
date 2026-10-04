"""Render one or more legs' READ-TABLE.json into the markdown this hub reports.

⛔ It renders; it does not decide.  Every verdict it prints comes from
``exp_gen_read.build_table``'s paired estimand against the margin declared in the PREREG,
and every row carries its own class, so "a win over a RIVAL is a RESULT and a win over
anything else is an ABLATION" (w4-common section 1) is a property of the table and not of
the prose around it.

    python scripts/hub_read_report.py LEG [LEG ...] > table.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = [
    "oracle_span", "tensor_addr", "span_head", "tensor_addr_sum", "corr_linear",
    "pvm_span", "pvm_table", "knn_proto",
    "rival_table", "rival_mlp", "rival_cont_mlp", "rival_transformer", "rival_lstm",
    "rival_lace", "rival_poe_ebm",
    "lookup_argmax", "lookup_convex",
    "landscape_removed", "landscape_shuffled", "random_span_refit", "span_rotated",
]


def fmt(x, n=3):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def load(leg: Path) -> dict:
    with open(leg / "READ-TABLE.json", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv) -> int:
    legs = [Path(a) for a in argv]
    tabs = {str(p): load(p) for p in legs}
    print("## Rows 3 and 4 — every arm, every leg\n")
    print("**Labels are used ONCE at fit time to align wells and at the address book, "
          "and nowhere else.** Bytes: the store's ledger and the address book's are "
          "printed separately and never netted together.\n")
    head = ("| arm | class | store B | address B | "
            + " | ".join(f"r4 novel/corner ({Path(l).name})" for l in tabs)
            + " | " + " | ".join(f"r3 novel/corner ({Path(l).name})" for l in tabs) + " |")
    print(head)
    print("|" + "---|" * (4 + 2 * len(tabs)))
    names = [a for a in ORDER if any(a in t["arms"] or a in t["not_run"]
                                     for t in tabs.values())]
    for a in names:
        any_t = next((t for t in tabs.values() if a in t["arms"]), None)
        if any_t is None:
            reason = next(t["not_run"][a] for t in tabs.values() if a in t["not_run"])
            print(f"| `{a}` | DECLARED NOT-RUN | — | — |"
                  + " — |" * (2 * len(tabs)) + f" {reason}")
            continue
        b = any_t["arms"][a]
        r4 = [fmt((t["arms"].get(a, {}).get("r4_comp_nway") or {})
                  .get("novel_tuples_per_corner")) for t in tabs.values()]
        r3 = [fmt((t["arms"].get(a, {}).get("r3_comp_2way") or {})
                  .get("novel_tuples_per_corner")) for t in tabs.values()]
        print(f"| `{a}` | {b['class']} | {fmt(b.get('store_bytes'))} | "
              f"{fmt(b.get('address_bytes'))} | " + " | ".join(r4) + " | "
              + " | ".join(r3) + " |")

    print("\n## The paired estimand — the corner is the unit, never a single cell\n")
    for leg, t in tabs.items():
        print(f"\n### {Path(leg).name} (margin declared in advance: "
              f"{t['margin_declared']:+.2f} novel tuples/corner)\n")
        print("| row | comparison | class of the comparison | mean paired Δ | "
              "bootstrap 95% | sign-test p | corners won/tied/lost | beats the margin |")
        print("|" + "---|" * 8)
        for key, e in sorted(t["estimands"].items()):
            row, rest = key.split(":", 1)
            ci = e["bootstrap_ci95"]
            print(f"| {row} | {e['arm']} − {e['control']} | {e['comparison_class']} | "
                  f"{e['mean_paired_delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                  f"{e['sign_test_p_two_sided']:.3g} | "
                  f"{e['corners_arm_wins']}/{e['corners_tied']}/{e['corners_control_wins']}"
                  f" | {'YES' if e['beats_beyond_margin'] else 'no'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
