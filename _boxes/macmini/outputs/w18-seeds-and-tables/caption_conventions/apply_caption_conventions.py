"""w18 seeds-and-tables: name the byte convention INSIDE five draft tables (captions / one ledger label).

Guarded, idempotent, exact-string edits. Each edit must match exactly once; a second run reports
`already applied` and writes nothing. No `\\PLACEHOLDER{}` is touched and the script refuses to run if
the count of PLACEHOLDER tokens would change. Only banked conventions are added: no w15 or w16 number
enters the draft (AMENDMENT-1 gate).

Sources of each convention (build-verified cells in ../TABLES.md §8):
  tab:1x         w13-seeds-and-tables/TABLES.md:642-643; w12b-nr6-and-kstar/FINDINGS.md:133
  tab:hole       w13-seeds-and-tables/TABLES.md:431; w12-deletion-baselines/FINDINGS.md:122-123
  tab:rivals-del w12-deletion-baselines/FINDINGS.md:122-123
  tab:verb       w14-read-verb/FINAL-REPORT.md:262-266
  tab:maze-del   w13-seeds-and-tables/TABLES.md:162, 164

Usage: python apply_caption_conventions.py DRAFT.tex [OUT.tex]   (writes OUT, or DRAFT in place)
"""
import sys
from pathlib import Path

EDITS = [
    ("tab:1x",
     "3 seeds, held out, 0 scored runs.} Per-seed margins (seeds 0 / 1 / 2) and 3-seed mean.}",
     "3 seeds, held out, 0 scored runs.} Per-seed margins (seeds 0 / 1 / 2) and 3-seed mean. Opponent\n"
     "parameters are float32, matched to the object's own four-part total (symmetric-triangle float64\n"
     "statistic).}"),
    ("tab:hole",
     "the same post-deletion requests is a declared not-run.}",
     "the same post-deletion requests is a declared not-run. Bytes rewritten: ours is one byte per hole in\n"
     "the 1,322 B coupling table; the comparator's is its whole parameter blob.}"),
    ("tab:rivals-del",
     "seed 0.} Data rows read and bytes rewritten are seed-invariant.",
     "seed 0.} Data rows read and bytes rewritten are seed-invariant; bytes rewritten is the rival's whole\n"
     "parameter blob."),
    ("tab:verb",
     "(unseen pair).}\n\\label{tab:verb}",
     "(unseen pair). The lookup floor is matched to the store's read floats; at this configuration the\n"
     "match does not bind, and the floor is the whole training set.}\n\\label{tab:verb}"),
    ("tab:maze-del",
     "ledger: atoms $\\cdot$ statistic (float64) $\\cdot$ address book & 3 &",
     "ledger: atoms (float32) $\\cdot$ statistic (float64) $\\cdot$ address book (float32) & 3 &"),
]


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    text = src.read_text(encoding="utf-8")
    n_ph = text.count("\\PLACEHOLDER")
    for label, old, new in EDITS:
        if text.count(new) == 1:
            print(f"{label}: already applied")
            continue
        if text.count(old) != 1:
            sys.exit(f"{label}: anchor found {text.count(old)} times, expected 1; nothing written")
        text = text.replace(old, new)
        print(f"{label}: applied")
    if text.count("\\PLACEHOLDER") != n_ph:
        sys.exit("PLACEHOLDER count changed; nothing written")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} (PLACEHOLDER tokens: {n_ph}, unchanged)")


if __name__ == "__main__":
    main()
