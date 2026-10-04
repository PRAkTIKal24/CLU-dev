"""w12 -- THE FOUR-PART LEDGER ACROSS K, ARITHMETIC ONLY.

Everything here is exact arithmetic on the partition shape, so it exists for EVERY K in
the sweep whether or not that K's composition row finished.  It is written separately
from the sweep for exactly that reason: the byte half of the crossover figure is not
hostage to the wall clock.

The three parts that are pure arithmetic:
  1. atoms                 A x D float32, A = min(64, min_k d_k - 1)
  2. covariance + count    packed float64 triangle per well + first moment + count,
                           plus the cardinality-dependent count table
  4. coupling / hole table cardinality-dependent, K-independent
Part 3, the ADDRESS BOOK, is measured -- it depends on the fitted TensorAddress -- and is
filled in from the sweep rows where they exist and marked DECLARED NOT-RUN where they do
not.  It is never estimated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from w12_crossover_lib import (  # noqa: E402
    COUNTS_BYTES, COUPLING_BYTES, D, REFERENCE_PIN, atoms_for,
)

OUT = Path("C:/chlu/outputs/w12-single-design-crossover")
SWEEP = (5, 8, 10, 12, 16, 20, 24, 28, 32)


def main() -> int:
    rows = []
    hdr = (f"{'K':>4} {'d_min':>6} {'d_max':>6} {'A':>4} {'rank/width':>11} "
           f"{'atoms B':>10} {'cov+count B':>12} {'coupling B':>11} "
           f"{'addr book B':>13} {'TOTAL B':>12} {'x524288':>8} {'1x params':>11}")
    print(hdr)
    print("-" * len(hdr))
    for K in SWEEP:
        d0, rem = divmod(D, K)
        ds = [d0 + 1] * rem + [d0] * (K - rem)
        A = atoms_for(ds)
        tri = sum(d * (d + 1) // 2 for d in ds)
        cov = (tri + D + K) * 8
        atoms = A * D * 4
        p = OUT / "val_sweep" / f"val_K{K}_seed0" / "row.json"
        if p.exists():
            addr = int(json.loads(p.read_text(encoding="utf-8"))["address_bytes"])
            addr_s = f"{addr:,}"
        else:
            addr = None
            addr_s = "NOT-RUN"
        tot = atoms + cov + COUNTS_BYTES + COUPLING_BYTES + (addr or 0)
        rows.append({
            "K": K, "well_width_min": min(ds), "well_width_max": max(ds),
            "A_requested": A, "rank_over_width_max": round(A / min(ds), 4),
            "p1_atom_bytes": atoms,
            "p2_covariance_plus_count_bytes": cov + COUNTS_BYTES,
            "p3_address_book_bytes": addr,
            "p3_status": ("measured" if addr is not None else
                          "DECLARED NOT-RUN -- no sweep row for this K on seed 0"),
            "p4_coupling_hole_table_bytes": COUPLING_BYTES,
            "total_bytes_parts_1_2_4_only": atoms + cov + COUNTS_BYTES + COUPLING_BYTES,
            "honest_total_bytes": tot if addr is not None else None,
            "times_reference_pin_parts_1_2_4": round(
                (atoms + cov + COUNTS_BYTES + COUPLING_BYTES) / REFERENCE_PIN, 4),
            "baseline_match_1x_params_float32": (tot // 4) if addr is not None else None,
            "netted": False,
        })
        print(f"{K:>4} {min(ds):>6} {max(ds):>6} {A:>4} {A / min(ds):>11.4f} "
              f"{atoms:>10,} {cov + COUNTS_BYTES:>12,} {COUPLING_BYTES:>11,} "
              f"{addr_s:>13} "
              f"{(f'{tot:,}' if addr is not None else 'partial'):>12} "
              f"{(atoms + cov + COUNTS_BYTES + COUPLING_BYTES) / REFERENCE_PIN:>8.3f} "
              f"{(f'{tot // 4:,}' if addr is not None else '--'):>11}")
    doc = {
        "SMOKE_NOT_A_CLAIM": True,
        "what": ("the four-part ledger across the swept K. Parts 1, 2 and 4 are exact "
                 "arithmetic on the partition shape; part 3 is measured from the fitted "
                 "address book and is a DECLARED NOT-RUN where that K has no sweep row."),
        "reference_pin_bytes": REFERENCE_PIN,
        "reference_pin_note": ("the 524,288 B figure is a REFERENCE COLUMN this wave, "
                               "not a budget. The object is 1x its own total."),
        "never_netted": True,
        "rows": rows,
    }
    p = OUT / "LEDGER-ACROSS-K.json"
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
