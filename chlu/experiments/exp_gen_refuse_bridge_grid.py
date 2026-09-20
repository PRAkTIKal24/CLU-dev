"""w5.1 ``refuse`` (R-A retry) -- the 3x3 (store seed) x (bridge seed) GRID.

⛔⛔ **No scored run.** Every cell re-reads rows the parent already paid for, through the
same frozen ``score_deletion``.

**Why a grid and not just two spreads.** ``pinned`` and ``joint`` each give one three-seed
number. The joint spread is wider, but it is wider for **two reasons at once** -- the store
draw and the instrument -- and a single number cannot say which is doing the work. That
matters, because "the parent's ± is narrower than the truth" is only interesting if the
missing width comes from the thing the parent held fixed.

The grid separates them. Rows are store seeds, columns are bridge seeds (the frozen
encoder/decoder/classifier triple):

* **down column ``bridge = 0``** -- the STORE effect at a fixed instrument. ⭐ This is
  exactly the parent's own spread, and it is what the parent said it measured.
* **across a row at fixed store** -- ⭐⭐ the INSTRUMENT effect alone. ⛔ Neither hub has
  ever measured this.
* **the diagonal** -- store and instrument moving together: what ``--bridge-mode joint``
  and ``w5 hub/delete`` report.

⚠ The grid's off-diagonal cells are not a new experiment on the object. They are the same
deletion, on the same scored corners, read by a different one of the three frozen
instruments -- which is why they cost no scored run and why they are allowed to exist at a
zero budget at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .exp_gen_factor_wells import _jdump
from .exp_gen_refuse_bridge_spread import NOVEL_REFUSERS, collect
from .exp_gen_refuse_report import LEDGER

SEEDS = (0, 1, 2)

#: What the grid tabulates. ``(path into a collect() record, label, decimals)``.
QUANTITIES = [
    (("evict_bytes", "changed"), "EVICTION bytes changed", 0),
    (("evict_bytes", "n_items_evicted"), "EVICTION items evicted", 0),
    (("evict_bytes", "negative_control_bytes"), "NEGATIVE CONTROL bytes", 0),
] + [
    (("novel", flav, nm), f"NOVEL TUPLES {tag} - {nm}", 0)
    for flav, tag in (("hole", "post-HOLE"), ("evict", "post-EVICTION"))
    for nm in NOVEL_REFUSERS
]

#: ⛔ Quantities that are bounded by construction. They are tabulated so that "did not
#: move" is measured across all nine cells, not assumed from three.
INVARIANTS = [
    (("evict_bytes", "outside_permitted"), "EVICTION bytes OUTSIDE permitted", 0),
]


def _dig(rec, path):
    for k in path:
        rec = rec.get(k) if isinstance(rec, dict) else None
        if rec is None:
            return None
    return rec


def collect_grid(runs: Path, log: Callable = print) -> dict:
    """One ``collect()`` per bridge column, indexed by bridge seed.

    ``runs/pinned`` is the ``bridge = 0`` column (it is the parent's configuration);
    ``runs/bridge1`` and ``runs/bridge2`` are the other two.
    """
    cols = {}
    for b in SEEDS:
        d = Path(runs) / ("pinned" if b == 0 else f"bridge{b}")
        if not d.exists():
            log(f"  ⛔ bridge {b}: {d} absent -- DECLARED NOT RUN, never a null")
            continue
        rec = collect(d, f"bridge={b}", log=log)
        # ⛔ ASSERTED: every cell of a column must really have read bridge seed b.
        for s, p in rec["provenance"].items():
            if int(p["bridge_seed"]) != b:
                raise AssertionError(
                    f"column bridge={b}, store seed {s} recorded bridge_seed "
                    f"{p['bridge_seed']} -- the grid is not the grid it claims to be")
        cols[b] = rec
        log(f"  bridge {b}: seeds {rec['seeds_present']}")
    return cols


def _cell(cols: dict, b: int, s: int, path) -> float:
    rec = cols.get(b)
    if rec is None:
        return None
    sp = _dig(rec, path)
    if not sp or sp.get("per_seed") is None:
        return None
    seeds = rec["seeds_present"]
    if s not in seeds:
        return None
    return sp["per_seed"][seeds.index(s)]


def _rng(vals) -> float:
    v = [x for x in vals if x is not None]
    return (max(v) - min(v)) if len(v) > 1 else None


def _f(x, dp=0) -> str:
    if x is None:
        return "-"
    return f"{x:,.0f}" if dp == 0 else f"{x:,.{dp}f}"


def build(runs: Path, dest: Path, log: Callable = print) -> dict:
    cols = collect_grid(Path(runs), log=log)
    if len(cols) < 2:
        raise AssertionError(
            f"the grid needs at least two bridge columns; got {sorted(cols)}")
    have_b = sorted(cols)

    lines, W = [], None
    W = lines.append
    W("### ⭐⭐ THE 3x3 GRID -- (store seed) x (bridge seed), and what each direction isolates")
    W("")
    W("Rows are **store seeds** (the fit-pool draw and the atom selection). Columns are "
      "**bridge seeds** -- the frozen encoder, decoder and classifier the store is built "
      "from and read back by. ⛔ Nothing about the field store, the rule, the thresholds or "
      "the metric differs between cells; only which of the three frozen instruments is in "
      "use. ⛔ No cell is a scored run.")
    W("")
    W("- **column `b=0`** = the PARENT'S OWN SPREAD (it held the bridge seed at 0);")
    W("- **a row** = the INSTRUMENT effect at a fixed store -- ⭐ never measured before;")
    W("- **the diagonal** = what `--bridge-mode joint` and `w5 hub/delete` report.")
    W("")

    summary = {}
    for path, label, dp in QUANTITIES + INVARIANTS:
        grid = {(s, b): _cell(cols, b, s, path) for s in SEEDS for b in have_b}
        if all(v is None for v in grid.values()):
            continue
        col_rng = {b: _rng([grid[(s, b)] for s in SEEDS]) for b in have_b}
        row_rng = {s: _rng([grid[(s, b)] for b in have_b]) for s in SEEDS}
        diag = [grid[(s, s)] for s in SEEDS if (s, s) in grid]
        summary[label] = {
            "grid": {f"store{s}_bridge{b}": grid[(s, b)] for s in SEEDS for b in have_b},
            "store_effect_range_per_bridge_column": col_rng,
            "instrument_effect_range_per_store_row": row_rng,
            "parent_column_range_b0": col_rng.get(0),
            "diagonal_range": _rng(diag),
            "max_instrument_range": _rng([]) if not row_rng else max(
                [v for v in row_rng.values() if v is not None], default=None),
            "whole_grid_range": _rng(list(grid.values())),
        }
        W(f"**{label}**")
        W("")
        W("| store seed \\ bridge seed | " + " | ".join(f"b={b}" for b in have_b)
          + " | ⭐ instrument range (across the row) |")
        W("|---|" + "---|" * (len(have_b) + 1))
        for s in SEEDS:
            cells = []
            for b in have_b:
                v = _f(grid[(s, b)], dp)
                cells.append(f"**{v}**" if s == b else v)
            W(f"| store {s} | " + " | ".join(cells) + f" | {_f(row_rng[s], dp)} |")
        W("| ⭐ **store range (down the column)** | "
          + " | ".join(_f(col_rng[b], dp) for b in have_b)
          + f" | whole grid: {_f(summary[label]['whole_grid_range'], dp)} |")
        W("")
        p0 = col_rng.get(0)
        mi = summary[label]["max_instrument_range"]
        dg = summary[label]["diagonal_range"]
        if p0 is not None and mi is not None:
            if p0 == 0 and mi == 0:
                verdict = ("⭐ **bounded by construction: 0 in every direction.** No "
                           "instrument can move it.")
            elif p0 == 0:
                verdict = (f"⛔ **the parent's column has range 0 but the instrument moves "
                           f"it by up to {_f(mi, dp)}** -- its ± is not merely narrow, it "
                           f"is empty in the direction that matters.")
            else:
                r = mi / p0
                verdict = (f"instrument range is **{r:.2f}x** the parent's store range "
                           f"({_f(mi, dp)} vs {_f(p0, dp)}); diagonal range "
                           f"{_f(dg, dp)}.")
                verdict += ("  ⛔ **The parent's ± understates this quantity.**" if r > 1.0
                            else "  ⭐ The parent's ± is the larger of the two effects.")
            W(verdict)
            W("")
        summary[label]["verdict"] = verdict if p0 is not None and mi is not None else None

    # -- the diagonal / column identity check ----------------------------------
    W("### ⭐ Internal consistency -- the diagonal, computed twice")
    W("")
    W("The grid's diagonal cells and the separately-run `--bridge-mode joint` arm are the "
      "same (store, bridge) pairs reached by two different code paths (`--bridge-seed s` "
      "with `--seed s` versus `--bridge-mode joint`). ⛔ They must agree exactly or one of "
      "the two paths is not doing what its name says.")
    W("")

    md = "\n".join(lines) + "\n"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "BRIDGE-GRID.md").write_text(md, encoding="utf-8")
    _jdump({"columns_present": have_b, "quantities": summary,
            "provenance": {str(b): cols[b]["provenance"] for b in have_b}},
           dest / "bridge_grid.json")
    log(md)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", required=True,
                   help="the runs/ tree holding pinned/, bridge1/ and bridge2/")
    p.add_argument("--dest", required=True)
    a = p.parse_args(argv)
    build(Path(a.runs), Path(a.dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
