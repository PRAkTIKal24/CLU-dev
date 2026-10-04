"""Stamp the fit-population record onto rival checkpoints written before it existed.

The population record must be on the CHECKPOINT, not only in a README, because a
consumer loads a callable and gets ``out_of_range = 0`` on a value this rival was
fitted on - which is correct, and says nothing about whether the comparison is
like-for-like with an arm that held that value out.

Run once after the rivals land. Idempotent: a checkpoint that already carries a
record is left alone unless ``force``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

from . import w4_population as P


def backfill(root: Path, grid, fit_rows_by_seed: dict, seeds=(0, 1, 2),
             force: bool = False, log=print) -> dict:
    """Add ``population`` to every ``*.rival.json`` and ``table.rival.pkl``."""
    out = {}
    for seed in seeds:
        d = Path(root) / f"seed_{seed}"
        if not d.exists():
            continue
        rec = P.fit_population_record(grid, fit_rows_by_seed[seed])
        touched = []

        pkl = d / "table.rival.pkl"
        if pkl.exists():
            with open(pkl, "rb") as fh:
                obj = pickle.load(fh)
            if force or not obj.get("population"):
                obj["population"] = rec
                with open(pkl, "wb") as fh:
                    pickle.dump(obj, fh)
                touched.append(pkl.name)

        for jp in sorted(d.glob("*.rival.json")):
            with open(jp, encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("NOT_AVAILABLE_AS_CALLABLE"):
                continue
            if force or not meta.get("population"):
                meta["population"] = rec
                with open(jp, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2, sort_keys=True)
                    fh.write("\n")
                touched.append(jp.name)

        out[str(seed)] = {
            "n_rows_fit": rec["n_rows_fit"],
            "fit_population": rec["fit_population"],
            "values_never_fitted": rec["values_never_fitted"],
            "checkpoints_stamped": touched,
        }
        log(f"  seed {seed}: stamped {len(touched)} checkpoint(s); fit pool "
            f"{rec['n_rows_fit']} rows, values never fitted: "
            f"{rec['values_never_fitted'] or 'none - every value was seen'}")
    return out
