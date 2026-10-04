"""w6 hub/retention -- THE COST LINE, and the charter section C.4 check.

Separate from ``exp_w6_retention`` on purpose: the scored seeds are run in resumable chunks
that RE-IMPORT the experiment module on every chunk, so editing that module while a seed is
in flight would change the object mid-run.  This script only READS finished artifacts.

Churn is a COST LINE only, one definition, never a claim.  The definition is fixed in
``PREREG-w6-retention-1.md`` section 6 and is restated in the output:

    churn = (bytes of ``atom_life`` written per decay step) x (steps to the refusal horizon)

i.e. what the lifetime channel costs to RUN, against eviction's single write.

Runnable::

    python scripts/w6_retention_costline.py --out C:/chlu/outputs/w6-retention/run1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chlu.core import gen_retention as LAW
from chlu.experiments import exp_gen_delete_row7 as R7
from chlu.experiments import exp_w6_retention as W6


def cost_line(out: Path, seeds=(0, 1, 2), log=print) -> dict:
    out = Path(out)
    sched = W6.horizon_schedule()
    res = {"churn_definition": (
        "churn = (bytes of atom_life written per decay step) x (steps to the refusal "
        "horizon). ONE definition, fixed in PREREG section 6. It is a COST LINE and is "
        "claimed nowhere."),
        "schedule": {k: sched[k] for k in ("mu2", "rho", "n_half", "n_refuse_steps",
                                           "n_refuse_half_lives", "band", "eps", "gamma")},
        "per_seed": {}}
    for seed in seeds:
        d = out / f"seed_{int(seed)}"
        S = {}
        base = R7.build_field_base(out, seed, log=(lambda *_: None))
        f0 = base["field"]
        from chlu.core.gen_field import _items_holding
        ids, prov, geom = _items_holding(f0, int(W6.PINS["delete_factor"]),
                                         int(W6.PINS["delete_value"]))
        n_items = int(ids.size)
        life_bytes = sum(e - s for s, e in
                         [f0.lifetime_byte_range(int(i)) for i in ids])
        item_bytes = sum(e - s for s, e in
                         [r for i in ids for r in f0.item_byte_ranges(int(i))])
        n_ref = int(sched["n_refuse_steps"])
        S["deletion_target"] = {
            "factor": int(W6.PINS["delete_factor"]), "value": int(W6.PINS["delete_value"]),
            "name": "object_hue = 0.0 (red)",
            "n_items": n_items, "n_by_provenance": int(prov.size),
            "n_by_geometry": int(geom.size)}
        S["churn"] = {
            "atom_life_bytes_written_per_step": int(life_bytes),
            "steps_to_refusal_horizon": n_ref,
            "total_lifetime_bytes_written": int(life_bytes) * n_ref,
            "instant_eviction_bytes_written_once": int(item_bytes),
            "ratio_decay_writes_over_eviction_writes":
                (float(life_bytes * n_ref) / float(item_bytes)) if item_bytes else None,
            "total_image_bytes": int(np.asarray(f0.buf).size)}
        # -- the store's own cost, and the rivals' -----------------------------
        p = d / "scores.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            tc = z["table_cost"].item() if "table_cost" in z.files else None
            S["table_cost"] = tc
        # -- retrain-from-scratch, the GOLD standard, with its cost line -------
        t0 = time.time()
        try:
            S["learner_cost"] = R7.learner_cost(out, seed, log=(lambda *_: None))
        except Exception as e:                      # pragma: no cover - reported, not hidden
            S["learner_cost"] = {"NOT_RUN": f"{type(e).__name__}: {e}"}
        S["learner_cost_wall_seconds"] = time.time() - t0
        # -- charter section C.4: is the settled state sitting on its nearest atom? --
        try:
            S["nearest_atom_check"] = R7.nearest_atom_check(out, seed, log=(lambda *_: None))
        except Exception as e:                      # pragma: no cover
            S["nearest_atom_check"] = {"NOT_RUN": f"{type(e).__name__}: {e}"}
        # -- the BAR 4 rebuild cost, read back from the commutation artifact ----
        cp = d / "commutation.json"
        if cp.exists():
            with open(cp, encoding="utf-8") as fh:
                c = json.load(fh)
            S["never_inserted_rebuild"] = {
                "seconds": c["B_never_inserted"]["rebuild_seconds"],
                "rows_removed": c["B_never_inserted"]["n_rows_removed"],
                "rows_refit": c["B_never_inserted"]["n_pool_rows_after"],
                "bytes_differing": c["B_never_inserted"]["bytes_differing_from_evicted_store"]}
        res["per_seed"][str(seed)] = S
        log(f"  cost(seed {seed}): churn = {life_bytes} B/step x {n_ref} steps = "
            f"{life_bytes * n_ref:,} B written, against eviction's single "
            f"{item_bytes:,} B write "
            f"({life_bytes * n_ref / max(item_bytes, 1):.4f}x)")
    with open(out / "costline.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: (
            float(o) if isinstance(o, (np.floating, np.integer)) else
            (o.tolist() if isinstance(o, np.ndarray) else str(o))))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    cost_line(Path(a.out), tuple(int(s) for s in a.seeds.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
