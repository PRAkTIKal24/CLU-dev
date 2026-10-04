"""w15 spoke/lace-poe-rows -- launcher for the GENERATION TABLE (``exp_gen_cell``) on this box.

⛔ Nothing in ``exp_gen_cell`` / ``exp_gen_store`` is edited by this script.

THE ONE THING IT DOES: the no-extension ruling, applied exactly as the hub that spent
``generation-table`` 1/3 on this box applied it
(``outputs/hub-generation-table/relaunch/scored/run_arm.py`` +
``REENCODE-VERDICT-iter2.md``).  On FORGIS_14 an ENFORCED ``_assert_cache_reencodes``
aborts (51 float16 ULP on sub-2^-4 post-ReLU dust; every element >= 2^-4 moves by
exactly 1.000 ULP).  The gate's own docstring says it binds on a run that EXTENDS the
cache.  So, per seed:

  1. compute every row the cell will gather (store pool ∪ B ∪ MS ∪ fit ∪ val), with the
     harness's OWN row functions;
  2. ``n_shortfall`` = rows not in the frozen cache;
  3. shortfall > 0  -> STOP, nothing encoded, nothing scored;
     shortfall == 0 -> the gate is called with ``will_extend=False`` and a reason that
     names the measured shortfall.  Its statistics are still computed, logged and put in
     the manifest (``would_have_failed`` is recorded), the bounds are untouched, and an
     unenforced pass does NOT register in ``_REENCODE_VERIFIED`` -- so if any uncached row
     is ever gathered after all, ``gather_features`` re-runs the check at full strength
     and aborts.

usage: python scripts/w15_lace_poe_gencell.py --out DIR --seed 0 [--seed 1 ...] --arm A ...
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chlu  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if _ROOT not in Path(chlu.__file__).resolve().parents:
    raise SystemExit(f"chlu resolves to {chlu.__file__}, OUTSIDE {_ROOT} -- refusing.")

from chlu.config import get_default_config  # noqa: E402
from chlu.core.gen_bridge import dataset_module, load_feature_cache  # noqa: E402
from chlu.experiments import exp_gen_cell as C  # noqa: E402
from chlu.experiments import exp_gen_store as S  # noqa: E402


def shortfall(config, seed: int) -> dict:
    cfg = config.gen_cell
    dataset, pretext = str(cfg.dataset), str(cfg.pretext)
    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    rows = S.select_store_rows(split, config, int(seed))
    pool = np.asarray(rows["store_train"], np.int64)
    ms = S.matched_seen_rows(dataset, split, values, rows["store_train"])
    need = np.unique(np.concatenate([pool, np.asarray(rows["eval_b"], np.int64),
                                     np.asarray(ms, np.int64)]))
    bdir = S.bridge_dir(config, dataset, pretext, int(seed))
    crows, _f, cman = load_feature_cache(bdir, verify=False)
    miss = int((~np.isin(need, np.asarray(crows, np.int64))).sum())
    return {"seed": int(seed), "n_needed": int(need.size), "n_shortfall": miss,
            "cache_scope": cman.get("scope"), "cache_n_rows": int(cman["n_rows"]),
            "bridge_dir": str(bdir)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, action="append", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--stage", default="cell", choices=["cell", "aggregate", "all"])
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config = get_default_config()
    # the refusal record of an unrelated experiment gates the other rows of the generation
    # table, not the K* row this launcher adds; it is not part of this package
    config.gen_cell.require_refusal_complete = False
    print(f"chlu.__file__ = {chlu.__file__}", flush=True)
    cov = {}
    if a.stage in ("cell", "all"):
        for s in a.seed:
            cov[s] = shortfall(config, s)
            print(f"seed {s}: cache coverage {cov[s]}", flush=True)
            if cov[s]["n_shortfall"] > 0:
                raise SystemExit(f"STOP: seed {s} needs {cov[s]['n_shortfall']} uncached "
                                 "rows; this run would EXTEND the cache. Nothing encoded.")
        with open(out / "cache_coverage.json", "w", encoding="utf-8") as fh:
            json.dump(cov, fh, indent=1)
        reason = ("w15/lace-poe-rows launcher: n_shortfall == 0 measured for seeds "
                  f"{sorted(cov)} ({ {k: v['n_needed'] for k, v in cov.items()} } rows "
                  "needed, all in the frozen cache) -- this run cannot extend the cache "
                  "(no-extension ruling, hub-generation-table REENCODE-VERDICT-iter2.md)")
        C._assert_cache_reencodes = functools.partial(
            S._assert_cache_reencodes, will_extend=False, reason=reason)
    C.run_experiment_gen_cell(config, quick=False, out_dir=str(out), stage=a.stage,
                              seeds=a.seed, arms=a.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
