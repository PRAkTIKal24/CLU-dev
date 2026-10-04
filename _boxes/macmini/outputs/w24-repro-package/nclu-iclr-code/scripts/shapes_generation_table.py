"""shapes generation table."""

from __future__ import annotations
import argparse
import functools
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nclu

_ROOT = Path(__file__).resolve().parents[1]
if _ROOT not in Path(nclu.__file__).resolve().parents:
    raise SystemExit(f"nclu resolves to {nclu.__file__}, OUTSIDE {_ROOT} -- refusing.")
from nclu.config import get_default_config
from nclu.core.gen_bridge import dataset_module, load_feature_cache
from nclu.experiments import exp_gen_cell as C
from nclu.experiments import exp_gen_store as S


def shortfall(config, seed: int) -> dict:
    cfg = config.gen_cell
    dataset, pretext = (str(cfg.dataset), str(cfg.pretext))
    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    rows = S.select_store_rows(split, config, int(seed))
    pool = np.asarray(rows["store_train"], np.int64)
    ms = S.matched_seen_rows(dataset, split, values, rows["store_train"])
    need = np.unique(
        np.concatenate([pool, np.asarray(rows["eval_b"], np.int64), np.asarray(ms, np.int64)])
    )
    bdir = S.bridge_dir(config, dataset, pretext, int(seed))
    crows, _f, cman = load_feature_cache(bdir, verify=False)
    miss = int((~np.isin(need, np.asarray(crows, np.int64))).sum())
    return {
        "seed": int(seed),
        "n_needed": int(need.size),
        "n_shortfall": miss,
        "cache_scope": cman.get("scope"),
        "cache_n_rows": int(cman["n_rows"]),
        "bridge_dir": str(bdir),
    }


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
    config.gen_cell.require_refusal_complete = False
    print(f"nclu.__file__ = {nclu.__file__}", flush=True)
    cov = {}
    if a.stage in ("cell", "all"):
        for s in a.seed:
            cov[s] = shortfall(config, s)
            print(f"seed {s}: cache coverage {cov[s]}", flush=True)
            if cov[s]["n_shortfall"] > 0:
                raise SystemExit(
                    f"STOP: seed {s} needs {cov[s]['n_shortfall']} uncached rows; this run would EXTEND the cache. Nothing encoded."
                )
        with open(out / "cache_coverage.json", "w", encoding="utf-8") as fh:
            json.dump(cov, fh, indent=1)
        reason = f"lace-poe-rows launcher: n_shortfall == 0 measured for seeds {sorted(cov)} ({ {k: v['n_needed'] for k, v in cov.items()} } rows needed, all in the frozen cache) -- this run cannot extend the cache (no-extension, study-generation-table REENCODE-VERDICT-.md)"
        C._assert_cache_reencodes = functools.partial(
            S._assert_cache_reencodes, will_extend=False, reason=reason
        )
    C.run_experiment_gen_cell(
        config, quick=False, out_dir=str(out), stage=a.stage, seeds=a.seed, arms=a.arm
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
