"""exp gen dream."""

from __future__ import annotations
from typing import Callable
import matplotlib

matplotlib.use("Agg")
import numpy as np
from nclu.config import NCLUConfig
from nclu.core.gen_store import channel_correlation, dead_channels, grouping_labels


def resolve_grouping(
    plan, X_store, config: NCLUConfig, seed: int, encoder_hash: str, log: Callable = print
) -> tuple:
    cfg = config.gen_dream
    K = int(plan["K"])
    if str(cfg.grouping_source) == "plan":
        labels = np.asarray(plan["labels"], np.int64)
        return (
            labels,
            {
                "grouping_source": "plan (VERBATIM)",
                "well_sizes": [int(x) for x in np.bincount(labels, minlength=K)],
                "warning": "⛔ the plan's labels index the k_choice seed's encoder channels; on any other seed this is a size-matched RANDOM partition",
            },
        )
    if str(cfg.grouping_source) != "per_seed":
        raise ValueError(
            f"grouping_source must be 'per_seed' or 'plan', got {cfg.grouping_source!r}"
        )
    gs = config.gen_store
    Xc = np.asarray(X_store, np.float32)[: int(gs.n_corr)]
    dead = dead_channels(Xc)
    corr = channel_correlation(Xc)
    labels, info = grouping_labels(
        corr,
        K,
        "correlation",
        int(seed),
        gs.grouping_method,
        dead=dead,
        min_well_frac=float(gs.min_well_frac),
    )
    out = {
        "grouping_source": "per_seed (re-derived at the plan's K)",
        "n_corr_rows": int(Xc.shape[0]),
        "n_dead_channels": int(dead.sum()),
        "grouping_info": info,
        "well_sizes": [int(x) for x in np.bincount(labels, minlength=K)],
        "plan_encoder_param_hash": plan.get("encoder_param_hash"),
        "this_encoder_param_hash": str(encoder_hash),
    }
    if str(encoder_hash) == str(plan.get("encoder_param_hash")):
        same = bool(np.array_equal(labels, np.asarray(plan["labels"], np.int64)))
        out["reproduces_the_plan_bit_identically"] = same
        if not same:
            raise AssertionError(
                "the per-seed grouping does NOT reproduce store_plan.json's labels on the seed whose encoder wrote it — ⛔ this is not the same procedure and the ladder comparison would be void"
            )
        log("  ⭐ grouping re-derivation reproduces the plan BIT-IDENTICALLY on its own seed")
    else:
        out["reproduces_the_plan_bit_identically"] = None
        log(
            f"  ⚠ this seed's encoder differs from the plan's ({str(encoder_hash)[:12]}… vs {str(plan.get('encoder_param_hash'))[:12]}…), so the grouping is re-derived on its own channels — ⛔ the plan's label vector would be a size-matched RANDOM partition here"
        )
    return (labels, out)
