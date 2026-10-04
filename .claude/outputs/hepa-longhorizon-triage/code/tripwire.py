"""Matched-byte tripwire for the HEPA long-horizon triage.

The instrument that decides whether a long-context h-AUROC gain is a gain of
mechanism or a gain of bytes. It ships before the sweep it will judge.

USAGE (this is the exact call signature wave 2 uses at an arbitrary rung)
------------------------------------------------------------------------
    PYTHONPATH=<this dir> <venv>/bin/python tripwire.py \
        --dataset FD001 --context 512 \
        --seeds 42 123 456 \
        --out ../tripwire_smoke.json

Optional: --budgets token_only residual_stream kv_cache   (default: all three)
          --descriptors chan_stats last_patch patch_means raw_window
          --no-diagnostic        (skip the PREREG-non-compliant surface store)

Everything else -- splits, cut points, labels, horizon grid, normalisation,
degenerate-horizon handling -- comes from the HEPA clone and is never
reimplemented here (see hepa_iface.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

import features as F
import hepa_iface as HI
import tw_config as C
from arms import (
    ExemplarStore,
    last_event_age_surface,
    prevalence_surface,
    seasonal_surface,
)
from byte_ledger import ledger


def _stats(vals: List[float], seed_active: bool) -> Dict[str, object]:
    """Seed aggregation that refuses to fabricate spread (task §5)."""
    a = np.asarray(vals, dtype=float)
    if not seed_active:
        return {
            "regime": "deterministic",
            "h_auroc": float(a[0]),
            "n_seeds_run": int(len(a)),
            "note": "store fits the budget without subsampling; the seed is inert, "
                    "so one number is reported rather than a +/-0.00 over three "
                    "identical runs",
        }
    return {
        "regime": "seeded",
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "range": float(a.max() - a.min()),
        "per_seed": [float(v) for v in a],
        "n_seeds_run": int(len(a)),
    }


def run(
    dataset: str,
    context: int,
    seeds: List[int],
    budgets: List[str],
    descriptors: List[str],
    diagnostic: bool = True,
) -> Dict[str, object]:
    t0 = time.time()
    bundle = HI.load_bundle(dataset, context)
    H = bundle.horizons
    y_tr = HI.label_surface(bundle.train.tte, H)
    y_va = HI.label_surface(bundle.val.tte, H)
    y_te = HI.label_surface(bundle.test.tte, H)

    led = ledger(context, bundle.n_channels)

    out: Dict[str, object] = {
        "schema": "hepa-lh-tripwire/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "context_len": context,
        "seeds": list(seeds),
        "provenance": HI.hepa_provenance(),
        "protocol": {
            "source": "hepa/utils/config.py + REPRODUCIBILITY.md (clone-actual), PREREG §1",
            "norm_mode": bundle.norm_mode,
            "K": len(H),
            "horizon_grid": [int(H[0]), int(H[-1])],
            "horizon_conditioning": "scalar-concat: raw dt in native units "
                                    "(PREREG §2.4); no learned horizon embedding",
            "degenerate_horizon_convention": (
                "SHIPPED CODE: a horizon is dropped only if it is EXACTLY degenerate "
                "(n_pos == 0 or n_pos == n_total). The paper and PROTOCOL.md instead "
                "threshold prevalence at 0.001/0.999 (scout D4). We use the code's."
            ),
            "min_context": C.MIN_CONTEXT,
            "stride_train_val": C.STRIDE_TRAIN,
            "stride_test": C.STRIDE_TEST,
            "k_selected_on": "ft_val h-AUROC; test is never used for selection",
            "k_grid": C.K_NEIGHBOURS_GRID,
        },
        "splits": {
            "n_entities": {k: len(v) for k, v in bundle.entity_lengths.items()},
            "n_windows": {
                "train": len(bundle.train), "val": len(bundle.val), "test": len(bundle.test)
            },
            "entity_len_train": {
                "min": int(np.min(bundle.entity_lengths["train"])),
                "median": float(np.median(bundle.entity_lengths["train"])),
                "max": int(np.max(bundle.entity_lengths["train"])),
            },
            "effective_context_test": {
                "mean": float(bundle.test.effective_context.mean()),
                "median": float(np.median(bundle.test.effective_context)),
                "max": int(bundle.test.effective_context.max()),
                "frac_windows_at_full_rung": float(
                    (bundle.test.effective_context >= context).mean()
                ),
                "note": "if this fraction is 0 the rung is not reachable on this "
                        "dataset and any difference vs a shorter rung is noise",
            },
            "n_nondegenerate_horizons_test": HI.n_nondegenerate_horizons(y_te),
            "label_surface_prevalence_test": float(y_te.mean()),
        },
        "byte_ledger": led,
        "trivial_baselines": {},
        "exemplar_store": {},
    }

    # ---- trivial references (all deterministic: no subsample, no ties) ------
    triv: Dict[str, object] = {}
    p = prevalence_surface(y_tr, len(bundle.test))
    triv["prevalence"] = {
        "status": "OK", "regime": "deterministic",
        "h_auroc": HI.score_h_auroc(p, y_te, H),
        "definition": "per-horizon base rate on ft_train, constant in t",
    }

    key = C.dataset_key(dataset)
    period = C.SEASONAL_PERIOD.get(key, None)
    if period is None:
        reason = C.SEASONAL_PERIOD_REASON.get(
            "cmapss" if key.startswith("FD") else key, "no native period defined"
        )
        triv["seasonal"] = {"status": "NOT_APPLICABLE", "reason": reason, "period": None}
    else:
        p = seasonal_surface(
            y_tr, bundle.train.t_index, bundle.test.t_index, period, C.N_SEASONAL_BINS
        )
        triv["seasonal"] = {
            "status": "OK", "regime": "deterministic",
            "h_auroc": HI.score_h_auroc(p, y_te, H),
            "period_steps": int(period), "n_bins": C.N_SEASONAL_BINS,
            "definition": "per-horizon base rate on ft_train conditioned on the "
                          "phase bin of t within one native period",
        }

    p = last_event_age_surface(bundle.test.age, len(H))
    triv["last_event_age"] = {
        "status": "OK", "regime": "deterministic",
        "h_auroc": HI.score_h_auroc(p, y_te, H),
        "definition": "rank by steps since this entity's previous event; if the "
                      "entity has had none, steps since its record began. No model, "
                      "no encoder, no window.",
        "age_test": {
            "min": int(bundle.test.age.min()),
            "median": float(np.median(bundle.test.age)),
            "max": int(bundle.test.age.max()),
        },
    }
    out["trivial_baselines"] = triv

    # ---- matched-byte exemplar store ---------------------------------------
    store_res: Dict[str, object] = {}
    for budget_name in budgets:
        budget = led["budget_bytes"][budget_name]
        per_desc: Dict[str, object] = {}
        for desc in descriptors:
            payloads = ["tte"] + (["surface"] if diagnostic and budget_name == C.PRIMARY_BUDGET else [])
            for payload in payloads:
                d = F.feature_dim(desc, bundle.n_channels, context)
                pf = 1 if payload == "tte" else len(H)
                if budget // ((d + pf) * C.DTYPE_BYTES) < 1:
                    per_desc[f"{desc}::{payload}"] = {
                        "status": "NOT_RUN",
                        "reason": f"budget {budget} B < one exemplar "
                                  f"({(d + pf) * C.DTYPE_BYTES} B: dim {d} + payload {pf})",
                    }
                    continue
                val_h, test_h, ks, infos = [], [], [], []
                for seed in seeds:
                    st = ExemplarStore(
                        descriptor=desc, budget_bytes=budget, context_len=context,
                        norm_mode=bundle.norm_mode, payload=payload, n_horizons=len(H),
                    ).fit(bundle.train.contexts, bundle.train.tte,
                          y_tr if payload == "surface" else None,
                          bundle.n_channels, seed)
                    pv = st.predict_ks(bundle.val.contexts, H, C.K_NEIGHBOURS_GRID)
                    scored = {k: HI.score_h_auroc(v, y_va, H) for k, v in pv.items()}
                    best_k = max(scored, key=lambda k: (scored[k], -k))
                    pt = st.predict(bundle.test.contexts, H, best_k)
                    val_h.append(scored[best_k])
                    test_h.append(HI.score_h_auroc(pt, y_te, H))
                    ks.append(best_k)
                    infos.append(st.info)
                    if not st.info["seed_active"]:
                        break   # deterministic store: running seeds 2 and 3 is theatre
                seed_active = bool(infos[0]["seed_active"])
                per_desc[f"{desc}::{payload}"] = {
                    "status": "OK",
                    "prereg_compliant": payload == "tte",
                    "store": {k: v for k, v in infos[0].items() if k != "seed"},
                    "k_selected_per_seed": [int(k) for k in ks],
                    "val": _stats(val_h, seed_active),
                    "test": _stats(test_h, seed_active),
                }
        store_res[budget_name] = {"budget_bytes": int(budget), "arms": per_desc}
    out["exemplar_store"] = store_res

    # ---- the single number the kill rule consumes --------------------------
    out["tripwire"] = _select_tripwire(out, C.PRIMARY_BUDGET)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    return out


def _score(entry: Dict[str, object], split: str) -> float:
    s = entry[split]
    return float(s["h_auroc"] if s["regime"] == "deterministic" else s["mean"])


def _select_tripwire(out: Dict[str, object], budget_name: str) -> Dict[str, object]:
    """Best PREREG-compliant reference, selected on VAL, reported on TEST.

    The kill rule (PREREG §4b) needs one number: the strongest same-byte or
    no-model reference. Descriptor and k are both chosen on ft_val, so the test
    number is not selected on itself.
    """
    arms = out["exemplar_store"][budget_name]["arms"]
    cands = {
        name: e for name, e in arms.items()
        if e.get("status") == "OK" and e.get("prereg_compliant")
    }
    best_store = max(cands, key=lambda n: _score(cands[n], "val")) if cands else None
    triv = {
        n: e["h_auroc"] for n, e in out["trivial_baselines"].items()
        if e.get("status") == "OK"
    }
    best_triv = max(triv, key=lambda n: triv[n]) if triv else None
    store_test = _score(cands[best_store], "test") if best_store else float("nan")
    triv_test = triv[best_triv] if best_triv else float("nan")
    winner = "exemplar_store" if store_test >= triv_test else "trivial"
    return {
        "budget_used": budget_name,
        "budget_bytes": out["exemplar_store"][budget_name]["budget_bytes"],
        "best_exemplar_arm": best_store,
        "best_exemplar_h_auroc_test": store_test,
        "best_trivial_arm": best_triv,
        "best_trivial_h_auroc_test": triv_test,
        "tripwire_h_auroc": float(max(store_test, triv_test)),
        "tripwire_source": winner,
        "selection_rule": "descriptor and k chosen by ft_val h-AUROC; the reported "
                          "number is ft_test at that choice; PREREG-non-compliant "
                          "diagnostic arms are excluded",
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="FD001")
    ap.add_argument("--context", type=int, default=C.DEFAULT_CONTEXT)
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--budgets", nargs="+", default=C.BUDGET_VARIANTS)
    ap.add_argument("--descriptors", nargs="+", default=C.DESCRIPTORS)
    ap.add_argument("--no-diagnostic", action="store_true")
    ap.add_argument("--out", default=str(C.OUT_DIR / "tripwire_smoke.json"))
    a = ap.parse_args(argv)

    res = run(a.dataset, a.context, a.seeds, a.budgets, a.descriptors,
              diagnostic=not a.no_diagnostic)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    tw = res["tripwire"]
    print(f"[tripwire] {a.dataset} @ context {a.context} -> {a.out}")
    print(f"  budget ({tw['budget_used']}) = {tw['budget_bytes']} B")
    for n, e in res["trivial_baselines"].items():
        print(f"  trivial {n:16s} {e.get('h_auroc', e.get('status'))}")
    print(f"  best exemplar arm  {tw['best_exemplar_arm']} -> {tw['best_exemplar_h_auroc_test']:.4f}")
    print(f"  TRIPWIRE h-AUROC   {tw['tripwire_h_auroc']:.4f}  (from {tw['tripwire_source']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
