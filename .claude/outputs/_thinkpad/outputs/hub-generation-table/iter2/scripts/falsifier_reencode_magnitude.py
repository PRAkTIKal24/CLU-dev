"""§5.1 FALSIFIER — reencode_magnitude.json.

Loads seed 0's frozen cache, takes crows[:64], re-encodes EXACTLY as
`_assert_cache_reencodes` does (same dtype round-trip, same np.spacing ULP at the
cache dtype), and records the MAGNITUDE of every differing element.

⛔ STOP CONDITION for all four hubs: any element with |cached| >= 2^-4 showing
   ulp_delta > 2.
"""
import json, os, sys
from pathlib import Path
import numpy as np

from chlu.config import get_default_config
from chlu.core.gen_bridge import (
    load_feature_cache, features_for_rows, stream_images, encode_images,
)
from chlu.core.gen_bridge import param_hash
from chlu.experiments.exp_gen_store import bridge_dir

DATASET, PRETEXT, SEED = "shapes3d", "recon", 0
N_ROWS = 64
THRESH = 2.0 ** -4  # the magnitude split the falsifier is defined on

def main() -> int:
    cfg = get_default_config()
    bdir = bridge_dir(cfg, DATASET, PRETEXT, SEED)
    print(f"bridge dir: {bdir}", flush=True)

    from chlu.utils.checkpoints import load_model  # noqa
    crows, feats, cman = load_feature_cache(bdir, verify=True)
    print(f"cache: n_rows={cman['n_rows']} scope={cman.get('scope')} D={cman['D']} "
          f"dtype={cman['dtype']} feat_shape={cman['feat_shape']}", flush=True)

    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    encoder = load_model(bdir / "encoder.pkl")
    eh = param_hash(encoder)
    assert eh == bman["encoder_param_hash"], (
        f"encoder param hash mismatch: {eh} != {bman['encoder_param_hash']}")
    print(f"encoder param hash OK: {eh}", flush=True)

    # ---- EXACTLY the arithmetic of _assert_cache_reencodes -------------------
    take = np.asarray(crows[:N_ROWS], np.int64)
    cached = features_for_rows(crows, feats, take).reshape((take.size, -1))
    cache_dtype = np.dtype(cman["dtype"])
    imgs = stream_images(DATASET, take, cfg.gen_data.data_root,
                         int(cfg.gen_bridge.cache_block))
    fresh = (
        encode_images(encoder, imgs, int(cfg.gen_bridge.eval_batch))
        .astype(cache_dtype).astype(np.float32).reshape((take.size, -1))
    )
    delta = np.abs(cached.astype(np.float64) - fresh.astype(np.float64))
    ulp = np.spacing(np.abs(cached).astype(cache_dtype)).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ulp_delta = np.where(delta > 0.0, delta / ulp, 0.0)

    n_elem = int(cached.size)
    diff = delta > 0.0
    n_diff = int(np.count_nonzero(diff))
    absc = np.abs(cached).astype(np.float64)

    # argmax-ULP element
    flat = int(np.argmax(ulp_delta))
    arg = {
        "flat_index": flat,
        "row_in_take": int(flat // cached.shape[1]),
        "dataset_row": int(take[flat // cached.shape[1]]),
        "channel": int(flat % cached.shape[1]),
        "abs_cached": float(absc.flat[flat]),
        "cached": float(cached.astype(np.float64).flat[flat]),
        "fresh": float(fresh.astype(np.float64).flat[flat]),
        "delta": float(delta.flat[flat]),
        "ulp_at_element": float(ulp.flat[flat]),
        "ulp_delta": float(ulp_delta.flat[flat]),
        "abs_cached_ge_2pow_minus4": bool(absc.flat[flat] >= THRESH),
    }

    def band(mask_band):
        m = diff & mask_band
        n = int(np.count_nonzero(m))
        out = {
            "n_elements_in_band": int(np.count_nonzero(mask_band)),
            "n_differing": n,
            "max_ulp_delta": float(ulp_delta[m].max()) if n else 0.0,
            "max_abs_delta": float(delta[m].max()) if n else 0.0,
            "mean_ulp_delta": float(ulp_delta[m].mean()) if n else 0.0,
        }
        if n:
            i = int(np.argmax(np.where(m, ulp_delta, -1.0)))
            out["argmax_abs_cached"] = float(absc.flat[i])
            out["argmax_delta"] = float(delta.flat[i])
        return out

    hi_mask = absc >= THRESH
    lo_mask = ~hi_mask
    hi, lo = band(hi_mask), band(lo_mask)

    violators = diff & hi_mask & (ulp_delta > 2.0)
    n_viol = int(np.count_nonzero(violators))

    # extra context: how much of the diff mass sits where
    payload = {
        "falsifier": "LAUNCH-hub-generation-table-2 §5.1 reencode_magnitude",
        "produced_by": "hub/generation-table-2, Windows AMD64 (FORGIS_14)",
        "dataset": DATASET, "pretext": PRETEXT, "seed": SEED,
        "bridge_dir": str(bdir),
        "cache": {"n_rows": int(cman["n_rows"]), "scope": cman.get("scope"),
                  "D": int(cman["D"]), "dtype": str(cache_dtype),
                  "feat_shape": [int(x) for x in cman["feat_shape"]]},
        "encoder_param_hash": eh,
        "n_rows_checked": int(take.size),
        "n_elements": n_elem,
        "n_differing": n_diff,
        "frac_differing": n_diff / n_elem if n_elem else 0.0,
        "max_abs_delta": float(delta.max()) if n_elem else 0.0,
        "max_ulp_delta": float(ulp_delta.max()) if n_elem else 0.0,
        "threshold_abs_cached": THRESH,
        "argmax_ulp_element": arg,
        "band_ge_2pow_minus4": hi,
        "band_lt_2pow_minus4": lo,
        "STOP_CONDITION": {
            "rule": "any element with |cached| >= 2^-4 and ulp_delta > 2",
            "n_violators": n_viol,
            "TRIGGERED": bool(n_viol > 0),
        },
        "gate_bounds_for_reference": {"ULP_BOUND": 1.0, "FRAC_BOUND": 1e-2},
        "spacing_at_threshold_float16": float(np.spacing(np.float16(THRESH))),
        "float16_subnormal_gap": float(np.spacing(np.float16(0.0))),
    }

    out = Path(os.environ["CHLU_HUB_OUTPUTS"]) / "hub-generation-table" / "reencode_magnitude.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps({k: payload[k] for k in (
        "n_elements", "n_differing", "frac_differing", "max_abs_delta",
        "max_ulp_delta", "argmax_ulp_element", "band_ge_2pow_minus4",
        "band_lt_2pow_minus4", "STOP_CONDITION")}, indent=2), flush=True)
    print(f"\nWROTE {out}", flush=True)
    print("VERDICT: " + ("STOP - ALL FOUR HUBS HALT SCORED WORK" if n_viol
                         else "CLEAR - divergence is sub-2^-4 dust, scored work may proceed"))
    return 0

if __name__ == "__main__":
    sys.exit(main())
