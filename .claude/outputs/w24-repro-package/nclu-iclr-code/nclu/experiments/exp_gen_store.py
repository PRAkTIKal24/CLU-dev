"""exp gen store."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Callable
import matplotlib

matplotlib.use("Agg")
import numpy as np
from nclu.config import NCLUConfig
from nclu.core.gen_bridge import encode_images, features_for_rows, stream_images
from nclu.experiments.exp_gen_bridge import select_rows


def bridge_root(config: NCLUConfig) -> Path:
    r = config.gen_store.bridge_root or os.environ.get("NCLU_ENCODER_ROOT")
    if r:
        return Path(r).expanduser()
    from nclu.paths import ROOT

    return ROOT / "weights" / "encoder"


def bridge_dir(config: NCLUConfig, dataset: str, pretext: str, seed: int) -> Path:
    d = bridge_root(config) / f"{dataset}_{pretext}" / f"seed_{seed}"
    if not (d / "bridge_manifest.json").exists():
        raise FileNotFoundError(
            f"no frozen bridge at {d} — this study consumes the FROZEN cache and ⛔ never rebuilds it (set gen_store.bridge_root or $NCLU_ENCODER_ROOT)"
        )
    return d


def select_store_rows(split, config: NCLUConfig, seed: int) -> dict:
    bridge_rows = select_rows(split, config.gen_bridge, int(seed))
    pool = np.setdiff1d(np.asarray(split.train_rows, np.int64), bridge_rows["eval_a"])
    rng = np.random.default_rng(int(seed) * 7919 + 20260910)
    n = min(int(config.gen_store.n_store_train), pool.size)
    return {
        "store_train": np.sort(rng.permutation(pool)[:n]),
        "eval_a": bridge_rows["eval_a"],
        "eval_b": bridge_rows["eval_b"],
        "bridge_train": bridge_rows["train"],
    }


def matched_seen_rows(dataset: str, split, values: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    if dataset != "shapes3d":
        raise NotImplementedError(
            f"the matched-seen control is defined by the Shapes3D Montero corner; ⛔ do not invent one for {dataset!r} — return to the study"
        )
    v = np.asarray(values)
    corner = (
        (v[:, 0] >= 0.8) & (v[:, 1] >= 0.8) & (v[:, 2] >= 0.8) & (v[:, 3] >= 1.1) & (v[:, 5] > 20)
    )
    matched = np.where(corner & (v[:, 4] != 1.0))[0]
    matched = np.setdiff1d(matched, np.asarray(exclude, np.int64))
    if np.intersect1d(matched, split.test_rows).size:
        raise AssertionError("the matched-seen control overlaps the held-out split")
    return matched


_REENCODE_VERIFIED: set[tuple] = set()


def _cache_identity(cache_rows, cman: dict) -> tuple:
    """A stable key for "this cache, written by this encoder"."""
    return (
        str(cman.get("encoder_param_hash", "")),
        str(cman.get("dtype", "")),
        int(cman.get("D", 0)),
        int(cman.get("n_rows", -1)),
        int(np.asarray(cache_rows).size),
    )


def gather_features(
    dataset: str,
    config: NCLUConfig,
    encoder,
    cache_rows: np.ndarray,
    features,
    cman: dict,
    rows: np.ndarray,
    log: Callable = print,
) -> np.ndarray:
    rows = np.asarray(rows, np.int64)
    d = int(cman["D"])
    out = np.zeros((rows.size, d), np.float32)
    pos = np.searchsorted(cache_rows, rows)
    safe = np.minimum(pos, cache_rows.size - 1)
    hit = (pos < cache_rows.size) & (cache_rows[safe] == rows)
    if hit.any():
        out[hit] = features_for_rows(cache_rows, features, rows[hit]).reshape((-1, d))
    n_miss = int((~hit).sum())
    if n_miss:
        if _cache_identity(cache_rows, cman) not in _REENCODE_VERIFIED:
            log("    ⭐ cache extension requested — re-encode check runs ENFORCED here")
            _assert_cache_reencodes(
                dataset,
                config,
                encoder,
                cache_rows,
                features,
                cman,
                log=log,
                reason="gather_features is about to EXTEND the cache (point of extension)",
            )
        log(f"    extending cache coverage: {n_miss} row(s) encoded through the frozen encoder")
        missing = rows[~hit]
        slot = np.flatnonzero(~hit)
        chunk = int(config.gen_store.encode_chunk)
        for i in range(0, missing.size, chunk):
            sel = slice(i, min(i + chunk, missing.size))
            imgs = stream_images(
                dataset, missing[sel], config.gen_data.data_root, int(config.gen_bridge.cache_block)
            )
            f = encode_images(encoder, imgs, int(config.gen_bridge.eval_batch))
            out[slot[sel]] = f.astype(np.dtype(cman["dtype"])).astype(np.float32).reshape((-1, d))
            del imgs, f
    return out


def _assert_cache_reencodes(
    dataset,
    config,
    encoder,
    cache_rows,
    features,
    cman,
    n=64,
    log=print,
    *,
    will_extend: bool = True,
    reason: str = "",
):
    ULP_BOUND = 1.0
    FRAC_BOUND = 0.01
    if not will_extend and (not str(reason).strip()):
        raise ValueError(
            "⛔ _assert_cache_reencodes(will_extend=False) requires a `reason`. An unenforced check that leaves no trace is indistinguishable in the log and in the manifest from a check that passed — which is how this comes back as a mystery. Name the evidence that this run cannot extend the cache."
        )
    take = np.asarray(cache_rows[: int(n)], np.int64)
    cached = features_for_rows(cache_rows, features, take).reshape((take.size, -1))
    cache_dtype = np.dtype(cman["dtype"])
    imgs = stream_images(
        dataset, take, config.gen_data.data_root, int(config.gen_bridge.cache_block)
    )
    fresh = (
        encode_images(encoder, imgs, int(config.gen_bridge.eval_batch))
        .astype(cache_dtype)
        .astype(np.float32)
        .reshape((take.size, -1))
    )
    n_nonfinite = int(np.count_nonzero(~(np.isfinite(cached) & np.isfinite(fresh))))
    delta = np.abs(cached.astype(np.float64) - fresh.astype(np.float64))
    ulp = np.spacing(np.abs(cached).astype(cache_dtype)).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ulp_delta = np.where(delta > 0.0, delta / ulp, 0.0)
    n_elem = int(cached.size)
    n_diff = int(np.count_nonzero(delta > 0.0))
    stats = {
        "n_checked": int(take.size),
        "n_elements": n_elem,
        "max_abs_delta": float(np.max(delta)) if n_elem else 0.0,
        "max_ulp_delta": float(np.max(ulp_delta)) if n_elem else 0.0,
        "n_differing": n_diff,
        "frac_differing": n_diff / n_elem if n_elem else 0.0,
        "n_nonfinite": n_nonfinite,
        "cache_dtype": str(cache_dtype),
        "ulp_bound": float(ULP_BOUND),
        "frac_bound": float(FRAC_BOUND),
        "enforced": bool(will_extend),
        "enforced_reason": str(reason),
    }
    log(
        f"  ⭐ cache re-encode check: {stats['n_checked']} rows, {n_diff}/{n_elem} elements differ ({stats['frac_differing']:.3%}), max |delta| = {stats['max_abs_delta']:.6g} = {stats['max_ulp_delta']:.3f} {cache_dtype} ULP (bounds: {ULP_BOUND:g} ULP, {FRAC_BOUND:.1%} of elements)"
    )
    log(
        f"  {('⭐ ENFORCED' if will_extend else '⚠ NOT ENFORCED')} (enforced={str(bool(will_extend)).lower()}): {reason or 'no caller opinion — the strict default applies'}"
    )
    verdicts: list[tuple[str, str]] = []
    if n_nonfinite:
        verdicts.append(
            (
                "nonfinite",
                f"re-encode check: {n_nonfinite} of {n_elem} element(s) are NaN or Inf in the cache or in the re-encoded features, so the comparison is undefined — {stats}",
            )
        )
    if stats["max_ulp_delta"] > ULP_BOUND:
        verdicts.append(
            (
                "ulp",
                f"re-encoding {stats['n_checked']} cached rows did not reproduce the cache to within one {cache_dtype} ULP: max |delta| = {stats['max_abs_delta']:.6g} = {stats['max_ulp_delta']:.3f} ULP of the cache dtype (bound {ULP_BOUND:g}); {n_diff}/{n_elem} elements ({stats['frac_differing']:.3%}) differ.  ⭐ That is further than a rounding step of the stored format, so the loaded encoder does not reproduce these cached values — {stats}",
            )
        )
    if stats["frac_differing"] > FRAC_BOUND:
        verdicts.append(
            (
                "fraction",
                f"re-encoding {stats['n_checked']} cached rows stayed within one {cache_dtype} ULP (max {stats['max_ulp_delta']:.3f} ULP) but {n_diff}/{n_elem} elements ({stats['frac_differing']:.3%}) moved, over the {FRAC_BOUND:.3%} bound: a rounding-boundary crossing touches a handful of elements, a population this size is a different computation — {stats}",
            )
        )
    stats["would_have_failed"] = bool(verdicts)
    stats["would_have_failed_why"] = [k for k, _ in verdicts]
    if will_extend:
        if verdicts:
            raise AssertionError(verdicts[0][1])
        _REENCODE_VERIFIED.add(_cache_identity(cache_rows, cman))
        return stats
    if verdicts:
        log(
            f"  ⛔⛔ THE RE-ENCODE CHECK WOULD HAVE ABORTED THIS RUN ({', '.join((k for k, _ in verdicts))}) and was NOT enforced, because: {reason}. ⭐ The run encodes nothing new, so no inconsistent feature can be written — but the encoder does NOT reproduce this cache on this machine, and any run that extends the cache here WILL abort.\n      {verdicts[0][1]}"
        )
    else:
        log("  ⭐ (the check was not enforced, and it would have PASSED anyway)")
    return stats
