"""cache build."""

from __future__ import annotations
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Callable, Optional
import numpy as np
from nclu.core.gen_bridge import (
    dataset_module,
    encode_images,
    encode_precision_record,
    param_hash,
    stream_images,
)
from nclu.data.gen_splits import file_sha256

GEN2_SCHEMA = "nclu-gen-feature-cache/2"
GENERATION = 2


def device_record() -> dict:
    """Everything about *where* these bytes were computed, for the manifest."""
    import jax

    devs = jax.devices()
    return {
        "backend": jax.default_backend(),
        "n_devices": int(len(devs)),
        "devices": [f"{d.platform}:{d.id}" for d in devs],
        "device_kinds": sorted({str(d.device_kind) for d in devs}),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "host": platform.node(),
        "platform": platform.platform(),
        "precision": encode_precision_record(),
    }


def device_guard(log: Callable = print) -> dict:
    """Refuse to produce generation 2 on anything but a single CPU device.

    Raises:
        AssertionError: if JAX sees more than one device, or a non-CPU one.
    """
    rec = device_record()
    if rec["n_devices"] != 1:
        raise AssertionError(
            f"generation 2 must be built on ONE device; JAX sees {rec['n_devices']}: {rec['devices']}. A cache split across devices is silently void (CPU/GPU features differ by ~53 ULP on ~1 pct of elements)."
        )
    if rec["backend"] != "cpu" or not rec["devices"][0].startswith("cpu"):
        raise AssertionError(
            f"generation 2 must be built on the local CPU; JAX backend is {rec['backend']!r} on {rec['devices']}. Set JAX_PLATFORMS=cpu."
        )
    log(
        f"  device guard PASSES: one device, {rec['devices'][0]} ({rec['device_kinds']}), precision={rec['precision']['resolved']}, jax {rec['jax_version']}"
    )
    return rec


def encode_shard(
    encoder,
    dataset: str,
    rows: np.ndarray,
    root,
    dtype,
    eval_batch: int,
    block: int,
    handle=None,
    log: Optional[Callable] = None,
) -> np.ndarray:
    """Encode ``rows`` and quantise to the cache dtype - the exact cache bytes.

    Quantisation happens HERE, before hashing, so the determinism check hashes
    what actually lands in ``features.npy`` rather than a float32 intermediate
    the cache never stores.
    """
    imgs = stream_images(dataset, np.asarray(rows, np.int64), root, block=block, handle=handle)
    feats = encode_images(encoder, imgs, int(eval_batch), log=log)
    return np.asarray(feats).astype(np.dtype(dtype))


def shard_sha256(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def determinism_check(
    encoder,
    dataset: str,
    rows: np.ndarray,
    root,
    dtype,
    eval_batch: int,
    block: int,
    log: Callable = print,
) -> dict:
    rows = np.asarray(rows, np.int64)
    t0 = time.time()
    log(f"  determinism check: encoding {rows.size} rows, pass 1/2 ...")
    a = encode_shard(encoder, dataset, rows, root, dtype, eval_batch, block, log=None)
    ha = shard_sha256(a)
    log(f"     pass 1 sha256={ha}")
    log(f"  determinism check: encoding the SAME {rows.size} rows, pass 2/2 ...")
    b = encode_shard(encoder, dataset, rows, root, dtype, eval_batch, block, log=None)
    hb = shard_sha256(b)
    log(f"     pass 2 sha256={hb}")
    ok = bool(ha == hb)
    out = {
        "check": "re-encode one shard twice, sha256 of the quantised cache bytes",
        "n_rows": int(rows.size),
        "row_first": int(rows[0]),
        "row_last": int(rows[-1]),
        "dtype": str(np.dtype(dtype)),
        "sha256_pass1": ha,
        "sha256_pass2": hb,
        "passed": ok,
        "wall_clock_s": float(time.time() - t0),
        "device": device_record(),
    }
    if not ok:
        af, bf = (np.asarray(a, np.float64), np.asarray(b, np.float64))
        diff = af != bf
        out["n_elements"] = int(af.size)
        out["n_differing"] = int(diff.sum())
        out["frac_differing"] = float(diff.mean())
        out["max_abs_diff"] = float(np.abs(af - bf).max())
        out["VERDICT"] = (
            "GENERATION 2 IS VOID - the encode path is not reproducible on this box; stop and report"
        )
        log(
            f"  DETERMINISM CHECK FAILED: {out['n_differing']}/{out['n_elements']} elements differ ({out['frac_differing']:.3%}), max abs diff {out['max_abs_diff']:.6g}. Generation 2 is void."
        )
    else:
        out["VERDICT"] = "reproducible: the same rows encode to the same bytes"
        log(f"  determinism check PASSES ({out['wall_clock_s']:.1f}s): identical sha256")
    return out


def _progress_path(out_dir: Path) -> Path:
    return out_dir / "_progress.json"


def _load_progress(out_dir: Path, expect: dict) -> dict:
    """Resume state, but only if it describes the *same* build."""
    p = _progress_path(out_dir)
    if not p.exists():
        return {"blocks": {}, **expect}
    with open(p) as fh:
        prog = json.load(fh)
    for k, v in expect.items():
        if prog.get(k) != v:
            raise AssertionError(
                f"refusing to resume {out_dir}: progress {k}={prog.get(k)!r} but this build has {k}={v!r}. Delete the partial cache and start it again."
            )
    return prog


def build_gen2_cache(
    encoder,
    dataset: str,
    rows: np.ndarray,
    out_dir: str | Path,
    root,
    dtype: str = "float16",
    eval_batch: int = 256,
    block: int = 8192,
    seed: int = 0,
    source_manifest: Optional[dict] = None,
    max_shard_seconds: Optional[float] = None,
    log: Callable = print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device_guard(log)
    rows = np.asarray(rows, np.int64)
    order = np.argsort(rows, kind="stable")
    rows_sorted = rows[order]
    dt = np.dtype(dtype)
    feat_shape = tuple((int(x) for x in encoder.feat_shape))
    enc_hash = param_hash(encoder)
    expect = {
        "schema": GEN2_SCHEMA,
        "generation": GENERATION,
        "dataset": dataset,
        "seed": int(seed),
        "scope": "full",
        "n_rows": int(rows_sorted.size),
        "dtype": str(dt),
        "feat_shape": list(feat_shape),
        "encoder_param_hash": enc_hash,
        "block": int(block),
    }
    prog = _load_progress(out_dir, expect)
    feat_path = out_dir / "features.npy"
    rows_path = out_dir / "feature_rows.npy"
    if feat_path.exists() and prog["blocks"]:
        arr = np.lib.format.open_memmap(feat_path, mode="r+")
        if arr.shape != (rows_sorted.size,) + feat_shape or arr.dtype != dt:
            raise AssertionError(
                f"{feat_path} is {arr.shape}/{arr.dtype}, this build wants {(rows_sorted.size,) + feat_shape}/{dt}"
            )
    else:
        arr = np.lib.format.open_memmap(
            feat_path, mode="w+", dtype=dt, shape=(rows_sorted.size,) + feat_shape
        )
        prog["blocks"] = {}
    n_blocks = int(np.ceil(rows_sorted.size / block))
    done0 = len(prog["blocks"])
    log(
        f"  generation 2 / seed {seed}: {rows_sorted.size} rows, {n_blocks} blocks of {block}, dtype={dt} - {done0} block(s) already done"
    )
    mod = dataset_module(dataset)
    handle = mod.open_images(root)
    t0 = time.time()
    complete = True
    try:
        for bi in range(n_blocks):
            if str(bi) in prog["blocks"]:
                continue
            if max_shard_seconds is not None and time.time() - t0 > max_shard_seconds:
                log(
                    f"  wall-clock cap {max_shard_seconds:.0f}s reached at block {bi}/{n_blocks}; progress flushed, relaunch to resume"
                )
                complete = False
                break
            lo, hi = (bi * block, min((bi + 1) * block, rows_sorted.size))
            chunk = rows_sorted[lo:hi]
            feats = encode_shard(
                encoder, dataset, chunk, root, dt, eval_batch, block, handle=handle, log=None
            )
            arr[lo:hi] = feats
            prog["blocks"][str(bi)] = {
                "lo": int(lo),
                "hi": int(hi),
                "n": int(hi - lo),
                "sha256": shard_sha256(feats),
            }
            arr.flush()
            with open(_progress_path(out_dir), "w") as fh:
                json.dump(prog, fh, sort_keys=True)
            el = time.time() - t0
            nd = len(prog["blocks"])
            log(
                f"    block {bi + 1}/{n_blocks} rows {lo}:{hi}  [{nd}/{n_blocks} done, {el / 60:.1f} min elapsed]"
            )
    finally:
        handle.close()
        arr.flush()
        del arr
    if not complete:
        return {
            "complete": False,
            "blocks_done": len(prog["blocks"]),
            "n_blocks": n_blocks,
            "out_dir": str(out_dir),
        }
    np.save(rows_path, rows_sorted)
    manifest = {
        "schema": GEN2_SCHEMA,
        "generation": GENERATION,
        "generation_note": "GENERATION 2. Never compare a number computed on this cache with a number computed on a generation-1 cache, in either direction. Generation 1 remains on disk, untouched and still hashed.",
        "dataset": dataset,
        "seed": int(seed),
        "scope": "full",
        "n_rows": int(rows_sorted.size),
        "feat_shape": [int(x) for x in feat_shape],
        "D": int(np.prod(feat_shape)),
        "dtype": str(dt),
        "encoder_param_hash": enc_hash,
        "encode_precision": encode_precision_record(),
        "device": dev,
        "built_on_one_device": True,
        "features_file": feat_path.name,
        "features_sha256": file_sha256(feat_path),
        "features_bytes": int(feat_path.stat().st_size),
        "rows_file": rows_path.name,
        "rows_sha256": file_sha256(rows_path),
        "block": int(block),
        "n_blocks": n_blocks,
        "block_sha256": {
            k: v["sha256"] for k, v in sorted(prog["blocks"].items(), key=lambda kv: int(kv[0]))
        },
        "wall_clock_s": float(time.time() - t0),
        "source_generation1": source_manifest,
    }
    with open(out_dir / "feature_cache.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log(
        f"  generation 2 / seed {seed} COMPLETE: {rows_sorted.size} rows, features_sha256={manifest['features_sha256']}"
    )
    return {"complete": True, **manifest}


def compare_to_generation1(
    gen1_dir: str | Path, gen2_dir: str | Path, log: Callable = print
) -> dict:
    """How far generation 2 moved from generation 1 on the rows they share.

    This is an instrument measurement and is reported as one. It is NOT a licence
    to quote a generation-1 number beside a generation-2 number; it is the
    evidence for how much re-scoring was required.
    """
    gen1_dir, gen2_dir = (Path(gen1_dir), Path(gen2_dir))
    with open(gen1_dir / "feature_cache.json") as fh:
        m1 = json.load(fh)
    with open(gen2_dir / "feature_cache.json") as fh:
        m2 = json.load(fh)
    out = {
        "gen1": {
            k: m1.get(k)
            for k in ("scope", "n_rows", "dtype", "features_sha256", "encoder_param_hash")
        },
        "gen2": {
            k: m2.get(k)
            for k in ("scope", "n_rows", "dtype", "features_sha256", "encoder_param_hash")
        },
        "gen1_records_encode_precision": bool(m1.get("encode_precision") is not None),
    }
    if m1["encoder_param_hash"] != m2["encoder_param_hash"]:
        out["comparable"] = False
        out["reason"] = "different encoder parameter hashes - not the same bridge"
        log("  gen1/gen2 encoders differ; no element comparison is meaningful")
        return out
    r1 = np.load(gen1_dir / m1["rows_file"])
    r2 = np.load(gen2_dir / m2["rows_file"])
    shared = np.intersect1d(r1, r2)
    f1 = np.load(gen1_dir / m1["features_file"], mmap_mode="r")
    f2 = np.load(gen2_dir / m2["features_file"], mmap_mode="r")
    p1 = np.searchsorted(r1, shared)
    p2 = np.searchsorted(r2, shared)
    n_diff = 0
    n_elem = 0
    max_abs = 0.0
    step = 4096
    for i in range(0, shared.size, step):
        a = np.asarray(f1[p1[i : i + step]], np.float64)
        b = np.asarray(f2[p2[i : i + step]], np.float64)
        d = a != b
        n_diff += int(d.sum())
        n_elem += int(a.size)
        if d.any():
            max_abs = max(max_abs, float(np.abs(a - b).max()))
    out.update(
        {
            "comparable": True,
            "n_shared_rows": int(shared.size),
            "n_elements_compared": n_elem,
            "n_elements_differing": n_diff,
            "frac_elements_differing": float(n_diff / max(n_elem, 1)),
            "max_abs_diff": max_abs,
            "bit_identical": bool(n_diff == 0),
        }
    )
    if out["bit_identical"]:
        log(
            f"  gen1 and gen2 are BIT-IDENTICAL on all {shared.size} shared rows - generation 1 for this seed was CPU-written by the same path"
        )
    else:
        log(
            f"  gen1 vs gen2 differ on {n_diff}/{n_elem} elements ({out['frac_elements_differing']:.3%}), max abs diff {max_abs:.6g} over {shared.size} shared rows - a finding about the instrument, and the reason every arm is re-scored"
        )
    return out
