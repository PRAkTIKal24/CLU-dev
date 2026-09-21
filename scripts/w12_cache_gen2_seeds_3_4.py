"""w12 spoke/cache-gen2-seeds-3-4 - generation-2 feature caches for seeds 3 and 4.

THE ONE JOB: build the generation-2 full-scope feature caches for seeds 3 and 4,
with the SHIPPED re-encode gate applied, in the layout every downstream stage reads
(``<root>/shapes3d_recon/seed_<n>/{features.npy,feature_rows.npy,feature_cache.json}``).

This module measures nothing and scores nothing. It builds an instrument.

THE BLOCKER IT HAD TO ROUTE AROUND. ``$CHLU_GEN_BRIDGE_ROOT/shapes3d_recon`` holds
seeds 0, 1 and 2 only. Generation 2 is a *re-encode through the frozen bridge encoder*,
so seeds 3 and 4 have nothing to re-encode: no encoder, no generation-1 manifest to
assert its hash against. The encoders for seeds 3 and 4 are therefore FIT HERE by the
shipped ``chlu.experiments.exp_gen_bridge --stage all``, the same module that built
seeds 0-2, and that fact is written into every manifest this script produces.

Nothing here edits a shipped module. ``w4_cache2.build_gen2_cache`` and
``exp_gen_store._assert_cache_reencodes`` are called verbatim.

Stages::

    python scripts/w12_cache_gen2_seeds_3_4.py --stage det    --seeds 3 4
    python scripts/w12_cache_gen2_seeds_3_4.py --stage cache2 --seeds 3 4
    python scripts/w12_cache_gen2_seeds_3_4.py --stage gate   --seeds 3 4
    python scripts/w12_cache_gen2_seeds_3_4.py --stage ready  --seeds 3 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# One device, and it is the CPU. Pinned before JAX is imported by anything below,
# so w4_cache2.device_guard can never be satisfied by accident.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np  # noqa: E402

import chlu  # noqa: E402

# STANDING RULE: chlu must resolve inside THIS worktree. The shared site-packages
# carries an editable .pth pointing at another checkout; if that one wins, every byte
# this script writes is void.
WORKTREE = Path(__file__).resolve().parents[1]
_chlu = Path(chlu.__file__).resolve()
if WORKTREE not in _chlu.parents:
    raise SystemExit(
        f"chlu resolves to {_chlu}, which is NOT inside this worktree ({WORKTREE}). "
        "Refusing to build a byte.")

from chlu.config import get_default_config  # noqa: E402
from chlu.core.gen_bridge import dataset_module, param_hash  # noqa: E402
from chlu.eval.rivals import w4_cache2  # noqa: E402
from chlu.experiments import exp_gen_store  # noqa: E402
from chlu.utils.checkpoints import load_model  # noqa: E402

OUT = Path(os.environ.get("W12_OUT", "C:/chlu/outputs/w12-cache-gen2-seeds-3-4"))
DATASET = "shapes3d"
PRETEXT_DIR = "shapes3d_recon"

#: The generation-1 root, read ONLY to pin what seeds 0-2 are.
GEN1_ROOT = Path(os.environ["CHLU_GEN_BRIDGE_ROOT"]) / PRETEXT_DIR
#: The generation-2 root seeds 0-2 already live in - READ ONLY, never written.
GEN2_REF_ROOT = Path("C:/chlu/outputs/spoke-a-rivals/cache_generation2") / PRETEXT_DIR

BRIDGE_ROOT = OUT / "bridge" / PRETEXT_DIR
GEN2_ROOT = OUT / "cache_generation2" / PRETEXT_DIR


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def bridge_dir(seed: int) -> Path:
    return BRIDGE_ROOT / f"seed_{seed}"


def gen2_dir(seed: int) -> Path:
    return GEN2_ROOT / f"seed_{seed}"


def load_encoder(seed: int):
    """The frozen seed-N encoder, hash-asserted against its own bridge manifest."""
    d = bridge_dir(seed)
    with open(d / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    enc = load_model(d / "encoder.pkl")
    h = param_hash(enc)
    if h != bman["encoder_param_hash"]:
        raise AssertionError(
            f"seed {seed}: encoder.pkl hash {h} != bridge manifest "
            f"{bman['encoder_param_hash']} - this is not the frozen encoder")
    log(f"  seed {seed}: frozen encoder loaded, param_hash={h[:16]}... "
        f"(asserted against {d / 'bridge_manifest.json'})")
    return enc, bman


def full_rows(config) -> np.ndarray:
    mod = dataset_module(DATASET)
    split, _ = mod.build_split(config.gen_data.data_root)
    return np.arange(split.n_total, dtype=np.int64)


def _write(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    log(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Stage det - THE FALSIFIER, run BEFORE the expensive pass
# ---------------------------------------------------------------------------


def stage_det(args, config) -> dict:
    gb = config.gen_bridge
    rows = full_rows(config)
    block = int(gb.cache_block)
    out = {}
    for seed in args.seeds:
        enc, bman = load_encoder(seed)
        shard = rows[:block]
        log(f"  falsifier: seed {seed}, FIRST shard = rows {int(shard[0])}..{int(shard[-1])} "
            f"({shard.size} rows), dtype={gb.cache_dtype}")
        rec = w4_cache2.determinism_check(
            enc, DATASET, shard, config.gen_data.data_root, gb.cache_dtype,
            int(gb.eval_batch), block, log=log)
        rec["seed"] = int(seed)
        rec["encoder_param_hash"] = param_hash(enc)
        rec["bridge_manifest"] = str(bridge_dir(seed) / "bridge_manifest.json")
        out[str(seed)] = rec
        if not rec["passed"]:
            _write(out, OUT / "determinism_check.json")
            raise SystemExit(
                f"STOP: the re-encode determinism check FAILED on seed {seed}. "
                "Generation 2 would be noise; nothing is built.")
    _write(out, OUT / "determinism_check.json")
    return out


# ---------------------------------------------------------------------------
# Stage cache2 - the full-scope build, one CPU device, resumable
# ---------------------------------------------------------------------------


def stage_cache2(args, config) -> dict:
    gb = config.gen_bridge
    rows = full_rows(config)
    block = int(gb.cache_block)
    results = {}
    for seed in args.seeds:
        d = gen2_dir(seed)
        done = d / "feature_cache.json"
        if done.exists() and not args.force:
            with open(done) as fh:
                results[str(seed)] = {"complete": True, **json.load(fh)}
            log(f"  seed {seed}: generation 2 already complete, skipping")
            continue
        enc, bman = load_encoder(seed)
        t0 = time.time()
        res = w4_cache2.build_gen2_cache(
            enc, DATASET, rows, d, config.gen_data.data_root,
            dtype=gb.cache_dtype, eval_batch=int(gb.eval_batch), block=block,
            seed=seed, source_manifest={
                "NOTE": ("there is NO generation-1 cache for this seed; the source is "
                         "the w12 bridge built by exp_gen_bridge --stage all on this box"),
                "bridge_manifest": str(bridge_dir(seed) / "bridge_manifest.json"),
                "encoder_param_hash": bman["encoder_param_hash"],
                "decoder_param_hash": bman["decoder_param_hash"],
                "probe_param_hash": bman["probe_param_hash"],
                "bridge_gate": bman["result"]["gate"],
                "scope": bman["feature_cache"]["scope"],
                "n_rows": bman["feature_cache"]["n_rows"],
            },
            max_shard_seconds=args.max_seconds, log=log)
        res["elapsed_min"] = (time.time() - t0) / 60.0
        results[str(seed)] = res
        if not res.get("complete"):
            log(f"  seed {seed}: PARTIAL ({res['blocks_done']}/{res['n_blocks']} blocks); "
                "relaunch to resume")
    return results


# ---------------------------------------------------------------------------
# Stage gate - the SHIPPED re-encode gate, applied to the new caches
# ---------------------------------------------------------------------------


def stage_gate(args, config) -> dict:
    """Run ``exp_gen_store._assert_cache_reencodes`` - the gate exp_gen_cell fires at
    the top of every run_seed - against each new generation-2 cache.

    ``will_extend=False`` because this call encodes nothing new: it is an AUDIT, and the
    verdict is the deliverable. The bounds are unchanged (1.0 ULP, 1e-2 fraction) and
    ``would_have_failed`` records exactly what an enforced call would have done.
    """
    out = {"bounds": {"per_element_ulp": 1.0, "differing_fraction": 1e-2},
           "check": ("exp_gen_store._assert_cache_reencodes, the SHIPPED gate, run "
                     "against the w12 generation-2 caches with the encoder stored "
                     "beside each cache"),
           "generations": {"generation_2_w12": {}}}
    for seed in args.seeds:
        d = gen2_dir(seed)
        if not (d / "feature_cache.json").exists():
            out["generations"]["generation_2_w12"][str(seed)] = {
                "NOT_RUN": "no generation-2 cache for this seed"}
            continue
        with open(d / "feature_cache.json") as fh:
            cman = json.load(fh)
        enc, _ = load_encoder(seed)
        crows = np.load(d / cman["rows_file"])
        feats = np.load(d / cman["features_file"], mmap_mode="r")
        log(f"  seed {seed}: re-encode gate on {d} ...")
        st = exp_gen_store._assert_cache_reencodes(
            DATASET, config, enc, crows, feats, cman, log=log, will_extend=False,
            reason=("AUDIT: this call encodes nothing new, so it cannot write an "
                    "inconsistent row. The gate's verdict on seeds 3/4 is the "
                    "deliverable of w12 spoke/cache-gen2-seeds-3-4."))
        st["passed"] = not bool(st["would_have_failed"])
        st["cache_dir"] = str(d)
        out["generations"]["generation_2_w12"][str(seed)] = st
        del feats
    per = out["generations"]["generation_2_w12"]
    ran = {k: v for k, v in per.items() if "passed" in v}
    out["verdict"] = {
        "seeds_gated": sorted(ran),
        "passes": sorted(k for k, v in ran.items() if v["passed"]),
        "failures": sorted(k for k, v in ran.items() if not v["passed"]),
        "max_ulp_range": ([min(v["max_ulp_delta"] for v in ran.values()),
                           max(v["max_ulp_delta"] for v in ran.values())]
                          if ran else None),
    }
    _write(out, OUT / "reencode_gate_seeds_3_4.json")
    return out


# ---------------------------------------------------------------------------
# Stage ready - CACHE-READY.json, the acceptance criterion
# ---------------------------------------------------------------------------

PIN_KEYS = ("schema", "generation", "scope", "n_rows", "dtype", "feat_shape", "D",
            "block", "n_blocks", "features_bytes")


def _ref_shape(seed: int) -> dict:
    """Shape/dtype of an EXISTING seed-0..2 generation-2 cache, for the match assertion."""
    p = GEN2_REF_ROOT / f"seed_{seed}" / "feature_cache.json"
    with open(p) as fh:
        m = json.load(fh)
    d = {"source": str(p), "seed": int(m["seed"]),
         "features_sha256": m["features_sha256"]}
    d.update({k: m[k] for k in PIN_KEYS})
    return d


def stage_ready(args, config) -> dict:
    ref = {str(s): _ref_shape(s) for s in (0, 1, 2)}
    pin = {k: ref["0"][k] for k in PIN_KEYS}
    # Every seed 0-2 must already agree with the pin, or the pin is not a pin.
    pin_disagreements = {s: {k: [pin[k], ref[s][k]] for k in PIN_KEYS
                             if ref[s][k] != pin[k]} for s in ref}
    pin_disagreements = {s: v for s, v in pin_disagreements.items() if v}

    gate_p = OUT / "reencode_gate_seeds_3_4.json"
    gate = json.load(open(gate_p)) if gate_p.exists() else None
    det_p = OUT / "determinism_check.json"
    det = json.load(open(det_p)) if det_p.exists() else None

    seeds_out = {}
    for seed in args.seeds:
        d = gen2_dir(seed)
        man_p = d / "feature_cache.json"
        if not man_p.exists():
            seeds_out[str(seed)] = {
                "status": "NOT_RUN",
                "reason": "generation-2 cache for this seed was not completed",
            }
            continue
        with open(man_p) as fh:
            m = json.load(fh)
        mismatch = {k: {"seeds_0_2": pin[k], "this_seed": m.get(k)} for k in PIN_KEYS
                    if m.get(k) != pin[k]}
        g = (gate or {}).get("generations", {}).get("generation_2_w12", {}).get(str(seed))
        bman_p = bridge_dir(seed) / "bridge_manifest.json"
        bman = json.load(open(bman_p))
        seeds_out[str(seed)] = {
            "status": "READY" if (not mismatch and g and g.get("passed")) else "SEE_FLAGS",
            "cache_dir": str(d.resolve()),
            "features_file": str((d / m["features_file"]).resolve()),
            "rows_file": str((d / m["rows_file"]).resolve()),
            "manifest_file": str(man_p.resolve()),
            "n_rows": m["n_rows"],
            "dtype": m["dtype"],
            "feat_shape": m["feat_shape"],
            "array_shape": [m["n_rows"]] + list(m["feat_shape"]),
            "D": m["D"],
            "scope": m["scope"],
            "schema": m["schema"],
            "generation": m["generation"],
            "block": m["block"],
            "n_blocks": m["n_blocks"],
            "features_bytes": m["features_bytes"],
            "features_sha256": m["features_sha256"],
            "rows_sha256": m["rows_sha256"],
            "block_sha256": m["block_sha256"],
            "device": m["device"],
            "built_on_one_device": m["built_on_one_device"],
            "encoder_param_hash": m["encoder_param_hash"],
            "encode_precision": m["encode_precision"],
            "wall_clock_s": m["wall_clock_s"],
            "shape_dtype_matches_seeds_0_2": not mismatch,
            "shape_dtype_mismatches": mismatch or None,
            "reencode_gate": g,
            "determinism_check": (det or {}).get(str(seed)),
            "bridge": {
                "manifest": str(bman_p.resolve()),
                "built_by": ("chlu.experiments.exp_gen_bridge --stage all, on THIS box, "
                             "on one CPU device - there was no generation-1 bridge for "
                             "this seed"),
                "encoder_param_hash": bman["encoder_param_hash"],
                "decoder_param_hash": bman["decoder_param_hash"],
                "probe_param_hash": bman["probe_param_hash"],
                "gate": bman["result"]["gate"],
                "rows": bman["rows"],
                "training": bman["training"],
            },
        }
    ready = {
        "schema": "chlu-w12-cache-ready/1",
        "spoke": "w12 spoke/cache-gen2-seeds-3-4",
        "produced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chlu_file": str(_chlu),
        "worktree": str(WORKTREE),
        "generation2_root": str(GEN2_ROOT.resolve()),
        "downstream_layout": (
            "<generation2_root>/seed_<n>/{features.npy,feature_rows.npy,"
            "feature_cache.json} - identical to "
            "C:/chlu/outputs/spoke-a-rivals/cache_generation2/shapes3d_recon/seed_<0,1,2>. "
            "A downstream stage reads it by pointing gen_store.bridge_root at "
            "<generation2_root>/.. after copying the four BRIDGE_SIDECARS "
            "(encoder.pkl, probe.pkl, decoder.pkl, bridge_manifest.json) beside the "
            "cache - exp_w4_rivals.stage_bridgify does exactly this for seeds 0-2."),
        "shape_dtype_pin": pin,
        "shape_dtype_pin_sources": ref,
        "shape_dtype_pin_disagreements_within_seeds_0_2": pin_disagreements or None,
        "seeds": seeds_out,
    }
    _write(ready, OUT / "CACHE-READY.json")
    return ready


STAGES = {"det": stage_det, "cache2": stage_cache2, "gate": stage_gate,
          "ready": stage_ready}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 4])
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-seconds", type=float, default=None, dest="max_seconds")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    config = get_default_config()
    log(f"=== w12 cache-gen2 stage={args.stage} seeds={args.seeds} ===")
    log(f"  chlu package: {_chlu}")
    log(f"  gen2 root: {GEN2_ROOT}")
    log(f"  data_root: {config.gen_data.data_root}")
    STAGES[args.stage](args, config)
    log(f"=== stage {args.stage} done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
