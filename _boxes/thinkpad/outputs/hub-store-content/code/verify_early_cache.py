"""hub/store-content — VERIFY the earlier-layer cache against the frozen encoder.

⛔ Written because this hub once had two writers on the same memmap.  A cache that
has to be argued about is not an instrument; this re-encodes a random sample of rows
from pixels through the frozen encoder and requires the cache to reproduce them to
within ONE representable step of the cache's own dtype — the same standard, and the
same derivation, as ``exp_gen_store._assert_cache_reencodes``.

It also re-checks the sha256 of each block against the manifest, which catches a
torn write the sampling might miss.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earlylayer import CONTENT_BLOCKS, encode_blocks   # noqa: E402

from chlu.data import shapes3d as ds                    # noqa: E402
from chlu.utils.checkpoints import load_model           # noqa: E402
from chlu.core.gen_bridge import param_hash             # noqa: E402

OUT = Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs")) / "hub-store-content"
BRIDGE = Path(os.environ.get(
    "CHLU_GEN_BRIDGE_ROOT",
    "C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-feature-bridge/runs"))
#: derived exactly as the pin derives it: a float32 reorder flips a float16 element
#: only near a rounding midpoint, expected ~4e-4; the pin uses 1e-2 and so do we.
MAX_DIFF_FRACTION = 1e-2


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=384, help="rows re-encoded from pixels")
    ap.add_argument("--skip-sha", action="store_true")
    a = ap.parse_args(argv)

    cdir = OUT / "cache" / f"shapes3d_recon_seed_{a.seed}"
    man = json.load(open(cdir / "earlylayer_cache.json"))
    bdir = BRIDGE / "shapes3d_recon" / f"seed_{a.seed}"
    enc = load_model(bdir / "encoder.pkl")
    eh = param_hash(enc)
    assert eh == man["encoder_param_hash"], "encoder != the cache's own manifest"

    report = {"seed": int(a.seed), "encoder_param_hash": eh, "n_sampled": int(a.n),
              "blocks": {}, "sha256": {}}

    if not a.skip_sha:
        t0 = time.time()
        for b, meta in man["blocks"].items():
            got = sha256_file(cdir / meta["file"])
            report["sha256"][b] = {"manifest": meta["sha256"], "measured": got,
                                   "match": bool(got == meta["sha256"])}
            print(f"  sha256 {b}: {'OK' if got == meta['sha256'] else 'MISMATCH'}")
        print(f"  (sha pass {time.time()-t0:.0f}s)")

    # spread the sample over the whole grid, including both ends and the block seams
    n_rows = int(man["n_rows"])
    rng = np.random.default_rng(20_260_919)
    rows = np.unique(np.concatenate([
        np.array([0, 1, n_rows - 2, n_rows - 1], np.int64),
        np.arange(0, n_rows, 15_000, dtype=np.int64),          # every h5 chunk seam
        np.arange(1, n_rows, 15_000, dtype=np.int64),
        rng.choice(n_rows, max(a.n - 70, 8), replace=False).astype(np.int64),
    ]))
    h = ds.open_images(None)
    try:
        d = h["images"]
        imgs = np.stack([np.asarray(d[int(r)]) for r in rows])
    finally:
        h.close()
    x = imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    got = encode_blocks(enc, x, list(man["blocks"].keys()), batch=128)

    ok = True
    for b in man["blocks"]:
        cached = np.asarray(np.load(cdir / f"{b}.npy", mmap_mode="r")[rows], np.float32)
        ref = got[b].astype(np.float16).astype(np.float32)
        diff = np.abs(cached - ref)
        # tolerance derived from the cache dtype at each element's own magnitude
        tol = np.spacing(np.abs(cached).astype(np.float16)).astype(np.float32)
        over = diff > tol
        frac = float(over.mean())
        exact = float((diff == 0).mean())
        good = bool(frac <= MAX_DIFF_FRACTION)
        ok &= good
        report["blocks"][b] = {
            "n_elements": int(diff.size), "frac_exact": exact,
            "frac_over_one_ulp": frac, "max_abs": float(diff.max()),
            "bound": MAX_DIFF_FRACTION, "passed": good}
        print(f"  {b}: exact {exact:.6f}  over-1-ULP {frac:.2e}  maxabs "
              f"{float(diff.max()):.3e}  {'OK' if good else 'FAIL'}")

    report["n_rows_checked"] = int(rows.size)
    report["passed"] = bool(ok and all(v["match"] for v in report["sha256"].values())
                            if report["sha256"] else ok)
    json.dump(report, open(cdir / "verify_report.json", "w"), indent=1)
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}  "
          f"({rows.size} rows re-encoded from pixels)")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
