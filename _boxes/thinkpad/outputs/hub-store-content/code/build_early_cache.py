"""hub/store-content — build the EARLIER-LAYER feature cache, once, on local CPU.

⭐ Mirrors the frozen cache's schema (`chlu-gen-feature-cache/1`) so the same
downstream code can read it: float16, sorted rows, sha256 over the array, a manifest
carrying the encoder parameter hash.  ⛔ It is written into
``$CHLU_HUB_OUTPUTS/hub-store-content/cache`` and nothing in the frozen instrument is
touched.

WHY A CACHE AT ALL.  The frozen cache holds only the penultimate map; an earlier
block's activation is not derivable from it.  Measured on this box: a *scattered*
h5 read runs at 110 rows/s and a *contiguous* one at 8,800 rows/s, while the encoder
itself runs at 1,670 rows/s.  ⇒ one contiguous full-grid pass (~6 min) is cheaper
than any per-run scatter, it makes every later run a pure cache READ, and it means
the re-encode gate can never fire inside a scored run.

RESUMABLE: the memmap is written in contiguous blocks and ``progress.json`` records
the high-water mark.  A kill costs one block, not the pass.  Re-invoke to continue.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earlylayer import CONTENT_BLOCKS, encode_blocks  # noqa: E402

from chlu.data import shapes3d as ds                   # noqa: E402
from chlu.utils.checkpoints import load_model          # noqa: E402
from chlu.core.gen_bridge import param_hash            # noqa: E402

BRIDGE = os.environ.get(
    "CHLU_GEN_BRIDGE_ROOT",
    "C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-feature-bridge/runs",
)
OUT = Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs")) / "hub-store-content" / "cache"
N_ROWS = 480_000
BLOCKS = ("b1", "b2", "b3")          # ⛔ 'p' is NOT rebuilt — it comes from the frozen cache


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
    ap.add_argument("--block", type=int, default=15_000, help="h5 chunk-aligned read block")
    ap.add_argument("--sub", type=int, default=500, help="encode sub-batch (memory cap)")
    ap.add_argument("--n-rows", type=int, default=N_ROWS)
    ap.add_argument("--max-seconds", type=float, default=3600.0, help="wall-clock cap; exits resumable")
    a = ap.parse_args(argv)

    bdir = Path(BRIDGE) / "shapes3d_recon" / f"seed_{a.seed}"
    bman = json.load(open(bdir / "bridge_manifest.json"))
    enc = load_model(bdir / "encoder.pkl")
    eh = param_hash(enc)
    assert eh == bman["encoder_param_hash"], "encoder does not match the bridge manifest"

    out = OUT / f"shapes3d_recon_seed_{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    # -- SINGLE-WRITER LOCK.  Two builders once ran against this memmap at the same
    # -- time (2026-09-19); a cache with two writers is not an instrument.
    lock = out / "WRITER.lock"
    if lock.exists():
        other = lock.read_text().strip()
        alive = False
        try:
            import subprocess
            r = subprocess.run(["tasklist", "/FI", f"PID eq {other.split()[0]}"],
                               capture_output=True, text=True)
            alive = other.split()[0] in r.stdout
        except Exception:
            alive = True
        if alive:
            print(f"REFUSING: another writer holds the lock ({other})")
            return 4
        print(f"stale lock from {other} - taking it over")
    lock.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    prog_path = out / "progress.json"
    done = json.load(open(prog_path))["done"] if prog_path.exists() else 0

    mm = {}
    for b in BLOCKS:
        c = CONTENT_BLOCKS[b][1]
        p = out / f"{b}.npy"
        mode = "r+" if p.exists() and done else "w+"
        mm[b] = np.lib.format.open_memmap(
            p, mode=mode, dtype=np.float16, shape=(a.n_rows, c, 4, 4)
        )
    if done:
        print(f"resuming at row {done:,}/{a.n_rows:,}")

    t0 = time.time()
    h = ds.open_images(None)
    try:
        d = h["images"]
        i = done
        while i < a.n_rows:
            j = min(i + a.block, a.n_rows)
            raw = np.asarray(d[i:j])                                  # (n,64,64,3) uint8
            for s in range(0, j - i, a.sub):
                e = min(s + a.sub, j - i)
                x = (raw[s:e].transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
                got = encode_blocks(enc, x, BLOCKS, batch=a.sub)
                for b in BLOCKS:
                    mm[b][i + s : i + e] = got[b].astype(np.float16)
                del x, got
            del raw
            i = j
            for b in BLOCKS:
                mm[b].flush()
            json.dump({"done": i, "n_rows": a.n_rows}, open(prog_path, "w"))
            el = time.time() - t0
            rate = (i - done) / max(el, 1e-9)
            print(f"  {i:,}/{a.n_rows:,}  {rate:7.1f} rows/s  eta {(a.n_rows-i)/max(rate,1e-9)/60:5.1f} min",
                  flush=True)
            if el > a.max_seconds:
                print("wall-clock cap reached - exiting RESUMABLE")
                return 2
    finally:
        h.close()
        for b in BLOCKS:
            mm[b].flush()
        del mm
        try:
            lock.unlink()
        except OSError:
            pass

    man = {
        "schema": "chlu-gen-earlylayer-cache/1",
        "dataset": "shapes3d", "pretext": "recon", "seed": int(a.seed),
        "n_rows": int(a.n_rows), "dtype": "float16",
        "rows": "implicit arange(n_rows) - the full grid, in order",
        "encoder_param_hash": eh,
        "bridge_dir": str(bdir),
        "pooling": "exact 4x4 block-mean of the post-ReLU block output "
                   "(chlu.experiments <- hub code earlylayer._pool_to)",
        "device": "local CPU",
        "precision": "gen_bridge.ENCODE_PRECISION, entered at trace time - the "
                     "last block is BIT-IDENTICAL to encode_images (verified)",
        "blocks": {b: {"file": f"{b}.npy",
                       "shape": [int(a.n_rows), CONTENT_BLOCKS[b][1], 4, 4],
                       "dims": CONTENT_BLOCKS[b][2],
                       "sha256": sha256_file(out / f"{b}.npy")} for b in BLOCKS},
        "note": "The penultimate map ('p') is NOT here: it is read from the FROZEN "
                "cache so every arm's penultimate content is bit-identical to the "
                "incumbent's.",
    }
    json.dump(man, open(out / "earlylayer_cache.json", "w"), indent=1)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
