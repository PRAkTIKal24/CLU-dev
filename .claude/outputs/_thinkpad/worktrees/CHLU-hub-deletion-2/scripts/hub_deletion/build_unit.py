#!/usr/bin/env python
"""ONE resumable unit of `hub/deletion`: build one store from one live row set.

    "a kill costs ONE UNIT, never the run"  (scripts/csf3/job_cpu_gen.sh)

Writes `<out>/<tag>/build.json` + `<out>/<tag>/modes.npz` and then STOPS.  A unit
whose `build.json` already exists is skipped, so the driver is restartable.

*** THE RE-ENCODE HAZARD CANNOT ARISE HERE, STRUCTURALLY. ***
This script never imports and never constructs an encoder.  Features come from
`gen_bridge.features_for_rows`, which RAISES on any row the frozen cache does not
hold.  There is no extension path to take, so there is nothing for the 1-ULP gate
to bind on.  Seed 0's cache is scope="full" (480,000 rows = the whole grid), and
the coverage shortfall is asserted == 0 below and recorded in every build.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from chlu.config import CHLUConfig  # noqa: E402
from chlu.core.gen_bridge import (  # noqa: E402
    assert_no_holdout_leak,
    dataset_module,
    features_for_rows,
    load_feature_cache,
)
from chlu.experiments.exp_gen_deletion import (  # noqa: E402
    canonical_build,
    delete_rows,
    factor_tuples,
    row_predicate,
    store_digest,
)
from chlu.experiments.exp_gen_store import bridge_dir, select_store_rows  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# THE HUB-WIDE BUILD LOCK.  ⚠ Added 11:22Z after a real collision, not in advance.
# ---------------------------------------------------------------------------
# Two drivers were found running `--tag det_B` at the same moment, both about to
# `np.savez` the same `modes.npz`.  Nothing had been written yet, so no artefact was
# damaged — but a torn `modes.npz` would have silently corrupted the DETERMINISM
# GATE itself, which is the one result everything else rests on.
#
# The lock is HUB-WIDE, not per-tag, and that is deliberate: it also enforces
# LAUNCH-COMMON §3's "no hub exceeds its 4 threads", because two concurrent builds
# at OMP_NUM_THREADS=4 would be 8.  A second driver blocks here, and when it wakes it
# re-checks whether the unit it wanted was finished by the other driver in the
# meantime -- so two drivers are now SAFE and mutually redundant rather than racing.


class BuildLock:
    def __init__(self, path: Path, poll: float = 20.0, stale_s: float = 7200.0):
        self.path, self.poll, self.stale_s, self.fd = Path(path), poll, stale_s, None

    def _holder_alive(self) -> bool:
        try:
            info = json.loads(self.path.read_text())
        except Exception:
            return False           # unreadable/half-written -> treat as stale
        if time.time() - float(info.get("t", 0)) > self.stale_s:
            return False
        pid = int(info.get("pid", -1))
        try:                        # Windows-safe liveness probe
            import subprocess
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                 capture_output=True, text=True).stdout
            return str(pid) in out
        except Exception:
            return True             # cannot tell -> assume alive, never steal blindly

    def acquire(self):
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, json.dumps(
                    {"pid": os.getpid(), "t": time.time(),
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                ).encode())
                os.close(self.fd)
                return self
            except FileExistsError:
                if not self._holder_alive():
                    log(f"⚠ stale build lock at {self.path} — reclaiming")
                    try:
                        os.unlink(self.path)      # the LOCK only; ⛔ never an artefact
                    except OSError:
                        pass
                    continue
                log(f"another build holds the hub lock; waiting {self.poll:.0f}s "
                    f"(⭐ this is the 4-thread budget being enforced, not a hang)")
                time.sleep(self.poll)

    def release(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass


#: ⭐ THE CODE FINGERPRINT.  ⚠ Added 11:26Z after the sibling thread pointed out that
#: `exp_gen_deletion.py` was edited at 11:20:14Z — AFTER `det_A` was built at 11:13:12Z.
#: The edit was comment-only, but nothing in the artefacts *proved* that, and
#: "stale code vs stale artefact" is exactly how a collateral number gets confounded by
#: an edit instead of by the deletion.  Every unit now records the sha256 of every file
#: on its build path, so the analysis can ASSERT one fingerprint across the cell and any
#: unit that differs is visible rather than silent.
CODE_FILES = [
    "chlu/experiments/exp_gen_deletion.py",
    "chlu/core/gen_store.py",
    "chlu/core/gen_dream.py",
    "chlu/core/gen_request.py",
    "chlu/core/gen_bridge.py",
    "scripts/hub_deletion/build_unit.py",
]


def code_fingerprint() -> dict:
    out = {}
    for rel in CODE_FILES:
        p = _REPO / rel
        try:
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            out[rel] = None
    h = hashlib.sha256()
    for rel in CODE_FILES:
        h.update(f"{rel}:{out[rel]}\n".encode())
    return {"files": out, "combined": h.hexdigest()}


def write_atomic(path: Path, text: str) -> None:
    """⛔ Never leave a half-written artefact: temp file in the same dir, then rename."""
    tmp = Path(str(path) + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# The live set: one place, so every arm and every rival takes the SAME predicate
# ---------------------------------------------------------------------------


def resolve_live(spec: dict, pool: np.ndarray, tuples: np.ndarray) -> dict:
    """`spec` -> the sorted live row vector, plus what was removed and why."""
    kind = spec["kind"]
    if kind == "full":
        return {"live": np.sort(pool), "n_deleted": 0, "deleted_rows": np.array([], np.int64),
                "predicate": None}
    if kind == "delete":
        # [[factor, value], ...]; length 1 = a factor value, length >= 2 = a conjunction
        pairs = [(int(f), int(v)) for f, v in spec["spec"]]
        out = delete_rows(pool, tuples, pairs)
        out["predicate"] = pairs
        return out
    if kind in ("sequence", "union"):
        # ⭐ PREREG §4c, PATH-INDEPENDENCE — the part of the provenance claim that is
        # NOT free.  `sequence` deletes one predicate at a time, each applied to the
        # ALREADY-REDUCED live set; `union` removes everything matching ANY of them in
        # a single pass over the ORIGINAL pool.  Two genuinely different code paths.
        # They must land on the same live set and therefore the same digest — which is
        # the whole content of "history-independent placement".
        specs = [[(int(f), int(v)) for f, v in s] for s in spec["specs"]]
        if kind == "sequence":
            cur, tup = np.sort(pool), tuples
            for s in specs:
                keep = ~row_predicate(tup, s)
                cur, tup = cur[keep], tup[keep]
            kept = cur
        else:
            mask = np.zeros(pool.shape[0], bool)
            for s in specs:
                mask |= row_predicate(tuples, s)
            kept = np.sort(pool[~mask])
        return {"live": kept, "n_deleted": int(pool.size - kept.size),
                "deleted_rows": np.sort(np.setdiff1d(pool, kept)), "predicate": specs}
    if kind == "placebo":
        # THE CHURN CONTROL.  Same number of rows, chosen at random, so a rebuild's
        # own noise is measured instead of being attributed to the deletion.
        n = int(spec["n"])
        rng = np.random.default_rng(int(spec.get("seed", 0)) * 6_700_417 + 11)
        drop = rng.choice(pool.size, size=n, replace=False)
        mask = np.zeros(pool.size, bool)
        mask[drop] = True
        return {"live": np.sort(pool[~mask]), "n_deleted": int(n),
                "deleted_rows": np.sort(pool[mask]), "predicate": None}
    raise ValueError(f"unknown live-set kind {kind!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--spec", required=True, help="JSON string or @file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grouping", default="correlation")
    ap.add_argument("--dataset", default="shapes3d")
    ap.add_argument("--pretext", default="recon")
    ap.add_argument("--train-steps", type=int, default=None, help="smoke only")
    ap.add_argument("--n-store-train", type=int, default=None, help="smoke only")
    ap.add_argument("--n-corr", type=int, default=None, help="smoke only")
    ap.add_argument("--n-modes", type=int, default=None, help="smoke only")
    ap.add_argument("--no-verify-cache", action="store_true",
                    help="skip the 1.9 GB sha256 (already verified this session)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    t_unit = time.perf_counter()
    spec_txt = args.spec
    if spec_txt.startswith("@"):
        spec_txt = Path(spec_txt[1:]).read_text()
    spec = json.loads(spec_txt)

    out_dir = Path(args.out) / args.tag
    done = out_dir / "build.json"
    if done.exists() and not args.force:
        log(f"SKIP {args.tag} — already built ({done})")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    # ⛔ one build at a time across the whole hub (see BuildLock). Re-check after
    # acquiring: another driver may have finished this very unit while we waited.
    lock = BuildLock(Path(args.out).parent / ".build.lock").acquire()
    try:
        if done.exists() and not args.force:
            log(f"SKIP {args.tag} — built by another driver while this one waited")
            return 0
        return _build(args, out_dir, done, spec, t_unit)
    finally:
        lock.release()


def _build(args, out_dir: Path, done: Path, spec: dict, t_unit: float) -> int:
    config = CHLUConfig()
    g = config.gen_store
    g.dataset, g.pretext = args.dataset, args.pretext
    smoke = False
    for attr, val in (("train_steps", args.train_steps), ("n_store_train", args.n_store_train),
                      ("n_corr", args.n_corr), ("n_modes", args.n_modes)):
        if val is not None:
            setattr(g, attr, int(val))
            smoke = True

    bdir = bridge_dir(config, args.dataset, args.pretext, args.seed)
    log(f"frozen bridge: {bdir}")
    crows, feats, cman = load_feature_cache(bdir, verify=not args.no_verify_cache)
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    log(f"cache: {cman['n_rows']} rows scope={cman['scope']} D={cman['D']} {cman['dtype']}")

    mod = dataset_module(args.dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    rows = select_store_rows(split, config, args.seed)
    pool = np.sort(np.asarray(rows["store_train"], np.int64))

    leak = assert_no_holdout_leak(split, pool)
    log(f"no-leak: {leak['n_checked']} rows checked, {leak['n_holdout_leaked']} leaked")

    tuples = factor_tuples(grid, pool)
    live_info = resolve_live(spec, pool, tuples)
    live = live_info["live"]

    # --- THE COVERAGE ASSERTION.  shortfall > 0 => STOP, never encode. ---------
    need = np.unique(np.concatenate([live, rows["eval_a"], rows["eval_b"]]))
    pos = np.searchsorted(crows, need)
    hit = (pos < crows.size) & (crows[np.minimum(pos, crows.size - 1)] == need)
    shortfall = int((~hit).sum())
    if shortfall:
        raise SystemExit(
            f"STOP: cache coverage shortfall = {shortfall} rows. Per LAUNCH-COMMON §4 a "
            "shortfall > 0 means STOP and report, never encode."
        )
    log(f"coverage shortfall = 0 over {need.size} required rows (no encode path exists here)")

    log(f"live = {live.size} rows  (deleted {live_info['n_deleted']}, "
        f"predicate={live_info['predicate']})")
    # (n, C, H, W) -> (n, D).  The store's own reshape is identical; doing it once
    # here keeps `canonical_build` taking a single unambiguous 2-D array.
    X_live = features_for_rows(crows, feats, live).reshape((live.size, -1))
    log(f"gathered X_live {X_live.shape} {X_live.dtype}")

    # The live rows' OWN factor tuples: the address book's supervision, and the one
    # place the deleted value's *name* could survive a rebuild.
    lab_live = factor_tuples(grid, live)
    res = canonical_build(
        X_live, g, seed=args.seed, k=5, grouping=args.grouping,
        feat_shape=feat_shape, labels_live=lab_live,
        cardinalities=list(grid.cardinalities), log=log,
    )
    d = res["digest"]
    log(f"store_digest = {d['store_digest']}")
    log(f"param_hash   = {d['param_hash']}  book_hash = {d.get('book_hash', '-')}")
    log(f"well sizes   = {d['well_sizes']}  (train {res['timing_s']['train']:.1f} s, "
        f"book {res['timing_s']['book']:.1f} s, total {res['timing_s']['total']:.1f} s)")

    book = res["book"]
    _mtmp = out_dir / f"modes.npz.tmp.{os.getpid()}.npz"
    np.savez(
        _mtmp,
        **{f"well_{j}": np.asarray(m, np.float32) for j, m in enumerate(res["store"].modes)},
        **{f"chan_{j}": np.asarray(w, np.int64) for j, w in enumerate(res["well_channels"])},
        **{f"cond_{f}": np.asarray(lc, np.float32) for f, lc in enumerate(book.log_cond)},
        log_prior=np.asarray(book.log_prior, np.float32),
        codes=np.asarray(res["codes"], np.int32),
        lab_live=np.asarray(lab_live, np.int32),
        live=live, deleted=live_info["deleted_rows"], labels=np.asarray(res["labels"], np.int64),
    )
    os.replace(_mtmp, out_dir / "modes.npz")

    payload = {
        "schema": "chlu-hub-deletion/build/1",
        "tag": args.tag,
        "smoke": smoke,
        "spec": spec,
        "predicate": live_info["predicate"],
        "seed": args.seed,
        "grouping": args.grouping,
        "n_live": int(live.size),
        "n_deleted": int(live_info["n_deleted"]),
        "live_sha256": __import__("hashlib").sha256(live.tobytes()).hexdigest(),
        "deleted_sha256": __import__("hashlib").sha256(
            np.asarray(live_info["deleted_rows"], np.int64).tobytes()).hexdigest(),
        "digest": d,
        "grouping_info": res["grouping_info"],
        "build_inputs": res["build_inputs"],
        "timing_s": res["timing_s"],
        "unit_wall_s": time.perf_counter() - t_unit,
        "loss_first": float(res["history"][0]) if res["history"] else None,
        "loss_final": float(res["history"][-1]) if res["history"] else None,
        "cache": {
            "scope": cman["scope"], "n_rows": int(cman["n_rows"]),
            "features_sha256": cman["features_sha256"],
            "encoder_param_hash": cman["encoder_param_hash"],
            "coverage_shortfall": 0,
            "encoder_constructed": False,
            "note": "features_for_rows only; this unit has no encode path.",
        },
        "code": code_fingerprint(),
        "machine": {
            "platform": sys.platform,
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "python": sys.version.split()[0],
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # ⛔ `build.json` is the unit's DONE marker, so it is written last and atomically:
    # a build.json that exists must mean a modes.npz that is complete.
    write_atomic(done, json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    log(f"WROTE {done}  ({payload['unit_wall_s']:.1f} s wall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
