#!/usr/bin/env python3
"""Trivial, deterministic, checkpointed GPU payload -- the cloud path's smoke test.

This runs no science. Its entire job is to be a *real* GPU workload that is
(a) long enough to be interrupted, (b) exactly reproducible, so that
"the resumed run continued" and "the resumed run restarted" are
distinguishable by comparing bytes rather than by reading a log and hoping.

The determinism device is a **hash chain**. Every step folds the step's GPU
result into a running SHA-256:

    chain_t = sha256(chain_{t-1} || bytes(acc_t))

A run that silently restarted from zero, dropped a step, repeated a step, or
produced a different float anywhere in its history cannot arrive at the same
final ``chain``. One 64-character string therefore certifies the entire
history, not just the endpoint -- which is what a resume claim actually needs.

Pre-emption is handled twice over, because on Spot it is the expected ending:
  * a SIGTERM handler (GCE's shutdown sequence, and `docker stop`), and
  * a thread long-polling the metadata server's ``instance/preempted`` flag.
Either one sets the stop flag; the loop then FORCE-checkpoints and exits 0
with ``preempted: true`` in its progress file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ckpt import Checkpointer, FingerprintMismatch, config_fingerprint  # noqa: E402

_METADATA = "http://metadata.google.internal/computeMetadata/v1/instance"
_STOP = threading.Event()
_STOP_REASON: list[str] = []


def _stop(reason: str) -> None:
    if not _STOP.is_set():
        _STOP_REASON.append(reason)
        _STOP.set()
        print(f"[payload] STOP requested: {reason}", flush=True)


def _install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _f: _stop(f"signal {signal.Signals(s).name}"))


def _preemption_watcher() -> None:
    """Long-poll the GCE metadata server until this VM is pre-empted.

    `wait_for_change=true` blocks server-side, so this costs one idle socket
    rather than a polling loop, and fires within a second of the notice --
    well inside Spot's ~30 s budget.
    """
    import urllib.error
    import urllib.request

    url = f"{_METADATA}/preempted?wait_for_change=true&timeout_sec=300"
    while not _STOP.is_set():
        try:
            req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(req, timeout=330) as resp:
                if resp.read().decode().strip().upper() == "TRUE":
                    _stop("GCE pre-emption notice")
                    return
        except urllib.error.URLError:
            return  # not on GCE (laptop/test): nothing to watch
        except Exception:
            time.sleep(5)


# ----------------------------------------------------------------- the work


def _build_step_fn(n: int, dtype, seed: int):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def step_fn(acc, t):
        # Deterministic in (seed-folded key, t) alone -- carries nothing from
        # the process, so step t is identical whether it is the 1st step of a
        # resumed run or the 400th of an uninterrupted one.
        key = jax.random.fold_in(jax.random.PRNGKey(seed), t)
        a = jax.random.normal(key, (n, n), dtype=dtype)
        b = jax.random.normal(jax.random.fold_in(key, 1), (n, n), dtype=dtype)
        return acc + jnp.tanh(a @ b).sum(axis=0)

    return step_fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True, help="job name (namespaces the GCS prefix)")
    ap.add_argument("--out", default="/var/chlu/out", help="local output dir")
    ap.add_argument("--remote", default=None, help="gs://bucket/prefix for checkpoints+results")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1024, help="matmul edge length")
    ap.add_argument("--ckpt-every-s", type=float, default=20.0)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--step-sleep-s", type=float, default=0.0,
                    help="artificial per-step delay; stretches the run so it can be "
                         "interrupted. Not part of the fingerprint -- it cannot change a result.")
    ap.add_argument("--fresh", action="store_true", help="discard checkpoints and start over")
    ap.add_argument("--require-gpu", action="store_true", help="fail unless JAX sees a GPU")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    _install_signal_handlers()
    threading.Thread(target=_preemption_watcher, daemon=True).start()

    import jax
    import jax.numpy as jnp

    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    print(f"[payload] jax {jax.__version__} backend={backend} devices={devices}", flush=True)
    if args.require_gpu and backend != "gpu":
        print(f"[payload] FATAL: --require-gpu but backend is {backend!r}", file=sys.stderr)
        return 3

    # The fingerprint covers exactly the fields that determine the RESULT.
    # --step-sleep-s and --ckpt-every-s are deliberately excluded: a relaunch
    # is allowed to checkpoint more often or run flat out, and refusing that
    # would make the resume path unusable in the situation it exists for.
    cfg = {"steps": args.steps, "seed": args.seed, "size": args.size, "dtype": "float32",
           "payload": "matmul-tanh-sum-v1"}
    fp = config_fingerprint(cfg)
    print(f"[payload] config fingerprint {fp} :: {cfg}", flush=True)

    remote = f"{args.remote.rstrip('/')}/{args.job}" if args.remote else None
    cp = Checkpointer(os.path.join(args.out, "ckpt"), fp,
                      remote_uri=f"{remote}/ckpt" if remote else None,
                      every_s=args.ckpt_every_s, keep=args.keep)
    if args.fresh:
        cp.discard()

    n, dtype = args.size, jnp.float32
    try:
        found = cp.latest()
    except FingerprintMismatch as exc:
        print(f"[payload] FATAL: {exc}", file=sys.stderr)
        return 4

    if found is None:
        start_step = 0
        acc = jnp.zeros((n,), dtype=dtype)
        chain = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).digest()
        print("[payload] no checkpoint -> starting from step 0", flush=True)
    else:
        start_step, state = found
        acc = jnp.asarray(state["acc"])
        chain = bytes(state["chain"])
        print(f"[payload] RESUMED from step {start_step} "
              f"(chain {chain.hex()[:16]}...) -- {args.steps - start_step} steps remain",
              flush=True)

    step_fn = _build_step_fn(n, dtype, args.seed)
    t_start = time.time()
    resumed_from = start_step

    def write_progress(step: int, done: bool) -> None:
        """Mid-run breadcrumb. syncback.sh ships this on a timer, so the job's
        state is visible (and survives) well before it finishes."""
        payload = {
            "job": args.job, "step": step, "steps": args.steps, "done": done,
            "resumed_from": resumed_from, "fingerprint": fp,
            "chain": chain.hex(), "elapsed_s": round(time.time() - t_start, 3),
            "backend": backend, "devices": devices,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "hostname": os.uname().nodename, "n_saves": cp.n_saves,
            "preempted": bool(_STOP.is_set()), "stop_reason": list(_STOP_REASON),
            "wall_clock": time.time(),
        }
        tmp = os.path.join(args.out, ".progress.tmp")
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, os.path.join(args.out, "progress.json"))

    step = start_step
    write_progress(step, done=False)
    for step in range(start_step + 1, args.steps + 1):
        acc = step_fn(acc, step)
        acc.block_until_ready()
        chain = hashlib.sha256(chain + np.asarray(acc).tobytes()).digest()
        if args.step_sleep_s:
            time.sleep(args.step_sleep_s)

        if _STOP.is_set():
            cp.save(step, {"acc": acc, "chain": _b(chain), "step": step})
            write_progress(step, done=False)
            _sync(args.out, remote)
            print(f"[payload] checkpointed at step {step} on {_STOP_REASON} -> exit 0", flush=True)
            return 0

        if cp.maybe_save(step, {"acc": acc, "chain": _b(chain), "step": step}):
            write_progress(step, done=False)
            _sync(args.out, remote)
            print(f"[payload] step {step}/{args.steps} checkpointed "
                  f"(chain {chain.hex()[:16]}...)", flush=True)

    result = {
        "job": args.job, "config": cfg, "fingerprint": fp, "steps": args.steps,
        "chain": chain.hex(),
        "acc_sum": float(acc.sum()), "acc_first": float(acc[0]), "acc_last": float(acc[-1]),
        "resumed_from": resumed_from, "backend": backend, "devices": devices,
        # Provenance for the bit-identity question: XLA autotuning picks matmul
        # algorithms per process, so two runs of the same graph on two VMs can
        # differ in the last bits unless determinism is forced. Recording the
        # flags is the difference between "the resume is wrong" and "the GPU is
        # non-deterministic", which look identical in a diff of the results.
        "jax_version": jax.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    write_progress(args.steps, done=True)
    _sync(args.out, remote)
    print("[payload] RESULT " + json.dumps(result, sort_keys=True), flush=True)
    print(f"[payload] DONE chain={chain.hex()}", flush=True)
    return 0


def _b(chain: bytes):
    """SHA-256 digest as a uint8 array -- np.savez stores bytes exactly."""
    return np.frombuffer(chain, dtype=np.uint8)


def _sync(out: str, remote: str | None) -> None:
    """Push the output dir now. Called on every checkpoint, not only at exit:
    a pre-empted VM never reaches its last write.

    `ckpt/` is excluded. The Checkpointer already publishes each checkpoint to
    `<remote>/ckpt/` atomically at save time, and it prunes that prefix to the
    newest N; a second copy under `<remote>/out/ckpt/` is not pruned by anything
    and grows without bound over a campaign of pre-emptions. (Measured: the
    exclusion has to be `.*ckpt.*`, not `^ckpt/.*` -- gcloud matches the regex
    against the full path, so the anchored form silently matches nothing.)
    """
    if not remote:
        return
    subprocess.run(
        [os.environ.get("GCLOUD_BIN", "gcloud"), "storage", "rsync", "-r",
         "-x", ".*ckpt.*", out, f"{remote}/out"],
        check=False, capture_output=True, text=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
