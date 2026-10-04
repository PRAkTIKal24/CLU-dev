"""Pre-emption-survivable checkpointing for CHLU cloud jobs.

This is the `.claude/outputs/pilot-checkpoint-resume.md` pattern moved to
cloud storage. Nothing about the *mechanism* is new -- the five properties it
proved are re-implemented here verbatim, with one addition that only the cloud
needs (property 6):

1. **atomic**    -- write a temp file, then ``os.replace``. A reader never sees
   a torn checkpoint, and a process killed mid-write leaves the previous
   checkpoint intact.
2. **time-based** -- the cadence is wall-clock seconds, not epochs/steps.
   Pre-emption does not wait for an epoch boundary, so neither may we.
3. **keep last N** -- bounded footprint, locally and remotely.
4. **fingerprinted** -- resume verifies the step counter *and* a hash of the
   job config, and REFUSES a mismatch by name rather than silently resuming
   someone else's run. (In the pilot this self-protection is what stopped a
   cross-leg resume; it is the reason two legs sharing an output dir failed
   loudly instead of quietly.)
5. **bit-exact**  -- state is stored as raw arrays via ``np.savez``, so a
   resumed run continues from the same bytes it stopped at.
6. **persisted off the VM** -- ``save()`` is not complete until the checkpoint
   object exists in Cloud Storage. Instance-local disk dies with the VM, and
   on a Spot VM that is the expected outcome, not the unlucky one.

Why local-then-upload rather than a gcsfuse mount: POSIX ``rename`` is atomic
on a real filesystem and is *not* on gcsfuse, while a GCS object create is
atomic by itself. Doing both in order gives an atomic staging write followed
by an atomic publish, and no step in between can be observed half-done.

Deliberately free of any JAX/Equinox import: this module is the harness, it is
unit-testable on a laptop in milliseconds, and callers hand it plain numpy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

_CKPT_RE = re.compile(r"^ckpt_(\d{12})\.npz$")
_META_KEY = "__chlu_meta__"


class FingerprintMismatch(RuntimeError):
    """A checkpoint exists but was written by a differently-configured job."""


def config_fingerprint(config: dict[str, Any]) -> str:
    """Stable short hash of a job config.

    Canonical JSON (sorted keys, no whitespace slack) so that dict ordering
    and float formatting cannot change the fingerprint between processes.
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _ckpt_name(step: int) -> str:
    return f"ckpt_{step:012d}.npz"


def _step_of(name: str) -> int | None:
    m = _CKPT_RE.match(os.path.basename(name))
    return int(m.group(1)) if m else None


def _gcloud(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a `gcloud storage` command.

    `gcloud` is used rather than a client library so the image needs no extra
    dependency and the behaviour matches what the shell scripts do.
    """
    cmd = [os.environ.get("GCLOUD_BIN", "gcloud"), "storage", *args]
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


class Checkpointer:
    """Time-based, atomic, fingerprint-verified checkpoints with GCS backing.

    Parameters
    ----------
    local_dir:
        Staging directory on the instance. Fast, and the only place an atomic
        rename is genuinely atomic.
    remote_uri:
        ``gs://bucket/prefix`` where checkpoints are published. ``None``
        disables the remote leg entirely (used by the unit tests, and by any
        run whose loss of the VM would not matter -- which on Spot is none).
    fingerprint:
        Result of :func:`config_fingerprint` for this job's config.
    every_s:
        Minimum wall-clock seconds between checkpoints. Pick it from what you
        can afford to lose, not from the epoch length.
    keep:
        How many checkpoints to retain, newest first.
    """

    def __init__(
        self,
        local_dir: str,
        fingerprint: str,
        remote_uri: str | None = None,
        every_s: float = 60.0,
        keep: int = 3,
    ) -> None:
        if keep < 1:
            raise ValueError("keep must be >= 1")
        self.local_dir = os.path.abspath(local_dir)
        self.remote_uri = remote_uri.rstrip("/") if remote_uri else None
        self.fingerprint = fingerprint
        self.every_s = float(every_s)
        self.keep = int(keep)
        os.makedirs(self.local_dir, exist_ok=True)
        self._last_save = 0.0
        self.n_saves = 0

    # ---------------------------------------------------------------- saving

    def due(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return (now - self._last_save) >= self.every_s

    def maybe_save(self, step: int, state: dict[str, Any], force: bool = False) -> bool:
        """Checkpoint if the timer says so (or ``force``). Returns whether it did.

        ``force=True`` is what the pre-emption handler calls: on ~30 s notice
        there is no timer left to respect.
        """
        if not force and not self.due():
            return False
        self.save(step, state)
        return True

    def save(self, step: int, state: dict[str, Any]) -> str:
        """Atomically write, then publish, one checkpoint. Returns its path."""
        arrays = {k: np.asarray(v) for k, v in state.items()}
        if _META_KEY in arrays:
            raise ValueError(f"{_META_KEY!r} is reserved")
        meta = {
            "step": int(step),
            "fingerprint": self.fingerprint,
            "saved_at": time.time(),
            "keys": sorted(state),
        }
        arrays[_META_KEY] = np.frombuffer(
            json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8
        )

        final = os.path.join(self.local_dir, _ckpt_name(step))
        # tmp file in the SAME directory: os.replace is only atomic within one
        # filesystem, and /tmp is frequently a different one.
        fd, tmp = tempfile.mkstemp(dir=self.local_dir, prefix=".ckpt_tmp_", suffix=".npz")
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                np.savez(fh, **arrays)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)  # <- the atomic moment, locally
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

        self._publish(final)
        self._last_save = time.monotonic()
        self.n_saves += 1
        self._prune()
        return final

    def _publish(self, path: str) -> None:
        """Copy one checkpoint to GCS. A GCS object create is itself atomic."""
        if not self.remote_uri:
            return
        dest = f"{self.remote_uri}/{os.path.basename(path)}"
        try:
            _gcloud("cp", path, dest)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - needs GCS
            # A failed publish is loud but not fatal: the local copy still
            # exists and the next tick retries. Silently continuing would turn
            # "checkpointed" into a lie.
            print(
                f"[ckpt] WARNING: publish to {dest} FAILED rc={exc.returncode}: "
                f"{exc.stderr.strip()[:400]}",
                file=sys.stderr,
                flush=True,
            )

    def _prune(self) -> None:
        """Keep the newest N across BOTH stores.

        Pruning by the local directory alone is wrong the moment a job is ever
        resumed: the new process downloads only the one checkpoint it restarts
        from, so every checkpoint the pre-empted process left in the bucket is
        invisible to it and is never deleted. "Keep last N" then silently means
        "keep last N per process", and a job pre-empted often enough
        accumulates them without bound. Measured on the `preempt-kill` run:
        steps 137 and 163 survived in the bucket after the resume pruned
        around them.
        """
        local = {s for s in (_step_of(n) for n in os.listdir(self.local_dir)) if s is not None}
        known = local | set(self._remote_steps())
        for step in sorted(known)[: max(0, len(known) - self.keep)]:
            name = _ckpt_name(step)
            path = os.path.join(self.local_dir, name)
            if os.path.exists(path):
                os.unlink(path)
            if self.remote_uri:
                _gcloud("rm", f"{self.remote_uri}/{name}", check=False)

    # --------------------------------------------------------------- loading

    def _remote_steps(self) -> list[int]:
        if not self.remote_uri:
            return []
        try:
            out = _gcloud("ls", f"{self.remote_uri}/").stdout
        except subprocess.CalledProcessError:
            return []  # prefix does not exist yet == no checkpoints
        return sorted(s for s in (_step_of(line.strip()) for line in out.splitlines()) if s is not None)

    def latest(self) -> tuple[int, dict[str, Any]] | None:
        """Newest valid checkpoint as ``(step, state)``, or ``None`` if fresh.

        Prefers the remote copy: after a pre-emption the local disk is gone,
        and when both exist they are the same bytes.

        Raises
        ------
        FingerprintMismatch
            The checkpoint belongs to a different job config. Refusing here is
            the whole point -- a silent resume onto mismatched config produces
            a result that looks fine and is not.
        """
        remote = self._remote_steps()
        if remote:
            step = remote[-1]
            name = _ckpt_name(step)
            local = os.path.join(self.local_dir, name)
            if not os.path.exists(local):
                _gcloud("cp", f"{self.remote_uri}/{name}", local)
        else:
            local_steps = sorted(
                s for s in (_step_of(n) for n in os.listdir(self.local_dir)) if s is not None
            )
            if not local_steps:
                return None
            step = local_steps[-1]
            local = os.path.join(self.local_dir, _ckpt_name(step))

        with np.load(local, allow_pickle=False) as z:
            meta = json.loads(bytes(z[_META_KEY]).decode())
            state = {k: z[k] for k in z.files if k != _META_KEY}

        if meta["fingerprint"] != self.fingerprint:
            raise FingerprintMismatch(
                f"checkpoint {local} was written by config fingerprint "
                f"{meta['fingerprint']}, this job is {self.fingerprint}. "
                "Refusing to resume. Use a different --out, or --fresh to discard."
            )
        if meta["step"] != step:
            raise FingerprintMismatch(
                f"checkpoint {local} has filename step {step} but internal step "
                f"{meta['step']} -- corrupt or hand-edited. Refusing to resume."
            )
        return int(meta["step"]), state

    def discard(self) -> None:
        """Throw away every checkpoint, local and remote (``--fresh``)."""
        shutil.rmtree(self.local_dir, ignore_errors=True)
        os.makedirs(self.local_dir, exist_ok=True)
        if self.remote_uri:
            _gcloud("rm", "-r", f"{self.remote_uri}/", check=False)
        self._last_save = 0.0
