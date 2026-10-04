"""Industrial-dataset loader protocol + download helpers (F3).

Every dataset exposes the same small surface so the evaluation harness
(``chlu.eval.harness``) is dataset-agnostic and new sources (e.g. FactoryWave)
can drop in later:

- ``unit_ids() / train_ids() / test_ids()``: physical **units** (a run, a
  bearing, a simulation seed) — the only legal split granularity.
- ``load_unit(uid) -> UnitRecord``: lazily load one unit's multichannel series
  with labels + metadata.
- ``label_kind``: "point" (per-timestep 0/1 labels — VUS-PR primary) or
  "episode" (one label per unit — episode AUROC primary).
- ``protocol``: "cross_unit" (fit on train units, score test units) or
  "per_unit_prefix" (TSB-AD style: each series carries its own normal training
  prefix; fit and score within the unit).

Downloads are fetch-or-point-at-path: constructors take ``root`` (defaulting
to ``$CHLU_DATA_ROOT`` or ``~/.cache/chlu/datasets/<name>``) and
``download=True`` invokes ``fetch()`` where automatic download is legal and
feasible; otherwise ``fetch()`` raises with manual instructions. Checksums are
verified whenever the upstream publishes or we have pinned them.
"""
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
def file_digest(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_file(
    url: str,
    dest: str | Path,
    sha256: str | None = None,
    md5: str | None = None,
    timeout: int = 60,
) -> Path:
    """Stream ``url`` to ``dest`` and verify checksums — **concurrency-safe**.

    Safe under N parallel processes sharing one cache (the CSF3 flagship
    launches ~6 ``--download`` jobs against the same networked home cache). The
    strategy is unique-temp + atomic-rename + check-final-first — deliberately
    **not** ``flock`` (advisory locks are unreliable on the networked FS):

    1. **Check final first.** If ``dest`` already exists and verifies, return it
       immediately (the common case once one job has published the file). No
       shared ``.part`` for a peer to read or clobber.
    2. **Download to a process-unique temp** (``mkstemp`` in ``dest``'s own
       directory, so it is on the same filesystem), verify the checksum on that
       temp, then **atomically** ``os.replace(temp, dest)``. Concurrent
       processes each write their own temp; the renames are last-writer-wins,
       but every temp is an identical *verified* file, so ``dest`` is always
       valid. The process's own temp is removed on any failure.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _verify(p: Path) -> None:
        if sha256 and file_digest(p, "sha256") != sha256:
            raise IOError(f"sha256 mismatch for {p}")
        if md5 and file_digest(p, "md5") != md5:
            raise IOError(f"md5 mismatch for {p}")

    # 1. check-final-first: use an already-published, verified file as-is.
    if dest.exists():
        try:
            _verify(dest)
            return dest
        except IOError:
            # A corrupt/legacy final (or a mid-migration leftover): fall through
            # and re-download to a fresh temp, then atomically overwrite it.
            pass

    # 2. download to a process-unique temp on the SAME filesystem as dest, so
    #    os.replace() below is an atomic same-fs rename.
    fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=dest.name + ".", suffix=f".{os.getpid()}.part"
    )
    tmp = Path(tmp_name)
    try:
        print(f"downloading {url} -> {dest}")
        with urllib.request.urlopen(url, timeout=timeout) as resp, os.fdopen(
            fd, "wb"
        ) as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        _verify(tmp)
        # atomic publish; last-writer-wins is safe (every temp is identical).
        os.replace(tmp, dest)
    except BaseException:
        # clean up only our own temp; never touch a peer's temp or the final.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return dest


