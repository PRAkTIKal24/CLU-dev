"""base."""

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
    url: str, dest: str | Path, sha256: str | None = None, md5: str | None = None, timeout: int = 60
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _verify(p: Path) -> None:
        if sha256 and file_digest(p, "sha256") != sha256:
            raise IOError(f"sha256 mismatch for {p}")
        if md5 and file_digest(p, "md5") != md5:
            raise IOError(f"md5 mismatch for {p}")

    if dest.exists():
        try:
            _verify(dest)
            return dest
        except IOError:
            pass
    fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=dest.name + ".", suffix=f".{os.getpid()}.part"
    )
    tmp = Path(tmp_name)
    try:
        print(f"downloading {url} -> {dest}")
        with urllib.request.urlopen(url, timeout=timeout) as resp, os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        _verify(tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return dest
