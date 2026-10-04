"""Concurrency safety for ``download_file`` (CSF3 parallel-``--download`` race).

Reproduces the failure that killed 5/6 of the first CSF3 flagship launch: N
parallel processes calling ``download_file`` on the same URL into one shared
cache used to write a *shared* ``…​.part`` and clobber each other -> ``sha256
mismatch`` / ``FileNotFoundError``. The fix (unique-temp + atomic-rename +
check-final-first) must let all N succeed.

Pure stdlib, no real network: a throttled localhost HTTP server serves a small
fixture slowly enough that the workers genuinely overlap mid-download (that
overlap is exactly what triggered the shared-``.part`` race on the old code).
"""

import hashlib
import http.server
import multiprocessing as mp
import threading
import time
from pathlib import Path

import pytest

# Payload big enough (with the per-chunk sleep below) that 4 workers started
# together are all mid-download at once — the condition that races the cache.
_PAYLOAD = (b"voraus-fixture-block-0123456789abcdef" * 4096)  # ~152 KB
_SHA256 = hashlib.sha256(_PAYLOAD).hexdigest()


class _SlowHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (stdlib API)
        self.send_response(200)
        self.send_header("Content-Length", str(len(_PAYLOAD)))
        self.end_headers()
        chunk = 16 * 1024
        for i in range(0, len(_PAYLOAD), chunk):
            self.wfile.write(_PAYLOAD[i : i + chunk])
            self.wfile.flush()
            time.sleep(0.02)  # throttle -> guarantees inter-process overlap

    def log_message(self, *args):  # silence the test log
        pass


def _serve():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _worker(args):
    """Top-level (picklable for the 'spawn' start method) download worker."""
    url, dest, sha = args
    from chlu.data.industrial.base import download_file

    try:
        p = download_file(url, dest, sha256=sha)
        # verify the final we ended up with is the real payload
        got = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        return "ok" if got == sha else f"bad-final-digest {got}"
    except Exception as exc:  # noqa: BLE001 — report, don't crash the pool
        return f"{type(exc).__name__}: {exc}"


def test_download_file_concurrent_shared_cache(tmp_path):
    """>=4 processes downloading the same URL into one shared cache all succeed.

    On the pre-fix code (shared ``…​.part``) this fails with a mix of
    ``sha256 mismatch`` / ``FileNotFoundError`` from the clobbered partial.
    """
    server, _ = _serve()
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/voraus-fixture.parquet"
    dest = tmp_path / "cache" / "voraus-fixture.parquet"  # one SHARED final path

    n = 4
    try:
        ctx = mp.get_context("spawn")  # match macOS default; fully isolated procs
        with ctx.Pool(processes=n) as pool:
            results = pool.map(_worker, [(url, str(dest), _SHA256)] * n)
    finally:
        server.shutdown()

    assert results == ["ok"] * n, results
    assert dest.exists()
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == _SHA256
    # no shared/leftover temp or ``.part`` files remain in the cache dir
    leftovers = [p.name for p in dest.parent.iterdir() if p.name != dest.name]
    assert leftovers == [], f"stray temp files left behind: {leftovers}"


def test_download_file_short_circuits_verified_final(tmp_path):
    """If a verified final already exists, no download is attempted."""
    from chlu.data.industrial.base import download_file

    dest = tmp_path / "cache" / "voraus-fixture.parquet"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(_PAYLOAD)
    # An unreachable URL: if the short-circuit works we never touch the network.
    out = download_file("http://127.0.0.1:1/never", dest, sha256=_SHA256)
    assert out == dest


if __name__ == "__main__":  # allow: python tests/test_download_concurrency.py
    raise SystemExit(pytest.main([__file__, "-q"]))
