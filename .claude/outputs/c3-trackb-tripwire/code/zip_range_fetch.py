"""Fetch a single member of a remote ZIP over HTTP range requests.

Used to pull N-CMAPSS_DS02-006.h5 out of the 15.76 GB NASA PCoE archive
'17. Turbofan Engine Degradation Simulation Data Set 2.zip' without
downloading the whole thing.  Read-only, no repo code touched.
"""
import io
import struct
import subprocess
import sys
import zlib

URL = ("https://phm-datasets.s3.amazonaws.com/NASA/"
       "17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip")


def rng(url, start, end):
    """Inclusive byte range -> bytes."""
    out = subprocess.run(
        ["curl", "-sS", "--retry", "5", "--retry-delay", "3",
         "-r", f"{start}-{end}", url],
        capture_output=True, check=True)
    return out.stdout


def total_size(url):
    out = subprocess.run(["curl", "-sSI", url], capture_output=True, check=True)
    for line in out.stdout.decode("latin1").splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1])
    raise RuntimeError("no content-length")


def find_eocd(url, size):
    tail = rng(url, size - 200000, size - 1)
    # ZIP64 EOCD locator
    i = tail.rfind(b"PK\x06\x07")
    if i < 0:
        raise RuntimeError("no zip64 eocd locator")
    (_, cd64_off, _) = struct.unpack("<IQI", tail[i + 4:i + 20])
    z64 = rng(url, cd64_off, cd64_off + 55)
    assert z64[:4] == b"PK\x06\x06", z64[:4]
    (nent, nent_tot, cdsize, cdoff) = struct.unpack("<QQQQ", z64[24:56])
    return nent_tot, cdsize, cdoff


def central_dir(url, cdoff, cdsize):
    data = rng(url, cdoff, cdoff + cdsize - 1)
    entries = []
    p = 0
    while p < len(data) and data[p:p + 4] == b"PK\x01\x02":
        (method, ) = struct.unpack("<H", data[p + 10:p + 12])
        csize, usize = struct.unpack("<II", data[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", data[p + 28:p + 34])
        lho, = struct.unpack("<I", data[p + 42:p + 46])
        name = data[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = data[p + 46 + nlen:p + 46 + nlen + elen]
        # zip64 extra field
        if 0xFFFFFFFF in (csize, usize, lho):
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    body = extra[q + 4:q + 4 + hsz]
                    r = 0
                    if usize == 0xFFFFFFFF:
                        usize, = struct.unpack("<Q", body[r:r + 8]); r += 8
                    if csize == 0xFFFFFFFF:
                        csize, = struct.unpack("<Q", body[r:r + 8]); r += 8
                    if lho == 0xFFFFFFFF:
                        lho, = struct.unpack("<Q", body[r:r + 8]); r += 8
                    break
                q += 4 + hsz
        entries.append(dict(name=name, method=method, csize=csize,
                            usize=usize, lho=lho))
        p += 46 + nlen + elen + clen
    return entries


def fetch_member(url, ent, dest):
    hdr = rng(url, ent["lho"], ent["lho"] + 29)
    assert hdr[:4] == b"PK\x03\x04", hdr[:4]
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    start = ent["lho"] + 30 + nlen + elen
    end = start + ent["csize"] - 1
    print(f"fetching {ent['name']} bytes {start}-{end} "
          f"({ent['csize']/1e6:.1f} MB compressed, method {ent['method']})",
          flush=True)
    # stream in 32 MB chunks, inflate on the fly
    dec = zlib.decompressobj(-15) if ent["method"] == 8 else None
    CH = 32 << 20
    with open(dest, "wb") as fh:
        pos = start
        while pos <= end:
            hi = min(pos + CH - 1, end)
            blob = rng(url, pos, hi)
            if len(blob) == 0:
                raise RuntimeError("empty range response")
            fh.write(dec.decompress(blob) if dec else blob)
            pos += len(blob)
            print(f"  {(pos-start)/1e6:8.1f} / {ent['csize']/1e6:.1f} MB",
                  flush=True)
        if dec:
            fh.write(dec.flush())
    return dest


if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else "DS02"
    dest = sys.argv[2] if len(sys.argv) > 2 else None
    size = total_size(URL)
    print("archive bytes:", size)
    nent, cdsize, cdoff = find_eocd(URL, size)
    print("entries:", nent, "cd at", cdoff, "size", cdsize)
    ents = central_dir(URL, cdoff, cdsize)
    for e in ents:
        print(f"  {e['usize']:>14,}  {e['csize']:>14,}  m{e['method']}  {e['name']}")
    if dest:
        hit = [e for e in ents if want in e["name"] and e["usize"] > 0]
        assert len(hit) == 1, hit
        fetch_member(URL, hit[0], dest)
