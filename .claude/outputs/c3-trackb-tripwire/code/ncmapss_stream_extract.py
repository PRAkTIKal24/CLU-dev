"""Stream the NASA PCoE archive '17. Turbofan ... Data Set 2.zip' and pull ONE
member out of the *inner* data_set.zip without ever storing the 15.76 GB blob.

Outer zip: 2 entries, the payload being 'data_set.zip' stored with deflate
(method 8) at offset LHO.  We range-request the outer compressed bytes, inflate
them on the fly, and parse the inner zip's local file headers sequentially,
writing only the wanted member and stopping as soon as it is complete.

CPU-only, network-only, writes nothing into the repo.
"""
import hashlib
import struct
import subprocess
import sys
import zlib

URL = ("https://phm-datasets.s3.amazonaws.com/NASA/"
       "17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip")
OUTER_LHO = None  # discovered
WANT = sys.argv[1] if len(sys.argv) > 1 else "DS02"
DEST = sys.argv[2] if len(sys.argv) > 2 else "/tmp/N-CMAPSS_DS02-006.h5"


def sh(args):
    return subprocess.run(args, capture_output=True, check=True).stdout


def rng(start, end):
    return sh(["curl", "-sS", "--retry", "5", "--retry-delay", "3",
               "-r", f"{start}-{end}", URL])


def outer_payload_range():
    # from zip_range_fetch.py: single payload entry, method 8
    import zip_range_fetch as z
    size = z.total_size(URL)
    _, cdsize, cdoff = z.find_eocd(URL, size)
    ents = z.central_dir(URL, cdoff, cdsize)
    e = [x for x in ents if x["usize"] > 0][0]
    hdr = rng(e["lho"], e["lho"] + 29)
    assert hdr[:4] == b"PK\x03\x04"
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    start = e["lho"] + 30 + nlen + elen
    return start, start + e["csize"] - 1, e


class Inflater:
    """Yields inflated bytes of the outer member, chunk by chunk."""

    def __init__(self, start, end, chunk=16 << 20):
        self.pos, self.end, self.chunk = start, end, chunk
        self.dec = zlib.decompressobj(-15)
        self.raw_read = 0

    def __iter__(self):
        while self.pos <= self.end:
            hi = min(self.pos + self.chunk - 1, self.end)
            blob = rng(self.pos, hi)
            if not blob:
                raise RuntimeError("empty range response")
            self.pos += len(blob)
            self.raw_read += len(blob)
            out = self.dec.decompress(blob)
            if out:
                yield out


def main():
    start, end, ent = outer_payload_range()
    print(f"outer payload: {ent['name']} {ent['csize']:,}B compressed "
          f"-> {ent['usize']:,}B", flush=True)
    buf = bytearray()
    it = iter(Inflater(start, end))
    state = "hdr"
    fh = None
    remaining = 0
    h = None
    total_seen = 0
    inner_dec = None

    def refill(n):
        nonlocal buf
        while len(buf) < n:
            try:
                buf += next(it)
            except StopIteration:
                return False
        return True

    while True:
        if state == "hdr":
            if not refill(30):
                break
            sig = bytes(buf[:4])
            if sig == b"PK\x01\x02" or sig == b"PK\x06\x06":
                print("reached central directory; done", flush=True)
                break
            if sig != b"PK\x03\x04":
                raise RuntimeError(f"unexpected sig {sig!r} at {total_seen}")
            flags, method = struct.unpack("<HH", buf[6:10])
            csize, usize = struct.unpack("<II", buf[18:26])
            nlen, elen = struct.unpack("<HH", buf[26:30])
            if not refill(30 + nlen + elen):
                break
            name = bytes(buf[30:30 + nlen]).decode("utf-8", "replace")
            extra = bytes(buf[30 + nlen:30 + nlen + elen])
            if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:
                q = 0
                while q + 4 <= len(extra):
                    hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                    if hid == 1:
                        body = extra[q + 4:q + 4 + hsz]
                        r = 0
                        if usize == 0xFFFFFFFF:
                            usize, = struct.unpack("<Q", body[r:r + 8]); r += 8
                        if csize == 0xFFFFFFFF:
                            csize, = struct.unpack("<Q", body[r:r + 8]); r += 8
                        break
                    q += 4 + hsz
            del buf[:30 + nlen + elen]
            print(f"member {name}  method={method} flags={flags} "
                  f"csize={csize:,} usize={usize:,}", flush=True)
            if flags & 0x8:
                raise RuntimeError("data descriptor (streamed) zip: sizes "
                                   "unknown in local header; cannot skip")
            remaining = csize
            want = WANT in name and usize > 0
            inner_dec = None
            if want:
                fh = open(DEST, "wb")
                h = hashlib.sha256()
                if method == 8:
                    inner_dec = zlib.decompressobj(-15)
                elif method != 0:
                    raise RuntimeError(f"member method {method} unsupported")
            state = "data"
        elif state == "data":
            if remaining == 0:
                state = "hdr"
                if fh:
                    if inner_dec is not None:
                        tail = inner_dec.flush()
                        if tail:
                            fh.write(tail)
                            h.update(tail)
                    fh.close()
                    print(f"WROTE {DEST} sha256={h.hexdigest()}", flush=True)
                    return 0
                continue
            if not buf:
                try:
                    buf += next(it)
                except StopIteration:
                    break
            take = min(remaining, len(buf))
            if fh:
                chunk = bytes(buf[:take])
                if inner_dec is not None:
                    chunk = inner_dec.decompress(chunk)
                fh.write(chunk)
                h.update(chunk)
            del buf[:take]
            remaining -= take
            total_seen += take
    return 1


if __name__ == "__main__":
    sys.exit(main())
