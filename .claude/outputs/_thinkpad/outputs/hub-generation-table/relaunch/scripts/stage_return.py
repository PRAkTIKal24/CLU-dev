"""Stage the RESULTS of hub/generation-table-2 for --return-pack, deleting nothing.

LAUNCH-COMMON 5.1 is absolute: YOU NEVER DELETE.  But `--return-pack hub-generation-table`
on the real outputs directory would pack ~3.6 GB, of which >99 % is regenerable
channel-geometry / feature cache (iteration 1's `dev/cache/**` and my `relaunch/cache/**`).

So: COPY the results into a staging tree inside my own write prefix and pack THAT,
leaving every original byte in place on FORGIS_14.  Everything left behind is listed
with its size and sha256 in RETURN-EXCLUSIONS-2.json, together with how to regenerate it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

OUT = Path(os.environ["CHLU_HUB_OUTPUTS"].replace("/c/", "C:/", 1))
SLUG = "hub-generation-table"
SRC = OUT / SLUG
STAGE = SRC / "return_stage_2"
DST = STAGE / ".claude" / "outputs" / SLUG

#: directories whose contents are regenerable and are NOT returned
EXCLUDE_DIRS = {
    "dev/cache": "iteration 1's per-corner channel geometry, PQ codebooks, Jacobian "
                 "probe samples and gathered pool features. Regenerable by "
                 "`dev/fast.py::channel_geometry` in ~4 s per corner.",
    "relaunch/cache": "MY per-corner channel geometry (correlation + ward/average "
                      "dendrogram leaf orders). Regenerable by "
                      "`relaunch/dev/fast.py::channel_geometry` in ~4.1 s per corner.",
    "return_stage": "iteration 1's own staging copy - already returned once.",
    "return_stage_2": "this staging tree itself.",
}
EXCLUDE_SUFFIX = (".pyc",)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(rel: str):
    r = rel.replace("\\", "/")
    for d, why in EXCLUDE_DIRS.items():
        if r == d or r.startswith(d + "/"):
            return why
    if r.endswith(EXCLUDE_SUFFIX) or "__pycache__/" in r:
        return "a Python bytecode cache; not an artefact."
    if ".return" in r and r.endswith(".tar.gz"):
        # a return TARBALL is packaging, not a result - and packing one inside another
        # is how a 15 MB pack becomes a 3.8 GB one. The git bundles ARE returned.
        return ("a return tarball (packaging, not an artefact). It is still on the box "
                "at its original path.")
    return None


def main():
    if DST.exists():
        shutil.rmtree(DST)          # only ever MY staging copy, never a source byte
    DST.mkdir(parents=True, exist_ok=True)
    kept, left = [], []
    for p in sorted(SRC.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(SRC))
        why = excluded(rel)
        if why:
            left.append({"path": f"{SLUG}/{rel.replace(os.sep, '/')}",
                         "bytes": p.stat().st_size, "why": why})
            continue
        tgt = DST / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, tgt)
        kept.append({"path": f"{SLUG}/{rel.replace(os.sep, '/')}",
                     "bytes": p.stat().st_size})
    # the one-page report lives one level up; carry a copy INSIDE the slug so the
    # return pack (which only takes `<slug>/**` and `<slug>.md`) brings it home.
    rep = OUT / "hub-generation-table-2.md"
    if rep.exists():
        shutil.copy2(rep, DST / "REPORT-hub-generation-table-2.md")
        kept.append({"path": f"{SLUG}/REPORT-hub-generation-table-2.md",
                     "bytes": rep.stat().st_size})

    # sha the big excluded files so the far side can verify a later re-send
    big = sorted(left, key=lambda r: -r["bytes"])[:40]
    for r in big:
        p = SRC / r["path"][len(SLUG) + 1:]
        if p.exists():
            r["sha256"] = sha256(p)

    man = {
        "what": "hub/generation-table-2 (RELAUNCH) return staging",
        "produced_by": "FORGIS_14, Windows 11 AMD64",
        "rule": "LAUNCH-COMMON 5.1 - NOTHING IS DELETED. Every excluded file is still "
                "on the box at its original path; this is a COPY-based staging tree.",
        "n_returned": len(kept), "bytes_returned": sum(r["bytes"] for r in kept),
        "n_left_in_place": len(left), "bytes_left_in_place": sum(r["bytes"] for r in left),
        "excluded_directories": EXCLUDE_DIRS,
        "left_in_place_largest_40": big,
        "returned": kept,
    }
    (OUT / SLUG / "RETURN-EXCLUSIONS-2.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8")
    print(f"staged {len(kept)} files / {man['bytes_returned']:,} B -> {DST}")
    print(f"left in place {len(left)} files / {man['bytes_left_in_place']:,} B "
          f"(listed in RETURN-EXCLUSIONS-2.json)")


if __name__ == "__main__":
    main()
