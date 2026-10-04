#!/usr/bin/env python
"""Independent verification of the w10 acceptance criterion.

The criterion: `TABLES.md` contains tables where EVERY cell traces to a file path recorded in
`TABLES.json`, and a reader can regenerate every figure from that manifest alone.

This script does not import build_tables.py.  It re-reads TABLES.json, re-opens every source by
the path recorded there, re-resolves every `provenance` pointer, and asserts the value stored in
the manifest is the value on disk.  It also re-hashes every source and every figure.

    python verify_tables.py          # exits 0 only if every check passes
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

def _default_out() -> Path:
    here = Path(__file__).resolve().parent
    return here if (here / "TABLES.json").exists() else Path("C:/chlu/outputs/w10-seeds-and-tables")


OUT = Path(os.environ.get("W10_OUT_DIR") or
           (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "")
           or _default_out())
MAN = json.loads((OUT / "TABLES.json").read_text(encoding="utf-8"))


def native(p: str) -> Path:
    m = re.match(r"^/([a-zA-Z])/(.*)$", p)
    return Path(f"{m.group(1).upper()}:/{m.group(2)}") if m else Path(p)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def ptr(obj, pointer: str):
    cur = obj
    for tok in pointer.strip("/").split("/"):
        tok = tok.replace("~1", "/").replace("~0", "~")
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


fails: list[str] = []
counts = {"sources": 0, "source_hashes_ok": 0, "cells": 0, "cell_values_checked": 0,
          "figures": 0, "figure_hashes_ok": 0, "reductions": 0}

# ---- 1. every source exists and still hashes to what the manifest recorded -----------------
DOCS = {}
for sid, meta in MAN["sources"].items():
    counts["sources"] += 1
    p = native(meta["path"])
    if not p.exists():
        fails.append(f"SOURCE MISSING  {sid}  {meta['path']}")
        continue
    h = sha256(p)
    if h != meta["sha256"]:
        fails.append(f"SOURCE HASH DRIFT  {sid}  recorded {meta['sha256'][:16]} now {h[:16]}")
    else:
        counts["source_hashes_ok"] += 1
    if meta["path"].endswith(".json"):
        DOCS[sid] = json.loads(p.read_text(encoding="utf-8"))


# ---- 2. every cell's every provenance entry re-resolves to the recorded value ---------------
def is_cell(o) -> bool:
    return isinstance(o, dict) and "values" in o and "provenance" in o


def check_cell(o, where: str):
    counts["cells"] += 1
    vals, prov = o["values"], o["provenance"]
    if len(vals) != len(prov):
        fails.append(f"CELL SHAPE  {where}: {len(vals)} values vs {len(prov)} provenance entries")
        return
    for v, pv in zip(vals, prov):
        sid = pv["source"]
        if sid not in DOCS:
            fails.append(f"CELL SOURCE NOT A JSON SOURCE  {where}  {sid}")
            continue
        if native(MAN["sources"][sid]["path"]) != native(pv["path"]):
            fails.append(f"CELL PATH MISMATCH  {where}  {pv['path']}")
            continue
        try:
            got = ptr(DOCS[sid], pv["pointer"])
        except Exception as exc:                                   # noqa: BLE001
            fails.append(f"CELL POINTER UNRESOLVABLE  {where}  {sid}{pv['pointer']}  ({exc})")
            continue
        red = pv.get("reduction")
        if red == "max over the 5 wells":
            got = max(got)
            counts["reductions"] += 1
        elif red:
            fails.append(f"CELL UNKNOWN REDUCTION  {where}  {red}")
            continue
        ok = (got == v) or (isinstance(got, float) and isinstance(v, float) and abs(got - v) <= 1e-12 * max(1.0, abs(v)))
        if not ok:
            fails.append(f"CELL VALUE MISMATCH  {where}  {sid}{pv['pointer']}  manifest {v!r} disk {got!r}")
        else:
            counts["cell_values_checked"] += 1


def walk(o, where=""):
    if is_cell(o):
        check_cell(o, where)
        return
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{where}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{where}[{i}]")


walk(MAN["tables"], "tables")

# ---- 3. every pin and every correction/not-run pointer resolves -----------------------------
def check_ptr_list(entries, where):
    for e in entries:
        if "pointer" not in e:
            continue
        sid = e.get("source")
        if sid not in DOCS:
            continue      # markdown/text sources carry locators, not pointers
        try:
            ptr(DOCS[sid], e["pointer"])
        except Exception as exc:                                   # noqa: BLE001
            fails.append(f"POINTER UNRESOLVABLE  {where}  {sid}{e['pointer']}  ({exc})")


for tid, t in MAN["tables"].items():
    check_ptr_list(t.get("pin_provenance", []), f"{tid}.pin_provenance")
    if "verdict" in t:
        check_ptr_list(t["verdict"].get("provenance", []), f"{tid}.verdict")
for c in MAN["corrections"]:
    check_ptr_list(c["provenance"], f"correction {c['id']}")
for nr in MAN["declared_not_run"]:
    check_ptr_list(nr["evidence"], f"not-run {nr['id']}")

# ---- 4. every figure exists, hashes, and names sources that are in the register --------------
for fid, f in MAN["figures"].items():
    counts["figures"] += 1
    p = OUT / f["file"]
    if not p.exists():
        fails.append(f"FIGURE MISSING  {fid}  {f['file']}")
        continue
    if sha256(p) != f["sha256"]:
        fails.append(f"FIGURE HASH DRIFT  {fid}")
    else:
        counts["figure_hashes_ok"] += 1
    for srcref in f["sources"]:
        if srcref["source"] not in MAN["sources"]:
            fails.append(f"FIGURE SOURCE NOT IN REGISTER  {fid}  {srcref['source']}")

# ---- 5. every declared not-run carries a reason AND a launch line ----------------------------
for nr in MAN["declared_not_run"]:
    for field in ("status", "why", "launch_line", "evidence"):
        if not nr.get(field):
            fails.append(f"NOT-RUN INCOMPLETE  {nr['id']}  missing {field}")

# ---- 6. TABLES.md exists and every source path it prints is in the register ------------------
md = (OUT / "TABLES.md").read_text(encoding="utf-8")
registered = {m["path"] for m in MAN["sources"].values()}
def expand_braces(pat: str):
    """`a{0,1}b{x,y}` -> every concrete spelling.  The tables print brace-globs for brevity;
    each expansion must still be a registered source."""
    m = re.search(r"\{([^{}]*)\}", pat)
    if not m:
        return [pat]
    out = []
    for alt in m.group(1).split(","):
        out += expand_braces(pat[:m.start()] + alt + pat[m.end():])
    return out


for path in sorted(set(re.findall(r"`(/c/chlu/[^`]+?)`", md))):
    for concrete in expand_braces(path):
        if concrete in registered:
            continue
        if "*" in concrete and any(r.startswith(concrete.split("*")[0]) for r in registered):
            continue
        fails.append(f"MD PATH NOT IN REGISTER  {concrete}")

# ---- report -----------------------------------------------------------------------------------
print("w10 seeds-and-tables — acceptance verification")
print(f"  sources registered .............. {counts['sources']}")
print(f"  source hashes re-verified ....... {counts['source_hashes_ok']}")
print(f"  cells found in TABLES.json ...... {counts['cells']}")
print(f"  per-seed values re-resolved ..... {counts['cell_values_checked']}  "
      f"(of which {counts['reductions']} via a declared reduction)")
print(f"  figures ......................... {counts['figures']}  "
      f"({counts['figure_hashes_ok']} hashes re-verified)")
print(f"  declared NOT-RUN rows ........... {len(MAN['declared_not_run'])}")
print(f"  corrections ..................... {len(MAN['corrections'])}")
print()
if fails:
    print(f"FAIL — {len(fails)} problem(s):")
    for f_ in fails:
        print("  " + f_)
    sys.exit(1)
print("PASS — every cell in TABLES.json re-resolves to the value on disk at the path and pointer "
      "it records, every source and figure hash matches, and every declared NOT-RUN carries a "
      "reason and a launch line.")
