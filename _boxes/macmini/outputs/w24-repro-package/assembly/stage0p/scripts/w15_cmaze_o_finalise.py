"""Finalise the CMAZE-O spoke: TABLES.md, HARNESS-READY.json, and a console summary.

Usage:  python scripts/w15_cmaze_o_finalise.py <out_dir> <repo_root>
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chlu.exp_w15_cmaze_o_readiness import build                     # noqa: E402


def main(argv):
    out_dir, repo_root = argv[1], argv[2]
    smoke_path = os.path.join(out_dir, "valsmoke", "VAL-SMOKE.json")
    span_path = os.path.join(out_dir, "valsmoke", "SPAN-SWEEP.json")

    # 1. tables
    rc = subprocess.call([sys.executable,
                          os.path.join(repo_root, "scripts",
                                       "w15_cmaze_o_report.py"), out_dir])
    print("report rc=%d" % rc)

    # 2. readiness
    ready = build(os.path.join(out_dir, "valsmoke"), smoke_path, repo_root,
                  span_path=span_path)
    dest = os.path.join(out_dir, "HARNESS-READY.json")
    with io.open(dest, "w", encoding="utf-8") as f:
        json.dump(ready, f, indent=1, default=str)
    print("wrote %s  status=%s" % (dest, ready["status"]))
    for k, v in ready["ACCEPTANCE"].items():
        print("  %-62s %s" % (k, v["pass"]))
    for w in ready["instrument_warnings"]:
        print("  WARNING: %s %s" % (w["warning"],
                                    {k: v for k, v in w.items()
                                     if k not in ("warning", "means")}))
    print("  headline:", json.dumps(ready["headline"], default=str)[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
