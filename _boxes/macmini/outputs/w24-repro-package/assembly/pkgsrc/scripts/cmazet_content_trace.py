"""cmazet content trace."""

from __future__ import annotations
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
K = 4
SEED = 5
from nclu.paths import run_dir

PD = str(run_dir("cmaze_t", "symmetric", "test", "persist", "BOTH_seed5", "test"))


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_file(pdir, name, man):
    """.sha256 side file and MANIFEST must agree, and the file must hash to them."""
    side = open(os.path.join(pdir, name + ".sha256")).read().split()
    side_param, side_file = (side[1], side[3])
    fh = fsha(os.path.join(pdir, name))
    rec = {
        "file_sha256_computed": fh,
        "file_sha256_sidecar": side_file,
        "file_sha256_manifest": man[name]["file_sha256"],
        "param_sha256_sidecar": side_param,
        "param_sha256_manifest": man[name]["param_sha256"],
    }
    ok = fh == side_file == man[name]["file_sha256"] and side_param == man[name]["param_sha256"]
    rec["file_match"] = ok
    assert ok, (name, rec)
    return rec
