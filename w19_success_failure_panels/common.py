"""Shared plumbing: redirect the w17 package's side-effect paths into THIS spoke's dirs."""
from __future__ import annotations

import os
from pathlib import Path

SPOKE_OUT = Path(os.environ.get("W19_SF_OUT", "/Users/batmac/clu/outputs/w19-success-failure-panels"))
FIG_OUT = Path(os.environ.get("W19_SF_FIG", "/Users/batmac/clu/outputs/w17-paper/figures/rendered"))


def w17():
    """Import the w17 render module with its log / comparator-assembly paths pointed here,
    so nothing is written into the w17 spoke's output directory."""
    from w17_rendered_compositions import render as RR
    RR.SPOKE_OUT = SPOKE_OUT
    RR.ASSEMBLED = SPOKE_OUT / "_assembled_comparators"
    return RR
