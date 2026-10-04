"""Frozen configuration for the matched-byte tripwire.

Every constant here is either (a) read verbatim from the HEPA clone
(``.claude/hepa-src``, commit recorded in the output JSON) or (b) a tripwire
design choice declared once, here, and never inlined in an experiment body.

Nothing in this module imports HEPA or torch; it is pure declaration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Paths (overridable by env for wave 2)
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]          # /Users/user/Desktop/CHLU
HEPA_SRC = Path(os.environ.get("HEPA_SRC", REPO_ROOT / ".claude" / "hepa-src"))
HEPA_DATA_DIR = Path(os.environ.get("HEPA_DATA_DIR", REPO_ROOT / ".claude" / "data" / "hepa"))
OUT_DIR = Path(
    os.environ.get("TRIPWIRE_OUT", REPO_ROOT / ".claude" / "outputs" / "hepa-longhorizon-triage")
)

# --------------------------------------------------------------------------
# HEPA architecture constants -- source: hepa/utils/config.py::PROTOCOL and
# hepa/model/encoder.py (CausalEncoder). Used ONLY for the byte ledger.
# --------------------------------------------------------------------------
PATCH_SIZE: int = 16          # PROTOCOL["patch_size"]
D_MODEL: int = 256            # PROTOCOL["d_model"]
N_LAYERS: int = 2             # PROTOCOL["n_layers"]  (_TransformerBlock count)
DTYPE_BYTES: int = 4          # torch default dtype float32; encoder.py declares no dtype

# Which byte-ledger reading sizes the exemplar store. All three are computed and
# reported at every rung; PRIMARY is the one the kill rule uses unless the Hub
# rules otherwise after reviewing the ledger.
BUDGET_VARIANTS: List[str] = ["token_only", "residual_stream", "kv_cache"]
PRIMARY_BUDGET: str = "residual_stream"

# --------------------------------------------------------------------------
# HEPA evaluation-protocol constants -- source: hepa/training/finetune.py
# (EventDataset defaults) and scripts/train.py::_build_event_loader.
# --------------------------------------------------------------------------
MIN_CONTEXT: int = 128        # EventDataset(min_context=128)
STRIDE_TRAIN: int = 4         # _build_event_loader(..., stride=4) for ft_train/ft_val
STRIDE_TEST: int = 1          # _build_event_loader(..., stride=1) for ft_test
MIN_ENTITY_LEN: int = 129     # `if len(e["test"]) <= 128 + 1: continue`

# --------------------------------------------------------------------------
# Tripwire design choices
# --------------------------------------------------------------------------
SEEDS: List[int] = [42, 123, 456]      # PREREG §1
DEFAULT_CONTEXT: int = 512             # PREREG §3 reference rung

# kNN neighbour counts searched on the finetune VAL split (never on test).
K_NEIGHBOURS_GRID: List[int] = [1, 4, 16, 64]

# Window descriptors. dim is a callable of (n_channels, n_tokens, context_len).
# `patch_means` and `raw_window` grow with the context; `chan_stats` and
# `last_patch` do not -- that asymmetry is deliberate and reported.
DESCRIPTORS: List[str] = ["chan_stats", "last_patch", "patch_means", "raw_window"]

# Per-exemplar stored payload beyond the feature vector: the scalar
# time-to-event, in native units. One float. This is what keeps the store
# PREREG-§2.4-compliant (scalar-concat horizon conditioning): the store holds NO
# per-horizon table, and dt enters only through the comparison tte <= dt.
STORE_PAYLOAD_FLOATS: int = 1

# Non-compliant diagnostic arm: store the full K-vector label surface per
# exemplar instead of the scalar tte. Reported, but flagged, never used by the
# kill rule (it is a per-horizon free-parameter table, i.e. the mechanism
# PREREG §2.4 bans).
RUN_SURFACE_STORE_DIAGNOSTIC: bool = True

# --------------------------------------------------------------------------
# `seasonal` baseline: the period, in native steps, of the dataset's own rate.
# None => the baseline is NOT_APPLICABLE for that dataset and is reported as
# such with the reason, never as a number.
# Native rates from hepa-benchmark-scout.md §3 (paper Table 3).
# --------------------------------------------------------------------------
SEASONAL_PERIOD: Dict[str, Optional[int]] = {
    # 1 sample per flight cycle; run-to-failure, no calendar time. No period.
    "FD001": None, "FD002": None, "FD003": None, "FD004": None,
    "PSM": 1440,        # 1/min  -> daily
    "GECCO": 1440,      # 1/min  -> daily
    "BATADAL": 24,      # 1/hour -> daily
    "ETTm1": 96,        # 15 min -> daily
    "Weather": 144,     # 10 min -> daily
    "BeijingAQ": 24,    # 1/hour -> daily
    "MBA": None,        # 275 Hz ECG; the cardiac cycle is not a fixed period
}
SEASONAL_PERIOD_REASON: Dict[str, str] = {
    "cmapss": "native rate is 1 sample per flight cycle on a run-to-failure "
              "stream; there is no calendar or diurnal period to condition on",
    "MBA": "275 Hz single ECG recording; the only quasi-period is the cardiac "
           "cycle, which is not a fixed number of steps",
}

N_SEASONAL_BINS: int = 24     # phase bins within one period


def dataset_key(name: str) -> str:
    """Normalise a dataset name to the key used by SEASONAL_PERIOD."""
    n = name.upper()
    if n in ("FD001", "FD002", "FD003", "FD004"):
        return n
    return {"BEIJINGAQ": "BeijingAQ", "BEIJING_AQ": "BeijingAQ",
            "ETTM1": "ETTm1", "WEATHER": "Weather"}.get(n, n)
