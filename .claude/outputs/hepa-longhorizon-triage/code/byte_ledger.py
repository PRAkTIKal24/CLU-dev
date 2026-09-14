"""HEPA per-example state-byte ledger, and the matched store sizing.

THE ONE THING THIS FILE MUST GET RIGHT (task §4): the exemplar store is sized to
the bytes HEPA itself carries at context L, so a HEPA win is a win of mechanism,
not of capacity. The arithmetic is spelled out so the Hub can audit it.

Source of every constant: hepa/utils/config.py::PROTOCOL and
hepa/model/encoder.py::CausalEncoder (patch_size=16, d_model=256, n_layers=2,
torch default dtype float32 = 4 B).

Definitions
-----------
n_tokens(L) = ceil(L / P)                       # PatchEmbedding right-pads T to a multiple of P

Three readings of "the state HEPA carries at context L", all reported:

  token_only      = n_tokens * d_model * 4
      The patch-embedded token stream (post PatchEmbedding + sinusoidal PE), i.e.
      the minimum HEPA must materialise for a window of length L.

  residual_stream = (n_layers + 1) * n_tokens * d_model * 4          <-- PRIMARY
      The residual stream at the input of each of the n_layers blocks plus the
      final LayerNorm'd stream. This is what a forward pass over the window
      actually holds; it is the reading we size the store to.

  kv_cache        = (1 + 2*n_layers) * n_tokens * d_model * 4  +  d_model * 4
      The streaming/deployment reading: the embedded tokens, plus K and V for
      every token at every layer (causal attention), plus the pooled h_t that
      leaves the encoder. This is the largest defensible reading.

NOT a reading we use: the pooled output h_t alone (d_model*4 = 1024 B). It is
context-independent, so it makes every rung of the context sweep share one
budget and turns the 8192 rung into a strawman (task §4.3 forbids exactly this).

Store sizing
------------
The store holds `n_exemplars` rows of (feature vector, payload). Feature dtype is
float32 (matching HEPA's). Payload is the scalar time-to-event (1 float) for the
PREREG-compliant arm, or the K-vector label surface for the flagged diagnostic.

  bytes_per_exemplar = (feature_dim + payload_floats) * 4
  n_exemplars        = floor(budget_bytes / bytes_per_exemplar)

Index/array overhead (numpy headers, the scaler's 2*feature_dim floats) is NOT
charged. It is O(1) per store, at most a few hundred bytes, and charging it
would only shrink the tripwire; we declare it rather than hide it.
"""

from __future__ import annotations

import math
from typing import Dict

from tw_config import D_MODEL, DTYPE_BYTES, N_LAYERS, PATCH_SIZE


def n_tokens(context_len: int, patch_size: int = PATCH_SIZE) -> int:
    """Tokens HEPA's PatchEmbedding produces for a window of `context_len`."""
    return int(math.ceil(context_len / patch_size))


def state_bytes(context_len: int) -> Dict[str, int]:
    """All three readings of HEPA's per-example carried state at context L."""
    n = n_tokens(context_len)
    tok = n * D_MODEL * DTYPE_BYTES
    return {
        "token_only": tok,
        "residual_stream": (N_LAYERS + 1) * tok,
        "kv_cache": (1 + 2 * N_LAYERS) * tok + D_MODEL * DTYPE_BYTES,
    }


def ledger(context_len: int, n_channels: int) -> Dict[str, object]:
    """Full auditable ledger row for one context rung."""
    n = n_tokens(context_len)
    b = state_bytes(context_len)
    return {
        "context_len": int(context_len),
        "patch_size": PATCH_SIZE,
        "n_tokens": n,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "dtype_bytes": DTYPE_BYTES,
        "n_channels": int(n_channels),
        "raw_window_bytes": int(context_len * n_channels * DTYPE_BYTES),
        "arithmetic": {
            "n_tokens": f"ceil({context_len}/{PATCH_SIZE}) = {n}",
            "token_only": f"{n} * {D_MODEL} * {DTYPE_BYTES} = {b['token_only']}",
            "residual_stream": (
                f"({N_LAYERS}+1) * {n} * {D_MODEL} * {DTYPE_BYTES} = {b['residual_stream']}"
            ),
            "kv_cache": (
                f"(1+2*{N_LAYERS}) * {n} * {D_MODEL} * {DTYPE_BYTES} "
                f"+ {D_MODEL}*{DTYPE_BYTES} = {b['kv_cache']}"
            ),
        },
        "budget_bytes": b,
    }


def store_capacity(budget_bytes: int, feature_dim: int, payload_floats: int) -> Dict[str, int]:
    """How many exemplars fit in `budget_bytes`, and what that realises."""
    per = (feature_dim + payload_floats) * DTYPE_BYTES
    n_ex = int(budget_bytes // per)
    return {
        "feature_dim": int(feature_dim),
        "payload_floats": int(payload_floats),
        "bytes_per_exemplar": per,
        "n_exemplars_budgeted": n_ex,
        "budget_bytes": int(budget_bytes),
    }
