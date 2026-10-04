"""hub/store-content — CONTENT SPECS: what goes into the store, at matched bytes.

⛔ The one dial this hub turns.  Everything else (grouping rule, atom selection,
address book, read verb, decoder, judge) is the pinned instrument.

THE BYTE ARITHMETIC, which is the whole constraint
--------------------------------------------------
``store_bytes = N · D · 4`` and the budget is **524,288 B at tolerance 0**, so
``N · D = 131,072 = 2^17``.  More content is therefore paid for in ATOMS and in
nothing else.  Every spec below asserts its own product.

WHERE THE CONTENT COMES FROM
----------------------------
* ``p``  — the **frozen** penultimate cache.  ⛔ Never recomputed, so every arm's
  penultimate content is bit-identical to the incumbent's.
* ``b1``/``b2``/``b3`` — this hub's earlier-layer cache, one contiguous CPU pass of
  the frozen encoder, 4×4 block-mean pooled onto the penultimate map's own grid.

SCALING, and why it costs zero bytes
------------------------------------
Blocks have different activation scales, and a raw concatenation would let whichever
block is larger dominate every squared distance in ``soft_atom_assignment`` — i.e.
the content dial would secretly be a *weighting* dial.  Each block is therefore
standardised to unit RMS **by a scalar computed from the stored atoms' own rows**.
⭐ That is one float per block and it is a function of the atoms, so it is free by
exactly the convention ``coupling_byte_ledger`` already declares for ``m_k``/``s_k``
("functions of the ATOMS (0 additional bytes)").  ⛔ It is a per-block scalar, never
a per-coordinate vector, which would be a fitted diagonal and would cost D floats.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUDGET_BYTES = 524_288
_ND = BUDGET_BYTES // 4          # 131_072 = N * D

BRIDGE = os.environ.get(
    "CHLU_GEN_BRIDGE_ROOT",
    "C:/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/gen-feature-bridge/runs",
)
EARLY = Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs")) / "hub-store-content" / "cache"

#: dims of each source block on the shared 4x4 grid
BLOCK_DIMS = {"p": 2048, "b3": 2048, "b2": 1024, "b1": 512}
BLOCK_CH = {"p": 128, "b3": 128, "b2": 64, "b1": 32}


@dataclass(frozen=True)
class ContentSpec:
    """One declared answer to "what is written into the store"."""

    name: str
    blocks: tuple
    half_channels: tuple = ()
    shuffle: tuple = ()
    shuffle_seed: int = 0
    note: str = ""

    @property
    def dims_per_block(self) -> dict:
        return {b: (BLOCK_DIMS[b] // 2 if b in self.half_channels else BLOCK_DIMS[b])
                for b in self.blocks}

    @property
    def D(self) -> int:
        return int(sum(self.dims_per_block.values()))

    @property
    def n_atoms(self) -> int:
        d = self.D
        if _ND % d:
            raise ValueError(
                f"{self.name}: D={d} does not divide N*D={_ND}, so this content "
                "CANNOT be byte-matched at tolerance 0 - not a legal arm")
        return _ND // d

    @property
    def store_bytes(self) -> int:
        return int(self.n_atoms * self.D * 4)

    @property
    def n_penult(self) -> int:
        """Width of the leading slice the frozen decoder is handed."""
        if self.blocks[0] != "p":
            raise ValueError(f"{self.name}: 'p' must come first")
        return int(self.dims_per_block["p"])

    def assert_matched(self) -> dict:
        b = self.store_bytes
        if b != BUDGET_BYTES:
            raise AssertionError(
                f"{self.name}: {b:,} B != the {BUDGET_BYTES:,} B budget")
        return {"name": self.name, "blocks": list(self.blocks),
                "half_channels": list(self.half_channels),
                "shuffle": list(self.shuffle), "shuffle_seed": int(self.shuffle_seed),
                "D": self.D, "n_atoms": self.n_atoms, "store_bytes": b,
                "budget_bytes": BUDGET_BYTES, "byte_matched": True,
                "dims_per_block": self.dims_per_block,
                "bytes_formula": "n_atoms * D * 4", "note": self.note}


# --------------------------------------------------------------------------
# the declared arms
# --------------------------------------------------------------------------

SPECS = {
    "P": ContentSpec(
        "P", ("p",),
        note="THE INCUMBENT CONTENT: the penultimate map alone, 64 atoms. Every "
             "number in the programme so far was measured on this."),
    "WIDEN": ContentSpec(
        "WIDEN", ("p", "b3"),
        note="THE ARM. All of the penultimate map PLUS conv block 3 (the map one "
             "block earlier, 8x8, block-mean pooled to 4x4). Paid for in atoms: "
             "64 -> 32. D=4096 <= the 20,000 LACE whitening limit, so LACE stays fair."),
    "WIDEN_SHUF": ContentSpec(
        "WIDEN_SHUF", ("p", "b3"), shuffle=("b3",), shuffle_seed=20_260_919,
        note="THE LAUNDERING CONTROL for WIDEN. Identical D, N, bytes and marginal "
             "statistics; the b3 rows are permuted so the block carries NO information "
             "about the row it sits beside. A WIDEN gain that survives here is not content."),
    "WIDEN_B2": ContentSpec(
        "WIDEN_B2", ("p", "b2", "b1"), half_channels=("b1",),
        note="Two and three blocks earlier instead of one: b2 (1024) + half of b1 (256). "
             "D=3328 does NOT divide 2^17 - kept to show it was considered and rejected "
             "on the byte arithmetic, not on taste."),
    "WIDEN_MIX": ContentSpec(
        "WIDEN_MIX", ("p", "b3", "b2"), half_channels=("b3",),
        note="The added 2048 dims spread over TWO earlier depths instead of one: "
             "half of b3's channels (1024) + all of b2's (1024). D=4096, N=32, "
             "byte-matched. Answers 'is one block back enough, or is it depth?'"),
    "WIDEN_MIX_SHUF": ContentSpec(
        "WIDEN_MIX_SHUF", ("p", "b3", "b2"), half_channels=("b3",),
        shuffle=("b3", "b2"), shuffle_seed=20_260_919,
        note="The laundering control for WIDEN_MIX."),
    # -- considered and KILLED, kept in the table so the record shows why ----
    "SWAP": ContentSpec(
        "SWAP", ("p", "b3"), half_channels=("p", "b3"),
        note="KILLED BY THE FROZEN DECODER, not by the byte arithmetic. It is "
             "byte-legal (D=2048, N=64) and it was the cleanest reallocation axis - "
             "same D, same N as the incumbent - but it keeps only 64 of the 128 "
             "penultimate channels, and the shared decoder's input is (128,4,4). "
             "Feeding it a half-empty map is a content the decoder never saw. "
             "DIAGNOSTIC ONLY, never a scored row."),
    "SWAP_B2": ContentSpec(
        "SWAP_B2", ("p", "b2"), half_channels=("p",),
        note="Same kill as SWAP: half a penultimate map cannot be rendered."),
}

#: Byte-matched at tolerance 0 AND renderable by the frozen decoder (full 2048-dim
#: penultimate slice).  These are the only specs that may reach a scored row.
LEGAL = ("P", "WIDEN", "WIDEN_SHUF", "WIDEN_MIX", "WIDEN_MIX_SHUF")
#: Byte-legal but NOT renderable - kept for the record.
KILLED_BY_DECODER = ("SWAP", "SWAP_B2")


# --------------------------------------------------------------------------
# materialising a spec
# --------------------------------------------------------------------------

def open_sources(seed: int = 0):
    bdir = Path(BRIDGE) / "shapes3d_recon" / f"seed_{seed}"
    src = {"p": np.load(bdir / "features.npy", mmap_mode="r")}
    edir = EARLY / f"shapes3d_recon_seed_{seed}"
    for b in ("b1", "b2", "b3"):
        p = edir / f"{b}.npy"
        if p.exists():
            src[b] = np.load(p, mmap_mode="r")
    return src


def _channel_slice(b: str, half: bool) -> np.ndarray:
    c = BLOCK_CH[b]
    return np.arange(0, c, 2) if half else np.arange(c)


def build_Z(spec: ContentSpec, rows, seed: int = 0, src=None,
            scales=None, chunk: int = 8192):
    """``rows -> ((n, D) float32, scales)`` joint content for ``spec``.

    ``scales`` (per-block RMS) is computed once from the ATOM rows and passed in for
    every later call, so the standardisation is a function of the stored atoms and
    not of whatever set happens to be encoded at the time.
    """
    rows = np.asarray(rows, np.int64)
    src = src if src is not None else open_sources(seed)
    parts, got = [], {}
    for b in spec.blocks:
        ch = _channel_slice(b, b in spec.half_channels)
        r = rows
        if b in spec.shuffle:
            rng = np.random.default_rng(int(spec.shuffle_seed) + 1009 * int(seed))
            r = rows[rng.permutation(rows.size)]
        out = np.empty((rows.size, ch.size * 16), np.float32)
        order = np.argsort(r, kind="stable")
        rs = r[order]
        for i in range(0, rs.size, chunk):
            sl = slice(i, min(i + chunk, rs.size))
            a = np.asarray(src[b][rs[sl]], np.float32)[:, ch]
            out[order[sl]] = a.reshape(a.shape[0], -1)
        if scales is not None and b in scales:
            s = float(scales[b])
        else:
            s = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) or 1.0
        got[b] = s
        parts.append(out / s)
    Z = np.concatenate(parts, axis=1)
    if Z.shape[1] != spec.D:
        raise AssertionError(f"{spec.name}: built width {Z.shape[1]} != D {spec.D}")
    return Z, got
