"""⭐⭐ **Feature-factored launches** — C2W11's structural change (§A34.1).

> **ONE PARTICLE PER SEMANTIC FEATURE CHANNEL of ``phi``.** ``k`` is *structured
> by the encoder's decomposition, not free.*

**What this module is, and what it is deliberately not.** It builds the channel
decomposition of ``phi``, the per-channel launch head, and the launch-geometry
instrumentation (K0, K6, M5's launch side, M6's launch side, and the C2W9
coverage statistic). ⛔ **It builds NO binding structure.** Whether the ``k``
landed particles can be pooled back into one answer is the READ's job (a
set-level DeepSets ``psi``) and belongs to another spoke. Co-activation /
wormhole edges are a C2W9 pointer and are not here either.

## Why this is not C2W5's launch again, stated so it can be checked

C2W5 launched ``P = 4`` particles from **one** set-code at fixed designed
offsets ``o_p``. Measured, before any store existed: the particles occupied
**2.20** of the ``F = 4`` required distinct wells, ``>= F`` distinct wells were
reachable on **5.0 %** of queries, and exact-set occupancy was
**0.0000 / 2 560** (``orgdiv-null-arms`` §3). *That is a launch-geometry cap
that existed before any store was written* — no reader and no organizer can
repair it, because you cannot sum four vectors you never visited.

The head here makes the particles **structurally distinct by construction**
instead of by offset noise, by decomposing the set-code against ``phi``'s own
frozen code dictionary with greedy matched-filter **deflation**::

    r_0 = phi(x)
    for c = 1..k:   j_c = argmax_j <r_{c-1}, e_j>
                    r_c = r_{c-1} - <r_{c-1}, e_{j_c}> e_{j_c}
    q_c = (R * e_{j_c} + sigma_q * xi_c ,  0_m)

After a full deflation ``<r_c, e_{j_c}> = 0`` exactly, so the ``k`` selected
code directions are distinct with probability 1.

## ⛔ What the head is allowed to read (the leak boundary, stated up front)

It reads ``phi(x)`` and ``phi``'s **own frozen parameters** ``{e_j}`` (already
ledgered in ``phi_bytes``) — and nothing else. It never reads ``A(x)``, never
reads a payload, and never reads the store. It is the **address head**, which
§A31.4 makes its own head: task features are not address features, and
cheap unfitted address geometry is a legitimate default (the measured inversion:
the task-strong encoder was address-**worst** beyond 2 SE).

⚠ **The consequence, registered rather than discovered:** a matched filter over
the codes is a *good* launch geometry, so ``K6`` — the fraction of queries whose
asserted set is *already exactly right* before any reader is fitted — must be
reported beside every downstream number. Raising addressability (K0) without
raising precision (K6) is the whole point; raising both would mean the launch
head, not the store, is answering the question.

⛔ **Wells are never named semantically** (``PREREG-TierII.md`` §2.6). Wells,
channels and features carry integer indices and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "FeatureLaunchHead",
    "build_launch_head",
    "launch_points",
    "k0_stats",
    "asserted_sets",
    "k6_already_right",
    "coverage_stats",
    "wells_visited",
    "occupancy_precision_from_points",
]


# ==========================================================================
# the head
# ==========================================================================
class FeatureLaunchHead(eqx.Module):
    """``k`` launch points, one per semantic feature channel of ``phi``.

    Holds **no parameters of its own** — it reuses ``phi``'s frozen codes, so
    its byte cost is exactly ``0`` and the byte ledger is unchanged by the
    switch from designed offsets to feature-factored launches. (The designed
    offsets it replaces *were* parameters, so the swap is byte-negative; both
    numbers are emitted by :meth:`n_bytes` and its ``replaces_bytes``.)
    """

    codes: jnp.ndarray  # (N_a, d) unit query codes -- phi's, not ours
    radius: float = eqx.field(static=True)
    addr_dim: int = eqx.field(static=True)
    payload_dim: int = eqx.field(static=True)
    k: int = eqx.field(static=True)
    collapse_to_one_channel: bool = eqx.field(static=True)

    def __init__(self, codes, radius: float, addr_dim: int, payload_dim: int,
                 k: int, collapse_to_one_channel: bool = False):
        self.codes = jnp.asarray(codes)
        self.radius = float(radius)
        self.addr_dim = int(addr_dim)
        self.payload_dim = int(payload_dim)
        self.k = int(k)
        # ⛔ M1's DESIGNED NEGATIVE, shipped inside the head so it cannot drift
        # away from the thing it is meant to falsify: every channel collapses
        # onto channel 1. A launch set that cannot fail here does not ship.
        self.collapse_to_one_channel = bool(collapse_to_one_channel)

    def set_code(self, indicator: jnp.ndarray) -> jnp.ndarray:
        """``(..., N_a) -> (..., d)``: ``phi``'s lossy set-code, on the shell."""
        s = indicator @ self.codes
        return self.radius * s / (jnp.linalg.norm(s, axis=-1, keepdims=True) + 1e-12)

    def channels(self, code: jnp.ndarray) -> jnp.ndarray:
        """``(d,) -> (k,)`` the channel code indices, by matched-filter deflation."""
        n = self.codes.shape[0]
        r = code
        mask = jnp.zeros((n,), dtype=code.dtype)
        picks = []
        for _ in range(self.k):
            align = self.codes @ r - 1e9 * mask
            j = jnp.argmax(align)
            e = self.codes[j]
            r = r - jnp.dot(r, e) * e
            mask = mask + jax.nn.one_hot(j, n, dtype=code.dtype)
            picks.append(j)
        out = jnp.stack(picks)
        if self.collapse_to_one_channel:
            out = jnp.full_like(out, out[0])
        return out

    def launch(self, indicator: jnp.ndarray, key=None,
               sigma_q: float = 0.0) -> jnp.ndarray:
        """``(N_a,) -> (k, dim)`` launch positions; payload block pinned to 0.

        ⚠ The payload block is pinned to ``0`` — the anti-decoration guard,
        inherited verbatim from ``FrozenPhi.launch``. The read must *dissipate
        up to* ``v_j``; nothing hands it the answer.
        """
        code = self.set_code(indicator)
        js = self.channels(code)
        pts = self.radius * self.codes[js]  # (k, d)
        if key is not None and sigma_q > 0.0:
            pts = pts + sigma_q * jax.random.normal(key, pts.shape)
        pad = jnp.zeros((pts.shape[0], self.payload_dim), dtype=pts.dtype)
        return jnp.concatenate([pts, pad], axis=-1)

    def n_bytes(self) -> Dict[str, int]:
        """⭐ ZERO parameters of its own. Emitted from the code, never a doc."""
        return {"head_bytes": 0, "reuses_phi_codes_bytes": int(self.codes.size * 4)}


def build_launch_head(phi, cfg, *, collapse_to_one_channel: bool = False
                      ) -> FeatureLaunchHead:
    """Build the head from a frozen ``phi`` and a :class:`CatTestConfig`."""
    k = int(cfg.n_channels) if cfg.n_channels is not None else int(cfg.f_subset)
    return FeatureLaunchHead(phi.codes, float(cfg.ball_radius), int(cfg.addr_dim),
                             int(cfg.payload_dim), k,
                             collapse_to_one_channel=collapse_to_one_channel)


def launch_points(head_or_phi, indicators: np.ndarray, cfg, key,
                  *, sigma_q: Optional[float] = None) -> np.ndarray:
    """``(B, N_a) -> (B, k or P, dim)`` raw launch geometry. **No store.**"""
    sq = float(cfg.query_sigma) if sigma_q is None else float(sigma_q)
    ind = jnp.asarray(indicators, dtype=jnp.float32)
    keys = jax.random.split(key, ind.shape[0])
    fn = jax.jit(jax.vmap(lambda i, kk: head_or_phi.launch(i, kk, sq)))
    return np.asarray(fn(ind, keys))


# ==========================================================================
# ⭐ K0 — launch expressivity, computed from launch geometry with NO store
# ==========================================================================
def _occ(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """``(B, k, dim) -> (B, k)`` nearest-anchor index (a per-read transient)."""
    a = np.asarray(anchors, dtype=float)
    z = np.asarray(points, dtype=float)[..., : a.shape[1]]
    d2 = ((z[:, :, None, :] - a[None, None, :, :]) ** 2).sum(-1)
    return d2.argmin(-1)


def k0_stats(points: np.ndarray, anchors: np.ndarray, subsets: np.ndarray,
             f_subset: int) -> Dict[str, Any]:
    """⭐⭐ **K0** — the leg C2W5 never had, and the cheapest kill in the wave.

    ``>= F`` **distinct** feature wells reachable, and the mean number reachable
    — from the launch geometry alone, with **no store written**. Reference
    (C2W5's ``P = 4`` designed offsets, 5 seeds, 2 560 queries): distinct-``F``
    fraction **0.050**, mean distinct **2.202**, exact-set occupancy
    **0.0000 / 2 560**.

    ⭐ The **full distribution** of distinct wells reachable is returned, not
    just the mean, and the per-channel breakdown beside it (the task's explicit
    requirement: a mean hides a bimodal launch set).
    """
    occ = _occ(points, anchors)  # (B, k)
    B, k = occ.shape
    distinct = np.array([len(np.unique(occ[i])) for i in range(B)], dtype=int)
    # ⭐⭐ THE DECISIVE STATISTIC, and it is neither of the two obvious ones.
    # `mean_distinct_wells` counts how many DIFFERENT wells the launch set
    # reaches; `occupancy_precision` counts what fraction of particles land in a
    # right well. A read that must express ``y = sum_{j in A} v_j`` needs
    # neither on its own -- it needs the number of **distinct wells that are
    # also CORRECT**, i.e. ``|set(occupancy) & A(x)|``. A launch set can raise
    # the first without raising this one (by spreading onto wrong wells) or
    # raise the second without raising this one (by piling every particle onto
    # one right well). ⛔ Quote this beside every K0 number.
    correct_distinct = np.array(
        [len(set(occ[i].tolist()) & set(np.asarray(subsets)[i].tolist()))
         for i in range(B)], dtype=int)
    hist = np.bincount(distinct, minlength=k + 1)[: k + 1]
    sub = np.asarray(subsets)
    exact = np.array([set(occ[i].tolist()) == set(sub[i].tolist())
                      for i in range(B)], dtype=bool)
    # per-channel: does channel c land in a well of A(x)?
    per_ch = np.array([[bool(occ[i, c] in set(sub[i].tolist())) for c in range(k)]
                       for i in range(B)], dtype=bool)
    per_ch_distinct = np.array(
        [[len(np.unique(occ[i, : c + 1])) for c in range(k)] for i in range(B)])
    return {
        "k": int(k), "n_queries": int(B), "F": int(f_subset),
        "mean_distinct_wells": float(distinct.mean()),
        "sd_distinct_wells": float(distinct.std(ddof=1)) if B > 1 else 0.0,
        "frac_ge_F_distinct": float((distinct >= int(f_subset)).mean()),
        "mean_correct_distinct_wells": float(correct_distinct.mean()),
        "sd_correct_distinct_wells": (float(correct_distinct.std(ddof=1))
                                      if B > 1 else 0.0),
        "correct_distinct_histogram": np.bincount(
            correct_distinct, minlength=int(f_subset) + 1)[
                : int(f_subset) + 1].tolist(),
        "frac_all_F_correct_distinct": float(
            (correct_distinct >= int(f_subset)).mean()),
        "distinct_histogram": hist.tolist(),
        "distinct_hist_labels": list(range(k + 1)),
        "exact_set_occupancy": float(exact.mean()),
        "per_channel_precision": per_ch.mean(axis=0).tolist(),
        "per_channel_cumulative_distinct": per_ch_distinct.mean(axis=0).tolist(),
        "occupancy_precision": float(per_ch.mean()),
    }


# ==========================================================================
# ⭐ K6 — the fifth-session slip, now owned: computed BEFORE any reader
# ==========================================================================
def asserted_sets(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """``(B, k, dim) -> (B, k)`` the set the launch geometry *asserts*."""
    return _occ(points, anchors)


def k6_already_right(points: np.ndarray, anchors: np.ndarray,
                     subsets: np.ndarray) -> Dict[str, Any]:
    """⭐ **K6** — the fraction of queries whose asserted set is already right.

    One line, computed **before any reader is fitted**. It is not a kill; it is
    a mandatory reported precondition that scopes the interpretation of every
    fitted-reader score: the reader-fitting pathology destroys signal only in
    proportion to this fraction. Reference fractions: **2/2560 · 3/1280 ·
    0/2560** (C2W5's cells) versus **~18 %** (C2W7's).
    """
    occ = _occ(points, anchors)
    sub = np.asarray(subsets)
    B = occ.shape[0]
    right = np.array([set(occ[i].tolist()) == set(sub[i].tolist())
                      for i in range(B)], dtype=bool)
    n_right = int(right.sum())
    return {"k6_frac_already_right": float(right.mean()),
            "k6_n_right": n_right, "k6_n_queries": int(B),
            "k6_as_fraction": f"{n_right}/{B}"}


# ==========================================================================
# ⭐ THE C2W9 COVERAGE STATISTIC (§7 of PREREG-C2W11 — spoke A owns this half)
# ==========================================================================
def coverage_stats(points: np.ndarray, anchors: np.ndarray, subsets: np.ndarray,
                   reach: float, *, threshold: float = 0.20,
                   full_targets: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Is the needed feature well inside the union of the ``k`` launch diamonds?

    A **coverage failure** is a launch-head problem (as opposed to a traversal
    failure, which is in-flight and is spoke B's). Channel ``c``'s diamond is
    the ball of radius ``reach`` about its launch point; a needed well ``j in
    A(x)`` is covered iff ``min_c ||u_j - q_c|| <= reach``.

    ⛔ The threshold is registered **before** the run: the trigger fires iff the
    mean fraction of *uncovered needed wells* exceeds ``threshold``.
    """
    a = np.asarray(anchors, dtype=float)
    q = np.asarray(points, dtype=float)[..., : a.shape[1]]
    sub = np.asarray(subsets)
    B, k, _ = q.shape
    F = sub.shape[1]
    # (B, k, N_a) launch-point-to-anchor distances
    dist = np.linalg.norm(q[:, :, None, :] - a[None, None, :, :], axis=-1)
    dmin = dist.min(axis=1)  # (B, N_a): nearest launch point per well
    needed = np.take_along_axis(dmin, sub, axis=1)  # (B, F)
    covered = needed <= float(reach)
    frac_uncovered = 1.0 - covered.mean(axis=1)  # per query
    per_slot = 1.0 - covered.mean(axis=0)  # per position in A(x) (not semantic)
    mean_unc = float(frac_uncovered.mean())
    # ⭐ THE FULL-SPACE VARIANT (added after the registered address-space form
    # was measured, and LABELLED as added). The registered diamond lives in the
    # ADDRESS block; but the read launches with the payload block pinned to 0
    # and a well's full target sits at payload ``v_j``, so the address-space
    # statistic cannot see the distance the read actually has to cross. Both are
    # reported; ⛔ the registered address-space one remains the primary and the
    # trigger's threshold is applied to it.
    full = None
    if full_targets is not None:
        ft = np.asarray(full_targets, dtype=float)
        qf = np.asarray(points, dtype=float)
        dist_f = np.linalg.norm(qf[:, :, None, :] - ft[None, None, :, :], axis=-1)
        dmin_f = dist_f.min(axis=1)
        needed_f = np.take_along_axis(dmin_f, sub, axis=1)
        unc_f = 1.0 - (needed_f <= float(reach)).mean(axis=1)
        full = {"mean_frac_needed_wells_uncovered": float(unc_f.mean()),
                "median_distance_to_needed_full_target": float(np.median(needed_f)),
                "label": ("ADDED after the registered address-space form was "
                          "measured; reported beside it, never instead of it")}
    return {
        "reach_radius": float(reach),
        "full_space_variant": full,
        "mean_frac_needed_wells_uncovered": mean_unc,
        "sd_frac_uncovered": float(frac_uncovered.std(ddof=1)) if B > 1 else 0.0,
        "frac_queries_fully_covered": float((frac_uncovered == 0.0).mean()),
        "frac_queries_zero_coverage": float((frac_uncovered == 1.0).mean()),
        "per_slot_frac_uncovered": per_slot.tolist(),
        "median_distance_to_needed_well": float(np.median(needed)),
        "n_queries": int(B), "k": int(k), "F": int(F),
        "threshold": float(threshold),
        "coverage_trigger_fired": bool(mean_unc > float(threshold)),
    }


# ==========================================================================
# ⭐ M5 — anti-collapse: DIRECT wells-visited, TWO-SIDED (§A26.4)
# ==========================================================================
def wells_visited(points: np.ndarray, anchors: np.ndarray, n_wells: int,
                  *, band_low: float = 0.75) -> Dict[str, Any]:
    """``W / N_a`` — the wells ever occupied over a query set.

    ⛔ The ``S_eff in [8, 16]`` band is **RETIRED** (§A26.4): its lower half is
    unreachable by construction, because ``S_eff = K*F/W`` with ``W <= N_a``.
    The instrument here is the direct fraction, **two-sided**, and the two
    failure modes are named apart:

    * ``concentration`` (a few wells absorb everything) is **COLLAPSED**;
    * ``under-usage`` is labelled **under-usage** and is not called collapse.

    ⚠ ``S_eff`` is still emitted for continuity with the banked cells, and is
    explicitly **not** the leg. Reference concentration failure: C2W5's
    15 / 10 / 14 of 32 wells ever occupied.
    """
    occ = _occ(points, anchors)
    counts = np.bincount(occ.reshape(-1), minlength=int(n_wells))
    W = int((counts > 0).sum())
    frac = W / float(n_wells)
    total = counts.sum()
    p = counts / max(total, 1)
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    # participation ratio of the marginal well usage: 1 => one well, N_a => flat
    part = float(1.0 / max((p ** 2).sum(), 1e-12))
    return {"wells_visited": W, "n_wells": int(n_wells),
            "wells_visited_frac": float(frac),
            "band_low": float(band_low),
            "verdict": "OK" if frac >= band_low else "COLLAPSED",
            "label": ("ok" if frac >= band_low else
                      ("concentration" if part < 0.5 * n_wells else "under-usage")),
            "marginal_participation_ratio": part,
            "marginal_entropy_nats": entropy,
            "max_entropy_nats": float(np.log(max(n_wells, 1))),
            "s_eff_legacy_do_not_use_as_a_leg": (
                float("inf") if W == 0 else None)}


def occupancy_precision_from_points(points: np.ndarray, anchors: np.ndarray,
                                    subsets: np.ndarray) -> float:
    """Fraction of particles landing in a well **belonging to ``A(x)``**.

    ⛔ Scored against the **BLANK STORE / raw launch geometry**, never against
    ``F/N_a`` (C2W5 reconciliation 4: the store was simultaneously above chance,
    0.297 vs 0.125, *and below its own launder*, 0.406). ⛔ Reported as a
    co-activation statistic against task structure — never as a semantic
    identification of a well.
    """
    occ = _occ(points, anchors)
    sub = np.asarray(subsets)
    return float(np.mean([np.isin(occ[i], sub[i]).mean()
                          for i in range(occ.shape[0])]))


@dataclass
class LaunchProtocol:
    """The frozen launch protocol, emitted into the interfaces JSON."""

    mode: str
    k: int
    rule: str
    launch_key: int
    sigma_q: float
    radius: float
    payload_block: str = "pinned to 0 (the anti-decoration guard)"

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "k": self.k, "rule": self.rule,
                "launch_key": int(self.launch_key), "sigma_q": float(self.sigma_q),
                "shell_radius": float(self.radius),
                "payload_block": self.payload_block}
