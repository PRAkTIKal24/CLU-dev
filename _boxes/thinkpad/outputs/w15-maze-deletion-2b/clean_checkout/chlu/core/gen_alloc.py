"""⭐⭐ **D6 — spend codewords where the distortion is**, at CONSTANT TOTAL BYTES.

⭐ **DIAL: codeword allocation across wells, at constant total bytes.**
- **Laundering control:** uniform ``M = 64`` at the same total bytes — the shipped
  arm, **re-fitted through the identical code path**.
- **Falsifies:** the scaling law's *actionability*.  ⛔ This is a **CONTROL ARM for
  the dynamical read, NOT a gate on it.**

**The law under test** (`gen-variance-equalisation` §4, measured parameter-free): a
vector quantiser with ``M`` codewords on a source of **effective dimension**
``PR = (Σλ)²/Σλ²`` has relative distortion

    ε(M) = M^(−2/PR).

The wells' ``PR`` spans **3.12 … 13.63** against nominal widths ``d_k = 283 … 744``,
so ⛔ **the wells are not equally hard and a uniform ``M`` is not obviously the right
design.**  This module derives three allocations from the **measured, label-free**
``PR`` at one byte budget ``Σ_k M_k·d_k·4``:

``uniform``            ``M_k = M``.  ⭐ the control.
``equal_distortion``   ``M_k = ε^(−PR_k/2)`` with ``ε`` solved to the budget — the
                       rule D6 names.
``mse_optimal``        the Lagrangian minimiser of ``Σ_k v_k·ε_k(M_k)`` at the same
                       budget — ⭐ the allocation that actually minimises the read
                       error, and the one that says whether re-allocation is a lever
                       at all.

⛔ **What this CANNOT do, stated up front rather than discovered** (§0.5 consequence
2): at ε = 0.10 a ``PR = 13.63`` well needs ~6.5 **million** codewords.  ⭐ No
re-allocation at 524,288 B rescues the high-``PR`` wells.  A partial fix is not a
fix and this module's reporting says so on every row.

⛔ **No factor labels anywhere.**  ``PR`` comes from the covariance of the well's own
store-training features, so the allocation is part of the store's own (generative)
fit and not a supervised choice.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

#: ⭐ The shipped store's byte budget, ``M·D·4 = 64·2048·4``.  Every allocation here
#: is measured against the UNIFORM arm's own byte count, never against this constant
#: — ``D`` is a per-seed property of the grouping.  Kept for reporting only.
STORE_BYTE_BUDGET = 524_288

#: The allocations this module knows how to build.  ⛔ ``uniform`` is the CONTROL,
#: not a placeholder.
ALLOCATIONS = ("uniform", "equal_distortion", "mse_optimal")


# ---------------------------------------------------------------------------
# 1. The measured input — ⛔ label-free
# ---------------------------------------------------------------------------


def well_spectrum(X: np.ndarray, channels: Sequence[int]) -> dict:
    """⭐ One well's **label-free** variance spectrum: ``PR``, ``Σλ``, ``d_k``.

    ``PR = (Σλ)²/Σλ²`` is the participation ratio (effective dimension).  ⛔ No
    labels are read: the allocation must be derivable from the store's own training
    features or it is a supervised choice dressed up as a store.

    ⚠ Slice FIRST, then widen to float64 — casting the whole ``(N, D)`` block costs
    ~270 MB per call on a 16 GB machine (``gen-variance-equalisation`` §12 L1's
    neighbour).
    """
    idx = np.asarray(channels, np.int64)
    Z = np.asarray(np.asarray(X)[:, idx], np.float64)
    n, d = Z.shape
    Zc = Z - Z.mean(axis=0)
    C = (Zc.T @ Zc) / max(1, n - 1)
    lam = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    tot = float(lam.sum())
    sq = float((lam**2).sum())
    return {
        "d_k": int(d),
        "n_rows": int(n),
        "total_variance": tot,
        "participation_ratio": float(tot**2 / sq) if sq > 0 else float("nan"),
        "eigenvalues_head": [float(x) for x in np.sort(lam)[::-1][:16]],
        "label_free": True,
    }


def well_spectra(X: np.ndarray, well_channels) -> list:
    """:func:`well_spectrum` for every well, in store order."""
    return [well_spectrum(X, w) for w in well_channels]


# ---------------------------------------------------------------------------
# 2. The law
# ---------------------------------------------------------------------------


def predicted_distortion(pr: float, m: int) -> float:
    """``ε(M) = M^(−2/PR)`` — the high-rate law, parameter-free."""
    p, mm = float(pr), float(m)
    if not np.isfinite(p) or p <= 0 or mm < 1:
        return float("nan")
    return float(mm ** (-2.0 / p))


def predicted_distortion_vec(pr: Sequence[float], m: Sequence[int]) -> np.ndarray:
    return np.asarray([predicted_distortion(p, mm) for p, mm in zip(pr, m, strict=True)],
                      np.float64)


def variance_weighted_distortion(
    pr: Sequence[float], m: Sequence[int], var: Sequence[float]
) -> float:
    """``Σ_k v_k ε_k / Σ_k v_k`` — the whole feature vector's predicted relative error.

    ⭐ This is the quantity `gen-variance-equalisation` measured parameter-free
    (predicted 0.2533, measured 0.2783).
    """
    e = predicted_distortion_vec(pr, m)
    v = np.asarray(var, np.float64)
    tot = float(v.sum())
    return float((v * e).sum() / tot) if tot > 0 else float("nan")


def modes_for_target(pr: float, eps: float) -> float:
    """``M = ε^(−PR/2)`` — ⛔ the number that shows the high-``PR`` wells are lost."""
    return float(float(eps) ** (-float(pr) / 2.0))


def infeasibility_table(
    pr: Sequence[float], d: Sequence[int], targets: Sequence[float] = (0.278, 0.10, 0.05)
) -> dict:
    """⛔ **The honest ceiling.**  ``M`` needed per well to reach each target ε, and
    what it would cost in bytes.  ⭐ Reported *beside* every allocation so a partial
    fix is never presented as a fix (§0.5 consequence 2)."""
    d = np.asarray(d, np.float64)
    rows = []
    for t in targets:
        m = np.asarray([modes_for_target(p, t) for p in pr], np.float64)
        rows.append({
            "target_eps": float(t),
            "modes_needed_per_well": [float(x) for x in m],
            "bytes_needed": float((m * d).sum() * 4.0),
            "vs_uniform64_budget": float((m * d).sum() * 4.0 / (64.0 * d.sum() * 4.0)),
            "feasible_at_budget": bool((m * d).sum() <= 64.0 * d.sum()),
        })
    return {
        "targets": rows,
        "ruling": (
            "⛔ NO re-allocation at a fixed budget rescues a high-PR well: at ε = 0.10 "
            "a PR = 13.63 well needs ~6.5 million codewords. A partial fix is NOT a "
            "fix and must not be reported as one."
        ),
    }


# ---------------------------------------------------------------------------
# 3. ⭐ The allocations — all at ONE byte budget
# ---------------------------------------------------------------------------


def _budget_words(well_sizes: Sequence[int], m_uniform: int) -> int:
    """The uniform arm's codeword-times-width budget, ``Σ_k M·d_k = M·D``."""
    return int(m_uniform) * int(np.asarray(well_sizes, np.int64).sum())


def _greedy_topup(m: np.ndarray, d: np.ndarray, budget: int, gain) -> np.ndarray:
    """Spend the integer-rounding slack on the well with the best ``gain(k, M_k)``.

    ⛔ Never exceeds ``budget``: a store that is one byte bigger is *a bigger store*,
    not an allocation, and that is this spoke's own falsifier #1.
    """
    m = np.asarray(m, np.int64).copy()
    d = np.asarray(d, np.int64)
    while True:
        used = int((m * d).sum())
        cand = [k for k in range(len(m)) if used + int(d[k]) <= budget]
        if not cand:
            return m
        best = max(cand, key=lambda k: gain(k, int(m[k])))
        m[best] += 1


def allocate_uniform(well_sizes: Sequence[int], m_uniform: int = 64) -> np.ndarray:
    """⭐ **The laundering control.**  ``M_k = M`` — the shipped arm."""
    return np.full(len(list(well_sizes)), int(m_uniform), np.int64)


def allocate_equal_distortion(
    pr: Sequence[float],
    well_sizes: Sequence[int],
    m_uniform: int = 64,
    m_min: int = 2,
    m_max: int = 100_000,
) -> np.ndarray:
    """⭐ **The rule D6 names:** equalise ``ε_k`` across wells at constant bytes.

    ``M_k = ε^(−PR_k/2)`` with ``ε`` bisected so ``Σ M_k d_k ≤ M·D``.

    ⛔ **Equalising distortion is NOT minimising it.**  At a fixed budget, levelling
    the hard wells down means levelling the **easy** wells *up*: the low-``PR`` wells
    give up most of their codewords for a very small reduction in the hard wells'
    error.  :func:`allocate_mse_optimal` is the allocation that minimises the read
    error, and the two are reported side by side precisely so this is visible.
    """
    PR = np.asarray(pr, np.float64)
    d = np.asarray(well_sizes, np.int64)
    budget = _budget_words(d, m_uniform)

    def words(e: float) -> np.ndarray:
        return np.clip(np.round(np.power(float(e), -PR / 2.0)), m_min, m_max).astype(np.int64)

    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if int((words(mid) * d).sum()) > budget:
            lo = mid
        else:
            hi = mid
    m = words(hi)
    if int((m * d).sum()) > budget:      # degenerate budget (m_min alone overshoots)
        return m
    return _greedy_topup(
        m, d, budget,
        # give the next codeword to the well with the WORST current distortion
        gain=lambda k, mk: predicted_distortion(PR[k], mk),
    )


def allocate_mse_optimal(
    pr: Sequence[float],
    well_sizes: Sequence[int],
    var: Sequence[float],
    m_uniform: int = 64,
    m_min: int = 2,
    m_max: int = 100_000,
) -> np.ndarray:
    """⭐ **The byte-constrained minimiser of the read error**, for reference.

    ``min_M Σ_k v_k M_k^(−2/PR_k)`` s.t. ``Σ_k M_k d_k ≤ M·D``.  Stationarity gives
    ``M_k = (2 v_k / (λ d_k PR_k))^{PR_k/(2+PR_k)}``; ``λ`` is bisected in log space.
    ⭐ If this is within seed spread of ``uniform``, **uniform M is already optimal
    and re-allocation is not a lever** — the pre-registered NULL reading for D6.
    """
    PR = np.asarray(pr, np.float64)
    d = np.asarray(well_sizes, np.int64)
    v = np.asarray(var, np.float64)
    budget = _budget_words(d, m_uniform)

    def words(lam: float) -> np.ndarray:
        m = np.power(2.0 * v / (float(lam) * d.astype(np.float64) * PR), PR / (2.0 + PR))
        return np.clip(np.round(m), m_min, m_max).astype(np.int64)

    lo, hi = 1e-15, 1e15
    for _ in range(400):
        mid = float(np.sqrt(lo * hi))
        if int((words(mid) * d).sum()) > budget:
            lo = mid
        else:
            hi = mid
    m = words(hi)
    if int((m * d).sum()) > budget:
        return m
    return _greedy_topup(
        m, d, budget,
        # best marginal variance-weighted error reduction PER BYTE
        gain=lambda k, mk: (
            float(v[k]) * (predicted_distortion(PR[k], mk)
                           - predicted_distortion(PR[k], mk + 1)) / float(d[k])
        ),
    )


def build_allocation(
    name: str,
    pr: Sequence[float],
    well_sizes: Sequence[int],
    var: Sequence[float],
    m_uniform: int = 64,
    m_min: int = 2,
    m_max: int = 100_000,
) -> np.ndarray:
    """Dispatch on the allocation name.  ⛔ Unknown names raise rather than default."""
    if name == "uniform":
        return allocate_uniform(well_sizes, m_uniform)
    if name == "equal_distortion":
        return allocate_equal_distortion(pr, well_sizes, m_uniform, m_min, m_max)
    if name == "mse_optimal":
        return allocate_mse_optimal(pr, well_sizes, var, m_uniform, m_min, m_max)
    raise ValueError(f"unknown allocation {name!r} — have {ALLOCATIONS}")


# ---------------------------------------------------------------------------
# 4. ⛔ THE BYTE GATE — this spoke's own falsifier #1
# ---------------------------------------------------------------------------


def allocation_ledger(
    name: str,
    m: Sequence[int],
    well_sizes: Sequence[int],
    pr: Sequence[float],
    var: Sequence[float],
    m_uniform: int = 64,
) -> dict:
    """Everything about one allocation that a reviewer must be able to check."""
    M = np.asarray(m, np.int64)
    d = np.asarray(well_sizes, np.int64)
    ref = _budget_words(d, m_uniform) * 4
    got = int((M * d).sum()) * 4
    return {
        "allocation": str(name),
        "n_modes_per_well": [int(x) for x in M],
        "well_sizes": [int(x) for x in d],
        "participation_ratio": [float(x) for x in pr],
        "total_variance": [float(x) for x in var],
        "store_bytes": int(got),
        "uniform_reference_bytes": int(ref),
        "bytes_delta": int(got - ref),
        "bytes_shortfall": int(ref - got),
        "bytes_shortfall_frac": float((ref - got) / ref) if ref else float("nan"),
        "bits_per_read": float(np.log2(M.astype(np.float64)).sum()),
        "uniform_bits_per_read": float(len(M) * np.log2(float(m_uniform))),
        "onehot_width": int(M.sum()),
        "uniform_onehot_width": int(len(M) * int(m_uniform)),
        "predicted_eps_per_well": [float(x) for x in predicted_distortion_vec(pr, M)],
        "predicted_variance_weighted_eps": variance_weighted_distortion(pr, M, var),
    }


def assert_bytes_not_exceeded(ledgers: Sequence[dict], reference: str = "uniform") -> dict:
    """⛔⛔ **THE GATE.**  No arm may use more bytes than the uniform arm.

    ⭐ The task's falsifier #1 is *"total bytes are not exactly equal ⇒ it is a
    bigger store, not an allocation."*  With integer ``M_k`` and unequal ``d_k``,
    ``Σ M_k d_k = M·D`` **exactly** is a Diophantine problem with no guaranteed
    solution, so the constraint that is actually enforceable — and pre-registered —
    is ``bytes ≤ bytes(uniform)``, **never ``>``**, with the exact shortfall on the
    row.  A *smaller* store cannot be "a bigger store", so the falsifier's intent is
    preserved and the allocated arm is, if anything, handicapped.
    """
    by = {str(x["allocation"]): x for x in ledgers}
    if reference not in by:
        raise ValueError(f"no {reference!r} arm in the ledgers — the control is mandatory")
    ref = int(by[reference]["store_bytes"])
    rows = {}
    for nm, led in by.items():
        got = int(led["store_bytes"])
        rows[nm] = {
            "store_bytes": got,
            "delta_vs_reference": int(got - ref),
            "shortfall_frac": float((ref - got) / ref) if ref else float("nan"),
            "within_budget": bool(got <= ref),
        }
        if got > ref:
            raise AssertionError(
                f"⛔ allocation {nm!r} uses {got:,} B against the {reference!r} arm's "
                f"{ref:,} B (+{got - ref:,} B) — that is a BIGGER STORE, not a "
                "re-allocation, and the comparison is VOID"
            )
    worst = max(rows.values(), key=lambda r: r["shortfall_frac"])
    return {
        "reference_arm": str(reference),
        "reference_bytes": ref,
        "per_arm": rows,
        "max_shortfall_frac": float(worst["shortfall_frac"]),
        "verdict": "ALL ARMS WITHIN THE UNIFORM ARM'S BYTES",
        "note": (
            "⭐ constant total bytes is enforced as `<=`, never `==`: integer M_k at "
            "unequal d_k cannot in general hit the budget exactly. The shortfall is "
            "on every row and the allocated arms are handicapped, never helped."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Per-well M plumbing the shipped helpers assume is scalar
# ---------------------------------------------------------------------------


def onehot_codes_per_well(codes: np.ndarray, n_modes_per_well: Sequence[int]) -> np.ndarray:
    """``(N, K)`` codes -> ``(N, Σ_k M_k)`` one-hot, wells concatenated.

    ⭐ The per-well generalisation of
    :func:`chlu.eval.gen_attribution.onehot_codes`, which assumes one ``M``.
    ⚠ **The probe's input width is therefore an ARM-DEPENDENT quantity** (uniform
    ``K·M = 320``; an equal-distortion arm is typically *narrower*).  That is a
    capacity confound in either direction and is why every arm also gets a
    random-code control **at its own widths**.
    """
    C = np.asarray(codes, np.int64)
    if C.ndim != 2:
        raise ValueError(f"codes must be (N, K), got {C.shape}")
    ms = [int(x) for x in n_modes_per_well]
    n, k = C.shape
    if k != len(ms):
        raise ValueError(f"codes have {k} wells but n_modes_per_well has {len(ms)}")
    off = np.concatenate([[0], np.cumsum(ms)]).astype(np.int64)
    if C.size:
        for j in range(k):
            lo, hi = int(C[:, j].min()), int(C[:, j].max())
            if lo < 0 or hi >= ms[j]:
                raise ValueError(
                    f"well {j}: codes out of range [0, {ms[j]}) — saw [{lo}, {hi}]"
                )
    out = np.zeros((n, int(off[-1])), np.float32)
    if n:
        rows = np.repeat(np.arange(n, dtype=np.int64), k)
        cols = (off[None, :k] + C).ravel()
        out[rows, cols] = 1.0
    return out


def random_codes_per_well(
    n: int, n_modes_per_well: Sequence[int], seed: int
) -> np.ndarray:
    """⭐ The mandatory control at **this arm's own** widths: same code shape, zero
    information.  ⛔ A uniform-``M`` random control would not bound an allocated
    arm's probe."""
    rng = np.random.default_rng(int(seed))
    ms = [int(x) for x in n_modes_per_well]
    return np.stack([rng.integers(0, m, size=int(n), dtype=np.int64) for m in ms], axis=1)


def mask_posterior_to_allocation(
    posterior: np.ndarray, n_modes_per_well: Sequence[int]
) -> np.ndarray:
    """⛔ **Required before :func:`chlu.core.gen_compose.compose_codes` on an
    allocated store.**

    :func:`chlu.core.gen_dream.fit_address_book` builds a **rectangular** ``(K, M)``
    table with ``M = max_k M_k`` and Laplace smoothing, so a well with ``M_k < M``
    gets a small but **non-zero** posterior on modes it does not own — and
    ``argmax`` can then emit an **out-of-range code**, which
    :func:`chlu.core.gen_store.read_from_codes` would silently gather as a wrapped
    index.  This sets those entries to ``0`` and renormalises.
    """
    P = np.asarray(posterior, np.float64).copy()
    ms = [int(x) for x in n_modes_per_well]
    if P.ndim != 3 or P.shape[1] != len(ms):
        raise ValueError(f"posterior must be (N, K={len(ms)}, M), got {P.shape}")
    for j, m in enumerate(ms):
        if m < P.shape[2]:
            P[:, j, m:] = 0.0
    s = P.sum(axis=-1, keepdims=True)
    P = np.divide(P, s, out=np.zeros_like(P), where=s > 0)
    return P.astype(np.float32)


def mode_usage_per_well(codes: np.ndarray, n_modes_per_well: Sequence[int]) -> dict:
    """Per-well mode occupancy at per-well ``M_k`` — a dead codebook is a silent
    failure, and an **over-allocated** well is exactly how one is created."""
    C = np.asarray(codes, np.int64)
    ms = [int(x) for x in n_modes_per_well]
    out = []
    for j, m in enumerate(ms):
        c = np.bincount(C[:, j], minlength=m).astype(np.float64)
        p = c / max(c.sum(), 1.0)
        nz = p[p > 0]
        out.append({
            "well": int(j),
            "n_modes": int(m),
            "used_modes": int((c > 0).sum()),
            "used_fraction": float((c > 0).sum() / m) if m else float("nan"),
            "entropy_bits": float(-(nz * np.log2(nz)).sum()),
            "max_bits": float(np.log2(m)) if m > 0 else float("nan"),
        })
    return {
        "per_well": out,
        "total_used_modes": int(sum(o["used_modes"] for o in out)),
        "total_allocated_modes": int(sum(ms)),
        "mean_used_fraction": float(np.mean([o["used_fraction"] for o in out])),
        "realised_bits": float(sum(o["entropy_bits"] for o in out)),
    }


# ---------------------------------------------------------------------------
# 6. ⭐ The fit — "report the fit, not just the win" (acceptance 5)
# ---------------------------------------------------------------------------


def per_well_distortion(store, X: np.ndarray, codes: np.ndarray) -> dict:
    """⭐ **Measured** per-well relative distortion ``E‖z_k − ẑ_k‖² / E‖z_k − μ_k‖²``.

    This is the quantity the law predicts as ``M_k^(−2/PR_k)``, measured **per well**
    rather than aggregated — ⛔ the aggregate law was already confirmed by
    `gen-variance-equalisation`; the per-well one is the untested part.
    """
    Xf = np.asarray(X, np.float32)
    Xf = Xf.reshape((Xf.shape[0], -1))
    C = np.asarray(codes, np.int64)
    rows = []
    for j, (w, mo) in enumerate(zip(store.well_channels, store.modes, strict=True)):
        z = np.asarray(Xf[:, np.asarray(w, np.int64)], np.float64)
        zh = np.asarray(np.asarray(mo)[C[:, j]], np.float64)
        err = float(np.mean(np.sum((z - zh) ** 2, axis=1)))
        var = float(np.mean(np.sum((z - z.mean(axis=0)) ** 2, axis=1)))
        rows.append({
            "well": int(j),
            "d_k": int(len(w)),
            "n_modes": int(np.asarray(mo).shape[0]),
            "sq_error": err,
            "total_variance": var,
            "relative_distortion": float(err / var) if var > 0 else float("nan"),
        })
    tv = float(sum(r["total_variance"] for r in rows))
    return {
        "per_well": rows,
        "variance_weighted_relative_distortion": float(
            sum(r["sq_error"] for r in rows) / tv) if tv > 0 else float("nan"),
        "n_rows": int(Xf.shape[0]),
    }


def law_fit(predicted: Sequence[float], measured: Sequence[float]) -> dict:
    """⭐ OLS of ``log ε_measured`` on ``log ε_predicted`` + the ratio distribution.

    ⛔ Reported whether or not any arm wins: acceptance 5 asks for **the fit**, and a
    ratio table alone cannot say whether the law's *exponent* is right.
    """
    p = np.asarray(predicted, np.float64)
    m = np.asarray(measured, np.float64)
    ok = np.isfinite(p) & np.isfinite(m) & (p > 0) & (m > 0)
    p, m = p[ok], m[ok]
    if p.size < 2:
        return {"n_points": int(p.size), "status": "too few finite points to fit"}
    lp, lm = np.log(p), np.log(m)
    A = np.stack([lp, np.ones_like(lp)], axis=1)
    coef, *_ = np.linalg.lstsq(A, lm, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(((lm - pred) ** 2).sum())
    ss_tot = float(((lm - lm.mean()) ** 2).sum())
    r = float(np.corrcoef(lp, lm)[0, 1]) if p.size > 1 else float("nan")
    ratio = m / p
    return {
        "n_points": int(p.size),
        "slope": slope,
        "intercept": intercept,
        "multiplicative_offset": float(np.exp(intercept)),
        "pearson_r_log": r,
        "r2_log": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "ratio_measured_over_predicted": {
            "median": float(np.median(ratio)),
            "min": float(ratio.min()),
            "max": float(ratio.max()),
            "frac_within_half_to_double": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
        },
        "reading": (
            "slope 1 + offset 1 would be the law exactly; slope < 1 means the measured "
            "distortion is FLATTER in M than M^(-2/PR) predicts, i.e. extra codewords "
            "buy LESS than the law promises."
        ),
    }


def allocation_summary_row(
    led: dict, measured: Optional[dict] = None
) -> dict:
    """One compact row per arm: bytes, bits, predicted ε, measured ε."""
    row = {
        "allocation": led["allocation"],
        "n_modes_per_well": led["n_modes_per_well"],
        "store_bytes": led["store_bytes"],
        "bytes_delta_vs_uniform": led["bytes_delta"],
        "bits_per_read": led["bits_per_read"],
        "onehot_width": led["onehot_width"],
        "predicted_variance_weighted_eps": led["predicted_variance_weighted_eps"],
    }
    if measured is not None:
        row["measured_variance_weighted_eps"] = measured[
            "variance_weighted_relative_distortion"]
        row["measured_eps_per_well"] = [
            r["relative_distortion"] for r in measured["per_well"]]
    return row
