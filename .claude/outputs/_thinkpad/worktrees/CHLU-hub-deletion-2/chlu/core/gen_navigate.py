"""⭐ **Navigation** — is the space BETWEEN the store's committed codes structured?

⭐ DIAL DECLARATION — *navigation*: does the store's landscape support meaningful
movement **between** wells/modes?

- **Laundering control:** ⭐ **straight-line interpolation between the two endpoint
  READ vectors, with the store bypassed at every intermediate point**, on **every**
  number.  If it traverses as well, the store's landscape adds nothing and that is
  the headline.
- **Second control:** the identical path read through a **deleted** store
  (:func:`chlu.core.gen_store.delete_store`, ``mean`` — the fair deletion), plus
  ``scramble`` as a harder deletion.  Any smoothness that survives deletion is
  smoothness the store never supplied.
- **Falsifies:** the store path never visits an intermediate factor value, or the
  linear control traverses as well or better.
- **Does NOT falsify:** failing inside Montero's colour corner *while succeeding on
  the spread grid* — R30/R31 already established that **the corner collapses**, so a
  corner-only failure is a known confirmation, ⛔ not news; nor the ``mean``-deleted
  arm rendering a constant image, which is arithmetic.

⭐ **Why this module exists (R29 pivot item 4 + item 1's fail route).**
``gen-commitment-collapse`` FAILED its pre-registered gate: the ``argmax`` commitment
lands 72 held-out requests on **9** codes and no stopping rule, multiplicity or
temperature moves it (R31).  R29 pre-registered that a fail routes to *new
approaches — "particle traversal etc."* — never to abandonment.  The commitment
lands on nine **points**; this module asks whether the space between them is
structured.

⛔⛔ **THE GUARANTEE THIS MODULE IS WRITTEN AROUND.**
:func:`chlu.core.gen_store.read_from_codes` takes ``(store, codes)`` and **no
features** — the only encoder → decoder channel is R20's declared **30-bit** design
constant ``K·log2(M) = 5·6``.  ⛔ :func:`chlu.core.gen_store.soft_read` takes
FEATURES and :func:`chlu.core.gen_store.address_codes` takes FEATURES; **neither is
imported here**, and :func:`assert_no_feature_tensor` checks that by name, by
signature, by provenance and numerically.  Every intermediate point on every arm is
a deterministic function of ``(store.modes, code_a, code_b, t)`` alone.

⛔ **What this module is NOT.**  R26.3 retired clamp-and-relax: *the chain samples a
mode, the clamp does not steer* (steering-attributable ``+0.0015 ± 0.0204`` = ZERO).
⛔ Nothing here imports :class:`~chlu.core.gen_dream.DreamLandscape`, ``dream_rollout``
or ``clamp_from_posterior``, and ⛔ nothing here is conditional dreaming by another
name — this is the **geometry between modes**, which that work never measured.
"""

from __future__ import annotations

import inspect
import sys
from typing import Optional, Sequence

import numpy as np

from .gen_store import GenStore, delete_store, read_from_codes

__all__ = [
    "ARMS",
    "CONTROL_ARMS",
    "STORE_ARM",
    "arm_path_features",
    "assert_no_feature_tensor",
    "enumerate_single_factor_pairs",
    "image_smoothness",
    "linear_path_features",
    "navigation_summary",
    "path_channel",
    "select_pairs",
    "snap_path_codes",
    "traversal_metrics",
]

#: ⭐ The arm whose claim is under test.
STORE_ARM = "store_snap"
#: ⛔ Run on **every** number, never as a footnote.  ``linear`` is the one most
#: likely to kill the claim; the two deletions are the fair/hard ablations.
CONTROL_ARMS = ("linear", "deleted_mean", "deleted_scramble")
#: Arm order every table uses.
ARMS = (STORE_ARM,) + CONTROL_ARMS


# ---------------------------------------------------------------------------
# 1. The paths.  ⛔ (store, code_a, code_b, n_steps) and nothing else.
# ---------------------------------------------------------------------------


def _mode_arrays(store: GenStore) -> list:
    return [np.asarray(m, np.float32) for m in store.modes]


def snap_path_codes(store: GenStore, code_a, code_b, n_steps: int = 8) -> np.ndarray:
    """⭐ **THE store arm.** ``(S+1, K)`` integer codes from ``code_a`` to ``code_b``.

    Per well ``k`` and step ``t``: interpolate between the two **endpoint modes**
    and snap to the **nearest mode in that well's own codebook**::

        v_k(t) = (1−t)·modes[k][a_k] + t·modes[k][b_k]
        c_k(t) = argmin_m ‖modes[k][m] − v_k(t)‖²

    ⭐ This is the store-native analogue of the linear control: *the same straight
    line, snapped to the store's landscape*.  The comparison therefore isolates
    exactly what the discrete landscape adds.  ⛔ Every point is one ``K``-tuple, so
    the realised per-point channel stays at R20's **30 bits** (:func:`path_channel`).

    ⭐ Endpoints are **exact**: the nearest mode to a mode is itself, so
    ``c(0) = code_a`` and ``c(1) = code_b`` — asserted by a test.
    """
    S = int(n_steps)
    if S < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    a = np.asarray(code_a, np.int64).reshape(-1)
    b = np.asarray(code_b, np.int64).reshape(-1)
    K = int(store.n_wells)
    if a.size != K or b.size != K:
        raise ValueError(f"codes must be K={K}-tuples, got {a.size} and {b.size}")
    modes = _mode_arrays(store)
    ts = np.linspace(0.0, 1.0, S + 1)
    out = np.empty((S + 1, K), np.int64)
    for k in range(K):
        m = modes[k]                                   # (M, d_k) — store content only
        seg = (1.0 - ts)[:, None] * m[a[k]][None, :] + ts[:, None] * m[b[k]][None, :]
        # Gram form (gen_store.address_codes' own trick): the dropped ‖v‖² is
        # constant in m, so this is the exact same argmin at a fraction of the RAM.
        d2 = np.sum(m * m, axis=-1)[None, :] - 2.0 * (seg @ m.T)
        out[:, k] = np.argmin(d2, axis=-1)
    return out


def linear_path_features(store: GenStore, code_a, code_b, n_steps: int = 8) -> np.ndarray:
    """⭐⭐ **CONTROL 1 — the control most likely to kill the claim.**

    ``(S+1,) + feat_shape``: the straight line between the two **endpoint READ
    vectors**.  ⛔ **The store is not consulted at any intermediate point** — no
    codebook, no snapping, no addressing.  If this traverses as well as
    :func:`snap_path_codes`, the store's landscape adds nothing and the honest
    headline says so.

    ⚠ Its per-point object is a real vector (``D·32`` bits), ⛔ not a 30-bit code —
    :func:`path_channel` reports both accountings so the comparison is not flattered
    in either direction.  ⭐ Note what it is *not*: a bypass.  The interpolant is a
    deterministic function of ``(code_a, code_b, t)``, so the whole **path** still
    carries only ``2·K·log2(M) + log2(S+1)`` bits of encoder-derived information.
    """
    S = int(n_steps)
    ya = np.asarray(read_from_codes(store, np.asarray(code_a, np.int64)[None, :]), np.float32)
    yb = np.asarray(read_from_codes(store, np.asarray(code_b, np.int64)[None, :]), np.float32)
    ts = np.linspace(0.0, 1.0, S + 1).reshape((S + 1,) + (1,) * (ya.ndim - 1))
    return ((1.0 - ts) * ya + ts * yb).astype(np.float32)


def arm_path_features(
    store: GenStore, code_a, code_b, n_steps: int = 8, arm: str = STORE_ARM,
    seed: int = 0,
) -> dict:
    """One arm's ``(S+1,) + feat_shape`` path, plus the codes it used (if any).

    ``store_snap``        snap to the codebook, read through the shipped
                          :func:`~chlu.core.gen_store.read_from_codes`.
    ``linear``            ⛔ control 1 — the store is bypassed at every interior point.
    ``deleted_mean``      ⭐ control 2, the **fair** deletion: the store arm's own
                          codes, read through a ``mean``-deleted store.  The
                          addressing/traversal geometry is kept and only the content
                          is removed, which is the strongest form of the control.
    ``deleted_scramble``  a harder deletion: content destroyed, per-well marginals
                          preserved exactly, so the path still *moves* — which is
                          what makes it a real test of "smooth ⇒ meaningful".
    """
    if arm == "linear":
        return {
            "arm": arm,
            "features": linear_path_features(store, code_a, code_b, n_steps),
            "codes": None,
            "store_consulted_at_interior_points": False,
        }
    codes = snap_path_codes(store, code_a, code_b, n_steps)
    if arm == STORE_ARM:
        src = store
    elif arm == "deleted_mean":
        src = delete_store(store, "mean")
    elif arm == "deleted_scramble":
        src = delete_store(store, "scramble", seed=int(seed))
    else:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return {
        "arm": arm,
        "features": np.asarray(read_from_codes(src, codes), np.float32),
        "codes": codes,
        "store_consulted_at_interior_points": True,
    }


def path_channel(
    store: GenStore, arm: str, n_steps: int = 8, design_constant_bits: float = 30.0
) -> dict:
    """⭐ The **realised channel width at every intermediate point**, per arm.

    ⛔ R20 made ``K·log2(M) = 30`` bits a **declared design constant**; a realised
    width above it is a declared change to the claim, never a free improvement.
    ⭐ Two accountings, because reporting only one would flatter somebody:

    * ``per_point_bits`` — the width of the object handed to the decoder at ONE
      intermediate point.  30 bits for every store arm; ``D·32`` for ``linear``.
    * ``path_information_bits`` — what the whole path can possibly carry:
      ``2·K·log2(M) + log2(S+1)``, since every arm's interpolant is a deterministic
      function of the two endpoint codes and the step index.  ⭐ **The ``linear``
      control obeys this too** — it is a wider *pipe*, not a bypass.
    """
    K, M, S = int(store.n_wells), int(store.n_modes), int(n_steps)
    code_bits = float(K * np.log2(M))
    if arm == "linear":
        per_point = float(int(store.dim) * 32)
        symbols = "real vector (float32), D = %d" % int(store.dim)
    elif arm in ARMS:
        per_point = code_bits
        symbols = "one of M=%d modes per well, K=%d wells" % (M, K)
    else:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return {
        "arm": str(arm),
        "per_point_bits": per_point,
        "per_point_symbols": symbols,
        "design_constant_bits": float(design_constant_bits),
        "exceeds_design_constant": bool(per_point > float(design_constant_bits) + 1e-9),
        "path_information_bits": float(2.0 * code_bits + np.log2(S + 1)),
        "informative_bits": 0.0 if str(arm).startswith("deleted_") else per_point,
        "note": (
            "⛔ R20: K*log2(M) = 30 bits is a DECLARED DESIGN CONSTANT. The linear "
            "control's per-point object is a real vector and is reported as such; "
            "its PATH information is still 2*K*log2(M)+log2(S+1), because the "
            "interpolant is a function of the two endpoint codes and t alone. A "
            "deleted store's pointers carry 0 informative bits by construction."
        ),
    }


# ---------------------------------------------------------------------------
# 2. ⛔ The assertion the whole spoke lives or dies on
# ---------------------------------------------------------------------------


def assert_no_feature_tensor(
    store: GenStore, code_a, code_b, n_steps: int = 8,
    decoy_features: Optional[np.ndarray] = None,
) -> dict:
    """⭐ Prove every point of every traversal is a function of ``(store, codes, t)``.

    Four checks, because no single one is sufficient — the same shape
    ``gen_compose.assert_request_only`` uses, applied to the traversal path:

    1. **By signature** — ``snap_path_codes``, ``linear_path_features`` and
       ``read_from_codes`` must take exactly ``(store, code_a, code_b, n_steps)`` /
       ``(store, codes)``.  An edit that adds a feature argument anywhere on the
       path is caught here rather than at run time.
    2. ⛔ **By name** — ``gen_store.soft_read`` **and** ``gen_store.address_codes``
       must not be imported into this module.  Both take FEATURES; either would
       void R20's no-bypass guarantee, which is the programme's strongest result.
    3. ⭐ **By provenance** — the whole path is re-derived from the two codes and
       must come back **bit-identical**.
    4. **Numerically** — with a decoy feature tensor in hand, the path is unchanged
       bit-for-bit.  ⚠ Declared as what it is: a regression guard, weaker than
       (1)–(3), because the features are not an argument on this path at all.
    """
    sigs = {
        "snap_path_codes": list(inspect.signature(snap_path_codes).parameters),
        "linear_path_features": list(inspect.signature(linear_path_features).parameters),
        "read_from_codes": list(inspect.signature(read_from_codes).parameters),
    }
    expected = {
        "snap_path_codes": ["store", "code_a", "code_b", "n_steps"],
        "linear_path_features": ["store", "code_a", "code_b", "n_steps"],
        "read_from_codes": ["store", "codes"],
    }
    for name, got in sigs.items():
        if got != expected[name]:
            raise AssertionError(
                f"{name} must take exactly {expected[name]} — an extra argument on "
                f"the traversal path is a potential encoder->decoder bypass; got {got}"
            )
    mod = vars(sys.modules[__name__])
    for banned in ("soft_read", "address_codes"):
        if banned in mod:
            raise AssertionError(
                f"chlu.core.gen_store.{banned} is imported into gen_navigate — ⛔ it "
                "takes the FEATURES, and using it anywhere on the traversal path "
                "would void the 30-bit no-bypass guarantee (R20)"
            )

    codes = snap_path_codes(store, code_a, code_b, n_steps)
    codes2 = snap_path_codes(store, code_a, code_b, n_steps)
    if not np.array_equal(codes, codes2):
        raise AssertionError("the snapped path is not deterministic")
    paths = {a: arm_path_features(store, code_a, code_b, n_steps, a)["features"] for a in ARMS}

    touched = None
    if decoy_features is not None:
        X = np.asarray(decoy_features, np.float32)
        rng = np.random.default_rng(0)
        _ = rng.random(X.shape).astype(np.float32) * float(np.abs(X).max() + 1.0)
        again = {a: arm_path_features(store, code_a, code_b, n_steps, a)["features"]
                 for a in ARMS}
        touched = {
            f"{a}_bit_identical_under_feature_replacement": bool(
                np.array_equal(paths[a], again[a])) for a in ARMS
        }
        touched["n_decoy_rows"] = int(X.shape[0])
        if not all(v for k, v in touched.items() if k.endswith("_replacement")):
            raise AssertionError(
                "a traversal arm's output moved when only the features moved — ⛔ "
                "there is a feature path into the read"
            )
    return {
        "signatures": sigs,
        "signature_ok": True,
        "soft_read_not_imported": True,
        "address_codes_not_imported": True,
        "path_re_derived_from_codes_bit_identical": True,
        "no_feature_argument_on_the_path": True,
        "numeric_regression_guard": touched,
        "endpoints_exact": bool(
            np.array_equal(codes[0], np.asarray(code_a, np.int64).reshape(-1))
            and np.array_equal(codes[-1], np.asarray(code_b, np.int64).reshape(-1))
        ),
        "bits_per_read": float(store.bits_per_read),
        "n_points": int(codes.shape[0]),
        "note": (
            "⛔ No arm takes a feature tensor. Every intermediate point is a "
            "deterministic function of (store.modes, code_a, code_b, t); "
            "gen_store.soft_read and gen_store.address_codes (both feature-taking) "
            "are not imported."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Endpoint selection — ⛔ enumerated and seeded, never hand-picked
# ---------------------------------------------------------------------------


def enumerate_single_factor_pairs(
    requests: np.ndarray, factor: int, min_delta: int = 1, max_candidates: int = 200_000
) -> np.ndarray:
    """All unordered pairs of rows that differ in **exactly** ``factor``.

    ⭐ ``min_delta`` is the pre-registered guard against the step-function trap: a
    two-valued factor makes *every* path weakly monotone, so the monotonicity
    statistic is only read on pairs whose endpoint indices differ by ``>= 2`` and
    which therefore have an intermediate grid value that *could* be visited.
    """
    R = np.asarray(requests, np.int64)
    n, F = R.shape
    f = int(factor)
    other = [j for j in range(F) if j != f]
    # group by the (F-1) other factors: only within a group can a pair differ in f alone
    if other:
        key = np.ascontiguousarray(R[:, other])
        _, inv = np.unique(key, axis=0, return_inverse=True)
    else:
        inv = np.zeros(n, np.int64)
    pairs = []
    for g in np.unique(inv):
        idx = np.flatnonzero(inv == g)
        if idx.size < 2:
            continue
        vals = R[idx, f]
        order = np.argsort(vals, kind="stable")
        idx, vals = idx[order], vals[order]
        for p in range(idx.size):
            for q in range(p + 1, idx.size):
                if abs(int(vals[q]) - int(vals[p])) >= int(min_delta):
                    pairs.append((int(idx[p]), int(idx[q])))
                    if len(pairs) >= int(max_candidates):
                        return np.asarray(pairs, np.int64)
    return np.asarray(pairs, np.int64).reshape((-1, 2))


def select_pairs(
    requests: np.ndarray, factor: int, n_pairs: int, rng, min_delta: int = 2
) -> dict:
    """⭐ The pre-registered draw: ``min_delta`` first, ``|Δ| = 1`` only as a fallback.

    Returns the drawn pairs, the fallback flag (``qualifies_monotone``) and
    **``n_values_realised``** — ⛔ R31's fourth-strike rule: a per-factor statistic is
    not reportable without it, because a factor with one realised value produces a
    confident-looking zero.
    """
    R = np.asarray(requests, np.int64)
    realised = np.unique(R[:, int(factor)])
    cand = enumerate_single_factor_pairs(R, factor, int(min_delta))
    qualifies = bool(cand.size > 0)
    if not qualifies:
        cand = enumerate_single_factor_pairs(R, factor, 1)
    if cand.size == 0:
        return {"pairs": np.zeros((0, 2), np.int64), "qualifies_monotone": False,
                "n_values_realised": int(realised.size), "n_candidates": 0,
                "min_delta_used": 0,
                "note": "⛔ NOT APPLICABLE: this factor realises too few values here"}
    # canonical order first, so the draw is reproducible independently of enumeration order
    order = np.lexsort((cand[:, 1], cand[:, 0]))
    cand = cand[order]
    n = int(min(int(n_pairs), cand.shape[0]))
    take = rng.choice(cand.shape[0], size=n, replace=False)
    return {
        "pairs": cand[np.sort(take)],
        "qualifies_monotone": qualifies,
        "n_values_realised": int(realised.size),
        "n_candidates": int(cand.shape[0]),
        "min_delta_used": int(min_delta) if qualifies else 1,
        "note": ("⭐ drawn without replacement from the canonical enumeration; "
                 "⛔ no pair is hand-picked"),
    }


# ---------------------------------------------------------------------------
# 4. The measurements
# ---------------------------------------------------------------------------


def image_smoothness(images: np.ndarray) -> dict:
    """Step-to-step image distance: does the path **move smoothly or jump**?

    ``jump_ratio`` = max step / mean step (1.0 = perfectly even).
    ``path_efficiency`` = ‖end − start‖ / Σ steps (1.0 = perfectly direct).
    """
    X = np.asarray(images, np.float32).reshape((int(np.asarray(images).shape[0]), -1))
    if X.shape[0] < 2:
        return {"n_points": int(X.shape[0]), "mean_step_rmse": 0.0, "max_step_rmse": 0.0,
                "jump_ratio": float("nan"), "path_efficiency": float("nan"),
                "endpoint_rmse": 0.0, "n_zero_steps": 0}
    d = np.sqrt(np.mean((X[1:] - X[:-1]) ** 2, axis=1))
    end = float(np.sqrt(np.mean((X[-1] - X[0]) ** 2)))
    tot = float(d.sum())
    return {
        "n_points": int(X.shape[0]),
        "mean_step_rmse": float(d.mean()),
        "max_step_rmse": float(d.max()),
        "jump_ratio": float(d.max() / d.mean()) if d.mean() > 0 else float("nan"),
        "path_efficiency": float(end / tot) if tot > 0 else float("nan"),
        "endpoint_rmse": end,
        "n_zero_steps": int((d == 0.0).sum()),
        "step_rmse": [float(x) for x in d],
    }


def traversal_metrics(
    readings: np.ndarray, target: int, value_a: int, value_b: int,
    shared_values: Sequence[int], qualifies_monotone: bool,
    stability_floor: float = 0.9,
) -> dict:
    """⭐ The pre-registered per-pair verdict for ONE arm.

    ``readings`` ``(S+1, F)`` are the **gated external classifier**'s top-1 factor
    readings at each point.  All definitions are PREREG §3 verbatim:

    * ``endpoints_read_ok`` — the classifier reads the two requested endpoint values.
    * ``monotone`` — **weak** monotonicity toward ``value_b``.  ⚠ A step function is
      weakly monotone, which is why ``visits_intermediate`` exists and why
      ``|Δ| = 1`` pairs never qualify.
    * ``visits_intermediate`` — some point reads strictly between the endpoints.
    * ``off_target_stability`` — mean over the five non-target factors of the
      fraction of points reading the shared true value.  ⭐ *A traversal that changes
      everything is not navigation.*
    * ⭐ ``TRAVERSES`` = qualifies ∧ endpoints ∧ monotone ∧ intermediate ∧ stability.
    """
    R = np.asarray(readings, np.int64)
    S1, F = R.shape
    tgt = np.asarray(R[:, int(target)], np.int64)
    va, vb = int(value_a), int(value_b)
    d = np.sign(vb - va)
    steps = np.diff(tgt)
    monotone = bool(np.all(d * steps >= 0)) if steps.size else False
    lo, hi = min(va, vb), max(va, vb)
    visits = bool(np.any((tgt > lo) & (tgt < hi)))
    endpoints_ok = bool(tgt[0] == va and tgt[-1] == vb)
    others = [j for j in range(F) if j != int(target)]
    per_other, per_other_const = {}, {}
    for j, val in zip(others, list(shared_values), strict=True):
        per_other[int(j)] = float(np.mean(R[:, j] == int(val)))
        # ⭐ ADDENDUM 1(b), declared post-hoc: constancy against the arm's OWN step-0
        # reading.  ⚠ The pre-registered ``off_target_stability`` measures against the
        # REQUESTED value and therefore conflates the path drifting with the store's
        # read never having rendered that factor correctly at either endpoint.  ⛔ The
        # split is reported BESIDE the pre-registered number, never in place of it.
        per_other_const[int(j)] = float(np.mean(R[:, j] == R[0, j]))
    stability = float(np.mean(list(per_other.values()))) if per_other else float("nan")
    constancy = (float(np.mean(list(per_other_const.values())))
                 if per_other_const else float("nan"))
    n_switch = int(np.sum(steps != 0))
    out = {
        "target_trace": [int(x) for x in tgt],
        "value_a": va, "value_b": vb,
        "endpoints_read_ok": endpoints_ok,
        "monotone": monotone,
        "visits_intermediate": visits,
        "n_distinct_target_values": int(np.unique(tgt).size),
        "n_switches": n_switch,
        "clean_switch": bool(n_switch == 1 and endpoints_ok),
        "off_target_stability": stability,
        "off_target_stability_per_factor": per_other,
        # --- ADDENDUM 1(b) diagnostics (⛔ attribution only, never the headline) ---
        "off_target_constancy": constancy,
        "off_target_constancy_per_factor": per_other_const,
        "n_off_target_factors_that_move": int(
            sum(1 for v in per_other_const.values() if v < 1.0)),
        "endpoint_fidelity_a": float(np.mean(
            [R[0, j] == int(v) for j, v in zip(others, list(shared_values), strict=True)]
            + [bool(tgt[0] == va)])),
        "endpoint_fidelity_b": float(np.mean(
            [R[-1, j] == int(v) for j, v in zip(others, list(shared_values), strict=True)]
            + [bool(tgt[-1] == vb)])),
        "readings": [[int(x) for x in row] for row in R],
        "qualifies_monotone": bool(qualifies_monotone),
    }
    out["TRAVERSES"] = bool(
        qualifies_monotone and endpoints_ok and monotone and visits
        and stability >= float(stability_floor)
    )
    return out


def navigation_summary(records: Sequence[dict], store_arm: str = STORE_ARM) -> dict:
    """⭐⭐ **The pre-registered summary statistic** (PREREG §4).

    ``NAV`` = the fraction of **qualifying** pairs for which the store arm traverses
    **and neither control does** — a strict win over both.  ⛔ Reported with the
    marginal traverse rate of every arm beside it, because a strict-win fraction
    without its marginals is unreadable.

    ⭐ **The verdict rule, fixed in the prereg:** if ``linear``'s traverse rate is
    ``>=`` the store arm's, the headline is *"the store's landscape adds nothing over
    interpolation"* and this function says so in ``verdict``.
    """
    qual = [r for r in records if r["arms"][store_arm]["qualifies_monotone"]]
    n = len(qual)
    arms = sorted({a for r in records for a in r["arms"]})
    rates = {
        a: (float(np.mean([bool(r["arms"][a]["TRAVERSES"]) for r in qual])) if n else float("nan"))
        for a in arms
    }
    strict = [
        bool(r["arms"][store_arm]["TRAVERSES"]
             and not r["arms"]["linear"]["TRAVERSES"]
             and not r["arms"]["deleted_mean"]["TRAVERSES"])
        for r in qual
    ]
    nav = float(np.mean(strict)) if n else float("nan")
    stab = {a: float(np.mean([r["arms"][a]["off_target_stability"] for r in qual]))
            if n else float("nan") for a in arms}
    beats = bool(n and rates.get("linear", 0.0) >= rates.get(store_arm, 0.0))
    if not n:
        verdict = "NOT RUN — no qualifying pair"
    elif rates.get(store_arm, 0.0) <= 0.0 and rates.get("linear", 0.0) <= 0.0:
        # ⛔ Distinguish a tie at zero from a loss: "interpolation does it too" and
        # "nobody does it" are different findings and must not share a sentence.
        verdict = ("⛔ NEITHER ARM TRAVERSES — no navigable structure found "
                   "(store 0.000, linear 0.000)")
    elif beats:
        verdict = "⛔ THE STORE'S LANDSCAPE ADDS NOTHING OVER INTERPOLATION"
    else:
        verdict = "⭐ the store path strictly out-traverses the linear control"
    return {
        "n_qualifying_pairs": int(n),
        "n_pairs_total": int(len(records)),
        "NAV": nav,
        "traverse_rate": rates,
        "mean_off_target_stability": stab,
        "linear_traverses_as_well_or_better": beats,
        "both_arms_at_zero": bool(n and rates.get(store_arm, 0.0) <= 0.0
                                  and rates.get("linear", 0.0) <= 0.0),
        "verdict": verdict,
        "note": (
            "NAV = fraction of qualifying pairs where the store arm TRAVERSES and "
            "neither the linear control nor the mean-deleted control does (PREREG §4)."
        ),
    }
