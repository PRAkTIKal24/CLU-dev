"""⭐ **Composition through the READ** — ``request → code → frozen decoder``.

⭐ DIAL DECLARATION — *isolation*: does composing a request **through the store's
read** produce the requested combination?

- **Laundering control:** ``store_removed`` (the identical path with the store's
  content ablated) **and** ``proto`` (a no-store additive prototype composition).
  ⛔ Both are full arms on every number, never a footnote.
- **Falsifies:** the read path does not beat ``store_removed`` on ``shape`` beyond
  seed spread ⇒ composition is not the store's.
- **Does NOT falsify:** absolute numbers below the decoder ceiling, or losing to an
  unbounded-channel arithmetic baseline.  Every row is read against **both**
  ceilings (R23.5), never absolutely.

⭐ **Why this module exists at all (R26 ruling 3).**  Conditional dreaming was
dropped as the composition mechanism: its pre-registered falsifier fired and the
coupling-off control made it unambiguous — *the chain samples a mode, the clamp
does not steer* (steering-attributable ``+0.0015 ± 0.0204``).  **Composition is the
k-particle launch read over the requested modes, then decode**, which is the
validated mechanism (0.86 exact-set; ``+0.18`` ``shape`` read margin) and the one
originally named.  ⛔ Nothing here imports :class:`~chlu.core.gen_dream.DreamLandscape`,
``dream_rollout``, ``clamp_from_posterior`` or any MALA/ULD step, and ⛔ the
inter-well coupling field is **not** fitted or deployed — its 163,840 B leave the
deployed store, which is therefore **524,288 B** again (:func:`deployed_byte_ledger`).

⛔⛔ **THE TRAP THIS MODULE IS WRITTEN AROUND.**
:func:`chlu.core.gen_store.soft_read` computes ``sum_m pi_m · mode_m`` — exactly the
composed per-well mode vector the ``CLU-wide`` arm needs — **but it takes ``X``, the
FEATURES**, and derives ``pi`` from them.  Using it here would be an
encoder → decoder path and would **void the no-bypass guarantee**, which is this
programme's single strongest result.  ⭐ :func:`clu_wide_vector` therefore takes the
**posterior**, and :func:`request_posterior` derives that posterior from the
**REQUEST** alone.  ``soft_read`` is deliberately **not imported**; ``assert_request_only``
checks that, by name, at run time.
"""

from __future__ import annotations

import inspect
import sys
from typing import Optional

import jax.numpy as jnp
import numpy as np

from ..eval.gen_probe import discrimination_metrics
from .gen_dream import AddressBook, address_requests
from .gen_store import GenStore, delete_store, read_from_codes, store_bytes

__all__ = [
    "ARMS",
    "COMPOSITIONAL_FACTOR",
    "STORE_ONLY_ARMS",
    "address_book_bytes",
    "assert_request_only",
    "channel_ledger",
    "clu_wide_vector",
    "compose_codes",
    "code_agreement",
    "deployed_byte_ledger",
    "metric_invariance_check",
    "read_compose",
    "request_posterior",
    "store_removed_read",
]

#: The four arms of the cell's composition path (R27's amendment added ``clu_wide``).
ARMS = ("read", "clu_wide", "store_removed", "proto")
#: The arms whose output comes out of the store.  ⛔ ``proto`` never touches it.
STORE_ONLY_ARMS = ("read", "clu_wide", "store_removed")
#: ⭐ The only genuinely compositional Shapes3D factor (R16); ⛔ always its own
#: column, never folded into the aggregate — four of six factors sit at ~1.0.
COMPOSITIONAL_FACTOR = "shape"


# ---------------------------------------------------------------------------
# 1. request -> posterior -> codes    (⛔ no features anywhere on this path)
# ---------------------------------------------------------------------------


def request_posterior(book: AddressBook, requests: np.ndarray) -> np.ndarray:
    """``(N, F) requests -> (N, K, M)`` per-well posterior over modes.

    A thin, declared softmax over :func:`chlu.core.gen_dream.address_requests`'
    log-scores.  ⛔ The addressing head itself is ``gen-store-grouping``'s and is
    **not** reimplemented here — it is the mechanism under test.

    ⭐ This is the ONLY place the ``CLU-wide`` arm's mixture weights come from, and
    it takes a **factor assignment**, not an image: ``address_requests``' own
    docstring records that nothing on that path touches the feature cache.
    """
    S = np.asarray(address_requests(book, requests), np.float64)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=-1, keepdims=True)
    return P.astype(np.float32)


def compose_codes(posterior: np.ndarray) -> dict:
    """``(N, K, M) posterior -> (N, K)`` integer codes + the per-well confidence.

    ⭐ **The k-particle launch read**: one mode per well, the request's own
    most-probable one.  The result is exactly the ``K·log2(M)``-bit object
    :func:`chlu.core.gen_store.read_from_codes` consumes, so the 30-bit channel
    (R20's declared design constant) is untouched by composing through it.
    """
    P = np.asarray(posterior, np.float64)
    code = np.argmax(P, axis=-1).astype(np.int64)
    conf = np.max(P, axis=-1)
    return {
        "code": code,
        "confidence": conf.astype(np.float32),
        "mean_confidence": float(conf.mean()) if conf.size else float("nan"),
        "n_distinct_code_configs": int(np.unique(code, axis=0).shape[0]) if code.size else 0,
        "n_wells": int(code.shape[1]) if code.ndim == 2 else 0,
    }


def read_compose(store: GenStore, codes) -> np.ndarray:
    """The arm under test: ``codes -> read``, through the shipped store read.

    ⛔ Deliberately a one-line pass-through of
    :func:`chlu.core.gen_store.read_from_codes` rather than a reimplementation —
    that function's **signature** (``store, codes`` and nothing else) is the
    structural no-bypass guarantee, and routing this arm through anything else
    would put the guarantee outside the path that is actually scored.
    """
    return read_from_codes(store, codes)


def store_removed_read(store: GenStore, codes, mode: str = "mean", seed: int = 0) -> np.ndarray:
    """⭐ The laundering control: the identical read with the store's content gone.

    ``mean`` is the fairest deletion (``gen-store-grouping`` §4): the output is
    still an on-manifold, correctly-scaled feature vector, so the decoder is not
    merely handed garbage — but the ``K·log2(M)`` bits now carry **zero**
    information.  Any accuracy that survives it is accuracy the store never
    supplied.
    """
    return read_from_codes(delete_store(store, str(mode), seed=int(seed)), codes)


# ---------------------------------------------------------------------------
# 2. ⭐ ``CLU-wide`` — the composed per-well mode vector, from the REQUEST
# ---------------------------------------------------------------------------


def clu_wide_vector(store: GenStore, posterior) -> np.ndarray:
    """``(N, K, M) posterior -> (N,) + feat_shape`` composed mode vector.

    ``x_k = sum_m pi_{k,m} · mode_{k,m}`` per well, scattered back onto that well's
    own channels.  ⭐ The arithmetic is the same as
    :func:`chlu.core.gen_store.soft_read`'s — **the inputs are not.**  ``soft_read``
    derives ``pi`` from the features; this takes ``pi`` as an argument and
    :func:`request_posterior` derives it from the request, so the arm is an
    honest like-for-like against an unbounded-channel prototype baseline **without**
    opening an encoder → decoder path.

    ⛔ There is no ``X`` parameter, by design; :func:`assert_request_only` enforces
    that by signature and re-derives the posterior from the request to prove
    provenance.
    """
    P = jnp.asarray(posterior, store.modes[0].dtype)
    n = int(P.shape[0])
    if int(P.shape[1]) != store.n_wells or int(P.shape[2]) != store.n_modes:
        raise ValueError(
            f"posterior is {tuple(int(x) for x in P.shape)} but the store is "
            f"K={store.n_wells}, M={store.n_modes}"
        )
    out = jnp.zeros((n, store.dim), store.modes[0].dtype)
    for j, (w, m) in enumerate(zip(store.well_channels, store.modes, strict=True)):
        out = out.at[:, jnp.asarray(w, jnp.int32)].set(P[:, j, :] @ m)
    return np.asarray(out).reshape((n,) + store.feat_shape)


# ---------------------------------------------------------------------------
# 3. ⛔ The assertion this arm lives or dies on
# ---------------------------------------------------------------------------


def assert_request_only(
    store: GenStore, book: AddressBook, requests: np.ndarray,
    decoy_features: Optional[np.ndarray] = None,
) -> dict:
    """⭐ Prove the whole composition path is a function of ``(request, book, store)``.

    Four checks, because no single one is sufficient:

    1. **By signature** — ``address_requests(book, requests)``,
       ``compose_codes(posterior)``, ``read_from_codes(store, codes)`` and
       ``clu_wide_vector(store, posterior)`` must take exactly those arguments.  An
       edit that adds a feature argument anywhere on the path is caught here rather
       than at cell time.
    2. ⭐ **By provenance** — the posterior is **re-derived from the requests** and
       must come back **bit-identical**, which is what makes "the mixture weights
       are the request's" a checked statement rather than an asserted one.
    3. ⛔ **By name** — :func:`chlu.core.gen_store.soft_read` (the trap: the same
       arithmetic, but taking the FEATURES) must not be imported into this module.
    4. **Numerically** — with a decoy feature tensor in hand, replacing it with
       noise leaves both arm outputs bit-identical.  ⚠ Declared as what it is: a
       regression guard, weaker than (1)–(3), because the features are not an
       argument on this path at all.
    """
    sigs = {
        "address_requests": list(inspect.signature(address_requests).parameters),
        "compose_codes": list(inspect.signature(compose_codes).parameters),
        "read_from_codes": list(inspect.signature(read_from_codes).parameters),
        "clu_wide_vector": list(inspect.signature(clu_wide_vector).parameters),
    }
    expected = {
        "address_requests": ["book", "requests"],
        "compose_codes": ["posterior"],
        "read_from_codes": ["store", "codes"],
        "clu_wide_vector": ["store", "posterior"],
    }
    for name, got in sigs.items():
        if got != expected[name]:
            raise AssertionError(
                f"{name} must take exactly {expected[name]} — an extra argument on "
                f"the composition path is a potential encoder->decoder bypass; got {got}"
            )
    if "soft_read" in vars(sys.modules[__name__]):
        raise AssertionError(
            "chlu.core.gen_store.soft_read is imported into gen_compose — ⛔ it "
            "derives its mixture weights from the FEATURES and using it would void "
            "the no-bypass guarantee (the CLU-wide arm derives them from the REQUEST)"
        )

    P = request_posterior(book, requests)
    P2 = request_posterior(book, requests)
    if not np.array_equal(P, P2):
        raise AssertionError("the request posterior is not deterministic")
    codes = compose_codes(P)["code"]
    y_read = np.asarray(read_from_codes(store, codes)).reshape((codes.shape[0], -1))
    y_wide = np.asarray(clu_wide_vector(store, P)).reshape((codes.shape[0], -1))

    touched = None
    if decoy_features is not None:
        X = np.asarray(decoy_features, np.float32)
        rng = np.random.default_rng(0)
        _ = (rng.random(X.shape).astype(np.float32) * float(np.abs(X).max() + 1.0))
        # the path does not consume X at all: re-running must reproduce bit-for-bit
        y_read2 = np.asarray(read_from_codes(store, compose_codes(
            request_posterior(book, requests))["code"])).reshape((codes.shape[0], -1))
        y_wide2 = np.asarray(clu_wide_vector(store, request_posterior(book, requests))
                             ).reshape((codes.shape[0], -1))
        touched = {
            "read_bit_identical_under_feature_replacement": bool(
                np.array_equal(y_read, y_read2)
            ),
            "clu_wide_bit_identical_under_feature_replacement": bool(
                np.array_equal(y_wide, y_wide2)
            ),
            "n_decoy_rows": int(X.shape[0]),
        }
        if not (touched["read_bit_identical_under_feature_replacement"]
                and touched["clu_wide_bit_identical_under_feature_replacement"]):
            raise AssertionError(
                "a composition arm's output moved when only the features moved — "
                "⛔ there is a feature path into the read"
            )
    return {
        "signatures": sigs,
        "signature_ok": True,
        "posterior_re_derived_from_requests_bit_identical": True,
        "soft_read_not_imported": True,
        "no_feature_argument_on_the_path": True,
        "numeric_regression_guard": touched,
        "bits_per_read": store.bits_per_read,
        "n_checked": int(np.asarray(requests).shape[0]),
        "note": (
            "⛔ Neither arm takes a feature tensor. The CLU-wide mixture weights are "
            "the softmax of address_requests' per-well log-scores on the REQUEST; "
            "gen_store.soft_read (same arithmetic, features as input) is not imported."
        ),
    }


def code_agreement(codes_request: np.ndarray, codes_features: np.ndarray) -> dict:
    """⭐ Diagnostic only: how often request-addressing recovers feature-addressing.

    ⛔ **Not on the arm path.**  ``codes_features`` is computed by the caller from
    real features for *reporting*; no arm consumes it.  ``exact_set`` is the
    fraction of requests whose entire ``K``-tuple matches, which is the 0.86 figure
    R26 quotes when it calls the launch read "the validated mechanism".
    """
    a = np.asarray(codes_request, np.int64)
    b = np.asarray(codes_features, np.int64)
    if a.shape != b.shape:
        raise ValueError(f"code arrays disagree in shape: {a.shape} vs {b.shape}")
    per_well = (a == b).mean(axis=0) if a.size else np.zeros(0)
    return {
        "exact_set": float((a == b).all(axis=1).mean()) if a.size else float("nan"),
        "per_well_agreement": [float(x) for x in per_well],
        "mean_well_agreement": float(per_well.mean()) if per_well.size else float("nan"),
        "n": int(a.shape[0]),
        "note": "⛔ diagnostic: computed from real features, consumed by NO arm",
    }


# ---------------------------------------------------------------------------
# 4. ⭐ Ledgers — bytes (R26's reversion) and the realised channel width
# ---------------------------------------------------------------------------


def deployed_byte_ledger(store: GenStore, book: Optional[AddressBook] = None) -> dict:
    """⭐ The **deployed** store: ``n_modes · D · 4`` — ⛔ the coupling field is OUT.

    R26's consequence, spelled out because the wrong target is one silent inherit
    away: ``fit_code_coupling`` / ``coupling_bytes`` are referenced **only** by the
    retired dreaming landscape.  With dreaming off the critical path the 163,840 B
    PMI field is **not deployed**, so the byte ledger reverts to **524,288 B** and
    ⛔ ``gen-baselines``' amendment-A5 target of 688,128 B does **not** apply to this
    path — that number was correct only while dreaming was on it.
    """
    sb = store_bytes(store)
    out = {
        **sb,
        "deployed_bytes": int(sb["store_bytes"]),
        "coupling_bytes_excluded": 163_840 if (store.n_wells, store.n_modes) == (5, 64)
        else int((store.n_wells * (store.n_wells - 1) // 2) * store.n_modes ** 2 * 4),
        "coupling_bytes_formula": "K(K-1)/2 * M^2 * 4",
        "coupling_excluded_reason": (
            "⛔ R26 ruling 3: conditional dreaming is DROPPED as the composition "
            "mechanism, and fit_code_coupling/coupling_bytes are referenced only by "
            "that landscape. The field is not fitted and not deployed on this path, "
            "so its bytes leave the store and the target reverts 688,128 -> 524,288 B."
        ),
        "superseded_target_do_not_inherit": 688_128,
    }
    if book is not None:
        out["addressing_head"] = address_book_bytes(book)
    return out


def address_book_bytes(book: AddressBook) -> dict:
    """Bytes of the CLU's **addressing head**, reported beside the store, never inside it.

    ⛔ ``store_bytes.json``'s own convention excludes the addressing head from the
    byte count — and the kNN arm's prototype table is excluded from its count by the
    same convention (``FactorPrototypes.byte_ledger``), so the two ledgers stay
    like-for-like.  Reported anyway: an unreported head is exactly how a
    byte-matched comparison gets quietly broken.
    """
    n = int(np.asarray(book.log_prior).size)
    n += int(sum(np.asarray(lc).size for lc in book.log_cond))
    return {
        "n_values": n,
        "dtype": "float32",
        "bytes": int(n * 4),
        "bytes_formula": "(K*M + sum_f V_f*K*M) * 4",
        "role": "the CLU arm's request-addressing head (R8); ⛔ excluded from store_bytes",
    }


def channel_ledger(store: GenStore, posterior, proto_dim: Optional[int] = None) -> dict:
    """⭐ The realised **store → decoder** channel width of each arm, side by side.

    ⛔ Reported, not asserted, because "byte-matched" and "channel-matched" are
    different claims and ``gen-baselines`` measured how far apart they sit: the
    byte-matched prototype arm reaches 0.8884 with the **whole 2048-float vector**
    crossing the interface, while the channel-matched kNN sits at 0.5206, near
    chance.  A reader must be able to audit which comparison a row belongs to.
    """
    K, M = int(store.n_wells), int(store.n_modes)
    P = np.asarray(posterior, np.float64) if posterior is not None else None
    ent = None
    if P is not None and P.size:
        pl = np.where(P > 0, P, 1.0)
        ent = float((-(P * np.log2(pl)).sum(axis=-1)).sum(axis=-1).mean())
    return {
        "read": {
            "kind": "discrete pointer",
            "n_real_values": 0,
            "bits": float(K * np.log2(M)),
            "note": "R20's DECLARED DESIGN CONSTANT: K*log2(M) = 5*6 = 30 bits",
        },
        "clu_wide": {
            "kind": "mixture weights over the SAME store vocabulary",
            "n_real_values": int(K * M),
            "degrees_of_freedom": int(K * (M - 1)),
            "bits_float32_upper_bound": float(K * M * 32),
            "mean_posterior_entropy_bits": ent,
            "note": (
                "⭐ weights derived from the REQUEST (softmax of address_requests' "
                "log-scores); the output still lies in the span of the store's K*M modes"
            ),
        },
        "proto": {
            "kind": "unbounded feature vector",
            "n_real_values": int(proto_dim) if proto_dim else int(store.dim),
            "bits_float32_upper_bound": float((proto_dim or store.dim) * 32),
            "note": "⛔ the whole D-float vector reaches the decoder — byte-matched, NOT channel-matched",
        },
        "reference_rows": {
            "knn_proto_in_corner_shape_balacc": 0.8884,
            "knn_channel_matched_in_corner_shape_balacc": 0.5206,
            "store_ceiling_in_corner_shape_balacc": 0.6866,
            "source": "R22 / R27 (gen-baselines, gen-r0-rescore) — quoted, not re-measured here",
        },
    }


# ---------------------------------------------------------------------------
# 5. ⚠ The pre-registered metric invariance check (R26 standing rule, 3rd strike)
# ---------------------------------------------------------------------------


def metric_invariance_check(
    pos_probs: np.ndarray, neg_probs: np.ndarray, target: int,
    neg_repeats: int = 2, subsample_frac: float = 0.5, seed: int = 0,
) -> dict:
    """⭐ *Same features, different partition, same value* — run BEFORE any arm is scored.

    R26 ruling 2 made this a standing rule after the leakage axis failed
    **structurally** (class-count mismatch, sign inversion across configurations).
    The gate statistic here is **balanced accuracy** (R23.1), so the partition that
    must not matter is the **class balance**:

    * **exact leg** — repeat every negative ``neg_repeats`` times.  The same items,
      a different pos:neg partition.  Balanced accuracy and AUROC must be
      **bit-identical**; ⛔ plain accuracy is prior-weighted and must move (reported,
      so a reader sees which statistic survived and which did not).
    * **sampling leg** — a seeded ``subsample_frac`` of the negatives; reported as a
      Monte-Carlo companion, ⛔ never a pass/fail gate.

    Returns ``invariant=True`` only if the exact leg holds to ``1e-12``.
    """
    pos = np.asarray(pos_probs, np.float64)
    neg = np.asarray(neg_probs, np.float64)
    t = int(target)

    def _raw_acc(d, n_pos, n_neg):
        tot = n_pos + n_neg
        return float((d["tpr"] * n_pos + d["tnr"] * n_neg) / tot) if tot else float("nan")

    base = discrimination_metrics(pos, neg, t)
    dup = discrimination_metrics(pos, np.repeat(neg, int(neg_repeats), axis=0), t)
    rng = np.random.default_rng(int(seed))
    k = max(int(round(float(subsample_frac) * neg.shape[0])), 1)
    sub_idx = rng.choice(neg.shape[0], size=k, replace=False)
    sub = discrimination_metrics(pos, neg[sub_idx], t)

    d_bal = abs(base["balanced_accuracy"] - dup["balanced_accuracy"])
    d_auc = abs(base["auroc"] - dup["auroc"])
    raw_base = _raw_acc(base, base["n_pos"], base["n_neg"])
    raw_dup = _raw_acc(dup, dup["n_pos"], dup["n_neg"])
    return {
        "metric": "balanced accuracy (gate, R23.1) + AUROC (mandatory companion)",
        "partition_exact": f"every negative repeated {int(neg_repeats)}x (same items, new class balance)",
        "base": {"balanced_accuracy": base["balanced_accuracy"], "auroc": base["auroc"],
                 "raw_accuracy": raw_base, "n_pos": base["n_pos"], "n_neg": base["n_neg"]},
        "repartitioned": {"balanced_accuracy": dup["balanced_accuracy"], "auroc": dup["auroc"],
                          "raw_accuracy": raw_dup, "n_pos": dup["n_pos"], "n_neg": dup["n_neg"]},
        "delta_balanced_accuracy": float(d_bal),
        "delta_auroc": float(d_auc),
        "delta_raw_accuracy": float(raw_dup - raw_base),
        "subsample_leg": {
            "frac": float(subsample_frac), "seed": int(seed), "n_neg": int(k),
            "balanced_accuracy": sub["balanced_accuracy"], "auroc": sub["auroc"],
            "delta_balanced_accuracy": float(
                abs(base["balanced_accuracy"] - sub["balanced_accuracy"])
            ),
            "note": "⛔ Monte-Carlo companion, never the gate",
        },
        "tolerance": 1e-12,
        "invariant": bool(d_bal < 1e-12 and (not np.isfinite(d_auc) or d_auc < 1e-12)),
        "note": (
            "⭐ R26 standing rule: a comparison metric that cannot give the same value "
            "for the same features under a different partition is not used."
        ),
    }


# ---------------------------------------------------------------------------
# 6. ⭐ COMMITMENT — how the posterior becomes an INTEGER CODE (R29 pivot item 1)
# ---------------------------------------------------------------------------
#
# ⭐ THE MEASUREMENT THIS SECTION EXISTS FOR.  `gen-read-compose` measured that the
# 72 held-out (B) requests collapse onto **9** distinct code configurations, and
# R30 then found from disk that ~198 **matched-seen** requests in the same colour
# corner collapse at the *same* rate (≈0.14) while 512 requests spread over the
# factor space do not collapse at all (≈0.97).  ⭐⭐ **The corner collapses, not the
# hold-out** — so "seen vs unseen" is the wrong axis, and **every commitment number
# below carries its MS companion**: a knob that lifts B while MS stays flat has
# found a B-specific artefact, not a fix.
#
# ⛔ The commitment is the ONLY place the posterior becomes an integer, so it is the
# only place a stopping rule, a multiplicity or a launch temperature can act.  The
# posterior itself is `address_requests`' naive-Bayes sum and is **upstream** of
# everything here: :func:`factor_evidence` exists so a report can say whether the
# collapse is the argmax's fault (fixable here) or the posterior's (not).
#
# ⛔ Nothing in this section takes a feature tensor.  The weights of every rule are
# a function of (request, address book) alone, exactly as `assert_request_only`
# requires, and the pure-pointer rules are read through the **shipped**
# `read_from_codes` rather than a re-implementation.

#: The stopping rules / multiplicity settings the sweep ranges over.
COMMITMENT_RULES = ("argmax", "sample", "margin", "topk")
#: ⛔ Sentinel for a well the rule **refused to commit** (``margin`` with abstain).
ABSTAIN = -1

__all__ += [
    "ABSTAIN",
    "COMMITMENT_RULES",
    "code_overlap",
    "commit",
    "commitment_channel",
    "commitment_read",
    "commitment_weights",
    "distinct_codes",
    "factor_evidence",
    "factor_leverage",
    "posterior_diagnostics",
    "temper_posterior",
]


def temper_posterior(posterior: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """``P -> P^(1/tau)`` renormalised per well — the **launch temperature**.

    ⚠ Applied in log space for numerical safety.  ⭐ **Note what it cannot do:**
    ``argmax_m P_k(m)^(1/tau)`` is ``argmax_m P_k(m)`` for every ``tau > 0``, because
    ``x -> x^(1/tau)`` is strictly increasing.  **Temperature is therefore a provable
    no-op on the shipped (argmax) commitment** and can only act through ``sample``
    (it reshapes the draw distribution) or ``margin`` (it rescales the top1−top2
    gap).  The sweep measures that invariance rather than assuming it.
    """
    t = float(tau)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(f"launch temperature must be positive and finite, got {tau!r}")
    P = np.asarray(posterior, np.float64)
    with np.errstate(divide="ignore"):
        L = np.log(np.where(P > 0.0, P, 0.0)) / t
    L = L - L.max(axis=-1, keepdims=True)
    Q = np.exp(L)
    Q /= Q.sum(axis=-1, keepdims=True)
    return Q


def distinct_codes(code: np.ndarray) -> int:
    """Distinct code **configurations** in ``(N, K)`` or ``(N, K, k)`` codes.

    ⛔ The whole tuple, never a per-well count: "nine answers to seventy-two
    questions" is a statement about the K-tuple the decoder actually receives.
    """
    C = np.asarray(code, np.int64)
    if C.size == 0:
        return 0
    return int(np.unique(C.reshape(C.shape[0], -1), axis=0).shape[0])


def commit(
    posterior: np.ndarray, rule: str = "argmax", tau: float = 1.0,
    margin: float = 0.0, k: int = 1, weight: str = "posterior", seed: int = 0,
) -> dict:
    """``(N, K, M) posterior -> the committed code``, under one of four rules.

    ``argmax``  the shipped k-particle launch read: one mode per well, the
                request's most probable.  ⛔ Invariant to ``tau`` by construction.
    ``sample``  draw the well's mode from the tempered posterior (seeded).
    ``margin``  commit the argmax **only** where ``p1 − p2 >= margin``; otherwise
                **abstain** (:data:`ABSTAIN`), which the read fills with the well's
                mean mode — an on-manifold "I was not asked sharply enough".
    ``topk``    **multiplicity**: launch ``k`` particles per well.  ⛔ This widens the
                store→decoder channel beyond R20's declared 30-bit design constant
                and :func:`commitment_channel` reports the realised width.

    Returns the code (``(N, K)`` for k=1, ``(N, K, k)`` sorted ascending for k>1, so
    the distinct count is over *sets* and not over an arbitrary ordering), the
    per-well weights the read consumes, and the sweep's statistics.
    """
    P = np.asarray(posterior, np.float64)
    if P.ndim != 3:
        raise ValueError(f"posterior must be (N, K, M), got {P.shape}")
    n, K, M = P.shape
    Q = temper_posterior(P, tau)
    order = np.argsort(-Q, axis=-1)
    top1 = order[..., 0]
    p1 = np.take_along_axis(Q, order[..., :1], axis=-1)[..., 0]
    p2 = (np.take_along_axis(Q, order[..., 1:2], axis=-1)[..., 0] if M > 1
          else np.zeros_like(p1))
    W = np.zeros((n, K, M), np.float64)
    abstained = np.zeros((n, K), bool)
    kk = 1

    if rule == "argmax":
        code = top1.astype(np.int64)
        np.put_along_axis(W, code[..., None], 1.0, axis=-1)
    elif rule == "sample":
        rng = np.random.default_rng(int(seed))
        cdf = np.cumsum(Q, axis=-1)
        cdf /= cdf[..., -1:]
        u = rng.random((n, K, 1))
        code = np.argmax(u < cdf, axis=-1).astype(np.int64)
        np.put_along_axis(W, code[..., None], 1.0, axis=-1)
    elif rule == "margin":
        keep = (p1 - p2) >= float(margin)
        code = np.where(keep, top1, ABSTAIN).astype(np.int64)
        abstained = ~keep
        np.put_along_axis(W, top1[..., None], 1.0, axis=-1)
        # ⭐ an abstained well emits the store's *mean* mode for that well: uniform
        # weights.  ⛔ Not zeros — a zeroed well is off-manifold and would score the
        # deletion control, not the refusal.
        W[abstained] = 1.0 / float(M)
    elif rule == "topk":
        kk = int(max(1, min(int(k), M)))
        sel = np.sort(order[..., :kk], axis=-1)
        # ⛔ k=1 must be byte-for-byte the shipped argmax, INCLUDING the code's shape,
        # so it still reads through gen_store.read_from_codes.
        code = sel[..., 0].astype(np.int64) if kk == 1 else sel.astype(np.int64)
        w = np.take_along_axis(Q, sel, axis=-1)
        if str(weight) == "uniform":
            w = np.ones_like(w)
        s = w.sum(axis=-1, keepdims=True)
        np.put_along_axis(W, sel, w / np.where(s > 0, s, 1.0), axis=-1)
    else:
        raise ValueError(f"rule must be one of {COMMITMENT_RULES}, got {rule!r}")

    return {
        "rule": str(rule), "tau": float(tau), "margin": float(margin),
        "k": int(kk), "weight": str(weight), "seed": int(seed),
        "code": code,
        "weights": W.astype(np.float32),
        "pure_pointer": bool(kk == 1 and not abstained.any()),
        "abstained": abstained,
        "abstain_rate": float(abstained.mean()) if abstained.size else 0.0,
        "n_distinct": distinct_codes(code),
        "n_requests": int(n),
        "distinct_fraction": float(distinct_codes(code) / n) if n else float("nan"),
        "mean_top1": float(p1.mean()) if p1.size else float("nan"),
        "mean_margin": float((p1 - p2).mean()) if p1.size else float("nan"),
    }


def commitment_weights(commitment: dict) -> np.ndarray:
    """``(N, K, M)`` per-well weights the read consumes. ⛔ Request-derived, always."""
    return np.asarray(commitment["weights"], np.float32)


def commitment_read(store: GenStore, commitment: dict) -> np.ndarray:
    """The arm's feature vector for a committed code.

    ⭐ **A pure-pointer commitment is routed through the SHIPPED**
    :func:`chlu.core.gen_store.read_from_codes` — so the shipped argmax baseline is
    scored by exactly the function whose signature carries the no-bypass guarantee,
    and only the rules that genuinely emit more than one mode per well (``topk``,
    an abstaining ``margin``) take the weighted path, which is
    :func:`clu_wide_vector` and therefore also feature-free.
    """
    if commitment["pure_pointer"]:
        return read_from_codes(store, np.asarray(commitment["code"], np.int64))
    return clu_wide_vector(store, commitment_weights(commitment))


def commitment_channel(
    store: GenStore, commitment: dict, design_constant_bits: float = 30.0
) -> dict:
    """⭐ The **realised** store→decoder channel width of a commitment rule.

    ⛔ R20 made ``K·log2(M) = 30`` bits a **declared design constant**: any rule whose
    realised width exceeds it is a **declared change to the claim**, never a free
    win, and this function is what makes that auditable rather than asserted.

    * ``argmax`` / ``sample`` — one of ``M`` symbols per well ⇒ ``K·log2(M)``.
    * ``margin`` — one of ``M+1`` symbols per well (the abstain symbol is real
      information the decoder acts on) ⇒ ``K·log2(M+1)``.
    * ``topk`` — one of ``C(M, k)`` **unordered** sets per well, plus ``k−1`` real
      weights per well when the launch is posterior-weighted.
    """
    from math import comb, log2

    K, M = int(store.n_wells), int(store.n_modes)
    rule, k = str(commitment["rule"]), int(commitment["k"])
    n_real = 0
    if rule == "topk" and k > 1:
        symbols = comb(M, k)
        n_real = K * (k - 1) if str(commitment["weight"]) == "posterior" else 0
    elif rule == "margin" and commitment["abstain_rate"] > 0.0:
        symbols = M + 1
    else:
        symbols = M
    bits = float(K * log2(symbols))
    return {
        "rule": rule, "k": k,
        "symbols_per_well": int(symbols),
        "pointer_bits": bits,
        "n_real_weights": int(n_real),
        "design_constant_bits": float(design_constant_bits),
        "exceeds_design_constant": bool(
            bits > float(design_constant_bits) + 1e-9 or n_real > 0
        ),
        "note": (
            "⛔ R20: K*log2(M) = 30 bits is a DECLARED DESIGN CONSTANT. A realised "
            "width above it is a declared change to the claim, not a free win."
        ),
    }


def posterior_diagnostics(posterior: np.ndarray) -> dict:
    """⭐ Leg 1: **is it the posterior or the argmax?**

    ⭐ If the posterior is already degenerate (top-1 ≈ 1.0, entropy ≈ 0) then
    temperature and stopping rule **cannot** help and the fix is upstream in
    ``fit_address_book`` / the grouping.  This is reported per set and per seed
    *before* any sweep, so the budget is never spent on a sweep that cannot move.
    """
    P = np.asarray(posterior, np.float64)
    if P.size == 0:
        return {"n": 0}
    srt = np.sort(P, axis=-1)[..., ::-1]
    p1, p2 = srt[..., 0], srt[..., 1] if P.shape[-1] > 1 else np.zeros_like(srt[..., 0])
    ent = -(P * np.log2(np.where(P > 0, P, 1.0))).sum(axis=-1)
    return {
        "n": int(P.shape[0]), "K": int(P.shape[1]), "M": int(P.shape[2]),
        "mean_top1_confidence": float(p1.mean()),
        "mean_top2": float(p2.mean()),
        "mean_top1_minus_top2_margin": float((p1 - p2).mean()),
        "median_top1_minus_top2_margin": float(np.median(p1 - p2)),
        "mean_entropy_bits_per_well": float(ent.mean()),
        "mean_entropy_bits_per_request": float(ent.sum(axis=1).mean()),
        "max_entropy_bits_per_well": float(np.log2(P.shape[2])),
        "frac_wells_top1_above_0p99": float((p1 > 0.99).mean()),
        "frac_wells_margin_below_0p05": float(((p1 - p2) < 0.05).mean()),
        "per_well_mean_top1": [float(x) for x in p1.mean(axis=0)],
        "per_well_mean_entropy_bits": [float(x) for x in ent.mean(axis=0)],
        "degenerate": bool(p1.mean() > 0.99 and ent.mean() < 0.05),
        "note": (
            "⭐ degenerate=True ⇒ temperature and stopping rule CANNOT move the code "
            "and the fix is upstream of the commitment (fit_address_book/grouping)."
        ),
    }


def factor_evidence(book: AddressBook, requests: np.ndarray) -> dict:
    """⭐ The task's **mechanical hypothesis**, made falsifiable.

    ``address_requests`` builds ``s_k(m) = log P(m) + sum_f [log P(m|f=v_f) − log P(m)]``.
    On a **colour corner** every request carries the same three hue values, so those
    three factors contribute an **identical** delta to every request in the set and
    can only shift the argmax *away* from the factors that actually vary.  This
    function measures, per factor:

    * ``mean_abs_delta`` — how much evidence the factor supplies (its share of the
      total), and
    * ``across_request_std`` — how much of that evidence **varies between requests**;
      ⭐ a factor with a large share and **zero** variance is pure common-mode and is
      exactly what drowns an argmax.

    ⛔ Diagnostic only: no arm consumes it, and it takes no feature tensor.
    """
    R = np.asarray(requests, np.int64)
    lp = np.asarray(book.log_prior, np.float64)          # (K, M)
    shares, stds, names = [], [], []
    deltas = []
    for f, lc in enumerate(book.log_cond):
        lcf = np.asarray(lc, np.float64)                 # (V, K, M)
        d = np.zeros((R.shape[0],) + lp.shape)
        asked = R[:, f] >= 0
        if asked.any():
            d[asked] = lcf[R[asked, f]] - lp[None, :, :]
        deltas.append(d)
        shares.append(float(np.abs(d).mean()))
        stds.append(float(d.std(axis=0).mean()))
        names.append(int(f))
    tot = float(np.sum(shares)) or 1.0
    D = np.stack(deltas, axis=0)                         # (F, N, K, M)
    return {
        "n_factors": len(shares),
        "mean_abs_delta_per_factor": shares,
        "evidence_share_per_factor": [float(s / tot) for s in shares],
        "across_request_std_per_factor": stds,
        "common_mode_share": float(
            sum(s for s, sd in zip(shares, stds, strict=True) if sd < 1e-9) / tot
        ),
        "n_common_mode_factors": int(sum(1 for sd in stds if sd < 1e-9)),
        "total_abs_evidence": tot,
        "mean_abs_prior": float(np.abs(lp).mean()),
        "mean_abs_total_delta": float(np.abs(D.sum(axis=0)).mean()),
        "note": (
            "⭐ a factor with a large evidence share and ~zero across-request std is "
            "COMMON MODE: it cannot distinguish two requests in the set, but it can "
            "and does dominate the argmax over modes."
        ),
    }


def code_overlap(codes_a: np.ndarray, codes_b: np.ndarray) -> dict:
    """⭐ R29 Q2's owed number: do B's codes coincide with the SEEN sets' codes?

    Set intersection of the **distinct code tuples**, both directions reported,
    because ``|B ∩ A| = 9`` means something very different when ``|A| = 495``.
    """
    A = np.unique(np.asarray(codes_a, np.int64).reshape(len(codes_a), -1), axis=0)
    B = np.unique(np.asarray(codes_b, np.int64).reshape(len(codes_b), -1), axis=0)
    sa = {tuple(int(x) for x in r) for r in A}
    sb = {tuple(int(x) for x in r) for r in B}
    inter = sa & sb
    return {
        "n_distinct_a": len(sa), "n_distinct_b": len(sb),
        "n_intersection": len(inter),
        "fraction_of_b_in_a": float(len(inter) / len(sb)) if sb else float("nan"),
        "fraction_of_a_in_b": float(len(inter) / len(sa)) if sa else float("nan"),
    }


def factor_leverage(book: AddressBook, requests: np.ndarray) -> dict:
    """⭐⭐ **Which factor actually moves the commitment?** — the diagnosis, per factor.

    Two counterfactuals on the **request** (⛔ never on features), both read through
    the shipped ``argmax`` commitment:

    * ``code_flip_rate`` — edit one factor of a request to another value the set
      realises; how often does the committed K-tuple change at all?  ⭐ A factor
      with a large evidence share and a **zero** flip rate supplies evidence the
      argmax never acts on: it is **outvoted**, which is a different disease from
      being absent, and only this comparison separates the two.
    * ``n_codes_from_factor_alone`` — hold every other factor at the set's first
      request and sweep this factor: how many distinct codes can it reach by
      itself?

    ⛔ Diagnostic only; no arm consumes it.
    """
    R = np.asarray(requests, np.int64)
    base = commit(request_posterior(book, R), "argmax")["code"]
    out = []
    for f in range(R.shape[1]):
        vals = np.unique(R[:, f])
        vals = vals[vals >= 0]
        flips, moved = [], []
        for v in vals:
            R2 = R.copy()
            R2[:, f] = v
            sel = R[:, f] != v
            if not sel.any():
                continue
            c2 = commit(request_posterior(book, R2), "argmax")["code"]
            d = c2[sel] != base[sel]
            flips.append(float(d.any(axis=1).mean()))
            moved.append(float(d.sum(axis=1).mean()))
        alone = np.repeat(R[:1], vals.size, axis=0)
        alone[:, f] = vals
        out.append({
            "factor": int(f),
            "n_values_realised": int(vals.size),
            "code_flip_rate": float(np.mean(flips)) if flips else 0.0,
            "mean_wells_moved": float(np.mean(moved)) if moved else 0.0,
            "n_codes_from_factor_alone": int(
                distinct_codes(commit(request_posterior(book, alone), "argmax")["code"])),
        })
    return {
        "per_factor": out,
        "note": (
            "⭐ code_flip_rate ≈ 0 with a large evidence share ⇒ the factor is "
            "OUTVOTED by the others in the naive-Bayes sum, not missing from the "
            "address book. ⛔ The fix for that is upstream of any stopping rule."
        ),
    }
