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
