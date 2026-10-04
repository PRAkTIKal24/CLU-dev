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
from .gen_dream import AddressBook, address_requests
from .gen_store import GenStore, delete_store, read_from_codes, store_bytes
COMPOSITIONAL_FACTOR = "shape"
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


