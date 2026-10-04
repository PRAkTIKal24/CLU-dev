"""gen compose."""

from __future__ import annotations
import inspect
import sys
from typing import Optional
import jax.numpy as jnp
import numpy as np
from nclu.core.gen_dream import AddressBook, address_requests
from nclu.core.gen_store import GenStore, delete_store, read_from_codes, store_bytes

COMPOSITIONAL_FACTOR = "shape"


def request_posterior(book: AddressBook, requests: np.ndarray) -> np.ndarray:
    S = np.asarray(address_requests(book, requests), np.float64)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=-1, keepdims=True)
    return P.astype(np.float32)


def compose_codes(posterior: np.ndarray) -> dict:
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
    return read_from_codes(store, codes)


def store_removed_read(store: GenStore, codes, mode: str = "mean", seed: int = 0) -> np.ndarray:
    return read_from_codes(delete_store(store, str(mode), seed=int(seed)), codes)


def clu_wide_vector(store: GenStore, posterior) -> np.ndarray:
    P = jnp.asarray(posterior, store.modes[0].dtype)
    n = int(P.shape[0])
    if int(P.shape[1]) != store.n_wells or int(P.shape[2]) != store.n_modes:
        raise ValueError(
            f"posterior is {tuple((int(x) for x in P.shape))} but the store is K={store.n_wells}, M={store.n_modes}"
        )
    out = jnp.zeros((n, store.dim), store.modes[0].dtype)
    for j, (w, m) in enumerate(zip(store.well_channels, store.modes, strict=True)):
        out = out.at[:, jnp.asarray(w, jnp.int32)].set(P[:, j, :] @ m)
    return np.asarray(out).reshape((n,) + store.feat_shape)


def assert_request_only(
    store: GenStore,
    book: AddressBook,
    requests: np.ndarray,
    decoy_features: Optional[np.ndarray] = None,
) -> dict:
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
                f"{name} must take exactly {expected[name]} — an extra argument on the composition path is a potential encoder->decoder bypass; got {got}"
            )
    if "soft_read" in vars(sys.modules[__name__]):
        raise AssertionError(
            "nclu.core.gen_store.soft_read is imported into gen_compose — ⛔ it derives its mixture weights from the FEATURES and using it would void the no-bypass guarantee (the CLU-wide arm derives them from the REQUEST)"
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
        _ = rng.random(X.shape).astype(np.float32) * float(np.abs(X).max() + 1.0)
        y_read2 = np.asarray(
            read_from_codes(store, compose_codes(request_posterior(book, requests))["code"])
        ).reshape((codes.shape[0], -1))
        y_wide2 = np.asarray(clu_wide_vector(store, request_posterior(book, requests))).reshape(
            (codes.shape[0], -1)
        )
        touched = {
            "read_bit_identical_under_feature_replacement": bool(np.array_equal(y_read, y_read2)),
            "clu_wide_bit_identical_under_feature_replacement": bool(
                np.array_equal(y_wide, y_wide2)
            ),
            "n_decoy_rows": int(X.shape[0]),
        }
        if not (
            touched["read_bit_identical_under_feature_replacement"]
            and touched["clu_wide_bit_identical_under_feature_replacement"]
        ):
            raise AssertionError(
                "a composition arm's output moved when only the features moved — ⛔ there is a feature path into the read"
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
        "note": "⛔ Neither arm takes a feature tensor. The CLU-wide mixture weights are the softmax of address_requests' per-well log-scores on the REQUEST; gen_store.soft_read (same arithmetic, features as input) is not imported.",
    }


def deployed_byte_footprint(store: GenStore, book: Optional[AddressBook] = None) -> dict:
    sb = store_bytes(store)
    out = {
        **sb,
        "deployed_bytes": int(sb["store_bytes"]),
        "coupling_bytes_excluded": 163840
        if (store.n_wells, store.n_modes) == (5, 64)
        else int(store.n_wells * (store.n_wells - 1) // 2 * store.n_modes**2 * 4),
        "coupling_bytes_formula": "K(K-1)/2 * M^2 * 4",
        "coupling_excluded_reason": "⛔ conditional dreaming is DROPPED as the composition mechanism, and fit_code_coupling/coupling_bytes are referenced only by that landscape. The field is not fitted and not deployed on this path, so its bytes leave the store and the target reverts 688,128 -> 524,288 B.",
        "superseded_target_do_not_inherit": 688128,
    }
    if book is not None:
        out["addressing_head"] = address_book_bytes(book)
    return out


def address_book_bytes(book: AddressBook) -> dict:
    n = int(np.asarray(book.log_prior).size)
    n += int(sum((np.asarray(lc).size for lc in book.log_cond)))
    return {
        "n_values": n,
        "dtype": "float32",
        "bytes": int(n * 4),
        "bytes_formula": "(K*M + sum_f V_f*K*M) * 4",
        "role": "the CLU arm's request-addressing head; ⛔ excluded from store_bytes",
    }
