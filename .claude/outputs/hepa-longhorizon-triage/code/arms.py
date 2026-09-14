"""The tripwire arms: one matched-byte exemplar store, three trivial references.

Horizon conditioning (PREREG §2.4, binding on every arm here): dt enters as ONE
RAW SCALAR IN NATIVE UNITS. The exemplar store carries a scalar time-to-event
per exemplar and answers p(t, dt) = weighted fraction of retrieved neighbours
whose time-to-event is <= dt. There is no per-horizon table and no learned
horizon embedding anywhere in the compliant path.

The one exception is `ExemplarStore(payload="surface")`, a DIAGNOSTIC that
stores the full K-vector label surface per exemplar. That IS per-horizon free
capacity, it is flagged `prereg_compliant: false` everywhere it appears, and the
kill rule must not use it. It is reported because a suppressed reference is a
suppressed result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

import features as F
from byte_ledger import store_capacity


# ---------------------------------------------------------------------------
# Arm 1 -- matched-byte kNN exemplar store
# ---------------------------------------------------------------------------


@dataclass
class ExemplarStore:
    """kNN over window descriptors, sized to a byte budget.

    The seed controls (a) which exemplars survive the budget subsample and
    (b) tie-breaking in the neighbour argsort. When the pool already fits the
    budget the store is EXACTLY deterministic and the seed is inert -- callers
    must report that as `deterministic`, not as a spread of zero.
    """

    descriptor: str
    budget_bytes: int
    context_len: int
    norm_mode: str
    payload: str = "tte"            # "tte" (PREREG-compliant) | "surface" (diagnostic)
    n_horizons: int = 1

    # fitted state
    feats: np.ndarray = field(default=None, repr=False)
    mu: np.ndarray = field(default=None, repr=False)
    sd: np.ndarray = field(default=None, repr=False)
    tte: np.ndarray = field(default=None, repr=False)
    surface: Optional[np.ndarray] = field(default=None, repr=False)
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def payload_floats(self) -> int:
        return 1 if self.payload == "tte" else int(self.n_horizons)

    @property
    def prereg_compliant(self) -> bool:
        return self.payload == "tte"

    def fit(
        self,
        contexts: Sequence[np.ndarray],
        tte: np.ndarray,
        y_surface: Optional[np.ndarray],
        n_channels: int,
        seed: int,
    ) -> "ExemplarStore":
        d = F.feature_dim(self.descriptor, n_channels, self.context_len)
        cap = store_capacity(self.budget_bytes, d, self.payload_floats)
        n_pool = len(contexts)
        n_keep = min(cap["n_exemplars_budgeted"], n_pool)
        if n_keep < 1:
            raise ValueError(
                f"budget {self.budget_bytes} B cannot hold one {self.descriptor} "
                f"exemplar of dim {d} (+{self.payload_floats} payload floats)"
            )
        subsampled = n_keep < n_pool
        if subsampled:
            idx = np.sort(
                np.random.default_rng(seed).choice(n_pool, size=n_keep, replace=False)
            )
        else:
            idx = np.arange(n_pool)

        self.feats = F.build(
            [contexts[i] for i in idx], self.descriptor, self.context_len, self.norm_mode
        )
        self.mu = self.feats.mean(axis=0, keepdims=True)
        self.sd = self.feats.std(axis=0, keepdims=True) + 1e-6
        self.feats = ((self.feats - self.mu) / self.sd).astype(np.float32)
        self.tte = np.asarray(tte, dtype=np.float64)[idx]
        self.surface = (
            np.asarray(y_surface, dtype=np.float32)[idx] if self.payload == "surface" else None
        )
        self.info = {
            **cap,
            "descriptor": self.descriptor,
            "payload": self.payload,
            "prereg_compliant": self.prereg_compliant,
            "n_pool": int(n_pool),
            "n_exemplars_realised": int(n_keep),
            "store_bytes_realised": int(n_keep * cap["bytes_per_exemplar"]),
            "budget_utilisation": float(n_keep * cap["bytes_per_exemplar"] / self.budget_bytes),
            "subsampled": bool(subsampled),
            "seed_active": bool(subsampled),
            "seed": int(seed),
        }
        return self

    def predict_ks(
        self,
        contexts: Sequence[np.ndarray],
        horizons: List[int],
        ks: Sequence[int],
        chunk: int = 512,
    ) -> Dict[int, np.ndarray]:
        """{k: (N, K) probability surface} for several neighbour counts in one pass.

        dt enters only through `tte <= h` (payload="tte"), i.e. HEPA's
        scalar-concat conditioning: one raw scalar in native units, no
        per-horizon parameters.
        """
        n_store = self.feats.shape[0]
        ks = sorted({int(min(k, n_store)) for k in ks})
        k_max = max(ks)
        h = np.asarray(horizons, dtype=np.float64)[None, None, :]        # (1,1,K)
        out = {k: np.empty((len(contexts), len(horizons)), dtype=np.float64) for k in ks}
        e2 = (self.feats ** 2).sum(axis=1)[None, :]
        for s, Q in F.iter_chunks(
            contexts, self.descriptor, self.context_len, self.norm_mode, chunk
        ):
            Q = ((Q - self.mu) / self.sd).astype(np.float32)
            d2 = (Q ** 2).sum(axis=1)[:, None] + e2 - 2.0 * (Q @ self.feats.T)
            np.maximum(d2, 0.0, out=d2)
            if k_max < n_store:
                part = np.argpartition(d2, k_max - 1, axis=1)[:, :k_max]
            else:
                part = np.tile(np.arange(n_store), (d2.shape[0], 1))
            dpart = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(dpart, axis=1, kind="stable")             # ties -> lowest index
            nn_all = np.take_along_axis(part, order, axis=1)             # (B, k_max) sorted
            d_all = np.sqrt(np.take_along_axis(dpart, order, axis=1))
            B = nn_all.shape[0]
            if self.payload == "tte":
                hit_all = (self.tte[nn_all][:, :, None] <= h)            # (B,k_max,K)
            else:
                hit_all = self.surface[nn_all].astype(np.float64)        # (B,k_max,K)
            for k in ks:
                w = 1.0 / (d_all[:, :k] + 1e-6)
                w = w / w.sum(axis=1, keepdims=True)
                out[k][s : s + B] = np.einsum("bk,bkh->bh", w, hit_all[:, :k].astype(np.float64))
        return out

    def predict(
        self,
        contexts: Sequence[np.ndarray],
        horizons: List[int],
        k: int,
        chunk: int = 512,
    ) -> np.ndarray:
        """(N, K) probability surface for a single neighbour count."""
        res = self.predict_ks(contexts, horizons, [k], chunk)
        return res[min(int(k), self.feats.shape[0])]


# ---------------------------------------------------------------------------
# Arms 2-4 -- the trivial references
# ---------------------------------------------------------------------------


def prevalence_surface(y_train: np.ndarray, n_query: int) -> np.ndarray:
    """Per-horizon base rate from train, constant in t.

    Mechanically this scores every test row identically at a given horizon, so
    every non-degenerate horizon AUROC is exactly 0.5 and h-AUROC is exactly
    0.500. It is the metric's own floor and is reported as the sanity check that
    the harness is wired to the labels the way HEPA's is.
    """
    p = y_train.mean(axis=0)[None, :]
    return np.repeat(p.astype(np.float64), n_query, axis=0)


def seasonal_surface(
    y_train: np.ndarray,
    t_train: np.ndarray,
    t_query: np.ndarray,
    period: int,
    n_bins: int,
) -> np.ndarray:
    """Per-horizon base rate conditioned on phase within the dataset's period."""
    def _bin(t: np.ndarray) -> np.ndarray:
        return ((t % period) * n_bins // period).astype(np.int64)

    glob = y_train.mean(axis=0)
    table = np.repeat(glob[None, :], n_bins, axis=0)
    bt = _bin(np.asarray(t_train))
    for b in range(n_bins):
        m = bt == b
        if m.any():
            table[b] = y_train[m].mean(axis=0)
    return table[_bin(np.asarray(t_query))].astype(np.float64)


def last_event_age_surface(age_query: np.ndarray, n_horizons: int) -> np.ndarray:
    """Rank by time since the entity's previous event (see `_prev_event_index`).

    The score is constant across horizons; per-horizon AUROC is rank-based, so
    any strictly increasing function of age -- including a dt-dependent one --
    gives the identical h-AUROC. We therefore use the age itself, and note that
    this arm needs no model, no encoder, and no window at all.
    """
    return np.repeat(
        np.asarray(age_query, dtype=np.float64)[:, None], n_horizons, axis=1
    )
