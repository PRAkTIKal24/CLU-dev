"""⭐ **The dynamical read** — a continuous landscape over the **written atoms**.

``request -> address-book clamp + REQUEST ENERGY -> launch particles -> settle in a
per-well landscape over the stored TRAIN DATA POINTS with a LOW-RANK inter-well
coupling -> decode the SETTLED CONTINUOUS state``.

⭐ **DIAL DECLARATION — *the read*.**  Continuous settling over stored atoms,
replacing codebook copy-back.
- **Laundering control:** ⭐ **request-off on the SAME chain** — same launch, same
  key, same landscape, ``request_strength = 0``.  **Steering-attributable gain is
  the falsifier.**  ⛔ This is dreaming's exact lesson: `gen-dream-sampler`'s clamp
  produced an apparent gain that its ``no_coupling`` control took *entirely*
  (steering-attributable ``+0.0015 ± 0.0204`` = ZERO).
- **Falsifies:** the three falsifiers in this spoke's ``PREREG.md`` §5.
  ⛔ **This module renders no verdict**; the scored run does.
- **Does NOT falsify:** losing to ``knn_proto`` on a metric-native protocol — that
  is the metric-native-ceiling theorem, not news.

---

⛔ **THE BINDING CONSTRAINT (Advisor, 2026-09-14), and it removes the obvious design.**
The landscape must **NOT** take the ``M`` k-means centres as its minima, or the
dynamical read **inherits the codebook loss**: a continuous read over discrete
minima is still a discrete read.  ⭐ **The atoms here are stored TRAIN DATA POINTS**
— :func:`assert_atoms_are_train_rows` proves it **bit-for-bit** against the train
feature matrix, so "you built it over the centres" is a *measurement*, not a
promise.  ⛔ Per-value weights over centres would be the old flat count table
wearing a different hat.

⭐ **WHY THE ARM EXISTS AT ALL (the measured reason, R45 / design note §0.5).**
`gen-variance-equalisation` measured the well features' effective dimension
``PR = (Σλ)²/Σλ²`` at only **3.12 … 13.63**, and high-rate quantisation theory then
gives ``ε ≈ N^(−2/PR)`` parameter-free (predicted 0.253 vs measured 0.278).
Interpolation over atoms is higher order, ``ε ≈ N^(−4/PR)``.  ⭐ **The claim is
therefore a BETTER EXPONENT AT MATCHED BYTES — never "no `M` in the error term",
which was struck.**  The landscape still pays *interpolation error along the factor
manifold*; at matched bytes it gets **no extra points** and must win on the exponent
alone.

⭐ **THE REQUEST IS AN ENERGY TERM, NOT ONLY AN INITIALISATION** (non-negotiable).
Every requested term contributes one expert over the well's atoms, at a **declared
strength** ``request_strength``:

    ``E(x) = E_store(x) + β · Σ_f [ E_f(x) − E_store(x) ]  −  λ · E_couple(x)``

with ``E_store(x) = Σ_k −σ_k² logsumexp_i(−d²_ki / 2σ_k²)`` the kernel energy over
the well's atoms and ``E_f`` the same energy with the atoms **re-weighted** by the
address book's ``log P(atom_i | f = v_f)``.  At ``β = 1`` the bracket is exactly the
naive-Bayes log-likelihood-ratio :func:`chlu.core.gen_dream.address_requests`
composes in code space — which is why the two arms' addressing stories are the same
story.  ⭐ **``β = 0`` is the request-off control and it runs on the SAME chain.**

⛔ **NOT the PMI field again.**  Dreaming's coupling was a PMI table over *discrete
codes* — a table over the very quantisation this arm is trying to escape (and
"the coupling was never fitted" is **not** the correct autopsy: it *was* fitted, at
+163,840 B).  Here the coupling is **continuous, low-rank, label-free and fitted on
train features only**: per pair ``W_kl = U_klᵀ V_kl`` at rank ``r``, costing
``Σ_{k<l} r(d_k + d_l)·4`` bytes — the ledger `PREREG.md` §3 declares.
⭐ **``λ = 0`` is the coupling-off control and it is mandatory.**

⭐ **THE DECLARED CHANNEL IS THE STORE'S FULL WIDTH IN FLOATS** — ``D`` float32 per
read, i.e. **parity with ``knn_proto``**.  ⛔ No number in this module is
``K·log₂M = 30 bits``: a continuous settled state is not described by it and that
width may not be quietly inherited.  The **empirical spread of the settled states**
is reported beside the declared width (:func:`settled_state_spread`).

⛔ **THE BYPASS GUARANTEE IS OVER LAUNCH + SETTLE + READ AS ONE PATH.**
:func:`assert_request_only` proves no feature tensor enters **any** of the three, and
it is demonstrated **failing** on a deliberately bypassed variant at each stage
(:data:`BYPASS_STAGES`).  ⛔ ``chlu.core.gen_store.soft_read(store, X, τ)`` takes
FEATURES and would void the guarantee — it is never on this path.

⚠ **JAX LANDMINE, measured.**  Under ``jax_threefry_partitionable=True``
``jax.random.split`` is **prefix-stable**: ``split(k, 2)[1] is split(k, 3)[1]``.
⛔ **Arm keys are therefore derived with :func:`arm_key` (``fold_in``), never by
lengthening a split**, or two arms are bit-identical rather than decorrelated.

⚠ **Dynamics reporting is binding** (`PREREG.md` §6): acceptance, **per-step
displacement from step 1**, the **energy trace every step**, and the fraction of
chains **still descending at 1000 steps**.  ⭐ A chain still descending at 1000 steps
is reported **NOT CONVERGED** — ⛔ never as a null.  Dreaming's acceptance was
**0.952**: a chain that barely moved.  Without per-step displacement we cannot tell
settling from sitting still.

⛔ **Standing convention (inherited from** :mod:`chlu.core.gen_store` **):** wells are
never named semantically.  Integer indices only.

Runnable through :mod:`chlu.experiments.exp_gen_dynamical_read`.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .gen_store import GenStore

#: How the ``N`` atoms per well are chosen out of the train rows.  ⛔ Both options
#: select **rows**; neither ever averages anything (that would be a centre).
ATOM_SELECTIONS = ("random", "kmeanspp", "first")
#: How the settled continuous state becomes the decoder's input.
#: ``state`` — the particle position itself (`PREREG.md` §1's literal wording).
#: ``atom_mixture`` — the kernel-weighted mean of the well's atoms at the settled
#: position, i.e. the softmax generalisation of ``read_from_codes``' argmax gather.
READ_MODES = ("state", "atom_mixture")
#: Where a deliberately bypassed variant injects a feature tensor.  ⛔ Used ONLY by
#: :func:`assert_request_only` to demonstrate the assertion **failing**.
BYPASS_STAGES = ("none", "launch", "settle", "read")
#: Launch rules.  Every one of them is a function of ``(request, store)`` ONLY.
LAUNCH_MODES = ("clamp_atom", "posterior_mean", "prior_atom")
#: The arms this module owns.  ⭐ ``request_off`` and ``coupling_off`` are the two
#: mandatory controls; ``zero_step`` decodes the launch itself (dreaming's
#: laundering control, which took *all* of its apparent gain).
ARMS = ("full", "request_off", "request_off_launch", "coupling_off", "zero_step")

__all__ = [
    "ARMS",
    "ATOM_SELECTIONS",
    "BYPASS_STAGES",
    "LAUNCH_MODES",
    "READ_MODES",
    "AtomAddressBook",
    "AtomLandscape",
    "AtomStore",
    "LowRankCoupling",
    "SettleCfg",
    "arm_key",
    "assert_atoms_are_train_rows",
    "assert_request_only",
    "atom_posterior",
    "check_no_feature_signature",
    "convergence_report",
    "coupling_byte_ledger",
    "decode_width",
    "dynamical_read",
    "fit_atom_address_book",
    "fit_low_rank_coupling",
    "landscape_byte_ledger",
    "launch_state",
    "read_settled",
    "select_atom_rows",
    "settle_chain",
    "settled_state_spread",
    "store_byte_ledger",
    "total_energy",
]


# ---------------------------------------------------------------------------
# 0. ⚠ Keys — the measured JAX landmine
# ---------------------------------------------------------------------------


def arm_key(key, label: str):
    """⭐ Derive an arm's key by ``fold_in`` on a stable hash of its **label**.

    ⚠ **Measured landmine.**  With ``jax_threefry_partitionable=True`` (JAX's
    default since 0.4.36) ``jax.random.split`` is **prefix-stable**: the second key
    of ``split(k, 2)`` is bit-identical to the second key of ``split(k, 3)``.  A
    harness that "adds one more arm" by lengthening the split therefore hands two
    arms **the same key** and reports their agreement as a finding.  ⛔ Never
    lengthen a split to add an arm; fold the arm's name in.
    """
    h = hashlib.sha256(str(label).encode("utf-8")).digest()
    return jax.random.fold_in(key, int(np.frombuffer(h[:4], np.uint32)[0]))


# ---------------------------------------------------------------------------
# 1. ⭐ The atoms — stored TRAIN DATA POINTS.  ⛔ NEVER the k-means centres.
# ---------------------------------------------------------------------------


def _kmeanspp_rows(Z: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ **seeding** over the rows of ``Z``: returns ROW INDICES.

    ⭐ Seeding selects *existing rows*; it never averages.  The returned atoms are
    therefore genuine stored data points and
    :func:`assert_atoms_are_train_rows` holds bit-for-bit.
    """
    m = int(Z.shape[0])
    n = int(min(n, m))
    if n <= 0:
        return np.zeros((0,), np.int64)
    idx = [int(rng.integers(m))]
    d2 = np.sum((Z - Z[idx[0]]) ** 2, axis=1)
    for _ in range(1, n):
        tot = float(d2.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            rest = np.setdiff1d(np.arange(m), np.asarray(idx))
            idx.extend(rng.permutation(rest)[: n - len(idx)].tolist())
            break
        j = int(rng.choice(m, p=d2 / tot))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((Z - Z[j]) ** 2, axis=1))
    return np.asarray(sorted(set(idx)), np.int64)


def select_atom_rows(
    Z_train: np.ndarray, n_atoms: int, selection: str, seed: int
) -> np.ndarray:
    """Choose which ``n_atoms`` **train rows** are written into the store.

    ⛔ Every option returns row indices into ``Z_train``.  There is deliberately no
    option that returns a centroid: that is the design §0 excludes.
    """
    if selection not in ATOM_SELECTIONS:
        raise ValueError(f"selection must be one of {ATOM_SELECTIONS}, got {selection!r}")
    Z = np.asarray(Z_train, np.float32)
    Z = Z.reshape((Z.shape[0], -1))
    n = int(min(int(n_atoms), Z.shape[0]))
    rng = np.random.default_rng(int(seed) * 9_176_083 + 20_260_914)
    if selection == "first":
        # ⛔ Only for fixtures. The Shapes3D grid is a lexicographic bijection, so a
        # PREFIX restricts the outermost factor (R31's erratum). Never for a run.
        return np.arange(n, dtype=np.int64)
    if selection == "random":
        return np.sort(rng.permutation(Z.shape[0])[:n]).astype(np.int64)
    return _kmeanspp_rows(Z, n, rng)


class AtomStore(eqx.Module):
    """``K`` wells over a channel partition; ``N`` **stored train atoms** per well.

    ``atoms[k]`` is ``(N, d_k)`` — the well's own slice of ``N`` genuine train
    feature rows.  ⛔ This is the one structural difference from
    :class:`chlu.core.gen_store.GenStore`, whose ``modes[k]`` are ``M`` *fitted
    centres*: the centres are what the Advisor's constraint forbids as minima.

    The wells PARTITION the coordinates, so ``Σ_k d_k = D`` and
    ``store_bytes = N·D·4`` independently of ``K`` — which is exactly the
    ``M·D·4`` of the codebook at ``N = M``.  ⭐ **That is what "matched bytes"
    means here: the landscape gets NO extra points.**
    """

    atoms: list
    well_channels: tuple = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    feat_shape: tuple = eqx.field(static=True)
    atom_rows: tuple = eqx.field(static=True)
    selection: str = eqx.field(static=True)

    def __init__(self, well_channels, feat_shape, Z_train, atom_rows,
                 selection: str = "kmeanspp"):
        self.well_channels = tuple(tuple(int(i) for i in w) for w in well_channels)
        self.feat_shape = tuple(int(x) for x in feat_shape)
        self.dim = int(np.prod(self.feat_shape))
        flat = np.concatenate([np.asarray(w, np.int64) for w in self.well_channels])
        if flat.size != self.dim or np.unique(flat).size != self.dim:
            raise ValueError(
                f"the wells must PARTITION the {self.dim} coordinates "
                f"(got {flat.size} entries, {np.unique(flat).size} distinct)"
            )
        rows = np.asarray(atom_rows, np.int64)
        Z = np.asarray(Z_train, np.float32).reshape((np.asarray(Z_train).shape[0], -1))
        if Z.shape[1] != self.dim:
            raise ValueError(f"Z_train has width {Z.shape[1]}, store dim is {self.dim}")
        if rows.size == 0:
            raise ValueError("an atom store with zero atoms is not a store")
        self.atom_rows = tuple(int(r) for r in rows)
        self.selection = str(selection)
        self.atoms = [
            jnp.asarray(Z[np.ix_(rows, np.asarray(w, np.int64))], jnp.float32)
            for w in self.well_channels
        ]

    @property
    def n_wells(self) -> int:
        return len(self.well_channels)

    @property
    def n_atoms(self) -> int:
        return int(self.atoms[0].shape[0])

    @property
    def well_sizes(self) -> list:
        return [len(w) for w in self.well_channels]

    def n_bytes(self) -> int:
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_inexact_array))
        return int(sum(int(np.asarray(x).size) for x in leaves) * 4)


def build_atom_store(
    well_channels, feat_shape, Z_train: np.ndarray, n_atoms: int,
    selection: str = "kmeanspp", seed: int = 0,
) -> AtomStore:
    """``(train features, N) -> AtomStore``.  ⛔ Nothing is fitted; rows are chosen."""
    rows = select_atom_rows(Z_train, int(n_atoms), str(selection), int(seed))
    return AtomStore(well_channels, feat_shape, Z_train, rows, selection=str(selection))


def assert_atoms_are_train_rows(store: AtomStore, Z_train: np.ndarray) -> dict:
    """⛔⭐ **Prove the atoms are stored TRAIN DATA POINTS, bit-for-bit.**

    This is the assertion that answers the Advisor's binding constraint with a
    measurement.  Two legs, because either alone is weak:

    1. **Reconstruction.**  Re-gathering ``Z_train[atom_rows]`` through the well
       channels must reproduce every atom to the last bit.  A centroid, a mean, a
       smoothed prototype or a k-means centre fails this by construction.
    2. ⭐ **Non-degeneracy.**  The atoms must not all be the same row, and the
       per-well atom matrix must not equal its own column-mean broadcast (the
       failure mode "the atoms collapsed to a centre" looks fine under leg 1 if
       ``N = 1``).
    """
    Z = np.asarray(Z_train, np.float32).reshape((np.asarray(Z_train).shape[0], -1))
    rows = np.asarray(store.atom_rows, np.int64)
    worst = 0.0
    for w, a in zip(store.well_channels, store.atoms, strict=True):
        ref = Z[np.ix_(rows, np.asarray(w, np.int64))]
        worst = max(worst, float(np.max(np.abs(np.asarray(a) - ref))) if ref.size else 0.0)
    if worst != 0.0:
        raise AssertionError(
            "⛔ THE ATOMS ARE NOT STORED TRAIN ROWS (max |Δ| vs Z_train[atom_rows] = "
            f"{worst}). A landscape whose minima are fitted CENTRES inherits the "
            "codebook loss and the arm is void (Advisor, 2026-09-14)."
        )
    n_distinct = int(np.unique(rows).size)
    collapsed = []
    for k, a in enumerate(store.atoms):
        A = np.asarray(a)
        if A.shape[0] > 1 and float(np.max(np.abs(A - A.mean(axis=0, keepdims=True)))) == 0.0:
            collapsed.append(k)
    if collapsed:
        raise AssertionError(
            f"⛔ wells {collapsed} have atoms identical to their own mean — the "
            "store has collapsed to a centre"
        )
    return {
        "atoms_are_train_rows": True,
        "max_abs_delta_vs_train_rows": worst,
        "n_atoms": int(store.n_atoms),
        "n_distinct_atom_rows": n_distinct,
        "atom_selection": str(store.selection),
        "note": (
            "⛔ The minima are NOT the k-means centres: every atom is a verbatim "
            "train feature row. Per-value weights over centres would be the old "
            "flat count table wearing a different hat."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Byte ledgers — ⛔ declared BEFORE the run, and they must match PREREG §3
# ---------------------------------------------------------------------------


def store_byte_ledger(n_atoms: int, well_sizes: Sequence[int]) -> dict:
    """``Σ_k N·d_k·4 = N·D·4``.  ⭐ At ``N = M`` this IS the codebook's budget."""
    d = int(sum(int(x) for x in well_sizes))
    return {
        "n_atoms_per_well": int(n_atoms),
        "dim": d,
        "well_sizes": [int(x) for x in well_sizes],
        "store_bytes": int(int(n_atoms) * d * 4),
        "dtype": "float32",
        "bytes_formula": "N * sum_k d_k * 4 = N * D * 4",
        "note": (
            "⭐ matched bytes with the codebook at N = M: the landscape gets NO "
            "extra points and must win on the EXPONENT alone (PREREG §2)."
        ),
    }


def coupling_byte_ledger(rank: int, well_sizes: Sequence[int]) -> dict:
    """``Σ_{k<l} r(d_k + d_l)·4``.  ⛔ EXTRA to the store; never folded into it.

    ⚠ The per-well centre ``m_k`` and radius ``s_k`` the coupling standardises by
    are **derived from the atoms** and therefore cost **0 additional bytes** — that
    is declared here rather than discovered by a referee.
    """
    ds = [int(x) for x in well_sizes]
    K = len(ds)
    total = 0
    pairs = []
    for k in range(K):
        for lo in range(k + 1, K):
            b = int(rank) * (ds[k] + ds[lo]) * 4
            pairs.append({"k": k, "l": lo, "bytes": int(b)})
            total += b
    return {
        "rank": int(rank),
        "n_pairs": len(pairs),
        "coupling_bytes": int(total),
        "per_pair": pairs,
        "dtype": "float32",
        "bytes_formula": "sum_{k<l} r*(d_k + d_l)*4",
        "identity": "= r*(K-1)*D*4 when the wells partition D",
        "derived_free": (
            "the per-well centre m_k and radius s_k are functions of the ATOMS "
            "(0 additional bytes)"
        ),
        "note": "⛔ EXTRA to store_bytes — a byte-matched rival must be told about it.",
    }


def landscape_byte_ledger(n_atoms: int, well_sizes: Sequence[int], rank: int,
                          budget_bytes: int, label: str = "") -> dict:
    """⭐ The whole declared cost, against the budget D3 binds it to.

    D3's rule is *"store + coupling ≤ 2× budget"*; `PREREG.md` §3 declares it
    binding **exactly** at ``r = 16`` (budget A) and ``r = 128`` (budget B).
    """
    st = store_byte_ledger(n_atoms, well_sizes)
    cp = coupling_byte_ledger(rank, well_sizes)
    total = int(st["store_bytes"]) + int(cp["coupling_bytes"])
    b = int(budget_bytes)
    return {
        "label": str(label),
        "store": st,
        "coupling": cp,
        "total_bytes": total,
        "budget_bytes": b,
        "multiple_of_budget": (total / b) if b else float("nan"),
        "d3_rule": "store + coupling <= 2x budget",
        "d3_rule_satisfied": bool(b and total <= 2 * b),
        "d3_rule_binding": bool(b and total == 2 * b),
    }


def decode_width(store: AtomStore, read_mode: str = "state") -> dict:
    """⭐ **The declared channel: the store's FULL WIDTH IN FLOATS** (`PREREG.md` §7).

    ⛔ ``K·log₂M = 30 bits`` does not describe a continuous settled state and is
    **not** inherited here.  The declared width is ``D`` float32 — i.e. **parity
    with ``knn_proto``**, which also hands the decoder ``D`` floats.
    """
    d = int(store.dim)
    return {
        "declared_channel_floats": d,
        "declared_channel_bits": int(d * 32),
        "dtype": "float32",
        "parity_with": "knn_proto (also D floats to the decoder)",
        "read_mode": str(read_mode),
        "note": (
            "⛔ NOT K*log2(M): the settled state is continuous. The width is "
            "declared at the store's full float width and the empirical spread of "
            "the settled states is reported beside it (settled_state_spread)."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Geometry: bandwidths, centres, radii — all functions of the ATOMS
# ---------------------------------------------------------------------------


def _pair_d2(A: np.ndarray) -> np.ndarray:
    g = (A * A).sum(axis=1)
    return np.maximum(g[:, None] + g[None, :] - 2.0 * (A @ A.T), 0.0)


def atom_bandwidths(store: AtomStore, frac: float = 1.0, rule: str = "nn_median") -> np.ndarray:
    """Per-well kernel bandwidth ``σ_k``, from the atoms' own geometry.

    ``nn_median``       median nearest-neighbour distance between atoms (default:
                        the kernel then just barely overlaps neighbouring atoms, so
                        the landscape interpolates rather than sitting on spikes).
    ``pairwise_median`` median pairwise distance — a much smoother landscape.

    ⚠ The bandwidth is the interpolation knob.  ⛔ Too small and the minima ARE the
    atoms and the read is discrete again (the Advisor's constraint, re-entering
    through the back door); too large and every well collapses to one blob.  It is
    a **declared** config knob for exactly that reason.
    """
    out = []
    for a in store.atoms:
        A = np.asarray(a, np.float64)
        if A.shape[0] < 2:
            out.append(1.0)
            continue
        d2 = _pair_d2(A)
        np.fill_diagonal(d2, np.inf)
        if rule == "nn_median":
            s = float(np.median(np.sqrt(np.min(d2, axis=1))))
        elif rule == "pairwise_median":
            iu = np.triu_indices(A.shape[0], 1)
            s = float(np.median(np.sqrt(np.where(np.isfinite(d2), d2, 0.0)[iu])))
        else:
            raise ValueError(f"unknown bandwidth rule {rule!r}")
        out.append(s if s > 0 else 1.0)
    return np.asarray(out, np.float32) * float(frac)


def atom_centres(store: AtomStore) -> list:
    """``m_k`` — the mean of well ``k``'s atoms.  ⭐ Derived, 0 extra bytes."""
    return [np.asarray(a, np.float64).mean(axis=0) for a in store.atoms]


def atom_radii(store: AtomStore) -> np.ndarray:
    """``s_k`` — the RMS radius of well ``k``'s atoms about ``m_k``.  0 extra bytes."""
    out = []
    for a, m in zip(store.atoms, atom_centres(store), strict=True):
        A = np.asarray(a, np.float64)
        r = float(np.sqrt(np.mean((A - m[None, :]) ** 2)))
        out.append(r if r > 0 else 1.0)
    return np.asarray(out, np.float64)


# ---------------------------------------------------------------------------
# 4. ⭐ The coupling — LOW-RANK, label-free, TRAIN FEATURES ONLY
# ---------------------------------------------------------------------------


class LowRankCoupling(eqx.Module):
    """``E_couple(x) = −Σ_{k<l} x̃_kᵀ U_klᵀ V_kl x̃_l`` at rank ``r``.

    ``x̃_k = (x_k − m_k)/s_k`` with ``m_k``/``s_k`` derived from the atoms (free).
    ``U_kl`` is ``(r, d_k)`` and ``V_kl`` is ``(r, d_l)`` — so the declared cost is
    ``Σ_{k<l} r(d_k + d_l)·4``, exactly `PREREG.md` §3's ledger.

    ⛔ **Fitted on the TRAIN FEATURES ONLY, label-free** — the truncated SVD of the
    empirical cross-covariance of the standardised well blocks, i.e. the rank-``r``
    part of the Gaussian interaction.  ⛔ Not the PMI table over discrete codes that
    dreaming used: that is a table over the very quantisation this arm escapes.
    """

    u: list
    v: list
    pairs: tuple = eqx.field(static=True)
    centres: list
    radii: jnp.ndarray
    rank: int = eqx.field(static=True)

    def __init__(self, u, v, pairs, centres, radii, rank):
        self.u = [jnp.asarray(x, jnp.float32) for x in u]
        self.v = [jnp.asarray(x, jnp.float32) for x in v]
        self.pairs = tuple((int(a), int(b)) for a, b in pairs)
        self.centres = [jnp.asarray(x, jnp.float32) for x in centres]
        self.radii = jnp.asarray(radii, jnp.float32)
        self.rank = int(rank)

    def n_bytes(self) -> int:
        """⛔ Counts ``u``/``v`` ONLY — ``centres``/``radii`` are derived from the
        atoms and are declared free in :func:`coupling_byte_ledger`."""
        return int(sum(int(np.asarray(x).size) for x in self.u + self.v) * 4)


def fit_low_rank_coupling(
    store: AtomStore, Z_train: np.ndarray, rank: int, normalise: bool = True,
) -> LowRankCoupling:
    """Fit the rank-``r`` inter-well coupling on ``Z_train``.  ⛔ No labels anywhere.

    For each pair ``(k, l)`` take the standardised cross-covariance
    ``C = (1/n) X̃_kᵀ X̃_l`` and its top-``r`` SVD ``C ≈ Û Σ V̂ᵀ``; store
    ``U = Σ^{1/2} Ûᵀ`` and ``V = Σ^{1/2} V̂ᵀ``.  With ``normalise=True`` every pair
    is divided by its own leading singular value, so ``coupling_strength`` is
    **dimensionless** and comparable across pairs — otherwise one wide well
    silently owns the whole coupling term.
    """
    Z = np.asarray(Z_train, np.float32).reshape((np.asarray(Z_train).shape[0], -1))
    cen = atom_centres(store)
    rad = atom_radii(store)
    blocks = [
        (Z[:, np.asarray(w, np.int64)].astype(np.float64) - cen[k][None, :]) / rad[k]
        for k, w in enumerate(store.well_channels)
    ]
    n = max(int(Z.shape[0]), 1)
    K = store.n_wells
    r = int(rank)
    us, vs, pairs = [], [], []
    for k in range(K):
        for lo in range(k + 1, K):
            C = (blocks[k].T @ blocks[lo]) / n
            rr = int(min(r, min(C.shape)))
            U, S, Vt = np.linalg.svd(C, full_matrices=False)
            U, S, Vt = U[:, :rr], S[:rr], Vt[:rr]
            s0 = float(S[0]) if S.size and S[0] > 0 else 1.0
            sc = np.sqrt(S / (s0 if normalise else 1.0))
            Uk = (U * sc[None, :]).T                     # (rr, d_k)
            Vl = (Vt * sc[:, None])                      # (rr, d_l)
            if rr < r:                                   # pad so the ledger is exact
                Uk = np.pad(Uk, ((0, r - rr), (0, 0)))
                Vl = np.pad(Vl, ((0, r - rr), (0, 0)))
            us.append(Uk)
            vs.append(Vl)
            pairs.append((k, lo))
    return LowRankCoupling(us, vs, pairs, cen, rad, r)


# ---------------------------------------------------------------------------
# 5. ⭐ The request address book — over ATOMS, not over centres
# ---------------------------------------------------------------------------


class AtomAddressBook(eqx.Module):
    """``request -> per-well distribution over the well's ATOMS``.

    ``log_prior``  ``(K, N)`` — ``log P(atom_i)`` on the train split.
    ``log_cond``   per factor, ``(V_f, K, N)`` — ``log P(atom_i | f = v)``.

    ⚠ **Declared supervision, identical to** :class:`chlu.core.gen_dream.AddressBook`:
    the store, the coupling and the read are label-free; grounding the *words* of a
    request in the store is not, and every arm pays it (R8 gives the kNN rival
    composed per-factor prototypes built from the same labels).  It enters at the
    interface and ⛔ never in the generative path.
    """

    log_prior: jnp.ndarray
    log_cond: list

    def __init__(self, log_prior, log_cond):
        self.log_prior = jnp.asarray(log_prior, jnp.float32)
        self.log_cond = [jnp.asarray(x, jnp.float32) for x in log_cond]

    @property
    def n_wells(self) -> int:
        return int(self.log_prior.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.log_prior.shape[1])

    @property
    def n_factors(self) -> int:
        return len(self.log_cond)


def soft_atom_assignment(store: AtomStore, Z: np.ndarray, sigma: np.ndarray,
                         batch: int = 4096) -> list:
    """``[ (n, N) ]`` per-well soft assignment of TRAIN rows to atoms.

    ⛔ This is a **fit-time** quantity: it is how the address book learns which
    atoms a factor value lives on.  It is never evaluated at read time — nothing on
    the request-only path calls it.
    """
    X = np.asarray(Z, np.float32).reshape((np.asarray(Z).shape[0], -1))
    out = [np.zeros((X.shape[0], store.n_atoms), np.float64) for _ in store.atoms]
    for i in range(0, X.shape[0], int(batch)):
        chunk = X[i : i + int(batch)]
        for k, (w, a) in enumerate(zip(store.well_channels, store.atoms, strict=True)):
            z = chunk[:, np.asarray(w, np.int64)].astype(np.float64)
            A = np.asarray(a, np.float64)
            d2 = (
                (z * z).sum(axis=1)[:, None]
                - 2.0 * (z @ A.T)
                + (A * A).sum(axis=1)[None, :]
            )
            s = -np.maximum(d2, 0.0) / (2.0 * float(sigma[k]) ** 2)
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            out[k][i : i + chunk.shape[0]] = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    return out


def fit_atom_address_book(
    store: AtomStore, Z_train: np.ndarray, labels: np.ndarray,
    cardinalities: Sequence[int], sigma: np.ndarray, smoothing: float = 1.0,
) -> AtomAddressBook:
    """Fit :class:`AtomAddressBook` from train features + train factor labels.

    ⭐ The atom responsibilities are **soft** (a train row spreads over the atoms it
    is near) rather than argmax, because an argmax here would re-introduce exactly
    the piecewise-constant assignment the arm exists to escape.
    """
    pis = soft_atom_assignment(store, Z_train, sigma)
    Y = np.asarray(labels, np.int64)
    K, N = store.n_wells, store.n_atoms
    a = float(smoothing)
    prior = np.zeros((K, N), np.float64)
    for k in range(K):
        m = pis[k].sum(axis=0)
        prior[k] = (m + a) / (m.sum() + a * N)
    conds = []
    for f, card in enumerate(cardinalities):
        V = int(card)
        cond = np.zeros((V, K, N), np.float64)
        for v in range(V):
            sel = Y[:, f] == v
            for k in range(K):
                m = pis[k][sel].sum(axis=0) if bool(sel.any()) else np.zeros(N)
                cond[v, k] = (m + a) / (m.sum() + a * N)
        conds.append(np.log(cond))
    return AtomAddressBook(np.log(prior), conds)


def atom_posterior(book: AtomAddressBook, requests: np.ndarray) -> np.ndarray:
    """``(n, F) requests -> (n, K, N)`` log-scores.  ``-1`` means "not requested".

    ⛔ A request is a *factor assignment*, not an image: nothing here touches a
    feature cache, which is what lets the whole path be request-only.
    """
    R = np.asarray(requests, np.int64)
    lp = np.asarray(book.log_prior)
    out = np.broadcast_to(lp, (R.shape[0],) + lp.shape).copy()
    for f, lc in enumerate(book.log_cond):
        lc = np.asarray(lc)
        asked = R[:, f] >= 0
        if not asked.any():
            continue
        out[asked] += lc[R[asked, f]] - lp[None, :, :]
    return out.astype(np.float32)


def _softmax_last(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, np.float64)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    return P / np.maximum(P.sum(axis=-1, keepdims=True), 1e-300)


# ---------------------------------------------------------------------------
# 6. ⭐ The landscape
# ---------------------------------------------------------------------------


class AtomLandscape(eqx.Module):
    """``E(x) = E_store(x) + β Σ_f [E_f(x) − E_store(x)] − λ E_couple(x)``.

    ``E_store(x) = Σ_k −σ_k² logsumexp_i(−‖x_k − a_ki‖² / 2σ_k²)`` — the kernel
    energy over the **written atoms**; the wells are **emergent**, and the minima
    are *not* enumerated.  ⭐ That is the whole point: the settled state can land
    **between** atoms, which is precisely what ``scale`` and ``orientation`` need —
    they were quantised away, not absent (the encoder carries them at
    **0.9352 / 0.9915**).

    ``request_strength = β`` is the **declared strength** of the request energy;
    ``coupling_strength = λ`` the declared strength of the low-rank coupling.
    ⭐ ``β = 0`` is the request-off control; ``λ = 0`` is the coupling-off control.
    """

    store: AtomStore
    coupling: Optional[LowRankCoupling]
    sigma: jnp.ndarray
    request_strength: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)

    def __init__(self, store, sigma, coupling=None, request_strength: float = 1.0,
                 coupling_strength: float = 1.0):
        self.store = store
        self.coupling = coupling
        self.sigma = jnp.asarray(np.asarray(sigma, np.float32), jnp.float32)
        if int(self.sigma.shape[0]) != int(store.n_wells):
            raise ValueError(
                f"sigma must have one bandwidth per well ({store.n_wells}), "
                f"got {self.sigma.shape}"
            )
        self.request_strength = float(request_strength)
        self.coupling_strength = float(coupling_strength)

    @property
    def n_wells(self) -> int:
        return int(self.store.n_wells)

    @property
    def n_atoms(self) -> int:
        return int(self.store.n_atoms)

    def with_strengths(self, request_strength=None, coupling_strength=None) -> "AtomLandscape":
        """⭐ The controls, built from the SAME landscape object — so request-off and
        coupling-off differ from the full arm in exactly one declared float."""
        return AtomLandscape(
            self.store, self.sigma, self.coupling,
            self.request_strength if request_strength is None else float(request_strength),
            self.coupling_strength if coupling_strength is None else float(coupling_strength),
        )


def _well_slices(store: AtomStore, X):
    idx = [jnp.asarray(w, jnp.int32) for w in store.well_channels]
    Z = X.reshape((X.shape[0], -1))
    return [Z[:, i] for i in idx]


def _well_logk(store: AtomStore, X, sigma_mult):
    """``[ (n, N) ]`` per-well ``−‖x_k − a_ki‖² / 2σ_k²``.  ⚠ Gram form, not the
    ``(n, N, d)`` broadcast: the broadcast is hundreds of MB per well at the shipped
    cell and the run dies on a 16 GB laptop (``gen_store.address_codes`` carries the
    same note)."""
    out = []
    for z, a, s in zip(_well_slices(store, X), store.atoms, sigma_mult, strict=True):
        d2 = (
            jnp.sum(z * z, axis=-1)[:, None]
            - 2.0 * (z @ a.T)
            + jnp.sum(a * a, axis=-1)[None, :]
        )
        out.append(-jnp.maximum(d2, 0.0) / (2.0 * s**2))
    return out


def store_energy(land: AtomLandscape, X, sigma_scale=1.0):
    """``E_store`` — the kernel free energy over the well's atoms."""
    sig = land.sigma * jnp.asarray(sigma_scale, land.sigma.dtype)
    e = jnp.zeros((X.shape[0],), X.dtype)
    for lk, s in zip(_well_logk(land.store, X, sig), sig, strict=True):
        e = e - (s**2) * jax.nn.logsumexp(lk, axis=-1)
    return e


def request_energy(land: AtomLandscape, X, log_w, sigma_scale=1.0):
    """⭐ **The request as an ENERGY TERM** — one expert per requested term.

    ``log_w`` is ``(n_terms, n, K, N)``: one **expert per requested term**, each a
    log-weighting over the well's atoms taken from the address book.  The term
    returned is ``Σ_t [E_t(x) − E_store(x)]``, i.e. the naive-Bayes
    log-likelihood-*ratio* — so at ``β = 1`` the request energy is exactly the
    composition :func:`chlu.core.gen_dream.address_requests` performs in code space,
    lifted to the continuous landscape.

    ⛔ ``log_w`` is a function of the REQUEST and the address book only.  No feature
    tensor can reach it; that is asserted over the whole path by
    :func:`assert_request_only`.
    """
    sig = land.sigma * jnp.asarray(sigma_scale, land.sigma.dtype)
    lks = _well_logk(land.store, X, sig)
    base = jnp.zeros((X.shape[0],), X.dtype)
    for lk, s in zip(lks, sig, strict=True):
        base = base - (s**2) * jax.nn.logsumexp(lk, axis=-1)
    total = jnp.zeros((X.shape[0],), X.dtype)
    n_terms = int(log_w.shape[0])
    for t in range(n_terms):
        e = jnp.zeros((X.shape[0],), X.dtype)
        for k, (lk, s) in enumerate(zip(lks, sig, strict=True)):
            e = e - (s**2) * jax.nn.logsumexp(lk + log_w[t, :, k, :], axis=-1)
        total = total + (e - base)
    return total


def coupling_energy(land: AtomLandscape, X):
    """``−Σ_{k<l} x̃_kᵀ U_klᵀ V_kl x̃_l`` at rank ``r``.  ``0`` when no coupling."""
    c = land.coupling
    if c is None:
        return jnp.zeros((X.shape[0],), X.dtype)
    zs = _well_slices(land.store, X)
    std = [
        (z - c.centres[k][None, :]) / c.radii[k]
        for k, z in enumerate(zs)
    ]
    e = jnp.zeros((X.shape[0],), X.dtype)
    for (k, lo), U, V in zip(c.pairs, c.u, c.v, strict=True):
        e = e - jnp.sum((std[k] @ U.T) * (std[lo] @ V.T), axis=-1)
    return e


def total_energy(land: AtomLandscape, X, log_w, sigma_scale=1.0):
    """⛔ Traceable: no Python branch on a traced value (``integrators.langevin_step``
    convention).  The two strengths are **static** floats, so ``β = 0`` and
    ``λ = 0`` really do remove the term rather than multiply it by zero at cost."""
    e = store_energy(land, X, sigma_scale)
    if land.request_strength != 0.0 and log_w is not None and int(log_w.shape[0]) > 0:
        e = e + land.request_strength * request_energy(land, X, log_w, sigma_scale)
    if land.coupling_strength != 0.0 and land.coupling is not None:
        e = e + land.coupling_strength * coupling_energy(land, X)
    return e


def request_experts(book: AtomAddressBook, requests: np.ndarray) -> np.ndarray:
    """``(n, F) requests -> (T, n, K, N)`` log-weights, **one expert per requested term**.

    ``T = F``; a term the request does not ask for contributes a **flat** (zero)
    expert, so the tensor shape is static across requests and the energy stays
    ``jit``-friendly.  ⛔ Rows are never dropped from the batch on a Python branch.
    """
    R = np.asarray(requests, np.int64)
    n, F = R.shape
    K, N = book.n_wells, book.n_atoms
    lp = np.asarray(book.log_prior)
    out = np.zeros((F, n, K, N), np.float32)
    for f, lc in enumerate(book.log_cond):
        lc = np.asarray(lc)
        asked = R[:, f] >= 0
        if not asked.any():
            continue
        out[f, asked] = (lc[R[asked, f]] - lp[None, :, :]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 7. ⭐ LAUNCH — ⛔ a function of (request, store) ONLY
# ---------------------------------------------------------------------------


def launch_state(land: AtomLandscape, book: AtomAddressBook, requests: np.ndarray,
                 launch_mode: str = "clamp_atom", clamp_threshold: float = 0.5,
                 seed: int = 0) -> dict:
    """``requests -> x0`` — the address-book clamp on the addressed wells.

    ⛔ **There is no argument through which a feature tensor could enter**, and the
    signature is checked structurally by :func:`check_no_feature_signature`.

    ``clamp_atom``     the argmax atom where the posterior concentrates above
                       ``clamp_threshold``; the prior's argmax atom elsewhere.
    ``posterior_mean`` the posterior-weighted mean of the well's atoms.
    ``prior_atom``     ⛔ the request is ignored — the *launch-off* control.

    ⭐ The clamp is the launch, **not** the steering: the steering is the request
    ENERGY (non-negotiable 2).  Dreaming clamped and then froze the clamped wells,
    which is why it had ``0.58`` of 5 wells left to steer; here every coordinate
    stays free and the request holds the addressed wells by energy, so the arm's
    job — **resolution INSIDE the addressed wells** (D1) — is actually reachable.
    """
    if launch_mode not in LAUNCH_MODES:
        raise ValueError(f"launch_mode must be one of {LAUNCH_MODES}, got {launch_mode!r}")
    store = land.store
    P = _softmax_last(atom_posterior(book, requests))           # (n, K, N)
    prior = _softmax_last(np.asarray(book.log_prior)[None])[0]  # (K, N)
    n = int(P.shape[0])
    conf = P.max(axis=-1)
    idx = P.argmax(axis=-1)
    clamped = conf >= float(clamp_threshold)
    x0 = np.zeros((n, store.dim), np.float32)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        A = np.asarray(store.atoms[k], np.float32)
        if launch_mode == "posterior_mean":
            x0[:, cols] = (P[:, k, :] @ A).astype(np.float32)
        elif launch_mode == "prior_atom":
            x0[:, cols] = A[int(np.argmax(prior[k]))][None, :]
        else:
            take = np.where(clamped[:, k], idx[:, k], int(np.argmax(prior[k])))
            x0[:, cols] = A[take]
    return {
        "x0": x0,
        "atom_index": idx.astype(np.int64),
        "confidence": conf.astype(np.float32),
        "clamped": clamped,
        "posterior": P.astype(np.float32),
        "launch_mode": str(launch_mode),
        "clamp_threshold": float(clamp_threshold),
        "mean_n_clamped": float(clamped.sum(axis=1).mean()) if clamped.size else 0.0,
        "mean_n_relaxed": float((~clamped).sum(axis=1).mean()) if clamped.size else 0.0,
        "mean_confidence": float(conf.mean()) if conf.size else float("nan"),
        "n_wells": int(store.n_wells),
    }


# ---------------------------------------------------------------------------
# 8. ⭐ SETTLE — 1000 steps, and the audit trail that convicts a frozen chain
# ---------------------------------------------------------------------------


class SettleCfg:
    """Settling knobs in absolute units.  ⛔ Kept out of the config dataclass on
    purpose: ``sigma`` is a *multiple* of a quantity measured from the atoms, and
    shipping the fraction into the sampler is exactly the defect that froze
    `gen-dream-sampler`'s first run (``moved = 0.05``)."""

    __slots__ = ("steps", "step_size", "sigma_scale_start", "sigma_scale_end",
                 "temp_start", "temp_end", "metropolis", "trace_every")

    def __init__(self, steps=1000, step_size=0.2, sigma_scale_start=4.0,
                 sigma_scale_end=1.0, temp_start=0.0, temp_end=0.0,
                 metropolis=True, trace_every=1):
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.sigma_scale_start = float(sigma_scale_start)
        self.sigma_scale_end = float(sigma_scale_end)
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.metropolis = bool(metropolis)
        self.trace_every = int(trace_every)

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__slots__}


def _geom(a: float, b: float, i: int, n: int) -> float:
    a, b = float(a), float(b)
    if n <= 1 or a <= 0.0 or b <= 0.0:
        return b
    return float(a * (b / a) ** (i / (n - 1)))


@eqx.filter_jit
def _settle_step(land, x, log_w, key, eps, temp, sigma_scale, sigma_ref, metropolis):
    """One MALA step on the FULL state (⛔ nothing is frozen).

    ⚠ ``eps``/``temp``/``sigma_scale`` arrive as traced jnp scalars, not Python
    floats: annealing them as Python floats retraces the whole step every step.
    ``metropolis`` is a Python bool (2 values, static).

    ⚠ Two energies come back.  ``E`` is the *annealed* landscape's energy at this
    step; ``E_ref`` is the SAME state scored at the fixed reference bandwidth.
    ⛔ **Only ``E_ref`` may be differenced across steps** — with σ annealing,
    ``E[0]`` and ``E[-1]`` are values of two different functions and the trace can
    rise while the state strictly improves (the trap `gen-store-grouping` filed
    against annealed training losses, and `gen-dream-sampler` re-filed).
    """
    ef = lambda z: jnp.sum(total_energy(land, z, log_w, sigma_scale))  # noqa: E731
    g = jax.grad(ef)(x)
    mean_f = x - eps * g
    key, nk, uk = jax.random.split(key, 3)
    noise = jax.random.normal(nk, x.shape, x.dtype) * jnp.sqrt(2.0 * eps * temp)
    xp = mean_f + noise
    gp = jax.grad(ef)(xp)
    mean_r = xp - eps * gp
    denom = jnp.maximum(4.0 * eps * temp, jnp.asarray(1e-30, x.dtype))
    logq_f = -jnp.sum((xp - mean_f) ** 2, axis=-1) / denom
    logq_r = -jnp.sum((x - mean_r) ** 2, axis=-1) / denom
    de = total_energy(land, xp, log_w, sigma_scale) - total_energy(land, x, log_w, sigma_scale)
    # ⚠ T = 0 is a legitimate (and the default) SETTLING regime: the proposal is a
    # pure gradient step and MH degenerates to "accept iff the energy fell".  The
    # finite-T expression is evaluated anyway (``where`` has no lazy branch) and can
    # produce inf-inf there, so it is sanitised before the select.
    log_alpha_mh = jnp.nan_to_num(
        -de / jnp.maximum(temp, 1e-30) + logq_r - logq_f, nan=-1e30,
        posinf=0.0, neginf=-1e30)
    log_alpha = jnp.where(temp > 0.0, log_alpha_mh, jnp.where(de < 0.0, 0.0, -1e30))
    u = jax.random.uniform(uk, (x.shape[0],), x.dtype)
    accepted = jnp.log(jnp.maximum(u, jnp.asarray(1e-30, x.dtype))) < log_alpha
    take = accepted if metropolis else jnp.ones_like(accepted)
    x_new = jnp.where(take[:, None], xp, x)
    disp = jnp.sqrt(jnp.mean((x_new - x) ** 2, axis=-1))
    return (
        x_new, jnp.exp(jnp.minimum(log_alpha, 0.0)), accepted, disp, key,
        total_energy(land, x_new, log_w, sigma_scale),
        total_energy(land, x_new, log_w, sigma_ref),
    )


def settle_chain(land: AtomLandscape, x0: np.ndarray, log_w, cfg: SettleCfg, key) -> dict:
    """⭐ The settling rollout, with its binding audit trail (`PREREG.md` §6).

    ``steps = 0`` is the **zero-step control**: the state comes back exactly as
    handed in, so the caller decodes the launch itself.  ⛔ It is never optional —
    it is the control that took *all* of dreaming's apparent gain.

    ⛔ **Logged EVERY step from step 1:** the energy trace (annealed *and* at the
    fixed reference bandwidth), the acceptance probability, whether the step was
    accepted, and the **per-step displacement**.  ⚠ Dreaming's acceptance was
    ``0.952`` — a chain that barely moved; without per-step displacement we cannot
    tell settling from sitting still.

    ⛔ **There is no argument through which a feature tensor could enter** — the
    signature is checked structurally by :func:`check_no_feature_signature`.
    """
    x = jnp.asarray(np.asarray(x0, np.float32))
    n_steps = int(cfg.steps)
    lw = None if log_w is None else jnp.asarray(np.asarray(log_w, np.float32))
    sig_ref = jnp.asarray(float(cfg.sigma_scale_end), jnp.float32)
    eps = jnp.asarray(float(cfg.step_size), jnp.float32)
    e_hist, er_hist, a_hist, acc_hist, disp_hist = [], [], [], [], []
    sig_hist, temp_hist = [], []
    s0 = jnp.asarray(_geom(cfg.sigma_scale_start, cfg.sigma_scale_end, 0, max(n_steps, 1)),
                     jnp.float32)
    e_hist.append(np.asarray(total_energy(land, x, lw, s0), np.float64))
    er_hist.append(np.asarray(total_energy(land, x, lw, sig_ref), np.float64))
    for i in range(n_steps):
        sig_now = jnp.asarray(
            _geom(cfg.sigma_scale_start, cfg.sigma_scale_end, i, n_steps), jnp.float32)
        temp_now = jnp.asarray(
            _geom(cfg.temp_start, cfg.temp_end, i, n_steps)
            if cfg.temp_start > 0 else float(cfg.temp_end), jnp.float32)
        key, sk = jax.random.split(key)
        x, aprob, acc, disp, _k, e_new, er_new = _settle_step(
            land, x, lw, sk, eps, temp_now, sig_now, sig_ref, bool(cfg.metropolis))
        a_hist.append(np.asarray(aprob, np.float64))
        acc_hist.append(np.asarray(acc, bool))
        disp_hist.append(np.asarray(disp, np.float64))
        e_hist.append(np.asarray(e_new, np.float64))
        er_hist.append(np.asarray(er_new, np.float64))
        sig_hist.append(float(sig_now))
        temp_hist.append(float(temp_now))
    n = int(x.shape[0])
    trace = {
        "energy": np.stack(e_hist) if e_hist else np.zeros((0, n)),
        "energy_ref": np.stack(er_hist) if er_hist else np.zeros((0, n)),
        "accept_prob": np.stack(a_hist) if a_hist else np.zeros((0, n)),
        "accepted": np.stack(acc_hist) if acc_hist else np.zeros((0, n), bool),
        "displacement": np.stack(disp_hist) if disp_hist else np.zeros((0, n)),
        "sigma_scale": np.asarray(sig_hist, np.float64),
        "temperature": np.asarray(temp_hist, np.float64),
    }
    return {"x": np.asarray(x, np.float32), "trace": trace}


def convergence_report(trace: dict, window: int = 100, tol: float = 1e-3) -> dict:
    """⭐ **Is the chain settled, or still descending?**  (`PREREG.md` §6, binding.)

    A chain is **STILL DESCENDING** iff the OLS slope of its ``energy_ref`` over the
    last ``window`` steps, multiplied by the window, is below ``−tol × scale`` with
    ``scale`` the mean ``|E_ref|`` over the same window.  ⛔ **A chain still
    descending at the final step is reported NOT CONVERGED — never as a null.**
    """
    E = np.asarray(trace["energy_ref"], np.float64)
    disp = np.asarray(trace["displacement"], np.float64)
    acc = np.asarray(trace["accept_prob"], np.float64)
    took = np.asarray(trace["accepted"])
    if E.shape[0] < 2:
        # ⭐ The zero-step control still fills EVERY key, so the dynamics table has
        # no ragged row and a missing number cannot be mistaken for a zero.
        e0 = float(E[0].mean()) if E.shape[0] else float("nan")
        nan = float("nan")
        return {
            "n_steps": 0, "window": 0, "tol": float(tol),
            "rule": "zero-step control: no step was taken, so nothing can descend",
            "zero_step": True,
            "frac_still_descending": nan, "n_still_descending": 0,
            "n_chains": int(E.shape[1]) if E.ndim > 1 and E.shape[0] else 0,
            "relative_tail_drop": {"mean": nan, "min": nan, "max": nan},
            "energy_first": e0, "energy_final": e0,
            "energy_absolute_change": 0.0, "energy_relative_drop": 0.0,
            "energy_is_flat": True,
            "mean_accept_prob": nan, "accepted_fraction": nan,
            "displacement_step_1": nan, "displacement_final_step": nan,
            "displacement_mean": nan, "total_displacement": 0.0,
            "verdict": "ZERO-STEP CONTROL (the launch, decoded; no chain was run)",
            "note": ("⭐ dreaming's laundering control: it took ESSENTIALLY ALL of "
                     "that spoke's apparent gain, so it is never optional."),
        }
    w = int(min(max(window, 2), E.shape[0]))
    tail = E[-w:]
    t = np.arange(w, dtype=np.float64)
    t = t - t.mean()
    slope = (t[:, None] * (tail - tail.mean(axis=0, keepdims=True))).sum(axis=0) / (t * t).sum()
    scale = np.maximum(np.abs(tail).mean(axis=0), 1e-12)
    rel = slope * w / scale
    still = rel < -float(tol)
    frac = float(np.mean(still))
    return {
        "n_steps": int(E.shape[0] - 1),
        "window": w,
        "tol": float(tol),
        "rule": ("still descending iff OLS slope of energy_ref over the last "
                 "`window` steps x window < -tol * mean|E_ref| over that window"),
        "frac_still_descending": frac,
        "n_still_descending": int(still.sum()),
        "n_chains": int(E.shape[1]),
        "relative_tail_drop": {"mean": float(rel.mean()), "min": float(rel.min()),
                               "max": float(rel.max())},
        "energy_first": float(E[0].mean()),
        "energy_final": float(E[-1].mean()),
        "energy_absolute_change": float(E[-1].mean() - E[0].mean()),
        "energy_relative_drop": float(
            (E[0].mean() - E[-1].mean()) / max(abs(E[0].mean()), abs(E[-1].mean()), 1e-12)),
        "energy_is_flat": bool(
            float(np.mean(np.abs(E[0] - E[-1]))) / max(float(np.abs(E).mean()), 1e-12) < 1e-6),
        "mean_accept_prob": float(acc.mean()) if acc.size else float("nan"),
        "accepted_fraction": float(took.mean()) if took.size else float("nan"),
        "displacement_step_1": float(disp[0].mean()) if disp.size else float("nan"),
        "displacement_final_step": float(disp[-1].mean()) if disp.size else float("nan"),
        "displacement_mean": float(disp.mean()) if disp.size else float("nan"),
        "total_displacement": float(disp.sum(axis=0).mean()) if disp.size else float("nan"),
        "verdict": (
            "⛔ NOT CONVERGED — chains are still descending at the final step"
            if frac > 0.0 else "settled (no chain descending in the final window)"
        ),
        "note": (
            "⭐ Per-step displacement is reported FROM STEP 1. ⚠ dreaming's "
            "acceptance was 0.952 on a chain that barely moved: acceptance alone "
            "cannot distinguish settling from sitting still."
        ),
    }


# ---------------------------------------------------------------------------
# 9. ⭐ READ — the settled continuous state, at the store's full float width
# ---------------------------------------------------------------------------


def read_settled(land: AtomLandscape, x, read_mode: str = "state",
                 read_sigma_scale: float = 1.0) -> np.ndarray:
    """``settled state -> the decoder's input`` at ``feat_shape``.

    ``state``        the settled particle position itself (`PREREG.md` §1's literal
                     wording: *decode the SETTLED CONTINUOUS state*).
    ``atom_mixture`` the kernel-weighted mean of the well's atoms at the settled
                     position — the softmax generalisation of
                     :func:`chlu.core.gen_store.read_from_codes`' argmax gather, so
                     the output lies in the convex hull of the **stored atoms**.

    ⛔ **There is no argument through which a feature tensor could enter.**
    ⛔ ``gen_store.soft_read(store, X, τ)`` takes FEATURES and would void the
    guarantee — it is not on this path (`PREREG.md` §7).
    """
    if read_mode not in READ_MODES:
        raise ValueError(f"read_mode must be one of {READ_MODES}, got {read_mode!r}")
    store = land.store
    X = jnp.asarray(np.asarray(x, np.float32)).reshape((np.asarray(x).shape[0], -1))
    if read_mode == "state":
        return np.asarray(X, np.float32).reshape((X.shape[0],) + store.feat_shape)
    sig = land.sigma * float(read_sigma_scale)
    lks = _well_logk(store, X, sig)
    out = jnp.zeros((X.shape[0], store.dim), jnp.float32)
    for w, a, lk in zip(store.well_channels, store.atoms, lks, strict=True):
        out = out.at[:, jnp.asarray(w, jnp.int32)].set(jax.nn.softmax(lk, axis=-1) @ a)
    return np.asarray(out, np.float32).reshape((X.shape[0],) + store.feat_shape)


def settled_state_spread(x, store: Optional[AtomStore] = None) -> dict:
    """⭐ The **empirical** spread of the settled states, beside the declared width.

    Reports the participation ratio ``PR = (Σλ)²/Σλ²`` of the settled set — the
    same statistic `gen-variance-equalisation` measured at **3.12 … 13.63** on the
    well features, and the exponent in ``ε ≈ N^(−2/PR)``.  ⛔ This is the honest
    companion to "the channel is ``D`` floats": ``D`` floats is the *declared* width,
    not the realised one.
    """
    X = np.asarray(x, np.float64).reshape((np.asarray(x).shape[0], -1))
    if X.shape[0] < 2:
        return {"n": int(X.shape[0]), "note": "fewer than 2 states — no spread"}
    Xc = X - X.mean(axis=0, keepdims=True)
    var = (Xc * Xc).sum(axis=0) / max(X.shape[0] - 1, 1)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = (s**2) / max(X.shape[0] - 1, 1)
    pr = float((lam.sum() ** 2) / max(float((lam**2).sum()), 1e-300))
    out = {
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "total_variance": float(var.sum()),
        "mean_coordinate_std": float(np.sqrt(np.maximum(var, 0)).mean()),
        "participation_ratio": pr,
        "n_distinct_states": int(np.unique(np.asarray(X, np.float32), axis=0).shape[0]),
        "note": (
            "⛔ n_distinct_states is NOT falsifier 2: continuous states are "
            "trivially distinct and that number is worthless. Falsifier 2 is the "
            "distinct FACTOR TUPLES the EXTERNAL CLASSIFIER reads off the DECODED "
            "IMAGES (PREREG §5.2)."
        ),
    }
    if store is not None:
        A = np.concatenate(
            [np.asarray(a, np.float64) for a in store.atoms], axis=1)
        out["atom_participation_ratio"] = float(
            (lambda l: (l.sum() ** 2) / max(float((l**2).sum()), 1e-300))(
                (np.linalg.svd(A - A.mean(axis=0, keepdims=True), compute_uv=False) ** 2)
                / max(A.shape[0] - 1, 1)
            )
        )
    return out


# ---------------------------------------------------------------------------
# 10. ⛔⭐ THE PATH, AND THE BYPASS GUARANTEE OVER ALL THREE STAGES AT ONCE
# ---------------------------------------------------------------------------

#: Parameter names that could carry a feature tensor.  ⛔ None of them may appear in
#: the signature of ``launch_state``, ``settle_chain`` or ``read_settled``.
_FEATURE_ARG_NAMES = (
    "x_features", "features", "feats", "feature", "zf", "z_features",
    "encoded", "embeddings", "images", "imgs", "X_train", "Z_train", "X", "Z",
)


def check_no_feature_signature(fns: Sequence[Callable], banned=_FEATURE_ARG_NAMES) -> dict:
    """⛔ **Structural leg** — no stage of the path may *accept* a feature tensor.

    A future edit that adds ``X`` to ``settle_chain`` is caught here rather than at
    cell time.  ⭐ Demonstrated FAILING in ``tests/test_gen_dynamical_read.py`` on a
    function that takes one.
    """
    bad = {}
    for fn in fns:
        params = list(inspect.signature(fn).parameters)
        hits = [p for p in params if p in set(banned)]
        if hits:
            bad[getattr(fn, "__qualname__", str(fn))] = hits
    if bad:
        raise AssertionError(
            "⛔ REQUEST-ONLY SIGNATURE CHECK FAILED — these stages accept an "
            f"argument that could carry a feature tensor: {bad}. The launch, the "
            "settle and the read are ONE path and none of them may see features."
        )
    return {
        "signature_ok": True,
        "stages_checked": [getattr(f, "__qualname__", str(f)) for f in fns],
        "banned_names": list(banned),
    }


def dynamical_read(
    land: AtomLandscape,
    book: AtomAddressBook,
    requests: np.ndarray,
    cfg: SettleCfg,
    key,
    launch_mode: str = "clamp_atom",
    clamp_threshold: float = 0.5,
    read_mode: str = "state",
    *,
    bypass_stage: str = "none",
    bypass_features: Optional[np.ndarray] = None,
    bypass_alpha: float = 0.0,
) -> dict:
    """⭐ **The whole path, as one function**: launch -> settle -> read.

    ⛔ The three ``bypass_*`` arguments are **keyword-only, default-off and exist
    solely so :func:`assert_request_only` can DEMONSTRATE THE ASSERTION FAILING.**
    At ``bypass_stage="none"`` (the shipped path) ``bypass_features`` is never read
    — which is what the assertion measures, bit-for-bit, rather than asserts.
    ⚠ They follow the precedent of ``gen_bridge.quantise_through_store``'s
    ``bypass_alpha``: a guard that has never been seen to fail is decoration.
    """
    if bypass_stage not in BYPASS_STAGES:
        raise ValueError(f"bypass_stage must be one of {BYPASS_STAGES}, got {bypass_stage!r}")
    R = np.asarray(requests, np.int64)
    lw = request_experts(book, R)
    lau = launch_state(land, book, R, launch_mode, clamp_threshold)
    x0 = lau["x0"]

    def _inject(v):
        B = np.asarray(bypass_features, np.float32).reshape(
            (np.asarray(bypass_features).shape[0], -1))
        if B.shape[0] < v.shape[0]:
            B = np.repeat(B, int(np.ceil(v.shape[0] / B.shape[0])), axis=0)
        return (v + float(bypass_alpha) * B[: v.shape[0], : v.shape[1]]).astype(np.float32)

    if bypass_stage == "launch" and bypass_features is not None:
        x0 = _inject(x0)
    out = settle_chain(land, x0, lw, cfg, key)
    xs = out["x"]
    if bypass_stage == "settle" and bypass_features is not None:
        xs = _inject(xs)
    r = read_settled(land, xs, read_mode)
    if bypass_stage == "read" and bypass_features is not None:
        flat = _inject(np.asarray(r, np.float32).reshape((r.shape[0], -1)))
        r = flat.reshape(r.shape)
    return {
        "launch": lau,
        "x0": np.asarray(x0, np.float32),
        "x_settled": np.asarray(xs, np.float32),
        "read": np.asarray(r, np.float32),
        "trace": out["trace"],
        "bypass_stage": str(bypass_stage),
        "bypass_alpha": float(bypass_alpha),
    }


def assert_request_only(
    land: AtomLandscape,
    book: AtomAddressBook,
    requests: np.ndarray,
    cfg: SettleCfg,
    key,
    features_a: np.ndarray,
    features_b: np.ndarray,
    launch_mode: str = "clamp_atom",
    clamp_threshold: float = 0.5,
    read_mode: str = "state",
    bypass_stage: str = "none",
    bypass_alpha: float = 0.0,
) -> dict:
    """⛔⭐ **THE BYPASS GUARANTEE, over LAUNCH + SETTLE + READ AS ONE PATH.**

    Run the *whole* path twice with two **different** feature tensors available to
    it.  All three stage outputs — ``x0``, ``x_settled`` and the decoder's input —
    must come back **bit-identical**, because no feature tensor may enter any of
    them.  ⭐ Reporting the three stages separately is deliberate: it says *which*
    stage leaked, and it is why the guarantee covers the path rather than the read
    alone (`PREREG.md` §7, R40.C).

    ⭐ **Demonstrated failing.**  Call with ``bypass_stage`` in ``("launch",
    "settle", "read")`` and ``bypass_alpha > 0`` and this raises — that is the point
    of the argument's existence.  A guard that has never been seen to fail is
    decoration.

    ⛔ ``gen_store.soft_read(store, X, τ)`` takes FEATURES and would void this; it
    is not on the path, which :func:`check_no_feature_signature` enforces
    structurally.
    """
    sig = check_no_feature_signature([launch_state, settle_chain, read_settled])
    A = np.asarray(features_a, np.float32)
    B = np.asarray(features_b, np.float32)
    if A.shape != B.shape:
        raise ValueError("the two feature tensors must have the same shape")
    if float(np.max(np.abs(A - B))) == 0.0:
        raise AssertionError(
            "⛔ the two feature tensors are IDENTICAL — the check could not "
            "discriminate and would pass vacuously"
        )
    kw = dict(launch_mode=launch_mode, clamp_threshold=clamp_threshold,
              read_mode=read_mode, bypass_stage=str(bypass_stage),
              bypass_alpha=float(bypass_alpha))
    ra = dynamical_read(land, book, requests, cfg, key, bypass_features=A, **kw)
    rb = dynamical_read(land, book, requests, cfg, key, bypass_features=B, **kw)
    deltas = {
        "launch": float(np.max(np.abs(ra["x0"] - rb["x0"]))),
        "settle": float(np.max(np.abs(ra["x_settled"] - rb["x_settled"]))),
        "read": float(np.max(np.abs(ra["read"] - rb["read"]))),
    }
    leaked = [k for k, v in deltas.items() if v != 0.0]
    feature_difference = float(np.max(np.abs(A - B)))
    if leaked:
        raise AssertionError(
            "⛔ REQUEST-ONLY GUARANTEE VIOLATED at stage(s) "
            + ", ".join(leaked)
            + f": two DIFFERENT feature tensors (max |Δ| = {feature_difference}) "
            + f"produced DIFFERENT outputs {deltas}. A feature tensor reaches the "
            "launch/settle/read path, so the declared channel is not the store's "
            "float width and every cross-arm comparison through it is void."
        )
    return {
        **sig,
        "request_only_over_launch_settle_read": True,
        "max_abs_delta_per_stage": deltas,
        "feature_tensor_max_abs_difference": feature_difference,
        "n_requests": int(np.asarray(requests).shape[0]),
        "steps": int(cfg.steps),
        "bypass_stage": str(bypass_stage),
        "bypass_alpha": float(bypass_alpha),
        "declared_channel": decode_width(land.store, read_mode),
        "note": (
            "⭐ The guarantee is over the WHOLE path — launch, settle and read — "
            "not over the read alone. ⛔ soft_read(store, X, tau) takes FEATURES "
            "and is not on this path."
        ),
    }


# ---------------------------------------------------------------------------
# 11. Artifacts
# ---------------------------------------------------------------------------


def save_trace(trace: dict, path) -> str:
    """⭐ Write the audit trail to disk.  ⛔ Not summarised in prose — on disk.

    ⚠ The energy trace is written **every step** (`PREREG.md` §6): a summary
    cannot show that a chain was still descending at the final step.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        energy=np.asarray(trace["energy"], np.float32),
        energy_ref=np.asarray(trace["energy_ref"], np.float32),
        accept_prob=np.asarray(trace["accept_prob"], np.float32),
        accepted=np.asarray(trace["accepted"], np.bool_),
        displacement=np.asarray(trace["displacement"], np.float32),
        sigma_scale=np.asarray(trace["sigma_scale"], np.float32),
        temperature=np.asarray(trace["temperature"], np.float32),
    )
    return str(path)
