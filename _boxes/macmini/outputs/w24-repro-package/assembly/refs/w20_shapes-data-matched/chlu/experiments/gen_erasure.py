"""``hub/erasure`` -- is deleted content GONE from the CLU store, or merely GATED?

⛔ **Read this header before any number in** ``outputs/hub-erasure/LOG.md``.

A gate and an erasure are behaviourally identical and physically different.  An
8-byte value-presence bitmap, a deletion log and a refusal list all **still hold the
content** and decline to serve it; the CLU's store is **refitted with no row carrying
the value**, so there is nothing left to serve.  This module measures that difference
under **one adversary** applied identically to every arm.

⭐ **The one structural fact that makes the question measurable here.**
``AtomStore.atoms[k] = Z_train[atom_rows][:, cols_k]`` and
``assert_atoms_are_train_rows`` enforces it bit-for-bit; the atom row index is
**shared across wells**.  ⇒ the CLU store's entire content is ``N`` verbatim train
feature rows, and an adversary holding the weights can decode every one of them.
Symmetrically :class:`~chlu.core.gen_request.FactorPrototypes` holds one mean feature
vector **per (factor, value)** -- including the deleted value.  So the contrast is not
a metaphor: it is ``D*4`` bytes of the prototype table that *are* the deleted content,
against a set of ``N`` rows none of which carries it.

⛔⛔ **THE SCOPE RULING, AND IT IS A MEASURED CALIBRATION, NOT AN OPINION.**
"Recoverable from the store" ⛔ cannot mean "the frozen decoder can be driven to draw
it".  Measured, 2026-09-18: 64 free signed coefficients over a **random Gaussian**
subspace matched only in per-coordinate mean/std reach ``object_hue = 5`` at **0.672
unanimous over three independent judges** -- *better* than the same sweep over real
atoms from a pool that never held the value (0.594), and better than coordinate-
shuffled atoms (0.562).  The frozen decoder was trained on every hue and can render
any of them; the split artifact's own scope note says so.  => an adversary free to
choose the decoder's input recovers any value from any store, **including one whose
parameters are noise**.  Therefore:

* a move is **admissible** only if its own positive control -- the same move against a
  dimension-matched random span, and against the landscape-removed store -- returns
  ~ 0, while its sensitivity control -- the same move against a store that **holds**
  the value -- returns high;
* every move's (sensitivity, specificity) pair is measured and written into the
  artifact **beside** its result, and ⛔ a move that fails calibration is reported and
  its number is never read as evidence.

**What is frozen and read-only** (charter A.4): the encoder feature cache, the
decoder, the three external classifiers, the store grouping, the repaired splits.
⛔ Nothing here re-fits any of them.  This module only *reads* the cache, so it can
never extend it and the re-encode gate cannot bind.

**Bytes.**  Every arm is held to ``byte_budget`` (524,288 B, R26.3's corrected budget;
⛔ never the retired 688,128 B) under ``exp_gen_cell.arm_byte_ledger``'s convention:
``clu`` deploys the atoms (``N*D*4``) and reports its addressing head beside;
``knn_proto`` deploys the prototype table itself; ``basis_head`` deploys dictionary +
head.  The gate is **8 bytes** and is ADDED to the gated arms' deployed total -- a
declared handicap on our side, never netted off.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 0. Frozen-instrument locations -- ⛔ environment only, never a literal path
# ---------------------------------------------------------------------------

#: R26.3's corrected budget.  ⛔ Do NOT inherit the retired 688,128 B.
BYTE_BUDGET = 524_288
#: The gate of the target file's section 1, in bytes: 57 (factor, value) bits, padded.
GATE_BYTES = 8
#: Factor axis indices in the Shapes3D grid.
F_FLOOR_HUE, F_WALL_HUE, F_OBJECT_HUE, F_SCALE, F_SHAPE, F_ORIENTATION = range(6)
#: ⛔ Forbidden as deletion targets (target file section 5): both are PHYSICAL
#: DEFAULTS the model degrades towards, so a held-out default is unremovable for
#: reasons that have nothing to do with the store.  ⚠ Independently corroborated by
#: this hub's own instrument check.
FORBIDDEN_DELETION_FACTORS = ("scale", "orientation")


def _env_dir(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(
            f"⛔ {name} is not set. The frozen instrument is located by environment "
            "only; a literal path here would silently re-point the pin."
        )
    return Path(v)


def instrument_paths(bridge_seed: int = 0) -> dict:
    """Every frozen artefact this module reads, resolved and existence-checked."""
    bridge = _env_dir("CHLU_GEN_BRIDGE_ROOT") / "shapes3d_recon" / f"seed_{int(bridge_seed)}"
    declift = _env_dir("CHLU_GEN_DECLIFT_DIR")
    data = _env_dir("CHLU_GEN_DATA_ROOT") / "shapes3d"
    clf_root = _env_dir("CHLU_CLAUDE_ROOT") / "outputs" / "gen-external-classifier"
    out = {
        "features": bridge / "features.npy",
        "feature_cache": bridge / "feature_cache.json",
        "decoder": declift / f"decoder_v1_s{int(bridge_seed)}.pkl",
        "store_plan": _env_dir("CHLU_GEN_STORE_RUNS") / "shapes3d_recon" / "store_plan.json",
        "value_holdout": data / "value_holdout",
        "factor_index": data / "value_holdout" / "factor_index.npz",
        "classifier_gate": clf_root / "classifier_gate.json",
    }
    for s in (0, 1, 2):
        out[f"classifier_{s}"] = clf_root / f"seed_{s}" / "external_classifier.pkl"
    missing = sorted(str(p) for p in out.values() if not Path(p).exists())
    if missing:
        raise FileNotFoundError(
            "⛔ the frozen instrument is incomplete; nothing here is scorable:\n  "
            + "\n  ".join(missing)
        )
    return out


def assert_cache_is_read_only(paths: Mapping) -> dict:
    """⛔ w3 section 8 hazard 2: record cache coverage FIRST.

    ``shortfall == 0`` means no run here can extend the cache, so the re-encode gate
    cannot bind and there is no GPU hazard to manage.  ⛔ Asserted, not assumed.
    """
    with open(paths["feature_cache"]) as fh:
        fc = json.load(fh)
    scope, n_rows = str(fc.get("scope")), int(fc.get("n_rows", 0))
    shortfall = 0 if (scope == "full" and n_rows == 480_000) else 480_000 - n_rows
    if shortfall:
        raise AssertionError(
            f"⛔ the feature cache covers {n_rows}/480,000 rows (scope={scope!r}); a "
            f"shortfall of {shortfall} means a run here would EXTEND the cache and "
            "the re-encode gate binds at full strength (w3 section 8)."
        )
    return {"scope": scope, "n_rows": n_rows, "shortfall": 0, "D": int(fc["D"]),
            "dtype": fc["dtype"], "features_sha256": fc["features_sha256"],
            "feat_shape": tuple(int(x) for x in fc["feat_shape"]),
            "consequence": ("⭐ read-only: no run in this hub can extend the cache, so "
                            "the re-encode gate cannot bind and there is no GPU case")}


# ---------------------------------------------------------------------------
# 1. Config -- ⭐ w3-hub-common section 3: the store fit is pinned COMPLETELY
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErasureConfig:
    """⛔ Every knob that moves a number is here, and the artifact carries all of it.

    ⭐ w3-hub-common section 3: *"Pin the store fit completely in your PREREG --
    partition, RNG stream, Lloyd budget, ridge, every one."*  Wave 2's composition hub
    pinned ridge and left the rest free, and its headline level failed to replicate.
    """

    # -- the frozen instrument ------------------------------------------------
    dataset: str = "shapes3d"
    #: ⛔ ONE bridge seed for every quantity in a paired comparison (w3 section 8).
    bridge_seed: int = 0
    #: The attacked judge.  ⭐ The other two are HELD OUT and score, never attacked.
    attacked_judge: int = 0
    held_out_judges: tuple = (1, 2)

    # -- the live set and the store fit ---------------------------------------
    #: ⭐ The programme's store pool size (``config.gen_store.n_store_train``).
    n_store_train: int = 32_768
    #: ``N*D*4 = 64*2048*4 = 524,288`` -- the budget, exactly.
    n_atoms: int = 64
    atom_selection: str = "kmeanspp"
    #: ⛔ Inherited grouping only (w3 section 4 fact 1: learned placement was a null).
    grouping_source: str = "store_plan.json (ward, K=5, frozen)"
    address_smoothing: float = 1.0
    sigma_rule: str = "nn_median"
    sigma_frac: float = 1.0
    #: ⭐ The read this hub measures.  Zero settling steps -- ``x0`` IS the read.
    launch_mode: str = "clamp_atom"
    read_mode: str = "state"
    #: Store seeds.  ⛔ >= 3 is binding for the churn claim (target section 3).
    store_seeds: tuple = (0, 1, 2)
    #: The RNG stream every pool draw and atom selection derives from.
    rng_base: int = 20_260_919

    # -- bytes -----------------------------------------------------------------
    byte_budget: int = BYTE_BUDGET
    byte_match_tolerance: int = 0
    #: basis_head dictionary size: the largest ``m`` the budget allows once the
    #: head carries singles + ALL PAIRWISE interactions + bias.  Asserted.
    basis_m: int = 38
    #: Ridge for the basis_head regression head.  ⭐ Pinned, per w3 section 3.
    basis_ridge: float = 1.0
    #: Lloyd iterations for the basis dictionary.  ⭐ Pinned; a free budget here is
    #: exactly the "left the rest free" fault w3 section 3 names.
    basis_lloyd_iters: int = 25

    # -- the adversary (declared in the PREREG BEFORE it runs) -----------------
    adv_n_candidates: int = 96
    adv_steps: int = 100
    adv_lr: float = 0.05
    #: Learning rate for the CONVEX sweep (softmax parameterisation needs a larger
    #: step than the raw-coefficient one).  Pinned.
    adv_lr_convex: float = 0.2
    #: L1 cap on the signed coefficient vector.  ⛔ ``None`` = the UNCONSTRAINED
    #: sweep, measured to fail its own positive control (module docstring); it is
    #: reported as a failed move and ⛔ never as evidence.
    adv_l1_cap: Optional[float] = 2.0
    adv_seed: int = 4_242
    #: Requests / samples per interface move (M1, M3, M5).  ⭐ One knob for
    #: every judge pass, so a contended box is handled in ONE declared place.
    adv_n_probe: int = 256
    #: Cap on records per pinned query set, for plumbing smoke ONLY. 0 = the whole
    #: set, which is what every reported scoreboard uses.
    sb_max_records: int = 0
    #: Per-cell wall-clock cap in seconds (charter: cap it on EVERY launch).
    adv_cell_seconds: float = 900.0

    # -- churn -----------------------------------------------------------------
    #: Wave 2's deletion size, reproduced exactly: 8,234 of 32,768.
    churn_delete_n: int = 8_234
    churn_n_requests: int = 4_096

    meta: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping] = None) -> "ErasureConfig":
        mapping = dict(mapping or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"ErasureConfig: unknown key(s) {unknown}")
        for k in ("store_seeds", "held_out_judges"):
            if k in mapping:
                mapping[k] = tuple(mapping[k])
        return cls(**mapping)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["store_seeds"] = list(self.store_seeds)
        d["held_out_judges"] = list(self.held_out_judges)
        return d


def request_feature_names(cardinalities) -> list:
    """The head's request encoding: singles, then ALL PAIRWISE interactions, then bias.

    The pair-interaction block is not decoration: wave 2 measured that **the pair
    block, not the store, is what beats knn_proto** on this object.  A ``basis_head``
    without it is a weaker control than the one w3-hub-common section 1 names, and an
    under-powered like-for-like is not a like-for-like.
    """
    cs = [int(c) for c in cardinalities]
    names = [("single", f, v) for f, c in enumerate(cs) for v in range(c)]
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            for a in range(cs[i]):
                for b in range(cs[j]):
                    names.append(("pair", (i, a), (j, b)))
    names.append(("bias", -1, -1))
    return names


def assert_byte_budget(cfg: ErasureConfig, dim: int, cardinalities) -> dict:
    """The budget arithmetic, asserted rather than asserted-in-a-comment.

    ``basis_head`` is DECLARED AT THE FULL BUDGET and its ledger reports what it
    actually uses.  Its natural parameterisation cannot spend the last few hundred
    bytes, and over-stating the control's allowance is the conservative direction for
    a claim made against it.  The alternative -- quietly scoring it at a smaller
    number -- would be under-resourcing the like-for-like.
    """
    cs = [int(c) for c in cardinalities]
    sum_c = sum(cs)
    clu = int(cfg.n_atoms) * int(dim) * 4
    proto = (sum_c + 1) * int(dim) * 4
    P = len(request_feature_names(cs))
    per_atom = (int(dim) + P) * 4
    m_max = int(cfg.byte_budget) // per_atom
    used = int(cfg.basis_m) * per_atom
    if clu != cfg.byte_budget:
        raise AssertionError(f"clu deploys {clu} B, not the {cfg.byte_budget} B budget")
    if used > cfg.byte_budget:
        raise AssertionError(f"basis_head uses {used} B over budget")
    if int(cfg.basis_m) != m_max:
        raise AssertionError(
            f"basis_m={cfg.basis_m} is not the largest m the budget allows "
            f"({m_max}); an UNDER-resourced like-for-like is not a like-for-like")
    return {
        "budget_bytes": int(cfg.byte_budget),
        "clu": {"deployed_bytes": clu, "formula": "N*D*4", "n_atoms": int(cfg.n_atoms),
                "exactly_the_budget": True},
        "knn_proto": {"deployed_bytes": proto, "formula": "(sum_f C_f + 1)*D*4",
                      "headroom_bytes": int(cfg.byte_budget - proto)},
        "basis_head": {"deployed_bytes": int(cfg.byte_budget),
                       "used_bytes": used,
                       "unused_bytes": int(cfg.byte_budget - used),
                       "m": int(cfg.basis_m), "n_request_features": P,
                       "formula": "m*D*4 + P*m*4, P = singles + all pairs + bias",
                       "m_max_at_budget": m_max,
                       "declared_at": ("the FULL budget: the control is allotted "
                                       "524,288 B and cannot spend the last "
                                       f"{int(cfg.byte_budget - used)} B under its own "
                                       "parameterisation. Over-stating the control is "
                                       "the conservative direction")},
        "gate_bytes": GATE_BYTES,
        "note": ("the 8-byte gate is ADDED to the gated arms' deployed total and "
                 "declared; it is never netted off the CLU's side"),
    }


# ---------------------------------------------------------------------------
# 2. Deletion targets -- ⛔ declared, with the forbidden ones refused in code
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeletionTarget:
    """One value to delete, on one factor axis.

    ⛔ ``scale`` and ``orientation`` are refused here rather than avoided by
    convention: target file section 5 rules them out as physical defaults, and a
    convention is what a later hub breaks by accident.
    """

    factor: str
    factor_index: int
    value_index: int
    value: float
    label: str
    why: str

    def __post_init__(self):
        if self.factor in FORBIDDEN_DELETION_FACTORS:
            raise ValueError(
                f"⛔ {self.factor!r} is a PHYSICAL DEFAULT the model degrades towards "
                "(target file section 5). A generative model cannot delete the value "
                "it degrades TOWARDS, and the failure would be the renderer's, not "
                "the store's. ⛔ Not an eligible deletion target."
            )


#: ⭐ The two value targets, inherited verbatim from the pinned value-hold-out
#: condition (``chlu.data.gen_value_holdout``) so the split and the deletion agree.
TARGET_HUE = DeletionTarget(
    factor="object_hue", factor_index=F_OBJECT_HUE, value_index=5, value=0.5,
    label="object_hue = 0.5",
    why=("the interior mid-point of the CIRCULAR 10-value hue ring -- no hue is a "
         "physical default -- and the axis this hub's instrument reads back at 1.000 "
         "on real cached features, so a recovery is the store's and not the judge's"),
)
TARGET_CUBE = DeletionTarget(
    factor="shape", factor_index=F_SHAPE, value_index=0, value=0.0,
    label="shape = cube",
    why=("CATEGORICAL with no retained near-neighbour among {cylinder, sphere, "
         "oblong}: the one shape with flat faces and hard edges. ⛔ The sphere is "
         "excluded as the shape a degraded render degrades TOWARDS"),
)
VALUE_TARGETS = (TARGET_HUE, TARGET_CUBE)


# ---------------------------------------------------------------------------
# 3. Live sets -- the pools the stores are fitted on
# ---------------------------------------------------------------------------


def load_grid_and_split(paths: Mapping):
    """The pinned value-hold-out grid and its train/test row split."""
    from ..data.gen_splits import load_index

    grid, train_rows, test_rows, meta = load_index(str(paths["factor_index"]))
    return grid, np.asarray(train_rows, np.int64), np.asarray(test_rows, np.int64), meta


def build_live_sets(grid, train_rows: np.ndarray, target: DeletionTarget,
                    cfg: ErasureConfig, seed: int) -> dict:
    """The four matched pools one deletion is measured on.

    ``L_full``     the live set BEFORE the deletion: ``n_store_train`` rows drawn from
                   the universe of rows that are not in any held-out COMBINATION cell.
                   ⭐ It contains the target value at its natural rate.
    ``L_deleted``  ``L_full`` minus **every** row carrying the target value.  This is
                   the CLU's deletion: an operation on rows, then a refit.
    ``L_placebo``  ``L_full`` minus an equal number of rows chosen uniformly from the
                   rows that do **not** carry the target value.  ⭐ Matched deletion
                   SIZE, so a behaviour change is attributable to *which* rows left,
                   never to *how many*.
    ``L_identical`` ``L_full`` itself -- the churn NULL, whose churn must be 0.0000.

    ⛔ The universe excludes the pair/n-way test cells, so no store ever sees a
    held-out combination and rows 3-4 stay scorable.
    """
    fi = np.asarray(grid.factor_indices_of(np.arange(int(grid.n_rows) if hasattr(grid, "n_rows")
                                                     else 480_000)), np.int64)
    n_all = fi.shape[0]
    carries = fi[:, int(target.factor_index)] == int(target.value_index)
    # universe = train pool U every row carrying ANY held-out VALUE, minus the
    # held-out COMBINATION cells.  Equivalently: everything the value hold-out
    # removed for VALUE reasons is allowed back; nothing removed for COMBINATION
    # reasons is.
    in_train = np.zeros(n_all, bool)
    in_train[train_rows] = True
    value_held = np.zeros(n_all, bool)
    for t in VALUE_TARGETS:
        value_held |= fi[:, int(t.factor_index)] == int(t.value_index)
    universe = np.where(in_train | value_held)[0]
    rng = np.random.default_rng(int(cfg.rng_base) + 1_000_003 * int(seed)
                                + 7919 * int(target.factor_index))
    L_full = np.sort(rng.choice(universe, int(cfg.n_store_train), replace=False))
    hit = carries[L_full]
    n_hit = int(hit.sum())
    L_deleted = L_full[~hit]
    keep_idx = np.where(~hit)[0]
    drop = rng.choice(keep_idx, n_hit, replace=False)
    mask = np.ones(L_full.shape[0], bool)
    mask[drop] = False
    L_placebo = L_full[mask]
    if n_hit == 0:
        raise AssertionError(
            f"⛔ L_full carries no row with {target.label}; the deletion would be a "
            "no-op and every number below it would be meaningless"
        )
    return {
        "target": target.label,
        "seed": int(seed),
        "universe_n": int(universe.size),
        "L_full": L_full,
        "L_deleted": L_deleted,
        "L_placebo": L_placebo,
        "L_identical": L_full,
        "n_carrying_target": n_hit,
        "frac_carrying_target": float(n_hit / L_full.size),
        "sizes": {"L_full": int(L_full.size), "L_deleted": int(L_deleted.size),
                  "L_placebo": int(L_placebo.size), "L_identical": int(L_full.size)},
        "matched_deletion_size": bool(L_deleted.size == L_placebo.size),
        "note": ("⭐ L_deleted and L_placebo are the SAME size, so any difference "
                 "between them is which rows left, not how many"),
    }


def assert_value_absent(grid, rows: np.ndarray, target: DeletionTarget) -> dict:
    """⛔ The deletion actually happened, checked on the pool rather than believed."""
    fi = np.asarray(grid.factor_indices_of(np.asarray(rows, np.int64)), np.int64)
    n = int((fi[:, int(target.factor_index)] == int(target.value_index)).sum())
    if n:
        raise AssertionError(
            f"⛔ {n} of {rows.size} pool rows still carry {target.label}; the pool is "
            "not a deleted pool and nothing fitted on it is an erasure"
        )
    return {"target": target.label, "n_rows": int(rows.size), "n_carrying": 0,
            "absent": True}


# ---------------------------------------------------------------------------
# 4. The 8-byte gate -- the target file's section 1 object, built as a rival
# ---------------------------------------------------------------------------


class ValueBitmap:
    """57 bits -- one per (factor, grid value) -- saying "the live set holds this".

    ⭐ This is the object that killed refusal as a byte-matched claim (target file
    section 1): it scores 1.0000 on rows 1, 2, 5 and 6 at **8 bytes** against a
    524,288-byte budget.  It is built here as a **rival**, not as a straw man: a
    gated arm keeps its full-pool parameters and merely declines to serve a request
    naming a cleared bit.  ⛔ That is precisely why it is not an erasure -- the
    content is still in the parameters, and this hub's adversary goes and gets it.
    """

    def __init__(self, cardinalities, present: np.ndarray):
        self.cardinalities = tuple(int(c) for c in cardinalities)
        self.present = np.asarray(present, bool)
        if self.present.size != sum(self.cardinalities):
            raise ValueError("one bit per (factor, value) or it is not the 8-byte rule")

    @classmethod
    def from_pool(cls, grid, rows: np.ndarray) -> "ValueBitmap":
        fi = np.asarray(grid.factor_indices_of(np.asarray(rows, np.int64)), np.int64)
        bits = []
        for f, c in enumerate(grid.cardinalities):
            seen = np.zeros(int(c), bool)
            seen[np.unique(fi[:, f])] = True
            bits.append(seen)
        return cls(grid.cardinalities, np.concatenate(bits))

    def clear(self, factor_index: int, value_index: int) -> "ValueBitmap":
        """The deletion, as the GATE performs it: one bit, no refit, no forgetting."""
        off = int(sum(self.cardinalities[: int(factor_index)]))
        p = self.present.copy()
        p[off + int(value_index)] = False
        return ValueBitmap(self.cardinalities, p)

    def n_bytes(self) -> int:
        return int(np.ceil(self.present.size / 8.0))

    def accepts(self, value_indices: np.ndarray) -> np.ndarray:
        """``(n, F) on-grid value indices -> (n,) accept?``  ``-1`` = not requested."""
        V = np.asarray(value_indices, np.int64)
        ok = np.ones(V.shape[0], bool)
        for f, c in enumerate(self.cardinalities):
            off = int(sum(self.cardinalities[:f]))
            asked = V[:, f] >= 0
            if not asked.any():
                continue
            ok[asked] &= self.present[off + V[asked, f]]
        return ok

    def byte_ledger(self) -> dict:
        return {"bits": int(self.present.size), "bytes": self.n_bytes(),
                "n_present": int(self.present.sum()),
                "what": ("the target file's section 1 rule: one bit per (factor, grid "
                         "value). ⛔ It HOLDS nothing and FORGETS nothing -- which is "
                         "the whole point of this hub")}


# ---------------------------------------------------------------------------
# 5. The three arms.  The CLU arm calls the PINNED primitives, never a copy.
# ---------------------------------------------------------------------------


def load_features(paths: Mapping) -> np.ndarray:
    """The frozen cache, memory-mapped.  Opened read-only; never written."""
    return np.load(str(paths["features"]), mmap_mode="r")


def gather(F, rows: np.ndarray) -> np.ndarray:
    """``rows -> (n, D) float32``, chunked so a big pool never spikes resident memory."""
    r = np.asarray(rows, np.int64)
    out = np.empty((r.size, int(np.prod(F.shape[1:]))), np.float32)
    for i in range(0, r.size, 8192):
        sl = r[i: i + 8192]
        out[i: i + sl.size] = np.asarray(F[sl], np.float32).reshape(sl.size, -1)
    return out


def well_channels_from_plan(paths: Mapping) -> tuple:
    """The FROZEN ward grouping, read from ``store_plan.json``.  Never re-derived."""
    with open(paths["store_plan"]) as fh:
        plan = json.load(fh)
    lab = np.asarray(plan["labels"], np.int64)
    k = int(plan["K"])
    wells = tuple(tuple(int(i) for i in np.where(lab == j)[0]) for j in range(k))
    if sum(len(w) for w in wells) != int(plan["D"]):
        raise AssertionError("the frozen labels do not partition D")
    return wells, plan


def fit_clu(paths, F, grid, rows: np.ndarray, cfg: ErasureConfig, seed: int) -> dict:
    """The CLU arm: ``N`` verbatim train rows as atoms + the request address book.

    Every primitive is the pinned one -- ``build_atom_store``,
    ``assert_atoms_are_train_rows``, ``atom_bandwidths``, ``fit_atom_address_book``,
    ``AtomLandscape``, ``launch_state``.  Nothing is re-implemented here.
    """
    from ..core.gen_landscape import (AtomLandscape, assert_atoms_are_train_rows,
                                      atom_bandwidths, build_atom_store,
                                      fit_atom_address_book, store_byte_ledger)

    wells, plan = well_channels_from_plan(paths)
    Z = gather(F, rows)
    store = build_atom_store(wells, tuple(plan["feat_shape"]), Z, int(cfg.n_atoms),
                             str(cfg.atom_selection), int(seed))
    proof = assert_atoms_are_train_rows(store, Z)
    sigma = atom_bandwidths(store, float(cfg.sigma_frac), str(cfg.sigma_rule))
    labels = np.asarray(grid.factor_indices_of(rows), np.int64)
    book = fit_atom_address_book(store, Z, labels, grid.cardinalities, sigma,
                                 float(cfg.address_smoothing))
    land = AtomLandscape(store, sigma, None, 1.0, 1.0)
    led = store_byte_ledger(int(cfg.n_atoms), store.well_sizes)
    # The atoms are whole train rows: record WHICH rows, so the adversary's
    # weight-read move is auditable against the pool it was fitted on.
    pool_rows = np.asarray(rows, np.int64)[np.asarray(store.atom_rows, np.int64)]
    return {"arm": "clu", "store": store, "book": book, "land": land, "sigma": sigma,
            "atom_pool_rows": pool_rows, "byte_ledger": led,
            "deployed_bytes": int(led["store_bytes"]),
            "atoms_are_train_rows": proof, "n_fit_rows": int(rows.size)}


def clu_read(arm: dict, request_indices: np.ndarray, cfg: ErasureConfig) -> np.ndarray:
    """``(n, F) on-grid value indices -> (n, D)``.  The PINNED launch, zero steps."""
    from ..core.gen_landscape import launch_state

    lau = launch_state(arm["land"], arm["book"], np.asarray(request_indices, np.int64),
                       str(cfg.launch_mode), 0.5)
    return np.asarray(lau["x0"], np.float32)


def clu_weight_vectors(arm: dict) -> np.ndarray:
    """EVERYTHING the CLU store holds, as decodable feature vectors.

    The atom row index is shared across wells, so re-concatenating the per-well
    slices returns the ``N`` **whole stored train rows**.  There is nothing else in
    the store: this IS the adversary's complete weight read.
    """
    store = arm["store"]
    X = np.zeros((store.n_atoms, store.dim), np.float32)
    for k, w in enumerate(store.well_channels):
        X[:, np.asarray(w, np.int64)] = np.asarray(store.atoms[k], np.float32)
    return X


def fit_knn_proto(F, grid, rows: np.ndarray) -> dict:
    """``knn_proto``: the pinned per-factor prototype table, decoded directly."""
    from ..core.gen_request import FactorPrototypes

    Z = gather(F, rows)
    fp = FactorPrototypes.fit(grid, Z, np.asarray(rows, np.int64))
    led = fp.byte_ledger()
    return {"arm": "knn_proto", "proto": fp, "byte_ledger": led,
            "deployed_bytes": int(led["bytes"]), "n_fit_rows": int(rows.size)}


def knn_proto_read(arm: dict, request_indices: np.ndarray, grid) -> np.ndarray:
    fp = arm["proto"]
    V = np.asarray(request_indices, np.int64)
    out = np.empty((V.shape[0], fp.dim), np.float32)
    for i in range(V.shape[0]):
        terms = [{"factor_index": f, "value": float(grid.values[f][V[i, f]])}
                 for f in range(V.shape[1]) if V[i, f] >= 0]
        out[i] = fp.embed(terms)
    return out


def knn_proto_weight_vectors(arm: dict) -> tuple:
    """Every deployed prototype, as a decodable feature vector, WITH its label.

    This is the move the gate cannot stop: ``protos[object_hue][5]`` is the mean
    cached feature of every deleted row and it is still sitting in the table.
    """
    fp = arm["proto"]
    vecs, tags = [np.asarray(fp.mu_global, np.float32)], [("global_mean", -1, -1)]
    for f, P in enumerate(fp.protos):
        for v in range(P.shape[0]):
            vecs.append(np.asarray(P[v], np.float32))
            tags.append((str(fp.names[f]), int(f), int(v)))
    return np.stack(vecs), tags


def _nearest(Z: np.ndarray, B: np.ndarray, chunk: int = 4096) -> np.ndarray:
    g = (B * B).sum(1)
    out = np.empty(Z.shape[0], np.int64)
    for i in range(0, Z.shape[0], chunk):
        z = Z[i: i + chunk]
        out[i: i + z.shape[0]] = np.argmin(g[None, :] - 2.0 * (z @ B.T), axis=1)
    return out


def _onehot(fi: np.ndarray, cardinalities) -> np.ndarray:
    """``(n, F) value indices -> (n, P)``: singles, all pairwise interactions, bias.

    ``-1`` means "not requested" and contributes to no single and to no pair.
    """
    cs = [int(c) for c in cardinalities]
    offs, o = [], 0
    for c in cs:
        offs.append(o)
        o += c
    n_single = o
    pair_off, po = {}, n_single
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            pair_off[(i, j)] = po
            po += cs[i] * cs[j]
    P = po + 1
    out = np.zeros((fi.shape[0], P), np.float32)
    for f, c in enumerate(cs):
        asked = fi[:, f] >= 0
        out[np.where(asked)[0], offs[f] + fi[asked, f]] = 1.0
    for (i, j), base in pair_off.items():
        asked = (fi[:, i] >= 0) & (fi[:, j] >= 0)
        if asked.any():
            out[np.where(asked)[0], base + fi[asked, i] * cs[j] + fi[asked, j]] = 1.0
    out[:, -1] = 1.0
    return out


def fit_basis_head(F, grid, rows: np.ndarray, cfg: ErasureConfig, seed: int) -> dict:
    """**THE LIKE-FOR-LIKE** (w3-hub-common section 1).

    The wave-2 object: a **factored dictionary** over frozen encoder features plus a
    **request-conditioned linear regression head**, at matched bytes.

    * dictionary ``B`` ``(m, D)``: Lloyd/k-means over the live set's cached features,
      ``basis_lloyd_iters`` iterations from a seeded k-means++ init.  Both pinned.
    * head ``W`` ``(P, m)``: ridge regression of each row's dictionary coefficients on
      its one-hot factor encoding, ``P = sum_f C_f + 1`` (bias last).  Ridge pinned.

    Read: ``x_hat(r) = phi(r) @ W @ B``.  Deployed bytes: ``m*D*4 + P*m*4``.
    """
    Z = gather(F, rows)
    m = int(cfg.basis_m)
    rng = np.random.default_rng(int(cfg.rng_base) + 31 * int(seed) + 101)
    idx = [int(rng.integers(Z.shape[0]))]
    d2 = ((Z - Z[idx[0]]) ** 2).sum(1)
    for _ in range(m - 1):
        p = d2 / max(float(d2.sum()), 1e-30)
        j = int(rng.choice(Z.shape[0], p=p))
        idx.append(j)
        d2 = np.minimum(d2, ((Z - Z[j]) ** 2).sum(1))
    B = Z[np.asarray(idx, np.int64)].copy()
    for _ in range(int(cfg.basis_lloyd_iters)):
        a = _nearest(Z, B)
        for j in range(m):
            sel = a == j
            if sel.any():
                B[j] = Z[sel].mean(0)
    G = B @ B.T + 1e-3 * np.eye(m, dtype=np.float32)
    C = np.linalg.solve(G, (Z @ B.T).T).T.astype(np.float32)
    fi = np.asarray(grid.factor_indices_of(rows), np.int64)
    Phi = _onehot(fi, grid.cardinalities)
    P = Phi.shape[1]
    A = Phi.T @ Phi + float(cfg.basis_ridge) * np.eye(P, dtype=np.float32)
    W = np.linalg.solve(A, Phi.T @ C).astype(np.float32)
    used = int(B.size * 4 + W.size * 4)
    return {"arm": "basis_head", "B": B, "W": W, "P": P, "m": m,
            "cardinalities": tuple(int(c) for c in grid.cardinalities),
            "deployed_bytes": int(cfg.byte_budget),
            "byte_ledger": {"deployed_bytes": int(cfg.byte_budget),
                            "budget_bytes": int(cfg.byte_budget),
                            "used_bytes": used,
                            "unused_bytes": int(cfg.byte_budget) - used,
                            "m": m, "P": P,
                            "bytes_formula": "m*D*4 + P*m*4, P = singles+pairs+bias",
                            "dtype": "float32",
                            "what": ("THE LIKE-FOR-LIKE: a fitted dictionary over "
                                     "frozen encoder features + a request-conditioned "
                                     "linear head WITH the pair-interaction block "
                                     "wave 2 measured to be what beats knn_proto")},
            "n_fit_rows": int(rows.size)}


def basis_head_read_offgrid(arm: dict, request_indices: np.ndarray,
                            factor: int, value: float, grid) -> np.ndarray:
    """``basis_head`` answering a genuinely OFF-GRID term, by interpolation.

    The head's encoding is one-hot over grid values, so an off-grid value enters as
    the linear blend of its two bracketing one-hots -- the same interpolation
    :meth:`FactorPrototypes.prototype` performs, so the two table arms are asked the
    off-grid question in the same way.  Pair rows containing the factor are blended
    with the same weights, so the encoding stays consistent with ``_onehot``.
    """
    V = np.asarray(request_indices, np.int64)
    vals = np.asarray(grid.values[int(factor)], np.float64)
    j = int(np.searchsorted(vals, float(value)))
    lo = int(max(0, min(j - 1, vals.size - 2)))
    hi = lo + 1
    w = float(np.clip((float(value) - vals[lo]) / (vals[hi] - vals[lo]), 0.0, 1.0))
    Vlo, Vhi = V.copy(), V.copy()
    Vlo[:, int(factor)] = lo
    Vhi[:, int(factor)] = hi
    Phi = ((1.0 - w) * _onehot(Vlo, arm["cardinalities"])
           + w * _onehot(Vhi, arm["cardinalities"]))
    return (Phi @ arm["W"] @ arm["B"]).astype(np.float32)


def basis_head_read(arm: dict, request_indices: np.ndarray) -> np.ndarray:
    Phi = _onehot(np.asarray(request_indices, np.int64), arm["cardinalities"])
    return (Phi @ arm["W"] @ arm["B"]).astype(np.float32)


def basis_head_weight_vectors(arm: dict) -> tuple:
    """Every deployed parameter of the like-for-like, as decodable vectors.

    Three families, because the content could be in any of them and a weak adversary
    would check only one: the ``m`` **dictionary atoms**; every **single** head row
    ``(bias + w_v) @ B``; and every **pair-interaction** head row, which is where a
    conjunction's content would sit.
    """
    B, W = arm["B"], arm["W"]
    names = request_feature_names(arm["cardinalities"])
    vecs = [np.asarray(b, np.float32) for b in B]
    tags = [("dictionary_atom", -1, int(j)) for j in range(B.shape[0])]
    bias = W[-1]
    for r, nm in enumerate(names):
        if nm[0] == "bias":
            continue
        vecs.append(np.asarray((bias + W[r]) @ B, np.float32))
        if nm[0] == "single":
            tags.append(("head_single", int(nm[1]), int(nm[2])))
        else:
            tags.append((f"head_pair_f{nm[1][0]}v{nm[1][1]}_f{nm[2][0]}v{nm[2][1]}",
                         int(nm[1][0]), int(nm[1][1])))
    vecs.append(np.asarray(bias @ B, np.float32))
    tags.append(("head_bias", -1, -1))
    return np.stack(vecs), tags


def arm_span(arm: dict) -> np.ndarray:
    """The deployed vectors a coefficient sweep is allowed to combine, per arm.

    For ``clu`` it is the stored atoms; for ``knn_proto`` the prototype table; for
    ``basis_head`` the dictionary.  Each arm's sweep therefore ranges over exactly
    what that arm deploys -- which is what makes the sweep comparable across arms.
    """
    if arm["arm"] == "clu":
        return clu_weight_vectors(arm)
    if arm["arm"] == "knn_proto":
        return knn_proto_weight_vectors(arm)[0]
    if arm["arm"] == "basis_head":
        return np.asarray(arm["B"], np.float32)
    raise ValueError(f"unknown arm {arm['arm']!r}")


# ---------------------------------------------------------------------------
# 6. The judges.  ONE judge is attacked; the other two are HELD OUT and score.
# ---------------------------------------------------------------------------


def load_judges(paths: Mapping, cfg: ErasureConfig) -> dict:
    """The frozen decoder and the three external classifiers, hash-asserted.

    The attacked judge is named in the config; the other two never see a gradient.
    A recovery counts only when **all three** read the target value, which is what
    separates a recovered value from an adversarial example against one net.
    """
    from ..core.gen_bridge import param_hash
    from ..utils.checkpoints import load_model

    with open(paths["classifier_gate"]) as fh:
        gate = json.load(fh)
    if str(gate.get("gate")) != "PASS":
        raise AssertionError(
            f"PRECONDITION: classifier_gate.json reads gate={gate.get('gate')!r}, "
            "not 'PASS'. The external instrument is ungated, so nothing here is "
            "scorable. Report blocked."
        )
    dec = load_model(str(paths["decoder"]))
    dh = param_hash(dec)
    expect = str((gate.get("decoder") or {}).get("param_hash_per_seed", {})
                 .get(str(int(cfg.bridge_seed))))
    if expect and dh != expect:
        raise AssertionError(
            f"the decoder on disk hashes {dh[:16]} but the gate recorded {expect[:16]} "
            "-- this is not the decoder that was gated"
        )
    clfs = [load_model(str(paths[f"classifier_{s}"])) for s in (0, 1, 2)]
    return {"decoder": dec, "decoder_hash": dh,
            "classifiers": clfs,
            "classifier_hashes": [param_hash(c) for c in clfs],
            "gate": {"verdict": str(gate.get("gate")), "floor": gate.get("floor"),
                     "per_seed": gate.get("per_seed"),
                     "statistic": gate.get("statistic")},
            "attacked": int(cfg.attacked_judge),
            "held_out": [int(s) for s in cfg.held_out_judges]}


def judge_features(judges: dict, X: np.ndarray, feat_shape, batch: int = 128) -> dict:
    """``(n, D) feature vectors -> every judge's per-factor read-back.``"""
    from ..eval.gen_probe import decode_features, predict_factors

    img = decode_features(judges["decoder"],
                          np.asarray(X, np.float32).reshape((-1,) + tuple(feat_shape)),
                          int(batch))
    preds = [predict_factors(c, img, int(batch)) for c in judges["classifiers"]]
    return {"images": img, "preds": preds}


def recovery_scores(preds, target: DeletionTarget, attacked: int,
                    held_out) -> dict:
    """Did the adversary recover the deleted value?  Three readings, all reported.

    ``attacked_hit``   the attacked judge alone -- the loosest reading, and the one
                       an adversarial example inflates.
    ``heldout_hit``    BOTH held-out judges, neither of which saw a gradient.
    ``unanimous_hit``  all three.  **This is the headline number.**
    """
    f, v = int(target.factor_index), int(target.value_index)
    hits = [np.asarray(p)[:, f] == v for p in preds]
    a = hits[int(attacked)]
    ho = np.ones_like(a)
    for s in held_out:
        ho = ho & hits[int(s)]
    una = a & ho
    return {"n": int(a.size),
            "attacked_hit": float(a.mean()) if a.size else 0.0,
            "heldout_hit": float(ho.mean()) if ho.size else 0.0,
            "unanimous_hit": float(una.mean()) if una.size else 0.0,
            "n_unanimous": int(una.sum()),
            "unanimous_mask": una,
            "headline": "unanimous_hit",
            "note": ("the attacked judge alone is inflated by adversarial examples; "
                     "the held-out judges never saw a gradient")}


# ---------------------------------------------------------------------------
# 7. THE ADVERSARY -- built first, budget and move-set declared in the PREREG
# ---------------------------------------------------------------------------

#: The move set.  Declared BEFORE any move runs (target file section 2:
#: "an adversary invented after seeing the result proves nothing").
ADVERSARY_MOVES = {
    "M1_direct_request": (
        "ask the deployed read for the deleted value directly, on-grid, 512 requests "
        "structurally matched to row-1 (the pinned row5_held_out set)"),
    "M2_weight_read": (
        "THE decisive move against a gate. Decode EVERY deployed parameter vector as "
        "a feature vector and judge it: for clu the N stored train rows; for "
        "knn_proto the (sum_f C_f + 1) prototypes; for basis_head the m dictionary "
        "atoms AND the P head rows pushed through the dictionary. No gate can stop "
        "it, because it does not use the interface"),
    "M3_indirect_request": (
        "name the value without naming it: off-grid terms bracketing the deleted "
        "value, and requests with the deleted factor's term OMITTED so the arm must "
        "fill it in from whatever it still holds"),
    "M4_span_sweep": (
        "signed-coefficient gradient ascent on the ATTACKED judge's log p(target), "
        "over the arm's OWN deployed span, under a declared L1 cap. ADMISSIBLE ONLY "
        "IF its positive control returns ~0 -- measured, both ways, every run"),
    "M5_read_reachable": (
        "enumerate what the deployed read can actually emit -- for clu the per-well "
        "atom-index tuples, sampled -- and judge every one. This is the reachable "
        "set of the interface, with no gradient anywhere"),
}

#: The calibration every sweep-style move must pass before its number is read.
ADVERSARY_CALIBRATION = {
    "sensitivity": ("the SAME move against a store fitted on a pool that HOLDS the "
                    "value must recover it. A move that cannot is too weak and its "
                    "zero on the deleted store means nothing"),
    "specificity": ("the SAME move against (a) a dimension-matched RANDOM span and "
                    "(b) the landscape-removed store must recover ~nothing. A move "
                    "that recovers from noise is measuring the decoder"),
    "verdict_rule": ("a move is ADMISSIBLE iff sensitivity - specificity exceeds the "
                     "pre-registered separation. Otherwise it is reported as a FAILED "
                     "MOVE and its number is never quoted as evidence"),
}


def random_span(reference: np.ndarray, n: int, seed: int) -> np.ndarray:
    """The specificity control: ``n`` random vectors matched to the feature marginals.

    Matched in per-coordinate mean and std to the pool the real span came from, so
    the only thing that differs from the real span is **what is in it**.
    """
    rng = np.random.default_rng(int(seed))
    mu = reference.mean(0)
    sd = reference.std(0)
    return (rng.standard_normal((int(n), reference.shape[1])).astype(np.float32)
            * sd + mu).astype(np.float32)


def shuffled_span(span: np.ndarray, seed: int) -> np.ndarray:
    """The landscape-removed control: each coordinate permuted across the vectors.

    Same bytes, same marginals, the joint structure destroyed.  Charter C.2 requires
    a landscape-removed control on every arm; this is its span-sweep form.
    """
    rng = np.random.default_rng(int(seed))
    out = np.asarray(span, np.float32).copy()
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(out.shape[0]), j]
    return out


def span_sweep(span: np.ndarray, judges: dict, target: DeletionTarget,
               feat_shape, cfg: ErasureConfig, seed: int,
               deadline: Optional[float] = None) -> dict:
    """M4: signed coefficients over ``span``, ascending the attacked judge's logit.

    The coefficient vector is initialised at a single basis vector (one-hot) plus
    small noise and projected back onto the L1 ball after every step, so the sweep
    starts at something the arm could actually emit and stays within a declared
    distance of it.  ``adv_l1_cap = None`` removes the cap and reproduces the
    UNCONSTRAINED sweep, which is retained only so its failure can be shown.
    """
    import time as _time

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from ..core.gen_bridge import model_dtype

    dec = judges["decoder"]
    clf = judges["classifiers"][int(cfg.attacked_judge)]
    dt = model_dtype(dec)
    f, v = int(target.factor_index), int(target.value_index)
    A = jnp.asarray(np.asarray(span, np.float32))
    n_basis = int(A.shape[0])
    cap = cfg.adv_l1_cap

    def proj(c):
        if cap is None:
            return c
        s = jnp.abs(c).sum(-1, keepdims=True)
        return c * jnp.minimum(1.0, float(cap) / jnp.maximum(s, 1e-9))

    def loss(c):
        x = proj(c) @ A
        lg = jax.vmap(clf)(jax.vmap(dec)(
            x.reshape((-1,) + tuple(feat_shape)).astype(dt)))[f].astype(jnp.float32)
        return -(lg[:, v] - jax.nn.logsumexp(lg, axis=-1)).sum()

    g = eqx.filter_jit(jax.value_and_grad(loss))
    rng = np.random.default_rng(int(cfg.adv_seed) + 7919 * int(seed))
    nb = int(cfg.adv_n_candidates)
    c0 = np.zeros((nb, n_basis), np.float32)
    c0[np.arange(nb), rng.integers(0, n_basis, nb)] = 1.0
    c = jnp.asarray(c0 + rng.normal(0, 0.02, (nb, n_basis)).astype(np.float32))
    t0 = _time.time()
    steps_done = 0
    for _ in range(int(cfg.adv_steps)):
        val, grad = g(c)
        c = c - float(cfg.adv_lr) * grad
        steps_done += 1
        if deadline is not None and _time.time() > deadline:
            break
    X = np.asarray(proj(c) @ A, np.float32)
    jf = judge_features(judges, X, feat_shape)
    sc = recovery_scores(jf["preds"], target, cfg.attacked_judge, cfg.held_out_judges)
    sc.pop("unanimous_mask", None)
    return {"move": "M4_span_sweep", **sc,
            "n_basis": n_basis, "n_candidates": nb,
            "steps_requested": int(cfg.adv_steps), "steps_done": steps_done,
            "l1_cap": (None if cap is None else float(cap)),
            "lr": float(cfg.adv_lr), "seconds": float(_time.time() - t0),
            "hit_deadline": bool(steps_done < int(cfg.adv_steps))}


def weight_read(span_vecs: np.ndarray, tags, judges: dict, target: DeletionTarget,
                feat_shape, cfg: ErasureConfig) -> dict:
    """M2: decode every deployed parameter and judge it.  No gradient, no interface.

    This is the move that separates a gate from an erasure, and it is the cheapest
    move in the set: a gate cannot refuse it, because it never asks.

    **Two rates, and the TARGETED one is the headline.**  The rate over *all*
    parameters has the wrong denominator: an adversary does not need half the
    parameters to leak, it needs **one**, and it knows which one to read.
    ``targeted_hit`` is the rate over the parameters *labelled with the deleted
    value* -- ``protos[object_hue][5]`` for the prototype table, the
    ``object_hue = 5`` single and every pair row containing it for the like-for-like.
    An arm whose parameters carry no value labels at all (``clu``: its atoms are whole
    train rows, indexed by nothing) has no targeted set, and the all-parameter rate is
    then the right and only statistic -- which is exactly the asymmetry this hub is
    about.
    """
    jf = judge_features(judges, span_vecs, feat_shape)
    sc = recovery_scores(jf["preds"], target, cfg.attacked_judge, cfg.held_out_judges)
    mask = sc.pop("unanimous_mask")
    f, v = int(target.factor_index), int(target.value_index)
    targeted = np.zeros(len(tags), bool) if tags is not None else np.zeros(0, bool)
    if tags is not None:
        for i, t in enumerate(tags):
            targeted[i] = (int(t[1]) == f and int(t[2]) == v)
    hits = [dict(zip(("kind", "factor_index", "value_index"), tags[i]))
            for i in np.where(mask)[0]] if tags is not None else []
    out = {"move": "M2_weight_read", **sc,
           "n_parameters_decoded": int(span_vecs.shape[0]),
           "recovering_parameters": hits[:32],
           "n_recovering_parameters": int(mask.sum()),
           "any_parameter_recovers": bool(mask.any()),
           "n_targeted_parameters": int(targeted.sum()),
           "note": ("a gate cannot refuse this move because it never asks; what it "
                    "finds is what the deployed parameters still CONTAIN")}
    if int(targeted.sum()):
        out["targeted_hit"] = float(mask[targeted].mean())
        out["targeted_note"] = (
            f"{int((mask & targeted).sum())}/{int(targeted.sum())} of the parameters "
            f"LABELLED with {target.label} decode to it")
        out["headline_rate"] = "targeted_hit"
    else:
        out["targeted_hit"] = None
        out["targeted_note"] = (
            "this arm's parameters carry no value labels, so there is no targeted "
            "set and the all-parameter rate is the only statistic")
        out["headline_rate"] = "unanimous_hit"
    return out


def move_recovery(move) -> Optional[float]:
    """The number a move's recovery is READ at.  One place, so nothing drifts.

    * an interface move is read **AFTER THE GATE**, because that is what the deployed
      arm actually emits -- crediting an adversary with a request the gate refuses
      would overstate the gated baseline's leakage, and that is the one direction
      this hub must not err in;
    * ``M2`` is read at its **targeted** rate where the arm's parameters carry value
      labels, and at the all-parameter rate otherwise.
    """
    if not isinstance(move, Mapping):
        return None
    if move.get("move") == "M2_weight_read":
        t = move.get("targeted_hit")
        if t is not None:
            return float(t)
        return None if move.get("unanimous_hit") is None else float(move["unanimous_hit"])
    if "unanimous_hit_after_gate" in move:
        return float(move["unanimous_hit_after_gate"])
    if "unanimous_hit" in move:
        return float(move["unanimous_hit"])
    return None

def convex_sweep(span: np.ndarray, judges: dict, target: DeletionTarget,
                 feat_shape, cfg: ErasureConfig, seed: int,
                 deadline: Optional[float] = None) -> dict:
    """M4a: the CONVEX span sweep -- ``c = softmax(theta)``, always on the simplex.

    Why this exists and the signed sweep does not survive: measured 2026-09-18, with
    **no optimisation at all**, 128 random *signed* L1 = 1 combinations of atoms from
    a pool that never held the value already read as that value at 0.125 (hue) and
    0.195 (cube), while 128 random *convex* combinations of the same atoms read 0.000
    and 0.000.  Free signed coefficients leave the feature manifold and the frozen
    decoder renders the off-manifold point as whatever is asked for.  The simplex is
    the largest coefficient set whose zero-move floor is 0.000.
    """
    import time as _time

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from ..core.gen_bridge import model_dtype

    dec = judges["decoder"]
    clf = judges["classifiers"][int(cfg.attacked_judge)]
    dt = model_dtype(dec)
    f, v = int(target.factor_index), int(target.value_index)
    A = jnp.asarray(np.asarray(span, np.float32))
    n_basis = int(A.shape[0])

    def loss(t):
        x = jax.nn.softmax(t, axis=-1) @ A
        lg = jax.vmap(clf)(jax.vmap(dec)(
            x.reshape((-1,) + tuple(feat_shape)).astype(dt)))[f].astype(jnp.float32)
        return -(lg[:, v] - jax.nn.logsumexp(lg, axis=-1)).sum()

    g = eqx.filter_jit(jax.value_and_grad(loss))
    rng = np.random.default_rng(int(cfg.adv_seed) + 7919 * int(seed) + 13)
    nb = int(cfg.adv_n_candidates)
    t0 = np.zeros((nb, n_basis), np.float32)
    t0[np.arange(nb), rng.integers(0, n_basis, nb)] = 4.0
    t = jnp.asarray(t0 + rng.normal(0, 0.05, t0.shape).astype(np.float32))
    start = _time.time()
    steps = 0
    for _ in range(int(cfg.adv_steps)):
        val, grad = g(t)
        t = t - float(cfg.adv_lr_convex) * grad
        steps += 1
        if deadline is not None and _time.time() > deadline:
            break
    X = np.asarray(jax.nn.softmax(t, axis=-1) @ A, np.float32)
    jf = judge_features(judges, X, feat_shape)
    sc = recovery_scores(jf["preds"], target, cfg.attacked_judge, cfg.held_out_judges)
    sc.pop("unanimous_mask", None)
    return {"move": "M4a_convex_span_sweep", **sc, "n_basis": n_basis,
            "n_candidates": nb, "steps_requested": int(cfg.adv_steps),
            "steps_done": steps, "lr": float(cfg.adv_lr_convex),
            "seconds": float(_time.time() - start),
            "hit_deadline": bool(steps < int(cfg.adv_steps))}


def read_reachable(arm: dict, judges: dict, target: DeletionTarget, feat_shape,
                   cfg: ErasureConfig, seed: int, n: Optional[int] = None) -> dict:
    """M5: what the DEPLOYED read can actually emit, sampled, with no gradient.

    For ``clu`` the read picks one atom index per well, so the reachable set is the
    ``N ** K`` per-well recombinations; ``n`` of them are drawn uniformly and judged.
    For the table arms the read is a fixed linear map of the request, so the
    reachable set is swept by the request moves (M1/M3) and this move returns
    ``NOT APPLICABLE`` rather than a zero -- a zero here would read as evidence.
    """
    n = int(cfg.adv_n_probe) if n is None else int(n)
    if arm["arm"] != "clu":
        return {"move": "M5_read_reachable", "status": "NOT APPLICABLE",
                "why": ("the read is a fixed linear map of the request, so its "
                        "reachable set is exactly what M1/M3 sweep; a 0.0 here "
                        "would be an artefact of the move, not a property of the arm")}
    store = arm["store"]
    rng = np.random.default_rng(int(cfg.adv_seed) + 104_729 * int(seed))
    idx = rng.integers(0, store.n_atoms, (int(n), store.n_wells))
    X = np.zeros((int(n), store.dim), np.float32)
    for k, w in enumerate(store.well_channels):
        X[:, np.asarray(w, np.int64)] = np.asarray(store.atoms[k], np.float32)[idx[:, k]]
    jf = judge_features(judges, X, feat_shape)
    sc = recovery_scores(jf["preds"], target, cfg.attacked_judge, cfg.held_out_judges)
    sc.pop("unanimous_mask", None)
    return {"move": "M5_read_reachable", **sc, "n_sampled": int(n),
            "reachable_set_size": f"{store.n_atoms}**{store.n_wells}",
            "gradient_used": False}


def request_moves(arm: dict, grid, judges: dict, target: DeletionTarget, feat_shape,
                  cfg: ErasureConfig, seed: int, gate: Optional["ValueBitmap"] = None,
                  n: Optional[int] = None) -> dict:
    """M1 + M3: drive the deployed interface, with and without the gate in the way.

    ``M1`` names the deleted value on-grid, in an otherwise ordinary 6-term request.
    ``M3`` names it without naming it, two ways: off-grid terms that bracket it, and
    requests with the deleted factor's term **omitted** so the arm must fill the slot
    from whatever it still holds.
    """
    n = int(cfg.adv_n_probe) if n is None else int(n)
    rng = np.random.default_rng(int(cfg.adv_seed) + 15_485_863 * int(seed))
    F_ = len(grid.cardinalities)
    f, v = int(target.factor_index), int(target.value_index)
    base = np.stack([rng.integers(0, int(c), int(n)) for c in grid.cardinalities], 1)

    direct = base.copy()
    direct[:, f] = v
    omitted = direct.copy()
    omitted[:, f] = -1

    def _read(V):
        if arm["arm"] == "clu":
            return clu_read(arm, V, cfg)
        if arm["arm"] == "knn_proto":
            return knn_proto_read(arm, V, grid)
        return basis_head_read(arm, V)

    out = {}
    for name, V, move in (("M1_direct_request", direct, "M1"),
                          ("M3_term_omitted", omitted, "M3")):
        served = np.ones(V.shape[0], bool)
        if gate is not None:
            served = gate.accepts(V)
        X = _read(V)
        jf = judge_features(judges, X, feat_shape)
        sc = recovery_scores(jf["preds"], target, cfg.attacked_judge,
                             cfg.held_out_judges)
        mask = sc.pop("unanimous_mask")
        out[name] = {"move": move, **sc, "n_requests": int(V.shape[0]),
                     "gate_served_frac": float(served.mean()),
                     "unanimous_hit_after_gate": float((mask & served).mean()),
                     "note": ("`unanimous_hit` is what the arm WOULD emit; "
                              "`unanimous_hit_after_gate` is what the deployed, "
                              "gated arm actually emits")}
    # M3, second form: a GENUINE off-grid request bracketing the deleted value.
    # This is where the 8-byte gate is structurally blind: its vocabulary is the
    # factor GRID, so a term that is not a grid value has no bit to test and the rule
    # can only accept it.  Whether that buys the adversary anything depends on whether
    # the arm's read INTERPOLATES, which differs by arm and is reported per arm.
    vals = np.asarray(grid.values[f], np.float64)
    if arm["arm"] == "clu":
        out["M3_offgrid_bracket"] = {
            "move": "M3", "applicable": False,
            "why": ("NOT APPLICABLE to `clu`, and this is a property of the interface, "
                    "not a gap in the adversary: the pinned request is a factor -> "
                    "GRID VALUE assignment (`atom_posterior` indexes "
                    "`log_cond[f][value_index]`), so there is no off-grid channel to "
                    "exploit. Reported as inapplicable, never as 0.0")}
    elif not (vals.size > 2 and 0 < v < vals.size - 1):
        out["M3_offgrid_bracket"] = {
            "move": "M3", "applicable": False,
            "why": (f"{target.factor} is CATEGORICAL or the value is on the range "
                    "boundary, so no off-grid value brackets it. Reported as "
                    "inapplicable, never as 0.0")}
    else:
        step = float(min(vals[v] - vals[v - 1], vals[v + 1] - vals[v]))
        nb = min(64, n)
        Xs, labels = [], []
        for e in (0.02, 0.10, 0.25, 0.45):
            for sgn in (-1.0, 1.0):
                off = float(vals[v] + sgn * e * step)
                labels.append({"value": off, "frac_of_grid_step": e})
                if arm["arm"] == "knn_proto":
                    terms = [[{"factor_index": k,
                               "value": float(off if k == f
                                              else grid.values[k][base[i, k]])}
                              for k in range(F_)] for i in range(nb)]
                    Xs.append(arm["proto"].embed_many(terms))
                else:
                    Xs.append(basis_head_read_offgrid(arm, base[:nb], f, off, grid))
        Xc = np.concatenate(Xs)
        jf = judge_features(judges, Xc, feat_shape)
        sc = recovery_scores(jf["preds"], target, cfg.attacked_judge,
                             cfg.held_out_judges)
        sc.pop("unanimous_mask", None)
        out["M3_offgrid_bracket"] = {
            "move": "M3", **sc, "applicable": True, "offsets": labels,
            "gate_served_frac": 1.0,
            "unanimous_hit_after_gate": float(sc["unanimous_hit"]),
            "gate_note": ("the 8-byte rule is keyed on GRID values, so it has no bit "
                          "for an off-grid term and serves every one of these. "
                          "Whatever this move recovers, the gate did not stop it")}
    return out

# ---------------------------------------------------------------------------
# 8. CLAIM 2 -- rebuild churn, with per-request predictions PERSISTED
# ---------------------------------------------------------------------------

#: Wave 2 measured 18.8% +- 0.8% against 1.40% +- 0.05% -- a 13.4x ratio -- but kept
#: AGGREGATES ONLY, so the ratio was never auditable.  Target file section 3: persist
#: per-request predictions, use the identical-live-set rebuild as the null, >= 3 store
#: seeds, pre-register before looking.
CHURN_NULL = "identical_live_set"
CHURN_PRIOR = {"wave2_clu": 0.188, "wave2_clu_sd": 0.008,
               "wave2_table": 0.0140, "wave2_table_sd": 0.0005,
               "wave2_ratio": 13.4,
               "caveat": ("wave 2 kept aggregates only; these are quoted as the prior "
                          "this hub tests, never as this hub's numbers")}


def churn_requests(grid, n: int, seed: int) -> np.ndarray:
    """``n`` ordinary on-grid 6-term requests.  The same set for every arm and store."""
    rng = np.random.default_rng(20_260_919 + 7 * int(seed))
    return np.stack([rng.integers(0, int(c), int(n)) for c in grid.cardinalities], 1)


def predict_for_store(arm: dict, grid, judges: dict, requests: np.ndarray,
                      feat_shape, cfg: ErasureConfig) -> np.ndarray:
    """``(n_judges, n_requests, n_factors)`` -- the PERSISTED per-request predictions."""
    if arm["arm"] == "clu":
        X = clu_read(arm, requests, cfg)
    elif arm["arm"] == "knn_proto":
        X = knn_proto_read(arm, requests, grid)
    else:
        X = basis_head_read(arm, requests)
    jf = judge_features(judges, X, feat_shape)
    return np.stack([np.asarray(p, np.int16) for p in jf["preds"]])


def churn_between(before: np.ndarray, after: np.ndarray) -> dict:
    """Fraction of requests whose ANSWER changed, per judge and unanimously.

    The answer is the judge's read-back factor tuple.  ``per_factor`` is reported too,
    because "18.8% of answers changed" hides whether one factor moved or all six.
    """
    out = {}
    for j in range(before.shape[0]):
        ch = (before[j] != after[j])
        out[f"judge_{j}"] = {
            "tuple_changed": float(ch.any(axis=1).mean()),
            "per_factor": [float(x) for x in ch.mean(axis=0)],
            "mean_factors_changed": float(ch.sum(axis=1).mean()),
        }
    any_j = np.zeros(before.shape[1], bool)
    all_j = np.ones(before.shape[1], bool)
    for j in range(before.shape[0]):
        c = (before[j] != after[j]).any(axis=1)
        any_j |= c
        all_j &= c
    out["any_judge_changed"] = float(any_j.mean())
    out["all_judges_changed"] = float(all_j.mean())
    out["headline"] = "judge_0.tuple_changed"
    out["n_requests"] = int(before.shape[1])
    return out


# ---------------------------------------------------------------------------
# 9. CLAIM 3 -- conjunction deletion, on a DECLARED non-cylinder cell
# ---------------------------------------------------------------------------

#: The cell, declared before any conjunction runs.  Target file section 4: the wave-2
#: scored cell is entirely ``shape = cylinder`` and cannot test a shape conjunction.
#: ``shape = 3`` (oblong/capsule) is the non-cylinder cell used here, and it is
#: neither the cube (this hub's VALUE-deletion target) nor the sphere (the shape a
#: degraded render degrades towards).
CONJUNCTION_CELL = {
    "factor": "shape", "factor_index": F_SHAPE, "value_index": 3,
    "label": "shape = 3 (oblong/capsule)",
    "why": ("NON-CYLINDER, as the target file requires -- the wave-2 scored cell is "
            "entirely shape = cylinder. Not the cube, which is this hub's value "
            "deletion target and must stay independent; not the sphere, which is the "
            "shape a degraded render degrades towards"),
}


@dataclass(frozen=True)
class Conjunction:
    """One ``(factor_a = v_a AND factor_b = v_b)`` deletion, with its placebo."""

    name: str
    factor_a: int
    value_a: int
    factor_b: int
    value_b: int
    why: str

    def mask(self, fi: np.ndarray) -> np.ndarray:
        return ((fi[:, self.factor_a] == self.value_a)
                & (fi[:, self.factor_b] == self.value_b))


#: >= 3 conjunctions, pre-registered, all inside the declared non-cylinder cell and
#: all avoiding ``scale`` and ``orientation``.
CONJUNCTIONS = (
    Conjunction("objhue2_x_oblong", F_OBJECT_HUE, 2, F_SHAPE, 3,
                "the object's own photometric property crossed with the cell's shape"),
    Conjunction("floorhue7_x_oblong", F_FLOOR_HUE, 7, F_SHAPE, 3,
                "a BACKGROUND hue crossed with the cell's shape -- the contrast with "
                "conjunction 1 says whether the effect needs the object's own axis"),
    Conjunction("wallhue4_x_oblong", F_WALL_HUE, 4, F_SHAPE, 3,
                "the second background axis, as a within-family replicate of 2"),
)


def conjunction_pools(grid, L: np.ndarray, conj: Conjunction, seed: int) -> dict:
    """``L`` minus the conjunction, and the size-matched placebo inside the cell.

    The placebo removes the SAME number of rows, drawn from rows that sit in the
    declared cell but do **not** satisfy the conjunction -- so the cell's density
    falls by the same amount and only the conjunction's own rows differ.
    """
    fi = np.asarray(grid.factor_indices_of(L), np.int64)
    hit = conj.mask(fi)
    n_hit = int(hit.sum())
    if n_hit == 0:
        raise AssertionError(f"{conj.name}: the live set holds no such row")
    in_cell = fi[:, int(CONJUNCTION_CELL["factor_index"])] == int(
        CONJUNCTION_CELL["value_index"])
    eligible = np.where(in_cell & ~hit)[0]
    if eligible.size < n_hit:
        raise AssertionError(
            f"{conj.name}: only {eligible.size} placebo-eligible rows in the cell for "
            f"{n_hit} deleted rows -- a size-matched placebo is not constructible")
    rng = np.random.default_rng(20_260_919 + 977 * int(seed) + hash(conj.name) % 9973)
    drop = rng.choice(eligible, n_hit, replace=False)
    m = np.ones(L.size, bool)
    m[drop] = False
    return {"name": conj.name, "n_deleted": n_hit,
            "L_deleted": L[~hit], "L_placebo": L[m],
            "size_matched": bool(int((~hit).sum()) == int(m.sum()))}


def conjunction_retained(arm: dict, grid, judges: dict, conj: Conjunction,
                         feat_shape, cfg: ErasureConfig, seed: int,
                         n: Optional[int] = None) -> dict:
    """Ask for the conjunction; does the output still carry BOTH values?

    ``retained`` is unanimous over three judges on **both** factors at once, which is
    what "the conjunction survived" has to mean.  The two single-factor rates are
    reported beside it: a drop in one of them is a different failure from a drop in
    the pair, and wave 2 could not tell them apart.
    """
    n = int(cfg.adv_n_probe) if n is None else int(n)
    rng = np.random.default_rng(int(cfg.adv_seed) + 6_700_417 * int(seed))
    V = np.stack([rng.integers(0, int(c), int(n)) for c in grid.cardinalities], 1)
    V[:, conj.factor_a] = conj.value_a
    V[:, conj.factor_b] = conj.value_b
    if arm["arm"] == "clu":
        X = clu_read(arm, V, cfg)
    elif arm["arm"] == "knn_proto":
        X = knn_proto_read(arm, V, grid)
    else:
        X = basis_head_read(arm, V)
    jf = judge_features(judges, X, feat_shape)
    P = jf["preds"]
    a = np.ones(int(n), bool)
    b = np.ones(int(n), bool)
    for p in P:
        a &= (np.asarray(p)[:, conj.factor_a] == conj.value_a)
        b &= (np.asarray(p)[:, conj.factor_b] == conj.value_b)
    return {"conjunction": conj.name, "n_requests": int(n),
            "retained_pair": float((a & b).mean()),
            "retained_factor_a": float(a.mean()),
            "retained_factor_b": float(b.mean()),
            "criterion": "unanimous over 3 judges on BOTH factors"}


def bitmap_cannot_express_a_conjunction(grid, L: np.ndarray, conj: Conjunction) -> dict:
    """The section-4 contrast, computed exactly rather than argued.

    A per-factor bitmap has one bit per (factor, value).  After a conjunction
    deletion **both** values still occur in the live set on their own, so **no bit
    changes** and the gate serves the deleted conjunction at its full rate.  The only
    gate that could express it is the full joint table -- which then refuses every
    combination the live set does not contain, i.e. every row-3 and row-4 request.
    """
    fi = np.asarray(grid.factor_indices_of(L), np.int64)
    before = ValueBitmap.from_pool(grid, L)
    after = ValueBitmap.from_pool(grid, L[~conj.mask(fi)])
    n_changed = int((before.present != after.present).sum())
    n_combinations = int(np.prod([int(c) for c in grid.cardinalities]))
    joint_bytes = int(np.ceil(n_combinations / 8.0))
    return {
        "conjunction": conj.name,
        "bitmap_bits_changed_by_the_deletion": n_changed,
        "bitmap_bytes": before.n_bytes(),
        "bitmap_can_express_it": bool(n_changed > 0),
        "joint_table_bytes": joint_bytes,
        "joint_table_n_combinations": n_combinations,
        "joint_table_refuses_every_unseen_combination": True,
        "consequence": (
            "the 8-byte per-factor bitmap changes NO bit, so the gated arm keeps "
            "serving the deleted conjunction; the joint table can express it but "
            "then refuses every combination absent from the live set, which is "
            "every row-3 and row-4 request -- so it buys the deletion by zeroing "
            "compositional generation"),
    }


# ---------------------------------------------------------------------------
# 10. The run -- resumable by artefact, wall-clock capped, one cell at a time
# ---------------------------------------------------------------------------

#: The conditions of claim 1.  Every one is a (arm, pool, gate) triple, and the two
#: controls the target file makes mandatory are conditions here rather than options.
CLAIM1_CONDITIONS = (
    ("clu_full", "clu", "L_full", False,
     "SENSITIVITY CEILING: the CLU store fitted on a pool that HOLDS the value. An "
     "adversary that cannot recover here is too weak and every zero below is void"),
    ("clu_deleted", "clu", "L_deleted", False,
     "THE TREATMENT: the CLU store refitted with no row carrying the value"),
    ("clu_placebo", "clu", "L_placebo", False,
     "PLACEBO: the same number of rows deleted, none of them carrying the value"),
    ("clu_shuffled", "clu", "L_deleted", False,
     "LANDSCAPE-REMOVED (charter C.2): the deleted store's atoms, coordinate-"
     "shuffled. Same bytes, same marginals, the content destroyed"),
    ("basis_head_gated", "basis_head", "L_full", True,
     "THE LIKE-FOR-LIKE (w3 section 1): the fitted basis + linear head, full-pool "
     "parameters, an 8-byte deletion bitmap bolted on. It refuses correctly and "
     "STILL HOLDS THE CONTENT"),
    ("basis_head_refit", "basis_head", "L_deleted", False,
     "the like-for-like doing what the CLU does -- refitting. Reported so the claim "
     "is about GATING vs ERASURE, not about our arm vs their handicap"),
    ("knn_proto_gated", "knn_proto", "L_full", True,
     "the second control: the prototype table with the same 8-byte gate"),
    ("knn_proto_refit", "knn_proto", "L_deleted", False,
     "the second control, refitted"),
    ("random_span", "random", "L_deleted", False,
     "SPECIFICITY CONTROL: a dimension-matched random span, matched to the deleted "
     "pool's per-coordinate mean and std. Anything recovered here is the decoder's"),
)


def _write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=_jsonable)
    tmp.replace(path)
    return path


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def build_condition(cond_id, arm_name, pool_key, gated, paths, F, grid, pools,
                    target, cfg, seed):
    """One condition's fitted object, its gate and its byte ledger."""
    rows = pools[pool_key]
    arm = None
    gate = None
    if arm_name == "clu":
        arm = fit_clu(paths, F, grid, rows, cfg, seed)
        if cond_id == "clu_shuffled":
            X = shuffled_span(clu_weight_vectors(arm),
                              int(cfg.adv_seed) + 1_000_003 * int(seed))
            arm = dict(arm)
            arm["shuffled_span"] = X
    elif arm_name == "knn_proto":
        arm = fit_knn_proto(F, grid, rows)
    elif arm_name == "basis_head":
        arm = fit_basis_head(F, grid, rows, cfg, seed)
    elif arm_name == "random":
        Z = gather(F, rows[: min(rows.size, 4096)])
        arm = {"arm": "random", "span": random_span(
            Z, int(cfg.n_atoms), int(cfg.adv_seed) + 99_991 * int(seed)),
            "deployed_bytes": int(cfg.n_atoms) * Z.shape[1] * 4,
            "byte_ledger": {"bytes": int(cfg.n_atoms) * Z.shape[1] * 4,
                            "what": "a dimension-matched RANDOM span; not an arm"}}
    if gated:
        gate = ValueBitmap.from_pool(grid, rows).clear(target.factor_index,
                                                       target.value_index)
    return arm, gate


def condition_span(cond_id, arm):
    if cond_id == "clu_shuffled":
        return arm["shuffled_span"]
    if arm["arm"] == "random":
        return arm["span"]
    return arm_span(arm)


def run_claim1_cell(cond, paths, F, grid, judges, pools, target, cfg, seed,
                    feat_shape, out_dir: Path) -> dict:
    """One (condition, target, seed) cell of claim 1.  Written the moment it ends."""
    import time as _time

    cond_id, arm_name, pool_key, gated, why = cond
    stem = f"claim1__{target.factor}_{target.value_index}__{cond_id}__seed{seed}"
    path = out_dir / "claim1" / f"{stem}.json"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        rec["resumed"] = True
        return rec
    t0 = _time.time()
    arm, gate = build_condition(cond_id, arm_name, pool_key, gated, paths, F, grid,
                                pools, target, cfg, seed)
    span = condition_span(cond_id, arm)
    deadline = t0 + float(cfg.adv_cell_seconds)
    rec = {
        "cell": stem, "condition": cond_id, "arm": arm_name, "pool": pool_key,
        "gated": bool(gated), "why": why, "target": target.label, "seed": int(seed),
        "n_fit_rows": int(pools[pool_key].size),
        "deployed_bytes": int(arm.get("deployed_bytes", 0)) + (GATE_BYTES if gated else 0),
        "byte_ledger": arm.get("byte_ledger"),
        "gate_bytes": GATE_BYTES if gated else 0,
        "moves": {},
    }
    # -- M2: read the parameters.  The decisive move, and the cheapest. ------
    if arm_name == "knn_proto":
        vecs, tags = knn_proto_weight_vectors(arm)
    elif arm_name == "basis_head":
        vecs, tags = basis_head_weight_vectors(arm)
    elif arm_name == "clu":
        vecs = span
        tags = [("stored_train_row", -1, int(i)) for i in range(vecs.shape[0])]
    else:
        vecs, tags = span, [("random_vector", -1, int(i)) for i in range(span.shape[0])]
    rec["moves"]["M2_weight_read"] = weight_read(vecs, tags, judges, target,
                                                 feat_shape, cfg)
    # -- M4a: the convex span sweep (admissible; calibrated every run) -------
    rec["moves"]["M4a_convex_span_sweep"] = convex_sweep(
        span, judges, target, feat_shape, cfg, seed, deadline)
    # -- M5: the deployed read's reachable set -------------------------------
    if arm_name == "clu":
        rec["moves"]["M5_read_reachable"] = read_reachable(
            arm, judges, target, feat_shape, cfg, seed)
    # -- M1 / M3: drive the interface ----------------------------------------
    if arm_name in ("clu", "knn_proto", "basis_head"):
        rec["moves"].update(request_moves(arm, grid, judges, target, feat_shape,
                                          cfg, seed, gate))
    rec["seconds"] = float(_time.time() - t0)
    scored = {k: move_recovery(m) for k, m in rec["moves"].items()}
    scored = {k: v for k, v in scored.items() if v is not None}
    rec["move_recovery"] = scored
    rec["best_unanimous_hit"] = max(scored.values()) if scored else 0.0
    rec["best_move"] = (max(scored.items(), key=lambda kv: kv[1])[0]
                        if scored else "NONE")
    rec["recovery_rule"] = (
        "interface moves are read AFTER the gate (what the deployed arm emits); M2 "
        "is read at its targeted rate where the parameters carry value labels")
    _write_json(rec, path)
    return rec


def open_run(cfg: ErasureConfig, out_root: Optional[str] = None) -> dict:
    """Load the frozen instrument once and assert everything that gates a number."""
    out_dir = Path(out_root or os.environ.get("CHLU_HUB_OUTPUTS", ".")) / "hub-erasure"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = instrument_paths(int(cfg.bridge_seed))
    cache = assert_cache_is_read_only(paths)
    grid, train_rows, test_rows, meta = load_grid_and_split(paths)
    F = load_features(paths)
    judges = load_judges(paths, cfg)
    ledger = assert_byte_budget(cfg, int(cache["D"]), grid.cardinalities)
    ctx = {"cfg": cfg, "paths": paths, "cache": cache, "grid": grid,
           "train_rows": train_rows, "test_rows": test_rows, "F": F,
           "judges": judges, "byte_ledger": ledger, "out_dir": out_dir,
           "feat_shape": cache["feat_shape"]}
    _write_json({"config": cfg.to_dict(), "cache": cache, "byte_ledger": ledger,
                 "decoder_hash": judges["decoder_hash"],
                 "classifier_hashes": judges["classifier_hashes"],
                 "classifier_gate": judges["gate"],
                 "adversary_moves": ADVERSARY_MOVES,
                 "adversary_calibration": ADVERSARY_CALIBRATION,
                 "conjunction_cell": CONJUNCTION_CELL,
                 "churn_prior": CHURN_PRIOR},
                out_dir / "run_manifest.json")
    return ctx


def run_claim1(ctx: dict, targets=VALUE_TARGETS, seeds=None, conditions=None,
               log=print) -> list:
    """Claim 1 in full: every condition x target x seed, resumable, capped."""
    cfg = ctx["cfg"]
    seeds = list(cfg.store_seeds if seeds is None else seeds)
    conds = list(CLAIM1_CONDITIONS if conditions is None else conditions)
    recs = []
    for target in targets:
        for seed in seeds:
            pools = build_live_sets(ctx["grid"], ctx["train_rows"], target, cfg, seed)
            assert_value_absent(ctx["grid"], pools["L_deleted"], target)
            _write_json({k: v for k, v in pools.items()
                         if not isinstance(v, np.ndarray)},
                        ctx["out_dir"] / "claim1" /
                        f"pools__{target.factor}_{target.value_index}__seed{seed}.json")
            for cond in conds:
                rec = run_claim1_cell(cond, ctx["paths"], ctx["F"], ctx["grid"],
                                      ctx["judges"], pools, target, cfg, seed,
                                      ctx["feat_shape"], ctx["out_dir"])
                recs.append(rec)
                log(f"  [{target.factor}={target.value_index} s{seed}] "
                    f"{rec['condition']:18s} best={rec['best_unanimous_hit']:.3f} "
                    f"({rec['best_move']})"
                    + ("  [resumed]" if rec.get("resumed") else
                       f"  {rec['seconds']:.0f}s"))
    return recs


def claim1_verdict(recs, margin: float = 0.10) -> dict:
    """The verdict, with BOTH calibration gates applied BEFORE it is read.

    ``sensitivity`` is the ceiling condition's best recovery.  ``specificity`` is the
    **worst case over the two negative controls** -- the dimension-matched random span
    and the charter C.2 landscape-removed store -- because either one recovering the
    value means the move is not reading store content.

    ⭐ This is not a formality.  Measured on seed 0: for ``object_hue = 0.5`` the two
    controls agree (0.000 and 0.016) and the cell is readable; for ``shape = cube``
    the random span returns **0.000** while the coordinate-shuffled store returns
    **0.344**, nearly the ceiling's 0.385.  Taking the random span alone would have
    declared an erasure result on a target where a store whose content has been
    destroyed recovers the value almost as well as one that holds it.  ⛔ A Gaussian
    matched only to the marginals is a WEAKER control than shuffled real atoms, and
    the weaker one must not be the one that licenses the claim.
    """
    by = {}
    for r in recs:
        by.setdefault((r["target"], r["seed"]), {})[r["condition"]] = r
    out = []
    for (tgt, seed), cells in sorted(by.items()):
        def best(name):
            v = cells.get(name, {}).get("best_unanimous_hit")
            return None if v is None else float(v)

        sens = best("clu_full")
        ctrls = {"random_span": best("random_span"),
                 "clu_shuffled": best("clu_shuffled")}
        have = [v for v in ctrls.values() if v is not None]
        spec = max(have) if have else None
        calibrated = (sens is not None and spec is not None
                      and (sens - spec) >= float(margin))
        row = {"target": tgt, "seed": seed,
               "sensitivity_clu_full": sens,
               "specificity_controls": ctrls,
               "specificity_worst_case": spec,
               "specificity_binding_control": (
                   None if spec is None else
                   max((k for k, v in ctrls.items() if v is not None),
                       key=lambda k: ctrls[k])),
               "separation": (None if sens is None or spec is None else sens - spec),
               "calibrated": bool(calibrated),
               "levels": {k: v.get("best_unanimous_hit") for k, v in cells.items()},
               "best_move": {k: v.get("best_move") for k, v in cells.items()}}
        if not calibrated:
            row["verdict"] = "ADVERSARY NOT CALIBRATED"
            row["why"] = (
                f"sensitivity {sens} does not clear the worst-case specificity "
                f"{spec} (binding control: {row['specificity_binding_control']}) by "
                f"the pre-registered margin {margin}. A store whose content has been "
                "destroyed recovers the value about as well as one that holds it, so "
                "this move is not reading store content on this target. "
                "No erasure claim is readable from this cell.")
        else:
            clu = best("clu_deleted")
            gated = [best(c) for c in ("basis_head_gated", "knn_proto_gated")]
            gated = [g for g in gated if g is not None]
            g = max(gated) if gated else None
            row["clu_deleted"] = clu
            row["best_gated_baseline"] = g
            row["gap_gated_minus_clu"] = (None if clu is None or g is None
                                          else g - clu)
            row["refit_baselines"] = {c: best(c) for c in
                                      ("basis_head_refit", "knn_proto_refit")}
            row["verdict"] = (
                "ERASED vs GATED"
                if (clu is not None and g is not None and g - clu >= margin)
                else "NOT SEPARATED")
        out.append(row)
    return {"margin": float(margin), "rows": out,
            "rule": ("a cell is read ONLY if sensitivity clears the WORST of the two "
                     "negative controls (random span, landscape-removed store) by the "
                     "margin; the verdict is ERASED vs GATED iff the best gated "
                     "baseline's recovery exceeds the refitted CLU's by the margin")}

# ---------------------------------------------------------------------------
# 11. Claim 2 and claim 3 runners
# ---------------------------------------------------------------------------


def run_claim2(ctx: dict, seeds=None, log=print) -> list:
    """Rebuild churn, per-request predictions PERSISTED, identical-live-set null."""
    import time as _time

    cfg = ctx["cfg"]
    grid, F, paths = ctx["grid"], ctx["F"], ctx["paths"]
    seeds = list(cfg.store_seeds if seeds is None else seeds)
    R = churn_requests(grid, int(cfg.churn_n_requests), 0)
    recs = []
    for seed in seeds:
        rng = np.random.default_rng(int(cfg.rng_base) + 5_003 * int(seed))
        universe = np.concatenate([ctx["train_rows"]])
        L = np.sort(rng.choice(universe, int(cfg.n_store_train), replace=False))
        drop = rng.choice(L.size, int(cfg.churn_delete_n), replace=False)
        keep = np.ones(L.size, bool)
        keep[drop] = False
        L_after = L[keep]
        for arm_name in ("clu", "basis_head", "knn_proto"):
            stem = f"claim2__{arm_name}__seed{seed}"
            path = ctx["out_dir"] / "claim2" / f"{stem}.json"
            npz = ctx["out_dir"] / "claim2" / f"{stem}__predictions.npz"
            if path.exists() and npz.exists():
                with open(path, encoding="utf-8") as fh:
                    rec = json.load(fh)
                rec["resumed"] = True
                recs.append(rec)
                log(f"  [churn s{seed}] {arm_name:11s} [resumed] "
                    f"churn={rec['churn_deleted']['judge_0']['tuple_changed']:.4f}")
                continue
            t0 = _time.time()

            def _fit(rows):
                if arm_name == "clu":
                    return fit_clu(paths, F, grid, rows, cfg, seed)
                if arm_name == "knn_proto":
                    return fit_knn_proto(F, grid, rows)
                return fit_basis_head(F, grid, rows, cfg, seed)

            P_before = predict_for_store(_fit(L), grid, ctx["judges"], R,
                                         ctx["feat_shape"], cfg)
            P_null = predict_for_store(_fit(L), grid, ctx["judges"], R,
                                       ctx["feat_shape"], cfg)
            P_del = predict_for_store(_fit(L_after), grid, ctx["judges"], R,
                                      ctx["feat_shape"], cfg)
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz, requests=R, before=P_before, null=P_null,
                                deleted=P_del)
            rec = {"arm": arm_name, "seed": int(seed),
                   "n_live": int(L.size), "n_deleted_rows": int(cfg.churn_delete_n),
                   "delete_frac": float(cfg.churn_delete_n / L.size),
                   "n_requests": int(R.shape[0]),
                   "churn_null": churn_between(P_before, P_null),
                   "churn_deleted": churn_between(P_before, P_del),
                   "predictions_file": npz.name,
                   "seconds": float(_time.time() - t0),
                   "null_is_exactly_zero": bool(
                       churn_between(P_before, P_null)["judge_0"]["tuple_changed"] == 0.0)}
            _write_json(rec, path)
            recs.append(rec)
            log(f"  [churn s{seed}] {arm_name:11s} "
                f"null={rec['churn_null']['judge_0']['tuple_changed']:.4f} "
                f"deleted={rec['churn_deleted']['judge_0']['tuple_changed']:.4f} "
                f"{rec['seconds']:.0f}s")
    return recs


def run_claim3(ctx: dict, seeds=None, log=print) -> list:
    """Conjunction deletion on the declared non-cylinder cell, with placebos."""
    import time as _time

    cfg = ctx["cfg"]
    grid, F, paths = ctx["grid"], ctx["F"], ctx["paths"]
    seeds = list(cfg.store_seeds if seeds is None else seeds)
    recs = []
    for seed in seeds:
        rng = np.random.default_rng(int(cfg.rng_base) + 5_003 * int(seed))
        L = np.sort(rng.choice(ctx["train_rows"], int(cfg.n_store_train),
                               replace=False))
        for conj in CONJUNCTIONS:
            stem = f"claim3__{conj.name}__seed{seed}"
            path = ctx["out_dir"] / "claim3" / f"{stem}.json"
            if path.exists():
                with open(path, encoding="utf-8") as fh:
                    rec = json.load(fh)
                rec["resumed"] = True
                recs.append(rec)
                log(f"  [conj s{seed}] {conj.name:20s} [resumed]")
                continue
            t0 = _time.time()
            pools = conjunction_pools(grid, L, conj, seed)
            rec = {"conjunction": conj.name, "seed": int(seed),
                   "cell": CONJUNCTION_CELL, "why": conj.why,
                   "n_deleted": pools["n_deleted"],
                   "size_matched_placebo": pools["size_matched"],
                   "gate_analysis": bitmap_cannot_express_a_conjunction(grid, L, conj),
                   "arms": {}}
            for arm_name in ("clu", "basis_head", "knn_proto"):
                def _fit(rows):
                    if arm_name == "clu":
                        return fit_clu(paths, F, grid, rows, cfg, seed)
                    if arm_name == "knn_proto":
                        return fit_knn_proto(F, grid, rows)
                    return fit_basis_head(F, grid, rows, cfg, seed)

                pre = conjunction_retained(_fit(L), grid, ctx["judges"], conj,
                                           ctx["feat_shape"], cfg, seed)
                post = conjunction_retained(_fit(pools["L_deleted"]), grid,
                                            ctx["judges"], conj, ctx["feat_shape"],
                                            cfg, seed)
                plac = conjunction_retained(_fit(pools["L_placebo"]), grid,
                                            ctx["judges"], conj, ctx["feat_shape"],
                                            cfg, seed)
                rec["arms"][arm_name] = {
                    "pre_deletion": pre, "post_deletion": post, "placebo": plac,
                    "drop_real": pre["retained_pair"] - post["retained_pair"],
                    "drop_placebo": pre["retained_pair"] - plac["retained_pair"],
                    "specific_drop": ((pre["retained_pair"] - post["retained_pair"])
                                      - (pre["retained_pair"] - plac["retained_pair"])),
                }
                log(f"  [conj s{seed}] {conj.name:20s} {arm_name:11s} "
                    f"pre={pre['retained_pair']:.3f} post={post['retained_pair']:.3f} "
                    f"placebo={plac['retained_pair']:.3f}")
            rec["seconds"] = float(_time.time() - t0)
            _write_json(rec, path)
            recs.append(rec)
    return recs


# ---------------------------------------------------------------------------
# 12. The seven-row scoreboard (w3-hub-common section 2)
# ---------------------------------------------------------------------------


def _nearest_value_index(values: np.ndarray, v: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, np.float64) - float(v))))


def load_query_set(paths: Mapping, name: str) -> list:
    with open(Path(paths["value_holdout"]) / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)["records"]


def records_to_indices(records, grid) -> tuple:
    """``records -> ((n, F) nearest-grid value indices, (n, F) exactly-on-grid mask)``.

    A duplicate term keeps the LAST one, which is what an interface that reads terms
    in order does; the duplicate itself is what makes the record invalid, and the
    scoreboard scores that through the refusal row, not through the encoding.
    """
    n, F = len(records), len(grid.cardinalities)
    V = np.full((n, F), -1, np.int64)
    exact = np.zeros((n, F), bool)
    for i, rec in enumerate(records):
        for term in rec["request"]:
            f = int(term["factor_index"])
            vals = np.asarray(grid.values[f], np.float64)
            V[i, f] = _nearest_value_index(vals, float(term["value"]))
            exact[i, f] = bool(np.any(vals == float(term["value"])))
    return V, exact


def judge_satisfies(preds, V: np.ndarray) -> np.ndarray:
    """Unanimous over three judges: every REQUESTED factor read back as requested."""
    ok = np.ones(V.shape[0], bool)
    asked = V >= 0
    for p in preds:
        P = np.asarray(p)
        ok &= np.all((P == V) | ~asked, axis=1)
    return ok


def arm_outputs(arm: dict, grid, V: np.ndarray, cfg: ErasureConfig) -> np.ndarray:
    if arm["arm"] == "clu":
        return clu_read(arm, V, cfg)
    if arm["arm"] == "knn_proto":
        return knn_proto_read(arm, V, grid)
    return basis_head_read(arm, V)


def cell_requests(cell: Mapping, grid, n: int, seed: int) -> np.ndarray:
    """``n`` requests inside one held-out cell: the cell's values fixed, the rest free."""
    rng = np.random.default_rng(20_260_919 + 7 * int(seed) + abs(hash(cell["cell_id"])) % 9_973)
    V = np.stack([rng.integers(0, int(c), int(n)) for c in grid.cardinalities], 1)
    for fname, vals in cell["values"].items():
        f = list(grid.names).index(fname)
        gv = np.asarray(grid.values[f], np.float64)
        choices = [_nearest_value_index(gv, float(x)) for x in vals]
        V[:, f] = rng.choice(choices, int(n))
    return V


def score_arm_rows(arm: dict, grid, judges: dict, paths, cfg: ErasureConfig,
                   feat_shape, seen_tuples: set, gate: Optional[ValueBitmap],
                   seed: int, n_per_cell: int = 32) -> dict:
    """Every one of the seven rows for one arm, as ``make_outcome`` objects.

    ``accept`` is the gate's decision (all-accept for an ungated arm); ``valid`` is
    the three-judge unanimous read-back of the arm's own output.  Rows 3 and 4 draw
    requests inside the pinned held-out cells.
    """
    from .gen_scoreboard import make_outcome

    led = dict(arm["byte_ledger"])
    led["deployed_bytes"] = int(arm["deployed_bytes"]) + (GATE_BYTES if gate else 0)
    led["budget_bytes"] = int(cfg.byte_budget)
    if led["deployed_bytes"] != int(cfg.byte_budget):
        led["budget_note"] = (
            f"deployed {led['deployed_bytes']:,} B against the {cfg.byte_budget:,} B "
            "budget; DECLARED, not silent (R26.3)")
    setting = {"arm": arm["arm"], "launch_mode": cfg.launch_mode,
               "read_mode": cfg.read_mode, "n_atoms": cfg.n_atoms,
               "basis_m": cfg.basis_m, "gated": bool(gate is not None)}
    threshold = 0.5

    with open(Path(paths["value_holdout"]) / "value_cells.json", encoding="utf-8") as fh:
        cells = json.load(fh)["cells"]
    pair_cells = [c for c in cells if c["meta"]["family"] == "pair"]
    nway_cells = [c for c in cells if c["meta"]["family"] == "nway"]

    per_factor_detail = {}

    def _outcome(V, ids=None, row_id=None):
        served = gate.accepts(V) if gate is not None else np.ones(V.shape[0], bool)
        X = arm_outputs(arm, grid, V, cfg)
        jf = judge_features(judges, X, feat_shape)
        good = judge_satisfies(jf["preds"], V) & served
        tuples = np.asarray(jf["preds"][0], np.int64).copy()
        tuples[~served] = -1
        if row_id is not None:
            # ⭐ `valid` is UNANIMOUS-over-3-judges on EVERY requested factor at once,
            # which is far stricter than the programme's usual "unweighted mean over
            # factors". The strict rate is what the row scores; the per-factor mean is
            # recorded beside it so the row is readable and so nobody compares this
            # hub's row 1 with another hub's without noticing the difference.
            asked = V >= 0
            acc = []
            for pr in jf["preds"]:
                P = np.asarray(pr)
                hit = ((P == V) & asked).sum(axis=0) / np.maximum(asked.sum(axis=0), 1)
                acc.append(hit)
            per_factor_detail[row_id] = {
                "criterion": ("valid = unanimous over 3 judges on EVERY requested "
                              "factor at once (strict exact match)"),
                "per_factor_accuracy_judge0": [float(x) for x in acc[0]],
                "mean_over_factors_judge0": float(np.mean(acc[0])),
                "mean_over_factors_all_judges": float(np.mean(acc)),
                "strict_exact_match": float(good.mean()),
            }
        return make_outcome(served, good, tuples, threshold, setting, led, ids)

    rows = {}
    for row_id, qs in (("r1_seen", "row1_seen"), ("r2_unseen_valid", "row2_unseen"),
                       ("r5_comp_refusal", "row5_held_out"),
                       ("r6_invalid_refusal", "c_invalid")):
        recs = load_query_set(paths, qs)
        if int(cfg.sb_max_records):
            recs = recs[: int(cfg.sb_max_records)]
        V, _ = records_to_indices(recs, grid)
        rows[row_id] = _outcome(V, [r["id"] for r in recs], row_id)
    for row_id, cs in (("r3_comp_2way", pair_cells), ("r4_comp_nway", nway_cells)):
        Vs = [cell_requests(c, grid, int(n_per_cell), seed) for c in cs]
        rows[row_id] = _outcome(np.concatenate(Vs), None, row_id)
    rows["_per_factor_detail"] = per_factor_detail
    return rows


def deletion_row_inputs(arm_pre: dict, arm_post: dict, grid, judges, paths, cfg,
                        feat_shape, seen_tuples, seed) -> dict:
    """Row 7's three pieces: survivors, the deleted value, and the PRE-deletion refusal.

    ``fit_fingerprint`` is ``(pre, post)`` and row 7 requires them EQUAL -- "no
    retraining".  This hub's CLU deletion **is** a refit, so the fingerprints differ
    by construction and row 7 records that fault rather than hiding it.  That is the
    honest reading and it is the whole point of this hub: **the CLU buys erasure with
    a rebuild, and row 7 as specified scores the no-retraining variant, which is the
    GATE's move.**  The scoreboard is scored UNGATED for every arm, so the deployed
    ledger is constant across the seven rows; the gate belongs to claim 1.
    """
    pre_rows = score_arm_rows(arm_pre, grid, judges, paths, cfg, feat_shape,
                              seen_tuples, None, seed)
    post_rows = score_arm_rows(arm_post, grid, judges, paths, cfg, feat_shape,
                               seen_tuples, None, seed)
    pre_rows.pop("_per_factor_detail", None)
    post_rows.pop("_per_factor_detail", None)
    return {"survivor": post_rows["r4_comp_nway"],
            "deleted": post_rows["r5_comp_refusal"],
            "pre": pre_rows["r5_comp_refusal"],
            "fit_fingerprint": (arm_pre.get("n_fit_rows"), arm_post.get("n_fit_rows"))}


def bitmap_arm_rows(grid, paths, cfg, seed: int) -> dict:
    """The target file section 1 object as a SCORED ARM: the 8-byte value bitmap.

    It generates nothing, so its ``valid`` is False everywhere and rows 1-4 score 0
    by construction.  Its refusal pair is what the section-1 negative is about, and
    printing it beside the store's is mandatory (target file section 6).
    """
    from .gen_scoreboard import make_outcome

    rng = np.random.default_rng(int(cfg.rng_base) + 5_003 * int(seed))
    L = np.sort(rng.choice(np.arange(480_000), int(cfg.n_store_train), replace=False))
    bm = ValueBitmap.from_pool(grid, L)
    # ⛔ EVERY held-out value, not just the hue. Row 5 carries both a held_out_hue
    # and a held_out_shape half; clearing one bit refuses only half the row and
    # under-reports the section-1 rule, which is the rule this hub must NOT flatter
    # itself against.
    for _t in VALUE_TARGETS:
        bm = bm.clear(_t.factor_index, _t.value_index)
    led = {"deployed_bytes": bm.n_bytes(), "budget_bytes": int(cfg.byte_budget),
           "what": "the 8-byte value-presence bitmap: one bit per (factor, grid value)",
           "budget_note": ("8 B against a 524,288-byte budget -- 0.0015 % of it. It "
                           "HOLDS nothing and GENERATES nothing, which is why its "
                           "rows 1-4 are 0 and its refusal rows are not evidence "
                           "for any store")}
    setting = {"arm": "value_bitmap_8B", "bits": int(bm.present.size)}
    rows = {}
    for row_id, qs in (("r1_seen", "row1_seen"), ("r2_unseen_valid", "row2_unseen"),
                       ("r5_comp_refusal", "row5_held_out"),
                       ("r6_invalid_refusal", "c_invalid")):
        recs = load_query_set(paths, qs)
        if int(cfg.sb_max_records):
            recs = recs[: int(cfg.sb_max_records)]
        V, exact = records_to_indices(recs, grid)
        # an OFF-grid term is not in the bitmap's vocabulary at all; the rule accepts
        # it, which is what makes row 2 free for it and is reported, not hidden.
        served = bm.accepts(np.where(exact, V, -1))
        rows[row_id] = make_outcome(served, np.zeros(V.shape[0], bool), None,
                                    0.5, setting, led, [r["id"] for r in recs])
    return rows


def run_scoreboard(ctx: dict, seed: int = 0, log=print) -> dict:
    """The seven-row table: `clu` primary, `basis_head` and `knn_proto` on every row.

    The laundering control (`grid_syntax_rule`) is the pinned no-model rule and is
    mandatory on the refusal rows.
    """
    import time as _time

    from .gen_scoreboard import (ScoreboardConfig, build_scoreboard,
                                 grid_syntax_outcome, write_scoreboard)

    cfg = ctx["cfg"]
    grid, F, paths = ctx["grid"], ctx["F"], ctx["paths"]
    out = ctx["out_dir"] / "scoreboard" / f"scoreboard__seed{seed}.json"
    if out.exists():
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)
    t0 = _time.time()
    target = TARGET_HUE
    pools = build_live_sets(grid, ctx["train_rows"], target, cfg, seed)
    L, Ld = pools["L_full"], pools["L_deleted"]
    fi = np.asarray(grid.factor_indices_of(L), np.int64)
    seen = {tuple(int(x) for x in r) for r in fi}
    arms_pre = {"clu": fit_clu(paths, F, grid, L, cfg, seed),
                "basis_head": fit_basis_head(F, grid, L, cfg, seed),
                "knn_proto": fit_knn_proto(F, grid, L)}
    arms_post = {"clu": fit_clu(paths, F, grid, Ld, cfg, seed),
                 "basis_head": fit_basis_head(F, grid, Ld, cfg, seed),
                 "knn_proto": fit_knn_proto(F, grid, Ld)}
    outcomes, deletion, detail = {}, {}, {}
    for name, arm in arms_pre.items():
        outcomes[name] = score_arm_rows(arm, grid, ctx["judges"], paths, cfg,
                                        ctx["feat_shape"], seen, None, seed)
        detail[name] = outcomes[name].pop("_per_factor_detail", {})
        log(f"  [scoreboard s{seed}] {name} rows scored  {_time.time()-t0:.0f}s")
        deletion[name] = deletion_row_inputs(arm, arms_post[name], grid,
                                             ctx["judges"], paths, cfg,
                                             ctx["feat_shape"], seen, seed)
        # build_scoreboard requires the KEY for all seven rows; the row-7 SCORE comes
        # from `deletion`. The representative outcome carries the same ledger as the
        # other six, so the per-arm ledger stays constant.
        outcomes[name]["r7_deletion"] = deletion[name]["deleted"]
        log(f"  [scoreboard s{seed}] {name} row 7 scored  {_time.time()-t0:.0f}s")
    # the mandatory no-model laundering control, on the pinned query sets
    sb_cfg = ScoreboardConfig(byte_budget=int(cfg.byte_budget), byte_match_tolerance=0)
    lr = {}
    for row_id, qs in (("r1_seen", "row1_seen"), ("r2_unseen_valid", "row2_unseen"),
                       ("r5_comp_refusal", "row5_held_out"),
                       ("r6_invalid_refusal", "c_invalid")):
        _r = load_query_set(paths, qs)
        if int(cfg.sb_max_records):
            _r = _r[: int(cfg.sb_max_records)]
        lr[row_id] = grid_syntax_outcome(_r, grid, sb_cfg)
    for row_id in ("r3_comp_2way", "r4_comp_nway", "r7_deletion"):
        lr[row_id] = grid_syntax_outcome([], grid, sb_cfg)
    outcomes[LAUNDERING_ARM_NAME] = lr
    outcomes["value_bitmap_8B"] = bitmap_arm_rows(grid, paths, cfg, seed)
    table = build_scoreboard("clu", outcomes, sorted(seen), sb_cfg, deletion,
                             provenance={"seed": int(seed),
                                         "bridge_seed": int(cfg.bridge_seed),
                                         "decoder_hash": ctx["judges"]["decoder_hash"],
                                         "pools": pools["sizes"],
                                         "per_factor_detail": detail,
                                         "hub": "hub/erasure"})
    out.parent.mkdir(parents=True, exist_ok=True)
    write_scoreboard(table, out)
    log(f"  [scoreboard s{seed}] written {out.name}  {_time.time()-t0:.0f}s")
    return table


LAUNDERING_ARM_NAME = "grid_syntax_rule"
