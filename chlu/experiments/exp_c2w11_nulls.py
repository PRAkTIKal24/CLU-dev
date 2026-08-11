"""⭐⭐ **C2W11 spoke C — the ORGANIZER SWAP's NULL SIDE**, at F3-grade tuning.

> **A hobbled null is the same referee attack in mirror image.**

This module produces the ``null*`` side of ``PREREG-C2W11.md`` §5's three VALUE
legs — **V1** (generalization), **V2a/V2b** (the graded-novelty confidence
channel) and **V3** (the anytime curve) — for the five registered non-physics
organizers N1–N5, on the substrate spoke A froze in
``.claude/outputs/c2w11/FROZEN-INTERFACES-C2W11.json``.

⛔ **THIS SPOKE IS THE CONTROL AND COMPUTES NO VERDICT.** No ``OD``, no
``OD_min``, no swap verdict, no tier-ii verdict, no paper number. It emits
``null*`` and the arm-side tables that a verdict would be computed *from*.

## What is new here relative to C2W5's ``exp_null_arms`` (which this inherits)

1. **The launch is feature-factored.** Every arm consumes
   :class:`~chlu.core.feature_launch.FeatureLaunchHead`'s launch points, produced
   by :func:`~chlu.core.null_arms.feature_launch_states`, which reproduces
   ``multi_particle_read``'s launch stage — *including its key-chunking* — so the
   arms and the physics arm are on **bit-identical** points (pytest-asserted).
2. ⭐ **A matched confidence channel on every arm** (VALUE leg ii's null side).
   N1 = the winning atom's read-objective weight · N2 = distance-to-codebook ·
   N3 = the fitted rule's margin · N4 = neighbour distance · N5 = the surprise
   gate's read-time form. ⛔ An arm with no principled channel would be a
   declared NOT-RUN for V2; all five have one, and N5's derivation is stated
   where it can be checked.
3. ⭐⭐ **VALUE leg iii's null side is NOT a flat line.** The swap hands the null
   arm the **same reader class and the same read**, so N1/N2/N3's organization is
   **instantiated as a landscape** (:func:`~chlu.core.null_arms.instantiate_landscape`,
   which calls the physics arm's own placing write) and read with the identical
   k-particle anytime read at the **frozen** budget grid. N4/N5 admit no
   landscape and are **declared NOT-RUN for V3**, reported as flat reference
   lines — ⛔ never scored as "un-navigable".
4. ⛔ **The decodability ceiling is recomputed on the new launches**
   (:func:`~chlu.core.null_arms.feature_decodability_ceiling`) and belongs in the
   same sentence as any "no arm clears" statement.

## ⛔ The payload-repair trap

Spoke A moved ``payload_radius`` 1.0 → **0.60** to close a measured reach gap.
``tol``, ``chance`` and every y-scale are homogeneous of degree 1 in it and moved
with it (``tol = 0.28696``, **not** ``run1``'s pre-repair 0.47827). ⛔ Nothing
here hard-codes them: :func:`load_frozen` reads them from the frozen artifact and
:func:`assert_frozen_match` **refuses to score** a cell whose measured ``tol``
disagrees with the frozen one.

⛔ Wells are never named semantically (``PREREG-TierII.md`` §2.6): *"Wells {j} are
co-activated by queries whose ground-truth factor set contains factor f, with
co-activation correlation rho = … (95 % CI …), measured against a permutation
null. No well is identified with any factor; the claim is a correlation between
co-activation/wormhole/shell-position statistics and task structure."*
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import numpy as np

from chlu.core.factored_store import (
    CatTestConfig,
    build_family,
    build_phi,
    chance_accuracy,
    exact_set_accuracy,
    multi_particle_read,
    occupancy,
    place_wells,
)
from chlu.core.feature_launch import build_launch_head
from chlu.core.null_arms import (
    ARMS,
    NullArmGrid,
    anytime_read,
    arm_ledger,
    expected_calibration_error,
    feature_decodability_ceiling,
    feature_keys,
    feature_launch_states,
    fit_readers_plus_identity,
    instantiate_landscape,
    n1_confidence,
    n1_gradient_placed,
    n2_confidence,
    n2_vq,
    n3_confidence,
    n3_static_geometric,
    n4_confidence,
    n4_knn,
    n5_confidence,
    n5_titans,
    novelty_auroc,
    read_flops,
    score_readers_plus_identity,
    shuffle_launches,
)
from chlu.experiments.exp_null_arms import (
    _dump,
    _grid_configs,
    _native,
    _rule4_val_split,
    _z_native,
)

__all__ = [
    "run_c2w11_nulls",
    "C2W11_GRID",
    "load_frozen",
    "c2w11_null_config",
    "seed_setup",
    "assert_frozen_match",
    "build_novelty_family",
    "stage_guards",
    "stage_grid",
    "stage_score",
    "stage_gridmax",
    "stage_v2",
    "stage_v3",
    "stage_ceiling",
    "stage_oracle",
    "ALL_STAGES",
]

ALL_STAGES = ("guards", "grid", "score", "gridmax", "v2", "v3", "ceiling",
              "oracle")

#: ⛔ **R1 (declared rider).** ``FROZEN-INTERFACES-C2W11.json`` freezes launch
#: keys for k0/k6_k7cap/k3_k4_k5/m6/coverage and registers **none** for the VALUE
#: legs. This spoke adopts the ``k3_k4_k5`` key — the stage that fits the reader
#: class and scores unseen exact-set accuracy, i.e. the exact operation V1 is —
#: with SEEN on the key itself and ``Q_unseen`` on ``fold_in(key, 1)``, which is
#: ``stage_k3_k4_k5``'s own pattern. ⚠ Spoke B must use the same key or V1 and V3
#: are scored on different launches; the launch-point hash is emitted so it is
#: byte-checkable.
VALUE_LAUNCH_KEY = 7000

#: The registered F3 grid. ⚠ ``n4_keys`` gains ``launch_flat``: a mean over
#: channels destroys exactly the per-channel structure the C2W11 launch was built
#: to create, so scoring the nulls on the mean alone would hobble them.
C2W11_GRID = NullArmGrid(
    n4_keys=("set_code", "launch_mean", "launch_flat"),
)

#: N5's key space, fixed (not an axis) to the **richest arm-visible** statistic.
#: ⛔ ``set_code`` is the *noiseless-key* variant and is N4's declared axis only.
N5_KEY = "launch_flat"

#: V2's novel/known split — ⛔ **R2 (declared rider):** not frozen by spoke A.
N_NOVEL_WELLS = 4
NOVELTY_SEED_BASE = 20260811

#: V3's instantiation sweep (the mirror-image referee attack, closed by
#: measurement): 5 widths x 3 depths x 2 atom budgets = 30 configs per arm.
V3_WIDTH_FRACS = (0.20, 0.25, 0.37, 0.50, 0.75)
V3_DEPTHS = (0.15, 0.30, 0.60)
V3_ATOM_BUDGETS = (384, 768)
#: the subset the ``null*_V3`` grid-max runs over at EVERY budget point (the full
#: 30 would be 2 700 settles; the 6 kept are the selected config plus the width
#: axis at the selected depth/budget, which is the axis the attack names).
V3_GRIDMAX_WIDTHS = V3_WIDTH_FRACS


# ==========================================================================
# the frozen interfaces — read, never re-derived
# ==========================================================================
_FROZEN_REL = Path(".claude/outputs/c2w11/FROZEN-INTERFACES-C2W11.json")


def _frozen_default_path() -> Path:
    """Locate spoke A's frozen artifact.

    ⚠ **A worktree does not carry `.claude/`** (it is gitignored and lives in the
    main checkout only), so searching the *module's* parents silently finds
    nothing when this spoke runs from `../CHLU-<slug>` — which would degrade a
    claim cell to "frozen not loaded" without anyone noticing. The search
    therefore covers the environment variable, the module's parents **and the
    cwd's parents**, and the resolved path is emitted into every artifact.
    """
    env = os.environ.get("CHLU_C2W11_FROZEN")
    if env:
        return Path(env)
    for root in (Path(__file__).resolve(), Path.cwd().resolve()):
        for p in (root, *root.parents):
            cand = p / _FROZEN_REL
            if cand.exists():
                return cand
    return _FROZEN_REL


def load_frozen(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Spoke A's frozen family / launches / phi / reader class / byte ledger.

    ⛔ Returns ``None`` when the artifact is absent (a test fixture, a fresh
    clone) so the module stays importable; every cell then records
    ``frozen_verified = False`` and the harness says so loudly. It never
    substitutes a remembered constant for the file.
    """
    p = Path(path) if path else _frozen_default_path()
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["_resolved_path"] = str(p)
    return d


def c2w11_null_config(frozen: Optional[Dict[str, Any]] = None, **kw
                      ) -> CatTestConfig:
    """The **claim cell**: spoke A's repaired substrate at its selected operating point.

    ⛔ Every value here is spoke A's, read from the frozen artifact when it is
    present: the re-selected co-scaled width (repair (b) + the refuse-at-
    unselected-width guard (d)) and the **repaired payload radius** with its
    co-scaled ``atom_payload_init_radius``.
    """
    w = 0.37
    r = 0.6
    if frozen:
        w = float(frozen["selected_atom_width"]["atom_width_frac_spacing"])
        r = float(frozen["family"]["payload_radius"])
    base = dict(
        write_mode="placing",
        launch_mode="feature_factored",
        atoms_per_well=12,
        payload_dim=8,
        payload_radius=r,
        atom_payload_init_radius=r,
        atom_width_frac_spacing=w,
        atom_width_selected_frac=w,
    )
    base.update(kw)
    cfg = CatTestConfig(**base)
    if cfg.launch_mode == "feature_factored":
        k = int(cfg.n_channels) if cfg.n_channels is not None else int(cfg.f_subset)
        cfg = replace(cfg, n_particles=k)
    return cfg


class FrozenCellMismatch(RuntimeError):
    """⛔ Raised when a cell's measured family disagrees with the frozen one.

    The payload-repair trap in mechanical form: ``run1``'s artifacts carry the
    **pre-repair** ``tol = 0.47827`` while the repaired cell is ``0.28696``, and
    ``tol`` is what every score in this module is thresholded at. A cell that
    silently ran at the wrong ``tol`` would produce a plausible, wrong table.
    """


def assert_frozen_match(cfg: CatTestConfig, fam, seed: int,
                        frozen: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare the measured cell against the frozen artifact and REFUSE on drift."""
    info: Dict[str, Any] = {"frozen_verified": bool(frozen),
                            "tol_measured": float(fam.tol), "seed": int(seed)}
    if not frozen:
        info["note"] = ("⛔ FROZEN-INTERFACES-C2W11.json not found: the cell ran "
                        "on its own construction and NOTHING here is verified "
                        "against spoke A.")
        return info
    f = frozen["family"]
    info["tol_frozen"] = float(f["tol"])
    info["tol_ratio_to_frozen"] = float(fam.tol) / float(f["tol"])
    # ⭐ THE PAYLOAD-REPAIR TRAP, in its correct mechanical form.
    # `tol = tol_frac * RMS||y - ybar||` is a PER-SEED sample statistic, so it is
    # exact only on the seed spoke A froze (seed 0, verified: 0.286960063782279)
    # and varies by ~1-2 % on the others. The thing that must NOT drift is the
    # SCALE: the pre-repair tol is 0.47827, i.e. a ratio of 1.667, so a 10 % band
    # separates "another seed of the repaired family" from "a pre-repair
    # constant leaked in" by a factor of six.
    info["payload_radius_frozen"] = float(f["payload_radius"])
    if abs(float(cfg.payload_radius) - float(f["payload_radius"])) > 1e-12 or \
            abs(float(cfg.atom_payload_init_radius)
                - float(f["atom_payload_init_radius"])) > 1e-12:
        raise FrozenCellMismatch(
            f"payload radius drift: cell {cfg.payload_radius} / "
            f"{cfg.atom_payload_init_radius} vs frozen {f['payload_radius']} / "
            f"{f['atom_payload_init_radius']}. ⛔ The reach repair moved it "
            "1.0 -> 0.60 and tol, chance and every y-scale moved with it.")
    if int(seed) == 0 and abs(float(fam.tol) - float(f["tol"])) > 1e-6:
        raise FrozenCellMismatch(
            f"tol drift at the FROZEN seed: measured {fam.tol!r} vs frozen "
            f"{f['tol']!r}. ⛔ A pre-repair constant has leaked into this cell.")
    if not (0.90 <= info["tol_ratio_to_frozen"] <= 1.10):
        raise FrozenCellMismatch(
            f"tol scale drift at seed {seed}: measured {fam.tol!r} is "
            f"{info['tol_ratio_to_frozen']:.3f}x the frozen {f['tol']!r} "
            "(pre-repair would be 1.667x). ⛔ A pre-repair constant has leaked "
            "into this cell.")
    for k in ("n_wells", "f_subset", "n_items", "payload_dim", "addr_dim",
              "atoms_per_well", "n_unseen"):
        want = {"n_wells": "N_a", "f_subset": "F", "n_items": "K",
                "payload_dim": "m", "addr_dim": "d_addr",
                "atoms_per_well": "a", "n_unseen": "n_unseen_sampled"}[k]
        got, exp = int(getattr(cfg, k)), int(f[want])
        info[f"{k}_ok"] = bool(got == exp)
        if got != exp:
            raise FrozenCellMismatch(f"{k}: cell {got} vs frozen {exp}")
    if seed < len(f["chance_per_seed"]):
        info["chance_frozen"] = float(f["chance_per_seed"][seed])
    return info


def _phi_bytes_hash(phi) -> Tuple[int, str]:
    """φ's byte count and **spoke A's own hash of the same bytes**.

    ⛔ Deliberately imports ``exp_c2w11_substrate._phi_hash`` instead of hashing
    here: a byte-comparison against a frozen digest is only a comparison if both
    sides use the same digest. (Measured the hard way — an md5 of identical bytes
    reported ``match = False`` against spoke A's sha256 and looked exactly like a
    φ mismatch.)
    """
    from chlu.experiments.exp_c2w11_substrate import _phi_hash

    return int(phi.n_bytes()), _phi_hash(phi)


# ==========================================================================
# the per-seed setup — the ONLY place Q_unseen is constructed
# ==========================================================================
def seed_setup(cfg: CatTestConfig, seed: int, n_val: int = 32,
               frozen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything an arm may see, and nothing else.

    ⛔ **THE SELECTION GUARD, mechanically.** ``Q_unseen`` (``ind_u``/``q0_u``/
    ``y_unseen``) is constructed **here and nowhere else**, and is read in exactly
    three functions: :func:`stage_score`, :func:`stage_gridmax` and the declared
    out-of-class diagnostics (:func:`stage_ceiling`, :func:`stage_oracle`). No
    fit, no hyperparameter and no arm ever sees it: every ``_fit_arm`` call in
    this module is passed ``S["tr"]`` (grid) or all of SEEN (score), and the
    selection statistic is the arm's own read on ``S["va"]`` — a **rule-4-valid**
    slice carved out of SEEN, so selection runs on the same *problem* that is
    scored and not on an easier near-neighbour one.
    """
    fam = build_family(cfg, seed=int(seed))
    phi = build_phi(cfg)
    head = build_launch_head(phi, cfg)
    ruler = cfg.s_measured if cfg.s_measured is not None else cfg.atom_width
    anchors = place_wells(phi, cfg, sep=float(cfg.target_ds * ruler))
    ind_s = fam.indicator(fam.seen, cfg.n_wells)
    ind_u = fam.indicator(fam.unseen, cfg.n_wells)
    k = jax.random.PRNGKey(VALUE_LAUNCH_KEY + int(seed))
    q0_s = feature_launch_states(head, cfg, ind_s, k)
    q0_u = feature_launch_states(head, cfg, ind_u, jax.random.fold_in(k, 1))
    va, tr, k2val = _rule4_val_split(fam, int(cfg.f_subset), int(n_val), int(seed))
    return {"family": fam, "phi": phi, "head": head, "anchors": anchors,
            "ind_s": ind_s, "ind_u": ind_u, "q0_s": q0_s, "q0_u": q0_u,
            "tr": tr, "va": va, "key": k, "seed": int(seed),
            "chance": chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol),
            "val_rule4": k2val,
            "frozen_check": assert_frozen_match(cfg, fam, int(seed), frozen)}


# ==========================================================================
# fitting one configuration of one arm — SEEN only
# ==========================================================================
def _fit_arm(arm: str, conf: Dict[str, Any], S: Dict[str, Any],
             cfg: CatTestConfig, g: NullArmGrid, *, idx: np.ndarray,
             q0_override: Optional[np.ndarray] = None,
             family=None) -> Dict[str, Any]:
    """Fit ``arm`` at ``conf`` on the SEEN rows ``idx``.

    ⚠ Deliberately NOT ``exp_null_arms._fit_arm``: that one routes N4/N5 through
    ``n4_keys``, which de-offsets by ``phi.offsets`` — a designed offset the
    feature-factored launch does not have. Everything else is the same call into
    :mod:`chlu.core.null_arms`, so N1/N2/N3 are the C2W5 arms unchanged.
    """
    fam = S["family"] if family is None else family
    anchors, head = S["anchors"], S["head"]
    q0_all = S["q0_s"] if q0_override is None else q0_override
    q0_tr, y_tr = q0_all[idx], fam.y_seen[idx]
    seed = S["seed"]

    if arm == "N1":
        fit = n1_gradient_placed(cfg, fam, anchors, q0_tr, y_tr, lr=conf["lr"],
                                 tau=conf["tau"], init=conf["init"],
                                 steps=g.steps, seed=seed,
                                 atoms_per_well=conf["atoms_per_well"])
        from chlu.core.null_arms import n1_apply
        hard = conf["read"] == "hard"
        return {"z": lambda q0, ind=None: n1_apply(fit, q0, hard=hard),
                "conf": lambda q0, ind=None: n1_confidence(fit, q0),
                "codebook": fit["codebook"], "ledger": fit["ledger"], "fit": fit,
                "train": {"loss_first": fit["loss_first"],
                          "loss_last": fit["loss_last"]}}
    if arm == "N2":
        fit = n2_vq(cfg, fam, q0_tr, y_tr, variant=conf["variant"],
                    n_codes=conf["n_codes"], commitment=conf["commitment"],
                    lr=max(conf["lr"], 1e-6), steps=g.steps, seed=seed,
                    restarts=g.n2_restarts, payload_source=conf["payload_source"],
                    anchors=anchors)
        return {"z": lambda q0, ind=None: fit["apply"](q0),
                "conf": lambda q0, ind=None: n2_confidence(fit, q0),
                "codebook": fit["codebook"], "ledger": fit["ledger"],
                "assign": fit["assign"], "fit": fit,
                "train": {k: fit[k] for k in ("loss_first", "loss_last")
                          if k in fit}}
    if arm == "N3":
        fit = n3_static_geometric(cfg, fam, anchors, q0_tr, y_tr,
                                  level=conf["level"], lr=conf["lr"],
                                  tau=conf["tau"], steps=g.steps, seed=seed,
                                  payload_source=conf["payload_source"])
        return {"z": lambda q0, ind=None: fit["apply"](q0),
                "conf": lambda q0, ind=None: n3_confidence(fit, q0, "evidence"),
                # the task's registered form, reported beside the sound one
                "conf_variant": lambda q0, ind=None: n3_confidence(fit, q0,
                                                                   "margin"),
                "codebook": fit["codebook"], "ledger": fit["ledger"],
                "assign": fit["assign"], "fit": fit,
                "train": {"loss_first": fit["loss_first"],
                          "loss_last": fit["loss_last"]}}
    if arm == "N4":
        keys_tr = feature_keys(conf["key"], head, cfg, S["ind_s"][idx], q0_all[idx])

        def predict(q0, ind):
            kq = feature_keys(conf["key"], head, cfg, ind, q0)
            return n4_knn(cfg, keys_tr, y_tr, kq, k=conf["k"],
                          weight=conf["weight"])

        def conf_fn(q0, ind):
            # per-CHANNEL neighbour distance: the key space restricted to one
            # channel, so the arm emits a per-feature quantity like every other.
            out = []
            for c in range(int(cfg.n_particles)):
                kq = np.asarray(q0)[:, c, : cfg.addr_dim]
                kt = keys_tr if keys_tr.shape[1] == cfg.addr_dim else \
                    np.asarray(q0_all[idx])[:, c, : cfg.addr_dim]
                out.append(n4_confidence(kt, kq))
            return np.stack(out, 1)

        return {"predict": predict, "conf": conf_fn,
                "ledger": arm_ledger("N4", cfg, n_params=0,
                                     n_state=int(len(idx) * (cfg.addr_dim
                                                             + cfg.payload_dim)),
                                     k=conf["k"], weight=conf["weight"],
                                     key_space=conf["key"],
                                     noiseless_key=bool(conf["key"] == "set_code"),
                                     read_flops=read_flops("N4", cfg)),
                "train": {}}
    if arm == "N5":
        keys_tr = feature_keys(N5_KEY, head, cfg, S["ind_s"][idx], q0_all[idx])
        rank = np.empty(len(fam.seen), dtype=int)
        rank[np.asarray(fam.order)] = np.arange(len(fam.seen))
        order = np.argsort(rank[idx])
        fit = n5_titans(cfg, keys_tr, y_tr, hidden=conf["hidden"], lr=conf["lr"],
                        momentum=conf["momentum"], decay=conf["decay"],
                        gate=conf["gate"], chunk=conf["chunk"], passes=g.n5_passes,
                        pretrain_steps=g.n5_pretrain_steps, order=order, seed=seed)

        def predict(q0, ind):
            return fit["apply"](feature_keys(N5_KEY, head, cfg, ind, q0))

        def conf_fn(q0, ind):
            # the read-time surprise, per channel: the key space with only that
            # channel's launch point live (the others zeroed), so the deviation
            # M - M_0 is evaluated where that feature landed.
            z = np.asarray(q0)[..., : cfg.addr_dim]
            out = []
            for c in range(int(cfg.n_particles)):
                kk = np.zeros_like(z)
                kk[:, c] = z[:, c]
                out.append(n5_confidence(fit, kk.reshape(len(z), -1)))
            return np.stack(out, 1)

        return {"predict": predict, "conf": conf_fn, "ledger": fit["ledger"],
                "fit": fit,
                "train": {k: fit[k] for k in ("pre_loss_first", "pre_loss_last",
                                              "stream_loss_first",
                                              "stream_loss_last")}}
    raise ValueError(arm)


def _arm_predict(fitted: Dict[str, Any], q0: np.ndarray, ind: np.ndarray,
                 cfg: CatTestConfig) -> np.ndarray:
    if "predict" in fitted:
        return fitted["predict"](q0, ind)
    return _z_native(fitted["z"](q0, ind), cfg)


def _mean_2se(xs) -> Dict[str, Any]:
    a = np.asarray([float(x) for x in xs], dtype=float)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return {"mean": float(a.mean()), "sd": sd, "n": int(n),
            "two_se": float(2.0 * sd / math.sqrt(max(n, 1))), "values": a.tolist()}


def _byte_ledger(arm: str, led: Dict[str, Any], reader_params: Dict[str, int],
                 phi_bytes: int) -> Dict[str, Any]:
    """The **two-sided** per-arm byte ledger (frozen template's six rows).

    ⛔ Every number is emitted by the code that computes it, never from a doc
    (the C2W5 ``FROZEN-interfaces.md`` failure: its ledger row *and* its reader
    parameter counts were both wrong).
    """
    rp = int(max(reader_params.values())) if reader_params else 0
    rows = {"store": int(led.get("param_bytes", 0)),
            "phi": int(phi_bytes),
            "launch_head": 0,  # FeatureLaunchHead holds ZERO parameters of its own
            "projection": 0,   # no arm learns a projection (declared)
            "reader_params_bytes": int(rp) * 4,
            "state": int(led.get("state_bytes", 0))}
    rows["total_bytes"] = int(sum(rows.values()))
    rows["arm"] = arm
    rows["reader_params_max"] = rp
    rows["read_mult_adds_per_query"] = int(led.get("read_flops", 0))
    return rows


# ==========================================================================
# STAGE guards — phi, launches, tol: byte-compared and pytest-mirrored
# ==========================================================================
def stage_guards(cfg: CatTestConfig, seeds: Sequence[int] = (0, 1, 2),
                 frozen: Optional[Dict[str, Any]] = None,
                 out: Optional[Path] = None) -> Dict[str, Any]:
    """The matching obligations, measured before any arm is fitted."""
    import hashlib

    cells = []
    for s in seeds:
        S = seed_setup(cfg, s, n_val=C2W11_GRID.n_val, frozen=frozen)
        nb, h = _phi_bytes_hash(S["phi"])
        # bit-identity of the launch against the PHYSICS read path: a zero-step
        # multi_particle_read returns exactly its launch stage.
        from chlu.core.factored_store import FactoredStore
        c0 = replace(cfg, address_steps=0, read_steps=0)
        blank = FactoredStore(cfg, S["anchors"], jax.random.PRNGKey(int(s)),
                              atom_width=0.3)
        z0 = multi_particle_read(blank, S["head"], c0, S["ind_u"],
                                 jax.random.fold_in(S["key"], 1))
        same = bool(np.array_equal(np.asarray(z0, dtype=np.float32),
                                   np.asarray(S["q0_u"], dtype=np.float32)))
        cells.append({
            "seed": int(s), "phi_bytes": nb, "phi_byte_hash": h,
            "phi_hash_matches_frozen": (
                bool(frozen and h == frozen["phi"]["byte_hash"])),
            "launch_bit_identical_to_physics_read": same,
            "launch_point_hash": hashlib.md5(
                np.asarray(S["q0_u"], dtype=np.float32).tobytes()).hexdigest(),
            "tol": float(S["family"].tol), "chance": float(S["chance"]),
            "frozen_check": S["frozen_check"],
            "val_rule4": S["val_rule4"],
            "head_bytes": S["head"].n_bytes(),
        })
        print(f"[guards] seed={s} phi={nb}B hash={h[:8]} "
              f"match={cells[-1]['phi_hash_matches_frozen']} "
              f"launch_bit_identical={same} tol={S['family'].tol:.6f}", flush=True)
    res = {"stage": "guards", "cells": cells,
           "phi_byte_identical_across_seeds": len({c["phi_byte_hash"]
                                                   for c in cells}) == 1,
           "all_launches_bit_identical": all(
               c["launch_bit_identical_to_physics_read"] for c in cells),
           "all_phi_match_frozen": all(c["phi_hash_matches_frozen"]
                                       for c in cells),
           "VALUE_LAUNCH_KEY": VALUE_LAUNCH_KEY,
           "rider_R1": ("the VALUE launch key is NOT frozen by spoke A; this "
                        "spoke adopts k3_k4_k5's PRNGKey(7000+seed). Spoke B "
                        "must match or V1/V3 are scored on different launches.")}
    if out:
        _dump(res, out / "stage_guards.json")
    return res


# ==========================================================================
# STAGE grid — the registered F3 budget, COMPUTED
# ==========================================================================
def stage_grid(cfg: CatTestConfig, grid: NullArmGrid, arms: Sequence[str] = ARMS,
               out: Optional[Path] = None, seeds: Optional[Sequence[int]] = None,
               frozen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """≥ 5 lr × 3 capacity × 3 seeds per arm, selected on the held-out-from-seen split.

    ⛔ ``Q_unseen`` is not read anywhere in this function. Selection statistic:
    the arm's **own read** on ``S["va"]`` — accuracy first, MSE as the declared
    tie-break (⚠ C2W5's honest-power note: at 32 validation rows the grain is
    1/32 = 0.031 and every config scored 0.0000, so selection ran entirely on the
    tie-break; :func:`stage_gridmax` exists so the verdict does not depend on the
    selection statistic having resolved anything).
    """
    seeds = tuple(grid.tune_seeds if seeds is None else seeds)
    res: Dict[str, Any] = {"stage": "grid", "seeds": list(seeds), "records": {},
                           "selected": {}, "n_configs": {}, "n_diverged": {}}
    setups = {s: seed_setup(cfg, s, n_val=grid.n_val, frozen=frozen) for s in seeds}
    for arm in arms:
        confs = _grid_configs(arm, grid)
        res["n_configs"][arm] = len(confs)
        recs, t0 = [], time.time()
        for ci, conf in enumerate(confs):
            per_seed = []
            for s in seeds:
                S = setups[s]
                fam = S["family"]
                fitted = _fit_arm(arm, conf, S, cfg, grid, idx=S["tr"])
                pred = _arm_predict(fitted, S["q0_s"][S["va"]],
                                    S["ind_s"][S["va"]], cfg)
                sc = _native(pred, fam.y_seen[S["va"]], fam.tol)
                pred_tr = _arm_predict(fitted, S["q0_s"][S["tr"]],
                                       S["ind_s"][S["tr"]], cfg)
                sc_tr = _native(pred_tr, fam.y_seen[S["tr"]], fam.tol)
                per_seed.append({"seed": s, "val": sc, "train": sc_tr,
                                 "n_params": fitted["ledger"]["n_params"]})
            recs.append({"config": conf,
                         "val_acc": float(np.mean([p["val"]["acc"] for p in per_seed])),
                         "val_mse": float(np.mean([p["val"]["mse"] for p in per_seed])),
                         "train_acc": float(np.mean([p["train"]["acc"]
                                                     for p in per_seed])),
                         "train_mse": float(np.mean([p["train"]["mse"]
                                                     for p in per_seed])),
                         "n_params": per_seed[0]["n_params"],
                         "diverged": bool(any(p["val"]["diverged"]
                                              for p in per_seed)),
                         "per_seed": per_seed})
            if (ci + 1) % 20 == 0 or ci + 1 == len(confs):
                print(f"[grid:{arm}] {ci+1}/{len(confs)} "
                      f"best_val_acc={max(r['val_acc'] for r in recs):.4f} "
                      f"best_train_acc={max(r['train_acc'] for r in recs):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        best = max(recs, key=lambda r: (r["val_acc"], -r["val_mse"]))
        res["records"][arm] = recs
        res["n_diverged"][arm] = int(sum(r["diverged"] for r in recs))
        res["selected"][arm] = {"config": best["config"],
                                "val_acc": best["val_acc"],
                                "val_mse": best["val_mse"],
                                "train_acc": best["train_acc"],
                                "train_mse": best["train_mse"],
                                "n_params": best["n_params"],
                                "wall_s": round(time.time() - t0, 1)}
        # ⚠ a diverged config must LOSE selection, never disappear (asserted)
        res["selected"][arm]["diverged_can_win"] = bool(best["diverged"])
        print(f"[grid:{arm}] SELECTED {best['config']} "
              f"val_acc={best['val_acc']:.4f} train_acc={best['train_acc']:.4f} "
              f"diverged={res['n_diverged'][arm]}/{len(confs)}", flush=True)
    if out:
        _dump(res, out / "stage_grid.json")
    return res


# ==========================================================================
# STAGE score — V1, plus the L1-L4 anchors and the two-sided ledgers
# ==========================================================================
def stage_score(cfg: CatTestConfig, grid: NullArmGrid, selected: Dict[str, Any],
                arms: Sequence[str] = ARMS, out: Optional[Path] = None,
                seeds: Optional[Sequence[int]] = None,
                frozen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """**V1**: 5 seeds, the frozen reader class **plus its zero-parameter twins**.

    Every arm is refit on **all** of SEEN at its selected config, readers are
    fitted on SEEN only, and only then is ``Q_unseen`` scored. The shuffle-φ
    launder re-runs the identical pipeline with the launch blocks permuted across
    queries (⛔ DIAGNOSTIC — no launder margin is a pass condition anywhere).
    """
    seeds = tuple(grid.score_seeds if seeds is None else seeds)
    cells: List[Dict[str, Any]] = []
    for s in seeds:
        S = seed_setup(cfg, s, n_val=grid.n_val, frozen=frozen)
        fam = S["family"]
        allidx = np.arange(len(fam.seen))
        for arm in arms:
            t0 = time.time()
            conf = selected[arm]["config"]
            fitted = _fit_arm(arm, conf, S, cfg, grid, idx=allidx)
            row: Dict[str, Any] = {"seed": int(s), "arm": arm, "config": conf,
                                   "chance": S["chance"], "tol": float(fam.tol),
                                   "ledger": fitted["ledger"],
                                   "train": fitted.get("train", {})}
            pu = _arm_predict(fitted, S["q0_u"], S["ind_u"], cfg)
            ps = _arm_predict(fitted, S["q0_s"], S["ind_s"], cfg)
            row["native_unseen"] = _native(pu, fam.y_unseen, fam.tol)
            row["native_seen"] = _native(ps, fam.y_seen, fam.tol)
            row["native_curve"] = {
                f"x{m:g}": exact_set_accuracy(pu, fam.y_unseen, fam.tol * m)
                for m in (0.25, 0.5, 1.0, 2.0, 4.0)}
            # -- the frozen reader class (+ zero-parameter twins) -----------
            if "z" in fitted:
                z_s, z_u = fitted["z"](S["q0_s"]), fitted["z"](S["q0_u"])
                anc, pay = fitted["codebook"]
                rd = fit_readers_plus_identity(z_s, fam.y_seen, anchors=anc,
                                               well_payloads=pay,
                                               addr_dim=int(cfg.addr_dim), seed=s)
                row["readers_unseen"] = score_readers_plus_identity(
                    rd, z_u, fam.y_unseen, fam.tol)
                row["readers_seen"] = score_readers_plus_identity(
                    rd, z_s, fam.y_seen, fam.tol)
                row["reader_params"] = {k: int(v.get("n_params", 0))
                                        for k, v in rd.items()}
                row["best_reader_unseen"] = float(max(row["readers_unseen"].values()))
            else:
                row["readers_unseen"] = {"direct_predictor": row["native_unseen"]["acc"]}
                row["reader_params"] = {"direct_predictor": 0}
                row["best_reader_unseen"] = float(row["native_unseen"]["acc"])
                row["readers_note"] = ("⛔ N4/N5 are DIRECT PREDICTORS: they have "
                                       "no latent z and therefore no reader-class "
                                       "column. Reported as such, never dressed "
                                       "up as a latent map.")
            row["best_unseen"] = float(max(row["best_reader_unseen"],
                                           row["native_unseen"]["acc"]))
            # -- the shuffle-phi launder (DIAGNOSTIC) ----------------------
            q0_sh = shuffle_launches(S["q0_u"], s)
            row["launder_shuffle_phi"] = _native(
                _arm_predict(fitted, q0_sh, S["ind_u"], cfg),
                fam.y_unseen, fam.tol)["acc"]
            row["bytes"] = _byte_ledger(arm, fitted["ledger"],
                                        row.get("reader_params", {}),
                                        int(S["phi"].n_bytes()))
            row["wall_s"] = round(time.time() - t0, 1)
            cells.append(row)
            print(f"[score] seed={s} {arm} unseen={row['best_unseen']:.5f} "
                  f"seen={row['native_seen']['acc']:.4f} "
                  f"launder={row['launder_shuffle_phi']:.5f} "
                  f"({row['wall_s']}s)", flush=True)

    # -- per-arm aggregation ------------------------------------------------
    per_arm: Dict[str, Any] = {}
    for arm in arms:
        rs = [c for c in cells if c["arm"] == arm]
        per_arm[arm] = {
            "unseen": _mean_2se([c["best_unseen"] for c in rs]),
            "seen": _mean_2se([c["native_seen"]["acc"] for c in rs]),
            "launder": _mean_2se([c["launder_shuffle_phi"] for c in rs]),
            "config": rs[0]["config"], "bytes": rs[0]["bytes"]}
    chance = float(np.mean([c["chance"] for c in cells]))
    res = {"stage": "score", "cells": cells, "per_arm": per_arm,
           "chance": chance, "bar_context_only": chance + 0.05,
           "max_arm_selected": float(max(c["best_unseen"] for c in cells)),
           "note": ("⛔ `max over the SELECTED configs` is NOT null*. null* is "
                    "the grid-max of stage_gridmax, an oracle-selected upper "
                    "bound over the ENTIRE registered grid.")}
    if out:
        _dump(res, out / "stage_score.json")
    return res


# ==========================================================================
# STAGE gridmax — ⭐ null*, COMPUTED over the ENTIRE registered grid
# ==========================================================================
def stage_gridmax(cfg: CatTestConfig, grid: NullArmGrid,
                  arms: Sequence[str] = ARMS, out: Optional[Path] = None,
                  seeds: Optional[Sequence[int]] = None,
                  frozen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """⭐⭐ ``null* = max over ALL arms AND their ENTIRE registered tuning grid``.

    ⛔ **Computed, not estimated** (C2W5's 584 configs × 5 seeds is the permanent
    standard). Every configuration is refit on all of SEEN and scored on
    ``Q_unseen``. It is an explicitly **oracle-selected upper bound**, reported so
    the verdict reads *"no configuration clears"* rather than *"the one we picked
    didn't"* — ⛔ and it is never any arm's own score.
    """
    seeds = tuple(grid.score_seeds if seeds is None else seeds)
    setups = {s: seed_setup(cfg, s, n_val=grid.n_val, frozen=frozen) for s in seeds}
    res: Dict[str, Any] = {"stage": "gridmax", "seeds": list(seeds), "arms": {},
                           "n_configs_total": 0}
    for arm in arms:
        confs = _grid_configs(arm, grid)
        res["n_configs_total"] += len(confs)
        rows, t0 = [], time.time()
        for ci, conf in enumerate(confs):
            accs = []
            for s in seeds:
                S = setups[s]
                fam = S["family"]
                fitted = _fit_arm(arm, conf, S, cfg, grid,
                                  idx=np.arange(len(fam.seen)))
                pu = _arm_predict(fitted, S["q0_u"], S["ind_u"], cfg)
                accs.append(_native(pu, fam.y_unseen, fam.tol)["acc"])
            rows.append({"config": conf, "mean": float(np.mean(accs)),
                         "max_seed": float(np.max(accs)), "per_seed": accs})
            if (ci + 1) % 25 == 0 or ci + 1 == len(confs):
                print(f"[gridmax:{arm}] {ci+1}/{len(confs)} "
                      f"max={max(r['mean'] for r in rows):.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        best = max(rows, key=lambda r: r["mean"])
        res["arms"][arm] = {"n_configs": len(confs), "grid_max_mean": best["mean"],
                            "argmax": best["config"],
                            "best_single_seed": float(max(r["max_seed"]
                                                          for r in rows)),
                            "rows": rows, "wall_s": round(time.time() - t0, 1)}
        print(f"[gridmax:{arm}] GRID-MAX {best['mean']:.5f} at {best['config']}",
              flush=True)
    ns = {a: res["arms"][a]["grid_max_mean"] for a in arms}
    res["null_star"] = float(max(ns.values())) if ns else float("nan")
    res["null_star_arm"] = max(ns, key=ns.get) if ns else None
    res["n_score_seeds"] = len(seeds)
    res["label"] = ("⛔ ORACLE-SELECTED UPPER BOUND over the entire registered "
                    "grid x score seeds. Never any arm's own score.")
    if out:
        _dump(res, out / "stage_gridmax.json")
    return res


# ==========================================================================
# STAGE v2 — the matched confidence channel (VALUE leg ii's NULL side)
# ==========================================================================
def build_novelty_family(cfg: CatTestConfig, seed: int,
                         n_novel: int = N_NOVEL_WELLS,
                         n_eval_per_class: int = 128):
    """A SEEN split over **written** wells only, and an eval set with 0/1/2 novel channels.

    ⛔ **R2 (declared rider): this split is NOT frozen by spoke A.** Registered
    here: ``n_novel = 4`` wells drawn by ``default_rng(20260811 + seed)`` are
    **never written**; SEEN is ``K`` rule-4-valid ``F``-subsets of the remaining
    ``N_a - n_novel``; the eval set carries exactly 0, 1 or 2 novel wells and is
    rule-4-valid against every SEEN row. φ is untouched — it carries a code for
    every well, written or not — so the launch head is unchanged and the arms are
    still on the frozen launch protocol.

    Returns a :class:`~chlu.core.factored_store.CatFamily`-shaped object plus the
    novel-well set and the per-query novel count.
    """
    from chlu.core.factored_store import CatFamily

    rng = np.random.default_rng(NOVELTY_SEED_BASE + int(seed))
    N_a, F, K, m = int(cfg.n_wells), int(cfg.f_subset), int(cfg.n_items), \
        int(cfg.payload_dim)
    novel = np.sort(rng.choice(N_a, size=int(n_novel), replace=False))
    known = np.setdiff1d(np.arange(N_a), novel)

    combos_known = np.array(list(itertools.combinations(known.tolist(), F)),
                            dtype=np.int32)
    pick = rng.choice(len(combos_known), size=min(K, len(combos_known)),
                      replace=False)
    seen = np.sort(combos_known[pick].astype(int), axis=1)

    # payloads on the sphere (the family's own law), all N_a wells
    g = rng.normal(size=(N_a, m))
    payloads = float(cfg.payload_radius) * g / np.linalg.norm(g, axis=1,
                                                              keepdims=True)
    ind_seen = np.zeros((len(seen), N_a), dtype=np.int32)
    np.put_along_axis(ind_seen, seen, 1, axis=1)

    def rule4_ok(A):
        a = np.zeros(N_a, dtype=np.int32)
        a[np.asarray(A)] = 1
        return bool((ind_seen @ a).max() <= F - 2)

    ev, ev_novel = [], []
    all_c = np.array(list(itertools.combinations(range(N_a), F)), dtype=np.int32)
    n_nov_per = np.isin(all_c, novel).sum(1)
    for j in (0, 1, 2):
        cand = all_c[n_nov_per == j]
        rng.shuffle(cand)
        took = 0
        for A in cand:
            if took >= int(n_eval_per_class):
                break
            if rule4_ok(A):
                ev.append(np.sort(A))
                ev_novel.append(j)
                took += 1
    ev = np.asarray(ev, dtype=int)
    ev_novel = np.asarray(ev_novel, dtype=int)
    y_seen = ind_seen.astype(np.float64) @ payloads
    ind_ev = np.zeros((len(ev), N_a), dtype=np.int32)
    np.put_along_axis(ind_ev, ev, 1, axis=1)
    y_ev = ind_ev.astype(np.float64) @ payloads
    tol = float(cfg.tol_frac * np.sqrt(np.mean(np.sum(
        (y_ev - y_ev.mean(0)) ** 2, -1))))
    fam = CatFamily(seen=seen, unseen=ev, payloads=payloads, y_seen=y_seen,
                    y_unseen=y_ev, tol=tol, n_valid_heldout=int(len(ev)),
                    n_total_combos=int(len(all_c)),
                    order=rng.permutation(len(seen)),
                    k2={"construction": "C2W11 spoke C novelty split (rider R2)"})
    return fam, novel, ev_novel


def stage_v2(cfg: CatTestConfig, grid: NullArmGrid, selected: Dict[str, Any],
             arms: Sequence[str] = ARMS, out: Optional[Path] = None,
             seeds: Optional[Sequence[int]] = None,
             frozen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """**V2a** per-feature novelty AUROC and **V2b** set-level ECE, per arm.

    ⛔ Every arm here HAS a principled confidence channel (see
    :mod:`chlu.core.null_arms`); none is a NOT-RUN, and none is scored as
    "uncalibrated".

    ⚠ **The designed negative, corrected where it does not bind.**
    ``PREREG-C2W11.md`` §5 registers *permuted payloads ⇒ AUROC ≈ 0.5*. That
    negative is built for a novelty head that reads **payloads**; four of the five
    null channels are functions of the **address** geometry alone and are
    therefore *structurally invariant* under a payload permutation — which is a
    measurement, reported as such, and NOT a pass. The negative that can fail for
    these channels is the **label permutation** (shuffle which wells are novel),
    and it is run and asserted for every arm.
    """
    seeds = tuple(grid.score_seeds if seeds is None else seeds)
    cells: List[Dict[str, Any]] = []
    for s in seeds:
        cfgn = cfg
        fam, novel, ev_novel = build_novelty_family(cfgn, s)
        phi = build_phi(cfgn)
        head = build_launch_head(phi, cfgn)
        ruler = cfgn.s_measured if cfgn.s_measured is not None else cfgn.atom_width
        anchors = place_wells(phi, cfgn, sep=float(cfgn.target_ds * ruler))
        ind_s = fam.indicator(fam.seen, cfgn.n_wells)
        ind_u = fam.indicator(fam.unseen, cfgn.n_wells)
        k = jax.random.PRNGKey(VALUE_LAUNCH_KEY + int(s))
        q0_s = feature_launch_states(head, cfgn, ind_s, k)
        q0_u = feature_launch_states(head, cfgn, ind_u, jax.random.fold_in(k, 1))
        S = {"family": fam, "phi": phi, "head": head, "anchors": anchors,
             "ind_s": ind_s, "ind_u": ind_u, "q0_s": q0_s, "q0_u": q0_u,
             "seed": int(s)}
        # per-particle novel label: the channel's own launch code is an
        # UNWRITTEN well. (The per-query novel count is `ev_novel`.)
        cd = float(cfgn.ball_radius) * np.asarray(head.codes, dtype=np.float64)
        picks = ((np.asarray(q0_u)[..., : cfgn.addr_dim][:, :, None, :]
                  - cd[None, None]) ** 2).sum(-1).argmin(-1)   # (B, k)
        lab = np.isin(picks, novel)
        for arm in arms:
            t0 = time.time()
            conf_cfg = selected[arm]["config"]
            fitted = _fit_arm(arm, conf_cfg, S, cfgn, grid,
                              idx=np.arange(len(fam.seen)), family=fam)
            cu = np.asarray(fitted["conf"](S["q0_u"], S["ind_u"]), dtype=np.float64)
            cs = np.asarray(fitted["conf"](S["q0_s"], S["ind_s"]), dtype=np.float64)
            auroc = novelty_auroc(cu, lab)
            # -- label-permutation designed negative (the one that can FAIL) --
            rng = np.random.default_rng(int(s) + 555)
            nov_p = rng.permutation(int(cfgn.n_wells))[: len(novel)]
            auroc_perm_label = novelty_auroc(cu, np.isin(picks, nov_p))
            # -- the registered permuted-payload negative --------------------
            fam_p = replace_payload_permutation(fam, int(s))
            fitted_p = _fit_arm(arm, conf_cfg, S, cfgn, grid,
                                idx=np.arange(len(fam.seen)), family=fam_p)
            auroc_perm_pay = novelty_auroc(
                np.asarray(fitted_p["conf"](S["q0_u"], S["ind_u"])), lab)
            # -- V2b: set-level ECE, calibrated on the SEEN split ONLY -------
            agg_u, agg_s = cu.mean(1), cs.mean(1)
            q = np.sort(agg_s)                       # the empirical CDF of SEEN
            p_u = np.searchsorted(q, agg_u, side="right") / max(len(q), 1)
            pred_u = _arm_predict(fitted, S["q0_u"], S["ind_u"], cfgn)
            correct = (np.linalg.norm(np.asarray(pred_u) - fam.y_unseen, axis=-1)
                       <= fam.tol)
            ece = expected_calibration_error(p_u, correct)
            per_class = {int(j): float(np.mean(agg_u[ev_novel == j]))
                         for j in (0, 1, 2) if (ev_novel == j).any()}
            variant = float("nan")
            if "conf_variant" in fitted:
                variant = novelty_auroc(
                    np.asarray(fitted["conf_variant"](S["q0_u"], S["ind_u"])), lab)
            cells.append({
                "seed": int(s), "arm": arm, "config": conf_cfg,
                "V2a_auroc": float(auroc),
                "V2a_auroc_registered_variant": float(variant),
                "V2a_negative_label_permutation": float(auroc_perm_label),
                "V2a_negative_permuted_payloads": float(auroc_perm_pay),
                "V2b_ece": ece, "mean_conf_by_novel_count": per_class,
                "n_novel_wells": int(len(novel)),
                "n_particles_novel": int(lab.sum()),
                "n_particles": int(lab.size),
                "tol": float(fam.tol), "wall_s": round(time.time() - t0, 1)})
            print(f"[v2] seed={s} {arm} AUROC={auroc:.4f} "
                  f"(label-perm {auroc_perm_label:.4f}, pay-perm "
                  f"{auroc_perm_pay:.4f}) ECE={ece['ece']:.4f} "
                  f"acc={ece['acc']:.4f}", flush=True)
    per_arm = {}
    for arm in arms:
        rs = [c for c in cells if c["arm"] == arm]
        per_arm[arm] = {"V2a": _mean_2se([c["V2a_auroc"] for c in rs]),
                        "V2a_registered_variant": _mean_2se(
                            [c["V2a_auroc_registered_variant"] for c in rs]),
                        "V2a_negative_label_permutation": _mean_2se(
                            [c["V2a_negative_label_permutation"] for c in rs]),
                        "V2a_negative_permuted_payloads": _mean_2se(
                            [c["V2a_negative_permuted_payloads"] for c in rs]),
                        "V2b_ece": _mean_2se([c["V2b_ece"]["ece"] for c in rs]),
                        "V2b_acc": _mean_2se([c["V2b_ece"]["acc"] for c in rs]),
                        "V2b_degenerate": bool(all(c["V2b_ece"]["degenerate"]
                                                   for c in rs))}
    res = {"stage": "v2", "cells": cells, "per_arm": per_arm,
           "null_star_V2a": float(max(per_arm[a]["V2a"]["mean"] for a in arms)),
           "null_star_V2a_arm": max(arms, key=lambda a: per_arm[a]["V2a"]["mean"]),
           "best_V2b_ece": float(min(per_arm[a]["V2b_ece"]["mean"] for a in arms)),
           "rider_R2": ("the novel/known split is NOT frozen: n_novel=4 wells "
                        "by default_rng(20260811+seed), never written. Spoke B "
                        "must use the same rule."),
           "designed_negative_note": (
               "⛔ permuted payloads does NOT bind an address-geometry "
               "confidence channel and is reported as structurally invariant, "
               "never as a pass. The binding negative is the label permutation.")}
    if out:
        _dump(res, out / "stage_v2.json")
    return res


def replace_payload_permutation(fam, seed: int):
    """The registered permuted-payload null: same wells, ``v_j`` permuted."""
    from chlu.core.factored_store import CatFamily

    rng = np.random.default_rng(int(seed) + 99)
    perm = rng.permutation(len(fam.payloads))
    pay = np.asarray(fam.payloads)[perm]
    ind_s = fam.indicator(fam.seen, len(pay))
    ind_u = fam.indicator(fam.unseen, len(pay))
    return CatFamily(seen=fam.seen, unseen=fam.unseen, payloads=pay,
                     y_seen=ind_s @ pay, y_unseen=ind_u @ pay, tol=fam.tol,
                     n_valid_heldout=fam.n_valid_heldout,
                     n_total_combos=fam.n_total_combos, order=fam.order,
                     k2={"permuted_payloads": True})


# ==========================================================================
# STAGE v3 — the anytime curve on the NULL side (leg iii)
# ==========================================================================
def _v3_budget_grid(frozen: Optional[Dict[str, Any]]) -> List[int]:
    """⛔ Quoted from the frozen artifact — a mismatched axis VOIDS leg iii."""
    if frozen and "v3_budget_grid" in frozen:
        return [int(b) for b in frozen["v3_budget_grid"]["points_total_verlet_steps"]]
    return [50, 100, 200, 400, 800, 1200]


def _v3_score(store, S, cfg, budget: int, arm_codebook, fam, *, key,
              fit_readers_too: bool = False) -> Dict[str, Any]:
    """One (store, budget) cell: the identical k-particle anytime read, scored."""
    z_u = anytime_read(store, S["head"], cfg, S["ind_u"], jax.random.fold_in(key, 1),
                       budget=budget)
    anc, pay = arm_codebook
    out = {"native": _native(_z_native(z_u, cfg), fam.y_unseen, fam.tol)["acc"]}
    from chlu.core.null_arms import well_identity_apply
    out["well_identity"] = exact_set_accuracy(
        well_identity_apply({"anchors": np.asarray(anc)[:, : cfg.addr_dim],
                             "well_payloads": np.asarray(pay)}, z_u),
        fam.y_unseen, fam.tol)
    if fit_readers_too:
        z_s = anytime_read(store, S["head"], cfg, S["ind_s"], key, budget=budget)
        rd = fit_readers_plus_identity(z_s, fam.y_seen,
                                       anchors=np.asarray(anc)[:, : cfg.addr_dim],
                                       well_payloads=np.asarray(pay),
                                       addr_dim=int(cfg.addr_dim), seed=S["seed"])
        out["readers"] = score_readers_plus_identity(rd, z_u, fam.y_unseen, fam.tol)
        out["reader_params"] = {k: int(v.get("n_params", 0)) for k, v in rd.items()}
    out["best"] = float(max([out["native"], out["well_identity"]]
                            + list(out.get("readers", {}).values())))
    return out


def stage_v3(cfg: CatTestConfig, grid: NullArmGrid, selected: Dict[str, Any],
             arms: Sequence[str] = ("N1", "N2", "N3"),
             out: Optional[Path] = None, seeds: Optional[Sequence[int]] = None,
             frozen: Optional[Dict[str, Any]] = None,
             static_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """⭐⭐ **V3's null side** — instantiated landscapes, read with the IDENTICAL read.

    Two phases:

    1. **The instantiation sweep (F3-grade).** ⚠⚠ *"You gave the null a badly
       instantiated landscape"* is the same referee attack as *"you hobbled the
       competition"*, so the three knobs that could hobble it — atom budget,
       co-scaled width, amplitude — are swept (5 × 3 × 2 = 30 configs) and
       selected on the **held-out-from-seen** split at the top budget.
    2. **The curve**, at the **frozen** budget grid, 5 seeds, plus the
       ``null*_V3`` grid-max over the width axis at every budget point.

    ⛔ **N4/N5 are DECLARED NOT-RUN here** — no landscape exists — and their
    static scores are reported as flat reference lines, never as "un-navigable".
    """
    seeds = tuple(grid.score_seeds if seeds is None else seeds)
    budgets = _v3_budget_grid(frozen)
    res: Dict[str, Any] = {"stage": "v3", "budgets": budgets,
                           "budget_grid_source": ("FROZEN-INTERFACES-C2W11.json::"
                                                  "v3_budget_grid" if frozen
                                                  else "module default"),
                           "instantiation_sweep": {}, "curves": {},
                           "NOT_RUN": {
                               "N4": "no landscape exists (kNN over raw rows)",
                               "N5": "no landscape exists (Titans fast weights)"},
                           "flat_reference_lines": static_scores or {}}

    # ⚠ ONE setup and ONE arm fit per seed, reused across the whole stage: a
    # `seed_setup` enumerates C(N_a,F) = 35 960 combinations and re-verifies
    # rule 4, so rebuilding it inside the sweep's inner loop (30 configs x 3
    # seeds x 3 arms) would spend the entire stage's budget on family
    # construction and none of it on the read under test.
    setups = {s: seed_setup(cfg, s, n_val=grid.n_val, frozen=frozen)
              for s in set(tuple(grid.tune_seeds)) | set(seeds)}

    # -- phase 1: the instantiation sweep, on the SEEN validation slice ------
    tune_seeds = tuple(grid.tune_seeds)
    for arm in arms:
        rows, t0 = [], time.time()
        tr_fit = {s: _fit_arm(arm, selected[arm]["config"], setups[s], cfg, grid,
                              idx=setups[s]["tr"]) for s in tune_seeds}
        for wf, dep, tot in itertools.product(V3_WIDTH_FRACS, V3_DEPTHS,
                                              V3_ATOM_BUDGETS):
            accs = []
            for s in tune_seeds:
                S = setups[s]
                fam = S["family"]
                anc, pay = tr_fit[s]["codebook"]
                store, _info = instantiate_landscape(
                    cfg, np.asarray(anc), np.asarray(pay), seed=s,
                    total_atoms=tot, width_frac=wf, place_depth=dep)
                z_va = anytime_read(store, S["head"], cfg, S["ind_s"][S["va"]],
                                    S["key"], budget=budgets[-1])
                accs.append(_native(_z_native(z_va, cfg), fam.y_seen[S["va"]],
                                    fam.tol)["acc"])
            rows.append({"width_frac": wf, "place_depth": dep, "total_atoms": tot,
                         "val_acc": float(np.mean(accs))})
        best = max(rows, key=lambda r: r["val_acc"])
        res["instantiation_sweep"][arm] = {
            "n_configs": len(rows), "rows": rows, "selected": best,
            "wall_s": round(time.time() - t0, 1),
            "note": ("selected on the held-out-from-seen split at the top "
                     "budget; ⛔ Q_unseen is not read in this phase")}
        print(f"[v3:{arm}] instantiation SELECTED {best} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # -- phase 2: the curve + the null*_V3 grid-max --------------------------
    for arm in arms:
        sel = res["instantiation_sweep"][arm]["selected"]
        curve: Dict[str, Any] = {}
        gridmax: Dict[str, Any] = {}
        t0 = time.time()
        # one fit and one store per (seed, width) — the READ is the object under
        # test, so everything upstream of it is built once.
        book, stores = {}, {}
        for s in seeds:
            S = setups[s]
            fitted = _fit_arm(arm, selected[arm]["config"], S, cfg, grid,
                              idx=np.arange(len(S["family"].seen)))
            anc, pay = fitted["codebook"]
            book[s] = (np.asarray(anc), np.asarray(pay))
            for wf in V3_GRIDMAX_WIDTHS:
                stores[(s, wf)] = instantiate_landscape(
                    cfg, book[s][0], book[s][1], seed=s,
                    total_atoms=sel["total_atoms"], width_frac=wf,
                    place_depth=sel["place_depth"])
        for b in budgets:
            per_seed, per_seed_max = [], []
            for s in seeds:
                S, fam = setups[s], setups[s]["family"]
                store, info = stores[(s, sel["width_frac"])]
                cell = _v3_score(store, S, cfg, b, book[s], fam, key=S["key"],
                                 fit_readers_too=True)
                cell["store_bytes"] = info["store_bytes"]
                cell["read_mult_adds_per_query"] = int(
                    int(cfg.n_particles) * b * info["total_atoms"]
                    * (int(cfg.dim) + 1))
                per_seed.append(cell)
                # the width-axis grid-max at this budget (zero-parameter reads)
                mx = cell["best"]
                for wf in V3_GRIDMAX_WIDTHS:
                    if wf == sel["width_frac"]:
                        continue
                    c2 = _v3_score(stores[(s, wf)][0], S, cfg, b, book[s], fam,
                                   key=S["key"])
                    mx = max(mx, c2["best"])
                per_seed_max.append(mx)
            curve[str(b)] = {"best": _mean_2se([c["best"] for c in per_seed]),
                             "native": _mean_2se([c["native"] for c in per_seed]),
                             "well_identity": _mean_2se(
                                 [c["well_identity"] for c in per_seed]),
                             "store_bytes": per_seed[0]["store_bytes"],
                             "read_mult_adds_per_query":
                                 per_seed[0]["read_mult_adds_per_query"],
                             "reader_params": per_seed[0].get("reader_params", {})}
            gridmax[str(b)] = _mean_2se(per_seed_max)
            print(f"[v3:{arm}] b={b} best={curve[str(b)]['best']['mean']:.5f} "
                  f"gridmax={gridmax[str(b)]['mean']:.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        vals = [curve[str(b)]["best"]["mean"] for b in budgets]
        res["curves"][arm] = {
            "selected_instantiation": sel, "curve": curve, "gridmax": gridmax,
            "flat": bool(max(vals) - min(vals) <= 0.004),
            "monotone": bool(all(vals[i] <= vals[i + 1] + 1e-12
                                 for i in range(len(vals) - 1))),
            "span": float(max(vals) - min(vals)), "wall_s": round(time.time() - t0, 1)}
    res["null_star_V3_by_budget"] = {
        str(b): float(max(res["curves"][a]["gridmax"][str(b)]["mean"]
                          for a in arms)) for b in budgets}
    res["null_star_V3"] = float(max(res["null_star_V3_by_budget"].values()))
    # the physics arm's read cost at the frozen full budget, for the RATIO
    res["physics_read_mult_adds_per_query"] = int(read_flops("physics", cfg))
    if out:
        _dump(res, out / "stage_v3.json")
    return res


# ==========================================================================
# STAGE ceiling — ⛔ out-of-class, and never quoted apart from the verdict
# ==========================================================================
def stage_ceiling(cfg: CatTestConfig, grid: NullArmGrid,
                  out: Optional[Path] = None,
                  seeds: Optional[Sequence[int]] = None,
                  frozen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """⛔ **THE DECODABILITY CEILING, RECOMPUTED ON THE C2W11 LAUNCHES.**

    Declared **out-of-class** and never scored as an arm. It is what turns "no
    arm clears" into an **attributable** finding, and ⛔ it is never quoted
    without it. Banked at C2W5's launches: 1.0000 noiseless, 0.2719 ± 0.0126
    as-launched.
    """
    seeds = tuple(grid.score_seeds if seeds is None else seeds)
    cells = []
    for s in seeds:
        S = seed_setup(cfg, s, n_val=grid.n_val, frozen=frozen)
        t0 = time.time()
        c = feature_decodability_ceiling(S["head"], cfg, S["family"], S["q0_u"])
        c["seed"] = int(s)
        c["wall_s"] = round(time.time() - t0, 1)
        cells.append(c)
        print(f"[ceiling] seed={s} noiseless={c['noiseless_accuracy']:.4f} "
              f"as_launched={c['as_launched_accuracy']:.4f} "
              f"asserted_exact={c['asserted_set_exact']:.4f} "
              f"({c['wall_s']}s)", flush=True)
    res = {"stage": "ceiling", "cells": cells,
           "noiseless": _mean_2se([c["noiseless_accuracy"] for c in cells]),
           "as_launched": _mean_2se([c["as_launched_accuracy"] for c in cells]),
           "asserted_set_exact": _mean_2se([c["asserted_set_exact"]
                                            for c in cells]),
           "banked_C2W5": {"noiseless": 1.0, "as_launched": 0.2719},
           "label": "⛔ DECLARED OUT-OF-CLASS — never an arm, never a score."}
    if out:
        _dump(res, out / "stage_ceiling.json")
    return res


# ==========================================================================
# STAGE oracle — N3 fitted on the physics store's OWN assignments (T5.2 (i))
# ==========================================================================
def stage_oracle(cfg: CatTestConfig, grid: NullArmGrid,
                 out: Optional[Path] = None,
                 seeds: Optional[Sequence[int]] = None,
                 frozen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The **oracle-imitation** null: N3 fitted on the physics arm's assignments.

    *A physics arm that cannot beat an imitation of itself has no organization
    claim.* ⚠ Banked, and carried verbatim wherever F5 is discussed: the
    agreement measured **0.8888** fitted to the assignments directly but only
    **0.2576** fitted to the read objective — **the 0.89-vs-0.26 gap is an
    optimisation gap in fitting the diagram, NOT evidence of a structurally
    non-VQ channel.**

    ⛔ **R3 (declared rider):** the *trained* organizer is spoke B's, so the
    target here is the **written but un-organized** physics store's own
    assignments (spoke A's `build_arm` + `multi_particle_read`). Declared as
    such, never as "the physics arm".
    """
    from chlu.experiments.exp_c2w11_substrate import build_arm

    seeds = tuple(seeds if seeds is not None else grid.tune_seeds)
    cells = []
    for s in seeds:
        S = seed_setup(cfg, s, n_val=grid.n_val, frozen=frozen)
        fam = S["family"]
        arm = build_arm(cfg, fam, s, phi=S["phi"])
        z_s = multi_particle_read(arm["store"], S["head"], cfg, S["ind_s"], S["key"])
        z_u = multi_particle_read(arm["store"], S["head"], cfg, S["ind_u"],
                                  jax.random.fold_in(S["key"], 1))
        a_s = occupancy(z_s, arm["anchors"])
        a_u = occupancy(z_u, arm["anchors"])
        row = {"seed": int(s)}
        for tag, kw in (("fitted_on_assignments", {"target_assign": a_s}),
                        ("fitted_on_read_objective", {})):
            fit = n3_static_geometric(cfg, fam, arm["anchors"], S["q0_s"],
                                      fam.y_seen, level="csb", lr=1e-2, tau=0.2,
                                      steps=grid.steps * 2, seed=s,
                                      payload_source="fitted", **kw)
            agree_u = float((fit["assign"](S["q0_u"]) == a_u).mean())
            agree_s = float((fit["assign"](S["q0_s"]) == a_s).mean())
            row[tag] = {"agreement_unseen": agree_u, "agreement_seen": agree_s,
                        "native_unseen": _native(
                            _z_native(fit["apply"](S["q0_u"]), cfg),
                            fam.y_unseen, fam.tol)["acc"]}
        row["physics_store_unseen_zero_param"] = float(exact_set_accuracy(
            np.asarray(fam.payloads)[a_u].sum(1), fam.y_unseen, fam.tol))
        cells.append(row)
        print(f"[oracle] seed={s} assign={row['fitted_on_assignments']['agreement_unseen']:.4f} "
              f"read_obj={row['fitted_on_read_objective']['agreement_unseen']:.4f}",
              flush=True)
    res = {"stage": "oracle", "cells": cells,
           "agreement_fitted_on_assignments": _mean_2se(
               [c["fitted_on_assignments"]["agreement_unseen"] for c in cells]),
           "agreement_fitted_on_read_objective": _mean_2se(
               [c["fitted_on_read_objective"]["agreement_unseen"] for c in cells]),
           "banked_C2W5": {"assignments": 0.8888, "read_objective": 0.2576},
           "F5_fires_bar": 0.99,
           "rider_R3": ("the target is the WRITTEN, UN-ORGANIZED physics store "
                        "(the trained organizer is spoke B's)"),
           "carry_this_sentence": (
               "The 0.89-vs-0.26 gap is an optimisation gap in fitting the "
               "diagram, NOT evidence of a structurally non-VQ channel.")}
    if out:
        _dump(res, out / "stage_oracle.json")
    return res


# ==========================================================================
# the runner
# ==========================================================================
def run_c2w11_nulls(project: Optional[str] = None,
                    seeds: Sequence[int] = (0, 1, 2, 3, 4),
                    quick: bool = False, out_dir: Optional[str] = None,
                    stages: Sequence[str] = ALL_STAGES,
                    arms: Sequence[str] = ARMS,
                    frozen_path: Optional[str] = None) -> Dict[str, Any]:
    """⭐ Run order: guards → grid → score → gridmax → v2 → v3 → ceiling → oracle."""
    del project
    out = Path(out_dir) if out_dir else Path("outputs/c2w11_nulls")
    out.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen(frozen_path)
    cfg = c2w11_null_config(frozen)
    grid = C2W11_GRID
    frozen_note = ""
    if quick:
        # ⛔ QUICK IS NOT A CLAIM CELL. It runs a deliberately different family
        # (N_a = 16, K = 32), so the frozen cross-check MUST be disabled rather
        # than "passed" — a guard that fires on a cell nobody claims teaches the
        # harness to run with the guard off.
        frozen_note = ("⛔ quick mode: the frozen cross-check is DISABLED "
                       "because the quick family is deliberately not the claim "
                       "cell. No number from a quick run is quotable.")
        frozen = None
    if quick:
        cfg = replace(cfg, n_wells=16, n_items=32, n_unseen=64, atoms_per_well=6)
        grid = replace(grid, lrs=(1e-2,), tune_seeds=(0,), score_seeds=(0, 1),
                       steps=20, n_val=8, n1_atoms_per_well=(6,), n1_taus=(1.0,),
                       n1_inits=("written",), n2_variants=("kmeans",),
                       n2_codes=(16,), n2_commitments=(0.0,), n2_restarts=2,
                       n3_levels=("b",), n3_payloads=("fitted",), n4_ks=(1,),
                       n4_weights=("idw",), n5_hidden=(8,), n5_momentum=(0.0,),
                       n5_decay=(0.0,), n5_gate=("surprise",), n5_chunk=(1,),
                       n5_passes=1, n5_pretrain_steps=10)
        seeds = tuple(seeds)[:2]
    res: Dict[str, Any] = {"config": cfg.as_dict(),
                           "flags_vs_default": cfg.as_flag_table(),
                           "grid": grid.as_dict(), "quick": bool(quick),
                           "seeds": list(map(int, seeds)),
                           "frozen_loaded": bool(frozen),
                           "frozen_path": (frozen or {}).get("_resolved_path"),
                           "frozen_note": frozen_note,
                           "arms": list(arms)}
    if not quick and not frozen:
        raise FrozenCellMismatch(
            "⛔ a CLAIM cell was requested but FROZEN-INTERFACES-C2W11.json was "
            "not found. Every family constant (tol, chance, the selected width, "
            "the payload radius, the V3 budget grid) is spoke A's and is READ, "
            "never remembered. Set CHLU_C2W11_FROZEN or pass frozen_path.")
    want = set(stages)
    score_seeds = tuple(seeds) if seeds else grid.score_seeds

    for name, fn in (("grid", "stage_grid"), ("score", "stage_score"),
                     ("gridmax", "stage_gridmax"), ("v2", "stage_v2"),
                     ("v3", "stage_v3"), ("ceiling", "stage_ceiling"),
                     ("guards", "stage_guards"), ("oracle", "stage_oracle")):
        p = out / f"{fn}.json"
        if name not in want and p.exists():
            res[name] = json.loads(p.read_text())

    if "guards" in want:
        print("\n=== GUARDS (phi bytes, launch bit-identity, the tol trap) ===",
              flush=True)
        res["guards"] = stage_guards(cfg, seeds=tuple(seeds)[:3], frozen=frozen,
                                     out=out)
    if "grid" in want:
        print("\n=== GRID (the registered F3 budget, COMPUTED) ===", flush=True)
        res["grid_stage"] = stage_grid(cfg, grid, arms=arms, out=out, frozen=frozen)
    selected = (res.get("grid_stage") or res.get("grid") or {}).get("selected")
    if "score" in want:
        if not selected:
            raise RuntimeError("stage_score needs stage_grid's selection")
        print("\n=== SCORE (V1 + L1-L4 + the two-sided ledgers) ===", flush=True)
        res["score"] = stage_score(cfg, grid, selected, arms=arms, out=out,
                                   seeds=score_seeds, frozen=frozen)
    if "gridmax" in want:
        print("\n=== GRIDMAX (null*, over the ENTIRE registered grid) ===",
              flush=True)
        res["gridmax"] = stage_gridmax(cfg, grid, arms=arms, out=out,
                                       seeds=score_seeds, frozen=frozen)
    if "v2" in want:
        if not selected:
            raise RuntimeError("stage_v2 needs stage_grid's selection")
        print("\n=== V2 (the matched confidence channel) ===", flush=True)
        res["v2"] = stage_v2(cfg, grid, selected, arms=arms, out=out,
                             seeds=score_seeds, frozen=frozen)
    if "v3" in want:
        if not selected:
            raise RuntimeError("stage_v3 needs stage_grid's selection")
        print("\n=== V3 (instantiated landscapes, the identical read) ===",
              flush=True)
        static = {}
        for a in ("N4", "N5"):
            sc = res.get("score", {}).get("per_arm", {}).get(a, {})
            if sc:
                static[a] = float(sc["unseen"]["mean"])
        res["v3"] = stage_v3(cfg, grid, selected,
                             arms=tuple(a for a in arms if a in ("N1", "N2", "N3")),
                             out=out, seeds=score_seeds, frozen=frozen,
                             static_scores=static)
    if "ceiling" in want:
        print("\n=== CEILING (out-of-class, recomputed on the new launches) ===",
              flush=True)
        res["ceiling"] = stage_ceiling(cfg, grid, out=out, seeds=score_seeds,
                                       frozen=frozen)
    if "oracle" in want:
        print("\n=== ORACLE-IMITATION (T5.2 rider (i)) ===", flush=True)
        res["oracle"] = stage_oracle(cfg, grid, out=out, frozen=frozen)

    _dump(res, out / "c2w11_nulls_summary.json")
    return res
