"""⭐⭐ Experiment **CAPTURE AT STRONG φ** (C2W8 pass 3, the SPINE) — *does the
physics add anything once the encoder is not the bottleneck?*

**What this is.** The **frozen** C2W8 census
(:func:`chlu.experiments.exp_well_lifecycle.run_census_cell`) plus the
**COMPLETED** gate — **G-CAP · G-DEC · G-DRIFT · G-ADDR** — run on **arm A's
co-scaled-width store** over a **strong-φ** rig: the Split-CIFAR-10 encoders that
`c2w8-cifar-strong-phi` built and priced (``simclr``, ``enc_steps = 8000``; the
unfitted ``randconv`` as the cheap control), read through **wt2's declared
``φ_dim → addr_dim`` projection**, against an **internal ``pca``-φ reference arm
at the same ``addr_dim`` in this same run**.

⛔⛔ **Three things this file does NOT do, by construction.**

* It does **not** touch the census or the G-ADDR instrument. Every cell is
  ``run_census_cell`` called on the unmodified :mod:`chlu.core.well_lifecycle`,
  so all three arms are measured on **one arithmetic**. What is substituted is
  the ``cl_config`` / ``build_phi`` / ``store_config`` triple — the rig's φ and
  the arm's store — inside a ``try/finally``, exactly as ``exp_capture_armA``
  substitutes its store factory.
* It does **not** compare anything to a pass-1 or pass-2 number. Those are
  **MNIST** (Head ruling R1); a cross-run comparison here would be cross-dataset
  **and** cross-encoder **and** cross-checkout at once. The weak-φ baseline is the
  ``pca`` arm **in this run**.
* It does **not** produce a paper number, a tier-ii verdict, a full-CLU verdict,
  or an adjudication of the arm A / arm B race (all declared NOT-RUNs).

**Both branches are pre-registered as REPORTABLE** (``PREREG-C2W8-PASS3`` §6):

* **(a) DAYLIGHT** — measurable separation opens between the settle and its **own
  same-keys launder** once both can address (registered prior Q5 = 0.15);
* **(b) NO DAYLIGHT** — the CIFAR spoke's ``±0.0007``-class result reproduced on
  the census rig ⇒ **the tier-i thesis measured at the CL substrate. A REPORTABLE
  FINDING, never a shortfall to be tuned away** (registered prior Q6 = 0.70).

⚠⚠ **The D2a warning travels** (§A29.6). ``G-DRIFT → 0`` means the settled point
approaches a *deterministic function of the stored key* = **D2a =
table-expressible**, which the configuration intervention §8.2 prohibits and
which is exactly what the CIFAR arm already measured at strong φ. ⛔ No leg,
objective or tuning choice here treats drift → 0 as a target; **G-DRIFT is
reported TWO-SIDED**, beside an explicit :func:`d2a_probe` that measures how
often the settled point resolves to the *same item the same-keys kNN launder
resolves to*. **If the best-scoring cell is also the lowest-drift cell, that
co-occurrence is the D2a signature, not a success**, and
:func:`d2a_cooccurrence` says so in the artifact.

⛔ **The joint dial.** ``(addr_dim, atom budget)`` is ONE dial:
``n_atoms = round(512·√2^d)`` ⇒ ``d = 12`` is **32 768 atoms**. The
geometry-favoured ``d = 16`` is **NOT RUN**: wt2 measured the store **inert**
there (median fitted depth 5.44e-7 at a fully honoured 131 072-atom budget), and
an inert store makes a census vacuous for a reason that is **not** the gate's
reason.

Runnable::

    uv run python -m chlu.experiments.exp_capture_strong_phi --quick
    uv run python -m chlu.experiments.exp_capture_strong_phi --seeds 0,1,2 \
        --arms randconv,simclr,pca --save-dir <dir>
    chlu exp-capture-strong-phi [--project N] [--seeds 0,1,2] [--quick]
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional

import jax
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.clu_system import CluSystemConfig
from chlu.core.well_lifecycle import cue_queries, gate_addr_verdict
from chlu.experiments import exp_capture_armA as armA
from chlu.experiments import exp_well_lifecycle as ewl
from chlu.experiments.exp_cl_entry import apply_cifar10
from chlu.experiments.exp_phi_geometry import (
    ATOM_BASE,
    ATOM_C,
    assert_no_truncation,
    atom_budget,
    stream_fingerprint,
)
from chlu.experiments.phi_encoders import PhiProjection, ProjectedReadIn

#: ⛔ Bound ONCE at import so a substitution can never see (or recurse into) its
#: own replacement, whatever else a caller has patched. Same discipline as
#: ``exp_capture_armA._FROZEN_STORE_CONFIG``.
_FROZEN_CL_CONFIG = ewl.cl_config
_FROZEN_BUILD_PHI = ewl.build_phi
_FROZEN_STORE_CONFIG = ewl.store_config

#: the arms this spine knows how to build. ``pca`` is the INTERNAL weak-φ
#: reference at the same ``addr_dim`` (Head ruling R1), never a banked row.
STRONG_ARMS = ("simclr", "randconv", "convae")


# ---------------------------------------------------------------------------
# the rig: the CL config, the φ, and the declared φ_dim → addr_dim map
# ---------------------------------------------------------------------------
def phi_dim_for(cfg: CHLUConfig, arm: str) -> int:
    """φ's OWN width for ``arm``, before the map.

    The conv arms run at their priced width (256) and are **projected** down; the
    ``pca`` reference is built directly at ``addr_dim`` and mapped by an identity,
    which is provably neutral (PCA-of-PCA-``k`` fit on the same pool *is* PCA-``d``).
    """
    g = cfg.experiment_capture_strong_phi
    return int(g.phi_dim_strong) if str(arm) in STRONG_ARMS else int(g.addr_dim)


def cl_config_for(config: CHLUConfig, arm: str) -> Any:
    """The CL-entry config one cell drives — the substitution for ``ewl.cl_config``.

    ⚠ The **preset trap** (`cl-encoder` §10, pinned in ``tests/``):
    :func:`apply_cifar10` must run **before** the explicit knobs, or a run
    silently executes on a different φ fit pool than the one that was asked for.

    ⚠ The stream depends on **none** of ``(arm, phi_dim)`` — only on the
    dataset/protocol knobs — so it is bit-identical across arms at a seed and
    every arm comparison is **PAIRED** (:func:`stream_fingerprint` is recorded on
    every cell as the evidence).
    """
    g = config.experiment_capture_strong_phi
    c2 = copy.deepcopy(config)
    if str(g.dataset) == "cifar10":
        apply_cifar10(c2)
    cl = c2.experiment_cl_entry
    cl.dataset = str(g.dataset)
    cl.phi_arm = str(arm)
    cl.phi_dim = phi_dim_for(config, arm)
    cl.enc_steps = int(g.enc_steps)
    cl.n_fit_region = int(g.n_fit_region)
    cl.n_fit_pool = int(g.n_fit_pool)
    cl.phi_regimes = [str(g.phi_regime)]
    return cl


def build_projected_phi(regime: str, stream, cl, seed: int, *, addr_dim: int,
                        form: str, sink: Optional[Dict[str, Any]] = None):
    """``φ`` at its own width, composed with the DECLARED ``φ_dim → addr_dim`` map.

    This is the substitution for ``ewl.build_phi``, and it is the whole reason the
    spine can run at all: ``exp_well_lifecycle.PhiAddress`` forces
    ``phi_dim = addr_dim`` and **truncates**, so "strong φ at ``addr_dim``" would
    otherwise be 8-of-256 coordinates rather than the encoder that was priced
    (``PREREG-C2W8-PASS3`` §3, Head ruling R2).

    ⛔ **R2(b) is honoured mechanically, not by intention.** The object returned is
    a :class:`ProjectedReadIn` whose output is already ``addr_dim``-dimensional, so
    ``PhiAddress``'s ``f[:, :addr_dim]`` is the **identity** — and the census hands
    that *same object* to its kNN-in-φ launder through ``embed.keys(...)``. The
    launder therefore **cannot** read the 256-dim φ. :func:`assert_no_truncation`
    (wt2's) raises here if the map ever emits a different width, and
    :func:`launder_audit` re-asserts bit-identity after the cell.

    ⚠ The map is fitted on the **regime's own** pool (``task1_only``), disjoint
    from every stream item and drawn from task-1 classes only — no leakage.
    """
    phi, prov = _FROZEN_BUILD_PHI(regime, stream, cl, seed)
    pool = np.asarray(stream[f"fit_pool_{regime}"], np.float32)
    if int(cl.phi_dim) == int(addr_dim) and str(form) in ("identity", "none"):
        proj = PhiProjection(np.asarray(phi(pool[:1]), np.float32), int(addr_dim),
                             form="identity")
    else:
        proj = PhiProjection(np.asarray(phi(pool), np.float32), int(addr_dim),
                             form=str(form), seed=int(seed))
    phi_proj = ProjectedReadIn(phi, proj)
    assert_no_truncation(phi_proj, int(addr_dim))
    prov = {**prov, "projection": proj.provenance(),
            "phi_param_floats_total": int(phi_proj.param_floats()),
            "phi_own_param_floats": int(phi_proj.phi.param_floats()),
            "map_param_floats": int(proj.param_floats())}
    if sink is not None:
        sink["phi"] = phi_proj
        sink["phi_provenance"] = prov
    return phi_proj, prov


def strong_store_config(cfg: CHLUConfig, seed: int, d_safe: float,
                        overrides: Optional[Dict[str, Any]] = None) -> CluSystemConfig:
    """⭐ **Arm A's CO-SCALED-WIDTH store, declared explicitly rather than inherited.**

    ⚠⚠ The banked pass-2 arm-A census ran at ``atom_width_frac_spacing = 1.5``,
    passed on the CLI — but the shipped :class:`ExperimentCaptureArmAConfig`
    **default is 0.5** (the pilot cell), which `c2w8p3-gate-addr` measured **does
    not clear the pass-2 gate**. A spine that merely imported arm A's factory
    would silently score a different store. So the width fraction (and the kernel
    axis) come from **this** experiment's own config and are written into a copy
    of ``experiment_capture_arm_a`` before arm A's factory is called.

    ⭐ The width is co-scaled to the **MEASURED** key spacing of this seed's own
    run — ``d_safe / d_safe_frac``, recovered inside the cell — never a hardcoded
    number, and never MNIST's spacing.
    """
    g = cfg.experiment_capture_strong_phi
    c2 = copy.deepcopy(cfg)
    a = c2.experiment_capture_arm_a
    a.atom_width_frac_spacing = (None if g.atom_width_frac_spacing is None
                                 else float(g.atom_width_frac_spacing))
    a.atom_kernel = str(g.atom_kernel)
    a.atom_kernel_cutoff = float(g.atom_kernel_cutoff)
    a.site_local_init = bool(g.site_local_init)
    a.site_local_radius_frac = float(g.site_local_radius_frac)
    return armA.arm_store_config(c2, seed, d_safe, overrides=overrides)


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------
def run_cell(cfg: CHLUConfig, seed: int, arm: str, *, data=None,
             verbose: bool = True) -> Dict[str, Any]:
    """One frozen census cell at ``(arm, seed)`` on the strong-φ rig.

    The three substitutions (``cl_config`` → this dataset/φ-width; ``build_phi``
    → φ composed with the declared map; ``store_config`` → arm A's co-scaled
    width) are scoped by ``try/finally`` so nothing leaks into a later cell or a
    later arm. **The census itself is untouched.**
    """
    g = cfg.experiment_capture_strong_phi
    d = int(cfg.experiment_well_lifecycle.addr_dim)
    form = "identity" if phi_dim_for(cfg, arm) == d else str(g.projection)
    sink: Dict[str, Any] = {}
    captured: Dict[str, Any] = {}

    def _post_build(system, **kw):
        # the system object is mutated in place by the census, so holding the
        # reference here gives the FINAL store for the additive D2a probe below.
        captured["system"] = system
        captured["embed"] = kw.get("embed")
        captured["stream"] = kw.get("stream")

    orig = (ewl.cl_config, ewl.build_phi, ewl.store_config)
    ewl.cl_config = lambda c: cl_config_for(c, arm)
    ewl.build_phi = lambda regime, stream, cl, s: build_projected_phi(
        regime, stream, cl, s, addr_dim=d, form=form, sink=sink)
    ewl.store_config = lambda c, s, ds, overrides=None: strong_store_config(
        c, s, ds, overrides=overrides)
    try:
        cell = ewl.run_census_cell(cfg, seed, data=data, verbose=verbose,
                                   post_build=_post_build)
    finally:
        ewl.cl_config, ewl.build_phi, ewl.store_config = orig

    phi = sink["phi"]
    cell["arm"] = str(arm)
    cell["phi_dim_own"] = phi_dim_for(cfg, arm)
    cell["projection_form"] = form
    cell["stream_fingerprint"] = (stream_fingerprint(captured["stream"])
                                  if captured.get("stream") is not None else None)
    cell["launder_audit"] = launder_audit(phi, captured.get("embed"),
                                          captured.get("stream"), d)
    cell["bytes_with_phi"] = byte_ledger(cell, phi, d)
    cell["d2a"] = (d2a_probe(captured["system"], cell, cfg, seed)
                   if bool(g.d2a_probe) and captured.get("system") is not None
                   else {"status": "NOT RUN — d2a_probe = False (declared, not a null)"})
    return cell


def launder_audit(phi_proj, embed, stream, addr_dim: int) -> Dict[str, Any]:
    """⛔ **Head ruling R2(b), asserted — the launder reads the PROJECTED φ.**

    The census feeds its :class:`RingBufferKNN` launder from ``embed.keys(x)``,
    which is ``phi(x)[:, :addr_dim] * scale`` on the *same* ``phi`` object the
    store addresses through. This re-derives that array from the projected φ and
    asserts **bit-identity**, raising rather than reporting a soft flag: a launder
    reading 256 dims while the store reads ``d`` is not a launder, it is a
    handicap match (fairness invariant §A4.3).
    """
    if embed is None or stream is None:
        return {"status": "NOT RUN — no embed captured", "checked": False}
    x = np.asarray(stream["train_X"][0][: min(32, len(stream["train_X"][0]))],
                   np.float32)
    keys = np.asarray(embed.keys(x))
    # the same arithmetic PhiAddress.keys performs, on the SAME projected φ —
    # dtypes included, so "bit-identical" means bit-identical and not "close".
    direct = (np.asarray(phi_proj(x), np.float32)[:, : int(addr_dim)]
              * embed.scale)
    if keys.shape[1] != int(addr_dim):
        raise AssertionError(
            f"the census launder reads {keys.shape[1]}-dim keys while the store "
            f"addresses {addr_dim}: that is a handicap match, not a launder "
            f"(Head ruling R2(b))"
        )
    if not np.array_equal(keys, direct):
        raise AssertionError(
            "the census launder's keys are not bit-identical to the projected φ: "
            "R2(b) is asserted in code, never merely intended"
        )
    return {
        "checked": True,
        "launder_key_dim": int(keys.shape[1]),
        "store_address_dim": int(addr_dim),
        "launder_reads_projected_phi": True,
        "bit_identical_to_store_addresses": True,
        "phi_dim_before_map": int(phi_proj.projection.in_dim),
        "rule": ("R2(b): the launder reads the SAME projected φ as the store; a "
                 "256-dim launder against a d-dim store is a handicap match"),
    }


def byte_ledger(cell: Dict[str, Any], phi_proj, addr_dim: int) -> Dict[str, Any]:
    """The census ledger **plus the φ term, on EVERY arm including the launder**.

    ⛔ §A4.3 / Head ruling R2: the map's floats ride on the ledger of every arm
    that reads through it. The store and the launder read the *same* object, so
    the φ term is the **same number** on both rows — which is precisely why the
    matched-**bytes** ratio below is not improved by the encoder.

    ⚠ **Every quotation states matched-ITEMS vs matched-BYTES.** These cells are
    matched-items (same keys, same queries, same φ); matched-bytes is **NOT met**,
    and pass 1's **1 253×** stream-launder byte-ratio caveat travels unchanged.
    """
    b = dict(cell["bytes"])
    floats_phi = int(phi_proj.phi.param_floats())
    floats_map = int(phi_proj.projection.param_floats())
    phi_bytes = 4 * (floats_phi + floats_map)
    n_atoms = int(cell["flags"]["n_atoms"])
    b.update({
        "phi_param_floats": floats_phi,
        "map_param_floats": floats_map,
        "phi_total_param_floats": floats_phi + floats_map,
        "phi_total_bytes": int(phi_bytes),
        "clu_total_bytes_with_phi": int(b["clu_total_bytes"] + phi_bytes),
        "knn_launder_bytes_with_phi": int(b["knn_launder_bytes"] + phi_bytes),
        "ratio_clu_over_knn_launder": float(
            b["clu_total_bytes"] / max(b["knn_launder_bytes"], 1)),
        "ratio_clu_over_knn_launder_with_phi": float(
            (b["clu_total_bytes"] + phi_bytes) / max(b["knn_launder_bytes"] + phi_bytes, 1)),
        "joint_dial_(d, atom_budget)": {
            "addr_dim": int(addr_dim), "n_atoms": n_atoms,
            "rule": f"n_atoms >= round({ATOM_BASE} * {ATOM_C:.6f}**d)",
            "priced_budget": int(atom_budget(int(addr_dim))),
            "budget_honoured": bool(n_atoms >= int(atom_budget(int(addr_dim)))),
            "note": "ONE dial: d and the atom budget are never quoted apart",
        },
        "matching": "matched-ITEMS (same keys, same queries, same φ)",
        "matched_bytes": False,
        "caveat": ("pass 1's 1253x stream-launder byte ratio travels; ⛔ no "
                   "performance claim is made at any of these ratios"),
        "gate_reads_bytes": False,
    })
    return b


# ---------------------------------------------------------------------------
# ⭐ the D2a diagnostic (§A29.6) — reported TWO-SIDED, never a target
# ---------------------------------------------------------------------------
def d2a_probe(system, cell: Dict[str, Any], cfg: CHLUConfig,
              seed: int) -> Dict[str, Any]:
    """⚠⚠ *Is the settled point becoming a deterministic function of the key?*

    ``G-DRIFT → 0`` means the settled point approaches a deterministic function of
    the stored key — **D2a, table-expressible**, which the configuration
    intervention §8.2 prohibits, and exactly what the CIFAR arm already measured
    at strong φ (settle = same-keys kNN to ``±0.0007``). ⛔ **This is a
    diagnostic, not an objective**; nothing in this file optimises it in either
    direction.

    On **G-ADDR's own cue set** (same ``seed``/``kappa_q``/``n_query_per_item``, so
    the queries are bit-identical to the ones the gate scored) it measures:

    * ``agreement_rate`` — how often the store's settled point resolves to the
      **same item** the same-keys kNN-in-φ launder resolves to. → 1.0 is the
      table-expressible limit;
    * ``median_settle_to_launder_key_over_spacing`` — the distance from the settle
      to the launder's chosen key, as a **dimensionless ratio** of the measured
      key spacing (``PREREG-C2W8-PASS3`` §4: every geometric quantity is a ratio
      with the scale stated). → 0 is the same limit, measured in space rather than
      in labels.
    """
    w = cfg.experiment_well_lifecycle
    ids, centers, pays = system.codebook()
    n = int(len(ids))
    if n == 0:
        return {"status": "NOT RUN — empty codebook", "n_items": 0}
    centers = np.asarray(centers, float)
    spacing = float(cell["geometry"]["median_nn_task1"])
    q0, tgt, _ = cue_queries(centers, int(system.store.dim), spacing=spacing,
                             kappa_q=float(w.gaddr_kappa_q),
                             n_per_item=int(w.gaddr_n_query_per_item),
                             seed=int(seed))
    res = system.read(q0)
    q_star = np.asarray(res.state.q_star, float)
    a_star = q_star[:, : int(system.store.addr_dim)]
    store_resolved = np.argmin(
        np.linalg.norm(a_star[:, None, :] - centers[None, :, :], axis=-1), axis=1)
    launder_resolved = np.argmin(
        np.linalg.norm(q0[:, None, : int(system.store.addr_dim)] - centers[None, :, :],
                       axis=-1), axis=1)
    agree = float(np.mean(store_resolved == launder_resolved))
    dist = np.linalg.norm(a_star - centers[launder_resolved], axis=-1)
    return {
        "status": "MEASURED — two-sided diagnostic, ⛔ never a target (§A29.6)",
        "n_items": n, "n_queries": int(tgt.size),
        "cue_set": "bit-identical to G-ADDR's (same seed, kappa_q, n_query_per_item)",
        "agreement_rate": agree,
        "agreement_chance": float(1.0 / n),
        "median_settle_to_launder_key_over_spacing": float(
            np.median(dist) / spacing) if spacing > 0 else float("nan"),
        "spacing_ref": spacing,
        "reading": ("agreement -> 1.0 AND drift -> 0 is the D2a signature "
                    "(settle = a deterministic function of the stored key = "
                    "TABLE-EXPRESSIBLE, intervention §8.2 prohibits it as a "
                    "target); agreement near chance means the settle is doing "
                    "something the table does not"),
    }


def d2a_cooccurrence(legs: List[Dict[str, Any]], gaddr: List[Dict[str, Any]]
                     ) -> Dict[str, Any]:
    """⚠ **Does the best-scoring cell coincide with the lowest-drift cell?**

    The pass-2 gate as written **REWARDS** the degenerate configuration: a store
    whose settled point collapses onto the stored key scores a perfect G-DRIFT.
    So if the best-scoring seed is also the lowest-drift seed, that co-occurrence
    is the **D2a signature, not a success**, and it is said prominently rather
    than left for a reader to notice.
    """
    if not legs:
        return {"status": "NOT RUN — no cells"}
    if len(legs) < 2:
        return {"status": ("NOT INFORMATIVE — one cell cannot co-occur with "
                           "itself (declared, not a null)"),
                "n_cells": len(legs), "best_is_also_lowest_drift": None}
    score = [int(bool(lg["G_CAP"]["pass"])) + int(bool(lg["G_DEC"]["pass"]))
             + int(bool(lg["G_DRIFT"]["pass"]))
             + int(bool(g.get("gate_addr_pass", False))) for lg, g in zip(legs, gaddr, strict=True)]
    drift = [float(lg["G_DRIFT"]["ratio"]) for lg in legs]
    a1 = [float(g["A1"]["correct_basin_rate"]) if isinstance(g.get("A1"), dict)
          else float("nan") for g in gaddr]
    best = int(np.argmax(score)) if len(set(score)) > 1 else int(np.nanargmax(a1))
    low = int(np.argmin(drift))
    return {
        "best_scoring_index": best, "lowest_drift_index": low,
        "legs_passed_by_index": score,
        "A1_by_index": a1,
        "G_DRIFT_ratio_by_index": drift,
        "best_is_also_lowest_drift": bool(best == low),
        "warning": ("⛔ the pass-2 gate REWARDS the degenerate configuration: if "
                    "the best-scoring cell is also the lowest-drift cell that "
                    "co-occurrence is the D2a signature, NOT a success (§A29.6)"),
    }


# ---------------------------------------------------------------------------
# ⭐ the branch — computed mechanically, never argued
# ---------------------------------------------------------------------------
def daylight_verdict(gaddr: List[Dict[str, Any]], *, min_seeds: int = 3
                     ) -> Dict[str, Any]:
    """⭐ **Branch (a) DAYLIGHT or branch (b) NO DAYLIGHT** — the registered reading.

    Both branches are pre-registered as reportable (``PREREG-C2W8-PASS3`` §6);
    **branch (b) is a FINDING, not a shortfall.**

    Registered rule, computed here and never argued:

    > **(a) DAYLIGHT** iff a launder margin is **POSITIVE beyond 2 SE on
    > ``min_seeds`` seeds** — on the cue protocol (``A3a``, McNemar SE) or on the
    > held-out stream (``A3b``, pooled binomial SE). **(b) NO DAYLIGHT**
    > otherwise.

    ⚠ Losing to the launder on the **cue** protocol is the metric-native-ceiling
    theorem, not news: the cue draws ``c_i + σ·ε`` with equal priors, under which
    1-NN over the stored keys **is** the Bayes rule. That is why the *gate* leg
    asks "does not lose", and why the branch question — *is there daylight ABOVE
    it?* — is asked separately and two-sided.
    """
    rows = []
    for g in gaddr:
        if not isinstance(g.get("A3"), dict):
            continue
        a3 = g["A3"]
        a3a = float(a3["A3a_cue_margin"])
        se_a = float(a3.get("A3a_se_paired", 0.0) or 0.0)
        a3b = a3.get("A3b_stream_margin")
        se_b = float(a3.get("A3b_se_pooled") or 0.0)
        rows.append({
            "A3a_cue_margin": a3a, "A3a_2se": 2.0 * se_a,
            "A3a_strict_margin": float(a3.get("A3a_strict_margin", float("nan"))),
            "A3a_store_rate": float(a3.get("A3a_store_rate", float("nan"))),
            "A3a_launder_rate": float(a3.get("A3a_launder_rate", float("nan"))),
            "A3a_positive_beyond_2se": bool(a3a > 2.0 * se_a),
            "A3b_stream_margin": (None if a3b is None else float(a3b)),
            "A3b_2se": 2.0 * se_b,
            "A3b_applicable": bool(a3.get("A3b_applicable", False)),
            "A3b_positive_beyond_2se": bool(a3b is not None and float(a3b) > 2.0 * se_b),
        })
    n_a = sum(r["A3a_positive_beyond_2se"] for r in rows)
    n_b = sum(r["A3b_positive_beyond_2se"] for r in rows)
    daylight = bool(len(rows) >= int(min_seeds)
                    and (n_a >= int(min_seeds) or n_b >= int(min_seeds)))
    return {
        "branch": "(a) DAYLIGHT" if daylight else "(b) NO DAYLIGHT",
        "daylight": daylight,
        "n_seeds": len(rows), "min_seeds": int(min_seeds),
        "n_seeds_A3a_positive_beyond_2se": int(n_a),
        "n_seeds_A3b_positive_beyond_2se": int(n_b),
        "rows_by_seed": rows,
        "rule": ("(a) iff a launder margin is POSITIVE beyond 2 SE on >= "
                 f"{int(min_seeds)} seeds (A3a McNemar / A3b pooled binomial); "
                 "(b) otherwise"),
        "prereg": {
            "both_branches_registered_reportable": True,
            "Q5_prior_daylight": 0.15,
            "Q6_prior_no_daylight": 0.70,
            "status_of_branch_b": ("A REPORTABLE FINDING — the tier-i thesis "
                                   "measured at the CL substrate. ⛔ NOT a "
                                   "shortfall, ⛔ never to be tuned away"),
        },
        "not_a_verdict": ("⛔ neither branch is a tier-ii verdict (no organizer "
                          "swap exists here), a full-CLU verdict (§A28.4), or an "
                          "arm-race adjudication; ⛔ no paper number"),
    }


# ---------------------------------------------------------------------------
# the arm
# ---------------------------------------------------------------------------
def completed_gate(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The **COMPLETED** gate: G-CAP · G-DEC · G-DRIFT (arm A's, unmodified) plus
    **G-ADDR** (wt1's, straight off the census cell). All four legs, per seed."""
    legs = [armA.gate_legs(c) for c in cells]
    gaddr = [c["g_addr"] for c in cells]
    all_pass = [bool(lg["G_CAP"]["pass"] and lg["G_DEC"]["pass"]
                     and lg["G_DRIFT"]["pass"] and g.get("gate_addr_pass", False))
                for lg, g in zip(legs, gaddr, strict=True)]
    return {
        "legs_by_seed": legs,
        "g_addr_by_seed": gaddr,
        "G_CAP_pass_seeds": int(sum(lg["G_CAP"]["pass"] for lg in legs)),
        "G_DEC_pass_seeds": int(sum(lg["G_DEC"]["pass"] for lg in legs)),
        "G_DRIFT_pass_seeds": int(sum(lg["G_DRIFT"]["pass"] for lg in legs)),
        "G_ADDR": gate_addr_verdict(gaddr, min_seeds=max(len(cells), 1)),
        "all_four_same_seed": int(sum(all_pass)),
        "n_seeds": len(cells),
        "gate_pass": bool(len(cells) >= 3 and all(all_pass)),
        "rule": ("COMPLETED gate = the pass-2 three (PREREG-C2W8-PASS2 §3) AND "
                 "G-ADDR (PREREG-C2W8-PASS3 §2), every leg on every seed, >= 3 seeds"),
        "d_drift_warning": ("⛔ G-DRIFT is TWO-SIDED: drift -> 0 is D2a = "
                            "table-expressible (§A29.6), NEVER a target"),
    }


def run_arm(cfg: CHLUConfig, arm: str, seeds: List[int], *, data=None,
            verbose: bool = True) -> Dict[str, Any]:
    cells = []
    for s in seeds:
        if verbose:
            print(f"[strong-phi:{arm}] census seed {s} "
                  f"(d={cfg.experiment_well_lifecycle.addr_dim}) ...", flush=True)
        cells.append(run_cell(cfg, s, arm, data=data, verbose=verbose))
    gate = completed_gate(cells)
    return {
        "arm": str(arm),
        "role": ("PRIMARY strong φ" if arm == cfg.experiment_capture_strong_phi.phi_arm_strong
                 else "CONTROL (unfitted encoder)" if arm == cfg.experiment_capture_strong_phi.phi_arm_control
                 else "⛔ INTERNAL weak-φ REFERENCE at the SAME d, in THIS run"),
        "seeds": [int(s) for s in seeds],
        "phi_dim_own": phi_dim_for(cfg, arm),
        "projection_form": cells[0]["projection_form"] if cells else None,
        "gate": gate,
        "daylight": daylight_verdict(gate["g_addr_by_seed"],
                                     min_seeds=max(len(cells), 1)),
        "d2a": {
            "by_seed": [c["d2a"] for c in cells],
            "cooccurrence": d2a_cooccurrence(gate["legs_by_seed"],
                                             gate["g_addr_by_seed"]),
        },
        "bytes_by_seed": [c["bytes_with_phi"] for c in cells],
        "launder_audit_by_seed": [c["launder_audit"] for c in cells],
        "geometry_by_seed": [c["geometry"] for c in cells],
        "self_probe_by_seed": [c["self_probe"] for c in cells],
        "depth_raw_median_by_seed": [c["census"]["depth_raw_median"] for c in cells],
        "store_inert_by_seed": [bool(float(c["census"]["depth_raw_median"]) < 1e-6)
                                for c in cells],
        "own_foreign_by_seed": [armA.own_foreign(c) for c in cells],
        "stream_by_seed": [c["stream"] for c in cells],
        "phi_provenance_by_seed": [c["phi_provenance"] for c in cells],
        "stream_fingerprint_by_seed": [c["stream_fingerprint"] for c in cells],
        "store_flags_by_seed": [c["flags"]["clu_system_non_defaults"] for c in cells],
        "cells": cells,
        "wall_s": float(sum(c["wall_s"] for c in cells)),
    }


# ---------------------------------------------------------------------------
# the deliverable
# ---------------------------------------------------------------------------
def apply_quick(cfg: CHLUConfig) -> None:
    """Smoke mode: a real census + a real gate on a tiny stream, tiny φ, tiny store.

    ⛔ Never a claim cell — the atom budget alone (``round(512·√2^d)``) is what a
    real ``d`` costs, and quick mode does not pay it.
    """
    ewl.apply_quick(cfg)
    g = cfg.experiment_capture_strong_phi
    g.quick = True
    g.seeds = [0]
    g.addr_dim = int(cfg.experiment_well_lifecycle.addr_dim)
    g.phi_dim_strong = 16
    g.enc_steps = 20
    g.n_fit_region = 600
    g.n_fit_pool = 200
    g.d2a_probe = True


def run_experiment_capture_strong_phi(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "plots",
    seeds: Optional[List[int]] = None,
    arms: Optional[List[str]] = None,
    quick: bool = False,
    data=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """The spine: the frozen census + the completed gate, per arm, per seed."""
    cfg = config or get_default_config()
    g = cfg.experiment_capture_strong_phi
    if quick:
        apply_quick(cfg)
    # ⭐ the joint dial, written into the census's own knobs BEFORE any cell
    w = cfg.experiment_well_lifecycle
    w.dataset = str(g.dataset)
    w.addr_dim = int(g.addr_dim)
    w.run_gate_addr = True
    if quick:
        g.addr_dim = int(w.addr_dim)

    seeds = [int(s) for s in (seeds if seeds is not None else g.seeds)]
    arms = [str(a) for a in (arms if arms is not None else g.arms)]
    t0 = time.time()

    out_arms: Dict[str, Any] = {}
    for a in arms:
        out_arms[a] = run_arm(cfg, a, seeds, data=data, verbose=verbose)

    results = {
        "experiment": "capture_strong_phi",
        "wave": "C2W8 pass 3 — THE SPINE",
        "question": ("does the physics add anything once the encoder is not the "
                     "bottleneck?"),
        "dial_declaration": {
            "dial": ("none as a new claim — a COMPONENT BUILD measuring whether "
                     "the physics adds anything once the encoder is not the "
                     "bottleneck"),
            "laundering_control": ("same-keys kNN-in-φ launder in the SAME "
                                   "projected φ on every cell, with the byte "
                                   "ledger beside it (φ params + projection "
                                   "params on EVERY arm including the launder); "
                                   "matched-ITEMS, not matched-bytes; the 1253x "
                                   "caveat travels"),
            "falsifies": ("nothing — BOTH branches are pre-registered as "
                          "reportable; what would be non-compliant is tuning "
                          "away branch (b)"),
            "does_not_falsify": ("losing to the kNN launder on the metric-native "
                                 "cue protocol (1-NN is the Bayes rule there — "
                                 "the metric-native-ceiling theorem, not news)"),
            "no_paper_number": True,
            "no_tier_ii_verdict": True,
            "no_full_clu_verdict": "§A28.4",
            "no_arm_race_adjudication": "§A30.1 — the race stays VOID/unadjudicated",
            "depth_is_not_feature_importance": "§A23.5 ACTIVE",
        },
        "substrate_ruling": {
            "R1": ("SPLIT-CIFAR-10 (reduced protocol). ⛔ pass-1/pass-2 census "
                   "numbers are MNIST and are NOT the baseline; the weak-φ "
                   "comparison is the INTERNAL pca arm at the same d in THIS run"),
            "R2": "the projection is wt2's and the launder reads the PROJECTED φ",
            "R3": ("RESOLVED BY MEASUREMENT: d = 12. The geometry-favoured d = 16 "
                   "is INERT (wt2: median depth 5.44e-7 at a fully honoured "
                   "131072-atom budget) and is a declared NOT-RUN"),
            "R4": "geometry_go re-labels, it does not block",
        },
        "joint_dial_d_atom_budget": {
            "addr_dim": int(w.addr_dim),
            "priced_atom_budget": int(atom_budget(int(w.addr_dim))),
            "rule": f"n_atoms >= round({ATOM_BASE} * sqrt(2)**d)",
            "note": "ONE dial — d and its atom budget are never quoted apart",
        },
        "seeds": seeds,
        "arms": out_arms,
        "arm_roles": {a: out_arms[a]["role"] for a in out_arms},
        "branch_by_arm": {a: out_arms[a]["daylight"]["branch"] for a in out_arms},
        "gate_pass_by_arm": {a: out_arms[a]["gate"]["gate_pass"] for a in out_arms},
        "store_inert_by_arm": {a: out_arms[a]["store_inert_by_seed"] for a in out_arms},
        "declared_not_runs": [
            "d = 16 (the geometry-favoured dimension): the store is INERT there "
            "(wt2, censused cell) — a declared NOT-RUN, never a null",
            "merge / prune / restoration verbs and every §2.7 claim cell — "
            "still deferred (no population; monitor #3 defect open)",
            "the arm A vs arm B race — VOID as a comparison, stays UNADJUDICATED",
            "the tier-ii organizer swap; any full-CLU or I2 verdict",
            "generic_frozen φ (the leaking reference regime); convae",
            "MNIST — excluded by Head ruling R1, not merely unrun",
            "any paper number, any performance claim",
        ],
        "flags": _flag_table(cfg, seeds, arms),
        "wall_s": float(time.time() - t0),
    }

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "capture_strong_phi.json")
    with open(path, "w") as f:
        json.dump(ewl._jsonable(results), f, indent=2)
    results["json_path"] = path
    if verbose:
        print(f"[strong-phi] wrote {path} ({results['wall_s']:.0f}s)", flush=True)
        for a, arm_res in out_arms.items():
            print(f"  {a}: gate_pass={arm_res['gate']['gate_pass']} "
                  f"branch={arm_res['daylight']['branch']} "
                  f"A1={arm_res['gate']['G_ADDR']['A1_by_seed']}", flush=True)
    return results


def _flag_table(cfg: CHLUConfig, seeds: List[int], arms: List[str]) -> Dict[str, Any]:
    """The mandatory flag-provenance table (protocol §5) — every non-default knob."""
    g = cfg.experiment_capture_strong_phi
    w = cfg.experiment_well_lifecycle
    return {
        "experiment_capture_strong_phi": dataclasses.asdict(g),
        "experiment_well_lifecycle": dataclasses.asdict(w),
        "arms_run": arms, "seeds": seeds,
        "store_arm": ("arm A's CO-SCALED WIDTH "
                      f"(atom_width_frac_spacing={g.atom_width_frac_spacing}, "
                      f"kernel={g.atom_kernel} — ⭐ the BANKED census value, NOT "
                      "the shipped default 0.5, which does not clear the pass-2 "
                      "gate; kernel form is a DECLARED SECONDARY axis)"),
        "jax": str(jax.__version__),
    }


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--arms", type=str, default=None)
    p.add_argument("--addr-dim", type=int, default=None)
    p.add_argument("--save-dir", type=str, default="plots")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    cfg = get_default_config()
    if a.addr_dim is not None:
        cfg.experiment_capture_strong_phi.addr_dim = int(a.addr_dim)
    run_experiment_capture_strong_phi(
        config=cfg, save_dir=a.save_dir,
        seeds=[int(s) for s in a.seeds.split(",")] if a.seeds else None,
        arms=[s.strip() for s in a.arms.split(",")] if a.arms else None,
        quick=bool(a.quick),
    )


if __name__ == "__main__":
    main()
