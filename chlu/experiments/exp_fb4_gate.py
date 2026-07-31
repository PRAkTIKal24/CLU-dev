"""⭐ Experiment **FB4-GATE** — is the B′ protocol measuring the memory, or the task?

`PREREG-Bprime.md` §6, falsifier **FB4**: *"the +0 B substitute is at ceiling for
every family **including full attention** ⇒ the protocol measures the task, not
the memory"* — and *"run this first: it is cheap and it validates the protocol
before it is spent on six families."* ``bprime-rivals`` is gated on this verdict.

Five arms per family, **on identical ``phi``, with ``phi``-bytes ledgered**:

===============  =========================================================  ==================
arm              what it is                                                 bytes
===============  =========================================================  ==================
``full``         the shipped CLU read                                       full ledger
``launder``      settle-deleted / matched-bytes table launder               table
``substitute``   the family's strongest **+0 B** reader of the same table   table **+0 B**
``blank``        ``blank_store_control`` — the family's floor               —
``attention``    ⭐ a **full-attention** reader of the same table            table **+ 4 B**
===============  =========================================================  ==================

⭐ **The attention arm is the point of FB4.** It is the strongest metric-native
reader anyone would reach for; if it is *also* at ceiling everywhere, the families
are not discriminating readers at all. It is a **table** reader — it never sees a
trajectory, it is **not** ``AttentionPsi``, and it inherits none of that
quarantine.

The verdict is **computed, not argued**: :mod:`chlu.eval.fb4_gate` holds the
pre-registered rule (``S(f) ≥ 0.95`` and the 2-SE attention leg), filed before
this module was ever run.

⛔ **No cell here is a byte-matched dividend** — the minimum ratio measured
anywhere in C2W1/C2W3 is **17.11×**, and matched bytes is unreachable by
construction under a masked write (``ratio = 1.4·atoms_per_item + 0.8``).

Runnable directly::

    PYTHONPATH=. python -m chlu.experiments.exp_fb4_gate [--quick]

⚠ **Ownership (C2W3).** This module and :mod:`chlu.eval.fb4_gate` are
``bprime-fb4-gate``'s. It *imports* the gym (:mod:`chlu.experiments.memory_gym`,
:mod:`chlu.experiments.exp_memory_gym`) and ``chlu.eval.dividend`` and edits
neither — the write/read/query/score path is therefore the **shipped** one,
byte-for-byte, which is also what makes the shared arms reproduce the on-disk
C2W1 artefact digit-for-digit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.eval.dividend import byte_account, dividend
from chlu.eval.fb4_gate import (
    ATTENTION_TEMPERATURE_BYTES,
    METRIC_MAX,
    attention_read,
    family_saturated,
    fb4_verdict,
    fit_attention_temperature,
)
from chlu.experiments.exp_memory_gym import (
    _build_queries,
    _clu_predictions,
    _insertion_order,
    _launder_predictions,
)
from chlu.experiments.memory_gym import (
    PRIMARY_METRIC,
    GymConfig,
    QuerySet,
    gym_config,
    make_gym_stream,
    readout_settled,
    restrict_to_pair,
    score,
)

#: ⭐ The FB4 plan: four families, **3 seeds**, at the SHIPPED anchor.
#: ``overload`` runs at ``load1x_shipped`` (the 478× cell) — reconciliation 6:
#: ``overload`` at the base atom budget is **unusable** (0/18 admissible,
#: *including* the Gaussian control), so the base-budget cell is a declared
#: NOT-RUN here and is never reported as a null.
DEFAULT_PLAN: Tuple[Tuple[str, str], ...] = (
    ("overload", "load1x_shipped"),
    ("aggregate", "base"),
    ("recency", "base"),
    ("manifold", "base"),
)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)

#: ⛔ C2W2 D4: the recency family's shipped default grades a **K-way** answer on a
#: **2-way** curve (the CLU's answer falls outside its own pair 19.4 % of the
#: time). The pre-fix ``0.3019 ± 0.0679`` is a scoring-domain **defect** and is
#: never-quote as a null. FB4 runs the family **post-fix**, and emits both
#: coverages so the switch is auditable.
FAMILY_GYM_OVERRIDES: Dict[str, dict] = {"recency": {"restrict_index_to_pair": True}}

#: ``--quick`` shrinks the store and the stream, never the code path.
QUICK_GYM = dict(n_offer=6, capacity=6, budget=6, reference_capacity=3,
                 n_query_per_item=2, n_query_per_pair=3, consolidate_every=2,
                 min_consolidations=4, n_manifold_launch=4)
QUICK_CLU = dict(write_steps=30, address_steps=80, read_steps=120,
                 n_query_per_item=2, quick=True, atoms_per_item=32)


# --------------------------------------------------------------------------
# the identical-phi invariant, for a gym whose phi is the identity launch
# --------------------------------------------------------------------------
def phi_ledger_row(qs: QuerySet) -> Dict[str, Any]:
    """``phi_id`` / ``phi_bytes`` for the gym's read-in, on **every** arm.

    The gym's ``phi`` is the **identity launch** ``_launch(ccfg, addr)``: the
    address block copied into ``q0``, payload block zeroed. It carries **no
    learnable parameters**, so ``phi_bytes = 0`` — but it is still ledgered on
    every arm, because "identical ``phi`` and identical ``phi``-bytes on every
    arm" is the fairness invariant (enforced in code as ``assert_identical_phi``
    / ``PhiMismatchError`` for the parametric ``phi`` families) and an arm that
    quietly consumed a different read-in would confound every number.

    Here the invariant is *structural*: all five arms consume **the same
    ``QuerySet`` object**, so the row hashes its ``q0``/``keys`` and every arm
    reports that hash.
    """
    h = hashlib.sha256()
    for a in (np.asarray(qs.q0, dtype=np.float64), np.asarray(qs.keys, dtype=np.float64)):
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return {"phi_family": "identity_launch", "phi_id": h.hexdigest()[:16],
            "phi_params": 0, "phi_bytes": 0,
            "note": ("all five arms consume the SAME QuerySet object; phi_id is a "
                     "content hash of (q0, keys)")}


def _assert_identical_phi_rows(rows: Dict[str, Dict[str, Any]]) -> str:
    """Every arm's ``phi`` row must be identical — raise, never warn."""
    from chlu.core.psi_readout import PhiMismatchError

    ids = {k: (v["phi_id"], v["phi_bytes"]) for k, v in rows.items()}
    uniq = sorted(set(ids.values()))
    if len(uniq) != 1:
        raise PhiMismatchError(
            "identical-phi invariant VIOLATED across FB4 arms: "
            + "; ".join(f"{k}: {v}" for k, v in sorted(ids.items())))
    return uniq[0][0]


# --------------------------------------------------------------------------
# the +0 B substitute set, per family
# --------------------------------------------------------------------------
def order_aware_pair_launder(born: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """⭐ The recency family's strongest **+0 B** substitute *after the D4 fix*.

    Of the query's **own two candidates** — the same pair information
    ``restrict_index_to_pair`` hands the CLU arm — return the one inserted
    **later**. A table's row order already encodes insertion order, so this costs
    **nothing**, and once every arm answers the same 2-way question this reader
    answers it **exactly, by construction**.

    That is intervention §8.3 in its purest form and it is exactly what FB4 is
    built to detect: it is reported as a *capability of the table*, never as a
    dividend against it.
    """
    pr = np.asarray(pairs, dtype=int)
    b = np.asarray(born, dtype=float).reshape(-1)
    take = b[pr]
    pick = np.argmax(take, axis=1)
    return np.asarray(np.take_along_axis(pr, pick[:, None], axis=1).ravel())


def zero_byte_candidates(family: str, launder_scores: Dict[str, float]) -> List[str]:
    """The declared **+0 B** reader set per family (PREREG §1.1(c)).

    ⭐ **Includes ``settle_deleted`` itself**: the ``substitute`` arm is *a
    different reader of the same table at the same bytes*, and FB4's own text
    ("the +0 B substitute is at ceiling ... including full attention") is a
    statement about **any** zero-extra-byte reader of the launder's table. The
    exclusive variant (``sub_excl_launder``) is reported beside it as the
    declared secondary. Including the launder makes saturation *easier* to
    reach — it is the choice that cannot flatter the program.
    """
    names = ["settle_deleted"]
    names += [k for k in launder_scores if k.endswith("+0B")]
    return [n for n in names if n in launder_scores]


# --------------------------------------------------------------------------
# one FB4 cell
# --------------------------------------------------------------------------
def run_fb4_cell(family: str, arm: str = "base", seed: int = 0,
                 gym_overrides: Optional[dict] = None, quick: bool = False,
                 loud: bool = False) -> dict:
    """Run one ``family/arm@seed`` cell and emit all five arms + the byte ledger.

    The write / read / query / score path is the **shipped gym's**, imported
    unchanged; only the two extra readers (attention, and the pair-restricted
    order-aware substitute) are new, and both are pure table readers.
    """
    import jax

    over = dict(FAMILY_GYM_OVERRIDES.get(family, {}))
    over.update(dict(gym_overrides or {}))
    if quick:
        qk = dict(QUICK_GYM)
        clu = dict(QUICK_CLU)
        clu.update(dict(over.pop("clu_overrides", {})))
        qk.update(over)
        over = dict(qk, clu_overrides=clu)
    gcfg: GymConfig = gym_config(family, arm, seed=seed, **over)
    ccfg: CluSystemConfig = gcfg.build_clu()
    label = f"{family}/{arm}@s{seed}"
    metric = PRIMARY_METRIC[family]
    t0 = time.time()

    # --- the shipped write stream, with consolidation windows ---------------
    system = build_system(ccfg, key=jax.random.PRNGKey(seed), loud=loud)
    stream = make_gym_stream(gcfg, ccfg)
    key = jax.random.PRNGKey(seed + 1)
    admitted: List[int] = []
    prev = 0
    for b in stream.chunks:
        if b > prev:
            key, k_w = jax.random.split(key)
            wrep = system.write_stream(stream.items[prev:b], key=k_w)
            admitted += wrep.admitted
            prev = b
        system.consolidate()
    write_s = time.time() - t0

    ids, centers, pays = system.codebook()
    order_map = _insertion_order(system)
    born = np.asarray([order_map.get(int(i), -1) for i in ids], dtype=float)
    if int(len(ids)) < 2:
        return {"cell": label, "family": family, "arm": arm, "seed": seed,
                "degenerate": True, "n_live": int(len(ids)),
                "note": "fewer than 2 live items cannot support ANY reported metric"}

    # --- the queries (the SAME rng stream the gym uses) ---------------------
    rng = np.random.default_rng(seed + 7717)
    qs = _build_queries(gcfg, ccfg, stream, system, centers, pays, born, rng)
    sep = float(system.certificates().get("sep", float("nan")))
    t1 = time.time()
    res = system.read(qs.q0)
    read_s = time.time() - t1
    restrict_pairs = (np.asarray(qs.meta.get("pairs"))
                      if (gcfg.restrict_index_to_pair and qs.kind == "index"
                          and qs.meta.get("pairs") is not None) else None)

    # --- arm 1: full (the shipped CLU read) ---------------------------------
    preds = _clu_predictions(gcfg, ccfg, res, centers, sep, restrict_pairs)
    occupancy = preds.pop("_occupancy", None)
    full = float(score(qs, preds["clu"])[metric])

    # --- arms 2/3: the launder and every +0 B reader of its table ------------
    launders = _launder_predictions(qs, centers, pays, born, rng, None, restrict_pairs)
    if family == "recency" and qs.meta.get("pairs") is not None:
        launders["order_aware_pair_+0B"] = order_aware_pair_launder(
            born, np.asarray(qs.meta["pairs"], dtype=int))
    lnd_scores = {k: float(score(qs, p)[metric]) for k, p in launders.items()}
    launder = float(lnd_scores["settle_deleted"])
    cands = zero_byte_candidates(family, lnd_scores)
    sub_name = max(cands, key=lambda n: lnd_scores[n])
    sub = float(lnd_scores[sub_name])
    excl = [n for n in cands if n != "settle_deleted"]
    sub_excl_name = max(excl, key=lambda n: lnd_scores[n]) if excl else ""
    sub_excl = float(lnd_scores[sub_excl_name]) if excl else float("nan")

    # --- arm 4: the blank store (the family's floor) ------------------------
    blank_sys = build_system(replace(ccfg, seed=ccfg.seed + 991),
                             key=jax.random.PRNGKey(seed + 991), loud=False)
    b_res = blank_sys.read(qs.q0)
    b_preds = _clu_predictions(gcfg, ccfg, b_res, centers, sep, restrict_pairs)
    b_preds.pop("_occupancy", None)
    blank = float(score(qs, b_preds["clu"])[metric])

    # --- arm 5: ⭐ ATTENTION over the launder's OWN (key, payload) table ------
    #  values: the column the family's question is asked of. For `manifold` the
    #  table stores ONE point per item and the spectator coordinate it stored is
    #  the written one (zero) — exactly what `_launder_predictions` gives the
    #  frozen launder — so the attention read of it is a convex combination of
    #  zeros. That is the table's honest answer, not a handicap.
    values = (np.zeros((centers.shape[0], 1), dtype=float) if qs.kind == "coord"
              else np.asarray(pays, dtype=float))
    rng_fit = np.random.default_rng(seed + 20260731)
    qs_fit = _build_queries(gcfg, ccfg, stream, system, centers, pays, born, rng_fit)
    fit_pairs = (np.asarray(qs_fit.meta.get("pairs"))
                 if (gcfg.restrict_index_to_pair and qs_fit.kind == "index"
                     and qs_fit.meta.get("pairs") is not None) else None)
    tfit = fit_attention_temperature(
        centers, values, qs_fit.keys,
        lambda p: float(score(qs_fit, p)[metric]),
        kind=qs.kind, pairs=fit_pairs)
    attn_pred = attention_read(centers, values, qs.keys, temperature=tfit["tau"],
                               kind=qs.kind, pairs=restrict_pairs)
    attn = float(score(qs, attn_pred)[metric])

    # --- the two-sided byte ledger, on every arm ----------------------------
    ba = byte_account(system, centers, pays)
    phi_row = phi_ledger_row(qs)
    table_bytes = int(ba.launder_bytes)
    ledger = {
        "full_bytes": int(ba.full_bytes), "launder_bytes": table_bytes,
        "ratio": float(ba.ratio), "matched": bool(ba.matched()),
        "n_atoms": int(system.store.V.learned.centers.shape[0]),
        "atoms_per_live_item": float(system.store.V.learned.centers.shape[0]
                                     / max(len(ids), 1)),
        "breakdown": dict(ba.breakdown),
        "per_arm": {
            "full": {"bytes": int(ba.full_bytes), **phi_row},
            "launder": {"bytes": table_bytes, **phi_row},
            "substitute": {"bytes": table_bytes, "extra_bytes": 0, **phi_row},
            "blank": {"bytes": 0, **phi_row},
            "attention": {"bytes": table_bytes + ATTENTION_TEMPERATURE_BYTES,
                          "extra_bytes": ATTENTION_TEMPERATURE_BYTES, **phi_row},
        },
        "note": ("⛔ NOT a byte-matched dividend: the minimum ratio measured "
                 "anywhere is 17.11×, and `ratio = 1.4·atoms_per_item + 0.8` makes "
                 "matched bytes unreachable by construction under a masked write."),
    }
    phi_id = _assert_identical_phi_rows(
        {k: v for k, v in ledger["per_arm"].items()})

    rec: Dict[str, Any] = {
        "cell": label, "family": family, "arm": arm, "seed": seed,
        "degenerate": False, "metric": metric, "metric_max": METRIC_MAX[metric],
        "n_live": int(len(ids)), "n_queries": int(len(qs)), "sep": sep,
        "sep_over_sigma_q": float(sep / max(ccfg.query_sigma, 1e-12)),
        "arms": {"full": full, "launder": launder, "substitute": sub,
                 "blank": blank, "attention": attn},
        "substitute_name": sub_name,
        "substitute_excl_launder": {"name": sub_excl_name, "score": sub_excl},
        "all_zero_byte_readers": {n: lnd_scores[n] for n in cands},
        "all_launder_scores": lnd_scores,
        "attention_fit": {k: v for k, v in tfit.items() if k != "curve"},
        "attention_curve": tfit["curve"],
        "dividend_vs_launder": dividend(full, launder, metric=metric).as_dict(),
        "byte_ledger": ledger, "phi_id": phi_id,
        "gym_config_non_default": gcfg.as_flag_table(),
        "clu_config_non_default": ccfg.as_flag_table(),
        "n_admitted": len(admitted),
        "wall_write_s": write_s, "wall_read_s": read_s,
    }

    # --- the recency switch, made auditable (both coverages) ----------------
    if family == "recency" and occupancy is not None and qs.meta.get("pairs") is not None:
        pairs = np.asarray(qs.meta["pairs"], dtype=int)
        unres = np.argmax(occupancy, axis=1)
        inside = np.mean([int(u) in set(p.tolist()) for u, p in zip(unres, pairs,
                                                                   strict=True)])
        q_star = np.asarray(res.state.q_star)[:, : centers.shape[1]]
        neg_d = -np.linalg.norm(q_star[:, None, :] - centers[None, :, :], axis=-1)
        rec["recency_coverage"] = {
            "restrict_index_to_pair": bool(gcfg.restrict_index_to_pair),
            "acc_unrestricted_K_way(traj)": float(score(qs, unres)[metric]),
            "acc_restricted_2_way(traj)": full,
            "acc_restricted_2_way(point)": float(
                score(qs, restrict_to_pair(neg_d, pairs))[metric]),
            "coverage_answer_inside_own_pair": float(inside),
            "out_of_pair_rate": float(1.0 - inside),
            "note": ("⛔ the PRE-FIX unrestricted number is a SCORING-DOMAIN "
                     "DEFECT (a K-way answer graded on a 2-way curve) and is "
                     "never-quote as a null; both are emitted so the switch is "
                     "auditable."),
        }
    if family == "overload":
        # the auxiliary decode of the blank store, for continuity with C1W1
        b_val = readout_settled(b_res, ccfg)
        b_dec = np.argmin(np.linalg.norm(b_val[:, None, :] - pays[None, :, :], axis=-1),
                          axis=1)
        rec["blank_aux_decode_labels_argmin"] = float(
            np.mean(b_dec == np.argmin(np.linalg.norm(
                qs.keys[:, None, :] - centers[None, :, :], axis=-1), axis=1)))
    return rec


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
def gate(records: Sequence[dict]) -> Dict[str, Any]:
    """Apply the **pre-registered** D0.1 rule to the finished cells."""
    by_fam: Dict[str, List[dict]] = {}
    for r in records:
        if not r.get("degenerate"):
            by_fam.setdefault(r["family"], []).append(r)
    rows = []
    for fam in ("overload", "aggregate", "recency", "manifold"):
        cells = sorted(by_fam.get(fam, []), key=lambda c: c["seed"])
        if not cells:
            continue
        names = {c["substitute_name"] for c in cells}
        rows.append(family_saturated(
            fam, cells[0]["metric"],
            sub_seeds=[c["arms"]["substitute"] for c in cells],
            attn_seeds=[c["arms"]["attention"] for c in cells],
            blank_seeds=[c["arms"]["blank"] for c in cells],
            sub_name="/".join(sorted(names)),
            detail={
                "arm": cells[0]["arm"], "seeds": [c["seed"] for c in cells],
                "full_seeds": [c["arms"]["full"] for c in cells],
                "launder_seeds": [c["arms"]["launder"] for c in cells],
                "byte_ratio_seeds": [c["byte_ledger"]["ratio"] for c in cells],
                "attention_tau": [c["attention_fit"]["tau"] for c in cells],
                "attention_tau_degenerate": [c["attention_fit"]["degenerate"]
                                             for c in cells],
                "sub_excl_launder_seeds": [c["substitute_excl_launder"]["score"]
                                           for c in cells],
                "sub_excl_launder_name": cells[0]["substitute_excl_launder"]["name"],
            }))
    out = fb4_verdict(rows)
    # the declared SECONDARY: the same rule with the launder excluded from the
    # +0 B reader set (PREREG §1.1(c)) — reported beside the primary, never instead
    sec_rows = []
    for fam in ("overload", "aggregate", "recency", "manifold"):
        cells = sorted(by_fam.get(fam, []), key=lambda c: c["seed"])
        if not cells or not np.all([np.isfinite(c["substitute_excl_launder"]["score"])
                                    for c in cells]):
            continue
        sec_rows.append(family_saturated(
            fam, cells[0]["metric"],
            sub_seeds=[c["substitute_excl_launder"]["score"] for c in cells],
            attn_seeds=[c["arms"]["attention"] for c in cells],
            blank_seeds=[c["arms"]["blank"] for c in cells],
            sub_name=cells[0]["substitute_excl_launder"]["name"]))
    out["secondary_excl_launder"] = fb4_verdict(sec_rows)
    return out


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------
def run_experiment_fb4_gate(config=None, save_dir: str = "results",
                            models_dir: Optional[str] = None,
                            seed: Optional[int] = None,
                            families: Optional[Sequence[str]] = None,
                            seeds: Optional[Sequence[int]] = None,
                            quick: bool = False,
                            overrides: Optional[dict] = None) -> dict:
    """Run the FB4 plan and write the verdict artifact."""
    os.makedirs(save_dir, exist_ok=True)
    base_seed = int(seed if seed is not None else getattr(
        getattr(config, "project", None), "seed", 0) or 0)
    plan = [(f, a) for f, a in DEFAULT_PLAN
            if (not families or f in set(families))]
    cell_seeds = tuple(int(s) for s in (seeds if seeds else DEFAULT_SEEDS))
    if base_seed and not seeds:
        cell_seeds = tuple(s + base_seed for s in cell_seeds)

    results: Dict[str, Any] = {
        "experiment": "fb4_gate",
        "question": ("FB4: is the +0 B substitute at ceiling for every family, "
                     "INCLUDING full attention? If so the protocol measures the "
                     "task, not the memory."),
        "plan": [{"family": f, "arm": a, "seeds": list(cell_seeds)} for f, a in plan],
        "quick": bool(quick),
        "sd_convention": "sample sd (ddof=1); SE = sd/sqrt(n); n = 3",
        "prereg": (".claude/outputs/bprime-fb4-gate/PREREG.md — the rule, its two "
                   "constants and the predicted S(f) were filed BEFORE this ran"),
        "byte_note": ("⛔ no cell here is a byte-matched dividend; the minimum "
                      "ratio measured anywhere is 17.11×"),
        "cells": [],
    }
    # the metrics artifact sits WITH the figure (the gym's sibling-``results``
    # convention exists for project runs; ``--save-dir`` is used directly here so
    # a report's artifacts are one directory).
    results_dir = (os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
                   if os.path.basename(os.path.abspath(save_dir)) == "plots"
                   else save_dir)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_fb4_gate_metrics.json")

    def _dump():
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=_json_default)

    for fam, arm in plan:
        for s in cell_seeds:
            t0 = time.time()
            try:
                rec = run_fb4_cell(fam, arm, int(s), quick=quick,
                                   gym_overrides=overrides)
            except Exception as exc:  # a failed cell is reported, never silent
                import traceback

                traceback.print_exc()
                rec = {"cell": f"{fam}/{arm}@s{s}", "family": fam, "arm": arm,
                       "seed": int(s), "degenerate": True, "error": repr(exc)}
            rec["wall_s"] = time.time() - t0
            results["cells"].append(rec)
            a = rec.get("arms", {})
            print(f"[{rec['cell']}] metric={rec.get('metric')} "
                  f"full={a.get('full')} launder={a.get('launder')} "
                  f"sub={a.get('substitute')} ({rec.get('substitute_name')}) "
                  f"attn={a.get('attention')} blank={a.get('blank')} "
                  f"ratio={rec.get('byte_ledger', {}).get('ratio')}x "
                  f"({rec['wall_s']:.0f}s)")
            _dump()

    results["gate"] = gate(results["cells"])
    print("\n⭐ FB4 VERDICT: " + results["gate"]["verdict"])
    print("   surviving families: " + ", ".join(results["gate"]["surviving_families"]))
    try:
        results["figures"] = _plot(results, save_dir)
    except Exception as exc:  # pragma: no cover - figures are not the result
        results["figures"] = []
        results["figure_error"] = repr(exc)
    _dump()
    results["metrics_path"] = out_path
    return results


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _plot(results: dict, save_dir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["gate"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    x = np.arange(len(rows))
    names = ["blank", "launder", "sub", "attn", "full"]
    w = 0.16
    for i, n in enumerate(names):
        if n in ("launder", "full"):
            vals = [float(np.mean(r["detail"][f"{n}_seeds"])) for r in rows]
        else:
            vals = [r[n] for r in rows]
        ax.bar(x + i * w, vals, w, label=n)
    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels([r["family"] for r in rows], fontsize=8)
    ax.set_ylabel("family primary metric")
    ax.set_title("FB4: every arm on identical phi, same scorer", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    S = [r["S"] for r in rows]
    ax.bar(x, S, 0.6, color=["tab:red" if s >= 0.95 else "tab:green" for s in S])
    ax.axhline(0.95, color="k", ls="--", lw=1, label="pre-registered 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels([r["family"] for r in rows], fontsize=8)
    ax.set_ylabel("S(f) = (sub - blank)/(M - blank)")
    ax.set_title("saturation — red = struck as protocol-invalid\n"
                 + results["gate"]["verdict"], fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(save_dir, "exp_fb4_gate.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment FB4-GATE: is the B′ protocol measuring the memory?")
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Base seed offset")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--families", nargs="+", help="Run only these families")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the seed list")
    parser.add_argument("--save-dir", default="results", help="Where figures go")
    args = parser.parse_args()

    config = None
    save_dir = args.save_dir
    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        save_dir = str(pm.get_paths(args.project)["plots"])
    res = run_experiment_fb4_gate(config=config, save_dir=save_dir, seed=args.seed,
                                  families=args.families, seeds=args.seeds,
                                  quick=args.quick)
    print(json.dumps(res["gate"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
