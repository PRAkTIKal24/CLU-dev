"""⭐⭐ Experiment **B′-RIVALS** — the audit paper's spine.

    *"When does test-time dynamics buy anything over a table at matched bytes?"* —
    one protocol (matched-byte table launder + two-sided byte ledger + a **+0 B**
    substitute audit + same-keys null + blank store), applied **uniformly** to the
    CLU and to the modern neural-memory family.

Charter ADDENDUM 3 §A14.2 / §A15 task 1, `PREREG-Bprime.md` §2. This module builds
the **rival rows**; the CLU column is **banked** (`PREREG-Bprime.md` §7) and is
reproduced here only as a fidelity check, never re-derived.

**The family set is RULED and is not re-opened here** (FB4 returned ◐ PARTIAL):

* ``aggregate@base`` — ``S(f) = 0.5068``, the **sole reader-discrimination
  family**. Every dividend number in this experiment is on this family.
* ``overload@load1x_shipped`` — ⛔ **BYTE-FRONTIER COLUMN ONLY**, labelled at every
  appearance, whose defensibility is the declared secondary reading
  ``S_excl = 0.6500``. Never a dividend family, never a headline.
* ``recency`` / ``manifold`` — ⛔ **NOT RUN** (substitute-saturated at +0 B ⇒
  protocol-invalid).

⚠ **The thinness is owned, in writing:** two rival families audited against **one**
surviving synthetic family is a **thin cross-family audit**. That sentence belongs
in the paper's Limitations verbatim, not softened.

⛔ **No cell here is a byte-matched dividend for the CLU** — the minimum ratio
measured anywhere is **17.11×**, and the corrected byte law
``ratio = [A(D+2) + d]/(d+m)`` makes matched bytes unreachable by construction
under a masked write (floor **2.20×**, **2.40×** at ``n_spectator = 1``).
⭐ The rivals *can* be byte-matched to their own tables, and that asymmetry is one
of the audit's findings rather than a nuisance.

Runnable directly::

    PYTHONPATH=. python -m chlu.experiments.exp_bprime_rivals [--quick]

⚠ **Ownership (C2W4).** This module and :mod:`chlu.eval.rivals` are
``bprime-rivals``'s. It *imports* the gym (:mod:`chlu.experiments.memory_gym`,
:mod:`chlu.experiments.exp_memory_gym`) and edits neither — ⛔ ``memory_gym.py``'s
live ``byte_ratio_law`` bug is ``harness-debt``'s this wave, so where this module's
ledger and the gym's printed law disagree (any ``n_spectator > 0`` cell) **the
corrected law is used here and the disagreement is reported, not fixed.**
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.eval.dividend import assert_ledger_identity, byte_account
from chlu.eval.rivals import (
    RIVALS,
    FitExample,
    assert_identical_phi,
    clu_two_sided_ledger,
    fit_best_of_grid,
    matched_table_rows,
    phi_row,
    rival_arms,
    table_ledger,
)
from chlu.eval.rivals.deltanet import metric_native_verdict as delta_verdict
from chlu.eval.rivals.fit import DEFAULT_BUDGET_FLOATS, LR_GRID
from chlu.eval.rivals.ttt import measured_state_floats
from chlu.eval.rivals.ttt import metric_native_verdict as ttt_verdict
from chlu.experiments.exp_memory_gym import (
    _build_queries,
    _clu_predictions,
    _insertion_order,
    _launder_predictions,
)
from chlu.experiments.memory_gym import (
    PRIMARY_METRIC,
    GymConfig,
    GymStream,
    QuerySet,
    gym_config,
    make_gym_stream,
    score,
)

#: ⭐ The ruled family set (§A14.2). ``overload`` is a **frontier column**, not a
#: dividend family, and carries its label at every appearance.
DEFAULT_PLAN: Tuple[Tuple[str, str], ...] = (
    ("aggregate", "base"),
    ("overload", "load1x_shipped"),
)
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)

#: ⛔ The label that must travel with every ``overload`` number.
FRONTIER_LABEL = ("BYTE-FRONTIER COLUMN — not a dividend family; declared "
                  "secondary reading S_excl = 0.6500 (arg-min launder excluded "
                  "from the +0 B reader set)")

#: The head-width sweep for the frontier column (declared, filed in the PREREG).
FRONTIER_HEADS: Tuple[int, ...] = (2, 4, 8, 16, 36)
FRONTIER_RIVALS: Tuple[str, ...] = ("ttt_linear", "gdn2")

#: ⭐ **Banked** (`PREREG-Bprime.md` §7) — quoted, **never re-measured**. Source:
#: `bprime-fb4-gate.md` §A2/§A3 (which itself reproduced the on-disk C2W1 artefact
#: digit-for-digit on every shared arm).
BANKED_CLU: Dict[str, Dict[str, Any]] = {
    "aggregate/base": {
        "metric": "neg_mae",
        "full": [-0.682608, -0.384693, -0.511032],
        "launder": [-0.496261, -0.413103, -0.432255],
        "blank": [-0.438906, -0.404201, -0.423079],
        "best_zero_byte_substitute": {"name": "knn2_idw_+0B", "mean": -0.2081},
        "S": 0.5068,
        "byte_ledger": {"full_bytes": 5456, "launder_bytes": 100, "ratio": 54.56},
    },
    "overload/load1x_shipped": {
        "metric": "decode",
        "full": [1.000000, 0.958333, 0.958333],
        "launder": [1.0, 1.0, 1.0],
        "blank": [0.166667, 0.166667, 0.166667],
        "best_zero_byte_substitute": {"name": "knn2_idw_+0B",
                                      "seeds": [0.9167, 0.5833, 0.625]},
        "S": 1.0000, "S_excl": 0.6500,
        "byte_ledger": {"full_bytes": 57384, "launder_bytes": 120, "ratio": 478.2},
        "accuracy_vs_bytes_curve": {"decode": [0.972, 0.097],
                                    "ratio": [478.0, 2.28],
                                    "note": "banked C1W1/C2W1 curve — NOT re-measured"},
    },
}

#: ``--quick`` shrinks the store, the stream and the outer loop, never the code path.
QUICK_GYM = dict(n_offer=6, capacity=6, budget=6, reference_capacity=3,
                 n_query_per_item=2, n_query_per_pair=3, consolidate_every=2,
                 min_consolidations=4)
QUICK_CLU = dict(write_steps=30, address_steps=80, read_steps=120,
                 n_query_per_item=2, quick=True, atoms_per_item=32)
QUICK_FIT = dict(steps=40, lrs=(3.16e-3,))


# --------------------------------------------------------------------------
# the write stream, as a rival sees it
# --------------------------------------------------------------------------
def stream_tokens(stream: GymStream, ccfg: CluSystemConfig
                  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """The gym's write stream as ``(T, dim)`` tokens ``[address | payload | 0]``.

    ⚠ **Declared protocol asymmetry, in the rivals' favour:** the stream's
    **delete** row is *skipped* — no rival family in §A14.2 has a deletion verb, so
    a rival is never asked to forget, while the CLU is. Stated here rather than
    left for a referee. The collision offer and the revisit **are** kept: they are
    ordinary tokens.
    """
    rows, n_del = [], 0
    for it in stream.items:
        if it.get("delete"):
            n_del += 1
            continue
        z = np.zeros((ccfg.dim,), dtype=np.float32)
        a = np.asarray(it["address"], dtype=float)[: ccfg.addr_dim]
        z[: ccfg.addr_dim] = a
        z[ccfg.addr_dim: ccfg.addr_dim + ccfg.payload_dim] = float(it["payload"])
        rows.append(z)
    xs = np.stack(rows) if rows else np.zeros((0, ccfg.dim), dtype=np.float32)
    return xs, np.ones((xs.shape[0],), dtype=np.float32), {
        "n_tokens": int(xs.shape[0]), "n_delete_rows_skipped": int(n_del),
        "note": ("delete rows are skipped for the rival arms (no rival family has "
                 "a deletion verb); this is an asymmetry IN THE RIVALS' FAVOUR"),
    }


def aux_fit_examples(family: str, arm: str, seed: int, ccfg: CluSystemConfig,
                     gcfg: GymConfig, *, n_streams: int = 2,
                     offset: int = 101) -> Tuple[List[FitExample], Dict[str, Any]]:
    """⭐ **F2a's binding guard**: fit streams built from *different* seeds.

    Different farthest-point sites, different designed payloads, the same query
    law. The rival's outer parameters therefore cannot memorise the eval cell's
    items — which is what makes the state-byte axis (F2) mean anything. No
    published rival baseline runs this guard; we do, and we say so.
    """
    out: List[FitExample] = []
    used = []
    for i in range(int(n_streams)):
        s = int(seed) + int(offset) + i
        used.append(s)
        g2 = replace(gcfg, seed=s)
        st = make_gym_stream(g2, ccfg)
        xs, mask, _ = stream_tokens(st, ccfg)
        centers = st.addresses
        pays = st.payloads
        rng = np.random.default_rng(s + 7717)
        born = np.asarray(st.order, dtype=float)
        qs = _build_queries(g2, ccfg, st, None, centers, pays, born, rng)
        out.append(FitExample(xs=xs, mask=mask,
                              xq=np.asarray(qs.q0, dtype=np.float32),
                              target=np.asarray(qs.target, dtype=np.float32
                                                ).reshape(len(qs), -1)))
    # vmap over examples needs a common shape; trim to the shortest draw
    t = min(e.xs.shape[0] for e in out)
    n = min(e.xq.shape[0] for e in out)
    out = [FitExample(xs=e.xs[:t], mask=e.mask[:t], xq=e.xq[:n], target=e.target[:n])
           for e in out]
    return out, {"fit_stream_seeds": used, "n_tokens": int(t), "n_queries": int(n),
                 "guard": "F2a — outer parameters never see the eval cell's items"}


# --------------------------------------------------------------------------
# one (family, arm, seed) cell: the CLU side, then every rival
# --------------------------------------------------------------------------
def run_rivals_cell(family: str, arm: str = "base", seed: int = 0, *,
                    rivals: Sequence[str] = RIVALS, quick: bool = False,
                    budget_floats: Optional[int] = None,
                    fit_kwargs: Optional[dict] = None,
                    loud: bool = False) -> dict:
    """Run the shipped CLU write/read path, then audit every rival on the SAME
    queries, the same ``phi`` and the same scorer."""
    import jax

    over: Dict[str, Any] = {}
    if quick:
        qk = dict(QUICK_GYM)
        clu = dict(QUICK_CLU)
        qk["clu_overrides"] = clu
        over = qk
    gcfg: GymConfig = gym_config(family, arm, seed=seed, **over)
    ccfg: CluSystemConfig = gcfg.build_clu()
    label = f"{family}/{arm}@s{seed}"
    metric = PRIMARY_METRIC[family]
    t0 = time.time()

    # --- the shipped CLU write stream (unmodified path) ---------------------
    system = build_system(ccfg, key=jax.random.PRNGKey(seed), loud=loud)
    v_before = system.store.V
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

    rng = np.random.default_rng(seed + 7717)
    qs: QuerySet = _build_queries(gcfg, ccfg, stream, system, centers, pays, born, rng)
    sep = float(system.certificates().get("sep", float("nan")))
    res = system.read(qs.q0)

    # --- the CLU's own arms: a FIDELITY CHECK against the banked column ------
    preds = _clu_predictions(gcfg, ccfg, res, centers, sep, None)
    preds.pop("_occupancy", None)
    clu_full = float(score(qs, preds["clu"])[metric])
    launders = _launder_predictions(qs, centers, pays, born, rng, None, None)
    clu_lnd = {k: float(score(qs, p)[metric]) for k, p in launders.items()}
    blank_sys = build_system(replace(ccfg, seed=ccfg.seed + 991),
                             key=jax.random.PRNGKey(seed + 991), loud=False)
    b_preds = _clu_predictions(gcfg, ccfg, blank_sys.read(qs.q0), centers, sep, None)
    b_preds.pop("_occupancy", None)
    clu_blank = float(score(qs, b_preds["clu"])[metric])

    # --- the CLU's ledger: T1's identity as integers (D7) + the SAME
    #     learned-initial-state rule the rivals are held to -------------------
    ba = byte_account(system, centers, pays)
    identity = assert_ledger_identity(system, centers, pays, account=ba)
    clu_ledger = clu_two_sided_ledger(v_before, system.store.V, int(centers.shape[0]),
                                      int(ccfg.addr_dim))

    # ⭐ the raw-space **constant predictor**: ignore the query, return the mean of
    # the stored payloads. +0 B (a function of the table), and the control the
    # partial-input / trivial-baseline tradition demands. It is scored by the
    # family's own scorer like every other arm.
    raw_mean_pred = np.broadcast_to(np.asarray(pays, dtype=float).mean(
        axis=0, keepdims=True), (len(qs), int(pays.shape[1])))
    raw_mean_score = (float(score(qs, raw_mean_pred)[metric])
                      if qs.kind == "value" else float("nan"))
    clu_lnd["raw_table_mean_+0B"] = raw_mean_score

    # --- the rivals ---------------------------------------------------------
    xs, mask, tok_note = stream_tokens(stream, ccfg)
    fkw = dict(fit_kwargs or {})
    if quick:
        fkw = dict(QUICK_FIT, **fkw)
    examples, fit_note = aux_fit_examples(family, arm, seed, ccfg, gcfg)
    budget = int(budget_floats if budget_floats is not None else DEFAULT_BUDGET_FLOATS)
    d_in, m = int(ccfg.dim), int(ccfg.payload_dim)
    phi = phi_row(qs.q0, qs.keys)

    rows: Dict[str, Any] = {}
    for name in rivals:
        t1 = time.time()
        # ⚠ deterministic per-rival key offset. ⛔ NEVER Python's ``hash(name)``:
        # string hashing is salted per process (``PYTHONHASHSEED``), which makes
        # the whole run irreproducible at a fixed seed. Caught in review.
        k_fit = jax.random.PRNGKey(seed * 1000 + 7 * (RIVALS.index(name) + 1))
        model, fit_rec = fit_best_of_grid(
            name, d_in, m, examples, key=k_fit, budget_floats=budget,
            lrs=fkw.get("lrs", LR_GRID), steps=int(fkw.get("steps", 400)),
            b_grid=fkw.get("b_grid"))
        st_floats = int(model.declared_state_floats())
        n_rows = matched_table_rows(st_floats, model.d_k, model.d_v)
        arms_pred = rival_arms(model, xs, mask, np.asarray(qs.q0, dtype=np.float32),
                               rng=np.random.default_rng(seed + 31337),
                               n_rows=n_rows)
        sc = {k: float(score(qs, v)[metric]) for k, v in arms_pred.items()}
        zero_byte = {k: v for k, v in sc.items() if k.endswith("+0B")}
        best_zb = max(zero_byte, key=lambda n: zero_byte[n]) if zero_byte else ""
        # FB4's own convention, for uniformity: the +0 B reader set INCLUDES the
        # arg-min launder (a different reader of the same table at the same bytes).
        incl = dict(zero_byte, settle_deleted=sc["launder"])
        best_incl = max(incl, key=lambda n: incl[n])
        # ⭐ THE STRONGER CONTROL A REFEREE WILL ASK FOR. The registered P5
        # construction reads the table through the memory's OWN projections, which
        # are trained for the recurrence and not for a table. So the same state
        # budget is ALSO spent on the gym's raw ``(address, payload)`` rows and read
        # by the same +0 B readers in the raw metric — i.e. literally the CLU's own
        # launder set, on the same queries. It is strictly stronger and it cannot
        # be accused of handicapping the control.
        raw_rows = int(st_floats // max(int(ccfg.addr_dim + ccfg.payload_dim), 1))
        raw_cands = {k: v for k, v in clu_lnd.items()
                     if k.endswith("+0B") or k == "settle_deleted"}
        best_raw = max(raw_cands, key=lambda n: raw_cands[n]) if raw_cands else ""
        state_final = model.write(xs, mask)
        moved = measured_state_floats(model.init_state(), state_final)
        led = model.ledger(moved=moved)
        tab = table_ledger(n_rows, model.d_k, model.d_v,
                           param_floats=led.param_floats,
                           param_breakdown=led.param_breakdown)
        verdict = (ttt_verdict("linear" if name == "ttt_linear" else "mlp")
                   if name.startswith("ttt") else delta_verdict(name))
        rows[name] = {
            "rival": name, "d_head": int(model.d_head),
            "arms": sc,
            "dividend_vs_own_table": float(sc["full"] - sc["launder"]),
            "zero_byte_margin": {
                "name": best_zb,
                "value": (float(zero_byte[best_zb]) if best_zb else float("nan")),
                "signed_margin_full_minus_sub": (float(sc["full"] - zero_byte[best_zb])
                                                 if best_zb else float("nan")),
                "incl_argmin_name": best_incl,
                "incl_argmin_value": float(incl[best_incl]),
                "signed_margin_incl_argmin": float(sc["full"] - incl[best_incl]),
                "convention": ("primary = the EXCLUSIVE +0 B reader set (as "
                               "registered in PREREG R5); the inclusive set adds "
                               "the arg-min launder, FB4's own convention"),
            },
            "raw_table_control": {
                "note": ("the SAME state budget spent on the gym's raw "
                         "(address, payload) rows and read by the same +0 B "
                         "readers in the raw metric — strictly stronger than the "
                         "registered projected-table launder, and identical to "
                         "the CLU's own launder set on these queries"),
                "rows_affordable": raw_rows,
                "table_is_lossless": bool(raw_rows >= int(len(ids))),
                "best_reader": best_raw,
                "best_reader_score": (float(raw_cands[best_raw]) if best_raw
                                      else float("nan")),
                "signed_margin_full_minus_raw": (
                    float(sc["full"] - raw_cands[best_raw]) if best_raw
                    else float("nan")),
                "all_raw_readers": dict(raw_cands),
            },
            "byte_ledger": {
                "rival": led.as_dict(), "matched_table": tab.as_dict(),
                "state_floats_declared": st_floats,
                "state_floats_measured_moved": int(moved),
                "table_rows_affordable": int(n_rows),
                "table_rows_used": int(min(n_rows, xs.shape[0])),
                "table_is_lossless": bool(n_rows >= xs.shape[0]),
                "state_over_own_table_bytes": (
                    float(led.state_bytes) / max(float(tab.state_bytes), 1.0)),
                "state_bytes_vs_clu_full_bytes": (
                    float(led.state_bytes) / max(float(ba.full_bytes), 1.0)),
                **phi,
            },
            "metric_native": verdict,
            "fit": fit_rec,
            "wall_s": time.time() - t1,
        }

    # identical-phi across EVERY arm — raises, never warns
    phi_rows = {f"{n}/{a}": dict(phi) for n in rows for a in
                ("full", "launder", "substitute", "null", "blank")}
    phi_rows["clu/full"] = dict(phi)
    phi_rows["clu/launder"] = dict(phi)
    phi_id = assert_identical_phi(phi_rows)

    n_attempt = int(qs.meta.get("n_pairs", 0)) * int(gcfg.n_query_per_pair) \
        if family == "aggregate" else len(qs)
    return {
        "cell": label, "family": family, "arm": arm, "seed": seed,
        "degenerate": False, "metric": metric,
        "frontier_only": bool(family == "overload"),
        "label": (FRONTIER_LABEL if family == "overload" else
                  "reader-discrimination family (S = 0.5068) — the sole dividend family"),
        "n_live": int(len(ids)), "n_queries": int(len(qs)),
        "admissible_coverage": {
            "admissible": int(len(qs)), "attempted": int(max(n_attempt, len(qs))),
            "fraction": float(len(qs) / max(n_attempt, len(qs))),
            "n_offered": int(gcfg.n_offer), "n_admitted_by_store": int(len(ids)),
            "store_admission_fraction": float(len(ids) / max(gcfg.n_offer, 1)),
        },
        "clu_reproduction": {
            "full": clu_full, "launder": clu_lnd.get("settle_deleted"),
            "blank": clu_blank, "all_launder_scores": clu_lnd,
            "banked": BANKED_CLU.get(f"{family}/{arm}", {}),
            "note": ("the CLU column is BANKED (PREREG-Bprime §7); this is a "
                     "fidelity reproduction, never a re-derivation"),
        },
        "clu_byte_ledger": {**ba.as_dict(), "identity_T1": identity,
                            "two_sided_learned_init_rule": clu_ledger.as_dict()},
        "stream": tok_note, "fit_streams": fit_note, "phi_id": phi_id,
        "rivals": rows,
        "gym_config_non_default": gcfg.as_flag_table(),
        "clu_config_non_default": ccfg.as_flag_table(),
        "n_admitted": len(admitted), "wall_write_s": write_s,
    }


# --------------------------------------------------------------------------
# the byte-frontier column (D4) — labelled at every appearance
# --------------------------------------------------------------------------
def run_frontier_cell(seed: int = 0, *, heads: Sequence[int] = FRONTIER_HEADS,
                      rivals: Sequence[str] = FRONTIER_RIVALS,
                      quick: bool = False,
                      fit_kwargs: Optional[dict] = None) -> dict:
    """⛔ **BYTE-FRONTIER COLUMN.** Accuracy vs state bytes for the rivals, beside
    the CLU's **banked** ``decode 0.972 -> 0.097`` as the ratio falls
    ``478x -> 2.28x`` (§7 banked — **not** re-measured here)."""
    import jax

    family, arm = "overload", "load1x_shipped"
    over: Dict[str, Any] = {}
    if quick:
        over = dict(QUICK_GYM, clu_overrides=dict(QUICK_CLU))
    gcfg = gym_config(family, arm, seed=seed, **over)
    ccfg = gcfg.build_clu()
    metric = PRIMARY_METRIC[family]

    system = build_system(ccfg, key=jax.random.PRNGKey(seed), loud=False)
    stream = make_gym_stream(gcfg, ccfg)
    key = jax.random.PRNGKey(seed + 1)
    prev = 0
    for b in stream.chunks:
        if b > prev:
            key, k_w = jax.random.split(key)
            system.write_stream(stream.items[prev:b], key=k_w)
            prev = b
        system.consolidate()
    ids, centers, pays = system.codebook()
    order_map = _insertion_order(system)
    born = np.asarray([order_map.get(int(i), -1) for i in ids], dtype=float)
    rng = np.random.default_rng(seed + 7717)
    qs = _build_queries(gcfg, ccfg, stream, system, centers, pays, born, rng)
    xs, mask, tok = stream_tokens(stream, ccfg)
    examples, fit_note = aux_fit_examples(family, arm, seed, ccfg, gcfg)
    fkw = dict(QUICK_FIT) if quick else {}
    fkw.update(dict(fit_kwargs or {}))
    d_in, m = int(ccfg.dim), int(ccfg.payload_dim)

    points = []
    for name in rivals:
        for d in heads:
            k_fit = jax.random.PRNGKey(seed * 1000 + int(d))
            model, fit_rec = fit_best_of_grid(
                name, d_in, m, examples, key=k_fit, d_head=int(d),
                lrs=fkw.get("lrs", LR_GRID), steps=int(fkw.get("steps", 400)),
                b_grid=fkw.get("b_grid"))
            st = int(model.declared_state_floats())
            n_rows = matched_table_rows(st, model.d_k, model.d_v)
            pred = rival_arms(model, xs, mask, np.asarray(qs.q0, dtype=np.float32),
                              rng=np.random.default_rng(seed + 31337), n_rows=n_rows)
            sc = {k: float(score(qs, v)[metric]) for k, v in pred.items()}
            points.append({
                "rival": name, "d_head": int(d), "state_floats": st,
                "state_bytes": st * 4, "table_rows_affordable": int(n_rows),
                "table_rows_used": int(min(n_rows, xs.shape[0])),
                "table_is_lossless": bool(n_rows >= xs.shape[0]),
                "arms": sc, "dividend_vs_own_table": float(sc["full"] - sc["launder"]),
                "fit": fit_rec["best"],
            })
    return {"cell": f"{family}/{arm}@s{seed}", "seed": int(seed),
            "label": FRONTIER_LABEL, "metric": metric,
            "n_stream_tokens": int(xs.shape[0]), "n_queries": int(len(qs)),
            "n_live": int(len(ids)), "stream": tok, "fit_streams": fit_note,
            "clu_banked_curve": BANKED_CLU["overload/load1x_shipped"][
                "accuracy_vs_bytes_curve"],
            "points": points}


# --------------------------------------------------------------------------
# aggregation + the PREREG scorecard
# --------------------------------------------------------------------------
def _mean_se(v: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return float(np.mean(a)), float(sd / np.sqrt(a.size))


def audit_table(records: Sequence[dict]) -> Dict[str, Any]:
    """⭐ The audit table — one row per (family, rival), 3-seed mean ± SE."""
    out: Dict[str, Any] = {}
    for fam in ("aggregate", "overload"):
        cells = [r for r in records
                 if r.get("family") == fam and not r.get("degenerate")]
        if not cells:
            continue
        fam_rows: Dict[str, Any] = {}
        for name in RIVALS:
            per = [c["rivals"][name] for c in cells if name in c.get("rivals", {})]
            if not per:
                continue
            full_m, full_se = _mean_se([p["arms"]["full"] for p in per])
            lnd_m, lnd_se = _mean_se([p["arms"]["launder"] for p in per])
            div_m, div_se = _mean_se([p["dividend_vs_own_table"] for p in per])
            zb_m, zb_se = _mean_se(
                [p["zero_byte_margin"]["signed_margin_full_minus_sub"] for p in per])
            raw_m, raw_se = _mean_se(
                [p["raw_table_control"]["signed_margin_full_minus_raw"] for p in per])
            incl_m, _ = _mean_se(
                [p["zero_byte_margin"]["signed_margin_incl_argmin"] for p in per])
            null_m, _ = _mean_se([p["arms"]["same_keys_null"] for p in per])
            blank_m, _ = _mean_se([p["arms"]["blank"] for p in per])
            # ⭐ the RESCUE / informativeness gate (`rival-recon` F3's sanity gate):
            # an arm within noise of its OWN blank store is not rescued, and ⛔ no
            # margin against it is quotable.
            lift_m, lift_se = _mean_se([p["arms"]["full"] - p["arms"]["blank"]
                                        for p in per])
            rescued = bool(np.isfinite(lift_m) and np.isfinite(lift_se)
                           and lift_m > 2.0 * lift_se)
            fam_rows[name] = {
                "n_seeds": len(per), "d_head": per[0]["d_head"],
                "full": full_m, "full_se": full_se,
                "launder": lnd_m, "launder_se": lnd_se,
                "dividend_vs_own_table": div_m, "dividend_se": div_se,
                "zero_byte_substitute": per[0]["zero_byte_margin"]["name"],
                "zero_byte_margin": zb_m, "zero_byte_margin_se": zb_se,
                "zero_byte_margin_incl_argmin": incl_m,
                "beats_own_plus0B": bool(np.isfinite(zb_m) and zb_m > 0),
                "raw_table_reader": per[0]["raw_table_control"]["best_reader"],
                "raw_table_margin": raw_m, "raw_table_margin_se": raw_se,
                "beats_raw_table_plus0B": bool(np.isfinite(raw_m) and raw_m > 0),
                "same_keys_null": null_m, "blank": blank_m,
                "lift_over_own_blank": lift_m, "lift_se": lift_se,
                "RESCUED_above_own_blank_2se": rescued,
                "state_bytes": per[0]["byte_ledger"]["rival"]["state_bytes"],
                "param_bytes": per[0]["byte_ledger"]["rival"]["param_bytes"],
                "table_bytes": per[0]["byte_ledger"]["matched_table"]["state_bytes"],
                "state_over_own_table_bytes":
                    per[0]["byte_ledger"]["state_over_own_table_bytes"],
                "table_is_lossless": per[0]["byte_ledger"]["table_is_lossless"],
                "metric_native": per[0]["metric_native"]["verdict"],
            }
        clu_full = [c["clu_reproduction"]["full"] for c in cells]
        clu_lnd = [c["clu_reproduction"]["launder"] for c in cells]
        cf, cfse = _mean_se(clu_full)
        cl, clse = _mean_se(clu_lnd)
        out[fam] = {
            "is_dividend_family": bool(fam == "aggregate"),
            "label": cells[0]["label"],
            "rivals": fam_rows,
            "clu_banked": BANKED_CLU.get(f"{fam}/{cells[0]['arm']}", {}),
            "clu_reproduced": {"full": cf, "full_se": cfse, "launder": cl,
                               "launder_se": clse,
                               "dividend": float(cf - cl) if np.isfinite(cf) else
                               float("nan")},
            "admissible_coverage": [c["admissible_coverage"] for c in cells],
        }
    return out


def prereg_scorecard(table: Dict[str, Any]) -> Dict[str, Any]:
    """Registered vs measured vs verdict, **including every NOT-RUN**."""
    agg = table.get("aggregate", {}).get("rivals", {})
    div = {k: v["dividend_vs_own_table"] for k, v in agg.items()}
    zb = {k: v["zero_byte_margin"] for k, v in agg.items()}
    raw = {k: v.get("raw_table_margin", float("nan")) for k, v in agg.items()}
    n_lose_plus0b = sum(1 for k, v in zb.items() if np.isfinite(v) and v <= 0)
    delta_arms = [k for k in ("deltanet", "gdn", "gdn2") if k in zb]
    delta_lose = sum(1 for k in delta_arms if np.isfinite(zb[k]) and zb[k] <= 0)
    delta_lose_raw = sum(1 for k in delta_arms
                         if np.isfinite(raw[k]) and raw[k] <= 0)
    rows = [
        {"id": "P2 (measured half)",
         "registered": (">= 2 of the 3 measured (k,v)-shaped-state families "
                        "(DeltaNet, GDN, GDN-2) lose to their own byte-matched "
                        "table's strongest +0 B reader"),
         "measured": f"{delta_lose} of {len(delta_arms)}",
         "verdict": ("SUPPORTED" if delta_lose >= 2 else "REFUTED"),
         "second_reading_raw_metric_table": (
             f"{delta_lose_raw} of {len(delta_arms)} lose to the RAW-metric "
             "+0 B table at the same bytes (the strictly stronger control, "
             "reported beside the registered one and pre-committed to)"),
         "scope": ("⚠ FIRST HALF ONLY. The real-data-LM half (0 of 4 lose on bpc) "
                   "belongs to `cluformer-pilot` and is NOT tested here. "
                   "Mamba-2 and SDM are adjudicated from their equations only, "
                   "never blurred with the measured three.")},
        {"id": "P3", "registered": ("the two FUNCTION-VALUED memories (TTT-MLP, "
                                    "Titans L_M>=2) show the largest positive "
                                    "dividend"),
         "measured": "—", "verdict": "⛔ NOT-RUN",
         "scope": ("no Titans arm (D5 Hub ruling: no official code, chunk size "
                   "never numeric, no seeds), so the PAIR cannot be formed. "
                   "TTT-MLP alone is reported as a single-arm datum. "
                   "⛔ NOT-RUN is NOT refuted.")},
        {"id": "P5", "registered": ("the launder transfers to all five rival state "
                                    "types; predicted failures 0 of 5"),
         "measured": f"{len(agg)} of 5 state types carry a byte-matched table",
         "verdict": ("SUPPORTED" if len(agg) == 5 else "PARTIAL"),
         "scope": ("state types run: d_head^2 (TTT-Linear), 8*d_head^2 (TTT-MLP), "
                   "n_head*d_k*d_v x3 (DeltaNet, GDN, GDN-2)")},
        {"id": "R4 (mine): dividend vs own ARG-MIN table = +0.27 [+0.05,+0.45]",
         "registered": "+0.27", "measured": _fmt_map(div),
         "verdict": _band_verdict(np.mean([v for v in div.values()
                                           if np.isfinite(v)]) if div else np.nan,
                                  0.05, 0.45)},
        {"id": "R5 (mine): signed +0 B margin = -0.02 [-0.15,+0.08], >=3 of 5 <= 0",
         "registered": "-0.02; >=3 of 5 arms <= 0",
         "measured": _fmt_map(zb) + f"; {n_lose_plus0b} of {len(zb)} <= 0",
         "verdict": _band_verdict(np.mean([v for v in zb.values()
                                           if np.isfinite(v)]) if zb else np.nan,
                                  -0.15, 0.08)},
        {"id": "R5-raw (the stronger control, reported beside R5)",
         "registered": ("not separately banded — the raw-metric table was added "
                        "after the PREREG's R5 band was fixed and is reported as "
                        "a SECOND READING, never substituted for it"),
         "measured": _fmt_map(raw),
         "verdict": (f"{sum(1 for v in raw.values() if np.isfinite(v) and v <= 0)}"
                     f" of {len(raw)} arms <= 0")},
    ]
    return {"rows": rows,
            "not_run": {
                "Titans": ("D5 Hub ruling — NeurIPS 2025, peer-reviewed (⛔ never "
                           "'a preprint'); no official code, chunk size b never "
                           "given a numeric value, no seeds reported. Positioning "
                           "only. Its 2*|M_theta| momentum accounting remains OUR "
                           "reconstruction and stays ⚠ UNPINNED."),
                "Sparse Delta Memory": ("D5 — official code needs Torch>=2.8, "
                                        "Triton>=3.4, SM 80+; cannot run on this "
                                        "machine. Positioning only. ⛔ quote NONE "
                                        "of its Table 1 state/param ratios."),
                "recency / manifold families": ("⛔ substitute-saturated at +0 B "
                                                "(S = 1.0000) ⇒ struck as "
                                                "protocol-invalid by FB4."),
                "Mamba-2 / GRU / sliding-window attention": (
                    "not in §A14.2's ruled family set."),
            }}


def _fmt_map(d: Dict[str, float]) -> str:
    return ", ".join(f"{k} {v:+.4f}" for k, v in sorted(d.items()))


def _band_verdict(x: float, lo: float, hi: float) -> str:
    if not np.isfinite(x):
        return "NOT-RUN"
    return f"mean {x:+.4f} — {'IN BAND' if lo <= x <= hi else 'OUT OF BAND'}"


def falsifier_adjudication(table: Dict[str, Any]) -> Dict[str, Any]:
    """FB1 / FB2 / FB3 / FB5, each with its evidence (D3 + §5)."""
    agg = table.get("aggregate", {}).get("rivals", {})
    clu = table.get("aggregate", {}).get("clu_banked", {})
    clu_div = (float(np.mean(clu["full"]) - np.mean(clu["launder"]))
               if clu else float("nan"))
    resc = {k: v for k, v in agg.items() if v.get("RESCUED_above_own_blank_2se")}
    pos = [k for k, v in resc.items() if v["dividend_vs_own_table"] > 0]
    beats_zb = [k for k, v in resc.items() if v["beats_own_plus0B"]]
    beats_raw = [k for k, v in resc.items() if v.get("beats_raw_table_plus0B")]
    fb3_strong = bool(len(beats_raw) >= 4 and clu_div <= 0)
    return {
        "rescue_gate": {
            "rule": ("`rival-recon` F3's sanity gate, applied per (family, rival): "
                     "an arm within 2 SE of its OWN blank-store control is NOT "
                     "RESCUED and ⛔ no margin against it is quotable. Only "
                     "rescued arms enter the FB2/FB3 adjudications below."),
            "rescued_on_aggregate": sorted(resc),
            "not_rescued_on_aggregate": sorted(set(agg) - set(resc)),
        },
        "FB1": {"verdict": "DOES NOT FIRE",
                "evidence": ("inherited from `bprime-fb1-recon` (14 candidates: "
                             "0 HIT, 2 PARTIAL both out-of-family, 7 NEAR-MISS, "
                             "5 NO) and nothing seen from inside these "
                             "implementations changes it. ⭐ The surviving claim "
                             "is the NARROWED one: seven independent groups built "
                             "the adjacent instrument and none closed the loop. "
                             "The audit-at-equal-bits discipline IS standard "
                             "OUTSIDE the family (learned Bloom filters, learned "
                             "indexes, SOSD) — cited as B′'s methodological "
                             "ancestry, never suppressed — and a token-matched "
                             "trivial control was published 7 days before filing "
                             "(arXiv:2607.21962). The substitute-audit IDEA in "
                             "general form is Poliak et al. 2018 / Feng, Wallace "
                             "& Boyd-Graber ACL 2019: conceded.")},
        "FB2": {"verdict": "DOES NOT FIRE",
                "measured_families": sorted(agg.keys()),
                "reasoned_from_equations_only": ["Mamba-2", "Sparse Delta Memory",
                                                 "Titans"],
                "evidence": ("a byte-matched table is definable WITHOUT an "
                             "arbitrary modelling choice for every state type "
                             "measured: each has an explicit float state and an "
                             "explicit (theta_K x, theta_V x) stream, so "
                             "n_rows = floor(state_floats/(d_k+d_v)) is forced, "
                             "not chosen. ⚠ 2 of the 5 §2 families were NOT "
                             "adjudicated by measurement and are stated as such.")},
        "FB3": {"verdict": ("FIRES (strong form)" if fb3_strong else
                            "DOES NOT FIRE in the strong form"),
                "rivals_with_positive_dividend_vs_argmin_table": sorted(pos),
                "rivals_beating_their_own_plus0B_reader": sorted(beats_zb),
                "rivals_beating_the_RAW-metric +0B table at the same bytes":
                    sorted(beats_raw),
                "clu_banked_dividend_vs_argmin_table": clu_div,
                "evidence": ("⭐ We pre-committed IN WRITING to saying so rather "
                             "than re-framing. The distinction that decides it was "
                             "registered BEFORE measurement (PREREG R4 vs R5): a "
                             "dividend against the ARG-MIN control is not the same "
                             "claim as a dividend against the family's own +0 B "
                             "reader, and only the latter would make B′ a "
                             "different paper.")},
        "FB5": {"verdict": "DOES NOT FIRE",
                "evidence": ("arXiv:2501.12352 (test-time regression) is purely "
                             "theoretical — softmax attention appears as the "
                             "nonparametric special case ANALYTICALLY, with no "
                             "experiments and no baselines. They unify mechanisms; "
                             "B′ prices them.")},
    }


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------
def run_experiment_bprime_rivals(config=None, save_dir: str = "results",
                                 models_dir: Optional[str] = None,
                                 seed: Optional[int] = None,
                                 families: Optional[Sequence[str]] = None,
                                 rivals: Optional[Sequence[str]] = None,
                                 seeds: Optional[Sequence[int]] = None,
                                 quick: bool = False,
                                 frontier: bool = True,
                                 overrides: Optional[dict] = None) -> dict:
    """Run the audit plan and write the artifact."""
    os.makedirs(save_dir, exist_ok=True)
    base_seed = int(seed if seed is not None else getattr(
        getattr(config, "project", None), "seed", 0) or 0)
    plan = [(f, a) for f, a in DEFAULT_PLAN if (not families or f in set(families))]
    cell_seeds = tuple(int(s) for s in (seeds if seeds else DEFAULT_SEEDS))
    if base_seed and not seeds:
        cell_seeds = tuple(s + base_seed for s in cell_seeds)
    riv = tuple(rivals) if rivals else RIVALS

    results: Dict[str, Any] = {
        "experiment": "bprime_rivals",
        "question": ("B′ TIER i: when does test-time dynamics buy anything over a "
                     "table at matched bytes? One protocol, applied uniformly to "
                     "the CLU and to the TTT / delta-rule families."),
        "family_set_ruling": (
            "§A14.2: `aggregate` is the SOLE reader-discrimination family; "
            "`overload@load1x_shipped` runs ONLY as a labelled byte-frontier "
            "column (secondary reading S_excl = 0.6500); `recency` and `manifold` "
            "are NOT RUN (substitute-saturated ⇒ protocol-invalid)."),
        "thinness": ("⚠ OWNED, VERBATIM: two rival families audited against ONE "
                     "surviving synthetic family is a THIN cross-family audit. "
                     "This belongs in the paper's Limitations unsoftened."),
        "reimplementation_caption": (
            "minimal FAITHFUL reimplementations on the gym harness — faithful to "
            "the update equation and the state size, minimal in everything else "
            "(n_head=1, no conv branch, no SWA hybrid, no chunkwise kernel, no "
            "backbone). NOT a vendored training stack."),
        "equations_implemented": {
            "ttt_linear/ttt_mlp": ("arXiv:2407.04620 Eqs. 1, 2, 4, 5; §2.4 "
                                   "mini-batch (b=16, in the tuning grid); §2.7 "
                                   "f_res = x + LN(f(x)), learnable W_0 and eta, "
                                   "f_MLP = 2 layers / 4x hidden / GELU"),
            "deltanet": "arXiv:2605.22791 Eq. 5 (= Yang et al. 2024)",
            "gdn": "arXiv:2605.22791 Eq. 6 (= Yang et al. 2025, the ABLATION)",
            "gdn2": ("arXiv:2605.22791 **Eq. 10** (boxed) with Eqs. 8, 9, 11, 12 "
                     "and §3.1's negative-eigenvalue erase range [0,2]^{d_k}; "
                     "§3.5 block design (L2 on q/k, SiLU on v). ⭐ STATE-SIZE "
                     "CONVENTION VERIFIED FROM ITS OWN Eq. 90 (H*d_k*d_v = "
                     "16*128*128 = 262,144 floats per layer): the -2 revision "
                     "PRESERVES the DeltaNet/GDN accounting."),
        },
        "byte_law": ("corrected: ratio = [A(D+2) + d]/(d+m); floor 2.20x "
                     "(n_spec=0) / 2.40x (n_spec=1). ⛔ the published 'verified to "
                     "1e-9 in all 28 cells' is 24/28 — the 4 n_spectator=1 cells "
                     "miss by +8.6667. The error was CONSERVATIVE."),
        "banked_not_remeasured": sorted(BANKED_CLU.keys()),
        "plan": [{"family": f, "arm": a, "seeds": list(cell_seeds)} for f, a in plan],
        "rivals": list(riv), "quick": bool(quick),
        "sd_convention": "sample sd (ddof=1); SE = sd/sqrt(n); n = 3",
        "prereg": ".claude/outputs/bprime-rivals/PREREG.md — filed before any run",
        "cells": [], "frontier": [],
    }
    results_dir = (os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
                   if os.path.basename(os.path.abspath(save_dir)) == "plots"
                   else save_dir)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_bprime_rivals_metrics.json")

    def _dump():
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=_json_default)

    for fam, arm in plan:
        for s in cell_seeds:
            t0 = time.time()
            try:
                rec = run_rivals_cell(fam, arm, int(s), rivals=riv, quick=quick,
                                      fit_kwargs=(overrides or {}).get("fit"))
            except Exception as exc:  # a failed cell is reported, never silent
                import traceback

                traceback.print_exc()
                rec = {"cell": f"{fam}/{arm}@s{s}", "family": fam, "arm": arm,
                       "seed": int(s), "degenerate": True, "error": repr(exc)}
            rec["wall_s"] = time.time() - t0
            results["cells"].append(rec)
            for name, r in rec.get("rivals", {}).items():
                print(f"[{rec['cell']}] {name:11s} full={r['arms']['full']:+.4f} "
                      f"launder={r['arms']['launder']:+.4f} "
                      f"div={r['dividend_vs_own_table']:+.4f} "
                      f"+0B_margin={r['zero_byte_margin']['signed_margin_full_minus_sub']:+.4f} "
                      f"blank={r['arms']['blank']:+.4f} "
                      f"state={r['byte_ledger']['rival']['state_bytes']}B "
                      f"table={r['byte_ledger']['matched_table']['state_bytes']}B")
            print(f"[{rec['cell']}] done in {rec['wall_s']:.0f}s")
            _dump()

    if frontier and any(f == "overload" for f, _ in plan):
        for s in cell_seeds:
            try:
                results["frontier"].append(run_frontier_cell(
                    int(s), quick=quick, fit_kwargs=(overrides or {}).get("fit")))
            except Exception as exc:  # pragma: no cover
                import traceback

                traceback.print_exc()
                results["frontier"].append({"seed": int(s), "error": repr(exc)})
            _dump()

    results["audit_table"] = audit_table(results["cells"])
    results["prereg_scorecard"] = prereg_scorecard(results["audit_table"])
    results["falsifiers"] = falsifier_adjudication(results["audit_table"])
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

    tab = results.get("audit_table", {}).get("aggregate", {})
    rows = tab.get("rivals", {})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    names = sorted(rows)
    x = np.arange(len(names) + 1)
    clu = tab.get("clu_banked", {})
    full = [rows[n]["full"] for n in names] + [float(np.mean(clu.get("full", [np.nan])))]
    lnd = [rows[n]["launder"] for n in names] + [
        float(np.mean(clu.get("launder", [np.nan])))]
    zb = [rows[n]["full"] - rows[n]["zero_byte_margin"] for n in names] + [
        clu.get("best_zero_byte_substitute", {}).get("mean", np.nan)]
    ax.bar(x - 0.25, full, 0.25, label="full (learned dynamics)")
    ax.bar(x, lnd, 0.25, label="own byte-matched table (arg-min)")
    ax.bar(x + 0.25, zb, 0.25, label="best +0 B reader of that table")
    ax.set_xticks(x)
    ax.set_xticklabels(names + ["CLU (banked)"], fontsize=7, rotation=20)
    ax.set_ylabel("neg_mae (higher is better)")
    ax.set_title("B′ tier i — `aggregate`, the SOLE dividend family (S=0.5068)\n"
                 "one protocol, every family", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name in sorted({p["rival"] for c in results.get("frontier", [])
                        for p in c.get("points", [])}):
        pts = [(p["state_bytes"], p["arms"]["full"], p["arms"]["launder"])
               for c in results.get("frontier", []) for p in c.get("points", [])
               if p["rival"] == name]
        if not pts:
            continue
        pts.sort()
        b = np.asarray([p[0] for p in pts], dtype=float)
        ax.plot(b, [p[1] for p in pts], "o-", label=f"{name} full")
        ax.plot(b, [p[2] for p in pts], "s--", alpha=0.6, label=f"{name} own table")
    ax.set_xscale("log")
    ax.set_xlabel("rival state bytes")
    ax.set_ylabel("decode")
    ax.set_title("⛔ BYTE-FRONTIER COLUMN (overload@load1x_shipped)\n"
                 "not a dividend family; secondary reading S_excl = 0.6500",
                 fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(save_dir, "exp_bprime_rivals.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment B′-RIVALS: the cross-family matched-byte audit")
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Base seed offset")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument("--families", nargs="+", help="Run only these families")
    parser.add_argument("--rivals", nargs="+", help="Run only these rivals")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the seed list")
    parser.add_argument("--no-frontier", action="store_true",
                        help="Skip the byte-frontier column")
    parser.add_argument("--save-dir", default="results", help="Where figures go")
    args = parser.parse_args()

    config = None
    save_dir = args.save_dir
    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        save_dir = str(pm.get_paths(args.project)["plots"])
    res = run_experiment_bprime_rivals(
        config=config, save_dir=save_dir, seed=args.seed, families=args.families,
        rivals=args.rivals, seeds=args.seeds, quick=args.quick,
        frontier=not args.no_frontier)
    print(json.dumps({"audit_table": res["audit_table"],
                      "prereg_scorecard": res["prereg_scorecard"],
                      "falsifiers": res["falsifiers"]},
                     indent=2, default=_json_default))


if __name__ == "__main__":
    main()
