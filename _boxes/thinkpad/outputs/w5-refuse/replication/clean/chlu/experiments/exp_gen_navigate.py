"""Experiment: ⭐ **navigation** — the commitment lands on nine points; is the space
BETWEEN them structured?

⭐ DIAL DECLARATION — *navigation*: does the store's landscape support meaningful
movement **between** wells/modes?

- **Laundering control:** ⭐ **straight-line interpolation between the two endpoint
  READ vectors, with the store bypassed at every intermediate point**, on EVERY
  number.  ⛔ If it traverses as well, "navigation" is interpolation with machinery
  attached and that is the headline.
- **Second control:** the identical path read through a **deleted** store
  (``mean`` — the fair deletion) and through a ``scramble``-deleted store.
- **Falsifies:** the store path never visits an intermediate factor value, or the
  linear control traverses as well or better.
- **Does NOT falsify:** a corner-only failure that is absent on the spread grid
  (R30/R31 already established *the corner collapses, not the hold-out*), or the
  ``mean``-deleted arm rendering a constant image (arithmetic, not a finding).

⭐ **Why this exists, stated honestly.**  ``gen-commitment-collapse`` FAILED its
pre-registered gate: 72 held-out requests receive **9** codes and no stopping rule,
multiplicity or temperature moves it (R31; ``argmax`` is provably temperature-blind,
and at 160.2 bits — 5.34× the design constant — the same store still scores ``scale``
0.000).  R29 pre-registered that a fail routes to **new approaches** ("particle
traversal etc."), ⛔ never to abandonment and ⛔ never to re-tuning.  This is that
spoke, and it is also pivot item 4.

⛔ **Two traps this module is written around, both already paid for.**
1. ``GenBridgeConfig.dec_recipe`` defaults to ``"baseline"`` — the decoder that
   **FAILED** the gate.  The winner is ``v1_capacity_schedule``; :func:`resolve_decoder`
   asserts the parameter **hash** and the **recipe name**, so a void run cannot look
   normal in the log.
2. ⭐ The instrument is the **gated external classifier** (``classifier_gate.json`` =
   ``PASS``, 3/3 seeds at 1.0000 on real held-out images), ⛔ never the linear probe —
   it shares the decoder's lineage.  Its parameter hash is asserted per seed.

⛔ **The guarantee.**  ``read_from_codes(store, codes)`` takes no features and the
only encoder→decoder channel is R20's declared **30 bits**.  ``soft_read`` and
``address_codes`` take FEATURES and are ⛔ not imported anywhere on this path;
:func:`~chlu.core.gen_navigate.assert_no_feature_tensor` checks that four ways, and
the realised per-point channel width is reported for every arm at every point.

Runnable::

    python -m chlu.experiments.exp_gen_navigate --dataset shapes3d --pretext recon
    python -m chlu.experiments.exp_gen_navigate --quick        # ⚠ plumbing smoke only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, get_default_config, gen_navigate_config  # noqa: E402
from ..core.gen_bridge import PRETEXTS, param_hash  # noqa: E402
from ..core.gen_compose import (  # noqa: E402
    assert_request_only,
    compose_codes,
    request_posterior,
)
from ..core.gen_dream import requests_from_labels  # noqa: E402
from ..core.gen_navigate import (  # noqa: E402
    ARMS,
    STORE_ARM,
    arm_path_features,
    assert_no_feature_tensor,
    image_smoothness,
    navigation_summary,
    path_channel,
    select_pairs,
    traversal_metrics,
)
from ..core.gen_store import load_store_plan  # noqa: E402
from ..eval.gen_probe import decode_features, predict_factors  # noqa: E402
from ..experiments.exp_gen_compose import (  # noqa: E402
    _claude_root,
    _ms,
    _write_json,
    build_commit_cache,
    load_commit_store,
    resolve_decoder,
    store_plan_path,
)
from ..utils.checkpoints import load_model  # noqa: E402

#: PREREG §2.  ``CORNER`` = the 72 held-out (B) requests (Montero's cell);
#: ``GRID`` = lines through seen ``eval_a`` anchors, ⛔ corner anchors excluded.
POOLS = ("CORNER", "GRID")


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------


def _cfg(config: CHLUConfig):
    return gen_navigate_config(config)


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only.  ⛔ A ``--quick`` number is NEVER reported."""
    c = _cfg(config)
    c.seeds = list(c.quick_seeds)
    c.n_pairs_per_factor = int(c.quick_n_pairs_per_factor)
    c.n_steps = int(c.quick_n_steps)
    c.pools = list(c.quick_pools)
    return config


def nav_out_root(config: CHLUConfig) -> Path:
    c = _cfg(config)
    if c.out_root:
        return Path(c.out_root).expanduser()
    return _claude_root(config) / "outputs" / "gen-navigation"


def _abs(config: CHLUConfig, p: str) -> Path:
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else _claude_root(config) / q


def commit_cache_dir(config: CHLUConfig, dataset: str, pretext: str, seed: int) -> Path:
    c = _cfg(config)
    return _abs(config, c.commit_cache_pattern.format(
        dataset=dataset, pretext=pretext, seed=int(seed)))


def resolve_classifier(config: CHLUConfig, seed: int, log: Callable = print):
    """⭐ Load the **gated** external classifier and refuse an un-gated instrument.

    ⛔ Never the linear probe: R29's headline table needs a reader that is not of
    the decoder's own lineage, and ``gen-external-classifier`` built exactly that
    (gate PASS 3/3 at 1.0000 on REAL held-out images, floor 0.95).  The gate file is
    read, the verdict asserted, and the checkpoint's parameter hash asserted against
    the seed's own manifest.
    """
    c = _cfg(config)
    root = _abs(config, c.classifier_root)
    gate_path = root / str(c.gate_file)
    with open(gate_path) as fh:
        gate = json.load(fh)
    verdict = str(gate.get("gate", "")).upper()
    if bool(c.require_gate_pass) and verdict != "PASS":
        raise AssertionError(
            f"⛔ {gate_path} says gate={verdict!r} — refusing to read a traversal with "
            "an un-gated instrument"
        )
    path = root / c.classifier_pattern.format(seed=int(seed))
    clf = load_model(path)
    h = param_hash(clf)
    man_path = path.parent / "classifier_manifest.json"
    expected = None
    if man_path.exists():
        with open(man_path) as fh:
            man = json.load(fh)
        expected = (man.get("classifier_param_hash")
                    or (man.get("classifier") or {}).get("param_hash"))
    if expected and expected != h:
        raise AssertionError(
            f"the external classifier at {path} hashes {h[:16]}… but its manifest "
            f"records {str(expected)[:16]}… — ⛔ different instrument"
        )
    log(f"  ⭐ external classifier: gate={verdict} hash={h[:16]}… "
        f"(⛔ not the probe — the probe shares the decoder's lineage)")
    return clf, {"gate": verdict, "gate_path": str(gate_path),
                 "classifier_path": str(path), "classifier_param_hash": h,
                 "classifier_hash_asserted_against_manifest": bool(expected)}


# ---------------------------------------------------------------------------
# ⭐ Pool construction — ⛔ enumerated and seeded, never hand-picked (PREREG §2)
# ---------------------------------------------------------------------------


def corner_index_sets(grid) -> list:
    """Montero's cell expressed as per-factor **index** sets.

    ⛔ The inequalities are not restated here: they come from
    :func:`chlu.data.shapes3d.montero_cell`, so the corner used for pool
    construction is the same object the split itself is built from.
    """
    from ..data.shapes3d import montero_cell

    cell = montero_cell(grid)
    out = []
    for j, name in enumerate(grid.names):
        vals = np.asarray(grid.values[j], np.float64)
        allowed = np.asarray(cell.values[name], np.float64)
        out.append(np.flatnonzero(np.isin(np.round(vals, 10), np.round(allowed, 10))))
    return out


def pool_requests(pool: str, factor: int, corner_req: np.ndarray,
                  anchors: np.ndarray, cardinalities) -> np.ndarray:
    """The request set a pool enumerates its single-factor pairs over.

    ``CORNER`` the 72 held-out requests verbatim.
    ``GRID``   the **line** through each seen anchor along ``factor`` — every grid
               value of that factor, all other five held at the anchor's own values.
               ⭐ That makes "all unordered pairs differing in exactly one factor"
               (PREREG §2) a finite, mechanically enumerable set.
    """
    if pool == "CORNER":
        return np.asarray(corner_req, np.int64)
    if pool != "GRID":
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
    A = np.asarray(anchors, np.int64)
    card = int(cardinalities[int(factor)])
    rows = np.repeat(A, card, axis=0)
    rows[:, int(factor)] = np.tile(np.arange(card, dtype=np.int64), A.shape[0])
    return rows


def grid_anchors(config: CHLUConfig, a_requests: np.ndarray, grid) -> dict:
    """Seen ``eval_a`` configurations, ⛔ with Montero's corner removed."""
    R = np.asarray(a_requests, np.int64)
    sets = corner_index_sets(grid)
    inside = np.ones(R.shape[0], bool)
    for j, allowed in enumerate(sets):
        inside &= np.isin(R[:, j], allowed)
    return {"anchors": R[~inside], "n_dropped_in_corner": int(inside.sum()),
            "n_anchors": int((~inside).sum())}


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


def _codes_for(book, requests: np.ndarray) -> np.ndarray:
    """``requests -> committed K-tuples``.  ⛔ Request-only; no feature tensor exists
    on this path (``assert_request_only`` is run over it before any number)."""
    return compose_codes(request_posterior(book, np.asarray(requests, np.int64)))["code"]


def run_pair_group(
    store, decoder, clf, book, pairs: np.ndarray, requests: np.ndarray,
    factor: int, names, qualifies: bool, cfg, seed: int, log: Callable = print,
) -> list:
    """Every arm's traversal for one (pool, factor) group, in one decode pass per arm."""
    S = int(cfg.n_steps)
    req = np.asarray(requests, np.int64)
    codes = _codes_for(book, req[np.unique(pairs)]) if pairs.size else None
    idx_map = {int(r): i for i, r in enumerate(np.unique(pairs))} if pairs.size else {}
    recs = []
    for (ia, ib) in pairs:
        ca, cb = codes[idx_map[int(ia)]], codes[idx_map[int(ib)]]
        recs.append({
            "factor": int(factor), "factor_name": str(names[int(factor)]),
            "request_a": [int(x) for x in req[int(ia)]],
            "request_b": [int(x) for x in req[int(ib)]],
            "code_a": [int(x) for x in ca], "code_b": [int(x) for x in cb],
            "endpoint_codes_identical": bool(np.array_equal(ca, cb)),
            "code_hamming": int(np.sum(np.asarray(ca) != np.asarray(cb))),
            "qualifies_monotone": bool(qualifies),
            "arms": {},
        })
    if not recs:
        return recs

    for arm in ARMS:
        feats, per_pair_codes = [], []
        for rec in recs:
            got = arm_path_features(store, rec["code_a"], rec["code_b"], S, arm,
                                    seed=int(cfg.scramble_seed) + int(seed))
            feats.append(got["features"])
            per_pair_codes.append(got["codes"])
        Z = np.concatenate(feats, axis=0)
        imgs = decode_features(decoder, Z, int(cfg.eval_batch))
        reads = predict_factors(clf, imgs, int(cfg.eval_batch))
        ch = path_channel(store, arm, S, float(cfg.design_constant_bits))
        for i, rec in enumerate(recs):
            sl = slice(i * (S + 1), (i + 1) * (S + 1))
            others = [j for j in range(req.shape[1]) if j != int(factor)]
            shared = [int(rec["request_a"][j]) for j in others]
            m = traversal_metrics(
                reads[sl], int(factor), int(rec["request_a"][int(factor)]),
                int(rec["request_b"][int(factor)]), shared, bool(qualifies),
                float(cfg.stability_floor))
            m["smoothness"] = image_smoothness(imgs[sl])
            m["channel"] = ch
            m["per_point_bits"] = [float(ch["per_point_bits"])] * (S + 1)
            cds = per_pair_codes[i]
            if cds is None:
                m["n_distinct_codes_on_path"] = None
                m["n_interior_codes_off_endpoints"] = None
            else:
                u = np.unique(cds, axis=0)
                interior = cds[1:-1] if cds.shape[0] > 2 else cds[:0]
                off = [tuple(int(x) for x in c) for c in interior
                       if not np.array_equal(c, cds[0]) and not np.array_equal(c, cds[-1])]
                m["n_distinct_codes_on_path"] = int(u.shape[0])
                m["n_interior_codes_off_endpoints"] = int(len(set(off)))
            rec["arms"][arm] = m
        del feats, Z, imgs
    return recs


def run_seed(seed: int, out_dir: Path, config: CHLUConfig, log: Callable = print) -> dict:
    """⭐ One seed, written **per (pool, factor) unit** — ⛔ nothing is held to the end."""
    c = _cfg(config)
    dataset, pretext = str(c.dataset), str(c.pretext)
    sdir = Path(out_dir) / f"seed_{int(seed)}"
    sdir.mkdir(parents=True, exist_ok=True)
    man_path = sdir / f"navigate_manifest_s{seed}.json"
    if man_path.exists():
        with open(man_path) as fh:
            done = json.load(fh)
        if done.get("complete"):
            log(f"  ⏭ seed {seed}: manifest already complete — skipping (incremental)")
            return done

    t0 = time.time()

    def tlog(msg):
        log(f"[{time.time() - t0:8.1f}s] {msg}")

    # ---- the frozen artefacts, every one asserted --------------------------
    cache_dir = commit_cache_dir(config, dataset, pretext, seed)
    cache = build_commit_cache(dataset, pretext, int(seed), cache_dir, config, log=tlog)
    book, lab, names = cache["book"], cache["lab"], cache["names"]
    grid, cards = cache["grid"], cache["cards"]
    plan = load_store_plan(store_plan_path(config, dataset, pretext))
    store = load_commit_store(dataset, pretext, int(seed), cache_dir, config,
                              cache["labels_vec"], plan, tlog)
    decoder, dec_info, _ = resolve_decoder(config, cache["bdir"], int(seed), log=tlog)
    clf, clf_info = resolve_classifier(config, int(seed), log=tlog)

    # ---- the requests, and the two guarantees ------------------------------
    corner_req = requests_from_labels(lab["B"], "full", 0, int(seed))
    a_req = requests_from_labels(lab["A"], "full", 0, int(seed))
    anchors = grid_anchors(config, a_req, grid)
    tlog(f"⭐ pools: CORNER={corner_req.shape[0]} held-out requests · GRID anchors="
         f"{anchors['n_anchors']} of {a_req.shape[0]} "
         f"({anchors['n_dropped_in_corner']} dropped as in-corner)")

    req_only = assert_request_only(store, book, corner_req, decoy_features=None)
    tlog("⭐ endpoint path asserted request-only (no feature argument on ANY arm)")
    c0 = _codes_for(book, corner_req[:2])
    decoy = np.zeros((4, store.dim), np.float32)
    nofeat = assert_no_feature_tensor(store, c0[0], c0[1], int(c.n_steps),
                                      decoy_features=decoy)
    tlog(f"⭐ traversal path asserted feature-free (4 checks) · endpoints exact="
         f"{nofeat['endpoints_exact']} · {nofeat['bits_per_read']:.0f} bits/read")
    channels = {a: path_channel(store, a, int(c.n_steps), float(c.design_constant_bits))
                for a in ARMS}
    for a in ARMS:
        tlog(f"  channel [{a}]: {channels[a]['per_point_bits']:.1f} bits at EVERY "
             f"intermediate point (design constant {channels[a]['design_constant_bits']:.0f})"
             + (" ⛔ EXCEEDS" if channels[a]["exceeds_design_constant"] else " ⭐ within"))

    # ---- the traversals, unit by unit -------------------------------------
    records, selection = [], []
    partial = sdir / f"records_s{seed}.json"
    if partial.exists():
        with open(partial) as fh:
            prev = json.load(fh)
        records, selection = prev.get("records", []), prev.get("selection", [])
        tlog(f"  resuming: {len(records)} pair records already on disk")
    done_units = {(r["pool"], int(r["factor"])) for r in records}

    for pool in list(c.pools):
        for f in range(len(names)):
            if (pool, f) in done_units:
                tlog(f"  ⏭ {pool}/{names[f]} already done — skipping (per-unit)")
                continue
            reqs = pool_requests(pool, f, corner_req, anchors["anchors"], cards)
            rng = np.random.default_rng(
                int(seed) * int(c.pair_seed_mult) + int(c.pair_seed_offset)
                + 101 * POOLS.index(pool) + f)
            sel = select_pairs(reqs, f, int(c.n_pairs_per_factor), rng, int(c.min_delta))
            categorical = str(names[f]) in set(c.categorical_factors)
            if categorical:
                # ⛔ PREREG §2: a categorical factor is switch-only. "Monotone in
                # shape" is not a statement about anything.
                sel["qualifies_monotone"] = False
            sel_row = {"pool": pool, "factor": int(f), "factor_name": str(names[f]),
                       "categorical": bool(categorical),
                       "n_values_realised": int(sel["n_values_realised"]),
                       "n_candidate_pairs": int(sel["n_candidates"]),
                       "min_delta_used": int(sel["min_delta_used"]),
                       "qualifies_monotone": bool(sel["qualifies_monotone"]),
                       "n_pairs_drawn": int(sel["pairs"].shape[0]),
                       "n_pool_requests": int(reqs.shape[0])}
            selection.append(sel_row)
            if sel["pairs"].size == 0:
                tlog(f"  ⛔ {pool}/{names[f]}: NOT APPLICABLE — n_values_realised="
                     f"{sel['n_values_realised']}, 0 single-factor pairs exist "
                     "(⛔ reported as NOT RUN, never as a zero)")
                _write_json({"records": records, "selection": selection}, partial)
                continue
            got = run_pair_group(store, decoder, clf, book, sel["pairs"], reqs, f,
                                 names, bool(sel["qualifies_monotone"]), c, int(seed),
                                 log=tlog)
            for r in got:
                r["pool"] = pool
            records.extend(got)
            s = navigation_summary(got)
            ident = float(np.mean([r["endpoint_codes_identical"] for r in got]))
            if not sel["qualifies_monotone"]:
                # ⛔ n_values_realised < 3 ⇒ no intermediate value EXISTS, so every
                # correct path is a step function and a traverse rate here would be
                # a vacuous success.  Reported as the switch statistic instead.
                why = ("CATEGORICAL" if str(names[f]) in set(c.categorical_factors)
                       else "n_values_realised < 3")
                tlog(f"  ⚠ {pool}/{names[f]} (n_values_realised="
                     f"{sel['n_values_realised']}, {why} ⇒ SWITCH-ONLY, ⛔ excluded from the "
                     f"headline; {len(got)} pairs): clean-switch "
                     + " ".join(
                         f"{a}={np.mean([r['arms'][a]['clean_switch'] for r in got]):.2f}"
                         for a in ARMS)
                     + f" · identical endpoint codes={ident:.2f}")
            else:
                tlog(f"  ⭐ {pool}/{names[f]} (n_values_realised={sel['n_values_realised']}, "
                     f"|Δ|≥{sel['min_delta_used']}, {len(got)} pairs): traverse "
                     + " ".join(f"{a}={s['traverse_rate'][a]:.2f}" for a in ARMS)
                     + f" · NAV={s['NAV']:.3f} · identical endpoint codes={ident:.2f}")
            _write_json({"records": records, "selection": selection}, partial)

    man = {
        "schema": "chlu-gen-navigate/1", "complete": True,
        "dataset": dataset, "pretext": pretext, "seed": int(seed),
        "dial": ("navigation — does the store's landscape support meaningful "
                 "movement BETWEEN wells/modes? (R29 pivot item 4)"),
        "laundering_control": (
            "⭐ straight-line interpolation between the two endpoint READ vectors, "
            "store bypassed at every interior point, on EVERY number; plus the "
            "mean-deleted store (the fair deletion) and a scrambled store."),
        "decoder": dec_info,
        "classifier": clf_info,
        "store_param_hash": param_hash(store),
        "K": int(store.n_wells), "n_modes": int(store.n_modes),
        "bits_per_read": float(store.bits_per_read),
        "n_steps": int(c.n_steps),
        "request_only_path": req_only,
        "no_feature_tensor": nofeat,
        "channel_by_arm": channels,
        "pools": {"corner_requests": int(corner_req.shape[0]), **anchors,
                  "anchors": None},
        "selection": selection,
        "summary_overall": navigation_summary(records),
        "summary_by_pool": {
            p: navigation_summary([r for r in records if r["pool"] == p])
            for p in {r["pool"] for r in records}},
        "summary_by_pool_factor": {
            f"{p}/{nm}": navigation_summary(
                [r for r in records if r["pool"] == p and r["factor_name"] == nm])
            for p in {r["pool"] for r in records}
            for nm in {r["factor_name"] for r in records if r["pool"] == p}},
        "wall_clock_s": float(time.time() - t0),
    }
    man["pools"].pop("anchors", None)
    _write_json(man, man_path)
    tlog(f"seed {seed} done: {len(records)} pairs, NAV="
         f"{man['summary_overall']['NAV']:.4f} ⇒ {man['summary_overall']['verdict']}")
    return man


# ---------------------------------------------------------------------------
# Aggregate + plot
# ---------------------------------------------------------------------------


def _nanmean(vals) -> float:
    """``np.nanmean`` without the empty-slice warning.

    ⚠ Not cosmetic: ``jump_ratio`` is legitimately ``nan`` for the ``mean``-deleted
    arm (its steps are all exactly zero), so an all-``nan`` column is the **expected**
    reading of a dead control, ⛔ not a defect to be hidden.
    """
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], np.float64)
    return float(v.mean()) if v.size else float("nan")


def _records(sdir: Path, seed: int) -> list:
    p = Path(sdir) / f"seed_{int(seed)}" / f"records_s{int(seed)}.json"
    if not p.exists():
        return []
    with open(p) as fh:
        return json.load(fh).get("records", [])


def aggregate_runs(out_dir: Path, config: CHLUConfig, log: Callable = print) -> dict:
    """⭐ Across seeds.  ⛔ A seed that did not run is reported as ``NOT RUN``."""
    c = _cfg(config)
    per_seed, missing = {}, []
    for s in list(c.seeds):
        man = Path(out_dir) / f"seed_{int(s)}" / f"navigate_manifest_s{int(s)}.json"
        if not man.exists():
            missing.append(int(s))
            continue
        with open(man) as fh:
            per_seed[int(s)] = json.load(fh)
    seeds = sorted(per_seed)
    recs = {s: _records(out_dir, s) for s in seeds}
    pools = sorted({r["pool"] for s in seeds for r in recs[s]})
    factors = sorted({r["factor_name"] for s in seeds for r in recs[s]})

    def _nvr(pool=None, factor=None):
        """⛔ R31, fourth strike: a per-factor number is void without this."""
        vals = []
        for s_ in seeds:
            for row in per_seed[s_].get("selection", []):
                if pool is not None and row["pool"] != pool:
                    continue
                if factor is not None and row["factor_name"] != factor:
                    continue
                vals.append(int(row["n_values_realised"]))
        return sorted(set(vals)) or None

    def agg(sel, pool=None, factor=None) -> dict:
        rows = {s_: [r for r in recs[s_] if sel(r)] for s_ in seeds}
        summ = {s_: navigation_summary(rows[s_]) for s_ in seeds if rows[s_]}
        if not summ:
            return {"status": "NOT RUN", "n_qualifying_pairs": 0,
                    "n_values_realised": _nvr(pool, factor)}
        live = [s_ for s_ in seeds if rows[s_]]
        return {
            "NAV": _ms([summ[s_]["NAV"] for s_ in summ]),
            "traverse_rate": {a: _ms([summ[s_]["traverse_rate"].get(a) for s_ in summ])
                              for a in ARMS},
            "off_target_stability": {
                a: _ms([summ[s_]["mean_off_target_stability"].get(a) for s_ in summ])
                for a in ARMS},
            "n_qualifying_pairs": int(sum(summ[s_]["n_qualifying_pairs"] for s_ in summ)),
            "n_pairs_total": int(sum(len(rows[s_]) for s_ in seeds)),
            "n_values_realised": _nvr(pool, factor),
            "endpoint_codes_identical": _ms(
                [float(np.mean([r["endpoint_codes_identical"] for r in rows[s_]]))
                 for s_ in live]),
            "interior_codes_off_endpoints": _ms(
                [float(np.mean([r["arms"][STORE_ARM]["n_interior_codes_off_endpoints"]
                                for r in rows[s_]])) for s_ in live]),
            "endpoints_read_ok": {a: _ms([
                float(np.mean([bool(r["arms"][a]["endpoints_read_ok"]) for r in rows[s_]]))
                for s_ in live]) for a in ARMS},
            "visits_intermediate": {a: _ms([
                float(np.mean([bool(r["arms"][a]["visits_intermediate"]) for r in rows[s_]]))
                for s_ in live]) for a in ARMS},
            "jump_ratio": {a: _ms([
                _nanmean([r["arms"][a]["smoothness"]["jump_ratio"] for r in rows[s_]])
                for s_ in live]) for a in ARMS},
            "path_efficiency": {a: _ms([
                _nanmean([r["arms"][a]["smoothness"]["path_efficiency"] for r in rows[s_]])
                for s_ in live]) for a in ARMS},
            "off_target_constancy": {a: _ms([
                _nanmean([r["arms"][a]["off_target_constancy"] for r in rows[s_]])
                for s_ in live]) for a in ARMS},
            "endpoint_fidelity": {a: _ms([
                _nanmean([0.5 * (r["arms"][a]["endpoint_fidelity_a"]
                                 + r["arms"][a]["endpoint_fidelity_b"])
                          for r in rows[s_]]) for s_ in live]) for a in ARMS},
            "n_off_target_factors_that_move": {a: _ms([
                _nanmean([r["arms"][a]["n_off_target_factors_that_move"]
                          for r in rows[s_]]) for s_ in live]) for a in ARMS},
            "clean_switch_rate": {a: _ms([
                float(np.mean([bool(r["arms"][a]["clean_switch"]) for r in rows[s_]]))
                for s_ in live]) for a in ARMS},
        }

    summary = {
        "schema": "chlu-gen-navigate-summary/1",
        "seeds_run": seeds, "seeds_not_run": missing,
        "n_steps": int(c.n_steps),
        "dial": "navigation (R29 pivot item 4)",
        "overall": agg(lambda r: True),
        "by_pool": {p: agg(lambda r, p=p: r["pool"] == p, pool=p) for p in pools},
        "by_pool_factor": {
            f"{p}/{f}": agg(lambda r, p=p, f=f: r["pool"] == p and r["factor_name"] == f,
                            pool=p, factor=f)
            for p in pools for f in factors},
        "selection": {int(s_): per_seed[s_].get("selection", []) for s_ in seeds},
        "channel_by_arm": per_seed[seeds[0]]["channel_by_arm"] if seeds else {},
        "decoder": per_seed[seeds[0]]["decoder"] if seeds else {},
        "classifier": per_seed[seeds[0]]["classifier"] if seeds else {},
        "note": ("⛔ Every number carries BOTH controls (linear interpolation with "
                 "the store bypassed, and the mean-deleted store). ⛔ A per-factor "
                 "number is void without its n_values_realised (R31)."),
    }
    gp = summary["by_pool"].get("GRID", {})
    if gp and gp.get("n_qualifying_pairs"):
        lin = gp["traverse_rate"]["linear"]["mean"]
        sto = gp["traverse_rate"][STORE_ARM]["mean"]
        summary["verdict"] = (
            "⛔ THE STORE'S LANDSCAPE ADDS NOTHING OVER INTERPOLATION "
            f"(GRID: linear {lin:.3f} >= store {sto:.3f})" if lin >= sto else
            f"⭐ the store path out-traverses interpolation on GRID ({sto:.3f} > {lin:.3f})")
    else:
        summary["verdict"] = "NOT RUN — no qualifying GRID pair"
    _write_json(summary, Path(out_dir) / "summary.json")
    log(f"⭐ summary: seeds={seeds} not_run={missing} ⇒ {summary['verdict']}")
    return summary


def _plot(summary: dict, out_dir: Path, config: CHLUConfig) -> None:
    pools = [p for p in summary.get("by_pool", {})
             if summary["by_pool"][p].get("n_qualifying_pairs")]
    if not pools:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(pools))
    w = 0.2
    for i, a in enumerate(ARMS):
        m = [summary["by_pool"][p]["traverse_rate"][a]["mean"] for p in pools]
        e = [summary["by_pool"][p]["traverse_rate"][a]["std"] for p in pools]
        axes[0].bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=3, label=a)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pools)
    axes[0].set_ylabel("traverse rate")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("⭐ TRAVERSES, by arm (both controls on every bar)")
    axes[0].legend(fontsize=7)
    for i, a in enumerate(ARMS):
        m = [summary["by_pool"][p]["off_target_stability"][a]["mean"] for p in pools]
        e = [summary["by_pool"][p]["off_target_stability"][a]["std"] for p in pools]
        axes[1].bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=3, label=a)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pools)
    axes[1].set_ylabel("off-target stability")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.9, ls="--", c="k", lw=0.8)
    axes[1].set_title("a traversal that changes everything is not navigation")
    keys = [k for k in summary.get("by_pool_factor", {})
            if summary["by_pool_factor"][k].get("n_qualifying_pairs")]
    if keys:
        y = np.arange(len(keys))
        for i, a in (0, STORE_ARM), (1, "linear"):
            v = [summary["by_pool_factor"][k]["traverse_rate"][a]["mean"] for k in keys]
            axes[2].barh(y + (i - 0.5) * 0.4, v, 0.4, label=a)
        axes[2].set_yticks(y)
        axes[2].set_yticklabels(keys, fontsize=6)
        axes[2].set_xlabel("traverse rate")
        axes[2].set_xlim(0, 1)
        axes[2].set_title("by factor (⛔ read with n_values_realised)")
        axes[2].legend(fontsize=7)
    fig.suptitle(f"gen-navigation — {summary.get('verdict', '')}", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "navigation.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_experiment_gen_navigate(
    config: Optional[CHLUConfig] = None, quick: bool = False,
    out_dir: Optional[str] = None, seeds: Optional[list] = None,
    log: Callable = print,
) -> dict:
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    c = _cfg(config)
    root = Path(out_dir) if out_dir else (
        nav_out_root(config) / "runs" / f"{c.dataset}_{c.pretext}")
    root.mkdir(parents=True, exist_ok=True)
    if seeds is not None:
        c.seeds = [int(s) for s in seeds]
    log(f"[gen-navigate] dataset={c.dataset} pretext={c.pretext} seeds={c.seeds} "
        f"steps={c.n_steps} pools={c.pools} -> {root}")
    for s in list(c.seeds):
        run_seed(int(s), root, config, log=log)
    summary = aggregate_runs(root, config, log=log)
    if bool(c.make_plots):
        _plot(summary, root, config)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="⭐ The navigation dial: is the space BETWEEN the nine codes structured?")
    p.add_argument("--dataset", default=None, choices=["shapes3d", "dsprites"])
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS))
    p.add_argument("--seed", type=int, default=None, action="append", help="repeatable")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--pairs-per-factor", type=int, default=None)
    p.add_argument("--pool", default=None, action="append", choices=list(POOLS))
    p.add_argument("--out", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--quick", action="store_true",
                   help="⚠ plumbing smoke only — ⛔ a quick number is never reported")
    p.add_argument("--aggregate-only", action="store_true")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = get_default_config()
    c = _cfg(config)
    if args.dataset:
        c.dataset = args.dataset
    if args.pretext:
        c.pretext = args.pretext
    if args.steps:
        c.n_steps = int(args.steps)
    if args.pairs_per_factor:
        c.n_pairs_per_factor = int(args.pairs_per_factor)
    if args.pool:
        c.pools = list(args.pool)
    if args.data_root:
        # ⚠ §7 (gen-external-classifier): ``shapes3d_root(root)`` treats an explicit
        # root as the **dataset** directory (…/gen/shapes3d) while
        # ``$CHLU_GEN_DATA_ROOT`` is its **parent** (…/gen).  ⛔ Do NOT forward the
        # environment variable here — leaving ``data_root=None`` lets
        # ``default_gen_data_root()`` read it with the right meaning.  Setting it
        # from the env var costs a FileNotFoundError that looks like a missing
        # dataset; it bit this spoke on its first smoke run.
        config.gen_data.data_root = args.data_root
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    c = _cfg(config)
    if args.aggregate_only:
        root = Path(args.out) if args.out else (
            nav_out_root(config) / "runs" / f"{c.dataset}_{c.pretext}")
        s = aggregate_runs(root, config)
        if bool(c.make_plots):
            _plot(s, root, config)
        return 0
    run_experiment_gen_navigate(config, quick=bool(args.quick), out_dir=args.out,
                                seeds=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
