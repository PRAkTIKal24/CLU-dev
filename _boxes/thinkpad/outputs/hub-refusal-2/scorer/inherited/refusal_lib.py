"""hub/refusal -- the landscape refusal scorer.

WHAT THIS IS.  A scorer that settles the SHIPPED dynamical read on a request set and
reports, per request, a statistic built ONLY from the read's own settled energy and its
convergence.  Nothing here re-implements the read: the landscape is
``exp_gen_dynamical_read.build_landscape``'s, the chain is
``gen_landscape.settle_chain``, the per-term energy is ``gen_landscape.request_energy``.

NOT ONE MANIFEST-PINNED FILE IS EDITED.  This module lives entirely under
``$CHLU_HUB_OUTPUTS/hub-refusal/`` and imports the pin read-only.

THE ONE PIECE OF NEW MACHINERY, declared here and in PREREG section 2.
``gen_landscape.request_experts`` builds ONE EXPERT PER FACTOR from a dense ``(n, F)``
array of grid class indices.  That form cannot express either invalid kind:
``duplicate_factor`` names one factor twice, and ``out_of_range`` names a value that is
not on the grid and so has no class index.  :func:`term_experts` builds ONE EXPERT PER
REQUESTED TERM instead -- which is what ``request_energy``'s ``(T, n, K, N)`` contract
already asks for -- and it resolves an off-grid value by the SAME declared extrapolation
rule R8 gives the kNN rival: ``gen_request._extrapolating_index``, imported, not
re-derived.

Two properties that make this a generalisation and not a different arm:
  1. for an on-grid value ``_extrapolating_index`` returns ``w == 0`` (an exact hit), so
     the extrapolation is the identity and the renormalisation is a no-op;
  2. ``requests_from_rows`` emits terms in factor order, so for a VALID request the
     ``(T, n, K, N)`` tensor is elementwise equal to ``request_experts``'.
:func:`assert_reproduces_shipped_experts` checks (2) numerically on every valid unit.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CHLU_ROOT = os.environ["CHLU_ROOT"]
if CHLU_ROOT not in sys.path:
    sys.path.insert(0, CHLU_ROOT)

import jax  # noqa: E402

from chlu.config import (  # noqa: E402
    gen_dynamical_read_config,
    gen_row_split_config,
    get_default_config,
)
from chlu.core.gen_bridge import dataset_module, load_feature_cache, param_hash  # noqa: E402
from chlu.core.gen_landscape import (  # noqa: E402
    SettleCfg,
    arm_key,
    convergence_report,
    request_energy,
    request_experts,
    settle_chain,
    store_energy,
    total_energy,
    _softmax_last,
)
from chlu.core.gen_request import (  # noqa: E402
    _extrapolating_index,
    control_map_of,
    deduplicate_terms,
    load_c_requests,
    load_query_set_registry,
    load_registered_control_requests,
    requests_from_rows,
)
from chlu.core.gen_store import well_channels_from_labels  # noqa: E402
from chlu.data.gen_rowsplit import split_fit_val  # noqa: E402
from chlu.eval.gen_probe import factor_labels  # noqa: E402
from chlu.experiments.exp_gen_cell import load_store_plan, store_plan_path  # noqa: E402
from chlu.experiments.exp_gen_dream import resolve_grouping  # noqa: E402
from chlu.experiments.exp_gen_dynamical_read import build_landscape  # noqa: E402
from chlu.experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from chlu.utils.checkpoints import load_model  # noqa: E402

CLAUDE_ROOT = Path(os.environ["CHLU_CLAUDE_ROOT"])
DATA_ROOT = Path(os.environ["CHLU_GEN_DATA_ROOT"])
OUT = Path(os.environ["CHLU_HUB_OUTPUTS"]) / "hub-refusal"

DATASET, PRETEXT = "shapes3d", "recon"
#: The MATCHED rung -- 64 atoms/well, rank 16, 524,288 B = the deployed byte target.
HEADLINE_BUDGET = "A"
#: Steps per rung, exactly ``exp_gen_cell.DYNREAD_STEPS``.
DYNREAD_STEPS = {"A": 1000, "B": 2000}
#: Term slots.  6 factors + at most one duplicate term (``DenseRequests``' contract).
T_MAX = 7


def tlog(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# The one piece of new machinery -- one expert per requested TERM
# ---------------------------------------------------------------------------


def _logsumexp_last(A: np.ndarray) -> np.ndarray:
    m = A.max(axis=-1, keepdims=True)
    return m + np.log(np.exp(A - m).sum(axis=-1, keepdims=True))


def term_experts(book, grid, requests, t_max: int = T_MAX) -> tuple:
    """``requests -> ((T, n, K, N) log-weights, (n,) n_terms)``.

    One expert per requested term.  A term ``(f, v)``'s expert is
    ``logsoftmax_a[(1-w) * log_cond[f][i] + w * log_cond[f][j]] - log_prior``, with
    ``(i, j, w)`` from the SHIPPED ``gen_request._extrapolating_index``.

    The renormalisation (``logsoftmax`` over the well's atoms) is declared, and it
    matters: ``log_cond[f][v]`` is a distribution over the atoms, and a log-linear
    EXTRAPOLATION of two distributions is not normalised.  Without it the term energy is
    no longer ``-sigma^2 log E_p[e^W]`` with ``W`` a genuine likelihood ratio, and an
    off-grid value could be made to look MORE likely by an unnormalised spike.
    On-grid (``w == 0``) it is the identity, so no valid request is touched by it.
    """
    lp = np.asarray(book.log_prior, np.float64)  # (K, N)
    K, N = lp.shape
    n = len(requests)
    out = np.zeros((int(t_max), n, K, N), np.float32)
    n_terms = np.zeros(n, np.int64)
    n_offgrid = 0
    for i, req in enumerate(requests):
        terms = req["request"]
        if len(terms) > int(t_max):
            raise ValueError(
                f"request {req.get('id')!r} has {len(terms)} terms > t_max={t_max}")
        n_terms[i] = len(terms)
        for t, term in enumerate(terms):
            f = int(term["factor_index"])
            v = float(term["value"])
            vals = np.asarray(grid.values[f], np.float64)
            a, b, w = _extrapolating_index(vals, v)
            lc = np.asarray(book.log_cond[f], np.float64)  # (V, K, N)
            if a == b or w == 0.0:
                L = lc[a]
            else:
                n_offgrid += 1
                L = (1.0 - w) * lc[a] + w * lc[b]
                L = L - _logsumexp_last(L)
            out[t, i] = (L - lp).astype(np.float32)
    return out, n_terms, {"n_offgrid_terms": int(n_offgrid)}


def assert_reproduces_shipped_experts(book, requests, log_w) -> dict:
    """For a VALID one-term-per-factor request the term tensor must equal
    ``request_experts``'.  This is the proof that this hub did not quietly build a
    different arm for the accept half.
    """
    if any("class_indices" not in r for r in requests):
        return {"skipped": "not a row-derived valid request set"}
    R = np.asarray([[int(c) for c in r["class_indices"]] for r in requests], np.int64)
    ref = np.asarray(request_experts(book, R), np.float32)
    got = np.asarray(log_w, np.float32)[: ref.shape[0]]
    d = float(np.max(np.abs(got - ref))) if ref.size else 0.0
    return {"max_abs_delta_vs_request_experts": d,
            "bit_identical": bool(np.array_equal(got, ref)),
            "shape_ref": list(ref.shape), "shape_got": list(got.shape)}


def launch_from_terms(land, book, log_w, n_terms, clamp_threshold: float = 0.5) -> dict:
    """``clamp_atom`` launch generalised to terms.

    Identical to ``gen_landscape.launch_state`` for a valid request: the posterior there
    is ``log_prior + sum_f (log_cond[f][v] - log_prior)``, which is this sum with one
    term per factor.
    """
    lp = np.asarray(book.log_prior, np.float64)
    n = int(log_w.shape[1])
    S = np.broadcast_to(lp, (n,) + lp.shape).copy()
    for t in range(int(log_w.shape[0])):
        active = (n_terms > t)[:, None, None]
        S = S + np.where(active, np.asarray(log_w[t], np.float64), 0.0)
    P = _softmax_last(S)
    conf = P.max(axis=-1)
    idx = P.argmax(axis=-1)
    store = land.store
    x0 = np.zeros((n, store.dim), np.float32)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        A = np.asarray(store.atoms[k], np.float32)
        x0[:, cols] = A[idx[:, k]]
    clamped = conf >= float(clamp_threshold)
    return {"x0": x0, "atom_index": idx.astype(np.int64),
            "confidence": conf.astype(np.float32),
            "mean_n_clamped": float(clamped.sum(axis=1).mean()),
            "mean_confidence": float(conf.mean()), "n_wells": int(store.n_wells)}


# ---------------------------------------------------------------------------
# The statistics -- ALL of them functions of the settled energy and the trace
# ---------------------------------------------------------------------------


def per_term_residuals(land, X, log_w, n_terms, sigma_scale: float = 1.0) -> np.ndarray:
    """``(n, T)`` ``E_t(x) - E_store(x)``, one column per term slot.

    Computed by calling the SHIPPED ``request_energy`` with a one-term tensor, so the
    arithmetic is the harness's and not a re-derivation.  Inactive slots are ``nan``.
    """
    Xj = np.asarray(X, np.float32)
    T = int(log_w.shape[0])
    out = np.full((int(Xj.shape[0]), T), np.nan, np.float64)
    for t in range(T):
        lw1 = np.ascontiguousarray(np.asarray(log_w[t: t + 1], np.float32))
        r = np.asarray(request_energy(land, Xj, lw1, sigma_scale), np.float64)
        out[:, t] = np.where(n_terms > t, r, np.nan)
    return out


def energy_parts(land, X, log_w, n_terms, sigma_scale: float = 1.0) -> dict:
    """The settled energy, decomposed exactly as ``total_energy`` composes it."""
    Xj = np.asarray(X, np.float32)
    e_store = np.asarray(store_energy(land, Xj, sigma_scale), np.float64)
    e_tot = np.asarray(total_energy(land, Xj, log_w, sigma_scale), np.float64)
    res = per_term_residuals(land, Xj, log_w, n_terms, sigma_scale)
    with np.errstate(invalid="ignore"):
        s_max = np.nanmax(res, axis=1)
        s_sum = np.nansum(res, axis=1)
        s_mean = np.nanmean(res, axis=1)
    return {"e_store": e_store, "e_total": e_tot, "residuals": res,
            "S_max": s_max, "S_sum": s_sum, "S_mean": s_mean}


def per_request_convergence(trace: dict, window: int, tol: float) -> dict:
    """The PER-REQUEST form of ``convergence_report``'s pooled statistic.

    ``rel_tail_drop[i]`` is exactly the ``rel`` the shipped report thresholds, kept per
    chain instead of averaged; ``still_descending[i] = rel[i] < -tol`` reproduces the
    shipped boolean chain by chain.
    """
    E = np.asarray(trace["energy_ref"], np.float64)
    w = int(min(max(int(window), 2), E.shape[0]))
    tail = E[-w:]
    t = np.arange(w, dtype=np.float64)
    t = t - t.mean()
    slope = (t[:, None] * (tail - tail.mean(axis=0, keepdims=True))).sum(axis=0) / (t * t).sum()
    scale = np.maximum(np.abs(tail).mean(axis=0), 1e-12)
    rel = slope * w / scale
    disp = np.asarray(trace["displacement"], np.float64)
    nanv = np.full(E.shape[1], np.nan)
    return {"rel_tail_drop": rel,
            "still_descending": (rel < -float(tol)),
            "energy_ref_final": E[-1].copy(),
            "energy_ref_first": E[0].copy(),
            "displacement_final": disp[-1].copy() if disp.size else nanv,
            "total_displacement": disp.sum(axis=0) if disp.size else nanv}


# ---------------------------------------------------------------------------
# The cell context -- mirrors exp_gen_cell.run_seed up to the well partition
# ---------------------------------------------------------------------------


def build_context(seed: int, config=None) -> dict:
    """Rows, cache, well partition.  Nothing is re-encoded and nothing is extended."""
    config = config or get_default_config()
    cfg = config.gen_cell
    # The shipped resolution: ``gen_data.data_root = None`` -> ``default_gen_data_root()``
    # -> ``$CHLU_GEN_DATA_ROOT``.  Overriding it here would point ``shapes3d_root`` at
    # the parent of the dataset directory and the h5 would not be found.
    t0 = time.time()
    mod = dataset_module(DATASET)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    rows = select_store_rows(split, config, int(seed))
    pool = np.asarray(rows["store_train"], np.int64)
    rs = gen_row_split_config(config)
    fit_rows, val_rows, split_info = split_fit_val(
        pool, grid.factor_indices_of(pool), int(cfg.n_fit), int(cfg.n_val), int(seed),
        mode=str(rs.mode), tiebreak_jitter=float(rs.tiebreak_jitter),
        seed_mult=int(rs.seed_mult), seed_offset=int(rs.seed_offset), log=tlog)
    if int(val_rows.size) != int(cfg.n_val):
        raise AssertionError(
            f"SILENT TRUNCATION: n_val realised {val_rows.size} != requested {cfg.n_val}")

    bdir = bridge_dir(config, DATASET, PRETEXT, int(seed))
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    crows, feats_cache, cman = load_feature_cache(
        bdir, verify=bool(cfg.verify_cache_checksum))
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    config.gen_store.encode_chunk = int(cfg.encode_chunk)
    # The re-encode gate, called through its OWN declared scoping API.
    #   * ``will_extend=False`` because this hub reads the cache and never writes it:
    #     it gathers only ``store_train`` rows, which the cache already holds, and it
    #     needs NO feature at all for C / A7 (a read is request-only).
    #   * This is NOT a weakened check.  An unenforced pass deliberately does not
    #     register in ``_REENCODE_VERIFIED``, so if any uncached row reached
    #     ``gather_features`` the check would re-run there AT FULL STRENGTH and abort.
    #   * ``would_have_failed`` is recorded verbatim in every manifest this hub writes.
    #     On this box it DOES fail the enforced bound, at ~51 float16 ULP on ~0.98% of
    #     elements -- which is charter declared property #3 ("CPU/GPU feature agreement
    #     is ~53 ULP on ~1% of elements, not bit-exact") measured cross-ISA against a
    #     Mac-produced cache.  It is reported, never waved away.
    reenc = _assert_cache_reencodes(
        DATASET, config, encoder, crows, feats_cache, cman, log=tlog,
        will_extend=False,
        reason=("hub/refusal reads the frozen cache and never extends it: it gathers "
                "only store_train rows (already cached) and needs no feature for the "
                "invalid or control request sets, because the read is request-only. "
                "The enforced bound is re-applied at full strength by gather_features "
                "if any uncached row is ever requested."))

    X_store = gather_features(DATASET, config, encoder, crows, feats_cache, cman, pool,
                              log=tlog)
    lab_store = factor_labels(grid, pool)
    plan = load_store_plan(store_plan_path(config, DATASET, PRETEXT))
    K = int(plan["K"])
    config.gen_dream.grouping_source = str(cfg.grouping_source)
    labels_vec, ginfo = resolve_grouping(plan, X_store, config, int(seed),
                                         bman["encoder_param_hash"], log=tlog)
    wc = well_channels_from_labels(labels_vec, K)
    ms_rows = matched_seen_rows(DATASET, split, values, pool)

    return {"config": config, "seed": int(seed), "split": split, "grid": grid,
            "values": values,
            "names": list(grid.names), "cards": list(grid.cardinalities),
            "feat_shape": feat_shape, "pool": pool,
            "fit_rows": fit_rows, "val_rows": val_rows,
            "eval_a": np.asarray(rows["eval_a"], np.int64),
            "eval_b": np.asarray(rows["eval_b"], np.int64),
            "ms_rows": np.asarray(ms_rows, np.int64),
            "X_store": X_store, "lab_store": lab_store, "well_channels": wc,
            "encoder_param_hash": enc_hash,
            "grouping": {k: v for k, v in ginfo.items() if not hasattr(v, "shape")},
            "row_split": {k: v for k, v in split_info.items() if not hasattr(v, "shape")},
            "reencode_check": reenc,
            "cache_meta": {k: cman.get(k) for k in ("n_rows", "scope", "D", "dtype")},
            "build_s": float(time.time() - t0)}


def request_sets(ctx: dict, which) -> dict:
    """The request DICT lists, by set name.  C / A7 are read from disk verbatim."""
    grid = ctx["grid"]
    which = set(which)
    out = {}
    if "val" in which:
        out["val"] = requests_from_rows(grid, ctx["val_rows"])
    if "A" in which:
        out["A"] = requests_from_rows(grid, ctx["eval_a"])
    if "B" in which:
        out["B"] = requests_from_rows(grid, ctx["eval_b"])
    if "MS" in which:
        out["MS"] = requests_from_rows(grid, ctx["ms_rows"])
    if "C" in which:
        out["C"] = load_c_requests(DATA_ROOT / DATASET / "c_invalid.json")
    if which & {"A7", "A7D"}:
        reg = load_query_set_registry(DATA_ROOT / DATASET / "query_sets.json")
        cmap = control_map_of(reg)
        named = sorted({v for v in cmap.values() if v != "base"})
        loaded = load_registered_control_requests(DATA_ROOT / DATASET, reg, named)
        a7 = loaded.get("a7_redundant")
        if a7 is None:
            raise FileNotFoundError("a7_redundant is not file-backed in the registry")
        if "A7" in which:
            out["A7"] = a7
        if "A7D" in which:
            out["A7D"] = deduplicate_terms(a7)
    return out


# ---------------------------------------------------------------------------
# The charter section C.2 control -- LANDSCAPE REMOVED
# ---------------------------------------------------------------------------


def shuffled_store(store, seed: int = 0):
    """The atoms' CONTENT destroyed, everything else held fixed (charter section C.2).

    Each channel column of each well's atom matrix is permuted independently among the
    ``N`` atoms, with a seeded RNG.  Consequences, all of them deliberate:
      * ``store_bytes`` is unchanged to the byte -- the control is byte-matched;
      * every channel's marginal over the atoms is the SAME MULTISET of values;
      * no atom is a train row any more, so the joint structure -- the landscape's
        entire content -- is gone, and ``assert_atoms_are_train_rows`` would convict it.
    The address book, the bandwidths and the coupling are untouched: exactly ONE declared
    quantity differs from the full arm.
    """
    from chlu.core.gen_landscape import AtomStore

    rng = np.random.default_rng(int(seed) * 9176 + 20260916)
    N = int(store.n_atoms)
    Z = np.zeros((N, int(store.dim)), np.float32)
    for k, w in enumerate(store.well_channels):
        cols = np.asarray(w, np.int64)
        A = np.asarray(store.atoms[k], np.float32).copy()
        for j in range(A.shape[1]):
            A[:, j] = A[rng.permutation(N), j]
        Z[:, cols] = A
    return AtomStore(store.well_channels, store.feat_shape, Z, np.arange(N, dtype=np.int64),
                     selection=f"SHUFFLED_CONTROL(from {store.selection})")


# ---------------------------------------------------------------------------
# The landscape and one settling unit
# ---------------------------------------------------------------------------


def build_land(ctx: dict, budget: str = HEADLINE_BUDGET, log=tlog) -> dict:
    """``exp_gen_dynamical_read.build_landscape``, called -- never re-implemented."""
    return build_landscape(ctx["config"], str(budget), ctx["X_store"], ctx["lab_store"],
                           ctx["well_channels"], ctx["feat_shape"], ctx["cards"],
                           int(ctx["seed"]), log=log)


def unit_key(seed: int, budget: str, set_name: str, arm: str, config):
    """The harness's own key derivation, reproduced exactly (``arm_key``/``fold_in``).

    ``exp_gen_cell.run_dynread_budget``: ``PRNGKey(row_seed_offset + seed)`` ->
    ``arm_key(., "budget_<b>")`` -> ``arm_key(., <request set>)``; then
    ``run_arm``: ``arm_key(., <inner arm>)`` -> ``fold_in(., chunk_offset)``.
    """
    c = gen_dynamical_read_config(config)
    k0 = jax.random.PRNGKey(int(c.row_seed_offset) + int(seed))
    return arm_key(arm_key(arm_key(k0, f"budget_{budget}"), str(set_name)), str(arm))


def settle_unit(ctx: dict, built: dict, set_name: str, requests, arm: str = "full",
                budget: str = HEADLINE_BUDGET, steps: int | None = None,
                chunk: int | None = None, log=tlog) -> dict:
    """One resumable unit: settle ``requests`` and return every per-request statistic.

    ``arm``: ``full`` (beta = 1), ``request_off`` (beta = 0, the steering control),
    ``zero_step`` (the launch, decoded -- the laundering control).
    """
    config = ctx["config"]
    c = gen_dynamical_read_config(config)
    n_steps = int(DYNREAD_STEPS[str(budget)] if steps is None else steps)
    chunk = int(c.request_chunk if chunk is None else chunk)
    land0 = built["land"]
    book = built["book"]

    log_w, n_terms, tinfo = term_experts(book, ctx["grid"], requests, T_MAX)
    repro = assert_reproduces_shipped_experts(book, requests, log_w)

    # Arms COMPOSE, so a control can be applied at the launch as well as at the settled
    # state: "store_removed_zero_step" is the landscape-removed control evaluated at the
    # launch, which is the right floor check for a launch-time statistic.
    land = land0
    if "request_off" in arm:
        land = land.with_strengths(request_strength=0.0)
    if "store_removed" in arm:
        from chlu.core.gen_landscape import AtomLandscape
        land = AtomLandscape(shuffled_store(land.store, int(ctx["seed"])),
                             np.asarray(land.sigma), land.coupling,
                             float(land.request_strength),
                             float(land.coupling_strength))
    cfg = SettleCfg(steps=(0 if "zero_step" in arm else n_steps),
                    step_size=float(c.step_size),
                    sigma_scale_start=float(c.sigma_scale_start),
                    sigma_scale_end=float(c.sigma_scale_end),
                    temp_start=float(c.temp_start), temp_end=float(c.temp_end),
                    metropolis=bool(c.metropolis))
    key = unit_key(int(ctx["seed"]), budget, set_name, arm, config)

    t0 = time.time()
    xs_parts, x0_parts, traces = [], [], []
    n = len(requests)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        lw = np.ascontiguousarray(log_w[:, sl])
        nt = n_terms[sl]
        lau = launch_from_terms(land, book, lw, nt, float(c.clamp_threshold))
        out = settle_chain(land, lau["x0"], lw, cfg, jax.random.fold_in(key, i))
        x0_parts.append(lau["x0"])
        xs_parts.append(np.asarray(out["x"], np.float32))
        traces.append(out["trace"])
        log(f"      [{set_name}/{arm}] chunk {i}..{sl.stop} done "
            f"({time.time() - t0:.1f}s)")
    x0 = np.concatenate(x0_parts, axis=0)
    xs = np.concatenate(xs_parts, axis=0)
    trace = {k: (np.concatenate([np.asarray(t[k]) for t in traces], axis=1)
                 if np.asarray(traces[0][k]).ndim == 2 else np.asarray(traces[0][k]))
             for k in ("energy", "energy_ref", "accept_prob", "accepted", "displacement")}
    trace["sigma_scale"] = np.asarray(traces[0]["sigma_scale"])
    trace["temperature"] = np.asarray(traces[0]["temperature"])

    # -- the statistics, at the REFERENCE bandwidth (the only one differenceable) --
    sig_ref = float(c.sigma_scale_end)
    settled = energy_parts(land, xs, log_w, n_terms, sig_ref)
    launch = energy_parts(land, x0, log_w, n_terms, sig_ref)
    if cfg.steps > 0:
        conv = per_request_convergence(trace, int(c.descent_window), float(c.descent_tol))
        pooled = convergence_report(trace, int(c.descent_window), float(c.descent_tol))
    else:
        nanv = np.full(n, np.nan)
        conv = {"rel_tail_drop": nanv, "still_descending": np.zeros(n, bool),
                "energy_ref_final": settled["e_total"], "energy_ref_first": settled["e_total"],
                "displacement_final": nanv, "total_displacement": np.zeros(n)}
        pooled = convergence_report(trace, int(c.descent_window), float(c.descent_tol))

    radii = np.asarray([np.asarray(a).std() for a in land.store.atoms], np.float64)
    ref = float(np.sqrt(np.mean(radii ** 2))) or 1.0
    rms_state = float(np.sqrt(np.mean(xs ** 2)))
    diverged = bool((not np.isfinite(rms_state))
                    or rms_state > float(c.divergence_radius_mult) * ref)

    return {
        "meta": {"seed": int(ctx["seed"]), "budget": str(budget), "set": str(set_name),
                 "arm": str(arm), "n": int(n), "steps": int(cfg.steps),
                 "chunk": int(chunk), "sigma_ref": sig_ref,
                 "request_strength": float(land.request_strength),
                 "coupling_strength": float(land.coupling_strength),
                 "term_info": tinfo, "experts_reproduction": repro,
                 "wall_clock_s": float(time.time() - t0),
                 "diverged": diverged, "settled_rms": rms_state,
                 "atom_rms_radius": ref,
                 "pooled_convergence": pooled},
        "ids": [str(r["id"]) for r in requests],
        "kinds": [str(r.get("kind", "valid")) for r in requests],
        "n_terms": n_terms,
        "settled": settled, "launch": launch, "conv": conv,
        "x_settled": xs, "x0": x0, "trace": trace,
    }


def save_unit(rec: dict, path: Path) -> dict:
    """Persist a unit the moment its chain finishes -- a kill costs ONE UNIT."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(rec["ids"]), kinds=np.asarray(rec["kinds"]),
        n_terms=rec["n_terms"],
        S_max=rec["settled"]["S_max"], S_sum=rec["settled"]["S_sum"],
        S_mean=rec["settled"]["S_mean"],
        e_store=rec["settled"]["e_store"], e_total=rec["settled"]["e_total"],
        residuals=rec["settled"]["residuals"],
        launch_S_max=rec["launch"]["S_max"], launch_S_sum=rec["launch"]["S_sum"],
        launch_e_total=rec["launch"]["e_total"], launch_e_store=rec["launch"]["e_store"],
        rel_tail_drop=rec["conv"]["rel_tail_drop"],
        still_descending=rec["conv"]["still_descending"],
        energy_ref_final=rec["conv"]["energy_ref_final"],
        energy_ref_first=rec["conv"]["energy_ref_first"],
        displacement_final=rec["conv"]["displacement_final"],
        total_displacement=rec["conv"]["total_displacement"],
        x_settled=rec["x_settled"].astype(np.float32),
        x0=rec["x0"].astype(np.float32),
        meta_json=np.asarray(json.dumps(rec["meta"], default=str)),
    )
    return {"path": str(path), "bytes": int(path.stat().st_size)}
