"""Experiment: ⭐ **the dynamical read** — ``request -> clamp + request ENERGY ->
launch -> settle over the WRITTEN ATOMS -> decode the settled continuous state``.

⭐ **DIAL DECLARATION — *the read*.**
- **Dial:** continuous settling over stored atoms, replacing codebook copy-back.
- **Laundering control:** ⭐ **request-off on the SAME chain** (same launch, same
  key, same landscape, ``request_strength = 0``).  **Steering-attributable gain is
  the falsifier.**  ⛔ Dreaming's exact lesson: its clamp's apparent gain was taken
  *entirely* by the coupling-off control.  ``coupling_off`` is mandatory beside it.
- **Falsifies:** the three falsifiers in this spoke's ``PREREG.md`` §5.
- **Does NOT falsify:** losing to ``knn_proto`` on a metric-native protocol (the
  metric-native-ceiling theorem, not news).

⛔⛔ **BUILD ONLY. THE SCORED RUN IS GATED AND THIS MODULE RENDERS NO VERDICT.**
It emits the path, the controls, the byte ledger, the bypass assertion and the
dynamics trace.  It deliberately does **not** compute falsifier answers, run three
seeds, or call the external classifier: those belong to the scored run, after
``PREREG.md`` is signed.

⛔ **The three non-negotiables it implements, from the rulings:**
1. ⭐ **Atoms = stored TRAIN DATA POINTS**, proved bit-for-bit by
   :func:`chlu.core.gen_landscape.assert_atoms_are_train_rows`.  ⛔ Never the
   ``k``-means centres — a landscape over discrete minima inherits the codebook loss.
2. ⭐ **The request enters as an ENERGY TERM** at a declared strength, one expert per
   requested term over the well's atoms — ⛔ not only as an initialisation.  The
   request-off control is wired **on the same chain** from the start.
3. ⭐ **The bypass guarantee covers LAUNCH + SETTLE + READ AS ONE PATH**, and it is
   demonstrated **failing** on deliberately bypassed variants at each of the three
   stages.  ⛔ ``soft_read(store, X, τ)`` takes FEATURES and would void it.

⚠ **Three environment facts** (paid for already by R31/R34/`gen-loss-attribution`):
a worktree has no ``.claude/``, so ``$CHLU_GEN_DATA_ROOT``, ``$CHLU_CLAUDE_ROOT``
**and** ``$CHLU_GEN_BRIDGE_ROOT`` must all be set, or the run dies at ``bridge_dir``.
⚠ Feature caches are NON-UNIFORM: seed 0 is ``scope="full"``, seeds 1-2 ``"used"``.
⚠ JAX cold start is ~20 min on this box and it has 16 GB — ⛔ never start a full suite.

Runnable::

    python -m chlu.experiments.exp_gen_dynamical_read --quick   # ⚠ plumbing smoke only
    python -m chlu.experiments.exp_gen_dynamical_read
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, gen_dynamical_read_config, get_default_config  # noqa: E402
from ..core.gen_bridge import PRETEXTS, load_feature_cache, param_hash  # noqa: E402
from ..core.gen_dream import (  # noqa: E402
    WINNING_DECODER_RECIPE,
    assert_dream_decoder,
    requests_from_labels,
)
from ..core.gen_landscape import (  # noqa: E402
    AtomLandscape,
    SettleCfg,
    arm_key,
    assert_atoms_are_train_rows,
    assert_request_only,
    atom_bandwidths,
    build_atom_store,
    convergence_report,
    decode_width,
    dynamical_read,
    fit_atom_address_book,
    fit_low_rank_coupling,
    landscape_byte_ledger,
    pairwise_separation,
    safe_coupling_strength,
    save_trace,
    settled_state_spread,
)
from ..core.gen_store import address_codes, read_from_codes, store_bytes, well_channels_from_labels  # noqa: E402
from ..eval.gen_attribution import n_values_realised  # noqa: E402
from ..eval.gen_probe import decode_features, factor_labels  # noqa: E402
from ..experiments.exp_gen_attribution import load_attrib_store, load_grouping  # noqa: E402
from ..experiments.exp_gen_compose import _claude_root, _write_json  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    select_store_rows,
)
from ..utils.checkpoints import load_model  # noqa: E402

#: The request sets.  ⭐ ``CORNER_B`` is the phase's "seventy-two questions" and is
#: always run; ``SPREAD`` is the global row set, capped by config.
REQUEST_SETS = ("CORNER_B", "SPREAD")


def _cfg(config: CHLUConfig):
    return gen_dynamical_read_config(config)


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only.  ⛔ A ``--quick`` number is NEVER reported.

    ⛔ ``N`` and ``r`` are deliberately **not** reduced: the byte ledger is a
    PRE-REGISTERED number and a smoke that shrinks it proves nothing about it.
    """
    c = _cfg(config)
    c.seeds = list(c.quick_seeds)
    c.budgets = list(c.quick_budgets)
    c.n_store_train = int(c.quick_n_store_train)
    c.n_coupling_fit = int(c.quick_n_coupling_fit)
    c.n_spread_requests = int(c.quick_n_spread_requests)
    c.steps = int(c.quick_steps)
    c.descent_window = int(c.quick_descent_window)
    c.decode_rows = int(c.quick_decode_rows)
    c.request_chunk = int(c.quick_request_chunk)
    c.verify_cache_checksum = bool(c.quick_verify_cache_checksum)
    c.assert_byte_ledger = bool(c.quick_assert_byte_ledger)
    return config


def dynread_out_root(config: CHLUConfig) -> Path:
    c = _cfg(config)
    if c.out_root:
        return Path(c.out_root).expanduser()
    return _claude_root(config) / "outputs" / "gen-dynamical-read"


def _abs(config: CHLUConfig, p: str) -> Path:
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else _claude_root(config) / q


# ---------------------------------------------------------------------------
# 1. Rows
# ---------------------------------------------------------------------------


def select_rows(config: CHLUConfig, dataset: str, split, seed: int) -> dict:
    """``store_train`` (the atom pool + the coupling's fit set) and the request rows.

    ⛔ ``eval_a``/``eval_b`` are outside the pool structurally, so a request is never
    a row the atoms were drawn from.
    """
    c = _cfg(config)
    if int(c.n_store_train):
        config.gen_store.n_store_train = int(c.n_store_train)
    rows = select_store_rows(split, config, int(seed))
    pool = np.setdiff1d(np.asarray(split.train_rows, np.int64), rows["store_train"])
    rng = np.random.default_rng(int(seed) * int(c.row_seed_mult) + int(c.row_seed_offset))
    n = min(int(c.n_spread_requests), pool.size)
    # ⛔ A PREFIX of a sorted row set is NOT a subsample: the Shapes3D grid is a
    # lexicographic bijection, so leading rows restrict the OUTERMOST factor (R31's
    # erratum). Every subsample here is a seeded permutation.
    spread = np.sort(rng.permutation(pool)[:n]) if n else np.zeros((0,), np.int64)
    return {
        "store_train": rows["store_train"],
        "CORNER_B": np.asarray(rows["eval_b"], np.int64),
        "SPREAD": spread,
    }


# ---------------------------------------------------------------------------
# 2. One budget: build the landscape, run the arms, decode
# ---------------------------------------------------------------------------


def build_landscape(config: CHLUConfig, budget: str, X_train: np.ndarray,
                    Y_train: np.ndarray, well_channels, feat_shape, cards,
                    seed: int, log: Callable = print) -> dict:
    """⭐ Atoms, bandwidths, address book, low-rank coupling — and the byte ledger.

    ⛔ Every assertion that can be made *before* a chain is run is made here: the
    atoms are train rows, the ledger matches ``PREREG.md`` §3, and the declared
    channel is the store's float width.
    """
    c = _cfg(config)
    spec = c.budget_spec(budget)
    t0 = time.time()
    store = build_atom_store(well_channels, feat_shape, X_train, int(spec["n_atoms"]),
                             str(c.atom_selection), int(seed))
    atoms_ok = assert_atoms_are_train_rows(store, X_train)
    sigma = atom_bandwidths(store, float(c.bandwidth_frac), str(c.bandwidth_rule))
    book = fit_atom_address_book(store, X_train, Y_train, cards, sigma,
                                 float(c.address_smoothing))
    nfit = min(int(c.n_coupling_fit), X_train.shape[0])
    sel = np.sort(np.random.default_rng(
        int(seed) * int(c.row_seed_mult) + int(c.row_seed_offset) + 313
    ).permutation(X_train.shape[0])[:nfit])
    coupling = fit_low_rank_coupling(store, X_train[sel], int(spec["rank"]),
                                     bool(c.coupling_normalise))
    land = AtomLandscape(store, sigma, coupling, float(c.request_strength),
                         float(c.coupling_strength))
    # ⛔⭐ THE COERCIVITY CERTIFICATE.  A bilinear coupling is an indefinite
    # quadratic form: on this spoke's first smoke the lambda = 1 chain ran to
    # E = -1.6e28 with a per-step displacement of 8e11.  The strength actually used
    # is capped by the atoms' own geometry, and the cap is REPORTED.
    cert = safe_coupling_strength(land, float(c.coupling_strength),
                                  float(c.coupling_safety_frac))
    if bool(c.enforce_coupling_certificate):
        land = land.with_strengths(coupling_strength=float(cert["coupling_strength_used"]))
    if cert["certificate_bound_the_strength"]:
        log(f"    [{spec['label']}] ⚠ coupling strength CAPPED BY THE CERTIFICATE: "
            f"declared {cert['declared_coupling_strength']:g} -> used "
            f"{cert['coupling_strength_used']:.4g} (lambda* = "
            f"{cert['max_safe_coupling_strength']:.4g})")
    else:
        log(f"    [{spec['label']}] coupling certificate: lambda* = "
            f"{cert['max_safe_coupling_strength']:.4g}, using "
            f"{cert['coupling_strength_used']:g} (bounded below="
            f"{cert['certificate']['bounded_below']})")
    ledger = landscape_byte_ledger(int(spec["n_atoms"]), store.well_sizes,
                                   int(spec["rank"]), int(spec["budget_bytes"]),
                                   label=f"budget {spec['label']} ({spec['role']})")
    measured = {"store_bytes": int(store.n_bytes()),
                "coupling_bytes": int(coupling.n_bytes())}
    if measured["store_bytes"] != ledger["store"]["store_bytes"]:
        raise AssertionError(
            f"⛔ the store's measured bytes {measured['store_bytes']} do not match "
            f"the declared ledger {ledger['store']['store_bytes']}")
    if measured["coupling_bytes"] != ledger["coupling"]["coupling_bytes"]:
        raise AssertionError(
            f"⛔ the coupling's measured bytes {measured['coupling_bytes']} do not "
            f"match the declared ledger {ledger['coupling']['coupling_bytes']}")
    exp = c.byte_ledger_expectations(budget)
    got = {
        "store_bytes": int(ledger["store"]["store_bytes"]),
        "coupling_bytes": int(ledger["coupling"]["coupling_bytes"]),
        "total_bytes": int(ledger["total_bytes"]),
    }
    ledger_matches = {k: bool(got[k] == int(exp[k])) for k in got}
    ledger_matches["multiple_of_budget"] = bool(
        abs(float(ledger["multiple_of_budget"]) - float(exp["multiple_of_budget"])) < 1e-9)
    if bool(c.assert_byte_ledger) and not all(ledger_matches.values()):
        raise AssertionError(
            "⛔ THE BYTE LEDGER DOES NOT MATCH PREREG §3 — declared before the run "
            f"and therefore binding. expected={exp} got store="
            f"{ledger['store']['store_bytes']} coupling="
            f"{ledger['coupling']['coupling_bytes']} total={ledger['total_bytes']} "
            f"({ledger['multiple_of_budget']:.4f}x budget)")
    log(f"    [{spec['label']}] N={spec['n_atoms']} r={spec['rank']} "
        f"store={ledger['store']['store_bytes']:,} B coupling="
        f"{ledger['coupling']['coupling_bytes']:,} B total={ledger['total_bytes']:,} B "
        f"= {ledger['multiple_of_budget']:.2f}x budget "
        f"(PREREG §3 match: {all(ledger_matches.values())})")
    log(f"    [{spec['label']}] sigma per well = "
        + " ".join(f"{float(s):.4f}" for s in sigma))
    return {
        "spec": spec, "store": store, "book": book, "coupling": coupling,
        "land": land, "sigma": [float(s) for s in sigma],
        "atoms_are_train_rows": atoms_ok,
        "coupling_certificate": cert,
        "byte_ledger": ledger,
        "byte_ledger_measured": measured,
        "byte_ledger_matches_prereg": ledger_matches,
        "byte_ledger_prereg": exp,
        "byte_ledger_got": got,
        "declared_channel": decode_width(store, str(c.read_mode)),
        "n_coupling_fit_rows": int(nfit),
        "build_wall_clock_s": float(time.time() - t0),
    }


def _arm_landscape(land: AtomLandscape, arm: str, cfg: SettleCfg, launch_mode: str):
    """⭐ Each arm differs from ``full`` in exactly ONE declared quantity."""
    if arm == "full":
        return land, cfg, launch_mode
    if arm == "request_off":
        return land.with_strengths(request_strength=0.0), cfg, launch_mode
    if arm == "request_off_launch":
        return land.with_strengths(request_strength=0.0), cfg, "prior_atom"
    if arm == "coupling_off":
        return land.with_strengths(coupling_strength=0.0), cfg, launch_mode
    if arm == "zero_step":
        z = SettleCfg(**{**cfg.as_dict(), "steps": 0})
        return land, z, launch_mode
    raise ValueError(f"unknown arm {arm!r}")


def run_arm(config: CHLUConfig, built: dict, arm: str, requests: np.ndarray,
            key, log: Callable = print) -> dict:
    """One arm, chunked over requests.  ⛔ Arm keys come from ``fold_in``.

    ⚠ **JAX LANDMINE, measured on this box (JAX 0.9.0):** with
    ``jax_threefry_partitionable=True`` ``split(k, 2)[1]`` **IS** ``split(k, 3)[1]``,
    so lengthening a split to add an arm hands two arms the same key.
    :func:`chlu.core.gen_landscape.arm_key` folds the arm's name in instead.
    """
    c = _cfg(config)
    base = SettleCfg(steps=int(c.steps), step_size=float(c.step_size),
                     sigma_scale_start=float(c.sigma_scale_start),
                     sigma_scale_end=float(c.sigma_scale_end),
                     temp_start=float(c.temp_start), temp_end=float(c.temp_end),
                     metropolis=bool(c.metropolis))
    land, cfg, launch_mode = _arm_landscape(built["land"], arm, base, str(c.launch_mode))
    k_arm = arm_key(key, arm)
    chunk = max(int(c.request_chunk), 1)
    parts, traces, launches = [], [], []
    t0 = time.time()
    for i in range(0, int(requests.shape[0]), chunk):
        R = requests[i : i + chunk]
        out = dynamical_read(land, built["book"], R, cfg, jax.random.fold_in(k_arm, i),
                             launch_mode=launch_mode,
                             clamp_threshold=float(c.clamp_threshold),
                             read_mode=str(c.read_mode))
        parts.append(out)
        traces.append(out["trace"])
        launches.append(out["launch"])
    merge = lambda kk: np.concatenate([np.asarray(p[kk]) for p in parts], axis=0)  # noqa: E731
    trace = {
        kk: (np.concatenate([np.asarray(t[kk]) for t in traces], axis=1)
             if np.asarray(traces[0][kk]).ndim == 2 else np.asarray(traces[0][kk]))
        for kk in ("energy", "energy_ref", "accept_prob", "accepted", "displacement")
    }
    trace["sigma_scale"] = np.asarray(traces[0]["sigma_scale"])
    trace["temperature"] = np.asarray(traces[0]["temperature"])
    conv = convergence_report(trace, int(c.descent_window), float(c.descent_tol))
    lau = {
        "launch_mode": launch_mode,
        "clamp_threshold": float(c.clamp_threshold),
        "mean_n_clamped": float(np.mean([lo["mean_n_clamped"] for lo in launches])),
        "mean_n_relaxed": float(np.mean([lo["mean_n_relaxed"] for lo in launches])),
        "mean_confidence": float(np.mean([lo["mean_confidence"] for lo in launches])),
        "n_wells": int(launches[0]["n_wells"]),
    }
    x0, xs, rd = merge("x0"), merge("x_settled"), merge("read")
    # ⛔ A diverged chain's energy trace measures nothing — say so explicitly.
    radii = np.asarray([np.asarray(a).std() for a in built["store"].atoms], np.float64)
    ref = float(np.sqrt(np.mean(radii**2))) or 1.0
    rms_state = float(np.sqrt(np.mean(xs**2)))
    diverged = bool(
        (not np.isfinite(rms_state))
        or rms_state > float(c.divergence_radius_mult) * ref)
    rec = {
        "arm": arm,
        "request_strength": float(land.request_strength),
        "coupling_strength": float(land.coupling_strength),
        "settle_cfg": cfg.as_dict(),
        "launch": lau,
        "dynamics": conv,
        "stability": {
            "settled_rms": rms_state,
            "atom_rms_radius": ref,
            "radius_ratio": rms_state / ref if ref else float("nan"),
            "divergence_radius_mult": float(c.divergence_radius_mult),
            "diverged": diverged,
            "verdict": ("⛔ DIVERGED — the landscape is not bounded below at this "
                        "coupling strength and NO number from this arm is quotable"
                        if diverged else "bounded"),
        },
        "displacement_from_launch": {
            "rms": float(np.sqrt(np.mean((xs - x0) ** 2))),
            "max": float(np.max(np.abs(xs - x0))) if xs.size else float("nan"),
        },
        "settled_spread": settled_state_spread(xs, built["store"]),
        # ⭐ launch vs settled: does the chain SEPARATE requests the discrete clamp
        # collided, or collapse them?  ⛔ The one the scored run's falsifier 2 rests
        # on, and the one PR alone got wrong.
        "separation": {"launch": pairwise_separation(x0),
                       "settled": pairwise_separation(xs)},
        "wall_clock_s": float(time.time() - t0),
    }
    log(f"      [{arm}] beta={land.request_strength:g} lambda={land.coupling_strength:g} "
        f"clamped={lau['mean_n_clamped']:.2f}/{lau['n_wells']} "
        f"E {conv['energy_first']:.3f}->{conv['energy_final']:.3f} "
        f"acc={conv['accepted_fraction']} "
        f"disp[1]={conv['displacement_step_1']} disp[T]={conv['displacement_final_step']} "
        f"still_descending={conv['frac_still_descending']} · {conv['verdict']}"
        + ("  ⛔ DIVERGED" if diverged else ""))
    log(f"        separation launch -> settled: "
        f"{rec['separation']['launch']['mean_pairwise_rms_distance']:.4f} -> "
        f"{rec['separation']['settled']['mean_pairwise_rms_distance']:.4f} "
        f"(min {rec['separation']['launch']['min_pairwise_rms_distance']:.4f} -> "
        f"{rec['separation']['settled']['min_pairwise_rms_distance']:.4f}, "
        f"collapsed pairs {rec['separation']['launch']['frac_pairs_collapsed']:.3f} -> "
        f"{rec['separation']['settled']['frac_pairs_collapsed']:.3f})")
    return rec, {"x0": x0, "x_settled": xs, "read": rd, "trace": trace}


def steering_table(arms: dict, states: dict) -> dict:
    """⭐ **The laundering control, made arithmetic.**

    ⛔ Dreaming's clamp produced a gain the coupling-off control took *entirely*
    (steering-attributable ``+0.0015 ± 0.0204`` = ZERO).  This reports, at the level
    of the settled STATE (⛔ not of any score — the scored run owns scores), how far
    each control's trajectory sits from the full arm's.  **If request-off already
    gives the same trajectory, that is dreaming again and it is said here.**
    """
    if "full" not in states:
        return {"skipped": "no full arm"}
    F = states["full"]["x_settled"]
    scale = float(np.sqrt(np.mean((F - F.mean(axis=0, keepdims=True)) ** 2)))
    out = {"reference": "full", "settled_state_rms_spread": scale, "per_arm": {}}
    for a, s in states.items():
        if a == "full":
            continue
        d = s["x_settled"] - F
        rms = float(np.sqrt(np.mean(d**2)))
        out["per_arm"][a] = {
            "rms_delta_vs_full": rms,
            "max_abs_delta_vs_full": float(np.max(np.abs(d))) if d.size else float("nan"),
            "rms_delta_relative_to_spread": rms / scale if scale > 0 else float("nan"),
            "identical_to_full": bool(rms == 0.0),
        }
    ro = out["per_arm"].get("request_off", {})
    out["reading"] = (
        "⛔ REQUEST-OFF GIVES THE SAME TRAJECTORY — the chain is settling onto a "
        "mode, not being steered. THIS IS DREAMING AGAIN."
        if ro.get("identical_to_full") else
        "request-off moves the trajectory; the request energy is doing work. ⛔ Whether "
        "that work is a GAIN is a scored-run question and is not answered here."
    )
    return out


# ---------------------------------------------------------------------------
# 3. ⛔⭐ The bypass guarantee, over launch + settle + read as ONE path
# ---------------------------------------------------------------------------


def bypass_report(config: CHLUConfig, built: dict, requests: np.ndarray, key,
                  X_feat: np.ndarray, log: Callable = print) -> dict:
    """⭐ Assert on the real path; ⛔ then **demonstrate the assertion FAILING**.

    A guard that has never been seen to fail is decoration — so the report carries
    one line per bypassed stage, each of which had to raise.
    """
    c = _cfg(config)
    n = int(min(requests.shape[0], 32))
    R = requests[:n]
    cfg = SettleCfg(steps=int(min(int(c.steps), 25)), step_size=float(c.step_size),
                    sigma_scale_start=float(c.sigma_scale_start),
                    sigma_scale_end=float(c.sigma_scale_end),
                    temp_start=float(c.temp_start), temp_end=float(c.temp_end),
                    metropolis=bool(c.metropolis))
    D = int(built["store"].dim)
    rng = np.random.default_rng(20_260_914)
    A = np.asarray(X_feat[:n], np.float32).reshape((n, -1))[:, :D]
    if A.shape[0] < n:
        A = np.repeat(A, int(np.ceil(n / max(A.shape[0], 1))), axis=0)[:n]
    B = rng.standard_normal((n, D)).astype(np.float32) * float(np.abs(A).max() + 1.0)
    out = {"real_path": assert_request_only(
        built["land"], built["book"], R, cfg, key, A, B,
        launch_mode=str(c.launch_mode), clamp_threshold=float(c.clamp_threshold),
        read_mode=str(c.read_mode))}
    log(f"      ⭐ request-only guarantee HOLDS over launch+settle+read: "
        f"{out['real_path']['max_abs_delta_per_stage']} "
        f"(feature tensors differed by {out['real_path']['feature_tensor_max_abs_difference']:.3f})")
    if bool(c.demonstrate_bypass_failure):
        demos = {}
        for stage in ("launch", "settle", "read"):
            try:
                assert_request_only(
                    built["land"], built["book"], R, cfg, key, A, B,
                    launch_mode=str(c.launch_mode),
                    clamp_threshold=float(c.clamp_threshold),
                    read_mode=str(c.read_mode), bypass_stage=stage,
                    bypass_alpha=float(c.bypass_alpha))
            except AssertionError as e:
                demos[stage] = {"raised": True, "message": str(e)[:400]}
                log(f"      ✓ deliberately bypassed at {stage}: assertion FIRED")
                continue
            demos[stage] = {"raised": False}
            raise AssertionError(
                f"⛔⛔ THE BYPASS ASSERTION DID NOT FIRE on a variant deliberately "
                f"bypassed at the {stage} stage (alpha={c.bypass_alpha}). The guard "
                "is DECORATION and the structural claim is unsupported.")
        out["deliberately_bypassed_variants"] = demos
        out["all_stages_caught"] = bool(all(v["raised"] for v in demos.values()))
    return out


# ---------------------------------------------------------------------------
# 4. One seed
# ---------------------------------------------------------------------------


def run_seed(config: CHLUConfig, dataset: str, pretext: str, seed: int,
             out_dir: Path, log: Callable = print) -> dict:
    c = _cfg(config)
    t0 = time.time()
    bdir = bridge_dir(config, dataset, pretext, int(seed))
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    crows, feats, cman = load_feature_cache(bdir, verify=bool(c.verify_cache_checksum))
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    log(f"  bridge {bdir} · cache scope={cman.get('scope')!r} rows={crows.size}")

    from ..core.gen_bridge import dataset_module

    mod = dataset_module(dataset)
    split, _values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    rows = select_rows(config, dataset, split, int(seed))
    config.gen_store.encode_chunk = int(c.encode_chunk)
    reencode = _assert_cache_reencodes(dataset, config, encoder, crows, feats, cman, log=log)

    X, Y, NVR = {}, {}, {}
    for name in ("store_train",) + REQUEST_SETS:
        r = rows[name]
        if r.size == 0:
            continue
        log(f"    gathering {name}: {r.size} rows")
        X[name] = gather_features(dataset, config, encoder, crows, feats, cman, r, log=log)
        Y[name] = factor_labels(grid, r)
        NVR[name] = n_values_realised(Y[name], names, cards)

    # ⛔ The partition: loaded from gen-commitment-collapse (fitted on UNWHITENED
    # features, and EXONERATED by gen-loss-attribution) and asserted against the
    # shipped codebook store, whose feat_shape the decoder expects.
    labels_vec = load_grouping(config, dataset, pretext, int(seed))
    shipped, sinfo = load_attrib_store(config, dataset, pretext, int(seed), labels_vec, log)
    wc = well_channels_from_labels(labels_vec, int(shipped.n_wells))
    feat_shape = tuple(shipped.feat_shape)

    rec = {
        "schema": "chlu-gen-dynamical-read/1",
        "seed": int(seed), "dataset": dataset, "pretext": pretext,
        "dial": "the read — continuous settling over stored atoms, ⛔ not codebook copy-back",
        "build_only": ("⛔ BUILD ONLY. The scored run is gated: no falsifier verdict, "
                       "no 3-seed cell and no external-classifier number is emitted here."),
        "factor_names": names,
        "cardinalities": [int(x) for x in cards],
        "bridge_dir": str(bdir),
        "encoder_param_hash": enc_hash,
        "cache": {"scope": cman.get("scope"), "n_rows": int(crows.size),
                  "D": int(cman["D"]), "reencode_check": reencode},
        "shipped_codebook_store": {**sinfo, "store_bytes": store_bytes(shipped)},
        "rows": {k: int(np.asarray(v).size) for k, v in rows.items()},
        "n_values_realised": NVR,
        "well_sizes": [len(w) for w in wc],
        "budgets": {},
    }

    # ⭐ the reference the arm is measured against later: the codebook's own read
    # error on the request rows.  ⛔ Reported as context; the scored run owns the
    # comparison.
    for rs in REQUEST_SETS:
        if rs not in X:
            continue
        codes = address_codes(shipped, np.asarray(X[rs], np.float32))
        r = np.asarray(read_from_codes(shipped, codes), np.float32).reshape((codes.shape[0], -1))
        Xf = np.asarray(X[rs], np.float32).reshape((r.shape[0], -1))
        rec.setdefault("codebook_reference", {})[rs] = {
            "relative_read_error": float(np.mean((r - Xf) ** 2)
                                         / max(float(np.mean((Xf - Xf.mean(0)) ** 2)), 1e-12)),
            "n_rows": int(r.shape[0]),
            "n_distinct_codes": int(np.unique(codes, axis=0).shape[0]),
        }

    decoder, dguard = _load_decoder(config, int(seed), log=log)
    key0 = jax.random.PRNGKey(int(c.row_seed_offset) + int(seed))

    for budget in list(c.budgets):
        log(f"  budget {budget}")
        built = build_landscape(config, budget, X["store_train"], Y["store_train"],
                                wc, feat_shape, cards, int(seed), log=log)
        brec = {k: v for k, v in built.items()
                if k not in ("store", "book", "coupling", "land")}
        bkey = arm_key(key0, f"budget_{budget}")
        brec["request_sets"] = {}
        for rs in REQUEST_SETS:
            if rs not in X or X[rs].shape[0] == 0:
                continue
            requests = requests_from_labels(Y[rs], "full", 0, int(seed))
            log(f"    request set {rs}: {requests.shape[0]} FULL requests")
            arms, states = {}, {}
            for arm in list(c.arms):
                arms[arm], states[arm] = run_arm(
                    config, built, arm, requests, arm_key(bkey, rs), log=log)
                sp = out_dir / f"energy_traces_s{int(seed)}_{budget}_{rs}_{arm}.npz"
                save_trace(states[arm]["trace"], sp)
            np.savez_compressed(
                out_dir / f"settled_states_s{int(seed)}_{budget}_{rs}.npz",
                **{f"{a}_x_settled": np.asarray(s["x_settled"], np.float32)
                   for a, s in states.items()},
                **{f"{a}_x0": np.asarray(s["x0"], np.float32) for a, s in states.items()},
                requests=np.asarray(requests, np.int16))
            entry = {
                "n_requests": int(requests.shape[0]),
                "protocol": "full (⭐ HEADLINE: the arm's job is RESOLUTION INSIDE the "
                            "addressed wells, D1)",
                "n_values_realised": NVR[rs],
                "arms": arms,
                "steering": steering_table(arms, states),
            }
            log("      ⭐ " + entry["steering"]["reading"])
            if bool(c.decode_reads) and decoder is not None:
                entry["decoded"] = _decode(config, decoder, states, feat_shape,
                                           out_dir, seed, budget, rs, log=log)
            entry["bypass"] = bypass_report(config, built, requests, bkey,
                                            X[rs], log=log)
            brec["request_sets"][rs] = entry
        brec["decoder_guard"] = dguard
        rec["budgets"][budget] = brec

    rec["wall_clock_s"] = float(time.time() - t0)
    rec["complete"] = True
    _write_json(rec, Path(out_dir) / f"dynread_s{int(seed)}.json")
    return rec


def _load_decoder(config: CHLUConfig, seed: int, log: Callable = print):
    """⭐ ``gen-decoder-lift``'s WINNER, hash-asserted.  ⛔ The standing decoder trap:
    ``GenBridgeConfig.dec_recipe`` still defaults to ``"baseline"`` — the recipe that
    FAILED the gate (0.7191, 0/3 seeds).  A run through it looks entirely normal in
    the logs and is **void**."""
    c = _cfg(config)
    dpath = _abs(config, c.winner_decoder_pattern.format(seed=int(seed)))
    hpath = _abs(config, c.winner_hashes_path)
    if not dpath.exists() or not hpath.exists():
        return None, {"skipped": f"⛔ no winner decoder at {dpath} — the decoded column "
                                 "is DECLARED ABSENT, never silently replaced by the "
                                 "baseline recipe"}
    with open(hpath) as fh:
        hashes = json.load(fh)
    decoder = load_model(dpath)
    guard = assert_dream_decoder(decoder, hashes[f"seed_{int(seed)}"]["param_hash"],
                                 str(c.dec_recipe), WINNING_DECODER_RECIPE)
    log(f"  ⭐ decoder guard: recipe={guard['decoder_recipe']} "
        f"hash={guard['decoder_param_hash'][:16]}…")
    return decoder, guard


def _decode(config: CHLUConfig, decoder, states: dict, feat_shape, out_dir: Path,
            seed: int, budget: str, rs: str, log: Callable = print) -> dict:
    """Render every arm's settled state through the frozen winner decoder.

    ⛔ **No probe, no classifier, no accuracy.**  This proves the path reaches an
    image; the numbers on those images belong to the scored run.
    """
    c = _cfg(config)
    n = int(c.decode_rows) or None
    out = {"n_rows": 0, "arms": {}}
    grids = {}
    for arm, s in states.items():
        r = np.asarray(s["read"], np.float32)
        r = r[:n] if n else r
        r = r.reshape((r.shape[0],) + tuple(feat_shape))
        imgs = decode_features(decoder, r, int(c.decode_batch))
        grids[arm] = imgs
        out["n_rows"] = int(imgs.shape[0])
        out["arms"][arm] = {
            "image_shape": [int(x) for x in imgs.shape[1:]],
            "mean": float(imgs.mean()), "std": float(imgs.std()),
            "min": float(imgs.min()), "max": float(imgs.max()),
        }
    if grids and bool(c.make_plots):
        out["grid_png"] = str(_grid_png(
            grids, out_dir / f"decoded_s{int(seed)}_{budget}_{rs}.png"))
    log(f"      decoded {out['n_rows']} rows/arm through the frozen winner decoder")
    return out


def _grid_png(grids: dict, path: Path, n_show: int = 6) -> Path:
    arms = list(grids)
    n = int(min(n_show, min(g.shape[0] for g in grids.values())))
    fig, axes = plt.subplots(len(arms), n, figsize=(1.4 * n, 1.5 * len(arms)),
                             squeeze=False)
    for i, a in enumerate(arms):
        for j in range(n):
            im = np.asarray(grids[a][j])
            im = np.transpose(im, (1, 2, 0)) if im.ndim == 3 and im.shape[0] <= 4 else im
            axes[i][j].imshow(np.clip(im, 0, 1))
            axes[i][j].axis("off")
        axes[i][0].set_ylabel(a, fontsize=6)
        axes[i][0].axis("on")
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])
    fig.suptitle("settled state -> frozen winner decoder (⛔ build smoke, not scored)",
                 fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. Across seeds — ⛔ a build roll-up, NOT a scored cell
# ---------------------------------------------------------------------------


def summarize(records: list, config: CHLUConfig) -> dict:
    """⛔ **No verdict.**  The dynamics, the ledger and the guarantee, per budget."""
    out = {
        "schema": "chlu-gen-dynamical-read-summary/1",
        "n_seeds": len(records), "seeds": [r["seed"] for r in records],
        "build_only": records[0]["build_only"] if records else "",
        "budgets": {},
    }
    for b in records[0].get("budgets", {}):
        rows = [r["budgets"][b] for r in records if b in r.get("budgets", {})]
        if not rows:
            continue
        entry = {
            "byte_ledger": rows[0]["byte_ledger"],
            "byte_ledger_matches_prereg": rows[0]["byte_ledger_matches_prereg"],
            "byte_ledger_prereg": rows[0]["byte_ledger_prereg"],
            "declared_channel": rows[0]["declared_channel"],
            "atoms_are_train_rows": rows[0]["atoms_are_train_rows"],
            "request_sets": {},
        }
        for rs in rows[0].get("request_sets", {}):
            rr = [x["request_sets"][rs] for x in rows if rs in x.get("request_sets", {})]
            entry["request_sets"][rs] = {
                "n_requests": rr[0]["n_requests"],
                "n_values_realised": rr[0]["n_values_realised"],
                "dynamics": {
                    a: {
                        "frac_still_descending": [
                            x["arms"][a]["dynamics"]["frac_still_descending"] for x in rr],
                        "accepted_fraction": [
                            x["arms"][a]["dynamics"]["accepted_fraction"] for x in rr],
                        "displacement_step_1": [
                            x["arms"][a]["dynamics"]["displacement_step_1"] for x in rr],
                        "displacement_final_step": [
                            x["arms"][a]["dynamics"]["displacement_final_step"] for x in rr],
                        "verdict": [x["arms"][a]["dynamics"]["verdict"] for x in rr],
                        "mean_pairwise_rms_launch": [
                            x["arms"][a]["separation"]["launch"][
                                "mean_pairwise_rms_distance"] for x in rr],
                        "mean_pairwise_rms_settled": [
                            x["arms"][a]["separation"]["settled"][
                                "mean_pairwise_rms_distance"] for x in rr],
                        "frac_pairs_collapsed_settled": [
                            x["arms"][a]["separation"]["settled"][
                                "frac_pairs_collapsed"] for x in rr],
                    } for a in rr[0]["arms"]
                },
                "steering": [x["steering"] for x in rr],
                "bypass_all_stages_caught": [
                    x["bypass"].get("all_stages_caught") for x in rr],
            }
        out["budgets"][b] = entry
    return out


def run_experiment_gen_dynamical_read(config: Optional[CHLUConfig] = None,
                                      out_dir: Optional[str] = None,
                                      log: Callable = print) -> dict:
    config = config or get_default_config()
    c = _cfg(config)
    root = Path(out_dir) if out_dir else dynread_out_root(config)
    runs = root / "runs" / f"{c.dataset}_{c.pretext}"
    runs.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in list(c.seeds):
        sdir = runs / f"seed_{int(seed)}"
        sdir.mkdir(parents=True, exist_ok=True)
        path = sdir / f"dynread_s{int(seed)}.json"
        if path.exists() and bool(c.skip_if_exists):
            with open(path) as fh:
                rec = json.load(fh)
            if rec.get("complete"):
                log(f"  seed {seed}: artefact HIT {path.name} — skipped")
                records.append(rec)
                continue
        log(f"  seed {seed}: running")
        records.append(run_seed(config, c.dataset, c.pretext, int(seed), sdir, log=log))
    summary = summarize(records, config)
    _write_json(summary, root / "summary.json")
    log("⛔ BUILD ONLY — no falsifier verdict is rendered here; the scored run is gated.")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=None)
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS) + [None])
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--budgets", nargs="+", default=None, choices=["A", "B", None])
    p.add_argument("--arms", nargs="+", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--no-decode", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    config = get_default_config()
    c = _cfg(config)
    # ⛔ ORDER MATTERS AND IT WAS A MEASURED BUG: ``apply_quick`` REPLACES the seed,
    # budget, arm and step lists, so an explicit flag applied *before* it is silently
    # discarded.  A `--quick --budget A B` run reported budget A only and looked
    # entirely normal in the logs.  ⭐ Quick first, explicit flags last, always.
    if args.quick:
        apply_quick(config)
    if args.dataset:
        c.dataset = args.dataset
    if args.pretext:
        c.pretext = args.pretext
    if args.seeds:
        c.seeds = list(args.seeds)
    if args.budgets:
        c.budgets = list(args.budgets)
    if args.arms:
        c.arms = list(args.arms)
    if args.steps:
        c.steps = int(args.steps)
    if args.no_decode:
        c.decode_reads = False
    run_experiment_gen_dynamical_read(config, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
