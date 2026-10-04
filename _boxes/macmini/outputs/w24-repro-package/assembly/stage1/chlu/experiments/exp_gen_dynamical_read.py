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
import time
from typing import Callable
import jax
import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
from ..config import CHLUConfig, gen_dynamical_read_config
from ..core.gen_landscape import AtomLandscape, SettleCfg, arm_key, assert_atoms_are_train_rows, atom_bandwidths, build_atom_store, convergence_report, decode_width, dynamical_read, fit_atom_address_book, fit_low_rank_coupling, landscape_byte_ledger, pairwise_separation, safe_coupling_strength, settled_state_spread
def _cfg(config: CHLUConfig):
    return gen_dynamical_read_config(config)


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


