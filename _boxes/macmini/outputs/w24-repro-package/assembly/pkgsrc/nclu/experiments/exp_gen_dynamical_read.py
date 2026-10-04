"""exp gen dynamical read."""

from __future__ import annotations
import time
from typing import Callable
import jax
import matplotlib

matplotlib.use("Agg")
import numpy as np
from nclu.config import NCLUConfig, gen_dynamical_read_config
from nclu.core.gen_landscape import (
    AtomLandscape,
    SettleCfg,
    arm_key,
    assert_atoms_are_train_rows,
    atom_bandwidths,
    build_atom_store,
    convergence_report,
    decode_width,
    dynamical_read,
    fit_atom_address_book,
    fit_low_rank_coupling,
    landscape_byte_footprint,
    pairwise_separation,
    safe_coupling_strength,
    settled_state_spread,
)


def _cfg(config: NCLUConfig):
    return gen_dynamical_read_config(config)


def build_landscape(
    config: NCLUConfig,
    budget: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    well_channels,
    feat_shape,
    cards,
    seed: int,
    log: Callable = print,
) -> dict:
    c = _cfg(config)
    spec = c.budget_spec(budget)
    t0 = time.time()
    store = build_atom_store(
        well_channels, feat_shape, X_train, int(spec["n_atoms"]), str(c.atom_selection), int(seed)
    )
    atoms_ok = assert_atoms_are_train_rows(store, X_train)
    sigma = atom_bandwidths(store, float(c.bandwidth_frac), str(c.bandwidth_rule))
    book = fit_atom_address_book(store, X_train, Y_train, cards, sigma, float(c.address_smoothing))
    nfit = min(int(c.n_coupling_fit), X_train.shape[0])
    sel = np.sort(
        np.random.default_rng(
            int(seed) * int(c.row_seed_mult) + int(c.row_seed_offset) + 313
        ).permutation(X_train.shape[0])[:nfit]
    )
    coupling = fit_low_rank_coupling(
        store, X_train[sel], int(spec["rank"]), bool(c.coupling_normalise)
    )
    land = AtomLandscape(
        store, sigma, coupling, float(c.request_strength), float(c.coupling_strength)
    )
    cert = safe_coupling_strength(land, float(c.coupling_strength), float(c.coupling_safety_frac))
    if bool(c.enforce_coupling_certificate):
        land = land.with_strengths(coupling_strength=float(cert["coupling_strength_used"]))
    if cert["certificate_bound_the_strength"]:
        log(
            f"    [{spec['label']}] ⚠ coupling strength CAPPED BY THE CERTIFICATE: declared {cert['declared_coupling_strength']:g} -> used {cert['coupling_strength_used']:.4g} (lambda* = {cert['max_safe_coupling_strength']:.4g})"
        )
    else:
        log(
            f"    [{spec['label']}] coupling certificate: lambda* = {cert['max_safe_coupling_strength']:.4g}, using {cert['coupling_strength_used']:g} (bounded below={cert['certificate']['bounded_below']})"
        )
    footprint = landscape_byte_footprint(
        int(spec["n_atoms"]),
        store.well_sizes,
        int(spec["rank"]),
        int(spec["budget_bytes"]),
        label=f"budget {spec['label']} ({spec['role']})",
    )
    measured = {"store_bytes": int(store.n_bytes()), "coupling_bytes": int(coupling.n_bytes())}
    if measured["store_bytes"] != footprint["store"]["store_bytes"]:
        raise AssertionError(
            f"⛔ the store's measured bytes {measured['store_bytes']} do not match the declared footprint {footprint['store']['store_bytes']}"
        )
    if measured["coupling_bytes"] != footprint["coupling"]["coupling_bytes"]:
        raise AssertionError(
            f"⛔ the coupling's measured bytes {measured['coupling_bytes']} do not match the declared footprint {footprint['coupling']['coupling_bytes']}"
        )
    exp = c.byte_footprint_expectations(budget)
    got = {
        "store_bytes": int(footprint["store"]["store_bytes"]),
        "coupling_bytes": int(footprint["coupling"]["coupling_bytes"]),
        "total_bytes": int(footprint["total_bytes"]),
    }
    footprint_matches = {k: bool(got[k] == int(exp[k])) for k in got}
    footprint_matches["multiple_of_budget"] = bool(
        abs(float(footprint["multiple_of_budget"]) - float(exp["multiple_of_budget"])) < 1e-09
    )
    if bool(c.assert_byte_footprint) and (not all(footprint_matches.values())):
        raise AssertionError(
            f"⛔ THE BYTE FOOTPRINT DOES NOT MATCH PROTOCOL §3 — declared before the run and therefore binding. expected={exp} got store={footprint['store']['store_bytes']} coupling={footprint['coupling']['coupling_bytes']} total={footprint['total_bytes']} ({footprint['multiple_of_budget']:.4f}x budget)"
        )
    log(
        f"    [{spec['label']}] N={spec['n_atoms']} r={spec['rank']} store={footprint['store']['store_bytes']:,} B coupling={footprint['coupling']['coupling_bytes']:,} B total={footprint['total_bytes']:,} B = {footprint['multiple_of_budget']:.2f}x budget (PROTOCOL §3 match: {all(footprint_matches.values())})"
    )
    log(f"    [{spec['label']}] sigma per well = " + " ".join((f"{float(s):.4f}" for s in sigma)))
    return {
        "spec": spec,
        "store": store,
        "book": book,
        "coupling": coupling,
        "land": land,
        "sigma": [float(s) for s in sigma],
        "atoms_are_train_rows": atoms_ok,
        "coupling_certificate": cert,
        "byte_footprint": footprint,
        "byte_footprint_measured": measured,
        "byte_footprint_matches_declared": footprint_matches,
        "byte_footprint_protocol": exp,
        "byte_footprint_got": got,
        "declared_channel": decode_width(store, str(c.read_mode)),
        "n_coupling_fit_rows": int(nfit),
        "build_wall_clock_s": float(time.time() - t0),
    }


def _arm_landscape(land: AtomLandscape, arm: str, cfg: SettleCfg, launch_mode: str):
    if arm == "full":
        return (land, cfg, launch_mode)
    if arm == "request_off":
        return (land.with_strengths(request_strength=0.0), cfg, launch_mode)
    if arm == "request_off_launch":
        return (land.with_strengths(request_strength=0.0), cfg, "prior_atom")
    if arm == "coupling_off":
        return (land.with_strengths(coupling_strength=0.0), cfg, launch_mode)
    if arm == "zero_step":
        z = SettleCfg(**{**cfg.as_dict(), "steps": 0})
        return (land, z, launch_mode)
    raise ValueError(f"unknown arm {arm!r}")


def run_arm(
    config: NCLUConfig, built: dict, arm: str, requests: np.ndarray, key, log: Callable = print
) -> dict:
    c = _cfg(config)
    base = SettleCfg(
        steps=int(c.steps),
        step_size=float(c.step_size),
        sigma_scale_start=float(c.sigma_scale_start),
        sigma_scale_end=float(c.sigma_scale_end),
        temp_start=float(c.temp_start),
        temp_end=float(c.temp_end),
        metropolis=bool(c.metropolis),
    )
    land, cfg, launch_mode = _arm_landscape(built["land"], arm, base, str(c.launch_mode))
    k_arm = arm_key(key, arm)
    chunk = max(int(c.request_chunk), 1)
    parts, traces, launches = ([], [], [])
    t0 = time.time()
    for i in range(0, int(requests.shape[0]), chunk):
        R = requests[i : i + chunk]
        out = dynamical_read(
            land,
            built["book"],
            R,
            cfg,
            jax.random.fold_in(k_arm, i),
            launch_mode=launch_mode,
            clamp_threshold=float(c.clamp_threshold),
            read_mode=str(c.read_mode),
        )
        parts.append(out)
        traces.append(out["trace"])
        launches.append(out["launch"])
    merge = lambda kk: np.concatenate([np.asarray(p[kk]) for p in parts], axis=0)
    trace = {
        kk: np.concatenate([np.asarray(t[kk]) for t in traces], axis=1)
        if np.asarray(traces[0][kk]).ndim == 2
        else np.asarray(traces[0][kk])
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
    x0, xs, rd = (merge("x0"), merge("x_settled"), merge("read"))
    radii = np.asarray([np.asarray(a).std() for a in built["store"].atoms], np.float64)
    ref = float(np.sqrt(np.mean(radii**2))) or 1.0
    rms_state = float(np.sqrt(np.mean(xs**2)))
    diverged = bool(not np.isfinite(rms_state) or rms_state > float(c.divergence_radius_mult) * ref)
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
            "verdict": "⛔ DIVERGED — the landscape is not bounded below at this coupling strength and NO number from this arm is quotable"
            if diverged
            else "bounded",
        },
        "displacement_from_launch": {
            "rms": float(np.sqrt(np.mean((xs - x0) ** 2))),
            "max": float(np.max(np.abs(xs - x0))) if xs.size else float("nan"),
        },
        "settled_spread": settled_state_spread(xs, built["store"]),
        "separation": {"launch": pairwise_separation(x0), "settled": pairwise_separation(xs)},
        "wall_clock_s": float(time.time() - t0),
    }
    log(
        f"      [{arm}] beta={land.request_strength:g} lambda={land.coupling_strength:g} clamped={lau['mean_n_clamped']:.2f}/{lau['n_wells']} E {conv['energy_first']:.3f}->{conv['energy_final']:.3f} acc={conv['accepted_fraction']} disp[1]={conv['displacement_step_1']} disp[T]={conv['displacement_final_step']} still_descending={conv['frac_still_descending']} · {conv['verdict']}"
        + ("  ⛔ DIVERGED" if diverged else "")
    )
    log(
        f"        separation launch -> settled: {rec['separation']['launch']['mean_pairwise_rms_distance']:.4f} -> {rec['separation']['settled']['mean_pairwise_rms_distance']:.4f} (min {rec['separation']['launch']['min_pairwise_rms_distance']:.4f} -> {rec['separation']['settled']['min_pairwise_rms_distance']:.4f}, collapsed pairs {rec['separation']['launch']['frac_pairs_collapsed']:.3f} -> {rec['separation']['settled']['frac_pairs_collapsed']:.3f})"
    )
    return (rec, {"x0": x0, "x_settled": xs, "read": rd, "trace": trace})
