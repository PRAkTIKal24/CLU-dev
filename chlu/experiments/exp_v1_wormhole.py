"""Experiment V1-Wormhole: energy-gated sparse non-local routing (V1 pillar 3).

The third V1 inference-time mechanism (after the calibrated gate and the parked
squeeze retries): **wormhole routing**. A chain lattice of N CHLU associative
memories; a query arrives at the query unit (unit 0) and its answer is either
*local* (bound in unit 0) or *distant* (bound only in the archive unit, N-1).
An energy-gated sparse edge opens a non-local path when the query relaxes badly
in its local unit — attention-like long-range access priced in energy.

Mechanism (F5 §7.4 smooth gate; Def-7 escalation beyond a single shell):
  1. Each unit i is written (generative PCD, no MSE) as an EBM over its own
     disjoint set of [key||value] patterns.
  2. Phase 1 — clamp the query cue into the KEY half of unit 0 and relax under
     the energy governor; decode unit 0's value half (the local answer). The
     routing signal is the LOCAL residual R0 = H_0(settled) - floor_0: low when
     unit 0 holds the answer (local hit), high when it does not (distant).
  3. Phase 2 (routing arms) — the wormhole is a GATED KEY-channel spring between
     unit 0 and the archive that TRANSPORTS the clamped query key to the
     archive's free key half. When open, the archive relaxes to the stored
     pattern whose key matches the query and its value half fills in; the answer
     is then read at the *terminal* (archive) unit. (A weak position coupling
     cannot drag a unit's value out of its own attractor basin, so the honest
     read is the unit that actually retrieves — the wormhole's value is being a
     DIRECT 1-hop edge, not force-writing a foreign value into unit 0.)
  4. The gate weight g is a smooth, label-free z-normalized sigmoid of R0, held
     fixed per query during the routed rollout, so H stays C^1 and conformally
     symplectic (F5 §7.4 no-energy-ledger regime). Route (and read the archive)
     iff g > threshold; else keep the local answer.

Opening the gate on a *local* query mis-routes it (the archive's key is pulled
to a cue it does not store => garbage) — gate selectivity therefore has real
teeth, and the always-open "dense" arm pays for it in local accuracy as well as
compute.

Five arms:
  (a) local-only     : phase-1 only, read unit 0 (ceiling local, floor distant)
  (b) gated wormhole : smooth energy gate on R0 -> route the direct edge (mech.)
  (c) dense open     : always-open edge (g = 1), same connectivity, no gate
  (d) chain multi-hop: no wormhole; the query key must diffuse hop-by-hop along
                       the chain to reach the archive (cost scales with N)
  (e) calibrated tau : learned per-model head (R0[,margin]) -> route iff p > tau

Squeeze retries stay PARKED (Head 2026-07-07); no top-k selection (smooth gates
only => everything conformally symplectic, no energy ledger); Lambda shell-lift
stays open — this is routing *within* one lattice.
"""

import json
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.chlu_unit import CHLU
from chlu.core.integrators import velocity_verlet_step
from chlu.core.lattice import channel_spring_coupling
from chlu.data.mqar import make_token_embeddings
from chlu.experiments.exp_v1_gate import _auroc, _decode_values, _settle_batch
from chlu.training.calibration import fit_calibration_head
from chlu.training.train_generative import train_generative
from chlu.utils.checkpoints import save_checkpoint
from chlu.utils.plotting import (
    plot_v1_wormhole_cost_accuracy,
    plot_v1_wormhole_selectivity,
)


# ---------------------------------------------------------------------------
# The smooth energy gate (residual-driven variant of the lattice's gate)
# ---------------------------------------------------------------------------


def smooth_gate(drive, threshold: float = 0.0, width: float = 1.0):
    """Smooth gate g = sigmoid((drive - threshold)/width) — opens (->1) as the
    driving energy `drive` increases (F5 §7.4 smooth gate). Monotone in drive;
    bounded in [0, 1]. NumPy-backed so it works on either array type post-hoc.
    """
    drive = np.asarray(drive, dtype=float)
    return 1.0 / (1.0 + np.exp(-(drive - threshold) / width))


def _zscore(x: np.ndarray) -> np.ndarray:
    """Label-free within-deployment normalization (median / IQR) — the
    v1-l0-gate finding: raw residual R is not cross-model comparable, so the
    gate consumes a per-deployment-standardized residual instead."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = (q75 - q25) + 1e-8
    return (x - med) / iqr


# ---------------------------------------------------------------------------
# Joint routed relaxation on the concatenated lattice state
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _joint_settle(
    units,
    couplings,
    edges,
    offsets,
    q0,
    p0,
    gate,
    steps,
    dt,
    floor,
    sensitivity,
    clamp_mask,
):
    """Governed joint Verlet relaxation of a concatenated lattice state.

    H_net(q, p) = sum_i [T_i(p_i) + V_i(q_i)] + sum_e gate_e * V_c^e(q_i, q_j).
    Positions-only couplings (F5 §7.2 cond 1); one global step (cond 2); scalar
    governor gamma = sensitivity * tanh(max(0, H - floor)). `gate` is per-query
    (B, E) so each query can scale its wormhole edge independently — a fixed
    scalar during the rollout, so H stays smooth and conformally symplectic
    (F5 §7.4 no-ledger regime). `clamp_mask` (D,) freezes clamped coordinates
    at their initial values with zero momentum (the query cue pinned into key
    halves). offsets/edges/steps are static (Python).
    """
    n = len(units)
    E = len(edges)

    # Consistent float dtype for the scan carry (gate may arrive float64 under
    # x64 while q0/p0 are float32 — promote all three to their common type so
    # the Verlet carry types match in both x32 and x64 regimes).
    dtype = jnp.result_type(q0, p0, gate)
    q0 = q0.astype(dtype)
    p0 = p0.astype(dtype)
    gate = gate.astype(dtype)

    def H(q, p, gates):
        total = units[0].T(p[offsets[0] : offsets[1]]) + units[0].potential_net(
            q[offsets[0] : offsets[1]]
        )
        for i in range(1, n):
            sl = slice(offsets[i], offsets[i + 1])
            total = total + units[i].T(p[sl]) + units[i].potential_net(q[sl])
        for e in range(E):
            i, j = edges[e]
            si = slice(offsets[i], offsets[i + 1])
            sj = slice(offsets[j], offsets[j + 1])
            total = total + gates[e] * couplings[e](q[si], q[sj])
        return total

    def one(q_init, p_init, gvec):
        def scan_fn(state, _):
            q, p = state
            H_now = H(q, p, gvec)
            gamma = sensitivity * jnp.tanh(jnp.maximum(0.0, H_now - floor))

            def Hg(qq, pp):
                return H(qq, pp, gvec)

            q_next, p_next = velocity_verlet_step(Hg, q, p, dt, gamma)
            q_next = jnp.where(clamp_mask, q_init, q_next)
            p_next = jnp.where(clamp_mask, 0.0, p_next)
            return (q_next, p_next), None

        (qf, pf), _ = jax.lax.scan(scan_fn, (q_init, p_init), None, length=steps)
        return qf, pf

    return jax.vmap(one)(q0, p0, gate)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def _build_and_write_units(cfg, config, key, n_units, e, dim):
    """Sample disjoint per-unit KV dictionaries and write each unit as an EBM.

    Returns (units, floors, dictionaries) where dictionaries[i] holds the
    per-unit keys/values (tokens + embeddings).
    """
    k_dict, k_units = jax.random.split(key)
    V = cfg.vocab_size
    n_keys_total = n_units * cfg.kv_per_unit
    kd1, kd2 = jax.random.split(k_dict)
    # Disjoint global draws: keys from [1, V/2), values from [V/2, V).
    all_keys = jax.random.choice(
        kd1, jnp.arange(1, V // 2), shape=(n_keys_total,), replace=False
    )
    all_vals = jax.random.choice(
        kd2, jnp.arange(V // 2, V), shape=(n_keys_total,), replace=False
    )

    units, floors, dicts = [], [], []
    ukeys = jax.random.split(k_units, n_units)
    for i in range(n_units):
        sl = slice(i * cfg.kv_per_unit, (i + 1) * cfg.kv_per_unit)
        keys_tok = all_keys[sl]
        vals_tok = all_vals[sl]
        key_emb = _embed_tokens(keys_tok)
        val_emb = _embed_tokens(vals_tok)
        stored = jnp.concatenate([key_emb, val_emb], axis=1)  # (kv, dim)
        km, kt = jax.random.split(ukeys[i])
        model = CHLU(
            dim=dim,
            hidden=cfg.hidden_dim,
            rest_mass=config.model.rest_mass,
            c=config.model.speed_of_causality,
            kinetic_mode=cfg.kinetic_energy_mode,
            potential_type=cfg.potential_type,
            key=km,
        )
        model, _losses, floor = train_generative(
            model,
            stored,
            key=kt,
            config=config,
            epochs=cfg.train_epochs,
            lr=cfg.train_lr,
            batch_size=cfg.train_batch_size,
            dt=cfg.dt,
            buffer_capacity=cfg.train_buffer_capacity,
            k_steps=cfg.train_k_steps,
            sleep_friction=cfg.train_friction,
            sleep_temperature=cfg.train_temperature,
            input_noise_sigma=cfg.train_input_noise_sigma,
        )
        units.append(model)
        floors.append(float(floor))
        dicts.append(
            {
                "keys_tok": np.asarray(keys_tok),
                "vals_tok": np.asarray(vals_tok),
                "key_emb": key_emb,
                "val_emb": val_emb,
                "stored": stored,
            }
        )
    return units, floors, dicts


# module-global embedding table set per (N, seed); avoids threading through
# every helper. Set in the driver before unit construction.
_EMBEDS = None


def _embed_tokens(tokens):
    return _EMBEDS[tokens]


def _make_queries(cfg, key, e, dict_local, dict_archive):
    """Assemble jittered local + distant queries.

    Returns dict with cue_emb (Q, e), true_tok (Q,), is_distant (Q,).
    """
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    per = cfg.trials_per_type
    sig = cfg.query_cue_noise

    def build(d, distant):
        kv = len(d["keys_tok"])
        idx = rng.integers(0, kv, size=per)
        base = np.asarray(d["key_emb"])[idx]  # (per, e)
        noise = sig * rng.standard_normal((per, e))
        cue = base + noise
        true_tok = np.asarray(d["vals_tok"])[idx]
        return cue, true_tok, np.full(per, int(distant))

    cl, tl, dl = build(dict_local, False)
    cd, td, dd = build(dict_archive, True)
    cue = np.concatenate([cl, cd], axis=0)
    true_tok = np.concatenate([tl, td])
    is_distant = np.concatenate([dl, dd])
    return {
        "cue": jnp.asarray(cue),
        "true_tok": true_tok.astype(np.int64),
        "is_distant": is_distant.astype(bool),
    }


def _fit_router_head(
    cfg, unit0, floor0, dict_local, other_dicts, e, val_embeds, val_tokens, dt, sens
):
    """Write-time self-test: fit a per-model head R0[,margin] -> p(route).

    Own-key jittered probes (label: route=False, the local memory holds them);
    impostor probes = keys from OTHER units (label: route=True, unit 0 has no
    binding). Reuses fit_calibration_head (p_wrong == p_route here).
    """
    rng = np.random.default_rng(0)
    scales = list(cfg.calib_cue_noise_scales)
    cues, labels = [], []
    # own keys -> should NOT route
    for kemb in np.asarray(dict_local["key_emb"]):
        for pi in range(cfg.calib_probes_per_key):
            s = scales[pi % len(scales)]
            cues.append(kemb + s * rng.standard_normal(e))
            labels.append(False)
    # impostor keys (other units) -> SHOULD route
    imp = np.concatenate([np.asarray(d["key_emb"]) for d in other_dicts], axis=0)
    for kemb in imp:
        for pi in range(cfg.calib_probes_per_key):
            s = scales[pi % len(scales)]
            cues.append(kemb + s * rng.standard_normal(e))
            labels.append(True)
    cues = np.stack(cues)
    labels = np.asarray(labels, dtype=bool)

    q0 = jnp.concatenate([jnp.asarray(cues), jnp.zeros((len(cues), e))], axis=1)
    p0 = jnp.zeros_like(q0)
    qf, _pf, Hf = _settle_batch(
        unit0, q0, p0, cfg.relax_steps, dt, jnp.asarray(floor0), sens, e
    )
    R = np.asarray(Hf) - float(floor0)
    _pred, margin = _decode_values(qf[:, e:], val_embeds, val_tokens)
    # "wrong" for own-key probes = decoded != any own value; simpler: use the
    # route label directly (impostors always route, own keys never) — the head
    # learns R (and margin) separation between the two.
    head = fit_calibration_head(
        R=R,
        margin=np.asarray(margin),
        wrong=labels,
        features=cfg.calib_features,
        l2=cfg.calib_l2,
    )
    return head


def run_experiment_v1_wormhole(
    config: Optional[CHLUConfig] = None,
    save_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    quick: Optional[bool] = None,
):
    """Run the V1 wormhole-routing experiment (see module docstring)."""
    global _EMBEDS
    if config is None:
        config = get_default_config()
    if save_dir is not None:
        config.project.save_dir = save_dir
    if seed is not None:
        config.project.seed = seed

    cfg = config.experiment_v1_wormhole
    if quick:
        cfg.n_units_values = [4]
        cfg.n_seeds = 1
        cfg.trials_per_type = 16
        cfg.kv_per_unit = 3
        cfg.train_epochs = min(cfg.train_epochs, 150)
        cfg.relax_steps = min(cfg.relax_steps, 150)
        cfg.route_steps = min(cfg.route_steps, 150)
        cfg.hidden_dim = min(cfg.hidden_dim, 64)

    save_dir = config.project.save_dir or "results/"
    models_dir = models_dir or os.path.join(save_dir, "..", "models")
    results_dir = os.path.join(save_dir, "..", "results")
    for d in (save_dir, models_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    base_seed = config.project.seed
    e = cfg.embed_dim
    dim = 2 * e
    dt = cfg.dt
    sens = cfg.governor_sensitivity

    print("\n" + "=" * 64)
    print("EXPERIMENT V1-WORMHOLE: energy-gated sparse non-local routing")
    print("=" * 64)
    print(
        f"vocab={cfg.vocab_size} embed_dim={e} -> unit dim={dim} | "
        f"kinetic={cfg.kinetic_energy_mode} potential={cfg.potential_type}"
    )
    print(
        f"N sweep={cfg.n_units_values} seeds={cfg.n_seeds} "
        f"kv/unit={cfg.kv_per_unit} trials/type={cfg.trials_per_type}"
    )

    arms = ["local_only", "gated", "dense", "chain", "calibrated"]
    # runs[N] = list of per-seed dict records
    runs = {}

    for N in cfg.n_units_values:
        runs[N] = []
        for si in range(cfg.n_seeds):
            seed_i = base_seed + si
            master = jax.random.PRNGKey(seed_i)
            k_embed, k_units, k_query = jax.random.split(master, 3)

            _EMBEDS = make_token_embeddings(
                k_embed, cfg.vocab_size, e, scale=cfg.embed_scale
            )
            val_tokens = jnp.arange(cfg.vocab_size // 2, cfg.vocab_size)
            val_embeds = _EMBEDS[val_tokens]

            print(f"\n[N={N} seed={seed_i}] writing {N} unit memories...")
            units, floors, dicts = _build_and_write_units(
                cfg, config, k_units, N, e, dim
            )
            archive = N - 1

            queries = _make_queries(cfg, k_query, e, dicts[0], dicts[archive])
            cue = queries["cue"]  # (Q, e)
            Q = cue.shape[0]
            is_distant = queries["is_distant"]
            true_tok = queries["true_tok"]

            offsets = tuple(int(i * dim) for i in range(N + 1))
            D = N * dim

            # ---- Phase 1: local relaxation in unit 0 (the routing signal) ----
            q0_local = jnp.concatenate([cue, jnp.zeros((Q, e))], axis=1)
            p0_local = jnp.zeros_like(q0_local)
            qf0, pf0, Hf0 = _settle_batch(
                units[0],
                q0_local,
                p0_local,
                cfg.relax_steps,
                dt,
                jnp.asarray(floors[0]),
                sens,
                e,
            )
            R0 = np.asarray(Hf0) - floors[0]
            pred_local, margin0 = _decode_values(qf0[:, e:], val_embeds, val_tokens)
            pred_local = np.asarray(pred_local)
            margin0 = np.asarray(margin0)

            # smooth gate on z-normalized R0
            z = _zscore(R0)
            g_smooth = smooth_gate(z, cfg.gate_z_threshold, cfg.gate_z_width)

            # ---- Build couplings on the KEY channel (query transport) ----
            # The wormhole is a GATED key-channel spring: it transports the
            # clamped query key from unit 0 to the archive's (free) key half, so
            # the archive relaxes to the matching stored pattern and its value
            # half fills in — the readout is then the *terminal* (archive) unit.
            # (A weak position coupling cannot drag a unit's value OUT of its
            # own attractor basin, so the honest read is the unit that actually
            # retrieves; the wormhole's value is being a DIRECT 1-hop edge vs the
            # chain's N-1-hop diffusion.)
            key_chan = tuple(range(0, e))
            wh_coupling = channel_spring_coupling(
                dim, dim, cfg.kappa_wormhole, channel=key_chan
            )
            chain_couplings = tuple(
                channel_spring_coupling(dim, dim, cfg.kappa_chain, channel=key_chan)
                for _ in range(N - 1)
            )

            floor_pair = floors[0] + floors[archive]
            floor_chain = float(sum(floors))
            route_thr = cfg.gate_route_threshold

            # --- wormhole sub-lattice (unit 0 <-> archive), only unit0 key pinned
            wh_units = (units[0], units[archive])
            wh_offsets = (0, dim, 2 * dim)
            wh_edges = ((0, 1),)
            wh_clamp = np.zeros(2 * dim, dtype=bool)
            wh_clamp[0:e] = True
            wh_clamp_j = jnp.asarray(wh_clamp)
            q0_wh = np.zeros((Q, 2 * dim), dtype=np.float32)
            q0_wh[:, 0:e] = np.asarray(cue)
            q0_wh_j = jnp.asarray(q0_wh)
            p0_wh_j = jnp.zeros((Q, 2 * dim), dtype=jnp.float32)

            # --- full chain lattice, only unit0 key pinned
            chain_clamp = np.zeros(D, dtype=bool)
            chain_clamp[0:e] = True
            chain_clamp_j = jnp.asarray(chain_clamp)
            q_init = np.zeros((Q, D), dtype=np.float32)
            q_init[:, 0:e] = np.asarray(cue)
            q_init_j = jnp.asarray(q_init)
            p_init_j = jnp.zeros((Q, D), dtype=jnp.float32)

            def route_wormhole(
                gate_vec,
                steps,
                _u=wh_units,
                _c=(wh_coupling,),
                _e=wh_edges,
                _o=wh_offsets,
                _q=q0_wh_j,
                _p=p0_wh_j,
                _cl=wh_clamp_j,
                _fl=floor_pair,
                _ve=val_embeds,
                _vt=val_tokens,
            ):
                """Route via the direct wormhole edge; read the ARCHIVE value
                half (the route terminal). Loop vars bound via defaults."""
                qf, _pf = _joint_settle(
                    _u,
                    _c,
                    _e,
                    _o,
                    _q,
                    _p,
                    jnp.asarray(gate_vec)[:, None],
                    steps,
                    dt,
                    jnp.asarray(_fl),
                    sens,
                    _cl,
                )
                pred, _m = _decode_values(qf[:, dim + e : 2 * dim], _ve, _vt)
                return np.asarray(pred), qf

            arm_pred = {}
            arm_cost = {}  # unit-steps (FLOP proxy: steps * active units)
            arm_verlet = {}  # raw Verlet steps

            # (a) local-only: no route, read unit 0
            arm_pred["local_only"] = pred_local
            arm_cost["local_only"] = np.full(Q, cfg.relax_steps * 1.0)
            arm_verlet["local_only"] = np.full(Q, float(cfg.relax_steps))

            # (b) gated wormhole: route iff g > threshold, then read the archive
            route_b = g_smooth > route_thr
            pred_arch_b, qf_b = route_wormhole(g_smooth, cfg.route_steps)
            arm_pred["gated"] = np.where(route_b, pred_arch_b, pred_local)
            arm_cost["gated"] = cfg.relax_steps * 1.0 + route_b * cfg.route_steps * 2.0
            arm_verlet["gated"] = (cfg.relax_steps + route_b * cfg.route_steps).astype(
                float
            )

            # (c) dense always-open (g = 1): always route + read archive, no gate
            pred_arch_c, _qf_c = route_wormhole(np.ones(Q), cfg.route_steps)
            arm_pred["dense"] = pred_arch_c
            arm_cost["dense"] = np.full(Q, cfg.route_steps * 2.0)
            arm_verlet["dense"] = np.full(Q, float(cfg.route_steps))

            # (d) chain multi-hop: same gate, but route THROUGH the chain (the
            # query key must diffuse hop-by-hop to reach the archive). Read
            # archive if routed else unit 0.
            chain_edges_lst = tuple((i, i + 1) for i in range(N - 1))
            qf_chain, _pf_chain = _joint_settle(
                tuple(units),
                chain_couplings,
                chain_edges_lst,
                offsets,
                q_init_j,
                p_init_j,
                jnp.ones((Q, N - 1)),
                cfg.route_steps,
                dt,
                jnp.asarray(floor_chain),
                sens,
                chain_clamp_j,
            )
            arch_val_sl = slice(offsets[archive] + e, offsets[archive] + dim)
            pred_chain_arch, _mc = _decode_values(
                qf_chain[:, arch_val_sl], val_embeds, val_tokens
            )
            arm_pred["chain"] = np.where(
                route_b, np.asarray(pred_chain_arch), pred_local
            )
            arm_cost["chain"] = (
                cfg.relax_steps * 1.0 + route_b * cfg.route_steps * float(N)
            )
            arm_verlet["chain"] = (cfg.relax_steps + route_b * cfg.route_steps).astype(
                float
            )

            # (e) calibrated tau-gate: learned head decides route via wormhole
            other_dicts = [dicts[archive]] + [dicts[k] for k in range(1, N - 1)]
            head = _fit_router_head(
                cfg,
                units[0],
                floors[0],
                dicts[0],
                other_dicts,
                e,
                val_embeds,
                val_tokens,
                dt,
                sens,
            )
            p_route = head.p_wrong(R=R0, margin=margin0)
            route_e = p_route > cfg.calib_p_route
            pred_arch_e, _qf_e = route_wormhole(route_e.astype(float), cfg.route_steps)
            arm_pred["calibrated"] = np.where(route_e, pred_arch_e, pred_local)
            arm_cost["calibrated"] = (
                cfg.relax_steps * 1.0 + route_e * cfg.route_steps * 2.0
            )
            arm_verlet["calibrated"] = (
                cfg.relax_steps + route_e * cfg.route_steps
            ).astype(float)

            # ---- energy injected through the OPEN gate (arm b, routed) ----
            # E_wh = g * V_c(q0_key, q_arch_key) at the settled routed state.
            base_vals = np.asarray(
                jax.vmap(wh_coupling)(qf_b[:, 0:dim], qf_b[:, dim : 2 * dim])
            )
            e_inj = np.asarray(g_smooth) * base_vals  # (Q,)

            # ---- per-arm accuracy split ----
            rec = {
                "N": N,
                "seed": seed_i,
                "Q": int(Q),
                "is_distant": is_distant,
                "true_tok": true_tok,
                "R0": R0,
                "margin0": margin0,
                "z": z,
                "g_smooth": np.asarray(g_smooth),
                "route_b": np.asarray(route_b),
                "route_e": np.asarray(route_e),
                "e_inj": e_inj,
                "arms": {},
                "auroc_R0_distant": _auroc(is_distant.astype(int), R0),
                "auroc_p_route_distant": _auroc(is_distant.astype(int), p_route),
                "floors": floors,
                "n_local": int((~is_distant).sum()),
                "n_distant": int(is_distant.sum()),
            }
            for a in arms:
                corr = arm_pred[a] == true_tok
                rec["arms"][a] = {
                    "correct": corr,
                    "cost": np.asarray(arm_cost[a], dtype=float),
                    "verlet": np.asarray(arm_verlet[a], dtype=float),
                    "acc": float(corr.mean()),
                    "acc_local": float(corr[~is_distant].mean()),
                    "acc_distant": float(corr[is_distant].mean()),
                    "mean_cost": float(np.mean(arm_cost[a])),
                    "mean_verlet": float(np.mean(arm_verlet[a])),
                }
            runs[N].append(rec)
            print(
                f"  Q={Q} (local {rec['n_local']}, distant {rec['n_distant']}) "
                f"AUROC(R0->distant)={rec['auroc_R0_distant']:.3f}"
            )
            for a in arms:
                ar = rec["arms"][a]
                print(
                    f"    {a:12s} acc={ar['acc']:.3f} "
                    f"(local {ar['acc_local']:.3f} / distant {ar['acc_distant']:.3f}) "
                    f"cost={ar['mean_cost']:.0f} unit-steps"
                )

    # -----------------------------------------------------------------
    # Aggregate + summarize
    # -----------------------------------------------------------------
    summary = {"seed": base_seed, "arms": arms, "by_N": {}}
    for N in cfg.n_units_values:
        recs = runs[N]
        entry = {"n_seeds": len(recs), "arms": {}}
        for a in arms:
            accs = np.array([r["arms"][a]["acc"] for r in recs])
            accs_l = np.array([r["arms"][a]["acc_local"] for r in recs])
            accs_d = np.array([r["arms"][a]["acc_distant"] for r in recs])
            costs = np.array([r["arms"][a]["mean_cost"] for r in recs])
            entry["arms"][a] = {
                "acc_mean": float(accs.mean()),
                "acc_std": float(accs.std()),
                "acc_local_mean": float(accs_l.mean()),
                "acc_distant_mean": float(accs_d.mean()),
                "cost_mean": float(costs.mean()),
            }
        entry["auroc_R0_distant_mean"] = float(
            np.nanmean([r["auroc_R0_distant"] for r in recs])
        )
        entry["auroc_p_route_distant_mean"] = float(
            np.nanmean([r["auroc_p_route_distant"] for r in recs])
        )
        # gate-selectivity confusion matrix (arm b): open=g>0.5 vs is_distant
        cm = np.zeros((2, 2), dtype=int)  # [open/closed, distant/local]
        e_inj_all = []
        for r in recs:
            opened = r["route_b"]
            dist = r["is_distant"]
            cm[0, 0] += int((opened & dist).sum())  # open, distant (TP)
            cm[0, 1] += int((opened & ~dist).sum())  # open, local (FP)
            cm[1, 0] += int((~opened & dist).sum())  # closed, distant (FN)
            cm[1, 1] += int((~opened & ~dist).sum())  # closed, local (TN)
            e_inj_all.append(r["e_inj"])
        e_inj_all = np.concatenate(e_inj_all)
        entry["gate_confusion"] = cm.tolist()
        opened_n = cm[0, 0] + cm[0, 1]
        entry["gate_precision_distant"] = (
            float(cm[0, 0] / opened_n) if opened_n else float("nan")
        )
        entry["gate_recall_distant"] = (
            float(cm[0, 0] / (cm[0, 0] + cm[1, 0]))
            if (cm[0, 0] + cm[1, 0])
            else float("nan")
        )
        entry["energy_injected_mean"] = float(np.mean(e_inj_all))
        entry["energy_injected_max"] = float(np.max(e_inj_all))
        summary["by_N"][str(N)] = entry

    # -----------------------------------------------------------------
    # Save metrics + summary
    # -----------------------------------------------------------------
    npz = {}
    for N in cfg.n_units_values:
        for si, r in enumerate(runs[N]):
            pre = f"N{N}_s{si}_"
            npz[pre + "is_distant"] = r["is_distant"]
            npz[pre + "R0"] = r["R0"]
            npz[pre + "z"] = r["z"]
            npz[pre + "g_smooth"] = r["g_smooth"]
            npz[pre + "e_inj"] = r["e_inj"]
            for a in arms:
                npz[pre + a + "_correct"] = r["arms"][a]["correct"]
                npz[pre + a + "_cost"] = r["arms"][a]["cost"]
    metrics_path = os.path.join(results_dir, "exp_v1_wormhole_metrics.npz")
    np.savez(metrics_path, **npz)
    summary_path = os.path.join(results_dir, "exp_v1_wormhole_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}\nSaved summary to {summary_path}")

    # a lightweight checkpoint of the last-built lattice's unit 0 (provenance)
    try:
        ckpt = os.path.join(models_dir, f"v1wormhole_unit0_seed{base_seed}.pkl")
        save_checkpoint(
            units[0],
            ckpt,
            epoch=cfg.train_epochs,
            loss=0.0,
            config=None,
            target_energy=floors[0],
        )
    except Exception as ex:  # non-fatal
        print(f"  (checkpoint skipped: {ex})")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    plot_v1_wormhole_cost_accuracy(
        summary,
        arms,
        os.path.join(save_dir, "exp_v1_wormhole_cost_accuracy.png"),
    )
    plot_v1_wormhole_selectivity(
        runs,
        summary,
        cfg.n_units_values,
        os.path.join(save_dir, "exp_v1_wormhole_selectivity.png"),
    )

    # -----------------------------------------------------------------
    # Headline read
    # -----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("V1 WORMHOLE READ (heuristics; final read is the Hub's)")
    print("=" * 64)
    for N in cfg.n_units_values:
        ent = summary["by_N"][str(N)]
        print(
            f"\n N={N} (AUROC R0->distant={ent['auroc_R0_distant_mean']:.3f}, "
            f"gate precision/recall for distant = "
            f"{ent['gate_precision_distant']:.2f}/{ent['gate_recall_distant']:.2f})"
        )
        for a in arms:
            ar = ent["arms"][a]
            print(
                f"   {a:12s} acc={ar['acc_mean']:.3f}±{ar['acc_std']:.3f} "
                f"(local {ar['acc_local_mean']:.3f} / distant "
                f"{ar['acc_distant_mean']:.3f}) cost={ar['cost_mean']:.0f}"
            )
        print(
            f"   energy injected (open gate): mean={ent['energy_injected_mean']:.3f} "
            f"max={ent['energy_injected_max']:.3f} (bounded)"
        )
    print("=" * 64 + "\n")

    return {"runs": runs, "summary": summary}
