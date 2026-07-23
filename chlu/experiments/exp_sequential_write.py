"""Experiment SEQUENTIAL-WRITE: does an admission gate stop writes destroying items?

**The claim under test (w21).** w20's worst result — write A, write B, A is
destroyed (strict ``1.000 -> 0.000``, `learned-landscape-write-read` §3) — was
measured with **ungated** writes. The theorist measured the same contrast with an
MVC-0 admission gate and got max fixed-point drift ``8.0e-5`` and selectivity
``1.000`` at every K (`clu-controller-spec` §C3/§C5). If the gate transfers, our
worst negative becomes a statement about *everyone else's* primitive, since
MLPs / transformers / RNNs all write to latent state with no such gate.

Four items:

1. ``item1_gate_on_w20``   the EXACT w20 §3 protocol (write A on a K-site ring,
                           write B, re-read A) plus the MVC-0 rules — the
                           spacing gate with refuse-and-relocate (C5-A1/A2) and
                           the C3 admissibility check on stored items. Gated vs
                           ungated, >= 5 seeds.
2. ``item2_sequential``    K = 1..16 items written ONE AT A TIME, re-reading all
                           previously stored items after every write. Retention
                           of item 1 vs number of subsequent writes; arms
                           designed/learned x gated/ungated.
3. ``item3_cross_primitive`` the same sequential-write protocol against
                           transformer / GRU / MLP / CLU in the `primitive-harness`
                           drop-in slot at matched parameters, with
                           compute-to-criterion and retrieval-cost scaling in K.
4. ``item4_retrieval_cost`` parametric vs contextual read cost as K grows.

⚠ **SCOPE (required, `Item 4` of the task; a referee will raise it).** Every CLU
write in this module is **PARAMETRIC** — into ``V_theta`` or into a block's
weights, at training time. Attention's headline memory is **CONTEXTUAL** — a KV
cache written at inference. These are different capabilities. The defensible
comparison here is against other *parametric* stores (MLP/FFN) and fixed-state
recurrences (GRU); the transformer's parametric arm is the one compared on
retention, and its contextual arm appears only in the cost measurement, labelled.

⚠ **Method (w20 finding, enforced here).** Blank controls are taken over the
STRONGEST read on every reported cell, and the primary metric is
**value recovery** (did the stored NUMBER come back), never classification —
an arbitrarily small address leak is a perfect item code.

Runnable directly::

    uv run python -m chlu.experiments.exp_sequential_write --quick

or via the CLI: ``chlu exp-sequential-write [--project N] [--seed I] [--quick]``.
"""

import copy
import json
import os
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import CHLUConfig, get_default_config
from chlu.core.admission import (
    admit_site,
    c3_drift,
    disk_proposer,
    min_separation,
    ring_proposer,
)
from chlu.core.memory_potentials import (
    AtomDictionaryPotential,
    designed_payloads,
    ring_sites,
)
from chlu.experiments.exp_learned_memory import build_landscape, model_for
from chlu.experiments.exp_retrieval import nearest_centroid_read, tail_features
from chlu.training.train_memory import train_memory_landscape

# ---------------------------------------------------------------------------
# Shared: site-explicit queries, reads and retention scoring
# ---------------------------------------------------------------------------


def make_queries_at(key, sites, n_per_item: int, cfg, dim: int = 3):
    """Perturbed queries around explicit sites. ``q2 = p2 = 0`` ALWAYS (the guard).

    Unlike ``exp_learned_memory.make_queries`` this takes the sites verbatim
    instead of regenerating a ring, because sequential writes put items at
    arbitrary (gate-chosen) locations.
    """
    s = jnp.asarray(sites, dtype=jnp.float32)
    K = s.shape[0]
    k_q, k_p = jax.random.split(key, 2)
    n = K * n_per_item
    labels = np.repeat(np.arange(K), n_per_item)
    Q0 = s[jnp.asarray(labels)]
    jit = jax.random.normal(k_q, (n, dim)) * (cfg.f * cfg.query_sigma_theta)
    Q0 = (Q0 + jit).at[:, 2].set(0.0)
    P0 = jnp.zeros((n, dim))
    P0 = P0.at[:, :2].set(jax.random.normal(k_p, (n, 2)) * cfg.query_sigma_p)
    return Q0, P0, labels


def basin_of(addr_q, sites):
    """Nearest stored site in the ADDRESS PLANE (q0,q1) — scored against ALL
    stored sites, so landing in a freshly written neighbour counts as a miss."""
    s = np.asarray(sites)[:, :2]
    d = ((np.asarray(addr_q)[:, None, :2] - s[None, :, :]) ** 2).sum(-1)
    return np.argmin(d, axis=1)


def two_phase(model, Q0, P0, cfg, gamma_address, gamma_read):
    """``query -> [gamma_address relaxation] -> ADDRESS -> [gamma_read rollout]``."""
    dim = Q0.shape[1]

    def relax(q, p):
        tr = model(q, p, cfg.address_steps, cfg.dt, gamma_address)
        return tr[-1, :dim], tr[-1, dim:]

    addr_q, addr_p = jax.vmap(relax)(Q0, P0)
    traj = jax.vmap(lambda q, p: model(q, p, cfg.read_steps, cfg.dt, gamma_read))(
        addr_q, addr_p
    )
    return addr_q, traj


def effective_payload_tol(cfg, payloads) -> float:
    """Value-recovery tolerance, capped at a fraction of the CODEBOOK SPACING.

    ⚠ A tolerance wider than half the codebook spacing makes "the stored value
    came back" ambiguous: a read that is within ``payload_tol`` of item i can be
    closer to item j's codeword. w20 ran K <= 8 on ``[-1, 1]`` (spacing 0.286)
    with ``payload_tol = 0.1``, i.e. 0.35 x spacing; at K = 16 the same absolute
    tolerance would be 0.75 x spacing. The cap keeps the criterion at the w20
    *ratio*, which is stricter, never looser -- it cannot manufacture a positive.
    """
    p = np.unique(np.asarray(payloads))
    if p.size < 2:
        return float(cfg.payload_tol)
    return float(min(cfg.payload_tol, cfg.payload_tol_frac * np.min(np.diff(p))))


def evaluate_items(model, sites, payloads, cfg, seed, n_query=None, dim: int = 3):
    """Retention scoring on explicit sites. **Per-item**, never only aggregated.

    Primary metric is the leak-immune VALUE criterion: an item is retained iff
    the relaxed query lands in its own basin AND the read-out value is within
    ``payload_tol`` of the stored number.
    """
    n_query = n_query or cfg.n_query_per_item
    K = len(payloads)
    key = jax.random.PRNGKey(seed)
    Q0, P0, labels = make_queries_at(key, sites, n_query, cfg, dim=dim)
    addr_q, traj = two_phase(model, Q0, P0, cfg, cfg.gamma_address, cfg.gamma_read)

    basin_ok = basin_of(addr_q, sites) == labels
    f_pay = tail_features(traj, cfg.tail_frac, cfg.n_subsample, coords=[2])
    read_val = np.asarray(f_pay).mean(axis=1)
    pay = np.asarray(payloads)
    err = np.abs(read_val - pay[labels])
    tol = effective_payload_tol(cfg, payloads)
    strict = basin_ok & (err < tol)

    per_item = [
        {
            "item": int(k),
            "basin": float(np.mean(basin_ok[labels == k])),
            "strict": float(np.mean(strict[labels == k])),
            "abs_err": float(np.mean(err[labels == k])),
        }
        for k in range(K)
    ]
    acc_nc = (
        nearest_centroid_read(f_pay[::2], labels[::2], f_pay[1::2], labels[1::2], K)
        if K > 1
        else float("nan")
    )
    return {
        "K": K,
        "payload_tol_used": tol,
        "finite": bool(np.all(np.isfinite(np.asarray(traj)))),
        "per_item": per_item,
        "mean_strict": float(np.mean(strict)),
        "mean_basin": float(np.mean(basin_ok)),
        "mean_abs_err": float(np.mean(err)),
        "acc_nearest_centroid": acc_nc,
    }


# ---------------------------------------------------------------------------
# The write operators (one designed, one learned) + the controller around them
# ---------------------------------------------------------------------------


def _write_kwargs(cfg, anchors=None):
    return dict(
        n_perturb=cfg.write_n_perturb,
        sigma_addr=cfg.write_sigma_addr,
        sigma_pay=cfg.write_sigma_pay,
        margin=cfg.write_margin,
        barrier=cfg.write_barrier,
    )


def learned_write(V, targets, key, cfg, steps=None, anchors=None):
    """One learned write: Adam on ``V.learned`` to make ``targets`` minima.

    ``anchors`` (C3 option (b) of the controller spec — "anchors R-2/R-3 on
    stored items during writes") are appended to the target set, i.e. the
    controller REHEARSES from its codebook while writing. This is a *structured
    write operator*, not an admission gate, and is reported as such.
    """
    tg = jnp.asarray(targets, dtype=jnp.float32)
    if anchors is not None and len(anchors) > 0:
        tg = jnp.concatenate([tg, jnp.asarray(anchors, dtype=jnp.float32)], axis=0)
    return train_memory_landscape(
        V,
        tg,
        key,
        steps=cfg.write_steps if steps is None else steps,
        lr=cfg.write_lr,
        weight_decay=cfg.write_weight_decay,
        loss_kwargs=_write_kwargs(cfg),
    )


def c3_gated_learned_write(V, targets, key, cfg, stored_states, steps, anchors=None):
    """A learned write run in CHUNKS, stopped the moment C3 would be violated.

    The controller cannot make a global-support write local, but it *can* compute
    ``||H_i^-1 grad dV(q*_i)||`` for every stored item (C3: "the controller can
    check this even when it cannot guarantee it") and stop. This turns an
    inadmissible write into a **refusal or a truncation** — the only two things
    an admission gate is able to do about a write operator it cannot change.
    """
    V0 = V
    chunk = max(1, int(cfg.c3_chunk_steps))
    done, drift = 0, 0.0
    while done < steps:
        n = min(chunk, steps - done)
        key, k = jax.random.split(key)
        V_try, _ = learned_write(V, targets, k, cfg, steps=n, anchors=anchors)
        if len(stored_states) > 0:
            d = float(np.max(c3_drift(V0, V_try, stored_states)))
        else:
            d = 0.0
        if d > cfg.delta_budget:
            # Refuse this chunk and stop. `drift` is the drift of the landscape
            # actually committed; `d` (returned as the rejected drift) is what
            # the controller declined -- reporting it is what makes "the very
            # first chunk already violates the budget" a measurement.
            return V, done, drift, True, d
        V, done, drift = V_try, done + n, d
    return V, done, drift, False, drift


# ---------------------------------------------------------------------------
# ITEM 1 — the gate on the EXACT w20 failing setup
# ---------------------------------------------------------------------------


def item1_gate_on_w20(cfg, seeds, dim: int = 3):
    """w20 §3 verbatim (write A on a K-site ring, write B, re-read A), + MVC-0.

    Arms:
      ``ungated``            — w20 exactly: continue training on B's target.
      ``gated_spacing``      — C5-A1/A2 admission (refuse-and-relocate) on B's site.
      ``gated_c3``           — spacing gate + the C3 chunked-stop check.
      ``anchored``           — spacing gate + C3 + stored items anchored in the
                               write loss (a structured write operator, C3(b)).

    Proposal modes: ``ring`` (B at the K-th ring site — the w20 geometry, where
    the spacing gate is arithmetically vacuous) and ``disk`` (B proposed
    uniformly in a disk — the theorist's crowded regime, where it is not).
    """
    K = cfg.interference_K
    pay_all = designed_payloads(K, seed=cfg.payload_seed)
    pay_A = pay_all[:-1]
    sites_all = np.asarray(ring_sites(K, f=cfg.f, dim=dim, payloads=pay_all))
    d_safe = cfg.d_safe_mult * cfg.write_sigma_addr

    rows, blanks = [], []
    for rung in cfg.gate_rungs:
        for proposal in cfg.gate_proposals:
            for arm in cfg.gate_arms:
                for seed in seeds:
                    rows.append(
                        _gate_cell(
                            cfg,
                            rung,
                            proposal,
                            arm,
                            seed,
                            K,
                            pay_all,
                            pay_A,
                            sites_all,
                            d_safe,
                            dim,
                        )
                    )
                # ⚠ Blank control on every (rung, proposal, arm), at the first
                # seed: identical writes, NOTHING stored, scored against the
                # real codebook. A cell whose blank passes the value read is not
                # a measurement (w20 method finding).
                blanks.append(
                    _gate_cell(
                        cfg,
                        rung,
                        proposal,
                        arm,
                        seeds[0],
                        K,
                        pay_all,
                        pay_A,
                        sites_all,
                        d_safe,
                        dim,
                        blank=True,
                    )
                )
    return {
        "d_safe": float(d_safe),
        "ring_min_spacing": float(2.0 * cfg.f * np.sin(np.pi / K)),
        "spacing_gate_is_vacuous_on_ring": bool(
            2.0 * cfg.f * np.sin(np.pi / K) >= d_safe
        ),
        "w20_reference": {
            "designed_ungated_corruption": 0.0,
            "learned_ungated_corruption_band": [2.9e-2, 5.0e-1],
            "sites_learned_payload_strict_after_B": 0.0,
        },
        "theorist_reference": {"designed_gated_max_drift": 8.0e-5},
        "rows": rows,
        "blank_rows": blanks,
    }


def _gate_cell(
    cfg,
    rung,
    proposal,
    arm,
    seed,
    K,
    pay_all,
    pay_A,
    sites_all,
    d_safe,
    dim,
    blank: bool = False,
):
    """One (rung, proposal, arm, seed) cell of the w20 interference protocol.

    ``blank=True`` is the CONTROL: identical architecture, identical writes, but
    **nothing stored** (all written payloads zero) -- and it is scored against
    the REAL codebook, so any strict success it shows is the read recovering the
    address rather than the content. ⚠ Scoring a blank against the zeros it was
    written with reports 1.000 for an empty landscape; that is a tautology, not
    a control, and it is one of the two blank-control bugs this harness hit.
    """
    key = jax.random.PRNGKey(seed)
    k_v, k_a, k_b, k_prop, k_reloc = jax.random.split(key, 5)

    write_pay = np.zeros_like(np.asarray(pay_all)) if blank else np.asarray(pay_all)
    sites_w = np.array(sites_all, dtype=float)
    sites_w[:, 2] = write_pay

    V = build_landscape(rung, cfg, write_pay, k_v, dim=dim)
    V, _ = learned_write(V, sites_w[:-1], k_a, cfg)
    stored = sites_all[:-1]

    # ---- the PROPOSAL for item B ----
    if proposal == "ring":
        q_new = sites_all[-1].copy()
        proposer = ring_proposer(cfg.f, dim)
    else:
        p = disk_proposer(cfg.proposal_radius, dim)(k_prop, 1)
        q_new = p[0]
        proposer = disk_proposer(cfg.proposal_radius, dim)
    q_new = np.asarray(q_new, dtype=float)
    q_new[2] = float(pay_all[-1])

    # ---- the CONTROLLER ----
    if arm == "ungated":
        decision = {
            "decision": "admit",
            "site": q_new[:2],
            "d_min_proposed": min_separation(q_new[:2], stored[:, :2]),
            "d_min_written": min_separation(q_new[:2], stored[:, :2]),
            "n_candidates_examined": 0,
        }
    else:
        decision = admit_site(
            q_new[:2],
            stored[:, :2],
            d_safe,
            key=k_reloc,
            proposer=lambda k, n: np.asarray(proposer(k, n))[:, :2],
            n_candidates=cfg.n_relocation_candidates,
        )

    before = evaluate_items(
        model_for(V, dim), sites_all[:-1], pay_A, cfg, seed, dim=dim
    )
    q_star_stored = stored  # the write targets ARE the recorded addresses (MVC-0)

    truncated, c3_max, steps_used, c3_rejected = False, 0.0, 0, 0.0
    if decision["decision"] == "refuse":
        V2 = V
    else:
        site_B = np.zeros(dim)
        site_B[:2] = decision["site"]
        site_B[2] = float(write_pay[-1])
        anchors = sites_w[:-1] if arm == "anchored" else None
        if arm in ("gated_c3", "anchored"):
            V2, steps_used, c3_max, truncated, c3_rejected = c3_gated_learned_write(
                V,
                site_B[None, :],
                k_b,
                cfg,
                q_star_stored,
                cfg.interference_write_steps,
                anchors=anchors,
            )
        else:
            V2, _ = learned_write(
                V,
                site_B[None, :],
                k_b,
                cfg,
                steps=cfg.interference_write_steps,
                anchors=anchors,
            )
            steps_used = cfg.interference_write_steps
            c3_max = (
                float(np.max(c3_drift(V, V2, q_star_stored)))
                if len(q_star_stored)
                else 0.0
            )
            c3_rejected = c3_max

    after = evaluate_items(
        model_for(V2, dim), sites_all[:-1], pay_A, cfg, seed, dim=dim
    )
    # was B itself stored?
    if decision["decision"] == "refuse":
        strict_B = 0.0
    else:
        site_B = np.zeros((1, dim))
        site_B[0, :2] = decision["site"]
        site_B[0, 2] = float(pay_all[-1])
        all_sites = np.concatenate([stored, site_B], axis=0)
        b_eval = evaluate_items(
            model_for(V2, dim), all_sites, np.asarray(pay_all), cfg, seed, dim=dim
        )
        strict_B = b_eval["per_item"][-1]["strict"]

    # measured vs predicted drift of the stored fixed points
    meas = _measured_drift(model_for(V2, dim), q_star_stored, cfg, dim)
    pred = c3_drift(V, V2, q_star_stored) if len(q_star_stored) else np.zeros((0,))
    ratio = (
        float(np.median((pred + 1e-12) / (meas + 1e-12))) if len(meas) else float("nan")
    )

    return {
        "rung": rung,
        "proposal": proposal,
        "arm": arm,
        "seed": int(seed),
        "blank": bool(blank),
        "decision": decision["decision"],
        "d_min_proposed": float(decision["d_min_proposed"]),
        "d_min_written": float(decision["d_min_written"]),
        "write_steps_used": int(steps_used),
        "write_steps_budget": int(cfg.interference_write_steps),
        "c3_truncated": bool(truncated),
        "c3_predicted_max_drift": float(c3_max),
        "c3_rejected_drift": float(c3_rejected),
        "measured_max_drift": float(np.max(meas)) if len(meas) else 0.0,
        "drift_pred_over_meas_median": ratio,
        "read_err_A_before_B": before["mean_abs_err"],
        "read_err_A_after_B": after["mean_abs_err"],
        "corruption_of_A_by_writing_B": abs(
            after["mean_abs_err"] - before["mean_abs_err"]
        ),
        "strict_A_before_B": before["mean_strict"],
        "strict_A_after_B": after["mean_strict"],
        "strict_drop": before["mean_strict"] - after["mean_strict"],
        "strict_B": float(strict_B),
    }


def _measured_drift(model, q_stars, cfg, dim):
    """Where the stored minima actually move to under the post-write landscape."""
    q = np.atleast_2d(np.asarray(q_stars, dtype=np.float32))
    if q.shape[0] == 0:
        return np.zeros((0,))
    p0 = jnp.zeros_like(jnp.asarray(q))

    def relax(qq, pp):
        tr = model(qq, pp, cfg.address_steps, cfg.dt, cfg.gamma_address)
        return tr[-1, :dim]

    end = np.asarray(jax.vmap(relax)(jnp.asarray(q), p0))
    return np.sqrt(((end - q) ** 2).sum(-1))


# ---------------------------------------------------------------------------
# ITEM 2 — the sequential-write curve (the deliverable)
# ---------------------------------------------------------------------------


def _propose_sequence(key, n, cfg, dim):
    """The SAME proposal sequence for every arm at a given seed (paired design)."""
    return np.asarray(disk_proposer(cfg.proposal_radius, dim)(key, n))


def sequential_run(arm, cfg, seed, dim: int = 3, blank: bool = False):
    """Write K items ONE AT A TIME, re-reading every stored item after each write.

    ``arm`` in ``{designed_gated, designed_ungated, learned_gated, learned_ungated}``.
    Returns the per-write history: which items were admitted, and the retention
    of every stored item after every write.
    """
    K = cfg.n_sequential_items
    designed = arm.startswith("designed")
    gated = "_gated" in arm
    # "anchored" = the controller REHEARSES the stored items from its codebook
    # while writing (C3 option (b), a structured write operator -- NOT a gate).
    anchored = "anchored" in arm
    key = jax.random.PRNGKey(seed)
    k_prop, k_v, k_w, k_reloc = jax.random.split(key, 4)

    proposals = _propose_sequence(k_prop, K, cfg, dim)
    # ⚠ The blank control writes ZEROS but is scored against the REAL codebook.
    # Scoring it against the zeros it was written with reports 1.000 for an empty
    # landscape (measured: designed_gated blank 1.000) -- that is a tautology,
    # not a control. `write_pay` goes into V; `payloads_all` does the scoring.
    payloads_all = np.asarray(designed_payloads(K, seed=cfg.payload_seed))
    write_pay = np.zeros_like(payloads_all) if blank else payloads_all

    if designed:
        V = AtomDictionaryPotential(
            dim=dim,
            capacity=K,
            alpha=cfg.atom_alpha,
            s=cfg.atom_width,
            kappa=cfg.payload_kappa,
        )
        d_safe = cfg.d_safe_mult * cfg.atom_width
    else:
        V = build_landscape(cfg.sequential_rung, cfg, write_pay, k_v, dim=dim)
        d_safe = cfg.d_safe_mult * cfg.write_sigma_addr

    stored_sites, stored_pay, history, decisions = [], [], [], []
    for i in range(K):
        q_new = proposals[i].copy()
        q_new[2] = payloads_all[i]
        stored_arr = np.stack(stored_sites) if stored_sites else np.zeros((0, dim))
        if gated:
            k_reloc, kr = jax.random.split(k_reloc)
            dec = admit_site(
                q_new[:2],
                stored_arr[:, :2] if len(stored_sites) else np.zeros((0, 2)),
                d_safe,
                key=kr,
                proposer=lambda k, n, d=dim: np.asarray(
                    disk_proposer(cfg.proposal_radius, d)(k, n)
                )[:, :2],
                n_candidates=cfg.n_relocation_candidates,
            )
        else:
            dm = min_separation(
                q_new[:2], stored_arr[:, :2] if len(stored_sites) else np.zeros((0, 2))
            )
            dec = {
                "decision": "admit",
                "site": q_new[:2],
                "d_min_proposed": dm,
                "d_min_written": dm,
                "n_candidates_examined": 0,
            }
        decisions.append(
            {
                "write": i,
                "decision": dec["decision"],
                "d_min_proposed": float(dec["d_min_proposed"]),
            }
        )
        if dec["decision"] == "refuse":
            # A refusal is a correct controller output. Nothing is written, so
            # nothing can be damaged; the item simply does not enter the store.
            if stored_sites:
                ev = evaluate_items(
                    model_for(V, dim),
                    np.stack(stored_sites),
                    np.asarray(stored_pay),
                    cfg,
                    seed,
                    n_query=cfg.n_query_sequential,
                    dim=dim,
                )
                history.append(_hist_row(i, "refuse", ev, stored_pay, 0))
            continue

        site = np.zeros(dim)
        site[:2] = dec["site"]
        site[2] = write_pay[i]

        steps_used = 0
        if designed:
            V = V.with_item(site[:2], float(write_pay[i]), amp=cfg.atom_amp)
        else:
            k_w, kw = jax.random.split(k_w)
            anchors = np.stack(stored_sites) if (anchored and stored_sites) else None
            if gated:
                V, steps_used, _, _, _ = c3_gated_learned_write(
                    V,
                    site[None, :],
                    kw,
                    cfg,
                    np.stack(stored_sites) if stored_sites else np.zeros((0, dim)),
                    cfg.sequential_write_steps,
                    anchors=anchors,
                )
            else:
                V, _ = learned_write(
                    V,
                    site[None, :],
                    kw,
                    cfg,
                    steps=cfg.sequential_write_steps,
                    anchors=anchors,
                )
                steps_used = cfg.sequential_write_steps
        stored_sites.append(site)
        stored_pay.append(float(payloads_all[i]))

        ev = evaluate_items(
            model_for(V, dim),
            np.stack(stored_sites),
            np.asarray(stored_pay),
            cfg,
            seed,
            n_query=cfg.n_query_sequential,
            dim=dim,
        )
        history.append(_hist_row(i, dec["decision"], ev, stored_pay, steps_used))

    # ⚠ "admitted" is a PLACEMENT decision. On the learned arms the C3 check can
    # admit a site and then refuse the write itself, so the number of items
    # actually committed to the landscape is a separate count.
    n_written = sum(
        1
        for h in history
        if h["decision"] != "refuse" and (designed or h["write_steps_used"] > 0)
    )
    return {
        "arm": arm,
        "seed": int(seed),
        "d_safe": float(d_safe),
        "n_proposed": int(K),
        "n_admitted": len(stored_sites),
        "n_written": int(n_written),
        "decisions": decisions,
        "history": history,
        "final_sites": np.stack(stored_sites).tolist() if stored_sites else [],
    }


def _hist_row(i, decision, ev, stored_pay, steps_used=0):
    return {
        "write_index": int(i),
        "decision": decision,
        "write_steps_used": int(steps_used),
        "n_stored": len(stored_pay),
        "mean_strict": ev["mean_strict"],
        "mean_basin": ev["mean_basin"],
        "mean_abs_err": ev["mean_abs_err"],
        "item1_strict": ev["per_item"][0]["strict"],
        "item1_abs_err": ev["per_item"][0]["abs_err"],
        "per_item_strict": [d["strict"] for d in ev["per_item"]],
    }


def item2_sequential(cfg, seeds, dim: int = 3):
    """Retention of item 1 vs number of subsequent writes, per arm, >= 5 seeds."""
    runs, blanks = [], []
    for arm in cfg.sequential_arms:
        for seed in seeds:
            runs.append(sequential_run(arm, cfg, seed, dim=dim))
        # ONE blank control per arm (identical writes, all payloads zero, scored
        # against the REAL codebook) — the leak-immune control of the w20 method.
        blanks.append(sequential_run(arm, cfg, seeds[0], dim=dim, blank=True))
    pay = np.asarray(designed_payloads(cfg.n_sequential_items, seed=cfg.payload_seed))
    return {
        "runs": runs,
        "blank_runs": blanks,
        "curve": _retention_curve(runs, cfg),
        "blank_is_informative": blank_is_informative(cfg, pay),
        "min_abs_codeword": float(np.min(np.abs(pay))),
        "payload_tol_used": effective_payload_tol(cfg, pay),
    }


def blank_is_informative(cfg, payloads) -> bool:
    """Can a landscape holding NOTHING accidentally match a codeword?

    A blank store reads ~0 in the payload channel, so if any codeword lies
    within the value tolerance of 0 the blank "retains" that item for free and
    the control is uninformative for it. (Measured at K=5, where
    ``designed_payloads`` puts an exact 0 in the codebook: blank item-1
    retention 1.000 on an empty landscape.) At the reported K = 4 and K = 16 the
    grid excludes 0 by 0.333 and 0.067 respectively, both above the tolerance.
    """
    tol = effective_payload_tol(cfg, payloads)
    return bool(np.min(np.abs(np.asarray(payloads))) > tol)


def _retention_curve(runs, cfg):
    """Retention of item 1 (and mean over stored items) vs #subsequent writes."""
    out = {}
    for arm in sorted({r["arm"] for r in runs}):
        rs = [r for r in runs if r["arm"] == arm]
        n_after = cfg.n_sequential_items
        item1, meanret, nstored = [], [], []
        for j in range(n_after):
            v1, vm, ns = [], [], []
            for r in rs:
                # State after write attempt j (j = 0 is immediately after the
                # first item was written; a REFUSED attempt still produces a row,
                # because the controller re-reads the store either way).
                h = {h["write_index"]: h for h in r["history"]}.get(j)
                if h is None:
                    continue
                v1.append(h["item1_strict"])
                vm.append(h["mean_strict"])
                ns.append(h["n_stored"])
            if v1:
                item1.append([float(np.mean(v1)), float(np.std(v1))])
                meanret.append([float(np.mean(vm)), float(np.std(vm))])
                nstored.append(float(np.mean(ns)))
        out[arm] = {
            "item1_strict_mean_std": item1,
            "mean_retention_mean_std": meanret,
            "n_stored": nstored,
            "n_admitted_mean": float(np.mean([r["n_admitted"] for r in rs])),
            "n_admitted_std": float(np.std([r["n_admitted"] for r in rs])),
        }
    return out


# ---------------------------------------------------------------------------
# ITEM 3 — the cross-primitive comparison (the `primitive-harness` slot, reused)
# ---------------------------------------------------------------------------


def kv_dataset(key, n_items: int, key_len: int, vocab: int):
    """``n_items`` distinct key-sequences, each with a distinct value token.

    Keys are sequences (not single tokens) so the sequence-mixing machinery of
    every primitive is actually exercised; the value is read at the last
    position. This is a **parametric** store: an item is written by gradient
    updates into the weights, not placed in a context window.
    """
    k_k, k_v = jax.random.split(key, 2)
    keys = jax.random.randint(k_k, (n_items, key_len), 1, vocab)
    # distinct values, sampled without replacement
    vals = jax.random.choice(k_v, vocab - 1, (n_items,), replace=False) + 1
    return np.asarray(keys), np.asarray(vals)


def _kv_slot(vocab, key_len):
    class _Slot:
        model_kwargs = dict(vocab_size=vocab, out_dim=vocab, max_len=key_len)
        name = "kv_store"

    return _Slot()


def sequential_write_primitive(primitive, cfg, hcfg, seed, lr, n_items=None):
    """Write ``n_items`` key->value pairs into a primitive ONE AT A TIME.

    Each write runs Adam on the single new item until it is correct (criterion),
    up to ``kv_max_write_steps``. After every write, retention of ALL stored
    items is probed. Two compute measurements are taken:

    * ``steps_to_criterion``   — steps for item i alone to become correct.
    * ``steps_to_joint``       — steps for item i to become correct **without**
      any of items 1..i-1 being incorrect (the Head's "wasted compute
      reorganising to conserve key info"). ``None`` = never reached in budget
      (censored — reported, not dropped).
    """
    import equinox as eqx
    import optax

    from chlu.experiments.exp_primitive_harness import build_model, match_width

    n_items = n_items or cfg.kv_n_items
    slot = _kv_slot(cfg.kv_vocab, cfg.kv_key_len)
    mkey = jax.random.PRNGKey(seed)
    width, bp, tp, err = match_width(primitive, hcfg, slot, mkey)
    model = build_model(primitive, hcfg, slot, width, mkey)

    keys_arr, vals = kv_dataset(
        jax.random.PRNGKey(seed + 7919), n_items, cfg.kv_key_len, cfg.kv_vocab
    )
    X = jnp.asarray(keys_arr)
    Y = jnp.asarray(vals)

    opt = optax.chain(optax.clip_by_global_norm(hcfg.grad_clip), optax.adam(lr))
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_fn(m, x, y):
        logits = jax.vmap(m)(x)[:, -1, :]  # read at the last position
        ll = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.take_along_axis(ll, y[:, None], axis=-1))

    @eqx.filter_jit
    def step(m, st, x, y):
        loss, g = eqx.filter_value_and_grad(loss_fn)(m, x, y)
        upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, upd), st, loss

    @eqx.filter_jit
    def correct(m, x, y):
        return jnp.argmax(jax.vmap(m)(x)[:, -1, :], axis=-1) == y

    history, diverged = [], False
    for i in range(n_items):
        xi, yi = X[i : i + 1], Y[i : i + 1]
        s_crit, s_joint = None, None
        for t in range(1, cfg.kv_max_write_steps + 1):
            model, opt_state, loss = step(model, opt_state, xi, yi)
            if not np.isfinite(float(loss)):
                diverged = True
                break
            if t % cfg.kv_check_every == 0 or t == cfg.kv_max_write_steps:
                c = np.asarray(correct(model, X[: i + 1], Y[: i + 1]))
                if s_crit is None and bool(c[i]):
                    s_crit = t
                if s_joint is None and bool(np.all(c)):
                    s_joint = t
                if s_crit is not None and s_joint is not None:
                    break
        if diverged:
            break
        c = np.asarray(correct(model, X[: i + 1], Y[: i + 1]))
        history.append(
            {
                "write_index": int(i),
                "n_stored": int(i + 1),
                "steps_to_criterion": s_crit,
                "steps_to_joint": s_joint,
                "item1_retained": float(c[0]),
                "mean_retention": float(np.mean(c)),
                "per_item_retained": [float(v) for v in c],
            }
        )
    return {
        "primitive": primitive,
        "seed": int(seed),
        "lr": lr,
        "width": int(width),
        "block_params": int(bp),
        "total_params": int(tp),
        "param_match_err": float(err),
        "diverged": diverged,
        "history": history,
        "final_mean_retention": history[-1]["mean_retention"]
        if history
        else float("nan"),
        "final_item1_retained": history[-1]["item1_retained"]
        if history
        else float("nan"),
    }


def item3_cross_primitive(cfg, hcfg, seeds):
    """Equal-LR-grid selection + symmetric monotone rescue, `primitive-harness` §4.

    The tuning budget is equal by construction: every primitive sees the same LR
    grid, the same step budgets and the same seeds, and the rescue pass (re-run
    the non-selected LRs at full length, adopt only if the FULL n-seed mean
    improves — the winner's-curse fix) is applied to every primitive.
    """
    out = []
    for prim in cfg.kv_primitives:
        # --- selection: one seed, every LR, identical budget ---
        sel = []
        for lr in hcfg.lr_grid:
            r = sequential_write_primitive(
                prim, cfg, hcfg, seeds[0], lr, n_items=cfg.kv_select_items
            )
            sel.append((lr, r["final_mean_retention"], r))
        best_lr = max(sel, key=lambda t: (t[1] if np.isfinite(t[1]) else -1))[0]

        def full(lr, prim=prim):
            rs = [sequential_write_primitive(prim, cfg, hcfg, s, lr) for s in seeds]
            m = float(np.nanmean([r["final_mean_retention"] for r in rs]))
            return m, rs

        best_m, best_rs = full(best_lr)
        rescued = None
        for lr in hcfg.lr_grid:
            if lr == best_lr:
                continue
            m, rs = full(lr)
            if m > best_m:  # monotone: adopt ONLY if the full n-seed mean wins
                best_m, best_rs, best_lr, rescued = m, rs, lr, lr
        # --- extended sweep: same selected LR, same seeds, larger K. Reported
        # separately, never merged into the matched-K headline. ---
        ext = [
            sequential_write_primitive(
                prim, cfg, hcfg, s, best_lr, n_items=cfg.kv_extended_items
            )
            for s in seeds
        ]
        out.append(
            {
                "primitive": prim,
                "selected_lr": best_lr,
                "extended_K": cfg.kv_extended_items,
                "extended_mean_retention_at_K": _mean_over_seeds(ext, "mean_retention"),
                "extended_item1_retention_at_K": _mean_over_seeds(
                    ext, "item1_retained"
                ),
                "extended_steps_to_criterion_at_K": _mean_over_seeds(
                    ext, "steps_to_criterion"
                ),
                "extended_joint_censoring_at_K": _censor_rate(ext),
                "rescued_to_lr": rescued,
                "lr_selection": [(lr, m) for lr, m, _ in sel],
                "runs": best_rs,
                "mean_retention_at_K": _mean_over_seeds(best_rs, "mean_retention"),
                "item1_retention_at_K": _mean_over_seeds(best_rs, "item1_retained"),
                "steps_to_criterion_at_K": _mean_over_seeds(
                    best_rs, "steps_to_criterion"
                ),
                "joint_censoring_rate_at_K": _censor_rate(best_rs),
            }
        )
    return {
        "primitives": out,
        "n_items": cfg.kv_n_items,
        "seeds": list(map(int, seeds)),
    }


def _mean_over_seeds(runs, field):
    n = max((len(r["history"]) for r in runs), default=0)
    res = []
    for j in range(n):
        vals = [
            r["history"][j][field]
            for r in runs
            if len(r["history"]) > j and r["history"][j][field] is not None
        ]
        res.append(
            [float(np.mean(vals)), float(np.std(vals))] if vals else [float("nan")] * 2
        )
    return res


def _censor_rate(runs):
    n = max((len(r["history"]) for r in runs), default=0)
    res = []
    for j in range(n):
        vals = [
            r["history"][j]["steps_to_joint"] is None
            for r in runs
            if len(r["history"]) > j
        ]
        res.append(float(np.mean(vals)) if vals else float("nan"))
    return res


# ---------------------------------------------------------------------------
# ITEM 3b / 4 — retrieval cost scaling in K (parametric vs contextual)
# ---------------------------------------------------------------------------


def item4_retrieval_cost(cfg, hcfg, seed=0, dim: int = 3):
    """Read cost as a function of the number of STORED items.

    Three measurements, deliberately kept apart because they exercise different
    capabilities (task Item 4):

    * **parametric** — one forward pass of each harness primitive. The stored
      items live in the weights, so cost cannot depend on K. Measured, not asserted.
    * **contextual** — attention with K key/value pairs *in the context window*.
      This is the capability CLU is NOT exercising anywhere in this module.
    * **CLU landscape rollout** — the two-phase retrieval on a store holding K
      items. O(steps), and for a parametric landscape independent of K.
    """
    from chlu.experiments.exp_primitive_harness import (
        build_model,
        forward_flops,
        match_width,
    )

    ks = list(cfg.cost_K_grid)
    res = {"K_grid": ks, "parametric": {}, "contextual": {}, "clu_rollout": {}}

    # --- parametric: forward pass over one query, model fixed ---
    slot = _kv_slot(cfg.kv_vocab, cfg.kv_key_len)
    x_one = jnp.ones((1, cfg.kv_key_len), dtype=jnp.int32)
    for prim in cfg.kv_primitives:
        mkey = jax.random.PRNGKey(seed)
        width, _, _, _ = match_width(prim, hcfg, slot, mkey)
        model = build_model(prim, hcfg, slot, width, mkey)
        # forward_flops vmaps internally, so it wants a BATCHED input (1, T).
        fl = forward_flops(model, x_one)
        read = jax.jit(lambda xx, m=model: jax.vmap(m)(xx))
        wall = _time_call(lambda r=read, xx=x_one: r(xx))
        res["parametric"][prim] = {
            "flops_per_read": [fl] * len(ks),
            "wall_ms_per_read": [wall] * len(ks),
            "note": "K-independent by construction: the store is in the weights",
        }

    # --- contextual: attention over a K-pair context (LABELLED, not a CLU claim) ---
    fl_c, wall_c, lens = [], [], []
    for K in ks:
        T = max(4, 2 * K + 2)  # K key/value pairs + one query
        cslot = _kv_slot(cfg.kv_vocab, T)
        mkey = jax.random.PRNGKey(seed)
        width, _, _, _ = match_width("attention", hcfg, cslot, mkey)
        model = build_model("attention", hcfg, cslot, width, mkey)
        xc = jnp.ones((1, T), dtype=jnp.int32)
        fl_c.append(forward_flops(model, xc))
        read = jax.jit(lambda xx, m=model: jax.vmap(m)(xx))
        wall_c.append(_time_call(lambda r=read, xx=xc: r(xx)))
        lens.append(T)
    res["contextual"]["attention"] = {
        "seq_len": lens,
        "flops_per_read": fl_c,
        "wall_ms_per_read": wall_c,
        "note": "CONTEXTUAL memory (KV cache at inference) — a DIFFERENT capability "
        "from the parametric writes measured everywhere else in this module",
    }

    # --- CLU landscape rollout on a designed store holding K items ---
    fl_r, wall_r = [], []
    for K in ks:
        V = AtomDictionaryPotential(
            dim=dim,
            capacity=max(ks),
            alpha=cfg.atom_alpha,
            s=cfg.atom_width,
            kappa=cfg.payload_kappa,
        )
        prop = np.asarray(
            disk_proposer(cfg.proposal_radius, dim)(jax.random.PRNGKey(seed), 4 * K)
        )
        pay = np.asarray(designed_payloads(K, seed=cfg.payload_seed))
        placed = []
        d_safe = cfg.d_safe_mult * cfg.atom_width
        for c in prop:
            if len(placed) >= K:
                break
            if (
                min_separation(
                    c[:2], np.stack(placed)[:, :2] if placed else np.zeros((0, 2))
                )
                >= d_safe
            ):
                V = V.with_item(c[:2], float(pay[len(placed)]), amp=cfg.atom_amp)
                placed.append(c)
        model = model_for(V, dim)
        q = jnp.zeros((1, dim)).at[0, :2].set(jnp.asarray(placed[0][:2]))
        p = jnp.zeros((1, dim))
        fn = jax.jit(
            lambda qq, pp, m=model: two_phase(
                m, qq, pp, cfg, cfg.gamma_address, cfg.gamma_read
            )[1]
        )
        fn(q, p)[0].block_until_ready()
        wall_r.append(_time_call(lambda f=fn, qq=q, pp=p: f(qq, pp)))
        fl_r.append(float("nan"))
    res["clu_rollout"] = {
        "wall_ms_per_read": wall_r,
        "n_placed": ks,
        "steps": int(cfg.address_steps + cfg.read_steps),
        "note": "O(steps); the landscape is parametric so the rollout does not "
        "grow with the number of stored items",
    }
    return res


def _time_call(fn, n: int = 20):
    fn()  # warm/compile
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        jax.block_until_ready(r)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


# ---------------------------------------------------------------------------
# Figures (local, per the exp_paid_access / exp_learned_memory precedent)
# ---------------------------------------------------------------------------


def _plot_all(results, save_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths = []
    curve = (results.get("item2_sequential") or {}).get("curve")
    cross = (results.get("item3_cross_primitive") or {}).get("primitives")
    if not curve and not cross:
        return paths
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    a = axes[0]
    for arm, c in (curve or {}).items():
        y = np.array(c["item1_strict_mean_std"])
        if y.size == 0:
            continue
        x = np.arange(len(y))
        a.errorbar(x, y[:, 0], yerr=y[:, 1], marker="o", ms=3, capsize=2, label=arm)
    a.set_xlabel("number of subsequent write attempts")
    a.set_ylabel("retention of item 1 (strict)")
    a.set_title("CLU: sequential-write retention")
    a.set_ylim(-0.05, 1.05)
    a.legend(fontsize=7)

    a = axes[1]
    for arm, c in (curve or {}).items():
        y = np.array(c["mean_retention_mean_std"])
        if y.size == 0:
            continue
        a.errorbar(
            np.arange(len(y)),
            y[:, 0],
            yerr=y[:, 1],
            marker="s",
            ms=3,
            capsize=2,
            label=f"{arm} (n adm {c['n_admitted_mean']:.1f})",
        )
    a.set_xlabel("number of subsequent write attempts")
    a.set_ylabel("mean retention over stored items")
    a.set_title("CLU: mean retention")
    a.set_ylim(-0.05, 1.05)
    a.legend(fontsize=7)

    a = axes[2]
    for p in cross or []:
        y = np.array(p["item1_retention_at_K"])
        if y.size == 0:
            continue
        a.errorbar(
            np.arange(1, len(y) + 1),
            y[:, 0],
            yerr=y[:, 1],
            marker="o",
            ms=3,
            capsize=2,
            label=p["primitive"],
        )
    a.set_xlabel("number of items written (K)")
    a.set_ylabel("retention of item 1")
    a.set_title("cross-primitive, PARAMETRIC sequential writes")
    a.set_ylim(-0.05, 1.05)
    a.legend(fontsize=7)

    fig.tight_layout()
    p = os.path.join(save_dir, "sequential_write_fig1_retention.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_sequential_write(
    config: Optional[CHLUConfig] = None,
    save_dir: str = "results",
    models_dir: Optional[str] = None,
    seed: Optional[int] = None,
    items: Optional[list] = None,
):
    config = config or get_default_config()
    cfg = config.experiment_sequential_write
    hcfg = config.experiment_primitive_harness
    seed = config.project.seed if seed is None else seed
    items = items or ["1", "2", "3", "4"]
    os.makedirs(save_dir, exist_ok=True)
    seeds = list(cfg.seeds)

    results = {
        "seed": seed,
        "seeds": seeds,
        "config": {
            k: getattr(cfg, k)
            for k in (
                "f",
                "dt",
                "gamma_address",
                "gamma_read",
                "address_steps",
                "read_steps",
                "tail_frac",
                "n_subsample",
                "n_query_per_item",
                "n_query_sequential",
                "query_sigma_theta",
                "query_sigma_p",
                "payload_tol",
                "payload_seed",
                "write_steps",
                "write_lr",
                "write_sigma_addr",
                "write_margin",
                "write_barrier",
                "interference_K",
                "interference_write_steps",
                "d_safe_mult",
                "delta_budget",
                "c3_chunk_steps",
                "atom_width",
                "atom_alpha",
                "atom_amp",
                "proposal_radius",
                "n_sequential_items",
                "sequential_write_steps",
                "sequential_rung",
                "kv_n_items",
                "kv_vocab",
                "kv_key_len",
                "kv_max_write_steps",
            )
        },
    }
    results_dir = os.path.join(os.path.dirname(os.path.abspath(save_dir)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp_sequential_write_metrics.json")

    def _dump():
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)

    if "1" in items:
        results["item1_gate_on_w20"] = item1_gate_on_w20(cfg, seeds)
        _dump()
    if "2" in items:
        results["item2_sequential"] = item2_sequential(cfg, seeds)
        _dump()
    if "3" in items:
        results["item3_cross_primitive"] = item3_cross_primitive(cfg, hcfg, seeds)
        _dump()
    if "4" in items:
        results["item4_retrieval_cost"] = item4_retrieval_cost(cfg, hcfg, seed=seed)
        _dump()

    try:
        results["figures"] = _plot_all(results, save_dir)
    except Exception as exc:  # pragma: no cover - figures are not the result
        results["figures"] = []
        results["figure_error"] = repr(exc)
    _dump()
    results["metrics_path"] = out_path
    return results


def apply_quick(config: CHLUConfig) -> None:
    """Quick smoke settings — same code path, smaller sweeps.

    ⚠ NOT shorter than this on the settling budget: the relaxation length is what
    makes the DESIGNED reference arm work at all, and a smoke run whose baseline
    fails prints what looks like a scientific negative (w20 lesson).
    """
    cfg = config.experiment_sequential_write
    cfg.seeds = [0, 1]
    cfg.gate_rungs = ["sites_learned_payload"]
    cfg.gate_proposals = ["ring"]
    cfg.gate_arms = ["ungated", "gated_c3"]
    cfg.write_steps = 60
    cfg.interference_write_steps = 40
    cfg.c3_chunk_steps = 20
    cfg.address_steps = 400
    cfg.read_steps = 200
    cfg.n_query_per_item = 8
    cfg.n_query_sequential = 8
    cfg.n_sequential_items = 4
    cfg.sequential_write_steps = 30
    cfg.sequential_arms = ["designed_gated", "learned_ungated"]
    cfg.kv_primitives = ["mlp", "attention"]
    cfg.kv_n_items = 4
    cfg.kv_select_items = 2
    cfg.kv_max_write_steps = 40
    cfg.cost_K_grid = [2, 4]
    config.experiment_primitive_harness.lr_grid = [1e-3, 3e-3]


def _replace(cfg, **kw):
    c = copy.copy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def main():
    """Documented script entry (see module docstring)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment SEQUENTIAL-WRITE: does an admission gate stop "
        "sequential writes destroying stored items?"
    )
    parser.add_argument("--project", help="Project name to use (default: ./results)")
    parser.add_argument("--seed", type=int, help="Random seed (project-level)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke mode")
    parser.add_argument(
        "--items",
        nargs="+",
        choices=["1", "2", "3", "4"],
        help="Run only these items (default: all)",
    )
    args = parser.parse_args()

    if args.project:
        from chlu.project import ProjectManager

        pm = ProjectManager()
        config = pm.load(args.project)
        paths = pm.get_paths(args.project)
        save_dir, models_dir = str(paths["plots"]), str(paths["models"])
    else:
        config = get_default_config()
        save_dir, models_dir = "results", None
        os.makedirs(save_dir, exist_ok=True)

    if args.quick:
        apply_quick(config)

    res = run_experiment_sequential_write(
        config=config,
        save_dir=save_dir,
        models_dir=models_dir,
        seed=args.seed,
        items=args.items,
    )
    print(
        json.dumps({k: v for k, v in res.items() if k != "item2_sequential"}, indent=2)[
            :3000
        ]
    )


if __name__ == "__main__":
    main()
