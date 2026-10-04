"""One CMAZE-O seed end to end: bank -> pool -> frozen AE -> every arm -> exact judge.

This module produces, for one seed, the per-row record of EVERY judged state for EVERY
arm, plus a compact replayable record of every state it judged, so that
``exp_w15_cmaze_o_verify`` can re-derive every verdict from a second, independently
driven code path and the disagreement count can be printed.

Nothing here selects anything on a held-out row.  The rivals early-stop on a held-IN
10 % of the TRAIN requests; the CLU's ridge, ``K`` and ``n_atoms`` are pinned in
:data:`CONFIG_A` and are never swept.
"""

from __future__ import annotations

import json
import os
import time

import jax
import numpy as np

from chlu.exp_w15_cmaze_o_arms import (BLIND_NAMES, CONTROLS, BlindArms, OCluRead,
                                       OLookupRead, ORival, continuous_inputs,
                                       main_effect_design, make_grids,
                                       nearest_item_audit, token_inputs)
from chlu.exp_w15_cmaze_o_encoder import (LATENT_D, ae_param_count, decode_all,
                                          encode_all, train_ae)
from chlu.exp_w15_cmaze_o_env import (admissible, build_layout_bank, decode_state,
                                      item_tensor, judge_state, render_state,
                                      state_record)
from chlu.exp_w15_cmaze_o_splits import build_pool, disjointness_report

#: THE SINGLE DESIGN, pinned.  Never swept, never selected on a held-out row.
CONFIG_A = {"K": 8, "n_atoms": 12, "ridge": 1.0, "pair_block": True,
            "latent_d": LATENT_D, "partition": "ward_balanced"}

ROWS = ("seen", "unseen_2way", "unseen_nway", "value_holdout", "value_slot_holdout")
LATENT_ARMS_NOTE = ("every latent arm is rendered by the SAME frozen decoder and read "
                    "back by the SAME parameterless state decoder")


def _items(bank, reqs):
    return np.stack([item_tensor(render_state(bank[r.layout], r)) for r in reqs])


def _judge_rows(bank, reqs, states):
    """Judge a list of predicted states against their references.  Exact, no tolerance."""
    recs, packed, verdicts = [], [], []
    for r, pred in zip(reqs, states):
        ref = render_state(bank[r.layout], r)
        j = judge_state(pred, ref)
        recs.append(j)
        packed.append(state_record(pred))
        verdicts.append([j["valid"], j["exact_objects"], j["exact_layout"], j["legal"]])
    return recs, packed, verdicts


def _agg(recs):
    if not recs:
        return {"n": 0}
    out = {"n": len(recs)}
    for k in ("valid", "exact_objects", "exact_layout", "exact_all", "legal",
              "door_load_bearing"):
        out[k] = float(np.mean([r[k] for r in recs]))
    for k in recs[0]["per_fact"]:
        out["fact_" + k] = float(np.mean([r["per_fact"][k] for r in recs]))
    for k in recs[0]["clauses"]:
        out["clause_" + k] = float(np.mean([r["clauses"][k] for r in recs]))
    out["floor_cells_wrong_mean"] = float(np.mean([r["floor_cells_wrong"]
                                                   for r in recs]))
    out["tint_cells_wrong_mean"] = float(np.mean([r["tint_cells_wrong"] for r in recs]))
    out["tolerance_used"] = None
    return out


def run_seed(seed: int, out_dir: str, n_layouts: int = 12, n_train: int = 8000,
             n_row: int = 400, n_seen: int = 600, ae_steps: int = 2000,
             rival_steps: int = 2500, deletion: bool = True,
             verbose: bool = True) -> dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    log = (lambda *a: print("[seed %d]" % seed, *a, flush=True)) if verbose \
        else (lambda *a: None)

    # ---------------------------------------------------------------- instrument
    bank, tried = build_layout_bank(n_layouts, 90_000 + 1000 * int(seed))
    log("layout bank: %d layouts from %d generator seeds" % (len(bank), tried))
    pool = build_pool(bank, seed, n_train=n_train, n_row=n_row, n_seen=n_seen,
                      verbose=verbose)
    disj = disjointness_report(pool)
    log("disjointness pass=%s %s" % (disj["pass"], disj["failures"][:2]))

    # no impossible request is ever shown to a non-refuser arm -- re-checked here
    n_checked, n_impossible = 0, 0
    for name in ROWS:
        for r in pool.rows[name]:
            n_checked += 1
            if not admissible(bank[r.layout], r)[0]:
                n_impossible += 1
    for r in pool.train:
        n_checked += 1
        if not admissible(bank[r.layout], r)[0]:
            n_impossible += 1
    log("impossible-request check: %d requests, %d impossible" % (n_checked,
                                                                  n_impossible))

    # ---------------------------------------------------------------- frozen AE
    X = _items(bank, pool.train)
    t = time.time()
    ae, hist = train_ae(X, jax.random.PRNGKey(seed), steps=ae_steps, batch=64,
                        log_every=max(1, ae_steps // 4), verbose=verbose)
    ae_secs = time.time() - t
    Z = encode_all(ae, X)
    log("AE trained in %.1fs; latents %s" % (ae_secs, Z.shape))

    # ---------------------------------------------------------------- the arms
    grids = make_grids(len(bank))
    clu = OCluRead(Z, pool.train, grids, K=CONFIG_A["K"], n_atoms=CONFIG_A["n_atoms"],
                   ridge=CONFIG_A["ridge"], pair_block=CONFIG_A["pair_block"],
                   control=None, seed=seed)
    led = clu.ledger()
    log("CLU read floats %d (atoms %d + head %d), span %.3f"
        % (led["read_floats"], led["store_atom_floats"], led["head_floats"],
           led["span_fraction"]))
    request_only = clu.assert_request_only(pool.train)

    arms = {"CLU": clu}
    for c in CONTROLS:
        arms["CTRL:" + c] = OCluRead(Z, pool.train, grids, K=CONFIG_A["K"],
                                     n_atoms=CONFIG_A["n_atoms"],
                                     ridge=CONFIG_A["ridge"],
                                     pair_block=CONFIG_A["pair_block"],
                                     control=c, seed=seed)
    lookup = OLookupRead(Z, pool.train, budget_floats=led["read_floats"], seed=seed)
    log("lookup floor holds %d items = %d floats (CLU read %d)"
        % (lookup.n_stored, lookup.floats, led["read_floats"]))

    # rivals, matched to the CLU's deployed read floats
    Phi_main, mspec = main_effect_design(pool.train, grids)
    Tok = token_inputs(Phi_main, mspec)
    Cont = continuous_inputs(pool.train, grids)
    target = int(led["read_floats"])
    rivals, rival_secs = {}, {}
    for kind in ORival.KINDS:
        key = jax.random.PRNGKey(1000 + seed)
        if kind == "mlp":
            Xin = Phi_main
            rv = ORival(kind, key, Phi_main.shape[1], Z.shape[1], target)
        elif kind == "cont_mlp":
            Xin = Cont
            rv = ORival(kind, key, Cont.shape[1], Z.shape[1], target)
        else:
            Xin = Tok
            rv = ORival(kind, key, None, Z.shape[1], target, seq_in=Tok.shape[2],
                        n_tokens=Tok.shape[1])
        t = time.time()
        rv.fit(Xin, Z, steps=rival_steps, seed=seed, verbose=False)
        rival_secs[kind] = time.time() - t
        rivals[kind] = (rv, Xin)
        log("rival %-12s width %4d params %7d (target %7d, matched=%s) %.0fs"
            % (kind, rv.width, rv.n_params, target, rv.matched, rival_secs[kind]))

    blind = BlindArms(bank, pool.train)

    # ---------------------------------------------------------------- evaluation
    results, packs, index, verds = {}, [], [], []
    rng_rand = np.random.default_rng(777 + seed)
    for row in ROWS:
        reqs = pool.rows[row]
        if not reqs:
            continue
        Xrow = _items(bank, reqs)
        Zt = encode_all(ae, Xrow)                       # the AE CEILING's read
        row_reads = {"AE-CEILING": Zt, "CLU": clu.read(reqs),
                     "LOOKUP": lookup.read(reqs)}
        for c in CONTROLS:
            row_reads["CTRL:" + c] = arms["CTRL:" + c].read(reqs)
        Pm, _ = main_effect_design(reqs, grids)
        Tk = token_inputs(Pm, mspec)
        Cn = continuous_inputs(reqs, grids)
        for kind, (rv, _x) in rivals.items():
            xin = Pm if kind == "mlp" else (Cn if kind == "cont_mlp" else Tk)
            row_reads["RIVAL:" + kind] = rv.read(xin)
        for arm, Zr in row_reads.items():
            LG = decode_all(ae, np.asarray(Zr, np.float32))
            states = [decode_state(LG[i]) for i in range(len(reqs))]
            recs, packed, vd = _judge_rows(bank, reqs, states)
            results.setdefault(arm, {})[row] = _agg(recs)
            verds += vd
            for i, pk in enumerate(packed):
                packs.append(pk)
                index.append((arm, row, i) + tuple(reqs[i].words()))
        for bname in BLIND_NAMES:
            states = [blind.state(bname, r, rng_rand) for r in reqs]
            recs, packed, vd = _judge_rows(bank, reqs, states)
            results.setdefault(bname, {})[row] = _agg(recs)
            verds += vd
            for i, pk in enumerate(packed):
                packs.append(pk)
                index.append((bname, row, i) + tuple(reqs[i].words()))
        log("row %-20s judged %d requests x %d arms"
            % (row, len(reqs), len(row_reads) + len(BLIND_NAMES)))

    # ---------------------------------------------------------------- deletion
    del_out = None
    if deletion:
        from chlu.exp_w15_cmaze_o_deletion import deletion_study
        del_out = deletion_study(bank, pool, ae, Z, seed, CONFIG_A,
                                 rival_steps=rival_steps, verbose=verbose)

    # charter C.4: is the CLU's read a lookup table in disguise?
    audit = nearest_item_audit(clu.read(pool.rows["unseen_nway"][:128]), Z[:1024])

    # ---------------------------------------------------------------- artefacts
    np.savez_compressed(
        os.path.join(out_dir, "states_seed%d.npz" % seed),
        floor_bits=np.stack([p["floor_bits"] for p in packs]),
        tint_bits=np.stack([p["tint_bits"] for p in packs]),
        ints=np.stack([p["ints"] for p in packs]),
        verdict=np.asarray(verds, np.uint8),
        words=np.array([tuple(x[3:]) for x in index], np.int64),
        arm=np.array([x[0] for x in index]),
        row=np.array([x[1] for x in index]),
    )
    np.savez_compressed(os.path.join(out_dir, "bank_seed%d.npz" % seed),
                        grids=np.stack([L.grid for L in bank]),
                        tints=np.stack([L.tint_mask for L in bank]),
                        slots=np.stack([L.slot_cell for L in bank]))

    out = {
        "seed": int(seed),
        "config_A": dict(CONFIG_A),
        "layout_bank": {"n_layouts": len(bank), "generator_seeds_tried": int(tried),
                        "seed0": 90_000 + 1000 * int(seed)},
        "pool_stats": pool.stats,
        "disjointness": disj,
        "impossible_request_check": {"requests_checked": int(n_checked),
                                     "impossible_shown": int(n_impossible)},
        "ae": {"steps": ae_steps, "seconds": ae_secs, "loss_history": hist,
               **ae_param_count(ae)},
        "clu_ledger": led,
        "lookup_ledger": lookup.ledger(),
        "rivals": {k: {**v[0].ledger(), "fit_seconds": rival_secs[k],
                       "heldin_val_mse": v[0].val_mse} for k, v in rivals.items()},
        "request_only_assertion": request_only,
        "clu_out_of_range_designs": int(getattr(clu, "last_out_of_range", 0)),
        "nearest_item_audit_unseen_nway": audit,
        "results": results,
        "deletion": del_out,
        "wall_seconds": time.time() - t0,
        "tolerance_used": None,
    }
    with open(os.path.join(out_dir, "seed%d.json" % seed), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    log("done in %.0fs" % out["wall_seconds"])
    return out
