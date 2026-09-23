"""w16 hub/maze-wm -- CMAZE-O, Task 2.  AE cache, VAL sweep, pinned full runs on VAL or TEST.

    python scripts/w16_cmaze_o.py prep  <seed> <cache_dir>
    python scripts/w16_cmaze_o.py sweep <seed> <cache_dir> <out_dir>
    python scripts/w16_cmaze_o.py full  <seed> <cache_dir> <out_dir> --leg val|test \
        --K K --atoms A [--pair 1|0] [--tag name] [--deletion 1|0]

Everything below the request layer is the w15 CMAZE-O harness, imported unmodified:
layout bank, pool, hold-outs, AE (same recipe: 6,000 steps, seed key), OCluRead, the
charter-C controls, rivals, lookup floor, BLIND arms, exact judge, deletion study.

What is new:
* ``prep`` trains the frozen AE ONCE per seed and caches it (eqx serialisation) with the
  bank, the pool and the latents, so every config reads the SAME frozen decoder.
* The TEST leg: w15 CMAZE-O shipped no test split.  TEST rows are drawn with the w15
  row recipe (same bank, same hold-out patterns, same stratification) from a DISJOINT rng
  stream (20,000 + seed) and exclude every train request AND every VAL-row request, so a
  TEST request has been seen by nothing.  ``seen`` on TEST is a fresh draw of train items
  (it is a sanity row by construction, as in w15).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import equinox as eqx                                                # noqa: E402
import jax                                                           # noqa: E402

from chlu.exp_w15_cmaze_o_arms import (BLIND_NAMES, CONTROLS_SHIPPED, BlindArms,  # noqa
                                       OCluRead, OLookupRead, ORival,
                                       continuous_inputs, main_effect_design,
                                       make_grids, token_inputs)
from chlu.exp_w15_cmaze_o_encoder import (AE, LATENT_D, ae_param_count,  # noqa: E402
                                          decode_all, encode_all, train_ae)
from chlu.exp_w15_cmaze_o_env import (N_CH, build_layout_bank, decode_state,  # noqa
                                      item_tensor, judge_state, render_state)
from chlu.exp_w15_cmaze_o_splits import (AGENT_SLOT_HOLDOUT_COLOUR,  # noqa: E402
                                         VALUE_HOLDOUT_COLOUR, NWAY_FACTORS,
                                         _draw, _stratified, _with, build_pool,
                                         disjointness_report)

ROWS = ("seen", "unseen_2way", "unseen_nway", "value_holdout", "value_slot_holdout")
KW = dict(n_layouts=12, n_train=8000, n_row=400, n_seen=600, ae_steps=6000,
          rival_steps=3000)
SPAN_CAP = 0.30


def log(*a):
    print(*a, flush=True)


def _items(bank, reqs):
    return np.stack([item_tensor(render_state(bank[r.layout], r)) for r in reqs])


# ------------------------------------------------------------------ TEST rows
def test_rows(bank, pool, seed, n_row=400, n_seen=600, lbf=0.25):
    """The w15 row recipe, a disjoint rng stream, excluding train AND val requests."""
    rng = np.random.default_rng(20_000 + int(seed))
    hold, n_layouts = pool.hold, len(bank)
    excl = {r.words() for r in pool.train}
    for rs in pool.rows.values():
        excl |= {r.words() for r in rs}
    trw = {r.words() for r in pool.train}

    def fresh(r):
        return r.words() not in excl
    rows = {"seen": [pool.train[int(i)]
                     for i in rng.permutation(len(pool.train))[:n_seen]]}
    out2 = []
    per = max(1, n_row // max(1, len(hold.two_way)))
    for (fi, vi, fj, vj) in hold.two_way:
        g = lambda rr, a={fi: vi, fj: vj}: _with(_draw(rr, n_layouts), a)
        acc = lambda r: (not hold.carries_value(r)) and (not hold.matches_n_way(r)) \
            and fresh(r)
        got, _d = _stratified(rng, bank, hold, per, g, acc, lbf, seen=set(excl))
        out2 += got
    rows["unseen_2way"] = [r for (r, _f) in out2]
    outn = []
    pern = max(1, n_row // max(1, len(hold.n_way)))
    for t in hold.n_way:
        a = {f: v for f, v in zip(NWAY_FACTORS, t)}
        g = lambda rr, a=a: _with(_draw(rr, n_layouts), a)
        acc = lambda r: (not hold.carries_value(r)) and (not hold.matches_two_way(r)) \
            and fresh(r)
        got, _d = _stratified(rng, bank, hold, pern, g, acc, lbf, seen=set(excl))
        outn += got
    rows["unseen_nway"] = [r for (r, _f) in outn]

    def gv(rr):
        r = _draw(rr, n_layouts)
        return _with(r, {int(rr.integers(4)) + 1: VALUE_HOLDOUT_COLOUR})
    accv = lambda r: (not hold.matches_two_way(r)) and (not hold.matches_n_way(r)) \
        and (not hold.carries_agent_slot_holdout(r)) and fresh(r)
    rows["value_holdout"] = [r for (r, _f) in _stratified(rng, bank, hold, n_row, gv,
                                                          accv, lbf, seen=set(excl))[0]]
    gs = lambda rr: _with(_draw(rr, n_layouts), {1: AGENT_SLOT_HOLDOUT_COLOUR})
    accs = lambda r: (not hold.matches_two_way(r)) and (not hold.matches_n_way(r)) \
        and (not hold.carries_value(r)) and fresh(r)
    rows["value_slot_holdout"] = [r for (r, _f) in _stratified(rng, bank, hold, n_row,
                                                               gs, accs, lbf,
                                                               seen=set(excl))[0]]
    # disjointness receipts
    rec = {}
    for k, rs in rows.items():
        w = [r.words() for r in rs]
        rec[k] = {"n": len(rs),
                  "in_train": int(sum(x in trw for x in w)),
                  "in_val_rows": int(sum(any(x == y.words() for y in pool.rows[k])
                                         for x in w)) if k != "seen" else None}
    return rows, rec


# ------------------------------------------------------------------- PREP
def prep(seed, cache):
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    bank, tried = build_layout_bank(KW["n_layouts"], 90_000 + 1000 * int(seed))
    pool = build_pool(bank, seed, n_train=KW["n_train"], n_row=KW["n_row"],
                      n_seen=KW["n_seen"], verbose=True)
    trows, trec = test_rows(bank, pool, seed)
    log("[prep %d] test rows %s" % (seed, trec))
    X = _items(bank, pool.train)
    ae, hist = train_ae(X, jax.random.PRNGKey(seed), steps=KW["ae_steps"], batch=64,
                        log_every=1500, verbose=True)
    Z = encode_all(ae, X)
    eqx.tree_serialise_leaves(os.path.join(cache, "ae_seed%d.eqx" % seed), ae)
    np.save(os.path.join(cache, "Z_seed%d.npy" % seed), Z)
    with open(os.path.join(cache, "pool_seed%d.pkl" % seed), "wb") as fh:
        pickle.dump({"bank": bank, "tried": tried, "pool": pool, "test_rows": trows,
                     "test_receipt": trec, "ae_hist": hist}, fh)
    log("[prep %d] done %.0fs" % (seed, time.time() - t0))


def load(seed, cache):
    with open(os.path.join(cache, "pool_seed%d.pkl" % seed), "rb") as fh:
        d = pickle.load(fh)
    ae = eqx.tree_deserialise_leaves(os.path.join(cache, "ae_seed%d.eqx" % seed),
                                     AE(jax.random.PRNGKey(seed), N_CH, LATENT_D))
    Z = np.load(os.path.join(cache, "Z_seed%d.npy" % seed))
    return d, ae, Z


def judge_valid(ae, bank, reqs, Zr):
    LG = decode_all(ae, np.asarray(Zr, np.float32))
    v, lg = [], []
    for i, r in enumerate(reqs):
        j = judge_state(decode_state(LG[i]), render_state(bank[r.layout], r))
        v.append(j["valid"])
        lg.append(j["legal"])
    return {"n": len(reqs), "valid": float(np.mean(v)), "legal": float(np.mean(lg))}


# ------------------------------------------------------------------ SWEEP
SWEEP = []
for K in (1, 2, 4, 8, 16):
    dk = LATENT_D // K
    for na in sorted({1, 2, 4, 6, 8, 12, 16, 24, 32, 38}):
        if na / dk <= SPAN_CAP + 1e-9:
            SWEEP.append((K, na))


def sweep(seed, cache, out):
    os.makedirs(out, exist_ok=True)
    d, ae, Z = load(seed, cache)
    bank, pool = d["bank"], d["pool"]
    grids = make_grids(len(bank))
    res = {"seed": seed, "grid": []}
    rows = ("seen", "unseen_2way", "unseen_nway")
    # the AE ceiling on VAL, once
    res["AE_CEILING"] = {r: judge_valid(ae, bank, pool.rows[r],
                                        encode_all(ae, _items(bank, pool.rows[r])))
                         for r in rows}
    log("[sweep %d] ceiling %s" % (seed, {k: v["valid"] for k, v in res["AE_CEILING"].items()}))
    for pair in (True, False):
        for (K, na) in SWEEP:
            t = time.time()
            clu = OCluRead(Z, pool.train, grids, K=K, n_atoms=na, ridge=1.0,
                           pair_block=pair, seed=seed)
            led = clu.ledger()
            row = {"K": K, "n_atoms": na, "pair_block": pair,
                   "span": na / (LATENT_D // K), "read_floats": led["read_floats"],
                   "atoms": led["store_atom_floats"], "head": led["head_floats"],
                   "CLU": {r: judge_valid(ae, bank, pool.rows[r], clu.read(pool.rows[r]))
                           for r in rows},
                   "controls": {}}
            for c in CONTROLS_SHIPPED:
                if c == "landscape_shuffled":
                    continue            # proved identity for this read (w15 s3.6a)
                co = OCluRead(Z, pool.train, grids, K=K, n_atoms=na, ridge=1.0,
                              pair_block=pair, control=c, seed=seed)
                row["controls"][c] = {r: judge_valid(ae, bank, pool.rows[r],
                                                     co.read(pool.rows[r])) for r in rows}
            res["grid"].append(row)
            log("  pair %d K %2d at %2d span %.3f rf %7d | CLU %s | worst ctl nway %.4f [%.0fs]"
                % (pair, K, na, row["span"], row["read_floats"],
                   " ".join("%.4f" % row["CLU"][r]["valid"] for r in rows),
                   max(v["unseen_nway"]["valid"] for v in row["controls"].values()),
                   time.time() - t))
    with open(os.path.join(out, "sweep_seed%d.json" % seed), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    log("[sweep %d] wrote" % seed)


# ------------------------------------------------------------------- FULL
def full(seed, cache, out, leg, K, na, pair=True, tag="A", deletion=True):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    d, ae, Z = load(seed, cache)
    bank, pool = d["bank"], d["pool"]
    rows_src = pool.rows if leg == "val" else d["test_rows"]
    grids = make_grids(len(bank))
    cfg = {"K": int(K), "n_atoms": int(na), "ridge": 1.0, "pair_block": bool(pair),
           "latent_d": LATENT_D, "partition": "ward_balanced"}
    res = {"seed": seed, "leg": leg, "tag": tag, "config": cfg,
           "scored_run": leg == "test", "tolerance_used": None,
           "test_receipt": d["test_receipt"] if leg == "test" else None,
           "disjointness_val": disjointness_report(pool)["pass"]}
    clu = OCluRead(Z, pool.train, grids, K=K, n_atoms=na, ridge=1.0, pair_block=pair,
                   seed=seed)
    led = clu.ledger()
    res["clu_ledger"] = led
    res["rank_over_width_printed"] = led.get("span_fraction")
    res["request_only"] = clu.assert_request_only(pool.train[:64])
    arms = {"CLU": clu}
    for c in CONTROLS_SHIPPED:
        arms["CTRL:" + c] = OCluRead(Z, pool.train, grids, K=K, n_atoms=na, ridge=1.0,
                                     pair_block=pair, control=c, seed=seed)
    lookup = OLookupRead(Z, pool.train, budget_floats=led["read_floats"], seed=seed)
    res["lookup_ledger"] = lookup.ledger()
    Phi_main, mspec = main_effect_design(pool.train, grids)
    Tok = token_inputs(Phi_main, mspec)
    Cont = continuous_inputs(pool.train, grids)
    target = int(led["read_floats"])
    rivals = {}
    res["rivals"] = {}
    for kind in ORival.KINDS:
        key = jax.random.PRNGKey(1000 + seed)
        if kind == "mlp":
            rv, Xin = ORival(kind, key, Phi_main.shape[1], Z.shape[1], target), Phi_main
        elif kind == "cont_mlp":
            rv, Xin = ORival(kind, key, Cont.shape[1], Z.shape[1], target), Cont
        else:
            rv = ORival(kind, key, None, Z.shape[1], target, seq_in=Tok.shape[2],
                        n_tokens=Tok.shape[1])
            Xin = Tok
        t = time.time()
        rv.fit(Xin, Z, steps=KW["rival_steps"], seed=seed, verbose=False)
        rivals[kind] = rv
        res["rivals"][kind] = dict(rv.ledger(), fit_seconds=time.time() - t)
        log("[%s %d %s] rival %s params %d matched=%s" % (tag, seed, leg, kind,
                                                        rv.n_params, rv.matched))
    blind = BlindArms(bank, pool.train)
    rng_rand = np.random.default_rng(777 + seed)
    R = {}
    for row in ROWS:
        reqs = rows_src[row]
        if not reqs:
            continue
        reads = {"AE-CEILING": encode_all(ae, _items(bank, reqs)),
                 "LOOKUP (printed floor)": lookup.read(reqs)}
        for a, obj in arms.items():
            reads[a] = obj.read(reqs)
        Pm, _ = main_effect_design(reqs, grids)
        Tk = token_inputs(Pm, mspec)
        Cn = continuous_inputs(reqs, grids)
        for kind, rv in rivals.items():
            reads["RIVAL:" + kind] = rv.read(Pm if kind == "mlp" else
                                             (Cn if kind == "cont_mlp" else Tk))
        R[row] = {a: judge_valid(ae, bank, reqs, Zr) for a, Zr in reads.items()}
        for b in BLIND_NAMES:
            st = [blind.state(b, r, rng_rand) for r in reqs]
            ok = [judge_state(s, render_state(bank[r.layout], r))["valid"]
                  for s, r in zip(st, reqs)]
            R[row][b] = {"n": len(reqs), "valid": float(np.mean(ok))}
        best = max(v["valid"] for k, v in R[row].items() if k.startswith("RIVAL:"))
        log("[%s %d %s] %-18s n=%d CLU %.4f best-rival %.4f worst-ctl %.4f ceil %.4f"
            % (tag, seed, leg, row, len(reqs), R[row]["CLU"]["valid"], best,
               max(v["valid"] for k, v in R[row].items()
                   if k.startswith("CTRL:") and k != "CTRL:landscape_shuffled"),
               R[row]["AE-CEILING"]["valid"]))
    res["ROWS"] = R
    if deletion:
        from chlu.exp_w15_cmaze_o_deletion import deletion_study
        from types import SimpleNamespace
        p2 = SimpleNamespace(train=pool.train, rows=rows_src, hold=pool.hold)
        res["DELETION"] = deletion_study(bank, p2, ae, Z, seed, cfg,
                                         rival_steps=KW["rival_steps"])
    res["seconds"] = time.time() - t0
    p = os.path.join(out, "%s_%s_seed%d.json" % (tag, leg, seed))
    with open(p, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    log("[%s %d %s] wrote %s [%.0fs]" % (tag, seed, leg, p, res["seconds"]))


if __name__ == "__main__":
    mode, seed = sys.argv[1], int(sys.argv[2])
    if mode == "prep":
        prep(seed, sys.argv[3])
    elif mode == "sweep":
        sweep(seed, sys.argv[3], sys.argv[4])
    else:
        kw = {sys.argv[i].lstrip("-"): sys.argv[i + 1] for i in range(5, len(sys.argv), 2)}
        full(seed, sys.argv[3], sys.argv[4], kw["leg"], int(kw["K"]), int(kw["atoms"]),
             kw.get("pair", "1") == "1", kw.get("tag", "A"), kw.get("deletion", "1") == "1")
