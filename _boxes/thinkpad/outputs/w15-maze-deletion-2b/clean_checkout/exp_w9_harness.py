"""w9 maze harness -- build and self-test the shared instrument.

One run of this file = one seed of the whole instrument:

  judge audit -> stepper audit -> shared encoder (reconstruction only) -> shared
  observation -> matched-parameter MLP / LSTM / transformer on that observation with
  the shared action head -> table floor -> reported columns (fragment lookup,
  composition-blind) -> supervised refuser.

Everything is scored by `w9_harness.judge_maze.JUDGE`, which is env-scout's exact
integer judge, unwrapped.

Usage:
    python exp_w9_harness.py --seed 0 --scale full --out /c/chlu/outputs/w9-harness
    python exp_w9_harness.py --seed 0 --scale smoke
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import numpy as np

from w8_envscout.maze_colour import generate, solve
from w9_harness import arms, columns, data, dynamics, encoder, features, render, tabular
from w9_harness.judge_maze import ACTIONS, JUDGE, audit
from w9_harness.rollout import HEADER, NOTE, print_row, rollout, score

SCALES = {
    "smoke": dict(n_train=60, n_eval=40, n_imp_train=30, n_imp_eval=30,
                  enc_steps=200, bc_steps=150, target_params=20000, n_audit=8,
                  train_T=64),
    "mid":   dict(n_train=300, n_eval=120, n_imp_train=150, n_imp_eval=120,
                  enc_steps=800, bc_steps=800, target_params=60000, n_audit=20,
                  train_T=96),
    "std":   dict(n_train=500, n_eval=200, n_imp_train=250, n_imp_eval=200,
                  enc_steps=1200, bc_steps=2000, target_params=60000, n_audit=25,
                  train_T=104),
    "full":  dict(n_train=700, n_eval=200, n_imp_train=350, n_imp_eval=200,
                  enc_steps=1500, bc_steps=2500, target_params=60000, n_audit=30,
                  train_T=112),
}

FEAT_DIM = 8

# Two DECLARED tiers of the same environment, same generator, same exact judge.
# The defaults are env-scout's adopted configuration, unchanged.  The easy tier is
# the same generator with a smaller grid and two colours: the composition (rule A x
# rule B x colour) is intact, the routes are ~18 steps instead of ~43, and that is
# the difference between an instrument with dynamic range for a learned arm and one
# that reads 0.0000 for every arm because 43 consecutive actions must all be right.
TIERS = {
    "cmaze1": {},
    "cmaze1-easy": dict(h=4, w=4, n_colours=2),
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(seed, scale, outdir, arm_kinds=("mlp", "lstm", "transformer"),
        do_refuser=True, tag="", tier="cmaze1"):
    cfg = SCALES[scale]
    gen = TIERS[tier]
    t_all = time.time()
    rep = {"seed": seed, "scale": scale, "config": cfg, "utc_start": utc(),
           "arms_run": list(arm_kinds), "refuser_run": do_refuser,
           "tier": tier, "generator_kwargs": gen}
    base = 1_000_000 + 5_000_000 * seed

    # ---------------------------------------------------------------- 1. data
    print("\n" + "=" * 78)
    print("w9 HARNESS  seed=%d  scale=%s  tier=%s %s"
          % (seed, scale, tier, gen or "(env-scout adopted defaults)"))
    print("=" * 78)
    t0 = time.time()
    tr_insts, tr_seeds = data.make_instances(cfg["n_train"], base, **gen)
    ev_insts, ev_seeds = data.make_instances(cfg["n_eval"], base + 2_000_000, **gen)
    tr_routes = data.oracle_routes(tr_insts)
    ev_routes = data.oracle_routes(ev_insts)
    print("instances: %d train / %d eval  (CMAZE-1 adopted defaults)  [%.1fs]"
          % (len(tr_insts), len(ev_insts), time.time() - t0))
    opt = np.array([i.opt_len for i in ev_insts])
    print("  eval opt_len mean %.1f max %d ; budget mean %.1f max %d"
          % (opt.mean(), opt.max(),
             np.mean([i.budget for i in ev_insts]), max(i.budget for i in ev_insts)))
    rep["instances"] = {"n_train": len(tr_insts), "n_eval": len(ev_insts),
                        "eval_opt_len_mean": float(opt.mean()),
                        "eval_opt_len_max": int(opt.max())}

    # ------------------------------------------------------------- 2. audits
    print()
    a = audit(ev_insts[: cfg["n_audit"]])
    s = dynamics.audit_stepper(ev_insts[: cfg["n_audit"]])
    rep["judge_audit"] = {k: v for k, v in a.items() if k != "source_findings"} | \
                         {"source_findings": a["source_findings"]}
    rep["stepper_audit"] = s
    if not a["exact"]:
        rep["STOP"] = "judge is not exact -- everything downstream inherits this"
        return rep

    # ------------------------------------------------------- 3. shared encoder
    print()
    tr_imgs = render.render_batch(tr_insts)
    ev_imgs = render.render_batch(ev_insts)
    enc, enc_rep = encoder.train(tr_imgs, feat_dim=FEAT_DIM, steps=cfg["enc_steps"],
                                 seed=seed)
    rep["encoder"] = enc_rep
    rep["encoder_independence"] = encoder.independence_report(enc)
    rep["encoder_probe"] = encoder.probe_report(
        enc, ev_imgs, encoder.plane_names(int(tr_insts[0].n_colours)))
    tr_feats = encoder.features(enc, tr_imgs)
    ev_feats = encoder.features(enc, ev_imgs)

    k = int(tr_insts[0].n_colours)
    obs_dim = features.obs_dim(FEAT_DIM, k)
    Tmax = max(max(len(r) for r in tr_routes), max(int(i.budget) for i in ev_insts)) + 1
    print("  shared observation: %d dims (%d local patch + %d global pool + %d agent state), "
          "rollout horizon %d"
          % (obs_dim, FEAT_DIM * 25, FEAT_DIM * 25, features.dyn_dim(k), Tmax))
    rep["observation"] = {"obs_dim": obs_dim, "encoder_local": FEAT_DIM * 25,
                          "encoder_global": FEAT_DIM * 25,
                          "agent_state_part": features.dyn_dim(k), "Tmax": Tmax}

    t0 = time.time()
    trT = min(cfg["train_T"], Tmax)
    X, Y, M, sinfo = data.build_sequences(tr_insts, tr_feats, tr_routes, obs_dim, trT)
    vX, vY, vM, vinfo = data.build_sequences(ev_insts, ev_feats, ev_routes, obs_dim, trT)
    print("  supervision: %d of %d train steps kept (horizon %d; %d of %d routes run "
          "past it), %d eval steps  [%.1fs]"
          % (sinfo["steps_supervised"], sinfo["steps_available"], trT,
             sinfo["n_truncated_routes"], len(tr_routes), vinfo["steps_supervised"],
             time.time() - t0))
    rep["supervision"] = {"train": sinfo, "eval": vinfo}

    # -------------------------------------------------- 4. matched parameters
    print()
    widths = arms.match_widths(obs_dim, arms.N_ACTIONS, cfg["target_params"],
                               kinds=arm_kinds, tmax=Tmax + 4)
    rep["matched_widths"] = {k2: {"width": w, "params": p, "kw": kw}
                             for k2, (w, p, kw) in widths.items()}

    # ------------------------------------------------------------- 5. baselines
    print("\nTRAINING (behaviour cloning on exact oracle routes)")
    policies, train_reps = {}, {}
    for kind in arm_kinds:
        w, _, kw = widths[kind]
        p = arms.build(kind, obs_dim, w, arms.N_ACTIONS, seed=seed, **kw)
        p, tr = arms.train_policy(p, X, Y, M, steps=cfg["bc_steps"], seed=seed,
                                  label=kind, val=(vX, vY, vM))
        policies[kind] = p
        train_reps[kind] = tr
    rep["training"] = train_reps
    rep["head_identity"] = arms.head_identity_report(policies)
    print("  shared action head: identical shape across arms = %s (%s)"
          % (rep["head_identity"]["identical_shape_across_arms"],
             rep["head_identity"]["same_init_key"]))

    print("\nEVALUATION -- exact judge, %d instances" % len(ev_insts))
    print(HEADER)
    results = {}
    for kind in arm_kinds:
        routes, refused, _ = rollout(policies[kind], ev_insts, ev_feats, obs_dim,
                                     arms.N_ACTIONS, tmax=Tmax)
        r = score(ev_insts, routes, label="%s (matched, %d params)"
                  % (kind.upper(), train_reps[kind]["params"]))
        r["teacher_forced_action_acc"] = train_reps[kind]["val_teacher_forced_action_acc"]
        results[kind] = r
        print_row(r)
        print("       teacher-forced per-step action accuracy %.4f ; validity by "
              "optimal length %s"
              % (r["teacher_forced_action_acc"],
                 {b: round(v["validity"], 4)
                  for b, v in r["validity_by_optimal_length"].items()}))

    # ------------------------------------------------------------ 6. table floor
    rng = np.random.default_rng(100 + seed)
    tab = tabular.build(tr_insts, tr_routes, tabular.raw_key)
    routes, tinfo = tabular.table_rollout(ev_insts, tab, tabular.raw_key, rng)
    r = score(ev_insts, routes, label="TABLE FLOOR (raw state, %d keys)" % tinfo["size"])
    r["coverage"] = tinfo["coverage"]
    results["table"] = r
    print_row(r)

    fmap = {id(i): f for i, f in list(zip(tr_insts, tr_feats)) + list(zip(ev_insts, ev_feats))}
    ekey = tabular.make_enc_key(fmap)
    tab2 = tabular.build(tr_insts, tr_routes, ekey)
    routes, t2 = tabular.table_rollout(ev_insts, tab2, ekey, rng)
    r = score(ev_insts, routes, label="TABLE-ENC (encoder bits, %d keys)" % t2["size"])
    r["coverage"] = t2["coverage"]
    results["table_enc"] = r
    print_row(r)

    # -------------------------------------------------------- 7. reported columns
    print("\nREPORTED COLUMNS (never abort conditions)")
    print(HEADER)
    cols = {}
    for name, spec in columns.BLIND.items():
        routes = [columns.plan(i, spec) or "" for i in ev_insts]
        r = score(ev_insts, routes, label=name)
        cols[name] = r
        print_row(r)
    for radius in (1, 2):
        routes, info = tabular.fragment_column(tr_insts, tr_routes, ev_insts, radius, rng)
        r = score(ev_insts, routes, label="fragment-lookup r=%d" % radius)
        r.update(info)
        cols["fragment-lookup r=%d" % radius] = r
        print_row(r)
        print("       %d signatures, train coverage %.4f"
              % (info["signatures"], info["train_coverage"]))
    print(NOTE)
    rep["columns"] = cols
    hname, hres = columns.headline(cols)
    rep["column_headline"] = {
        "composition_blind_planner": hname,
        "composition_blind": hres["score"],
        "composition_blind_validity": hres["validity"],
        "fragment_lookup": max(cols["fragment-lookup r=1"]["score"],
                               cols["fragment-lookup r=2"]["score"]),
        "fragment_lookup_validity": max(cols["fragment-lookup r=1"]["validity"],
                                        cols["fragment-lookup r=2"]["validity"])}
    print("  headline composition-blind = %s (score %.4f); fragment-lookup = %.4f"
          % (hname, hres["score"], rep["column_headline"]["fragment_lookup"]))

    # ------------------------------------------------------------ 8. the refuser
    if not do_refuser:
        rep["results"] = results
        rep["seconds"] = time.time() - t_all
        rep["utc_end"] = utc()
        print("\nseed %d (partial: refuser NOT RUN) done in %.1fs"
              % (seed, rep["seconds"]))
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir,
                                   "harness_seed%d%s.json" % (seed, tag)), "w") as f:
                json.dump(rep, f, indent=1, default=str)
        return rep

    print("\nSUPERVISED REFUSER (the ONLY comparator for DISPLAY rows)")
    t0 = time.time()
    imp_tr, fam_tr = data.make_impossible(cfg["n_imp_train"], base + 4_000_000,
                                          rng_seed=seed, **gen)
    imp_ev, fam_ev = data.make_impossible(cfg["n_imp_eval"], base + 4_500_000,
                                          rng_seed=100 + seed, **gen)
    bad = data.certify_impossible(imp_tr + imp_ev, fam_tr + fam_ev)
    unseen = data.assert_unseen(list(tr_seeds) + list(ev_seeds), imp_tr + imp_ev)
    print("  impossible set: %d train / %d eval, certification failures %d, "
          "families %s  [%.1fs]"
          % (len(imp_tr), len(imp_ev), len(bad),
             {f: (fam_tr + fam_ev).count(f) for f in set(fam_tr + fam_ev)},
             time.time() - t0))
    print("  no impossible example in any non-refuser arm's data: %s" % unseen["clean"])
    rep["impossible"] = {"n_train": len(imp_tr), "n_eval": len(imp_ev),
                         "certification_failures": bad[:5],
                         "n_certification_failures": len(bad),
                         "families": {f: (fam_tr + fam_ev).count(f)
                                      for f in set(fam_tr + fam_ev)},
                         "unseen_by_other_arms": unseen}

    imp_tr_imgs = render.render_batch(imp_tr)
    imp_ev_imgs = render.render_batch(imp_ev)
    imp_tr_feats = encoder.features(enc, imp_tr_imgs)
    imp_ev_feats = encoder.features(enc, imp_ev_imgs)
    rX, rY, rM = data.refuse_sequences(imp_tr, imp_tr_feats, obs_dim, trT,
                                       arms.N_ACTIONS)
    # equal total loss weight for "act" and "refuse", stated rather than left to
    # the accident of how many steps each population happens to contribute
    w_refuse = float(M.sum()) / max(float(rM.sum()), 1.0)
    rM = rM * w_refuse
    print("  class balance: %d action-steps vs %d refuse-steps -> refuse weight "
          "x%.1f (equal total weight)"
          % (int(M.sum()), int(rM.sum() / w_refuse), w_refuse))
    rep["refuser_class_balance"] = {"action_steps": int(M.sum()),
                                    "refuse_steps": int(rM.sum() / w_refuse),
                                    "refuse_weight": w_refuse}
    RX = np.concatenate([X, rX])
    RY = np.concatenate([Y, rY])
    RM = np.concatenate([M, rM])
    wR = arms.match_widths(obs_dim, arms.N_OUT_REFUSER, cfg["target_params"],
                           kinds=("transformer",), tmax=Tmax + 4,
                           verbose=False)["transformer"]
    print("  refuser: transformer width %d, %d params (matched to the same target)"
          % (wR[0], wR[1]))
    ref_p = arms.build("transformer", obs_dim, wR[0], arms.N_OUT_REFUSER, seed=seed,
                       **wR[2])
    ref_p, ref_tr = arms.train_policy(ref_p, RX, RY, RM, steps=cfg["bc_steps"],
                                      seed=seed, label="refuser")
    rep["refuser_training"] = ref_tr

    refuser_rows = {}
    for mode, at_start in (("decision at t=0", True), ("any step", False)):
        ro, refd, _ = rollout(ref_p, ev_insts, ev_feats, obs_dim, arms.N_OUT_REFUSER,
                              refuse_index=arms.N_ACTIONS, tmax=Tmax,
                              refuse_only_at_start=at_start)
        rp = score(ev_insts, ro, refused=refd,
                   label="REFUSER on POSSIBLE [%s]" % mode)
        ro2, refd2, _ = rollout(ref_p, imp_ev, imp_ev_feats, obs_dim,
                                arms.N_OUT_REFUSER, refuse_index=arms.N_ACTIONS,
                                tmax=Tmax, refuse_only_at_start=at_start)
        ri = score(imp_ev, ro2, refused=refd2,
                   label="REFUSER on IMPOSSIBLE [%s]" % mode)
        refuser_rows[mode] = (rp, ri)
        results["refuser_possible_%s" % mode.replace(" ", "_")] = rp
        results["refuser_impossible_%s" % mode.replace(" ", "_")] = ri
    r_poss, r_imp = refuser_rows["decision at t=0"]
    tpr, fpr = r_imp["refusal_rate"], r_poss["refusal_rate"]
    rep["refuser_headline"] = {
        "refusal_rate_on_impossible": tpr,
        "refusal_rate_on_possible": fpr,
        "separation": tpr - fpr,
        "reading": "decision at t=0; the any-step reading is also recorded"}
    print("  %-38s %8s %8s   %s" % ("", "validity", "refusal", "note"))
    print("  %-38s %8.4f %8.4f   false refusals on solvable mazes"
          % (r_poss["label"], r_poss["validity"], r_poss["refusal_rate"]))
    print("  %-38s %8.4f %8.4f   correct refusals (the DISPLAY comparator)"
          % (r_imp["label"], r_imp["validity"], r_imp["refusal_rate"]))
    for mode, (rp, ri) in refuser_rows.items():
        if mode != "decision at t=0":
            print("  %-38s %8.4f %8.4f   alternative reading"
                  % (rp["label"], rp["validity"], rp["refusal_rate"]))
            print("  %-38s %8.4f %8.4f   alternative reading"
                  % (ri["label"], ri["validity"], ri["refusal_rate"]))
    print("  refusal separation (impossible - possible) = %+.4f"
          % rep["refuser_headline"]["separation"])
    results["refuser_possible"] = r_poss
    results["refuser_impossible"] = r_imp

    for kind in arm_kinds:
        ro3, _, _ = rollout(policies[kind], imp_ev, imp_ev_feats, obs_dim,
                            arms.N_ACTIONS, tmax=Tmax)
        rr = score(imp_ev, ro3, label="%s on IMPOSSIBLE (cannot refuse)" % kind.upper())
        results["%s_impossible" % kind] = rr
        print("  %-38s %8.4f %8s   no refuse token by construction"
              % (rr["label"], rr["validity"], "-"))

    rep["results"] = results
    rep["seconds"] = time.time() - t_all
    rep["utc_end"] = utc()
    print("\nseed %d done in %.1fs" % (seed, rep["seconds"]))

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "harness_seed%d%s.json" % (seed, tag)), "w") as f:
            json.dump(rep, f, indent=1, default=str)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", default="smoke", choices=list(SCALES))
    ap.add_argument("--out", default="")
    ap.add_argument("--arms", default="mlp,lstm,transformer")
    ap.add_argument("--no-refuser", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tier", default="cmaze1", choices=list(TIERS))
    a = ap.parse_args()
    run(a.seed, a.scale, a.out, tuple(a.arms.split(",")), not a.no_refuser, a.tag,
        a.tier)


if __name__ == "__main__":
    main()
