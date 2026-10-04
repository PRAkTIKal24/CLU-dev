"""w23 spoke/seed5-diagnosis -- QUESTION 2.  ZERO-FIT: why n-CLU-REL (CLU-REL-A10) loses the
value row, shown on the design, not asserted.

    python scripts/w23_rel_value.py <out_dir>

Part A (numbers, from the run JSONs only): the value row per seed 0-5 for the CLU and every
arm; the brief's figures (0.4888 ... 0.6038, mean 0.5537, best-arm mean 0.8676) re-derived
under each plausible "best arm" definition; the wave-plan section-9 pair "0.500 vs 0.792"
located.

Part B (mechanics, persisted objects; codec + clu.pkl file and param sha256 asserted against
sidecar and MANIFEST): for every seed 0-5
  * the 396-column design (318 base [pair block x0.5] + 6 eq columns + 6 x (7 door + 5 dir)
    eq-crosses) labelled column by column;
  * C4 = every column that NAMES the value colour 4 (a one-hot or pair cell or eq x door
    cross with a colour word = 4): all-zero on the FIT rows?  head rows W_k[C4] == 0?
  * the eq columns: fraction of fit rows / value rows on which each fires, and on value
    rows whether it fires BECAUSE of colour 4 (both words = 4);
  * read(value request) == read(value request with C4 zeroed)?  (the value colour's
    identity contributes nothing to the read, to the float);
  * value rows split by whether the TRUTH names colour 4 (pocket_next = 4 or key event on
    colour 4): CLU exact per group, per wrong fact;
  * the store: can the deployed atoms hold a colour-4 code at all (span residual,
    L2-best in-span decode) -- no colour-4 code is in the fit pool.
⛔ Nothing is fitted.
"""

from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "peer_w21_clu_improve_e7fdb9b"))   # w21_clu (pickled)

import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
import chlu.exp_w15_cmaze_t_arms as A                               # noqa: E402
import w16_cmaze_t as W16                                           # noqa: E402
import w21_clu                                                      # noqa: E402,F401
import w21_symmetric as S                                           # noqa: E402
from w23_seed5_trace import K, verify_file                          # noqa: E402

OUT = "C:/chlu/outputs"
SRC = {s: (OUT + "/w21-maze-t-clu-improve/test/persist/BOTH_seed%d/test" % s,
           OUT + "/w21-maze-t-clu-improve/test/BOTH_test_seed%d.json" % s) for s in (0, 1, 2)}
SRC.update({s: (OUT + "/w21-rel-newseeds/test/persist/BOTH_seed%d/test" % s,
                OUT + "/w21-rel-newseeds/test/BOTH_test_seed%d.json" % s) for s in (3, 4, 5)})
BRIEF_CLU = [0.4888, 0.5008, 0.5118, 0.5252, 0.6917, 0.6038]
LINES = []
WN = T.WORD_NAMES
FACTS = ("legal", "disp", "event", "pocket")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


# ----------------------------------------------------------------- Part A
def part_a():
    J = {s: json.load(open(SRC[s][1])) for s in range(6)}
    for s in range(6):
        assert J[s]["w21_clu_cfg"]["sha256"].startswith("0d4ec5b9"), s
    V = {s: J[s]["ROWS"]["value"] for s in range(6)}
    arms = [a for a in V[0] if a not in ("n",) and not a.endswith(":coverage")]
    defs = {
        "matched rivals (8k + 80k = rival: / rival@10x-steps:)":
            lambda a: a.startswith(("rival: ", "rival@10x-steps: ")),
        "+ 10x-read-floats diagnostic":
            lambda a: a.startswith(("rival: ", "rival@10x-steps: ", "diagnostic@10x-read-floats")),
        "+ 10x-floats + '+rel' diagnostics":
            lambda a: a.startswith(("rival: ", "rival@10x-steps: ", "diagnostic")),
        "every trained arm (rivals, diagnostics, controls, learned blind arms, floors; "
        "planners excluded)":
            lambda a: not a.startswith("column: ") and a != "CLU",
    }
    P("=== PART A: value row, CLU-REL-A10, BOTH TEST (run JSON ROWS.value) ===")
    P("| seed | n | CLU (JSON) | brief | abs diff | " + " | ".join(
        "best: %s" % d.split(" (")[0] for d in defs) + " |")
    P("|---|---|---|---|---|" + "---|" * len(defs))
    best = {d: [] for d in defs}
    for s in range(6):
        cells = []
        for d, f in defs.items():
            cand = {a: V[s][a] for a in arms if f(a)}
            a = max(cand, key=cand.get)
            best[d].append((a, cand[a]))
            cells.append("%.4f (%s)" % (cand[a], a))
        P("| %d | %d | %.4f | %.4f | %.1e | %s |" % (s, V[s]["n"], V[s]["CLU"], BRIEF_CLU[s],
                                                   abs(V[s]["CLU"] - BRIEF_CLU[s]),
                                                   " | ".join(cells)))
    clu = [V[s]["CLU"] for s in range(6)]
    P("| mean 6 |  | %.4f | %.4f |  | %s |" % (np.mean(clu), np.mean(BRIEF_CLU), " | ".join(
        "%.4f" % np.mean([b[1] for b in best[d]]) for d in defs)))
    P("| mean s0-2 |  | %.4f |  |  | %s |" % (np.mean(clu[:3]), " | ".join(
        "%.4f" % np.mean([b[1] for b in best[d][:3]]) for d in defs)))
    # every arm's 3-seed mean (s0-2) -- the section-9 pair
    m3 = {a: float(np.mean([V[s][a] for s in range(3)])) for a in arms}
    near = sorted(((a, v) for a, v in m3.items() if abs(v - 0.792) < 0.0005), key=lambda x: x[1])
    P("\nsection-9 '0.500 vs 0.792': CLU mean over seeds 0-2 = %.4f (%s) ; arms whose seeds-0-2 "
      "MEAN rounds to 0.792: %s" % (m3["CLU"], " / ".join("%.4f" % V[s]["CLU"] for s in range(3)),
                                    ["%s = %.4f (%s)" % (a, v, " / ".join("%.4f" % V[s][a]
                                                                           for s in range(3)))
                                     for a, v in near]))
    per_seed_0792 = [(s, a, V[s][a]) for s in range(6) for a in arms
                     if abs(V[s][a] - 0.792) < 0.0005]
    P("per-seed cells (any arm, any seed) that round to 0.792: %s" % per_seed_0792)
    return {"value_row": {s: V[s] for s in range(6)}, "clu": clu, "brief_clu": BRIEF_CLU,
            "clu_mean6": float(np.mean(clu)),
            "best_by_definition": {d: {"per_seed": best[d],
                                       "mean6": float(np.mean([b[1] for b in best[d]])),
                                       "mean_s0_2": float(np.mean([b[1] for b in best[d][:3]]))}
                                   for d in defs},
            "section9": {"clu_mean_s0_2": m3["CLU"], "arms_mean_s0_2_round_0792": near,
                         "per_seed_cells_round_0792": per_seed_0792,
                         "source": OUT + "/w21-maze-t-clu-improve/TABLES.md:105 and :252; "
                                         "FINDINGS.md:183"}}


# ----------------------------------------------------------------- Part B
def labels(dx):
    cards = dx.cards
    lay = dx.spec.layout
    lab, c4 = ["bias"], [False]
    colour_word = {0, 1, 2, 3}        # agent/pocket, key, door (closed colour 1..k), tint

    def names4(f, v):
        return f in colour_word and v == K       # door word v == K is the CLOSED door of colour 4
    for f, c in enumerate(cards):
        for v in range(c):
            lab.append("%s=%d" % (WN[f], v))
            c4.append(names4(f, v))
    for (i, j) in lay["pairs"]:
        for a in range(cards[i]):
            for b in range(cards[j]):
                lab.append("%sx%s=(%d,%d)" % (WN[i], WN[j], a, b))
                c4.append(names4(i, a) or names4(j, b))
    assert len(lab) == dx.width_base
    kind = ["base"] * dx.width_base
    for p in dx.rel_pairs:
        lab.append("eq(%s)" % ",".join(WN[q] for q in p))
        c4.append(False)
        kind.append("eq")
    for f in dx.rel_x:
        for p in dx.rel_pairs:
            for v in range(cards[f]):
                lab.append("eq(%s)x%s=%d" % (",".join(WN[q] for q in p), WN[f], v))
                c4.append(names4(f, v))
                kind.append("eqx")
    assert len(lab) == dx.width, (len(lab), dx.width)
    return lab, np.array(c4), np.array(kind)


def names4_truth(r):
    return r.pocket_next == K or r.ev in (K, 2 * K)


def part_b_seed(seed):
    import equinox as eqx
    import jax
    pd, rj = SRC[seed]
    man = json.load(open(os.path.join(pd, "MANIFEST.json")))
    hc = {n: verify_file(pd, n, man) for n in ("codec.eqx", "clu.pkl")}
    codec = eqx.tree_deserialise_leaves(os.path.join(pd, "codec.eqx"),
                                        A.Codec(jax.random.PRNGKey(0), int(T.code_dim(K)), 128))
    assert S.Persist._leaf_hash(jax.tree_util.tree_leaves(
        eqx.filter(codec, eqx.is_array))) == man["codec.eqx"]["param_sha256"]
    with open(os.path.join(pd, "clu.pkl"), "rb") as fh:
        clu = pickle.load(fh)
    assert S.Persist._leaf_hash(list(S._array_leaves(clu))) == man["clu.pkl"]["param_sha256"]
    for v in hc.values():
        v["param_match"] = True
    J = json.load(open(rj))
    _b, _e, (tr_insts, tr_rows, _n, _m), (va_insts, va_rows, _vn, _m2) = W16.banks(seed, "test")
    ho = T.make_holdout(K, seed=seed)
    assert ho.value_colour == K
    train_rows = T.split_rows(tr_rows, ho, K).get("seen", [])
    rs = W16.cap_split(T.split_rows(va_rows, ho, K), seed).get("value", [])
    assert len(rs) == J["ROWS"]["value"]["n"], (len(rs), J["ROWS"]["value"]["n"])
    Wtr, Wva = A.build_words(train_rows), A.build_words(rs)
    dx = clu.dx
    lab, C4, kind = labels(dx)
    Phi_tr = np.asarray(clu.design(Wtr)[0], np.float64)
    Phi_va = np.asarray(clu.design(Wva)[0], np.float64)
    assert np.abs(Phi_tr.T @ Phi_tr - clu.A).max() < 1e-6
    # reproduce the value row
    Zva = clu.read(Wva)
    em = T.decode_codes(K, A.codec_decode(codec, Zva))
    jd = [T.judge_transition(va_insts[r.maze], r.state(), r.action, e) for r, e in zip(rs, em)]
    ok = np.array([j[0] for j in jd])
    assert float(ok.mean()) == J["ROWS"]["value"]["CLU"], (seed, ok.mean())
    # C4 on the fit rows, and the head rows
    c4_tr_max = float(np.abs(Phi_tr[:, C4]).max())
    assert int((Wtr[:, :4] == K).any(axis=1).sum()) == 0      # no fit row names colour 4
    heads = [np.asarray(W, np.float64) for W in clu.heads]
    w_c4_max = [float(np.abs(W[C4]).max()) for W in heads]
    w_other_rms = [float(np.sqrt((W[~C4] ** 2).mean())) for W in heads]
    # read invariance: zero C4 on the value requests
    Phi0 = Phi_va.copy()
    Phi0[:, C4] = 0.0
    from chlu.core.gen_request_head import head_read
    Z0 = head_read(clu.heads, clu.store, Phi0.astype(np.float32))
    Zf = head_read(clu.heads, clu.store, Phi_va.astype(np.float32))
    inv = float(np.abs(Z0 - Zf).max())
    em0 = T.decode_codes(K, A.codec_decode(codec, Z0))
    same_emit = float((em0 == em).all(axis=1).mean())
    # eq columns
    eqcols = np.flatnonzero(kind == "eq")
    eq_rows = []
    for j in eqcols:
        p = dx.rel_pairs[list(eqcols).index(j)]
        both4 = np.all([Wva[:, q] == K for q in p], axis=0)
        eq_rows.append({"col": int(j), "label": lab[j],
                        "fit_rows_active": int((Phi_tr[:, j] != 0).sum()),
                        "fit_rows_active_with_colour_4": int(
                            ((Phi_tr[:, j] != 0) & (Wtr[:, :4] == K).any(axis=1)).sum()),
                        "value_rows_active": int((Phi_va[:, j] != 0).sum()),
                        "value_rows_active_because_both_words_are_4": int(
                            ((Phi_va[:, j] != 0) & both4).sum()),
                        "head_row_norm_per_well": [float(np.linalg.norm(W[j])) for W in heads]})
    eqx_c4 = np.flatnonzero((kind == "eqx") & C4)
    eqx_c4_rec = {"n_cols": int(eqx_c4.size), "labels": [lab[j] for j in eqx_c4],
                  "fit_rows_nonzero": int((Phi_tr[:, eqx_c4] != 0).any(axis=1).sum()),
                  "value_rows_nonzero": int((Phi_va[:, eqx_c4] != 0).any(axis=1).sum()),
                  "head_rows_max_abs": [float(np.abs(W[eqx_c4]).max()) for W in heads]}
    # value rows: truth names colour 4?
    t4 = np.array([names4_truth(r) for r in rs])
    truth = np.array([[r.legal, r.disp, r.ev, r.pocket_next] for r in rs])
    wrong = (em != truth)
    grp = {}
    for nm, m in (("truth names colour 4", t4), ("truth does not name colour 4", ~t4)):
        grp[nm] = {"n": int(m.sum()), "clu_exact": int(ok[m].sum()),
                   "clu_exact_frac": float(ok[m].mean()) if m.any() else None,
                   "wrong_fact_counts": {FACTS[q]: int(wrong[m, q].sum()) for q in range(4)},
                   "emitted_colour_4_anywhere": int(((em[m, 3] == K) | np.isin(em[m, 2], [K, 2 * K])).sum())}
    # per colour-4 role
    roles = defaultdict(list)
    for i, w in enumerate(Wva):
        rl = [WN[f] for f in (0, 1, 2, 3) if w[f] == K]
        roles["+".join(rl) + (" | truth names 4" if t4[i] else " | truth colour-free")].append(i)
    role_tab = {}
    for rname, ii in sorted(roles.items()):
        ii = np.array(ii)
        eq_act = [lab[j] for j in eqcols if (Phi_va[ii, j] != 0).any()]
        role_tab[rname] = {"n": int(ii.size), "clu_exact": int(ok[ii].sum()),
                           "eq_cols_active_on_some_rows": eq_act,
                           "frac_rows_with_any_eq_active": float(
                               (Phi_va[np.ix_(ii, eqcols)] != 0).any(axis=1).mean()),
                           "wrong_facts": {FACTS[q]: int(wrong[ii, q].sum()) for q in range(4)}}
    # per 4-tuple (w_agent, w_door, w_tint, w_dir) -- the per-tuple table
    tup = defaultdict(list)
    for i, w in enumerate(Wva):
        tup["%d,%d,%d,%d" % (w[0], w[2], w[3], w[4])].append(i)
    tup_tab = []
    for t, ii in tup.items():
        ii = np.array(ii)
        tup_tab.append({"tuple": t, "n": int(ii.size), "clu_exact": int(ok[ii].sum()),
                        "truth_names_4": int(t4[ii].sum()),
                        "eq_active_rows": int((Phi_va[np.ix_(ii, eqcols)] != 0).any(axis=1).sum()),
                        "C4_active_rows": int((Phi_va[np.ix_(ii, np.flatnonzero(C4))] != 0)
                                              .any(axis=1).sum())})
    tup_tab.sort(key=lambda r: -r["n"])
    # store: can the atoms hold a colour-4 code?
    codes_fit = sorted({(r.legal, r.disp, r.ev, r.pocket_next) for r in train_rows})
    codes_va = sorted({(int(a), int(b), int(c), int(d)) for a, b, c, d in truth})
    codes = sorted(set(codes_fit) | set(codes_va))
    Z = A.codec_encode(codec, np.stack([T.encode_outcome(K, *c) for c in codes])).astype(np.float64)
    Zp = np.zeros_like(Z)
    for G, ch in zip(clu.store.modes, clu.store.well_channels):
        G = np.asarray(G, np.float64)
        Uu, sv, _ = np.linalg.svd(G.T, full_matrices=False)
        Q = Uu[:, sv > sv.max() * 1e-8]
        Zp[:, ch] = (Z[:, ch] @ Q) @ Q.T
    rel = np.linalg.norm(Z - Zp, axis=1) / np.linalg.norm(Z, axis=1)
    dp = T.decode_codes(K, A.codec_decode(codec, Zp.astype(np.float32)))
    rt = T.decode_codes(K, A.codec_decode(codec, Z.astype(np.float32)))
    c4c = [i for i, c in enumerate(codes) if c[3] == K or c[2] in (K, 2 * K)]
    nc4 = [i for i, c in enumerate(codes) if i not in c4c]
    store = {"n_codes_naming_4_in_value_truth": len(c4c),
             "n_codes_naming_4_in_fit_pool": int(sum(1 for c in codes_fit
                                                      if c[3] == K or c[2] in (K, 2 * K))),
             "codec_roundtrip_ok_codes_naming_4": int(sum((rt[i] == codes[i]).all() for i in c4c)),
             "span_rel_resid_codes_naming_4_mean": float(rel[c4c].mean()) if c4c else None,
             "span_rel_resid_other_codes_mean": float(rel[nc4].mean()),
             "l2_oracle_ok_codes_naming_4": int(sum((dp[i] == codes[i]).all() for i in c4c)),
             "l2_oracle_ok_other_codes": int(sum((dp[i] == codes[i]).all() for i in nc4)),
             "n_other_codes": len(nc4)}
    return {"seed": seed, "hash_checks": hc, "persist": pd, "run_json": rj,
            "design_width": int(dx.width), "n_C4_cols": int(C4.sum()),
            "n_eq_cols": int(eqcols.size), "n_eqx_cols": int((kind == "eqx").sum()),
            "value_n": len(rs), "clu_value_exact_reproduced": float(ok.mean()),
            "C4_fit_rows_max_abs": c4_tr_max, "head_rows_C4_max_abs": w_c4_max,
            "head_rows_other_rms": w_other_rms,
            "read_invariance_zeroing_C4_max_abs": inv, "emissions_identical_frac": same_emit,
            "eq_columns": eq_rows, "eqx_crosses_naming_4": eqx_c4_rec,
            "by_truth": grp, "by_role": role_tab, "per_tuple": tup_tab, "store": store}


def main(out_dir):
    t0 = time.time()
    res = {"script": "scripts/w23_rel_value.py", "zero_fit": True, "partA": part_a()}
    P("\n=== PART B: mechanics, CLU-REL-A10, value row (colour 4 = the never-trained colour) ===")
    B = {}
    for s in range(6):
        B[s] = r = part_b_seed(s)
        P("\n--- seed %d (hashes OK; value exact %.4f reproduced from the object; design width %d;"
          " %d columns name colour 4; %d eq + %d eq-crosses) ---"
          % (s, r["clu_value_exact_reproduced"], r["design_width"], r["n_C4_cols"],
             r["n_eq_cols"], r["n_eqx_cols"]))
        P("  colour-4 columns on the %s FIT rows: max|Phi| = %.1f  -> head rows W_k[C4] max|.| = %s"
          " (other rows rms %s)" % ("all", r["C4_fit_rows_max_abs"],
                                    ["%.1e" % v for v in r["head_rows_C4_max_abs"]],
                                    ["%.3f" % v for v in r["head_rows_other_rms"]]))
        P("  eq x door=4 crosses: %d cols, fit rows non-zero %d, value rows non-zero %d, head rows "
          "max|.| %s" % (r["eqx_crosses_naming_4"]["n_cols"],
                         r["eqx_crosses_naming_4"]["fit_rows_nonzero"],
                         r["eqx_crosses_naming_4"]["value_rows_nonzero"],
                         ["%.1e" % v for v in r["eqx_crosses_naming_4"]["head_rows_max_abs"]]))
        P("  read(value) vs read(value with every colour-4 column zeroed): max|dz| = %.2e ; "
          "emissions identical on %.4f of rows" % (r["read_invariance_zeroing_C4_max_abs"],
                                                   r["emissions_identical_frac"]))
        P("  | eq column | fit rows active | value rows active | ...of which both words = 4 | "
          "head row norm (w0, w1) |")
        P("  |---|---|---|---|---|")
        for e in r["eq_columns"]:
            P("  | %s | %d | %d | %d | %.3f, %.3f |"
              % (e["label"], e["fit_rows_active"], e["value_rows_active"],
                 e["value_rows_active_because_both_words_are_4"], *e["head_row_norm_per_well"]))
        P("  | value rows by truth | n | CLU exact | wrong legal/disp/event/pocket | CLU emits "
          "colour 4 anywhere |")
        P("  |---|---|---|---|---|")
        for g, v in r["by_truth"].items():
            P("  | %s | %d | %d (%.3f) | %s | %d |"
              % (g, v["n"], v["clu_exact"], v["clu_exact_frac"] or 0,
                 "/".join(str(v["wrong_fact_counts"][f]) for f in FACTS),
                 v["emitted_colour_4_anywhere"]))
        st = r["store"]
        P("  store: codes naming 4 in value truth %d, in fit pool %d; codec round-trip ok %d/%d; "
          "span rel-resid mean %.3f (codes naming 4) vs %.3f (others); L2-best in-span decode ok "
          "%d/%d (naming 4) vs %d/%d (others)"
          % (st["n_codes_naming_4_in_value_truth"], st["n_codes_naming_4_in_fit_pool"],
             st["codec_roundtrip_ok_codes_naming_4"], st["n_codes_naming_4_in_value_truth"],
             st["span_rel_resid_codes_naming_4_mean"] or -1, st["span_rel_resid_other_codes_mean"],
             st["l2_oracle_ok_codes_naming_4"], st["n_codes_naming_4_in_value_truth"],
             st["l2_oracle_ok_other_codes"], st["n_other_codes"]))
        P("  | colour-4 role | truth | n | CLU exact | rows with any eq active | wrong "
          "legal/disp/event/pocket |")
        P("  |---|---|---|---|---|---|")
        for rn, v in r["by_role"].items():
            role, tr = rn.split(" | ")
            P("  | %s | %s | %d | %d | %.3f | %s |"
              % (role or "(none)", tr, v["n"], v["clu_exact"], v["frac_rows_with_any_eq_active"],
                 "/".join(str(v["wrong_facts"][f]) for f in FACTS)))
    res["partB"] = B
    # pooled
    pool = defaultdict(lambda: [0, 0])
    for s in range(6):
        for g, v in B[s]["by_truth"].items():
            pool[g][0] += v["n"]
            pool[g][1] += v["clu_exact"]
    P("\npooled 6 seeds: " + "; ".join("%s: %d/%d = %.4f" % (g, v[1], v[0], v[1] / v[0])
                                       for g, v in pool.items()))
    res["pooled_by_truth"] = {g: {"n": v[0], "clu_exact": v[1]} for g, v in pool.items()}
    res["seconds"] = time.time() - t0
    with open(os.path.join(out_dir, "Q2-REL-VALUE.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(out_dir, "Q2-REL-VALUE.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    P("[done %.0fs]" % res["seconds"])


if __name__ == "__main__":
    main(sys.argv[1])
