"""w21 spoke/maze-t-seed4 -- ZERO-RUN diagnosis of the held-out n-way tuples (no fitting).

    python scripts/w21_seed_diagnosis.py <out_json> [--seeds 0,1,2,3,4,5] [--legs test,val]

For every seed and every held-out n-way 4-tuple (w_agent, w_door, w_tint, w_dir) this reads
ONLY the banks and the design (W16.banks, T.make_holdout, T.split_rows, W16.cap_split -- the
instrument, unchanged) and prints what the TRAIN rows (the ``seen`` split, which is every
arm's training data) say about that tuple:

* n_eval       -- scored eval rows of the tuple (capped split, exactly what is scored);
* truth        -- the eval rows' outcome ``(legal, ev, pocket_next)`` distribution
                  (displacement is a function of w_dir alone and is left out);
* triple_support -- train rows sharing (w_agent, w_door, w_tint) with ANY direction, and how
                  many of them carry the SAME outcome as the eval truth;
* pair_support -- for each of the six word pairs of the tuple, train rows carrying that pair
                  and the modal outcome among them (does the pair's evidence AGREE with the
                  truth?).

Classification (a definition, stated here and applied mechanically -- it is not a bar):
  A  REACHABLE-FROM-TRIPLE  the (agent, door, tint) triple occurs in train with the SAME
                            outcome (only the direction differs);
  B  REACHABLE-FROM-PAIRS   no such triple, but every colour pair (agent,door), (agent,tint),
                            (door,tint) has train support and its modal outcome equals the
                            truth;
  C  UNREACHABLE-CONFLICT   no such triple; every colour pair has support, but at least one
                            pair's train evidence says a DIFFERENT outcome -- the answer needs
                            a 3-way (agent,door,tint) interaction that train never shows;
  D  UNREACHABLE-NO-SUPPORT no such triple and at least one colour pair has ZERO train rows.

Second, colour-FREE criterion (``relation``): the tuple's signature = which roles are empty and
which colours are EQUAL (pocket==door, pocket==tint, door==tint); its outcome is re-expressed
colour-free (legal, event relative to the pocket, pocket_next empty/same/other).  A tuple is
REACHABLE-BY-RELATION if train rows with the same signature (any colours) have that outcome as
their mode.  C/D tuples that are reachable by relation are unreachable for a colour-IDENTITY
(one-hot + pair) design but not for a representation that codes colour equality.
"""
from __future__ import annotations
import json
import os
import sys
import time
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
import w15_cmaze_t_run as R                                         # noqa: E402
import w16_cmaze_t as W16                                           # noqa: E402
K = R.K_COLOURS
PAIRS = {"agent,door": (0, 1), "agent,tint": (0, 2), "door,tint": (1, 2),
         "agent,dir": (0, 3), "door,dir": (1, 3), "tint,dir": (2, 3)}
COLOUR_PAIRS = ("agent,door", "agent,tint", "door,tint")
def key4(w):
    return (int(w[0]), int(w[2]), int(w[3]), int(w[4]))


def outc(r):
    return (int(r.legal), int(r.ev), int(r.pocket_next))


def rel_sig(a, d, t, k=None):
    """Colour-FREE signature of (agent, door, tint): which roles are empty and which colours
    are EQUAL -- the relation, not the colour identity."""
    k = K if k is None else k
    dk = "none" if d == 0 else ("closed" if 1 <= d <= k else ("open" if d == k + 1 else "wall"))
    return (a == 0, dk, t == 0,
            bool(a and dk == "closed" and a == d), bool(a and t and a == t),
            bool(dk == "closed" and t and d == t))


def rel_out(a, o):
    """Colour-FREE outcome: (legal, event kind relative to the pocket, pocket_next relation)."""
    legal, ev, pn = o
    evk = "none" if ev == 0 else ("pickup" if ev <= K else
                                  ("consume_pocket" if ev - K == a else "consume_other"))
    pr = "empty" if pn == 0 else ("same" if pn == a else "other")
    return (legal, evk, pr)


def family_of(tup, value_colour):
    fam = T._nway_candidates(K, value_colour)
    return [f for f in T.NWAY_FAMILIES if tuple(tup) in fam[f]]


def diagnose(seed, leg):
    t0 = time.time()
    _b, _e, (tr_insts, tr_rows, _n, _m), (ev_insts, ev_rows, _n2, _m2) = W16.banks(seed, leg)
    ho = T.make_holdout(K, seed=seed)
    train = T.split_rows(tr_rows, ho, K).get("seen", [])
    ev = W16.cap_split(T.split_rows(ev_rows, ho, K), seed).get("nway", [])
    tk = [key4(r.words) for r in train]
    to = [outc(r) for r in train]
    out = []
    for tup in ho.nway:
        tup = tuple(int(x) for x in tup)
        rs = [r for r in ev if key4(r.words) == tup]
        truth = Counter(outc(r) for r in rs)
        tmode = truth.most_common(1)[0][0] if truth else None
        tri = [o for k4, o in zip(tk, to) if k4[:3] == tup[:3]]
        rec = {"tuple": list(tup), "family": family_of(tup, ho.value_colour),
               "n_eval": len(rs),
               "truth_outcomes": {str(k): v for k, v in truth.most_common()},
               "truth_modal_outcome": list(tmode) if tmode else None,
               "triple_support_n": len(tri),
               "triple_support_same_outcome_n": int(sum(o == tmode for o in tri)) if tmode else 0,
               "triple_support_directions": sorted(set(k4[3] for k4 in tk if k4[:3] == tup[:3]))}
        ps = {}
        for nm, (i, j) in PAIRS.items():
            sup = [o for k4, o in zip(tk, to) if k4[i] == tup[i] and k4[j] == tup[j]]
            c = Counter(sup)
            mode = c.most_common(1)[0][0] if c else None
            ps[nm] = {"n": len(sup),
                      "modal_outcome": list(mode) if mode else None,
                      "modal_frac": (c.most_common(1)[0][1] / len(sup)) if sup else None,
                      "frac_same_as_truth": (sum(o == tmode for o in sup) / len(sup))
                      if (sup and tmode) else None,
                      "agrees_with_truth": bool(mode == tmode) if (mode and tmode) else None}
        rec["pair_support"] = ps
        #: which held-out 2-WAY pair (ho.pairs, word index i/j of the 5-word tuple) removes a
        #: colour pair of this tuple from train by construction
        blk = []
        for (i, va, j, vb) in ho.pairs:
            if i == 0 and va == tup[0] and ((j == 2 and vb == tup[1]) or (j == 3 and vb == tup[2])):
                blk.append([int(i), int(va), int(j), int(vb)])
        rec["blocked_by_heldout_2way_pair"] = blk
        if tmode is None:
            cls = "NO-EVAL-ROWS"
        elif rec["triple_support_same_outcome_n"] > 0:
            cls = "A REACHABLE-FROM-TRIPLE"
        elif any(ps[p]["n"] == 0 for p in COLOUR_PAIRS):
            cls = "D UNREACHABLE-NO-SUPPORT"
        elif all(ps[p]["agrees_with_truth"] for p in COLOUR_PAIRS):
            cls = "B REACHABLE-FROM-PAIRS"
        else:
            cls = "C UNREACHABLE-CONFLICT"
        rec["class"] = cls
        # colour-RELATION reachability (w21 addition): train rows with the same colour-free
        # signature under OTHER colours, and whether their colour-free outcome = the truth's
        if tmode is not None:
            sig = rel_sig(*tup[:3])
            rtruth = rel_out(tup[0], tmode)
            rsup = [rel_out(k4[0], o) for k4, o in zip(tk, to) if rel_sig(*k4[:3]) == sig]
            rc = Counter(rsup)
            rmode = rc.most_common(1)[0][0] if rc else None
            rec["relation"] = {"signature": [str(x) for x in sig], "truth_rel_outcome": list(rtruth),
                               "train_rows_same_signature": len(rsup),
                               "train_modal_rel_outcome": list(rmode) if rmode else None,
                               "frac_same_rel_outcome": (rc[rtruth] / len(rsup)) if rsup else None,
                               "reachable_by_relation": bool(rmode == rtruth)}
        else:
            rec["relation"] = None
        rec["why"] = why(rec)
        out.append(rec)
    n_ev = sum(r["n_eval"] for r in out)
    summ = Counter(r["class"] for r in out)
    rows_by = Counter()
    for r in out:
        rows_by[r["class"]] += r["n_eval"]
    unreach = [r for r in out if r["class"].startswith(("C", "D"))]
    unreach_rel = [r for r in unreach if not (r["relation"] or {}).get("reachable_by_relation")]
    return {"seed": seed, "leg": leg, "value_colour": int(ho.value_colour),
            "n_train_rows_seen": len(train), "n_eval_nway_rows": n_ev,
            "n_tuples": len(out), "class_count_tuples": dict(summ),
            "class_count_eval_rows": dict(rows_by),
            "n_unreachable_tuples": len(unreach),
            "unreachable_eval_rows": int(sum(r["n_eval"] for r in unreach)),
            "unreachable_row_fraction": (sum(r["n_eval"] for r in unreach) / n_ev) if n_ev else None,
            "identified_row_fraction":
                (1 - sum(r["n_eval"] for r in unreach) / n_ev) if n_ev else None,
            "NOTE_identified": ("NOT an upper bound on any arm's score: an arm can be right on an "
                                "unidentified row by prior/default (e.g. predicting the modal "
                                "outcome); it says only that train does not DETERMINE the answer"),
            "heldout_2way_pairs": [list(p) for p in ho.pairs],
            "n_unidentified_tuples_ALSO_unreachable_by_colour_relation": len(unreach_rel),
            "unidentified_rows_ALSO_unreachable_by_colour_relation":
                int(sum(r["n_eval"] for r in unreach_rel)),
            "unreachable_by_relation_list": [[r["tuple"], r["family"], r["n_eval"]]
                                             for r in unreach_rel],
            "unreachable_list": [[r["tuple"], r["family"], r["n_eval"], r["class"]] for r in unreach],
            "tuples": out, "seconds": time.time() - t0}


def why(r):
    if r["class"].startswith("A"):
        return ("(agent,door,tint) seen in train with the same outcome (directions %s); only "
                "the direction is new" % r["triple_support_directions"])
    if r["class"].startswith("B"):
        return "no triple in train; all three colour pairs present and each says the true outcome"
    if r["class"].startswith("D"):
        z = [p for p in COLOUR_PAIRS if r["pair_support"][p]["n"] == 0]
        return ("no triple in train and colour pair(s) %s have ZERO train rows; blocked by "
                "held-out 2-way pair(s) %s (word idx 0=agent,2=door,3=tint)"
                % (z, r["blocked_by_heldout_2way_pair"]))
    if r["class"].startswith("C"):
        bad = ["%s (n=%d, train says %s, %.0f%% of its rows match truth)"
               % (p, r["pair_support"][p]["n"], r["pair_support"][p]["modal_outcome"],
                  100 * (r["pair_support"][p]["frac_same_as_truth"] or 0))
               for p in COLOUR_PAIRS if not r["pair_support"][p]["agrees_with_truth"]]
        return ("no (agent,door,tint) triple with this outcome in train (triple rows: %d); "
                "truth %s but pair evidence disagrees: %s"
                % (r["triple_support_n"], r["truth_modal_outcome"], "; ".join(bad)))
    return ""


if __name__ == "__main__":
    op = sys.argv[1]
    kw = {sys.argv[i].lstrip("-"): sys.argv[i + 1] for i in range(2, len(sys.argv), 2)}
    seeds = [int(s) for s in kw.get("seeds", "0,1,2,3,4,5").split(",")]
    legs = kw.get("legs", "test,val").split(",")
    res = {"script": "scripts/w21_seed_diagnosis.py", "zero_run": True,
           "fitting": "NONE -- banks + hold-out design only", "k_colours": K, "runs": []}
    for leg in legs:
        for s in seeds:
            d = diagnose(s, leg)
            res["runs"].append(d)
            print("[diag seed %d %s] tuples %d eval-rows %d  unreachable tuples %d rows %d (%.3f)  "
                  "identified-frac %.3f  classes %s [%.0fs]"
                  % (s, leg, d["n_tuples"], d["n_eval_nway_rows"], d["n_unreachable_tuples"],
                     d["unreachable_eval_rows"], d["unreachable_row_fraction"] or 0,
                     d["identified_row_fraction"] or 0, d["class_count_tuples"],
                     d["seconds"]), flush=True)
            print("    ...of which ALSO unreachable by colour relation: %d tuples, %d rows"
                  % (d["n_unidentified_tuples_ALSO_unreachable_by_colour_relation"],
                     d["unidentified_rows_ALSO_unreachable_by_colour_relation"]), flush=True)
            for r in d["tuples"]:
                rl = r["relation"]
                print("    %-14s %-16s n=%-4d %-26s rel:%s(%s/%s) %s"
                      % (r["tuple"], ",".join(r["family"]), r["n_eval"], r["class"],
                         "-" if rl is None else ("Y" if rl["reachable_by_relation"] else "N"),
                         "-" if rl is None else rl["train_rows_same_signature"],
                         "-" if rl is None or rl["frac_same_rel_outcome"] is None
                         else "%.2f" % rl["frac_same_rel_outcome"], r["why"]), flush=True)
            with open(op, "w") as fh:
                json.dump(res, fh, indent=1, default=R._json_default)
