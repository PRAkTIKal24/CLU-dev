"""CMAZE-T self-test: the four audits on a small bank, in seconds.

    python scripts/w15_cmaze_t_selftest.py [n_mazes]

⛔ This is not the VAL smoke and it produces no row.  It is the cheap check that the
instrument is still the instrument -- run it after ANY change to
``chlu/exp_w15_cmaze_t*.py`` and before trusting a number.  Exit status is 0 iff every
audit passes.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chlu.exp_w15_cmaze_t as T                                    # noqa: E402
K = 4
def main(n_mazes=10):
    t0 = time.time()
    insts, rows, nat, meta = T.build_bank(4_242_000, int(n_mazes), per_tuple=2,
                                          n_natural=150)
    print("bank: %d mazes, %d stratified rows, %d natural rows  [%.1fs]"
          % (meta["n_mazes"], meta["n_rows"], meta["n_natural_rows"],
             time.time() - t0))
    print("door-tint receipt:", meta["door_tint"])

    ok = True

    r = T.replay_audit(insts, rows)
    print("1 INDEPENDENT REPLAY   %7d transitions, %d disagreements ......... %s"
          % (r["transitions_replayed"], r["disagreements"],
             "PASS" if r["pass"] else "FAIL"))
    for d in r["detail"][:5]:
        print("      ", d)
    ok &= r["pass"]

    r = T.impossible_example_audit(insts, rows)
    print("2 NO IMPOSSIBLE EXAMPLE %6d rows checked, %d without an accepting "
          "emission ... %s" % (r["rows_checked"],
                               r["rows_with_no_accepting_emission"],
                               "PASS" if r["pass"] else "FAIL"))
    ok &= r["pass"]

    ho = T.make_holdout(K, seed=0)
    sp = T.split_rows(rows, ho, K)
    rep = T.disjointness_report(sp.get("seen", []),
                                {k: v for k, v in sp.items() if k != "seen"}, ho, K)
    print("3 HOLD-OUT DISJOINTNESS colour-leak %d, n-way leak %d, 2-way leak %d ... %s"
          % (rep["train_rows_mentioning_the_held_out_colour"],
             rep["nway_combos_present_in_train"],
             rep["pair_combos_present_in_train"],
             "PASS" if rep["pass"] else "FAIL"))
    print("      rows by class:", {k: len(v) for k, v in sorted(sp.items())})
    ok &= rep["pass"]

    w = T.words_determine_transition(rows)
    print("3b WORDS DETERMINE      %d distinct tuples, %d with two outcomes ...... %s"
          % (w["distinct_word_tuples"], w["word_tuples_with_two_outcomes"],
             "PASS" if w["pass"] else "FAIL"))
    ok &= w["pass"]

    b = T.rule_b_column(rows)
    n = T.rule_b_column(nat)
    print("4 RULE B BITES (REPORTED COLUMN, never a gate)")
    print("      stratified bank %.4f   natural distribution %.4f   "
          "3-way (door=pocket=tint) rows %d"
          % (b["fraction"], n["fraction"], b["n_door_eq_pocket_eq_tint"]))

    #: the blind columns, so a change that flattens the instrument is visible here too
    print("   blind columns on the stratified bank (exact-match against the judge):")
    for mode in T.BLIND_MODES:
        good = sum(T.judge_transition(insts[r_.maze], r_.state(), r_.action,
                                      T.blind_outcome(insts[r_.maze], r_.state(),
                                                      r_.action, mode))[0]
                   for r_ in rows)
        print("      %-20s %.4f" % (mode, good / max(len(rows), 1)))

    print("\nSELF-TEST %s   [%.1fs]" % ("PASS" if ok else "FAIL", time.time() - t0))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2])))
