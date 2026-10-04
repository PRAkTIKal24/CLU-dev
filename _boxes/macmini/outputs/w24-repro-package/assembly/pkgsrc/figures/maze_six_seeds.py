"""maze six seeds."""

import json
import os
from nclu.paths import out_root as _out_root

EVID = str(_out_root())
SUMMARY = "cmaze_t/relational/REL-NEWSEEDS.json"
CLASSES = "cmaze_t/symmetric/MAZE-T-SEED4.json"
RUNS = {
    "A1": {
        0: "cmaze_t/symmetric/test/BOTH_test_seed0.json",
        1: "cmaze_t/symmetric/test/BOTH_test_seed1.json",
        2: "cmaze_t/symmetric/test/BOTH_test_seed2.json",
        3: "cmaze_t/symmetric/test/BOTH_test_seed3.json",
        4: "cmaze_t/symmetric/test/BOTH_test_seed4.json",
        5: "cmaze_t/symmetric/test/BOTH_test_seed5.json",
    },
    "REL": {
        0: "cmaze_t/relational/test/BOTH_test_seed0.json",
        1: "cmaze_t/relational/test/BOTH_test_seed1.json",
        2: "cmaze_t/relational/test/BOTH_test_seed2.json",
        3: "cmaze_t/relational/test/BOTH_test_seed3.json",
        4: "cmaze_t/relational/test/BOTH_test_seed4.json",
        5: "cmaze_t/relational/test/BOTH_test_seed5.json",
    },
}
PERTUPLE = "cmaze_t/relational/replay/BOTH_test_seed{}_PERTUPLE.json"
SEEDS = [0, 1, 2, 3, 4, 5]


def _load(rel):
    with open(os.path.join(EVID, rel), "r", encoding="utf-8") as f:
        return json.load(f)


def _argmax(d, pred):
    ks = [k for k in d if pred(k)]
    m = max((d[k] for k in ks))
    return (m, [k for k in ks if d[k] == m])


def read():
    summ = _load(SUMMARY)
    cls = _load(CLASSES)["classification"]
    ceil_pre = tuple(summ["ceiling_definition"])
    ceil_rel_pre = tuple(summ["ceiling_incl_rel_definition"])
    out = {
        "seeds": SEEDS,
        "class": {int(k): v for k, v in cls["class"].items()},
        "cut": cls["cut"]["cut"],
        "cut_rule": cls["cut"]["rule"],
        "a1_ceiling_for_class": {int(k): v for k, v in cls["ceilings_BOTH_TEST_nway"].items()},
        "sources": [SUMMARY, CLASSES],
        "config": {},
    }
    assert out["class"] == {int(k): v for k, v in summ["class_from_maze_t_seed4"].items()}
    for cfg in ("A1", "REL"):
        per = {}
        for s in SEEDS:
            run = _load(RUNS[cfg][s])
            out["sources"].append(RUNS[cfg][s])
            assert run["seed_index"] == s and run["leg"] == "test" and (run["condition"] == "BOTH")
            nw = {k: v for k, v in run["ROWS"]["nway"].items() if k != "n"}
            allarms = summ["nway_all_arms"][cfg][str(s)]
            for k, v in allarms.items():
                assert nw[k] == v, (cfg, s, k, nw[k], v)
            rec = {"CLU": nw["CLU"], "arms": nw, "source": RUNS[cfg][s]}
            rec["ceiling"], rec["ceiling_arms"] = _argmax(nw, lambda k: k.startswith(ceil_pre))
            rec["best_matched"], rec["best_matched_arms"] = _argmax(
                nw, lambda k: k.startswith(("rival: ", "rival@10x-steps: "))
            )
            rec["max_control"], rec["max_control_arms"] = _argmax(
                nw, lambda k: k.startswith("control: ")
            )
            if cfg == "REL":
                rec["best_rel"], rec["best_rel_arms"] = _argmax(
                    nw, lambda k: k.startswith("diagnostic+rel@")
                )
                rec["ceiling_incl_rel"], _ = _argmax(nw, lambda k: k.startswith(ceil_rel_pre))
            per[s] = rec
        out["config"][cfg] = per
    S = summ["nway_summary"]
    chk = {
        "A1 CLU": ("A1", "CLU"),
        "A1 ceiling": ("A1", "ceiling"),
        "A1 best matched rival": ("A1", "best_matched"),
        "REL CLU": ("REL", "CLU"),
        "REL ceiling": ("REL", "ceiling"),
        "REL ceiling incl +rel": ("REL", "ceiling_incl_rel"),
        "REL best matched rival": ("REL", "best_matched"),
        "REL best +rel": ("REL", "best_rel"),
        "REL max control": ("REL", "max_control"),
    }
    for lab, (cfg, key) in chk.items():
        for s in SEEDS:
            assert S[lab][str(s)] == out["config"][cfg][s][key], (lab, s)
    for s in SEEDS:
        assert out["a1_ceiling_for_class"][s] == out["config"]["A1"][s]["ceiling"]
        assert (out["a1_ceiling_for_class"][s] < out["cut"]) == (out["class"][s] == "hard")
    return out


def pertuple(seed):
    """{arm: {tuple: (correct, n)}} from the hash-linked zero-fit replay of one REL seed."""
    d = _load(PERTUPLE.format(seed))
    assert d["zero_fit"] and d["all_reproduce"] and d["all_param_hashes_match"]
    assert d["seed"] == seed and d["leg"] == "test" and (d["condition"] == "BOTH")
    out = {}
    for a, tt in d["PER_TUPLE_NWAY"].items():
        assert sum((v[0] for v in tt.values())) == d["n_nway_rows"], (seed, a)
        assert all((0 <= v[1] <= v[0] for v in tt.values()))
        out[a] = {t: (v[1], v[0]) for t, v in tt.items()}
    return (out, PERTUPLE.format(seed))
