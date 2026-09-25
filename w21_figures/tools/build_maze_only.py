"""PW-1 plot-maze: build ONLY the maze figures (fig_maze_conditions, fig_cmazet_families,
fig_cmazet_per_arm, fig_cmazet_rel_per_arm, fig_cmazet_door_then_tint) into --out, with
PROVENANCE-maze.json and DOOR-SEARCH.json. Loads no model; no scoring.

Before drawing, the read-only gate:
  * every arm's n-way score recomputed from the per-item files (correct / n) is float-equal to the
    scored run JSON's ROWS.nway[arm] (base n-CLU seeds 0-2, n-CLU-REL seeds 0-5);
  * the per-item files' sha256 equal the ones recorded in their REPRO files.
The staged-object hash gate (cmazet_verify.py) and the ambiguity count (cmazet_ambiguity.py) are
run by the caller (see LOG), not here.
usage: python tools/build_maze_only.py --out <dir>
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import make_figures_w22 as F  # noqa: E402
import cmazet_panels  # noqa: E402

W = F.W
STEMS = ("fig_maze_conditions", "fig_cmazet_families", "fig_cmazet_per_arm",
         "fig_cmazet_rel_per_arm", "fig_cmazet_door_then_tint")
ITEMS = ("cmazet_items.json", "cmazet_items_rel.json")


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def gate():
    rep = {}
    for fn in ITEMS:
        p = os.path.join(HERE, "data", fn)
        d = json.load(open(p))
        assert d["summary"]["all_arms_equal_artefact"]
        per = {}
        for s in d["seeds"]:
            run = W.load(s["artefact"])
            nw = run["ROWS"]["nway"]
            assert len(s["items"]) == s["n_nway"]
            bad = []
            for a, V in s["verdicts"].items():
                assert len(V) == s["n_nway"]
                if sum(v[0] for v in V) / len(V) != nw[a]:
                    bad.append(a)
            assert not bad, (fn, s["seed"], bad)
            per[str(s["seed"])] = dict(artefact=s["artefact"], artefact_sha256=W.sha(s["artefact"]),
                                       recorded_artefact_sha256=s["artefact_sha256"],
                                       arms_float_equal=len(s["verdicts"]))
            assert per[str(s["seed"])]["artefact_sha256"] == s["artefact_sha256"], s["artefact"]
        repro = os.path.join(HERE, "data", fn.replace(".json", ".REPRO.json"))
        rrec = json.load(open(repro)) if os.path.exists(repro) else {}
        rep[fn] = dict(sha256=fsha(p), seeds=per,
                       arms_float_equal_total=sum(v["arms_float_equal"] for v in per.values()),
                       repro_file=os.path.basename(repro),
                       repro_mentions_this_sha=fsha(p) in json.dumps(rrec))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = gate()
    F.fig_maze_conditions(a.out)
    door = cmazet_panels.build(a.out, os.path.join(HERE, "data", "cmazet_items.json"), W.PROV,
                               save_fn=F.save)
    built = sorted(W.PROV)
    assert built == sorted(STEMS), built
    with open(os.path.join(a.out, "DOOR-SEARCH.json"), "w", encoding="utf-8") as f:
        json.dump(door, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(os.path.join(a.out, "PROVENANCE-maze.json"), "w", encoding="utf-8") as f:
        json.dump({"evidence_root": W.M.EVIDENCE, "matplotlib": W.M.matplotlib.__version__,
                   "backend": W.M.matplotlib.get_backend(), "gate": g,
                   "code_sha256": {n: fsha(os.path.join(HERE, n)) for n in (
                       "make_figures_w22.py", "make_figures_w21.py", "make_figures.py",
                       "maze_six.py", "cmazet_panels.py", "cmazet_door_search.py",
                       "data/cmazet_panels.json", "data/maze_conditions.json",
                       "tools/build_maze_only.py")},
                   "figures": W.PROV}, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
