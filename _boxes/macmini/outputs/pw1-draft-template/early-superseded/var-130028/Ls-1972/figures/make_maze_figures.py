"""w18 placeholder-fill -- regenerate the maze figures from the landed data.

* fig_maze_wm (NEW; CMAZE-T only): w16-maze-wm/t1/A1_test_seed{0,1,2}.json -- the scored Config-A1
  TEST runs 1-3. (a) unseen n-way per arm, (b) rollout validity beside per-step exactness,
  (c) post-deletion composition on the surviving colours.
* fig_maze_readverb, fig_maze_requestlayer (CMAZE-2b / request-layer audit): the w14 data is final;
  they are rebuilt by the unchanged w17 code with the PLACEHOLDER tag switched off, because the w16
  rows they were waiting on now live in their own figure and tables.

Every plotted value is read from the JSON; bars are the mean of the three per-seed values and the
dots are the seeds. Style, colours and the save/provenance helpers are the w17 module's (vendored
unchanged as w17_make_figures.py).  usage: python make_maze_figures.py --out <dir>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w17_make_figures as M  # noqa: E402

plt = M.plt
Patch = M.Patch

M.tag_placeholder = lambda fig: None  # the maze figures are no longer placeholders

RUNS = ["w16-maze-wm/t1/A1_test_seed%d.json" % s for s in range(3)]

NWAY_ARMS = [
    ("CLU", "CLU", M.ARMS["clu"]["color"], None),
    ("rival: transformer", "Transformer", M.ARMS["transformer"]["color"], None),
    ("rival: mlp", "MLP", M.ARMS["mlp"]["color"], None),
    ("rival: lstm", "LSTM", M.ARMS["lstm"]["color"], None),
    ("control: landscape_removed", "landscape removed", M.CONTROL, None),
    ("control: landscape_shuffled", "landscape shuffled", M.CONTROL, None),
    ("control: landscape_shuffled_across_wells", "across wells (deranged)", M.CONTROL, None),
    ("control: random_atoms", "random atoms", M.CONTROL, None),
    ("floor: lookup cross-maze WORDS", "Lookup, cross-maze (floor)", M.FLOOR, "xx"),
]
ROLL_ARMS = [
    ("CLU", "CLU", M.ARMS["clu"]["color"], ""),
    ("rival: transformer", "Transformer", M.ARMS["transformer"]["color"], ""),
    ("rival: mlp", "MLP", M.ARMS["mlp"]["color"], ""),
    ("rival: lstm", "LSTM", M.ARMS["lstm"]["color"], ""),
    ("diagnostic: transformer at 10x", "Transformer 10× floats\n(diagnostic, not matched)",
     M.ARMS["transformer"]["color"], "...."),
]


def mean(v):
    return sum(v) / len(v)


def fig_maze_wm(out):
    ds = [M.load(r) for r in RUNS]
    for s, d in enumerate(ds):
        assert d["seed_index"] == s and d["leg"] == "test" and d["tag"] == "A1" and d["scored_run"]
        assert d["tolerance_used"] is None
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(M.TEXT_W, 2.7), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    # (a) unseen n-way
    for i, (k, lab, col, hatch) in enumerate(NWAY_ARMS):
        v = [d["ROWS"]["nway"][k] for d in ds]
        a1.bar(i, mean(v), 0.7, color="white" if hatch else col, edgecolor=col, lw=0.8,
               hatch=hatch or "")
        a1.scatter([i] * 3, v, s=6, color=M.INK, zorder=3, lw=0)
    a1.set_xticks(range(len(NWAY_ARMS)))
    a1.set_xticklabels([a[1] for a in NWAY_ARMS], rotation=45, ha="right", fontsize=5.6)
    a1.set_ylim(0, 1.38)
    a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a1.set_ylabel("validity, exact judge (fraction)")
    a1.set_title("(a) Unseen n-way (store row)", loc="left")

    # (b) rollout validity (solid) beside per-step exactness (hatched)
    for i, (k, lab, col, hatch) in enumerate(ROLL_ARMS):
        val = [d["ROLLOUT"]["arms"][k]["validity"] for d in ds]
        pse = [d["ROLLOUT"]["arms"][k]["per_step_exactness"] for d in ds]
        a2.bar(i - 0.2, mean(val), 0.38, color=col, edgecolor=col, lw=0.8)
        a2.bar(i + 0.2, mean(pse), 0.38, color="white", edgecolor=col, lw=0.8, hatch="////")
        a2.scatter([i - 0.2] * 3, val, s=6, color=M.INK, zorder=3, lw=0)
        a2.scatter([i + 0.2] * 3, pse, s=6, color=M.INK, zorder=3, lw=0)
    a2.set_xticks(range(len(ROLL_ARMS)))
    a2.set_xticklabels([a[1] for a in ROLL_ARMS], rotation=45, ha="right", fontsize=5.6)
    a2.set_ylim(0, 1.38)
    a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a2.set_title("(b) Rollouts", loc="left")
    a2.legend(handles=[Patch(facecolor=M.INK2, edgecolor=M.INK2, label="rollout validity"),
                       Patch(facecolor="white", edgecolor=M.INK2, hatch="////",
                             label="per-step exactness")], loc="upper left", fontsize=5.3)

    # (c) post-deletion composition on surviving colours (n-way), both scopes
    scopes = [("value", "value"), ("conjunction_colour_AND_tinted_floor", "conj.")]
    groups = []
    for j, (sk, slab) in enumerate(scopes):
        clu = [d["DELETION"][sk]["composition_on_surviving_colours"]["nway"] for d in ds]
        riv = [d["DELETION"][sk]["rival_composition_on_surviving_colours"]["nway"] for d in ds]
        for c in clu:  # the downdated store equals the store fitted without the rows
            assert c["CLU after deletion (downdate)"] == c["CLU fitted WITHOUT the rows"]
        series = [
            ("store, downdated (= fitted without)", [c["CLU after deletion (downdate)"] for c in clu],
             M.ARMS["clu"]["color"], ""),
            ("rival, retrained without", [r["rival retrain-without"] for r in riv],
             M.ARMS["transformer"]["color"], ""),
            ("rival, as trained", [r["rival as-trained"] for r in riv],
             M.ARMS["transformer"]["color"], "////"),
        ]
        for i, (lab, v, col, hatch) in enumerate(series):
            x = j * 1.2 + (i - 1) * 0.3
            a3.bar(x, mean(v), 0.28, color="white" if hatch else col, edgecolor=col, lw=0.8,
                   hatch=hatch)
            a3.scatter([x] * 3, v, s=6, color=M.INK, zorder=3, lw=0)
            if j == 0:
                groups.append(Patch(facecolor="white" if hatch else col, edgecolor=col,
                                    hatch=hatch, label=lab))
    a3.set_xticks([0, 1.2])
    a3.set_xticklabels([s[1] for s in scopes])
    a3.set_ylim(0, 1.38)
    a3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a3.set_title("(c) After deletion", loc="left")
    a3.legend(handles=groups, loc="upper left", fontsize=5.0)
    M.save(fig, out, "fig_maze_wm", RUNS, False, None,
           "CMAZE-T only, Config A1, TEST, scored runs 1-3, 3 seeds; bar = mean of the three "
           "per-seed values, dots = seeds. Rivals matched on read floats (transformer shown in "
           "(c), the best matched rival). The across-wells control is the w16 DERANGED one. "
           "(b): validity is 0 for the CLU and every matched rival on every seed; the diagnostic "
           "transformer at 10x read floats is not a matched rival (seed 2 non-zero). (c): "
           "surviving-colour n-way after deleting colour 1 (value) or colour 1 AND tinted floor "
           "(conjunction); the CLU downdate equals the fitted-without read (asserted).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_maze_wm(args.out)
    M.fig_maze_readverb(args.out)
    M.fig_maze_requestlayer(args.out)
    for stem, env in (("fig_maze_readverb", "CMAZE-2b only; w14 data, final"),
                      ("fig_maze_requestlayer", "request-layer audit (CMAZE-2 vs CMAZE-2b layer, "
                                                "no learned arm); w14 data, final; appendix")):
        M.PROV[stem]["placeholder"] = False
        M.PROV[stem]["awaiting"] = None
        M.PROV[stem]["w18"] = ("rebuilt by w18 placeholder-fill without the PLACEHOLDER tag: " + env +
                               ". The w16 world-model rows are in fig_maze_wm and Tables "
                               "tab:cmazet / tab:cmazeo, never in this figure.")
    with open(os.path.join(args.out, "PROVENANCE-maze-w18.json"), "w", encoding="utf-8") as f:
        json.dump({"evidence_root": M.EVIDENCE, "matplotlib": M.matplotlib.__version__,
                   "figures": M.PROV}, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
