"""pw1/plot-shapes: build ONLY the five figures this branch owns, into --out, with their provenance.

  fig_crossover (make_figures_w21), fig_ladder / fig_headline_steps / fig_deletion (make_figures_w22),
  fig_encoder_adapt (make_figures). The same functions build.sh calls; nothing else is drawn.
Writes <out>/PROVENANCE-shapes.json (sources + sha256 per figure, rendered strings, PW-1 notes).
usage: python tools/build_shapes_only.py --out <dir>
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import make_figures_w22 as W22  # noqa: E402

W = W22.W
M = W.M
STEMS = ["fig_crossover", "fig_deletion", "fig_encoder_adapt", "fig_headline_steps", "fig_ladder"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rendered = {}
    for stem, fn in (("fig_crossover", W.fig_crossover), ("fig_ladder", W22.fig_ladder),
                     ("fig_headline_steps", W22.fig_headline_steps),
                     ("fig_deletion", W22.fig_deletion), ("fig_encoder_adapt", M.fig_encoder_adapt)):
        n0 = len(M.PW1_TEXT)
        fn(args.out)
        assert len(M.PW1_TEXT) == n0 + 1, stem  # the vocabulary check ran once on this figure
        rendered[stem] = M.PW1_TEXT[-1]
    figs = {}
    for stem in STEMS:
        e = dict(W.PROV[stem]) if stem in W.PROV else dict(M.PROV[stem])
        if "source_sha256" not in e:
            e["source_sha256"] = {s: W.sha(s) for s in e["sources"]}
        e["built_by"] = "make_figures.py" if stem == "fig_encoder_adapt" else (
            "make_figures_w21.py" if stem == "fig_crossover" else "make_figures_w22.py")
        e["rendered_text"] = rendered[stem]
        e["outputs_sha256"] = {}
        for ext in ("pdf", "png"):
            with open(os.path.join(args.out, f"{stem}.{ext}"), "rb") as f:
                e["outputs_sha256"][f"{stem}.{ext}"] = hashlib.sha256(f.read()).hexdigest()
        figs[stem] = e
    with open(os.path.join(args.out, "PROVENANCE-shapes.json"), "w", encoding="utf-8") as f:
        json.dump({"evidence_root": M.EVIDENCE, "matplotlib": M.matplotlib.__version__,
                   "backend": M.matplotlib.get_backend(), "branch": "pw1/plot-shapes",
                   "vocabulary_check": "make_figures.check_vocab_pw1 (PW1_BANNED) passed on every "
                                       "rendered string of every figure here",
                   "figures": figs}, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
