#!/usr/bin/env python3
r"""w17 cite-tagger: insert \citep / \nocite into DRAFT-v2.tex and DRAFT-v2-style.tex.

INSERTION ONLY. Each rule is (anchor, command). The anchor is a phrase of the draft's own prose;
it must match EXACTLY ONCE in each file, or the script aborts without writing. The command is
inserted immediately after the anchor, and nothing else in the file changes.

  \citep{k}  -> inserted as "~\citep{k}" where the draft names a method/claim without author-year.
  \nocite{k} -> inserted where the draft ALREADY prints a hand-typed author-year string
                ("Tenenbaum and Freeman, 2000"); a visible cite would print it twice, and
                replacing the string would be a prose change. \nocite puts the work in the
                reference list and leaves the rendered sentence unchanged.

Anchors are given as alternatives "v2 wording|style wording" where the paraphrase differs.
Usage: insert_cites.py <in.tex> <out.tex>
"""
import re
import sys

# (anchor alternatives, kind, keys) in document order.
RULES = [
    # ---- §1.1 positioning
    (["held-out combinations of factors is an old and active problem",
      "held-out combinations of factors is a long-standing and active problem"],
     "citep", "du2020compositional,liu2022compositional,du2023reduce,okawa2023compositional"),
    (["Tenenbaum and Freeman, 2000"], "nocite", "tenenbaum2000separating"),
    (["Vasilescu and Terzopoulos, 2002"], "nocite", "vasilescu2002tensorfaces"),
    (["Vlasic et al., 2005"], "nocite", "vlasic2005face"),
    (["MLC-VAE, 2020"], "citep", "georgopoulos2020multilinear"),
    (["(GANSpace, 2020"], "citep", "harkonen2020ganspace"),
    (["TCA, 2021"], "citep", "oldfield2021tensor"),
    (["PandA, 2023"], "citep", "oldfield2023panda"),
    (["its 2024 journal version"], "citep", "oldfield2024bilinear"),
    (["Cao and Yang (2015)"], "nocite", "cao2015towards"),
    (["Exact unlearning is also established", "Exact unlearning is likewise established"],
     "citep", "nguyen2025survey"),
    (["Ginart et al., 2019"], "nocite", "ginart2019making"),
    (["for bilinear matrix completion"], "citep", "xu2023netflix"),
    (["ridge heads on frozen backbones"], "citep", "quan2026exact"),
    (["The nearest generative attempts are approximate",
      "The closest generative attempts are approximate"], "citep", "kong2022approximate"),
    (["feature unlearning for pretrained GANs and VAEs",
      "Feature unlearning for pretrained GANs and VAEs"], "citep", "moon2024feature"),
    (["data redaction from pretrained GANs"], "citep", "kong2023data"),
    (["GAN unlearning by substitution"], "citep", "sun2025generative"),
    (["forward-citation crawl"], "citep", "priem2022openalex"),
    (["(Hall, Marshall and Martin, 2002"], "nocite", "hall2002adding,hall1999adding"),
    (["and successors"], "citep", "jin2014evd,lee2021fast"),
    # ---- §1.2 world models
    (["A world model‸ is useful"], "citep",
     "ha2018world,hafner2019dream,hafner2025mastering,schrittwieser2020mastering"),
    (["asked for states and transitions it has not seen",
      "queried for states and transitions it has not seen"], "citep",
     "micheli2023transformers,alonso2024diffusion,valevski2025diffusion,bruce2024genie"),
    (["with no tolerance, accepts it"], "citep", "vafa2024evaluating"),
    (["that it has seen only separately"], "citep", "kipf2020contrastive,zhao2022toward"),
    (["and approximate unlearning‸ misses"], "citep",
     "guo2020certified,izzo2021approximate,golatkar2020eternal"),
    # ---- §1.3 contributions: first body mention of Shapes3D
    (["On a Shapes3D‸ value hold-out at"], "citep", "burgess2018shapes,kim2018disentangling"),
    # ---- §2 object
    (["Cartesian-product training requirement"], "citep",
     "wiedemer2023compositional,lachapelle2023additive,montero2022lost"),
    (["not claim certified $(\\varepsilon,\\delta)$ unlearning"], "citep", "guo2020certified"),
    (["Learned rivals (MLP, LSTM"], "citep", "hochreiter1997long"),
    (["Learned rivals (MLP, LSTM~\\citep{hochreiter1997long}, transformer"], "citep",
     "vaswani2017attention"),
    (["it is printed, never matched"], "citep",
     "borgeaud2021improving,khandelwal2020generalization,lample2019large,wu2022memorizing,berges2024memory"),
    # ---- §3 Shapes3D
    (["on the Shapes3D value hold-out‸, with the partition fixed"], "citep",
     "montero2021role,schott2021visual"),
    (["The paired estimand with a bootstrap interval"], "citep", "efron1979bootstrap"),
    (["lace-poe-rows}: LACE‸ and PoE rows"], "citep", "nie2021controllable"),
    (["lace-poe-rows}: LACE~\\citep{nie2021controllable} and PoE‸ rows"], "citep",
     "hinton2002training"),
    # ---- §5 theory
    (["relaxing a latent state under a designed energy"], "citep",
     "hopfield1982neural,krotov2016dense,ramsauer2021hopfield,hoover2023energy,kong2024reservoir"),
    # ---- appendix: corrections absorbed (14)
    (["Ginart Def.~A.5"], "citep", "ginart2019making"),
]


def pattern(phrase):
    # "‸" marks the insertion point inside the anchor (default: the anchor's end)
    head, _, tail = phrase.partition("‸")
    enc = lambda p: r"\s+".join(re.escape(t) for t in p.split())
    sep = r"\s+" if tail[:1].isspace() else ""
    return "(" + enc(head) + ")" + (sep + enc(tail) if tail.strip() else "")


def strip_comments(s):
    # blank out LaTeX comments (unescaped %) so anchors never match inside a comment
    return re.sub(r"(?<!\\)%[^\n]*", lambda m: " " * len(m.group(0)), s)


def main(src, dst):
    s = open(src).read()
    log = []
    for alts, kind, keys in RULES:
        hits = []
        visible = strip_comments(s)
        for a in alts:
            hits += [(m.start(), m.end(1)) for m in re.finditer(pattern(a), visible)]
        # the same span can be hit by two alternatives only if they are identical text
        hits = sorted(set(hits))
        if len(hits) != 1:
            raise SystemExit(f"ABORT {src}: anchor {alts} matched {len(hits)} times")
        _, end = hits[0]
        cmd = f"~\\citep{{{keys}}}" if kind == "citep" else f"\\nocite{{{keys}}}"
        s = s[:end] + cmd + s[end:]
        line = s.count("\n", 0, end) + 1
        log.append(f"{line}\t{kind}\t{keys}")
    open(dst, "w").write(s)
    print("\n".join(log))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
