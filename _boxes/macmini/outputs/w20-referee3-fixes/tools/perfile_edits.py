# Per-file edits for prose that the style file paraphrases. Same numbers, same SRC comments, same order.
SRCX = "% SRC: w15-crossover-seeds/CROSSOVER-3SEED.md"
KSTAR_MAIN = (
 "$\\Kstar=32$ is what the frozen rule selects, and on three seeds the rule cannot separate the plateau\n"
 "$K=5\\ldots32$: the seed standard deviation of validation composition exceeds the rule's 0.02\n"
 "tolerance at $K=10$, 16, 20, 24 and 28 (0.0211, 0.0203, 0.0285, 0.0235, 0.0276) and at $\\Kstar=32$ " + SRCX + "\n"
 "(0.0259), so three seeds do not distinguish $\\Kstar$ from its neighbours. " + SRCX + "\n"
 "The finding is the plateau itself: composition is flat across $K$ to within its seed spread, while\n"
 "the honest four-part ledger falls to its minimum near $K=24$, 2,094,130 / 2,080,090 / 2,085,810 B " + SRCX + "\n"
 "on seeds 0 / 1 / 2, a minimum that is resolvable (seed band 7,060 B against neighbour gaps of " + SRCX + "\n"
 "12,203 B and 30,507 B; Appendix~\\ref{app:kcurve}, Table~\\ref{tab:kcurve3}). " + SRCX + "\n")
KSTAR_STYLE = (
 "The frozen rule selects $\\Kstar=32$; on three seeds, however, it cannot separate the plateau\n"
 "$K=5\\ldots32$. The seed standard deviation of validation composition exceeds the rule's 0.02\n"
 "tolerance at $K=10$, 16, 20, 24 and 28 (0.0211, 0.0203, 0.0285, 0.0235, 0.0276) and at $\\Kstar=32$ " + SRCX + "\n"
 "(0.0259); three seeds therefore do not distinguish $\\Kstar$ from its neighbours. " + SRCX + "\n"
 "The finding is the plateau itself: composition is flat across $K$ to within its seed spread,\n"
 "while the honest four-part ledger falls to its minimum near $K=24$, at 2,094,130 / 2,080,090 / 2,085,810 B " + SRCX + "\n"
 "on seeds 0 / 1 / 2. This minimum is resolvable, with a seed band of 7,060 B against neighbour gaps of " + SRCX + "\n"
 "12,203 B and 30,507 B (Appendix~\\ref{app:kcurve}, Table~\\ref{tab:kcurve3}). " + SRCX + "\n")
KANCHOR = "manufacture a\ncrossover. % SRC: w13-seeds-and-tables/TABLES.md\n"

CDM_COMMENT = "% cite-tagger: bib entry needed -- Golatkar, Achille, Swaminathan, Soatto, \"Training Data Protection with Compositional Diffusion Models\", arXiv:2308.01937 (not in references.bib)\n"
PHYS_COMMENT = "% cite-tagger: bib entries needed -- Raissi, Perdikaris, Karniadakis, \"Physics-informed neural networks\", J. Comput. Phys. (2019); Greydanus, Dzamba, Yosinski, \"Hamiltonian Neural Networks\", NeurIPS 2019 (neither in references.bib)\n"

SYS_MAIN = (
 "These are the screen's counts, not a claim that no such model exists: abstract-level screening\n"
 "cannot see a match buried in a method section, and indexing of preprints younger than about\n"
 "three months is incomplete (\\S\\ref{sec:limits}).\n"
 "Three families of systems built on top of units are out of scope, one sentence each.\n"
 "SISA training~\\citep{bourtoule2021machine} is a system that makes any unit, a generator included,\n"
 "exactly unlearnable by splitting its data into shards and retraining the unit on the shard that\n"
 "held the deleted rows, which reads that shard's retained data; we compare with the unit it wraps,\n"
 "whose only exact route is the retrain of \\S\\ref{sec:del-comparators}.\n"
 "Compartmentalized diffusion models (Golatkar, Achille, Swaminathan and Soatto) are a system that\n"
 "trains separate diffusion models on separate data sources, composes them at inference and forgets\n"
 "a source by dropping its model, so what is composed is models of data sources, not held-out\n"
 "combinations of factor values within one unit.\n" + CDM_COMMENT +
 "Physics-informed variants of an MLP, such as physics-informed neural networks (Raissi, Perdikaris\n"
 "and Karniadakis, 2019) and Hamiltonian neural networks (Greydanus, Dzamba and Yosinski, 2019), are\n"
 "systems that add a physical loss or structure to an MLP unit; the plain MLP is our rival, and the\n"
 "store's only relation to physics is the zero-dynamics identity of \\S\\ref{sec:clue}.\n" + PHYS_COMMENT + "\n")
SYS_STYLE = (
 "These counts describe the screen; they do not claim that no such model exists. Abstract-level\n"
 "screening cannot see a match buried in a method section, and the indexing of preprints younger\n"
 "than about three months is incomplete (\\S\\ref{sec:limits}).\n"
 "Three families of systems built on top of units are out of scope, and we distinguish each in one sentence.\n"
 "SISA training~\\citep{bourtoule2021machine} is a system: it makes any unit, a generator included,\n"
 "exactly unlearnable by splitting its data into shards and retraining the unit on the shard that\n"
 "held the deleted rows, reading that shard's retained data, and we compare with the unit it wraps,\n"
 "whose only exact route is the retrain of \\S\\ref{sec:del-comparators}.\n"
 "Compartmentalized diffusion models (Golatkar, Achille, Swaminathan and Soatto) are a system that\n"
 "trains separate diffusion models on separate data sources, composes them at inference and forgets a\n"
 "source by dropping its model; what they compose is models of data sources, not held-out\n"
 "combinations of factor values within one unit.\n" + CDM_COMMENT +
 "Physics-informed variants of an MLP, such as physics-informed neural networks (Raissi, Perdikaris\n"
 "and Karniadakis, 2019) and Hamiltonian neural networks (Greydanus, Dzamba and Yosinski, 2019), are\n"
 "systems that add a physical loss or structure to an MLP unit; our rival is the plain MLP, and the\n"
 "store relates to physics only through the zero-dynamics identity of \\S\\ref{sec:clue}.\n" + PHYS_COMMENT + "\n")

UNIT_MAIN = (
 "The Naive CLU is a \\emph{naive AI unit}: one model with no system built around it, in the sense\n"
 "in which an MLP, an LSTM or a transformer is a unit. We compare it only with other naive units, an\n"
 "MLP, an LSTM, a transformer and a continuous-input MLP, matched on the read path\n"
 "(\\S\\ref{sec:object}). We do not claim the best compositional-generation system or the best\n"
 "unlearning system. Systems built on top of units, such as sharded retraining, compositional\n"
 "diffusion and physics-informed variants of an MLP, are out of scope by definition; each is cited\n"
 "in one sentence below, with the unit/system distinction stated.\n\n")
UNIT_STYLE = (
 "The Naive CLU is a \\emph{naive AI unit}: one model with no system built around it, in the sense\n"
 "in which an MLP, an LSTM or a transformer is a unit. We therefore compare it only with other naive\n"
 "units: an MLP, an LSTM, a transformer and a continuous-input MLP, each matched on the read path\n"
 "(\\S\\ref{sec:object}). We claim neither the best compositional-generation system nor the best\n"
 "unlearning system. Systems built on top of units, such as sharded retraining, compositional\n"
 "diffusion and physics-informed variants of an MLP, are out of scope by definition; we cite each\n"
 "in one sentence below and state the unit/system distinction.\n\n")

NOV = "% SRC: _thinkpad/w8/w8-novelty/NOVELTY.md\n"
MAIN = [
 ("A-abs",
  "Compositional generation is a crowded area\nand exact unlearning is an established one, but their intersection is empty in a precise sense: a mechanical\n"
  "screen of approximately 4,050 works finds no \\emph{factor-compositional} generative model (one that generates held-out combinations of factor values) that deletes training rows\n"
  "exactly while reading no data and retraining nothing. " + NOV,
  "Compositional generation is a crowded area\nand exact unlearning is an established one. We work at the level of a single \\emph{naive unit} and\n"
  "compare it only with other naive units (an MLP, an LSTM, a transformer and a continuous-input MLP):\n"
  "we do not claim the best compositional-generation or unlearning system, and systems built on top of\n"
  "units, such as sharded retraining or compositional diffusion, are out of scope.\n"),
 ("A-unit", "Generating held-out combinations of factors is an old and active problem",
  UNIT_MAIN + "Generating held-out combinations of factors is an old and active problem"),
 ("A-screen", "The intersection of the two literatures is empty on a systematic screen. A forward-citation\ncrawl",
  "A systematic screen of both literatures records what we could find. A forward-citation\ncrawl"),
 ("A-dist", "One distinction in this intersection is ours and worth a sentence.", "One distinction is ours and worth a sentence."),
 ("D1", "Pre-declared landscape controls\n(\\texttt{landscape\\_removed}",
  "Four controls, which we call the\n\\emph{pre-declared landscape controls} throughout (\\texttt{landscape\\_removed}"),
 ("E28", "one of six pre-registered bars (the V-curve argmin)", "one of six pre-registered criteria (the V-curve argmin)"),
 ("E29", "The full statements, bars and misses are in", "The full statements, criteria and misses are in"),
 ("E31", "Two caveats live in the gate file it cites:", "Two caveats live in the pre-registration record it cites:"),
 ("F1", KANCHOR, KANCHOR + KSTAR_MAIN),
]
STYLE = [
 ("A-abs",
  "Compositional generation is extensively studied and exact unlearning established; their intersection, however, is empty in a precise sense: a mechanical\n"
  "screen of approximately 4,050 works finds no \\emph{factor-compositional} generative model (one that generates held-out combinations of factor values) that deletes training rows\n"
  "exactly while reading no data and retraining nothing. " + NOV,
  "Compositional generation is extensively studied and exact unlearning established. We work at the level of a single \\emph{naive unit} and\n"
  "compare it only with other naive units (an MLP, an LSTM, a transformer and a continuous-input MLP).\n"
  "We do not claim the best compositional-generation or unlearning system, and systems built on top of\n"
  "units, such as sharded retraining or compositional diffusion, are out of scope.\n"),
 ("A-unit", "Generating held-out combinations of factors is a long-standing and active problem",
  UNIT_STYLE + "Generating held-out combinations of factors is a long-standing and active problem"),
 ("A-screen", "However, on a systematic screen the intersection of the two literatures is empty. A\nforward-citation crawl",
  "A systematic screen of both literatures records what we could find. A\nforward-citation crawl"),
 ("A-dist", "One distinction within this intersection is ours and merits a sentence.", "One distinction is ours and merits a sentence."),
 ("D1", "store, the pre-declared landscape controls (\\texttt{landscape\\_removed}",
  "store, four controls, which we call the \\emph{pre-declared landscape controls} throughout (\\texttt{landscape\\_removed}"),
 ("E28", "One of six pre-registered bars (the V-curve", "One of six pre-registered criteria (the V-curve"),
 ("E29", "Full statements, bars and misses:", "Full statements, criteria and misses:"),
 ("E31", "The gate file it cites records two caveats.", "The pre-registration record it cites holds two caveats."),
 ("F1", KANCHOR, KANCHOR + KSTAR_STYLE),
]
# block replacement: from "Two works outside the screen" up to and including "three months is incomplete.\n" (+ following blank line kept)
BLOCK = ("Two works outside the screen", "three months is incomplete.\n")
