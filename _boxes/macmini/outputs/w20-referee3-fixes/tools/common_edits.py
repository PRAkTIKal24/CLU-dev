# (old, new) pairs applied IDENTICALLY to DRAFT-v2.tex and DRAFT-v2-style.tex.
# Each old string must occur exactly once in each file.
COMMON = [
 # ---- (e) pipeline language: wave / spoke labels ----
 ("E1", r"LACE and PoE (w15 \texttt{lace-poe-rows}) are added beside", r"LACE and PoE are added beside"),
 ("E2", r"\textbf{Result (w15 \texttt{coverage-control}).}", r"\textbf{Result.}"),
 ("E3", r"\paragraph{LACE and PoE, at their own budgets (w15 \texttt{lace-poe-rows}).}", r"\paragraph{LACE and PoE, at their own budgets.}"),
 ("E4", r"in the lace-poe-rows spoke's own words, a", r"in the words of that experiment's own record, a"),
 ("E5", r"added to the generation-table registry and scored once", r"added to the arms of the generation table and scored once"),
 ("E13", r"\textbf{At three seeds (w15 \texttt{crossover-seeds}, 0 scored runs).}", r"\textbf{At three seeds (0 scored runs).}"),
 ("E14a", "The across-wells control here is\nw12's, a uniform", "The across-wells control here is\nthe ladder study's, a uniform"),
 ("E14b", "for comparability with\nw12. Seeds 3 and 4", "for comparability with\nthat study. Seeds 3 and 4"),
 ("E15", r"\paragraph{Criterion D on the longer leg, scored (w15 \texttt{bar-d-score}).}", r"\paragraph{Criterion D on the longer leg, scored.}"),
 ("E17", r"\paragraph{The decomposition re-stated on the longer leg (w15 \texttt{bar-d-score}).}", r"\paragraph{The decomposition re-stated on the longer leg.}"),
 ("E23", r"\textbf{CMAZE-T with symmetric inputs (wave 19, \texttt{maze-t-symmetric}): pending.}", r"\textbf{CMAZE-T with symmetric inputs: pending.}"),
 # ---- (e) "the Hub" ----
 ("E10", "\\textbf{That rival margin is withdrawn by the\nHub.}", "\\textbf{We withdraw that rival\nmargin.}"),
 ("E20", r"The rival margin is WITHDRAWN BY THE HUB; the store gate holds.", r"The rival margin is WITHDRAWN; the store gate holds."),
 ("E21", r"\textbf{composition over rivals: WITHDRAWN BY THE HUB.}", r"\textbf{composition over rivals: WITHDRAWN.}"),
 # ---- (e) "charter-C controls" -> the defined term ----
 ("E9", r"All four charter-C controls collapse below the bar of 0.575", r"All four pre-declared landscape controls collapse below the bar of 0.575"),
 ("E11", r"all four charter-C controls read 0.000 on every seed", r"all four pre-declared landscape controls read 0.000 on every seed"),
 ("E12", r"the four charter-C controls and the cross-maze", r"the four pre-declared landscape controls and the cross-maze"),
 ("E19", r"all four charter-C controls at most store", r"all four pre-declared landscape controls at most store"),
 ("E22", r"all four charter-C controls 0.0000 on every seed", r"all four pre-declared landscape controls 0.0000 on every seed"),
 # ---- (e) "bar k" -> "pre-registered criterion k" ----
 ("E6", "collapse on every seed (bar S1 ", "collapse on every seed (pre-registered criterion S1, "),
 ("E7a", "The pre-declared bar against the shipped\nread (every seed, margin 0.020) \\textsc{fails}", "Pre-registered criterion S2, against the shipped\nread (every seed, margin 0.020), \\textsc{fails}"),
 ("E8", r"bars were pre-registered before any number", r"criteria were pre-registered before any number"),
 ("E24", r"only after its validation bars passed", r"only after its validation criteria passed"),
 ("E25", r"Pre-declared bars: S1", r"Pre-registered criteria: S1"),
 ("E26a", r"\textbf{Pre-registered bars of the retention law", r"\textbf{Pre-registered criteria of the retention law"),
 ("E26b", r"bar & seeds & statistic & seed 0 & seed 1 & seed 2 \\", r"criterion & seeds & statistic & seed 0 & seed 1 & seed 2 \\"),
]

COMMON += [
 ("E16", r"by the bar-d-score artefact, that leg's", r"by the criterion-D scoring record, that leg's"),
 ("E27", r"One of the six pre-registered bars missed", r"One of the six pre-registered criteria missed"),
 ("E30", r"departs from the gate file's declared primary", r"departs from the pre-registration's declared primary"),
 ("E18", "the charter-C controls\ndo not collapse at the span fraction", "the pre-declared landscape\ncontrols do not collapse at the span fraction"),
 # ---- (a) headings (identical in both files) ----
 ("H1", r"\section{Introduction: positioning, and why the intersection matters}", r"\section{Introduction: a naive unit, and why its three properties matter}"),
 ("H2", r"\subsection{Positioning: compositional generation is crowded; with data-free exact deletion it is empty}", r"\subsection{Positioning: a naive unit, compared with naive units}"),
 ("H3", r"\subsection{Why the intersection matters: world models need all three at once}", r"\subsection{Why the three properties matter: world models need all three at once}"),
 # ---- (e) define "pre-registered criterion" once (sec:object, judge paragraph) ----
 ("D2", "\nSeed count is a column of every table.",
  "\nWhere an experiment was pre-registered, its success and falsification criteria were written before\n"
  "its first number; we refer to each as a \\emph{pre-registered criterion}, labelled as in that\n"
  "pre-registration (for example, criterion D or criterion S1).\n"
  "Seed count is a column of every table."),
 # ---- (b) M2: the placeholder naming both rival sets and their pools ----
 ("B1", "\nscored in no artefact.}\n",
  "\nscored in no artefact.}\n"
  "% SRCBLOCK: w12-deletion-baselines/FINDINGS.md; _thinkpad/w4/spoke-a-rivals/rivals/seed_0/mlp.rival.json; w13-seeds-and-tables/TABLES.md; ../../tasks/w20/shapes-data-matched.md\n"
  "\\PLACEHOLDER{w20 \\texttt{shapes-data-matched}: the one data-matched rival set that replaces both\n"
  "rival sets above. The two sets it replaces, and their fit pools: the headline rivals of\n"
  "Table~\\ref{tab:t7}, fit on 16,384 rows (the deletion study's declared build/smoke scale, reused as\n"
  "the headline comparators), and the ladder's rivals of Table~\\ref{tab:ladder}, fit on 32,256 rows of\n"
  "the combination-split pool; the store's own fit pool is 131,072 rows. Pending: the four naive units\n"
  "refit on the store's exact 131,072-row pool, and the store refit on the rivals' 16,384- and\n"
  "32,256-row pools, rows 3--4 and every ladder rung, 3 seeds, each rival with its convergence\n"
  "evidence; which of the two tables leads is not yet decided. May land as a NEGATIVE: against\n"
  "data-matched rivals the store's margin may shrink, vanish or reverse on some or all corners}\n"
  "% ENDSRC\n"),
 # ---- placeholder index (identical in both files) ----
 ("P1", "Ten rows remain pending, each marked in the text by a red \\textsc{placeholder} whose argument names\n"
        "the row and states that it may land as a negative or not at all; all ten await the w19 wave: the\n"
        "CMAZE-T per-arm generated states in Figures~\\ref{fig:cmazet-families}, \\ref{fig:cmazet-door-tint}\n"
        "and~\\ref{fig:cmazet-deletion} (three), and the seven rows of Table~\\ref{tab:cmazet-sym}\n"
        "(\\texttt{maze-t-symmetric}). The rows that were pending when the w15 and w16 rows were first\n"
        "written in are now written in from their artefacts:",
        "Eleven rows remain pending, each marked in the text by a red \\textsc{placeholder} whose argument names\n"
        "the row and states that it may land as a negative or not at all, and each awaits a further run: the\n"
        "data-matched rival set that replaces both rival sets of \\S\\ref{sec:shapes-t7} (one); the\n"
        "CMAZE-T per-arm generated states in Figures~\\ref{fig:cmazet-families}, \\ref{fig:cmazet-door-tint}\n"
        "and~\\ref{fig:cmazet-deletion} (three); and the seven rows of Table~\\ref{tab:cmazet-sym}. Rows\n"
        "that were pending earlier are now written in from their artefacts:"),
]
