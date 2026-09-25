r"""PW-2 item 1: every edit, as (id, tells, section, [(scope, old, new)], note).
scope: 'both' (the same old/new in both drafts), 'v2' or 'style' (the style twin keeps its own wording; the edit is
made at the same place in its own words). Each `old` must occur exactly once in the draft it is applied to.
Tells: T1 triad, T2 contrast, T3 self-narration, T5 heading, T6 coinage, T7 rhythm (a: split for a short sentence;
b: connective opener; c: 'This section'); B = one of the three borrowings from the style twin (plan item 4).
Every edit deletes, splits or plainly restates; the only added text is the borrowings and, at a split, the capital."""

E = []
def add(id, tells, section, pairs, note=''):
    E.append(dict(id=id, tells=tells, section=section, pairs=pairs, note=note))

# ------------------------------------------------------------------ BORROWINGS (plan item 4, last bullet)
add('B1', 'B', 'Abstract', [('v2',
r"""A world model that is to be trusted with requests it has never seen needs three things at
once: its outputs must be \emph{valid} under the task's judge, it must \emph{compose}
held-out combinations of factors from factors it has seen, and it must be able to
\emph{forget} specific training rows. Compositional""",
r"""A world model trusted with requests it has never seen needs three things at once: its outputs are \emph{valid} under the task's judge, it \emph{composes}
held-out combinations of factors from factors it has seen, and it can \emph{forget} specific
training rows. Compositional""")],
'the abstract\'s first sentence, verbatim from the style twin (line layout copied so the Head-held sentence is byte-identical to the style twin\'s)')
add('B2', 'B', 'Methods', [('v2',
r"""as a virtue of the design itself.

\paragraph{The n-CLU.}""",
r"""as a virtue of the design itself.

A claim about any of the three properties is only as precise as the definitions of the object,
its judge and its memory footprint; we therefore fix these here, before any result.

\paragraph{The n-CLU.}""")],
'the Sec. 2 (Methods) signpost, verbatim from the style twin, at the same place (second paragraph of Methods)')
add('B3', 'B', 'Discussion', [('v2',
r"""\section{Discussion}
\label{sec:limits}
% =======================================================================================

\subsection{Limitations}""",
r"""\section{Discussion}
\label{sec:limits}
% =======================================================================================

We now state what the naive unit does not do, the one measured clue, and the theory we carry.

\subsection{Limitations}""")],
'the Sec. 5 (Discussion) signpost, verbatim from the style twin; it becomes v2\'s Head-held Discussion first paragraph')
add('B4', 'B', 'Introduction', [('v2',
r"""We do not claim the best compositional-generation system or the best
unlearning system.""",
r"""We claim neither the best compositional-generation system nor the best
unlearning system.""")],
'"we claim neither ... nor", verbatim from the style twin\'s Introduction first paragraph (Head-held)')

# ------------------------------------------------------------------ ABSTRACT
add('A1', 'T3 T7a', 'Abstract', [('both',
r"""object. Deletion has limits, which we print: the book's retained statistic is""",
r"""object. Deletion has limits. The book's retained statistic is""")],
'self-narration "which we print" deleted; the colon becomes a full stop, so "Deletion has limits." (4 words) follows the 30-word sentence before it')
add('A2', 'T6', 'Abstract', [('v2',
r"""We close with what the naive unit cannot do""",
r"""We close with what the n-CLU cannot do"""), ('style',
r"""We conclude with what the naive unit cannot do""",
r"""We conclude with what the n-CLU cannot do""")],
'"naive unit" after its first mention -> n-CLU')

# ------------------------------------------------------------------ INTRODUCTION
add('I1', 'T7a', 'Introduction', [('v2',
r"""different task from deletion; GAN unlearning by substitution""",
r"""different task from deletion. GAN unlearning by substitution"""), ('style',
r"""from deletion; and GAN unlearning by substitution""",
r"""from deletion. GAN unlearning by substitution""")],
'split at the semicolon: "GAN unlearning by substitution is approximate by construction." (8 words)')
add('I2', 'T2 T7a', 'Introduction', [('v2',
r"""These are the screen's counts, not a claim that no such model exists: abstract-level screening
cannot see""",
r"""These are the screen's counts. Abstract-level screening
cannot see"""), ('style',
r"""These counts describe the screen; they do not claim that no such model exists. Abstract-level""",
r"""These counts describe the screen. Abstract-level""")],
'contrast restated as its positive half; v2: "These are the screen\'s counts." (5 words) follows the 21-word zero-works sentence')
add('I3', 'T3', 'Introduction', [('v2',
r"""Three families of systems built on top of units are out of scope, one sentence each.""",
r"""Three families of systems built on top of units are out of scope."""), ('style',
r"""Three families of systems built on top of units are out of scope, and we distinguish each in one sentence.""",
r"""Three families of systems built on top of units are out of scope.""")],
'announcement "one sentence each" deleted')
add('I4', 'T2', 'Introduction', [('v2',
r"""so what is composed is models of data sources, not held-out
combinations of factor values within one unit.""",
r"""so what is composed is models of data sources."""), ('style',
r"""what they compose is models of data sources, not held-out
combinations of factor values within one unit.""",
r"""what they compose is models of data sources.""")],
'"X, not Y" -> X')
add('I5', 'T3', 'Introduction', [('v2',
r"""One distinction is ours and worth a sentence.""",
r"""One distinction is ours."""), ('style',
r"""One distinction is ours and merits a sentence.""",
r"""One distinction is ours.""")],
'announcement "worth a sentence" deleted')
add('I6', 'T7a', 'Introduction', [('v2',
r"""Our route matches the refit, for the opposite reason:
\emph{downdate the statistic, not the truncated factors}.""",
r"""Our route matches the refit.
\emph{Downdate the statistic, not the truncated factors}."""), ('style',
r"""Our route matches the refit, for the opposite reason: \emph{downdate the statistic, not the truncated factors}.""",
r"""Our route matches the refit. \emph{Downdate the statistic, not the truncated factors}.""")],
'", for the opposite reason" deleted and split at the colon: "Our route matches the refit." (5 words) follows the 29-word decremental-PCA sentence; the kept contrast is the Introduction\'s one')
add('I7', 'T7b', 'Introduction', [('style',
r"""However, the three pull against each other:""",
r"""The three pull against each other:""")],
'style only (its itemize makes this a paragraph opener): connective "However" deleted')
add('I8', 'T3', 'Introduction', [('both',
r"""fitted on its own rows, with each baseline's convergence printed beside it: by about seven tuples""",
r"""fitted on its own rows: by about seven tuples""")],
'announcement "with each baseline\'s convergence printed beside it" deleted')
add('I9', 'T7a', 'Introduction', [('both',
r"""(Tab.~\ref{tab:ladder}, Sec.~\ref{sec:ladder}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.""",
r"""(Tab.~\ref{tab:ladder}, Sec.~\ref{sec:ladder}). For $n\ge3$ the rungs are a training-data axis.""")],
'split at the semicolon and "and not a depth axis" deleted: "For n>=3 the rungs are a training-data axis." (8 words)')
add('I10', 'T3', 'Introduction', [('both',
r"""\item \textbf{Deletion on the n-CLU and on its address book, with its limitations printed.} After""",
r"""\item \textbf{Deletion on the n-CLU and on its address book.} After""")],
'announcement "with its limitations printed" deleted')
add('I11', 'T7a', 'Introduction', [('both',
r"""The deletion reads no
  training row, but the book's retained statistic""",
r"""The deletion reads no
  training row. The book's retained statistic""")],
'split at ", but": "The deletion reads no training row." (6 words) follows the 28-word sentence')
add('I12', 'T2', 'Introduction', [('both',
r"""(Tabs.~\ref{tab:del} and~\ref{tab:rivals-del}), so this is neither a size nor a speed claim. Refusal""",
r"""(Tabs.~\ref{tab:del} and~\ref{tab:rivals-del}). Refusal""")],
'"neither a size nor a speed claim" deleted (the two facts it summarises stand in the sentence)')
add('I13', 'T2 T7a', 'Introduction', [('both',
r"""same columns close most of that gap, so it is an input-representation result, not an n-CLU result""",
r"""same columns close most of that gap. It is an input-representation result""")],
'"X, not Y" -> X, and split: "It is an input-representation result (Sec. 4.8)." (7 words)')
add('I14', 'T6', 'Introduction', [('v2',
r"""\item \textbf{Two ablations and a leaving note.}""",
r"""\item \textbf{Two ablations.}"""), ('style',
r"""\item \emph{Two ablations and a leaving note.}""",
r"""\item \emph{Two ablations.}""")],
'coinage "leaving note" dropped')
add('I15', 'T6', 'Introduction', [('v2',
r"""The naive unit's
  limitations""",
r"""The n-CLU's
  limitations"""), ('style',
r"""The
  limitations of the naive unit,""",
r"""The
  limitations of the n-CLU,""")],
'"naive unit" after its first mention -> n-CLU')
add('I16', 'T6 T3', 'Introduction', [('both',
r"""\noindent\textbf{Thickness, stated in the introduction.} The positive""",
r"""\noindent The positive""")],
'coinage "Thickness" and the announcement "stated in the introduction" dropped with the run-in label')
add('I17', 'T7a', 'Introduction', [('both',
r"""no external classifier, so no
combination of theirs can be judged.""",
r"""no external classifier. No
combination of theirs can be judged.""")],
'split: "No combination of theirs can be judged." (7 words)')

# ------------------------------------------------------------------ METHODS
add('M1', 'T7a', 'Methods', [('both',
r"""its judge and its memory footprint; we therefore fix these here, before any result.""",
r"""its judge and its memory footprint. We therefore fix these here, before any result.""")],
'the borrowed signpost (both drafts) split at its semicolon: "We therefore fix these here, before any result." (8 words)')
add('M2', 'T7a', 'Methods', [('both',
r"""read is request-only: no feature tensor enters it""",
r"""read is request-only. No feature tensor enters it""")],
'split: "The read is request-only." (4 words) follows the 31-word address-book sentence')
add('M3', 'T5 T3', 'Methods', [('both',
r"""\paragraph{The memory footprint, never netted.}""",
r"""\paragraph{The memory footprint.}""")],
'heading: comma apposition and "never netted" deleted')
add('M4', 'T3', 'Methods', [('v2',
r"""We never net them into one number, and every
byte figure names its convention""",
r"""Every
byte figure names its convention"""), ('style',
r"""We never net them into a single number, and every byte figure names its convention""",
r"""Every byte figure names its convention""")],
'announcement "We never net them into one number" deleted')
add('M5', 'T7a', 'Methods', [('v2',
r"""It is not a limit the deletable object meets.""",
r"""It is not a limit the object meets."""), ('style',
r"""It is not a limit that the deletable object meets.""",
r"""It is not a limit the object meets.""")],
'"deletable" (and style "that") deleted: the closing sentence is 8 words after the 30-word reference sentence')
add('M6', 'T5', 'Methods', [('both',
r"""\paragraph{Why a hold-out draw can remove a whole colour relation.}""",
r"""\paragraph{Colour relations under the hold-out draw.}""")],
'heading: clause -> noun phrase (plain restatement)')
add('M8', 'T3', 'Methods', [('both',
r"""criterion S1 or criterion S2).
Seed count is a column of every table.""",
r"""criterion S1 or criterion S2).""")],
'announcement "Seed count is a column of every table." deleted (the brief\'s "every row prints its seed count")')
add('M9', 'T7a', 'Methods', [('both',
r"""contribution from the statistic, and the atoms are re-eigendecomposed""",
r"""contribution from the statistic. The atoms are re-eigendecomposed""")],
'split: "Deleting rows subtracts their contribution from the statistic." (8 words) follows the 18-word sentence')
add('M10', 'T7a', 'Methods', [('v2',
r"""measured separately; we do not claim certified""",
r"""measured separately. We do not claim certified"""), ('style',
r"""measured separately; and we do not claim certified""",
r"""measured separately. We do not claim certified""")],
'split: "We do not claim certified (eps, delta) unlearning." (7 words)')

# ------------------------------------------------------------------ EXPERIMENTS
add('X1', 'T5', 'Experiments', [('both',
r"""\paragraph{Baselines: naive against naive.}""",
r"""\paragraph{Baselines.}""")],
'heading: colon-as-sentence deleted')
add('X2', 'T2 T3 T7a', 'Experiments', [('v2',
r"""The memory footprint is not a read-path parameter; it is printed, never matched~\citep{""",
r"""The memory footprint is not a read-path parameter~\citep{"""), ('style',
r"""The memory footprint is not a read-path parameter: it is printed, never matched~\citep{""",
r"""The memory footprint is not a read-path parameter~\citep{""")],
'"it is printed, never matched" deleted (contrast and self-narration)')
add('X3', 'T2', 'Experiments', [('both',
r"""It is printed beside the data-matched set, never in its place.""",
r"""It is printed beside the data-matched set.""")],
'"X, never Y" -> X; the 7-word sentence follows the 51-word sentence')
add('X4', 'T3', 'Experiments', [('both',
r"""landscape shuffled (Tab.~\ref{tab:ladder}), which we print
rather than round away.""",
r"""landscape shuffled (Tab.~\ref{tab:ladder}).""")],
'announcement "which we print rather than round away" deleted')
add('X5', 'T7c', 'Experiments', [('both',
r"""This section asks the n-CLU to \emph{generate}""",
r"""The task asks the n-CLU to \emph{generate}""")],
'no sentence opens with "This section" (plain restatement: "This section" -> "The task")')
add('X6', 'T3 T7a', 'Experiments', [('both',
r"""Baselines are matched on read floats, the landscape controls and the blind columns
are printed, and every row prints its seed count (Tabs.""",
r"""Baselines are matched on read floats. The landscape controls and the blind columns
are printed (Tabs.""")],
'announcement "every row prints its seed count" deleted; split: "Baselines are matched on read floats." (6 words)')
add('X7', 'T5', 'Experiments', [('both',
r"""\paragraph{CMAZE-T, a minor contribution: a small world-model task for composition.}""",
r"""\paragraph{A small world-model task for composition.}""")],
'heading: colon-as-sentence and apposition deleted')
add('X8', 'T7a', 'Experiments', [('both',
r"""The request is five words; the
held-out families""",
r"""The request is five words. The
held-out families""")],
'split: "The request is five words." (5 words) follows the 21-word sentence')
add('X9', 'T2', 'Experiments', [('both',
r"""reasons to report it per seed, not to rank systems on it.""",
r"""reasons to report it per seed.""")],
'"X, not Y" -> X (the "a printed floor, never a baseline" contrast is Experiments\' one)')

# ------------------------------------------------------------------ RESULTS
add('R1', 'T2 T3', 'Results', [('v2',
r"""the deletable object
itself is larger, and its memory footprint (Tab.~\ref{tab:ledger-kstar}) is printed, never
matched.""",
r"""the deletable object
itself is larger (Tab.~\ref{tab:ledger-kstar})."""), ('style',
r"""The deletable
object itself is larger, and its memory footprint (Tab.~\ref{tab:ledger-kstar}) is printed,
never matched.""",
r"""The deletable
object itself is larger (Tab.~\ref{tab:ledger-kstar}).""")],
'"is printed, never matched" deleted (contrast and self-narration); the table pointer stays')
add('R2', 'T5', 'Results', [('both',
r"""\paragraph{What the counts look like.}""",
r"""\paragraph{The counts as images.}""")],
'heading: clause -> noun phrase')
add('R3', 'T2', 'Results', [('both',
r"""drawn by a fixed-seed rule, stated in the captions, never by appearance.""",
r"""drawn by a fixed-seed rule, stated in the captions.""")],
'"X, never Y" -> X')
add('R4', 'T5 T3', 'Results', [('both',
r"""\paragraph{The inference note, verbatim in substance.}""",
r"""\paragraph{The inference note.}""")],
'heading: apposition "verbatim in substance" deleted')
add('R5', 'T2', 'Results', [('both',
r"""probe the same combinations and differ in fit-pool coverage, not composition depth.}""",
r"""probe the same combinations and differ in fit-pool coverage.}""")],
'"X, not Y" -> X')
add('R6', 'T3 T7a', 'Results', [('both',
r"""(Tab.~\ref{tab:ladder}). Two caveats travel with % SRC: w20-shapes-data-matched/FINDINGS.md
the baseline rows. The data-matched set is not proven converged everywhere: a ten-times spot check""",
r"""(Tab.~\ref{tab:ladder}). % SRC: w20-shapes-data-matched/FINDINGS.md
The data-matched set is not proven converged everywhere. A ten-times spot check""")],
'announcement "Two caveats travel with the baseline rows." deleted; split: "The data-matched set is not proven converged everywhere." (8 words) follows the 24-word interval sentence')
add('R7', 'T3', 'Results', [('both',
r"""moves the LSTM by about one tuple at one rung and seed (Tab.~\ref{tab:ladder}), printed and not carried. The earlier set's""",
r"""moves the LSTM by about one tuple at one rung and seed (Tab.~\ref{tab:ladder}). The earlier set's""")],
'announcement "printed and not carried" deleted (the table note says "not carried")')
add('R8', 'T2', 'Results', [('both',
r"""Its TEST cell is therefore
\emph{derived, not measured}: it is""",
r"""Its TEST cell is therefore
\emph{derived}: it is""")],
'"X, not Y" -> X')
add('R9', 'T2', 'Results', [('both',
r"""That TEST cell is \emph{derived, not
measured} (a new TEST""",
r"""That TEST cell is \emph{derived} (a new TEST""")],
'"X, not Y" -> X')
add('R10', 'T3', 'Results', [('both',
r"""\textsc{no rise to decompose on val}, and we report it as
that; validation sits near its ceiling. The inference note travels with % SRC: w15-coverage-control/FINDINGS.md""",
r"""\textsc{no rise to decompose on val}; validation sits near its ceiling. % SRC: w15-coverage-control/FINDINGS.md""")],
'announcements "and we report it as that" and "The inference note travels with" deleted')
add('R11', 'T3', 'Results', [('both',
r"""Tab.~\ref{tab:ladder}: \emph{``We beat the naive fair baselines at every $n$; the n-CLU's own 2-way
$\to$ 6-way rise is not to be inferred, because rungs 3--6 share combinations and differ in coverage.''}
""",
r"""""")],
'the repeated quotation of the inference note (it opens the previous paragraph) deleted with its "travels with" announcement')
add('R12', 'T7a', 'Results', [('both',
r"""Check (d)
holds as well: the downdated object""",
r"""Check (d)
holds as well. The downdated object""")],
'split: "Check (d) holds as well." (5 words) follows the 68-word sentence')
add('R13', 'T5', 'Results', [('both',
r"""\paragraph{Deleting a conjunction changes what the book knows about the shared value.}""",
r"""\paragraph{Conjunction deletion and the shared value.}""")],
'heading: sentence -> noun phrase')
add('R14', 'T7a', 'Results', [('both',
r"""left the survivors unchanged, because it did
not delete from the book at all.""",
r"""left the survivors unchanged. It did
not delete from the book.""")],
'split at "because", "at all" deleted: "It did not delete from the book." (7 words)')
add('R15', 'T2 T7a', 'Results', [('v2',
r"""The n-CLU has changed: without the address-book carry, the read""",
r"""The n-CLU has changed. Without the address-book carry, the read"""), ('v2',
r"""so the basis change is generic, not targeted.""",
r"""so the basis change is generic."""), ('style',
r"""the basis change is therefore generic, not targeted.""",
r"""the basis change is therefore generic.""")],
'"X, not Y" -> X; v2 split: "The n-CLU has changed." (4 words) follows the 19-word opening sentence (the style twin already splits there)')
add('R16', 'T7a', 'Results', [('both',
r"""and it is a flag: a request for a
holed value is refused by construction, so refusal is measurement (c) of Tab.~\ref{tab:del}, a
display row, and symbolic refusal is not claimed.""",
r"""and it is a flag: a request for a
holed value is refused by construction, so refusal is measurement (c) of Tab.~\ref{tab:del}, a
display row. Symbolic refusal is not claimed.""")],
'split: "Symbolic refusal is not claimed." (4 words)')
add('R17', 'T7a', 'Results', [('both',
r"""Nor is the
n-CLU's route faster end to end: its downdate takes seconds""",
r"""Nor is the
n-CLU's route faster. Its downdate takes seconds""")],
'"end to end" deleted and split: "Nor is the n-CLU\'s route faster." (6 words) follows the 76-word sentence')
add('R18', 'T3', 'Results', [('both',
r"""Two caveats travel with these rows: the baselines
retrain at their selected schedule, so the MLP (seeds 0 and 1) and continuous-MLP convergence flags
of Tab.~\ref{tab:t7} travel with them; and approximate unlearning""",
r"""The baselines
retrain at their selected schedule, so the MLP (seeds 0 and 1) and continuous-MLP convergence flags
of Tab.~\ref{tab:t7} apply to them. Approximate unlearning""")],
'announcements "Two caveats travel with these rows" deleted and "travel with them" -> "apply to them"; split at the semicolon')
add('R19', 'T2', 'Results', [('v2',
r"""The comparator's job is to measure what the alternative to a hole requires, not to lose.""",
r"""The comparator's job is to measure what the alternative to a hole requires."""), ('style',
r"""The comparator serves to measure what the alternative to a hole requires, not to lose.""",
r"""The comparator serves to measure what the alternative to a hole requires.""")],
'"X, not Y" -> X')
add('R20', 'T2', 'Results', [('v2',
r"""\noindent The operation that rewrites only the bytes it names and leaves behaviour unchanged is the
hole, not deletion in general.""",
r"""\noindent The operation that rewrites only the bytes it names and leaves behaviour unchanged is the
hole."""), ('style',
r"""\noindent The hole, not deletion in general, is the operation""",
r"""\noindent The hole is the operation""")],
'"X, not Y" -> X')
add('R21', 'T2', 'Results', [('both',
r"""The claim is a match with a refit without the rows, never erasure without loss.""",
r"""The claim is a match with a refit without the rows.""")],
'"X, never Y" -> X')
add('R22', 'T5', 'Results', [('both',
r"""\paragraph{Composition-only n-CLU (composes, cannot forget).}""",
r"""\paragraph{Composition-only n-CLU.}""")],
'heading: parenthetical deleted')
add('R23', 'T5', 'Results', [('both',
r"""\paragraph{Item memory (forgets, loses composition).}""",
r"""\paragraph{Item memory.}""")],
'heading: parenthetical deleted')
add('R24', 'T7a', 'Results', [('v2',
r"""deletes by construction and
composes poorly: on row 4 it sits below the learned""",
r"""deletes by construction and
composes poorly. On row 4 it sits below the learned"""), ('v2',
r"""loses, because it is a ceiling and not an arm.""",
r"""loses. It is a ceiling and not an arm."""), ('style',
r"""loses, because this n-CLU is a ceiling and not an arm.""",
r"""loses. This n-CLU is a ceiling and not an arm.""")],
'the one-sentence paragraph split at its colon and at "because": "It is a ceiling and not an arm." (8 words)')
add('R25', 'T5', 'Results', [('both',
r"""\paragraph{LACE and PoE, at their own settings.}""",
r"""\paragraph{LACE and PoE.}""")],
'heading: apposition deleted')
add('R26', 'T2 T3', 'Results', [('both',
r"""the level of our landscape controls; we report them for completeness, not as evidence, in
Tab.~\ref{tab:lace-poe} (App.~\ref{app:t7more}).""",
r"""the level of our landscape controls
(Tab.~\ref{tab:lace-poe}, App.~\ref{app:t7more}).""")],
'"we report them for completeness, not as evidence" deleted (self-narration and contrast)')
add('R27', 'T5', 'Results', [('both',
r"""\paragraph{The generation table, on the $\Kstar$ single design.}""",
r"""\paragraph{The generation table.}""")],
'heading: apposition deleted')
add('R28', 'T2', 'Results', [('both',
r"""The generation table is a second Shapes3D instrument, never merged with Tab.~\ref{tab:t7}: a fixed set of""",
r"""The generation table is a second Shapes3D instrument: a fixed set of""")],
'"X, never Y" -> X')
add('R29', 'T7a', 'Results', [('both',
r"""though its mean is positive, and the""",
r"""though its mean is positive. The"""), ('both',
r"""declared protocol's falsification clause fires: the claim that""",
r"""declared protocol's falsification clause fires. The claim that""")],
'split twice: "The declared protocol\'s falsification clause fires." (6 words) follows the 20-word S2 sentence')
add('R30', 'T5', 'Results', [('both',
r"""\paragraph{CMAZE-T with symmetric inputs: six seeds.}""",
r"""\paragraph{CMAZE-T with symmetric inputs.}""")],
'heading: colon deleted')
add('R31', 'T6', 'Results', [('both',
r"""to every arm the n-CLU is behind the best matched baseline on all three TEST seeds""",
r"""to every arm it is behind the best matched baseline on all three TEST seeds""")],
'n-CLU density: pronoun (the subject of the previous sentence is the n-CLU)')
add('R32', 'T5', 'Results', [('both',
r"""\paragraph{A relational request coding for the n-CLU is an input-representation result.}""",
r"""\paragraph{Relational request coding.}""")],
'heading: sentence -> noun phrase')
add('R33', 'T6', 'Results', [('both',
r"""(n-CLU-REL), lifts the n-CLU on every seed""",
r"""(n-CLU-REL), lifts it on every seed""")],
'n-CLU density: pronoun (the sentence\'s subject is the change to the n-CLU\'s coding)')
add('R34', 'T2', 'Results', [('both',
r"""relational request coding carried through the n-CLU's closed-form head, not the n-CLU.""",
r"""relational request coding carried through the n-CLU's closed-form head.""")],
'"X, not Y" -> X')
add('R35', 'T5', 'Results', [('both',
r"""\paragraph{CMAZE-T: what is not claimed.}""",
r"""\paragraph{Claims not made on CMAZE-T.}""")],
'heading: colon-as-sentence -> noun phrase')
add('R36', 'T7a', 'Results', [('both',
r"""n-CLU is below the best matched baseline, so nothing is claimed on either row.""",
r"""n-CLU is below the best matched baseline. Nothing is claimed on either row.""")],
'split at "so": "Nothing is claimed on either row." (6 words)')
add('R37', 'T5', 'Results', [('both',
r"""\paragraph{Reading the CMAZE-T panel figures.}""",
r"""\paragraph{The CMAZE-T panel figures.}""")],
'heading: gerund clause -> noun phrase')
add('R38', 'T7a', 'Results', [('both',
r"""the pocket after the move, and a picture need
not show all four.""",
r"""the pocket after the move. A picture need
not show all four.""")],
'split: "A picture need not show all four." (7 words)')

# ------------------------------------------------------------------ DISCUSSION (v2's limitations list is no longer
# Head-held once B3 gives v2 the style twin's Discussion signpost; the list is edited in lock-step)
add('D1', 'T7a', 'Discussion', [('both',
r"""with its minimum 3.9942 times the reference at $K=24$, so
  $\Kstar$ is not the point of smallest footprint; and on the one configuration""",
r"""with its minimum 3.9942 times the reference at $K=24$.
  $\Kstar$ is not the point of smallest footprint. On the one configuration""")],
'split twice ("so", "; and" deleted): "K* is not the point of smallest footprint." (8 words) follows the 40-word sentence')
add('D2', 'T2', 'Discussion', [('both',
r"""(the validation sweep labels itself a smoke, not a claim)""",
r"""(the validation sweep labels itself a smoke)""")],
'"X, not Y" -> X')
add('D3', 'T2', 'Discussion', [('both',
r"""we judge the risk low, not zero.""",
r"""we judge the risk low.""")],
'"X, not Y" -> X')
add('D4', 'T5', 'Discussion', [('both',
r"""\paragraph{Toward a CMAZE suite.} CMAZE-T is the first instrument""",
r"""CMAZE-T is the first instrument""")],
'run-in title deleted: a prepositional phrase that repeats its subsection heading')
add('D5', 'T1', 'Discussion', [('v2',
r"""the same content, the same wells,
the same read, with the settling loop's""",
r"""the same content, wells and read, with the settling loop's""")],
'rhetorical anaphoric triad "the same X, the same Y, the same Z" -> one list (the style twin already reads "the same content, wells and read")')
add('D6', 'T7a', 'Discussion', [('both',
r"""We asserted this in code: the zero-step arm was required""",
r"""We asserted this in code. The zero-step arm was required""")],
'split: "We asserted this in code." (5 words) follows the 44-word identity sentence')
add('D7', 'T2 T7a', 'Discussion', [('both',
r"""environment's dynamics; here the dynamics are the read, not a model of the data.)""",
r"""environment's dynamics. Here the dynamics are the read.)""")],
'"X, not Y" -> X and split: "Here the dynamics are the read." (6 words)')
add('D8', 'T5', 'Discussion', [('both',
r"""\paragraph{Two laws measured on this study's own n-CLU (\textsc{confirmed [this study]}).}""",
r"""\paragraph{Two laws measured on this study's own n-CLU.}""")],
'heading: parenthetical deleted')
add('D9', 'T2', 'Discussion', [('v2',
r"""prediction is a \emph{time}, not a score: with nothing fitted""",
r"""prediction is a \emph{time}: with nothing fitted"""), ('style',
r"""prediction is a \emph{time}, not a score. With nothing fitted""",
r"""prediction is a \emph{time}. With nothing fitted""")],
'"X, not Y" -> X (the Summary\'s "a law about the n-CLU, not a lever on the task" is Discussion\'s one)')
add('D10', 'T1', 'Discussion', [('both',
r"""They occupy no row of the evaluation, they
beat no model, and we make no capability claim from them}""",
r"""We make no capability claim from them}""")],
'rhetorical triad of one claim ("occupy no row, beat no model, make no claim") -> the claim')
add('D11', 'T6', 'Discussion', [('both',
r"""The naive unit provides deletion by subtraction""",
r"""The n-CLU provides deletion by subtraction""")],
'"naive unit" after its first mention -> n-CLU')
add('D12', 'T6', 'Discussion', [('style',
r"""The leaving note states what the naive unit does not provide; the follow-up is therefore a full evaluation""",
r"""The follow-up is a full evaluation""")],
'style only: coinage "leaving note" (and "naive unit") dropped with the clause that announced it')

# ------------------------------------------------------------------ APPENDICES
add('P1', 'T3', 'App-preface', [('v2',
r"""Appendix-maximalism is in force: every row, corollary, negative and control of the study that
is not in the main text is here, and nothing is omitted.""",
r"""Every row, corollary, negative and control of the study that
is not in the main text is here."""), ('style',
r"""Appendix-maximalism applies: every row, corollary, negative and control of the study that is
not in the main text appears here, and nothing is omitted.""",
r"""Every row, corollary, negative and control of the study that is
not in the main text appears here.""")],
'announcements "Appendix-maximalism is in force" and "nothing is omitted" deleted')
add('P2', 'T7a', 'App A', [('v2',
r"""Numbers are quoted as the record prints them; none is re-derived.""",
r"""Numbers are quoted as the record prints them. None is re-derived."""), ('style',
r"""Numbers are quoted as the record prints them, and none is
re-derived.""",
r"""Numbers are quoted as the record prints them. None is
re-derived.""")],
'split: "Numbers are quoted as the record prints them." (8 words) follows the 29-word sentence')
add('P3', 'T7a', 'App A', [('v2',
r"""They differ in partition and
in convention; the convention alone""",
r"""They differ in partition and
in convention. The convention alone"""), ('style',
r"""They differ in partition and in convention; the convention alone""",
r"""They differ in partition and in convention. The convention alone""")],
'split: "They differ in partition and in convention." (7 words)')
add('P4', 'T7a', 'App B', [('both',
r"""parameters only; no baseline matched to the memory footprint is printed.""",
r"""parameters only. No baseline matched to the memory footprint is printed.""")],
'split: "Baselines are matched on read-path parameters only." (7 words)')
add('P5', 'T7a', 'App B', [('both',
r"""(Tab.~\ref{tab:lace-poe}), 3 seeds; no
learned-baseline column was replaced.""",
r"""(Tab.~\ref{tab:lace-poe}), 3 seeds. No
learned-baseline column was replaced.""")],
'split: "No learned-baseline column was replaced." (5 words)')
add('P6', 'T7a', 'App C', [('both',
r"""above the 1.0 bar, so % SRC: w20-shapes-data-matched/FINDINGS.md
its $n=4$ and $n=5$ rows are not claimable.""",
r"""above the 1.0 bar. % SRC: w20-shapes-data-matched/FINDINGS.md
Its $n=4$ and $n=5$ rows are not claimable.""")],
'split at "so": "Its n=4 and n=5 rows are not claimable." (8 words)')
add('P7', 'T7a', 'App D', [('v2',
r"""0.4375--0.9844 on seed 2, and the controls still collapse, so rank/width is a warning and only the
control decides.""",
r"""0.4375--0.9844 on seed 2, and the controls still collapse. Rank/width is a warning. Only the
control decides."""), ('style',
r"""0.4375--0.9844 on seed 2, yet the controls still collapse; rank/width is therefore a warning, and
only the control decides.""",
r"""0.4375--0.9844 on seed 2, yet the controls still collapse. Rank/width is therefore a warning.
Only the control decides.""")],
'split twice: "Rank/width is a warning." (4 words) follows the 35-word sentence')
add('P8', 'T3 T2', 'App E', [('both',
r"""is declared, not netted.""",
r"""is declared.""")],
'"declared, not netted" -> "declared" (self-narration and contrast)')
add('P9', 'T2', 'App E', [('v2',
r"""its record states that it is neither a test
run nor a claim.""",
r"""its record states that it is not a test
run or a claim."""), ('style',
r"""neither a test run nor a claim.""",
r"""not a test run or a claim.""")],
'"neither ... nor" -> plain negation (App E keeps "a property of the content, not of the downdate")')
add('P10', 'T7a', 'App E', [('v2',
r"""content, not of the downdate; the top-level ``all assertions passed'' excludes it.""",
r"""content, not of the downdate. The top-level ``all assertions passed'' excludes it."""), ('style',
r"""not of the downdate, and the top-level ``all assertions
passed'' excludes it.""",
r"""not of the downdate. The top-level ``all assertions
passed'' excludes it.""")],
'split: "The top-level ``all assertions passed\'\' excludes it." (7 words)')
add('P11', 'T7a', 'App E', [('v2',
r"""before the re-run and not moved; the numerator is the failing leg's own worst.""",
r"""before the re-run and not moved. The numerator is the failing leg's own worst."""), ('style',
r"""declared before the re-run and not moved; its numerator is the failing leg's own worst.""",
r"""declared before the re-run and not moved. Its numerator is the failing leg's own worst.""")],
'split: "The numerator is the failing leg\'s own worst." (8 words)')
add('P12', 'T7a', 'App F', [('v2',
r"""3.4 times the n-CLU pin, while every other rung carries 0.""",
r"""3.4 times the n-CLU pin. Every other rung carries 0."""), ('style',
r"""3.4 times the n-CLU pin, whereas every other rung carries 0.""",
r"""3.4 times the n-CLU pin. Every other rung carries 0.""")],
'split at "while"/"whereas": "Every other rung carries 0." (5 words)')
add('P13', 'T3 T2', 'App H', [('v2',
r"""aggregate / exact-match, per seed, never
pooled: frozen""",
r"""aggregate / exact-match, per seed:
frozen"""), ('style',
r"""(aggregate / exact-match, per seed,
never pooled) are: frozen""",
r"""(aggregate / exact-match, per seed) are: frozen""")],
'announcement "never pooled" deleted')
add('P14', 'T3 T2', 'App H', [('v2',
r"""The anchor arm's distance from the frozen encoder
(reported, never tuned): feature relative""",
r"""The anchor arm's distance from the frozen encoder:
feature relative"""), ('style',
r"""(reported, never tuned) is a feature relative""",
r"""is a feature relative""")],
'announcement "(reported, never tuned)" deleted')
add('P15', 'T7a', 'App I', [('v2',
r"""images) is a separate table: equal compute passes on all three seeds; its equal-seconds control is""",
r"""images) is a separate table. Equal compute passes on all three seeds. Its equal-seconds control is""")],
'v2 split twice: "Equal compute passes on all three seeds." (6 words) follows the 19-word sentence (the style twin already splits there)')
add('P16', 'T7a', 'App J', [('v2',
r"""(C3) The state is carried in span coordinates; the same
energy""",
r"""(C3) The state is carried in span coordinates. The same
energy"""), ('style',
r"""(C3) The state is carried in span
coordinates; the same energy""",
r"""(C3) The state is carried in span
coordinates. The same energy""")],
'split: "(C3) The state is carried in span coordinates." (8 words)')
add('P17', 'T7a', 'App J', [('v2',
r"""524,288 B on every seed; a finite-difference Hessian check reads""",
r"""524,288 B on every seed. A finite-difference Hessian check reads"""), ('style',
r"""524,288 B on every seed; and a finite-difference Hessian check reads""",
r"""524,288 B on every seed. A finite-difference Hessian check reads""")],
'split: the Hessian sentence (8 words) follows the 40-word sentence')
add('P18', 'T7a', 'App K', [('v2',
r"""AUC 0.457 to 0.559); what breaks the
read is motion inside it.""",
r"""AUC 0.457 to 0.559). What breaks the
read is motion inside it."""), ('style',
r"""AUC 0.457 to 0.559); what breaks
the read is motion inside it.""",
r"""AUC 0.457 to 0.559). What breaks
the read is motion inside it.""")],
'split: "What breaks the read is motion inside it." (8 words)')
add('P19', 'T7a', 'App K', [('v2',
r"""a byte multiple of 59,392; no claim is made on those
rows.""",
r"""a byte multiple of 59,392. No claim is made on those
rows."""), ('style',
r"""a byte multiple of 59,392; we make no claim on those
rows.""",
r"""a byte multiple of 59,392. We make no claim on those
rows.""")],
'split: "No claim is made on those rows." (7 words)')
add('P20', 'T3', 'App L', [('v2',
r"""\textsc{refuted} and \textsc{untested} as stated. Status discipline is part of the section.""",
r"""\textsc{refuted} and \textsc{untested} as stated."""), ('style',
r"""
Status discipline is part of the section.
""",
r"""
""")],
'announcement "Status discipline is part of the section." deleted')
add('P21', 'T7a', 'App L', [('both',
r"""priority- or attribute-based eviction; recency-based eviction is excluded as""",
r"""priority- or attribute-based eviction. Recency-based eviction is excluded as""")],
'split inside the parenthesis: "Recency-based eviction is excluded as intrinsically history-dependent." (7 words)')
add('P22', 'T7a', 'App L', [('both',
r"""derived object inverts it: the retention floor is curvature-independent.""",
r"""derived object inverts it. The retention floor is curvature-independent.""")],
'split: "The retention floor is curvature-independent." (5 words)')
add('P23', 'T3', 'App L', [('both',
r"""each with the rider that must travel with it.""",
r"""each with its rider.""")],
'announcement "that must travel with it" deleted')
add('P24', 'T3 T7a', 'App O', [('v2',
r"""Light adaptation is optional, and this is what it yields.""",
r"""Light adaptation is optional."""), ('style',
r"""Light adaptation is optional; this subsection reports what it yields.""",
r"""Light adaptation is optional.""")],
'signpost "and this is what it yields" deleted: "Light adaptation is optional." (4 words) follows the 19-word sentence')
add('P25', 'T3 T2', 'App O', [('both',
r"""aggregate, per seed, never pooled).""",
r"""aggregate, per seed).""")],
'announcement "never pooled" deleted')
add('P26', 'T7a', 'App O', [('both',
r"""This is a finding about the encoder, and the claim
that the n-CLU shapes the representation is not made.""",
r"""This is a finding about the encoder. The claim
that the n-CLU shapes the representation is not made.""")],
'split: "This is a finding about the encoder." (7 words) follows the 59-word sentence')
add('P27', 'T2', 'App O', [('both',
r"""(Tab.~\ref{tab:bard-gpu}, a block of its own, never merged with the
shortened-decoder leg of Tab.~\ref{tab:encloop})""",
r"""(Tab.~\ref{tab:bard-gpu})""")],
'"a block of its own, never merged with ..." deleted (contrast; the table caption states it)')
add('P28', 'T7a', 'App O', [('both',
r"""of 6 on the longer one); the ``unfreezing hurts'' half does not survive.""",
r"""of 6 on the longer one). The ``unfreezing hurts'' half does not survive.""")],
'split: "The ``unfreezing hurts\'\' half does not survive." (7 words)')
add('P29', 'T2', 'App P', [('both',
r"""whose prediction is a
\emph{time}, not a score.""",
r"""whose prediction is a
\emph{time}.""")],
'"X, not Y" -> X (App P keeps "on this memory, not the law")')
add('P30', 'T3', 'App P', [('both',
r"""its maximum over $\gamma$, is
printed and not moved: 59.1158""",
r"""its maximum over $\gamma$, is
not moved: 59.1158""")],
'announcement "printed and" deleted')
add('P31', 'T1', 'App P', [('both',
r"""It occupies no row of the evaluation, it
beats no model, and we make no capability claim from it}""",
r"""We make no capability claim from it}""")],
'rhetorical triad of one claim -> the claim (as D10)')
add('P32', 'T7a', 'App Q', [('both',
r"""the same post-deletion requests is not run; \texttt{pca\_tensor} is not run at $\Kstar$""",
r"""the same post-deletion requests is not run. \texttt{pca\_tensor} is not run at $\Kstar$""")],
'split: "pca_tensor is not run at K* (App. S)." (8 words)')
add('P33', 'T7a', 'App R', [('both',
r"""permanently blocked, not pending: the run behind""",
r"""permanently blocked, not pending. The run behind""")],
'split: "They are permanently blocked, not pending." (6 words; App R\'s one contrast) follows the 32-word sentence')
add('P34', 'T3', 'App N', [('both',
r"""The harness ships three warnings, all printed here: a composition-blind""",
r"""The harness ships three warnings: a composition-blind""")],
'announcement "all printed here" deleted')
add('P35', 'T2', 'App L', [('style',
r"""have neither a derivation nor an
impossibility argument""",
r"""have no derivation and no
impossibility argument""")],
'style only: "neither ... nor" -> plain negation (App L keeps "parameter separation, not block structure")')
add('R39', 'T6', 'Results', [('both',
r"""the n-CLU's own loss on that deletion, about six tuples""",
r"""its own loss on that deletion, about six tuples""")],
'n-CLU density: pronoun (the clause\'s subject "it" is the n-CLU)')
