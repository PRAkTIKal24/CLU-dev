# EDITS — every edit of PW-2 item 1, before → after, tagged with its tell

Source: `pw2_draft_tells/edits.py` (applied by `apply.py`; each `old` occurs exactly once in its draft). Scope `both` = the same bytes in both drafts; otherwise the style twin's own wording at the same place.
Tells: T1 triad, T2 contrast, T3 self-narration, T5 heading, T6 coinage, T7a split for a short sentence, T7b connective opener, T7c "This section"; B = borrowing from the style twin.
Words added by each edit are listed (case-insensitive multiset, from `audit.json`); a capital made by a split is not an addition.

### B1 — B — Abstract
the abstract's first sentence, verbatim from the style twin (line layout copied so the Head-held sentence is byte-identical to the style twin's)
- **v2**: `A world model that is to be trusted with requests it has never seen needs three things at once: its outputs must be \emph{valid} under the task's judge, it must \emph{compose} held-out combinations of factors from factors it has seen, and it must be able to \emph{forget} specific training rows. Compositional`
  → `A world model trusted with requests it has never seen needs three things at once: its outputs are \emph{valid} under the task's judge, it \emph{composes} held-out combinations of factors from factors it has seen, and it can \emph{forget} specific training rows. Compositional`
- words added: DRAFT-v2 adds are×1, composes×1, can×1

### B2 — B — Methods
the Sec. 2 (Methods) signpost, verbatim from the style twin, at the same place (second paragraph of Methods)
- **v2**: `as a virtue of the design itself. \paragraph{The n-CLU.}`
  → `as a virtue of the design itself. A claim about any of the three properties is only as precise as the definitions of the object, its judge and its memory footprint; we therefore fix these here, before any result. \paragraph{The n-CLU.}`
- words added: DRAFT-v2 adds as×2, a×1, of×2, the×3, claim×1, about×1, any×2, three×1, properties×1, is×1, only×1, precise×1, definitions×1, object×1, its×2, judge×1, and×1, memory×1, footprint×1, we×1, therefore×1, fix×1, these×1, here×1, before×1, result×1

### B3 — B — Discussion
the Sec. 5 (Discussion) signpost, verbatim from the style twin; it becomes v2's Head-held Discussion first paragraph
- **v2**: `\section{Discussion} \label{sec:limits} % ======================================================================================= \subsection{Limitations}`
  → `\section{Discussion} \label{sec:limits} % ======================================================================================= We now state what the naive unit does not do, the one measured clue, and the theory we carry. \subsection{Limitations}`
- words added: DRAFT-v2 adds we×2, now×1, state×1, what×1, the×3, naive×1, unit×1, does×1, not×1, do×1, one×1, measured×1, clue×1, and×1, theory×1, carry×1

### B4 — B — Introduction
"we claim neither ... nor", verbatim from the style twin's Introduction first paragraph (Head-held)
- **v2**: `We do not claim the best compositional-generation system or the best unlearning system.`
  → `We claim neither the best compositional-generation system nor the best unlearning system.`
- words added: DRAFT-v2 adds neither×1, nor×1

### A1 — T3 T7a — Abstract
self-narration "which we print" deleted; the colon becomes a full stop, so "Deletion has limits." (4 words) follows the 30-word sentence before it
- **both**: `object. Deletion has limits, which we print: the book's retained statistic is`
  → `object. Deletion has limits. The book's retained statistic is`

### A2 — T6 — Abstract
"naive unit" after its first mention -> n-CLU
- **v2**: `We close with what the naive unit cannot do`
  → `We close with what the n-CLU cannot do`
- **style**: `We conclude with what the naive unit cannot do`
  → `We conclude with what the n-CLU cannot do`
- words added: DRAFT-v2 adds n-clu×1; DRAFT-v2-style adds n-clu×1

### I1 — T7a — Introduction
split at the semicolon: "GAN unlearning by substitution is approximate by construction." (8 words)
- **v2**: `different task from deletion; GAN unlearning by substitution`
  → `different task from deletion. GAN unlearning by substitution`
- **style**: `from deletion; and GAN unlearning by substitution`
  → `from deletion. GAN unlearning by substitution`

### I2 — T2 T7a — Introduction
contrast restated as its positive half; v2: "These are the screen's counts." (5 words) follows the 21-word zero-works sentence
- **v2**: `These are the screen's counts, not a claim that no such model exists: abstract-level screening cannot see`
  → `These are the screen's counts. Abstract-level screening cannot see`
- **style**: `These counts describe the screen; they do not claim that no such model exists. Abstract-level`
  → `These counts describe the screen. Abstract-level`

### I3 — T3 — Introduction
announcement "one sentence each" deleted
- **v2**: `Three families of systems built on top of units are out of scope, one sentence each.`
  → `Three families of systems built on top of units are out of scope.`
- **style**: `Three families of systems built on top of units are out of scope, and we distinguish each in one sentence.`
  → `Three families of systems built on top of units are out of scope.`

### I4 — T2 — Introduction
"X, not Y" -> X
- **v2**: `so what is composed is models of data sources, not held-out combinations of factor values within one unit.`
  → `so what is composed is models of data sources.`
- **style**: `what they compose is models of data sources, not held-out combinations of factor values within one unit.`
  → `what they compose is models of data sources.`

### I5 — T3 — Introduction
announcement "worth a sentence" deleted
- **v2**: `One distinction is ours and worth a sentence.`
  → `One distinction is ours.`
- **style**: `One distinction is ours and merits a sentence.`
  → `One distinction is ours.`

### I6 — T7a — Introduction
", for the opposite reason" deleted and split at the colon: "Our route matches the refit." (5 words) follows the 29-word decremental-PCA sentence; the kept contrast is the Introduction's one
- **v2**: `Our route matches the refit, for the opposite reason: \emph{downdate the statistic, not the truncated factors}.`
  → `Our route matches the refit. \emph{Downdate the statistic, not the truncated factors}.`
- **style**: `Our route matches the refit, for the opposite reason: \emph{downdate the statistic, not the truncated factors}.`
  → `Our route matches the refit. \emph{Downdate the statistic, not the truncated factors}.`

### I7 — T7b — Introduction
style only (its itemize makes this a paragraph opener): connective "However" deleted
- **style**: `However, the three pull against each other:`
  → `The three pull against each other:`

### I8 — T3 — Introduction
announcement "with each baseline's convergence printed beside it" deleted
- **both**: `fitted on its own rows, with each baseline's convergence printed beside it: by about seven tuples`
  → `fitted on its own rows: by about seven tuples`

### I9 — T7a — Introduction
split at the semicolon and "and not a depth axis" deleted: "For n>=3 the rungs are a training-data axis." (8 words)
- **both**: `(Tab.~\ref{tab:ladder}, Sec.~\ref{sec:ladder}); for $n\ge3$ the rungs are a training-data axis and not a depth axis.`
  → `(Tab.~\ref{tab:ladder}, Sec.~\ref{sec:ladder}). For $n\ge3$ the rungs are a training-data axis.`

### I10 — T3 — Introduction
announcement "with its limitations printed" deleted
- **both**: `\item \textbf{Deletion on the n-CLU and on its address book, with its limitations printed.} After`
  → `\item \textbf{Deletion on the n-CLU and on its address book.} After`

### I11 — T7a — Introduction
split at ", but": "The deletion reads no training row." (6 words) follows the 28-word sentence
- **both**: `The deletion reads no training row, but the book's retained statistic`
  → `The deletion reads no training row. The book's retained statistic`

### I12 — T2 — Introduction
"neither a size nor a speed claim" deleted (the two facts it summarises stand in the sentence)
- **both**: `(Tabs.~\ref{tab:del} and~\ref{tab:rivals-del}), so this is neither a size nor a speed claim. Refusal`
  → `(Tabs.~\ref{tab:del} and~\ref{tab:rivals-del}). Refusal`

### I13 — T2 T7a — Introduction
"X, not Y" -> X, and split: "It is an input-representation result (Sec. 4.8)." (7 words)
- **both**: `same columns close most of that gap, so it is an input-representation result, not an n-CLU result`
  → `same columns close most of that gap. It is an input-representation result`

### I14 — T6 — Introduction
coinage "leaving note" dropped
- **v2**: `\item \textbf{Two ablations and a leaving note.}`
  → `\item \textbf{Two ablations.}`
- **style**: `\item \emph{Two ablations and a leaving note.}`
  → `\item \emph{Two ablations.}`

### I15 — T6 — Introduction
"naive unit" after its first mention -> n-CLU
- **v2**: `The naive unit's limitations`
  → `The n-CLU's limitations`
- **style**: `The limitations of the naive unit,`
  → `The limitations of the n-CLU,`
- words added: DRAFT-v2 adds n-clu's×1; DRAFT-v2-style adds n-clu×1

### I16 — T6 T3 — Introduction
coinage "Thickness" and the announcement "stated in the introduction" dropped with the run-in label
- **both**: `\noindent\textbf{Thickness, stated in the introduction.} The positive`
  → `\noindent The positive`

### I17 — T7a — Introduction
split: "No combination of theirs can be judged." (7 words)
- **both**: `no external classifier, so no combination of theirs can be judged.`
  → `no external classifier. No combination of theirs can be judged.`

### M1 — T7a — Methods
the borrowed signpost (both drafts) split at its semicolon: "We therefore fix these here, before any result." (8 words)
- **both**: `its judge and its memory footprint; we therefore fix these here, before any result.`
  → `its judge and its memory footprint. We therefore fix these here, before any result.`

### M2 — T7a — Methods
split: "The read is request-only." (4 words) follows the 31-word address-book sentence
- **both**: `read is request-only: no feature tensor enters it`
  → `read is request-only. No feature tensor enters it`

### M3 — T5 T3 — Methods
heading: comma apposition and "never netted" deleted
- **both**: `\paragraph{The memory footprint, never netted.}`
  → `\paragraph{The memory footprint.}`

### M4 — T3 — Methods
announcement "We never net them into one number" deleted
- **v2**: `We never net them into one number, and every byte figure names its convention`
  → `Every byte figure names its convention`
- **style**: `We never net them into a single number, and every byte figure names its convention`
  → `Every byte figure names its convention`

### M5 — T7a — Methods
"deletable" (and style "that") deleted: the closing sentence is 8 words after the 30-word reference sentence
- **v2**: `It is not a limit the deletable object meets.`
  → `It is not a limit the object meets.`
- **style**: `It is not a limit that the deletable object meets.`
  → `It is not a limit the object meets.`

### M6 — T5 — Methods
heading: clause -> noun phrase (plain restatement)
- **both**: `\paragraph{Why a hold-out draw can remove a whole colour relation.}`
  → `\paragraph{Colour relations under the hold-out draw.}`
- words added: DRAFT-v2 adds relations×1, under×1, the×1; DRAFT-v2-style adds relations×1, under×1, the×1

### M8 — T3 — Methods
announcement "Seed count is a column of every table." deleted (the brief's "every row prints its seed count")
- **both**: `criterion S1 or criterion S2). Seed count is a column of every table.`
  → `criterion S1 or criterion S2).`

### M9 — T7a — Methods
split: "Deleting rows subtracts their contribution from the statistic." (8 words) follows the 18-word sentence
- **both**: `contribution from the statistic, and the atoms are re-eigendecomposed`
  → `contribution from the statistic. The atoms are re-eigendecomposed`

### M10 — T7a — Methods
split: "We do not claim certified (eps, delta) unlearning." (7 words)
- **v2**: `measured separately; we do not claim certified`
  → `measured separately. We do not claim certified`
- **style**: `measured separately; and we do not claim certified`
  → `measured separately. We do not claim certified`

### X1 — T5 — Experiments
heading: colon-as-sentence deleted
- **both**: `\paragraph{Baselines: naive against naive.}`
  → `\paragraph{Baselines.}`

### X2 — T2 T3 T7a — Experiments
"it is printed, never matched" deleted (contrast and self-narration)
- **v2**: `The memory footprint is not a read-path parameter; it is printed, never matched~\citep{`
  → `The memory footprint is not a read-path parameter~\citep{`
- **style**: `The memory footprint is not a read-path parameter: it is printed, never matched~\citep{`
  → `The memory footprint is not a read-path parameter~\citep{`

### X3 — T2 — Experiments
"X, never Y" -> X; the 7-word sentence follows the 51-word sentence
- **both**: `It is printed beside the data-matched set, never in its place.`
  → `It is printed beside the data-matched set.`

### X4 — T3 — Experiments
announcement "which we print rather than round away" deleted
- **both**: `landscape shuffled (Tab.~\ref{tab:ladder}), which we print rather than round away.`
  → `landscape shuffled (Tab.~\ref{tab:ladder}).`

### X5 — T7c — Experiments
no sentence opens with "This section" (plain restatement: "This section" -> "The task")
- **both**: `This section asks the n-CLU to \emph{generate}`
  → `The task asks the n-CLU to \emph{generate}`
- words added: DRAFT-v2 adds the×1, task×1; DRAFT-v2-style adds the×1, task×1

### X6 — T3 T7a — Experiments
announcement "every row prints its seed count" deleted; split: "Baselines are matched on read floats." (6 words)
- **both**: `Baselines are matched on read floats, the landscape controls and the blind columns are printed, and every row prints its seed count (Tabs.`
  → `Baselines are matched on read floats. The landscape controls and the blind columns are printed (Tabs.`

### X7 — T5 — Experiments
heading: colon-as-sentence and apposition deleted
- **both**: `\paragraph{CMAZE-T, a minor contribution: a small world-model task for composition.}`
  → `\paragraph{A small world-model task for composition.}`

### X8 — T7a — Experiments
split: "The request is five words." (5 words) follows the 21-word sentence
- **both**: `The request is five words; the held-out families`
  → `The request is five words. The held-out families`

### X9 — T2 — Experiments
"X, not Y" -> X (the "a printed floor, never a baseline" contrast is Experiments' one)
- **both**: `reasons to report it per seed, not to rank systems on it.`
  → `reasons to report it per seed.`

### R1 — T2 T3 — Results
"is printed, never matched" deleted (contrast and self-narration); the table pointer stays
- **v2**: `the deletable object itself is larger, and its memory footprint (Tab.~\ref{tab:ledger-kstar}) is printed, never matched.`
  → `the deletable object itself is larger (Tab.~\ref{tab:ledger-kstar}).`
- **style**: `The deletable object itself is larger, and its memory footprint (Tab.~\ref{tab:ledger-kstar}) is printed, never matched.`
  → `The deletable object itself is larger (Tab.~\ref{tab:ledger-kstar}).`

### R2 — T5 — Results
heading: clause -> noun phrase
- **both**: `\paragraph{What the counts look like.}`
  → `\paragraph{The counts as images.}`
- words added: DRAFT-v2 adds as×1, images×1; DRAFT-v2-style adds as×1, images×1

### R3 — T2 — Results
"X, never Y" -> X
- **both**: `drawn by a fixed-seed rule, stated in the captions, never by appearance.`
  → `drawn by a fixed-seed rule, stated in the captions.`

### R4 — T5 T3 — Results
heading: apposition "verbatim in substance" deleted
- **both**: `\paragraph{The inference note, verbatim in substance.}`
  → `\paragraph{The inference note.}`

### R5 — T2 — Results
"X, not Y" -> X
- **both**: `probe the same combinations and differ in fit-pool coverage, not composition depth.}`
  → `probe the same combinations and differ in fit-pool coverage.}`

### R6 — T3 T7a — Results
announcement "Two caveats travel with the baseline rows." deleted; split: "The data-matched set is not proven converged everywhere." (8 words) follows the 24-word interval sentence
- **both**: `(Tab.~\ref{tab:ladder}). Two caveats travel with % SRC: w20-shapes-data-matched/FINDINGS.md the baseline rows. The data-matched set is not proven converged everywhere: a ten-times spot check`
  → `(Tab.~\ref{tab:ladder}). % SRC: w20-shapes-data-matched/FINDINGS.md The data-matched set is not proven converged everywhere. A ten-times spot check`

### R7 — T3 — Results
announcement "printed and not carried" deleted (the table note says "not carried")
- **both**: `moves the LSTM by about one tuple at one rung and seed (Tab.~\ref{tab:ladder}), printed and not carried. The earlier set's`
  → `moves the LSTM by about one tuple at one rung and seed (Tab.~\ref{tab:ladder}). The earlier set's`

### R8 — T2 — Results
"X, not Y" -> X
- **both**: `Its TEST cell is therefore \emph{derived, not measured}: it is`
  → `Its TEST cell is therefore \emph{derived}: it is`

### R9 — T2 — Results
"X, not Y" -> X
- **both**: `That TEST cell is \emph{derived, not measured} (a new TEST`
  → `That TEST cell is \emph{derived} (a new TEST`

### R10 — T3 — Results
announcements "and we report it as that" and "The inference note travels with" deleted
- **both**: `\textsc{no rise to decompose on val}, and we report it as that; validation sits near its ceiling. The inference note travels with % SRC: w15-coverage-control/FINDINGS.md`
  → `\textsc{no rise to decompose on val}; validation sits near its ceiling. % SRC: w15-coverage-control/FINDINGS.md`

### R11 — T3 — Results
the repeated quotation of the inference note (it opens the previous paragraph) deleted with its "travels with" announcement
- **both**: `Tab.~\ref{tab:ladder}: \emph{''We beat the naive fair baselines at every $n$; the n-CLU's own 2-way $\to$ 6-way rise is not to be inferred, because rungs 3--6 share combinations and differ in coverage.''}`
  → `(deleted)`

### R12 — T7a — Results
split: "Check (d) holds as well." (5 words) follows the 68-word sentence
- **both**: `Check (d) holds as well: the downdated object`
  → `Check (d) holds as well. The downdated object`

### R13 — T5 — Results
heading: sentence -> noun phrase
- **both**: `\paragraph{Deleting a conjunction changes what the book knows about the shared value.}`
  → `\paragraph{Conjunction deletion and the shared value.}`
- words added: DRAFT-v2 adds deletion×1, and×1; DRAFT-v2-style adds deletion×1, and×1

### R14 — T7a — Results
split at "because", "at all" deleted: "It did not delete from the book." (7 words)
- **both**: `left the survivors unchanged, because it did not delete from the book at all.`
  → `left the survivors unchanged. It did not delete from the book.`

### R15 — T2 T7a — Results
"X, not Y" -> X; v2 split: "The n-CLU has changed." (4 words) follows the 19-word opening sentence (the style twin already splits there)
- **v2**: `The n-CLU has changed: without the address-book carry, the read`
  → `The n-CLU has changed. Without the address-book carry, the read`
- **v2**: `so the basis change is generic, not targeted.`
  → `so the basis change is generic.`
- **style**: `the basis change is therefore generic, not targeted.`
  → `the basis change is therefore generic.`

### R16 — T7a — Results
split: "Symbolic refusal is not claimed." (4 words)
- **both**: `and it is a flag: a request for a holed value is refused by construction, so refusal is measurement (c) of Tab.~\ref{tab:del}, a display row, and symbolic refusal is not claimed.`
  → `and it is a flag: a request for a holed value is refused by construction, so refusal is measurement (c) of Tab.~\ref{tab:del}, a display row. Symbolic refusal is not claimed.`

### R17 — T7a — Results
"end to end" deleted and split: "Nor is the n-CLU's route faster." (6 words) follows the 76-word sentence
- **both**: `Nor is the n-CLU's route faster end to end: its downdate takes seconds`
  → `Nor is the n-CLU's route faster. Its downdate takes seconds`

### R18 — T3 — Results
announcements "Two caveats travel with these rows" deleted and "travel with them" -> "apply to them"; split at the semicolon
- **both**: `Two caveats travel with these rows: the baselines retrain at their selected schedule, so the MLP (seeds 0 and 1) and continuous-MLP convergence flags of Tab.~\ref{tab:t7} travel with them; and approximate unlearning`
  → `The baselines retrain at their selected schedule, so the MLP (seeds 0 and 1) and continuous-MLP convergence flags of Tab.~\ref{tab:t7} apply to them. Approximate unlearning`
- words added: DRAFT-v2 adds apply×1, to×1; DRAFT-v2-style adds apply×1, to×1

### R19 — T2 — Results
"X, not Y" -> X
- **v2**: `The comparator's job is to measure what the alternative to a hole requires, not to lose.`
  → `The comparator's job is to measure what the alternative to a hole requires.`
- **style**: `The comparator serves to measure what the alternative to a hole requires, not to lose.`
  → `The comparator serves to measure what the alternative to a hole requires.`

### R20 — T2 — Results
"X, not Y" -> X
- **v2**: `\noindent The operation that rewrites only the bytes it names and leaves behaviour unchanged is the hole, not deletion in general.`
  → `\noindent The operation that rewrites only the bytes it names and leaves behaviour unchanged is the hole.`
- **style**: `\noindent The hole, not deletion in general, is the operation`
  → `\noindent The hole is the operation`

### R21 — T2 — Results
"X, never Y" -> X
- **both**: `The claim is a match with a refit without the rows, never erasure without loss.`
  → `The claim is a match with a refit without the rows.`

### R22 — T5 — Results
heading: parenthetical deleted
- **both**: `\paragraph{Composition-only n-CLU (composes, cannot forget).}`
  → `\paragraph{Composition-only n-CLU.}`

### R23 — T5 — Results
heading: parenthetical deleted
- **both**: `\paragraph{Item memory (forgets, loses composition).}`
  → `\paragraph{Item memory.}`

### R24 — T7a — Results
the one-sentence paragraph split at its colon and at "because": "It is a ceiling and not an arm." (8 words)
- **v2**: `deletes by construction and composes poorly: on row 4 it sits below the learned`
  → `deletes by construction and composes poorly. On row 4 it sits below the learned`
- **v2**: `loses, because it is a ceiling and not an arm.`
  → `loses. It is a ceiling and not an arm.`
- **style**: `loses, because this n-CLU is a ceiling and not an arm.`
  → `loses. This n-CLU is a ceiling and not an arm.`

### R25 — T5 — Results
heading: apposition deleted
- **both**: `\paragraph{LACE and PoE, at their own settings.}`
  → `\paragraph{LACE and PoE.}`

### R26 — T2 T3 — Results
"we report them for completeness, not as evidence" deleted (self-narration and contrast)
- **both**: `the level of our landscape controls; we report them for completeness, not as evidence, in Tab.~\ref{tab:lace-poe} (App.~\ref{app:t7more}).`
  → `the level of our landscape controls (Tab.~\ref{tab:lace-poe}, App.~\ref{app:t7more}).`

### R27 — T5 — Results
heading: apposition deleted
- **both**: `\paragraph{The generation table, on the $\Kstar$ single design.}`
  → `\paragraph{The generation table.}`

### R28 — T2 — Results
"X, never Y" -> X
- **both**: `The generation table is a second Shapes3D instrument, never merged with Tab.~\ref{tab:t7}: a fixed set of`
  → `The generation table is a second Shapes3D instrument: a fixed set of`

### R29 — T7a — Results
split twice: "The declared protocol's falsification clause fires." (6 words) follows the 20-word S2 sentence
- **both**: `though its mean is positive, and the`
  → `though its mean is positive. The`
- **both**: `declared protocol's falsification clause fires: the claim that`
  → `declared protocol's falsification clause fires. The claim that`

### R30 — T5 — Results
heading: colon deleted
- **both**: `\paragraph{CMAZE-T with symmetric inputs: six seeds.}`
  → `\paragraph{CMAZE-T with symmetric inputs.}`

### R31 — T6 — Results
n-CLU density: pronoun (the subject of the previous sentence is the n-CLU)
- **both**: `to every arm the n-CLU is behind the best matched baseline on all three TEST seeds`
  → `to every arm it is behind the best matched baseline on all three TEST seeds`
- words added: DRAFT-v2 adds it×1; DRAFT-v2-style adds it×1

### R32 — T5 — Results
heading: sentence -> noun phrase
- **both**: `\paragraph{A relational request coding for the n-CLU is an input-representation result.}`
  → `\paragraph{Relational request coding.}`

### R33 — T6 — Results
n-CLU density: pronoun (the sentence's subject is the change to the n-CLU's coding)
- **both**: `(n-CLU-REL), lifts the n-CLU on every seed`
  → `(n-CLU-REL), lifts it on every seed`
- words added: DRAFT-v2 adds it×1; DRAFT-v2-style adds it×1

### R34 — T2 — Results
"X, not Y" -> X
- **both**: `relational request coding carried through the n-CLU's closed-form head, not the n-CLU.`
  → `relational request coding carried through the n-CLU's closed-form head.`

### R35 — T5 — Results
heading: colon-as-sentence -> noun phrase
- **both**: `\paragraph{CMAZE-T: what is not claimed.}`
  → `\paragraph{Claims not made on CMAZE-T.}`
- words added: DRAFT-v2 adds claims×1, made×1, on×1; DRAFT-v2-style adds claims×1, made×1, on×1

### R36 — T7a — Results
split at "so": "Nothing is claimed on either row." (6 words)
- **both**: `n-CLU is below the best matched baseline, so nothing is claimed on either row.`
  → `n-CLU is below the best matched baseline. Nothing is claimed on either row.`

### R37 — T5 — Results
heading: gerund clause -> noun phrase
- **both**: `\paragraph{Reading the CMAZE-T panel figures.}`
  → `\paragraph{The CMAZE-T panel figures.}`

### R38 — T7a — Results
split: "A picture need not show all four." (7 words)
- **both**: `the pocket after the move, and a picture need not show all four.`
  → `the pocket after the move. A picture need not show all four.`

### D1 — T7a — Discussion
split twice ("so", "; and" deleted): "K* is not the point of smallest footprint." (8 words) follows the 40-word sentence
- **both**: `with its minimum 3.9942 times the reference at $K=24$, so $\Kstar$ is not the point of smallest footprint; and on the one configuration`
  → `with its minimum 3.9942 times the reference at $K=24$. $\Kstar$ is not the point of smallest footprint. On the one configuration`

### D2 — T2 — Discussion
"X, not Y" -> X
- **both**: `(the validation sweep labels itself a smoke, not a claim)`
  → `(the validation sweep labels itself a smoke)`

### D3 — T2 — Discussion
"X, not Y" -> X
- **both**: `we judge the risk low, not zero.`
  → `we judge the risk low.`

### D4 — T5 — Discussion
run-in title deleted: a prepositional phrase that repeats its subsection heading
- **both**: `\paragraph{Toward a CMAZE suite.} CMAZE-T is the first instrument`
  → `CMAZE-T is the first instrument`

### D5 — T1 — Discussion
rhetorical anaphoric triad "the same X, the same Y, the same Z" -> one list (the style twin already reads "the same content, wells and read")
- **v2**: `the same content, the same wells, the same read, with the settling loop's`
  → `the same content, wells and read, with the settling loop's`
- words added: DRAFT-v2 adds and×1

### D6 — T7a — Discussion
split: "We asserted this in code." (5 words) follows the 44-word identity sentence
- **both**: `We asserted this in code: the zero-step arm was required`
  → `We asserted this in code. The zero-step arm was required`

### D7 — T2 T7a — Discussion
"X, not Y" -> X and split: "Here the dynamics are the read." (6 words)
- **both**: `environment's dynamics; here the dynamics are the read, not a model of the data.)`
  → `environment's dynamics. Here the dynamics are the read.)`

### D8 — T5 — Discussion
heading: parenthetical deleted
- **both**: `\paragraph{Two laws measured on this study's own n-CLU (\textsc{confirmed [this study]}).}`
  → `\paragraph{Two laws measured on this study's own n-CLU.}`

### D9 — T2 — Discussion
"X, not Y" -> X (the Summary's "a law about the n-CLU, not a lever on the task" is Discussion's one)
- **v2**: `prediction is a \emph{time}, not a score: with nothing fitted`
  → `prediction is a \emph{time}: with nothing fitted`
- **style**: `prediction is a \emph{time}, not a score. With nothing fitted`
  → `prediction is a \emph{time}. With nothing fitted`

### D10 — T1 — Discussion
rhetorical triad of one claim ("occupy no row, beat no model, make no claim") -> the claim
- **both**: `They occupy no row of the evaluation, they beat no model, and we make no capability claim from them}`
  → `We make no capability claim from them}`

### D11 — T6 — Discussion
"naive unit" after its first mention -> n-CLU
- **both**: `The naive unit provides deletion by subtraction`
  → `The n-CLU provides deletion by subtraction`
- words added: DRAFT-v2 adds n-clu×1; DRAFT-v2-style adds n-clu×1

### D12 — T6 — Discussion
style only: coinage "leaving note" (and "naive unit") dropped with the clause that announced it
- **style**: `The leaving note states what the naive unit does not provide; the follow-up is therefore a full evaluation`
  → `The follow-up is a full evaluation`

### P1 — T3 — App-preface
announcements "Appendix-maximalism is in force" and "nothing is omitted" deleted
- **v2**: `Appendix-maximalism is in force: every row, corollary, negative and control of the study that is not in the main text is here, and nothing is omitted.`
  → `Every row, corollary, negative and control of the study that is not in the main text is here.`
- **style**: `Appendix-maximalism applies: every row, corollary, negative and control of the study that is not in the main text appears here, and nothing is omitted.`
  → `Every row, corollary, negative and control of the study that is not in the main text appears here.`

### P2 — T7a — App A
split: "Numbers are quoted as the record prints them." (8 words) follows the 29-word sentence
- **v2**: `Numbers are quoted as the record prints them; none is re-derived.`
  → `Numbers are quoted as the record prints them. None is re-derived.`
- **style**: `Numbers are quoted as the record prints them, and none is re-derived.`
  → `Numbers are quoted as the record prints them. None is re-derived.`

### P3 — T7a — App A
split: "They differ in partition and in convention." (7 words)
- **v2**: `They differ in partition and in convention; the convention alone`
  → `They differ in partition and in convention. The convention alone`
- **style**: `They differ in partition and in convention; the convention alone`
  → `They differ in partition and in convention. The convention alone`

### P4 — T7a — App B
split: "Baselines are matched on read-path parameters only." (7 words)
- **both**: `parameters only; no baseline matched to the memory footprint is printed.`
  → `parameters only. No baseline matched to the memory footprint is printed.`

### P5 — T7a — App B
split: "No learned-baseline column was replaced." (5 words)
- **both**: `(Tab.~\ref{tab:lace-poe}), 3 seeds; no learned-baseline column was replaced.`
  → `(Tab.~\ref{tab:lace-poe}), 3 seeds. No learned-baseline column was replaced.`

### P6 — T7a — App C
split at "so": "Its n=4 and n=5 rows are not claimable." (8 words)
- **both**: `above the 1.0 bar, so % SRC: w20-shapes-data-matched/FINDINGS.md its $n=4$ and $n=5$ rows are not claimable.`
  → `above the 1.0 bar. % SRC: w20-shapes-data-matched/FINDINGS.md Its $n=4$ and $n=5$ rows are not claimable.`

### P7 — T7a — App D
split twice: "Rank/width is a warning." (4 words) follows the 35-word sentence
- **v2**: `0.4375--0.9844 on seed 2, and the controls still collapse, so rank/width is a warning and only the control decides.`
  → `0.4375--0.9844 on seed 2, and the controls still collapse. Rank/width is a warning. Only the control decides.`
- **style**: `0.4375--0.9844 on seed 2, yet the controls still collapse; rank/width is therefore a warning, and only the control decides.`
  → `0.4375--0.9844 on seed 2, yet the controls still collapse. Rank/width is therefore a warning. Only the control decides.`

### P8 — T3 T2 — App E
"declared, not netted" -> "declared" (self-narration and contrast)
- **both**: `is declared, not netted.`
  → `is declared.`

### P9 — T2 — App E
"neither ... nor" -> plain negation (App E keeps "a property of the content, not of the downdate")
- **v2**: `its record states that it is neither a test run nor a claim.`
  → `its record states that it is not a test run or a claim.`
- **style**: `neither a test run nor a claim.`
  → `not a test run or a claim.`
- words added: DRAFT-v2 adds not×1, or×1; DRAFT-v2-style adds not×1, or×1

### P10 — T7a — App E
split: "The top-level ``all assertions passed'' excludes it." (7 words)
- **v2**: `content, not of the downdate; the top-level ''all assertions passed'' excludes it.`
  → `content, not of the downdate. The top-level ''all assertions passed'' excludes it.`
- **style**: `not of the downdate, and the top-level ''all assertions passed'' excludes it.`
  → `not of the downdate. The top-level ''all assertions passed'' excludes it.`

### P11 — T7a — App E
split: "The numerator is the failing leg's own worst." (8 words)
- **v2**: `before the re-run and not moved; the numerator is the failing leg's own worst.`
  → `before the re-run and not moved. The numerator is the failing leg's own worst.`
- **style**: `declared before the re-run and not moved; its numerator is the failing leg's own worst.`
  → `declared before the re-run and not moved. Its numerator is the failing leg's own worst.`

### P12 — T7a — App F
split at "while"/"whereas": "Every other rung carries 0." (5 words)
- **v2**: `3.4 times the n-CLU pin, while every other rung carries 0.`
  → `3.4 times the n-CLU pin. Every other rung carries 0.`
- **style**: `3.4 times the n-CLU pin, whereas every other rung carries 0.`
  → `3.4 times the n-CLU pin. Every other rung carries 0.`

### P13 — T3 T2 — App H
announcement "never pooled" deleted
- **v2**: `aggregate / exact-match, per seed, never pooled: frozen`
  → `aggregate / exact-match, per seed: frozen`
- **style**: `(aggregate / exact-match, per seed, never pooled) are: frozen`
  → `(aggregate / exact-match, per seed) are: frozen`

### P14 — T3 T2 — App H
announcement "(reported, never tuned)" deleted
- **v2**: `The anchor arm's distance from the frozen encoder (reported, never tuned): feature relative`
  → `The anchor arm's distance from the frozen encoder: feature relative`
- **style**: `(reported, never tuned) is a feature relative`
  → `is a feature relative`

### P15 — T7a — App I
v2 split twice: "Equal compute passes on all three seeds." (6 words) follows the 19-word sentence (the style twin already splits there)
- **v2**: `images) is a separate table: equal compute passes on all three seeds; its equal-seconds control is`
  → `images) is a separate table. Equal compute passes on all three seeds. Its equal-seconds control is`

### P16 — T7a — App J
split: "(C3) The state is carried in span coordinates." (8 words)
- **v2**: `(C3) The state is carried in span coordinates; the same energy`
  → `(C3) The state is carried in span coordinates. The same energy`
- **style**: `(C3) The state is carried in span coordinates; the same energy`
  → `(C3) The state is carried in span coordinates. The same energy`

### P17 — T7a — App J
split: the Hessian sentence (8 words) follows the 40-word sentence
- **v2**: `524,288 B on every seed; a finite-difference Hessian check reads`
  → `524,288 B on every seed. A finite-difference Hessian check reads`
- **style**: `524,288 B on every seed; and a finite-difference Hessian check reads`
  → `524,288 B on every seed. A finite-difference Hessian check reads`

### P18 — T7a — App K
split: "What breaks the read is motion inside it." (8 words)
- **v2**: `AUC 0.457 to 0.559); what breaks the read is motion inside it.`
  → `AUC 0.457 to 0.559). What breaks the read is motion inside it.`
- **style**: `AUC 0.457 to 0.559); what breaks the read is motion inside it.`
  → `AUC 0.457 to 0.559). What breaks the read is motion inside it.`

### P19 — T7a — App K
split: "No claim is made on those rows." (7 words)
- **v2**: `a byte multiple of 59,392; no claim is made on those rows.`
  → `a byte multiple of 59,392. No claim is made on those rows.`
- **style**: `a byte multiple of 59,392; we make no claim on those rows.`
  → `a byte multiple of 59,392. We make no claim on those rows.`

### P20 — T3 — App L
announcement "Status discipline is part of the section." deleted
- **v2**: `\textsc{refuted} and \textsc{untested} as stated. Status discipline is part of the section.`
  → `\textsc{refuted} and \textsc{untested} as stated.`
- **style**: `Status discipline is part of the section.`
  → `(deleted)`

### P21 — T7a — App L
split inside the parenthesis: "Recency-based eviction is excluded as intrinsically history-dependent." (7 words)
- **both**: `priority- or attribute-based eviction; recency-based eviction is excluded as`
  → `priority- or attribute-based eviction. Recency-based eviction is excluded as`

### P22 — T7a — App L
split: "The retention floor is curvature-independent." (5 words)
- **both**: `derived object inverts it: the retention floor is curvature-independent.`
  → `derived object inverts it. The retention floor is curvature-independent.`

### P23 — T3 — App L
announcement "that must travel with it" deleted
- **both**: `each with the rider that must travel with it.`
  → `each with its rider.`
- words added: DRAFT-v2 adds its×1; DRAFT-v2-style adds its×1

### P24 — T3 T7a — App O
signpost "and this is what it yields" deleted: "Light adaptation is optional." (4 words) follows the 19-word sentence
- **v2**: `Light adaptation is optional, and this is what it yields.`
  → `Light adaptation is optional.`
- **style**: `Light adaptation is optional; this subsection reports what it yields.`
  → `Light adaptation is optional.`

### P25 — T3 T2 — App O
announcement "never pooled" deleted
- **both**: `aggregate, per seed, never pooled).`
  → `aggregate, per seed).`

### P26 — T7a — App O
split: "This is a finding about the encoder." (7 words) follows the 59-word sentence
- **both**: `This is a finding about the encoder, and the claim that the n-CLU shapes the representation is not made.`
  → `This is a finding about the encoder. The claim that the n-CLU shapes the representation is not made.`

### P27 — T2 — App O
"a block of its own, never merged with ..." deleted (contrast; the table caption states it)
- **both**: `(Tab.~\ref{tab:bard-gpu}, a block of its own, never merged with the shortened-decoder leg of Tab.~\ref{tab:encloop})`
  → `(Tab.~\ref{tab:bard-gpu})`

### P28 — T7a — App O
split: "The ``unfreezing hurts'' half does not survive." (7 words)
- **both**: `of 6 on the longer one); the ''unfreezing hurts'' half does not survive.`
  → `of 6 on the longer one). The ''unfreezing hurts'' half does not survive.`

### P29 — T2 — App P
"X, not Y" -> X (App P keeps "on this memory, not the law")
- **both**: `whose prediction is a \emph{time}, not a score.`
  → `whose prediction is a \emph{time}.`

### P30 — T3 — App P
announcement "printed and" deleted
- **both**: `its maximum over $\gamma$, is printed and not moved: 59.1158`
  → `its maximum over $\gamma$, is not moved: 59.1158`

### P31 — T1 — App P
rhetorical triad of one claim -> the claim (as D10)
- **both**: `It occupies no row of the evaluation, it beats no model, and we make no capability claim from it}`
  → `We make no capability claim from it}`

### P32 — T7a — App Q
split: "pca_tensor is not run at K* (App. S)." (8 words)
- **both**: `the same post-deletion requests is not run; \texttt{pca\_tensor} is not run at $\Kstar$`
  → `the same post-deletion requests is not run. \texttt{pca\_tensor} is not run at $\Kstar$`

### P33 — T7a — App R
split: "They are permanently blocked, not pending." (6 words; App R's one contrast) follows the 32-word sentence
- **both**: `permanently blocked, not pending: the run behind`
  → `permanently blocked, not pending. The run behind`

### P34 — T3 — App N
announcement "all printed here" deleted
- **both**: `The harness ships three warnings, all printed here: a composition-blind`
  → `The harness ships three warnings: a composition-blind`

### P35 — T2 — App L
style only: "neither ... nor" -> plain negation (App L keeps "parameter separation, not block structure")
- **style**: `have neither a derivation nor an impossibility argument`
  → `have no derivation and no impossibility argument`
- words added: DRAFT-v2-style adds no×2, and×1

### R39 — T6 — Results
n-CLU density: pronoun (the clause's subject "it" is the n-CLU)
- **both**: `the n-CLU's own loss on that deletion, about six tuples`
  → `its own loss on that deletion, about six tuples`
- words added: DRAFT-v2 adds its×1; DRAFT-v2-style adds its×1

