# LIMITATIONS — part 7 (item 5 applied to the Discussion limitations list), both drafts

## DRAFT-v2: the list before

```latex
\subsection{Limitations}
\label{sec:limits-list}

\begin{enumerate}
\item \textbf{On the maze transition task the n-CLU's lead is small and seed-dependent.}
  % pw1-draft-cuts (item 14): removed limitation 1: "no n-CLU row on the route tasks" (CMAZE-2, CMAZE-2b); its numbers stay in: w13-seeds-and-tables/TABLES.md
  On CMAZE-T with symmetric inputs the n-CLU is ahead of the best matched baseline on three of six seeds,
  level on two and behind on one, and behind on every seed of the full word design % SRC: w21-maze-t-seed4/FINDINGS.md
  (Sec.~\ref{sec:maze-wm}, Tab.~\ref{tab:cmazet-sym}).
\item \textbf{Deleting a conjunction changes what the book knows about the shared value.} After deletion
  the n-CLU and the address book match those never fitted on the deleted rows, but deleting the red
  sphere lowers composition on
  the surviving combinations by 6.13 / 6.43 / 6.30 tuples per held-out combination, 32.1 on the combinations that request red % SRC: w20-shapes-data-matched/FINDINGS.md
  with another shape at seed 0, while a random 1\% deletion moves them by at most 0.11; retraining a % SRC: w20-shapes-data-matched/FINDINGS.md
  baseline without the same rows is cleaner on collateral damage (Sec.~\ref{sec:shapes-del},
  Sec.~\ref{sec:del-comparators}).
\item \textbf{Deletion is not cheap in memory or time: the address book's refit.} The book's retained statistic is 0.97 GB, % SRC: w20-shapes-data-matched/FINDINGS.md
  the size of its fit data, and its refit takes 2,367 to 5,913 s, longer than a baseline's % SRC: w20-shapes-data-matched/FINDINGS.md
  retrain-without; ``reads no training row'' is not a size or a speed claim.
\item \textbf{Deletability requires a large number of bytes.} The deletable object at $\Kstar$ is 4.0768 to 4.0897
  times the reference (Tab.~\ref{tab:ledger-kstar}); across nine swept partitions the honest
  four-part total is U-shaped in $K$, with its minimum 3.9942 times the reference at $K=24$.
  $\Kstar$ is not the point of smallest footprint. On the one configuration at $K\ge36$ whose composition
  was measured ($K=40$, 7 atoms per well, 0.991 times the pin) composition collapses to 0.0648.
  No object with a float64 sufficient statistic composes and forgets inside 524,288 bytes at rank
  64 on any swept partition (the validation sweep labels itself a smoke).
  % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
\item \textbf{The baseline search may be truncated.} Every data-matched family selected the largest
  learning rate and the longer epoch count of its grids, and the MLP at seed 0 still gains 1.47 tuples % SRC: w20-shapes-data-matched/FINDINGS.md
  per held-out combination from ten to thirty times its steps (Sec.~\ref{sec:shapes-t7}).
\item \textbf{The encoder loop licenses no n-CLU claim.} Training the encoder through the n-CLU
  fired the pre-declared falsifier on row 3 of the shortened leg, a falsifier that involves no baseline % SRC: w13-seeds-and-tables/TABLES.md
  (App.~\ref{app:encloop}).
\item \textbf{No functioning generative baseline.} LACE and PoE, re-implemented at their own settings
  in this feature space, render no exact-match request on the generation table and score at the level
  of our landscape controls (Tab.~\ref{tab:lace-poe}); the study compares naive units only and has no
  working compositional-generation baseline.
% pw1-draft-cuts (item 14): removed limitations: post-deletion composition on the maze (CMAZE-2) and maze refusal as novelty detection; its numbers stay in: w13-seeds-and-tables/TABLES.md; w14-advisor-report.md; w14-refusal-displays-closeout/FINDINGS.md
\item \textbf{At most three seeds on Shapes3D.} Seeds 3 and 4 cannot be judged there (no decoder, no
  external classifier), so no Shapes3D row can exceed three seeds; CMAZE-T prints six.
\item \textbf{Further limitations} (an unseen-valid row that belongs to a generic learner, an
  exhausted placement dial, and comparators not run) are listed in App.~\ref{app:morelimits}.
\item \textbf{Residual novelty risk.} Abstract-level screening cannot catch a match buried in a
  method section, and very recent preprints and non-indexed workshop papers are incompletely
  covered; we judge the risk low. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
\end{enumerate}
```

## DRAFT-v2: the list after

```latex
\subsection{Limitations}
\label{sec:limits-list}

\begin{enumerate}
\item \textbf{On the maze transition task the n-CLU's lead is small and seed-dependent.}
  % pw1-draft-cuts (item 14): removed limitation 1: "no n-CLU row on the route tasks" (CMAZE-2, CMAZE-2b); its numbers stay in: w13-seeds-and-tables/TABLES.md
  On CMAZE-T with symmetric inputs the n-CLU is ahead of the best matched baseline on three of six seeds,
  level on two and behind on one, and behind on every seed of the full word design % SRC: w21-maze-t-seed4/FINDINGS.md
  (Sec.~\ref{sec:maze-wm}, Tab.~\ref{tab:cmazet-sym}).
\item \textbf{Deleting a conjunction changes what the book knows about the shared value.} After deletion
  the n-CLU and the address book match those never fitted on the deleted rows, but deleting the red
  sphere lowers composition on
  the surviving combinations by about six tuples per held-out combination, about thirty on the combinations that request red % SRC: w20-shapes-data-matched/FINDINGS.md
  with another shape at seed 0, while a random 1\% deletion moves them by at most about a tenth of a tuple; retraining a % SRC: w20-shapes-data-matched/FINDINGS.md
  baseline without the same rows is cleaner on collateral damage (Tab.~\ref{tab:del}, Sec.~\ref{sec:shapes-del},
  Sec.~\ref{sec:del-comparators}).
  % PW2-5 moved -> Tab. del (column (d): -6.128 / -6.431 / -6.303; item -0.018 / -0.110 / -0.009; caption notes: the 20 red-with-another-shape combinations -32.1 each at seed 0) and Fig. deletion caption (6.13, 6.43, 6.30; 0.02, 0.11, 0.01): 6.13 / 6.43 / 6.30; 32.1; 0.11; SRC: w20-shapes-data-matched/FINDINGS.md
\item \textbf{Deletion is not cheap in memory or time: the address book's refit.} The book's retained statistic is nearly a gigabyte, % SRC: w20-shapes-data-matched/FINDINGS.md
  the size of its fit data, and its refit takes longer than a baseline's % SRC: w20-shapes-data-matched/FINDINGS.md
  retrain-without (Tabs.~\ref{tab:del} and~\ref{tab:rivals-del}); ``reads no training row'' is not a size or a speed claim.
  % PW2-5 moved -> Tab. del (cost rows: reads the 967,304,800 B statistic; book refit 2,590.7 ... 2,367.3 ... 5,912.9 s) and Tab. rivals-del (retrain times): 0.97 GB; 2,367 to 5,913 s; SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.json
\item \textbf{Deletability requires a large number of bytes.} The deletable object at $\Kstar$ is about four
  times the reference (Tab.~\ref{tab:ledger-kstar}); across nine swept partitions the honest
  four-part total is U-shaped in $K$, with its minimum at $K=24$ (Tab.~\ref{tab:kcurve}).
  $\Kstar$ is not the point of smallest footprint. On the one configuration at $K\ge36$ whose composition
  was measured, composition collapses (Tab.~\ref{tab:kcurve}, caption).
  No object with a float64 sufficient statistic composes and forgets inside the reference at rank
  64 on any swept partition (the validation sweep labels itself a smoke).
  % PW2-5 moved -> Tab. ledger-kstar (last row: 4.0897 / 4.0768 / 4.0847 x the 524,288 B reference), Tab. kcurve (K=24 row: 3.9942) and Tab. kcurve caption (ADDED: K=40, 7 atoms per well, 0.991 times the reference, validation composition 0.0648); 524,288 is the reference defined in Methods: 4.0768 to 4.0897; 3.9942; K=40; 7 atoms; 0.991; 0.0648; 524,288; SRC: w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
  % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
\item \textbf{The baseline search may be truncated.} Every data-matched family selected the largest
  learning rate and the longer epoch count of its grids, and the MLP at seed 0 still gains more than a tuple % SRC: w20-shapes-data-matched/FINDINGS.md
  per held-out combination from ten to thirty times its steps (Sec.~\ref{sec:shapes-t7}, Tab.~\ref{tab:t7}).
  % PW2-5 moved -> Sec. results first paragraph (Head-held: "+1.47") and Tab. t7 (diagnostic MLP row at thirty times the steps, 41.477 at seed 0): 1.47; SRC: w20-shapes-data-matched/FINDINGS.md
\item \textbf{The encoder loop licenses no n-CLU claim.} Training the encoder through the n-CLU
  fired the pre-declared falsifier on row 3 of the shortened leg, a falsifier that involves no baseline % SRC: w13-seeds-and-tables/TABLES.md
  (App.~\ref{app:encloop}).
\item \textbf{No functioning generative baseline.} LACE and PoE, re-implemented at their own settings
  in this feature space, render no exact-match request on the generation table and score at the level
  of our landscape controls (Tab.~\ref{tab:lace-poe}); the study compares naive units only and has no
  working compositional-generation baseline.
% pw1-draft-cuts (item 14): removed limitations: post-deletion composition on the maze (CMAZE-2) and maze refusal as novelty detection; its numbers stay in: w13-seeds-and-tables/TABLES.md; w14-advisor-report.md; w14-refusal-displays-closeout/FINDINGS.md
\item \textbf{At most three seeds on Shapes3D.} Seeds 3 and 4 cannot be judged there (no decoder, no
  external classifier), so no Shapes3D row can exceed three seeds; CMAZE-T prints six.
\item \textbf{Further limitations} (an unseen-valid row that belongs to a generic learner, an
  exhausted placement dial, and comparators not run) are listed in App.~\ref{app:morelimits}.
\item \textbf{Residual novelty risk.} Abstract-level screening cannot catch a match buried in a
  method section, and very recent preprints and non-indexed workshop papers are incompletely
  covered; we judge the risk low. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
\end{enumerate}
```

## DRAFT-v2-style: the list before

```latex
\subsection{Limitations}
\label{sec:limits-list}

\begin{enumerate}
\item \textbf{On the maze transition task the n-CLU's lead is small and seed-dependent.}
  % pw1-draft-cuts (item 14): removed limitation 1: "no n-CLU row on the route tasks" (CMAZE-2, CMAZE-2b); its numbers stay in: w13-seeds-and-tables/TABLES.md
  On CMAZE-T with symmetric inputs the n-CLU is ahead of the best matched baseline on three of six seeds,
  level on two and behind on one, and behind on every seed of the full word design % SRC: w21-maze-t-seed4/FINDINGS.md
  (Sec.~\ref{sec:maze-wm}, Tab.~\ref{tab:cmazet-sym}).
\item \textbf{Deleting a conjunction changes what the book knows about the shared value.} After deletion
  the n-CLU and the address book match those never fitted on the deleted rows, but deleting the red
  sphere lowers composition on
  the surviving combinations by 6.13 / 6.43 / 6.30 tuples per held-out combination, 32.1 on the combinations that request red % SRC: w20-shapes-data-matched/FINDINGS.md
  with another shape at seed 0, while a random 1\% deletion moves them by at most 0.11; retraining a % SRC: w20-shapes-data-matched/FINDINGS.md
  baseline without the same rows is cleaner on collateral damage (Sec.~\ref{sec:shapes-del},
  Sec.~\ref{sec:del-comparators}).
\item \textbf{Deletion is not cheap in memory or time: the address book's refit.} The book's retained statistic is 0.97 GB, % SRC: w20-shapes-data-matched/FINDINGS.md
  the size of its fit data, and its refit takes 2,367 to 5,913 s, longer than a baseline's % SRC: w20-shapes-data-matched/FINDINGS.md
  retrain-without; ``reads no training row'' is not a size or a speed claim.
\item \textbf{Deletability requires a large number of bytes.} The deletable object at $\Kstar$ is 4.0768 to 4.0897
  times the reference (Tab.~\ref{tab:ledger-kstar}); across nine swept partitions the honest
  four-part total is U-shaped in $K$, with its minimum 3.9942 times the reference at $K=24$.
  $\Kstar$ is not the point of smallest footprint. On the one configuration at $K\ge36$ whose composition
  was measured ($K=40$, 7 atoms per well, 0.991 times the pin) composition collapses to 0.0648.
  No object with a float64 sufficient statistic composes and forgets inside 524,288 bytes at rank
  64 on any swept partition (the validation sweep labels itself a smoke).
  % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
\item \textbf{The baseline search may be truncated.} Every data-matched family selected the largest
  learning rate and the longer epoch count of its grids, and the MLP at seed 0 still gains 1.47 tuples % SRC: w20-shapes-data-matched/FINDINGS.md
  per held-out combination from ten to thirty times its steps (Sec.~\ref{sec:shapes-t7}).
\item \textbf{The encoder loop licenses no n-CLU claim.} Training the encoder through the n-CLU
  fired the pre-declared falsifier on row 3 of the shortened leg, a falsifier that involves no baseline % SRC: w13-seeds-and-tables/TABLES.md
  (App.~\ref{app:encloop}).
\item \textbf{No functioning generative baseline.} LACE and PoE, re-implemented at their own settings
  in this feature space, render no exact-match request on the generation table and score at the level
  of our landscape controls (Tab.~\ref{tab:lace-poe}); the study compares naive units only and has no
  working compositional-generation baseline.
% pw1-draft-cuts (item 14): removed limitations: post-deletion composition on the maze (CMAZE-2) and maze refusal as novelty detection; its numbers stay in: w13-seeds-and-tables/TABLES.md; w14-advisor-report.md; w14-refusal-displays-closeout/FINDINGS.md
\item \textbf{At most three seeds on Shapes3D.} Seeds 3 and 4 cannot be judged there (no decoder, no
  external classifier), so no Shapes3D row can exceed three seeds; CMAZE-T prints six.
\item \textbf{Further limitations} (an unseen-valid row that belongs to a generic learner, an
  exhausted placement dial, and comparators not run) are listed in App.~\ref{app:morelimits}.
\item \textbf{Residual novelty risk.} Abstract-level screening cannot catch a match buried in a
  method section, and very recent preprints and non-indexed workshop papers are incompletely
  covered; we judge the risk low. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
\end{enumerate}
```

## DRAFT-v2-style: the list after

```latex
\subsection{Limitations}
\label{sec:limits-list}

\begin{enumerate}
\item \textbf{On the maze transition task the n-CLU's lead is small and seed-dependent.}
  % pw1-draft-cuts (item 14): removed limitation 1: "no n-CLU row on the route tasks" (CMAZE-2, CMAZE-2b); its numbers stay in: w13-seeds-and-tables/TABLES.md
  On CMAZE-T with symmetric inputs the n-CLU is ahead of the best matched baseline on three of six seeds,
  level on two and behind on one, and behind on every seed of the full word design % SRC: w21-maze-t-seed4/FINDINGS.md
  (Sec.~\ref{sec:maze-wm}, Tab.~\ref{tab:cmazet-sym}).
\item \textbf{Deleting a conjunction changes what the book knows about the shared value.} After deletion
  the n-CLU and the address book match those never fitted on the deleted rows, but deleting the red
  sphere lowers composition on
  the surviving combinations by about six tuples per held-out combination, about thirty on the combinations that request red % SRC: w20-shapes-data-matched/FINDINGS.md
  with another shape at seed 0, while a random 1\% deletion moves them by at most about a tenth of a tuple; retraining a % SRC: w20-shapes-data-matched/FINDINGS.md
  baseline without the same rows is cleaner on collateral damage (Tab.~\ref{tab:del}, Sec.~\ref{sec:shapes-del},
  Sec.~\ref{sec:del-comparators}).
  % PW2-5 moved -> Tab. del (column (d): -6.128 / -6.431 / -6.303; item -0.018 / -0.110 / -0.009; caption notes: the 20 red-with-another-shape combinations -32.1 each at seed 0) and Fig. deletion caption (6.13, 6.43, 6.30; 0.02, 0.11, 0.01): 6.13 / 6.43 / 6.30; 32.1; 0.11; SRC: w20-shapes-data-matched/FINDINGS.md
\item \textbf{Deletion is not cheap in memory or time: the address book's refit.} The book's retained statistic is nearly a gigabyte, % SRC: w20-shapes-data-matched/FINDINGS.md
  the size of its fit data, and its refit takes longer than a baseline's % SRC: w20-shapes-data-matched/FINDINGS.md
  retrain-without (Tabs.~\ref{tab:del} and~\ref{tab:rivals-del}); ``reads no training row'' is not a size or a speed claim.
  % PW2-5 moved -> Tab. del (cost rows: reads the 967,304,800 B statistic; book refit 2,590.7 ... 2,367.3 ... 5,912.9 s) and Tab. rivals-del (retrain times): 0.97 GB; 2,367 to 5,913 s; SRC: w20-shapes-data-matched/FINDINGS.md; w20-shapes-data-matched/SHAPES-MATCHED.json
\item \textbf{Deletability requires a large number of bytes.} The deletable object at $\Kstar$ is about four
  times the reference (Tab.~\ref{tab:ledger-kstar}); across nine swept partitions the honest
  four-part total is U-shaped in $K$, with its minimum at $K=24$ (Tab.~\ref{tab:kcurve}).
  $\Kstar$ is not the point of smallest footprint. On the one configuration at $K\ge36$ whose composition
  was measured, composition collapses (Tab.~\ref{tab:kcurve}, caption).
  No object with a float64 sufficient statistic composes and forgets inside the reference at rank
  64 on any swept partition (the validation sweep labels itself a smoke).
  % PW2-5 moved -> Tab. ledger-kstar (last row: 4.0897 / 4.0768 / 4.0847 x the 524,288 B reference), Tab. kcurve (K=24 row: 3.9942) and Tab. kcurve caption (ADDED: K=40, 7 atoms per well, 0.991 times the reference, validation composition 0.0648); 524,288 is the reference defined in Methods: 4.0768 to 4.0897; 3.9942; K=40; 7 atoms; 0.991; 0.0648; 524,288; SRC: w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md
  % SRC: w13-referee-2/REFEREE-v1.md; w13-seeds-and-tables/TABLES.md; w13-draft-revise/DRAFT-v1.md
\item \textbf{The baseline search may be truncated.} Every data-matched family selected the largest
  learning rate and the longer epoch count of its grids, and the MLP at seed 0 still gains more than a tuple % SRC: w20-shapes-data-matched/FINDINGS.md
  per held-out combination from ten to thirty times its steps (Sec.~\ref{sec:shapes-t7}, Tab.~\ref{tab:t7}).
  % PW2-5 moved -> Sec. results first paragraph (Head-held: "+1.47") and Tab. t7 (diagnostic MLP row at thirty times the steps, 41.477 at seed 0): 1.47; SRC: w20-shapes-data-matched/FINDINGS.md
\item \textbf{The encoder loop licenses no n-CLU claim.} Training the encoder through the n-CLU
  fired the pre-declared falsifier on row 3 of the shortened leg, a falsifier that involves no baseline % SRC: w13-seeds-and-tables/TABLES.md
  (App.~\ref{app:encloop}).
\item \textbf{No functioning generative baseline.} LACE and PoE, re-implemented at their own settings
  in this feature space, render no exact-match request on the generation table and score at the level
  of our landscape controls (Tab.~\ref{tab:lace-poe}); the study compares naive units only and has no
  working compositional-generation baseline.
% pw1-draft-cuts (item 14): removed limitations: post-deletion composition on the maze (CMAZE-2) and maze refusal as novelty detection; its numbers stay in: w13-seeds-and-tables/TABLES.md; w14-advisor-report.md; w14-refusal-displays-closeout/FINDINGS.md
\item \textbf{At most three seeds on Shapes3D.} Seeds 3 and 4 cannot be judged there (no decoder, no
  external classifier), so no Shapes3D row can exceed three seeds; CMAZE-T prints six.
\item \textbf{Further limitations} (an unseen-valid row that belongs to a generic learner, an
  exhausted placement dial, and comparators not run) are listed in App.~\ref{app:morelimits}.
\item \textbf{Residual novelty risk.} Abstract-level screening cannot catch a match buried in a
  method section, and very recent preprints and non-indexed workshop papers are incompletely
  covered; we judge the risk low. % SRC: _thinkpad/w8/w8-novelty/NOVELTY.md
\end{enumerate}
```

**Lock-step:** the list is byte-identical between the two drafts after the edit: True (before: True).

## Numbers that left the list, and where each lives now

| numbers | where they live now; what the text says | SRC |
|---|---|---|
| 6.13 / 6.43 / 6.30 | Tab. del column (d) (−6.128 / −6.431 / −6.303) and Fig. deletion caption (6.13, 6.43, 6.30); text: "about six" | `w20-shapes-data-matched/FINDINGS.md` |
| 32.1 (seed 0, red with another shape) | Tab. del caption notes (the seed-0 breakdown, −32.1 each); text: "about thirty" | `w20-shapes-data-matched/FINDINGS.md` |
| at most 0.11 (random 1% deletion) | Tab. del column (d), item row (−0.110) and Fig. deletion caption (0.11); text: "at most about a tenth of a tuple" | `w20-shapes-data-matched/FINDINGS.md` |
| 0.97 GB | Tab. del cost row ("reads the 967,304,800 B statistic"); text: "nearly a gigabyte" | `w20-shapes-data-matched/SHAPES-MATCHED.json` |
| 2,367 to 5,913 s | Tab. del cost row (book refit, per scope and seed: 2,367.3 … 5,912.9); text: "longer than a baseline's retrain-without" (Tab. rivals-del) | `w20-shapes-data-matched/FINDINGS.md` |
| 4.0768 to 4.0897 | Tab. ledger-kstar (last row, 4.0897 / 4.0768 / 4.0847); text: "about four times the reference" | `w13-referee-2/REFEREE-v1.md` |
| 3.9942 (minimum at K=24) | Tab. kcurve (K=24 row, four-part ×); text keeps the partition name $K=24$ | `w13-seeds-and-tables/TABLES.md` |
| K=40; 7 atoms per well; 0.991 times the pin; 0.0648 | Tab. kcurve caption (ADDED); text: "composition collapses" | `w13-seeds-and-tables/TABLES.md; w13-referee-2/REFEREE-v1.md` |
| 524,288 bytes | not the reference definition here (that is Methods, "The 524,288-byte figure ... is a reference column"); text: "the reference" | `w13-referee-2/REFEREE-v1.md` |
| 1.47 (MLP, seed 0, ten to thirty times its steps) | Sec. 4.1 first paragraph (Head-held, "+1.47") and Tab. t7 (diagnostic MLP row); text: "more than a tuple" | `w20-shapes-data-matched/FINDINGS.md` |
