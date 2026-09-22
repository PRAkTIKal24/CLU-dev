# w17 spoke/draft-v2 — FINDINGS

**Deliverable:** `~/clu/outputs/w17-paper/DRAFT-v2.tex` (ICLR 2027 template, anonymised, `\iclrfinalcopy` commented out).
Compiled PDF (for reference only, not a deliverable of the shared dir): `~/clu/outputs/w17-draft-v2/DRAFT-v2.pdf`, **31 pages**; main text ends on **p. 12** (the ≤12 pp working bound; the venue's own limit is 9 pp at submission, VENUE.md §1, not trimmed to — Head's later pass). AI-use / ethics / reproducibility statements follow the main text; appendices start p. 14 after `\clearpage`.
Compile: `cd ~/clu/outputs/w17-paper && pdflatex DRAFT-v2.tex` (the file sets `\input@path` to `iclr2027/`). pdfTeX, TeX Live 2026: exit 0, 0 undefined references, 0 overfull boxes. No `references.bib` yet ⇒ bibliography block is guarded by `\IfFileExists` and emits nothing; no `\cite{}` is used (cite-tagger inserts them).

**Scored runs: 0. Compute: 0. No sub-agents. No background jobs (PIDS.txt empty).**

## Acceptance criterion — status
| clause | status |
|---|---|
| `DRAFT-v2.tex` exists, anonymised | ✅ no names / institution / URL / acknowledgements in the text (grep-checked); author block "Anonymous authors" |
| six-part spine in order | ✅ §1 positioning (+§1.2 why it matters) · §2 object/judge/ledger · §3 Naive CLU on Shapes (T7 at K\*, ladder + inference note, deletion value/conj/item, probe, hole refusal, ablations, encoder-adapt anchor, BAR D) · §4 maze (deletion transfer CMAZE-2, request-layer fix CMAZE-2b, verb = INTERFACE claim, lookup 0.97 within-maze floor, w16 placeholders) · §5 limitations → clue (zero-dynamics bit-identical) → theory (class name "Latent Dynamics Model"; store-level scoping quoted in §2) → leaving note · §6 follow-up |
| every number has an artefact comment | ✅ `check_provenance.py` (in this dir): **1,821 number tokens checked, 0 not found in their cited artefact**; the one reported miss is a tokeniser artefact (the set `{5,8,…,32}` read as one token); the 10 "unsourced" lines carry only wave labels / template paths, no result numbers. Output: `provenance_check.txt`. ⚠ The checker verifies the number STRING occurs in the cited file, not that it is the right cell — a referee should spot-check. |
| every w15/w16 gap is a `\PLACEHOLDER{}` naming the row and that it may land negative | ✅ 29 placeholders, listed below and in `placeholders.txt`; **no w15 or w16 number anywhere** (no SRC comment points at any w15/w16 artefact; the only w15–w18 file cited is `advisor-wave-plan-w15-w18.md`, for the planned 18-run budget, which is not a result) |
| FINDINGS lists every placeholder and every TODO | ✅ below |

## Decisions taken (pre-authorised; also in LOG.md)
- **D1 byte convention:** ONE convention in the main text — symmetric-triangle float64 (the K\*/crossover ledger). The w12 ladder's K = 5 object (full-square) appears only in App. B, labelled, re-priced on the triangle convention (REFEREE-v1 N1) with the **938,413–1,710,324** bracket printed. The 3.13–3.26 M "inflated" opponent is **not** printed as a row.
- **D2 rivals:** main text = naive vs naive (read-path, 130,433–131,891 params). The 1×-four-part-total learner at K\* is an appendix **stress comparison** (App. C).
- **D3** one K in main text (K\* = 32); K = 5 = App. B "sweep endpoint".
- **D4** T7 (rows 3–4, n = 3) from `w12b-nr6-and-kstar/NR6-T7.md`; K = 5 TEST n = 1 only in App. (not-runs).
- **D5 BAR D:** the banked w8 CPU decomposition (TABLES.md T2) and the w8 GPU-leg probe (w8-BAR-D-fold.md) written in full in App.; the **w15 scored BAR D row + its decomposition paragraph = PLACEHOLDER**, per AMENDMENT-1.
- **D6 maze:** CMAZE-2 (banked deletion transfer), CMAZE-2b (verb, request layer) never in one result table; the request-layer before/after table is the only place both names appear, as the before/after of one rule with no result row.
- **D7 positioning count:** the w12 positioning scout carries **no counted intersection figure** (it is a positioning-shape brief). The count used (≈4,050 screened; 0 of 3,038; compositional∧generative∧exact-deletion → 0) is from the w8 novelty scout's forward-citation crawl, `_thinkpad/w8/w8-novelty/NOVELTY.md` Addendum B.
- **D8** "~1/7×" composition-only store ratio: **no artefact** (it is a planning figure in the advisor plan) → `\TODO`; byte totals printed instead.
- **D9** "Latent Dynamics Model" used per Head ruling, with a one-sentence disambiguation of the term's other uses (latent-dynamics-umbrella.md flags the collision).
- **D10** refusal comparator: TABLES.md T10 prints the refuse-token transformer's off-grid-included diagnostic (AUROC 0.81–0.88) as if primary; the source (w12-deletion-baselines FINDINGS §5) says the primary set is AUROC 1.0000 / false refusal 0.0000. The draft prints **both, labelled**.
- Wide tables auto-fit with `adjustbox`; figures included via a guarded `\figslot` (renders a box if the file is absent).

## Placeholders (29) — wave 18's `placeholder-fill` greps `\PLACEHOLDER{`
Line numbers refer to DRAFT-v2.tex as installed.
- L355	w15 \texttt{lace-poe-rows}: LACE and PoE at their own budgets; may beat the store
- L457	w15 \texttt{coverage-control} (one scored run): the $n=6$ store score at $\Kstar$ with the fit pool restricted to $n=3$'s train index, 3 seeds, beside the $n=6$ and $n=3$ rows of Table~\ref{tab:ladder}. It may land showing that the 2-way$\to$6-way rise is a coverage effect, that it is not, or neither; the ladder table and its inference note stand unchanged in every case.
- L500	w15 \texttt{single-design-seeds}: the three deletion rows at seeds 1--2; any may land as a negative
- L509	w15 \texttt{single-design-seeds} (\texttt{single-design} scored runs 2/3 and 3/3): the value, red-sphere and 1\% item deletion rows of Table~\ref{tab:del} at seeds 1 and 2, with the retrain-without rival cost, so that the table prints $n=3$. Any of them may land as a negative: exactness above floating-point roundoff, a survivor composition change beyond the declared margin, or a refused leg; Table~\ref{tab:del} then prints it as such.
- L654	w15 \texttt{lace-poe-rows}: LACE and PoE rows (at their own declared budgets) added beside the learned rivals on rows 3--4 (Table~\ref{tab:t7}) and on every ladder rung, 3 seeds, plus \texttt{generation-table} run 3 on the $\Kstar$ single design. LACE or PoE may beat the store on some or all rows; they are added and never replace the read-path rival columns.
- L717	w15 \texttt{maze-deletion-2b} (free replication): the exact-deletion rows of Table~\ref{tab:maze-del} re-run on CMAZE-2b, the environment on which the verb result of \S\ref{sec:verb} is measured, downdate against refit-without and the projector residual, replicated from a clean checkout, 3 seeds, printed in its own table. It may land as a negative: a residual above roundoff, a failed replication, or a changed ledger; it is then printed as such and Table~\ref{tab:maze-del} stays the CMAZE-2 result.
- L1203	w15 \texttt{crossover-seeds}: the seven middle rows of Table~\ref{tab:kcurve} at seeds 1 and 2, so that every row carries 3 seeds; the curve's shape, its minimum and the selected $\Kstar$ may change, and a crossover above the 0.10 threshold may appear.
- L1401	w15 \texttt{single-design-seeds}: the rival retrain-without cost line at seeds 1 and 2 beside the store's deletion rows; it may differ from the seed-0 line above.
- L1486	w15 \texttt{bar-d-score}: the scored BAR D row of the longer leg (store minus matched-parameter learner inside each encoder arm, and whether it grows from \texttt{alone} to \texttt{loop\_swap}), 3 seeds, its own table; it may show no growth.
- L1925	w15 \texttt{cmaze-t-harness} and \texttt{cmaze-o-harness}: the harness audits (independent replay, stepper disagreements, no impossible example seen by a non-refuser arm) and the seeded validation smoke of the first store configuration against every rival, on each instrument. Either harness may land not ready, in which case its table below is a declared not-run.
- L1948	w16 CMAZE-T seen row, 3 seeds, or a declared negative
- L1948	w16 CMAZE-O seen row, 3 seeds, or a declared negative
- L1949	w16 CMAZE-T 2-way composition: store vs rivals, or a declared negative with its failing link
- L1949	w16 CMAZE-O 2-way composition: store vs rivals, or a declared negative with its failing link
- L1950	w16 CMAZE-T $n$-way composition, or a declared negative with its failing link
- L1950	w16 CMAZE-O $n$-way composition, or a declared negative with its failing link
- L1951	w16 CMAZE-T rollout validity over chained generated states, or a declared negative
- L1952	w16 CMAZE-T never-seen colour (scored, no refusal display), or a declared negative
- L1952	w16 CMAZE-O never-seen colour, or a declared negative
- L1953	w16 CMAZE-T charter-C controls (must collapse for a store row), within- and cross-maze lookup floors, BLIND columns; else interface row or nothing
- L1953	w16 CMAZE-O controls, lookup floor, BLIND columns; else interface row or nothing
- L1954	w16 CMAZE-T value deletion: exactness, post-deletion composition vs retrain-without, cost; or a declared negative
- L1954	w16 CMAZE-O value deletion, same columns; or a declared negative
- L1955	w16 CMAZE-T conjunction deletion, same columns; or a declared negative
- L1955	w16 CMAZE-O conjunction deletion, same columns; or a declared negative
- L1960	w16 \texttt{maze-wm} configuration ladder and budget: which configuration (the Shapes recipe unchanged; the same with the task-shaped interface given to every rival; a $K$/rank re-sweep on the maze latent with item and field-store ablations) passed validation and was scored on each task, and how many of the 18 scored runs each consumed. It may be that no configuration passes on either task.
- L1965	w16 fallback: if neither CMAZE-T nor CMAZE-O yields a win, the route task on CMAZE-2b with the store against learned rivals only, the merge of \S\ref{sec:verb} as the shared interface and the lookup printed as the floor beside the cross-maze fragment-lookup column; this too may land as a declared negative.
- L1969	w16 declared negatives: for every task and configuration that did not yield a row, the negative stated with its mechanism (which link failed: content, factorisation, read or interface). If there are none, this sentence is removed.
- L2008	w15 \texttt{bar-d-score} (one \texttt{encoder-loop} scored run, new pre-registration): BAR D on the longer, full-decoder leg, i.e.\ the store's margin over the matched-parameter learner inside each encoder arm and whether it grows from \texttt{alone} to \texttt{loop\_swap}, 3 seeds, in a table never merged with the CPU table; and the decomposition paragraph (unfreezing hurts / swap loss repairs) re-stated on that leg. It may land as a negative: the margin may not grow, in which case the paragraph states that the features improved for every arm.

## TODOs (4)
- L447	no artefact for the paired $\Kstar$ ladder margin (with CI) against the read-path-matched rivals; the $\Kstar$ replication pairs the store only against the $1\times$ opponent of Appendix~\ref{app:1x}.
- L622	no artefact for the ``$\sim$1/7$\times$'' byte ratio of the composition-only store named in the task spine; the artefacts give the two totals printed here and in Table~\ref{tab:ledger-kstar}, and this spoke does not divide them.
- L963	AI use statement --- the Head to confirm which of the venue's required disclosure categories apply. This draft, the evidence pipeline and most of the code were produced with substantial LLM-agent assistance (implementing methods, designing and critiquing experiments, interpreting results, drafting the paper); the statement must say so accurately.
- L981	anonymised code archive not yet built.

## Flags for the Hub / Head / referee-3
1. ⚠ **`fig_ladder` panel (b)** (plots spoke) plots the margin vs the **1×-total-bytes learner**, not vs the read-path rivals the main text uses. The caption says so explicitly. Recommend plots / wave 18 add (or swap to) a read-path-rival margin panel — or the Head moves the figure beside App. C.
2. ⚠ **TABLES.md T10** presents the refuse-token comparator's off-grid-included numbers without the primary-set numbers (D10). A tables fix for wave 18.
3. ⚠ **The paired K\* ladder margin vs the read-path rivals (with CI) does not exist in any artefact** (the K\* replication paired only against the 1× learner). The main text prints both sides' scores and the K = 5 paired margins, and marks the gap `\TODO`. Zero-run arithmetic on per-corner files would close it — not this spoke's to derive.
4. ⚠ **Venue:** VENUE.md reports the ICLR 2027 abstract deadline (Sep 18) as passed. Not acted on.
5. The theory section carries shorts-derived laws as "confirmed [prior derivation]" with their riders; the programme never re-ran them on this store (status discipline in §5.3 / App.).
6. `_thinkpad` appears only inside LaTeX source comments (evidence paths), never in the PDF.

## What did NOT run / was not done (DECLARED NOT-DONE)
- No `\cite{}` inserted (cite-tagger's job, after this END_UTC). Related-work prose names the works so keys can be slotted.
- No bibliography compiled (no `references.bib` exists yet).
- AI-use statement content: `\TODO` for the Head (facts belong to the authors).
- Anonymised code archive: not built (`\TODO`).
- No number re-derived; the four places where a derived quantity would have been convenient (paired K\* vs read-path rivals; "~1/7×"; per-seed CI at K\*; T7 1× column) are TODOs or appendix pointers, not numbers.
- Style paraphrase, bib, figures, referee: not this spoke's files; not touched.
