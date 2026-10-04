# v1-scoreboard-removal — paper-writer report
Task + acceptance criterion: remove the Hopfield scoreboard from `.claude/NIPSsubmission/v1-ttcl/pj_sub.tex`, re-homing §4.3's CLU-internal content into §4.1; diff must contain **only** the enumerated changes R1–R5, orphan list empty, build clean.
Status: **done** (all 5 items executed; 0 unenumerated diffs; orphan list EMPTY; build 0 errors / 0 undefined refs).

## ⚠ DOWNSTREAM RECONCILIATION LIST (needs an owner — protocol §5 corollary, stated in the first 10 lines)
1. **`claims_matrix.md` CM-8** (line 566) is sourced *"V1 §4.3 (final wording)"*. **V1 has no §4.3 regime map any more** — its comparative half (Hopfield cost ceiling, Δ+0.02 reversal, Δ+0.03 at kv96, "Hopfield keeps cost AND noise-robustness") is now asserted **nowhere in V1**. Needs a `doc-curator` pass to re-point/rescope the row. ⛔ I did not touch the matrix.
2. **`pj_sub.pdf` in the submission dir is one revision stale** (task scoped me to `pj_sub.tex` and nothing else). Owner: whoever ships. Command: `cd .claude/NIPSsubmission/v1-ttcl && tectonic pj_sub.tex`.
3. **App A.4's heading `\S4.3 Regime Map Configuration` now points at the wrong section** — LaTeX auto-renumbering moved *Matched-Compute Anytime Read* from §4.4 to §4.3 (verified: label test shows `4.3` pre → `4.2` post for the subsection preceding it). Not enumerated → untouched. Needs one Head-authorized word (`\S4.1`, or drop the pointer).
4. **Three comparative sites survive because they were not enumerated** (§5 Position/Scope; App A.4 Hopfield-config row; App D "Documented Negative Results" ×4 Hopfield mentions incl. the surviving `reverses` clause). See Findings §F1 — the Head must rule whether the scoreboard removal extends to them.

## DIAL DECLARATION (echoed)
**Dials touched: NONE.** Manuscript-only pass: one `.tex` file, removal of a comparison + re-homing of CLU-internal content + deletion of one figure and one appendix table. No experiment, no configuration, no new measurement. Laundering control: n/a (no performance number produced). Falsifies: n/a.

## Pin check (§0) — PASSED
- Required `bb98439d4dfdbfc279aa2988e0ecc5b8`; observed `bb98439d4dfdbfc279aa2988e0ecc5b8`; 410 lines / 6,450 tex-words. Match ⇒ proceeded.
- Post-pass: `da1b067b920b0f300b8f774bdc1b1506`, 383 lines / 6,046 tex-words.
- Pre-pass snapshot preserved at `.claude/scratch/v1-scoreboard-removal/pj_sub.PRE.tex` (same md5) for independent review-time diffing.

## What I did
- **Method:** a single scripted pass (`.claude/scratch/v1-scoreboard-removal/apply.py`) with **15 single-occurrence substitutions, each asserting `count == 1` before writing** (a 0- or 2-match pattern exits without writing). All 15 matched exactly once. Edit log serialized to `edits.json`; `BUILD-NOTE-R2.md` is generated from that log, so its before/after blocks are the literal strings the script matched.
- **R1** — dissolved `\subsection{Settled Regime Mapping vs. Baseline Retrievers}` (heading + both body paragraphs). Deleted: the Hopfield one-matrix-vector cost claim and its `0.947`–`0.979` range and `O(\text{kv}\cdot d)` floor; *"The Hopfield baseline remains the strictly cheaper retriever at matched accuracy"*; the `Δ+0.02` "matches or marginally reverses" clause; the kv96 `Δ+0.03` surpassing clause; *"noise-robustness remains a core asset of the Hopfield architecture"*; the three-"operational bounds" comparative framing.
- **R1 (move)** — one new paragraph at the end of §4.1 carrying, verbatim where the sentence allowed: storage-fidelity convergence + *"the initial underperformance was an under-training artifact"*; *"Performance degradation beyond kv64 is constrained by epoch-budgets rather than hard capacity limits"*; the 4000-epoch **over-training** result (`1.00 → 0.89`); the savings `9.9\times, 9.5\times, 6.2\times` across kv32/kv64/kv96 **with the "against a full-budget CLU baseline" scope kept**; and the noise wall stated comparison-free (`0.36` at σ=0.6/kv32, fidelity `1.0`, "governed relaxation over-commits to the corrupted cue") **named as an open problem, with no Hopfield number beside it**.
- **R2** — App C.3: deleted the `Hopfield Acc` and `$\Delta$` columns from the header, all six data rows and the `tabular` specifier; rewrote the lead line (the Δ definition died with the column) to re-home the table under the §4.1 dial. `Cell`/`Epochs`/`CLU Fidelity`/`Gate Acc`/`Intra-CLU Savings` and the epoch-frontier prose are byte-unchanged.
- **R3** — App C.4 "Regime Map: Stress Axes" deleted entirely (ρ table + both commentary paragraphs). Its noise-wall content is **not lost**: it is replaced by R1's comparison-free sentence in §4.1.
- **R4** — Figure 2 (`figs/fig2_regime_map.png`) deleted whole: environment, caption, `\label{fig:regime}`. **No panel-(a) salvage attempted** (a `results-analyst` re-render, per the task's override note). Sweep for surviving references: the document contains **zero `\ref{}` commands at all**, so the label had no referents — nothing repointed, silently or otherwise.
- **R5** — abstract: minimal excision of *"and mapping a settled performance regime against modern Hopfield networks"* (the surrounding sentence is otherwise untouched; the excision forces `, ` → ` and ` between the two surviving gerunds). Contribution bullet 6 rewritten CLU-internally as *"Capacity and epoch-budget map of the compute dial"* — **contribution count stays 6, so no prose renumbering was needed**; `six`/`6 contributions` appear nowhere in the text (0 occurrences pre and post).
- Wrote `.claude/NIPSsubmission/v1-ttcl/BUILD-NOTE-R2.md` (every edit by item ID, before → after, with ancestor; two-way numeric check; sweep; build; judgement calls).

## How I verified
**1. Diff shape.** `diff -u` pre → post = **4 hunks, 53 changed lines**, every one attributable to R1–R5. No typo/grammar/reflow/terminology edits; nothing renumbered by hand; no heading retitled except the enumerated ones.

**2. Two-way numeric check** (tokenizer `\d+(\.\d+)?`; pre 792 tokens / 210 distinct, post 730 / 198):
- **ORPHAN LIST: EMPTY.** Every distinct numeric token in the post-pass file exists in the pre-pass file.
- Three tokens rose in *count*, all with pre-pass ancestors and none a new measurement: `4.1` (1→2, the `\S4.1` cross-reference in the new C.3 lead line), `3` (11→12, the `Appendix C.3` cross-reference in the new §4.1 paragraph), `1.0` (18→19, the fidelity value now stated in both the §4.1 paragraph and the rewritten contribution bullet).
- Values that left the paper entirely, each attributed to its item (full table in the build note): `0.947`, `0.979` (R1); `0.98`, `0.67`, `0.91`, `0.93` (R2 columns); `0.52`, `0.59`, `0.08`, `0.16`, `5.6`, `1.3` (R3); `0.71` (R1 + R3).
- **No surviving number changed value, precision, ±, seed count or unit.**

**3. Residual-comparison sweep, positive-controlled** (`/usr/bin/grep -o -F | wc -l` — never `grep -c`; the shell `grep` here is `ugrep` and is untrustworthy on these long lines):

| pattern | PRE | POST | note |
|---|---|---|---|
| `Hopfield` | 32 | **9** | all 9 read in context, table below |
| `matrix-vector` | 1 | **0** | positive control: pre 1 > 0 |
| `matvec` | 0 | **0** | ⚠ **false friend confirmed live** — 0 in both; the claim was spelled `matrix-vector multiplication`. Controlled by the row above; a `matvec`-only sweep would have produced the prior pass's wrong finding. |
| `\Delta+0.02` | 2 | **0** | positive control: pre 2 > 0 |
| `0.947` / `0.979` | 1 / 1 | **0 / 0** | positive control: pre 1 > 0 each |
| `reverses` | 2 | **1** | survivor in App D — not enumerated, see F1 |
| `cheaper` | 1 | **0** | positive control: pre 1 > 0 |
| `0.18` / `0.88` | 1 / 4 | **1 / 4** | ✅ acceptance criterion: memory-agnostic transfer intact |
| `fig2_regime_map` / `fig:regime` | 1 / 1 | **0 / 0** | positive control: pre 1 > 0 each |
| `\ref{` | 0 | **0** | ⚠ 0 in both — the paper uses no `\ref` at all; hence R4 left no dangling reference |

Surviving `Hopfield` sites, read in context: **L132** §4.1 memory-agnostic transfer `0.18→0.88` (**must survive; untouched**) · **L148** §5 Position/Scope (not enumerated) · **L202** App A.2 flag row *"Hopfield baseline & Platt-calibrated logit margin"* (provenance for L132; legitimate) · **L232** App A.4 flag row *"Hopfield configuration"* (not enumerated) · **L347 ×4** App D negative results (not enumerated) · **L374** References, Ramsauer et al. 2021 (cited by L132).

**4. Build** — `tectonic` at `/opt/homebrew/bin/tectonic` (no `pdflatex`/`latexmk` on this machine). Pre and post built in toolchain-matched isolated scratch copies with the same `.sty` and `figs/`:

| | errors (`^!`) | undefined refs | pages | main text | appendix |
|---|---|---|---|---|---|
| PRE | 0 | 0 | **14** | pp. 1–8 | pp. 9–14 (6 pp) |
| POST | 0 | 0 | **13** | pp. 1–8 | pp. 9–13 (5 pp) |

Page split measured with temporary `\label`s injected **into scratch copies only**. Net −1 page, taken out of the appendix (C.4's table + commentary); the main-text savings (Figure 2 + §4.3 prose, minus the re-homed paragraph) were absorbed inside page 8's existing slack. The shipped `pj_sub.pdf` independently reports 14 pages, consistent with the PRE rebuild. The one `not found` log line (`\pdfdraftmode`, pdftexcmds) is identical in both builds and benign.

**5. Untouched-file manifest** (acceptance criterion):
```
caef2272f9dc96d349b46486563d24ee  .claude/NIPSsubmission/v1-ttcl/submission.tex     (unchanged)
47a15010c3d643f4014d5d616708f5da  .claude/NIPSsubmission/v1-ttcl/pj_sub.pdf         (unchanged, now stale)
8f4a6422a1719d787d59b8d181f6f14d  .claude/NIPSsubmission/v1-ttcl/BUILD-NOTE-R1.md   (unchanged)
f447d3302c8719cb27619a074c876b44  .claude/NIPSsubmission/v1-ttcl/neurips_2026.sty   (unchanged)
a2bc48c0e0c2f3ceab1fa7cf34f655c8  .claude/papers/v1-short/CHANGELOG.md              (unchanged)
00d703d58a15c0cb77051a9c55674684  .claude/papers/v1-short/draft.md                  (unchanged)
208797d113fa9d6efa6de67d05705ea6  .claude/papers/v1-short/draft.tex                 (unchanged)
141f2c37ee8089c814932ce27e5f4fa4  .claude/papers/v1-short/draft.pdf                 (unchanged)
b0cfbf53651ac187bacee0f977d93f1e  .claude/papers/v1-short/fig2_regime_map.png       (unchanged)
+ fig1_certificate / fig4_bibo / fig_frontier_clean / fig_regime_map / paid_access_reach, draft.log — all byte-identical
```
`ls -la` on the submission dir shows only `pj_sub.tex` with a new mtime (plus the new `BUILD-NOTE-R2.md`).

**6. Head-owned edits (§0) — none made.** Verified absent and left absent: machine-precision qualifier, the score sentence, the §A20.5 substrate-scope sentence, the App-F grade line, the `2.2\times10^{-8}` reconstruction bound (the file still carries only `2.2\times10^{-16}` in App E), and the two §4.2 hedges (*"However, we clearly establish the boundary…"* and *"but raw kinetic energy is not the optimal routing signal in this regime"*) — both still present, untouched.

## Findings

**F1 — Comparative material that survives because it was not enumerated (the largest open item).** The scoreboard is gone from the abstract, the contributions, the main body, the figure and the appendix grids, but three sites still carry the comparative story and were outside my worklist:
- **§5 Position, Scope, and Horizon:** *"…they do not universally dominate simpler neural routing heuristics or baseline Hopfield retrievers under noisy cue constraints."* (This one is now the only main-text Hopfield comparison.)
- **App A.4 flag table:** heading `\S4.3 Regime Map Configuration` + row `Hopfield configuration & $\beta\in\{2,5,20\}$; iteration sweep {1,2,3,5,10}`. The β/iteration sweep is now provenance for **no surviving number**; the `\S4.3` pointer is now wrong (see reconciliation item 3). The rest of the row block (epochs, architecture, retrieval, seeds) *is* still the provenance for the surviving C.3 table, so the block must not simply be deleted.
- **App D Documented Negative Results, finding 2:** four Hopfield mentions including *"the CLU gate **reverses** the Hopfield baseline only on clean or correlated cues at low capacities"* — i.e. the Δ+0.02 reversal claim, in prose, still stands in the appendix while the main text no longer states it. This is the sharpest internal inconsistency left in the document. Under Head policy 4 (appendix maximalism) a negative-result appendix is exactly where such a row *should* live — so this may well be deliberate — but it needs an explicit ruling, not an inference.

**F2 — Also referencing §4.3 by number: App A.4's heading** (only site; `\S4.3` occurred twice pre-pass, the other being the deleted Figure 2 caption). Since *Matched-Compute Anytime Read* auto-promoted from §4.4 → §4.3, the A.4 heading now silently mislabels its own contents. I did not repoint it (not enumerated, and prohibition 4 bars renumbering).

**F3 — §4.1's opening flag table (A.2) still lists a Hopfield baseline row, and that is correct.** It is the provenance of the memory-agnostic transfer result the task requires to survive. Flagging it only so a later "clean the Hopfield out" pass does not delete it by pattern.

**F4 — The savings figures now appear twice** (contribution bullet 6 and the §4.1 paragraph), as the pre-pass file did for the §4.1 4.81× figure (bullet 4 + body). Consistent with house style; both carry the "against a full-budget CLU baseline" scope. If the Head prefers the bullet to point rather than restate, that is a one-line trim.

**F5 — Charter compliance of the new/moved prose.** No defensive audit-confession paragraph was added anywhere (**C-1 as reversed 2026-07-07**); the moved material is learned-system evidence and is not labelled verification (**C-2**); scale qualifiers ride in-sentence (every new claim names its kv cells, and §4.1's opening states MQAR/vocab-256/5-seeds while §1 states laptop CPUs) (**C-5**); no certificate claim was moved, so no fine print travelled (**C-6**); the CLU continuity sentence in §1 is untouched; no unpublished sibling short is referenced (**M1/C-8** — I read no other draft). Against the claims matrix: the surviving CLU-internal statements match **CM-8**'s approved wordings (under-training artifact, epoch-budget wall not capacity, kv32 over-trains 1.00→0.89, "6–10× savings = intra-CLU rationing", noise wall at gate 0.36 with fidelity ≈1.0) and **CM-2**'s memory-agnostic wording is preserved verbatim; **CM-3** appears nowhere (the §4.1 disclaimer *"we do not assert that the CLU possesses an inherently superior energy signal"* is intact).

**F6 — Things I noticed and deliberately did not touch** (all outside the worklist): the §4.1 → new-paragraph transition has no subheading (deliberate; a `\paragraph{}` would be restructuring); App C.3's `tabular` spec still carries one spare column specifier, which it did pre-pass (I deleted exactly the two the task named rather than "fixing" the spare); §4.4→§4.3 promotion is LaTeX's, not mine; the abstract's remaining *"against a physics-free baseline"* is §4.2's, not the scoreboard's, and stays.

## Open editorial questions for the Hub/Head
1. **F1/App D:** does the scoreboard removal extend to Appendix D's negative-results prose (the surviving `reverses` clause and the noise-robustness attribution to Hopfield), or does the Head's "show what works and name open problems" ruling deliberately keep the comparison alive as a *negative* result in the appendix? One-word answer sufficient; it is a single sentence-level edit.
2. **F1/§5:** same question for the §5 closing clause *"or baseline Hopfield retrievers under noisy cue constraints"* — currently the last comparative sentence in the main text.
3. **F2/App A.4:** repoint the heading to `\S4.1` and drop the now-unsourced `Hopfield configuration` row, or leave the whole block as historical provenance?
4. **R2 re-home, interpretation check:** I re-homed C.3 *by wording and cross-reference* (lead line now cites the §4.1 dial; §4.1 cites Appendix C.3) and left the subsection in place, because physically re-ordering appendix subsections is barred by §0's re-ordering prohibition. If a physical move (C.3 → immediately after C.2's §4.1-adjacent material, or into a new appendix section) was intended, say so and I will do it as a one-item pass.
5. **Epoch-budget wall, stated once:** the finer form (kv32 saturates by 1000 ep, kv96 needs 4000 ep for 0.975) lives verbatim in App C.3's surviving prose, which §4.1 now points at; the main text carries the coarse form (*"epoch-budgets rather than hard capacity limits"* + the 4000-epoch over-training result). Duplicating the kv32/kv96 epoch numbers into the main text would state the same numbers twice — do you want them promoted anyway?
6. **Figure count:** the paper now has **exactly one figure** — Fig. 1, the certificate stack (`\begin{figure}` count 2 → 1; `\includegraphics` 2 → 1). The other PNGs in `figs/` (`fig4_bibo`, `fig_frontier_clean`, `fig_regime_map`, `paid_access_reach`) were already un-included by `pj_sub.tex` pre-pass. The headline figure is unambiguously Fig. 1. If the Head wants panel (a) of the old Fig. 2 (fidelity vs epochs) back as a standalone, that is the `results-analyst` re-render flagged in the task — I did not attempt it.
7. **Stale `pj_sub.pdf`** — I left it untouched per the "one file and nothing else" bound. Confirm who rebuilds.

## Missing-experiment notes
None. This pass needed no number that did not already exist in the file; nothing was improvised, rounded or smoothed.

## Git footprint
**None.** No tracked file was touched; all work is under `.claude/` (gitignored). `git status` unchanged by this pass. No branch created (protocol §3: research-only agents with nothing to commit).

## Proposed handover updates (for the Hub)
- V1 `pj_sub.tex` is at `da1b067b920b0f300b8f774bdc1b1506` (383 lines, 6,046 tex-words, 13 pages: main pp. 1–8, appendix pp. 9–13). Pre-pass state preserved at `.claude/scratch/v1-scoreboard-removal/pj_sub.PRE.tex` (`bb98439d…`) for review-time diffing; the Head's six in-flight edits were still absent at pass time and remain so.
- **V1 no longer contains a Hopfield scoreboard in abstract, contributions, main body, figures or the C.3/C.4 grids.** §4.3 is dissolved; *Matched-Compute Anytime Read* is now §4.3. Figure 2 and Appendix C.4 no longer exist.
- Open Head rulings needed before V1 is internally consistent: Appendix D's surviving reversal prose, §5's closing comparative clause, and App A.4's `\S4.3` heading + Hopfield-config row (see reconciliation list, items 1–4). Each is a single-site edit; none should be executed by a spoke without an explicit enumeration.
- `claims_matrix.md` CM-8 is now mis-sourced (points at V1 §4.3). Recommend a `doc-curator` line-item.
