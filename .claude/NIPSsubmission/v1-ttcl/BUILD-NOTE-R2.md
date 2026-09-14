# BUILD-NOTE-R2 — Hopfield scoreboard removal (`pj_sub.tex`)

Pass: `v1-scoreboard-removal` (paper-writer spoke), 2026-08-27. Task: `.claude/tasks/v1-scoreboard-removal.md`.

## Pin check

| | value |
|---|---|
| required md5 at spawn | `bb98439d4dfdbfc279aa2988e0ecc5b8` |
| observed md5 at spawn | `bb98439d4dfdbfc279aa2988e0ecc5b8` — **PASS** |
| pre-pass size | 410 lines / 6,450 tex-words |
| post-pass md5 | `da1b067b920b0f300b8f774bdc1b1506` |
| post-pass size | 384 lines / 6046 tex-words |
| files written | `pj_sub.tex` (only), plus this note |

Method: one scripted single-occurrence substitution per site, each asserting `count == 1` before writing (`.claude/scratch/v1-scoreboard-removal/apply.py`). 15/15 sites matched exactly once. No regex-wide edits, no reflow.

## Edits, before → after

### R5 · `R5b-abstract`
**Ancestor:** abstract, final sentence

**BEFORE**
```tex
demonstrating memory-agnostic calibrated compute-rationing, evaluating scaling boundaries for non-local routing against a physics-free baseline, and mapping a settled performance regime against modern Hopfield networks.
```

**AFTER**
```tex
demonstrating memory-agnostic calibrated compute-rationing and evaluating scaling boundaries for non-local routing against a physics-free baseline.
```

### R5 · `R5a-contribution`
**Ancestor:** Contributions list, bullet 6

**BEFORE**
```tex
\item \textbf{Regime mapping against standard baselines:} We establish a settled comparative regime map. Modern Hopfield networks provide a more cost-effective and noise-robust retrieval baseline at matched accuracy. The proposed CLU gate achieves parity only on clean and correlated cues at small capacity limits, demonstrating that the gate operates as a tool for clean retrieval while generalized noise robustness remains a property of the Hopfield architecture.
```

**AFTER**
```tex
\item \textbf{Capacity and epoch-budget map of the compute dial:} We map the rationing gate across capacity and training budget. The measured savings ($9.9\times, 9.5\times, 6.2\times$ across kv32, kv64, kv96) are intra-CLU rationing against a full-budget CLU baseline, performance beyond kv64 is constrained by epoch-budgets rather than hard capacity limits, and gate accuracy degrades sharply under evaluation cue noise while storage fidelity remains at $1.0$.
```

### R1 · `R1-move-into-4.1`
**Ancestor:** new paragraph appended to §4.1 (Memory-Agnostic Calibrated Compute-Rationing), after its \end{enumerate}

**BEFORE**
```tex
\end{enumerate}

\subsection{Boundary Analysis for Non-Local Routing}
```

**AFTER**
```tex
\end{enumerate}

We map the same gate across capacity and training budget on a 198-job evaluation grid (Appendix C.3). Storage fidelity converges to near-perfect levels across training epochs, indicating that the initial underperformance was an under-training artifact. Performance degradation beyond kv64 is constrained by epoch-budgets rather than hard capacity limits; training models to 4000 epochs, however, causes smaller cells to over-train, degrading accuracy from $1.00$ to $0.89$. The measured savings of the CLU gate ($9.9\times, 9.5\times, 6.2\times$ across kv32, kv64, kv96) represent intra-CLU rationing against a full-budget CLU baseline. Under evaluation cue noise ($\sigma\in\{0.3,0.6\}$), the gate degrades sharply: at $\sigma=0.6$ on kv32, the CLU gate yields an accuracy of $0.36$, despite underlying CLU storage fidelity remaining at $1.0$. The governed relaxation over-commits to the corrupted cue, and we record this noise wall as an open problem.

\subsection{Boundary Analysis for Non-Local Routing}
```

### R4 · `R4-figure2`
**Ancestor:** Figure 2 environment (figs/fig2_regime_map.png) + caption + \label{fig:regime}

**BEFORE**
```tex
\begin{figure}[t]\centering
\includegraphics[width=\textwidth]{figs/fig2_regime_map.png}
\caption{\textbf{Regime Mapping against Hopfield Baseline (\S4.3).} Evaluated at 2000 epochs across 198 jobs. \emph{(a)} Storage fidelity converges to near-perfect levels across training epochs, indicating that the initial underperformance was an under-training artifact. \emph{(b)} On clean cues, the CLU gate marginally exceeds the Hopfield baseline for capacities kv$\le64$ ($\Delta+0.02$). The indicated computational savings ($9.9\times, 9.5\times, 6.2\times$) represent intra-CLU rationing, not a comparative win over the Hopfield network. \emph{(c)} \textbf{The noise wall constraint:} Under evaluation cue noise ($\sigma\in\{0.3,0.6\}$), the CLU gate strictly underperforms the Hopfield baseline across all capacities, despite maintaining perfect storage fidelity. The governed CLU relaxation trajectory over-commits to the corrupted initial cue, confirming that noise-robustness remains a property of the Hopfield architecture.}
\label{fig:regime}
\end{figure}
```

**AFTER**
```tex
(deleted — nothing substituted)
```

### R1 · `R1-dissolve-4.3`
**Ancestor:** §4.3 heading 'Settled Regime Mapping vs. Baseline Retrievers' + its two body paragraphs

**BEFORE**
```tex
\subsection{Settled Regime Mapping vs. Baseline Retrievers}
We construct a comprehensive 198-job evaluation grid comparing the governed CLU gate against a modern Hopfield network. The Hopfield baseline achieves optimal recall in approximately one matrix-vector multiplication, yielding an accuracy range of $0.947$ to $0.979$ at a computational floor of $O(\text{kv}\cdot d)$. The measured savings of the CLU gate ($9.9\times, 9.5\times, 6.2\times$ across kv32, kv64, kv96) represent intra-CLU rationing against a full-budget CLU baseline, not a comparative computational win over the Hopfield network. The Hopfield baseline remains the strictly cheaper retriever at matched accuracy.

The evaluation yields three distinct operational bounds. First, for clean and correlated cues at low capacity (kv$\le64$), the CLU gate matches or marginally reverses the Hopfield baseline ($\Delta+0.02$). Second, performance degradation beyond kv64 is constrained by epoch-budgets rather than hard capacity limits. Training models to 4000 epochs allows kv96 to marginally surpass the Hopfield baseline ($\Delta+0.03$), although smaller cells begin to over-train, degrading accuracy from $1.00$ to $0.89$. Finally, under the dominant constraint of cue noise ($\sigma\in\{0.3,0.6\}$), the CLU gate fails to achieve parity with the Hopfield baseline at any capacity limit. At $\sigma=0.6$ on kv32, the CLU gate yields an accuracy of $0.36$ against Hopfield's $0.71$, despite underlying CLU storage fidelity remaining at $1.0$. The governed relaxation over-commits to the corrupted cue. Therefore, the CLU rationing gate acts as a tool for efficient clean retrieval, while global noise-robustness remains a core asset of the Hopfield architecture.
```

**AFTER**
```tex
(deleted — nothing substituted)
```

### R2 · `R2-lead-line`
**Ancestor:** Appendix C.3 lead line

**BEFORE**
```tex
Evaluated with zero correlation across 8 pooled seeds. $\Delta$ represents the CLU gate accuracy minus the Hopfield accuracy.
```

**AFTER**
```tex
Evaluated with zero correlation across 8 pooled seeds. The grid characterizes the capacity and epoch-budget axes of the compute dial of \S4.1.
```

### R2 · `R2-colspec`
**Ancestor:** Appendix C.3 tabular column specifier

**BEFORE**
```tex
\begin{tabular}{@{}lccccccc@{}}
```

**AFTER**
```tex
\begin{tabular}{@{}lccccc@{}}
```

### R2 · `R2-header`
**Ancestor:** Appendix C.3 header row

**BEFORE**
```tex
Cell (N,kv) & Epochs & CLU Fidelity & Gate Acc & Hopfield Acc & $\Delta$ & Intra-CLU Savings \\
```

**AFTER**
```tex
Cell (N,kv) & Epochs & CLU Fidelity & Gate Acc & Intra-CLU Savings \\
```

### R2 · `R2-row1`
**Ancestor:** Appendix C.3 data row 1

**BEFORE**
```tex
N128/kv32 & 500 & $0.76\pm0.09$ & $0.31\pm0.04$ & $0.98\pm0.01$ & $-0.67$
```

**AFTER**
```tex
N128/kv32 & 500 & $0.76\pm0.09$ & $0.31\pm0.04$
```

### R2 · `R2-row2`
**Ancestor:** Appendix C.3 data row 2

**BEFORE**
```tex
N128/kv32 & 2000 & $1.00\pm0.00$ & $1.00\pm0.00$ & $0.98\pm0.01$ & $+0.02$
```

**AFTER**
```tex
N128/kv32 & 2000 & $1.00\pm0.00$ & $1.00\pm0.00$
```

### R2 · `R2-row3`
**Ancestor:** Appendix C.3 data row 3

**BEFORE**
```tex
N256/kv64 & 500 & $0.43\pm0.05$ & $0.06\pm0.02$ & $0.97\pm0.01$ & $-0.91$
```

**AFTER**
```tex
N256/kv64 & 500 & $0.43\pm0.05$ & $0.06\pm0.02$
```

### R2 · `R2-row4`
**Ancestor:** Appendix C.3 data row 4

**BEFORE**
```tex
N256/kv64 & 2000 & $1.00\pm0.00$ & $0.99\pm0.00$ & $0.97\pm0.01$ & $+0.02$
```

**AFTER**
```tex
N256/kv64 & 2000 & $1.00\pm0.00$ & $0.99\pm0.00$
```

### R2 · `R2-row5`
**Ancestor:** Appendix C.3 data row 5

**BEFORE**
```tex
N384/kv96 & 500 & $0.40\pm0.03$ & $0.02\pm0.01$ & $0.95\pm0.01$ & $-0.93$
```

**AFTER**
```tex
N384/kv96 & 500 & $0.40\pm0.03$ & $0.02\pm0.01$
```

### R2 · `R2-row6`
**Ancestor:** Appendix C.3 data row 6

**BEFORE**
```tex
N384/kv96 & 2000 & $0.97\pm0.01$ & $0.91\pm0.02$ & $0.95\pm0.01$ & $-0.04$
```

**AFTER**
```tex
N384/kv96 & 2000 & $0.97\pm0.01$ & $0.91\pm0.02$
```

### R3 · `R3-C4`
**Ancestor:** Appendix C.4 'Regime Map: Stress Axes' subsection (lead, tabular, both commentary paragraphs)

**BEFORE**
```tex
\subsection*{C.4 Regime Map: Stress Axes}
Evaluated at 2000 epochs with 5 seeds.

\noindent{\footnotesize
\begin{tabular}{@{}lcccccc@{}}
\toprule
Cell (N,kv) & Correlation $\rho$ & Gate Acc & Hopfield Acc & $\Delta$ & Intra-CLU Savings \\
\midrule
N128/kv32 & $0.9$ & $0.87\pm0.06$ & $0.72$ & $+0.16$ & $5.6\times$ \\
N256/kv64 & $0.9$ & $0.67\pm0.07$ & $0.59$ & $+0.08$ & $2.6\times$ \\
N384/kv96 & $0.9$ & $0.36\pm0.05$ & $0.52$ & $-0.16$ & $1.3\times$ \\
\bottomrule
\end{tabular}}

At extreme correlation ($\rho=0.9$), the Hopfield baseline collapses heavily, causing the performance gap to widen mathematically, though this indicates Hopfield fragility rather than intrinsic CLU superiority.

Crucially, under evaluation cue noise ($\sigma\in\{0.3,0.6\}$), no cell closes at any capacity, even at kv32. For example, at $\sigma=0.6$ on kv32, the CLU gate manages only an accuracy of $0.36$ against the Hopfield network's $0.71$, resulting in a large deficit ($\Delta=-0.35$). The governed CLU relaxation over-commits to the corrupted initial state.
```

**AFTER**
```tex
(deleted — nothing substituted)
```

## Two-way numeric check

Tokenizer: `\d+(\.\d+)?` over the whole file. Pre: 792 tokens / 210 distinct. Post: 730 tokens / 198 distinct.

### (a) Orphans — numeric tokens in POST with no ancestor in PRE

**ORPHAN LIST: EMPTY (0 tokens)**

Every distinct numeric token in the post-pass file occurs in the pre-pass file. Three tokens rose in *count* (all with pre-pass ancestors, none a new measurement):

| token | pre → post | why |
|---|---|---|
| `4.1` | 1 → 2 | section cross-reference `\S4.1` added to the C.3 lead line (R2 re-home) |
| `3` | 11 → 12 | section cross-reference `Appendix C.3` added in the §4.1 paragraph (R1/R2 re-home) |
| `1.0` | 18 → 19 | the storage-fidelity value `$1.0$` is now stated in the §4.1 paragraph **and** in the rewritten contribution bullet; both carry the same pre-pass value |

### (b) Numbers removed, attributed to the item that removed them

| item | numeric tokens removed |
|---|---|
| `R5b-abstract` | — (none) |
| `R5a-contribution` | — (none) |
| `R1-move-into-4.1` | — (none) |
| `R4-figure2` | 0.02 0.3 0.6 2 4.3 6.2 9.5 9.9 64 198 2000 |
| `R1-dissolve-4.3` | 0.02 0.03 0.3 0.36 0.6 0.6 0.71 0.89 0.947 0.979 1.00 1.0 6.2 9.5 9.9 32 32 64 64 64 96 96 198 4000 |
| `R2-lead-line` | — (none) |
| `R2-colspec` | — (none) |
| `R2-header` | — (none) |
| `R2-row1` | 0.01 0.67 0.98 |
| `R2-row2` | 0.01 0.02 0.98 |
| `R2-row3` | 0.01 0.91 0.97 |
| `R2-row4` | 0.01 0.02 0.97 |
| `R2-row5` | 0.01 0.93 0.95 |
| `R2-row6` | 0.01 0.04 0.95 |
| `R3-C4` | 0.05 0.06 0.07 0.08 0.16 0.16 0.3 0.35 0.36 0.36 0.52 0.59 0.6 0.6 0.67 0.71 0.72 0.87 0.9 0.9 0.9 0.9 1.3 2.6 4 5 5.6 32 32 32 64 96 128 256 384 2000 |

**Values that left the paper entirely (count → 0), with the item that removed them:**

| value | pre count | removed by |
|---|---|---|
| `0.08` | 1 | R3-C4 |
| `0.16` | 2 | R3-C4 |
| `0.52` | 1 | R3-C4 |
| `0.59` | 1 | R3-C4 |
| `0.67` | 2 | R2-row1, R3-C4 |
| `0.71` | 2 | R1-dissolve-4.3, R3-C4 |
| `0.93` | 1 | R2-row5 |
| `0.947` | 1 | R1-dissolve-4.3 |
| `0.979` | 1 | R1-dissolve-4.3 |
| `0.98` | 2 | R2-row1, R2-row2 |
| `1.3` | 1 | R3-C4 |
| `5.6` | 1 | R3-C4 |

**No surviving number changed value, precision, ±, seed count or unit.** Every kept token in C.3 (`Cell`, `Epochs`, `CLU Fidelity`, `Gate Acc`, `Intra-CLU Savings`) is byte-identical to its pre-pass form.

## Residual-comparison sweep (positive-controlled)

Tool: `/usr/bin/grep -o -F` + `wc -l` (occurrence counts, **not** `grep -c`), per the task's grep hazard note —
the shell `grep` here is `ugrep`, which can silently exit 0 on long `.tex` lines.
Each pattern's **pre-pass count is its own positive control**: a post-pass 0 is only trusted where the same
command returned > 0 on the pre-pass file.

| pattern | PRE | POST | positive control | verdict |
|---|---|---|---|---|
| `Hopfield` | 32 | **9** | pre 32 > 0 | 9 survivors, all read in context below |
| `matrix-vector` | 1 | **0** | pre 1 > 0 ✅ | Hopfield cost claim gone (R1) |
| `matvec` | 0 | **0** | ⚠ **false friend** — 0 in *both*; the claim was spelled `matrix-vector multiplication`, controlled by the row above | n/a |
| `\Delta+0.02` | 2 | **0** | pre 2 > 0 ✅ | reversal claim gone (R1 body + R4 caption) |
| `0.947` | 1 | **0** | pre 1 > 0 ✅ | Hopfield accuracy range gone (R1) |
| `0.979` | 1 | **0** | pre 1 > 0 ✅ | Hopfield accuracy range gone (R1) |
| `reverses` | 2 | **1** | pre 2 > 0 ✅ | 1 survivor in Appendix D — **not enumerated for this pass**, see Findings |
| `cheaper` | 1 | **0** | pre 1 > 0 ✅ | "strictly cheaper retriever" gone (R1) |
| `0.18` / `0.88` | 1 / 4 | **1 / 4** | — | ✅ memory-agnostic transfer intact (acceptance criterion) |
| `198-job` | 1 | **1** | — | grid size survives, re-homed to §4.1 |
| `fig2_regime_map` | 1 | **0** | pre 1 > 0 ✅ | figure deleted (R4) |
| `fig:regime` | 1 | **0** | pre 1 > 0 ✅ | label deleted with it |
| `ref{` | 0 | **0** | ⚠ 0 in both — the document contains **no `\ref` at all**; `\label{fig:regime}` had zero referents, so R4 left no dangling reference | n/a |

### The 9 surviving `Hopfield` hits, read in context

| line | site | status |
|---|---|---|
| 132 | §4.1 bullet 1, memory-agnostic transfer: *"…applied to a baseline modern Hopfield memory (Ramsauer et al. 2021) yields analogous improvements (raw `$0.18\to$` calibrated `$0.88$`…)"* | ✅ **must survive** — the disclosure that the gate mechanism is not CLU-specific. Untouched. |
| 148 | §5 Position/Scope: *"…do not universally dominate simpler neural routing heuristics or baseline Hopfield retrievers under noisy cue constraints."* | ⚠ not enumerated → untouched. **Finding for the Head.** |
| 202 | App A.2 flag table, row *"Hopfield baseline & Platt-calibrated logit margin, identically probed"* | ✅ provenance for the surviving line-132 transfer result. Untouched. |
| 232 | App A.4 flag table, row *"Hopfield configuration & `$\beta\in\{2,5,20\}$`; iteration sweep…"* | ⚠ not enumerated → untouched. **Finding for the Head.** |
| 347 (×4) | App D "Documented Negative Results", second finding: abstention rates, the `reverses` clause, the noise-robustness clause | ⚠ not enumerated → untouched. **Finding for the Head.** |
| 374 | References, Ramsauer et al. 2021 | ✅ still cited by line 132. Untouched. |

## Build

`tectonic 0.15`-class binary at `/opt/homebrew/bin/tectonic` (no `pdflatex`/`latexmk` on this machine).
Both builds run in isolated scratch copies (`.claude/scratch/v1-scoreboard-removal/build/{pre,post}/`) with the
same `neurips_2026.sty` and `figs/` — so pre and post are toolchain-matched.

| | errors (`^!`) | undefined refs | pages | main text | appendix |
|---|---|---|---|---|---|
| PRE (`bb98439d…`) | **0** | **0** | **14** | pp. 1–8 | pp. 9–14 (6 pp) |
| POST (`da1b067b…`) | **0** | **0** | **13** | pp. 1–8 | pp. 9–13 (5 pp) |

- Page split measured by temporary `\label`s injected **into the scratch copies only** (`\appendix`, `\end{document}`);
  the shipped `pj_sub.tex` carries no such labels.
- The single `not found` log line (`\pdfdraftmode not found`, pdftexcmds) is present identically in both builds; benign.
- The shipped `pj_sub.pdf` also reports 14 pages, consistent with the PRE rebuild.
- ⚠ **`pj_sub.pdf` in the submission directory was NOT regenerated** — the task scopes this pass to `pj_sub.tex`
  and nothing else. It is now one revision stale. Regenerate with
  `cd .claude/NIPSsubmission/v1-ttcl && tectonic pj_sub.tex`.

## Diff shape

`diff -u` pre → post: **4 hunks, 53 changed lines**, all 15 accounted for by items R1–R5 above.
Zero unenumerated diffs. `submission.tex` (`caef2272f9dc96d349b46486563d24ee`) and every file under
`.claude/papers/v1-short/` are byte-untouched (md5 manifest in `.claude/outputs/v1-scoreboard-removal.md`).

## Judgement calls made inside the enumerated scope

1. **R1 — where the moved content landed.** As one new paragraph immediately after §4.1's `\end{enumerate}`,
   with no new heading (a new `\paragraph{}` would have been restructuring). Sentences are carried verbatim
   with the comparative clause excised; connective prose is limited to the opener
   *"We map the same gate across capacity and training budget on a 198-job evaluation grid (Appendix C.3)."*
   — which would otherwise leave the moved sentences without a subject — and the closing
   *"…and we record this noise wall as an open problem."* (the Head-ruled open-problem framing). No intensifiers,
   no new claim, no new number.
2. **R1 — the epoch-budget wall is stated once, not twice.** Main text carries the verbatim §4.3 sentence
   (*"Performance degradation beyond kv64 is constrained by epoch-budgets rather than hard capacity limits"*)
   plus the 4000-epoch over-training clause. The finer form the task quotes (kv32 saturates by 1000, kv96 by 4000)
   **already lives verbatim in App C.3's epoch-frontier prose**, which R2 keeps and which §4.1 now points at.
   Duplicating it into the main text would have introduced a second statement of the same numbers. Flagged for
   the Hub in case the Head wants it in the main text too.
3. **R2 — "re-home under §4.1's material" executed as a pointer, not a move.** Appendix C.3 keeps its position
   and heading (physically re-ordering appendix subsections is barred by §0), and is re-homed by wording:
   its lead line now reads *"The grid characterizes the capacity and epoch-budget axes of the compute dial of §4.1"*,
   and §4.1 cites it. Flagged in case a physical move was intended.
4. **R2 — column specifier.** The pre-pass spec was `@{}lccccccc@{}` = 8 specifiers for 7 columns (one spare,
   pre-existing). Exactly two `c`s were deleted → `@{}lccccc@{}` = 6 for 5 columns, preserving the pre-existing
   spare rather than silently "fixing" it.
5. **R5 — the contribution bullet was rewritten, not deleted,** so the contributions count stays at **6** and no
   prose renumbering was needed. It now states only the CLU-internal dial result (savings scoped to the
   full-budget CLU baseline, the epoch-budget wall, the noise wall). No count of contributions is stated
   anywhere in the text (`six` returns 0 occurrences pre and post), so nothing downstream depends on it.
6. **R4 — no salvage of panel (a) attempted.** Re-rendering a fidelity-only figure is a `results-analyst` pass.
   The whole `figure` environment, caption and `\label` are gone; nothing referenced the label.
