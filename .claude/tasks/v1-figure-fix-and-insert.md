# V1 — fix three figures, then insert two of them

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-27.**

**Agent:** `results-analyst` (Bash-capable: renders **and** makes the two `\includegraphics` insertions).
**Writes:** PNGs under `.claude/NIPSsubmission/v1-ttcl/figs/` **and** `pj_sub.tex` — ⛔ **only the two figure environments enumerated in §3, nothing else.**
**Report:** `.claude/outputs/v1-figure-fix-and-insert.md` · **Deliverable #1:** `BUILD-NOTE-R4.md`

---

## 0. ⛔ Pin check

`pj_sub.tex` md5 at scoping = **`08d31733b5648ed6ab4a6bbc5dc07ed8`** (382 lines). **Compute it first; if it differs the Head has edited — STOP and report.** This file has moved five times this session.

⭐ **The prior replot pass's scratch is staged and you should reuse it, not rebuild it:** `.claude/scratch/v1-replot-pass/` holds `new_fig1.py`, `new_fig4.py`, `new_frontier.py`, `tap.py`, `size.py`, `pagesplit.py`, `digests/`, `backup_figs/`. ⛔ **Its `tap.py` digests are the baseline: the values are already proven unchanged, and they must stay that way.**

---

## PART A — three presentation fixes. ⛔ No plotted value may move.

### A1 ⛔⛔ `fig1_certificate.png` — a retired word is back in the paper
Panel (a)'s annotation currently reads **`squeeze priced out of the swept ζ ≤ 2.0`**. ⛔ **`priced` is the exact word a prose pass just removed from the whole document** (8 → 0). The figure re-introduces it in the headline figure.

**Change to exactly:** `squeeze: beyond the swept ζ ≤ 2.0`

⚠ *(This was an Advisor error — the corrected string was agreed with the Head but never written into the prior task file, so the spoke rendered what it was told. Nothing else in fig1 is wrong: title, both panel headings, `ΔV = 0`, and `state-replacing map` ×3 all landed correctly and ⛔ must not be touched.)*

### A2 `fig_frontier_clean.png` — two layout defects
1. ⛔ **The `500` x-tick renders detached below the axis** on **both** panels, as stray text rather than a tick label. Fix the tick placement so all four values (500/1000/2000/4000) sit on the axis as proper ticks.
2. ⚠ **The y-axis labels wrap awkwardly** (`CLU-EBM storage / fidelity`, `CLU gated / accuracy`), and the left panel's title wraps to two lines while the right's does not. Make the two panels visually symmetric.

✅ **Leave correct:** the Hopfield band is gone and stays gone; the right title is `Gated acc vs epochs`; the left title keeps `3 seeds`; all four kv curves and both panels remain.

### A3 `fig4_bibo.png` — legend overlaps the data
The two-line title compressed the axes and the legend box now sits over the **b = 1.0–3.0** curve region. Move or shrink the legend so no curve is occluded. ✅ Everything else in fig4 is correct and ⛔ must not change — title, the three `screen`/`state-replacing` legend entries, and the red annotation with **`free`** intact.

⛔ **Re-run `tap.py` on all three and prove the digests still match the staged baseline.** A presentation fix that moves a value is a failure, not a fix.

---

## PART B — type and footprint

- Type targets **at printed size**: ticks ≥ 7 pt, axis labels and legend ≥ 8 pt, titles ≥ 9 pt effective.
- Measure printed boxes from a **built PDF with `mutool`**, ⛔ never from `\linewidth` arithmetic. Report deltas; a prior pass in this estate found a one-character title change moved a canvas by 0.93 pt.

---

## PART C — insert two figures into `pj_sub.tex`

⛔ **Exactly two `figure` environments. No other edit to the file, of any kind.**

### C1 — `fig4_bibo.png` into Appendix B.2
**Insert after** `\subsection*{B.2 The BIBO Battery}` (currently l.270), before its existing prose.

```latex
\begin{figure}[t]\centering
\includegraphics[width=0.68\linewidth]{figs/fig4_bibo.png}
\caption{\textbf{The BIBO battery (App.~B.2, verification grade).} Maximum excursion radius
$r^\ast=\max_t\|q_t\|$ at horizon $2T$, against the requested exit locus $b$. Exits inside the
coercive component stay bounded for every arm. Beyond the coercive edge $x_b=3.54$ the
unscreened arms grow as $r^\ast\propto T$, while the screened wormhole holds at $r^\ast=0.09$
because it \emph{refuses} the jump rather than making it safe. The state-replacing map
coincides exactly with the screen-ignored ablation, so what buys boundedness is the screen and
not the jump. Note $b=5.0$: the energy change is exactly zero and the exit still escapes.}
\label{fig:bibo}
\end{figure}
```

### C2 — `fig_frontier_clean.png` into Appendix C.3
**Insert after** `\subsection*{C.3 Regime Map: Capacity Axis and Epoch Frontier}` (currently l.326), before its table.

```latex
\begin{figure}[t]\centering
\includegraphics[width=0.82\linewidth]{figs/fig_frontier_clean.png}
\caption{\textbf{Epoch-scaling frontier (App.~C.3, evidence grade; $n=3$ seeds).} CLU-EBM
storage fidelity and gated accuracy against training budget for kv$\in\{32,64,96,128\}$.
Required epochs scale with capacity rather than saturating at a fixed ceiling, and kv$32$
over-trains between $2000$ and $4000$ epochs. ⛔ Note this panel is $n=3$, not the $8$ pooled
seeds of the capacity axis below.}
\label{fig:frontier}
\end{figure}
```

⚠ **Strip the `⛔` glyph** — it is a task-file marker, not paper text. Render that final sentence as plain prose.

⛔ **These captions are the Advisor's draft and the Head may amend them at spawn. Render exactly what the task says; ⛔ do not improve, extend, or re-word them.**

⚠ **Placement note, report do not act:** the frontier is the compute dial's epoch story and could sit in §4.1 main text instead. It is placed in the appendix because main text already runs ~8.3 pp against a ~5 pp target. Flag the alternative; the Head decides.

---

## Deliverables

1. **`BUILD-NOTE-R4.md`** — every string changed (before → after), every layout change, both insertions quoted, and the `tap.py` digest table proving values unmoved.
2. **Printed-box measurements** before and after, all three figures.
3. **Build**: 0 errors, 0 undefined references; total pages and the main-text split before and after.
4. **A residual sweep, positive-controlled**: `priced` · `receipt` · `ledger` · `no-physics router` — expected **0** in every PNG and in the two new captions.
5. **Any caption edit you believe is needed but did not make** — listed, not applied.

## Acceptance criteria

- Pin check passed, or aborted.
- `diff` on `pj_sub.tex` shows **exactly two hunks**, both figure environments. ⛔ **Zero other changes** — checked independently at review.
- ⛔ `submission.tex` and `.claude/papers/v1-short/**` byte-untouched (md5 manifest printed).
- `tap.py` digests match the staged baseline for all three figures.
- `priced` appears in **no** PNG.

## ⚠ Grep hazards

⛔ `grep` here is a shell function resolving to **`ugrep 7.5.0`**: on bounded-context patterns over long lines it either **errors "exceeds complexity limits" and exits 0** — a silent false negative — or **hangs**. Use **`/usr/bin/grep`**; count with `grep -o … | wc -l`, ⚠ **never `grep -c`** (lines, not occurrences). ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — sweep per-file. ⚠ zsh does not word-split: quote any glob. **Positive-control every negative.**

## DIAL DECLARATION
**Dials touched: NONE.** Presentation-only re-renders from banked data plus two figure insertions. No experiment, no configuration change, and — proven by data tap — no plotted value moved.
