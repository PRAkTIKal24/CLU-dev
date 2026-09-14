# V1 — remove the Hopfield scoreboard; re-home its CLU-internal content into §4.1

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-27.** The Head has ruled: **§4.1 stays and is presented as a CLU inference-time adaptable compute dial with no comparative claim; the Hopfield scoreboard goes.**

**Agent:** `paper-writer` · **Edits `.claude/NIPSsubmission/v1-ttcl/pj_sub.tex` and nothing else.**
**Deliverable #1:** `.claude/NIPSsubmission/v1-ttcl/BUILD-NOTE-R2.md` · **Report:** `.claude/outputs/v1-scoreboard-removal.md`

---

## 0. ⛔⛔ BOUNDING RULE, and a live-file hazard

**Every edit is enumerated in §2. Anything not enumerated is FORBIDDEN, however beneficial it looks.** A pass in this program once executed an entire audit inventory faithfully and additively and **was reverted in full**, because volume alone read as a rewrite. The Head has approved *this* list and only this list.

⛔ **Specifically forbidden however tempting:** typo fixes, grammar, capitalisation, rewording, re-ordering, re-wrapping, terminology harmonisation, and any touch to a number, caption, label or heading not named below.

⛔⛔ **THE HEAD IS EDITING THIS FILE AND SIX OTHER EDITS ARE THEIRS, NOT YOURS.** They are separately adding: the machine-precision qualifier, the score sentence, the §A20.5 substrate-scope sentence, the App-F grade line, the `2.2\times10^{-8}` reconstruction bound, and the removal of two hedges in §4.2. ⛔ **Do not add, pre-empt, or "helpfully" insert any of these.** If you notice one is missing, that is expected — it is in flight elsewhere.

⛔ **PIN CHECK — ABORT AND REPORT IF IT FAILS.** At spawn, `pj_sub.tex` md5 = **`bb98439d4dfdbfc279aa2988e0ecc5b8`** (410 lines, 6,450 tex-words). **Compute it first. If it differs, the Head has edited since scoping — STOP, write the report saying so, change nothing.** This file has already moved once mid-session, and a prior pass in this program was nearly run against a stale baseline.

⚠ **Line numbers below are indicative only and WILL shift as you delete.** ⛔ **Locate every site by content, never by line number.**

---

## 1. Why (so your judgement calls land right)

The Hopfield comparison is V1's only external scoreboard, and the paper loses it: cheaper on cost (final), 0/6 under cue noise, below at kv96 clean; the single win is `Δ+0.02` in one named regime. The Head's ruling is that a short should show what works and name open problems, not stage a scoreboard against a one-shot retriever the CLU was never built to beat.

⭐ **But most of §4.3's *data* is not a comparison — it is the dial's own characterisation, and it must survive.** Fidelity convergence, the epoch-budget wall, and the intra-CLU savings are all CLU-internal. **Your job is a separation, not a deletion.**

---

## 2. THE ENUMERATED WORKLIST

### R1 — dissolve §4.3, moving its CLU-internal content into §4.1
Locate `\subsection{Settled Regime Mapping vs. Baseline Retrievers}` (1 occurrence).

**⛔ DELETE (comparative):** the Hopfield cost claim (*"achieves optimal recall in approximately one matrix-vector multiplication… accuracy range $0.947$ to $0.979$ at a computational floor of $O(\text{kv}\cdot d)$"*) · *"The Hopfield baseline remains the strictly cheaper retriever at matched accuracy"* · the `Δ+0.02` reversal and *"matches or marginally reverses"* · the kv96 `Δ+0.03` surpassing claim · *"noise-robustness remains a core asset of the Hopfield architecture"* · the three-"operational bounds" framing insofar as it is built on comparison.

**✅ MOVE INTO §4.1 (CLU-internal — carry VERBATIM where the sentence allows):**
- storage fidelity converging with training, and that the early underperformance was an **under-training artifact**;
- the **epoch-budget wall** — required epochs scale with capacity (kv32 saturates by ~1000, kv96 by ~4000), **not a hard capacity limit**;
- **kv32 over-trains** (`1.00 → 0.89`);
- the intra-CLU savings **`9.9\times, 9.5\times, 6.2\times` across kv32/kv64/kv96**, ⛔ **keeping the scope "against a full-budget CLU baseline"**. *(The "not a comparative win over the Hopfield network" half becomes moot once Hopfield is gone; the full-budget-CLU scope is the substantive half and stays.)*

**✅ KEEP, comparison-free — the noise-wall mention the Head ruled for.** One brief sentence, stated CLU-internally, e.g.: *under evaluation cue noise ($\sigma\in\{0.3,0.6\}$) gate accuracy degrades sharply — to $0.36$ at $\sigma=0.6$/kv32 — despite storage fidelity remaining at $1.0$: the governed relaxation over-commits to the corrupted cue.* ⛔ **No Hopfield number beside it. Named as an open problem, not a scoreboard row.**

### R2 — App C.3: delete two columns, keep the table
Locate `\subsection*{C.3 Regime Map: Capacity Axis and Epoch Frontier}`.
- ⛔ **Delete the `Hopfield Acc` and `$\Delta$` columns** (and their `tabular` column specifiers) from all six data rows.
- ⛔ **Rewrite the lead line**, which currently defines `Δ` as *"the CLU gate accuracy minus the Hopfield accuracy"* — that definition dies with the column.
- ✅ **Keep** `Cell`, `Epochs`, `CLU Fidelity`, `Gate Acc`, `Intra-CLU Savings` and every value in them, unchanged to the digit.
- ✅ Keep the epoch-frontier prose that follows insofar as it is CLU-internal.
- **Re-home the subsection under §4.1's material** (it is now the dial's characterisation table).

### R3 — App C.4: delete entirely
Locate `\subsection*{C.4 Regime Map: Stress Axes}`. It is a pure ρ-vs-Hopfield comparison table plus its commentary. **Delete the subsection.** ⚠ Its noise-wall sentence is replaced by R1's comparison-free mention — ⛔ **do not delete the noise wall from the paper, only from this table's framing.**

### R4 — Figure 2: delete
Locate `fig2_regime_map.png` (1 occurrence). The figure has three panels: (a) fidelity, (b) the clean-cue reversal, (c) the noise wall. **(b) and (c) are the comparison ledger, and you cannot re-render an image — so a partial keep would leave panels contradicting the text.** ⛔ **Delete the whole `figure` environment, its caption and its `\label`.** Then **sweep for any surviving `\ref` to that label and report it** (do not repoint it silently).
⚠ **Head one-word override available:** salvaging panel (a) alone is a `results-analyst` re-render, a separate pass. Note it in your report; do not attempt it.

### R5 — contributions list and abstract
- **Contribution bullet** on the regime map / corrected cost story: ⛔ delete the comparative claim. ✅ If a bullet is to remain it states only the CLU-internal dial result. ⚠ **If removing it changes the contribution count stated anywhere in the text, report that — do not renumber prose you were not asked to touch.**
- **Abstract:** delete the Hopfield clause. ⛔ Minimal excision — do not rewrite the surrounding sentence.

---

## 3. Method

⭐ **Scripted, assertion-guarded, single-occurrence edits, each asserting `count == 1` before writing.** A pattern matching 0 or 2 sites must fail loudly rather than write. This is the method that closed a comparable pass in this program with an empty orphan list.

- **Relocated text is carried VERBATIM wherever the sentence allows.** Connective prose is permitted **only** where a moved sentence would otherwise dangle, must match the surrounding register, and ⛔ **must contain no intensifiers and no new claim.**
- ⛔ **Zero new numbers.** Every numeric token in the result must already exist in `pj_sub.tex`. **The orphan list must be EMPTY.**
- ⛔ **No number changes value, precision, ±, seed count or units.**

---

## 4. Deliverables

1. **`BUILD-NOTE-R2.md`** — every edit by item ID (R1…R5), printed **before → after**, with its ancestor. Empty orphan list.
2. **A two-way numeric check**: every number surviving traces to the pre-pass file; every number removed is listed with the item that removed it.
3. **A residual-comparison sweep, positive-controlled**: `Hopfield` · `matrix-vector` · `matvec` · `$\Delta+0.02$` · `0.947` · `0.979` · `reverses` · `cheaper`. ⛔ **Report the surviving count for each and read every hit in context** — §4.1's memory-agnostic transfer (`0.18\to0.88`) legitimately names Hopfield and **must survive**; it is the disclosure that the gate mechanism is not CLU-specific. ⚠ **Do not "clean" it.**
4. **Build**: 0 errors, 0 undefined references; report total pages and the main-text/appendix split before and after.
5. **`.claude/outputs/v1-scoreboard-removal.md`** — the report, including a findings section for anything you noticed and did not touch.

## 5. Acceptance criteria

- `diff` against the pre-pass state contains **only** the enumerated changes. ⛔ **Zero unenumerated diffs** — checked independently at review.
- ⛔ `submission.tex` and `.claude/papers/v1-short/**` **byte-untouched** (md5 manifest printed).
- The pin check in §0 passed, or the pass aborted.
- `0.18`/`0.88` memory-agnostic transfer still present.
- Every negative positive-controlled.

## 6. ⛔ Prohibitions

1. ⛔ **Do not make any of the six Head-owned edits** (§0).
2. ⛔ **Do not add a comparison, a baseline, or a "compared to" clause anywhere.** The point of this pass is their removal.
3. ⛔ **Do not retire the noise wall** — it becomes a comparison-free open-problem sentence (R1).
4. ⛔ **Do not renumber, re-title or restructure sections** beyond dissolving §4.3 as instructed.
5. ⛔ **C-8 hermetic:** do not read V2's or V5's drafts.
6. ⛔ **Treat every C3-era number as PENDING.** Nothing here should need one.

## 7. ⚠ Grep hazards on this machine

⛔ `grep` here is a shell function resolving to **`ugrep 7.5.0`**: on bounded-context patterns over long `.tex` lines it either **errors *"exceeds complexity limits"* and exits 0** — a silent false negative that looks like success — or **hangs**. ⇒ use **`/usr/bin/grep`** for context patterns; count with `grep -o … | wc -l`, ⚠ **never `grep -c`** (lines, not occurrences). ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — sweep per-file. ⚠ zsh does not word-split: quote any `--include=*.tex` glob or omit it. ⚠ **False friends are live here**: `matvec` returns 0 in this file while the claim is present spelled out as *"matrix-vector multiplication"* — that exact miss produced a wrong finding in a prior pass. **Read every hit in context.**

## DIAL DECLARATION
**Dials touched: NONE.** This pass edits one `.tex` file: it removes a comparison, re-homes CLU-internal content, and deletes one figure and one table. It runs no experiment, changes no configuration, and produces no new measurement.
