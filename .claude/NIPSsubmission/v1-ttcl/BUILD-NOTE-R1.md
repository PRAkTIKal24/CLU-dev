# BUILD-NOTE-R1 — V1 revision pass 1 (`v1-revision-1`)

**Agent:** `paper-writer` · **Date:** 2026-08-26 · **File edited:** `.claude/NIPSsubmission/v1-ttcl/submission.tex` (that file only).
**Method:** scripted, assertion-guarded, single-occurrence replacements (`assert count == 1` before every write; `assert new-text not already present`). Script: `.claude/scratch/v1-revision-1/apply_edits.py`; machine-readable log: `.claude/scratch/v1-revision-1/edits.json`.

⭐ **ORPHAN LIST: EMPTY.** Every added token carries an ancestor below (a `draft.md` line, a registry clause, or the Head ruling in the task file). No number was invented, adjusted, rounded or smoothed.

## 0. Mechanical proof of scope (the acceptance criterion)

| check | result |
|---|---|
| enumerated edits applied | **22** scripted replacements + **3** post-patches (2 typesetting, 1 phrasing hardening) — all listed below |
| reconstruction test (BEFORE + logged edits ⟹ live file, byte-for-byte) | **IDENTICAL**, md5 `caef2272f9dc96d349b46486563d24ee` ⇒ **zero unenumerated diffs** |
| `papers/v1-short/**` byte-untouched | **11/11 md5 digests identical** (manifest §6) |
| build | `pdflatex` (TeX Live 2026, `/Library/TeX/texbin`), 2 passes, **0 errors, 0 LaTeX warnings, 0 undefined references** |
| page split | **before 18 pp** (main 1–12, App. A from p. 13) → **after 25 pp** (main 1–14, App. A from p. 15). Reported, not optimised (task §D). |
| overfull boxes | 3, **exactly the 3 present in the baseline** (32.90 / 2.28 / 1.13 pt); the two my restored tables introduced were removed by column-spacing only |

## 1. Edits, item by item — before → after, with ancestor

### A1
**Ancestor:** `draft.md:78` (§3.2 heading, verbatim)
**BEFORE:**
```
\subsection{The discriminating experiment: reach steps then collapses; the wormhole is flat --- and only it hands back a receipt}
```
**AFTER:**
```
\subsection{The discriminating experiment: squeeze reach is priced, the wormhole is flat-priced --- and only the wormhole hands back a receipt}
```

### A2
**Ancestor:** `draft.md:31` ("An untrained **state-replacing** map $(q,p)\mapsto(b,p)$")
**BEFORE:**
```
The router's map $(q,p)\mapsto(b,p)$ has $\detJ=0$
```
**AFTER:**
```
An untrained \textbf{state-replacing} map $(q,p)\mapsto(b,p)$ has $\detJ=0$
```

### A3
**Ancestor:** `draft.md:448` (N2 entry, settled form — condensed per task A3)
**BEFORE:**
```
\textbf{N2} abstention-vs-Hopfield unwinnable; Hopfield dominant at $500$ ep (\emph{provisional --- under-trained map}).
```
**AFTER:**
```
\textbf{N2} abstention-vs-Hopfield unwinnable; the $500$-ep ``Hopfield-dominant'' map was an \textbf{under-training artifact} --- at convergence the gate reverses Hopfield \textbf{only on clean/correlated cues at kv$\le64$} ($\Delta+0.02$), the barrier beyond it is an \textbf{epoch-budget wall, not a capacity wall}, and the tally is $\mathbf{6/15}$ cells closing at $2000$ ep (settled, full grid, $198$ jobs).
```

### A4
**Ancestor:** `draft.md:72` ("(the theory note's coset content, $X$ the broken generator)")
**BEFORE:**
```
For a protected direction carrying Goldstone charge $Q=p^\top X q$, under the wormhole
```
**AFTER:**
```
For a protected direction carrying Goldstone charge $Q=p^\top X q$ (the theory note's coset content, $X$ the broken generator), under the wormhole
```

### A5a
**Ancestor:** `draft.md:49` ("The theory note proves (Prop-A2) that…")
**BEFORE:**
```
\textbf{Access} is provably enlarging $Q_T$. In relativistic mode one drift advances
```
**AFTER:**
```
\textbf{Access} is provably enlarging $Q_T$. The theory note proves (Prop-A2) that in relativistic mode one drift advances
```

### A5b
**Ancestor:** `draft.md:27` ("**[proven; theory note + Anonymous 2026.]**")
**BEFORE:**
```
different receipts. \emph{[proven.]}
```
**AFTER:**
```
different receipts. \emph{[proven; theory note + Anonymous 2026.]}
```

### B1-43
**Ancestor:** `draft.md:388/390/392` (C.4.a rows: 9.9× / 9.5× / 6.2×) · rule CM-22(bb) · CM-8 `claims_matrix.md:556`
**BEFORE:**
```
the CLU gate's $9$--$10\times$ figure is \emph{intra-CLU} rationing vs a full-budget CLU
```
**AFTER:**
```
the CLU gate's savings figure --- $9.9$/$9.5$/$6.2\times$ across kv$32$/$64$/$96$, falling with load --- is \emph{intra-CLU} rationing vs a full-budget CLU
```

### B1-151
**Ancestor:** `draft.md:388/390/392` (C.4.a rows) · CM-22(bb)
**BEFORE:**
```
The CLU gate's ``$9$--$10\times$ savings'' is an \textbf{intra-CLU} number
```
**AFTER:**
```
The CLU gate's savings --- $9.9$/$9.5$/$6.2\times$ across kv$32$/$64$/$96$ (App.~C.4.a), a curve that falls with load, not a constant --- is an \textbf{intra-CLU} number
```

### B1-153
**Ancestor:** `draft.md:388/390/392` (C.4.a rows) · CM-22(bb)
**BEFORE:**
```
(every ``$9$--$10\times$'' here is \emph{intra-CLU} rationing, not a win over Hopfield)
```
**AFTER:**
```
(every savings number here --- $9.9$/$9.5$/$6.2\times$ at kv$32$/$64$/$96$ --- is \emph{intra-CLU} rationing, not a win over Hopfield)
```

### B1-157
**Ancestor:** `draft.md:388/390/392` (C.4.a rows) · CM-22(bb)
**BEFORE:**
```
at $9$--$10\times$ \emph{intra-CLU} rationing
```
**AFTER:**
```
at $9.9$/$9.5$/$6.2\times$ \emph{intra-CLU} rationing across kv$32$/$64$/$96$
```

### B2
**Ancestor:** `draft.md:388` (kv32 @2000 ep = 9.9×) vs `draft.md:136` (§4.1 kv32 @400 ep = 1.14±0.06×) — flag only, no number moved
**BEFORE:**
```
the robust payoff is \emph{rationing} ($9.9\times$ intra-CLU)
```
**AFTER:**
```
the robust payoff is \emph{rationing} ($9.9\times$ intra-CLU at kv$32$/$2000$ ep --- \S4.3's grid, a different epoch budget from this subsection's)
```

### C1
**Ancestor:** Head ruling, task file §C1 (restatement, non-comparative) + the measured replacement it points at = CM-23(r) `claims_matrix.md:574`
**BEFORE:**
```
The honest pillars of \S4 delimit where this buys a real ML advantage (escalatable rationing) and where a cheaper black box wins (routing): a certificate stack is a design discipline, not a guarantee of dominance.
```
**AFTER:**
```
The honest pillars of \S4 delimit what the extra compute does \emph{on this memory}: a rationed relaxation budget buys accuracy on the memory we trained (\S4.1, \S4.3), and we offer that as an \textbf{empirical proof of concept}, not as a comparative result --- \textbf{no matched-compute floor and no matched-bytes exemplar store is run against \S4.1--4.3 anywhere in this paper}, and that comparison is named here as \textbf{future work} (the one matched-compute comparison we do report, \S4.4, comes back a \emph{tie}). They also delimit where a cheaper black box wins (routing): a certificate stack is a design discipline, not a guarantee of dominance.
```

### C2
**Ancestor:** CM-23 scoreboard sentence, `claims_matrix.md:574` — VERBATIM
**BEFORE:**
```
Where the receipt says a cheaper black box wins, we say so.
```
**AFTER:**
```
Where the receipt says a cheaper black box wins, we say so. \textbf{The scoreboard sentence, stated up front: external benchmarks won on their own headline metric = ZERO.}
```

### C3
**Ancestor:** §A20.5 substrate-scope sentence, `claims_matrix.md:607` — VERBATIM (φ → $\varphi$)
**BEFORE:**
```
No claim here is at scale, and none uses learned placement.
```
**AFTER:**
```
No claim here is at scale, and none uses learned placement. And the substrate scope, stated once, in our own voice: \textbf{these laws govern the store; end-to-end performance additionally depends on the encoder, measured separately, $\varphi$-bytes ledgered.}
```

### D1
**Ancestor:** `draft.md:193–282` (Appendix A, five flag-provenance tables + cross-section note)
**BEFORE** (1839 chars): `All results inherit their exact non-default configuration; see \texttt{draft.md} Appendix~A for the full four tables (\S3 certificate stack, commit \texttt{6f2384c}; \S4.1 gate, commit \texttt{572c708}, with the memory-agnostic Hopfield transfer $0.18\to0.88$ from \texttt{minus-the-physics} Part~B,  …`
**AFTER** (8836 chars): full restored block — see `.claude/scratch/v1-revision-1/blocks/`; head: `All results inherit the exact non-default configuration in effect. Repositories read-only for the analysis reports; branches for the code-producing reports named in Appendix~A.4.

\subsection*{A.1\quad \S3 certificate stack (\texttt{paid-access-theory}, \texttt{paid-access-experiments})}
\noindent{\ …`

### D2
**Ancestor:** `draft.md:328–442` (Appendix C, C.1 / C.1.b / C.2 / C.3 / C.4.a–c)
**BEFORE** (1347 chars): `See \texttt{draft.md} Appendix~C for: (C.1) the full reach battery grid + per-arm certificates (wormhole $\detJ=[1.0]{\times}6$ exact, ledger err $[0.0]{\times}6$; \textbf{state-replacing map (\texttt{no\_physics\_router}) $\detJ=[0.0]{\times}6$, measured by forward-mode autodiff --- volume-annihila …`
**AFTER** (11383 chars): full restored block — see `.claude/scratch/v1-revision-1/blocks/`; head: `\subsection*{C.1\quad Reach battery, full grid (Table~2; \texttt{paid-access-experiments} \S7.1, dim $2$, $5$ seeds, $L=2.5$)}
\noindent{\footnotesize
\begin{tabular}{@{}lcccccc@{}}
\toprule
arm & $d{=}0.8$ & $1.6$ & $2.4$ & $3.2$ & $4.0$ & $5.0$ \\
\midrule
\texttt{plain\_relax} & 0 & 0 & 0 & 0 & 0 …`

### D3
**Ancestor:** `draft.md:449` (N2b, THE NOISE WALL) — CM-8 rider: "travels with every reversal claim"
**BEFORE:**
```
and the tally is $\mathbf{6/15}$ cells closing at $2000$ ep (settled, full grid, $198$ jobs). \textbf{N3}
```
**AFTER:**
```
and the tally is $\mathbf{6/15}$ cells closing at $2000$ ep (settled, full grid, $198$ jobs). \textbf{N2b THE NOISE WALL: the accuracy reversal does not survive cue noise (dominant negative).} Under eval-noise $\sigma\in\{0.3,0.6\}$ \textbf{no cell closes at any capacity, even kv$32$} --- gate $0.36$ vs Hopfield $0.71$ at $\sigma=0.6$/kv$32$ ($\Delta$ from $-0.05$ to $-0.35$) --- \emph{despite} CLU storage fidelity $\approx1.0$. The governed relaxation over-commits to the corrupted cue; the gate's rationing is a \textbf{clean-retrieval} asset and \textbf{noise-robustness is Hopfield's}. This is the axis most relevant to real retrieval and the sharpest negative in the study; a referee finding it absent would be right to fault the paper (App.~C.4.c). Diagnosing/curing the noise-robustness gap (noise-aware $\tau$, longer relax budget, denoising init) is future work. \textbf{N3}
```

### E1
**Ancestor:** CM-23(r) `claims_matrix.md:574` (VERBATIM core) · N103 `negative_results.md:1002` · riders: N95 `negative_results.md:899` (⟲ Head ruling 2026-07-25, corrected status) and N308 `negative_results.md:3297` (flat-curve disjunction, 3-point curve)
**BEFORE** (39 chars): `
\section{Position, scope, and horizon} …`
**AFTER** (4027 chars): full restored block — see `.claude/scratch/v1-revision-1/blocks/`; head: `
\subsection{A matched-compute anytime read (beyond the three pillars): it ties}
\emph{Reporting grade: evidence, with the substrate stated --- a \textbf{designed} store addressed through a \textbf{learned, frozen} encoder $\varphi$, not a trained CLU checkpoint. Every number in this subsection is $ …`

### E2
**Ancestor:** CM-23(y) `claims_matrix.md:574` (VERBATIM) · N119 `negative_results.md:1166` (both fixes refuted / not available)
**BEFORE:**
```
They also delimit where a cheaper black box wins (routing): a certificate stack is a design discipline, not a guarantee of dominance.
```
**AFTER:**
```
They also delimit where a cheaper black box wins (routing): a certificate stack is a design discipline, not a guarantee of dominance. \textbf{The position has a third corner, and it is an impossibility result.} Beyond a single access mechanism, the same accounting binds the store's \emph{read}: with a quadratic payload channel, the $q_2(0)=0$ anti-decoration guard and a fixed read budget, a store cannot have all three of \textbf{exact value fidelity}, \textbf{amplitude-independent address hold} and \textbf{amplitude-independent read latency} --- pick two. The shipped store drops the second, which is why effective lifetime correlates $r=-0.85$ with $a_i^2$; the recommended gated-stiffness channel drops the third. \textbf{Both proposed fixes to that corner are refuted, and neither may be described as available.} The corner to lead with is the third: \emph{dropping amplitude-independent latency \textbf{is} the compute-adaptive-read dial --- a faded memory costs more integration steps to read, which is a physical, measurable statement a timestamped row cannot make.} (Proved and measured on this program's decaying-store instantiation --- a designed store with a quadratic payload channel and a fixed read budget, $3$ seeds, laptop-CPU --- not on the \S3 unit.)
```

### E3
**Ancestor:** CM-23(g) `claims_matrix.md:574` (VERBATIM) · N90 `negative_results.md:854` · CM-23(b) shape sentence (VERBATIM) · CM-23 v2.3 5-seed range (mask −3.9…−20.7 pp; noise −9.7…−48.2 pp; 40/40; +0…+79.7 pp; 0/40) · CM-23(l)/N95 rider
**BEFORE** (141 chars): `use $\tfrac12\,\mathrm{MALA}(\sigma^\star)+\tfrac12$ sign-symmetrized squeeze-MH (a mixture of $\pi$-reversible kernels is $\pi$-reversible). …`
**AFTER** (1905 chars): full restored block — see `.claude/scratch/v1-revision-1/blocks/`; head: `use $\tfrac12\,\mathrm{MALA}(\sigma^\star)+\tfrac12$ sign-symmetrized squeeze-MH (a mixture of $\pi$-reversible kernels is $\pi$-reversible). \textbf{Measured, on a designed store (evidence; MNIST, designed/closed-form on every line, an $8$-cell ladder at seed $0$ re-measured across $5$ seeds \emph{ …`

### E3b
**Ancestor:** mechanical consequence of E3, required by task §E3 ("App F.6's 'not run' language must be updated in the same edit")
**BEFORE:**
```
four design rules (theory-complete on toy EBMs; no runs on trained CLU checkpoints are claimed)
```
**AFTER:**
```
four design rules (theory-complete on toy EBMs; rule~2's mechanism controls are measured on a designed store; no runs on trained CLU checkpoints are claimed)
```

### E3c
**Ancestor:** same as E3b (App. F.6 closing sentence)
**BEFORE:**
```
\textbf{Not run; no CLU-checkpoint result is claimed.}
```
**AFTER:**
```
\textbf{Partially answered, and on a different substrate:} the \emph{mechanism} half --- directed re-launch versus equal-energy random kick, ensemble-of-$k$ restarts and ungated retry-all --- is measured on a \textbf{designed} store (\S5, design rule~2; MNIST). \textbf{What remains unrun is this specification itself: the certified mixture kernel at $\gamma=0$ with coset projection, on a trained CLU checkpoint, with the latch-erosion decay curve. No trained-CLU-checkpoint result is claimed.}
```

### post-patch 1 — typesetting only (inside the D2 restoration)
`\noindent{\footnotesize` → `\noindent{\scriptsize\setlength{\tabcolsep}{3pt}` on the **C.1.b latch-payoff table**. Reason: the 6-column table overran the text block by 28.95 pt. **No content, no number, no wording changed.**

### post-patch 2 — typesetting only (inside the D2 restoration)
`\noindent{\footnotesize` → `\noindent{\footnotesize\setlength{\tabcolsep}{3pt}` on the **C.4.a capacity-axis table** (8 columns; overran by 56.90 pt). **No content changed.**

### post-patch 3 — phrasing hardening (inside the E3 fold)
`the gated read wins no cell of $8$` → `the gated read does \textbf{not} win in any of $8$ cells`.
Reason: although negated, the literal substring *"gated read wins"* would trip a forbidden-form grep at review (CM-23(r): the word is **ties**, never *wins*). The new phrasing matches CM-23(l)'s own *"does NOT beat … in any of 8 cells"*.

---

## 2. Two-way numeric check (deliverable #2)

Automated token-level comparison (`.claude/scratch/v1-revision-1/numcheck.py`) between each restored block and its `draft.md` ancestor range.

| block | ancestor tokens | restored tokens | missing | extra |
|---|---|---|---|---|
| D1 (App. A, `draft.md:193–282`) | 311 | 312 | `7` ×1 | `1` ×2 (+ column-width specs) |
| D2 (App. C, `draft.md:328–442`) | 588 | 580 | `3`×2, `32`, `64`, `96`, `128`, `0.947`, `0.976`, `2` | `4.3` ×1 |
| D3 (N2b, `draft.md:449`) | 10 distinct | 10 distinct | **none** | **none** |

**Every difference is accounted for; none is a changed measurement:**

1. **D1 `7` missing** — the markdown appendix *title* reads "Appendix A — Flag-provenance tables **(C-7)**". The `.tex` section heading (`\section{Flag-provenance tables}`, line 192) is **pre-existing and not on the worklist**, so the "(C-7)" tag is not in the restored body; the C-7 label survives inside the restored *cross-section reproducibility note (C-7)*.
2. **D1 `1` ×2 extra** — `draft.md`'s "$128$–$144$ trials·seed⁻¹·level⁻¹" uses **Unicode superscript-1**; LaTeX renders it `seed$^{-1}\cdot$level$^{-1}$`. Same exponents, ASCII digits.
3. **D1 column-width specs (`0.24`, `0.70` ×5)** — `p{0.24\textwidth}` / `p{0.70\textwidth}` tabular geometry. Typesetting, not data.
4. **D2 `3`×2, `32`, `64`, `96`, `128`, `0.947`, `0.976`** — these are exactly the tokens of the **duplicated Figure-3 float** in `draft.md` C.4.b (`![Figure 3](fig_frontier_clean.png)` + its caption). **That figure already exists in the `.tex` main text as Figure~\ref{fig:frontier} (= Figure 3, p. 11)**; re-inserting the float would have created a duplicate figure — an unenumerated new float. It is replaced by a cross-reference inside C.4.b's own heading: *"`fig_frontier_clean.png`, plotted as Figure 3 in §4.3"*. Re-running the check with those two source lines excluded leaves **missing = {`2`}**, item 5.
5. **D2 `2` missing** — `draft.md`'s C.4.a heading says "Figure **2**a–b"; the `.tex` uses `Figure~\ref{fig:regime}a--b`, which the `.aux` resolves to **Figure 2** (`newlabel{fig:regime}{{2}{10}}`). Identical referent.
6. **D2 `4.3` extra** — the "\S4.3" in that same cross-reference.

⇒ **No value, precision, ±, seed count or unit differs from its ancestor anywhere in the restorations.** Numbers that *left* the file: the four `9$--$10\times` strings (B1, replaced by the measured curve 9.9/9.5/6.2×, ancestors `draft.md:388/390/392`) — nothing else was deleted.

---

## 3. Positive-controlled forbidden-form sweep (deliverable #3)

Tool discipline per the task's grep hazard note: pure-Python `re` counting over the whole file (not `grep`/`ugrep`), occurrence counts (not line counts). Script: `.claude/scratch/v1-revision-1/sweep.py`.

**Zero-hit list (all 0 in the edited file):**
`beats feedforward` · `beats the feedforward` · `beat the feedforward` · `beats a feedforward` · `wins via test-time compute` · `test-time compute wins` · `the anytime read wins` · `anytime read wins` · `the gated read wins` · `9$--$10\times` · `9--10\times` · `9-10x` · `$9$--$10\times$` · `only memory with` · `the first memory` · `uniquely` · `unique among` · `no other architecture` · `superior signal` · `energy-gated router wins` · `energy is the routing signal` · `r3 failed` · `r3 leaderboard` · `gated-stiffness channel is available` · `gated-stiffness fix is available` · `capacity multiplies by sharding` · `the write operator is the ceiling` · `24.5$\times$ fewer floats` · `0.99985` · `slack 1.08` · `deletion-compliant` · `unlearning` · `exact deletion` · `real ML advantage` · `cm-23` · `cm-22` · `n95` · `n103` · `n90` · `n308`

**Positive controls (the sweep can see the file):** `ties` 4 · `auto-stopping` 3 · the scoreboard sentence 1 · `noise wall` 5 · `9.9` 6 · `state-replacing` 25 · `proof of concept` 1 · `these laws govern the store` 1 · `prop-a2` 1 · `broken generator` 1.

**Three non-zero hits, each adjudicated as NOT a violation (all verified against the baseline copy):**

| hit | count new / baseline | adjudication |
|---|---|---|
| `energy is a better` / `better confidence signal` / `energy-as-a-superior` | 1 / **1** each | **Pre-existing**, and both are the paper *disclaiming* CM-3: *"not a claim that CLU energy is a better confidence signal"* and *"energy-as-a-superior-signal is a claim we do not make anywhere (N3)"*. Untouched by this pass. |
| `2.6` | 6 / 4 | The two new instances are (a) C.4.c row `N256/kv64, ρ=0.9 … 2.6×` savings — a measured ancestor value (`draft.md`), and (b) the pre-existing `ζ≈2.6441` restated in C.1's footnote. **Not** the program-retracted "2.6" (a different, non-V1 constant). |
| `draft.md` | 1 / 3 | Reduced 3 → 1 by D1/D2. The survivor is the **line-2 source comment** (`% … Canonical content: draft.md.`) — not typeset, not on the worklist. Flagged in the report's findings. |

**Required-wording exactness (each present exactly once, verbatim):** the scoreboard sentence · the substrate-scope sentence · *"no mask oracle can be constructed"* · *"dropping amplitude-independent latency **is** the compute-adaptive-read dial"* · *"the lift is the **directed** symplectic re-launch"* · *"dead flat in all 8 cells"* · *"ungated retry-all collapses 0.96→0.004 at 9× compute"* · *"cosine-to-nearest-well is a ranking, never an acceptance, signal"* · *"the store carries nothing **or** it cannot be addressed"*.

---

## 4. Build (deliverable #4) — BUILT, not pseudo-verified

`pdflatex` is **not on PATH** but **is installed** at `/Library/TeX/texbin/pdflatex` (pdfTeX 3.141592653-2.6-1.40.29, TeX Live 2026) — the same engine the Advisor's `submission.log` shows. Two passes run each time; `.aux/.log/.out/.pdf` regenerated in place (build artifacts only; `submission.tex` is the only *edited* file).

| | pages | §4 starts | §5 starts | App. A starts | errors | LaTeX warnings |
|---|---|---|---|---|---|---|
| **before** | **18** | 8 | 10 | 13 | 0 | 0 |
| **after** | **25** | 8 | 12 | 15 | 0 | 0 |

Full after-split: §1 p2 · §2 p3 · §3 p4 · §4 p8 · **§4.4 p11** · §5 p12 · App. A p15 · App. B p17 · App. C p19 · App. D p22 · App. E p22 · App. F p23 · References p24.
**Main text 12 → 14 pp (+2); appendices 6 → 11 pp (+5).** ⛔ Not optimised — per task §D the Head condenses in `pj_sub.tex`.

---

## 5. Three conversion decisions inside the enumerated items (declared, not hidden)

1. **E1's section placement and the word "Three".** §4 is titled *"Three honest pillars on learned memories"* and the **abstract** says *"three honest pillars"*. Renaming the section to "Four" would have contradicted the abstract; editing the abstract is not on the worklist. The fold therefore lands as **§4.4 "A matched-compute anytime read (beyond the three pillars): it ties"** — placed in §4 as the task requires, with a title that is consistent with both existing "three" strings. **No heading outside the enumerated list was touched.**
2. **D2's Figure-3 float** — replaced by a cross-reference to the identical, already-present main-text float (numeric check §2, item 4). No number lost.
3. **A5's line number.** The task names `submission.tex:56` for the Prop-A2 attribution; the clause that carries the $Q_T\subseteq C_T$ derivation is at **:50** (`:56` is the REACH-failure bullet). Applied at :50, which is `draft.md:49`'s actual counterpart. All other task line numbers matched exactly.

---

## 6. md5 manifest, `papers/v1-short/**` (before → after)

| file | md5 before | md5 after |
|---|---|---|
| CHANGELOG.md | a2bc48c0e0c2f3ceab1fa7cf34f655c8 | a2bc48c0e0c2f3ceab1fa7cf34f655c8 |
| draft.log | 86d8f80c82f90c4ab1fa0ce8a384cc0d | 86d8f80c82f90c4ab1fa0ce8a384cc0d |
| **draft.md** | **00d703d58a15c0cb77051a9c55674684** | **00d703d58a15c0cb77051a9c55674684** |
| draft.pdf | 141f2c37ee8089c814932ce27e5f4fa4 | 141f2c37ee8089c814932ce27e5f4fa4 |
| draft.tex | 208797d113fa9d6efa6de67d05705ea6 | 208797d113fa9d6efa6de67d05705ea6 |
| fig1_certificate.png | 679647f639bfb8b3b7ecfa1333f43b69 | 679647f639bfb8b3b7ecfa1333f43b69 |
| fig2_regime_map.png | b0cfbf53651ac187bacee0f977d93f1e | b0cfbf53651ac187bacee0f977d93f1e |
| fig4_bibo.png | 708b6fae2dadc755291b07c2962d102f | 708b6fae2dadc755291b07c2962d102f |
| fig_frontier_clean.png | bcc5f32dcd85e01740638c6608f26320 | bcc5f32dcd85e01740638c6608f26320 |
| fig_regime_map.png | 8b1dfddc54b4e48da0254a3bf35b9159 | 8b1dfddc54b4e48da0254a3bf35b9159 |
| paid_access_reach.png | fc372ae54cf39d7181f68837ad0e463b | fc372ae54cf39d7181f68837ad0e463b |

`submission.tex`: `05586b2db9652ea3c83964cb61284466` → `caef2272f9dc96d349b46486563d24ee`.

---

## 7. Riders discharged in-sentence (Part E)

| rider | where it landed |
|---|---|
| CM-23(r) *"ties", never "wins"* | §4.4, first paragraph: *"It ties; it does not win"*; zero `wins` substrings near the claim (sweep §3) |
| N95 placement obligation (same section as the retry claims) | stated **twice** — §4.4 *"The negative that travels with any anytime claim (C-9)"* and §5 rule 2's closing — because the retry claims land in two sections. Corrected status quoted (*static-retrieval instantiation closed; mechanism survives; losing to a told-the-mask oracle is the metric-native ceiling*); ⛔ *"R3 failed"* nowhere. |
| N308 flat-curve disjunction | §4.4 *"Quoting an anytime curve, including a flat one (C-6)"* — disjunction + the **three-point** curve 0.0223 → 0.8219 → 0.8711 → plateau vs shipped-flat 0.0004 (never an endpoint pair) |
| occupied-venue / no-uniqueness | §4.4 closing sentence (DEQs, EBTs, recurrent-depth), no `\cite` added (see report findings F-7) |
| seed counts never merged | §4.4 opens *"Every number in this subsection is 6 seeds of **its own** protocol, never §3/§4.1–4.3's 5 seeds"*; §5 rule 2 says *"an 8-cell ladder at seed 0 re-measured across 5 seeds **of that protocol** — never §3/§4's 5 seeds"* |
| CM-23(y) hard never-quote | §5: *"Both proposed fixes to that corner are refuted, and neither may be described as available."* immediately after the gated-stiffness clause |
| CM-8 noise-wall rider | restored as App. D **N2b** (D3) and present in §4.3 + App. C.4.c |
| CM-23(g) mandatory companions | §5 rule 2: (i) the verbatim shape/floor sentence + the **5-seed** gap range (−3.9…−20.7 mask, −9.7…−48.2 noise, 40/40); (ii) protocol split +36…+76 vs +6.6…+34.8 pp, nothing past σ≈0.4; (iii) cosine-to-nearest-well is a ranking, never an acceptance, signal |
| CM-23(aa)/N117 (τ-regime rule) | ⛔ **NOT folded** (task: out of scope). The τ sub-claim of CM-23(r) is therefore **omitted** from §4.4 rather than stated in a superseded form — see report finding F-8. |
