# v2-revision-3 — paper-writer report

**Task + acceptance criterion:** close the two number-hygiene MUST-FIX (MF2-A, MF2-B) from the `v2-referee-2` clean pass with the exact referee-recommended wordings + two trivial SF items (SF2-1, SF2-2); rebuild both PDFs; supply a diff summary. No experiments.
**Status:** done. Both MUST-FIX closed; both SHOULD-FIX closed; both PDFs build clean. **One flagged deviation from the referee's literal text** (a rounding, see §"Deviation" below) — Hub decision requested but I believe it is required by binding rule 3.

---

## What I did (5 items, all wording/label; zero numbers invented, zero runs)

### 1. MF2-A — the "5×" (abstract + §3.2) — CLOSED, referee option (b)
Replaced "mispredicts by up to $5\times$ past the crossover" in **both** the abstract and §3.2 with the trained-model number plus an explicit exact-map attribution.

- **Abstract now:** "…and, past the crossover, mispredicts in a calculable direction by **≈3.2× at the deepest trained-model breaking we test** (measured/predicted 0.31 at δ=4); **on the exact analytic map the same mechanism continues to ≈5× further underdamped** (theory note, check (k))."
- **§3.2 now:** "…fails only past the crossover — by **≈3.2×** at the deepest trained-model breaking we test (measured/predicted **0.309±0.012** at δ=4; **our 14-δ sweep goes no deeper on the underdamped side**), in the calculable ballistic direction, with the EP delay spike (2.2×) as the third signature. The failure mechanism is the retention-metric bifurcation verified in §3.1, so it does not stop at our deepest trained tilt: on the *exact analytic map* the same single-exponential predictor continues to ≈5× further underdamped (theory note, Prop. metric-bifurcation and check (k), at **γ=0.1** — a different damping and a deeper tilt than any trained run here). **The ≈3.2× is the trained-model number; the ≈5× is exact-map only, and we never quote it as evidence about a trained network.**"

Three hardenings beyond the literal ask, all C-2/C-5-motivated and all source-backed:
- the sweep-exhaustion clause ("goes no deeper on the underdamped side") pre-empts "why not test deeper?";
- the γ=0.1-vs-γ=0.05 disclosure makes the exact-map number non-transferable by construction (it is a *different regime*, not a *deeper measurement*);
- the bolded non-transfer sentence is the C-2 verification/evidence firewall stated in-line, so a reviewer cannot lift the 5× as a trained-model claim.

I also linked the misprediction to the §3.1 metric bifurcation, which is its actual mechanism (envelope vs first-crossing split 3.2× at δ=4, γ=0.05, §3.1) — this makes the ≈3.2× *predicted* rather than merely *reported*, and makes the exact-map ≈5× at γ=0.1 a coherent continuation rather than an unexplained larger number. Both facts already existed in the draft; I only connected them.

### 2. MF2-B — the 15×/92% splice — CLOSED, all three sub-fixes
- **(i) γ label.** §3.4 table header `CLU (symplectic)` → **`CLU (γ=0 conservative)`** (md + tex).
- **(ii) Non-comparability caveat, stated TWICE** (at both ends of the splice, since the contradiction is only constructible by a reader holding both):
  - new **§3.4** paragraph "*Two comparability caveats, so the ladder is not misread against the table above*" — (a) different experiments/MSE scales, verbatim per the task: App-G ladder is a separate wake-only experiment (`fit-gap-anatomy`, seeds {0,1,2}, `9a13455`); its scale and twin (0.0047) are **not cross-comparable** with §3.4 `minus-the-physics` (twin 0.0128, seeds {42,43,44}, `b41410f`); **only within-table gaps are meaningful**; the ladder's γ=0 rung (0.2066) is the ladder's counterpart of the table's γ=0 CLU (0.190), *not the same measurement*. (b) "92%" is an absolute-gap fraction, not a ratio.
  - new **boxed caveat atop App G.2**, additionally naming the differing training objective (wake-only vs wake+sleep) and **explicitly defusing the cross-division**: "the §3.4 ≈15× (0.190/0.0128) and this table's implicit **≈44×** (0.2066/0.0047) are the *same qualitative statement measured on two different scales, not two different findings*."
- **(iii) Abstract reworded:** "a licensed global damping recovers 92% of the **absolute** contraction-forbidden fit gap with every structural guarantee intact — **the recovered unit still trails the twin by ≈4.6× in ratio, but is bounded by construction**." Mirrored in §3.4 ("92% of the *absolute* twin fit gap") and App G.2 ("recovers 92% of the **absolute** gap … while still trailing this table's twin by ≈4.6× in *ratio* (0.0216 vs 0.0047); the two statements are compatible and both must travel with the claim").

By naming the ≈44× myself, the paper now owns the number a hostile reviewer would compute, and marks it non-load-bearing. Note the referee's own hostile sentence #2 ("Which unit, and which twin, is the paper's actual price of physics?") is now answered explicitly in-text: **§3.4's γ=0 CLU vs its own twin (≈15×) is the price of physics; the ladder is a within-experiment mechanism attribution.**

### 3. SF2-1 (F5 note) — CLOSED
`f5-note.tex` provenance appendix: "all **14** checks reproduce F5 App-N" → "all **13** checks **(a)–(m)** reproduce F5 App-N". Verified the table has exactly 13 rows (`grep -c "^(.) &"` → `13`). Note this was a *second* stale count: v0.2 had already fixed the abstract's "All twelve results"→"All results" but left this one.

### 4. SF2-2 — CLOSED
Abstract "≈57–69" → "**≈56–69**" (LEM 56.4, LSTM 68.7; not rounded up). md + tex.

### 5. PDFs rebuilt
Both rebuilt with `tectonic`, exit 0.

---

## Deviation from the referee's literal wording (ONE, deliberate, flagged for the Hub)

The referee's recommended abstract text (and the task, quoting it) was: *"the unit still trails the twin **~5×** by ratio."* **I wrote ≈4.6×.** Rationale, in priority order:

1. **Binding rule 3 forbids rounding up.** 0.0216 / 0.0047 = **4.596**. "~5×" rounds up by 8.8%. This task exists to close a number-hygiene defect; shipping a rounded-up number inside its fix would be self-defeating, and SF2-2 in the *same task* corrects an identical 56.4→57 round-up.
2. **It would have re-created the exact defect MF2-A just removed.** The revised abstract already contains one "≈5×" (the exact-map misprediction). A second, unrelated "~5×" (the fit ratio) two sentences later hands a reviewer a spurious coincidence to unpick — precisely the "headline-number drift" class the referee flagged. The two "5×"-shaped facts are unrelated (one is a lifetime misprediction on the analytic map at γ=0.1; one is an MSE ratio on a wake-only fit experiment).
3. It is arithmetically checkable from the App-G.2 table printed in the same paper, so a reviewer *will* check it.

**If the Hub prefers the referee's literal "~5×", it is a one-token revert in 3 places** (abstract md line 13, tex line 25; §3.4 md; App G.2 md+tex) — but I recommend against it and record it here rather than silently.

---

## How I verified

```
$ grep -n "up to \$5\\\\times" draft.md draft.tex        → (none)
$ grep -n "CLU (symplectic)" draft.md draft.tex           → (none)
$ grep -n 'approx57' draft.md draft.tex                   → (none)
$ grep -c "^(.) &" f5-note/f5-note.tex                    → 13
$ grep -n "13 checks\|14 checks" f5-note/f5-note.tex      → line 283: "all 13 checks (a)--(m)"
```
Residual-`5\times` audit — every surviving occurrence inspected in context, all legitimate:

| tex line | occurrence | status |
|---|---|---|
| 25 | `\approx5\times` further underdamped | **new, exact-map-attributed** (intended) |
| 25 | `\approx15\times` better raw MSE | CM-1 canonical |
| 40 | `\approx15\times` (contribution 4) | CM-1 canonical |
| 87 | `\approx5\times` ×2 (exact-map + non-transfer sentence) | **new, attributed** (intended) |
| 125 | `14`–`15\times` FLOPs; `23.5\times` wall | CM-4 (SF-2) |
| 156, 158 | `15\times` twin edge / loan | CM-1 canonical |
| 169 | `35\times` wake MSE (anchor λ=100) | CM-6 |

Builds:
```
$ tectonic draft.tex    → exit 0; "Writing `draft.pdf` (588.22 KiB)"; warnings = 2 Underfull \hbox only
$ tectonic f5-note.tex  → exit 0; "Writing `f5-note.pdf` (163.81 KiB)"; no warnings
```
Cross-checked every quoted number against its source report before editing: `v2-full-runs` L79 (0.309±0.012 — and confirmed as the "5× failure" **mislabel** origin), `v2-referee-experiments` L38 (δ=4 row: exact-gap 0.309 / λ̂ 0.304, deepest row of the 14-δ sweep), `fit-gap-anatomy` L120–122 + L16 (twin 0.0047, γ=0 0.2066, +γ 0.0216, `9a13455`, seeds 0–2), `minus-the-physics` L29 + L10 (twin 0.0128, CLU 0.190, `b41410f`, seeds 42–44), F5 check (k) (deepest ratio 0.19 → 5.26×, γ=0.1, ε=0.1), §3.3 table (LEM 56.4 / LSTM 68.7).

---

## Diff summary

| file | loc | change |
|---|---|---|
| `v2-short/draft.md` | abstract | MF2-A (≈3.2× + exact-map clause), MF2-B(iii) (absolute + ≈4.6× ratio), SF2-2 (56–69) |
| | §3.2 ¶2 | MF2-A full wording + mechanism link to §3.1 + non-transfer sentence |
| | §3.4 table | header → `CLU (γ=0 conservative)` |
| | §3.4 ladder ¶ | "92% of the **absolute** twin fit gap"; **new** two-part comparability-caveat paragraph |
| | App G.2 | **new** boxed non-comparability caveat; recovery sentence gains absolute-vs-ratio pairing |
| `v2-short/draft.tex` | 25, 87, 145, 160, 266, 279 | identical changes, LaTeX-native |
| `v2-short/draft.pdf` | — | rebuilt, 588 KiB, 5 figures |
| `v2-short/CHANGELOG.md` | — | v0.4 entry (incl. the flagged ≈4.6× deviation) |
| `f5-note/f5-note.tex` | 283 | "14 checks" → "13 checks (a)–(m)" |
| `f5-note/f5-note.pdf` | — | rebuilt, 164 KiB |
| `f5-note/CHANGELOG.md` | — | v0.3 entry |

**Not touched (correctly):** §3.2 table (0.309), Fig 2 caption (0.31), Fig 2b caption (0.30), contribution 2 (0.31±0.01) — all already correct and now consistent with the prose. SF2-3 (pruning) deferred per task.

## Charter/matrix compliance check
- **C-1** no audit paragraph (unchanged, still absent). **C-2** the new §3.2 sentence *strengthens* the verification/evidence firewall (exact-map ≠ trained-model). **C-5** scale qualifiers preserved and one added ("dim 4, 5 seeds" already in abstract; "at γ=0.1" added to the exact-map clause). **C-6** unchanged. **C-7** all new numbers trace to A.3/A.5 + the named commits, now cited *inline* at the caveat. **C-8/M1** no new citations; the theory note stays third-person "(theory note, check (k))" — **no cross-short citation introduced**. **C-10** nothing pruned.
- **CM-4:** the draft now matches the w10 canonical verbatim in substance ("≈3.2× at the deepest TRAINED-model breaking (δ=4, meas/pred 0.31) — the 5× is EXACT-MAP ONLY … NEVER a trained-model number").
- **CM-1:** "≈15×" preserved exactly; the 92% now carries the "absolute" qualifier that CM-1's own wording implies but does not state. **See open question 2.**
- **CM-3** forbidden claim: absent (grepped).

---

## Open questions / follow-ups / risks

1. **[Hub decision] The ≈4.6× vs "~5×" deviation** (see above). One-token revert if you disagree; I recommend keeping ≈4.6×.
2. **[Matrix] CM-1's "92%" wording should inherit the "absolute" qualifier.** CM-1 currently reads "+γ global recovers 92% of the twin fit gap" with no absolute/ratio marker, and blends the 15× (`minus-the-physics`) with the 92% (`fit-gap-anatomy`) — i.e. **the matrix row itself encodes the MF2-B splice.** V1/V3 may cite CM-1 verbatim and reproduce the defect. Recommend amending CM-1 to: "…recovers **92% of the absolute** twin fit gap (separate wake-only experiment, seeds 0–2; still ~4.6× by ratio; not cross-comparable with the 15× table)…". This is the recurrence-prevention the referee asked for on CM-4 (already done, v1.5) but not on CM-1.
3. **[Matrix] CM-4's misprediction canonical is now satisfied**; no further action.
4. **[Risk, low] §3.1 already reports "split by up to 3.2× at δ=4"** (metric bifurcation, envelope vs first-crossing) and §3.2 now reports "≈3.2×" (meas/pred). These are numerically near-identical (3.16× and 3.24×) because **they are the same phenomenon** — I stated the mechanism link explicitly, so this reads as coherence rather than a copy-paste error. A referee may still ask; the answer is in-text.
5. **[Unchanged, carried] SF2-3 pruning** (>5 pp with 4 floats) and the **F5 arXiv id** remain the only two open items before submission. The end-note figure list still promises appendix "Fig 4/5" PNGs that do not exist (`fig4_emergent`, `fig5_isotropy`, `fig1_erosion_curves`, `fig4_cures`) — no broken embed, but a camera-ready must either produce them or cut the promise (referee N2-4). **Recommend folding into the pruning pass.**
6. **No missing experiments.** Both MUST-FIX were wording/label as diagnosed; nothing in this task required a number that does not exist.

## Proposed handover updates (for the Hub)
- **V2 short → v0.4; both w10 MUST-FIX CLOSED.** MF2-A ("5×"→≈3.2× trained / ≈5× exact-map-only, with a bolded non-transfer sentence and the γ=0.1 disclosure); MF2-B (§3.4 table labeled `CLU (γ=0 conservative)`; non-comparability caveat stated at **both** ends of the splice, naming the ≈44× cross-division and defusing it; abstract/§3.4/App-G "92% of the **absolute** gap, still ≈4.6× by ratio, bounded by construction"). SF2-1 (F5 "14 checks"→13, rows a–m) and SF2-2 ("≈57–69"→"≈56–69") closed. Both PDFs rebuild clean (draft.pdf 588 KiB / f5-note.pdf 164 KiB).
- **V2 status: clean weak-accept, submission-ready modulo (a) the pruning pass (SF2-3 + the promised-but-absent appendix Fig 4/5) and (b) the F5 arXiv id.** No missing experiments.
- **One writer deviation for Hub ratification:** referee's "~5× by ratio" → **"≈4.6×"** (0.0216/0.0047=4.596). Grounds: no-round-up rule + avoiding a second, unrelated "5×" in the abstract MF2-A had just disambiguated. Revert is one token in 3 places if the Hub disagrees.
- **Matrix action recommended (CM-1):** CM-1 encodes the MF2-B splice at the source (blends `minus-the-physics` 15× with `fit-gap-anatomy` 92%, no absolute/ratio marker, no non-comparability note). V1/V3 cite CM-1 verbatim → the defect can propagate. Suggest amending CM-1's 92% clause to carry "**absolute** … (separate wake-only experiment, seeds 0–2; ≈4.6× by ratio; not cross-comparable with the 15× table)". CM-4 already carries its w10 canonical and needs nothing.
- **F5 note → v0.3** (count fix only). Its exact-map "∼5×" (Prop. metric-bifurcation / check (k), γ=0.1, ε=0.1, ratio 0.19) is untouched and is now the V2 short's explicitly-cited source for that clause — the two documents are cross-coherent on the misprediction for the first time.
