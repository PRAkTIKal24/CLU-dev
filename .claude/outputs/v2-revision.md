# v2-revision — paper-writer report

**Task + acceptance criterion:** apply the `v2-referee` punch-list (MF-1…5, SF-1…6, NICE) to the V2 short (`.claude/papers/v2-short/`) and F5 note (`.claude/papers/f5-note/`) in place; every MUST-FIX closed or (MF-5 Cor-3) resolved per Head DEC; both PDFs rebuild (tectonic); diff-summary mapping each item to what changed.
**Status: done.** Both PDFs build clean (exit 0). draft.md canonical + draft.tex synced; f5-note.tex revised. CHANGELOGs bumped. SF-1/2/3 left as marked `[pending: v2-referee-experiments]` slots (need small runs — not fabricated). Report-only files under `.claude/`; no tracked code touched (no git footprint).

## How I verified
- `tectonic draft.tex` → `Writing draft.pdf (395 KiB)`, exit 0. Only minor Underfull/one Overfull `\hbox` warnings (App D/G table rows); no errors; all 3 `\includegraphics` resolved (build errors if a figure were missing).
- `tectonic f5-note.tex` → `Writing f5-note.pdf (168 KiB)`, exit 0, zero warnings.
- `grep todohead|TODO-HEAD` across both `.tex` → empty (command def + all 5 uses + stale header comment removed).
- `grep audit|unmeasured|12–28|solved constitutive|separately-proven|subsumes|twelve|public instantiation|few-to-fifteen` → all cleared (remaining hits were the legit "$+12$–$15\%$" body text and the removed-phrase absence).
- Figures staged: `cp` fig1_gmor / fig2_mo (from `v2-full-runs/`) + figA_retention_overlay→fig3 (from `v2-prefreeze-baselines/`) into `.claude/papers/v2-short/figs/`.

## Diff-summary — each punch-list item → what changed

### MUST-FIX (all closed)
- **MF-1 (delete §1 audit paragraph + orphaned cross-refs).** Deleted the "A brief audit disclosure" paragraph in both draft.md (was L23) and draft.tex (was L33). App F reframed: N17/N18 legacy-mechanism confessions **demoted** to a single neutral "class-level design caveats (neutral theorems in the theory note)" line with **no legacy numbers** (dropped the ½ln(1−γ) inertness and the ≈11×-colder Langevin figures as evidence; kept only the FDT scale as neutral class theory). **N22 kept** but reframed as a "class scope limitation" with the `(physics-audit paragraph, §1)` back-ref removed. App F header "program-wide audit negatives" → "class-level design caveats". §2's "150 epochs" forward-ref kept (not audit framing), per task.
- **MF-2 (wire the measured crossing + recovery ladder).** §3.4 closing paragraph rewritten: the loan is **called at ≈700 steps** (twin leads 1–2 orders to ~500, crosses CLU ~700, diverges to **196 @5000** vs CLU bounded **0.20–0.23**); asset = **boundedness-by-construction**. Recovery ladder wired: **+γ global recovers 92%** with BIBO/latch/μ² preserved; **γ_φ −24%** wrong tool. **CM-1 guard applied:** explicit "not the lowest steady MSE — broken-vol 0.14, LSTM 0.13 < CLU 0.22". Abstract gained the crossing clause. New **Appendix G** holds the full loan-curve table (6 horizons × 4 arms), the ladder table (4 rungs × 5 structural columns), and the reach certificates (squeeze Δq=2.18/det=1; κ=0 bit-exact), with full flag provenance (commit 9a13455).
- **MF-3 (reconcile §3.4 ↔ App D).** "consistent +12–28%" **deleted**. §3.4 now states the autonomous-protocol triple (277/257/303 vs 247/227/263, canonical kick 0.1) = **+12–15% envelope**, and App D as a **distinct probe** (kick→0 amplitude sweep on the **same three checkpoints**) resolving a per-seed **+5 to +29% spread** (1.052/1.125/1.288) — settle-offset (2/3 seeds) + genuine anharmonicity (softest-μ² seed, ×1.05→×1.55 with amplitude). One coherent range **includes the +5% seed** and its endpoints are the displayed App D ratios. App D header now says it is a distinct probe and that its μ²_ang column is **not** cross-comparable with §3.4's softest-mode μ² (5.1/5.9/5.4×10⁻²) — kills the "seed-44 1.07e-1 ~2× §3.4 max" contradiction by attribution. §3.3's stray "+12–28%" mention synced to "+12–15% (per-seed +5 to +29%)". Appendix D title de-numbered ("resolved by a kick-amplitude probe", not "+12–28%").
- **MF-4 (embed figures).** Fig 1 (`fig1_gmor.png`, GMOR+C2) after §3.1; Fig 2 (`fig2_mo.png`, **headline**) after §3.2; Fig 3 (`fig3_retention_overlay.png`) after §3.3. `\includegraphics` in tex (float `[t]`, `\label`+`\caption`), markdown `![caption](figs/…)` in md. Fig 4/5 left text+table per task.
- **MF-5 (F5 note).** All 5 `\todohead` stripped + command definition + stale header comment removed. Q3: dropped the "few-to-fifteen-percent" parenthetical (§11) → neutral "measured in companion evidence-side work". Q5: row (c) given a neutral "supplementary check used by companion work" line. **Cor-3 footnote (Head DEC 2026-07-08):** CUT the "appears in at least one public instantiation, where it is inert by construction" de-anon clause; footnote reduced to the pure class fact (mean-spectrum penalty degenerate → chaos penalty must use a max/positive-part statistic) — the class fact also remains in the Cor. 2 body. N-2: abstract "All twelve results" → "All results" (13 checks a–m).

### SHOULD-FIX / NICE applied
- **SF-4** — evidence-first framing paragraph inserted before §3.1 (both files) naming §3.2/§3.3 as the load-bearing evidence and §3.1 as the verification that licenses them. (Full subsection reorder **not** done — would break cross-refs; task allowed the framing-paragraph minimum.)
- **SF-5** — reach-pricing "separately-proven" → "second, separately **measured** … secondary and aggregate-only at single-unit scale (+77% MSE when the cap binds, collapses once v_max≥req)".
- **SF-6** — §5 "solved constitutive problem" → "a constitutive problem with an exact solvable core and quantified anharmonic deviations".
- **N-1** — abstract "subsumes" → "contains … as its overdamped limit".
- **N-3** — §4 "our §3.5 contributes" → "reports".

### SF-1/2/3 — left as marked pending slots (NOT fabricated)
Per task ("may need small runs — DO NOT fabricate"). Confirmed the data is **not** in existing outputs (checked `v2-full-runs.md`: the 44% λ̂(T) figure is a protocol note, not a full-regime Mo-estimator sweep; no per-step FLOP ratio; no anchored-3000ep GMOR sweep). Inline `[pending: v2-referee-experiments]` slots added:
- **SF-1** §3.2 — Mo's own λ̂(T=128) estimator overlay across all regimes (predictor-substitution closer for Fig 2).
- **SF-2** §3.3 — per-step FLOP/wall-time ratio CLU-step vs LSTM/LEM-cell (to compute-normalize the 4×; qualitative triad noted as independent of it).
- **SF-3** §3.5 — GMOR tilt-sweep + EP onset re-verified on an anchored 3000-ep checkpoint.

## Charter / claims-matrix compliance notes
- **C-1 (reversed):** no audit paragraph anywhere; corrected mechanisms live only as neutral class theory (F5 Cor. 2 / §11); CLU continuity sentence intact ("the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)").
- **C-2:** §3.1 kept labeled *verification*; §3.2/§3.3/§3.4 labeled *evidence* (framing paragraph reinforces).
- **C-5:** scale qualifiers in-sentence (dim 4, ≤5 seeds, S¹, laptop-CPU) preserved in the rewritten §3.4/App G.
- **CM-1:** crossing + boundedness-by-construction claimed in the approved wording; lowest-plateau claim explicitly disclaimed. Reach term = "real, secondary, aggregate-only". Ladder (+γ 92% / γ_φ −24%) verbatim to CM-1.
- **CM-4:** §3.3 triad wording unchanged (already compliant).
- Every new number traces to `fit-gap-anatomy.md` (loan curve item 2; ladder item 3; reach item 1/3b) or `v2-prefreeze-baselines`/`v2-full-runs` (already-cited); App G carries the flag-provenance inline (commit 9a13455, seeds {0,1,2}, wake-only except γ_φ rung). No number was adjusted/rounded.

## Open editorial questions for the Hub/Head
1. **SF-1/2/3 are genuine small experiments** — recommend the Hub task the analyst (`v2-referee-experiments`) with: (a) Mo λ̂(T=128) full-regime overlay on the trained models; (b) per-step FLOP ratio CLU vs LSTM/LEM; (c) GMOR+EP sweep on an anchored 3000-ep checkpoint. Until then the three `[pending:]` slots stay in the draft; they are honestly flagged, not silent gaps.
2. **Page budget (N-4):** with 3 embedded figures + Appendix G the main text is now well over 5pp of body+appendix. This is intended under C-10 (appendix-maximalism); the dedicated pruning pass near the venue deadline should target §3.1 machine-precision enumerations and the emergent-lifetime double-reporting (§3.3/§3.4/App D). Not self-pruned now.
3. **MF-3 residual μ² mismatch:** §3.4's softest-mode μ² (5.1/5.9/5.4×10⁻²) and App D's per-seed angular μ²_ang (5.45/10.7/2.03×10⁻²) are genuinely different measurements on the same checkpoints. I resolved it by *attribution* (distinct probes, not cross-comparable) rather than by picking one — the honest move without a new run. If the Hub wants a single reconciled μ² table, that is a (small) re-measurement, flag it.
4. **F5 N-2:** I changed "twelve"→"all" rather than pruning row (c) or renumbering — zero-risk and removes the count contradiction; note in case the Head prefers an explicit "12 numbered results + 1 supplementary check" phrasing.
5. **Titles still `[WORKING TITLE: …]`** and authors `[AUTHORS PLACEHOLDER]` in both — untouched per placeholder rule; title workshop pending.

## Proposed handover updates (for the Hub)
- **V2 short at v0.2** (`.claude/papers/v2-short/`, draft.md + draft.tex + figs/ + draft.pdf 395 KiB): all four MF closed; SF-4/5/6, N-1/N-3 applied; SF-1/2/3 pending-slotted → **ready for `paper-referee` re-review**.
- **F5 note at v0.2** (`f5-note.tex` + f5-note.pdf 168 KiB): MF-5 fully closed incl. Head's Cor-3 DEC; **arXiv-clean of margin notes**; N-2 count fixed.
- **New Hub task candidate:** `v2-referee-experiments` scope = the three SF-1/2/3 runs above (all laptop-scale, none blocking submission of the qualitative claims).
- Claims-matrix unaffected; CM-1 crossing/ladder now realized in a shipped draft (was already updated in matrix v1.3).
