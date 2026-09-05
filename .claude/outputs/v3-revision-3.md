# v3-revision-3 — paper-writer report

Task + acceptance criterion: consolidating V3 revision — reframe MF-1 per CM-9-RESOLVED (near-reject retired), embed final figures with per-figure grade labels, close SF-1/2/3 + N-1, splice-or-flag the remaining bib. **Status: done** (bib item 4 = flagged, not spliced; see §5).

Artifacts: `.claude/papers/v3-short/{draft.md, draft.tex, draft.pdf, CHANGELOG.md, figures/}` → **v0.4**. No tracked files touched (no git footprint; `.claude/**` is gitignored).

---

## 1. What I did (by task item)

**Item 1 — MF-1 REFRAMED.** §3.2 rebuilt into three explicitly-ordered movements, structural-first per `v3-interference-extra` §4:

- **(i) Structural identity, upgraded from assertion to exact measurement.** R ≡ 0 off the coupling graph (0 of 4,656 off-graph entries nonzero over 72 chain runs; monolith 4,656/4,656) ⇒ boxed identity **S_B = deg(B)·R̄_edge**, verified `S/(deg·R̄_edge) = 1.000000` in every topology×N cell. Topology-control table (chain / ring / circulant-4, 8 seeds) in main text: **ring (deg≡2) b = +0.071 ± 0.183, p = 0.391 (flat)**; circulant-4 b = −0.066 ± 0.315, p = 0.636; degree doubling ⇒ S ×2.01 at N=16 (quoted as the identity, never as a "2.0× constant" — the ±30% R̄_edge init scatter is stated).
- **Chain residual stated preemptively**, as instructed: its own dedicated paragraph ("We state the one residual preemptively"), giving b = +0.46 ± 0.31 (12 seeds, N∈[2,16]) and +0.26 ± 0.17, p = 0.008 (8-seed topology control), the coordination prediction +0.289, and the degree-normalized b(S/deg) = +0.17 ± 0.31. Ends: *"We therefore never describe a chain as 'flat in N': the correct statement is degree-bounded, and the ring is where flatness is measured."*
- **(ii) Growth claim, 12 seeds × 6 sizes.** Full S-vs-N table (both arms) + slopes: modular +0.463 ± 0.311 vs monolith **N^{1.18±0.17}**; Welch p = 3.3e-4, paired p = 5.5e-4, 11/12 seeds; modular saturates (b[8,16] = −0.23 ± 0.37) vs monolith +0.79 ± 0.09; separation ×20,649 at N=16.
- **(iii) Block-monolith control folded into main text** (referee missing-exp #2): 5-arm table (params, S(8), S(16), per-pair R̄, slope, non-edge R). Verdict wording: **"parameter separation, not block structure"**; capacity decoupled (1,185-param tied = worst; 18,960-param untied = exactly zero).
- **`block_tied` promoted to co-primary foil** (per task's "consider"): reported in the main table and given its own closing paragraph (deep-sets, per-pair R̄ = 0.97, 6.2× the naive monolith at N=16, exceeds S=1 by N=4), with the explicit statement that *we quote the naive monolith in the headline, which understates our case*.
- **Honest framing sentence landed verbatim-in-spirit** in §3.2(iii), §4 and App G: *"nothing physics-specific buys the firewall — parameter separation does, and `block_untied` is a strictly better firewall than our lattice (S≡0 vs R̄_edge ≈ 6.5e-5). The modular lattice is `block_untied` plus an O(κ²) graph-local leak … what the physics buys is that the lattice pays that price while retaining a single joint symplectic Hamiltonian with priced, graph-local communication. A network of disconnected potentials has a perfect firewall and nothing to say across it."*
- **Terminology purge:** every "O(1)-in-N vs O(N)" → **"coordination-bounded vs width-linear"** (abstract, contributions, §3.2, §4, figure captions, title slug). Verified by grep: zero occurrences of "flat in N" applied to a chain; the only "statistically flat in N" instances are the **ring** (approved CM-9 wording) and the explicit disavowal sentence.

**Item 2 — Figures.** All 8 embedded in both `draft.md` (image + blockquote caption) and `draft.tex` (`\includegraphics` + `\caption`), each caption carrying a grade label (SF-3):

| # | file | placement | grade |
|---|---|---|---|
| 1 (headline) | `fig1_scaling_curve.png` (new 6-point, 12 seeds) | §3.2 | [evidence] |
| 2 | `fig2_banding.png` (from `v3-banding-figure`) | §3.4 | [evidence] |
| 3 | `fig3_pricing_parity.png` | §3.3 | [evidence] |
| 4 | `fig4_lattice_pricing.png` | §3.1 | [verification] |
| 5 | `mem_grad_summary.png` | §3.5 | [structural] |
| 6 | `fig1_interference_bars.png` (demoted from headline) | App D.1 | [evidence] |
| 7 | `fig_block_monolith.png` (new) | App H.1 | [evidence] |
| 8 | `fig_coordination.png` (new) | App H.2 | [evidence] |

v0.3's **swap note is discharged** (deleted, not carried) and the **Fig-2 missing-asset flag is discharged**. Note recorded in the asset map: the Fig-1 asset's own title reads "…vs *O(N)* blow-up" — that O(N) refers to the **monolith** (fitted 1.18±0.17), and no arm of the modular lattice is called O(1)/flat anywhere; I did not regenerate the analyst's figure.

**Item 3 — SF-1 / SF-2 / N-1.**
- **SF-1:** S=1 crossing now reads *"by N≈5.9 the aggregate cross-unit force perturbation a monolithic unit receives exceeds the magnitude of its own intended update"*, with the storage gloss explicitly refused and cross-referenced to the unmeasured dynamical half-life (App C). Same wording in the Fig-1 caption and §5.
- **SF-2:** "≤8%" → **"≤7.5% relative to the registered prediction"** in abstract, contribution 3, §3.3, Fig-3 caption. §3.3 additionally prints the five residuals {7.5, 0.9, 2.1, 4.5, 3.2}% and *volunteers the hostile row*: "211 predicted vs 195 measured — which is 8.2% if one instead normalizes by the measured value, so we state the denominator." Nothing rounded up.
- **N-1:** contribution 2 carries in-sentence scope: "measured at N≤16 on chain/ring/circulant-4 lattices of 2-dim units with MLP potentials, κ=0.05, 8–12 seeds, at initialization, laptop-CPU."

**Item 4 — Bib.** `.claude/outputs/venue-follow-up.md` **does not exist** at this revision (checked). Per instruction: slots left marked, nothing fabricated. Added an explicit ⚠ block at the head of the references in `draft.md` and an italic note in `draft.tex` naming the three unresolved strings (Mo 2026; Di Bernardo/Keller; checkpointing-O(√T) + RevNet/momentum-net) and the reason. Two new marked `[·]` reference lines added so the reversible §4 anchors are visible in the reference list rather than only in prose.

**Item 5 — PDF rebuilt.** `tectonic 0.15.0`, exit 0, **19 pp**, **zero overfull hboxes** (I shrank five wide tables to `\footnotesize` + reduced `\tabcolsep`; one `\\[$\cdot$]`-parsed-as-optional-arg error fixed with `{}`). Cosmetic underfull hboxes only. Toolchain present on this machine, so the build is real, not pseudo-verified.

**Not done, deliberately (C-10):** referee SF-4 / N-a pruning (demote §3.5/§3.6, Fig 3→appendix). Reserved for the dedicated pruning pass, as the task directs. The draft is *more* appendix-maximal after this revision, not less.

---

## 2. Evidence backing each changed section

| section | claim | source |
|---|---|---|
| Abstract, §1 C2, §3.2(i) | R≡0 off-graph, 0/4,656 over 72 runs; S=deg·R̄_edge to 6 dp | `v3-interference-extra` §1b, §1c, `analysis_stdout.txt` exact-zero audit + coordination identity block |
| §3.2(i) topology table | chain/ring/circulant-4 S, degrees, slopes, p-values | `v3-interference-extra` §1c; `analysis_stdout.txt` |
| §3.2 residual ¶ | b=+0.46±0.31, +0.26±0.17 (p=0.008), pred +0.289, b(S/deg)=+0.17±0.31 | `v3-interference-extra` §1, §1c |
| §3.2(ii) table + slopes | S(N) both arms, 12 seeds; Welch/paired p; b[8,16]; ×20,649 | `v3-interference-extra` §1, §5 handover |
| §3.2(ii) S=1 crossing | N≈5.9 (S(4)=0.640, S(6)=1.024) | `v3-interference-extra` §1 |
| §3.2(ii) κ², mass-independence, through-training | slope 1.99 @N=4; bit-identical banded≡uniform (max|ΔR| = 0.000e0); R_off 4.2e-5→8.9e-5 @N=2 epochs 0/150/300 | `v3-interference-ntk` items 1/3; `v3-interference-extra` §1d; `through_training.json` (verified 2×2 ⇒ N=2) |
| §3.2(iii), App H.1 | 5-arm block table, exact-zero audit, R[A,A]=1.000000, 129,033×, 6.2× | `v3-interference-extra` §2, `analysis_stdout.txt` |
| §3.3 | residuals {7.5,0.9,2.1,4.5,3.2}%, 211-vs-195 | `v3-referee` SF-2 (source `sync_rel_err`) |
| §3.4, Fig 2 | 1.180/2.416/6.924/12.791, gap 0.000 5/5 | `v3-band-selection` items 1–2; `v3-banding-figure` |
| App A.2b | commit `37dc664`, JAX 0.9.0, seeds {0..11}/{0..7}, N grids, probe path, bit-exact anchor | `v3-interference-extra` flag-provenance §5 (transcribed, not recomputed) |
| App D.1 | 3-seed table + reconciliation + the ×2.56/×2.18 ±2.27 process negative | `v3-interference-ntk`; `v3-interference-extra` §1 |
| §4 related-work prose | positioning lifted/adapted from **`scout-modular-interference`** (spliced in v0.3; re-worded here to degree-bounded/width-linear + parameter-isolation framing) | `scout-modular-interference.md` |

**M4 / cross-section-contradiction sweep (C-7).** Three latent contradictions found and closed:

1. **"parameter-matched monolith" (pre-existing, would have been fatal).** The abstract/§3.2/§4 called the monolith *parameter-matched* while the new §5 correctly admits **19,112 modular vs 2,177 monolith params at N=16** (`v3-interference-extra` limitation #3). Fixed everywhere: the arms are **width-matched (same family, hidden 32), not parameter-matched**; the abstract now says so in-sentence and points at the capacity controls. **This was a real reviewer-quotable defect in v0.3, not introduced by this revision.**
2. **Neighbour-leak magnitude drift.** v0.3 quoted `≈2×10⁻⁵` (3-seed R_off mean); the 12-seed per-edge quantity is `R̄_edge ≈ 6.5×10⁻⁵`. All main-text/abstract/§4 mentions now use the 12-seed R̄_edge; the 3-seed table survives in **App D.1** with an explicit reconciliation (bit-exact reproduction on shared seeds ⇒ seed sampling, not metric drift). The `1:9,000` ratio is retired in favour of the reported `×20,649 @N=16`. No ratio was recomputed by me.
3. **"persists through training" scope.** v0.3 implied this held for the reported N. Verified against `through_training.json`: the R matrices are 2×2 ⇒ **N=2 only, 3 seeds**. Now scoped in §3.2, §5 and App C, and added to the unmeasured list.

Also: §3.1's unrelated "flat in N" (symplectic error) reworded to "constant in N to within the f32 Hessian floor" so the forbidden phrase appears nowhere near an interference claim.

---

## 3. Charter / claims-matrix compliance

- **C-1:** no audit paragraph. Unchanged. ✅
- **C-2 / SF-3:** grades now on every figure caption *and* every §3.x header; new "structural" grade defined in §1's reporting-discipline sentence. §3.2 relabeled to "Evidence" with the honest addendum that the exact zeros are *analytic*, the finite magnitudes *measured at init*. ✅
- **C-5:** in-sentence scope on contribution 2, abstract claim (1), §4 related work, all interference captions. ✅
- **C-6:** certificate fine print sits next to its claim — the ≤7.5% denominator, the 8.2%-if-normalized-by-measurement row, the S=1 force-not-storage clause, the "exact zeros are training-invariant, finite magnitudes are init-scale" clause. ✅
- **C-7:** new **A.2b** flag-provenance table (commit, JAX version, seeds, N grid, probe path, measurement point, statistics, bit-exact metric anchor). A.2 gains a supersession row. ✅
- **C-8 / M1:** hermetic. No cross-short citation. I **declined to add a Deep Sets citation** for `block_tied` (I was confident of the string, but it is not in any scout report's verified list) — the architecture is described, not cited. Flag below if the Hub wants it. ✅
- **C-9:** two negatives added to App G (the superseded 2-point/3-seed reading; "the firewall is not a physics result"); App C's block-monolith item struck-through-and-retained as a discharge record; four new unmeasured items added. ✅
- **C-10:** nothing pruned. Main text grew (§3.2), appendices grew more (new App H, App D.1). ✅
- **CM-9 (v1.6):** the wording follows `v3-interference-extra` §4's recommendation. Canonical phrases used: *degree-bounded, not width-dependent*; *coordination-bounded vs width-linear*; *N^{1.18±0.17}*; *S = deg·R̄_edge*; *parameter separation, not block structure*; *force perturbation, not storage*. **CM-3 absent** (no energy-as-confidence claim anywhere). CM-10's ≤7.5% now matches the matrix's "≤8%" only in the honest direction (tighter) — **matrix §CM-10 still reads "pointwise to ≤8%"; recommend the Hub tighten it to ≤7.5% in lockstep** (see handover updates).

---

## 4. Verification (commands + observed output)

```
tectonic draft.tex   # exit 0
  → first pass: error draft.tex:379 "Missing number, treated as zero"   (\\ followed by [$\cdot$])
  → after {} fix: builds; 5 overfull hboxes (10.9 / 35.2 / 62.2 / 69.7 / 25.0 pt) in the new tables
  → after \footnotesize + \tabcolsep{4pt,3pt}: **0 overfull hboxes**, underfull only
mutool info draft.pdf → Pages: 19        (894,863 bytes, 2026-07-09 20:18)
grep 'flat in N' draft.md  → only: ring (deg≡2) + the explicit "we never call a chain flat in N" disavowal
grep 'le8\%|1:9,000|2\times10^{-5}|parameter-matched shared-potential' draft.{md,tex} → 0 hits
python: through_training.json R matrices are 2×2 → through-training N=2 (scope now stated)
```

Figures copied into `figures/`: `fig1_scaling_curve.png` (76,481 B, byte-identical to `v3-interference-extra/fig_scaling_curve.png`), `fig2_banding.png` (66,339 B), `fig_block_monolith.png`, `fig_coordination.png`. No analyst artifact modified.

---

## 5. Open editorial questions (for the Hub / Head)

1. **Bib blocker.** `venue-follow-up.md` does not exist. Three marked slots remain (Mo; Di Bernardo/Keller; checkpointing + RevNet/momentum-net). **The draft cannot go to `v3-referee-2` as camera-ready-ish with three `[·]` in the reference list** — the referee will (correctly) flag it. Either spawn the bib micro-pass or accept a referee finding.
2. **Deep Sets citation.** `block_tied` is a real, nameable architecture. May I cite Zaheer et al. (2017)? It is published and citable (C-8-legal), but no scout report verified the string, so I left it uncited. One-line Head/Hub decision.
3. **CM-10 lockstep.** The matrix row still says sync "pointwise to ≤8%". The draft now says ≤7.5% (of the prediction). **Matrix should be tightened** or a reviewer reading both finds a (benign, but real) mismatch.
4. **Headline-figure title.** The Fig-1 PNG's own title contains "O(N) blow-up". It is correct (it describes the monolith) and I documented it in the asset map, but if the Hub wants the figure text itself to read "width-linear", that is a `results-analyst` re-render, not a paper-writer edit.
5. **The strongest sentence in the paper is now a concession.** §3.2(iii) says, in the main text, that `block_untied` is a *better* firewall than the CLU lattice. I believe this is right (it is what the data say, it pre-empts the sharpest attack, and it converts the contribution from "physics gives you a firewall" to "physics gives you a *priced channel* through one"). But it is a genuine load-bearing reframe of the paper's selling point and the Head should see it. It also implies the **title** should perhaps foreground the *price list*, not the firewall — flagged for the title workshop.
6. **Foil choice.** I report both foils and headline the naive monolith (per CM-9's canonical numbers). If the Head prefers `block_tied` as the *primary* foil, the abstract's "≈0.19 / N^{1.18}" would become "0.97 / N^{1.29}" and the separation grows to 129,033×. Cheap swap; matrix wording would need updating first.

## 6. Risks

- §3.2 is now ~2.5 pp of a 4–5 pp short. The pruning pass has a real job (SF-4 already flags §3.5/§3.6).
- Everything in §3.2 except the κ²-sweep, the through-training panel and the 3-seed App-D table is **measured at initialization**. I have said so five times (§3.2, Fig-1 caption, §5, App A.2b, App C). If a referee still calls this the paper's throat, the fix is the through-training run at N∈{4,8,16}, not more prose.

---

## Proposed handover updates (for the Hub)

- **V3 draft → v0.4; MF-1 retired.** §3.2 reframed structural-first per CM-9-resolved; block-monolith control folded into main text; `block_tied` reported as co-primary foil; "O(1)-vs-O(N)" eliminated program-wide in this draft. SF-1/2/3 + N-1 closed. Figures 1–8 embedded with grade labels. PDF 19 pp, clean build.
- **Two pre-existing defects fixed, both reviewer-quotable:** (a) the monolith was described as **"parameter-matched"** in the abstract/§3.2/§4 while it is only width-matched (19,112 vs 2,177 params at N=16) — corrected everywhere; (b) **"persists through training"** was scope-free but the source JSON is N=2 only — now scoped and added to the unmeasured list. Suggest checking whether (a) leaked into any other short or into the matrix's CM-9 evidence column.
- **CM-10 needs a lockstep tightening:** "sync pointwise ≤8%" → **"≤7.5% relative to the registered prediction"** (max residual on the weakest-coupling lattice; 8.2% if normalized by measurement — state the denominator). The V3 draft has already moved.
- **CM-9 evidence column:** consider adding "arms are width-matched, not parameter-matched; param-matched sweep unrun" to the scope cell, so no future draft re-introduces the error.
- **Blocking for `v3-referee-2`:** the three bib strings (venue scout report absent). Recommend either running the bib micro-pass first, or telling the referee the slots are known-open so the finding isn't spent.
- **Head decision items:** (i) Deep Sets citation for `block_tied`? (ii) primary-foil choice (naive monolith vs deep-sets tied) — matrix wording follows whichever; (iii) the §3.2(iii) concession ("nothing physics-specific buys the firewall; the physics buys a priced channel") is now main text and arguably changes the paper's headline emphasis toward the price list.
