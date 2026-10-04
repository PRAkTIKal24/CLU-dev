# v3-revision-2 — paper-writer report

**Task + acceptance criterion:** splice the scout modular-interference bib into V3 §4 (close editorial Q5) + wire figures via `\includegraphics` (close Q3); acceptance = zero `[·]` anchors in the modular slot, figures embedded, PDF builds. Edit `.claude/papers/v3-short/` in place; CHANGELOG v0.3.
**Status: done.** All three task items executed. `draft.md` (canonical) + `draft.tex` synced; PDF rebuilds (tectonic exit 0; 530 KiB, up from 141 KiB with 4 embedded figures; cosmetic underfull-hbox warnings only). CHANGELOG v0.3 added. **No tracked repo files touched — no git footprint** (all under gitignored `.claude/papers/`).

## What I did (by task item)

### Item 1 — §4 modular-interference bib splice (closes editorial Q5)
- Replaced the placeholder note `*(Specific published anchors to be finalized from the scout bibliography; see report.)*` in the §4 "Modular vs. monolithic" paragraph with the **scout's guard-railed lift paragraph**, spliced with the 10 verified citations named in the task, in both `draft.md` and `draft.tex`.
- **Citations used (exactly the task list, from `scout-modular-interference`):** McCloskey & Cohen (1989); French (1999); Jacobs et al. (1991); Shazeer et al. (2017); Kirkpatrick et al. (2017, EWC); Mallya & Lazebnik (2018, PackNet); Doan et al. (2021, NTK-overlap); Riemer et al. (2019); Yu et al. (2020); Boopathy et al. (2025). Rendered author-year inline (matches the draft's manual-list citation style; no bibtex toolchain in the .tex).
- **Novelty framing per scout verdict (CLEAR at specific-claim / CROWDED at neighbourhood):** the paragraph credits the two remedy families (prevent-by-construction: Jacobs/Shazeer/Kirkpatrick/Mallya; measure-to-minimize: Doan/Riemer/Yu) as owned prior art, and positions CHLU as the *conjunction none owns* — interference **measured as a coupling-resolved kernel through training**, a *priced* graph-local firewall. **Doan et al. (2021)** (measured-but-monolithic data-overlap) and **Boopathy et al. (2025)** (modular *sample-complexity* separation, not cross-talk) named explicitly as the sharp differentiators.
- **All 10 refs added to the References list** in both files.
- **Numbers = CM-9 approved wording only:** monolith ≈20% (0.20), modular neighbour ≈2×10⁻⁵, non-adjacent *exactly* zero, leak ∝κ² (zero at κ=0), architectural **O(1)-in-width vs. O(N)**. No new/unapproved numbers; ratio 1:9,000 / slope 1.99 left to §3.2 where they already live.
- **C-5 in-sentence scope qualifier retained:** *"(Scope: 2-dim units, chain topology, MLP potentials, N≤8, learned potentials, laptop-CPU; see §3.2 and Appendix D.)"*

### Item 2 — Figures (closes editorial Q3)
- **Scaling-curve PNG has NOT landed.** Confirmed: no `.claude/outputs/v3-scaling-figure.md`, no `fig_scaling_curve.png` in `v3-interference-ntk/`. Per task fallback, **embedded the bars figure as Fig 1 with an explicit one-line swap note** (in both the tex caption and the md asset-map).
- Copied assets into a self-contained `figures/` subdir (paper dir is gitignored) and wired **4 `\includegraphics`** in `draft.tex`:
  - **Fig 1 (headline, §3.2)** = `figures/fig1_interference_bars.png` — firewall bars; caption carries the swap note (→ `fig_scaling_curve.png` once `v3-scaling-figure` ships).
  - **Fig 3 (§3.3)** = `figures/fig3_pricing_parity.png` — pred-vs-measured sync parity.
  - **Fig 4 (§3.1)** = `figures/fig4_lattice_pricing.png` (from `v3-lattice-build/exp_lattice_pricing.png`) — designed-lattice price law, labeled **verification**.
  - **Fig 5 (§3.5)** = `figures/mem_grad_summary.png` — reversible O(1)-memory trade, labeled **structural measurement on untrained models**.
- Captions carry source-report + seed-count + scope + verification/evidence/structural labels (C-2/C-5).

### Item 3 — F5 citation string (Q4)
- **No action** per task: the theory-note citation stays **"(Anonymous, 2026)"** until the F5 arXiv id exists. Unchanged in body and References.

## How I verified
- Build: `cd .claude/papers/v3-short && tectonic draft.tex` → **exit 0**, `draft.pdf` 530 KiB. All 4 graphics resolved (tectonic errors on missing graphics; it did not). Only 5 residual underfull-hbox warnings (cosmetic), zero overfull, zero errors.
- `grep -c includegraphics draft.tex` = **4**; all 4 `figures/*.png` present on disk.
- `grep '\[·\]' draft.tex` = **none**. `grep '\[·\]' draft.md` = only the two out-of-scope reference-string placeholders (Mo; Di Bernardo/Keller — see risks).
- Modular slot: the "to be finalized from the scout bibliography" note is **gone from the §4 modular paragraph** in both files; the three sharp-differentiator citations (McCloskey & Cohen 1989 / Doan et al. 2021 / Boopathy et al. 2025) confirmed present in the tex body.
- md↔tex parity: modular paragraph + 10 refs added to both; numbers identical.

## Compliance self-check
- **Claims-matrix CM-9:** modular paragraph uses only approved numbers (≈0.20 / ≈2e-5 / exact-0 / ∝κ² / O(1)-vs-O(N)); metric-discipline (report R, not NTK cosine) untouched in §3.2/App D.
- **CM-3 (forbidden) absent** — no energy-as-confidence/routing content added.
- **C-1:** no physics-audit paragraph introduced (unchanged; charter-compliant).
- **C-2:** Fig 4 caption = *verification* (designed testbed); Figs 1/3 = *evidence* (trained lattices); Fig 5 = *structural measurement on untrained models*.
- **C-5:** modular paragraph carries in-sentence scope qualifier; figure captions carry seed/dim/CPU scope.
- **C-8 / rule 5 (hermetic):** all 10 new citations are *published, external* prior art (verified by the scout against arXiv/dblp/DOI); no cross-short citation added; no fabricated strings.
- **Rule 3 (every number traces):** no new numbers improvised; all modular figures inherit App A flag-provenance tables (A.1/A.2/A.3/A.6 already present).

## Open questions / follow-ups / risks (for the Hub)
- **Fig 1 swap (Q3, still open):** the intended headline O(N)-vs-O(1) **scaling-curve** PNG (`v3-scaling-figure` → `fig_scaling_curve.png`) had not landed at build time. Bars figure is embedded meanwhile with an in-caption swap note. **When the analyst ships it, swap `figures/fig1_interference_bars.png` → the scaling curve** (drop the swap note). One-line change.
- **Fig 2 (banding) has NO PNG asset** — `.claude/outputs/v3-band-selection/` contains no figures. §3.4 text references "Figures 2–3"; Fig 2 is currently unembeddable. **Missing-asset flag to Hub:** needs a `results-analyst` render (banding degradation curve matched<uniform<orthogonal<anti + FFT-selector overlay), or drop the "Figure 2" pointer in §3.4. Data exists in `v3-band-selection` App B tables.
- **Residual `[·]` are OUT OF THIS TASK'S SCOUT SCOPE (not fabricated, rule 5):**
  1. `draft.md` References: `Mo, [·] (2026)` and `Di Bernardo et al.; Keller, [·]` — belong to `mo-deep-read` / `di-bernardo-skim`, which this task did not supply verified bib strings for.
  2. §4 **reversible** paragraph (both files): `*(Specific published anchors — checkpointing and RevNet/momentum-net — to be finalized from the scout bibliography.)*` — RevNet (Gomez et al.) / gradient-checkpointing (Chen et al.) / momentum-net anchors are a **separate bib gap** not covered by `scout-modular-interference`. **Recommend a follow-up scout** (or Hub-supplied strings) before the `v3-referee` w10 pass; I did not fabricate these well-known-but-unverified citations.
- **Editorial Q4 (F5 citation string):** still pending live arXiv id; body + References carry "(Anonymous, 2026)". V2 and V3 must adopt the identical final string.
- **Editorial Q6 (venue length):** unchanged — with 4 figures now embedded the short is longer; the §3.5 reversible section and §3.1 wormhole-skeleton sentence remain first pruning candidates for the dedicated pruning pass (do not prune now, C-10).

## Files touched (all gitignored `.claude/papers/v3-short/`)
- `draft.md` — canonical: §4 modular paragraph (bib splice); 10 References added; Figures asset-map rewritten (embed state + swap flag + Fig-2 missing flag).
- `draft.tex` — same modular paragraph + refs synced; 4 `\includegraphics` figures (Figs 1/3/4/5); References `\small\raggedright` + 10 entries; rebuilt → `draft.pdf` (exit 0).
- `figures/` (new) — 4 local PNG copies: `fig1_interference_bars.png`, `fig3_pricing_parity.png`, `fig4_lattice_pricing.png`, `mem_grad_summary.png`.
- `CHANGELOG.md` — v0.3 line.

## Proposed handover updates (for the Hub)
- **V3 short v0.3 ready for `v3-referee` (w10).** Editorial **Q5 (modular bib) CLOSED** — §4 spliced with 10 verified refs + guard-railed lift paragraph, Doan/Boopathy as differentiators, CM-9 numbers only. **Q3 (figures) substantially closed** — 4 figures embedded, PDF builds; the headline Fig 1 is the bars fig with a swap note pending the scaling-curve PNG.
- **Two residual bib gaps for a w9/w10 scout or Hub decision (NOT fabricated):** (i) RevNet/checkpointing/momentum-net anchors in the §4 reversible paragraph; (ii) Mo + Di Bernardo/Keller reference strings. Both are genuinely outside `scout-modular-interference`'s remit.
- **One missing figure asset:** Fig 2 (banding) has no PNG — analyst render needed or drop the pointer.
- **Q4 (F5 citation string) unchanged** — "(Anonymous, 2026)" until the arXiv id lands (matrix §4 open slot); V2/V3 must match.
