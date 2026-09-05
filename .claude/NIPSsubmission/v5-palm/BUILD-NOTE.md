# `NIPSsubmission/v5-palm` — BUILD NOTE (the clean iteration base for V5)

**Deliverable #1 of this pass.** Everything below is measured, with the command named; nothing is asserted.

**Source (read-only, verified byte-identical after this pass):** `.claude/papers/plain/v5/submission.tex` + `figs/` + `neurips_2025_ml4ps.sty` (the plain build; its own note records the device-stripping pass that produced it).
**Built with:** `/Library/TeX/texbin/pdflatex -interaction=nonstopmode submission.tex` ×3 (pdfTeX, TeX Live 2026), run **only inside this folder**.
**Result:** `Output written on submission.pdf (19 pages, 1778065 bytes)` — **0 errors, 0 undefined references/citations.**
**Scope of this pass:** framing/editorial only. **Zero number changes** (proved in §4), zero findings added or dropped, no experiment re-run.

---

## 1 — Page split: reported, ⛔ NOT optimised

| block | pages |
|---|---|
| main text (title → Limitations) | **pp. 1–6** |
| references | pp. 7–8 |
| appendices A–E | pp. 9–19 |
| **total** | **19 pp** |

**Instrument:** `pdftotext -bbox`, text block 72–720 pt — the same instrument the earlier build notes used. On the last main-text page (p. 6) the final text line ends at `yMax = 160.48 pt`, i.e. **13.7 % of the block**. So main text = **6 pages consumed / 5.14 pp of fill**. The source measured **5 pages consumed / 4.90 pp of fill** (p. 5, `yMax = 653.88`, 89.8 % of the block).

⛔ **The page count is reported and deliberately not fought.** For the record, and it is the fact this folder exists to stop us forgetting: **the source measures 5.00 pp of main text at default formatting against a venue limit of 4** — the earlier "4.00 pp" reading was typographic (proved by the plain pass's control build, not asserted). This build adds ~1/4 page of framing on top of that, so it is **~2 pp over a 4-pp limit on real content**. Compression is a later, deliberate pass; nothing here was fitted.

**Main-text length:** 3,400 → **3,673 words** (+273), all of it the §1/§2 audience scoping in §2 below.

**Boxes, reported and not fought:** 3 overfull `\hbox` — **196.50 pt / 604.92 pt / 49.44 pt, numerically identical to the source build's**, i.e. inherited, not introduced. They are the three tables that do not fit the text block at `\small` (the first two do not fit at *any* permitted size and need editorial restructuring — an owed fix, out of scope here). 0 overfull `\vbox`; 9 underfull `\hbox` (same as source); underfull `\vbox` 4 → 12 (more float pages, a consequence of the extra page).

---

## 2 — Audience scoping (the new work in this pass)

Source of the audience facts: the 2025–2026 audience refresh (first-edition venue; proxy = the nearest dedicated memory workshop, the 2025 adjacent workshops, and the organizers'/speakers' own output). **Nineteen exact-once string edits + two reference insertions**, each listed in the pass script; the substantive ones:

**(a) The white space is named, and no system property is claimed.**
- §1: *"The gap is widest at removal. Agent-memory work supplies mechanisms for writing, consolidating and retrieving; for deletion it supplies a timestamp or a dropped row (Rasmussen et al., 2025; Chhikara et al., 2025), whose residue is measurable after the fact (Chakraborttii et al., 2026; Wang & Zhang, 2026). The 2026 evaluation wave asks for update-and-deletion tests (Yang, 2026; Uddin et al., 2026), and a test needs a mechanism to exercise."*
- §2, closing the deletion strand: *"The two literatures leave a seam. The deployed stores delete best-effort and do not state what remains; the formal guarantees are stated for a learned model, not for the store that holds the item. Our result sits in that seam, one level down, and it is a mechanism with measured laws rather than a property of a deployed system."*
- §3.3 opener now ties back: *"…the store-level property an update-and-deletion test (§1) would exercise."*
- ⭐ **Every citation in the white-space framing was already in the paper.** No venue is named anywhere (`PALM` = 0 in the PDF), no accepted-paper census number is quoted, and the claim is a *scarcity of mechanisms*, not a claim about a deployed system.

**(b) The physics is stated in its own terms (the hedge is gone).** *"Physics enters only as the derivation apparatus"* → *"The store is a physical system and we describe it as one."* The paragraph now ends *"The three results below are read off that reduction — a retention law, a diffusion coefficient and a vault factor — and each is then stated as the policy quantity it fixes."* In §3.2 the diffusion coefficient is named as such before it is used (*"The diffusion coefficient of the stored coordinate is $D_\theta=\varepsilon T(2-\gamma)/(2F^2\gamma)$, verified to 1.0068 ± 0.0219 over 25 cells"*) and the vault is stated mechanism-first (refrigerator, then $(\gamma_{\rm eff}/\gamma)^2$). **The policy framing is untouched:** the ABT openings, the three operational questions, and the contribution labels (*retention dial · scoped retention · structural deletion guarantee*) all stand.

**(c) The three-sided reviewer pool, mapped to the three results.**
- *Privacy / membership-inference side:* the leakage paragraph and its TTL laundering control are unchanged and still lead with the control that **fires** (0.983 vs 1.000 exact; 0.559 vs 0.996 at σ_obs = 0.1) — the honesty there is an asset and was not softened.
- *Nearest published retention-dial neighbour:* now cited **and contrasted in one sentence** — *"Titans learns its gate; the same slot in our store is a budget in (μ, γ, T) whose setting is computed rather than trained."*
- *Brain–language-model alignment side (new):* a representational-drift bridge in §2, with the no-biological-claim sentence attached — *"Memory is also studied where it moves on its own: cross-session representational drift reorganizes which units carry a representation while decoding performance is preserved (Aitken et al., 2022; Jude et al., 2023). That is a different phenomenon from the motion we report, which is of one coset coordinate under our own dynamics, and we make no biological claim."*
- **Mandated-once check:** `we make no biological claim` = **1**; the honest-scope sentence (*"We report a mechanism with measured laws, not a system result."*) = **1**.

**(d) Style (`PJ_Writing_Style_Context.md`), applied without touching a claim.** The abstract is rebuilt on ABT with one idea per sentence and three signposted results (*The dial has a computable optimum · Retention is scopeable · Removal is structural*); eleven long compound sentences in §1/§3.1/§3.2/§3.3 were split. ⛔ **Simplify the prose, never the claim:** the do-not-cut list was checked verbatim afterwards (§6, 27/27).

**⚠ Two additions this pass deliberately did NOT make** (missing-verification notes, not omissions): the nearest speaker-side neighbours *ATLAS* (arXiv:2505.23735) and *Nested Learning* (arXiv:2512.24695) are **not** primary-verified in the audience report (search-summary sourced), so they are not cited. They are the strongest candidates for the next cite-check pass.

---

## 3 — Refused terms: verified, and the refusal stands

The source refused *"right-to-be-forgotten"* and *"memory provenance"* as claimed terms, because each names a compliance property of a deployed system while our result is store-level. **That judgment was preserved, and the white-space framing did not relax it.**

| term | source | this build |
|---|---|---|
| `right-to-be-forgotten` / `right to be forgotten` | 0 / 0 | **0 / 0** |
| `memory provenance` / `provenance` | 0 / 0 | **0 / 0** |

**Adopted vocabulary, unchanged:** retention policy ×3 · retention dial ×4 · TTL ×11 · expiry ×1 · scoping/scoped ×16 · deletion guarantee ×2 · membership ×4 · stale ×1 · consolidation ×2 (+1, the lifecycle framing) · best-effort ×2 (+1, the seam sentence).
**Forgetting-word discipline:** `forgetting` ×9 (unchanged) — every occurrence means value degradation or is quoting the cited literature; the paper says **deletion / removal** (×26) wherever it means deletion-on-request, which is the discipline this room requires.

---

## 4 — Numeric two-way check (printed in full)

Instrument: the plain pass's own `numcheck.py` (typography arguments excluded) plus a main-text/appendix split check.

| check | result |
|---|---|
| distinct numeric tokens, whole file | source 570 → **575** |
| **in source, not in build** | **none** |
| in build, not in source | `2022`×2, `2023`×2, `2025`×2, `2026`×5 (citation years re-used in the new framing sentences) + `18`, `11`, `1010716.`×2, `10.1371`, `197`, `234`, `257.` (the two new bibliography entries) |
| **NEW-main tokens absent from SOURCE-main** | **[] (empty)** |
| **SOURCE-main tokens absent from NEW-main** | **[] (empty)** |
| NEW-appendix tokens absent from SOURCE-appendix | `10.1371`, `1010716`, `197`, `234`, `257` — the two new reference records only |
| SOURCE-appendix tokens absent from NEW-appendix | **[] (empty)** |
| appendix + references diff | **2 lines, both insertions** (Aitken et al. 2022; Jude et al. 2023) — every appendix line otherwise byte-identical |

⇒ **The main text's numeric multiset is identical in both directions.** No result number was added, dropped, moved, rounded or re-scaled.

**Reference count:** 30 → **32**. The two additions are the drift bridge's, both primary-verified in the audience scouting record (Aitken et al.: abstract read from the PLOS article page; Jude et al.: title/authors/pages from the PMLR volume index) and both already carried in a verified program bibliography. ⚠ They have not been re-verified in *this* pass — flagged for the cite-check.

---

## 5 — Sweeps (per file, positive-controlled, printed)

Instrument: the V5 submission sweep script (38 zero-list patterns spanning never-quotes, internal apparatus/paths/report IDs, and semantic hermeticity + 7 context-check classes).

- **Positive controls: 14/14 fired ⇒ instrument LIVE** — `107.77`×9 · `106.1`×3 · `0.9001`×2 · `Blelloch`×10 · *"stops answering before it stops leaking"*×2 · `confines`×3 · `8.11`×6 · `Anonymous`×2 · *"introduced as CHLU"*×1 · `verification`×12 · `evidence`×15 · `9.5\times10^{15}`×1 · `0.4586`×2 · `ZERO`×2.
- **Zero-list: 3 hits, all false positives, none new in substance.**
  1–2. `n_{\rm R1}` / `\Gamma_{\rm R3}` in the Appendix-B table header — **this paper's own instrument names**, defined three paragraphs above; the identical false positive the source build reports (source total: 2).
  3. ⚠ **New, and it is a regex artifact:** the commit-hash pattern matches `e1010716` inside the new PLOS article number `18(11):e1010716`. Not a commit hash. (Instrument note for the next sweeper, not a defect in the draft.)
- **Positive control on a non-paper file** (this pass's task file): **9 zero-list hits** ⇒ the sweep is not vacuously clean.
- **Context-check classes, unchanged from source:** `certified` ×3 (2 literature + 1 denial) · `unlearning` ×6 (denial + 1 literature sentence + 4 reference entries) · *"deletion is exact"* ×2 (both store-level-qualified) · `CHLU` ×2 (the continuity sentence + its reference entry) · `0.99985` ×1 (carries "at full load") · `297.8` ×1 ("never the vault number") · `23.39` ×3 (all designed-only / falsifier-fired).
- **Semantic hermeticity (C-8):** `companion` / `sibling` / *our other short* / *the program* / `forthcoming` / *in preparation* = **0**. The theory note remains *(Anonymous, 2026)*; the continuity sentence *"the CLU (Causal Learning Unit), introduced as CHLU in Jawahar & Pierini (2026)"* is present verbatim, once.
- **Anonymization posture, unchanged:** `\author{}` empty · `\@notice` suppressed · PDF metadata scrub retained (`pdfinfo`: Title/Author/Subject/Keywords/Creator/Producer all empty) · decompressed-PDF string sweep `/Users` 0 · `Desktop` 0 · `.claude` 0 · `NIPSsubmission` 0 · `v5-palm` 0 · `chlu/` 0 · `ml4ps` 0 · **no venue string (`PALM` 0)** · `Pierini` 2 = the sanctioned continuity sentence + its reference entry. **Code-inclusive path neutralization** is carried by the closing note, unchanged: *"Any supplementary or linked material, including code, is anonymized; only an anonymized snapshot may be linked."*

---

## 6 — Carried over from the source: each item verified, not assumed

| item to carry | check run | result |
|---|---|---|
| no page-fitting typography | `grep -c` for `\scriptsize`, `\tiny`, `\footnotesize`, `multicols`, `\raggedbottom`, `\@startsection`, float-fraction and skip overrides, reduced `\includegraphics` widths | **0 in source, 0 here** |
| exactly **five** `\small` exceptions, on the tables that physically overflow | `grep -n '\small'` | **5 in source, 5 here, same five tables** (tex lines 130, 173, 188, 267, 308): the $T>0$ budget table · the four-instrument table · the instrument-gap table · the emergent-confinement/laundering table · the laundering-control table |
| no bold outside structural headers | `grep -c '\textbf'` | **0** |
| author token absent from prose/captions/labels/filenames | standalone-token sweep `\bMo\b` (occurrences, not lines) | **1 occurrence = the bibliography entry** (expressly kept); prose/captions/labels/filenames **0**. Positive control on the pre-plain source: **2 occurrences.** ⚠ "Morse"/"Moser" survive elsewhere in the program; **this file contains neither (0 and 0)**, so nothing could be lost to the regex. Pronoun sweep (`his/His/he/him`): **0** |
| seven restored banked figures with provenance-bearing captions | appendix diff (§4) | **byte-identical captions**; all seven present |
| single-seed figures labelled as such | string check | *"Multi-seed status: one trained designed-$SO(2)$ checkpoint (seed 44)…"* ×2 (Figs 2 and 3) and *"two emergent seeds and one matched designed control, an $n<3$ cell"* ×1 (Fig 8) — **all three present verbatim** |
| the `v5-referee-v02` do-not-cut list | 27 exact-string probes, source vs build | **27/27 present, counts equal, 0 failures** — the leakage sentence · the exact-deletion form with its three conditions and the recency exclusion · Blelloch–Golovin at **every** deletion site (abstract, contributions, §3.3, appendix, no-priority clause; `Blelloch` ×10 in both) · the lifecycle's two riders · the substrate-scope sentence · the score sentence · the designed-symmetry precondition · the `fdt`+Newtonian fine print (both sites) · the emergent-arm caveats (no σ_θ ratio; the θ=π confound; the contrast number designed-only) · the trilemma corner · the R₅₀ differentiator · the store-level-only clause · the no-certified-unlearning clause · scale-as-scope-choice · the CLU continuity sentence · the anonymization note |
| source folders untouched | sha manifests before/after, `diff` | `plain` (30 files) · `palm-variant` (16) · `v5-short` (25) · `v2-short` (21) · `v2-neurreps-descoped` (10) · `neurreps-variants` (11) — **all six byte-identical**; `pdflatex` ran **only** in this folder |

⚠ **Inherited, not introduced (all pre-existing in the source, listed so nobody re-discovers them as new):** the two tables that overflow at any permitted size · the instrument IDs `I-J` / `I-R1` / `I-R2` / `I-R3` still in the appendix text and the Fig. 9 caption (the caption-sync worklist is still unexecuted) · the `Anonymous (2026)` theory-note reference · the third-person self-citation in a double-blind build · the NeurIPS-family style file standing in for the venue template · the line numbers printed by that style file.

---

## 7 — Figure inventory (11, unchanged from the source build)

| # | file | home | evidences | multi-seed status printed in caption |
|---|---|---|---|---|
| 1 | `fig1_damping_optimum.png` | **main text (headline)** | the collapsed V-curve, 5 designed + 3 emergent | 5 designed (verification) + 3/3 emergent (evidence) |
| 2 | `figB_dlaw.png` | App. A | the 25-cell diffusion law 1.0068 ± 0.0219 | **single checkpoint (seed 44)**, stated |
| 3 | `figB_signflip.png` | App. A | the sign flip and $n_{1/2}\propto1/T$ | **single checkpoint (seed 44)**, stated |
| 4 | `figB_massive_vs_flat.png` | App. A | the two regimes and the exact latch | 5 designed seeds |
| 5 | `figA1_damping_optimum_full.png` | App. B | the full-size collapse with annotations | same eight curves as Fig. 1 |
| 6 | `figC_lambda_coset.png` | App. B | the emergent V-curve + the $10^{-3}$ latch failure | 3 emergent + designed control |
| 7 | `figC_register_capacity.png` | App. B | "no continuous coset register", ≈1–1.6 bits | 3 emergent + designed control |
| 8 | `figC_Tstar.png` | App. B | the crossover $T^\star\approx3\times10^{-3}$ | **two emergent seeds + one designed control, an $n<3$ cell**, stated |
| 9 | `fig2_two_instruments.png` | App. B | the V-curve on a second instrument | 3 emergent seeds |
| 10 | `fig2_vault.png` | App. C | the refrigerator, the 8× contrast, the 107.77 ± 4.78× vault | 3 designed seeds |
| 11 | `figC2_vault_emergent.png` | App. C | the vault on an emergent register | 3 emergent seeds × 2 temperatures |

---

## 8 — Open items for the Head's iteration on this base

1. **Title** — still the source's; the Head's call. Workshopped candidates are in the pass report.
2. **Compression** — deliberately not attempted. The honest gap to a 4-pp main text is ~2 pp of *content*, so the next pass is a scope decision (what leaves the main text), not a typography decision.
3. **Cite-check owed** — the two new drift references (re-verification) and the two speaker-side neighbours deliberately not cited (ATLAS, Nested Learning) if the Head wants them in.
4. **Inherited editorial debts** — the two unfittable tables; the `I-J`/`I-R*` caption-sync worklist.
