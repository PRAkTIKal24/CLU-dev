# nips-v5-clean — paper-writer report

Task + acceptance criterion: build `.claude/NIPSsubmission/v5-palm/` as V5's **clean iteration base** — the plain build carried forward, plus the 2025–26 audience scoping (white-space positioning without a system claim · stop hedging the physics · the three-sided reviewer pool) and a strict PJ-style prose pass — with **zero number changes**, all sweeps printed, page count reported but not optimised, `BUILD-NOTE.md` as deliverable #1.
Status: **done.** 19 pp (main text pp. 1–6), 0 errors, 0 undefined references, all checks printed below.

**DIAL DECLARATION (echoed): none — framing/editorial pass; zero number changes.** Laundering control: n/a (no performance number produced; the paper's own TTL laundering control is preserved, not re-run). Falsifies: n/a. Does not falsify: n/a.

> ## ⚠ DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — needs an owner)
> 1. **⛔ Scope decision owed on this base, not a typography decision.** Main text is **6 pages consumed / 5.14 pp of fill** here against the source's 5 pp / 4.90 pp — and the source's own 5.00 pp already sits **1 pp over a 4-pp venue limit on real content**. Getting to 4 pp now costs ~2 pp of *content*. **Owner: the Advisor/Head** (what leaves the main text), at the next V5 review.
> 2. **⚠ Cite-check owed on two additions and two deliberate non-additions.** Added: Aitken et al. (2022) and Jude et al. (2023) for the representational-drift bridge — both primary-verified in `outputs/neurreps-audience-scout.md`, **not re-verified in this pass**. Deliberately **not** cited: *ATLAS* (arXiv:2505.23735) and *Nested Learning* (arXiv:2512.24695), the speaker-side neighbours named in `outputs/audience-refresh-2025-2026.md` §2.2(C) — both search-summary sourced, **not primary-verified**, so citing them would breach the citation discipline. **Owner: the Advisor** (whether to commission verification and fold them in).
> 3. **⚠ Sweep-instrument artifact, not a draft defect.** `scratch/v5-palm-reframe/subsweep.py`'s commit-hash pattern fires on the PLOS article number `e1010716`. Any future sweeper will see 3 hits where the source shows 2. **Owner: whoever next edits the sweep script.**
> 4. **Inherited debts restated so they are not re-discovered as new:** the two appendix tables that overflow at *any* permitted size (196.50 pt / 604.92 pt), and the unexecuted `I-J`/`I-R*` caption-sync worklist. Both pre-exist in the source. **Owner: unchanged (the standing worklists).**

## What I did

- Read `AGENT_PROTOCOL.md`; the Positioning Charter as it reads today (`outputs/philosophy-synthesis.md` §"Positioning Charter", C-1…C-10 — including the **C-1 reversal: no defensive audit paragraph**, which this draft honours: there is none); `claims_matrix.md`; `PJ_Writing_Style_Context.md`; the shorts charter Add.45–48; `outputs/audience-refresh-2025-2026.md` (Parts 2 and 3.2 in full); `outputs/v5-referee-v02.md` §D (the do-not-cut list); `outputs/v5-palm-reframe.md` (the refused-terms judgment and the adopted vocabulary); `outputs/plain-text-builds.md` + `papers/plain/v5/BUILD-NOTE.md` (what must carry over); and the source `.tex` end to end.
- Created **`.claude/NIPSsubmission/v5-palm/`**: `BUILD-NOTE.md` · `submission.tex` · `submission.pdf` · `figs/` (11) · style file + build artifacts. `pdflatex` was run **only** in this folder.
- Applied **19 exact-once string edits + 2 reference insertions** (`scratch/nips-v5-clean/patch1.py`, `patch2.py`, each replacement asserted to match exactly once, so a silent mis-edit is impossible), then two small wording tightenings after reading the compiled PDF.
- Verified everything in §"How I verified" and wrote the build note **first among the deliverables' content** (page split, audience changes, refused terms, numeric check, sweeps, carried-over verifications, figure inventory).

## How I verified (commands + observed output)

| check | instrument | observed |
|---|---|---|
| build | `pdflatex ×3` | `Output written on submission.pdf (19 pages, 1778065 bytes)`; `^!` errors **0**; "undefined" **0** |
| page split | `pdftotext -bbox` (block 72–720 pt) | main pp. 1–6 (p. 6 ends at `yMax=160.48` = 13.7 % ⇒ **5.14 pp of fill**), refs pp. 7–8, appendices pp. 9–19. Source: main pp. 1–5 (`yMax=653.88` = 89.8 % ⇒ 4.90 pp fill) |
| boxes | `submission.log` | overfull `\hbox` **196.49718 / 604.92413 / 49.44029 pt — identical values to the source log**; overfull `\vbox` 0; underfull `\hbox` 9 (=source); underfull `\vbox` 12 (source 4) |
| numeric, whole file | `scratch/plain-text-builds/numcheck.py` | distinct 570 → 575; **in source not in build: none**; in build not in source: citation years (`2022`×2, `2023`×2, `2025`×2, `2026`×5) + the two new bib records (`18`,`11`,`1010716.`×2,`10.1371`,`197`,`234`,`257.`) |
| numeric, main text | `scratch/nips-v5-clean/numcheck2.py` | **NEW-main − SOURCE-main = [] and SOURCE-main − NEW-main = []** (identical multisets both ways) |
| appendix fidelity | line diff after `\clearpage` | **2 diff lines, both insertions** (the two references); every other line byte-identical |
| do-not-cut list | `scratch/nips-v5-clean/dnc.py`, 27 exact-string probes vs source | **27/27, counts equal, FAILURES: 0** |
| compliance sweep | `scratch/v5-palm-reframe/subsweep.py` (38 zero-list patterns + 7 context classes) | positive controls **14/14 ⇒ LIVE**; zero-list **3 hits, all false positives** (2 = the source's own known instrument-name FPs; 1 = the PLOS-article-number regex artifact). Positive control on the task file: **9 hits** |
| refused terms | `grep` | `right-to-be-forgotten` 0 · `right to be forgotten` 0 · `memory provenance` 0 · `provenance` 0 (source: 0/0/0/0) |
| mandated-once | `grep` | `we make no biological claim` **1** · honest-scope sentence **1** |
| author token | `grep -o '\bMo\b' \| wc -l` | **1 = the bibliography entry**; prose/captions/labels/filenames 0; positive control on the pre-plain source **2**; `Morse` 0, `Moser` 0; pronouns 0 |
| typography carry-over | `grep -c` | `\scriptsize`/`\tiny`/`\footnotesize`/`multicols`/`\raggedbottom`/`\@startsection`/float-fraction/skip overrides/reduced widths = **0**; `\small` = **5**, the same five tables; `\textbf` = **0** |
| figures | `grep includegraphics` | **11**, same files, captions byte-identical, all three single-seed labels present verbatim |
| anonymization | decompressed-PDF string sweep + `pdfinfo` | `/Users` 0 · `Desktop` 0 · `.claude` 0 · `NIPSsubmission` 0 · `v5-palm` 0 · `chlu/` 0 · `ml4ps` 0 · **`PALM` 0** · `Pierini` 2 (continuity sentence + reference); Title/Author/Subject/Keywords/Creator/Producer all empty |
| source folders | sha manifests before/after | `plain` 30 · `palm-variant` 16 · `v5-short` 25 · `v2-short` 21 · `v2-neurreps-descoped` 10 · `neurreps-variants` 11 — **all six byte-identical** |
| length | de-macro word count | main text 3,400 → **3,673 words** (+273, all §1/§2 framing) |

## Findings / results — what changed, and on what evidence

**1. The white space is claimed as a gap in *mechanisms*, never as a system property.** §1 now names removal as the widest gap and §2 closes the deletion strand with the seam sentence (*"Our result sits in that seam, one level down, and it is a mechanism with measured laws rather than a property of a deployed system"*). **Evidence:** `outputs/audience-refresh-2025-2026.md` §2.3 finding 2 (deletion/right-to-be-forgotten essentially absent from the 70-paper proxy while the venue's own call names deletion in four of seven topics) and §2.2(B) (2025's "provable/exact unlearning" is invited-speaker-level in the adjacent unlearning community). ⭐ **Neither the census numbers nor the venue is named in the paper** — the framing is carried entirely by citations the paper already had (Rasmussen, Chhikara, Chakraborttii, Wang & Zhang, Yang, Uddin), so nothing unciteable or venue-identifying entered the artifact.

**2. The physics no longer apologises.** *"Physics enters only as the derivation apparatus"* is gone; the store is introduced as a physical system, the reduction is stated, and the three results are named as *a retention law, a diffusion coefficient and a vault factor* before being mapped to policy quantities. §3.2 now names the diffusion coefficient explicitly and states the vault mechanism-first. **Evidence:** `audience-refresh` §2.4 (physics vocabulary now attested in the proxy: *thermodynamic arbitration*, *entropic memory*) and §2.5 (theory and negative results are named contribution types). C-3's ML-first spine is intact: the paper still opens on the policy question and the contribution labels are operational.

**3. The three-sided reviewer pool is served without a single number moving.** Privacy side: the TTL laundering control still leads the leakage paragraph and still fires. Retention-dial side: Titans is now contrasted in one sentence rather than only named. Brain–LM side: the representational-drift bridge is in, with the no-biological-claim sentence attached exactly once, citing only primary-verified records. **Prose lifted from:** `outputs/audience-refresh-2025-2026.md` §2.2(C) and §3.2 (the vocabulary map) for the framing; `outputs/neurreps-audience-scout.md` §2.2 D2/D6/N1 for the drift records and — importantly — for the trap it documents (drift-along-an-attractor vs cross-session representational drift are two phenomena sharing one word), which is why the bridge is written as a *disclaimer of scope*, not a bid.

**4. The prose is simpler and the claims are not.** Eleven compound sentences split, the abstract rebuilt on ABT with three signposted results. The do-not-cut list survives verbatim at 27/27, `Blelloch` appears at all 10 sites, and both `fdt`+Newtonian fine-print blocks, the three emergent-arm caveats and the exact-deletion form with its three conditions are byte-identical to the source.

**5. ⛔ The page fact, stated honestly and not fixed.** 6 pages consumed / 5.14 pp of fill for the main text, on top of the source's already-honest 5.00 pp against a 4-pp limit. Growth is 100 % framing text (+273 words); no figure width, font size or skip was touched. **This base is for content and framing; compression is a later, deliberate pass.**

## Open editorial questions for the Hub / Head

1. **Title.** Unchanged from the source (*"Retention You Can Predict, Scope and Delete: a (μ,γ,T) Law and a Structural Deletion Guarantee for a Physics-Structured Memory Store"*). With the white-space framing now explicit, a deletion-forward title is available, e.g. *"Deletion You Can Check, Retention You Can Predict: a (μ,γ,T) Law and a Structural Deletion Guarantee for a Physics-Structured Memory Store"*. **Head's call**; I changed nothing.
2. **Does the Head want the venue's topic named at all?** I kept the source's refusal intact and named nothing venue-specific. A one-clause form (*"the update-and-deletion tests this literature asks for are system-level; ours is a store-level property"*) is already effectively in §3.3 without the compliance term.
3. **The speaker-side neighbours (ATLAS, Nested Learning).** They are the strongest 2026-current citations available for the retention-dial contrast and they are *not verified*. Commission a cite-check, or leave the contrast on Titans alone?
4. **Where the ~2 pp comes from, when compression starts.** My reading of the cheapest content-preserving order: (a) the lifecycle contribution (4) → appendix pointer; (b) §2's third strand compressed to two sentences; (c) the §3.3 composition clause → appendix (it is already there verbatim). Each is a claim-preserving move; none is mine to make.

## Risks

- **The drift bridge adds citation surface.** Two references were inserted without re-verification in this pass (reconciliation item 2). If either record is wrong, it is wrong in a sentence whose function is to *disclaim* scope — low blast radius, but it is a new surface the source did not have.
- **`we make no biological claim` is new text in V5.** It is required by the audience brief and is consistent with the program's standing treatment of drift, but it is the one sentence in this build that asserts something about a literature the paper does not otherwise engage.
- **Inherited, not introduced:** the two unfittable tables, the `I-J`/`I-R*` tokens, `Anonymous (2026)`, the third-person self-citation in a double-blind build, the stand-in style file, and the style file's printed line numbers.

## Git footprint

**None.** No tracked file created, modified or deleted. Everything written by this pass lives under `.claude/` (`NIPSsubmission/v5-palm/**`, `scratch/nips-v5-clean/**`, this report).

## Proposed handover updates (for the Hub)

1. **`.claude/NIPSsubmission/v5-palm/` exists and is the V5 iteration base**: 19 pp, main text 6 pp consumed / 5.14 pp fill, 0 errors, `BUILD-NOTE.md` complete (page split · audience changes · refused terms · numeric check · sweeps · carried-over verifications · figure inventory), all six source paper folders byte-identical.
2. **Record the standing fact in the ledger:** V5's main text at default formatting is **5.00 pp in the source and 6 pages here**, against a 4-pp limit — the gap is content, not typography, and the next V5 pass is therefore a **scope** pass.
3. **Two owed cite-check items** (reconciliation item 2): re-verify Aitken 2022 / Jude 2023; rule on ATLAS + Nested Learning.
4. **Sweep-script fix** (reconciliation item 3): exclude DOI/article-number contexts from the commit-hash pattern, or the next sweep report will carry a phantom hit.
5. **The refused-terms judgment held under the white-space framing** — `right-to-be-forgotten` and `memory provenance` remain at 0, and the white space is stated as a scarcity of mechanisms. If the Advisor wants that judgment revisited, this build is the clean place to A/B it: the change is one sentence in §1.
