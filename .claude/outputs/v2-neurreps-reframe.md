# v2-neurreps-reframe — paper-writer report

**DIAL DECLARATION (echoed): none — reframing/editorial pass; zero content, number or claim changes.**
Laundering control: n/a. Falsifies: n/a. Does NOT falsify: n/a.

Task + acceptance criterion: re-express the stylized V2 r9 build in the NeurReps audience's vocabulary as a
SEPARATE variant under `papers/neurreps-variants/v2/`, same numbers/claims, EA shape from the start.
**Status: done, with acceptance criteria 1 (page budget) MISSED and the arithmetic + costed menu on record.**

## ⚠ RECONCILIATION LIST (first-10-lines rule) — needs an owner
1. **Acceptance criterion 1 is MISSED and cannot be met without a rule violation.** Main = **5.69 pp** vs ≤4;
   total **13 pp** vs 8–9. Every reduction lever was *built and measured* (BUILD-NOTE §3.2); only one of eight
   is takeable without breaking C-2/C-3/C-6/C-10. **Owner: the Advisor → the Head.** Same shape as Add.35, but
   now with per-item measured costs rather than projections.
2. **The scout's reconciliation item 1 was honoured and item 2 is untouched by me.** §2 is written to the 2026
   CFP topic lines ("dynamics of neural representations", "symmetries, dynamical systems, and learning"); the
   v197 census supplies *neighbour* citations only. The EA-track-purpose tension (scout item 2) is a Head call.
3. **Scout rev.-2 retraction honoured.** The retracted "we identify the perturbation and cure it" framing does
   not appear. The rev.-2 replacement sentence is in §2 in its narrowed form, and Renart 2003 / Vafidis 2022 are
   cited *against* our own claims in both §2 and §4.
4. **A process incident is logged in BUILD-NOTE §8** (a stray `pdflatex` in the source directory, detected,
   fully reverted, byte-identity re-proved). The Hub may want the general guard recorded.

## What I did
- Created `papers/neurreps-variants/v2/`: `submission.tex`, `submission.pdf`, `BUILD-NOTE.md`, `figs/`,
  plus the copied template and the anonymized supplementary theory note. Build artifacts removed.
- Rebuilt the main text geometry-first: energy landscape → group orbit → transverse curvature spectrum →
  half-life; friction and temperature as motion along the orbit. Title is now *"The Price of a Flat Direction:
  Transverse Curvature Sets Retention in a Trained Recurrent Memory"*.
- Rebuilt §2 for this audience from the scout's Part 2 and Part 3 (see "prose lifted from" below).
- Applied Add.35's proportion package **by construction**: Figure 2 never promoted (appendix), §5 written to EA
  proportion from the start, retirements compressed to one sentence each in §2 with elaborations relocated to a
  new Appendix D (nothing dropped — all their numbers are in the file, and the two-way check proves it).
- Ran the two-way numeric check, the 71-pattern never-quote/apparatus sweep, the semantic-hermeticity pass and
  the anonymization audit including a decompressed-stream sweep; all printed in BUILD-NOTE §7–§8.

## How I verified (real numbers)
- Build: `pdflatex` ×3, TeX Live 2026 — **0 errors, 0 undefined refs, 0 overfull boxes**, 13 pp.
- Page split (PDF word bounding boxes vs the text block, the same instrument as the source build, run on both
  PDFs so the numbers are directly comparable):
  | | source r9 | variant |
  |---|---|---|
  | main | 5.72 pp | **5.69 pp** |
  | references | 1.01 pp (28) | **1.63 pp (45)** |
  | appendices | 4.26 pp | **5.68 pp** |
  | total | 11 pp | **13 pp** |
  Main text 3,872 words + one figure.
- **Costed menu, each row an actual build** (BUILD-NOTE §3.2): M1 Figure 1 → appendix **−0.39**; M4 §5 scope box
  **−0.28**; M5 §2 bridge para **−0.28**; M6 §4.2 price-of-prior **−0.25**; M2 fine print (a)–(c) **−0.12**;
  M3 FDT box **−0.12**; M7 honest gap **−0.12**; M8 retirements **−0.12**. **Only M1 violates no rule** (lands
  5.30 pp); 4.0 pp needs M1 + ≥3 rule-violating items.
- **Two-way numeric check** (330 source vs 390 variant distinct tokens): source-only = **1 token, `0.9`** = the
  `width=0.9\linewidth` figure specs (typographic). Variant-only = **61 tokens, all citation metadata** of the
  15 new references or regex fragments of existing numbers. **Zero content numbers lost, added or changed.**
- **Never-quote + internal-apparatus sweep: 71 patterns, ZERO hits.** Two context-checked compliant hits
  (`workshop` ×2: a source comment and the J&P venue string, both as in the source build). Positive controls
  fired: GMOR 7 · "introduced as CHLU" 1 · Rusch 5 · verification 6 · evidence 9 · Anonymous 3 · "transverse
  curvature" 4 · "continuous attractor" 1 · "fine-tuning problem" 2 · 45 reference entries.
- **Semantic hermeticity: 9 patterns, zero hits**, positive control "separate note" ×1 (the sanctioned
  Anonymous theory-note pointer).
- **Anonymization:** `\author{}` blank; PDF Author/Title/Subject/Keywords all empty; decompressed-stream sweep
  over 81 streams / 17.4 MB inflated returns 0 for `Forgis`, `x10719pj`, `Users/user`, `Desktop`, `CERN`,
  `Manchester`, `.claude`, `neurreps-variants`, `/tmp/`, with positive control `Goldstone` 8.
- **`\textbf` in main text = 0.** No-biological-claim sentence count = **exactly 1**.
- **`papers/v2-short/**` byte-identical**: 20-file `shasum` manifest taken before work and re-taken at the end;
  `diff` empty. (See the incident note below and BUILD-NOTE §8.)

## Findings / editorial substance

**Audience-term compliance (acceptance criterion 2).** Every audience term used is on the scout's Part-3 map or
defined in one clause:
| term | map | how it is handled |
|---|---|---|
| flat direction / zero-curvature direction tangent to the orbit | [E] | S\'agodi's definition quoted verbatim in §2 |
| continuous attractor | [E] | used **once**, for the literature's object only — never for our unit (F7) |
| transverse curvature $\mu^2$ | [A] | led with; "spectral mass" kept as the parenthetical; **the units conversion is printed once** in §1 |
| coset coordinate / coset register | [A]/[⛔N] | defined in one clause: "the coset coordinate, the position along the flat direction" |
| latch vs neural integrator | [A] | the real difference stated, not smoothed: *we store by damping, they store by not damping* |
| erosion → fine-tuning problem | [A] | scoped as "an instance of what the attractor literature calls the fine-tuning problem, the perturbation here being the training objective itself"; **"solves the fine-tuning problem" = 0** |
| the settle → fast normal / slow tangent flow | [A→E] | one clause in §3 |
| exceptional point | [⛔N] | defined on first use as the overdamped→underdamped crossover where the block becomes defective |
| conformally symplectic | [A] | one clause; ⚠ residual F8 collision survives only in Xu et al.'s reference title |
| Wigner–Weyl / NG / pseudo-NG | [A] | glossed once each: single fixed point / exactly-flat direction / slow manifold |
| budget cube | [⛔N] | **not used** — written out as "the map from $(\mu,\gamma,T)$ to lifetime" |
| "drift" | trap §2.2 | 6 occurrences, every one scoped in-sentence or a kick–drift–kick / table header / prohibition |

**Where the reframe was forbidden from widening a claim, and did not.** §4.2 carries the Vafidis rider
("a measurement on our architecture class and training recipe, not a general statement that learning cannot
produce a tuned flat direction"); §4.3 carries the Renart rider ("that a corrective term can keep a flat
direction alive is not new … what we add is that every headline law still holds under the correction").
Both are the scout's B7/B8 obligations, and both *shrink* rather than widen the surrounding claims.

**Related-work prose lifted from the scout report** (per my brief's craft rule): §2's continuous-attractor
paragraph is built from `neurreps-audience-scout.md` §2.1 (B1–B6) and §2.1b (B7–B9), its closing positioning
sentence from §2.3's rev.-2 narrowed form, its census neighbours from §1.1/§1.3 (G3, G6, G9, T4, N3), its drift
disclaimer from §2.2 (D2, D6, N1), and every vocabulary decision from Part 3.

**Headline figure:** Figure 1 (the head-to-head, `fig1_mo_headtohead.png`) — unchanged in content, resized
0.9 → 0.68 linewidth.

**Nothing was dropped.** Two blocks moved main → appendix (retirement elaborations; the Mo-own-estimator
reproduction), both fully written in the new Appendix D with their numbers intact; Figure 2 moved main →
Appendix A.1 with its caption verbatim.

## Git footprint
None. No tracked code touched; all output under `.claude/`.

## Open questions / follow-ups / risks
1. **For the Head (numbered):** (i) take M1 (Figure 1 → appendix, main 5.30 pp) or hold the figure? (ii) accept
   main 5.69 pp on the Add.36 precedent (ship rather than trade measured evidence for pages), or authorise a
   rule-violating item from the menu? (iii) does the reframe's §2 justify the +0.62 pp of references, or should
   6 of the 15 new citations be dropped to pull the total toward 9 pp?
2. **Template risk unchanged from the source build:** no NeurReps/NeurIPS-2026 style file on this machine, so
   the count must be re-measured in the real template. The venue template is typically denser, which is the
   only unbudgeted lever left.
3. **Scout gaps that touch this draft:** Renart 2003 and Vafidis 2022 abstracts are still index-sourced and are
   now load-bearing in *both* §2 and §4. Recommend the primary-abstract read the scout listed as priority (3)
   before this variant is sent anywhere.
4. **Scout F8 residual:** "conformal" means two things in this room; the collision is confined to a reference
   title and cannot be edited away.
5. **Process risk, generalisable:** a `cd` into a source directory persisted across Bash calls and a build ran
   there. Caught by the pre-taken checksum manifest and reverted byte-for-byte. **Recommend the manifest-before,
   manifest-after discipline become standard for any task with a "byte-untouched" acceptance criterion** —
   without it the incident would have been silent.

## Proposed handover updates (for the Hub)
- `papers/neurreps-variants/v2/` exists and builds: main **5.69 pp** / refs **1.63** / appendices **5.68** /
  total **13 pp**; source build byte-verified untouched.
- Add.35's proportion package, applied by construction, is now **measured** rather than projected: it bought
  back the entire cost of the reframe's new §2 (main 5.69 vs r9's 5.72 while carrying a full bridge-literature
  section), and the residual gap to 4 pp is **protected content**, itemised and individually costed.
- The claims matrix and never-quote registry need **no new rows** from this pass: zero claims were added,
  changed or widened, and the two-way numeric check is clean in both directions.
