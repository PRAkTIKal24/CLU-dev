# v5-referee-final — paper-referee report

**Task + acceptance criterion:** final hostile-but-fair referee pass on `~/Desktop/V5_PALM_Submission/paper.pdf` exactly as PALM receives it; deliver MUST/SHOULD/NICE with costs, one unhedged verdict, a simulated score, the three most hostile quotes, a one-sentence contribution statement, and the main-text-standalone verdict.
**Status:** done.

**DIAL DECLARATION — echoed:** Dials touched: **NONE.** Read-only review. No experiment, no config, no registry, no charter, no paper file. Writes one report.

**`paper.tex` md5 — start of pass:** `9e94731cf3f43151d98dc213af834ccf`
**`paper.tex` md5 — end of pass:** `9e94731cf3f43151d98dc213af834ccf` *(identical; nothing in the submission folder was modified; mtimes unchanged: paper.tex 13:13, paper.pdf 13:14, refs.bib 02:28)*

---

## 0. Independence-bar statement and full disclosure

**The bar was honoured. No file on the barred list was opened.** I did not read `v5-referee-v02.md`, `pj-referee-v5-r2.md`, `pj-fidelity-v5-r2.md`, `pj-minimal-v5.md`, `v5-cite-pass.md`, `v5-derivation-appendix.md`, `v5-final-pass.md`, `v5-palm-reframe.md`, any `BUILD-NOTE*.md`, or `advisor-head-shorts-charter.md`. I did not open `.claude/outputs/` at all except to `ls` the directory listing (filenames only, no contents).

**Disclosure of one deliberate departure from the task's read-whitelist, made with the reason stated:** my spoke system prompt standingly directs me to read `.claude/critique_register.md`, `.claude/claims_matrix.md`, and the Positioning Charter in `.claude/outputs/philosophy-synthesis.md`. I resolved the conflict as follows:

- I read **`critique_register.md` only** — dated 2026-07-20, a standing attack catalog that predates this draft by five weeks and contains **no assessment of V5**. It is an instrument for being *more* hostile, not a prior opinion on this paper.
- I read it **only after** the blind main-text read was complete and written to disk. My independent findings are timestamped in `.claude/scratch/v5-referee-final/blind-first-read.md`, written before the register or any appendix was opened. Suspicions S1–S14 in that file are all mine, unaided; the register added exactly two items (the missing minus-the-physics control, and the "verification/evidence" taxonomy being a *program* convention rather than a public one).
- I did **not** read `claims_matrix.md` (398 KB; may contain V5-adjacent approved wordings) and did **not** read `philosophy-synthesis.md`.

**No leak occurred.** Every number in this report was checked against the PDF, `paper.tex`, `paper.bbl` and `refs.bib` — never against an internal report.

**Instrument-validation note (the grep/strings hazard, and it bit me).** My first anonymity sweep with `strings paper.pdf | grep -i jawahar|pierini|...` returned **zero hits**. The positive control (`grep -c "Anonymous"`) also returned **0** — i.e. the instrument was blind, because the PDF's text streams are Flate-compressed. I re-ran after `mutool clean -d`; the positive control then returned 1 and the sweep returned real content. **The clean anonymity result reported in §6 below is from the validated instrument, not the blind one.**

---

## 1. §4.2 — the paper's one contribution, in my own words, written after the main-text-alone read

> *In a small hand-designed physics-structured memory store, the retention half-life of a written value is a closed-form function of three physical dials — the stored direction's spectral mass µ, the damping γ, and the temperature T — with an analytically predictable optimum; retention can be scoped in space by a local friction hole; and if item placement is made canonical, deleting an item leaves the store byte-identical to one that never held it.*

I was able to state it, and it is a real contribution. The paper's framing question — *what actually is a retention policy, physically, and what exactly survives a delete* — is a good question and PALM is the right room to ask it in. **The paper earns its place by being about the right question.** It does not earn it by its evidence, and it does not pretend to. That is the honest summary and it is the one I would write as a reviewer.

---

## 2. §4.1 — MAIN-TEXT-STANDALONE VERDICT: **NO**

**Does every claim the main text makes carry its qualification in the main text? No — twelve times.** This is the highest-value section of this pass and it is where the paper is weakest. A PALM short-track reviewer is not required to open an appendix, and this paper's four pages currently assert more than the four pages support.

| # | Claim stated in MAIN TEXT | Qualification lives ONLY in | Severity |
|---|---|---|---|
| 1 | Abstract + §2.1 L84: "a V-curve **minimized at γ_crit = 2εµ**" | App. F.3 — γ\* = 2εµ·(1+O(εµ)); exact root is h\*(γ)=(1−√(1−γ))√(2/(2−γ)) | **MF-4** |
| 2 | §2.1 L86/L93: log-slopes "+1.23 to +1.27" and "+1.116±0.011" printed against Fig. 1's drawn "∓1 asymptotes" | App. F.3 — d ln n½/d ln γ = 1 + γ/(2−γ), "exceeds it at every finite γ" | **MF-4** |
| 3 | Abstract: "spans approximately **11 orders of magnitude** in µ²" | §2.1's own hedge is present, but Fig. 1's real span and the probe-floor tick are in App.-Fig. 6 | **MF-5** |
| 4 | §2.2 L119: "vault factor of (γ_eff/γ)², yielding 107.77±4.78×" | App. F.7 — the factor exists *because* damping and noise are deliberately mismatched (absorb-only); the FDT-consistent counterfactual is **13.88×** | **SF-2** |
| 5 | §2.1 L91: "This fundamental law **transfers directly to emergent units**" | App. C Fig. 8 — *"The emergent unit has no continuous coset register… nothing is retained"* (≈1–1.6 bits) | **SF-3** |
| 6 | §2.2 / §3 L167: the vault contrast arm, "where bounds remain strictly contained" | App. D — 23.39±10.06 vs pre-registered [6.5, 9.5], **"triggering the designated falsifier constraint"** | **MF-1** |
| 7 | §2.3 L145: "Against an **exact adversary**…" | App. E Table 6 — "exact" = zero observation noise; and the **white-box row scores 1.000** | **MF-3** |
| 8 | §2.3 L133: "with the waitlist engaged, store-level deletion remains exact" | App. E — with the waitlist **off** at overflow, byte-equal fraction → **0.0000** and AUC(n_live) → **1.00000** | SF |
| 9 | §2.3 / §3: deletion cost | App. E, final sentence — the shipped store does a **full O(n) rebuild per operation** | **SF-8** |
| 10 | Fig. 1 and Fig. 2 captions label arms "**verification**" / "**evidence**" | Nowhere. The taxonomy is never defined in the paper. | **SF-6** |
| 11 | §2.3 L134: "loads (0.29× to 1.71× of standard lattice capacity)", cited to App. E | **Not in App. E.** App. E reports n=4, n∈{8,16,40,64}, K=64, ×1.05 — neither 0.29 nor 1.71 appears | **MF-6** |
| 12 | Abstract: "a designed, non-learned **3-dimensional** datastore" | **Nowhere.** "3-dimensional" occurs exactly once in 18 pages — in the abstract | **MF-6** |

This is precisely the failure mode the task named: *a number moved and its scope clause did not follow.* Items 1, 2, 6 and 12 could not have been caught by any diff.

---

## 3. MUST-FIX — each with the cost of not fixing

### MF-1. A pre-registered falsifier fired, and the Limitations bullet describes it as contained. *(integrity — the most dangerous item in the paper)*
**Location:** App. D "Emergent Arm Translation" (paper.tex L285) vs §3 Limitations L167; §2.2 silent.
**Evidence:** App. D: *"the measured field-to-scalar control variance computes to 23.39±10.06 against a pre-registered bounds parameter of [6.5,9.5], **triggering the designated falsifier constraint**."* §3: *"remeasure the memory vault contrast arm **where bounds remain strictly contained**."* §2.2 reports 107.77× vs the 13.28× control with no indication that the corresponding emergent contrast failed its own pre-registered band by 2.5×.
**Attack:** a reviewer who reads App. D and then re-reads Limitations concludes the paper *knew* a pre-registered prediction failed and wrote a Limitations bullet whose plain reading says the opposite.
**Cost of not fixing:** this is the fastest available route from "borderline" to "reject", and it is the only finding in this report that a rebuttal cannot repair — you cannot un-write a sentence a reviewer has already read as concealment. Everything else in this paper is a clarity problem; this one is a credibility problem. The paper's whole differentiator at this venue is that it is more honest than its neighbours; this sentence spends that.
**Minimal compliant alternative:** state the fired falsifier in §2.2 in one clause, and make the Limitations bullet say the band was exceeded and why (the control arm delocalises).

### MF-2. "A system may only optimize two" — an impossibility theorem with no proof, no citation, and no appendix, anywhere in 18 pages.
**Location:** §2.3 L141–143.
**Evidence:** *"Memory lifetimes are governed by a strict trilemma: exact value fidelity, amplitude-independent address hold, and amplitude-independent read latency. A system may only optimize two."* App. F contains eight derivations; none is the trilemma. App. E's "Trilemma Parameters and Cost Evaluation" reports measurements on this one store and refutes **two attempted repairs** (and says "both … variants" while describing one).
**Attack:** a universal impossibility claim supported by two failed patches on the authors' own testbed. This is the single clearest instance of the designed-testbed result reading as general — and the paper's own "Claim Classification" bullet promises *"a fundamental mechanism bounded by measured physical laws"*, which a bare impossibility assertion is not.
**Cost of not fixing:** a theory-comfortable reviewer will quote this sentence as the paper's characteristic overreach and use it to discount the closed forms that *are* properly derived. It converts a strength (App. F is genuinely good) into a liability.
**Minimal compliant alternative:** demote to what was measured — "on this store we could not obtain all three; both repairs we tried failed, at [numbers]" — or add the proof.

### MF-3. The leakage sentence is inverted relative to its own table, the threat model is undefined, and the two adversary rows that matter are absent from the main text.
**Location:** §2.3 L145–148 vs App. E Table 6.
**Evidence:** Main text: *"the Boolean TTL flag remains within a 0.017 AUC margin of pure physical decay (0.983 versus 1.000)."* The natural parse of "*X* within a margin of *Y* (*a* versus *b*)" assigns TTL = 0.983, physical decay = 1.000. **Table 6 says the reverse**: Physical Decay = 0.983, TTL Flag = 1.000. Further: the main text reports only the middle row. It omits **"White-box architecture, fully exact: Physical Decay 1.000, TTL 1.000"** — perfect membership inference against the physical store — and it omits **"σ_obs = 0.1: Physical Decay 0.559, TTL 0.996"**, which is the paper's *best* privacy result and its strongest sentence. "Exact adversary" is never defined in the main text.
**Attack:** this is the privacy reviewer's section and it is the one they will read twice. Under the sentence's natural parse the paper states that its own store scores AUC 1.000 against a membership adversary — inside a subsection titled "**Absolute Guarantees**". Under the table's reading the paper has buried its best number in an appendix and put its most ambiguous one on page 4.
**Cost of not fixing:** the membership-inference reviewer's stated reason to reject, and the paper cannot answer it from what is printed because the sentence is genuinely ambiguous — the rebuttal would have to say "you misread us", which loses. It also forfeits, for free, the 0.559-vs-0.996 result that would have won that reviewer over.
**Minimal compliant alternative:** name the two entities with their numbers unambiguously, define "exact adversary" in one clause, and put the σ_obs=0.1 row in the main text.

### MF-4. §2.1's two central closed-form claims are visibly contradicted by the numbers printed beside them; both reconciliations exist only in App. F. *(one sentence fixes both)*
**Location:** Abstract L14; §2.1 L84, L86, L93; Fig. 1 caption.
**Evidence (a):** the abstract and §2.1 assert the V-curve is "minimized at γ_crit = 2εµ". Nine lines later §2.1 prints the measured argmin as **0.902 ± 0.003 × γ_crit** — 10% below the asserted optimum, excluding 1.0 by ~33 standard errors. App. B independently gives designed argmins 0.076–0.096 against predicted 0.082–0.116 (ratios ≈0.83–0.93). App. F.3 shows γ\* = 2εµ(1+O(εµ)) and gives the exact implicit root h\*(γ) — **but never evaluates it and never compares it to 0.90**. (I evaluated it: at µ²=0.670 the exact crossing is γ\*≈0.0786 vs 2εµ=0.08185, ratio ≈0.96 — so the O(εµ) correction accounts for roughly *half* the observed offset and the rest is unexplained anywhere in the document.)
**Evidence (b):** §2.1 prints log-slopes **+1.23 to +1.27** (designed) and **+1.116** (emergent) while Fig. 1's caption calls the drawn guides "the **∓1** asymptotes", and Fig. 1 visibly shows the right branch departing from its dotted +1 guide. App. F.3 derives exactly why — d ln n½/d ln γ = 1 + γ/(2−γ), which "exceeds [+1] at every finite γ" — and that sentence never reaches the main text.
**Attack:** a reviewer of the four pages alone sees a law asserted and then a number that falsifies it, twice, on the same page, with no comment. The headline figure appears to show data departing from the law it is captioned as confirming.
**Cost of not fixing:** this is the most quotable defect in the paper and it is entirely self-inflicted — the paper *has* the answer, in App. F.3, in one clause. Not stating it converts a fully-derived result into what looks like a suppressed disagreement, and invites the reviewer to check every other number.
**Minimal compliant alternative:** one sentence in §2.1 — that 2εµ is the leading-order root and the finite-εµ/finite-γ corrections of App. F.3 predict both the sub-unit argmin and slopes above +1.

### MF-5. The µ² interval printed in the main text excludes most of the curves in the headline figure, and its upper endpoint is sourced nowhere.
**Location:** §2.1 L94–95 vs Fig. 1 caption.
**Evidence:** §2.1: *"Spanning both architecture families, this curve holds consistently across µ² ∈ [1.7×10⁻¹², 7×10⁻²]."* Fig. 1's caption states the five designed radial modes are **µ²_rad = 0.670–1.348** — five of the eight curves, an order of magnitude *above* the stated upper endpoint. Separately, **7×10⁻² occurs exactly once in the document** and matches no measured value: Fig. 1's emergent range is 2.0–5.4×10⁻²; App. D's is 2.77–8.93×10⁻².
**Attack:** the sentence is the sole support for the abstract's "approximately 11 orders of magnitude", the paper's most quotable claim, and it disagrees with its own figure.
**Cost of not fixing:** a reviewer who spot-checks the headline against Fig. 1 finds them inconsistent. That reviewer then does not trust 107.77, 0.983, or 1.0068 either — which is unfair to a numeric spine that is otherwise sound, and is exactly the damage a two-minute edit prevents.

### MF-6. Two main-text/abstract facts have no backing anywhere in 18 pages, one of them cited to an appendix that does not contain it.
**Location:** Abstract L22 ("a designed, non-learned **3-dimensional** datastore at capacities 8–64"); §2.3 L134 ("all evaluated loads (**0.29× to 1.71×** of standard lattice capacity)", cited "see App. E").
**Evidence:** "3-dimensional" occurs once in the whole document — in the abstract. The word "lattice" occurs three times and never with a dimension. App. E reports n=4, n∈{8,16,40,64}, K=64, a ×1.05 inflation, 61/64 and 43/64; **neither 0.29 nor 1.71 appears there or anywhere else**. Compounding it, the abstract's "capacities 8–64" and §2.3's "0.29×–1.71× of capacity" are two different scale descriptors for the same experiment, reconciled nowhere — while §3 says the work is "constrained to a dimension of 4".
**Attack:** a reviewer who follows the pointer to App. E to check the deletion sweep cannot find the number they were sent to find; and one who reads "3-dimensional" in the abstract and "dimension of 4" in the Limitations concludes the paper is describing two different systems.
**Cost of not fixing:** an unbacked number in the abstract is the cheapest possible credibility loss and the one a referee is most likely to mention in writing. It is also the only class of defect here that looks like fabrication rather than compression, which is a much worse impression than it deserves.

---

## 4. SHOULD-FIX

- **SF-1 — the abstract quotes 107.77× naked; the matched control is main-text-only.** §2.2 gives the uniform-scalar-friction baseline as 13.28±0.12×, so the honest contrast is **8.11×** (and Fig. 2b says exactly this: "8×"). The abstract's "a 107.77±4.78× retention factor" reads to a systems reviewer as a 100× system win. *(Abstract L18; §2.2 L120.)*
- **SF-2 — the main text never says the vault works by locally breaking FDT.** App. F.7 is explicit and honest: absorb-only means the noise keeps the *scalar*-γ scale while damping runs at γ_eff, and the FDT-consistent coupled bath yields **13.88×** — essentially the scalar control. The 8.11× advantage *is* the 7.942× refrigerator factor, "identically". Meanwhile the main text's Scope bullet declares that T>0 *"strictly requires the fluctuation–dissipation-consistent noise scale σ\*"*. A dynamics-comfortable reviewer will spot that the headline vault departs from the consistency the same page mandates, and the paper never owns it. It is defensible physics (a cold finger is real) — but it must be said in §2.2, not only in App. F.7.
- **SF-3 — App. C states the emergent units cannot hold a value; the main text says the law "transfers directly".** Fig. 8 caption: *"The emergent unit has no continuous coset register… on every emergent seed the value relaxes to the washboard minimum and nothing is retained."* The emergent arm is the paper's only claim to generality beyond a hand-built testbed, and its own appendix says those units store ≈1–1.6 bits. What transfers is the *eigenvalue* V-curve, not the ability to store. Say so in §2.1.
- **SF-4 — dangling forward reference.** Fig. 8's caption ends *"This is the figure behind the capacity statement of ≈1–1.6 bits."* **There is no capacity statement in the paper.** The phrase "bits" occurs nowhere else.
- **SF-5 — Fig. 2(c) plots the number the paper disavows and not the number it quotes.** The panel's bars are the raw first-passage ratios (84×/86×/91× per seed) which the caption explicitly says are *not* the vault; 107.77× appears only as panel title text, plotted nowhere. Additionally the T=0 control — a load-bearing null — is delivered in a ~4 pt inset that will not be readable at printed size.
- **SF-6 — "verification" / "evidence" are used in main-text figure captions and defined nowhere.** This is a program-internal taxonomy (designed-testbed = verification, learned-system = evidence). Fig. 1 and Fig. 2 both carry it on page 3–4. A reviewer cannot know what it means and may read "evidence" as a hedge on the emergent arm — which, given SF-3, is worse than saying nothing.
- **SF-7 — App. E's prose is close to unparseable, and App. E is the appendix the privacy reviewer will read.** Representative: *"the resulting structural spacing enforces an exact boundary of 1.540000"*; *"the functional byte-equal architectural fraction plummets directly to 0.0000"*; *"We rigorously refute both standard previously proposed baseline repair variants"* followed by a description of one; *"generating hard death physical amplitudes of exactly 0.90/0.80/0.70/0.30"*. The underlying results are strong (24 exhaustive orders at n=4, 200 randomised orders at n∈{8,16,40,64}, 100 interleavings, byte-equality across 64 randomised queries). They are not currently legible.
- **SF-8 — App. E claims a "negative price" and admits an O(n) rebuild per operation in the same paragraph.** The "negative packing price" (61/64 → 64/64 canonical vs 43/64 stochastic) sits a few lines from *"the core canonical synchronization mandates a complete and structural O(n) rebuild evaluated entirely per operation."* The systems reviewer's first question at PALM is "what does this cost a deployed memory?", and the answer — O(n) per delete — appears once, in the last sentence of an appendix, and never in the main text or Limitations (which lists costing it as *future work*).
- **SF-9 — §2.3's heading is a bare superlative over the paper's weakest result.** "Structural Deletion: **Absolute Guarantees**, Lifetime Trade-offs, and Residual Leakage" sits above a table whose strongest adversary scores AUC 1.000. Headings are what skimmers read.
- **SF-10 — statistical vocabulary is misused in the main text, twice, in load-bearing sentences.** L101: *"a variance of 0.35%"* — it is a relative difference between two argmins. L111: *"verified to a **tolerance** of 1.0068 ± 0.0219"* — it is a ratio of measured to predicted. (App. D adds *"the measured field-to-scalar control **variance** computes to 23.39±10.06"* — also a ratio, and App. B *"records exponential factors"* for numbers that are not exponential.) A technical reviewer reads systematic misuse of "variance" and "tolerance" as either carelessness or machine-written prose, and discounts the numbers.
- **SF-11 — two verbs the evidence does not support.** L106: a coset *"latches **indefinitely**"* — inferred from 200k steps, with a drift bound (4.9×10⁻¹² rad) at the float64 floor, i.e. plausibly an instrument limit rather than a measurement. L123: the outside-boundary state fraction *"drops **definitively** to 0.0000"* — a measured zero to four decimals with no N stated in the main text (App. D: 1024 walkers).
- **SF-12 — there is no minus-the-physics baseline anywhere in the paper, and the V-curve's actual differentiator is never stated.** Every control here is internal (scalar friction, coupled bath, γ=0, T=0, TTL flag). The obvious reviewer question — *"would an ordinary first-order exponentially-decaying store show the same thing?"* — has a good answer (it would not: the V-curve requires second-order/momentum dynamics; first-order decay is monotone in the decay rate) and the paper never gives it. This is the cheapest available generality argument and it is left on the table.
- **SF-13 — the nearest neighbour in this room is engaged only in an appendix.** App. A does it well — *"While the learned gate in Titans is structurally similar to the retention dial discussed in Sec. 2.1, none of these prior works establish physical retention laws, state explicit half-lives, define read tolerances, or mathematically price the information loss."* That sentence is the paper's novelty defence against the single most likely "this already exists" objection, and it is on page 8. A PALM reviewer forming a score from four pages will not see it.
- **SF-14 — an orphan reference, hand-typeset as a bullet after `\bibliography`.** The Jude et al. (2023) entry (paper.tex L130) is cited nowhere in the document; 28 keys are used and 28 `\bibitem`s are produced, so this is a 29th entry outside BibTeX, rendered as a stray bullet in the reference list. It supports nothing.
- **SF-15 — the page overrun (see §7).**

---

## 5. NICE

- **N-1.** ε is overloaded: the integration step (fixed at 0.05, Scope bullet) and the unlearning parameter in "(ε,δ)" (§2.3 L140, App. A). A privacy reviewer reads (ε,δ) natively and will trip.
- **N-2.** The architecture is renamed without explanation — "the Causal Learning Unit (CLU), introduced as CHLU by [self-cite]" — while the cited title is "The Causal **Hamiltonian** Learning Unit". Dropping "Hamiltonian" from a paper whose entire thesis is Hamiltonian dynamics reads oddly.
- **N-3.** "Consequently, n½ ∝ γ^{+1}" (§2.2 L112) is exact only for γ≪2; App. F.6 correctly prints n½ ∝ γ/[(2−γ)T], a ~25% correction across the swept range γ∈[0.002, 0.5]. App. B's measured +0.9552±0.0422 is consistent with the exact form, not the stated one.
- **N-4.** App. B L147: *"The legacy reference default structure fundamentally violates these thermodynamic laws."* This is an erratum against the self-cited foundation, delivered in a subordinate clause. It is admirable and it is also the kind of sentence a reviewer stops on.
- **N-5.** Three different drift bounds for similar-sounding latch claims — 4.9×10⁻¹² (main text, no-hole latch across γ), 1.75×10⁻¹² (App. D, hole in/out), 1.8×10⁻¹² (Fig. 2c inset). These are genuinely different experiments; a reader cannot tell that from the text.

---

## 6. Non-issues — pre-empted so the Head does not spend hours on them

These all *look* like defects and are not. I checked each; do not touch them.

1. **The page footer reading "Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026)" rather than naming PALM is template-forced and cannot be fixed while staying anonymous.** `neurips_2026.sty` emits `\@trackname` (which does say "Workshop: PALM") only under `\if@neuripsfinal` (sty L396–402); and `[final]` sets `\@anonymousfalse` (sty L61). `[dblblindworkshop]` is the correct and only anonymous option. **Leave it.**
2. **The Jawahar & Pierini [2026] self-citation is the sanctioned third-person double-blind mechanism, not a violation.** It is written in the third person ("introduced as CHLU by Jawahar and Pierini [2026]"), with no "our previous work" construction, and it points to a public arXiv record. Per the task's own §5.7 this is correct practice and I am flagging it as *compliant*, not as a problem.
3. **Anonymity and metadata are clean — verified with a positive-controlled instrument.** After `mutool clean -d` (positive control passing), a sweep for author names, institutions, filesystem paths and acknowledgements returns only the sanctioned reference-list entry. `pdfinfo`: Title/Author/Subject/Keywords/Creator/Producer all empty; `\hypersetup` scrub present at paper.tex L20; `\ack` is suppressed by the submission option; no `/Users/` paths; figure filenames are generic.
4. **The build is clean.** 0 undefined citations, 0 undefined references, 0 overfull boxes, 28 cite keys / 28 bibitems (no uncited BibTeX entries — SF-14 is outside BibTeX). Four underfull hboxes in table cells, cosmetically invisible.
5. **The Zotero cite keys `goos_lower_2003` and `hutchison_uniquely_2008` are not mis-attributions.** Those keys carry LNCS *series editors*, but the entries' author fields are correct and the PDF renders "[Buchbinder and Petrank, 2003]" and "Blelloch et al. [2008]". Correct in the output; ugly only in the source.

---

## 7. §6 — the page limit, reported and priced, not solved

**True split, measured from the PDF:** main text runs to page 5 and ends at line 169; `References` begins at line 170, approximately **37% down page 5**. So **main text = 4.37 pp against a 4-pp limit**; total document 18 pp. The Head's 4.35 figure is accurate.

**Chair's likely mechanical response.** At a NeurIPS workshop a 0.37-page overrun is usually survivable: most PALM-class chairs either eyeball it or defer it to camera-ready. The realistic risk is not a human chair — it is an **automated OpenReview page-count check that counts pages-before-references**, which here returns **5**, and fails a mechanical `≤4` test without a human ever seeing the paper.

**My one statement, per the task's instruction to say it once and move on.** I think the Head's acceptance of this overrun is *slightly* wrong, and the reason is asymmetry rather than probability: the chance of enforcement is low (I would guess 10–20%), but enforcement is a total, un-rebuttable loss of the cycle, whereas recovering 0.37 pp is an hour's work. That is a bad trade even at 10%. **That said — I am not recommending content cuts, the overrun is not a MUST, and it does not change my verdict.** Moving on.

---

## 8. VERDICT (unhedged)

# `SUBMIT AFTER THE MUSTS`

**Why this and not `SUBMIT AS IS`:** MF-1 and MF-3 are not clarity defects. MF-1 puts a sentence in the Limitations whose plain reading contradicts a fired pre-registered falsifier disclosed in the appendix; MF-3 states the paper's privacy result in a form whose natural parse is the opposite of its own table, inside a subsection titled "Absolute Guarantees". Each independently converts a sympathetic reviewer into a hostile one, and neither can be repaired in rebuttal. **Why this and not `DO NOT SUBMIT THIS CYCLE`:** every MUST is a text-only edit. There is no missing experiment among them, no new figure, no re-run. MF-1 through MF-6 are on the order of eight sentences, all of which *reduce* claim strength or *import* material the paper already owns from its own appendices. The underlying work — App. F in particular — is sound and above workshop norm.

---

## 9. Simulated PALM short-track review

**Recommendation: 5 — Borderline accept** *(would become a confident 6 with the MUSTs; would fall to 3 if a reviewer hits MF-1 or MF-3 first)*. **Confidence: 4.**

> **Summary.** The paper models an agent memory store as a damped symplectic integrator of a learned Hamiltonian and derives retention half-life as a closed-form function of (µ, γ, T), with a predicted optimum at γ=2εµ, a spatially localised "memory vault" via a friction hole, and an exact store-level deletion guarantee obtained by making item placement canonical à la Blelloch & Golovin.
>
> **Strengths.** The question is the right one for this workshop and is genuinely under-served: nobody in the agent-memory literature can tell you the half-life of a stored value or what is left after a DELETE, and this paper can, in closed form, for its store. The derivation appendix is unusually good — every closed form the paper prints is derived from one 2×2 step matrix, in four pages, and the authors explicitly disclaim novelty for the parts that are not theirs. The honesty apparatus is well above workshop norm: pre-registered predictions, laundering controls, a γ=0 null, a T=0 null, per-figure seed counts and flag provenance, a cross-instrument re-measurement, and an explicit refusal to claim certified unlearning. The scope bullets on page 1 are exemplary.
>
> **Weaknesses.** No task, no baseline system, no comparison to any deployed memory. The scale is dim 4 / hidden 64 / 5+3 seeds on a laptop CPU — the authors say so plainly, which I credit, but it does bound what the workshop can take from this. The "emergent" arm is the paper's only bid for generality and Appendix C states that those units have no coset register at all and retain ≈1–1.6 bits — so what transfers is the eigenvalue curve, not the memory. Several of the main text's central assertions are contradicted by numbers printed beside them (the optimum is asserted at γ_crit and measured at 0.902 γ_crit; the ±1 asymptotes are drawn in Figure 1 and measured at +1.23–1.27); the reconciliations are correct and are in Appendix F, but a four-page reviewer will not see them. The headline vault number is quoted without its matched control in the abstract, and the 8× advantage over uniform friction is entirely the local FDT mismatch, which the main text does not say. The trilemma is asserted as an impossibility and never proved. Finally the leakage sentence is stated in a form that inverts its own table.
>
> **Verdict.** The mechanism is real and the analysis is careful; the writing currently overstates it in the four pages that count. I would accept a version whose main text says what its appendices say.

---

## 10. The three most hostile quotes a reviewer could write — verbatim — and whether the paper can answer them from what is printed

> **1. "Section 3 tells me the vault contrast arm has 'bounds [that] remain strictly contained'; Appendix D tells me the same measurement came in at 23.39 ± 10.06 against a pre-registered band of [6.5, 9.5] and 'triggered the designated falsifier constraint'. I would like the authors to explain which of these two sentences they intend me to believe."**
>
> **Can the paper answer it from what is printed? NO.** App. D gives a real mechanistic reason (the control arm delocalises while the field holds the absorb limit), but nothing in the main text or Limitations connects the two sentences, so the answer exists only in the rebuttal — by which time the reviewer has already formed the impression. **This is the one quote that ends the paper. Fix MF-1.**

> **2. "The abstract states the V-curve is 'minimized at γ_crit = 2εµ'. Nine lines into Section 2.1 the authors report the measured argmin as 0.902 ± 0.003 × γ_crit — ten percent away, at roughly thirty standard errors — and offer no comment. Figure 1 likewise draws '∓1 asymptotes' beside measured slopes of +1.23 to +1.27. A paper whose central contribution is a closed-form law should not print its own falsification twice on one page without remark."**
>
> **Can the paper answer it? YES — but only from an appendix the reviewer need not open.** App. F.3 has both answers, correctly: γ\* = 2εµ(1+O(εµ)) with the exact root, and d ln n½/d ln γ = 1 + γ/(2−γ) which "exceeds [+1] at every finite γ". One sentence in §2.1 disarms this entirely. *(Caveat the Head should know: the O(εµ) correction accounts for roughly half the argmin offset by my own evaluation of App. F.3's exact root; the remainder is unexplained anywhere in the document, so the added sentence should not claim the correction closes the gap fully.)*

> **3. "Section 2.3 is titled 'Absolute Guarantees' and reports that against an exact adversary the store sits at AUC 1.000. Appendix E's own table shows a white-box adversary at 1.000 as well. The authors are welcome to argue that physical decay is not a privacy mechanism — they do — but they should not report the number in a form that requires the appendix to determine which system it belongs to, nor omit the one adversary model (σ_obs = 0.1) under which their store actually beats the TTL baseline 0.559 to 0.996."**
>
> **Can the paper answer it? PARTIALLY.** The disclaimers are genuinely present in the main text — "no certified (ε,δ) unlearning", "store-level guarantee only", "architectural retrieval geometry rather than cryptographic privacy" — so the paper is not accused of hiding the limitation. But it cannot answer the *inversion* charge, because the sentence really is ambiguous, and it cannot answer the selective-reporting charge, because the σ_obs row really is absent. **Fix MF-3 and this becomes the paper's best section rather than its most attackable one.**

---

## 11. Missing-experiment list for the Hub

Distinguishing what is genuinely absent from what exists but is not cited.

**A. Genuinely missing — task candidates.**

1. **A minus-the-physics / first-order-decay arm.** No non-symplectic, non-Hamiltonian, or first-order-decay store appears anywhere in the paper. The claim that needs it: that the V-curve and its computable optimum are *consequences of second-order (momentum) dynamics* and cannot be obtained from a scalar decay rate. Cheapest, highest-value missing control in the paper; answers SF-12 and the register's standing G2.
2. **A vault ablation separating brake from refrigerator.** App. F.7 proves algebraically that the 8.11× field-over-scalar advantage *is* the 7.942× refrigerator factor. No experiment isolates them (e.g. a cold spot with no extra damping, vs extra damping at matched local temperature). Answers the "which component buys what" attack directly and is a one-config run.
3. **Re-measure the emergent vault contrast arm** so MF-1's falsifier is either cleared or reported as a standing negative. The Limitations already names this as future work; it should be a task, not a bullet.
4. **Per-delete cost measured against a flat datastore**, to convert App. E's "O(n) rebuild per operation" from a buried admission into a stated price with a number. Limitations lists it as future work.
5. **Any task-level result at all** — even one toy retrieval task where the predicted half-life predicts observed recall. The paper's Limitations concedes this; it is the single thing that would move PALM from "interesting" to "convincing".
6. **The white-box adversary characterised, not just scored.** AUC 1.000 white-box is reported without saying what the adversary sees or why it is not the operative threat model.

**B. Exists but is not wired into the main text — must-fix wiring notes, not experiments.**

7. σ_obs = 0.1 leakage row (App. E Table 6) → §2.3. *(This is the paper's best privacy number and it is not on page 4.)*
8. App. F.3's finite-εµ/finite-γ corrections → §2.1. *(MF-4.)*
9. App. F.7's coupled-bath counterfactual (13.88×) and the absorb-only design choice → §2.2. *(SF-2.)*
10. App. C Fig. 8's "no continuous coset register" → §2.1's "transfers directly". *(SF-3.)*
11. App. E's waitlist-off collapse (byte-equal → 0.0000, AUC → 1.00000) → §2.3's conditions list. *(Item 8 of the §2 table.)*
12. App. A's Titans-differentiation sentence → §1. *(SF-13.)*

---

## Open questions / follow-ups / risks

- **The 0.29×–1.71× and "3-dimensional" figures (MF-6) are the only two numbers in this pass I could not source anywhere in the document.** Given that prior passes established the numeric spine as clean, the most likely explanation is that they came from a source outside the shipped document. The Head should confirm they exist before the paper asserts them; if they cannot be sourced in the time available, deleting them costs nothing.
- **The residual ~5% argmin gap** (measured 0.90 vs App. F.3's exact root ≈0.96 at the lowest µ²) may simply be the resolution of the γ sweep. Worth knowing before the MF-4 sentence is written, so it does not overclaim.
- I did not re-audit the numeric spine, per §5. Spot-checks I did run all reconciled: the vault ladder (8.41/23.04/44.89/110.25 predicted vs 8.42/22.39/44.31/107.77 measured), T_local = 1.259×10⁻⁴, the refrigerator factor 1/0.12591 = 7.942, and the underdamped floor 2ln2/0.05 = 27.7 ≈ 27.03. **No fabricated or mis-transcribed value found.** The defects in this report are defects of *placement and phrasing*, not of measurement — which is worth the Head hearing plainly.

## Proposed handover updates (for the Hub)

- V5 status: **`SUBMIT AFTER THE MUSTS`** — six MUSTs, all text-only, no experiment blocked. MF-1 (buried fired falsifier) and MF-3 (inverted leakage sentence) are the two that decide the outcome.
- Standing lesson for the shorts pipeline, generalising beyond V5: **relocation-condensation silently strips scope clauses, and no diff detects it.** The §2 table here found twelve instances. Recommend a mechanical pre-submission check on every future short — *for each main-text number, does its qualifying clause also appear in the main text?* — since the failure is invisible to every other instrument we run.
- Register note: G2 ("which component buys what") is **unanswered in V5** — there is no minus-the-physics arm and no brake-vs-refrigerator ablation. G7 ("stickman") is also unanswered: both the designed and the emergent arm are SO(2) on the same symmetric data, so "transfers to emergent units" is a within-symmetry-class statement. Neither is a V5 blocker; both are live for the longs.
