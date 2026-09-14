# `v5-referee-final` — the last look before V5 goes to PALM

**Agent:** `paper-referee`
**Commissioned by the Shorts Advisor at the Head's direction, 2026-08-26 (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 90).**
**Object:** `~/Desktop/V5_PALM_Submission/paper.pdf` — review it **exactly as the venue receives it**.
**Report:** `.claude/outputs/v5-referee-final.md`

---

## 1. What this pass is, and what makes a finding worth writing

This is the **final referee pass**. The Head closes V5 after it. ⇒ **the only finding worth writing is one that could change the outcome.**

⛔ **Every MUST must state the cost of not fixing it** — desk reject / a reviewer's stated reason to reject / a credibility hit / a reader misunderstanding a claim. A MUST with no stated cost is a SHOULD. The Head is deciding what is worth touching hours before submission, and an unranked list is useless to them.

⭐ **Be harsh.** The Head asked for a strict, hostile-but-fair pass. Do not soften. A defect you decline to name here ships.

---

## 2. ⛔ THE INDEPENDENCE BAR — absolute

⛔ **You are barred from reading ANY prior report on this paper.** Specifically: `outputs/v5-referee-v02.md` · `outputs/pj-referee-v5-r2.md` · `outputs/pj-fidelity-v5-r2.md` · `outputs/pj-minimal-v5.md` · `outputs/v5-cite-pass.md` · `outputs/v5-derivation-appendix.md` · `outputs/v5-final-pass.md` · `outputs/v5-palm-reframe.md` · every `BUILD-NOTE*.md` in the submission folder · and the charter itself.

**Why, stated so you do not route around it:** this program's most trustworthy signal has repeatedly been **two blind instruments converging on the same defect**. A finding you reach unaided is worth far more than one you were handed, and a pass that has read its predecessors cannot produce that evidence. If a number or claim needs checking, check it **against the PDF and against `refs.bib`** — not against our internal reports.

✅ **You MAY read:** the PDF, `paper.tex`, `refs.bib`, and the venue template files in the folder.

---

## 3. Calibration — who is reviewing this

**Venue: PALM @ NeurIPS 2026 — "Personalized, Aligned, Long-Term Memory for AI Systems". Short-paper track: 4 pages main text**, references and supplementary excluded. Non-archival, double-blind, NeurIPS 2026 template.

Review it as a **hostile-but-fair composite of that room**, which is agent/LLM-memory-native, not physics-native:
- a **long-term-memory systems** reviewer (MemGPT / Mem0 / Zep / Titans class) who will ask what this buys a deployed agent memory;
- a **membership-inference / privacy** reviewer who will read the leakage claims adversarially;
- a **representational-geometry** reviewer comfortable with dynamics and attractors;
- and a chair who applies the page limit mechanically.

⚠ **The scale gap is the obvious attack and you must price it honestly:** this is a small designed store at laptop scale, submitted to a room that mostly works on LLM agent memory. Ask whether the paper earns its place **by being about the right question**, and say plainly whether the room will buy that.

---

## 4. ⭐ The reading protocol — do this literally, in this order

1. **Read the MAIN TEXT ALONE, in full, before opening a single appendix.** Then answer explicitly: **does every claim the main text makes carry its qualification IN THE MAIN TEXT?**
   ⭐ *This is the specific risk of a paper condensed by relocation: a number moves to an appendix and its scope clause does not follow, so the paper acquires an unqualified claim without a word being edited — and no diff can see it. A PALM short-track reviewer is **not required** to open an appendix.* Name every claim that is stated in main text and qualified only in an appendix.
2. **State this paper's ONE contribution in your own words, in one sentence, after that first read.** ⭐ If you cannot, that is itself the finding — say so.
3. Only then read the appendices, and review them as supporting evidence.

---

## 5. What to weight, and what not to

⛔ **Do NOT re-audit the numeric spine.** Multiple independent passes have traced every headline number and found **zero fabricated and zero mis-transcribed values**; that is established. **Spot-check only**, and flag anything **new, internally inconsistent, or contradicted between two places in the document**.

Spend the effort here instead:

1. ⭐⭐ **Main-text self-containment** (§4.1) — the highest-value question in the pass.
2. ⭐ **The derivation appendix** — is it *sufficient and honest* for the closed forms the paper asserts? ⛔ Not a line-by-line re-derivation. Ask the reviewer's question: **can a sceptic reproduce the paper's central results from what is printed?** Flag any closed form, threshold or constant asserted with no derivation and no citation.
3. **Claim strength.** Read every quantitative sentence adversarially: does the evidence support the verb? Watch specifically for a **designed-testbed result reading as general**, a **law-confirmation reading as a performance win**, an **honest null reading as a finding in our favour**, and a **store-level mechanism reading as a system-level guarantee**.
4. **The deletion and leakage section**, hardest of all — it is the claim a privacy reviewer will attack. Are the conditions, exclusions and scope stated where the claim is made? Does anything read as a formal unlearning guarantee?
5. **Citations, now that BibTeX is wired.** Is the right work cited at the right sentence; is anything cited that does not support its claim; is any citation decorative; is anything asserted that plainly needs support and has none.
6. **Figures** — do they carry the claims they are cited for, and are they legible at printed size? Check axis/tick type, not just content.
7. **Anonymity / desk-reject surface** — author identity, metadata, acknowledgements, the class option, anything that identifies the authors or their institution. ⚠ Note: a third-person self-citation to prior published work is the sanctioned double-blind mechanism and is **not** a violation; say so if you see one rather than flagging it.
8. **Related work** — is the nearest neighbour in this room engaged with, or dodged?

## 6. The page limit — report it, do not solve it

**Measured: main text 4.35 pp against a 4-pp limit; 18 pp total.** Report the true split and **the chair's likely mechanical response**, including whether a 0.35-pp overrun is realistically enforced at a NeurIPS workshop.
⛔ **Do not recommend content cuts to fit.** If you believe the overrun is a genuine risk, say so once, with the cost, and move on. The Head has seen this number and accepted it; your job is to tell them if that acceptance is wrong, not to re-plan the paper.

## 7. Deliverables

1. **MUST / SHOULD / NICE**, each MUST with its stated cost of non-fixing.
2. ⭐ **A single unhedged verdict: `SUBMIT AS IS` / `SUBMIT AFTER THE MUSTS` / `DO NOT SUBMIT THIS CYCLE`.** ⛔ No hedging, no "it depends".
3. **A simulated PALM short-track score/recommendation** with the reasoning a reviewer would actually write.
4. ⭐ **The three most hostile quotes** a reviewer could write against this paper, verbatim as they would appear in a review — and for each, whether the paper can answer it from what is printed.
5. Your one-sentence statement of the paper's contribution (§4.2).
6. The main-text-standalone verdict (§4.1) with its list.
7. An explicit statement that the independence bar was honoured, and disclosure of any leak.

## 8. Boundaries

- ⛔ **Read-only on the paper.** You edit nothing — not `paper.tex`, not `pj_sub.tex`, not `refs.bib`, not a figure. Report the `md5` of `paper.tex` at the start and end of your pass to prove it.
- ⛔ **Never write to `.claude/NIPSsubmission/v5-palm/**`** — that is the Head's live editing copy.
- Building the PDF yourself is permitted (`/Library/TeX/texbin/pdflatex`, `bibtex`; not on `PATH`) but unnecessary — the shipped PDF is current and clean (0 errors, 0 undefined citations, 0 undefined references, 0 overfull boxes).
- ⚠ **Grep hazard:** directory-level grep over `.claude/` silently returns nothing (gitignored). Sweep per-file, and **positive-control every negative** before reporting "zero occurrences" — this program has repeatedly had to defeat a false negative from an unvalidated instrument.
- ⛔ **Do not propose rewrites of the Head's prose.** Name the defect and, where a fix is not obvious, give one minimal compliant alternative. The Head writes the words.

## DIAL DECLARATION
**Dials touched: NONE.** Read-only review. No experiment, no config, no registry, no charter, no paper file. Writes one report.
