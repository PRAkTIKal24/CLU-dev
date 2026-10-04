# w20 spoke/referee3-fixes — ITEMS (a)–(g) IN BOTH DRAFTS
**Agent type: paper-writer. Zero compute, zero scored runs.** ⭐ PRE-AUTHORISED.
⛔⛔ **EXCLUSIVE `.tex` LOCK.** You edit `DRAFT-v2.tex` and `DRAFT-v2-style.tex`. ⛔ If any `w18-placeholder-fill`, `w19-draft-figure-slots` or `w20-citation-prune` session is live, **STOP and say so** — two agents on one file corrupts it.

## Apply in BOTH `.tex` files
**(a) Positioning — rewritten to the Head's framing.** The CLU is a **NAIVE AI UNIT**, compared against **other naive units** (MLP / LSTM / transformer / continuous MLP). ⛔ **The paper does NOT claim the best compositional-generation or unlearning system.** Systems built on top of units — sharded retraining, compositional diffusion, physics-informed variants of an MLP — are **out of scope by definition**, cited in **ONE sentence each** in related work **with the unit/system distinction made explicit**. ⛔⛔ **The sentence "the intersection is empty" is REMOVED EVERYWHERE.**
**(b) M2** — replaced by the wave-20 rival set. ⚠ **Until it lands, the placeholder NAMES BOTH rival sets and their pools** (16,384 · 32,256 · store 131,072).
**(c) M5** — split every shared-environment table and figure. ⛔ CMAZE-1 / CMAZE-2 / CMAZE-2b / CMAZE-T / CMAZE-O never share one.
**(d) M6** — a seeds column on every table. An n = 1 row says n = 1.
**(e) Strip pipeline language.** Remove: wave numbers, "bar k", "charter controls", "the Head", "spoke", "ruling". ⭐ Define **"pre-declared landscape controls"** and **"pre-registered criterion k"** ONCE and use them. ⭐ Move **"Corrections absorbed"** out of the submission.
**(f) K\*** — report K\* as selected by the frozen rule, **AND** state that on three seeds **the rule cannot separate the plateau K = 5…32** (seed sd > 0.02). ⭐ **The finding is the plateau itself: composition flat across K while the ledger falls to its minimum near K = 24.**
**(g) Deletion wording — ⛔ DO NOT TOUCH.** The Head rewrites it in the methods (exact on the store *content*; address book **carried, not refit**).

## ⛔ Explicitly NOT yours
The **AI-use statement**, the **page trim**, and the **rest of venue compliance** are the Head's. ⛔ Do not touch them.

## Invariants
⭐ Both drafts must still **compile with 0 undefined citations** — compile `pdflatex → bibtex` (with `BIBINPUTS`/`BSTINPUTS` = `w17-paper/`) `→ pdflatex ×3`. ⚠ bibtex had silently never succeeded before wave 19; **check the count, do not assume**.
⭐ **Re-run the style diff-check** and show all four diffs EMPTY: numbers, placeholders, artefact comments, section order.
⛔ Do not touch the wave-19 placeholders or any placeholder you cannot fill.

## Acceptance
`REFEREE3-FIXES.md`: (a)–(f) each with **before → after** and location; (g) listed as **untouched, by instruction**; the AI-use/page-trim/venue items listed as **the Head's**; four diffs EMPTY; both compile logs with the undefined-citation count shown.

## What would prove this wrong
⭐ If removing "the intersection is empty" leaves the positioning with **no claim at all**, say so — that means the contribution sentence depended on it, and the Head needs to know before the trim pass. ⛔ Do not invent a replacement claim.
