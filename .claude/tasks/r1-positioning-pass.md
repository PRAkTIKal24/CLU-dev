# r1-positioning-pass — R1 in existing vocabulary (Ginart Def A.5 · Sekhari Def 3)

**Agent:** paper-writer. **No worktree** (no tracked code). Base local `main @ 082d095` (post-w26).
**Campaign tag: [C1W27].** Head ruling 2026-07-29 (queue item 6 = YES; "R1 repositioning on Ginart
Def A.5 + Sekhari Def 3 vocabulary" in the also-in-scope list). Source of record:
`.claude/outputs/deletion-prior-art.md` (read §§ the wording gate, the Candidate-2 table, and the
banned-terms list in full) plus `.claude/outputs/placement-landing.md` and
`.claude/outputs/mia-decay-measurement.md` §1.6.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** none — **positioning / wording.** No new evidence is generated or claimed here.
- **Laundering control:** every R1 sentence you write must name the **flat-datastore row-delete** as
  the trivial substitute, and must not claim to beat it. A dict delete is exact by construction; our
  claim is that the *physical* store matches it, with a stated scope.
- **Falsifies the claim:** any sentence you produce that survives only because a scope qualifier was
  dropped. Self-check every sentence against its qualifier.
- **Does NOT falsify:** the fact that our theorems are instantiations of prior work — that is the
  finding, and the defensible novelty is stated below.

## 1. The prior-art position (adopt, do not coin)
⛔ **Blelloch–Golovin FOCS'07 own the algorithm outright** — our Thms 1–2 are **instantiations** of
their Thm 3.2 + `DELETE`, and the fix-up cascade is **theirs**, not ours. Our defensible novelty is
the **four-part composition**: (i) the packing certificate · (ii) decaying content + the commutation
proof · (iii) the canonical **energy function** · (iv) the **negative** SHI price, against
Buchbinder–Petrank's proven exponential cost.

**Candidate 2 is occupied in pieces — adopt existing vocabulary:**
- **Cost side:** Ginart et al. (NeurIPS 2019, arXiv:1907.05012) **Def. A.5** — α-deletion
  efficiency, amortized `O(n^{1−α})`.
- **Matched-utility side:** Sekhari et al. (NeurIPS 2021, arXiv:2103.03279) **Def. 3 — deletion
  capacity**, the max number of deletions at a fixed excess-risk budget. This is the closest existing
  formalisation of Candidate 2. **Adopt it, or state explicitly why the store setting needs a
  different one.**
- **Practice precedent:** SISA's shards-vs-accuracy curve. Also cite MUSE criteria 4+5+6 and
  CURE4Rec where the multi-dimensional framing is used.
- ⛔ **Do NOT coin a benchmark name.**
- ⚠ Hartline needs a **reversibility caveat** — our amplitude layer is not reversible.
- ⚠ **Re-typeset equations from source** (Sekhari Def. 3, Guo Defs 1–2) — the scout's quotes came
  through a text-extraction proxy and are faithful in substance but not in punctuation.

## 2. The banned-terms sweep
Sweep `.claude/papers/**` drafts and every R1 sentence for: **"certified"** (banned since w24) ·
**"unlearning"** · **"deletion-compliant"** (EUROCRYPT'20 term of art, newly banned) · **unqualified
"exact deletion"** (it invites a model-level test we fail on the encoder channel) · **"our fix-up
cascade"** as a possessive · any claim that **decay reduces distinguishability per se** · any claim
that **eviction removes the item** (under the shipped placement it does not). Produce a file:line
table of every hit and its replacement.

## 3. The vision-doc line (Head-approval deliverable — do NOT edit the doc)
`.claude/research_roadmap.md` Part 4 still reads *"deletion as a **certified** physical operation"* —
**triply forbidden**. Draft the replacement, built from the scout's approved §1.6 paragraph, and
deliver it **as an exact replacement string in your report** for Head approval. ⛔ The curator owns
`research_roadmap.md`; you do not edit it. Same for `claims_matrix.md`, `negative_results.md`,
`philosophy-synthesis.md`.

## 4. The scope qualifier — supply BOTH wordings
Every R1 sentence currently carries *"operating below capacity or under set-function eviction"*
(`AUC(z_hole)` 0.5000 ± 0.0000, byte-equal 3072/3072 below capacity; `AUC(n_live)` = 1.000 at
overflow). **`deletion-waitlist-stiffness` may land the P2 waitlist this wave.** Therefore deliver
**two** versions of each R1 claim sentence: the current (scoped) wording, and the post-waitlist
wording — the latter conditional on that task's measured load sweep, since
`carried-remeasurements` showed the leak is a **curve** (AUC 0.6715/0.9165/0.9961/0.99985 at 2/4/6/8
offers) ⇒ **the acceptance number must be stated at a load**. The Hub will relay that task's exact
scope sentence to you if it lands before you finish; if it has not landed, mark the second version
**PENDING** and do not guess a number.

## File ownership
**You own:** `.claude/outputs/r1-positioning-pass.md` and edits to `.claude/papers/**`.
⛔ **Do NOT edit** any transfer doc (curator-owned this wave), any task file, or any `chlu/` code.

## Deliverable
`.claude/outputs/r1-positioning-pass.md`: the adopted-vocabulary paragraph (with citations, BibTeX
keys reused from `deletion-prior-art.md`) · the file:line banned-terms table with replacements · the
exact Part-4 replacement string for Head approval · both scope wordings per §4 · a one-line statement
of the four-part novelty composition that any future draft can paste. Flag anything you could not
verify against a source rather than smoothing it over.
