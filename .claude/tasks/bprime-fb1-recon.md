# bprime-fb1-recon — FB1: has anyone already run the matched-byte non-parametric control?

**Campaign 2, wave C2W3. Agent:** web-scout. **NO WORKTREE, NO CODE, READ-ONLY on the repo.**
Launch **immediately** — you gate on nothing, and you are cheap.
**Funded by Head ruling, 2026-07-31 (C2W3 decision queue, Q2: "yes")** — a seventh task, added to
charter ADDENDUM 2 §A11's six. ⭐ **This is the only departure from §A11's task list in C2W3, and it is
by explicit Head ruling.**

**Read first:** `.claude/AGENT_PROTOCOL.md`;
`.claude/outputs/track2-admissibility/PREREG-Bprime.md` **IN FULL — §5 P4 and §6 FB1/FB5 are your
entire brief**; `.claude/outputs/track2-admissibility.md` **§2 (the evidence and citations behind the
prereg) and §3 (the SDM brief)**; `.claude/outputs/rival-recon.md` **in full — it is the C2W1 survey you
are extending, and its §F2 byte conventions are what "matched-byte" means here**;
`.claude/advisor-head-c2-charter.md` **ADDENDUM 2 §A9.1 (B′'s shape) and §A11**; the **live
`2026-07-31` `[C2W2]` §10 entry** in `.claude/handover_context.md` (the never-quote additions).

⛔ **REGISTRY LAG — THREE WAVES (C1W27 · C2W1 · C2W2).** Quote our own results **only** from
`.claude/outputs/*` and the §10 review entries — never from `negative_results.md` (N122),
`claims_matrix.md` (v2.5), `research_roadmap.md` (v0.9), the ledger (⟲ w26) or the primers.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **none — recon.** No dial, no leaderboard, no dividend, no measurement of ours.
- **Laundering control:** n/a — **but the object you are hunting IS a laundering control**, and you must
  hold candidates to *our* definition of it, not to a paper's own use of the word "baseline". A rival
  that reports state bytes, or normalises iso-state across **neural** architectures, or runs isoFLOP /
  isoParam, has **not** run a matched-byte **non-parametric** control. Those three near-misses are
  already on the record (Zoology/Based App. E.2 · MAD · SDM respectively) — do not re-file them as hits.
- **Falsifies:** §4. **Does NOT falsify:** finding a paper that runs a *parametric* baseline at matched
  bytes (that is not FB1) · finding a retrieval-augmented LM that beats a neural LM (that is not the
  same claim) · returning a clean negative (**a well-protocolled negative is this task's most likely
  and most valuable outcome**).

---

## 0. Why this task exists, in four sentences
B′'s claim is *"when does test-time dynamics buy anything over a table at matched bytes?"*, and its
novelty rests entirely on **P4: no published rival paper runs a non-parametric matched-byte control.**
P4's status at filing is **"none found — absence of evidence, medium confidence"**, which is the
weakest kind of claim a paper can be built on and the cheapest to check. **FB1 fires if ≥1 established
paper in the family already runs that control and reports the same verdict** — in which case **B′ is a
replication, not a contribution**, and the Head needs to know **before** `bprime-rivals` (the wave's
heaviest task) has spent a worktree on six families' worth of audit columns.

## 1. Deliverables

### D1 — ⭐ THE FB1 SWEEP, and it is a VERDICT, not a reading list
> **FB1 — "not news."** ≥1 established paper in the family already runs a **matched-byte non-parametric
> control** and reports the **same verdict** ⇒ **B′ is a replication.** *(Kills P4 and B′'s novelty.)*

Return **`FB1 = FIRES` or `FB1 = DOES NOT FIRE`**, in your first 10 lines, with the evidence.

**The search space** (extend `rival-recon`, do not repeat it):
- **Test-time / fast-weight memories:** TTT (TTT-Linear, TTT-MLP), **Titans (NeurIPS 2025 — ⛔ never
  "a preprint")**, Titans-Revisited (arXiv:2510.09551), fast-weight programmers, Sparse Delta Memory
  (arXiv:2607.07386).
- **Linear-attention / SSM state families:** DeltaNet, Gated DeltaNet, GLA, Mamba-1/2, and the FLA
  ecosystem's evaluation conventions.
- **The benchmark/analysis line, where a control like ours is most likely to already exist:** Zoology,
  Based, MAD, RULER, and any iso-state / iso-memory analysis paper.
- **The retrieval side, which is where a *non-parametric* baseline is native:** kNN-LM (Khandelwal),
  Xu/Alon/Neubig ICML'23, RETRO, retrieval-augmented and memory-augmented LMs, NTM/DNC-lineage work.
- ⭐ **FB5's neighbour and its citing literature:** Wang, Shi & Fox, *"Test-time regression: a unifying
  framework…"* (arXiv:2501.12352) already unifies linear attention, SSMs, fast-weight programmers,
  online learners and softmax attention as special cases. **B′ differentiates on the empirical
  byte-matched audit — they unify mechanisms, we price them. ⛔ If that line (or anything citing it) has
  since added a non-parametric matched-byte baseline, FB1 FIRES.**

**What counts as a hit — hold this line strictly, in both directions:**
| counts as FB1 | does NOT count |
|---|---|
| a **non-parametric** store (table / kNN / count-based / explicit (k,v) rows) | another **neural** architecture, however small |
| sized to **match the learned memory's state bytes**, declared | matched **parameters** only · **isoFLOP** · **isoParam** |
| run as a **control on the same task**, and its verdict reported | a *related-work* mention, or a retrieval baseline at a different budget |
| **and reports the same verdict** (i.e. prices the dynamics against it) | reports the baseline but never draws the comparison we draw |

⚠ **A partial hit is the most likely real outcome and it is the most useful thing you can return.**
If a paper runs the control but at a different budget convention, or runs it and does **not** report the
verdict, say so precisely — **that is a "P4 survives but narrows" finding**, and it changes how B′ is
worded rather than whether B′ exists. Grade every candidate `HIT / PARTIAL / NEAR-MISS / NO` with the
reason.

### D2 — ⭐ Upgrade P4 from "absence of evidence" to a DEFENSIBLE negative
A negative is only citable if the search that produced it is declared. Deliver:
- **the search protocol** — engines, queries, date range, venues, arXiv listings swept, and the
  **snowball** you ran (forward citations of the family's anchor papers; backward citations of the
  retrieval baselines);
- **what you could not reach and why** (⚠ **OpenReview is bot-blocked — it blocked C2W2's scout twice
  and that was reported both times.** Report it again if it blocks you; do **not** silently omit a venue);
- **a confidence grade** on P4 with the reasoning, replacing "medium confidence" with something a
  referee can check;
- ⭐ **the sentence B′ should actually print.** *"No published rival paper runs a non-parametric
  matched-byte control"* is a strong claim in a crowded field. Write the **narrowest true version** of
  it — scoped to the families surveyed, the date pinned, the protocol cited. That sentence is your
  single most valuable deliverable.

### D3 — The near-neighbour watch (pins are 2026-07-30; you are re-pinning)
- ⚠ **Sparse Delta Memory was three weeks old at C2W2 and is a near-collision.** Check for anything
  newer in the collision zone since the C2W2 pin, in the neural-memory *and* the audit/benchmark lines.
- ⚠ **Pillar 4's last differentiator:** sweep **MUNKEY's (arXiv:2603.15033, ICML 2026) citing
  literature** for **exact** deletion in a **sequence memory**. MUNKEY narrowed our claim already
  (MIA-AUROC → 0.5 **by design**, but **not exact** — gap to retraining **0.56 ± 0.21**, and a **ViT
  classifier**, not a sequence memory). ⛔ **If someone has since published exact deletion in a sequence
  memory, that is a positioning emergency — report it the same day** (the C2W2 precedent: this exact
  falsifier was registered as "report same-day" and it is still live).
- ⛔ **Quote NO SDM Table 1 state/param ratios** — two independent extractions disagree (156 % vs 168 %;
  111 % vs 98 %). If you can resolve the conflict from the source, that resolution is itself a
  deliverable; if you cannot, the quarantine stands.

### D4 — One question that is cheap while you are in the literature
**Has anyone published the *substitute audit* idea** — that a **+0 B** reader of the same bytes (row
order, the query itself, an echo) can match or beat the learned system, and that reporting only the
frozen control is a laundering by omission? Our audit went **0-for-4** and we treat it as ours. If it
is not ours, B′'s framing changes and it is far better to learn that now. Same grading as D1.

## 2. Method / discipline
- ⛔ **Read-only. Never edit the repo.** Your artifacts live under `.claude/outputs/bprime-fb1-recon/`.
- **Every claim carries a citation with a resolvable identifier** (arXiv ID, DOI, venue + year), and
  ⛔ **venue metadata is checked, not assumed** — C2W2 caught "Titans is a preprint" propagating through
  our own documents when it is **NeurIPS 2025, peer-reviewed**, and caught a **Guo citation defect**
  ("Def. 1/Def. 2" does not exist; it is §3 Eq. (1) inline) that had reached the running log as fact.
- **Where two extractions of the same table disagree, quarantine the number and say so** — that is now
  standing practice because of SDM.
- **BibTeX-ready refs for everything new**, appended to `rival-recon`'s and `track2-admissibility`'s
  lists rather than duplicating them.

## 3. PREREG
Not applicable in the measured-quantity sense. ⭐ **But declare your search protocol BEFORE you run it**
(`.claude/outputs/bprime-fb1-recon/PREREG.md`): the queries, the venues, the date range, and **what you
would accept as a hit** — written down before you start reading, so a marginal paper is graded against a
rule rather than against how you feel about B′ at that moment. State your prior on FB1 firing.

## 4. Falsifiers
- ⛔ **FB1 FIRES** — an established paper already runs the matched-byte non-parametric control and
  reports the same verdict ⇒ ⭐⭐ **B′ is a replication; P4 is dead; the wave's heaviest task needs
  re-scoping before it spends its worktree. REPORT TO THE HUB THE SAME HOUR** — do not finish the rest
  of the sweep first.
- ⛔ **FB5's line has added a non-parametric matched-byte baseline** ⇒ FB1 fires by that route.
- ⛔ **Someone has published exact deletion in a sequence memory** ⇒ **positioning emergency, same-day
  report** (pillar 4's last uncontested differentiator).
- ⛔ **The substitute audit is not ours** (D4) ⇒ B′'s framing changes; report with the citation.
- **Does NOT falsify:** a clean negative (**the expected and valuable outcome — it is what makes P4
  citable**) · a near-miss that narrows P4's wording without killing it · a venue you could not reach,
  **provided you declare it**.

## 5. ⛔ Never-quote (inherited — you are the agent most likely to introduce a citation defect)
**Titans as "a preprint"** (**NeurIPS 2025, peer-reviewed**) · any **SDM Table 1 state/param ratio**
(two extractions conflict) · **"MAD `compression` is the admissible synthetic"** (dead by arithmetic —
its own iso-state normalisation at 4096 dims exceeds the task's 224 B max payload by **73× at fp32 /
36× at bf16**, so compression in MAD is **never beyond-capacity**) · **"principled forgetting"** as a
novelty phrase · **"we alone delete"** (the MUNKEY narrowing) · any CHLU cell as a **byte-matched**
dividend (min ratio anywhere **17.11×**) · the recency family's **`0.3019 ± 0.0679`** as a null
(scoring-domain **defect**) · any **`AttentionPsi`** trajectory number (it leaks).

## 6. Output
`.claude/outputs/bprime-fb1-recon.md`, protocol §5 format, with:
- ⭐ **`FB1 = FIRES | DOES NOT FIRE` in the first 10 lines**, with the deciding evidence — the Hub reads
  that line and either releases or re-scopes `bprime-rivals`;
- the **candidate table**, one row per paper checked, graded `HIT / PARTIAL / NEAR-MISS / NO` with the
  reason (the near-misses are as informative as the hits: they are the field's own evidence that nobody
  runs this control);
- ⭐ **the narrowest true version of P4's sentence**, ready to print;
- the **declared search protocol** and everything you could not reach, with the reason;
- **BibTeX-ready refs**, new only;
- your reconciliation list in the **first 10 lines** if you produce one (protocol §5 corollary — a
  citation correction is exactly the kind of item that rots for two waves without an owner);
- ⛔ **declared NOT-SEARCHED areas, never presented as searched-and-empty.**
