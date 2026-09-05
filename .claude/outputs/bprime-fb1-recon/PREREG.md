# PREREG — `bprime-fb1-recon` (search protocol, declared BEFORE reading)

**Filed 2026-07-31, Campaign 2 wave C2W3, by `web-scout`, before the first query was issued.**
Task: `.claude/tasks/bprime-fb1-recon.md`. Protocol §3 of that file requires the search rule to be
written down before a marginal paper can be graded against a feeling.

## 0. Dial declaration (echoed)
- **Dial/pillar:** none — recon.
- **Laundering control:** n/a; the *object being hunted* is a laundering control. Held to OUR definition,
  not to a paper's own use of the word "baseline".
- **Falsifies:** an established paper in the family that runs a non-parametric store, sized to the learned
  memory's declared **state bytes**, as a control on the same task, and reports the comparison verdict.
- **Does NOT falsify:** parametric matched-byte baselines · isoFLOP/isoParam · retrieval-augmented LM
  beating a neural LM · a clean negative.

## 1. Prior (committed)
- **P(FB1 fires) = 0.12.** Reasoning: three independent near-misses are already on the record
  (Zoology/Based App. E.2 reports state bytes but varies state by hyperparameters; MAD normalises
  iso-state across **neural** architectures only; SDM reports isoFLOP + isoParam). Three separate groups
  arriving *next to* the control and not running it is weak evidence that the control is not a community
  convention. Against that: the retrieval side (kNN-LM lineage) has a native non-parametric baseline and
  the datastore's byte size is routinely reported, so a byte-axis comparison could exist there by accident.
- **P(≥1 PARTIAL) = 0.65** — I expect at least one paper that runs the control at a *different budget
  convention*, or runs it and never draws the comparison.
- ⭐ **P(the matched-byte learned-vs-classical audit exists in a NEIGHBOURING field) = 0.7** — specifically
  learned data structures (learned index vs B-tree; learned Bloom filter vs Bloom filter; learned sketch vs
  count-min), where "at equal bits" is the *native* fairness axis. **Declared in advance:** such a paper is
  **NOT** an FB1 hit (wrong family — no test-time dynamics, no sequence memory), but it **narrows P4's
  wording** and must be cited as methodological ancestry rather than suppressed. Committing to this now so
  that finding it cannot be re-graded as either a hit or as irrelevant after the fact.
- **P(D4 — the substitute-audit idea is already published in some form) = 0.85**, in the
  benchmark-artefact / trivial-baseline "reality check" tradition rather than in the memory literature.

## 2. What counts as a HIT (frozen grading rule)
All four must hold:
1. **Non-parametric store** — table / kNN / count-based / explicit (k,v) rows / classical data structure.
   Another *neural* architecture is not a hit, however small (this excludes sliding-window attention,
   small transformers, and every iso-state architecture sweep).
2. **Sized to the learned memory's state bytes (or explicitly to its state dimension × dtype), declared.**
   Matched parameters only, isoFLOP, isoParam, or "same wall-clock" are **NOT** hits.
3. **Run as a control on the same task**, in the same table, not a related-work mention.
4. **The verdict is reported** — the paper prices the learned dynamics against the store.

Grades: `HIT` (4/4) · `PARTIAL` (3/4, name the missing one) · `NEAR-MISS` (2/4 or adjacent convention) ·
`NO`.

## 3. Declared search protocol (engines, queries, scope)
**Engines:** web search (Google-index-backed) + direct fetch of arXiv `abs`/HTML/ar5iv, ACL Anthology,
PMLR, NeurIPS/ICML/ICLR proceedings, OpenReview (expected to bot-block — will be reported, not omitted),
Semantic Scholar / connected-papers-style forward citation where reachable, GitHub repos of the anchor
papers.

**Date range:** anything up to **2026-07-31**; priority to 2024-01-01 → 2026-07-31; the retrieval lineage
back to 2016 (NTM/DNC, kNN-LM 2020).

**Query families (declared):**
1. `matched state bytes baseline` / `iso-state lookup table` / `equal-bytes table baseline` × {linear
   attention, SSM, recurrent, memory}.
2. `non-parametric baseline` + {state size, memory budget, KV cache bytes}.
3. Anchor-paper **forward citations**: Zoology (2312.04927), Based (2402.18668), MAD (2403.17844), TTT
   (2407.04620), Titans (2501.00663), DeltaNet (2406.06484), GDN (2412.06464), SDM (2607.07386),
   **Test-time regression (2501.12352)**.
4. **Backward** citations of the retrieval baselines: kNN-LM (1911.00172), Xu/Alon/Neubig (2301.02828),
   RETRO, Memorizing Transformers, NPM.
5. Deletion sweep: `exact deletion sequence memory`, `unlearning recurrent/state-space/KV memory`,
   MUNKEY (2603.15033) forward citations.
6. Collision-zone re-pin since 2026-07-30: new arXiv cs.LG/cs.CL neural-memory + audit/benchmark items.
7. D4: `trivial baseline beats neural`, `hypothesis-only baseline`, `reality check`, `control task`,
   `laundering by omission`, `same-bytes substitute`.
8. Neighbouring-field ancestry (declared as NOT-a-hit in §1): learned index vs B-tree, learned Bloom
   filter, learned count-min sketch, at matched space.

**Venues explicitly in scope:** arXiv (cs.LG/cs.CL/cs.NE), NeurIPS, ICML, ICLR, COLM, ACL/EMNLP/NAACL,
TMLR, JMLR, SIGMOD/VLDB (for §3.8 ancestry only).

## 4. Declared NOT-SEARCHED (will be repeated in the report, never presented as searched-and-empty)
- Non-English literature; patents; theses.
- Paywalled venues without an arXiv/anthology mirror.
- OpenReview reviewer threads if the bot-check fires again (precedent: C2W2 blocked twice).
- Vision/audio memory literature except where it surfaced on a query.
- The *entire* forward-citation set of every anchor (thousands of citations); I sweep titles/abstracts of
  the citation sets I can reach and say which ones I could not enumerate.

## 5. Stopping rule
Stop when (a) FB1 fires — report the same hour, do not finish the sweep; or (b) the declared query
families are exhausted with no new candidate classes appearing in two consecutive query families.
