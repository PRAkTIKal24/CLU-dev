# `v1-ttcl` — V1's working folder

**Created 2026-08-26 by the Shorts Advisor at the Head's direction**, mirroring `v2-neurreps/` and `v5-palm/`.

| file | what it is |
|---|---|
| `submission.tex` | **the base.** A verbatim copy of `.claude/papers/v1-short/draft.tex` (v0.4, 2026-07-19) with `\includegraphics` repointed to `figs/`. Builds clean: 0 errors, 0 undefined references, 18 pp. |
| `figs/` | the six banked V1 PNGs (four are used by the base). |
| `neurips_2026.sty` | a NeurIPS-family style, present so a template port is possible later. ⛔ **Not TTCL's template** — TTCL's is not held and its venue fields are stale. |
| `pj_sub.tex` | **does not exist yet.** The Head creates it from `submission.tex`. |

---

## ⛔⛔ THE ONE THING THAT MAKES THIS FOLDER DIFFERENT FROM `v2-neurreps/` AND `v5-palm/`

In those two folders, `submission.tex` was a **registry-current clean base** — it had been through revision passes, cite checks and referee passes against the live claims matrix and negatives registry before the Head ever hand-edited it.

**`v1-ttcl/submission.tex` has NOT.** It is a **2026-07-19 artifact that predates the entire of Campaign 2** — six waves, ~90 registry entries, and a campaign boundary. Advisor-verified on disk at creation:

- ⛔ **the measured score sentence is ABSENT** (`external benchmark` = 0, `headline metric` = 0) — mandatory under charter §4.1;
- ⛔ **the §A20.5 substrate-scope sentence is ABSENT** (0 hits);
- ⛔ **the venue-class header is stale** — it reads *"ML4PS / NeurReps workshop short … final venue pending the Jul-11 scout"*; the ruled venue is **TTCL**, and ML4PS has no 2026 edition;
- ⚠ `[AUTHORS PLACEHOLDER]` is still in the draft;
- ⚠ **9 "theory note" mentions**, never assessed for self-containment (V5's equivalent assessment took it to zero);
- ⚠ **MQAR appears 7×** — the Add.10 F2 venue-admissibility question, queued for this pass and never taken.

✅ **Checked and NOT a defect:** the draft's `wins`/`beats` language (3 + 3 uses) is the paper **conceding losses** — the learned router beating our own energy-gated edge — not a CLU-performance overclaim. An audit need not spend a pass there.

⇒ **The first pass on V1 is a claims-currency audit, not a typeset or a hand-edit.** Hand-editing this base first would mean condensing a document that still contains claims the registries retired — the one failure mode the V2/V5 arc never had to face.
