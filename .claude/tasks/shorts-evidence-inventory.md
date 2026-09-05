# shorts-evidence-inventory — doc-curator

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 2 rulings) at the Head's direction, 2026-08-05.** Read the protocol, then the handover §10 header, then this file. This is a **documentation task**: repo is read-only, no git discipline needed, you write exactly one artifact.

## Why this exists
The Head is personally re-writing ALL workshop shorts for a simultaneous NeurIPS-workshop submission batch: **V1** (paid access / test-time compute), **V2** (mode-mass budget), **V3** (composition / firewall), **V5** (budget cube / forgetting-and-deletion), and **V6** (a short-form of the B′ audit paper). Every existing draft pre-dates C1W20+ and all of C2. The Head needs ONE verified document holding the full two-sided evidence base — every quotable positive and every binding null, with riders — so the re-writes and the Hub's owed A18.7 re-pass draw from the same source.

## Head rulings in force (Shorts-Advisor charter Addendum 2 — apply, do not re-litigate)
1. V5 is IN (re-scoped around forgetting + the R1 deletion estate). V6 = the B′ short exists as a target.
2. All shorts go out together to NeurIPS workshops; anonymization via third-person self-citation ("the CLU, introduced as CHLU in Jawahar & Pierini (2026)"); ⛔ shorts never cite each other (C-8 hermetic rule).
3. **Cross-borrowing is UNRESTRICTED** (non-archival venues; results may appear in multiple shorts' main text or appendices) — so the inventory must mark which results serve which shorts, not fence them.
4. Negatives are DISTRIBUTED: each short carries its own nulls in its APPENDICES (C-9). The inventory lists, per short, which nulls are mandatory riders.

## Sources of truth (precedence order — quote, never paraphrase numbers)
1. `.claude/claims_matrix.md` **v2.11** — §0.1–0.9 never-quotes are BINDING; CM-22/24/26/28/30 forbidden rows; CM-23/25/27/29/31 approved wordings (copy approved wordings VERBATIM where marked verbatim).
2. `.claude/negative_results.md` through **N224** (+ the DECLARED NOT-RUN blocks — never reportable as nulls).
3. `.claude/handover_context.md` §10 from `[C1-CLOSE]` (2026-07-30) forward — anything post-dating a registry version.
4. `.claude/advisor-head-c2-charter.md` §A13 (tiers/controls), §A18 (rulings), §A20 (re-labelling + substrate-scope sentence).
5. `.claude/papers/bprime/draft-r5.md` — the audit's frozen numbers (V6's source; referee pass still pending — mark V6 numbers "r5, pre-referee").
6. `.claude/outputs/*` for anything the registries point at.

## Deliverable — `.claude/outputs/shorts-evidence-inventory.md`
Structure (one section per short + three shared sections):

- **§0 Cross-cutting binding rules** (one page): the score sentence · the genuine-win bar (win-by-construction = supplementary) · §A20.2 re-labelling · the substrate-scope sentence (mandatory in every short, §A20.5) · pooled-n=9 + F3-before/after rule · modal-value byte rule · ⛔ CSF3 = pending, zero quotable numbers · "CLU-former" placeholder ban · quote-the-curve rule.
- **§1–§5 (V1 · V2 · V3 · V5 · V6)**, each with four sub-blocks:
  - **(a) Quotable positives** — every result in scope for that short, each as: the approved wording (or CM pointer where the wording is long), the numbers, the MANDATORY scope riders in the same entry, and the citation (CM row / N / `outputs/*` file / §10 date). Include supplementary-grade results explicitly labelled supplementary (e.g. the CL matched-bytes frontier, CM-23(w)).
  - **(b) Mandatory nulls** — the negatives that MUST ride in that short's appendices under distributed-negatives + registry-consistency discipline (e.g. V1: N1/N2/N3/N24/N37/N90/N95; V2: N46/N149-N150 blast radius; V5: N108/N112/N118 scope clauses; V6: the CLU-not-rescued row N218).
  - **(c) Highest-risk never-quotes** — the §0.x lines a writer of THIS short is most likely to trip (with the correcting form).
  - **(d) Stale-claim delta vs the current draft** — claim-level only (headline/contribution/abstract claims that the registry has since superseded, retired, or re-scoped), citing the superseding entry. ⚠ This is NOT a line-by-line referee pass — flag the claim-level deltas a re-write must handle; a referee pass comes later, per short.
  - For V6(d): instead, list what a 4–5 pp condensation must keep (the five mandatory columns, the rescue gate, 0-of-6 at n=9, the CLU's own row, the byte-floor identity) vs what stays long-form-only.
- **§6 V4 seed inventory** (theory-only): the parked "symmetry-allocated memory bank" candidate — `future_work.md` V4 section, `v2-symmetry-deepdive` §7 seeds (torus independence · O(α⁵) order-insensitivity · holonomy ∝α² · custodial n₁/n₂ = m₁/m₂ · GMO sum rules), each with its status tag (THEORY-ONLY / OPEN) and what a trained-checkpoint battery would need to exist before V4 is a V2-class short. No recommendation — the Head decides.
- **§7 Multi-short results** — the results that serve >1 short (e.g. the settle-costs-clean/pays-ambiguous sentence; the two-sided untrainability theorem; the trilemma), with the riders that travel wherever they go.
- **§8 Orphans** — quotable evidence with no natural home in V1–V6 (candidates for appendices or future work).

## Acceptance criteria (all mandatory)
1. Every number traces to a named source (CM row / N-number / `outputs/*` / §10 date) — zero unsourced numbers, zero numbers invented or recomputed.
2. Zero contradictions with matrix §0.1–0.9; where a draft and the registry conflict, the registry wins and the entry says so.
3. Every CSF3/tier-iii-scale item marked **PENDING — not quotable**; every pilot toy number carries its N94/TOY riders; declared NOT-RUNs never appear as nulls.
4. §A20.2 labelling enforced throughout ("the P-particle occupancy read protocol is refuted at P=4", never "the compositional family").
5. Approved wordings marked verbatim in the matrix are copied verbatim, not paraphrased.
6. Output is the single file above. Do not edit `.claude/papers/`, the registries, or the handover. Conflicts or registry gaps you find → a `## Flags for the Advisor/Hub` section at the end of your report, never fixed in place.

## Report
Write the deliverable itself as the output file; append the standard `## Proposed handover updates` + `## Flags` sections at the end. The Shorts Advisor reviews it against the registries before the Head relies on it.
