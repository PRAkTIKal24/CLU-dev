# pj-fidelity-v5-r2 — doc-curator — ROUND 2 fidelity audit of the Head's rewritten V5 `pj_sub.tex`

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 57, 2026-08-22).** Read `.claude/AGENT_PROTOCOL.md`, then this file. Output: `.claude/outputs/pj-fidelity-v5-r2.md`.

**Context:** the Head has completely rewritten `pj_sub.tex` since the round-1 audit (`outputs/pj-fidelity-v5.md`, 2026-08-22). The round-1 syntax errors are fixed — the file now compiles clean (the Advisor rendered `pj_sub.pdf`, **11 pages**, zero repairs; the old `pj_sub_buildcopy.*` is obsolete lineage, ignore it). Round 1 found the numeric spine clean and **the campaign's most serious claims error — an affirmative "certified removal"**; this round verifies the rewrite and re-adjudicates every round-1 finding.

**DIAL DECLARATION: none — read-and-report only; zero file edits anywhere.**

## Absolute constraints
- ⛔ **`pj_sub.tex` is EDIT-BARRED (Head ruling, Add.53, still in force). You issue ZERO writes against any file in `.claude/NIPSsubmission/` — your only write is your report.** Advisor-pinned md5 at task issue: `6c1902f74ee9611d718cc65b9fd1a031`; the Advisor re-verifies after your pass.
- ⚠ You have NO shell tool (Read/Grep/Glob/Write only). No render step exists in this task — the PDF already exists. If any check seems to need a shell, report the limitation honestly; never fake it.
- Object: `.claude/NIPSsubmission/v5-palm/pj_sub.tex`. Source of truth: `.claude/NIPSsubmission/v5-palm/submission.tex` (the accepted clean base, Add.52) — and, where the base inherits an approved wording, the registries as cited below.

## Part A — numeric fidelity (the Head's standing question)
Every numeric token in `pj_sub.tex` matched against `submission.tex` for **value, precision, units, ±/CI, seed counts, and scope**. ⛔ **Any number lacking an ancestor in the source is the most serious finding class available** — quote it with context and name the nearest source number.
⭐ **New in round 2 — the citation ancestry check:** the rewrite adds deployment-literature citations the clean base does not carry (observed at minimum: Rasmussen et al. 2025 (Zep) · Mem0 · Chakraborttii et al. 2026 · Yang 2026 · Uddin et al. 2026). For EVERY cited work: ancestor in the base's bibliography, or **NEW**? List every NEW citation (author, year, claimed venue/ID) with the claim it supports — new citations are UNVERIFIED records and the list feeds a cite-check spoke. Do not attempt to verify them (no web access); ancestry only. Also list any base entry DROPPED while its in-text claim survives.

## Part B — claims fidelity + the round-1 re-adjudication (⛔ the "certified" item above all)
1. **Re-adjudicate every round-1 V5 finding by direct quotation (new text beside base text): FIXED / PARTIALLY FIXED / UNFIXED:**
   (a) ⛔⛔ **the "certified removal" inversion.** The base's compliant form is an explicit DENIAL (*"we do not claim certified (ε,δ) unlearning"*). The rewrite's known passage (l.80 region) uses "certified removal" as the literature's term (Guo) and positions us as *"functionally between these approaches"* offering a *"store-level, bit-exact structural guarantee"*. Adjudicate precisely: does any sentence claim or imply certified removal as OUR property? Is the explicit denial still present anywhere, or only implied by contrast? Quote every "certified" occurrence with a verdict per occurrence. Authorities: charter §4 (never-quote) · N118 · CM-25(f) · the corrected Guo cite form (**§2 Eq. (1), ε-only**; the (ε,δ) form is the unnumbered display after it — ⚠ note the rewrite says "(ε,δ) relaxations" against Guo: check this against the base's corrected form).
   (b) store-level deletion WITH its three conditions and the recency exclusion — present beside the claim, or absent?
   (c) the score sentence (external benchmarks won on their own headline metric = zero) and the deletion section's trivial-substitute laundering control — present?
   (d) the Blelloch–Golovin no-priority clause (*"we claim no priority… the fix-up cascade is theirs"*) — the citation survived round 1; did the clause return?
2. **The do-not-cut walk (the `v5-referee-v02` list, same as round 1):** N108's sentence ("stops answering before it stops leaking") · the exact-deletion form with its three conditions + recency exclusion · Blelloch–Golovin at every deletion site · the lifecycle riders (§0.13 approved form only — no VALUE number, no C2W8 cell numbers) · the substrate-scope sentence (§A20.5) · the score sentence · the emergent-arm caveats (designed-only contrast scope; the vault's LAWS transfer, never "the vault transfers"; no emergent σ_θ ratio) · the k-regime clause · the corrected Guo form · the honest scope sentence (not a deployed agent memory, exactly once) · ⛔ "right-to-be-forgotten"/"memory provenance" never claimed as our properties (naming the topic is permitted). For each: **present / absent — and where absent, which claim now stands unqualified**, ranked by consequence.
3. **Claims table:** every surviving substantive claim side by side with its base form, ruled **IDENTICAL / NARROWER (safe) / WIDER (⛔) / CHANGED IN KIND**. V5 drift modes: store-level deletion reading as system-level unlearning · the vault's laws-transfer reading as the vault transferring · a designed-arm result reading as general · TTL-comparison scope drift.
4. **Mechanical inventory:** approximate page split; ⚠ **figure count = 0 in `pj_sub.tex` vs 11 `\includegraphics` in the base** — list which base figures carried claims that now stand figure-less (the referee prices the consequence; you report the mapping).

## Acceptance criteria
- Every Part-A mismatch and Part-B absence quoted with line context, ⛔ flagged never fixed.
- The round-1 re-adjudication table complete; the "certified" adjudication per-occurrence.
- The NEW-citation list complete (or explicitly empty).
- Zero writes outside `.claude/outputs/pj-fidelity-v5-r2.md`.
