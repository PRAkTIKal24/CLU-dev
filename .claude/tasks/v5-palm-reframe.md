# v5-palm-reframe — paper-writer (V5, reframed in the PALM audience's own terms; a SEPARATE variant)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 38; Head ruling 2026-08-20 — V5 reframes for PALM, V2 for NeurReps).** Read `.claude/AGENT_PROTOCOL.md`, then this file. **No gate: the PALM audience is already scouted and verified** (`outputs/v5-scope-scout.md`).

**You write ONLY into `papers/palm-variant/v5/`** (create it): `submission.tex` · `submission.pdf` · `BUILD-NOTE.md` · `figs/`. ⛔ **Nothing under `papers/v5-short/` is touched** — the canonical v0.4 and the current PALM build (10 pp; main text 4.00 pp, venue-compliant) stay exactly as the Head left them. This is a parallel variant, kept apart to avoid confusion.

**DIAL DECLARATION: none — reframing/editorial pass; zero content, number or claim changes.**

## Source and the one job
Source = `papers/v5-short/submission/submission.tex`. **Same results, same numbers, same claims — re-expressed in the vocabulary and priorities of PALM's audience**, and written more directly than the source.

**The audience, from its own CFP (`outputs/v5-scope-scout.md` Part 1, verbatim quotes there):** *"Personalized, aligned, long-term memory for AI systems"* — systems that *"retain information across sessions … and sometimes forget past experience"*, built on *"persistent memory layers that encode, retrieve, update, and sometimes forget."* Communities: ML, NLP, AI agents, HCI, cognitive science, neuroscience, privacy, security, AI safety. Their seven topics include **"memory update and deletion tests"**, **"right-to-be-forgotten mechanisms"**, **"interfaces for inspecting, editing, deleting, and scoping memories"**, **"privacy leakage"**, **"handling stale or contradictory memories"**, and consolidation/forgetting policies.

## What reframing means here, concretely
1. **Lead with the memory-policy question, not with the physics.** This audience's problem is: *what is the retention policy, is deletion real, and what leaks?* Our three results answer exactly those, so state them that way and let the physics be the mechanism that makes the answers computable — the ordering is policy question → mechanism → number, not mechanism first.
2. **Use their operational vocabulary where the mapping is exact** (the scout's Part 2 records back these terms): retention policy · **TTL / expiry** · consolidation · scoping · **deletion guarantee** vs best-effort deletion · **right-to-be-forgotten** · memory provenance · **privacy leakage / membership inference** · stale-memory handling. ⛔ Where our object has no operational equivalent, keep our term and define it in one clause; never borrow an operational term that would imply a deployed system we do not have.
3. **⭐ The TTL comparison is the honest centrepiece, not a footnote.** TTL/expiry is what this audience's systems actually do, and our own laundering control measured that a boolean TTL flag is **indistinguishable from physical decay against an exact adversary (0.983 vs 1.000 AUC; 0.559 vs 1.000 at σ = 0.1)**. Lead the leakage result with that: what a physical decay buys over a TTL flag is **retrieval geometry**, not privacy — and print N108's sentence, *"the store stops answering before it stops leaking"*, verbatim, in the same breath.
4. **Frame the three contributions in their terms:** (a) the damping optimum = a **retention dial with a computable optimum**, the policy statement being that retention is non-monotone in the dial and the optimum is predicted, not tuned; (b) the vault + the new emergent confinement result = **scoped retention** — a local change that confines one item's coordinate; (c) deletion = a **structural deletion guarantee** (canonical placement makes the store's state a function of its live set alone), stated as the composition claim — placement certificate + decaying content + delete/decay commutation under one energy function — with the Blelloch–Golovin attribution and all three conditions verbatim. The three-state lifecycle maps to their consolidation/stale-memory topic and stays in its approved mechanics-only form.
5. **Related work (§2, directly after the introduction) is rebuilt for this audience** from the scout's 13-work brief: external-memory agent architectures · decay/expiry/consolidation policies in deployed systems · the 2026 forgetting/deletion evaluation wave · the fixed-size-state retention line · **Titans' learned forget gate as the nearest published neighbour to a retention dial** (its first author is an invited speaker). Every citation from the scout's verified records; ⛔ nothing cited that the scout did not verify.
6. **⛔ The honest scope sentence, printed once and prominently:** our evidence is a small designed store — not a deployed agent memory, not an LLM system. State what class of claim this is (a mechanism with measured laws) and what it is not (a system result). The existing scale qualifiers and the score sentence stay verbatim.

## Style (strict — the Head's directive, tightened)
`.claude/PJ_Writing_Style_Context.md`, applied strictly and more directly than the source: ABT openings (abstract, §1, each results subsection) · macro-to-micro (policy question → mechanism → number) · **short declarative sentences; plain technical terms; one idea per sentence** · zero weasel words · signposting · "we" for our actions, passive for established facts · **no bold outside structural headers** · every number carries its scope in-sentence.

## Boundaries (absolute)
1. ⛔ **Approved wordings, mandatory riders, scope qualifiers and fine print: VERBATIM.** The `v5-referee-v02` do-not-cut list binds: N108's sentence · the exact-deletion form with its three conditions and the recency exclusion · Blelloch–Golovin at every deletion site · the lifecycle's two riders · the substrate-scope sentence · the score sentence · the designed-symmetry precondition · the fdt+Newtonian fine print · the emergent-arm caveats (⛔ no σ_θ ratio; the θ=π confound; the contrast number is designed-only). ⛔ **Where an operational term would widen a claim, our term stays.**
2. ⛔ Zero number changes, zero new claims, zero dropped findings. Two-way numeric-token check against the source build, printed.
3. ⛔ All sweeps (never-quote · internal-apparatus · semantic hermeticity), per-file, positive-controlled, printed.
4. Anonymization identical to the source build, including PALM's code-inclusive path neutralization.

## Shape
Main text **≤ 4 pp** (PALM short-track hard limit) · references and supplementary excluded · total **8–9 pp** target. Appendices keep only plots and results tables; the negatives table survives (their CFP names negative results as an accepted contribution type).

## Acceptance criteria
1. Main ≤ 4 pp; total 8–9 pp; page split printed.
2. Every operational term used is scout-backed or defined in one clause; every citation scout-verified; the honest scope sentence present exactly once.
3. The TTL comparison leads the leakage result, with N108's sentence verbatim beside it.
4. Numeric check + all sweeps printed; `papers/v5-short/**` byte-untouched (state the check).
