# Task: v3-revision-4 — the composition problem (w15, writer)

- **Agent:** `paper-writer` · **Output:** `.claude/outputs/v3-revision-4.md`
- **Read first:** **`.claude/outputs/v3-referee-2.md`** (whole report — MF-2 and the five reviewer-reachable defects) · Charter · `.claude/claims_matrix.md` **v1.9** (CM-9 RESOLVED, CM-13) · `.claude/outputs/v3-interference-extra.md` · `.claude/outputs/scout-modular-interference.md` (the bib strings for the related-work hole).
- **Draft:** `.claude/papers/v3-short/draft.md` (v0.4). Rebuild the PDF.
- **Runs in parallel with** `v3-pricing-n-scaling` (analyst), which is producing the one experiment that would move this to accept. **Write a marked slot for its result; do not block on it.**

## Where we are
**MF-1 is CLOSED and the near-reject is retired.** The referee's words: *"more honest than most accepted workshop shorts."* The structural identity leads, the fitted slope is corroboration, the chain's degree-ramp residual is stated preemptively in its own main-text paragraph, and the block-monolith control is folded in with the capacity confound killed cleanly.

## MF-2 — the composition the paper never confronts
§3.2(iii) makes a concession the referee admired: *"nothing physics-specific buys the firewall — parameter separation does, and `block-untied` is a strictly better firewall than our lattice."* Meanwhile the surviving physics-specific claim is the **priced channel** (§3.3), measured on **2-unit** trained lattices, 3 seeds — and App C concedes that on trained lattices the `κ_eff` exponents are **inconclusive**, so *"the exponent authority remains the designed lattice"* (a C-2 **verification**, not evidence).

So: **a paper titled "Scaling a Conservative Memory Primitive" establishes its scaling result for a mechanism it explicitly disclaims, and its surviving physics result at N = 2.** Each half is honest; they sit hundreds of lines apart. **No reviewer will fail to compose them.**

This is not fixed by hedging. Fix it by **owning the composition in the main text** and making the paper's actual thesis survive it. Two moves, and you may need both:
1. **Say what the firewall result *is*.** It is a *structural* guarantee about parameter separation that the lattice **realizes by construction and a monolith cannot** — and the honest comparison is that `block-untied` achieves it too, at the cost of having no shared Hamiltonian and therefore no pricing law, no reversibility, and no conserved quantities. **Modularity is the firewall; the physics is what you get to keep *while* having it.** If that is the paper's claim, put it in §1 and let §3.2(iii) be its evidence rather than its embarrassment.
2. **Scope the title and the abstract to what is established.** If the pricing law is `N=2`-verified and `N>2` is pending, the abstract must not let "scaling" attach to it. `v3-pricing-n-scaling` may close this within the wave — leave a marked slot.

## The five reviewer-reachable defects (three are M4 cross-section drift)
Take these verbatim from `v3-referee-2` §"Itemized" and close each:
1. a **main-text bound falsified by the paper's own appendix**;
2. a **§3.3 table from which none of its own five residuals can be recomputed** (publish the quantities that make them recomputable, or drop the residual column);
3. a **single-unit result attributed to "the lattice" in the abstract**;
4. the **related-work hole** at exactly the place the nearest architectural cousin lives — **the strings already exist in `scout-modular-interference.md`**; splice them, do not re-scout;
5. the remaining item as the referee states it.

## Also fold
- **CM-13 scope must ride with the 946× reversibility claim** wherever it appears: **exact only at γ=0** (γ>0 gives a finite horizon `(1−γ)^−n`), and it is **gradient mechanics only — not wired into `train_chlu`**.
- **MF-1 (theory note = `Anonymous (2026)`)** is a standing Head dependency across all three shorts. Mention once; do not itemize.
- **SF-4** (six contributions over-budget) remains deferred to the **C-10 pruning pass** once page limits are known. Do not prune now.

**Acceptance:** MF-2 confronted in the main text with a thesis that survives the composition; the five defects closed; related-work spliced from the existing scout strings; CM-13 scoped at every site; the `v3-pricing-n-scaling` slot marked and fillable; PDF clean. **The referee says none of this needs a new experiment to reach submission — so do not ask for one.**
