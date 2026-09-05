# nips-v5-clean — paper-writer (V5's clean iteration base: `.claude/NIPSsubmission/v5-palm/`)

**Commissioned by the Shorts Advisor (charter: `.claude/advisor-head-shorts-charter.md`, Addendum 48; Head directive 2026-08-21).** Read `.claude/AGENT_PROTOCOL.md`, then this file.

**Output — a NEW folder `.claude/NIPSsubmission/v5-palm/`:** `submission.tex` · `submission.pdf` · `figs/` · `BUILD-NOTE.md`. ⛔ **`BUILD-NOTE.md` is deliverable #1** (the previous pass shipped PDFs before its note). ⛔ Every other paper folder is READ-ONLY; never run `pdflatex` outside your own output folder. `pdflatex` is at `/Library/TeX/texbin/pdflatex`.

**Source:** `.claude/papers/plain/v5/submission.tex` (the plain build).

**This folder is the base the Head will ITERATE on.** Content and framing first; the page count is explicitly not a target in this pass.

**DIAL DECLARATION: none — framing/editorial pass; zero number changes.**

## What carries over unchanged from the source (verify, do not redo)
No page-fitting typography (⚠ the source declared **five** `\small` exceptions on tables that physically overflow — keep exactly those, list them) · no bold outside structural headers · the author token absent everywhere (⚠ "Morse"/"Moser" survive) · the seven restored banked figures with provenance-bearing captions · single-seed figures labelled as such. **Verify each and state the check.**

## 1 — Audience scoping on 2025–2026 data (the new work; source: `outputs/audience-refresh-2025-2026.md`)
PALM is a **first edition**, so its room is characterised from the immediately preceding memory workshop (**MemAgents @ ICLR 2026**, 70 papers — ⭐ **Weiwen Liu keynotes both**), the 2025 adjacent workshops, and the organizers'/speakers' own output. Three facts change the framing:
- ⭐⭐ **This is white space for V5, and the paper should be positioned as filling it.** The proxy set contains **almost nothing on deletion or right-to-be-forgotten** — exactly V5's ground, and two of PALM's own CFP topics name it (*"memory update and deletion tests"*, *"right-to-be-forgotten mechanisms"*). Say what is missing from the current literature and what we supply, without overclaiming: we supply a **mechanism with measured laws at small scale**, not a deployed-system guarantee.
- ⭐ **The room now speaks physics and accepts theory.** The proxy set contains *"thermodynamic arbitration"* and *"entropic memory"*, and the CFP names *"theoretical perspectives"* and *"negative results"* as accepted contribution types. ⇒ ⛔ **Stop hedging the physics.** The retention law, the diffusion coefficient and the vault can be stated in their own terms rather than translated into policy language first — the policy framing stays, but it no longer needs to hide the mechanism.
- **The reviewer pool is now three-sided and every side maps to one of our results:** a membership-inference/privacy expert (⇒ the leakage section and its TTL comparison are read by a specialist — the honesty there is an asset), the author of the nearest published retention-dial neighbour (⇒ cite it explicitly and contrast), and **new: a brain–language-model alignment researcher** (⇒ the representational-drift bridge is now live in this room too, with the no-biological-claim sentence still mandatory, exactly once).

## 2 — Keep the operational discipline that already works
The source refused *"right-to-be-forgotten"* and *"memory provenance"* as terms, because each names a compliance property of a deployed system and we have a store-level result. ⭐ **That judgment stands and must not be relaxed by the white-space framing** — the CFP topic may be named as the topic we address; the property may not be claimed. Keep the adopted vocabulary (retention policy · retention dial · TTL/expiry · consolidation · scoping · deletion guarantee · membership adversary · stale entry) and the honest scope sentence, exactly once.

## 3 — Wording: simple, direct, strictly PJ
`.claude/PJ_Writing_Style_Context.md` applied strictly — ABT openings (abstract, §1, each results subsection) · macro-to-micro (policy question → mechanism → number) · **short declarative sentences, one idea each** · plain technical terms · zero weasel words · signposting · "we" for our actions, passive for established facts · `\texttt{}` for software/flags/files · italics sparingly. ⛔ **Simplify the prose, never the claim** — the `v5-referee-v02` do-not-cut list stays **verbatim**: the leakage sentence · the exact-deletion form with its three conditions and the recency exclusion · Blelloch–Golovin at every deletion site · the lifecycle's two riders · the substrate-scope sentence · the score sentence · the designed-symmetry precondition · the fdt+Newtonian fine print · the emergent-arm caveats (no σ_θ ratio; the θ=π confound; the contrast number designed-only).

## Boundaries
1. ⛔ Zero number changes; zero findings added or dropped; two-way numeric-token check against the source, printed.
2. ⛔ All sweeps (never-quote · internal-apparatus · semantic hermeticity), per-file, positive-controlled, printed; anonymization posture unchanged, including PALM's code-inclusive path neutralization.
3. ⛔ Page count reported but **not optimised**. ⚠ For the record in the build note: the source measured **5.00 pp of main text at default formatting**, against a venue limit of 4 — i.e. the earlier 4.00 pp was typographic. State the honest number; do not fix it in this pass.

## Acceptance criteria
1. `BUILD-NOTE.md` present and complete: page split (marked not-optimised, with the 5.00 pp fact stated) · the audience-scoping changes · the refused-terms verification · numeric check · sweeps · carried-over verifications · figure inventory.
2. The white-space framing is present and does not claim a system property; the no-biological-claim and honest-scope sentences each appear exactly once.
3. All source folders byte-untouched (state the check).
