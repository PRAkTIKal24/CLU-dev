# Task: f5-arxiv-note — distill F5 into a standalone, citable preprint (critique P1/M1; Head-approved 2026-07-07)

- **Agent:** `physics-theorist` · **Output:** `.claude/outputs/f5-arxiv-note.md` (= the draft itself) + a coverage table (F5 item → note section → numerical check).
- **Read first:** protocol · `.claude/critique_register.md` (M1 + M2 — the citation architecture this note exists to solve) · `.claude/outputs/formalism-note.md` (F5 v1.1, the source) · `.claude/outputs/mo-deep-read.md` §4 (attribution obligations).
- **Why:** the three workshop shorts cannot cite each other under double-blind (M1). A single arXiv'd theory note becomes the one citable common ancestor. Target: draft for Head review ~Jul 20, arXiv early Aug (before shorts submission).

## Scope — what goes IN
The exactly-solved theory + its numerical checks, self-contained: the damped-Verlet single-mode solution; the flat-direction latch theorem + Noether-charge decay; GMOR spectral-mass law + retention n₁/₂ ∝ μ⁻² + the mass-independent floor + first-crossing/envelope metric distinction (Cor-14); critical-damping retention minimum (Cor-13) + exceptional-point signatures (Cor-15); kinetic-isotropy/Schur constraint; the two negative results (friction never stabilizes a saddle; governor blindness); position-gated volume accounting (Prop-11); discrete equivariant neutrality (Prop-16); Def-2 inertial-vs-spectral mass. Each with its App-N-grade numerical verification.

## Scope — what stays OUT
- Program/roadmap/vertical structure, wormholes, gate/calibration machinery, lattice results, erosion study — **the shorts keep their own empirical novelty**; this note is theory + sanity checks only.
- Anything from unpublished spoke experiments beyond the minimal numerical checks.

## Anonymization constraints (Head directive: minimize de-anonymization)
1. **NO "CLU" coinage** — that debut is reserved for the V2 short (Thread 6). Use neutral vocabulary throughout: "damped symplectic recurrences / Hamiltonian recurrent units."
2. Neutral, descriptive title (e.g., the register's "memory budgets of damped symplectic recurrences" register — final title = Head's call, propose 3).
3. Cite Jawahar & Pierini 2026 in **third person**, as *one instantiation* of the class — never "our previous work."
4. No program-distinctive branding as headline terms (in-text descriptive use of e.g. "budget of modes" is fine; avoid making our private vocabulary the paper's title/section names).
5. Author list + acknowledgments = **placeholder, Head decides.**

## Attribution duties (from mo-deep-read / f5-v11)
Mo 2026 (overdamped-face relationship, stated neutrally), Golubitsky/Krupa/Rumberger (equivariant-dynamics lineage), Minami–Hidaka, Welling et al. 2605.14685, Di Bernardo (geometry, scoped per the skim's guard-rail: never bare "we choose G/H").

## Format
Publication-grade **markdown** (GitHub-clean equations, colleague convention) — LaTeX conversion happens later via the paper-writer agent. Structure as a real paper: abstract, setup, results, checks, related work, limitations (scope: linear/exactly-solvable core + measured deviations on learned anharmonic vacua at stated %).
