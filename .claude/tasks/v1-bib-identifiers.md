# V1 — the identifier block for the bibliography

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-27.** The Head pastes the result into Zotero's identifier tool to generate `refs.bib`, which a later cite pass wires in.

**Agent:** `web-scout` — **read-only. ⛔ Edits no file in `NIPSsubmission/` or `papers/`.**
**Output:** `.claude/outputs/v1-bib-identifiers.md`
⚠ `web-scout` has **no shell**. If you need a hash or a file measurement, say so plainly rather than faking one — a prior scout in this program did exactly that and it was the right call.

---

## 0. ⛔⛔ THE ONE RULE THAT MATTERS: ZERO GUESSED IDENTIFIERS

**An invented identifier silently yields a wrong `.bib` entry the Head cannot catch.** Every identifier you return must be **verified against a primary source at the moment of use** — never recalled, never inferred from an author name, never carried from a neighbouring line.

**Three failure modes this program has already been bitten by. Read them before you start.**

1. ⛔⛔ **Crossref never says "no result" — it always returns its best match.** A deliberately nonsensical query once returned a confident, plausible, fully-formed record (*"Saint Katherine of Alexandria"*, DOI and all). ⇒ **Every DOI must be eyeball-matched against the title Crossref returns.** One first-pass hit in a sibling pass was caught this way — a lookup returned the **bioRxiv preprint** DOI rather than the journal record the paper cites.
2. ⛔⛔ **Author-surname matching produces wrong works.** The Advisor tried to reuse banked records for this very list and **two of four matches were the wrong work**: `Graves → arXiv:1410.5401` is *Neural Turing Machines*, **not** *Adaptive Computation Time*; and `Platt → arXiv:2202.02164` is **a different person** (Aslan, Platt & Sheard 2023, PMLR v197). ⇒ **Match on TITLE, never on author alone.** Graves, Platt, Neal and Roberts are all common surnames with multiple relevant works.
3. ⚠ **The `http://` arXiv endpoint 301-redirects to nothing.** Use **HTTPS**. A sibling pass reported "none of these IDs resolve" from that alone, and only a positive control caught it.

⭐ **Positive-control your negatives.** Before reporting that a work has no DOI, confirm your method finds a DOI for a work you know has one.

---

## 1. The 17 works — the complete list, read off the paper

⭐ **Five already have identifiers the Advisor verified. CARRY these; do not re-derive them, but DO confirm each title matches:**

| # | work | identifier | provenance |
|---|---|---|---|
| 1 | Gladstone et al. (2025), *Energy-Based Transformers are Scalable Learners and Thinkers* | `2507.02092` | read out of the paper's own text |
| 2 | Graves (2016), *Adaptive Computation Time for Recurrent Neural Networks* | `1603.08983` | read out of the paper's own text |
| 3 | Raposo et al. (2024), *Mixture-of-Depths* | `2404.02258` | read out of the paper's own text |
| 4 | Ramsauer et al. (2021), *Hopfield Networks is All You Need*, ICLR | `2008.02217` | banked record, author list confirmed |
| 5 | Jawahar & Pierini (2026), *CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning* | `2603.01768` | banked record, title + authors confirmed |

**⛔ These eleven need identifiers and are your job:**

| # | work as the paper cites it | note |
|---|---|---|
| 6 | Angelopoulos, Bates et al. (2021), *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control* | arXiv likely; check for a later venue |
| 7 | Banino, Balaguer, Blundell (2021), *PonderNet: Learning to Ponder* | ICML workshop / arXiv |
| 8 | Duane, Kennedy, Pendleton, Roweth (1987), *Hybrid Monte Carlo*, **Physics Letters B** | ⭐ journal — DOI expected |
| 9 | Geifman & El-Yaniv (2017), *Selective Classification for Deep Neural Networks*, NeurIPS | NeurIPS proceedings; DOI may not exist |
| 10 | Lieb & Robinson (1972), *The Finite Group Velocity of Quantum Spin Systems*, **Comm. Math. Phys.** | ⭐ journal — DOI expected |
| 11 | Neal (2011), *MCMC using Hamiltonian dynamics*, **Handbook of MCMC** | ⚠ book chapter; ⛔ do not confuse with Neal's other MCMC works |
| 12 | Platt (1999), *Probabilistic Outputs for Support Vector Machines* | ⚠ **MSR tech report — a DOI may not exist.** ⛔ Do NOT return the Aslan/Platt/Sheard PMLR record |
| 13 | Roberts & Tweedie (1996), *Exponential Convergence of Langevin Distributions and Their Discrete Approximations*, **Bernoulli** | ⭐ journal — DOI expected |
| 14 | Schuster et al. (2022), *Confident Adaptive Language Modeling*, NeurIPS | arXiv likely |
| 15 | Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR | arXiv likely |
| 16 | Wales & Doye (1997), *Global Optimization by Basin-Hopping*, **J. Phys. Chem. A** | ⭐ journal — DOI expected |

**#17 — `Anonymous (2026)`, the theory note.** ⛔ **No identifier exists and none may be invented.** Report it as a Head decision, not a lookup. *(Context: a sibling short dropped its equivalent entry entirely, because an uncited hand-built entry still prints and points at a document the reviewer cannot open. V1's case differs — an audit found one load-bearing dependency, which a commissioned proofs appendix is meant to close.)*

⭐ **Reuse before re-verify**, with failure mode 2 live: `.claude/outputs/v2-cite-check.md` holds 26 primary-source-verified records and `v5-scope-scout.md` ~30 more. ⛔ **Any record you carry from them must be title-matched to the work above, not author-matched.**

---

## 2. ⛔⛔ THE CITE-OR-CUT DECISION, AND WHY IT IS URGENT

**Six of the seventeen are cited nowhere in the paper's prose:** **Duane · Geifman · Lieb · Neal · Roberts · Wales.**

⚠ **They are NOT decorative.** Advisor-verified: every one has a live conceptual home in the text — `Monte Carlo` ×2, `MALA` ×1, `Langevin` ×5, `basin-hopping` ×1, `light-cone` ×1, `selective classification` ×1. **Their author names were stripped by the Head's condensation; the ideas they support are still being discussed.**

⛔⛔ **THE TRAP, and it is why this must be ruled before the `.bib` is built.** The paper's reference list is currently **hand-built** (`\item` entries), so an uncited entry **still prints**. The cite pass will convert it to `\bibliography{}` — at which point **any entry without a `\cite` silently disappears.** ⇒ **If these six are not attached to a citation site first, six legitimate references vanish from the paper and nobody sees it happen.**

**Deliverable: for each of the six, name the exact sentence that should carry the citation**, quoted, with its section. ⛔ Recommend only — the Head rules attach-or-cut.

---

## 3. Deliverables

1. ⭐ **THE BLOCK** — one identifier per line, ready to paste into Zotero. **DOI preferred and bare** (`10.xxxx/yyyy`, no `https://doi.org/` prefix), **arXiv ID as fallback** (`NNNN.NNNNN`, no `arXiv:` prefix). ⛔ Zero guessed identifiers. ⚠ **The Head has asked for ONE combined block, not two** — DOIs and arXiv IDs together, in the order of §1.
2. **The coverage table** — every work marked **CARRIED** (from §1's five or a banked record, title-confirmed) or **NEW** (you resolved it), with the source you verified against and the returned title quoted so the match is auditable.
3. **The residual list** — any work with no resolvable identifier, and the exact fields the Head must hand-enter. ⛔ Better an honest residual than a plausible wrong ID.
4. ⭐ **The cite-or-cut list** (§2) — the six orphans, each with its proposed citation site quoted.
5. **Never-copy traps** — any near-miss you hit, recorded so the next pass does not repeat it.

## 4. Acceptance criteria

- Every returned identifier is title-matched against a primary source, and the returned title is **quoted in the report**.
- Every DOI is **re-resolved** (a fabricated DOI as negative control, reported).
- ⛔ No file under `NIPSsubmission/` or `papers/` is modified.
- The six orphans each carry a proposed site or an explicit cut recommendation.

## 5. ⚠ Notes

- ⛔ **C-8 hermetic:** do not read the sibling shorts' drafts. Their *cite-check outputs* are fair game as verified records (§1), the drafts are not.
- The paper is **double-blind**. ⛔ If you meet an author-identifying artifact, report it; do not propagate it.
- ⚠ A sibling pass found the exporter silently **dropped arXiv-only entries** and imported **arXiv posting years for venue-published works**. Where a work has both a preprint and a venue record, **say which the paper cites** so the year does not silently change.

## DIAL DECLARATION
**Dials touched: NONE.** Read-only literature lookup producing one report. No experiment, no configuration change, no paper file edited.
