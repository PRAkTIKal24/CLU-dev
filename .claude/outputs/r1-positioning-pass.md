# r1-positioning-pass — paper-writer report

Task + acceptance criterion: reposition every R1 claim sentence in **existing** deletion vocabulary (Ginart Def. A.5 · Sekhari Def. 3), sweep `.claude/papers/**` for banned terms, draft the `research_roadmap.md` Part-4 replacement string, and deliver both scope wordings. **No new evidence is generated or claimed here.**
Status: **done**, with two citation corrections and one PENDING (§4B, as instructed).

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). SIX items.**
> 1. ⭐⭐ **CITATION DEFECT, program-wide: "Guo et al., ICML 2020, Def. 1 / Def. 2" DOES NOT EXIST.** I pulled the arXiv LaTeX source (`arxiv.org/e-print/1911.03030`) and the paper contains **no `definition` environment at all** (`\newtheorem` declares only lemma/theorem/corollary/appendix_theorem). ε-certified removal is an **inline definition with a numbered equation, §3 Eq. (1)**; (ε,δ)-certified removal is the **unnumbered displayed pair immediately after it**. Everywhere we currently write "Guo Def. 1–2" (claims_matrix **CM-22(m)**, `deletion-prior-art` §Group-4 + its BibTeX `note`, `unlearning-recon` if it inherits) must become **"Guo et al. (2020), §3, Eq. (1) and the (ε,δ) relaxation following it."** Owner: **doc-curator** (CM-22(m) + the bibtex note). ⚠ Verified against the arXiv source only; the PMLR camera-ready was not opened (§7).
> 2. ✅ **Sekhari Def. 3 CONFIRMED as Definition 3, verbatim, and re-typeset (§1.2).** Counted in source order: Def. 1 = sample complexity of learning, Def. 2 = (ε,δ)-unlearning, **Def. 3 = deletion capacity**. `Thm`-side bound `m ≥ c·n√ε/(d log(1/δ))^{1/4}` confirmed verbatim in `files/sec_unlearning_algorithms.tex`. No action.
> 3. ✅ **Ginart Def. A.5 CONFIRMED and re-typeset (§1.1).** It is the 5th `definition` env under `\appendix\section{Supplementary Materials}` ⇒ **A.5**, and reads *"we say an algorithm A is α-deletion efficient if it runs Algorithm 3 in amortized time O(n^{1−α})"* under `m = Θ(n^α)`. No action.
> 4. ⛔ **A Ginart-Def-A.5 cost claim is NOT currently supportable and must not be written** — the algorithmic cascade is ~3–4 survivor moves/delete at full lattice load, but the **shipped `_canonical_sync` is a full O(n) `with_item` rebuild per operation** (`placement-landing` §1 design note + §Open-questions 3). Def. A.5 is about **amortized time**, so quoting α-deletion efficiency today would be quoting an algorithm we have not implemented. **Missing experiment (§8, item M1): amortized wall-clock delete cost vs n.** Owner: Hub → experiment-engineer.
> 5. ⚠ **`.claude/papers/**` is CLEAN of R1 vocabulary** — zero hits for "unlearning", "deletion-compliant", "exact deletion", "fix-up cascade", "evict", membership/privacy language, in every draft. **Exactly one must-fix hit existed** (`iclr-long/outline.md` L10, *certified* × *lifetimes*) and I fixed it (§2a). Everything else labelled "certified" in the drafts is **symplectic/BIBO/LTT certificate language** — a different technical sense — and there is a **live contradiction between three governing documents** about whether that is banned (§2b). **Head ruling requested.** Owner: Head, relayed by Hub.
> 6. ⚠ **claims_matrix CM-22(q) and CM-22(r) are now stale.** (q) blocked *every* deletion-flavoured sentence pending N99's three blockers — all three cleared in w26 (`placement-landing` landed `Controller.delete`; `deletion-prior-art` verified the prior art; the acceptance test drove `AUC(z_hole)` → 0.5000 ± 0.0000). (r) blocked "0.953" pending the real two-phase Verlet read — measured, 0.9531. The §4 wordings below are written **on the assumption the curator unblocks them at this wave's review**; until then they are drafted-but-not-cleared. Owner: **doc-curator**.

---

## ⭐ DIAL DECLARATION (echo, protocol §7)
- **Dial:** none — **positioning / wording.** No measurement.
- **Laundering control:** every R1 sentence below names the **flat-datastore row-delete** as the trivial substitute and claims only to *match* it. Executed: the control clause is baked into S1/S2 (§4) and into the vocabulary paragraph (§1.4); no sentence I wrote claims to beat a dict on a deletion-exactness axis.
- **Falsifies:** any sentence surviving only because a scope qualifier was dropped. **Self-check executed sentence-by-sentence in §4c** (the qualifier-audit table).
- **Does NOT falsify:** our Theorems 1–2 being instantiations of Blelloch–Golovin — that is the finding, and §5 states the defensible novelty.

---

## 1. The adopted-vocabulary paragraphs (paste-ready; BibTeX keys reused from `deletion-prior-art`)

### 1.1 Ginart Def. A.5, re-typeset from primary (`arxiv.org/e-print/1907.05012`, `neurips_2019.tex` L613–620)

Verbatim from source (the definition body; the surrounding fractional-power-regime setup is L609):

> **Definition A.5 (Deletion Efficient Learning Algorithm).** *Recall the Ω(n/m) lower bound on amortized computation for any sequential learning algorithm in the online deletion setting (Section 2). Given some fractional power scaling m = Θ(n^α), we say an algorithm A is **α-deletion efficient** if it runs Algorithm 3 in amortized time O(n^{1−α}).*

Companion facts, also verbatim from source, that must travel if we cite it: the Ω(n/m) bound is stated as a **Remark** in §3 (*"for n datapoints and m deletion requests we establish an asymptotic lower bound of Ω(n/m) for the amortized computation time of any (sequential) learning algorithm"*), and Ginart's own exactness notion (§3, the main-text `Data Deletion Operation` definition) is **equality in distribution**: `A(D_{−i}) =_d R_A(D, A(D), i)`.

### 1.2 Sekhari Def. 3, re-typeset from primary (`arxiv.org/e-print/2103.03279`, `files/definitions.tex` L81–90)

> **Definition 3 (Deletion capacity).** *Let ε, δ ≥ 0. Let S be a dataset of size n drawn i.i.d. from D, and let f(w, z) be a loss function. For a pair of learning and unlearning algorithms A, Ā that are (ε,δ)-unlearning, the deletion capacity `m^{A,Ā}_{ε,δ}(d, n)` is defined as the maximum number of samples U that can be unlearnt, while still ensuring an excess population risk of 0.01. Specifically,*
>
> ```
> m^{A,Ā}_{ε,δ}(d,n) := max { m : E[ max_{U ⊆ S, |U| ≤ m}  F(Ā(U, A(S), T(S))) − F* ] ≤ 0.01 }
> ```
> *where the expectation above is with respect to S ∼ D^n and output of the algorithms A and Ā.*

Its precondition, **Definition 2** (same file, L60–71), is the (ε,δ) two-sided likelihood-ratio condition between `Ā(U, A(S), T(S))` and `Ā(∅, A(S∖U), T(S∖U))`; `T(S)` is the retained-statistics budget, which Sekhari explicitly requires **not to grow with n** (*"this precludes strategies which involve storing and reusing the entire training set, or aggressive model checkpointing"*). Their capacity bound (`files/sec_unlearning_algorithms.tex` L76): `m ≥ c · n√ε / (d log(1/δ))^{1/4}`.

### 1.3 ⭐ We adopt Sekhari's FORM, and here is why the store needs an instantiation, not the literal definition

*(The task said "adopt it, or state explicitly why the store setting needs a different one." Answer: adopt the form, state three instantiation changes. This is the paste-ready justification paragraph.)*

> We report deletion cost and utility on the two axes the literature already owns, rather than naming a new benchmark. On the cost axis we use Ginart et al.'s deletion efficiency [ginart2019forget, Def. A.5]: against the Ω(n/m) amortized lower bound for any sequential learner, an algorithm is α-deletion efficient if a stream of m = Θ(n^α) deletions costs O(n^{1−α}) amortized. On the utility axis we use the form of Sekhari et al.'s deletion capacity [sekhari2021remember, Def. 3] — *the largest number of deletions a system absorbs at a fixed utility budget* — which is the closest existing formalisation of what we want to report. We instantiate rather than apply it literally, and we state the three substitutions. **(i)** Def. 3 is stated for a pair of (ε,δ)-unlearning algorithms; our object is a store, whose deletion is exact at the byte level within the scope stated below, so the (ε,δ) precondition degenerates (ε = δ = 0 at the store layer) and the capacity question becomes one about *retrieval*, not about a likelihood ratio. **(ii)** Def. 3's utility is the **excess population risk** `F(·) − F*` of a learned predictor at a fixed budget of 0.01; a store has no population risk, so we substitute the store's own utility — per-offered retrieval accuracy on the live set, reported together with per-admitted accuracy, since the two are different metrics and a single number is not interpretable. **(iii)** Def. 3's budget `T(S)` (retained statistics that must not grow with n) has a direct analogue we report: the store *is* the retained statistic, and its size is the design parameter. With those substitutions the quantity we report is Sekhari's: **how many items the store holds and returns, at a deletion cost we state.** We do not claim (ε,δ)-unlearning in the sense of [sekhari2021remember, Def. 2] or (ε,δ)-certified removal in the sense of [guo2020certified, §3, Eq. (1)], and we supply no such parameters.

### 1.4 ⭐ The trivial substitute, stated first (mandatory laundering clause — paste-ready)

> **The comparison we are matching, not beating.** A flat (brute-force) datastore deletes a row exactly, by construction, in O(1); this is the operation a nonparametric opt-out already provides [min2024silo], and it is the control that accompanies every number below. Our claim is only that a *physical* store — one whose contents are continuously decaying amplitudes in an energy landscape, read by a dynamical relaxation rather than by lookup — can be given the same claim structure as `del d[k]` on a canonical dictionary, within the scope stated. Nothing here beats a dictionary at being a dictionary. *(The "flat" qualifier is load-bearing rather than pedantic: graph-based ANN indexes delete lazily, marking a deletion vector and deferring graph repair to a batch consolidation pass [singh2021freshdiskann], and a recent preprint reports that soft-deleted embeddings in three HNSW implementations remain recoverable from the raw index files at the storage layer [chakraborttii2026ghost]. We flag both as context and rely on neither; the second is a six-week-old, non-peer-reviewed preprint and is cited as "a recent preprint reports".)*

### 1.5 The practice precedent + the multi-dimensional convention (paste-ready)

> Reporting the two axes jointly is established practice rather than a new protocol: SISA's sharding study is a deletion-cost-versus-utility Pareto curve in all but name — 4.63× / 2.45× / 1.36× retraining speedups on Purchase / SVHN / ImageNet, paid for in accuracy (up to 16.14 pp top-5 degradation on ImageNet at standard sharding) [bourtoule2021sisa, §VII] — and evaluation suites already treat cost-and-utility as several dimensions rather than one scalar: MUSE scores, among six criteria, (4) utility preservation on retained data, (5) scalability in the size of the removal request and (6) sustainability over sequential requests [shi2024muse], and CURE4Rec defines completeness, utility, efficiency and fairness [chen2024cure4rec]. We adopt this convention and deliberately do not name a benchmark.

### 1.6 ⭐ The prior-art attribution paragraph (the scout's approved §1.6 wording, with ONE flagged amendment)

Reproduced from `deletion-prior-art` §1.6, which the Hub approved for verbatim paper use. **I changed exactly one clause and flag it rather than doing it silently:** the final sentence's *"not a claim … about certified (ε,δ) unlearning"* uses two program-banned words even in denial, and the task's §2 sweep bans them without a use/mention exemption. Replacement clause in **bold**; everything else is the scout's text unaltered.

> *Order-independent placement is not new: strongly history-independent (uniquely represented) data structures were introduced by Micciancio (STOC'97) and Naor & Teague (STOC'01), characterised as canonical representations by Hartline et al. (Algorithmica 2005), and realised for open-addressed hash tables by Blelloch & Golovin (FOCS'07), whose table is a stable matching between keys and slots under a global key priority — the same priority-greedy rule and the same delete-time fix-up cascade we use here. Our placement rule is an instance of that framework, obtained by replacing the hash probe sequence with a probe order induced by distance in the store's metric, and we claim no new result about unique representation per se. What is new is the composition: the slots are a lattice packing, so the canonical representation carries a minimum-separation certificate and hence a quantitative interference bound; the stored content is not a static key but a continuously decaying amplitude, and we prove that deletion commutes with the decay flow, so that a deleted item's survivors are bit-identical to a history in which it was never written, at every point of their schedules; the canonical object is therefore not a memory layout but an energy function, which a dynamical read relaxes into; and, unlike the general case, where strong history independence is known to cost as much as an exponential slowdown (Buchbinder & Petrank, CRYPTO'03), here it is free — the designed lattice packs strictly better than the stochastic relocation rule it replaces. This is a store-level structural property,* ***not a claim about the trained encoder, and not a claim of (ε,δ)-indistinguishability from retraining in the sense of Guo et al. (2020).***

⚠ **Hartline reversibility caveat — mandatory footnote wherever that paragraph appears** (paste-ready):

> Hartline et al.'s characterisation (their Theorem 1: a *reversible* data structure is strongly history independent only if a canonical representation for each state is fixed at initialisation) requires a strongly connected state-transition graph. Our **placement layer** is reversible; our **amplitude layer is not** — amplitudes are monotone in elapsed time and `born` is monotone, so a state once left is unreachable. We therefore cite Hartline et al. only for the definitional equivalence between the Naor–Teague and canonical-representation formulations (exactly as Blelloch & Golovin do), never inside a proof; the canonicity of placement is a direct construction and does not need it.

⚠ **Do not cite a Naor–Teague definition number** — the scout could not obtain the STOC'01 primary (ePrint bot-check 403, `.ps` HTTP 422, ACM DL paywalled). Cite N&T for the *notions*; cite Hartline/BG07 for numbered definitions.

---

## 2. The banned-terms sweep — file:line table

**Sweep executed over** `.claude/papers/**` (`*.md`, `*.tex`, incl. CHANGELOGs; the `* 2/` Finder-duplicate directories are empty and were excluded) **for:** `certif*` · `unlearn*` · `deletion-compliant` · `exact deletion` · `fix-up|fixup cascade` · `evict*` · `distinguishab*` · `membership` · `privacy|GDPR|right to be forgotten` · `tombstone` · `delete|deletion|erasur*`.

### 2a. MUST-FIX hits (found: 1; fixed: 1)

| file:line | offending string | verdict | replacement (applied) |
|---|---|---|---|
| `iclr-long/outline.md:10` | `G7c "Pareto-not-podium: physically-motivated, **certified, predictable-lifetime** memory, Pareto-competitive"` | ⛔ **CM-22(m) exactly** — *certified* applied to *lifetimes* is the named forbidden construction, and it is the ICLR thesis fallback, i.e. lead position | ✅ **APPLIED:** `"Pareto-not-podium: a physically-motivated memory with **set-at-write-time lifetimes and an exact decay law**, Pareto-competitive"` + an inline `[⛔ …CM-22(m)…]` marker so it cannot silently regress. Logged in the new `iclr-long/CHANGELOG.md` |

### 2b. ⚠ CONTESTED hits — certificate language in the physics sense (found: **112 lines across 10 files** — line counts, not occurrence counts; **NOT edited**, Head ruling requested)

| file | lines with `certif*` | sense in context | verdict |
|---|---|---|---|
| `v1-short/draft.md` | 13, 21, 23, 29, 31, 33, 39, 41, 47, 62, 68, 70, 72, 74, 93, 97, 101, 103, 138, 140, 170, 172, 176, 177, 178, 180, 182, 185, 187, 189, 197, 259, 265, 283, 299, 308, 318, 319, 320, 324, 345, 456, 478, 484, 486, 496, 500, 502, 506, 510, 514 (51) | paid-access **certificate stack**: `det J = 1` symplectic receipts, bounded energy injection `≤ e^{2\|ζ\|}H`, BIBO coercive-component screen, LTT distribution-free **coverage** certificates (30/30 cells), "certified Markov kernel" | **no deletion/lifetime sense anywhere**; C-6 explicitly sanctions *"certified within [stated scope]"* and V1's fine print is already inline |
| `v1-short/draft.tex` | 26, 32, 34, 39, 40, 41, 45, 63, 68, 72, 76, 78, 102, 105, 107, 141, 143, 168, 175, 176, 177, 180, 182, 185, 187, 189, 193, 195, 211, 222, 235, 236, 237, 242, 247, 256, 261, 264, 266, 270, 272, 274 (42) | LaTeX mirror of the above | same |
| `v2-short/draft.md` | 310 | *"G.3 Reach rungs (certificates only; falsifiable a/c)"* — analytic squeeze/wormhole receipts | same |
| `v2-short/draft.tex` | 338 | same | same |
| `v3-short/draft.md` | 171, 195, 219, 337, 410 | *"per our certificate discipline"*, matched-quadratic-H **certificate**, **Certificate altitude** paragraph | same; L219 is the C-6 fine-print paragraph itself |
| `v3-short/draft.tex` | 221, 227, 327 | same | same |
| `iclr-long/outline.md` | 9, 18, 28 | *"certified test-time compute"* (V1 paid-access), *"Certificate derivations"* appendix | same |
| `v1-short/CHANGELOG.md` | 3–6 · `v2-short/CHANGELOG.md` 11 · `v3-short/CHANGELOG.md` 4 | revision log entries naming the certificate sections | historical log; editing would falsify the record |

⭐ **The contradiction that needs a Head ruling (stated plainly, not smoothed over).** Three governing documents disagree:
- `deletion-prior-art` §Group-4 says *"the word stays banned **program-wide**"*;
- claims_matrix **CM-22(m)** bans it specifically *"for CLU's decay/eviction"* — i.e. scoped to lifetimes/removal;
- Positioning Charter **C-6** *requires* the form *"certified within [stated scope]"* and V1's entire §3 is built on it (a conformal-prediction / LTT coverage certificate **is** a certificate in the standard statistical sense, and a `det J = 1` receipt is a proof, not a claim about a learning algorithm's output distribution).

**My recommendation (not applied):** narrow the ban to its referent — *"certified"* is forbidden **for any statement about an item's lifetime, its removal, or an adversary's ability to detect it**, and permitted for symplectic/BIBO/LTT receipts **provided the fine print is adjacent (C-6)**. If the Head instead wants the word gone program-wide, the mechanical substitutions are: `certified X` → `receipted X` / `X with a stated-scope certificate`; `certificate stack` → `receipt stack`; `certified Markov kernel` → `receipt-carrying Markov kernel` — **a 112-line edit across four papers (+3 CHANGELOGs) that I did not perform on my own authority**, and which would need a fresh V1 pass because the word is structural to that paper's contribution list.

### 2c. Terms with ZERO hits anywhere in `.claude/papers/**` (verified, not assumed)

`unlearn*` **0** · `deletion-compliant` **0** · `exact deletion` **0** · `fix-up cascade` / `fixup cascade` **0** · `our fix-up cascade` **0** · `evict*` **0** · `tombstone` **0** · `GDPR` / `right to be forgotten` **0** · any claim that decay reduces distinguishability **0** · any claim that eviction removes the item **0**.
The only `membership` hits (`v1-short` L13/74/317/455, `.tex` L26/78/234) are *"coercive-**component** membership"*, and the only `distinguishab*` hits (`v3-short` L15/179/185/209, `.tex` L27/203/210/224, `v2-short` L411, `.tex` L432, `v5-short` L181) are *"float-indistinguishable gradients"* / *"training-indistinguishability"*. **No false-positive was counted as a hit and no true hit was reclassified as a false positive.**

### 2d. ⚠ Adjacent-vocabulary collision to watch when R1 is drafted (no action now)

`v5-short/draft.md` L97/135/149/199/226, `v5-short/draft.tex` L281/510/513 and `f5-note/f5-note.tex` L173 use **"erasure"/"erase"/"delete"** in the *physics* sense (T = 0 erasure of coset content, the friction-hole vault, the unbuilt `T_φ` "shredder", *"a friction-only mechanism … cannot delete an exactly-flat coset coordinate"*). These are correct in their own paper and must not be edited — but a reviewer reading R1 **and** V5 will see two different senses of "erase". **Drafting note for R1:** use **"deletion / removal"** for the store-level record operation and never "erasure", reserving "erasure" for the V5 physics sense; add one disambiguating sentence if both appear in the ICLR long.

---

## 3. The `research_roadmap.md` Part-4 replacement string (⛔ Head approval required; the curator applies it — I did not edit the doc)

**Location:** `.claude/research_roadmap.md` **line 110**, the **R1 row** of *"⭐ THE RESULT SET (Head, Part 4) — five results, each owning a dial"*.

**CURRENT (triply forbidden — `certified` × 2, and *"deletion as a certified physical operation"* additionally asserts an unqualified exactness we only hold in a stated scope):**

```
| **R1** | **Memory with a dial: certified per-item lifetimes.** Half-lives set at write time and measured to match the physics across orders of magnitude; permanent + scheduled-fade in one store; **deletion as a certified physical operation**, not an approximate fine-tune. Lands on **machine unlearning / right-to-be-forgotten**. | **2 — lifetimes** | the Part III retention law (`exp(−leak·t)`, half-life ↔ μ²/γ), measured exact | **Closest to done** — `controller-mvp` demonstrated the machinery; missing = packaging against the unlearning literature's benchmarks. ⚠ **`unlearning-recon` (w24) is testing whether "mostly framing" is true** — what *"certified"* formally requires, and whether "deletion by construction" is already owned (a kNN datastore also deletes exactly). |
```

**REPLACEMENT — exact string, cell 2 (the deliverable):**

```
**Memory with a dial: set-at-write-time item lifetimes, and a store whose deletion is exact at the store layer.** Half-lives fixed at write time and measured to match the physics across orders of magnitude; permanent and scheduled-fade items coexisting in one store; and — for stores operating below capacity or under set-function (priority/attribute-based) eviction — removal that reproduces, bit for bit, the store holding exactly the remaining records, with survivors' schedules unaffected (deletion and decay commute). The comparison is a flat datastore's row delete, which is exact by construction: we match its claim structure in a physical store, we do not beat it. The placement rule that buys this is an instance of Blelloch & Golovin's strongly-history-independent table (FOCS'07), not a new result; the contribution is the composition around it (packing certificate · decaying content with a commutation proof · a canonical energy function · a negative price of strong history independence). Lands on the data-deletion literature (SISA, SILO, Ticketed L–U, PALL) as a **store-level** result: the frozen encoder and any residue in a learned landscape are separate channels, measured separately.
```

**REPLACEMENT — exact string, cell 5 (status; OPTIONAL, offered because the current cell is stale and also contains the banned word in a live-question framing):**

```
**Machinery landed and measured (w26).** `Controller.delete` + canonical placement ship; on the paired-world membership harness the post-deletion history column falls to AUC 0.5000 ± 0.0000 on every statistic (TPR 0.000 @ FPR 1 %), byte-equal to never-written in 3 072/3 072 worlds, at 8 offers into a store sized below capacity. ⛔ **Not exact at overflow** (8 offers, 7 cells: AUC(n_live) = 1.000) — the waitlist is the open build. Missing for the paper: the amortized cost measurement (Ginart Def. A.5 is about amortized time; the shipped re-pack is O(n)), and the deletion-cost-vs-utility table with the flat-datastore control in it. **`deletion-prior-art` (w26) settled the prior-art question:** the discrete skeleton is owned outright by Blelloch–Golovin (FOCS'07); "deletion by construction" is occupied (SILO/SISA/Ticketed L–U/PALL); the four-part composition is not.
```

**Judgement call flagged for the Head (do not let this pass silently):** I kept a *field-name* pointer to the deletion literature but wrote it as **"the data-deletion literature (SISA, SILO, Ticketed L–U, PALL)"** rather than *"machine unlearning / right-to-be-forgotten"*. Naming the field is not naming our mechanism, so the ban arguably permits *"machine unlearning"* as a venue pointer — but *"right-to-be-forgotten"* is the legal framing Garg–Goldwasser–Vasudevan formalised as **deletion-compliance**, whose dependency clause we **fail** (the encoder φ saw the item), so pointing at it invites exactly the test we lose. **If the Head wants the venue named explicitly, the safe form is "lands on the machine-unlearning literature" — and never "right-to-be-forgotten".**

---

## 4. The scope qualifier — both wordings, per §4 of the task

### 4A. CURRENT (scoped) wordings — landed, measured, quotable **once the curator lifts CM-22(q)/(r)**

**S1 — the deletion claim (the load-bearing sentence).**
> *"Placement in the store is canonical — a deterministic function of the live item records and the store geometry alone — so **store-level** deletion is exact **for stores operating below capacity or under set-function (priority/attribute-based) eviction**: removing an item reproduces, bit for bit, the store that holds exactly the remaining records, with each survivor's scheduled decay and permanence unaffected, because deletion and decay commute. Recency-based (LRU/staleness) eviction is intrinsically history-dependent and is excluded by construction. A flat datastore's row delete is exact by construction and is the control throughout: the claim is that a physical store **matches** that claim structure, not that it improves on it."*

**S2 — the measured membership result (the acceptance number, STATED AT A LOAD).**
> *"On a paired-world membership-inference harness (3 seeds × 8 targets × 128 paired worlds = 3 072 worlds per arm), **at 8 offers into a capacity-8 store sized to 13 lattice cells** — i.e. below capacity — the post-deletion history column is **AUC 0.5000 ± 0.0000 on all six statistics** (distance-to-nearest-live-site, live count, two query-side statistics and two white-box statistics), with **TPR 0.000 at FPR 1 %**, and the store after deletion is **byte-equal to the never-written store in 3 072/3 072 worlds** — including when the deletion fires a survivor displacement cascade (target keyed to the highest priority: mean 1.132 moves, max 5, still byte-equal in 3 072/3 072). The statistics are tied because the two stores are identical, not because the adversary is weak."*

**S2-companion (mandatory; the two clauses that stop S2 being quoted out of scope).**
> *"Two boundaries travel with that number. **(i) Load.** Under the shipped history-dependent placement rule the same harness gives a post-deletion membership oracle whose strength scales with occupancy — AUC 0.6715 ± 0.0405, 0.9165 ± 0.0265, 0.9961 ± 0.0040, 0.99985 ± 0.00070 at 2, 4, 6 and 8 offers into a capacity-8 store (TPR at FPR 1 %: 0.029, 0.118, 0.924, 1.000) — so a placement rule evaluated at low load can look far closer to done than it is; we report the top of the curve. **(ii) Overflow.** At 8 offers into a 7-cell store the canonical rule is **not** exact: AUC(live count) = 1.000 and AUC(white-box address depth) = 0.914, because a background item refused in the world where the target was written does not counterfactually return when the target is deleted. That is an unbuilt admission waitlist, not a property of the placement rule."*

**S3 — the packing/admission sentence (unaffected by the capacity qualifier; carries the two-metric rule instead, CM-22(k)).**
> *"At K = 64 offers under the shipped two-phase Verlet read, canonical lattice placement admits **61/64 items deterministically (σ = 0)** with **per-admitted retrieval 1.0000 ± 0.0000** and therefore **per-offered 0.9531 ± 0.0000**; inflating the lattice radius by 1.05 (73 cells) gives **64/64** and per-offered **1.0000**. The refuse-and-relocate rule it replaces admits **43/64** on the same seeds (per-offered 0.6719). Per-admitted and per-offered are different metrics and both are stated: per-admitted alone would claim a perfect memory that holds a fraction of what it was offered. Minimum live spacing is **1.540000** by construction. Delete-time churn is **~3–4 survivor moves at full lattice load and ~0.2–0.4 at half load** (2.836 on the single-key-set protocol, 3.865 averaged over 200 independent key sets)."*

**S4 — the lifetimes sentence (CM-22(p) approved substitutes; both halves mandatory).**
> *"Decay does not make an item harder to detect: against an exact adversary the per-example white-box AUC is **1.000 at all 18 amplitude levels** down to the floor — what decays is the effect size, not the attack's success. What is true, and is the sentence we use, is that **the store stops answering before it stops leaking** (retention **0.832** at the last amplitude before self-eviction, while per-example distinguishability is still AUC **0.983** / TPR **0.858** at FPR 1 % on the query channel), and that decay **contracts an item's addressing tolerance** — the 50 % retrieval radius runs **1.146 → 0.752** (a factor 1.52) as amplitude falls from 1 to 0.06, where a TTL dictionary's lookup radius is a constant step at ≈0.77. The second is a retrieval-geometry statement and needs no adversary model; the first does."*
> ⚠ **Supersession risk:** `deletion-waitlist-stiffness` Part B re-measures `R₅₀` under a gated-stiffness payload channel. If the contraction does not survive the gate, S4's second clause must be rewritten from that report's numbers.

### 4B. POST-WAITLIST wordings — **PENDING** (as instructed; no number guessed)

`deletion-waitlist-stiffness` had **not landed** when I finished: `.claude/tasks/deletion-waitlist-stiffness.md` exists, `.claude/outputs/deletion-waitlist-stiffness.md` **does not** (checked 2026-07-29 15:16 BST). Per task §4 I mark this **PENDING** and supply the template with named slots and the rule that decides which branch applies. **Do not fill these in from anything but that report.**

**S1-post (template).** Two branches, chosen by that task's measured load sweep:
- *Branch FLAT* — **if and only if** the post-waitlist history column is 0.5000 with byte-equality restored **at every load in the sweep (2 / 4 / 6 / 8 offers)**:
  > *"Placement in the store is canonical …, so **store-level** deletion is exact **under set-function (priority/attribute-based) eviction, at every store load we measured (⟦LOAD SWEEP: … offers into a capacity-⟦C⟧ store⟧)**: removing an item reproduces, bit for bit, …"* — i.e. **"below capacity" is deleted and replaced by the measured load range**, never by an unqualified "exact".
- *Branch BOUNDED* — if the sweep is flat only up to some load L:
  > *"… deletion is exact **for stores at or below ⟦L⟧ ⟦unit: offers / live items⟧ into a capacity-⟦C⟧ store, or under set-function eviction**; above ⟦L⟧ the ⟦STATISTIC⟧ channel reopens (AUC ⟦x.xxxx ± x.xxxx⟧ at ⟦load⟧)."*

**S2-post (template).**
> *"With the admission waitlist enabled, the same harness at **⟦LOAD⟧ into a capacity-⟦C⟧ store (⟦N⟧ lattice cells, i.e. the overflow geometry)** gives history-column AUC ⟦x.xxxx ± x.xxxx⟧ on the live-count statistic (from 1.000) and ⟦x.xxxx ± x.xxxx⟧ on white-box address depth (from 0.914), with byte-equality ⟦f⟧/⟦N_worlds⟧ (from 0/3 072). Across the load sweep: ⟦2 offers: …⟧, ⟦4: …⟧, ⟦6: …⟧, ⟦8: …⟧."*

**Slots that MUST be filled from `deletion-waitlist-stiffness` and from nowhere else:** ⟦LOAD SWEEP⟧ ⟦L⟧ ⟦C⟧ ⟦N⟧ ⟦STATISTIC⟧ ⟦all AUC ± σ⟧ ⟦byte-equal fraction⟧ ⟦N_worlds⟧, **plus** that report's own "exact replacement scope sentence" (its Deliverable requires one — prefer it verbatim over these templates if it differs).
**Unchanged by the waitlist either way:** S3 (packing) and S4 (lifetimes) — the waitlist changes admission bookkeeping at overflow, not the rematch numbers or the decay channel. **Always retained regardless of branch:** the LRU exclusion, the store-level-only clause, and the flat-datastore control clause.

### 4c. ⭐ Qualifier self-audit (the falsifier from the Dial Declaration, executed)

| sentence | qualifier it cannot survive without | present? |
|---|---|---|
| S1 | "store-level" + "below capacity or under set-function eviction" + LRU exclusion + flat-datastore control | ✅ all four in-sentence |
| S2 | the **load** ("at 8 offers into a capacity-8 store sized to 13 cells") + n (3 072 worlds, 3 seeds × 8 targets × 128) | ✅ in-sentence |
| S2-companion | that the 0.6715…0.99985 curve is the **shipped/relocate** rule, not the canonical one | ✅ in-sentence |
| S3 | per-admitted **and** per-offered together (CM-22(k)); "under the shipped two-phase Verlet read"; the 1.05 sizing stated separately | ✅ in-sentence |
| S4 | "against an exact adversary"; "the graded curve exists only relative to a stated adversary resolution" is implied by the first clause; `R₅₀` clause labelled retrieval-geometry, not privacy | ✅ in-sentence |
| §3 roadmap cell 2 | "store layer" + "below capacity or under set-function eviction" + "we match, do not beat" + "instance of Blelloch–Golovin" | ✅ all four |
**No sentence I wrote survives only because a qualifier was dropped.** The one sentence I could not write at all is a Ginart-Def-A.5 cost claim (recon item 4) — flagged as a missing experiment rather than hedged into existence.

---

## 5. The four-part novelty composition — one-line statement for any future draft (paste-ready)

> **"Order-independent placement is not our contribution — it is Blelloch & Golovin's strongly-history-independent table (FOCS'07), instantiated here by replacing their hash probe order with a probe order induced by distance in the store's metric — and our contribution is the four-part composition around it: (i) the slots are a lattice packing, so the canonical representation carries a minimum-separation certificate (live spacing 1.540000 by construction) and hence a quantitative interference bound; (ii) the stored content is a continuously decaying amplitude rather than a static key, and deletion provably commutes with the decay flow, so survivors are bit-identical to a history in which the deleted item was never written, at every point of their schedules; (iii) the canonical object is therefore an energy function read by a dynamical relaxation, not a memory layout; and (iv) the price of strong history independence here is negative — the designed lattice admits 61/64 where the stochastic relocation rule it replaces admits 43/64 — against a literature in which strong history independence is proven to cost up to an exponential slowdown (Buchbinder & Petrank, CRYPTO'03)."**

Short form (abstract-length): *"We instantiate a known strongly-history-independent placement rule in a physical store, and show that the composition — a packing certificate, decaying content whose deletion commutes with decay, a canonical energy function, and a negative price for strong history independence — is what is new."*

⛔ **Never write, in any form:** "we prove that placement is order-independent" (say *follows from*) · "the first order-independent / uniquely-represented store" (false: BG07; and BGV08 already took unique representation into computational geometry) · "**our** fix-up cascade" (possessive; it is theirs) · "deletion by construction" as a novelty · "deletion-compliant" · unqualified "exact deletion" · any deletion claim in the same breath as LRU/staleness eviction.

---

## 6. Citation architecture for an R1 draft (keys as in `deletion-prior-art`; hermetic per M1/C-8)

**Must appear:** `blelloch2007shi` (⭐ owns the algorithm; also `blelloch2006shideletion` for the TR) · `hartline2005characterizing` (definitional equivalence only + reversibility caveat) · `naor2001antipersistence` (notions, no def. number) · `micciancio1997oblivious` (origin; the randomisation route, a contrast) · `buchbinder2006lower` (SHI has a proven price ⇒ our negative price is reportable) · `blelloch2008geometry` (unique representation already reached geometry — must differentiate) · `karger1997consistent` (one line) · `bourtoule2021sisa` · `min2024silo` (the hardest preemption; also the trivial substitute) · `ghazi2023ticketed` · `brophy2021dare` · `ozdenizci2025pall` (the CL ∩ deletion cell) · `ginart2019forget` + `sekhari2021remember` (Candidate-2 vocabulary) · `guo2020certified` (the thing we explicitly do not claim) · `thudi2022auditable` (why the claim must be algorithm-level) · `garg2020formalizing` (the meaning of "data deletion" we do not claim) · `shi2024muse` + `chen2024cure4rec` (multi-dimensional convention). **Optional framing:** `singh2021freshdiskann`, `chakraborttii2026ghost` (preprint-grade only).
**Naming/house rules for the R1 draft:** title `[WORKING TITLE: …]`, authors `[AUTHORS PLACEHOLDER]`; continuity sentence *"the CLU, introduced as CHLU in Jawahar & Pierini (2026)"*; **inertial M vs spectral μ**, never bare "mass"; physics-audit paragraph placement per C-1 (no defensive audit paragraph); designed-store results labelled **verification**, not evidence (C-2) — **every number in §4 is designed-store, zero learning, so an R1 short is a verification-grade paper unless a learned-φ arm is added**; appendix maximalism (C-10): the overflow cell, the load curve, the cascade-cost two-protocol discrepancy and the D2/D3 defects all go to appendices, fully written, never pruned now.

---

## 7. Could not verify / did not smooth over

1. **Guo et al. PMLR camera-ready not opened.** The "no numbered definitions" finding (recon 1) is from the arXiv LaTeX source of `1911.03030` (e-print tarball, last-modified 2023-11-09). It is possible — though unlikely, since the source declares no `definition` environment at all — that the ICML proceedings version numbers them. **Someone should open the PMLR PDF before camera-ready.** Until then cite *"§3, Eq. (1)"*.
2. **Blelloch–Golovin Thm 3.1/3.2, `DELETE`, Hartline Thm 1, SISA Def. III.1, MUSE's six criteria, CURE4Rec's four dimensions, and the FreshDiskANN/Ghost-Vectors content are INHERITED from `deletion-prior-art`**, not re-verified by me. I re-verified only the three the task named for re-typesetting (Sekhari Def. 3, Guo, and Ginart Def. A.5, the last two being where the defects were).
3. **Naor–Teague definition numbers remain unobtainable** (scout's ePrint 403 / `.ps` 422 / paywall). Unchanged.
4. **No number in this report was measured by me.** Every quantity traces to `placement-landing` (§2, §2b, §3, §5 + its flag-provenance table: base local `main` `ff85573`, measurement commit `e2d44cd`, main venv, **JAX 0.9.0**, eqx 0.13.4, seeds 0/1/2, `AtomStorePotential(dim=3, capacity=8, α=0.02, s=0.35, s_pay=s, κ=1.0)`, `d_safe = 4.4·s = 1.540`, `evict_policy="depth"`, two-phase read `dt 0.05`, `γ_address 0.05×400 → γ_read 0.0×800`, tail 0.25, 8 subsamples, 16 queries/item, σ_θ = 0.15, σ_p = 0.05, **no training anywhere — N94 does not apply**), to `carried-remeasurements` (§0.4, §2 table; base `ff85573`, JAX 0.9.0, seeds 0/1/2, 24 per-example values × 128 paired worlds × 4 loads, proposal disk `R = radius_for_capacity(8, 1.54) = 2.2869`), and to `mia-decay-measurement` (§1.6 handover items 1–2 for S4). **Any R1 draft inherits those two flag-provenance tables verbatim into its appendix (C-7); they are not reproduced in full here because they must be copied from the source reports, not from me.**
5. **I did not adjust, round or reconcile any number.** Where two protocols disagree (cascade cost 2.836 vs 3.865) both are quoted with their protocols, per `placement-landing` §5.

---

## 8. Missing-experiment notes for the Hub (do not improvise these into prose)

- **M1 — amortized deletion cost vs n.** Required before *any* Ginart-Def-A.5 sentence. The shipped `_canonical_sync` is a full O(n) `with_item` rebuild per op, so today's wall-clock is O(n) per delete regardless of the ~3–4-move cascade. Either implement the in-place slot move and measure, or write the cost claim purely as **survivor-move count** and say explicitly that amortized time is not claimed. *(I chose the latter in S3; the former is the paper-grade fix.)*
- **M2 — the deletion-cost-vs-utility table itself.** §1 gives the vocabulary; there is no table yet with the **flat-datastore row-delete** row in it. Without that row present, S1/S2 are compliant but the paper has no Candidate-2 result.
- **M3 — address quantization cost under a learned φ.** Canonical placement snaps every write to a lattice cell (≤ 0.889 at `d_safe = 1.540`); harmless for allocator-chosen addresses (per-admitted 1.0000) and **unmeasured** when the address carries similarity. This blocks R1 from composing with the CL entry.
- **M4 — the R1 evidence grade.** Every number in §4 is designed-store with **no learning anywhere**. Under C-2 the whole result set is **verification**, not evidence. If R1 is to be an evidence-grade short, it needs one learned-content arm; if not, the paper must say "verification" in its own contribution list. **This is an editorial question for the Head, not a wording fix.**

---

## Open editorial questions (for the Hub/Head)

1. ⭐ **The "certified" ban's scope** (§2b) — narrow-to-referent (my recommendation) or program-wide word ban (112 lines, four papers + 3 CHANGELOGs, a fresh V1 pass)? Nothing else in this report depends on the answer; V1's contribution list does.
2. **Venue pointer in the Part-4 R1 cell** (§3) — may "machine unlearning" be used as a *field name*? My replacement avoids it and avoids "right-to-be-forgotten" entirely. Head's call.
3. **One-clause amendment to the scout's approved §1.6 paragraph** (§1.6) — I replaced *"certified (ε,δ) unlearning"* with *"(ε,δ)-indistinguishability from retraining in the sense of Guo et al. (2020)"*. Approve or revert; if reverted, the sweep in §2 must carry a use/mention exemption or the paragraph itself becomes a banned-terms hit.
4. **R1's paper vehicle.** Given M4, is R1 a standalone short (verification-grade, store-level, with the composition as the contribution) or an ICLR-long section? The §1/§4/§5 blocks are vehicle-neutral and drop into either.

---

## Git footprint

**No tracked code touched; no branch, no commits.** Files created/edited (all under gitignored `.claude/`):
- `.claude/outputs/r1-positioning-pass.md` (this report) — created.
- `.claude/papers/iclr-long/outline.md` — **one line edited** (L10, §2a).
- `.claude/papers/iclr-long/CHANGELOG.md` — created (the directory had none; protocol requires one line per revision).
No transfer doc (`research_roadmap.md`, `claims_matrix.md`, `negative_results.md`, `philosophy-synthesis.md`), no task file, and no `chlu/` file was opened for writing. Scratch: `/tmp/r1src/` (arXiv e-print tarballs for 2103.03279, 1911.03030, 1907.05012 — outside the repo, disposable).

---

## Proposed handover updates (for the Hub)

1. ⛔ **`claims_matrix.md` CM-22(m): fix the Guo citation** — *"Guo et al., ICML 2020, Def 1–2"* → *"Guo et al., ICML 2020, §3, Eq. (1) (ε-certified removal) and the (ε,δ) relaxation immediately following"*. Same fix in the `guo2020certified` BibTeX `note` field in `deletion-prior-art`. **The definitions do not exist as numbered objects** (arXiv source verified; PMLR unchecked). Owner: doc-curator.
2. **CM-22(q) can be unblocked with a scope** — N99's three blockers all cleared in w26. Proposed successor text: *"deletion-flavoured sentences are permitted ONLY in the S1/S2 forms of `r1-positioning-pass` §4A: store-level, below capacity or under set-function eviction, LRU excluded, the acceptance number stated at a load (8 offers into a capacity-8 store sized to 13 cells), and the flat-datastore row-delete named as the control we match rather than beat."* Under the **shipped** placement, (q) stands unchanged.
3. **CM-22(r) can be lifted for "0.953"** — measured under the real two-phase Verlet read (`placement-landing` §3), and it travels with per-admitted 1.0000 per CM-22(k). Approved form = **S3** (§4A).
4. **New CM-22 candidate (v):** ⛔ *"α-deletion efficient" / any amortized-time deletion-cost claim* — the shipped re-pack is O(n) per operation; only the **survivor-move count** is measured (2.836 single-key-set / 3.865 over 200 key sets at full lattice load, 0.219–0.425 at half). Owner: doc-curator, at the review that accepts this report.
5. **New standing vocabulary rule:** in R1 prose, *deletion / removal* = the store-level record operation; *erasure* is reserved for the V5 physics sense (T = 0 coset erasure, the `T_φ` shredder). Prevents a cross-paper collision that already exists in the corpus (§2d).
6. **The Part-4 replacement string (§3) needs a Head decision, then a curator edit.** I did not touch `research_roadmap.md`. Line 110 currently carries *"certified"* twice and an unqualified *"deletion as a certified physical operation"* — it is the highest-visibility banned-term site in the program.
7. **Relay to `r1-positioning-pass`'s successor:** when `deletion-waitlist-stiffness` lands, §4B's slots must be filled from **that report's own "exact replacement scope sentence"**, preferentially verbatim. If its Part-B `R₅₀` result contradicts S4's second clause, S4 must be rewritten, not patched.
