# bprime-fb1-recon — web-scout report

**Task + acceptance criterion:** FB1 sweep — does ≥1 established paper in the neural-memory family already run a **matched-byte non-parametric control** and report the same verdict? Return a VERDICT, upgrade P4 from "absence of evidence" to a declared negative, re-pin the collision zone, and answer D4.
**Status: done.** (Two venue endpoints unreachable — declared in §5.)

## ⭐⭐ VERDICT — FIRST 10 LINES

> ## **FB1 = DOES NOT FIRE.**
> **No paper found, in any of the six surveyed families, that sizes a non-parametric store to a learned memory's declared STATE-BYTE budget, runs it as a control on the same task, and reports the verdict.** 14 candidates graded: **0 HIT · 2 PARTIAL (both out-of-family) · 7 NEAR-MISS · 5 NO.** **FB5's route does not fire** — arXiv:2501.12352 is a purely theoretical unification with no experiments (§2, row 1). **P4 survives, materially narrowed on two axes** (§3): the audit-at-equal-bits discipline *is* standard **outside** the family (learned data structures), and a **token-matched trivial control** was published **7 days ago** in LLM-agent memory evaluation (arXiv:2607.21962, 24 Jul 2026). **`bprime-rivals` is RELEASED — no re-scope needed.**
> ## ⛔ RECONCILIATION LIST (owner needed — protocol §5 corollary)
> 1. ⛔ **CITATION DEFECT: "MUNKEY (arXiv:2603.15033, **ICML 2026**)" is WRONG and appears in ≥4 of our documents** (charter §A9.9; `PREREG-Bprime.md` §8; `track2-admissibility.md` §3.3 + never-quote 13; this wave's task files). **arXiv v3 (2 Apr 2026) carries an EMPTY comments field — no venue.** The authors' own ETH group page lists it as **"Oral at ICLR Workshop TTU, 2026"**; an independent secondary says **"ICLR 2026 Workshop RSI"**. ⇒ It is an **ICLR-2026 *workshop* paper (oral), not an ICML main-track paper**, and the workshop's identity is **conflicting between two sources ⇒ quarantine the workshop name**. Owner: curator (same class of defect as "Titans is a preprint").
> 2. **MUNKEY's description also needs correcting:** v3 says "a **memory-augmented transformer**" evaluated on "natural image benchmarks, fine-grained recognition, and medical datasets" — our record says "ViT classifier" and cites v2. Owner: curator.
> 3. **Six rival papers absent from our registry** that `bprime-rivals` must at least name (§4): **Gated DeltaNet-2** (arXiv:2605.22791) supersedes GDN as the delta-rule reference arm. Owner: Hub → `bprime-rivals`.
> 4. **D4: the substitute-audit *idea* is NOT ours in general form** (§6) — it is the partial-input-baseline / trivial-baseline audit tradition (Poliak 2018; Feng, Wallace & Boyd-Graber ACL 2019). Owner: whoever drafts B′'s framing.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form — echoed before the first result)
- **Dial / pillar:** **none — recon.** No dial, no leaderboard, no dividend, no measurement of ours.
- **Laundering control:** n/a — the object hunted **is** a laundering control; candidates held to *our* four-part definition (PREREG §2), not to a paper's own use of "baseline". Zoology/Based (state bytes, hyperparameter-varied), MAD (iso-state across **neural** arch. only) and SDM (isoFLOP/isoParam) are re-graded as the known NEAR-MISSes, not re-filed as hits.
- **Falsifies:** an in-family paper meeting all four HIT criteria. **Does NOT falsify:** parametric matched-byte baselines · retrieval-augmented LM beating a neural LM · a clean negative (**the outcome obtained**).

## What I did
Filed `PREREG.md` (search protocol, grading rule, **stated prior P(FB1)=0.12**, and — pre-declared — that a learned-data-structures hit would be graded NOT-a-hit-but-narrowing) **before the first query**. Then ran the eight declared query families across web search + direct arXiv `abs`/HTML/ar5iv fetches, the authors' group pages, and OpenReview (blocked). Pulled abstracts/quotes from source, not summaries. Two load-bearing items adversarially re-extracted: **Based App. E/Fig. 2** (second extraction confirms the byte axis carries only neural sequence mixers) and **MUNKEY's venue** (second extraction **contradicted** our record — reconciliation 1).

**PREREG scorecard:** prior P(FB1)=0.12 → **did not fire** ✅. P(≥1 PARTIAL)=0.65 → **2 PARTIALs**, both out-of-family ✅. P(matched-space audit exists in a neighbouring field)=0.7 → **confirmed** ✅ (learned indexes/Bloom filters/sketches). P(D4 idea already published in some form)=0.85 → **confirmed** ✅.

---

# D1 — THE FB1 SWEEP (the candidate table)

**HIT = all four:** (1) non-parametric store; (2) sized to the learned memory's **state bytes** (declared); (3) run as a control on the **same task**; (4) **verdict reported**.

| # | paper (id, year, venue) | what it does at the byte/budget axis | grade | reason |
|---|---|---|---|---|
| 1 | **Wang, Shi & Fox, "Test-time regression"**, arXiv:2501.12352 (2025) — **FB5's neighbour** | unifies linear attention / SSMs / FWPs / online learners / softmax attention as test-time regression | **NO** | **Purely theoretical — no experiments, no baselines, no memory-budget discussion.** Softmax attention appears as the *nonparametric* (Nadaraya–Watson) special case **analytically**, never as a byte-matched control. ⇒ **FB1 does not fire by the FB5 route.** ⚠ fetched HTML v1 was truncated after §4; graded NO at medium-high confidence |
| 2 | **Arora et al., Based**, arXiv:2402.18668, **ICML 2024** | recall vs **"state size (bytes) during generation"**, the field's only explicit byte axis | **NEAR-MISS** (re-confirmed, 2nd extraction) | the axis is populated by **six neural mixers only** — attention, sliding-window attention, Mamba, H3, Hyena, Based. **No lookup table, kNN, n-gram or oracle scan is placed on it.** Criterion (1) fails |
| 3 | **Arora et al., Zoology**, arXiv:2312.04927, **ICLR 2024** | state varied by hyperparameters (`d_model`), best-of-lr-sweep | **NEAR-MISS** | varies state, never matches it *to* a non-parametric store |
| 4 | **Poli et al., MAD**, arXiv:2403.17844 (2024) | *"normalized to an iso-state and iso-parameter setting … a common total state dimension of 4,096"* | **NEAR-MISS** | iso-state **across neural architectures only**. Criterion (1) fails — already on the record, not re-filed |
| 5 | **Cabannes et al., Sparse Delta Memory**, arXiv:2607.07386 (2026) | isoFLOP + isoParam + a state/parameter ratio column | **NEAR-MISS** | no table baseline; ratio column is state-vs-*params*, not state-vs-*store*. ⛔ its Table 1 ratios stay quarantined |
| 6 | ⭐ **Cui, "A Hippocampus for Linear Attention" (HOLA)**, arXiv:2607.02303 (2 Jul 2026, preprint) | *"adds a bounded exact KV cache, forming a **semiparametric test-time memory**"*; compares against *"a **matched** HOLA+**recency** cache"* on RULER | **NEAR-MISS — the closest in-family cousin** | the exact KV cache is **non-parametric and in-family**, and the recency variant is a **matched-budget trivial-policy control** — but it is a *trivial WRITE rule inside their own architecture*, not a non-parametric store priced against the learned state, and the cache is sized by tokens (`w=64/layer`), never to the delta-rule state's bytes. Criteria (2) and (4)-as-we-draw-it fail |
| 7 | **Behrouz, Zhong & Mirrokni, Titans**, arXiv:2501.00663, **NeurIPS 2025** ⛔ *never "a preprint"* | baselines = Transformer++, RetNet, GLA, Mamba/Mamba-2, DeltaNet, TTT | **NO** | all-neural; **no state-byte convention stated at all** (our `2·\|M_θ\|` remains our reconstruction) |
| 8 | **Behrouz et al., ATLAS**, arXiv:2505.23735 (2025) — the Titans line's successor | Table 1 vs Transformers, RetNet, GLA, RWKV-7, DeltaNet, Titans, SWA | **NO** | full-text sweep of the experiments section: **no non-parametric baseline; no explicit statement of state/memory-size matching** |
| 9 | **Sun et al., TTT**, arXiv:2407.04620 (2024) | `b=16` mini-batch speed/quality dial; baselines Transformer/Mamba | **NO** | no non-parametric control |
| 10 | **Yang et al., DeltaNet / Gated DeltaNet** (arXiv:2406.06484 NeurIPS 2024; arXiv:2412.06464 ICLR 2025) + **Gated DeltaNet-2** (arXiv:2605.22791) | head-dim / chunk ablations, LM + recall benchmarks | **NO** | all-neural comparison sets throughout the delta-rule line |
| 11 | **Khandelwal et al., kNN-LM**, arXiv:1911.00172, **ICLR 2020** + **Xu, Alon & Neubig**, ICML 2023 | a genuinely non-parametric datastore (103 M entries, **keys quantised to 64 B**) *interpolated into* an LM | **NEAR-MISS** | the store is an **augmentation**, never a byte-matched **control** on a learned state; λ is tuned (0.25), and the λ=1 pure-kNN read is degenerate (∞ ppl) — this is the evidence **for** our P1, not a rival's control |
| 12 | **Shao et al., MassiveDS**, arXiv:2407.12854, **NeurIPS 2024** | compute-optimal curves over **datastore size × model size × pretraining tokens**; *"a smaller model augmented with a large datastore outperforms a larger LM-only model"* | **NEAR-MISS — the strongest "price the table" precedent** | the trade is priced against **training compute and parameters**, never against a learned memory's **state bytes**; and the datastore is orders of magnitude larger than any state. Criterion (2) fails |
| 13 | ⭐ **Spencer, "Ground Truth First…"**, arXiv:2607.21962 (**24 Jul 2026**, single-author preprint) | *"a **token-matched recency window** (most recent whole events under hybrid v2's **mean read budget of ∼1,807 tokens**)"* + a full-history baseline, against 5 agent-memory backends; verdict drawn: *"Full history scores 97.9% (808/825) — statistically indistinguishable from hybrid v2's 96.8%"*, *"small-history evaluations like this one **may fail to separate the tested memory systems from a trivial strategy**"* | ⭐ **PARTIAL — out-of-family** | 3 of 4 criteria met (non-parametric ✅, same task ✅, verdict ✅); **budget is matched in READ TOKENS, not state bytes**, and the systems audited are **LLM-agent memory pipelines**, not test-time-dynamics memories. **Not "established"** (8 days old, single author, no venue). ⇒ **does not fire FB1, but narrows P4's wording** |
| 14 | ⭐ **The learned-data-structures line** — Mitzenmacher, *"A Model for Learned Bloom Filters… Sandwiching"*, **NeurIPS 2018** (arXiv:1901.00902); Kipf/Marcus et al., **SOSD** (arXiv:1911.13014) & *Benchmarking Learned Indexes*, **PVLDB 14** | learned structure vs classical structure **at matched space**, false-positive-rate-vs-bits and size-vs-lookup-time Pareto; SOSD's verdict: PGM *"30–80× larger than B-trees"* on real data, *"4 orders-of-magnitude more time to build"* | ⭐ **PARTIAL — out-of-family, declared in PREREG §1 in advance** | this **is** the matched-bits learned-vs-nonparametric audit, mature and adversarial — but there is **no sequence, no test-time dynamics, no state**. ⇒ **does not fire FB1; it is B′'s methodological ancestry and must be cited as such rather than suppressed** |

**Reading of the table.** Seven independent groups (Stanford/Hazy, Together, MIT/Kim, Google, Meta FAIR, CMU, Zurich) built the *adjacent* instrument — a byte axis (Based), an iso-state normaliser (MAD), a state/param ratio column (SDM), a matched trivial-policy control (HOLA), a compute-priced datastore (MassiveDS) — and **none of them closed the loop by putting a non-parametric store on the learned memory's own byte budget.** The near-misses are the field's own evidence that the control is not a convention. That sentence is B′'s novelty argument and it is now supported by named papers rather than by silence.

---

# D2 — P4, upgraded from "absence of evidence" to a declared negative

## ⭐ 2.1 The sentence B′ should actually print (narrowest true version — the deliverable)

> **Across the modern neural sequence-memory family surveyed here — delta-rule and linear-attention models (DeltaNet, Gated DeltaNet, Gated DeltaNet-2), SSMs (Mamba-1/2/3), test-time-trained memories (TTT, Titans, ATLAS), explicit-slot memories (Sparse Delta Memory), semiparametric hybrids (HOLA), and the family's own recall and architecture-search benchmarks (Zoology, Based, MAD, RULER) — we find no paper, as of 31 July 2026, that sizes a *non-parametric* store (a table, kNN index, count-based model, or explicit (k, v) rows) to a learned memory's **declared state-byte budget**, runs it as a control on the same task, and reports the comparison. The nearest existing conventions are iso-state normalisation *across neural architectures only* (MAD §3.2), a state-bytes axis populated exclusively by neural sequence mixers (Based, App. E / Fig. 2), and isoFLOP/isoParameter reporting with a state-to-parameter ratio (Sparse Delta Memory). Budget-matched controls against non-learned alternatives are, by contrast, routine **outside** this family: at matched space in learned data structures (learned Bloom filters, learned indexes, learned sketches), and — concurrently with this work — as a token-matched recency window in LLM-agent memory evaluation. We therefore position this audit as **importing an established discipline into a family that has not adopted it**, not as inventing it.**

**Why this wording and not the stronger one.** *"No published rival paper runs a non-parametric matched-byte control"* is unsupportable as written: it quantifies over all papers, and two families outside the survey (learned data structures; agent memory) run recognisable versions of the control. The scoped version is **checkable by a referee** — it names the families, the date, the convention (state bytes, not tokens/FLOPs/params), and it concedes the ancestry before a reviewer finds it. A conceded ancestor is worth more than a contested monopoly.

## 2.2 Confidence grade on P4 (replacing "medium confidence")

| scope | grade | basis |
|---|---|---|
| **No in-family paper runs the control at the STATE-BYTE convention** | **medium-high** | 14 candidates graded; every family's *evaluation-convention* section read at source; the two papers that own the byte axis (Based) and the iso-state axis (MAD) re-extracted and confirmed neural-only. Residual risk: appendices of ~6 papers not read line-by-line (§5) |
| **No in-family paper runs it at ANY budget convention** | **medium** | HOLA's matched recency cache (arXiv:2607.02303) is a matched-budget trivial control *inside* an architecture; a similar ablation could exist unadvertised in an appendix |
| **The control is unknown outside the family** | ⛔ **FALSE — do not claim** | learned Bloom filters / learned indexes / learned sketches; arXiv:2607.21962 |
| **B′'s cross-family, uniform, state-byte protocol is unprecedented** | **medium-high** | no paper found applying one byte-matched non-parametric protocol across ≥3 memory families |

## 2.3 Declared search protocol (this is what makes the negative citable)
- **Engines:** general web search (Google-index-backed) + direct fetch of `arxiv.org/abs`, `/html`, ar5iv, PMLR/NeurIPS proceedings, authors' institutional pages, GitHub READMEs, OpenReview (blocked).
- **Date range:** to **2026-07-31**; priority 2024-01→2026-07; retrieval lineage back to 2018.
- **Query families run (all eight declared in PREREG §3):** matched-state-bytes/iso-state/equal-bytes table baselines × {linear attention, SSM, recurrent, memory}; non-parametric baseline × {state size, memory budget, KV bytes}; **forward sweeps** of Zoology / Based / MAD / TTT / Titans / DeltaNet / GDN / SDM / **Test-time regression**; **backward** sweep of kNN-LM / Xu-Alon-Neubig / MassiveDS; the deletion sweep (MUNKEY forward citations); the collision-zone re-pin; the D4 trivial-baseline tradition; and the declared out-of-family ancestry sweep.
- **Snowball actually executed:** anchor→citing via search-index surfacing (not a citation-graph API — see §5); Titans→ATLAS/Miras/It's-All-Connected; DeltaNet→GDN→**GDN-2**→Erase-then-Delta→Preconditioned DeltaNet; SDM→HOLA; MUNKEY→ (unlearning-2026 set); kNN-LM→MassiveDS.
- **Stopping rule applied:** two consecutive query families produced no new candidate *class*.

---

# D3 — Near-neighbour watch (re-pinned 2026-07-31; the C2W2 pin was 2026-07-30)

## 3.1 ⛔ Exact deletion in a sequence memory — **NOT FOUND. No positioning emergency.**
The pillar-4 falsifier registered as "report same-day" **did not fire**. The 2026 unlearning sweep surfaced representation-level, parameter-level and optimizer-state work, none of it a sequence memory:
- **MUNKEY** (arXiv:2603.15033v3, 2 Apr 2026) — still the nearest; **image classification**, memory-augmented transformer, not exact. ⚠ **venue defect, reconciliation 1.**
- **Stewart, "Form and Function: Machine Unlearning as a Problem of Misaligned States"** (arXiv:2605.17590, 17 May 2026, single-author preprint) — **online L-BFGS optimizer state**, not a sequence memory. *Relevant only as vocabulary:* it formalises unlearning as **counterfactual state alignment** — *"the target of unlearning is the optimizer state that would have arisen had the deleted samples never been processed"* — which is **exactly our byte-equality-to-the-never-written-counterfactual instrument, in someone else's words.** Cite it as convergent framing; it strengthens the instrument's legitimacy and costs us nothing (different object, and they explicitly *cannot* achieve exactness).
- **"Exact Unlearning in Reinforcement Learning"** (OpenReview `R975odFtp0`) — surfaced, **unreadable (bot-block)**; RL setting, not sequence memory. Declared unresolved.
- Verdict: **byte-exact deletion in a sequence memory remains uncontested**, with the MUNKEY narrowing now *weakened* (workshop, not ICML).

## 3.2 Collision zone — nothing newer than the C2W2 pin in the neural-memory line
Newest in-family items: **HOLA** 2 Jul 2026 · **SDM** 7–8 Jul 2026 (unchanged as the frontier). **Nothing dated after 2026-07-30** surfaced in the neural-memory or audit/benchmark lines. The only post-pin item anywhere in scope is **arXiv:2607.21962 (24 Jul 2026)**, in agent memory (row 13).

## 3.3 ⭐ Six rival papers not in our registry (input to `bprime-rivals`)
| paper | why it matters |
|---|---|
| **Gated DeltaNet-2**, arXiv:2605.22791 — channel-wise **erase** gate `b_t` + channel-wise **write** gate `w_t`, decoupling the scalar erase/write tie; *"achieves the strongest overall results among Mamba-2, Gated DeltaNet, KDA, and Mamba-3 variants"* at 1.3 B | ⚠ **the delta-rule reference arm has moved.** A "GDN" baseline in B′ is now one generation stale; name GDN-2 at minimum in limitations (same rule as "real Mamba = Mamba-2 min.") |
| **Erase-then-Delta Attention**, arXiv:2606.26560 | same erase/write decoupling theme — pillar-4-adjacent (erase is *not* deletion; it is gated overwrite) |
| **Preconditioned DeltaNet**, arXiv:2604.21100 | curvature-aware linear recurrence; TTT-line neighbour |
| **ATLAS**, arXiv:2505.23735 + **"It's All Connected"**, arXiv:2504.13173 (the Miras line) | the Titans successor line; **swept: no non-parametric baseline, no state-size matching statement** (row 8). Relevant to any "Titans-class" arm |
| **HOLA**, arXiv:2607.02303 | ⭐ the **semiparametric** framing (learned compressive state **+** exact non-parametric store, jointly) is the nearest published relative of "price the dynamics against the table" — B′ must cite it, and can cite it *favourably*: the field is already conceding that part of the payload belongs in an exact store |

## 3.4 SDM Table 1 conflict — **NOT RESOLVED; quarantine stands**
Not reachable from an extractable source in this session (PDF endpoints for this paper have defeated two agents). **⛔ No SDM state/param ratio is quoted anywhere in this report.**

---

# D4 — Is the **substitute audit** ours? **NO, not in its general form. Partly, in its specific form.**

**Answer:** the idea *"a trivial reader of the same (or less) information matches the learned system, therefore the reported gain is an artefact"* is a **well-established audit tradition since 2018**, under the name **partial-input baselines**. What we have found **no publication of** is the specific instantiation: a **+0 B reader over a learned memory's own stored bytes** (insertion order · the query echoed · an aggregate), applied to a *memory architecture*, with **reporting only the frozen/ablated control named as laundering by omission**.

| candidate | grade vs D4 | evidence |
|---|---|---|
| **Poliak, Naradowsky, Haldar, Rudinger, Van Durme (2018), "Hypothesis Only Baselines in Natural Language Inference", \*SEM 2018, arXiv:1805.01042** | ⭐ **HIT (the idea's ancestor)** | trains on hypotheses alone, i.e. a reader denied the premise, and shows it beats the majority baseline on 6/10 NLI datasets ⇒ the *benchmark*, not the model, was doing the work |
| **Feng, Wallace & Boyd-Graber (2019), "Misleading Failures of Partial-input Baselines", ACL 2019, arXiv:1905.05778** | ⭐ **HIT — and it is a caveat B′ must print** | *"When a partial-input baseline gets high accuracy, a dataset is cheatable. However, the converse is not necessarily true: the failure of a partial-input baseline does not mean a dataset is free of artifacts."* ⚠ quoted from a search-surfaced PDF extract — **single-sourced, re-verify before printing.** **Direct consequence for B′:** our audit went **0-for-4** (substitutes *won*) ⇒ by this logic the finding is "the protocol/task is cheatable", which is what we say; but for any *future* family where the substitute **loses**, we may **not** conclude the memory is doing real work. **FB4 is exactly this argument** and should be credited to it |
| **Spencer (2026), arXiv:2607.21962** (row 13) | **PARTIAL — closest live instantiation** | a **token-matched** trivial reader (recency window) + a full-history reader, run against memory systems, with the explicit "may fail to separate … from a trivial strategy" verdict — published **7 days before this report** |
| **Cui (2026), HOLA**, arXiv:2607.02303 | **PARTIAL** | *"a **matched** HOLA+recency cache"* — the same bytes with a trivial policy, as a control, in-family |
| **"laundering by omission" — reporting only the frozen control** | **NOT FOUND** | no paper found naming the *omission* of the trivial-reader column as a methodological failure in memory work |

**⇒ B′'s framing change (do this now, it is cheap):** drop any implication that the substitute audit is a new instrument. Print it as *"we apply the partial-input-baseline discipline (Poliak et al. 2018; Feng et al. 2019) to a memory's own stored bytes at a +0 B budget"*, and **carry Feng et al.'s converse caveat explicitly** — it is the honest ceiling on what a *passed* substitute audit can license, and printing it pre-empts the strongest available referee attack on B′'s positive cells.

---

# §5 — What I could not reach, and NOT-SEARCHED areas (never presented as searched-and-empty)

**Blocked / unreachable (declared, not omitted):**
1. ⛔ **OpenReview — bot-blocked AGAIN (third consecutive wave: C2W2 ×2, now C2W3 ×1).** `openreview.net/forum?id=gGH3Xp1lHR` (MUNKEY) returned *"Complete the check below to continue to OpenReview"*. ⇒ **MUNKEY's venue could not be resolved from the venue's own record**; the correction in reconciliation 1 rests on (a) an **empty arXiv comments field** and (b) the **authors' institutional page**. Also unreachable: `R975odFtp0` (Exact Unlearning in RL), and the GDN/Titans reviewer threads (seed conventions, still owed from C2W2 R4). **This now needs a human or an authenticated fetch — it has cost three waves.**
2. **arXiv PDF endpoint for arXiv:2402.18668** returned compressed binary (unextractable); Based App. E was read via **ar5iv** instead — the per-architecture byte formulas remain as pinned in `rival-recon` (from App. E.2), and my re-extraction confirms only the *figure's model list* and the *bytes-during-generation* axis.
3. **arXiv:2501.12352 HTML truncated** after §4; "purely theoretical, no experiments" is graded medium-high, not pinned to a full read.

**⛔ DECLARED NOT-SEARCHED:**
- Non-English literature; patents; theses; paywalled venues without an arXiv/anthology mirror.
- **Full-text appendices** of: Titans (camera-ready), TTT, DeltaNet, GDN, GDN-2, Mamba-2/3, RULER. I read their **baseline/normalisation conventions**, not every appendix line. A byte-matched table baseline hidden in an unadvertised appendix would have been missed.
- **A true citation-graph sweep.** I had no Semantic Scholar/OpenAlex API in this session; "forward citations" here means *search-index surfacing of citing work*, which is recall-limited. **This is the single biggest residual risk to P4** and the cheapest thing to close next.
- Vision/audio/RL memory literature except where it surfaced (POPGym, world-model memory papers surfaced but were not graded).
- The **agent-memory** field beyond the two papers graded — ⚠ note this field is where the *other* PARTIAL lives, so it is under-swept relative to its hit rate.
- **SDM's Table 1** (quarantine unresolved) and **SDM's appendix**.

---

# BibTeX-ready refs (NEW only — appends to `rival-recon`'s and `track2-admissibility`'s lists)
```bibtex
@article{cui2026hola, title={A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets},
  author={Cui, Wanyun}, journal={arXiv preprint arXiv:2607.02303}, year={2026},
  note={2 Jul 2026; 12 pages; preprint, no venue; "semiparametric test-time memory"}}

@article{spencer2026groundtruthfirst, title={Ground Truth First: A Longitudinal Evaluation Instrument for Agent Memory, and the Tenure Crossover in Memory-Architecture Rankings},
  author={Spencer, Quentin}, journal={arXiv preprint arXiv:2607.21962}, year={2026},
  note={24 Jul 2026; single author; preprint, no venue; token-matched recency-window baseline}}

@inproceedings{poliak2018hypothesisonly, title={Hypothesis Only Baselines in Natural Language Inference},
  author={Poliak, Adam and Naradowsky, Jason and Haldar, Aparajita and Rudinger, Rachel and Van Durme, Benjamin},
  booktitle={Proceedings of *SEM}, year={2018}, note={arXiv:1805.01042}}

@inproceedings{feng2019misleading, title={Misleading Failures of Partial-input Baselines},
  author={Feng, Shi and Wallace, Eric and Boyd-Graber, Jordan},
  booktitle={ACL}, year={2019}, note={arXiv:1905.05778}}

@inproceedings{mitzenmacher2018learnedbloom, title={A Model for Learned Bloom Filters and Optimizing by Sandwiching},
  author={Mitzenmacher, Michael}, booktitle={NeurIPS}, year={2018}, note={arXiv:1901.00902}}

@article{kipf2019sosd, title={SOSD: A Benchmark for Learned Indexes},
  author={Kipf, Andreas and Marcus, Ryan and van Renen, Alexander and Stoian, Mihail and Kemper, Alfons and Kraska, Tim and Neumann, Thomas},
  journal={arXiv preprint arXiv:1911.13014}, year={2019}, note={NeurIPS 2019 ML for Systems workshop; extended as Marcus et al., PVLDB 14(1):1--13, 2021}}

@inproceedings{shao2024massiveds, title={Scaling Retrieval-Based Language Models with a Trillion-Token Datastore},
  author={Shao, Rulin and He, Jacqueline and Asai, Akari and Shi, Weijia and Dettmers, Tim and Min, Sewon and Zettlemoyer, Luke and Koh, Pang Wei},
  booktitle={NeurIPS}, year={2024}, note={arXiv:2407.12854}}

@article{behrouz2025atlas, title={ATLAS: Learning to Optimally Memorize the Context at Test Time},
  author={Behrouz, Ali and others}, journal={arXiv preprint arXiv:2505.23735}, year={2025}}

@article{gdn2_2026, title={Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention},
  journal={arXiv preprint arXiv:2605.22791}, year={2026},
  note={channel-wise erase gate b_t and write gate w_t; author list not extracted in this session}}

@article{erasethendelta2026, title={Erase-then-Delta Attention: Decoupling Erase and Write Addresses in Delta-Rule Linear Attention},
  journal={arXiv preprint arXiv:2606.26560}, year={2026}, note={author list not extracted}}

@article{stewart2026formfunction, title={Form and Function: Machine Unlearning as a Problem of Misaligned States},
  author={Stewart, Kennon}, journal={arXiv preprint arXiv:2605.17590}, year={2026},
  note={17 May 2026; online L-BFGS optimizer state; counterfactual state alignment}}
```
⚠ **Two refs carry incomplete author lists** (GDN-2, Erase-then-Delta) — flagged rather than invented. Complete before any draft cites them.

**⛔ CORRECTED ref (replaces the entry in `track2-admissibility.md`):**
```bibtex
@article{laguna2026munkey, title={Rethinking Machine Unlearning: Models Designed to Forget via Key Deletion},
  author={Laguna, Sonia and da Silva Gon{\c c}alves, Jorge and Vandenhirtz, Moritz and Ryser, Alain and Cannistraci, Irene and Vogt, Julia E.},
  journal={arXiv preprint arXiv:2603.15033}, year={2026},
  note={v3, 2 Apr 2026; arXiv comments field EMPTY; authors' institutional page lists "Oral at ICLR Workshop TTU, 2026"; a secondary source says "ICLR 2026 Workshop RSI" -- workshop identity CONFLICTING. NOT ICML.}}
```

---

# Open questions / follow-ups / risks
1. ⭐ **The only cheap way to strengthen P4 further is a real citation-graph sweep** (Semantic Scholar / OpenAlex API) over the 9 anchors. Search-index snowballing is recall-limited and is P4's largest residual risk. ~1 hour with an API key.
2. ⛔ **OpenReview has now blocked three consecutive waves.** Three separate deliverables are stalled behind it (seed conventions; MUNKEY's venue; Exact-Unlearning-in-RL). Escalate to the Head as a tooling item, not a research item.
3. **HOLA is the paper most likely to become a genuine FB1 hit in its next version** — it already frames memory as *semiparametric* (learned state + exact store). Re-check at camera-ready.
4. **Risk to B′'s framing (D4):** if a referee knows the partial-input-baseline literature and we do not cite it, the substitute audit looks reinvented. Citing it costs one sentence and buys the Feng et al. caveat, which we need anyway.
5. **GDN-2 vs GDN** changes which delta-rule arm `bprime-rivals` should build. Hub decision.

## Proposed handover updates (for the Hub)
1. **[C2W3] `bprime-fb1-recon` landed. ⭐ FB1 = DOES NOT FIRE; `bprime-rivals` is released unchanged.** 14 candidates graded, 0 HIT. **FB5's route also does not fire** — arXiv:2501.12352 is purely theoretical. **P4 survives, narrowed:** the matched-space learned-vs-classical audit is standard in **learned data structures** (Mitzenmacher NeurIPS 2018; SOSD), and a **token-matched trivial control** was published in agent-memory evaluation on **24 Jul 2026** (arXiv:2607.21962). **The printable P4 sentence is in `.claude/outputs/bprime-fb1-recon.md` §D2.1 — use it verbatim; the unscoped version is now on the never-quote list.**
2. ⛔ **NEVER-QUOTE additions:** **"MUNKEY (ICML 2026)"** — it is an **ICLR 2026 *workshop* paper (oral)**, arXiv comments field empty, workshop identity conflicting between two sources ⇒ quote as *"an ICLR 2026 workshop paper"* and quarantine the workshop name · **"No published rival paper runs a non-parametric matched-byte control"** unscoped — only the §D2.1 scoped version is true · **"the +0 B substitute audit is our instrument"** — it is the partial-input-baseline tradition (Poliak et al. 2018; Feng et al. ACL 2019) applied to a store's own bytes · **"a failed substitute audit shows the memory is doing real work"** (Feng et al.'s converse caveat forbids it).
3. **Curator work order (reconciliations 1–2):** replace every "MUNKEY … ICML 2026" occurrence (charter §A9.9; `PREREG-Bprime.md` §8; `track2-admissibility.md` §3.3 + never-quote 13; C2W3 task files) and change "ViT classifier" → "memory-augmented transformer, image-classification benchmarks (v3, 2 Apr 2026)". **Pillar 4's narrowing is now weaker than recorded** — a workshop oral, not an ICML main-track paper — but the mechanism is still published and the claim still must be phrased on verified byte-exactness.
4. **`bprime-rivals` rider:** the delta-rule reference arm has moved — **Gated DeltaNet-2 (arXiv:2605.22791)** decouples erase/write with channel-wise gates and reports the strongest results among Mamba-2/GDN/KDA/Mamba-3 at 1.3 B. Same rule as "real Mamba = Mamba-2 minimum": build GDN, **name GDN-2 in limitations**. Also add **HOLA (arXiv:2607.02303)** to related work — a *semiparametric* memory (learned state + bounded exact KV cache) is the field conceding B′'s premise, and it runs a **matched recency-cache** control of its own.
5. **B′ framing change (cheap, do it in the outline):** cite the partial-input-baseline ancestry for the substitute audit and print Feng et al.'s converse caveat; cite Stewart (arXiv:2605.17590) as convergent vocabulary for deletion-as-counterfactual-state-alignment; cite the learned-data-structures line as the matched-space ancestor. B′'s contribution is then *"we import a mature audit discipline into a family that has not adopted it, at the state-byte convention"* — narrower, and unattackable.
6. ⛔ **Tooling escalation:** OpenReview has bot-blocked **three consecutive waves**; three deliverables are stalled behind it. Needs a human/authenticated fetch, or the affected claims stay permanently single-sourced.
