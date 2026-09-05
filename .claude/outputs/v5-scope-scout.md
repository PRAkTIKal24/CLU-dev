# v5-scope-scout — web-scout report

**Task + acceptance criterion:** (1) PALM's topic scope from the venue's own site, verbatim (the standing Add.6/A5.7 Q7 known-gap, now blocking V5 §4); (2) a cited venue-native agent/LLM long-term-memory brief for §4 (which has zero venue-native citations, SF-7/ME-7); (3) fresh verification + BibTeX for V5's deletion/unlearning/physics citation set, with usage-checks on every content-lean; (4) SF-10's competing-work citation recovery.
**Status:** done (one item **could-not-verify**, flagged: MUNKEY's presentation type).
**DIAL DECLARATION (echoed): none — venue/literature scouting + citation verification; no performance claim; no laundering control applies.**

> ## ⚠ DOWNSTREAM RECONCILIATION LIST — READ FIRST (needs a named owner; default `v5-revision-1`)
> 1. **⛔ HEADLINE — `Guo et al. §3 Eq. (1)` is WRONG at all three V5 sites (§4 l.71, E.7 l.194, K.2 l.277), and the N131 citation fence itself encodes the error.** In the published ICML/PMLR version, **§3 is "Removal Mechanisms"**; ε-certified removal is **Equation (1) in §2 "Certified Removal"**, and the **(ε,δ)** relaxation is an *unnumbered* display immediately after it. V5 also conflates the two: Eq. (1) is the **ε**-only condition. Correct form below (§3.2 item 3). **The matrix/registry fence must change with the draft, or v0.3 will "comply" with a wrong rule.**
> 2. **⛔ HEADLINE — K.2's history-independence provenance is misattributed.** *"strongly history-independent (uniquely represented) data structures were introduced by Micciancio (STOC'97) and Naor & Teague (STOC'01)"*: Micciancio (STOC'97) introduced ***oblivious*** data structures (a distributional condition on pointer representations, explicitly **not** canonical representation); unique/canonical representation goes back to **Snyder (FOCS'77)**, Sundar & Tarjan (STOC'90), Andersson & Ottmann (SICOMP 1995); the **two history-independence notions (WHI/SHI) are Naor & Teague's**. Verified from Blelloch–Golovin's own "Previous Work". Rewrite in §3.2 item 6.
> 3. **⚠ U1 (Mo's theorem) is UNCORRECTED IN V5 at two sites** — §4 l.71 and K.2 l.273 both say *"has $\dim(G/\mathcal H)$ zero Lyapunov exponents"*; Mo says **"at least dim(G/H) ... not exactly that number."** `v2-cite-check` reconciliation item 1 was applied to V2 v0.8 (Add.25) and **never transferred to V5**. The `v5-referee-v02` pass did not catch it.
> 4. **⚠ U14 (conformal symplecticity ← HLW) is also uncorrected in V5** (K.2 l.283: *"The conformal-symplectic structure and the $h<2$ stability limit are standard for damped leapfrog integrators (Hairer, Lubich & Wanner)"*, **cited with no year**). Origin is **McLachlan & Perlmutter 2001**; HLW is right for leapfrog/stability and Ch. XII only.
> 5. **⚠ SF-10's "(oral)" is NOT VERIFIABLE.** MUNKEY's arXiv record carries **no comments field and no venue**; OpenReview is bot-walled; one weak secondary says **Poster**, N168 says **oral**. Recommend citing venue-free ("a 2026 preprint, arXiv:2603.15033") or "an ICLR-2026 workshop paper" **without** a presentation type. Also: **0.56 ± 0.21 is MUNKEY's "Average Gap" to the retrained oracle across four metrics (CIFAR-10, 10% forget), not a membership-inference gap** — V5's phrasing invites a reviewer to read it as the latter.
> 6. **⚠ Two named laws in V5 have no citation at all:** the **Gell-Mann–Oakes–Renner** law (App A.0 l.87) and **Coleman / Mermin–Wagner** (App G heading + first line l.218-220). Records supplied in §3.4.
> 7. **ℹ SF-7 is over-stated in one direction and under-stated in another** — see Part 1 §1.3: V5's deletion contribution maps onto **two named CFP topics verbatim** ("memory update and deletion tests"; "right-to-be-forgotten mechanisms"), so the fit risk is *citational*, not *topical*. The §4 gap is real; the "wrong venue" reading is not supported by the CFP.

**What I did:** fetched the PALM site (three passes, different prompts) for the CFP verbatim; verified 19 citation records fresh against publisher/proceedings/arXiv primaries (two PDFs read page-by-page: Blelloch–Golovin FOCS'07 and Guo et al. PMLR 119); carried forward 11 records already verified in `v2-cite-check` without re-verification; assembled a 13-work venue-native brief, every record primary-verified; recovered SF-10's competing work; swept the V5 draft per-file (positive-controlled) for citations the referee's MF-6 list missed. **Retrieval date for everything below: 2026-08-19.**

**How I verified:** primary sources only where one exists (arXiv abs/HTML, PMLR, DBLP API, ACM DL DOI, APS/PTEP journal-refs, author-hosted PDFs read directly with the PDF reader). Labelled fallbacks where blocked: `link.springer.com` (303 SSO → DBLP API), `sciencedirect.com` (403 → author-hosted Technion PDF), `openreview.net` (bot challenge on `/forum`, `/pdf`, `api.openreview.net`, `api2.openreview.net` → **could-not-verify**, recorded as such).

---

# PART 1 — PALM's topic scope (closes the Add.6/A5.7 Q7 known gap)

**Source:** https://palm-neurips-2026.github.io/ (the workshop's own site), retrieved 2026-08-19. ⛔ Facts only; every fit judgment below is labelled as such and is the Advisor's/Head's to make.

## 1.1 Identity and remit
- Full name (site header): **"PALM · NeurIPS 2026"**, expanded as **"Personalized, aligned, long-term memory for AI systems"**.
- Scope sentence, verbatim: systems that *"retain information across sessions, personalize to users, reason over long horizons, and act consistently across tasks, tools, and modalities"*, built on *"persistent memory layers that encode, retrieve, update, and sometimes forget past experience."*
- Communities invited, verbatim: *"machine learning, NLP, AI agents, HCI, cognitive science, neuroscience, privacy, security, and AI safety."*

## 1.2 The seven CFP topics (headings verbatim; example-lists verbatim where retrieved)
1. **"Memory Architectures for Conversational Assistants"** — *"how conversational assistants should write, retrieve, update, consolidate, and forget memories across sessions"*; memory stores, retrieval policies, consolidation, *"handling stale or contradictory memories."*
2. **"Memory for LLM Agents & Multi-Agent Systems"** — *"persistent task histories, tool-use traces, shared memory across agents, memory provenance, memory isolation between agents."*
3. **"Multimodal, Visual, Video & Embodied Memory"** — *"long-term video memory, visual retrieval for agents, spatial memory for embodied systems, multimodal event memory."*
4. **"Neuroscience-Inspired & Cognitive Memory Models"** — *"complementary learning systems, episodic-to-semantic consolidation, replay, forgetting, abstraction, cognitive maps."*
5. **"Benchmarking & Evaluation"** — *"long-horizon memory benchmarks, temporal reasoning over past events, **memory update and deletion tests**, contradiction handling."*
6. **"Safety, Privacy & Security"** — *"memory poisoning, prompt injection through stored memories, sleeper memories, privacy leakage"*, *"alignment drift."*
7. **"User Control & Transparency"** — *"interfaces for inspecting, editing, deleting, and scoping memories; consent and access-control mechanisms"*, *"memory provenance"*, **"right-to-be-forgotten mechanisms"**, *"human-centered evaluations of memory transparency."*

## 1.3 Systems vs theory — the answer to the standing question
**No stated preference; theory is explicitly welcome.** Verbatim: submissions *"may present new architectures, benchmarks, datasets, evaluations, systems, theoretical perspectives, position papers, negative results, or interdisciplinary analyses."*
⛔ Fact, not judgment: **"negative results"** is a named accepted contribution type — V5's 15-item negatives ledger is inside the CFP's own vocabulary. And **"forget"/"forgetting"/"deleting"/"deletion"/"right-to-be-forgotten"** appear in **four of the seven topic headings' example lists** (1, 4, 5, 7).

## 1.4 Reviewer-pool predictors (invited speakers + organizers, verbatim from the site)
**Invited speakers:** Weiwen Liu (Shanghai Jiao Tong University) · **Tsendsuren Munkhdalai (Google)** · **Niloofar Mireshghallah (CMU / humans&)** · **Ali Behrouz (Cornell University)** · one TBA.
**Organizers:** Mario Fritz (CISPA) · Seong Joon Oh (KAIST) · Sahar Abdelnabi (ELLIS Institute Tübingen & MPI-IS) · Shawn Shen (Memories.ai & Univ. of Bristol) · Hugo D. Lopes (Google DeepMind) · Haritz Puerto (ELLIS Tübingen & MPI-IS) · Ivaxi Sheth (CISPA) · Seokwon Jung (KAIST).

⚠ **Three of these are directly load-bearing on V5's claims and are stated as facts, not predictions:**
- **Ali Behrouz** is first author of **Titans** (NeurIPS 2025), whose update is exactly *test-time gradient descent with momentum plus a **learned forget gate*** $\mathcal M_t=(1-\alpha_t)\mathcal M_{t-1}+S_t$ — the nearest published neighbour to V5's "damping as a priced retention dial". **V5 does not cite Titans at all** (V2 does). Record carried in `v2-cite-check` §6.
- **Tsendsuren Munkhdalai** is first author of **Infini-attention** (compressive memory with a bounded-memory update; arXiv:2404.07143) — the fixed-size-state retention line.
- **Niloofar Mireshghallah** works on memorization/privacy leakage and membership inference. **App E.5 (AUC 0.983 / 1.000, $d'$ 1560→79.6, the σ_obs-is-our-choice admission) will be read by an expert in that instrument.** The E.5 honesty posture is an asset here; the omitted TTL-flag laundering control (MF-9) is the exposure.

## 1.5 Submission mechanics not previously banked
- **Template:** *"Use the NeurIPS 2026 template"* (formatting-instructions zip linked from the site). ⚠ The 4-pp measurement in `scratch/v5-rebuild/maincount.log` was taken in generic `article`/10pt/1in — **re-measure in the NeurIPS 2026 style before freeze** (already the referee's §D caveat 1; now confirmed as the named template).
- **Page limits (re-confirms Add.20 verbatim):** *"Full-Length Papers: Up to 9 pages (excluding references and supplementary materials). Short Papers: Up to 4 pages (excluding references and supplementary materials)."*
- **Anonymization (re-confirms the charter's warning verbatim):** *"Submissions must be fully anonymized. This policy applies to any supplementary or linked material as well, including code."* ⇒ **SF-12 stands** (`chlu/core/placement.py`, `../CHLU-waitlist`, `../CHLU-c2w10`, `.claude/scratch/...` in App A).
- **Archival status:** *"non-archival"*; *"Accepted papers will be made public, but rejected submissions and reviews will not."*
- **Dual submission, verbatim:** *"The workshop will adopt a non-archival policy, welcoming ongoing and unpublished work, as well as papers under review or recently accepted at other venues"*; workshop submissions *"can be subsequently or concurrently submitted to other venues."* ⇒ ⭐ **This is a directly relevant fact for the Add.27/D2 M3 ruling: PALM explicitly permits concurrent submission elsewhere.** (The V2↔V5 overlap question is unaffected — that is about *two of our own papers*, not about dual submission — but the venue's stance removes one imagined hazard.)
- **Dates on the site:** paper submission **August 24, 2026**; notification **September 29, 2026**; workshop **December 12 or 13, 2026**. (Recorded as retrieved facts; timelines are the Head's, not tracked here.)

---

# PART 2 — the venue-native long-term-memory brief (13 works, for §4 / K.2)

**Answer first.** The literature a PALM reviewer expects contact with splits into (i) **external-memory agent architectures** (MemGPT, Mem0, Zep, Generative Agents), (ii) **decay/expiry policies inside them** (Generative Agents' 0.995 recency factor, MemoryBank's Ebbinghaus curve, Expire-Span's learned span, Titans' learned forget gate), and (iii) a **2026 forgetting-and-deletion evaluation wave** that is almost purpose-built for V5's two hooks (ForgetEval, Memora/FAMA, MemLeak, Ghost Vectors, Agentic Unlearning). **Every decay mechanism in (ii) is a heuristic score or a learned hyperparameter with no closed-form retention law, no read tolerance, and no temperature; every deletion mechanism in (i)/(iii) is a flag, a tombstone or an invalidation timestamp, and the 2026 wave's headline finding is that these leave the content physically recoverable.** That is exactly the pair of gaps V5 addresses — which makes the *absence* of these citations in §4 the single most damaging fixable defect for this venue, and their *presence* the cheapest available fit gain.

## 2.1 External-memory agent architectures (the "MemGPT class")
- **Packer, Wooders, Lin, Fang, Patil, Stoica & Gonzalez (2023), "MemGPT: Towards LLMs as Operating Systems", arXiv:2310.08560** (v1 12 Oct 2023; v2 12 Feb 2024; no venue on the record). Claim: *"virtual context management, a technique drawing inspiration from hierarchical memory systems in traditional operating systems that provide the appearance of large memory resources through data movement between fast and slow memory."*
  **V5 relation (contrast):** MemGPT pages content between tiers; eviction is a control-flow decision by the LLM. V5's forgetting is a *dynamical* property of the store with a computable half-life. Cite as the canonical external-memory agent, not as a competitor.
- **Chhikara, Khant, Aryan, Singh & Yadav (2025), "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory", arXiv:2504.19413** (28 Apr 2025). The update phase is an LLM tool call over four operations, verbatim: *"ADD for creation of new memories when no semantically equivalent memory exists; UPDATE for augmentation of existing memories with complementary information; **DELETE for removal of memories contradicted by new information**; and NOOP when the candidate fact requires no modification to the knowledge base."*
  **V5 relation (direct contrast, and the honest one):** this is the deployed form of "deletion" in the venue's own stack — an LLM-issued row operation over a discrete store. V5's E.2 result (deletion = set-minus, bit-exactly, with a *proof* that placement is a set function) is a statement of a kind Mem0 does not make and does not need — **and V5's flat-table trivial-substitute sentence is precisely the right way to cite it.**
- **Rasmussen, Paliychuk, Beauvais, Ryan & Chalef (2025), "Zep: A Temporal Knowledge Graph Architecture for Agent Memory", arXiv:2501.13956** (20 Jan 2025). Edges carry four timestamps: *"$t'_{created}$ and $t'_{expired}$ monitor when facts are created or invalidated in the system, while $t_{valid}$ and $t_{invalid}$ track the temporal range during which facts held true"*; and *"when the system identifies temporally overlapping contradictions, it invalidates the affected edges by setting their $t_{invalid}$ to the $t_{valid}$ of the invalidating edge."*
  ⭐ **V5 relation — the strongest single contrast in this brief:** the state of the art in production agent memory **invalidates rather than deletes**; nothing is removed, and the historical record is deliberately preserved. V5's claim ("the post-deletion store is byte-identical to the store that never held the item") is the *opposite design point*, and saying so in one sentence converts §3.3 from "a correctness test of a 2007 table" into a positioned contribution.
- **Park, O'Brien, Cai, Morris, Liang & Bernstein (2023), "Generative Agents: Interactive Simulacra of Human Behavior", arXiv:2304.03442** (UIST 2023). Retrieval score = recency + importance + relevance, with recency an *"exponential decay function over the number of sandbox game hours since the memory was last retrieved. **Our decay factor is 0.995**"*, and *"all α's are set to 1."*
  ⭐ **V5 relation (the cleanest "controllable decay" contrast):** the field's most-cited memory decay is **a scalar hyperparameter in a ranking heuristic** — no half-life, no tolerance $\Delta$, no temperature, no statement of what is lost. V5's $(\mu,\gamma,T)$ budget is the same functional role with a law attached. This is the sentence §4 is missing.

## 2.2 Decay / expiry / consolidation policies
- **Sukhbaatar, Ju, Poff, Roller, Szlam, Weston & Fan (2021), "Not All Memories are Created Equal: Learning to Forget by Expiring", ICML 2021, arXiv:2105.06548.** Learned per-memory expiration span in a Transformer. **V5 relation:** the nearest *per-item lifetime* prior; expiry is learned for efficiency, not specified for control, and there is no permanence class. (Record carried from `unlearning-recon`.)
- **Zhong, Guo, Gao, Ye & Wang (2024), "MemoryBank: Enhancing Large Language Models with Long-Term Memory", AAAI 2024, arXiv:2305.10250.** Ebbinghaus-curve exponential decay of memory strength. **V5 relation:** an explicitly *physics-analogy* decay in a deployed LLM memory — the closest venue-native rhetoric to V5's, and it is a heuristic score, not a retrieval amplitude. Cite it to show the program knows the analogy is already in the room.
- **Behrouz, Zhong & Mirrokni (2025), "Titans: Learning to Memorize at Test Time", NeurIPS 2025, arXiv:2501.00663.** $\mathcal M_t=(1-\alpha_t)\mathcal M_{t-1}+S_t$, $S_t=\eta_tS_{t-1}-\theta_t\nabla\ell$; the paper itself says this is *"similar to gradient descent with momentum."* **V5 relation:** damping and inertia are **learned gates** there and **read out and priced** ($\gamma,M,\mu$) here. ⭐ **Also: its first author is a PALM invited speaker.** Not citing it is a live risk.
- **Munkhdalai, Faruqui & Gopal (2024), "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention", arXiv:2404.07143** (no venue on the record — cite as a preprint). Compressive memory with bounded parameters. **V5 relation:** the fixed-size-state retention line; identity-class cite, invited-speaker relevance.

## 2.3 The 2026 forgetting/deletion evaluation wave (the part V5 can actually claim contact with)
- **Yang (2026), "Control-Plane Placement Shapes Forgetting: An Architectural Study of Agent Memory Across Thirteen System Configurations", arXiv:2606.15903** (v1 14 Jun, v2 16 Jun 2026; 25 pp; MIT-licensed code/benchmark). Verbatim: *"**Production failures are predominantly forgetting failures rather than recall failures, yet existing benchmarks measure only recall.**"* Introduces **ForgetEval** (1000 templated + 385 adversarial cases); a mutation-time hook recovers intent-aware deletion (78–85%) and lifts overall coverage to 91.7–93.2%.
  ⭐ **V5 relation — this is the single best framing citation available.** It is the venue's own literature saying, with numbers, that *forgetting is the failure mode nobody measures*. One clause in §1 citing it converts V5's opening from "forgetting is usually a side effect" (an assertion) into a positioned claim.
- **Uddin, Shubham, Blanco, Baral & Wang (2026), "From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents", arXiv:2604.20006** (Memora; **FAMA** = Forgetting-Aware Memory Accuracy). Finding: *"failures to forget outdated memory"* account for **64%** of sampled recommendation errors. (Carried from `unlearning-recon`, [VERIFIED via HTML fetch] there.) **V5 relation:** same framing lever, benchmark side.
- **Chakraborttii, García Alvarado, Abdulofizova & Dwivedi (2026), "Ghost Vectors: Soft-Deleted Embeddings Remain Reconstructible in HNSW Vector Databases", arXiv:2606.18497** (16 Jun 2026; cs.CR/cs.AI/cs.DB; comments *"13 pages, 5 figures, 12 tables. Prepared for submission"*). Soft-deleted (tombstoned) vectors remain physically present in ChromaDB, FAISS and Weaviate after the API confirms deletion, and are reconstructible from raw index files via Vec2Text: 25.5% of exact names / 46.4% of locations (Wikipedia BLP), 100% of structured patient markers (Synthea), 99% identity recovery from face embeddings. Proposed fix: *Epoch Key Rotation* (encrypt, discard key on delete) → 0% PII recovery.
  ⭐ **V5 relation — this is the citation K.2 already leans on and does not have.** K.2 currently says *"the report that soft-deleted vectors in graph ANN indexes (which delete lazily via tombstones) remain physically reconstructible — the latter narrowing the 'a vector store deletes just as exactly' preemption to *flat/brute-force* stores."* That sentence is **exactly supported** by this paper and must carry its identifier.
- **Wang & Zhang (2026), "MemLeak: Diagnosing Information Leaks in Multimodal Agent Memory", arXiv:2606.29788** (29 Jun 2026; 23 pp). Verbatim: *"When a multimodal AI agent is asked to forget a fact, current memory systems usually delete the text entry and report success. We find that the fact can remain recoverable from retained user images…"* Deletion cascade: direct probing <1%, retained correlated text 18.3%, retained images 12.0% (0.0% blind baseline, 0.3% FPR); 47% of image leaks not text-recoverable.
  **V5 relation (and it is a *defensive* cite):** MemLeak is the venue-native statement of V5's own **encoder-excluded** caveat — "the store deleted it; the system did not". Citing it makes §3.3's *"the frozen encoder and any residue of past writes in a learned landscape are separate channels"* read as engagement with the literature rather than as a hedge.
- **Wang, Wang, Wang, Cong, Yu, Yin, Han & Wei (2026), "Agentic Unlearning: When LLM Agent Meets Machine Unlearning", arXiv:2602.17692.** Deletes from **both** weights and persistent agent memory; names the **"parameter–memory backflow"** problem (retrieval reactivates forgotten content); dependency-aware pruning with blocklists + reference counting. (Carried from `unlearning-recon`, [VERIFIED via HTML fetch] there.) **V5 relation:** the closest live competitor to "delete from the store, not the weights", and the named source of the φ-retains-the-datum objection §3.3 already concedes.
- **Garg, Kolhe, Song & Zhao (2026), "MemFail: Stress-Testing Failure Modes of LLM Memory Systems", arXiv:2605.26667** (26 May 2026). Formalizes memory systems as *"the composition of three canonical operations — summarization, storage, and retrieval"* and builds five adversarial datasets per operation. **V5 relation:** optional; use only if §4 needs a second "black-box benchmarks hide mechanism" citation. ⚠ Its abstract does **not** claim deletion/forgetting coverage — do not cite it for that.

## 2.4 Recommended minimum set for a 4-pp short (Advisor/writer's call, stated as a recommendation)
If only **four** venue-native cites fit: **Yang 2026 (ForgetEval)** in §1 for the framing · **Park et al. 2023 (0.995)** in §3.1/§4 for the decay contrast · **Rasmussen et al. 2025 (Zep invalidation)** in §3.3/§4 for the deletion contrast · **Chakraborttii et al. 2026 (Ghost Vectors)** in K.2 where the un-cited lean already is. Add **Behrouz et al. 2025 (Titans)** if a fifth fits (invited-speaker exposure). The rest belong in K.2's extended discussion, which is supplementary and free.

---

# PART 3 — V5's citation records

Legend: **✔ verified fresh this pass** · **↻ carried from `v2-cite-check` (not re-verified, per task)** · **⛔ trap / never-copy** · **FAIL/PARTIAL** = a draft content-lean its source does not support.

## 3.1 Carried forward from `v2-cite-check` (records are in that report; do not re-derive)
Mo 2026 (arXiv:2605.03338, **single author, preprint, "at least dim(G/H)"**) ↻ · Rusch & Mishra 2021a/b ↻ · Hairer, Lubich & Wanner 2006 ↻ · Fischer & Igel 2010 + 2011 ↻ · Nijkamp et al. 2020 ↻ · Golubitsky, Stewart & Schaeffer 1988 ↻ · Krupa 1990 ↻ · Decelle, Furtlehner & Seoane 2021 ↻ · Agoritsas, Catania, Decelle & Seoane 2023 ↻ · Toledo-Marin et al. 2025 ↻ · Kong et al. 2024 ↻ (**not cited in V5** — no action) · Hinton et al. 1995 ↻ · Tieleman 2008 ↻ · Hochreiter & Schmidhuber 1997 ↻ · Rusch et al. 2022 (LEM) ↻ · Di Bernardo et al. 2025 ↻ · Iqbal, Keller, Song, Miyato & Welling 2026 ↻ · Jawahar & Pierini 2026 ↻ · McLachlan & Perlmutter 2001 ↻ (needed for the U14 fix) · Bhatt, Floyd & Moore 2016 ↻ (optional).
⚠ **Two of these carry uncorrected V5 usage errors** — reconciliation items 3 (Mo) and 4 (HLW).

## 3.2 Verified fresh — the history-independence set

### 1. Blelloch & Golovin (FOCS'07) — the load-bearing attribution ✔
**Record ✔** Guy E. Blelloch, Daniel Golovin, "Strongly History-Independent Hashing with Applications", **48th Annual IEEE Symposium on Foundations of Computer Science (FOCS 2007), pp. 272–282**, DOI **10.1109/FOCS.2007.36** (DBLP; full text read from the authors' PDF at cs.cmu.edu/~dgolovin/papers/focs07.pdf).
**Usage-check — every V5 lean PASSES, and the paper is *more* supportive than the draft claims:**
| V5 says | source says | verdict |
|---|---|---|
| "the **Blelloch–Golovin (FOCS'07) stable-matching table** with a global priority order" (§3.3, E.2, K.2 ×7 sites) | *"Our framework reveals a subtle connection between history independent hashing and the **Gale-Shapley stable marriage algorithm** [9]"*; *"interpret the keys as men and the slots of the hash table as women"*; Thm 3.1 [9]: *"Every execution of the Gale-Shapley algorithm results in the same stable matching."* | **PASS** |
| "a **global priority order** and a metric-induced probe order" | *"we fix $p=(1+\epsilon)n$ … and a total ordering on the keys… each slot prefers $\mathbf k$ to $\mathbf k'$ if $\mathbf k>\mathbf k'$"* — i.e. slot preferences ARE one global total order on keys | **PASS — exact** |
| "the **fix-up cascade** is theirs" | `DELETE(key k)` pseudocode (Fig. 1): *"While (NEXT(x) is not null) Set y = NEXT(x); Set A[x] = A[y]; Set x = y; Set A[x] to be empty"* | **PASS on substance.** ⚠ BG never use the phrase "fix-up cascade"; it is our name for their `NEXT`-chain. Safe (V5 already says "our fix-up cascade" is a never-quote), but the attribution sentence should read "their delete-time repair (the `NEXT`-chain of Fig. 1)". |
| "Hartline et al. … cited for the **definitional equivalence only**, never inside a proof, since **his** theorem requires a *reversible* structure and our amplitude layer is not" (E.2) | *"Our definition of strong history independence differs from that of Naor and Teague, the two definitions were proved equivalent by Hartline et al. [10] **for reversible data structures** (i.e., those for which there always exists some sequence of operations which returns the data structure to its initial state)"* | **PASS — and it is an unusually precise piece of citation hygiene.** ⚠ **"his" → "their"**: Hartline et al. is **five** authors. |
| SHI as "canonical representation" | Def. 2.2 verbatim: *"A **reversible** data structure is strongly history independent (SHI) if it has canonical representations up to initial randomness. That is, for each sequence of initial random bits and for each state of the data structure, there is a unique memory representation."* | **PASS** |

### 2. Micciancio (STOC'97) — ⛔ **the misattribution** ✔
**Record ✔** Daniele Micciancio, "Oblivious Data Structures: Applications to Cryptography", **STOC '97, pp. 456–464**, DOI **10.1145/258533.258638**.
**Usage FAIL (reconciliation item 2).** BG's own §Previous Work: *"**Micciancio [13] defined the notion of *oblivious* data structures** in which the *pointer structure* reveals nothing about its history. **Oblivious data structures need not have canonical pointer representation** since they only require that the probability distribution over possible pointer representations is independent of the sequence of operations."* And, separately: *"The two main notions of history independence … **were advanced by Naor and Teague [14]**, and further studied by Hartline et al."*; *"**Canonical representations were studied by Snyder [22], Sundar and Tarjan [23], and Andersson and Ottmann [3]**."*
⇒ V5's *"strongly history-independent (uniquely represented) data structures were introduced by Micciancio (STOC'97) and Naor & Teague (STOC'01)"* is **wrong twice**: Micciancio's notion is the *weaker, distributional* one and is explicitly not canonical; unique representation predates both (Snyder, FOCS'77).
**Recommended replacement sentence (drop-in, 41 words):** *"Uniquely represented (canonical) data structures date to Snyder (1977); obliviousness — a distributional condition on the memory representation — is due to Micciancio (1997); the weak and strong history-independence notions used here are Naor & Teague's (2001), characterised by Hartline et al. (2005) and realised for open-addressed hash tables by Blelloch & Golovin (2007)."*
⚠ Note the countervailing datum, stated so the Hub can weigh it: **Buchbinder & Petrank's own abstract says "History independent data structures, presented by Micciancio"** — so the loose attribution exists in the literature. It is still not what BG say, and BG is the paper V5 builds on.

### 3. Naor & Teague (STOC'01) ✔
**Record ✔** Moni Naor, Vanessa Teague, "Anti-persistence: History Independent Data Structures", **STOC '01, pp. 492–501**, DOI **10.1145/380752.380844**. ⚠ DBLP indexes only the **IACR ePrint 2001/036** version; the STOC record is confirmed via the ACM DL DOI and BG's reference list [14].

### 4. Hartline, Hong, Mohr, Pentney & Rocke (2005) ✔
**Record ✔** "Characterizing History Independent Data Structures", **Algorithmica 42(1):57–74 (2005)**, DOI **10.1007/s00453-004-1140-z**; conference version **ISAAC 2002, pp. 229–240**, DOI 10.1007/3-540-36136-7_21. ⛔ **Five authors — never "Hartline's theorem"; write "Hartline et al.'s".**

### 5. Blelloch, Golovin & Vassilevska (SWAT 2008) ✔
**Record ✔** Guy E. Blelloch, Daniel Golovin, Virginia Vassilevska, "Uniquely Represented Data Structures for Computational Geometry", **SWAT 2008, pp. 17–28**, DOI **10.1007/978-3-540-69903-3_4** (LNCS 5124). ⚠ Springer returned a 303 SSO redirect; record taken from the **DBLP API** (labelled fallback). V5's use is an identity/priority-disclaimer cite (App J #8, K.2) — nothing to usage-check beyond the record.

### 6. Buchbinder & Petrank (CRYPTO'03) — ✔ verified verbatim, **needs a scope clause**
**Record ✔** Niv Buchbinder, Erez Petrank, "Lower and Upper Bounds on Obtaining History Independence", **CRYPTO 2003, LNCS 2729, pp. 445–462**, DOI **10.1007/978-3-540-45146-4_26**; journal version **Information and Computation 204(2):291–337 (2006)**, DOI 10.1016/j.ic.2005.11.001.
**Usage-check (K.2, E.2, §3.3):** V5 says *"a literature that proves strong history independence can cost an exponential slowdown (Buchbinder–Petrank)"*. Abstract, **verbatim** (author-hosted Technion PDF, page 1 read directly): *"**The gap we obtain is exponential: some operations may be executed in logarithmic time (or even in constant time) with the weaker definition, but require linear time with the stronger definition.**"* — **PASS on the word "exponential."**
⚠ **But the axis is not V5's axis, and a referee in this area will say so.** BP's separation is **weak-HI vs strong-HI**, *"for comparison based algorithms"*, *"for a large class of data structures, including, for example, the heap and the queue abstract data structures."* V5's "negative price" compares canonical placement against a **history-*dependent* stochastic relocation rule** — a different comparison. **Recommended scoping (18 words, drop into K.2 and E.2's attribution block):** *"…where strong history independence is known to cost up to an exponential slowdown **relative to the weak notion, for heaps and queues in a comparison-based model** (Buchbinder & Petrank 2003), here it is free…"* Without it, the sentence is a fair paraphrase with an unstated baseline; with it, it is unimpeachable.

## 3.3 Verified fresh — the unlearning set

### 7. Guo, Goldstein, Hannun & van der Maaten (2020) — ⛔ **the §-number error** ✔
**Record ✔** "Certified Data Removal from Machine Learning Models", **ICML 2020, PMLR 119:3832–3842**; arXiv:1911.03030 (comments: *"Accepted to ICML 2020"*).
**Usage FAIL (reconciliation item 1).** Read page-by-page from the PMLR PDF:
- **§2 is "Certified Removal"**; **§3 is "Removal Mechanisms"** (§3.1 "Linear Classifiers").
- Eq. **(1)** appears on **p.1, inside §2**, and is the **ε-only** condition, verbatim: *"Given $\epsilon>0$, we say that removal mechanism $M$ performs $\epsilon$-certified removal ($\epsilon$-CR) for learning algorithm $A$ if $\forall\mathcal T\subseteq\mathcal H,\mathcal D\subseteq\mathcal X,\mathbf x\in\mathcal D$: $e^{-\epsilon}\le\frac{P(M(A(\mathcal D),\mathcal D,\mathbf x)\in\mathcal T)}{P(A(\mathcal D\setminus\mathbf x)\in\mathcal T)}\le e^{\epsilon}$."* **(1)**
- The **$(\epsilon,\delta)$** notion is the **next, unnumbered** display on p.2: *"We also define a more relaxed notion of $(\epsilon,\delta)$-certified removal for $\delta>0$ if $\forall\mathcal T\subseteq\mathcal H,\mathcal D\subseteq\mathcal X,\mathbf x\in\mathcal D$: $P(M(A(\mathcal D),\mathcal D,\mathbf x)\in\mathcal T)\le e^{\epsilon}P(A(\mathcal D\setminus\mathbf x)\in\mathcal T)+\delta$, and $P(A(\mathcal D\setminus\mathbf x)\in\mathcal T)\le e^{\epsilon}P(M(A(\mathcal D),\mathcal D,\mathbf x)\in\mathcal T)+\delta$."* Eq. (2) is differential privacy; Eq. (3) is the Newton update.
⇒ **Both halves of V5's citation are off:** the section is **§2**, and Eq. (1) is **not** the $(\epsilon,\delta)$ statement.
**Recommended replacement (drop-in, both §4/E.7 and K.2):** *"*Certified* removal is introduced by Guo et al. (ICML 2020) in §2 as $\epsilon$-certified removal, Eq. (1), with an unnumbered $(\epsilon,\delta)$ relaxation stated immediately after — neither is a numbered Definition, and we cite it that way."*
⚠ This also **retires N131's "Guo §3 Eq. (1)" fence and the ⛔"Guo Def. 1/2" ban's justification text.** The Def-1/2 ban is *correct* (there are no such definitions); the §-number in the replacement rule is not.

### 8. Ginart, Guan, Valiant & Zou (2019) ✔ — **Def. A.5 PASSES**
**Record ✔** "Making AI Forget You: Data Deletion in Machine Learning", **NeurIPS 2019**, arXiv:1907.05012 (arXiv page states *"To appear in NeurIPS 2019"*).
**Usage-check (E.7, K.2: "Def. A.5, amortized time"):** Def. A.5 verbatim: *"Given some fractional power scaling $m=\Theta(n^\alpha)$, we say an algorithm $\mathcal A$ is $\alpha$-deletion efficient if it runs Algorithm 3 in **amortized time** $O(n^{1-\alpha})$."* Neighbours: A.1 Learning Algorithm · A.2 Data Deletion Operation · A.3 Robust Data Deletion Operation · A.4 Online Data Deletion (Average-Case) · A.6 Approximate deletion. **PASS — exact.**

### 9. Sekhari, Acharya, Kamath & Suresh (2021) ✔ — **Def. 3 PASSES**
**Record ✔** "Remember What You Want to Forget: Algorithms for Machine Unlearning", **NeurIPS 2021, pp. 18075–18086**; arXiv:2103.03279.
**Usage-check (E.7, K.2: "Def. 3 — capacity at fixed excess risk"):** Def. 3 verbatim: *"Let $\epsilon,\delta\ge0$. Let $S$ be a dataset of size $n$ drawn i.i.d. from $\mathcal D$… For a pair of learning and unlearning algorithms $A,\bar A$ that are $(\epsilon,\delta)$-unlearning, the **deletion capacity** $m^{A,\bar A}_{\epsilon,\delta}(d,n)$ is defined as the maximum number of samples $U$ that can be unlearnt, while still ensuring an excess population risk of **0.01**."* **PASS** — and the fixed risk level is literally 0.01, so V5's "at fixed excess risk" is right.

### 10. Bourtoule et al. (2021) — SISA ✔ — **Def. III.1 PASSES**
**Record ✔** Lucas Bourtoule, Varun Chandrasekaran, Christopher A. Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, Nicolas Papernot, "Machine Unlearning", **42nd IEEE Symposium on Security and Privacy (S&P) 2021**; arXiv:1912.03817.
**Usage-check (K.2: "whose Def. III.1 is distributional and model-level"):** Def. III.1 verbatim: *"…Let $\mathbb D_M$ denote the distribution of models learned using mechanism $M$ on $\mathcal D'$ and then unlearning $d_u$. Let $\mathbb D_{real}$ be the distribution of models learned using $M$ on $\mathcal D$. The mechanism $M$ facilitates unlearning when **these two distributions are identical**."* **PASS — distributional and model-level, exactly as V5 says.**

### 11–14. SILO · PALL · Ticketed L–U · MUSE · CURE4Rec
- **SILO ✔↻** Min, Gururangan, Wallace, Shi, Hajishirzi, Smith & Zettlemoyer, "SILO Language Models: Isolating Legal Risk in a Nonparametric Datastore", **ICLR 2024 (spotlight)**, arXiv:2308.04430. Abstract verbatim (verified in `unlearning-recon`): *"…enables data producers to opt out from the model by removing content from the store."* **V5's "delete by isolation" grouping is fair.**
- **PALL ✔↻** Özdenizci, Rueckert & Legenstein, "Privacy-Aware Lifelong Learning", **ICLR 2025**, arXiv:2505.10941. Task-specific sparse subnetworks + episodic rehearsal ⇒ **"exact task unlearning"** in continual learning. ⚠ **Scope caveat V5 must not blur: PALL is exact *task* unlearning, not per-item.** V5 groups it under "delete by isolation" — correct, but the per-task/per-item distinction is the differentiator worth one clause.
- **Ticketed L–U ✔** Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Ayush Sekhari, Chiyuan Zhang, "Ticketed Learning–Unlearning Schemes", **COLT 2023, PMLR 195:5110–5139**; arXiv:2306.15744.
- **MUSE ✔** Weijia Shi, Jaechan Lee, Yangsibo Huang, Sadhika Malladi, Jieyu Zhao, Ari Holtzman, Daogao Liu, Luke Zettlemoyer, Noah A. Smith, Chiyuan Zhang, "MUSE: Machine Unlearning Six-Way Evaluation for Language Models", **ICLR 2025**; arXiv:2407.06460 (2024). (DBLP.)
- **CURE4Rec ✔** Chaochao Chen, Jiaming Zhang, Yizhao Zhang, Li Zhang, Lingjuan Lyu, Yuyuan Li, Biao Gong, Chenggang Yan, "CURE4Rec: A Benchmark for Recommendation Unlearning with Deeper Influence", **NeurIPS 2024**; arXiv:2408.14393. (DBLP.)

### 15. SF-10's competing work — **RECOVERED, with a caveat** ✔/⚠
**Record ✔** Sonia Laguna, Jorge da Silva Goncalves, Moritz Vandenhirtz, Alain Ryser, Irene Cannistraci, Julia E. Vogt, **"Rethinking Machine Unlearning: Models Designed to Forget via Key Deletion"**, **arXiv:2603.15033** (v1 16 Mar 2026, v2 24 Mar, v3 2 Apr 2026; cs.LG). System name **MUNKEY**. Abstract verbatim: *"We propose **unlearning by design**… We instantiate this idea with Machine UNlearning via KEY deletion (MUNKEY), a **memory augmented transformer** that decouples instance-specific memorization from model weights. Here, unlearning corresponds to removing the instance-identifying key, enabling direct zero-shot forgetting **without weight updates or access to the original samples or labels**."*
**Numbers, from the paper's own tables (HTML v3):** MIA AUROC on the forget set **51.40 ± 0.44** (CIFAR-10, 10% forget) and **51.19 ± 1.19** (DermaMNIST, 10%); **Average Gap 0.56 ± 0.21** (CIFAR-10, 10%), where Average Gap = mean absolute difference across four metrics (test acc., retain acc., forget acc., MIA) **against the retrained oracle**.
⛔ **Two traps.**
(a) **Venue/presentation type is NOT VERIFIABLE.** The arXiv record carries **no comments field and no journal-ref**; Semantic Scholar's `publicationVenue` is **arXiv.org** with `publicationTypes: ["JournalArticle"]`; the paper's own LaTeX keyword line reads *"Machine Learning, ICML"* (the ICML template default — **this is almost certainly the source of the C2W2 "ICML 2026" error N168 already corrected**). OpenReview forum **gGH3Xp1lHR** exists but every access route is bot-challenged (`/forum`, `/pdf`, `api.openreview.net`, `api2.openreview.net` all return a verification page or 302). One search-engine summary states **"ICLR 2026 Workshop RSI … Poster"**; N168 states **"(oral)"**. RSI = *ICLR 2026 Workshop on AI with Recursive Self-Improvement* (verified separately). **Both presentation-type claims are single-sourced and mutually contradictory; I could not verify either from a primary source.**
(b) **The 0.56 ± 0.21 is the Average-Gap-to-oracle, not a membership gap.** V5's *"evaluated on the same membership instrument (AUROC → 0.5; not exact, gap 0.56 ± 0.21)"* reads as though 0.56 is a membership quantity. It is not.
**Recommended N168-compliant, defensible form (drop-in for E.7 and K.2):** *"…and a 2026 preprint publishes unlearning-by-design for a memory-augmented transformer, evaluated on the same membership instrument (forget-set MIA AUROC $\to$ 51.4/51.2%); it is not exact — its average gap to the retrained oracle is $0.56\pm0.21$ (Laguna et al., 2026, arXiv:2603.15033)."* ⚠ **Do not print a presentation type until someone with an OpenReview session confirms it.** If the Head wants the workshop framing, *"an ICLR-2026 workshop paper"* is the most that is defensible, and even that rests on a single search-engine assertion.

## 3.4 Verified fresh — the physics set, and two owed citations

### 16. Minami & Hidaka (2018) ✔ — **PASS, including the "two Noether-charge types" lean**
**Record ✔** Yuki Minami, Yoshimasa Hidaka, "Spontaneous symmetry breaking and Nambu-Goldstone modes in dissipative systems", **Phys. Rev. E 97, 012130 (2018)**, DOI **10.1103/PhysRevE.97.012130**; arXiv:1509.05042 (v1 2015, v2 2018).
**Usage-check (§4 l.71, K.2 l.271, B.7):** abstract verbatim: *"there exist **two types of NG modes** in dissipative systems corresponding to type-A and type-B… we show that the **type-A NG modes in the dissipative system are diffusive modes, while they are propagating modes in Hamiltonian systems**… this difference is caused by the existence of **two types of Noether charges**, $Q_R^\alpha$ and $Q_A^\alpha$… the NG modes are propagating modes if $Q_R^\alpha$ are conserved, while those are diffusive modes if they are not conserved."* **PASS on every clause, including "two Noether-charge types."**

### 17. Hidaka & Minami (2020) — ⛔ **AUTHOR ORDER IS REVERSED, and it is four types, not two** ✔
**Record ✔** **Yoshimasa Hidaka, Yuki Minami** (Hidaka **first**), "Spontaneous symmetry breaking and Nambu–Goldstone modes in open classical and quantum systems", **Prog. Theor. Exp. Phys. 2020(3):033A01**, DOI **10.1093/ptep/ptaa005**; arXiv:1907.08241.
⛔ **Trap:** V5 writes **"Minami & Hidaka (2018, 2020)"** at §4 l.71 and K.2 l.271. The **2020** paper is **Hidaka & Minami**. Correct form: *"Minami & Hidaka (2018); Hidaka & Minami (2020)"*.
⚠ **Second, smaller usage note:** the 2020 paper classifies NG modes into **four** types — abstract verbatim: *"we classify the Nambu-Goldstone modes into four types: type-A propagation, type-A diffusion, type-B propagation, and type-B diffusion modes"* — so a "two types" sentence should be sourced to **2018**, not to the pair.

### 18. Du & Mordatch (2019) ✔ — **PASS**
**Record ✔** Yilun Du, Igor Mordatch, "Implicit Generation and Modeling with Energy Based Models", **NeurIPS 2019, pp. 3603–3613**; arXiv:1903.08689. ⛔ **Trap: the arXiv v1 title is different** — *"Implicit Generation and **Generalization** in Energy-Based Models"*. Cite the NeurIPS title.
**Usage-check (App H(d): "the energy-magnitude regularization of Du & Mordatch (2019), used there for partition-function stability rather than to preserve a structural prior"):** §3.3 verbatim: *"We found it useful to weakly **L2 regularize energy magnitudes for both positive and negative samples** during training, as otherwise while the difference between positive and negative samples was preserved, the actual values would fluctuate to numerically unstable values"*; and *"Both forms of regularization also serve to ensure that **partition function is integrable** over the domain of the input."* Algorithm 1 term: $\alpha(E_\theta(\mathbf x_i^+)^2+E_\theta(\mathbf x_i^-)^2)$. **PASS — both halves ("partition function" and "stability") are literally in the source.**

### 19–21. Owed citations (named laws/theorems with no reference in V5)
- **Gell-Mann–Oakes–Renner** (App A.0 l.87, *"a Gell-Mann–Oakes–Renner law in spectral mass"*): M. Gell-Mann, R. J. Oakes, B. Renner, "Behavior of Current Divergences under $SU_3\times SU_3$", **Phys. Rev. 175, 2195–2199 (1968)**, DOI **10.1103/PhysRev.175.2195** ✔.
- **Coleman** (App G heading + first line): Sidney Coleman, "There are no Goldstone bosons in two dimensions", **Comm. Math. Phys. 31(4):259–264 (1973)**, DOI **10.1007/BF01646487** ✔.
- **Mermin–Wagner** (same site): N. D. Mermin, H. Wagner, "Absence of Ferromagnetism or Antiferromagnetism in One- or Two-Dimensional Isotropic Heisenberg Models", **Phys. Rev. Lett. 17, 1133–1136 (1966)**, DOI 10.1103/PhysRevLett.17.1133 ✔ (⚠ with **Erratum, PRL 17, 1307 (1966)**; and note the literature increasingly writes **"Hohenberg–Mermin–Wagner"**).
⚠ **App G's argument is load-bearing** (it is the paper's answer to "you claim a $T=0$ latch in 0+1 dimensions") and currently rests on two uncited eponyms. Cheapest possible fix; highest per-word credibility gain in the appendix.

## 3.5 Sweep completeness
Per-file grep over `draft.md` (296 lines), positive-controlled (`et al.|Blelloch|…` → 18 hits; year pattern → 19 hits; `Coleman|Mermin|Gell-Mann|TTL|…` → 9 hits, all inspected). **Beyond the referee's MF-6 list of ~30, the sweep found three additional citable items with no reference: GMOR, Coleman, Mermin–Wagner** (§3.4 above). Non-citation proper nouns excluded: splitmix64, Wendland kernel, LSTM/LEM/coRNN as system names, JAX/equinox/numpy/scipy/tectonic. **`Gale–Shapley` does not appear in the draft** — an option worth considering, since naming it makes the BG attribution unmissable in one word.

## 3.6 BibTeX (App-Q house pattern — caveats in `note`, retrieval date on every entry)

```bibtex
% ---------- history independence ----------
@inproceedings{blelloch2007shi,
  title={Strongly History-Independent Hashing with Applications},
  author={Blelloch, Guy E. and Golovin, Daniel},
  booktitle={Proceedings of the 48th Annual IEEE Symposium on Foundations of Computer Science (FOCS)},
  pages={272--282}, year={2007}, doi={10.1109/FOCS.2007.36},
  note={The table is a Gale--Shapley stable matching: keys=men (probe order), slots=women (one GLOBAL total order on keys). Delete-time repair = the NEXT-chain of Fig. 1 (they do not call it a ``fix-up cascade''). Def. 2.2 defines SHI for REVERSIBLE structures. Retrieved 2026-08-19.}}

@inproceedings{micciancio1997oblivious,
  title={Oblivious Data Structures: Applications to Cryptography},
  author={Micciancio, Daniele},
  booktitle={Proceedings of the 29th Annual ACM Symposium on Theory of Computing (STOC)},
  pages={456--464}, year={1997}, doi={10.1145/258533.258638},
  note={⛔ NEVER cite as the origin of SHI/unique representation. Micciancio defines OBLIVIOUS data structures, which per Blelloch--Golovin ``need not have canonical pointer representation''. Retrieved 2026-08-19.}}

@inproceedings{snyder1977unique,
  title={On Uniquely Representable Data Structures},
  author={Snyder, Lawrence},
  booktitle={18th Annual IEEE Symposium on Foundations of Computer Science (FOCS)},
  pages={142--146}, year={1977},
  note={The actual origin of canonical/uniquely-represented data structures, per Blelloch--Golovin's Previous Work section. Record taken from BG's reference list [22]; DOI not independently verified. Retrieved 2026-08-19.}}

@inproceedings{naor2001antipersistence,
  title={Anti-persistence: History Independent Data Structures},
  author={Naor, Moni and Teague, Vanessa},
  booktitle={Proceedings of the 33rd Annual ACM Symposium on Theory of Computing (STOC)},
  pages={492--501}, year={2001}, doi={10.1145/380752.380844},
  note={THE source of the weak/strong history-independence notions. DBLP indexes only IACR ePrint 2001/036; STOC record via ACM DL DOI + BG ref [14]. Retrieved 2026-08-19.}}

@article{hartline2005characterizing,
  title={Characterizing History Independent Data Structures},
  author={Hartline, Jason D. and Hong, Edwin S. and Mohr, Alexander E. and Pentney, William R. and Rocke, Emily},
  journal={Algorithmica}, volume={42}, number={1}, pages={57--74}, year={2005},
  doi={10.1007/s00453-004-1140-z},
  note={FIVE authors -- ``Hartline et al.'s theorem'', never ``his''. Conference version ISAAC 2002, pp. 229--240. The WHI/SHI equivalence holds for REVERSIBLE structures only. Retrieved 2026-08-19.}}

@inproceedings{blelloch2008geometry,
  title={Uniquely Represented Data Structures for Computational Geometry},
  author={Blelloch, Guy E. and Golovin, Daniel and Vassilevska, Virginia},
  booktitle={11th Scandinavian Workshop on Algorithm Theory (SWAT)}, series={LNCS}, volume={5124},
  pages={17--28}, year={2008}, doi={10.1007/978-3-540-69903-3_4},
  note={Record via DBLP API (link.springer.com returned a 303 SSO redirect). Retrieved 2026-08-19.}}

@inproceedings{buchbinder2003bounds,
  title={Lower and Upper Bounds on Obtaining History Independence},
  author={Buchbinder, Niv and Petrank, Erez},
  booktitle={Advances in Cryptology --- CRYPTO 2003}, series={LNCS}, volume={2729},
  pages={445--462}, year={2003}, doi={10.1007/978-3-540-45146-4_26},
  note={Journal version: Information and Computation 204(2):291--337 (2006), doi 10.1016/j.ic.2005.11.001. ⚠ SCOPE: the exponential gap is WEAK-vs-STRONG HI, in a COMPARISON-BASED model, for heaps and queues. Verbatim: ``The gap we obtain is exponential: some operations may be executed in logarithmic time (or even in constant time) with the weaker definition, but require linear time with the stronger definition.'' Retrieved 2026-08-19.}}

% ---------- unlearning ----------
@inproceedings{guo2020certified,
  title={Certified Data Removal from Machine Learning Models},
  author={Guo, Chuan and Goldstein, Tom and Hannun, Awni and van der Maaten, Laurens},
  booktitle={Proceedings of the 37th International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={119}, pages={3832--3842}, year={2020},
  note={⛔ Eq. (1) is in SECTION 2 (``Certified Removal''), NOT section 3 (``Removal Mechanisms''), and Eq. (1) is the EPSILON-only condition; the (eps,delta) relaxation is the UNNUMBERED display immediately after. There are no numbered Definitions 1/2. arXiv:1911.03030. Retrieved 2026-08-19.}}

@inproceedings{ginart2019making,
  title={Making AI Forget You: Data Deletion in Machine Learning},
  author={Ginart, Antonio and Guan, Melody Y. and Valiant, Gregory and Zou, James},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2019},
  note={arXiv:1907.05012. Def. A.5: ``alpha-deletion efficient if it runs Algorithm 3 in amortized time O(n^{1-alpha})''. Retrieved 2026-08-19.}}

@inproceedings{sekhari2021remember,
  title={Remember What You Want to Forget: Algorithms for Machine Unlearning},
  author={Sekhari, Ayush and Acharya, Jayadev and Kamath, Gautam and Suresh, Ananda Theertha},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={18075--18086}, year={2021},
  note={arXiv:2103.03279. Def. 3 = deletion capacity at excess population risk 0.01. Retrieved 2026-08-19.}}

@inproceedings{bourtoule2021machine,
  title={Machine Unlearning},
  author={Bourtoule, Lucas and Chandrasekaran, Varun and Choquette-Choo, Christopher A. and Jia, Hengrui and Travers, Adelin and Zhang, Baiwu and Lie, David and Papernot, Nicolas},
  booktitle={42nd IEEE Symposium on Security and Privacy (S\&P)}, year={2021},
  note={SISA. arXiv:1912.03817. Def. III.1 is distributional/model-level: the two model DISTRIBUTIONS must be identical. Retrieved 2026-08-19.}}

@inproceedings{min2024silo,
  title={SILO Language Models: Isolating Legal Risk in a Nonparametric Datastore},
  author={Min, Sewon and Gururangan, Suchin and Wallace, Eric and Shi, Weijia and Hajishirzi, Hannaneh and Smith, Noah A. and Zettlemoyer, Luke},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2024},
  note={Spotlight. arXiv:2308.04430. ``enables data producers to opt out from the model by removing content from the store''. Record carried from outputs/unlearning-recon.md. Retrieved 2026-08-19.}}

@inproceedings{ozdenizci2025pall,
  title={Privacy-Aware Lifelong Learning},
  author={{\"O}zdenizci, Ozan and Rueckert, Elmar and Legenstein, Robert},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2025},
  note={PALL. arXiv:2505.10941. ⚠ exact TASK unlearning (sparse subnetworks + episodic rehearsal), not per-item. Record carried from outputs/unlearning-recon.md. Retrieved 2026-08-19.}}

@inproceedings{ghazi2023ticketed,
  title={Ticketed Learning--Unlearning Schemes},
  author={Ghazi, Badih and Kamath, Pritish and Kumar, Ravi and Manurangsi, Pasin and Sekhari, Ayush and Zhang, Chiyuan},
  booktitle={Proceedings of the 36th Conference on Learning Theory (COLT)},
  series={PMLR}, volume={195}, pages={5110--5139}, year={2023},
  note={arXiv:2306.15744. Retrieved 2026-08-19.}}

@inproceedings{shi2025muse,
  title={MUSE: Machine Unlearning Six-Way Evaluation for Language Models},
  author={Shi, Weijia and Lee, Jaechan and Huang, Yangsibo and Malladi, Sadhika and Zhao, Jieyu and Holtzman, Ari and Liu, Daogao and Zettlemoyer, Luke and Smith, Noah A. and Zhang, Chiyuan},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2025},
  note={arXiv:2407.06460 (2024). Retrieved 2026-08-19.}}

@inproceedings{chen2024cure4rec,
  title={CURE4Rec: A Benchmark for Recommendation Unlearning with Deeper Influence},
  author={Chen, Chaochao and Zhang, Jiaming and Zhang, Yizhao and Zhang, Li and Lyu, Lingjuan and Li, Yuyuan and Gong, Biao and Yan, Chenggang},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2024},
  note={arXiv:2408.14393, doi 10.48550/arXiv.2408.14393. Retrieved 2026-08-19.}}

@article{laguna2026munkey,
  title={Rethinking Machine Unlearning: Models Designed to Forget via Key Deletion},
  author={Laguna, Sonia and da Silva Goncalves, Jorge and Vandenhirtz, Moritz and Ryser, Alain and Cannistraci, Irene and Vogt, Julia E.},
  journal={arXiv preprint arXiv:2603.15033}, year={2026},
  note={MUNKEY. ⛔ NO venue/comments field on arXiv; S2 venue = arXiv.org. The LaTeX keyword line ``Machine Learning, ICML'' is the ICML template default, NOT an acceptance. OpenReview forum gGH3Xp1lHR is bot-walled; presentation type (oral vs poster) COULD NOT BE VERIFIED. Forget-set MIA AUROC 51.40+-0.44 (CIFAR-10 10\%); ``Average Gap'' to retrained oracle 0.56+-0.21 -- NOT a membership gap. Retrieved 2026-08-19.}}

% ---------- physics ----------
@article{minami2018dissipative,
  title={Spontaneous symmetry breaking and Nambu-Goldstone modes in dissipative systems},
  author={Minami, Yuki and Hidaka, Yoshimasa},
  journal={Physical Review E}, volume={97}, number={1}, pages={012130}, year={2018},
  doi={10.1103/PhysRevE.97.012130},
  note={arXiv:1509.05042. THIS is the ``two types of Noether charges Q_R, Q_A'' + ``type-A NG modes are diffusive'' source. Retrieved 2026-08-19.}}

@article{hidaka2020open,
  title={Spontaneous symmetry breaking and Nambu--Goldstone modes in open classical and quantum systems},
  author={Hidaka, Yoshimasa and Minami, Yuki},
  journal={Progress of Theoretical and Experimental Physics}, volume={2020}, number={3}, pages={033A01},
  year={2020}, doi={10.1093/ptep/ptaa005},
  note={⛔ AUTHOR ORDER REVERSED vs the 2018 paper: HIDAKA is first here. Classifies NG modes into FOUR types (type-A/B x propagation/diffusion) -- do not cite this one for ``two types''. arXiv:1907.08241. Retrieved 2026-08-19.}}

@inproceedings{du2019implicit,
  title={Implicit Generation and Modeling with Energy Based Models},
  author={Du, Yilun and Mordatch, Igor},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={3603--3613}, year={2019},
  note={⛔ arXiv:1903.08689 v1 has a DIFFERENT title (``Implicit Generation and Generalization in Energy-Based Models''); use the NeurIPS title. Sec. 3.3 L2-regularizes energy magnitudes for numerical stability AND to ``ensure that partition function is integrable''. Retrieved 2026-08-19.}}

@article{gellmann1968behavior,
  title={Behavior of Current Divergences under $SU_3\times SU_3$},
  author={Gell-Mann, Murray and Oakes, R. J. and Renner, B.},
  journal={Physical Review}, volume={175}, number={5}, pages={2195--2199}, year={1968},
  doi={10.1103/PhysRev.175.2195}, note={The GMOR relation. Retrieved 2026-08-19.}}

@article{coleman1973nogoldstone,
  title={There are no Goldstone bosons in two dimensions},
  author={Coleman, Sidney},
  journal={Communications in Mathematical Physics}, volume={31}, number={4}, pages={259--264},
  year={1973}, doi={10.1007/BF01646487}, note={Retrieved 2026-08-19.}}

@article{merminwagner1966absence,
  title={Absence of Ferromagnetism or Antiferromagnetism in One- or Two-Dimensional Isotropic Heisenberg Models},
  author={Mermin, N. David and Wagner, Herbert},
  journal={Physical Review Letters}, volume={17}, number={22}, pages={1133--1136}, year={1966},
  doi={10.1103/PhysRevLett.17.1133},
  note={Erratum: PRL 17, 1307 (1966). Often written ``Hohenberg--Mermin--Wagner''. Retrieved 2026-08-19.}}

% ---------- venue-native (Part 2) ----------
@article{packer2023memgpt,
  title={MemGPT: Towards LLMs as Operating Systems},
  author={Packer, Charles and Wooders, Sarah and Lin, Kevin and Fang, Vivian and Patil, Shishir G. and Stoica, Ion and Gonzalez, Joseph E.},
  journal={arXiv preprint arXiv:2310.08560}, year={2023},
  note={Preprint; no venue on the arXiv record. Retrieved 2026-08-19.}}

@article{chhikara2025mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413}, year={2025},
  note={Preprint. Update phase = LLM tool call over ADD/UPDATE/DELETE/NOOP; ``DELETE for removal of memories contradicted by new information''. Retrieved 2026-08-19.}}

@article{rasmussen2025zep,
  title={Zep: A Temporal Knowledge Graph Architecture for Agent Memory},
  author={Rasmussen, Preston and Paliychuk, Pavlo and Beauvais, Travis and Ryan, Jack and Chalef, Daniel},
  journal={arXiv preprint arXiv:2501.13956}, year={2025},
  note={Preprint. Edges carry t_valid/t_invalid (+ t'_created/t'_expired); contradictions INVALIDATE rather than delete. Retrieved 2026-08-19.}}

@inproceedings{park2023generative,
  title={Generative Agents: Interactive Simulacra of Human Behavior},
  author={Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},
  booktitle={Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST)},
  year={2023}, note={arXiv:2304.03442. Recency = exponential decay over sandbox hours since last retrieval, ``Our decay factor is 0.995''; score = recency+importance+relevance, ``all alpha's are set to 1''. Retrieved 2026-08-19.}}

@inproceedings{sukhbaatar2021expirespan,
  title={Not All Memories are Created Equal: Learning to Forget by Expiring},
  author={Sukhbaatar, Sainbayar and Ju, Da and Poff, Spencer and Roller, Stephen and Szlam, Arthur and Weston, Jason and Fan, Angela},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)}, year={2021},
  note={arXiv:2105.06548. Learned per-memory expiration span. Record carried from outputs/unlearning-recon.md. Retrieved 2026-08-19.}}

@inproceedings{zhong2024memorybank,
  title={MemoryBank: Enhancing Large Language Models with Long-Term Memory},
  author={Zhong, Wanjun and Guo, Lianghong and Gao, Qiqi and Ye, He and Wang, Yanlin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}, year={2024},
  note={arXiv:2305.10250. Ebbinghaus-curve exponential decay of memory strength. Record carried from outputs/unlearning-recon.md. Retrieved 2026-08-19.}}

@article{munkhdalai2024infini,
  title={Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention},
  author={Munkhdalai, Tsendsuren and Faruqui, Manaal and Gopal, Siddharth},
  journal={arXiv preprint arXiv:2404.07143}, year={2024},
  note={Preprint; no venue on the arXiv record. First author is a PALM 2026 invited speaker. Retrieved 2026-08-19.}}

@article{yang2026controlplane,
  title={Control-Plane Placement Shapes Forgetting: An Architectural Study of Agent Memory Across Thirteen System Configurations},
  author={Yang, Dongxu}, journal={arXiv preprint arXiv:2606.15903}, year={2026},
  note={SINGLE author -- never ``Yang et al.''. Introduces ForgetEval. ``Production failures are predominantly forgetting failures rather than recall failures, yet existing benchmarks measure only recall.'' Retrieved 2026-08-19.}}

@article{uddin2026memora,
  title={From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents},
  author={Uddin, Md Nayem and Shubham, Kumar and Blanco, Eduardo and Baral, Chitta and Wang, Gengyu},
  journal={arXiv preprint arXiv:2604.20006}, year={2026},
  note={Memora benchmark; FAMA metric; 64\% of sampled recommendation errors are failures to forget. Record carried from outputs/unlearning-recon.md. Retrieved 2026-08-19.}}

@article{chakraborttii2026ghost,
  title={Ghost Vectors: Soft-Deleted Embeddings Remain Reconstructible in HNSW Vector Databases},
  author={Chakraborttii, Chandranil and Garc{\'i}a Alvarado, Jackeline and Abdulofizova, Sitora and Dwivedi, Shivanshu},
  journal={arXiv preprint arXiv:2606.18497}, year={2026},
  note={cs.CR/cs.AI/cs.DB; ``13 pages, 5 figures, 12 tables. Prepared for submission''. Tombstoned vectors remain physically present in ChromaDB/FAISS/Weaviate and are Vec2Text-reconstructible. THIS is the source for V5 K.2's currently un-cited soft-delete lean. Retrieved 2026-08-19.}}

@article{wang2026memleak,
  title={MemLeak: Diagnosing Information Leaks in Multimodal Agent Memory},
  author={Wang, Kuan and Zhang, Chao}, journal={arXiv preprint arXiv:2606.29788}, year={2026},
  note={Deletion cascade: <1\% direct, 18.3\% via retained correlated text, 12.0\% via retained images. ⚠ NOT about vector-DB tombstones -- do not conflate with Ghost Vectors. Retrieved 2026-08-19.}}

@article{wang2026agenticunlearning,
  title={Agentic Unlearning: When LLM Agent Meets Machine Unlearning},
  author={Wang, Bin and Wang, Fan and Wang, Pingping and Cong, Jinyu and Yu, Yang and Yin, Yilong and Han, Zhongyi and Wei, Benzheng},
  journal={arXiv preprint arXiv:2602.17692}, year={2026},
  note={Names the ``parameter--memory backflow'' problem. Record carried from outputs/unlearning-recon.md. Retrieved 2026-08-19.}}

@article{garg2026memfail,
  title={MemFail: Stress-Testing Failure Modes of LLM Memory Systems},
  author={Garg, Ishir and Kolhe, Neel and Song, Dawn and Zhao, Xuandong},
  journal={arXiv preprint arXiv:2605.26667}, year={2026},
  note={⚠ Formalizes summarization/storage/retrieval failure modes; the abstract makes NO deletion/forgetting claim. Retrieved 2026-08-19.}}
```

---

## Findings/results — dense summary

- **PALM's scope is now banked** (Part 1). V5's deletion contribution maps verbatim onto CFP topics 5 (*"memory update and deletion tests"*) and 7 (*"right-to-be-forgotten mechanisms"*); its decay contribution onto topic 1 (*"…and forget memories across sessions"*) and 4 (*"replay, forgetting"*); **"negative results" and "theoretical perspectives" are named accepted contribution types.** The Q7 known-gap closes.
- **The fit risk is citational, not topical.** SF-7 is confirmed as a real defect (zero venue-native cites) but the "wrong venue" reading it invites is not supported by the CFP. 13 primary-verified venue-native works supplied; a 4-cite minimum set recommended.
- **21 citation records verified fresh; 20 carried; 0 fabricated citations found in V5.**
- **Three usage FAILs, all fixable by editing:** Guo `§3 Eq. (1)` (→ §2, and Eq. (1) is the ε-only form) · Micciancio/Naor-Teague "introduced" (→ Snyder 1977 / obliviousness ≠ canonicity) · Minami & Hidaka **2020** (→ Hidaka & Minami; four types, not two).
- **Two uncorrected carry-overs the referee pass missed:** Mo's "at least dim(G/H)" (2 sites) and conformal-symplecticity-attributed-to-HLW (1 site, no year).
- **One scope clause owed:** Buchbinder–Petrank's exponential gap is WHI-vs-SHI, comparison-based model, heaps/queues.
- **Three owed citations** for named laws already in the text: GMOR, Coleman, Mermin–Wagner.
- **SF-10 recovered** (arXiv:2603.15033, MUNKEY, Laguna et al. 2026) **with the presentation type unverifiable and the 0.56 ± 0.21 mislabelled in the draft.**
- **Five V5 usage-checks PASS exactly and should be left alone:** Blelloch–Golovin stable matching + global priority + delete cascade · Hartline-reversibility caveat (an unusually good piece of hygiene) · Ginart Def. A.5 · Sekhari Def. 3 · SISA Def. III.1 · Du & Mordatch partition-function stability · Minami & Hidaka 2018's two-Noether-charge mechanism.

**Git footprint:** none — read-only; the only file created is this report.

---

## Proposed handover updates (for the Hub)

1. **Q7 (PALM topic scope) is CLOSED.** Facts in Part 1 above, retrieved from the venue site 2026-08-19: seven CFP topics verbatim, contribution types (incl. "theoretical perspectives" and "negative results"), NeurIPS-2026 template, non-archival, concurrent-submission permitted, anonymization-including-code re-confirmed, invited speakers/organizers. The charter's "V2-vs-V5 venue question" input is now available.
2. **N131's citation fence must be amended**: `Guo §3 Eq. (1)` → **`Guo §2 Eq. (1)` (ε-form), with the (ε,δ) relaxation unnumbered in the same section**. The ⛔"Guo Def. 1/2" ban stays (correct). ⚠ `unlearning-recon.md` §Proposed-updates item 1 also prints "Guo et al., ICML 2020, **Def. 1–2**" — that phrasing is the thing the ban exists to stop and should be annotated, never edited.
3. **New never-copy entries proposed:** (a) *"SHI was introduced by Micciancio"* → **Snyder FOCS'77 for unique representation; Micciancio 1997 = obliviousness ≠ canonical**; (b) *"Minami & Hidaka (2020)"* → **Hidaka & Minami (2020)**, and the 2020 paper gives **four** NG types; (c) *"MUNKEY … (oral)"* → **presentation type unverified; arXiv record has no venue at all**; (d) *"Du & Mordatch, 'Implicit Generation and Generalization…'"* → that is the arXiv v1 title; the NeurIPS title is *"…Modeling with Energy Based Models"*.
4. **Carry-item for the V5 writer beyond the referee's MF list:** U1 (Mo "at least", 2 sites) and U14 (HLW conformal symplecticity + missing year, 1 site) were fixed in V2 v0.8 and **never transferred to V5**. Recommend a standing rule: *every `v2-cite-check` usage fix is checked against V5/V1/V6 at the wave it lands, not only against the draft that commissioned it.*
5. **§4/K.2 rewrite input is ready.** 13 venue-native works with verified records, per-work "what it claims / how V5 relates (contrast)" one-liners, and a recommended 4-cite minimum for the 4-pp main text. The two currently-uncited leans in K.2 (soft-deleted vectors; the competing preprint) now both have identifiers.
6. **`unlearning-recon.md`'s own note — *"the sub-line is ~6 months old and moving — re-scout before any freeze"* — is discharged as of 2026-08-19** (ForgetEval, MemLeak, Ghost Vectors, MemFail added; Memora/Agentic-Unlearning re-confirmed). It should be re-checked again at camera-ready if that is months away.
7. **Anonymization item, now venue-confirmed:** PALM's *"applies to any supplementary or linked material as well, including code"* is verbatim on the site. **SF-12 is a hard blocker for the supplement**, not a nicety.

## Flags

- ⛔ **HEADLINE — Guo `§3 Eq. (1)` is wrong at three draft sites and in the program's own citation fence.** A referee who works in unlearning (and PALM has three privacy/security organizers plus a memorization-focused invited speaker) opens Guo, finds §3 is "Removal Mechanisms", and marks the paper's most-repeated external citation as unchecked.
- ⛔ **HEADLINE — the history-independence provenance sentence in K.2 misattributes the origin of both notions.** Verified against Blelloch–Golovin's own "Previous Work" section, which V5 cites 11 times.
- ⚠ **`v2-cite-check` fixes U1 and U14 never reached V5** — a lockstep leak the `v5-referee-v02` pass did not catch, and the class of leak the Hub should systematize.
- ⚠ **MUNKEY's "(oral)" could not be verified from any primary source.** OpenReview blocked on four routes; arXiv has no comments/journal-ref; S2 says arXiv.org. One search-engine summary says "Poster". **Recommend printing no presentation type.**
- ⚠ **Buchbinder–Petrank's exponential gap needs its WHI-vs-SHI / comparison-model / heaps-and-queues scope**, exactly as the Decelle/Agoritsas *k*-regime clause was needed for App H(c). Same failure class.
- ⚠ **V5 does not cite Titans, whose first author is a PALM invited speaker**, while making a retention-dial claim that Titans' learned forget gate is the nearest published neighbour to. V2 carries the record already.
- ⚠ **App G's Coleman / Mermin–Wagner defence is uncited**, and it is the paper's answer to the strongest available physics objection.
- ℹ **Fallback routing used and labelled:** link.springer.com (303 SSO → DBLP API) · sciencedirect.com (403 → author-hosted Technion PDF, read page 1 directly) · openreview.net (bot challenge on `/forum`, `/pdf`, `api.openreview.net`, `api2.openreview.net` → **could-not-verify**, recorded) · api.semanticscholar.org (one 429, retried successfully).
- ℹ **Per-file greps used throughout** (directory-level Grep over `.claude/` was positive-control-tested this session and returned a **false negative** on a known-present token — the standing doctrine holds). Every negative sweep reported here was positive-controlled on the same file.
