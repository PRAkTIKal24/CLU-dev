# deletion-prior-art — web-scout report

Task + acceptance criterion: verify from **primaries** the load-bearing SHI/deletion citations asserted by `order-independent-placement`, pin the unlearning arena's per-deletion cost model, test whether "deletion cost at matched utility" is already occupied, re-confirm the "certified" ban — **before any drafting**.
Status: **done** (read-only; no git footprint). 4 of 5 Group-1 primaries verified; Naor–Teague STOC'01 own numbered definitions **NOT** obtained from its primary (stated plainly, §Could-not-verify).

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). SIX items.**
> 1. ⭐ **The theorist's self-assessment was CORRECT and if anything understated.** PGCP is not merely "SHI hashing transplanted" — **Theorems 1 and 2 of `order-independent-placement` are re-derivations of Blelloch–Golovin (FOCS'07) Theorems 3.1/3.2 and their DELETE routine**, in a metric probe space. The paper must present them as *instantiation*, not as new theorems. Owner: whoever drafts the R1/controller section. **`placement-landing`'s wording gate: PASS with mandatory attribution wording (§1.6).**
> 2. **Theorem numbering/credit fix:** our "Thm 1 (unique representation)" = BG07 Thm 3.2; our "Thm 2 (exact deletion / fix-up cascade)" = BG07 §3 `DELETE` + `next(x)` chain. Restate as *"our placement rule is the Blelloch–Golovin stable-matching table with a global priority order and a metric-induced probe order; Theorems 1–2 follow from theirs"*. Owner: theorist/curator.
> 3. **Hartline is being leaned on incorrectly.** Hartline et al. Thm 1 requires a **reversible** (strongly-connected state graph) structure. Our *placement layer* is reversible; the **amplitude layer is not** (decay + `born` are monotone in time). Cite Hartline only for the SHI⇄canonical *definitional equivalence* (which is exactly how BG07 cite it), never inside a proof. Owner: theorist.
> 4. **`Controller.delete` naming hazard:** "deletion by construction" / "exact deletion" both have defended technical meanings (Garg–Goldwasser–Vasudevan EUROCRYPT'20 *deletion-compliance*; the exact-unlearning = distributional-equivalence-to-retraining convention). Recommended safe string in §Do-not-claim. Owner: curator (quote-block rule, same class as the "certified" ban).
> 5. **Candidate 2 ("deletion cost at matched utility") is NOT a free axis** — it is un-named but occupied in pieces (Ginart Def A.5 α-deletion-efficiency; Sekhari Def 3 deletion capacity at fixed excess risk 0.01; SISA's accuracy-vs-shards curve; MUSE criteria 4+5+6; CURE4Rec's 4 dimensions). We must **adopt existing vocabulary, not invent a benchmark**. Owner: whoever scopes the R1 experiment.
> 6. ⭐ **New ammunition against the "a kNN store deletes just as exactly" preemption** (the #0 preemption in `unlearning-recon`): it is true only for a *flat/brute-force* store. **Graph ANN indexes (HNSW/DiskANN) delete lazily via tombstones and are order-dependent**, and soft-deleted embeddings have just been shown to be **physically reconstructible** (arXiv:2606.18497, June 2026: 99% top-1 face identity recovery, 100% on medical age/gender). Owner: whoever writes the baselines paragraph — this changes the framing of baseline #1 in `unlearning-recon`.

---

## ⭐ DIAL DECLARATION (echo, protocol §7)
- **Dial:** lifetimes/admission (R1) — guards *claim wording*, not a measurement.
- **Laundering control (literature analogue):** novelty checked against the field that owns the technique (theory-of-computing data structures), not only the field we publish into (ML unlearning). **Executed:** §1 is entirely TCS-side.
- **Falsifies:** an asserted citation does not say what we claim / the continuous-landscape composition is also owned. **Outcome: no citation failed; the composition is NOT owned (§1.5), but the discrete skeleton is owned more completely than we wrote.**
- **Does NOT falsify:** the discrete skeleton being prior art — that was the expected finding and it is confirmed.

---

## Answer first
Every asserted Group-1 citation checks out, and **Blelloch & Golovin (FOCS 2007) own our algorithm outright** — their SHI open-addressed table is a Gale–Shapley stable matching in which *"each slot prefers k to k′ if k > k′"*, inserts by displacement, and deletes by following a `next(x)` chain: that is PGCP's priority-greedy placement and fix-up cascade, line for line, with our metric probe order substituted for their hash probe order. Our Theorems 1–2 are therefore **instantiations, not results**. What is *not* in that literature: (i) the slot set being a **geometric packing whose adjacency carries a physical isolation certificate**, (ii) the stored content being a **continuously decaying amplitude with a proven delete/decay commutation** (SHI literature stores static keys — no analogue exists), (iii) the canonical object being an **energy function `V(q)` read by a dynamical relaxation**, and (iv) a setting where the **price of strong history independence is negative** (our lattice packs 61/64 vs 43/64), against a literature that proves SHI can cost an exponential blow-up (Buchbinder–Petrank). On the ML side, "deletion cost at matched utility" is un-named but effectively occupied in pieces; adopt existing metrics rather than coin a benchmark.

---

## Group 1 — history-independence / uniquely-represented data structures

### 1.1 Blelloch & Golovin, FOCS 2007 ⭐ (the load-bearing one) — **CONFIRMED, and stronger than the theorist stated**
Verified from **two independent primaries**: the FOCS'07 paper PDF (`cs.cmu.edu/~dgolovin/papers/focs07.pdf`) and the earlier CMU tech report **CMU-CS-06-156, "Strongly History Independent Hashing with Deletion" (Oct 2006)**. Both agree on every structural point.

- **Definition 2.2 (their SHI):** *"A reversible data structure is strongly history independent (SHI) if it has canonical representations up to initial randomness. That is, for each sequence of initial random bits and for each state of the data structure, there is a unique memory representation."*
- **Framework (§3):** *"interpret the keys as men and the slots of the hash table as women, and construct a distribution on stable marriage instances between U and the set of all slots."* Key preference list over slots = its probe sequence (`RANK(k,w) < RANK(k,w′)` ⇒ k prefers w). **Slot preference over keys: *"each slot prefers k to k′ if k > k′"* — a single GLOBAL total order on keys.**
- **Theorem 3.1:** *"Every execution of the Gale-Shapley algorithm results in the same stable matching."*
- **Theorem 3.2:** *"For any hash table following our framework, after fixing the random bits there is a unique representation of the slots array for each set of p−1 or fewer keys."*
- **INSERT (Fig. 1):** probe; on collision, *if slot x prefers A[x] to k′* → keep probing; *else* → **swap (displace)**, set `i = RANK(k′,x)`, continue. **This is PGCP's "first cell not taken by a higher-priority key", executed incrementally.**
- **DELETE:** locate the key, then `While(next(x) ≠ null){ y = next(x); A[x] = A[y]; x = y; }` where `next(x)` = *"the slot x′ containing the largest key k′ that probed x but was rejected (or displaced) in favor of another key."* **This is PGCP's fix-up cascade.**
- **Bounds:** Thm 4.1 — SHI hash table, insert/delete expected O(1), search worst-case O(1), O(n) space; TR Thm 1 — with (1+ε)n slots, expected O(1/ε²) per op; Thm 3.4 — linear probing O(1/(1−α)³) at load α; Thm 5.1 SHI ordered dictionary O(log log n); Thm 5.4 SHI order maintenance.
- **They cite Hartline for the equivalence, exactly as we should:** *"Our definition of strong history independence differs from that of Naor and Teague [14], however the two definitions were proved equivalent by Hartline et al. [10] for reversible data structures."*

⛔ **Verdict: our fix-up cascade IS their algorithm.** The only algorithmic differences: (a) their probe sequence is a random hash `PROBE(k,i) = (h(k)+i) mod p` from a 5-universal `h`, ours is *cells of a hex lattice sorted by distance to a hash point* (a metric-induced preference list — a legal instance of their framework, which is stated for arbitrary preference lists); (b) their priority is the key value itself, ours is a splitmix64 priority hash (cosmetic); (c) their slots hold keys, ours hold continuous atoms.

### 1.2 Hartline, Hong, Mohr, Pentney, Rocke — **CONFIRMED, with a condition we must respect**
"Characterizing History Independent Data Structures", **ISAAC 2002; journal version Algorithmica 42:57–74 (2005)**.
- **Definition 2 (WHI):** *"…for any two sequences of operations X and Y that take the data structure from initialization to state A, the distribution over memory after X is performed is identical to the distribution after Y."*
- **Definition 3 (SHI):** *"…for any two (possibly empty) sequences of operations X and Y that take a data structure in state A to state B, the distribution over representations of B after X is performed on a representation a is identical to the distribution after Y is performed on a."*
- **Theorem 1:** *"For a reversible data structure to be SHI, a canonical representation for each state must be determined during the data structure's initialization."* — **condition: reversible = the state-transition graph is strongly connected.**
- **Corollary 1:** when representations a and b are mutually reachable, *"for all representations b′ of B, a ≡ b′ if and only if b′ = b."*
⚠ **Reconciliation item 3:** our decay/`born` layer is **not reversible** (amplitudes are monotone in elapsed time; a state once left is unreachable), so Hartline's characterization applies to the *placement* layer only. Our Theorem 1 is a direct construction and does not need Hartline; cite Hartline for the definitional equivalence only.

### 1.3 Naor & Teague, STOC 2001 — **claim confirmed, exact numbered definitions NOT obtained from the primary**
"Anti-persistence: history independent data structures", **STOC '01, pp. 492–501**; ePrint 2001/036.
- Abstract (from the IACR ePrint record, primary): *"Many data structures give away much more information than they were intended to. Whenever privacy is important, we need to be concerned that it might be possible to infer information from the memory representation of a data structure that is not available through its 'legitimate' interface."*
- Content confirmed: a **history-independent open-addressing hash table** with O(1) insert/search; a **history-independent dynamic perfect hash table**, linear space, expected amortized O(1) insert *and delete*; and a **general scheme for history-independent memory allocation**.
- The WHI/SHI distinction (one-time observer vs. multiple-observation observer, with the SHI observer additionally seeing the random bits) is verified from **two independent sources**: Hartline et al. (Defs 2–3 above) and **Naor's own later paper** (Naor, Segev, Wieder, "History-Independent Cuckoo Hashing", ICALP 2008), whose Def 2.1/2.2 read: *"A data structure implementation is weakly history independent if any two sequences of operations that yield the same content induce the same distribution on the memory representation"*, and for SHI, *"the distributions of the memory representation at the points of P1 and at the corresponding points of P2 are identical."* That paper also states: *"An alternative characterization of strong history independence was provided by Hartline et al. [14]… strong history independence is equivalent to having a canonical representation up to the choice of initial randomness."*
- **Which do we satisfy? SHI (the strong one), and trivially so**: PGCP is *deterministic* — no initial randomness at all — so there is literally one representation per live set. Below capacity we satisfy SHI unconditionally; under LRU/staleness eviction we satisfy **neither** (the state itself becomes history-dependent).
- ⚠ **Could not fetch the STOC'01/ePrint full text** (ePrint PDF behind a bot-check returning 403/CAPTCHA; ACM DL paywalled; the `.ps` returned HTTP 422). **Do not cite a Naor–Teague *definition number*** — cite the paper for the *notions*, and cite Hartline/BG07 for numbered definitions.

### 1.4 Micciancio, STOC 1997 — **CONFIRMED, and it is a contrast, not a precedent**
"Oblivious data structures: applications to cryptography", **STOC '97, pp. 456–464**. *"An oblivious data structure yields no knowledge about the sequence of operations that have been applied to it other than the final result"*; the Oblivious Tree is 2–3-tree-like with the property that *"the only information conveyed by an Oblivious Tree is the set of values stored at its leaves."* Crucially it achieves this **by randomization in the update algorithms, not by canonicalization** — i.e. it is the *weak*/randomized route to history independence, the opposite design choice from ours. Cite as the origin of the idea; **do not** cite it as the precedent for canonical placement.

### 1.5 Karger et al., STOC 1997 — **CONFIRMED as the correct weak neighbour**
Karger, Lehman, Leighton, Panigrahy, Levine, Lewin, "Consistent hashing and random trees: distributed caching protocols for relieving hot spots on the World Wide Web", **STOC '97, pp. 654–663**, DOI 10.1145/258533.258660. Order-independent *assignment* of keys to buckets with minimal disruption under bucket churn — **no spacing, no geometry semantics, no per-item content interaction**. Accurate as characterised; a one-line cite, nothing more.

### 1.6 ⭐ Adversarial closure — what is left, and the exact wording that survives
Two further probes, both material:
- **Blelloch, Golovin & Vassilevska, "Uniquely Represented Data Structures for Computational Geometry", SWAT 2008, pp. 17–28 (DOI 10.1007/978-3-540-69903-3_4)** — the same authors *already took uniquely-represented structures into geometry*: uniquely represented ordered subsets, range trees, horizontal point location, orthogonal segment intersection, 2-D dynamic convex hull. ⚠ **This narrows our margin: "canonical placement in a geometric setting" is not virgin ground.** What it does *not* do: place items to satisfy a *packing/minimum-separation* constraint, and it does not store continuous decaying quantities. (I could not extract this paper's abstract verbatim — PDF was Flate-compressed and Springer redirects to auth; bibliographic record verified via dblp, structure list from the tech-report ToC.)
- **Buchbinder & Petrank, "Lower and upper bounds on obtaining history independence", CRYPTO 2003, pp. 445–462; Information and Computation 204(2):291–337, 2006** — *"the first separation between the two notions of history independence, with an exponential gap: some operations may be executed in logarithmic time (or even in constant time) with the weaker definition, but require linear time with the stronger definition"*, with strong lower bounds for comparison-based heaps/queues. ⭐ **This is a gift, not a threat:** it establishes that SHI generally has a price, which makes our measured **negative** price (lattice 61/64 vs stochastic relocate 43/64) a reportable contrast rather than an unremarkable engineering detail.
- **Null result (stated as a null result):** I found **no work applying history independence / unique representation to a continuous energy landscape, to memories with decay, or to a store read by dynamical relaxation.** Searches over "history independent + energy landscape / metric packing / continuous placement", and over neural/vector memory, returned nothing in that intersection. This is absence of evidence from ~6 query formulations, not a proof of absence.

**⭐ The one-paragraph novelty statement the paper may use verbatim** (this replaces the current wording, which over-claims Theorems 1–2):

> *Order-independent placement is not new: strongly history-independent (uniquely represented) data structures were introduced by Micciancio (STOC'97) and Naor & Teague (STOC'01), characterised as canonical representations by Hartline et al. (Algorithmica 2005), and realised for open-addressed hash tables by Blelloch & Golovin (FOCS'07), whose table is a stable matching between keys and slots under a global key priority — the same priority-greedy rule and the same delete-time fix-up cascade we use here. Our placement rule is an instance of that framework, obtained by replacing the hash probe sequence with a probe order induced by distance in the store's metric, and we claim no new result about unique representation per se. What is new is the composition: the slots are a lattice packing, so the canonical representation carries a minimum-separation certificate and hence a quantitative interference bound; the stored content is not a static key but a continuously decaying amplitude, and we prove that deletion commutes with the decay flow, so that a deleted item's survivors are bit-identical to a history in which it was never written, at every point of their schedules; the canonical object is therefore not a memory layout but an energy function, which a dynamical read relaxes into; and, unlike the general case, where strong history independence is known to cost as much as an exponential slowdown (Buchbinder & Petrank, CRYPTO'03), here it is free — the designed lattice packs strictly better than the stochastic relocation rule it replaces. This is a store-level structural property, not a claim about the trained encoder or about certified (ε,δ) unlearning.*

---

## Group 2 — the unlearning / exact-deletion arena, with per-deletion cost

| work | mechanism | guarantee (exact wording) | **what happens at deletion time** | per-deletion cost |
|---|---|---|---|---|
| **SISA** — Bourtoule et al., IEEE S&P 2021, arXiv:1912.03817 | shard → isolate → slice → aggregate, + per-slice checkpoints | §III-B **Def. III.1**: distribution of models after learn-then-unlearn matches the distribution from learning on D∖d_u; explicitly *"the definition does not necessarily require that the owner retrain the model M′ from scratch on D∖d_u, as long as they are able to provide evidence that model M′ could have been trained from scratch"* | locate shard → locate slice → **retrain that shard from the checkpoint saved before that slice** (not from random init) | **4.63× (Purchase), 2.45× (SVHN), 1.36× (ImageNet)** vs retraining — modest; **paid for in accuracy**: >5 pp drop for S>20 (simple tasks), **16.14 pp top-5 degradation on ImageNet** at standard sharding (§VII-A1) |
| **SILO** — Min et al., ICLR 2024 (spotlight), arXiv:2308.04430 | parametric LM trained on permissive data + nonparametric datastore | *"enables data producers to opt out from the model by removing content from the store"* | **row delete from the datastore; no retraining** | ~O(1); the reason this preemption is dangerous. ⚠ **but see recon item 6** |
| **Ticketed Learning–Unlearning** — Ghazi et al., COLT 2023, arXiv:2306.15744 | per-example encrypted "ticket" + small central state | *"a good predictor that is identical to the predictor that would have been produced when learning from scratch on the surviving examples"* | *"the examples that wish to be unlearnt present their tickets to the unlearning algorithm, which additionally uses the central information to return a new predictor"* | space-efficient for restricted concept classes (thresholds, parities, intersection-closed); no general-model result |
| **DaRE forests** — Brophy & Lowd, ICML 2021, arXiv:2009.05567 | random upper levels + greedy lower levels; cached node statistics and leaf data | *"model updates for each DaRE tree in the forest are exact, meaning that removing instances from a DaRE model yields exactly the same model as retraining from scratch on updated data"* | update cached counts along the root-to-leaf path; **retrain only affected subtrees** | *"orders of magnitude faster than retraining from scratch while sacrificing little to no predictive power"* |
| **PALL** — Özdenizci, Rueckert & Legenstein, ICLR 2025, arXiv:2505.10941 | task-specific **sparse subnetworks** with parameter sharing in one net + episodic-memory rehearsal | *"exact task unlearning without performance degradations"* — **task-level, not example-level** | drop/isolate the task's subnetwork (parameter isolation) | cheap, but the unit of deletion is a **task**; rehearses ⇒ stores raw data |
| **Ginart et al.**, NeurIPS 2019, arXiv:1907.05012 | quantized/divide-and-conquer k-means | **Def. A.5**: *"an algorithm A is α-deletion efficient if it runs Algorithm 3 in amortized time O(n^(1−α))"* | recompute only destabilised centroids | *"over 100X improvement in deletion efficiency across 6 datasets"* at *"comparable statistical quality"* |
| **Sekhari et al.**, NeurIPS 2021, arXiv:2103.03279 | convex ERM + second-order update | **Def. 3 (deletion capacity)**: max m s.t. `E[max_{U⊆S,|U|≤m} F(Ā(U,A(S),T(S))) − F*] ≤ 0.01` | one bounded update per deletion using retained statistics | **Thm 2:** capacity ≥ c·n√ε/(d log(1/δ))^{1/4}, vs Θ(n/√d) for DP |

**Reading for us:** the arena's cost model is *"retrain a bounded piece"* — nobody in the exact camp achieves O(1)-per-item with a *parametric* model, and every one of them pays for it either in accuracy (SISA, 16.14 pp) or in scope (concept-class restriction, task granularity, trees only). A store whose deletion is O(cascade)=2.84 moves and costs **zero** utility sits in a cell no parametric method occupies — but it sits in the *same* cell as SILO/kNN, which is exactly why the win must be framed as *utility at matched deletion cost*, not deletion cost.

---

## Group 3 — is "deletion cost at matched utility" already occupied?

**Answer: there is no benchmark of that name, and no leaderboard someone already wins — but the axis is occupied in pieces, and inventing a new metric would be scored as reinvention.** Findings:
- **Formal cost side, already owned:** Ginart et al. **Def. A.5** ("α-deletion efficient", amortized O(n^{1−α})).
- **Formal "at matched utility" side, already owned:** Sekhari et al. **Def. 3 deletion capacity** — literally *the number of deletions attainable subject to a fixed excess-risk budget (0.01)*. **This is the closest existing formalisation of Candidate 2 and we should adopt it or explicitly say why the store setting needs a different one.**
- **Benchmark side:** **MUSE** (Shi et al., arXiv:2407.06460, 2024) enumerates six properties including *"(4) utility preservation on data not intended for removal, (5) scalability with respect to the size of removal requests, (6) sustainability over sequential unlearning requests"* — i.e. cost-vs-utility is criteria 4+5+6, not a single scalar. **CURE4Rec** (NeurIPS 2024 D&B, arXiv:2408.14393) defines four dimensions: *"unlearning Completeness, recommendation Utility, unleaRning efficiency, and recommendation fairnEss"*. **OpenUnlearning** (NeurIPS 2025 D&B, arXiv:2506.12618) standardises >12 metrics for TOFU/WMDP but is not deletion-cost-centric.
- **Practice side:** SISA §VII already *is* a deletion-cost-vs-utility Pareto study (accuracy vs number of shards vs retraining time), it just isn't named as a benchmark.
- ⚠ **The trap:** on raw deletion cost, a nonparametric datastore wins trivially (O(1) row delete, SILO). Any claim must therefore be **utility (or capacity/interference) at matched deletion cost**, or **deletion cost at matched utility with the datastore's utility as the comparator** — and the datastore baseline must be present in the table (already mandated by `unlearning-recon`).
- ⭐ **New, and it materially strengthens the position (recon item 6):** the "kNN row-delete is exactly as exact" preemption holds only for flat stores. **Graph-based ANN indexes delete lazily**: FreshDiskANN (Singh, Subramanya, Krishnaswamy, Simhadri, arXiv:2105.09613) marks deletions in a deletion vector and defers graph repair to a batch **consolidation** pass — deletion is *not* order-independent and recall degrades between consolidations (recall-degradation numbers I saw are from secondary/derivative sources — **single-sourced, do not quote**). And **"Ghost Vectors: Soft-Deleted Embeddings Remain Reconstructible in HNSW Vector Databases"** (Chakraborttii et al., arXiv:2606.18497, June 2026) shows *"deleted vectors remain physically recoverable by accessing the raw index files at the storage layer, bypassing API access"* across three HNSW implementations — **99% top-1 identity recovery on facial embeddings, 100% recovery of patient age/gender markers on NIH Synthea, 25.5%/46.4% name/location recovery on Wikipedia bios**. ⚠ Preprint, not peer-reviewed, 6 weeks old, single-sourced — usable as *"a recent preprint reports…"*, not as an established fact.

---

## Group 4 — vocabulary check (the standing ban)

- ✅ **The "certified" ban HOLDS — re-verified from the primary this session.** Guo, Goldstein, Hannun & van der Maaten, "Certified Data Removal from Machine Learning Models", ICML 2020, arXiv:1911.03030, **Def. 1** `e^{−ε} ≤ P(M(A(D),D,x)∈T)/P(A(D∖x)∈T) ≤ e^{ε}` and **Def. 2** `P(M(A(D),D,x)∈T) ≤ e^{ε}P(A(D∖x)∈T)+δ` (and the reverse). We supply no (ε,δ) over a *learning algorithm's output distribution*; the word stays banned program-wide. One line, as instructed.
- ⚠ **"exact deletion" / "exact unlearning" carries a defended meaning** and it is *distributional*: the unlearned model's parameter distribution must equal that of retraining on the retain set (SISA Def. III.1; the convention across the exact-unlearning literature). **Our claim is about a store's byte-level state, not about a model's output distribution.** Saying "exact deletion" unqualified invites the referee to apply the model-level test and fail us on the encoder channel.
- ⚠ **"data deletion" / "deletion by construction" collides with a cryptographic definition:** Garg, Goldwasser & Vasudevan, "Formalizing Data Deletion in the Context of the Right to be Forgotten", **EUROCRYPT 2020, pp. 373–402** (arXiv:2002.10635; ePrint 2020/254) define **deletion-compliance** (their statistical Def. 2.2): the collector's state after a deletion request must be statistically indistinguishable, to an unbounded distinguisher, from an execution in which the requester **never interacted at all**, including removal of *"dependencies that other data could have on the data that is requested for deletion"*. We satisfy the store-layer analogue but **not** the dependency clause (the encoder φ saw the item). Do not use the phrase "deletion-compliant".
- ✅ Thudi et al. (USENIX Sec'22, arXiv:2110.11891) still governs the only auditable form: **algorithm-level** deletion claims. PGCP's Theorem 2 corollary is exactly that form and should be worded that way.

---

## Must-cite list (one line each on why)

1. **Blelloch & Golovin, FOCS 2007** — ⭐ *owns our algorithm*; cite as the source of the priority/stable-matching placement and the delete fix-up cascade.
2. **Hartline, Hong, Mohr, Pentney & Rocke, Algorithmica 2005** — SHI ⇄ canonical representation (definitional equivalence only; reversibility condition).
3. **Naor & Teague, STOC 2001** — origin of the WHI/SHI notions and of history-independent hashing/allocation.
4. **Micciancio, STOC 1997** — origin of oblivious data structures; contrast (randomisation route, not canonicalisation).
5. **Buchbinder & Petrank, CRYPTO 2003 / Inf.&Comp. 2006** — SHI has a proven price in general; our negative price is meaningful against it.
6. **Blelloch, Golovin & Vassilevska, SWAT 2008** — unique representation already reached computational geometry; must differentiate (no packing constraint, no decaying content).
7. **Karger et al., STOC 1997** — order-independent assignment without geometry semantics; one line.
8. **Bourtoule et al., S&P 2021 (SISA)** — the arena's reference exact method and its accuracy price.
9. **Ghazi et al., COLT 2023 (Ticketed L–U)** — formal per-example exact deletion, restricted classes.
10. **Brophy & Lowd, ICML 2021 (DaRE)** — exact deletion by cached-structure surgery.
11. **Min et al., ICLR 2024 (SILO)** — the nonparametric-datastore opt-out; our hardest preemption.
12. **Özdenizci et al., ICLR 2025 (PALL)** — exact *task* unlearning inside CL; the CL∩forgetting cell.
13. **Ginart et al., NeurIPS 2019** + **Sekhari et al., NeurIPS 2021** — the two existing formalisations of deletion cost / deletion-at-fixed-utility (Candidate 2's vocabulary).
14. **Guo et al., ICML 2020** — the definition of "certified" we are explicitly not claiming.
15. **Thudi et al., USENIX Sec 2022** — why the claim must be algorithm-level.
16. **Garg, Goldwasser & Vasudevan, EUROCRYPT 2020** — the defended meaning of "data deletion" we are not claiming.
17. *(optional, framing)* **Singh et al., arXiv:2105.09613 (FreshDiskANN)** and **Chakraborttii et al., arXiv:2606.18497 (Ghost Vectors, preprint)** — the datastore baseline is not as clean as it looks.

## ⛔ Do-not-claim list

1. ⛔ **"certified"** anything (Guo Def. 1/2). Unchanged.
2. ⛔ **"we prove that placement is order-independent"** as a novel theorem — it is Blelloch–Golovin Thm 3.2 instantiated. Say "follows from".
3. ⛔ **"the first order-independent / uniquely-represented store"** — false (BG07; and BGV08 for geometry).
4. ⛔ **"exact deletion"** unqualified — always "exact **store-level** deletion" + the sentence excluding encoder and learned-landscape channels.
5. ⛔ **"deletion by construction"** as a novelty claim (`unlearning-recon`: it is a table row) — and now additionally ⛔ **"deletion-compliant"** (EUROCRYPT'20 term of art with a dependency clause we fail).
6. ⛔ **"our fix-up cascade"** as a possessive — it is theirs.
7. ⛔ Any deletion claim in the same breath as **LRU/staleness eviction** (already scoped out by the theorist; the literature would call the structure not even WHI in that mode).
8. ⛔ Quoting FreshDiskANN recall-degradation numbers or Ghost-Vectors recovery rates as established facts (preprint / secondary).
9. ⛔ Coining a new benchmark name for "deletion cost at matched utility" without citing Ginart Def. A.5, Sekhari Def. 3, MUSE and CURE4Rec.

## Could not verify (stated plainly, per task)

- **Naor & Teague's own numbered definitions** of WHI/SHI from the STOC'01 primary. ePrint 2001/036 PDF returns a bot-check (403), the `.ps` returned HTTP 422, ACM DL is paywalled. The *notions* are verified from two independent primaries (Hartline Defs 2–3; Naor–Segev–Wieder Defs 2.1–2.2, i.e. Naor restating himself). **Do not cite an N–T definition number.**
- **Blelloch–Golovin–Vassilevska SWAT'08 abstract verbatim** — PDF Flate-compressed (no local pdftoppm), Springer behind auth redirect. Bibliographic record verified from dblp; the covered-structures list comes from the tech-report table of contents, so treat "what it covers" as high-confidence but not quote-grade.
- ⚠ **Extraction-pipeline caveat (integrity note):** the FOCS'07, CMU-TR, Hartline, Sekhari, SISA, Garg et al. and Guo et al. quotes were obtained via a text-extraction proxy over the PDFs and are faithful in substance — the FOCS'07 content is corroborated point-for-point by the independent CMU-CS-06-156 tech report — but **exact punctuation/line-breaks should be re-checked against the PDFs before camera-ready**, and *equation renderings* (Sekhari Def. 3, Guo Defs 1–2) should be re-typeset from source rather than copy-pasted from this report.
- **SISA's `Definition III.1` exact formal statement** — I have its paraphrase plus the verbatim caveat clause, not the full displayed formula.
- Whether any *venue-published* work measures deletion cost of a **continual-learning memory buffer** at matched utility — searched, nothing found; treated as an open cell, not as a null.

## Bibtex-ready refs

```bibtex
@inproceedings{naor2001antipersistence,
  title={Anti-persistence: History Independent Data Structures},
  author={Naor, Moni and Teague, Vanessa},
  booktitle={Proceedings of the 33rd Annual ACM Symposium on Theory of Computing (STOC)},
  pages={492--501}, year={2001}, doi={10.1145/380752.380844},
  note={Cryptology ePrint Archive 2001/036}}

@article{hartline2005characterizing,
  title={Characterizing History Independent Data Structures},
  author={Hartline, Jason D. and Hong, Edwin S. and Mohr, Alexander E. and Pentney, William R. and Rocke, Emily C.},
  journal={Algorithmica}, volume={42}, number={1}, pages={57--74}, year={2005},
  doi={10.1007/s00453-004-1140-z},
  note={Preliminary version ISAAC 2002; Thm 1: reversible SHI requires canonical representations}}

@inproceedings{blelloch2007shi,
  title={Strongly History-Independent Hashing with Applications},
  author={Blelloch, Guy E. and Golovin, Daniel},
  booktitle={48th Annual IEEE Symposium on Foundations of Computer Science (FOCS)},
  pages={272--282}, year={2007},
  note={Def. 2.2 SHI = canonical up to initial randomness; Thm 3.1--3.2 stable-matching uniqueness; Thm 4.1 O(1) insert/delete. TR version: CMU-CS-06-156}}

@techreport{blelloch2006shideletion,
  title={Strongly History Independent Hashing with Deletion},
  author={Blelloch, Guy E. and Golovin, Daniel},
  institution={Carnegie Mellon University}, number={CMU-CS-06-156}, year={2006}}

@inproceedings{micciancio1997oblivious,
  title={Oblivious Data Structures: Applications to Cryptography},
  author={Micciancio, Daniele},
  booktitle={Proceedings of the 29th Annual ACM Symposium on Theory of Computing (STOC)},
  pages={456--464}, year={1997}, doi={10.1145/258533.258638}}

@inproceedings{karger1997consistent,
  title={Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web},
  author={Karger, David and Lehman, Eric and Leighton, Tom and Panigrahy, Rina and Levine, Matthew and Lewin, Daniel},
  booktitle={Proceedings of the 29th Annual ACM Symposium on Theory of Computing (STOC)},
  pages={654--663}, year={1997}, doi={10.1145/258533.258660}}

@article{buchbinder2006lower,
  title={Lower and Upper Bounds on Obtaining History Independence},
  author={Buchbinder, Niv and Petrank, Erez},
  journal={Information and Computation}, volume={204}, number={2}, pages={291--337}, year={2006},
  note={Preliminary version CRYPTO 2003, pp. 445--462; exponential WHI/SHI separation}}

@inproceedings{blelloch2008geometry,
  title={Uniquely Represented Data Structures for Computational Geometry},
  author={Blelloch, Guy E. and Golovin, Daniel and Vassilevska, Virginia},
  booktitle={11th Scandinavian Workshop on Algorithm Theory (SWAT)},
  pages={17--28}, year={2008}, doi={10.1007/978-3-540-69903-3_4}}

@inproceedings{naor2008cuckoohi,
  title={History-Independent Cuckoo Hashing},
  author={Naor, Moni and Segev, Gil and Wieder, Udi},
  booktitle={ICALP}, year={2008}}

@inproceedings{bourtoule2021sisa,
  title={Machine Unlearning},
  author={Bourtoule, Lucas and Chandrasekaran, Varun and Choquette-Choo, Christopher A. and Jia, Hengrui and Travers, Adelin and Zhang, Baiwu and Lie, David and Papernot, Nicolas},
  booktitle={IEEE Symposium on Security and Privacy (S\&P)}, year={2021},
  note={arXiv:1912.03817; Def. III.1; 4.63x Purchase / 2.45x SVHN / 1.36x ImageNet; 16.14 PP top-5 degradation}}

@inproceedings{brophy2021dare,
  title={Machine Unlearning for Random Forests},
  author={Brophy, Jonathan and Lowd, Daniel},
  booktitle={ICML}, year={2021}, note={arXiv:2009.05567; exact = identical to retraining}}

@inproceedings{ghazi2023ticketed,
  title={Ticketed Learning-Unlearning Schemes},
  author={Ghazi, Badih and Kamath, Pritish and Kumar, Ravi and Manurangsi, Pasin and Sekhari, Ayush and Zhang, Chiyuan},
  booktitle={COLT}, year={2023}, note={arXiv:2306.15744}}

@inproceedings{min2024silo,
  title={SILO Language Models: Isolating Legal Risk in a Nonparametric Datastore},
  author={Min, Sewon and Gururangan, Suchin and Wallace, Eric and Shi, Weijia and Hajishirzi, Hannaneh and Smith, Noah A. and Zettlemoyer, Luke},
  booktitle={ICLR (spotlight)}, year={2024}, note={arXiv:2308.04430}}

@inproceedings{ozdenizci2025pall,
  title={Privacy-Aware Lifelong Learning},
  author={{\"O}zdenizci, Ozan and Rueckert, Elmar and Legenstein, Robert},
  booktitle={ICLR}, year={2025}, note={arXiv:2505.10941; exact TASK unlearning, sparse subnetworks + episodic rehearsal}}

@inproceedings{ginart2019forget,
  title={Making AI Forget You: Data Deletion in Machine Learning},
  author={Ginart, Antonio and Guan, Melody Y. and Valiant, Gregory and Zou, James},
  booktitle={NeurIPS}, year={2019}, note={arXiv:1907.05012; Def. A.5 alpha-deletion efficiency; 100x on k-means}}

@inproceedings{sekhari2021remember,
  title={Remember What You Want to Forget: Algorithms for Machine Unlearning},
  author={Sekhari, Ayush and Acharya, Jayadev and Kamath, Gautam and Suresh, Ananda Theertha},
  booktitle={NeurIPS}, year={2021},
  note={arXiv:2103.03279; Def. 3 deletion capacity at excess risk 0.01; Thm 2 capacity >= c n sqrt(eps)/(d log(1/delta))^{1/4}}}

@inproceedings{guo2020certified,
  title={Certified Data Removal from Machine Learning Models},
  author={Guo, Chuan and Goldstein, Tom and Hannun, Awni and van der Maaten, Laurens},
  booktitle={ICML}, year={2020}, note={arXiv:1911.03030; Def. 1 eps-certified, Def. 2 (eps,delta)-certified removal}}

@inproceedings{thudi2022auditable,
  title={On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning},
  author={Thudi, Anvith and Jia, Hengrui and Shumailov, Ilia and Papernot, Nicolas},
  booktitle={31st USENIX Security Symposium}, year={2022}, note={arXiv:2110.11891}}

@inproceedings{garg2020formalizing,
  title={Formalizing Data Deletion in the Context of the Right to Be Forgotten},
  author={Garg, Sanjam and Goldwasser, Shafi and Vasudevan, Prashant Nalini},
  booktitle={EUROCRYPT}, pages={373--402}, year={2020},
  note={arXiv:2002.10635; ePrint 2020/254; deletion-compliance, Def. 2.2}}

@article{shi2024muse,
  title={MUSE: Machine Unlearning Six-Way Evaluation for Language Models},
  author={Shi, Weijia and Lee, Jaechan and Huang, Yangsibo and Malladi, Sadhika and Zhao, Jieyu and Holtzman, Ari and Liu, Daogao and Zettlemoyer, Luke and Smith, Noah A. and Zhang, Chiyuan},
  journal={arXiv:2407.06460}, year={2024},
  note={six criteria incl. scalability wrt removal-request size and sustainability over sequential requests}}

@inproceedings{chen2024cure4rec,
  title={CURE4Rec: A Benchmark for Recommendation Unlearning with Deeper Influence},
  booktitle={NeurIPS Datasets and Benchmarks}, year={2024},
  note={arXiv:2408.14393; Completeness, Utility, unlearning Efficiency, Fairness}}

@article{singh2021freshdiskann,
  title={FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search},
  author={Singh, Aditi and Subramanya, Suhas Jayaram and Krishnaswamy, Ravishankar and Simhadri, Harsha Vardhan},
  journal={arXiv:2105.09613}, year={2021}}

@article{chakraborttii2026ghost,
  title={Ghost Vectors: Soft-Deleted Embeddings Remain Reconstructible in HNSW Vector Databases},
  author={Chakraborttii, Chandranil and Garc{\'i}a Alvarado, Jackeline and Abdulofizova, Sitora and Dwivedi, Shivanshu},
  journal={arXiv:2606.18497}, year={2026}, note={PREPRINT, not peer-reviewed}}
```

---

## Proposed handover updates (for the Hub)

- **`placement-landing` wording gate: PASS, conditional.** The engineer may land the code and use the deletion claim **only** with the §1.6 attribution paragraph and the §Do-not-claim list applied. Theorems 1–2 must be re-labelled as instantiations of Blelloch–Golovin (FOCS'07) Thm 3.2 + their `DELETE` routine.
- **N99 update block should be amended**: the theorist's "SHI hashing transplanted" is right, but the paper must go further — *the same rule, the same cascade, a different probe metric*. The defensible novelty is the four-part composition (packing certificate · decaying content with a commutation proof · canonical **energy function** · negative price of SHI against Buchbinder–Petrank).
- **Add to the standing vocabulary ban** (alongside "certified"): "deletion-compliant" (EUROCRYPT'20 term of art), and unqualified "exact deletion"/"exact unlearning" (they are distributional, model-level terms).
- **Candidate 2 rescoped:** don't invent a benchmark. Report deletion cost vs utility using **Ginart Def. A.5** (cost) and **Sekhari Def. 3 deletion capacity** (volume at fixed utility) as the cited formalisms, with SISA's shards-vs-accuracy curve as the practice precedent and the flat-datastore row-delete as the mandatory trivial-substitute control.
- **New baseline-framing asset:** the datastore preemption is weaker than `unlearning-recon` assumed for *graph* indexes (lazy tombstone deletion; a June-2026 preprint reports soft-deleted HNSW vectors are physically reconstructible). Worth one sentence in the R1 positioning — flagged as preprint-grade.
