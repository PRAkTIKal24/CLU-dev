# Task: unlearning-recon — is "certified per-item lifetimes" a real slot in machine unlearning? (w24)

- **Agent:** `web-scout` · **Output:** `.claude/outputs/unlearning-recon.md` · **Branch:** none (read-only)
- **Read first:** `.claude/outputs/continual-learning-recon.md` — ⭐ **this is your template and your standard.** It found **SQHN (Nature Comms 2024)** already occupying the slot we assumed was open, and scoped the win honestly as a niche. Do exactly that here. · `.claude/outputs/controller-mvp.md` §3(b) (the decay/eviction machinery you are checking for novelty) · `.claude/negative_results.md` **N91/N93**

## Why
The Head's result set names **R1 — "memory with a dial: certified per-item lifetimes"** as the result **closest to done**, on the grounds that `controller-mvp` already demonstrated the machinery (half-lives set at write time, `exp(−leak·t)` measured exactly, permanent + leaky wells coexisting in one store, self-eviction below an amplitude floor, permanent items never evicted) and that what remains is **"mostly framing"** against the machine-unlearning / right-to-be-forgotten literature.

⚠ **That "mostly framing" claim is exactly what this recon must test.** We do not currently know the field's benchmarks, metrics, baselines, or whether our word *"certified"* collides with an established and much stronger technical meaning. The CL recon proved this due diligence changes the plan.

## Item 1 — the benchmark map (pin protocols from primary sources)
Canonical unlearning benchmarks, datasets, and **metrics**: forgetting quality, retain-set accuracy, **membership-inference-attack** success, relearn time, and the standard reference point of **exact retraining from scratch**. Pin the evaluation protocols the way the CL recon pinned van de Ven's taxonomy — a referee expects the field's own instruments.

## Item 2 — ⭐ "certified" is the load-bearing word: pin what it means there
In this literature *certified* generally implies a **formal guarantee** (differential-privacy-style bounds, or provable indistinguishability from a model retrained without the datum) — **not** "we measured a clean exponential decay." Establish:
- what **certified unlearning** formally requires, and who claims it (certified-removal lineage and its successors);
- the distinction between **exact** unlearning (SISA-style retraining/sharding) and **approximate** unlearning (gradient/influence-based);
- ⛔ **whether CLU's per-item decay would qualify as "certified" under any accepted definition, or whether we must call it something else.** If our decay is a *measured physical schedule* rather than a formal guarantee, say so bluntly and propose honest wording. **Getting this wrong is a one-line referee kill.**

## Item 3 — the winnability audit (the CL-recon format: ranked, with predicted loss modes)
Where does a landscape store with **exact, scheduled, per-item deletion by construction** actually win? Candidate angle: most unlearning methods are *approximate and expensive* because they must undo a datum's influence spread across shared weights — whereas a store that never entangled the item can delete it *structurally*. Audit that honestly:
- is **"deletion by construction"** already claimed by memory-augmented / retrieval-augmented / non-parametric methods (where deleting a datastore row is trivially exact)? **This is the sharpest preemption risk — a kNN datastore also deletes exactly, and it is a strong, simple baseline.**
- what does the store's **payload** actually retain after eviction, and does the *learned* `φ` retain information about a deleted item? (If `φ` saw the datum, deletion from the store may not be deletion from the system — flag this as a possible fatal scoping issue for R1.)
- weight class, harness availability, referee community.

## Item 4 — prior art on memory-module / KV-store unlearning
Who has run unlearning on episodic-memory modules, KV stores, retrieval-augmented systems, or Hopfield-class energy stores? Include the **SQHN** neighbour again if it touches deletion. Enumerate CLU's remaining novelty surface the way the CL recon did (per-item *schedules* and permanence-as-a-designed-coset are the candidates — decay as a **physical amplitude law** rather than a bookkeeping delete).

## Item 5 — collision check on the framing
Is **"memory with dials"** / capacity–lifetime–compute–admission as a *set of laws* already anyone's pitch? Check adjacent framings (controllable memory, editable memory, model editing / knowledge editing — note this is a **distinct and crowded** field, so pin the boundary).

## Acceptance
**ONE recommended entry** (or a plainly-stated ⛔ "this slot is occupied / this is not winnable as framed") + the benchmark map with pinned protocols + the certified-terminology verdict + the winnability audit with predicted loss modes + the mandatory-baseline list + bibtex-ready refs. Mark every claim **VERIFIED / SECONDARY / could-not-fetch**, exactly as the CL recon did.

## ⚠ The standard
The CL recon's value was that it said *"Class-IL is solved by replay and even by a dumb buffer; the winnable claim is 'best replay-free', not 'beats replay'"* — **before** we built anything. **Deliver that same honesty here.** If R1 is not "mostly framing" but "needs a formal guarantee we cannot currently give," that is the most valuable sentence you can write, and the Head wants it early rather than late.
