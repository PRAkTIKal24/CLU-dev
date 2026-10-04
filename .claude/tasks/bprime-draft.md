# bprime-draft — the B′ audit paper (TIER i), drafted from the evidence base

> ## ✅✅ V2 RELEASE — 2026-08-01, second review: **THE NUMBER-FREEZE GATE IS CLOSED. F3 LANDED.**
> `draft-v1` is accepted (the `⟦F3⟧` discipline was exactly right — 47 markers, banner, §2.6, L4).
> **Your v2 pass is the mechanical integration your own report §"What draft-v2 must do" specifies**,
> against `.claude/outputs/bprime-rivals-f3.md` (Hub-verified digit-for-digit from its raw JSONs).
> The adjudicated outcomes you integrate:
> - ⭐ **The headline SURVIVES AND STRENGTHENS:** 0 of 5 rivals beat the raw-metric +0 B table under the
>   full F3 grid, at 5× budget, under held-out selection, and **at n = 9 every raw margin is ≥ 4.4 SE
>   below zero** (−0.259 … −0.460). §4.2's sentence stands, now quoted at n = 9.
> - ⛔ **HUB RULING (binding): every rescue verdict and every rival mean in the paper is quoted at the
>   POOLED n = 9, never at n = 3** — f3 showed the n = 3 rescue gate is a coin flip (three legitimate
>   configurations give three different rescued sets). At n = 9, both code paths agree: **rescued =
>   {deltanet, gdn, gdn2}; ttt_mlp NOT RESCUED in any configuration; ttt_linear ambiguous (0.09–0.32
>   lift, code-path-dependent) — print it as UNSTABLE with both readings, quote no margin against it.**
> - **R5's scorecard line updates: "3 of 5 ≤ 0" → "4 of 5 ≤ 0" at n = 9** (gdn crosses; gdn2's +0.047 ±
>   0.028 is < 2 SE ⇒ **a tie, never "the one rival that beats its table"**).
> - **§4's P5-vs-raw magnitudes re-quote at n = 9:** 0.276 / 0.263 / 0.425 / 0.856 / 0.942 (all > 2 SE;
>   direction unchanged 5 of 5).
> - ⭐ **New methods material (§2.6 rewrite):** the referee attack is closed **by measurement** — the F3
>   grid's added points are chosen in **0 of 45** cells under the incumbent selection rule, the tuning
>   effect on `full` is ≤ 0.031 on every arm, and a 64 % fit-loss cut (ttt_mlp @2000 steps) moves eval
>   < 1 SE. ⭐ Plus f3's finding-about-F3 (fit-split best-of-grid makes 6×2 operationally 6×1 — the
>   held-out selection secondary) — one paragraph, methods or appendix, it is a real contribution.
> - ⚠ **Disclose the init-redraw effect** exactly as f3 does: the declared init-key change moved arms
>   4–35× more than tuning did; the `f3_lite_control` column is what makes the tuning effect readable.
> - **New never-quotes for your §6/L-list:** any C2W4 rescue verdict at n = 3 · any margin against
>   `ttt_mlp` · *"no margin against DeltaNet is quotable"* (RETRACTED at n = 9).
> Then: delete the banner, move the before/after table to App. I.1, and hand the Hub `draft-v2` for the
> **referee pass** (§A15.5 — before anything else).

**Campaign 2, wave C2W4. Agent:** paper-writer. **No worktree, no repo code** (protocol §3: research-only
agents skip the git discipline — nothing to commit). Writes under `.claude/papers/bprime/` and
`.claude/outputs/bprime-draft.md`.
Charter **ADDENDUM 3 §A15 task 5**, framed by **§A13 tier i**.

> ## ✅ RELEASED — 2026-07-31, at the C2W4 review. Your gate has opened.
> `bprime-rivals` **landed and was accepted**; the §A14.3 checkpoint passes. **B′ has its rival rows.**
> Base: `main @ 21a6dc4` (three merges, zero conflicts, 1136 tests). Your spine is
> `.claude/outputs/bprime-rivals.md` + its raw artifact
> `.claude/outputs/bprime-rivals/run/exp_bprime_rivals_metrics.json` (**the Hub re-derived every
> headline number from that JSON — quote the report, but the JSON is the authority**).
>
> **The falsifier adjudications you were promised, as ruled at review:**
> - **FB1 — does not fire.** **FB2 — does not fire**, ⚠ **but on 3 of 5 §2 families by measurement
>   only** (`ttt_linear`, `ttt_mlp`, and the delta arms sharing one state type); **Mamba-2, SDM and
>   Titans were adjudicated from their equations alone.** ⛔ **Never blur the two in the paper — say
>   which families were measured and which were reasoned.**
> - ⭐ **FB3 — does NOT fire in the strong form; it DOES fire in the WEAK form, and you write the weak
>   form plainly.** Verbatim from the measurement: *against the arg-min control, `gdn` (+1.02) and
>   `gdn2` (+0.88) show large positive dividends while the CLU shows −0.0789 — test-time dynamics pays
>   for the delta-rule family and does not pay for ours.* **That sentence is true as measured, it is in
>   the report, and it goes in the paper.** The strong form fails only because **0 of 5 rivals beat the
>   raw-metric +0 B table either** — and *only because R5 was registered before measurement*. ⭐ **Say
>   that last part in the methods: had the raw control been added after seeing R4, it would have been
>   indistinguishable from a re-frame.** That is the paper's strongest credibility move.
>
> ⛔ **ONE BINDING EDIT, and it is the only place the tier rule bites a good paragraph.**
> `.claude/outputs/bprime-c6.md` **§5 is written for you and you may lift it VERBATIM — except its
> final clause.** Cut *"…and it is the sharpest reason to move the claim from inference-time reads to
> training-time organisation."* **That clause is a tier-ii forward reference and §A13 forbids the
> reframe anywhere in tier i.** Everything before it — including *"a store organised well enough to be
> safe is organised well enough to be a table"*, which IS a tier-i conclusion — stays.

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **ADDENDUM 3 in full
(§A12 · §A13 — read the tier-i paragraph twice · §A14 · §A15)** plus **§A3 (B′ is a contribution, not a
retreat)**, **§A9.1/§A9.9/§A9.11**, **§A6 (the weak-proceed precedent, for how this program states
grades)**; `.claude/advisor-head-intervention.md` **§7 (papers) and §8 (prohibitions)**;
`.claude/outputs/track2-admissibility/PREREG-Bprime.md` **— this is the paper's pre-registration and the
table it IS**; `.claude/outputs/bprime-rivals.md` (**your spine, handed to you at release**);
`.claude/outputs/bprime-fb4-gate.md`; `.claude/outputs/bprime-fb1-recon.md` (**the novelty argument and
its concessions**); `.claude/outputs/bprime-theory.md` (**the theorem set: T1 byte floor + the
sharing/deletion frontier · T2 Prop D2a · T3 the one statement · T4 the seven protocol caveats with
domains · §9 DECLARED NOT DERIVED**); `.claude/outputs/bprime-c6.md` and
`.claude/outputs/harness-debt.md` (**both land this wave and both change numbers you will print**);
`.claude/claims_matrix.md` **§0 (the consolidated dated never-quote list) and CM-22/CM-23 (forbidden vs
approved wordings)**; `.claude/negative_results.md`; the **`2026-07-31 (later still)` `[C2W3]` §10
entry**.

⭐ **REGISTRY STATUS — CURRENT, for the first time in four waves.** You may quote the registries. **Three
live errata override them**: the byte law is **24/28** (corrected `[A(D+2)+d]/(d+m)`; floor **2.40×** at
`n_spec=1`) · **MUNKEY is an ICLR-2026 workshop paper (oral) — workshop name QUARANTINED — not ICML
2026, and it self-describes as "a memory-augmented transformer"** · **monitor #6's "27 post-repair" is
PROVISIONAL** unless `harness-debt` has landed its diff (check, then quote the corrected count or omit
the count entirely).

---

## 0. ⛔⛔ THE FRAMING RULE — read this before you write a word

**This paper is TIER i and ONLY tier i.** §A13 restructured the program's claims into three tiers:
- **tier i = B′, this paper** — *"when does test-time dynamics buy anything over a table at matched
  bytes?"*, applied uniformly to the CLU **and** the TTT/Delta family. **Its control is the
  matched-bytes / settle-deleted launder.**
- **tier ii** = the organization dividend / the cat test (C2W5; only *pre-registered* this wave).
- **tier iii** = the block on real streams (⚠ **"CLU-former" is a PLACEHOLDER NAME — Head ruling — and
  it must NEVER be baked into any draft**).

⭐⭐ **The rule, stated as the charter states it: tier i's CONCLUSION is the published justification for
the reframe — and the reframe is claimed NOWHERE in tier i.** You are writing the audit. You are **not**
writing "…and therefore the real dividend is at training time", **not** the organization dividend,
**not** the cat test, **not** the block. ⛔ **No forward-reference to tier ii or tier iii in any form**,
including as "future work" phrased as a promise. If the audit's conclusion invites the reframe, the
*reader* may draw it; the paper does not.

⚠ **And say the hard thing in the paper's own voice, early:** the settle-deleted launder tests whether
inference-time dynamics beat a table **given the organization** — both arms inherit the same placement.
That is what this paper measures, and stating its scope precisely is the paper's integrity, not its
weakness.

## 1. What the paper is

**The claim, one sentence (`PREREG-Bprime.md` §1):** one protocol — **matched-byte table launder +
two-sided byte ledger + a +0 B substitute audit + same-keys null** — applied uniformly to the modern
neural-memory family **and to the CLU**, reporting for each the dividend of its learned dynamics over a
byte-matched non-parametric store.

**Contribution structure, in the order a referee will test it:**
1. ⭐ **The protocol** — and the **learned-initial-state rule** is ours and is a contribution in its own
   right: *an initialisation is PARAMETERS; only the per-sequence deviation is STATE; both declared.*
   Counting the init as state **inflates**; counting the deviation as parameters **launders**. ⭐ **We
   apply it to our own `V_θ` init in the same table** — say so explicitly; it is the sentence that makes
   the audit credible.
2. ⭐ **The theorems** (from `bprime-theory.md`): the **byte-floor theorem** (corrected form
   `[A(D+2)+d]/(d+m)`, an accounting identity, exact 28/28 in rational arithmetic) and its corollary
   that **compression and byte-exact deletion are the same trade** with a computable exchange rate
   (`S* = (D+2)A_tot/m` items per atom — **7** at `A_tot=1`, **2387** at the shipped anchor; deletable
   fraction capped at **0.042 %** at `r=1`, `A_tot=341`); **Prop D2a**; and ⭐⭐ **T3 as ONE statement,
   both directions — a settled-point read is untrainable end-to-end** (`∂q*/∂q₀ = 0` **and** the
   mass-gauge dissolution), with the `e^{−C}` finite-budget law that **retro-explains w19 with zero
   fitted parameters**.
3. **The measurements** — the rival rows, the CLU column, the byte-frontier curve.
4. **The negative, stated as the finding it is.**

## 2. The numbers, and the frames they must never leave

- ⭐ **The CLU column (banked, C2W1):** the dividend is **≈0 or negative on every family measured**;
  substitute audit **0-for-4**; **exactly 0.0000 with `D = 0`** where the store works perfectly (Prop
  D2a, three independent confirmations); byte-exact deletion **AUC 0.5000 ± 0.0000**, byte-equal
  **3072/3072**; the accuracy-vs-bytes curve `decode 0.972 → 0.097` as the ratio falls `478× → 2.28×`.
- ⭐ **FB4 does not fire, and its content is the paper's honesty:** *"everything is at ceiling" is
  FALSE* — but on **three of four** designed families something costing **≤4 B** sits at the metric's
  exact maximum, **and it is never the CLU.** On `overload` the CLU reads **0.9722 ± 0.0139**, below
  three different readers of a table costing **1/478th** of its bytes. `S(f)`: overload **1.0000** ·
  **aggregate 0.5068 (the sole survivor)** · recency **1.0000** · manifold **1.0000**.
- ⭐ **The generalisable design rule, and it belongs in the paper:** `aggregate` survives for one reason
  — **its target is constructed to be absent from the table.** *"The answer is provably not in the
  table"* is the **only** property that has survived a +0 B audit in two waves (C2W1 0-for-4; C2W3
  1-of-4).
- ⛔ **`overload@load1x_shipped` is a BYTE-FRONTIER column, never a dividend family**, and every
  appearance carries the label plus the declared secondary reading **`S_excl = 0.6500`**.
- ⛔ **The one-family thinness goes in Limitations VERBATIM** (§A14.2, binding): two rival families
  audited against **one** surviving synthetic family is a thin cross-family audit. Do not bury it.
- ⛔ **No C2W3-or-later cell is a byte-matched dividend** — the minimum ratio anywhere is **17.11×**.

### 2.1 ⭐ The novelty argument — concede the ancestry, it is stronger that way
`bprime-fb1-recon`: FB1 does **not** fire (14 candidates: 0 HIT · 2 PARTIAL, both out-of-family · 7
NEAR-MISS · 5 NO) and FB5 does not fire (arXiv:2501.12352 is purely theoretical). But **P4 is narrowed
and the narrowed sentence is what you write**:
- ⛔ **Cite, do not suppress, B′'s methodological ancestry** — audit-at-equal-bits is standard **outside**
  the family: learned Bloom filters, learned indexes, SOSD.
- ⛔ **Cite the token-matched trivial control** published days before filing in LLM-agent memory
  evaluation (arXiv:2607.21962).
- ⛔ **The substitute-audit idea is not ours in general form** — it is the partial-input / trivial-baseline
  audit tradition (**Poliak et al. 2018; Feng, Wallace & Boyd-Graber, ACL 2019**). **Concede it in the
  related-work section, in our own voice.** (Reconciliation 8, owner: *whoever drafts B′*. That is you.)
- ⭐ **The surviving claim:** *seven independent groups built the adjacent instrument and none closed the
  loop* — a byte axis (Based), an iso-state normaliser (MAD), a state/param ratio (SDM), a matched
  trivial-policy control (HOLA), a compute-priced datastore (MassiveDS). **"A conceded ancestor is worth
  more than a contested monopoly."**

### 2.2 Deletion stays in the "and also" position
Charter §A9.9 + intervention §8.3. Phrase on **verified byte-exactness**, never *"we alone delete."*
⚠ **Say it before a referee does: a table deletes exactly by construction** — exact deletion is a result
only for a *learned/superposed* store. ⚠ Record the **MUNKEY narrowing with the corrected venue**
(ICLR-2026 workshop, oral; **name the workshop nowhere**; "a memory-augmented transformer"; MIA-AUROC→0.5
by design but **not exact**, gap to retraining **0.56 ± 0.21**).

### 2.3 What `bprime-c6` gives you, and its exact frame
The one coupling a per-slot table **provably cannot express** is **third-party store attribution** — the
change in a read caused by a stored item the query did **not** select; a row-selecting table's derivative
w.r.t. a non-selected row is **exactly 0** (Prop T5.4). ⭐ **This is an audit column — *"here is what a
table structurally cannot do, measured"* — and it is NOT a Route-3 revival.** §A9.5's kill stands,
scoped by §A14.1 to inference-read claims. ⚠ **Carry its magnitude honestly:** the coupling is suppressed
as `exp(−½(d/s)²)`, so **our own admission gate puts it at ~7e-4** while the rig actually run puts it at
**O(1)** — quote `bprime-c6`'s measured curve, not the two-well toy's prediction, and carry the caveat
that **`s` for a learned multi-atom well is an unsolved modelling question**.

## 3. Policy on drafting (charter + protocol, non-negotiable)
- **Appendix maximalism:** main results in main text; **everything else** — corollaries, negatives,
  extra plots, the full cell tables — in appendices. **Nothing is omitted before the dedicated pruning
  passes.**
- ⛔ **You do NOT invent results, run experiments, or touch `chlu/` code.** Every number traces to a
  named artifact. If a number you want does not exist, it is a **NOT-RUN** and you write it as one.
- ⛔ **Declared NOT-RUNs are never reported as nulls.** The C2W3 NOT-RUN list is long and explicit —
  reproduce it in an appendix: no Titans arm · TTT-MLP if skipped ⇒ **P3 NOT-RUN, not refuted** ·
  `overload` at the base atom budget · a **trained** attention reader · a soft-certificate sweep over
  `B` · OQ-2 · `route3-stage2` in its entirety · whatever `bprime-rivals` declares.
- **Multi-seed before any paper number.** A single-seed cell is printed as `n = 1` and is never a
  headline. **Quote the curve, not the endpoint.**
- **Flag-provenance tables travel into the paper** (protocol §5) — reviewers reproducing across sections
  must not find apparent contradictions.
- ⚠ Carry the **theorist's DECLARED NOT DERIVED list** (`bprime-theory.md` §9) as caveats, especially:
  the harness's own particle-gradient **prefactor is open**; **`s` for a learned multi-atom well is
  unsolved and gates the transfer of every domain statement**; **T5.4's coupling list is not proven
  exhaustive**; `B = 0.33`'s edge was located with a broken ruler (`bprime-c6` re-locates it this wave —
  use their number).

## 4. ⛔ Never-quote (full dated list: `claims_matrix.md` §0; and check CM-22 forbidden wordings)
⭐ **"CLU-former"**, or any tier-ii/tier-iii claim (§A13) · **Titans as "a preprint"** (NeurIPS 2025) ·
any **SDM Table 1 state/param ratio** (two extractions conflict) · **MUNKEY as "ICML 2026"** or with a
named workshop · **"verified to 1e-9 in all 28 cells"** (**24/28**; use the corrected law) ·
**"MAD `compression` is the admissible synthetic"** (dead by arithmetic: **73× at fp32 / 36× at bf16**
over the 224 B max payload) · **"principled forgetting"** as a novelty phrase · **"we alone delete"** ·
the anytime curve as a **uniqueness** claim (**shape only** — the figure is occupied by DEQs / EBTs /
Titans-Revisited) · **"certified"** unlearning / **"exact deletion"** / **"deletion by construction"**
(CM-22 (m)/(n)/(o)) · **"Guo et al. Def. 1 / Def. 2"** (does not exist — ε-certified removal is **§3
Eq. (1)**, inline) · **monitor #6's counts** ("58 trips" without *"pre-repair"*; "27 post-repair" is
**PROVISIONAL**) · the recency family's **`0.3019 ± 0.0679`** as a null (scoring-domain **defect**;
post-fix **−0.0028 ± 0.0619**) · the ridge saddle **`λ_min = −0.5946`** as multi-seed (seed 0; 3-seed
mean **+0.177 ± 0.469**) · **`sep/2`** as a certified inradius · **`λ_min > 0`** as certifying a nonempty
basin (**0.000** at `λ_min = +0.910`) · any **`AttentionPsi`** trajectory number · **`ε` as the
manifold-payload lifetime dial ∝ 1/ε** without the `2α` ceiling (`τ_max = Γ/2α`) · **`k*`** without *"of
`∂q_N/∂θ`, and only where the fixed-point sensitivity dominates the transient"* · **"Prop D1 is
violated"** (retired) · any C2W3-or-later cell as a **byte-matched** dividend (**17.11×** min).

## 5. Deliverables
1. **The draft** — `.claude/papers/bprime/draft-v1.md`. Length/venue framing per intervention §7: scope
   by venue expectations, **not by calendar** (venue timing and titles are the Head's alone — do not
   name a venue, a deadline, or an author list).
2. **A claims-consistency check** — every claim mapped to its artifact + its caveat, in a table, in
   `.claude/outputs/bprime-draft.md`. **Any claim without an artifact is cut or marked NOT-RUN.**
3. **The Limitations section**, containing verbatim: the one-family thinness · the ≥2.20×/2.40× byte
   ratio caveat on every dividend/byte claim (§A3, until a shared substrate lands) · the conceded
   ancestry (§2.1) · "a table deletes exactly by construction" · the launder's scope (§0's hard
   sentence) · the theorist's not-derived list · ⭐ **FB2's measured-vs-reasoned split** (3 of 5
   families measured) · ⭐⭐ **the F3-lite tuning grid, and the direction of its bias, stated by us.**

### 3.1 ⭐⭐ The F3-lite limitation — write it before a referee finds it
The rival arms were tuned on a **reduced grid** (Adam, 400 steps, `lr ∈ {1e-3, 3.16e-3, 1e-2}`; TTT also
`b ∈ {1,16}`), declared by the engineer as *"a budget choice, not presented as `rival-recon` F3
compliance"*. ⚠ **The audit's finding is "rivals lose to their own byte-matched tables", and
under-tuning a rival produces exactly that finding.** The bias runs **toward** our headline, which is
the dangerous direction, and **N78 (rescued baselines) is a standing program value.**
✅ **RULED (Head, 2026-08-01): the full-F3 pass IS FUNDED and runs this wave as `bprime-rivals-f3`**
(the three rescued arms mandatory; the two non-rescued as a rider). ⛔ **Your NUMBER-FREEZE is gated on
its before/after table** — you may start framing, related work, theorems, and limitations now, but **no
rival number is frozen into the draft until `.claude/outputs/bprime-rivals-f3.md` lands**, and ⭐ **the
Head's pre-commitment is binding: if tuning changes any outcome, the paper's claim changes with it.**
Quote the full-F3 numbers; describe the tuning as the F3 standard met, with the C2W4 reduced-grid run
disclosed as the audit's first pass.
⚠ **Related and mandatory: the RESCUE GATE is first-class in every table.** An arm within 2 SE of its
own blank-store control is **NOT RESCUED and no margin against it is quotable** — that disqualified
**`ttt_mlp` and `deltanet`** on `aggregate` and **all 5 arms** on the byte-frontier column. ⛔ **You may
not write "the CLU reads 0.972 vs the rivals' 0.10" on the frontier column** — the engineer explicitly
declined to, and so do you.
4. ⭐ **A referee pass before anything else** (§A15.5: *"referee pass before anything else"*) — the Hub
   hands the draft to `paper-referee` and you revise against the report. **Budget for the revision; the
   draft is not done when v1 exists.**

## 6. Output — `.claude/outputs/bprime-draft.md`, protocol §5 format
- the **claims→artifact→caveat table** (deliverable 2), first screen;
- **every NOT-RUN the draft declares**, listed;
- **every place you could not source a number** and what you did instead;
- **your reconciliation list in the FIRST 10 LINES** if you produce one — including any place the
  registries, the charter and the outputs disagree (you are reading all three; you will find some);
- the FB2/FB3 framing you were handed and how the draft reflects it.
