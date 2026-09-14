# bprime-referee — paper-referee report (C2W5, on `draft-r2.md`)

Task + acceptance criterion: adversarial referee pass on the B′ tier-i audit paper (`.claude/papers/bprime/draft-r2.md`), scored against frozen n=9 numbers; itemized MUST/SHOULD/NICE + simulated verdict; nothing fixed by me.
Status: **done** (read-only; no draft or repo file touched).

## ⚠ RECONCILIATION LIST — needs a Hub owner, first 10 lines
1. ⛔ **MF-1 is a genuine arithmetic/logic defect in the audit table:** the CLU row prints ✅ RESCUED "against blank −0.4221" while its own full read is **−0.5261 ± 0.0863 — BELOW its blank** — i.e. the CLU **fails its own rescue gate** under B.5 as written. Owner: **r3 writer** (verdict + honest sentence) and **Hub** (whether to fund the CLU n=9 column, which also closes the paper's visible n-asymmetry).
2. ⛔ **MF-3 is a claims-matrix compliance gap:** CM-27(c)'s mandatory caveats (the anchor-vs-family objection; the coverage clause) do not appear anywhere in the draft. Owner: **r3 writer**; Hub to confirm which coverage statement CM-27(c)'s "(0/3 on two families)" refers to.
3. ⚠ **CM-28(ccc) exemption needs a ruling:** App H prints first-pass (C2W4) rival numbers with no F3 before/after verdict beside them (the column was NOT-RUN under F3). Either the Hub ratifies the in-place NOT-RUN note as (ccc)-compliant, or funds the ~5-min labelled deltanet frontier row (A18.6 rider does the latter anyway). Owner: **Hub**.

---

# 0. DIAL DECLARATION (echoed, protocol §7)
- **Dial:** none — instrument/review (referee pass; no new claim).
- **Laundering control:** n/a; the pass itself audits the draft's controls.
- **Falsifies:** n/a. **Does NOT falsify:** a finding of mine being closable by drafting does not downgrade the draft (that is what r3 is for).

---

# 1. VERDICT (simulated)

**Weak-accept at an analysis/benchmark/audit-style venue, conditional on MF-1–MF-5. Borderline-reject at a first-tier main conference main track** — not on honesty (the honesty apparatus is the best I have refereed in this program) but on G3 coverage: one designed synthetic family at `d_in = 5`, three effective rival arms sharing one state type, CPU scale, no real-data leg.

**Meta-review (as the venue would write it).** The paper asks a question the neural-memory family genuinely does not ask of itself, builds a defensible protocol (the learned-initial-state rule and the projected-vs-raw control finding are real methodological contributions), pre-registers both controls, tunes the rivals properly and proves it, applies the audit to its own system, and reports a negative headline for everyone including itself. The statistics discipline (n=9 rescue verdicts, withdrawn n=3 verdicts, the init-redraw disclosure) is exemplary. Against that: the entire empirical verdict rests on **one** author-designed synthetic family that the authors' own validation left standing out of four, evaluated at toy scale, with two of five rival arms failing or straddling the authors' own functioning-check — and, fatally for internal consistency as drafted, **the authors' own store fails the same functioning-check and is printed as passing it**. The task family and the rival implementations are never actually specified in the paper, so the audit is not reviewable as submitted. The pre-registration ordering carries the paper's central admissibility argument yet no registration artifact is cited. Fix those and this is a useful, citable audit; the scale and coverage ceiling on the contribution remains.

---

# 2. MUST-FIX (blocks submission)

**MF-1 — The CLU fails its own rescue gate and is printed ✅ RESCUED. (§4.1.1 audit table, CLU row; §2.2; B.5.)**
The attack: B.5 defines RESCUED as *full read exceeds own blank-store control by more than 2 SE of the lift*. The CLU's full read is **−0.5261 ± 0.0863**; its blank is **−0.4221** (printed in the same table's parenthetical, in I.1c, and in §2.5). The lift is **−0.104 — negative**: the written store reads *worse* than the empty one. Under the paper's own gate the CLU is NOT RESCUED; the printed "✅ (against blank −0.4221)" is arithmetically unsupported by the paper's own numbers, and the fact itself (store below blank) is never stated anywhere. This is the single sentence a hostile reviewer builds the review around: *"the authors disqualify rival arms sitting at their blank-store floor, while their own arm sits below its blank floor and keeps a tick in the same table."* It also collides with §2.2's "we report no three-seed rescue verdict, including our own first pass's" — the CLU ✅ *is* a three-seed rescue verdict (A18.1 never-quote: any rescue verdict at n=3).
Evidence: §4.1.1 (full −0.5261 = launder −0.4472 + dividend −0.0789, both banked); I.1c (blank −0.4221); B.5 (gate definition); A18.1.
Fix direction (writer, no new run needed): print the CLU's gate status honestly (NOT RESCUED / below blank), state the below-blank fact as the finding it is (it is *consistent* with the paper's thesis — the written content does not lift the read), and extend §4.2's "honest qualifier on the arm count" to the CLU exactly as it is extended to TTT-MLP. The clean upgrade is the CLU n=9 column (Hub call — also closes SF-4/risk-2). **Note: fixed properly, this strengthens the paper.**

**MF-2 — The paper never specifies the task family or the rival implementations. (whole draft; no appendix exists for either.)**
The attack: the sole surviving family, `aggregate`, is the entire empirical basis — and it is never defined. What is stored, what a query is, what the aggregation target is, what neg-MAE is computed over, what chance level is, how the "answer provably not in the table" construction rule works operationally — none of it is in the paper (only fragments: §4.1.1's coverage note, §2.5's one-line rule). Likewise the rival arms: "minimal faithful TTT-class / delta-rule memories" appear with `d_head` and byte counts but no architecture, inner objective, update rule, or faithfulness argument. A reviewer cannot check the metric-nativeness table (§4.3) or the losslessness of the tables (B.2) without them. Appendices A–M cover provenance/theorems/caveats but not the two objects the audit is *about*.
Triage: **MUST-FIX** — this is "exists in outputs but not cited": the specifications live in `bprime-rivals.md` / `memory-gym` artifacts; the r3 writer ports them into two new appendices (task-family definitions incl. the struck families; rival-arm specifications incl. the iso-state head-width derivation). No new experiment.

**MF-3 — CM-27(c)'s mandatory caveats are absent. (§2.5.)**
The attack (and the matrix's): CM-27(c) approves the FB4 wording *only* "always with the anchor-vs-family objection and the one-family thinness" and "never without the coverage (0/3 on two families)". L1 carries the thinness; **the engineer's unresolved objection — the rule cannot distinguish "the family is substitutable" from "the family's ANCHOR is substitutable" (N170/N171) — appears nowhere**, and neither does the coverage clause. This is precisely the objection a referee raises unprompted against §2.5's "three of four struck" result: perhaps the *anchors* were substitutable, not the families. The register's own doctrine is to say it first.
Fix: add the objection in §2.5 (one sentence, in our voice) + the coverage clause; Hub confirms the referent of "(0/3 on two families)" (the N186 write-admissibility record) so the wording is exact.

**MF-4 — §2.2 states an n=3 frontier rescue verdict as fact, contradicting §4.5/App H. (§2.2, last sentence of the gate paragraph.)**
"…and on the byte-frontier column no arm clears it at all (§4.5)" is exactly the three-seed rescue verdict that §4.5 explicitly refuses to draw ("we do not report '0 of 5 not rescued' as a verdict either") and that A18.1 forbids at n=3. §6 L2 has the correctly-labelled observational form; §2.2 does not.
Fix: rewrite the clause to the labelled observational form ("no arm separated from its control at the three seeds run; the column carries no quotable verdict, §4.5").

**MF-5 — The pre-registration ordering is load-bearing and unverifiable as printed; "previously published" misdescribes internal artifacts. (§2.4, §3.1, §4.6, App I preamble.)**
The attack: §2.4 states "the registration order is what makes §4.3 admissible" and §4.3 repeats it — the paper's central credibility move — yet no registration artifact is cited, dated, or promised as supplementary. A reviewer has only the authors' word for the ordering. Separately, §3.1's erratum says "a previously published closed form" (and §4.6 "a prior estimate", §5.4 "our own earlier notes"): to an external reader "published" implies a public prior paper that does not exist — the closed form lived in an internal pre-registration. That is both inaccurate and an anonymity/consistency landmine (M2: a reviewer hunting for the "prior publication" finds nothing).
Fix: (a) one sentence + appendix note committing the pre-registration documents (with dates) to supplementary material; (b) reword "previously published" → "the closed form stated in our pre-registration" (ERRATA-Bprime's replacement sentence licenses this content; only the register changes).

---

# 3. SHOULD-FIX

**SF-1 — "No margin against TTT-MLP is quotable anywhere in this paper" is literally contradicted by the abstract.** (§4.1.1 vs Abstract/§4.2/§7.) The abstract's headline range −0.2592…−0.4602 *contains* TTT-MLP's −0.4425 and TTT-Linear's −0.4602. The draft's implicit rule — the gate suppresses *comparative* margins in the flattering direction, not an arm's own loss-to-its-table (CM-29(a) blesses the 5-arm range) — is workable but never stated. Define the direction rule once in B.5 and reword §4.1.1's sentence ("no comparative margin *in favour of any other arm over* TTT-MLP…"); otherwise a referee quotes the two sentences side by side.

**SF-2 — TTT-Linear's per-seed configs break the single-row ledger (C-7 apparent-contradiction risk).** §4.1.1 prints ttt_linear as `d_head = 29`, F2 = 5220 B, table 5104 B; A.1's chosen configs include **b = 1** cells, and A.1b states b1 implies **d = 36** (F2 = 5184 B). A reviewer reproducing the ledger from the chosen configs constructs exactly the cross-section contradiction C-7 exists to prevent. Fix: state the `b → (d_head, F2, rows)` mapping once (B.2 or a table footnote) and label the §4.1.1 row as the b16 configuration (iso-state 5456 B is the invariant either way — say so).

**SF-3 — The CLU dividend −0.0789 is quoted ≥4× (abstract-adjacent §4.2, §4.3 twice, §7) with no SE and no n at point of use.** It is load-bearing for the weak-FB3 adjudication ("does not pay for ours"). Attach `n = 3` and a per-seed spread (exists in the banked artifact) at first use.

**SF-4 — Power/stopping optics.** (a) §4.2's robustness sentence quotes the 5× (−0.2184…−0.2630) and held-out (−0.24…−0.49) ranges without in-sentence n=3 labels (labelled only in I.1d). (b) Seeds 3–8 are a post-hoc power addition pooled into the primary; a referee will raise optional stopping. Both closable in one sentence: the addition was declared before pooling, the registered n=3 primary yields the same headline sign on every arm (I.1a shows it), and the verdict changes at n=9 ran *against* the flattering direction (deltanet rescued ⇒ more arms functioning). The material exists; wire it.

**SF-5 — "Seven independent groups" lists five.** (§5.3 closing ⭐ paragraph names Based, MAD, SDM, HOLA, MassiveDS.) Name all seven (fb1-recon's list — presumably + Zoology and kNN-LM) or count five.

**SF-6 — §4.6.1 quotes "1.8" and "4.3 well-widths" without naming the ruler** (1.8/4.3 are the atom-width ruler; the fitted ruler reads 1.10/3.72) while L9(ii) claims "every `d/s` statement here names its ruler" — CM-28(ppp)'s exact form. The paragraph is CM-29(f) approved wording, so the fix is a two-word bracket "(atom-width ruler)" at each, keeping the approved sentence intact.

**SF-7 — App H / CM-28(ccc):** first-pass rival numbers printed with no F3 counterpart. The in-place NOT-RUN note is honest; the ratified clause is absolute. Hub ruling or the 5-min row (reconciliation 3). The A18.6 deltanet frontier rider, when it lands at r3, partially restores the column anyway — make sure App H is re-pointed at it.

**SF-8 — Figures: five specifications, zero renders.** Blocking at submission; Figure 1 is renderable now (gate closed). Engineer/analyst task, not the writer's.

**SF-9 — Citations beyond the parallel scout's list:** (a) arXiv:2605.17590 (counterfactual-state-alignment unlearning, §4.7) appears in **no** program artifact I audited — verify it exists and says what §4.7 claims before printing; (b) §1's "Gated DeltaNet-2, 2026" is authorless (known GDN-2 BibTeX gap — confirm `bprime-cite-check` covers the in-text form too); (c) "Guo et al." (§5.4) has no year in text.

**SF-10 — Commission the 10-min n=9 re-aggregation (f3 follow-up 2, already riding with r3).** It removes App I.1c's mixed-n residue, lets §4.3 quote the dividend directly instead of the two-component sign argument (which a referee will call an inference the authors declined to compute), and discharges the writer's reconciliations 1–2.

---

# 4. NICE

- **N-1:** strip ledger glyphs (⭐⛔⚠✅◐) and meta-commentary register before LaTeX; several paragraphs read as internal notes ("this is a contribution, not a caveat").
- **N-2:** cut "a mis-citation we found in our own earlier notes" (§5.4) — internal-process leakage with no reader value.
- **N-3:** a half-page "the store under audit" subsection (write procedure + read map in one place; currently scattered across §3.1/§3.4/A.2–A.3).
- **N-4:** App H's CLU 0.9722 lacks its ±0.0139 (present in §2.5).
- **N-5:** length — main text is conference-short-plus; the C-10 pruning pass decides, nothing to do now.
- **N-6:** the "design identity" generalization (§4.6.1/§7) is bounded by L9(iii) (coupling list not exhaustive) but the conclusion's phrasing "rather than a defect of one implementation" invites "you measured one implementation"; consider "of this implementation class" — optional, CM-29(f) covers the current form.

---

# 5. Compliance summary (checked item by item)

- **A18.1 never-quotes:** no n=3 rescue verdict as a claim **except MF-1's CLU tick and MF-4's §2.2 clause**; "≥ 4.4 SE" correct (min = 0.4602/0.1038 = 4.43; the stale "3.6 SE" appears nowhere in the draft); no ∞ gradient ratio anywhere (T3 quotes trainability-spike's 2.42e6 and the C2W2 bitwise-0.0 mass gradient — correctly sourced, no pilot conflation); no pilot number in the draft at all; γ statements carry read budgets (A.3, §3.4's C = 18.34); rival numbers are F3 numbers with the before/after in §2.6 + I.1a (App H exception = SF-7).
- **Mixed-n labelling (Hub attack surface 2):** near-airtight — abstract, §4 preamble, §4.1/4.1.1, §4.2, §7, App K caption rule all label. Residue: SF-3 (−0.0789 bare), SF-4a (§4.2 stress ranges), App I.1c (by design, claim-free, labelled). Verdict: **honest-and-temporary as ruled; the r3 riders close most of it.**
- **Tier-i discipline:** verified independently — no "CLU-former", no tier-ii/iii vocabulary, no future-work section, no reframe (§1.2/L3 state the launder's scope and stop; §4.6.1 ends at "design identity" with the forbidden clause cut); C-1 post-reversal form holds (no defensive audit paragraph; the errata live where the numbers live, which is C-9/appendix-maximalism, not audit-confession).
- **Positioning Charter:** C-2 exemplary (§1.3 verification/evidence split is a model); C-5 headline/conclusion carry scale qualifiers in-sentence, no scope-free "CLUs provide" found; C-6 fine print adjacent to claims (deletion conditions in the same sentence, §4.7); C-7 five provenance tables (SF-2 is the one crack); C-8 hermetic (J&P 2026 only, third person); C-9 App L = 17 negatives incl. three own refuted preregs; C-10 held.
- **CM-26/27/28/29:** CM-29(a)/(b)/(c)/(d)/(f) reproduced within approved wording; CM-26(kk)–(oo), (uu)/(vv) (scoped §5.2 sentence, dated), (ww) (GDN-2 reference arm) all compliant; CM-28(aaa)/(bbb)/(ddd)/(eee) compliant; MUNKEY per ERRATA E2 (no workshop named, "memory-augmented transformer", gap 0.56±0.21); byte law in corrected form everywhere with the E1 erratum printed; Titans NeurIPS 2025; SDM ratios quoted nowhere. **Gaps: MF-3 (CM-27(c) caveats), SF-6 (ppp), SF-7 (ccc).**
- **Statistical honesty (attack surface 5):** gate-power finding first-class (§2.2, L2a, I.1d); ttt_linear UNSTABLE with both readings printed; gdn2 +0.0473±0.0277 stated as a tie and never "the one rival that beats its table"; "4 of 5 ≤ 0" correct; withdrawn first-pass verdicts stated as withdrawn. Clean except MF-1.
- **Number spot-audit:** headline range, handicaps, tuning/redraw decomposition (P3′ = sum verified), 2000-step deltas, rescue lifts, byte identities (5456/100/54.56; floors 2.20/2.40; S* 7/2387; p ≤ 4.19e-4; dp/dr 2.10e-3), ledger rows (5220/5104 = 1.023), C6 (4.34/3.72, 1.53e-2, 525×, s three-way 0.7%), SE multiples (8.9/9.4/6.9) — **all reproduce from the artifacts. No silently rounded or scope-widened number found beyond the items above.**

---

# 6. Missing-experiment list (for the Hub; "genuinely missing", not wiring)

1. **CLU column at n=9** — closes MF-1 cleanly + the paper's visible n-asymmetry (writer's own risk 2). Cheap; highest-leverage.
2. **n=9 re-aggregation of projected-launder/dividend/blank/same-keys columns** (~10 min, f3 follow-up 2 — already an r3 rider; confirm it actually runs).
3. **Labelled deltanet frontier row** (~5 min, A18.6 rider) — restores App H and settles SF-7.
4. **TTT-Linear verdict resolution** — paired per-seed blank control or init-averaged control (f3 open question 1); it is load-bearing for L2 ("three rescued arms" vs "four").
5. **Mamba-2 arm; GRU/SWA arm** — priced, unfunded (§A17.3; Head call at the C2W5 review). My read on brief item 4: **L5 as written survives review at an analysis/audit venue** (measured-vs-reasoned split is explicit, the closure is priced in the paper's own voice); a first-tier reviewer will demand Mamba-2 specifically, since the survey sentence (§5.2) names SSMs and none is measured — if only one closure is funded, fund Mamba-2.
6. **A second dividend family built to the §2.5 rule** — L1 names it as the cheapest strengthening; also the only real answer to the anchor-vs-family objection (MF-3).
7. **Figure renders** (Fig 1–5; engineer/analyst with plotting scope).
8. Not experiments but blocking: Feng et al. quote verification (in flight, `bprime-cite-check`); arXiv:2605.17590 verification (SF-9a); supplementary prereg package (MF-5).

---

# 7. The three sentences a hostile reviewer would quote

1. *"Every measured cell in this paper runs at `d_in = 5`, 5–6 stored items, ~10-token streams, on CPU"* — an audit that indicts a family of language-model memories without running any arm within six orders of magnitude of its operating regime.
2. *"Two rival families audited against **one** surviving synthetic family is a thin cross-family audit"* — by the authors' own accounting the verdict rests on three arms sharing one state type, on one task the authors designed, after their own instrument struck the other three tasks they designed.
3. *"An arm whose full read sits within 2 SE of its own blank-store control is NOT RESCUED, and no margin against it is quotable"* — a rule the authors apply to disqualify two rival arms while their own store reads **below** its blank-store control (−0.5261 vs −0.4221) and carries a ✅ in the same table.

(3) is erased by MF-1's fix; (1) and (2) are the paper's honest ceiling and are already stated in its own voice — which is the correct defense, and the only one.

---

## Proposed handover updates (for the Hub)
1. **§10:** `bprime-referee` delivered on r2 — simulated **weak-accept (analysis/benchmark venue) conditional on 5 MUST-FIX**; borderline-reject at a first-tier main track on coverage/scale. Headline numbers all reproduce; the one substantive defect is the CLU rescue-gate tick (MF-1), which is a *drafting* error whose honest fix strengthens the thesis (the store reads below its own blank — the paper's point, unstated).
2. **r3 task must include:** MF-1–MF-5 + SF-1–SF-6 (writer-closable); the two funded riders (n=9 re-aggregation, deltanet frontier row) wired to I.1c/App H; the supplementary prereg package decision (MF-5) is Head/Hub-owned (interacts with M2 anonymity).
3. **Ruling owed:** CM-28(ccc) exemption for App H (SF-7); referent of CM-27(c)'s coverage clause (MF-3); whether to fund the CLU n=9 column (missing-experiment 1) and Mamba-2 (5) before venue selection.
4. **Never-quote candidate:** the CLU's "✅ rescued" at n=3 — forbidden on the same A18.1 basis as the rival n=3 verdicts, and additionally false under B.5's own definition.
