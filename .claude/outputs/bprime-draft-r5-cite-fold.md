# bprime-draft-r5-cite-fold — paper-writer report
Task + acceptance criterion: fold `bprime-cite-check-2` + `bprime-cite-check-3` into `draft-r4.md` → produce `draft-r5.md` (citation layer only; ⛔ no measured number moves), remove all 8 `CITE2` markers, fix the §5.3 mis-attribution, lift the GDN quotation restriction, paste the verified bibliography, and print the claims-matrix v2.11 §0.9/CM-30 sweep.
Status: **done** — all 7 fold items applied; 48-pattern sweep printed; **0 real never-quote violations**; **0 measured numbers moved** (verified mechanically, §3).
**Dial declaration (protocol §7, echoed):** **none — instrument/editorial** (a citation-layer revision). No performance number produced, no laundering control applicable, nothing falsifiable in the dial sense. Nothing here touches admission · lifetimes · isolation · compute-adaptive reads.

## ⛔ DOWNSTREAM RECONCILIATION LIST (owner needed — protocol §5 corollary, first 10 lines)
1. ⛔⛔ **THE MIS-ATTRIBUTION IS UPSTREAM OF THE DRAFT AND IS STILL LIVE THERE.** `.claude/outputs/bprime-fb1-recon.md` **line 49 (survey row 14)** is where the draft got it, and its version is **worse than the draft's was**: it attributes *"30–80× larger than B-trees"* to **SOSD** *and* names **PGM** as the subject — PGM is the one index Chesetti & Pandey report as **4× *smaller***. The draft is now fixed; **fb1-recon is not**, and it is the survey of record that any future positioning artifact will read. **Owner needed** (curator erratum or a one-line correction block in that report). Sites checked and clean: `bprime-rivals.md` l.252 and `philosophy-synthesis.md` l.2112 mention SOSD/learned indexes with **no numbers** — no other artifact in `.claude/**` or `docs/**` carries the quotes.
2. ⚠ **I edited one negotiated sentence: §5.2's survey blockquote** ("the exact sentence we are entitled to"). *"at matched space in learned data structures (learned Bloom filters, learned indexes, learned sketches)"* → *"at matched or explicitly accounted space … (learned Bloom filters, where the learned model's own bytes are a **condition on the verdict**; learned indexes and sketches, where the structure's own space is a **reported column** beside its speed)"*. Rationale: it carried the same unsafe "matched space" attribution §1 did, and cite-check-3 §3.3 is explicit that **nobody in SOSD equalizes bytes**. No survey verdict, count or grade changed. **Hub ratification requested** — this is the one edit I would not have made unilaterally if the sentence were not factually implicated.
3. **The draft still has no reference list.** Appendix Q is scoped as *"the entries verified in a dedicated pass"*, **not** the bibliography. Assembling and verifying the remaining ~25 entries is a separate task (owner: a scout batch + the LaTeX pass). Editorial item **6b** now names every residual unverified/single-sourced citation so the task is pre-scoped.

---

# 1. What I did (per fold item)

Output: **`.claude/papers/bprime/draft-r5.md`** (2621 → **2825 lines**), CHANGELOG line appended per A18.5 with per-item disposition. `draft-r4.md` untouched.

| # | fold item | disposition |
|---|---|---|
| 1 | ⛔ §5.3 mis-attribution | **FIXED.** Built on cite-check-3 §3.3's suggested rewrite. Quotes re-attributed to **Chesetti & Pandey (ACDA 2025)**, pinned **arXiv:2407.00590v2 §6.6** in-sentence with *"the SIAM camera-ready is paywalled and we did not read it"*. Both caveats ride along: subject = **RadixSpline and RMI** (PGM **4× smaller** in the same paragraph) and the **every-256th-key** sparsified B-tree. Added that paper's own *"roughly a tie"* headline, and ⛔ **SOSD's opposite verdict printed beside it** (RMI 3 %, RS < 1 % vs B-tree 16 %; *"learned models indeed often outperform state-of-the-art implementations"*). Framing changed from *"the SOSD verdict that a learned index is 30–80× larger"* to *"we cite the adversarial study for the **tone of the accounting**, not for a verdict against learned structures."* |
| 2 | 8 × `CITE2` markers | **ALL REMOVED** (§1 ×2, §2.4, §5.1 ×2, §5.3, A.1f, editorial item 6). **Poliak:** §5.3's *"beat the majority baseline on 6 of 10"* → ***"significantly outperform** the majority-class baseline on **six of the ten** NLI datasets"*; §2.4's inline cite gains the venue (`\*SEM 2018` / ACL 2019). **Mamba-2:** §1 and §5.1 now cite **Dao & Gu, ICML 2024, PMLR 235:10041–10071, arXiv:2405.21060**; **A.1f's ⚠ provenance row became a ✅ discharge row** carrying the author-order trap, the triple-sourced `d_head × d_state` convention (the `P × N` special case our arm runs at `d_state = head_dim = 36`), the confirmed `chunk_size = 256` reference default we deviate from — and the honest residual that **the Mamba-2 paper body was never read**. |
| 3 | ⭐ GDN restriction lifted | **QUOTED, WITH THE HEDGE.** §5.1 now reads: Gated DeltaNet (**Yang, Kautz & Hatamizadeh**, ICLR 2025) introduces Mamba-2 as *"a simple gated update rule, `S_t = α_t S_{t−1} + v_t k_tᵀ`, which uniformly decays all key-value associations at each time step by a dynamic ratio, `α_t ∈ (0,1)`"*. ⚠ Immediately followed by GDN's own *"up to specific parameterization"* hedge **plus** the substantive consequence I added so the hedge is not decorative: Mamba-2's shipped parameterisation is a per-head `P × N` structured state, and **our arm's identity claim rests on Appendix O.2b's asserted-and-tested implementation, not on that sentence**. |
| 4 | §1 cite upgrades | **APPLIED, with cite-check-3 §3.3's safe/unsafe split.** The unsafe blanket (*"…compare at matched space (Mitzenmacher, 2018; Kipf et al., 2019)"*) is gone. Replaced by: Mitzenmacher **quoted from the published §3.1** as the line that makes total size a **condition on the verdict**; index benchmarks as **space-accounted reporting** (**Kipf et al., 2019, NeurIPS ML-for-Systems workshop, non-archival**; the accompanying PVLDB study, **Marcus et al., 2020** — named separately, never collapsed). A new short paragraph states the distinction explicitly and notes **this paper takes the stronger of the two** (bytes equalised, not merely reported) — which is a positioning gain, not just a correction. ⛔ No number is quoted from Marcus et al. |
| 5 | Bibliography | **NEW Appendix Q.** Q.1 = the **seven** verified entries (Mitzenmacher · Kipf/SOSD · Marcus · Chesetti & Pandey · Poliak · Dao & Gu · Yang et al.) verbatim from the scout reports, **caveat notes intact**. Q.2 = what is deliberately **not** printed: the **Belinkov** entry (unverified venue/pages, and this draft does not cite it — omitted rather than carried), and Poliak's ar5iv-sourced Table 2 decimals. Q's preamble states it is **not** the reference list. ⛔ Never-copy recorded in two places (Q.1's Poliak note + editorial item 6): **S2 mislabels \*SEM as SemEval; the ACL Anthology record governs.** |
| 6 | ⭐ §0.9 / CM-30 sweep | **RUN AND PRINTED (§2 below).** 48 patterns over the full 2825-line draft. **0 real violations**; 13 hit-clusters, every one adjudicated. Both named items are clean: **(a)** the §A20.2 form — the draft contains **no** occurrence of *"compositional family"*, *"occupancy read protocol"*, or any tier-ii object (this is the tier-i audit paper; the tier-ii wave never entered it); **(b)** `null*` = **0.00117** and the 584-config grid appear **nowhere** — the draft's "null" is exclusively the **same-keys null** column, a different object, and no "best null arm scored…" construction exists. |
| 7 | CHANGELOG | **Appended** (r-convention, per-item disposition, ~40 lines). |

**One clarity fix arose from the sweep and is the only non-citation edit:** App L 19's *"0 of 15"* now points at the column of record — *"(Appendix H.1 — the three-arm sub-table; **the column of record is 0 of 20** once the SSD arm is added, entry 23 below)"*. No number changed; it removes an apparent cross-section contradiction (C-7).

---

# 2. The claims-matrix v2.11 §0.9 / CM-30 sweep — printed

Script: `.claude/scratch/bprime-draft-r5/sweep.py` (48 patterns, case-insensitive, whole-document matching with line mapping so cross-line context windows resolve correctly). Raw output: `.claude/scratch/bprime-draft-r5/sweep-out.txt`.

```
clean  §0.9/CM-30(www) · 'the compositional family is refuted' in any form
clean  §0.9/CM-30(www) · 'compositional family' mentioned at all (must use the read-protocol form)
clean  §0.9/CM-30(www) · P-particle occupancy read protocol named without 'P = 4'
HIT*   §0.9/CM-30(xxx) · 0.272 / 27.2 % as an arm's bar or live launder bar   (1 — FALSE POSITIVE)
clean  §0.9/CM-30(xxx) · the 0.05039 arm bar quoted out of context
clean  §0.9/CM-30(yyy) · 'no matched-capacity organizer clears'
clean  §0.9/CM-30(zzz) · 'physics loses' / tier-ii null from the cat test
clean  §0.9/CM-30(zzz) · 'K3 ✅ K4 ✅' as evidence of soundness
clean  §0.9      · null* = 0.00117 as an arm score / 'the best null arm scored'
HIT*   §0.9      · occupancy scored against F/N_a (= 0.125)                   (2 — FALSE POSITIVE)
clean  §0.9      · P > 4 sweep results quoted as arm scores
clean  §0.9      · OD_min = −0.0016 quoted as the wave's null*
HIT*   §0.9/CM-30(aaaa) · the CLU 'RESCUED' at n = 3                          (4 — all NOT RESCUED)
clean  §0.9/CM-30(bbbb) · headline '0 of 5' rival arms
HIT*   §0.9/CM-30(cccc) · unlabelled 54.56× / 5456 B (modal-value rule)       (22 sites — all labelled)
HIT*   §0.9/CM-30(cccc) · a single TTT byte figure as *the* n = 9 value       (1 — labelled per-seed)
clean  §0.9/CM-30(dddd) · deltanet or gdn2 quoted as plain RESCUED
clean  §0.9      · a dividend sentence saying the launder 'beats' the store
clean  §0.9      · a quotable margin against Mamba-2 on the overload frontier
HIT*   §0.9      · d/s stated without naming its ruler                        (1 — designed sweep)
clean  §0.9      · s = 0.40 quoted as refuted rather than FLAGGED
clean  §0.9      · '77–84 % of wall-clock is the Python plan pass'
clean  §0.9      · 'localized init / a trajectory write term fixes the store'
clean  §0.9      · a probe or cat-test reading quoted without its scope
clean  A18.1     · '≥ 3.6 SE' in any form (the minimum is 4.4 SE)
clean  A18.1     · an n = 3 rescue verdict quoted as live
HIT*   A18.1     · 'R5: 3 of 5' unlabelled                                    (4 — labelled before/after)
clean  A18.1     · the n = 3 weak-inversion dividends +1.02 / +0.88 unpaired
clean  A18.1     · 'CLU-former' as a name / cluformer numbers
HIT*   r4        · CLU still reading *below* its own blank (n = 3 drama)      (1 — `recency` family)
HIT*   r4        · stale CLU −0.3180 / −0.5261 outside a supersession block   (3 — all inside one)
clean  r4        · stale d/s 3.72 as the audited cell
clean  r4        · stale ±0.40 on the 0.814 row
clean  r4        · Mamba-2 still listed as NOT-RUN / outside the ruled set
HIT*   r4        · '0 of 15' unqualified frontier claim                       (4 — all scoped)
clean  charter C-1 (REVERSED 2026-07-07) · defensive audit-confession paragraph
clean  charter   · legacy paper's mechanism-numbers asserted as evidence
clean  charter   · bare 'mass' where inertial M vs spectral μ is meant
clean  r5        · the 30–80× / 4-orders quotes still attributed to SOSD or Kipf
clean  r5        · any leftover CITE2 marker
clean  r5        · Mamba-2 attributed to 'Gu & Dao' (author-order trap)
HIT*   r5        · Poliak's venue given as SemEval                            (2 — never-copy warnings)
HIT*   r5        · 30–80× printed without its RadixSpline/RMI subject         (1 — editorial item)
HIT*   r5        · Chesetti & Pandey without the arXiv version pin            (3 — BibTeX + editorial)
clean  r5        · the GDN equation quoted without its 'up to spec. param.' hedge
clean  r5        · Kipf/SOSD cited as an archival/proceedings paper
clean  r5        · Marcus/PVLDB numbers quoted (body text unverified)
```
**Verdict: 48 patterns, 13 hit-clusters, 0 real violations.**

**Adjudications, one line each (nothing waved through):**
| hit | line(s) | adjudication |
|---|---|---|
| `27.2 %` | 2114 | substring of a tuning-grid cell value **`0.2729`**. False positive. |
| `F/N_a` (0.125) | 432, 1945 | substring of the re-draw vector **`+0.125`** (initialisation-redraw magnitudes, §2.6). False positive. |
| CLU RESCUED | 716, 1592, 2270, 2295 | all four read **NOT RESCUED**; the regex's negative lookahead sits on the wrong side of the phrase. Correct form everywhere. |
| 54.56× / 5456 B | 22 sites | **all labelled.** 19 carry `modal` / `8 of 9` / `45.60×` / `seed 8` within ±6 lines. Three do not and are a different object: §2.5's validation-table row labels *`aggregate (54.56×)`* / *`recency (54.56×)`* (a **3-seed** run, and A.2's provenance row states *"seeds 0–2, i.e. inside the modal set"*), and l.1566's **`5456 B` iso-state budget**, which is a registered constant, not the seed-dependent ledger. |
| TTT byte figure | 1444 | the line is A.1's ⚠ ledger-caveat row and says **"per-seed"** in its first clause, then gives both values. Compliant with SF-2/R4. |
| `d/s` without ruler | 541 | Appendix D's dichotomy-verification table, row (H1): a **designed parametric sweep** of `d/s ∈ {2.86, 2.29, 1.71}` on a constructed store, not an estimator reading of the shipped store. Every *measurement* statement names its ruler (l.968, 993–995, 1002, 1020–1022, 1531, 2285) and l.1324 states the convention explicitly. |
| `3 of 5` | 793, 1885, 1930–1931 | all four are the **mandated labelled before/after** ("3 of 5 → **4 of 5** at nine seeds") in §4.2 and App I.1a/I.1b. |
| reads *below* blank | 2318 | App L 2, about the **struck `recency` family** (0.4769 written vs 0.5463 blank) — a protocol-validation result, not the CLU n = 3 drama. |
| −0.3180 / −0.5261 | 1473, 2089–2090 | A.1e's numpy fidelity check ("seeds 0–2 reproduce the banked three-seed values digit-for-digit") and I.1c's labelled supersession block — both **required** by A18.1 beside a moved number. |
| `0 of 15` | 1777, 1953, 2358, 2379 | l.1777 is scoped *"in this sub-table"*; l.1953 is a **different object** (prereg P1: cells picking a new `lr`); l.2358 now points at the 0-of-20 column of record (my edit); l.2379 **is** the 0-of-20 entry. |
| SemEval | 2725, 2799 | the two ⛔ **never-copy warnings** I added (Q.1's Poliak note + editorial item 6). Intentional. |
| 30–80× w/o subject | 2808 | editorial item 6 (deleted before circulation), which names Chesetti & Pandey **and** the arXiv v2 §6.6 pin. §5.3, the printed site, carries the full subject caveat. |
| Chesetti w/o pin | 2681, 2683, 2818 | the BibTeX title/author lines (the pin is in the same entry's `note`) and editorial 6b, which says *"quotes pinned to arXiv v2"* in-sentence. |

**Charter compliance spot-check (unchanged from r4 except where this pass touched it).** **C-1 as REVERSED 2026-07-07** — no defensive audit-confession paragraph; **J&P 2026 is cited exactly once, for the primitive's introduction only**, with the mandatory continuity sentence *"the CLU, introduced as CHLU in Jawahar & Pierini (2026)"* (§1, l.65); no legacy mechanism-number is asserted as evidence. **C-2** verification/evidence labels untouched (§1.3, §3, §4 preambles). **C-3** ML-first; contributions on page 1; Figure 1 named as the headline. **C-5** scale qualifiers untouched. **C-6** the new fine print (RadixSpline/RMI subject, sparsified baseline, arXiv-version pin, GDN's hedge, the Mamba-2 body-text residual) sits **next to the claim**, not in a footnote — that is the whole shape of this revision. **M1 hermetic** — no unpublished program short is cited; Chesetti & Pandey, Marcus et al. and Yang et al. are published/citable; the F5 note and J&P 2026 stay third person. **C-9/C-10** nothing pruned; all new material is an appendix (Q) or an in-place precision expansion.

---

# 3. How I verified

| check | method | observed |
|---|---|---|
| ⛔ **no measured number moved** | scripted set-difference of every decimal with ≥ 3 fractional digits between `draft-r4.md` and `draft-r5.md` | **numbers only in r4: `[]` (empty).** Numbers only in r5: **15, every one a citation identifier** (`10.1137`, `10.14778`, `10.5555`, `1802.00884`, `1803.01474`, `1805.01042`, `1901.00902`, `1907.04389`, `1911.13014`, `2006.12804`, `2407.00590`, `2412.06464`, `3326943.3326986`, `3421424.3421425`, `1.9781611978759`). ⇒ the pass is provably citation-layer. |
| all 8 markers gone | `grep -c "CITE2"` | **0** occurrences of the `⟦…⟧` marker form; one plain-text mention inside editorial item 6 that *records* the discharge (deliberately written without the glyph so a future sweep cannot trip on it). |
| every edit unique/intentional | 9 `Edit` calls, each on an exact unique string; no `replace_all` | 9/9 applied, 0 ambiguous. |
| the mis-attribution is gone | sweep pattern `SOSD verdict\|(Kipf\|SOSD)…(30–80\|4 orders)` | **clean.** |
| upstream propagation hunt | `grep -rl "30–80×\|30-80×"` over `.claude/**` + `docs/**` | one non-draft, non-task hit: **`bprime-fb1-recon.md` l.49** (reconciliation 1). `bprime-rivals.md` and `philosophy-synthesis.md` mention the line with **no numbers**. |
| never-quote / CM-30 sweep | 48-pattern script, whole-document matching | printed above; **0 real violations**. |
| line count | `wc -l` | 2621 → **2825**. |

**Flag provenance (protocol §5).** ⛔ **This pass produced no quantitative result and re-derived none.** Every number in `draft-r5.md` is inherited unchanged from `draft-r4.md` and keeps its existing provenance tables (**Appendix A.1, A.1b, A.1c, A.1d, A.1e, A.1f, A.2, A.3, A.4**); A.1f's citation-provenance row is the only provenance cell edited, and the edit is a verification status, not a measurement. No harness was run, no environment was resolved, no commit was made (papers live under gitignored `.claude/papers/`).

**Git footprint:** none. No tracked file touched, no branch created — per the task file (papers are gitignored).

---

# 4. Residual single-sourced / unverified items, named (task's closing request)

Now printed in the draft itself as **editorial item 6b**, so the next scout batch is pre-scoped:
- **Never verified against published text, cited qualitatively, no number quoted from any:** Wang/Shi/Fox (2501.12352), ATLAS (2505.23735), Miras (2504.13173), HOLA (2607.02303), Based, MAD, Zoology, RULER, Sparse Delta Memory, Erase-then-Delta (2606.26560), the LLM-agent-memory trivial control (2607.21962), kNN-LM, Xu/Alon/Neubig, MassiveDS, Sun et al. (TTT), and DeltaNet/GDN/GDN-2 beyond the loci verified above.
- **Single-sourced but printed:** Mitzenmacher's **pp. 462–471** (DBLP + S2, correlated provenance; NeurIPS's own page prints no pagination — safe, NeurIPS entries are routinely cited without pages).
- **Verified identity, body text unread:** **Marcus et al., PVLDB 14(1):1–13** (⛔ the draft quotes **no** number from it — enforced by a sweep pattern), and **Mamba-2's paper body** (the state convention rests on the reference implementation + the authors' companion post; disclosed in A.1f).
- **Quotes pinned to a preprint because the camera-ready is paywalled:** **Chesetti & Pandey, arXiv:2407.00590v2 §6.6** (= v3 §5.2). Stated in-text.
- **Not printed at all:** the **Belinkov et al. (2019)** corroboration entry (unverified venue/pages, and uncited) — recorded in Q.2 with the reason.
- **Blocked, 6 consecutive waves:** MUNKEY's workshop identity (OpenReview bot wall). ⚠ Unchanged by this pass, and **MUNKEY is still cited 0× in the draft**, so it is not exposure.

---

# 5. Open editorial questions for the Hub / Head

1. ⚠ **Ratify (or revert) the §5.2 blockquote edit** — reconciliation 2. It is the one negotiated sentence I changed, and I changed it because it carried the same defect §5.3 did.
2. **Does the §1 Mitzenmacher block-quote earn its length?** I quoted the published sentence verbatim (2.5 lines in the intro) because it is the *only* verified source for the "matched space as a condition on the verdict" claim, and a paraphrase re-opens exactly the attribution question this fold closed. A shorter paraphrase + the quote moved to §5.3 is the alternative; I did not take it unilaterally.
3. **Appendix Q's status.** I scoped it as "verified entries", not "references", because assembling a real bibliography means printing ~25 entries this program has never verified — which would violate the same standard this pass just enforced. Confirm that's the intended shape, and whether the reference-list task is scoped now or at the LaTeX pass.
4. **§5.3's rhetorical position has shifted slightly and the Head may want to see it.** r4 borrowed a (mis-attributed) anti-learned-index verdict as a rhetorical prop. r5 cannot: SOSD's real verdict is *pro*-learned-index. The bullet now imports the **accounting discipline** while conceding the field's verdict runs the other way. I think that is strictly stronger — the audit is not claiming learned structures lose, it is claiming they are *priced* — but it is a change of tone in a positioning section.
5. **Carried forward, unchanged from r4 and still open:** Fig 1's Mamba-2 re-render (does it gate circulation?); whether `{GDN, Mamba-2}` is the right restricted form for §4.2; the title workshop (*bounded-state memories* vs *test-time dynamics*, now that the arm set includes an SSM); whether the CLU column belongs in the harness rather than a scratch aggregation; and the App L / App J pruning rule.

---

## Proposed handover updates (for the Hub)

1. **`draft-r5.md` exists** (2825 lines), CHANGELOG appended. **r4 is superseded for citations only** — ⛔ **no measured number differs between r4 and r5** (mechanically verified, §3), so any number quoted from r4 remains correct.
2. ⛔ **A live error was corrected in the draft and is STILL LIVE UPSTREAM:** `bprime-fb1-recon.md` l.49 attributes *"30–80× larger than B-trees"* to **SOSD** and names **PGM** as the subject. Both are wrong (it is **Chesetti & Pandey**, and the subject is **RadixSpline/RMI**; PGM is 4× *smaller*). **Needs an owner** — this is the survey of record.
3. **Both `CITE2` items are discharged; the markers are gone.** Poliak = \*SEM 2018, S18-2023, pp. 180–191, DOI 10.18653/v1/S18-2023, verb **"significantly outperform"**, count **six of ten**. Mamba-2 = **Dao & Gu, ICML 2024, PMLR 235:10041–10071** (⚠ Mamba-1 = Gu & Dao). GDN's Mamba-2 equation is now **quoted** with its *"up to specific parameterization"* hedge.
4. **Sweep result for the wave record: claims-matrix v2.11 §0.9 / CM-30 + A18.1 + r4 patterns — 48 patterns, 0 violations** on a 2825-line draft. Both items the Hub named are structurally clean: the draft contains **no tier-ii object at all** (no "compositional family", no occupancy read protocol, no 0.272, no `null*`/0.00117, no cat test), because B′ is the tier-i audit paper and the tier-ii wave never entered it. **That is worth recording as a standing fact so the sweep is not re-scoped onto B′ every wave** — the exposure surface for §0.9 in this draft is the tier-i clauses only (aaaa/bbbb/cccc/dddd + the modal-value and selection-stability rules), all of which r4 already satisfied and r5 preserves.
5. **New, pre-scoped work item:** the paper has **no reference list**. Appendix Q holds the seven verified entries; editorial item 6b names the ~25 that are unverified. A scout batch + a LaTeX-pass assembly task closes it.
