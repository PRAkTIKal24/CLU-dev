# bprime-draft-r3 — paper-writer report (C2W5, the POST-REFEREE revision)

Task + acceptance criterion: revise `draft-r2.md` against `bprime-referee` (MF-1…MF-5, SF-1…SF-7, SF-9,
SF-10 + nice-to-haves), fold the two riders (uniform n = 9 columns; the labelled deltanet frontier row),
apply `bprime-cite-check`, run the A18.1 never-quote sweep → **`.claude/papers/bprime/draft-r3.md`** +
CHANGELOG line.
**Status: done.** `draft-r3.md` (2231 lines, from r2's 1697); `CHANGELOG.md` appended; `draft-r2.md` and
`draft-v1.md` left intact as the record. No repo code touched (research-only agent, protocol §3 — **git
footprint: none**).

## ⚠ RECONCILIATION LIST — needs a Hub owner (first-10-lines rule)
1. ⛔ **The n = 9 fold CHANGED A CLAIM'S MAGNITUDE, per the F3 pre-commitment pattern (flagged, not
   smoothed):** §4.3's weak-inversion dividends are now quoted directly at n = 9 —
   **deltanet +0.1515 ± 0.0600 · gdn +0.5960 ± 0.0933 · gdn2 +0.6824 ± 0.0756** — against the first pass's
   n = 3 **+0.2006 / +1.0197 / +0.8771**. Signs hold 5 of 5; **gdn's magnitude falls 42 %, gdn2's 22 %**;
   **deltanet's dividend is now positive beyond 2 SE**, which it was not adjudicated as before. Both
   readings are printed side by side (CM-28(ccc)). **The C2W4-era sentence "gdn (+1.02) and gdn2 (+0.88)"
   is superseded and should be retired from the registries** — owner: **curator**.
2. ⛔ **New rival-side finding with no home outside this draft:** at n = 9 **both TTT arms read below their
   own same-keys null** (−0.2063 ± 0.1016 / −0.1995 ± 0.0665) while all three delta arms read above it
   (+0.2174 / +0.5642 / +0.7438). Filed in §4.1.1 + App. L 18. Owner: **curator** (an N-entry) — it is
   arguably a stronger statement of malfunction than the blank-store gate and it is not in any registry.
3. ⚠ **MF-3's coverage clause creates a fact the program should look at, not just print:** N186's
   write-admissibility criterion says **`aggregate` is 0/3 write-admissible** — i.e. *the family carrying
   the entire dividend column is one our store never writes to its own 0.05 endpoint-loss bar*. I wrote it
   as a **candidate mechanism for the CLU's below-blank read** (§2.5, §4.1.1). If the Hub disagrees with
   that causal framing, it is one sentence to cut. Owner: **Hub**.
4. ⚠ **SF-9a: arXiv:2605.17590 was CUT** (counterfactual-state-alignment unlearning, §4.7). Grep over all
   of `.claude/**` returns **zero** occurrences outside the referee report: the citation exists in no
   program artifact and could not be verified here. Owner: **Hub** — either commission a scout check and
   re-add, or leave cut.
5. ⚠ **One citation remains single-sourced and is now the only one:** Poliak et al. (\*SEM 2018), incl.
   the "6 of 10 NLI datasets" figure quoted in §5.3. `bprime-cite-check` flagged it as out of its scope.
   Owner: **Hub** (a ~10-minute scout item; listed in the draft's own open-editorial list).

---

# 0. DIAL DECLARATION (echoed, protocol §7)
- **Dial:** none — instrument/manuscript (a drafting pass; no new claim, no new measurement).
- **Laundering control:** n/a for the writer; the draft's own controls are the matched-byte launder, the
  +0 B substitute set, the same-keys null and the blank-store gate, all reported for every arm including
  ours.
- **Falsifies:** n/a. **Does NOT falsify:** a referee item I close by drafting does not upgrade the
  paper's evidence; MF-1's fix strengthens the thesis but adds no measurement.

---

# 1. Referee disposition — every item, with where it landed

| item | disposition | where in `draft-r3.md` |
|---|---|---|
| **MF-1** CLU printed ✅ RESCUED while below its own blank | **CLOSED** — row now ⛔ **NOT RESCUED**, lift **−0.104** printed; the below-blank fact stated as a finding consistent with the thesis; §4.2's arm-count qualifier extended to us; **plus** the honest power caveat (it is a three-seed *sign* fact, not an n = 9 verdict) | §4.1.1 (row + 3 ¶), §4.2, §6 L2, B.5, App. J, App. L 1a, Abstract |
| **MF-2** task family + rival arms never specified | **CLOSED** — two new appendices, fully written, sourced from `memory_gym` + `bprime-rivals` §8 | **App. N** (families), **App. O** (rival arms + iso-state derivation); pointers in §2.1/§2.5 |
| **MF-3** CM-27(c) caveats absent | **CLOSED** — anchor-vs-family objection (N170) + coverage clause (N186 referent, Hub-supplied) in our own voice, both scoped | §2.5 (two bullets), §6 L1 |
| **MF-4** n = 3 frontier verdict in §2.2 | **CLOSED** — replaced by the n = 9 form the rider licenses (0 of 15 cells) | §2.2 |
| **MF-5** prereg unverifiable + "previously published" | **CLOSED** — supplementary-commitment sentence + **dated 5-row registration table**; register fixed at all sites | §2.4, App. I preamble, §3.1, App. C, App. L 4, §4.6 |
| **SF-1** direction rule undeclared | **CLOSED** — stated once in B.5 as a display rule; §2.2/§4.1.1/§4.3/§6 L2 reworded to "no *comparative* margin in favour of another arm over X". ⭐ Consequence: §4.3's metric-native table now **prints** the TTT arms' own losses (0.4602 ± 0.1038, 0.4425 ± 0.0869), removing the contradiction with the abstract's range | B.5, §2.2, §4.1.1, §4.3, §6 L2 |
| **SF-2** TTT per-seed ledger | **CLOSED** — `b → (d_head, state, rows, table)` mapping table; §4.1.1 row shows both values; R4's never-quote stated in-draft | B.2, §4.1.1, §4.4, I.1c(e) |
| **SF-3** −0.0789 bare | **CLOSED** — **−0.0789 ± 0.0620, n = 3, per-seed −0.1863 / +0.0284 / −0.0788** at all four uses | §4.1.1, §4.2, §4.3, I.1c |
| **SF-4a** unlabelled stress ranges | **CLOSED** — in-sentence "three seeds" + an explicit "less powered than the result they defend" clause | §4.2 |
| **SF-4b** optional stopping | **CLOSED** — one paragraph: declared before pooling, registered n = 3 gives the same signs, and the verdict change ran against the flattering direction (DeltaNet rescued); the one change that ran our way is named | §4.2 |
| **SF-5** "seven groups" listed five | **CLOSED** — reworded to *"seven of the fourteen candidates in our survey"* and all seven named (Based · Zoology · MAD · SDM · HOLA · kNN-LM · MassiveDS). ⚠ I dropped *"independent groups"*: Based and Zoology share authors, so the original phrasing was doubly wrong | §5.3 |
| **SF-6** rulers unnamed | **CLOSED** — "(atom-width ruler)" at both sites, with the fitted-ruler value beside the second | §4.6.1 |
| **SF-7 / CM-28(ccc)** App H | **CLOSED by rider 2** — App. H split into **H.1** (n = 9, labelled null) and **H.2** (first pass, history, NOT-RUN note kept) | §4.5, App. H |
| **SF-8** figures | **not the writer's** — spec updated only (Fig 1 caption must now carry the CLU's NOT-RESCUED hatch; Fig 5 caption re-stated at n = 9) | App. K |
| **SF-9** citations | **CLOSED** — (a) 2605.17590 **cut** (see reconciliation 4); (b) GDN-2 → **Hatamizadeh et al., 2026** (§1 + §5.1); (c) **Guo et al. (ICML 2020)** with "§3 Eq. (1) + the (ε, δ) relaxation following it" | §1, §5.1, §5.4, §4.7 |
| **SF-10** n = 9 re-aggregation | **CLOSED by rider 1** | §4 preamble, §4.1, §4.1.1, §4.3, App. I.1c, App. J, App. M |
| **N-1** glyph strip | **DEFERRED** per Hub ruling 9 (LaTeX pass) | — |
| **N-2** internal-process clause | **CLOSED** (cut) | §5.4 |
| **N-3** "the store under audit" | **CLOSED** — as **Appendix P** rather than main text, to keep main text at main results (C-10) | App. P |
| **N-4** App H CLU without ±0.0139 | **CLOSED** | App. H.1/H.2 |
| **N-5** length | no action (C-10 pruning pass owns it) | — |
| **N-6** "one implementation" | **CLOSED** — "of this implementation class" in §7; §4.6.1's CM-29(f) block untouched | §7 |

**Should-fixes skipped: none.** Nice-to-haves skipped: N-1 (Hub-deferred), N-5 (policy).

---

# 2. CLAIMS → ARTIFACT → CAVEAT — the r3 delta (extends the r2 table; rows C1–C32 stand unless listed)

| # | draft site | claim as written in `draft-r3` | artifact | caveat carried in-draft |
|---|---|---|---|---|
| **C33** | §4.1.1, §4.2, §6 L2, B.5, App. L 1a, Abstract | ⛔ **NEW/CHANGED** — the CLU is **NOT RESCUED**: full **−0.5261 ± 0.0863** vs blank **−0.4221**, lift **−0.104** | `memory-gym-v0.md` §(aggregate/base row) via `n9_full_columns.json → clu_banked`; the same values already in r2's own table | (i) three-seed **sign** fact, not an n = 9 verdict; (ii) the banked artifact carries **no paired per-seed SE** on the blank, so no SE on the lift is quoted; (iii) §2.2's power finding applied to our own column; (iv) the n = 9 CLU column declared NOT-RUN (App. J) |
| **C34** | §4.1.1, App. I.1c(a)/(c), App. L 18 | ⭐ **NEW** — `full − null` (paired, n = 9): **−0.2063 ± 0.1016 / −0.1995 ± 0.0665 / +0.2174 ± 0.0749 / +0.5642 ± 0.1032 / +0.7438 ± 0.1242**; both TTT arms below their own same-keys null | `pilot-placement-probe/n9_full_columns{.json,_table.md}` §A (+ §B for the second code path) | rival-side only, changes no CLU claim; second code path printed beside it |
| **C35** | §4.3, App. I.1c(a) | **CHANGED** — the projected-control dividend at n = 9: **−0.1840 ± 0.1069 / −0.1794 ± 0.0748 / +0.1515 ± 0.0600 / +0.5960 ± 0.0933 / +0.6824 ± 0.0756** | `n9_full_columns.json → columns.f3_n9.table.rivals[*].dividend_vs_own_table (+ _se)` | first-pass n = 3 values printed beside them (CM-28(ccc)); magnitudes fall 42 %/22 %; sign unchanged 5 of 5 |
| **C36** | §4.1.1, §4.4, B.2, I.1c(e), App. O | **NEW** — the TTT byte ledger is **per-seed**: `b = 16 → (29, 5220 B, 5104 B)`, `b = 1 → (36, 5328 B, 5184 B)`; ttt_mlp `16 → (12, 5376, 5376)`, `1 → (12, 4656, 4608)` | rider 1's ledger columns + per-seed `mini_batch_per_seed`; arithmetic reproduces the recorded ledger exactly | ⛔ never quote a single TTT byte figure as *the* n = 9 value; the invariant is the 5456 B budget and the state/table ratio ≤ 1.028 |
| **C37** | §4.1.1 | **CHANGED** — admissible-cell coverage now **all nine seeds**: 0.806 / 0.825 / 0.688 / 0.563 / 0.667 / 0.700 / 0.800 / 0.750 / 0.455; 5 of 8 items admitted (6 of 8 on seed 8) | `n9_full_columns.json → admissible_coverage` (9 records) | the added seeds' coverage spread is **wider** than the registered three — stated in-draft |
| **C38** | §4.5, App. H.1, §2.2, B.5, App. L 19 | **CHANGED** — byte-frontier column at **n = 9**: **0 of 15** (arm × head-width) cells rescued; best lift **+0.0694 ± 0.0491** (deltanet @ d4) | `pilot-placement-probe/n9_deltanet_frontier{.json,_table.md}` | byte-frontier label at every appearance + `S_excl = 0.6500`; deltanet's `aggregate` rescue does **not** transfer; `d ≤ 8` cells not table-lossless; `ttt_mlp`/`gdn` NOT-RUN there |
| **C39** | §2.5, §6 L1 | **NEW** — the anchor-vs-family objection, unresolved, in our voice; `overload` substitutable at 12.0× too (launder 1.000 vs store 0.333) but the rule cannot tell | N170 (engineer's unresolved objection), `bprime-fb4-gate` §A3.6 | stated as unresolved; L1 names the second family as the only real answer |
| **C40** | §2.5, §4.1.1, §6 L1 | **NEW** — coverage: `aggregate` **0/3** and `manifold` **0/3** write-admissible (endpoint loss 0.2463–0.3612 / 0.2494–0.3808 vs tolerance 0.05; escalation 300→900 moves ≤ 0.005); only `overload@load1x_shipped` 3/3 | N186 (`route3-stage1-plus-2x2` §1; reproduced by `traj-write-objective` D5) | scoped as a **write-quality** instrument, explicitly distinguished from §4.1.1's query-level coverage; framed as a candidate mechanism for C33, a fact about our store not about rivals |
| **C41** | App. I preamble, §2.4 | **NEW** — five dated pre-registration documents committed to supplementary material (2026-07-30 / 07-31 / 07-31 / 08-01 / undated-but-pre-run) | `PREREG-Bprime.md`, `bprime-fb4-gate/PREREG.md`, `bprime-rivals/PREREG.md`, `bprime-rivals-f3/PREREG.md`, `bprime-c6/PREREG.md` (headers read this pass) | the fifth carries no date in its header and is described exactly that way; ⛔ none is edited after the fact |
| **C42** | App. N | **NEW** — the four task families defined (construction rule, query law, metric, chance, `S(f)`, why struck) | `chlu/experiments/memory_gym.py` docstrings/configs/query builders (read-only), `bprime-fb4-gate` §A3, N170/N171/N172 | `overload` label as frontier-only; struck families' numbers never reported as dividends or nulls |
| **C43** | App. O | **NEW** — rival-arm specifications + the iso-state head-width derivation | `bprime-rivals` §8 (equations verified by the engineer from the primaries), §5.1's ledger formulas, rider 1's ledger columns | measured-vs-reasoned split preserved; NOT-RUN arms listed with reasons |
| **C44** | App. P | **NEW** — the store under audit in one place (potential, write, read, ledger) | §3.1 + A.2/A.3 flag tables (already in-draft), `memory-gym-v0` flags | no new claim; every constant already appeared in a provenance table |
| **C5′** | §5.1 | **AMENDED** — GDN-2 cited with authors; **EDA (Li et al., arXiv:2606.26560)** named as concurrent | `bprime-cite-check` §1–§2 (double-sourced) | "supersedes GDN" remains **our** characterisation, and the EDA clause makes it a trend rather than a pick |
| **C45** | §5.3, §6 L8, §2.4 | **AMENDED** — Feng et al. paraphrase kept, + "by the same logic" + the datasets-not-memories scope clause | `bprime-cite-check` §3 (ACL 2019, pp. 5533–5538, DOI 10.18653/v1/P19-1554) | ⛔ if a later revision quotes, it must use the ACL wording, not the arXiv one — recorded in the draft's open-editorial list |

**Unsourced claims: 0.** Every quantitative statement in `draft-r3.md` traces to a named artifact.

**Two derived (not improvised) quantities, declared:**
1. the CLU's **per-seed dividends** (−0.1863 / +0.0284 / −0.0788) are `full − launder` per seed from the
   banked per-seed columns; their mean and sample SE reproduce the recorded **−0.0789 ± 0.0620** exactly
   (0.10738/√3 = 0.062) — which is the check that licenses printing them;
2. **App. O's head-width table** (`d² + b·d`, `8d² + b·d`, `n_head·d_k·d_v` under the 1364-float budget)
   reproduces every recorded ledger byte count in rider 1 digit-for-digit (5220/5328/5376/4656/5184 and
   tables 5104/5184/5376/4608/5184). It is an accounting reconstruction of recorded numbers, not a new
   measurement, and is presented as such.

---

# 3. Every NOT-RUN the draft declares (App. J — r3 changes in bold)

Titans arm (⇒ P3 NOT-RUN, not refuted) · SDM arm (Table-1 ratios quarantined) · Mamba-2 / GRU / SWA · a
deletion column for any rival · `recency` and `manifold` (protocol-struck) · `overload` at base atom budget
· a *trained* attention reader · a soft-certificate sweep over `B` · the live-launch-momentum probe (target
factor 65) · a `d/s` sweep by atom width · the eviction-path deletion arm · any real-data or LM leg · any
change to a shipped default · the `β = (0.9, 0.98)` and cosine-decay sub-clauses · held-out selection as
primary · **`ttt_mlp` and `gdn` on the byte-frontier column** · **the byte-frontier column under the full
tuning grid** (H.1 is nine-seed but not re-tuned — stated, not implied) · ⭐ **the CLU column at nine
seeds** (new; the referee's missing-experiment 1, Head-owned).
**RETIRED as NOT-RUNs (now run):** the n = 9 aggregate of the projected-launder / dividend / blank /
same-keys-null columns; the frontier column at n = 9 for three of five arms.

---

# 4. How I verified

| check | result |
|---|---|
| never-quote sweep (A18.1 + claims_matrix §0 + CM-28), literal grep | `CLU-former` **0** · `3.6 SE` / `≥ 3.6` **0** · `ICML 2026` **0** · `we alone delete` **0** · `principled forgetting` **0** · `deletion by construction` **0** · `Def. 1`/`Def. 2` **0** · `(oral)` **0** · `MUNKEY` **0** (cited 0×) · `2605.17590` **0** · `0 of 5 not rescued` **0** · `no margin against DeltaNet` **0** · `the one rival that beats` **0** · `cluformer` **0** · tier-ii/iii vocabulary (`cat test`, `organizer swap`, `tier ii`, `tier iii`, `organization dividend`, `training-time organis/zation`, `emergen*`) **0** · `future work` **0** · pilot vocabulary (`enwik8`, `bpc`, `0.16 M`) **0** |
| the three "context-needed" hits, inspected individually | `verified to 1e-9` ×2 — both are the **erratum quoting the wrong sentence** (§3.1) and the never-quote note in A.1 ✅ · `∞` ×1 — the **`γ → ∞` static-watershed** statement in E.1, not a gradient ratio ✅ · `1089` ×1 — the **corrected-to-525×** sentence ✅ · `certified` ×7 / `exact deletion` ×9 — every one inside a **negation or a quarantine** (§4.7's "we do not describe this as…", E.2/E.3, §5.4's ε-certified removal citation) ✅ |
| n = 3 rescue verdicts (A18.1/CM-28(aaa)) | **0 rival verdicts at n = 3** anywhere; the only n = 3 gate statement is the **CLU's NOT RESCUED**, which is Hub-directed (addendum item 1) and is printed as a *sign* fact with §2.2's power caveat attached |
| CM-28(ccc) (no C2W4 rival number without the before/after) | ✅ — first-pass numbers survive only in A.1b, H.2, I.1, I.1a, I.1c(f); §4.3's new before/after pair is printed inline |
| CM-29 approved wordings | (a) headline unchanged at pooled n = 9 · (b) tuning paragraph unchanged · (c) launder-by-omission unchanged · (d) ledger asymmetry: **1.000–1.023 kept as printed**, with "≤ 1.028 in every seed's selected configuration" added beside it (per-seed truth, no approved number altered) · (f) §4.6.1 block untouched **minus its tier-ii clause**, still ending at "design identity" |
| CM-27(c) compliance | ✅ FB4 wording now carries **both** mandatory caveats (anchor objection + coverage) and L1's thinness |
| C-1 (post-reversal) | ✅ no defensive audit-confession paragraph; the errata and negatives live where the numbers live (§3.1, App. C/I/L) |
| C-2 verification/evidence | ✅ unchanged; App. N/O/P are specifications, not results, and are labelled as such |
| C-5 scale qualifiers | ✅ abstract, §1.2, §4.2, §6 L11, §7 all carry `d_in = 5` / 5–6 items / ~10 tokens / CPU **and** the seed counts; App. N adds the per-family scale |
| C-6 fine print adjacency | ✅ deletion conditions in-sentence (§4.7); the frontier label in the same block as every frontier number |
| C-7 flag provenance | ✅ **seven** provenance tables now (A.1, A.1b, **A.1c**, **A.1d**, A.2, A.3, A.4). Cross-section contradiction check: the TTT ledger (the one crack the referee found) is now reconciled by B.2's mapping, and §4.4's table declares which `b` it prints |
| C-8 hermetic | ✅ J&P 2026 in third person, once, with the continuity sentence; no unpublished sibling work referenced |
| C-9 negatives | ✅ App. L now **20** items (added 1a the CLU gate failure, 18 the TTT null-crossing, 19 the frontier null at n = 9, 20 the dividend-magnitude shrink) |
| C-10 appendix maximalism | ✅ nothing pruned — the replaced I.1c table is **retained as I.1c(f)**, the first-pass frontier table as **H.2**; three new appendices added |
| markdown structural check | 0 ragged tables (scripted column-count check over all 345 table rows); two pre-existing `|`-in-cell defects fixed (Titans `2·|M_θ|` cell, App. D `|dq|` cells) |
| figures | **not rendered** — specifications only; not pseudo-verified |
| LaTeX | **not built** — not requested; markdown only |

---

# 5. Where I could not source a number, and what I did instead

| wanted | status | what I did |
|---|---|---|
| SE on the CLU's rescue lift | **does not exist** (banked blank is a 3-seed mean without a paired per-seed spread) | printed the **point lift −0.104** and argued the verdict from its **sign**, saying in-draft why no SE is quoted |
| CLU column at n = 9 | **NOT-RUN** (Head/Hub decision, per the Hub addendum) | declared in App. J; labelled at every appearance; §4.2 and §4.1.1 say the CLU statement stands on its own three seeds |
| the frontier column's tuning grid provenance | not declared in the rider | described H.1 as "nine seeds on the current code path" and **explicitly did not** call it a full-grid column |
| `ttt_mlp` / `gdn` frontier rows at n = 9 | not run | NOT-RUN with a reason; no "all five arms" sentence written anywhere about that column |
| verification of arXiv:2605.17590 | impossible here (0 occurrences in `.claude/**`) | **cut the sentence**; reconciliation 4 |
| Poliak et al. \*SEM 2018 anthology ID / pages / "6 of 10" | registry-sourced only | left as-is (it was in r2), flagged in the draft's own open-editorial list and reconciliation 5 |
| rendered figures | not produced | App. K specs updated; `figures/` still empty |

---

# 6. Open questions / follow-ups / risks

1. ⭐ **MF-1's fix is the paper's best new paragraph and it costs nothing** — but it makes the CLU column's
   `n = 3` more conspicuous, not less. A reviewer will now ask "does your own arm fail at nine seeds too?"
   and the honest answer is *untested*. The n = 9 CLU column is the single highest-leverage remaining
   experiment; it is priced at the cost of one banked-column re-run.
2. ⚠ **The write-admissibility clause (C40) is a loaded gun pointed at our own store**, and I chose to fire
   it in our own voice with a causal hypothesis attached. It is the most consequential *new* framing in
   this revision. If the Hub wants it purely descriptive, delete one clause in §2.5 and one in §4.1.1.
3. ⚠ **§4.3's magnitude change (42 %/22 %) is the third time a B′ number has moved under more power.** The
   pattern is now stable and worth a program-level sentence somewhere: *dividends measured against a
   handicapped control shrink with power; margins against a raw table do not move at all.* That is a
   methodological claim the paper is entitled to but does not currently make.
4. ⚠ **Length.** Main text is unchanged in shape; appendices grew ~500 lines (N/O/P + I.1c). C-10 says do
   not prune now; the pruning pass will need a real cut list, and App. N/O are the parts a
   benchmark/analysis venue is *most* likely to want kept.
5. **Figure 1 is renderable and its caption spec changed** (the CLU bar is now hatched NOT RESCUED). If the
   engineer renders before the caption is re-read, the figure will contradict §4.1.1.

## Proposed handover updates (for the Hub)

1. **§10:** `bprime-draft-r3` delivered — **`draft-r3.md`**, all five referee MUST-FIXes closed, SF-1…SF-7
   + SF-9/SF-10 closed, both riders folded, cite-check applied, **never-quote sweep 0 violations**.
   ⭐ The headline is untouched; the CLU now **fails its own rescue gate in print**, which the referee
   scored as the single defect and which strengthens the thesis.
2. ⛔ **New never-quotes to file (registry, curator):** (i) the CLU's *"✅ rescued"* at n = 3 (Hub-ruled,
   now honoured in-draft); (ii) **any single TTT byte figure as *the* n = 9 value** (the ledger is
   per-seed); (iii) the C2W4 dividend magnitudes **"+1.02 / +0.88"** — superseded by
   **+0.5960 ± 0.0933 / +0.6824 ± 0.0756** at n = 9; (iv) *"no margin against TTT-MLP/TTT-Linear"* in its
   unqualified form — the suppressed quantity is a **comparative** margin, and the arms' own losses to
   their own tables are quotable (B.5's direction rule).
3. ⭐ **Two findings that belong in the registries, not only in the draft:** both TTT arms below their own
   same-keys null at n = 9 (reconciliation 2), and the frontier column's n = 9 null (0 of 15 cells).
4. ⚠ **Rulings owed / decisions outstanding:** the CLU n = 9 column (Head); the causal framing of the
   write-admissibility clause (Hub, reconciliation 3); the 2605.17590 re-verification (Hub,
   reconciliation 4); the Poliak scout micro-check (reconciliation 5); figure renders (engineer, SF-8).
5. **Deferred by ruling, still open:** the glyph/register strip (N-1) at the LaTeX pass; the "CLU" name
   debut (A18.7, post-C2W5 shorts re-pass).
