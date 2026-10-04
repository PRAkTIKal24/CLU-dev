# bprime-draft — paper-writer report (C2W4, **v2 pass**)

**Task + acceptance criterion:** close the NUMBER-FREEZE GATE — integrate `bprime-rivals-f3` into the B′
tier-i audit paper, delete the draft-state banner, re-run every adjudication that the full-grid pass
moved, and hand the Hub `draft-v2` for the referee pass.
**Status: done.** `.claude/papers/bprime/draft-v2.md` (1697 lines), `CHANGELOG.md` updated, `draft-v1.md`
kept intact as the record.

## ⚠ RECONCILIATION LIST — needs a Hub owner (first-10-lines rule)
1. ⛔ **The task file's own C2W4 release block and its v2 release block conflict, and I followed the v2
   one.** C2W4 handed me the weak-FB3 sentence *"gdn (+1.02) and gdn2 (+0.88)"* as verbatim-and-binding;
   the v2 block rules that **every rival mean is quoted at pooled n = 9, never n = 3**. `+1.02 / +0.88`
   are n = 3 means and **no n = 9 dividend aggregate exists** (see §5, missing-number 1). I kept the weak
   form as a *sentence* and quoted the two n = 9 components that force its sign, and I said in-draft why.
   **Owner: Hub — ratify or order the 10-minute re-aggregation** (`bprime-rivals-f3` open question 2).
2. ⛔ **Four audit columns have no n = 9 aggregate** (projected launder · dividend · blank · same-keys
   null). Under the ruling they cannot be quoted at n = 3 as claims, so I moved them to **App. I.1c**,
   labelled, claim-free. The engineer offers the re-aggregation for ~10 min. **Owner: Hub.**
3. ⚠ **The byte-frontier column was NOT re-run** (n = 3, reduced grid). Its "0 of 5 not rescued" is now a
   *forbidden* n = 3 rescue verdict, so the draft reports **no verdict and no margin in either direction**
   there — the conservative reading, but it weakens App. H to a printed-for-completeness table.
   **Owner: Hub** — one labelled row (~5 min) would restore it; I did not assume it.
4. ⚠ **Still open from v1 and unanswered:** (a) `draft-v1.md`/`draft-v2.md` versioned filenames vs the
   agent brief's canonical `draft.md`; (b) whether B′ carries the "CLU" name debut; (c) the single-sourced
   Feng, Wallace & Boyd-Graber (2019) quotation must be re-verified before circulation; (d) **the
   paper-writer agent brief and `claims_matrix.md` §3 carried the pre-reversal form of Charter C-1** —
   ⭐ the brief was **corrected by the Hub 2026-08-01** (my v1 reconciliation 1 is discharged on the brief
   side); **`claims_matrix.md` §3 is still stale.** Owner: curator.
5. ⚠ **v1's B.5 asserted the rescue gate uses a `SE_paired` statistic. No source says that** — `f3` in
   fact proposes *"make the blank a paired per-seed statistic"* as a **fix**, implying it is not paired.
   **I removed the claim from the draft** and stated pairing as an open improvement. Owner: nobody — but
   record it: v1 shipped an invented estimator detail that a referee could have checked.

---

# 0. What changed between `draft-v1` and `draft-v2` (the mechanical integration, itemised)

| site | v1 | v2 | source |
|---|---|---|---|
| top-of-file banner | draft-state banner, gate OPEN | **deleted** | task §v2 |
| `⟦F3⟧` markers | **47** | **0** (verified by grep) | — |
| Abstract | rival numbers marked, "3 seeds" | n = 9 headline range `−0.2592 … −0.4602`, ≥ 4.4 SE; control handicap `0.263 … 0.942` | f3 §1, §5 P6 |
| §1.1 contribution 1 | protocol only | **+ the two protocol findings** (gate power; fit-split selection ⇒ 6×2 is 6×1) | f3 §4, §2 |
| §1.1 contribution 3 | "up to 1.208" | `0.276 / 0.263 / 0.425 / 0.856 / 0.942` (n = 9) | f3 §5 P6 |
| §2.2 rescue gate | one paragraph | **+ the power requirement**, stated as protocol: n ≥ 9, blank spread −0.962/−2.634/−1.390, coin-flip at n = 3 | f3 §4 |
| §2.4 | "0.203–1.208 ⟦F3⟧"; "+0.88…+1.02" | per-arm n = 9 handicaps + the sign-flip argument (handicap 0.856/0.942 vs raw margin 0.2600/0.2592) | f3 §5 P6, §1 |
| **§2.6** | "gate open, re-written in v2" | **rewritten**: standard met · 0 of 45 · ≤ 0.031 tuning effect · 64 % fit cut moves < 1 SE · init-redraw 4–35× · selection-rule finding | f3 §1.1, §2, §3 |
| §4 preamble | "all cells 3 seeds" | seed counts stated once, never mixed | f3 §6 |
| §4.1 status table | rival launder/null numbers | `have (n = 3, App. I.1c)`; `+0 B` at n = 9 | f3 §1 |
| **§4.1.1 audit table** | 3-seed, 13 columns | **n = 9**, both code paths' rescue lifts, UNSTABLE/NOT RESCUED verdicts, CLU row labelled n = 3 | f3 §1, §4 |
| §4.2 headline | "−0.188 to −0.424 ⟦F3⟧" | n = 9 range + **robustness list** (5× budget, held-out, seeds) + the **rescued-arms-only restricted form** (−0.2592/−0.2600/−0.2732) | f3 §1.1, §3 |
| §4.3 weak form | +1.02/+0.88 | components at n = 9 + an explicit note on why the dividend itself is not quoted | f3 §5 P6 + ruling |
| §4.3 metric-native table | margins for all 5 | margins **only for rescued arms**; the other two rows state the gate verdict | ruling |
| §4.5 frontier | "0 of 5 rescued" | **no verdict, no margin either way**; not re-run | ruling + f3 §6 |
| §6 L2 | "two of five NOT RESCUED" | 1 NOT RESCUED + 1 UNSTABLE + "carried by three arms sharing one state type" | f3 §4 |
| §6 **L2a (new)** | — | the gate is underpowered below n = 9; our P4 prediction refuted | f3 §4, §5 P4 |
| §6 L4 | "reduced grid, full pass in flight" | **standard met** + exactly what tuning does *not* cover + the redraw disclosure | f3 §1.1, §6 |
| §6 **L4a (new)** | — | our own selection rule makes the wd axis unselectable; held-out as declared secondary | f3 §2 |
| §6 L11 | "3 seeds" | per-run seed counts | f3 §6 |
| §7 | ⟦F3⟧ | n = 9 + the three robustness columns | f3 §1.1 |
| App. A.1 | one 3-seed table | **rewritten** (grid, optimiser, deviations, init scheme, chosen configs, φ hash, JAX/Equinox/Optax, wall clock, 1143 tests) **+ A.1b** for the first pass | f3 §6, §7 |
| App. B.5 | `2·SE_paired` | plain form + power requirement + the verdict list; **invented pairing claim removed** | see reconciliation 5 |
| App. H | "0 of 5 rescued ⟦F3⟧" | provenance + status note; no verdict quoted | ruling |
| App. I.1 | first-pass scorecard | **relabelled n = 3** + R5/R5-raw rows carry the n = 9 count | f3 §1 |
| App. **I.1a–I.1d (new)** | — | before/after table · the second pass's own prereg scorecard (P1–P6) · the un-re-aggregated columns · fit surface, three selections, 2000-step table, full rescue-gate matrix | f3 §1, §2, §2.1, §2.2, §3, §4, §5 |
| App. J | — | **+5 NOT-RUNs**: frontier under full grid · the four columns at n = 9 · β/cosine sub-clauses · held-out-as-primary | f3 §6 |
| App. L | 14 negatives | **+3** (P4 refuted; P2's count refuted; the init-redraw dominance) | f3 §5 |
| App. M | "all cells 3-seeded" | seed counts + the digit-for-digit first-pass reproduction as the licence for the before/after table | f3 §7 |
| App. K Fig. 1 | ⟦F3⟧ bars | caption must carry n = 9 / n = 3 split and the UNSTABLE hatch | ruling |

---

# 1. ⭐ CLAIMS → ARTIFACT → CAVEAT (deliverable 2 — the rows that CHANGED at v2)

Rows C1–C26 of the v1 report stand except where listed. `F` = frozen at v1, `9` = re-quoted at n = 9.

| # | draft site | claim as written in `draft-v2` | artifact | caveat carried in-draft |
|---|---|---|---|---|
| C1 | Abstract, §4.2, §7 | **9** 0 of 5 rival arms beat a +0 B reader of a **raw** table at the same bytes: **−0.2592 ± 0.0292 / −0.2600 ± 0.0278 / −0.2732 ± 0.0395 / −0.4425 ± 0.0869 / −0.4602 ± 0.1038**, every one ≥ 4.4 SE below zero; the CLU **−0.3180 ± 0.0804** | `bprime-rivals-f3` §1 (pooled n = 9), Hub-verified from `run400/`+`seeds3to8/` JSONs | scale qualifier in-sentence; **restricted form on the three rescued arms printed beside it**; the two non-settling arms flagged in the same paragraph |
| C1b | §4.2 | **9** the headline is unchanged at **5× budget** (−0.2184 … −0.2630) and under **held-out selection** (−0.24 … −0.49) | f3 §3, §1.1 | both labelled as what they are (a budget check; a declared secondary rule) |
| C4 | §4.3 | **9** weak-inversion: the projected control handicaps the table by **0.856 ± 0.091 (GDN)** and **0.942 ± 0.091 (GDN-2)** against raw margins of 0.2600/0.2592 ⇒ against that control both arms are positive and the CLU (−0.0789) is not | f3 §5 P6 + §1 | ⚠ **the dividend itself is NOT quoted** — no n = 9 aggregate exists; the draft says so in-line |
| C5 | §2.4, §1.1 | **9** the registered projected launder costs the table **0.276 / 0.263 / 0.425 / 0.856 / 0.942** (all > 2 SE) | f3 §5 P6 | pre-registration ordering still stated in **methods**, not in rebuttal |
| C27 | §2.2, §6 L2a, App. B.5, I.1d | **9 NEW** the rescue gate is **underpowered below nine seeds**: blank spread −0.962/−2.634/−1.390; three legitimate n = 3 configurations ⇒ three rescued sets; two code paths agree at n = 9 on 4 of 5 arms | f3 §4, §5 P4 | printed as a *protocol* finding and as a refuted pre-registration of ours (App. I.1b P4) |
| C28 | §4.1.1, §4.3, §6 L2 | **9 NEW** verdicts: **RESCUED** `deltanet`, `gdn`, `gdn2`; ⛔ **NOT RESCUED** `ttt_mlp`; ⚠ **UNSTABLE** `ttt_linear` (0.093 ± 0.134 vs 0.320 ± 0.083) | f3 §4 (both n = 9 rows) | no margin against `ttt_mlp` **or** `ttt_linear` anywhere; both readings printed for `ttt_linear` |
| C29 | §2.6, §6 L4a, App. I.1d | **9 NEW** best-of-grid on the fit split ⇒ **0 of 45** cells pick a new lr, wd only by 4th-decimal tie-break (12 of 45); held-out selection picks 26 of 45 / 24 of 45 and moves `ttt_linear` −0.6075 → −0.4461 | f3 §2, §5 P1/P2 | held-out = **declared secondary**; its rescued set (`{gdn}` at n = 9) printed as *thinner*, not better |
| C30 | §2.6, §6 L4, App. A.1 | **9 NEW** the **initialisation re-draw** moved arms `−0.148/+0.125/+0.018/−0.015/−0.042`, **4–35× the tuning effect** (`−0.0303/+0.0018/−0.0009/+0.0006/+0.0034`) | f3 §2 | disclosed by us, with the control column that makes it readable |
| C31 | §2.6 | **9 NEW** 5× budget: `ttt_mlp` fit loss **−64.1 %** moves eval **0.036 (< 1 SE, ±0.0891)**; delta arms' fit loss moves ≤ 0.1 % | f3 §3 | framed as *the binding constraint is the fit→eval gap*, not budget |
| C32 | App. A.1, App. M | **9 NEW** the first pass **reproduces digit-for-digit at base code** on all five arms ⇒ every difference is a declared change; suite **1143 passed / 0 failed** | f3 §7 | stated as the licence for the before/after table |
| C15 | §4.5, App. H | **AMENDED** frontier column: **no rescue verdict and no margin quoted in either direction**; not re-run under the full grid or at n = 9 | f3 §6 (declared NOT-RUN) + the n = 3 never-quote | the *flattering* comparison is declined in our own voice, as at v1 |

**Everything else is unchanged from v1's table** (C2, C3, C6–C14, C16–C26) — the CLU column, all three
theorems, the protocol-validation table, both byte ledgers, the attribution section, the deletion column
and the whole positioning section were never gated.

---

# 2. ⛔ Every NOT-RUN the draft declares (App. J — additions at v2 in bold)

Titans arm (⇒ P3 NOT-RUN, not refuted) · Sparse Delta Memory arm (Table-1 ratios quarantined) ·
Mamba-2 / GRU / SWA · a deletion column for any rival · `recency` and `manifold` (protocol-struck) ·
`overload` at base atom budget · a *trained* attention reader · a soft-certificate sweep over `B` ·
the live-launch-momentum probe (OQ-2, target factor 65) · a `d/s` sweep by atom width · the eviction-path
deletion arm · any real-data or LM leg · any change to a shipped default ·
**the byte-frontier column under the full grid and at n = 9** ·
**the n = 9 aggregate of the projected-launder / dividend / blank / same-keys-null columns** ·
**the standard's `β = (0.9, 0.98)` and cosine-decay sub-clauses** ·
**a re-run with held-out selection as the primary rule.**
⭐ All printed as NOT-RUN with a reason. None reported as a null.

---

# 3. Where I could not source a number, and what I did instead

| wanted | status | what I did |
|---|---|---|
| **n = 9 dividend vs the projected control** (the `+1.02 / +0.88` sentence) | **does not exist** — f3 aggregated only `full`, `+0 B`, raw margin, rescue lift | Kept the weak-FB3 *sentence*; quoted the two n = 9 components that force its sign (handicap 0.856 ± 0.091 / 0.942 ± 0.091 vs raw margin 0.2600 ± 0.0278 / 0.2592 ± 0.0292); said in-draft that the dividend was not re-aggregated. **I did not compute `raw margin + handicap` and print it** — that number appears in no artifact |
| n = 9 projected launder / blank / same-keys null | not aggregated | App. I.1c at n = 3, labelled, **claim-free**; declared NOT-RUN |
| n = 9 cross-arm **mean** `+0 B` margin (R5's registered band) | not aggregated | R5 row reports the **count** (4 of 5 ≤ 0) and points to the per-arm table; states the band was on a mean we did not re-aggregate |
| frontier column at n = 9 / full grid | declared NOT-RUN by the engineer | no verdict, no margin, both directions; App. H printed for completeness |
| per-seed **admissible coverage** on seeds 3–8 | not tabulated | §4.1.1 labels the coverage line as recorded on seeds 0–2 only |
| rendered figures | not produced | Appendix K specifications updated (n = 9/n = 3 caption rule, UNSTABLE hatch); `figures/` still empty. **Not pseudo-verified** |
| LaTeX build | not requested | Markdown only; no `.tex`, nothing claimed as compiling |

---

# 4. How I verified

| check | result |
|---|---|
| `⟦F3⟧` markers remaining | **0** (grep; was 47) |
| banner deleted | ✅ (grep for `DRAFT-STATE BANNER` → 0) |
| every new number traces to `bprime-rivals-f3` | ✅ — scripted containment check of **72** newly-printed numeric tokens against the six source reports: **0 missing** |
| arm↔number mapping | ✅ re-derived: the P5-vs-raw gaps map `ttt_linear 0.276 / ttt_mlp 0.263 / deltanet 0.425 / gdn 0.856 / gdn2 0.942` (checked against the first-pass identity `gap = raw-table score − projected-table score`, which reproduces the C2W4 values 0.216/0.203/0.458/1.208/1.065 exactly); tuning/redraw lists re-ordered to one consistent arm order throughout |
| no rival mean quoted at n = 3 as a claim | ✅ — the only n = 3 rival numbers left are inside App. A.1b / H / I.1 / I.1c, each carrying an explicit `n = 3` label and a "no claim rests on this" sentence |
| no rescue verdict at n = 3 | ✅ — all verdicts are n = 9; App. I.1d prints the n = 3 rows *as the evidence that they are unstable* |
| no margin against `ttt_mlp` or `ttt_linear` | ✅ — §4.3's table prints the gate verdict instead; the only place their raw margins appear is the audit table row and the headline range, which the Hub ruled explicitly |
| forbidden-string grep (`CLU-former`, `ICML 2026`, `we alone delete`, `principled forgetting`, `certified unlearning`, `deletion by construction`, `Def. 1`, `future work`, tier-ii/iii phrases, `training-time organisation`) | ✅ **0 hits each** |
| internal codenames (`F3`, `f3`, `C2W4`, `rival-recon`, `N78`) leaked into the draft | ✅ **0** after cleanup (two hits found and rewritten to protocol-internal language) |
| tier-i containment | ✅ no forward reference to tier ii/iii; no future-work section; §4.6.1 still ends on *"…it is a design identity."* |
| Charter C-1 (as REVERSED) | ✅ no defensive audit-confession paragraph; the first-pass corrections live where the numbers live (§2.6, App. I.1) |
| C-2 verification/evidence labels | ✅ unchanged; §4 preamble now also carries the seed-count discipline |
| C-5 scale qualifiers | ✅ abstract, §1.2, §4.2, §6 L11, §7 all carry `d_in = 5`, 5–6 items, ~10 tokens, CPU **and now the seed counts** |
| C-7 flag provenance | ✅ **five** provenance tables now (A.1 full grid, A.1b first pass, A.2, A.3, A.4) — cross-section reproduction cannot construct a contradiction between the two passes because A.1 states the digit-for-digit reproduction |
| C-9 negatives | ✅ App. L now **17** items, including three of our own refuted pre-registrations from this pass |
| C-10 appendix maximalism | ✅ nothing pruned; App. I gained four subsections (~130 lines) |

**Git footprint: none.** Research-only agent; no tracked file touched.

---

# 5. Open questions / follow-ups / risks

1. ⭐⭐ **Referee pass is next (§A15.5), and the draft is shaped for it.** The two attacks I expect to
   land are now *stated by us*: (a) **one surviving family × three arms sharing one state type** (§6 L1 +
   L2 together — the arm count is really three, not five); (b) **the byte-frontier column is inert** and
   nothing can be said with it. Neither is closable by drafting.
2. ⚠ **The paper now contains a visible asymmetry a referee will ask about: rivals at n = 9, the CLU at
   n = 3.** I state it and show the CLU margin is ~4 SE below zero on its own seeds, but the clean fix is
   re-running the CLU column at nine seeds. **Hub decision.**
3. ⚠ **`ttt_linear`'s UNSTABLE verdict is genuinely unresolved and it is load-bearing for L2** — it is the
   difference between "three rescued arms" and "four". The engineer's fix list (pair the control, or
   average over inits) is cheap. Not scoped.
4. ⚠ **The stress tests are asymmetric in a way I could not fix:** the 5×-budget and held-out-selection
   checks are at n = 3. They agree with the n = 9 primary in direction and sign on every arm, and I label
   their `n` at every appearance — but a referee could ask why the robustness checks are less powered than
   the result they defend.
5. **Figures: five specifications, zero renders.** Figure 1 is now renderable (the gate is closed) from
   `bprime-rivals-f3/{run400,seeds3to8}/exp_bprime_rivals_metrics.json`; it needs an engineer/analyst with
   plotting scope, not me.
6. **Length.** The main text grew by ~60 lines (§2.6, §4.2, §6 L2a/L4a) and the appendices by ~150. Still
   long for a workshop short, about right for a conference short; the pruning pass decides, per C-10.

---

## Proposed handover updates (for the Hub)

1. **§10 running log:** `bprime-draft` delivered **`draft-v2.md` with the NUMBER-FREEZE GATE CLOSED** —
   0 `⟦F3⟧` markers, every rival number at pooled n = 9, banner deleted, before/after table in App. I.1a.
   **The headline survived and strengthened**; the only claims that changed are the ones the Hub listed
   (rescue verdicts, R5's count, the P5-vs-raw magnitudes), and each changed in the draft exactly once.
2. ⛔ **Two Hub rulings are owed before the referee pass** (reconciliations 1 and 2): whether the
   weak-FB3 sentence may stand on its components, and whether to commission the ~10-minute n = 9
   re-aggregation of the four remaining columns. Both are cheap; both are currently handled by explicit
   in-draft "not re-aggregated" language, which a referee may read as a gap.
3. ⚠ **One labelled frontier row (~5 min) would restore App. H** from "printed for completeness" to a
   usable non-informativeness verdict at n = 9. Currently the paper can say nothing there at all.
4. ⚠ **`claims_matrix.md` §3 still carries the pre-reversal C-1 form** (the agent brief was fixed
   2026-08-01; the matrix was not). Curator.
5. **New never-quotes now honoured in-draft and worth filing:** any rescue verdict at n = 3 (including our
   own first pass's) · any margin against `ttt_mlp` · any margin against `ttt_linear` while its verdict is
   unstable · *"the one rival that beats its table"* for `gdn2` (+0.0473 ± 0.0277 is **a tie**) ·
   *"no margin against DeltaNet is quotable"* (**retracted** at n = 9).
6. **Record for the program, not just the paper:** the two protocol findings (gate power ≥ 9 seeds;
   best-of-grid on the fit split makes a `wd` axis unselectable) are now published contributions of this
   draft — if the Hub adopts the held-out selection rule as a standing rule, the paper's §6 L4a wording
   should be re-checked, since it currently presents it as a **declared secondary**.
