# bprime-draft-r4 — paper-writer report

**Task + acceptance criterion:** fold the three post-`r3` artifacts (the CLU n = 9 column · the Mamba-2 row
· the closure cracks + the Hub's selection-stability ruling) into **`.claude/papers/bprime/draft-r4.md`**,
with a CHANGELOG line carrying per-item disposition, every n = 9 claim change flagged, and the A18.1 sweep
printed. **Status: done.** No code touched, no branch, no commits (research-only spoke, protocol §3).

> ## ⚠ RECONCILIATION / OWNER LIST (first-10-lines rule)
> 1. ⛔ **FIGURE RE-RENDER REQUESTED — Fig 1 is one bar short.** The render (16:11) predates the Mamba-2 arm
>    (17:34): it carries **six bars** (5 rivals + CLU, uniform n = 9, CLU hatched NOT-RESCUED). The r4 spec
>    is **seven bars** with three hatch classes (NOT-RESCUED `ttt_mlp`+CLU · INIT-UNSTABLE `ttt_linear` ·
>    SELECTION-DEPENDENT `deltanet`,`gdn2`). Renderer:
>    `.claude/scratch/bprime-referee-closures/render_figures.py`; the mamba2 values live in
>    `.claude/outputs/bprime-mamba2-arm/run_agg_n9/`. **Fig 2** also wants the SSD bar; **Fig 5**'s caption
>    wants the 0-of-20 count. *(Owner: an engineer with the renderer — ~20 s of compute.)*
> 2. ⚠ **Two numbers the draft needed and does NOT have** (declared NOT-RUN in App J rather than
>    improvised): (a) the SSD arm's **paired `full − null`** (only unpaired means exist ⇒ I print no SE);
>    (b) its **per-reader `+0 B` means** and per-head-width frontier `+0 B` margins. Both are one
>    re-aggregation of banked cells, not new measurement. *(Owner: Hub — fund or leave declared.)*
> 3. ⚠ **⟦CITE2⟧ markers are live in the draft** at §1, §2.4, §5.1, §5.3, App A.1f and editorial item 6
>    (Poliak `*SEM 2018` + the Mamba-2 venue/id/state-accounting). They must be removed **only** by
>    `bprime-cite-check-2`. *(Owner: `bprime-cite-check-2`.)*
> 4. ⚠ **A cross-section contradiction I found and closed by labelling, not by editing a number:** App N.1
>    defines `aggregate`'s blank store as **−0.4221** (protocol-validation run, 3 seeds) while §4.1.1 now
>    reports the same control at **−0.3906 ± 0.0124** (audit cell, 9 seeds). Both are printed with their
>    runs named (App N.1 gained the reconciliation sentence). *(Owner: none — closed; flagged so the
>    curator does not "fix" one of them.)*

## DIAL DECLARATION (echoed, protocol §7)
- **Dial:** **none — manuscript/fold work.** No new measurement, no new claim; every number is transcribed
  from a named report + its flag-provenance table.
- **Laundering control:** inherited per number (the audit's own column set). The drafting-side analogue of
  a laundering control is the **A18.1 + never-quote sweep**, printed in §4 below.
- **Falsifies (mine):** a number in `draft-r4.md` that does not appear in
  `bprime-referee-closures` / `bprime-mamba2-arm` / `bprime-rivals-f3` / the r3 draft; or a rescue verdict
  quoted where it is not selection-stable.
- **Does NOT falsify:** the CLU still failing its own gate, or the audit still being 0-of-6 — those are the
  measurements, and the fold's job is to state them.

---

# 1. What I wrote (per fold item, with the evidence behind each)

**Source of record for every new number:** `.claude/outputs/bprime-referee-closures.md` §1 + §5 (CLU n = 9,
figures) and `.claude/outputs/bprime-mamba2-arm.md` §1, §1.2, §2, §3.1–§3.6, §4, §5 (the SSD arm). Hub
rulings from the `[C2W5]` second-review §10 entry (rulings 1, 3) via the task file.

### (1) The CLU column at n = 9 — the biggest claim change in the revision
| quantity | r3 (n = 3) | **r4 (n = 9)** | where it moved in the draft |
|---|---|---|---|
| rescue verdict | ⛔ NOT RESCUED, *"reads below its own blank"* | ⛔ **NOT RESCUED — "statistically indistinguishable from an empty store"**, lift **−0.0465 ± 0.0406** (\|t\| = 1.14), point estimate on the wrong side of zero | abstract, §2.2, §4.1.1 (row + 2 rewritten paragraphs), §4.2, §6 L2/L2a, §7, B.5, App L 1a |
| `full` | −0.5261 ± 0.0863 | **−0.4370 ± 0.0417** | §4.1.1, I.1c(a) |
| `+0 B` / raw margin | −0.3180 ± 0.0804 (≈4 SE) | **−0.2897 ± 0.0328 (8.84 SE)**, raw ≡ +0 B float-identical **9/9 seeds** | abstract, §4.1, §4.1.1, §4.2, §4.3, §7 |
| dividend | −0.0789 ± 0.0620 | **−0.0561 ± 0.0315 — 1.78 SE** | §4.1.1, §4.2, §4.3, I.1c(a), App L 1 |
| launder / blank / null / `full−null` | −0.4472 / −0.4221 / −0.8175 / — | **−0.3810 ± 0.0345 / −0.3906 ± 0.0124 / −0.6512 ± 0.0383 / +0.2141 ± 0.0443** | §4.1, §4.1.1, I.1c(a) |

- ⛔ **Every "the launder beats the store" is softened to "reads no worse than"** (task item 1). Grep for
  `beats` in the CLU context returns zero such sentences; §4.1.1 states the 1.78 SE explicitly and says
  *"no sentence in this paper says the launder beats it."*
- ⭐ The *"(its own table is already raw)"* parenthetical is **retired into a measured statement**
  (float-identical on 9/9 seeds; the arg-min launder never beats the `knn2` readers).
- The two-conventions agreement (−0.2897 vs −0.2862, Δ = 0.0035) and the write-vs-blank variance
  asymmetry (SE 0.0417 vs 0.0124) are carried as the closures report suggested.
- ⚠ **The one honesty item I added unprompted:** the CLU column is *bit-identical across both code paths*,
  so cross-path agreement is **not** independent evidence for our own verdict (closures §6 risk 3). Stated
  in §4.1.1 and A.1e.
- The **n-asymmetry is closed** in the main text; the frontier curve stays banked at n = 3 and is labelled
  at every appearance (§4.5, App H.1/H.2, K Fig 5, M, J).

### (2) The Mamba-2 (SSD) row
- Headline **0-of-5 → 0-of-6**; range **−0.2563 … −0.4602**; ⛔ **minimum SE multiple unchanged at 4.43**
  (still `ttt_linear`) — the A18.1 "4.4 SE" wording is untouched.
- Row added to §4.1 (protocol table), §4.1.1 (full row incl. the ledger), §4.3 (apples-to-apples + the
  metric-native table with the unnormalised-key argument), §4.4 (ledger, state/param 0.619), App O.1.
- **New App O.2b** (equations; three implementations asserted equal incl. the dual/SSD-duality read;
  sizing; the four declared deviations with the direction each cuts; the block-level ablation table).
- **New App H.1b**: frontier NOT RESCUED at **5/5 head widths** and, on the audit cell, in **3/3
  registered selections** ⇒ the column moves **0 of 15 → 0 of 20**; ⛔ no margin quotable there, including
  against the CLU's banked 0.972.
- **New App A.1f** flag-provenance table (seeds, grid, selections, chunk = 16 asserted inert, the six exact
  ledger integers, the bit-identical incumbent regression check, 1261-test suite, wall clocks).
- §5.1 gains a state-space paragraph incl. **the rival authors' own placement** of Mamba-2 as the
  erase-free delta rule ⇒ the row's reading: *the delta-erase term buys 0.003 ± 0.037 of `full`*.
- §5.2's SSM limitation **RETIRES** (with the "one member of the family; Mamba-1/3 are different state
  types" qualifier); §6 L5 moves it from reasoned to measured (reasoned = SDM, Titans).
- ⭐ **Both honest qualifiers carried verbatim:** best `full` is *a tie with gdn2, not a win* (§4.1.1
  prose + App L 24: no ordering among the four delta/SSD arms is quotable); the +0 B margin
  **+0.0047 ± 0.0519 is a tie with zero** and flips sign under held-out selection.
- §2.6 gains three tuning-defence measurements the arm bought: **0 of 45 → 1 of 54** (the widened grid is
  decorative *for the delta/TTT arms, not for the rig*), the **interior fit-optimum**, and the **block
  ablation** (−36 % fit loss, worse eval).

### (3) Selection stability (Hub ruling 1, extends A17.4)
Stated as protocol in **§2.2** and formally in **B.5**, applied at every verdict site:
✅ `gdn`, `mamba2` **RESCUED** (stable) · ⛔ `ttt_mlp` **NOT** · ⚠ `ttt_linear` **INIT-UNSTABLE**
(relabelled from "UNSTABLE" for precision) · ⚠ `deltanet`, `gdn2` **SELECTION-DEPENDENT**
(+0.0768 ± 0.0446, +0.6685 ± 0.3389 held-out — numbers already in I.1c(d), so nothing was improvised).
Downstream: §4.2's restricted form now rests on **{GDN, Mamba-2}** (−0.2600 ± 0.0278, −0.2563 ± 0.0416);
§4.1.1's "only the rescued arms enter §4.3's adjudications" → two arms; §6 L2 comparative half carried by
**two** arms; L2a, L4a, I.1a (two table cells + derived outcomes), I.1c(a), I.1d updated.
⛔ **No margin is quoted against a selection-dependent arm in the flattering direction** — verified by
sweep (§4 below): the only quantities quoted for those arms are *their own* losses to *their own* tables,
which B.5's direction rule explicitly leaves quotable.

### (4) Closure cracks
- **R2 (modal ledger):** `5456 B / 100 B / 54.56×` labelled **modal (8 of 9)** with seed 8's
  `5472 / 120 / 45.60×` beside it at **10 sites** — §4.1 table, §4.1.1 (new paragraph), §4.4 (table note +
  the identity paragraph), §6 L6, A.1, A.1e, A.2, I.1c(e), P.4, Fig 2 spec. Exactly the TTT R4 pattern
  ("no single figure is *the* nine-seed value").
- **R3:** §4.6's audited cell **3.72 → `d/s_fit = 3.59`**; the column header now names the ruler; the prose
  and §4.6.1 both re-quote 3.59 and **name `sep/s_fit = 3.71` as a different ruler that is never used as
  `d/s`** (SF-6 pattern). Fig 3's spec states 3.59 (the render already plots it).
- **R4:** the `0.814` row's interval **±0.40 → ±0.31** (2·grad_se = 0.3076, printed to the table's 2 dp).
- **R5:** Fig 1's mixed-`n` caption rule **deleted**; the spec now says uniform `n = 9` and explicitly
  forbids carrying the old rule.

### (5) r3's own reconciliations (Hub ruling 3)
- **N186 causal framing KEPT and softened** to the n = 9 form at both sites (§2.5 (i) and §4.1.1): *"a
  candidate mechanism for why the written store reads no better than blank."*
- **2605.17590 cut RATIFIED** — nothing re-added (verified: 0 occurrences).
- **Poliak ⟦CITE2⟧** at §1, §2.4, §5.3 (+ the Mamba-2 citation facts at §1, §5.1, A.1f) — non-blocking.

### (6) Figures
App K gains a **render-status block** (Figs 1–5 rendered, PNG + PDF, 48-entry provenance table, the two
declared non-artifact quantities named). Fig 1 spec = the target seven-bar state with the three hatch
classes + the ⚠ note that the current render has six; Fig 2 gains modal-ledger caps and the SSD bar; Fig 3
names `d/s_fit = 3.59` and the presentational shading edge; Fig 5 gains the **two write loads** (closures
§2) and the **0-of-20** rival null. §1.1's headline-figure line no longer says "not rendered".

### Also folded (appendix maximalism, C-10 — nothing pruned)
New negatives **App L 21–24**: the seed-dependent CLU ledger (a *refuted* prereg, P9) · rescue verdicts
depending on the selection rule · the frontier staying a labelled null at 20 cells · the SSD arm's tie
margins + the un-quotable ordering. App J rewritten: Mamba-2 **out** of NOT-RUN; **in**: Mamba-1/Mamba-3,
the SSD 5×-budget re-check, the SSD paired `full − null` and per-reader means, the SSD per-head-width +0 B
margins, and the CLU frontier curve at n = 9. App M's reproducibility paragraph re-stated at uniform n.

---

# 2. Claim changes flagged (the F3 pre-commitment pattern)

| # | claim | before | after | verdict moved? |
|---|---|---|---|---|
| 1 | CLU rescue verdict | NOT RESCUED (n = 3 sign fact, "below blank") | **NOT RESCUED (n = 9, indistinguishable from blank)** | ⛔ **no** — direction unchanged, *content* changed |
| 2 | CLU dividend significance | "the launder is *better*" | **1.78 SE — "reads no worse than"** | ⚠ **yes, softened** |
| 3 | audit headline arm count | 0 of 5 | **0 of 6** | strengthened, no sign change |
| 4 | headline margin range | −0.2592 … −0.4602 | **−0.2563 … −0.4602** (min SE mult. **4.43 unchanged**) | no |
| 5 | rescued set | {deltanet, gdn, gdn2} | **{gdn, mamba2}** quotable; {deltanet, gdn2} SELECTION-DEPENDENT | ⚠ **yes** |
| 6 | frontier null | 0 of 15 | **0 of 20** | no |
| 7 | CLU byte ratio | 54.56× (stated constant) | **54.56× modal (8 of 9)**; 45.60× on seed 8 | ⚠ **yes, labelling** |
| 8 | audited `d/s` (fitted ruler) | 3.72 | **3.59** | no (the atom-width 4.34 and the 1.53e-2 coupling are unchanged) |
| 9 | §4.6's 0.814 interval | ±0.40 | **±0.31** | no (inadmissible row) |
| 10 | measured vs reasoned families | 3 measured / 3 reasoned | **3 measured (SSM now measured) / 2 reasoned** | ⚠ **yes** |

---

# 3. How I verified

| check | command / method | observed |
|---|---|---|
| every replacement is unique and intentional | six scripted fold batches (`.claude/scratch/bprime-draft-r4/fold{1..6}.py`), each asserting `count(old) == 1` before writing | **110 replacements applied, 0 ambiguous matches**; one mismatch caught and fixed on batch 1 (a line-wrap difference) rather than force-matched |
| number provenance | every new figure transcribed from `bprime-referee-closures.md` §1/§5 and `bprime-mamba2-arm.md` §1/§1.2/§3.2/§3.5/§3.6/§4.2/§5 | ⛔ **no number in r4 is computed by me.** Two quantities I could have computed (the SSD paired `full − null`; the CLU seed-8 coverage-to-ledger correlation) are declared NOT-RUN / stated unpaired instead |
| markdown table integrity | pipe-count check over all 42 tables | consistent everywhere except two cells containing escaped `\|Δ\|` (intentional, renders correctly) |
| A18.1 + never-quote sweep | 23-pattern regex sweep (§4) | **0 violations** |
| cross-section contradiction hunt (C-7) | grep for every quantity that appears in ≥2 sections at different `n` | one found (blank store −0.4221 vs −0.3906) and closed by labelling both runs in App N.1 |
| line count | `wc -l` | 2231 → **2621** |

---

# 4. The A18.1 / never-quote sweep (task item 7) — printed

```
clean  A18.1 · '3.6 SE' in any form
clean  A18.1 · any n=3 rescue verdict quoted as live
clean  A18.1 · 'no margin against DeltaNet is quotable'
clean  A18.1 · 'R5: 3 of 5' unlabelled
HIT*   A18.1 · P5-vs-raw n=3 magnitudes            (2 — both FALSE POSITIVES: "0.2037" and "−0.5052")
clean  A18.1 · 'the frontier column shows'
clean  A18.1 · 'P3 was refuted'
HIT*   A18.1 · 'verified to 1e-9 in all 28 cells'  (1 — inside §3.1's erratum, quoting our own wrong text)
clean  A18.1 · cluformer numbers / CLU-former name
clean  A18.1 · 'needs > 500 M'
clean  r4   · headline still '0 of 5' rival arms
HIT*   r4   · CLU n=3 rescue verdict               (1 — I.1c's labelled before/after supersession block)
clean  r4   · below-blank drama survived
HIT*   r4   · stale CLU +0B margin / full          (3 — all inside I.1c's supersession block + A.1e's
                                                    fidelity check, i.e. the mandated before/after)
clean  r4   · stale d/s 3.72 as the audited cell
clean  r4   · stale ±0.40 on the 0.814 row
clean  r4   · Mamba-2 still NOT-RUN / outside ruled set
clean  r4   · '0 of 15' unqualified frontier claim
clean  r4   · unlabelled 54.56 without modal (main text)
clean  r4   · selection-dependent arm quoted as plain RESCUED
clean  charter C-1 · defensive audit-confession paragraph
```
**Verdict: 0 real violations.** The seven hits are (a) two substring false positives, (b) the erratum that
*must* quote our wrong published form, and (c) four occurrences inside the **labelled before/after
supersession** blocks that A18.1 itself requires beside a moved number.

**Charter compliance spot-check:** C-1 (no audit-confession paragraph — none present; J&P 2026 cited once,
for the primitive's introduction only, with the mandatory continuity sentence) · C-2 (verification/evidence
labels intact at §1.3, §3 preamble, §4 preamble, §4.6) · C-3 (ML-first; contributions on page 1; one
headline figure named) · C-5 (`d_in = 5`, 5–6 items, ~10-token streams, CPU, seed counts in-sentence on the
abstract, §4.2, §7, L11) · C-6 (the gate's power + selection requirements sit next to the verdicts they
qualify, not in a footnote) · C-7 (three new flag-provenance tables: A.1e, A.1f, and A.1's new arms row) ·
C-8 (hermetic — no other short cited; the SSD arm cited to Dao & Gu, the GDN authors' own characterisation
quoted) · C-9/C-10 (negatives extended to 24 items; nothing pruned; all new material in appendices).

---

# 5. Open editorial questions for the Hub / Head

1. ⭐ **Fig 1 re-render (reconciliation 1)** — do you want r4 circulated with a six-bar Fig 1 + the
   in-caption note, or does the re-render gate circulation?
2. **Is `{GDN, Mamba-2}` the right restricted form for §4.2?** It is the honest intersection, but it drops
   the paper's restricted headline from three arms to two. The alternative — quoting the primary rule's
   `{deltanet, gdn, gdn2, mamba2}` and labelling — is *less* conservative and I did not take it.
2b. Related: §6 L2 now says the comparative half of the audit is carried by **two** arms. That is the
   thinnest the audit has read. It is also true. Flagging in case the Head wants the framing softened —
   I did not soften it.
3. **Titles.** The working title still leads on *"test-time dynamics vs a table at matched bytes"*. With
   the SSM arm in, "test-time dynamics" is now slightly narrow for the arm set (Mamba-2 does no test-time
   learning). Candidate reframe for the title workshop: *bounded-state memories* rather than
   *test-time dynamics*. Not changed unilaterally.
4. **Does the CLU column belong in the harness?** The closures spoke flagged that `audit_table` emits only
   `clu_reproduced.{full,launder,dividend}` while the blank/null/+0 B/lift columns are aggregated by a
   scratch script. The paper now leans on those columns for its own arm's verdict. That is a **code** task
   (~30 lines + a test), and App A.1e discloses the aggregation route honestly in the meantime.
5. **App L is at 24 entries and App J at 20+ declared NOT-RUNs.** Both are per C-9/C-10 policy, but the
   pruning pass will need a rule for which negatives are main-text-adjacent.

---

## Proposed handover updates (for the Hub)

1. **`draft-r4.md` exists** (2621 lines) with the CHANGELOG line carrying per-item disposition. `r3` is
   superseded for every number; nothing from `r3` was deleted, only relabelled or moved to labelled
   history (I.1c(f), H.2, A.1b).
2. **Quotable paper-side forms after this fold:** *"0 of 6 rival arms, −0.2563 … −0.4602, every margin
   ≥ 4.4 SE below zero at n = 9, and the CLU at −0.2897 ± 0.0328 (8.8 SE)"* · *"the CLU is NOT RESCUED at
   n = 9 — statistically indistinguishable from an empty store (lift −0.0465 ± 0.0406)"* · *"the CLU's
   dividend is −0.0561 ± 0.0315, a 1.78 SE sign statement — the launder reads no worse than the store"* ·
   *"rescued and selection-stable: `{gdn, mamba2}`"* · *"the byte-frontier column is 0 of 20 at n = 9."*
3. ⛔ **New never-quote candidates this fold makes necessary** (curator): (a) *"the CLU reads below its own
   blank store"* — retired, the n = 9 lift is inside noise; (b) *"the launder beats the store"* — 1.78 SE,
   the licensed form is *"reads no worse than"*; (c) *"deltanet / gdn2 are RESCUED"* **unlabelled** — they
   are SELECTION-DEPENDENT; (d) *"5456 B / 100 B / 54.56×"* as an unqualified n = 9 CLU ledger — modal
   (8 of 9); (e) *"0 of 15 frontier cells"* — it is 0 of 20; (f) *"the audited configuration sits at
   `d/s = 3.72`"* — that is `sep/s_fit`; the fitted-width ruler gives **3.59**.
4. **Render request** (reconciliation 1) and the **two declared missing aggregations** (reconciliation 2)
   are the only open dependencies the draft has on new work. Neither blocks a referee pass.
5. **`bprime-cite-check-2` inherits four ⟦CITE2⟧ sites** (§1, §2.4/§5.3 Poliak; §1/§5.1/A.1f Mamba-2) —
   the marker must be removed only by that spoke.
