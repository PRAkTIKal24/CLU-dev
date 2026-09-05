# v5-palm-reframe — paper-writer report

**Task + acceptance criterion:** produce a PALM-audience reframe of V5 as a separate variant in
`papers/palm-variant/v5/` — same results, same numbers, same claims, re-expressed policy-question-first —
with main text ≤ 4 pp, total 8–9 pp, the TTL comparison leading the leakage result, and
`papers/v5-short/**` byte-untouched.
**Status: done.** Main text **4.00 pp**, total **9 pp**, all four acceptance criteria met and printed below.
**DIAL DECLARATION (echoed): none — reframing/editorial pass; zero content, number or claim changes.**
No laundering control applies (no new performance number is produced); the paper's own TTL laundering
control is *promoted*, not re-run.

> ## ⚠ DOWNSTREAM RECONCILIATION LIST — READ FIRST (needs a named owner)
> 1. **⛔ The task brief mis-states one measured number, and I did not follow it.** The brief (and Add.38)
>    quote the σ-limited TTL comparison as *"0.559 vs 1.000 at σ = 0.1"*. The measured table in
>    `papers/v5-short/submission/submission.tex` Appendix D reads **physical decay 0.559 / TTL flag 0.996 /
>    separation 0.437**. This build prints **0.559 versus 0.996**. ⇒ **Add.38's text and any Hub note that
>    inherited it need the correction**, or the next writer will "comply" with a wrong number.
> 2. **⚠ Three reference entries in the LIVE V5 submission build carry placeholder titles** — Yang (2026)
>    *"What breaks in production agent memory"*, Uddin et al. (2026) *"Failure modes of memory in deployed
>    recommendation agents"*, Mo (2026) *"Symmetry-protected memory in recurrent dynamics"*. None is the real
>    title; the canonical `draft.md` has all three correct. **This variant uses the canonical records; the
>    live PALM build still has the placeholders and needs a one-line fix.**
> 3. **⚠ A figure-generation job falls out of this pass.** Meeting 4.00 pp forced the headline figure to
>    0.60 `\linewidth` (measured maximum; 0.62 and above spill). `fig1_damping_optimum.png` should be
>    re-rendered with larger axis/tick fonts for this variant. Text-only pass; not attempted here.

## What I did

- Read `AGENT_PROTOCOL.md`, the Positioning Charter (`outputs/philosophy-synthesis.md` §"Positioning
  Charter", C-1…C-10 as they read today, incl. the **C-1 reversal — no audit-confession paragraph**, which
  this draft honours: there is none), `claims_matrix.md`, `PJ_Writing_Style_Context.md`,
  `advisor-head-shorts-charter.md` Add.26/28/34/36/37/38, `outputs/v5-scope-scout.md` (all four parts),
  `outputs/v5-referee-v02.md` §D (the do-not-cut list), `outputs/v2-cite-check.md` (carried records), and
  both the source submission build and the canonical `draft.md` (for K.2 and the verified bibliography).
- Created `papers/palm-variant/v5/` with `submission.tex`, `submission.pdf`, `BUILD-NOTE.md`, `figs/`,
  the four component `.tex` files (kept for auditability, per the source build's convention) and the `.sty`.
- **Reframed:** abstract · §1 · §2 (rebuilt from the scout's 13-work brief) · the three §3 subsections'
  titles and opening frames · §3.3's leakage paragraph re-ordered so the TTL comparison leads.
- **Left byte-identical:** Appendices A–E in full, the nomenclature block, every fine-print block, every
  protected wording, the Limitations block, the figures, the style file.

## How I verified (commands + observed output)

| check | command / instrument | result |
|---|---|---|
| build | `pdflatex ×3` (TeX Live 2026, `/Library/TeX/texbin`) | `Output written on submission.pdf (9 pages)`, **0 errors, 0 undefined references** |
| boxes | `grep Overfull/Underfull submission.log` | 2 overfull (`91.6832pt`, `406.18022pt`) + 1 underfull — the two overfull values are **numerically identical to the source build's log**, i.e. inherited from Appendix B's wide `\tiny` tables |
| main ≤ 4 pp | `pdftotext -f 5 -l 5` | p. 5 line 1 = `References` ⇒ **main text ends at the foot of p. 4 = 4.00 pp** |
| total | PDF | **9 pp** (main 4.00 · references 0.75 · appendices A–E 4.25) |
| numeric two-way | `scratch/v5-palm-reframe/check_numeric.py` (comment-safe tokenizer) | see below |
| compliance sweep | `scratch/v5-palm-reframe/subsweep.py` (the source build's own instrument), per-file, positive-controlled | **zero-list hits = 2/38 patterns**, both the known false positive; **instrument LIVE** (14/14 positive controls fired) |
| `papers/v5-short/**` untouched | 25-file `md5` manifest before/after, `diff` | **empty diff — byte-identical** |
| appendix fidelity | `diff <(sed -n '115,312p' source) appendix.tex` | **empty** |

**Numeric two-way check, printed in full.**
- (A) tokens in this build **not** in the source build: `0.4 0.7 0.9` (LaTeX lengths) + `10.1007 10.1109
  10.1145 10.1162 1997.9 2007.36 2024 2105.06548 2305.10250 2310.08560 2404.07143 258533.258638 2602.17692
  2604.20006 2605.03338 2606.15903 2606.29788 380752.380844 540 69903 8.1735 978` — **all bibliographic
  identifiers of the six added / three restored references, every one present in the canonical `draft.md`.**
  Of the 25, exactly **1** (`0.7`) is absent from `draft.md`, and it is a `\@startsection` length.
- (B) source tokens absent here: **`0.84`** only — the source's headline-figure width.
- (C) source **main-text** tokens absent from this **main text**: **`0.84`** only ⇒ **no content number
  left the main text.**
- (D) new main-text tokens vs source main text: `0.1 0.559 0.996` (the promoted TTL row — **verified
  present in the SOURCE build's Appendix D table**), `2024` (two new citation years), and `0.4 0.60 0.7 0.9
  6` (LaTeX lengths).

**Sweep detail.** Positive controls: 107.77 ×8 · 106.1 ×3 · 0.9001 ×2 · Blelloch ×10 · N108's sentence ×2 ·
"confines" ×3 · 8.11 ×6 · Anonymous ×2 · "introduced as CHLU" ×1 · verification ×8 · evidence ×12 ·
9.5e15 ×1 · 0.4586 ×2 · ZERO ×2. Zero-list: clean on 36 of 38 patterns; the two hits are `n_{\rm R1}` /
`\Gamma_{\rm R3}` header cells in Appendix B Table 3 — **this paper's own instrument names**, defined three
paragraphs above, and the identical false positive the source build reported. Context-check class:
`certified` ×3 (2 literature + the denial) · `unlearning` ×6 (denial + 1 literature sentence + 4 reference
entries) · "deletion is exact" ×2 (both store-level-qualified) · `CHLU` ×2 (continuity sentence + reference
entry) · `0.99985` ×1 (carries "at full load") · `297.8` ×1 ("never the vault number") · `23.39` ×3 (all
designed-only / falsifier-fired). **Semantic hermeticity (C-8): `companion` / `sibling` / `our other short` /
`the program` / `forthcoming` / `in preparation` = 0.**

**Anonymization.** `\author{}` blank · `\textbf` = 0 · no `[WORKING TITLE` / `[AUTHORS PLACEHOLDER]` ·
no acknowledgment, funding, URL or repository string · PDF Title/Author/Subject/Keywords/Creator/Producer
all empty · decompressed-PDF string sweep: `/Users/` 0 · username 0 · `Desktop` 0 · `.claude` 0 · `CHLU-` 0 ·
`palm-variant` 0 · `chlu/` 0 · `ml4ps` 0 · **`PALM` 0** (no venue string anywhere) · `Pierini` 2 = the
sanctioned continuity sentence and its reference entry, exactly as in the source build. PALM's
code-inclusive requirement is carried by the closing anonymization note.

## Findings/results — the reframe itself

**1. The spine is policy question → mechanism → number.** §1 opens on the retention policy every deployed
store already runs, then states the three operational questions (*what is the retention policy and where is
its optimum · can retention be scoped to one item · is deletion real and what still leaks*) and maps them to
§3.1/§3.2/§3.3. Physics enters one sentence later, explicitly *"only as the derivation apparatus"* (C-3).

**2. The three contributions are restated in the audience's terms**, with the ⭐ policy statement carried
in-sentence: (1) **a retention dial with a computable optimum** — *"retention is non-monotone in the dial and
the optimum is predicted, not tuned"*; (2) **scoped retention** — *"a local change to the dial confines one
item's coordinate"*; (3) **a structural deletion guarantee**, stated as the composition claim with the
Blelloch–Golovin attribution and all three conditions verbatim, and phrased as *"makes the store's state a
function of its live set alone"*; (4) the lifecycle, in its approved mechanics-only form, labelled *"the
consolidation and stale-entry face of the same store"* (CFP topics 1 and 4).

**3. ⭐ The TTL comparison leads the leakage result, with N108's sentence verbatim beside it.** §3.3's final
paragraph now opens: *"The tighter laundering control is the policy already deployed elsewhere, a boolean TTL
flag inside the same store, and it fires: against an exact adversary the TTL flag stays within 0.017 AUC of
physical decay (0.983 versus 1.000), the two separating only against a resolution-limited adversary (0.559
versus 0.996 at σ_obs = 0.1), whose resolution is our own modelling choice."* — immediately followed by
**"The store stops answering before it stops leaking"** verbatim, and closing on *"What physical decay buys
over a TTL flag is retrieval geometry, not privacy"* with the R₅₀ 1.146→0.752 (1.52×) differentiator.

**4. §2 is rebuilt for this audience from the scout's brief, in three strands** — (i) retention policies in
deployed systems (MemGPT, Mem0, Infini-attention; Generative Agents 0.995, MemoryBank, Expire-Span, **Titans
named as "the nearest published neighbour to the dial of §3.1"**); (ii) deletion, deployed and formal (Zep's
invalidation as the opposite design point, Ghost Vectors as why byte-identity matters, the 2026 wave —
Yang/Uddin/MemLeak/agentic unlearning — then Guo §2 Eq. (1), SISA/Ginart/Sekhari, and the Snyder → Naor &
Teague → Blelloch & Golovin → BGV lineage); (iii) forgetting laws, learned and physical (LSTM/LEM,
Minami & Hidaka 2018, Mo 2026). **Prose lifted and compressed from the canonical `draft.md` K.2 and from
`outputs/v5-scope-scout.md` Part 2. Mechanically checked: 13 of the scout's 13 non-optional venue-native
works are cited (MemFail/Garg et al., which the scout marks optional, is not), and no citation appears that
the scout did not verify.**

**5. The honest scope sentence is present exactly once, in §1, in the paper's voice:** *"What class of claim
this is. Our evidence is a small designed store measured at laptop scale: not a deployed agent memory, not
an LLM system, not a benchmark result. We report a mechanism with measured laws, not a system result."*
It sits directly under the contributions. The `Scale is a scope choice` sentence and the score sentence
(*"external benchmarks won on their own headline metric = ZERO"*) are both present, verbatim, unmoved.

**6. Operational vocabulary — adopted vs refused.** Adopted (scout-backed): *retention policy* ×3 ·
*retention dial* ×4 · *TTL* ×11 · *expiry* · *consolidation* · *scoping/scoped* ×16 · *deletion guarantee* ×2
vs *best-effort* · *membership* ×4 · *stale*. ⛔ **Refused: "right-to-be-forgotten" (0) and "memory
provenance" (0).** Both name compliance properties of a deployed system; using either would convert a
store-level bit-exactness statement with the encoder excluded into a system guarantee — the Add.37 FLAG-2
boundary (*where an audience term would widen a claim, our term stays*). This is a judgment call and the
Advisor may want to review it: it costs two verbatim CFP-topic keyword hits.

**7. Every do-not-cut item is present and verbatim — mechanically checked, 19/19 exact string matches in
both this build and the source build, 0 failures:**
N108's sentence · the exact-deletion quote with its three conditions and the recency exclusion · the
Blelloch–Golovin attribution sentence · the lifecycle's two riders (7/7 legs + "one leg unexercised";
"no value or benchmark number is claimed, and none was run") · the substrate-scope sentence · the score
sentence · the designed-symmetry precondition · the `fdt`+Newtonian fine print · the emergent-arm caveats
(no σ_θ ratio, the θ=π confound in Appendix C/E, the contrast number designed-only) · the trilemma corner ·
the R₅₀ differentiator · the CLU continuity sentence · scale-as-scope-choice · the quote-the-curve load on
0.99985.

**8. Style (Add.30 / `PJ_Writing_Style_Context.md`), applied more directly than the source.** ABT openings:
*abstract* — AND what a system forgets is set by policy, BUT a policy states when an entry expires and not
how fast/what remains, THEREFORE three answers with numbers. *§1* — AND every store runs a retention policy,
BUT forgetting is the failure mode benchmarks do not measure and a policy is not a law, THEREFORE a memory
whose forgetting is a dynamical property. *§3.1* — AND damping is the dial, BUT more friction reads as
faster forgetting, THEREFORE the optimum is predicted, not tuned. *§3.2* — AND scoping needs a local change,
BUT that needs the sign of the dial's effect, THEREFORE friction preserves and a hole is a vault. *§3.3* —
AND deployed deletion is best-effort, BUT whether anything remains is empirical, THEREFORE make the state a
function of the live set. Macro-to-micro enforced per section (policy → mechanism → number → fine print);
`\textbf` = 0; signposting explicit; "we" for our actions, passive for established facts; every generalizing
claim carries its scale qualifier in-sentence (C-5).

## Deltas vs the source build, stated so nothing is smuggled

1. **TTL row promoted** Appendix D → main text (`0.559` / `0.996` at σ_obs = 0.1). Numbers unchanged; both
   appear verbatim in the source build's Appendix D table.
2. **Three reference records restored** to the canonical/scout-verified form (Yang, Uddin et al., Mo).
3. **Six references added**, all cited in §2 and all scout-verified: Packer 2023, Munkhdalai 2024,
   Zhong 2024, Sukhbaatar 2021, Wang & Zhang 2026, Wang et al. 2026. 24 → 30 entries.
4. **One typo corrected**: the source §2's `"and and exact methods"`.
5. **Headline figure 0.84 → 0.60 `\linewidth`** — measured maximum compatible with 4.00 pp (0.62/0.64/0.66/
   0.70/0.76/0.84 each built and each spills to p. 5, total 10 pp). Appendix figure widths unchanged.
6. **Typography** (content-neutral, listed in `BUILD-NOTE.md` §6): §2 + nomenclature + all fine print +
   Limitations at `\scriptsize`; appendix body `\scriptsize` (tables keep their own explicit sizes, so no
   table changed); tighter heading/float skips; references in two `multicol` columns.

⇒ **This variant is 9 pp where the live PALM build is 10 pp**, with the same content. Items 6's levers
(two-column references, `\scriptsize` appendix body) are the reason, and they are **directly transferable to
the live build** if the Head wants its 1-pp overshoot closed without touching content — that is the
Advisor's recommendation in Add.37 that a reframe should also improve the primary build, discharged
concretely.

## Acceptance criteria — one line each

1. **Main ≤ 4 pp ✔ (4.00, measured) · total 8–9 pp ✔ (9) · split printed ✔** (main 4.00 / refs 0.75 /
   appendices 4.25).
2. **Every operational term scout-backed or defined in one clause ✔; every citation scout-verified ✔
   (30/30, records from `v5-scope-scout` or `v2-cite-check` via canonical `draft.md`); honest scope
   sentence present exactly once ✔.**
3. **TTL comparison leads the leakage result ✔, with N108's sentence verbatim beside it ✔.**
4. **Numeric check + all sweeps printed ✔; `papers/v5-short/**` byte-untouched ✔ (25-file md5 manifest,
   empty diff).**

## Open editorial questions for the Hub / Head

1. **Title — the Head's call.** In the artifact: *"Retention You Can Predict, Scope and Delete: a (μ,γ,T)
   Law and a Structural Deletion Guarantee for a Physics-Structured Memory Store."* Alternatives, in the
   order I would rank them: (b) *"What Is the Retention Policy, Is Deletion Real, and What Leaks? Three
   Measured Answers for a Physics-Structured Memory Store"* (most PALM-native, most unusual); (c)
   *"A Retention Dial with a Computable Optimum, Scoped Retention, and a Structural Deletion Guarantee"*
   (flattest, most literal). I used a real title with a blank author block, matching the source build's
   anonymized-submission convention rather than the `[WORKING TITLE]` placeholder convention.
2. **Do we want "right-to-be-forgotten" anywhere?** I refused it (item 6 above). It is a verbatim CFP topic
   heading and would be a keyword hit; it is also the one term most likely to be read as a compliance claim.
   ⇒ **Advisor/Head ruling wanted**; a one-clause form (*"the deletion tests and right-to-be-forgotten
   mechanisms this venue asks about are system-level; ours is a store-level property"*) would be defensible
   and costs ~1 line, which currently does not exist in the 4-pp budget.
3. **Which build goes to PALM?** This variant and the live build are now two venue-compliant artifacts for
   the same venue with the same numbers. The Head ruled the live build primary; if this reframe is preferred,
   the live build's three placeholder references (reconciliation item 2) become moot, and if the live build
   is kept, they must be fixed there.
4. **Figure 1 re-render** (reconciliation item 3) — worth doing whichever build ships, since 0.84 was already
   a 1.8× downscale of the source PNG.

## Risks

- **The 4-pp fit is typographic, not textual.** Main text is 3,051 words against the live build's 2,572 (same
  tokenizer) and fits only because §2, the nomenclature and all fine print run at `\scriptsize`. In the real
  NeurIPS-2026 style file this could tip either way. **Re-measure before freeze** — the same caveat the live
  build carries, and it is tighter here.
- **`\scriptsize` for §2** is small for a related-work section a reviewer will actually read. If the Head
  dislikes it, the cheapest compensation is item 2 of the referee's own move-menu (§3.3's composition clause
  to Appendix D, ≈25 words) plus dropping the §3.2 opening frame — about 3 lines, which is what §2 costs to
  go back to `\footnotesize`.
- **I did not re-verify any citation record.** Every entry is transcribed from `outputs/v5-scope-scout.md`
  (fresh 2026-08-19) or the canonical `draft.md`. Three references still lack primary verification upstream
  (Snyder 1977, Sundar & Tarjan 1990, Andersson & Ottmann 1995 — all taken from Blelloch & Golovin's
  reference list), as `v5-revision-1` already flagged; Snyder and Andersson & Ottmann are cited here.
- **Inherited, not introduced:** the two overfull hboxes; the `Anonymous (2026)` theory-note reference; the
  third-person self-citation appearing in a double-blind build; the NeurIPS-family style file standing in for
  the unobtainable venue template.

## Git footprint

**None.** No tracked file was created, modified or deleted; everything written by this pass lives under
`.claude/` (`papers/palm-variant/v5/**`, `scratch/v5-palm-reframe/**`, `outputs/v5-palm-reframe.md`).

## Proposed handover updates (for the Hub)

1. **`papers/palm-variant/v5/` exists and is venue-compliant**: main 4.00 pp, total 9 pp, 0 errors,
   0 undefined references, all sweeps clean, `papers/v5-short/**` byte-untouched (md5 manifest, empty diff).
   Add.38's spoke is discharged.
2. **⛔ Correct Add.38's TTL number.** It says *"0.559 vs 1.000 at σ = 0.1"*; the measured value for the TTL
   arm is **0.996**. Any note that inherited the brief's phrasing needs the same correction.
3. **Open a one-line fix task on the LIVE V5 build**: Yang (2026), Uddin et al. (2026) and Mo (2026) carry
   placeholder titles in `papers/v5-short/submission/submission.tex`; the canonical `draft.md` has the
   verified records.
4. **Transferable page win.** Two-column `\tiny` references + `\scriptsize` appendix body take the live V5
   build from 10 pp to inside the 8–9 pp band **without touching content**; the Head's Add.36 costed menu
   (which required dropping an appendix) can be retired if this is adopted.
5. **New figure task (small):** re-render `fig1_damping_optimum.png` with larger axis/tick fonts; at 0.60
   `\linewidth` (the measured maximum for 4.00 pp here) the current PNG is a 3.7× downscale.
6. **Ruling wanted** on whether "right-to-be-forgotten" may appear in a one-clause, explicitly system-level
   form (open question 2). I refused it on the Add.37 FLAG-2 boundary; that is a judgment, not a rule.
7. **Framing language worth back-porting to the live build** (Add.37's Advisor recommendation, made concrete):
   the three-question spine in §1; the contribution labels *retention dial / scoped retention / structural
   deletion guarantee*; the TTL-leads-the-leakage ordering in §3.3; the *"what class of claim this is"*
   sentence. All four are content-neutral and each is a strict improvement in reviewer-legibility.
