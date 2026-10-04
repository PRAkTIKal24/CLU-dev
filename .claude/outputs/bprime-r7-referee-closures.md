# bprime-r7-referee-closures — paper-writer report

Task + acceptance criterion: close the `bprime-r6-referee` findings (MF-1/MF-3, SF-1…SF-10, NICE at judgment) + the `bprime-r6-cite-check` reconciliation items 1–3 + 5 into a new `papers/bprime/draft-r7.md` with a CHANGELOG entry listing every changed passage by finding number; ⛔ zero new measurements; `draft-r6.md` byte-untouched.
Status: **done** — every MF/SF closed or explicitly routed; 7 of 7 NICE handled (6 applied, **N-1 deferred to the Hub with reasons, not silently dropped**).

**DIAL DECLARATION (echoed): none — revision pass; no new measurement; no laundering control applies.** Falsifies the deliverable: a referee finding neither closed in `draft-r7.md` nor named as deferred in the CHANGELOG entry.

## ⛔ RECONCILIATION LIST — needs a Hub owner (protocol §5 corollary, first 10 lines)
1. **MF-4 is unowned outside this pass and I made no draft edit for it** (as tasked). The §0.12 "no paper number" vs Head-commissioned-fold **permission** ruling must be written into the matrix as a dated clause covering exactly the N276/N294–N296 material; §0.12's vehicle **v2.15 is PROPOSED** and **v2.16/v2.17 confirmation is owed**. Owner: **Hub/Head**.
2. **N-1 is a ruling question on an approved wording and I did not invent a clause.** (a) Does §0.12's approved form subsume N276's scope rider *"pass 1's 1,253× stream-launder caveat travels unchanged"*? (b) The approved wording pairs an **832 B cue-side** table with **stream** reads; a reviewer will ask what the stream-side table's bytes were, and **that number is not in any artifact I hold** (A.5 gives 832 B cue / 416 B ring only). Owner: **Hub** — improvising either would have been a number I could not trace.
3. **`negative_results.md` N294 carries the same "a permitted 3,000" wording the draft inherited** (SAM-kNN STM cap). The draft is corrected at r7; the **registry line is not**, and the cite-check's finding (it is a reference-implementation default `ltm_size = 0.4`, absent from the ICDM paper) should be annotated there. Owner: **curator**.
4. **Cite-check reconciliation item 4 is a curator item, not mine:** the r5-era standing flag *"the Metro Interstate DOI … treat it as unverified"* (`c2w10-benchmark-scout.md` l.203) is **discharged** (DOI resolves 302 → dataset 492) and wants a one-line erratum on that report. Owner: **curator**.
5. **Environment defect, program-wide, disclosed:** the **main venv cannot import matplotlib on this machine** — `pyparsing/__init__.py` and several matplotlib deps are macOS **`dataless`** (cloud-evicted) and every read times out (`TimeoutError: [Errno 60]`), including after `brctl download`; the same failure reproduces in three other project venvs. Any future figure/plot spoke will hit this. My workaround is recorded below. Owner: **Head** (machine state).

## What I did
- Read, in order: `AGENT_PROTOCOL.md` · the task file · `bprime-r6-referee.md` (all findings) · `bprime-r6-cite-check.md` (all 371 lines incl. the BibTeX block) · `claims_matrix.md` §0.9's N224 block + CM-30's instrument clause + §3.1's §A20.5 row + the CM-27/N190 approved companion clause · `advisor-head-shorts-charter.md` §A20.5/Addendum-12 lines · registry **N276 / N294 / N295 / N296** (for the N-2 and N-5 adjudications) · **all 3,218 lines of `draft-r6.md`** · `papers/bprime/CHANGELOG.md`.
- Created `draft-r7.md` as a byte-copy of `draft-r6.md`, then applied **34 edits** through a harness that **asserts an exact, unique match** for every replacement (`.claude/scratch/bprime-r7-referee-closures/edits_batch*.py`); zero fuzzy edits, zero manual retyping of any existing number.
- **Re-rendered Figure 1 only** with the Mamba-2 bar, from banked artifacts, with a renderer that fails loudly rather than drawing if the bar count ≠ 7 or any bar is not at `n = 9`.
- Ran the never-quote sweep **per file with positive controls**, an approved-wording integrity check, a rider-count regression against r6, and a `d/s`-rider adjacency scan.

## How I verified (commands + observed output)
**r6 immutability.** `md5 draft-r6.md` = `bc441dc29b9814b8bdb60506988caef2` **before and after** the pass (recorded at copy time and re-checked at the end). r6 = 3,218 lines, r7 = 3,658 lines; `diff` = **+508 / −68** lines, and I read every one of the 68 deleted lines: all 68 are the edit sites listed in the CHANGELOG, and **no approved-wording blockquote line is among them**.

**Figure 1 (MF-1), rendered — real numbers, all read from banked JSON:**
```
TTT-Linear        -0.460206 ± 0.103807  n=9  INIT-UNSTABLE
TTT-MLP           -0.442517 ± 0.086899  n=9  NOT RESCUED
DeltaNet          -0.273165 ± 0.039452  n=9  SELECTION-DEPENDENT
Gated DeltaNet    -0.259986 ± 0.027793  n=9  RESCUED
Gated DeltaNet-2  -0.259189 ± 0.029232  n=9  SELECTION-DEPENDENT
Mamba-2 (SSD)     -0.256317 ± 0.041557  n=9  RESCUED
CLU (ours)        -0.289729 ± 0.032793  n=9  NOT RESCUED
```
Every value matches §4.1.1's printed table to the digit. Gate statuses are **derived in code**, not typed: `RESCUED` iff true under all three registered selection rules; `SELECTION-DEPENDENT` iff true under both fit-split rules and false under held-out (`f3_val`); `INIT-UNSTABLE` iff the two code paths disagree; `NOT RESCUED` otherwise. Derivation inputs, observed:
```
f3           ttt_linear F  ttt_mlp F  deltanet T  gdn T  gdn2 T   | mamba2 T
f3_lite      ttt_linear F  ttt_mlp F  deltanet T  gdn T  gdn2 T   | mamba2 T
f3_val       ttt_linear F  ttt_mlp F  deltanet F  gdn T  gdn2 F   | mamba2 T
c2w4_code    ttt_linear T  ttt_mlp F  deltanet T  gdn T  gdn2 T   | mamba2 (not scored)
```
⇒ exactly the r6 draft's own verdict set. **Zero cells were measured or re-measured.**

**Never-quote sweep (per-file over `draft-r7.md` only; single-file `grep -c`).** Positive controls, all returning hits: `54.56` (14) · `306.76` (3) · `14.35` (2) · `0.2897` (13) · `no-daylight` (4) · `2,364` (6) · `ARF's shoulder` (3). Zero-hit patterns: `3.6 SE` · `CLU-former`/`cluformer` · `K5` · `UNASKABLE` · `0.2719` · `0.7254` · `300.09` · `S_eff` · `4.95e-63` · `n_never_read`/`frac_never_read` · `any_basin` · `best_is_also_lowest_drift` · "the completed gate" · `gate_pass` · `91.25` · "58 trips" · "principled forgetting" · "we alone delete" · `MUNKEY` · `ICML 2026` · `0.3019` · `AttentionPsi` · "Def. 1" · "0 of 5 rival" · `compositional` · `organizer` · `s10618-016-` (the mis-"corrected" Webb DOI) · "certified unlearning". **Context-adjudicated hits, all compliant:** `1089` ×1 = the self-negating erratum site (*"not the 1089× previously stated"*) · *"it beats ARF"* ×1 = **inside the mandatory never-form** I added at §2.5 · *"river ships NO SAM-kNN"* and *never "de Souza et al."* ×1 each = the verbatim **never-copy warnings inside the new Q.1 notes** (both are negations; the author field reads `Souza, Vinicius M. A.` and the ARF field reads `Pfahringer, Bernhard`, verified by printing the fields).

**Approved-wording integrity (whitespace- and markdown-normalised comparison r6 → r7).** §0.12 no-daylight form **2 → 2**, character-identical. §0.13 admissibility form **2 → 3** (the third is the new §1.1 contribution 7; character-identical to the other two). Framing rider *"a fallback being retired, not a venue crisis"* 2 → 4. N190 companion clause added ×1. §A20.5 substrate-scope sentence added ×1 (sentence-initial capital only). Rider-count regression, r6 → r7, **no count fell**: `matched-items` 2→2 · `byte-matched to nothing` 2→2 · `registered k = 5` 1→1 · `not a drift-free data source` 1→1 · `pass-3` 2→2 · `N290` 2→2 · `fifth and sixth firings` 1→1 · `one-line baseline` 2→2 · `0 of 6` 4→4 · `component-build` 5→7 · `declared NOT-RUN` 10→14.

**`d/s`-rider adjacency scan** (regex over every `d/s[_fit] = N` occurrence, ±1,500 chars): **8 sites, 8 carrying a ruler and/or the subtraction convention** after this pass (7 in body text, 1 in the delete-before-circulation editorial item 4, which I left as an internal note). The §3.3 site was the one gap and is now labelled a **designed** width rather than a fitted one.

**Bibliography fold.** The six new sources were **extracted programmatically from the cite-check report's BibTeX block and pasted byte-verbatim** (verified: `bib_text in draft` → `True`), so no `note` caveat could be lost in transcription. Q.1 now holds **14 records / 13 sources** (`river` = paper + software `@misc`).

## Findings / disposition (referee numbering)
| # | disposition | location in `draft-r7.md` |
|---|---|---|
| **MF-1** | **CLOSED** — Fig 1 re-rendered, 7 bars, uniform n = 9, banked artifacts only; disclaimer deleted | App K render-status + Fig 1 spec (+ bar table), §1.1, editorial items 4 & 11 |
| **MF-2** | **CLOSED** — six sources folded verbatim with caveats; item 10 rewritten as discharged | App Q.1 (+ preamble), Q.2 (two new not-printed entries), editorial item 10 |
| **MF-3** | **CLOSED** — subtraction convention stated; `s = 0.40` FLAGGED not discharged; direction (conservative) stated; NOT-RUN filed | §4.6, §4.6.1, §6 L9(ii), App J, abstract, §7, §3.3 |
| **MF-4** | **ROUTED, no draft edit** (as tasked) | CHANGELOG only; reconciliation item 1 above |
| **SF-1** | CLOSED — *"indistinguishable from — or below —"* + analogy pointed at the dividend column | §1.2(i), §4.8 heading + opening |
| **SF-2** | CLOSED — ordinal enumerated, substrate named *real-image* | §1.2(i), §4.8 heading/body/close |
| **SF-3** | CLOSED — sixth handicap wired (0.614; **+0.6139 ± 0.1386**) | §2.4 |
| **SF-4** | CLOSED — "three estimates by **two independent methods**" | §4.6 (+ L9(ii) harmonised) |
| **SF-5** | CLOSED — §A20.5 sentence verbatim, own voice | §1.2 (first of three hard boundaries) |
| **SF-6** | CLOSED — declaration **located in the run record, not in a dated document**; no dated prereg claimed | App I preamble |
| **SF-7** | CLOSED — six-bullet "what could have come out differently" rebuttal + pointer | App N.1, §4.2 |
| **SF-8** | CLOSED — decode sentence adjacent, approved wording untouched | §2.5 (also §1.1 contribution 7) |
| **SF-9** | CLOSED (Head-approved) — 2,364× replication into the abstract; embargo + venue admissibility into contributions; §7 closes on the strengthened base | abstract, §1.1 (contribution 1 + new 7), §7 |
| **SF-10** | CLOSED — voice struck at §4.1 and (same class) §R.2.1; item 8 names sites | §4.1, §R.2.1, editorial item 8 |
| **cite 1** | CLOSED — Webb replacement clause **verbatim** as recommended | §R.2.2 |
| **cite 2** | CLOSED — MOA-vs-river at **both** sites; F1 softening applied | §R.2.1, A.6 |
| **cite 3** | CLOSED — "a permitted 3,000" relabelled | §R.2.1 |
| **cite 5** | CLOSED — item 10 discharged | editorial item 10 |
| **N-1** | **DEFERRED to Hub** (ruling on an approved wording; the stream-side byte figure does not exist in my inputs) | — |
| **N-2** | CLOSED — INSECTS half = registered ruling, Metro half = our own inference | §2.5, App R.2 |
| **N-3** | CLOSED — criterion-4 parenthetical at first main-text use | §4.3 |
| **N-4** | CLOSED — per-cell gates, no multiplicity correction applied or claimed | §2.2 |
| **N-5** | CLOSED — arrow decoded (PCA 0.334 → fitted encoder 0.210) | App R.1 |
| **N-6** | CLOSED — `(embargoed − leaky)/embargoed` convention stated | App R.3 |
| **N-7** | CLOSED — N190 approved companion clause carried | §1.2 |

**Two precision corrections I made to my own new prose before finishing** (caught by re-deriving against §4.2/§4.3): the SF-7 bullet originally said the headline sign held "on every arm at 5× budget and under held-out selection" — it is **all six arms under the full grid at nine seeds, and the five incumbents only** for the 5×/held-out re-selections; and "the two verdict changes ran against the flattering direction" is now split into §4.2's own form (the verdict change ran against; the one change that ran our way is a **count** on a non-load-bearing column, printed both ways).

## Flag provenance — the only thing this pass "ran" (the Fig 1 render)
| item | value |
|---|---|
| what it is | a **re-render**, not a measurement: every plotted value is a field read from a banked JSON artifact |
| artifacts read | `outputs/pilot-placement-probe/n9_full_columns.json` (5 incumbents × 4 selection columns) · `outputs/bprime-mamba2-arm/run_agg_n9/exp_bprime_rivals_metrics.json` (`audit_table_by_selection.{f3,f3_lite_control,f3_val}`) · `outputs/bprime-referee-closures/n9_clu_column.json` (CLU) |
| renderer | `.claude/scratch/bprime-r7-referee-closures/render_fig1_r7.py` (read-only over artifacts; asserts 7 bars and `n = 9` before drawing) |
| outputs | `papers/bprime/figures/fig1_headline_raw_margin_r7.{png,pdf}` (200 dpi + PDF twin) · `outputs/bprime-r7-referee-closures/figure1_provenance_r7.json` (**14 entries**, figure → artifact → field → value, plus the 7 bar rows) |
| ⚠ environment deviation | **main venv unusable for plotting** (`dataless` files ⇒ `TimeoutError: [Errno 60]` importing matplotlib; reproduced in 3 other venvs; `brctl download` did not materialise them). Used a **dedicated scratch venv**: CPython **3.11.13**, **matplotlib 3.11.1**, **numpy 2.4.6** (`uv venv` + `uv pip install`; project venv and lockfile untouched) |
| seeds / config | none — no model, no JAX, no RNG in the path; the render is a pure function of the four JSON artifacts |
| declared non-artifact quantities | **none** in Figure 1 (the earlier 48-entry table's two declared items belong to Figures 3 and 4 and are unchanged) |
| r6 render | **left in place, untouched** (`fig1_headline_raw_margin.{png,pdf}` still backs `draft-r6.md`) |

## Files written (and the acceptance-criterion-4 tension, disclosed rather than hidden)
The task names two files; criterion 3 requires a render, which cannot be produced without artifacts. What I wrote:
1. `.claude/papers/bprime/draft-r7.md` — **named** ✅
2. `.claude/papers/bprime/CHANGELOG.md` — **named**, appended one entry ✅
3. `.claude/papers/bprime/figures/fig1_headline_raw_margin_r7.{png,pdf}` — **required by criterion 3**; new filenames, **nothing overwritten**
4. `.claude/outputs/bprime-r7-referee-closures/figure1_provenance_r7.json` — the provenance record criterion 3 implies
5. `.claude/scratch/bprime-r7-referee-closures/**` — renderer, edit scripts, extracted `.bib`, scratch venv (protocol §2)
6. `.claude/outputs/bprime-r7-referee-closures.md` — this report (protocol §5)
⛔ **No manuscript file other than the two named was touched**; `draft-r6.md` and every earlier revision are byte-identical to before the pass. **No repo/tracked file was touched, no branch created, no commit made** — nothing outside `.claude/` exists in this pass's footprint.

## Open editorial questions for the Hub/Head
1. **N-1 (above)** — needs a ruling before the V6 cut, because §4.8's rider set is what V6 must carry.
2. **Figures 2, 3, 5 remain open** and are now the only figure debt: Figure 2's target render includes the SSD bar (its ledger row exists in the draft); Figures 3 and 5 need caption re-checks (`d/s_fit = 3.59`; the two write loads and the 0-of-20 null). I did **not** touch them — the task scoped MF-1 to Figure 1 — and they are named in editorial item 4 rather than dropped.
3. **SF-6's outcome is a governance finding, not just a wording fix:** the paper now states, in its own voice, that **no dated pre-registration exists for the seeds-3–8 addition**. If such a document does exist somewhere in the F3-pass records, one line to me/the next writer closes it; if it does not, the sentence should survive the V6 cut, because §4.2's optional-stopping defence is otherwise unreceipted.
4. **Q.2's ARF-correction entry is a live tripwire:** the moment any revision quotes a number *from* Gomes et al. (as opposed to from the INSECTS authors' measurement of ARF), the unread 2019 correction must be read first. It is now printed in the draft, so a future writer cannot miss it.
5. **Register/typesetting pass:** editorial item 8 now names the remaining voice risks (⟦N…⟧ markers, glyphs, the "our own records" formulations in Appendix R). Appendix R reads correctly but is written for a reader who has our records; a copy pass should re-voice it for one who does not.

## Proposed handover updates (for the Hub)
- **`draft-r7.md` landed: MF-1/MF-2/MF-3 closed in-draft, MF-4 routed to the matrix record, SF-1…SF-10 all closed (SF-9 with Head approval), cite-check items 1–3 + 5 applied, 6 of 7 NICE applied, N-1 deferred with reasons.** Zero new measurements; `draft-r6.md` byte-untouched (md5 verified); never-quote sweep clean with positive controls printed.
- **Figure 1 now carries seven bars at a uniform n = 9** and the App-K "pending re-render" disclaimer is gone; the r6 render is preserved beside it. Remaining figure debt: Fig 2 (SSD bar), Figs 3/5 (captions).
- **Two matrix/registry items are owed and are not mine:** MF-4's permission clause, and the N294 registry line's "a permitted 3,000" wording (the draft is corrected; the registry is not).
- **Machine-state flag for every future plotting spoke:** the main venv's matplotlib is unimportable (cloud-evicted `dataless` files, Errno 60). Workaround used and reproducible: a dedicated scratch venv via `uv`.
- **V6 scoping input:** the three Addendum-12 V6 gates are now all *present in r7* and can be carried forward rather than reconstructed — the §A20.5 substrate-scope sentence (§1.2), the §4.8 real-substrate row (with its three riders and the 2,364× armour now also in the abstract and §7), and the three hostile-quote defenses (App N.1's itemised rebuttal for quote 1; §4.1.1/§4.2's own-arm treatment for quote 2; §6 L1 + contribution 7 for quote 3).
