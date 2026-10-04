# FINDINGS — pw2-draft-numbers (PW-2 item 5: numbers out of the main text, BOTH drafts)

Session `pw2-draft-numbers`, branch `pw2/draft-numbers` (from `pw2/draft-methods` 2e52807), batmac. Start 2026-09-25T16:05:50Z.
Snapshot `snap-start/` (sha256 in `snap-start/SHA256`; equal to pw2-draft-methods' `INSTALLED-SHA256.txt`). Final stage `stage-20260925T162725Z`; install verified in `verify-live-20260925T162932Z`.

## Status: item 5 DONE in both drafts, installed. The Head-held paragraphs were left byte-identical and still carry numbers (see §4).

| brief point | result |
|---|---|
| No per-seed or table-readable number left in the main text (title to end of Discussion; abstract included; captions and tables excluded) | **Done, outside the Head-held paragraphs.** Every remaining number is listed with its class in `NUMBERS-LEFT.md`; 0 UNCLASSIFIED in either draft. Outside Head-held text, the only digit tokens left are identifiers, protocol constants, one definition and two rounded values ("about 0.6", "just under 1.4") |
| Every removed number is in a table or caption cited by `Tab.~\ref` / `Fig.~\ref`, or keeps a `%` pointer | **Done.** 41 edits, 87 before→after pairs (`NUMBERS-MOVED.md`). Every edit leaves a `% PW2-5 moved -> <where>: <numbers>; SRC: <files>` comment beside its sentence, identical in both drafts. There are 97 moved-number rows: 71 point to a table, caption or appendix that already carried the number; 19 were ADDED to the nearest existing caption or table (§2); 7 survive only as `%` pointers (§3) |
| Every rounded number follows from a record | **Done.** `ROUNDED.md`: 51 phrases; each recorded value is found verbatim in its file and each rule holds (0 FAIL). Positive control: a planted "about seven" against a record of 6.9 → FAIL. No phrase rounds up past its record: "nearly X" is used wherever X exceeds the record (nearly five = 4.989; nearly a gigabyte = 0.967 GB; nearly two thirds = 65.39%) |
| Abstract: at most five numbers; print the before and after lists | **Done: 5 quantities** (2 of them Head-held); 0 digit tokens (35 before). Lists below |
| Head-held paragraphs byte-identical; their numbers counted | **Done.** `headheld.py` PASS 7/7 on both installed drafts against pw2-draft-methods' `HEAD-HELD.md`, which is unchanged and copied here. Counts in §4 |
| Identifiers that stay are listed with a justification | **Done** (`NUMBERS-LEFT.md`, class column; summary in §5) |
| Only the numeric phrase and the minimal words around it changed | Done. No sentence was rewritten for style. Tells, self-narration and coinages were left for item 1 (e.g. "which we print" stays) |
| Compile checks (COMMON) | Both drafts: exit 0; 0 undefined citations; 0 undefined references; 0 errors; 0 multiply defined; **0 overfull**. Underfull: v2 21→17, style 17→21, all `\vbox … while \output is active`, i.e. page breaks. 56 bibitems; no Rerun; 43 / 44 pages (unchanged) |
| Style diff-check (numbers multiset, placeholders, TODO, artefact paths, SRC strings and order, headings, labels, floats, preamble) | **ALL DIFFS EMPTY** (`STYLE-DIFF-CHECK.md`). **Number multiset between the drafts: 3,381 tokens (1,761 distinct) in each, lost ∅, added ∅.** Before this pass it was 3,631 each; every float is byte-identical between the drafts |
| Provenance findings identical or improved | **Identical**, line numbers masked (v2: 3 unsourced lines, 9 not-found tokens; style: 3 and 10, all pre-existing). Checked tokens went 2,597 → 2,355. The pre-existing "106" finding moved with its number into Tab. cmazet-sym |
| `\TODO` / `\PLACEHOLDER` | Unchanged against the snapshot: 4 `\TODO`, 2 `\PLACEHOLDER` per draft, byte-identical arguments |
| Figure / table / section numbering | 77 of 77 labels, 0 renumbered (`LABELS.txt`). Some floats moved by a page; the Introduction now starts on p. 1 (the abstract is shorter); Discussion still starts p. 17, main text still ends p. 19, appendices p. 25 |
| Installed byte-identically into both live folders | `.tex` + PDF in `~/clu/pw2/live/{v2,style}`, cmp-equal to the stage. The lock guard was clean and the live files were cmp-equal to the snapshot just before install. sha256 in `INSTALLED-SHA256.txt`. A fresh build from a copy of the live folders gives identical `\newlabel`s and the same metrics |
| Bib | `refs_clu_iclr.bib` b2cf715f… in both folders at start and end; never written |

## 1. Before → after, per section (digit tokens and spelled cardinals, main text; `COUNTS.md`)

| draft | section | digit before | digit after | of which Head-held | word before | word after |
|---|---|---|---|---|---|---|
| DRAFT-v2 | Abstract | 35 | 0 | 0 | 14 | 8 |
| DRAFT-v2 | Introduction | 30 | 4 | 0 | 28 | 32 |
| DRAFT-v2 | Methods | 12 | 10 | 0 | 5 | 5 |
| DRAFT-v2 | Experiments | 56 | 34 | 31 | 20 | 19 |
| DRAFT-v2 | Results | 389 | 114 | 38 | 74 | 88 |
| DRAFT-v2 | Discussion | 64 | 33 | 27 | 32 | 35 |
| DRAFT-v2 | **total** | **586** | **195** | **96** | **173** | **187** |
| DRAFT-v2-style | total | 586 | 195 | 96 (69 Head-held + 27 in the limitations list, held for lock-step) | 170 | 185 |

The word count rises because rounded values are written as words ("about seven", "nearly five"). Of v2's 195 remaining digit tokens, 96 sit in Head-held text. The other 99 are: seed indices, rung names ($n=2 \ldots 6$), the design names $K^\star=32$, $K=5$, 6-atom, 10-atom; "95%", "$10\times$", "random 1%", row 3 / row 4, list enumerators (1)–(4), math notation, a citation year, tokeniser artefacts inside float64/float32, protocol sizes ("64 checks", "256 requests", "(3 seeds"), the declared margin 1.0, the reference definition (524,288 B / 131,072 float32), "coverage share is 1 by construction", and two rounded values ("about 0.6", "just under 1.4").

### Abstract, before (v2, 49 numeral tokens, in order)
three · one · three · 32 · 5 · 2 · three · 32.350 · 0.815 · six · 67.177 · 1.320 · three · four · 0.0000 · four · 131,072 · ten · 7.03 · 109 · 7.49 · 8.60 · 5.02 · three · 95 · 3.73 · 6.54 · 10.468 · 21.936 · 5.133 · 12.550 · 1 · 5.38e-07 · 0 · 0.97 · 2,367 · 5,913 · 4.0768 · 4.0897 · 524,288 · 6.13 · 6.43 · 6.30 · 0.11 · 1 · six · one · zero · zero

### Abstract, after (both drafts): 5 quantities
1. "three things at once" — **Head-held** (opening sentence)
2. "all three properties" — count (the paper's frame)
3. "about **seven** more distinct valid novel tuples per held-out combination than the strongest, an MLP" — rounded; record +7.034 (`w20-shapes-data-matched/SHAPES-MATCHED.md`)
4. "survivors fall by about **six** tuples per combination" — rounded; record 6.13 / 6.43 / 6.30 (`w20-shapes-data-matched/FINDINGS.md`)
5. "the six-seed symmetric-input condition table" — **Head-held** (the maze `\TODO`)

Not counted: "zero-dynamics limit" and "zero-step arm" (names of the limit and of the arm) and the pronoun "one" ("an established one"). The abstract's "the one measured clue" became "the measured clue" (A4), so it no longer carries a numeral. The Head may prefer to restore it; item 1 (tell 6) names this coinage anyway. The after-abstract keeps its claims without numbers: "the finest partition on a validation plateau that the seeds do not resolve"; "every landscape control collapsing"; "each also trained far beyond its selected optimiser steps"; "to floating-point roundoff"; "as large as its fit data and its refit is slower than retraining a baseline"; "several times the earlier design's byte reference"; "far more than after a random item deletion".

## 2. Tables and captions that received numbers (ADDED; same-line `% SRC`, identical in both drafts)
- **Tab. t7** caption: the n-CLU's row-3 / row-4 means with seed SD, 32.350 ± 0.815 and 67.177 ± 1.320. Previously these appeared only in the abstract. ± is the sample SD: checked by recomputing from the per-seed values in `NR6-T7.md` (0.8148, 1.3196).
- **Fig. shapes-per-model** caption: validity counts at seed 0 (5,688 requests; n-CLU 5,177, MLP 4,361, LSTM 3,703, Transformer 2,786, continuous MLP 3,110, table 897) and the factor-miss counts (497 of 511; scale 329, orientation 125, shape 71).
- **Tab. ladder** caption, a new *Notes* block:
  - earlier-set hyperparameters (117,878–119,858 parameters, 40 epochs, batch 256, lr 0.003);
  - control collapse (0.0000 on 13 of 15 legs; 0.0083 at $n=3$ seeds 0 and 1);
  - probe geometry (8,640 rows; 60 × 36 against 120 × 72; row counts 300,000 / 235,200 / 213,600 / 200,640; $r=-0.96$ to $-0.98$);
  - +2.46 lower bound; the LSTM spot check 20.55 → 21.85;
  - n-CLU-content over-rung means (+10.28, +10.75, +9.97, +17.53) and +1.98;
  - the seed-to-seed MLP spread (2,560 steps, −2.15 to +4.13, 12 of 15);
  - the coverage-control validation leg and TEST identity check (22.767 …, 23.478 …, +1.200 / +0.567 / +0.367, 1.00, 0.000, 1.000000, 3,600 of 3,600).
- **Tab. del** caption, a new *Notes* block: 117,964 fit rows; the seed-0 red-sphere breakdown (20 / −32.1, 36 / −1.47, 53 / +0.51); the carry probe (0.610083 / 0.614277, 720 / 4,968, 6.194e-08 / 6.163e-08, 1,015,808 of 1,060,864, 0 rows); the $K=5$ targeting (rank/width 0.156, cos_min 0.1999 against 0.9976).
- **Tab. hole** caption: 0.0000 s, the 1,322-byte table and its 1,322 pair slots, the 8-byte bitmap at 1.0000 with 0 rows, and 10.48 s.
- **Tab. ablations** caption: 712,072–719,352 B (1.358–1.372×), and the item-memory eviction (1.0000; −2.0899 / −0.9101 / −6.7416; ±1.0 bar; hole +0.0000).
- **Tab. gentable-kstar** caption: the S2 margin 0.020.
- **Tab. cmazet-sym**:
  - the words + observation n-CLU row gains the seed-5 note (score 0; 0.04 margin; 106 of 317 fit rows);
  - a new row "value hold-out, n-CLU-REL (TEST)": 0.554 against 0.797, six-seed means, behind on every seed. This is the brief's own example, now "stays well behind the best matched baseline (Tab.~\ref{tab:cmazet-sym})".

The provenance checker confirms that every added number occurs in its cited source: the not-found count is unchanged.

## 3. Numbers that now survive only as `%` pointers (no float carries them; for the Head)
- The literature screen (Sec. 1.1): 4,050; 3,038; 759; 254; 21 queries; about 100; 10 of 3,038 (0.3%); 0 of 3,038; 4 of 6. The prose keeps "about four thousand", "ten … most of them false positives", "four of the six claim atoms".
- Derived values: the TEST rise +4.989 (+5.758 / +3.467 / +5.742; derivable from Tab. ladder) and the n-CLU-REL lead over +rel per seed (derivable from Tab. cmazet-sym). The shares of the margin recovered, 0.97 and 1.00, are not derivable from any table.
- Sec. 5.3: the zero-step arm's scores (row 4: 63.612 / 64.327 / 67.653; row 3: 29.717 / 31.617 / 31.450). They are numerically the composition-only row of Tab. ablations, but no table states the identity. Also the median displacements 0.009 / 0.39 and the upper byte bound 84,216. App. physics prose has the rest.
- "scores 0.000 on two tuples" is now "scores nothing"; the per-tuple replay stays in the existing SRC note.

## 4. Head-held paragraphs (byte-identical; numbers counted, not touched)
| paragraph | v2 digit / word | style digit / word |
|---|---|---|
| abstract, opening sentence | 0 / 1 | 0 / 1 |
| abstract, maze `\TODO` | 0 / 1 | 0 / 1 |
| Introduction, first paragraph | 0 / 2 | 0 / 2 |
| Methods, first paragraph (the Head's item-3 text) | 0 / 0 | 0 / 0 |
| Experiments, first paragraph (the $K^\star$ rule and plateau: per-$K$ seed SDs 0.0211 … 0.0259; per-seed footprints 2,094,130 / 2,080,090 / 2,085,810 B; 7,060 / 12,203 / 30,507 B) | 31 / 4 | 31 / 5 |
| Results, first paragraph (per-seed margins +7.486 / +8.596 / +5.018, mean +7.034, +3.73, wins 76 / 84 / 69, losses 22 / 16 / 13, 20,480 steps, +1.47, +6.018 …, LSTM / continuous MLP / transformer ranges, 130,433–131,891, 131,072, 10%) | 38 / 12 | 38 / 12 |
| Discussion, first paragraph (v2: the whole limitations list: 6.13 / 6.43 / 6.30, 32.1, 0.11, 0.97 GB, 2,367–5,913 s, 4.0768–4.0897, 3.9942, $K=24$, $K=40$, 0.991, 0.0648, 524,288, 1.47, …) | 27 / 13 | 0 / 1 (style's signpost sentence) |

**Choice logged (lock-step).** In the style twin the limitations list is not Head-held: its Discussion opens with its own signpost. The list is identical to v2's, however, and editing only the style copy would break the number multiset between the drafts, which COMMON requires to be equal. I held it byte-identical in both drafts. Its 27 digit tokens are classed "lock-step" in `NUMBERS-LEFT.md`.

**For the Head:** the three Head-held paragraphs carry 96 digit tokens that item 5 would otherwise remove. Most are per-seed values already printed elsewhere:
- Results first paragraph → Tab. t7;
- Experiments first paragraph → Tab. kcurve3 / App. kcurve;
- limitations → Tab. del, Tab. ledger-kstar, App. kcurve.

Three numbers live **only** in the Head-held Results paragraph: +7.034 (mean margin), +6.544 (best fit of any length) and the +3.73 lower bound. The abstract and Contribution 1 pointers now cite them there.

## 5. Identifiers and protocol numbers that stay, with justification (summary; every row is in `NUMBERS-LEFT.md`)
- **Design names:**
  - $K^\star=32$ (Contribution 1; Sec. 4.1 figure paragraph);
  - $K=5$ (the named endpoint);
  - 6-atom / 10-atom (the base and relational designs);
  - "random 1%" (the item-deletion scope name: Experiments definition, Sec. 4.3, Sec. 4.5);
  - "$10\times$ training steps" (the schedule name; "a transformer at 80,000 steps" became "$10\times$ training steps", PW-1 vocabulary).
- **Indices:** seed numbers (0–5); rung names $n=2\ldots6$, "2-way → 6-way", "rungs 3–6"; row 3 / row 4; the outcome code "(legal, move south, pocket 2)"; the enumerators (1)–(4); $5\times5$ (panel crop); $s_{t+1}$, $(\cdot)^{-1}$, $\lambda=0$; the bucket name $h/h^\star\in[0.5,2]$; the year 2024 in the citation phrase.
- **Protocol:** "95%" (interval level); "64 checks", "256 requests" (assertion sizes; no table carries them); "(3 seeds"; "3-seed mean"; the declared 1.0-tuple margin (defined once in Methods; elsewhere "the pre-declared margin").
- **Definition:** "The 524,288-byte figure … is a reference column … about 131,072 float32 parameters at it" (Methods). This is the one place the reference is defined; everywhere else says "the reference". Also "the coverage share is 1 by construction".
- **Counts in words (high-level):** "three properties", "three / six seeds", "three wins, two ties, one loss", "two of three seeds", "eight other arms", "four facts", "five words", "four controls", and similar.

## 6. Choices logged
1. Rounded values are written as words: "about seven", as in the Summary's existing "about seven tuples per held-out combination" and in the plan's tell-4 example. The Head's example uses "about 7".
2. Where the rounded value would exceed the record, the text says "nearly": five, a gigabyte, two thirds. Where rounding down would flatter an error metric, no rounded number is written; the settle half-life became "well within its declared bar (Tab.~\ref{tab:crr})" rather than "within about ten percent", because the median factor is 1.105.
3. Contribution 1 names "an MLP" as the strongest baseline, following the plan's corrected example.
4. Tab. del's cost rows and Tab. rivals-del support the relative claim "the book's refit is slower than any baseline's retrain": 2,367 s is the fastest refit, 2,064.5 s the slowest retrain. This claim replaces the timing ranges in the abstract, Contribution 2 and Sec. 4.4.
5. Caption growth. Tab. ladder and Tab. del now carry *Notes* blocks. Page count is unchanged at 43 / 44, and the main text still ends on p. 19. No text was cut for length; the nine-page cut is the Head's.

## What did NOT run
- No other PW-2 item (1, 4) was touched. The tells were deliberately left alone: self-narration ("which we print"), "not X but Y", coinages ("leaving note", "Thickness"), triads.
- No Head-held paragraph was edited. The 96 digit tokens in v2's Head-held Experiments, Results and Discussion paragraphs remain (§4). No proposed replacement text was written for them, since the Head writes those paragraphs.
- The AI-use, ethics and reproducibility statements were not scanned or edited. They follow Discussion, and the brief's scope is title through the end of Discussion. The reproducibility statement keeps "1,308 of the 2,293".
- Appendix prose was not edited, except that captions and tables in the appendix received numbers (§2).
- No page-by-page visual review of the PDFs. Checks are text-level: pypdf extraction, the rendered word diff (`WORDDIFF-*.txt`), aux labels and build logs. The rendered diff was read for swallowed text; none remains. An early staging (`try1/`) had two `%` pointer comments that would have commented out the rest of their line. The stage check caught it before any build reached the live folders, and `apply.py` now refuses such an edit.
- No compute, fits or GPU. The bib, `outputs/w17-paper`, `outputs/pw1-*`, `outputs/w*/`, TABLES.md, the registry and the claims matrix were not written.

## Deliverables in this directory
`NUMBERS-LEFT.md`, `NUMBERS-MOVED.md`, `ROUNDED.md`, `COUNTS.md`, `STYLE-DIFF-CHECK.md`, `prov-{before,after}-*.txt`, `COMPILE.txt`, `LABELS.txt`, `WORDDIFF-{v2,style}.txt`, `INSTALLED-SHA256.txt`, `HEAD-HELD.md` (unchanged copy), `controls/` (planted number → UNCLASSIFIED and style-diff exit 1), `snap-start/`, `stage-*/`, `verify-live-*/`, `LOG.md`.

## Branch and bundle
Tooling commit 83596f2 on `pw2/draft-numbers` (`pw2_draft_numbers/`). Bundle `pw2-draft-numbers.bundle`, sha256 ddfa0c47…9eaf. `git bundle verify` is okay, the history is complete, and the head is refs/heads/pw2/draft-numbers = 83596f2. Outputs are not committed. At the end, the bib is b2cf715f… in both folders and the live files still match `INSTALLED-SHA256.txt`.
