# V1 — retire every hard-coded cross-reference before the structure moves again

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-27.** Charter basis: Add.97-amendment (the conversion must precede structural edits) + Add.98 §98.2 + Add.99.

**Agent:** `paper-writer` (Bash-capable — it must build).
**Writes:** `.claude/NIPSsubmission/v1-ttcl/pj_sub.tex` only. **Deliverable #1:** `BUILD-NOTE-R6.md` · **Report:** `.claude/outputs/v1-ref-conversion.md`

⛔⛔ **LAUNCH GATE — MECHANICAL, CHECK IT FIRST.** This pass converts headings, and two of them are headings the Head is about to reword; converting a heading mid-reword wastes both edits. **The gate is a string test on `pj_sub.tex`, not a promise:**

```
/usr/bin/grep -c 'Latch Payoff'                     .../v1-ttcl/pj_sub.tex   # expect 0
/usr/bin/grep -c 'Cashing Out the Certificate'      .../v1-ttcl/pj_sub.tex   # expect 0
```

**If either returns non-zero, STOP and report** — the Head's §97.5.2 register edits have not landed. ⛔ **Override, and the ONLY one:** if the line below reads `RULED: HEADINGS STAY AS WRITTEN`, the gate is discharged and you proceed with those strings present.

> **HEAD RULING ON THE TWO HEADINGS: ⭐ GATE DISCHARGED ON THE FACTS, 2026-08-28 — both strings are now 0 in the file.** The Head reworded them rather than exempting them: `Latch Payoff` → `Latch Transport`, and `Cashing Out the Certificate` → `The Certificate's Consequence`. ⇒ **the gate test passes mechanically; no override is needed or given.** ⚠ Run the test anyway — it is the check, not this sentence.

⚠ *Why this is mechanical: a task file that says "launch only after X" names no file and cannot fire — this program has watched a "GATED on X" spoke launch in parallel and go stale. A gate names a testable string or it is not a gate.*

⛔ No other spoke may be in flight on this file.

⛔ **Pin (RE-PINNED 2026-08-28 after the ć/T1 fix).** `pj_sub.tex` md5 = **`023febbf94a6a7a1e5c59a52c164bb89`** · `refs.bib` = **`4ce68c08109f990362751eb9f5132764`**. *(The earlier pin `de4559a3…` is SUPERSEDED.)* **The Head may edit again — the pin is a REPORT item, not an abort condition.** ⚠ **Locate every site by CONTENT, never by line number.**

⛔⛔ **THE PROBE'S BANKED BASELINE IS INVALID — REBUILD IT, DO NOT REUSE IT.** The first (gate-blocked) launch banked a before-baseline at `scratch/v1-ref-conversion/probe/` (`before.words`, 8 537 tokens, 16 pp) and recommended reusing it. **Since then the Advisor restored `\usepackage[T1]{fontenc}` and fixed a `refs.bib` glyph**, which changed hyphenation and line-breaking: the same instrument now measures **8 504 tokens over 15 pp**. ⇒ **Build your own before-baseline from the re-pinned file.** ⚠ *This is the shelf-life rule in the act: a provenance fact decays the moment the next pass touches its object — the probe's conversion-pattern proof and its instrument remain valid; only the baseline numbers died.*

---

## 1. Why this exists

The paper is about to be condensed (~8.3 pp → ~5 pp intent), which relocates blocks and renumbers appendices. **Every hard-coded pointer breaks silently under that edit.** A sibling paper proved both halves of this lesson: a theorist hard-coded `D.1…D.8` into an appendix that rendered as G — seven references mis-pointed; after label conversion, a later appendix deletion moved G→F and **nine references renumbered themselves for free**.

**Advisor-verified inventory at scoping (the closed worklist):**
⭐ **The blocked first launch verified this whole inventory against the file and it held (23/23, 5/5, 10/10, 7 labels, 4 refs, 0 table floats) — and corrected two arithmetic slips in the Advisor's prose below, which are fixed in place here: it is 11 unlabelled `\section`s (not 12 — `app:deriv` already has one), and W1's own target set is FIVE named subsections plus the `\subsubsection` = 6 sub-structure labels (not four). ⇒ W1 creates 17 labels, W2 a further 10 = 27, retiring 28 hard-coded pointers.** It also proved the W1+W2+W3 pattern renders identically on a copy and positive-controlled the acceptance instrument in both polarities, and it banked a ready-to-use site register with semantic keys — **read `outputs/v1-ref-conversion.md` §F2/§F4 first; do not re-derive them.** ⚠ Its §F3 hazard is binding: `tectonic` reports *"rerun seems needed, but stopping at 6 passes"* — **assert `0 undefined` from the after-log and grep the after-text for `??` explicitly; never infer resolution from exit 0.**

⚠⚠ **RE-MEASURED 2026-08-28 AFTER AN ADVISOR PASS — THE NUMBERS BELOW SUPERSEDE EVERY EARLIER COUNT IN THIS FILE AND IN `outputs/v1-ref-conversion.md`.** On the Head's instruction the Advisor already did **Appendix A's share of W1+W3**: five `\label`s added (`sec:certificates` · `sec:erasure` · `sec:rationing` · `sec:routing` · `sec:anytime`) and the five A.1–A.5 heading `\S<n>` converted to `Sec.~\ref{}`. ⛔ **Do not redo them; do not "restore" the `\S` form in those headings — `Sec.~\ref{}` there is deliberate and Head-approved.**

- **18 `\S<n>` occurrences remain** (was 23): `\S2` ×3 · `\S3` ×3 · `\S3.1` ×6 · `\S3.2` ×3 · `\S4.1` ×2 · `\S5` ×1. ⚠ **14 of the 18 now sit in Appendix G** (the derivations appendix) — that is the largest remaining block, and ⛔ **none is inside a heading any more.**
- **5 literal appendix strings, UNCHANGED**: `Appendix B.` · `Appendix B.2` · `App.~B.2` · `Appendix C.3` · `App.~C.3`. ⚠ One of them (`Appendix B.2`) sits **inside the A.5 heading**, which now also carries a `Sec.~\ref{}` — that heading is a mixed site: convert the literal string, ⛔ leave the `Sec.~\ref{}` alone.
- **10 hand-numbered starred subsections, UNCHANGED** — W2 is untouched and is still the whole of its own worklist item.
- **Existing labels: 12** (the original 7 + the Advisor's 5). **Existing `\ref`s: 9** (the original 4 + the Advisor's 5). ⛔ Leave all of them alone.
- **Unlabelled structure remaining: 11 `\section`s minus none done = still 11 for the appendix/section set MINUS the 5 now labelled ⇒ re-derive on the day** — ⚠ this line is the one count you must measure yourself rather than trust, because two passes have now edited it.

## 2. The worklist (closed set — nothing else)

**W1 — label the structure.** `\label{sec:…}` on the five numbered sections; `\label{sec:…}` on the four numbered subsections of §3/§4 that pointers name (3.1, 3.2, 4.1, 4.2, 4.3) and the §3.2.1 `\subsubsection`; `\label{app:…}` on the six appendix `\section`s that lack one (flag tables · certificate/BIBO · grids · negatives · verification · Markov kernels). Choose short semantic keys; ⛔ never encode today's number or letter in a key (`app:flags`, not `app:a`).

**W2 — un-hard-code the 10 starred subsections.** `\subsection*{A.1 <title>}` → `\subsection{<title>}` + `\label{…}` (drop the typed `A.1` prefix; `\appendix` auto-numbering reproduces it). ⚠ The class must actually render `A.1`-style numbers — verify on the first one before doing the other nine. ⛔ **Do not reword any heading text** — titles move verbatim, including any `\S` pointer inside them, which W3 then converts in place.

**W3 — convert the 23 `\S<n>`** → `\S\ref{sec:…}` (renders identically), and **the 5 literal appendix strings** → `Appendix~\ref{…}` / `App.~\ref{…}` matching each site's existing abbreviation style. `B.2` and `C.3` point at subsection level — their targets are W2's new labels.

**W4 — report-only (⛔ no edit):** `fig:bibo` and `fig:frontier` are labelled floats with **zero** inbound references — where a `\ref` belongs is content placement, the Head's call. Likewise the hand-titled "Table 1" (there is no `table` float in the file, `\begin{table}` = 0) — floatifying it is structural, out of scope; report it.

## 3. ⛔ The diff contract

Every hunk carries exactly one label: **`LABEL`** (W1) · **`STRUCTURE`** (W2) · **`CONVERT`** (W3). **`OTHER` = 0.** Forbidden however tempting: rewording, typo fixes, re-wrapping, terminology changes, any touch to a number, caption, or citation.

⭐ **The acceptance instrument is rendered-text identity:** `pdftotext` of the before and after builds must differ **only in whitespace/line-breaking — zero word-level differences** (the auto-numbers reproduce the typed prefixes; `\S\ref` reproduces `\S<n>`). If a word-level difference appears, that hunk is wrong — stop and find it. ⚠ Run both builds with the same engine (`tectonic`); line numbers from the submission class appear in both, so they cancel.

## 4. Deliverables

1. `BUILD-NOTE-R6.md` — the hunk register (every hunk labelled), the pdftotext identity check, before/after label+ref inventory.
2. **Sweeps, positive-controlled:** literal `\S[0-9]` = **0** remaining · literal `Appendix [A-Z]`/`App.~[A-Z]` = **0** remaining (⛔ excluding `Appendix~\ref` forms) · hand-typed `[A-C].[0-9]` heading prefixes = **0** · every `\ref` resolves (`0 undefined`).
3. **Build:** 0 errors · 0 undefined citations · 0 undefined references · pages and main-text split before/after (expect unchanged).
4. Anything noticed and not touched, listed.
5. Report: pin at start (md5), whether the Head had edited since scoping, and the full before→after md5 chain.

## 5. Acceptance criteria

- `OTHER` hunks = 0; pdftotext word-level identity holds.
- All sweeps in deliverable 2 pass with both control polarities (a seeded positive AND a known-present negative).
- `submission.tex`, `refs.bib`, `.claude/papers/v1-short/**` byte-untouched (md5 manifest printed).

## 6. ⚠ Grep hazards (carried, they have fired repeatedly)

⛔ `grep` here resolves to **ugrep 7.5.0** — silent false negatives/hangs on long `.tex` lines: use **`/usr/bin/grep`**; count with `grep -o … | wc -l`, never `grep -c` (lines ≠ occurrences). ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — per-file only. ⚠ zsh does not word-split; quote every glob. ⚠ Positive-control every negative, **both polarities**.

## DIAL DECLARATION
**Dials touched: NONE.** Mechanical cross-reference conversion; no experiment, no measured value, no claim wording.
