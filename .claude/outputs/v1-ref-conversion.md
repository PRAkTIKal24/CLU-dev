# v1-ref-conversion — paper-writer report

Task + acceptance criterion: Convert every hard-coded cross-reference in `NIPSsubmission/v1-ttcl/pj_sub.tex` to `\label`/`\ref` (W1–W3), acceptance = pdftotext word-level identity + zero `OTHER` hunks.
**Status: BLOCKED — LAUNCH GATE FAILED. `pj_sub.tex` NOT EDITED (md5 unchanged, byte-identical to the scoping pin).**

## ⛔ FIRST 10 LINES — THE VERDICT AND ITS OWNER
- **Gate test result: BOTH strings PRESENT ⇒ STOP fired as written.** `Latch Payoff` = 1 (line 240, the A.5 heading) · `Cashing Out the Certificate` = 1 (line 108, the §3.2.1 heading). Required: 0 and 0.
- **Override NOT available:** the ruling line in the task file still reads `_(unset — blank means not ruled; …)_`. It does not read `RULED: HEADINGS STAY AS WRITTEN`. Swept the shorts charter, the C3 charter, and the C3 handover for a ruling recorded elsewhere — **no hit**; the three C3 docs are stale (Aug 14) relative to this task (Aug 28).
- **Root cause, confirmed independently of the gate: `pj_sub.tex` md5 is EXACTLY the scoping pin `de4559a36af659bada4a56ea05156db7`.** The Head has **not edited the file at all** since the Advisor scoped this pass (mtime Aug 27 21:48). So this is not "the register edits landed and the gate is stale" — **the §97.5.2 prose edits have not been made.**
- **⚠ RECONCILIATION LIST / OWNER NEEDED (protocol §5 corollary):** this pass is blocked on **three Head-owned prose edits** (Add.97 §97.5.2, located exactly): (1) A.5 heading `Latch Payoff` → `Latch Transport`; (2) the §3.2.1 heading `Cashing Out the Certificate: State Erasure vs. Transport`; (3) the table row `Latch payoff` at line 252. Per Add.99 §99.3 the Head also holds the seven §98.1 prose items. **The Hub must give these an owner and a due wave, or this pass stays parked.**
- **Unblock is one line either way:** the Head makes the three edits (gate then passes mechanically), **or** the Advisor writes `RULED: HEADINGS STAY AS WRITTEN` into the task file's ruling line. No third path — I did not exercise judgment about the headings.

## DIAL DECLARATION (echoed per protocol §7)
**Dials touched: NONE.** Mechanical cross-reference conversion; no experiment, no measured value, no claim wording. Laundering control: n/a. Falsifies: n/a. Does NOT falsify: n/a.

## What I did
- Read `AGENT_PROTOCOL.md`, the Positioning Charter (`philosophy-synthesis.md` §"Positioning Charter", C-1…C-10 as they read today — C-1 in its **REVERSED / no-audit-confession** form), confirmed `claims_matrix.md` present, then the task file.
- **Ran the launch gate first, before opening the file for edit.** It failed. Stopped all editing.
- Ran the full **read-only** before-state census (below) so the worklist is verified-current the moment the gate opens.
- Ran a **scratch-only feasibility probe** in `.claude/scratch/v1-ref-conversion/probe/` (copies only) that (a) proves the W1+W2+W3 conversion pattern renders identically and (b) validates the acceptance instrument with **both polarities**. `pj_sub.tex` was never a build input target and never written.
- Wrote **no `BUILD-NOTE-R6.md`** — see Open questions Q1.

## How I verified
```
$ /usr/bin/grep -c 'Latch Payoff'                pj_sub.tex   → 1     (gate expects 0)
$ /usr/bin/grep -c 'Cashing Out the Certificate' pj_sub.tex   → 1     (gate expects 0)
   line 108: \subsubsection{Cashing Out the Certificate: State Erasure vs. Transport}
   line 240: \subsection*{A.5 \S3.2.1 Latch Payoff and Appendix B.2 BIBO Battery Configuration}
   line 252: Latch payoff & 16 incoming states, ball radius $0.3<\rho=0.35$ \\   (third §97.5.2 site)
```
**md5 chain (pin → start → end): `de4559a3…` → `de4559a3…` → `de4559a3…` (UNCHANGED throughout).**

**Byte-untouched manifest (acceptance criterion §5, verified post-probe):**
| file | md5 | status |
|---|---|---|
| `v1-ttcl/pj_sub.tex` | `de4559a36af659bada4a56ea05156db7` | untouched (= scoping pin) |
| `v1-ttcl/submission.tex` | `caef2272f9dc96d349b46486563d24ee` | untouched |
| `v1-ttcl/refs.bib` | `58c75795e1fa8f5e46a74cbc2902e457` | untouched |
| `v1-ttcl/pj_sub.pdf` | `f50923b7854616eb90b0ca900708a1a6` | untouched |
| `.claude/papers/v1-short/**` (11 files) | manifest below | untouched — `find -newermt 2026-08-27` = **0 files** |

`papers/v1-short/`: `CHANGELOG.md a2bc48c0…` · `draft.md 00d703d5…` · `draft.tex 208797d1…` · `draft.pdf 141f2c37…` · `draft.log 86d8f80c…` · `fig1_certificate.png 679647f6…` · `fig2_regime_map.png b0cfbf53…` · `fig4_bibo.png 708b6fae…` · `fig_frontier_clean.png bcc5f32d…` · `fig_regime_map.png 8b1dfdd…` · `paid_access_reach.png fc372ae5…`

`git status --porcelain` = empty (all work under gitignored `.claude/`; **no code changes, no branch, no commits** — §3 n/a).

## Findings

### F1 — The scoped worklist is VERIFIED-CURRENT (nothing drifted; the file never moved)
Counts by `/usr/bin/grep -o … | wc -l` (never `grep -c`, per §6):
| item | task says | measured | ✓ |
|---|---|---|---|
| `\S<n>` occurrences | 23 | **23** | ✓ |
| breakdown | `\S2`×3 `\S3`×4 `\S3.1`×6 `\S3.2`×3 `\S3.2.1`×1 `\S4.1`×3 `\S4.2`×1 `\S4.3`×1 `\S5`×1 | **identical** | ✓ |
| literal appendix strings | 5 | **5** (`Appendix B.` L86 · `Appendix B.2` L240 · `App.~B.2` L277 · `Appendix C.3` L141 · `App.~C.3` L344) | ✓ |
| hand-numbered `\subsection*` | 10 | **10** (A.1–A.5, B.1, B.2, C.1–C.3) | ✓ |
| existing labels | 7 | **7** (`fig:reach` `fig:bibo` `fig:frontier` `app:deriv` `app:deriv:box` `app:deriv:squeeze` `app:deriv:gate`) | ✓ |
| existing `\ref`s | 4 | **4** (`app:deriv:box`×2, `app:deriv:squeeze`, `fig:reach`) | ✓ |
| `\begin{table}` floats | 0 | **0** | ✓ |
| existing `Appendix~\ref` forms (must be excluded from sweeps) | — | **3** | — |

**Two arithmetic slips in the task's §1 inventory prose** (worklist unaffected — W1/W2/W3 are unambiguous by content; recording them so the executing pass doesn't chase a phantom):
1. *"12 `\section`s … carry no label"* — there are **12 sections total**, of which **11 lack a label** (`app:deriv` already labels *Derivations for the Reach and Injection Certificates*). W1's own text is right: **6** unlabelled appendix sections + **5** unlabelled numbered sections = 11.
2. *"the **four** numbered subsections of §3/§4 that pointers name (3.1, 3.2, 4.1, 4.2, 4.3)"* — the parenthesis lists **five**. Measured unlabelled numbered sub-structures = **6** = those five + the §3.2.1 `\subsubsection`, exactly W1's stated target set. **W1 therefore creates 17 labels** (5 sec + 6 subsec/subsubsec + 6 app), and W2 a further 10 = **27 new labels**, retiring 23 + 5 = **28 hard-coded pointers**.

### F2 — ⭐ THE CONVERSION PATTERN IS PROVEN, AND SO IS THE ACCEPTANCE INSTRUMENT (scratch, both polarities)
The task flags *"the class must actually render `A.1`-style numbers — verify on the first one before doing the other nine."* **Verified now, on copies, so the gated pass starts from a known-good pattern.**

Baseline build (`tectonic 0.15.0`, the same engine both sides): **exit 0 · 0 errors · 0 undefined citations · 0 undefined references · 16 pp · 8 537 words of `pdftotext -layout`.**

Probe edit (one site, the exact W1+W2+W3 triple):
```latex
\section{Verification of the Certificate Stack}
\label{sec:certstack}                                        % W1  LABEL
...
\subsection{\S\ref{sec:certstack} Certificate Stack Configuration}   % W2 STRUCTURE + W3 CONVERT
\label{app:flags:certstack}
```
Result: **exit 0 · 0 errors · 0 undefined refs/cites · 16 pp (unchanged) · 8 537 words (unchanged)**.
- Renders `A.1  §3 Certificate Stack Configuration` — the `\appendix` auto-number **reproduces the typed `A.1`**, and `\S\ref{}` **reproduces `\S3`**, exactly as the task predicted.
- **Word-level identity: `diff` of whitespace-normalised token streams = 0 differences** (8 537 vs 8 537 tokens). The only rendered change is inter-word spacing after the auto-number, which the instrument correctly ignores.
- **Positive control (the other polarity):** seeding one word change (`Configuration`→`CONFIGX`) into the after-text makes the same instrument fire (`4824c4824`). **The instrument is not vacuously green.**
- **No hyperref `Token not allowed in a PDF string` warning** from `\S\ref{}` inside a heading — a real risk for W2/W3 (bookmarks are generated: `.out` = 4.74 KiB) that **did not materialise**. This was the probe's main unknown.

Reusable instrument (recommend the gated pass lift it verbatim):
```
pdftotext -layout X.pdf X.txt; tr -s '[:space:]' '\n' < X.txt | /usr/bin/grep -v '^$' > X.words
diff before.words after.words        # must be empty; positive-control it before trusting the empty
```
Artifacts banked: `.claude/scratch/v1-ref-conversion/probe/` (`pj_sub.pdf`/`before.txt`/`before.words` = the **before-baseline**, reusable so the gated pass need not rebuild it; `probe.*` = the pattern proof; `build_before.log`, `build_probe.log`).

### F3 — ⚠ A build-engine hazard the gated pass must not read past
Both builds emit `warning: internal consistency problem when checking if pj_sub.bbl changed` and **`warning: TeX rerun seems needed, but stopping at 6 passes`**. Refs still resolved (`0 undefined`) here — but this pass **adds 27 labels and 28 refs**, and a non-converged `tectonic` run is exactly how a `??` reaches a PDF while the log looks calm. **The gated pass must assert `0 undefined` from the after-log explicitly and grep the after-text for `??`, never infer resolution from exit 0.**

### F4 — Ready-to-execute site register (targets resolved by content; ⛔ line numbers are a snapshot and will shift)
Proposed semantic keys — ⛔ none encodes today's letter/number, per W1:
- **Sections:** `sec:intro` (Introduction) · `sec:setup` (Setup: A Conservative Memory…) · `sec:certificates` (Verification of the Certificate Stack) · `sec:learned` (Translating Mechanisms to Learned Memories) · `sec:position` (Position, Scope, and Horizon).
- **Sub-structures:** `sec:mechanisms` (3.1 Mechanism Certificates and Limitations) · `sec:access` (3.2 Differentiating Access Mechanisms) · `sec:erasure` (3.2.1 — ⚠ **key deliberately not derived from the heading text, which the Head is rewording**) · `sec:rationing` (4.1) · `sec:routing` (4.2) · `sec:anytime` (4.3).
- **Appendix sections (the 6 lacking labels):** `app:flags` · `app:certtable` · `app:grids` · `app:negatives` · `app:protocols` · `app:markov`. (⛔ `app:deriv` exists — leave it.)
- **W2's 10:** `app:flags:certstack` `app:flags:calibgate` `app:flags:routing` `app:flags:regime` `app:flags:latch` · `app:certtable:table` `app:certtable:bibo` · `app:grids:reach` `app:grids:routing` `app:grids:regime`.
  ⚠ **`app:flags:calibgate`, not `app:flags:gate`** — `app:deriv:gate` already exists (the hard-gate Jacobian) and a near-collision across two different "gates" is a future mis-point.
- **Pointer targets:** `Appendix B.` (L86) → `app:certtable` · `Appendix B.2`/`App.~B.2` → `app:certtable:bibo` · `Appendix C.3`/`App.~C.3` → `app:grids:regime` — matching each site's existing abbreviation style (`Appendix~\ref` vs `App.~\ref`), which the two figure captions (L277, L344) use in the `App.~` form.
- ⚠ **The A.5 heading is the single hardest hunk: it is simultaneously a W2 site, a W3 `\S` site, a W3 literal-appendix site, and a Head reword site** — `\subsection*{A.5 \S3.2.1 Latch Payoff and Appendix B.2 BIBO Battery Configuration}` contains `A.5` + `\S3.2.1` + `Appendix B.2` + the word the Head is changing. **This is precisely why the gate exists.** Its `Appendix B.2` also points at a W2 target, so W2 must land before W3 resolves it.

### F5 — W4 (report-only, ⛔ untouched as instructed)
- **`fig:bibo` and `fig:frontier` have ZERO inbound `\ref`s** — confirmed: the only `\ref`s are `app:deriv:box`×2, `app:deriv:squeeze`×1, `fig:reach`×1. Two of the three labelled floats are unpointed. **Where a `\ref` belongs is content placement — the Head's call.**
- **"Table 1" is a hand-typed heading string** (`\subsection*{B.1 Table 1: Certificate per Mechanism}`) with **`\begin{table}` = 0** in the file. Floatifying it is structural, out of scope. ⚠ Note the interaction: after W2 this heading renders **"B.1 Table 1: Certificate per Mechanism"** — a section number and a float-style number in one title, which reads oddly and will read worse after the condensation renumbers B. **Flagged, not touched** (fixing it would be an `OTHER` hunk and a reword).

### F6 — Noticed and not touched (deliverable 4)
- 2 overfull + 3 underfull `\hbox` warnings (L262–271, L289–303 overfull by 9.64pt/1.92pt; L309–323, L327–340 underfull) and 7 in the `.bbl`. Cosmetic; out of scope; will move under condensation anyway.
- The main-text/appendix split at present: **16 pp total**, `\appendix` at L167 of 4xx. Against the ~5 pp intent this is the Add.97 §97.5 page question — untouched, and unaffected by this pass (identity acceptance ⇒ pages must not move).
- `.DS_Store` and stale `submission.{aux,log,out,pdf}` (Aug 26) sit in the submission dir alongside the live `pj_sub.*`. Housekeeping only; not mine to remove.

## Grep hazards actually encountered (carrying them forward)
- ⚠ **`grep -o` with a context window UNDER-COUNTS overlapping matches.** Dumping the `\S` sites with `-o '.\{0,55\}\\S[0-9]…'` displayed 22 windows for **23** true occurrences (lines 386 and 411 each carry multiple hits inside one window). **The authoritative count is the bare-pattern `grep -o … | wc -l`; never count from a context dump.** This is a new instance of the §6 family and is exactly how an inventory silently drifts by one.
- ⚠ **Bash `cd` persists across tool calls in this harness.** A `cd` into the submission dir made a later repo-root-relative path report "No such file or directory" for the task file — a false "the file is gone". Absolute paths, or re-`cd` every call.
- ✅ `/usr/bin/grep` used throughout (never bare `grep`/ugrep); per-file only over `.claude/` (never directory-level); every negative sweep positive-controlled.

## Open questions / follow-ups / risks
1. **Q1 — I did NOT write `BUILD-NOTE-R6.md`, deliberately. Confirm.** No hunk landed on `pj_sub.tex`, so an R6 note would be a hunk register with zero hunks — and it would **consume the R6 slot** that the real conversion pass (R1–R5 already exist) should own. My call: the R6 slot stays reserved; this report is the blocked-pass record. **Reverse it if the Hub wants a numbered artifact per launch rather than per landing.**
2. **Q2 — Who makes the three §97.5.2 edits, and by when?** They are the sole blocker. See the reconciliation list at the top.
3. **Q3 — Or does the Head simply rule the headings stay?** If so the Advisor writes `RULED: HEADINGS STAY AS WRITTEN` into the task file's ruling line and this pass is immediately executable — F2 shows the pattern and the instrument are already proven, so it becomes short.
4. **Risk — the pass is still correctly ordered before the condensation, and that ordering is now the binding constraint.** Every wave this sits parked is a wave in which someone may start the ~8.3→~5 pp condensation first, which is the exact edit Add.97-amendment says must come second. **If the Head's prose edits are going to be slow, the Hub should consider Q3 rather than letting the condensation overtake the conversion.**
5. **Risk — `fig:bibo`/`fig:frontier` stay unpointed through the condensation.** Unpointed floats are the ones that get orphaned when blocks relocate. W4 says report-only, so it is reported: **two of three labelled floats have no inbound reference.**

## Proposed handover updates (for the Hub)
- **`tasks/v1-ref-conversion.md`: the mechanical gate WORKED AS DESIGNED and should be recorded as such.** It fired on the first launch, named the exact blocker, and cost zero wasted edits — the Add.99-amendment lesson ("a gate names a testable string, or it is not a gate") has now paid out once, in production. Recommend the charter records the payout next to the rule.
- **Charter (shorts) — add to the Add.99 §99.3 entry:** ref-conversion **launched 2026-08-28 and returned BLOCKED-BY-GATE**; `pj_sub.tex` byte-identical to the `de4559a3…` scoping pin ⇒ **the §97.5.2 register edits and the seven §98.1 prose items are all still outstanding on the Head's side**; the file has been frozen since Aug 27 21:48.
- **Two inventory arithmetic slips in the task's §1 prose are corrected in F1** (11 unlabelled sections not 12; five named subsections not four ⇒ 6 sub-structure labels). The **worklist itself is unchanged and verified-current**; only the prose counts were off. Fold into the charter so the next scoping doesn't re-derive them.
- **Bank F2 as a reusable asset:** the before-baseline (`before.words`, 8 537 tokens, 16 pp) and the validated identity instrument live in `.claude/scratch/v1-ref-conversion/probe/`. **The gated pass should reuse, not rebuild, the baseline** — same engine, same inputs, md5-pinned.
- **Add the two new grep hazards** (context-window under-count; persistent `cd`) to the §6 carried-hazard list.
