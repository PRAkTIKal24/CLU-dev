# v1-terms-swap — paper-writer report
Task + acceptance criterion: retire the receipt/price/cost/ledger register from `.claude/NIPSsubmission/v1-ttcl/pj_sub.tex` while preserving all registry-mandated content, and correct the "energy is exponential in distance" physics error to the rapidity form. Pin check must pass; no number moves; no claim widens or narrows.
Status: **done** — with **3 items needing a Head/Hub ruling** (all flagged, none silently absorbed).

## ⚠ RECONCILIATION LIST — THIS REPORT HAS ONE, AND IT NEEDS AN OWNER (protocol §5 corollary, in the first 10 lines)
Three sites in `pj_sub.tex` are **left as-is pending a ruling**, because acting on them unilaterally would have broken a stricter fence: (1) **the CM-7 "or even zero" half is missing at two BIBO-insufficiency sites** (§4 below — a claim change, forbidden in a vocabulary pass); (2) **`price`/`prices` survives at 2 sites** including contribution 2 (§5 — an unlisted class); (3) **`unpaid contraction`** at L76 (§5 — a false friend of the `paid access` class). All three are one-line fixes with wordings pre-drafted below. **Owner needed at the review that accepts this report.**

## DIAL DECLARATION (echoed before first result)
**Dials touched: NONE** — instrument/manuscript pass. Laundering control: n/a (no performance number produced). Falsifies the claim: n/a. Does NOT falsify: n/a. No experiment run, no config changed, no measured value moved.

## What I did
- **Pin check PASSED before writing.** md5 `de3585a6794add42c657600c9aa022db`, 382 lines — matched the task's pin exactly.
- **§0 sequencing check:** `govern the store` = 0, `φ-bytes ledgered` = 0, `ledgered` = 0 ⇒ the Head's §A20.5 insertion was **not yet in the file**. This pass ran first, exactly as sequenced; there was no approved-verbatim wording present to preserve or wrongly "fix".
- Applied **52 scripted, assertion-guarded, single-occurrence replacements** (`.claude/scratch/v1-terms-swap/swap.py` + `swap2.py`). Each asserts `count == 1` over the whole file; the file is written **only after all 52 assertions pass**. No global find-and-replace on any stem.
- Corrected the physics at **7 sites** (3 named in the task + 1 unnamed equivalent I found + 3 clarifications).
- Wrote **`BUILD-NOTE-R3.md`** (255 lines) into `.claude/NIPSsubmission/v1-ttcl/` — every changed site, before → after, tagged S1–S9/P.
- Built with `tectonic`; compared against a build of the pre-swap file.

## How I verified (commands + observed output)
- **Per-class tally, measured:** S1 `receipt` **12/12** · S2 `paid access` **5** (task said 4 — see below) · S3 title **1**, `metered` → 0 · S4 `savings` **3/3** · S5 `buys escape` **1/1** · S6 `cost`(energy) **6** · S7 `cost`(FLOP) **2** · S8 `priced`/`pricing` **10/10** · S9 `ledger` **19/19**.
- **Two-way numeric check (printed):** `total tokens before=641 after=646` · **BEFORE-only (LOST) = `{}` — empty ✅** · AFTER-only (GAINED) = `{'2': 5}`, all five being the exponent in newly written `e^{2|\zeta|}`/`e^{2|\zeta|}H` (the paper's own pre-existing expression, already at L73/L264; occurrences 4 → 9 = +5, matching exactly). **Zero measured values, ±, seeds or units moved.**
- **Residual sweep, positive-controlled** (`/usr/bin/grep -o … | wc -l`, never `grep -c`): `receipt 0 · ledger 0 · metered 0 · priced 0 · pricing 0 · savings 0 · buys 0 · paid 1`. Positive controls live: `certificate 32 · rationing 12 · wormhole 29` ⇒ the zeros are real, not a ugrep silent-false-negative.
- **DO-NOT-SWEEP verified byte-identical before → after:** `physics-free 4→4 · distribution-free 1→1 · budget 21→21 · Goldstone charge 3→3 · rationing 12→12 · account (abstract) 1→1`.
- **Forbidden phrases:** `cannot beat the box` 0 · `collapses past the box` 0 · `cannot reach` 0 (all already 0, kept at 0) · `exponentially in distance` and equivalents **0**.
- **Intensifiers:** `strictly 15→14` (one incidental removal, listed) · `clearly 1→1` · `conclusively 1→1` · `fully 5→5`. **None introduced.**
- **Build:** `tectonic -X compile` → **0 errors, 0 undefined references**, both before and after. 18 `hbox` badness warnings, **identical set before and after** (pre-existing B.1/C.1 table lines).

## Findings/results

### 1. The physics error was real, and there was a FOURTH site the task did not name
The task named the abstract, the §3.2 body and the Fig-1 caption. Sweeping by content (not line number) I found a fourth carrying the same wrong form: **§5 conclusion, *"A squeeze buys escape bounding but prices distance exponentially"***. Acceptance criterion §7 requires `exponentially in distance` **and equivalents** = 0, so it was corrected in the same restatement that carries its S2/S5/S6/S9 changes. All four now state energy growth **in rapidity ζ** with the **quadratic-in-excess-distance** bracket. ✅ The already-correct site (contribution 2, *"prices reach exponentially in rapidity"*) was **left untouched** as instructed — only its `ledger` clause changed.

### 2. MF-B fence held at every restated site — no regression to a reach-impossibility claim
Each of the 7 pricing-law sites still asserts (a) squeeze energy `≤ e^{2|ζ|}H` growing with ζ **and** (b) the wormhole's `ΔV` independent of `Δ`; the per-site quote table is BUILD-NOTE-R3 §3. The anti-*"cannot reach"* guard is preserved verbatim: *"…**not that it is fundamentally incapable of reaching them if provided sufficient energy**."* **Zero BLOCKED sites** — every site was restatable without changing what it asserts.

### 3. Free-ledger fence: the zero survived, and `free` was replaced by the literal value, never softened
`free` → **`exactly zero`** / **`zero energy`** (never "low"/"small"/"modest"). `ΔH = 0.0 exactly` is retained verbatim at §3.1 and B.2, together with the full CM-7 clause (energy-only sub-level test **admits** the `b=5.0` jump, it **escapes anyway**, coercive-component membership is the operative clause). B.2's `Energy sub-level test` row is byte-unchanged. `ΔV` vs `ΔH` was chosen **per site**; the B.1 table's `ΔV = V(b) − V(a)` is preserved.

### 4. ⚠ RULING NEEDED — task §7 asked for the zero at "three sites"; the pinned file had it at **two**
The zero was explicit at **two** sites (§3.1 L82, B.2 L290) — both preserved. The other two BIBO-insufficiency sites carry only the *bounded* half and **never carried the zero**: contribution 3 (L41) and the §5 conclusion (L148). CM-7's rider reads *"a bounded/even-FREE energy ledger isn't sufficient for BIBO"*, so the sharp half is missing there. ⛔ **I did not add it** — task §5 forbids widening a claim in a vocabulary pass, and "or even zero" makes both sentences assert strictly more. **This is a pre-existing gap, not one this pass created.** Proposed one-line fixes, ready to apply on a ruling:
- L41 → *"…a bounded — **or even exactly zero** — energy change is necessary but insufficient for BIBO stability without coercive-component screening."*
- L148 → *"…bounding the energy change — **even to exactly zero** — is an insufficient condition for BIBO stability; strict coercive-component screening is required."*

### 5. ⚠ RULING NEEDED — three out-of-scope residuals in the retired register
§1 rules that *"a class not listed is out of scope"*, so I did not sweep these. All appear **unmeasured** rather than **protected** (none is in §4's DO-NOT-SWEEP table):
- **`price` / `prices` — 6 occurrences, a different set from S8's 10.** S8 names `priced`/`pricing` and counts them at 10 = exactly `priced`(8) + `pricing`(2). Four `price(s)` were removed **incidentally** where a mandated S/P restatement rewrote the clause (L26, L59, L148 ×2 — all listed in the build note). **Two survive:** L32 *"the physical properties of the phase space strictly **price** every access mechanism"* and L40 *"…and **prices** reach exponentially in rapidity"* (⛔ the site the task explicitly says is physics-correct — only its wording is at issue, never its content). Proposed: → *"strictly **determine the energy required by** every access mechanism"* and → *"cures escape with bounded injection **at an energy growing exponentially in rapidity**."*
- **`unpaid`** (L76, *"broken by an **unpaid** contraction"*) — a false friend of the `paid access` class; same register. Proposed: *"**uncompensated** contraction"*.
- **`cashed out`** (2 sites: §3.2.1 heading, Fig-1 caption (b)) — S1 swapped the noun; the idiom is the same register but not a listed class. Proposed: *"The Certificate's Consequence: State Erasure vs. Transport"* and *"(b) The certificate in force"*.

### 6. ⚠ S2 count discrepancy: the file has **5** `paid access` body sites, not 4
The fifth is the appendix **section heading** `\section{The Paid-Access Certificate Table and BIBO Battery}` — plausibly counted with the titles at scoping. Same class, so swapping it is not an unlisted change. A literal swap gives the stutter *"Certified-Access Certificate Table"*; I set it to **`The Access-Certificate Table and BIBO Battery`** (keeps both concepts, no claim change, matches B.1's *"Certificate per Mechanism"*). **Reversible in one line if the Head prefers the literal form.**

### 7. `cost` disposition — 8 changed, **2 deliberate survivors**
Per-site table in BUILD-NOTE-R3 §1.3. The survivors: **L26** *"quantify the **costs** incurred against the model's fundamental stability guarantees"* (neither energy nor FLOP sense — the intro's generic framing, not the retired register) and **L200** *"**cost** checkpoints 300/1200/2100/3000"* (a **flag-provenance table descriptor** naming the harness's actual gate config; renaming it would break the reader's mapping back to the run config — protocol §5 / charter C-7).

### 8. Build: main-text split UNCHANGED, total +1 page
| | errors | undef. refs | pages | main text | appendix A | references |
|---|---|---|---|---|---|---|
| pre-swap | 0 | 0 | 13 | pp. 1–8 | p. 9 | p. 13 |
| **post-swap** | **0** | **0** | **14** | **pp. 1–8** | **p. 9** | pp. 13–14 |

**No main-text or appendix boundary moved.** The extra page is purely the last five reference entries spilling over, because the MF-B restatements run a few words longer. Toolchain: `tectonic` only (no `pdflatex`/`latexmk` on this machine) — the build is real, not pseudo-verified.

### 9. Charter compliance
**C-1** no audit-confession paragraph added or present ✅ · **C-2** designed-testbed results still labeled *verification* (§3 heading unchanged), learned-memory results still *evidence* ✅ · **C-5** no scale qualifier removed; the quadratic-in-excess-distance bracket is a consequence of the paper's own reach bracket, not a new generalization ✅ · **C-6** certificate fine print still adjacent to its claim (the `e^{2|ζ|}` matched-quadratic caveat at L73, the LTT exchangeability caveat at L134 — both byte-unchanged) ✅ · **CM-8 adjacency rider** `intra-CLU` remains adjacent to **all three** step-reduction figures (contribution 6, §4.1 body, C.3 column header `Intra-CLU Step-Reduction`) ✅ · **CM-7 must-travel fine print** intact at both latch-certificate and BIBO sites ✅.

## Scope proof (md5 manifest, final)
```
NIPSsubmission/v1-ttcl/pj_sub.tex        de3585a6794add42c657600c9aa022db -> 08d31733b5648ed6ab4a6bbc5dc07ed8  (EDITED, intended)
NIPSsubmission/v1-ttcl/submission.tex    caef2272f9dc96d349b46486563d24ee -> UNCHANGED ✅
papers/v1-short/CHANGELOG.md             a2bc48c0e0c2f3ceab1fa7cf34f655c8 -> UNCHANGED ✅
papers/v1-short/draft.log                86d8f80c82f90c4ab1fa0ce8a384cc0d -> UNCHANGED ✅
papers/v1-short/draft.md                 00d703d58a15c0cb77051a9c55674684 -> UNCHANGED ✅
papers/v1-short/draft.pdf                141f2c37ee8089c814932ce27e5f4fa4 -> UNCHANGED ✅
papers/v1-short/draft.tex                208797d113fa9d6efa6de67d05705ea6 -> UNCHANGED ✅
papers/v1-short/fig1_certificate.png     679647f639bfb8b3b7ecfa1333f43b69 -> UNCHANGED ✅
papers/v1-short/fig2_regime_map.png      b0cfbf53651ac187bacee0f977d93f1e -> UNCHANGED ✅
papers/v1-short/fig4_bibo.png            708b6fae2dadc755291b07c2962d102f -> UNCHANGED ✅
papers/v1-short/fig_frontier_clean.png   bcc5f32dcd85e01740638c6608f26320 -> UNCHANGED ✅
papers/v1-short/fig_regime_map.png       8b1dfddc54b4e48da0254a3bf35b9159 -> UNCHANGED ✅
papers/v1-short/paid_access_reach.png    fc372ae54cf39d7181f68837ad0e463b -> UNCHANGED ✅
```
Build outputs were written to `.claude/scratch/v1-terms-swap/build*/` — **no stray `.aux`/`.log`/`.pdf` artifacts in `v1-ttcl/`** (dir listing verified: only `pj_sub.tex` modified, only `BUILD-NOTE-R3.md` added).

## Git footprint
**None.** No tracked code touched; all work under `.claude/` (gitignored), per protocol §3 (research-only agent).

## Open questions / follow-ups / risks
1. **[Head] The CM-7 "or even zero" half at L41 and L148** — add or leave? Wordings drafted in §4. Risk if left: the two most prominent statements of the BIBO negative (contribution bullet + conclusion) are weaker than the appendix that proves it, and a reviewer reading only the main text never learns the jump was *free*.
2. **[Head] `price`/`prices` at L32 and L40** — sweep or keep? Wordings drafted in §5. Risk if left: two sentences in the intro and the contribution list still read in the register the pass exists to retire.
3. **[Head] `unpaid contraction` (L76) and `cashed out` (×2)** — same question, same drafted fixes.
4. **[Head] The appendix heading** `The Access-Certificate Table…` — confirm, or revert to the literal `The Certified-Access Certificate Table…`.
5. **[Hub] The title is now `[WORKING TITLE: Certified Access: Test-Time Compute on a Conservative Memory]`** per S3, verbatim as scoped. Charter C-10 keeps titles as placeholders until the end-of-program workshopping pass; flagging that this one is now *pinned by a Head ruling* rather than open for workshopping.
6. **Risk, low:** the paper is now **14 pages** vs 13. If a page cap binds this venue, the five spilled reference lines are the cheapest thing to reclaim — no content pruning needed.
7. **Note for the next editor:** the file's md5 is now `08d31733b5648ed6ab4a6bbc5dc07ed8`. The Head's §A20.5 insertion (`govern the store` / `φ-bytes ledgered`) is **still pending** and is unaffected by this pass.

## Proposed handover updates (for the Hub)
- `pj_sub.tex` R3 landed: economic register retired (`receipt`/`ledger`/`paid`/`metered`/`priced`/`pricing`/`savings`/`buys` all at **0**), and the **"energy exponential in distance" error is corrected to rapidity at all 4 sites** (task named 3; a 4th was found in the §5 conclusion). MF-B fence verified per-site; free-ledger `ΔH = 0.0` preserved. Build 0/0, main-text split unchanged, 13 → 14 pages.
- **Three Head rulings are outstanding and are the only thing between this file and "register fully retired":** CM-7's "or even zero" at 2 sites · `price`/`prices` at 2 sites · `unpaid`/`cashed out` at 3 sites. All are one-line, all have drafted wordings in `.claude/outputs/v1-terms-swap.md` §4–§5. **They need an owner at this review.**
- `pj_sub.tex` md5 for the next pass's pin: **`08d31733b5648ed6ab4a6bbc5dc07ed8`** (382 lines). §A20.5 still not inserted.
