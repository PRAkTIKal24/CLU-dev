# V1 — replot pass: close a claims regression in the headline figure, retire two vocabularies, drop one comparison

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-27.**

**Agent:** `results-analyst` · **Writes: PNGs under `.claude/NIPSsubmission/v1-ttcl/figs/` and its report only.**
**Report:** `.claude/outputs/v1-replot-pass.md`

⛔⛔ **THE HEAD IS EDITING `pj_sub.tex` RIGHT NOW. YOU MAY NOT TOUCH ANY `.tex` FILE.** Not the caption, not the `\includegraphics` line, nothing. If a caption needs changing, **list the exact edit in your report for the Head to make.** A concurrent write would clobber live work.

---

## 1. Why this pass exists — the lead item is a claims regression, not styling

⛔⛔ **`fig1_certificate.png` panel (a) carries the annotation "squeeze collapses past the box."** That is the **MF-B falsified framing.** Per `papers/v1-short/CHANGELOG.md` v0.4 it was removed from *"abstract, contributions 2, §2, §3.2 heading/table/**caption**"* and replaced by the pricing law — **and it was, in the text.** Nobody fixed the PNG.

**The figure now contradicts its own caption.** Advisor-verified on disk:
- caption, `pj_sub.tex:64`: *"…priced out past the swept rapidity budget $\zeta\le2.0$. Its theoretical reach spans $[L, L+p_0\sinh\zeta/M_0]$…"*
- body, `pj_sub.tex:102`: *"…priced out of the swept rapidity budget, **not that it is fundamentally incapable of reaching them** if provided sufficient energy."*
- `collapses past` in the text: **0 occurrences.**

⭐ **This is the single most quotable defect available to a referee** — a headline figure asserting the claim its own caption retracts. Fixing it is the first job here.

Two further vocabularies are also baked into pixels: **MF-A's retired *"no-physics router"*** (the paper now says `state-replacing` ×13 and *"no-physics router"* ×0), and the **receipt/ledger register the Head is removing from the prose**.

---

## 2. The generator situation — two of three are covered by a LIVE generator

⭐ **`.claude/scratch/v1-revision-2/make_figs.py` is the live generator for `fig1_certificate.png` and `fig4_bibo.png`** (it also emits `fig2_regime_map.png`). **Advisor-verified as current, not stale** — every string in it reproduces the shipped PNGs exactly: `l.48` `"squeeze collapses\npast the box"` · `l.53` the panel-(a) title · `l.81` the panel-(b) title · `l.168` the Fig-4 title · `l.171` the red annotation. It reads the JSON below.

⛔⛔ **TWO TRAPS IN THAT FILE — read before you run it.**
1. **`OUT` is hard-coded to `/Users/user/Desktop/CHLU/.claude/papers/v1-short`** (l.15) — the **canonical** paper directory, which is byte-untouched by acceptance criterion. ⛔ **Running it unmodified writes into the wrong tree.** Point `OUT` at a scratch dir, then copy only the approved PNGs into `NIPSsubmission/v1-ttcl/figs/`.
2. **It also emits `fig2_regime_map.png`**, which a prior pass removed from the paper. ⛔ **Do not copy fig2 into `figs/`**; leave the shipped one untouched.

⚠ **`fig_frontier_clean.png` has NO generator.** `.claude/scratch/regime-remap-2000ep/analyze.py` emits **`fig_frontier.png`** — a *different, earlier* figure with no Hopfield band and no titles. ⛔ **Do not confuse the two.** The `_clean` variant must be **reconstructed** from data.

**Data sources:**

| figure | source |
|---|---|
| `fig1_certificate.png` (a reach, b latch payoff) | generator above → `.claude/outputs/v1-certificate-payoff/paid_access_metrics.json` (`reach` with L=2.5/`distances`/`landing_rates`/`certificates`; `latch_and_certs.squeeze_injection`; `certificate_payoff.latch`) |
| `fig4_bibo.png` | generator above → same JSON, `certificate_payoff.bibo` (12 entries) |
| `fig_frontier_clean.png` | ⛔ **reconstruct** from `.claude/outputs/regime-remap-2000ep/tables.md` §"Item 2 — epoch-scaling frontier" |

⭐ **Reuse the banked instruments from the last render pass — do not rewrite them:**
`.claude/scratch/figure-render-pass/tap.py` (array-hashing data tap: records every numeric array handed to matplotlib, order-insensitive, ignores text/colour) · `size.py` · `pagesplit.py`.

⛔⛔ **THE VALIDATION BAR, and it is the point of the pass:** re-render each figure and **prove with `tap.py` that the new PNG plots exactly the same VALUES as the shipped one.** Only text, labels and styling may change. A precedent pass in this estate reproduced five of six figures byte-identically and reconstructed the sixth under exactly this discipline. ⛔ **If a value moves, STOP and report — do not adjust the data to match the picture.**

---

## 3. The string changes — ⛔ EXACT, and you may not deviate or invent

These are the Head's terms. ⛔ **Render precisely these strings. If one will not fit, wrap or rotate; ⛔ never abbreviate and never substitute your own wording — report the constraint instead.**

### 3a. `fig1_certificate.png`
| where | from | to |
|---|---|---|
| ⛔⛔ panel (a) annotation | `squeeze collapses past the box` | `squeeze: beyond the swept ζ ≤ 2.0` |
| panel (a) title | `(a) Reach: who lands — and with which receipt` | `(a) Landing rate vs. basin distance` |
| panel (b) title | `(b) The receipt cashed out: latch transported vs erased` | `(b) Goldstone charge: transported vs. erased` |
| (a) legend | `wormhole (det J = 1, ledger = 0)` | `wormhole (det J = 1, ΔV = 0)` |
| (a) legend | `no-physics router (det J = 0)` | `state-replacing map (det J = 0)` |
| (b) legend | `no-physics router (det J=0)` | `state-replacing map (det J = 0)` |
| (b) inset box | `router: std(Q_out) = 0.0 (erasure)` | `state-replacing: std(Q_out) = 0.0 (erasure)` |

✅ **Leave unchanged:** `causal box L = 2.5`, `crossover bracket`, both axis labels, `identity (Q preserved)`, `random shift (det J=1, no channel)`, `std(Q_in) = 0.0803`, and the `wormhole: std(Q_out) = 0.0803 (transport)` line.

### 3b. `fig4_bibo.png`
| where | from | to |
|---|---|---|
| title | `BIBO: an uncertified exit escapes; the receipt refuses it` | `Maximum excursion radius vs. destination locus of the wormhole jump` |
| legend | `wormhole, receipt ignored (ablation)` | `wormhole, screen ignored (ablation)` |
| legend | `no-physics router (coincides with ablation)` | `state-replacing map (coincides with ablation)` |
| legend | `wormhole + receipt (screened; refuses)` | `wormhole + screen (refuses exit)` |
| red annotation | `b = 5.0: energy ledger ΔH = 0 (FREE) — and the blind exit still escapes` | `b = 5.0: energy change ΔH = 0 (free) — the unscreened exit still escapes` |

✅ Leave unchanged: `coercive edge x_b = 3.54`, `escape radius`, both axis labels.

### 3c. `fig_frontier_clean.png`
- ⛔ **Remove the shaded Hopfield band from the right panel and its `Hopfield band` legend entry**, and drop `(Hopfield band shaded)` from that panel's title. The paper no longer makes a Hopfield comparison; the band re-introduces it visually.
- ✅ **Keep both panels and every CLU curve** (kv32/64/96/128, fidelity and gated accuracy).
- ⚠ The left panel title already reads `3 seeds` — **keep it**, and state **n=3** in the caption you propose. This is a MUST-FIX inherited from the fidelity audit: the frontier is 3 seeds, not the 8 pooled seeds quoted elsewhere.

---

## 4. Rendering standards

- **Type targets at printed size:** ticks ≥ 7 pt, axis labels and legend ≥ 8 pt, titles ≥ 9 pt **effective** (i.e. after the `\includegraphics` scale factor, not in the raw canvas).
- ⛔ **Preserve each figure's printed footprint** so pagination cannot move: measure the printed box from a **built PDF with `mutool`**, ⛔ never from `\linewidth` arithmetic. ⚠ A prior pass in this estate found the real shrink was worse than the arithmetic implied, and another found a title-length change moved the canvas width by 0.93 pt — **if a string change moves the box, report the delta; do not silently accept it.**
- Build in a **scratch copy**, ⛔ never inside `NIPSsubmission/v1-ttcl/`.
- **Back up all current PNGs with an md5 manifest before writing anything.**

---

## 5. Deliverables

1. **The re-rendered PNGs** in `figs/`, replacing the current ones.
2. ⭐ **The `tap.py` digest table** — per figure, old vs new, proving values identical. This is the pass's core evidence.
3. **A before → after string table** for every text element changed, so the Head can check it against their prose pass.
4. **Printed-box measurements** before and after, from built PDFs.
5. **Caption edits owed** — ⛔ **listed for the Head, never made.** At minimum: Figure 1's caption may need its wording checked against the new annotation, and the frontier figure needs a caption stating **n=3** if the Head inserts it.
6. **A note on `fig_regime_map.png` and `paid_access_reach.png`** — both banked and unused. ⛔ **Do not re-render them**; they are superseded (the latter was the v0.1 headline). One line confirming you left them alone.

## 6. Acceptance criteria

- ⛔ **No `.tex` file modified** — md5 manifest of `NIPSsubmission/v1-ttcl/*.tex` before and after, printed and identical.
- Every re-rendered figure's `tap.py` digest **matches its predecessor's**.
- `collapses past the box` appears in **no** PNG. `no-physics router` appears in **no** PNG.
- Type targets met at printed size; footprint deltas reported.
- Every negative positive-controlled.

## 7. ⚠ Two open items — report, do not act

1. **Figure 2 (`fig2_regime_map.png`) is no longer in the paper** — a prior pass removed it along with the Hopfield scoreboard. Its panel (a) (storage fidelity) is CLU-internal and could return as a single-panel figure. ⛔ **Do not render it unless the Head says so; note the option in your report.**
2. **`fig4_bibo.png` is not currently in `pj_sub.tex`.** You are re-rendering it so it is ready; the Head inserts it.

## 8. ⚠ Grep hazards on this machine

⛔ `grep` is a shell function resolving to **`ugrep 7.5.0`**: on bounded-context patterns over long lines it either **errors *"exceeds complexity limits"* and exits 0** — a silent false negative — or **hangs**. ⇒ use **`/usr/bin/grep`**; count with `grep -o … | wc -l`, ⚠ **never `grep -c`** (lines, not occurrences). ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — sweep per-file. ⚠ zsh does not word-split: quote any `--include='*.py'` glob. **Positive-control every negative.**

## DIAL DECLARATION
**Dials touched: NONE.** This pass re-renders three figures from banked data, changing text and styling only. It runs no experiment, changes no configuration, edits no paper file, and — by construction, proven with the data tap — changes no plotted value.
