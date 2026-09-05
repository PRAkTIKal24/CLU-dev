# BUILD-NOTE-R3 — retire the economic register; state the physics directly

**Pass:** `v1-terms-swap` (paper-writer spoke) · **Date:** 2026-08-27 · **File edited:** `pj_sub.tex` **only**.

## 0. Pin check
| | md5 | lines |
|---|---|---|
| required pin (task §0) | `de3585a6794add42c657600c9aa022db` | 382 |
| **measured at start** | `de3585a6794add42c657600c9aa022db` ✅ | 382 ✅ |
| after this pass | `08d31733b5648ed6ab4a6bbc5dc07ed8` | 382 (unchanged) |

**§0 Head-insertion check:** `govern the store` = **0**, `φ-bytes ledgered` = **0**, `ledgered` = **0**. The §A20.5 sentence was **not** present — the Head has not yet inserted it, so this pass ran before it exactly as sequenced, and there was nothing protected to preserve. ⛔ The approved `φ-bytes ledgered` wording is therefore untouched-by-absence, not touched-and-restored.

**Method:** one scripted, assertion-guarded pass (`.claude/scratch/v1-terms-swap/swap.py`, + a 2-site refinement `swap2.py`). Every replacement asserted `count == 1` against the whole file **before** any write; the file is written only after all 52 assertions pass. A pattern matching 0 or 2 sites aborts the run. No global find-and-replace on any stem was used.

---

## 1. Every changed site, before → after, tagged by class

Classes are the task's S1–S9, plus **P** = the §2 physics correction (rapidity, not distance), which the task mandates separately in its own section. **No other class of change was applied.**

| # | class | line (pre-swap) | before | after |
|---|---|---|---|---|
| R1 | S3 | 17 | `\title{\textbf{[WORKING TITLE: Paid Access: Test-Time Compute on a\\ Conservative Memory as a Physically-Metered Resource]}}` | `\title{\textbf{[WORKING TITLE: Certified Access: Test-Time Compute on a\\ Conservative Memory]}}` |
| R2 | S2 | 26 | `framework for paid access in conservative` | `framework for certified access in conservative` |
| R3 | S9 | 26 | `a bounded energy ledger, and an exact latch-transport law` | `a bounded energy change, and an exact latch-transport law` |
| R4 | P | 26 | `governor-re-absorbable energy and prices reach exponentially rather than capping it strictly` | `governor-re-absorbable energy and expands reach at an energy that grows exponentially in the rapidity $\zeta$ rather than capping it strictly` |
| R5 | S9+S6 | 26 | `This incurs a fixed discrete ledger cost and transports latched content precisely.` | `This requires a fixed, discrete energy change and transports latched content precisely.` |
| R6 | P+S8 | 26 | `confirming that the squeeze reach is exponentially priced in distance, whereas the wormhole reach is flat-priced` | `confirming that reaching beyond the box with the squeeze requires energy growing exponentially in rapidity $\zeta$, bounded by $e^{2\|\zeta\|}H$ and therefore quadratic in the excess distance, whereas the wormhole's energy requirement is fixed and independent of distance` |
| R7 | S2+S1 | 32 | `We formalize this paradigm as paid access, where each mechanism is accompanied by a physical receipt. This receipt comprises` | `We formalize this paradigm as certified access, where each mechanism is accompanied by a physical certificate. This certificate comprises` |
| R8 | S9 | 32 | `a bounded energy ledger to quantify injected energy` | `a bounded energy change to quantify injected energy` |
| R9 | S9 | 32 | `the exactness claims ($\det J$, the energy ledger and the latch-transport law)` | `the exactness claims ($\det J$, the energy change and the latch-transport law)` |
| R10 | S1 | 32 | `Crucially, these receipts are verifiable theorems` | `Crucially, these certificates are verifiable theorems` |
| R11 | S1 | 39 | `These are resolved by distinct mechanisms carrying different receipts.` | `These are resolved by distinct mechanisms carrying different certificates.` |
| R12 | S9 | 40 | `cures reach continuously with $\detJ=1$, a fixed ledger exact to zero, and exact latch transport.` | `cures reach continuously with $\detJ=1$, a fixed energy change exact to zero, and exact latch transport.` |
| R13 | S9 | 41 | `a bounded energy ledger is necessary but insufficient for Bounded-Input Bounded-Output (BIBO) stability` | `a bounded energy change is necessary but insufficient for Bounded-Input Bounded-Output (BIBO) stability` |
| R14 | S7 | 43 | `demonstrates a flat computational cost scaling relative to multi-hop diffusion` | `demonstrates a flat FLOP count under distance scaling relative to multi-hop diffusion` |
| R15 | S4 | 44 | `The measured savings ($9.9\times, 9.5\times, 6.2\times$ across kv32, kv64, kv96) are intra-CLU rationing` | `The measured step-reductions ($9.9\times, 9.5\times, 6.2\times$ across kv32, kv64, kv96) are intra-CLU rationing` |
| R16 | S8+P | 59 | `it does so at an exponential energy price, establishing a pricing law rather than bypassing the flow cap.` | `it does so at an energy that grows exponentially in the rapidity $\zeta$ and is bounded by $e^{2\|\zeta\|}H$, establishing an energy law rather than bypassing the flow cap.` |
| R17 | S8 | 64 | `stepping up until it is priced out past the swept rapidity budget $\zeta\le2.0$` | `stepping up until the required energy exceeds the swept rapidity budget $\zeta\le2.0$` |
| R18 | P+S6 | 64 | `(shaded), indicating an exponential energy cost for increasing distance.` | `(shaded); because reach grows as $\sinh\zeta$ while the injected energy is bounded by $e^{2\|\zeta\|}H$, the energy required grows exponentially in rapidity $\zeta$ and therefore quadratically in the excess distance.` |
| R19 | S9+S6 | 64 | `across all tested distances with exact volume preservation ($\detJ=1$) and zero ledger cost.` | `across all tested distances with exact volume preservation ($\detJ=1$) and zero energy change.` |
| R20 | S1 | 64 | `\textbf{(b) The receipt cashed out:}` | `\textbf{(b) The certificate cashed out:}` |
| R21 | S1 | 71 | `\subsection{Mechanism Receipts and Limitations}` | `\subsection{Mechanism Certificates and Limitations}` |
| R22 | S6+S9 | 76 | `The energetic cost is a discrete jump $\Delta H_{\rm wh}=V_\theta(q+\Delta)-V_\theta(q)$ requiring explicit ledgering, where matched loci result in free transport.` | `The energy required is a discrete jump $\Delta H_{\rm wh}=V_\theta(q+\Delta)-V_\theta(q)$ that must be accounted explicitly, where matched loci result in transport at zero energy.` |
| R23 | S9 | 76 | `the exact ledger evaluates to $0.0$` | `the exact energy change evaluates to $0.0$` |
| R24 | S9 | 79 | `alongside the ledger constraint` | `alongside the energy constraint` |
| R25 | S9 | 82 | `even if the jump is symplectic and the ledger is strictly free` | `even if the jump is symplectic and the energy change is exactly zero` |
| R26 | S9 | 82 | `an exit at $b=5.0$ yields a free ledger ($\Delta H=0.0$ exactly)` | `an exit at $b=5.0$ requires zero energy ($\Delta H=0.0$ exactly)` |
| R27 | S9+S6 | 82 | `an admissible exit at $b=3.0$ dictates a ledger cost of $\Delta H=2.88$` | `an admissible exit at $b=3.0$ requires an energy change of $\Delta H=2.88$` |
| R28 | S1 | 82 | `The full receipt parameters are detailed in Appendix B.` | `The full certificate parameters are detailed in Appendix B.` |
| R29 | S8+P | 93 | `Reach beyond $C_T$ is exponentially priced` | `Reach beyond $C_T$ needs energy $\le e^{2\|\zeta\|}H$` |
| R30 | S9 | 94 | `Flat access scaling; exact $0.0$ ledger` | `Flat access scaling; exact $0.0$ energy change` |
| R31 | P+S8 | 102 | `Consequently, reach via squeeze is exponentially priced in distance, whereas reach via wormhole is flat-priced.` | `Consequently, reach via squeeze requires energy growing exponentially in rapidity $\zeta$ and therefore quadratically in the excess distance, whereas the wormhole's energy requirement is fixed and independent of distance.` |
| R32 | S8 | 102 | `simply indicates it is priced out of the swept rapidity budget` | `simply indicates that the energy it requires exceeds the swept rapidity budget` |
| R33 | S1 | 104 | `\subsubsection{Cashing Out the Receipt: State Erasure vs. Transport}` | `\subsubsection{Cashing Out the Certificate: State Erasure vs. Transport}` |
| R34 | S1 | 124 | `volume preservation alone is not the latch receipt` | `volume preservation alone is not the latch certificate` |
| R35 | S8 | 127 | `strictly pricing every mechanism and establishing boundaries` | `strictly accounting the energy required by every mechanism and establishing boundaries` |
| R36 | S4 | 137 | `The measured savings of the CLU gate` | `The measured step-reductions of the CLU gate` |
| R37 | S7 | 140 | `maintains a flat computational cost as distance $N$ scales` | `maintains a flat FLOP count as distance $N$ scales` |
| R38 | S2+S5+P+S9+S6 | 148 | `test-time compute is paid access governed by physical prices. A squeeze buys escape bounding but prices distance exponentially, whereas a non-local wormhole achieves global reach at exact volume preservation ($\detJ=1$), zero ledger cost, and strictly preserved latch transport.` | `test-time compute is certified access: the phase-volume, energy and latch consequences of every access mechanism are fixed by the dynamics. A squeeze cures escape with a bounded energy injection but requires energy growing exponentially in rapidity $\zeta$, and therefore quadratically in the excess distance, whereas a non-local wormhole achieves global reach at exact volume preservation ($\detJ=1$), an energy requirement that is fixed and independent of distance (exactly zero on the matched channel measured here), and strictly preserved latch transport.` |
| R39 | S9 | 148 | `bounding the energy ledger is an insufficient condition for BIBO stability` | `bounding the energy change is an insufficient condition for BIBO stability` |
| R40 | S2 | 157 | `Extending the paid-access framework to repeated test-time retries` | `Extending the certified-access framework to repeated test-time retries` |
| R41 | S2 | 255 | `\section{The Paid-Access Certificate Table and BIBO Battery}` | `\section{The Access-Certificate Table and BIBO Battery}` |
| R42 | S1 | 256 | `\subsection*{B.1 Table 1: Receipt per Mechanism}` | `\subsection*{B.1 Table 1: Certificate per Mechanism}` |
| R43 | S9 | 262 | `$\Delta V{=}V(b){-}V(a)$, discrete ledger` | `$\Delta V{=}V(b){-}V(a)$, discrete jump` |
| R44 | S1 | 278 | `Wormhole with receipt` | `Wormhole with certificate` |
| R45 | S1 | 284 | `Wormhole receipt logic` | `Wormhole certificate logic` |
| R46 | S9 | 290 | `where the ledger is exactly zero` | `where the energy change is exactly zero` |
| R47 | S9 | 290 | `$\detJ=1$ combined with a free ledger is insufficient for BIBO stability` | `$\detJ=1$ combined with a zero energy change is insufficient for BIBO stability` |
| R48 | S8 | 309 | `The squeeze is priced out past the swept rapidity grid at $d=4.0$.` | `The squeeze requires more energy than the swept rapidity grid provides at $d=4.0$.` |
| R49 | P | 309 | `requires a rapidity of $\zeta\ge2.0105$, corresponding to an exponential energy scale.` | `requires a rapidity of $\zeta\ge2.0105$, corresponding to an energy bound growing as $e^{2\|\zeta\|}$ in the rapidity.` |
| R50 | S4 | 332 | `Intra-CLU Savings` | `Intra-CLU Step-Reduction` |
| R51 | S9 | 76 (2nd stage of R24) | `that must be accounted explicitly` | `that must be tracked explicitly` |
| R52 | S8 | 127 (2nd stage of R44) | `strictly accounting the energy required by every mechanism` | `strictly tracking the energy required by every mechanism` |
### 1.1 Per-class tally (measured, against the pinned file)

| class | task's expected sites | changed | reconciliation |
|---|---|---|---|
| S1 `receipt` → `certificate` | 12 | **12** ✅ | exact (`receipt` 9 + `Receipt` 3, incl. 3 plurals) |
| S2 `paid access` → `certified access` | 4 | **5** ⚠ | see §1.2 — the file carries 5 body sites, not 4 |
| S3 title | 1 (+ kills the only `metered`) | **1** ✅ | `metered` residual = **0** |
| S4 `savings` → step reduction | 3 | **3** ✅ | matched the file's existing hyphenated form `step-reduction` |
| S5 `buys escape` → `cures escape` | 1 | **1** ✅ | §3.1's own verb |
| S6 `cost` (energy) → energy required | per-site | **6** | see §1.3 |
| S7 `cost` (FLOP) → FLOP count | per-site | **2** | see §1.3 |
| S8 `priced`/`pricing` → energy | 10 | **10** ✅ | exact (`priced` 8 + `pricing` 2) |
| S9 `ledger` → explicit energy change | 19 | **19** ✅ | exact (incl. the one `ledgering`) |
| **P** physics fix (task §2) | 3 named + equivalents | **7** | see §2 — 3 named + 1 unnamed equivalent + 3 clarifications |

`rationing` (Head ruling: STAYS) = **12 before → 12 after**, byte-identical. The `ration`/`duration`/`iteration` false-friend class did not arise.

### 1.2 ⚠ S2 count discrepancy — reported, not silently absorbed
The task scoped S2 at **4** sites. The pinned file carries **5** non-title sites of the `paid access` class:

1. L26 abstract `framework for paid access` 2. L32 `We formalize this paradigm as paid access` 3. L148 `test-time compute is paid access` 4. L157 `the paid-access framework` 5. **L255 `\section{The Paid-Access Certificate Table and BIBO Battery}`**

Site 5 is an appendix **section heading**, which is plausibly why it was counted with the titles rather than with S2. It is the same class (`paid access` → `certified access`), so swapping it is **not** an unlisted class of change. A literal swap would have produced the stutter *"The Certified-Access Certificate Table"*; the heading was therefore set to **`The Access-Certificate Table and BIBO Battery`**, which keeps both concepts, changes no claim, and matches B.1's *"Certificate per Mechanism"*. ⚠ **Flagged for the Head** in case the literal form is preferred.

### 1.3 `cost` — per-site disposition (10 occurrences of the stem)

| line | text | class | disposition |
|---|---|---|---|
| 26 | *"This incurs a fixed discrete ledger **cost**"* | S6+S9 | → *"requires a fixed, discrete energy change"* |
| 43 | *"a flat computational **cost** scaling"* (contribution 5, mirrors §4.2) | S7 | → *"a flat FLOP count under distance scaling"* |
| 64 | *"an exponential energy **cost** for increasing distance"* | S6+**P** | → rapidity form, see §2 |
| 64 | *"zero ledger **cost**"* | S6+S9 | → *"zero energy change"* |
| 76 | *"The energetic **cost** is a discrete jump"* | S6 | → *"The energy required is a discrete jump"* |
| 82 | *"dictates a ledger **cost** of $\Delta H=2.88$"* | S6+S9 | → *"requires an energy change of $\Delta H=2.88$"* |
| 140 | *"a flat computational **cost** as distance $N$ scales"* (§4.2) | S7 | → *"a flat FLOP count as distance $N$ scales"* |
| 148 | *"zero ledger **cost**"* | S6+S9 | → restated, see §2/§3 |
| **26** | *"nor do they quantify the **costs** incurred against the model's fundamental stability guarantees"* | — | ⛔ **LEFT.** Neither the energy sense (S6) nor the FLOP sense (S7): it is the intro's generic framing — what the extra compute costs *in stability terms*. Not receipt/price/ledger register. |
| **200** | *"**cost** checkpoints $300/1200/2100/3000$"* | — | ⛔ **LEFT.** A **flag-provenance table entry** naming the harness's actual gate configuration (relaxation-step checkpoints). Renaming a provenance descriptor would break the reader's mapping back to the run config (protocol §5 / charter C-7). |

8 changed + 2 justified survivors = 10 ✅

---

## 2. ⛔ The physics fix (task §2) — sites, and what each now says

The pinned file asserted energy growth **in distance**. Corrected to **rapidity ζ, with distance in brackets**, per the Head's approved form. Located by content, not line number.

**The three named sites:**
1. **Abstract** — *"confirming that the squeeze reach is exponentially priced in distance, whereas the wormhole reach is flat-priced"* → *"confirming that reaching beyond the box with the squeeze requires energy growing exponentially in rapidity $\zeta$, bounded by $e^{2|\zeta|}H$ and therefore quadratic in the excess distance, whereas the wormhole's energy requirement is fixed and independent of distance."*
2. **§3.2 body (L102)** — *"Consequently, reach via squeeze is exponentially priced in distance, whereas reach via wormhole is flat-priced."* → *"Consequently, reach via squeeze requires energy growing exponentially in rapidity $\zeta$ and therefore quadratically in the excess distance, whereas the wormhole's energy requirement is fixed and independent of distance."*
3. **Figure-1 caption (L64)** — *"(shaded), indicating an exponential energy cost for increasing distance."* → *"(shaded); because reach grows as $\sinh\zeta$ while the injected energy is bounded by $e^{2|\zeta|}H$, the energy required grows exponentially in rapidity $\zeta$ and therefore quadratically in the excess distance."*

**⚠ A FOURTH site carrying the same wrong form, not named in the task, found by content sweep:**
4. **§5 conclusion (L148)** — *"A squeeze buys escape bounding but **prices distance exponentially**"*. This is an "equivalent" under acceptance criterion §7 (`exponentially in distance` **and equivalents** = 0), so it was corrected: → *"A squeeze **cures escape** with a bounded energy injection but requires energy growing exponentially in rapidity $\zeta$, and therefore quadratically in the excess distance…"*. (Same site also carries S2, S5, S6 and S9 ×2.)

**Three further clarifications** so no residual reads as distance-exponential: L59 (*"an exponential energy price"* → *"an energy that grows exponentially in the rapidity $\zeta$ and is bounded by $e^{2|\zeta|}H$"*), L93 table diagnostic (*"Reach beyond $C_T$ is exponentially priced"* → *"Reach beyond $C_T$ needs energy $\le e^{2|\zeta|}H$"*), L309 (*"corresponding to an exponential energy scale"* → *"corresponding to an energy bound growing as $e^{2|\zeta|}$ in the rapidity"*).

✅ **The already-correct site was LEFT untouched**, exactly as instructed: contribution 2 (L40) still reads *"prices reach exponentially in rapidity"*. Only the `ledger` clause on that line changed (*"a fixed ledger exact to zero"* → *"a fixed energy change exact to zero"*).

**Verification:** `exponentially in distance` · `exponentially priced in distance` · `priced in distance` = **0**.

---

## 3. ⛔⛔ MF-B compliance statement — the retracted-claim fence

The referee MUST-FIX closure retracted *"cannot beat the box"* / *"collapses past the box"* and replaced them with the pricing law. **The word went; the content did not.** Every restated pricing-law site still asserts (a) squeeze reach requires energy `≤ e^{2|ζ|}H`, growing with ζ, and (b) the wormhole's `ΔV` is independent of `Δ`.

| site | (a) squeeze energy `≤ e^{2|ζ|}H`, grows with ζ | (b) wormhole independent of Δ |
|---|---|---|
| **Abstract** | *"requires energy growing exponentially in rapidity $\zeta$, **bounded by $e^{2\|\zeta\|}H$**"* ✅ | *"the wormhole's energy requirement is **fixed and independent of distance**"* ✅ |
| **L59 setup** | *"an energy that grows exponentially in the rapidity $\zeta$ and **is bounded by $e^{2\|\zeta\|}H$**, establishing an energy law rather than bypassing the flow cap"* ✅ | n/a (squeeze-only paragraph; unchanged scope) |
| **Fig-1 caption** | *"the **injected energy is bounded by $e^{2\|\zeta\|}H$**, the energy required grows exponentially in rapidity $\zeta$"* ✅ | *"flat landing rate of 1.0 **across all tested distances** … **zero energy change**"* ✅ |
| **L93 table** | *"Reach beyond $C_T$ **needs energy $\le e^{2\|\zeta\|}H$**"* ✅ | adjacent row: *"Flat access scaling; exact $0.0$ energy change"* ✅ |
| **§3.2 body (L102)** | preceding sentence retained verbatim: *"because the energy scales as $e^{2\|\zeta\|}H$, reaching $d=4.0$ … requires $\zeta\approx2.01$"* ✅ | *"the wormhole's energy requirement is **fixed and independent of distance**"* ✅ |
| **§5 conclusion (L148)** | *"requires energy growing exponentially in rapidity $\zeta$, and therefore quadratically in the excess distance"* ✅ | *"an energy requirement that is **fixed and independent of distance** (exactly zero on the matched channel measured here)"* ✅ |
| **C.1 (L309)** | *"corresponding to an **energy bound growing as $e^{2\|\zeta\|}$** in the rapidity"* ✅ | n/a |

⛔ **The anti-"cannot reach" guard is intact and was deliberately preserved.** §3.2 still reads: *"The squeeze failing to reach the outer distances simply indicates that the energy it requires exceeds the swept rapidity budget ($\zeta\le2.0$), **not that it is fundamentally incapable of reaching them if provided sufficient energy**."* Residual counts: `cannot beat the box` = 0, `collapses past the box` = 0, `cannot reach` = 0 (all were already 0; kept at 0). **No site regressed to a reach-impossibility claim.**

**Zero BLOCKED sites.** Every site was restatable without changing what it asserts.

---

## 4. ⛔⛔ The free-ledger fence (task §3 / CM-7 must-travel rider)

`ΔV` (potential difference) and `ΔH` (energy change) were chosen **per site** from what each sentence is about; no global swap. The wormhole's `ΔV = V(b) − V(a)` is preserved verbatim in the B.1 table.

**The zero is explicit, and `free` was NOT softened.** It was replaced with the *literal value*, never with "low"/"small"/"modest":

| site | before | after | zero explicit? |
|---|---|---|---|
| §3.1 BIBO (L82) | *"even if the jump is symplectic and the ledger is **strictly free**"* | *"…and the energy change is **exactly zero**"* | ✅ |
| §3.1 BIBO (L82) | *"an exit at $b=5.0$ yields a **free ledger** ($\Delta H=0.0$ exactly)"* | *"an exit at $b=5.0$ **requires zero energy** ($\Delta H=0.0$ exactly)"* | ✅ **`ΔH = 0.0` retained verbatim** |
| App. B.2 (L290) | *"where the **ledger is exactly zero**"* | *"where the **energy change is exactly zero**"* | ✅ |
| App. B.2 (L290) | *"$\detJ=1$ combined with a **free ledger** is insufficient for BIBO stability"* | *"$\detJ=1$ combined with a **zero energy change** is insufficient for BIBO stability"* | ✅ |

The full CM-7 clause survives at both places: a `ΔH = 0.0` exit at `b = 5.0` is **admitted** by the energy-only sub-level test and **escapes anyway** (`r* ∝ T`), so **coercive-component membership is the operative clause**. Appendix B.2's `Energy sub-level test` row (`admit` at `b=5.0`) is byte-unchanged.

### ⚠ 4.1 OPEN ITEM for the Head — two insufficiency sites carry no zero
Task §7 requires the zero *"still explicit at all three sites."* Measured against the pinned file, the zero was explicit at **two** sites (both above, in §3.1 and B.2) — both preserved. The other two sites that assert BIBO-insufficiency carry only the **bounded** half and never carried the zero:

- **Contribution 3 (L41):** *"a bounded energy change is necessary but insufficient for BIBO stability without coercive-component screening."*
- **§5 conclusion (L148):** *"bounding the energy change is an insufficient condition for BIBO stability; strict coercive-component screening is required."*

CM-7's rider reads *"a bounded/even-FREE energy ledger isn't sufficient for BIBO"* — so the sharp "or even zero" half is **missing at these two sites**. ⛔ **I did not add it**, because task §5 forbids widening or narrowing a claim in a vocabulary pass, and adding "or even zero" makes both sentences assert strictly more. **This needs a one-line Head ruling** (proposed wording in the report's open-questions section). It is a pre-existing gap, not one this pass created.

---

## 5. Two-way numeric check (printed)

```
total numeric tokens: before=641  after=646
BEFORE-only (numbers LOST):   {}          <-- must be empty: EMPTY ✅
AFTER-only  (numbers GAINED): {'2': 5}
```
⛔ **Zero numeric values changed, dropped, rounded, re-precisioned or re-signed.** The only gain is **five occurrences of the literal `2`**, every one of which is the exponent inside a newly written `e^{2|\zeta|}` / `e^{2|\zeta|}H` — the paper's own pre-existing certificate expression (already at L73 and L264), added at the five MF-B sites (abstract, L59, L64, L93, L309). No `±`, no seed count, no unit, no measured value moved. Verified: `e^{2|\zeta|}` occurrences 4 → 9 = +5, matching exactly.

## 6. Residual sweep (positive-controlled)

```
receipt = 0     ledger  = 0     metered = 0     priced  = 0
pricing = 0     savings = 0     buys    = 0     paid    = 1   <-- see below
positive controls (must be > 0):  certificate = 32   rationing = 12   wormhole = 29
```
The positive controls confirm the sweep is live (i.e. the zeros are real, not a `ugrep` silent-false-negative). All sweeps used `/usr/bin/grep -o … | wc -l`, never `grep -c`.

**Every survivor, justified:**

| survivor | count | why it stays |
|---|---|---|
| `paid` inside **`unpaid`** (L76) | 1 | *"volume preservation is broken by an **unpaid** contraction ($\detJ=1+\nabla g\cdot\Delta\neq1$)"* — the state-dependent-gate design guard. **A false friend of the `paid access` class**, and not a listed class. ⚠ Flagged: it is the same register and a one-word fix (*"uncompensated contraction"*) if the Head wants it. |
| `cost` (energy sense) | 0 | all 8 energy/FLOP-sense sites changed |
| `costs` / `cost` (other senses) | 2 | L26 generic stability framing; L200 provenance-table flag descriptor — see §1.3 |
| `price` / `prices` | 6 | ⛔ **NOT SWEPT — out of scope.** See §7. |

## 7. ⚠ OUT-OF-SCOPE RESIDUAL the Head should rule on: `price` / `prices` (6 sites)

The swap map's **S8 names `priced` / `pricing` and counts them at 10** — which is *exactly* `priced` (8) + `pricing` (2). The bare **`price` (2)** and **`prices` (4)** forms are a **different 6 occurrences**, they are **not** in any S-class, they are **not** in §4's protected list, and §6's residual-sweep list does not name them. §1 rules that *"a class not listed is out of scope"*, so **I did not sweep them.** They appear to have been *unmeasured*, not *protected*.

**Four were removed incidentally** because a mandated S/P restatement rewrote the whole clause (listed here so nothing is silent): L26 *"prices reach exponentially"* (P), L59 *"exponential energy price"* (S8), L148 *"governed by physical prices"* (S2) and L148 *"prices distance exponentially"* (P).

**Two survive, and they read in exactly the register the Head is retiring:**
1. **L32 (intro):** *"the physical properties of the phase space strictly **price** every access mechanism."*
2. **L40 (contribution 2):** *"the Lorentz squeeze cures escape with bounded injection and **prices** reach exponentially in rapidity."* — ⛔ this is the site task §2 explicitly says is **physics-correct, leave it**; only its wording is in question, never its content.

⇒ **Head ruling requested.** Proposed one-line fixes, ready to apply: (1) → *"…strictly **determine the energy required by** every access mechanism"*; (2) → *"…cures escape with bounded injection at an energy growing exponentially in rapidity."*

## 8. Also out of scope, also flagged: the `cashed out` idiom (2 sites)
S1 swapped the noun but the surrounding idiom is the same economic register and is not a listed class: **§3.2.1 heading** *"Cashing Out the **Certificate**: State Erasure vs. Transport"* and the **Fig-1 caption** *"(b) The **certificate** cashed out"*. Minimal in-class swap applied; idiom left. Proposed fixes if wanted: *"The Certificate Cashed Out"* → *"What the Certificate Buys: State Erasure vs. Transport"* is **worse** (buys is the same register) — suggest *"The Certificate's Consequence: State Erasure vs. Transport"* and *"(b) The certificate in force"*.

## 9. Guards
- **No claim widened or narrowed.** Zero BLOCKED sites; the two places where widening *would* have been required (§4.1) were left alone and escalated instead.
- **No intensifiers introduced.** `strictly` 15 → **14**, `clearly` 1 → 1, `conclusively` 1 → 1, `fully` 5 → 5. ⚠ **One incidental removal, as required to be listed:** L82's *"the ledger is **strictly** free"* → *"the energy change is **exactly zero**"*. `strictly` was dropped because the replacement states the literal measured value (`ΔH = 0.0 exactly`), which is stronger, not softer.
- **DO-NOT-SWEEP list verified byte-identical, before → after:** `physics-free` 4 → 4 · `distribution-free` 1 → 1 · `budget` 21 → 21 · `Goldstone charge` 3 → 3 · `rationing` 12 → 12 · `account` (abstract, = *explanation*) 1 → 1.
- **CM-8 adjacency rider held:** `intra-CLU` remains adjacent to **every** step-reduction figure — contribution 6 (*"The measured step-reductions ($9.9\times, 9.5\times, 6.2\times$ across kv32, kv64, kv96) are **intra-CLU** rationing against a full-budget CLU baseline"*), §4.1 body (same, *"represent **intra-CLU** rationing against a full-budget CLU baseline"*), and the C.3 table column header (**`Intra-CLU Step-Reduction`**). S4 changed the paper's word, not the registry's clause.
- **Charter:** C-1 (no audit-confession paragraph — none added, none present) · C-2 (designed testbed still labeled *verification*, learned-memory results still *evidence*; §3 heading "Verification of the Certificate Stack" unchanged) · C-5 (no scale qualifier removed; the "quadratically in the excess distance" bracket is a stated consequence of the paper's own reach bracket, not a new generalization) · C-6 (certificate fine print still adjacent: the `e^{2|ζ|}` matched-quadratic caveat at L73 and the LTT exchangeability caveat at L134 are byte-unchanged).

## 10. Build

`tectonic 0.15`-class toolchain (`/opt/homebrew/bin/tectonic`); no `pdflatex`/`latexmk` on this machine.

| | errors | undefined refs/citations | pages | main text | appendices | references |
|---|---|---|---|---|---|---|
| pre-swap (`pj_sub.BEFORE.tex`) | 0 | 0 | 13 | pp. 1–8 | A starts p. 9 | p. 13 |
| **post-swap (`pj_sub.tex`)** | **0** | **0** | **14** | **pp. 1–8** | **A starts p. 9** | pp. 13–14 |

⚠ **Total grew 13 → 14 pages; the main-text split is UNCHANGED.** Appendix A still begins on p. 9 and the reference list still begins on p. 13 — the extra page is purely the last five reference entries spilling over, because the MF-B restatements are a few words longer than the phrases they replaced. **No main-text or appendix boundary moved.**

Warnings: 18 `hbox` badness warnings, **identical set before and after** (B.1 table lines 260–263, C.1 table line 308) — pre-existing typesetting, not introduced here.

## 11. Scope proof
```
pj_sub.tex        de3585a67… -> 08d31733b5648ed6ab4a6bbc5dc07ed8   (this pass)
submission.tex    caef2272f9dc96d349b46486563d24ee -> UNCHANGED
.claude/papers/v1-short/**  (11 files)            -> ALL UNCHANGED
```
Full manifest in `.claude/outputs/v1-terms-swap.md` §Scope proof.

## DIAL DECLARATION
**Dials touched: NONE.** Vocabulary pass plus one physics correction on one `.tex` file. No experiment run, no configuration changed, no measured value moved.
