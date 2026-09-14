# V1 — retire the economic register; state the physics directly

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-27.** The Head is removing the receipt/price/cost/ledger vocabulary because it reads as generated filler and undercuts the work. ⛔ **The words go; the CONTENT they carry is registry-mandated and must survive intact.**

**Agent:** `paper-writer` · **Edits `.claude/NIPSsubmission/v1-ttcl/pj_sub.tex` and nothing else.**
**Deliverable #1:** `BUILD-NOTE-R3.md` in that folder · **Report:** `.claude/outputs/v1-terms-swap.md`

---

## 0. ⛔ Pin check, and why this pass runs BEFORE the Head's own edits

At scoping, `pj_sub.tex` md5 = **`de3585a6794add42c657600c9aa022db`** (382 lines, 6,014 tex-words). **Compute it first. If it differs, STOP and report — do not write.** ⚠ **This file is live-edited and moved four times during this session, including once between the drafting and the verification of this very task file.** The swap counts in §1 were re-measured against this exact md5.

⭐ **This pass is deliberately sequenced first.** The Head has pending insertions, one of which — the §A20.5 sentence — **contains the word `ledgered` inside an approved wording that binds verbatim**. Running the swap first means that sentence is not yet in the file, so there is nothing for you to wrongly "fix." ⛔ **If you find `govern the store` or `φ-bytes ledgered` present, the Head has already inserted it: LEAVE IT EXACTLY AS IS and say so in your report.**

---

## 1. The swap map — this is a RULE-BASED pass, so the guards are stricter

Unlike an enumerated pass, you are applying classes of change. ⇒ **Every single changed site must appear in your build note as before → after.** ⛔ A class not listed below is out of scope.

| # | from | → to | sites |
|---|---|---|---|
| S1 | `receipt` | `certificate` | 12 |
| S2 | `paid access` | `certified access` | 4 |
| S3 | title `Paid Access… Physically-Metered Resource` | `Certified Access: Test-Time Compute on a Conservative Memory` | 1 (+ kills the only `metered`) |
| S4 | `savings` | `step reduction` | 3 — ⭐ the file already uses `step-reduction`; match its existing form |
| S5 | `buys escape` | `cures escape` | 1 — ⭐ the paper's own §3.1 verb |
| S6 | `cost` (energy sense) | `energy required` | per-site |
| S7 | `cost` (FLOP sense, §4.2) | `FLOP count` | per-site |
| S8 | `priced` / `pricing` | restate as energy — see §2 | 10 |
| S9 | `ledger` | the explicit energy change — see §3 | 19 |

⛔ **`rationing` STAYS (12 sites) — Head ruling.** It names what the gate does (stopping early, spending less than budgeted); *"allocation"* names something the paper does not do, and it borrows from a CM-2 clause this paper deliberately does not invoke. ⇒ the `ration`/`duration`/`iteration` false-friend class does not arise.

---

## 2. ⛔⛔ THE PHYSICS FIX — the current text is WRONG and the swap must correct it

**The paper says energy grows "exponentially in distance" in three places. That is false.** Energy is bounded by `e^{2|ζ|}H`, i.e. **exponential in RAPIDITY ζ**. Because reach grows like `sinh ζ`, the energy required is **quadratic in the excess distance**, not exponential.

**Advisor-verified against the paper's own printed predictions** (the bracket reproduces both exactly: ζ = 2.0105 at d=4.0, ζ = 2.6441 at d=5.0):

| | energy ratio, d = 4 → 5 |
|---|---|
| from the paper's own ζ values, `e^{2ζ}` | **3.55** |
| quadratic in excess distance | **3.64** ✓ |
| exponential in distance | 7.39 ✗ |

⇒ **Head ruling: state it in rapidity, with distance in brackets.** The correct form, e.g.:
> *"reaching beyond the box requires energy growing exponentially in rapidity ζ (and therefore quadratically in the excess distance), whereas the wormhole's energy requirement is fixed and independent of distance."*

**Sites carrying the wrong form** (locate by content, ⛔ not by line number): the **abstract**, the §3.2 body, and the Figure-1 caption's *"exponential energy cost for increasing distance"*. ✅ The site already reading *"exponentially in rapidity"* is **correct — leave it**.

⛔⛔ **MF-B FENCE.** The pricing law was a referee MUST-FIX closure: the phrases *"cannot beat the box"* / *"collapses past the box"* were **falsified and retracted**, and replaced by this content. **The word may go; the content may not.** Every site must still say that squeeze reach costs energy `≤ e^{2|ζ|}H` growing with ζ, and that the wormhole's `ΔV` is **independent of Δ**. ⛔ **If you cannot restate a site while keeping that, STOP and report it** — regressing to *"cannot reach"* re-opens a closed referee finding.

---

## 3. ⚠ `ledger` — per-site, and ⛔ never conflate ΔV with ΔH

`ΔV = V(b) − V(a)` is the **potential** difference; `ΔH` is the **energy** change. They coincide for the wormhole (momentum is unchanged) but they are **not the same object**. ⛔ **Choose per site from what the sentence is actually about; never swap globally.**

⛔⛔ **THE FREE-LEDGER FENCE — CM-7's must-travel rider, and the zero is load-bearing.** Three sites carry *"a bounded, or even FREE, energy ledger is not sufficient for BIBO."* The sharp form is that at `b = 5.0` the jump costs **exactly nothing** (`ΔH = 0.0`), an energy-only sub-level test **admits** it, and it **escapes anyway** — so coercive-component membership, not the energy, is the operative clause.
⇒ Rename the noun if you like, but ⛔ **keep the zero explicit.** **Do not soften `free` to "low", "small" or "modest" — that inverts the negative into its opposite.**

---

## 4. ⛔ DO NOT SWEEP — false friends and terms of art

| leave alone | why |
|---|---|
| `physics-free` (4) | CM-7's own name for the 449-param baseline |
| `distribution-free` (1) | standard statistics (LTT) |
| `budget` (21) | epoch/compute/read budget is a term of art, not metaphor |
| `account` at the abstract | means *explanation* — *"without providing a structural account of the capabilities"* |
| `Goldstone charge` | physics |
| `rationing` (12) | Head ruling, §1 |
| ⛔ **`φ-bytes ledgered`**, if present | **approved wording, binds VERBATIM** (§0) |

⚠ **CM-8's registry clause reads `"6–10× savings" = intra-CLU rationing`.** S4 changes the paper's word, not the registry's. ⛔ **The `intra-CLU` scope must remain adjacent to every step-reduction figure** — that adjacency is a standing rider and is independent of the noun.

---

## 5. Method

⭐ **Scripted, assertion-guarded, single-occurrence replacements, each asserting `count == 1` before writing.** A pattern matching 0 or 2 sites must fail loudly rather than write. ⛔ **No global find-and-replace on any stem** — every site is read in context first.

- ⛔ **Zero numbers change.** No value, precision, ±, seed count or unit moves. Two-way numeric check, printed.
- ⛔ **No claim widens or narrows.** This is a vocabulary pass; if restating a sentence would change what it asserts, **STOP and report the site.**
- ⛔ **No intensifiers introduced** (*strictly*, *clearly*, *conclusively*, *fully*). If you remove one incidentally, list it.

## 6. Deliverables

1. **`BUILD-NOTE-R3.md`** — every changed site, before → after, tagged by class S1–S9. ⛔ Any site you could not restate without changing meaning, listed separately as BLOCKED.
2. **Two-way numeric check**, printed.
3. **A residual sweep, positive-controlled**: `receipt` · `ledger` · `paid` · `metered` · `priced` · `pricing` · `savings` · `buys` — with surviving counts and every survivor justified (some `cost` uses in the energy sense may legitimately remain; `ledgered` inside §A20.5 must).
4. **An MF-B compliance statement**: quote each restated pricing-law site and show the energy content survived.
5. **Build**: 0 errors, 0 undefined references; report pages and the main-text split.

## 7. Acceptance criteria

- Pin check passed (§0), or aborted.
- ⛔ `submission.tex` and `.claude/papers/v1-short/**` byte-untouched (md5 manifest printed).
- Every changed site is in the build note. ⛔ **No unlisted class of change.**
- `collapses past the box` / `cannot beat the box` = **0** (they are already 0 — keep it that way).
- `exponentially in distance` and equivalents = **0**; the rapidity form present.
- The free-ledger zero (`ΔH = 0`) still explicit at all three sites.

## 8. ⚠ Grep hazards on this machine

⛔ `grep` is a shell function resolving to **`ugrep 7.5.0`**: on bounded-context patterns over long `.tex` lines it either **errors *"exceeds complexity limits"* and exits 0** — a silent false negative — or **hangs**. ⇒ use **`/usr/bin/grep`**; count with `grep -o … | wc -l`, ⚠ **never `grep -c`** (lines, not occurrences). ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — sweep per-file. ⚠ zsh does not word-split: quote any glob. ⚠ **False friends have already produced one wrong finding this session** (`matvec` returned 0 while the claim was present spelled out). **Read every hit in context.**

## DIAL DECLARATION
**Dials touched: NONE.** This pass edits one `.tex` file's vocabulary and corrects one physics statement. It runs no experiment, changes no configuration, and changes no measured value.
