# V1 — the proofs appendix: make the paper self-contained on the two results it attributes to a note that will not ship

**Scoped by the V1 Shorts Advisor at the Head's direction, 2026-08-27.** ⛔ **This supersedes an earlier draft of this task file entirely — its line anchors were five passes stale.**

**Agent:** `physics-theorist` — chosen deliberately: this is algebra and you can **numerically self-check** it. That is the pattern that made two sibling appendices referee-proof.
**Writes:** `.claude/NIPSsubmission/v1-ttcl/pj_sub.tex` (one appendix block + at most two `\ref` insertions) and `.claude/outputs/v1-derivation-appendix.md`.

---

## 0. ⛔ Pin check

`pj_sub.tex` md5 at scoping = **`727ebee2b8498b4095f8bb7159258f90`**. **Compute it first; if it differs, STOP and report.** This file is live-edited and has moved seven times this session. ⚠ **Locate every site by content, never by line number** — anchors in the superseded draft of this task were already dead.

---

## 1. Why this exists — the note is being cut, and it takes two proofs with it

**Head ruling: no theory note ships with V1.** Its bibliography entry is cut. But the paper does not merely *mention* the note — **it attributes a proof to it, in main text**:

> *"The **theory note proves** that in relativistic mode, a single drift step advances position $q$ by at most $\varepsilon c/\sqrt{M_i}$ per coordinate. Consequently, the reachable set is strictly bounded…"*

⇒ **Cut the note without replacing this and the paper asserts a theorem the reader cannot reach, using the word "proves".** A sibling short hit exactly this — a referee found *"self-containment fails; the central closed forms are underived in-submission"* — and it was closed by an appendix of this kind.

### The two objects, and they are the whole job

| # | object | where it is asserted | why load-bearing |
|---|---|---|---|
| **P1** | ⛔ **The causal box** — one relativistic drift advances $q$ by at most $\varepsilon c/\sqrt{M_i}$ per coordinate, hence $Q_T \subseteq C_T$ with $L_i = T\varepsilon c/\sqrt{M_i}$ | main text, the *"theory note proves"* sentence (locate by that phrase) | **The paper's central theorem.** It is what makes reach a *kinematic* failure mode and the reach/escape dichotomy falsifiable. Contribution 1 and the whole of §3 rest on it. |
| **P2** | ⛔ **The bounded-injection certificate** $H(S_\zeta z) \le e^{2\vert\zeta\vert}H(z)$ | **six sites**, incl. the abstract, §3.1 ×2 and the Figure-1 caption | Carries the abstract, contribution 2, §3.1, §5 and Table 1. The paper gives the squeeze's action and its symplecticity but **never the energy inequality's derivation.** |

⚠ **A third candidate, judge it yourself and declining is a valid result:** the ledger's **hard-gate regime** — the design guard that a gate varying *during* the jump gives $\det J = 1 + \nabla g\cdot\Delta \ne 1$ (unit test $2.05$). Include only if genuinely underived and cheap.

⛔ **"Lean" is binding.** Derive only what the paper asserts and cannot reach. **A correct derivation of something the paper does not claim is scope creep.**

⚠ **Vocabulary has changed.** A prose pass removed the receipt/ledger/price register: the paper now says **certificate**, **screen**, and states energy changes as **$\Delta V$ / $\Delta H$** directly. ⛔ Match the paper's current words; do not reintroduce retired ones.

---

## 2. ⛔ The two rules that make this trustworthy

**2a. PRE-REGISTER YOUR CHECK VALUES BEFORE ANY SCRIPT RUNS, mtime-provably.** Write `PREREG.md` first. A sibling pass was provable to the second (`PREREG.md` 02:43 · `check.py` 02:45 · results 02:54), and that is what made it referee-proof rather than merely correct.

⭐ **Check against the COMPOSED MAP, never against the closed form itself.** Build the damped velocity-Verlet step from its elementary kick–drift–kick–damping pieces and test your closed forms against *that*. A closed form checked against itself proves nothing.

**2b. ⛔⛔ THE STOP CLAUSE.** **A derivation that disagrees with a number the paper prints is a FINDING ABOUT THE PAPER — never a reason to adjust the algebra.** Stop, report, fix neither side. In both prior instances the clause never fired and the papers' constants were right all along; that outcome is only meaningful because the clause was real.

**Constants your derivations must reproduce** (all verified present on disk):

| target | printed value |
|---|---|
| the causal box | $L = T\varepsilon c/\sqrt{M_0} = 2.5$ at $T{=}100$, $\varepsilon{=}0.05$, $c{=}1$, $M_0{=}4.0$ |
| injection ratio vs bound, $\zeta = 0.25/0.5/1.0/2.0$ | ratio `1.13 / 1.55 / 3.79 / 27.5` against bound `1.65 / 2.72 / 7.39 / 54.6` |
| ⭐ the bracket predictions | $\zeta = \mathbf{2.0105}$ for $d{=}4.0$; $\zeta = \mathbf{2.6441}$ for $d{=}5.0$ |

⭐ **The bracket pair is your sharpest target.** The Advisor has independently reproduced both from $[L,\,L+p_0\sinh\zeta/M_0]$ with $L{=}2.5$, $p_0{=}1.2$, $M_0{=}4.0$ **and the landing tolerance $0.4$** — i.e. solving $d - 0.4 = L + (p_0/M_0)\sinh\zeta$. If your algebra reproduces them, the bracket is confirmed by a second instrument.

⚠ **A correction already in the paper that you must not undo:** energy is exponential in **rapidity** and only **quadratic in the excess distance**. The paper's own two $\zeta$ values imply an energy ratio of **3.55** from $d{=}4{\to}5$, against **3.64** quadratic and **7.39** exponential-in-distance. ⛔ Do not restate it as exponential in distance.

⚠ **P2's scope is already stated and you must respect it:** the $e^{2|\zeta|}$ bound is a **matched-quadratic-$H$** certificate; on the quartic well the raw ratio can exceed it, and the paper says so. ⛔ Derive it in that scope; do not prove more than the paper claims.

---

## 3. Footprint

- **The new appendix will be G.** The block currently runs **A–F** (A flag-provenance · B certificate table + BIBO · C evaluation grids · D negatives · E analytic verification · F Markov-kernel derivations).
- ⛔⛔ **NEVER hard-code the appendix letter into prose or proof numbering.** Use `\subsection` + `\label{app:deriv:n}` and refer with `\ref`. **Directly earned:** a sibling theorist hard-coded proofs as `D.1…D.8` in an appendix that rendered as **G**, while `app:vault` genuinely *was* Appendix D — seven cross-references pointed at the wrong appendix. After conversion, deleting an appendix later moved G→F and **nine references renumbered themselves for free.** ⚠ This paper is about to be ported into a venue template, which is exactly that kind of structural edit.
- ⭐ **At most TWO `\ref` insertions** into main text so the appendix is reachable — a sibling shipped a derivation appendix nothing pointed at. ⛔ **Nothing else in main text moves**: no rewording, no re-ordering, no re-flowing.
- ⛔ **Zero new numbers.** Every numeric token must have an ancestor in the paper. A value without one is a finding, not a licence.
- **Report main-text page count before and after.** Main text currently runs ~8.3 pp; appendices are excluded from TTCL's 4–9 pp, so appendix length is free and **main-text growth is not**.

---

## 4. Deliverables

1. **`PREREG.md`**, written before any script runs.
2. **The Appendix G block**, `\subsection` + `\label` per proof.
3. **≤2 `\ref` insertions**, each reported before → after with surrounding words shown byte-identical.
4. **`.claude/outputs/v1-derivation-appendix.md`** — the derivations, the numerical check table (every row against the composed map), mtimes proving prereg-first, whether the STOP clause fired, and ⭐ **a required section: what you could NOT derive from the paper's own stated assumptions.** That section is as valuable as the proofs; a sibling's flagged an undefined symbol and an unproven impossibility claim, and both became real findings.
5. **The three "theory note" sites listed with the exact rewording each needs** now that the note is cut — ⛔ **listed for the Head, not made.** The *"theory note proves"* sentence is the one that matters.
6. **Build**: 0 errors, 0 undefined references; page split reported.

## 5. Acceptance criteria

- `diff` shows **exactly**: one appendix block + at most two single-line `\ref` insertions. ⛔ Nothing else.
- ⛔ `.claude/papers/v1-short/**` and `submission.tex` byte-untouched (md5 manifest printed).
- Every check row is against the **composed map**, not the closed form.
- Prereg mtimes precede every result artifact.

## 6. ⛔ Prohibitions

1. ⛔ **Never invent a definition or a constant.** If a symbol is undefined, **say so and refuse** — a sibling theorist did exactly that and was right; the definition existed elsewhere and was found by search, not composition.
2. ⛔ **No main-text edits beyond the two `\ref` insertions.**
3. ⛔ **Do not adjust algebra to match a printed number** (§2b).
4. ⛔ **C-8 hermetic:** do not read the sibling shorts' drafts.
5. ⛔ Treat every C3-era number as PENDING. Nothing here should need one.

## 7. ⚠ Grep hazards

⛔ `grep` here is a shell function resolving to **`ugrep 7.5.0`**: on bounded-context patterns over long `.tex` lines it either **errors "exceeds complexity limits" and exits 0** — a silent false negative — or **hangs**. Use **`/usr/bin/grep`**; count with `grep -o … | wc -l`, ⚠ **never `grep -c`** (lines, not occurrences). ⛔ Directory-level grep over `.claude/` returns nothing (gitignored) — sweep per-file. ⚠ zsh does not word-split: quote any glob. ⚠ **Case-sensitivity has produced three false findings in this estate this session alone.** Positive-control every negative.

## DIAL DECLARATION
**Dials touched: NONE.** Derives algebra already implied by the paper, adds one appendix and at most two cross-references. No experiment, no configuration change, no new measurement.
