# controller-doctrine — 13 collapse modes → monitor invariants → productive bands → controller verbs

**Campaign 2, wave C2W1. Agent:** physics-theorist. **No worktree, no production-code edits**
(numerical sanity checks live in `.claude/scratch/controller-doctrine/`). **Cheap task** — this is a
formalisation pass, not a research campaign. Charter §6.5, formalising §3.1–3.2.

Read first: `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-intervention.md` (**§5 — the 13 modes
are the whole input to this task**), `.claude/advisor-head-c2-charter.md` (§3), your own prior report
`.claude/outputs/clu-controller-spec.md` (**this task supersedes/extends it — you are amending your own
C1 work, not restarting it**), and `.claude/negative_results.md` for the source N-entries named below.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial (C2 form):** none — doctrine/theory. This task makes **no performance claim** and measures no
  benchmark. Its output is the specification the harness implements against.
- **Laundering control:** n/a. ⚠ But one of the 13 modes (**#2, settle → arg-min**) *is* the laundering
  control promoted to a runtime monitor — treat its invariant as the most important one you write.
- **Falsifies the deliverable:** (a) any of the 13 modes has **no invariant computable at runtime from
  quantities CLU actually has** — then it is not a guard, and you must say so explicitly and name what
  the system would need to acquire for it to become one; or (b) **two modes' productive bands are
  provably disjoint** — then staged activation (charter §3.1) is impossible as specified and the Head
  needs a ruling. **Pre-register which modes you expect to fail (a) before you work through them.**
- **Does NOT falsify:** a band that is narrow, or one that can only be stated empirically (a measured
  band from the 26-wave map is a legitimate band — say "measured, not derived" and cite the wave).
  Needing a monitor that costs a diagnostic pass (cost is a design input, not a refutation).

## The deliverable, in one sentence
**A table with 13 rows and five columns, plus the propositions that justify the hard entries.**

| # | collapse mode | monitor invariant | productive band | restoring verb | source |

- **monitor invariant** — an explicit function of runtime observables (state, landscape, codebook,
  stream statistics) with its **trip predicate**, its cost, and its **false-trip mode** (what benign
  situation fires it — a monitor that cries wolf is a monitor that gets disabled).
- **productive band** — the interval/region in which the lever is known to be productive, with its
  provenance (**measured at wave n** / **derived**). Where a band is derived, give the derivation.
- **restoring verb** — which of the designed verbs {admit, place, evict, decay, route, retry, stop}
  restores the invariant, **and in which direction**. If no verb in the set restores it, that is a
  finding: name the verb the action space is missing.

## The 13 rows (intervention §5) and what is already known about each
The Hub has drafted a **provisional** monitor table into `full-clu-harness` §Monitors so the engineer is
not blocked. **Your spec supersedes it at wave review** — treat the provisional table as a strawman to
correct, and say per row whether you *confirm*, *sharpen*, or *replace* it.

1 overdamping → "the last observation" (C17-3, N85: corr(q\*,q_last)→0.97, and γ won by turning the
physics off) · 2 settle → arg-min (w26 same-keys launder beats CLU 6/6 — **the mode that defines the
dividend**) · 3 vacuous gate (N74: spacing 1.414 vs d_safe 1.10; and N91's finding that **the address
space, not the controller, was the binding constraint**) · 4 blank controls passing (N68: blanks
0.992–1.000; and w26's **`tol` metric vacuous at m>1**, blank scores 1.0000) · 5 learned addressing dies
(w19 0/18, 4.2%; ⭐ and w26's mechanism result — the whole annealed-read gain is **address acquisition**,
`readonly` reproduces baseline to 4 dp) · 6 objective/goal divergence (w25/w26: write loss → 0 while
retrieval fails) · 7 mass stores nothing (Prop F1, ×3 — an **exact gauge** verified 6.2e-16; the
invariant here is a gauge assertion, and it should be a *test*, not a runtime cost) · 8 learning erases
design (w20; the C1–C5 / N1–N4 certificates from your own spec are the natural invariant) · 9
payload-dependent lifetimes (w25 r=−0.85; N108 — and the recommended fix, option (d) gated stiffness,
is **C1W27's to build; C2 monitors only**) · 10 degenerate axes / silent knobs (N19/N58; read-mode axis
dead at `clu_steps=1`) · 11 reach failure (⭐ **superseded by your own saddle criterion on
`L=√(|c|²+a²)`, verified 31/32 on the shipped trained `V` with zero free parameters** — this row should
become the strongest in the table; reach is *logarithmically un-buyable*, κ 4→5 costs 55× depth) · 12
starve-and-overwrite (w26: naive sequential/masked writes give each item `atoms/K`) · 13 under-trained
artefacts (N94 — this one is a **provenance field**, not a runtime trip; say so if you agree).

## Required propositions (the parts that need proof, not tabulation)
**P1 — the compatibility question.** Do the 13 bands have a **non-empty simultaneous intersection**?
This is the formal content of "staged activation, not big-bang" (charter §3.1). Attack it honestly:
identify every *pairwise* tension you can (the obvious candidates: #1 damping vs #11 reach; #2
non-separability vs #8 design-preservation; #3 admission strictness vs #12 starvation; #9 lifetimes vs
#2's requirement that basins interact). For each tension, either exhibit a joint band or prove it empty.
⚠ A proven-empty pair is **the most valuable output this task can produce** — it converts a whole
campaign's engineering risk into one ruling.

**P2 — the verb-completeness question.** Is {admit, place, evict, decay, route, retry, stop} **complete**
for the 13 modes? Name any mode with no restoring verb and propose the minimal addition.

**P3 — the trigger-ordering question.** When two monitors trip in the same step, which verb fires first?
Give a **partial order with a justification**, not a heuristic. (Precedent from your own spec: deletion
is the one irreversible verb and gets a persistence window + hysteresis; the deadband is verified free
and necessary; re-derivation must be by relaxation with a λ_min > 0 check, **never** a critical-point
solver — C4.3, which caught a saddle written into a codebook.)

**P4 — the designed/learned boundary.** Charter §3.2: the verbs and guards are **designed**; *when* and
*how hard* to pull them is **learned**. State this as a formal split: for each verb, what is the designed
invariant it must never violate (a hard constraint on the learned policy), and what is the free
parameter the policy sets. ⭐ **This is the mechanism that prevents relearning w20's lesson at the
controller level** — free learning erased the structure a designed landscape provided, and the same
failure at the controller level would look like a controller that learns to never fire its guards.

## Numerical sanity checks (only where they settle something)
Pure numpy/jax/sympy in `.claude/scratch/controller-doctrine/`, as in your C1 spec. Do not import repo
code into the checks unless a check is *about* shipped behaviour. Candidates worth the time: the
pairwise band-tension checks in P1; the trip/false-trip rate of any invariant you newly propose; the
gauge assertion for #7. **Do not re-verify what your C1 spec already verified** — cite it.

## File ownership
**You own:** `.claude/outputs/controller-doctrine.md` + `.claude/scratch/controller-doctrine/`.
⛔ **You edit no tracked code at all.** If a monitor requires an observable the code does not expose,
that is a line item in your **implementation requests** section — the engineer implements it, not you.
⛔ **Do not touch** `.claude/outputs/clu-controller-spec.md` (C1 artifact — supersede it *by reference*
from your new file, do not rewrite it in place).

## Deliverable
`.claude/outputs/controller-doctrine.md`, protocol §5 format. Must contain, in order:
(1) the pre-registered list of modes you expected to fail criterion (a); (2) **the 13-row table**;
(3) P1–P4 with proofs or explicit "measured, not derived" labels; (4) **implementation requests** — the
observables the harness must expose, one line each, written so the engineer can work from them directly;
(5) a **diff against the Hub's provisional table** (confirm / sharpen / replace, per row); (6) the
reconciliation list in the first 10 lines.

⛔ **Do-not-quote, carried:** "the write operator is the ceiling" · width-lock-as-cause · "~32,
d-independent" as settled · the √2 / `d^1.62` exponent · any `tol`-metric number at m>1 · decay-reduces-
distinguishability *per se* (N108: white-box AUC 1.000 at all 18 levels) · "certified" / "unlearning" /
"deletion-compliant" / unqualified "exact deletion".
**Standing:** quote the curve, not the endpoint.
</content>
