# rival-recon — the neural-memory rival map + the Track-2 fairness checklist

**Campaign 2, wave C2W1. Agent:** web-scout. **No worktree, no code, no repo edits** (writes only under
`.claude/`). Charter §2.5 verbatim. **⭐ THIS TASK RUNS FIRST. Nothing in Track 2 freezes before it
lands** (charter §2.5) — the harness design, the baseline set, and the byte-matching convention are
all downstream of your brief.

Read first: `.claude/AGENT_PROTOCOL.md`, `.claude/advisor-head-c2-charter.md` (§2.2, §2.5, §6.3),
`.claude/advisor-head-intervention.md` (§6 benchmark criteria — **all five are binding**), then this file.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** none — recon. No performance claim is made or measured by this task.
- **Laundering control:** n/a (no measurement). But **you must report, for every rival, whether ITS
  memory is metric-native** — that is the laundering exposure of the *rival*, and it is a deliverable
  (§3 below), not an aside.
- **Falsifies the deliverable:** the fairness checklist is falsified if it cannot state, for each named
  rival, (a) its state-byte accounting convention and (b) its metric-native status, **each traced to a
  primary source with a section/equation number**. A convention you cannot pin from a primary source is
  reported as **UNPINNED**, never inferred, never averaged across papers.
- **Does NOT falsify:** finding that a rival is *not* metric-native (that is information, and it makes
  the fight harder and fairer). Finding that our positioning is already occupied (say so plainly —
  w26's `deletion-prior-art` found Blelloch–Golovin own our deletion algorithm outright and that report
  is one of the wave's most valuable).

## Why this task exists
The program has never fought the modern neural-memory family. Twenty-six waves of laundering controls
have taught us that the *trivial* substitute usually wins; we now have to find out what the
*non-trivial* substitutes actually do, on what conventions they report, and where their own memory is
metric-native (i.e. where a classical method is *their* provable ceiling too). Get this wrong and
Track 2 is either unfair to them (a referee kills it) or unfair to us (we hand them our state budget).

## Deliverable 1 — the write-mechanism map
For each family below: **what is written, when, into what, and with what update rule.** Equations, not
prose summaries. Cite section/equation numbers.
1. **TTT / test-time-training memories** — the inner-loop objective, what the "memory" parameter set is,
   how many inner steps per token/chunk, and what is reset between sequences.
2. **Titans-class test-time memory** — the surprise/momentum/forget-gate write rule, the
   **chunk granularity** (charter §2.2 leans on this being standard practice — verify it, with the
   citation, or report that it is not), and the persistent/long-term/short-term split.
3. **Fast weights / Hebbian outer-product memories** (incl. the linear-attention-as-fast-weights
   equivalence) — the write rule, the delta-rule variants, and the capacity claim each makes.
4. **Mamba / SSM state** — for completeness, since **real Mamba is a standing rule for any SSM claim**
   (handover §10, carried). Where is the reference implementation, what does its recurrent state
   consist of, and what is the state-byte count as a function of (d_model, d_state, expand, n_layer)?

## Deliverable 2 — task definitions, metrics, and tuning conventions
**MAD / zoology / MQAR**: the exact task generators (vocab, kv pairs, sequence length, query
placement), the reported metric, and — critically — **the baseline-tuning convention** each paper uses
(lr sweeps per architecture? matched params? matched state? matched FLOPs? how many seeds?). We have
been burned once already by an un-rescued baseline (**N78**: an un-rescued GRU turned a 2.1× margin
into a quoted "19×" — never quotable). ⭐ **Report the specific tuning protocol we must adopt to make
our GRU/attention/Mamba baselines rescued-by-their-own-standard**, with the citation.
Also: **enwik8 / WikiText small-scale conventions** — what "small scale" means in this literature
(params, tokens, context), what BPC/PPL numbers a from-scratch laptop-class run should expect, and
what the community treats as a non-embarrassing floor.

## Deliverable 3 — the metric-native audit (the sharpest part of this task)
Intervention §6 criterion 4 is a **theorem about our situation**, confirmed four times: *if the query
lives in the same metric space as the stored keys, a classical method is the provable ceiling.*
For each rival memory, answer: **is its read a similarity computation in the key metric?** If yes, the
rival is metric-native too, and a kNN/lookup launder is a ceiling for *them* — which is either a shared
weakness (so the fight is on other axes) or an argument we must pre-empt. Give the answer per family,
with the mechanism that decides it.
Then: **for each candidate Track-2 task, does the task itself admit a metric-native classical ceiling?**
Any task that does is **inadmissible as a primary claim** (§8.4, non-negotiable).

## Deliverable 4 — the failure modes they publish
What does each family cite as its own limitation? (state saturation, forgetting behaviour, length
generalization, compute per token, recall-vs-length curves). This is where our four pillars have to
land to be interesting, and it is also where a referee will look for our missing baseline.

## Deliverable 5 — THE FAIRNESS CHECKLIST (the artifact that gates Track 2)
A checklist the harness designer executes, one line per item, each traceable to Deliverables 1–4:
- **matched parameters** — what counts, what is excluded (embeddings? readout?), per the literature's
  own convention.
- **matched state-bytes** — the accounting formula for each baseline **and** the corresponding formula
  for a CLU store (atoms × dims × dtype + codebook + controller state). ⭐ If the two conventions are
  not commensurable, say so and propose the honest reconciliation — do not paper over it.
- **baseline tuning** — the minimum sweep each baseline must receive before any comparison is quoted
  (N78 rescue standard).
- **chunk granularity** — is chunked memory-update standard practice in this family, yes/no, cited.
  Charter §2.2 rests on this being fair; if it is not standard, that is a finding the Head must hear.
- **seeds and reporting** — the community norm, versus our multi-seed-before-any-paper-number rule.
- **the anytime curve** — does any rival draw an accuracy-vs-compute curve at fixed weights? If one
  does, the "signature figure no baseline can draw" (charter §2.2) needs re-wording, and you must say so.

## Constraints and traps
- **Primary sources only for every number and convention.** Blog posts and secondary summaries may
  orient you; they may not be the citation. Mark preprint-grade evidence as preprint-grade (w26
  precedent: the HNSW soft-delete reconstruction claim).
- ⛔ **Do not coin a benchmark name or a term of art.** w26 ruling: adopt existing vocabulary
  (Ginart Def A.5 / Sekhari Def 3 precedent). If we need a name, propose it as a question for the Head.
- ⛔ **Never-quote list, carried:** "certified" · "unlearning" · "deletion-compliant" · unqualified
  "exact deletion" · "our fix-up cascade" as a possessive (it is Blelloch–Golovin's) · the "19×/17×
  chance" MQAR figure (N78) · "the write operator is the ceiling" · width-lock-as-cause · the √2 / d^1.62
  exponent · any `K_learned` at `pscale ≠ 1` without the payload-noise condition.
- **Novelty positioning:** flag anything that preempts *any* of the four pillars (charter §4) —
  expressive latents (trajectory/manifold as the stored object), structured exploration
  (wormholes/boosts), physics-intuited hyperparameters (zero-free-parameter predictions), explicit
  memory + principled forgetting. **A preemption found now is cheap; found by a referee it is fatal.**

## Deliverable location & format
`.claude/outputs/rival-recon.md`, standard §5 format, plus a `## Fairness checklist` section written so
it can be lifted verbatim into the Track-2 harness task file. Cite inline. State confidence per claim
(pinned / inferred / UNPINNED). Reconciliation list — anything in our docs this brief contradicts — in
the **first 10 lines**.
</content>
</invoke>
