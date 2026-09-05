# Task: primitive-harness — evaluate CLU as a general primitive against MLP / GRU / Mamba / attention (w20)

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/primitive-harness.md` · **Branch:** `agent/experiment-engineer/primitive-harness`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · `.claude/outputs/clu-retrieval-demo.md` · `.claude/outputs/scout-dynamical-memory-priorart.md` · the CM-8 / MQAR machinery already in the repo (`regime-remap-2000ep` used it — reuse, do not rebuild)

## Why
**The program's framing is that CLU is a general AI primitive** — at the same level as MLP / GRU / Mamba / attention / DeepSVDD, *not* a special case of one, and a substrate on which special cases (equivariant variants etc.) can later be built. That framing dictates the evaluation: **drop CLU into a standard architecture slot and compare against other primitives on the same harness at matched parameter budget.** A single-benchmark win does not support a primitive claim; a primitive has to be *general*.

This task builds the harness. It does **not** need CLU to win yet — it needs the comparison to be **fair, reusable, and honest**, so that every later result lands in a frame reviewers accept.

## Item 1 — the drop-in slot
Define a single interface such that CLU, an MLP, a GRU, a Mamba/S4 block, and a self-attention block are **interchangeable** in the same model, with everything else (embedding, head, optimizer, schedule, data) held fixed. Report the interface and what had to be conceded to make CLU fit it. ⚠ **If CLU cannot be made drop-in without special-casing, that is itself a finding about the primitive claim** — report it plainly rather than engineering around it.

## Item 2 — matched budgets, honestly
Match on **parameter count**, and report **wall-clock and FLOPs separately** — do not hide a compute advantage inside a parameter match. Per Head's ruling, **training speed/efficiency is explicitly NOT a first-paper requirement** and CLU is expected to be slower; the requirement is that the cost is **stated**, not that it is competitive. Report the honest multiple.

## Item 3 — task families (≥2, ideally 3)
1. **Associative recall / selective retrieval (MQAR-style)** — the primary diagnostic. This is where CLU's demonstrated strengths live: exact retention (durability 1.000 out to 1200 steps), retrieval as a fixed point rather than a decaying trace, and **Prop 2 hard read isolation** (a sub-barrier particle *cannot* read a foreign item; softmax mixes every key).
   ⭐ **The headline figure to build: recall accuracy vs number of distractors / sequence length.** The structural prediction is that attention degrades from interference and CLU does not, because of barrier confinement. **This is a claim with a proof behind it (Prop 2, verified within one grid cell) and it cannot be explained away by parameter count.**
2. **A sequence-modelling family** where HiPPO/S4/Mamba are the natural competitor — this is the comparison the primitive framing makes central.
3. **Optional third** if cheap.
⚠ **Report per-family results separately and never average across families.** A primitive that wins one and loses two is a primitive that wins one.

## Item 4 — the baselines must be real
Tune the baselines at least as hard as CLU and **report the tuning budget spent on each**. ⚠ The program has a standing lesson here: on voraus, CLU scored 0.51–0.62 against LOF 0.81 / kNN 0.75, and w19 found the best CAFE config was the one with the *least* dynamics. **A weak baseline is worse than no baseline** — it produces a number we later have to retract.

## Acceptance
The drop-in interface, the matched-budget table with compute cost stated, per-family results with real baselines and their tuning budgets, and the recall-vs-distractors figure. Tests green.

⚠ **Do not tune toward a CLU win.** This harness will be used for every subsequent claim in the flagship; its value is entirely in being trustworthy. If CLU loses everywhere, report that — it redirects the program, which is worth more than a flattering number. **Do not quote** the "8-item ceiling" as CLU's capacity (it is a 2-D-ring artifact; `address-space-dimension-scaling` is measuring the real one), and note that capacity may bound what recall lengths are honestly attemptable — **coordinate with that task's result rather than working around a ceiling.**
