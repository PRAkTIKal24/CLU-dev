# WAVE 14 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-22. Three agents, FORGIS_14, `base/iter8 = 6699665`.**
All finished. **Scored spend: 3/3 on `read-verb`** — the target is now exhausted.
**Zero GPU. Local CPU only.** Record: `.claude/outputs/w14-*`.
⚠ Every number is quoted from an agent's own artefact. ⛔ **The Hub re-derived nothing.**

---

## 0. The scoreboard

| agent | verdict | scored |
|---|---|---|
| ⭐⭐ `hub/read-verb` | **THE VERB IS THE LEVER — confirmed, 3 seeds.** ⛔ **And it belongs to the INTERFACE, not the store** | **3/3** |
| ⭐ `spoke/request-layer-fix` | **rule B now bites: multi-key 14.3 % → 100 %, blind-seq 0.91 → 0.18–0.32** | 0 |
| ⭐ `spoke/refusal-displays-closeout` | maze refusal displays **retired with a real revival bar** | 0 |

**Budgets after w14:** ⛔ `read-verb` **exhausted** · `encoder-adapt` 2 · `generation-table` 1 ·
`encoder-loop` 2 (one reserved for BAR D) · ⛔ `single-design` and `maze` exhausted.

---

## 1. ⭐⭐ `hub/read-verb` — wave 9's diagnosis was RIGHT, and the claim it earns is narrower than hoped

> ⭐⭐ **Moving the composition of two half-addresses from LATENT space to PLAN space — on IDENTICAL
> latents, through the IDENTICAL frozen decoder — lifts held-out `unseen_pair` validity from
> 0.0704 / 0.1313 / 0.1298 to 0.5126 / 0.6667 / 0.5769.**
> **+0.4422 / +0.5354 / +0.4471 — a 4.4× / 5.1× / 4.4× multiplication**, under the unchanged exact
> integer judge, **three seeds, all or none.**

⭐ The ablation is the tightest possible: `AVG` = `decode((Z_A + Z_B)/2)` — **w9's read on the
identical two latents**. ⛔ Nothing about the store is a dial this wave: w9's own scored configuration
carried unchanged (`n_atoms=31, K=8, ridge=0.01, ae_steps=12000, LATENT_D=256`).
⭐ It also moves the other rows: `n-way` **0.0000 / 0.0160 / 0.0000 → 0.4891 / 0.4652 / 0.3915**.

### ⛔⛔ AND THE HUB ARGUED AGAINST ITS OWN HEADLINE — this is the finding
**The same fixed merge lifts EVERY arm, and it lifts the matched-bytes LOOKUP FLOOR more than twice
as much as the CLU:** mean **+0.8299** against the CLU's **+0.4749**, carrying that floor to
**0.9648 / 0.9646 / 0.9808** — ⛔ **past the CLU by ~0.4, and past the single-decode pipeline ceiling
itself.** ⛔ **The charter-C controls do NOT collapse ⇒ NO ROW IS CLAIMED FOR THE STORE.**

⭐ The hub's own formulation, and the Hub endorses it verbatim:
> ***"`MERGE` and `AVG` consume the same two latents; a verb claim is not a store claim, and only the
> verb claim is made."***

⇒ ⭐⭐ **The composition operator is part of the INTERFACE and it is TASK-SHAPED.** Images compose by
mixing latents; routes compose by overlaying constraints. ⭐ **This FITS the interface-split claim
rather than contradicting it** — but ⛔ **it is not a store result, and the paper must not make it one.**

⭐ **This is exactly the outcome the brief pre-declared as still-a-result:** *if a baseline plus the
same merge also reaches the ceiling, the CLU's differentiator in the maze is DELETION* — which
already transferred (7.5e-16 / 8.8e-16 / 8.2e-16, projector 0.000e+00, replicated from a clean
checkout with zero numeric differences).

⭐ **Discipline:** three ledger rows written **BEFORE** each run, each naming the PREREG ·
`chlu.__file__` asserted against `w9_maze`, `w8_envscout` and `w14_request_layer` **before any
number** · no other agent's session, worktree or process touched · no box-wide kill · zero GPU.

---

## 2. ⭐ `spoke/request-layer-fix` — the environment defect is fixed, and the fix is measured

⛔ **The defect (w9):** 1,242 of 1,449 routes used **exactly one key**, so rule B had no key ORDER to
trap — which is why blind-sequential read **0.9077 / 0.8905 / 0.9130** on w9's split.

| acceptance condition | before | after |
|---|---|---|
| fraction of routes needing **≥ 2 keys** | 0.1428 | ⭐ **1.0000** (2 keys ×3825, 3 ×751, 4 ×35) |
| **BLIND-SEQUENTIAL**, worst held-out row | 0.9130 | ⭐ **0.1810 – 0.3188** |
| **rule-B bend fraction** | — | ⭐ **0.9536** |
| **ORACLE** | 1.0000 | ⭐ **1.0000** |
| `tolerance_used` | null | ⭐ **null** |

⭐ **28 of 28 independent verification checks pass.** **0 scored runs.**
⭐ **It named the result `CMAZE-2b`** — *"the CMAZE-2 request layer with ONE acceptance rule added"*,
engine, solver and exact judge imported unmodified — ⛔ rather than silently reusing "CMAZE-2".
⭐⭐ **And it explained why it lands BELOW 0.4980 rather than claiming a bullseye:** 0.4980 is a
**mixture** over CMAZE-1's own key counts (73.6 % one-key at 0.6532, the rest at 0.1351 / 0.0000);
CMAZE-2b is **100 % multi-key**, so it sits at CMAZE-1's **multi-key band**, not its marginal.
⚠ **Declared cost:** fixing the layer made routes longer, so the budget grid is re-derived **per
seed** (s0 55/85/105, s1 60/85/100, s2 55/75/90). Both grids are reported on VAL.

## 3. ⭐ `spoke/refusal-displays-closeout` — retired, with the line drawn where it matters
⛔ **The maze refusal displays are RETIRED** (Head ruling), and the measurement that forces it is on
the record: separation **+0.047**, and ⭐⭐ **a ridge probe on the same observation at t=0 scores
AUC 0.48–0.55 — CHANCE — across all six seed × tier runs**, the budget scalar alone 0.38–0.46.
⇒ ⭐ **Deciding solvability-within-budget is close to SOLVING the maze** — a property of the task,
⛔ not a defect of the model.
⭐⭐ **It drew the line that stops the misreading:** the retired thing is a **LEARNED refusal score on
a route request in CMAZE-2b**; the **Shapes3D coupling HOLE is UNTOUCHED** — a **structural** hole at
**1 byte per hole, 0 data rows read, no refit**. ⛔ Different object, different environment.
⭐ It also refused a soft bar — *"an option is not a bar"* — and wrote a revival condition a future
wave can be held to, after auditing that the old entry had **none**.

## 4. Limits
`read-verb` is **exhausted at 3/3** — ⛔ **no budget remains behind this result** · the merge is a
**fixed operator**, and a tuned one would not carry a row · the environment is **CMAZE-2b**, ⛔ which
never shares a table with CMAZE-1 or with w9's CMAZE-2 · the budget grid is **per-seed**, not one grid ·
⛔ **no row is claimed for the store on this wave.**

## 5. What the Advisor is asked to rule on
1. ⭐⭐ **The verb claim is real and the store claim is not.** ⛔ **Does the paper carry the merge as an
   INTERFACE contribution — "composition is task-shaped, and the operator belongs to the interface" —
   given the lookup floor gains MORE from it than the CLU does?**
2. ⛔ **The matched-bytes lookup reaches 0.96–0.98 with the same merge, past the single-decode
   pipeline ceiling.** **Is that a headline negative the paper must print, or an appendix row?**
3. ⭐ **Wave 9's diagnosis is now confirmed on a held-out estimand.** **Does the maze section become
   "the read verb, diagnosed and fixed", with deletion as the CLU's only maze differentiator?**
4. ⚠ **`read-verb` is exhausted and so are `single-design` and `maze`.** **If a follow-up is wanted,
   it needs a new target — which is the Head's call, not the Hub's.**
