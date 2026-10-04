
## §I8.8 ⊙⊙ THE WAVE-14 CLOSEOUT ROWS — **the maze refusal display is RETIRED**, with the number that retired it and the bar to revive it (filed 2026-09-22 by `w14 spoke/refusal-displays-closeout`, `doc-curator`, fold symbol **⊙**; ⚠ CURATOR-PROPOSED; commissioned by `tasks/w14-spoke-refusal-displays-closeout.md` on the Head ruling at `outputs/advisor-wave-plan-w12-w14.md:56`; registry `N337`)

> ⛔⛔ **`CM-R12-3` IS NOT REPLACED AND NOT EDITED.** It said *"not yet informative"*. ⭐ **`CM-W14-1`
> below carries it the rest of the way to RETIRED**, adds the **bar to revive**, and is read together
> with it. ⛔ **No `CM-` number moved; no row above was touched.**

### **CM-W14-1** ⛔⛔ RETIREMENT ROW — **the maze refusal DISPLAY is RETIRED from the paper. ⭐ Here is the number that retired it, and here is what would revive it.** *(Registry: `N337`, retiring `N323` and `N-I8-19`; matrix companion `CM-R12-3`, which stays. Authority: `outputs/advisor-wave-plan-w12-w14.md:56`, Head-ruled 2026-09-21.)*

⛔⛔ **FORBIDDEN, ABSOLUTELY:** ⭐ any wave-9 maze refusal DISPLAY row **as a results-table row**, as a
**comparison**, or as **evidence that the object refuses**. ⛔ This covers **both** environments — the
`hub/maze` CLU rows (**CMAZE-2**, the request layer) and the `spoke/harness` refuser rows (**CMAZE-1
proper**) — and ⛔ **they still never share a table** (w9 AMENDMENT 2).

⭐⭐ **APPROVED WORDING — the retirement, stated as a FINDING and not as an apology:**
*"We built a refusal display on the maze and we retire it, because its only comparator is at chance on
the observation the arms themselves receive: a ridge probe on the shared observation at **t = 0**,
possible vs all impossible, scores **AUC 0.4822–0.5518 across all six seed × tier runs** at
**`n_test = 200`** per run, and the **budget scalar alone reaches only 0.3840–0.4589**, so there is no
shortcut either. ⭐ **Deciding solvability-within-budget is close to solving the maze — a property of
the task, not a wiring bug and not a defect of the model.** The supervised refuser's separation is
**+0.047 [0.000, 0.115]** (`cmaze1`) and **+0.020 [−0.005, 0.050]** (`cmaze1-easy`), both intervals
containing zero, and class weighting moves only the operating point."*
(`outputs/_thinkpad/w9/w9-harness/refusal_separability.json`;
`outputs/_thinkpad/w9/w9-harness/LOG.md:222-224,228-236`.)

⛔⛔ **THE NUMBERS THAT MAY BE QUOTED ONLY WITH THEIR `n`, because the registry carried them as bare
values and they are 3-SEED MEANS** (`N337` Part C, from
`outputs/_thinkpad/w9/w9-maze/runs/scored{1,2,3}_seed{0,1,2}_test.json` → `display.refusal.*`):
`budget_infeasible` **0.971429 / 0.969231 / 0.939655** (mean 0.9601) · `walled` **0.960573 / 0.984634 /
0.995221** (mean 0.9801) · `mt_unseen` **0.972077 / 0.979401 / 0.982919** (mean 0.9781) ·
`grounded_possible` **0.797945 / 0.855960 / 0.800633** (mean 0.8182).
⛔ **`0.8182` is the MEAN, never seed 2** — seed 2 is **0.800633**. ⛔ *"0.7979 / 0.8560 / 0.8182"* as
three seeds is **wrong** and must not be written.
⭐⭐ **The novelty control fires on EVERY seed**, which the mean alone could not show: `mt_unseen` is
**POSSIBLE** and is refused at **0.9721 / 0.9794 / 0.9829**, inside the IMPOSSIBLE conditions' own
per-seed range on all three ⇒ **the refusal is novelty detection, not wall detection**, with a
false-refusal floor of **0.7979** against a **pre-registered 0.20**.

⭐⭐ **THE BAR TO REVIVE IT — three conditions, and a fourth that is a trap** (`N337` Part D):
1. ⭐ **BINDING:** a comparator that **separates above chance on a probe of the SAME observation** —
   the ridge probe must clear **0.50 on every seed × tier run**, `n_test` printed, ⛔ **declared before
   any DISPLAY row is scored.** ⛔ Today **0.4822–0.5518 at `n_test = 200`** — ⚠ **only 3 of the 6
   point estimates are above 0.50**, so **3 runs fail the bar outright**, and `N323` reads the spread
   as chance.
2. ⚠ **AND** the refuser's separation interval must stop containing 0.
3. ⚠ **AND** the CLU's `mt_unseen` must separate from `walled`, with false refusal on
   `grounded_possible` **below 0.20**.
4. ⛔⛔ **THE TRAP:** reviving it by making impossibility **perceptually detectable** is **choosing the
   answer**. w9 said so itself; the legality probe already reaches **0.8474** on `cmaze1/seed1`
   *because extra tints change the image*. ⛔ **Any wave that takes this route must label the impossible
   family a DESIGNED giveaway, in the row.**
⭐ **1 alone re-opens the comparator. 1+2+3 re-opens the display row.**

⛔ **NEVER DELETE THE RETIRED ROWS.** `N323`, `N-I8-19` and `CM-R12-3` stay at their own numbers and
are **marked retired in place**. ⭐ The negative is the useful object.
⭐ **WHAT RETIREMENT DOES NOT TOUCH, and may still be stated** (as an instrument property, ⛔ never as a
refusal result): **0** impossible instances in any non-refuser arm's data across all seeds and both
tiers (`assert_unseen`), and **0** certification failures over **2700** impossible instances
(`certify_impossible`).

### **CM-W14-2** ⭐⭐ SCOPE ROW — **the Shapes3D refusal-via-the-HOLE is UNTOUCHED by the maze's retirement. ⛔ Do not generalise.** *(Registry: `N337` Part F, with `N-I8-6`, `N332` and `N334`.)*

⛔⛔ **FORBIDDEN:** *"refusal does not work"*, *"the refusal mechanism is retired"*, or any sentence that
lets a reader carry the maze's failure onto the store's hole. ⛔ **They are different mechanisms, on
different objects, on different environments.**

⭐⭐ **APPROVED WORDING:**
*"The maze refusal display is retired. ⛔ **The Shapes3D refusal mechanism is not, and nothing measured
in wave 9 or wave 14 bears on it.** The maze display is a **learned** refusal score on a route request;
the Shapes3D refusal is a **structural hole** in the request count / coupling table. The hole refuses
`red ∧ cylinder` at **1.0000** for **ONE byte**, with **0 bytes changed outside the permitted range**,
**no refit**, and costing **EXACTLY 0.0000** composition, while all **101** other never-held
conjunctions are untouched, at **zero seed spread**; at wave 12 **two** holes — `red ∧ cube` and
`red sphere` — cost **1 byte each, 2 bytes total, 0 data rows read, no refit**, over an index space
**asserted** equal to `gen_field`'s (`layout_equals_gen_field: true`, **1,322 pair slots, 60 indices
checked**). ⭐ **`red ∧ cube` is refusable although no red cube was ever written** — refusal is a
separate mechanism from forgetting and does not need a deleted row to have existed."*
(`N-I8-6`, from `outputs/w5-advisor-report.md` §2 and `outputs/w4-advisor-report.md` §4; `N332`, from
`outputs/w12-single-design-crossover/FINDINGS.md:294-300,455-462`.)

⚠⚠ **AND THE HONEST OTHER HALF, printed with it or not at all — *"untouched" is not *"complete"*:** the
Shapes3D display row's **comparator is a DECLARED NOT-RUN** (`N334`, `CM-W12-9`): the refuse-token
transformer `transformer__refuse_token__{value,conj}` is present in `BASELINES-READY.json` and the wall
clock ran out after the scored rows and the deletion legs
(`outputs/w12-single-design-crossover/FINDINGS.md:464-466,482`).
⭐⭐ **THE DISTINCTION THAT DECIDES BOTH ROWS, and it is the sentence to keep:** ***the Shapes3D gap is a
BUDGET not-run, which a wave can close by spending; the maze's is a TASK property, which spending does
not close.*** ⛔ That is why one is retired and the other is not.

---

## §I8.9 ⚠ OPEN FLAGS FOR THE HUB — the w14 refusal-display closeout (⛔ raised, deliberately not resolved by the curator)

1. ⚠ **`CM-R12-3` now has a successor but keeps its own wording.** A reader who loads only `CM-R12-3`
   gets *"not yet informative"* and **not** *"retired"*. ⛔ **The curator did not edit it.** ⭐ **The
   Hub's call:** leave the pointer-only arrangement, or have a later fold add a one-line in-place
   `⊙` note to `CM-R12-3` as well.
2. ⛔⛔ **A number was quoted without its `n` in three places at once** — `N-I8-19`, `CM-R12-3` and
   `outputs/w9-advisor-report.md` all print `0.8182` (and three companions) as a bare value when it is
   a **3-seed mean**. ⭐ The registry and matrix copies are corrected here. ⛔ **The w9 advisor report is
   NOT this spoke's file and was not edited** — the Hub decides whether it takes an amendment.
3. ⚠ **The task file that commissioned this spoke carries the same defect** (*"false refusal 0.7979 /
   0.8560 / 0.8182"* as three seeds). ⭐ Recorded here rather than inherited; ⛔ the task file was not
   edited.
4. ⚠ **`N337`'s tier is CURATOR-PROPOSED A** and `v2.21` is **PROPOSED**. ⛔ The curator never
   self-confirms, and ⛔ **`v2.20`, `v2.19`, `v2.18` and below stay PROPOSED for their own Hubs.**
5. ⚠ **The revival bar in `N337` Part D is a CURATOR DRAFT of a bar, not a ratified pre-registration.**
   ⭐ It is stated so a future wave can be held to something; ⛔ **only the Head can make it binding.**
6. ⚠ **`N-I8-21`(b) touches this pass and is left where `w13` left it.** The w9 refuser is an
   instrument whose own verdict about itself is the finding — ⭐ **caught by its own spoke, against its
   own interest**, which is the healthy form. ⛔ Not filed as a seventh instance; the promotion
   question remains the Head's (`outputs/w13-registry-fold/negative_results.md` `⊛.4`).
