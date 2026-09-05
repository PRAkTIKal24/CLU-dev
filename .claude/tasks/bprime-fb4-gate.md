# bprime-fb4-gate — FB4 first: is the protocol measuring the memory, or the task? (+ the two harness-debt riders)

**Campaign 2, wave C2W3. Agent:** experiment-engineer (**small on FB4, medium with the riders**).
**Worktree MANDATORY** — you hold **B′ worktree slot 1 of 2** (charter §A9.3).
Base local `main` (`6ff4c1d` at scoping). Branch `agent/experiment-engineer/bprime-fb4-gate`.
Charter **ADDENDUM 2 §A11 task 1** (*"runs first"*), implementing **§A9.1** (FB4 before rivals) and the
**harness-debt riders** named at the foot of §A11 (`store_write_mask_factory`; SC-1…SC-7 per §A9.8).

**⚖ YOU ARE THE WAVE'S FIRST RUN.** `PREREG-Bprime.md` names FB4 as *"the first thing B′ should run —
it is cheap and it validates the protocol before it is spent on six families."* `bprime-rivals` is
**gated on your D0 verdict** and will not be spawned until you report it.

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **in full, especially
ADDENDUM 2 §A7 (the gate verdict), §A9.1/§A9.8 (your two rulings) and §A11**;
`.claude/advisor-head-intervention.md` **§5 (the 13 modes), §6 (the five criteria), §8 (prohibitions)**;
the **live `2026-07-31` `[C2W2]` §10 entry** in `.claude/handover_context.md` (the 11 reconciliations —
**#5 is yours** — the NOT-RUN list, and the never-quote additions);
`.claude/outputs/track2-admissibility/PREREG-Bprime.md` **§6 FB4, verbatim — it is your acceptance
criterion**; `.claude/outputs/memory-gym-v0.md` (the gym you measure on);
`.claude/outputs/doctrine-repairs.md` **§4.4 (SC-1…SC-7, your rider D2's spec, verbatim)**;
`.claude/outputs/traj-write-objective.md` **§5 (the recency defect fix you must switch ON)**.

⛔ **REGISTRY LAG — THREE WAVES (C1W27 · C2W1 · C2W2).** `negative_results.md` (N122),
`claims_matrix.md` (v2.5), `research_roadmap.md` (v0.9), the ledger (⟲ w26) and the primers are **three
passes behind**. Quote results **only** from `.claude/outputs/*` and the §10 review entries — **never**
from the matrix, the registry or the roadmap, and **never infer from a registry's silence that a result
does not exist.** `doc-curator-c1w27-c2w1-sync` (the unparked 3-wave pass) runs in parallel with you;
do not wait for it. If the Hub tells you mid-task that the curator has landed, you may quote the
registries from that point on — **not before, and not on your own judgement.**

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **none — instrument validation + harness infrastructure.** You make **no**
  performance claim, enter **no** leaderboard, and claim **no** dividend. D0 measures whether the
  *protocol itself* is valid; D1/D2 are debt that blocks other people's work.
- **Laundering control:** you are *building the control's audit*. Every family carries its **+0 B
  substitute**, its **settle-deleted / matched-bytes launder**, its **blank-store control**, and (new)
  a **full-attention arm**. Identical φ and φ-bytes on every arm, ledgered (enforced in code:
  `assert_identical_phi` / `PhiMismatchError`, landed C2W2).
- **Falsifies:** §5. **Does NOT falsify:** an individual family being substitute-saturated — that is a
  *finding about that family*, and the pre-registered rule below says exactly what happens to it.

---

## 0. Why this task exists, in four sentences
The C2W2 gate fired negative and B′ (the audit paper) is the wave's spine: *"when does test-time
dynamics buy anything over a table at matched bytes?"*, one protocol applied to CLU **and** to the
TTT/Titans/Delta family. Before that protocol is spent on six rival families, it has to be shown to
measure the **memory** and not the **task**. C2W1's substitute audit already went **0-for-4** — a **+0 B**
substitute of the launder's own table matched or beat the CLU on every family (insertion order **0.776**
vs 0.302; echo **1.0000** vs −0.180) — and a protocol on which *every* reader including full attention
sits at ceiling is not an instrument, it is a saturated task. That is FB4, and you fire it or clear it.

## 1. Deliverables

### D0 — ⭐ FB4, and it is your FIRST commit's experiment (report to the Hub before you touch D1/D2)

Run the existing memory gym — **four families** `overload · aggregate · recency · manifold`, **3 seeds
{0,1,2}**, at the **shipped anchor** (`overload` at `load1x_shipped`, the 478× cell — reconciliation 6:
`overload` at the base atom budget is **unusable**, 0/18 admissible *including the Gaussian control*).

⛔ **The recency family runs with `restrict_index_to_pair=True`.** C2W2's D4 proved the shipped default
grades a **K-way** answer on a **2-way** curve (the CLU's answer falls outside its own pair **19.4 %** of
the time). The pre-fix number **`0.3019 ± 0.0679` is a scoring-domain defect and is never-quote as a
null**; post-fix the dividend is **−0.0028 ± 0.0619**. Emit both coverages so the switch is auditable.

Per family, emit **five arms on identical φ with φ-bytes ledgered**:
| arm | what it is | bytes |
|---|---|---|
| `full` | the shipped CLU read | full ledger |
| `launder` | settle-deleted / matched-bytes table launder (`chlu.eval.dividend.settle_deleted_launder`) | table |
| `substitute` | the family's strongest **+0 B** substitute (`knn_mean_launder` uniform **and** inverse-distance for `aggregate`; `order_aware_launder(k=2)` for `recency`; `echo_launder` for `manifold`; for `overload` the best of the frozen table readers at +0 B — declare which) | table **+0 B** |
| `blank` | `blank_store_control` — the family's floor | — |
| ⭐ `attention` | **NEW: a full-attention reader** over the launder's own (key, payload) table — softmax over `q·kᵀ/√d`, value-weighted read, no learned parameters beyond a scalar temperature fit on the family's own train split. It is a **reader of the same bytes**, so its ledger is the table's ledger **+ 4 B** (the temperature) — declare it | table + 4 B |

⭐ **The attention arm is the point of FB4.** It is the strongest metric-native reader anyone would
reach for. If it is *also* at ceiling everywhere, the families are not discriminating readers at all.

### D0.1 — ⭐ THE PRE-REGISTERED "AT CEILING" RULE (Hub decision, filed BEFORE you run; Head may overrule)

FB4's text — *"the +0 B substitute is at ceiling for every family including full attention"* — has no
number in it. It gets one **now**, in your `PREREG.md`, so the verdict is **computed, not argued**
(the C2W2 precedent: the gate's admissibility rulings did real work in both directions *because* they
were ratified before adjudication).

The gym's metrics have different scales (`overload:decode` ∈[0,1] · `aggregate:neg_mae` ≤0 ·
`recency:acc` ∈[0,1] · `manifold:r2` ≤1), so the rule is **normalised against the family's own floor**:

> Let `M(f)` = the metric's exact maximum (**1.0** for `decode`/`acc`/`r2`; **0.0** for `neg_mae`),
> `blank(f)` = the blank-store control, `sub(f)` = the best +0 B substitute, `attn(f)` = the attention arm.
> **Saturation** `S(f) = (sub(f) − blank(f)) / (M(f) − blank(f))`.
>
> **Family `f` is SUBSTITUTE-SATURATED iff `S(f) ≥ 0.95` AND `sub(f) ≥ attn(f) − 2 SE`** (3 seeds,
> `SE = sd/√3`, sample sd `ddof=1`).
>
> - ⛔ **FB4 FIRES iff ALL FOUR families are substitute-saturated.** ⇒ the protocol measures the task,
>   not the memory. **STOP. Report to the Hub the same hour. The wave pauses for a Head/Advisor
>   protocol ruling before `bprime-rivals` is built** (§A11 task 1, verbatim). Do not start D1/D2 until
>   the Hub answers — actually, **do** start them (they are orthogonal harness debt), but ship D0's
>   verdict first, in its own commit and its own interim report.
> - ◐ **FB4 PARTIAL (1–3 families saturated).** Report it as a first-class finding. Each saturated
>   family is **struck from B′'s cross-family audit as protocol-invalid** and named as such in your
>   report, with its arithmetic. The wave proceeds on the survivors. ⚠ **`manifold` is expected to
>   saturate** — `echo_launder` scores **1.0000** at +0 B by construction, which is intervention §8.3
>   in its purest form. A partial that is exactly `{manifold}` is the *predicted* outcome, not news.
> - ✅ **FB4 CLEARS (0 families, or only the expected `manifold`) ⇒ the protocol is validated** on the
>   surviving families and `bprime-rivals` is released against exactly those families.

⛔ **`S(f) ≥ 0.95` and the 2-SE attention leg are pre-registered constants. Do not tune them after
seeing the data.** If you believe the rule is wrong once you see the numbers, report *both* the rule's
verdict and your objection — the Hub takes the objection to the Head. It does not get edited in flight.

### D0.2 — the two-sided byte ledger on every arm
Every arm above emits `full_bytes / launder_bytes / ratio` and `phi_id / phi_bytes`. Reuse
`chlu.eval.dividend.byte_account` and the banked byte-floor law `ratio = 1.4·atoms_per_item + 0.8`
(verified to **1e-9** in all 28 C2W1 cells, floor **2.20×**). ⛔ **No C2W3 cell is a byte-matched
dividend** — the minimum ratio measured anywhere is **17.11×**; say so on every table.

---

### D1 — ⭐ HARNESS-DEBT RIDER A: `store_write_mask_factory` (§A11 rider · reconciliation 5)
`chlu/core/clu_system.py` already resolves `store_potential_factory` (an import path) so a new store
family can supply its own potential without editing a line it does not own — **but there is no matching
hook for the write mask.** A new store family therefore cannot supply its own update mask, an unmasked
leaf breaks C3 locality, and a test asserts it. **This one missing config field blocks any future store
family — including `route3-stage2`'s slotted store, which is the same wave.**

- Add `store_write_mask_factory: Optional[str] = None` + `store_write_mask_kwargs`, resolved by the
  same `pkg.module:attr` mechanism as `resolve_store_potential_factory` (mirror its error messages and
  its non-callable `TypeError`).
- **Default `None` ⇒ shipped behaviour bit-identical.** A blocking regression test proves it.
- A second test: a toy store family that supplies its own mask **preserves C3 locality** where the
  unmasked leaf breaks it — i.e. the test that currently asserts the failure now has a passing partner.

### D2 — ⭐ HARNESS-DEBT RIDER B: land SC-1…SC-7 as a MONITORED SOFT CONSTRAINT (§A9.8)
The soft certificate is **standing harness doctrine, orthogonal to paper shape** — Head ruling §A9.8:
*"it is what makes any non-separable measurement legal."* The spec is written and you implement it
verbatim from `doctrine-repairs.md` §4.4; you do **not** redesign it.

The load-bearing structural finding you are landing: **the shipped harness sets
`d_safe := 2s_max + κ′σ_q`, so the admission radius IS the certificate radius** — the two "mutually
exclusive" bands are **one object**, and that is why §A2.5 was confirmed 7/7 under the shipped rule and
**refuted 11/12** under the soft certificate.

- **SC-1 — break the identification.** `d_safe` becomes an **independent, declared** admission radius
  `d_safe = ζ·sep_expected` with `ζ = 0.6` (the harness's own S4 convention). `R_cert = 2s_max + κ′σ_q`
  is **still computed and still reported** — it is no longer the gate. This retires the
  `d_safe_override` hack (the gym needed it twice, the harness once, always "deliberately out of band"
  — **the override *was* the soft certificate, undeclared**).
- **SC-3 — the violation budget, declared per run.** `deficit_rel ≤ B` with **`B = 0.33`** (the measured
  edge) and `mean(deficit_rel over live items) ≤ B/2`. ⛔ **Exceeding the budget is a TRIP of monitor
  #3 — not a refusal.** A soft constraint that refuses is a hard constraint with extra steps.
- **SC-2 / SC-4…SC-6** land as specified in `doctrine-repairs.md` §4.4. **SC-7 is a falsifier, not
  code** — record it in your report: *if a shared/factored store's wells cannot hold `λ_min > λ_floor`
  at any admissible `B`, basin interaction and non-degeneracy are genuinely disjoint and that is a Head
  escalation.* On the theorist's grid they are **not** disjoint: at `B = 0.33`, `λ_min = +3.19` with
  `ρ_ex = 0.294`.
- ⛔ **DEFAULT-OFF, additive, with a bit-identical-shipped-behaviour regression test.** With the soft
  certificate disabled the harness must be bit-identical to `6ff4c1d` — the C1W27 `payload_gate`
  precedent, and it is blocking.
- ⚠ **Carry the price tag into your report, unsoftened:** decoupling buys `ρ_ex` up to **6.3×** at a
  `λ_min` cost of **2.2–6.0×**, and **the dividend in that region stays ≈0** (+0.0043 … −0.0067). The
  relaxation is a **precondition, not a result** — §A2.1 predicted exactly this and the theorist said so.
- ⚠ **The budget's outer edge was located by a broken proxy.** R1's corrected inradius is the
  prerequisite for setting `B`, not an independent cleanup; `sep/2` is **never-quote** as a certified
  inradius, and the corrected proxy is valid only inside `s/sep ∈ [0.15, 0.30]`. State the domain
  wherever `B = 0.33` appears.

### D3 — Monitor #3's replacement leg (already specified; land it with SC)
`doctrine-repairs.md` §1.3 retires monitor #3's **correlation leg** (`corr(gate_margin,
post_write_drift)`, sign-unstable for four named causes) and replaces it with a **C3 first-order
calibration test**: spearman **+0.914 vs +0.412**, **0/12 sign flips vs 1/12**, **at zero extra cost**
(`_c3_check` already computes it). This was queued to C2W3 by the C2W2 freeze rule (*"a rider that grows
while its host runs is how a 'small' task eats a worktree slot"*) — it is now in scope, and it belongs
with SC-1 because SC-1 is what stops the fire-rate band and the certificate being one object.

⭐ **ACCEPTANCE for D1–D3, the C2W2 pattern that converted "repaired" into an artifact:** re-run the
**C2W1 shipped anchor** and **DIFF the monitor trip-states** against the on-disk C2W1 artefacts
(`.claude/outputs/{memory-gym-v0,full-clu-harness}/*.json`) — **never** against a freshly generated
baseline. **Every changed trip must map one-to-one to a named repair; everything else bit-identical.**
An unexplained trip-state change is a **regression, not a repair.** ⚠ Diff against the **post-C2W2**
state where `phi-particle-head` already moved monitor #6 (**58 pre-repair → 27 post-repair, 0 new
trips**); "58 trips" without *"pre-repair"* is never-quote, and the #6 artefact count is **31 of 58**,
not 29.

## 2. Compute
D0 is hours on the existing gym (28-cell scale, 3 seeds). D1–D3 are code + regression runs. **JAX
cold-start on this machine is ~20 min** — keep the session warm, use `--quick` for smoke runs, and do
not mistake a slow import for a hang. Reuse the main venv:
`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …` with cwd in the worktree. If you
must `uv sync`, use `--frozen` and report the resolved JAX version in your flag-provenance table.

## 3. PREREG (`.claude/outputs/bprime-fb4-gate/PREREG.md`, before ANY measured run)
Mandatory. It must contain, at minimum:
1. **The D0.1 saturation rule, verbatim**, with `0.95` and the 2-SE attention leg fixed.
2. Your predicted `S(f)` per family and the reasoning. (The Hub's own prior, for the record: `manifold`
   saturates — `echo` = 1.0000 by construction; `recency` is close after the defect fix — insertion
   order at +0 B answers the question *exactly*; `aggregate` and `overload` do not. Predicted verdict:
   **PARTIAL = {manifold}**, possibly `{manifold, recency}`. Beat that prior or confirm it.)
3. Predicted attention-arm ordering vs the +0 B substitute per family.
4. D1/D2's bit-identity predictions (both are **exact**, not statistical — a partial pass is a fail).

## 4. Falsifiers
- ⛔ **FB4 fires on all four families** ⇒ the B′ protocol measures the task. **Wave pauses; Head/Advisor
  ruling before rivals.** This is decision-grade and you report it the same hour you have it.
- ⛔ **The attention arm cannot be given a byte ledger commensurate with the table** (i.e. it needs
  parameters the table does not have) ⇒ FB4 is **undecidable as specified**; report that, do not
  substitute a weaker reader and call it attention.
- ⛔ **D1 or D2's flag-off regression is not bit-identical** ⇒ blocking; nothing built on either hook is
  interpretable, and `route3-stage2` inherits a corrupted foundation.
- ⛔ **A monitor trip-state changes with no named repair behind it** ⇒ regression, not repair.
- **Does NOT falsify:** a family saturating (that is a finding with a defined consequence) · the
  attention arm winning on a metric-native family (that is the **metric-native-ceiling theorem**,
  confirmed four times, and it is not news — intervention §6 criterion 4) · the soft certificate
  showing **no dividend** in the non-separable region (§A2.1 predicts it; it is a precondition).

## 5. File ownership (the zero-conflict standing practice — 9 branches / 3 waves / 0 conflicts)
**Yours, exclusively this wave:**
`chlu/core/clu_system.py` · `chlu/core/monitors.py` · `chlu/core/admission.py` ·
`chlu/core/soft_certificate.py` (**new**) · `chlu/eval/fb4_gate.py` (**new**) ·
`chlu/experiments/exp_fb4_gate.py` (**new**) · `tests/test_{clu_system,monitors,soft_certificate,fb4_gate}.py`

**Read-only to you** (owned by others this wave, or standing):
`chlu/config.py` (**standing read-only to all C2 engineers** — C2 config objects live in the C2-owned
module that uses them) · `chlu/core/psi_readout.py`, `chlu/experiments/exp_{traj_write,ssb_shell}.py`,
`chlu/eval/attribution.py` (`route3-stage1-plus-2x2`) · `chlu/eval/rivals/**`,
`chlu/experiments/exp_bprime_rivals.py`, `chlu/eval/dividend.py` (`bprime-rivals`; you **import** from
`dividend.py` freely, you do not edit it) · `chlu/core/{memory_potentials,placement,integrators,
controller,shell_atoms}.py`.

⛔ **If your work requires editing a file you do not own, STOP and report to the Hub.** Do not edit it
and mention it afterwards. Nine branches over three waves have produced zero conflicts by this rule.

## 6. ⛔ Never-quote (inherited — violating these sends the report back)
Monitor #6's **"58 trips"** without *"pre-repair"* (post-repair **27**) · **"29 of 58"** as the #6
artefact count (it is **31**) · the ridge saddle **`λ_min = −0.5946`** as a multi-seed result (seed 0
only; 3-seed mean **+0.177 ± 0.469**) · the recency family's **`0.3019 ± 0.0679`** as a null (it is a
scoring-domain **defect**) · **"Prop D1 is violated (1.5–7.44×)"** (retired — `U` was computed against a
certificate that did not exist) · **`sep/2`** as a certified inradius, and the corrected proxy outside
`s/sep ∈ [0.15, 0.30]` · **`λ_min > 0`** as certifying a nonempty basin (measured basin **0.000** at
`λ_min = +0.910`) · **`k*`** without *"of `∂q_N/∂θ`, and only where the fixed-point sensitivity
dominates the transient"* · any **`AttentionPsi`** trajectory number (**it leaks** — `q0_only`
0.35–0.45 vs a bar of 0.19, at every stride; `route3-stage1-plus-2x2` is quarantining it this wave.
⚠ **This does not bar your D0 `attention` arm** — that is a *table reader*, not `AttentionPsi`, and it
never sees a trajectory. Say so explicitly in your report so no one conflates them) · **`ε` as "the
manifold-payload lifetime dial ∝ 1/ε"** without the `2α` coercivity ceiling (`τ_max = Γ/2α`) · any
**SDM Table 1 state/param ratio** (two extractions conflict) · **Titans as "a preprint"** (it is
**NeurIPS 2025, peer-reviewed**) · **"MAD `compression` is the admissible synthetic"** (dead by
arithmetic: iso-state 4096 dims vs a 224 B max payload — **73× at fp32**) · **"principled forgetting"**
as a novelty phrase · any C2W3 cell as a **byte-matched** dividend (min ratio anywhere **17.11×**).

## 7. Output
`.claude/outputs/bprime-fb4-gate.md`, protocol §5 format, with:
- ⭐ **the FB4 verdict in the first 10 lines** — `FIRES` / `PARTIAL = {…}` / `CLEARS`, with `S(f)`,
  `sub(f)`, `attn(f)`, `blank(f)` and the SE per family, and the surviving-family list `bprime-rivals`
  will be released against;
- the flag-provenance table (commit, seeds, every non-default flag — protocol §5, mandatory);
- the trip-state diff table for D1–D3, one row per changed trip mapped to its named repair;
- your reconciliation list in the **first 10 lines** if you produce one (protocol §5 corollary — it gets
  an owner at review, or it rots for two waves like the "2.6" retraction did);
- ⛔ **declared NOT-RUNs, never reported as nulls.**

⚠ **Report D0 to the Hub as an INTERIM the moment you have it** — do not wait for D1–D3. A whole
engineer task is gated on it.
