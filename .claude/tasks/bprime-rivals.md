# bprime-rivals — the audit paper's spine: does test-time dynamics buy anything over a table at matched bytes?

**Campaign 2, wave C2W4. Agent:** experiment-engineer (**the wave's first act and its heavy task**).
**Worktree MANDATORY** — you hold **worktree slot 1 of 3**, and you are the wave's spine (charter §A15.1).
Base local `main` @ **`d4f56c8`**. Branch `agent/experiment-engineer/bprime-rivals`.
Worktree: `git worktree add ../CHLU-rivals -b agent/experiment-engineer/bprime-rivals`.
Charter **ADDENDUM 3 §A15 task 1**, implementing **§A14.2** (the family set) and **§A9.1** (B′ is the spine).

> ## ⭐ YOU ARE NOT GATED. SPAWN IMMEDIATELY.
> This file was written in C2W3 and **released by both its gates** — `bprime-fb4-gate` (FB4 does **not**
> fire) and `bprime-fb1-recon` (FB1 does **not** fire) — and then **the wave ended before you were
> handed over.** That is the only reason B′ has no rival rows. **C2W3 closed INCOMPLETE because of
> this task, and C2W4 opens with it.** Nothing is gating you now; the family set below is already ruled
> (§A14.2) and is not yours to re-litigate.
>
> ⚠ **What CAN still stop you, mid-flight:** **FB2** and **FB3** (§5). Both re-shape the paper rather
> than refine it. Report either the same day, to the Hub, with evidence.

**Read first:** `.claude/AGENT_PROTOCOL.md`; `.claude/advisor-head-c2-charter.md` **in full, especially
ADDENDUM 3 (§A12 the C2W3 adjudication · §A13 the claim architecture v3 — you are TIER i · §A14.2 your
family set · §A15)** and **ADDENDUM 2 §A9.1 / §A9.9 / §A9.11**;
`.claude/advisor-head-intervention.md` **§6 (five criteria) and §8 (prohibitions)**;
`.claude/outputs/track2-admissibility/PREREG-Bprime.md` **in full — §2 is the table this paper IS, §4 is
your byte conventions, §5 is P1–P5, §6 is FB1–FB5, §7 is what you must NOT re-measure**;
`.claude/outputs/bprime-fb4-gate.md` **§A3 in full (your family set's evidence, incl. §A3.7's four
construction rules)**; `.claude/outputs/bprime-fb1-recon.md` **(FB1's verdict + the P4 narrowing you
must cite, not suppress)**; `.claude/outputs/bprime-theory.md` **T1 (the corrected byte law) and §6 C3**;
`.claude/outputs/track2-admissibility.md` §2.3 + §3; `.claude/outputs/rival-recon.md` §F2;
the **`2026-07-31 (later still)` `[C2W3]` §10 entry** in `.claude/handover_context.md`.

⭐ **REGISTRY STATUS — CURRENT (the C2W3 curator pass landed).** The wave-wide *"quote only
`.claude/outputs/*` and §10"* restriction is **LIFTED**. `claims_matrix.md` §0 carries the consolidated,
dated never-quote list — **read it**. ⚠ **Three live errata override the registries wherever they
disagree**, and all three touch you:
1. ⛔ **The byte law's published verification is wrong.** *"Verified to 1e-9 in all 28 cells"* is in fact
   **24 of 28**. The corrected law **`ratio = [A(D+2) + d]/(d+m)`** is exact in all 28 in rational
   arithmetic; the floor **RISES** from 2.20× to **2.40×** at `n_spectator = 1`. **The error is
   conservative — no claim was inflated — and `PREREG-Bprime.md` §7's reuse licence STANDS: you do NOT
   re-measure the byte-floor theorem.** You quote the **corrected** law and the corrected floor.
   `PREREG-Bprime.md` is deliberately **NOT** edited (a revised pre-registration stops being one).
2. ⛔ **MUNKEY (arXiv:2603.15033) is an ICLR-2026 *workshop* paper (oral), NOT ICML 2026.** The
   **workshop's identity is QUARANTINED** (two sources disagree — cite it as "an ICLR-2026 workshop
   paper" and name no workshop). Its v3 self-describes as **"a memory-augmented transformer"**, not a
   ViT classifier. The **narrowing itself stands** (MIA-AUROC→0.5 by design, not exact, gap to
   retraining 0.56 ± 0.21).
3. ⚠ **Monitor #6's "27 post-repair" is PROVISIONAL** until `harness-debt` lands the `+eps_acq` half
   this wave. Do not quote a monitor-#6 count.

---

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial / pillar:** **none in the CHLU sense — this is a cross-family AUDIT, and it is TIER i** of the
  claim architecture v3 (§A13). You are not claiming a CLU win. You are pricing **test-time dynamics
  against its own byte-matched table, uniformly, for the CLU and for the TTT/Delta family.** B′ is *"a
  contribution, not a retreat"* (§A3) **exactly because the protocol is uniform** — which is why this
  task is one engineer and not two (Hub ruling §6.1).
- ⛔ **You do NOT claim, mention, or imply the §A13 reframe** (the organization dividend, the cat test,
  tier ii, tier iii, "CLU-former"). Tier i's *conclusion* is what later justifies the reframe; the
  reframe is claimed **nowhere** in tier i. If your results suggest it, put that in *Open questions*,
  not in a finding.
- **Laundering control:** the launder **is** the deliverable. Every family, every arm: matched-byte
  table launder · **+0 B** substitute · two-sided byte ledger · same-keys null · blank-store control ·
  **identical φ and φ-bytes on every arm** (enforce in code — raise, do not warn).
- **Falsifies:** §5. **Does NOT falsify:** a rival beating CLU (that is a *result of the audit*; FB3
  pre-commits us to saying so rather than re-framing) · a rival losing to its own table on a
  metric-native probe (that is the metric-native-ceiling theorem, predicted by P2, not news).

---

## 0. Why this task exists, in four sentences
Every rival family surveyed by `rival-recon` is **metric-native or weakly so** — *"the launder ceiling
is the field's problem, not ours"* — and **P4 (narrowed, see §0.1) says no published paper in this
family runs a non-parametric matched-byte control.** The audit paper's claim is therefore not "CLU
wins"; it is *"here is the price of test-time dynamics, measured the same way for everyone, and the
field has never measured it."* C2W1 banked the CLU column (28 cells, the byte-floor law, substitute
audit 0-for-4, exact deletion AUC 0.5000 ± 0.0000); **you build the other rows.** The paper exists when
`PREREG-Bprime.md` §2 has no `need` cells left in the rows you own.

### 0.1 ⭐ P4 as `bprime-fb1-recon` narrowed it — you inherit the narrowed sentence, not the original
FB1 does **not** fire (14 candidates: 0 HIT · 2 PARTIAL, both out-of-family · 7 NEAR-MISS · 5 NO), and
FB5's route does not fire either (arXiv:2501.12352 is purely theoretical — softmax attention appears as
the nonparametric special case **analytically**, with no experiments and no baselines). But P4 survives
**materially narrowed on two axes**, and the narrowed sentence is what you write:
- ⛔ **The audit-at-equal-bits discipline IS standard *outside* the family** — learned Bloom filters,
  learned indexes, SOSD. **That is B′'s methodological ancestry and it is CITED, never suppressed.**
- ⛔ **A token-matched trivial control was published 7 days before filing** in LLM-agent memory
  evaluation (arXiv:2607.21962).
- ⭐ **The surviving claim, and it is stronger than silence:** *seven independent groups built the
  adjacent instrument and none closed the loop* — a byte axis (Based), an iso-state normaliser (MAD), a
  state/param ratio (SDM), a matched trivial-policy control (HOLA), a compute-priced datastore
  (MassiveDS). **"A conceded ancestor is worth more than a contested monopoly."**
- ⛔ **The substitute-audit *idea* is not ours in general form** — it is the partial-input /
  trivial-baseline audit tradition (Poliak et al. 2018; Feng, Wallace & Boyd-Graber, ACL 2019).
  Concede it. (`bprime-draft` owns the prose; you own not contradicting it.)

---

## 1. ⛔ THE FAMILY SET — RULED (§A14.2). Not yours to re-open.

FB4 returned **◐ PARTIAL** and **does not fire**: three of four designed gym families are **saturated**
(something costing ≤ 4 B sits at the metric's exact maximum, and it is never the CLU) and are **struck
from B′'s cross-family audit as protocol-invalid.** Verbatim from the adjudication:

| family | `S(f)` | verdict | your instruction |
|---|---|---|---|
| **`aggregate`**@54.56× | **0.5068** | ✅ **SURVIVES** | ⭐ **the sole reader-discrimination family. Every dividend number you report is on this family.** |
| `overload`@478.2× | 1.0000 | ⛔ SATURATED | ⭐ **runs, but ONLY as an explicitly labelled BYTE-FRONTIER column — never as a dividend family, never in a dividend table, never as a headline.** Its defensibility is the declared secondary reading `S_excl = 0.6500` (arg-min launder excluded from the +0 B reader set); state that whenever you print it. |
| `recency`@54.56× | 1.0000 | ⛔ SATURATED | ⛔ **DO NOT RUN.** A reader of the table's **own row order** answers it at 1.0000, 3/3 seeds, at **+0 B**. The fix that made it scorable is the fix that makes it protocol-invalid. |
| `manifold`@52.0× | 1.0000 | ⛔ SATURATED (predicted) | ⛔ **DO NOT RUN.** The ≤4 B `echo_+0B` substitute is at 1.0000 while full attention reads 0.00. |

⚠ **The thinness is owned, in writing, in the paper's own limitations — verbatim, not softened:** *two*
rival families audited against **one** surviving synthetic family is a thin cross-family audit. Your
report says so in the first screen; `bprime-draft` puts it in Limitations. **Building a replacement
family is a wave's work, not a rider — §A14.2 defers that decision to the C2W4 review with your rival
rows in hand.** If you find yourself building a new family, STOP and report.

⚠ **The engineer's filed objection, carried and still unresolved (do not resolve it, just don't lean on
the rule beyond its reach):** the FB4 rule cannot distinguish *"the family is substitutable"* from
*"the family's **anchor** is substitutable"*. `overload` saturates at `load1x_shipped`, the cell chosen
because the base budget is unusable — which is exactly why it is a frontier column, not a claim.

⭐ **And the one design rule that generalises, for everything downstream:** `aggregate` survives for one
reason — **its target is constructed to be absent from the table** (a convex combination, dropped at
construction if it lands within `payload_tol` of a stored payload). *"The answer is provably not in the
table"* is **the only property that has survived a +0 B audit in two waves** (C2W1 0-for-4; C2W3
1-of-4).

## 2. Deliverables

### D1 — ⭐ A minimal faithful **TTT-class** memory on the gym harness
TTT-Linear (and TTT-MLP **only if** it is cheap at this weight class — **declare if you skip it; a skip
is a NOT-RUN, never a null, and it makes P3 NOT-RUN rather than refuted**). The inner-loop,
self-supervised, per-sequence gradient-descent memory.
- **Matched parameters AND matched state-bytes** to the CLU arm it is audited beside.
- ⭐ **The learned-initial-state rule (`PREREG-Bprime.md` §4.1) — the audit's sharpest edge:** **the
  initialisation `W₀` is PARAMETERS (F1); only the per-sequence deviation is STATE (F2). Both
  declared.** Counting the init as state **inflates**; counting the deviation as parameters
  **launders**. This rule is *ours* and is one of the paper's contributions — **apply it to CLU's `V_θ`
  init too, in the same table**, so no referee can say we scored ourselves generously.
- State ledger per `PREREG-Bprime.md` §2: `d_head²` (Linear) / `8·d_head²` (MLP) **+ the `b = 16`
  buffer**.

### D2 — ⭐ A minimal faithful **delta-rule** memory: DeltaNet + **Gated DeltaNet-2**
⛔ **§A14.2 ruling: Gated DeltaNet-2 (arXiv:2605.22791) REPLACES GDN as the delta-rule reference arm.**
GDN(-1) may still be run as the ablation that isolates what the -2 revision changed, but **the reference
arm — the one that appears in the audit table's delta-rule row — is Gated DeltaNet-2.** ⚠ It was
**absent from our registry** until C2W3 (reconciliation 7); `doc-curator-c2w3-sync` is filing it this
wave. **Verify its state-size convention and its update equation from the paper yourself** — do not
inherit GDN(-1)'s Eq. 8 accounting without checking that the -2 revision preserves it, and **say in your
report which equation numbers you implemented.**
- State ledger `n_head·d_k·d_v` (§2), adjusted to whatever -2 actually specifies.
- Where the delta rule's metric-nativeness lives is an **equation-level argument that you then
  MEASURE** against its own table — assert nothing.
- ⚠ Reference implementations exist and are portable (FLA — the 41-model table). **You are writing a
  minimal faithful reimplementation on the gym harness, not vendoring a training stack.** Faithful to
  the update equation and the state size; minimal in everything else. **Caption it as such in every
  table.**

### D3 — the audit columns, on every (family × rival) cell — the actual deliverable
At **3 seeds minimum** (⛔ **multi-seed before any paper number** — standing rule, no exceptions; a
single-seed cell is reported as `n=1` and is never a headline):
1. **dividend** = (the rival's learned dynamics) − (its own byte-matched table launder), same harness,
   same bytes, same φ — **on `aggregate` only**;
2. **matched-byte table launder** — the family's own; for a weight-valued memory this is the byte-equal
   table of the `(θ_K x, θ_V x)` pairs (**P5** predicts **0 of 5** failures — *test it, do not assume*);
3. **two-sided byte ledger** — parameters *and* state, both sides, with the §4 conventions and the
   **corrected** byte law;
4. **+0 B substitute** — the family's strongest zero-byte reader, **signed margin reported** (the C2W1
   audit went 0-for-4 and the margin is what makes that computable rather than arguable);
5. **same-keys null** and **blank-store control**;
6. **metric-native verdict** — is this rival's read a metric operation on its own state? Argued at
   equation level, then **measured** against its own table.
7. ⭐ **admissible-cell coverage per family, first-class** — the C2W2 standing rule: *what must not
   happen is admissibility filtering quietly gutting coverage.* Report `admissible/total` per cell.

### D4 — the byte-frontier column (`overload@load1x_shipped`), labelled at every appearance
Not a dividend family. Its job is one curve: **accuracy vs byte ratio**, with the CLU's banked
`decode 0.972 → 0.097` as the ratio falls `478× → 2.28×` reused (§7 banked — do **not** re-measure it),
and the rivals' equivalents measured beside it. ⛔ **Every appearance carries the label and the
`S_excl = 0.6500` sentence.** ⛔ **No C2W3-or-later cell is a *byte-matched* dividend** — the minimum
ratio anywhere is **17.11×**.

### D5 — SDM and Titans: positioning only. The caveats are BINDING.
- ⛔ **Sparse Delta Memory is NOT built and NOT run.** Official code exists
  (`facebookresearch/sparse-delta-memory`, CC-BY-NC 4.0) but needs **Torch ≥2.8, Triton ≥3.4, SM 80+
  (Ampere/Hopper)** ⇒ **it cannot run on this machine.** It enters as **positioning** only.
  ⛔ **Quote NONE of its Table 1 state/param ratios** — two independent extractions disagree (156 % vs
  168 %; 111 % vs 98 %). Safe to state: `M_size = (d_qk^tot)²·d_v^tot/(4H²)` (Eq. 6); *"`α_t` is
  per-head, not per-slot"*; *"top-W/top-R index sets are per-read transients (F4, not F2)"*.
  ⚠ SDM is a **near-collision** (arXiv:2607.07386) — write the positioning **as if a referee has read
  it**, because one will have.
- ⛔ **Titans is NeurIPS 2025, peer-reviewed. Every "preprint" citation is wrong.** ⭐ **Hub ruling
  (C2W4): NO Titans arm is built this wave.** §A14.2 names TTT-class and Gated-DeltaNet-2-class and
  nothing else; there is **no official code**, it is **not in FLA**, its **chunk size `b` is never given
  a numeric value**, and **no seeds are reported** — an arm on those terms would be *our reconstruction
  audited against our reconstruction's table*, which is not evidence. **Titans is positioning, and the
  Titans row in `PREREG-Bprime.md` §2 is declared NOT-RUN with this reason.** ⚠ Its `2·|M_θ|` momentum
  accounting **remains our reconstruction and is captioned every time — the paper states no
  convention** (§2 already marks that cell ⚠ UNPINNED; leave it unpinned).
  ⚠ **Consequence, stated by you, not left for a referee: with no Titans arm and no TTT-MLP, P3 is
  NOT-RUN.** Say so in the prereg scorecard.

### D6 — the deletion column, in the "and also" position, with its narrowing
Charter §A9.9 + intervention §8.3: byte-exact deletion stays in the **"and also"** position, and the
**MUNKEY narrowing is recorded with the corrected venue** (erratum 2 above — ICLR-2026 workshop, oral;
workshop name quarantined; "a memory-augmented transformer"). Our claim **survives, materially
narrowed**, phrased on **verified byte-exactness**, never on "we alone delete."
⚠ **Say it before a referee does: a table deletes exactly by construction.** Exact deletion is a result
only for a *learned/superposed* store.
⚠ §A9.9 standing: any future **shared-substrate** work measures deletion as a **curve** (exactness on
the private-atom fraction; measured degradation on the shared fraction). ⭐ The theorist has now
**priced** that frontier: **compression and byte-exact deletion are the same trade**, matched bytes
needs `S* = (D+2)A_tot/m` items per atom (**7** at `A_tot = 1`, **2387** at the shipped anchor), and the
byte-exactly-deletable fraction is capped at `p ≤ [(d+m)r − d]/[(D+2)A_tot]` (**0.042 %** at `r = 1`,
`A_tot = 341`). You are not building a shared substrate; you are recording the rule and its exchange
rate so the deletion column cannot be quoted out of that frame.

### D7 — rider: the structural ledger identity (theorist code request C3)
In `chlu/eval/dividend.py::byte_account` (**your file, append-only surface**): assert the ledger
identity **structurally as integers** rather than by comparing a float ratio —
`full == 4·[N_at(D+2) + K·d]` and `launder == 4·K·(d+m)`. It is a blocking test, it is the guard that
makes your two-sided ledger auditable, and it costs minutes.
⛔ **You do NOT touch `chlu/experiments/memory_gym.py`** — its live `byte_ratio_law` bug is
`harness-debt`'s, this wave, by Hub assignment. If your ledger and the gym's printed law disagree,
**that is expected until `harness-debt` merges**; use the corrected law, note the disagreement, and do
not fix it in your branch.

## 3. What you must NOT re-measure (`PREREG-Bprime.md` §7 — banked; re-deriving burns the wave's spine)
matched-bytes launder record (28 cells; `matched=False` is **architectural**) · the byte-floor theorem
(**corrected form `[A(D+2)+d]/(d+m)`**, exact 28/28 in rational arithmetic; floor **2.20×** gauss,
**2.40×** at `n_spec=1`, **2.40/2.60×** shell; measured min 2.28×) · substitute audit **0-for-4**
(insertion order **0.776** vs 0.302; echo **1.0000** vs −0.180) · **Prop D2a**, three independent
confirmations · byte-exact deletion **AUC 0.5000 ± 0.0000**, byte-equal **3072/3072** · **`D` is the
dividend's variance, not its magnitude** (the `D = 0.931` cell has dividend **−0.875**; monitor #2
bounds where a dividend *could* live and is **never** a progress signal) · the accuracy-vs-bytes curve
(`decode 0.972 → 0.097`, ratio `478× → 2.28×`) · the FB4 `S(f)` values.

## 4. PREREG (`.claude/outputs/bprime-rivals/PREREG.md`) — **before ANY measured run**
Mandatory (protocol §5). Restate and commit to, **per family/arm you actually run**:
- **P2.** Of the four families with an explicit (k,v)-shaped state (Mamba-2, DeltaNet, Gated DeltaNet-2,
  SDM): **≥3 lose to their own byte-matched table on a metric-native probe**, and **0 of 4 lose to it on
  real-data LM bpc.** ⚠ **You test the first half only** — the second half belongs to the real-data leg
  (`cluformer-pilot`, conditional) — **and you say so explicitly.**
- **P3.** The two **function-valued** memories (TTT-MLP, Titans `L_M ≥ 2`) show the **largest positive
  dividend** over their matched-byte tables. ⚠ **With no Titans arm (D5) and TTT-MLP optional, expect to
  declare P3 NOT-RUN. NOT-RUN is not refuted.**
- **P5.** The launder **transfers to all five** rival state types. **Predicted failures: 0 of 5.**
- **Your own per-cell numeric predictions, derived (not guessed), with the derivation written down.**
  A pre-registered prediction that survives is evidence; one that fails is a *finding*; an
  un-pre-registered agreement is neither.

## 5. Falsifiers (of B′ ITSELF — this is why the prereg exists)
- ⛔ **FB2 — "not apples-to-apples."** For **≥2 of 5** rival families no byte-matched table is definable
  without an arbitrary modelling choice ⇒ **the cross-family comparison is invalid and B′ collapses to a
  CLU-only negative.** Report the same day; it re-shapes the paper. ⚠ **You are running two families,
  so a "≥2 of 5" verdict is partly extrapolated — state which of the 5 you actually adjudicated and
  which you are reasoning about from their equations, and never blur the two.**
- ⛔ **FB3 — "the finding inverts."** Every rival shows a large positive dividend and **only CLU does
  not** ⇒ B′ is a *different paper* (*"test-time dynamics pays, except for ours"*). ⭐ **We pre-commit,
  in writing, to SAYING SO rather than re-framing. This is the single most important sentence in your
  task file.** If it happens, write it plainly and let the Hub take it to the Head.
- ⛔ **FB1 — "not news."** The literature sweep says it does not fire, **but you can see from inside an
  implementation what a sweep misses.** If, while implementing, you find that ≥1 established paper in
  the family already runs a matched-byte non-parametric control and reports the same verdict, **report
  it the same hour** — it kills P4 and re-scopes this task.
- **Does NOT falsify:** a rival winning · a family being hard · a single arm failing to train (report it
  with its evidence and its budget — **a truthful partial beats a confident wrong one**) · the CLU
  reading below a table (that is the whole point of the instrument).

## 6. Compute, scope and the two hard stops
### 6.1 One engineer, not two — Hub ruling (Head may overrule)
The charter gives B′ *"first claim on 2 worktrees"*. **I am spending that claim on `bprime-rivals` +
`bprime-c6`, not on splitting you.** Reason: **the deliverable is protocol uniformity across families.**
Two engineers applying the §4 byte conventions independently manufactures exactly the inconsistency the
paper is about, and a cross-family audit whose two rows were scored by two agents is worth less than one
row scored by one. **You get the wave's longest run and its first spawn instead.**
### 6.2 ⛔ No real-data leg in this task
Stay on the gym harness at the C2W1/C2W2 weight class. The real-data leg is a **separate, conditional
task** (`cluformer-pilot`, §A14.3, **gated on your audit columns landing**) and it is **tier iii**, not
tier i. ⛔ **If you find yourself sizing a language-model run, STOP and report.** Your audit columns
landing on `aggregate` with the protocol holding **is** that task's gate — which makes finishing yours
the wave's critical path.
### 6.3 Environment
⚠ **JAX cold-start ~20 min** — keep the session warm, `--quick` for smoke runs, reuse the main venv
(`PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …`, cwd in the worktree). If you must
`uv sync`, use `--frozen` and report the resolved JAX version in your flag-provenance table.
⚠ **N94 epoch discipline.** ⚠ Use `git -C <worktree>` always.

## 7. File ownership (the zero-conflict standing practice — **11 branches / 4 waves / 0 conflicts**)
**Yours, exclusively this wave:**
`chlu/eval/rivals/` (**new package** — `__init__.py`, `ttt.py`, `deltanet.py`, `ledger.py`) ·
`chlu/eval/dividend.py` (**append-only** — the C2W1 signatures are frozen; the gym-side callers were
appended for exactly this reason) · `chlu/experiments/exp_bprime_rivals.py` (**new**) ·
`tests/test_bprime_rivals.py`, `tests/test_rivals_ledger.py` (**new**).

**Read-only to you (owned by a concurrent C2W4 spoke or frozen):**
`chlu/experiments/memory_gym.py`, `chlu/core/monitors.py`, `tests/test_memory_gym.py`,
`tests/test_monitors.py` (**`harness-debt`**) · `chlu/eval/attribution.py`,
`chlu/experiments/exp_route3_attribution.py`, `chlu/core/soft_certificate.py`,
`chlu/cli/experiment_cmd.py`, `tests/test_attribution.py` (**`bprime-c6`**) ·
`chlu/core/blocks.py`, `chlu/data/` (**`cluformer-pilot`, if it spawns**) ·
`chlu/config.py` (**standing read-only to all C2 engineers**) ·
`chlu/eval/race.py` (**frozen schema** — the C2W2 RACE+SEAM FREEZE; you may import and emit `RaceCell`s,
you may **not** change the schema) · `chlu/eval/fb4_gate.py` · `chlu/core/psi_readout.py` ·
`chlu/core/{clu_system,admission,memory_potentials,placement,integrators,controller}.py`.

⛔ **If your work requires editing a file you do not own, STOP and report to the Hub.** Do not
opportunistically reformat or refactor shared files.

## 8. ⛔ Never-quote (inherited; violating these sends the report back)
Read `claims_matrix.md` §0 for the full dated list. The ones that will bite *you*:
**Titans as "a preprint"** (NeurIPS 2025) · any **SDM Table 1 state/param ratio** · **MUNKEY as "ICML
2026"** or with a named workshop (erratum 2) · **"verified to 1e-9 in all 28 cells"** (it is 24/28 —
use the corrected law) · **"MAD `compression` is the admissible synthetic"** (dead by arithmetic: its
iso-state normalisation at 4096 dims exceeds the task's 224 B max payload by **73× at fp32 / 36× at
bf16**) · **"principled forgetting"** as a novelty phrase · **"we alone delete"** · any C2W3-or-later
cell as a **byte-matched** dividend (min ratio anywhere **17.11×**) · **monitor #6's counts** (both
"58 trips" without *"pre-repair"* and "27 post-repair" — the latter is **PROVISIONAL** until
`harness-debt` lands) · the recency family's **`0.3019 ± 0.0679`** as a null (scoring-domain **defect**;
post-fix **−0.0028 ± 0.0619**) · the ridge saddle **`λ_min = −0.5946`** as multi-seed (seed 0; 3-seed
mean **+0.177 ± 0.469**) · **"Prop D1 is violated"** (retired) · **`sep/2`** as a certified inradius ·
**`λ_min > 0`** as certifying a nonempty basin (measured **0.000** at `λ_min = +0.910`) · any
**`AttentionPsi`** trajectory number (it leaks and now **raises** — `q0_only` 0.35–0.45 vs a bar of
0.19) · **`ε` as "the manifold-payload lifetime dial ∝ 1/ε"** without the `2α` coercivity ceiling
(`τ_max = Γ/2α`) · **`k*`** without *"of `∂q_N/∂θ`, and only where the fixed-point sensitivity dominates
the transient"* · ⭐ **any tier-ii/tier-iii claim whatsoever** (§A13 — not yours, and "CLU-former" is a
placeholder name that must never appear in a draft).

## 9. Output — `.claude/outputs/bprime-rivals.md`, protocol §5 format
- ⭐ **The audit table in the FIRST SCREEN** — one row per (family, rival), columns exactly as
  `PREREG-Bprime.md` §2, **every `need` you closed marked `have`, every one you did not marked NOT-RUN
  with its reason.** This table is the paper.
- **admissible-cell coverage per family, first-class**;
- the **signed +0 B substitute margin** per cell;
- the **two-sided byte ledger** per arm with the learned-initial-state rule applied to CLU as well;
- the **flag-provenance table** (commit, seeds, every non-default flag — mandatory, protocol §5);
- the **PREREG scorecard** (registered vs measured vs verdict, incl. every NOT-RUN);
- the **FB1 / FB2 / FB3 / FB5 adjudication**, explicitly, each with its evidence;
- ⭐ **the one-family thinness stated in your own words, in the first screen** (§1);
- **your reconciliation list in the FIRST 10 LINES** if you produce one (protocol §5 corollary — the
  "2.6" retraction sat live for two waves because a reconciliation had no owner);
- ⛔ **declared NOT-RUNs, never reported as nulls.**
- **Report your git footprint**: branch, commit hashes, files touched, tests run. Before removing your
  worktree, **verify from the MAIN repo that your branch ref shows your commits** (`git -C
  /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/bprime-rivals`) — the w4
  lesson cost 8 commits.
- ⛔ **Never push `origin`.** Do not push at all; leave the branch for Hub review.
