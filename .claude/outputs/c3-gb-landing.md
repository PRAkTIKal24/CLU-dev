# c3-gb-landing — experiment-engineer report

**Task + acceptance criterion:** land the ratified **G-B** geometry (`store_layers`, 1,380,864 B / 0.658×
pinned by test, pilot flag table provably unchanged), **clear run 3's launch blocker B2**
(`PREREG_CONT` in `job_gpu_cluformer.sh`), amend `PREREG-C3-LADDER.md` with the three binding additions
(placement decision · phase-1 declaration · the pre-registered store-liveness diagnostic with falsifiers
and a kill condition), and fix `byte_ledger.py`'s wrong remedy text.

**Status: done.** ⛔ One thing I did **not** do, deliberately and per the task's own STOP rule: I did not
touch the exemption's verification logic — and it is exactly there that I found the one blocking defect
below.

> ## ⚠⚠ RECONCILIATION LIST — NEEDS A HUB/ADVISOR OWNER (protocol §5 corollary, in the first 10 lines)
> 1. ⛔⛔ **RUN 3 IS *STILL* NOT SUBMITTABLE AS WRITTEN — AND B2 WAS NOT THE ONLY BLOCKER.**
>    `RUN3-LAUNCH.md` §4's three `sbatch` lines carry `ttt_normalized_write=true` **inside `MEM=`**.
>    `ttt_normalized_write` is a **`PilotConfig`** field, not a `StreamMemoryConfig` one, so measured
>    against the **real** run-2 journal it (a) is **silently dropped** (the TTT fix is *not* on) and (b)
>    still lands in `pilot.memory`, making it a **SECOND differing key** ⇒ the exemption **REFUSES the
>    leg**. Moving it to `SET=` is refused too (`pilot.ttt_normalized_write`). ⛔ **There is no way to set
>    that flag on run 3 and keep the exemption** — this is a genuine conflict between PILOT-TTT-RULINGS
>    **ruling 1** and PREREG-LeakAblation **§4**, not a plumbing bug, and it is **not mine to rule**.
>    ⭐ Recommendation + the exact one-deletion fix + two alternatives, costed:
>    **`.claude/outputs/c3-gb-landing/RUN3-LAUNCH-DELTA.md`**. **Owner: the Hub (→ Advisor).**
> 2. ⛔ **B1 is still open and is not mine:** this branch (which *extends* the exemption branch) is
>    unmerged, so a CSF3 `git pull` still fetches code without `--prereg-continuation`. **One merge lands
>    both, as the launch package intends.**
> 3. ⚠ **Three store-side instruments hard-coded layer 0** and would have broken or gone *silently* dark
>    under G-B (found by running it, not by reading it): the gradient probe **raised**, the §7.27 store
>    watch and the monitor pass would have reported `applicable: false`, the selector readout **raised**.
>    Fixed here (`first_store_layer`). ⚠ **Related, NOT fixed, flagged:** `calibrate_atom_group_centers`
>    and `calibrate_phi_gain` still calibrate off **layer 0's** latents, which under G-B is a layer with no
>    store. That was already an approximation across 12 layers; under G-B it is arguably wrong. **Needs a
>    ruling, not a patch** — it moves an initialisation, i.e. a claims-relevant number.
> 4. ⚠ **`BUDGET_IS_INTERIM = True` is still live and still blocks every ladder arm.** ⛔ Untouched by me
>    (it is one edit that belongs with *accepting* `PREREG-C3-LADDER`, not with building its geometry).
> 5. ⚠⚠ **NOT DISJOINT FROM `c3-rival-mamba2` AFTER ALL** (the task predicted disjointness "by
>    construction"; ⛔ **verified, and it is false**). That branch also edits `chlu/training/train_cluformer.py`
>    (a `PilotConfig` field — *different* region from mine — and **`build_arm`, whose hunk is ~3 lines from
>    mine**) and `chlu/experiments/exp_cluformer_pilot.py` (`main()` only; mine is `_selectors` — clean).
>    ⇒ **likely a clean merge, but `build_arm` is the one place to check**, and after merging both, re-run
>    `tests/test_c3_gb_geometry.py` **and** theirs. **Owner: the Hub, at merge time.**
> 6. ⚠⚠ **`store_layers` MUST NOT be applied to a published rival arm** (phase 2). A Mamba-2/GDN-2 layer is
>    the block's **sequence mixer**, not a memory in a slot — running one in 3 of 12 blocks is not that
>    model, and byte-matching by deleting nine of a rival's layers is **hobbling** (the anti-hobbling rule
>    has already inverted one verdict). The rival's matched-bytes control is the one already built: keep 12
>    layers, shrink its own declared knob. ⚠ As shipped, `StreamModel` applies the selection **uniformly**,
>    so a rival arm built through `build_arm` *would* inherit it. Noted in `PREREG-C3-LADDER` §4.3 and
>    handed to the rival spokes' reviewer. **Owner: the Hub (→ the phase-2 rival tasks).**
> 7. ⚠ **A full-suite run died silently under contention**, exactly as handover 7.46 predicts (§Suite).

---

## ⭐ DIAL DECLARATION (protocol §7, echoed before the first result)

- **Dial:** **none — instrument/plumbing + prereg amendment.** ⛔ No claim, no ladder arm trained, no bpc
  that any table could quote. Every number below is config arithmetic, a property of the instrument, or a
  toy-scale execution trace.
- **Laundering control:** n/a for a geometry landing — but the **inverse obligation** applies and is
  discharged: the deliverable is what the ledger still **refuses** and what the flag table still **shows**
  (§2, §3). ⭐ And the amendment I filed *creates* a laundering control for someone else: §7.1's
  **random-site read**, without which an oracle-addressed bpc would launder an off-distribution
  perturbation as an addressing effect.
- **Falsifies the task:** a `store_layers` selection that moves the byte arithmetic off **1,380,864 B**;
  run 3 still refused after §2; the prereg amendments not surviving a re-read by someone who was not here.
  ⭐ **The second one FIRED — and not for the reason the task expected** (item 1 above): B2 is cleared and
  asserted, and run 3 is still refused, by a *different* second key that was already in the launch package.
- **Does NOT falsify:** G-B being architecturally unusual. **It is ratified**; I implemented it and did not
  re-litigate it.

**Pre-registration:** ⛔ none owed *by me* — my acceptance criterion is an instrument, not a measured
ratio/slope/law, and **zero arms were trained**. ⭐ I *filed* one **for others**: `PREREG-C3-LADDER.md`
**§7.1** (the store-liveness diagnostic: four numeric predictions, three declared verdicts, kill condition
**K6**) and **§2.5**'s P10–P12 — all written **before** the runs they govern exist.

---

## 1. ⭐ `store_layers` — the ratified G-B geometry, as BUILT

**Artifact:** `.claude/outputs/c3-gb-landing/gb-geometry-ledger.json` (measured off **constructed** cells,
not a formula).

| arm | store layers | per-layer cell | **total state B** | occupancy of 2 MiB | within |
|---|---|---|---|---|---|
| `clu_store` | **3 of 12** `(2,6,10)` | 460,288 | **1,380,864** ✅ | **0.65845×** | true |
| `ttt_matched` | 3 of 12 | 456,976 | **1,370,928** | 0.65371× | true |
| `gru_matched` | 3 of 12 | 916 | 2,748 | 0.00131× | true |
| `none` / `echo` | 3 of 12 | 0 | 0 | 0 | true |
| *(same config, 12 layers)* `clu_store` | 12 | 460,288 | 5,523,456 | **2.63380×** ⛔ | false |

⭐ **The ratified arithmetic reproduces to the byte**: `3 × 460,288 = 1,380,864 B = 0.658×`, and the
two-sided **CLU/TTT match ratio is 1.00725×**, inside the pre-registered `[0.99, 1.01]` band — both members
lose the same nine layers, so the swap survives the selection. Pinned by
`tests/test_c3_gb_geometry.py` (**30 cases**).

**How it works.** `resolve_store_layers(n_layers, sel)` (in `blocks.py`, beside the model) resolves the
selection; `StreamModel` puts the arm's cell in those blocks and a `NullMemoryCell` in the rest, **keeping
the same per-layer shell key `ks[2+i]`** so `assert_shared_shell_identical` still holds — across arms *and*
across selections (asserted). `PilotConfig.store_layers` carries it; `arm_ledger` sums the cell term over
the store-bearing layers and the φ term over all twelve, and **prints `n_store_layers`,
`store_layer_fraction` and `store_layer_indices` on every row** so no reader of an artifact ever has to
infer the denominator.

⭐ **`None` is the default and means EVERY layer — never `(0,1,2)`.** The count came from the ceiling; the
*placement* is a design decision, argued in the prereg (§3.1 below) and required on the launch line.

### 1.1 ⭐ The pilot path is provably undisturbed (the task's STOP condition — it did not fire)

`as_flag_table()` emits **non-default fields only**, so an unset `store_layers` is invisible to the resume
fingerprint. Asserted three ways, the last one end-to-end through the **real** verifier:

- `"store_layers" not in PilotConfig().as_flag_table()` and not in the pilot config's;
- against a run-2-shaped journal, `verify_preregistered_continuation` still moves **exactly**
  `['memory.erosion_partition', 'pilot.memory']`;
- ⭐ and the *converse*, which is the protection that matters: a **G-B config is REFUSED** as a
  continuation of a pilot journal, **by name** (`pilot.store_layers`). The ladder cannot ride run 3's
  exemption, and that is now a test.

⚠ **The tuple/list JSON round-trip is safe** (journal stores `[2,6,10]`, config holds `(2,6,10)`; the
fingerprint compares `json.dumps` of both) — verified against a **real journal written by a real toy run**,
because a false refusal here costs a cluster job.

### 1.2 ⚠ G-B's declared price — the TTT criterion still fires (recorded, **not** re-decided)

`solve_matched_ttt` at G-B returns **(k, n) = (2197, 52)** — identical to the pilot's, because G-B shrinks
the *number* of cells, not the cell — so with `η = softplus(0) = 0.69315` and `d = 12`:

**`η·n/d = 3.00364 ≥ 2` ⇒ the criterion FIRES ⇒ `ttt_normalized_write=True` on the TTT arm.**

⛔ **Already ruled** (PILOT-TTT-RULINGS ruling 1). Recorded in the artifact with its basis and stamped as
*not a new decision*; the launch line in `PREREG-C3-LADDER` §6.4 sets it.

### 1.3 ⭐ It actually runs — a full toy S1+S2+S3 under a selection

`--scale toy --stage s3 --quick --set store_layers=1`: **all five arms trained and evaluated, 278 s**,
`monitors_final applicable: true` (at layer 1 — with the old hard-coded 0 it would have been silently
`false`), `selectors_final` reporting `layers: [1]`, and the byte ledger showing `n_store_layers: 1`,
`store_layer_indices: [1]` for every arm. ⛔ Toy bpc values exist in that log and appear in **no** table
here — they are execution evidence only.

⚠ **Three real defects were found by running it and are fixed** (reconciliation item 3), plus a fourth
found by the smoke run itself: `--set store_layers=1` arrives as an **int**, and the first version of the
parser rejected it. It now means *layer 1* — documented at both sites, because "3" meaning *three layers*
is the obvious misreading.

⭐ **One compute change, declared:** the plan pass **skips the host-side controller for non-store layers**.
That is decision-inert *by construction* — a `NullMemoryCell`'s `read`/`write` ignore the plan entirely —
and it is what makes "the store's compute is per store-bearing layer" true of pass 1 as well as of the
forward. ⛔ With the default selection the branch never fires and the pass is bit-identical.

---

## 2. ⛔ RUN 3's LAUNCH BLOCKER B2 — CLEARED (and the one behind it, which is not)

```
$ grep -c 'prereg-continuation' scripts/csf3/job_gpu_cluformer.sh
1
$ bash -n scripts/csf3/job_gpu_cluformer.sh          # clean
```

Two lines, mirroring `job_gpu_c3_seeds.sh:84` and `:169`, plus a comment block saying **why run 3 cannot be
routed through the ladder script**. Asserted by two tests: the script text, and the **emitted command
line** word-splitting into the four/five `KEY=VALUE` argv entries argparse wants (the zsh trap, §7.45,
checked not assumed) — plus a third that drives the **real CLI** end-to-end with those tokens and shows the
refusal comes from the *exemption*, not from argparse.

### 2.1 ⛔⛔ `RUN3-LAUNCH.md` §1–§4, re-read line by line against this branch — **§4 is broken**

Everything else verifies (full table in `RUN3-LAUNCH-DELTA.md`): §2's `.eqx`/`S4.json` names match
`ckpt_path`/`partial_path` exactly for all five arms; **§3's flag re-derivation was executed against the
banked seed-0 journal and reproduces the document's MEM/STORE/SET block byte-for-byte**; §4's exported
variables are all consumed by the script and it passes no `ARMS` (so `arms` stays default); §5's grep keys
exist in the artifact.

**§4 does not work**, measured against the real journal:

```
pcfg.ttt_normalized_write = False        <- passed via --mem: SILENTLY DROPPED (unknown key filtered)
raw pilot.memory carries it: True        <- but the OVERRIDE DICT keeps it

run2 + erosion_partition only        : ACCEPTED  ['memory.erosion_partition', 'pilot.memory']
§4 literal (MEM="… ttt_nw=true")     : REFUSED   pilot.memory: journal={…} now={… ttt_normalized_write:true}
SET="… ttt_normalized_write=true"    : REFUSED   pilot.ttt_normalized_write: journal=<absent> now=true
```

⇒ as written, §4 submits a leg that **does not have the TTT fix on** *and* **is refused before training**.
⛔ **Not fixable by plumbing:** the exemption admits exactly one registered flag and PREREG-LeakAblation §4
admits exactly one token, so *any* route for that flag is a second token — which would invalidate the leak
ablation's attribution **even if the exemption did not exist**. ⭐ **Recommendation (option A): delete
` ttt_normalized_write=true` from the three `MEM=` strings.** Run 3 is the **leak ablation**; ruling 1's
flag belongs to the TTT arm's own leg, and the **C3 ladder is that leg** — fresh geometry, no exemption,
flag set there (§6.4 of the prereg). Nothing run 3 exists for is lost. Two other options costed in the
delta doc. ⛔ **The Hub's call; I changed no exemption logic and did not edit the Hub's document.**

---

## 3. ⭐ `PREREG-C3-LADDER.md` — amended in place, dated, change-marked

⛔ The accepted text is **not rewritten**: an **AMENDMENT 1 block** at the top lists every change with a
pointer, each addition is a **new marked section**, and the only edits to accepted prose are four status
markers (K0 discharged, K0b inert, the G-B row, §9's item 1 closed).

### 3.1 **NEW §2.5 — the layer placement is a DESIGN DECISION, argued**

**`store_layers = (2, 6, 10)`**, generated by a stated rule — *period ⌊12/n⌋, offset so ≥2 blocks sit below
the first store layer and ≥1 above the last* — so that a different count does not reopen the argument
(`n=4` ⇒ `(1,4,7,10)`, **derived, not re-fitted**). Three clauses, each argued: ≥2 below (a layer-0 address
space is close to a bag-of-bytes); ≥1 above (a store at layer 11 can only re-rank — this is why not
`(3,7,11)`); spread not clustered (adjacent chunk summaries are correlated ⇒ three near-copies of one
memory at 3× the bytes). **Five alternatives tabulated with the clause each violates**, including
`(0,1,2)` — named explicitly as *"the default one would get for free by writing `range(3)`"*, which is the
byte-fitting default the section exists to forbid. Precedent named (periodic/hybrid placement: Griffin,
Jamba, Samba, Zamba; "attention at ≈¼,½,¾ depth") **with the novelty stated honestly**: not the
periodicity, but that the interleaved layer is an *addressable store* and the count comes from a
pre-registered byte ceiling.
⭐ **Pre-registered**: **P10** `bpc(2,6,10) − bpc(0,1,2) = 0.000 ± 0.010` (falsifier `|Δ| > 0.02`, whose
firing would be good news twice: placement matters *and* P1 is partly falsified) — ⚠ **conditional on the
§7.1 diagnostic saying the store is in the loss at all**, else it measures one null against another for
11 h; **P11** `frac_recovered` monotone in depth (falsifier: layer 2 strictly highest on ≥2/3 seeds ⇒ the
"compose first" clause is wrong and the repair is `(0,4,8)`); **P12** the cost, with K3 as its falsifier.

### 3.2 **NEW §4.3 — PHASE 1 IS DECLARED, AND PHASE 1 IS NOT THE CLAIM**

States what phase 1 trains (CLU + the TTT swap + GRU/null/echo + dyn-eval + slices + the diagnostic) and
⛔ **that the six pinned rivals are NOT trained here** — none is implemented; `RIVAL_SPECS` is a ledger.
**Therefore charter §2's tier-iii primary claim WAITS for phase 2, and no phase-1 number, table, figure or
abstract sentence may be quoted as it** — set as a block quote, with a ✅/⛔ table of exactly what phase 1
may and may not say, and the operational rule that follows: ⛔ **the rival reference table must never be
printed in the same table as our arms' bpc** (bytes and bpc in one grid is the whole confusion). K1 is
restated as the gate on phase 2 *in the section where someone will look for permission to proceed*.

### 3.3 ⭐⭐ **NEW §7.1 — the store-liveness diagnostic, pre-registered on the FIRST rungs (+ K6)**

The C2 flat-curve disjunction — *carries nothing* **vs** *cannot be addressed* — separated **at real scale,
early**, because the two worlds imply **opposite** next moves and **neither is "train 12 more arms"**.
Built on C2W11's banked separator (same store, same physics, same budgets: shipped read flat at 0.0004,
**oracle-addressed 0.0223 → 0.8219 → 0.8711**, a ~2,000× separation from addressing alone) — ⭐ **the idea
reused, not the toy code**:

- **L0 — is anything written?** `depth_ratio` + `qstar_payload_spread`, from `store_health_probe` **as
  shipped**, read at the three store layers. Zero new physics.
- **L1 — is it ADDRESSABLE?** ⭐ **Oracle-addressed payload recovery**: launch each live item at **its own
  recorded site** (`plan.sites[slot]`, same jitter), settle with the **identical** budget/friction, compare
  `pay(q*)` with **that item's** written payload; `tol` = ½ the minimum between-item payload separation,
  **fixed from the geometry before the measurement**. ⭐ It is a *small extension of an instrument that
  already runs* — `qstar_payload_spread` already launches every item at its own site and throws the
  pairing away. ⛔ Mandatory negative control: the **shuffled** site→item pairing (must be ≈0).
- **L2 — is it in the LOSS?** `bpc` under shipped / oracle-site / **random-site** / blank-store reads on
  the same batches. ⛔ The addressing effect is `random − oracle`, **never** `shipped − oracle`: the oracle
  read is off-distribution for the trained `assim` head and the random-site read is the same perturbation
  *without* the answer. Priced at **≈16 min per job** against a 29.7 h job.

**Four numeric predictions with derivations** (L-P1 `frac_recovered ≥ 0.50` — C2W11's 0.8621 discounted for
live §7.27 erosion; L-P2 `Δbpc ≤ 0.005`; L-P3 `depth_ratio ≥ 0.10` from the 4-seed median 0.5497; L-P4 = P1
re-measured 16,000 steps early) and **three declared verdicts, each a conjunction** so no soft reading can
be talked into a preferred outcome — plus an explicit ⚠ **UNRESOLVED** band that must be *reported as
unresolved, not rounded to the nearest verdict*:

> ⛔ **K6 (new kill condition): `frac_recovered ≤ 0.05` AND `depth_ratio < 0.10` on ≥2 of 3 seeds ⇒ STOP —
> do not submit the remaining 12 jobs.** The store carries nothing, so every other arm measures the shell.
> Re-scope to the write (φ's launch head / `erosion_partition`), where C2 Add.16 already localized the
> blocker. ⛔ **Do not answer it by adding rivals** (K1). ⭐ Cost avoided: ~1.7 days of makespan, 12 A100
> jobs.

Declared limits stated before it runs: `frac_recovered` is a **DIAGNOSTIC** and may never be a capability
number (it hands the read the answer — the same bar C2W11 put on its own 0.8621); the three layers are
reported **separately**, never pooled.

### 3.4 **NEW §6.4 — G-B's launch line** (§6.3 said it did not exist; the work item is done)

Three literal one-line submissions: stage → **the first rungs alone** (`clu_store`, 3 seeds, with the §7.1
diagnostic read off their 4,000-step checkpoints **before** the other 12 jobs go out) → the remaining 12
**only if K6 did not fire**. ⭐ With the ledger check that proves the geometry actually landed:
`n_store_layers: 3`, `store_layer_indices: [2,6,10]`, `total_state_bytes: 1380864`, `occupancy: 0.65845` —
⛔ *if it reads 12, the `--set` did not land, kill the job.*

---

## 4. ⛔ The ledger's remedy text — it was wrong, and a wrong remedy is worse than none

`StateByteBudgetError` told the operator to shrink `capacity`/`atoms_per_item`. Both move **ZERO BYTES** at
`addr_dim = 8`: `n_atoms` is a `max()` including the w23 floor `512·√2⁸ = 8192`, which the pilot's `32×256`
ties exactly (handover 7.38). It now names the levers that **do** move bytes — **`store_layers`** (with the
ratified `1,380,864 B = 0.658×`), `dim` (bounded by the `d ≤ 12` reach ceiling and arithmetically unable to
fit 8192 atoms alone), **`min_atoms_base`** (⛔ marked as a claims-relevant descent below a design-ruled
floor: *pre-register it*), and `n_layers` (⛔ *not* a byte knob — it moves the weight class) — and names
`capacity`/`atoms_per_item` **only to warn that they are inert**. Asserted, including that the old wording
does not survive anywhere in the message.

---

## 5. How I verified (commands + observed output)

| check | result |
|---|---|
| `tests/test_c3_gb_geometry.py` (new, **30 cases**) | ✅ **30 passed** in 53 s |
| **Targeted regression over every touched subsystem** — `test_anti_erosion, test_blocks, test_c3_csf3_harness, test_c3_gb_geometry, test_c3_run3_budget_exemption, test_cluformer_pilot, test_csf3_memory_fit, test_lane_parallel_controller, test_pilot_checkpoint_resume, test_placement, test_placement_probe, test_psi_readout, test_psi_residual, test_ttt_stability_and_d5_wiring` | ✅ **328 passed / 0 failed** in **21:47** |
| **Full suite** | ✅ **1,846 passed / 0 failed** in **45:11** at HEAD `7d33308` — §5.1 (incl. one disclosure) |
| `ruff check chlu/ tests/ scripts/` | ✅ All checks passed |
| `grep -c 'prereg-continuation' scripts/csf3/job_gpu_cluformer.sh` · `bash -n` | ✅ **1** · clean |
| G-B byte ledger off **built** cells → `gb-geometry-ledger.json` | ✅ 1,380,864 B / 0.65845× / ratio 1.00725 / `η·n/d` 3.00364 |
| Toy S1 and **S1+S2+S3** end-to-end under `store_layers` | ✅ 27 s / **278 s**, five arms, monitors applicable at the store layer |
| `RUN3-LAUNCH.md` §3's flag re-derivation vs the banked seed-0 journal | ✅ byte-for-byte identical to the document |
| The §4 submission line vs the **real** run-2 journal, three variants | ⛔ **1 accepted, 2 refused** — §2.1 |
| A G-B journal resuming (tuple/list round-trip) + a different selection refused | ✅ both, against a journal written by a real run |

### 5.1 §Suite — the full test suite: ✅ **1,846 passed / 0 failed**

```
$ PYTHONPATH=/Users/user/Desktop/CHLU-wt1 .venv/bin/python -m pytest -q -p no:cacheprovider --no-cov
1846 passed, 36 warnings in 2711.59s (0:45:11)
EXIT=0
HEAD_BEFORE=7d33308  MAIN_BEFORE=0644c48  DIRTY=0   START=21:11:54
HEAD_AFTER =7d33308  MAIN_AFTER =0644c48  DIRTY=1   END  =21:57:12
```

✅ **Against a named, re-verified HEAD** — `7d33308`, with `main` and the tree checked on **both** sides.

**Count arithmetic, checked not assumed:** `c3-run3-budget-exemption` measured **1,819** at `a656746`
(my base, whose tip is the parent of my first commit). **1,819 + 27 = 1,846** ⇒ ⭐ **exactly my new test
file and nothing else; no existing test changed.**

⚠⚠ **DISCLOSURE — `DIRTY=1` at the end, and it was me.** ~35 % into the run I added three parametrize
cases to **my own new test file** (the bare-int `store_layers` form), so the tree was dirty for the second
half. ⛔ **Nothing under test moved** — the diff is `tests/test_c3_gb_geometry.py | 6 +-`, a test file
pytest had already imported at collection. Handled rather than hidden: the file was **re-run standalone
after the edit → 30 passed**, and the addition is its own commit (`6d1a227`) whose message records that it
post-dates the suite. ⚠ I broke my own "a suite needs a stable tree" discipline here; the fix would have
cost 45 minutes, so I am reporting the exact scope instead of claiming a clean run.

⚠ **One earlier full-suite attempt DIED SILENTLY** (`EXIT=144`, no summary line, no traceback) while
**two** other spokes' `pytest` processes were live on the shared venv — the exact failure handover **7.46**
describes. ⛔ I killed nothing of theirs; I re-ran mine when `ps` showed space. Recorded because a silent
death must never be read as a failing test.

---

## 6. Flag-provenance table (every quantitative result above)

**Commit `6d1a227`** (branch tip; every measured result below was taken at `7d33308`, its parent —
they differ by three test cases only), base `agent/experiment-engineer/c3-run3-budget-exemption @ a656746`,
`main @ 0644c48`. **JAX 0.9.0**, the **main venv reused** via `PYTHONPATH=<worktree>
/Users/user/Desktop/CHLU/.venv/bin/python` (⛔ **no `uv sync` in the worktree** — the w6 hazard),
CPU/float32, macOS, worktree `/Users/user/Desktop/CHLU-wt1`.

| | (A) G-B byte ledger | (B) TTT criterion | (C) toy smoke S1/S3 | (D) run-3 launch probe | (E) tests |
|---|---|---|---|---|---|
| seed | 0 | 0 (`PRNGKey(0)`) | 0 | 0 (+1, 2 read only) | 0 |
| scale | **PILOT** (arithmetic only, ⛔ **no training**) | cell only | **TOY** ⛔ never a claim venue | PILOT (config only) | toy + pilot arithmetic |
| `store_layers` | **`(2,6,10)`** and `None` (both) | `(2,6,10)` | `(1,)` of 2 | `None` (run 3 is pilot geometry) | both |
| non-default `PilotConfig` | PILOT dict as shipped | — | `--quick`: `steps=6, warmup=2, eval_batches=2, dyneval_batches=2, data_bytes=1e6` | reconstructed **from the real journal's own flag table** | per test |
| non-default memory | PILOT | — | TOY | run 2's verbatim **+ `erosion_partition=True`** (± the contaminant) | per test |
| budget | `2,097,152 B` (**INTERIM**, unchanged) | — | same, `enforce=True` | same | same |
| `ttt_normalized_write` | **False** (shipped default) | **False** | **False** | **False** / **True** (the probed variants) | **False** |
| corpus | n/a | n/a | `enwik8`, real | n/a | n/a |
| journal | n/a | n/a | written + resumed by the run itself | **REAL** `csf3_outs/run2/pilot_pilot_seed0_PARTIAL.json` | synthetic fixtures |

⛔ **No number in this report is a claim.** The toy bpc values from (C) appear in no table here.

---

## 7. Git footprint

**Branch `agent/experiment-engineer/c3-gb-landing`**, off **`agent/experiment-engineer/c3-run3-budget-exemption` @ `a656746`** (accepted, unmerged — extended so the Hub merges **once**, with the §2 fix inside),
in worktree **`/Users/user/Desktop/CHLU-wt1`**. ⛔ Not pushed, no PR. ⚠ Per §3.5 I did **not** rebase onto
the stale `origin/main`; rebase onto my named base is a **no-op** (base unmoved at `a656746`, and `main`
unmoved at `0644c48` for the whole task — checked on both sides).

| commit | files | note |
|---|---|---|
| `a8e51fd` | `chlu/core/blocks.py`, `chlu/training/train_cluformer.py`, `chlu/experiments/exp_cluformer_pilot.py` | the `store_layers` selection + `first_store_layer` for the three instruments that hard-coded layer 0 |
| `be99817` | `chlu/eval/byte_ledger.py` | the store-bearing denominator **+** the corrected `StateByteBudgetError` remedy |
| `e0a212f` | `scripts/csf3/job_gpu_cluformer.sh` | **blocker B2**: the `PREREG_CONT` passthrough (+3 lines, +15 of comment) |
| `7d33308` | **new** `tests/test_c3_gb_geometry.py` (+473) | 27 cases |
| `6d1a227` | same (+6/−1) | ⚠ 3 more cases, added **after** the full-suite run (§5.1 discloses it); file re-run standalone → **30 passed** |

```
$ git -C /Users/user/Desktop/CHLU log --oneline c3-run3-budget-exemption..c3-gb-landing
6d1a227  7d33308  e0a212f  be99817  a8e51fd        (5 commits)
$ git -C /Users/user/Desktop/CHLU diff --stat c3-run3-budget-exemption..c3-gb-landing
 chlu/core/blocks.py | 114+ | chlu/eval/byte_ledger.py | 67+ |
 chlu/experiments/exp_cluformer_pilot.py | 19+ | chlu/training/train_cluformer.py | 66+ |
 scripts/csf3/job_gpu_cluformer.sh | 18+ | tests/test_c3_gb_geometry.py | 478+
 6 files changed, 739 insertions(+), 24 deletions(-)
```

**Shared-file hunks, exactly** (⚠ **`chlu/training/train_cluformer.py` and
`chlu/experiments/exp_cluformer_pilot.py` are OUTSIDE the task's named ownership list** — declared, in the
same spirit as the exemption spoke's own out-of-list script hunk, because a `store_layers` that is not a
**config value** would fail the task's §1, and a config value has to be read *somewhere*):

- `train_cluformer.py` (**+66/−9**, five hunks): the `store_layers` field (default `None` ⇒ **no behaviour
  change, no flag-table entry**), its `from_mapping` coercion, the `build_arm` passthrough, the plan-pass
  skip (guarded — inert at the default), and `first_store_layer` in `gradient_probe` / the two probe
  `layer` defaults (`0` → resolved, **identical at the default**).
- `exp_cluformer_pilot.py` (**+19/−6**, one hunk): `_selectors` reads the **store-bearing** blocks and now
  names them (`layers`), instead of raising on a null cell.
- `byte_ledger.py`: `ArmLedger.n_store_layers` (default `None` ⇒ `n_layers` ⇒ **the shipped arithmetic**),
  two new row keys, the arithmetic string, and the remedy text.
- ⛔ **Not touched:** `load_journal`/`_flag_dict`/`_flag_defaults`/`_RESUME_FLAG_EXEMPT` and
  `verify_preregistered_continuation` (**called, never modified** — the STOP held), `chlu/config.py`,
  `chlu/eval/rivals*`, the corpora/registry surface, `job_gpu_c3_seeds.sh`, `MATCHED_STATE_BYTE_BUDGET`,
  `BUDGET_IS_INTERIM`, and run 3's config.

⚠⚠ **Concurrency — verified, and the task's "disjoint by construction" is FALSE.** Two rival worktrees are
live (`wt2` = `c3-rival-mamba2` @ `6464ae4`, `wt3` = `c3-rival-gdn2` @ `baf0166`). Diffing all three
branches against `main`:

| file | me | mamba2 | gdn2 |
|---|---|---|---|
| `chlu/core/blocks.py`, `chlu/eval/byte_ledger.py`, `scripts/csf3/job_gpu_cluformer.sh` | ✅ | — | — |
| `chlu/training/train_cluformer.py` | ✅ (field @~98, `build_arm`, `plan_pass`, probes) | ⚠ **also** (field @~185, **`build_arm`**, `solve_arms`) | — |
| `chlu/experiments/exp_cluformer_pilot.py` | ✅ (`_selectors` @~959) | ⚠ **also** (`main()` @~969) | — |
| `chlu/eval/rivals/*`, `exp_c3_rival_gdn2.py`, `scripts/smoke_c3_*.sh` | ⛔ **never touched** | ✅ | ✅ |

⇒ **`gdn2` is genuinely disjoint from me; `mamba2` is not.** Their `build_arm` hunk ends ~3 unchanged lines
before mine (they replace the `cells = [...]` list; I add `store_layers=` to the `StreamModel(...)` call),
so a textual conflict is unlikely but **possible**, and it is the one hunk a reviewer should look at.
⛔ I did **not** touch their worktrees or branches.

**Worktree-ref verification (protocol §3.2, the lost-8-commits precedent):** all **5** commits confirmed on
the shared ref **from the main repo BEFORE removal** (output above), the worktree confirmed clean
(`git status --porcelain` = 0 lines), then `git worktree remove ../CHLU-wt1`, then **re-verified after**:
tip `6d1a227`, `main..agent/experiment-engineer/c3-gb-landing` = **9 commits** (my 5 + the 4 inherited
exemption commits). ⭐ **wt1 is released** — `wt2`/`wt3` are the live rival spokes' and I never touched
them — so the next spoke told to "take wt1" does not collide with my checkout.

**Artifacts** (all under `.claude/`): `.claude/outputs/c3-gb-landing/gb-geometry-ledger.json`,
`.claude/outputs/c3-gb-landing/RUN3-LAUNCH-DELTA.md`; amendments **in place** in
`.claude/outputs/c3-rival-ladder-prereg/PREREG-C3-LADDER.md`; scratch (probes, logs, smoke artifacts) in
`.claude/scratch/c3-gb-landing/`. Nothing left in the repo.

---

## 8. Open questions / follow-ups / risks

1. ⛔⛔ **Run 3's §4 line (reconciliation 1) — the Hub must rule before submission.** My recommendation is
   option A (delete the token); the alternative that keeps the flag also breaks the leak ablation's
   attribution, so it is not a trade between "safety" and "the fix" — it is a trade between run 3's
   *purpose* and a flag that has a better home on the ladder.
2. ⚠ **`calibrate_phi_gain` / `calibrate_atom_group_centers` still calibrate off layer 0** (reconciliation
   3). Under G-B layer 0 has no store. It moves an **initialisation**, so it needs a ruling, not a patch.
3. ⚠ **`store_layers` cannot be given per-arm.** It applies uniformly, which is what keeps the swap a swap
   and the byte denominators equal — but it also means a future "CLU in 3 layers vs TTT in 12" comparison
   is not expressible. That is deliberate; recording it so nobody discovers it as a limitation later.
4. 🔍 **Not done, out of scope:** §7.1's diagnostic is **designed and pre-registered, not implemented** —
   L1 needs a small extension of `store_health_probe` (keep the site→item pairing) and L2 needs three extra
   eval passes. **That is the next engineering task, and it is on the ladder's critical path**, because K6
   is supposed to fire *before* 12 jobs are submitted.
5. ⚠ **P12's honesty check:** G-B does **not** cut the per-step cost to 3/12 of the pilot's. φ, conv, the
   MLP and the norms still run in all twelve layers; only the store's read/write and its host-side plan are
   per store-bearing layer. The prereg's projected **4.18 s/step** already came from that model, but the
   distinction is now written down where a reader will meet it.

---

## Proposed handover updates (for the Hub)

**§3 (CLI & config) — new surface**

- **`PilotConfig.store_layers: Optional[Tuple[int, ...]] = None`** (CLI: `--set store_layers=2,6,10`;
  a bare int is **one index**, not a count). `None` = **every layer** = the shipped behaviour bit-for-bit,
  so it emits nothing into `as_flag_table()` and **cannot move a banked journal's resume fingerprint**.
  ⛔ Set on the C3 ladder, ⛔ **never** on run 3.
- **`chlu.core.blocks.resolve_store_layers` / `parse_store_layers` / `first_store_layer`** — one resolver,
  used by the model *and* the byte ledger, so the two cannot drift.
- **Byte-ledger rows gained `n_store_layers`, `store_layer_fraction`, `store_layer_indices`**, and the
  `arithmetic` string now states the denominator. ⛔ No new top-level artifact key.

**§7 — entries**

- **7.47 [NEW, RESOLVES 7.39] G-B is built: the store need not live in every layer.** `store_layers` puts
  the full-size 8192-atom cell in **3 of 12** layers ⇒ **1,380,864 B = 0.658× of 2 MiB**, two-sided match
  **1.0072×**, w23 floor intact, no design rule broken. Placement **`(2,6,10)`** is a design decision
  argued in `PREREG-C3-LADDER` §2.5, generated by a stated rule, **not a byte-fitting default**.
- **7.48 [NEW, ⛔ BLOCKS RUN 3] `ttt_normalized_write` cannot be set on run 3 at all.** It is a
  `PilotConfig` field, so via `--mem` it is **silently dropped** *and* contaminates `pilot.memory`; via
  `--set` it is a second key outright. Both are **REFUSED** by the exemption against the real run-2
  journal. ⇒ PILOT-TTT-RULINGS ruling 1 and PREREG-LeakAblation §4 **conflict on run 3**; the flag's home
  is the C3 ladder's own leg. Fix + options: `.claude/outputs/c3-gb-landing/RUN3-LAUNCH-DELTA.md`.
- **7.49 [NEW, corrective, RESOLVES 7.40] `byte_ledger`'s `StateByteBudgetError` remedy is now correct** —
  it names `store_layers`/`dim`/`min_atoms_base`/`n_layers` and names `capacity`/`atoms_per_item` **only as
  byte-inert**. Asserted by test.
- **7.50 [NEW, latent-defect class] Store-side instruments hard-coded layer 0.** The gradient probe and the
  selector readout **raised**; the §7.27 store watch and the monitor pass would have gone **silently
  `applicable: false`** — the worse failure. All now resolve `first_store_layer`. ⚠ **Still open:**
  `calibrate_phi_gain` / `calibrate_atom_group_centers` calibrate off layer 0, which under G-B carries no
  store; it moves an initialisation and needs a ruling.
- **7.51 [NEW, ops, confirms 7.46] A full-suite run died silently at `EXIT=144`** with **two** other
  spokes' suites live on the shared venv — no summary, no traceback. The re-run with space was clean.
  ⇒ check `ps` for `.venv/bin/python -m pytest` before starting one, and ⛔ never read a silent death as a
  failing test.

**§10 (C3W1 block):** G-B is **ratified, built and pinned**; run 3's **B2 is cleared** and **B1 is one
merge**; ⛔ **a third blocker (7.48) is open on run 3's own submission line and needs an Advisor ruling**;
`PREREG-C3-LADDER` now carries the placement argument, the phase-1/phase-2 boundary and a
**pre-registered, kill-conditioned store-liveness diagnostic (K6) on the ladder's first rungs** — whose
implementation is the next task on the ladder's critical path.
