# PREREG — `gen-wells-route-b`

**Filed:** 2026-09-15, **before** any measurement of the quantities below.
**Tree at filing:** worktree `../CHLU-routeb`, branch `agent/experiment-engineer/gen-wells-route-b`, base
`main @ 115d4b5`. ⛔ No harness existed at filing time: `chlu/core/gen_wells.py`,
`chlu/experiments/exp_gen_wells_route_b.py` and `tests/test_gen_wells.py` were all absent from the tree
(`git ls-tree main | grep gen_wells` → empty). This document is therefore a prediction, not a description.

---

## 0. ⭐ DIAL DECLARATION (echoed verbatim from the task file, protocol §7)

- **Dial:** ⭐ **WHERE THE WELLS GO** — well placement learned with the store in the loop, replacing a partition
  inherited from clustering. ⛔ Not the read, not the decoder, not the encoder.
- **Laundering control:** ⭐⭐ **D-matched RANDOM placement at matched bytes.** Binding; see §4.
- **Falsifies:** the claim that our wells are semantic. ⚠ It may show placement was never the lever.
- **Does NOT falsify:** losing to the *pooled-code* retention ceiling, or failing to recover `scale`/`orientation`
  information that `gen-loss-attribution` already localised at the **codebook** rather than the partition. That is a
  known confirmation of a measured result (§2), not news, and it does not falsify a placement claim.

## 0a. ⛔ Terminological correction, filed before it can propagate

The task file says the partition is *"inherited from `k`-means"*. **It is not.** At `main @ 115d4b5` the shipped
partition is **hierarchical linkage** on the channel-correlation matrix — `gen_store.correlation_grouping` walks
`GROUPING_ORDER = ("average", "ward")` and the shipped Shapes3D/`recon` plan records `method_used: "ward"`
(`.claude/outputs/gen-store-grouping/runs/shapes3d_recon/store_plan.json`). **k-means++ seeds the MODES**
(`gen_store.init_store` → `_kmeanspp_init`), not the partition. Both are inherited and both are in scope for
"placement", but a report that calls the partition k-means is wrong and would mislead the next reader.
⇒ **Reconciliation item, owner Hub** (also listed in the report's first 10 lines).

---

## 1. What is being learned, exactly

The shipped store plan pins the contract and says so in its own `note` field:

> "Every later factorisation-ladder rung loads this file VERBATIM. K, n_modes, D and store_bytes are pinned here;
> **only the grouping (the `labels` field) is allowed to change between rungs.**"

⭐ **Route B is exactly that: a new `labels` vector.** `K = 5`, `n_modes = 64`, `D = 2048`,
`store_bytes = 524,288 B`, `bits_per_read = 30.0` are pinned and are **identical in every arm**.

The learned object is a soft assignment `A ∈ R^{D×K}`, `a_k = softmax_k(A)[:, k]`, optimised jointly with a
relaxed store (§3), then **hardened to a partition whose well-size histogram is EXACTLY the reference
`[303, 305, 573, 501, 366]`** (§4). Every scored number then comes from the ordinary
`init_store` + `train_store` path on that hard partition, so an arm differs from another arm **only in its label
vector** — the discipline `exp_gen_store.run_arm` already enforces.

⚠ **Declared cost asymmetry, stated here so a referee does not have to find it.** The *relaxation* holds `K·M·D`
floats while it is being optimised; the *delivered store* holds `Σ_k M·d_k = M·D`. The byte ledger (§5) governs the
delivered store, which is byte-identical across arms. The relaxation's transient cost is a **training** cost, is
reported as such, and is ⛔ never netted against the store budget.

## 2. ⚠ THE PRIOR EVIDENCE THIS RUN MUST NOT PRETEND NOT TO HAVE

`gen-loss-attribution` (3/3 seeds) already measured where the information dies:

| factor | encoder | after partition | after codebook | partition's share of the loss | codebook's share |
|---|---|---|---|---|---|
| `scale` | 0.9352 | 0.8753 | 0.3069 | **9.5 %** | ⭐ **90.5 %** |
| `orientation` | 0.9915 | ~1.0037 retention | 0.2909 | **0.0 %** | ⭐ **100 %** |

⇒ **The partition is already exonerated as the carrier of the pooled loss.** Any honest prereg for a
placement spoke must predict accordingly, and this one does (§6).

## 3. The objective — label-free primary

`L = L_rec + λ_occ · L_occ + λ_fact · L_fact`, all three computed from **features and the store alone**:

1. **`L_rec` — within-well fit.** `Σ_k mean_n ‖ z_k − π_k(z) M_k ‖²` under the membership-weighted distance
   `d²_k(n,m) = Σ_d a_k[d] (x[n,d] − M_k[m,d])²`, with `π = softmax(−d²/2τ²)` annealed `τ_start → τ_end`. This is
   `train_store`'s own soft term, generalised from a hard gather to a soft membership.
2. **`L_occ` — within-well MULTIMODALITY.** `−Σ_k H(mean_n π_k)`: a well whose mass sits on one mode is not
   multimodal, it is collapsed. ⭐ This term is the direct counter-pressure to the collapse falsifier (§7).
3. **`L_fact` — between-well FACTORISATION.** `Σ_{k<l} I(π_k ; π_l)` estimated on the batch from
   `P̄(m,m') = mean_n π_k[n,m] π_l[n,m']`. Minimising it pushes the wells' codes towards independence.

⛔ **No factor label enters any of the three.** Enforced two ways and both must hold (§8).

**Arms, declared here so neither can be swapped in later:**

| arm | what it is | may it be quoted as "the label-free result"? |
|---|---|---|
| `learned_label_free` | ⭐ **THE PRIMARY.** `L` as above. | yes |
| `learned_weak_sup` | `L + λ_sup · L_sup`, `L_sup` a factor-label term | ⛔ **NEVER.** Separate row, always. |
| `random_dmatched` | ⭐⭐ **THE FLOOR.** `random_grouping(reference, seed)` | n/a — it is the floor |
| `correlation` | the shipped `ward` partition (route A's) | n/a — context row |

⭐ If the label-free arm fails and the weakly-supervised arm works, **that is the finding, and it will be reported
in those words.**

## 4. ⛔⛔ The control that can kill this — built BEFORE the objective

**D-matched random placement, matched bytes, same everything else.** `random_grouping` permutes the *reference
label vector*, so the well-size histogram is held **exactly** and the only thing that varies is which coordinate
sits in which well. The learned arm is hardened to that same histogram by a **size-constrained assignment**, so all
four arms are D-matched by construction rather than by luck.

⭐ **Every placement number in the report is a MARGIN OVER THIS FLOOR. ⛔ Never an absolute.**
This programme has been burned exactly here: an untrained random-init encoder scored 0.9361 / 0.9865 against a
trained encoder's 0.935 / 0.991 for **1,246× less compute**. A sharpness number without its floor beside it is void.

## 5. Byte discipline

`store_bytes = Σ_k M·d_k·4 = M·D·4 = 64·2048·4 = 524,288 B` for **any** partition of `D`; the coupling identity
`Σ_{k<l} r(d_k+d_l)·4 = r(K−1)D·4` likewise holds for any partition. ⭐ Asserted **in code**, per arm, per seed,
against the pinned plan; **the run aborts on a mismatch.** A per-seed drift in well sizes may not silently
renegotiate the budget.

## 6. ⭐⭐ THE PREDICTIONS — committed before the harness exists

**The metric.** Per factor `f`, per well `k`:
`S_fk = 1 − H(p)/log(n_values_realised(f))`, `p(v) = P(f = v | well k's mode)` averaged over modes under `P(m|k)`.
Equivalently `S_fk = 1 − H(f | code_k)/log V_realised`; at a uniform realised marginal this is
`I(f ; code_k)/H(f)`. ⛔ `n_values_realised` on **every row**; `V_realised = 1` ⇒ `log 1 = 0` ⇒ **S is UNDEFINED
and is reported as `null` with `degenerate: true`** — ⛔ never 0, ⛔ never 1. (Montero's corner pins `shape` to 1 of
4 and has already produced one wrong headline in this programme.)

### 6a. Predicted floor levels — `random_dmatched`, `S_f = max_k S_fk`, Shapes3D/`recon`, seed 0

Derived from `gen-loss-attribution`'s measured code-level retention (the code keeps ~99.7 % of the hue signal,
0.665 `object_hue`, 0.555 `shape`, 0.225 `scale`, 0.243 `orientation` of above-chance probe signal) mapped onto a
normalised-MI scale, capped by `log2 M = 6` bits per well against `H(f)`:

| factor | `V` | `H(f)` bits | predicted `S_f` (floor) |
|---|---|---|---|
| `floor_hue` | 10 | 3.32 | **0.35 – 0.60** |
| `wall_hue` | 10 | 3.32 | **0.35 – 0.60** |
| `object_hue` | 10 | 3.32 | **0.15 – 0.40** |
| `shape` | 4 | 2.00 | **0.08 – 0.25** |
| `scale` | 8 | 3.00 | **0.02 – 0.12** |
| `orientation` | 15 | 3.91 | **0.01 – 0.10** |

### 6b. ⭐⭐ THE HEADLINE PREDICTION — I predict the NULL, and I say why

**Clause 1 (pooled).** Pooled code retention of `scale`/`orientation` is **unchanged** by learned placement:
`|Δ| ≤ 0.03` absolute. *Derivation:* `gen-loss-attribution` localised 90.5 % / 100 % of that loss at the
**codebook**. A partition repair cannot recover a loss that is not at the partition.

**Clause 2 (per-well sharpness) — the genuinely open one.** `max_k S_fk` is a *different* quantity from pooled
retention: a partition change can concentrate a factor into one well without changing total information. This is
the only clause route B can win on.
⭐ **Predicted:** the label-free margin over the D-matched random floor is **small — mean over factors
`ΔS < 0.05` absolute, and no factor exceeds `+0.10`.**
⭐ **Predicted verdict: route B does NOT clear its own bar on the label-free arm.**

**What would refute this prereg** (i.e. what makes route B a real result):
`learned_label_free` beats `random_dmatched` by **`ΔS ≥ 0.10` on at least one of `scale`/`orientation`/`shape`,
3/3 seeds, with well occupancy NOT collapsed** (§7). ⭐ That is a finding and it sends route B to the cluster.

**What refutes SEMANTIC PLACEMENT** (the dial's own falsifier, stated in the task's words):
⭐ **learned placement does not beat D-matched random placement.** Then placement was never the lever, the
inherited partition was not the problem, and ⛔ **route B does not go to the cluster.** ⚠ On the evidence in §2
this is the outcome I expect, and saying so now is the point of filing this.

## 7. ⭐ The collapse trap — pre-declared, not discovered later

Sharpness can rise by **collapsing wells**: one mode eats the mass and the address book looks sharp because there
is nothing left to choose between. ⛔ **Well occupancy is reported BESIDE every sharpness number** — per well:
`used_modes` (of 64), occupancy entropy in bits (of `log2 64 = 6`), and max mode share.
⭐ **Pre-registered guard:** a sharpness gain accompanied by `mean_used_modes` falling below **80 % of the floor
arm's** is reported as **COLLAPSE, not as a win**, in those words.

## 8. ⭐ The label-free assertion, and the leaky variant that must make it FIRE

Two independent checks, because either alone is weak:

1. **Structural** — the label-free objective's signature must not admit a labels argument.
2. **Numerical** — evaluate the objective on **permuted factor labels**; loss and gradients must be
   **bit-identical**. If a label leaked through the partition, the bridge or the atom selection, they would move.

⛔ **An assertion that has never fired is not an assertion.** The same check is run against
`learned_weak_sup` (the deliberately leaky variant) and **must FAIL there**, and that failure is shown in the
report as evidence the check has teeth.

## 9. ⛔ Preconditions that fail at STARTUP or not at all

Every one of these is asserted before the run does work — ⛔ never after a scheduler has allocated a GPU:

1. **Artefact roots exist**, by name, with the missing name printed (`env.sh` + the jobscript).
2. ⭐ **Cache coverage.** Measured, not assumed: seed 0's cache is `scope="full"` (**480,000** rows, 1.83 GiB of the
   1.95 GiB staged); seeds 1 and 2 are `scope="used"` (**9,288 each**), `dtype=float16`, `D=2048`, and
   **`encoder_param_hash` DIFFERS per seed** — they are not interchangeable. A production run needs
   `n_store_train = 32,768` rows plus eval/matched-seen, so **seeds 1 and 2 fall short** and would silently enter
   `gather_features`' extend-by-re-encode path. ⛔ The run **names the seed and the shortfall and dies at startup.**
   ⚠ If that path is taken deliberately it is reported as an **exercised untested path**, ⛔ not a normal run.
3. **Byte ledger** matches the pinned plan (§5) — abort on mismatch.
4. ⭐ **`dec_recipe = "v1_capacity_schedule"`**, asserted by **name AND per-seed param hash** from
   `winner_hashes.json` (seed 0: `01ef91f7…`). ⛔ The default is `"baseline"` — the decoder that FAILED the gate;
   a baseline run looks normal in the logs and is **void**.
5. ⭐ **Arm keys by `fold_in`** (`gen_landscape.arm_key`), ⛔ never by lengthening a split — `split(k,2)[1]` is
   bit-identical to `split(k,3)[1]` on JAX 0.9.0 here, so lengthening hands two arms the same key.

## 10. ⛔⛔ Checkpointing — a standing Head requirement and an acceptance item

A run that dies at hour 70 of 96 **must resume**, ⛔ never restart. Following
`.claude/outputs/pilot-checkpoint-resume.md` (⛔ not a second invented pattern):

1. **Time-based** checkpoints, not only epoch-based — an OOM does not wait for an epoch boundary.
2. ⭐ **Atomic writes**: temp file + `os.replace`. A crash *during* a write must not corrupt the last good one.
3. **Keep the last N good checkpoints**, not just one.
4. **Resume verifies** step and config hash and ⛔ **refuses a checkpoint from a different config.**
5. ⭐ **`~/scratch`** — ⛔ never `$TMPDIR` (wiped at job end), ⛔ never `$HOME` (quota, backed up).
6. ⭐ **Instrumented**: the resume path prints what it recovered and from which step.

⭐⭐ **And it is demonstrated by KILLING it**: start the verification run, `kill -9` mid-flight, restart with
`--resume`, and show it picks up rather than restarting — with a **bit-identity check against the uninterrupted
run**, the same gate `pilot-checkpoint-resume` passed. ⛔ A resume path that has never survived an actual kill is
not a resume path. ⛔ This item may not be waived for time.

## 11. Cost and scope

⛔ **CSF3 needs 2FA and I cannot reach it.** Every cluster leg is reported **`NOT RUN`, naming the leg**; the
deliverable is submit-ready lines, not results. Partition `-p gpuA` (A100 SXM4 80 GB, free at point of use, batch
max `-t 4-0`, `-G N` required); ⛔ never `-p serial`.
⚠ The §12 "A100-80GB does not exist" ruling is about **GCP**, where it is quota-zero. CSF3's `gpuA` A100-80GB is a
different resource and is free at point of use — both lines are given separately in the report, and ⛔ neither is
quoted as the other.
The **verification run** (acceptance item 3) is local/L4-class, is ⛔ **not scored**, and ⛔ **no number from it
enters any table.**

## 12. Artefacts

`.claude/outputs/gen-wells-route-b/` — `PREREG.md` (this) · `runs/<dataset>_<pretext>/seed_*/` ·
`floor_first/` (the random floor, measured first) · `sharpness.json` · `byte_ledger.json` · `resume_proof/` ·
`submit.sh` · the report at `.claude/outputs/gen-wells-route-b.md`.
⭐ The verification run's artefacts live under `resume_proof/` (`uninterrupted/` and `killed/`) and are
⛔ **deliberately not** at the scored `runs/<dataset>_<pretext>/seed_*/` paths, so a smoke can never be mistaken for
a scored run. Every such `summary.json` additionally carries `"scored": false` and a `NOT RENDERED` verdict.
