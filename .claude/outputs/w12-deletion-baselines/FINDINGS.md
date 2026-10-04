# w12 spoke/deletion-baselines — FINDINGS

**Box FORGIS_14 · branch `w12/deletion-baselines` · worktree
`C:/chlu/worktrees/CHLU-w12-deletion-baselines` · 3 threads · 2026-09-21.**
⛔ **SCORED RUNS SPENT: 0.** No held-out test corner, no query set, no scored request was read.
Every row touched comes from the value-holdout condition's **fit pool**.
⭐ `chlu.__file__` = `C:\chlu\worktrees\CHLU-w12-deletion-baselines\chlu\__init__.py` — asserted
in code (`assert_chlu_in_worktree`) before any number, and recorded in every artefact.
⛔ **Nothing in this spoke read `$CHLU_CLAUDE_ROOT`** — see §7.

---

## 0. The one job, and whether it is done

| acceptance clause | status | evidence |
|---|---|---|
| three comparators built | ⭐ **DONE** (four, counting both unlearning methods) | `chlu/eval/rivals/w12_deletion_baselines.py` |
| wired into the rows-3/4 harness, called BY NAME | ⭐ **DONE** | `BASELINES-READY.json` → `how_the_hub_calls_them`; checked from a process that never imports w12 |
| `BASELINES-READY.json` names each comparator + entry point + matched param count + cost fields | ⭐ **DONE** | `/c/chlu/outputs/w12-deletion-baselines/BASELINES-READY.json` |
| a smoke call of each returns a number | ⭐ **DONE** — **150 records, seeds 0 / 1 / 2** | `BASELINES-READY.json` → `per_seed.*.records.*.smoke` |
| every row prints seconds · bytes rewritten · data rows read | ⭐ **DONE, and ENFORCED** | `assert_cost_line_complete` refuses an incomplete cost line |

---

## 1. ⭐⭐ THE HEADLINE: the two approximate unlearners BRACKET the exact reference

Forget minus retain, feature MSE against the cached generation-2 ground truth. **Bold/large figure
= mean over seeds 0 / 1 / 2; the small figures beneath are the three seeds in order.** Source:
`BASELINES-READY.json` → `per_seed.{0,1,2}.records.*.smoke.forget_minus_retain`, and the three
`build_seed{0,1,2}.log`. Higher = more forgetting of the deleted region relative to the survivors.
⭐ **Every conclusion below holds on 3 of 3 seeds**; none rests on a single seed.

**VALUE deletion — `object_hue = 0.0` ("red"), 1,352 of 16,384 fit rows:**

| arm | as-trained | ⭐ retrain_without (EXACT) | unlearn_neggrad | unlearn_finetune |
|---|---|---|---|---|
| mlp | -0.00189 <br><sub>-0.0009 / -0.0013 / -0.0034</sub> | **+0.03661** <br><sub>+0.0332 / +0.0406 / +0.0360</sub> | +0.51518 <br><sub>+0.4951 / +0.5042 / +0.5463</sub> | +0.00740 <br><sub>+0.0078 / +0.0066 / +0.0078</sub> |
| cont_mlp | +0.00935 <br><sub>+0.0100 / +0.0112 / +0.0068</sub> | **+0.03817** <br><sub>+0.0388 / +0.0392 / +0.0365</sub> | +0.35688 <br><sub>+0.3447 / +0.3652 / +0.3608</sub> | +0.03832 <br><sub>+0.0357 / +0.0402 / +0.0390</sub> |
| transformer | -0.00054 <br><sub>+0.0003 / +0.0003 / -0.0023</sub> | **+0.06057** <br><sub>+0.0585 / +0.0661 / +0.0571</sub> | +1.26223 <br><sub>+1.4503 / +1.0718 / +1.2646</sub> | +0.01394 <br><sub>+0.0192 / +0.0092 / +0.0135</sub> |
| lstm | -0.00101 <br><sub>-0.0012 / +0.0000 / -0.0019</sub> | **+0.04111** <br><sub>+0.0277 / +0.0197 / +0.0759</sub> | +0.63285 <br><sub>+0.5671 / +0.5928 / +0.7386</sub> | +0.01174 <br><sub>+0.0117 / +0.0078 / +0.0157</sub> |

⛔ **NegGrad+ over-forgets by 9.4× – 20.8× against the exact reference** (mlp 14.1× · cont_mlp
9.4× · transformer 20.8× · lstm 15.4×, on the 3-seed means). ⛔ **Fine-tune-on-retain
under-forgets on three arms of four** (0.20× · 1.00× · 0.23× · 0.29× of the exact answer). Neither
approximates the exact answer, and they miss it in **opposite directions** — on every seed.
⭐ **Both were given every gradient row-visit the exact retrain got**: 296,960, asserted equal
across all eight arm×method combinations on all three seeds
(`per_seed.*.matched_compute.value.matched = true`). ⛔ So this is **not** a starved-comparator
artefact; it is what these methods do at full budget.

⚠ The as-trained column is the control and it behaves: −0.001 … +0.010, i.e. an arm that has
deleted nothing shows no forget/retain gap. ⛔ Without that column the retrain numbers would be
unreadable.

---

## 2. ⭐⭐ THE CONJUNCTION ROW CONFIRMS THE HEAD'S FRAMING, ON THE BASELINES THEMSELVES

**CONJUNCTION deletion — `object_hue = 0.0 AND shape = 2.0`, 667 of 16,384 fit rows:**

| arm | as-trained | ⭐ retrain_without (EXACT) | unlearn_neggrad | unlearn_finetune |
|---|---|---|---|---|
| mlp | -0.00914 <br><sub>-0.0078 / -0.0091 / -0.0105</sub> | **-0.00289** <br><sub>-0.0020 / -0.0029 / -0.0037</sub> | +0.48965 <br><sub>+0.4696 / +0.4919 / +0.5075</sub> | +0.00128 <br><sub>+0.0029 / -0.0000 / +0.0010</sub> |
| cont_mlp | +0.00750 <br><sub>+0.0086 / +0.0099 / +0.0040</sub> | **+0.01907** <br><sub>+0.0202 / +0.0212 / +0.0158</sub> | +0.33962 <br><sub>+0.3268 / +0.3500 / +0.3420</sub> | +0.01459 <br><sub>+0.0116 / +0.0183 / +0.0139</sub> |
| transformer | -0.00708 <br><sub>-0.0056 / -0.0072 / -0.0084</sub> | **-0.00179** <br><sub>-0.0011 / -0.0025 / -0.0018</sub> | +1.33207 <br><sub>+1.4910 / +1.4085 / +1.0968</sub> | +0.00243 <br><sub>+0.0022 / +0.0024 / +0.0027</sub> |
| lstm | -0.00649 <br><sub>-0.0056 / -0.0063 / -0.0076</sub> | **-0.00026** <br><sub>+0.0007 / -0.0004 / -0.0011</sub> | +0.59847 <br><sub>+0.5738 / +0.6077 / +0.6139</sub> | +0.00168 <br><sub>+0.0010 / +0.0016 / +0.0025</sub> |

⭐⭐ **Retraining a learned arm from scratch WITHOUT the red∧shape-2 rows leaves red∧shape-2
reconstruction essentially unchanged** (3-seed means −0.00289 / +0.01907 / −0.00179 / −0.00026;
sign-stable and below 0.004 in magnitude on every seed for mlp, transformer and lstm), because red
and shape-2 each survive separately and the arm composes them. ⭐ That is the wave-12 plan's
sentence — *"a compositional store can still compose 'red cube' from 'red' and 'cube' after the
red-cube rows are gone — that is composition working, not deletion failing"* — **measured on the
LEARNED BASELINES**, so it is a property of compositional generation and not a special pleading
for our store. ⛔ **It also means the conjunction RESULT row cannot carry the deletion claim; the
refusal DISPLAY row is what the hole is for.**

---

## 3. ⛔⛔ AN ITEM DELETION MAKES NOTHING IMPOSSIBLE — DECLARED NOT-RUN, NOT A ZERO

**ITEM deletion — a random 1% of fit rows, 164 of 16,384.**

The first seed-0 pass **stopped itself** with
`AssertionError: 164 of 164 IMPOSSIBLE requests were NOT certified impossible`. ⭐ That was my own
certifier refusing to train a refuser on a label it cannot support, and the fact behind it is a
finding:

> ⛔ A random item deletion clears **no** `(factor, value)` bit and punches **no** conjunction
> hole. Every deleted row's factor tuple is still composable from values the pool still holds.
> **0 of 164** requests built from the deleted rows are certifiably impossible — **on all three
> seeds** (`BASELINES-READY.json` → `per_seed.*.DECLARED_NOT_RUN.measured_not_applicable`).

⭐ So **the item-deletion row has no refusal DISPLAY row at all.** Refusing a deleted *item* would
need a per-tuple deletion table — a store bolted to a generator, which is exactly the objection
`chlu/experiments/exp_gen_refuse.py` already makes one level coarser about the pair-deletion
table. `refusal_row_applies(grid, part, bitmap)` returns this as a **measurement**, and the runner
prints a DECLARED NOT-RUN carrying the count. ⛔ Never a zero.

The item row's other comparators did run:

| arm | as-trained | ⭐ retrain_without (EXACT) | unlearn_neggrad | unlearn_finetune |
|---|---|---|---|---|
| mlp | -0.00227 <br><sub>-0.0027 / -0.0029 / -0.0013</sub> | **-0.00210** <br><sub>-0.0025 / -0.0027 / -0.0011</sub> | +0.26322 <br><sub>+0.2409 / +0.2685 / +0.2803</sub> | -0.00243 <br><sub>-0.0035 / -0.0029 / -0.0009</sub> |
| cont_mlp | +0.00292 <br><sub>+0.0024 / +0.0020 / +0.0044</sub> | **+0.00304** <br><sub>+0.0025 / +0.0020 / +0.0045</sub> | +0.08767 <br><sub>+0.0871 / +0.0840 / +0.0919</sub> | +0.00385 <br><sub>+0.0045 / +0.0020 / +0.0051</sub> |
| transformer | -0.00222 <br><sub>-0.0023 / -0.0027 / -0.0016</sub> | **-0.00208** <br><sub>-0.0023 / -0.0026 / -0.0014</sub> | +0.83286 <br><sub>+0.8703 / +0.9469 / +0.6814</sub> | -0.00237 <br><sub>-0.0025 / -0.0032 / -0.0015</sub> |
| lstm | -0.00172 <br><sub>-0.0024 / -0.0023 / -0.0005</sub> | **-0.00152** <br><sub>-0.0021 / -0.0019 / -0.0005</sub> | +0.37437 <br><sub>+0.3829 / +0.3698 / +0.3703</sub> | -0.00115 <br><sub>-0.0014 / -0.0005 / -0.0015</sub> |

⭐ **The exact reference and the as-trained arm agree to within 0.000354 — the maximum over all
four arms and all three seeds** — i.e. deleting 1% of rows at random changes a matched-bytes
learned arm by nothing measurable, and the exact retrain confirms it. ⛔ NegGrad+ still destroys
the deleted region (+0.088 … +0.833 on the 3-seed means) **for a deletion the exact answer says
costs nothing**, which is the clearest statement of what "approximate" buys here.

---

## 4. ⭐ THE COST LINE — seconds · data rows read · bytes rewritten, for every arm

16,384 fit rows. ⭐ **`data_rows_read` and `bytes_rewritten` are seed-INVARIANT** — the partition
comes from the pinned pool draw, not from the seed — and both were checked identical across seeds
0/1/2. Only `seconds` varies with the seed; the figures below are **seed 0**
(`BASELINES-READY.json` → `per_seed.0.records.*.cost`). `bytes_rewritten` is the arm's whole
parameter blob; `param_bytes_changed` is **measured** against the as-trained checkpoint.

**VALUE deletion (`object_hue = 0.0`):**

| comparator | seconds | data rows read | bytes rewritten | param bytes CHANGED | gradient row-visits |
|---|---|---|---|---|---|
| mlp retrain_without | 8.86 | **15,032** | 522,056 | 522,056 | 296,960 |
| mlp unlearn_neggrad | 9.25 | 2,704 | 522,056 | 521,568 | 296,960 |
| mlp unlearn_finetune | 9.88 | 1,352 | 522,056 | 521,324 | 296,960 |
| cont_mlp retrain_without | 8.87 | **15,032** | 527,564 | 527,564 | 296,960 |
| cont_mlp unlearn_neggrad | 9.99 | 2,704 | 527,564 | 527,564 | 296,960 |
| cont_mlp unlearn_finetune | 7.25 | 1,352 | 527,564 | 527,564 | 296,960 |
| transformer retrain_without | 34.62 | **15,032** | 521,732 | 521,732 | 296,960 |
| transformer unlearn_neggrad | 31.77 | 2,704 | 521,732 | 520,292 | 296,960 |
| transformer unlearn_finetune | 29.33 | 1,352 | 521,732 | 520,112 | 296,960 |
| lstm retrain_without | 15.99 | **15,032** | 522,884 | 522,884 | 296,960 |
| lstm unlearn_neggrad | 18.01 | 2,704 | 522,884 | 521,252 | 296,960 |
| lstm unlearn_finetune | 14.47 | 1,352 | 522,884 | 521,048 | 296,960 |
| transformer refuse_token | 10.52 | 4,096 | 521,916 | 521,916 | 92,160 |

Conjunction leg: retrain_without reads **15,717** rows (8.51 / 9.22 / 31.80 / 15.66 s at seed 0);
item leg: **16,220** rows (10.28 / 10.62 / 39.66 / 22.25 s). Full tables for all three seeds in
`BASELINES-READY.json` → `per_seed`. Whole-spoke wall clock **884.9 / 834.1 / 566.3 s** for seeds
0 / 1 / 2 — 3 threads, CPU only, ⭐ **0 GPU**.

⭐ **The sentence this buys for every deletion row:** the exact reference reads **15,032 / 15,717 /
16,220 rows of data and rewrites ~522 KB of parameters**, taking 8.5–39.7 s per arm, while
approximate unlearning at the *same* compute still reads 164–2,704 rows and still rewrites
essentially every parameter byte (520,112–527,564 of 520,112–527,564 B). ⛔ **There is no learned
comparator here whose post-deletion adaptation reads zero rows of data.**

⚠ The store's own cost line (0 rows read) is a **DECLARED NOT-RUN** for this spoke — see §6.

---

## 5. ⚠ THE REFUSE-TOKEN TRANSFORMER SEPARATES AT +0.99 — AND THAT IS NOT IMPRESSIVE

`BASELINES-READY.json` → `per_seed.*.records.transformer__refuse_token__*.separation`:

| deletion | separation, seeds 0 / 1 / 2 | AUROC | refuse @ impossible | false refusal @ possible | n_imp / n_pos |
|---|---|---|---|---|---|
| value `object_hue = 0.0` | **+0.9911 / +0.9924 / +0.9928** | 1.0000 (3/3) | 1.0000 (3/3) | 0.0000 (3/3) | 676 / 4,096 |
| conjunction `red ∧ shape-2` | **+0.9887 / +0.9903 / +0.9906** | 1.0000 (3/3) | 1.0000 (3/3) | 0.0000 (3/3) | 334 / 4,096 |
| item (1% random) | ⛔ **DECLARED NOT-RUN** (3/3) | — | — | — | 0 certified impossible of 164 |

⭐ **Reported against w9's bar, which this spoke was told to expect:** w9's maze refuser separated
at **+0.047** (cmaze1) / +0.020 (easy) with a ridge probe at chance. ⛔ **Do not read +0.99 as the
Shapes3D refuser being 20× better than the maze one.** They are different problems:

- For a **VALUE** deletion the impossible set is exactly "the request carries the deleted
  `object_hue` token". A token-vocabulary transformer sees that token directly, so the task is a
  one-token lookup. ⛔ It is a *learned reproduction of something free*: the 8-byte value-presence
  bitmap plus bracket check already scores this at **1.0000** for 8 bytes and 0 rows read
  (`exp_gen_factor_wells.value_presence_bitmap` + `accept_rule`).
- For a **CONJUNCTION** deletion the free rule **cannot express the question at all** —
  `exp_gen_refuse.bitmap8_bits_changed_by_a_conjunction_deletion` returns the integer **0** by
  construction. ⭐ **This is the number that matters:** the refuse-token transformer learns the
  conjunction hole at **+0.9887 / AUROC 1.0000**, and it costs **130,479 parameters, 4,096 data
  rows read, 521,916 B rewritten and 10.48 s**. That is the price of the alternative to a coupling
  hole. ⛔ The comparator's job is to price it, not to lose.

**⚠ DECLARED DIAGNOSTIC — the out-of-range leg the family structurally cannot see.** A `tokens`
input is a closed vocabulary of grid classes, so an off-grid request is encoded as its *nearest
grid class* and this architecture cannot represent out-of-range-ness. Including it is therefore
excluded from the primary set by declaration, and measured separately rather than argued:
separation drops to **+0.4549 / +0.4717 / +0.4673** (AUROC 0.8692 / 0.8747 / 0.8755) on the value
leg and **+0.3671 / +0.3747 / +0.3455** (AUROC 0.8269 / 0.8340 / 0.8133) on the conjunction leg,
with false refusal rising from 0.0000 to 0.0947 / 0.0781 (seed 0). ⛔ **A 2.1×–2.9× drop in
separation, caused purely by an input encoding that cannot represent the question.**
⭐ Out-of-range refusal already has its own exact, free comparator; it is not this one's row.

**⛔ USAGE CAVEAT the hub must carry:** the refuse-token arm's FEATURE head is trained with the
feature term **masked on impossible requests** (there is no ground-truth feature for a request
the pool cannot answer, and regressing towards a fabricated one would be inventing data). Its
smoke feature MSE on deleted requests is therefore **1.05** by construction.
⛔ **It is a refusal comparator, not a generative comparator for the deleted region.**

---

## 6. ⛔ DECLARED NOT-RUN — every one with a reason, none a zero or a blank

1. **Our own arm's deletion cost line (0 data rows read).** NOT MEASURED here. It belongs to the
   object the crossover hub builds at **K\***, which does not exist yet; measuring it against a
   K=5 stand-in would print a cost for an object no table carries. The comparator side of every
   cost line is delivered; our side is the hub's to fill at K\*.
2. **The refusal DISPLAY row for ITEM deletion.** Measured as inapplicable — 0 of 164 deleted-row
   requests are certifiably impossible (§3). ⛔ Not a zero refusal rate; **no row exists**.
3. **Any judge / external-classifier accuracy.** These comparators emit FEATURES. Rendering and
   judging is the crossover hub's, with its own frozen decoder and hash-asserted external
   classifier. The smoke statistic is feature MSE against the cached ground truth, which needs no
   instrument. ⛔ No number here is a judge number.
4. **Seeds 3 and 4.** ⭐ Seeds **0, 1 and 2 are BUILT** (`BASELINES-READY.json` →
   `seeds_present = [0, 1, 2]`, 150 records). Seeds 3–4 are a DECLARED NOT-RUN: their
   generation-2 caches are `spoke/cache-gen2-seeds-3-4`'s deliverable and did not exist when this
   spoke ran. When they do, `scripts/w12_deletion_baselines.py --seed 3` merges into the same
   ready file — ⛔ nothing here needs rebuilding for a 5-seed row.
5. **`table`, `lace`, `poe_ebm`.** Out of scope: the task names the LEARNED arms. `table` already
   has a retrain-without comparator in `chlu/eval/rivals/w4_deletion.py`; `lace` and `poe_ebm` are
   sampler draws and are not callables at all (`w4_api` refuses them by construction).
6. **Any tuning of any comparator.** Deliberate, not omitted: epochs / lr / weight decay are
   INHERITED from the shipped `w4_learners` grids. ⛔ A comparator whose strength we selected is a
   comparator whose strength we chose.

---

## 7. ⭐ What was checked rather than claimed

- **`chlu` resolves inside this worktree** — `assert_chlu_in_worktree()`, called before any
  number, recorded in every `BASELINES-READY*.json` under `worktree_assertion`.
- **No read of `$CHLU_CLAUDE_ROOT`.** `CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2`
  redirects the harness's feature source (`exp_gen_factor_wells.features_root` sanctions exactly
  this), and `CHLU_GEN_DATA_ROOT=C:/chlu/data/gen` is outside it too. Cache asserted
  **generation 2, scope=full**, `D = 2048`, sha256 per seed `ef6359a4cd54b082…` (s0) /
  `86b44d0f1e6470e7…` (s1) / `3ab14e583c3f500d…` (s2) — ⭐ three DIFFERENT feature spaces, so the
  three seeds are a real spread and not three RNG draws on one instrument.
- **Matched parameters, MEASURED by building the model** (`w4_learners.match_width`), against the
  524,288 B pin = 131,072 params: `mlp` **130,514** (−0.43%) · `cont_mlp` **131,891** (+0.63%) ·
  `transformer` **130,433** (−0.49%) · `lstm` **130,721** (−0.27%) · refuse-token transformer
  (`out_dim = D+1`) **130,479** (−0.45%). All inside the shipped ±10% tolerance, and identical on
  all three seeds (the width search sees only the architecture and the target).
- **Matched compute, asserted equal rather than asserted in prose.** `assert_compute_matched`
  raises if any two arms' unlearners differ: **296,960** (value) / **312,320** (conjunction) /
  **322,560** (item) gradient row-visits, identical across all 8 arm×method combinations on each
  leg **and across all three seeds** (`BASELINES-READY.json` → `per_seed.*.matched_compute`).
- ⭐⭐ **`BASELINES-READY.json` VERIFIED by something that did not write it.**
  `python scripts/w12_verify_baselines_ready.py --load-every` imports `w4_api` only — never the
  builder — and passes: 5/5 entry points resolve to real callables · **150/150 records carry a
  complete cost line** · every arm inside ±10% of the 131,072-param pin (worst **+0.625%**) ·
  matched compute equal across 8 arms on all 3 legs on all 3 seeds · every measured NOT-RUN
  carries a reason · ⭐ **every one of the 150 records loads BY NAME through `w4_api.load_rival`
  and returns a finite (1, 2048) feature vector** (126 distinct checkpoints, 42 per seed: the
  4 `as_trained` arms are shared across the three deletion legs and so are loaded once per leg) — (1, 2049) for the refuse-token arms, which is the feature
  block plus the refuse token. Output: `BASELINES-READY VERIFIED`.
- **15/15 property checks pass** — `python test_w12_deletion_baselines.py`. Hermetic (its own
  4-factor grid, its own random targets; no feature cache, decoder or judge). Includes: a no-op
  deletion is refused; an unmeetable matched-compute budget raises; an incomplete cost line is
  refused; the impossible certifier decides deleted / surviving / out-of-range / off-grid exactly;
  AUROC matches brute force to 1e-12; **a saved comparator loads BY NAME through `w4_api.load_rival`
  with no w12 import**.

---

## 8. ⛔ THREE DEFECTS IN SHIPPED CODE, FOUND AND FIXED — all three blocked this job outright

1. ⛔⛔ **`chlu/eval/rivals/w4_learners.py::train` cast every input to float32.** `tokens` inputs
   are vocabulary indices; under jax 0.9.0 `eqx.nn.Embedding` then raises
   `TypeError: Indexer must have integer or boolean type, got indexer with type float32`.
   **`transformer` and `lstm` could not be trained through that function at all.** Fixed with
   `_as_model_input` — dtype preserved for integers, the float path byte-identical.
   ⚠ **Anyone re-fitting a token-family rival on this box before this fix was blocked, not wrong** —
   it raises rather than silently mis-training.
2. ⛔ **Same file, the LR schedule.** `decay_steps=total`; at `total == 1` optax computes
   `decay_steps - warmup_steps == 0` and raises. Guard added, a no-op for every `total >= 2`.
3. ⛔ **`chlu/eval/rivals/w4_api.py::load_rival` refused any name not on the hand-maintained
   `LEARNER_FAMILIES` list** — the exact failure its own source comment warns about
   ("loadability is decided by what the checkpoint SAYS IT IS"). Extended so a checkpoint
   declaring `base_family` loads; `save_learner` gained `extra=` for self-describing manifests.

---

## 9. ⚠ A PIN HAD TO MOVE, AND IT IS DECLARED RATHER THAN SLIPPED IN

The wave-12 plan's illustration is **"red ∧ cube"**. ⛔ On the realised value-holdout fit pool
that deletes **ZERO rows**: shape index 0 (cube) is the condition's **own held-out shape value**
and is absent from the pool entirely (measured on the 131,072-row pinned pool: shape marginal
`[0, 43616, 43527, 43929]`). The conjunction pin is therefore
**`object_hue = 0.0 ∧ shape = 2.0`** (5,511 rows in the full pool; 667 in the 16,384-row build
pass), declared in `PINS_W12["conjunction_deletion"]` and overridable by the hub.
⭐ `partition_fit_rows` **refuses** any no-op deletion with the per-term counts in the message, so
a zero-row deletion can never be printed as a deletion result.

---

## 10. What the crossover hub does with this

```python
from chlu.eval.rivals.w12_deletion_baselines import load_comparator
r = load_comparator(seed, "transformer__retrain_without__value-object_hue-0",
                    root="C:/chlu/outputs/w12-deletion-baselines/comparators")
feats, out_of_range = r(values)          # the rows-3/4 arm_features contract, unchanged

# or with no w12 import at all:
from chlu.eval.rivals.w4_api import load_rival
r = load_rival(seed, "<name>", root="C:/chlu/outputs/w12-deletion-baselines/comparators")

# refusal DISPLAY rows only:
r = load_comparator(seed, "transformer__refuse_token__conj-object_hue0-shape2", root=...)
r.refuse_score(values)      # the refuse token in [0, 1]
r.refuses(values)           # at the declared threshold 0.5
```

Names are `<arm>__<kind>__<spec-label>` (`<arm>__as_trained` has no spec). Kinds:
`as_trained`, `retrain_without`, `unlearn_neggrad`, `unlearn_finetune`, `refuse_token`.
Every checkpoint's manifest carries its `cost_line`, `deletion_spec`, `compute`, `param_target`
and `param_error`, so a table can be built from the checkpoints alone.

**To rebuild or extend:**
```
CHLU_GEN_FEATURES_ROOT=C:/chlu/outputs/spoke-a-rivals/cache_generation2 \
python scripts/w12_deletion_baselines.py --seed S --n-fit N --epochs E --out DIR
```
`--n-fit 16384` is the declared BUILD/SMOKE scale; nothing in the comparators changes with it.
