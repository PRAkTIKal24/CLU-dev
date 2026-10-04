# WAVE 12 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-21. Seven agents + a parallel revision pass, FORGIS_14,
`base/iter8 = aa91785`.** All finished. **Scored spend: ONE run** (`single-design 1/3`).
Record: `.claude/outputs/w12-*` and `w12r-*`.
⚠ Every number is quoted from an agent's own artefact. ⛔ **The Hub re-derived nothing** — and where
the Hub *did* paraphrase, it was **wrong three times**; see §6.

---

## 0. The scoreboard

| agent | verdict | scored |
|---|---|---|
| ⭐ `hub/single-design-crossover` | **K\* = 32; the saturation trap does NOT fire at rank/width 0.9844** | **1/3** |
| ⛔ `hub/composition-ladder` | **the store loses to a same-bytes MLP at 4 of 5 rungs** | 0 (replications) |
| ⭐ `spoke/kway-ladder-harness` | ladder built, both ends reproduce exactly | 0 |
| ⭐ `spoke/deletion-baselines` | comparators built; ⚠ **"red ∧ cube" is unmeasurable** | 0 |
| ⭐ `spoke/cache-gen2-seeds-3-4` | seeds 3 **and** 4 built, gate passed | 0 |
| ⭐ `spoke/positioning-scout` | **position dynamics as a measured law, never as a primitive** | 0 |
| ⭐ `spoke/clue-theorist` | zero-dynamics limit stated with its scope | 0 |
| ⭐ `w12r/artefact-mustfix` | MF3/4/7/10/11 closed; **verifier byte-identical and still PASS** | 0 |
| ⭐ `w12r/registry-repair` | N89/N90 renumbered with redirects; 10 open items declared | 0 |

**Budgets after w12:** `single-design` **2/3 left** · `generation-table` run 3 **unspent** ·
`composition` 2/3 · `deletion` 2/3 · `encoder-loop` 2 · `maze` 0 · ⭐ `encoder-adapt` **0/3 (new)**.

---

## 1. ⛔ THE COMPOSITION CLAIM DOES NOT SURVIVE AN HONEST BYTE MATCH

**The n-way ladder, fraction of per-corner ceiling, 3 seeds, all five rungs complete (195/195 rows):**

| arm | n=2 | n=3 | n=4 | n=5 | n=6 |
|---|---|---|---|---|---|
| **`learned_1x`** (MLP at the store's own bytes) | **0.926** | 0.771 | **0.944** | **0.953** | **0.953** |
| `single_design_K5` | 0.848 | **0.830** | 0.871 | 0.899 | 0.897 |
| `pca_tensor` | 0.848 | 0.830 | 0.871 | 0.899 | 0.897 |
| `rival_lstm` (524,288 B) | 0.579 | 0.604 | 0.604 | 0.604 | 0.604 |
| `rival_mlp` (524,288 B) | 0.577 | 0.586 | 0.586 | 0.586 | 0.586 |
| *`landscape_shuffled`* | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| *`random_atoms`* | 0.010 | 0.006 | 0.009 | 0.015 | 0.004 |

**Margin (store − `learned_1x`): −0.077 · +0.059 · −0.073 · −0.054 · −0.056.**

1. ⛔ **The store loses at 4 of 5 rungs**, in a tight band. It wins only at **n=3**, ⚠ and there the
   **LEARNER dipped** to 0.771 from a tight 0.94–0.95 elsewhere. ⭐ Report n=3 as **unexplained, not
   as a win**.
2. ⭐⭐ **The controls collapse at EVERY rung** — `landscape_shuffled` exactly 0.000 throughout.
   ⛔ **This is NOT wave 9's decorative failure.** The store genuinely carries its score; it is simply
   **beaten by a learner of the same true size**.
3. ⭐⭐ **`single_design_K5` == `pca_tensor` to three decimals at every rung.** ⭐ **The four-part
   ledger buys DELETION, not composition.** This is the cleanest statement of what the bytes are for.
4. ⭐ The store **does not deteriorate with depth — it improves** (0.848 → 0.899). ⛔ **The wave's own
   question presumed deterioration; the premise is false for this arm.**

### ⛔⛔ AND THE LADDER IS NOT A COMPOSITION-DEPTH AXIS FOR n ≥ 3
`spoke/seeds-and-tables` found that **rungs n = 3, 4, 5, 6 probe the IDENTICAL 120 corners** — what
varies is **how much training data is held out**. ⭐ **Only n=2 vs n=6 is a genuine depth contrast.**
⚠ The harness spoke declared this confound when it built the instrument (*"a lower-k box pins fewer
factors, so it removes more rows from the fit pool"*) and the Hub failed to carry it into the reading.
⛔ **The smooth 0.848 → 0.899 rise may be a held-out-fraction trend, not a depth trend.**

### The byte budgets, measured
- **524,288 B reference = 131,072 float32 params.** Rivals matched to within **±0.63 %**.
- **1× = the store's own measured total.** On the scored rows the single design is
  **13,041,864 B = 23.89× – 24.88×** the pin; ⭐ **`pca_tensor` is 712,072 B = 1.36×.**
- ⇒ ⭐⭐ **`pca_tensor` at 1.37× scores IDENTICALLY to the single design at 24.4×.** An 18-fold byte
  difference buys **zero** composition.

---

## 2. ⭐ `hub/single-design-crossover` — K\* = 32, and the trap does not fire

**K\* = 32** by a selection rule **frozen and timestamped at 11:10:24Z before any val composition
number existed**: the finest partition whose val composition is within 0.02 of K=5.

⭐⭐ **THE SATURATION CHECK PASSES AT rank/width 0.9844 — ABOVE wave 9's 0.938.**
`landscape_shuffled`, `landscape_shuffled_across_wells` and `random_atoms` **all collapse to 0.000**,
a drop of **+0.9016** against the arm's 0.9016, on **109/109 corners**.
⇒ ⛔ **Wave 9's decorative failure is NOT a property of high rank/width per se.**

⭐ **The sphere hazard was MEASURED and does not reproduce.** The Hub flagged that a degraded render
might be judged "sphere" and flatter a damaged baseline on the deleted conjunction. On 24 val corners
the two maximally-degraded arms bound it: a degraded render is judged **cube** (1.0000 / 0.9196)
not sphere (0.0000 / 0.0023); **sphere excess −0.3333 / −0.3310**, i.e. **negative**.

⚠ **T7, the headline scored row, is ONE SEED.** Run 1/3 was spent at 4 seeds declared; seed 4's cache
landed an hour after the hub closed. ⛔ **The main table prints `n = 1` until NR6 lands.**

---

## 3. ⚠ `spoke/deletion-baselines` — a commissioned row was unmeasurable
⛔ **"red ∧ cube" deletes ZERO rows.** Cube is shape 0 — **the condition's own held-out shape**,
absent from the pool entirely (marginal `[0, 43616, 43527, 43929]`).
⭐ The conjunction pin is **`object_hue = 0.0 ∧ shape = 2.0` — which IS "red ∧ sphere"** (Head ruling:
name it *red sphere* everywhere). ⭐ `partition_fit_rows` **refuses any no-op deletion**, so a
zero-row deletion can never be printed as a deletion result.

⚠ **AMENDED mid-wave (Head): naive vs naive.** Paper rows carry **(i)** retrain-from-scratch-without
— the naive rival's only deletion route, cost printed — **(ii)** rival as-trained, **(iii)** the
refuse-token transformer plus **one extra abstain output** for MLP / cont-MLP / LSTM.
⛔ **DROPPED from paper rows:** approximate-unlearning on the rival's head; sharded ensembles.
⚠ **The abstain output is NOT YET BUILT** — the spoke ran before the amendment. **DECLARED NOT-RUN.**

⭐⭐ **Correction the Hub owes:** the unlearners are **not "unrun".** They were **measured on the fit
pool** — 0 scored runs, no held-out corner — and the numbers are real: **NegGrad+ over-forgets
9.4×–20.8×; fine-tune under-forgets on three arms of four**, at **296,960 matched gradient row-visits
asserted equal across all eight arm × method combinations on all three seeds** ⇒ ⛔ **not a
starved-comparator artefact.** What is unrun is **scoring** them. ⭐ **HELD for post-submission by
Head ruling — availability is not authorisation.**

## 4. ⭐ The two scouts
- **`positioning-scout`:** ⭐ position the latent-dynamics material as **a measured law about our own
  object**, under rung 3 of the claim ladder. ⛔ **Never as a primitive, never as a scoreboard row.**
  Grounded in our own record: the charter already ruled it out of the scoreboard *before* any number
  was read, and ⭐ **the same dynamics measurably COST composition** on the same object and seeds.
  9 declared not-runs; Semantic Scholar 429 reported rather than worked around.
- **`clue-theorist`:** the zero-dynamics limit is **measured, not conjectured** — a zero-step arm was
  asserted **bit-identical** to the static read on 256 requests and passed. ⚠ It re-ran all four
  checks from a **clean clone of its own bundle**, byte-identical.

## 5. ⭐ The revision pass (`w12r`), run in parallel at zero compute
- **`artefact-mustfix`:** MF3 (byte row), MF4 (define *exact* = **store-scoped**, encoder/decoder not
  refit), MF7, MF10 (**CMAZE-2** relabel), MF11. ⭐ It edited the **generator**, never outputs by hand,
  and left `verify_tables.py` **byte-identical** — so the check stayed independent. **PASS** before and
  after. The **failure agreement** is now a citable row.
- **`registry-repair`:** `N89`/`N90` renumbered to `N316`/`N317` with **redirects at every old number
  in both directions**, cross-file sweep run **before** the move, one residual declared not silently
  fixed. Registry 4,739 → 5,268 lines. ⚠ **`N-I8-21(b)` is on its THIRD instance** — promoting it to a
  standing rule is the Head's call; the spoke declined to do so unilaterally.

## 6. ⛔⛔ THREE ERRORS THE HUB MADE, CORRECTED BY ITS OWN SPOKES
1. ⛔ **The byte curve.** The Hub reported *"the honest total falls 7.48× → 2.07×, monotonically"*.
   ⭐ That is the **`times_reference_pin_parts_1_2_4`** column — it **EXCLUDES the address book** — read
   from a file whose **first key is `SMOKE_NOT_A_CLAIM: true`**. The honest four-part total falls
   **7.855× → 4.088×** and is **U-SHAPED, bottoming at 3.994× at K = 24**. ⛔ **K\* = 32 is NOT the
   cheapest point.** ⭐ The direction survives; the magnitude, the shape and the endpoint do not.
   ⚠ And on the **scored rows** the realised figure is **24.4×**, which is the referee's own MF3 number.
2. ⛔ **The unlearners** were reported as "built but not run". They were **measured** (§3).
3. ⛔ **Notation.** The Hub wrote **k** for both the ladder rungs and the well count. ⭐ Convention now:
   **n** = jointly composed factors · **K** = wells. Recorded so no draft inherits it.

## 7. Limits
One scored run in the whole wave · **T7/T9/T10 rest on ONE seed** · the ladder is **3 seeds** and for
n ≥ 3 is a **training-data axis** · the abstain output is **not built** · the K\* store-arm
replication is **not yet run**, so the ladder's verdict stands against a **1,029,508-param** opponent
where the main text will use **535,784** · the Hub re-derived nothing and was wrong three times where
it paraphrased.

## 8. What the Advisor is asked to rule on
1. ⭐⭐ **`pca_tensor` at 1.37× equals the single design at 24.4× on composition.** ⛔ **Should the
   composition rows quote `pca_tensor` and let the single design carry DELETION alone?**
2. ⛔ **Which byte column leads** — 1× (recommended, with NR8 stated) or the 524,288 B reference?
3. ⛔ **Do the n ≥ 3 rungs stay**, given they probe the identical 120 corners and are a training-data
   axis, or does the paper report **n=2 vs n=6** as the only genuine depth contrast?
4. ⚠ **Is "finer partitions are cheaper" still safe to state**, given the honest curve is U-shaped and
   K\* = 32 is not the minimum?
5. ⚠ **`N-I8-21(b)`** — duplicate-number allocation, third instance. **Standing rule?**
