# WAVE 13 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-22. Four agents, FORGIS_14, `base/iter8 = aa91785 → 6699665`.**
All finished. **Scored spend: ONE run** (`encoder-adapt 1/3`, its own new target).
Record: `.claude/outputs/w13-*`.
⚠ Every number is quoted from an agent's own artefact. ⛔ **The Hub re-derived nothing** — and where
the Hub paraphrased, its own spokes corrected it **twice more**; see §5.

---

## 0. The scoreboard

| agent | verdict | scored |
|---|---|---|
| ⭐ `hub/encoder-adapt` | **adaptation GROWS the margin** — `anchor` 3/3 seeds, `adapter` 2/3 | **1/3** |
| ⭐ `spoke/seeds-and-tables` | T1…T12 rebuilt; ⛔ **the ladder is a TRAINING-DATA axis for n ≥ 3** | 0 |
| ⭐ `spoke/registry-fold` | w12 folded; ⛔ **corrected the Hub's byte curve and its "unrun" claim** | 0 |
| ⭐ `spoke/draft-revise` | all 16 must-fix closed, each with the change named | 0 |
| ⛔⛔ `spoke/referee-2` | **MAJOR REVISION — no longer a reject.** ⭐ R1 **3 → 7** | 0 |

**Budgets after w13:** `encoder-adapt` **2 left** · `generation-table` 1 · `encoder-loop` 2 (one
reserved for BAR D) · `read-verb` 3 · ⛔ `single-design` and `maze` **exhausted**.

---

## 1. ⭐ `hub/encoder-adapt` — light adaptation buys a real, small margin gain

**The margin = single design − best learned baseline (`mlp_180k`), per seed, Δ vs `frozen`:**

| seed | frozen | adapter | anchor |
|---|---|---|---|
| 0 | +0.5688 | **+1.6239** (Δ **+1.0550**) | **+1.3303** (Δ +0.7615) |
| 1 | −0.0275 | **+0.3670** (Δ +0.3945) | −0.0092 (Δ +0.0183) |
| 2 | +1.1009 | +0.3028 (⛔ Δ **−0.7982**) | **+1.3028** (Δ +0.2018) |

⭐ **`anchor` grows the margin on all three seeds; `adapter` on two of three.** ⛔ The bar —
*the margin must grow from frozen to adapted* — is met by `anchor` 3/3.
⚠ **This does NOT contradict wave 12.** The comparator here is `mlp_180k` at 180 k params, ⛔ not
`learned_1x` at 0.5–3.2 M. **Different opponent ⇒ the two tables must never be merged.**

⭐ **Equal compute is PRINTED, not claimed:** **400 steps and 25,600 images identical across all nine
arm-runs**, with encoder params in the optimiser (0 / 131,072 / 428,576) and seconds beside them.
⭐ The artefact says in its own header: *"3 seeds × 3 configurations = 9 arm-runs. ⛔ That is 3 SEEDS,
never '9 seeds'."* — the w8 phrasing defect pre-empted **in the artefact**.

⛔ **The factor probe buys nothing.** Margins over random-init are +0.0001 … −0.0160 aggregate,
reported **per seed, never pooled**. ⇒ ⭐ adaptation is not buying a better representation by that
measure, which is consistent with w8.

⭐ Two PREREG amendments, both **before any number on any condition and before any ledger row**, and
both say so: *"No bar is moved by this amendment."* Amendment 1 was a **measurement, not a choice** —
§3 let a reader assume the encoder trains *through* the store; measured on a quick warm smoke, **it
cannot**, so the description was corrected rather than the experiment quietly changed.

---

## 2. ⛔⛔ `spoke/referee-2` — MAJOR REVISION, and it broke the wave-12 flip

> **MAJOR REVISION. ⛔ No longer a reject — and ⛔ not yet an accept. The revision costs 0 scored runs
> and 0 fitted models: every new must-fix is arithmetic on a field already on disk.**

| reviewer | v0 | v1 |
|---|---|---|
| **R1 — machine unlearning** | 3 (reject) | ⭐ **7 (accept)** — *"the best-defended unlearning claim I have refereed this cycle"* |
| **R2 — compositional generation** | 5 | 5 → **7 after N1–N4** |

⭐ **14 of 16 must-fix genuinely CLOSED**; ⚠ **MF2 and MF14 are PARTIAL** — and it found both by
checking the artefacts, ⛔ not by reading the draft's own disposition table.

### ⛔⛔ N1–N3: the wave-12 flip is mostly a PRICING ARTEFACT
- **N1:** the **K = 5 defeat is priced by `Σ_k d_k²`** while the **K\* win is priced by the symmetric
  triangle** — ⛔ **and the draft itself says, two sections later, that `Σ_k d_k²` is the wrong cost
  function.**
- ⭐⭐ **N2:** `w12b` had **already fitted** a same-convention **1,029,776-param** opponent at K = 5,
  on the **identical corner set** (verified equal as sets), same seed, same cache, same schedule.
  Recomputed per corner, **with zero new runs**:

| rung | margin **as printed** | margin, **same convention** |
|---|---|---|
| n=2 | −4.567 | **−1.233** |
| n=3 | +4.817 | **+10.658** |
| n=4 | −6.242 | **−0.392** |
| n=5 | −4.808 | **−0.125** |
| n=6 | −5.775 | **−0.583** |

  ⭐ The honest bracket for that opponent is **938,413–1,710,324** params; the one used,
  **3,241,076, sits 1.9×–3.5× OUTSIDE it.** The store still fails the +1.0 bar at four rungs, ⛔ but
  in the referee's words: ***"'loses by 2–6 novel tuples per corner' and 'loses by 0.1–1.2' are
  different papers."***
- ⛔ **N3:** the bridge table **drops two columns its own source provides.** They show the store
  **FALLS** across the construction step at n=3 and n=4 (−2.23, −0.63), and that
  ⭐⭐ **71.8 – 77.3 % of the swing is the OPPONENT SHRINKING, not the store improving.**
- **N4:** the control gate is a **weaker control at K\* than at K = 5**, and the cited artefact's own
  LIMITS says so.

⭐ **What it could NOT break, and says so:** the n ≥ 3 axis is labelled honestly (r = −0.96…−0.98
against held-out row counts); n=3 is reported as **unexplained, not as a win**; and
**`verify_tables.py` PASSES on its own fresh run at exactly the counts the draft claims**
(52/52/872/1,786/8).

---

## 3. ⭐ `spoke/seeds-and-tables` — and the finding that reframes the ladder
T1…T12 rebuilt, **seed count as a COLUMN of every row**, both byte columns on every row.
⛔⛔ **For n ≥ 3 the ladder rungs probe the IDENTICAL 120 corners** — what varies is **held-out
fraction**. ⇒ ⭐ **only n=2 vs n=6 is a genuine composition-depth contrast**; the rest is a
training-data axis. ⚠ The harness spoke declared this confound when it built the instrument and the
Hub failed to carry it into the reading.
⭐ On the scored rows the single design measures **13,041,864 B = 23.89× – 24.88×** the pin;
**`pca_tensor` is 712,072 B = 1.36×** — ⭐⭐ **and they score IDENTICALLY.**

## 4. ⭐ `spoke/registry-fold` — N316…N336, and two corrections aimed at the Hub
⭐ It verified the w8 items were already folded rather than re-folding them, then recorded wave 12.
⭐ It **declined to spend**: closing `N325` needs 5 val legs on `single-design` — *"not mine to
spend; I spend none and recommend none."*
⚠ **`N-I8-21(b)` (duplicate-number allocation) is on its THIRD instance.** ⛔ Promoting it to a
standing rule is the **Head's** call; the spoke refused to do it unilaterally.

## 5. ⛔⛔ TWO MORE HUB ERRORS, CORRECTED BY ITS OWN SPOKES
1. ⛔ **The byte curve.** The Hub reported *"7.48× → 2.07×, monotonic"*. That is the
   **parts-1-2-4 column** — it **excludes the address book** — read from a file whose **first key is
   `SMOKE_NOT_A_CLAIM: true`**. ⭐ The honest four-part total falls **7.855× → 4.088×** and is
   **U-SHAPED, bottoming at 3.994× at K = 24**. ⛔ **K\* = 32 is NOT the cheapest point.**
2. ⛔ **The unlearners** were reported as *"built but not run"*. They were **MEASURED on the fit
   pool** — 0 scored runs, no held-out corner — **NegGrad+ over-forgets 9.4×–20.8×; fine-tune
   under-forgets on three arms of four**, at **296,960 matched gradient row-visits asserted equal
   across all eight arm × method combinations on all three seeds** ⇒ ⛔ **not a starved-comparator
   artefact.** What is unrun is **scoring** them. ⭐ **HELD for post-submission (Head ruling).**
⚠ Counting w12's three, that is **five byte/scope figures the Hub stated wrongly and its spokes
caught.** ⭐ Every byte figure now travels with its convention named.

## 6. Limits
One scored run · `encoder-adapt`'s comparator is `mlp_180k`, ⛔ not `learned_1x` — **separate table** ·
the probe margins are ~0 · **2 must-fix remain PARTIAL** · **8 new must-fix (N1–N8) are OPEN** ·
the draft is **not yet an accept**.

## 7. What the Advisor is asked to rule on
1. ⛔⛔ **The byte convention is now load-bearing, not cosmetic.** The ladder and the crossover priced
   the 1× opponent **differently**, and the flip is **71–77 % opponent-shrinkage**. **Does the paper
   adopt ONE convention, or print the 938,413–1,710,324 bracket explicitly?**
2. ⛔ **N2's appendix row costs 0 runs and changes the story from "loses by 2–6" to "loses by
   0.1–1.2".** **Is it added?**
3. ⚠ **`anchor` beats `adapter` 3/3 vs 2/3.** **Does the secondary paragraph lead with `anchor`?**
4. ⚠ **`N-I8-21(b)`, third instance — standing rule?**
5. ⭐ **R1 is at 7 and the remaining revision costs no compute.** **Is the paper's lead now
   exact deletion, with composition demoted to a measured negative?**
