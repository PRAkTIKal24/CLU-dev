# WAVE 21 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-25. Seven units, machines `forgis-14` (all scored work) and `batmac` (figures, registry).**
**12 scored runs spent: `maze-t-symmetric` 4–9/9 (target now EXHAUSTED at 9/9) · `maze-t-seed4` 3/3 · `maze-t-rel-newseeds` 3/3. Zero GPU, zero cloud.**
Code landed on the Mac, all additive except where a unit edits the CLU by design: `w21/words-full-test 6a3db43` · `w21/maze-t-clu-improve f491a04` · `w21/maze-t-seed4 bd2468f` · `w21/rel-newseeds 31eeadd` · `w21/ladder-rivals-32k 3cd0110` · `w21/figures-redraw cc17f15` · `w21/registry-fold f1ca948`.
⭐ **The Hub re-derived every load-bearing number in this report from the run JSONs.** ⛔ **No pre-registration governs it** — the Head banned pre-declarations on 2026-09-24; **everything measured is printed**, including what earlier practice would have withheld.

---

## 0. ⭐⭐ THE ANSWER

**The maze composition claim survives, and it is not a memory result.**

1. **A1 on six seeds is a WIN by the Head's rule** — ahead on the mean and on 4 of 6 seeds, gate holding — but the mean margin is **+0.046**, not the +0.099 that three seeds suggested.
2. ⭐ **The improved CLU is a large, robust, six-seed improvement**: **0.925 vs A1's 0.677**, fixing exactly the two seeds A1 was weakest on.
3. ⛔⛔ **But three independent measurements say the STORE is not what produces it.** Hand the rivals the same relational columns and they recover **97–100 %** of the margin; on seed 4 a **control ties the CLU exactly**; on seed 5 a **rival ties it**.
4. ⭐ **The defensible contribution is therefore an INPUT REPRESENTATION — colour-relational request coding carried through the closed-form head** — not a claim about memory.
5. ⭐ **MF-2 is answered and immaterial:** rivals refitted on the store's own rows leave the ladder margin where it was. **The operative asymmetry was never data; it was training steps** — confirming w20 independently.

---

## 1. The six-seed condition table (`maze-t-seed4`, `words-full-test`, w19)

**BOTH TEST, n-way exact, CLU A1 unchanged. Seeds column = 6.**

| | s0 | s1 | s2 | s3 | s4 | s5 | **mean** |
|---|---|---|---|---|---|---|---|
| **CLU (A1)** | 0.800 | 0.566 | 0.808 | 0.674 | 0.533 | 0.682 | **0.677** |
| best matched rival | 0.627 | 0.567 | 0.784 | 0.412 | 0.531 | **0.867** | 0.631 |
| **CLU − rival** | +0.173 | −0.001 | +0.023 | **+0.262** | +0.001 | ⛔ **−0.185** | **+0.046** |
| **per-seed CEILING** | 0.830 | 0.567 | 0.808 | 0.714 | 0.535 | 0.939 | 0.732 |

⭐ **3 clear wins · 2 ties at ceiling · 1 loss.** ⭐ Easy {0, 2, 3, 5} / hard {1, 4}, classified by **measured ceiling** (largest-gap cut 0.641), ⛔ never by the CLU's own score; the matched-arms-only sensitivity (hard {1,3,4}) is printed beside it.
⚠ **Two asymmetries that cut opposite ways and must both print:** seed 3's **+0.262 is inflated** — its matched MLP never converged (loss **0.180@8k → 0.205@80k**, worse with 10× training). Seed 5's **−0.185 is diagnosed**: the CLU scores **0.000 on two fully-supported class-A tuples** (170 rows) where **8 trained arms score 1.000** — ⛔ a CLU-specific miss on reachable data, not a hard seed.
⭐ **`words-full-test`: WORDS-FULL on TEST is a LOSS** — CLU 0.7246 vs matched transformer-8k 0.7421, Δ̄ **−0.0175**, ahead on **0/3** seeds, store gate holding. TEST reproduces VAL almost exactly (−0.0175 vs −0.0173), and **nothing was re-tuned**.
⚠ **It declared its own label's fragility:** a σ from the CLU's per-seed spread (0.137) or R\*'s (0.151) reads **COMPARABLE**; the paired-difference spread was fixed before any TEST number, so **LOSS** stands. ⭐ Print both readings.
⚠ **Rollout validity is 0.000 on all six seeds** (rivals reach 0.267). ⚠ `pair`: CLU 0.384 vs ceiling 0.550, with the learned blind arm **beating** the CLU on several seeds.

## 2. ⭐⭐ THE IMPROVED CLU — a real gain, and an honest mechanism

| | s0 | s1 | s2 | s3 | s4 | s5 | **mean** |
|---|---|---|---|---|---|---|---|
| **`CLU-REL-A10`** | 0.908 | **0.894** | 0.984 | 0.927 | 0.900 | 0.939 | ⭐ **0.925** |
| best matched rival | 0.803 | 0.567 | 0.685 | 0.723 | 0.624 | ⛔ **0.939** | 0.724 |
| best **`+rel`** rival | 0.915 | 0.869 | 0.819 | 0.921 | 0.900 | 0.939 | 0.894 |
| max control | 0.778 | 0.563 | 0.477 | 0.698 | ⛔ **0.900** | 0.827 | 0.707 |

⭐ **It generalises and it fixes A1's weak seeds** — seed 1 **0.566 → 0.894**, seed 5 **0.682 → 0.939** — and reaches the **ceiling on all six**. ⭐ These margins do **not** rest on broken rivals: no WORSE@80k on seeds 3–5, best rival unflagged.
⛔⛔ **THE THREE MEASUREMENTS THAT LOCATE THE CAUSE:**
1. **`+rel` erases it.** CLU − best `+rel` = −0.007 / +0.024 / +0.165 / **+0.006 / 0.000 / 0.000**; six-seed mean **+0.031**. On the new seeds the coding recovers **97–100 %** of the margin.
2. ⛔ **Seed 4: `control: random_atoms` = CLU = 0.9004**, matching **exactly on all 16 held-out tuples** — the VAL tie, repeated on TEST. **The store is not load-bearing there.**
3. ⛔ **Seed 5: CLU 0.939 == best matched rival 0.939.** No margin.
⭐ **Mechanism, measured:** for a colour *c*, `(pocket=c, tint=c)` never occurs in train, so **every colour-IDENTITY design has nothing to read** — which is why every matched rival sits at exactly **0.5674** on seed 1. The new CLU adds one column, `eq(x,y)`, learnable from other colours; those rows go **0 → 1.000**. ⛔ The `(c,c,c)` family stays **0 on every seed**, for every arm.
⭐ **Search discipline: 68 candidates tried, 57 cleared the VAL gate, all logged with numbers, kept or not.** ⚠ Its own ledger rows say 69 — the spoke corrected itself to 68 and ⛔ did not edit the hashed rows.
⚠ **The store gate weakens with atom count:** A1 max control 0.250 → `CLU-REL-A10` **0.615** (bar 0.787). It passes, but the margin to the bar shrank threefold.

## 3. The ladder — MF-2 answered (`ladder-rivals-32k`, free replication)

K\*=32 minus the strongest rival, mean over rungs n=2…6:

| set | pool · steps | margin |
|---|---|---|
| **A** (the draft's) | 32,256 rows · 5,040 | **+17.53** |
| **B** (w20) | 131,072 rows · 20,480/204,800 | **+9.97** |
| **C** (new) | **32,768 rows — the store's own content rows** · 5,120/51,200 | **+10.28** |
| **C step-matched** | 32,768 rows, matched to B | **+10.75** |

⭐ **Referee-4 was right that set B gave rivals 4× the store's rows — and it does not matter.** Set C lands within **±1.5** of B at every rung, **in both directions**. Every CI95 lower bound in B/C/C-sm is **above 0** (lowest **+1.98**).
⭐ **Pool identity proven on 15/15 legs** (`n_fit_rows = n_store_train = 32768`; rung pool identical to the w12 partitions), so set C read **exactly the store's content rows**.
⭐⭐ **Independent confirmation of w20: at matched steps the pool barely moves these rivals; the step budget does.**

## 4. ⚠⚠ A RESULT THAT BEARS ON EVERY SINGLE-FIT MARGIN IN THE PAPER
The ladder's **X1** check held rows, steps, width, parameter count, trainer and schedule **identical**, leaving **only the PRNG draw** (init and shuffle keys). Per-seed deltas reach **±4.1 novel tuples/corner, with CIs excluding 0 on 10 of 15 legs.**
⇒ **A single 2,560-step rival fit has random-seed variance up to ±4.1 — against a headline margin of ≈ +7.** ⭐ The pattern is systematic, not symmetric (higher on seed 0 at 5/5 rungs, lower on seed 1 at 5/5). ⛔ **Fit-level variance must print beside any single-fit margin**, or a referee will ask what one fit's margin means.

## 5. Integrity checks
- ⭐⭐ **Instrument unchanged:** `words-full-test` reproduced w19's WORDS-FULL VAL **bit-identically** — max |Δ| = 0 over CLU + 4 controls × 4 rows × 3 seeds — and its TEST CLU + controls equal w19's BOTH TEST **bit-for-bit**. **Only the rivals' input changed.**
- ⭐ **Zero-fit replay, across units:** every persisted object's param hash matched and every number reproduced bit-exactly for w19 seeds 0–2, `words-full-test` seeds 0–2, w19 OBS VAL, `seed4` seeds 3–5, and the peer's REL seeds 0–2. `figures-redraw` re-read 132/132 objects and reproduced the scored artefact exactly on a **different machine**.
- ⭐ `landscape_shuffled_across_wells` asserted a **DERANGEMENT** `[1,0]` on every seed of every run.
- ⭐ Ledger rows written **before** every scored run, append-only, stating seed and condition with **no expected outcome** (the post-ban format).

## 6. What did NOT run — DECLARED (⭐ per the Head's omit-nothing order)
- **C-512** under any condition — ⛔ declared not-run, **not** a loss; it is exposed to the same input asymmetry as A1 and ⛔ **may not be promoted in its place**.
- With the improved CLU: **WORDS-FULL TEST, OBS, WORDS-MAIN**, and four design variants (dyn-interactions, two-stage fit, item-store, read-verb).
- **LACE/PoE @131k on ladder rungs** (15 legs) — cost measured, not estimated: a 36-trial selection of 5,955–11,034 s plus a 50-epoch classifier at ~7.5 h on the val leg.
- ⚠ **Previously withheld, now printed:** the **10×-read-floats** and **10×-steps** diagnostic columns (*"never a bar"*); the **planner columns** excluded from every ceiling (**ORACLE = 1.000 by construction**); **smoke runs** (plumbing); and `words-full-test`'s **σ-sensitivity**.

## 7. What the Advisor is asked to rule on
1. ⭐⭐ **The maze contribution is an input-representation result, not a memory result.** Three measurements say so. Does the paper claim it as such — and what, if anything, remains claimable for the store on this task?
2. ⭐⭐ **Fit-level variance (±4.1) against the headline margin (≈+7).** Does every single-fit margin print its RNG spread? This is a rule question, not a w21 one.
3. ⭐ **Seed 5's CLU-specific miss** — 0.000 on two fully-supported tuples where 8 arms score 1.000. Printed as a limitation, or investigated before submission?
4. ⚠ **Seed 3's +0.262 rests on a rival that never converged** (worse at 80k than 8k). The matched-float rule gives a 954-dim-input rival **4 hidden units**. Is that rule still defensible, and does the row stand?
5. ⚠ **C-512 is unverified under symmetric inputs.** Confirm it is not promoted in A1's place.
