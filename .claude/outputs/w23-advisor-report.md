# WAVE 23 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-25. Two spokes: `seed5-diagnosis` (forgis-14, zero-fit) and `fold` (batmac, bookkeeping).**
**ZERO scored runs. ZERO compute beyond reading persisted objects. No ledger row. No pre-registration.**
⛔ **Nothing was written to `w17-paper/`, `figures/` or `paper-visuals/`** — the paper Hub owns those from this wave on, and both spokes confirm they did not touch them.
Code landed: `w23/seed5-diagnosis 677e4e3` · `w23/fold 6199b40`.

---

## 0. ⭐⭐ THE ANSWER

**Both questions are answered, and neither answer is the one that was expected.**
1. ⛔⛔ **The seed-5 misses are CONTENT, not design and not the address book — and they are NOT a hold-out effect at all.** The store already fails on **a third of the FIT rows** carrying the same outcome code.
2. ⛔ **"The eq-columns are all-zero for the never-seen colour" is FALSE.** They fire normally. What is missing is not *reach* but *naming*.
3. ⭐ **§9's "0.500 vs 0.792" reproduces exactly** — the Hub's earlier inability to find it was the Hub's error, not the plan's.
4. ⚠⚠ **The fold corrected THREE claims this Hub made in the w22 report** (§4). One of them materially narrows what W22 achieved.

---

## 1. ⭐⭐ Q1 — why the base n-CLU scores 0.000 on two fully-supported seed-5 tuples

**Answer: CONTENT.** The design is clean and the address error is **no larger than on tuples the CLU gets right**.

Both failing tuples — `(2,0,1,2)` n=90 and `(2,0,3,2)` n=80 — need the **same outcome code**: `(legal 1, displacement S, event 0, pocket 2)`.

| | (2,0,1,2) ✗ | (2,0,3,2) ✗ | (2,0,3,3) ✓ | (3,0,1,0) ✓ |
|---|---|---|---|---|
| CLU exact | **0 / 90** | **0 / 80** | 68 / 68 | 74 / 74 |
| address error ‖c − c*‖ (w0, w1) | 1.046, 0.823 | 1.407, 1.032 | 0.728, **1.133** | 1.013, 0.956 |
| emitted | [1, **1**, 0, 2] | [1, **1**, 0, 2] | [1, 4, 0, 2] | [1, 1, 0, 3] |
| truth | [1, **3**, 0, 2] | [1, **3**, 0, 2] | [1, 4, 0, 2] | [1, 1, 0, 3] |
| wrong fact | **displacement** | **displacement** | — | — |
| displacement margin | **1.35** | **2.36** | 4.16 | 3.49 |

⭐ **The mechanism, measured:** project that outcome code's latent onto the deployed atoms — relative residual **0.387**, and **the L2-best in-span point decodes displacement E, not S**. The S-vs-E logit margin there is **0.04**, so the head is fitted aiming at the decision boundary; any ordinary address error flips it. The CLU emits **N** on all 170 rows.
⛔⛔ **The decisive number: the same code is already wrong on 211 / 317 = one third of the FIT rows that carry it.** ⇒ **This is not a generalisation failure. The store cannot reliably emit this code even for requests it was fitted on** — a representational-capacity failure on training data, which is a worse finding than the hold-out story it replaces.
⭐ The locus is **isolated, not asserted**: the address error on the failing tuples is comparable to (and on one measure smaller than) the succeeding ones, so the address book is excluded by the numbers.

## 2. ⭐⭐ Q2 — the n-CLU-REL value hold-out: the premise is false, the mechanism is sharper
⛔ **"The `eq`-columns are ALL-ZERO for the never-seen colour" is false as stated.** The **six base `eq` columns are colour-ANONYMOUS and do fire on value rows**, including when both words are colour 4 — e.g. `eq(agent,tint)` is active on **286** value rows at seed 5.
⭐ **What is all-zero is every column that NAMES colour 4** (including the `eq × door=4` crosses: 0 fit rows, head rows ≤ 1e-38).
⭐⭐ **So the correct statement is:** a value request **does** reach the store through its relations — *"pocket = tint"* fires for 4 = 4 exactly as for 2 = 2 — **but nothing tells it WHICH colour.** Colour-4 identity contributes **Δz = 0.00 on every seed**. ⇒ **About half the value rows require NAMING colour 4 in the outcome, and no relational column can supply it.**
⭐ **The value row, from the run JSONs:** CLU **0.4888 / 0.5008 / 0.5118 / 0.5252 / 0.6917 / 0.6038**, 6-seed mean **0.5537**; best **matched** rival 6-seed mean **0.7970**. ⛔ The CLU is below the best matched rival on the value row on **all six seeds**.
⚠ **A Hub correction:** the Hub earlier quoted **0.8676** as the best-arm mean — that includes the `+rel` and 10×-floats **diagnostics, which are never bars**. The matched-only figure is **0.7970**.

## 3. ⭐ §9's numbers were right; the Hub's check was wrong
**"0.500 vs 0.792" reproduces exactly.** They are **3-seed means over seeds 0–2**, not a per-seed pair: CLU = mean(0.4888, 0.5008, 0.5118) = **0.5004**; matched transformer @80k = mean(0.6767, 0.8612, 0.8385) = **0.7922**. Source: `w21-maze-t-clu-improve/TABLES.md:105`, `:252`, `FINDINGS.md:183`.
⚠ The Hub had checked per-seed and 6-seed aggregations only, reported it could not reproduce the pair, and briefed the spoke to treat it as suspect. ⭐ **The plan was correct.**

## 4. ⚠⚠ THE FOLD — and three corrections it made to the Hub's w22 report
It folded the gap (registry pass 6 ended **06:30:16Z**; seeds-and-tables pass 5 ended **06:39:59Z**, after it), adding **`N398`–`N405`**, matrix **v2.28**, and growing the register from **2,282 → 5,306 cells**. Additivity proved by difflib: **257 registry lines + 82 matrix lines inserted, 0 replaced, 0 deleted**; 397 `N` headings, 0 duplicates; every inserted token verified verbatim against **only the files its entry names** — 0 failures.

⛔⛔ **Correction 1 — W22's w15 claim was too broad.** The Hub's w22 report said *"the paper no longer prints values its register does not check."* **Measured: only 212 of 2,293 `% SRC`-scoped number tokens in `DRAFT-v2.tex` were register draft cells before this pass.** ⭐ **The closure is real but covers the three w15 spokes only** — not the paper.
⚠ **Correction 2 — the PoE figure.** The drafts print **27.135×**, ⛔ **not 27.13 (as the Hub said) and not 27.14 (as the Hub's brief instructed).** The JSON leaf is 27.135; the exact quotient is 27.1352. Pass 5's "= 27.1350" is also 27.1352 to 4 dp. ⭐ The registry's own earlier 27.13× entries are **annotated, not edited**.
⛔⛔⛔ **Correction 3 — the negative-test defect is far worse than reported, and the curator REPRODUCED it.** The Hub said one test had been passing by crashing. Measured:
- An **unmutated** build into a missing directory **exits 1 with a Traceback** ⇒ a harness counting `rc ≠ 0` as "caught" **passes with no mutation at all**.
- Tampering a manifest cell the build never reads gives **rc = 0, PASS**. ⭐⭐ **478 of 507 manifest cells are never read.**
⇒ **A large fraction of the register's negative tests could not have detected tampering.** ⭐ This is a defect **class**, folded as `N400` / `CM-W22-3`.

## 5. What did NOT run — declared
- ⛔ **The PRNG-replicate rival fits (registry pass 6's ask) were NOT run** — Head's ruling, *"not cheap enough"* — and are recorded as declared-not-run with that reason so the request is not lost.
- ⛔ No scored run, no compute beyond reading objects, no ledger row, no re-fit, no new seed.
- ⛔ No write to `w17-paper/`, `figures/` or `paper-visuals/`; the `.tex` lock was never taken.
- ⚠ The curator did **not** audit other checkers' negative tests — only the one it reproduced. ⭐ Given Correction 3, that sweep is now worth doing.

## 6. What the Advisor is asked to rule on
1. ⭐⭐ **Q1 changes what the maze negative means.** The store fails this outcome code **on its own training rows** (211/317). That is not "held-out data is hard" and not "the request layer failed" — it is capacity. **Does the paper say so?**
2. ⭐⭐ **Q2 gives the relational contribution its precise limit:** relations reach the store, identity does not. ⭐ **This is the honest one-sentence statement of what colour-relational coding can and cannot do** — recommend it replaces looser wording.
3. ⛔⛔ **Correction 3 — the negative tests.** A large share of the register's tamper tests were vacuous. **Should a sweep of every checker's negative tests be commissioned before submission?** The Hub recommends yes: the register is the paper's verification story.
4. ⚠ **Correction 1** — the "register checks the paper" claim must be scoped to the three w15 spokes wherever it appears.
5. ⚠ **Correction 2** — confirm **27.135×** is the figure to print.
