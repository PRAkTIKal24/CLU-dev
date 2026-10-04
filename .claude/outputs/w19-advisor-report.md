# WAVE 19 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-23. One hub spoke, machine `forgis-14` (local CPU, 4 shared cores), base `w16/maze-wm = 9f80f59` — the tip w16 actually scored on, ⛔ not `base/iter8`.**
**3 of 9 scored runs spent. 6 DECLARED NOT-RUN. Zero GPU, zero cloud.**
Record: `/c/chlu/outputs/w19-maze-t-symmetric/` (956 files, 173 MB); primary artefacts mirrored to `.claude/outputs/w19-maze-t-symmetric/` (78 files). Code: `w19/maze-t-symmetric = 0455961`, bundle verified "complete history", fetched to the Mac, **4 files / 1,144 insertions / 0 deletions** against its base.
⭐ **The Hub re-derived the load-bearing numbers from the run JSONs** (bars, per-seed n-way, all six deletion cells) rather than quoting the spoke's prose. They agree to every printed digit.
⛔ **CMAZE-T only. No wave-16 row appears in any table here.**

> ⟲ **CORRECTION, 2026-09-24 (source: `w18/registry-fold` pass 4, registry `N371`).** An earlier version of §2 and §7 of this report said the charter-C controls collapse on *every* condition. They do not: **WORDS-MAIN fails the control bar** (`B2_store_gate = false`). The error came from w19's own §0 prose, which contradicts w19's own §2 table — ⛔ the table and the JSON are right. Both places are fixed below. ⭐ No other number in this report changed.

---

## 0a. ⭐⭐ RELABELLED UNDER THE HEAD'S RULE (§5b, 2026-09-24) — ⛔ THIS SUPERSEDES §0's VERDICTS

§0 below reports the **w19 PREREG margin bars**, which were the rule when w19 ran. ⛔ **The Head's label rule (§5b) replaces them for the paper**: the label is the measured relation to the best matched rival over both budgets — **WIN** (mean ahead + majority of seeds, store gate holds) · **COMPARABLE** (within one seed's spread) · **LOSS** (behind). No numeric margin bar.
⭐ Recomputed by the Hub from the per-seed arrays in the BARS JSONs (σ = sd, ddof 1, of the **paired per-seed difference**):

| condition | CLU | R\* | mean Δ | σ | CLU ahead | **label (Head's rule)** | was (PREREG bar) |
|---|---|---|---|---|---|---|---|
| WORDS-MAIN (VAL) | 0.3344 | 0.7473 | −0.4130 | 0.1537 | 0/3 | **LOSS** | LOSS |
| WORDS-FULL (VAL) | 0.7247 | 0.7420 | −0.0173 | 0.0157 | 0/3 | **LOSS** | LOSS |
| **OBS (VAL)** | 0.3513 | 0.4566 | −0.1053 | 0.1068 | 0/3 | ⚠ **COMPARABLE** | ~~LOSS~~ |
| *OBS-X10 (diagnostic)* | 0.4471 | 0.4608 | −0.0137 | 0.1644 | 2/3 | *COMPARABLE* | *would-be LOSS* |
| **BOTH (TEST, scored)** | 0.7246 | 0.6258 | **+0.0988** | 0.0751 | **3/3** | ⭐⭐ **WIN** | ~~LOSS (SB1)~~ |
| WORDS-FULL (TEST, w21 runs 4–6) | 0.7246 | 0.7421 | −0.0175 | 0.0140 | 0/3 | **LOSS** | — |

⛔⛔ **Two rows change.** **OBS becomes COMPARABLE** — its spread (0.1068) exceeds its mean gap (0.1053), driven by seed 0 where the CLU reads 0.2407 against 0.4641. **BOTH TEST becomes a WIN** — the CLU is ahead of R\* on all three seeds, which is what the Head's rule asks, even though w19's own SB1 bar failed at +0.0988 < 0.10.
⚠⚠ **The caveat that must travel with the WIN:** R\* is the arm with the best **mean** (lstm, 80k). ⛔ Against the **per-seed best** arm, seed 1 flips — the 8k lstm reads **0.5674** against the CLU's **0.5660**. That is exactly why SB1's every-seed clause failed. ⭐ Print both readings; do not let "ahead 3/3" stand unqualified.
⭐ Source: `val/BARS-*-val.json` and `test/BARS-BOTH-test.json`, `rows.nway.per_seed`. Independently recomputed by `w21/figures-redraw`, which flagged the OBS relabel first.

## 0. The answer

**The w16 CMAZE-T win does not survive symmetric inputs. No condition wins by the pre-registered rule.**

w16 gave its rivals a 636-dim conv-feature observation while the CLU read five words through a 318-wide `[bias | main | pair]` design. Wave 19 gave every arm the identical input, four ways:

| condition (identical input to every arm) | n_seeds | CLU | R\* (best matched rival) | margin | VAL | scored | TEST |
|---|---|---|---|---|---|---|---|
| **WORDS-MAIN** — 27 main one-hots | 3 | 0.3344 | 0.7473 (transformer, 80k) | **−0.4130** | LOSS (VB1, VB2) | 0 | — |
| **WORDS-FULL** — the 318-wide design to everyone | 3 | 0.7247 | 0.7420 (transformer, 8k) | **−0.0173** | LOSS (VB1) | 0 | — |
| **OBS** — conv-AE ⊕ dynamics ⊕ action | 3 | 0.3513 | 0.4566 (transformer, 80k) | **−0.1053** | LOSS (VB1, VB4) | 0 | — |
| **BOTH** — words + observation to everyone | 3 | 0.7247 | 0.6168 (lstm, 80k) | +0.1079 | PASS (knife-edge) | **3** | **LOSS (SB1)** |
| *OBS-X10 (diagnostic, AMENDMENT 1; never a bar)* | 3 | 0.4471 | 0.4608 | −0.0137 | *would read LOSS* | 0 | — |

Source: `val/BARS-<cond>-val.json`, `test/BARS-BOTH-test.json`.

⭐⭐ **The mechanism, and it is the finding: the pair block carries the w16 margin, not the store.** Hand a matched transformer the CLU's own 318-wide design and it reads **0.742 against the CLU's 0.725** (WORDS-FULL). Hand rivals only the main effects and they still compose held-out 4-tuples (lstm 80k, 0.923 on seed 0) where a ridge head over the same main effects cannot. So what the CLU buys over a ridge head is the pair indicators — and a network given those same indicators does as well or better.

---

## 1. The scored leg: BOTH, TEST, runs 1–3/9

Ledger rows written **before** launch (`2026-09-23T18:43:46Z`, three rows). Unseen n-way, `exact`, n = 761 / 742 / 686.

| arm | mean | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| **CLU (A1)** | **0.7246** | 0.800 | 0.5660 | 0.808 |
| rival lstm, 80k = **R\*** | 0.6258 | 0.627 | 0.466 | 0.784 |
| rival lstm, 8k | 0.429 | 0.361 | **0.5674** | 0.359 |
| rival transformer, 80k | 0.594 | 0.562 | 0.557 | 0.662 |
| rival mlp, 80k | 0.568 | 0.612 | 0.457 | 0.636 |
| *diag: transformer at 10× read floats (never a bar)* | 0.710 | **0.830** | 0.567 | 0.733 |

**SB1 FAILS, two ways.** The mean margin is **+0.0988** against a bar of +0.10 — it misses by 0.0012. And the every-seed clause fails: on seed 1 the best matched rival is the **8k** lstm at **0.56739**, ahead of the CLU's **0.56604**. `test/BARS-BOTH-test.json`: `SB1=false, SB2=true, SB3=true, SB4=true, PASS=false`.

⚠ **Two reasons for caution the spoke printed rather than hid:**
1. The BOTH rivals are matched to **4,584 read-path floats on a 954-dim input**, which leaves them a **4-unit hidden layer**. The matched-float rule, not the store, may be what produced the +0.108 VAL margin.
2. The 10×-read-floats transformer **passes the CLU on seed 0** (0.830 vs 0.800).

**Other TEST rows, same gate:** `seen` CLU 0.968 vs rival 1.000 (sanity, claims nothing) · `pair` CLU 0.410 but learned blind 0.557 **beats the CLU — the falsifier fires** · `value` CLU 0.465 vs 0.585. Per-step exactness: CLU 0.925 vs best rival 0.953.
⚠ **Rollout validity is 0.000 for the CLU** (rivals ≤ 0.015 on TEST). Per-step error compounds over ~40-step routes. It is printed once, as a limitation.

## 2. ⭐ What DOES survive — and it is not a rival margin

- **The store gate — on THREE of the four conditions, not all four.** ⚠ `landscape_removed` does read **0.000 on every condition**, but the full control bar **FAILS on WORDS-MAIN**: its worst control (`landscape_shuffled`) reads **0.18744** against a bar of **0.18437**, so `B2_store_gate = false` there (`val/BARS-WORDS-MAIN-val.json`). On **WORDS-FULL, OBS and BOTH** the controls do collapse — on the scored TEST leg they read 0.000 / 0.225 / 0.244 / 0.166 against the CLU's 0.725, and SB2 passes with the worst control at 0.244 against a 0.575 bar. ⭐ `landscape_shuffled_across_wells` was asserted a **DERANGEMENT** — `[1, 0]` on every seed of every run, the runner refusing a fixed point (this is `N350`, fixed and now printed).
- **Exact store-scoped deletion, re-run on all three scored seeds.** Max relative ΔS_k across all six cells (3 seeds × 2 modes) is **7.727e-16 … 1.118e-15** against a 1e-5 bar; deployed-atom max |Δ| is **0.0**; all six pass. The downdated read equals the fitted-without read on **100 %** of surviving-colour n-way rows.
- **The cost asymmetry is real and measured.** CLU adaptation is **0.09–0.39 s and data-free** (A, Mx, S_k, t_k, n_k downdated, same closed form re-solved — ⛔ not "no changes"). The rival deletes only by retrain-without: **100–140 s and 8,000 gradient steps**.
- Byte ledger, convention named: deployed = 4 × (atom + address-book floats), float32 — A1's 4,584 read floats = **18,336 B deployed (f32)**, 34,320 B statistic (f64) printed beside it, never netted.
- ⛔ **The w15 caveat travels unchanged:** this is store-scoped exactness, **not capability erasure**. Nothing here measures whether deleted content can still be composed.

**So: the store is load-bearing for the CLU's own read. On this task it is not better than a matched network given the same input.**

## 3. OBS — the failure has a named link, and it is READ

The declared word extractor (636 → 32 → 27, 6,000 steps, TRAIN rows only, 21,275 floats added to the matched target) cannot recover the request from the observation on held-out combinations:

| row | whole-tuple accuracy (seeds 0/1/2) | the word it fails on |
|---|---|---|
| TRAIN | 0.404 / 0.683 / 0.767 | — |
| `seen` | 0.359 / 0.600 / 0.628 | tint 0.778, door 0.729 |
| `nway` | **0.065 / 0.224 / 0.191** | **tint 0.249** |
| `pair` | 0.000 / 0.000 / 0.020 | tint 0.158, door 0.470 |
| `value` | 0.000 / 0.000 / 0.000 | cannot emit the never-trained colour |

⭐ **This is why AMENDMENT 1 exists** and it did not rescue the condition: a 10× extractor (60,000 steps) lifts TRAIN to 0.483 / 0.847 / 0.922 and n-way only to 0.148 / 0.296 / 0.328, leaving the CLU at 0.447 vs rivals 0.461 — *it would still read LOSS*. ⛔ It was never a bar.
⭐ **The extractor is no private channel either way:** rivals handed the same extractor output as an extra row do no better than on the raw observation (0.452 vs 0.457 at 80k).
**The link that breaks is READ — the request fails to compose before the store is ever reached.**

## 4. Integrity checks the spoke ran on itself

- ⭐⭐ **Instrument-unchanged, and this is what makes the comparison legitimate:** the CLU's own numbers are byte-identical to w16 A1. WORDS-FULL and BOTH CLU on VAL n-way = 0.8019 / 0.5689 / 0.8034; BOTH CLU + four controls on TEST = 0.800 / 0.566 / 0.808 with controls 0.000 / 0.225 / 0.244 / 0.166. Both triples match `w16-maze-wm/t1/A1_*` to every printed digit. **Only the rivals' input changed.**
- `chlu.__file__` asserted inside the worktree before any number (the import trap).
- Rival float-matching residuals −2 % … +6 % (TABLES §4). R\* is the best matched rival over **both** budgets (8k and 80k) and every kind/variant — ⭐ the spoke did not get to pick the weak one.
- 384 persisted objects, each with a param+file sha256 and a rebuild MANIFEST; all 384 re-verified, **0 mismatches**; rebuild round-trip verified for rival and CLU.
- ⚠ Measured and corrected in passing: the w9 observation's dynamics block is **31** wide at k=4, not the 29 in the brief and the w9 docstring. The observation is used unchanged (636 = 600 + 31 + 5).
- ⚠ Known cosmetic defect: the `persist_dir` field inside each result JSON names the **pre-move** path; the objects were moved to `persist/` after the runs and LOG.md records the move.

## 5. Budget and what did NOT run — declared

| runs | what | status |
|---|---|---|
| 1–3/9 | BOTH, TEST, seeds 0/1/2 | **SPENT — LOSS (SB1)** |
| 4–9/9 | WORDS-MAIN, WORDS-FULL, OBS | ⛔ **DECLARED NOT-RUN** — each failed VAL (VB1). The spending rule forbids seeding a VAL-failing condition, and BOTH already had its 3 seeds (all or none). **No re-tuning was done to rescue any condition.** |

Also not run, and declared as such rather than reported as zeros:
- **Config C-512 under any condition.** The brief changed only the input; A1 is the headline config and C-512 would share the WORDS-FULL mechanism. ⛔ Declared not-run, **not** a loss.
- The state encoder in WORDS-MAIN / WORDS-FULL (no arm there consumes the observation); `cont_mlp` in OBS / BOTH (input already continuous).
- Deletion rows on VAL legs and on non-scored conditions (`DELETION: DECLARED_NOT_RUN` in those JSONs).
- GPU, cloud, any seed beyond 3, and any edit to the w15/w16 modules — imported only.
- Smoke runs (60 steps, 150 rows) are plumbing checks and are not reported as results.

⚠ **A data ceiling, not a store property:** seed 1 caps *every* strong arm at 0.5689 on VAL — CLU, rivals and the 10×-float diagnostics alike. It is that seed's hold-out, and it is why the every-seed clause is decided on a 0.0013 gap.

## 6. What would have proved the w16 win input-independent — and what happened

| the test | what would have proved it | what happened |
|---|---|---|
| WORDS-FULL | matched rivals given the CLU's exact design stay ≥ 0.10 below it | ⛔ they did not — transformer **0.742** vs CLU 0.725 |
| OBS | the CLU keeps its margin once its request must be read from the observation | ⛔ it did not — **0.351** vs 0.457 |
| BOTH | margin ≥ 0.10 on TEST, ahead on every seed | ⛔ it did not — **+0.0988**, and behind on seed 1 |

## 7. What the Advisor is asked to rule on

1. ⭐⭐ **Does the CMAZE-T composition claim come out of the paper entirely?** The w16 margin is now attributable to an input asymmetry, and under symmetric inputs no condition clears its bar. The Hub's reading is that no composition-margin claim on CMAZE-T survives. ⛔ Not the Hub's call to decide what replaces it.
2. ⭐⭐ **Is the matched-float rule itself now the problem?** Matching on read-path floats gives a 954-dim-input rival a 4-unit hidden layer. That makes the surviving +0.108 VAL margin hard to attribute to the store, and it cuts the other way too — it may be *understating* rivals throughout the programme. This is a rule question, not a w19 question.
3. ⭐ **How should the store gate be reported now?** Deletion is exact on all three seeds, and the controls collapse on **three of the four conditions** — ⚠ **not on WORDS-MAIN**, where the bar fails by 0.003. Both are store-over-control facts with **no rival margin behind them**. Are they a contribution, or supplementary material under the genuine-win bar?
4. ⚠ **The `pair` row falsifier fired** — learned blind (0.557) beats the CLU (0.410). It is printed and claims nothing, but the Advisor should decide whether it needs saying out loud in the paper.
5. ⚠ **Rollout validity is 0.000.** Currently a one-line limitation. Is that sufficient, given a reviewer will ask whether the thing can actually act?

---

**Downstream, already in motion (standing authorisation, no approval sought):** `registry-fold` pass 4 ran and is harvested; `seeds-and-tables` pass 3/4 is running on batmac applying the deletion-rule amendment. The **10 w19 `\PLACEHOLDER{}`** in the draft are filled by a `placeholder-fill` pass once w20 also lands — ⭐ every one of them was written to stand if the row landed as a negative, and **they now do.**
