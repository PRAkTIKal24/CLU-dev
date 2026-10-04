# WAVE 16 — REPORT FOR THE MAIN ADVISOR
**Compiled by the ICLR-long Hub, 2026-09-23. One autonomous hub (`hub/maze-wm`), FORGIS_14, local CPU only, 4 cores.**
Base `base/iter8 = 148f3b7` + the two w15 harness branches merged (`b1ca534`). Record: `.claude/outputs/w16-maze-wm/` (on the box: `/c/chlu/outputs/w16-maze-wm/`).
**Scored spend: 9 of 18. The other 9 are DECLARED NOT-RUN with reasons (§4).** Zero GPU, zero cloud.
⚠ Every number is quoted from the hub's own artefact. ⛔ **The Hub re-derived nothing.**

---

## 0. The answer

| # | task / config | runs | verdict by the pre-registered rule | what survives scrutiny |
|---|---|---|---|---|
| 1 | ⭐⭐ **CMAZE-T · Config A1** — the Shapes recipe UNCHANGED (latent 128, K 2, 6 atoms, rank/width 0.094) | 3 (1–3) | **WIN — STORE row, unseen n-way** | ⭐ **survives** rivals trained 10× longer |
| 2 | **CMAZE-O · Config C (C-field)** (K 8, 64 atoms) | 3 (4–6) | WIN by the rule (0.693 vs 0.167) | ⛔ **rival margin WITHDRAWN** |
| 3 | **CMAZE-T · Config C-512** (latent 512, K 2, 8 atoms) vs 10×-trained rivals | 3 (7–9) | **WIN — STORE row** (0.725 vs 0.453) | replication of A1; ⚠ one per-seed caveat |
| — | CMAZE-T Config B (match-bit interface) | 0 | FAILED VAL (VB3) | the interface lifts the rival **2× more** than the CLU |
| — | CMAZE-O Config A (rank/width ≤ 0.3) | 0 | FAILED VAL (VB0) | the AE latent cannot be carried by a ≤ 0.3 store |
| — | CMAZE-O Config C2 (rank/width 0.5) | 0 | FAILED VAL (VB1, 10×-trained rivals) | — |
| — | Task 3 (CMAZE-2b route) | 0 | DECLARED NOT-RUN | gated on "no win in Tasks 1–2"; Task 1 won |

## 1. ⭐⭐ THE HEADLINE: a STORE row on CMAZE-T unseen n-way, and the controls DID collapse
⚠ **This was not expected.** The w15 harness shipped ready **with a warning** that a charter-C control did not collapse on the pair row. On the scored TEST leg at Config A1 **all four collapse**:

| arm | mean | per seed |
|---|---|---|
| ⭐ **CLU (A1)** | **0.7246** | 0.800 / 0.566 / 0.808 |
| best matched rival (transformer) | 0.4035 | 0.394 / 0.395 / 0.421 |
| rival mlp / lstm | 0.2194 / 0.1413 | — |
| control `landscape_removed` | **0.0000** | 0 / 0 / 0 |
| control `landscape_shuffled` | 0.2248 | — |
| control `across_wells` | 0.2435 | — |
| control `random_atoms` | 0.1655 | — |
| learned **BLIND-SEQUENTIAL** | 0.3418 | — |
| learned **BLIND-COLOUR** | 0.0716 | — |
| floor: **cross-maze lookup** | **0.0000** | 0 / 0 / 0 |

Bars: **SB1 PASS** (margin 0.321, ahead on every seed) · **SB2 PASS** (every control ≤ CLU − 0.15) · **SB3 PASS** (learned blind max 0.342 vs bar 0.675) · **SB4 PASS**. Replay **0 disagreements**, **0 impossible rows**, disjointness passes on all three TEST banks.
⇒ ⭐⭐ **The Shapes recipe, unchanged, transfers to a maze transition task and earns a store row.**

## 2. ⭐⭐ THE HUB WITHDREW ITS OWN WIN — and this is referee M1 reappearing live
⚠ **CORRECTED 2026-09-23 against the primary record** (`w16-maze-wm/`) by `w18/seeds-and-tables` pass 2. The Hub's first version of this section paraphrased and got eight things wrong; **the artefact wins in every one.** The corrected reading:

Runs 4–6 recorded a CMAZE-O **C-field** win (0.693 vs 0.167) **against rivals at the recipe's training budget**. Against its **own** 30k-step converged rivals, C-field loses: the matched **MLP reads 0.988 / 0.990 / 0.990** (`w16-maze-wm/FINDINGS.md:212`).
⚠ **The triples 0.088 / 0.150 / 0.173 vs 0.315 / 0.413 / 0.463 belong to Config C2 (latent), NOT to C-field** (`w16-maze-wm/FINDINGS.md:219–220`). ⛔ An earlier Hub statement printed them as the C-field comparison. **That row is withdrawn.**

⛔ **The CMAZE-O rival margin is WITHDRAWN BY THE HUB** (`MAZE-WM.json:22`). Runs 4–6 are attached to a claim that does not stand.
⭐ **But the C-field STORE GATE HOLDS** (`MAZE-WM.json:369`). ⛔ The earlier claim *"CMAZE-O yields no store row at any config"* **contradicts the record** and is withdrawn. What is withdrawn is the **composition-over-rivals** claim. It is still not a win, and no table prints it as one.
⚠ **CMAZE-O has THREE failing links, not one** (`MAZE-WM.json:377–379`): Config A **content/read**, C-field **interface** (vs converged rivals), C2 **read**.

The hub raised its own bar mid-wave (`PREREG-T2 AMENDMENT 4` requires 30k-step rivals). ⚠ **The 80,000-step re-test is Config A1 on VAL — a diagnostic, not a TEST re-score** (`TABLES.md:1281`). ⭐ **The scored leg against 80k-step rivals is C-512 on TEST** (W16-T6), and it won.
⚠ **This is exactly referee M1** (`w17`: Shapes rivals fit on 16,384 rows vs the store's 131,072). It cost **three scored runs on a claim that does not stand**, and it is the strongest argument yet that M1 needs a ruling.

## 3. ⛔ The negatives, printed — ⚠ corrected against the record
- ⛔ **CMAZE-T rollout validity = 0.000 for every arm that is the CLU or a MATCHED rival.** Per-step CLU 0.925 vs transformer 0.818; it compounds over 43 steps. ⚠ **Qualifier the Hub first omitted:** the 10×-float **diagnostic** transformer reads a NON-ZERO validity on seed 2 (`TABLES.md:138`). **Named link: READ.**
- ⛔ **CMAZE-O wins no composition-over-rivals row at any config** once rivals converge. ⭐ Its C-field **store gate still holds** — ⛔ *not* "no store row at any config".
- ⛔ **Config B's task-shaped interface lifts the RIVAL twice as much as the CLU** — the same shape as w14's verb finding.
- ⚠ Config C-512 seed 1: `random_atoms` **0.464** against the **CLU's 0.567** (`FINDINGS.md:169`) — a per-seed caveat that travels with the row.
- ⚠ **Exact deletion on the CMAZE-T winner (A1): statistic residual 7.7e-16 … 1.2e-15; only the ATOMS Δ is 0.0** (`FINDINGS.md:110`). ⛔ The Hub's earlier *"relative ΔS = 0.0 on every seed"* is the **CMAZE-O C-field** deletion's wording, not the CMAZE-T winner's.
- ⭐ **CMAZE-O deletion RAN** — C-field TEST, both scopes, every seed (`TABLES.md:877–888`). ⛔ An earlier Hub reading declared it not-run from its own silence. **The declared not-runs are only B, C2 and C2r.**

## 4. Budget — all 18 accounted
| runs | status |
|---|---|
| 1–3, 4–6, 7–9 | **SPENT** |
| 10–12 (T2 cap) | ⛔ **DECLARED NOT-RUN** — no CMAZE-O config passes VAL against converged rivals; **any field config would need n-way ≥ 1.02 to clear the 0.99 MLP**, which is impossible on a bounded metric. Seeding a VAL-failing config would break the val-first rule. |
| 13–18 (Task 3) | ⛔ **DECLARED NOT-RUN** — the brief opens Task 3 *"ONLY if Tasks 1–2 yield no win"*; Task 1 won. |
⭐ **The Hub's judgement: this is a correct stop, not an early one.** "Keep trying configs" is subordinate to "scored runs go ONLY to a config that passes its val bars", and the hub proved no remaining CMAZE-O config *could* pass.

## 5. What the Advisor is asked to rule on
1. ⭐⭐ **The maze section now has a positive result.** CMAZE-T carries a store row that survives 10×-trained rivals. **Does it become the maze section's headline, with CMAZE-O and rollout validity as the printed negatives?**
2. ⭐⭐ **M1, again and now twice.** A recorded win was withdrawn purely because rivals were under-trained. **Is "naive vs naive" defensible as parameter-only, or must every rival be trained to convergence and the Shapes headline restated at the data-matched margin (+3.117…+17.875 rather than +39…+49)?**
3. ⚠ **Rollout validity is 0.000 for every arm.** ⛔ Is a world-model claim sustainable when no arm can chain 43 generated steps? Or does the paper scope the claim to single-step transitions and print the rollout row as the limitation?
4. ⚠ **CMAZE-T and CMAZE-O never share a table.** Confirm they are reported as two instruments, with the CMAZE-O negative given equal prominence.
