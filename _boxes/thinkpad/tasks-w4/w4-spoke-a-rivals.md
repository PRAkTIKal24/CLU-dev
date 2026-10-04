# Task: w4-spoke-a-rivals — the rival set, and cache generation 2

- **Agent:** `experiment-engineer` · **Branch:** `agent/experiment-engineer/w4-rivals`, off **`base/iter4 = 9dfed5e`**
- **Machine:** ThinkPad (FORGIS_14), local CPU. **Output dir:** `/c/chlu/outputs/spoke-a-rivals/`
- **Report:** `/c/chlu/outputs/spoke-a-rivals/REPORT.md` · running notes in `LOG.md`, dated bullets.
- ⭐⭐ **You are the critical path for wave 4.** Four hubs may do DEV work now but ⛔ **cannot spend a
  scored run until you publish §5's readiness file.** Publish it the moment it is true, not at the end.

## DIAL DECLARATION
**Dial: the instrument and the comparison set.** You run no arm of ours and make no claim.
**Falsified if** a rival cannot be brought to matched parameters within ±10 %, or if generation-2
re-encoding does not reproduce its own hashes on a second pass — either makes the wave's numbers
unusable and the Head needs to know **immediately**, not in the report.

---

## 1. The one job
**Build the rival set every wave-4+ row is scored against, and produce cache generation 2 so the
programme can finally run three seeds.**

## 2. The rival set — §1 of the wave plan
**Specialists:**
- **prototype table** (per-value means) — the incumbent.
- **LACE, selection REPAIRED on the stratified split.** ⚠ `GenCellConfig.lace_selection_source` points
  at the broken-split historical selection; the harness flags `lace` as regressed (0.6136 measured vs
  0.7077 published). ⭐ **Repair it in your own module — ⛔ do NOT edit the pinned config.** Print both
  the repaired and the historical number so the change is auditable.
- **product-of-experts EBM at its OWN budget.** Print the byte multiple beside every row. Run it
  **twice**. ⭐ **If it wins everything outright it goes in the appendix WITH its numbers — ⛔ never hidden.**

**Generic learners, matched TOTAL parameters (±10 %, count printed), same split, same labels, same
frozen decoder, same frozen judge, 3 seeds:**
- **MLP** — request one-hots → 2048 features
- **transformer** — 6 factor tokens, 2 layers → features
- **LSTM** — 6 tokens → features
- **continuous-input MLP** — hue as a *number*; ⭐ this one exists for row 2 and must accept an
  off-grid hue.

⛔ **Rows a rival cannot enter (5–7) get a dash and a footnote. ⛔ A dash is NEVER a win for us.**
⭐ **`retrain-from-scratch without the value` is the gold-standard deletion comparator** — build it and
**print its cost line**; wave 5 needs it.

## 3. ⭐⭐ Cache generation 2 — the seeds, and the one rule that matters
Produce **bridge seeds 1 and 2** by **re-encoding from pixels**, hashed, as a declared **generation 2**.
- ⛔⛔ **Generation 2 must be produced ENTIRELY ON ONE DEVICE, and that device is the local CPU.**
  CPU and GPU features agree to ~53 ULP on ~1 % of elements. A cache half-built on each is silently
  void. ⭐ **There is no GPU case here: one 480,000-row pass measured 26 min on this box in wave 3.**
- ⛔ **Never mix generation-1 and generation-2 numbers in one comparison**, in either direction.
- ⭐ **Re-score every existing arm on generation 2** so the corpus is internally consistent.
- **Hash everything** and record: encoder param hash, per-seed `features.npy` / `feature_rows.npy`
  sha256, row counts, dtype, and the exact command that produced them.
- ⭐ **Determinism check, and it is your own falsifier:** re-encode **one** seed's first shard a second
  time and assert the sha256 matches. If it does not, ⛔ **stop and report** — generation 2 is not a
  generation, it is noise, and every wave-4+ number would inherit it.

⚠ Generation 1 stays on disk, untouched and still hashed. ⛔ Do not delete or overwrite it.

## 4. Files
- **You own:** a new `chlu/eval/rivals/` module tree and a new `chlu/experiments/exp_w4_rivals.py`;
  everything under `/c/chlu/outputs/spoke-a-rivals/`; and the generation-2 cache under a **new**
  directory that names its generation.
- ⛔ **Read-only — editing any of these voids the wave:** `exp_gen_cell.py` · `gen_splits.py` ·
  `gen_rowsplit.py` · `gen_store.py` · `gen_landscape.py` · `gen_bridge.py` · `config.py` ·
  `gen_scoreboard.py` · `gen_value_holdout.py`.
- ⚠ `chlu/eval/rivals/gen_knn.py` already exists at the pin. **Extend around it; do not rewrite it.**

## 5. ⭐⭐ THE READINESS FILE — four hubs are blocked on this
Write `/c/chlu/outputs/spoke-a-rivals/RIVALS-READY.json` **as soon as it is true**, and update it as
each piece lands:
```json
{"status": "ready", "generation": 2,
 "rivals": {"table": true, "lace_repaired": true, "poe_ebm": true,
            "mlp": true, "transformer": true, "lstm": true, "cont_mlp": true},
 "seeds": [0, 1, 2], "cache_gen2_root": "<abs path>",
 "hashes": {"...": "..."}, "param_counts": {"...": 0}, "utc": "..."}
```
⭐ **Partial is useful: set `status: "partial"` with per-rival booleans the moment the first rival is
scorable.** A hub can start scoring against the table while the transformer still trains.

## 6. Acceptance
1. Every rival in §2 built, at matched parameters **±10 % with the count printed**, 3 seeds.
2. LACE repaired, **both** numbers printed.
3. PoE-EBM run twice, byte multiple printed.
4. Generation 2 for seeds 1–2, hashed, **CPU-only**, with the re-encode determinism check passing.
5. Every existing arm re-scored on generation 2, generation-1 numbers never mixed in.
6. `RIVALS-READY.json` on disk and accurate.
7. `retrain-from-scratch` comparator built, with its cost line.

## 7. ⛔ What would prove this spoke wrong
- A rival cannot reach matched parameters within ±10 % ⇒ report the achievable count and **stop**;
  do not quietly ship a mismatched rival.
- The re-encode determinism check fails ⇒ **generation 2 is void. Stop and report immediately.**
- Re-scoring on generation 2 moves a published generation-1 arm by more than its seed spread ⇒ that is
  a finding about the instrument, ⭐ **report it before any hub scores against it.**
- The PoE-EBM beats everything at its own budget ⇒ ⛔ not a reason to bury it. Appendix, with numbers.
