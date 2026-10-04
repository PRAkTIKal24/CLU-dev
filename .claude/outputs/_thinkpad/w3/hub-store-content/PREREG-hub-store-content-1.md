# PREREG — hub/store-content — SCORED RUN 1

**Written 2026-09-19, BEFORE any held-out corner has been read.**
Target budget drawn on: **`generation-table`**, 1 already spent, **2 left** → this is **scored 2/3**.
Charter §A.1–A.3 · w3-hub-common §3, §5.
⛔ Amendments are appended below with a date; ⛔ nothing above an amendment is ever edited.

---

## 0. The dial and the question

**Dial: WHAT IS STORED.** Does writing **earlier-layer content alongside the penultimate map, at
matched bytes**, recover `scale`, `orientation` and `shape` on scoreboard rows 1–4?

⭐ This is a **reallocation, not extra capacity**: the budget is 524,288 B at
`byte_match_tolerance = 0`, `store_bytes = N·D·4`, so `N·D = 131,072 = 2^17` exactly, and every
added dimension is paid for in atoms.

## 1. The arms, completely pinned

Content is concatenated on the **channel axis at the penultimate map's own 4×4 grid**; earlier
blocks are brought there by an exact 4×4 block-mean pool.

| label | content | D | N | store bytes | role |
|---|---|---|---|---|---|
| `P` | `p` | 2048 | 64 | 524,288 | incumbent content (the thing being reallocated away from) |
| `WIDEN` | `p ⊕ pool(b3)` | 4096 | 32 | 524,288 | **THE ARM** |
| `WIDEN_SHUF` | `p ⊕ rowshuffle(pool(b3))` | 4096 | 32 | 524,288 | laundering control for `WIDEN` |
| `WIDEN_MIX` | `p ⊕ half(pool(b3)) ⊕ pool(b2)` | 4096 | 32 | 524,288 | two earlier depths |
| `WIDEN_MIX_SHUF` | as above, shuffled | 4096 | 32 | 524,288 | its laundering control |
| `basis_head` | `p` | 2048 | 64 | 524,288 | ⭐ **THE LIKE-FOR-LIKE** (w3-hub-common §1) |
| `basis_head@WIDEN` | `p ⊕ pool(b3)` | 4096 | 32 | 524,288 | ⭐ the dial at a **fixed strong read** |
| `knn_proto` | `p` | — | — | 475,136 (0.906×, declared) | second control |

**Every knob of the store fit, pinned here** (w3-hub-common §3 — wave 2 pinned ridge and left the
rest free, and its headline then failed to replicate):

* partition: `gen_store.correlation_grouping`, `method="auto"` (walks `average` then `ward`),
  `min_well_frac = 0.10`, dead coordinates from `gen_store.dead_channels`, **K = 5 pinned** from the
  frozen `store_plan.json` — ⛔ K is **not** re-chosen on the wider matrix.
* atoms: `gen_landscape.select_atom_rows`, `selection = "kmeanspp"`, `seed = 0`. Atoms are genuine
  stored train rows and `assert_atoms_are_train_rows` runs on every arm.
* bandwidth: `atom_bandwidths(frac = 1.0, rule = "nn_median")`.
* address book: `fit_atom_address_book`, Laplace `smoothing = 1.0`.
* read: `launch_state(mode = "clamp_atom")`, `clamp_threshold = 0.5` (reporting-only),
  **`settling_steps = 0`**, `read_mode = "state"` — the pin's own `x0` arm.
* per-block RMS standardisation: one scalar per block, measured **on the atom rows**, free by the
  `coupling_byte_ledger` convention (`m_k`/`s_k` are functions of the atoms).
* `basis_head`: `RIDGE_PROJ = 1e-2`, `RIDGE_HEAD = 1e-1`, one-hot design over Σ_f V_f = 57,
  signed linear read. ⛔ No simplex.
* store fit pool: value-holdout train split → `rng(seed·7919 + 20,260,910)` → 32,768 rows →
  `stratified_fit_val(n_fit = 32,256, n_val = 512, seed = 0)`. The store fits on **`fit` only**.
* **Seed 0 only** (w3-hub-common §5). ⇒ this run buys a **verdict, not a paper number**.

## 2. The estimand — ⛔ never a single cell

**The 49 `nway-*` corners of the value-holdout condition, 72 rows each, PAIRED per corner.**
The statistic is the **per-corner paired difference** `Δ_c = metric(arm, c) − metric(control, c)`,
reported as mean ± the paired standard error over the 49 corners, plus the **sign count**
(corners won / tied / lost) and a two-sided Wilcoxon signed-rank p on the 49 paired differences.
Row 3 uses the 3 `pair-*` corners the same way and is reported, never claimed at n = 3.

## 3. ⭐ THE BARS, declared in advance

**Primary (rows 1–4, the target):** `WIDEN` beats `basis_head`, both at 524,288 B, on the paired
49-corner estimand, by a **mean paired margin ≥ +0.02 aggregate accuracy** with the lower end of
the paired 95 % interval above 0.
⛔ `decoration_margin = 0.02`. A tie is reported in the scoreboard's own words: *"A CLU mechanism
that ties basis+head is decoration"*.

**⭐ THE HYPOTHESIS BAR — this is the one the hub is actually about, and the aggregate does not
substitute for it.** The dial claim is specifically about two factors. `WIDEN` must beat `P` on

* `scale` by **≥ +0.05** paired mean over 49 corners, and
* `orientation` by **≥ +0.05** paired mean over 49 corners,

with the same content read (`WIDEN` vs `P`, both the `clamp_atom` arm) **and** at a fixed read
(`basis_head@WIDEN` vs `basis_head`). ⭐ Two legs because a move on only one of them is a read
effect, not a content effect.

**Direction expected:** `scale` and `orientation` **up**; `shape` up or flat; hues flat or slightly
**down** (they are already at 0.94–1.00 and the atom count halves).

**Floors (charter §C, every arm, every row):** `full` beats `atom_mean_lookup` **and**
`row_lookup` beyond seed spread, or the store is decorative and is reported as such;
`atoms_shuffled` must collapse; an arm whose read sits on its nearest atom is a lookup table.

## 4. ⛔ What falsifies this hub

1. `WIDEN` at matched bytes leaves `scale` and `orientation` within ±0.05 of `P` ⇒ **the loss is the
   quantiser's / the read's, not the layer's**. Clean negative; the content dial is the wrong dial.
2. The aggregate moves but `scale` and `orientation` do not ⇒ won on something else; say what.
3. `WIDEN` ≤ `WIDEN_SHUF` ⇒ the gain is geometry (width × fewer atoms), not content.
4. `WIDEN` ≤ `atom_mean_lookup` or ≤ `row_lookup` ⇒ §C.1, reported as a lookup.

## 5. ⚠ Declared deviations and known limits, before the run

* ⛔ **`SWAP` is unavailable.** Replacing half the penultimate channels is byte-legal and was the
  cleanest isolation of content, but the frozen decoder's input is `(128,4,4)` and no arm may
  substitute its own decoder. ⇒ at a 2^17 budget with a fixed 2048-dim render interface the content
  dial has exactly one setting: widen and halve the atoms.
* ⚠ The well partition here is **not** the frozen plan's `[303,305,573,501,366]`; the pool is the
  value-holdout train split, not Montero's, so the correlation matrix differs. Same rule, different
  condition, declared.
* ⚠ `D = 4096 ≤ WHITENING_D_LIMIT = 20,000`, so a LACE row would remain fair at this width. LACE is
  not run by this hub.
* ⛔ CPU-only. Every feature this hub uses is produced on local CPU by the frozen encoder; the last
  block of the re-implementation is **bit-identical** to `gen_bridge.encode_images`. No comparison
  mixes devices.
* ⛔ Rows 5, 6 and 7 are filled and reported for comparability (w3-hub-common §2) and **no row-5 or
  row-6 number is quoted as evidence for the store** (w3-hub-common §7).

## 6. Selection already done, on `val` only (n = 512, store pool held back)

`P` 0.5618 · `basis_head` 0.7891 · `knn_proto` 0.8424 (aggregate). These chose nothing about
`WIDEN`; they established the bench, reproduced the known `scale`/`orientation` deficit, and
reproduced common §4.5 (44.3 % negative coefficients, 0.00 % probability vectors).
⭐ The `WIDEN` selection numbers will also be taken on `val` and recorded in `LOG.md` **before**
this scored run executes.

---
