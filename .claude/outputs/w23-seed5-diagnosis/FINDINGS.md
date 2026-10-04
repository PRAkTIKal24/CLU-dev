# FINDINGS — w23 spoke/seed5-diagnosis (FORGIS_14)

⛔ **Zero scored runs, zero re-fits, no new seed, no ledger row, no PREREG.** Every object was
loaded from disk and hash-checked before it was read. Its file sha256 and param sha256 had to
match both its `.sha256` sidecar and `MANIFEST.json`, and the param hash was also recomputed. That
covers the codec and `clu.pkl` for A1 seeds 0–5 and CLU-REL-A10 seeds 0–5, plus the four seed-5
A1 controls. Every re-read object reproduced its run-JSON number exactly (the float `==` is
asserted in code). Nothing was written outside this directory and the worktree. Nothing was written
to `w17-paper/`, `figures/` or `paper-visuals/`.

Branch `w23/seed5-diagnosis` (from `w21/rel-newseeds` 31eeadd). Scripts:
`scripts/w23_seed5_trace.py`, `w23_seed5_content.py`, `w23_content_allseeds.py`,
`w23_rel_value.py`, `w23_summary.py`. Machine record: `W23-SEED5-DIAGNOSIS.json`.
Full printouts: `Q1-TRACE.txt`, `Q1-CONTENT.txt`, `Q1-CONTENT-ALLSEEDS.txt`, `Q2-REL-VALUE.txt`.
Raw per-row JSON: `Q1-TRACE.json`, `Q1-CONTENT.json`, `Q1-CONTENT-ALLSEEDS.json`,
`Q2-REL-VALUE.json`.

---

## QUESTION 1 — why the seed-5 base n-CLU scores 0 on `(2,0,1,2)` and `(2,0,3,2)`

### Answer: **CONTENT.** The design is fine, and the address book has an error of ordinary size (not larger than on tuples the CLU gets right).

Both failing tuples need the same outcome code, **`(legal 1, disp 3 = S, event 0, pocket 2)`**. The
seed-5 store holds that code badly. Project its latent onto the deployed atoms (mean + 5
directions in each 64-wide well; deployed rank [5, 5]): the relative residual is **0.387** and the
L2-best in-span point **decodes displacement E**. The S-vs-E logit margin there is **0.04**. The head
is fitted in L2 against these projected targets, so it aims at the S/E decision boundary. Any
ordinary address error then flips the displacement. The CLU emits **N** (margin 1.35 / 2.36) on
all 170 rows. The same code is **already wrong on 1/3 of the FIT rows that carry it**: 211 / 317
exact. So this is **not a hold-out effect at all**: the store cannot reliably emit this code even
for requests it was fitted on.

### Steps 1–4, printed (main request of each tuple; every request is in `Q1-TRACE.txt`)

Words = `(w_agent, w_key, w_door, w_tint, w_dir)`. Design = `[bias | 27 main | 290 pair]` = 318.
Every request has exactly 16 non-zero columns (all = 1): bias, 5 main one-hots and 10 pair cells.
The full 318-vector of each request is `design_row_318` in `Q1-TRACE.json`. K = 2 wells, so the
request addresses **both** wells with 6 coefficients each. The first atom is the well mean
(coefficient ≈ 1.0 on every request).

| | **(2,0,1,2) ✗** | **(2,0,3,2) ✗** | **(2,0,3,3) ✓ comparison** | **(3,0,1,0) ✓ comparison** |
|---|---|---|---|---|
| rows / CLU exact | 90 / **0** | 80 / **0** | 68 / 68 | 74 / 74 |
| main request (rows) | (2,0,0,1,2) (76) | (2,0,0,3,2) (75) | (2,0,0,3,3) (68) | (3,0,0,1,0) (69) |
| non-zero design cols | [0] bias; [3] agent=2; [6] key=0; [11] door=0; [19] tint=1; [25] dir=2; [38] a×k(2,0); [67] a×d(2,0); [99] a×t(2,1); [125] a×dir(2,2); [138] k×d(0,0); [174] k×t(0,1); [200] k×dir(0,2); [224] d×t(0,1); [260] d×dir(0,2); [300] t×dir(1,2) | same with tint=3: [21], [101], [226], [310] | same as (2,0,3,2) but dir=3: [26], [126], [201], [261], [311] | see `Q1-TRACE.txt` |
| **c well 0** = φW₀ | [0.998, −2.476, −1.494, 0.026, −0.276, −0.605] | [0.999, −2.687, −1.715, 0.148, −0.477, −0.624] | [1.020, −4.728, −3.244, 0.349, −2.738, −1.144] | [0.977, −5.760, −0.363, 0.550, −0.365, 3.160] |
| c* well 0 (L2-best for truth) | [0.986, −1.996, −1.091, −0.097, 0.542, −0.738] | same code → same c* | [1.035, −5.186, −3.454, 0.207, −3.134, −1.460] | [0.972, −5.813, −0.819, 0.891, −1.005, 3.697] |
| **c well 1** = φW₁ | [1.010, −2.373, −1.872, 1.841, 3.286, −0.213] | [1.009, −2.354, −1.793, 1.794, 3.215, −0.395] | [0.968, −1.518, −0.167, 1.370, 1.245, −2.834] | [1.024, −4.166, 0.635, −3.737, 2.181, −0.604] |
| c* well 1 | [0.999, −2.572, −2.219, 2.006, 3.332, 0.486] | same | [0.957, −1.210, 0.318, 1.321, 0.438, −3.382] | [1.046, −4.591, 0.616, −3.830, 2.987, −0.874] |
| ‖c − c*‖ (w0, w1) = **address error** | 1.046, 0.823 | 1.407, 1.032 | 0.728, 1.133 | 1.013, 0.956 |
| **emitted** [legal, disp, event, pocket] | **[1, 1, 0, 2]** | **[1, 1, 0, 2]** | [1, 4, 0, 2] | [1, 1, 0, 3] |
| truth | [1, 3, 0, 2] | [1, 3, 0, 2] | [1, 4, 0, 2] | [1, 1, 0, 3] |
| wrong fact | **displacement** (N for S) | **displacement** | — | — |
| emitted margins (per block) | 16.35 / **1.35** / 13.35 / 12.76 | 16.61 / **2.36** / 13.34 / 13.29 | 15.53 / 4.16 / 12.48 / 15.72 | 15.17 / 3.49 / 16.34 / 14.58 |

### Step 5 — the locus tests (same columns)

| test | (2,0,1,2) ✗ | (2,0,3,2) ✗ | (2,0,3,3) ✓ | (3,0,1,0) ✓ | reads as |
|---|---|---|---|---|---|
| **DESIGN:** null-space fraction of φ in the train design (rank 122 of 318) | 1.2e-15 | 1.6e-15 | 1.6e-15 | 7.0e-16 | φ is **estimable** for all four |
| DESIGN: rank of the train design restricted to rows sharing ≥ 1 pair cell with φ | 114 (12,362 rows) | 114 (12,345) | 113 (12,361) | 111 (12,293) | same order for all four |
| DESIGN: φ in the row space of that restricted design (rel. residual) | 1.3e-14 | 1.0e-14 | 1.2e-14 | 5.4e-15 | yes, for all four |
| DESIGN: rank drop when any one of the 16 non-zero columns is removed | 0 for all 16 | 0 | 0 | 0 | the structural one-hot collinearity (bias = Σ main block); **identical for right and wrong tuples, so not discriminating** |
| DESIGN: ridge leverage φ(A+λI)⁻¹φᵀ | 0.0068 | 0.0086 | 0.0076 | 0.0058 | no extrapolation outlier |
| **ADDRESS:** ‖c − c*‖ (w0, w1) | 1.05, 0.82 | 1.41, 1.03 | 0.73, 1.13 | 1.01, 0.96 | the same size as on the right tuples. Other tuples the CLU gets right go up to 4.88 (Q1-TRACE SUMMARY) |
| **CODEC:** decode(encode(truth)) | ok | ok | ok | ok | the frozen codec is not the cause |
| **CONTENT:** truth code's span rel. residual | **0.387** | **0.387** | 0.239 | 0.164 | |
| CONTENT: decode of L2-best in-span point | **[1, 2, 0, 2] ✗** | **[1, 2, 0, 2] ✗** | [1, 4, 0, 2] ✓ | [1, 1, 0, 3] ✓ | |
| CONTENT: its displacement margin | **0.04** | **0.04** | 7.26 | 5.60 | |
| CONTENT: CLU **in-sample** exact on the FIT rows with this truth code | **0.666** (317 rows; emits S 211, E 63, N 43) | 0.666 | 1.000 (486) | 1.000 (455) | the failure exists **inside the fit pool** |

**The isolating pair.** `(2,0,3,2)` against `(2,0,3,3)`: same pocket, no door, same tint, same key
word. Only the direction differs, so only 5 design columns differ. The design is equally estimable
for both (1.6e-15 vs 1.6e-15), and the address error is the same size. The one thing that
separates them is the **truth code** each must emit: `(1,3,0,2)` sits on a decision boundary of
the store's span (margin 0.04), while `(1,4,0,2)` sits well inside (margin 7.26).

**Block-swap counterfactual** (Q1-TRACE ISOLATION): take the failing request and replace one factor
block with the comparison request's block. Replacing `w_tint × w_dir` flips the emission to the
truth `[1,3,0,2]`. Replacing `w_door × w_dir` moves it to W (`[1,4,0,2]`). All other blocks leave
it at N. So a point in the span that decodes S **does exist**. The span is not empty of S, but the
L2 target the head is fitted to lies on the S/E boundary. That makes the emitted displacement
depend on small address perturbations: it is correct for 2/3 of the fit rows and wrong for these
two held-out requests.

### The content failure over all 27 codes (seed 5; `Q1-CONTENT.txt`)
- Codes whose L2-best in-span point decodes wrong: **7 of 27**. Four are the unreachable
  `interaction` codes (0 fit rows). The other three are `(1,0,3,3)`, `(1,0,1,1)` (key pick-ups,
  **in-sample 0.000** on their 163 / 166 fit rows) and **`(1,3,0,2)`**. It is the only such code
  that an n-way tuple needs.
- Over the 23 codes with ≥ 20 fit rows, Pearson(span residual, CLU in-sample exact) = **−0.858**.
  Every code with residual ≤ 0.301 is 1.000 in-sample.
- ⚠ **The "eighth baseline" is a CLU control.** Of the 8 trained arms at 1.000 on both tuples
  (PERTUPLE: rival mlp, rival transformer, @10×-steps mlp and transformer, three 10×-floats
  diagnostics, and `control: landscape_shuffled_across_wells`), one is the CLU's own
  across-wells control. It is right here **by accident**, not because its store is better. Its span
  residual on `(1,3,0,2)` is 0.935, its L2-best point decodes wrong on 16 of 27 codes, and it scores
  0/68 and 0/74 on the two comparison tuples. The other three controls score 0 or 1 of 80–90 on the
  failing tuples. So **seven** genuine trained baselines solve them, not eight.

### Is this seed-5-specific? (`Q1-CONTENT-ALLSEEDS.txt`, A1 BOTH TEST seeds 0–5)
- Across all 6 seeds, seed 5's `(1,3,0,2)` is the **only** code with ≥ 20 fit rows whose L2-best
  in-span point decodes wrong for a held-out n-way tuple. The other wrong codes are interaction
  codes with 0–15 fit rows and in-sample 0.000.
- **Cross-seed control:** the **same tuple `(2,0,3,2)` with the same truth code `(1,3,0,2)` is 72/72
  on seed 3**. There the code's span residual is **0.197** (margin 6.55; 247 fit rows; in-sample
  1.000). Same design, same request; a different store (and a different seed's codec) holds the
  code. The failure moves with the content.
- Every other A1 n-way miss on seeds 0–4 whose code has fit rows (838 rows) has content-OK codes.
  Those misses are the hold-out-coverage misses (class D, blocked 2-way pairs) that the maze-t-seed4
  diagnosis already located.

### What would have proved this wrong
- φ in the null space of the train design, or a restricted support rank that drops only for the
  failing tuples → DESIGN. Measured 1e-15 everywhere; ranks 111–114 for all four tuples.
- ‖c − c*‖ markedly larger on the failing tuples → ADDRESS BOOK. Measured 0.82–1.41, against
  0.73–1.13 on the comparisons.
- A correct L2-best in-span decode and in-sample 1.000 on the truth code → CONTENT cleared.
  Measured: wrong decode (margin 0.04) and in-sample 0.666.
- A codec that cannot round-trip `(1,3,0,2)` → the frozen codec, not the CLU. Measured: it
  round-trips.

---

## QUESTION 2 — the `n-CLU-REL` (CLU-REL-A10) value hold-out

### The numbers (run JSONs `ROWS.value`; `Q2-REL-VALUE.txt` Part A)
| seed | n | CLU (JSON) | brief | best matched rival (8k/80k) | best incl. 10×-floats | best incl. 10×-floats + `+rel` |
|---|---|---|---|---|---|---|
| 0 | 4000 | 0.48875 | 0.4888 | 0.6817 transformer 8k | 0.7242 10×-floats transformer | 0.8460 transformer +rel 80k |
| 1 | 4000 | 0.50075 | 0.5008 | 0.8612 transformer 80k | 0.8942 10×-floats transformer | 0.9107 transformer +rel 8k |
| 2 | 4000 | 0.51175 | 0.5118 | 0.8385 transformer 80k | 0.8960 10×-floats transformer | 0.8960 |
| 3 | 4000 | 0.52525 | 0.5252 | 0.8223 transformer 8k | 0.8738 10×-floats transformer | 0.9018 transformer +rel 8k |
| 4 | 4000 | 0.69175 | 0.6917 | 0.7480 transformer 80k | 0.7480 | 0.7480 |
| 5 | 4000 | 0.60375 | 0.6038 | 0.8305 transformer 80k | 0.8305 | 0.9032 transformer +rel 8k |
| **mean 6** | | **0.5537** | 0.5537 | **0.7970** | 0.8278 | **0.8676** |

- ✅ The brief's six CLU values and the 0.5537 mean reproduce. The per-seed 5e-5 differences are
  the brief's 4-dp rounding of x.xxxx5 values.
- ⚠ **The brief's "best arm per seed mean 0.8676" is the best over the `+rel` and 10×-floats
  DIAGNOSTICS**, which are never bars. The **best matched rival** 6-seed mean is **0.7970**. Both
  are printed above. The CLU is below the best matched rival on the value row on all 6 seeds.
- ⭐ **§9's "0.500 vs 0.792" is located and reproduces.** It is a pair of **3-seed means over seeds
  0–2**, not a per-seed pair. CLU = mean(0.4888, 0.5008, 0.5118) = **0.5004**; `rival@10x-steps:
  transformer` (= matched transformer @80,000 steps) = mean(0.6767, 0.8612, 0.8385) = **0.7922**.
  The source is `w21-maze-t-clu-improve/TABLES.md:105` (R\* column), `:252`, and `FINDINGS.md:183`.
  No single seed and arm cell equals 0.792. The Hub's non-reproduction came from comparing it
  against per-seed cells.

### The mechanism, shown (`Q2-REL-VALUE.txt` Part B; all six seeds, identical pattern)
Design: 396 = 318 base (pair block × 0.5) + 6 `eq` columns + 72 `eq × {door (7), dir (5)}` crosses.
**90 columns name colour 4**: identity one-hots, pair cells, and the 6 `eq × door=4` crosses.

| seed | colour-4 cols on FIT rows max\|Φ\| | head rows W[C4] max\|·\| (w0, w1) | other head rows rms | read(value) − read(value, C4 zeroed) max\|Δz\| | emissions identical |
|---|---|---|---|---|---|
| 0 | 0.0 | 3.6e-10, 4.5e-10 | 0.52, 0.55 | **0.00** | 1.0000 |
| 1 | 0.0 | 6.4e-10, 6.4e-10 | 0.47, 0.46 | **0.00** | 1.0000 |
| 2 | 0.0 | 7.4e-11, 3.4e-10 | 0.47, 0.51 | **0.00** | 1.0000 |
| 3 | 0.0 | 5.2e-10, 5.8e-10 | 0.54, 0.52 | **0.00** | 1.0000 |
| 4 | 0.0 | 2.2e-10, 3.4e-10 | 0.47, 0.49 | **0.00** | 1.0000 |
| 5 | 0.0 | 6.0e-10, 4.3e-10 | 0.48, 0.46 | **0.00** | 1.0000 |

**⚠ The brief's premise, corrected by the artefact.** "The `eq`-columns are ALL-ZERO for the
never-seen colour" is **false as stated**. What is all-zero is every column that **names** colour
4. The `eq × door=4` crosses belong to that set: 0 fit rows, head rows ≤ 1e-38, yet non-zero on
198–257 value rows per seed. The **six base `eq` columns are colour-anonymous and do fire on value
rows, including when both words are 4**. For seed 5 (other seeds in `Q2-REL-VALUE.txt`):

| eq column (seed 5) | fit rows active | value rows active | …because both words = 4 | head row norm (w0, w1) |
|---|---|---|---|---|
| eq(agent, key) | 0 | 0 | 0 | 0.000, 0.000 |
| eq(agent, door) | **0** | 81 | 57 | **0.000, 0.000** (zero in fit on seeds 1 and 5) |
| eq(agent, tint) | 1314 | 286 | 262 | 1.660, 2.442 |
| eq(key, door) | 4 | 6 | 5 | 0.246, 0.171 |
| eq(key, tint) | 286 | 55 | 32 | 0.112, 0.107 |
| eq(door, tint) | 770 | 276 | 204 | 0.129, 0.162 |

So a value request **does** reach the store through its relations: "pocket = tint" fires for 4 = 4
exactly as it does for 2 = 2. What it **cannot** reach is **which** colour. Colour-4 identity
contributes nothing to the read, to the float (Δz = 0.00 on every seed). An `eq` column says
"same colour" but not "colour 4".

**Why that loses the value row: about half its rows must *name* colour 4 in the outcome.**

| seed | truth names colour 4 (pocket_next = 4 or key event on 4): n / CLU exact | truth colour-free: n / CLU exact | dominant wrong fact (names-4 rows) |
|---|---|---|---|
| 0 | 2020 / 17 (0.008) | 1980 / 1938 (0.979) | pocket 1907 |
| 1 | 1986 / 3 (0.002) | 2014 / 2000 (0.993) | pocket 1892 |
| 2 | 1943 / 0 (0.000) | 2057 / 2047 (0.995) | pocket 1819 |
| 3 | 1946 / 115 (0.059) | 2054 / 1986 (0.967) | pocket 1747 |
| 4 | 2071 / 863 (0.417) | 1929 / 1904 (0.987) | pocket 1139 |
| 5 | 2066 / 505 (0.244) | 1934 / 1910 (0.988) | pocket 1502 |
| **pooled** | **12032 / 1503 (0.125)** | **11968 / 11785 (0.985)** | |

The value row is lost on the rows whose answer is a colour-4 **name**. The largest block is
"pocket carries colour 4, so pocket_next = 4": 1,635–1,728 rows per seed, `w_agent` role in the
per-role tables. Rows whose answer is colour-free are 0.967–0.995 exact. The CLU emits colour 4 at
all only where the colour-anonymous read happens to decode to 4 (seeds 4 and 5: 863 / 572 rows).
**Store, secondary:** no colour-4 code is in any fit pool (0 of 10). The span residual of the 10
colour-4 codes is 0.21–0.39 against 0.08–0.17 for the others, and their L2-best in-span decode is
right for 6–10 of 10. So even a head that could name 4 would face a weaker store on those codes.

**Per-tuple table, value row, seed 5** (top 15 by n; every tuple of every seed is under
`partB.<seed>.per_tuple` in `Q2-REL-VALUE.json`; tuple = (w_agent, w_door, w_tint, w_dir)):

| tuple | n | CLU exact | truth names 4 | rows with an eq column active | rows with a colour-4 column active |
|---|---|---|---|---|---|
| (4,0,0,4) | 133 | 48 | 133 | 0 | 133 |
| (4,6,0,0) | 100 | 16 | 100 | 0 | 100 |
| (4,6,0,3) | 99 | 75 | 99 | 0 | 99 |
| (4,6,0,1) | 96 | 75 | 96 | 0 | 96 |
| (4,6,0,2) | 89 | 14 | 89 | 0 | 89 |
| (4,0,0,1) | 84 | 0 | 84 | 0 | 84 |
| (4,0,0,3) | 81 | 0 | 81 | 0 | 81 |
| (4,0,0,0) | 75 | 0 | 75 | 0 | 75 |
| (0,0,4,0) | 73 | 73 | 0 | 3 | 73 |
| (4,0,0,2) | 70 | 62 | 70 | 0 | 70 |
| (0,0,4,2) | 69 | 69 | 0 | 1 | 69 |
| (0,0,4,1) | 64 | 64 | 0 | 2 | 64 |
| (0,0,4,3) | 63 | 63 | 0 | 2 | 63 |
| (4,0,2,3) | 61 | 0 | 61 | 2 | 61 |
| (4,0,1,2) | 61 | 51 | 61 | 5 | 61 |

### What would have proved Q2 wrong
- A non-zero colour-4 column on any fit row, a non-zero head row there, or Δz ≠ 0 when the colour-4
  columns are zeroed. Measured: 0.0 / ≤ 6.4e-10 / 0.00 on every seed.
- Colour-free-truth value rows failing as badly as names-4 rows. Measured: 0.985 vs 0.125 pooled.

---

## What did NOT run (and why)
- **No scored run, no re-fit, no new seed, no ledger row, no PREREG.** Banned by the brief.
- No counterfactual re-fit (e.g. more atoms per well to test whether `(1,3,0,2)` becomes
  representable). That would be a re-fit. The span-residual / L2-oracle / in-sample tests and the
  seed-3 cross-seed comparison stand in for it.
- Rivals were not traced. Their per-tuple counts were read from `check_BOTH_test_seed5_PERTUPLE.json`,
  not re-scored.
- Q2 re-scored only the CLU's value row, per seed. Controls and rivals on the value row were read
  from the run JSONs, not re-scored.
- The A1 seed 0–2 objects were read from `w19-maze-t-symmetric/persist/` (not listed in the brief).
  This was a zero-fit read for the cross-seed check, hash-verified like the rest; LOG.md records the
  choice.
- The seed-5 value-row per-tuple table and the Q1 trace table above were transcribed from the
  JSONs / printouts. The source files are authoritative.
