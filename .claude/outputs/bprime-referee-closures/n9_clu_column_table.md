# The **CLU column at n = 9** (referee missing-experiment 1 — MF-1's clean upgrade)

⭐ **What this closes:** the paper's visible **n-asymmetry** (rival arms at n = 9, the CLU at n = 3) and
**MF-1** (the CLU row printed `✅ RESCUED` while its own full read sat below its blank). ⛔ **The n = 3
"✅ rescued" is a never-quote** (Hub ruling, C2W5 review); the verdict below replaces it.

⛔ **Nothing was re-measured.** `run_rivals_cell` runs the shipped CLU write/read path on **every**
(family, arm, seed) cell regardless of the rival tuning grid, so the F3 rider's seeds 3–8 runs already
contained `clu_reproduction` at those seeds — rider 1 simply excluded the CLU from its re-aggregation
("the CLU column is BANKED, never re-derived"). This is that aggregation, run with **rider 1's own rule**.

**Family:** `aggregate@base` — the sole reader-discrimination family (`S = 0.5068`).
**Convention:** sample sd (ddof = 1); SE = sd/√n; **n = 9** (seeds 0–8). Metric `neg_mae`
(higher = better). `full`/`launder`/`dividend` come out of the shipped
`exp_bprime_rivals.audit_table`; every margin/lift is **paired per seed**, then averaged.

⭐ **Fidelity:** seeds 0/1/2 reproduce the banked column **digit-for-digit**
(`full = [-0.682608, -0.384693, -0.511032]`, `launder = [-0.496261, -0.413103, -0.432255]`, `blank = [-0.438906, -0.404201, -0.423079]`), and the CLU cells are
**bit-identical across both code paths** on all 9 seeds (max |Δ| = 0.0 on `full`, `launder`, `blank`,
`same_keys_null`) — the CLU path never touches the rival grid.

---

## A. The column (n = 9), beside the banked n = 3 the paper currently prints

| quantity | **n = 9 (seeds 0–8)** | SE multiple | n = 3 (banked, currently in §4.1.1) | moves |
|---|---|---|---|---|
| `full` | **-0.4370 ± 0.0417** | 10.47 SE | −0.5261 ± 0.0863 | **+0.089 better** |
| `launder` (its own raw `(key, payload)` table) | **-0.3810 ± 0.0345** | 11.05 SE | −0.4472 | +0.066 better |
| `blank` (empty store, same read) | **-0.3906 ± 0.0124** | 31.57 SE | −0.4221 | +0.032 better |
| `same-keys null` | **-0.6512 ± 0.0383** | 17.01 SE | −0.8175 | +0.166 better |
| **dividend** = `full − launder` (paired) | **-0.0561 ± 0.0315** | **1.78 SE** | −0.0789 (no SE printed) | −0.023 |
| **+0 B margin** (arg-max +0 B reader, paired) | **-0.2897 ± 0.0328** | **8.84 SE** | −0.3180 ± 0.0804 | +0.028 |
| **raw-table margin** (+0 B set ∪ arg-min launder) | **-0.2897 ± 0.0328** | **8.84 SE** | — (not printed: "its own table is already raw") | — |
| **lift over own blank** = `full − blank` (paired) | **-0.0465 ± 0.0406** | 1.14 SE | −0.1040 (implied, never stated) | +0.057 |
| `full − same-keys null` (paired) | **+0.2141 ± 0.0443** | 4.83 SE | +0.2914 (implied) | −0.077 |

## B. ⛔ THE RESCUE-GATE VERDICT AT n = 9

> **lift = -0.0465 +/- 0.0406 (n=9); 2 SE = 0.0813 => NOT RESCUED**

**⛔ NOT RESCUED.** Under the paper's own gate (B.5: *an arm within 2 SE of its own blank-store control is
NOT RESCUED, and no margin against it is quotable*) the CLU's written store is **statistically
indistinguishable from its own empty store** on this family: lift
-0.0465 ± 0.0406 (|t| =
1.14, n = 9), i.e. the same
category as **`ttt_mlp`** — and the point estimate is on the **wrong side of zero**. Same verdict under
both code paths (-0.0465 ± 0.0406, bit-identical).

⭐ **This is a stronger, cleaner statement than the n = 3 one it replaces, and it is *consistent with the
paper's thesis*:** the written content does not lift the read above an empty store on the one family that
survives protocol validation. It also removes the referee's third quotable sentence (MF-1) at the root:
the CLU no longer carries a ✅ while rival arms are disqualified on the same test.

## C. Per-seed, so the aggregation is auditable

| seed | `full` | `launder` | `blank` | `same-keys null` | dividend | +0 B margin | raw margin | lift |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.6826 | -0.4963 | -0.4389 | -0.8175 | -0.1863 | -0.4676 | -0.4676 | -0.2437 |
| 1 | -0.3847 | -0.4131 | -0.4042 | -0.7142 | +0.0284 | -0.1921 | -0.1921 | +0.0195 |
| 2 | -0.5110 | -0.4323 | -0.4231 | -0.6261 | -0.0788 | -0.2941 | -0.2941 | -0.0880 |
| 3 | -0.4216 | -0.2505 | -0.3951 | -0.4450 | -0.1711 | -0.3819 | -0.3819 | -0.0264 |
| 4 | -0.4584 | -0.5150 | -0.3983 | -0.5794 | +0.0566 | -0.2800 | -0.2800 | -0.0601 |
| 5 | -0.3709 | -0.3798 | -0.3442 | -0.6378 | +0.0089 | -0.2107 | -0.2107 | -0.0267 |
| 6 | -0.2817 | -0.3322 | -0.4061 | -0.7446 | +0.0504 | -0.1648 | -0.1648 | +0.1244 |
| 7 | -0.5262 | -0.4041 | -0.3202 | -0.7392 | -0.1221 | -0.3586 | -0.3586 | -0.2059 |
| 8 | -0.2964 | -0.2057 | -0.3852 | -0.5567 | -0.0907 | -0.2576 | -0.2576 | +0.0888 |

**The +0 B reader set, at n = 9** (all three are readers of the store's own table at **+0 extra bytes**):

| reader | n = 9 mean ± SE | selected as arg-max in |
|---|---|---|
| `knn2_idw_+0B` | -0.1508 ± 0.0231 | 6 of 9 seeds |
| `knn2_mean_+0B` | -0.1552 ± 0.0245 | 3 of 9 seeds |
| `raw_table_mean_+0B` | -0.3442 ± 0.0153 | 0 of 9 seeds |

⚠ **Two conventions, both reported, they agree.** Primary (the rivals' own per-cell rule: **per-seed
arg-max** over the exclusive +0 B set) = **-0.2897 ± 0.0328**. The banked convention (one reader
fixed column-wide, here `knn2_idw_+0B`) = **-0.2862 ± 0.0317**. Difference
**-0.0035** — no claim turns on the choice.

⭐ **The raw-table margin equals the +0 B margin on 9 of 9 seeds** (float-identical): for the CLU the
projected-vs-raw distinction genuinely does not arise, because the arg-min launder (≈ −0.38) never beats
the `knn2` readers (≈ −0.15) on any seed. The draft's parenthetical *"(its own table is already raw)"* is
now **measured**, not asserted.

## D. ⛔ The byte ledger is NOT seed-constant at n = 9 (a NEW finding — P9 refuted)

| quantity | 8 of 9 seeds | seed 8 | modal |
|---|---|---|---|
| `full_bytes` | **5456** | **5472** | 5456 |
| `launder_bytes` | **100** | **120** | 100 |
| **ratio** | **54.56×** | **45.60×** | 54.56× |
| items admitted (`n_live`) | 5 | **6** | 5 |

`identity_T1` (`ratio = [A(D+2)+d]/(d+m)`) holds **on every one of the 9 seeds** — the ledger is exact,
it is simply a function of how many items the store's own admission gate let in, and seed 8 admitted
**6 of 8** offered items rather than 5. This is the CLU-side analogue of rider 1's **R4** (the TTT arms'
seed-dependent state bytes): ⛔ **`5456 B / 100 B / 54.56×` must be labelled as the modal (8 of 9 seeds)
configuration**, not as *the* n = 9 value. The two-sided learned-initial-state split moves with it:
F1 = 5376 B (`V_theta_init`, constant), F2 = **5200 B** (8 seeds) / **5472 B** (seed 8).

## E. Admissible-cell coverage at all 9 seeds (the draft tabulates only seeds 0–2)

| seed | admissible / attempted | fraction | items admitted / offered |
|---|---|---|---|
| 0 | 58/72 | 0.806 | 5/8 |
| 1 | 66/80 | 0.825 | 5/8 |
| 2 | 55/80 | 0.688 | 5/8 |
| 3 | 45/80 | 0.562 | 5/8 |
| 4 | 48/72 | 0.667 | 5/8 |
| 5 | 56/80 | 0.700 | 5/8 |
| 6 | 64/80 | 0.800 | 5/8 |
| 7 | 60/80 | 0.750 | 5/8 |
| 8 | 51/112 | 0.455 | 6/8 |

Mean admissible fraction **0.695 ± 0.041** (n = 9); store
admission **0.639 ± 0.014** of offered items.

