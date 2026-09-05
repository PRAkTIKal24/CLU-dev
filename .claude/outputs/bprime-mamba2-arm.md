# bprime-mamba2-arm — experiment-engineer report (C2W5, Head-funded ruling 5)

**Task + acceptance criterion:** build B′'s **Mamba-2 (SSD)** rival arm at matched state bytes and land
its full audit row — every column at **n = 9**, aggregated through the **shipped** `audit_table` rule,
rescue-gate verdict at n = 9, ledger identities green, tests green. **Status: done.**

## ⛔ DOWNSTREAM RECONCILIATION LIST (protocol §5 corollary — owner needed, in my first 10 lines)
1. ⭐ **`draft-r2/r3` §4.1, §4.1.1, §4.2, App. J: "Mamba-2 — NOT-RUN, outside the ruled arm set" is now
   FALSE.** The row exists at n = 9. **The headline's arm count changes 0-of-5 → 0-of-6 and its margin
   range changes `−0.2592 … −0.4602` → `−0.2563 … −0.4602`** (the minimum "≥ 4.4 SE" is unchanged —
   it is still `ttt_linear`). **Owner:** `bprime-draft` (r3).
2. ⭐ **The draft's stated n = 9 gap is CLOSED as a by-product.** §4.1 says the projected-launder and
   same-keys-null columns "were aggregated at the first pass's three seeds only" (App. I.1c). This run
   produced them at **n = 9 for all six arms** (§3.2 below). **Owner:** `bprime-draft` (r3) — decide
   whether to promote them out of I.1c.
3. ⚠ **A THIRD code path disagrees with the draft's "RESCUED under both code paths".** Under
   **held-out (`f3_val`) selection at n = 9**, `deltanet` (+0.077 ± 0.045) and `gdn2` (+0.669 ± 0.339)
   fall **below** the 2-SE rescue bar; only `gdn` and `mamba2` clear it under all three selections.
   The draft's §4.1.1 sentence *"RESCUED: deltanet, gdn, gdn2 — under both code paths"* needs the
   third path stated. **Owner:** `bprime-draft` + Hub (which selection rule is the standing one).
4. **Citation provenance:** the Mamba-2 venue/year/id and the reference-implementation state accounting
   come from `rival-recon.md` §1.4's pinned record, **not re-verified this session** (this agent has no
   web tool). Per `bprime-cite-check`'s pattern: **double-source before print.** **Owner:** `web-scout`.

---

# 0. ⭐ DIAL DECLARATION (echoed before the first result, protocol §7)
- **Dial:** none — tier-i audit coverage (a rival row, not a CLU claim).
- **Laundering control:** the full B′ column set — projected (P5) launder **and** the raw +0 B table
  (the pre-registered R4/R5 distinction), blank store, same-keys null, rescue gate, two-sided ledger.
- **Falsifies:** nothing of ours. ⛔ Selection on the eval split would invalidate the row — **it never
  happened**: both selections read only auxiliary streams from seeds `+101/+102` (fit) and `+103`
  (held-out), never the eval cell's stream.
- **Does NOT falsify:** Mamba-2 beating the raw table would have been the audit's first positive rival
  row, to be reported as such. **It did not** — it loses by **−0.2563 ± 0.0416 (6.2 SE, negative in
  9/9 seeds)**, which is the metric-native-ceiling theorem, not news.

---

# 1. ⭐ THE ROW (the paper's table line) — `aggregate@base`, seeds 0–8, full F3 grid

| arm | `d_head` | F1 param B | F2 state B | own table B | **full** | **+0 B margin** | ⭐ **raw-metric +0 B margin** | lift over own blank | **RESCUED?** |
|---|---|---|---|---|---|---|---|---|---|
| **mamba2 (SSD)** | **36** | **8380** | **5184** | **5184** | **−0.4036 ± 0.0329** | **+0.0047 ± 0.0519** | **−0.2563 ± 0.0416** | **+1.421 ± 0.463** | ✅ **RESCUED** |

**Beside the five incumbents (this same run, same code, `f3` selection, n = 9) — reproduced
digit-for-digit from `bprime-rivals-f3`/`draft-r2` §4.1.1, not re-derived:**

| arm | full | +0 B margin | raw margin | \|raw\|/SE | lift over blank | RESCUED |
|---|---|---|---|---|---|---|
| ttt_linear | −0.6075 ± 0.1096 | −0.2213 ± 0.1062 | −0.4602 ± 0.1038 | 4.43 | +0.093 ± 0.134 | ⛔ |
| ttt_mlp | −0.5898 ± 0.0731 | −0.2095 ± 0.0683 | −0.4425 ± 0.0869 | 5.09 | −0.071 ± 0.090 | ⛔ |
| deltanet | −0.4205 ± 0.0299 | −0.0172 ± 0.0263 | −0.2732 ± 0.0395 | 6.92 | +0.294 ± 0.077 | ✅ |
| gdn | −0.4073 ± 0.0120 | −0.0102 ± 0.0229 | −0.2600 ± 0.0278 | 9.35 | +0.880 ± 0.227 | ✅ |
| **gdn2** | −0.4065 ± 0.0178 | +0.0473 ± 0.0277 | −0.2592 ± 0.0292 | 8.87 | +1.025 ± 0.329 | ✅ |
| ⭐ **mamba2** | **−0.4036 ± 0.0329** | **+0.0047 ± 0.0519** | **−0.2563 ± 0.0416** | **6.17** | **+1.421 ± 0.463** | ✅ |

## 1.1 The finding, in one sentence
> ⭐ **Mamba-2 is the best `full` score in the audit (−0.4036) and it still loses to a zero-extra-byte
> reader of a raw table holding the same 5184 bytes, by −0.2563 ± 0.0416 — 6.2 SE below zero and
> negative in 9 of 9 seeds. The headline becomes "0 of 6 rival arms", and the SSM family, which the
> survey named and no one had measured, behaves exactly like the delta-rule family it is the
> erase-free degenerate case of.**

⚠ **Two honest qualifiers, first-class.**
- Mamba-2's `full` beats `gdn2`'s by **0.0029 ± 0.037** — that is a **tie, not a win**, and I quote no
  ordering among the four delta/SSD arms.
- Its **+0 B margin (+0.0047 ± 0.0519) is a tie with zero**, not a win. It is *not* a positive rival
  row: the load-bearing column is the raw one, which is decisively negative.

## 1.2 ⛔ The `overload` byte-frontier row (labelled at every appearance)
> ⛔ **BYTE-FRONTIER COLUMN — not a dividend family, never a headline; declared secondary reading
> `S_excl = 0.6500`.**

`overload@load1x_shipped`, decode, n = 9, chance = 0.1667:

| selection | full | launder | +0 B margin | raw margin | blank | lift | **RESCUED?** |
|---|---|---|---|---|---|---|---|
| `f3` | 0.2083 ± 0.0354 | 0.1898 | +0.0046 ± 0.0419 | −0.7917 ± 0.0354 | 0.1806 | +0.028 ± 0.035 | ⛔ **NO** |
| `f3_lite_control` | 0.1944 ± 0.0354 | 0.2083 | +0.0093 ± 0.0353 | −0.8056 ± 0.0354 | 0.1852 | +0.009 ± 0.037 | ⛔ **NO** |
| `f3_val` (held-out) | 0.1806 ± 0.0380 | 0.1991 | −0.0278 ± 0.0481 | −0.8194 ± 0.0380 | 0.1667 | +0.014 ± 0.045 | ⛔ **NO** |

⛔ **NOT RESCUED in every configuration — within noise of its own blank store and of chance.** Exactly
as all five incumbents were on this family. **I therefore draw no curve, and I quote no margin against
Mamba-2 here — including the CLU's banked `decode 0.972`.** Ledger: 5184 B state / 5184 B table /
8380 B params, table lossless (7 stream tokens, 18 rows affordable).

---

# 2. ⭐ THE PREREG SCORECARD (registered → measured → verdict)
`.claude/outputs/bprime-mamba2-arm/PREREG.md`, filed **before `run_rivals_cell` was ever invoked on the
arm**. Its only inputs were the banked n = 9 rival rows, Mamba-2's equations, and integer arithmetic.

| # | registered | band | measured (n = 9, `f3`) | verdict |
|---|---|---|---|---|
| **M1** | full **−0.42** | [−0.55, −0.33] | **−0.4036 ± 0.0329** | ✅ **IN BAND** |
| **M2** | raw margin **−0.26** | [−0.40, −0.15] | **−0.2563 ± 0.0416** | ✅✅ **IN BAND, 0.004 from the point** |
| **M2b** | raw margin ≤ 0 by > 2 SE | — | **6.17 SE below zero; negative in 9/9 seeds** | ✅ |
| **M3** | +0 B margin **+0.06** | [−0.05, +0.20] | **+0.0047 ± 0.0519** | ✅ **IN BAND** (near its lower edge) |
| **M3b** | M3 **>** `gdn`'s (−0.0102) | — | **+0.0047 > −0.0102** | ◐ **direction CONFIRMED, magnitude wrong** — the gap is 0.015, ≪ 1 SE. The no-normalisation mechanism is *not* refuted, but it is **not demonstrated** either |
| **M4** | launder **−1.0** | [−2.0, −0.40] | **−0.7612 ± 0.1316** | ✅ **IN BAND** |
| **M5** | dividend **+0.60** | [0.00, +1.60] | **+0.3575 ± 0.1451** | ✅ **IN BAND** |
| **M6** | same-keys null **−1.0** | [−2.2, −0.40] | **−0.7739** | ✅ **IN BAND** |
| **M7** | blank **−1.1** | [−2.4, −0.40] | **−1.8249** | ✅ **IN BAND** |
| **M8** | lift **+0.70** | [+0.10, +1.60] | **+1.4212 ± 0.4632** | ✅ **IN BAND** |
| **M9** | **RESCUED** at n = 9 | — | **RESCUED** (and lift > 0 in **9/9** seeds) | ✅ |
| **M10** | P5-vs-raw gap **+0.55** | [+0.15, +1.20] | **+0.6139 ± 0.1386** | ✅ **IN BAND** |
| **M11** | frontier: within noise of blank, decode ∈ [0.00, 0.35] | — | **0.181–0.208, NOT RESCUED in 3/3 selections** | ✅ |
| **M12** | no verdict differs between `f3` and `f3_val` | — | rescue ✅ same, raw-margin sign ✅ same, **+0 B sign flips** (+0.0047 → −0.0045) | ◐ **PARTIAL** — the flip is between two ties (both < 0.15 SE from 0); no *verdict* moves |
| **ledger** | `d = 36`, state **1296 f / 5184 B**, table **18 rows / 5184 B**, ratio **1.000**, params **2095 f / 8380 B**, lossless | exact integers | **every integer exact** | ✅✅ **EXACT, 6 of 6** |
| **F1–F6** | faithfulness pass/fail (chunk ≡ sequential; chunk inert; dual ≡ recurrent; no-decay ≡ linear attention; mask no-op; verdict weaker than the delta arms') | — | **6 of 6 pass** (unit tests) | ✅ |

**Score: 11 confirmed (7 of them in-band, 6 ledger integers exact, 6 faithfulness checks exact) ·
2 partial (M3b, M12) · 0 refuted.** ⚠ **I flag the obvious risk in my own favour:** a scorecard with
zero refutations is weaker evidence than one with a refutation, because the bands were mine and the
delta-arm cluster was visible when I drew them. **The two sharp, non-inherited predictions are M2b
(sign + significance) and the six ledger integers**; M1/M4/M6/M7 are "it lands in the cluster I copied
the band from" and should be read as such.

---

# 3. VERIFICATION — the things that make the row trustworthy

## 3.1 ⭐ The five incumbent arms reproduce BIT-IDENTICALLY (the append-only regression check)
Adding `mamba2` to `RIVALS` re-indexes nothing only if it is appended **last** (the per-rival fit key is
`jax.random.PRNGKey(seed*1000 + 7*(RIVALS.index(name)+1))`). Verified two ways:

| check | observed |
|---|---|
| per-seed, per-arm `arms` dicts at `aggregate/base@s0` vs the banked `bprime-rivals-f3/run400` artifact | ⭐ **bit-identical (`abs diff < 1e-12`) for all 5 arms, all 7 arm-columns**; chosen configs identical too |
| n = 3 (seeds 0–2) `f3` means vs the banked report's `full` / raw-margin column | −0.6332 / −0.5052 / −0.4478 / −0.4104 / −0.4350 and −0.4251 / −0.2971 / −0.2396 / −0.2022 / −0.2269 — **all Δ ≤ 5e-5 (the report's own rounding)** |
| n = 9 `f3` means vs `draft-r2` §4.1.1 | **every printed digit reproduces**, including the lift column (+0.093 ± 0.134 / −0.071 ± 0.090 / +0.294 ± 0.077 / +0.880 ± 0.227 / +1.025 ± 0.329) |
| CLU fidelity (banked, never re-derived) | `full` **−0.682608 / −0.384693 / −0.511032**, `launder` **−0.496261 / −0.413103 / −0.432255** — digit-for-digit; `overload` **1.000000 / 0.958333 / 0.958333**, launder **1.0** |
| CLU ledger identity (D7, integers) | green on **all 9** aggregate cells and all 9 overload cells: **5456 B / 100 B / 54.56×** and **57384 B / 120 B** |
| identical-φ invariant | enforced in code on every cell; 9 distinct `phi_id`s for 9 seeds, one per cell across all 13 arm rows (`aggregate@s0 = 09dc0ee5726e6a8d`, the same id `bprime-rivals-f3` reports) |
| degenerate / errored cells | **0 of 18** |

## 3.2 The full n = 9 column set (the draft's stated gap, closed as a by-product)
`draft-r2` §4.1 marks the projected launder and same-keys-null columns `have (n = 3)`. This run
aggregates **every** column at n = 9 through the shipped `audit_table`:

| arm | launder (n=9) | same-keys null (n=9) | dividend (n=9) | blank (n=9) | P5-vs-raw gap (n=9) |
|---|---|---|---|---|---|
| ttt_linear | −0.4235 ± 0.0145 | −0.4012 | −0.1840 ± 0.1069 | −0.7008 | +0.2762 ± 0.0285 |
| ttt_mlp | −0.4104 ± 0.0174 | −0.3903 | −0.1794 ± 0.0748 | −0.5189 | +0.2631 ± 0.0307 |
| deltanet | −0.5720 ± 0.0653 | −0.6379 | +0.1515 ± 0.0600 | −0.7147 | +0.4246 ± 0.0672 |
| gdn | −1.0033 ± 0.0952 | −0.9715 | +0.5960 ± 0.0933 | −1.2869 | +0.8560 ± 0.0907 |
| gdn2 | −1.0889 ± 0.0815 | −1.1503 | +0.6824 ± 0.0756 | −1.4319 | +0.9416 ± 0.0913 |
| ⭐ **mamba2** | **−0.7612 ± 0.1316** | **−0.7739** | **+0.3575 ± 0.1451** | **−1.8249** | **+0.6139 ± 0.1386** |

⚠ The n = 9 P5-vs-raw gaps differ slightly from `bprime-rivals-f3`'s handover item 4 (`0.276 / 0.263 /
0.425 / 0.856 / 0.942` there — I measure `0.2762 / 0.2631 / 0.4246 / 0.8560 / 0.9416`): **the same
numbers to 4 dp.** The §4 finding survives on **6 of 6** arms, all > 2 SE.

## 3.3 Robustness of the two verdicts that carry the row (per-seed sign tests, not just means)
The rescue gate's known weakness (`bprime-rivals-f3` §9.1) is that its blank control is a single init
draw whose variance can exceed the lift it gates — and **Mamba-2 has the most negative blank in the
audit (−1.82; one seed at −4.62)**, so its `RESCUED` verdict deserves more than a mean.

| quantity | mean ± SE | **per-seed sign test** |
|---|---|---|
| lift over own blank | +1.421 ± 0.463 | **positive in 9/9 seeds** (median +1.353), p = 2/2⁹ ≈ 0.004 |
| raw-table margin | −0.256 ± 0.042 | **negative in 9/9 seeds** (median −0.2165), p ≈ 0.004 |

Both verdicts are **unanimous across seeds**; neither depends on the outlier blank. (For contrast:
`ttt_linear` 6/9 and `ttt_mlp` 4/9 on lift — consistent with their UNSTABLE / NOT-RESCUED status.)

## 3.4 ⭐ The arm is NOT hobbled — four independent pieces of evidence
The Head funded this row specifically to avoid *"you hobbled Mamba-2"*. Answers, all measured:

1. **The full F3 grid, same standard as every other arm:** `lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3,
   1e-2} × wd ∈ {0, 0.1}`, 12 points per seed, 108 fits, best-of-grid **on the fit split** (primary)
   and on a **held-out** stream (declared secondary). Both reported; neither ever sees the eval split.
2. ⭐ **Its fit-split optimum is INTERIOR to the grid** (arg-min at `lr = 1e-3`, with `1e-4` worse by
   0.097 and `1e-2` flat) — the arm is **not lr-limited at either edge**, which is the sharpest
   available refutation of "under-tuned". The incumbents' arg-mins all sit at the grid's *top* edge.
3. ⭐ **Mamba-2 is the first arm in this rig for which F3's widened grid is not decorative.**
   `bprime-rivals-f3` measured **0 of 45** incumbent cells selecting `lr < 1e-3`. Adding this arm:
   **1 of 54** overall — and that one cell is Mamba-2's. Under held-out selection it is **7 of 9**
   Mamba-2 cells (and its held-out surface's arg-min is at `lr = 1e-4`, a point only F3 contains).
   `wd = 0.1` is selected in **5 of 9** Mamba-2 cells.
4. **The declared block-level ablation** (§3.5) — Mamba-2's `D` skip and `z` gate, dropped by default
   exactly as every other arm's block-level parts are, run through the **same** outer loop and scorer.

### The tuning surfaces (so "best-of-grid" is auditable), 9-seed means, `*` = arg-min
```
mamba2   fit  wd=0.0   0.3657  0.3082  0.2845 *0.2684  0.2685  0.2686
              wd=0.1   0.3658  0.3080  0.2841 *0.2684  0.2684  0.2686
         val  wd=0.0  *0.4145  0.4393  0.4478  0.4543  0.4607  0.4543
              wd=0.1  *0.4143  0.4399  0.4480  0.4547  0.4614  0.4554
gdn2     fit  wd=0.0   0.3770  0.2632  0.2517  0.2433 *0.2417  0.2417
         val  wd=0.0   0.4032 *0.3980  0.4002  0.4033  0.4085  0.4128
                       1e-4    3.16e-4 5e-4    1e-3    3.16e-3 1e-2
```
⭐ The **fit** and **held-out** surfaces disagree about the direction of the optimum for this arm
(1e-3 vs 1e-4) — a concrete instance of `bprime-rivals-f3`'s §2 finding that selecting on the fit
objective makes F3's own added points unselectable.

## 3.5 The declared block-level ablation (`use_D`, `gate_z`) — the "you dropped half the block" answer
Both configurations run through the **same** `fit_grid` → `select_best` → `rival_arms` → `score` path
(`make_rival(..., **arm_kwargs)`), same F3 grid, same 9 seeds, same fit streams. `D` and `W_z` are
**parameters and already counted in F1**, so the ablation costs **zero extra state bytes** (5184 B / 8380 B
in both). Script + artifact: `.claude/outputs/bprime-mamba2-arm/block_ablation.{py,json,log}`.

| config | fit-split loss | **full** | **raw-table margin** | blank |
|---|---|---|---|---|
| **minimal (the audited arm)** | 0.2684 ± 0.0198 | **−0.4036 ± 0.0329** | **−0.2563 ± 0.0416** | −1.8249 ± 0.4607 |
| **+block (`use_D`, `gate_z`)** | **0.1721 ± 0.0068** (−36%) | **−0.5985 ± 0.0860** | **−0.4512 ± 0.0996** | −2.6789 ± 0.6035 |

⭐ **The attack is answered by measurement, in the direction that closes it: restoring Mamba-2's
block-level parts fits 36% better on the fit split and scores WORSE on the eval metric (−0.195 of `full`,
≈ 2.3 SE), so the minimal configuration is the arm's best configuration on the audited metric.** This is
the same fit→eval generalisation gap the audit already measured for TTT-MLP (a 64% fit cut buying nothing),
now measured on the SSD arm. ⚠ And the `+block` reading is if anything *generous*: the `D` skip gives its
`full` a query-dependent path the byte-matched table structurally cannot have — and it still loses by more.
✅ **Cross-check:** the `minimal` column reproduces the harness's row exactly (−0.4036 / −0.2563 / −1.8249),
i.e. the standalone script and `run_rivals_cell` agree digit-for-digit.

## 3.6 ### The labelled frontier sweep, `overload@load1x_shipped`, n = 9, chance = 0.1667
⛔ **BYTE-FRONTIER COLUMN — not a dividend family, never a headline.**

| `d_head` | state B | table rows | lossless | **full** | own table | blank | lift | RESCUED |
|---|---|---|---|---|---|---|---|---|
| 2 | 16 | 1 | no | 0.1528 ± 0.0354 | 0.1667 | 0.1111 | +0.042 ± 0.033 | ⛔ |
| 4 | 64 | 2 | no | 0.1620 ± 0.0383 | 0.1852 | 0.1806 | −0.019 ± 0.044 | ⛔ |
| 8 | 256 | 4 | no | 0.1296 ± 0.0306 | 0.0741 | 0.1250 | +0.005 ± 0.025 | ⛔ |
| 16 | 1024 | 8 | **yes** | 0.1759 ± 0.0359 | 0.1435 | 0.1343 | +0.042 ± 0.037 | ⛔ |
| 36 | 5184 | 18 | yes | 0.2037 ± 0.0306 | 0.1898 | 0.1667 | +0.037 ± 0.036 | ⛔ |
| **CLU** (banked, ⛔ not re-measured) | 57384 | — | — | **0.972 → 0.097 as the ratio falls 478× → 2.28×** | 1.000 | 0.1667 | — | — |

⛔ **NOT RESCUED at every one of the five head widths** — every point is within noise of its own blank
store and of chance. **I draw no curve and I quote no margin against Mamba-2 here, including against the
CLU's banked 0.972.** This reproduces, on a sixth arm, exactly what C2W4 found for `ttt_linear` and `gdn2`
on this family, and it is a NOT-RESCUED verdict, not a result about the rivals.

## 3.7 Tests
| check | command | observed |
|---|---|---|
| **full suite** | `PYTHONPATH=. pytest tests/ -q --no-cov` | ✅ **`1261 passed, 0 failed, 31 warnings in 1517.88s`** (1249 pre-existing + 12 new; ⚠ run concurrently with another spoke's suite in a different worktree, hence the wall clock) |
| the arm's own tests | `pytest tests/test_bprime_rivals.py tests/test_rivals_ledger.py -q` | **85 passed in 23.19 s** |
| mamba2-only | `pytest tests/test_bprime_rivals.py -k mamba2 -q` | **22 passed** (11 new + 11 parametrized incumbents extended to the new arm) |
| lint | `ruff check chlu/ tests/` | **All checks passed** |
| smoke | `python -m chlu.experiments.exp_bprime_rivals --quick --seeds 0 --families aggregate --rivals mamba2 gdn2 --no-frontier --grid f3 --steps 12` | exit 0, 29 s |
| the runs | `--families aggregate --no-frontier --grid f3 --seeds 0..8` and `--families overload --rivals mamba2 --frontier-rivals mamba2 --grid f3 --seeds 0..8` | **18/18 audit cells + 9/9 frontier cells, 0 degenerate, 0 errors** |
| D7 ledger identity + identical-φ | in-code asserts, every cell | green **18/18** |

⚠ **One pre-existing test changed behaviour and I flipped it deliberately:**
`tests/test_rivals_ledger.py::test_head_width_rejects_unknown_kind` used `"mamba2"` as its example of an
*unknown* sizing kind. It is now a real arm, so the example moved to `"titans"` and a new test asserts the
new law (`head_width_for_budget("mamba2", 1364) == head_width_for_budget("delta", 1364) == 36`). The
docstring records why, so the flip is auditable rather than silent.

---

# 4. WHAT WAS BUILT, AND THE DECISIONS A REFEREE WILL ATTACK

## 4.1 The equations (Dao & Gu, ICML 2024, arXiv:2405.21060)
At `n_head = ngroups = 1`, in this rig's notation:
```
Delta_t = softplus(w_Delta . x_t + Delta_bias)        (Mamba-1 §3.2 selection, carried over)
B_t     = theta_K x_t ,  C_t = theta_Q x_t ,  v_t = silu(theta_V x_t)
a_t     = exp(Delta_t A),  A = -exp(A_log) < 0        (SSD: A_t = a_t I, SCALAR x identity)
h_t     = a_t h_{t-1} + B_t (Delta_t v_t)^T           h in R^{N x P}
o_q     = h_T^T C_q                                   then the shared head theta_O
```
Implemented **three ways, all asserted equal**: the **chunked SSD block pass** (§6, the shipped path,
chunk = **16**, matched to the rig's mini-batch; reference default is 256), the naive sequential
recurrence (reference), and the **quadratic/dual read** `o_q = Σ_j γ_j (C_q·B_j) Δ_j v_j`. That third
identity *is* state-space duality, and it is verified in a unit test rather than cited.
Reference-implementation init: `A ~ U(1, 16)`, `Δ ~ exp(U(log 1e-3, log 1e-1))` inverse-softplused.

⭐ **Mamba-2 is the delta arms' erase-free degenerate case, and the rival authors say so, not us:**
Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025) presents *"Mamba2 as `S_t = α_t S_{t−1} + v_t
k_tᵀ`"*. That is exactly what this module computes — which is why the row's cleanest reading is **what
the delta-erase term buys at byte-identical state: 0.003 ± 0.037 of `full` (i.e. nothing measurable)**.

## 4.2 Sizing: matched **state** bytes; params reported, not matched
The rig's convention is **iso-state** (budget = the CLU's banked `aggregate@base` **1364 float32 =
5456 B**), and I did not change it. With `d_state = head_dim = d` (declared) the SSM state is `d²`, so
the sizing law is *arithmetically identical* to the delta arms' and the arm lands on **byte-identical
state (5184 B)** — the cleanest possible isolation of the update rule.

| | F1 params | F2 state | own table | state/table | state/param |
|---|---|---|---|---|---|
| mamba2 (`d=36`) | **8380 B** = 2095 f (`θ_K,θ_Q,θ_V` 540 · `θ_O` 36 · **`S₀` 1296** · `w_Δ` 5 · `Δ_bias` 1 · `A_log` 1 · `D` 36 unused · `W_z` 180 unused) | **5184 B** (measured moved: **1296/1296 floats**, 9/9 seeds) | 5184 B, **18 rows, lossless** | **1.000** | 0.619 |
| delta arms (`d=36`) | 9956 B | 5184 B | 5184 B | 1.000 | 0.521 |
| CLU (`aggregate@base`) | 5376 B | 5200 B | **100 B** | ⛔ **52.0×** | 0.967 |

⚠ **Params are NOT matched** — no arm in this rig is param-matched, and Mamba-2's F1 (8380 B) is
**lower** than the delta arms' (9956 B), i.e. the asymmetry runs **in Mamba-2's favour** on F1. Stated
rather than left for a referee. Both sides of the ledger are printed, per D3.3.

## 4.3 The three declared deviations, and which way each one cuts

| deviation | why | direction |
|---|---|---|
| **no short conv branch** (the reference's `conv_state`, `(d_inner + 2·ngroups·d_state)·d_conv` floats, is excluded) | the same minimality caption every arm carries (TTT: "no convolution branch"; delta: "no short convolution") — protocol uniformity *is* the deliverable | ⭐ **in the rival's favour**: at a fixed byte budget every byte goes to the SSM state instead of 9/16 of them to a 4-tap window |
| **no `D` skip / `z` gate / gated-RMSNorm by default** | block-level, not update-rule; TTT's arm likewise has "no gating over a residual stream". Also, a `D` skip gives `full` a query path the byte-matched table structurally cannot have | measured in §3.5, both directions reported |
| **`d_state = head_dim`** | the rig has one width knob; this choice is what makes the state byte-identical to the delta arms | neutral (it spends the whole budget) |
| **SSD chunk = 16, not the reference's 256** | matched to the rig's ~7–19-token streams | ⭐ **provably neutral** — chunking is an exact re-association; asserted at `Q ∈ {1,2,3,7,16,256}` |
| **delete rows skipped** (inherited, all rivals) | no rival family has a deletion verb | in the rivals' favour (inherited, unchanged) |

## 4.4 The metric-native verdict — **weaker** than the delta arms', by construction
`"metric-native (unnormalised)"`. The read is a **dot-product kernel smoother with an exponential
recency weighting**, so criterion 4 closes in the same sense as DeltaNet Eq. 5. ⚠ **But Mamba-2 does
not L2-normalise its `B`/`C` paths** (GDN-2 §3.5 does), so in its own key space `arg-min ‖q−k‖` and
`arg-max q·k` **do not coincide** — the key-norm term survives (asserted in a test: GDN-2's key norms
are 1.000 ± 1e-3, Mamba-2's have sd > 1e-3). That is *why* its PREREG predicted a higher +0 B margin
than `gdn`'s: the P5 projected table is read by a worse-matched reader for this arm.
**M3b: direction confirmed (+0.0047 > −0.0102), magnitude ≪ 1 SE — not demonstrated.**

---

# 5. FLAG-PROVENANCE TABLE (mandatory, protocol §5)

| item | value |
|---|---|
| commits (results produced at) | **`5a71105`** (code identical to `da715d3` for every measured run), branch `agent/experiment-engineer/bprime-mamba2-arm`, base local `main @ eaecc91` |
| worktree / venv | `../CHLU-mamba2`; **main venv reused** (`/Users/user/Desktop/CHLU/.venv`), **no `uv sync`** (w6 trap avoided) |
| **JAX / Equinox / Optax / NumPy** | **0.9.0 / 0.13.4 / 0.2.6 / 2.4.1** — printed this session; identical to C2W4's and the f3 rider's |
| seeds | **0–8 (n = 9) on every column, from the start** (§A18.1: no n = 3 verdict exists to retract). SE = sample sd (ddof=1)/√9 |
| fit-stream seeds (F2a guard) | `seed+101`, `seed+102` (training) · `seed+103` (**held-out**, secondary selection only) — ⛔ never the eval stream |
| families | `aggregate@base` (the sole dividend family) · `overload@load1x_shipped` (⛔ **labelled byte-frontier column only**). ⛔ `recency` / `manifold` **NOT RUN** (protocol-invalid, FB4) |
| arms | **all six** — `mamba2` (new) + the five incumbents, re-run as the bit-identity regression check |
| **tuning grid** | `lr ∈ {1e-4, 3.16e-4, 5e-4, 1e-3, 3.16e-3, 1e-2} × wd ∈ {0, 0.1}` = **12 points/arm** (`--grid f3`); **400 outer steps**; `is_full_F3 = True` in the artifact |
| optimiser | `optax.adam` at `wd = 0`, `optax.adamw` (decoupled) at `wd = 0.1`. ⚠ F3's `β = (0.9, 0.98)` and cosine decay **NOT adopted** — the same declared deviation the f3 rider made, kept so the columns stay comparable |
| selections scored (all from the same fits) | **`f3` (fit-split, PRIMARY — the shipped rule the banked rows use)** · `f3_lite_control` (C2W4's 3-lr sub-grid) · `f3_val` (**held-out**, declared secondary) |
| iso-state budget | **1364 float32 = 5456 B**; head widths **29 / 12 / 36 / 36 / 36 / 36** (mamba2 = 36) |
| SSD chunk | **16** (rig-matched; reference default 256) — asserted **inert** |
| Mamba-2 arm config | `n_head = ngroups = 1`, `d_state = head_dim = 36`, `use_D = False`, `gate_z = False`, `d_conv = none`, init `A ~ U(1,16)`, `Δ ~ exp(U(log 1e-3, log 1e-1))` |
| gym / CLU flags | unchanged: `family=aggregate`, `capacity=6`, `consolidate_every=2`, `clu_overrides={stage_admission: True}`; CLU cell = the shipped `aggregate@base` |
| byte law | **corrected** `ratio = [A(D+2)+d]/(d+m)`; ledger identity green on all 18 cells (`5456 B / 100 B / 54.56×`). ⛔ never *"verified to 1e-9 in all 28 cells"* (it is 24/28) |
| CLU column | **banked, never re-derived** — reproduced digit-for-digit as a fidelity check |
| dtype | float32 throughout, both sides of the ledger |
| admissible coverage | `aggregate` s0/s1/s2 **58/72 · 66/80 · 55/80** (0.806 · 0.825 · 0.688), store **5/8**; `overload` **24/24** and **6/6** on every seed — identical to C2W4's |
| wall clock | aggregate n=9 (6 arms, 648 fits) **1447 s** · overload n=9 (9 audit cells **195 s**) + the 45-point frontier sweep ≈ **9 min** · block ablation (216 fits) ≈ **20 min** · full suite (see §3.7) |
| artifacts | `.claude/outputs/bprime-mamba2-arm/{PREREG.md, run_agg_n9/, run_overload_n9/, block_ablation.{py,json,log}, analyse.py, smoke/}` |

**⛔ DECLARED NOT-RUNs (never nulls):** `recency` / `manifold` (protocol-invalid, FB4) · **Titans**
(D5: no official code, chunk size never numeric, no seeds) · **Sparse Delta Memory** (D5: needs Torch
≥2.8 / Triton ≥3.4 / SM 80+) · **GRU / sliding-window attention** (still outside the ruled set) ·
**Mamba-1 and Mamba-3** (only Mamba-2 is funded; Mamba-3's complex/rotational state is a *different*
state type and would need its own ledger row) · **the deletion probe** (no rival family has a deletion
verb, Mamba-2 included) · **any language-model-scale run** (⛔ nothing here transfers to an LM claim) ·
**F3's β/cosine sub-clauses** (declared deviation, inherited) · **no CLU number re-measured**.

---

# 6. GIT FOOTPRINT

**Branch** `agent/experiment-engineer/bprime-mamba2-arm`, base local `main @ eaecc91`. Worked in a
scoped worktree `../CHLU-mamba2`, **removed after verifying the branch ref from the MAIN repo** (protocol
§3.2, the w4 lesson) — the slot is free and the branch carries all 3 commits. ⛔ Not pushed, no PR. Rebase onto **local `main`** (⚠ never `origin/main`, §7.21): **no-op — "up to date"**, no conflicts. Verified from the MAIN repo (`git -C /Users/user/Desktop/CHLU log --oneline main..agent/experiment-engineer/bprime-mamba2-arm`) — the w4 lesson — all 3 commits present.

| commit | subject |
|---|---|
| `e8f4efb` | `[experiment-engineer] add chlu/eval/rivals/mamba2: the Mamba-2 (SSD) rival arm` |
| `da715d3` | `[experiment-engineer] exp_bprime_rivals: score the SSD arm, and stop calling it NOT-RUN` |
| `5a71105` | `[experiment-engineer] tests: SSD faithfulness, the inert chunk, and the iso-state ledger` |

**Files touched — all inside my declared ownership, nothing else:**
- **new:** `chlu/eval/rivals/mamba2.py` (the arm)
- **additive:** `chlu/eval/rivals/fit.py` (RIVALS **append**, `LEDGER_KIND`, `make_rival` branch +
  `**arm_kwargs` passthrough) · `chlu/eval/rivals/ledger.py` (**2 lines**: the `"mamba2"` sizing-kind
  alias + its docstring bullet) · `chlu/eval/rivals/__init__.py` (export only) ·
  `chlu/experiments/exp_bprime_rivals.py` (verdict dispatch, `equations_implemented`, the NOT-RUN /
  FB2 / P5 honesty hunks, the NEW-ARM `before_after` row, `--frontier-rivals`, dynamic `sd_convention`)
- **tests:** `tests/test_bprime_rivals.py` (+11 tests, +1 branch in the parametrized ledger test) ·
  `tests/test_rivals_ledger.py` (the "unknown kind" example moves off `"mamba2"`; +1 test)

⛔ **NOT touched:** `chlu/config.py` · `memory_gym.py` / `exp_memory_gym.py` · `monitors.py` ·
`train_cluformer.py` · any factored-store file · `chlu/eval/dividend.py` · `chlu/eval/race.py`.
⭐ **`audit_table` is byte-untouched**, so the mamba2 columns aggregate through exactly the shipped
rule — the acceptance criterion's load-bearing requirement.
**Concurrent worktrees seen and avoided:** `../CHLU-lane-parallel-controller`, `../CHLU-null-arms`. No
collision.

---

# 7. OPEN QUESTIONS / FOLLOW-UPS / RISKS

1. ⭐ **The rescue gate now disagrees across THREE selections** (reconciliation 3). At n = 9,
   `f3` rescues {deltanet, gdn, gdn2, mamba2}; the C2W4 code path rescues {ttt_linear, deltanet, gdn,
   gdn2}; **held-out selection rescues only {gdn, mamba2}**. The draft's "under both code paths"
   sentence predates the third. **This is the most consequential thing I found that is not about
   Mamba-2.**
2. ⚠ **My PREREG has zero refutations**, which is weaker evidence than one with a refutation (§2). The
   non-inherited predictions are M2b and the six ledger integers; treat M1/M4/M6/M7 accordingly.
3. **Mamba-2 has the most negative blank store in the audit (−1.82, one seed at −4.62).** Its RESCUED
   verdict survives a 9/9 per-seed sign test, so it is not an artefact — but it is another instance of
   `bprime-rivals-f3` §9.1's warning that the gate's control is a single init draw. A **paired** blank
   (several inits averaged) remains the right fix and is still unbuilt.
4. **What Mamba-2 does NOT close.** FB2's "≥2 of 5 families" adjudication moves from 3-reasoned to
   **2-reasoned (Titans, SDM)** — it is not zero. And this is still **one synthetic family, `d_in = 5`,
   5–6 items, ~10-token streams, CPU scale**: ⛔ nothing here transfers to a language-model claim, and
   the SSM family is represented by **one** member (Mamba-1's `d_conv + d_state` state and Mamba-3's
   complex/rotational state are different state types and are declared NOT-RUN).
5. **Cheapest remaining tier-i coverage:** a GRU / sliding-window-attention arm — both trivial in this
   rig now that the arm interface has three implementations, and both would take the audit's measured
   state-type count from 6 to 8.
6. **`--frontier-rivals` is new API surface** with a default that preserves current behaviour; nothing
   else in the repo calls it yet.

---

## Proposed handover updates (for the Hub)

1. **§10 running log / `claims_matrix.md`:** ⭐ **B′'s SSM row exists.** Quotable with this report's
   provenance table: *at byte-matched state on `aggregate` at n = 9, **Mamba-2 (SSD) posts the audit's
   best `full` (−0.4036 ± 0.0329) and still loses to a zero-extra-byte reader of a raw table holding
   the same 5184 bytes by −0.2563 ± 0.0416 (6.2 SE, negative in 9/9 seeds)**. The headline becomes
   **0 of 6 rival arms**, range **−0.2563 … −0.4602**, minimum still **≥ 4.4 SE**.*
2. **Draft edits required (reconciliation 1):** every *"Mamba-2 — NOT-RUN / outside the ruled arm set"*
   site in `draft-r2/r3` (§4.1 table row, §4.1.1, §4.3, §5.1, §5.2's survey sentence, App. J, and the
   "Mamba-2 and a GRU/SWA arm are both cheap" follow-up) must change. **FB2's "reasoned from equations
   only" set is now 2 of 5 (Titans, SDM), not 3 of 5** — the harness emits this automatically.
3. **New never-quote candidates:** (a) ⛔ *any ordering among `deltanet` / `gdn` / `gdn2` / `mamba2` on
   `full`* — they span 0.017 with SEs of 0.012–0.033; (b) ⛔ *any margin against Mamba-2 on the
   `overload` frontier* — **NOT RESCUED in all three selections**; (c) ⛔ *"Mamba-2's +0 B margin is
   positive"* — **+0.0047 ± 0.0519 is a tie**, and it flips sign under held-out selection.
4. **§7 Known Issues — a new entry to consider:** *the rival rescue gate's verdict depends on the
   best-of-grid **selection rule**, not just on seed count.* At n = 9 the held-out selection drops
   `deltanet` and `gdn2` below the bar that fit-split selection clears. Any future rival work must
   state which selection its rescue verdict is from.
5. **§7 Known Issues — closed-by-measurement:** `bprime-rivals-f3`'s *"F3's widened lr axis is never
   selected (0 of 45)"* is **no longer universal** — the SSM arm selects `lr < 1e-3` (1/9 on the fit
   split, 7/9 held-out, and its held-out surface's arg-min is `1e-4`, a point only F3 contains). The
   grid is decorative for delta/TTT arms specifically, not for the rig.
6. **Citation ledger (reconciliation 4):** Mamba-2 = **Dao & Gu, "Transformers are SSMs: Generalized
   Models and Efficient Algorithms Through Structured State Space Duality", ICML 2024,
   arXiv:2405.21060**, from `rival-recon` §1.4's pinned record — ⚠ **not re-verified this session; no
   web tool in this agent's kit.** `web-scout` should double-source before print, per
   `bprime-cite-check`'s pattern.
7. **Config defaults:** none changed. `chlu/config.py` untouched (standing read-only to C2 engineers).
   The new knobs are CLI/API-level (`--frontier-rivals`, `arm_kwargs`) and default to prior behaviour.
