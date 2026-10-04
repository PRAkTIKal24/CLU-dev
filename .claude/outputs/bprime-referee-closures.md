# bprime-referee-closures — experiment-engineer report

**Task + acceptance criterion:** deliver (1) the **CLU column at a uniform n = 9** under the F3 protocol
with rider-1's aggregation rule (full · launder · dividend · +0 B margin · raw-table margin · blank ·
same-keys null · **rescue-gate verdict**), and (2) **renders of the five App-K figure specs** from banked
artifacts with a figure→artifact→field provenance table. **Status: done** (2 declared additions, 1
pre-registered prediction REFUTED, 0 `chlu/` edits, 0 commits — nothing tracked was touched).

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (first-10-lines rule)
> **R1 — ⛔ THE CLU IS *NOT RESCUED* AT n = 9.** lift over its own blank store = **−0.0465 ± 0.0406**
> (n = 9, paired; |t| = 1.14; 2 SE = 0.0813) ⇒ under B.5 the written store is **statistically
> indistinguishable from an empty one** on `aggregate`, the same category as `ttt_mlp`, with the point
> estimate on the wrong side of zero. **MF-1 is closed by measurement, not by drafting**, and the n = 3
> "✅ rescued" is replaced. *(Owner: `bprime-draft-r3` — §4.1.1 CLU row, §4.2, App I/L.)*
> **R2 — ⛔ THE CLU BYTE LEDGER IS NOT SEED-CONSTANT** (my prereg P9, **REFUTED**): 8 of 9 seeds give
> `5456 B / 100 B / 54.56×`, **seed 8 gives `5472 B / 120 B / 45.60×`** because the store's own admission
> gate admitted **6** items instead of 5. `identity_T1` is green on all 9. ⛔ `5456/100/54.56×` must be
> labelled **modal (8 of 9)**, exactly as rider 1's R4 forced for the TTT arms. *(Owner:
> `bprime-draft-r3` — §4.1.1, §4.4/T1 sites, App A.)*
> **R3 — ⚠ `draft-r2` §4.6's fitted-ruler cell for the audited configuration (`d/s = 3.72`) does not
> reproduce**: `c6_summary.json` gives `ds_fit = 3.590` (= `d/s_fit`), while `3.713 = sep/s_fit`. The other
> five rows of that table ARE `d/s_fit` (1.102/1.706/2.048/2.678/4.409 → printed 1.10/1.71/2.05/2.68/4.41),
> so the audited row is the **one cell that switches ruler mid-table** — and it is the load-bearing one
> (SF-6's neighbourhood). Figure 3 plots **3.59**. *(Owner: `bprime-draft-r3` + whoever owns SF-6.)*
> **R4 — ⚠ the `±0.40` on §4.6's `0.814` row does not reproduce** from the merged `c6_summary.json`
> (`grad_se = 0.15379` ⇒ 2 SE = **0.3076**; the pre-topup 2-cell artifact gives `0.8886 ± 0.4656`). It is
> the inadmissible 3/6 row, so nothing rests on it, but it is a C-7 crack. *(Owner: `bprime-draft-r3`.)*
> **R5 — App K's Figure-1 caption rule ("n = 9 on the rival bars and n = 3 on the CLU bar — the caption
> must say so") is SUPERSEDED**: the render is uniform **n = 9** on all six bars. The mixed-n caption
> language for Fig 1 must be deleted, not carried. *(Owner: `bprime-draft-r3`.)*

---

# 0. DIAL DECLARATION (echoed before the first result, protocol §7)

- **Dial:** **none — instrument / closure work.** No new dial claim: item 1 is a **re-aggregation of
  already-banked per-seed cells** at uniform n, item 2 is rendering. The one *new* content is a **verdict**
  (the CLU's own rescue-gate status at power) and a **byte-ledger seed-dependence finding**.
- **Laundering control:** unchanged and inherited per cell — projected/arg-min launder · three **+0 B**
  readers of the store's own table · **raw-metric table at the same bytes** · same-keys null · **blank
  store** · two-sided byte ledger · identical `φ` (`phi_id = 09dc0ee5…`, enforced in code).
- **Falsifies (mine):** the CLU clearing its own blank by > 2 SE at n = 9 would have *rescued* the n = 3
  "✅" and dissolved MF-1's premise. It did not.
- **Does NOT falsify:** the CLU losing to a **raw table read at its own bytes** — that is the
  metric-native-ceiling theorem on a metric-native protocol and is the paper's own headline (§4.2). Nor
  does the NOT-RESCUED verdict falsify any tier-ii/iii claim: it is one designed synthetic family at
  `d_in = 5`, 5–6 items, CPU.

---

# 1. ⭐ THE CLU COLUMN AT n = 9 (referee missing-experiment 1)

**Nothing was measured.** `run_rivals_cell` runs the shipped CLU write/read path on **every** cell
regardless of the rival tuning grid, so the F3 rider's `seeds3to8` runs **already contained**
`clu_reproduction` at seeds 3–8; rider 1 excluded the CLU by policy ("banked, never re-derived"). This is
that aggregation, under rider 1's rule. Full artifact: `n9_clu_column.json` +
`n9_clu_column_table.md`; per-seed rows and the reader set are in the table file.

| quantity (n = 9, seeds 0–8) | value | SE mult. | n = 3 (what §4.1.1 prints now) |
|---|---|---|---|
| `full` | **−0.4370 ± 0.0417** | 10.5 | −0.5261 ± 0.0863 |
| `launder` (its own raw (key,payload) table) | **−0.3810 ± 0.0345** | 11.1 | −0.4472 |
| `blank` (empty store, same read) | **−0.3906 ± 0.0124** | 31.6 | −0.4221 |
| `same-keys null` | **−0.6512 ± 0.0383** | 17.0 | −0.8175 |
| **dividend** (`full − launder`, paired) | **−0.0561 ± 0.0315** | **1.78** | −0.0789 (no SE, quoted ≥4×) |
| **+0 B margin** (paired, per-seed arg-max reader) | **−0.2897 ± 0.0328** | **8.84** | −0.3180 ± 0.0804 |
| **raw-table margin** (+0 B set ∪ arg-min launder) | **−0.2897 ± 0.0328** | **8.84** | not printed |
| **lift over own blank** (paired) | **−0.0465 ± 0.0406** | 1.14 | −0.1040 (implied, never stated) |
| `full − same-keys null` (paired) | **+0.2141 ± 0.0443** | 4.83 | +0.2914 (implied) |
| **RESCUE GATE (B.5)** | ⛔ **NOT RESCUED** | — | "✅ (against blank −0.4221)" ⛔ never-quote |

**What changes for the paper, one line each.**
1. ⭐ **The CLU row's rescue tick becomes ⛔ NOT RESCUED** — measured, at the same n as every rival arm.
   The referee's third quotable sentence (MF-1) is erased at the root, and the honest replacement is
   *stronger*: on the one family that survives protocol validation, **the written content does not lift the
   read above an empty store**. Same verdict under both code paths (bit-identical, §3 P8).
2. **The headline number set improves and the sign never moves:** `full` −0.4370 (was −0.5261), `+0 B
   margin` −0.2897 ± 0.0328 = **8.8 SE below zero** (was ≈4 SE at n = 3), so §4.2's *"…and the CLU over
   three seeds (−0.3180 ± 0.0804)"* becomes *"…and the CLU over nine seeds (−0.2897 ± 0.0328)"*, sitting
   inside the rival range instead of beside it at a different n.
3. **SF-3 is closed:** the dividend now carries `n = 9` and an SE — **−0.0561 ± 0.0315**, i.e. negative but
   **only 1.78 SE**, so it is a *sign statement*, not a significant effect. ⛔ Any sentence claiming the
   launder "beats" the store must be softened to "reads no worse than" or carry the 1.78 SE.
4. ⭐ **The draft's parenthetical "(its own table is already raw)" is now MEASURED:** the raw-table margin
   is **float-identical to the +0 B margin on 9 of 9 seeds** — the arg-min launder (≈ −0.38) never beats
   the `knn2` readers (≈ −0.15) on any seed. The projected-vs-raw distinction genuinely does not arise.
5. **Two +0 B conventions agree:** per-seed arg-max (the rivals' own rule) −0.2897 ± 0.0328 vs the banked
   fixed-reader rule (`knn2_idw_+0B`) −0.2862 ± 0.0317; Δ = 0.0035. No claim turns on the choice.
6. ⚠ **R2 (new):** the byte ledger is **not** seed-constant (see reconciliation list).
7. **Free bonus (declared addition):** admissible-cell coverage at **all 9 seeds** (the draft tabulates
   0–2 only): fractions 0.806/0.825/0.688/0.562/0.667/0.700/0.800/0.750/**0.455**, mean **0.695 ± 0.041**;
   store admission 5/8 on eight seeds and **6/8 on seed 8** (mean 0.639 ± 0.014). Seed 8 is the outlier on
   both — one mechanism (a sixth admitted item) explains its coverage drop *and* its byte ledger.
8. ⭐ **A structural observation the Hub may want in §4.1.1's prose:** the **blank** store's read is far
   more stable across seeds (SE 0.0124) than the **written** store's (SE 0.0417). The variance the rescue
   gate fights on the CLU side comes from the *write*, not the control — the opposite of the rival side,
   where rider 1 found the blank arm dominating the variance.

---

# 2. THE FIVE FIGURE RENDERS (SF-8)

Files (PNG @200 dpi + PDF twin) in `.claude/papers/bprime/figures/`:
`fig1_headline_raw_margin` · `fig2_two_sided_byte_ledger` · `fig3_thirdparty_attribution` ·
`fig4_protocol_validation` · `fig5_frontier_curve`.
Renderer: `.claude/scratch/bprime-referee-closures/render_figures.py` (read-only over artifacts).
Machine-readable provenance: **`figure_provenance.json`, 48 entries**, each `figure → artifact → field →
value`. ⚠ **No `dataviz` skill is installed on this machine** (searched `~/.claude/skills`,
`~/.claude/plugins/marketplaces`, and a filesystem sweep — absent), so styling follows
`chlu/utils/plotting.py`'s conventions (matplotlib, `dpi`, `bbox_inches="tight"`).

| fig | spec (App K) | source artifact(s) | fields used | deviations / declarations |
|---|---|---|---|---|
| **1** | signed +0 B raw-metric margin per arm, zero line, gate hatching, CLU distinct fill, ±1 SE, n in caption | `pilot-placement-probe/n9_full_columns.json` (rivals) · **`bprime-referee-closures/n9_clu_column.json`** (CLU) | `columns.f3_n9.table.rivals.<arm>.raw_table_margin(_se)`; `RESCUED_above_own_blank_2se` under **both** paths; `columns.f3_n9.clu.raw_table_margin.{mean,se,n}` | ⭐ **uniform n = 9** (spec said n = 3 for CLU — R5). `UNSTABLE` = rescued under exactly one code path (the draft's own rule). **CLU is hatched NOT RESCUED** |
| **2** | two-sided ledger, F1 (learned-init hatched) + F2, own-table tick, CLU 54.56× on a broken axis, "unreachable by construction (T1)" | `bprime-rivals-f3/{run400,seeds3to8}/exp_bprime_rivals_metrics.json` | `rivals.<arm>.byte_ledger.rival.{param_bytes,state_bytes,param_breakdown.W0_init}`, `.matched_table.state_bytes`; `clu_byte_ledger.{two_sided_learned_init_rule.{param,state}_bytes,launder_bytes,ratio}` | bars = **modal of 9 seeds**, black caps = the arm's other per-seed configuration (R4/R2: TTT arms and the CLU are not seed-constant). Delta arms have **no** learned-init component (`W0_init = 0`), correctly drawn as all-parameter |
| **3** | κ vs `d/s`, both rulers, fitted `exp(−½(d/s)²)` + R², table's exact zero on the axis, audited config marked, `λ_min < 0` region shaded | `bprime-c6/c6_summary.json` | `rows[*].{R,ds_fit,ds_proxy,grad,grad_se,lam,n_adm,n_cells,sep,s_fit}`; `fits.gradient_ratio.{slope,prefactor,s_implied,r2,decades}` | plots **`ds_fit` = 3.590** at the audited cell, not the draft's 3.72 (**R3**); ±2 SE from `grad_se` (**R4**); the shaded region's right edge is the **geometric midpoint** between the 3/6 and the next 3/3 point — **presentational, declared, logged in the provenance JSON** |
| **4** | `S(f)` per family vs the saturation threshold, substitute byte cost annotated, full-attention reader overlaid | `bprime-fb4-gate/exp_fb4_gate_metrics.json` | `gate.rows[*].{family,metric,S,se_paired,saturated,sub_name,blank,sub,attn,metric_max,detail.byte_ratio_seeds}`; `gate.secondary_excl_launder.rows[*].S` | the attention overlay is **DERIVED** — `(attn − blank)/(M − blank)`, i.e. the gate's own `S` rule applied to the attention reader (logged as derived); the declared **secondary** reading (`S_excl`, incl. `overload`'s 0.65) is plotted as open circles |
| **5** | CLU accuracy-vs-bytes curve (0.972 → 0.097 as 478× → 2.28×), rival points omitted + caption reason | `memory-gym-v0/exp_memory_gym_metrics.json` | `byte_frontier[*].{arm,byte_ratio,primary,n_live,n_atoms}`; `cells[*].dividend.launder` | ⚠ the swept arms differ in **write load** (6 vs 17 live items) — plotted as **two series**, and only the 1× series is connected; joining them would imply a sweep that was not run. `17.11×` carries **two arms** at the same ratio (`base` n = 3, `reach_free` n = 1) — both plotted, neither dropped. Byte floor **2.20×** drawn from T1 |

**Every plotted value traces to a named JSON field.** The only non-artifact quantities in any figure are
(a) Fig 4's attention normalisation, computed with the artifact's own published rule, and (b) Fig 3's
shading boundary, which is presentational — both are declared above and inside
`figure_provenance.json`.

---

# 3. PREREG SCORECARD (`bprime-referee-closures/PREREG.md`, filed before the aggregation ran)

Disclosure in the PREREG §0: seeds 0–2 were public (banked in the draft) and I had incidentally seen
**seed 3's** `full/launder/blank` while inspecting the artifact schema; **seeds 4–8 were unseen** (6 of 9
cells blind).

| # | registered | measured | verdict |
|---|---|---|---|
| **P1** | rescue verdict = **NOT RESCUED** (P(RESCUED) ≤ 10 %) | lift −0.0465 ± 0.0406 ⇒ **NOT RESCUED** | ✅ **CONFIRMED** |
| **P2** | `lift` ∈ [−0.20, +0.02], point −0.09 | **−0.0465 ± 0.0406** | ✅ in band |
| **P3** | dividend negative, ∈ [−0.22, −0.03], point −0.11 | **−0.0561 ± 0.0315** | ✅ in band (⚠ only 1.78 SE — weaker than the n=3 wording implies) |
| **P4** | +0 B margin negative, ≥ 2 SE, ∈ [−0.40, −0.18], point −0.28 | **−0.2897 ± 0.0328** (8.84 SE) | ✅✅ **near-exact** (predicted −0.28) |
| **P5** | raw margin == +0 B margin on ≥ 8 of 9 seeds | **9 of 9, float-identical** | ✅✅ **EXACT** |
| **P6** | `full − null` positive and > 2 SE | **+0.2141 ± 0.0443** (4.8 SE) | ✅ |
| **P7** | `full` ∈ [−0.56, −0.42], point −0.50 | **−0.4370 ± 0.0417** | ◐ in band, at the edge (my point estimate was 0.063 too pessimistic — the unseen seeds ran *better* than seeds 0–3) |
| **P8** | CLU bit-identical across both code paths, all 9 seeds | **max |Δ| = 0.0** on `full`/`launder`/`blank`/`null` | ✅✅ **EXACT** |
| **P9** | byte ledger constant across 9 seeds (`5456/100/54.56×`) | ⛔ **seed 8 = `5472/120/45.60×`** (6 items admitted) | ⛔ **REFUTED — and it is a paper-facing finding (R2)** |

**Score: 4 exact/near-exact · 3 confirmed-in-band · 1 partial · 1 REFUTED.** The refutation (P9) is the
one thing here a reader could not have predicted from the banked column, and it is reported as the finding,
not smoothed: I predicted the CLU's byte ledger was seed-independent because its `A` and `K` looked fixed;
**the admission gate is stochastic across seeds and the ledger follows it.**

---

# 4. HOW I VERIFIED (commands + observed output)

| check | command | observed |
|---|---|---|
| n = 9 CLU aggregation | `PYTHONPATH=. .venv/bin/python .claude/scratch/bprime-referee-closures/n9_clu_column.py` | 2 columns × 9 seeds; `full −0.4370 ± 0.0417`, `lift −0.0465 ± 0.0406 ⇒ NOT RESCUED`, ledger `[5456,5472]/[100,120]/[45.6,54.56] constant=False` |
| **fidelity to the banked n = 3** | same run | seeds 0–2 = `−0.682608 / −0.384693 / −0.511032` **digit-for-digit**; their mean **−0.526111** = the draft's −0.5261 |
| **shipped-rule identity** | `audit_table(cells)["aggregate"]["clu_reproduced"]` inside the same script | `full −0.4370470471`, `launder −0.3809775327`, `dividend −0.0560695144` — identical to my paired computation |
| **independent hand-check** (numpy, no shipped code) | `.venv/bin/python -c "…np.std(ddof=1)/3…"` | `full −0.437047 ± 0.041739` · `blank −0.390587 ± 0.012374` · `lift −0.046460 ± 0.040631`, `2 SE = 0.081261`, `RESCUED = False` · `dividend −0.056070 ± 0.031549` — **reproduces the script exactly** |
| **P8 cross-path identity** | same run | `{'full': True, 'launder': True, 'blank': True, 'same_keys_null': True, 'max_abs_diff': 0.0}` |
| **rider-1 rule reproduces on this machine** | re-ran `pilot-placement-probe/n9_aggregate.py`, diffed against the banked artifact | `rider-1 re-run byte-identical to banked artifact: True` (so Fig 1's rival source is reproducible and my rule is the same rule) |
| figure renders | `PYTHONPATH=. .venv/bin/python .claude/scratch/bprime-referee-closures/render_figures.py` | 5 PNG + 5 PDF written, `figure_provenance.json` 48 entries, **no missing-glyph warnings** after the ⛔/✅ characters were removed from figure text |
| visual inspection | read back all five PNGs | all legends/labels non-overlapping after 3 layout iterations (recorded in the scratch script) |
| **tests / lint** | *(not run — no code touched)* | `git status --short` **empty**; working tree clean at `eaecc91`; no `chlu/`, `tests/` or `pyproject` file opened for writing |

---

# 5. FLAG-PROVENANCE TABLE (protocol §5 — applies to every number above)

| item | value |
|---|---|
| commit | **`eaecc91`** (local `main`, clean tree). ⛔ **no branch, no commit, no tracked file touched** — every artifact is under `.claude/` (protocol §2/§3: research-only agents have nothing to commit) |
| venv | **main venv reused** (`/Users/user/Desktop/CHLU/.venv`), `PYTHONPATH=/Users/user/Desktop/CHLU`, **no `uv sync`** (w6 hazard avoided) |
| JAX / Equinox / Optax / numpy / matplotlib | **0.9.0 / 0.13.4 / 0.2.6 / 2.4.1 / 3.10.8** — resolved and printed this session (importing `exp_bprime_rivals.audit_table` *does* pull JAX at module scope, verified: `'jax' in sys.modules == True`; **no JAX computation is executed** — the aggregation is pure numpy over JSON). Identical stack to the one that produced the underlying `bprime-rivals-f3` measurements |
| seeds | **0–8 (n = 9)**, uniform on every CLU column; SE = sample sd (ddof = 1)/√n; every margin/lift **paired per seed** |
| family / arm | **`aggregate@base` only** (S = 0.5068, the sole reader-discrimination family). `capacity=6`, `consolidate_every=2`, `clu_overrides={stage_admission: True}`, shipped CLU cell |
| CLU read/write flags | the **shipped** `aggregate@base` configuration, unchanged (`clu_config_non_default` recorded per cell in the source artifacts); `langevin`/temperature **N/A** (deterministic read, T = 0) — §7.22's discipline does not apply |
| rival grid (context only) | F3 `lr ∈ {1e-4…1e-2} × wd ∈ {0,0.1}`, 400 steps, best-of-grid on the fit split; **the CLU column is grid-independent (P8, bit-identical across both code paths)** |
| iso-state budget | 1364 float32 = **5456 B**; CLU two-sided split F1 **5376 B** / F2 **5200 B** (⚠ 5472 B on seed 8) |
| byte law | corrected `ratio = [A(D+2)+d]/(d+m)`; `identity_T1.ok = true` on **all 9 seeds**; ratio **54.56× on 8 seeds, 45.60× on seed 8** ⛔ never "verified to 1e-9 in all 28 cells" |
| identical-φ | `phi_id = 09dc0ee5…` asserted in code on every source cell |
| metric | `neg_mae` (higher = better) on `aggregate`; `decode` on the Fig-5 `overload` frontier |
| wall clock | ≈ **4 min** total compute (2 aggregation runs + 4 render passes). ⛔ **0 new measurements, 0 A100-hours** |
| artifacts | `.claude/outputs/bprime-referee-closures/{PREREG.md, n9_clu_column.json, n9_clu_column_table.md, n9_coverage.json, figure_provenance.json}` · `.claude/papers/bprime/figures/fig{1..5}*.{png,pdf}` · scripts in `.claude/scratch/bprime-referee-closures/` |

**⛔ DECLARED NOT-RUNs (never nulls).**
- **No CLU cell re-measured** — the column is a re-aggregation of banked per-seed cells (that is *why* it
  is cheap; the task's "≈31 min for one arm" budget was not needed).
- **`overload` byte-frontier CLU column NOT re-aggregated at n = 9** — it is banked at 3 seeds and Fig 5
  plots the banked curve; extending it *would* require new runs (out of scope).
- **`recency` / `manifold`** — protocol-invalid (FB4), plotted in Fig 4 only as struck families.
- **Rival arms untouched** — every rival number in Fig 1/2 is read from rider 1's artifact; no refit.
- **No Mamba-2 / GRU / SWA / Titans / SDM arm** (§A14.2/D5 rulings, unchanged).
- **No test suite run and no lint run** — justified only because **no tracked file was modified**
  (`git status` clean, evidenced in §4). If the Hub wants the aggregation shipped inside
  `exp_bprime_rivals.audit_table` (so the CLU column comes out of the harness rather than a scratch
  script), that is a **code task** and would need tests; I flag it in §7 rather than doing it unilaterally.

---

# 6. RISKS ON THE RECORD

1. Everything is `d_in = 5`, 5–6 stored items, ~10-token streams, one designed synthetic family, CPU.
   ⛔ Nothing here transfers to a language-model claim.
2. The CLU's NOT-RESCUED verdict is **one family**. It is *not* evidence that the store cannot beat a blank
   store in general — it is evidence that it does not on `aggregate@base` at the shipped configuration,
   which is the only family the protocol left standing.
3. `n = 9` is still small for a gate whose control has this much spread; the verdict is stable across the
   two code paths **because the CLU path is literally identical across them** (P8), so the cross-path
   agreement is *not* independent evidence the way it is for the rivals. Say so if the draft leans on it.
4. Fig 5's curve mixes two write loads (declared, plotted separately). A single-load frontier is a new run.
5. The figures are renders of **banked** numbers: if any upstream artifact is re-measured (e.g. the
   deltanet frontier row, or a re-run of `bprime-c6`), Fig 1/3/5 must be re-rendered — the renderer is
   deterministic and takes ~20 s.

---

# 7. OPEN QUESTIONS / FOLLOW-UPS

1. **Should the CLU column come out of the harness?** Right now `audit_table` emits only
   `clu_reproduced.{full,launder,dividend}`; the blank/null/+0 B/raw/lift/gate columns exist per cell but
   are aggregated by my scratch script. A ~30-line addition to `audit_table` (plus a test) would make the
   CLU row a first-class harness output and remove the "aggregated in a spoke script" provenance step.
   **I did not do it** — the task said no `chlu/` edits expected. Hub call.
2. **Does the n = 9 CLU verdict change §4.3's falsifier adjudication?** `falsifier_adjudication` computes
   `fb3_strong` from `clu_banked` (n = 3, dividend −0.0789). At n = 9 the dividend is −0.0561 ± 0.0315 —
   same sign, so `clu_div <= 0` is unchanged and **no adjudication flips**; but the code reads the banked
   constant, not the n = 9 column, so if the Hub adopts n = 9 the constant should be updated in the same
   edit as item 1.
3. **`BANKED_CLU["aggregate/base"]` is now a 3-of-9 subset of a 9-seed column.** Leaving it as the
   "banked" reference is defensible (it is what the first pass published) but it will read as a
   contradiction next to the n = 9 table unless labelled.
4. **Seed 8 deserves one sentence somewhere**: it is simultaneously the lowest-coverage cell (0.455), the
   only 6-item cell, and the only different byte ledger. It is not an outlier to drop — it is the
   admission gate doing its job — but it is the single cell a referee will ask about.
5. `reach_free` at 17.11× in Fig 5 is an ablation arm sharing a byte ratio with `base`. If the writer
   prefers a clean frontier, dropping it is a **declared** choice, not a silent one.

---

## Proposed handover updates (for the Hub)

1. **§10 / `claims_matrix.md`:** the **CLU column now exists at n = 9** on `aggregate@base` and the
   **rescue-gate verdict is ⛔ NOT RESCUED** (lift **−0.0465 ± 0.0406**, |t| = 1.14). Quotable with this
   report's provenance table. Headline set at n = 9: `full −0.4370 ± 0.0417` · `+0 B / raw margin
   −0.2897 ± 0.0328` (**8.8 SE below zero**) · `dividend −0.0561 ± 0.0315` (1.78 SE) · `full − null
   +0.2141 ± 0.0443`. **The paper's n-asymmetry is closed: every arm in §4.1.1 is now n = 9.**
2. ⛔ **Never-quote — CONFIRMED AND SUPERSEDED:** the CLU's *"✅ rescued (against blank −0.4221)"* at
   n = 3 is dead twice over (A18.1 power rule + false under B.5). The replacement text is *"NOT RESCUED at
   nine seeds; the written store is within noise of its own blank store, and below it in point estimate."*
3. ⛔ **New never-quote candidate:** *"5456 B / 100 B / 54.56×"* **as an unqualified n = 9 value** — it is
   the **modal (8 of 9)** ledger; seed 8 is `5472/120/45.60×`. (Same class as rider 1's R4 for the TTT
   arms.) `identity_T1` is green on all nine, so this is a labelling rule, not a defect.
4. **§7 Known Issues — candidate entry:** *the gym's admission gate is seed-dependent (5 or 6 of 8 items
   admitted on `aggregate@base` across seeds 0–8), and the CLU byte ledger is a function of it.* Any
   byte-ratio statement about a gym cell must be per-seed or modal-labelled.
5. **Figures exist** (SF-8 discharged): 5 PNG + 5 PDF in `.claude/papers/bprime/figures/` with a 48-entry
   `figure_provenance.json`. App K needs three edits: Fig 1's mixed-n caption rule is superseded (R5), Fig
   3 must name `d/s = 3.59` at the audited cell or explain the ruler switch (R3), and Fig 5's caption must
   state the two write loads.
6. **Two draft numbers do not reproduce from their artifacts** (R3: `d/s = 3.72` → 3.590; R4: `±0.40` →
   ±0.3076). Both are in §4.6, both low-stakes, both C-7 cracks — the r3 writer should fix or annotate.
7. **Config defaults:** none changed. `chlu/config.py` and all of `chlu/` untouched; no branch created.
   The one code opportunity (promoting the CLU column into `audit_table`) is filed as §7 item 1.
