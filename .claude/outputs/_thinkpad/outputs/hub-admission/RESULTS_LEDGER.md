# `hub/admission` — RESULTS LEDGER

**Every run this hub performed, scored or not.** Machine on every row (R3): **FORGIS_14 · Windows AMD64 ·
local CPU · 4 threads**. Object on every row: `shapes3d`/`recon`, **seed 0**, unless stated.
⛔ `bandwidth_rule = nn_median`, `bandwidth_frac = 1.0`, **never altered**.

---

## ⭐⭐ SCORED-RUN BUDGET

| | |
|---|---|
| Budget for this target | **3**, across every attempt and every machine |
| Spent before this hub | **0** |
| **Spent by this hub** | ⭐ **0** |
| **Remaining** | ⭐ **3** |
| `eval_b` (the held-out compositional set) | ⛔ **never read, never embedded, never scored by this hub** |

**Why nothing was spent.** PREREG-1 §10 binds the run to *"only after the DEV number clears the §5 bar"*.
The DEV number missed the bar on two of three legs (pair: **LOSS**, Δ = −0.1383; `accept_A7` gap 0.194 vs
the permitted 0.05; localisation 0.3441 vs ≥ 0.50). ⛔ Spending a held-out run to confirm a failure I
could already measure on DEV would have cost the programme 1 of 3 and bought nothing.

---

## UNITS RUN — all DEV, all `val`/`fit` corners, all zero cost against the budget

| # | unit | artefact | wall clock | what it settles |
|---|---|---|---|---|
| 0c | `u0c_probe.py` | `U0C_PROBE_seed0_A.json` | 57 s | §0c; the three PREREG self-checks; the launch-point atlas |
| 1 | `u1_severity.py` | `U1_SEVERITY_seed0_A.json` | 56 s | §5.10, severity sweep, `unseen_value`, `duplicate_factor` |
| 2 | `u2_map.py` | `FACTOR_WELL_MAP_seed0.json` | 43 s | §3 step 1 — the factor→well map over shuffled-label floors |
| 3 | `u3_dev.py` | `DEV_TABLE_seed0_A.json` | 66 s | the DEV table, all §4 controls, R1 pairs, localisation |
| 4 | `u4_liveset.py` | `LIVESET_seed0_A.json` | 219 s | control (d): six held-out-value rebuilds; §5.6, §5.9 |
| 5 | `u5_atlas.py` | `RADIUS_ATLAS_seed0_A.json` | 79 s | the radius atlas; threshold-free AUROC; the fitted-beside |
| 6 | `u6_closeout.py` | `CLOSEOUT_seed0.json` | 553 s | capacity control (rung B); OOV-symbol; A7 vs A7D; localisation anatomy |
| 7 | `u7_uncertainty.py` | `UNCERTAINTY_seed0_A.json` | 114 s | within-seed bootstrap; the matched-operating-point comparison |
| 8 | `u8_a7_pair.py` | `A7_PAIR_seed0_A.json` | 91 s | the A7-controlled pair across the rival's whole operating curve |
| 9 | `u9_pooled.py` | `POOLED_SUPPORT_seed0_A.json` | 301 s | ⭐ the bar's ACTUAL pooled `out_of_support`; free mode |
| S1 | `u6_figure.py` (sub-agent, Opus) | `CERTIFIED_ACCESS_FIGURE_seed0_A.json`, `certified_access_seed0_A.png` | — | §3 step 6, the certified-access figure |
| S2 | `verify_*.py` (sub-agent, Opus) | `VERIFY_*.json`, `VERIFY_ADMISSION.md` | — | adversarial re-derivation of my headline claims |

**Total compute: ≈ 26.2 minutes** of my own CPU wall clock, 4 threads, one box, plus the two sub-agents
inside the same 4-thread share. ⛔ No GPU. ⛔ No GCloud spend. ⛔ CSF3 not reachable (2FA) and
**NOT RUN** — no submit line was needed; nothing here wants a GPU.

---

## THE HEADLINE NUMBERS, WITH PROVENANCE

| quantity | value | artefact |
|---|---|---|
| `max_i E_k(a_ki)` worst over wells (the §0 identity) | **−30.394** ✅ holds | `U0C_PROBE` |
| `Σ_k E_k` vs shipped `store_energy` | `max_rel_delta = 1.22e-07` ✅ | `U0C_PROBE` |
| `Σ_{t,k}` residual vs shipped `request_energy` | `max_rel_delta = 5.34e-06` ✅ | `DEV_TABLE` |
| `term_experts` vs shipped `request_experts` | **bit-identical** ✅ | `U0C_PROBE` |
| real encoder features sit at | **`d/σ_k = 0.787`** | `RADIUS_ATLAS` |
| derived zero level set sits at | **`d/σ_k = 1.89 – 2.42`** | `RADIUS_ATLAS` |
| certified radius `σ_k√(2 ln N)` | `d/σ_k = 2.884` (rung A) | `RADIUS_ATLAS` |
| rivals' fitted `q = 0.95` quantile | `d/σ_k = 1.05 – 1.29` | `RADIUS_ATLAS` |
| **certified fraction of refusals** | **0.6936** (rung A) · **0.7264** (rung B) | `CLOSEOUT` |
| `accept_val` (arm) | **1.0000** | `DEV_TABLE` |
| `accept_A7` (arm) | **0.8056** → **1.0000** deduplicated | `DEV_TABLE`, `CLOSEOUT` |
| `refuse_oor` (arm) | **0.3714** | `DEV_TABLE` |
| ⭐ `refuse_out_of_support` **POOLED** (arm) — the bar's own population | ⛔ **0.2653** (rival 0.7766) | `POOLED_SUPPORT` |
| Δ vs factored kNN, R1 paired, on `refuse_oor` | **−0.1383**, CI [−0.1564, −0.1203] → **LOSS** | `DEV_TABLE` |
| ⭐ Δ vs factored kNN, R1 paired, **POOLED** | **−0.1502**, CI [−0.1675, −0.1317] → **LOSS** | `POOLED_SUPPORT` |
| arm at rival's **matched** `accept_val = 1.0` | arm `refuse_oor` 0.3714 vs rival **0.6449** | `UNCERTAINTY` |
| free mode accept, 1/2/3 factors free (arm) | ⭐ **1.0000 / 1.0000 / 1.0000** (rival 0.549/0.363/0.217) | `POOLED_SUPPORT` |
| A7-controlled pair vs `knn_factored` q=0.95 | Δ = −0.0355 [−0.0908, +0.0225] → TIE ⛔ **not the bar** | `A7_PAIR` |
| ⚠ Δ at the rival's **own best** `q = 0.98` (verifier) | ⛔ **−0.1772** — my pre-registered figure **understated the loss by 0.039** | `VERIFY_ADMISSION` |
| ⭐ operating-point-free: CLU **distance** vs CLU **energy** | AUROC **0.9107 vs 0.8619** (rival 0.9198) — the `logsumexp` costs separation | `VERIFY_ADMISSION` |
| ⭐ loss decomposition of the 0.177 deficit | **0.114 radius + 0.063 signal** | `VERIFY_ADMISSION` |
| settled-state launch (the one I missed) | refuse: valid **0.1094** vs oor w=2/3 **0.0000** — ⛔ dominated, not an arm | `VERIFY_ADMISSION` |
| full 2048-dim real feature vs **rebuilt** store | refuse **0.0007** — *less* than vs the undamaged store (0.0013) | `VERIFY_ADMISSION` |
| certificate flips / plain decode collapses | **2.06** / **4.08** in `d/σ_k`; certificate flips 2.02 earlier | `CERTIFIED_ACCESS_FIGURE` |
| AUROC `oor` vs `val`, arm | **0.8619** (rival 0.9196; margin `M` 0.4183) | `RADIUS_ATLAS` |
| live-set refuse (held-out exemplar) | ⛔ **0.0000** (bar ≥ 0.90) | `LIVESET` |
| live-set accept (write-back) | 1.0000 (bar ≥ 0.90) — ⚠ **vacuous**, nothing is ever refused | `LIVESET` |
| term localisation | **0.3441** (bar ≥ 0.50); single-well control 0.1146 | `DEV_TABLE` |

---

## FALSIFIERS — every one answered with a number

| falsifier | verdict | the number |
|---|---|---|
| **§5.6** shortfall > 0 on a held-out-value rebuild | ⭐ **DOES NOT FIRE** — ⚠ but **by construction and by the verifier's independent route**, ⛔ **not** by my assertion, which was a tautology (`u4_liveset.py:99`) | `shortfall = 0` on all six; `gather_features` is never re-entered because `build_landscape` gets a slice of what it already returned |
| **§5.7** the gate does not calibrate | ⭐ **DOES NOT FIRE** | `accept_val = 1.0000` ≫ 0.50; the gate over-*accepts*, it does not over-refuse |
| **§5.8** no design advantage | ⚠ **PARTIAL** | factored kNN does not *tie* the pair, it **wins** it (Δ = −0.1383); but localisation **does** beat the single-well control (0.3441 vs 0.1146). The literal conjunction fails, the spirit holds. |
| **§5.9** the encoder does not separate | ⛔ **FIRES, on all six factors** | refuse(held-out) − refuse(in-vocab) = **+0.0000** on every factor; both are 0.0000 |
| **§5.10** the affine head does not cross the level set | ⭐ **DOES NOT FIRE** at `gate@r8_affine` | `max_k E_k` rises monotonically −34 → +2060 over `w = 2 → 13` on all six factors |
| **§5.10** at the SHIPPED `clamp_atom` launch | ⛔ **FIRES, by algebra** | `d = 0` exactly ⇒ `E_k ≤ 0` always ⇒ `refuse ≡ 0` |
| **§5.10** at `posterior_mean` | ⛔ **FIRES** | `accept = 1.0000` at every factor and every severity |

---

## THE BAR (§5) — as set by the task file, ⛔ never moved

| leg | bar | measured (DEV) | verdict |
|---|---|---|---|
| Pair vs factored kNN under R1 | win-or-tie | Δ = **−0.1383**, CI [−0.1564, −0.1203] | ⛔ **FAIL (loss)** |
| `accept_A7 ≥ accept_B − 0.05` | gap ≤ 0.05 | gap = **0.194** (1.0000 → 0.8056) | ⛔ **FAIL** |
| Localisation ≥ 0.50 | ≥ 0.50 | **0.3441** | ⛔ **FAIL** |
| Localisation beats single-well control | beyond seed spread | 0.3441 vs **0.1146** (chance 0.1667) | ⭐ **PASS** |
| Live-set refuse ≥ 0.90 | ≥ 0.90 | ⛔ **0.0000** | ⛔ **FAIL** |
| Live-set write-back accept ≥ 0.90 | ≥ 0.90 | 1.0000 | ⚠ **PASS, but vacuously** |
| No tuning | gate reported is `E_k > 0` | ⭐ it is; the fitted alternative is reported beside and **not adopted** | ⭐ **PASS** |

---

## COST LINE

| item | cost |
|---|---|
| Local CPU, this box, 4 threads | **≈ 17.9 min** compute; ≈ 0.5 h wall clock to the last unit |
| GPU | ⛔ **none** |
| GCloud | ⛔ **none — £0 spent, nothing submitted** |
| CSF3 | ⛔ unreachable (2FA); **NOT RUN**; nothing here required it |
| Scored-run budget | ⭐ **0 of 3 spent** |
| Sub-agents | 2, both Opus, both inside the 4-thread share |
