# memory-gym-v0 — experiment-engineer report

> ## ⚠⚠ ERRATUM BANNER — appended 2026-07-31 (C2W4) by `doc-curator-c2w3-sync`. **The body below is UNEDITED; read this first.**
> ⛔ **§2 / §3.1 / `PREREG-B1`'s byte law is WRONG on 4 of 28 cells.** The published closed form `ratio = atoms_per_item·(dim+2)/dim + d/dim = 1.4·A + 0.8`, and the sentence *"verified to 1e-9 in all 28 cells"*, reproduce the measured ledger in **24 of 28** cells only. The four `manifold` (`n_spectator = 1`) cells measure **52.00×** against a published **43.33×** (**+8.6667, +20 %**), and the printed floor there (**2.00×**) is wrong — the true floor **RISES to 2.40×**. Cause: `memory_gym.byte_ratio_law` divides by the **store** dim `D` where the launder row is `(d+m)` floats.
> ⭐ **Corrected law — exact in all 28 cells in rational arithmetic (0 ulp): `ratio = [A(D+2) + d]/(d+m)`.**
> ⭐ **NOTHING MEASURED HERE IS RETRACTED.** Every measured ratio in this report is correct; it is the *law* that was wrong, and **the error is CONSERVATIVE** (the store costs *more* relative to the table than published) — so **no claim in this report was inflated**, the `n_spec = 0` floor **2.20×** and the measured minimum **2.28×** are **unchanged**, and the byte-floor **theorem STANDS**.
> **Full erratum + the replacement sentence: `.claude/outputs/track2-admissibility/ERRATA-Bprime.md` §E1.** Sources: `bprime-theory` R-BYTE/T1.1/T1.2; `[C2W3]` §10 Hub re-derivation. Registry: **N167**, and **N134**'s dated correction block. ⚠ Code fix + published diff is `harness-debt`'s (C2W4, charter §A14.4) — **if their landed numbers ever differ from this banner, theirs are the numbers.**

Task + acceptance criterion: build Track 1's internal memory gym (one family per charter §2.1 opening) on the `full-clu-harness` harness and **measure the baseline dividend on every family with all three controls firing automatically and the byte ledger published, multi-seed**.
Status: **done.** Acceptance **MET**: 28 cells, 4 families + 10 arms, 3 seeds on every headline cell, three harness-native controls + a family-specific **+0 B** substitute + a two-sided byte ledger on **every** cell, 140 consolidation windows. ⭐ **The v0 dividend is ≈0 or negative on every family** — the charter's own stated expectation, i.e. a successful outcome. ⛔ **The gym falsifier did NOT fire** (the same-keys null beats CLU on 2 of 4 families, not all 4), **but the trivial-substitute audit is 0 for 4: the CLU is not the best reader of its own byte budget on any family.** ⭐⭐ The Hub's hard problem (bytes unmatched by 359×) is answered as a **measured frontier with a proof-shaped floor**: matched bytes is **unreachable by construction**, and the curve is `decode 0.972 (478×) → 0.458 (45.6×) → 0.333 (12.0×) → 0.097 (2.28×)` against a launder at **1.000 throughout**.

> ## ⚠ RECONCILIATION LIST — needs a Hub-assigned owner (protocol §5 corollary, first-10-lines rule)
> **R1 — the harness's own R1 gets worse and now has a SCOPE.** `controller-doctrine`'s Prop D1 bound `D ≤ U` under the `sep/2` inradius proxy is violated by up to **7.44×** here (harness measured 1.5–2.5×). ⭐ And it is now *scoped by measurement*: D1 **holds** (D/U = 0.81–1.04, 6/6 cells) exactly where `U` is large — queries placed *between* wells — and **fails 4.4–7.4×** where `U` is small (queries near centres, U = 0.125). *(Owner: theorist `controller-doctrine` + Hub.)*
> **R2 — monitor #6 has no dead-band and trips on numerically-zero slopes.** 29 of its 58 first-ever trips fire at `slope_write_loss = −5.2e−17`, `slope_acq = −5.9e−17` (predicate `slope_loss < 0 and slope_acq <= 0`). The other ~29 are genuine (e.g. `overload/base@s2`: slope_acq −0.214, slope_loss −0.055). `monitors.py` is **read-only to me**; proposed predicate: `slope_loss < −eps and slope_acq <= 0` with `eps ~ 1e−9 * scale`. *(Owner: the monitors' owner / next C2 engineer.)*
> **R3 — the spacing certificate's validity leg does not predict drift on a learned `V_θ`.** First-ever measurement of #3's leg (ii): `corr(gate margin, post-write drift)` ranges **−0.99 … +0.56** across cells and is **sign-unstable**, so the leg fires ~half the time for the wrong reason. A certificate that does not predict drift is not a certificate (the monitor's own words). *(Owner: theorist + engineer.)*
> **R4 — the task file's byte route 2 is closed by arithmetic and must be restated.** "Load the store far beyond the launder's table" cannot happen: at 20 B/row the byte-matched table holds ≥ 11× the CLU's items at every reachable configuration. `matched=False` is **architectural**, not a configuration choice. *(Owner: Hub — task-file/handover wording.)*
> **R5 — my dedicated #11 probe (`overload/reach_free`) was a no-op**: the over-excursion item was refused by the *merge* gate, not by reach, so the arm reproduced `base` bit-for-bit. #11 fired anyway from ordinary geometry (7 trips). A deliberate reach probe needs a site that also clears `d_safe`. *(Owner: me/next engineer, if still wanted.)*

---

## ⭐ DIAL DECLARATION (echoed, protocol §7, C2 form)
- **Dial:** the **dynamics dividend** is the only KPI. No accuracy number appears below without its dividend; no dividend appears without its bytes.
- **Laundering control:** harness-native on all 28 cells — settle-deleted launder · same-keys null (same keys, **permuted payloads**, per the Hub's ruling) · blank/empty store — **plus** one family-specific strong classical substitute, three of which cost **+0 B**.
- **Falsifies (the gym, not the CLU):** same-keys null ≥ CLU on **every** family + every family admits a classical ceiling. **Did not fire** (§4).
- **Does NOT falsify:** ≈0/negative dividend at v0 (charter §6.2); losing to a classical method on a metric-native protocol.
- **A positive cell is suspicious, not a win.** Only two cells are non-negative and both are ≈0 within noise (§3.2).

---

## 0. Flag provenance (every number in this report)

| item | value |
|---|---|
| branch / commits | `agent/experiment-engineer/memory-gym-v0` @ **`e89406c`**, 5 commits off `main @ 4160cf7` |
| artifacts | `.claude/outputs/memory-gym-v0/exp_memory_gym_metrics.json` (28 cells) · `exp_memory_gym.png` · `run_full.log` · `PREREG.md` |
| command | `PYTHONPATH=<worktree> python -u -m chlu.experiments.exp_memory_gym` (CLI: `chlu exp-memory-gym`) |
| env | main venv `/Users/user/Desktop/CHLU/.venv`, **JAX 0.9.0** (no worktree `uv sync`), **1 worktree** (`../CHLU-gym`) |
| **seeds** | **{0,1,2} on every headline cell**; single-seed cells are labelled `SINGLE SEED — not a claim` in the artifact and here |
| statistics | mean ± **sample** sd (`ddof=1`), `SE = sd/√n`, `n = 3`. Convention declared because the repo carries a cross-wave population-vs-sample split |
| geometry | `d = 4`, `m = 1`, `n_spectator = 0` (**1** on the manifold family ⇒ `dim = 6`), `ball_radius 1.0` (**0.45** on `aggregate/tight`) |
| **query law** | **`σ_q = 0.15` ISOTROPIC** (the harness's shipped value), `payload_tol 0.1`, `sep/σ_q` reported per cell (**3.42–9.06**) |
| store | learned `V_θ` = `DesignFreedomPotential(rung="free_mlp", family="atoms")`, one atom group per item slot, `atom_init_width 0.3`, `atom_init_scale 1.0`, `atom_depth_init 1e-4`, `confine (α) 0.05`. **`n_atoms` is the swept axis: 18 / 48 / 54 / 108 / 192 / 198 / 2046** |
| write | masked/local (C3-local), **300 steps**, Adam(W) lr 3e-3, wd 1e-4, `n_perturb 32`, `σ_addr 0.25`, `σ_pay 0.6`, `margin 0.15`, `barrier 0.2`, `barrier_pairs "nn"`, crowd targets = live set |
| read | `dt 0.05`, `γ_address 0.05`, `γ_read 0.02`, 400 + 800 steps, `traj_stride 8`, `kinetic_mode newtonian_learned` (M = I). **Two variants per cell**: shipped (pre-registered primary) and **annealed `[4,2,1]`** via the designed `anneal` verb, **+0 B, 1199 vs 1200 steps** |
| control | `stage_admission = True` everywhere; `d_safe_override` **0.58** on the overload family and **0.32** on `aggregate/tight` (both **deliberately out of band** — see §2), `amp_floor 0.05`, `evict_policy "depth"`, no retry (`retry_max_rounds 0`) |
| lifetimes | `stage_lifetimes = True`, **`leak = 0.06`** on the recency family only; 0 elsewhere |
| langevin / temperature | **N/A** — deterministic, `T = 0`, `p₀ = 0`, no Langevin step |
| lyapunov / wake–sleep | **N/A** — the write is `train_memory_landscape`'s static objective; consolidation is the harness's `consolidate()`, not `train.py`/`train_generative.py` |
| N94 maturity | 300 write steps ≥ 40 ⇒ **promotable**; ⚠ but see §3.1 — **at every atom budget below the shipped one the write does not converge** (final loss 0.20–0.24 vs **0.0002**), which is a *fit-budget* caveat on those cells |
| stream | per family: 6–18 offers + a **deletion demand** (naming a still-live item) + a **revisit** of an earlier address + (aggregate/recency) a near-duplicate **collision offer**; ≥ 5 **consolidation windows** per cell (140 total) |
| wall clock | **7.05 min for all 28 cells** (9–36 s each); pytest suite 1 h 08 m (§6) |

---

## 1. What I did

1. **`chlu/experiments/memory_gym.py`** (new) — `GymConfig` (+`from_mapping`/`as_flag_table`, config-driven via a `memory_gym:` YAML block, **not** `chlu/config.py`), the four families with **a written metric-native argument each in the module docstring**, the stream builder (deletion + revisit + collision + consolidation windows), the four query builders, five gym-side read-outs (including the **trajectory occupancy** ψ), three scorers, and `byte_ratio_law`.
2. **`chlu/eval/dividend.py`** (append-only; the `full-clu-harness` signatures untouched) — four gym-side callers: `knn_mean_launder` (uniform + IDW), `order_aware_launder`, `echo_launder`, `fit_shared_metric`. **Three of the four cost +0 B.**
3. **`chlu/experiments/exp_memory_gym.py`** (new) — the runner: 28-cell plan, three harness-native controls + the family substitute + `shared_metric_launder` on every value-family cell, two-sided byte ledger, the 13-monitor registry over ≥5 consolidation windows, **two read variants at shared write cost**, multi-seed aggregation, the trivial-substitute audit, JSON + 4-panel figure.
4. **`chlu/cli/experiment_cmd.py`** — `chlu exp-memory-gym` (purely additive: one parser block + one command function, in the one shared file I hold sole ownership of this wave).
5. **`tests/test_memory_gym.py`** — 18 tests, including the byte-floor theorem, the aggregate family's construction guarantee, and the documented same-keys-null degeneracy.
6. ⭐ **First-ever runs:** `shared_metric_launder` (doctrine I-12), monitor **#3**, monitor **#6** (applicable at all), monitor **#11**.

## 2. How I verified

| check | command | observed |
|---|---|---|
| plumbing | `python -m chlu.experiments.exp_memory_gym --quick` | 4 families, all controls, exit 0 |
| the run | the command in §0 | **28/28 cells, 0 degenerate, 0 errors**, 7.05 min |
| byte ledger vs closed form | `tests/test_memory_gym.py` + the artifact | `ratio == 1.4·atoms_per_item + 0.8` **in all 28 cells to 1e−9** |
| the aggregate construction guarantee | same | every target ≥ `payload_tol` from every stored payload |
| the recency label = insertion order | same | reconstructed from the controller's **public** verb log; matches |
| the shared-metric launder | same | `det M = 1.0000`; beats plain arg-min under an anisotropic law; **ties it (>97 % identical) under an isotropic one** |
| full suite | `pytest -q` (main venv, worktree) | see §6 |
| lint | `ruff check chlu/ tests/` | clean |
| read-only compliance | `git diff --stat main..HEAD` + per-file check | **zero violations**; `chlu/config.py` untouched |

⚠ **Two deliberate out-of-band settings, declared.** `d_safe_override = 0.58` (overload) and `0.32` (`aggregate/tight`). The derived radius is `2 s_max + 2.576 σ_q = 0.9864`, while 18 farthest-point sites in the unit 4-ball achieve `sep ≈ 0.97` and 8 sites in a 0.45-ball achieve `0.535` — the store would **fail its own admission gate** (doctrine I-13) and refuse every offer after the first, producing a spotless *empty* store, which is the degenerate configuration the task forbids settling into. This is the harness's own S4 convention (`0.6 × sep_expected`). `sep/σ_q` stays **in band (5.91–5.94)** on the overload family, so monitor #8's N2 is not the thing under test there.

---

## 3. Results

### 3.1 ⭐⭐ The byte axis — the Hub's hard problem, answered as a theorem plus a curve

**The theorem (pre-registered as PREREG-B1, confirmed, and sharpened).** With one atom group per item — which is *what makes the write masked/C3-local* — the ledger is

    ratio = full/launder = atoms_per_item·(dim+2)/dim + d/dim = 1.4·atoms_per_item + 0.8   (d=4, m=1)

**independent of K**, verified to 1e−9 in all 28 cells. `n_atoms` is forced to a multiple of the capacity, so `atoms_per_item ≥ 1` and

> ⛔ **`ratio ≥ 2.20×`. Matched bytes is UNREACHABLE BY CONSTRUCTION, not merely unachieved** — it would require atoms *shared between items*, which the masked write forbids. Measured minimum: **2.28×** (18 atoms, 17 live items). **No gym cell is quotable as a dividend**, and that is now a structural statement rather than a caveat.

**The curve (quote the curve, not the endpoint).** Overload family, `decode` over the full offered payload alphabet; the settle-deleted launder is **0.944 (3× load) / 1.000 (1× load) at every point on this axis**:

| byte ratio | arm | atoms/live item | n_atoms | **CLU decode** | launder | load |
|---|---|---|---|---|---|---|
| **2.28×** | `ref3` | 1.06 | 18 | **0.0972** | 0.944 | 3× |
| 5.00× | `load1x_ref3` | 3.00 | 18 | 0.1667 | 1.000 | 1× |
| 5.25× | `ref8` | 3.18 | 54 | 0.0694 | 0.944 | 3× |
| 9.69× | `ref16` | 6.35 | 108 | 0.2083 | 0.944 | 3× |
| **12.00×** | `load1x_ref8` | 8.00 | 48 | **0.3333** | 1.000 | 1× |
| 17.11× | `base` | 11.65 | 198 | 0.2593 ± 0.0668 | 0.944 | 3× |
| 45.60× | `load1x` | 32.00 | 192 | 0.4583 ± 0.0722 | 1.000 | 1× |
| **478.20×** | `load1x_shipped` | 341.0 | 2046 | **0.9722 ± 0.0139** | 1.000 | 1× |

⭐ **The anchor is the finding.** At the harness's own atom budget (2046 ≈ its 2048; ratio **478.2×** = the harness's own 478.7× at K=6) the gym's store reads **0.9722 ± 0.0139**, `acq = 1.000`, `strict = 1.000`, `δ_read = 0.0002`, `λ_min = +3.24` — and **under the annealed read it reads 1.0000 on all three seeds with the dividend exactly 0.0000 and `D = 0.000/0.042/0.042`.** That is **Prop D2a reproduced independently, on a different stream, in a different experiment**: the settle and the arg-min launder agree on essentially every query when the store works.

⭐ **The mechanism of the frontier is the WRITE, not the read** (measured, not inferred): `final write loss` **0.0002** (478×) → 0.063 (45.6×) → **0.20–0.24** (≤17×), and `λ_min(Hess V)` at the recorded sites **+3.24** (478×) → **−0.21 … −1.20** (everything else). Below ~340 atoms/item the write cannot dig a genuine minimum at every site, so the recorded addresses are **not minima** (N3 fails) — and at the 5.00× cell `λ_min = 0.1000` *exactly* `= 2α`, i.e. **the landscape is arithmetically just the confinement bowl with no wells at all.** So "beyond-capacity compression" in this architecture does not present as graceful read degradation; it presents as write failure.
⚠ **Fit-budget caveat (N94):** all sub-shipped cells ran the same 300 write steps as the shipped one, so this is a *parameter-budget* result, not a converged-fit result at those budgets; a longer write might move the curve and that is NOT RUN.

### 3.2 The dividend, per family (the acceptance criterion)

`neg_mae` = −mean|read − target| in payload units (higher is better); `acc`/`decode`/`r2` are native. **Every row carries its byte ledger and `matched=False`.**

| family / arm | metric | **full** | settle-deleted launder | **dividend** | sign | annealed dividend | same-keys null | blank | best **+0 B** substitute | ratio | sep/σ_q | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **overload/base** (3× overload) | decode | 0.2593 ± 0.0668 | 0.9444 | **−0.6852 ± 0.0668** | **neg** | −0.6852 | 0.0370 | 0.0556 | 0.2130 (knn2-IDW) | 17.1× | 5.94 | 3 |
| overload/load1x (1× ref) | decode | 0.4583 ± 0.0722 | 1.0000 | −0.5417 ± 0.0722 | neg | −0.4167 | 0.2778 | 0.1667 | 0.7083 | 45.6× | 8.97 | 3 |
| ⭐ overload/load1x_shipped | decode | 0.9722 ± 0.0139 | 1.0000 | **−0.0278 ± 0.0139** | ~0 | **+0.0000** | 0.2778 | 0.1667 | 0.7083 | 478.2× | 8.97 | 3 |
| **aggregate/base** | neg_mae | −0.5261 ± 0.0863 | −0.4472 | **−0.0789 ± 0.0620** | **~0** | −0.1426 | −0.7193 | −0.4221 | **−0.2081** (knn2-IDW) | 54.6× | 7.98 | 3 |
| aggregate/tight (`sep/σ_q` 3.6) | neg_mae | −0.5239 ± 0.0238 | −0.5324 | **+0.0084 ± 0.0348** | **~0** | −0.1174 | −0.6628 | −0.4221 | −0.2892 | 54.6× | 3.59 | 3 |
| **recency/base** | acc (chance 0.5) | 0.3019 ± 0.0679 | 0.4495 | **−0.1477 ± 0.0495** | **neg** | −0.1306 | 0.4495 | **0.3065** | **0.7764** (order-aware) | 54.6× | 7.98 | 3 |
| **manifold/base** | r2 | −0.1802 ± 0.1708 | 0.0000 | **−0.1802 ± 0.1708** | **~0** | −0.1743 | 0.0000 | −0.0001 | **1.0000** (echo) | 52.0× | 8.97 | 3 |
| manifold/ridge (blocker arm) | r2 | −1.2046 | 0.0000 | −1.2046 | neg (1 seed) | −1.2046 | 0.0000 | −0.0001 | 1.0000 | 52.0× | 9.06 | 1 |

**No cell is positive beyond 2 SE.** The two non-negative cells are `load1x_shipped` **annealed = exactly 0.0000** (`D = 0`, structurally zero — Prop D2a) and `aggregate/tight` **+0.0084 ± 0.0348** (indistinguishable from 0). ⭐ **This is the charter's expected honest starting line, measured on four independent task families.**

⛔ **The trivial-substitute audit is 0 for 4.** The **settle-deleted launder wins or ties on every family**; on the two families where the frozen launder is uninformative *by construction* (recency: it has no time column; manifold: it has one point per item), a **+0 B substitute of the same table wins instead** — insertion order **0.776** vs CLU 0.302, echo **1.0000** vs CLU −0.180. ⇒ **The CLU is not the best reader of its own byte budget on any of the four families.** The one place it beats the +0 B substitutes is `load1x_shipped` (0.972 vs 0.708) — but there the launder itself is at 1.000.

⛔ **The single most alarming row: on the recency family the CLU (0.3019 ± 0.0679) is statistically indistinguishable from its own BLANK store (0.3065), and both are BELOW the 2-way chance of 0.5.** The read is not reading the store on that family. (`acq = 0.458` there, i.e. mode #5 is live.)

### 3.3 ⭐⭐ `D` is not the dividend's magnitude — it is its variance (the result the doctrine should take)

Monitor #2's runtime pair, over all 28 cells:

| family | `D` (settle vs arg-min disagreement) | `U` (uncovered mass) | `D/U` | Prop D1 `D ≤ U` |
|---|---|---|---|---|
| overload, 3× load (7 cells) | **0.611–0.931** | 0.125–0.139 | **4.40–7.44** | ⛔ **violated** |
| overload, 1× load (5 cells) | 0.000–0.667 | **0.000** | — | inapplicable (0/0) |
| overload, shipped anchor (3 cells) | **0.000 / 0.042 / 0.042** | 0.000 | — | inapplicable; **D ≈ 0 = Prop D2a** |
| aggregate (6 cells) | 0.515–0.727 | **0.561–0.741** | **0.81–1.04** | ✅ **holds** (5/6; 1.02/1.04 are 2–4 % over) |
| recency (3 cells) | 0.569–0.625 | 0.625–0.681 | 0.84–0.94 | ✅ holds |

Two statements follow, both new:
1. **Prop D1's `sep/2` proxy fails precisely where `U → 0`** (queries near centres) and holds where `U` is large (queries *between* wells). The harness saw 1.5–2.5×; the gym reaches **7.44×**. The proxy is not merely loose, its error is *structured*. (R1.)
2. ⭐ **A large `D` is not a dividend.** `overload/ref8` has the largest disagreement in the whole gym (`D = 0.931`) and **the worst dividend (−0.875)** — the settle disagrees with the arg-min on 93 % of queries and is wrong on almost all of them. Monitor #2's `D` is therefore a **necessary-but-nowhere-near-sufficient** statistic: it bounds where a dividend *could* live but says nothing about its sign. Corollary for the controller: **#2 rightly has no restoring verb, and it should not be read as a progress signal either.**

### 3.4 Pillar (c), trajectory information — the harness's dead axis, independently replicated

`recency`, handcrafted ψ (soft time-occupancy over the phase-2 trajectory, radius `0.5 sep`) vs the settled point:

| seed | trajectory ψ | settled-point ψ | Δ | occupancy peakiness | annealed (traj / point) |
|---|---|---|---|---|---|
| 0 | 0.4306 | 0.4306 | **0.0000** | 0.685 | 0.4444 / 0.4444 |
| 1 | 0.2000 | 0.2000 | **0.0000** | 0.507 | 0.2250 / 0.2250 |
| 2 | 0.2750 | 0.2750 | **0.0000** | 0.771 | 0.2875 / 0.2875 |

⭐ **Exactly zero on every seed and every read variant**: the occupancy argmax and the nearest-centre-to-`q*` agree on **every single query**. PREREG predicted ≈+0.02 — confirmed, and more sharply than predicted. This **independently replicates `full-clu-harness`'s monitor-#10 finding that the trajectory axis is semantically dead at this geometry**, now on a task whose target is *not* a stored item and with a ψ that reads the whole phase-2 buffer. ⚠ Scope: one handcrafted ψ, one occupancy radius (pre-declared, **not tuned**). A learned ψ is `trainability-spike`'s and this is not evidence against it — it is evidence that the *signal the handcrafted ψ can see* is exhausted by the endpoint.

### 3.5 Pillar (d), manifold-valued memory — the blocker MEASURED, not asserted

- **There is no flat direction.** At 14 of 18 sites `λ_min = 0.0846–0.1000` ≈ **`2α = 0.10`** — the "unconstrained" spectator axis carries **exactly the confinement's curvature**, and the softest eigenvector's spectator participation is 0.000–0.579. So the settle drives the spectator coordinate to 0: `manifold_r2 = −0.180 ± 0.171` (0.0001 / −0.019 / −0.522), while the **+0 B echo attains 1.0000**.
- ⭐⭐ **The ridge write produces a SADDLE, not a valley.** Writing 5 collinear targets along the spectator axis into the item's own atom mask gives, at that item, `λ_min = −0.5946` with **spectator participation 1.000** — an *unstable* direction, so reads run away along it and `r2 = −1.205`. **Named blocker, now quantified: `write_loss` minimises `V` at each target independently and never constrains the connecting path to equal depth, so multi-target ridge writes buy instability rather than degeneracy.** A manifold-valued write needs an objective term on the *path* (equal-depth / zero-gradient-along-tangent), and there is no controller verb for it.

### 3.6 Monitors — four first fires, and a defect found

| # | monitor | state in the gym | evidence |
|---|---|---|---|
| 1 | overdamping | TRIP ×142 | `ρ_conv` ≫ 1e−6 whenever the write has not converged |
| 2 | settle→arg-min | **TRIP ×9** (all in consolidation windows), applicable 32/56 | never trips on a final reading (`ρ_ex` 0.81–7.44 ≫ 0.10); §3.3 |
| 3 | vacuous gate | ⭐ **FIRST FIRE EVER — ×190, both legs** | fire-rate `f = 0.000` (1×-load / manifold streams) and `f = 0.053–0.200` elsewhere; **validity `corr` −0.99 … +0.56 (sign-unstable, R3)**; utilisation **1.146 > 0.95** on `aggregate/tight@s2` |
| 4 | blank | **CLEAR, 0/56 — PASSES everywhere** | scores 0.056–0.327 vs bars 0.159–0.517 (monitor #4 never trips; the instrument does not leak) |
| 5 | addressing | TRIP ×138 | `acq` 0.132–1.000; **1.000 at the shipped anchor**, 0.13–0.51 below it |
| 6 | objective divergence | ⭐ **APPLICABLE FOR THE FIRST TIME (56/56) and TRIPS ×58** | genuine at e.g. `overload/base@s2` (slope_acq −0.214, slope_loss −0.055); ⚠ **29 of 58 are an ε-artefact (R2)** |
| 7 | mass gauge | n/a by design (pytest gauge) | — |
| 8 | certificates | TRIP ×128 | N3 fails wherever `λ_min < 0`; N2 fails on `aggregate/tight` (`sep/σ_q` 3.42–3.72 < 5.15) |
| 9 | lifetimes | TRIP ×128 | pre-declared uncleanable; the recency family's own mechanism |
| 10 | dead axis | ⚠ **not exercised by the gym** (no knob sweep; the harness owns it) — reported inapplicable, **not green** | `knob_reads` absent |
| 11 | reach | ⭐ **FIRST FIRE EVER — ×7** | `overload/base@s2` item 17, worst margin **−0.3439**; `aggregate/tight@s0` item 7, **−0.1209**. ⛔ my dedicated probe was a no-op (R5) |
| 12 | starvation | TRIP ×170 | fairness `min D/max D → 0` at reduced atom budgets |
| 13 | maturity | provenance field, never a trip | `write_steps = 300 ≥ 40` ⇒ promotable (with §3.1's fit caveat) |
| M14 | guard liveness | ⚠ **not exercised by the gym** (no canary stream; the harness owns it) — inapplicable, **not green** | — |

⭐ **The gym's own clean configuration** (`overload/load1x_shipped`) trips **only #3 and #6**, both with characterised false-trip modes, one of which (R2) is a defect rather than a signal.

### 3.7 `shared_metric_launder` — doctrine I-12's launder, run for the first time

Fitted per cell from a held-out split as `M = Cov[q − c_label]^{-1}`, normalised to `det M = 1` (**+40 B**). Eigenvalues came out **0.670–1.904** — up to 90 % from isotropy purely from finite-sample noise (32–72 fit samples) — and the launder still **ties plain arg-min exactly** on the overload family (0.9444 / 1.0000 in every cell) and is **marginally worse** on the aggregate family (−0.4563 vs −0.4472). ✅ Exactly as pre-registered: **with an isotropic query law a shared Mahalanobis metric cannot be a stronger launder, and it is robust to a badly-fitted `M`.** The tested-and-declared way to make it bite is an anisotropic query law (asserted in `tests/test_memory_gym.py`); that cell is **NOT RUN** (declared in PREREG §6 as optional).

---

## 4. The falsifier, adjudicated

> *"same-keys launder ≥ CLU on **every** gym task **and** each task admits a classical provable ceiling ⇒ the gym is metric-native in disguise ⇒ redesign."*

**It does NOT fire.** The same-keys null beats the CLU on **2 of 4** families (recency 0.4495 vs 0.3019; manifold 0.0000 vs −0.1802) and loses on the other two (overload 0.0370 vs 0.2593; aggregate −0.7193 vs −0.5261). The gym is therefore **not** metric-native in disguise on its own letter, and F2/F3/F4 each carry a written argument for why a classical arg-min is not their ceiling (§F-arguments in `memory_gym.py`'s docstring).

⚠ **But the weaker, real version of the same worry is confirmed, and it is the finding the Hub should act on:** every family is answered at least as well by a **+0 B substitute of the launder's own table** (§3.2). The gym's families are not metric-native — they are **cheaply-substitutable**, which for development-currency purposes has the same consequence. Two concrete redesign levers follow from the measurement rather than from taste: (i) F3 needs a target whose ground truth is **not** recoverable from row order *or* the metric (e.g. a count/aggregate over *history*, not over items); (ii) F4 needs a write objective with a path term before the family can measure anything at all.

---

## 5. PREREG scorecard (`.claude/outputs/memory-gym-v0/PREREG.md`, written before any cell ran)

| prediction | outcome |
|---|---|
| **PREREG-B1**: `ratio = 1.4·A + 0.8`, independent of K; matched bytes unreachable | ✅✅ **CONFIRMED to 1e−9 in all 28 cells**, and **SHARPENED**: `atoms_per_item ≥ 1` is itself architectural ⇒ the floor is **2.20×**; measured minimum **2.28×** |
| PREREG-B1(3): minimum ratio reaching decode ≥ 0.80 = **12.0×**, range [6.4×, 23.2×] | ⛔ **REFUTED, by ~40×.** At 12.0× decode = **0.333**; nothing reaches 0.80 below **478.2×** |
| PREREG-B1(4): route 2 ("overload the launder's table") is closed by arithmetic | ✅ confirmed (R4) |
| F1 dividend **negative**, point −0.30, range [−0.75, −0.05] | ✅ **CONFIRMED in sign, inside the range: −0.685 ± 0.067** (2.3× the point estimate) |
| F1 fails criterion 4 as a dividend task (declared in advance) | ✅ confirmed — the launder is 0.944–1.000 at *every* byte budget |
| F2 dividend ≈0, point **−0.02**, range [−0.25, +0.10] | ✅✅ **CONFIRMED — −0.0789 ± 0.0620**, inside the range, 0.06 from the point estimate |
| F2 mechanism: at in-band separation the settle *is* the arg-min ⇒ ≈0 | ✅ consistent: `D/U` = 0.81–0.94 with D1 holding |
| **F2-tight more negative** than F2-wide | ⛔ **REFUTED — +0.0084 ± 0.0348**, the *least* negative cell in the gym. Both sides degrade together (launder −0.447 → −0.532) |
| F3 `recency_acc` = **0.72**, range [0.50, 0.90] | ⛔⛔ **REFUTED — 0.3019 ± 0.0679, BELOW 2-way chance and indistinguishable from its own blank store (0.3065)** |
| F3 dividend **POSITIVE** +0.22, range [0.00, +0.40] | ⛔ **REFUTED — −0.1477 ± 0.0495** |
| F3's +0 B order-aware substitute = 1.000 ⇒ any positive F3 is a laundering artefact | ✅ mechanism confirmed (**0.7764**, not 1.000, because the two nearest keys to a jittered midpoint are not always the intended pair) — and moot, F3 being negative |
| F3 `acc(traj) − acc(point)` ≈ **+0.02**, range [−0.05, +0.10] | ✅✅ **CONFIRMED and sharper: exactly 0.0000 on 3/3 seeds and both read variants** |
| F4 `manifold_r2` = **0.15**, range [0.00, 0.60] | ⛔ **REFUTED — −0.180 ± 0.171** |
| F4 blocker: coercivity pulls the spectator axis to 0; no flat direction | ✅✅ **CONFIRMED QUANTITATIVELY: `λ_min = 0.0846–0.1000 ≈ 2α = 0.10` at 14/18 sites** |
| F4 echo = 1.000 at +0 B ⇒ capability, not dividend | ✅ measured **exactly 1.0000** |
| F4-ridge: higher `r2` if the ridge write takes | ⛔ **REFUTED and instructive — `r2 = −1.205`; the ridge item's `λ_min = −0.595` with spectator participation 1.000 ⇒ a SADDLE, not a valley** |
| **#3 fires for the first time** | ✅✅ ×190, on both the fire-rate and the (sign-unstable) validity leg |
| #4 blank does **not** fire | ✅ 0/56 |
| **#6 becomes applicable for the first time** and trips on F1 | ✅ applicable 56/56 and trips; ⚠ **half the trips are a defect (R2)** |
| **#11 fires for the first time** | ✅ ×7 — but ⛔ **my dedicated probe was a no-op** (R5), so the fires came from ordinary geometry |
| #2 trips-or-inapplicable on F2-wide; applicable with `D>0` on F2-tight | ◐ applicable with `D > 0` on **both**; **never trips on a final reading** (`ρ_ex` 0.81–1.04) |
| #5/#8/#9/#12 trip as listed | ✅ all four |
| the gym falsifier does **not** fire | ✅ (§4) |
| **summary verdict:** "≈0 or negative on every family once +0 B substitutes are admitted" | ✅ **CONFIRMED** — no cell positive beyond 2 SE; the two non-negative cells are exactly **0.0000** (`D = 0`) and **+0.008 ± 0.035** |

**Score: 13 confirmed (4 of them sharper than predicted) · 2 partial · 7 refuted.** Every refutation is in the same direction — **I predicted the physics would do more than it does** — which is the direction that cannot flatter the program.

---

## 6. Test suite

`tests/test_memory_gym.py`: **18 new tests, all passing** (`36 s` standalone).
✅ **Full suite on the branch: `843 passed, 0 failed` in 12 m 32 s** (`pytest -q`, worktree, main venv, JAX 0.9.0) — exactly `main`'s **825 + my 18**, no regressions. Log: `.claude/outputs/memory-gym-v0/pytest_full.log`. `ruff check chlu/ tests/` clean.
One known repo-wide hazard is pre-empted: the module carries the standard autouse `float32_dynamics` fixture (handover §7.2, the `x64-at-import` ordering trap that bit `test_clu_system` in the previous wave).

## 7. Git footprint

- **Branch** `agent/experiment-engineer/memory-gym-v0`, worktree `../CHLU-gym`, base local `main @ 4160cf7` (**unmoved during the task** ⇒ rebase is a no-op; verified).
- **Commits (5):** `7c6f7b8` gym task families · `18f8ccf` gym-side launders appended to `eval/dividend.py` · `f56e186` `exp_memory_gym` runner · `aa1121d` CLI hook · `e89406c` tests.
- **Files touched (5, all mine):** `chlu/experiments/memory_gym.py` (new) · `chlu/experiments/exp_memory_gym.py` (new) · `tests/test_memory_gym.py` (new) · `chlu/eval/dividend.py` (**+113 lines, append-only below the frozen surface**) · `chlu/cli/experiment_cmd.py` (**+61 lines, one parser block + one command function** — the shared file I hold sole ownership of this wave). **+2129 lines, 0 deletions.**
- **Read-only compliance: ZERO violations**, verified file-by-file against the task's list (`clu_system.py`, `monitors.py`, `clu_controller.py`, `exp_clu_system.py`, `memory_potentials.py`, `controller.py`, `placement.py`, `admission.py`, `exp_designed_mechanism.py`, `exp_cl_entry.py`, `cl_baselines.py`, `exp_phi_stream.py`, **`chlu/config.py`**) — all untouched.
- Results/figures live under `.claude/outputs/memory-gym-v0/`; **nothing in `results/` was committed.**
- **Not pushed, not merged.** Branch left for Hub review.

## 8. Open questions / follow-ups / risks

1. ⚠ **The gym's headline families all run below the shipped atom budget** (52–54.6× rather than 478×), because moving the byte ledger was a first-class deliverable. The consequence is measured and must travel with every family number: at those budgets `λ_min < 0` and the write loss plateaus at 0.20–0.24, i.e. **the families were measured on a store that is not fully written.** A re-run of F2/F3/F4 at the shipped budget (~478×) is **cheap (≈2 min/cell) and is the single highest-value follow-up** — it would separate "the family has no dividend" from "the store was not written".
2. **NOT RUN (declared, never a null):** the anisotropic-query-law cell for `shared_metric_launder`; F2/F3/F4 at the shipped atom budget; a longer write at reduced budgets; a deliberate #11 probe that also clears `d_safe` (R5); `trajectory_launder` (it needs a ψ that can see the address block — `trainability-spike`'s).
3. **Monitors #10 and M14 are still not exercised anywhere** — the gym has no knob sweep and no canary stream (both are the harness's). They remain **untested, not green**, across the whole of C2W1.
4. **Risk to flag:** the recency family's read is indistinguishable from a blank store. Before that family is used as development currency it needs either the shipped atom budget (item 1) or a different target (§4).
5. **For `trainability-spike`:** the point-vs-trajectory Δ is **exactly 0.0000** on a task whose answer is not a stored item, with a ψ that sees the whole phase-2 buffer. That strengthens the Hub's Stage-0 axis-liveness gate — the dead axis is not an artefact of the harness's stage geometry.
6. The occupancy radius (`0.5 sep`) and the ridge-write settings were **declared in PREREG and not tuned**; a radius sweep is a legitimate follow-up but must be pre-registered, since tuning it after seeing Δ = 0 would manufacture a laundering.

---

## Proposed handover updates (for the Hub)

**§2/§3 (architecture + config).**
- New module pair `chlu/experiments/{memory_gym,exp_memory_gym}.py` + `tests/test_memory_gym.py`, CLI **`chlu exp-memory-gym`** (`--families --arms --seeds --quick`). `chlu/eval/dividend.py` gains four **append-only** gym-side launders: `knn_mean_launder`, `order_aware_launder`, `echo_launder`, `fit_shared_metric`.
- **`GymConfig` deliberately does NOT live in `chlu/config.py`** (C2W1 ownership rule); the override path is a `memory_gym:` block in the project YAML, read directly — same convention as `CluSystemConfig`.

**§7 (known issues / live).**
- ⛔ **Matched bytes is unreachable by construction on the learned store.** `ratio = atoms_per_item·(dim+2)/dim + d/dim`, independent of K; one atom group per item forces `ratio ≥ 2.2×` at `d=4, m=1`. **No `V_θ`-vs-table dividend can ever be byte-matched under a masked write.** Retire the framing "load the store beyond the launder's table" (R4).
- ⚠ **Prop D1's `sep/2` proxy fails where `U → 0`** — up to **7.44×** (harness: 1.5–2.5×) — and **holds where `U` is large** (0.81–1.04). The error is structured, not loose (R1).
- ⚠ **Monitor #6 lacks a dead-band and trips on ~1e−17 slopes** — half its first-ever trips are artefacts (R2). `monitors.py` needs its owner.
- ⚠ **#3's validity leg does not predict drift on a learned `V_θ`** (`corr` −0.99 … +0.56, sign-unstable) (R3).
- ⚠ **The learned store's accuracy-vs-bytes curve is brutal**: `decode 0.972 → 0.458 → 0.333 → 0.097` as the ratio falls `478× → 45.6× → 12.0× → 2.28×`, launder 1.000 throughout; and the limiting step is the **write** (loss 0.0002 → 0.24, `λ_min` +3.24 → −1.20), not the read.

**§8/§10 (record).**
- **The C2W1 gym dividend at v0 is ≈0 or negative on all four openings** (best cells: **exactly 0.0000** with `D = 0` at the shipped budget under the annealed read, and **+0.008 ± 0.035** at overlapping basins). The **gym falsifier did not fire**, but the **trivial-substitute audit is 0 for 4**: a +0 B substitute of the launder's own table matches or beats the CLU on every family.
- ⭐ **Prop D2a reproduced independently** (annealed read, shipped atom budget, 3/3 seeds: full 1.0000 = launder 1.0000, `D = 0.000/0.042/0.042`).
- ⭐ **`D` is not the dividend's magnitude, it is its variance**: the cell with the largest settle-vs-arg-min disagreement in the gym (`D = 0.931`) has the *worst* dividend (−0.875). Monitor #2's `D` bounds where a dividend could live and says nothing about its sign — so it must not be used as a progress signal either.
- ⭐ **The trajectory axis is dead on a non-metric-native target too**: `acc(trajectory ψ) − acc(settled-point ψ) = 0.0000` on 3/3 seeds and both read variants — an independent replication of the harness's monitor-#10 result.
- ⭐ **Opening (d)'s blocker is now measured, not asserted**: there is no flat direction (`λ_min ≈ 2α = 0.10` on the "unconstrained" axis), and a multi-target **ridge write produces a saddle** (`λ_min = −0.595`, spectator participation 1.000), because `write_loss` never constrains the path between targets to equal depth. A manifold-valued write needs a path term and there is **no controller verb for it**.
- ⭐ **First fires, all three reportable:** monitor **#3** (×190, both legs), monitor **#6** (applicable at all for the first time; ×29 genuine), monitor **#11** (×7, worst margin −0.344). **#4 blank passes everywhere (0/56).** **#10 and M14 remain unexercised across the whole of C2W1.**
