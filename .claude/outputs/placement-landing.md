# placement-landing — experiment-engineer report

Task + acceptance criterion: land PGCP as `placement="canonical"` + the new `Controller.delete` verb, scrub the evicted row, and drive `mia-decay-measurement` §2's post-eviction `AUC(z_hole)` from **0.99985 → ~0.5**. **Status: done.**

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). FIVE items.**
> 1. **The deletion claim now has a code path and a number.** `AUC(z_hole)` 0.99985 → **0.5000 ± 0.0000**, every statistic 0.5000, `TPR@FPR1% ` 1.000 → **0.000**, and IN-after-delete is **byte-equal to never-written in 3 072/3 072 worlds**. Sites saying "no deletion-flavoured sentence is defensible" can be updated — **but only inside the scope in item 2.** Owner: `doc-curator` (N99 update block, R1 wording), gated on `deletion-prior-art`.
> 2. ⛔ **The scope is BELOW CAPACITY, and the boundary is now measured, not asserted.** On the un-inflated mia geometry (7 lattice cells, 8 offers ⇒ overflow) canonical placement is **not** exact: `AUC(n_live) = 1.000` and `AUC(s4) = 0.914` (byte-equal fraction 0.000). The P2 waitlist is **not built**. Any claim sentence must carry "operating below capacity or under set-function eviction". Owner: whoever writes the CL/controller paper section.
> 3. **The rematch per-offered number is now MEASURED under the real two-phase Verlet read** (theorist H5's scope gap closed): canonical `on_sized` K=64 = **61/64 admitted, per-admitted 1.0000, per-offered 0.9531**; ×1.05 sizing = **64/64, per-offered 1.0000**; incumbent refuse-and-relocate on the same seeds = **43/64, per-offered 0.6719**. "0.953" may now be quoted **as measured**. Owner: same section author.
> 4. **`controller-mvp` §3(b)'s decay-demo defect (mia D2) is NOT fixed here** — out of scope of this task; still open. Owner: `experiment-engineer` (next wave). Likewise **D3** (payload-dependent lifetimes) still needs a theorist ruling.
> 5. **The evict scrub (mia D1) is landed and is measurably neutral**: the relocate arm re-measured on the scrubbed code reproduces every published mia §2 number to 4–5 dp (0.99985 / 0.8114 / 0.6015 / 0.7828 / 0.7599 / 0.7106). Owner: `doc-curator` (the "`evict` is not erasure" line in §7 can be marked RESOLVED).

---

## ⭐ DIAL DECLARATION (echo, protocol §7)
- **Dial:** lifetimes + admission (R1's structural underpinning).
- **Laundering control:** the TTL-dict / canonical-dictionary claim-structure comparison. **Stated plainly: with PGCP, `Controller.delete` has exactly the claim structure of `del d[k]` on a canonical dictionary — that IS the deletion claim.** The differentiators (continuous amplitude law + Thm-4 commutation, the spacing certificate, packing economics) are named as properties *around* the claim, never smuggled into it.
- **Falsifies:** post-eviction `AUC(z_hole)` does not fall to ~0.5 under the real two-phase read; or the real read at exact-`d_safe` spacing underperforms the theorist's gradient-flow relaxation badly enough to force lattice inflation. **Neither fired:** 0.5000 ± 0.0000, and per-admitted = 1.0000 under the real read.
- **Does NOT falsify (declared in advance, all observed):** delete-time churn (measured 2.836 moves/delete at full 61-cell load); LRU staying outside the claim (now a hard error); per-offered 0.9531 < 1.0 at the un-inflated sizing.

## Flag-provenance table

| item | value |
|---|---|
| base | local `main` @ **`ff85573`**; branch `agent/experiment-engineer/placement-landing`, worktree `../CHLU-placement-landing` |
| code commit used for ALL measurements | **`e2d44cd`** (tip). Controller/placement/store code identical since `1cc55cb`. ⚠ `placement_mia.json`'s `meta.commit` reads `ff85573` because that run's **cwd** was the main repo (`PYTHONPATH` pointed at the worktree); the *code* executed was the worktree's. `placement_mia_hiprio.json` correctly records `e2d44cd` |
| venv | **main venv** (protocol §4, no worktree `uv sync`), **JAX 0.9.0**, eqx 0.13.4 |
| harnesses | `.claude/outputs/placement-landing/{mia_placement.py, rematch_cell.py, analyze.py}`, `.claude/scratch/placement-landing/cascade_cost.py`; adversary imported **verbatim** from `mia-decay-measurement/mia_harness.py` |
| PREREG | `.claude/outputs/placement-landing/PREREG.md`, written **before** any harness existed |
| seeds | acceptance: **0, 1, 2** (× 8 targets × 128 paired worlds = 24 per-example values, 3 072 worlds/arm); rematch: **0, 1, 2**; cascade: 777 (theorist protocol) + 0 (200 key sets) |
| store | `AtomStorePotential(dim=3, capacity=8, α=0.02, s=0.35, s_pay=s, κ=1.0)` — mia/controller-mvp values, **no learning anywhere** |
| controller | `d_safe = 4.4·s = 1.540`, `budget = capacity`, `amp = 1.0`, `n_relocation_candidates = 400` (relocate arm), `evict_policy="depth"` (canonical arms — LRU now forbidden), `leak = 0` in the acceptance run |
| geometry | acceptance: `R_mia = radius_for_capacity(8) = 2.28695` (**7 cells**) vs `R_sized = radius_for_cells(8) = 2.68163` (**13 cells**); rematch: `R(64) = 6.46846` (**61 cells**), ×1.05 = 6.79189 (**73 cells**) |
| read (shipped, unmodified) | two-phase, `dt 0.05`, `γ_address 0.05 × 400` → `γ_read 0.0 × 800`, tail 0.25, 8 subsamples, 16 queries/item, `σ_θ = 0.15`, `σ_p = 0.05` |
| training config | **N/A — no CHLU training anywhere** (no lyapunov / langevin / anchor / epoch flag is in effect; N94 does not apply) |
| runtimes | acceptance 638 s (3 arms) + 761 s (high-priority variant); rematch 26 s; full suite 831 s |

---

## 1. What I built

| # | thing | where |
|---|---|---|
| 1 | **`chlu/core/placement.py`** (286 lines, pure numpy) — `splitmix64`, `u01`, `hash_point`, `prio`, `hex_cells`, `n_cells_for`, **`radius_for_cells`**, `CanonicalPlacer` (probe orders, `_replace_from` suffix greedy, `insert`/`delete` with move log + drop log, `layout`/`centers`/`min_spacing`), `canonical_layout` | new file |
| 2 | **`Controller(placement=…, lattice_radius=…)`** — `"relocate"` default is byte-for-byte the w23 path; `"canonical"` routes `offer` to `_offer_canonical` (admission = "got a cell", no per-write spacing test) and re-packs the store into canonical priority order after every op (`_canonical_sync`, the H7 byte-identity condition) | `chlu/core/controller.py` |
| 3 | ⭐ **`Controller.delete(item_id)`** — Theorem 2: remove + fix-up cascade, survivors' `center`/`slot` updated, returns the move count. Hard-errors under `"relocate"` instead of pretending to be exact there | same |
| 4 | ⭐ **The array scrub (mia D1)** — `AtomStorePotential.evict` now zeroes `centers[slot]`/`payloads[slot]` as well as `active`/`amps` | `chlu/core/memory_potentials.py` |
| 5 | **Guards** — canonical + `evict_policy="staleness"` (LRU) **raises**; so do canonical without `lattice_radius`, canonical + `allow_relocation=False`, canonical + `peer_addresses_fn`, canonical + `addr_dim≠2`, and a duplicate live `item_id`. `"depth"` is allowed (item-intrinsic) | same |
| 6 | **`tick`** now self-evicts spent wells **by item id**, not by slot (a canonical removal re-packs slots; no-op under `"relocate"`) | same |
| 7 | **Rematch arm** `canon_sized` + config knobs `run_canonical_placement` (default **False**) and `canonical_radius_mult` (default **1.0**) — shipped chart unchanged | `chlu/experiments/exp_controller_mvp.py`, `chlu/config.py` |
| 8 | **`tests/test_placement.py`** — 26 tests, T1–T5 with `tobytes()` asserts at zero tolerance | new file |

**Design choices I made (task under-specified, smallest reasonable assumption, stated):**
- **The anchor `g(κ)` is the item's OFFERED address**, not `hash_point(key)` (which stays as the fallback when no address is given). A controller store is content-addressed; using the offered address keeps "write near where you asked" and still makes placement a function of the record set alone. Quantization cost = at most the covering radius `d_safe/√3 = 0.889`; it is **reported per write** in the decision row (`quantization`) and is **unmeasured as an accuracy cost** for φ-derived addresses — hence guard #5 (canonical + `allow_relocation=False` raises).
- **`decision` labelling**: probe index 0 ⇒ `"admit"`, later cell ⇒ `"relocate"`, no cell ⇒ `"refuse_spacing"`, so `intervention_rate` keeps meaning in the rematch chart.
- **LRU guard fires at construction**, not at `delete()` time: a store that *can* LRU-evict is history-dependent whether or not you delete from it.
- **`_canonical_sync` is a full O(n) `with_item` rebuild** (the theorist explicitly allowed this; in-place slot moves are an optimization). Cost at MVC-0 sizes: the 3 072-world acceptance arm takes 279 s vs 130 s for relocate.

---

## 2. ⭐ The acceptance test — `mia-decay-measurement` §2, history column

24 per-example values (3 seeds × 8 targets), 128 paired worlds each. AUCs **direction-calibrated** (`max(AUC, 1−AUC)`) exactly as in mia; raw AUCs in the JSON. Data: `placement_mia.json`, tables via `analyze.py`.

| statistic (history OUT) | mia published (relocate) | **relocate, re-measured here** | **`canon_sized` (below capacity)** | `canon_native` (at overflow) |
|---|---|---|---|---|
| **`z_hole`** = dist(`c_i`, nearest live site) | 0.99985 ± 0.00070 | **0.99985 ± 0.0007** (TPR@1% 1.000) | **0.50000 ± 0.0000** (TPR@1% **0.000**) | 0.57633 ± 0.0120 |
| **`n_live`** | 0.8114 ± 0.0349 | **0.81137 ± 0.0349** | **0.50000 ± 0.0000** | **1.00000 ± 0.0000** |
| `s1` value-return (query) | 0.6015 ± 0.0915 | **0.60149 ± 0.0915** | **0.50000 ± 0.0000** | 0.55411 ± 0.0269 |
| `s2` address-capture (query) | 0.7828 ± 0.1435 | **0.78277 ± 0.1435** | **0.50000 ± 0.0000** | 0.58442 ± 0.0785 |
| `s4` white-box address depth | 0.7599 ± 0.0644 | **0.75990 ± 0.0644** (TPR@1% 1.000) | **0.50000 ± 0.0000** | **0.91409 ± 0.0813** (TPR@1% 0.989) |
| `s5` white-box full `V` | 0.7106 ± 0.0796 | **0.71064 ± 0.0796** (TPR@1% 1.000) | **0.50000 ± 0.0000** | 0.58705 ± 0.0661 |
| **paired-placement column (sanity)** | 0.5000 ± 0.0000 | **0.5000 ± 0.0000** (all stats) | **0.5000 ± 0.0000** | **0.5000 ± 0.0000** |
| **IN-after-removal byte-equal to never-written** | — | 0.0000 | **1.0000 (3 072/3 072 worlds)** | 0.0000 |
| retention after removal | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Three readings, in order of importance.**
1. **The acceptance criterion is met, and by the strongest possible mechanism.** It is not that the adversary's statistic became weak — the two stores are **bit-identical**, so every present and future statistic is exactly tied. `AUC(z_hole)` 0.99985 → **0.5000**, `AUC(n_live)` 0.811 → **0.5000**, `TPR@FPR1%` 1.000 → **0.000**.
2. **My harness reproduces the published relocate numbers to 4–5 decimal places on all six statistics.** That is the control that says the swap, not the re-implementation, produced the change — and it doubles as proof that **the D1 scrub moves no measured number** (those numbers were published pre-scrub).
3. **At overflow the claim genuinely fails, exactly where the theorist said it would (§4b).** With 7 cells and 8 offers a background item is refused in the IN world and does not counterfactually return when the target is deleted ⇒ 6 live vs 7 (`AUC(n_live) = 1.000`, deterministic) and the cell nearest `c_i` is empty in one world and full in the other (`AUC(s4) = 0.914`). **This is not a defect of the build; it is the un-built P2 waitlist, and it is why the scope sentence is load-bearing.**

### 2b. The cascade actually fires — the strong version of the test
`prio(-1)` is the **lowest** priority among the acceptance run's keys `{-1, 0…6}`, so the target is always placed last and its deletion moves nobody (0.000 moves/delete). To rule out a trivial pass I re-ran `canon_sized` and `canon_native` with the target keyed `-5` (the **highest** priority of the set) — every survivor is now placed *after* the target and its deletion re-runs the greedy over all of them (`placement_mia_hiprio.json`):

| | moves/delete | byte-equal | `z_hole` | `n_live` | `s1` | `s2` | `s4` | `s5` |
|---|---|---|---|---|---|---|---|---|
| `canon_sized`, target = highest priority | **mean 1.132, max 5** | **1.0000** | **0.5000** | **0.5000** | **0.5000** | **0.5000** | **0.5000** | **0.5000** |
| `canon_native` (overflow), highest priority | mean 1.681, max 5 | 0.0000 | 0.5760 | **1.0000** | 0.5606 | 0.5757 | **0.9106** | 0.5737 |

Exactness survives a live displacement cascade. *(One artefact: `retention_post = 0.0007` in this cell — the retention scorer is indexed by the target's **pre-delete slot**, which after re-packing holds a moved survivor whose payload occasionally matches within `payload_tol`. It is a scorer indexing artefact, not retention.)*

---

## 3. ⭐ The rematch cell (theorist §6.4) — under the REAL two-phase Verlet read

`controller_line(K=64, arm=…)`, seeds 0/1/2, shipped `evaluate_items` strict criterion (own basin among live sites **and** `|v − a| < payload_tol`). Data: `rematch_cell.json`.

| arm | cells | **admitted** | **per-admitted** | **per-offered** | min spacing |
|---|---|---|---|---|---|
| **`canon_sized`, mult 1.00** | 61 | **61 / 64 = 0.953, σ = 0** | **1.0000 ± 0.0000** | **0.9531 ± 0.0000** | **1.540000** |
| **`canon_sized`, mult 1.05** | 73 | **64 / 64**, σ = 0 | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** | **1.540000** |
| `on_sized` (incumbent RR) | — | 43 / 64 = 0.672, σ = 0 | 1.0000 ± 0.0000 | 0.6719 ± 0.0000 | 1.5433 ± 0.0037 |

- **H5's scope gap is closed.** The theorist's per-admitted 1.0000 was gradient-flow relaxation; under the shipped two-phase Verlet read at *exactly* `d_safe` spacing it is **also 1.0000** (61 items × 16 queries × 3 seeds). **The lattice constant does not need to inflate; the packing win is not repaid.**
- **0.953 is now a measured retention number** (per-offered), and per N91 it travels with per-admitted 1.000 — never alone.
- The incumbent reproduces `controller-mvp`'s 42.8/64 (43 here on 3 seeds; they measured 42.8 ± on 5). **+18 items, +42 % over refuse-and-relocate, and the resulting store is exactly deletable.**

---

## 4. PREREG scorecard

| # | registered | measured | verdict |
|---|---|---|---|
| A1 `AUC(z_hole)`, `n_live`, `s1/s2/s4/s5` history = 0.5000 ± 0.0000 | all six | **0.50000 ± 0.00000, all six** | ✅ exact |
| A1 `TPR@FPR1%` = **0.000** exactly (registered *against* the task text's "~0.01", with the tie argument) | | **0.0000** on every statistic | ✅ **confirmed; the task's "~0.01" is the looser reading** |
| A1 paired column 0.5000 | | **0.5000 ± 0.0000** | ✅ |
| A1 byte-equality of IN-after-delete vs never-written | | **1.0000 (3 072/3 072)**; also 1.0000 with a cascade firing | ✅ |
| A2 `AUC(n_live)` ≥ 0.95 at overflow | point 1.00 | **1.00000 ± 0.0000** | ✅ |
| A2 `AUC(z_hole)` ≥ 0.85 at overflow | point 0.95 | **0.576** | ❌ **falsified as stated.** I predicted the leak would sit in the hole statistic; the fix-up cascade *fills* the freed cell with a re-placed survivor, so the hole moves away from `c_i`. The overflow leak is real but it lives in `n_live` (1.000) and `s4` (0.914), not in `z_hole` |
| Rematch: 61/64 admitted, zero variance | | **61, σ = 0** | ✅ |
| Rematch: per-admitted ≥ 0.98 (point 1.000) under the real read | | **1.0000** | ✅ |
| Rematch: per-offered 0.953, band [0.935, 0.960] | | **0.9531** | ✅ |
| Rematch: mult 1.05 → 64/64, per-offered ≥ 0.98 | | **64/64, 1.0000** | ✅ |
| `min_spacing_live` = 1.54 ± 1e−6 | | **1.540000** (float32 store; float64 placer 1.54 ± 2.2e−16) | ✅ |
| T1 24/24; T2 all orders; T3 interleavings; T4 mid-decay; T5 | | **all bit-identical, 0 mismatches at 0 tolerance** | ✅ |
| Packing `N_cells(R(K)·1.05) ≥ K`, K ∈ {16,32,64,128} → 19/37/73/139 | | **19/37/73/139** | ✅ |
| Cascade full load mean 2.84 (band [2,4]), max ≤ 15 | | **2.836, median 3, max 7, 8.2 % zero-move** (theorist protocol, replicated exactly); half load **0.219** vs their 0.22 | ✅ **port verified to 3 dp** |
| Scrub: max \|ΔV\| = 0.0 exactly | | **0.0** over 64 random queries; and the whole relocate arm reproduces mia's published numbers to 4–5 dp | ✅ |
| LRU guard raises | | raises `ValueError` | ✅ |

**Score: 14 confirmed, 1 falsified (A2's `z_hole`, in a direction that makes the mechanism clearer, not the claim weaker).**

**Erratum on my own PREREG §0:** the derivation table says "R = 2.2869 × 1.35 = 3.087 → **13 cells**" — 3.087 actually gives **19** cells (13 is the count at R ≈ 2.668). The prediction ("below capacity ⇒ exact") is unaffected. I then ran A1 at the **production** sizing rule `radius_for_cells(8) = 2.68163` (13 cells, mult 1.173) rather than the illustrative 1.35, i.e. **less** inflation than pre-registered.

---

## 5. Extra measurements (not pre-registered, flagged as such)

**Cascade cost, broader estimator.** The theorist's H6 uses **one** key set (seed 777) and deletes each of its 61 keys. Averaging instead over **200 independent key sets** (one random victim each) on the same 61-cell lattice gives **mean 3.865, median 4, max 10, 5 % zero-move** (32/61 load: 0.425; 16/61: 0.115). So `2.84` is one realization; the honest headline is **"~3–4 survivor moves per delete at full lattice load, ~0.2–0.4 at half load, and ~0 below half"**. Both protocols are in `.claude/scratch/placement-landing/cascade_cost.py`; the theorist's is reproduced to 3 dp, so this is a sampling-breadth difference, not a port discrepancy.

**The claim-structure control, run and reported as required.** With PGCP, `Controller.delete(i)` on this store has *exactly* the claim structure of `del d[k]` on a canonical dictionary — bit-identical to the never-written state, nothing more. What the dict does not have, named as what they are and **not** part of the deletion claim: the continuous amplitude law with the Thm-4 commutation (verified here: `test_T4_mid_decay_delete_leaves_survivors_bit_identical`, survivors at `exp(−0.35·5)` exactly), the spacing certificate (`min_spacing = 1.540000` by construction, neighbour leak `6.3e−5`), permanence and decay coexisting in one potential, and the packing economics (61 vs 43 at K=64).

---

## 6. The scoped paper sentence (theorist §4, verbatim — ⛔ CITATION-GATED)

> *"Placement in the store is canonical — a deterministic function of the live item records and the store geometry alone — so store-level deletion is exact: removing an item reproduces, bit for bit, the store that holds the remaining records, with each survivor's scheduled decay and permanence unaffected (deletion and decay provably commute). The claim covers stores operating below capacity or under set-function (priority/attribute-based) eviction; recency-based eviction is intrinsically history-dependent and is excluded. This is a store-level guarantee only: the frozen encoder and any residue of past writes in a learned landscape are separate channels, measured separately; we do not claim certified (ε,δ) unlearning."*

⛔ **Novelty wording is GATED on `deletion-prior-art` (web-scout, this wave).** The discrete skeleton (priority-displacement, strongly-history-independent hashing) is **not** novel; the contribution is exactness in a continuous designed landscape with decay/permanence coexisting, plus the spacing certificate. ⛔ "certified" is banned program-wide; no "unlearning", no "privacy guarantee". **Store-level scope only** — φ and learned-landscape residue are separate channels.

**What is now measurable and safe to say (with the scope clause):** *"under canonical placement the post-deletion membership signal falls from AUC 0.99985 (TPR 1.000 @ FPR 1 %) to 0.5000 ± 0.0000 on every statistic we can compute, because the two stores are byte-identical in 3 072/3 072 worlds."* **What must NOT be said:** anything about the overflow regime, LRU, φ, or a learned V.

---

## 7. How I verified

```
# unit
PYTHONPATH=. .venv/bin/python -m pytest tests/test_placement.py -q   -> 26 passed
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q                    -> 716 passed, 17 warnings, 831 s
.venv/bin/python -m ruff check chlu/ tests/                          -> All checks passed!
# measurement (cwd = worktree or main repo, PYTHONPATH = worktree)
python .claude/outputs/placement-landing/mia_placement.py            -> placement_mia.json (638 s)
python .../mia_placement.py --arms canon_sized,canon_native --target-id -5
                                                                     -> placement_mia_hiprio.json (761 s)
python .claude/outputs/placement-landing/rematch_cell.py --K 64 --seeds 0,1,2 -> rematch_cell.json (26 s)
python .claude/scratch/placement-landing/cascade_cost.py             -> H6 replication
```
`ruff format` is **not** run: the repo is not format-clean (82 files would reformat), so running it would emit a huge foreign diff. No NaN, no divergence, no non-finite value in any statistic; every number above is re-derived from a saved JSON by `analyze.py`, not transcribed from stdout.

## Git footprint

Branch **`agent/experiment-engineer/placement-landing`** (worktree `../CHLU-placement-landing`, base local `main` `ff85573`; verified from the main repo with `git log main..agent/experiment-engineer/placement-landing`). **Not pushed. 4 commits:**

| hash | subject | files |
|---|---|---|
| `9706d75` | add `chlu/core/placement.py`: PGCP canonical placement | +286 |
| `a8322ce` | scrub the evicted row in `AtomStorePotential.evict` (mia D1) | +15/−2 |
| `1cc55cb` | `Controller`: `placement="canonical"` + the `delete` verb (Thm 2) | controller +271/−5, tests +385 |
| `e2d44cd` | rematch cell: `canon_sized` arm in `exp_controller_mvp` | config +10, exp +28/−3 |

Files touched: `chlu/core/placement.py` (new), `chlu/core/controller.py`, `chlu/core/memory_potentials.py` (`evict` only), `chlu/config.py` (2 additive fields in `ExperimentControllerMvpConfig`), `chlu/experiments/exp_controller_mvp.py` (`controller_line` arm, arm list, plot colour, saved-config keys), `tests/test_placement.py` (new). **No conflicts; nothing outside scope; rebase onto local `main` is a no-op (base unmoved).** ⚠ `chlu/config.py` and `exp_controller_mvp.py` are the likely collision points with other w26 engineers — my hunks are purely additive (two dataclass fields; one `elif` arm + three one-line touches).

## Open questions / follow-ups / risks

1. ⚠ **The P2 waitlist is the difference between "exact below capacity" and "exact, period".** It is ~20 lines (keep refused records in a side dict; re-run admission by priority on any delete) and it would take `canon_native` from `AUC(n_live) = 1.000` to 0.5. Recommend tasking it — the overflow regime is the realistic one for a bounded store.
2. **Address quantization is unmeasured.** Canonical placement moves every write onto a lattice cell (≤ 0.889 away at `d_safe = 1.54`). Harmless when addresses are allocator-chosen (measured: per-admitted 1.0000), **unknown** when the address is `φ(x)` and carries similarity — the w25 CL entry uses `allow_relocation=False` precisely for that reason, so I made that combination raise. Needs a measurement before canonical placement touches the CL path.
3. **Cost.** `_canonical_sync` is O(n) `with_item` calls per op ⇒ the canonical acceptance arm ran 2.1× slower than relocate at n=8. At n=64 the rematch is still 3–6 s/seed, but a store with thousands of items needs the in-place slot move.
4. **`Controller.evict_item` under canonical** goes through the cascade too (it must), so the class-balanced CL policy would silently get exact-deletion semantics if someone flipped the flag — currently impossible, because `allow_relocation=False` raises. Flagged so it stays impossible by accident, not by luck.
5. **mia D2/D3 remain open** (see reconciliation item 4).

---

## Proposed handover updates (for the Hub)

1. **§7-CURRENT — mark mia D1 RESOLVED.** *"`AtomStorePotential.evict` now scrubs `centers`/`payloads` as well as `active`/`amps` (`a8322ce`). Verified neutral two ways: `max |ΔV| = 0.0` over random queries with `active` masked either way, and the full `mia-decay-measurement` §2 relocate column re-measured on the scrubbed code reproduces every published number to 4–5 dp."*
2. **N99 — the update block can now say the rule is LANDED and MEASURED, with the scope.** *"w26 `placement-landing`: PGCP is in `chlu/core/placement.py`; `Controller(placement="canonical", lattice_radius=R)` + the new `Controller.delete(item_id)` verb. Acceptance test (the `mia-decay-measurement` harness, 3 seeds × 8 targets × 128 paired worlds): post-deletion history-OUT `AUC(z_hole)` **0.99985 → 0.5000 ± 0.0000**, `AUC(n_live)` **0.811 → 0.5000**, every query/white-box statistic 0.5000, `TPR@FPR1%` **1.000 → 0.000**, IN-after-delete byte-equal to never-written in **3 072/3 072** worlds (and still 0.5000/byte-equal when the deletion fires a survivor cascade, mean 1.13 moves). ⛔ Scope: **below capacity or set-function eviction**. On the un-inflated 7-cell geometry with 8 offers the store is NOT exact — `AUC(n_live) = 1.000`, `AUC(s4) = 0.914` — because the P2 waitlist is not built. LRU + canonical is a hard error."*
3. **§1.6 / R1 — the rematch numbers are now real-read measurements.** canonical `on_sized` K=64: **61/64 admitted (deterministic), per-admitted 1.0000, per-offered 0.9531** under the shipped two-phase Verlet read; ×1.05 sizing → **64/64, 1.0000**; refuse-and-relocate on the same seeds → **43/64, 0.6719**. The theorist's H5 scope gap ("gradient-flow, not the real read") is **closed** — the lattice constant does not inflate.
4. **§3 config table — two new (default-preserving) knobs:** `experiment_controller_mvp.run_canonical_placement = False`, `experiment_controller_mvp.canonical_radius_mult = 1.0`. New `Controller` kwargs `placement="relocate"`, `lattice_radius=None`.
5. **Sizing-rule correction for anyone using the theorist's "×1.05":** it holds for K ∈ {16,32,64,128} but **not** small K (K=8 needs ×1.17, K=4 more). Use `chlu.core.placement.radius_for_cells(k, d_safe)`, which grows until the cells exist.
6. **Cascade-cost wording:** the theorist's `2.84 moves/delete` is one key-set realization (reproduced here to 3 dp under his protocol). Over 200 key sets the same lattice gives **3.87**. Quote it as *"~3–4 at full load, ~0.2–0.4 at half"*.
7. **Quotation guard, updated:** "0.953" **may** now be quoted as measured per-offered retention under canonical placement (with per-admitted 1.000 and the flags in §Flag-provenance). The **overflow** numbers must never be quoted as the deletion claim.
