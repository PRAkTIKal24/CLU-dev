# c2w8-well-lifecycle — experiment-engineer report

**Task + acceptance criterion (one line):** build the C2W8 stage-1 rig (CL stream on the full CLU),
the `U` telemetry and the well census, and file `census.json` with a **mechanically computed**
`stage2_unlock`; build stage 2 only if it unlocks.
**Status: partial — stage 1 COMPLETE and filed; K2 (§2.1, the trash region's first use) SHIPPED;
§2.2–2.7 DECLARED CUT, with reasons and evidence below (§6).**

⚠ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary), full detail in §7:**
**(R1)** prereg **N1 is falsified** (`P` predicted 0.15–0.45, measured **0.0208**);
**(R2)** the K1 unlock is carried **entirely by `M`, whose geometric leg never binds** — `M` equals
the same-class pair rate **exactly**, so the registered merge criterion must be re-registered before
any merge verb is built on it; **(R3)** the CL entry's binding `phi_dim >= 16` is **unreachable on
the learned-`V_theta` store** (measurably inert), which re-prices every "full CLU on the CL stream"
plan; **(R4)** §7.24's `atom_local_radius` is not merely dead, it is **unusable for a phi-addressed
stream** by construction.

⭐ **DIAL DECLARATION (echoed before the first result).**
**Dial:** lifetimes + isolation, as a full-CLU component build. ⛔ No paper number, no tier-ii
verdict, no full-CLU verdict. **Laundering control:** kNN-in-phi at matched memory carried on every
performance reading (here: diagnostic only — no claim cell was run). **Falsifies:** K1 kill (did not
fire), K3 failure, K2/K4 red. **Does NOT falsify:** losing to iCaRL/replay; a negative launder
margin (N5's registered prediction); an eroded well having no depth to restore.
⛔ Depth is not quoted as feature importance (§A23.5 ACTIVE). ⛔ "+0.510" never appears without
"−0.036 laundered" (CM-23(q)) — and no `+0.510`-class number is claimed anywhere in this report.

---

## 1. What I did

1. **`chlu/core/well_lifecycle.py` (new)** — the census instrument, built **before** the verbs it can
   kill: well states (depth **raw and decay-netted**, own-vs-foreign atom-sum split at the site,
   `lambda_min` at the **relaxed** site, measured SC-6 capture radius, site drift), a **measured**
   `theta_att`, `P`/`M` per prereg §3.3 with the two populations reported **separately**, the
   mechanical K1 rule, and `plant_item`/`flatten_unused_groups` (the designed negatives' kit).
2. **`chlu/experiments/usage_telemetry.py` (new)** — B2 at I2 grade: registered primary proxy
   `read_hits(i)`, **item-id keyed**, surviving eviction; LOO reported only beside `ICC(1,1)`,
   labelled `UNDEFINED` (no number quotable) when `ICC <= 0`. **Depth never enters `U`. No I2
   verdict is computed anywhere.**
3. **`chlu/core/controller.py`** — `touch()` also increments the item-id-keyed counter
   (decision-free; the LRU/staleness semantics are unchanged and pytest-pinned).
4. **`chlu/experiments/exp_well_lifecycle.py` (new)** — the rig: `exp_cl_entry`'s Split-MNIST
   Class-IL stream + `task1_only` phi + the kNN-in-phi launder, **imported read-only**, driven onto
   `CluSystem` with the **learned `V_theta`** as the store, run to `overdig = 2.00`, censused.
   Emits `census.json`. Per-cell: flag table (labelled non-promotable, with reason), byte ledger
   (launder included, `gamma_phi` hole line present), monitor trip state (#9/#12 named), pilot depth
   check, depth trace with the launder alongside.
5. **`chlu/config.py`** — new `ExperimentWellLifecycleConfig` group (**additive**; no existing
   default touched), wired into `load_config`/`save_config`; **`chlu/cli/experiment_cmd.py`** — one
   new `exp-well-lifecycle` command.
6. **`CluSystemConfig.gamma_phi*` + `CluSystem.trash/trash_route/trash_bytes` (§2.1, K2)** — the
   trash region's **genuine first use**: `friction_field.py` was built in C1 and referenced nowhere
   in `clu_system.py`. Shipped **OFF**, with OFF bit-identical *and* parameter-count-identical.
7. **`ERRATA-C2W8.md`** filed **before** the cells it governs (§1 `theta_att`'s arithmetic, §2 the
   merge criterion's two constants, §3 the census's address dimension **and why**, §4 label->payload,
   §5 B1 netting on this rig).

## 2. Flag provenance (every number below)

| item | value |
|---|---|
| commit | `d3a7570` (instrument) · `772c942` (rig) · `7840917` (K2); base `main @ d70898b` |
| seeds | **0, 1, 2** (census); tests seeded per-case |
| rig | `CluSystem`, learned `V_theta` (`DesignFreedomPotential`, rung `free_mlp`, family `atoms`) |
| phi | `exp_cl_entry` PCA read-in, regime **`task1_only`** (binding primary, no leakage), `phi_dim = addr_dim = 8` |
| addr_dim / payload_dim / dim | 8 / 1 / 9 · `n_atoms = 8192` (`min_atoms_base 512 · sqrt2^8`) |
| capacity (slots) / budget / **well_budget** | 16 / 16 / **8** ⇒ `overdig = 16/8 = 2.00` on every seed |
| lifetimes | `stage_lifetimes=True`, `leak=0.02`, `permanent_per_task=1` (2 protected items/seed) |
| admission | `d_safe_override = 0.88 · median-NN(task-1 phi keys)` = 0.1238 / 0.1210 / 0.1292 |
| write / read | `write_steps=300`, `lr=3e-3`, `write_margin=0.15`, `address_steps=400`, `read_steps=800`, `gamma_address=0.05`, `gamma_read=0.02`, `dt=0.05`, `kinetic_mode=newtonian_learned`, `query_sigma=0.15` |
| census instrument | `capture_dirs=16`, `capture_bisect_steps=8`, `payload_thresh = payload_tol = 0.1`, `R_cert = 2 s_max + 2.576 sigma_q` |
| payload map | `(label − 4.5)/9` (ERRATA §4) |
| `gamma_phi` | **OFF** in every census cell (`gamma_phi_hole_bytes = 0`) |
| **promotable** | **NO** — `phi_dim = 8 < 16` (ERRATA §3) and demoted write budget; carried in `census.json` as `flags.promotable=false` with its reason string |
| declared NOT-RUNs | I2 correlation test · cross-stream criterion · tier-ii verdict · full-CLU verdict · CSF3 · stage-2 §2.2–2.7 |

## 3. Stage 1 — the census (the deliverable)

**`.claude/outputs/c2w8-well-lifecycle/census.json`** — written, with `stage2_unlock` computed by
`unlock_verdict()`, i.e. by the prereg's arithmetic, not by a judgement call.

| seed | `overdig` | `n_live` | **`P`** | **`M`** | `theta_att` | depth med RAW | depth med NETTED | wall |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.00 | 16 | **0.0625** (1/16) | **0.2333** (28/120) | 2.087 | 1.659 | 1.898 | 1132 s |
| 1 | 2.00 | 16 | **0.0000** (0/16) | **0.2417** (29/120) | 1.563 | 1.239 | 1.490 | 960 s |
| 2 | 2.00 | 16 | **0.0000** (0/16) | **0.2417** (29/120) | 0.944 | 0.725 | 0.832 | 920 s |

> **K1 (mechanical): `P_mean = 0.0208`, `M_mean = 0.2389` ⇒ `stage2_unlock = TRUE`, `kill = FALSE`.**
> The rule fired on **`M` alone**; `P` is **below** the 0.05 threshold on the seed mean and is 0.0000
> on 2 of 3 seeds.

**Designed negatives (K1's own validity), both pytest-asserted and green:**
`test_census_sees_four_planted_unread_attractors` (4 planted never-read attractors ⇒ `P >= 4/n_live`)
and `test_census_sees_three_planted_near_duplicate_pairs` (3 planted near-duplicate pairs ⇒
`M >= 3/n_pairs`, and the far/payload-distinct pair is **not** admitted).

### 3.1 The pilot check first: the store is **NOT** inert here — but it does not *retrieve*
Depth after the first 3 writes: **0.742 / 0.565 / 0.744** (seed 0), **0.657 / 0.482 / 0.670**
(seed 1), **0.000 / 0.000 / 0.000** (seed 2 — wells appear only later in that seed). Final depth
medians 0.725–1.659. So this is **not** the `cluformer-pilot` inertness (0.045) at `addr_dim = 8`.

But:

* **measured capture radius = 0.000 on 15/16, 16/16 and 16/16 wells** (one well on seed 0 captured,
  at r = 0.508), while `lambda_min > 0` **everywhere** (0.79 … 8.87). This reproduces SC-6's own
  lesson at scale: *`lambda_min > 0` is necessary and NOT sufficient for a nonempty basin.*
* **site drift** (relaxed site vs recorded site) is **0.22 … 1.47**, against a median-NN key spacing
  of **0.14** — the read does not settle where the codebook says the item is.
* **self-probe: `acq` 0.086 / 0.117 / 0.078, `strict` identical, `decode` 0.0625 = `chance` 0.0625.**
  The store re-reads its own items **at chance**.
* monitors **#12 `starvation`** and **`vacuous_gate`** trip on **every** seed (refusal rate 0.000 —
  the admission gate never refused anything).

⇒ **`P` is ~0 for a mechanic-1 reason, not because everything was read**: 87.5 % of items were never
read on **every** seed (14/16), but only 1 well in 48 qualified as a *live attractor*.

### 3.2 `M` is real arithmetic on a criterion leg that never binds
Every admitted pair has `payload_dist = 0.000` **exactly** (the label->payload map gives 0 iff same
class). Independently recomputing the same-class pair count from the stream labels:

| seed | measured `n_mergeable_pairs` | same-class pairs in the admitted set | `R_cert` | median-NN spacing |
|---|---|---|---|---|
| 0 | 28 / 120 | **28 / 120** | 1.540 | 0.141 |
| 1 | 29 / 120 | **29 / 120** | 1.418 | 0.138 |
| 2 | 29 / 120 | **29 / 120** | 1.648 | 0.147 |

> **`M` ≡ the same-class pair rate, exactly, on all three seeds.** `R_cert` is 10–12× the key
> spacing, so the center-separation leg **admitted every same-class pair and refused nothing**
> (max admitted separation 1.271 vs `R_cert` 1.540). At this operating point `M` does **not** measure
> "the near-duplicate population over-digging produces"; it measures "a class-incremental stream
> contains two items of the same class", which is true by construction.

### 3.3 B1 — every depth curve raw AND netted, and the netting is exact
Netting replays the controller's own decay log (ERRATA §5). Medians move
**1.659 -> 1.898 (+14.4 %)**, **1.239 -> 1.490 (+20.2 %)**, **0.725 -> 0.832 (+14.8 %)** — the same
order as C2W6's 34 % correction, i.e. **not** negligible for any restoration claim.
C2W6's residual instrument (`exp_anti_erosion._interference_audit`, **imported, that file untouched**)
on 105 events per seed:

| quantity | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `rate_down_own_beyond_decay` (C3-locality violations) | **0.000** | **0.000** | **0.000** |
| `max_abs_own_residual_vs_decay_law` | **0.0** | **0.0** | **0.0** |
| `median_rel_drop_own` | 0.019801 | 0.019801 | 0.019801 (= `1 − e^{−0.02}`, exact) |
| `rate_up_foreign` (a neighbour crowding in) | 0.838 | 0.781 | 0.819 |

⇒ the designed decay is **exactly** what the own-leg loses; **but ~80 % of writes raise the FOREIGN
contribution at a live item's site**, and the own/foreign split shows foreign dominating (seed 2's
first six wells: own **0.00**, foreign 0.40–0.69). **The "depth" at an item's site is largely not
that item's own well** — which is also why depth *rises* over the stream (0.74 -> 1.66) while
retrieval *falls*.

### 3.4 `U` telemetry (B2) and the LOO leg
64 read events per seed (4 batches × 16 held-out Class-IL queries). `n_never_read` = 14/16 on every
seed (`frac_never_read = 0.875`); `n_unassigned` 58/62/62 (reads landing in no basin — credited to
nobody, by design). LOO `ICC(1,1)` = **0.298 / 0.873 / 0.939**, i.e. **positive on 3/3** — unlike
C2W6's negative ICC — so the leg is `status: usable`; **it was still not used for any decision**
(prereg §3.2), and no `rho(LOO)` is reported (that is the I2 test, a declared NOT-RUN).

### 3.5 Byte ledger, and the launder (diagnostic only)
CLU store **360 448 B** + codebook 512 B = **360 960 B**; kNN-in-phi launder at matched **items**
(8) = **288 B**; `gamma_phi_hole_bytes = 0`. ⚠ **This is matched-items, NOT matched-bytes**: at
matched *bytes* the launder would hold ≈ 10 000 phi keys against the store's 16 items.
Diagnostic read accuracy vs its launder over the depth trace (⛔ **not** a claim cell, no N94
compliance, no promotion): CLU 0.25 / 0.41 / 0.28 vs kNN 0.70 / 0.75 / 0.73 ⇒
**margin = −0.333 ± 0.072 (SE, n = 3 seeds)**, at matched items. Reported for the sign and the order
of magnitude only; the banked launder margin for the *designed* store remains **−0.036**, and
CM-23(q)'s three sentences travel together unchanged.

## 4. Stage 2 — K2 shipped (§2.1); §2.2–2.7 declared CUT

**K2 is built and green.** `friction_field.py` (C1) was referenced nowhere in `clu_system.py`; it is
now wired into the settle, **OFF by default**:

| K2 leg | assertion | result |
|---|---|---|
| OFF bit-identical | `test_gamma_phi_off_is_bit_identical_and_parameter_count_identical` | **green** |
| OFF parameter-count-identical | same test (`n_params(off) == n_params(shipped) == n_params(on, K=0)`) | **green** |
| (a) hole **at** a well destroys retrievability | `test_gamma_phi_hole_at_a_well_destroys_its_retrievability` | **green** |
| (b) hole **far** from every well ⇒ reads **bit-identical** | `test_gamma_phi_hole_far_from_every_well_leaves_reads_bit_identical` | **green** |
| the verb cannot silently un-ship OFF | `test_trash_route_refuses_when_the_flag_is_off` | **green** |

Two implementation facts the Hub should carry: **(i)** OFF means *no field attached at all* — even an
**empty** field is not bit-identical, because the integrator composes `1 − (1−γ)(1−γ_φ)` and
`1 − (1−g)·1.0 ≠ g` in floating point; **(ii)** the default gate is **compact**, not the field's own
`sigmoid`, because only the compact smoothstep is *exactly* zero beyond `r_k` — a sigmoid tail would
make designed negative (b) a "1e-30", i.e. a global friction change wearing a trash-region costume.
Holes are on the byte ledger (`trash_bytes = K·(dim+2)·4`).

**§2.2 merge-to-budget, §2.3 K4, §2.4 K3, §2.5 prune-to-trash, §2.6 restoration, §2.7 the claim
cells: NOT BUILT. Declared cut, with the cause, not a budget excuse:**

1. **§2.4/§2.5 (prune) has no population to act on.** `P_mean = 0.0208 < 0.05`; 0.0000 on 2/3 seeds.
   Building a prune verb here would be building a verb whose target set is empty at the only
   operating point measured — the exact failure the K1 gate exists to prevent, one leg down.
2. **§2.2/§2.3 (merge) would be built on a criterion that did not discriminate.** `M` equals the
   same-class pair rate exactly; the geometric leg refused **nothing**. A merge verb on that
   criterion is not "merge-to-budget on mechanical criteria", it is **collapse-to-one-well-per-class**
   wearing a certificate costume — the merge analogue of the depth-policy-in-a-usage-costume that K3
   exists to catch. **The criterion must be re-registered at an operating point where `R_cert` is
   commensurate with the key spacing before the verb is built.**
3. **§2.7's claim cells would measure a store that retrieves at chance** (`decode = chance = 0.0625`,
   capture radius 0.000 on 47/48 wells). A ± consolidation ablation on that rig cannot separate
   "consolidation helps" from "nothing was retrievable either way".

I did **not** override the mechanical gate: `census.json` records `stage2_unlock = true` and that is
what it says. What I am reporting is that **two of the three preconditions the stage-2 verbs need are
measurably absent**, which is a scoping decision for the Hub, not for me.

## 5. How I verified (commands + observed output)

* `uv`-equivalent env: main venv reused per protocol §4 (`PYTHONPATH=<worktree>
  /Users/user/Desktop/CHLU/.venv/bin/python …`), **no worktree `uv sync`** ⇒ no package drift.
* `python -m chlu.experiments.exp_well_lifecycle --seeds 0,1,2` — 3012 s total; the console output is
  quoted verbatim in §3 (`P`/`M`/`overdig`/`theta_att`/depth per seed, then the K1 line).
* `python -m chlu.experiments.exp_well_lifecycle --quick` — smoke, 53 s, writes a real `census.json`.
* `pytest tests/test_well_lifecycle.py tests/test_usage_telemetry.py` — **25 passed** (10 census +
  7 telemetry + 3 rig/e2e + 4 K2 + 1 x64 regression).
* `pytest tests/test_twins.py tests/test_usage_telemetry.py tests/test_well_lifecycle.py` —
  **33 passed** (the x64-ordering pairing that caught the two bugs in §8.1).
* `pytest` over every file I touched or wired into (`test_clu_system`, `test_soft_certificate`,
  `test_friction_field`, `test_clu_controller`, `test_controller_mvp` + mine) — **125 passed**.
* `pytest tests/test_config.py` — **7 passed** (the mutate-every-group round-trip, with the new group).
* CLI parser build + dispatch check: `exp-well-lifecycle --seeds 0,1 --quick` parses to
  `cmd_exp_well_lifecycle`.
* `ruff check chlu/ tests/…` — **All checks passed**.
* Full suite: see §8 (count arithmetic).
* Pricing probes (before committing to the operating point): `.claude/scratch/c2w8/timing.py`,
  `timing16.py`, `atomdist.py` — the ERRATA §3 tables.

## 6. The rig limitation that re-prices the plan (ERRATA §3, filed BEFORE the cells)

| `addr_dim` | `n_atoms` | depth after 3 designed-site writes | self-probe strict | nearest atom to a unit site | `exp(−r²/2s²)` |
|---|---|---|---|---|---|
| 4 | 2 048 | 0.480, 0.601 | 0.812 | 0.294 | 6.18e-01 |
| 8 | 8 192 | 0.443, 0.342, 0.458 | 0.333 | 0.738 | 4.86e-02 |
| 12 | 32 768 | 0.297, 0.345, **0.000** | 0.250 | 1.252 | 1.65e-04 |
| **16** | 131 072 | **2.1e-9, 6.8e-10, 0.000** | **0.000** | 1.483 | **4.98e-06** |

**At the CL entry's binding `phi_dim >= 16` the learned store is INERT.** Cause (arithmetic, not
conjecture): atom centers are drawn `N(0, atom_init_scale=1)` in `dim = addr_dim+1`, so the nearest
of *all* atoms recedes as `~sqrt(dim)` while `atom_width` stays 0.3, and the write gradient's
`exp(−r²/2s²)` underflows. The `min_atoms_base·c^d` floor buys atoms geometrically but does **not**
bring the nearest one closer. ⚠ `atom_local_radius` cannot rescue this for a phi-addressed stream:
it needs per-group localization targets **at init**, and a stream does not know its addresses until
write time (and §7.24 records it dead in the `LearnedVStore` path anyway).

## 7. Downstream reconciliation list (needs an owner)

| # | what changed | sites that must be reconciled |
|---|---|---|
| **R1** | **N1 falsified**: `P` predicted **0.15–0.45**, measured **0.0208** (0.0625/0.0/0.0). The prune leg of K1 is dead at this operating point. | `PREREG-C2W8.md` §6 N1/N3 scorecard; the C2W8 row of §A21; any wave summary quoting a prunable population |
| **R2** | The K1 **unlock is carried entirely by `M`**, and `M` ≡ the same-class pair rate exactly (`R_cert` 10–12× the key spacing; the geometric leg refused nothing). N2 is numerically "confirmed" (0.239 ∈ 0.10–0.35) but **degenerately**. | `PREREG-C2W8.md` §3.3 + §6 N2; **ERRATA §2 must be superseded before any merge verb ships**; §A20.3(d) design note |
| **R3** | The CL entry's binding `phi_dim >= 16` is **unreachable on the learned-`V_theta` store**. Every "full CLU on the CL stream" plan re-prices; the census (and any stage-2 cell on this rig) is **non-promotable**. | charter §A21 C2W8 row; `PREREG-C2W8.md` §4; `PREREG_CL_PHI` §7's interaction with the learned store; anything assuming the w25 CL entry ports to full CLU as-is |
| **R4** | §7.24's `atom_local_radius` is not merely "dead in the store path" — it is **unusable for a phi-addressed stream** (needs init-time group centers). | handover §7.24 |
| **R5** | New measured fact: **depth at an item's site is mostly foreign** (own 0.00 vs foreign 0.40–0.69 on seed 2's first six wells) and `rate_up_foreign` ≈ 0.8. Any future "depth" reading on this rig must state own-vs-foreign. | handover §7.28 (the effective-`s`/depth ruler thread); C2W6 erosion instruments |

## 8. Git footprint

* **Worktree** `../CHLU-c2w8`, **branch `c2w8-well-lifecycle`** off `main @ d70898b` (as the task
  file names it — note it is *not* the protocol's `agent/<type>/<slug>` form; the task file wins).
* Commits (verified present from the main repo, `git log main..c2w8-well-lifecycle`):
  * `d3a7570` — census instrument (`core/well_lifecycle.py`) + `U` telemetry
    (`experiments/usage_telemetry.py`) + `core/controller.py` touch counter + 17 tests
  * `772c942` — the rig (`experiments/exp_well_lifecycle.py`), `config.py` group (additive),
    `cli/experiment_cmd.py` command (additive), +3 tests
  * `7840917` — K2: `gamma_phi` wired into `CluSystem` (OFF), `trash_route`/`trash_bytes`, +4 tests
  * `4875002` — two **x64 ordering bugs** the full suite caught (§8.1), +1 regression test
* **Files touched (all owned by this task):** `chlu/core/well_lifecycle.py` (new),
  `chlu/core/clu_system.py`, `chlu/core/controller.py`, `chlu/experiments/usage_telemetry.py` (new),
  `chlu/experiments/exp_well_lifecycle.py` (new), `chlu/config.py` (additive only),
  `chlu/cli/experiment_cmd.py` (additive only), `tests/test_well_lifecycle.py` (new),
  `tests/test_usage_telemetry.py` (new).
* **Not touched:** every file on the DO-NOT-TOUCH list, including `exp_anti_erosion.py` (its
  residual instrument is **imported**), `exp_cl_entry.py`/`exp_phi_*` (imported read-only),
  `core/friction_field.py` (wired, not modified), and all C2W6/C2W7 files.
* **No conflicts.** The shared main checkout was never edited (`git -C <main> status` clean
  throughout). Nothing pushed; no PR.
* **Test count arithmetic:** baseline at `d70898b` = **1410** (handover), of which **2** are the
  network-hitting `tests/test_download_concurrency.py` (deselected here) ⇒ **1408 run**.
  New tests **+25** (`tests/test_well_lifecycle.py` **18**, `tests/test_usage_telemetry.py` **7**).
  **1408 + 25 = 1433.** First full-suite run: **1 failed, 1377 passed** (`-x`; the failure was
  **mine** — §8.1). After the fix: see §8.2.

### 8.1 The full suite caught two bugs of mine that the file-level run could not

Both are §7.23's **ordering** hazard — green in isolation, red the moment an x64-enabling module ran
first — and both are now fixed (`4875002`) with one **function-scoped** regression test (module-scoped
x64 fixtures are themselves the N211 hazard, so the remedy is deliberately not a fixture):

1. `flatten_unused_groups` mutated `np.asarray(jax_array, dtype=float)`. Under `jax_enable_x64` that
   conversion is **zero-copy** and returns a **read-only** view ⇒
   `ValueError: assignment destination is read-only`. Now `np.array(..., copy=True)`.
2. The C2W8 trash field was built in the ambient default dtype. Under x64 a float64 `gamma_phi`
   **promotes `p` inside the damping** and `lax.scan` rejects the carry
   (`input float32[d]` vs `output float64[d]`). The field is now pinned to **float32** — the dtype
   the read path actually runs in — at construction and after every `trash_route`.

I am reporting these because they are exactly the class of defect the protocol says to report: the
K2 flag would have shipped "green" on a file-level run and reddened the suite for the next agent.

### 8.2 Full-suite status after the fix

```
$ pytest -q --deselect tests/test_download_concurrency.py     # on 4875002
1433 passed, 2 deselected, 24 warnings in 2299.12s (0:38:19)
```
**GREEN: 1433 passed / 0 failed** = 1408 pre-existing (1410 baseline − 2 network tests) + **25 mine**.
Rebase onto the named base `main` is a **no-op** (`main` has not moved from `d70898b`); `origin/main`
was never used (§7.21).

## 9. Open questions / risks

1. **Should stage 2's merge verb be built at all before the criterion is re-registered?** (R2). My
   recommendation: no — re-register `R_cert` (or the operating point) so the geometric leg binds,
   then re-run the census cheaply (the instrument is built; a census is ~16 min/seed).
2. **The rig retrieves at chance.** Before any C2W8/C2W9 claim cell on `CluSystem` + phi streams,
   somebody must own "why does a store with depth 1.6 have capture radius 0.000". My measurement
   points at *foreign-atom domination at the site* + site drift ≫ key spacing, i.e. the wells are
   real but not **where** or **whose** the codebook thinks. That is a write-side question (§A26.6).
3. **`theta_att` > median depth** on all three seeds. That is a legitimate reading of the registered
   rule (ERRATA §1) but it means the attractor test is currently dominated by the capture leg. If
   the Hub prefers a different floor, it is a prereg-level decision, not an implementation one.
4. **Cost, declared:** the census is ~16 min/seed at `addr_dim=8` (dominated by per-write Hessians in
   `_c3_check`, which is O(n_live) per write, and by the LOO leg). I cut **nothing** from the census
   (3 seeds, full instrument); I cut the **stage-2 build** for the reasons in §4.
5. I ran the census **once** per seed; no seed was re-run after seeing results.

## Proposed handover updates (for the Hub)

* **§7 new entry — 7.30 [OPEN, program-wide]** *The learned `V_theta` store is inert at
  `addr_dim >= 12`* — measured 2026-08-06 (`c2w8-well-lifecycle`): depth `2.1e-9` at `addr_dim=16`
  vs 0.44–0.60 at `addr_dim<=8`; cause is the atom-init scatter (`nearest atom ~sqrt(dim)`,
  `exp(−r²/2s²) = 5e-6` at d=16) and the `min_atoms_base·c^d` floor does not fix it. **Any plan that
  runs the full CLU on a `phi_dim >= 16` read-in is currently unbuildable.** `atom_local_radius` does
  not help a stream (needs init-time group centers).
* **§7 amend 7.24** — add: `atom_local_radius` is additionally **unusable for phi-addressed streams**
  by construction, so "one-line forward in `clu_system.py`" would not make the N98 lever reachable
  for the CL rig.
* **§7 new entry — 7.31 [OPEN]** *On the `CluSystem`+phi rig, `lambda_min > 0` everywhere while the
  measured capture radius is 0.000 on 47/48 wells, and the site drift (0.22–1.47) exceeds the key
  spacing (0.14).* Self-probe `decode` = chance. Depth at a site is mostly **foreign** atoms.
* **§3 config** — new group `experiment_well_lifecycle` in `chlu/config.py` (additive, defaults
  preserve all existing behaviour) and new `CluSystemConfig.gamma_phi*` flags (**default OFF**,
  bit-identical and parameter-count-identical when off). New CLI command **`chlu exp-well-lifecycle`**.
* **§2 architecture** — two new modules: `chlu/core/well_lifecycle.py` (census/K1 instrument) and
  `chlu/experiments/usage_telemetry.py` (B2 usage `U`); `chlu/core/friction_field.py` is **no longer
  unused** — it has its first consumer in `CluSystem.model()`.
