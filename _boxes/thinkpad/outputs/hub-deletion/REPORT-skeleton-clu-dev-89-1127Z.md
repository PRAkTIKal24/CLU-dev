# `hub/deletion-2` — exact structural deletion on the generative store
**FORGIS_14 (Windows 11, AMD64, CPU) · branch `hub/deletion-2` @ `945a3e9` (base/iter2) · seed 0**
**Relaunch START_UTC `2026-09-17T11:13:33Z` · 12 h limit → `23:13:33Z`**
Running log: `hub-deletion/LOG.md` · Pre-registration: `hub-deletion/PREREG-deletion-cell.md`
(+ AMENDMENTS A1, A2) · Ledger: `hub-deletion/RESULTS-LEDGER.md`

> **STATUS: IN PROGRESS.** Step 1 is complete and passed. The deletion cell is running. Results
> sections below are marked `[PENDING]` until their artefacts exist; ⛔ no number appears here that is
> not re-derivable from `hub-deletion/runs/<tag>/{build,eval,rival}.json` and `preds.npz`.

---

## 1. What this hub set out to do, and the trap it was warned about
**Target.** Exact structural deletion of one factor value (and one conjunction) from the CLU's
generative store, on four conditions: unrecoverable from every composition · every other factor
preserved · byte-exact provenance · the rivals given the same deletion.

⛔ **The trap, quoted from the charter:** *store load-bearing **by construction** is a method-section
fact, never a headline*, and *"the rival has no delete operator" is worth nothing*.
⭐ **This report therefore does not claim either.** `knn_proto` is given an exact delete here — by
refit, which is canonical for it by construction — and the byte-exactness of the CLU's rebuild is
reported in the method section, where it belongs, because once determinism holds it is a theorem and
not a finding. **The headline this hub stakes itself on is COLLATERAL**, which cannot be true by
construction either way.

## 2. The mechanism — ported, ⛔ not invented
*Retention You Can Predict, Scope and Delete*, §2.3 / App. F / App. G.8: deletion is exact when the
store layout is a **canonical function of the live item set and the fixed geometry**
(history-independent placement). Implemented as `chlu/experiments/exp_gen_deletion.py`:

```
build(live) :  X      = frozen_cache[live]                       ⛔ never re-encoded
               corr   = channel_correlation(X[:16384])
               labels = grouping_labels(corr, K=5, auto->ward)    -> the channel partition
               store  = train_store(init_store(X, wells, M=64), X)   1200 steps, batch 128
               codes  = address_codes(store, X)
               book   = fit_address_book(codes, factor_tuples(live), cardinalities)
delete(S, P) := build( live(S) \ rows(P) )
```
**Cost: O(n) rebuild per delete**, accepted openly — see the cost line (§9).
⚠ **Caveat carried over verbatim:** *store-level guarantee only; the frozen encoder is a separate
channel; ⛔ no certified (ε, δ) unlearning is claimed.*

⭐ **The delete is a predicate over factor tuples**, so a factor value (`f == v`) and a conjunction
(`f1 == v1 ∧ f2 == v2`) are the **same code**, and the rivals take the **identical** predicate — read
off the CLU build's own artefact and sha256-checked against it, so the two arms cannot drift apart.

### 2.1 The read that is scored, and why it is not the shipped one
**Request-addressed:**
`request (a factor tuple) → address_requests(book) → argmax mode per well (K·log₂M = 30 bits) →
read_from_codes(store) → FROZEN decoder → FROZEN external probe.`
⛔ Nothing on that path touches the feature cache. Two reasons this and not `exp_gen_store`'s
feature-addressed `read`: (i) a feature-addressed read is **handed the answer in its input**, so it
cannot answer *"is `v` unrecoverable from every request naming it"*; (ii) it is the **only frame in
which `knn_proto` exists at all**, so it is the only frame in which the charter's rival comparison is
possible. The feature-addressed read is reported **beside** it as a companion, because the difference
between them separates *"the store lost `v`"* from *"the addressing lost `v`"*.

### 2.2 ⭐ The re-encode hazard is structurally unreachable here, not waived
`build_unit.py` / `eval_unit.py` / `rival_unit.py` **never construct an encoder**. Features come only
from `gen_bridge.features_for_rows`, which **raises** on any row the frozen cache does not hold; the
extension path (`gather_features`) is never called. Each unit additionally asserts a coverage
shortfall of exactly **0** over every row it touches and records
`"coverage_shortfall": 0, "encoder_constructed": false`. Seed 0's cache is `scope="full"`
(480,000 rows = the whole grid, `features_sha256 d6728f01…`, `encoder_param_hash 98d32602…`).
⛔ So `ULP_BOUND` / `FRAC_BOUND` were not touched, `will_extend=False` was never passed with a
shortfall, and `exp_gen_store.py` has a zero-line diff. There is no encode path in this hub to gate.

### 2.3 What `store_digest` covers, and why the existing hash was not enough
`store_digest = sha256( well_channels ‖ param_hash(modes) ‖ sha256(log_prior ‖ log_cond) )`.
⚠ `gen_bridge.param_hash` hashes an `eqx` module's **array leaves only**. Two things would have slipped
past it and both are part of the layout the mechanism claims is canonical: `GenStore.well_channels` is
an `eqx.field(static=True)` **tuple**, and the **address book** is not part of the store module at all —
yet it is fitted on the live rows' *labels*, i.e. it is precisely where a deleted value could survive.
⭐ Strictly stronger than the hash that already existed; ⛔ nothing was loosened.

---

## 3. ⭐⭐ STEP 1 — `train_store` determinism at real scale: **PASS**
*"Make `train_store` deterministic given the live row set, and PROVE it before anything else. Two
independent builds from the same live set must hash identical on this machine."*

| | process | started | wall | `store_digest` |
|---|---|---|---|---|
| `det_A` | sibling session `clu-dev-1e` | 11:03:55Z | **554.9 s** | `d0eacf3b9c2a10d0…` |
| `det_B` | this session | 11:13:36Z | **728.3 s** | `d0eacf3b9c2a10d0…` |

Identical live sets (`live_sha256 fa6106698d411b64…`, `n_live 32,768`), asserted rather than assumed.
**All 22 stored arrays bitwise identical** — `well_0..4` (the modes), `chan_0..4` (the partition),
`codes` (32768×5), `cond_0..5` (the address book), `log_prior`, `labels`, `live` — re-hashed
**independently of `store_digest`**, so the verdict does not rest on the digest agreeing with itself.
`param_hash 1bb1edb5…`, `book_hash 57bc8a31…`, `labels_sha256 27b49454…`,
`well_sizes [303, 305, 573, 501, 366]`.
`loss_first = 47.4907341003418` and `loss_final = 32.049007415771484` on both — full float32 precision.
Batching recorded: `train_batch=128, train_steps=1200, n_corr=16384, n_modes=64, k=5, tau 1.0→0.1`.

⭐⭐ **Why this is stronger than a repeat run.** The two builds differed by **31 % in wall time**
(554.9 s vs 728.3 s) because they ran under different contention from the three peer hubs. Identical
bytes out of differently-scheduled executions is evidence the build does not depend on thread timing —
which matters specifically because `gen_store.py:131`'s `(A.T @ A)` on (16384, 2048) feeds
`scipy.cluster.hierarchy.linkage`, a **greedy sequential merge** in which one flipped ULP flips a merge
and returns a wholly different channel partition. The partition came back identical.

**⚠ Scope of the claim, stated rather than implied.**
- Determinism is claimed **on this machine, in this session, under this XLA configuration** — ⛔ not
  across machines.
- **Measured, and it corrects my own earlier assumption:** the build runs **72 OS threads** and ~**2.9
  cores** (CPU 2,058 s / ~12 min). `OMP/MKL/OPENBLAS_NUM_THREADS=4` are exported and recorded but
  **XLA-CPU's intra-op pool ignores them**. The 4-thread budget was met by scheduling, ⛔ not by
  enforcement, and is reported that way. `XLA_FLAGS` was deliberately **not** pinned mid-cell, because
  changing the numeric environment would make the controls incomparable to the deletions.
- Cross-device (measured by `clu-dev-1e`): the **channel partition reproduces exactly** against the Mac
  manifest (`n_dead 71`, `n_live 1977`, `average` rejected at `[1973,1,1,1,1]`, `ward` accepted,
  `well_sizes [303,305,573,501,366]`) and `loss_first` is bit-identical — but `loss_final` differs by
  **+1.27 %**. ⭐ **Layout is ISA-invariant; trained contents are not ⇒ byte-exact provenance is a
  WITHIN-MACHINE property.** Every §4 hash claim inherits exactly that scope.
- ⛔ `smoke_A`/`smoke_B` are **not** cited as this gate: they predate the digest that includes
  `book_hash` and would be a pass for a weaker hash than the one in the tree.

**Falsifier §5.5: CLEAR.** The cell was authorised at 11:26:07Z.

---

## 4. Provenance — the hash equalities  `[PENDING]`
## 5. The pre-deletion control (falsifier 2)  `[PENDING]`
## 6. Unrecoverability  `[PENDING]`
## 7. ⭐⭐ COLLATERAL — the headline  `[PENDING]`
## 8. The conjunction — can a compositional store forget `(object_hue=5 ∧ shape=2)`?  `[PENDING]`
## 9. Cost line  `[PENDING]`
## 10. Verdict: solved truthfully / not solved  `[PENDING]`

---

## A. Pre-deletion levels — recorded before any deletion, because they frame everything
| arm | `eval_a` (1024) | `dev` (4096) | bytes |
|---|---|---|---|
| `clu` (request-addressed) | 0.5913 | 0.6003 | **524,288** |
| `knn_proto` | 0.7233 | *(pending)* | **475,136** |
| `store_removed` (laundering floor) | 0.1208 | 0.1217 | 524,288 |
| feature-addressed companion | 0.5913 | 0.6017 | — |

⛔⛔ **The rival is ahead of the arm and smaller.** On this cell, at these bytes, **the CLU does not
generate better than the byte-matched lookup table** — it is ~13 points behind at 90.6 % of the bytes.
⭐ This is stated at the top of the results because it is the most important context for the collateral
number and it would be easy to let it disappear behind a favourable Δ. It does **not** invalidate the
collateral object, which is a *paired change*; it does mean the floor-effect controls (AMENDMENT A1/A2 —
relative change, churn correction, and retention conditional on pre-deletion correctness) are
load-bearing rather than decorative.

## B. Attribution
⚠ A concurrent sibling session, `clu-dev-1e`, was driving this same hub for part of the window (see
`INCIDENT-two-agents-clu-dev-1e.md` and `LOG.md` 11:19:50Z). After direct contact it stood down and
handed over. Its contributions, named rather than absorbed: **`det_A`** (half the determinism gate);
**`AUDIT-01-determinism-sources.md`** (static audit of the build path, which corrected a factual error
in my own code comment); **`INHERITED-FACTS.md`** (the predecessor briefing, including the corrected
free floor on B); **`REVIEW-01-prereg-and-analysis-clu-dev-1e.md`** (which found the unequal-weighting
defect in my pooled statistic, supplied the two-bias analysis, the retention estimand and the
replacement pre-deletion bar, and verified my BCa implementation numerically against `scipy`).
