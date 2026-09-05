# deletion-waitlist-stiffness — experiment-engineer report

Task + acceptance: **(A)** the P2 waitlist ⇒ canonical deletion exact **at overflow**
(`AUC(n_live)` 1.000 → 0.500, byte-equality 0.000 → 1.000), quoted across a **load sweep**;
**(B)** option (d) behind a default-OFF flag + the `mia-decay` §1/§3(b)/§5 re-run **including
`R₅₀`**. **Status: done** — both criteria met; 2 pre-registered predictions falsified (B3 as
stated; the theorist's `τ_y`), 1 w26 published cell shown to be a harness artefact.

> ⛔ **DOWNSTREAM RECONCILIATION LIST — needs an owner (protocol §5 corollary). FIVE items.**
> 1. ⭐ **The exact-deletion claim loses its capacity qualifier.** With the waitlist the
>    post-delete store is byte-identical to the never-held-it store at **every load
>    measured, 2→12 offers into a 7-cell lattice** (0.29×–1.71× lattice capacity):
>    `AUC` = **0.5000 ± 0.0000** on all six statistics, byte-equal **1.0000** (3 072/3 072
>    worlds per cell), 3 seeds × 8 targets. **Exact replacement scope sentence in §2.4** —
>    the Hub relays it to `r1-positioning-pass`; I edited no doc.
> 2. ⛔ **One w26 `placement-landing` cell is a HARNESS ARTEFACT and must be re-stated.**
>    Its `canon_native` row (`AUC(n_live)` 1.000, `AUC(s4)` 0.914, byte-equal 0.000, target
>    id `-1`) came from deleting a **stale slot**: `prio(-1)` is the *lowest* of `{-1,0..6}`,
>    so at 8 offers the target itself is the key that fails to seat, `rec` is empty, and the
>    harness fell through to `store.evict(tslot)` on slot 0 — a *background* row. Re-measured
>    with an unconditional `delete()`: that cell is **0.5000 on every statistic, byte-equal
>    1.0000, even without the waitlist** (`target_seated_frac = 0.000`). The genuine overflow
>    failure is the **high-priority-target** cell (`-5`): 1.0000 / 0.9106 / 0.000, which my
>    p1 arm reproduces to 4 dp. Owner: `doc-curator` (any site quoting the overflow numbers).
> 3. ⛔ **"Payload-independent lifetimes" must be stated as an EFFECT SIZE, not a
>    correlation.** Under the gate at `g₀ = amp_floor` the correlation with `aᵢ²` only moves
>    **−0.846 → −0.667** (my B3 predicted \|r\| < 0.30 — **falsified**), but the spread
>    collapses: retention std at the floor **0.274 → 0.016**, worst codeword (`a = +1`)
>    **0.398 → 0.972**, min **0.007 → 0.925**. The ordering by `aᵢ²` survives; the *damage*
>    does not. Owner: `doc-curator` (mia §5 / §7 D3 wording) + theorist.
> 4. ⛔ **The theorist's read-length law is wrong for the shipped read.**
>    `τ_y = η/(κ(g₀+A))` is the overdamped form, but the shipped read phase has
>    **`γ_read = 0.0`**: the payload coordinate *oscillates*, and the measured period tracks
>    **`T = 2π/√(κ(g₀+A))`** to 3–4 s.f. at every point (19.100 vs 18.945; 24.650 vs 24.645;
>    26.550 vs 26.551), log-log slope **−0.5264** vs −0.5 (√ law) and −1.0 (overdamped).
>    Owner: theorist (`readout-channel-theory` §3.3) + whoever quotes `τ_y`.
> 5. ⭐ **`R₅₀` SURVIVES the gate** — the lifetimes dial keeps its adversary-caveat-free
>    differentiator: **1.135 → 0.771** at `g₀ = amp_floor` (baseline 1.146 → 0.752; TTL dict
>    constant 0.75–0.77). Owner: Hub (§1.6 / the N108 companion line).

---

## ⭐ DIAL DECLARATION (echo, protocol §7)
- **Dial:** **isolation/deletion** (Part A) and **lifetimes** (Part B).
- **Laundering control:** (A) a flat datastore row-delete is exact *by construction*; the
  only claim here is that the physical store **matches** it at overflow — no sentence in
  this report has the store beating a dict on deletion exactness. (B) **N108 stands**: the
  gate changes nothing about white-box detectability (`AUC(s4)` = 1.000 at every amplitude,
  every arm); the differentiator measured here is **retrieval geometry** (`R₅₀`), not
  privacy.
- **Falsifies:** (A) the waitlist fails to take `AUC(n_live)` 1.000 → 0.500 with byte-equality
  restored at 3 seeds — **did not fire** (0.5000 ± 0.0000, byte-equal 1.0000). (B) the `R₅₀`
  contraction disappears under the gate at `g₀ = amp_floor` **and** the graded variant does
  not restore it — **did not fire** (ratio 1.472 vs baseline 1.524).
- **Does NOT falsify (declared in advance, observed):** losing to a dict/TTL on any AUC axis
  (`AUC(s4)` = 1.000 everywhere); the step-shaped on-site retention at `g₀ = amp_floor`
  (0.9904 at the last amplitude — the trilemma's stated price); `R₅₀` being *larger* than
  baseline at small `A` under the gate (0.771 > 0.752).

## Flag-provenance table

| item | value |
|---|---|
| base | local `main` @ **`082d095`** (post-w26); branch `agent/experiment-engineer/deletion-waitlist-stiffness`; **worktree `../CHLU-waitlist`** (mandatory; `../CHLU-r2dsweep` was live on the same base) |
| **code commit for ALL measurements** | **`8731557`** (verified in every JSON's `meta.commit`). The later `2cac7b9` changes only the dtype a *waitlisted* amplitude is rounded through (`float32` → the store's dtype); in the default configuration used by every harness (no `jax_enable_x64`) it is provably a no-op, and no harness re-run is needed |
| venv | **main venv** (protocol §4, no worktree `uv sync`), **JAX 0.9.0**, eqx 0.13.4 |
| harnesses | `.claude/outputs/deletion-waitlist-stiffness/{waitlist_mia.py, gate_mia.py, osc_period.py, analyze.py}`; adversary, world construction, queries, statistics and the read imported **verbatim** from `mia-decay-measurement/mia_harness.py` |
| PREREG | `.claude/outputs/deletion-waitlist-stiffness/PREREG.md`, written **before either harness existed** (the shipped-code edits existed; no measurement had been taken) |
| seeds | **0, 1, 2** × 8 targets × **128** paired worlds = 24 per-example values, **3 072 worlds per cell** (identical draws to mia/`placement-landing`: `default_rng(4242+seed)`, `PRNGKey(1_000_003·(seed+1)+w)`) |
| store | `AtomStorePotential(dim=3, capacity=max(8, offers), α=0.02, s=0.35, s_pay=s, κ=1.0)`; **no learning anywhere** (N94 does not apply — no lyapunov/langevin/anchor/epoch flag exists in these runs) |
| controller (A) | `d_safe = 4.4·s = 1.540`, `budget = capacity`, `amp = 1.0`, `leak = 0`, `evict_policy="depth"`, `placement="canonical"`, `lattice_radius = R_mia = 2.28695` ⇒ **7 cells**; arms **`p1` = `waitlist=False`** (w26 rung P1) and **`p2` = `waitlist=True`** (w27 default) |
| controller (B) | shipped **relocate** allocator (`n_relocation_candidates = 400`), so §1/§3(b)/§5 are like-for-like with the published mia numbers; only `V`'s payload term and the read length change |
| gate arms (B) | `base` (`payload_gate=False`) · `g05` (`True, g₀=0.05 = amp_floor`) · `g005` (`True, g₀=0.005`) · `g05x2/g05x4/g005x4` (same, `read_steps ×2/×4`), `payload_eps = 1e-6` |
| read (shipped, unmodified) | two-phase, `dt 0.05`, `γ_address 0.05 × 400` → **`γ_read 0.0` × 800**, `tail_frac 0.25`, 8 subsamples, 16 queries/item, `σ_θ = 0.15`, `σ_p = 0.05`; `payload_tol = 0.1` |
| AUC convention | **direction-calibrated per example** (`max(AUC, 1−AUC)`) then averaged — exactly mia/`placement-landing`; raw AUCs are in the JSONs |
| data | `waitlist_mia_head.json` (402/373/400/530 s), `waitlist_mia_sweep.json`, `gate_mia_g05.json` (964 s), `gate_mia_g005.json` (2 253 s), `gate_mia_len.json` (921 s), `osc_period.json`; every table below is re-derived by `analyze.py`, nothing transcribed from stdout |
| machine | shared with `r2-d-sweep-close` (load average 190–300 throughout); ≤3 concurrent jobs held |

---

# Part A — the P2 waitlist

## A.1 What I built (≈70 lines of behaviour, 2 files)

| # | thing | where |
|---|---|---|
| 1 | `CanonicalPlacer(..., waitlist=False)`: an unseated key keeps `pos[k] = None` instead of being popped; `_replace_from` now records `entered_last_op` (None → cell) and `unplaced_last_op` (cell → None); `placed_keys()`/`waiting_keys()`; `layout`/`centers`/`min_spacing`/`keys` consider **placed** keys only; `insert()` returns "got a cell", not "still known" | `chlu/core/placement.py` |
| 2 | `Controller(..., waitlist=True)` (canonical only) + `waiting: {id → ItemRecord}`, `waiting_amps`, `n_waiting`, stats `waitlisted`/`reseated` | `chlu/core/controller.py` |
| 3 | **Both-directions reconciliation in `_canonical_sync`**: a record whose key lost its cell goes to `waiting` (with its current depth); a waiting record whose key took a freed cell re-enters at the depth it carried | same |
| 4 | `delete()` is legal for an **offered-but-unseated** item (a no-op on the store — which *is* the never-held-it counterfactual); returns `was_waiting` | same |
| 5 | `tick()` ages waiting records through the **same per-tick arithmetic** the live wells get and expires them at `amp_floor` (Theorem 4: amplitudes are item-intrinsic) | same |
| 6 | `_amp_cast` — round a waiting amplitude through the **store's** dtype (float32, or float64 under `jax_enable_x64`) | same |
| 7 | config `experiment_controller_mvp.canonical_waitlist = True` | `chlu/config.py` |

**Design choices (task under-specified; smallest reasonable assumption, stated):**
- **Budget eviction is a real removal, not a waitlisting.** Only *lattice* refusals wait. A
  budget-evicted item leaves the offered set. (Consequence, untested: with `budget < n_cells`
  a freed budget slot is immediately taken by the highest-priority waiting key, so `n_live`
  stays at `budget` — priority semantics, but unmeasured. My runs all have `budget ≥ n_cells`.)
- **A re-seated item's depth is the one its record carried**, ticked identically to a live
  well — not `base_amp·exp(−leak·age)` recomputed in one shot, which differs in the last ULP
  and breaks byte-identity. This is what makes A/`T4`-with-waitlist a *bit* test.
- **Duplicate-id guard extended to waiting items** (offering an id that is waiting raises).

## A.2 ⭐ The acceptance test (8 offers, 7 cells, reads ON)

3 seeds × 8 targets × 128 paired worlds = 3 072 worlds per cell. `p1` = no waitlist,
`p2` = waitlist. `paired-placement` control = **0.5000 ± 0.0000 on every statistic in every
cell** (all four cells), and post-delete retention = **0.0000** everywhere.

| arm | target id | seated | **byte-equal** | `n_live` | `z_hole` | `s4` | `s5` | `s1` | `s2` | moves/del |
|---|---|---|---|---|---|---|---|---|---|---|
| `p1` (w26) | **−5** (highest prio) | 1.000 | **0.0000** | **1.0000 ± 0.0000** | 0.5760 ± 0.0145 | **0.9106 ± 0.0841** | 0.5737 | 0.5606 | 0.5757 | 1.68 / 5 |
| **`p2` (w27)** | **−5** | 1.000 | **1.0000** | **0.5000 ± 0.0000** | **0.5000 ± 0.0000** | **0.5000 ± 0.0000** | **0.5000** | **0.5000** | **0.5000** | 1.68 / 5 |
| `p1` | −1 (lowest prio) | **0.000** | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.00 / 0 |
| `p2` | −1 | **0.000** | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.00 / 0 |

**Three readings.**
1. **The acceptance criterion is met at the cell where the mechanism actually fires.**
   `AUC(n_live)` **1.0000 → 0.5000 ± 0.0000**, `AUC(s4)` **0.9106 → 0.5000**, byte-equal
   fraction **0.0000 → 1.0000 (3 072/3 072)**, by the strongest mechanism: the two stores
   are *bit-identical*, so every present and future statistic is exactly tied.
2. **My `p1` arm reproduces `placement-landing`'s high-priority `canon_native` row to 4 dp**
   (their 1.0000 / 0.9106 / 0.5760 / 1.681 moves; mine 1.0000 / 0.9106 / 0.5760 / 1.68) —
   the control that says the *waitlist*, not the re-implementation, produced the change.
3. ⛔ **The published low-priority cell was an artefact** (reconciliation item 2). With the
   correct unconditional `delete()`, `target_id = −1` at 8 offers is exact **in both arms**,
   because the target is the key the lattice cannot seat (`seated = 0.000`) — there is
   nothing to leak. Under `p1` that same situation is a **silent forget**: `delete(-1)`
   raises `KeyError` (the controller no longer knows the item was ever offered). Under `p2`
   it is a legal delete of a waiting record. *The w26 harness hid this by evicting a stale
   slot; the failure it measured is real at `−5` and spurious at `−1`.*

## A.3 ⭐ The load sweep — "quote the curve, not the endpoint"

Offers = 1 target + (n−1) background into the **same 7-cell** lattice, i.e. load
**0.29× → 1.71×** the lattice capacity. Reads OFF except at 8 offers (statistics shown are
the allocator-trace/white-box ones, which are the load-sensitive ones).

| offers | 2 | 4 | 6 | **8** | 10 | 12 |
|---|---|---|---|---|---|---|
| **`p2` (waitlist), byte-equal** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **`p2`, `AUC(n_live)` / `s4` / `z_hole`** | **0.5000** | **0.5000** | **0.5000** | **0.5000** | **0.5000** | **0.5000** |
| `p1` (no waitlist), byte-equal | 1.0000 | 1.0000 | 1.0000 | **0.0000** | **0.0000** | **0.0000** |
| `p1`, `AUC(n_live)` | 0.5000 | 0.5000 | 0.5000 | **1.0000** | **1.0000** | **1.0000** |
| `p1`, `AUC(s4)` | 0.5000 | 0.5000 | 0.5000 | **0.9106** | **0.9106** | **0.9149** |
| `p1`, `AUC(z_hole)` | 0.5000 | 0.5000 | 0.5000 | 0.5760 | 0.5760 | 0.5750 |
| moves/delete (both arms) | 0.14 | 0.52 | 1.18 | 1.68 | 1.68 | 1.69 |
| *incumbent refuse-and-relocate `AUC(z_hole)` (`carried-remeasurements`, for reference)* | *0.6715* | *0.9165* | *0.9961* | *0.99985* | *—* | *—* |

(target id **−5**; the −1 column is 0.5000/byte-equal 1.0000 in **both** arms at every load,
with `seated = 1.000` up to 6 offers and `0.000` from 8 — table in `analyze.py` output.)

- **`p2` is flat at 0.5000 ± 0.0000 across the whole sweep, both target priorities.** The
  acceptance is not a point.
- **`p1` is a step, not a curve** — exact up to the lattice capacity (7 cells ⇒ ≤ 6 offers +
  target), broken from the first overflow and *flat thereafter* (10 and 12 offers give the
  same numbers as 8: the placed set is always the top-7 by priority, so exactly one cell
  fails to refill).
- ⚠ Note the axis difference from the incumbent curve: the relocate allocator's leak is
  **occupancy-scaled and never 0.5** (0.6715 at 2 offers); canonical placement is **0.5000
  at every load below overflow even without the waitlist**. The waitlist closes the *only*
  remaining load regime.

## A.4 ⭐ The exact replacement scope sentence (for `r1-positioning-pass`, via the Hub)

`placement-landing` §6's *"The claim covers stores operating below capacity or under
set-function (priority/attribute-based) eviction"* is replaced by:

> *"Placement in the store is canonical — a deterministic function of the live item records
> and the store geometry alone — so store-level deletion is exact: removing an item
> reproduces, bit for bit, the store that holds the remaining records, with each survivor's
> scheduled decay and permanence unaffected (deletion and decay provably commute). The
> capacity qualifier is no longer needed: an offer the lattice cannot seat is retained in the
> offered set and re-seated by priority as soon as a cell frees, so byte-identity holds at
> every load we measured — offers from 0.29× to 1.71× the lattice's cell count, AUC
> 0.5000 ± 0.0000 on all six membership statistics in 3 072 paired worlds per load. The claim
> covers set-function (priority / item-intrinsic-attribute) eviction; recency-based eviction
> is intrinsically history-dependent and is excluded. This is a store-level statement only:
> the frozen encoder and any residue of past writes in a learned landscape are separate
> channels, measured separately; we make no (ε,δ) claim."*

**Conditions the sentence carries (do not drop them):** measured with `budget ≥ n_cells` and
`leak = 0` in the sweep (decay-with-waitlist exactness is pinned by a **bit-identity unit
test**, not by the MIA harness); `evict_policy="depth"`; LRU remains a hard error.

---

# Part B — option (d), the gated-stiffness payload channel

Flag `payload_gate` **defaults OFF**; with it off the shipped `V` is **bit-identical**
(unit-tested, 32 random queries, exact `==`). No shipped default changed (B1.4).

## B.1 ⭐ The decisive number — `R₅₀` under the gate (mia §3(b))

Retention vs launch radius, 3 seeds × 8 targets × 128 worlds; `R₅₀` = radius at which
retention crosses 0.5 (linear interpolation, mia's convention).

| arm | `A` = 1.0 | 0.5 | 0.2 | 0.1 | 0.06 | **contraction** |
|---|---|---|---|---|---|---|
| **`base` (published mia)** | 1.146 | 1.083 | 0.979 | 0.874 | 0.752 | **1.524×** |
| **`g05` (`g₀ = amp_floor`)** | **1.135** | 1.062 | 0.942 | 0.852 | **0.771** | **1.472×** |
| `g005x4` (graded + ×4 read) | 1.152 | 1.086 | 0.972 | 0.881 | 0.796 | 1.447× |
| `g005` (graded, shipped read) | 1.152 | 1.085 | 0.965 | 0.850 | *0.221* | *5.2× — read-length artefact, see B.4* |
| *TTL vector-store (mia §3(b))* | *0.77* | *0.77* | *0.77* | *0.77* | *0.75* | *1.00× (a step)* |

⭐ **`R₅₀` survives the gate.** The contraction is essentially unchanged (1.472 vs 1.524),
and it is *the same physical mechanism*: `R₅₀` was pre-registered in mia P5 from a saddle
calculation on the **address** well (`α|q|² − A e^{−d²/2s²}`), which the payload gate does
not touch. **The lifetimes dial keeps the one differentiator that needs no adversary-model
caveat.** My `base` column reproduces the published `R₅₀` (1.146/1.083/0.979/0.874/0.752)
**element-for-element**, so the comparison is like-for-like.

## B.2 mia §1 — retention and distinguishability vs amplitude

| `A` | `τ = −ln A` | `base` ret | **`g05` ret** | `g005` ret | `g005x4` ret | `base` val-err | **`g05` val-err** | `base` AUC(s1) | `g05` AUC(s1) |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0001 | 0.0000 | 1.0000 | 1.0000 |
| 0.30 | 1.204 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0011 | 0.0001 | 1.0000 | 1.0000 |
| 0.15 | 1.897 | 0.9928 | **0.9996** | 0.9999 | 0.9999 | 0.0112 | 0.0015 | 0.9999 | 1.0000 |
| 0.10 | 2.303 | 0.9714 | **0.9989** | 0.9995 | 0.9995 | 0.0370 | 0.0037 | 0.9986 | 1.0000 |
| 0.08 | 2.526 | 0.9397 | **0.9976** | 0.9962 | 0.9989 | 0.0610 | 0.0204 | 0.9967 | 1.0000 |
| 0.07 | 2.659 | 0.9159 | **0.9966** | **0.5005** | 0.9983 | 0.0794 | 0.0289 | 0.9941 | 1.0000 |
| 0.06 | 2.813 | 0.8858 | **0.9945** | **0.4990** | 0.9968 | 0.1038 | 0.0318 | 0.9895 | 0.9997 |
| **0.051** | **2.976** | **0.8321** | **0.9904** | 0.7855 | 0.9912 | 0.1331 | 0.0239 | 0.9830 | 0.9985 |

- **B2 confirmed: at `g₀ = amp_floor` retention is a step** — ≥ 0.99 at every amplitude down
  to the floor, then 0 at self-eviction. That is the TTL *shape*, as the theorist priced it,
  and it is exactly what the `R₅₀` differentiator above is *not* affected by.
- The `base` column reproduces the published mia §1 retention (0.9714/0.9397/0.9159/0.8858/
  0.8321) **element-for-element** — the like-for-like control.
- **B6 confirmed: `AUC(s4)` (white-box address depth) = 1.0000 at every amplitude in every
  arm.** N108 stands: the gate buys nothing against an exact adversary, and I claim nothing
  there. Query-MIA `AUC(s1)` is, if anything, *slightly higher* under the gate (0.9985 vs
  0.9830 at the floor) — the store answers better, so it also leaks slightly better.

## B.3 mia §5 — the payload dependence, before and after

Pearson `r` over the **24 per-example** retention values at each amplitude.

| arm | `A` | **r(ret, `aᵢ²`)** | ret std | ret min | ret at `\|a\| = 1` |
|---|---|---|---|---|---|
| `base` | 0.060 | **−0.857** | 0.188 | 0.346 | 0.582 |
| `base` | **0.051** | **−0.846** *(the published number)* | 0.274 | 0.007 | **0.398** |
| **`g05`** | 0.060 | **−0.762** | **0.008** | 0.966 | **0.985** |
| **`g05`** | **0.051** | **−0.667** | **0.016** | **0.925** | **0.972** |
| `g005` | 0.060 | −0.873 | 0.499 | 0.000 | **0.000** |
| `g005x4` | 0.060 | −0.560 | 0.006 | 0.969 | 0.990 |
| `g005x4` | 0.051 | −0.466 | 0.024 | 0.880 | 0.970 |

❌ **B3 falsified as stated** (I registered \|r\| < 0.30). The correlation is scale-free and
survives at −0.667; what dies is the **effect size**: at the last amplitude before
self-eviction the worst codeword goes **0.398 → 0.972**, the spread **0.274 → 0.016** (17×),
the floor **0.007 → 0.925**. ⇒ *The honest claim is "the payload dependence is reduced to a
≤ 5 % ordering effect", not "removed".* (Reconciliation item 3.)

## B.4 ⭐ The read-length law — the compute-adaptive-read corner, measured

The theorist's `τ_y = η/(κ(g₀+A))` is an **overdamped** relaxation time, but the shipped read
phase runs at **`γ_read = 0.0`**. The payload coordinate therefore does not relax — it
**oscillates** about `ā` at `ω = √(κG)`, and the tail average returns `ā` only once the
averaging window covers a period. Measured directly from the read trajectory
(`osc_period.py`, zero-crossings of `q₂ − mean`, 4 queries per cell):

| gate | `g₀` | `A` | `G = g₀+A` | **T measured** | `2π/√(κG)` | osc. amplitude | tail err |
|---|---|---|---|---|---|---|---|
| off | — | any | κ = 1 (flat) | 6.28–7.01 | 6.283 | 0.00–0.25 | ≤ 0.030 |
| on | 0.05 | 0.20 | 0.2500 | **12.567** | 12.566 | 0.003 | 0.0003 |
| on | 0.05 | 0.10 | 0.1500 | **16.225** | 16.223 | 0.086 | 0.0032 |
| on | 0.05 | 0.06 | 0.1100 | **19.100** | 18.945 | 0.226 | 0.0557 |
| on | 0.005 | 0.10 | 0.1050 | **19.500** | 19.390 | 0.255 | 0.0500 |
| on | 0.005 | 0.06 | 0.0650 | **24.650** | 24.645 | 0.598 | 0.2180 |
| on | 0.005 | 0.051 | 0.0560 | **26.550** | 26.551 | 0.717 | 0.1737 |

**log-log slope `d ln T / d ln G` = −0.5264** (√ law: −0.5; overdamped law: −1.0). ⇒ **B5's
registered law confirmed and the theorist's refuted** (reconciliation item 4). The operative
requirement is

  **`tail_frac · read_steps · dt ≥ T = 2π/√(κ(g₀+A))`**, i.e. `read_steps ≥ 8π/(dt√(κ(g₀+A)))`.

Shipped read = 800 steps ⇒ window 10.0 time units. Required: **503** steps for the baseline
(satisfied), **1 528** at `g₀ = 0.05, A = 0.06` (**not** satisfied — ×2 needed), **1 972** at
`g₀ = 0.005, A = 0.06` (×4 needed). The full-harness sweep matches:

| arm | read_steps | `A = 0.06` val err | retention | verdict vs the law |
|---|---|---|---|---|
| `g05` | 800 | 0.0318 | 0.9945 | window 10 < T 19.1 — error visible but still ≪ `tol` |
| `g05x2` | 1600 | **0.0096** | 0.9947 | window 20 > T — error drops 3.3×, then plateaus |
| `g05x4` | 3200 | 0.0112 | 0.9947 | no further gain (plateau) |
| `g005` | 800 | 0.1146 | **0.4990** | window 10 ≪ T 24.6 — **the value criterion fails for `\|a\| ≥ 0.714`** |
| `g005x4` | 3200 | **0.0193** | **0.9968** | window 40 > T — restored |

⭐ **This is the compute-adaptive-read dial, reported as a measurement, not a pitch:** at
`g₀ = 0.005` a faded memory (`A = 0.06`) is unreadable at the shipped read length and fully
readable at **4× the integration steps**, and the required length scales as
**`(g₀+A)^{−1/2}`**. ⚠ It is also a *cost*: the `g005` arm at the shipped read length is
**worse than baseline** (retention 0.499 vs 0.886) and its payload dependence is *stronger*
(−0.873), because the failure is `\|a\|`-graded. `g₀ ≪ amp_floor` is only usable with the
longer read.

---

## PREREG scorecard

| # | registered | measured | verdict |
|---|---|---|---|
| **A1** | `p2`, tid −5, 8 offers: all AUC 0.5000 ± 0.005, byte-equal 1.0 | **0.5000 ± 0.0000** on all six, byte-equal **1.0000 (3072/3072)** | ✅ exact |
| **A2** | `p1`, tid −5: `AUC(n_live)` 1.000, `s4` 0.91 ± 0.05, byte-equal 0.000 | **1.0000 / 0.9106 / 0.0000** | ✅ (reproduces w26 to 4 dp) |
| **A3** | tid −1: the target is the unseated key ⇒ w26's 1.000 there is a stale-slot artefact; corrected ⇒ 0.5000, byte-equal 1.0, `seated = 0` | **`seated = 0.000`; 0.5000 on all six; byte-equal 1.0000 — in BOTH arms** | ✅ **confirmed, and it retires a published cell** |
| **A4** | the waitlisted sweep is FLAT: 0.5000 / byte-equal 1.0 at offers 2,4,6,8,10,12, both tids | flat, **0.5000 ± 0.0000 / 1.0000** at all 12 cells | ✅ |
| **A5** | without the waitlist: exact ≤ 6 offers, broken ≥ 8 | exact at 2/4/6; 1.0000 at 8/10/12 | ✅ (a step, as registered) |
| **A6** | moves/delete at 8 offers, tid −5 ∈ [0.5, 3.0], max ≤ 7 | **1.68, max 5** | ✅ |
| **A7** | full `pytest tests/` green with the waitlist ON by default; `test_T2_at_the_rematch_point` still 61 live | 61 live; **1 failure found and fixed** (float32 vs the store dtype under `jax_enable_x64`, `2cac7b9`), then green | ◐ **the prediction caught a real bug — see §"How I verified"** |
| **B1** ⭐ | `R₅₀` still contracts at `g₀ = amp_floor`: `R₅₀(1) = 1.15 ± 0.10`, `R₅₀(0.06) = 0.82 ± 0.15`, ratio 1.40 ± 0.30 | **1.135 / 0.771, ratio 1.472** | ✅ **all three inside the band; the falsifier did not fire** |
| **B2** | retention ≥ 0.99 at every `A ≥ 0.051` under `g05` (a step) | **0.9904** at 0.051, ≥ 0.9945 above | ✅ |
| **B3** | payload dependence removed: \|r(ret, `a²`)\| < 0.30 (or undefined) | **−0.667** (std 0.274 → 0.016, worst codeword 0.398 → 0.972) | ❌ **falsified as stated** — the correlation persists, the effect size collapses |
| **B4** | ×4 read restores payload-independence at `A = 0.06` (mean ≥ 0.90) | **0.9968** (vs 0.4990 at ×1) | ✅ |
| **B5** ⭐ | the read-length law is `T = 2π/√(κ(g₀+A))`, **not** the theorist's `η/(κ(g₀+A))`; thresholds 1 508 / 1 966 steps | **T measured to 3–4 s.f. at every point**, slope **−0.5264** vs −0.5 / −1.0; thresholds 1 528 / 1 972 | ✅ **confirmed; the theorist's form refuted** |
| **B6** | `AUC(s4)` unchanged by the gate (1.000 everywhere) | **1.0000** at every amplitude, every arm | ✅ (N108 stands) |
| **B7** | `payload_gate=False` ⇒ bit-identical `V` | exact `==` on 32 random queries (unit test) | ✅ |

**Score: 11 confirmed, 1 partial (A7 — the registered check found a bug), 1 falsified (B3).**

---

## How I verified

```
# unit (worktree, main venv, PYTHONPATH=.)
pytest tests/test_placement.py -q --no-cov        -> 39 passed (26 w26 + 13 new), 48 s
pytest tests/test_lattice.py tests/test_placement.py -q  -> 61 passed  (x64-contaminated order)
pytest tests/ -q --no-cov                          -> 1 failed, 759 passed (1646 s)  [BUG FOUND]
   -> fixed in 2cac7b9 (dtype); re-run   -> **760 passed, 24 warnings in 2307 s** (green)
ruff check chlu/ tests/                            -> All checks passed!
# measurement (cwd = worktree, PYTHONPATH = worktree, main venv, JAX 0.9.0)
waitlist_mia.py --arms p1,p2 --offers 8 --target-ids=-5,-1 --tag _head          (1705 s)
waitlist_mia.py --arms p1,p2 --offers 2,4,6,10,12 --target-ids=-5,-1 --no-reads (~55 min)
gate_mia.py --arms base,g05      --panels AB --tag _g05    (964 s)
gate_mia.py --arms g005,g005x4   --panels AB --tag _g005   (2253 s)
gate_mia.py --arms base,g05,g05x2,g05x4,g005,g005x4 --panels L --tag _len (921 s)
osc_period.py                                              (~60 s)
analyze.py A|B  -> every table above
```
`ruff format` is **not** run (the repo is not format-clean; it would emit a huge foreign
diff). No NaN, no divergence, no non-finite value in any statistic. **The one full-suite
failure is reported above, not hidden:** `test_P2_exactness_survives_decay_of_a_waiting_item`
passed in isolation and failed in `pytest tests/`, because `tests/test_lattice.py` and
`tests/test_goldstone.py` enable `jax_enable_x64` globally at module import and my waiting
amplitude was rounded through a hard-coded float32 while the live wells were rounding through
float64 — a genuine bug in my code, fixed by rounding through the store's own dtype. **This is
exactly what registered item A7 was for.** The post-fix full suite is **760 passed, 0 failed**.

## Git footprint

Branch **`agent/experiment-engineer/deletion-waitlist-stiffness`**, base local `main`
**`082d095`** (unmoved; `git rebase main` = "up to date", a no-op), worktree
`../CHLU-waitlist` — **branch ref verified from the MAIN repo** (`git log
main..agent/experiment-engineer/deletion-waitlist-stiffness` shows all four commits) before
the worktree was removed. **Not pushed. 4 commits:**

| hash | subject | files |
|---|---|---|
| `f650d67` | add the gated-stiffness payload channel behind `payload_gate` (OFF) | `memory_potentials.py` (+45/−4, `AtomStorePotential` only), `config.py` (+9) |
| `9b2c7ed` | P2 waitlist: canonical deletion stays exact AT OVERFLOW | `placement.py` (+86/−28), `controller.py` (+150/−32), `config.py` (+8) |
| `8731557` | tests: the P2 waitlist (6) and the payload gate (4) | `tests/test_placement.py` (+198/−9) |
| `2cac7b9` | waitlist amps: round through the STORE's dtype, not float32 | `controller.py` (+11/−4) |

Total `main..HEAD`: **5 files, +453/−54**. Files touched: `chlu/core/placement.py`,
`chlu/core/controller.py`, `chlu/core/memory_potentials.py` (**`AtomStorePotential` only** —
nothing above it, nothing below it), `chlu/config.py` (**`ExperimentControllerMvpConfig`
only**, 4 additive fields), `tests/test_placement.py`. ⛔ **`chlu/experiments/
exp_designed_mechanism.py` and `ExperimentDesignedMechanismConfig` were NOT touched**
(`r2-d-sweep-close`'s files this wave); neither was `exp_controller_mvp.py`. No conflicts;
nothing outside the ownership list; rebase onto local `main` is a no-op.

## Open questions / follow-ups / risks

1. ⚠ **The waitlist is unbounded.** With `leak = 0` a refused offer is remembered forever
   (one `ItemRecord` + one float per offer). At MVC-0 sizes this is nothing; a long-running
   store needs a waitlist policy (cap by priority, or expire), and **any such policy must
   itself be a set function** or it re-opens the exactness hole it just closed. Not built.
2. **`budget < n_cells` is untested** (see A.1). The interaction — a freed budget slot is
   taken by the highest-priority *waiting* key, not by the new offer — is coherent but
   unmeasured, and it changes what "refused because full" means for the caller.
3. **Cost.** `_canonical_sync` is still an O(n) rebuild per op; the waitlist adds one dict
   op per displaced key. The 3 072-world acceptance arm ran 400 s vs `p1`'s 402 s — **the
   waitlist is free at this size.**
4. **Part B is landed but not adopted.** `payload_gate` is OFF; nothing downstream reads it.
   Adopting it would (i) fix D3's effect size, (ii) leave `R₅₀` intact, (iii) *require*
   re-tuning `read_steps` (B.4) if `g₀ < amp_floor`, and (iv) change every published
   designed-store retention number. **Recommend a separate Head/Hub decision, with the
   read-length law in hand.**
5. **`g₀` between the two corners is unexplored.** I measured `g₀ ∈ {amp_floor, amp_floor/10}`
   because those are the theorist's; the trilemma is presumably continuous in `g₀`, and
   `g₀ = amp_floor/2` with a ×2 read may be the honest middle. One harness call each.
6. **Not re-measured under the gate:** the TTL-flag laundering line (mia §3(a)) and the TM-3
   resolution-limited adversary. Neither should move (the gate does not touch the address
   channel, and `s4` is flat), but I did not measure them and do not claim them.

---

## Proposed handover updates (for the Hub)

1. **§7-CURRENT / N99 — the exact-deletion claim loses its capacity qualifier.** *"w27
   `deletion-waitlist-stiffness`: the P2 waitlist is in `chlu/core/placement.py` +
   `Controller(waitlist=True)` (default under canonical placement). On the un-inflated 7-cell
   mia geometry the post-delete store is byte-identical to the never-held-it store at every
   load from 2 to 12 offers (0.29×–1.71× lattice capacity): `AUC` **0.5000 ± 0.0000** on
   `n_live`/`z_hole`/`s1`/`s2`/`s4`/`s5`, byte-equal **1.0000 (3 072/3 072 worlds per load)**,
   3 seeds × 8 targets. Without the waitlist the same sweep is a **step**: exact at ≤ 6
   offers, `AUC(n_live) = 1.0000` / `AUC(s4) = 0.9106` / byte-equal 0.0000 at 8, 10 and 12.
   Scope now reads 'exact under set-function eviction'; LRU remains a hard error."*
2. ⛔ **`placement-landing`'s `canon_native` (target id −1) row must be retired** — it is a
   stale-slot delete artefact (§A.2 reading 3). The defensible overflow-failure numbers are
   the **high-priority-target** ones (1.0000 / 0.9106 / 0.0000), which this report reproduces
   to 4 dp. Anyone quoting "canonical placement fails at overflow" should quote that cell.
3. **§1.6 / the N108 companion line — `R₅₀` survives the payload gate.** *"Under option (d)
   at `g₀ = amp_floor` the retrieval-geometry differentiator is intact: `R₅₀` contracts
   **1.135 → 0.771** as `A: 1 → 0.06` (baseline 1.146 → 0.752; a TTL dict is a constant step
   at 0.75–0.77), while white-box `AUC(s4)` stays 1.000 at every amplitude in every arm."*
4. **§7 / theory queue — mia-D3 is FIXED IN EFFECT SIZE, NOT IN CORRELATION.** r(ret, `a²`)
   moves only −0.846 → −0.667, but retention std at the floor collapses 0.274 → 0.016 and the
   worst codeword goes 0.398 → 0.972. Wording must say "reduced to a ≤ 5 % ordering effect".
5. ⛔ **Correction for `readout-channel-theory` §3.3:** the read-length requirement is
   **`τ_y = 2π/√(κ(g₀+A))`** (measured to 3–4 s.f., log-log slope −0.5264), *not*
   `η/(κ(g₀+A))` — the shipped read phase is undamped (`γ_read = 0`). Required `read_steps ≥
   8π/(dt√(κ(g₀+A)))`: 503 (baseline) / 1 528 (`g₀ = 0.05, A = 0.06`) / 1 972 (`g₀ = 0.005,
   A = 0.06`) against a shipped 800.
6. **§3 config table — three new default-preserving knobs** in `experiment_controller_mvp`:
   `canonical_waitlist = True` (inert below capacity and under `placement="relocate"`),
   `payload_gate = False`, `payload_gate_g0 = 0.05`, `payload_gate_eps = 1e-6`. New
   `Controller` kwarg `waitlist=True`; new `AtomStorePotential` kwargs `payload_gate`,
   `payload_g0`, `payload_eps`.
7. **Testing note worth keeping:** `tests/test_lattice.py` and `tests/test_goldstone.py`
   enable `jax_enable_x64` **globally at module import**, so any bit-identity test is run in
   float64 in a full-suite ordering and in float32 in isolation. My waitlist bug only
   appeared in the full suite. Bit-exactness code must read dtypes from the object, never
   hard-code them.
