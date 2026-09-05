# PREREG — deletion-waitlist-stiffness (w27, experiment-engineer)

Written **before** either harness was run (Part A harness `waitlist_mia.py` and Part B
harness `gate_mia.py` did not exist when this file was committed to disk; the shipped-code
edits existed, but no measurement had been taken). Base: local `main @ 082d095`, branch
`agent/experiment-engineer/deletion-waitlist-stiffness`, worktree `../CHLU-waitlist`.

---

## 0. What is being predicted, and from what

### Part A — the P2 waitlist (dial: isolation/deletion)
Mechanism as implemented: under `placement="canonical"`, a key the lattice cannot seat
keeps `pos[k] = None` in the placer and its `ItemRecord` in `Controller.waiting`; every
op re-runs the priority-suffix greedy, so the placed set is always **the seatable prefix
of the offered set in priority order** — a pure set function of the offered set at any
load. Therefore `Store(S∖{i})` after `delete(i)` must equal the store built from `S∖{i}`
**byte for byte, at any load**, and both the target's removal and the re-entry of a
displaced background key are forced.

⚠ Priority facts of the acceptance geometry (computed from `prio()` before predicting,
not measured from a harness): over the key set `{-1, 0…6}` the target `-1` has the
**lowest** priority of all 8 keys; over `{-5, 0…6}` the target `-5` has the **highest**.
Descending priority over `{-5,-1,0..11}`: `-5, 4, 1, 6, 5, 9, 2, 11, 3, 10, 0, 8, 7, -1`.
Consequence, predicted in advance: at 8 offers into a 7-cell lattice **the key that fails
to seat is the target itself when `target_id = -1`, and background key `0` when
`target_id = -5`.** These are two *different* failure modes and I predict different
mechanisms for the two fixes (A3/A4 below).

### Part B — the gated-stiffness payload channel (dial: lifetimes)
`V_pay = 0.5·κ·G(x)·(y − ā(x))²`, `G = g₀ + Σ mᵢAᵢeᵢ` (floored), `ā = Σ mᵢAᵢaᵢeᵢ /
(ε + Σ mᵢAᵢeᵢ)` (**not** floored). The hill/well the payload term puts on a site becomes
`κaᵢ²·(A/(g₀+A))·…` → amplitude-independent in the ratio sense, which is why the theorist
measured payload-**independent** retention. The `R₅₀` question is whether the *address*
channel's contraction survives: `R₅₀` was pre-registered in `mia-decay-measurement` P5
from a **saddle calculation on the address well alone** (`αq² − A e^{−d²/2s²}`) and
confirmed (1.146/1.083/0.979/0.874/0.752 vs 1.15/1.05/0.90/0.80/0.72). The gate does not
touch that term. **So my registered expectation is that the contraction SURVIVES.**

---

## 1. Part A — checkable items

Harness: `waitlist_mia.py`, a copy of `placement-landing/mia_placement.py` with
(i) an `--offers` knob (total offers = 1 target + (offers−1) background), (ii) a
`--waitlist on|off` arm knob, (iii) `delete(target_id)` called **unconditionally** under
canonical placement (the w26 harness only deleted when the target was *live*, which is why
`canon_native` deleted a background row instead — see A3), (iv) a `target_seated_frac`
statistic. Adversary/statistics imported verbatim from `mia-decay-measurement`.
Geometry: `R_mia = 2.28695` ⇒ **7 cells**, store capacity 8, `budget = 8`, `leak = 0`,
`amp = 1.0`, `evict_policy="depth"`, 3 seeds × 8 targets × 128 paired worlds.

| # | prediction | falsified if |
|---|---|---|
| **A1** | **waitlist ON, `target_id = -5`, 8 offers:** `AUC(n_live)` **0.5000 ± 0.0000**, `AUC(s4)` **0.5000**, `AUC(z_hole)` **0.5000**, byte-equal fraction **1.0000** (3072/3072) | any statistic ≠ 0.5000 ± 0.005, or byte-equal < 1.0 |
| **A2** | **waitlist OFF, `target_id = -5`, 8 offers** reproduces w26: `AUC(n_live)` 1.000, `AUC(s4)` 0.91 ± 0.05, byte-equal 0.000 | the OFF arm is already exact (⇒ my A1 measures nothing) |
| **A3** | **waitlist OFF, `target_id = -1`, 8 offers:** the target is the key that fails to seat, so the *undeleted* IN store already equals OUT; w26's `AUC(n_live) = 1.000` there is an artefact of deleting the target's **stale slot 0** (a background row). With the corrected unconditional delete + waitlist ON: all AUC **0.5000**, byte-equal **1.0000**, `target_seated_frac = 0.000` | target_seated_frac ≠ 0 at 8 offers, or the OFF arm's mechanism is not the stale-slot delete |
| **A4** | **the load sweep is FLAT with the waitlist on:** at offers = 2, 4, 6, 8, 10, 12 (7-cell lattice), every statistic 0.5000 ± 0.0000 and byte-equal 1.0000, for both target ids | any load at which byte-equality < 1.0 with the waitlist on |
| **A5** | **without the waitlist the sweep is a step, not a curve:** exact (0.5/byte-equal 1.0) at offers ≤ 6 (no overflow: 6 keys ≤ 7 cells), broken at offers ≥ 8 for `target_id = -5` | exactness fails at offers ≤ 6, or holds at offers ≥ 8 |
| **A6** | `moves/delete` at 8 offers, `target_id = -5` (highest priority ⇒ every survivor is re-run): mean ∈ [0.5, 3.0], max ≤ 7 | outside |
| **A7** | full `pytest tests/` stays green with the waitlist ON by default (w26's canonical tests are all below capacity except `test_T2_at_the_rematch_point`, where 64 offers into 61 cells now waitlists 3 keys instead of forgetting them — `n_live` must still be **61**) | any w26 test changes value |

**A-scope sentence (registered before measuring, to be confirmed or amended):** if A4
holds at every load, `placement-landing`'s *"exact below capacity or under set-function
eviction"* becomes *"exact under set-function eviction, at any load"*.

## 2. Part B — checkable items

Harness: `gate_mia.py` (Panel A amplitude ladder + Panel B radius sweep + payload
decomposition), same 3 seeds × 8 targets × 128 paired worlds, shipped two-phase read,
shipped `Controller` with the **relocate** allocator (so §1/§3(b)/§5 are compared
like-for-like against `mia-decay-measurement`). Arms: `base` (gate off, the published
baseline), `g05` (`payload_gate=True, g₀ = 0.05 = amp_floor`), `g005` (`g₀ = 0.005`),
`g005x4` (`g₀ = 0.005`, `read_steps ×4`).

| # | prediction | falsified if |
|---|---|---|
| **B1** ⭐ | **`R₅₀` still contracts under the gate at `g₀ = amp_floor`.** Registered numbers: `R₅₀(A=1) = 1.15 ± 0.10`, `R₅₀(A=0.06) = 0.82 ± 0.15`, **ratio 1.40 ± 0.30** (baseline 1.524). Reason: `R₅₀` was derived and confirmed from the **address** well's saddle, which the gate does not touch; the gate only removes the value-criterion failure, which can only *raise* `R₅₀` at small `A`, not flatten it | ratio ≤ 1.10 (i.e. `R₅₀` flat within ±5 %) ⇒ **the lifetimes dial has lost its only adversary-caveat-free differentiator, and I report exactly that** |
| **B2** | on-site retention at `g₀ = 0.05` is a **step**: ≥ 0.99 at every `A` down to 0.051, then 0 at self-eviction (reproduces the theorist's 1.000 column on the shipped controller-placed geometry rather than his ring) | retention < 0.95 at any `A ≥ 0.051` |
| **B3** | **payload dependence removed at `g₀ = 0.05`:** Pearson `r` between per-example retention at the floor and `aᵢ²` goes **−0.846 → \|r\| < 0.30** (or undefined because retention has zero variance — I will report that case as such, not as a small `r`) | \|r\| ≥ 0.5 |
| **B4** | `g₀ = 0.005` at the **shipped** read length is *graded but payload-dependent* near the floor (theorist: 0.750/0.643/0.508 at A = 0.08/0.06/0.051), and **×4 read restores payload-independence to `A = 0.06`** (≥ 0.95 mean) | ×4 read does not lift mean retention at `A = 0.06` above 0.90 |
| **B5** ⭐ (read-length law) | The theorist's `τ_y = η/(κ(g₀+A))` is the **overdamped** form and cannot be the operative law here: the shipped read phase has **`γ_read = 0.0`** (undamped), so the payload coordinate *oscillates* about `ā` with `ω = √(κG)` and the tail-average returns `ā` only if the averaging window covers a period. Registered law: **the read must satisfy `tail_frac · read_steps · dt ≥ T = 2π/√(κ(g₀+A))`**, i.e. `read_steps ≥ 8π/(dt·√(κ(g₀+A)))`. Predicted thresholds (κ=1, dt=0.05, tail_frac=0.25): baseline `G=1` ⇒ needs 503 steps (shipped 800 ✓); `g₀=0.05, A=0.06` ⇒ `T = 18.9`, needs **1508** steps ⇒ shipped 800 **fails**, ×2 (1600) passes; `g₀=0.005, A=0.06` ⇒ `T = 24.6`, needs **1966** ⇒ ×4 (3200) passes, ×2 fails. **Test: sweep `read_mult ∈ {1,2,4,8}` and locate the value-error threshold; it must track `1/√(g₀+A)`, not `1/(g₀+A)`** | the measured threshold read length scales as `1/(g₀+A)` (a factor ~4 change between `g₀+A` = 0.11 and 0.065 rather than the ~1.3 the √ law predicts), or is amplitude-independent |
| **B6** | white-box `AUC(s4)` (address-channel depth) is **unchanged by the gate** (1.000 at every `A`, the N108 standing result), because `s4` probes `U = α\|q\|² − ΣmAe` only | s4 differs between arms by > 0.01 |
| **B7** | the flag is inert when off: `payload_gate=False` reproduces the baseline `V` **bit-identically** (`max\|ΔV\| = 0.0` over random queries) | any nonzero difference |

**Registered answer to the task's decisive question:** *`R₅₀` survives the gate at
`g₀ = amp_floor`* (B1). If it does not, the honest report is that the lifetimes dial's
retrieval-geometry differentiator is gone at that setting, and the only remaining graded
configuration is `g₀ ≪ amp_floor` + long reads (B4), which is the compute-adaptive-read
corner.

## 3. What would NOT falsify anything (declared in advance)
- Losing to a dict / TTL flag on any AUC axis (N108, standing).
- A step-shaped **on-site** retention curve at `g₀ = amp_floor` (B2 — that is the
  trilemma's stated price).
- `R₅₀` being *larger* than baseline at small `A` under the gate (the value criterion no
  longer fails); only *flatness* falsifies B1.
- The waitlist making a *low-priority* offer wait longer under a finite budget (priority
  semantics, not a defect).
