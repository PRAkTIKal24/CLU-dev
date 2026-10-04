# PREREG-C2W10 — "The persistent store": memory across streams

**Filed by the C2W10 research-lead Hub, 2026-08-10, BEFORE any cell of this wave runs.**
Scope: charter **§A21's C2W10 row**, as amended by **ADDENDUM 12 §A34.3** (the THREE-STATE lifecycle)
and **§A20.6's Head addition** (k-streams-never-useful → `γ_φ(q)`), plus **Add.7 ruling 5** (the I2
re-measurement, deferred to this wave), **Add.9 §A27.1** (decay-netting as a BUILD REQUIREMENT), and
**§A23.2** (`refresh_monotonic`'s home is here). Bound additionally by the intervention doc §6 (the
five admissibility criteria) and §8 (the prohibitions), and by **§A33.1** (the MECHANICS/VALUE rule).

⛔ **Nothing in this file may be edited after the first cell runs.** Amendments are appended, dated,
and labelled as amendments (the `PREREG-Bprime.md` precedent: a revised pre-registration stops being
one).

---

## 0. DIAL DECLARATION (protocol §7 — echoed by every spoke before its first result)

- **Dial:** **lifetimes + admission** (retention of revisited regimes; admission/eviction/promotion/
  demotion/trash under capacity pressure across stream boundaries). Compute-adaptive reads are NOT a
  dial this wave (declared NOT-RUN; the anytime curve is C2W9's).
- **Laundering / tier control:** MECHANICS legs take **no** performance control (§A33.1 — a launder
  margin on a component gate is a DIAGNOSTIC, never a pass condition). VALUE legs take the **tier-iii
  system-level swap** (a matched-params AND matched-state-bytes TTT-class cell in the same harness;
  per **FB2**, a GRU can never be matched on both and is therefore a one-sided arm only) **plus** the
  benchmark-native byte-matched arms (SAM-kNN, kNN_S, ARF, No-Change persistence).
- **Claim scoping (Head ruling 2, 2026-08-10 — the control is defined by its PROPERTIES, not its
  file):** a system-level swap is valid iff the **whole memory** is replaced and the matching is
  **two-sided (params AND state-bytes)**; it must be **TTT-class, never GRU** (FB2: Θ(h²) vs Θ(h) ⇒ a
  GRU can never be two-sided matched; a TTT-class cell matches to ±0.1 %). On that definition **this
  wave's harness swap IS a valid tier control.** What is scoped is the *claim*, not the control:
  ⭐ **the ±persistence contrast is INTERNAL** — same harness, same scale, both arms — and it is
  **the wave's primary evidence**; ⚠ **the absolute competitiveness-against-rivals reading is
  SCALE-SCOPED and explicitly NON-PROMOTABLE to the CSF3 tier-iii claim**, and every artifact carries
  that sentence beside the number. This is not a downgrade to "labelled pilot": **valid control,
  clean internal primary, scoped absolute.**
- **Falsifies the wave's VALUE claim:** the persistent store's retention margin over its own episodic
  toggle is ≤ 0 at 2 SE at matched bytes; or the byte-matched exemplar store (SAM-kNN) matches/beats
  it on revisit-recovery (the scout's registered criterion-4 tripwire).
- **Does NOT falsify it:** losing to a per-regime oracle; losing on the drift-free `out-of-control`
  null (there the *correct* behaviour is no benefit — a win there is an artifact, not a result);
  losing on a stream where the persistent-vs-episodic contrast is absent (that is the metric-native
  ceiling theorem, not news).
- ⛔ **Depth is NEVER quoted as feature importance** (§A23.5 ACTIVE; lifts only per §4's refutation
  branch, and only the Hub lifts it). ⛔ **`M` is never quoted without naming its criterion** (K9 is
  the registered merge criterion; pass-1 `M` was vacuous). ⛔ **"CLU-former" stays a placeholder.**
  ⛔ **N94 epoch discipline** on any promotable reading. ⛔ Declared NOT-RUNs are never reported as
  nulls.

---

## 1. The rig (one capability toggled — the Add.6 standing rule)

**The FULL block for a streaming-prediction substrate:** address encoder → `CluSystem` store (learned
`V_θ`, atoms family) + `MemoryController` (admission / lifetimes / eviction / the new lifecycle verbs)
→ read → prediction head, driven prequentially over a stream of chunks.

**The single toggle: `persistent_store ∈ {True, False}`.**
- `False` (episodic) — the store is reset at every stream/document boundary; the encoder and head are
  not reset. This is the ablation arm, not a control.
- `True` (persistent) — the store survives boundaries, and the three-state lifecycle runs.

**Carried substrate (all pytest-pinned, inherited not rebuilt — §A34.10):** the placing write
(`atom_site_local_init`) · co-scaled Wendland widths · **d = 12 operational** (store measured INERT at
d = 16 with 131 072 atoms ⇒ **d = 16 is a declared NOT-RUN**) · P1 `erosion_partition=True` · the
item-id-keyed read-hit counter (`controller.touch` → `read_hits`, survives eviction) · the
`UsageTelemetry` instrument (registered primary proxy `read_hits(i)`; **depth never enters `U`**) ·
the trash region `CluSystem.gamma_phi / trash_route / trash_bytes` (**ships OFF; C2W10 is its first
experimental use ON**, and the OFF bit-identity + parameter-count-identity regressions stay green).

**Address block (Hub decision, from the §A31.4 inversion — task features ≠ address features):** the
**default arm is a cheap, unfitted projection** (random linear projection of the 33 features +
per-dimension standardisation + unit-ball normalisation — the tabular analogue of `randconv`, which
was the address-BEST arm at pass 3 for **0 fit steps**). A **learned-φ arm runs as an ablation, not as
the default** — the pass-3 measurement is that the task-strong encoder is the address-WORST arm beyond
2 SE, and this wave does not re-buy that lesson. φ parameters are on the byte ledger of **every** arm
including the launders.

**Read/write budget:** memory operations at **chunk granularity** (charter §2.2 — as Titans-class
memories do; fair). Chunk size `C`, write inner steps, and read steps are fixed by a **pricing probe
that runs before any claim cell** and are then frozen and declared in the flag-provenance table. Every
γ-band statement is **read-budget-scoped** (§A18.1). Any promotable cell uses **write inner steps ≥ 40**
(N94's undemoted floor); a cell below it is labelled non-promotable with its reason string.

---

## 2. The benchmark (from the landed scout shortlist; admissibility case in §2.3)

**PRIMARY: INSECTS `incremental-reoccurring` (balanced) + `incremental-abrupt-reoccurring` (balanced)
as the second condition; `out-of-control` as the published drift-free null.** Souza et al. (2020),
DMKD 34(6):1805–1858, USP Data Stream Repository, CC BY 4.0. Balanced reoccurring: **79 986**
instances, **33 features**, **6 classes**, published change points **26 568 / 53 364**; the regime
variable (temperature 20→40 °C, three cycles) is **withheld from the features** by construction.
Imbalanced reoccurring (452 044) is a declared **stretch condition**, from the USP repo directly
(river does not ship it).

⚠ **The null's label set differs:** `out-of-control` has **24 classes**, not 6. Therefore only the
**persistent − episodic contrast** is compared across streams; absolute accuracies are never compared
between streams.

**SYNTHETIC (the second stream, mandatory, and it is a MECHANICS INSTRUMENT ONLY):** a scripted
regime-switcher with `R` hidden regimes over a shared input space (same X region → different y per
regime), an exact revisit schedule, capacity pressure (`R` > well budget), and a drift-free control
condition. ⛔ **Per §A14.8 (gyms heavily demoted), the synthetic is a regression/mechanics instrument
and is NEVER a claim venue.** All designed negatives run here.

### 2.0 If B2 fires — the disposition is registered here (Head ruling 1, 2026-08-10)
⭐ **The Metro fallback is PRE-AUTHORIZED: it fires without a Head round-trip.**
⛔ **INSECTS is NOT re-argued via "our learned φ moves the keys out of the input metric."** That
escape route is closed **by measurement, not only by doctrine:** at pass 3 the settle equalled
same-keys nearest-neighbour to **±0.0007** at strong φ ⇒ **a better encoder does not escape
metric-nativeness, it relocates where nearest-neighbour operates.**
⭐ **And INSECTS is not discarded — it is FILED as a registered admissibility finding**
("INSECTS is metric-native at matched bytes"), the same class of product as FB4's protocol-invalidity
result: useful to the field, and the **fifth confirmation of the criterion-4 theorem**, obtained for
about an hour of CPU.

**FALLBACK (fires only on §3's B2): Metro Interstate Traffic Volume** (UCI, CC BY 4.0, 48 204 hourly
records) under the **hidden-clock, 24-h-horizon protocol** (`date_time` withheld so persistence
cannot win). ⚠ Its regime annotation would be **ours, not the literature's** — so the fallback
carries a mandatory precondition: a **drift map** (Webb-style drift magnitude between windows) is
published **before** any retention claim on it, and every artifact states that the annotation is ours.

### 2.1 Metrics (the scout's requested ruling, adopted)
**Prequential accuracy (sliding window 1 000, Souza's convention) + κ_per + κ⁺**, with the
**No-Change persistence baseline mandatory in every table**:
`κ_per = (p − p_per)/(1 − p_per)`; `κ⁺ = sqrt(max(0,κ)·max(0,κ_per))` (Žliobaitė et al. 2015, Eqs. 13–14).
⚠ **Žliobaitė Prop. 8 hazard, registered:** under temporal dependence, *false* drift alarms can RAISE
accuracy and drift-detector guarantees are void. ⇒ **every C2W10 table reports the controller's event
counts (admissions / evictions / promotions / demotions / trash routings) beside the accuracy**, so an
alarm-frequency artifact is visible rather than inferred.

### 2.2 Retention and adaptation (the two VALUE curves — operationalisation declared here)
The published schedule is: cycle 1 temperature 20→40, cycle 2 40→20, cycle 3 20→40, with change points
at 26 568 / 53 364. Each cycle is split into `B = 5` equal **bands** by position within the cycle
(position is linear in the hidden temperature by construction). Revisit pairing:
`(cycle 1, band b) ↔ (cycle 2, band B−1−b) ↔ (cycle 3, band b)`.
⚠ **Declared as OUR construction** derived from Souza's verbatim schedule text — band-level alignment
is not itself published, and every artifact says so.
- **Retention `R(b)`** = prequential accuracy over the **first 1 000 instances after re-entering** band
  `b`, **minus** the accuracy over the **last 1 000 instances of that band's first visit**.
- **Adaptation `A(b)`** = instances to reach **90 %** of the band's asymptotic accuracy after entry.
  ⛔⛔ **Measured PER-INSTANCE-SINCE-CHANGE, never per-stream-position** (Head ruling 5's hazard):
  decimation compresses the drift timeline — a change that took 1 000 instances takes `1 000/m` after
  decimation — so a position-indexed adaptation number **silently inflates apparent adaptation speed**.
  Internal comparisons are safe either way; **any literature-facing sentence is not**, and the
  per-instance-since-change form is the only one that may leave this program.
- ⛔ **Retention and adaptation are reported as a PAIR, always.** A retention gain bought with an
  adaptation cost beyond 2 SE is a stability-plasticity trade, not a win, and is reported as the trade.

### 2.2b Decimation — ACCEPTED with four binding conditions (Head ruling 5, 2026-08-10)
Truncation is **refused**: it would delete the third cycle, which is the revisit, which is the
benchmark. Uniform decimation (every `m`-th instance) is the funded alternative, under four conditions:
1. **`m` is declared in this prereg BEFORE any result.** The registered ladder is
   **`m ∈ {1, 2, 5, 10}`**; the pricing probe selects the smallest `m` meeting the wall-clock target,
   and the selected value is filed as a dated line in §9's amendment log — **before the first claim
   cell**, never after seeing one.
2. **Structure preservation is ASSERTED IN A TEST, not claimed:** at the chosen `m`, all three cycles
   and both change points are present, **with counts**, in a pytest that fails at an `m` that breaks
   either.
3. **Identical decimated stream for every arm and every baseline** (one frozen file, one sha256).
4. **Decimation travels in the ledger with every number** that derives from it.
⚠ Consequence carried into §2.2: adaptation is reported **per-instance-since-change**, never
per-stream-position.

### 2.3 Admissibility case (intervention §6, one criterion at a time)
1. **Strong baselines that do well** ✅ — ARF 77.13 % on inc-reoccurring (bal.) against 16.7 % chance;
   Leveraging Bagging 72.30 %; MLP/online-DL is a published laptop-scale family on the same stream
   (arXiv:2405.17222); Mamba/GRU/sliding-window attention drop in as sequence models over the 33-D
   stream. Our tier-iii **system-level swap** (matched-both TTT-class cell) is an additional live arm.
2. **Real headroom** ✅ — ARF ≈ 77 % vs a per-regime oracle ≈ 90 % (Souza Tables 3/4: pooled 84 % vs
   per-temperature 90 %; 66 % vs 86 % at 24 °C) ⇒ ~13 points of regime-conditioning headroom, and
   **persistence sits at 40.46 %**, i.e. the ELEC2 pathology is measured absent (this is criterion 2's
   clearance, and it is why ELEC2 itself is excluded — persistence 85.3 % there beats 10/12 MOA
   adaptive classifiers).
3. **Memory management over time is the difficulty** ✅ — three cycles over one hidden temperature
   sweep with published change points, and the *only* source of drift is that hidden variable (all
   others eliminated by uniform within-temperature sampling). Retention across a full cycle (≈26.5 k
   instances) is "did the store keep regime-1 knowledge alive through regimes 2–3", not single-shot
   lookup. The eleven INSECTS streams share one feature space and one hidden regime axis, so the
   persistent-vs-episodic contrast runs **across stream boundaries** — the §A21 wording ("the store
   survives document boundaries") satisfied literally.
4. **NOT metric-native** ⚠ **ARGUED, AND THEN TESTED — this is §3's B2 and it is a hard gate.** The
   favourable structure is measured: the label is not a function of the query point alone (36 % pooled
   class overlap vs 23 % per-regime), and the regime variable is withheld, so the addressing
   information exists only in accumulated stream context. But **no SAM-kNN/kNN_S number exists on
   INSECTS anywhere**, and that family beats ensembles on every other real tabular stream. ⇒ the
   scout's pre-registered tripwire runs FIRST, and it decides. ⛔ The admissibility case rests on
   *memory management over time*, **never** on the data being "physics-like" — that framing stays
   banned (charter §2.2).
5. **Every lever can be active** ✅ — address encoder, controller admit/evict/promote/demote/trash,
   lifetimes, capacity pressure over three cycles, and the trash region's first experimental use.

---

## 3. BENCHMARK GATE (spoke S1 — legs labelled; B-legs are MECHANICS **of the benchmark**)

| leg | label | statement | decision rule |
|---|---|---|---|
| **B1** loader positive control | MECHANICS | our prequential harness reproduces Souza Table 5 on inc-reoccurring (bal.): **No-Change 40.46**, **ARF 77.13** | within **±2.0 points** of both ⇒ PASS. FAIL ⇒ a loader/ordering defect (Souza's own MOA-Poker-hand warning) and **nothing on this stream is quotable** until fixed |
| **B2** criterion-4 tripwire | MECHANICS | SAM-kNN (published defaults k=5, L_min=50, **L_max=5000**) and kNN_S (L ∈ {5000, 1000}) vs ARF, prequential window 1 000 | exemplar store **within 2.0 points of ARF** (or above it) ⇒ **criterion 4 FIRES** ⇒ INSECTS cannot host the VALUE claim. Exemplar clearly **below** ARF ⇒ criterion 4 cleared **on evidence** (and, inverting §2.3's real-stream family, that is itself a reportable observation) |
| **B3** temporal-dependence sanity | MECHANICS | κ_per of ARF on the stream | κ_per > 0 required; κ_per ≤ 0 ⇒ the stream is persistence-trivial and is excluded like ELEC2 |
| **B4** byte ledger | MECHANICS | exemplar state bytes computed, not assumed: `L_max × (33×4 + 1) B` = **665 000 B ≈ 0.634 MiB** at L=5000; second budget point at L=1000 ≈ 133 kB | reported, both budget points, in the gate file |

**Environment isolation + the reproduction gate (Head ruling 6, 2026-08-10 — and the precedent is
stronger than the scout's citation).** `pyproject.toml` records that declaring **one** extra moved the
locked `pandas` **3.0.3 → 2.3.3 project-wide**, because uv resolves one version for the whole lock.
`river` pulls numpy/pandas/scipy ⇒ a re-resolution would change the numerical environment underneath
**every banked float32 result in the program**. Therefore:
- ⛔ **`river` never enters `pyproject.toml` / `uv.lock`.** Baselines run in a **scratch venv**, whose
  **exact `river` version AND full frozen package list** are recorded in the artifact.
- **The sha256'd stream file is the contract between the two environments.**
- ⭐ **REPRODUCTION GATE (the reader-fitting-audit pattern — reproduce first, compare second):** the
  harness's own loader must **reproduce that sha256** before **any** baseline number from the scratch
  venv is consumed by any C2W10 cell. A mismatch is a hard stop, not a tolerance.

**Deliverable (the mechanical precondition file, exact path):**
`.claude/outputs/c2w10-benchmark-gate/BENCHMARK-GATE.json` — containing `b1_pass`, `b2_margin_pts`,
`criterion4_cleared` (computed by B2's arithmetic, **not** by judgement), `b3_kappa_per`, the byte
ledger, the change-point index map, the band map of §2.2, and the sha256 of the frozen stream file.

⭐ **Hub ruling, registered here:** **criterion 4 gates the VALUE surface, not the MECHANICS build.**
The lifecycle mechanics (§4) run on the synthetic instrument and on INSECTS regardless of B2, because
they are component pass/fail legs with designed negatives and take no performance control. If B2
fires, the VALUE venue moves to the fallback and the MECHANICS work is unaffected.

---

## 4. LIFECYCLE MECHANICS (spoke S2 — every leg MECHANICS; kill-conditions BUILT FIRST)

**Standing doctrine applied (§A12): the designed negatives are committed and green BEFORE the verbs
they can kill are wired into the rig.** The spoke reports its commit order, and a leg whose designed
negative cannot fail **does not ship** (the defect class caught twice in C2W8: the vacuous `M`, the
addressability-blind gate).

The three states are **PROTECTED ⇄ ACTIVE → TRASH** (§A34.3). PROTECTED = no decay (`leak = 0`, the
existing permanent flag); ACTIVE = designed decay; TRASH = routed to `γ_φ(q)` / pruned.
⛔ **Demotion is PROTECTED → ACTIVE (re-exposure to designed decay), NEVER to trash.** Trash is
reserved for the never-useful / spurious.

| leg | label | statement | designed negative (pytest-asserted, mandatory) |
|---|---|---|---|
| **L1 PROMOTION** | MECHANICS | ACTIVE → PROTECTED on **sustained** usage: `read_hits` in the trailing window ≥ `h_hi` sustained over ≥ `d_dwell` chunks (hysteresis) | a well reaching `h_hi` in a **single burst** does not achieve PROTECTED; a well below `h_lo` never promotes |
| **L2 DEMOTION** ⭐ | MECHANICS | PROTECTED → ACTIVE within `d_demote` chunks of usage falling below `h_lo`, and the demoted well's depth then follows the designed decay law to within the netting tolerance | ⛔ **the rich-get-richer negative: an early-popular-then-abandoned well MUST demote.** A planted well with high stream-1 usage and zero usage thereafter must be ACTIVE (not PROTECTED) by `d_demote`, and must **NOT** be trashed by demotion |
| **L3 TRASH** | MECHANICS | a well **never useful since first appearance over `k` stream boundaries** routes to `γ_φ(q)` (§A20.6 Head addition), computed from cross-stream usage telemetry | (a) useful-in-stream-1-only ⇒ trashed at `k` (the intended positive); (b) **useful in every stream ⇒ NEVER trashed**; (c) **censoring guard: a well admitted in the last stream (age < `k` boundaries) is NEVER trashed** — never-useful-YET ≠ never-useful |
| **L4 PROTECTED FRACTION** | MECHANICS | the protected fraction is bounded at `f_max` (Hub default **0.25** of budget, spoke may re-derive and must declare); breaching it **refuses further promotion** and trips a new named monitor row **`protected_saturation`** | forcing every item's usage high must **trip the monitor and refuse**, never silently protect all (the anti-collapse leg of §A34.3) |
| **L5 I1 REFRESH-MONOTONICITY** | MECHANICS | §A23.2: a write into an existing well **never reduces its depth** (netted); rewrites refresh/deepen up to budget | with the guard OFF, a planted destructive rewrite **must** reduce depth — a guard that cannot be shown to fail is not a guard |
| **L5-b I1 CROSS-IMPLEMENTATION** ⭐ | MECHANICS | ⛔⛔ **[ERRATUM, 2026-08-11, Add.14 §A40(a) — ADVISOR-OWNED, RELAY ERROR, CORRECTED IN PLACE: the numbers in the struck text below are the `p1_off` GUARD-OFF BASELINE, not the I1 arm. ✅ THE I1 ARM IS `44 / 62 / 59` REWRITE EVENTS WITH `6 / 0 / 0` PRE-GUARD VIOLATIONS, and charter §A22 is CORRECT AS PRINTED — the error entered in the relay and propagated here and into the task file. L5-b's equivalence test binds to the CORRECTED I1 numbers. ⭐ And the bonus this wave's spoke earned: §A23.2's "zero post-guard violations" was NEVER in the artifact (`n_rewrite_violations_post_guard` is `null` in all three C2W6 files); the spoke recomputed it from raw as `0/0/0`, so that charter claim HAS EVIDENCE FOR THE FIRST TIME.]** ~~Head correction, 2026-08-10: the `blocks.py:683` flag is NOT unexercised — C2W6's `p1_on_i1_on` cell measured I1 directly through `exp_anti_erosion.py`: rewrite events 27/40/70 per run, violation rates 0.593/0.050/0.043, mean 0.228 ± 0.182 (inside the parent prereg's registered 10–40 % band), with zero post-guard violations.~~ **The substantive point stands unchanged: the flag IS exercised and its evidence is INHERITED, not re-derived.** That evidence is **INHERITED, not re-derived**, and the store-level guard is **validated against it** on the same designed rewrite events | the two implementations may not silently diverge: a divergence beyond the declared tolerance on the same events **fails the leg**. ⚠ Reconciliation the spoke owns first: Add.7 §A22's destructive-rewrite row (OFF `[0.593,0.050,0.043]` → ON `[0.027,0.0,0.0]`) is the **P1** arm while §A23.2's "0 post-guard violations" is the **I1** leg — **confirm from the raw C2W6 artifact which arm each number belongs to before writing the equivalence test against it** |
| **L6 NETTING** (BUILD REQUIREMENT) | MECHANICS | Add.9 §A27.1: **every** depth curve is emitted **raw AND netted**; the netting replays the per-item decay log with the exponent's `last_write_chunk` drift | netted ≡ raw **bitwise** when `leak = 0`; netted > raw when `leak > 0` and Δt > 0; a well with no writes nets to the analytic `exp(−leak·Δt)` to 1e-9 |
| **L7 OFF BIT-IDENTITY** | MECHANICS | `persistent_store=False` **and** every lifecycle verb OFF is **bit-identical and parameter-count-identical** to current `main` behaviour (the K2 pattern) | the existing `γ_φ` OFF regressions stay green; a new pairing test asserts the lifecycle's OFF path |

⛔ **No merge verb and no prune-below-budget-by-depth verb is built this wave.** C2W8's `M` was
vacuous (it equalled the same-class pair rate exactly) and **K9 is the registered merge criterion**;
merges stay deferred until K9 is re-registered at an operating point where its geometric leg binds.
⛔ **The optimizer's erosion is churn, not curation (§A28.3(iii)) — it is never a deletion policy.**

**K-C (registered in advance, the C2W8 lesson):** if the trash verb's target population is **empty**
at the measured operating point, the verb is reported as **UNEXERCISED**, not as working. A verb whose
target set is empty is not evidence.

**Deliverables (exact paths, and the consuming spokes gate on these strings verbatim):**
- `.claude/outputs/c2w10-lifecycle/LIFECYCLE-MECHANICS-DONE.json` — per-leg boolean table L1–L7 with
  `lifecycle_mechanics_done` computed **mechanically** as the AND; anything not landed is `false`
  **with its reason**, never omitted and never quietly true.
- `.claude/outputs/c2w10-lifecycle/USAGE-TELEMETRY.json` — the cross-stream usage record for §5:
  per-item `hits_by_stream`, `first_seen_stream`, per-item raw AND netted depth curves, `n_live` per
  seed per measurement point, `n_seeds`, `n_live_max`.

---

## 5. I2 RE-MEASUREMENT (spoke S3 — the caveat's only MEASUREMENT venue; the Advisor amends; Add.7 ruling 5)

**Context, stated exactly:** the depth-as-feature-importance caveat is ACTIVE because of a specific
hypothesised mechanism (the Head's: gradient magnitude ∝ contribution, so under a net-cost store the
MOST useful wells erode FASTEST). C2W6 returned `NO_USAGE_STRUCTURE` at live-well counts too small to
test it, and Add.7 ruling 5 deferred the re-measurement to **this wave, where live-well counts are
large enough**. ⛔ The caveat stays ACTIVE unless the refutation branch fires, **and the Hub lifts it,
not a spoke.**

| leg | label | statement | rule |
|---|---|---|---|
| **I2-a POWER** | MECHANICS (instrument) | the measurement point has **`n_live ≥ 64` per seed, ≥ 3 seeds** | not met ⇒ **UNDERPOWERED, a declared NOT-RUN**, caveat stays ACTIVE. At `n = 64` the Fisher-z 2-SE half-width is `2/√(n−3) ≈ 0.25` — the smallest honestly detectable \|ρ\| is ≈ 0.25, and that number is quoted with every ρ |
| **I2-b ICC** | MECHANICS (instrument) | `ICC(1,1)` of the usage proxy is reported **beside** any LOO number | **ICC ≤ 0 ⇒ the LOO leg is labelled `UNDEFINED` and no ρ is quotable** (the C2W6 lesson: `ρ(LOO) = +0.067` was UNDEFINED, not a null) |
| **I2-c EROSION** | the test | `ρ(U_i, E_i)` over live wells: `U` = item-id-keyed `read_hits` (**depth never enters `U`**), `E` = per-well erosion rate on the **netted** depth curve | **CONFIRM** (Head's hypothesis): ρ ≤ −0.2 beyond 2 SE on ≥ 2/3 seeds ⇒ useful wells erode fastest ⇒ ⛔ **the caveat HARDENS** (depth is anti-correlated with importance) |
| **I2-d DEPTH-USAGE** | the test | `ρ(U_i, depth_i)` on the netted curve, same population | positive evidence leg — see the lift rule |

**⚖ THE LIFT RULE (registered here, and it is deliberately hard):** the caveat lifts **only** if
**BOTH** hold, on **3/3 seeds**, with I2-a and I2-b satisfied:
1. **I2-c's negative correlation is refuted** — the 2-SE lower bound on ρ(U, E) is above **−0.10**; and
2. **I2-d is positive** — ρ(U, depth) ≥ **+0.30** beyond 2 SE.
Rationale: absence of an anti-correlation is not evidence of a positive one, and the caveat's content
is "depth may not be read as importance". Anything else ⇒ **INDETERMINATE ⇒ caveat stays ACTIVE**, and
that outcome is reported as a result, not as a shortfall.
⛔⛔ **AUTHORITY (Head ruling 3, 2026-08-10 — corrects this Hub's §5 as first filed): the caveat lives
in charter §A23.5, an ADVISOR-OWNED document ratified by the Head. ⇒ THE HUB MEASURES AND PROPOSES;
THE ADVISOR MAKES THE AMENDMENT.** No Hub, and no spoke, edits §A23.5.
**Two conditions carried from C2W6, binding:** ⭐ **only a POSITIVE-STRUCTURE finding lifts anything —
a second `NO_USAGE_STRUCTURE` leaves the caveat exactly where it is** (this is why the lift rule below
requires I2-d's positive leg and not merely the refutation of I2-c); and ⛔ **any ICC ≤ 0 reading is
`UNDEFINED`, not a null, and is never quotable as a measured correlation.**

**Scope of any proposed lift:** the proposal is **configuration-scoped** ("the persistent-store rig at d = 12, at
the declared chunk/read budget, at `n_live ≥ 64`"), stated with its scope every time it is invoked,
and **posted for Advisor ratification at the wave review**. ⛔ It is never a program-wide lift.

---

## 6. THE VALUE SURFACE (spoke S4 — tier iii, with the tier's own control)

⛔ **§A33.1: VALUE legs exist only at tier level with the tier's own control.** No component gate in
this wave has a launder margin as a pass condition; A3-style margins are DIAGNOSTIC columns.

### 6.1 GO / NO-GO precondition (mechanical, computed BEFORE any VALUE cell)
The C2W8-pass-3 lesson: a null is worth reporting only if it is **attributable**. All four required:
1. **Store non-inert** at the operating `d`: median depth after the first writes > 0.1 on 3/3 seeds
   (the d≥12 inertness arithmetic of §7.30 is why this is a precondition and not an assumption).
2. **Addressing lives**: the per-feature **G-ADDR MECHANICS instrument** (§A34.8 — permanently barred
   from VALUE duty) reports correct-basin rate above its designed floor on the stream-1 items.
3. **L6 netting green** (no erosion/flattening/recovery statement is legal without it).
4. **`BENCHMARK-GATE.json` present** with `criterion4_cleared` and the Hub's venue amendment line.
**NO-GO ⇒ the VALUE cells are a declared NOT-RUN naming the failing leg.** ⛔ Never a null.

### 6.2 The legs
⭐⭐ **Two readings, and they carry different weight (Head ruling 2):** the **±persistence contrast is
INTERNAL** (same harness, same scale, both arms) and is **this wave's PRIMARY evidence** — it is clean
and it is promotable as a within-harness result. The **absolute competitiveness against the rival arms
is SCALE-SCOPED and explicitly NON-PROMOTABLE to the CSF3 tier-iii claim**, and that sentence travels
beside every absolute number in every table and every draft.

| leg | label | statement | falsifier |
|---|---|---|---|
| **V1 RETENTION** | VALUE (tier iii) | `R(b)` (§2.2) at matched bytes: persistent CLU vs **its own episodic toggle** (the ablation) vs the **system-level swap** (matched-params AND matched-state-bytes TTT-class cell) vs **SAM-kNN** (persistent exemplar) vs **kNN_S** (episodic exemplar) vs **ARF** vs **No-Change** | persistent − episodic ≤ 0 at 2 SE; **or** SAM-kNN ≥ persistent CLU at matched bytes on revisit-recovery (the scout's tripwire) |
| **V2 ADAPTATION** | VALUE (tier iii) | `A(b)`, same arms, reported **paired with V1** | a retention gain with an adaptation cost beyond 2 SE is reported as a stability-plasticity trade, not a win |
| **V3 DRIFT-FREE NULL** | VALUE (control) | on `out-of-control` (published: *"this dataset must be drift-free"*), persistent ≈ episodic, \|Δ\| ≤ 2 SE | ⭐ **a "win" here falsifies the instrument, not the baseline** — it means the contrast is measuring something other than regime revisit |

### 6.3 The byte ledger — two-sided, both budget points, no side picks its favourite
`min_atoms ∝ √2^d` forces the store's size: at **d = 12**, `n_atoms = 512·√2^12 = 32 768` and store
bytes `= n_atoms × (dim+2) × 4` with `dim = addr+payload = 13` ⇒ **1 966 080 B ≈ 1.875 MiB**, i.e.
**2.95× SAM-kNN's published 0.634 MiB budget**. Therefore:
- the exemplar arms run at **BOTH** the published budget (L=5000) **and** the CLU's own byte count
  (≈ **14 782 exemplars** — the spoke computes the exact number), because a hobbled null is the same
  referee attack in mirror image (the **F3 anti-hobbling rule**);
- if the CLU cannot be shrunk to 0.634 MiB at d = 12 (the `min_atoms` floor), that is stated as
  **NOT-REACHABLE with its arithmetic** and reported as an explicitly labelled **byte-frontier
  column, never a dividend family** (the §A14.2 pattern);
- **φ parameters, the codebook, and `trash_bytes = K·(dim+2)·4` are on every arm's ledger**, launders
  included.

### 6.4 Seeds and pairing
Arms are **paired on the bit-identical stream** (pass-3 practice). **≥ 3 seeds for MECHANICS; ≥ 5 for
any VALUE number**; and per **A17.4**, any leg whose control carries learned-init variance uses paired
or multi-init controls, or `n ≥ 9`. ⛔ No rescue verdict at n = 3 (§A18.1).

---

## 7. Numeric predictions (registered BEFORE the cells; scored at the review either way)

| # | prediction | Hub's prior |
|---|---|---|
| **Q1** | B1 reproduces Souza's No-Change/ARF within ±2.0 pts | 0.75 |
| **Q2** | B2 clears criterion 4 (exemplar store clearly below ARF on INSECTS) | 0.55 — genuinely uncertain; it is the one number nobody has published |
| **Q3** | L2's rich-get-richer negative passes on the first build | 0.70 |
| **Q4** | L3's trash population is **non-empty** at the measured operating point (K-C does not fire) | 0.45 — C2W8 measured `P = 0.0208`; the cross-stream criterion is more permissive, but this is the leg most likely to be unexercised |
| **Q5** | I2 returns **INDETERMINATE** (caveat stays ACTIVE) | 0.60; CONFIRM 0.25; **a lift PROPOSAL goes to the Advisor** 0.15 |
| **Q6** | V1: persistent − episodic retention margin > 0 at 2 SE | 0.40 |
| **Q7** | V1: the CLU beats byte-matched SAM-kNN on revisit-recovery | 0.20 — four substrates of "no daylight" say the honest prior is low, and a null here is a reportable tier-iii datum, not a failure |
| **Q8** | V3's drift-free null reproduces (\|Δ\| ≤ 2 SE) | 0.80 |

---

## 8. Declared NOT-RUNs (never to be reported as nulls)

`d = 16` (measured inert, 131 072 atoms) · merge verbs / K9 re-registration · prune-below-budget by
depth · the anytime/compute-adaptive curve (C2W9's) · wormholes and learned-`p₀` traversal (C2W9's) ·
the compositional cat test and the organizer swap (C2W11's) · any tier-ii verdict · any full-CLU
verdict · CSF3 submission or any scale claim · the imbalanced-reoccurring stretch condition if the
pricing probe does not fund it · Metro Interstate unless B2 fires.

---

## 9. Amendment log

*(append-only; each amendment dated, labelled AMENDMENT, and naming the Hub who filed it)*

- **2026-08-10 — AMENDMENT 1 (C2W10 Hub), the Head's six rulings + one correction, folded in place:**
  (1) §2.0 added — the Metro fallback is **pre-authorized** (no round-trip), the learned-φ escape from
  criterion 4 is **closed by measurement** (settle = same-keys NN to ±0.0007 at strong φ ⇒ a better
  encoder relocates where nearest-neighbour operates, it does not escape), and a fired B2 **files
  INSECTS as a registered admissibility finding**, the fifth confirmation of the criterion-4 theorem.
  (2) §0 + §6.2 — the swap control is valid **by its properties, not its file** (whole memory
  replaced, two-sided match, **TTT-class never GRU**); **the ±persistence contrast is the internal
  PRIMARY**, the absolute rival reading is **scale-scoped and non-promotable to CSF3**.
  (3) §5 — **authority corrected: the Hub measures and proposes, the ADVISOR amends §A23.5**; only a
  **positive-structure** finding lifts anything (a second `NO_USAGE_STRUCTURE` changes nothing);
  ICC ≤ 0 is `UNDEFINED`, never a null.
  (4) sequencing — **C2W11 has first worktree claim** when its gate clears; C2W10 sequences behind and
  **does not idle**: preregs, the loader + sha256 freeze, the drift map and **above all the B2
  tripwire run early** (discovering an inadmissible venue after the lifecycle build is the expensive
  ordering, and it is free to avoid).
  (5) §2.2b added — **decimation accepted, truncation refused**, four conditions (`m` declared here
  before any result, ladder `m ∈ {1,2,5,10}`; structure preservation **asserted in a test**; one
  decimated stream for every arm; decimation in the ledger) + §2.2's hazard: **adaptation is
  per-instance-since-change, never per-stream-position**.
  (6) §3 — `river` stays out of the lock (the recorded project-wide `pandas` 3.0.3 → 2.3.3 precedent;
  a re-resolution would move the numerical environment under every banked float32 result), scratch
  venv with its **full frozen package list** recorded, and a **REPRODUCTION GATE**: the harness's own
  loader must reproduce the stream sha256 **before** any baseline number is consumed.
  (7) §4 — **L5-b added**: the `blocks.py:683` I1 flag is **NOT unexercised** (C2W6 `p1_on_i1_on`:
  27/40/70 rewrite events, violation rates 0.593/0.050/0.043, mean 0.228 ± 0.182 in the registered
  10–40 % band, zero post-guard violations); that evidence is **inherited** and the store-level guard
  is **validated against it**, with the P1-vs-I1 arm-labelling reconciliation owed first.
- **2026-08-11 — AMENDMENT 3 (C2W10 Hub), filed on the landed B-legs. ⚖ `criterion4_cleared = FALSE`:
  CRITERION 4 HAS FIRED ON THE PRIMARY.** Best registered exemplar arm SAM-kNN L=5000 (**0.634 MiB**)
  = **76.9157 %** vs ARF-100 **78.8139 %** ⇒ `b2_margin_pts = −1.8983`, inside the registered −2.0 rule.
  ⭐ **The verdict is ROBUST, not a hair's-breadth call: the tripwire fires against EVERY ARF reference
  tried** (−0.2143 vs the published ARF 77.13 · −0.4972 vs our ARF-10 · −1.8378/−1.9366 worst/best
  seed), and the margin closest to the threshold is the one against our **strongest** ARF, i.e. the
  most generous possible reference for clearing. B1 PASSED both conditions (No-Change **40.4526** vs
  published 40.46; ARF Δ +1.684). B3 PASSED (ARF κ_per **0.64422**).
  ⇒ **`AMENDMENT — VALUE VENUE:` the INSECTS streams are NOT the VALUE venue. They are FILED as a
  registered admissibility finding — "INSECTS is metric-native at matched bytes" — the fifth
  confirmation of the criterion-4 theorem, bought for ≈4 h of CPU.** The **Metro fallback fires**
  (pre-authorized, ruling 1). ⛔ **INSECTS remains fully admissible as a MECHANICS substrate** (§3's
  Hub ruling: criterion 4 gates the VALUE venue, not the mechanics build).
  ⛔⛔ **BUT THE VENUE IS NOT YET FIXED: Metro must pass THIS SAME TRIPWIRE FIRST.** A 24-h-horizon
  regression stream with a hidden clock has an obvious nearest-neighbour-over-past-windows attack, the
  scout flagged criterion 4 on it as ⚠, and it is **un-argued**. Firing a fallback without testing it
  is the exact mistake this leg just prevented at 1/50th the cost of the build it saved.
  **Also amended, on landed evidence:**
  (a) ⛔ **§6.2's V3 drift-free null: `out-of-control` HAS NO DATA SOURCE** (absent from the 2024-04-16
  USP archive and from river 0.25.0). The leg becomes a **declared NOT-RUN pending re-sourcing**, and
  the drift-free control is **re-constructed on the new venue, never quietly dropped.**
  (b) ⛔ **§2.2's band map is contaminated at `b = 4`**: the terminal band of every cycle is
  persistence-trivial and at ceiling for every arm (class entropy 0.84–1.05 bits vs 2.07–2.37 in bands
  0–3; No-Change 91.25/71.32/90.78 %; every fitted arm 97–99 %). **`R(4)`/`A(4)` are uninterpretable as
  registered and are EXCLUDED**; retention/adaptation are scored on bands 0–3.
  (c) ⚠ **ARF is not byte-matched to anything** — measured state **9 542 925 B = 14.35×** SAM-kNN's
  665 000 B. Every "ARF is the reference" sentence carries that caveat.
  (d) ⭐ **A finding worth carrying into the VALUE design: the exemplar store's byte-frontier is
  NON-MONOTONE under regime revisit** — kNN_S at the CLU's own budget (14 782 exemplars) scores
  **59.75 %** against **75.36 %** at 1 000 exemplars. More bytes HURT an exemplar store under drift.
  (e) **§4's L5-b numbers are CORRECTED** (see AMENDMENT 4).
  (f) **§4's L6 tolerance:** "nets to analytic `exp(−leak·Δt)` to 1e-9" holds only in **float64**; the
  shipped store is float32 with a floor of **≈5.3e-8**. Both are pinned; the registered 1e-9 was
  over-specified by this Hub.
  (g) **§4's L3 wording defect (mine):** "never useful **since first appearance** over `k` boundaries"
  and designed negative (a) "useful in stream 1 only ⇒ trashed at `k`" are **not jointly satisfiable**.
  Both readings ship, the default is declared, and the difference is pytest-asserted.
  (h) **§5's power precondition is MET at the floor:** `n_live_max = 64`, 3/3 seeds, reached by moving
  `d_safe_frac` 0.88 → **0.60**. No margin above the threshold ⇒ the detectable \|ρ\| ≈ 0.25 travels
  with every ρ, and INDETERMINATE remains the pre-priced modal outcome (Q5 = 0.60).
  (i) **Base test count at `9e0bb25` in a fresh worktree is 1564 collected / 1562 selected**, not the
  1555 carried from the C2W8 close entry.
- **2026-08-11 — AMENDMENT 4 (C2W10 Hub) — ⛔⛔ L5-b's INHERITED NUMBERS WERE MISATTRIBUTED, and the
  reconciliation this Hub attached to the leg is what caught it.** The figures carried into AMENDMENT 1
  (events **27/40/70**, violation rates **0.593/0.050/0.043**, mean 0.228 ± 0.182) are the
  **`p1_off` GUARD-OFF BASELINE**, not the `p1_on_i1_on` arm. **The I1 arm is 44 / 62 / 59 rewrite
  events with 6 / 0 / 0 PRE-guard violations.** Add.7 §A22's row is correct as printed; the
  misattribution entered downstream of it. ⇒ **L5-b's cross-implementation equivalence test binds to
  the corrected I1 numbers.** ⭐ **And a second finding for the Advisor:** §A23.2's *"0 post-guard
  violations"* **was never in the artifact** (`n_rewrite_violations_post_guard` is `null` in all three
  C2W6 files); it has now been **recomputed from raw as 0/0/0**, so the standing claim **has evidence
  for the first time**. ⚠ §A22/§A23.2 are **Advisor-owned: flagged here, not edited.**
- **2026-08-11 — AMENDMENT 5 (C2W10 Hub) — ⛔⛔ THE FALLBACK FIRED TOO: BOTH SHORTLISTED VENUES ARE
  INADMISSIBLE, AND THE VALUE SURFACE HAS NO VENUE.** `criterion4_cleared_metro = FALSE`, and where
  INSECTS was a 1.9-point miss, **Metro is not close**: k-NN over past windows at **0.634 MiB
  (MAE 314.58)** *beats* **every** strong baseline — tuned GBDT 335.20 (at 3.6 MB of state), GBDT 338.65,
  GRU-big 374.72, MLP 414.97 — **firing against all ~~8~~ NINE strong references** (margins −0.062 …
  −0.242), and it improves further at the CLU's own budget (1.875 MiB).
  ⛔ **[CORRECTION, 2026-08-11 — the Hub's own, caught by the curator's fold; two numeric slips, neither
  changing the verdict. (1) The robustness table carries NINE references** (`gbdt_tuned · gbdt ·
  gbdt_cat · gbdt_recent · rls · gru_big · ridge_batch · mlp · gru`), **all firing** — re-derived from
  `METRO-GATE.json`. **"all 8" was a Hub miscount from a truncated view of the array**; charter
  Add.14 §A39.2 and the source report both say nine. **(2) At the CLU's own budget the REGISTERED
  k = 5 arm (`knnwin_14894_raw`) is MAE 306.76; 300.09 is `knnk_14894_10_raw`, the best arm at that
  budget from the UNREGISTERED anti-hobbling k-ladder (k = 10).** Both beat `gbdt_tuned`'s 335.20, so
  the claim is unchanged — but **300.09 must be named as the anti-hobbling k-ladder arm and never
  attributed to the registered arm.**]** M1 passed; ⚠ M3 shows the
  tuned GBDT only **2.17 %** better than a seasonal-naive `t−168 h`, so Metro's criterion-2 headroom was
  thin independently.
  ⇒ ⛔ **THE VALUE SURFACE (§6, legs V1–V3) IS A DECLARED NOT-RUN WITH AN ATTRIBUTABLE CAUSE — no
  admissible venue — and it is NEVER reported as a null.** `c2w10-value-surface.md` is SUPERSEDED as
  scoped (it gates on a venue that does not exist); it is not deleted.
  ⛔⛔ **NO THIRD TABULAR STREAM IS SHOPPED.** Intervention §8.4 prohibits it after four purchases;
  these are the **fifth and sixth**. ⭐ **The finding is the product, and it converges with a ruling
  already on the books (Add.11 §A32.4):** the charter §2.2 fallback track — *non-stationary streaming
  prediction with regime revisits* — is, **by measurement on both shortlisted candidates, a
  metric-native family at matched bytes.** An exemplar store queried in the input metric sits at
  (INSECTS) or above (Metro) the tuned-baseline ceiling. **That retires a whole track cheaply and it is
  written up as a result.** ⚖ The venue question above this — what a persistent-store VALUE venue can
  be — is **Head-reserved** (options priced in the `[C2W10]` review entry); the Hub does not improvise.
- **2026-08-11 — AMENDMENT 6 (C2W10 Hub) — I2 = INDETERMINATE; ⛔ THE CAVEAT STAYS ACTIVE; and two
  defects in §5 are MINE.** `i2a_pass = true` (n_live 64/64/64, 3/3 seeds), `i2b_pass = true` (ICC
  0.280 / 0.447 / 0.466, positive 3/3 — unlike C2W6), `lift_rule_satisfied = false`.
  - **ρ(U, depth) = −0.036 / +0.103 / +0.020**, every CI straddling zero ⇒ **no positive structure ⇒
    nothing lifts**, exactly as the Head's ruling-3 condition requires. **This is the second
    independent no-usage-structure result for depth** ⇒ ⛔ **§A23.5 REMAINS ACTIVE.** ⭐ It also
    vindicates by measurement the L3 design choice to key the trash criterion on `read_hits` and
    **never** on depth.
  - **ρ(U, E) = −0.240 / −0.283 / −0.325** — **all three negative, and 2 of 3 have 2-SE upper bounds
    below zero** (−0.026, −0.065; perm p = 0.014 and < 0.05). ⇒ the Head's I2 hypothesis is
    **directionally supported and significant against ZERO on 2/3 seeds, but does not clear the
    registered −0.2 threshold**, so the registered branch is **INDETERMINATE** and that is what stands.
  - ⚠ **DEFECT (a), mine — an ambiguity the spoke resolved in the stricter direction and was right to:**
    §5's *"ρ ≤ −0.2 beyond 2 SE"* admits two readings (2-SE interval excludes **zero** with a point
    estimate ≤ −0.2 ⇒ CONFIRM fires 2/3; or 2-SE interval excludes **−0.2** ⇒ nothing fires). The
    charter's usage elsewhere (*"clears 0 beyond 2 SE"*) means **the interval excludes the threshold**,
    so the strict reading governs and **INDETERMINATE stands**. Registered here so the choice is on the
    record and not silent. Every future prereg states which quantity the interval must exclude.
  - ⚠ **DEFECT (b), mine — a threshold/power mismatch:** at the power precondition's floor (n = 64 ⇒
    2-SE half-width ≈ 0.27) clearing `hi_2se ≤ −0.2` needs ρ ≈ **−0.47**. The registered CONFIRM bar was
    therefore **close to unreachable by construction** at the very n the same prereg registered as
    sufficient. The INDETERMINATE verdict is honest but it is **partly an artifact of my own design**,
    and any future I2 measurement sets the threshold and the n **jointly**.
- **2026-08-11 — AMENDMENT 7 (C2W10 Hub), the close relay (charter ADDENDUM 14 §A39–§A41, Head-ratified).**
  ⚖ **C2W10 CLOSES AS A MECHANICS WAVE.** Banked and quotable: the three-state lifecycle (7/7 legs,
  kill-conditions committed before the verbs, **L4 labelled UNEXERCISED**) · the trash region's
  cross-stream plumbing · the I2 instrument and its INDETERMINATE verdict · both drift maps.
  ⛔ **The VALUE surface is a declared NOT-RUN for want of an admissible venue — never a null, in any
  artifact, ever.**
  ⚖ **The streaming-prediction VALUE track is CLOSED; no third venue.** Criterion 4 now stands at
  **six** confirmations (MAD/zoology/MQAR · three of four gym families via FB4 · ELEC2 + Covertype +
  Poker-hand · the SAM-kNN real-stream family · **INSECTS, measured here** · **Metro, measured here**).
  ⭐ **Framing, ruled: this is a FALLBACK BEING RETIRED, NOT A VENUE CRISIS** — §2.2's Track-2 primary
  (bounded-state sequence modelling) is live on CSF3 and the tier-ii compositional family is live in
  C2W11.
  ⭐ **Both tripwires bank as registry findings and route to B′ as protocol evidence (FB4's class).**
  Quotable form: *"at laptop byte budgets an exemplar store at matched bytes sits at or above the
  strong-baseline frontier on both of the best-documented real streaming venues — and on one of them,
  destroying temporal order does not hurt it."*
  ⭐ **Also filed as independently useful benchmarking methodology: the 24-h embargo finding** — plain
  prequential at a 24 h horizon leaks up to 23 h of future traffic to any continuously-updated learner
  (worth **+10.9 %** to a 250-exemplar k-NN and **−0.3 %** to GBDT). ⚠ The leak is **asymmetric IN THE
  DIRECTION OF FIRING**, i.e. the analyst closed a leak that was biased toward its own tripwire.
  **And:** seasonal-naive(`t−24 h`) is **DEGENERATE** at a 24 h horizon (bit-identical to persistence);
  the non-degenerate naive is **`t−168 h`**.
  ⛔⛔ **RULING 4(b) — THE I2 DEFERRAL WAS DEFECTIVE, AND IT RE-HOMES THE MECHANISM LEG.** Add.7
  deferred I2 here *"where live-well counts are large enough"* — the **count** was checked, the
  **channel** was not. **This rig runs no outer objective and no optimizer step on store parameters, so
  the Head's mechanism (gradient magnitude ∝ contribution) cannot act here at all**; the live-well count
  was in fact adequate (57–60 scored). ⇒ **I2-c's mechanism is a DECLARED NOT-RUN at this rig, NOT a
  null**, the INDETERMINATE is partly an Advisor scoping defect, and **the mechanism leg RE-HOMES to a
  rig with a live outer objective (C2W11's organizer, or the block).** ⛔ **§A23.5's
  depth-as-feature-importance caveat STAYS ACTIVE and TRAVELS until it is measured there.**
  **NEVER-QUOTES added (fold at the curator pass):**
  ⛔ `ρ(U, depth_raw) = +0.18 / +0.22 / +0.19` **is NOT positive evidence** — below the detection floor
  (0.272 / 0.265 / 0.272) **and** confounded with age (`ρ(E_raw, age) = +0.74 / +0.94 / +0.85`) ·
  ⛔ INSECTS' band **`b = 4`** is contaminated ⇒ `R(4)`/`A(4)` uninterpretable as registered ·
  ⛔ *"ARF is the reference"* without its byte caveat (**9 542 925 B = 14.35×** SAM-kNN's 665 000 B —
  **ARF is byte-matched to nothing**) · ⛔ the scout's §5 claim that **river ships SAM-kNN** (it does
  not; a ported reference implementation was required, and every cost estimate citing *"one-line
  baseline"* is **void**) · ⛔ **`out-of-control` has NO DATA SOURCE** (absent from the 2024-04-16 USP
  archive and river 0.25.0) ⇒ the V3 drift-free-null leg is a declared NOT-RUN ·
  ⚠ **I2-d as registered tests WRITE-TIME depth, not live depth** — netting makes depth time-invariant
  on this rig (`|last/first − 1| ≈ 4.5e-7`), which is a further reason the lift could not have fired
  here on the depth leg.
  ⭐ **HANDED FORWARD as a design input (the C2W5→C2W7 pattern): the persistent-store VALUE question is
  NOT dead — it moves to COMPOSITIONAL territory.** If C2W11's tier-ii claim lands, *"does a persistent
  store help across compositional streams"* is the right question, **on a venue where a table does not
  win by theorem.**
- **2026-08-11 — AMENDMENT 8 (C2W10 Hub): two items the spoke earned that the close relay does not yet
  carry, filed so they are not lost between waves.**
  1. ⭐⭐ **A PROPOSED NEW STANDING INSTRUMENT RULE — the twin of C2W6's "ICC ≤ 0 ⇒ UNDEFINED", and it
     is Advisor-facing (proposed by the Hub, not adopted by it):**
     > **Reliability is necessary but not sufficient.** On this rig `E_netted`'s split-half reliability
     > was **0.78–0.89** while the quantity itself was **≈4 ULP of float32 round-off** (median netted
     > total log-drop **4e-7 nats** against **0.58–0.62 nats** raw — a ratio of **6.8e-7**). ⇒ every
     > erosion/usage proxy must carry **two** checks: **(a) reliability** (ICC / split-half) **and
     > (b) magnitude against the designed channel.** A proxy at **≤1e-6 of the designed channel is
     > UNDEFINED regardless of its reliability.**
     ⚠ This is the *instrument* reason I2-d could not fire here, distinct from ruling 4(b)'s *channel*
     reason (no outer objective): **netting makes depth time-invariant on this rig**
     (`|last/first − 1| ≈ 4.5e-7`), so **I2-d as registered tests WRITE-TIME depth, not live depth.**
     Any re-registration must **name which depth it means**, and any future undesigned-erosion cell
     must run the store amplitudes in **float64** — at float32 the netted channel saturates at ~4 ULP,
     i.e. **the instrument's noise floor sits above any effect it could see.**
  2. ⚠ **§5's detection floor is CORRECTED:** the registered *"≈ 0.25 detectable |ρ| at n = 64"* should
     be quoted as **0.272 / 0.265 / 0.272**. `n_live = 64` is **transient** (`n_live_end = 63` on 3/3)
     and the ≥4-readings rule scores **57 / 60 / 57**. This makes AMENDMENT 6's defect (b) slightly
     worse than stated there, not better: clearing `hi_2se ≤ −0.2` needed ρ ≈ **−0.47**.
  ⭐ Also recorded, to the spoke's credit: **it registered never-quote (R4) on `ρ(U, depth_raw)` and
  declared I2-c's mechanism a NOT-RUN at this rig BEFORE the close relay ruled either** — the artifact
  and the ruling agree because the spoke got there first, not because it was told.
- **2026-08-10 — AMENDMENT 2 (C2W10 Hub), OWED BEFORE THE FIRST CLAIM CELL:** the selected decimation
  factor `m` from the registered ladder, filed here as a dated line by the Hub once the pricing probe
  reports. ⛔ No claim cell runs until this line exists.
  ✅ **CLOSED UNEXERCISED, 2026-08-11:** no claim cell ever ran (the VALUE surface is a declared NOT-RUN
  for want of an admissible venue, AMENDMENT 5), so **no `m` was ever selected and none is owed.** Both
  gate spokes characterised the ladder **structurally only**, and every published baseline number in
  this wave is at **`m = 1`**. ⛔ This amendment is closed as UNEXERCISED, not satisfied — the
  distinction matters if the ladder is ever revived on another venue.
