# c2w10-benchmark-gate — the criterion-4 tripwire, run before the wave's VALUE venue is fixed

**Campaign 2, C2W10 ("The persistent store"). Agent:** results-analyst. **ZERO worktrees.**
No tracked-code edits: everything you write lives under `.claude/` and a scratch venv.
Writes `.claude/outputs/c2w10-benchmark-gate.md` + artifacts to `.claude/outputs/c2w10-benchmark-gate/`.
**Budget:** ≈ half a day (the pre-run itself is ~1 h; the loader + freezing the stream is the rest).
**Runs FIRST. The VALUE venue of this wave is decided by your `criterion4_cleared` boolean.**

**Binding documents, read first, in this order:**
- `.claude/outputs/c2w10-persistent-store/PREREG-C2W10.md` **§2 and §3 IN FULL** — your legs are
  B1–B4 and their decision rules are already registered; you apply them, you do not re-derive them.
- `.claude/outputs/c2w10-benchmark-scout.md` **§2.1, §2.2, §2.3, §5, §6** — the verified numbers, the
  published byte budgets, and the pre-registered reading of this very pre-run (§6, last block).
- `.claude/advisor-head-intervention.md` §6 (the five criteria) and §8 (the prohibitions).

## ⭐ DIAL DECLARATION (protocol §7) — echo before your first result
- **Dial:** none — **benchmark admissibility instrument.** ⛔ No CLU cell, no store, no dividend, no
  performance claim about the CLU of any kind.
- **Laundering control:** N/A here — **you ARE the laundering control** (the exemplar store at
  matched bytes is the wave's launder, and this task is the measurement of how strong it is).
- **Falsifies:** B1 failing ⇒ our loader/ordering is defective and nothing on this stream is quotable.
- **Does NOT falsify:** the exemplar store beating ARF would NOT falsify the CLU — it falsifies the
  BENCHMARK's admissibility as a VALUE venue (criterion 4), which is a different and cheaper finding.
- ⛔ Depth is not feature importance (§A23.5 ACTIVE). ⛔ NOT-RUNs are declared, never nulls.

---

## What to do

### 1. Get the data, freeze it, and prove what you have
- Fetch **INSECTS `incremental-reoccurring` (balanced)** — 79 986 × 33 features × 6 classes. Two
  routes: `river`'s mirror (`river/datasets/insects.py`, `incremental_reoccurring_balanced`,
  21 433 047 B) or the USP Data Stream Repository directly (CC BY 4.0; the paper's archive password is
  `DMKD2018`). Also fetch **`incremental-abrupt-reoccurring` (balanced)** and **`out-of-control`**
  (905 145 instances, **24 classes** — note the different label cardinality).
- **Freeze each stream to a single local file with a recorded sha256** under a declared cache path
  (⛔ not in git; `.claude/**` and `projects/**` are gitignored — put the frozen streams somewhere
  declared and stable, and record the path + sha256 + byte size in the gate file). Every later spoke
  in this wave consumes **your frozen file**, so the stream is identical across arms by construction.
- Record: instance count, feature count, class count, class histogram, and the **published change
  points 26 568 / 53 364** verified against the file's own length.

### 2. Environment — keep `river` OUT of the project lock, and gate on reproduction
⛔⛔ **Do NOT add `river` to `pyproject.toml` / `uv.lock`** (Head ruling 6, 2026-08-10). The precedent
is stronger than a style preference: `pyproject.toml` records that declaring **one** extra moved the
locked `pandas` **3.0.3 → 2.3.3 project-wide**, because uv resolves one version for the whole lock.
`river` pulls numpy/pandas/scipy ⇒ **a re-resolution would change the numerical environment underneath
every banked float32 result in this program.**
Use a **scratch venv**: `uv venv .claude/scratch/c2w10/.venv && uv pip install --python .claude/scratch/c2w10/.venv river`
(or equivalent). Record **the exact `river` version AND the full frozen package list** (`uv pip freeze`)
in the artifact and in your flag-provenance table.
⭐ **THE REPRODUCTION GATE (the reader-fitting-audit pattern: reproduce first, compare second).** The
**sha256'd stream file is the contract between the two environments.** Write, into
`BENCHMARK-GATE.json`, everything the project harness needs to re-open the same bytes — and state
explicitly in your report that **no C2W10 cell may consume a baseline number from the scratch venv
until the harness's own loader reproduces that sha256.** A mismatch is a hard stop, not a tolerance.

### 3. B1 — the loader positive control (run this before B2 and report it first)
Run **No-Change (persistence)** and **ARF** on inc-reoccurring (bal.) under **prequential accuracy,
sliding window 1 000** (Souza's convention) and compare to the published values:
**No-Change 40.46**, **ARF 77.13**. **Within ±2.0 points of both ⇒ `b1_pass = true`.**
⚠ If B1 fails, **stop and report**: Souza §4.2 records that MOA's Poker-hand ordering carries temporal
dependence absent from the UCI original, i.e. an ordering defect is a real and known failure mode.
Do not tune anything to make B1 pass; report the discrepancy with both numbers.

### 4. B2 — the criterion-4 tripwire (the deciding measurement)
Run, on the **same frozen stream, same prequential protocol**:
- **SAM-kNN** at its published defaults — **k = 5, L_min = 50, L_max = 5000** (Losing et al., ICDM
  2016). This is a bounded dual exemplar store: STM + kMeans++-compressed LTM.
- **kNN_S** — plain distance-weighted kNN over a sliding window, at **L = 5000** and **L = 1000**.
- **ARF** (from B1) as the reference.
**The pre-registered reading, applied mechanically (do not argue it):**
- exemplar store **within 2.0 points of ARF, or above it** ⇒ `criterion4_cleared = false` — criterion
  4 has fired on the PRIMARY and the wave's VALUE venue moves to the fallback.
- exemplar store **clearly below** ARF ⇒ `criterion4_cleared = true`. ⭐ That result would **invert**
  the pattern of every other real tabular stream in the scout's §2.3 (where a 5 000-example windowed
  kNN is 2nd of 6 by average rank and an explicit exemplar store is 1st) — **say so explicitly**, it
  is a reportable observation in its own right.
⛔⛔ **If B2 fires, what happens is already ruled — you do not need a decision and neither does the Hub
(Head ruling 1, 2026-08-10):** the **Metro fallback is PRE-AUTHORIZED** and fires without a round-trip;
⛔ **INSECTS is NOT re-argued via "the learned φ moves the keys out of the input metric"** — that
escape is closed **by measurement**: at pass 3 the settle equalled same-keys nearest-neighbour to
**±0.0007** at strong φ, i.e. **a better encoder does not escape metric-nativeness, it relocates where
nearest-neighbour operates**; and ⭐ **INSECTS is then FILED as a registered admissibility finding**
("INSECTS is metric-native at matched bytes") — the same class of product as FB4's protocol-invalidity
result, and the **fifth confirmation of the criterion-4 theorem** for about an hour of CPU. **Write it
up as a finding, not as a disappointment.**
⭐ **This is why you run first and early.** Discovering the venue is inadmissible *after* the lifecycle
build is the expensive ordering, and it is the one ordering that is free to avoid.

⚠ **Anti-hobbling (the F3 rule):** the exemplar arms get the published defaults and a genuine
implementation. A weak baseline here would be the same referee attack in mirror image. If you must
deviate from a published default for a runtime reason, **declare it and report both**.

### 5. B3 / B4 — the temporal-dependence sanity leg and the byte ledger
- **B3:** report **κ_per** and **κ⁺** for every arm — `κ_per = (p − p_per)/(1 − p_per)`,
  `κ⁺ = sqrt(max(0,κ)·max(0,κ_per))` (Žliobaitė et al. 2015, Eqs. 13–14). ARF's **κ_per must be > 0**;
  κ_per ≤ 0 ⇒ the stream is persistence-trivial and is excluded exactly as ELEC2 is.
- **B4:** compute (do not assume) the exemplar arms' **state bytes**: `L_max × (33×4 + 1) B` ⇒
  **665 000 B ≈ 0.634 MiB** at L = 5000; the second budget point at L = 1000 ≈ 133 kB. Also report
  what **14 782 exemplars** would cost (the CLU's own d = 12 byte count, 1 966 080 B) — the wave's
  two-sided ledger needs that number and you are the cheapest place to get it.

### 5b. Decimation — price it and report the ladder (Head ruling 5, 2026-08-10)
The CLU harness cannot afford an 80 k-instance prequential pass at full settle budget, and
**truncation is REFUSED** (it would delete the third cycle, which is the revisit, which is the
benchmark). **Uniform decimation** (every `m`-th instance) is the funded alternative, ladder
**`m ∈ {1, 2, 5, 10}`**. Your part:
- report, per `m` in the ladder, the **instance count, the surviving change-point indices, and the
  per-cycle counts** — this is the evidence the Hub files `m` on, before any claim cell;
- confirm that at each `m` **all three cycles and both change points survive**, with counts (the
  engineer spokes turn this into a pytest — you provide the reference numbers);
- ⚠ **run your own B1/B2 baselines at `m = 1` (the undecimated stream)** so the loader positive
  control is against Souza's published numbers; the decimated arms are an internal comparison only.
⚠ **Hazard to state in your report:** decimation **compresses the drift timeline** — a change that
took 1 000 instances takes `1 000/m`. ⇒ adaptation must be reported **per-instance-since-change,
never per-stream-position**, or decimation silently inflates apparent adaptation speed.

### 6. Deliverables the rest of the wave gates on (exact paths — a precondition that names a path the
spoke does not write is a gate that fires on a file that exists; this cost the program a session)
`.claude/outputs/c2w10-benchmark-gate/BENCHMARK-GATE.json` containing **at minimum**:
```
b1_pass, b1_no_change_acc, b1_arf_acc,
b2_arms: {samknn_5000, knn_s_5000, knn_s_1000, arf} -> {acc, kappa, kappa_per, kappa_plus, state_bytes},
b2_margin_pts (best exemplar arm − ARF),  criterion4_cleared (bool, computed by B2's arithmetic),
b3_kappa_per_arf,  byte_ledger,  change_points, band_map (PREREG §2.2),
streams: [{name, path, sha256, n_instances, n_features, n_classes}],
river_version, notes
```
Plus the **band map of PREREG §2.2** as an explicit index table: three cycles, `B = 5` bands each, and
the revisit pairing `(c1,b) ↔ (c2,B−1−b) ↔ (c3,b)`. ⚠ **Label it as OUR construction** derived from
Souza's verbatim schedule text — band-level alignment is not published, and the artifact must say so.

## FILE OWNERSHIP (declared)
**You own:** `.claude/outputs/c2w10-benchmark-gate/**` · `.claude/scratch/c2w10/**` · the frozen
stream cache path you declare.
⛔ **You touch NO tracked code.** Not `chlu/`, not `tests/`, not `pyproject.toml`, not `uv.lock`.
If you believe tracked code is needed, **stop and report** — that is a Hub scoping decision.

## Acceptance (mechanical)
1. `BENCHMARK-GATE.json` exists at the exact path above with every key listed, and
   `criterion4_cleared` computed **arithmetically** from B2, not asserted.
2. B1 reported **before** B2 in the report, with both published values quoted beside ours.
3. κ_per and κ⁺ present for **every** arm; the No-Change baseline present in **every** table.
4. Byte ledger computed from the data's own shape, both budget points, plus the 1 966 080 B point.
5. Frozen streams recorded with sha256; `river` version reported; **zero** changes to tracked files.
6. Reconciliation list in the **first 10 lines** of your report; NOT-RUNs declared as NOT-RUNs.
⛔ You do not decide the wave's venue — you compute the boolean; the Hub rules on it.
⛔ Never push `origin`; you have nothing to push.
